"""Resultant and discriminant computation for multivariate Bernstein polynomials.

Provides :func:`_resultant` and :func:`_discriminant`, which compute the
resultant and discriminant of multivariate Bernstein polynomials by evaluating
the Sylvester/Bezout matrix determinant at modified Chebyshev nodes and
interpolating back to Bernstein form via SVD.

This implements the dimension-elimination step of the algoim-style implicit
quadrature pipeline (R. I. Saye, *J. Comput. Phys.* 448, 110720, 2022).

Supporting functions:

- :func:`_normalise` — Scale polynomial by its largest coefficient.
- :func:`_resultant_order` — Compute the output order of a resultant.
- :func:`_discriminant_order` — Compute the output order of a discriminant.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .._interpolation_utils import SVD_TOL_FACTOR
from ..quad import get_modified_chebyshev_nodes_1d
from ._bezier_derivative import _derivative_ctrl_nd
from ._bezier_interpolate import _bernstein_interpolate
from ._resultant_matrices import _bezout_matrix, _det_qr, _sylvester_matrix

_RESULTANT_REDUCTION_TOL_FACTOR: float = 1.0e4
"""Factor multiplied by machine epsilon for resultant auto-reduction."""


def _normalise(
    coeffs: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Scale polynomial coefficients by their maximum absolute value.

    Args:
        coeffs (npt.NDArray[np.floating[Any]]): Bernstein coefficients
            (any shape). Modified in place.

    Returns:
        npt.NDArray[np.floating[Any]]: The normalised array (same object as
        ``coeffs``).

    Note:
        Implementation follows algoim (Saye, J. Comput. Phys. 448, 2022).
    """
    x = float(np.max(np.abs(coeffs)))
    if x > 0.0:
        coeffs *= 1.0 / x
    return coeffs


def _resultant_order(
    order_p: tuple[int, ...],
    order_q: tuple[int, ...],
    dim: int,
) -> tuple[int, ...]:
    r"""Compute the output order of the resultant of two polynomials.

    The order of a polynomial is its degree plus one, i.e. the number of
    Bernstein coefficients per direction.

    The resultant of ``p`` and ``q`` along dimension ``dim`` is a polynomial
    in all remaining dimensions.  Its order in direction ``i``
    (skipping ``dim``) is:

    .. math::

        (P_{\mathrm{dim}} - 1)(Q_i - 1) + (Q_{\mathrm{dim}} - 1)(P_i - 1) + 1

    where ``P_i`` and ``Q_i`` are the orders (degree + 1) of ``p`` and ``q``.

    Args:
        order_p (tuple[int, ...]): Orders of polynomial ``p`` per direction.
        order_q (tuple[int, ...]): Orders of polynomial ``q`` per direction.
        dim (int): Elimination dimension.

    Returns:
        tuple[int, ...]: Orders of the resultant (one dimension fewer).
    """
    N = len(order_p)
    result = []
    for i in range(N):
        if i == dim:
            continue
        ord_i = (order_p[dim] - 1) * (order_q[i] - 1) + (order_q[dim] - 1) * (order_p[i] - 1) + 1
        result.append(max(ord_i, 1))
    return tuple(result)


def _discriminant_order(
    order_p: tuple[int, ...],
    dim: int,
) -> tuple[int, ...]:
    """Compute the output order of the discriminant of a polynomial.

    The order of a polynomial is its degree plus one, i.e. the number of
    Bernstein coefficients per direction.

    The discriminant along ``dim`` is the resultant of ``p`` with its
    partial derivative in direction ``dim``.

    Args:
        order_p (tuple[int, ...]): Orders (degree + 1) of polynomial ``p``.
        dim (int): Elimination dimension.

    Returns:
        tuple[int, ...]: Orders of the discriminant (one dimension fewer).
    """
    N = len(order_p)
    result = []
    for i in range(N):
        if i == dim:
            continue
        ord_i = (2 * order_p[dim] - 3) * (order_p[i] - 1) + 1
        result.append(max(ord_i, 1))
    return tuple(result)


def _collapse_along_axis_raw(
    coeffs: npt.NDArray[np.floating[Any]],
    x0: npt.NDArray[np.floating[Any]],
    dim: int,
) -> npt.NDArray[np.floating[Any]]:
    """Collapse a multivariate Bernstein polynomial along all axes except one.

    Evaluates Bernstein basis functions at ``x0`` for each direction other
    than ``dim``, contracting the coefficient tensor to produce a 1D array
    of Bernstein coefficients along ``dim``.

    Args:
        coeffs (npt.NDArray[np.floating[Any]]): N-D Bernstein coefficients.
        x0 (npt.NDArray[np.floating[Any]]): Parameter values for dimensions
            other than ``dim``.  Length ``N - 1``.
        dim (int): Direction to keep.

    Returns:
        npt.NDArray[np.floating[Any]]: 1D Bernstein coefficients along ``dim``,
        shape ``(coeffs.shape[dim],)``.

    Note:
        No input validation is performed.
    """
    from ..basis._basis_core import _tabulate_Bernstein_basis_1D_core  # noqa: PLC0415

    ndim = coeffs.ndim
    dtype = coeffs.dtype
    result = coeffs

    # Contract from highest dimension to lowest, skipping `dim`
    for d in range(ndim - 1, -1, -1):
        if d == dim:
            continue

        val_idx = d if d < dim else d - 1
        pts = np.array([x0[val_idx]], dtype=dtype)
        basis = np.empty((1, result.shape[d]), dtype=dtype)
        _tabulate_Bernstein_basis_1D_core(np.int32(result.shape[d] - 1), pts, basis)
        basis_1d = basis[0]

        result = np.tensordot(basis_1d, result, axes=([0], [d]))

    return result.ravel()


def _resultant(
    p: npt.NDArray[np.floating[Any]],
    q: npt.NDArray[np.floating[Any]],
    dim: int,
) -> npt.NDArray[np.floating[Any]]:
    """Compute the resultant of two multivariate Bernstein polynomials.

    Eliminates dimension ``dim`` by evaluating the Sylvester/Bezout matrix
    determinant at modified Chebyshev nodes in the remaining dimensions,
    then interpolating back to Bernstein form via SVD.

    For 1D inputs (``p.ndim == 1``), returns the scalar resultant as a
    0-D array.

    Args:
        p (npt.NDArray[np.floating[Any]]): Bernstein coefficients of the
            first polynomial.  Order (degree + 1) in ``dim`` must be >= 2.
        q (npt.NDArray[np.floating[Any]]): Bernstein coefficients of the
            second polynomial.  Must have the same number of dimensions as
            ``p``.
        dim (int): Dimension to eliminate (0-indexed).

    Returns:
        npt.NDArray[np.floating[Any]]: Resultant Bernstein coefficients with
        one fewer dimension.  Shape is determined by
        :func:`_resultant_order`.

    Raises:
        ValueError: If ``dim`` is out of range, orders are too small, or
            arrays have different numbers of dimensions.

    Note:
        Implementation follows algoim (Saye, J. Comput. Phys. 448, 2022).
    """
    _validate_resultant_inputs(p, q, dim)
    return _resultant_core(p, q, dim)


def _discriminant(
    p: npt.NDArray[np.floating[Any]],
    dim: int,
) -> npt.NDArray[np.floating[Any]]:
    """Compute the discriminant of a multivariate Bernstein polynomial.

    The discriminant along ``dim`` is the resultant of ``p`` with its
    partial derivative ``dp/dx_dim``.  It identifies the locus where ``p``
    and ``dp/dx_dim`` share a common root along ``dim``.

    For 1D inputs (``p.ndim == 1``), returns the scalar discriminant as a
    0-D array.

    Args:
        p (npt.NDArray[np.floating[Any]]): Bernstein coefficients.  Order
            (degree + 1) in ``dim`` must be >= 3.
        dim (int): Dimension to eliminate.

    Returns:
        npt.NDArray[np.floating[Any]]: Discriminant Bernstein coefficients
        with one fewer dimension.

    Raises:
        ValueError: If ``dim`` is out of range or order in ``dim`` < 3.

    Note:
        Implementation follows algoim (Saye, J. Comput. Phys. 448, 2022).
    """
    _validate_resultant_inputs_single(p, dim)

    if p.shape[dim] < 3:  # noqa: PLR2004
        raise ValueError(
            f"Discriminant requires order >= 3 in dimension {dim} "
            f"(degree >= 2), got {p.shape[dim]}."
        )

    # Compute partial derivative along dim
    dp = _derivative_ctrl_nd(p, dim)

    return _resultant_core(p, dp, dim)


def _validate_resultant_inputs(
    p: npt.NDArray[np.floating[Any]],
    q: npt.NDArray[np.floating[Any]],
    dim: int,
) -> None:
    """Validate inputs for resultant computation.

    Args:
        p (npt.NDArray[np.floating[Any]]): First polynomial.
        q (npt.NDArray[np.floating[Any]]): Second polynomial.
        dim (int): Elimination dimension.

    Raises:
        ValueError: If inputs are invalid.
    """
    if p.ndim != q.ndim:
        raise ValueError(
            f"p and q must have the same number of dimensions, got {p.ndim} and {q.ndim}."
        )
    if not np.issubdtype(p.dtype, np.floating):
        raise ValueError(f"p must have floating dtype, got {p.dtype}.")
    if not np.issubdtype(q.dtype, np.floating):
        raise ValueError(f"q must have floating dtype, got {q.dtype}.")
    if dim < 0 or dim >= p.ndim:
        raise ValueError(f"dim must be in [0, {p.ndim}), got {dim}.")
    if p.shape[dim] < 2:  # noqa: PLR2004
        raise ValueError(f"p must have order >= 2 in dimension {dim}, got {p.shape[dim]}.")
    if q.shape[dim] < 2:  # noqa: PLR2004
        raise ValueError(f"q must have order >= 2 in dimension {dim}, got {q.shape[dim]}.")


def _validate_resultant_inputs_single(
    p: npt.NDArray[np.floating[Any]],
    dim: int,
) -> None:
    """Validate inputs for discriminant computation.

    Args:
        p (npt.NDArray[np.floating[Any]]): Polynomial.
        dim (int): Elimination dimension.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not np.issubdtype(p.dtype, np.floating):
        raise ValueError(f"p must have floating dtype, got {p.dtype}.")
    if dim < 0 or dim >= p.ndim:
        raise ValueError(f"dim must be in [0, {p.ndim}), got {dim}.")
    if p.shape[dim] < 2:  # noqa: PLR2004
        raise ValueError(f"p must have order >= 2 in dimension {dim}, got {p.shape[dim]}.")


def _evaluate_det_on_grid(
    p: npt.NDArray[np.floating[Any]],
    q: npt.NDArray[np.floating[Any]],
    dim: int,
    grid_shape: tuple[int, ...],
) -> npt.NDArray[np.floating[Any]]:
    """Evaluate resultant-matrix determinants on a tensor-product Chebyshev grid.

    Collapses ``p`` and ``q`` to 1D along ``dim`` at each grid point, builds
    the Sylvester or Bezout matrix, and stores the determinant.  The result
    is normalised and interpolated back to Bernstein coefficients.

    Args:
        p (npt.NDArray[np.floating[Any]]): First polynomial (N-D, N >= 2).
        q (npt.NDArray[np.floating[Any]]): Second polynomial (same ndim).
        dim (int): Elimination dimension.
        grid_shape (tuple[int, ...]): Shape of the output Chebyshev grid
            (one entry per remaining dimension).

    Returns:
        npt.NDArray[np.floating[Any]]: Bernstein coefficients of the
        interpolated resultant on the given grid.

    Note:
        No input validation is performed.
    """
    dtype = np.result_type(p.dtype, q.dtype)
    eps = float(np.finfo(dtype).eps)
    ndim = p.ndim
    P = p.shape[dim]
    Q = q.shape[dim]

    f = np.empty(grid_shape, dtype=dtype)

    nodes_per_dim = []
    for d in range(len(grid_shape)):
        n = grid_shape[d]
        if n >= 2:  # noqa: PLR2004
            nodes_per_dim.append(get_modified_chebyshev_nodes_1d(n, dtype))
        else:
            nodes_per_dim.append(np.array([0.5], dtype=dtype))

    for flat_idx in range(f.size):
        multi_idx = np.unravel_index(flat_idx, grid_shape)

        x0 = np.empty(ndim - 1, dtype=dtype)
        for i, idx_val in enumerate(multi_idx):
            x0[i] = nodes_per_dim[i][idx_val]

        pk = _collapse_along_axis_raw(p, x0, dim)
        qk = _collapse_along_axis_raw(q, x0, dim)

        mat = _bezout_matrix(pk, qk) if P == Q else _sylvester_matrix(pk, qk)
        det, _ = _det_qr(mat)
        f[multi_idx] = det

    _normalise(f)
    interp_tol = (SVD_TOL_FACTOR * eps) ** (1.0 / max(ndim - 1, 1))
    return _bernstein_interpolate(f, interp_tol)


def _resultant_core(
    p: npt.NDArray[np.floating[Any]],
    q: npt.NDArray[np.floating[Any]],
    dim: int,
) -> npt.NDArray[np.floating[Any]]:
    """Core resultant computation via interpolation at Chebyshev nodes.

    Assumes inputs have already been validated by the caller.

    Args:
        p (npt.NDArray[np.floating[Any]]): First polynomial.
        q (npt.NDArray[np.floating[Any]]): Second polynomial.
        dim (int): Dimension to eliminate.

    Returns:
        npt.NDArray[np.floating[Any]]: Resultant coefficients.

    Note:
        Implementation follows algoim (Saye, J. Comput. Phys. 448, 2022).
        No input validation is performed.
    """
    dtype = np.result_type(p.dtype, q.dtype)
    p = p.astype(dtype, copy=False)
    q = q.astype(dtype, copy=False)
    ndim = p.ndim

    P = p.shape[dim]
    Q = q.shape[dim]

    if ndim == 1:
        # 1D case: direct matrix determinant, no interpolation needed
        mat = _bezout_matrix(p, q) if P == Q else _sylvester_matrix(p, q)
        det, _ = _det_qr(mat)
        return np.array(det, dtype=dtype)

    # N-D case: interpolate determinant values at modified Chebyshev nodes
    out_order = _resultant_order(p.shape, q.shape, dim)
    out = _evaluate_det_on_grid(p, q, dim, out_order)

    # Automatic degree reduction via Bezier.minimize_degree
    from . import Bezier as BezierCls  # noqa: PLC0415

    eps = float(np.finfo(dtype).eps)
    auto_tol = _RESULTANT_REDUCTION_TOL_FACTOR * eps
    out_bezier = BezierCls(out[..., np.newaxis], is_rational=False)
    reduced_bezier = out_bezier.minimize_degree(tol=auto_tol)

    if reduced_bezier.degree != out_bezier.degree:
        # Recompute on reduced grid for better conditioning
        reduced_shape = tuple(d + 1 for d in reduced_bezier.degree)
        return _evaluate_det_on_grid(p, q, dim, reduced_shape)

    return out
