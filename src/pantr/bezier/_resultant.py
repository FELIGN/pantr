"""Resultant and discriminant computation for multivariate Bernstein polynomials.

Provides :func:`_resultant` and :func:`_discriminant`, which compute the
resultant and discriminant of multivariate Bernstein polynomials by evaluating
the Sylvester/Bezout matrix determinant at modified Chebyshev nodes and
interpolating back to Bernstein form via SVD.

This implements the dimension-elimination step of the algoim-style implicit
quadrature pipeline (R. I. Saye, *J. Comput. Phys.* 448, 110720, 2022).

Supporting functions:

- :func:`_bernstein_interpolate_1d` — SVD-based 1D interpolation from
  modified Chebyshev node values to Bernstein coefficients.
- :func:`_bernstein_interpolate` — Tensor-product extension to N-D.
- :func:`_normalise` — Scale polynomial by its largest coefficient.
- :func:`_auto_reduction` — Automatically reduce polynomial degree in
  each direction when trailing coefficients are negligible.
- :func:`_resultant_extent` — Compute the output extent of a resultant.
- :func:`_discriminant_extent` — Compute the output extent of a discriminant.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from ..quad import get_modified_chebyshev_nodes_1d
from ._resultant_matrices import _bezout_matrix, _det_qr, _sylvester_matrix


def _bernstein_vandermonde_svd(
    n: int,
    dtype: npt.DTypeLike = np.float64,
) -> tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
]:
    """Compute the SVD of the Bernstein Vandermonde matrix at modified Chebyshev nodes.

    The Vandermonde matrix ``V`` has entries ``V[i, j] = B_{j,n-1}(x_i)``
    where ``x_i`` are the modified Chebyshev nodes and ``B_{j,n-1}`` is the
    ``j``-th Bernstein basis function of degree ``n - 1``.

    Args:
        n (int): Number of nodes/coefficients (degree + 1). Must be >= 1.
        dtype (npt.DTypeLike): Floating dtype. Defaults to ``float64``.

    Returns:
        tuple[npt.NDArray, npt.NDArray, npt.NDArray]: ``(U, sigma, Vt)`` where
        ``V = U @ diag(sigma) @ Vt``.
    """
    from ..basis._basis_core import _tabulate_Bernstein_basis_1D_core  # noqa: PLC0415

    nodes = get_modified_chebyshev_nodes_1d(max(n, 2), dtype)[:n]
    V = np.empty((n, n), dtype=dtype)
    _tabulate_Bernstein_basis_1D_core(np.int32(n - 1), nodes, V)
    U, sigma, Vt = np.linalg.svd(V, full_matrices=True)
    return U, sigma, Vt


def _bernstein_interpolate_1d(
    f: npt.NDArray[np.floating[Any]],
    tol: float | None = None,
) -> npt.NDArray[np.floating[Any]]:
    """Interpolate values at modified Chebyshev nodes to 1D Bernstein coefficients.

    Given function values ``f[i]`` sampled at the modified Chebyshev nodes
    of order ``len(f)``, compute the Bernstein coefficients of the
    interpolating polynomial via truncated SVD.

    Args:
        f (npt.NDArray[np.floating[Any]]): Function values at modified
            Chebyshev nodes.  Shape ``(n,)`` where ``n >= 1``.
        tol (float | None): SVD truncation tolerance (relative to the
            largest singular value).  If *None*, uses
            ``100 * eps`` where ``eps`` is machine epsilon.

    Returns:
        npt.NDArray[np.floating[Any]]: Bernstein coefficients, shape ``(n,)``.

    Note:
        Implementation follows algoim (Saye, J. Comput. Phys. 448, 2022).
    """
    n = f.shape[0]
    dtype = f.dtype

    if n == 1:
        return f.copy()

    eps = float(np.finfo(dtype).eps)
    if tol is None:
        tol = 100.0 * eps

    U, sigma, Vt = _bernstein_vandermonde_svd(n, dtype)

    # Apply U^T to f
    tmp = U.T @ f

    # Truncated pseudo-inverse: zero out small singular values
    min_sigma = tol * sigma[0]
    inv_sigma = np.where(sigma >= min_sigma, 1.0 / sigma, 0.0)
    tmp *= inv_sigma

    # Apply V to get coefficients
    out = Vt.T @ tmp
    return out.astype(dtype)


def _bernstein_interpolate(
    f: npt.NDArray[np.floating[Any]],
    tol: float | None = None,
) -> npt.NDArray[np.floating[Any]]:
    """Interpolate tensor-product values at modified Chebyshev nodes to Bernstein coefficients.

    Applies :func:`_bernstein_interpolate_1d` sequentially along each
    dimension of the input array.

    Args:
        f (npt.NDArray[np.floating[Any]]): Function values at the tensor
            product of modified Chebyshev nodes.  Shape
            ``(n_0, n_1, ..., n_{N-1})``.
        tol (float | None): SVD truncation tolerance.  If *None*, uses a
            default based on machine epsilon.

    Returns:
        npt.NDArray[np.floating[Any]]: Bernstein coefficients with the same
        shape as ``f``.

    Note:
        Implementation follows algoim (Saye, J. Comput. Phys. 448, 2022).
    """
    result = f.copy()
    ndim = f.ndim

    for dim in range(ndim):
        n = result.shape[dim]
        if n == 1:
            continue

        U, sigma, Vt = _bernstein_vandermonde_svd(n, f.dtype)

        eps = float(np.finfo(f.dtype).eps)
        actual_tol = tol if tol is not None else 100.0 * eps
        min_sigma = actual_tol * sigma[0]
        inv_sigma = np.where(sigma >= min_sigma, 1.0 / sigma, 0.0)

        # Pseudoinverse matrix: V @ diag(1/sigma) @ U^T
        pinv = (Vt.T * inv_sigma[np.newaxis, :]) @ U.T

        # Apply along dimension `dim`
        result = np.tensordot(pinv, result, axes=([1], [dim]))
        # tensordot puts the result dimension first; move it back to `dim`
        result = np.moveaxis(result, 0, dim)

    return result


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


def _bernstein_derivative_1d(
    coeffs: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Compute the derivative of a 1D Bernstein polynomial.

    Given Bernstein coefficients ``a[0], ..., a[P-1]`` of degree ``P - 1``,
    returns the degree ``P - 2`` Bernstein coefficients of the derivative:
    ``out[i] = (P - 1) * (a[i+1] - a[i])``.

    Args:
        coeffs (npt.NDArray[np.floating[Any]]): 1D Bernstein coefficients,
            shape ``(P,)`` with ``P >= 2``.

    Returns:
        npt.NDArray[np.floating[Any]]: Derivative coefficients, shape
        ``(P - 1,)``.

    Note:
        Implementation follows algoim (Saye, J. Comput. Phys. 448, 2022).
        No input validation is performed.
    """
    P = coeffs.shape[0]
    result = (coeffs[1:] - coeffs[:-1]) * coeffs.dtype.type(P - 1)
    return np.asarray(result, dtype=coeffs.dtype)


def _bernstein_derivative_nd(
    coeffs: npt.NDArray[np.floating[Any]],
    dim: int,
) -> npt.NDArray[np.floating[Any]]:
    """Compute the partial derivative of a multivariate Bernstein polynomial.

    Differentiates along dimension ``dim``, reducing the extent in that
    dimension by 1.  The result is in Bernstein form.

    Args:
        coeffs (npt.NDArray[np.floating[Any]]): Bernstein coefficients with
            shape ``(n_0, ..., n_{N-1})``.  ``n_dim >= 2``.
        dim (int): Direction to differentiate.

    Returns:
        npt.NDArray[np.floating[Any]]: Derivative coefficients with shape
        ``(n_0, ..., n_dim - 1, ..., n_{N-1})``.

    Note:
        Implementation follows algoim (Saye, J. Comput. Phys. 448, 2022).
        No input validation is performed.
    """
    P = coeffs.shape[dim]
    # Slice along dim: coeffs[..., 1:, ...] - coeffs[..., :-1, ...]
    slc_hi = [slice(None)] * coeffs.ndim
    slc_lo = [slice(None)] * coeffs.ndim
    slc_hi[dim] = slice(1, None)
    slc_lo[dim] = slice(None, -1)
    result = (coeffs[tuple(slc_hi)] - coeffs[tuple(slc_lo)]) * coeffs.dtype.type(P - 1)
    return np.asarray(result, dtype=coeffs.dtype)


def _squared_l2_norm(
    coeffs: npt.NDArray[np.floating[Any]],
) -> float:
    r"""Compute the squared L2 norm of a Bernstein polynomial.

    Uses the Bernstein inner product formula:

    .. math::

        \int_0^1 B_{i,n}(x) B_{j,n}(x)\,dx
        = \frac{1}{2n+1} \frac{\binom{n}{i} \binom{n}{j}}{\binom{2n}{i+j}}

    extended to tensor products for multivariate polynomials.

    Args:
        coeffs (npt.NDArray[np.floating[Any]]): Bernstein coefficients
            (any shape).

    Returns:
        float: The squared L2 norm ``||p||_2^2``.

    Note:
        Implementation follows algoim (Saye, J. Comput. Phys. 448, 2022).
    """
    ndim = coeffs.ndim
    shape = coeffs.shape

    # Precompute binomial rows and Gram factors per dimension
    binom_rows = []
    binom_double = []
    for d in range(ndim):
        n = shape[d] - 1
        brow = np.array([math.comb(n, i) for i in range(n + 1)], dtype=np.float64)
        bdbl = np.array([math.comb(2 * n, k) for k in range(2 * n + 1)], dtype=np.float64)
        binom_rows.append(brow)
        binom_double.append(bdbl)

    # Flatten and compute all pairwise products with Gram weights
    flat = coeffs.ravel().astype(np.float64)
    indices = np.array(np.unravel_index(np.arange(flat.size), shape)).T  # (size, ndim)

    # For each pair (i, j): gram_weight = product over dims of binom(n_d, i_d)*binom(n_d, j_d) /
    #                                      binom(2*n_d, i_d + j_d)
    # This is O(size^2) but polynomials are small in practice.
    delta = 0.0
    for idx_i in range(flat.size):
        for idx_j in range(flat.size):
            g = 1.0
            for d in range(ndim):
                ii = indices[idx_i, d]
                jj = indices[idx_j, d]
                g *= (binom_rows[d][ii] * binom_rows[d][jj]) / binom_double[d][ii + jj]
            delta += flat[idx_i] * flat[idx_j] * g

    for d in range(ndim):
        delta /= 2.0 * shape[d] - 1.0

    return abs(delta)


def _degree_elevate_1d(
    coeffs: npt.NDArray[np.floating[Any]],
    target_len: int,
) -> npt.NDArray[np.floating[Any]]:
    """Elevate a 1D Bernstein polynomial to a higher degree.

    Args:
        coeffs (npt.NDArray[np.floating[Any]]): Bernstein coefficients,
            shape ``(P,)``.
        target_len (int): Target length (target degree + 1). Must be >= P.

    Returns:
        npt.NDArray[np.floating[Any]]: Elevated coefficients, shape
        ``(target_len,)``.

    Note:
        No input validation is performed.
    """
    P = coeffs.shape[0]
    if target_len == P:
        return coeffs.copy()

    result = coeffs.copy()
    # Repeatedly elevate by 1
    while result.shape[0] < target_len:
        n = result.shape[0] - 1
        new = np.empty(n + 2, dtype=result.dtype)
        new[0] = result[0]
        new[n + 1] = result[n]
        for k in range(1, n + 1):
            t = k / (n + 1.0)
            new[k] = t * result[k - 1] + (1.0 - t) * result[k]
        result = new
    return result


def _degree_elevate_nd(
    coeffs: npt.NDArray[np.floating[Any]],
    target_shape: tuple[int, ...],
) -> npt.NDArray[np.floating[Any]]:
    """Elevate a multivariate Bernstein polynomial to match a target shape.

    Args:
        coeffs (npt.NDArray[np.floating[Any]]): Bernstein coefficients.
        target_shape (tuple[int, ...]): Target shape per direction.

    Returns:
        npt.NDArray[np.floating[Any]]: Elevated coefficients.

    Note:
        No input validation is performed.
    """
    result = coeffs
    for d in range(coeffs.ndim):
        if result.shape[d] == target_shape[d]:
            continue
        # Move dim d to axis 0, reshape to 2D, elevate each row, reshape back
        moved = np.moveaxis(result, d, 0)
        flat_shape = (moved.shape[0], -1) if moved.ndim > 1 else (moved.shape[0], 1)
        flat = moved.reshape(flat_shape)
        elevated = np.stack(
            [_degree_elevate_1d(flat[:, i], target_shape[d]) for i in range(flat.shape[1])],
            axis=1,
        )
        new_shape = (target_shape[d], *moved.shape[1:])
        result = np.moveaxis(elevated.reshape(new_shape), 0, d)
    return result


def _degree_reduce_1d(
    coeffs: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Reduce a 1D Bernstein polynomial degree by 1 using least-squares.

    Args:
        coeffs (npt.NDArray[np.floating[Any]]): Bernstein coefficients,
            shape ``(P,)`` with ``P >= 2``.

    Returns:
        npt.NDArray[np.floating[Any]]: Reduced coefficients, shape ``(P - 1,)``.

    Note:
        No input validation is performed.
    """
    P = coeffs.shape[0]
    n = P - 1  # current degree
    # Degree elevation matrix: T maps (n-1) coefficients to n coefficients
    # T[k, j] gives how coefficient j of degree n-1 maps to coefficient k of degree n
    # b_elevated[k] = (k/n)*a[k-1] + (1 - k/n)*a[k]
    # Build T and solve least-squares: T @ x = coeffs
    T = np.zeros((P, P - 1), dtype=coeffs.dtype)
    for k in range(P):
        if k > 0 and k - 1 < P - 1:
            T[k, k - 1] = k / float(n)
        if k < P - 1:
            T[k, k] = 1.0 - k / float(n)
    result: npt.NDArray[np.floating[Any]] = np.linalg.lstsq(T, coeffs, rcond=None)[0]
    return result


def _auto_reduction(
    coeffs: npt.NDArray[np.floating[Any]],
    tol: float | None = None,
) -> tuple[npt.NDArray[np.floating[Any]], bool]:
    """Automatically reduce polynomial degree while maintaining accuracy.

    Tries to reduce the degree in each dimension by 1, checking whether the
    round-trip (reduce then elevate) changes the polynomial by more than
    ``tol`` in relative L2 norm.

    Args:
        coeffs (npt.NDArray[np.floating[Any]]): Bernstein coefficients
            (any shape, N-D).
        tol (float | None): Relative tolerance. If *None*, uses
            ``1e3 * eps``.

    Returns:
        tuple[npt.NDArray[np.floating[Any]], bool]: ``(reduced_coeffs, changed)``
        where ``changed`` is *True* if any reduction occurred.

    Note:
        Implementation follows algoim (Saye, J. Comput. Phys. 448, 2022).
    """
    eps = float(np.finfo(coeffs.dtype).eps)
    if tol is None:
        tol = 1.0e3 * eps

    if tol <= 0.0:
        return coeffs, False

    changed = False
    result = coeffs

    for dim in range(result.ndim):
        while result.shape[dim] >= 2:  # noqa: PLR2004
            # Try to reduce in this dimension
            moved = np.moveaxis(result, dim, 0)
            flat_shape = (moved.shape[0], -1) if moved.ndim > 1 else (moved.shape[0], 1)
            flat = moved.reshape(flat_shape)

            # Reduce each slice
            reduced_slices = np.stack(
                [_degree_reduce_1d(flat[:, i]) for i in range(flat.shape[1])],
                axis=1,
            )

            new_shape = (moved.shape[0] - 1, *moved.shape[1:])
            reduced = np.moveaxis(reduced_slices.reshape(new_shape), 0, dim)

            # Elevate back and check error
            elevated = _degree_elevate_nd(reduced, result.shape)

            diff_norm = _squared_l2_norm(elevated - result)
            orig_norm = _squared_l2_norm(result)

            if orig_norm > 0.0:
                rel_error = math.sqrt(abs(diff_norm)) / math.sqrt(abs(orig_norm))
            else:
                rel_error = math.sqrt(abs(diff_norm))

            if rel_error < tol:
                result = reduced
                changed = True
            else:
                break

    return result, changed


def _resultant_extent(
    extent_p: tuple[int, ...],
    extent_q: tuple[int, ...],
    dim: int,
) -> tuple[int, ...]:
    r"""Compute the output extent of the resultant of two polynomials.

    The resultant of ``p`` and ``q`` along dimension ``dim`` is a polynomial
    in all remaining dimensions.  Its extent in direction ``i``
    (skipping ``dim``) is:

    .. math::

        (P_{\mathrm{dim}} - 1)(Q_i - 1) + (Q_{\mathrm{dim}} - 1)(P_i - 1) + 1

    where ``P_i`` and ``Q_i`` are the extents (degree + 1) of ``p`` and ``q``.

    Args:
        extent_p (tuple[int, ...]): Extents of polynomial ``p``.
        extent_q (tuple[int, ...]): Extents of polynomial ``q``.
        dim (int): Elimination dimension.

    Returns:
        tuple[int, ...]: Extents of the resultant (one dimension fewer).
    """
    N = len(extent_p)
    result = []
    for i in range(N):
        if i == dim:
            continue
        ext = (extent_p[dim] - 1) * (extent_q[i] - 1) + (extent_q[dim] - 1) * (extent_p[i] - 1) + 1
        result.append(max(ext, 1))
    return tuple(result)


def _discriminant_extent(
    extent_p: tuple[int, ...],
    dim: int,
) -> tuple[int, ...]:
    """Compute the output extent of the discriminant of a polynomial.

    The discriminant along ``dim`` is the resultant of ``p`` with its
    partial derivative in direction ``dim``.

    Args:
        extent_p (tuple[int, ...]): Extents of polynomial ``p``.
        dim (int): Elimination dimension.

    Returns:
        tuple[int, ...]: Extents of the discriminant (one dimension fewer).
    """
    N = len(extent_p)
    result = []
    for i in range(N):
        if i == dim:
            continue
        ext = (2 * extent_p[dim] - 3) * (extent_p[i] - 1) + 1
        result.append(max(ext, 1))
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
            first polynomial.  Extent in ``dim`` must be >= 2.
        q (npt.NDArray[np.floating[Any]]): Bernstein coefficients of the
            second polynomial.  Must have the same number of dimensions as
            ``p``.
        dim (int): Dimension to eliminate (0-indexed).

    Returns:
        npt.NDArray[np.floating[Any]]: Resultant Bernstein coefficients with
        one fewer dimension.  Shape is determined by
        :func:`_resultant_extent`.

    Raises:
        ValueError: If ``dim`` is out of range, extents are too small, or
            arrays have different numbers of dimensions.

    Note:
        Implementation follows algoim (Saye, J. Comput. Phys. 448, 2022).
    """
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
        p (npt.NDArray[np.floating[Any]]): Bernstein coefficients.  Extent
            in ``dim`` must be >= 2.
        dim (int): Dimension to eliminate.

    Returns:
        npt.NDArray[np.floating[Any]]: Discriminant Bernstein coefficients
        with one fewer dimension.

    Raises:
        ValueError: If ``dim`` is out of range or extent in ``dim`` < 2.

    Note:
        Implementation follows algoim (Saye, J. Comput. Phys. 448, 2022).
    """
    _validate_resultant_inputs_single(p, dim)

    if p.shape[dim] < 3:  # noqa: PLR2004
        raise ValueError(
            f"Discriminant requires extent >= 3 in dimension {dim} "
            f"(degree >= 2), got {p.shape[dim]}."
        )

    # Compute partial derivative along dim
    dp = _bernstein_derivative_nd(p, dim)

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
        raise ValueError(f"p must have extent >= 2 in dimension {dim}, got {p.shape[dim]}.")
    if q.shape[dim] < 1:
        raise ValueError(f"q must have extent >= 1 in dimension {dim}, got {q.shape[dim]}.")


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
        raise ValueError(f"p must have extent >= 2 in dimension {dim}, got {p.shape[dim]}.")


def _resultant_core(
    p: npt.NDArray[np.floating[Any]],
    q: npt.NDArray[np.floating[Any]],
    dim: int,
) -> npt.NDArray[np.floating[Any]]:
    """Core resultant computation via interpolation at Chebyshev nodes.

    Args:
        p (npt.NDArray[np.floating[Any]]): First polynomial.
        q (npt.NDArray[np.floating[Any]]): Second polynomial.
        dim (int): Dimension to eliminate.

    Returns:
        npt.NDArray[np.floating[Any]]: Resultant coefficients.

    Note:
        Implementation follows algoim (Saye, J. Comput. Phys. 448, 2022).
    """
    _validate_resultant_inputs(p, q, dim)

    dtype = np.result_type(p.dtype, q.dtype)
    p = p.astype(dtype, copy=False)
    q = q.astype(dtype, copy=False)
    eps = float(np.finfo(dtype).eps)
    ndim = p.ndim

    P = p.shape[dim]
    Q = q.shape[dim]

    # Compute output extent
    out_extent = _resultant_extent(p.shape, q.shape, dim)

    if ndim == 1:
        # 1D case: direct matrix determinant, no interpolation needed
        mat = _bezout_matrix(p, q) if P == Q else _sylvester_matrix(p, q)
        det, _ = _det_qr(mat)
        return np.array(det, dtype=dtype)

    # N-D case: interpolate determinant values at modified Chebyshev nodes
    # Allocate output for determinant values
    f = np.empty(out_extent, dtype=dtype)

    # Generate modified Chebyshev nodes for each output dimension
    nodes_per_dim = []
    for d in range(len(out_extent)):
        n = out_extent[d]
        if n >= 2:  # noqa: PLR2004
            nodes_per_dim.append(get_modified_chebyshev_nodes_1d(n, dtype))
        else:
            nodes_per_dim.append(np.array([0.5], dtype=dtype))

    # Iterate over all multi-indices in the output grid
    for flat_idx in range(f.size):
        multi_idx = np.unravel_index(flat_idx, out_extent)

        # Build the evaluation point x0 (all dims except `dim`)
        x0 = np.empty(ndim - 1, dtype=dtype)
        for i, idx_val in enumerate(multi_idx):
            x0[i] = nodes_per_dim[i][idx_val]

        # Collapse p and q to 1D along `dim` at point x0
        pk = _collapse_along_axis_raw(p, x0, dim)
        qk = _collapse_along_axis_raw(q, x0, dim)

        # Build resultant matrix and compute determinant
        mat = _bezout_matrix(pk, qk) if P == Q else _sylvester_matrix(pk, qk)
        det, _ = _det_qr(mat)
        f[multi_idx] = det

    # Normalise and interpolate to Bernstein coefficients
    _normalise(f)
    interp_tol = (100.0 * eps) ** (1.0 / max(ndim - 1, 1))
    out = _bernstein_interpolate(f, interp_tol)

    # Automatic degree reduction
    auto_tol = 1.0e4 * eps
    reduced, did_reduce = _auto_reduction(out, auto_tol)

    if did_reduce:
        # Recursive call on reduced space
        return _resultant_core_reduced(p, q, dim, reduced.shape)

    return out


def _resultant_core_reduced(
    p: npt.NDArray[np.floating[Any]],
    q: npt.NDArray[np.floating[Any]],
    dim: int,
    target_shape: tuple[int, ...],
) -> npt.NDArray[np.floating[Any]]:
    """Recompute resultant on a reduced output grid.

    After auto-reduction determines a smaller output shape, this function
    recomputes the resultant directly on the reduced grid for better
    conditioning.

    Args:
        p (npt.NDArray[np.floating[Any]]): First polynomial.
        q (npt.NDArray[np.floating[Any]]): Second polynomial.
        dim (int): Dimension to eliminate.
        target_shape (tuple[int, ...]): Reduced output shape.

    Returns:
        npt.NDArray[np.floating[Any]]: Resultant coefficients.
    """
    dtype = np.result_type(p.dtype, q.dtype)
    eps = float(np.finfo(dtype).eps)
    ndim = p.ndim
    P = p.shape[dim]
    Q = q.shape[dim]

    f = np.empty(target_shape, dtype=dtype)

    nodes_per_dim = []
    for d in range(len(target_shape)):
        n = target_shape[d]
        if n >= 2:  # noqa: PLR2004
            nodes_per_dim.append(get_modified_chebyshev_nodes_1d(n, dtype))
        else:
            nodes_per_dim.append(np.array([0.5], dtype=dtype))

    for flat_idx in range(f.size):
        multi_idx = np.unravel_index(flat_idx, target_shape)

        x0 = np.empty(ndim - 1, dtype=dtype)
        for i, idx_val in enumerate(multi_idx):
            x0[i] = nodes_per_dim[i][idx_val]

        pk = _collapse_along_axis_raw(p, x0, dim)
        qk = _collapse_along_axis_raw(q, x0, dim)

        mat = _bezout_matrix(pk, qk) if P == Q else _sylvester_matrix(pk, qk)
        det, _ = _det_qr(mat)
        f[multi_idx] = det

    _normalise(f)
    interp_tol = (100.0 * eps) ** (1.0 / max(ndim - 1, 1))
    return _bernstein_interpolate(f, interp_tol)
