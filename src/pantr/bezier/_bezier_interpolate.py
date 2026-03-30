"""Bernstein interpolation and fitting: construct a Bézier from function samples or values.

Provides :func:`interpolate_bezier` (callable-based) and :func:`fit_bezier`
(pre-evaluated values), which recover the Bernstein coefficients from the
Bernstein Vandermonde system.

The Bernstein Vandermonde matrix becomes increasingly ill-conditioned as the
polynomial degree grows, making a direct solve numerically unstable.  To handle
this, all solvers in this module use a **truncated SVD** pseudo-inverse:
singular values smaller than ``tol * sigma_max`` are zeroed out, which
regularises the inversion and yields robust results even at high degree.

Both functions support **tensor-product** grids (represented as a
:class:`~pantr.quad.PointsLattice` or a sequence of 1D node arrays) and
**scattered** point sets (represented as a 2D array of shape
``(n_pts, dim)``).

Supporting utilities (used internally and by the resultant pipeline):

- :func:`_bernstein_vandermonde_svd` — SVD of the Bernstein Vandermonde
  matrix at modified Chebyshev nodes.
- :func:`_bernstein_interpolate_1d` — 1D SVD-based interpolation from
  node values to Bernstein coefficients.
- :func:`_bernstein_interpolate` — Tensor-product extension to N-D.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from .._interpolation_utils import SVD_TOL_FACTOR, split_components
from ..quad import PointsLattice, get_modified_chebyshev_nodes_1d

if TYPE_CHECKING:
    from . import Bezier


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
        tol = SVD_TOL_FACTOR * eps

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
        actual_tol = tol if tol is not None else SVD_TOL_FACTOR * eps
        min_sigma = actual_tol * sigma[0]
        inv_sigma = np.where(sigma >= min_sigma, 1.0 / sigma, 0.0)

        # Pseudoinverse matrix: V @ diag(1/sigma) @ U^T
        pinv = (Vt.T * inv_sigma[np.newaxis, :]) @ U.T

        # Apply along dimension `dim`
        result = np.tensordot(pinv, result, axes=([1], [dim]))
        # tensordot puts the result dimension first; move it back to `dim`
        result = np.moveaxis(result, 0, dim)

    return result


def _resolve_nodes(
    n_pts: tuple[int, ...],
    nodes: (
        Literal["chebyshev", "uniform"]
        | PointsLattice
        | npt.NDArray[np.floating[Any]]
        | Sequence[npt.NDArray[np.floating[Any]]]
        | None
    ),
    dtype: np.dtype[np.float32] | np.dtype[np.float64],
) -> list[npt.NDArray[np.floating[Any]]]:
    """Resolve the *nodes* parameter into a list of 1D node arrays.

    Args:
        n_pts (tuple[int, ...]): Number of sample points per direction.
        nodes: Node specification — ``None`` or ``"chebyshev"`` for modified
            Chebyshev-Lobatto nodes, ``"uniform"`` for equispaced nodes, a
            :class:`~pantr.quad.PointsLattice`, a single 1D array (broadcast
            to every direction), or a sequence of 1D arrays (one per
            direction).
        dtype (np.dtype): Floating dtype for generated nodes.

    Returns:
        list[npt.NDArray[np.floating[Any]]]: One 1D node array per parametric
        direction.

    Raises:
        ValueError: If *nodes* is inconsistent with *n_pts*.
    """
    if nodes is None or (isinstance(nodes, str) and nodes == "chebyshev"):
        return [
            get_modified_chebyshev_nodes_1d(n, dtype)
            if n >= 2  # noqa: PLR2004
            else np.array([0.5], dtype=dtype)
            for n in n_pts
        ]

    if isinstance(nodes, str) and nodes == "uniform":
        result: list[npt.NDArray[np.floating[Any]]] = [
            np.linspace(0.0, 1.0, n, dtype=dtype) for n in n_pts
        ]
        return result

    # PointsLattice
    if isinstance(nodes, PointsLattice):
        if nodes.dim != len(n_pts):
            raise ValueError(f"PointsLattice has {nodes.dim} dimensions, expected {len(n_pts)}.")
        node_list: list[npt.NDArray[np.floating[Any]]] = [
            np.asarray(a, dtype=dtype) for a in nodes.pts_per_dir
        ]
        for i, (arr_i, n) in enumerate(zip(node_list, n_pts, strict=True)):
            if arr_i.shape[0] != n:
                raise ValueError(
                    f"PointsLattice direction {i} has {arr_i.shape[0]} nodes, expected n_pts={n}."
                )
        return node_list

    # User-provided nodes
    if isinstance(nodes, np.ndarray) and nodes.ndim == 1:
        # Single array — broadcast to all directions
        arr: npt.NDArray[np.floating[Any]] = nodes.astype(dtype, copy=False)
        for n in n_pts:
            if arr.shape[0] != n:
                raise ValueError(f"Node array length {arr.shape[0]} does not match n_pts={n}.")
        return [arr] * len(n_pts)

    # Sequence of arrays
    node_list_seq: list[npt.NDArray[np.floating[Any]]] = [np.asarray(a, dtype=dtype) for a in nodes]
    if len(node_list_seq) != len(n_pts):
        raise ValueError(f"Expected {len(n_pts)} node arrays, got {len(node_list_seq)}.")
    for i, (arr, n) in enumerate(zip(node_list_seq, n_pts, strict=True)):
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(
                f"Node array for direction {i} has shape {arr.shape}, expected ({n},)."
            )
    return node_list_seq


def _build_bernstein_pinv(
    nodes: npt.NDArray[np.floating[Any]],
    tol: float | None = None,
    degree: int | None = None,
) -> npt.NDArray[np.floating[Any]]:
    """Build the Bernstein pseudo-inverse matrix for arbitrary 1D nodes.

    Given a set of 1D interpolation nodes, constructs the matrix that maps
    function values at those nodes to Bernstein coefficients via truncated SVD.

    When ``degree`` is *None* or equals ``len(nodes) - 1``, the Vandermonde
    matrix is square and the result is an exact interpolation matrix.  When
    ``degree < len(nodes) - 1``, the Vandermonde is rectangular and the
    pseudo-inverse yields a least-squares fit.

    Args:
        nodes (npt.NDArray[np.floating[Any]]): 1D interpolation nodes on
            [0, 1], shape ``(n_pts,)``.
        tol (float | None): SVD truncation tolerance. If *None*, uses
            ``100 * eps``.
        degree (int | None): Polynomial degree of the output Bernstein
            representation.  If *None*, defaults to ``n_pts - 1`` (exact
            interpolation).

    Returns:
        npt.NDArray[np.floating[Any]]: Pseudo-inverse matrix, shape
        ``(degree + 1, n_pts)``.
    """
    from ..basis._basis_core import _tabulate_Bernstein_basis_1D_core  # noqa: PLC0415

    n_pts = nodes.shape[0]
    dtype = nodes.dtype
    deg = n_pts - 1 if degree is None else degree

    if n_pts == 1 and deg == 0:
        return np.ones((1, 1), dtype=dtype)

    n_coeffs = deg + 1
    V = np.empty((n_pts, n_coeffs), dtype=dtype)
    _tabulate_Bernstein_basis_1D_core(np.int32(deg), nodes, V)
    U, sigma, Vt = np.linalg.svd(V, full_matrices=False)

    eps = float(np.finfo(dtype).eps)
    actual_tol = tol if tol is not None else SVD_TOL_FACTOR * eps
    min_sigma = actual_tol * sigma[0]
    inv_sigma = np.where(sigma >= min_sigma, 1.0 / sigma, 0.0)

    pinv: npt.NDArray[np.floating[Any]] = (Vt.T * inv_sigma[np.newaxis, :]) @ U.T
    return pinv


def _build_nd_bernstein_vandermonde(
    pts: npt.NDArray[np.floating[Any]],
    degree_tuple: tuple[int, ...],
) -> npt.NDArray[np.floating[Any]]:
    """Build the full tensor-product Bernstein Vandermonde matrix at scattered points.

    For ``N``-dimensional points with degrees ``(p_0, ..., p_{N-1})``, the
    Vandermonde matrix has shape ``(n_pts, prod(p_d + 1))`` where each row
    contains the products of univariate Bernstein basis values across all
    directions.

    Args:
        pts (npt.NDArray[np.floating[Any]]): Evaluation points, shape
            ``(n_pts,)`` for 1D or ``(n_pts, ndim)`` for N-D.
        degree_tuple (tuple[int, ...]): Polynomial degree per parametric
            direction.

    Returns:
        npt.NDArray[np.floating[Any]]: Vandermonde matrix, shape
        ``(n_pts, prod(degree[d] + 1))``.
    """
    from ..basis._basis_core import _tabulate_Bernstein_basis_1D_core  # noqa: PLC0415

    ndim = len(degree_tuple)
    dtype = pts.dtype

    pts_2d = pts[:, np.newaxis] if pts.ndim == 1 else pts
    n_pts = pts_2d.shape[0]

    # Evaluate univariate Bernstein bases per direction
    basis_per_dir: list[npt.NDArray[np.floating[Any]]] = []
    for d in range(ndim):
        n_coeffs = degree_tuple[d] + 1
        B_d = np.empty((n_pts, n_coeffs), dtype=dtype)
        _tabulate_Bernstein_basis_1D_core(np.int32(degree_tuple[d]), pts_2d[:, d], B_d)
        basis_per_dir.append(B_d)

    # Build tensor-product Vandermonde via successive Kronecker-like expansion
    V = basis_per_dir[0]
    for d in range(1, ndim):
        # V has shape (n_pts, cols_so_far), basis_per_dir[d] has shape (n_pts, n_d)
        # Result: (n_pts, cols_so_far * n_d) via row-wise outer product
        V = (V[:, :, np.newaxis] * basis_per_dir[d][:, np.newaxis, :]).reshape(n_pts, -1)

    return V


def _fit_from_scattered(
    values: npt.NDArray[np.floating[Any]],
    pts: npt.NDArray[np.floating[Any]],
    degree_tuple: tuple[int, ...],
    tol: float | None,
) -> npt.NDArray[np.floating[Any]]:
    """Fit Bernstein coefficients from scattered (non-tensor-product) point values.

    Builds the full tensor-product Bernstein Vandermonde at the given points
    and recovers the coefficients via truncated SVD pseudo-inverse.

    Args:
        values (npt.NDArray[np.floating[Any]]): Function values at scattered
            points.  Shape ``(n_pts,)`` for scalar or ``(n_pts, rank)`` for
            vector-valued.
        pts (npt.NDArray[np.floating[Any]]): Evaluation points, shape
            ``(n_pts,)`` for 1D or ``(n_pts, ndim)`` for N-D.
        degree_tuple (tuple[int, ...]): Polynomial degree per parametric
            direction.
        tol (float | None): SVD truncation tolerance.

    Returns:
        npt.NDArray[np.floating[Any]]: Control points with shape
        ``(*orders, rank)`` where ``orders[d] = degree[d] + 1``.
    """
    dtype = values.dtype
    orders = tuple(d + 1 for d in degree_tuple)

    V = _build_nd_bernstein_vandermonde(pts, degree_tuple)

    # SVD pseudo-inverse with truncation
    U, sigma, Vt = np.linalg.svd(V, full_matrices=False)
    eps = float(np.finfo(dtype).eps)
    actual_tol = tol if tol is not None else SVD_TOL_FACTOR * eps
    min_sigma = actual_tol * sigma[0]
    inv_sigma = np.where(sigma >= min_sigma, 1.0 / sigma, 0.0)
    pinv: npt.NDArray[np.floating[Any]] = (Vt.T * inv_sigma[np.newaxis, :]) @ U.T

    # Solve for each component
    is_vector = values.ndim == 2  # noqa: PLR2004
    if is_vector:
        rank = values.shape[1]
        components: list[npt.NDArray[np.floating[Any]]] = []
        for r in range(rank):
            coeffs_flat = pinv @ values[:, r]
            components.append(coeffs_flat.reshape(orders).astype(dtype))
        return np.stack(components, axis=-1)

    coeffs_flat = pinv @ values
    ctrl = coeffs_flat.reshape(orders).astype(dtype)
    return ctrl[..., np.newaxis]


def _validate_n_pts(
    n_pts: int | Sequence[int],
) -> tuple[int, ...]:
    """Normalise and validate *n_pts* to a tuple of positive integers.

    Args:
        n_pts (int | Sequence[int]): Number of sample points per direction.

    Returns:
        tuple[int, ...]: Validated tuple.

    Raises:
        ValueError: If any value is < 1.
    """
    n_pts_tuple = (n_pts,) if isinstance(n_pts, int) else tuple(n_pts)
    for i, n in enumerate(n_pts_tuple):
        if n < 1:
            raise ValueError(f"n_pts[{i}] must be >= 1, got {n}.")
    return n_pts_tuple


def _validate_degree(
    degree: int | Sequence[int] | None,
    n_pts_tuple: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Normalise and validate the *degree* parameter.

    Args:
        degree (int | Sequence[int] | None): Requested degree per direction.
        n_pts_tuple (tuple[int, ...]): Number of sample points per direction.

    Returns:
        tuple[int, ...] | None: Validated degree tuple, or *None* if degree
        was not specified (meaning degree = n_pts - 1).

    Raises:
        ValueError: If degree >= n_pts in any direction.
    """
    if degree is None:
        return None
    deg_tuple = (degree,) * len(n_pts_tuple) if isinstance(degree, int) else tuple(degree)
    if len(deg_tuple) != len(n_pts_tuple):
        raise ValueError(f"degree has {len(deg_tuple)} entries but n_pts has {len(n_pts_tuple)}.")
    for i, (d, n) in enumerate(zip(deg_tuple, n_pts_tuple, strict=True)):
        if d < 0:
            raise ValueError(f"degree[{i}] must be >= 0, got {d}.")
        if d >= n:
            raise ValueError(
                f"degree[{i}]={d} must be < n_pts[{i}]={n} "
                f"(need at least degree + 1 sample points)."
            )
    return deg_tuple


def _fit_from_values(
    components: list[npt.NDArray[np.floating[Any]]],
    node_arrays: list[npt.NDArray[np.floating[Any]]],
    degree_tuple: tuple[int, ...] | None,
    tol: float | None,
) -> npt.NDArray[np.floating[Any]]:
    """Fit Bernstein coefficients from per-component value arrays.

    This is the shared core for both :func:`interpolate_bezier` and
    :func:`fit_bezier`.

    Args:
        components (list[npt.NDArray[np.floating[Any]]]): Per-component value
            arrays, each with shape ``(*n_pts_per_dir)``.
        node_arrays (list[npt.NDArray[np.floating[Any]]]): One 1D node array
            per parametric direction.
        degree_tuple (tuple[int, ...] | None): Degree per direction, or *None*
            for exact interpolation (``degree = n_pts - 1``).
        tol (float | None): SVD truncation tolerance.

    Returns:
        npt.NDArray[np.floating[Any]]: Control points with shape
        ``(*orders, rank)`` where ``orders[d] = degree[d] + 1``.
    """
    ndim = len(node_arrays)

    # Build pseudo-inverse matrices per direction
    pinvs = [
        _build_bernstein_pinv(
            node_arrays[d],
            tol,
            degree_tuple[d] if degree_tuple is not None else None,
        )
        for d in range(ndim)
    ]

    # Interpolate each component to Bernstein coefficients
    ctrl_components: list[npt.NDArray[np.floating[Any]]] = []
    for comp_values in components:
        result = comp_values.copy()
        for dim in range(ndim):
            if result.shape[dim] == 1 and pinvs[dim].shape == (1, 1):
                continue
            result = np.tensordot(pinvs[dim], result, axes=([1], [dim]))
            result = np.moveaxis(result, 0, dim)
        ctrl_components.append(result)

    return np.stack(ctrl_components, axis=-1)


def interpolate_bezier(
    func: Callable[..., npt.ArrayLike],
    n_pts: int | Sequence[int],
    *,
    degree: int | Sequence[int] | None = None,
    nodes: (
        Literal["chebyshev", "uniform"]
        | PointsLattice
        | npt.NDArray[np.floating[Any]]
        | Sequence[npt.NDArray[np.floating[Any]]]
        | None
    ) = None,
    tol: float | None = None,
) -> Bezier:
    """Interpolate a callable function into a Bézier in Bernstein form.

    Evaluates ``func`` on a tensor-product grid of interpolation nodes and
    recovers the Bernstein coefficients via truncated SVD.

    The parametric dimension is determined by the length of ``n_pts`` (when
    given as a sequence) or defaults to 1 (when given as a scalar ``int``).

    The output dtype is inferred from the return value of ``func``.

    Args:
        func (Callable[..., npt.ArrayLike]): Function to interpolate.  Called
            as ``func(lattice)`` where ``lattice`` is a
            :class:`~pantr.quad.PointsLattice` representing the tensor-product
            sampling grid.  The callable may also accept a plain ``ndarray``
            (for compatibility with :meth:`Bezier.evaluate` signatures).
            Must return an array of shape ``(n_total,)`` for a scalar-valued
            function or ``(n_total, rank)`` for a vector-valued function,
            where ``n_total = prod(n_pts)``.
        n_pts (int | Sequence[int]): Number of sample points per parametric
            direction.  A single ``int`` gives a 1D Bézier.
        degree (int | Sequence[int] | None): Polynomial degree per direction.
            If *None* (default), ``degree = n_pts - 1`` (exact interpolation).
            If provided, must satisfy ``degree < n_pts`` in each direction;
            the result is a least-squares approximation.
        nodes: Interpolation node selection.

            - ``None`` or ``"chebyshev"`` (default): modified Chebyshev-Lobatto
              nodes on [0, 1].
            - ``"uniform"``: equispaced nodes on [0, 1].
            - A :class:`~pantr.quad.PointsLattice`: custom tensor-product grid.
            - A 1D ``ndarray``: custom nodes broadcast to all directions.
            - A sequence of 1D ``ndarray`` values: per-direction custom nodes.
        tol (float | None): SVD truncation tolerance. If *None*, uses a
            default based on machine epsilon.

    Returns:
        ~pantr.bezier.Bezier: A non-rational Bézier whose evaluation
        approximates ``func``.

    Raises:
        ValueError: If ``n_pts`` values are < 1, *degree* >= *n_pts*, *nodes*
            is inconsistent with *n_pts*, or the callable returns an
            unexpected shape.

    Example:
        >>> from pantr.bezier import Bezier
        >>> import numpy as np
        >>> b = interpolate_bezier(lambda lattice: lattice.get_all_points()[:, 0] ** 2, [5])
        >>> b.degree
        (4,)
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    n_pts_tuple = _validate_n_pts(n_pts)
    degree_tuple = _validate_degree(degree, n_pts_tuple)

    # Resolve nodes and build PointsLattice (default to float64 for generated nodes)
    node_arrays = _resolve_nodes(n_pts_tuple, nodes, np.dtype(np.float64))
    lattice = PointsLattice(node_arrays)
    n_total = int(np.prod(n_pts_tuple))

    # Evaluate function and infer dtype from the return value
    raw_untyped = np.asarray(func(lattice))
    if not np.issubdtype(raw_untyped.dtype, np.floating):
        raw_untyped = raw_untyped.astype(np.float64)
    raw: npt.NDArray[np.floating[Any]] = raw_untyped

    if raw.ndim == 1:
        if raw.shape[0] != n_total:
            raise ValueError(
                f"Function returned shape {raw.shape}, expected ({n_total},) or ({n_total}, rank)."
            )
        values = raw.reshape(n_pts_tuple)
    elif raw.ndim == 2:  # noqa: PLR2004
        if raw.shape[0] != n_total:
            raise ValueError(
                f"Function returned shape {raw.shape}, expected ({n_total},) or ({n_total}, rank)."
            )
        values = raw.reshape(*n_pts_tuple, raw.shape[1])
    else:
        raise ValueError(
            f"Function returned shape {raw.shape}, expected ({n_total},) or ({n_total}, rank)."
        )

    components = split_components(values, n_pts_tuple)
    ctrl = _fit_from_values(components, node_arrays, degree_tuple, tol)
    return BezierCls(ctrl, is_rational=False)


def _resolve_nodes_from_user(
    nodes: PointsLattice | npt.NDArray[np.floating[Any]] | Sequence[npt.NDArray[np.floating[Any]]],
    n_pts_tuple: tuple[int, ...],
    dtype: np.dtype[np.float32] | np.dtype[np.float64],
) -> list[npt.NDArray[np.floating[Any]]]:
    """Resolve user-provided tensor-product nodes for :func:`fit_bezier`.

    Args:
        nodes: A :class:`~pantr.quad.PointsLattice`, a single 1D array
            (for 1D fitting), or a sequence of 1D arrays.
        n_pts_tuple (tuple[int, ...]): Expected number of points per direction.
        dtype (np.dtype): Floating dtype.

    Returns:
        list[npt.NDArray[np.floating[Any]]]: One 1D node array per direction.

    Raises:
        ValueError: If nodes are inconsistent with the expected grid shape.
    """
    if isinstance(nodes, PointsLattice):
        if nodes.dim != len(n_pts_tuple):
            raise ValueError(
                f"PointsLattice has {nodes.dim} dimensions, but values have "
                f"{len(n_pts_tuple)} parametric dimensions."
            )
        node_list: list[npt.NDArray[np.floating[Any]]] = [
            np.asarray(a, dtype=dtype) for a in nodes.pts_per_dir
        ]
        for i, (arr_i, n) in enumerate(zip(node_list, n_pts_tuple, strict=True)):
            if arr_i.shape[0] != n:
                raise ValueError(
                    f"PointsLattice direction {i} has {arr_i.shape[0]} nodes, expected {n}."
                )
        return node_list

    # Single 1D array
    if isinstance(nodes, np.ndarray) and nodes.ndim == 1:
        if len(n_pts_tuple) != 1:
            raise ValueError(
                f"A single node array implies 1D fitting, but values have "
                f"{len(n_pts_tuple)} parametric dimensions."
            )
        arr: npt.NDArray[np.floating[Any]] = nodes.astype(dtype, copy=False)
        if arr.shape[0] != n_pts_tuple[0]:
            raise ValueError(
                f"Node array length {arr.shape[0]} does not match "
                f"values grid size {n_pts_tuple[0]}."
            )
        return [arr]

    # Sequence of arrays
    node_list_seq: list[npt.NDArray[np.floating[Any]]] = [np.asarray(a, dtype=dtype) for a in nodes]
    if len(node_list_seq) != len(n_pts_tuple):
        raise ValueError(f"Expected {len(n_pts_tuple)} node arrays, got {len(node_list_seq)}.")
    for i, (arr_i, n) in enumerate(zip(node_list_seq, n_pts_tuple, strict=True)):
        if arr_i.ndim != 1 or arr_i.shape[0] != n:
            raise ValueError(
                f"Node array for direction {i} has shape {arr_i.shape}, expected ({n},)."
            )
    return node_list_seq


def _is_scattered_nodes(
    nodes: (
        PointsLattice | npt.NDArray[np.floating[Any]] | Sequence[npt.NDArray[np.floating[Any]]]
    ),
) -> bool:
    """Check whether *nodes* represents scattered (non-tensor-product) points.

    Args:
        nodes: The nodes argument passed to :func:`fit_bezier`.

    Returns:
        bool: ``True`` if *nodes* is a 2D array (scattered points),
        ``False`` if it is a :class:`~pantr.quad.PointsLattice`, a 1D array,
        or a sequence of 1D arrays (tensor-product).
    """
    if isinstance(nodes, PointsLattice):
        return False
    return isinstance(nodes, np.ndarray) and nodes.ndim == 2  # noqa: PLR2004


def fit_bezier(  # noqa: PLR0912
    values: npt.ArrayLike,
    nodes: (
        PointsLattice | npt.NDArray[np.floating[Any]] | Sequence[npt.NDArray[np.floating[Any]]]
    ),
    *,
    degree: int | Sequence[int] | None = None,
    tol: float | None = None,
) -> Bezier:
    """Construct a Bézier from pre-evaluated sample values at known nodes.

    The output dtype is inferred from *values*.

    Supports two point layouts:

    - **Tensor-product** (a :class:`~pantr.quad.PointsLattice`, a single 1D
      array, or a sequence of 1D arrays): values must have shape
      ``(*n_pts_per_dir)`` (scalar) or ``(*n_pts_per_dir, rank)`` (vector).
      Per-direction SVD pseudo-inverse is used (efficient).
    - **Scattered** (a 2D ``ndarray`` of shape ``(n_pts, dim)``): values must
      have shape ``(n_pts,)`` (scalar) or ``(n_pts, rank)`` (vector).
      ``degree`` is required. A full tensor-product Vandermonde is built and
      solved via SVD.

    Args:
        values (npt.ArrayLike): Sample values.
        nodes: Interpolation nodes.

            - A :class:`~pantr.quad.PointsLattice`: tensor-product grid.
            - A 1D ``ndarray``: 1D tensor-product (single direction).
            - A sequence of 1D ``ndarray`` values: N-D tensor-product.
            - A 2D ``ndarray`` of shape ``(n_pts, dim)``: scattered points.
        degree (int | Sequence[int] | None): Polynomial degree per direction.
            Required for scattered nodes.  If *None* (default) for
            tensor-product nodes, ``degree = n_pts - 1`` (exact
            interpolation).  Must satisfy
            ``prod(degree + 1) <= n_pts`` for scattered nodes and
            ``degree < n_pts`` per direction for tensor-product.
        tol (float | None): SVD truncation tolerance. If *None*, uses a
            default based on machine epsilon.

    Returns:
        ~pantr.bezier.Bezier: A non-rational Bézier.

    Raises:
        ValueError: If *nodes* are inconsistent with *values*, *degree* is
            invalid, or *degree* is missing for scattered nodes.
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    values_untyped = np.asarray(values)
    if not np.issubdtype(values_untyped.dtype, np.floating):
        values_untyped = values_untyped.astype(np.float64)
    values_arr: npt.NDArray[np.floating[Any]] = values_untyped
    dtype_obj = values_arr.dtype

    # --- Scattered path ---
    if _is_scattered_nodes(nodes):
        pts: npt.NDArray[np.floating[Any]] = np.asarray(nodes, dtype=dtype_obj)
        ndim_param = pts.shape[1]
        n_pts = pts.shape[0]

        if degree is None:
            raise ValueError("degree is required for scattered (non-tensor-product) nodes.")

        degree_tuple = (degree,) * ndim_param if isinstance(degree, int) else tuple(degree)
        if len(degree_tuple) != ndim_param:
            raise ValueError(
                f"degree has {len(degree_tuple)} entries but nodes have {ndim_param} columns."
            )
        n_coeffs = int(np.prod(tuple(d + 1 for d in degree_tuple)))
        if n_coeffs > n_pts:
            raise ValueError(
                f"Underdetermined: {n_coeffs} coefficients but only {n_pts} sample points."
            )

        # Validate values shape
        if values_arr.ndim == 1:
            if values_arr.shape[0] != n_pts:
                raise ValueError(
                    f"values has {values_arr.shape[0]} entries but nodes has {n_pts} points."
                )
        elif values_arr.ndim == 2:  # noqa: PLR2004
            if values_arr.shape[0] != n_pts:
                raise ValueError(
                    f"values has {values_arr.shape[0]} rows but nodes has {n_pts} points."
                )
        else:
            raise ValueError(
                f"For scattered nodes, values must be 1D (scalar) or 2D "
                f"(vector), got {values_arr.ndim}D."
            )

        ctrl = _fit_from_scattered(values_arr, pts, degree_tuple, tol)
        return BezierCls(ctrl, is_rational=False)

    # --- Tensor-product path ---
    # Determine parametric dimension from nodes
    if isinstance(nodes, PointsLattice):
        ndim_param = nodes.dim
    elif isinstance(nodes, np.ndarray) and nodes.ndim == 1:
        ndim_param = 1
    else:
        ndim_param = len(nodes)

    # Infer n_pts from values shape: first ndim_param dimensions are the grid
    if values_arr.ndim == ndim_param:
        # Scalar-valued
        n_pts_tuple = values_arr.shape
    elif values_arr.ndim == ndim_param + 1:
        # Vector-valued
        n_pts_tuple = values_arr.shape[:ndim_param]
    else:
        raise ValueError(
            f"values has {values_arr.ndim} dimensions, expected {ndim_param} "
            f"(scalar) or {ndim_param + 1} (vector) for {ndim_param}D fitting."
        )

    degree_tuple_tp = _validate_degree(degree, n_pts_tuple)
    node_arrays = _resolve_nodes_from_user(nodes, n_pts_tuple, dtype_obj)
    components = split_components(values_arr, n_pts_tuple)
    ctrl = _fit_from_values(components, node_arrays, degree_tuple_tp, tol)
    return BezierCls(ctrl, is_rational=False)
