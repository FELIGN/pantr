"""Bernstein interpolation: construct a Bézier from function samples.

Provides :func:`_interpolate_bezier`, which evaluates a callable on a
tensor-product grid of interpolation nodes and recovers the Bernstein
coefficients via truncated SVD.

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

from ..quad import get_modified_chebyshev_nodes_1d

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


def _resolve_nodes(
    n_pts: int | Sequence[int],
    nodes: (
        Literal["chebyshev", "uniform"]
        | npt.NDArray[np.floating[Any]]
        | Sequence[npt.NDArray[np.floating[Any]]]
        | None
    ),
    dtype: np.dtype[np.float32] | np.dtype[np.float64],
) -> list[npt.NDArray[np.floating[Any]]]:
    """Resolve the *nodes* parameter into a list of 1D node arrays.

    Args:
        n_pts (int | Sequence[int]): Number of sample points per direction.
        nodes: Node specification — ``None`` or ``"chebyshev"`` for modified
            Chebyshev-Lobatto nodes, ``"uniform"`` for equispaced nodes, a
            single 1D array (broadcast to every direction), or a sequence of
            1D arrays (one per direction).
        dtype (np.dtype): Floating dtype for generated nodes.

    Returns:
        list[npt.NDArray[np.floating[Any]]]: One 1D node array per parametric
        direction.

    Raises:
        ValueError: If *nodes* is inconsistent with *n_pts*.
    """
    # Normalise n_pts to a tuple
    if isinstance(n_pts, int):
        n_pts_tuple: tuple[int, ...] = (n_pts,)
    else:
        n_pts_tuple = tuple(n_pts)

    if nodes is None or (isinstance(nodes, str) and nodes == "chebyshev"):
        return [
            get_modified_chebyshev_nodes_1d(n, dtype)
            if n >= 2  # noqa: PLR2004
            else np.array([0.5], dtype=dtype)
            for n in n_pts_tuple
        ]

    if isinstance(nodes, str) and nodes == "uniform":
        result: list[npt.NDArray[np.floating[Any]]] = [
            np.linspace(0.0, 1.0, n, dtype=dtype) for n in n_pts_tuple
        ]
        return result

    # User-provided nodes
    if isinstance(nodes, np.ndarray) and nodes.ndim == 1:
        # Single array — broadcast to all directions
        arr: npt.NDArray[np.floating[Any]] = nodes.astype(dtype, copy=False)
        for n in n_pts_tuple:
            if arr.shape[0] != n:
                raise ValueError(f"Node array length {arr.shape[0]} does not match n_pts={n}.")
        return [arr] * len(n_pts_tuple)

    # Sequence of arrays
    node_list: list[npt.NDArray[np.floating[Any]]] = [np.asarray(a, dtype=dtype) for a in nodes]
    if len(node_list) != len(n_pts_tuple):
        raise ValueError(f"Expected {len(n_pts_tuple)} node arrays, got {len(node_list)}.")
    for i, (arr, n) in enumerate(zip(node_list, n_pts_tuple, strict=True)):
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(
                f"Node array for direction {i} has shape {arr.shape}, expected ({n},)."
            )
    return node_list


def _build_bernstein_pinv(
    nodes: npt.NDArray[np.floating[Any]],
    tol: float | None = None,
) -> npt.NDArray[np.floating[Any]]:
    """Build the Bernstein pseudo-inverse matrix for arbitrary 1D nodes.

    Given a set of 1D interpolation nodes, constructs the matrix that maps
    function values at those nodes to Bernstein coefficients via truncated SVD.

    Args:
        nodes (npt.NDArray[np.floating[Any]]): 1D interpolation nodes on
            [0, 1], shape ``(n,)``.
        tol (float | None): SVD truncation tolerance. If *None*, uses
            ``100 * eps``.

    Returns:
        npt.NDArray[np.floating[Any]]: Pseudo-inverse matrix, shape ``(n, n)``.
    """
    from ..basis._basis_core import _tabulate_Bernstein_basis_1D_core  # noqa: PLC0415

    n = nodes.shape[0]
    dtype = nodes.dtype

    if n == 1:
        return np.ones((1, 1), dtype=dtype)

    V = np.empty((n, n), dtype=dtype)
    _tabulate_Bernstein_basis_1D_core(np.int32(n - 1), nodes, V)
    U, sigma, Vt = np.linalg.svd(V, full_matrices=True)

    eps = float(np.finfo(dtype).eps)
    actual_tol = tol if tol is not None else 100.0 * eps
    min_sigma = actual_tol * sigma[0]
    inv_sigma = np.where(sigma >= min_sigma, 1.0 / sigma, 0.0)

    pinv: npt.NDArray[np.floating[Any]] = (Vt.T * inv_sigma[np.newaxis, :]) @ U.T
    return pinv


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


def _split_components(
    values: npt.NDArray[np.floating[Any]],
    grid_shape: tuple[int, ...],
) -> list[npt.NDArray[np.floating[Any]]]:
    """Split function values into per-component arrays.

    Args:
        values (npt.NDArray[np.floating[Any]]): Function output array.
        grid_shape (tuple[int, ...]): Expected grid shape.

    Returns:
        list[npt.NDArray[np.floating[Any]]]: One array per output component.

    Raises:
        ValueError: If values shape is incompatible with grid_shape.
    """
    if values.shape == grid_shape:
        return [values]
    if values.shape[: len(grid_shape)] == grid_shape and values.ndim == len(grid_shape) + 1:
        return [values[..., r] for r in range(values.shape[-1])]
    raise ValueError(
        f"Function returned shape {values.shape}, expected {grid_shape} "
        f"(scalar) or {(*grid_shape, 'rank')} (vector)."
    )


def _interpolate_bezier(
    func: Callable[..., npt.ArrayLike],
    n_pts: int | Sequence[int],
    *,
    nodes: (
        Literal["chebyshev", "uniform"]
        | npt.NDArray[np.floating[Any]]
        | Sequence[npt.NDArray[np.floating[Any]]]
        | None
    ) = None,
    dtype: npt.DTypeLike = np.float64,
    tol: float | None = None,
) -> Bezier:
    """Interpolate a callable function into a Bézier in Bernstein form.

    Evaluates ``func`` on a tensor-product grid of interpolation nodes and
    recovers the Bernstein coefficients via truncated SVD.

    The parametric dimension is determined by the length of ``n_pts`` (when
    given as a sequence) or defaults to 1 (when given as a scalar ``int``).

    Args:
        func (Callable[..., npt.ArrayLike]): Function to interpolate.  Called
            as ``func(x0, x1, ...)`` where each ``xi`` is a 1D array of
            shape ``(n_pts_i,)`` broadcast over a meshgrid (i.e. arrays with
            shape ``(*grid_shape)``).  Must return an array of shape
            ``(*grid_shape)`` for a scalar-valued function or
            ``(*grid_shape, rank)`` for a vector-valued function.
        n_pts (int | Sequence[int]): Number of sample points per parametric
            direction.  Determines degree = ``n_pts - 1`` in each direction.
            A single ``int`` gives a 1D Bézier.
        nodes: Interpolation node selection.

            - ``None`` or ``"chebyshev"`` (default): modified Chebyshev-Lobatto
              nodes on [0, 1].
            - ``"uniform"``: equispaced nodes on [0, 1].
            - A 1D ``ndarray``: custom nodes broadcast to all directions.
            - A sequence of 1D ``ndarray``s: per-direction custom nodes.
        dtype (npt.DTypeLike): Floating dtype for nodes and output.
            Defaults to ``float64``.
        tol (float | None): SVD truncation tolerance. If *None*, uses a
            default based on machine epsilon.

    Returns:
        ~pantr.bezier.Bezier: A non-rational Bézier whose evaluation
        approximates ``func``.  Degree is ``n_pts - 1`` per direction.

    Raises:
        ValueError: If ``n_pts`` values are < 1, or *nodes* is inconsistent
            with *n_pts*.
        ValueError: If the callable returns an unexpected shape.

    Example:
        >>> from pantr.bezier import Bezier
        >>> import numpy as np
        >>> b = Bezier.interpolate(lambda x: x**2, 5)
        >>> b.degree
        (4,)
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    dtype_raw = np.dtype(dtype)
    if not np.issubdtype(dtype_raw, np.floating):
        raise ValueError(f"dtype must be a floating type, got {dtype_raw}.")
    if dtype_raw == np.float32:
        dtype_obj: np.dtype[np.float32] | np.dtype[np.float64] = np.dtype(np.float32)
    else:
        dtype_obj = np.dtype(np.float64)

    n_pts_tuple = _validate_n_pts(n_pts)
    ndim = len(n_pts_tuple)

    # Resolve nodes and build meshgrid
    node_arrays = _resolve_nodes(n_pts_tuple, nodes, dtype_obj)
    grids = [node_arrays[0]] if ndim == 1 else list(np.meshgrid(*node_arrays, indexing="ij"))

    # Evaluate function and split into per-component arrays
    values: npt.NDArray[np.floating[Any]] = np.asarray(func(*grids), dtype=dtype_obj)
    components = _split_components(values, n_pts_tuple)

    # Build pseudo-inverse matrices per direction
    pinvs = [_build_bernstein_pinv(na, tol) for na in node_arrays]

    # Interpolate each component to Bernstein coefficients
    ctrl_components: list[npt.NDArray[np.floating[Any]]] = []
    for comp_values in components:
        result = comp_values.copy()
        for dim in range(ndim):
            if result.shape[dim] == 1:
                continue
            result = np.tensordot(pinvs[dim], result, axes=([1], [dim]))
            result = np.moveaxis(result, 0, dim)
        ctrl_components.append(result)

    ctrl = np.stack(ctrl_components, axis=-1)
    return BezierCls(ctrl, is_rational=False)
