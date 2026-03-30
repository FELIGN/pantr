"""B-spline interpolation, fitting, and L2 projection.

Private implementation for :meth:`~pantr.bspline.Bspline.interpolate`,
:meth:`~pantr.bspline.Bspline.fit`, and
:meth:`~pantr.bspline.Bspline.l2_project`.  All public entry points live on
the :class:`~pantr.bspline.Bspline` class; this module is not part of the
public API.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy import typing as npt

from ..quad import PointsLattice
from ._bspline_space_1d import BsplineSpace1D
from ._bspline_space_factory import greville_abscissae, greville_lattice
from ._bspline_space_nd import BsplineSpace

if TYPE_CHECKING:
    from ._bspline import Bspline


# ---------------------------------------------------------------------------
# 1D collocation matrix assembly
# ---------------------------------------------------------------------------


def _build_collocation_matrix_1d(
    space: BsplineSpace1D,
    nodes: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Build the dense collocation matrix for a 1D B-spline space.

    ``C[i, j] = N_j(node_i)`` where ``N_j`` is the ``j``-th basis function.

    Args:
        space (BsplineSpace1D): The 1D B-spline space.
        nodes (npt.NDArray): Evaluation nodes of shape ``(n_pts,)``.

    Returns:
        npt.NDArray: Dense matrix of shape ``(n_pts, num_basis)``.

    Note:
        No input validation is performed.
    """
    basis_vals, first_basis = space.tabulate_basis(nodes)
    n_pts = nodes.shape[0]
    n_basis = space.num_basis
    order = space.degree + 1

    mat = np.zeros((n_pts, n_basis), dtype=nodes.dtype)
    for i in range(n_pts):
        fb = first_basis[i]
        if space.periodic:
            for j in range(order):
                col = (fb + j) % n_basis
                mat[i, col] += basis_vals[i, j]
        else:
            mat[i, fb : fb + order] = basis_vals[i, :order]

    return mat


def _build_collocation_deriv_matrix_1d(
    space: BsplineSpace1D,
    node: np.float32 | np.float64,
    n_derivs: int,
) -> npt.NDArray[np.float32 | np.float64]:
    """Build rows of the collocation matrix for derivatives at a single point.

    Returns ``n_derivs`` rows, one per derivative order ``1, ..., n_derivs``.

    Args:
        space (BsplineSpace1D): The 1D B-spline space.
        node (float): Single evaluation point.
        n_derivs (int): Number of derivative orders to include.

    Returns:
        npt.NDArray: Matrix of shape ``(n_derivs, num_basis)``.

    Note:
        No input validation is performed.
    """
    pts = np.array([node], dtype=type(node))
    deriv_vals, first_basis = space.tabulate_basis_derivatives(pts, n_derivs)
    # deriv_vals shape: (1, n_derivs+1, degree+1)
    n_basis = space.num_basis
    order = space.degree + 1

    mat: npt.NDArray[np.float32 | np.float64] = np.zeros((n_derivs, n_basis), dtype=pts.dtype)
    fb = first_basis[0]
    for k in range(n_derivs):
        mat[k, fb : fb + order] = deriv_vals[0, k + 1, :order]

    return mat


# ---------------------------------------------------------------------------
# SVD pseudo-inverse solve
# ---------------------------------------------------------------------------

_DEFAULT_TOL_FACTOR: float = 100.0


def _solve_1d(
    mat: npt.NDArray[np.float32 | np.float64],
    rhs: npt.NDArray[np.float32 | np.float64],
    tol: float | None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Solve a 1D linear system via SVD pseudo-inverse with truncation.

    For square systems where the matrix is well-conditioned, uses a direct
    solve. Otherwise uses truncated SVD.

    Args:
        mat (npt.NDArray): Matrix of shape ``(m, n)`` with ``m >= n``.
        rhs (npt.NDArray): Right-hand side of shape ``(m,)`` or ``(m, k)``.
        tol (float | None): SVD truncation tolerance. If *None*, defaults to
            ``100 * machine_epsilon``.

    Returns:
        npt.NDArray: Solution of shape ``(n,)`` or ``(n, k)``.
    """
    m, n = mat.shape
    if m == n:
        # Try direct solve first for square systems.
        try:
            return np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:
            pass

    # Truncated SVD pseudo-inverse for rectangular or singular systems.
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    eps = float(np.finfo(mat.dtype).eps)  # type: ignore[misc]
    threshold = (tol if tol is not None else _DEFAULT_TOL_FACTOR * eps) * s[0]
    s_inv = np.where(s > threshold, 1.0 / s, 0.0)
    # pinv @ rhs = Vt.T @ diag(s_inv) @ U.T @ rhs
    result: npt.NDArray[np.float32 | np.float64]
    if rhs.ndim == 2:  # noqa: PLR2004
        result = vt.T @ (s_inv[:, np.newaxis] * (u.T @ rhs))
    else:
        result = vt.T @ (s_inv * (u.T @ rhs))
    return result


# ---------------------------------------------------------------------------
# Kronecker (per-direction) solve
# ---------------------------------------------------------------------------


def _solve_kronecker(
    matrices: list[npt.NDArray[np.float32 | np.float64]],
    rhs: npt.NDArray[np.float32 | np.float64],
    tol: float | None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Solve the tensor-product system by sequential 1D solves.

    Given 1D matrices ``A_0, A_1, ...`` and a right-hand side tensor of shape
    ``(m_0, m_1, ...)``, solves for the coefficient tensor ``C`` such that
    ``(A_{d-1} ⊗ ... ⊗ A_0) vec(C) = vec(rhs)``.

    Each direction is solved independently: contract ``A_d`` along axis ``d``
    of the current tensor.

    Args:
        matrices (list[npt.NDArray]): One matrix per parametric direction,
            each of shape ``(m_d, n_d)``.
        rhs (npt.NDArray): Tensor of shape ``(m_0, m_1, ...)``.
        tol (float | None): SVD truncation tolerance forwarded to
            :func:`_solve_1d`.

    Returns:
        npt.NDArray: Coefficient tensor of shape ``(n_0, n_1, ...)``.
    """
    result = rhs.copy()
    for d, mat in enumerate(matrices):
        # Move axis d to position 0, solve, move back.
        result = np.moveaxis(result, d, 0)
        shape_rest = result.shape[1:]
        result_2d = result.reshape(mat.shape[0], -1)
        solved = _solve_1d(mat, result_2d, tol)
        result = solved.reshape(mat.shape[1], *shape_rest)
        result = np.moveaxis(result, 0, d)
    return result


# ---------------------------------------------------------------------------
# Function evaluation helper
# ---------------------------------------------------------------------------


def _evaluate_func_on_lattice(
    func: Callable[..., npt.ArrayLike],
    lattice: PointsLattice,
    grid_shape: tuple[int, ...],
) -> tuple[list[npt.NDArray[np.floating[Any]]], np.dtype[np.float32] | np.dtype[np.float64]]:
    """Evaluate a callable on a PointsLattice and split into components.

    Args:
        func (Callable): Function receiving a :class:`PointsLattice`.
        lattice (PointsLattice): Tensor-product sampling grid.
        grid_shape (tuple[int, ...]): Expected grid shape
            ``(n_pts_0, ..., n_pts_{d-1})``.

    Returns:
        tuple: ``(components, dtype)`` where *components* is a list of arrays
        each with shape ``grid_shape``, and *dtype* is the inferred floating
        dtype.

    Raises:
        ValueError: If the callable returns an unexpected shape.
    """
    n_total = int(np.prod(grid_shape))
    raw_untyped = np.asarray(func(lattice))
    if not np.issubdtype(raw_untyped.dtype, np.floating):
        raw_untyped = raw_untyped.astype(np.float64)
    raw: npt.NDArray[np.floating[Any]] = raw_untyped

    if raw.ndim == 1:
        if raw.shape[0] != n_total:
            raise ValueError(
                f"Function returned shape {raw.shape}, expected ({n_total},) or ({n_total}, rank)."
            )
        values = raw.reshape(grid_shape)
    elif raw.ndim == 2:  # noqa: PLR2004
        if raw.shape[0] != n_total:
            raise ValueError(
                f"Function returned shape {raw.shape}, expected ({n_total},) or ({n_total}, rank)."
            )
        values = raw.reshape(*grid_shape, raw.shape[1])
    else:
        raise ValueError(
            f"Function returned shape {raw.shape}, expected ({n_total},) or ({n_total}, rank)."
        )

    components = _split_components(values, grid_shape)
    _f32: np.dtype[np.float32] = np.dtype(np.float32)
    _f64: np.dtype[np.float64] = np.dtype(np.float64)
    out_dtype = _f32 if raw.dtype == _f32 else _f64
    return components, out_dtype


def _split_components(
    values: npt.NDArray[np.floating[Any]],
    grid_shape: tuple[int, ...],
) -> list[npt.NDArray[np.floating[Any]]]:
    """Split function values into per-component arrays.

    Args:
        values (npt.NDArray): Function output array.
        grid_shape (tuple[int, ...]): Expected grid shape.

    Returns:
        list[npt.NDArray]: One array per output component.

    Raises:
        ValueError: If values shape is incompatible with *grid_shape*.
    """
    if values.shape == grid_shape:
        return [values]
    if values.shape[: len(grid_shape)] == grid_shape and values.ndim == len(grid_shape) + 1:
        return [values[..., r] for r in range(values.shape[-1])]
    raise ValueError(
        f"Values have shape {values.shape}, expected {grid_shape} "
        f"(scalar) or {(*grid_shape, 'rank')} (vector)."
    )


# ---------------------------------------------------------------------------
# Node resolution
# ---------------------------------------------------------------------------


def _resolve_nodes(
    space: BsplineSpace,
    nodes: (
        Literal["greville"]
        | PointsLattice
        | npt.NDArray[np.floating[Any]]
        | Sequence[npt.NDArray[np.floating[Any]]]
        | None
    ),
) -> list[npt.NDArray[np.float32 | np.float64]]:
    """Resolve the *nodes* argument into per-direction 1D arrays.

    Args:
        space (BsplineSpace): The target B-spline space.
        nodes: Node specification (see :meth:`Bspline.interpolate`).

    Returns:
        list[npt.NDArray]: One 1D node array per parametric direction.

    Raises:
        ValueError: If nodes are inconsistent with the space.
    """
    if nodes is None or nodes == "greville":
        lattice = greville_lattice(space)
        return list(lattice.pts_per_dir)

    if isinstance(nodes, PointsLattice):
        if nodes.dim != space.dim:
            raise ValueError(f"PointsLattice has {nodes.dim} dimensions, expected {space.dim}.")
        return list(nodes.pts_per_dir)

    if isinstance(nodes, np.ndarray) and nodes.ndim == 1:
        if space.dim != 1:
            raise ValueError(f"A single 1D array implies 1D, but space has {space.dim} dimensions.")
        return [nodes.astype(space.dtype, copy=False)]

    # Sequence of 1D arrays.
    node_list = list(nodes)
    if len(node_list) != space.dim:
        raise ValueError(f"Expected {space.dim} node arrays, got {len(node_list)}.")
    return [np.asarray(a, dtype=space.dtype) for a in node_list]


def _is_scattered_nodes(
    nodes: (
        PointsLattice | npt.NDArray[np.floating[Any]] | Sequence[npt.NDArray[np.floating[Any]]]
    ),
) -> bool:
    """Check whether *nodes* represents scattered (non-tensor-product) points.

    Args:
        nodes: The nodes argument.

    Returns:
        bool: ``True`` if *nodes* is a 2D ndarray (scattered), ``False``
        otherwise.
    """
    if isinstance(nodes, PointsLattice):
        return False
    return isinstance(nodes, np.ndarray) and nodes.ndim == 2  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def _interpolate_bspline(
    func: Callable[..., npt.ArrayLike],
    space: BsplineSpace,
    *,
    nodes: (
        Literal["greville"]
        | PointsLattice
        | npt.NDArray[np.floating[Any]]
        | Sequence[npt.NDArray[np.floating[Any]]]
        | None
    ) = None,
    boundary_derivatives: Sequence[tuple[int, ...] | None] | None = None,
    tol: float | None = None,
) -> Bspline:
    """Interpolate a callable onto a B-spline space.

    See :meth:`~pantr.bspline.Bspline.interpolate` for the full docstring.

    Args:
        func (Callable): Function to interpolate.
        space (BsplineSpace): Target B-spline space.
        nodes: Interpolation node selection.
        boundary_derivatives: Per-direction boundary derivative constraints.
        tol (float | None): SVD truncation tolerance.

    Returns:
        Bspline: The interpolated B-spline.
    """
    from ._bspline import Bspline  # noqa: PLC0415

    if not isinstance(space, BsplineSpace):
        raise TypeError(f"Expected BsplineSpace, got {type(space).__name__}")

    node_arrays = _resolve_nodes(space, nodes)

    # Build collocation matrices per direction (possibly with derivative rows).
    matrices = _build_collocation_matrices(space, node_arrays, boundary_derivatives)

    # Build sampling lattice and evaluate function.
    lattice = PointsLattice(node_arrays)
    grid_shape = tuple(a.shape[0] for a in node_arrays)
    components, out_dtype = _evaluate_func_on_lattice(func, lattice, grid_shape)

    # Modify RHS for boundary derivative constraints.
    if boundary_derivatives is not None:
        components = _apply_boundary_deriv_rhs(
            func, space, node_arrays, components, boundary_derivatives
        )

    # Solve per-component via Kronecker structure.
    ctrl_components: list[npt.NDArray[np.floating[Any]]] = []
    for comp in components:
        coeffs = _solve_kronecker(matrices, comp.astype(out_dtype), tol)
        ctrl_components.append(coeffs)

    ctrl = np.stack(ctrl_components, axis=-1)
    return Bspline(space, ctrl)


def _build_collocation_matrices(
    space: BsplineSpace,
    node_arrays: list[npt.NDArray[np.float32 | np.float64]],
    boundary_derivatives: Sequence[tuple[int, ...] | None] | None,
) -> list[npt.NDArray[np.float32 | np.float64]]:
    """Build per-direction collocation matrices, with optional derivative rows.

    Args:
        space (BsplineSpace): Target B-spline space.
        node_arrays (list[npt.NDArray]): Per-direction node arrays.
        boundary_derivatives: Per-direction ``(n_left, n_right)`` or ``None``.

    Returns:
        list[npt.NDArray]: One collocation matrix per direction.
    """
    matrices: list[npt.NDArray[np.float32 | np.float64]] = []
    for d, s1d in enumerate(space.spaces):
        mat = _build_collocation_matrix_1d(s1d, node_arrays[d])

        if boundary_derivatives is not None and d < len(boundary_derivatives):
            bd = boundary_derivatives[d]
            if bd is not None and not s1d.periodic:
                n_left, n_right = bd
                a, b = s1d.domain
                dtype = node_arrays[d].dtype

                if n_left > 0:
                    deriv_rows = _build_collocation_deriv_matrix_1d(s1d, dtype.type(a), n_left)
                    # Replace rows 1..n_left with derivative rows.
                    mat[1 : 1 + n_left, :] = deriv_rows

                if n_right > 0:
                    deriv_rows = _build_collocation_deriv_matrix_1d(s1d, dtype.type(b), n_right)
                    # Replace rows -1-n_right..-1 with derivative rows.
                    mat[-1 - n_right : -1, :] = deriv_rows

        matrices.append(mat)
    return matrices


def _apply_boundary_deriv_rhs(
    func: Callable[..., npt.ArrayLike],
    space: BsplineSpace,
    node_arrays: list[npt.NDArray[np.float32 | np.float64]],
    components: list[npt.NDArray[np.floating[Any]]],
    boundary_derivatives: Sequence[tuple[int, ...] | None],
) -> list[npt.NDArray[np.floating[Any]]]:
    """Replace RHS entries corresponding to derivative rows with zero.

    For boundary derivative constraints, the function values at the boundary
    nodes adjacent to the endpoints are replaced by the derivative values.
    Since we don't have derivative information from the callable (which only
    provides values), the derivative RHS entries are set to zero — meaning
    we constrain the derivatives to be zero at the boundaries.

    Args:
        func (Callable): The original callable.
        space (BsplineSpace): Target B-spline space.
        node_arrays (list[npt.NDArray]): Per-direction node arrays.
        components (list[npt.NDArray]): Per-component value arrays.
        boundary_derivatives: Per-direction ``(n_left, n_right)`` or ``None``.

    Returns:
        list[npt.NDArray]: Modified components with zero derivative entries.
    """
    modified = [c.copy() for c in components]
    for d, s1d in enumerate(space.spaces):
        if d >= len(boundary_derivatives):
            continue
        bd = boundary_derivatives[d]
        if bd is None or s1d.periodic:
            continue
        n_left, n_right = bd
        for comp in modified:
            if n_left > 0:
                slices = [slice(None)] * comp.ndim
                slices[d] = slice(1, 1 + n_left)
                comp[tuple(slices)] = 0.0
            if n_right > 0:
                slices = [slice(None)] * comp.ndim
                slices[d] = slice(-1 - n_right, -1)
                comp[tuple(slices)] = 0.0
    return modified


# ---------------------------------------------------------------------------
# Fit (from pre-evaluated values)
# ---------------------------------------------------------------------------


def _fit_bspline(
    values: npt.ArrayLike,
    nodes: (
        PointsLattice | npt.NDArray[np.floating[Any]] | Sequence[npt.NDArray[np.floating[Any]]]
    ),
    space: BsplineSpace,
    *,
    tol: float | None = None,
) -> Bspline:
    """Fit a B-spline from pre-evaluated sample values.

    See :meth:`~pantr.bspline.Bspline.fit` for the full docstring.

    Args:
        values (npt.ArrayLike): Sample values at the nodes.
        nodes: Interpolation nodes.
        space (BsplineSpace): Target B-spline space.
        tol (float | None): SVD truncation tolerance.

    Returns:
        Bspline: The fitted B-spline.
    """
    if not isinstance(space, BsplineSpace):
        raise TypeError(f"Expected BsplineSpace, got {type(space).__name__}")

    vals = np.asarray(values)
    if not np.issubdtype(vals.dtype, np.floating):
        vals = vals.astype(np.float64)

    if _is_scattered_nodes(nodes):
        return _fit_from_scattered(vals, nodes, space, tol)  # type: ignore[arg-type]

    return _fit_from_tensor_product(vals, nodes, space, tol)


def _fit_from_tensor_product(
    values: npt.NDArray[np.floating[Any]],
    nodes: (
        PointsLattice | npt.NDArray[np.floating[Any]] | Sequence[npt.NDArray[np.floating[Any]]]
    ),
    space: BsplineSpace,
    tol: float | None,
) -> Bspline:
    """Fit from tensor-product nodes via Kronecker solve.

    Args:
        values (npt.NDArray): Sample values.
        nodes: Tensor-product nodes.
        space (BsplineSpace): Target B-spline space.
        tol (float | None): SVD truncation tolerance.

    Returns:
        Bspline: The fitted B-spline.
    """
    from ._bspline import Bspline  # noqa: PLC0415

    # Resolve nodes to per-direction arrays.
    if isinstance(nodes, PointsLattice):
        if nodes.dim != space.dim:
            raise ValueError(f"PointsLattice has {nodes.dim} dimensions, expected {space.dim}.")
        node_arrays = list(nodes.pts_per_dir)
    elif isinstance(nodes, np.ndarray) and nodes.ndim == 1:
        if space.dim != 1:
            raise ValueError(f"A single 1D array implies 1D, but space has {space.dim} dimensions.")
        node_arrays = [nodes.astype(space.dtype, copy=False)]
    else:
        node_arrays = [np.asarray(a, dtype=space.dtype) for a in nodes]
        if len(node_arrays) != space.dim:
            raise ValueError(f"Expected {space.dim} node arrays, got {len(node_arrays)}.")

    grid_shape = tuple(a.shape[0] for a in node_arrays)
    components = _split_components(values, grid_shape)

    matrices = [
        _build_collocation_matrix_1d(s, n) for s, n in zip(space.spaces, node_arrays, strict=True)
    ]

    ctrl_components: list[npt.NDArray[np.floating[Any]]] = []
    for comp in components:
        coeffs = _solve_kronecker(matrices, comp, tol)
        ctrl_components.append(coeffs)

    ctrl = np.stack(ctrl_components, axis=-1)
    return Bspline(space, ctrl)


def _fit_from_scattered(
    values: npt.NDArray[np.floating[Any]],
    nodes: npt.NDArray[np.floating[Any]],
    space: BsplineSpace,
    tol: float | None,
) -> Bspline:
    """Fit from scattered (non-tensor-product) nodes via full SVD solve.

    Args:
        values (npt.NDArray): Sample values of shape ``(n_pts,)`` or
            ``(n_pts, rank)``.
        nodes (npt.NDArray): Scattered nodes of shape ``(n_pts, dim)``.
        space (BsplineSpace): Target B-spline space.
        tol (float | None): SVD truncation tolerance.

    Returns:
        Bspline: The fitted B-spline.

    Raises:
        ValueError: If the system is underdetermined.
    """
    from ._bspline import Bspline  # noqa: PLC0415

    n_pts = nodes.shape[0]
    ndim = nodes.shape[1] if nodes.ndim == 2 else 1  # noqa: PLR2004

    if ndim != space.dim:
        raise ValueError(f"Scattered nodes have {ndim} columns, expected {space.dim}.")

    n_total_basis = space.num_total_basis
    if n_pts < n_total_basis:
        raise ValueError(
            f"Underdetermined system: {n_pts} points < {n_total_basis} basis functions."
        )

    # Build the full collocation matrix by evaluating each 1D basis and
    # computing the row-wise outer product.
    colloc = _build_nd_collocation_matrix(space, nodes)

    # Determine rank from values.
    if values.ndim == 1:
        coeffs_flat = _solve_1d(colloc, values, tol)
        ctrl = coeffs_flat.reshape(*space.num_basis, 1)
    else:
        coeffs_flat = _solve_1d(colloc, values, tol)
        rank = values.shape[1]
        ctrl = coeffs_flat.reshape(*space.num_basis, rank)

    return Bspline(space, ctrl)


def _build_nd_collocation_matrix(
    space: BsplineSpace,
    pts: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.floating[Any]]:
    """Build the full N-D B-spline collocation matrix for scattered points.

    Args:
        space (BsplineSpace): Target B-spline space.
        pts (npt.NDArray): Points of shape ``(n_pts, dim)`` or ``(n_pts,)``
            for 1D.

    Returns:
        npt.NDArray: Collocation matrix of shape ``(n_pts, n_total_basis)``.
    """
    n_pts = pts.shape[0]
    ndim = space.dim

    if pts.ndim == 1:
        pts = pts[:, np.newaxis]

    # Evaluate 1D bases per direction.
    basis_per_dir: list[npt.NDArray[np.floating[Any]]] = []
    for d, s1d in enumerate(space.spaces):
        mat_1d = _build_collocation_matrix_1d(s1d, pts[:, d])
        basis_per_dir.append(mat_1d)

    # Row-wise outer product (Kronecker-like).
    result = basis_per_dir[0]
    for d in range(1, ndim):
        # result: (n_pts, prod(n_basis_0..d-1))
        # basis_per_dir[d]: (n_pts, n_basis_d)
        result = (result[:, :, np.newaxis] * basis_per_dir[d][:, np.newaxis, :]).reshape(n_pts, -1)

    return result


# ---------------------------------------------------------------------------
# L2 projection
# ---------------------------------------------------------------------------


def _l2_project_bspline(  # noqa: PLR0913
    func: Callable[..., npt.ArrayLike],
    space: BsplineSpace,
    *,
    n_quad: int | Sequence[int] | None = None,
    quadrature: Literal["gauss-legendre", "gauss-lobatto"] = "gauss-legendre",
    boundary_interpolation: bool | Sequence[tuple[bool, bool]] = False,
    tol: float | None = None,
) -> Bspline:
    """L2-project a callable onto a B-spline space.

    See :meth:`~pantr.bspline.Bspline.l2_project` for the full docstring.

    Args:
        func (Callable): Function to project.
        space (BsplineSpace): Target B-spline space.
        n_quad (int | Sequence[int] | None): Quadrature points per element.
        quadrature: Quadrature rule type.
        boundary_interpolation: Boundary interpolation flags.
        tol (float | None): SVD truncation tolerance.

    Returns:
        Bspline: The projected B-spline.
    """
    from ..quad import get_gauss_legendre_1d, get_gauss_lobatto_legendre_1d  # noqa: PLC0415
    from ._bspline import Bspline  # noqa: PLC0415

    if not isinstance(space, BsplineSpace):
        raise TypeError(f"Expected BsplineSpace, got {type(space).__name__}")

    ndim = space.dim

    # Resolve n_quad per direction.
    if n_quad is None:
        n_quads = tuple(s.degree + 1 for s in space.spaces)
    elif isinstance(n_quad, int):
        n_quads = tuple(n_quad for _ in range(ndim))
    else:
        n_quads = tuple(n_quad)
        if len(n_quads) != ndim:
            raise ValueError(f"n_quad has length {len(n_quads)}, expected {ndim}")

    # Resolve boundary interpolation flags.
    bi_flags = _resolve_boundary_interpolation(boundary_interpolation, space)

    # Build per-direction mass matrices and quadrature-sampled nodes.
    mass_matrices: list[npt.NDArray[np.float32 | np.float64]] = []
    quad_nodes_per_dir: list[npt.NDArray[np.float32 | np.float64]] = []
    quad_weights_per_dir: list[npt.NDArray[np.float32 | np.float64]] = []

    quad_func = (
        get_gauss_legendre_1d if quadrature == "gauss-legendre" else get_gauss_lobatto_legendre_1d
    )

    for d, s1d in enumerate(space.spaces):
        nq = n_quads[d]
        mass, q_nodes, q_weights = _assemble_mass_and_quad_1d(s1d, nq, quad_func)

        # Apply boundary interpolation if requested.
        bi_left, bi_right = bi_flags[d]
        if bi_left and not s1d.periodic:
            a = s1d.domain[0]
            colloc_row = _build_collocation_matrix_1d(s1d, np.array([a], dtype=s1d.dtype))
            mass[0, :] = colloc_row[0, :]

        if bi_right and not s1d.periodic:
            b = s1d.domain[1]
            colloc_row = _build_collocation_matrix_1d(s1d, np.array([b], dtype=s1d.dtype))
            mass[-1, :] = colloc_row[0, :]

        mass_matrices.append(mass)
        quad_nodes_per_dir.append(q_nodes)
        quad_weights_per_dir.append(q_weights)

    # Build quadrature lattice and evaluate function.
    quad_lattice = PointsLattice(quad_nodes_per_dir)
    quad_grid_shape = tuple(a.shape[0] for a in quad_nodes_per_dir)
    components, out_dtype = _evaluate_func_on_lattice(func, quad_lattice, quad_grid_shape)

    # Assemble load vectors per direction and solve.
    ctrl_components = _l2_solve_components(
        func,
        space,
        components,
        out_dtype,
        mass_matrices,
        quad_nodes_per_dir,
        quad_weights_per_dir,
        bi_flags,
        tol,
    )

    ctrl = np.stack(ctrl_components, axis=-1)
    return Bspline(space, ctrl)


def _l2_solve_components(  # noqa: PLR0913
    func: Callable[..., npt.ArrayLike],
    space: BsplineSpace,
    components: list[npt.NDArray[np.floating[Any]]],
    out_dtype: np.dtype[np.float32] | np.dtype[np.float64],
    mass_matrices: list[npt.NDArray[np.float32 | np.float64]],
    quad_nodes_per_dir: list[npt.NDArray[np.float32 | np.float64]],
    quad_weights_per_dir: list[npt.NDArray[np.float32 | np.float64]],
    bi_flags: list[tuple[bool, bool]],
    tol: float | None,
) -> list[npt.NDArray[np.floating[Any]]]:
    """Assemble load vectors and solve the L2 system for each component.

    Args:
        func (Callable): Function being projected.
        space (BsplineSpace): Target B-spline space.
        components (list[npt.NDArray]): Per-component quadrature values.
        out_dtype: Output floating dtype.
        mass_matrices (list[npt.NDArray]): Per-direction mass matrices.
        quad_nodes_per_dir (list[npt.NDArray]): Per-direction quadrature nodes.
        quad_weights_per_dir (list[npt.NDArray]): Per-direction quadrature weights.
        bi_flags (list[tuple[bool, bool]]): Per-direction boundary interpolation flags.
        tol (float | None): SVD truncation tolerance.

    Returns:
        list[npt.NDArray]: Per-component coefficient arrays.
    """
    ctrl_components: list[npt.NDArray[np.floating[Any]]] = []

    for comp in components:
        load_arr = comp.astype(out_dtype).copy()
        load: npt.NDArray[np.floating[Any]] = load_arr  # type: ignore[assignment]
        for d, s1d in enumerate(space.spaces):
            load = _assemble_load_1d(
                s1d,
                load,
                quad_nodes_per_dir[d],
                quad_weights_per_dir[d],
                d,
            )

        # Apply boundary interpolation to load vector.
        _apply_boundary_load(func, space, quad_nodes_per_dir, bi_flags, load)

        # Solve mass system per direction.
        coeffs = _solve_kronecker(mass_matrices, load, tol)
        ctrl_components.append(coeffs)

    return ctrl_components


def _apply_boundary_load(
    func: Callable[..., npt.ArrayLike],
    space: BsplineSpace,
    quad_nodes_per_dir: list[npt.NDArray[np.float32 | np.float64]],
    bi_flags: list[tuple[bool, bool]],
    load: npt.NDArray[np.floating[Any]],
) -> None:
    """Apply boundary interpolation values to the load tensor in-place.

    Args:
        func (Callable): Function being projected.
        space (BsplineSpace): Target B-spline space.
        quad_nodes_per_dir (list[npt.NDArray]): Per-direction quadrature nodes.
        bi_flags (list[tuple[bool, bool]]): Per-direction boundary flags.
        load (npt.NDArray): Load tensor to modify in-place.
    """
    for d, s1d in enumerate(space.spaces):
        bi_left, bi_right = bi_flags[d]
        if bi_left and not s1d.periodic:
            a = s1d.domain[0]
            boundary_val = _evaluate_func_at_boundary(func, space, quad_nodes_per_dir, d, a)
            slices: list[int | slice] = [slice(None)] * load.ndim
            slices[d] = 0
            load[tuple(slices)] = boundary_val
        if bi_right and not s1d.periodic:
            b = s1d.domain[1]
            boundary_val = _evaluate_func_at_boundary(func, space, quad_nodes_per_dir, d, b)
            slices = [slice(None)] * load.ndim
            slices[d] = -1
            load[tuple(slices)] = boundary_val


def _assemble_mass_and_quad_1d(
    space: BsplineSpace1D,
    n_quad: int,
    quad_func: Callable[..., tuple[npt.NDArray[Any], npt.NDArray[Any]]],
) -> tuple[
    npt.NDArray[np.float32 | np.float64],
    npt.NDArray[np.float32 | np.float64],
    npt.NDArray[np.float32 | np.float64],
]:
    """Assemble the 1D mass matrix and collect global quadrature nodes/weights.

    Uses per-element quadrature mapped from ``[0, 1]`` to each knot span.

    Args:
        space (BsplineSpace1D): The 1D B-spline space.
        n_quad (int): Quadrature points per element.
        quad_func (Callable): Quadrature rule factory (returns nodes, weights
            on ``[0, 1]``).

    Returns:
        tuple: ``(mass, global_nodes, global_weights)`` where *mass* has shape
        ``(num_basis, num_basis)`` and *global_nodes* / *global_weights* have
        shape ``(num_intervals * n_quad,)``.
    """
    ref_nodes, ref_weights = quad_func(n_quad, dtype=space.dtype)
    unique_knots = space.get_unique_knots_and_multiplicity(in_domain=True)[0]
    n_intervals = len(unique_knots) - 1
    n_basis = space.num_basis

    global_nodes_list: list[npt.NDArray[np.float32 | np.float64]] = []
    global_weights_list: list[npt.NDArray[np.float32 | np.float64]] = []
    mass = np.zeros((n_basis, n_basis), dtype=space.dtype)

    for e in range(n_intervals):
        a_e = unique_knots[e]
        b_e = unique_knots[e + 1]
        h = b_e - a_e
        if h <= 0:
            continue

        # Map reference nodes to element.
        elem_nodes = a_e + h * ref_nodes
        elem_weights = h * ref_weights

        global_nodes_list.append(elem_nodes)
        global_weights_list.append(elem_weights)

        # Evaluate basis at element quadrature nodes.
        basis_vals, first_basis = space.tabulate_basis(elem_nodes)
        order = space.degree + 1

        # Assemble element contribution to mass matrix.
        for q in range(n_quad):
            fb = first_basis[q]
            w = elem_weights[q]
            bv = basis_vals[q, :order]
            # Outer product contribution.
            for i in range(order):
                gi = (fb + i) % n_basis if space.periodic else fb + i
                for j in range(order):
                    gj = (fb + j) % n_basis if space.periodic else fb + j
                    mass[gi, gj] += w * bv[i] * bv[j]

    global_nodes = np.concatenate(global_nodes_list)
    global_weights = np.concatenate(global_weights_list)
    return mass, global_nodes, global_weights


def _assemble_load_1d(
    space: BsplineSpace1D,
    func_values: npt.NDArray[np.floating[Any]],
    quad_nodes: npt.NDArray[np.float32 | np.float64],
    quad_weights: npt.NDArray[np.float32 | np.float64],
    axis: int,
) -> npt.NDArray[np.floating[Any]]:
    """Contract function values along one direction to form load vector entries.

    For direction ``axis``, computes:
    ``L_i = sum_q w_q * N_i(x_q) * f(x_q, ...)``

    The result has the same shape as *func_values* except that axis ``axis``
    changes from ``n_quad_nodes`` to ``n_basis``.

    Args:
        space (BsplineSpace1D): The 1D B-spline space.
        func_values (npt.NDArray): Function values with quadrature-sized axis.
        quad_nodes (npt.NDArray): Global quadrature nodes.
        quad_weights (npt.NDArray): Global quadrature weights.
        axis (int): The axis to contract.

    Returns:
        npt.NDArray: Load contribution with basis-sized axis.
    """
    n_basis = space.num_basis
    n_quad_total = quad_nodes.shape[0]
    order = space.degree + 1

    basis_vals, first_basis = space.tabulate_basis(quad_nodes)

    # Build the full weighted-basis matrix: W[q, i] = w_q * N_i(x_q)
    weighted_basis = np.zeros((n_quad_total, n_basis), dtype=quad_nodes.dtype)
    for q in range(n_quad_total):
        fb = first_basis[q]
        w = quad_weights[q]
        for i in range(order):
            gi = (fb + i) % n_basis if space.periodic else fb + i
            weighted_basis[q, gi] += w * basis_vals[q, i]

    # Contract: result[..., i, ...] = sum_q weighted_basis[q, i] * func_values[..., q, ...]
    result = np.tensordot(weighted_basis.T, func_values, axes=([1], [axis]))
    # tensordot puts the contracted axis at position 0; move it back.
    result = np.moveaxis(result, 0, axis)
    return result


def _evaluate_func_at_boundary(
    func: Callable[..., npt.ArrayLike],
    space: BsplineSpace,
    quad_nodes_per_dir: list[npt.NDArray[np.float32 | np.float64]],
    direction: int,
    boundary_value: float | np.float32 | np.float64,
) -> npt.NDArray[np.floating[Any]] | np.floating[Any]:
    """Evaluate the function at a boundary point (or hyperplane).

    For the boundary interpolation row in the Kronecker L2 projection, we
    need the function value at the boundary. For 1D this is a scalar; for nD
    it's a tensor over the other directions' basis nodes.

    After the load contraction has already been applied for all directions,
    the load tensor has shape ``(n_basis_0, ..., n_basis_{d-1})``. The
    boundary interpolation replaces one slice along direction ``d`` with the
    function value at the boundary. For the Kronecker solve, this value must
    be the function evaluated at the boundary in direction ``d`` and at the
    Greville nodes in all other directions, then contracted with those other
    directions' load operators.

    For simplicity and correctness in the 1D case, we evaluate at the single
    boundary point using a minimal lattice.

    Args:
        func (Callable): Function to evaluate.
        space (BsplineSpace): Target B-spline space.
        quad_nodes_per_dir (list[npt.NDArray]): Quadrature nodes per direction.
        direction (int): The direction of the boundary.
        boundary_value: The boundary coordinate value.

    Returns:
        Value(s) at the boundary for insertion into the load tensor.
    """
    dtype = quad_nodes_per_dir[direction].dtype

    if space.dim == 1:
        # 1D: evaluate function at the single boundary point.
        lattice = PointsLattice([np.array([boundary_value], dtype=dtype)])
        raw = np.asarray(func(lattice))
        if not np.issubdtype(raw.dtype, np.floating):
            raw = raw.astype(np.float64)
        return raw.ravel()[0] if raw.size == 1 else raw.ravel()

    # nD: create a lattice with a single point in the boundary direction
    # and Greville nodes in the other directions.
    greville_nodes = [greville_abscissae(s) for s in space.spaces]
    boundary_nodes = [
        np.array([boundary_value], dtype=dtype) if d == direction else greville_nodes[d]
        for d in range(space.dim)
    ]
    lattice = PointsLattice(boundary_nodes)
    grid_shape = tuple(a.shape[0] for a in boundary_nodes)

    raw = np.asarray(func(lattice))
    if not np.issubdtype(raw.dtype, np.floating):
        raw = raw.astype(np.float64)

    values = raw.reshape(grid_shape) if raw.ndim == 1 else raw.reshape(*grid_shape, -1)
    # Squeeze out the single-point boundary direction.
    return values.squeeze(axis=direction)


def _resolve_boundary_interpolation(
    boundary_interpolation: bool | Sequence[tuple[bool, bool]],
    space: BsplineSpace,
) -> list[tuple[bool, bool]]:
    """Normalize boundary interpolation flags to per-direction pairs.

    Args:
        boundary_interpolation: Flags as provided by the user.
        space (BsplineSpace): The target B-spline space.

    Returns:
        list[tuple[bool, bool]]: Per-direction ``(left, right)`` flags.
    """
    ndim = space.dim
    if isinstance(boundary_interpolation, bool):
        if boundary_interpolation:
            return [(not s.periodic, not s.periodic) for s in space.spaces]
        return [(False, False)] * ndim

    flags = list(boundary_interpolation)
    if len(flags) != ndim:
        raise ValueError(f"boundary_interpolation has length {len(flags)}, expected {ndim}")
    return flags
