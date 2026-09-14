"""Distributed L2 projection onto tensor-product B-spline spaces.

Provides :func:`l2_project_bspline_distributed`, the MPI-parallel counterpart of
:func:`~pantr.bspline.l2_project_bspline`.  L2 *assembly* is per-element and maps
directly onto the cell partition.  The *evaluation* of ``func`` follows it where a
lattice can name the owned set: a rank whose owned cells form an axis-aligned box
evaluates ``func`` on that box's quadrature nodes only, and any other rank evaluates it on
the whole global quadrature lattice and masks the result to its owned cells.  Each rank
then contracts its values into a per-component load tensor, a single ``allreduce`` sums
the global load across ranks, and the (replicated) Kronecker solve recovers the global
coefficients.  The per-direction mass matrices are
partition-independent and built identically on every rank, so only the load is
communicated.  The result is a :class:`~pantr.mpi.DistributedFunction` whose
:attr:`~pantr.mpi.DistributedFunction.local` reproduces the serial L2 projection over the
rank's owned cells up to rounding.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from ..bspline import Bspline, BsplineSpace
from ..bspline._bspline_interpolate import (
    _apply_boundary_load,
    _assemble_load_1d,
    _build_l2_mass_and_quad,
    _evaluate_func_on_lattice,
    _solve_kronecker,
)
from ..quad import PointsLattice
from ._distributed_function import DistributedFunction
from ._distributed_space import DistributedSpace
from ._thread_policy import _ensure_default_thread_policy

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy.typing as npt


def _owned_quad_cell_mask(
    space: BsplineSpace,
    cell_owner: npt.NDArray[np.int32],
    rank: int,
    n_quads: tuple[int, ...],
    quad_grid_shape: tuple[int, ...],
) -> npt.NDArray[np.bool_]:
    """Build the boolean mask of quadrature points lying in this rank's owned cells.

    The global quadrature nodes are concatenated per element in each direction, so the
    ``n_quad`` consecutive nodes of direction-``d`` element ``e`` map to interval ``e``.
    A quadrature point's owning cell is the C-order flat id (over ``num_intervals``) of
    its per-direction element tuple; the point is owned iff ``cell_owner`` assigns that
    cell to ``rank``.

    Args:
        space (BsplineSpace): The target B-spline space.
        cell_owner (npt.NDArray[np.int32]): Owner rank of every global cell, C-order over
            ``num_intervals``.
        rank (int): This rank's id.
        n_quads (tuple[int, ...]): Quadrature points per element per direction.
        quad_grid_shape (tuple[int, ...]): Shape of the global quadrature grid
            (``num_intervals[d] * n_quads[d]`` per direction).

    Returns:
        npt.NDArray[np.bool_]: Boolean mask of shape ``quad_grid_shape``; ``True`` at
        quadrature points whose owning cell belongs to ``rank``.

    Note:
        Assumes a full partition: every cell is owned by exactly one rank (no inactive
        ``-1`` cells).  A future trimmed-grid caller with unowned cells would need to
        handle the ``-1`` owner before relying on this mask.
    """
    num_intervals = space.num_intervals
    elem_idx_per_dir = [
        np.repeat(np.arange(num_intervals[d], dtype=np.int64), n_quads[d]) for d in range(space.dim)
    ]
    mesh = np.meshgrid(*elem_idx_per_dir, indexing="ij")
    cell_flat = np.ravel_multi_index(tuple(mesh), num_intervals)
    return np.asarray(cell_owner[cell_flat] == rank).reshape(quad_grid_shape)


def _owned_cell_box(
    owned_cells: npt.NDArray[np.int64], num_intervals: tuple[int, ...]
) -> tuple[tuple[int, int], ...] | None:
    """Return the per-direction cell range of the owned cells if they form a box.

    The owned cells are a box iff their count equals the volume of their multi-index
    bounding box: every owned cell lies inside that box by construction, and the ids are
    distinct, so equal counts leave no box cell unowned.

    Args:
        owned_cells (npt.NDArray[np.int64]): Distinct flat ids (C-order over
            ``num_intervals``) of the cells this rank owns.
        num_intervals (tuple[int, ...]): Number of intervals per direction.

    Returns:
        tuple[tuple[int, int], ...] | None: One half-open ``(start, stop)`` interval-index
        range per direction when the owned cells are exactly that box; ``None`` when they
        are not a box, or when there are none.
    """
    if owned_cells.size == 0:
        return None
    multi = np.unravel_index(owned_cells, num_intervals)
    box = tuple((int(idx.min()), int(idx.max()) + 1) for idx in multi)
    volume = math.prod(stop - start for start, stop in box)
    return box if volume == owned_cells.size else None


def _evaluate_owned(  # noqa: PLR0913
    func: Callable[..., npt.ArrayLike],
    space: BsplineSpace,
    distributed_space: DistributedSpace,
    n_quads: tuple[int, ...],
    quad_nodes_per_dir: list[npt.NDArray[np.float32 | np.float64]],
    quad_weights_per_dir: list[npt.NDArray[np.float32 | np.float64]],
) -> tuple[
    list[npt.NDArray[np.floating[Any]]],
    np.dtype[np.float32] | np.dtype[np.float64],
    list[npt.NDArray[np.float32 | np.float64]],
    list[npt.NDArray[np.float32 | np.float64]],
]:
    """Evaluate ``func`` over this rank's owned cells, on the smallest lattice that names them.

    The global quadrature nodes are concatenated per element in each direction, so the
    nodes of interval range ``[start, stop)`` in direction ``d`` are the contiguous slice
    ``[start * n_quads[d], stop * n_quads[d])``.  When the owned cells form a box, ``func``
    is called on those slices only and every value it returns is owned.  Otherwise it is
    called on the whole global lattice and the values outside the owned cells are zeroed.

    A rank that owns no cells takes the whole-lattice path: ``func``'s return fixes the
    dtype and component count of this rank's ``allreduce`` contribution, so it must still
    be called, and no empty lattice exists to call it on.

    Args:
        func (Callable[..., npt.ArrayLike]): Function to project, ``func(lattice)``.
        space (BsplineSpace): The global B-spline space.
        distributed_space (DistributedSpace): The distributed space (owned cells,
            partition, rank).
        n_quads (tuple[int, ...]): Quadrature points per element per direction.
        quad_nodes_per_dir (list[npt.NDArray[np.float32 | np.float64]]): Per-direction
            global quadrature nodes.
        quad_weights_per_dir (list[npt.NDArray[np.float32 | np.float64]]): Per-direction
            global quadrature weights.

    Returns:
        tuple: ``(components, out_dtype, nodes_per_dir, weights_per_dir)`` -- the
        per-component values, zero outside the owned cells, each shaped like the evaluated
        lattice; the inferred floating dtype; and the per-direction nodes and weights of
        the evaluated lattice, which the load contraction must use.

    Raises:
        ValueError: If ``func`` returns an output with an invalid shape.
    """
    box = _owned_cell_box(distributed_space.owned_cells, space.num_intervals)
    if box is not None:
        ranges = [
            slice(start * nq, stop * nq) for (start, stop), nq in zip(box, n_quads, strict=True)
        ]
        nodes = [a[r] for a, r in zip(quad_nodes_per_dir, ranges, strict=True)]
        weights = [a[r] for a, r in zip(quad_weights_per_dir, ranges, strict=True)]
        shape = tuple(a.shape[0] for a in nodes)
        components, out_dtype = _evaluate_func_on_lattice(func, PointsLattice(nodes), shape)
        return components, out_dtype, nodes, weights

    shape = tuple(a.shape[0] for a in quad_nodes_per_dir)
    components, out_dtype = _evaluate_func_on_lattice(
        func, PointsLattice(quad_nodes_per_dir), shape
    )
    owned_mask = _owned_quad_cell_mask(
        space, distributed_space.partition.cell_owner, distributed_space.rank, n_quads, shape
    )
    masked = [np.where(owned_mask, comp, 0.0) for comp in components]
    return masked, out_dtype, quad_nodes_per_dir, quad_weights_per_dir


def _assemble_owned_load(
    space: BsplineSpace,
    components: list[npt.NDArray[np.floating[Any]]],
    out_dtype: np.dtype[np.float32] | np.dtype[np.float64],
    quad_nodes_per_dir: list[npt.NDArray[np.float32 | np.float64]],
    quad_weights_per_dir: list[npt.NDArray[np.float32 | np.float64]],
) -> list[npt.NDArray[np.floating[Any]]]:
    """Contract the owned-cell function values into a per-component load tensor.

    Each component's quadrature values, already zero outside this rank's owned cells, are
    contracted direction by direction with the weighted basis tabulated at the given
    nodes.  Because the contraction is linear in the function values and every quadrature
    point belongs to exactly one cell, summing these per-rank loads over all ranks
    reproduces the full serial L2 load.

    Args:
        space (BsplineSpace): The target B-spline space.
        components (list[npt.NDArray[np.floating[Any]]]): Per-component quadrature values,
            each shaped like the lattice of ``quad_nodes_per_dir``, zero outside the owned
            cells.
        out_dtype (np.dtype[np.float32] | np.dtype[np.float64]): Output floating dtype.
        quad_nodes_per_dir (list[npt.NDArray[np.float32 | np.float64]]): Per-direction
            quadrature nodes the values were taken at (global, or a contiguous slice).
        quad_weights_per_dir (list[npt.NDArray[np.float32 | np.float64]]): The matching
            per-direction quadrature weights.

    Returns:
        list[npt.NDArray[np.floating[Any]]]: Per-component load tensors of shape
        ``num_basis``.

    Note:
        This reproduces the full serial *interior* L2 load only.  Boundary-interpolation
        rows are not applied here: they are imposed later (via ``_apply_boundary_load``)
        on the reduced (global) load, after the ``allreduce``.
    """
    loads: list[npt.NDArray[np.floating[Any]]] = []
    for comp in components:
        load: npt.NDArray[np.floating[Any]] = np.asarray(comp, dtype=out_dtype)
        for d, s1d in enumerate(space.spaces):
            load = _assemble_load_1d(s1d, load, quad_nodes_per_dir[d], quad_weights_per_dir[d], d)
        loads.append(load)
    return loads


def l2_project_bspline_distributed(  # noqa: PLR0913
    func: Callable[..., npt.ArrayLike],
    distributed_space: DistributedSpace,
    *,
    n_quad: int | Sequence[int] | None = None,
    quadrature: Literal["gauss-legendre", "gauss-lobatto"] = "gauss-legendre",
    boundary_interpolation: bool | Sequence[tuple[bool, bool]] = False,
    tol: float | None = None,
) -> DistributedFunction:
    """L2-project a callable onto a distributed tensor-product B-spline space.

    The MPI-parallel counterpart of :func:`~pantr.bspline.l2_project_bspline`.  L2
    assembly is per-element and maps directly onto the cell partition: each rank
    contracts the quadrature values over its *owned* cells into a per-component load
    tensor; a single ``allreduce`` sums the global load across ranks; and the replicated
    Kronecker solve recovers the global coefficients.  The returned
    :class:`~pantr.mpi.DistributedFunction` agrees with the serial L2 projection
    pointwise over every owned cell.

    **Where ``func`` is evaluated depends on the shape of each rank's owned cells.**
    ``func`` receives a :class:`~pantr.quad.PointsLattice`, a tensor product of
    per-direction coordinates, so only an owned set that *is* an axis-aligned box of cells
    can be named by one.  A rank whose owned cells form such a box calls ``func`` on that
    box's quadrature nodes only.  Any other rank -- including one that owns no cells --
    calls it on the whole global lattice and discards the values outside its owned
    cells.  ``func`` is called exactly once per rank either way, and either way the
    rank's contribution to the load is the same up to rounding.

    **The default partitioner gives every rank a box**: ``partition_grid``'s ``block``
    backend, which ``create_distributed_space`` selects for a tensor-product grid with no
    cell weights or activity mask when the rank count factors onto the axes, splits the
    grid into axis-aligned boxes, so the points evaluated across the run total the serial
    count.  A partition from the ``rcb`` backend or from ``method="graph"`` need not be
    box-shaped; each of its non-box ranks evaluates the whole lattice, so for an expensive
    ``func`` the total can grow with the rank count.  Naming a non-box owned set would
    mean handing ``func`` a flat point array, which is a different signature --
    :func:`~pantr.mpi.quasi_interpolate_bspline_distributed` has that one.  No warning is
    issued when the whole-lattice path is taken.

    The per-direction mass matrices are partition-independent and built identically on
    every rank (cheap ``n_dofs_i x n_dofs_i`` systems), so only the load is communicated.
    ``boundary_interpolation`` rows are handled after the reduce: each boundary trace is
    partition-independent, so every rank recomputes the same boundary row (the boundary
    trace spans the whole face, so it is not attributable to a single owning cell -- every
    rank recomputes it identically from the global lattice).  Construction requires one
    MPI collective (``comm.allreduce``) after the local assembly.

    Args:
        func (Callable[..., npt.ArrayLike]): Function to project.  Called as
            ``func(lattice)`` where ``lattice`` is a :class:`~pantr.quad.PointsLattice`
            of quadrature points (the serial convention); must return an array of shape
            ``(n_total,)`` for scalar or ``(n_total, rank)`` for vector-valued functions.
        distributed_space (DistributedSpace): The distributed space to project onto.  Its
            ``global_space`` must be a :class:`~pantr.bspline.BsplineSpace`.
        n_quad (int | Sequence[int] | None): Quadrature points per element per direction.
            Defaults to ``degree + 1``.
        quadrature (Literal["gauss-legendre", "gauss-lobatto"]): Quadrature rule type.
            Defaults to ``"gauss-legendre"``.
        boundary_interpolation (bool | Sequence[tuple[bool, bool]]): Replace boundary rows
            with interpolation conditions.  ``False`` (default) is a pure L2 projection,
            ``True`` interpolates at all non-periodic boundaries, and a sequence of
            ``(left, right)`` pairs sets per-direction flags.
        tol (float | None): SVD truncation tolerance for the per-direction solves.  If
            ``None``, defaults to ``100 * machine_epsilon``.

    Returns:
        DistributedFunction: A distributed function whose
        :attr:`~pantr.mpi.DistributedFunction.local` L2-projects ``func`` over this
        rank's owned cells, and whose
        :attr:`~pantr.mpi.DistributedFunction.global_function` holds the full assembled
        global coefficient field (identical on every rank after the ``allreduce``).

    Raises:
        TypeError: If ``distributed_space.global_space`` is not a
            :class:`~pantr.bspline.BsplineSpace`.
        ValueError: If ``n_quad`` or ``boundary_interpolation`` is inconsistent with the
            global space, if ``func`` returns an output with an invalid shape, or if the
            reduced load does not match the expected stacked shape ``(*num_basis,
            n_components)`` (a symptom of ``func`` returning inconsistent shapes across
            ranks).

    Note:
        ``func`` MUST be rank-independent: its dtype and component count must not depend
        on the rank or on the size of the lattice it is handed, since ranks with a box
        receive only their box while other ranks receive the whole lattice.  A component
        count that varied with the lattice size could broadcast in the reduction rather
        than raise.  The reduction (``comm.allreduce``) is a
        collective that every rank must reach with a matching contribution; if ``func``
        raised on a subset of ranks (e.g. a shape error seen by some ranks only) those
        ranks would abort before the collective and deadlock the rest, and mixed dtypes
        would corrupt the reduction.

    Note:
        The output dtype is inferred from the return value of ``func`` (as in the serial
        :func:`~pantr.bspline.l2_project_bspline`).  Unlike the serial path -- where
        :class:`~pantr.bspline.Bspline` raises on a control-point/space dtype mismatch --
        the distributed path is more lenient: it coerces the assembled global control
        points to ``global_space.dtype`` rather than raising.

    Example:
        Needs a live MPI communicator and several ranks, so it is not run as a doctest.

        >>> from mpi4py import MPI  # doctest: +SKIP
        >>> import numpy as np  # doctest: +SKIP
        >>> from pantr.bspline import create_uniform_space  # doctest: +SKIP
        >>> from pantr.mpi import create_distributed_space  # doctest: +SKIP
        >>> from pantr.mpi import l2_project_bspline_distributed  # doctest: +SKIP
        >>> space = create_uniform_space([2, 2], [8, 8])  # doctest: +SKIP
        >>> ds = create_distributed_space(space, MPI.COMM_WORLD)  # doctest: +SKIP
        >>> dfn = l2_project_bspline_distributed(  # doctest: +SKIP
        ...     lambda lat: np.sin(lat.pts_per_dir[0]), ds
        ... )
        >>> local = dfn.local  # rank-local Bspline on the windowed space  # doctest: +SKIP
    """
    _ensure_default_thread_policy()

    global_space = distributed_space.global_space
    if not isinstance(global_space, BsplineSpace):
        raise TypeError(
            f"distributed_space.global_space must be a BsplineSpace; "
            f"got {type(global_space).__name__!r}."
        )

    comm = distributed_space.comm

    # Mass matrices, global quadrature nodes/weights, and resolved settings are
    # partition-independent: every rank builds them identically.
    mass_matrices, quad_nodes_per_dir, quad_weights_per_dir, bi_flags, n_quads = (
        _build_l2_mass_and_quad(global_space, n_quad, quadrature, boundary_interpolation)
    )

    # Evaluate func over this rank's owned cells -- on its box's nodes where the owned set
    # is a box, else on the whole lattice with the rest zeroed -- and contract the values
    # on the lattice they were taken at.
    components, out_dtype, eval_nodes, eval_weights = _evaluate_owned(
        func, global_space, distributed_space, n_quads, quad_nodes_per_dir, quad_weights_per_dir
    )
    n_components = len(components)
    local_loads = _assemble_owned_load(
        global_space, components, out_dtype, eval_nodes, eval_weights
    )

    # Stack components into a single tensor for one allreduce: (*num_basis, n_components).
    num_basis = tuple(global_space.num_basis)
    local_load = np.stack(local_loads, axis=-1).astype(out_dtype, copy=False)
    global_load = cast(
        "npt.NDArray[np.floating[Any]]",
        np.asarray(comm.allreduce(local_load), dtype=out_dtype),
    )
    expected_shape = (*num_basis, n_components)
    if global_load.shape != expected_shape:
        raise ValueError(
            f"Reduced load has shape {global_load.shape}, expected {expected_shape}; "
            f"this indicates 'func' returned an inconsistent shape across ranks."
        )

    # Apply boundary-interpolation rows on the reduced (global) load, replicated on every
    # rank.  Each boundary trace is partition-independent, matching the serial result.
    ctrl_components: list[npt.NDArray[np.floating[Any]]] = []
    for comp_idx in range(n_components):
        load: npt.NDArray[np.floating[Any]] = global_load[..., comp_idx].copy()
        _apply_boundary_load(
            func,
            global_space,
            quad_nodes_per_dir,
            quad_weights_per_dir,
            bi_flags,
            load,
            comp_idx,
            n_components,
        )
        coeffs = _solve_kronecker(mass_matrices, load, tol)
        ctrl_components.append(coeffs)

    ctrl = np.stack(ctrl_components, axis=-1).astype(global_space.dtype, copy=False)
    global_cp = ctrl.reshape(*num_basis, n_components)

    global_bspline = Bspline(global_space, global_cp)
    return DistributedFunction(global_bspline, distributed_space)


__all__ = ["l2_project_bspline_distributed"]
