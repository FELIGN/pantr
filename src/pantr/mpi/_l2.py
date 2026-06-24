"""Distributed L2 projection onto tensor-product B-spline spaces.

Provides :func:`l2_project_bspline_distributed`, the MPI-parallel counterpart of
:func:`~pantr.bspline.l2_project_bspline`.  L2 assembly is per-element, mapping directly
onto the cell partition: each rank evaluates ``func`` only on the quadrature points of
its *owned* cells and contracts them into a per-component load tensor, a single
``allreduce`` sums the global load across ranks, and the (replicated) Kronecker solve
recovers the global coefficients.  The per-direction mass matrices are
partition-independent and built identically on every rank, so only the load is
communicated.  The result is a :class:`~pantr.mpi.DistributedFunction` whose
:attr:`~pantr.mpi.DistributedFunction.local` reproduces the serial L2 projection exactly
over the rank's owned cells.
"""

from __future__ import annotations

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


def _assemble_owned_load(  # noqa: PLR0913
    space: BsplineSpace,
    components: list[npt.NDArray[np.floating[Any]]],
    owned_mask: npt.NDArray[np.bool_],
    out_dtype: np.dtype[np.float32] | np.dtype[np.float64],
    quad_nodes_per_dir: list[npt.NDArray[np.float32 | np.float64]],
    quad_weights_per_dir: list[npt.NDArray[np.float32 | np.float64]],
) -> list[npt.NDArray[np.floating[Any]]]:
    """Contract the owned-cell function values into a per-component load tensor.

    Each component's quadrature values are masked to this rank's owned cells (others set
    to zero), then contracted direction by direction with the weighted basis.  Because
    the contraction is linear in the function values and every quadrature point belongs
    to exactly one cell, summing these per-rank loads over all ranks reproduces the full
    serial L2 load exactly.

    Args:
        space (BsplineSpace): The target B-spline space.
        components (list[npt.NDArray[np.floating[Any]]]): Per-component quadrature values,
            each of shape ``quad_grid_shape``.
        owned_mask (npt.NDArray[np.bool_]): Owned-quadrature-point mask of shape
            ``quad_grid_shape``.
        out_dtype (np.dtype[np.float32] | np.dtype[np.float64]): Output floating dtype.
        quad_nodes_per_dir (list[npt.NDArray[np.float32 | np.float64]]): Per-direction
            global quadrature nodes.
        quad_weights_per_dir (list[npt.NDArray[np.float32 | np.float64]]): Per-direction
            global quadrature weights.

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
        load: npt.NDArray[np.floating[Any]] = np.where(owned_mask, comp, 0.0).astype(out_dtype)
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
    evaluates ``func`` only on the quadrature points of its *owned* cells and contracts
    them into a per-component load tensor; a single ``allreduce`` sums the global load
    across ranks; and the replicated Kronecker solve recovers the global coefficients.
    The returned :class:`~pantr.mpi.DistributedFunction` agrees with the serial L2
    projection pointwise over every owned cell.

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
        ``func`` MUST be rank-independent: for a given quadrature lattice it must return
        the same shape and dtype on every rank.  The reduction (``comm.allreduce``) is a
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
    rank = distributed_space.rank

    # Mass matrices, global quadrature nodes/weights, and resolved settings are
    # partition-independent: every rank builds them identically.
    mass_matrices, quad_nodes_per_dir, quad_weights_per_dir, bi_flags, n_quads = (
        _build_l2_mass_and_quad(global_space, n_quad, quadrature, boundary_interpolation)
    )

    # Evaluate func on the global quadrature lattice, restricted (by masking) to this
    # rank's owned cells, and contract into a per-component load tensor.
    quad_lattice = PointsLattice(quad_nodes_per_dir)
    quad_grid_shape = tuple(a.shape[0] for a in quad_nodes_per_dir)
    components, out_dtype = _evaluate_func_on_lattice(func, quad_lattice, quad_grid_shape)
    n_components = len(components)

    owned_mask = _owned_quad_cell_mask(
        global_space,
        distributed_space.partition.cell_owner,
        rank,
        n_quads,
        quad_grid_shape,
    )
    local_loads = _assemble_owned_load(
        global_space,
        components,
        owned_mask,
        out_dtype,
        quad_nodes_per_dir,
        quad_weights_per_dir,
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
