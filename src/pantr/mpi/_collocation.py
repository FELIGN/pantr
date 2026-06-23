"""Distributed collocation interpolation / fitting onto tensor-product B-spline spaces.

Provides :func:`interpolate_bspline_distributed` and :func:`fit_bspline_distributed`, the
MPI-parallel counterparts of :func:`~pantr.bspline.interpolate_bspline` and
:func:`~pantr.bspline.fit_bspline`.  The collocation solve itself is a cheap sequence of
1D SVD solves on small ``n_dofs_i x n_dofs_i`` systems; the bottleneck that motivates
parallelism is evaluating ``func`` on the tensor-product Greville-node grid (whose size is
the global DOF count).  Each rank therefore evaluates ``func`` on its block-assigned chunk
of the flattened grid, a single ``allgather`` assembles the full value field, and the
Kronecker solve runs **replicated** on every rank.  The result is a
:class:`~pantr.mpi.DistributedFunction` whose global coefficient field equals the serial
:func:`~pantr.bspline.interpolate_bspline` result bit-for-bit (and whose
:attr:`~pantr.mpi.DistributedFunction.local` reproduces it over the rank's owned cells).

Both entry points operate on tensor-product (lattice / Greville) nodes -- the case the
distributed algorithm parallelizes.  Scattered-node inputs fall back to a fully replicated
evaluation on every rank (no parallel speedup, but a correct result); this is documented
per function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._interpolation_utils import split_components
from ..bspline import Bspline, BsplineSpace, fit_bspline
from ..bspline._bspline_interpolate import (
    _apply_boundary_deriv_rhs,
    _build_collocation_matrices,
    _is_scattered_nodes,
    _resolve_nodes,
    _solve_kronecker,
)
from ..quad import PointsLattice
from ._distributed_function import DistributedFunction
from ._distributed_space import DistributedSpace
from ._thread_policy import _ensure_default_thread_policy

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy.typing as npt


def _block_bounds(n_total: int, n_parts: int, rank: int) -> tuple[int, int]:
    """Return this rank's half-open block ``[start, stop)`` of a contiguous range.

    Splits ``range(n_total)`` into ``n_parts`` contiguous blocks as evenly as possible:
    the first ``n_total % n_parts`` blocks get one extra element.  An empty range (or a
    rank beyond the populated blocks) yields a degenerate ``[start, start)`` slice.

    Args:
        n_total (int): Length of the range being split.
        n_parts (int): Number of blocks (ranks).
        rank (int): The rank whose block is requested; assumed in ``[0, n_parts)``.

    Returns:
        tuple[int, int]: The ``(start, stop)`` half-open bounds of this rank's block.

    Note:
        No input validation is performed.
    """
    base, extra = divmod(n_total, n_parts)
    start = rank * base + min(rank, extra)
    stop = start + base + (1 if rank < extra else 0)
    return start, stop


def _evaluate_block(
    func: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    all_points: npt.NDArray[np.float64],
    start: int,
    stop: int,
) -> npt.NDArray[np.float64]:
    """Evaluate ``func`` on a contiguous block of the flattened grid points.

    Args:
        func (Callable): Function called on a flat ``(M, dim)`` point array; must return
            ``(M,)`` (scalar) or ``(M, rank)`` (vector-valued).
        all_points (npt.NDArray[np.float64]): The full ``(n_total, dim)`` flattened grid.
        start (int): Inclusive start index of this rank's block.
        stop (int): Exclusive stop index of this rank's block.

    Returns:
        npt.NDArray[np.float64]: This rank's values, shape ``(stop - start,)`` (scalar) or
        ``(stop - start, rank)`` (vector); always ``float64`` and C-contiguous.

    Note:
        No input validation is performed; ``func``'s return shape is checked downstream by
        the replicated Kronecker solve.
    """
    block = np.ascontiguousarray(all_points[start:stop])
    values = np.asarray(func(block), dtype=np.float64)
    return np.ascontiguousarray(values)


def _assemble_full_values(
    blocks: list[npt.NDArray[np.float64]],
    n_total: int,
) -> npt.NDArray[np.float64]:
    """Concatenate per-rank value blocks into the full flattened value field.

    Args:
        blocks (list[npt.NDArray[np.float64]]): Per-rank value blocks in rank order, each
            shaped ``(block_len,)`` (scalar) or ``(block_len, rank)`` (vector); empty
            blocks may be ``(0,)`` or ``(0, rank)``.
        n_total (int): Expected total number of grid points (sum of block lengths).

    Returns:
        npt.NDArray[np.float64]: The full value field, shape ``(n_total,)`` (scalar) or
        ``(n_total, rank)`` (vector).

    Note:
        No input validation is performed.  The rank dimension is inferred from the first
        non-empty 2D block; if every block is empty the field is treated as scalar.
    """
    scalar = True
    rank_dim = 1
    for b in blocks:
        if b.shape[0] == 0:
            continue
        scalar = b.ndim == 1
        rank_dim = 1 if scalar else int(b.shape[1])
        break

    shape: tuple[int, ...] = (n_total,) if scalar else (n_total, rank_dim)
    full = np.empty(shape, dtype=np.float64)
    offset = 0
    for b in blocks:
        m = b.shape[0]
        if m == 0:
            continue
        full[offset : offset + m] = b if scalar else b.reshape(m, rank_dim)
        offset += m
    return full


def _solve_replicated(
    space: BsplineSpace,
    node_arrays: list[npt.NDArray[np.float32 | np.float64]],
    full_values: npt.NDArray[np.float64],
    boundary_derivatives: Sequence[tuple[int, ...] | None] | None,
    tol: float | None,
) -> Bspline:
    """Run the replicated tensor-product collocation solve from a full value field.

    Mirrors the serial :func:`~pantr.bspline.interpolate_bspline` solve: build the
    per-direction collocation matrices, reshape the flattened values to the grid, apply
    any boundary-derivative right-hand-side zeros, and solve each component via the
    Kronecker structure.  Every rank runs this identically on the gathered value field.

    Args:
        space (BsplineSpace): The global target space.
        node_arrays (list[npt.NDArray]): Per-direction node arrays (the Greville or custom
            tensor-product nodes).
        full_values (npt.NDArray[np.float64]): The full flattened value field, shape
            ``(n_total,)`` (scalar) or ``(n_total, rank)`` (vector), C-ordered to match the
            grid (last grid index varies fastest).
        boundary_derivatives (Sequence[tuple[int, ...] | None] | None): Per-direction
            ``(n_left, n_right)`` boundary-derivative constraints, or ``None``.
        tol (float | None): SVD truncation tolerance forwarded to the 1D solves.

    Returns:
        Bspline: The fitted global B-spline (control points cast to ``space.dtype``).

    Note:
        No input validation is performed.  ``full_values`` must already be shaped
        consistently with ``node_arrays``.
    """
    grid_shape = tuple(a.shape[0] for a in node_arrays)
    if full_values.ndim == 1:
        gridded: npt.NDArray[np.float64] = full_values.reshape(grid_shape)
    else:
        gridded = full_values.reshape(*grid_shape, full_values.shape[1])

    components = split_components(gridded, grid_shape)
    if boundary_derivatives is not None:
        components = _apply_boundary_deriv_rhs(space, components, boundary_derivatives)

    matrices = _build_collocation_matrices(space, node_arrays, boundary_derivatives)
    out_dtype = space.dtype
    ctrl_components = [
        _solve_kronecker(matrices, comp.astype(out_dtype), tol) for comp in components
    ]
    ctrl = np.stack(ctrl_components, axis=-1)
    return Bspline(space, ctrl)


def interpolate_bspline_distributed(
    func: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    distributed_space: DistributedSpace,
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
) -> DistributedFunction:
    """Interpolate a callable onto a distributed tensor-product B-spline space.

    The MPI-parallel counterpart of :func:`~pantr.bspline.interpolate_bspline`.  The
    collocation solve is a cheap sequence of small 1D SVD solves; the cost that motivates
    parallelism is evaluating ``func`` on the tensor-product node grid (whose size is the
    global DOF count).  Each rank evaluates ``func`` on its block-assigned chunk of the
    flattened grid, a single ``allgather`` assembles the full value field, and the
    Kronecker solve runs **replicated** on every rank.  The returned
    :class:`~pantr.mpi.DistributedFunction` carries the global coefficient field (identical
    on every rank, equal to the serial result) and its
    :attr:`~pantr.mpi.DistributedFunction.local` reproduces the serial interpolant over the
    rank's owned cells.

    Unlike serial :func:`~pantr.bspline.interpolate_bspline`, whose ``func`` receives a
    :class:`~pantr.quad.PointsLattice`, here ``func`` receives a **flat** ``(M, dim)`` point
    array -- a contiguous chunk of the grid is not itself a tensor product, so a flat array
    is the natural distributed convention (and matches the distributed quasi-interpolants).

    Construction requires one MPI collective (``comm.allgather``) after each rank's local
    evaluation.

    Args:
        func (Callable): Function to interpolate.  Called on a flat ``(M, dim)`` point
            array; must return ``(M,)`` (scalar) or ``(M, rank)`` (vector-valued).
        distributed_space (DistributedSpace): The distributed space to interpolate onto.
            Its ``global_space`` must be a :class:`~pantr.bspline.BsplineSpace`.
        nodes: Interpolation node selection, identical to
            :func:`~pantr.bspline.interpolate_bspline`.

            - ``None`` or ``"greville"`` (default): Greville abscissae.
            - A :class:`~pantr.quad.PointsLattice`: custom tensor-product grid.
            - A 1D ``ndarray``: custom nodes for a 1D space.
            - A sequence of 1D ``ndarray`` values: per-direction custom nodes.
        boundary_derivatives (Sequence[tuple[int, ...] | None] | None): Per-direction
            ``(n_left, n_right)`` boundary-derivative constraints (set to zero), or
            ``None``.  Defaults to ``None``.
        tol (float | None): SVD truncation tolerance for the collocation solve.  If
            ``None``, defaults to ``100 * machine_epsilon``.  Defaults to ``None``.

    Returns:
        DistributedFunction: A distributed function whose
        :attr:`~pantr.mpi.DistributedFunction.global_function` holds the full assembled
        global coefficient field (identical on every rank), and whose
        :attr:`~pantr.mpi.DistributedFunction.local` interpolates ``func`` over this rank's
        owned cells.

    Raises:
        TypeError: If ``distributed_space.global_space`` is not a
            :class:`~pantr.bspline.BsplineSpace`.
        ValueError: If ``nodes`` is inconsistent with the space, or ``func`` returns an
            output with an invalid shape.

    Note:
        Tensor-product (lattice / Greville) nodes are the parallelized path.  All internal
        computation uses ``float64``; the global control points are cast to
        ``global_space.dtype`` before assembly, consistent with serial
        :func:`~pantr.bspline.interpolate_bspline`.

    Example:
        >>> from mpi4py import MPI  # doctest: +SKIP
        >>> import numpy as np  # doctest: +SKIP
        >>> from pantr.bspline import create_uniform_space  # doctest: +SKIP
        >>> from pantr.mpi import create_distributed_space  # doctest: +SKIP
        >>> from pantr.mpi import interpolate_bspline_distributed  # doctest: +SKIP
        >>> space = create_uniform_space([2, 2], [8, 8])  # doctest: +SKIP
        >>> ds = create_distributed_space(space, MPI.COMM_WORLD)  # doctest: +SKIP
        >>> dfn = interpolate_bspline_distributed(  # doctest: +SKIP
        ...     lambda p: np.sin(p[:, 0]) * np.cos(p[:, 1]), ds
        ... )
    """
    _ensure_default_thread_policy()

    global_space = distributed_space.global_space
    if not isinstance(global_space, BsplineSpace):
        raise TypeError(
            f"distributed_space.global_space must be a BsplineSpace; "
            f"got {type(global_space).__name__!r}."
        )

    node_arrays = _resolve_nodes(global_space, nodes)
    lattice = PointsLattice(node_arrays)

    comm = distributed_space.comm
    n_parts = int(comm.size)
    rank = int(comm.rank)

    all_points = np.ascontiguousarray(lattice.get_all_points(order="C"), dtype=np.float64)
    n_total = all_points.shape[0]
    start, stop = _block_bounds(n_total, n_parts, rank)
    local_values = _evaluate_block(func, all_points, start, stop)

    blocks: list[npt.NDArray[np.float64]] = list(comm.allgather(local_values))
    full_values = _assemble_full_values(blocks, n_total)

    global_bspline = _solve_replicated(
        global_space, node_arrays, full_values, boundary_derivatives, tol
    )
    return DistributedFunction(global_bspline, distributed_space)


def fit_bspline_distributed(
    values: npt.ArrayLike,
    nodes: (
        PointsLattice | npt.NDArray[np.floating[Any]] | Sequence[npt.NDArray[np.floating[Any]]]
    ),
    distributed_space: DistributedSpace,
    *,
    values_distributed: bool = False,
    tol: float | None = None,
) -> DistributedFunction:
    """Construct a distributed B-spline from pre-evaluated sample values at known nodes.

    The MPI-parallel counterpart of :func:`~pantr.bspline.fit_bspline`.  The collocation
    solve is cheap (small 1D SVD solves); for tensor-product nodes the value field is the
    only large object, so this function supports values that **arrive already distributed**
    (one contiguous block of the flattened grid per rank), gathered with a single
    ``allgather`` before the replicated Kronecker solve.

    Two value layouts are supported via ``values_distributed``:

    - ``values_distributed=False`` (default, *replicated*): every rank passes the full
      value field (shape ``(*n_pts_per_dir)`` or ``(*n_pts_per_dir, rank)``), exactly as
      serial :func:`~pantr.bspline.fit_bspline`.  No ``allgather`` is needed; each rank
      solves locally.
    - ``values_distributed=True`` (*pre-distributed*): each rank passes only its
      block-assigned chunk of the **C-flattened** grid values, shape ``(block_len,)`` or
      ``(block_len, rank)``.  The blocks are concatenated (one ``allgather``) into the full
      field before the solve.  The per-rank split must match
      :func:`interpolate_bspline_distributed` (contiguous blocks of the C-order flattened
      grid, the first ``n_total % n_parts`` blocks one element longer).

    Args:
        values (npt.ArrayLike): Sample values.  Full tensor-product field when
            ``values_distributed`` is ``False``; this rank's flattened block when ``True``.
        nodes: Tensor-product interpolation nodes (identical on every rank).

            - A :class:`~pantr.quad.PointsLattice`: tensor-product grid.
            - A 1D ``ndarray``: 1D tensor-product (single direction).
            - A sequence of 1D ``ndarray`` values: N-D tensor-product.
        distributed_space (DistributedSpace): The distributed space to fit onto.  Its
            ``global_space`` must be a :class:`~pantr.bspline.BsplineSpace`.
        values_distributed (bool): Whether ``values`` is this rank's flattened block
            (``True``) or the full replicated field (``False``).  Defaults to ``False``.
        tol (float | None): SVD truncation tolerance.  If ``None``, defaults to
            ``100 * machine_epsilon``.  Defaults to ``None``.

    Returns:
        DistributedFunction: A distributed function whose
        :attr:`~pantr.mpi.DistributedFunction.global_function` holds the full assembled
        global coefficient field (identical on every rank), and whose
        :attr:`~pantr.mpi.DistributedFunction.local` reproduces the fit over the rank's
        owned cells.

    Raises:
        TypeError: If ``distributed_space.global_space`` is not a
            :class:`~pantr.bspline.BsplineSpace`.
        ValueError: If ``nodes`` are scattered (a 2D ``ndarray``) while
            ``values_distributed`` is ``True`` (scattered points have no tensor-product
            block layout), or if ``nodes`` / ``values`` are inconsistent with the space.

    Note:
        Scattered nodes (a 2D ``ndarray`` of shape ``(n_pts, dim)``) are only supported in
        the replicated path (``values_distributed=False``); they fall back to the serial
        :func:`~pantr.bspline.fit_bspline` solve run identically on every rank (correct,
        but with no parallel speedup).  The output dtype follows
        ``global_space.dtype``.

    Example:
        >>> from mpi4py import MPI  # doctest: +SKIP
        >>> from pantr.bspline import create_greville_lattice  # doctest: +SKIP
        >>> from pantr.bspline import create_uniform_space  # doctest: +SKIP
        >>> from pantr.mpi import create_distributed_space  # doctest: +SKIP
        >>> from pantr.mpi import fit_bspline_distributed  # doctest: +SKIP
        >>> space = create_uniform_space([2, 2], [8, 8])  # doctest: +SKIP
        >>> ds = create_distributed_space(space, MPI.COMM_WORLD)  # doctest: +SKIP
        >>> lat = create_greville_lattice(space)  # doctest: +SKIP
        >>> vals = (lat.get_all_points()[:, 0] ** 2)  # full field on every rank  # doctest: +SKIP
        >>> dfn = fit_bspline_distributed(vals, lat, ds)  # doctest: +SKIP
    """
    _ensure_default_thread_policy()

    global_space = distributed_space.global_space
    if not isinstance(global_space, BsplineSpace):
        raise TypeError(
            f"distributed_space.global_space must be a BsplineSpace; "
            f"got {type(global_space).__name__!r}."
        )

    if not values_distributed:
        # Replicated path: every rank holds the full field; delegate to the serial solve.
        global_bspline = fit_bspline(values, nodes, global_space, tol=tol)
        return DistributedFunction(global_bspline, distributed_space)

    if _is_scattered_nodes(nodes):
        raise ValueError(
            "values_distributed=True is only supported for tensor-product nodes; "
            "scattered (2D ndarray) nodes have no block layout. Pass the full value "
            "field with values_distributed=False instead."
        )

    node_arrays = _resolve_nodes(global_space, nodes)
    n_total = int(np.prod([a.shape[0] for a in node_arrays]))

    local_block = np.asarray(values, dtype=np.float64)
    blocks: list[npt.NDArray[np.float64]] = list(
        distributed_space.comm.allgather(np.ascontiguousarray(local_block))
    )
    full_values = _assemble_full_values(blocks, n_total)

    global_bspline = _solve_replicated(global_space, node_arrays, full_values, None, tol)
    return DistributedFunction(global_bspline, distributed_space)


__all__ = ["fit_bspline_distributed", "interpolate_bspline_distributed"]
