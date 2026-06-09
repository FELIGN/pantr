"""Serial windowing helpers for distributing a tensor-product B-spline space.

These functions compute, without any MPI, pieces a distributed local space is built
from:

- :func:`compute_halo`: the function-support closure of a set of owned cells -- the
  extra cells a rank must see so the B-spline functions touching its owned cells are
  fully represented.
- :func:`dof_owner`: the owner rank of every global DOF, by the
  lex-first-active-cell-in-support rule.

Both operate on the knot-span grid of a :class:`~pantr.bspline.BsplineSpace`: cells
are flat-indexed in C-order over ``num_intervals`` and DOFs in C-order over
``num_basis``, matching :func:`pantr.grid.tensor_product_grid` and
:class:`~pantr.bspline.SpanwiseElementExtraction`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ._thb_spline_space import _func_support_1d

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..grid import Partition
    from ._bspline_space_nd import BsplineSpace


def _reject_periodic(space: BsplineSpace) -> None:
    """Raise if any direction of ``space`` is periodic.

    Args:
        space (BsplineSpace): Tensor-product B-spline space.

    Raises:
        ValueError: If any 1D direction is periodic.
    """
    if any(sp.periodic for sp in space.spaces):
        raise ValueError("periodic B-spline spaces are not supported.")


def compute_halo(space: BsplineSpace, owned_cells: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Return the support-closure halo of ``owned_cells``.

    The halo is the set of knot-span cells, **excluding** the owned cells, covered by
    the support of any B-spline function non-zero on an owned cell. A rank owning
    ``owned_cells`` needs exactly these extra cells so every function touching its
    owned cells is fully represented over the union of the owned and halo cells. For
    open/uniform knots this is the ``degree``-wide halo; general or repeated knots are
    handled exactly via the per-direction support.

    Args:
        space (BsplineSpace): Tensor-product B-spline space (non-periodic). Its
            knot-span grid has ``num_total_intervals`` cells.
        owned_cells (npt.ArrayLike): Flat cell ids (C-order over ``num_intervals``)
            owned by the rank. Duplicates are ignored.

    Returns:
        npt.NDArray[np.int64]: Sorted, read-only flat ids of the halo cells -- those
        in the support closure but not in ``owned_cells``.

    Raises:
        ValueError: If any axis is periodic.
        TypeError: If ``owned_cells`` is not integer-valued.
        IndexError: If any owned cell id is out of range ``[0, num_total_intervals)``.
    """
    _reject_periodic(space)
    num_intervals = space.num_intervals
    n_cells = space.num_total_intervals
    owned = np.asarray(owned_cells).ravel()
    if owned.size == 0:
        result = np.empty(0, dtype=np.int64)
        result.flags.writeable = False
        return result
    if not np.issubdtype(owned.dtype, np.integer):
        raise TypeError(
            f"compute_halo: owned_cells must be integer-valued; got dtype {owned.dtype}."
        )
    owned = owned.astype(np.int64, copy=False)
    if int(owned.min()) < 0 or int(owned.max()) >= n_cells:
        raise IndexError(f"owned cell id out of range [0, {n_cells}).")

    # Per-axis interval -> inclusive support-cell range: functions non-zero on
    # interval c are [fb[c], fb[c] + degree]; their support is [fc[fb[c]], lc[fb[c]+p]].
    lo_axes: list[npt.NDArray[np.int64]] = []
    hi_axes: list[npt.NDArray[np.int64]] = []
    for sp in space.spaces:
        fb, fc, lc = _func_support_1d(sp)
        lo_axes.append(fc[fb].astype(np.int64))
        hi_axes.append(lc[fb + sp.degree].astype(np.int64))

    mask = np.zeros(num_intervals, dtype=np.bool_)
    owned_multi = np.unravel_index(owned, num_intervals)
    for i in range(owned.size):
        window = tuple(
            slice(int(lo_axes[d][owned_multi[d][i]]), int(hi_axes[d][owned_multi[d][i]]) + 1)
            for d in range(space.dim)
        )
        mask[window] = True
    halo = np.setdiff1d(np.flatnonzero(mask.ravel()), owned).astype(np.int64, copy=False)
    halo.flags.writeable = False
    return halo


def dof_owner(space: BsplineSpace, partition: Partition) -> npt.NDArray[np.int32]:
    """Return the owner rank of every global DOF (lex-first-active-cell rule).

    Each global B-spline DOF is owned by the rank that owns the active cell with the
    smallest flat id in the DOF's support. A DOF whose support contains no active cell
    (``cell_owner == -1`` throughout) is a dead DOF, assigned ``-1``.

    Args:
        space (BsplineSpace): Tensor-product B-spline space (non-periodic). DOFs are
            flat-indexed in C-order over ``num_basis``.
        partition (Partition): Owner of every knot-span cell; ``cell_owner`` must have
            length ``space.num_total_intervals``.

    Returns:
        npt.NDArray[np.int32]: Read-only ``(num_total_basis,)`` owner rank per DOF;
        ``-1`` for dead DOFs.

    Raises:
        ValueError: If any axis is periodic.
        ValueError: If ``partition.cell_owner`` length does not match
            ``space.num_total_intervals``.
    """
    _reject_periodic(space)
    cell_owner = partition.cell_owner
    if cell_owner.shape[0] != space.num_total_intervals:
        raise ValueError(
            f"partition has {cell_owner.shape[0]} cells; "
            f"expected {space.num_total_intervals} (space.num_total_intervals)."
        )
    num_intervals = space.num_intervals
    num_basis = space.num_basis
    dim = space.dim

    fc_axes: list[npt.NDArray[np.int64]] = []
    lc_axes: list[npt.NDArray[np.int64]] = []
    for sp in space.spaces:
        _, fc, lc = _func_support_1d(sp)
        fc_axes.append(fc.astype(np.int64))
        lc_axes.append(lc.astype(np.int64))

    owners = np.full(space.num_total_basis, -1, dtype=np.int32)
    dof_multi = np.unravel_index(np.arange(space.num_total_basis), num_basis)
    for dof in range(space.num_total_basis):
        axis_ranges = [
            np.arange(fc_axes[d][dof_multi[d][dof]], lc_axes[d][dof_multi[d][dof]] + 1)
            for d in range(dim)
        ]
        mesh = np.meshgrid(*axis_ranges, indexing="ij")
        support_cells = np.ravel_multi_index(tuple(m.ravel() for m in mesh), num_intervals)
        active = support_cells[cell_owner[support_cells] >= 0]
        if active.size:
            owners[dof] = cell_owner[int(active.min())]
    owners.flags.writeable = False
    return owners


class LocalSpace(NamedTuple):
    """The rank-local windowed view of a distributed tensor-product B-spline space.

    Produced by :func:`build_local`. A :class:`typing.NamedTuple` bundling a windowed
    :class:`~pantr.bspline.BsplineSpace` with the maps and masks relating it to the
    global space:

    - ``space`` -- the windowed B-spline space over the bounding box of the rank's
      owned cells and their support-closure halo; a real pantr object whose basis
      equals the global basis pointwise over those cells.
    - ``local_to_global_cell`` -- read-only ``(space.num_total_intervals,)`` map from
      each local knot-span cell to its global cell id.
    - ``local_to_global_dof`` -- read-only ``(space.num_total_basis,)`` map from each
      local DOF to its global DOF id.
    - ``owned_cell_mask`` -- read-only boolean ``(space.num_total_intervals,)`` mask;
      ``True`` for the cells this rank owns (the rest are halo / bounding-box fill).
    - ``owned_dof_mask`` -- read-only boolean ``(space.num_total_basis,)`` mask;
      ``True`` for the DOFs this rank owns.
    - ``n_global_cells`` -- total knot-span cells in the global space.
    - ``n_global_dofs`` -- total basis functions in the global space.
    """

    space: BsplineSpace
    local_to_global_cell: npt.NDArray[np.int64]
    local_to_global_dof: npt.NDArray[np.int64]
    owned_cell_mask: npt.NDArray[np.bool_]
    owned_dof_mask: npt.NDArray[np.bool_]
    n_global_cells: int
    n_global_dofs: int


def build_local(global_space: BsplineSpace, partition: Partition, rank: int) -> LocalSpace:
    """Build the rank-local windowed space of a distributed tensor-product B-spline space.

    Windows ``global_space`` to the bounding box of the cells owned by ``rank`` together
    with their support-closure halo (:func:`compute_halo`), so the local basis equals the
    global basis pointwise over those cells. Composes the grid restriction
    (:meth:`pantr.grid.TensorProductGrid.restrict`) for the cell map and the space
    restriction (:meth:`pantr.bspline.BsplineSpace.restrict`) for the windowed space and
    DOF map, and marks owned cells/DOFs via :func:`dof_owner`.

    Args:
        global_space (BsplineSpace): The global tensor-product B-spline space
            (non-periodic).
        partition (Partition): Owner of every knot-span cell; ``cell_owner`` must have
            length ``global_space.num_total_intervals``.
        rank (int): The rank whose local space is built; must be in
            ``[0, partition.n_parts)``.

    Returns:
        LocalSpace: The windowed space plus local-to-global cell/DOF maps and ownership
        masks.

    Raises:
        ValueError: If ``global_space`` is periodic, ``rank`` is out of range,
            ``partition`` does not match the space's cell count, or ``rank`` owns no
            cells.
    """
    from ..grid import tensor_product_grid  # noqa: PLC0415

    _reject_periodic(global_space)
    if not 0 <= rank < partition.n_parts:
        raise ValueError(f"rank must be in [0, {partition.n_parts}); got {rank}.")
    if partition.n_cells != global_space.num_total_intervals:
        raise ValueError(
            f"partition has {partition.n_cells} cells; "
            f"expected {global_space.num_total_intervals} (space.num_total_intervals)."
        )
    owned = partition.owned_cells(rank)
    if owned.size == 0:
        raise ValueError(f"rank {rank} owns no cells; cannot build a local space.")

    window = np.union1d(owned, compute_halo(global_space, owned))
    restr = global_space.restrict(window)
    grid_restr = tensor_product_grid(global_space).restrict(window)
    local_to_global_cell = grid_restr.local_to_global_cell
    local_to_global_dof = restr.local_to_global_dof
    assert local_to_global_cell.shape[0] == restr.space.num_total_intervals

    cell_owner = partition.cell_owner
    owned_cell_mask = cell_owner[local_to_global_cell] == rank
    owned_dof_mask = dof_owner(global_space, partition)[local_to_global_dof] == rank
    owned_cell_mask.flags.writeable = False
    owned_dof_mask.flags.writeable = False
    return LocalSpace(
        space=restr.space,
        local_to_global_cell=local_to_global_cell,
        local_to_global_dof=local_to_global_dof,
        owned_cell_mask=owned_cell_mask,
        owned_dof_mask=owned_dof_mask,
        n_global_cells=global_space.num_total_intervals,
        n_global_dofs=global_space.num_total_basis,
    )


__all__ = ["LocalSpace", "build_local", "compute_halo", "dof_owner"]
