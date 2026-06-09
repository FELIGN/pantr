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

from typing import TYPE_CHECKING

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
        IndexError: If any owned cell id is out of range ``[0, num_total_intervals)``.
    """
    _reject_periodic(space)
    num_intervals = space.num_intervals
    n_cells = space.num_total_intervals
    owned = np.asarray(owned_cells, dtype=np.int64).ravel()
    if owned.size and (int(owned.min()) < 0 or int(owned.max()) >= n_cells):
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
        ValueError: If any axis is periodic, or ``partition`` does not match the
            space's cell count.
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


__all__ = ["compute_halo", "dof_owner"]
