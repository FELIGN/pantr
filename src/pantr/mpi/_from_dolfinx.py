"""Build a :class:`~pantr.grid.Partition` from a dolfinx mesh's cell distribution.

:func:`from_dolfinx` consumes the cell ownership of a (distributed) ``dolfinx`` mesh
and returns the serial, redundant :class:`~pantr.grid.Partition` over a pantr grid's
cells that the distributed-space machinery consumes -- the bridge for dolfinx-based
consumers (e.g. QUGaR / tIGArx).

``dolfinx`` is never imported here: the mesh (and its MPI communicator) are supplied by
the caller, and only its attributes are read. This module therefore has no import-time
dependency on ``dolfinx`` or ``mpi4py`` -- ``import pantr.mpi`` still works without them.
At runtime, ``mesh.comm`` must behave as an ``mpi4py.MPI.Comm`` (providing ``size`` and
``allgather``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..grid import Partition
from ._thread_policy import _ensure_default_thread_policy

if TYPE_CHECKING:
    import numpy.typing as npt


def from_dolfinx(
    mesh: Any,  # noqa: ANN401 -- a dolfinx.mesh.Mesh; dolfinx is an optional, untyped dep
    n_cells: int,
    *,
    dolfinx_to_pantr: npt.ArrayLike | None = None,
) -> Partition:
    """Build a cell :class:`~pantr.grid.Partition` from a dolfinx mesh.

    Reads the locally-owned cells on every rank and MPI-allgathers their original
    (input) global indices over ``mesh.comm`` to assemble one global, redundant per-cell
    owner array -- the rank that owns each cell in dolfinx. The result is identical on
    every rank. Pantr cells absent from the mesh (e.g. exterior / trimmed cells excluded
    by an immersed consumer) get owner ``-1``.

    The correspondence between a dolfinx cell and a pantr grid cell is its *original*
    cell index (``mesh.topology.original_cell_index``): with ``dolfinx_to_pantr=None`` the
    mesh is assumed to have been built in pantr's C-order cell numbering, so the original
    index *is* the pantr flat cell id. Pass ``dolfinx_to_pantr`` to map the original
    dolfinx global cell index to a pantr cell id explicitly.

    Args:
        mesh (Any): A ``dolfinx.mesh.Mesh``. The required interface is duck-typed (no
            ``dolfinx`` import): ``mesh.comm`` (an MPI communicator with ``size`` and
            ``allgather``), ``mesh.topology.dim``,
            ``mesh.topology.index_map(dim).size_local``, and
            ``mesh.topology.original_cell_index`` (must support integer-slice indexing).
        n_cells (int): Total number of cells in the pantr grid (e.g.
            ``grid.num_cells``); must be ``>= 1``.
        dolfinx_to_pantr (npt.ArrayLike | None): Optional integer map from a dolfinx
            *original global* cell index to a pantr cell id. ``None`` means identity (the
            mesh is in pantr C-order).

    Returns:
        Partition: A per-cell owner assignment with ``mesh.comm.size`` parts; ``-1`` for
        pantr cells absent from the mesh.

    Raises:
        ValueError: If ``n_cells < 1``; if ``topology.index_map(dim).size_local`` is
            negative; if ``dolfinx_to_pantr`` is not a 1D integer array, is not injective,
            or an original index falls outside it; if any (original or mapped) pantr cell
            id is outside ``[0, n_cells)``; if ``allgather`` returns an inconsistent number
            of entries or a non-1D result for any rank; if any gathered cell id is out of
            range; or if two ranks own the same cell (inconsistent mesh/map).

    Note:
        Calling this engages the default MPI thread policy: unless threads were
        explicitly configured, this process's Numba thread pool is limited to one
        thread per rank. See :func:`pantr.mpi.configure_threads`.
    """
    _ensure_default_thread_policy()
    if n_cells < 1:
        raise ValueError(f"n_cells must be >= 1; got {n_cells}.")

    topology = mesh.topology
    tdim = int(topology.dim)
    size_local = int(topology.index_map(tdim).size_local)
    if size_local < 0:
        raise ValueError(
            f"topology.index_map({tdim}).size_local returned {size_local}; "
            "expected a non-negative count of locally-owned cells."
        )
    original = np.asarray(topology.original_cell_index, dtype=np.int64)[:size_local]
    local_ids = _map_to_pantr(original, dolfinx_to_pantr, n_cells)

    gathered = mesh.comm.allgather(local_ids)
    n_parts = int(mesh.comm.size)
    if len(gathered) != n_parts:
        raise ValueError(
            f"allgather returned {len(gathered)} entries but mesh.comm.size is {n_parts}; "
            "the communicator's allgather is inconsistent."
        )
    cell_owner = np.full(n_cells, -1, dtype=np.int32)
    for owner_rank, ids in enumerate(gathered):
        ids_arr = np.asarray(ids, dtype=np.int64)
        if ids_arr.ndim != 1:
            raise ValueError(
                f"allgather result for rank {owner_rank} is not a 1-D array "
                f"(got shape {ids_arr.shape})."
            )
        if ids_arr.size == 0:
            continue
        if int(ids_arr.min()) < 0 or int(ids_arr.max()) >= n_cells:
            raise ValueError(
                f"gathered cell ids from rank {owner_rank} are outside [0, {n_cells}): "
                f"got range [{int(ids_arr.min())}, {int(ids_arr.max())}]."
            )
        conflict_mask = cell_owner[ids_arr] != -1
        if bool(np.any(conflict_mask)):
            conflict_cell = int(ids_arr[conflict_mask][0])
            prev_owner = int(cell_owner[conflict_cell])
            raise ValueError(
                f"inconsistent partition: cell {conflict_cell} is claimed by both "
                f"rank {prev_owner} and rank {owner_rank}."
            )
        cell_owner[ids_arr] = owner_rank

    return Partition(cell_owner, n_parts)


def _map_to_pantr(
    original: npt.NDArray[np.int64],
    dolfinx_to_pantr: npt.ArrayLike | None,
    n_cells: int,
) -> npt.NDArray[np.int64]:
    """Map owned dolfinx original global cell indices to pantr cell ids.

    Args:
        original (npt.NDArray[np.int64]): Original global indices of the rank's owned
            cells.
        dolfinx_to_pantr (npt.ArrayLike | None): Optional dolfinx-global -> pantr-id map;
            ``None`` for identity.
        n_cells (int): Total pantr cell count, for range validation.

    Returns:
        npt.NDArray[np.int64]: Pantr cell ids of the owned cells.

    Raises:
        ValueError: If the map is not a 1D integer array, original indices are negative,
            an original index lies outside the map, the map is not injective, or a
            resulting pantr id lies outside ``[0, n_cells)``.
    """
    if dolfinx_to_pantr is None:
        ids = original
    else:
        mapping = np.asarray(dolfinx_to_pantr)
        if mapping.ndim != 1 or mapping.dtype.kind not in ("i", "u"):
            raise ValueError("dolfinx_to_pantr must be a 1D integer array.")
        if original.size and int(original.min()) < 0:
            raise ValueError(
                f"original cell indices must be non-negative; got {int(original.min())}."
            )
        if original.size and int(original.max()) >= mapping.shape[0]:
            raise ValueError(
                f"dolfinx_to_pantr has length {mapping.shape[0]} but an original cell "
                f"index reaches {int(original.max())}."
            )
        ids = mapping[original].astype(np.int64)
        if ids.size and np.unique(ids).size != ids.size:
            raise ValueError(
                "dolfinx_to_pantr is not injective: two dolfinx cells map to the same pantr id."
            )

    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= n_cells):
        raise ValueError(
            f"mapped pantr cell ids must lie in [0, {n_cells}); "
            f"got range [{int(ids.min())}, {int(ids.max())}]."
        )
    return ids


__all__ = ["from_dolfinx"]
