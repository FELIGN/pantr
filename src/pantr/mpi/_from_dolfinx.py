"""Build a :class:`~pantr.grid.Partition` from a dolfinx mesh's cell distribution.

:func:`from_dolfinx` consumes the cell ownership of a (distributed) ``dolfinx`` mesh
and returns the serial, redundant :class:`~pantr.grid.Partition` over a pantr grid's
cells that the distributed-space machinery consumes -- the bridge for dolfinx-based
consumers (e.g. qugar / tigarx).

``dolfinx`` is never imported here: the mesh (and its MPI communicator) are supplied by
the caller, and only its attributes are read. This module therefore has no import-time
dependency on ``dolfinx`` or ``mpi4py`` -- ``import pantr.mpi`` still works without them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..grid import Partition

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
            ``mesh.topology.original_cell_index``.
        n_cells (int): Total number of cells in the pantr grid (e.g. ``grid.num_cells``
            or ``space.num_total_intervals``); must be ``>= 1``.
        dolfinx_to_pantr (npt.ArrayLike | None): Optional integer map from a dolfinx
            *original global* cell index to a pantr cell id. ``None`` means identity (the
            mesh is in pantr C-order).

    Returns:
        Partition: A per-cell owner assignment with ``mesh.comm.size`` parts; ``-1`` for
        pantr cells absent from the mesh.

    Raises:
        ValueError: If ``n_cells < 1``; if ``dolfinx_to_pantr`` is not a 1D integer array
            or an original index falls outside it; if a mapped cell id is outside
            ``[0, n_cells)``; or if two ranks own the same cell (inconsistent mesh/map).
    """
    if n_cells < 1:
        raise ValueError(f"n_cells must be >= 1; got {n_cells}.")

    topology = mesh.topology
    tdim = int(topology.dim)
    size_local = int(topology.index_map(tdim).size_local)
    original = np.asarray(topology.original_cell_index, dtype=np.int64)[:size_local]
    local_ids = _map_to_pantr(original, dolfinx_to_pantr, n_cells)

    gathered = mesh.comm.allgather(local_ids)
    cell_owner = np.full(n_cells, -1, dtype=np.int32)
    for owner_rank, ids in enumerate(gathered):
        ids_arr = np.asarray(ids, dtype=np.int64)
        if ids_arr.size == 0:
            continue
        if bool(np.any(cell_owner[ids_arr] != -1)):
            raise ValueError(
                "inconsistent partition: a cell is owned by more than one rank "
                f"(rank {owner_rank} re-claims an already-owned cell)."
            )
        cell_owner[ids_arr] = owner_rank

    return Partition(cell_owner, int(mesh.comm.size))


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
        ValueError: If the map is not a 1D integer array, an original index lies outside
            the map, or a resulting pantr id lies outside ``[0, n_cells)``.
    """
    if dolfinx_to_pantr is None:
        ids = original
    else:
        mapping = np.asarray(dolfinx_to_pantr)
        if mapping.ndim != 1 or not np.issubdtype(mapping.dtype, np.integer):
            raise ValueError("dolfinx_to_pantr must be a 1D integer array.")
        if original.size and int(original.max()) >= mapping.shape[0]:
            raise ValueError(
                f"dolfinx_to_pantr has length {mapping.shape[0]} but an original cell "
                f"index reaches {int(original.max())}."
            )
        ids = mapping[original].astype(np.int64)

    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= n_cells):
        raise ValueError(
            f"mapped pantr cell ids must lie in [0, {n_cells}); "
            f"got range [{int(ids.min())}, {int(ids.max())}]."
        )
    return ids


__all__ = ["from_dolfinx"]
