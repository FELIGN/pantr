"""Cell-ownership partition for distributing a structured grid.

A :class:`Partition` records, for every cell of a grid (or the knot-span grid of a
B-spline space), which rank owns it -- the serial, communication-free descriptor
consumed by the distributed-space machinery. It is produced either by consuming an
external partition (for example a dolfinx mesh) or by a native graph partitioner,
and is intentionally space-agnostic: it stores only an integer owner per cell.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


class Partition:
    """A per-cell owner assignment over a grid's cells.

    Records, for every cell, the rank that owns it -- or ``-1`` for an inactive cell
    excluded from the partition (e.g. an exterior / trimmed cell). The owner array is
    coerced to a read-only ``int32`` array on construction and the object is otherwise
    immutable. Owners and counts are exposed through the :attr:`cell_owner`,
    :attr:`n_parts`, :attr:`n_cells`, and :attr:`active_mask` properties.
    """

    __slots__ = ("_cell_owner", "_n_parts")

    def __init__(self, cell_owner: npt.ArrayLike, n_parts: int) -> None:
        """Build a partition from a per-cell owner array.

        Args:
            cell_owner (npt.ArrayLike): Per-cell owner ranks (``-1`` for inactive
                cells); coerced to a read-only 1D ``int32`` array.
            n_parts (int): Number of parts (ranks); must be ``>= 1``.

        Raises:
            ValueError: If ``n_parts < 1``, ``cell_owner`` is not 1D integer, or any
                owner is outside ``[-1, n_parts)``.
        """
        if n_parts < 1:
            raise ValueError(f"n_parts must be >= 1; got {n_parts}.")
        owner = np.asarray(cell_owner)
        if owner.ndim != 1 or not np.issubdtype(owner.dtype, np.integer):
            raise ValueError("cell_owner must be a 1D integer array.")
        owner = np.ascontiguousarray(owner, dtype=np.int32)
        if owner.size and (int(owner.min()) < -1 or int(owner.max()) >= n_parts):
            raise ValueError(
                f"cell_owner values must lie in [-1, {n_parts}); "
                f"got range [{int(owner.min())}, {int(owner.max())}]."
            )
        owner.flags.writeable = False
        self._cell_owner = owner
        self._n_parts = int(n_parts)

    @property
    def cell_owner(self) -> npt.NDArray[np.int32]:
        """Get the read-only per-cell owner array.

        Returns:
            npt.NDArray[np.int32]: ``(n_cells,)`` owners; ``-1`` for inactive cells.
        """
        return self._cell_owner

    @property
    def n_parts(self) -> int:
        """Get the number of parts (ranks).

        Returns:
            int: The part count (``>= 1``).
        """
        return self._n_parts

    @property
    def n_cells(self) -> int:
        """Get the total number of cells (active and inactive).

        Returns:
            int: Length of :attr:`cell_owner`.
        """
        return int(self._cell_owner.shape[0])

    @property
    def active_mask(self) -> npt.NDArray[np.bool_]:
        r"""Get a boolean mask of the active cells (owned by some rank).

        Returns:
            npt.NDArray[np.bool\_]: Fresh ``(n_cells,)`` mask; ``True`` where the cell
            owner is not ``-1``.
        """
        return self._cell_owner >= 0

    def owned_cells(self, rank: int) -> npt.NDArray[np.int64]:
        """Return the flat ids of the cells owned by ``rank``, ascending.

        Args:
            rank (int): Owner rank in ``[0, n_parts)``.

        Returns:
            npt.NDArray[np.int64]: Sorted cell ids with ``cell_owner == rank``.

        Raises:
            ValueError: If ``rank`` is outside ``[0, n_parts)``.
        """
        if not 0 <= rank < self._n_parts:
            raise ValueError(f"rank must be in [0, {self._n_parts}); got {rank}.")
        return np.flatnonzero(self._cell_owner == rank).astype(np.int64)


__all__ = ["Partition"]
