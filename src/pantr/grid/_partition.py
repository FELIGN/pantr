"""Cell-ownership partition for distributing a structured grid.

A :class:`Partition` records, for every cell of a grid (or the knot-span grid of a
B-spline space), which rank owns it -- the serial, communication-free descriptor
consumed by the distributed-space machinery. It is produced either by consuming an
external partition (for example a dolfinx mesh) or by a native graph partitioner,
and is intentionally space-agnostic: it stores only an integer owner per cell.

A wrapper
---------

Since the 2026-08-27 amendment to ``design/cross_backend_types.md`` the partition
itself is owned by the C++ core (``cpp/include/pantr/grid/partition.hpp``) and this
class holds one. Ownership moved rather than being duplicated: there is one
implementation of a partition and one Python class in front of it. Under
``PANTR_BACKEND=python`` the thing held is :class:`_PartitionPython`, which is the
Python **backend** and not merely an oracle -- most of CI runs on it, and
:mod:`pantr.mpi` builds a partition on every rank.

Where the validation lives
--------------------------

Following the rule ``cpp/include/pantr/core/error.hpp`` states for the whole port:
the **dtype and rank** check ("``cell_owner`` must be a 1D integer array") is here,
because both facts are gone by the time the binding hands C++ a span, and so is the
narrowing cast to ``int32`` that the pre-port class performed; the **value** checks
-- ``n_parts >= 1``, every owner in ``[-1, n_parts)``, and ``owned_cells``' rank
range -- live in the C++ type, where a caller with no interpreter is protected by
them too.

One consequence, stated rather than left to be discovered: the pre-port class
checked ``n_parts`` *before* it looked at ``cell_owner``, and :func:`_new_impl` now
coerces first. An argument that violates **both** contracts at once therefore
reports the array complaint where it used to report the ``n_parts`` one. Both are
``ValueError``, and every input that violates only one contract is unaffected.

**The coercion runs before the backend is chosen, and that is what makes the two
agree.** Doing it inside the C++ branch alone left ``PANTR_BACKEND`` deciding which
of two simultaneous violations a caller was told about --
``Partition(numpy.array([1.5, 2.5]), 0)`` reported the ``n_parts`` complaint under
the Python backend and the array one under C++. That is the one thing the backend
switch must never change, and ``tests/test_grid_partition.py`` pins it on exactly
that input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from ._grid_utils import _python_backend_selected

if TYPE_CHECKING:
    import numpy.typing as npt

    from pantr._pantr_cpp import Partition as _CppPartition

    _Impl: TypeAlias = "_PartitionPython | _CppPartition"
    """The implementation a :class:`Partition` holds: the Python one, or the C++ one.

    Type-checking only. The two are unrelated nominal types that happen to offer the
    same surface, which is the port's whole claim; naming the union here is what
    lets the checker verify it instead of taking it on trust.
    """


def _as_int32_owners(cell_owner: npt.ArrayLike) -> npt.NDArray[np.int32]:
    """Coerce ``cell_owner`` to a C-contiguous 1-D ``int32`` array.

    The cast is narrowing, and that is the pre-port behaviour reproduced
    deliberately: an owner above ``2**31 - 1`` wraps here and is then rejected by
    the range check in the C++ type, in that order, under either backend.

    Args:
        cell_owner (npt.ArrayLike): Per-cell owner ranks.

    Returns:
        npt.NDArray[np.int32]: The owners, 1-D, ``int32`` and C-contiguous.

    Raises:
        ValueError: If ``cell_owner`` is not a 1-D integer array.
    """
    owner = np.asarray(cell_owner)
    if owner.ndim != 1 or not np.issubdtype(owner.dtype, np.integer):
        raise ValueError("cell_owner must be a 1D integer array.")
    return np.ascontiguousarray(owner, dtype=np.int32)


class _PartitionPython:
    """The pure-Python partition: the backend under ``PANTR_BACKEND=python``.

    This was the public :class:`Partition` until the 2026-08-27 amendment to
    ``design/cross_backend_types.md`` made the partition a C++-owned type. It is not
    merely a parity oracle: :func:`_impl_class` returns it whenever the Python
    backend is selected, which is how the package still works in a tree with no
    compiled extension.

    Its surface is deliberately the C++ type's, so that :class:`Partition` can call
    the same methods on either. :attr:`Partition.active_mask` is therefore not here:
    it is one comparison over :attr:`cell_owner`, computed by the wrapper so that
    the two backends cannot get it differently.

    Attributes:
        cell_owner (npt.NDArray[np.int32]): Per-cell owners, read-only.
        n_parts (int): Number of parts (ranks).
        n_cells (int): Number of cells.
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
        owner = _as_int32_owners(cell_owner)
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


def _impl_class() -> type[_PartitionPython] | type[_CppPartition]:
    """The implementation class the active backend selects.

    Returns:
        type[_PartitionPython] | type[_CppPartition]: :class:`_PartitionPython`
        under the Python backend, the bound C++ class otherwise.

    Raises:
        RuntimeError: If the C++ backend is requested and is not available.
    """
    if _python_backend_selected():
        return _PartitionPython
    from pantr import _pantr_cpp  # noqa: PLC0415  (optional, imported only when selected)

    return _pantr_cpp.Partition


def _new_impl(cell_owner: npt.ArrayLike, n_parts: int) -> _Impl:
    """Build a partition in whichever implementation the active backend selects.

    Args:
        cell_owner (npt.ArrayLike): Per-cell owner ranks.
        n_parts (int): Number of parts (ranks).

    Returns:
        _Impl: The implementation object; a :class:`_PartitionPython` or a C++
        handle.

    Raises:
        RuntimeError: If the C++ backend is requested and is not available.
    """
    # Coerced BEFORE the backend is looked at, and that ordering is the contract
    # rather than a convenience. Evaluating it inside the C++ branch alone made
    # `PANTR_BACKEND` decide which of two simultaneous violations was reported --
    # `Partition(numpy.array([1.5, 2.5]), 0)` said "n_parts must be >= 1" under one
    # backend and "cell_owner must be a 1D integer array" under the other. Doing it
    # once, unconditionally, is what `_bvh.py` and `_tags.py` already do.
    owner = _as_int32_owners(cell_owner)
    cls = _impl_class()
    if cls is _PartitionPython:
        return _PartitionPython(owner, n_parts)
    return cls(owner, int(n_parts))


class Partition:
    """A per-cell owner assignment over a grid's cells.

    Records, for every cell, the rank that owns it -- or ``-1`` for an inactive cell
    excluded from the partition (e.g. an exterior / trimmed cell). The owner array is
    coerced to a read-only ``int32`` array on construction and the object is otherwise
    immutable. Owners and counts are exposed through the :attr:`cell_owner`,
    :attr:`n_parts`, :attr:`n_cells`, and :attr:`active_mask` properties.

    **This class is a wrapper**, holding a partition owned by the C++ core (or, under
    ``PANTR_BACKEND=python``, a :class:`_PartitionPython`). See the module docstring
    for the ownership rule and for where validation lives.

    Attributes:
        cell_owner (npt.NDArray[np.int32]): Per-cell owners, read-only.
        n_parts (int): Number of parts (ranks).
        n_cells (int): Number of cells.
        active_mask (npt.NDArray[np.bool_]): Which cells some rank owns.
    """

    __slots__ = ("_impl",)

    _impl: _Impl
    """The implementation this wrapper holds; see :func:`_impl_class`."""

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
        object.__setattr__(self, "_impl", _new_impl(cell_owner, n_parts))

    def __setattr__(self, name: str, value: object) -> None:
        """Reject post-construction attribute writes.

        Args:
            name (str): The attribute being set.
            value (object): The value it would take.

        Raises:
            AttributeError: Always -- :class:`Partition` is immutable.
        """
        raise AttributeError(f"Partition is immutable; cannot set attribute {name!r}.")

    def __delattr__(self, name: str) -> None:
        """Reject attribute deletion.

        Args:
            name (str): The attribute being deleted.

        Raises:
            AttributeError: Always -- :class:`Partition` is immutable.
        """
        raise AttributeError(f"Partition is immutable; cannot delete attribute {name!r}.")

    def __reduce__(self) -> tuple[type[Partition], tuple[npt.NDArray[np.int32], int]]:
        """Pickle by the constructor's own arguments, never by the implementation.

        The C++ handle is not picklable and must not become part of the wire
        format: a pickle written under one backend has to load under the other, or
        the backend switch would silently become a data-format switch. This one is
        load-bearing beyond that -- :mod:`pantr.mpi` moves a partition between ranks
        through ``mpi4py``, which pickles.

        Returns:
            tuple: The class and the ``(cell_owner, n_parts)`` pair to rebuild it
            from.
        """
        return (type(self), (self.cell_owner, self.n_parts))

    def __repr__(self) -> str:
        """Return a compact representation naming the two counts.

        Formatted here rather than by the implementation, so that the two backends
        print identically. The owners are left out: there is one per cell, and a
        partition over a real mesh would print a page of them.

        Returns:
            str: ``"Partition(n_cells=..., n_parts=...)"``.
        """
        return f"Partition(n_cells={self.n_cells}, n_parts={self.n_parts})"

    @property
    def cell_owner(self) -> npt.NDArray[np.int32]:
        """Get the read-only per-cell owner array.

        Under the C++ backend this is a zero-copy view into the partition's own
        storage rather than a copy, which is what the pre-port class also handed
        back. A partition has no mutator, so nothing can move that storage while a
        view is alive.

        Returns:
            npt.NDArray[np.int32]: ``(n_cells,)`` owners; ``-1`` for inactive cells.
        """
        return self._impl.cell_owner

    @property
    def n_parts(self) -> int:
        """Get the number of parts (ranks).

        Returns:
            int: The part count (``>= 1``).
        """
        return int(self._impl.n_parts)

    @property
    def n_cells(self) -> int:
        """Get the total number of cells (active and inactive).

        Returns:
            int: Length of :attr:`cell_owner`.
        """
        return int(self._impl.n_cells)

    @property
    def active_mask(self) -> npt.NDArray[np.bool_]:
        r"""Get a boolean mask of the active cells (owned by some rank).

        Computed here rather than by the implementation: it is one comparison over
        :attr:`cell_owner`, which both backends hand back as the same values, so
        deriving it in one place is what makes it impossible for the two to
        disagree.

        Returns:
            npt.NDArray[np.bool\_]: Fresh ``(n_cells,)`` mask; ``True`` where the cell
            owner is not ``-1``.
        """
        return np.asarray(self.cell_owner >= 0)

    def owned_cells(self, rank: int) -> npt.NDArray[np.int64]:
        """Return the flat ids of the cells owned by ``rank``, ascending.

        Args:
            rank (int): Owner rank in ``[0, n_parts)``.

        Returns:
            npt.NDArray[np.int64]: A fresh, writeable array of the cell ids with
            ``cell_owner == rank``, in increasing order.

        Raises:
            ValueError: If ``rank`` is outside ``[0, n_parts)``.
        """
        return np.asarray(self._impl.owned_cells(int(rank)), dtype=np.int64)


__all__ = ["Partition"]
