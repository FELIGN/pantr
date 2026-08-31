"""Sparse named tags for grid cells and facets.

Two lazy registries attach integer labels to a sparse subset of a grid's
entities, following the model used by ``dolfinx.mesh.MeshTags``: each named tag
is a pair of parallel arrays ``(ids, values)`` (or ``(keys, values)`` for
facets). Entities not listed are untagged. This keeps memory proportional to the
number of *tagged* entities, which suits both whole-grid classification
(``in`` / ``out`` / ``cut``) and boundary-condition marking on a handful of
facets.

- :class:`CellTags` -- named ``(cell_ids, values)`` associations, with
  :meth:`CellTags.to_dense` to scatter a tag into a dense ``(num_cells,)`` array
  when a downstream Numba kernel needs one.
- :class:`FacetTags` -- named ``(keys, values)`` associations where each key is a
  ``(cell_id, local_facet_id)`` pair.

Both registries are created lazily by :class:`pantr.grid.Grid` and stay empty
(zero per-cell footprint) until the first :meth:`set` call. Classification logic
(deciding which cells are inside / outside / cut) is the consumer's
responsibility; these classes only store the result.

Both are wrappers
-----------------

Since the 2026-08-27 amendment to ``design/cross_backend_types.md`` the registries
themselves are owned by the C++ core (``cpp/include/pantr/grid/tags.hpp``) and
these classes hold one. Ownership moved rather than being duplicated: there is one
implementation of a registry and one Python class in front of it. Under
``PANTR_BACKEND=python`` the thing held is :class:`_CellTagsPython` or
:class:`_FacetTagsPython` instead, which is the Python **backend** and not merely
an oracle -- most of CI runs on it.

Where the validation lives, and why it is in two places
-------------------------------------------------------

The split is honest rather than tidy, and ``cpp/include/pantr/core/error.hpp``
states the rule for the whole port: **type-kind checks and key lookups stay in this
wrapper; value and range checks live in the C++ type.** nanobind 2.14.0 has no path
producing a ``TypeError`` and maps ``std::out_of_range`` to ``IndexError`` rather
than ``KeyError``, so neither a dtype complaint nor a missing-name lookup could be
raised from C++ with the message the oracle raises. So:

- **here:** the integer-dtype checks on ``ids``, ``keys``, ``values`` and
  ``to_dense``'s ``dtype`` (``TypeError``); the array *shape* checks, since a numpy
  shape is gone by the time the binding hands C++ a span; the scalar broadcast;
  ``KeyError`` for an unregistered name; and allocating and filling
  :meth:`CellTags.to_dense`'s destination, because ``numpy.full`` is what decides
  what an out-of-range ``fill`` does and reproducing its message by hand would be a
  second spelling of numpy's own contract;
- **in the type:** the id and key ranges, uniqueness, the two arrays' lengths, and
  the ``OverflowError`` when a stored value does not fit the destination dtype.

One consequence is worth stating rather than leaving to be discovered. The
wrapper's coercions all run *before* the type's value checks, while the pre-port
class interleaved them, so an input violating **two** contracts at once now reports
the wrapper's complaint where it used to report the value one --
``set("a", [5], [1, 2])`` on a five-cell grid, say, which is both out of range and
the wrong length. The two backends agree with each other, which is what the backend
switch has to guarantee; what changed is which of two simultaneous violations is
named first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, TypeAlias

import numpy as np

from ._grid_utils import _python_backend_selected

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy.typing as npt

    from pantr._pantr_cpp import CellTags as _CppCellTags
    from pantr._pantr_cpp import FacetTags as _CppFacetTags

    _CellImpl: TypeAlias = "_CellTagsPython | _CppCellTags"
    """The implementation a :class:`CellTags` holds: the Python one, or the C++ one.

    Type-checking only. The two are unrelated nominal types that happen to offer the
    same surface, which is the port's whole claim; naming the union here is what
    lets the checker verify it instead of taking it on trust.
    """

    _FacetImpl: TypeAlias = "_FacetTagsPython | _CppFacetTags"
    """The implementation a :class:`FacetTags` holds. See :data:`_CellImpl`."""

    _TagPair: TypeAlias = "tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]"
    """A tag's two parallel arrays, as :meth:`CellTags.__getitem__` returns them."""


# A facet key is the pair (cell_id, local_facet_id); stored as a (M, 2) array.
_FACET_KEY_WIDTH: Final[int] = 2

_NARROWER_THAN_INT64: Final[int] = 8
"""Destination width, in bytes, at or above which ``to_dense`` runs no range check.

Mirrors the `sizeof(IntT) < 8` condition in ``tags.hpp``, and both reproduce the
pre-port class's `out_dtype.itemsize < 8`. It is sound for ``int64``, whose range
contains every stored value, and unsound for ``uint64``, which is equally wide and
still cannot hold a negative one -- so a negative tag value scattered into a
``uint64`` destination wraps silently in both backends. Reproduced rather than
fixed: the docstrings state the limit as the contract ("only raised when ``dtype``
is narrower than ``int64``"), and closing it changes documented behaviour in both
backends at once, which is not a port's decision to take.
"""


def _as_int64_1d(values: npt.ArrayLike, *, name: str) -> npt.NDArray[np.int64]:
    """Coerce ``values`` to a 1-D ``int64`` array.

    Args:
        values (npt.ArrayLike): Integer array-like.
        name (str): Argument name, used in error messages.

    Returns:
        npt.NDArray[np.int64]: A C-contiguous 1-D ``int64`` array.

    Raises:
        TypeError: If ``values`` does not have an integer dtype.
        ValueError: If ``values`` is not 1-D.
    """
    arr = np.asarray(values)
    if arr.dtype.kind not in ("i", "u"):
        raise TypeError(f"{name} must have an integer dtype; got {arr.dtype!r}.")
    arr = np.ascontiguousarray(arr.astype(np.int64, copy=False))
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got shape {arr.shape}.")
    return arr


def _as_int64_keys(keys: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Coerce ``keys`` to a C-contiguous ``(M, 2)`` ``int64`` array.

    Args:
        keys (npt.ArrayLike): Integer array-like of ``(cell_id, local_facet_id)``
            rows.

    Returns:
        npt.NDArray[np.int64]: The keys, ``(M, 2)`` and C-contiguous.

    Raises:
        TypeError: If ``keys`` does not have an integer dtype.
        ValueError: If ``keys`` does not have shape ``(M, 2)``.
    """
    raw = np.asarray(keys)
    if raw.dtype.kind not in ("i", "u"):
        raise TypeError(f"keys must have an integer dtype; got {raw.dtype!r}.")
    arr = np.ascontiguousarray(raw.astype(np.int64, copy=False))
    if arr.ndim != _FACET_KEY_WIDTH or arr.shape[1] != _FACET_KEY_WIDTH:
        raise ValueError(f"keys must have shape (M, 2); got shape {arr.shape}.")
    return arr


def _broadcast_values(
    values: npt.ArrayLike,
    n: int,
    *,
    name: str,
) -> npt.NDArray[np.int64]:
    """Coerce ``values`` to an ``int64`` array of length ``n`` (scalars broadcast).

    Args:
        values (npt.ArrayLike): Scalar integer or length-``n`` integer
            array-like.
        n (int): Required length.
        name (str): Argument name, used in error messages.

    Returns:
        npt.NDArray[np.int64]: A fresh, writeable length-``n`` ``int64`` array.

    Raises:
        TypeError: If ``values`` does not have an integer dtype.
        ValueError: If ``values`` is array-like with a length other than ``n``.
    """
    arr = np.asarray(values)
    if arr.dtype.kind not in ("i", "u"):
        raise TypeError(f"{name} must have an integer dtype; got {arr.dtype!r}.")
    if arr.ndim == 0:
        return np.full(n, int(arr), dtype=np.int64)
    flat = np.ascontiguousarray(arr.astype(np.int64, copy=False)).ravel()
    if flat.shape[0] != n:
        raise ValueError(f"{name} must be a scalar or have length {n}; got length {flat.shape[0]}.")
    return flat


def _require_representable(values: npt.NDArray[np.int64], out_dtype: np.dtype[Any]) -> None:
    """Reject stored values the destination dtype cannot hold.

    Mirrors ``pantr::grid::detail::require_representable`` in ``tags.hpp``, message
    included, so that :meth:`CellTags.to_dense` reads the same under either
    backend. The width condition is :data:`_NARROWER_THAN_INT64`, which documents
    what it is unsound about.

    Args:
        values (npt.NDArray[np.int64]): The stored values.
        out_dtype (np.dtype[Any]): The destination integer dtype.

    Raises:
        OverflowError: If some value is outside ``out_dtype``'s range.
    """
    if values.shape[0] == 0 or out_dtype.itemsize >= _NARROWER_THAN_INT64:
        return
    info = np.iinfo(out_dtype)
    vmin, vmax = int(values.min()), int(values.max())
    if vmin < info.min or vmax > info.max:
        raise OverflowError(
            f"dtype {out_dtype!r} cannot represent all tag values without "
            f"truncation; value range [{vmin}, {vmax}] exceeds "
            f"dtype range [{info.min}, {info.max}]."
        )


class _CellTagsPython:
    """The pure-Python cell-tag registry: the backend under ``PANTR_BACKEND=python``.

    This was the public :class:`CellTags` until the 2026-08-27 amendment to
    ``design/cross_backend_types.md`` made the registry a C++-owned type. It is not
    merely a parity oracle: :func:`_cell_impl_class` returns it whenever the Python
    backend is selected, which is how the package still works in a tree with no
    compiled extension.

    Its surface is deliberately the one the C++ class exposes -- ``get``,
    ``scatter``, ``names``, ``__contains__``, ``__len__`` -- so that
    :class:`CellTags` can call the same methods on either. The Python-facing
    conveniences this class used to carry (``__getitem__``, ``to_dense``) moved up
    into that wrapper, where they are written once for both backends.

    Attributes:
        num_cells (int): Number of cells in the owning grid.
        names (tuple[str, ...]): Registered tag names, in insertion order.
    """

    __slots__ = ("_num_cells", "_tags")

    def __init__(self, num_cells: int) -> None:
        """Create an empty cell-tag registry.

        Args:
            num_cells (int): Number of cells in the owning grid (``>= 0``).

        Raises:
            ValueError: If ``num_cells`` is negative.
        """
        if int(num_cells) < 0:
            raise ValueError(f"num_cells must be >= 0; got {num_cells}.")
        self._num_cells = int(num_cells)
        self._tags: dict[str, _TagPair] = {}

    @property
    def num_cells(self) -> int:
        """Get the number of cells in the owning grid.

        Returns:
            int: The cell count; valid cell ids are ``[0, num_cells)``.
        """
        return self._num_cells

    @property
    def names(self) -> tuple[str, ...]:
        """Get the registered tag names.

        Returns:
            tuple[str, ...]: Tag names in insertion order.
        """
        return tuple(self._tags)

    def __len__(self) -> int:
        """Return the number of registered tags.

        Returns:
            int: Count of distinct tag names.
        """
        return len(self._tags)

    def __contains__(self, name: object) -> bool:
        """Return whether a tag named ``name`` exists.

        Args:
            name (object): Candidate tag name.

        Returns:
            bool: ``True`` iff ``name`` is a registered tag.
        """
        return name in self._tags

    def set(self, name: str, ids: npt.ArrayLike, values: npt.ArrayLike) -> None:
        """Create or replace the tag ``name`` with the association ``ids -> values``.

        A replacement keeps the name's position among :attr:`names`, which a Python
        ``dict`` gives for free and which the C++ registry reproduces on purpose.

        Args:
            name (str): Tag name. Replaces any existing tag with the same name.
            ids (npt.ArrayLike): 1-D integer array-like of cell ids; each must
                satisfy ``0 <= id < num_cells`` and be unique.
            values (npt.ArrayLike): Scalar integer (broadcast to every id) or a
                1-D integer array-like of the same length as ``ids``.

        Raises:
            TypeError: If ``ids`` or ``values`` is not integer-typed.
            ValueError: If ``ids`` is not 1-D, contains duplicates, or has an id
                out of range, or if ``values`` has a length other than
                ``len(ids)``.
        """
        id_arr = _as_int64_1d(ids, name="ids")
        if id_arr.shape[0] > 0 and (int(id_arr.min()) < 0 or int(id_arr.max()) >= self._num_cells):
            raise ValueError(
                f"cell ids must be in [0, {self._num_cells}); "
                f"got range [{int(id_arr.min())}, {int(id_arr.max())}]."
            )
        if np.unique(id_arr).shape[0] != id_arr.shape[0]:
            raise ValueError(f"cell ids for tag {name!r} must be unique; got duplicates.")
        val_arr = _broadcast_values(values, id_arr.shape[0], name="values")
        order = np.argsort(id_arr, kind="stable")
        sorted_ids = np.ascontiguousarray(id_arr[order])
        sorted_vals = np.ascontiguousarray(val_arr[order])
        sorted_ids.flags.writeable = False
        sorted_vals.flags.writeable = False
        self._tags[str(name)] = (sorted_ids, sorted_vals)

    def get(self, name: str) -> _TagPair:
        """Return the ``(ids, values)`` arrays for tag ``name``.

        Args:
            name (str): Tag name.

        Returns:
            tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: Read-only
            ``(ids, values)`` arrays sorted by ``ids``.

        Raises:
            KeyError: If no tag named ``name`` exists. :class:`CellTags` checks
                membership before calling, so no wrapped caller reaches it; the C++
                counterpart raises ``std::out_of_range`` here for the same reason.
        """
        return self._tags[name]

    def remove(self, name: str) -> None:
        """Delete the tag ``name``.

        Args:
            name (str): Tag name.

        Raises:
            KeyError: If no tag named ``name`` exists.
        """
        del self._tags[name]

    def scatter(self, name: str, out: npt.NDArray[Any]) -> None:
        """Write tag ``name``'s values into ``out`` at its ids.

        Does **not** fill: every entry the tag does not name is left as the caller
        left it. :meth:`CellTags.to_dense` allocates and fills, so that
        ``numpy.full`` keeps deciding what an out-of-range ``fill`` does.

        Args:
            name (str): Tag name.
            out (npt.NDArray[Any]): Destination of shape ``(num_cells,)`` with an
                integer dtype; written in place.

        Raises:
            KeyError: If no tag named ``name`` exists.
            ValueError: If ``out`` does not have shape ``(num_cells,)``.
            OverflowError: If a stored value is outside ``out.dtype``'s range,
                subject to the width limit :data:`_NARROWER_THAN_INT64` documents.
        """
        ids, values = self._tags[name]
        if out.shape != (self._num_cells,):
            raise ValueError(f"scatter: out must have length {self._num_cells}; got {out.shape}.")
        _require_representable(values, out.dtype)
        out[ids] = values


class _FacetTagsPython:
    """The pure-Python facet-tag registry: the backend under ``PANTR_BACKEND=python``.

    The counterpart of :class:`_CellTagsPython` for facets; see that class for why
    it survives the port and why its surface is the C++ one.

    Attributes:
        num_cells (int): Number of cells in the owning grid.
        facets_per_cell (int): Number of local facets per cell.
        names (tuple[str, ...]): Registered tag names, in insertion order.
    """

    __slots__ = ("_facets_per_cell", "_num_cells", "_tags")

    def __init__(self, num_cells: int, facets_per_cell: int) -> None:
        """Create an empty facet-tag registry.

        Args:
            num_cells (int): Number of cells in the owning grid (``>= 0``).
            facets_per_cell (int): Number of local facets per cell (``>= 1``).

        Raises:
            ValueError: If ``num_cells`` is negative or ``facets_per_cell`` is
                ``< 1``.
        """
        if int(num_cells) < 0:
            raise ValueError(f"num_cells must be >= 0; got {num_cells}.")
        if int(facets_per_cell) < 1:
            raise ValueError(f"facets_per_cell must be >= 1; got {facets_per_cell}.")
        self._num_cells = int(num_cells)
        self._facets_per_cell = int(facets_per_cell)
        self._tags: dict[str, _TagPair] = {}

    @property
    def num_cells(self) -> int:
        """Get the number of cells in the owning grid.

        Returns:
            int: The cell count.
        """
        return self._num_cells

    @property
    def facets_per_cell(self) -> int:
        """Get the number of local facets per cell.

        Returns:
            int: ``2 * ndim`` for an axis-aligned box grid.
        """
        return self._facets_per_cell

    @property
    def names(self) -> tuple[str, ...]:
        """Get the registered tag names.

        Returns:
            tuple[str, ...]: Tag names in insertion order.
        """
        return tuple(self._tags)

    def __len__(self) -> int:
        """Return the number of registered tags.

        Returns:
            int: Count of distinct tag names.
        """
        return len(self._tags)

    def __contains__(self, name: object) -> bool:
        """Return whether a tag named ``name`` exists.

        Args:
            name (object): Candidate tag name.

        Returns:
            bool: ``True`` iff ``name`` is a registered tag.
        """
        return name in self._tags

    def set(self, name: str, keys: npt.ArrayLike, values: npt.ArrayLike) -> None:
        """Create or replace the tag ``name`` with the association ``keys -> values``.

        Args:
            name (str): Tag name. Replaces any existing tag with the same name.
            keys (npt.ArrayLike): ``(M, 2)`` integer array-like of
                ``(cell_id, local_facet_id)`` rows; each row must be unique with
                ``0 <= cell_id < num_cells`` and
                ``0 <= local_facet_id < facets_per_cell``.
            values (npt.ArrayLike): Scalar integer (broadcast to every key) or a
                1-D integer array-like of length ``M``.

        Raises:
            TypeError: If ``keys`` or ``values`` is not integer-typed.
            ValueError: If ``keys`` does not have shape ``(M, 2)``, contains a
                duplicate or out-of-range key, or ``values`` has a length other
                than ``M``.
        """
        key_arr = _as_int64_keys(keys)
        if key_arr.shape[0] > 0:
            cids = key_arr[:, 0]
            lfids = key_arr[:, 1]
            if int(cids.min()) < 0 or int(cids.max()) >= self._num_cells:
                raise ValueError(
                    f"facet cell ids must be in [0, {self._num_cells}); "
                    f"got range [{int(cids.min())}, {int(cids.max())}]."
                )
            if int(lfids.min()) < 0 or int(lfids.max()) >= self._facets_per_cell:
                raise ValueError(
                    f"local facet ids must be in [0, {self._facets_per_cell}); "
                    f"got range [{int(lfids.min())}, {int(lfids.max())}]."
                )
            if np.unique(key_arr, axis=0).shape[0] != key_arr.shape[0]:
                raise ValueError(f"facet keys for tag {name!r} must be unique; got duplicates.")
        val_arr = _broadcast_values(values, key_arr.shape[0], name="values")
        # Lexicographic sort by (cell_id, local_facet_id) for deterministic order.
        order = np.lexsort((key_arr[:, 1], key_arr[:, 0]))
        sorted_keys = np.ascontiguousarray(key_arr[order])
        sorted_vals = np.ascontiguousarray(val_arr[order])
        sorted_keys.flags.writeable = False
        sorted_vals.flags.writeable = False
        self._tags[str(name)] = (sorted_keys, sorted_vals)

    def get(self, name: str) -> _TagPair:
        """Return the ``(keys, values)`` arrays for tag ``name``.

        Args:
            name (str): Tag name.

        Returns:
            tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: Read-only
            ``(keys, values)`` where ``keys`` has shape ``(M, 2)`` and is sorted
            lexicographically.

        Raises:
            KeyError: If no tag named ``name`` exists. See
                :meth:`_CellTagsPython.get`.
        """
        return self._tags[name]

    def remove(self, name: str) -> None:
        """Delete the tag ``name``.

        Args:
            name (str): Tag name.

        Raises:
            KeyError: If no tag named ``name`` exists.
        """
        del self._tags[name]

    def scatter(self, name: str, out: npt.NDArray[Any]) -> None:
        """Write tag ``name``'s values into ``out`` at its keys.

        Does **not** fill; see :meth:`_CellTagsPython.scatter`.

        Args:
            name (str): Tag name.
            out (npt.NDArray[Any]): Destination of shape
                ``(num_cells, facets_per_cell)`` with an integer dtype; written in
                place.

        Raises:
            KeyError: If no tag named ``name`` exists.
            ValueError: If ``out`` does not have the grid's shape.
            OverflowError: If a stored value is outside ``out.dtype``'s range,
                subject to the width limit :data:`_NARROWER_THAN_INT64` documents.
        """
        keys, values = self._tags[name]
        shape = (self._num_cells, self._facets_per_cell)
        if out.shape != shape:
            raise ValueError(f"scatter: out must be {shape}; got {out.shape}.")
        _require_representable(values, out.dtype)
        if keys.shape[0] > 0:
            out[keys[:, 0], keys[:, 1]] = values


def _cell_impl_class() -> type[_CellTagsPython] | type[_CppCellTags]:
    """The cell-registry implementation class the active backend selects.

    Returns:
        type[_CellTagsPython] | type[_CppCellTags]: :class:`_CellTagsPython` under
        the Python backend, the bound C++ class otherwise.

    Raises:
        RuntimeError: If the C++ backend is requested and is not available.
    """
    if _python_backend_selected():
        return _CellTagsPython
    from pantr import _pantr_cpp  # noqa: PLC0415  (optional, imported only when selected)

    return _pantr_cpp.CellTags


def _facet_impl_class() -> type[_FacetTagsPython] | type[_CppFacetTags]:
    """The facet-registry implementation class the active backend selects.

    Returns:
        type[_FacetTagsPython] | type[_CppFacetTags]: :class:`_FacetTagsPython`
        under the Python backend, the bound C++ class otherwise.

    Raises:
        RuntimeError: If the C++ backend is requested and is not available.
    """
    if _python_backend_selected():
        return _FacetTagsPython
    from pantr import _pantr_cpp  # noqa: PLC0415  (optional, imported only when selected)

    return _pantr_cpp.FacetTags


def _rebuild_cell_tags(
    num_cells: int,
    tags: tuple[tuple[str, npt.NDArray[np.int64], npt.NDArray[np.int64]], ...],
) -> CellTags:
    """Rebuild a :class:`CellTags` from its constructor argument and its tags.

    The reconstruction :meth:`CellTags.__reduce__` names. A module-level function
    rather than a ``__setstate__``, because a registry is built by :meth:`set` calls
    and replaying them in order is what restores the insertion order a pickle has to
    preserve.

    Args:
        num_cells (int): The owning grid's cell count.
        tags (tuple): One ``(name, ids, values)`` triple per tag, in insertion
            order.

    Returns:
        CellTags: The rebuilt registry, holding whichever implementation the
        *reading* process's backend selects.
    """
    registry = CellTags(num_cells)
    for name, ids, values in tags:
        registry.set(name, ids, values)
    return registry


def _rebuild_facet_tags(
    num_cells: int,
    facets_per_cell: int,
    tags: tuple[tuple[str, npt.NDArray[np.int64], npt.NDArray[np.int64]], ...],
) -> FacetTags:
    """Rebuild a :class:`FacetTags` from its constructor arguments and its tags.

    See :func:`_rebuild_cell_tags`.

    Args:
        num_cells (int): The owning grid's cell count.
        facets_per_cell (int): Local facets per cell.
        tags (tuple): One ``(name, keys, values)`` triple per tag, in insertion
            order.

    Returns:
        FacetTags: The rebuilt registry.
    """
    registry = FacetTags(num_cells, facets_per_cell)
    for name, keys, values in tags:
        registry.set(name, keys, values)
    return registry


class CellTags:
    """Sparse named integer tags over a grid's cells.

    Each tag named ``name`` is a pair of parallel ``int64`` arrays
    ``(ids, values)`` sorted by ``ids``; a cell not listed in ``ids`` is
    untagged under ``name``. Distinct tag names are independent. The owning
    grid's cell count is exposed through the :attr:`num_cells` property.

    **This class is a wrapper**, holding a registry owned by the C++ core (or, under
    ``PANTR_BACKEND=python``, a :class:`_CellTagsPython`). See the module docstring
    for the ownership rule and for where validation lives.

    Iteration order is insertion order, and a :meth:`set` that replaces an existing
    name leaves that name where it was. Both backends guarantee it, because
    :meth:`__iter__` and :attr:`names` are public and a caller can see it.

    Attributes:
        num_cells (int): Number of cells in the owning grid.
        names (tuple[str, ...]): Registered tag names, in insertion order.
    """

    __slots__ = ("_impl",)

    _impl: _CellImpl
    """The implementation this wrapper holds; see :func:`_cell_impl_class`."""

    def __init__(self, num_cells: int) -> None:
        """Create an empty cell-tag registry.

        Args:
            num_cells (int): Number of cells in the owning grid (``>= 0``).

        Raises:
            ValueError: If ``num_cells`` is negative.
        """
        object.__setattr__(self, "_impl", _cell_impl_class()(int(num_cells)))

    @classmethod
    def _wrap(cls, impl: _CellImpl) -> CellTags:
        """Adopt an implementation object that already exists and is already valid.

        The grid types need this: a grid OWNS its registry, so the wrapper in front
        of a grid must present the registry the implementation already holds rather
        than construct a fresh one. Constructing one would hand back an empty
        registry, and every assertion about a tag set earlier would then fail far
        from the cause.

        Args:
            impl (_CellImpl): The implementation object to adopt.

        Returns:
            CellTags: A wrapper around ``impl``, with no re-validation.
        """
        self = object.__new__(cls)
        object.__setattr__(self, "_impl", impl)
        return self

    def __setattr__(self, name: str, value: object) -> None:
        """Reject attribute writes.

        The registry accumulates through :meth:`set` and :meth:`remove`; the
        implementation it holds is not something a caller may swap, and neither is
        the cell count it was sized for.

        Args:
            name (str): The attribute being set.
            value (object): The value it would take.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError(f"CellTags has no settable attribute {name!r}.")

    def __delattr__(self, name: str) -> None:
        """Reject attribute deletion.

        Args:
            name (str): The attribute being deleted.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError(f"CellTags has no deletable attribute {name!r}.")

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Pickle by the constructor argument and the tags, never by the handle.

        The C++ handle is not picklable and must not become part of the wire
        format: a pickle written under one backend has to load under the other, or
        the backend switch would silently become a data-format switch.

        Returns:
            tuple: :func:`_rebuild_cell_tags` and the arguments to replay.
        """
        tags = tuple((name, *self[name]) for name in self.names)
        return (_rebuild_cell_tags, (self.num_cells, tags))

    def __repr__(self) -> str:
        """Return a concise representation showing the cell count and tag names.

        Formatted here rather than by the implementation, so that the two backends
        print identically.

        Note:
            The pre-port classes gave a ``__repr__`` to :class:`FacetTags` only.
            This one is a deliberate, small addition made in the same change, so
            that the two registries are not gratuitously different: a registry that
            prints as ``<... object at 0x...>`` beside a sibling that names its tags
            is a difference with no reason behind it.

        Returns:
            str: ``"CellTags(num_cells=..., tags=[...])"``.
        """
        return f"CellTags(num_cells={self.num_cells}, tags={list(self.names)!r})"

    @property
    def num_cells(self) -> int:
        """Get the number of cells in the owning grid.

        Returns:
            int: The cell count; valid cell ids are ``[0, num_cells)``.
        """
        return int(self._impl.num_cells)

    @property
    def names(self) -> tuple[str, ...]:
        """Get the registered tag names.

        Returns:
            tuple[str, ...]: Tag names in insertion order.
        """
        return tuple(self._impl.names)

    def __len__(self) -> int:
        """Return the number of registered tags.

        Returns:
            int: Count of distinct tag names.
        """
        return len(self._impl)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the registered tag names.

        Returns:
            Iterator[str]: Iterator over tag names in insertion order.
        """
        return iter(self.names)

    def __contains__(self, name: object) -> bool:
        """Return whether a tag named ``name`` exists.

        A non-string is reported as absent rather than passed on: :meth:`set`
        stores ``str(name)``, so no other kind of key can exist, and the C++
        registry's own ``__contains__`` takes a ``str``.

        Args:
            name (object): Candidate tag name.

        Returns:
            bool: ``True`` iff ``name`` is a registered tag.
        """
        return isinstance(name, str) and name in self._impl

    def set(self, name: str, ids: npt.ArrayLike, values: npt.ArrayLike) -> None:
        """Create or replace the tag ``name`` with the association ``ids -> values``.

        Args:
            name (str): Tag name. Replaces any existing tag with the same name,
                keeping the position it holds in :attr:`names`.
            ids (npt.ArrayLike): 1-D integer array-like of cell ids; each must
                satisfy ``0 <= id < num_cells`` and be unique.
            values (npt.ArrayLike): Scalar integer (broadcast to every id) or a
                1-D integer array-like of the same length as ``ids``.

        Raises:
            TypeError: If ``ids`` or ``values`` is not integer-typed.
            ValueError: If ``ids`` is not 1-D, contains duplicates, or has an id
                out of range, or if ``values`` has a length other than
                ``len(ids)``.
        """
        id_arr = _as_int64_1d(ids, name="ids")
        val_arr = _broadcast_values(values, id_arr.shape[0], name="values")
        self._impl.set(str(name), id_arr, val_arr)

    def __getitem__(self, name: str) -> _TagPair:
        """Return the ``(ids, values)`` arrays for tag ``name``.

        Under the C++ backend the two arrays are read-only views into the
        registry's own storage, and they stay valid past a later :meth:`set` or
        :meth:`remove` on the same name: a tag's storage is reference-counted, so a
        handed-out view holds it alive. That is what the Python backend gets from
        its own reference counting, and the port would otherwise have introduced a
        use-after-free the pre-port class could not have.

        Args:
            name (str): Tag name.

        Returns:
            tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: Read-only
            ``(ids, values)`` arrays sorted by ``ids``.

        Raises:
            KeyError: If no tag named ``name`` exists.
        """
        if name not in self:
            raise KeyError(name)
        return self._impl.get(name)

    def remove(self, name: str) -> None:
        """Delete the tag ``name``.

        Args:
            name (str): Tag name.

        Raises:
            KeyError: If no tag named ``name`` exists.
        """
        if name not in self:
            raise KeyError(name)
        self._impl.remove(name)

    def to_dense(
        self,
        name: str,
        *,
        fill: int = 0,
        dtype: npt.DTypeLike = np.int64,
    ) -> npt.NDArray[Any]:
        r"""Scatter tag ``name`` into a dense ``(num_cells,)`` array.

        Untagged cells receive ``fill``. Useful when a downstream Numba kernel
        wants a per-cell label array rather than the sparse representation.

        Allocated and filled here rather than by the implementation, so that
        ``numpy.full`` keeps deciding what an out-of-range ``fill`` does; the
        implementation writes only the tagged entries.

        Args:
            name (str): Tag name.
            fill (int): Value for untagged cells. Defaults to ``0``.
            dtype (npt.DTypeLike): Output integer dtype. Defaults to
                ``numpy.int64``. Values are stored as ``int64`` internally; if
                ``dtype`` is narrower than ``int64`` and any stored value falls
                outside the dtype's representable range, an ``OverflowError`` is
                raised rather than silently truncating the value.

        Returns:
            npt.NDArray[Any]: Fresh, writeable ``(num_cells,)`` array with
            ``dtype`` as the scalar type.

        Raises:
            KeyError: If no tag named ``name`` exists.
            TypeError: If ``dtype`` is not an integer or unsigned-integer dtype.
            OverflowError: If any stored value cannot be represented exactly in
                ``dtype`` (only raised when ``dtype`` is narrower than ``int64``),
                or if ``fill`` itself does not fit ``dtype``, which is
                ``numpy.full``'s own refusal rather than this method's.
        """
        out_dtype = np.dtype(dtype)
        if out_dtype.kind not in ("i", "u"):
            raise TypeError(f"dtype must be an integer dtype; got {out_dtype!r}.")
        if name not in self:
            raise KeyError(name)
        out = np.full(self.num_cells, fill, dtype=out_dtype)
        self._impl.scatter(name, out)
        return out


class FacetTags:
    """Sparse named integer tags over a grid's local facets.

    Each facet is addressed by a ``(cell_id, local_facet_id)`` key, with
    ``local_facet_id`` in ``[0, facets_per_cell)``. Each tag named ``name`` is a
    pair ``(keys, values)`` where ``keys`` is an ``(M, 2)`` ``int64`` array of
    ``(cell_id, local_facet_id)`` rows and ``values`` is a length-``M``
    ``int64`` array, sorted lexicographically by key. The owning grid's cell
    count and per-cell facet count are exposed through the :attr:`num_cells` and
    :attr:`facets_per_cell` properties.

    **This class is a wrapper**; see :class:`CellTags` and the module docstring.

    Attributes:
        num_cells (int): Number of cells in the owning grid.
        facets_per_cell (int): Number of local facets per cell.
        names (tuple[str, ...]): Registered tag names, in insertion order.
    """

    __slots__ = ("_impl",)

    _impl: _FacetImpl
    """The implementation this wrapper holds; see :func:`_facet_impl_class`."""

    def __init__(self, num_cells: int, facets_per_cell: int) -> None:
        """Create an empty facet-tag registry.

        Args:
            num_cells (int): Number of cells in the owning grid (``>= 0``).
            facets_per_cell (int): Number of local facets per cell (``>= 1``).

        Raises:
            ValueError: If ``num_cells`` is negative or ``facets_per_cell`` is
                ``< 1``.
        """
        impl = _facet_impl_class()(int(num_cells), int(facets_per_cell))
        object.__setattr__(self, "_impl", impl)

    @classmethod
    def _wrap(cls, impl: _FacetImpl) -> FacetTags:
        """Adopt an implementation object that already exists and is already valid.

        See :meth:`CellTags._wrap` for why the grid types need it.

        Args:
            impl (_FacetImpl): The implementation object to adopt.

        Returns:
            FacetTags: A wrapper around ``impl``, with no re-validation.
        """
        self = object.__new__(cls)
        object.__setattr__(self, "_impl", impl)
        return self

    def __setattr__(self, name: str, value: object) -> None:
        """Reject attribute writes.

        Args:
            name (str): The attribute being set.
            value (object): The value it would take.

        Raises:
            AttributeError: Always. See :meth:`CellTags.__setattr__`.
        """
        raise AttributeError(f"FacetTags has no settable attribute {name!r}.")

    def __delattr__(self, name: str) -> None:
        """Reject attribute deletion.

        Args:
            name (str): The attribute being deleted.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError(f"FacetTags has no deletable attribute {name!r}.")

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Pickle by the constructor arguments and the tags, never by the handle.

        See :meth:`CellTags.__reduce__`.

        Returns:
            tuple: :func:`_rebuild_facet_tags` and the arguments to replay.
        """
        tags = tuple((name, *self[name]) for name in self.names)
        return (_rebuild_facet_tags, (self.num_cells, self.facets_per_cell, tags))

    def __repr__(self) -> str:
        """Return a concise representation showing the cell/facet counts and tag names.

        Formatted here rather than by the implementation, so that the two backends
        print identically.

        Returns:
            str: ``"FacetTags(num_cells=..., facets_per_cell=..., tags=[...])"``
        """
        return (
            f"FacetTags(num_cells={self.num_cells}, "
            f"facets_per_cell={self.facets_per_cell}, "
            f"tags={list(self.names)!r})"
        )

    @property
    def num_cells(self) -> int:
        """Get the number of cells in the owning grid.

        Returns:
            int: The cell count.
        """
        return int(self._impl.num_cells)

    @property
    def facets_per_cell(self) -> int:
        """Get the number of local facets per cell.

        Returns:
            int: ``2 * ndim`` for an axis-aligned box grid.
        """
        return int(self._impl.facets_per_cell)

    @property
    def names(self) -> tuple[str, ...]:
        """Get the registered tag names.

        Returns:
            tuple[str, ...]: Tag names in insertion order.
        """
        return tuple(self._impl.names)

    def __len__(self) -> int:
        """Return the number of registered tags.

        Returns:
            int: Count of distinct tag names.
        """
        return len(self._impl)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the registered tag names.

        Returns:
            Iterator[str]: Iterator over tag names in insertion order.
        """
        return iter(self.names)

    def __contains__(self, name: object) -> bool:
        """Return whether a tag named ``name`` exists.

        Args:
            name (object): Candidate tag name.

        Returns:
            bool: ``True`` iff ``name`` is a registered tag. See
            :meth:`CellTags.__contains__` for why a non-string is absent rather
            than an error.
        """
        return isinstance(name, str) and name in self._impl

    def set(self, name: str, keys: npt.ArrayLike, values: npt.ArrayLike) -> None:
        """Create or replace the tag ``name`` with the association ``keys -> values``.

        Args:
            name (str): Tag name. Replaces any existing tag with the same name,
                keeping the position it holds in :attr:`names`.
            keys (npt.ArrayLike): ``(M, 2)`` integer array-like of
                ``(cell_id, local_facet_id)`` rows; each row must be unique with
                ``0 <= cell_id < num_cells`` and
                ``0 <= local_facet_id < facets_per_cell``.
            values (npt.ArrayLike): Scalar integer (broadcast to every key) or a
                1-D integer array-like of length ``M``.

        Raises:
            TypeError: If ``keys`` or ``values`` is not integer-typed.
            ValueError: If ``keys`` does not have shape ``(M, 2)``, contains a
                duplicate or out-of-range key, or ``values`` has a length other
                than ``M``.
        """
        key_arr = _as_int64_keys(keys)
        val_arr = _broadcast_values(values, key_arr.shape[0], name="values")
        self._impl.set(str(name), key_arr, val_arr)

    def __getitem__(self, name: str) -> _TagPair:
        """Return the ``(keys, values)`` arrays for tag ``name``.

        See :meth:`CellTags.__getitem__` for the lifetime the returned views have.

        Args:
            name (str): Tag name.

        Returns:
            tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: Read-only
            ``(keys, values)`` where ``keys`` has shape ``(M, 2)`` and is sorted
            lexicographically.

        Raises:
            KeyError: If no tag named ``name`` exists.
        """
        if name not in self:
            raise KeyError(name)
        return self._impl.get(name)

    def remove(self, name: str) -> None:
        """Delete the tag ``name``.

        Args:
            name (str): Tag name.

        Raises:
            KeyError: If no tag named ``name`` exists.
        """
        if name not in self:
            raise KeyError(name)
        self._impl.remove(name)

    def to_dense(
        self,
        name: str,
        *,
        fill: int = 0,
        dtype: npt.DTypeLike = np.int64,
    ) -> npt.NDArray[Any]:
        r"""Scatter tag ``name`` into a dense ``(num_cells, facets_per_cell)`` array.

        Untagged facets receive ``fill``. Useful when a downstream Numba kernel
        wants a per-facet label array rather than the sparse representation. See
        :meth:`CellTags.to_dense` for why the allocation stays here.

        Args:
            name (str): Tag name.
            fill (int): Value for untagged facets. Defaults to ``0``.
            dtype (npt.DTypeLike): Output integer dtype. Defaults to
                ``numpy.int64``. Values are stored as ``int64`` internally; if
                ``dtype`` is narrower than ``int64`` and any stored value falls
                outside the dtype's representable range, an ``OverflowError`` is
                raised rather than silently truncating the value.

        Returns:
            npt.NDArray[Any]: Fresh, writeable ``(num_cells, facets_per_cell)``
            array with ``dtype`` as the scalar type.

        Raises:
            KeyError: If no tag named ``name`` exists.
            TypeError: If ``dtype`` is not an integer or unsigned-integer dtype.
            OverflowError: If any stored value cannot be represented exactly in
                ``dtype`` (only raised when ``dtype`` is narrower than ``int64``),
                or if ``fill`` itself does not fit ``dtype``.
        """
        out_dtype = np.dtype(dtype)
        if out_dtype.kind not in ("i", "u"):
            raise TypeError(f"dtype must be an integer dtype; got {out_dtype!r}.")
        if name not in self:
            raise KeyError(name)
        out = np.full((self.num_cells, self.facets_per_cell), fill, dtype=out_dtype)
        self._impl.scatter(name, out)
        return out


__all__ = ["CellTags", "FacetTags"]
