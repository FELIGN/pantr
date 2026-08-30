"""The grid contract, and the Python implementation of it that is on its way out.

A grid is a partition of a parametric domain into cells with *implicit*
(computed, not stored) connectivity. It is the shared grid abstraction for the
PaNTr stack: background grids for immersed / unfitted discretizations, knot-span
grids of B-spline spaces, and hierarchical refinement grids all satisfy this
contract.

Two names, and the split is the point
-------------------------------------

:class:`Grid` is a :class:`typing.Protocol`. It is what a *consumer* may rely on
-- every public member, annotated as ``grid: Grid`` throughout the library -- and
it is satisfied structurally, so a class becomes a grid by having the members
rather than by inheriting. It carries real default implementations, so
``Grid.boundary_facets(g)`` computes the generic answer for any ``g`` that
satisfies it; that unbound call is how a specialization is tested against the
default it replaces.

``_GridPython`` is the abstract base class the concrete Python grids derive
from. It adds the three cache slots and their ``__init__`` to :class:`Grid` and
nothing else, so there is exactly one copy of every default. It is also where
the *implementer's* contract lives: its ``__abstractmethods__`` are the five
primitives below, and a subclass omitting one cannot be instantiated. The
:class:`Grid` protocol cannot express that -- structural typing has no
"inherited" half, so the protocol has to list every member a consumer may call,
and omitting an inherited default fails the same way as omitting a primitive.

**``_GridPython`` is scaffolding.** The grid layer is being ported to C++
(``cpp/include/pantr/grid/grid.hpp``), where the generic defaults are a CRTP
mixin and the primitives are a concept. Once the C++ grids are wrapped, this
class exists only as the parity oracle, and "keep parity with ``_GridPython``"
is never a reason not to improve the C++.

Design
------

A concrete grid must define only:

- :attr:`Grid.ndim`, :attr:`Grid.num_cells` -- size metadata;
- :meth:`Grid.cell_bounds` -- the axis-aligned ``(lo, hi)`` corners of a cell;
- :meth:`Grid.locate` -- the cell containing a point (or ``None``);
- :meth:`Grid.neighbor_across_facet` -- the cell across a given local facet.

Everything else has a concrete default built from those primitives, assuming the
common case of axis-aligned box cells with ``2 * ndim`` facets: cell AABBs,
reference maps, facet bounds, neighbour lists, boundary-facet detection, batch
point location, and an :class:`AABB` overlap query backed by a lazily-built
:class:`pantr.grid.BVH`. Subclasses override any default for which they have a
cheaper specialization (for example, :class:`pantr.grid.TensorProductGrid`
replaces :meth:`locate`, :meth:`neighbor_across_facet`, and the cell-bounds
collection with per-axis arithmetic).

Cells live in parametric coordinates; mapping to physical space is the
responsibility of the geometry (for example, a B-spline) layered on top of the
grid -- the grid itself stores no geometry map.

Tagging
-------

Each grid exposes two lazily-created sparse tag registries,
:attr:`Grid.cell_tags` and :attr:`Grid.facet_tags`, for attaching integer labels
to a subset of cells / facets (in / out / cut classification, boundary-condition
markers, ...). They stay empty until first use, so an untagged grid carries no
per-cell tag footprint. Deciding *what* to tag is the consumer's job.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

from ._grid_utils import _as_float64
from ._tags import CellTags, FacetTags

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy.typing as npt

    from ..geometry import AABB
    from ..transform import AffineTransform
    from ._bvh import BVH


class Grid(Protocol):
    """Structural contract for a cell grid with implicit connectivity.

    A :class:`typing.Protocol`: any object with these members is a grid, whether
    or not it inherits anything. See the module docstring for the split between
    this and ``_GridPython``, and for the five primitives a concrete grid
    must supply.

    The private members below are part of the structural contract, which is
    unusual enough to say why: the defaults are real implementations, and three
    of them memoize into named slots. Anything satisfying this protocol
    therefore owes those three slots under those three names -- which is also
    what the tag and BVH identity assertions in the test suite read directly.

    Those three writes carry a ``# type: ignore[misc]``. This class declares
    ``__slots__ = ()`` and never the memo slots themselves, because the storage
    belongs to whoever implements the protocol; mypy checks ``__slots__``
    against the class a write appears in rather than against the implementer's.

    Attributes:
        _bvh (BVH | None): Memoized spatial index; ``None`` until first built.
        _cell_tags (CellTags | None): Memoized cell-tag registry; ``None`` until
            first use.
        _facet_tags (FacetTags | None): Memoized facet-tag registry; ``None``
            until first use.
    """

    # `Generic` does not declare `__slots__`, so a Protocol that omits this one
    # line gives every explicit subclass a `__dict__` even where that subclass
    # declares `__slots__` itself. Nothing in the suite would catch it: the cost
    # is per-instance memory plus the quiet return of settable attributes.
    __slots__ = ()

    _bvh: BVH | None
    _cell_tags: CellTags | None
    _facet_tags: FacetTags | None

    # ------------------------------------------------------------------
    # Abstract contract
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def ndim(self) -> int:
        """Get the spatial dimension of the grid.

        Returns:
            int: Number of axes (``>= 1``).
        """

    @property
    @abc.abstractmethod
    def num_cells(self) -> int:
        """Get the number of cells in this (local) grid.

        Returns:
            int: Non-negative cell count.
        """

    @abc.abstractmethod
    def cell_bounds(
        self,
        cid: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the axis-aligned ``(lo, hi)`` corners of cell ``cid``.

        Args:
            cid (int): Cell identifier; must satisfy ``0 <= cid < num_cells``.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Fresh,
            writeable length-``ndim`` ``float64`` arrays.

        Raises:
            IndexError: If ``cid`` is out of range.
        """

    @abc.abstractmethod
    def locate(self, pt: npt.ArrayLike) -> int | None:
        """Return the cell containing ``pt``, or ``None`` if ``pt`` is outside.

        Args:
            pt (npt.ArrayLike): Length-``ndim`` point in parametric coordinates.

        Returns:
            int | None: Containing cell id, or ``None`` when ``pt`` lies outside
            every cell.

        Raises:
            ValueError: If ``pt`` does not have length ``ndim``.
        """

    @abc.abstractmethod
    def neighbor_across_facet(self, cid: int, lfid: int) -> int | None:
        """Return the cell across local facet ``lfid`` of ``cid``, or ``None``.

        The result is ``None`` iff the facet lies on the grid's outer boundary.

        Args:
            cid (int): Cell identifier.
            lfid (int): Local facet identifier in ``[0, num_local_facets(cid))``.

        Returns:
            int | None: Neighbouring cell id, or ``None`` on a boundary facet.

        Raises:
            IndexError: If ``cid`` or ``lfid`` is out of range.
        """

    # ------------------------------------------------------------------
    # Cell accessors (defaults built on the contract)
    # ------------------------------------------------------------------

    def iter_cells(self) -> Iterator[int]:
        """Yield every cell identifier exactly once, in id order.

        Returns:
            Iterator[int]: ``iter(range(num_cells))``.
        """
        return iter(range(self.num_cells))

    def cell_aabb(self, cid: int) -> AABB:
        """Return cell ``cid``'s axis-aligned bounding box.

        Args:
            cid (int): Cell identifier.

        Returns:
            AABB: Equivalent to ``AABB(*self.cell_bounds(cid))``.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        from ..geometry import AABB  # noqa: PLC0415

        lo, hi = self.cell_bounds(cid)
        return AABB(lo, hi)

    def cell_level(self, cid: int) -> int:
        """Return the refinement level of cell ``cid``.

        Non-hierarchical grids always return ``0``; hierarchical backends use
        level ``>= 1`` for refined cells.

        Args:
            cid (int): Cell identifier.

        Returns:
            int: Refinement level (``0`` for a flat grid).

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        self._check_cid(cid)
        return 0

    def child_cells(self, cid: int) -> tuple[int, ...]:
        """Return the immediate refinement children of cell ``cid``.

        For a flat (non-hierarchical) grid this is always empty.

        Args:
            cid (int): Cell identifier.

        Returns:
            tuple[int, ...]: Child cell ids; empty for a flat grid.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        self._check_cid(cid)
        return ()

    def reference_map(self, cid: int) -> AffineTransform:
        """Return the affine map ``[0, 1]^ndim -> cell`` for cell ``cid``.

        For an axis-aligned cell this is ``T(u) = diag(hi - lo) @ u + lo``.

        Args:
            cid (int): Cell identifier.

        Returns:
            AffineTransform: Push-forward from the unit cube to the cell.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        from ..transform import AffineTransform  # noqa: PLC0415

        lo, hi = self.cell_bounds(cid)
        return AffineTransform(np.diag(hi - lo), lo)

    def neighbors(self, cid: int) -> list[int]:
        """Return the facet-neighbour cell ids of ``cid``.

        Built by collecting :meth:`neighbor_across_facet` over every local facet
        and dropping boundary facets (``None``).

        Args:
            cid (int): Cell identifier.

        Returns:
            list[int]: Neighbouring cell ids; length between ``ndim`` (corner
            cell) and ``2 * ndim`` (interior cell) for a box grid.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        result: list[int] = []
        for lfid in range(self.num_local_facets(cid)):
            neighbor = self.neighbor_across_facet(cid, lfid)
            if neighbor is not None:
                result.append(neighbor)
        return result

    # ------------------------------------------------------------------
    # Restriction / windowing
    # ------------------------------------------------------------------

    def restrict(self, cell_ids: npt.ArrayLike) -> GridRestriction:
        """Return the structured sub-grid spanning a subset of cells.

        The sub-grid is the smallest grid of the same kind that contains every
        requested cell -- for a tensor-product grid, the multi-index bounding box
        of ``cell_ids``. Its breakpoints are *pure slices* of this grid's (never
        re-clamped or re-based), so a B-spline or field built on the sub-grid
        agrees pointwise with the global one over the shared cells.

        The bounding box may contain cells that were not requested (when
        ``cell_ids`` is non-convex); :attr:`GridRestriction.in_subset` flags, per
        local cell, whether it was requested (``True``) or is bounding-box fill
        (``False``).

        Restriction is an *optional* grid capability: this base implementation
        raises :class:`NotImplementedError`. Grids with structured windowing (for
        example :class:`pantr.grid.TensorProductGrid`) override it.

        Args:
            cell_ids (npt.ArrayLike): Flat cell identifiers to span; duplicates
                are ignored. Each must satisfy ``0 <= cid < num_cells``.

        Returns:
            GridRestriction: The windowed sub-grid, the ``local_to_global_cell``
            map, and the ``in_subset`` mask.

        Raises:
            NotImplementedError: If this grid kind does not support restriction.
            ValueError: If ``cell_ids`` is empty (in overriding implementations).
            IndexError: If any cell id is out of range (in overriding
                implementations).
            TypeError: If ``cell_ids`` is not integer-valued (in overriding
                implementations).
        """
        # Two statements rather than a bare `raise`, and the reason is not style.
        # mypy treats a protocol member whose body is a lone `raise
        # NotImplementedError` as *abstract*, which would make `restrict` a
        # required primitive for every explicit subclass -- the opposite of the
        # documented contract that restriction is optional.
        message = f"{type(self).__name__} does not support restrict()."
        raise NotImplementedError(message)

    # ------------------------------------------------------------------
    # Facet accessors (axis-aligned box defaults)
    # ------------------------------------------------------------------

    def num_local_facets(self, cid: int) -> int:
        """Return the number of local facets of cell ``cid``.

        Defaults to ``2 * ndim`` (an axis-aligned box).

        Args:
            cid (int): Cell identifier.

        Returns:
            int: ``2 * ndim``.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        self._check_cid(cid)
        return 2 * self.ndim

    def local_facet_axis_side(self, cid: int, lfid: int) -> tuple[int, int]:
        """Return ``(axis, side)`` for local facet ``lfid`` of cell ``cid``.

        Uses the conventional ``lfid = 2 * axis + side`` encoding, with
        ``axis in [0, ndim)`` and ``side in {0, 1}`` (``0`` = low face,
        ``1`` = high face).

        Args:
            cid (int): Cell identifier.
            lfid (int): Local facet identifier in ``[0, 2 * ndim)``.

        Returns:
            tuple[int, int]: ``(axis, side)``.

        Raises:
            IndexError: If ``cid`` or ``lfid`` is out of range.
        """
        self._check_lfid(cid, lfid)
        axis, side = divmod(int(lfid), 2)
        return axis, side

    def local_facet_bounds(
        self,
        cid: int,
        lfid: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the degenerate ``(lo, hi)`` AABB of facet ``lfid`` of cell ``cid``.

        On the facet's normal axis both corners coincide (the cell's lo corner
        for ``side == 0``, hi corner for ``side == 1``); the other axes span the
        cell's extent.

        Args:
            cid (int): Cell identifier.
            lfid (int): Local facet identifier in ``[0, 2 * ndim)``.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Fresh,
            writeable length-``ndim`` ``float64`` arrays.

        Raises:
            IndexError: If ``cid`` or ``lfid`` is out of range.
        """
        axis, side = self.local_facet_axis_side(cid, lfid)
        lo, hi = self.cell_bounds(cid)
        if side == 0:
            hi[axis] = lo[axis]
        else:
            lo[axis] = hi[axis]
        return lo, hi

    def is_mesh_boundary_facet(self, cid: int, lfid: int) -> bool:
        """Return whether facet ``lfid`` of cell ``cid`` is on the grid's outer boundary.

        Defaults to ``neighbor_across_facet(cid, lfid) is None``.

        Args:
            cid (int): Cell identifier.
            lfid (int): Local facet identifier in ``[0, num_local_facets(cid))``.

        Returns:
            bool: ``True`` iff no neighbouring cell shares the facet.

        Raises:
            IndexError: If ``cid`` or ``lfid`` is out of range.
        """
        return self.neighbor_across_facet(cid, lfid) is None

    def boundary_facets(self) -> npt.NDArray[np.int64]:
        """Return every facet on the grid's outer boundary as ``(cid, lfid)`` rows.

        Only the outer (domain) boundary: a facet shared with any neighbouring cell
        is excluded, whether that neighbour is conforming, coarser, or finer. For a
        hierarchical grid this means level interfaces are **not** boundary facets,
        even though the leaf-cell mesh makes them topologically exterior (hanging
        vertices) and so a consumer searching that mesh over-reports them.

        The default enumerates :meth:`is_mesh_boundary_facet` over every cell and
        local facet, costing ``O(num_cells * 2 * ndim)`` neighbour queries.
        Subclasses with exploitable structure should override it.

        Returns:
            npt.NDArray[np.int64]: Read-only array of shape ``(n, 2)`` whose rows are
            ``(cid, lfid)`` with ``lfid = 2 * axis + side`` (see
            :meth:`local_facet_axis_side`), sorted lexicographically by
            ``(cid, lfid)``.
        """
        rows = [
            (cid, lfid)
            for cid in self.iter_cells()
            for lfid in range(self.num_local_facets(cid))
            if self.is_mesh_boundary_facet(cid, lfid)
        ]
        out = np.array(rows, dtype=np.int64).reshape(len(rows), 2)
        out.flags.writeable = False
        return out

    def hanging_neighbors(self, cid: int, lfid: int) -> tuple[int, ...]:
        """Return all active cells sharing facet ``lfid`` of ``cid``.

        For conforming grids (flat :class:`TensorProductGrid`) this is
        equivalent to :meth:`neighbor_across_facet` wrapped in a tuple.  For
        hierarchical grids, a single coarse face may abut multiple fine cells
        (hanging nodes); this method returns all of them.

        Subclasses with hanging-node support should override this method.

        Args:
            cid (int): Cell identifier.
            lfid (int): Local facet identifier in ``[0, num_local_facets(cid))``.

        Returns:
            tuple[int, ...]: All neighbouring cell ids across the facet;
            empty when the facet lies on the grid's outer boundary.

        Raises:
            IndexError: If ``cid`` or ``lfid`` is out of range.
        """
        nbr = self.neighbor_across_facet(cid, lfid)
        return () if nbr is None else (nbr,)

    # ------------------------------------------------------------------
    # Point location and spatial queries
    # ------------------------------------------------------------------

    def locate_many(self, points: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Locate a batch of points, returning one cell id per point.

        Points outside the grid map to ``-1``. The default loops over
        :meth:`locate`; subclasses may override with a vectorized kernel.

        Args:
            points (npt.ArrayLike): ``(npts, ndim)`` array-like of points, or a
                single length-``ndim`` point.

        Returns:
            npt.NDArray[np.int64]: Shape ``(npts,)`` cell ids; ``-1`` for points
            outside the grid.

        Raises:
            ValueError: If the trailing axis of ``points`` is not ``ndim``.
        """
        pts = self._normalize_points(points)
        out = np.empty(pts.shape[0], dtype=np.int64)
        for i in range(pts.shape[0]):
            cid = self.locate(pts[i])
            out[i] = -1 if cid is None else cid
        return out

    def query_aabb(self, aabb: AABB) -> npt.NDArray[np.int64]:
        """Return the ids of every cell whose AABB overlaps ``aabb``.

        Backed by a :class:`pantr.grid.BVH` over the grid's cell AABBs, built
        lazily on first call and cached for the grid's lifetime. The overlap test
        is inclusive on every axis, so cells touching ``aabb`` on any face, edge,
        or corner are included.

        Args:
            aabb (AABB): Query box; must match :attr:`ndim`.

        Returns:
            npt.NDArray[np.int64]: Overlapping cell ids (unordered).

        Raises:
            ValueError: If ``aabb.ndim != self.ndim``.
        """
        return self.cell_bvh().query_aabb(aabb)

    def cell_bvh(self) -> BVH:
        """Return the cached :class:`pantr.grid.BVH` over the grid's cell AABBs.

        Built lazily on first call from ``collect_cell_bounds`` and cached.
        Building the BVH materializes ``O(num_cells)`` node arrays, so it is
        deferred until an :meth:`query_aabb` (or direct) call needs it -- an
        untagged, un-queried grid never pays this cost.

        Returns:
            BVH: The grid's spatial index over its cell AABBs.

        Warning:
            Not fully thread-safe. Under CPython (with the GIL), concurrent
            first calls may each build a valid :class:`BVH` and the second write
            silently wins — the only cost is redundant construction. Under
            free-threaded Python 3.13+ (``--disable-gil``) the assignment is not
            atomic and a concurrent caller could observe a partially-written
            reference. Call this method once on the main thread before sharing
            the grid across threads.
        """
        from ._bvh import BVH  # noqa: PLC0415

        if self._bvh is None:
            cell_lo, cell_hi = self.collect_cell_bounds()
            self._bvh = BVH.from_cell_bounds(cell_lo, cell_hi)  # type: ignore[misc]
        return self._bvh

    def collect_cell_bounds(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Materialize per-cell ``(lo, hi)`` as ``(num_cells, ndim)`` arrays.

        The default iterates :meth:`cell_bounds` over every cell in id order.
        Subclasses with structure (for example, tensor-product grids) should
        override this with a vectorized construction.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            ``(cell_lo, cell_hi)`` of shape ``(num_cells, ndim)``.
        """
        n = self.num_cells
        cell_lo = np.empty((n, self.ndim), dtype=np.float64)
        cell_hi = np.empty((n, self.ndim), dtype=np.float64)
        for cid in range(n):
            lo, hi = self.cell_bounds(cid)
            cell_lo[cid] = lo
            cell_hi[cid] = hi
        return cell_lo, cell_hi

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------

    @property
    def cell_tags(self) -> CellTags:
        """Get the grid's sparse cell-tag registry (created lazily).

        Returns:
            CellTags: A registry of named ``(cell_ids, values)`` associations,
            empty until first use.
        """
        if self._cell_tags is None:
            self._cell_tags = CellTags(self.num_cells)  # type: ignore[misc]
        return self._cell_tags

    @property
    def facet_tags(self) -> FacetTags:
        """Get the grid's sparse facet-tag registry (created lazily).

        Returns:
            FacetTags: A registry of named ``((cell_id, local_facet_id), value)``
            associations, empty until first use. Sized for ``2 * ndim`` facets
            per cell (an axis-aligned box grid).
        """
        if self._facet_tags is None:
            self._facet_tags = FacetTags(self.num_cells, 2 * self.ndim)  # type: ignore[misc]
        return self._facet_tags

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_cid(self, cid: int) -> None:
        """Raise :class:`IndexError` if ``cid`` is out of range.

        Args:
            cid (int): Candidate cell identifier.

        Raises:
            IndexError: If ``cid`` is negative or ``>= num_cells``.
        """
        if not 0 <= int(cid) < self.num_cells:
            raise IndexError(f"cell id {cid!r} is out of range [0, {self.num_cells}).")

    def _check_lfid(self, cid: int, lfid: int) -> None:
        """Raise :class:`IndexError` if ``lfid`` is not a valid facet of ``cid``.

        Args:
            cid (int): Cell identifier (validated first).
            lfid (int): Candidate local facet identifier.

        Raises:
            IndexError: If ``cid`` is out of range or ``lfid`` is not in
                ``[0, num_local_facets(cid))``.
        """
        self._check_cid(cid)
        n_facets = self.num_local_facets(cid)
        if not 0 <= int(lfid) < n_facets:
            raise IndexError(f"local facet id {lfid!r} is out of range [0, {n_facets}).")

    def _normalize_points(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Coerce ``points`` to a C-contiguous ``(npts, ndim)`` ``float64`` array.

        A single length-``ndim`` point is promoted to shape ``(1, ndim)``.

        Args:
            points (npt.ArrayLike): Points array-like.

        Returns:
            npt.NDArray[np.float64]: Shape ``(npts, ndim)``.

        Raises:
            ValueError: If the trailing axis is not ``ndim`` or the rank is not
                1 or 2.
        """
        arr = _as_float64(points, name="points")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != self.ndim:  # noqa: PLR2004
            raise ValueError(
                f"points must have shape (npts, {self.ndim}) or ({self.ndim},); "
                f"got shape {arr.shape}."
            )
        return np.ascontiguousarray(arr)


class _GridPython(Grid, abc.ABC):
    """The Python implementation of :class:`Grid`, and the port's parity oracle.

    Adds the three cache slots and their initializer to :class:`Grid` and
    nothing else, so every default has exactly one body and a specialization
    tested against ``Grid.<name>(g)`` is tested against the code that actually
    runs.

    This is where the *implementer's* contract is enforced:
    ``_GridPython.__abstractmethods__`` is the five primitives, so a subclass
    that omits one cannot be instantiated. :class:`Grid` cannot express that,
    because a protocol has no inherited half -- see the module docstring.

    Scaffolding: the grid layer's generic operations are moving to C++, and this
    class survives only as the oracle they are compared against.

    Attributes:
        _bvh (BVH | None): Lazily built spatial index.
        _cell_tags (CellTags | None): Lazily created cell-tag registry.
        _facet_tags (FacetTags | None): Lazily created facet-tag registry.
    """

    __slots__ = ("_bvh", "_cell_tags", "_facet_tags")

    def __init__(self) -> None:
        """Initialize the lazy spatial-index and tag-registry slots.

        Subclasses must call ``super().__init__()`` before use so the lazy
        :attr:`cell_tags`, :attr:`facet_tags`, and :meth:`query_aabb` caches are
        available.
        """
        self._bvh: BVH | None = None
        self._cell_tags: CellTags | None = None
        self._facet_tags: FacetTags | None = None


class GridRestriction(NamedTuple):
    """Result of :meth:`Grid.restrict`: a windowed sub-grid plus index maps.

    A :class:`typing.NamedTuple` returned by :meth:`Grid.restrict`:

    - ``grid`` -- the windowed sub-grid: the smallest structured grid containing
      every requested cell. For structured implementations (e.g.
      :class:`TensorProductGrid`) this is the same concrete type as the grid
      that produced it.
    - ``local_to_global_cell`` -- shape ``(grid.num_cells,)`` map from each
      sub-grid cell id to its id in the original grid, in the sub-grid's own
      cell-id order. Read-only.
    - ``in_subset`` -- shape ``(grid.num_cells,)`` boolean mask: ``True`` for
      cells that were in the requested ``cell_ids``, ``False`` for bounding-box
      fill cells (present only when the request was non-convex). Read-only.
    """

    grid: Grid
    local_to_global_cell: npt.NDArray[np.int64]
    in_subset: npt.NDArray[np.bool_]


__all__ = ["Grid", "GridRestriction"]
