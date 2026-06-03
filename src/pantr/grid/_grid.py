"""Abstract base class for structured cell grids.

A :class:`Grid` is a partition of a parametric domain into cells with *implicit*
(computed, not stored) connectivity. It is the shared grid abstraction for the
PaNTr stack: background grids for immersed / unfitted discretizations, knot-span
grids of B-spline spaces, and (later) hierarchical refinement grids all satisfy
this contract.

Design
------

The contract is deliberately small. A concrete grid must define only:

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
from typing import TYPE_CHECKING

import numpy as np

from ._grid_utils import _as_float64
from ._tags import CellTags, FacetTags

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy.typing as npt

    from ..geometry import AABB
    from ..transform import AffineTransform
    from ._bvh import BVH


class Grid(abc.ABC):
    """Abstract structured cell grid with implicit connectivity.

    See the module docstring for the contract and the set of methods a subclass
    must implement versus those provided as overridable defaults. The size
    metadata is exposed through the :attr:`ndim` and :attr:`num_cells`
    properties.
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
            self._bvh = BVH.from_cell_bounds(cell_lo, cell_hi)
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
            self._cell_tags = CellTags(self.num_cells)
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
            self._facet_tags = FacetTags(self.num_cells, 2 * self.ndim)
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


__all__ = ["Grid"]
