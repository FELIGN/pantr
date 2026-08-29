"""Bounding-volume hierarchy over grid cells.

A simple but efficient BVH that indexes a fixed collection of axis-aligned
bounding boxes (one per grid cell). The tree is built once by iterative
median-of-longest-axis splits and queried by iterative descent.

Layout
------

The BVH is held as five parallel arrays, matching the representation consumed by
the kernels:

- ``node_lo`` / ``node_hi``: per-node AABB corners, shape ``(n_nodes, ndim)``.
- ``node_left`` / ``node_right``: child indices; ``-1`` on leaves.
- ``node_cell``: cell identifier on leaves; ``-1`` on internal nodes.

The root is always node ``0`` and covers every cell. For ``N`` cells the tree
has exactly ``2 * N - 1`` nodes (``N`` leaves, ``N - 1`` internal nodes).
Internal-node AABBs are tight: the union of the children's AABBs. Construction
stops at one cell per leaf, so leaves and cells are in one-to-one correspondence.
The construction order (preorder) is deterministic, which keeps query results
reproducible.

``design/bvh.md`` establishes those five arrays as **public API rather than
implementation detail**, so the port reproduces the layout exactly: no leaf
clustering, no ``int32`` indices, no reordering.

Queries
-------

:meth:`BVH.query_aabb` returns the ids of cells whose AABB is not separated from
the query box. Queries run in two passes: a count-only descent sizes the output,
then an emit descent writes the cell ids. Both passes visit the same nodes in the
same order, so the result is a fresh, compact ``int64`` array with no Python-side
list growth.

**The overlap test is a separating-axis test and nothing else**, and it is not
:meth:`pantr.geometry.AABB.overlaps`. It compares inclusively, so a query box
sharing a face with a cell is reported; but it has no emptiness branch, so a
*reversed* query interval is reported against any cell whose own interval contains
it, where the box reports nothing. One cell ``[0, 10]`` in one dimension, queried
with ``lo = 5, hi = 3``, returns cell ``0`` here. Both backends behave that way and
agree with each other; this docstring used to claim agreement with the box, and the
claim is withdrawn rather than the behaviour changed, because reconciling the two
predicates has to move both backends at once.

A wrapper
---------

Since the 2026-08-27 amendment to ``design/cross_backend_types.md`` the hierarchy
itself is owned by the C++ core (``cpp/include/pantr/grid/bvh_tree.hpp``) and this
class holds one. Under ``PANTR_BACKEND=python`` the thing held is
:class:`_BVHPython`, which is the Python **backend** and not merely an oracle.

Where the validation lives, and the one exception the wrapper translates
------------------------------------------------------------------------

Following the rule ``cpp/include/pantr/core/error.hpp`` states for the whole port,
the **dtype** checks (``TypeError``) stay here, because nanobind has no path
producing one; so do the array **shape** messages, which the pre-port class worded
in terms of a numpy shape tuple that is gone once the binding holds a span. The
value and structural checks -- the node-count relation, the leaf encodings, the
child ranges and the tree depth -- live in the C++ type.

The depth is the exception. A tree deeper than the traversal stack is perfectly
well formed and this build cannot walk it, so the C++ type reports it as
``pantr._pantr_cpp.CapacityError``, which is what ``core/error.hpp`` reserves that
type for and the first site in the port to throw it. The pre-port class raised
``ValueError``, and ``PANTR_BACKEND`` must not decide which exception a caller
catches, so :meth:`BVH.__init__` translates it back, message unchanged. **That
translation is a seam**: it exists only while both backends do, and it goes when the
Python backend does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Self, TypeAlias

import numpy as np

from ._bvh_core import _BVH_STACK_DEPTH
from ._grid_backend import (
    bvh_build_kernel,
    bvh_query_count_kernel,
    bvh_query_emit_kernel,
)
from ._grid_utils import _as_float64, _python_backend_selected

if TYPE_CHECKING:
    import numpy.typing as npt

    from pantr._pantr_cpp import BVH as _CppBVH

    from ..geometry import AABB

    _Impl: TypeAlias = "_BVHPython | _CppBVH"
    """The implementation a :class:`BVH` holds: the Python one, or the C++ one.

    Type-checking only. The two are unrelated nominal types that happen to offer the
    same surface, which is the port's whole claim; naming the union here is what
    lets the checker verify it instead of taking it on trust.
    """

_NODE_ARRAY_NDIM: Final[int] = 2
"""``node_lo`` and ``node_hi`` are ``(n_nodes, ndim)`` tables, always rank 2."""


def _validate_node_array_types(
    node_lo: npt.NDArray[np.float64],
    node_hi: npt.NDArray[np.float64],
    node_left: npt.NDArray[np.int64],
    node_right: npt.NDArray[np.int64],
    node_cell: npt.NDArray[np.int64],
) -> tuple[int, int]:
    """Check the five node arrays' dtypes and shapes, in the pre-port order.

    Written once and called from both the wrapper and :class:`_BVHPython`, so the
    two cannot word the same complaint differently. Everything here is a question
    about a numpy array's *type* or *shape*, which is exactly what does not survive
    the crossing into a C++ span -- so this is the part that cannot move into the
    type, and the rest deliberately did.

    Args:
        node_lo (npt.NDArray[np.float64]): Per-node AABB lo corners.
        node_hi (npt.NDArray[np.float64]): Per-node AABB hi corners.
        node_left (npt.NDArray[np.int64]): Left-child indices.
        node_right (npt.NDArray[np.int64]): Right-child indices.
        node_cell (npt.NDArray[np.int64]): Leaf cell identifiers.

    Returns:
        tuple[int, int]: The ``(n_nodes, ndim)`` the arrays imply.

    Raises:
        TypeError: If any array has the wrong dtype.
        ValueError: If the shapes are inconsistent.
    """
    if node_lo.dtype != np.float64 or node_hi.dtype != np.float64:
        raise TypeError(
            f"node_lo / node_hi must be float64; got {node_lo.dtype!r} / {node_hi.dtype!r}."
        )
    if node_lo.ndim != _NODE_ARRAY_NDIM:
        raise ValueError(f"node_lo must be 2-D (n_nodes, ndim); got shape {node_lo.shape}.")
    if node_hi.shape != node_lo.shape:
        raise ValueError(f"node_hi shape {node_hi.shape} must match node_lo shape {node_lo.shape}.")
    n_nodes, ndim = int(node_lo.shape[0]), int(node_lo.shape[1])
    if ndim < 1:
        raise ValueError(f"BVH ndim must be >= 1; got {ndim}.")
    for arr, name in (
        (node_left, "node_left"),
        (node_right, "node_right"),
        (node_cell, "node_cell"),
    ):
        if arr.dtype != np.int64:
            raise TypeError(f"{name} must be int64; got {arr.dtype!r}.")
        if arr.shape != (n_nodes,):
            raise ValueError(f"{name} must have shape ({n_nodes},); got {arr.shape}.")
    return n_nodes, ndim


def _validate_cell_bounds(
    cell_lo: npt.ArrayLike, cell_hi: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Coerce and shape-check the per-cell bounds :meth:`BVH.from_cell_bounds` takes.

    The value checks -- finiteness and ``hi >= lo`` -- are deliberately *not* here:
    they live in the implementation, which is where a caller with no interpreter is
    protected by them too.

    Args:
        cell_lo (npt.ArrayLike): Per-cell lo corners.
        cell_hi (npt.ArrayLike): Per-cell hi corners.

    Returns:
        tuple: The two arrays as C-contiguous ``float64``.

    Raises:
        TypeError: If either input cannot be cast to ``float64``.
        ValueError: If the shapes are inconsistent or ``ndim < 1``.
    """
    lo = _as_float64(cell_lo, name="cell_lo")
    hi = _as_float64(cell_hi, name="cell_hi")
    if lo.ndim != _NODE_ARRAY_NDIM:
        raise ValueError(f"cell_lo must be 2-D (n_cells, ndim); got shape {lo.shape}.")
    if hi.shape != lo.shape:
        raise ValueError(f"cell_hi shape {hi.shape} must match cell_lo shape {lo.shape}.")
    if int(lo.shape[1]) < 1:
        raise ValueError(f"BVH ndim must be >= 1; got {int(lo.shape[1])}.")
    return np.ascontiguousarray(lo), np.ascontiguousarray(hi)


def _check_tree_structure(
    node_left: npt.NDArray[np.int64],
    node_right: npt.NDArray[np.int64],
    node_cell: npt.NDArray[np.int64],
    n_nodes: int,
    n_cells: int,
) -> None:
    """Check that the node arrays encode a tree the traversal kernels can walk.

    The kernels in :mod:`pantr.grid._bvh_core` decide "leaf or internal" from
    ``node_cell`` alone (``node_cell[i] < 0`` means internal) and then push
    ``node_left[i]`` and ``node_right[i]`` unconditionally, without testing either
    against ``-1``. So the two encodings of leafness have to agree, and an internal
    node's children have to be real indices: a child left at ``-1``, or any other
    negative value, wraps to the end of the node array rather than raising, and
    silently corrupts the query result.

    Args:
        node_left (npt.NDArray[np.int64]): Left-child indices; ``-1`` on leaves.
        node_right (npt.NDArray[np.int64]): Right-child indices; ``-1`` on leaves.
        node_cell (npt.NDArray[np.int64]): Leaf cell ids; ``-1`` on internal nodes.
        n_nodes (int): Number of nodes, i.e. the valid child-index range.
        n_cells (int): Number of indexed cells, i.e. the valid cell-id range.

    Raises:
        ValueError: If ``node_cell`` and the child pointers disagree about which
            nodes are leaves, if an internal node's child index lies outside
            ``[0, n_nodes)``, or if a leaf's cell id lies outside ``[0, n_cells)``.
    """
    is_leaf = node_cell >= 0
    if not np.array_equal(is_leaf, (node_left == -1) & (node_right == -1)):
        raise ValueError(
            "BVH: node_cell and the child pointers disagree about which nodes are "
            "leaves. A node is a leaf iff node_cell >= 0, and exactly then must "
            "node_left and node_right both be -1."
        )
    internal = ~is_leaf
    for arr, name in ((node_left, "node_left"), (node_right, "node_right")):
        child = arr[internal]
        if child.size > 0 and (int(child.min()) < 0 or int(child.max()) >= n_nodes):
            raise ValueError(
                f"BVH: {name} contains values outside [0, {n_nodes}) on internal nodes: "
                f"range is [{int(child.min())}, {int(child.max())}]."
            )
    leaf_cells = node_cell[is_leaf]
    if leaf_cells.size > 0 and int(leaf_cells.max()) >= n_cells:
        raise ValueError(
            f"BVH: node_cell contains values outside [0, {n_cells}) on leaves: "
            f"maximum is {int(leaf_cells.max())}."
        )


def _max_tree_depth(
    node_left: npt.NDArray[np.int64],
    node_right: npt.NDArray[np.int64],
    limit: int,
) -> int:
    """Measure a BVH node array's maximum root-to-leaf depth, capped at ``limit``.

    Walks from the root (node ``0``) with an explicit Python ``list`` as the
    traversal stack. That list has to be unbounded: the depth is precisely the
    unknown quantity being checked, so a fixed-size buffer would reproduce the very
    bug this function exists to catch.

    The walk stops as soon as it finds a path deeper than ``limit``, which is all the
    caller needs to know and which also makes it terminate on a node array that is
    not a tree. A cyclic child pointer would otherwise grow the stack without bound,
    and a validator that hangs is worse than the out-of-bounds write it replaces.

    Args:
        node_left (npt.NDArray[np.int64]): Left-child indices; ``-1`` on
            leaves. Shape ``(n_nodes,)``.
        node_right (npt.NDArray[np.int64]): Right-child indices; ``-1`` on
            leaves. Shape ``(n_nodes,)``.
        limit (int): Depth beyond which the exact value no longer matters.

    Returns:
        int: Number of nodes on the longest root-to-leaf path, the root counting as
        depth ``1``; ``0`` for empty node arrays. A returned value greater than
        ``limit`` means only "deeper than ``limit``", not the true maximum.
    """
    n_nodes = node_left.shape[0]
    if n_nodes == 0:
        return 0
    stack: list[tuple[int, int]] = [(0, 1)]
    max_depth = 0
    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth)
        if max_depth > limit:
            return max_depth
        left = int(node_left[node])
        right = int(node_right[node])
        if left != -1:
            stack.append((left, depth + 1))
        if right != -1:
            stack.append((right, depth + 1))
    return max_depth


class _BVHPython:
    """The pure-Python hierarchy: the backend under ``PANTR_BACKEND=python``.

    This was the public :class:`BVH` until the 2026-08-27 amendment to
    ``design/cross_backend_types.md`` made the hierarchy a C++-owned type. It is not
    merely a parity oracle: :func:`_impl_class` returns it whenever the Python
    backend is selected, which is how the package still works in a tree with no
    compiled extension.

    Its surface is deliberately the C++ type's, so that :class:`BVH` can call the
    same methods on either -- which is why :meth:`query_aabb` takes two corner
    arrays here and an :class:`~pantr.geometry.AABB` on the wrapper.

    Instances are immutable: the node arrays are flagged read-only and the three
    counts are read-only properties. **They used to be public writable attributes**,
    while the class docstring said instances were immutable; ``bvh.ndim = 7``
    defeated the dimension check in :meth:`query_aabb` and left the kernels indexing
    a mis-shaped array.

    Attributes:
        ndim (int): Spatial dimension of the indexed AABBs (``>= 1``).
        n_cells (int): Number of cells indexed (equal to the number of leaves).
        n_nodes (int): Total number of nodes (``2 * n_cells - 1``, else ``0``).
    """

    __slots__ = (
        "_n_cells",
        "_n_nodes",
        "_ndim",
        "_node_cell",
        "_node_hi",
        "_node_left",
        "_node_lo",
        "_node_right",
    )

    def __init__(  # noqa: PLR0913 -- BVH is a five-array flat struct
        self,
        node_lo: npt.NDArray[np.float64],
        node_hi: npt.NDArray[np.float64],
        node_left: npt.NDArray[np.int64],
        node_right: npt.NDArray[np.int64],
        node_cell: npt.NDArray[np.int64],
        n_cells: int,
    ) -> None:
        """Store the raw BVH arrays after validating their shapes.

        Args:
            node_lo (npt.NDArray[np.float64]): Per-node AABB lo corners, shape
                ``(n_nodes, ndim)``.
            node_hi (npt.NDArray[np.float64]): Per-node AABB hi corners, same shape.
            node_left (npt.NDArray[np.int64]): Left-child indices; ``-1`` on leaves.
            node_right (npt.NDArray[np.int64]): Right-child indices; ``-1`` on leaves.
            node_cell (npt.NDArray[np.int64]): Leaf cell identifiers; ``-1`` on
                internal nodes.
            n_cells (int): Number of indexed cells (leaves).

        Raises:
            TypeError: If any array has the wrong dtype.
            ValueError: If shapes are inconsistent, ``ndim`` is ``< 1``,
                ``n_nodes != 2 * n_cells - 1`` (``0`` when ``n_cells == 0``), if
                ``node_cell`` and the child pointers disagree about which nodes are
                leaves, if an internal node's children or a leaf's cell id are out of
                range, or if the tree's root-to-leaf depth exceeds the traversal
                kernels' stack depth (``_BVH_STACK_DEPTH``).
        """
        n_nodes, ndim = _validate_node_array_types(
            node_lo, node_hi, node_left, node_right, node_cell
        )
        n_cells_int = int(n_cells)
        expected_nodes = 2 * n_cells_int - 1 if n_cells_int > 0 else 0
        if n_nodes != expected_nodes:
            raise ValueError(
                f"BVH: n_cells={n_cells_int} implies n_nodes={expected_nodes}; "
                f"got node arrays with {n_nodes} rows."
            )
        # Runs before the depth walk below, which indexes the children itself.
        _check_tree_structure(node_left, node_right, node_cell, n_nodes, n_cells_int)

        # Guard the fixed-depth traversal stack in :mod:`pantr.grid._bvh_core`, whose
        # kernels push unconditionally.  Unlike from_cell_bounds -- whose median split
        # keeps the tree balanced, so depth follows from n_cells -- this constructor
        # takes arbitrary node arrays, and an unbalanced one overflows that stack with
        # an out-of-bounds write on the first query.  A depth-d traversal occupies at
        # most d slots, so depth == _BVH_STACK_DEPTH still fits.
        tree_depth = _max_tree_depth(node_left, node_right, _BVH_STACK_DEPTH)
        if tree_depth > _BVH_STACK_DEPTH:
            raise ValueError(
                f"BVH: the given node arrays encode a tree of depth {tree_depth} or more, "
                f"exceeding the stack depth {_BVH_STACK_DEPTH} that the traversal kernels "
                f"allow. Build the tree more evenly, or use BVH.from_cell_bounds, whose "
                f"median split keeps the depth logarithmic in the cell count."
            )
        self._node_lo = np.ascontiguousarray(node_lo, dtype=np.float64)
        self._node_hi = np.ascontiguousarray(node_hi, dtype=np.float64)
        self._node_left = np.ascontiguousarray(node_left, dtype=np.int64)
        self._node_right = np.ascontiguousarray(node_right, dtype=np.int64)
        self._node_cell = np.ascontiguousarray(node_cell, dtype=np.int64)
        for arr_ro in (
            self._node_lo,
            self._node_hi,
            self._node_left,
            self._node_right,
            self._node_cell,
        ):
            arr_ro.flags.writeable = False
        self._ndim = ndim
        self._n_cells = n_cells_int
        self._n_nodes = n_nodes

    @classmethod
    def from_cell_bounds(
        cls,
        cell_lo: npt.NDArray[np.float64],
        cell_hi: npt.NDArray[np.float64],
    ) -> Self:
        """Build a BVH over ``n_cells`` axis-aligned cell AABBs.

        Uses a top-down median-of-longest-axis split. Cells are sorted by centroid
        on the longest axis; the median splits the list into two halves of equal
        size (``+/- 1``). Each leaf indexes exactly one cell.

        Args:
            cell_lo (npt.NDArray[np.float64]): Per-cell lo corners, shape
                ``(n_cells, ndim)``, already ``float64`` and shape-checked.
            cell_hi (npt.NDArray[np.float64]): Per-cell hi corners, same shape.

        Returns:
            Self: The constructed hierarchy.

        Raises:
            ValueError: If any cell corner is non-finite, if some cell has
                ``hi < lo``, or if the implied tree exceeds the internal stack depth.
        """
        n_cells, ndim = int(cell_lo.shape[0]), int(cell_lo.shape[1])
        if not np.all(np.isfinite(cell_lo)) or not np.all(np.isfinite(cell_hi)):
            raise ValueError(
                "BVH.from_cell_bounds: cell_lo and cell_hi must contain only finite "
                "values; got NaN or Inf."
            )
        if np.any(cell_hi < cell_lo):
            raise ValueError(
                "Every cell must satisfy cell_hi >= cell_lo on every axis; "
                "at least one cell violates this."
            )
        if n_cells == 0:
            empty_lo = np.zeros((0, ndim), dtype=np.float64)
            empty_hi = np.zeros((0, ndim), dtype=np.float64)
            empty_i = np.zeros(0, dtype=np.int64)
            return cls(empty_lo, empty_hi, empty_i, empty_i, empty_i, 0)
        # Guard against the fixed-depth Numba stack in :mod:`pantr.grid._bvh_core`.
        # Median-of-longest-axis splits produce a balanced tree of height
        # ``ceil(log2(n_cells)) + 1``; the ``+ 1`` accounts for the root push.
        #
        # This keeps the pre-port float64 form on purpose, where the C++ side uses
        # `bit_width(n - 1) + 1` -- the same quantity in exact integers. The two
        # disagree only above a cell count needing petabytes for one coordinate
        # axis, which `scripts/measure_bvh_depth_arithmetic.py` enumerates; below
        # that they are equal, so the backends cannot differ on any reachable input
        # and this line stays the oracle's.
        max_depth = int(np.ceil(np.log2(n_cells))) + 1 if n_cells > 1 else 1
        if max_depth > _BVH_STACK_DEPTH:
            raise ValueError(
                f"BVH.from_cell_bounds: {n_cells} cells would produce a tree of depth "
                f">= {max_depth}, exceeding the internal stack depth {_BVH_STACK_DEPTH}. "
                f"This is a library limit; please report this as an issue."
            )
        max_nodes = 2 * n_cells - 1
        node_lo = np.empty((max_nodes, ndim), dtype=np.float64)
        node_hi = np.empty((max_nodes, ndim), dtype=np.float64)
        node_left = np.full(max_nodes, -1, dtype=np.int64)
        node_right = np.full(max_nodes, -1, dtype=np.int64)
        node_cell = np.full(max_nodes, -1, dtype=np.int64)
        bvh_build_kernel()(cell_lo, cell_hi, node_lo, node_hi, node_left, node_right, node_cell)
        return cls(node_lo, node_hi, node_left, node_right, node_cell, n_cells)

    def query_aabb(
        self, qlo: npt.NDArray[np.float64], qhi: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.int64]:
        """Return the ids of every leaf cell the query box is not separated from.

        Takes two corner arrays rather than an :class:`~pantr.geometry.AABB`, so
        that this class and the C++ type present one surface to :class:`BVH`.

        Args:
            qlo (npt.NDArray[np.float64]): Query box lo corner, length ``ndim``.
            qhi (npt.NDArray[np.float64]): Query box hi corner, same length.

        Returns:
            npt.NDArray[np.int64]: Matching cell ids, in the traversal's own order.

        Raises:
            ValueError: If the corners do not have length :attr:`ndim`.
            RuntimeError: If the count and emit passes disagree.
        """
        if qlo.shape[0] != self._ndim:
            raise ValueError(
                f"BVH.query_aabb: aabb.ndim ({qlo.shape[0]}) must match self.ndim "
                f"({self._ndim})."
            )
        if self._n_cells == 0:
            return np.zeros(0, dtype=np.int64)
        count = int(
            bvh_query_count_kernel()(
                qlo,
                qhi,
                self._node_lo,
                self._node_hi,
                self._node_left,
                self._node_right,
                self._node_cell,
            )
        )
        out = np.empty(count, dtype=np.int64)
        if count == 0:
            return out
        written = int(
            bvh_query_emit_kernel()(
                qlo,
                qhi,
                self._node_lo,
                self._node_hi,
                self._node_left,
                self._node_right,
                self._node_cell,
                out,
            )
        )
        if written != count:
            raise RuntimeError(
                f"BVH.query_aabb: internal count/emit mismatch (count pass returned {count}, "
                f"emit pass wrote {written}). This is a bug in the BVH kernel; please report it."
            )
        return out

    @property
    def ndim(self) -> int:
        """Get the spatial dimension of the indexed AABBs.

        Returns:
            int: The number of axes (``>= 1``).
        """
        return self._ndim

    @property
    def n_cells(self) -> int:
        """Get the number of cells indexed.

        Returns:
            int: Equal to the number of leaves.
        """
        return self._n_cells

    @property
    def n_nodes(self) -> int:
        """Get the total number of nodes.

        Returns:
            int: ``2 * n_cells - 1`` for ``n_cells > 0``, else ``0``.
        """
        return self._n_nodes

    @property
    def node_lo(self) -> npt.NDArray[np.float64]:
        """Get the read-only view of per-node AABB lo corners.

        Returns:
            npt.NDArray[np.float64]: Shape ``(n_nodes, ndim)``.
        """
        return self._node_lo

    @property
    def node_hi(self) -> npt.NDArray[np.float64]:
        """Get the read-only view of per-node AABB hi corners.

        Returns:
            npt.NDArray[np.float64]: Shape ``(n_nodes, ndim)``.
        """
        return self._node_hi

    @property
    def node_left(self) -> npt.NDArray[np.int64]:
        """Get the read-only view of per-node left-child indices.

        Returns:
            npt.NDArray[np.int64]: Shape ``(n_nodes,)``; ``-1`` on leaves.
        """
        return self._node_left

    @property
    def node_right(self) -> npt.NDArray[np.int64]:
        """Get the read-only view of per-node right-child indices.

        Returns:
            npt.NDArray[np.int64]: Shape ``(n_nodes,)``; ``-1`` on leaves.
        """
        return self._node_right

    @property
    def node_cell(self) -> npt.NDArray[np.int64]:
        """Get the read-only view of per-leaf cell identifiers.

        Returns:
            npt.NDArray[np.int64]: Shape ``(n_nodes,)``; ``-1`` on internal
            nodes, ``0 <= id < n_cells`` on leaves.
        """
        return self._node_cell


def _impl_class() -> type[_BVHPython] | type[_CppBVH]:
    """The implementation class the active backend selects.

    Returns:
        type[_BVHPython] | type[_CppBVH]: :class:`_BVHPython` under the Python
        backend, the bound C++ class otherwise.

    Raises:
        RuntimeError: If the C++ backend is requested and is not available.
    """
    if _python_backend_selected():
        return _BVHPython
    from pantr import _pantr_cpp  # noqa: PLC0415  (optional, imported only when selected)

    return _pantr_cpp.BVH


def _capacity_error() -> type[BaseException]:
    """The exception the C++ type raises for a tree its traversal stack cannot hold.

    Imported lazily and only on the C++ path, so a tree with no compiled extension
    never reaches it.

    Returns:
        type[BaseException]: ``pantr._pantr_cpp.CapacityError``.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (optional, imported only when selected)

    return _pantr_cpp.CapacityError


def _rebuild_bvh(  # noqa: PLR0913 -- BVH is a five-array flat struct
    node_lo: npt.NDArray[np.float64],
    node_hi: npt.NDArray[np.float64],
    node_left: npt.NDArray[np.int64],
    node_right: npt.NDArray[np.int64],
    node_cell: npt.NDArray[np.int64],
    n_cells: int,
) -> BVH:
    """Rebuild a :class:`BVH` from its five arrays and its cell count.

    The reconstruction :meth:`BVH.__reduce__` names. A module-level function rather
    than the class itself, because ``n_cells`` is keyword-only on the constructor
    and a ``__reduce__`` argument tuple is positional.

    Args:
        node_lo (npt.NDArray[np.float64]): Per-node AABB lo corners.
        node_hi (npt.NDArray[np.float64]): Per-node AABB hi corners.
        node_left (npt.NDArray[np.int64]): Left-child indices.
        node_right (npt.NDArray[np.int64]): Right-child indices.
        node_cell (npt.NDArray[np.int64]): Leaf cell identifiers.
        n_cells (int): Number of indexed cells.

    Returns:
        BVH: The rebuilt hierarchy, holding whichever implementation the *reading*
        process's backend selects.
    """
    return BVH(node_lo, node_hi, node_left, node_right, node_cell, n_cells=n_cells)


class BVH:
    """Bounding-volume hierarchy indexing a fixed collection of AABBs.

    Instances are immutable: the node arrays are read-only and the three counts are
    read-only properties. Queries allocate fresh ``int64`` output arrays per call.

    Build by passing per-cell AABBs to :meth:`from_cell_bounds`; direct
    construction from the raw array representation is supported via the default
    constructor but is mostly intended for tests and round-trip serialization.

    **This class is a wrapper**, holding a hierarchy owned by the C++ core (or, under
    ``PANTR_BACKEND=python``, a :class:`_BVHPython`). See the module docstring for
    the ownership rule, for where validation lives, and for what the overlap
    predicate actually tests.

    Attributes:
        ndim (int): Spatial dimension of the indexed AABBs (``>= 1``).
        n_cells (int): Number of cells indexed (equal to the number of leaves).
        n_nodes (int): Total number of nodes (``2 * n_cells - 1``, else ``0``).
    """

    __slots__ = ("_impl",)

    _impl: _Impl
    """The implementation this wrapper holds; see :func:`_impl_class`."""

    def __init__(  # noqa: PLR0913 -- BVH is a five-array flat struct
        self,
        node_lo: npt.NDArray[np.float64],
        node_hi: npt.NDArray[np.float64],
        node_left: npt.NDArray[np.int64],
        node_right: npt.NDArray[np.int64],
        node_cell: npt.NDArray[np.int64],
        *,
        n_cells: int,
    ) -> None:
        """Store the raw BVH arrays after validating their shapes.

        Callers should prefer :meth:`from_cell_bounds`; this constructor is
        useful for tests that need to poke specific tree shapes.

        Args:
            node_lo (npt.NDArray[np.float64]): Per-node AABB lo corners, shape
                ``(n_nodes, ndim)``.
            node_hi (npt.NDArray[np.float64]): Per-node AABB hi corners, shape
                ``(n_nodes, ndim)``.
            node_left (npt.NDArray[np.int64]): Left-child indices; ``-1`` on
                leaves. Shape ``(n_nodes,)``.
            node_right (npt.NDArray[np.int64]): Right-child indices; ``-1`` on
                leaves.
            node_cell (npt.NDArray[np.int64]): Leaf cell identifiers; ``-1`` on
                internal nodes.
            n_cells (int): Number of indexed cells (leaves).

        Raises:
            TypeError: If any array has the wrong dtype.
            ValueError: If shapes are inconsistent, ``ndim`` is ``< 1``,
                ``n_nodes != 2 * n_cells - 1`` (``0`` when ``n_cells == 0``), if
                ``node_cell`` and the child pointers disagree about which nodes are
                leaves, if an internal node's children or a leaf's cell id are out of
                range, or if the tree's root-to-leaf depth exceeds the traversal
                kernels' stack depth (``_BVH_STACK_DEPTH``).
        """
        _validate_node_array_types(node_lo, node_hi, node_left, node_right, node_cell)
        cls = _impl_class()
        if cls is _BVHPython:
            impl: _Impl = _BVHPython(
                node_lo, node_hi, node_left, node_right, node_cell, int(n_cells)
            )
        else:
            try:
                impl = cls(
                    np.ascontiguousarray(node_lo),
                    np.ascontiguousarray(node_hi),
                    np.ascontiguousarray(node_left),
                    np.ascontiguousarray(node_right),
                    np.ascontiguousarray(node_cell),
                    int(n_cells),
                )
            except _capacity_error() as exc:
                # The C++ type reports the traversal-stack limit as CapacityError,
                # which is right for a C++ caller: the tree is well formed and this
                # build cannot walk it. The pre-port class raised ValueError with
                # this same text, and PANTR_BACKEND must not decide which exception
                # a caller catches, so the wrapper presents the class it always did.
                # A seam: it lives exactly as long as the Python backend does.
                raise ValueError(str(exc)) from exc
        object.__setattr__(self, "_impl", impl)

    def __setattr__(self, name: str, value: object) -> None:
        """Reject post-construction attribute writes.

        ``ndim``, ``n_cells`` and ``n_nodes`` were public writable attributes before
        the port, while the class docstring said instances were immutable.
        ``bvh.ndim = 7`` defeated the dimension check in :meth:`query_aabb`, and the
        kernels then indexed a mis-shaped array.

        Args:
            name (str): The attribute being set.
            value (object): The value it would take.

        Raises:
            AttributeError: Always -- :class:`BVH` is immutable.
        """
        raise AttributeError(f"BVH is immutable; cannot set attribute {name!r}.")

    def __delattr__(self, name: str) -> None:
        """Reject attribute deletion.

        Args:
            name (str): The attribute being deleted.

        Raises:
            AttributeError: Always -- :class:`BVH` is immutable.
        """
        raise AttributeError(f"BVH is immutable; cannot delete attribute {name!r}.")

    @classmethod
    def _wrap(cls, impl: _Impl) -> BVH:
        """Wrap an implementation object that is already valid.

        Args:
            impl (_Impl): The implementation object to adopt.

        Returns:
            BVH: A wrapper around ``impl``, with no re-validation.
        """
        self = object.__new__(cls)
        object.__setattr__(self, "_impl", impl)
        return self

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Pickle by the five node arrays, never by the implementation.

        The C++ handle is not picklable and must not become part of the wire format:
        a pickle written under one backend has to load under the other, or the
        backend switch would silently become a data-format switch.

        Returns:
            tuple: :func:`_rebuild_bvh` and the arrays to rebuild from.
        """
        return (
            _rebuild_bvh,
            (
                self.node_lo,
                self.node_hi,
                self.node_left,
                self.node_right,
                self.node_cell,
                self.n_cells,
            ),
        )

    def __repr__(self) -> str:
        """Return a compact representation naming the three counts.

        Formatted here rather than by the implementation, so that the two backends
        print identically. The node arrays are left out: there are five of them and
        ``2 * n_cells - 1`` rows in each.

        Returns:
            str: ``"BVH(n_cells=..., n_nodes=..., ndim=...)"``.
        """
        return f"BVH(n_cells={self.n_cells}, n_nodes={self.n_nodes}, ndim={self.ndim})"

    @staticmethod
    def from_cell_bounds(cell_lo: npt.ArrayLike, cell_hi: npt.ArrayLike) -> BVH:
        """Build a BVH over ``n_cells`` axis-aligned cell AABBs.

        Uses a top-down median-of-longest-axis split. Cells are sorted by centroid
        on the longest axis; the median splits the list into two halves of equal
        size (``+/- 1``). Each leaf indexes exactly one cell.

        Args:
            cell_lo (npt.ArrayLike): Per-cell lo corners; shape
                ``(n_cells, ndim)`` with ``ndim >= 1``. Validated, not mutated.
            cell_hi (npt.ArrayLike): Per-cell hi corners; same shape and
                conventions as ``cell_lo``. Each entry must satisfy
                ``cell_hi >= cell_lo``.

        Returns:
            BVH: The constructed hierarchy.

        Raises:
            TypeError: If inputs cannot be cast to ``float64``.
            ValueError: If shapes are inconsistent, ``ndim`` is ``< 1``, any cell
                has ``hi < lo``, or the implied tree exceeds the internal stack
                depth.
        """
        lo, hi = _validate_cell_bounds(cell_lo, cell_hi)
        cls = _impl_class()
        if cls is _BVHPython:
            return BVH._wrap(_BVHPython.from_cell_bounds(lo, hi))
        try:
            return BVH._wrap(cls.from_cell_bounds(lo, hi))
        except _capacity_error() as exc:
            # See BVH.__init__ for why the class is translated back.
            raise ValueError(str(exc)) from exc

    def query_aabb(self, aabb: AABB) -> npt.NDArray[np.int64]:
        """Return the ids of every leaf cell the query box is not separated from.

        The predicate is inclusive on every face and has **no emptiness branch**, so
        it is not :meth:`pantr.geometry.AABB.overlaps`; the module docstring gives
        the case where the two disagree.

        Args:
            aabb (AABB): Query box; must match :attr:`ndim`.

        Returns:
            npt.NDArray[np.int64]: Matching cell ids. Order matches the
            internal preorder traversal; callers that need a particular order
            should sort the result.

        Raises:
            ValueError: If ``aabb.ndim != self.ndim``.
        """
        qlo = np.ascontiguousarray(aabb.lo, dtype=np.float64)
        qhi = np.ascontiguousarray(aabb.hi, dtype=np.float64)
        return np.asarray(self._impl.query_aabb(qlo, qhi), dtype=np.int64)

    @property
    def ndim(self) -> int:
        """Get the spatial dimension of the indexed AABBs.

        Returns:
            int: The number of axes (``>= 1``).
        """
        return int(self._impl.ndim)

    @property
    def n_cells(self) -> int:
        """Get the number of cells indexed.

        Returns:
            int: Equal to the number of leaves.
        """
        return int(self._impl.n_cells)

    @property
    def n_nodes(self) -> int:
        """Get the total number of nodes.

        Returns:
            int: ``2 * n_cells - 1`` for ``n_cells > 0``, else ``0``.
        """
        return int(self._impl.n_nodes)

    @property
    def node_lo(self) -> npt.NDArray[np.float64]:
        """Get the read-only view of per-node AABB lo corners.

        Under the C++ backend this aliases the hierarchy's own storage rather than
        copying it, which is what the pre-port class also handed back, and it is
        read-only against accident rather than against malice -- ``ctypes`` clears
        the flag in two lines. ``bvh_tree.hpp`` records what that costs under issue
        #359: on that backend a corrupted index is undefined behaviour, where the
        numba kernel merely returns a defined wrong answer.

        Returns:
            npt.NDArray[np.float64]: Shape ``(n_nodes, ndim)``.
        """
        return self._impl.node_lo

    @property
    def node_hi(self) -> npt.NDArray[np.float64]:
        """Get the read-only view of per-node AABB hi corners.

        Returns:
            npt.NDArray[np.float64]: Shape ``(n_nodes, ndim)``. See
            :attr:`node_lo` for the aliasing contract.
        """
        return self._impl.node_hi

    @property
    def node_left(self) -> npt.NDArray[np.int64]:
        """Get the read-only view of per-node left-child indices.

        Returns:
            npt.NDArray[np.int64]: Shape ``(n_nodes,)``; ``-1`` on leaves. See
            :attr:`node_lo` for the aliasing contract.
        """
        return self._impl.node_left

    @property
    def node_right(self) -> npt.NDArray[np.int64]:
        """Get the read-only view of per-node right-child indices.

        Returns:
            npt.NDArray[np.int64]: Shape ``(n_nodes,)``; ``-1`` on leaves. See
            :attr:`node_lo` for the aliasing contract.
        """
        return self._impl.node_right

    @property
    def node_cell(self) -> npt.NDArray[np.int64]:
        """Get the read-only view of per-leaf cell identifiers.

        Returns:
            npt.NDArray[np.int64]: Shape ``(n_nodes,)``; ``-1`` on internal
            nodes, ``0 <= id < n_cells`` on leaves. See :attr:`node_lo` for the
            aliasing contract.
        """
        return self._impl.node_cell


__all__ = ["BVH"]
