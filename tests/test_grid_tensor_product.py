"""Tests for ``pantr.grid.TensorProductGrid`` and its factories."""

from __future__ import annotations

import numpy as np
import pytest

from pantr.geometry import AABB
from pantr.grid import GridRestriction, TensorProductGrid, tensor_product_grid, uniform_grid
from pantr.transform import AffineTransform


def test_construction_metadata() -> None:
    """Per-axis breakpoints set ndim, counts, bounds, and num_cells."""
    g = TensorProductGrid([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0]])
    assert g.ndim == 2
    assert g.cells_per_axis == (3, 2)
    assert g.num_cells == 6
    assert g.bounds.tolist() == [[0.0, 3.0], [0.0, 4.0]]
    assert [b.tolist() for b in g.breakpoints] == [[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0]]


def test_breakpoints_are_read_only() -> None:
    """Stored breakpoint and bounds arrays are read-only."""
    g = uniform_grid([[0.0, 1.0]], 4)
    assert not g.breakpoints[0].flags.writeable
    assert not g.bounds.flags.writeable


@pytest.mark.parametrize(
    ("cells_per_axis", "cid", "multi"),
    [
        ((3, 2), 0, (0, 0)),
        ((3, 2), 1, (0, 1)),  # last axis fastest (C-order)
        ((3, 2), 2, (1, 0)),
        ((3, 2), 5, (2, 1)),
        ((2, 2, 2), 1, (0, 0, 1)),
        ((2, 2, 2), 4, (1, 0, 0)),
    ],
)
def test_c_order_addressing(
    cells_per_axis: tuple[int, ...], cid: int, multi: tuple[int, ...]
) -> None:
    """Flat cell ids are row-major (C-order) over cells_per_axis."""
    g = uniform_grid([[0.0, float(n)] for n in cells_per_axis], list(cells_per_axis))
    assert g.cell_multi_index(cid) == multi
    assert g.flat_cell_index(multi) == cid


def test_addressing_round_trip() -> None:
    """Multi -> flat -> multi is the identity over all cells."""
    g = uniform_grid([[0.0, 3.0], [0.0, 4.0], [0.0, 2.0]], [3, 4, 2])
    for cid in range(g.num_cells):
        assert g.flat_cell_index(g.cell_multi_index(cid)) == cid


def test_cell_bounds() -> None:
    """cell_bounds returns the box corners of the addressed cell."""
    g = TensorProductGrid([[0.0, 1.0, 3.0], [0.0, 10.0]])  # cells_per_axis (2, 1)
    lo, hi = g.cell_bounds(g.flat_cell_index((1, 0)))
    assert lo.tolist() == [1.0, 0.0]
    assert hi.tolist() == [3.0, 10.0]


def test_cell_bounds_writeable_and_fresh() -> None:
    """cell_bounds returns fresh, writeable arrays each call."""
    g = uniform_grid([[0.0, 1.0]], 2)
    lo1, _ = g.cell_bounds(0)
    lo2, _ = g.cell_bounds(0)
    assert lo1 is not lo2
    assert lo1.flags.writeable


def test_locate_interior_and_outside() -> None:
    """Locate returns the containing cell or None when outside."""
    g = uniform_grid([[0.0, 3.0], [0.0, 2.0]], [3, 2])
    assert g.locate([0.5, 1.5]) == g.flat_cell_index((0, 1))
    assert g.locate([2.9, 0.1]) == g.flat_cell_index((2, 0))
    assert g.locate([10.0, 10.0]) is None
    assert g.locate([-1.0, 0.0]) is None


def test_locate_breakpoint_tie_goes_to_lower_cell() -> None:
    """A point on an interior breakpoint maps to the lower-indexed cell."""
    g = TensorProductGrid([[0.0, 1.0, 3.0, 6.0]])
    assert g.locate(1.0) == 0  # face between cell 0 and 1 -> cell 0
    assert g.locate(3.0) == 1
    assert g.locate(0.0) == 0  # left boundary -> cell 0
    assert g.locate(6.0) == 2


def test_locate_wrong_shape_raises() -> None:
    """Locate rejects a point of the wrong length."""
    g = uniform_grid([[0.0, 1.0], [0.0, 1.0]], 2)
    with pytest.raises(ValueError, match="shape"):
        g.locate([0.5, 0.5, 0.5])


def test_locate_many_matches_locate() -> None:
    """locate_many agrees with per-point locate, with -1 for outside."""
    g = uniform_grid([[0.0, 4.0], [0.0, 3.0]], [4, 3])
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1.0, 5.0, size=(50, 2))
    batch = g.locate_many(pts)
    for i, p in enumerate(pts):
        single = g.locate(p)
        assert batch[i] == (-1 if single is None else single)


def test_locate_many_single_point() -> None:
    """locate_many accepts a single point and returns a length-1 array."""
    g = uniform_grid([[0.0, 2.0], [0.0, 2.0]], 2)
    out = g.locate_many([0.5, 1.5])
    assert out.shape == (1,)
    assert out[0] == g.locate([0.5, 1.5])


def test_neighbors_interior_and_corner() -> None:
    """Interior cells have 2*ndim neighbours; corner cells have ndim."""
    g = uniform_grid([[0.0, 3.0], [0.0, 3.0]], 3)
    center = g.flat_cell_index((1, 1))
    assert len(g.neighbors(center)) == 4
    corner = g.flat_cell_index((0, 0))
    assert len(g.neighbors(corner)) == 2


def test_neighbor_across_facet() -> None:
    """neighbor_across_facet steps one cell along the facet's axis/side."""
    g = uniform_grid([[0.0, 3.0], [0.0, 3.0]], 3)
    cid = g.flat_cell_index((1, 1))
    # lfid = 2*axis + side. axis 0 low -> (0,1); axis 0 high -> (2,1).
    assert g.neighbor_across_facet(cid, 0) == g.flat_cell_index((0, 1))
    assert g.neighbor_across_facet(cid, 1) == g.flat_cell_index((2, 1))
    assert g.neighbor_across_facet(cid, 2) == g.flat_cell_index((1, 0))
    assert g.neighbor_across_facet(cid, 3) == g.flat_cell_index((1, 2))


def test_neighbor_across_boundary_facet_is_none() -> None:
    """A boundary facet has no neighbour."""
    g = uniform_grid([[0.0, 3.0], [0.0, 3.0]], 3)
    corner = g.flat_cell_index((0, 0))
    assert g.neighbor_across_facet(corner, 0) is None  # axis 0 low
    assert g.neighbor_across_facet(corner, 2) is None  # axis 1 low
    assert g.is_mesh_boundary_facet(corner, 0)
    assert not g.is_mesh_boundary_facet(corner, 1)


def test_facet_accessors() -> None:
    """Facet count, axis/side decoding, and degenerate facet bounds."""
    g = uniform_grid([[0.0, 2.0], [0.0, 2.0]], 2)
    cid = g.flat_cell_index((0, 0))
    assert g.num_local_facets(cid) == 4
    assert g.local_facet_axis_side(cid, 0) == (0, 0)
    assert g.local_facet_axis_side(cid, 3) == (1, 1)
    lo, hi = g.local_facet_bounds(cid, 1)  # axis 0, high face
    assert lo[0] == hi[0] == 1.0
    assert (lo[1], hi[1]) == (0.0, 1.0)


def test_reference_map() -> None:
    """reference_map sends the unit cube to the cell box."""
    g = TensorProductGrid([[0.0, 2.0, 5.0], [0.0, 4.0]])
    cid = g.flat_cell_index((1, 0))  # cell [2,5] x [0,4]
    rm = g.reference_map(cid)
    assert isinstance(rm, AffineTransform)
    corner = rm(np.array([[0.0, 0.0], [1.0, 1.0]]))
    np.testing.assert_allclose(corner, [[2.0, 0.0], [5.0, 4.0]])


def test_cell_level_and_children() -> None:
    """A flat grid reports level 0 and no children."""
    g = uniform_grid([[0.0, 2.0]], 2)
    assert g.cell_level(0) == 0
    assert g.child_cells(0) == ()


def test_query_aabb_builds_lazy_bvh() -> None:
    """query_aabb builds the BVH lazily on first use and caches it."""
    g = uniform_grid([[0.0, 3.0], [0.0, 3.0]], 3)
    assert g._bvh is None  # not built at construction
    # cells (0,0),(0,1),(1,0),(1,1) touch the box [0,1]^2.
    expected = sorted(g.flat_cell_index(m) for m in [(0, 0), (0, 1), (1, 0), (1, 1)])
    result = g.query_aabb(AABB([0.0, 0.0], [1.0, 1.0]))
    assert sorted(int(c) for c in result) == expected
    assert g.cell_bvh() is g.cell_bvh()  # built once, then cached


def test_is_uniform_flag() -> None:
    """is_uniform reflects per-axis spacing regularity."""
    assert uniform_grid([[0.0, 4.0]], 4).is_uniform
    assert not TensorProductGrid([[0.0, 1.0, 3.0]]).is_uniform


def test_cell_aabb() -> None:
    """cell_aabb wraps cell_bounds as an AABB."""
    g = uniform_grid([[0.0, 2.0], [0.0, 2.0]], 2)
    box = g.cell_aabb(0)
    assert isinstance(box, AABB)
    assert box.lo.tolist() == [0.0, 0.0]
    assert box.hi.tolist() == [1.0, 1.0]


def test_out_of_range_cid_raises() -> None:
    """Out-of-range cell ids raise IndexError."""
    g = uniform_grid([[0.0, 2.0]], 2)
    with pytest.raises(IndexError):
        g.cell_bounds(5)
    with pytest.raises(IndexError):
        g.cell_multi_index(-1)


def test_invalid_construction() -> None:
    """Degenerate or non-monotone breakpoints are rejected."""
    with pytest.raises(ValueError, match="at least"):
        TensorProductGrid([[0.0]])
    with pytest.raises(ValueError, match="strictly increasing"):
        TensorProductGrid([[0.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        TensorProductGrid([[0.0, np.inf]])
    with pytest.raises(ValueError, match="at least one axis"):
        TensorProductGrid([])


def test_repr() -> None:
    """Repr reports ndim, cell counts, and uniformity."""
    g = uniform_grid([[0.0, 2.0], [0.0, 2.0]], 2)
    text = repr(g)
    assert "TensorProductGrid" in text
    assert "cells_per_axis=(2, 2)" in text


# ---------------------------------------------------------------- factories


def test_uniform_grid_scalar_and_sequence_cells() -> None:
    """uniform_grid broadcasts a scalar count and accepts a per-axis sequence."""
    g1 = uniform_grid([[0.0, 2.0], [0.0, 4.0]], 2)
    assert g1.cells_per_axis == (2, 2)
    g2 = uniform_grid([[0.0, 2.0], [0.0, 4.0]], [2, 4])
    assert g2.cells_per_axis == (2, 4)
    np.testing.assert_allclose(g2.breakpoints[1], [0.0, 1.0, 2.0, 3.0, 4.0])


def test_uniform_grid_validation() -> None:
    """uniform_grid rejects bad bounds, lo>=hi, wrong cell length, count<1."""
    with pytest.raises(ValueError, match="shape"):
        uniform_grid([0.0, 1.0], 2)
    with pytest.raises(ValueError, match="lo < hi"):
        uniform_grid([[1.0, 0.0]], 2)
    with pytest.raises(ValueError, match="length"):
        uniform_grid([[0.0, 1.0], [0.0, 1.0]], [2, 2, 2])
    with pytest.raises(ValueError, match=">= 1"):
        uniform_grid([[0.0, 1.0]], 0)


def test_tensor_product_grid_from_space() -> None:
    """tensor_product_grid uses a space's unique in-domain knots as breakpoints."""
    from pantr.bspline import BsplineSpace, BsplineSpace1D  # noqa: PLC0415

    sp = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)  # 2 intervals on [0, 2]
    space = BsplineSpace([sp, sp])
    g = tensor_product_grid(space)
    assert g.cells_per_axis == space.num_intervals
    assert g.ndim == 2
    np.testing.assert_allclose(g.bounds, [[0.0, 2.0], [0.0, 2.0]], atol=1e-9)


def test_tensor_product_grid_matches_extraction_ids() -> None:
    """Grid cell ids agree with SpanwiseElementExtraction's flat ordering."""
    from pantr.bspline import (  # noqa: PLC0415
        BsplineSpace,
        BsplineSpace1D,
        SpanwiseElementExtraction,
    )

    sp_x = BsplineSpace1D([0, 0, 0, 1, 2, 3, 3, 3], 2)  # 3 intervals
    sp_y = BsplineSpace1D([0, 0, 1, 2, 2], 1)  # 2 intervals
    space = BsplineSpace([sp_x, sp_y])
    g = tensor_product_grid(space)
    ext = SpanwiseElementExtraction(space, "bezier")
    assert g.cells_per_axis == ext.num_intervals
    assert g.num_cells == ext.num_total_intervals


def test_tensor_product_grid_periodic_rejected() -> None:
    """A periodic direction cannot become a bounded grid."""
    from pantr.bspline import (  # noqa: PLC0415
        BsplineSpace,
        BsplineSpace1D,
        create_uniform_periodic_knots,
    )

    knots = create_uniform_periodic_knots(num_intervals=4, degree=2)
    sp_per = BsplineSpace1D(knots, 2, periodic=True)
    space = BsplineSpace([sp_per])
    with pytest.raises(ValueError, match="periodic"):
        tensor_product_grid(space)


def test_locate_many_1d() -> None:
    """locate_many on a 1-D grid matches scalar locate, including boundaries."""
    g = uniform_grid([[0.0, 4.0]], 4)
    pts = np.array([[-0.5], [0.0], [0.999], [1.0], [2.5], [3.999], [4.0], [5.0]])
    result = g.locate_many(pts)
    for i, p in enumerate(pts[:, 0]):
        single = g.locate([p])
        assert result[i] == (-1 if single is None else single)


def test_construction_rejects_complex_breakpoints() -> None:
    """Complex-valued breakpoints are rejected with TypeError."""
    with pytest.raises(TypeError, match="numeric"):
        TensorProductGrid([np.array([0 + 0j, 1 + 0j, 2 + 0j])])


def test_construction_rejects_bool_breakpoints() -> None:
    """Boolean breakpoints are rejected with TypeError."""
    with pytest.raises(TypeError, match="numeric"):
        TensorProductGrid([np.array([False, True, True])])


def test_flat_cell_index_wrong_ndim_raises() -> None:
    """flat_cell_index raises ValueError when multi-index has wrong length."""
    g = uniform_grid([[0.0, 3.0], [0.0, 2.0]], [3, 2])
    with pytest.raises(ValueError, match="length"):
        g.flat_cell_index([0])


def test_flat_cell_index_out_of_range_raises() -> None:
    """flat_cell_index raises IndexError when a per-axis index is out of range."""
    g = uniform_grid([[0.0, 3.0], [0.0, 2.0]], [3, 2])
    with pytest.raises(IndexError):
        g.flat_cell_index([3, 0])  # axis 0 index 3 >= cells_per_axis[0]=3


def test_locate_many_interior_breakpoint_tie() -> None:
    """locate_many Numba kernel agrees with scalar locate at every interior breakpoint."""
    g = TensorProductGrid([[0.0, 1.0, 3.0, 6.0], [0.0, 2.0, 5.0]])
    # Interior breakpoints are at x=1.0, x=3.0 (axis 0) and y=2.0 (axis 1).
    breakpoints_2d = [[1.0, 0.5], [1.0, 2.0], [3.0, 0.5], [3.0, 2.0]]
    pts = np.array(breakpoints_2d, dtype=np.float64)
    batch = g.locate_many(pts)
    for i, pt in enumerate(breakpoints_2d):
        single = g.locate(pt)
        expected = -1 if single is None else single
        assert int(batch[i]) == expected, (
            f"locate_many disagreed with locate at breakpoint {pt}: "
            f"batch={batch[i]}, single={single}"
        )


def test_locate_many_3d() -> None:
    """locate_many works correctly for 3-D tensor-product grids."""
    g = uniform_grid([[0.0, 2.0], [0.0, 3.0], [0.0, 4.0]], [2, 3, 4])
    rng = np.random.default_rng(42)
    pts = rng.uniform(-0.5, 4.5, size=(60, 3))
    batch = g.locate_many(pts)
    for i, pt in enumerate(pts):
        single = g.locate(pt)
        assert int(batch[i]) == (-1 if single is None else single)


def test_query_aabb_3d() -> None:
    """query_aabb works correctly for 3-D tensor-product grids."""
    from pantr.geometry import AABB  # noqa: PLC0415

    g = uniform_grid([[0.0, 3.0], [0.0, 3.0], [0.0, 3.0]], 3)
    # Each axis has cells [0,1), [1,2), [2,3). A query box [0.5,1.5]^3
    # strictly overlaps only the 8 cells in the lower 2x2x2 block (indices 0
    # or 1 on every axis).
    box = AABB([0.5, 0.5, 0.5], [1.5, 1.5, 1.5])
    result = sorted(int(c) for c in g.query_aabb(box))
    expected = sorted(
        g.flat_cell_index(m)
        for m in [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    )
    assert result == expected


def test_uniform_grid_single_cell_per_axis() -> None:
    """uniform_grid with cells=1 produces a single-cell grid per axis."""
    g = uniform_grid([[0.0, 5.0], [1.0, 3.0]], 1)
    assert g.cells_per_axis == (1, 1)
    assert g.num_cells == 1
    assert g.locate([2.5, 2.0]) == 0
    assert g.locate([5.0, 3.0]) == 0  # right boundary → only cell
    assert g.locate([5.1, 2.0]) is None
    out = g.locate_many(np.array([[2.5, 2.0], [-1.0, 2.0]]))
    assert out.tolist() == [0, -1]


# ---------------------------------------------------------------- restrict


def _assert_window_matches_global(g: TensorProductGrid, r: GridRestriction) -> None:
    """Every sub-grid cell coincides with its global cell (bounds and locate)."""
    for k in range(r.grid.num_cells):
        gcid = int(r.local_to_global_cell[k])
        lo_s, hi_s = r.grid.cell_bounds(k)
        lo_g, hi_g = g.cell_bounds(gcid)
        np.testing.assert_allclose(lo_s, lo_g)
        np.testing.assert_allclose(hi_s, hi_g)
        center = 0.5 * (lo_s + hi_s)
        assert r.grid.locate(center) == k
        assert g.locate(center) == gcid


def test_restrict_convex_block_matches_global() -> None:
    """A contiguous block restricts to a sub-grid coinciding with the window."""
    g = uniform_grid([[0.0, 4.0], [0.0, 4.0]], 4)
    cell_ids = [g.flat_cell_index(m) for m in [(1, 1), (1, 2), (2, 1), (2, 2)]]
    r = g.restrict(cell_ids)
    assert isinstance(r, GridRestriction)
    assert isinstance(r.grid, TensorProductGrid)
    assert r.grid.cells_per_axis == (2, 2)
    np.testing.assert_array_equal(r.local_to_global_cell, [5, 6, 9, 10])
    assert bool(r.in_subset.all())
    _assert_window_matches_global(g, r)


def test_restrict_local_global_round_trip() -> None:
    """local_to_global composes with the bbox offset; in_subset recovers the request."""
    g = uniform_grid([[0.0, 5.0], [0.0, 4.0]], [5, 4])
    cell_ids = [g.flat_cell_index(m) for m in [(1, 1), (1, 2), (2, 1), (3, 2)]]
    r = g.restrict(cell_ids)  # bbox rows 1..3, cols 1..2 -> offset (1, 1)
    assert isinstance(r.grid, TensorProductGrid)
    for k in range(r.grid.num_cells):
        sm = r.grid.cell_multi_index(k)
        assert g.flat_cell_index((sm[0] + 1, sm[1] + 1)) == int(r.local_to_global_cell[k])
    assert {int(c) for c in r.local_to_global_cell[r.in_subset]} == set(cell_ids)


def test_restrict_non_convex_flags_fill_cells() -> None:
    """A non-convex request yields the full bbox with fill cells flagged False."""
    g = uniform_grid([[0.0, 3.0], [0.0, 3.0]], 3)
    corners = [g.flat_cell_index(m) for m in [(0, 0), (0, 2), (2, 0), (2, 2)]]
    r = g.restrict(corners)
    assert isinstance(r.grid, TensorProductGrid)
    assert r.grid.cells_per_axis == (3, 3)  # bbox spans the whole grid
    np.testing.assert_array_equal(r.local_to_global_cell, np.arange(9))
    expected_mask = np.zeros(9, dtype=bool)
    expected_mask[corners] = True
    np.testing.assert_array_equal(r.in_subset, expected_mask)
    _assert_window_matches_global(g, r)


def test_restrict_breakpoints_not_reclamped() -> None:
    """Sub-grid breakpoints are pure slices of the parent (never re-based)."""
    g = uniform_grid([[0.0, 4.0], [0.0, 4.0]], 4)  # breakpoints 0,1,2,3,4 per axis
    cell_ids = [g.flat_cell_index(m) for m in [(1, 1), (1, 2), (2, 1), (2, 2)]]
    sub = g.restrict(cell_ids).grid
    assert isinstance(sub, TensorProductGrid)
    np.testing.assert_allclose(sub.breakpoints[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(sub.breakpoints[1], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(sub.bounds, [[1.0, 3.0], [1.0, 3.0]])


def test_restrict_full_grid_is_identity() -> None:
    """Restricting to all cells reproduces the grid with an identity cell map."""
    g = TensorProductGrid([[0.0, 1.0, 3.0], [0.0, 2.0, 5.0, 6.0]])  # cells (2, 3)
    r = g.restrict(list(range(g.num_cells)))
    assert isinstance(r.grid, TensorProductGrid)
    assert r.grid.cells_per_axis == g.cells_per_axis
    for d in range(g.ndim):
        np.testing.assert_allclose(r.grid.breakpoints[d], g.breakpoints[d])
    np.testing.assert_array_equal(r.local_to_global_cell, np.arange(g.num_cells))
    assert bool(r.in_subset.all())


def test_restrict_single_cell() -> None:
    """Restricting one cell yields a single-cell grid mapping back to it."""
    g = uniform_grid([[0.0, 3.0], [0.0, 3.0]], 3)
    cid = g.flat_cell_index((1, 2))
    r = g.restrict([cid])
    assert isinstance(r.grid, TensorProductGrid)
    assert r.grid.cells_per_axis == (1, 1)
    assert int(r.local_to_global_cell[0]) == cid
    np.testing.assert_array_equal(r.in_subset, [True])
    lo_s, hi_s = r.grid.cell_bounds(0)
    lo_g, hi_g = g.cell_bounds(cid)
    np.testing.assert_allclose(lo_s, lo_g)
    np.testing.assert_allclose(hi_s, hi_g)


def test_restrict_1d() -> None:
    """Restriction works on a 1-D grid."""
    g = uniform_grid([[0.0, 6.0]], 6)  # cells 0..5
    r = g.restrict([2, 3, 4])
    assert isinstance(r.grid, TensorProductGrid)
    assert r.grid.cells_per_axis == (3,)
    np.testing.assert_allclose(r.grid.breakpoints[0], [2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(r.local_to_global_cell, [2, 3, 4])
    assert bool(r.in_subset.all())


def test_restrict_duplicates_ignored() -> None:
    """Duplicate cell ids do not change the result."""
    g = uniform_grid([[0.0, 4.0], [0.0, 4.0]], 4)
    ids = [g.flat_cell_index(m) for m in [(1, 1), (1, 2), (2, 1), (2, 2)]]
    r_dup = g.restrict(ids + ids)
    r = g.restrict(ids)
    np.testing.assert_array_equal(r_dup.local_to_global_cell, r.local_to_global_cell)
    np.testing.assert_array_equal(r_dup.in_subset, r.in_subset)


def test_restrict_empty_raises() -> None:
    """An empty cell_ids is rejected."""
    g = uniform_grid([[0.0, 2.0]], 2)
    with pytest.raises(ValueError, match="non-empty"):
        g.restrict([])


def test_restrict_out_of_range_raises() -> None:
    """Out-of-range cell ids raise IndexError."""
    g = uniform_grid([[0.0, 2.0]], 2)
    with pytest.raises(IndexError):
        g.restrict([0, 2])  # 2 == num_cells
    with pytest.raises(IndexError):
        g.restrict([-1, 0])


def test_restrict_non_integer_raises() -> None:
    """Non-integer cell ids raise TypeError."""
    g = uniform_grid([[0.0, 2.0]], 2)
    with pytest.raises(TypeError, match="integer"):
        g.restrict([0.0, 1.0])


def test_restrict_non_square_grid_exact_local_to_global() -> None:
    """On a non-square grid, local_to_global_cell uses the global strides."""
    g = uniform_grid([[0.0, 3.0], [0.0, 5.0]], [3, 5])  # 3x5, strides (5, 1)
    # rows 1-2, cols 1-3 -> local (0,0)..(1,2)
    cell_ids = [g.flat_cell_index(m) for m in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]]
    r = g.restrict(cell_ids)
    assert isinstance(r.grid, TensorProductGrid)
    assert r.grid.cells_per_axis == (2, 3)
    # global ids: row*5 + col for (1,1),(1,2),(1,3),(2,1),(2,2),(2,3) = 6,7,8,11,12,13
    np.testing.assert_array_equal(r.local_to_global_cell, [6, 7, 8, 11, 12, 13])
    assert bool(r.in_subset.all())
    _assert_window_matches_global(g, r)


def test_restrict_3d() -> None:
    """Restriction works on a 3-D grid."""
    g = uniform_grid([[0.0, 4.0], [0.0, 3.0], [0.0, 2.0]], [4, 3, 2])  # 4x3x2 = 24 cells
    cell_ids = [g.flat_cell_index(m) for m in [(1, 1, 0), (1, 1, 1), (2, 1, 0), (2, 1, 1)]]
    r = g.restrict(cell_ids)
    assert isinstance(r.grid, TensorProductGrid)
    assert r.grid.cells_per_axis == (2, 1, 2)
    assert bool(r.in_subset.all())
    _assert_window_matches_global(g, r)


def test_restrict_non_convex_non_square_grid() -> None:
    """Non-convex request in a non-square grid flags fill cells correctly."""
    g = uniform_grid([[0.0, 2.0], [0.0, 3.0]], [2, 3])  # 2x3, strides (3, 1)
    corners = [g.flat_cell_index((0, 0)), g.flat_cell_index((1, 2))]  # 0 and 5
    r = g.restrict(corners)
    assert isinstance(r.grid, TensorProductGrid)
    assert r.grid.cells_per_axis == (2, 3)
    np.testing.assert_array_equal(r.local_to_global_cell, np.arange(6))
    expected_mask = np.zeros(6, dtype=bool)
    expected_mask[[0, 5]] = True
    np.testing.assert_array_equal(r.in_subset, expected_mask)
    _assert_window_matches_global(g, r)


def test_restrict_non_uniform_breakpoints() -> None:
    """Breakpoint slicing is correct for non-uniform spacing."""
    g = TensorProductGrid([[0.0, 1.0, 3.0, 6.0, 10.0], [0.0, 5.0, 7.0]])  # 4x2 grid
    cell_ids = [g.flat_cell_index(m) for m in [(1, 0), (2, 0)]]
    r = g.restrict(cell_ids)
    assert isinstance(r.grid, TensorProductGrid)
    assert r.grid.cells_per_axis == (2, 1)
    np.testing.assert_allclose(r.grid.breakpoints[0], [1.0, 3.0, 6.0])
    np.testing.assert_allclose(r.grid.breakpoints[1], [0.0, 5.0])
    assert bool(r.in_subset.all())
    _assert_window_matches_global(g, r)


def test_restrict_result_arrays_are_read_only() -> None:
    """local_to_global_cell and in_subset are read-only."""
    g = uniform_grid([[0.0, 2.0]], 2)
    r = g.restrict([0, 1])
    with pytest.raises(ValueError, match="read-only"):
        r.local_to_global_cell[0] = 99
    with pytest.raises(ValueError, match="read-only"):
        r.in_subset[0] = False
