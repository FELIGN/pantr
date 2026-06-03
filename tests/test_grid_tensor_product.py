"""Tests for ``pantr.grid.TensorProductGrid`` and its factories."""

from __future__ import annotations

import numpy as np
import pytest

from pantr.geometry import AABB
from pantr.grid import TensorProductGrid, tensor_product_grid, uniform_grid
from pantr.transform import AffineTransform


def test_construction_metadata() -> None:
    """Per-axis breakpoints set ndim, counts, bounds, and num_cells."""
    g = TensorProductGrid([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0]])
    assert g.ndim == 2  # noqa: PLR2004
    assert g.cells_per_axis == (3, 2)
    assert g.num_cells == 6  # noqa: PLR2004
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
    assert g.locate(6.0) == 2  # noqa: PLR2004 -- right boundary -> last cell


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
    assert len(g.neighbors(center)) == 4  # noqa: PLR2004
    corner = g.flat_cell_index((0, 0))
    assert len(g.neighbors(corner)) == 2  # noqa: PLR2004


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
    assert g.num_local_facets(cid) == 4  # noqa: PLR2004
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
    assert g.ndim == 2  # noqa: PLR2004
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
