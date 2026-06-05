"""Tests for pantr.grid.HierarchicalGrid."""

from __future__ import annotations

import numpy as np
import numpy.testing as np_testing
import pytest

from pantr.geometry import AABB
from pantr.grid import HierarchicalGrid, hierarchical_grid, uniform_grid
from pantr.grid._hierarchical_grid import (
    _block_size,
    _in_block,
    _normalize_blocks,
    _peel,
    _rect_intersect,
    _try_merge,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────


def _grid_1d(n: int = 4, factor: int = 2) -> HierarchicalGrid:
    return hierarchical_grid(uniform_grid([[0.0, 1.0]], n), factor)


def _grid_2d(n: int = 4, factor: int = 2) -> HierarchicalGrid:
    return hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], n), factor)


# ──────────────────────────────────────────────────────────────────────────────
# Rectangle helpers
# ──────────────────────────────────────────────────────────────────────────────


class TestBlockHelpers:
    """Unit tests for the pure rectangle helper functions."""

    def test_block_size_1d(self) -> None:
        assert _block_size((2,), (5,)) == 3

    def test_block_size_2d(self) -> None:
        assert _block_size((0, 0), (3, 4)) == 12

    def test_in_block_inside(self) -> None:
        assert _in_block((1, 2), (0, 0), (3, 4))

    def test_in_block_on_lo(self) -> None:
        assert _in_block((0, 0), (0, 0), (2, 2))

    def test_in_block_on_hi_exclusive(self) -> None:
        assert not _in_block((2, 2), (0, 0), (2, 2))

    def test_rect_intersect_overlap(self) -> None:
        result = _rect_intersect((0, 0), (3, 3), (1, 1), (4, 4))
        assert result == ((1, 1), (3, 3))

    def test_rect_intersect_disjoint(self) -> None:
        assert _rect_intersect((0, 0), (2, 2), (3, 3), (5, 5)) is None

    def test_peel_2d_full_frame(self) -> None:
        slabs = _peel((0, 0), (5, 5), (1, 1), (4, 4))
        total = sum(_block_size(*s) for s in slabs)
        assert total == 25 - 9  # 5*5 minus 3*3

    def test_peel_1d(self) -> None:
        slabs = _peel((0,), (10,), (2,), (7,))
        assert sorted(slabs) == [((0,), (2,)), ((7,), (10,))]

    def test_peel_inner_equals_outer_empty(self) -> None:
        slabs = _peel((0, 0), (3, 3), (0, 0), (3, 3))
        assert slabs == []

    def test_try_merge_adjacent_1d(self) -> None:
        result = _try_merge((0,), (3,), (3,), (5,))
        assert result == ((0,), (5,))

    def test_try_merge_non_adjacent(self) -> None:
        assert _try_merge((0,), (2,), (3,), (5,)) is None

    def test_try_merge_misaligned_2d(self) -> None:
        assert _try_merge((0, 0), (2, 2), (2, 1), (4, 3)) is None

    def test_normalize_merges_adjacent(self) -> None:
        blocks: list[tuple[tuple[int, ...], tuple[int, ...]]] = [((0,), (3,)), ((3,), (7,))]
        assert _normalize_blocks(blocks) == [((0,), (7,))]


# ──────────────────────────────────────────────────────────────────────────────
# Construction
# ──────────────────────────────────────────────────────────────────────────────


class TestHierarchicalGridConstruction:
    """Tests for HierarchicalGrid construction and validation."""

    def test_1d_initial_state(self) -> None:
        g = _grid_1d(4, 2)
        assert g.ndim == 1
        assert g.num_cells == 4
        assert g.max_level == 0
        assert g.factor == (2,)

    def test_2d_initial_state(self) -> None:
        g = _grid_2d(3, 3)
        assert g.ndim == 2
        assert g.num_cells == 9
        assert g.factor == (3, 3)

    def test_3d_initial_state(self) -> None:
        root = uniform_grid([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], 2)
        g = hierarchical_grid(root, 2)
        assert g.ndim == 3
        assert g.num_cells == 8
        assert g.factor == (2, 2, 2)

    def test_scalar_factor_broadcast(self) -> None:
        root = uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4)
        g = hierarchical_grid(root, 3)
        assert g.factor == (3, 3)

    def test_anisotropic_factor(self) -> None:
        root = uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4)
        g = hierarchical_grid(root, [2, 3])
        assert g.factor == (2, 3)

    def test_factor_of_one_allowed(self) -> None:
        root = uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4)
        g = hierarchical_grid(root, [1, 2])
        assert g.factor == (1, 2)

    def test_invalid_factor_zero_raises(self) -> None:
        root = uniform_grid([[0.0, 1.0]], 4)
        with pytest.raises(ValueError, match="factor"):
            hierarchical_grid(root, 0)

    def test_invalid_factor_length_raises(self) -> None:
        root = uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4)
        with pytest.raises(ValueError, match="length"):
            hierarchical_grid(root, [2, 2, 2])

    def test_non_tensor_product_root_raises(self) -> None:
        with pytest.raises(TypeError, match="TensorProductGrid"):
            HierarchicalGrid("not a grid", 2)  # type: ignore[arg-type]

    def test_factory_function(self) -> None:
        root = uniform_grid([[0.0, 2.0]], 6)
        g = hierarchical_grid(root, 2)
        assert isinstance(g, HierarchicalGrid)
        assert g.root is root

    def test_repr(self) -> None:
        g = _grid_2d(3, 2)
        r = repr(g)
        assert "HierarchicalGrid" in r
        assert "ndim=2" in r
        assert "factor=(2, 2)" in r


# ──────────────────────────────────────────────────────────────────────────────
# Initial cell properties
# ──────────────────────────────────────────────────────────────────────────────


class TestHierarchicalGridInitialCells:
    """Tests for cell properties on an unrefined grid."""

    def test_cell_bounds_tile_domain_1d(self) -> None:
        g = _grid_1d(5, 2)
        all_lo = sorted(float(g.cell_bounds(cid)[0][0]) for cid in range(g.num_cells))
        all_hi = sorted(float(g.cell_bounds(cid)[1][0]) for cid in range(g.num_cells))
        np_testing.assert_allclose(all_lo[0], 0.0)
        np_testing.assert_allclose(all_hi[-1], 1.0)
        # Adjacent cells tile without gaps or overlaps.
        for lo, hi in zip(all_hi[:-1], all_lo[1:], strict=False):
            np_testing.assert_allclose(lo, hi)

    def test_cell_level_zero_before_refine(self) -> None:
        g = _grid_2d(3, 2)
        for cid in range(g.num_cells):
            assert g.cell_level(cid) == 0

    def test_cell_multi_index_matches_root(self) -> None:
        root = uniform_grid([[0.0, 1.0], [0.0, 1.0]], 3)
        g = hierarchical_grid(root, 2)
        for cid in range(g.num_cells):
            midx = g.cell_multi_index(cid)
            root_midx = root.cell_multi_index(cid)
            assert midx == root_midx


# ──────────────────────────────────────────────────────────────────────────────
# Refinement
# ──────────────────────────────────────────────────────────────────────────────


class TestHierarchicalGridRefine:
    """Tests for the refine method."""

    def test_refine_1d_num_cells(self) -> None:
        g = _grid_1d(4, 2)
        g.refine(0, [1], [3])
        assert g.num_cells == 4 - 2 + 2 * 2  # 6

    def test_refine_2d_num_cells(self) -> None:
        g = _grid_2d(4, 2)
        g.refine(0, [1, 1], [3, 3])
        assert g.num_cells == 16 - 4 + 4 * 4  # 28

    def test_refine_children_tile_parent(self) -> None:
        """Children of a refined cell exactly tile the parent's bounds."""
        root = uniform_grid([[0.0, 1.0], [0.0, 1.0]], 2)
        g = hierarchical_grid(root, 3)
        g.refine(0, [0, 0], [1, 1])  # refine root cell (0,0) only
        parent_lo = np.array([0.0, 0.0])
        parent_hi = np.array([0.5, 0.5])
        fine_los = []
        fine_his = []
        for cid in range(g.num_cells):
            if g.cell_level(cid) == 1:
                lo, hi = g.cell_bounds(cid)
                fine_los.append(lo.copy())
                fine_his.append(hi.copy())
        assert len(fine_los) == 9  # 3*3
        # Union of fine cells = parent
        all_lo = np.min(fine_los, axis=0)
        all_hi = np.max(fine_his, axis=0)
        np_testing.assert_allclose(all_lo, parent_lo)
        np_testing.assert_allclose(all_hi, parent_hi)

    def test_refine_cell_levels(self) -> None:
        g = _grid_1d(4, 2)
        g.refine(0, [1], [3])
        # Cells [0] and [3] at level 0; refined children at level 1.
        for cid in range(g.num_cells):
            lv = g.cell_level(cid)
            lo = g.cell_bounds(cid)[0][0]
            if lo < 0.25 or lo >= 0.75:
                assert lv == 0
            else:
                assert lv == 1

    def test_sequential_refinement(self) -> None:
        """Refine level 0, then refine a sub-region of the level-1 block."""
        g = _grid_1d(4, 2)
        g.refine(0, [1], [3])  # 6 cells
        g.refine(1, [2], [4])  # refine 2 of the 4 level-1 cells
        assert g.max_level == 2
        assert g.num_cells == 6 - 2 + 2 * 2  # 8

    def test_refine_full_domain(self) -> None:
        g = _grid_1d(4, 2)
        g.refine(0, [0], [4])
        assert g.max_level == 1
        assert g.num_cells == 8  # 4 * 2

    def test_refine_overlapping_noop_for_already_refined(self) -> None:
        """A second overlapping refine is a union — already-refined cells skipped."""
        g = _grid_1d(4, 2)
        g.refine(0, [1], [3])
        n1 = g.num_cells
        g.refine(0, [1], [2])  # fully within already-refined region
        assert g.num_cells == n1  # no change

    def test_two_disjoint_refines_same_level(self) -> None:
        g = _grid_2d(6, 2)
        g.refine(0, [0, 0], [2, 2])
        g.refine(0, [4, 4], [6, 6])
        assert g.max_level == 1
        assert len(g._blocks[1]) == 2  # two separate level-1 blocks

    def test_refine_invalid_level_raises(self) -> None:
        g = _grid_1d(4, 2)
        with pytest.raises(ValueError, match="level"):
            g.refine(1, [0], [2])  # level 1 doesn't exist yet

    def test_refine_lo_ge_hi_raises(self) -> None:
        g = _grid_1d(4, 2)
        with pytest.raises(ValueError, match="lo must be strictly less"):
            g.refine(0, [2], [2])

    def test_refine_out_of_bounds_raises(self) -> None:
        g = _grid_1d(4, 2)
        with pytest.raises(ValueError, match="out of bounds"):
            g.refine(0, [0], [5])

    def test_refine_cells_bounding_box(self) -> None:
        """refine_cells uses bounding box of the given cell ids."""
        g = _grid_1d(6, 2)
        # Cells 1 and 3 are at indices 1 and 3 (level 0, midx 1 and 3).
        g.refine_cells([1, 3])
        # Bounding box = [1, 4) → 3 cells refined → 6-3+3*2=9
        assert g.num_cells == 9

    def test_refine_cells_empty_noop(self) -> None:
        g = _grid_1d(4, 2)
        g.refine_cells([])
        assert g.num_cells == 4

    def test_refine_invalidates_bvh(self) -> None:
        g = _grid_2d(4, 2)
        _ = g.cell_bvh()  # build BVH
        g.refine(0, [1, 1], [3, 3])
        assert g._bvh is None  # invalidated

    def test_refine_invalidates_tags(self) -> None:
        g = _grid_2d(4, 2)
        g.cell_tags.set("test", [0, 1], 1)
        g.refine(0, [1, 1], [3, 3])
        assert g._cell_tags is None


# ──────────────────────────────────────────────────────────────────────────────
# locate
# ──────────────────────────────────────────────────────────────────────────────


class TestHierarchicalGridLocate:
    """Tests for locate on hierarchical grids."""

    def test_locate_in_frame_cell(self) -> None:
        g = _grid_1d(4, 2)
        g.refine(0, [1], [3])
        cid = g.locate([0.1])
        assert cid is not None
        assert g.cell_level(cid) == 0

    def test_locate_in_refined_cell(self) -> None:
        g = _grid_1d(4, 2)
        g.refine(0, [1], [3])
        cid = g.locate([0.4])  # inside [0.25, 0.75)
        assert cid is not None
        assert g.cell_level(cid) == 1

    def test_locate_outside_domain(self) -> None:
        g = _grid_1d(4, 2)
        assert g.locate([-0.1]) is None
        assert g.locate([1.1]) is None

    def test_locate_on_boundary(self) -> None:
        g = _grid_1d(4, 2)
        cid = g.locate([0.0])
        assert cid is not None
        cid2 = g.locate([1.0])
        assert cid2 is not None

    def test_locate_after_two_levels(self) -> None:
        g = _grid_1d(4, 2)
        g.refine(0, [1], [3])
        g.refine(1, [2], [4])
        cid = g.locate([0.3])  # in the doubly-refined region [0.25, 0.5)
        assert cid is not None
        assert g.cell_level(cid) == 2

    def test_locate_2d_consistent_with_bounds(self) -> None:
        """Every cell's interior point maps back to that cell."""
        g = _grid_2d(3, 2)
        g.refine(0, [1, 1], [2, 2])
        for cid in range(g.num_cells):
            lo, hi = g.cell_bounds(cid)
            mid = (lo + hi) / 2.0
            found = g.locate(mid)
            assert found == cid, f"cid={cid}, midpoint={mid}, locate={found}"

    def test_locate_wrong_shape_raises(self) -> None:
        g = _grid_2d(3, 2)
        with pytest.raises(ValueError, match="shape"):
            g.locate([0.5])


# ──────────────────────────────────────────────────────────────────────────────
# neighbor_across_facet and hanging_neighbors
# ──────────────────────────────────────────────────────────────────────────────


class TestHierarchicalGridNeighbors:
    """Tests for facet neighbour queries."""

    def test_conforming_neighbor_same_level(self) -> None:
        g = _grid_1d(4, 2)
        # Cell 0 (midx 0) right neighbor = cell 1 (midx 1).
        assert g.neighbor_across_facet(0, 1) == 1
        assert g.neighbor_across_facet(1, 0) == 0

    def test_boundary_facet_returns_none(self) -> None:
        g = _grid_1d(4, 2)
        assert g.neighbor_across_facet(0, 0) is None  # left boundary
        assert g.neighbor_across_facet(3, 1) is None  # right boundary

    def test_coarse_to_fine_neighbor(self) -> None:
        """Frame cell adjacent to a refined region → first fine neighbour."""
        g = _grid_1d(4, 2)
        g.refine(0, [1], [3])
        # Cell 0 (level 0, [0, 0.25)) has right face (lfid=1).
        # Neighbour at (level 0, midx 1) is not active (was refined).
        # First fine child of (0, 1) touching left face: midx 2.
        nbr = g.neighbor_across_facet(0, 1)
        assert nbr is not None
        lo_nbr, _hi_nbr = g.cell_bounds(nbr)
        np_testing.assert_allclose(lo_nbr[0], 0.25)

    def test_fine_to_coarse_neighbor(self) -> None:
        """Fine cell adjacent to a coarser frame cell → the coarse cell."""
        g = _grid_1d(4, 2)
        g.refine(0, [1], [3])
        # First fine cell (midx 2, level 1) has left face adjacent to level-0 frame.
        first_fine = next(cid for cid in range(g.num_cells) if g.cell_level(cid) == 1)
        nbr = g.neighbor_across_facet(first_fine, 0)
        assert nbr is not None
        assert g.cell_level(nbr) == 0

    def test_hanging_neighbors_2d_coarse_to_fine(self) -> None:
        """Factor-2 2D grid: coarse face abuts factor^(d-1) = 2 fine cells."""
        root = uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4)
        g = hierarchical_grid(root, 2)
        g.refine(0, [1, 0], [3, 4])  # refine a band; frame cells on left/right
        # Find a frame cell at level 0 adjacent to the refined band.
        # Cell with level-0 midx (0, k) for any k should have right face touching level-1 cells.
        frame_cid = next(
            cid
            for cid in range(g.num_cells)
            if g.cell_level(cid) == 0 and g.cell_multi_index(cid)[0] == 0
        )
        hn = g.hanging_neighbors(frame_cid, 1)  # right face
        assert len(hn) == 2  # factor^(2-1) = 2

    def test_hanging_neighbors_conforming_tuple_of_one(self) -> None:
        g = _grid_1d(4, 2)
        hn = g.hanging_neighbors(0, 1)
        assert len(hn) == 1
        assert hn[0] == g.neighbor_across_facet(0, 1)

    def test_hanging_neighbors_boundary_empty(self) -> None:
        g = _grid_1d(4, 2)
        assert g.hanging_neighbors(0, 0) == ()

    def test_grid_abc_hanging_neighbors_default(self) -> None:
        """TensorProductGrid inherits the Grid default for hanging_neighbors."""
        root = uniform_grid([[0.0, 1.0]], 4)
        nbr = root.neighbor_across_facet(0, 1)
        hn = root.hanging_neighbors(0, 1)
        assert hn == (nbr,)


# ──────────────────────────────────────────────────────────────────────────────
# BVH / query_aabb
# ──────────────────────────────────────────────────────────────────────────────


class TestHierarchicalGridBVH:
    """Tests for spatial query via inherited BVH."""

    def test_query_aabb_covers_refined_cells(self) -> None:
        g = _grid_2d(4, 2)
        g.refine(0, [1, 1], [3, 3])
        # Query the refined sub-region.
        q = AABB(np.array([0.25, 0.25]), np.array([0.75, 0.75]))
        hits = g.query_aabb(q)
        assert len(hits) > 0
        for cid in hits:
            lo, hi = g.cell_bounds(int(cid))
            # Every hit must overlap or touch the query box.
            assert np.all(lo <= q.hi) and np.all(hi >= q.lo)

    def test_bvh_rebuilt_after_refine(self) -> None:
        g = _grid_2d(4, 2)
        _ = g.cell_bvh()
        g.refine(0, [1, 1], [3, 3])
        # BVH is lazily rebuilt on next query — must not raise.
        bvh = g.cell_bvh()
        assert bvh is not None


# ──────────────────────────────────────────────────────────────────────────────
# Tags
# ──────────────────────────────────────────────────────────────────────────────


class TestHierarchicalGridTags:
    """Tests that tags are correctly invalidated after refinement."""

    def test_cell_tags_reset_after_refine(self) -> None:
        g = _grid_2d(4, 2)
        ct = g.cell_tags  # create
        ct.set("label", [0, 1, 2], 7)
        g.refine(0, [1, 1], [3, 3])
        assert g._cell_tags is None

    def test_cell_tags_usable_after_refine(self) -> None:
        g = _grid_2d(4, 2)
        g.refine(0, [1, 1], [3, 3])
        ct = g.cell_tags
        ct.set("cut", list(range(g.num_cells)), 1)
        assert "cut" in ct

    def test_facet_tags_reset_after_refine(self) -> None:
        g = _grid_1d(4, 2)
        _ = g.facet_tags  # create
        g.refine(0, [1], [3])
        assert g._facet_tags is None


# ──────────────────────────────────────────────────────────────────────────────
# Active-set accessors
# ──────────────────────────────────────────────────────────────────────────────


class TestActiveSetAccessors:
    """Tests for level_cells_per_axis, active_blocks, and the masks."""

    def test_level_cells_per_axis(self) -> None:
        g = _grid_1d(4, 2)
        assert g.level_cells_per_axis(0) == (4,)
        assert g.level_cells_per_axis(2) == (16,)

    def test_level_cells_per_axis_2d_anisotropic(self) -> None:
        g = hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4), (2, 1))
        assert g.level_cells_per_axis(1) == (8, 4)

    def test_level_cells_per_axis_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="level"):
            _grid_1d(4, 2).level_cells_per_axis(-1)

    def test_active_blocks_fresh(self) -> None:
        g = _grid_1d(4, 2)
        assert g.active_blocks(0) == (((0,), (4,)),)

    def test_active_blocks_after_refine(self) -> None:
        g = _grid_1d(4, 2)
        g.refine(0, [0], [2])
        assert g.active_blocks(0) == (((2,), (4,)),)
        assert g.active_blocks(1) == (((0,), (4,)),)

    def test_active_blocks_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="level"):
            _grid_1d(4, 2).active_blocks(1)

    def test_active_leaf_mask_total_equals_num_cells(self) -> None:
        g = _grid_2d(4, 2)
        g.refine(0, [0, 0], [2, 2])
        g.refine(1, [0, 0], [2, 2])
        total = sum(int(g.active_leaf_mask(level).sum()) for level in range(g.max_level + 1))
        assert total == g.num_cells

    def test_subdomain_mask_level0_all_true(self) -> None:
        g = _grid_2d(4, 2)
        g.refine(0, [0, 0], [2, 2])
        assert g.subdomain_mask(0).all()

    def test_mask_consistency_1d(self) -> None:
        g = _grid_1d(4, 2)
        g.refine(0, [0], [2])
        np.testing.assert_array_equal(g.active_leaf_mask(0), [False, False, True, True])
        np.testing.assert_array_equal(g.subdomain_mask(0), [True, True, True, True])
        np.testing.assert_array_equal(
            g.subdomain_mask(1), [True, True, True, True, False, False, False, False]
        )
        np.testing.assert_array_equal(
            g.active_leaf_mask(1), [True, True, True, True, False, False, False, False]
        )

    def test_subdomain_mask_out_of_range_raises(self) -> None:
        g = _grid_1d(4, 2)
        g.refine(0, [0], [2])
        with pytest.raises(ValueError, match="level"):
            g.subdomain_mask(2)

    def test_active_blocks_negative_level_raises(self) -> None:
        with pytest.raises(ValueError, match="level"):
            _grid_1d(4, 2).active_blocks(-1)

    def test_active_leaf_mask_negative_level_raises(self) -> None:
        with pytest.raises(ValueError, match="level"):
            _grid_1d(4, 2).active_leaf_mask(-1)

    def test_subdomain_mask_three_levels(self) -> None:
        # Refine the left half at level 0, then refine all level-1 cells.
        # Exercises the two-iteration accumulation path in subdomain_mask.
        g = _grid_1d(4, 2)
        g.refine(0, [0], [2])  # level-0 block [(2,), (4,)]; level-1 block [(0,), (4,)]
        g.refine(1, [0], [4])  # level-1 block emptied; level-2 block [(0,), (8,)]
        # Level-2 grid: 4 * 2^2 = 16 cells.
        # Subdomain mask: start all True, clear cells covered by coarser leaves.
        # Level-0 leaf block [(2,), (4,)) → scale 4 → slice [8, 16): cleared.
        # Level-1 has no leaf blocks → nothing more to clear.
        expected = np.zeros(16, dtype=bool)
        expected[:8] = True
        np.testing.assert_array_equal(g.subdomain_mask(2), expected)

    def test_is_active_leaf(self) -> None:
        g = _grid_1d(4, 2)
        g.refine(0, [0], [2])  # level-0 leaves at [2, 4); level-1 leaves at [0, 8)
        assert g.is_active_leaf(0, (2,))  # active level-0 leaf
        assert not g.is_active_leaf(0, (0,))  # refined away
        assert g.is_active_leaf(1, (0,))  # active level-1 leaf
        assert not g.is_active_leaf(1, (0, 0))  # wrong ndim
        assert not g.is_active_leaf(0, (-1,))  # out of range
        assert not g.is_active_leaf(5, (0,))  # nonexistent level

    def test_is_active_leaf_2d(self) -> None:
        g = _grid_2d(4, 2)
        # Refine level-0 cell (0, 0) -> children at level 1 in [0,2)x[0,2)
        g.refine(0, [0, 0], [1, 1])
        assert not g.is_active_leaf(0, (0, 0))  # refined away
        assert g.is_active_leaf(0, (1, 0))  # unrefined level-0 leaf
        assert g.is_active_leaf(0, (0, 1))  # unrefined level-0 leaf
        assert g.is_active_leaf(1, (0, 0))  # active level-1 leaf
        assert g.is_active_leaf(1, (1, 1))  # active level-1 leaf (sibling)
        assert not g.is_active_leaf(1, (0,))  # wrong ndim (1D tuple on 2D grid)
        assert not g.is_active_leaf(0, (-1, 0))  # negative index
        assert not g.is_active_leaf(5, (0, 0))  # nonexistent level


# ──────────────────────────────────────────────────────────────────────────────
# Coarsening
# ──────────────────────────────────────────────────────────────────────────────


def _grid_snapshot(g: HierarchicalGrid) -> tuple[object, ...]:
    """Capture the full structural state of a hierarchical grid."""
    return (g.num_cells, g.max_level, tuple(g.active_blocks(lv) for lv in range(g.max_level + 1)))


class TestHierarchicalGridCoarsen:
    """Tests for the coarsen method (inverse of refine)."""

    def test_coarsen_inverts_refine_1d(self) -> None:
        g = _grid_1d(4, 2)
        before = _grid_snapshot(g)
        g.refine(0, [1], [3])
        g.coarsen(0, [1], [3])
        assert _grid_snapshot(g) == before

    def test_coarsen_inverts_refine_2d(self) -> None:
        g = _grid_2d(4, 2)
        before = _grid_snapshot(g)
        g.refine(0, [1, 1], [3, 3])
        g.coarsen(0, [1, 1], [3, 3])
        assert _grid_snapshot(g) == before

    def test_coarsen_drops_trailing_level(self) -> None:
        g = _grid_1d(4, 2)
        g.refine(0, [0], [4])
        assert g.max_level == 1
        g.coarsen(0, [0], [4])
        assert g.max_level == 0
        assert g.num_cells == 4

    def test_coarsen_one_of_two_levels(self) -> None:
        g = _grid_1d(4, 2)
        g.refine(0, [0], [2])
        snap_one = _grid_snapshot(g)
        g.refine(1, [0], [4])  # refine all level-1 cells to level 2
        g.coarsen(1, [0], [4])  # undo just the level-1 refinement
        assert _grid_snapshot(g) == snap_one

    def test_coarsen_partial_region_raises(self) -> None:
        g = _grid_1d(4, 2)
        g.refine(0, [0], [1])  # only cell 0 refined
        with pytest.raises(ValueError, match="fully refined"):
            g.coarsen(0, [0], [2])  # cell 1 has no children

    def test_coarsen_level_out_of_range_raises(self) -> None:
        g = _grid_1d(4, 2)  # max_level 0, no level 1 to coarsen from
        with pytest.raises(ValueError, match="level"):
            g.coarsen(0, [0], [1])

    def test_coarsen_lo_ge_hi_raises(self) -> None:
        g = _grid_1d(4, 2)
        g.refine(0, [0], [4])
        with pytest.raises(ValueError, match="strictly less"):
            g.coarsen(0, [2], [2])

    def test_coarsen_out_of_bounds_raises(self) -> None:
        g = _grid_1d(4, 2)
        g.refine(0, [0], [4])
        with pytest.raises(ValueError, match="out of bounds"):
            g.coarsen(0, [0], [5])  # hi=5 > 4 cells at level 0

    def test_coarsen_wrong_ndim_raises(self) -> None:
        g = _grid_1d(4, 2)
        g.refine(0, [0], [4])
        with pytest.raises(ValueError, match="length"):
            g.coarsen(0, [0, 0], [4, 4])  # 1D grid, 2D lo/hi
