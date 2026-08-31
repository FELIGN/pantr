"""Tests for pantr.grid.HierarchicalGrid."""

from __future__ import annotations

import copy
import itertools
import re
from typing import TYPE_CHECKING

import numpy as np
import numpy.testing as np_testing
import pytest

from pantr.geometry import AABB
from pantr.grid import (
    Grid,
    GridRestriction,
    HierarchicalGrid,
    TensorProductGrid,
    hierarchical_grid,
    uniform_grid,
)
from pantr.grid._hierarchical_grid import (
    _MAX_DIAGNOSED_CELLS,
    _MAX_NAMED_CELLS,
    _block_size,
    _in_block,
    _mark_region_cells,
    _name_marked_cells,
    _normalize_blocks,
    _peel,
    _rect_intersect,
    _try_merge,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    def test_mark_region_cells_marks_only_the_overlap(self) -> None:
        mask = np.zeros((4, 3), dtype=np.bool_)
        _mark_region_cells(mask, (2, 1), (6, 4), ((4, 0), (9, 3)))  # partly outside
        expected = np.zeros((4, 3), dtype=np.bool_)
        expected[2:4, 0:2] = True  # region rows 2-3 (cells 4-5), columns 0-1 (cells 1-2)
        np_testing.assert_array_equal(mask, expected)

    def test_mark_region_cells_ignores_a_disjoint_block(self) -> None:
        mask = np.zeros((3,), dtype=np.bool_)
        _mark_region_cells(mask, (0,), (3,), ((5,), (8,)))
        assert not mask.any()

    def test_name_marked_cells_offsets_by_the_origin(self) -> None:
        mask = np.zeros((4,), dtype=np.bool_)
        mask[[1, 3]] = True
        assert _name_marked_cells(mask, (10,)) == "(11,), (13,)"

    def test_name_marked_cells_counts_the_remainder(self) -> None:
        mask = np.ones((_MAX_NAMED_CELLS + 2,), dtype=np.bool_)
        named = _name_marked_cells(mask, (0,))
        assert named.count("), (") == _MAX_NAMED_CELLS - 1
        assert named.endswith("and 2 more")

    def test_name_marked_cells_2d(self) -> None:
        mask = np.zeros((2, 2), dtype=np.bool_)
        mask[1, 0] = True
        assert _name_marked_cells(mask, (3, 7)) == "(4, 7)"


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
        g = g.refine(0, [1], [3])
        assert g.num_cells == 4 - 2 + 2 * 2  # 6

    def test_refine_2d_num_cells(self) -> None:
        g = _grid_2d(4, 2)
        g = g.refine(0, [1, 1], [3, 3])
        assert g.num_cells == 16 - 4 + 4 * 4  # 28

    def test_refine_children_tile_parent(self) -> None:
        """Children of a refined cell exactly tile the parent's bounds."""
        root = uniform_grid([[0.0, 1.0], [0.0, 1.0]], 2)
        g = hierarchical_grid(root, 3)
        g = g.refine(0, [0, 0], [1, 1])  # refine root cell (0,0) only
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
        g = g.refine(0, [1], [3])
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
        g = g.refine(0, [1], [3])  # 6 cells
        g = g.refine(1, [2], [4])  # refine 2 of the 4 level-1 cells
        assert g.max_level == 2
        assert g.num_cells == 6 - 2 + 2 * 2  # 8

    def test_refine_full_domain(self) -> None:
        g = _grid_1d(4, 2)
        g = g.refine(0, [0], [4])
        assert g.max_level == 1
        assert g.num_cells == 8  # 4 * 2

    def test_refine_overlapping_noop_for_already_refined(self) -> None:
        """A second overlapping refine is a union — already-refined cells skipped."""
        g = _grid_1d(4, 2)
        g = g.refine(0, [1], [3])
        n1 = g.num_cells
        g = g.refine(0, [1], [2])  # fully within already-refined region
        assert g.num_cells == n1  # no change

    def test_two_disjoint_refines_same_level(self) -> None:
        g = _grid_2d(6, 2)
        g = g.refine(0, [0, 0], [2, 2])
        g = g.refine(0, [4, 4], [6, 6])
        assert g.max_level == 1
        assert len(g._blocks[1]) == 2  # two separate level-1 blocks

    def test_refine_invalid_level_raises(self) -> None:
        g = _grid_1d(4, 2)
        with pytest.raises(ValueError, match="level"):
            g = g.refine(1, [0], [2])  # level 1 doesn't exist yet

    def test_refine_lo_ge_hi_raises(self) -> None:
        g = _grid_1d(4, 2)
        with pytest.raises(ValueError, match="lo must be strictly less"):
            g = g.refine(0, [2], [2])

    def test_refine_out_of_bounds_raises(self) -> None:
        g = _grid_1d(4, 2)
        with pytest.raises(ValueError, match="out of bounds"):
            g = g.refine(0, [0], [5])

    def test_refine_cells_bounding_box(self) -> None:
        """refine_cells uses bounding box of the given cell ids."""
        g = _grid_1d(6, 2)
        # Cells 1 and 3 are at indices 1 and 3 (level 0, midx 1 and 3).
        g = g.refine_cells([1, 3])
        # Bounding box = [1, 4) → 3 cells refined → 6-3+3*2=9
        assert g.num_cells == 9

    def test_refine_cells_empty_noop(self) -> None:
        g = _grid_1d(4, 2)
        g = g.refine_cells([])
        assert g.num_cells == 4

    def test_refine_gives_the_new_grid_a_fresh_bvh_and_leaves_the_receivers(self) -> None:
        g = _grid_2d(4, 2)
        _ = g.cell_bvh()  # build BVH
        new = g.refine(0, [1, 1], [3, 3])
        assert new._bvh is None  # the returned grid starts fresh
        assert g._bvh is not None  # the receiver keeps its own

    def test_refine_gives_the_new_grid_fresh_tags_and_leaves_the_receivers(self) -> None:
        g = _grid_2d(4, 2)
        g.cell_tags.set("test", [0, 1], 1)
        new = g.refine(0, [1, 1], [3, 3])
        assert new._cell_tags is None  # the returned grid starts fresh
        assert g._cell_tags is not None  # the receiver keeps its own


# ──────────────────────────────────────────────────────────────────────────────
# locate
# ──────────────────────────────────────────────────────────────────────────────


class TestHierarchicalGridLocate:
    """Tests for locate on hierarchical grids."""

    def test_locate_in_frame_cell(self) -> None:
        g = _grid_1d(4, 2)
        g = g.refine(0, [1], [3])
        cid = g.locate([0.1])
        assert cid is not None
        assert g.cell_level(cid) == 0

    def test_locate_in_refined_cell(self) -> None:
        g = _grid_1d(4, 2)
        g = g.refine(0, [1], [3])
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
        g = g.refine(0, [1], [3])
        g = g.refine(1, [2], [4])
        cid = g.locate([0.3])  # in the doubly-refined region [0.25, 0.5)
        assert cid is not None
        assert g.cell_level(cid) == 2

    def test_locate_2d_consistent_with_bounds(self) -> None:
        """Every cell's interior point maps back to that cell."""
        g = _grid_2d(3, 2)
        g = g.refine(0, [1, 1], [2, 2])
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
        g = g.refine(0, [1], [3])
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
        g = g.refine(0, [1], [3])
        # First fine cell (midx 2, level 1) has left face adjacent to level-0 frame.
        first_fine = next(cid for cid in range(g.num_cells) if g.cell_level(cid) == 1)
        nbr = g.neighbor_across_facet(first_fine, 0)
        assert nbr is not None
        assert g.cell_level(nbr) == 0

    def test_hanging_neighbors_2d_coarse_to_fine(self) -> None:
        """Factor-2 2D grid: coarse face abuts factor^(d-1) = 2 fine cells."""
        root = uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4)
        g = hierarchical_grid(root, 2)
        g = g.refine(0, [1, 0], [3, 4])  # refine a band; frame cells on left/right
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
        g = g.refine(0, [1, 1], [3, 3])
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
        g = g.refine(0, [1, 1], [3, 3])
        # BVH is lazily rebuilt on next query — must not raise.
        bvh = g.cell_bvh()
        assert bvh is not None


# ──────────────────────────────────────────────────────────────────────────────
# Tags
# ──────────────────────────────────────────────────────────────────────────────


class TestHierarchicalGridTags:
    """Tests that tags are correctly invalidated after refinement."""

    def test_cell_tags_reset_on_the_returned_grid_not_the_receiver(self) -> None:
        g = _grid_2d(4, 2)
        ct = g.cell_tags  # create
        ct.set("label", [0, 1, 2], 7)
        new = g.refine(0, [1, 1], [3, 3])
        assert new._cell_tags is None  # the returned grid starts fresh
        assert g._cell_tags is not None  # the receiver keeps its own
        ids, values = g.cell_tags["label"]
        np_testing.assert_array_equal(ids, [0, 1, 2])
        np_testing.assert_array_equal(values, [7, 7, 7])  # still readable

    def test_cell_tags_usable_after_refine(self) -> None:
        g = _grid_2d(4, 2)
        g = g.refine(0, [1, 1], [3, 3])
        ct = g.cell_tags
        ct.set("cut", list(range(g.num_cells)), 1)
        assert "cut" in ct

    def test_facet_tags_reset_on_the_returned_grid_not_the_receiver(self) -> None:
        g = _grid_1d(4, 2)
        g.facet_tags.set("cut", [[0, 0]], 1)
        new = g.refine(0, [1], [3])
        assert new._facet_tags is None  # the returned grid starts fresh
        assert g._facet_tags is not None  # the receiver keeps its own
        keys, values = g.facet_tags["cut"]
        np_testing.assert_array_equal(keys, [[0, 0]])
        np_testing.assert_array_equal(values, [1])  # still readable


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
        g = g.refine(0, [0], [2])
        assert g.active_blocks(0) == (((2,), (4,)),)
        assert g.active_blocks(1) == (((0,), (4,)),)

    def test_active_blocks_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="level"):
            _grid_1d(4, 2).active_blocks(1)

    def test_active_leaf_mask_total_equals_num_cells(self) -> None:
        g = _grid_2d(4, 2)
        g = g.refine(0, [0, 0], [2, 2])
        g = g.refine(1, [0, 0], [2, 2])
        total = sum(int(g.active_leaf_mask(level).sum()) for level in range(g.max_level + 1))
        assert total == g.num_cells

    def test_subdomain_mask_level0_all_true(self) -> None:
        g = _grid_2d(4, 2)
        g = g.refine(0, [0, 0], [2, 2])
        assert g.subdomain_mask(0).all()

    def test_mask_consistency_1d(self) -> None:
        g = _grid_1d(4, 2)
        g = g.refine(0, [0], [2])
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
        g = g.refine(0, [0], [2])
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
        g = g.refine(0, [0], [2])  # level-0 block [(2,), (4,)]; level-1 block [(0,), (4,)]
        g = g.refine(1, [0], [4])  # level-1 block emptied; level-2 block [(0,), (8,)]
        # Level-2 grid: 4 * 2^2 = 16 cells.
        # Subdomain mask: start all True, clear cells covered by coarser leaves.
        # Level-0 leaf block [(2,), (4,)) → scale 4 → slice [8, 16): cleared.
        # Level-1 has no leaf blocks → nothing more to clear.
        expected = np.zeros(16, dtype=bool)
        expected[:8] = True
        np.testing.assert_array_equal(g.subdomain_mask(2), expected)

    def test_is_active_leaf(self) -> None:
        g = _grid_1d(4, 2)
        g = g.refine(0, [0], [2])  # level-0 leaves at [2, 4); level-1 leaves at [0, 8)
        assert g.is_active_leaf(0, (2,))  # active level-0 leaf
        assert not g.is_active_leaf(0, (0,))  # refined away
        assert g.is_active_leaf(1, (0,))  # active level-1 leaf
        assert not g.is_active_leaf(1, (0, 0))  # wrong ndim
        assert not g.is_active_leaf(0, (-1,))  # out of range
        assert not g.is_active_leaf(5, (0,))  # nonexistent level

    def test_is_active_leaf_2d(self) -> None:
        g = _grid_2d(4, 2)
        # Refine level-0 cell (0, 0) -> children at level 1 in [0,2)x[0,2)
        g = g.refine(0, [0, 0], [1, 1])
        assert not g.is_active_leaf(0, (0, 0))  # refined away
        assert g.is_active_leaf(0, (1, 0))  # unrefined level-0 leaf
        assert g.is_active_leaf(0, (0, 1))  # unrefined level-0 leaf
        assert g.is_active_leaf(1, (0, 0))  # active level-1 leaf
        assert g.is_active_leaf(1, (1, 1))  # active level-1 leaf (sibling)
        assert not g.is_active_leaf(1, (0,))  # wrong ndim (1D tuple on 2D grid)
        assert not g.is_active_leaf(0, (-1, 0))  # negative index
        assert not g.is_active_leaf(5, (0, 0))  # nonexistent level

    def test_cell_id_inverts_cell_level_and_multi_index(self) -> None:
        """`cell_id` round-trips every active leaf, at every level, in both directions."""
        g = _grid_2d(4, 2)
        g = g.refine(0, [0, 0], [2, 2])
        g = g.refine(1, [0, 0], [2, 2])
        assert g.max_level == 2
        for cid in range(g.num_cells):
            assert g.cell_id(g.cell_level(cid), g.cell_multi_index(cid)) == cid

    def test_cell_id_none_for_cells_that_are_not_active_leaves(self) -> None:
        """The `None` cases are exactly `is_active_leaf`'s `False` cases."""
        g = _grid_1d(4, 2)
        g = g.refine(0, [0], [2])  # level-0 leaves at [2, 4); level-1 leaves at [0, 4)
        assert g.cell_id(0, (0,)) is None  # refined away
        assert g.cell_id(0, (-1,)) is None  # negative index
        assert g.cell_id(0, (9,)) is None  # past the level's extent
        assert g.cell_id(1, (0, 0)) is None  # wrong ndim
        assert g.cell_id(5, (0,)) is None  # nonexistent level
        assert g.cell_id(-1, (0,)) is None  # negative level

    def test_cell_id_is_resolved_afresh_after_a_mutation(self) -> None:
        """Ids move under refinement; the `(level, midx)` pair does not."""
        g = _grid_1d(4, 2)
        g = g.refine(0, [3], [4])  # refine the last root cell
        before = g.cell_id(0, (0,))
        g = g.refine(0, [0], [1])  # ids shift: level-0 loses a cell, level-1 gains two
        assert before == 0
        assert g.cell_id(0, (0,)) is None  # cell (0, 0) is gone, its id belongs elsewhere
        assert g.cell_id(0, (1,)) == 0
        assert g.cell_id(1, (0,)) == 2


# ──────────────────────────────────────────────────────────────────────────────
# Coarsening
# ──────────────────────────────────────────────────────────────────────────────


def _grid_snapshot(g: HierarchicalGrid) -> tuple[object, ...]:
    """Capture the full structural state of a hierarchical grid."""
    return (g.num_cells, g.max_level, tuple(g.active_blocks(lv) for lv in range(g.max_level + 1)))


def _obstacles_by_cellwise_walk(
    g: HierarchicalGrid,
    level: int,
    lo: Sequence[int],
    hi: Sequence[int],
) -> set[tuple[int, ...]]:
    """Return the cells of ``[lo, hi)`` that cannot be demoted, found cell by cell.

    An independent oracle for `coarsen`'s refusal: it uses only `is_active_leaf`, so it
    shares no arithmetic with the block scaling the error message is built from.
    """
    blocked: set[tuple[int, ...]] = set()
    for cell in itertools.product(*[range(a, b) for a, b in zip(lo, hi, strict=True)]):
        children = [
            tuple(cell[k] * g.factor[k] + off[k] for k in range(g.ndim))
            for off in itertools.product(*(range(g.factor[k]) for k in range(g.ndim)))
        ]
        if not all(g.is_active_leaf(level + 1, child) for child in children):
            blocked.add(cell)
    return blocked


_REFUSAL_REASONS = (
    "still active leaves",
    "refined beyond",
    "covered by a coarser active leaf",
)
"""The three reasons `coarsen`'s refusal can give, as they appear in its message.

Used to assert a randomized sweep actually exercised all three rather than grading
whichever one its draws happened to reach.
"""


def _cells_named_in(message: str) -> set[tuple[int, ...]]:
    """Return the cell indices a `coarsen` refusal names, excluding its region prefix."""
    listing = message.split("Offending cells", 1)[1] if "Offending cells" in message else ""
    return {
        tuple(int(value) for value in match.group(1).split(","))
        for match in re.finditer(r"\((\d+(?:,\s*\d+)*),?\)", listing)
    }


class TestHierarchicalGridCoarsen:
    """Tests for the coarsen method: argument validation and the accepted cases."""

    def test_coarsen_inverts_refine_1d(self) -> None:
        g = _grid_1d(4, 2)
        before = _grid_snapshot(g)
        g = g.refine(0, [1], [3])
        g = g.coarsen(0, [1], [3])
        assert _grid_snapshot(g) == before

    def test_coarsen_inverts_refine_2d(self) -> None:
        g = _grid_2d(4, 2)
        before = _grid_snapshot(g)
        g = g.refine(0, [1, 1], [3, 3])
        g = g.coarsen(0, [1, 1], [3, 3])
        assert _grid_snapshot(g) == before

    def test_coarsen_drops_trailing_level(self) -> None:
        g = _grid_1d(4, 2)
        g = g.refine(0, [0], [4])
        assert g.max_level == 1
        g = g.coarsen(0, [0], [4])
        assert g.max_level == 0
        assert g.num_cells == 4

    def test_coarsen_one_of_two_levels(self) -> None:
        g = _grid_1d(4, 2)
        g = g.refine(0, [0], [2])
        snap_one = _grid_snapshot(g)
        g = g.refine(1, [0], [4])  # refine all level-1 cells to level 2
        g = g.coarsen(1, [0], [4])  # undo just the level-1 refinement
        assert _grid_snapshot(g) == snap_one

    def test_coarsen_partial_region_raises(self) -> None:
        g = _grid_1d(4, 2)
        g = g.refine(0, [0], [1])  # only cell 0 refined
        with pytest.raises(ValueError, match="fully refined"):
            g = g.coarsen(0, [0], [2])  # cell 1 has no children

    def test_coarsen_level_out_of_range_raises(self) -> None:
        g = _grid_1d(4, 2)  # max_level 0, no level 1 to coarsen from
        with pytest.raises(ValueError, match="level"):
            g = g.coarsen(0, [0], [1])

    def test_coarsen_lo_ge_hi_raises(self) -> None:
        g = _grid_1d(4, 2)
        g = g.refine(0, [0], [4])
        with pytest.raises(ValueError, match="strictly less"):
            g = g.coarsen(0, [2], [2])

    def test_coarsen_out_of_bounds_raises(self) -> None:
        g = _grid_1d(4, 2)
        g = g.refine(0, [0], [4])
        with pytest.raises(ValueError, match="out of bounds"):
            g = g.coarsen(0, [0], [5])  # hi=5 > 4 cells at level 0

    def test_coarsen_wrong_ndim_raises(self) -> None:
        g = _grid_1d(4, 2)
        g = g.refine(0, [0], [4])
        with pytest.raises(ValueError, match="length"):
            g = g.coarsen(0, [0, 0], [4, 4])  # 1D grid, 2D lo/hi


class TestCoarsenIsNotAnUnconditionalInverse:
    """`coarsen` demotes its whole box, so it inverts `refine` only conditionally.

    `refine` promotes just the currently-active part of its box (union semantics), so it
    is not injective on the grid state and no single `coarsen` can invert it.  These
    tests pin both the conditional inverse and the consequence when the hypothesis
    fails, with the cell counts of the reproduction hardcoded.
    """

    def test_coarsen_inverts_refine_of_an_entirely_active_region_1d(self) -> None:
        """One refine of an all-active box, then coarsen, restores the grid exactly."""
        g = _grid_1d(6, 2)
        assert g.num_cells == 6
        g = g.refine(0, [0], [2])
        assert g.num_cells == 8
        g = g.coarsen(0, [0], [2])
        assert g.num_cells == 6

    def test_coarsen_inverts_refine_disjoint_from_an_earlier_one_1d(self) -> None:
        """Two disjoint refines: coarsening the second is still exact."""
        g = _grid_1d(6, 2)
        g = g.refine(0, [0], [2])
        g = g.refine(0, [3], [5])
        assert g.num_cells == 10
        g = g.coarsen(0, [3], [5])
        assert g.num_cells == 8

    def test_coarsen_after_overlapping_refines_demotes_the_whole_box_1d(self) -> None:
        """Coarsening a box built by overlapping refines also removes older children.

        The second refine promotes cell 2 alone, because cell 1 was already refined by
        the first one.  `coarsen` then demotes the *whole* box, so it also removes
        cell 1's children and the grid ends at 7 cells, not the 8 it had before the
        second refine.  That is `coarsen`'s documented contract (it demotes the box it
        is given), **not** a defect to "fix" by changing this number: the route that
        cannot lose refinement is :meth:`HierarchicalGrid.coarsen_cells`, where the
        caller names every cell being destroyed.  `TestCoarsenCells` below runs this
        same reproduction through it and gets the 8 back.
        """
        g = _grid_1d(6, 2)
        g = g.refine(0, [0], [2])
        assert g.num_cells == 8
        g = g.refine(0, [1], [3])
        assert g.num_cells == 9
        g = g.coarsen(0, [1], [3])
        assert g.num_cells == 7
        assert g.active_blocks(0) == (((1,), (6,)),)
        assert g.active_blocks(1) == (((0,), (2,)),)

    def test_coarsen_inverts_refine_of_an_entirely_active_region_2d_non_dyadic(self) -> None:
        """The 2D control on a non-dyadic factor: single refine, then coarsen, is exact."""
        g = _grid_2d(3, 3)
        assert g.num_cells == 9
        g = g.refine(0, [1, 1], [3, 3])
        assert g.num_cells == 41
        g = g.coarsen(0, [1, 1], [3, 3])
        assert g.num_cells == 9

    def test_coarsen_after_overlapping_refines_demotes_the_whole_box_2d_non_dyadic(self) -> None:
        """The 2D non-dyadic counterpart: cell (1, 1)'s 9 children become 1 cell again.

        Same contract as the 1D case above, on ``factor = 3`` so a factor-dependent
        regression cannot hide: 41 would mean `coarsen` had inverted the second refine.
        """
        g = _grid_2d(3, 3)
        g = g.refine(0, [0, 0], [2, 2])
        assert g.num_cells == 41
        g = g.refine(0, [1, 1], [3, 3])
        assert g.num_cells == 65
        g = g.coarsen(0, [1, 1], [3, 3])
        assert g.num_cells == 33
        # The shape, not only the count: dropping the block merge on reactivation leaves
        # 33 cells in a structurally different partition, which the count alone misses.
        assert g.active_blocks(0) == (((0, 2), (1, 3)), ((1, 1), (3, 3)), ((2, 0), (3, 1)))
        assert g.active_blocks(1) == (((0, 0), (3, 6)), ((3, 0), (6, 3)))

    def test_refine_undoes_coarsen_unconditionally(self) -> None:
        """The other direction holds with no hypothesis: refine always undoes coarsen."""
        g = _grid_1d(6, 2)
        g = g.refine(0, [0], [2])
        g = g.refine(0, [1], [3])
        before = _grid_snapshot(g)
        g = g.coarsen(0, [1], [3])
        g = g.refine(0, [1], [3])
        assert _grid_snapshot(g) == before

    def test_coarsen_names_cells_that_are_still_leaves(self) -> None:
        """The refusal names the box cells that have no children to remove."""
        g = _grid_1d(4, 2)
        g = g.refine(0, [0], [1])  # only cell 0 is refined
        with pytest.raises(ValueError, match="still active leaves at level 0") as excinfo:
            g = g.coarsen(0, [0], [2])
        named = str(excinfo.value).split("still active leaves at level 0")[1]
        assert "(1,)" in named
        assert "refined beyond" not in str(excinfo.value)
        # What the message says is true of the state: cell 1 is a leaf, cell 0 is not.
        assert g.is_active_leaf(0, (1,))
        assert not g.is_active_leaf(0, (0,))

    def test_coarsen_names_cells_refined_beyond_the_target_level(self) -> None:
        """The refusal names the box cells whose children are themselves refined."""
        g = _grid_1d(4, 2)
        g = g.refine(0, [0], [2])
        g = g.refine(1, [0], [2])  # cell 0's children go on to level 2
        with pytest.raises(ValueError, match="refined beyond level 1") as excinfo:
            g = g.coarsen(0, [0], [2])
        named = str(excinfo.value).split("refined beyond level 1")[1]
        assert "(0,)" in named
        assert "still active leaves" not in str(excinfo.value)
        # True of the state: cell 0's children are not leaves at level 1, but at level 2.
        assert not g.is_active_leaf(1, (0,))
        assert g.is_active_leaf(2, (0,))

    @pytest.mark.parametrize(
        ("cells_per_axis", "factor"),
        [([6], 3), ([4, 3], (2, 3)), ([3, 3], 2), ([2, 3, 2], (2, 1, 3))],
        ids=["1d-non-dyadic", "2d-anisotropic", "2d-dyadic", "3d-anisotropic"],
    )
    def test_named_cells_match_a_cellwise_oracle(
        self,
        cells_per_axis: list[int],
        factor: int | tuple[int, ...],
    ) -> None:
        """Every refusal names exactly the cells a cellwise walk finds, and no others.

        The message is built from the block lists with per-axis scaling; the oracle here
        instead asks :meth:`~pantr.grid.HierarchicalGrid.is_active_leaf` cell by cell, so
        the two disagree if the scaling is wrong for a non-dyadic or anisotropic factor.

        The draw count is not arbitrary. At 24 draws this seed leaves three of the four
        parametrizations never reaching one of the three refusal reasons, so the sweep
        graded less than it appears to; 96 is the first count at which all four reach all
        three, which is what the tally at the end asserts. Raising it further only repeats
        coverage, except that it also exercises the truncated-listing branch more often
        (never reachable in 1D, where a drawn region spans at most three cells).
        """
        root = uniform_grid([[0.0, 1.0]] * len(cells_per_axis), cells_per_axis)
        g = hierarchical_grid(root, factor)
        rng = np.random.default_rng(4)
        checked = 0
        reasons_seen: set[str] = set()
        for _ in range(96):
            coarsening = g.max_level > 0 and rng.random() >= 0.6
            level = int(rng.integers(0, g.max_level if coarsening else g.max_level + 1))
            extent = g.level_cells_per_axis(level)
            lo = [int(rng.integers(0, n)) for n in extent]
            hi = [
                int(rng.integers(a + 1, min(a + 3, n) + 1)) for a, n in zip(lo, extent, strict=True)
            ]
            if not coarsening:
                g = g.refine(level, lo, hi)
                continue
            expected = _obstacles_by_cellwise_walk(g, level, lo, hi)
            try:
                g = g.coarsen(level, lo, hi)
            except ValueError as exc:
                assert expected, f"refused a region the oracle finds coarsenable: {exc}"
                named = _cells_named_in(str(exc))
                assert named, f"refused without naming a cell: {exc}"
                if "more" in str(exc):
                    assert named <= expected  # truncated: a subset, never an invention
                else:
                    assert named == expected, f"named {sorted(named)}, blocked {sorted(expected)}"
                reasons_seen |= {phrase for phrase in _REFUSAL_REASONS if phrase in str(exc)}
                checked += 1
            else:
                assert not expected, "coarsened a region the oracle finds blocked"
        assert checked, "the sweep never exercised a refusal"
        # Without this the sweep can grade only one or two of the three reasons and still
        # report success, which is how a per-axis scaling bug in an unexercised branch
        # would survive: the message path is the only place that scaling is used.
        assert reasons_seen == set(_REFUSAL_REASONS), (
            f"the sweep never exercised {sorted(set(_REFUSAL_REASONS) - reasons_seen)}"
        )

    def test_coarsen_names_exactly_the_cap_without_a_remainder(self) -> None:
        """A listing landing exactly on the cap must not claim a remainder."""
        g = _grid_1d(8, 2)
        g = g.refine(0, [0], [1])  # cells 1..6 of the box below are leaves: exactly the cap
        with pytest.raises(ValueError) as excinfo:
            g = g.coarsen(0, [0], [1 + _MAX_NAMED_CELLS])
        message = str(excinfo.value)
        assert "more" not in message
        named = message.split("with no children to remove: ")[1]
        assert named.count("), (") == _MAX_NAMED_CELLS - 1
        # One cell further, the remainder is reported rather than dropped.
        with pytest.raises(ValueError, match="and 1 more"):
            g = g.coarsen(0, [0], [2 + _MAX_NAMED_CELLS])

    def test_coarsen_truncates_a_long_list_of_offending_cells(self) -> None:
        """A large rejected region names a few cells and counts the rest."""
        g = _grid_2d(6, 2)
        g = g.refine(0, [0, 0], [1, 1])  # 1 of the 36 level-0 cells is refined
        with pytest.raises(ValueError) as excinfo:
            g = g.coarsen(0, [0, 0], [6, 6])
        message = str(excinfo.value)
        named = message.split("with no children to remove: ")[1]
        assert named.count("), (") == _MAX_NAMED_CELLS - 1  # exactly the cap is spelled out
        assert f"and {36 - 1 - _MAX_NAMED_CELLS} more" in message

    def test_coarsen_reports_extent_instead_of_cells_for_a_huge_region(self) -> None:
        """A region far larger than the grid is described, not enumerated.

        Level ``l`` has ``factor ** l`` cells per root cell whether or not they exist, so
        a deep level admits an in-bounds region with more cells than the grid has.
        Naming them would need a mask per cell, so past the budget the message reports
        the extent and the refusal stays a `ValueError` rather than an out-of-memory.
        """
        g = _grid_1d(2, 2)
        for level in range(21):
            g = g.refine(level, [0], [1])  # deepen without growing the grid
        region = g.level_cells_per_axis(20)[0]
        assert region > _MAX_DIAGNOSED_CELLS
        with pytest.raises(ValueError, match=f"spans {region} cells, too many to name"):
            g = g.coarsen(20, [0], [region])

    def test_coarsen_names_offending_cells_with_an_anisotropic_factor(self) -> None:
        """The per-axis factor is respected when locating the offending cells."""
        g = hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], [2, 2]), (2, 3))
        assert g.factor == (2, 3)
        g = g.refine(0, [0, 0], [2, 2])  # every level-0 cell -> level 1
        g = g.refine(1, [0, 0], [1, 1])  # level-1 cell (0, 0) -> level 2
        with pytest.raises(ValueError, match="refined beyond level 1") as excinfo:
            g = g.coarsen(0, [0, 0], [2, 2])
        named = str(excinfo.value).split("refined beyond level 1")[1]
        assert "(0, 0)" in named
        # Only cell (0, 0) owns the level-2 cells, so no other cell may be named.
        assert named.count("(") == 1
        assert g.is_active_leaf(2, (0, 0))
        assert g.is_active_leaf(1, (1, 0))

    def test_coarsen_names_every_reason_and_only_offending_cells(self) -> None:
        """One box, all three reasons, and a coarsenable cell left unnamed.

        Level-1 cell 0 is refined past level 2, cells 4-7 are still leaves, cells 8-11
        sit inside level-0 leaves, and cell 1 is refined to exactly level 2, so it must
        not appear anywhere in the message.
        """
        g = _grid_1d(8, 2)
        g = g.refine(0, [0], [4])  # level-1 cells 0..7
        g = g.refine(1, [0], [4])  # level-1 cells 0..3 -> level-2 cells 0..7
        g = g.refine(2, [0], [2])  # level-2 cells 0, 1 -> level 3
        with pytest.raises(ValueError) as excinfo:
            g = g.coarsen(1, [0], [12])
        message = str(excinfo.value)
        assert "still active leaves at level 1" in message
        assert "refined beyond level 2" in message
        assert "covered by a coarser active leaf" in message
        assert "(0,)" in message.split("refined beyond level 2")[1]
        assert "(1,)" not in message.split("Offending cells")[1]
        # True of the state: cell 0 has a level-3 descendant, cell 1's children are
        # level-2 leaves, cells 4 and 8 are a level-1 leaf and a hidden cell.
        assert g.is_active_leaf(3, (0,))
        assert g.is_active_leaf(2, (2,))
        assert g.is_active_leaf(1, (4,))
        assert not g.is_active_leaf(1, (8,))

    def test_coarsen_names_cells_absent_at_the_requested_level(self) -> None:
        """The refusal names box cells that a coarser active leaf covers."""
        g = _grid_1d(4, 2)
        g = g.refine(0, [0], [2])
        g = g.refine(1, [0], [2])  # max_level 2, so coarsening level 1 is in range
        with pytest.raises(ValueError, match="covered by a coarser active leaf") as excinfo:
            g = g.coarsen(1, [4], [6])  # level-1 cells 4, 5 sit inside level-0 leaf cell 2
        named = str(excinfo.value).split("covered by a coarser active leaf")[1]
        assert "(4,)" in named
        assert "(5,)" in named
        # True of the state: cell 2 is a level-0 leaf, so level-1 cells 4, 5 do not exist.
        assert g.is_active_leaf(0, (2,))
        assert not g.is_active_leaf(1, (4,))
        assert not g.is_active_leaf(1, (5,))


def _caches_live(g: HierarchicalGrid) -> tuple[bool, bool]:
    """Return whether the BVH and cell-tag caches are currently populated.

    Read through a function on purpose: asserting on ``g._bvh`` directly narrows the
    attribute, after which mypy declares the opposite assertion later in the same test
    unreachable and fails the run.
    """
    return g._bvh is not None, g._cell_tags is not None


def _active_cells(g: HierarchicalGrid) -> set[tuple[int, tuple[int, ...]]]:
    """Return every active leaf as a ``(level, midx)`` pair.

    Unlike a flat id, the pair survives the reassignment every refine and coarsen
    performs, so two states of the same grid can be compared cell by cell.
    """
    return {(g.cell_level(cid), g.cell_multi_index(cid)) for cid in range(g.num_cells)}


class TestCoarsenCells:
    """`coarsen_cells` destroys exactly the cells it is given, and nothing else."""

    def test_round_trip_with_refine_cells_1d(self) -> None:
        """Refining one cell and coarsening its children restores the original grid."""
        g = _grid_1d(4, 2)
        before = _grid_snapshot(g)
        assert before == (4, 0, ((((0,), (4,)),),))
        g = g.refine_cells([0])
        assert (g.num_cells, g.max_level) == (5, 1)
        g = g.coarsen_cells([c for c in range(g.num_cells) if g.cell_level(c) == 1])
        assert _grid_snapshot(g) == before
        assert (g.num_cells, g.max_level, g.active_blocks(0)) == (4, 0, (((0,), (4,)),))

    def test_round_trip_with_refine_cells_2d_non_dyadic(self) -> None:
        """The 2D control on ``factor = 3``: nine children collapse back to one cell."""
        g = _grid_2d(3, 3)
        before = _grid_snapshot(g)
        g = g.refine_cells([4])  # the middle root cell
        assert (g.num_cells, g.max_level) == (17, 1)
        g = g.coarsen_cells([c for c in range(g.num_cells) if g.cell_level(c) == 1])
        assert _grid_snapshot(g) == before

    def test_partial_parent_is_skipped_without_raising(self) -> None:
        """A parent with only some children named is left exactly as it was."""
        g = _grid_1d(4, 2)
        g = g.refine_cells([0])
        before = _grid_snapshot(g)
        children = [c for c in range(g.num_cells) if g.cell_level(c) == 1]
        assert len(children) == 2
        g = g.coarsen_cells(children[:1])  # one of the two children
        assert _grid_snapshot(g) == before

    def test_level_zero_ids_are_ignored(self) -> None:
        """A root cell has no parent, so naming it does nothing (and does not raise)."""
        g = _grid_1d(4, 2)
        g = g.refine_cells([0])
        before = _grid_snapshot(g)
        g = g.coarsen_cells([c for c in range(g.num_cells) if g.cell_level(c) == 0])
        assert _grid_snapshot(g) == before

    def test_empty_and_repeated_ids(self) -> None:
        """An empty call is a no-op; a repeated id counts once, not twice."""
        g = _grid_1d(4, 2)
        g = g.refine_cells([0])
        before = _grid_snapshot(g)
        g = g.coarsen_cells([])
        assert _grid_snapshot(g) == before
        children = [c for c in range(g.num_cells) if g.cell_level(c) == 1]
        g = g.coarsen_cells([*children, *children])
        assert (g.num_cells, g.max_level) == (4, 0)

    def test_out_of_range_id_raises_before_anything_is_demoted(self) -> None:
        """Out of range is an `IndexError`, as for `refine_cells`, and nothing has moved.

        Every id is range-checked before the first parent is demoted, so a list whose
        bad id sits *after* a demotable family leaves the grid exactly as it was rather
        than half-coarsened.  A per-parent check would leave the first family gone.
        """
        g = _grid_1d(4, 2)
        with pytest.raises(IndexError, match="out of range"):
            g = g.coarsen_cells([4])
        with pytest.raises(IndexError, match="out of range"):
            g = g.coarsen_cells([-1])
        assert _grid_snapshot(g) == (4, 0, ((((0,), (4,)),),))

        g = g.refine_cells([0])
        before = _grid_snapshot(g)
        children = [c for c in range(g.num_cells) if g.cell_level(c) == 1]
        with pytest.raises(IndexError, match="out of range"):
            g = g.coarsen_cells([*children, g.num_cells])
        assert _grid_snapshot(g) == before

    def test_parents_at_two_levels_are_demoted_deepest_first(self) -> None:
        """The parent order is observable, so it is pinned rather than left to chance.

        The same set of active cells can be stored as different rectangle partitions:
        `_normalize_blocks` merges greedily, so what it produces depends on the order
        the reactivated parents were appended in.  Since flat ids are handed out block
        by block, the partition decides the id of every cell.

        On this mesh the two orders diverge.  Demoting deepest level first leaves the
        two level-1 blocks below; demoting shallowest first leaves the identical 18
        cells as *five* level-1 blocks, and so a different id for most of them.  The
        cell-set invariants of `test_never_destroys_a_cell_it_was_not_given` cannot see
        that difference, which is why this pins the blocks themselves.
        """
        g = _grid_2d(3, 2)
        g = g.refine(0, [0, 1], [2, 3])
        g = g.refine(1, [1, 3], [2, 5])
        g = g.refine(1, [1, 4], [3, 5])
        assert (g.num_cells, g.max_level) == (30, 2)
        g = g.coarsen_cells([c for c in range(g.num_cells) if g.cell_level(c) >= 1])
        assert (g.num_cells, g.max_level) == (18, 1)
        assert g.active_blocks(0) == (((0, 0), (2, 1)), ((1, 1), (2, 2)), ((2, 0), (3, 3)))
        assert g.active_blocks(1) == (((0, 2), (2, 6)), ((2, 4), (4, 6)))

    def test_coarsen_cells_returns_a_grid_with_fresh_caches_and_leaves_the_receivers(self) -> None:
        """Every returned grid starts with empty caches; only the receiver's stay live.

        There is no mutation counter left to tell a no-op apart from a real coarsen at
        the grid level -- `version` counted a mutation, and mutation is unrepresentable
        now that every call returns a fresh object.  The active-leaf set is the oracle
        instead: unchanged after a no-op, different after a real one.
        """
        g = _grid_1d(4, 2)
        g = g.refine_cells([0])
        g.cell_tags.set("test", [0, 1], 1)
        _ = g.cell_bvh()
        children = [c for c in range(g.num_cells) if g.cell_level(c) == 1]

        assert _caches_live(g) == (True, True)
        before = _active_cells(g)

        partial = g.coarsen_cells(children[:1])  # partial parent: nothing is demoted
        assert _active_cells(partial) == before
        assert _caches_live(partial) == (False, False)
        assert _caches_live(g) == (True, True)  # receiver untouched

        full = g.coarsen_cells(children)
        assert _active_cells(full) != before
        assert _caches_live(full) == (False, False)
        assert _caches_live(g) == (True, True)  # receiver still untouched

    def test_destroys_only_the_named_children_after_overlapping_refines(self) -> None:
        """The cell-exact counterpart of the box contrast test above.

        Same reproduction as `TestCoarsenIsNotAnUnconditionalInverse`'s box-overlap test,
        where `coarsen(0, [1], [3])` ends at 7 cells because it demotes the whole box.
        Naming only the children the second refine created gives back the 8-cell state
        that preceded it; naming all four children of level-0 cells 1 and 2 gives the
        box call's 7.  The difference between the two numbers is precisely what the
        caller controls by naming cells.
        """
        g = _grid_1d(6, 2)
        g = g.refine(0, [0], [2])
        after_first = _grid_snapshot(g)
        assert g.num_cells == 8
        g = g.refine(0, [1], [3])  # only cell 2 is still active, so only cell 2 is promoted
        assert g.num_cells == 9
        children_of = {
            parent: [
                c
                for c in range(g.num_cells)
                if g.cell_level(c) == 1 and g.cell_multi_index(c)[0] // 2 == parent
            ]
            for parent in (1, 2)
        }
        assert children_of == {1: [5, 6], 2: [7, 8]}

        undo_second = copy.deepcopy(g)
        undo_second = undo_second.coarsen_cells(children_of[2])
        assert undo_second.num_cells == 8
        assert _grid_snapshot(undo_second) == after_first

        both_parents = copy.deepcopy(g)
        both_parents = both_parents.coarsen_cells(children_of[1] + children_of[2])
        assert both_parents.num_cells == 7
        assert both_parents.active_blocks(0) == (((1,), (6,)),)
        assert both_parents.active_blocks(1) == (((0,), (2,)),)

    def test_ids_spanning_two_levels_are_handled_deepest_first(self) -> None:
        """One call over two levels coarsens the deeper parents before the shallower.

        Level-1 cells 0 and 1 are reborn by the level-2 coarsening, but they were not
        active leaves when the caller named its cells, so they cannot have been named:
        their parent (level-0 cell 0) therefore stays refined.  Coarsening never
        cascades past what the caller could name, which is the point of the method.
        """
        g = _grid_1d(4, 2)
        g = g.refine(0, [0], [2])  # level-1 cells 0..3
        g = g.refine(1, [0], [2])  # level-2 cells 0..3; level-1 leaves are 2, 3
        assert (g.num_cells, g.max_level) == (8, 2)
        g = g.coarsen_cells([c for c in range(g.num_cells) if g.cell_level(c) >= 1])
        assert (g.num_cells, g.max_level) == (5, 1)
        assert g.active_blocks(0) == (((1,), (4,)),)  # cell 0 is still refined
        assert g.active_blocks(1) == (((0,), (2,)),)

    def test_a_shallow_parent_still_goes_when_its_children_are_named(self) -> None:
        """The deepest-first order is not a restriction on which levels may be named."""
        g = _grid_1d(4, 2)
        g = g.refine(0, [0], [2])  # level-1 cells 0..3
        g = g.refine(1, [0], [2])  # level-2 cells 0..3; level-1 leaves are 2, 3
        # Name the level-2 children of level-1 cell 0 and the level-1 children of
        # level-0 cell 1: two parents at two different levels, neither nested.
        marked = [
            c
            for c in range(g.num_cells)
            if (g.cell_level(c) == 2 and g.cell_multi_index(c)[0] < 2)
            or (g.cell_level(c) == 1 and g.cell_multi_index(c)[0] >= 2)
        ]
        g = g.coarsen_cells(marked)
        assert g.active_blocks(0) == (((1,), (4,)),)  # level-0 cell 1 reborn
        assert g.active_blocks(1) == (((0,), (1,)),)  # cell 0 reborn; cell 1 still refined
        assert g.active_blocks(2) == (((2,), (4,)),)  # cell 1's children survive untouched

    @pytest.mark.parametrize(
        ("cells_per_axis", "factor"),
        [([6], 3), ([4, 3], (2, 3)), ([3, 3], 2), ([2, 3, 2], (2, 1, 3))],
        ids=["1d-non-dyadic", "2d-anisotropic", "2d-dyadic", "3d-anisotropic"],
    )
    def test_never_destroys_a_cell_it_was_not_given(
        self,
        cells_per_axis: list[int],
        factor: int | tuple[int, ...],
    ) -> None:
        """The guarantee that separates this from the box call, swept over meshes.

        A randomized walk of refinements and cell-id coarsenings checks two invariants
        after every call: no active leaf outside the named set disappears, and every
        cell that appears is a parent of named cells.  The box `coarsen` fails the
        first of these by design, which is what this method exists to avoid.

        The final tally is what stops a method that silently does nothing from passing
        the two invariants vacuously.  These 60 draws reach three levels and produce
        between 9 and 15 grid-changing coarsenings per parametrization; the floor is
        set well below that so a change in numpy's stream cannot turn a real
        regression into a spurious failure or the reverse.
        """
        root = uniform_grid([[0.0, 1.0]] * len(cells_per_axis), cells_per_axis)
        g = hierarchical_grid(root, factor)
        rng = np.random.default_rng(11)
        coarsened = 0
        for _ in range(60):
            if g.max_level == 0 or rng.random() < 0.5:
                level = int(rng.integers(0, g.max_level + 1))
                extent = g.level_cells_per_axis(level)
                lo = [int(rng.integers(0, n)) for n in extent]
                hi = [
                    int(rng.integers(a + 1, min(a + 3, n) + 1))
                    for a, n in zip(lo, extent, strict=True)
                ]
                g = g.refine(level, lo, hi)
                continue
            pool = [c for c in range(g.num_cells) if g.cell_level(c) >= 1]
            if not pool:
                continue
            pool_idx = rng.choice(
                len(pool), size=int(rng.integers(1, len(pool) + 1)), replace=False
            )
            marked = {
                (g.cell_level(pool[int(i)]), g.cell_multi_index(pool[int(i)])) for i in pool_idx
            }
            before = _active_cells(g)
            g = g.coarsen_cells([pool[int(i)] for i in pool_idx])
            after = _active_cells(g)
            assert before - marked <= after, "removed a cell that was not named"
            for level, midx in after - before:
                child = tuple(m * f for m, f in zip(midx, g.factor, strict=True))
                assert (level + 1, child) in marked, "revived a cell whose children were not named"
            coarsened += before != after
        assert coarsened >= 5, f"the sweep barely coarsened anything ({coarsened} calls changed)"


# ──────────────────────────────────────────────────────────────────────────────
# restrict
# ──────────────────────────────────────────────────────────────────────────────


def _check_restrict(g: HierarchicalGrid, r: GridRestriction, requested: list[int]) -> None:
    """Assert a restriction is internally consistent with the global grid."""
    assert isinstance(r.grid, HierarchicalGrid)
    assert not r.local_to_global_cell.flags.writeable
    assert not r.in_subset.flags.writeable
    sub = r.grid
    l2g = r.local_to_global_cell
    assert l2g.shape == (sub.num_cells,)
    assert len(set(l2g.tolist())) == sub.num_cells  # distinct global ids
    for k in range(sub.num_cells):
        gcid = int(l2g[k])
        lo_s, hi_s = sub.cell_bounds(k)
        lo_g, hi_g = g.cell_bounds(gcid)
        np_testing.assert_allclose(lo_s, lo_g)
        np_testing.assert_allclose(hi_s, hi_g)
        assert sub.cell_level(k) == g.cell_level(gcid)
        center = 0.5 * (lo_s + hi_s)
        assert sub.locate(center) == k
        assert g.locate(center) == gcid
    assert {int(c) for c in l2g[r.in_subset]} == set(requested)


class TestHierarchicalGridRestrict:
    """Tests for HierarchicalGrid.restrict."""

    def test_refined_region_matches_global(self) -> None:
        g = _grid_2d(4, 2)
        g = g.refine(0, [1, 1], [3, 3])
        requested = [c for c in range(g.num_cells) if g.cell_level(c) == 1]
        r = g.restrict(requested)
        _check_restrict(g, r, requested)
        assert r.grid.num_cells == len(requested)  # window == refined region
        assert bool(r.in_subset.all())

    def test_in_subset_flags_fill_cells(self) -> None:
        g = _grid_2d(4, 2)
        g = g.refine(0, [1, 1], [3, 3])
        # Request only the level-0 frame leaves inside root box [0,3)x[0,3).
        requested = [
            c
            for c in range(g.num_cells)
            if g.cell_level(c) == 0 and all(i < 3 for i in g.cell_multi_index(c))
        ]
        r = g.restrict(requested)
        _check_restrict(g, r, requested)
        assert not bool(r.in_subset.all())  # refined cells in the bbox are fill
        assert int(r.in_subset.sum()) == len(requested)

    def test_single_deep_cell_returns_root_subtree(self) -> None:
        g = _grid_2d(4, 2)
        g = g.refine(0, [1, 1], [2, 2])  # refine root cell (1,1) -> 4 level-1 leaves
        fine = [c for c in range(g.num_cells) if g.cell_level(c) == 1]
        r = g.restrict([fine[0]])
        _check_restrict(g, r, [fine[0]])
        assert r.grid.num_cells == 4  # whole root-cell subtree
        assert int(r.in_subset.sum()) == 1

    def test_coarse_only_region_trims_levels(self) -> None:
        g = _grid_2d(4, 2)
        g = g.refine(0, [1, 1], [3, 3])
        # Root row 0 is entirely level-0 frame, disjoint from the refined region.
        requested = [
            c for c in range(g.num_cells) if g.cell_level(c) == 0 and g.cell_multi_index(c)[0] == 0
        ]
        r = g.restrict(requested)
        _check_restrict(g, r, requested)
        sub = r.grid
        assert isinstance(sub, HierarchicalGrid)
        assert sub.max_level == 0  # finer level trimmed away
        assert bool(r.in_subset.all())

    def test_full_grid_is_identity(self) -> None:
        g = _grid_2d(4, 2)
        g = g.refine(0, [1, 1], [3, 3])
        r = g.restrict(list(range(g.num_cells)))
        assert r.grid.num_cells == g.num_cells
        assert set(r.local_to_global_cell.tolist()) == set(range(g.num_cells))
        assert bool(r.in_subset.all())
        _check_restrict(g, r, list(range(g.num_cells)))

    def test_full_grid_neighbors_match(self) -> None:
        g = _grid_2d(4, 2)
        g = g.refine(0, [1, 1], [3, 3])
        r = g.restrict(list(range(g.num_cells)))
        sub, l2g = r.grid, r.local_to_global_cell
        for k in range(sub.num_cells):
            for lfid in range(2 * sub.ndim):
                ns = sub.neighbor_across_facet(k, lfid)
                ng = g.neighbor_across_facet(int(l2g[k]), lfid)
                assert (None if ns is None else int(l2g[ns])) == ng

    def test_window_neighbors_map_to_global(self) -> None:
        g = _grid_2d(4, 2)
        g = g.refine(0, [1, 1], [3, 3])
        requested = [c for c in range(g.num_cells) if g.cell_level(c) == 1]
        r = g.restrict(requested)
        sub, l2g = r.grid, r.local_to_global_cell
        for k in range(sub.num_cells):
            for lfid in range(2 * sub.ndim):
                ns = sub.neighbor_across_facet(k, lfid)
                if ns is not None:
                    assert int(l2g[ns]) == g.neighbor_across_facet(int(l2g[k]), lfid)

    def test_sub_root_not_reclamped(self) -> None:
        g = _grid_2d(4, 2)  # root breakpoints [0, .25, .5, .75, 1]
        g = g.refine(0, [1, 1], [3, 3])
        requested = [c for c in range(g.num_cells) if g.cell_level(c) == 1]
        sub = g.restrict(requested).grid
        assert isinstance(sub, HierarchicalGrid)
        np_testing.assert_allclose(sub.root.breakpoints[0], [0.25, 0.5, 0.75])

    def test_1d(self) -> None:
        g = _grid_1d(4, 2)
        g = g.refine(0, [1], [3])
        requested = [c for c in range(g.num_cells) if g.cell_level(c) == 1]
        r = g.restrict(requested)
        _check_restrict(g, r, requested)
        assert r.grid.ndim == 1

    def test_returns_grid_restriction(self) -> None:
        g = _grid_2d(4, 2)
        g = g.refine(0, [1, 1], [3, 3])
        r = g.restrict([0, 1, 2])
        assert isinstance(r, GridRestriction)
        assert isinstance(r.grid, HierarchicalGrid)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _grid_2d(2, 2).restrict([])

    def test_out_of_range_raises(self) -> None:
        g = _grid_2d(2, 2)
        with pytest.raises(IndexError):
            g.restrict([g.num_cells])
        with pytest.raises(IndexError):
            g.restrict([-1])

    def test_non_integer_raises(self) -> None:
        with pytest.raises(TypeError, match="integer"):
            _grid_2d(2, 2).restrict([0.0, 1.0])

    def test_multilevel_restrict(self) -> None:
        # Two levels of refinement: level 0 → level 1 → level 2.
        g = _grid_2d(4, 2)
        g = g.refine(0, [1, 1], [3, 3])  # level-1 leaves in [1,3)x[1,3)
        # Refine one level-1 block to level 2.
        l1_cells = [c for c in range(g.num_cells) if g.cell_level(c) == 1]
        l1_midxs = [g.cell_multi_index(c) for c in l1_cells]
        lo = min(m[0] for m in l1_midxs)
        g = g.refine(1, [lo, lo], [lo + 1, lo + 1])
        l2_cells = [c for c in range(g.num_cells) if g.cell_level(c) == 2]
        assert len(l2_cells) > 0
        r = g.restrict([l2_cells[0]])
        _check_restrict(g, r, [l2_cells[0]])
        assert int(r.in_subset.sum()) == 1

    def test_non_contiguous_ids(self) -> None:
        # Cells from two disjoint corners force a large bounding box with fill cells.
        g = _grid_2d(6, 2)
        # Top-left corner root cell (0,0) and bottom-right corner root cell (5,5).
        corner_cells = [
            c
            for c in range(g.num_cells)
            if g.cell_level(c) == 0
            and (tuple(g.cell_multi_index(c)) == (0, 0) or tuple(g.cell_multi_index(c)) == (5, 5))
        ]
        assert len(corner_cells) == 2
        r = g.restrict(corner_cells)
        _check_restrict(g, r, corner_cells)
        # Bounding box spans the full 6x6 root; in_subset flags only the two corners.
        assert int(r.in_subset.sum()) == 2
        assert r.grid.num_cells > 2  # fill cells included in bbox


# ──────────────────────────────────────────────────────────────────────────────
# Numba kernel backing (locate_many / collect_cell_bounds / encode / decode)
# ──────────────────────────────────────────────────────────────────────────────


def _irregular_grid(ndim: int, factor: int | tuple[int, ...]) -> HierarchicalGrid:
    """Multi-level grid on irregular breakpoints with disjoint refined regions."""
    rng = np.random.default_rng(5 + ndim)
    bp = [np.sort(np.concatenate([[0.0, 1.0], rng.random(5)])) for _ in range(ndim)]
    g = hierarchical_grid(TensorProductGrid(bp), factor)
    for lev in range(3):
        n = g.level_cells_per_axis(lev)
        g = g.refine(lev, [0] * ndim, [max(1, n[k] // 2) for k in range(ndim)])
    n1 = g.level_cells_per_axis(1)
    g = g.refine(1, [max(0, n1[k] - 2) for k in range(ndim)], list(n1))
    return g


class TestHierKernelEquivalence:
    """Kernel-backed batch methods agree exactly with the scalar reference paths."""

    @pytest.mark.parametrize(("ndim", "factor"), [(1, 2), (2, 2), (2, (2, 3)), (3, 2)])
    def test_collect_cell_bounds_matches_scalar(
        self, ndim: int, factor: int | tuple[int, ...]
    ) -> None:
        """collect_cell_bounds is bitwise-identical to per-cell cell_bounds."""
        g = _irregular_grid(ndim, factor)
        assert g.max_level == 3
        lo_all, hi_all = g.collect_cell_bounds()
        for cid in range(g.num_cells):
            lo, hi = g.cell_bounds(cid)
            np_testing.assert_array_equal(lo_all[cid], lo)
            np_testing.assert_array_equal(hi_all[cid], hi)

    @pytest.mark.parametrize(("ndim", "factor"), [(1, 2), (2, 2), (2, (2, 3)), (3, 2)])
    def test_locate_many_matches_scalar_locate(
        self, ndim: int, factor: int | tuple[int, ...]
    ) -> None:
        """locate_many agrees with the per-point scalar locate on every point class."""
        g = _irregular_grid(ndim, factor)
        rng = np.random.default_rng(11)
        pts = rng.random((4000, ndim)) * 1.3 - 0.15  # interior + outside points
        lo_all, hi_all = g.collect_cell_bounds()
        # Cell corners exercise breakpoint / level-interface ties.
        pts = np.concatenate([pts, lo_all[:64], hi_all[:64]], axis=0)
        got = g.locate_many(pts)
        expected = np.array([-1 if (c := g.locate(p)) is None else c for p in pts], dtype=np.int64)
        np_testing.assert_array_equal(got, expected)

    def test_locate_many_nonfinite_points(self) -> None:
        """NaN / infinite coordinates map to -1."""
        g = _grid_2d()
        pts = np.array([[np.nan, 0.5], [np.inf, 0.5], [0.5, -np.inf], [0.5, 0.5]])
        np_testing.assert_array_equal(g.locate_many(pts)[:3], [-1, -1, -1])
        assert g.locate_many(pts)[3] >= 0

    @pytest.mark.parametrize(("ndim", "factor"), [(1, 2), (2, 2), (2, (2, 3)), (3, 2)])
    def test_decode_encode_roundtrip(self, ndim: int, factor: int | tuple[int, ...]) -> None:
        """_decode_flat_id and _encode_midx are mutually inverse over all cells."""
        g = _irregular_grid(ndim, factor)
        for cid in range(g.num_cells):
            level, midx = g._decode_flat_id(cid)
            assert g._encode_midx(level, midx) == cid

    def test_encode_midx_inactive_positions(self) -> None:
        """_encode_midx returns None for refined (non-leaf) and never-active positions."""
        g = _grid_2d(4)
        g = g.refine(0, [0, 0], [2, 2])
        # (0, (0, 0)) was refined away -> not an active leaf.
        assert g._encode_midx(0, (0, 0)) is None
        # A level beyond the hierarchy.
        assert g._encode_midx(5, (0, 0)) is None
        # Level-1 position outside the refined region is not active.
        assert g._encode_midx(1, (7, 7)) is None

    def test_decode_out_of_range_raises(self) -> None:
        """_decode_flat_id rejects out-of-range ids."""
        g = _grid_2d(4)
        with pytest.raises(IndexError, match="out of range"):
            g._decode_flat_id(g.num_cells)
        with pytest.raises(IndexError, match="out of range"):
            g._decode_flat_id(-1)

    def test_kernel_state_tracks_mutation(self) -> None:
        """Packed kernel arrays are rebuilt by refine/coarsen (results stay exact)."""
        g = _grid_2d(8)
        rng = np.random.default_rng(17)
        pts = rng.random((500, 2))
        g = g.refine(0, [0, 0], [4, 4])
        after_refine = g.locate_many(pts)
        expected = np.array([-1 if (c := g.locate(p)) is None else c for p in pts])
        np_testing.assert_array_equal(after_refine, expected)
        g = g.coarsen(0, [0, 0], [4, 4])
        after_coarsen = g.locate_many(pts)
        expected2 = np.array([-1 if (c := g.locate(p)) is None else c for p in pts])
        np_testing.assert_array_equal(after_coarsen, expected2)

    def test_restricted_grid_uses_fresh_packed_arrays(self) -> None:
        """Grids built via restrict() (the _from_blocks path) locate correctly."""
        g = _grid_2d(8)
        g = g.refine(0, [0, 0], [4, 4])
        restr = g.restrict(np.arange(min(20, g.num_cells)))
        sub = restr.grid
        rng = np.random.default_rng(23)
        pts = rng.random((300, 2))
        got = sub.locate_many(pts)
        expected = np.array([-1 if (c := sub.locate(p)) is None else c for p in pts])
        np_testing.assert_array_equal(got, expected)

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_locate_many_empty_input(self, ndim: int) -> None:
        """locate_many with zero rows returns a shape-(0,) array without error."""
        g = _irregular_grid(ndim, 2)
        out = g.locate_many(np.empty((0, ndim), dtype=np.float64))
        assert out.shape == (0,)
        assert out.dtype == np.int64

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_locate_many_single_point_1d_input(self, ndim: int) -> None:
        """A 1-D array (single point) is promoted to (1, ndim) and located."""
        g = _irregular_grid(ndim, 2)
        pt = np.full(ndim, 0.5)
        out = g.locate_many(pt)
        assert out.shape == (1,)
        expected = g.locate(pt)
        assert out[0] == (-1 if expected is None else expected)

    def test_collect_cell_bounds_unrefined(self) -> None:
        """collect_cell_bounds on a flat (level-0 only) grid matches per-cell bounds."""
        g = _grid_2d(4)  # 4x4 uniform grid, no refinement
        assert g.max_level == 0
        lo_all, hi_all = g.collect_cell_bounds()
        for cid in range(g.num_cells):
            lo, hi = g.cell_bounds(cid)
            np_testing.assert_array_equal(lo_all[cid], lo)
            np_testing.assert_array_equal(hi_all[cid], hi)


# ──────────────────────────────────────────────────────────────────────────────
# Boundary facets
# ──────────────────────────────────────────────────────────────────────────────


def _boundary_facets_bruteforce(grid: HierarchicalGrid) -> list[list[int]]:
    """Reference enumeration: one `is_mesh_boundary_facet` query per cell and facet.

    Independent of the block-vectorized implementation under test, which decides a
    whole block face at once and never queries a neighbour.
    """
    return [
        [cid, lfid]
        for cid in range(grid.num_cells)
        for lfid in range(grid.num_local_facets(cid))
        if grid.is_mesh_boundary_facet(cid, lfid)
    ]


def _refined_corner_grid() -> HierarchicalGrid:
    """2x2 unit grid with the low corner root cell refined once: 3 coarse + 4 fine."""
    g = hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 2), 2)
    g = g.refine(0, (0, 0), (1, 1))
    return g


class TestBoundaryFacets:
    """boundary_facets enumerates the outer boundary only, and agrees with the predicate."""

    @pytest.mark.parametrize(
        ("ndim", "factor"), [(1, 2), (2, 2), (2, (2, 3)), (3, 2), (1, 3), (2, 3)]
    )
    def test_matches_predicate(self, ndim: int, factor: int | tuple[int, ...]) -> None:
        """The block-vectorized enumeration equals the per-facet predicate, exactly."""
        g = _irregular_grid(ndim, factor)
        rows = g.boundary_facets()

        assert rows.tolist() == _boundary_facets_bruteforce(g)
        assert rows.shape[0] > 0  # not vacuously empty
        assert rows.dtype == np.int64
        assert rows.shape[1] == 2
        assert not rows.flags.writeable
        # Lexicographic in (cid, lfid).
        np_testing.assert_array_equal(rows[np.lexsort((rows[:, 1], rows[:, 0]))], rows)

    @pytest.mark.parametrize(("ndim", "factor"), [(1, 2), (2, 2), (3, 2)])
    def test_override_agrees_with_abc_default(
        self, ndim: int, factor: int | tuple[int, ...]
    ) -> None:
        """The specialized override returns exactly what the inherited default would."""
        g = _irregular_grid(ndim, factor)
        np_testing.assert_array_equal(g.boundary_facets(), Grid.boundary_facets(g))

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_unrefined_count_is_analytic(self, ndim: int) -> None:
        """A flat n^ndim grid has 2 * ndim * n^(ndim-1) boundary facets."""
        n = 3
        g = hierarchical_grid(uniform_grid([[0.0, 1.0]] * ndim, n), 2)
        assert g.boundary_facets().shape[0] == 2 * ndim * n ** (ndim - 1)

    def test_excludes_level_interfaces(self) -> None:
        """Level-interface facets are excluded, and the outer count is the analytic one."""
        g = _refined_corner_grid()
        rows = g.boundary_facets()
        assert g.num_cells == 7

        # 4 sides x 2 root cells = 8 coarse facets; the refined quadrant touches 2 of
        # those sides and on each *replaces* its coarse facet by 2 fine ones: 8 - 2 + 4.
        assert rows.shape[0] == 10

        # Every reported facet has no neighbour at all — a stricter check than the
        # predicate, since it would also catch a facet with hanging fine neighbours.
        for cid, lfid in rows.tolist():
            assert g.hanging_neighbors(cid, lfid) == ()

        # And the level interface, which does exist here, is entirely absent.
        interface = {
            (cid, lfid)
            for cid in range(g.num_cells)
            for lfid in range(g.num_local_facets(cid))
            for nbrs in [g.hanging_neighbors(cid, lfid)]
            if nbrs and any(g.cell_level(n) != g.cell_level(cid) for n in nbrs)
        }
        assert interface
        assert not interface & {tuple(row) for row in rows.tolist()}


# ──────────────────────────────────────────────────────────────────────────────
# Leaf-cell mesh export
# ──────────────────────────────────────────────────────────────────────────────

_CORNER_RTOL = 8.0 * np.finfo(np.float64).eps
"""Bound on ``|export_cells corner - cell_bounds corner|`` relative to the coordinate.

`cell_bounds` builds a corner as ``bp + s * (width / factor**level)``, with one more
addition for a ``hi`` corner; `export_cells` builds the same value as
``bp + o * (width / factor**max_level)``. Same quantity, different expression, so up to
four correctly-rounded operations per side, each at most ``eps/2`` relative to a
coordinate that dominates its own offset: ``<= 7 * eps / 2``, i.e. under ``4 * eps``.
Measured worst case over the cases below is 1 ulp; the 8 leaves a 2x margin.
"""


def _cell_corners(grid: HierarchicalGrid, cid: int) -> list[np.ndarray]:
    """Corners of `cell_bounds(cid)` in the export corner order (axis 0 = LSB)."""
    lo, hi = grid.cell_bounds(cid)
    return [
        np.array(
            [hi[k] if (corner >> k) & 1 else lo[k] for k in range(grid.ndim)], dtype=np.float64
        )
        for corner in range(1 << grid.ndim)
    ]


class TestExportCells:
    """export_cells deduplicates leaf corners exactly and reproduces the cell bounds."""

    @pytest.mark.parametrize(("ndim", "factor"), [(1, 2), (2, 2), (2, (2, 3)), (3, 2), (2, 3)])
    def test_roundtrip_matches_cell_bounds(self, ndim: int, factor: int | tuple[int, ...]) -> None:
        """points[conn[cid]] reproduces every cell's corners within the derived bound."""
        g = _irregular_grid(ndim, factor)
        points, conn = g.export_cells()

        assert conn.shape == (g.num_cells, 1 << ndim)
        assert points.shape[1] == ndim
        assert points.dtype == np.float64
        assert conn.dtype == np.int64
        assert not points.flags.writeable
        assert not conn.flags.writeable
        assert int(conn.min()) >= 0
        assert int(conn.max()) == points.shape[0] - 1

        for cid in range(g.num_cells):
            for corner, want in enumerate(_cell_corners(g, cid)):
                np_testing.assert_allclose(
                    points[conn[cid, corner]], want, rtol=_CORNER_RTOL, atol=0.0
                )

    def test_roundtrip_exact_on_dyadic_grid(self) -> None:
        """With dyadic breakpoints and a dyadic factor every corner agrees bitwise."""
        g = hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 2), 2)
        g = g.refine(0, (0, 0), (1, 1))
        g = g.refine(1, (0, 0), (2, 2))
        points, conn = g.export_cells()

        for cid in range(g.num_cells):
            for corner, want in enumerate(_cell_corners(g, cid)):
                np_testing.assert_array_equal(points[conn[cid, corner]], want)

    def test_dedup_exact_with_hanging_vertices(self) -> None:
        """The refined-corner example: 14 distinct vertices, hanging ones counted once."""
        g = _refined_corner_grid()
        points, conn = g.export_cells()

        assert conn.shape == (7, 4)
        # The 9 root corners of the 2x2 grid, plus the 5 level-1 nodes inside the
        # refined quadrant: 2 on the domain boundary, the quadrant centre, and the 2
        # hanging nodes on the level interface.
        assert points.shape == (14, 2)
        got = {tuple(p) for p in points.tolist()}
        assert len(got) == 14
        assert got == {
            (0.0, 0.0), (0.0, 0.25), (0.0, 0.5), (0.0, 1.0),
            (0.25, 0.0), (0.25, 0.25), (0.25, 0.5),
            (0.5, 0.0), (0.5, 0.25), (0.5, 0.5), (0.5, 1.0),
            (1.0, 0.0), (1.0, 0.5), (1.0, 1.0),
        }  # fmt: skip

    def test_dedup_exact_with_non_uniform_breakpoints(self) -> None:
        """Non-uniform root breakpoints deduplicate exactly as well."""
        g = hierarchical_grid(TensorProductGrid([np.array([0.0, 0.3, 1.0])] * 2), 2)
        g = g.refine(0, (0, 0), (1, 1))
        points, _ = g.export_cells()

        assert points.shape == (14, 2)
        assert len({tuple(p) for p in points.tolist()}) == 14
        # 0.15 is the level-1 node inside [0, 0.3]; it must appear exactly once.
        assert [tuple(p) for p in points.tolist()].count((0.15, 0.0)) == 1

    def test_shared_vertex_referenced_by_every_touching_cell(self) -> None:
        """An interior vertex is one index, referenced once per cell that touches it."""
        g = hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 2), 2)
        points, conn = g.export_cells()

        centre = int(np.flatnonzero((points == 0.5).all(axis=1))[0])
        assert int((conn == centre).sum()) == 4  # all four root cells meet there

    def test_corner_order_convention(self) -> None:
        """Corner c takes the hi bound on axis k iff bit k of c is set (axis 0 = LSB)."""
        # Deliberately anisotropic bounds, so a transposed convention cannot pass.
        g = hierarchical_grid(TensorProductGrid([np.array([0.0, 1.0]), np.array([0.0, 2.0])]), 2)
        points, conn = g.export_cells()

        assert points[conn[0]].tolist() == [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 2.0]]

    def test_unrefined_grid_matches_tensor_product_lattice(self) -> None:
        """On a flat n^ndim grid the vertex set is the full (n+1)^ndim breakpoint lattice."""
        n = 3
        g = hierarchical_grid(uniform_grid([[0.0, 1.0]] * 3, n), 2)
        points, conn = g.export_cells()

        assert points.shape == ((n + 1) ** 3, 3)
        assert conn.shape == (n**3, 8)
