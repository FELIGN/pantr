/// \file
/// Hierarchical cell addressing and point location.
///
/// Nothing here compares the two backends. The checks are properties that tie the
/// kernels to each other and to the geometry, which is what makes them an oracle
/// rather than a second copy of the code:
///
///  - **collect and locate agree.** Locating the midpoint of every cell that
///    `hier_collect_cell_bounds` produces must return that cell's own flat id.
///    Neither kernel can satisfy this by mirroring the other -- one walks blocks
///    and expands an odometer, the other descends levels and truncates a quotient.
///  - **the cells tile the domain.** Sorted by lower corner they must abut
///    exactly, with no gap and no overlap. An exact assertion, not a close one:
///    the shared corner is produced by the same expression on both sides.
///  - **decode inverts encode.** Round-tripping every flat id through
///    `decode_flat_id` and back through `block_of_midx` must return it unchanged.
///
/// The two grids below are worked out by hand in the comments so that a reader can
/// check the fixture itself rather than trusting it.

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include "check.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/grid/hierarchical.hpp"

namespace {

using pantr::span2d;
using pantr::grid::block_of_midx;
using pantr::grid::decode_flat_id;
using pantr::grid::hier_collect_cell_bounds;
using pantr::grid::hier_locate_points;

/// A hand-built hierarchical grid, 1-D.
struct Grid1D {
    std::vector<double> knots;          // root breakpoints
    std::vector<std::int64_t> starts;   // per-axis offset into knots
    std::vector<std::int64_t> root_cells;
    std::vector<std::int64_t> factor;
    std::vector<std::int64_t> blo, bhi;  // (n_blocks, 1)
    std::vector<std::int64_t> base;
    std::vector<std::int64_t> lbs;  // level_block_start
    std::int64_t num_cells;

    [[nodiscard]] std::size_t n_blocks() const { return base.size(); }
};

/// One level, two root cells, both active. Cells: [0,1] and [1,2].
Grid1D flat_grid() {
    return Grid1D{{0.0, 1.0, 2.0}, {0}, {2}, {2}, {0}, {2}, {0}, {0, 1}, 2};
}

/// Two levels. Root cell 1 stays coarse; root cell 0 splits in two.
///
/// level 0: one block over level-0 indices [1,2)  -> flat id 0, the cell [1,2]
/// level 1: one block over level-1 indices [0,2)  -> flat ids 1,2, cells
///          [0,0.5] and [0.5,1]
Grid1D refined_grid() {
    return Grid1D{{0.0, 1.0, 2.0}, {0}, {2}, {2}, {1, 0}, {2, 2}, {0, 1}, {0, 1, 2}, 3};
}

std::vector<std::int64_t> locate(const Grid1D& g, const std::vector<double>& xs) {
    std::vector<std::int64_t> out(xs.size(), -99);
    hier_locate_points<double>(
        span2d<const double>(xs.data(), xs.size(), 1), std::span<const double>(g.knots),
        std::span<const std::int64_t>(g.starts), std::span<const std::int64_t>(g.root_cells),
        std::span<const std::int64_t>(g.factor),
        span2d<const std::int64_t>(g.blo.data(), g.n_blocks(), 1),
        span2d<const std::int64_t>(g.bhi.data(), g.n_blocks(), 1),
        std::span<const std::int64_t>(g.base), std::span<const std::int64_t>(g.lbs),
        std::span<std::int64_t>(out));
    return out;
}

std::pair<std::vector<double>, std::vector<double>> collect(const Grid1D& g) {
    std::vector<double> lo(static_cast<std::size_t>(g.num_cells), -99.0);
    std::vector<double> hi(lo.size(), -99.0);
    hier_collect_cell_bounds<double>(
        std::span<const double>(g.knots), std::span<const std::int64_t>(g.starts),
        std::span<const std::int64_t>(g.factor),
        span2d<const std::int64_t>(g.blo.data(), g.n_blocks(), 1),
        span2d<const std::int64_t>(g.bhi.data(), g.n_blocks(), 1),
        std::span<const std::int64_t>(g.base), std::span<const std::int64_t>(g.lbs),
        span2d<double>(lo.data(), lo.size(), 1), span2d<double>(hi.data(), hi.size(), 1));
    return {lo, hi};
}

/// The fixture itself, checked against the hand computation in its comment.
void the_fixtures_are_what_they_claim() {
    const auto [lo, hi] = collect(refined_grid());
    PANTR_CHECK_MSG(lo[0] == 1.0 && hi[0] == 2.0, "flat id 0 is the coarse cell [1,2]");
    PANTR_CHECK_MSG(lo[1] == 0.0 && hi[1] == 0.5, "flat id 1 is [0,0.5]");
    PANTR_CHECK_MSG(lo[2] == 0.5 && hi[2] == 1.0, "flat id 2 is [0.5,1]");
}

/// Locating the midpoint of every collected cell returns that cell's own id.
void collect_and_locate_agree(const Grid1D& g, const char* what) {
    const auto [lo, hi] = collect(g);
    std::vector<double> mids(lo.size());
    for (std::size_t i = 0; i < lo.size(); ++i) {
        mids[i] = 0.5 * (lo[i] + hi[i]);
    }
    const auto got = locate(g, mids);
    for (std::size_t i = 0; i < mids.size(); ++i) {
        PANTR_CHECK_MSG(got[i] == static_cast<std::int64_t>(i),
                        std::string("the midpoint of cell ") + std::to_string(i) + " of " + what
                            + " located to " + std::to_string(got[i]));
    }
}

/// The collected cells tile the root domain exactly: no gap, no overlap.
void the_cells_tile_the_domain(const Grid1D& g, const char* what) {
    auto [lo, hi] = collect(g);
    std::vector<std::size_t> order(lo.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&lo](std::size_t a, std::size_t b) {
        return lo[a] < lo[b];
    });
    PANTR_CHECK_MSG(lo[order.front()] == g.knots.front(),
                    std::string("the first cell of ") + what + " must start at the domain");
    PANTR_CHECK_MSG(hi[order.back()] == g.knots.back(),
                    std::string("the last cell of ") + what + " must end at the domain");
    for (std::size_t i = 1; i < order.size(); ++i) {
        PANTR_CHECK_MSG(hi[order[i - 1]] == lo[order[i]],
                        std::string("cells of ") + what + " must abut exactly");
    }
}

/// decode_flat_id inverts block_of_midx for every id in the grid.
void decode_inverts_encode(const Grid1D& g, const char* what) {
    for (std::int64_t cid = 0; cid < g.num_cells; ++cid) {
        std::vector<std::int64_t> midx(1, -99);
        const std::int64_t level =
            decode_flat_id(cid, span2d<const std::int64_t>(g.blo.data(), g.n_blocks(), 1),
                           span2d<const std::int64_t>(g.bhi.data(), g.n_blocks(), 1),
                           std::span<const std::int64_t>(g.base),
                           std::span<const std::int64_t>(g.lbs), std::span<std::int64_t>(midx));
        const std::int64_t back =
            block_of_midx(level, std::span<const std::int64_t>(midx),
                          span2d<const std::int64_t>(g.blo.data(), g.n_blocks(), 1),
                          span2d<const std::int64_t>(g.bhi.data(), g.n_blocks(), 1),
                          std::span<const std::int64_t>(g.base),
                          std::span<const std::int64_t>(g.lbs));
        PANTR_CHECK_MSG(back == cid, std::string("round trip of id ") + std::to_string(cid)
                                         + " in " + what + " gave " + std::to_string(back));
    }
}

/// A position that is not an active leaf is -1, not a neighbouring cell's id.
void an_inactive_position_is_minus_one() {
    const Grid1D g = refined_grid();
    // level 0, index 0: that root cell was refined away, so it is not active.
    const std::vector<std::int64_t> midx{0};
    const std::int64_t got =
        block_of_midx(0, std::span<const std::int64_t>(midx),
                      span2d<const std::int64_t>(g.blo.data(), g.n_blocks(), 1),
                      span2d<const std::int64_t>(g.bhi.data(), g.n_blocks(), 1),
                      std::span<const std::int64_t>(g.base), std::span<const std::int64_t>(g.lbs));
    PANTR_CHECK_MSG(got == -1, "a refined-away position must not resolve to a cell");
}

/// Outside the root domain is -1 on either side, and the corners are inside.
void outside_the_root_domain_is_minus_one() {
    const Grid1D g = refined_grid();
    const auto got = locate(g, {-1e-12, 2.0 + 1e-12, 0.0, 2.0});
    PANTR_CHECK(got[0] == -1);
    PANTR_CHECK(got[1] == -1);
    PANTR_CHECK_MSG(got[2] == 1, "the lower corner is in the first refined cell");
    PANTR_CHECK_MSG(got[3] == 0, "the upper corner is in the coarse cell");
}

/// An interior breakpoint of the refined level takes the lower child.
///
/// The descent's truncation is a discrete verdict, so the tie gets its own case.
void a_child_boundary_takes_the_lower_child() {
    const Grid1D g = refined_grid();
    const auto got = locate(g, {0.5});
    PANTR_CHECK_MSG(got[0] == 2, "x = 0.5 is the lower corner of cell 2, so it takes cell 2");
}

}  // namespace

int main() {
    the_fixtures_are_what_they_claim();
    collect_and_locate_agree(flat_grid(), "the flat grid");
    collect_and_locate_agree(refined_grid(), "the refined grid");
    the_cells_tile_the_domain(flat_grid(), "the flat grid");
    the_cells_tile_the_domain(refined_grid(), "the refined grid");
    decode_inverts_encode(flat_grid(), "the flat grid");
    decode_inverts_encode(refined_grid(), "the refined grid");
    an_inactive_position_is_minus_one();
    outside_the_root_domain_is_minus_one();
    a_child_boundary_takes_the_lower_child();
    return pantr::test::summary("test_grid_hierarchical");
}
