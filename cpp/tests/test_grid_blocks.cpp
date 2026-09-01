/// \file
/// The rectangle algebra a hierarchical grid's active set is made of.
///
/// ## What is being defended, and it is not "the answer is right"
///
/// Every quantity here is an integer, so there is no rounding, no tolerance and no
/// regime where the answer is nearly correct. What can go wrong is one thing:
/// `normalize_blocks` produces a *different partition* from the Python oracle's, of the
/// same cells. That is not a wrong answer in any local sense -- both partitions cover
/// exactly the same set and both are pairwise non-mergeable -- and it would pass every
/// property test one could write about a partition. It would still be a defect, because
/// flat cell ids are handed out block by block, so the two backends would then number
/// the same cells differently with everything else identical.
///
/// So the assertions split three ways, and the middle group is the one that exists
/// because the first cannot do the job:
///
///  - **Hand-computed values**, for the pieces small enough to have them: `block_size`,
///    `rect_intersect`, `try_merge` and `peel` on cases worked out in the comments.
///  - **The order-dependence contract**, pinned on the smallest input that exhibits it.
///    An L-shaped tromino of three unit cells normalises to two different partitions
///    depending on which order it arrives in, and both are written out here. A
///    reimplementation that made normalisation canonical -- the tempting "improvement"
///    -- would pass every other test in this file and fail these two.
///  - **Properties over a random sweep**, where a hand case would only ever cover the
///    shapes someone thought of: idempotence, cell-count preservation, and pairwise
///    non-mergeability of the output. The sweep is an order of magnitude larger than the
///    oracle's own, which runs 400 decompositions.
///
/// **Nothing here compares the two backends.** That job is `tests/parity/`, which can
/// call both. What this file can do, and does, is pin the properties the parity claim
/// rests on, so that a divergence shows up as a specific broken property rather than as
/// a bare inequality in a sweep.
///
/// The two hand-pinned partitions were read off the oracle rather than invented, which
/// is worth saying because they look arbitrary: they are, and that is the point --
/// `normalize_blocks`'s output is a function of its input order, and these are the two
/// values that function takes.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "check.hpp"
#include "pantr/grid/blocks.hpp"

namespace {

using pantr::grid::BlockList;
using pantr::grid::BlockView;
using pantr::grid::block_size;
using pantr::grid::normalize_blocks;
using pantr::grid::peel;
using pantr::grid::rect_intersect;
using pantr::grid::try_merge;

/// One rectangle as `{lo..., hi...}`, for writing fixtures out by hand.
using Corners = std::vector<std::int64_t>;

/// Build a list from rectangles spelled `{lo0, lo1, ..., hi0, hi1, ...}`.
///
/// \param ndim Axis count.
/// \param rects The rectangles, each `2 * ndim` entries.
/// \return The list, in the order given -- which `normalize_blocks` is sensitive to.
BlockList make(std::int64_t ndim, const std::vector<Corners>& rects) {
    BlockList blocks(ndim);
    const auto d = static_cast<std::size_t>(ndim);
    for (const Corners& rect : rects) {
        blocks.push_back(BlockView{std::span<const std::int64_t>(rect.data(), d),
                                   std::span<const std::int64_t>(rect.data() + d, d)});
    }
    return blocks;
}

/// Flatten a list back to `{lo..., hi...}` per rectangle, for comparison.
///
/// \param blocks The list to flatten.
/// \return One `Corners` per rectangle, in list order.
std::vector<Corners> flatten(const BlockList& blocks) {
    std::vector<Corners> out;
    out.reserve(blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        Corners rect(blocks[i].lo.begin(), blocks[i].lo.end());
        rect.insert(rect.end(), blocks[i].hi.begin(), blocks[i].hi.end());
        out.push_back(std::move(rect));
    }
    return out;
}

/// Total cell count over a list.
///
/// \param blocks The list.
/// \return The sum of every rectangle's cell count.
std::int64_t total_cells(const BlockList& blocks) {
    std::int64_t total = 0;
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        total += block_size(blocks[i]);
    }
    return total;
}

// ---------------------------------------------------------------------------
// The pieces, on hand-computed values
// ---------------------------------------------------------------------------

/// `block_size` multiplies the extents, and is 1 over zero axes.
void test_block_size() {
    const BlockList one_d = make(1, {{2, 7}});
    PANTR_CHECK(block_size(one_d[0]) == 5);

    const BlockList three_d = make(3, {{0, 1, 2, 3, 3, 5}});  // extents 3, 2, 3
    PANTR_CHECK(block_size(three_d[0]) == 18);

    // A single cell, which is what every decomposition in the sweep is made of.
    PANTR_CHECK(block_size(make(2, {{4, 5, 5, 6}})[0]) == 1);
}

/// Intersection is the per-axis max/min, and touching counts as disjoint.
void test_rect_intersect() {
    Corners lo(2);
    Corners hi(2);

    const BlockList a = make(2, {{0, 0, 4, 4}});
    const BlockList b = make(2, {{2, 1, 6, 3}});
    PANTR_CHECK(rect_intersect(a[0], b[0], lo, hi));
    PANTR_CHECK(lo == Corners({2, 1}));
    PANTR_CHECK(hi == Corners({4, 3}));

    // Symmetric in its two arguments, which is what lets the signature take them
    // adjacent without a transposition hazard.
    Corners lo2(2);
    Corners hi2(2);
    PANTR_CHECK(rect_intersect(b[0], a[0], lo2, hi2));
    PANTR_CHECK(lo2 == lo);
    PANTR_CHECK(hi2 == hi);

    // Face-to-face is EMPTY, not a zero-width rectangle: `[0,4)` and `[4,8)` share no
    // cell. The half-open convention is what makes that the right answer, and getting it
    // wrong would leak an empty block into a level's list.
    const BlockList touching = make(2, {{4, 0, 8, 4}});
    PANTR_CHECK(!rect_intersect(a[0], touching[0], lo, hi));

    // Disjoint on one axis only is still disjoint.
    const BlockList apart = make(2, {{0, 9, 4, 12}});
    PANTR_CHECK(!rect_intersect(a[0], apart[0], lo, hi));

    // Containment gives the inner rectangle back.
    const BlockList inner = make(2, {{1, 1, 2, 2}});
    PANTR_CHECK(rect_intersect(a[0], inner[0], lo, hi));
    PANTR_CHECK(lo == Corners({1, 1}));
    PANTR_CHECK(hi == Corners({2, 2}));
}

/// Merging needs agreement on every axis but one, and adjacency on that one.
void test_try_merge() {
    Corners lo(2);
    Corners hi(2);

    // Adjacent along axis 0, in both argument orders.
    const BlockList left = make(2, {{0, 0, 2, 3}});
    const BlockList right = make(2, {{2, 0, 5, 3}});
    PANTR_CHECK(try_merge(left[0], right[0], lo, hi));
    PANTR_CHECK(lo == Corners({0, 0}));
    PANTR_CHECK(hi == Corners({5, 3}));
    PANTR_CHECK(try_merge(right[0], left[0], lo, hi));
    PANTR_CHECK(lo == Corners({0, 0}));
    PANTR_CHECK(hi == Corners({5, 3}));

    // Adjacent along axis 1.
    const BlockList below = make(2, {{0, 3, 2, 6}});
    PANTR_CHECK(try_merge(left[0], below[0], lo, hi));
    PANTR_CHECK(lo == Corners({0, 0}));
    PANTR_CHECK(hi == Corners({2, 6}));

    // Aligned on neither: differs on two axes.
    const BlockList diagonal = make(2, {{2, 3, 5, 6}});
    PANTR_CHECK(!try_merge(left[0], diagonal[0], lo, hi));

    // Differs on one axis but does not touch along it.
    const BlockList gapped = make(2, {{3, 0, 5, 3}});
    PANTR_CHECK(!try_merge(left[0], gapped[0], lo, hi));

    // Same extent on the merge axis but misaligned on the other: adjacency alone is not
    // enough, and this is the case a laxer test would miss.
    const BlockList offset = make(2, {{2, 1, 5, 4}});
    PANTR_CHECK(!try_merge(left[0], offset[0], lo, hi));

    // Two copies of one rectangle do not merge; there is no axis to merge along.
    PANTR_CHECK(!try_merge(left[0], left[0], lo, hi));
}

/// Peeling an interior box leaves the four surrounding slabs, in axis order.
///
/// Worked by hand on `[0,3) x [0,3)` minus the centre cell `[1,2) x [1,2)`. Axis 0 cuts
/// `[0,1) x [0,3)` then `[2,3) x [0,3)` and narrows the survivor to `[1,2)`; axis 1 then
/// cuts `[1,2) x [0,1)` and `[1,2) x [2,3)`. Four slabs, nine cells minus one.
void test_peel_hand_case() {
    const BlockList outer = make(2, {{0, 0, 3, 3}});
    const BlockList inner = make(2, {{1, 1, 2, 2}});
    BlockList out(2);
    peel(outer[0], inner[0], out);

    PANTR_CHECK(flatten(out)
                == std::vector<Corners>({{0, 0, 1, 3}, {2, 0, 3, 3}, {1, 0, 2, 1},
                                          {1, 2, 2, 3}}));
    PANTR_CHECK(total_cells(out) == 8);
}

/// Peeling a face-touching box emits fewer slabs, and never an empty one.
void test_peel_drops_empty_slabs() {
    // The inner box spans axis 1 entirely and sits at the low end of axis 0, so three of
    // the four candidate slabs are empty.
    const BlockList outer = make(2, {{0, 0, 3, 3}});
    const BlockList inner = make(2, {{0, 0, 1, 3}});
    BlockList out(2);
    peel(outer[0], inner[0], out);
    PANTR_CHECK(flatten(out) == std::vector<Corners>({{1, 0, 3, 3}}));

    // Peeling a box out of itself leaves nothing at all.
    BlockList nothing(2);
    peel(outer[0], outer[0], nothing);
    PANTR_CHECK(nothing.empty());
}

/// Whatever the shape, the remainder tiles `outer` minus `inner` exactly.
///
/// The property the hand cases above cannot cover: over every containment of a small box
/// in a small box, the slabs are non-overlapping, stay inside `outer`, avoid `inner`, and
/// account for every remaining cell. Checked by marking cells rather than by re-deriving
/// the decomposition, so it is not a mirror of `peel`'s own algorithm.
void test_peel_tiles_the_remainder() {
    constexpr std::int64_t kN = 4;
    std::int64_t checked = 0;
    for (std::int64_t a0 = 0; a0 < kN; ++a0) {
        for (std::int64_t a1 = a0 + 1; a1 <= kN; ++a1) {
            for (std::int64_t b0 = 0; b0 < kN; ++b0) {
                for (std::int64_t b1 = b0 + 1; b1 <= kN; ++b1) {
                    const BlockList outer = make(2, {{0, 0, kN, kN}});
                    const BlockList inner = make(2, {{a0, b0, a1, b1}});
                    BlockList out(2);
                    peel(outer[0], inner[0], out);

                    std::vector<int> marks(static_cast<std::size_t>(kN * kN), 0);
                    for (std::size_t s = 0; s < out.size(); ++s) {
                        const BlockView slab = out[s];
                        for (std::int64_t i = slab.lo[0]; i < slab.hi[0]; ++i) {
                            for (std::int64_t j = slab.lo[1]; j < slab.hi[1]; ++j) {
                                ++marks[static_cast<std::size_t>(i * kN + j)];
                            }
                        }
                    }
                    bool ok = out.size() <= 4;
                    for (std::int64_t i = 0; i < kN && ok; ++i) {
                        for (std::int64_t j = 0; j < kN && ok; ++j) {
                            const bool in_inner =
                                i >= a0 && i < a1 && j >= b0 && j < b1;
                            ok = marks[static_cast<std::size_t>(i * kN + j)]
                                 == (in_inner ? 0 : 1);
                        }
                    }
                    PANTR_CHECK(ok);
                    ++checked;
                }
            }
        }
    }
    PANTR_CHECK(checked == 100);
}

// ---------------------------------------------------------------------------
// normalize_blocks
// ---------------------------------------------------------------------------

/// The degenerate inputs, which take the early return and never enter the merge loop.
void test_normalize_trivial_inputs() {
    const BlockList empty(2);
    PANTR_CHECK(normalize_blocks(empty).empty());

    const BlockList single = make(2, {{1, 2, 3, 4}});
    PANTR_CHECK(flatten(normalize_blocks(single)) == std::vector<Corners>({{1, 2, 3, 4}}));
}

/// A row of unit cells collapses to one rectangle, in one pass.
void test_normalize_merges_a_row() {
    const BlockList row = make(1, {{0, 1}, {1, 2}, {2, 3}, {3, 4}});
    PANTR_CHECK(flatten(normalize_blocks(row)) == std::vector<Corners>({{0, 4}}));

    // Reversed input, same answer: chaining absorbs leftwards as readily as rightwards
    // when every merge is available, which is why a row is NOT a test of order
    // dependence. The tromino below is.
    const BlockList reversed = make(1, {{3, 4}, {2, 3}, {1, 2}, {0, 1}});
    PANTR_CHECK(flatten(normalize_blocks(reversed)) == std::vector<Corners>({{0, 4}}));
}

/// The output is sorted lexicographically by lower corner, whatever order it arrived in.
void test_normalize_sorts_its_output() {
    // Three rectangles that cannot merge with each other at all, shuffled.
    const BlockList blocks = make(2, {{4, 4, 5, 5}, {0, 0, 1, 1}, {2, 2, 3, 3}});
    PANTR_CHECK(flatten(normalize_blocks(blocks))
                == std::vector<Corners>({{0, 0, 1, 1}, {2, 2, 3, 3}, {4, 4, 5, 5}}));
}

/// **The contract.** The same three cells normalise two ways, and both are pinned.
///
/// An L-shaped tromino: cells `(0,0)`, `(1,0)` and `(0,1)`. The corner cell `(0,0)` can
/// merge with either neighbour, and which one it takes is decided by whichever comes
/// first in the list -- there is no tie-break, and no canonical answer.
///
/// Given `(0,0), (1,0), (0,1)` the corner merges along axis 0 into `[0,2) x [0,1)`,
/// leaving `[0,1) x [1,2)` alone. Given `(0,0), (0,1), (1,0)` it merges along axis 1
/// into `[0,1) x [0,2)`, leaving `[1,2) x [0,1)`. Two rectangles either way, three cells
/// either way, and neither pair is further mergeable.
///
/// Both values were read off the Python oracle rather than chosen. They are the assertion
/// that this port merges in the oracle's order, which is what the two backends agreeing
/// about flat cell ids rests on -- see the file header.
void test_normalize_is_order_dependent() {
    const BlockList axis0_first = make(2, {{0, 0, 1, 1}, {1, 0, 2, 1}, {0, 1, 1, 2}});
    PANTR_CHECK(flatten(normalize_blocks(axis0_first))
                == std::vector<Corners>({{0, 0, 2, 1}, {0, 1, 1, 2}}));

    const BlockList axis1_first = make(2, {{0, 0, 1, 1}, {0, 1, 1, 2}, {1, 0, 2, 1}});
    PANTR_CHECK(flatten(normalize_blocks(axis1_first))
                == std::vector<Corners>({{0, 0, 1, 2}, {1, 0, 2, 1}}));

    // Said as its own assertion, because it is the property a "canonical normalisation"
    // refactor would break while leaving every other test in this file green.
    PANTR_CHECK_MSG(flatten(normalize_blocks(axis0_first))
                        != flatten(normalize_blocks(axis1_first)),
                    "normalization has become order-independent; see this function's note");
}

/// A random non-overlapping decomposition of unit cells in a small lattice.
///
/// Unit cells, so the set is non-overlapping by construction and is a valid active-leaf
/// configuration for one level. Same shape as the oracle's own generator, so the two
/// sweeps cover the same population; the draws differ, which is fine, because what is
/// asserted below are properties rather than particular values.
///
/// \param rng The generator to draw from.
/// \return A list of unit rectangles, in a shuffled order.
BlockList random_decomposition(std::mt19937& rng) {
    const std::int64_t ndim = 1 + static_cast<std::int64_t>(rng() % 3);
    const std::int64_t n = 2 + static_cast<std::int64_t>(rng() % 4);
    std::int64_t count = 1;
    for (std::int64_t k = 0; k < ndim; ++k) {
        count *= n;
    }

    std::uniform_real_distribution<double> keep(0.0, 1.0);
    std::vector<Corners> rects;
    for (std::int64_t flat = 0; flat < count; ++flat) {
        if (keep(rng) >= 0.55) {
            continue;
        }
        Corners rect(static_cast<std::size_t>(2 * ndim));
        std::int64_t rest = flat;
        for (std::int64_t k = ndim - 1; k >= 0; --k) {
            rect[static_cast<std::size_t>(k)] = rest % n;
            rect[static_cast<std::size_t>(k + ndim)] = (rest % n) + 1;
            rest /= n;
        }
        rects.push_back(std::move(rect));
    }
    std::shuffle(rects.begin(), rects.end(), rng);
    return make(ndim, rects);
}

/// Over a large sweep: idempotent, cell-preserving, and pairwise non-mergeable.
///
/// The three together are what `_from_blocks`'s `unnormalized_levels` argument rests on.
/// A level holding a previous call's output must normalise to itself, or passing it
/// through unchanged and re-merging it would disagree and every flat cell id in the
/// result would move. Idempotence is the visible statement; non-mergeability is the
/// stronger one behind it, and it is checked directly rather than inferred, because a
/// merge loop that exited one pass early would still be idempotent on most inputs.
///
/// Ten times the oracle's own 400, and guarded so it cannot pass by merging nothing:
/// without merges this would be testing `sorted` and no more.
void test_normalize_properties_over_a_sweep() {
    std::mt19937 rng(20260901U);
    int checked = 0;
    int merged_something = 0;
    Corners lo;
    Corners hi;

    for (int trial = 0; trial < 4000; ++trial) {
        const BlockList blocks = random_decomposition(rng);
        if (blocks.empty()) {
            continue;
        }
        const BlockList once = normalize_blocks(blocks);

        PANTR_CHECK(normalize_blocks(once) == once);
        PANTR_CHECK(total_cells(once) == total_cells(blocks));

        lo.assign(static_cast<std::size_t>(once.ndim()), 0);
        hi.assign(static_cast<std::size_t>(once.ndim()), 0);
        bool any_mergeable = false;
        for (std::size_t i = 0; i < once.size() && !any_mergeable; ++i) {
            for (std::size_t j = i + 1; j < once.size() && !any_mergeable; ++j) {
                any_mergeable = try_merge(once[i], once[j], lo, hi);
            }
        }
        PANTR_CHECK(!any_mergeable);

        ++checked;
        merged_something += (once.size() < blocks.size()) ? 1 : 0;
    }
    PANTR_CHECK(checked > 3000);
    PANTR_CHECK(merged_something > checked / 2);
}

/// Over the same population: reshuffling changes the partition often enough to matter.
///
/// The sweep counterpart of the tromino above. Asserting `> 0` rather than a rate,
/// because the rate is a property of the generator; what is being defended is that the
/// order-dependence is still there at all, which is what a canonicalising rewrite would
/// remove. The oracle measures the same property on its own generator.
void test_normalize_order_dependence_survives_a_sweep() {
    std::mt19937 rng(20260902U);
    int differing = 0;
    for (int trial = 0; trial < 4000; ++trial) {
        BlockList blocks = random_decomposition(rng);
        if (blocks.size() < 2) {
            continue;
        }
        std::vector<Corners> shuffled = flatten(blocks);
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        if (flatten(normalize_blocks(blocks))
            != flatten(normalize_blocks(make(blocks.ndim(), shuffled)))) {
            ++differing;
        }
    }
    PANTR_CHECK_MSG(differing > 0,
                    "normalization has become order-independent; see the file header");
}

}  // namespace

int main() {
    test_block_size();
    test_rect_intersect();
    test_try_merge();
    test_peel_hand_case();
    test_peel_drops_empty_slabs();
    test_peel_tiles_the_remainder();
    test_normalize_trivial_inputs();
    test_normalize_merges_a_row();
    test_normalize_sorts_its_output();
    test_normalize_is_order_dependent();
    test_normalize_properties_over_a_sweep();
    test_normalize_order_dependence_survives_a_sweep();
    return pantr::test::summary("test_grid_blocks");
}
