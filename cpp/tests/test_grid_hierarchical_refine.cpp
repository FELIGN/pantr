/// \file
/// `HierarchicalGrid`'s refinement half: `refine`, `coarsen` and `restrict`.
///
/// Nothing here compares the two backends -- `tests/parity/` is where that happens, and
/// it needs the bindings. What this file does instead is tie the operations to
/// properties that hold whatever the implementation is.
///
/// ## The four kinds of check, and why none of them mirrors the code
///
/// **Closed-form counts.** Refining `m` active leaves at a level replaces each by
/// `prod(factor)` children, so the new cell count is `n - m + m * prod(factor)` exactly.
/// That is arithmetic the code never performs.
///
/// **An analytic invariant that survives any operation.** The active leaves partition the
/// root's domain, so their volumes sum to the root's whatever sequence of refinements and
/// coarsenings produced them. That is checked after each step, and it is what would catch
/// a block algebra that lost or double-counted a cell while keeping every count plausible.
///
/// **Round trips with their hypotheses.** `refine` inverts `coarsen` unconditionally, and
/// `coarsen` inverts `refine` only when the box was fully active first. Both directions
/// are asserted, and the second is asserted on a case where the hypothesis fails, so the
/// documented asymmetry is pinned rather than described.
///
/// **Hand-worked values.** The 7-cell refined-corner grid and the refusal messages are
/// small enough to work out in a comment, and are.
///
/// ## Where a bitwise assertion is used, and why it is legitimate there
///
/// One place. `restrict`'s sub-grid must reproduce its parent's cell bounds **bitwise**:
/// the sub-root is a pure slice of the parent's breakpoint buffer, and a windowed leaf
/// keeps the parent's level, its sub-index within the root cell and the root cell's own
/// two breakpoints, so `bounds_at` evaluates the *same expression on the same doubles*.
/// A tolerance there would accept a sub-root that had been re-based or re-clamped, which
/// is exactly the bug the assertion exists to catch. That is not a claim about
/// reproducibility across builds, which is what `design/backend_parity.md` reserves a
/// derived tolerance for; it is one program evaluating one expression twice.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/grid/blocks.hpp"
#include "pantr/grid/grid.hpp"
#include "pantr/grid/hierarchical_grid.hpp"
#include "pantr/grid/tensor_product_grid.hpp"

namespace {

using pantr::grid::BlockList;
using pantr::grid::BlockView;
using pantr::grid::GridRestriction;
using pantr::grid::HierarchicalGrid;
using pantr::grid::TensorProductGrid;

using Grid = HierarchicalGrid<double>;
using Ints = std::vector<std::int64_t>;

/// Half an ulp: the relative error of one correctly rounded `double` operation.
constexpr double kU = std::numeric_limits<double>::epsilon() / 2.0;

/// `gamma_m = m u / (1 - m u)`, the closed form for `m` accumulated roundings.
///
/// \param m The rounding count.
/// \return The factor.
constexpr double gamma_of(double m) { return m * kU / (1.0 - m * kU); }

/// A block list holding the given rectangles, each spelled `{lo..., hi...}`.
///
/// \param ndim Axis count.
/// \param rects The rectangles, in the order they are to be held.
/// \return The list.
BlockList level(std::int64_t ndim, const std::vector<Ints>& rects) {
    BlockList blocks(ndim);
    const auto d = static_cast<std::size_t>(ndim);
    for (const Ints& rect : rects) {
        blocks.push_back(BlockView{std::span<const std::int64_t>(rect.data(), d),
                                   std::span<const std::int64_t>(rect.data() + d, d)});
    }
    return blocks;
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// The unit square as a 2-by-2 grid of `0.5`-cells, factor 2, unrefined.
///
/// The C++ twin of `hierarchical_grid(uniform_grid([[0, 1], [0, 1]], 2), 2)`, which is
/// the fixture the Python export tests use. Every breakpoint is dyadic.
///
/// \return The grid.
Grid unit_2x2() {
    const Ints factor{2, 2};
    return Grid(TensorProductGrid<double>({{0.0, 0.5, 1.0}, {0.0, 0.5, 1.0}}), factor);
}

/// `unit_2x2` with the low-corner root cell refined once: three coarse cells and four fine.
///
/// level 0: `[0,1) x [1,2)` and `[1,2) x [0,2)` -- flat ids 0, 1, 2
/// level 1: `[0,2) x [0,2)`                     -- flat ids 3..6
///
/// \return The grid.
Grid refined_corner() {
    const Ints lo{0, 0};
    const Ints hi{1, 1};
    return unit_2x2().refine(0, lo, hi);
}

// ---------------------------------------------------------------------------
// Shared assertions
// ---------------------------------------------------------------------------

/// The sum of the active leaves' volumes.
///
/// \param g The grid.
/// \return The total volume, computed cell by cell from `cell_bounds`.
double total_volume(const Grid& g) {
    const auto d = static_cast<std::size_t>(g.ndim());
    std::vector<double> lo(d);
    std::vector<double> hi(d);
    double total = 0.0;
    for (std::int64_t cid = 0; cid < g.num_cells(); ++cid) {
        g.cell_bounds(cid, lo, hi);
        double volume = 1.0;
        for (std::size_t k = 0; k < d; ++k) {
            volume *= hi[k] - lo[k];
        }
        total += volume;
    }
    return total;
}

/// The active leaves still tile the root's domain, to a bound derived from the cell count.
///
/// Each cell's volume costs `ndim` subtractions and `ndim - 1` products, and the running
/// sum costs one addition per cell, so the accumulated relative error over `n` cells is at
/// most `gamma_(2 ndim + n)`. Stated against the root's own volume, which is the quantity
/// being reproduced and never vanishes.
///
/// \param g The grid.
/// \param what The fixture's name, for the failure message.
void check_the_leaves_tile_the_domain(const Grid& g, const std::string& what) {
    const auto d = static_cast<std::size_t>(g.ndim());
    double expected = 1.0;
    for (std::size_t k = 0; k < d; ++k) {
        const std::span<const double> bp = g.root().breakpoints(static_cast<std::int64_t>(k));
        expected *= bp.back() - bp.front();
    }
    const double bound =
        gamma_of(2.0 * static_cast<double>(d) + static_cast<double>(g.num_cells())) * expected;
    const double got = total_volume(g);
    PANTR_CHECK_MSG(std::abs(got - expected) <= bound,
                    what + ": volume " + std::to_string(got) + " against "
                        + std::to_string(expected));
}

/// Two grids hold the same cells, with the same ids, the same levels and the same bounds.
///
/// The bounds are compared **bitwise**: both sides run `bounds_at` on the same level, the
/// same sub-index and the same two breakpoints, so any difference at all is a difference
/// in the addressing rather than in the arithmetic.
///
/// \param a First grid.
/// \param b Second grid.
/// \param what The comparison's name, for the failure message.
void check_grids_agree(const Grid& a, const Grid& b, const std::string& what) {
    PANTR_CHECK_MSG(a.num_cells() == b.num_cells(), what + ": cell counts differ");
    PANTR_CHECK_MSG(a.to_string() == b.to_string(), what + ": " + a.to_string() + " vs "
                                                        + b.to_string());
    if (a.num_cells() != b.num_cells() || a.ndim() != b.ndim()) {
        return;
    }
    const auto d = static_cast<std::size_t>(a.ndim());
    std::vector<std::int64_t> ma(d);
    std::vector<std::int64_t> mb(d);
    std::vector<double> alo(d);
    std::vector<double> ahi(d);
    std::vector<double> blo(d);
    std::vector<double> bhi(d);
    for (std::int64_t cid = 0; cid < a.num_cells(); ++cid) {
        a.cell_multi_index(cid, ma);
        b.cell_multi_index(cid, mb);
        a.cell_bounds(cid, alo, ahi);
        b.cell_bounds(cid, blo, bhi);
        PANTR_CHECK_MSG(a.cell_level(cid) == b.cell_level(cid),
                        what + ": level differs at cell " + std::to_string(cid));
        PANTR_CHECK_MSG(ma == mb, what + ": index differs at cell " + std::to_string(cid));
        PANTR_CHECK_MSG(alo == blo && ahi == bhi,
                        what + ": bounds differ at cell " + std::to_string(cid));
    }
}

/// Run a call that must throw, and report the exception's message when it does.
///
/// \tparam Call The callable's type.
/// \param call The call.
/// \param needle A substring the message must contain.
/// \param what The check's name, for the failure message.
template <class Call>
void check_throws_containing(Call call, const std::string& needle, const std::string& what) {
    std::string message;
    bool threw = false;
    try {
        call();
    } catch (const std::exception& error) {
        threw = true;
        message = error.what();
    }
    PANTR_CHECK_MSG(threw, what + ": nothing was thrown");
    PANTR_CHECK_MSG(message.find(needle) != std::string::npos,
                    what + ": message was \"" + message + "\"");
}

// ---------------------------------------------------------------------------
// Refinement
// ---------------------------------------------------------------------------

/// The refined-corner fixture is exactly the seven cells its comment claims.
void test_the_fixture_is_what_it_claims() {
    const Grid g = refined_corner();
    PANTR_CHECK(g.num_cells() == 7);
    PANTR_CHECK(g.max_level() == 1);
    for (std::int64_t cid = 0; cid < 3; ++cid) {
        PANTR_CHECK(g.cell_level(cid) == 0);
    }
    for (std::int64_t cid = 3; cid < 7; ++cid) {
        PANTR_CHECK(g.cell_level(cid) == 1);
    }
    check_the_leaves_tile_the_domain(g, "the refined-corner fixture");
}

/// `refine` replaces each active leaf of the region by `prod(factor)` children.
///
/// A closed form the code never evaluates: with `m` active leaves inside the box and
/// `p = prod(factor)` children each, the new count is `n - m + m * p`. Checked on a box
/// that covers the whole level and on one that covers part of it.
void test_refine_obeys_the_cell_count_identity() {
    const Grid flat = unit_2x2();
    const Ints whole_lo{0, 0};
    const Ints whole_hi{2, 2};
    const Grid all = flat.refine(0, whole_lo, whole_hi);
    PANTR_CHECK_MSG(all.num_cells() == 4 - 4 + 4 * 4, std::to_string(all.num_cells()));

    const Ints part_lo{0, 0};
    const Ints part_hi{1, 2};
    const Grid half = flat.refine(0, part_lo, part_hi);
    PANTR_CHECK_MSG(half.num_cells() == 4 - 2 + 2 * 4, std::to_string(half.num_cells()));

    // Union semantics: only the still-active part of the box is promoted, so refining a
    // box that already holds fine cells costs only the coarse ones inside it.
    const Grid again = refined_corner().refine(0, whole_lo, whole_hi);
    PANTR_CHECK_MSG(again.num_cells() == 7 - 3 + 3 * 4, std::to_string(again.num_cells()));
    check_the_leaves_tile_the_domain(again, "the twice-refined grid");
}

/// The receiver keeps its cells, its ids and its tags; the result gets none of them.
///
/// This is what separates `rebuilt()` from the copy constructor: a copy would carry the
/// receiver's two tag registries into the result, and every value assertion would still
/// pass.
void test_refinement_never_touches_the_receiver() {
    Grid g = refined_corner();
    const Ints ids{0, 1};
    const Ints values{7, 9};
    g.cell_tags().set("marked", ids, values);
    PANTR_CHECK(g.cell_tags().contains("marked"));

    const Ints lo{1, 1};
    const Ints hi{2, 2};
    const Grid refined = g.refine(0, lo, hi);
    PANTR_CHECK_MSG(g.num_cells() == 7, "the receiver changed size");
    PANTR_CHECK_MSG(g.cell_tags().contains("marked"), "the receiver lost its tag");
    PANTR_CHECK_MSG(!refined.cell_tags().contains("marked"), "the result inherited a tag");
    PANTR_CHECK_MSG(refined.cell_tags().size() == 0, "the result has tags");
    PANTR_CHECK_MSG(refined.facet_tags().size() == 0, "the result has facet tags");
}

/// A region with no active cell yields an equal grid that is nonetheless a fresh one.
void test_refining_an_inactive_region_returns_a_cold_copy() {
    Grid g = refined_corner();
    const Ints ids{0};
    const Ints values{1};
    g.cell_tags().set("marked", ids, values);

    // `[0,1) x [0,1)` at level 0 is the corner that was already refined away.
    const Ints lo{0, 0};
    const Ints hi{1, 1};
    const Grid same = g.refine(0, lo, hi);
    check_grids_agree(g, same, "refining an inactive region");
    PANTR_CHECK_MSG(same.cell_tags().size() == 0, "the copy inherited a tag");
}

/// `refine_cells` refines the per-level bounding box, so it can promote unnamed cells.
///
/// On the flat 2-by-2 grid the four cells are `(0,0) (0,1) (1,0) (1,1)` as ids 0..3.
/// Naming the two on the diagonal, ids 0 and 3, gives the bounding box `[0,2) x [0,2)`,
/// so all four are refined and the count is `4 - 4 + 4 * 4`. Naming ids 0 and 1 gives
/// `[0,1) x [0,2)` and refines two.
void test_refine_cells_uses_the_bounding_box() {
    const Grid flat = unit_2x2();
    const Ints diagonal{0, 3};
    PANTR_CHECK_MSG(flat.refine_cells(diagonal).num_cells() == 16,
                    std::to_string(flat.refine_cells(diagonal).num_cells()));
    const Ints row{0, 1};
    PANTR_CHECK_MSG(flat.refine_cells(row).num_cells() == 10,
                    std::to_string(flat.refine_cells(row).num_cells()));
    const Ints repeated{0, 0, 1, 1, 0};
    check_grids_agree(flat.refine_cells(row), flat.refine_cells(repeated),
                      "repeated ids are ignored");
}

/// Ids from several levels are handled in one call, coarsest level first.
void test_refine_cells_spans_levels() {
    const Grid g = refined_corner();
    // Id 0 is a coarse cell at level 0; id 3 is a fine one at level 1.
    const Ints mixed{0, 3};
    const Grid refined = g.refine_cells(mixed);
    PANTR_CHECK(refined.max_level() == 2);
    // The level-0 box is the single cell (0,1) and the level-1 box the single cell (0,0),
    // so each contributes `4 - 1 + 4` to the count: 7 + 3 + 3.
    PANTR_CHECK_MSG(refined.num_cells() == 13, std::to_string(refined.num_cells()));
    check_the_leaves_tile_the_domain(refined, "refine_cells across two levels");
}

/// An empty id list yields a cold copy rather than the receiver.
void test_refine_cells_with_no_ids_returns_a_cold_copy() {
    Grid g = refined_corner();
    const Ints ids{2};
    const Ints values{5};
    g.cell_tags().set("marked", ids, values);
    const Ints none{};
    const Grid same = g.refine_cells(none);
    check_grids_agree(g, same, "refine_cells with no ids");
    PANTR_CHECK_MSG(same.cell_tags().size() == 0, "the copy inherited a tag");
}

/// Every documented refusal of `refine` fires, and says which rule it is.
void test_refine_rejects_bad_arguments() {
    const Grid g = refined_corner();
    const Ints lo{0, 0};
    const Ints hi{1, 1};
    check_throws_containing([&] { (void)g.refine(2, lo, hi); }, "level must be in [0, 1]",
                            "a level past the deepest");
    check_throws_containing([&] { (void)g.refine(-1, lo, hi); }, "level must be in [0, 1]",
                            "a negative level");
    const Ints short_lo{0};
    check_throws_containing([&] { (void)g.refine(0, short_lo, hi); }, "lo must have 2 entries",
                            "a short lo");
    const Ints empty_hi{0, 1};
    check_throws_containing([&] { (void)g.refine(0, lo, empty_hi); },
                            "lo must be strictly less than hi", "an empty box");
    const Ints far_hi{3, 1};
    check_throws_containing([&] { (void)g.refine(0, lo, far_hi); }, "out of bounds at level 0",
                            "a box outside the level");
    const Ints bad{0};
    check_throws_containing([&] { (void)g.refine_cells(Ints{99}); }, "out of range",
                            "an out-of-range id");
    static_cast<void>(bad);
}

// ---------------------------------------------------------------------------
// Coarsening
// ---------------------------------------------------------------------------

/// `refine` always undoes `coarsen`: coarsening leaves the whole box active at `level`.
void test_refine_inverts_coarsen_unconditionally() {
    const Grid g = refined_corner();
    const Ints lo{0, 0};
    const Ints hi{1, 1};
    const Grid coarse = g.coarsen(0, lo, hi);
    PANTR_CHECK_MSG(coarse.num_cells() == 4, std::to_string(coarse.num_cells()));
    check_grids_agree(g, coarse.refine(0, lo, hi), "refine after coarsen");
    check_the_leaves_tile_the_domain(coarse, "the coarsened grid");
}

/// `coarsen` undoes `refine` when every cell of the box was an active leaf beforehand.
void test_coarsen_inverts_a_fully_active_refine() {
    const Grid flat = unit_2x2();
    const Ints lo{0, 0};
    const Ints hi{2, 2};
    check_grids_agree(flat, flat.refine(0, lo, hi).coarsen(0, lo, hi), "coarsen after refine");
}

/// It does not, when the box held cells that were already refined away.
///
/// `refined_corner` promoted `(0,0)`. Refining the whole level then promotes only the
/// three cells that were still coarse, and coarsening the whole box afterwards demotes
/// all four -- including `(0,0)`, whose children the first `refine` created. So the round
/// trip lands on the flat grid rather than back on `refined_corner`, which is the
/// documented asymmetry.
void test_coarsen_does_not_invert_a_partial_refine() {
    const Grid g = refined_corner();
    const Ints lo{0, 0};
    const Ints hi{2, 2};
    const Grid round_trip = g.refine(0, lo, hi).coarsen(0, lo, hi);
    PANTR_CHECK_MSG(round_trip.num_cells() == 4, std::to_string(round_trip.num_cells()));
    check_grids_agree(unit_2x2(), round_trip, "the partial-refine round trip");
    PANTR_CHECK_MSG(round_trip.num_cells() != g.num_cells(),
                    "the round trip must not return the grid it started from");
}

/// A region that is not fully refined to exactly `level + 1` is refused, by reason.
///
/// On `refined_corner` the box `[0,2) x [0,2)` at level 0 covers all four root cells, but
/// only `(0,0)` has children. The other three are still active leaves at level 0, and the
/// message must name all three.
void test_coarsen_refuses_and_names_the_obstacles() {
    const Grid g = refined_corner();
    const Ints lo{0, 0};
    const Ints hi{2, 2};
    std::string message;
    bool threw = false;
    try {
        (void)g.coarsen(0, lo, hi);
    } catch (const std::invalid_argument& error) {
        threw = true;
        message = error.what();
    }
    PANTR_CHECK_MSG(threw, "coarsening a half-refined box must be refused");
    PANTR_CHECK_MSG(message.find("cannot coarsen [(0, 0), (2, 2)) at level 0")
                        != std::string::npos,
                    message);
    PANTR_CHECK_MSG(message.find("not fully refined to exactly level 1") != std::string::npos,
                    message);
    PANTR_CHECK_MSG(message.find("still active leaves at level 0, with no children to remove: "
                                 "(0, 1), (1, 0), (1, 1)")
                        != std::string::npos,
                    message);
    // Nothing is over-refined and nothing is hidden here, so neither clause may appear.
    PANTR_CHECK_MSG(message.find("refined beyond") == std::string::npos, message);
    PANTR_CHECK_MSG(message.find("covered by a coarser") == std::string::npos, message);
}

/// The over-refined and hidden reasons are reported too, and each names its own cells.
///
/// A 4-cell line, factor 2, with all three obstacles present at once when `[0, 8)` is
/// coarsened at level 1:
///
///   * root cell 3 stays a level-0 leaf, so the level-1 positions 6 and 7 under it do not
///     exist -- **hidden**;
///   * root cells 1 and 2 are refined exactly once, so the level-1 positions 2 to 5 are
///     themselves active leaves with no children to remove -- **still leaves**;
///   * root cell 0 is refined all the way to level 3, so the level-1 positions 0 and 1
///     have descendants below level 2 -- **over-refined**.
///
/// Nothing is refined to *exactly* level 2, which is why the call is refused at all.
void test_coarsen_reports_every_reason() {
    const Ints factor{2};
    std::vector<BlockList> blocks;
    blocks.push_back(level(1, {{3, 4}}));          // level 0: root cell 3
    blocks.push_back(level(1, {{2, 6}}));          // level 1: root cells 1 and 2
    blocks.push_back(level(1, {}));                // level 2: empty
    blocks.push_back(level(1, {{0, 8}}));          // level 3: under root cell 0
    const Grid g = Grid::from_blocks(TensorProductGrid<double>({{0.0, 1.0, 2.0, 3.0, 4.0}}),
                                     factor, std::move(blocks), std::nullopt);
    PANTR_CHECK(g.num_cells() == 1 + 4 + 8);
    check_the_leaves_tile_the_domain(g, "the three-obstacle fixture");

    const Ints lo{0};
    const Ints hi{8};
    std::string message;
    try {
        (void)g.coarsen(1, lo, hi);
    } catch (const std::invalid_argument& error) {
        message = error.what();
    }
    PANTR_CHECK_MSG(message.find("still active leaves at level 1, with no children to remove: "
                                 "(2,), (3,), (4,), (5,)")
                        != std::string::npos,
                    message);
    PANTR_CHECK_MSG(message.find("refined beyond level 2: (0,), (1,)") != std::string::npos,
                    message);
    PANTR_CHECK_MSG(message.find("covered by a coarser active leaf and absent at level 1: "
                                 "(6,), (7,)")
                        != std::string::npos,
                    message);
}

/// At most six cells are named per reason, and the remainder is reported as a count.
void test_coarsen_refusal_truncates_a_long_list() {
    const Ints factor{2};
    std::vector<Ints> rects;
    Ints whole{0, 16};
    std::vector<BlockList> blocks;
    blocks.push_back(level(1, {whole}));
    std::vector<double> bp(17);
    for (std::size_t i = 0; i < bp.size(); ++i) {
        bp[i] = static_cast<double>(i);
    }
    const Grid g = Grid::from_blocks(TensorProductGrid<double>({bp}), factor, std::move(blocks),
                                     std::nullopt);
    PANTR_CHECK(g.num_cells() == 16);
    static_cast<void>(rects);

    // Level 0 holds all sixteen cells and there is no level 1 at all, so every cell of the
    // box is a still-active leaf: six named and ten counted.
    const Ints lo{0};
    const Ints hi{16};
    check_throws_containing([&] { (void)g.coarsen(0, lo, hi); }, "level must be in [0, 0)",
                            "coarsening a grid with no finer level");

    const Grid deep = g.refine(0, Ints{0}, Ints{1});
    std::string message;
    try {
        (void)deep.coarsen(0, lo, hi);
    } catch (const std::invalid_argument& error) {
        message = error.what();
    }
    PANTR_CHECK_MSG(message.find("(1,), (2,), (3,), (4,), (5,), (6,) and 9 more")
                        != std::string::npos,
                    message);
}

/// A region too large to enumerate reports its extent instead of naming cells.
///
/// A level may hold far more cells than the grid does, so the diagnostic's cap is on the
/// region's extent rather than on the cell count. One root cell with factor 2 refined
/// straight to level 22 holds two cells, and its level-21 box spans `2 ** 21` positions --
/// twice the cap, from a grid of two cells.
void test_coarsen_refusal_reports_a_huge_region_as_an_extent() {
    const Ints factor{2};
    std::vector<BlockList> blocks;
    for (std::int64_t l = 0; l < 22; ++l) {
        blocks.push_back(level(1, {}));
    }
    blocks.push_back(level(1, {{0, 2}}));
    const Grid g = Grid::from_blocks(TensorProductGrid<double>({{0.0, 1.0}}), factor,
                                     std::move(blocks), std::nullopt);
    PANTR_CHECK(g.num_cells() == 2);
    const Ints lo{0};
    const Ints hi{std::int64_t{1} << 21};
    check_throws_containing([&] { (void)g.coarsen(21, lo, hi); },
                            "cells, too many to name", "a region above the diagnostic cap");
}

/// `coarsen_cells` demotes a parent only when every one of its children is named.
void test_coarsen_cells_demotes_only_complete_families() {
    const Grid g = refined_corner();
    const Ints three{3, 4, 5};
    check_grids_agree(g, g.coarsen_cells(three), "three of four children named");

    const Ints four{3, 4, 5, 6};
    const Grid demoted = g.coarsen_cells(four);
    check_grids_agree(unit_2x2(), demoted, "all four children named");
}

/// Ids at level 0 have no parent and are ignored rather than refused.
void test_coarsen_cells_ignores_level_zero_ids() {
    const Grid g = refined_corner();
    const Ints coarse_only{0, 1, 2};
    check_grids_agree(g, g.coarsen_cells(coarse_only), "only level-0 ids named");

    // Mixing them in does not stop the complete family from being demoted.
    const Ints mixed{0, 1, 2, 3, 4, 5, 6};
    check_grids_agree(unit_2x2(), g.coarsen_cells(mixed), "level-0 ids mixed in");
}

/// A call that demotes nothing returns a cold copy, not the receiver.
void test_coarsen_cells_that_demotes_nothing_returns_a_cold_copy() {
    Grid g = refined_corner();
    const Ints ids{1};
    const Ints values{4};
    g.cell_tags().set("marked", ids, values);
    const Ints none{};
    const Grid same = g.coarsen_cells(none);
    check_grids_agree(g, same, "coarsen_cells with no ids");
    PANTR_CHECK_MSG(same.cell_tags().size() == 0, "the copy inherited a tag");
}

/// Two complete families at two levels are demoted in one call, deepest first.
void test_coarsen_cells_spans_levels() {
    // A 2-cell line, factor 2: refine root cell 0 to level 1, then its left child to
    // level 2. Cells: level 0 -> root cell 1; level 1 -> the right child of root cell 0;
    // level 2 -> the two children of the left one.
    const Ints factor{2};
    Grid g(TensorProductGrid<double>({{0.0, 1.0, 2.0}}), factor);
    g = g.refine(0, Ints{0}, Ints{1});
    g = g.refine(1, Ints{0}, Ints{1});
    PANTR_CHECK(g.num_cells() == 4);
    PANTR_CHECK(g.max_level() == 2);

    // Naming every cell below level 0 collapses the whole hierarchy in one call: the two
    // level-2 cells rebuild the level-1 cell (0), which was not named and so cannot then
    // be part of a complete level-0 family. Coarsening does not cascade.
    Ints every;
    for (std::int64_t cid = 0; cid < g.num_cells(); ++cid) {
        if (g.cell_level(cid) > 0) {
            every.push_back(cid);
        }
    }
    const Grid demoted = g.coarsen_cells(every);
    PANTR_CHECK_MSG(demoted.max_level() == 1, std::to_string(demoted.max_level()));
    PANTR_CHECK_MSG(demoted.num_cells() == 3, std::to_string(demoted.num_cells()));
    check_the_leaves_tile_the_domain(demoted, "coarsen_cells across two levels");
}

/// Every documented refusal of `coarsen` fires.
void test_coarsen_rejects_bad_arguments() {
    const Grid g = refined_corner();
    const Ints lo{0, 0};
    const Ints hi{1, 1};
    check_throws_containing([&] { (void)g.coarsen(1, lo, hi); }, "level must be in [0, 1)",
                            "coarsening the deepest level");
    const Ints short_hi{1};
    check_throws_containing([&] { (void)g.coarsen(0, lo, short_hi); }, "hi must have 2 entries",
                            "a short hi");
    const Ints far_hi{3, 1};
    check_throws_containing([&] { (void)g.coarsen(0, lo, far_hi); }, "out of bounds at level 0",
                            "a box outside the level");
    check_throws_containing([&] { (void)g.coarsen_cells(Ints{99}); }, "out of range",
                            "an out-of-range id");
}

// ---------------------------------------------------------------------------
// Restriction
// ---------------------------------------------------------------------------

/// The hook replaces the mixin's default, which still throws when named explicitly.
void test_restrict_hook_replaces_the_throwing_default() {
    const Grid g = refined_corner();
    const Ints ids{0};
    const GridRestriction<Grid> restricted = g.restrict(ids);
    PANTR_CHECK(restricted.grid.num_cells() >= 1);
    check_throws_containing([&] { (void)g.Grid::Base::restrict(ids); },
                            "does not support restrict", "the hidden default");
}

/// The window is root-cell-aligned, so one deep leaf drags in its whole root cell.
///
/// On `refined_corner`, id 3 is the first of the four children of root cell `(0,0)`.
/// Restricting it returns the whole tiling of that root cell -- four level-1 cells -- with
/// only id 3 flagged.
void test_restrict_returns_the_whole_root_cell_of_a_deep_leaf() {
    const Grid g = refined_corner();
    const Ints ids{3};
    const GridRestriction<Grid> restricted = g.restrict(ids);
    PANTR_CHECK_MSG(restricted.grid.num_cells() == 4,
                    std::to_string(restricted.grid.num_cells()));
    PANTR_CHECK(restricted.grid.max_level() == 1);
    PANTR_CHECK(restricted.local_to_global_cell == std::vector<std::int64_t>({3, 4, 5, 6}));
    PANTR_CHECK(restricted.in_subset == std::vector<std::uint8_t>({1, 0, 0, 0}));
    // The sub-root is a pure slice: one root cell, `[0, 0.5]` on each axis.
    const std::span<const std::int64_t> sub_cells = restricted.grid.root().cells_per_axis();
    PANTR_CHECK(Ints(sub_cells.begin(), sub_cells.end()) == Ints({1, 1}));
}

/// Every windowed cell keeps its parent's geometry bitwise, and its own id map.
void test_restrict_preserves_cell_geometry_bitwise() {
    const Grid g = refined_corner();
    const Ints ids{0, 3};
    const GridRestriction<Grid> restricted = g.restrict(ids);
    // Root cells (0,0) and (0,1) -- the window is `[0,1) x [0,2)`, so it holds the coarse
    // cell (0,1) and the four children of (0,0): five cells.
    PANTR_CHECK_MSG(restricted.grid.num_cells() == 5,
                    std::to_string(restricted.grid.num_cells()));
    PANTR_CHECK(restricted.local_to_global_cell.size() == 5);

    std::vector<double> slo(2);
    std::vector<double> shi(2);
    std::vector<double> glo(2);
    std::vector<double> ghi(2);
    for (std::size_t local = 0; local < restricted.local_to_global_cell.size(); ++local) {
        const auto local_cid = static_cast<std::int64_t>(local);
        const std::int64_t global = restricted.local_to_global_cell[local];
        restricted.grid.cell_bounds(local_cid, slo, shi);
        g.cell_bounds(global, glo, ghi);
        PANTR_CHECK_MSG(slo == glo && shi == ghi,
                        "sub cell " + std::to_string(local) + " moved against global cell "
                            + std::to_string(global));
        PANTR_CHECK(restricted.grid.cell_level(local_cid) == g.cell_level(global));
    }
    std::size_t flagged = 0;
    for (const std::uint8_t flag : restricted.in_subset) {
        flagged += flag == 1U ? 1U : 0U;
    }
    PANTR_CHECK_MSG(flagged == 2, "exactly the two requested cells must be flagged");
}

/// Restricting the whole grid returns a grid equal to it.
void test_restrict_over_every_cell_is_the_identity() {
    const Grid g = refined_corner();
    Ints all;
    for (std::int64_t cid = 0; cid < g.num_cells(); ++cid) {
        all.push_back(cid);
    }
    const GridRestriction<Grid> restricted = g.restrict(all);
    check_grids_agree(g, restricted.grid, "restricting every cell");
    for (std::size_t local = 0; local < restricted.local_to_global_cell.size(); ++local) {
        PANTR_CHECK(restricted.local_to_global_cell[local]
                    == static_cast<std::int64_t>(local));
        PANTR_CHECK(restricted.in_subset[local] == 1U);
    }
}

/// The documented refusals fire.
void test_restrict_rejects_bad_arguments() {
    const Grid g = refined_corner();
    const Ints none{};
    check_throws_containing([&] { (void)g.restrict(none); }, "must be non-empty",
                            "an empty request");
    check_throws_containing([&] { (void)g.restrict(Ints{7}); }, "out of range",
                            "an out-of-range id");
    check_throws_containing([&] { (void)g.restrict(Ints{-1}); }, "out of range",
                            "a negative id");
}

// ---------------------------------------------------------------------------
// The invariant, over a deterministic sweep of operations
// ---------------------------------------------------------------------------

/// The leaves tile the domain after every step of a long refine-and-coarsen sweep.
///
/// Deterministic rather than random, so a failure is reproducible from the file alone.
/// The sweep is what would catch a block algebra that drops or double-counts a cell while
/// leaving every individual count plausible.
void test_the_partition_survives_a_long_sweep() {
    Grid g(TensorProductGrid<double>({{0.0, 0.3, 0.55, 1.0}, {0.0, 0.1, 0.4, 1.0}}),
           Ints{2, 3});
    check_the_leaves_tile_the_domain(g, "sweep step 0");

    for (std::int64_t step = 0; step < 6; ++step) {
        const std::int64_t lvl = step % (g.max_level() + 1);
        const std::int64_t nx = g.level_cells_per_axis(lvl, 0);
        const std::int64_t ny = g.level_cells_per_axis(lvl, 1);
        const Ints lo{step % nx, step % ny};
        const Ints hi{std::min(nx, lo[0] + 1 + step % 2), std::min(ny, lo[1] + 1 + step % 3)};
        g = g.refine(lvl, lo, hi);
        check_the_leaves_tile_the_domain(g, "sweep refine step " + std::to_string(step));
    }
    PANTR_CHECK(g.max_level() >= 2);

    // Coarsen every complete family, deepest level first, until nothing else demotes.
    for (std::int64_t round = 0; round < 4; ++round) {
        Ints deep;
        for (std::int64_t cid = 0; cid < g.num_cells(); ++cid) {
            if (g.cell_level(cid) == g.max_level()) {
                deep.push_back(cid);
            }
        }
        g = g.coarsen_cells(deep);
        check_the_leaves_tile_the_domain(g, "sweep coarsen round " + std::to_string(round));
    }
    PANTR_CHECK_MSG(g.max_level() == 0, "the sweep should have collapsed to one level; got "
                                            + std::to_string(g.max_level()));
    PANTR_CHECK_MSG(g.num_cells() == 9, std::to_string(g.num_cells()));
}

}  // namespace

/// Run every case.
///
/// \return `0` when all pass.
int main() {
    test_the_fixture_is_what_it_claims();
    test_refine_obeys_the_cell_count_identity();
    test_refinement_never_touches_the_receiver();
    test_refining_an_inactive_region_returns_a_cold_copy();
    test_refine_cells_uses_the_bounding_box();
    test_refine_cells_spans_levels();
    test_refine_cells_with_no_ids_returns_a_cold_copy();
    test_refine_rejects_bad_arguments();

    test_refine_inverts_coarsen_unconditionally();
    test_coarsen_inverts_a_fully_active_refine();
    test_coarsen_does_not_invert_a_partial_refine();
    test_coarsen_refuses_and_names_the_obstacles();
    test_coarsen_reports_every_reason();
    test_coarsen_refusal_truncates_a_long_list();
    test_coarsen_refusal_reports_a_huge_region_as_an_extent();
    test_coarsen_cells_demotes_only_complete_families();
    test_coarsen_cells_ignores_level_zero_ids();
    test_coarsen_cells_that_demotes_nothing_returns_a_cold_copy();
    test_coarsen_cells_spans_levels();
    test_coarsen_rejects_bad_arguments();

    test_restrict_hook_replaces_the_throwing_default();
    test_restrict_returns_the_whole_root_cell_of_a_deep_leaf();
    test_restrict_preserves_cell_geometry_bitwise();
    test_restrict_over_every_cell_is_the_identity();
    test_restrict_rejects_bad_arguments();

    test_the_partition_survives_a_long_sweep();
    return pantr::test::summary("test_grid_hierarchical_refine");
}
