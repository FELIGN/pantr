/// \file
/// The `HierarchicalGrid` type: addressing, geometry, neighbours and the five hooks.
///
/// Nothing here compares the two backends -- `tests/parity/` is where that happens, and
/// it needs the bindings. What this file can do, and does, is tie the type's answers to
/// each other and to closed-form geometry, so that a divergence shows up as a named
/// broken property rather than as a bare inequality in a parity sweep.
///
/// ## The four kinds of check, and why each is not a mirror of the code
///
/// **Two unrelated paths, compared bitwise.** `cell_bounds` decodes a flat id to
/// `(level, midx)` by binary search and evaluates one cell; `collect_cell_bounds` walks
/// blocks with an odometer and evaluates all of them. They share no code, so agreeing on
/// every cell is a real statement. `locate` descends levels and truncates quotients,
/// while `cell_bounds` does neither, so locating each cell's midpoint back to its own id
/// is another. And every hook is compared against the mixin default it hides, reached by
/// the qualified call `g.Base::name(...)` -- the C++ analogue of the unbound
/// `Grid.boundary_facets(g)` in `tests/test_grid_hierarchical.py`.
///
/// **Closed form, where the arithmetic is exact.** On a dyadic root with a dyadic
/// factor, a level-`l` cell's corners are exactly representable binary rationals, so the
/// computed bound must equal the hand-written value **bitwise**. That is an
/// oracle-independent statement about the geometry, not a comparison against another
/// implementation.
///
/// **The same check where the arithmetic is not exact.** A closed-form test that only
/// ever runs on exactly-representable inputs has not tested the arithmetic, it has
/// tested the cases that cannot fail. So the non-dyadic fixture asserts both halves: the
/// error against the exact rational is **nonzero** somewhere, and it is inside a bound
/// derived from the operation count rather than fitted to the observation.
///
/// **Hand-computed values** for the block-level answers -- `boundary_facets`, the two
/// masks, the neighbour cases -- on fixtures small enough to work out in a comment.
///
/// ## What is deliberately not here
///
/// Refinement, coarsening, restriction and the mesh export are not part of this type
/// yet. The mixin's throwing `restrict` default stands, and the traits bitmask says so.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "check.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/grid/blocks.hpp"
#include "pantr/grid/hierarchical_grid.hpp"
#include "pantr/grid/tensor_product_grid.hpp"

namespace {

using pantr::span2d;
using pantr::grid::BlockList;
using pantr::grid::BlockView;
using pantr::grid::HierarchicalGrid;
using pantr::grid::TensorProductGrid;

using Grid = HierarchicalGrid<double>;
using Ints = std::vector<std::int64_t>;

/// A block list holding one rectangle, spelled `{lo..., hi...}`.
///
/// \param ndim Axis count.
/// \param rects The rectangles.
/// \return The list, in the order given.
BlockList level(std::int64_t ndim, const std::vector<Ints>& rects) {
    BlockList blocks(ndim);
    const auto d = static_cast<std::size_t>(ndim);
    for (const Ints& rect : rects) {
        blocks.push_back(BlockView{std::span<const std::int64_t>(rect.data(), d),
                                   std::span<const std::int64_t>(rect.data() + d, d)});
    }
    return blocks;
}

/// The root of every 1-D fixture: four unit cells on `[0, 4]`.
///
/// \return The root grid.
TensorProductGrid<double> root_1d() {
    return TensorProductGrid<double>({{0.0, 1.0, 2.0, 3.0, 4.0}});
}

/// Four unit cells, unrefined, factor 2. Flat ids are the root's own.
///
/// \return The grid.
Grid flat_1d() {
    const Ints factor{2};
    return Grid(root_1d(), factor);
}

/// Root cells 0 and 1 refined once; cells 2 and 3 stay coarse.
///
/// level 0: block `[2, 4)` -- flat ids 0, 1, the cells `[2,3]` and `[3,4]`
/// level 1: block `[0, 4)` -- flat ids 2..5, the cells `[0,0.5] ... [1.5,2]`
///
/// Six cells. Note the ordering trap the numbering makes concrete: the *coarse* cells
/// come first because level 0 is packed first, even though they sit to the right.
///
/// \return The grid.
Grid refined_1d() {
    const Ints factor{2};
    std::vector<BlockList> blocks;
    blocks.push_back(level(1, {{2, 4}}));
    blocks.push_back(level(1, {{0, 4}}));
    return Grid::from_blocks(root_1d(), factor, std::move(blocks), std::nullopt);
}

/// A 2-by-2 root of unit cells on `[0,2] x [0,2]`, factor 2, with cell (0,0) refined.
///
/// level 0: blocks `[0,1) x [1,2)`, `[1,2) x [0,2)` -- three coarse cells
/// level 1: block `[0,2) x [0,2)` -- the four children of root cell (0,0)
///
/// Seven cells.
///
/// \return The grid.
Grid refined_2d() {
    const Ints factor{2, 2};
    std::vector<BlockList> blocks;
    blocks.push_back(level(2, {{0, 1, 1, 2}, {1, 0, 2, 2}}));
    blocks.push_back(level(2, {{0, 0, 2, 2}}));
    return Grid::from_blocks(TensorProductGrid<double>({{0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}}),
                             factor, std::move(blocks), std::nullopt);
}

/// The whole unit interval as one root cell, refined uniformly to `level`, factor 2.
///
/// Every corner is `k / 2**level`, a binary rational and therefore exact in `double`.
/// This is the fixture the closed-form assertions run on.
///
/// \param depth The level to refine to.
/// \return The grid.
Grid dyadic_unit(std::int64_t depth) {
    const Ints factor{2};
    std::vector<BlockList> blocks;
    for (std::int64_t l = 0; l < depth; ++l) {
        blocks.push_back(level(1, {}));
    }
    blocks.push_back(level(1, {{0, std::int64_t{1} << depth}}));
    return Grid::from_blocks(TensorProductGrid<double>({{0.0, 1.0}}), factor, std::move(blocks),
                             std::nullopt);
}

/// One root cell `[0, 0.1]`, refined uniformly to level 2 with factor 3.
///
/// Nothing here is exact: `0.1` is not representable, and dividing it by 9 is not
/// either. That is the point -- see the file header.
///
/// \return The grid.
Grid non_dyadic() {
    const Ints factor{3};
    std::vector<BlockList> blocks;
    blocks.push_back(level(1, {}));
    blocks.push_back(level(1, {}));
    blocks.push_back(level(1, {{0, 9}}));
    return Grid::from_blocks(TensorProductGrid<double>({{0.0, 0.1}}), factor, std::move(blocks),
                             std::nullopt);
}

/// One cell's corners.
///
/// \param g The grid.
/// \param cid Cell identifier.
/// \return `(lo, hi)`.
std::pair<std::vector<double>, std::vector<double>> bounds(const Grid& g, std::int64_t cid) {
    std::vector<double> lo(static_cast<std::size_t>(g.ndim()));
    std::vector<double> hi(lo.size());
    g.cell_bounds(cid, lo, hi);
    return {lo, hi};
}

/// Every cell's corners, through the hook.
///
/// \param g The grid.
/// \return `(cell_lo, cell_hi)`, each `num_cells * ndim` row-major.
std::pair<std::vector<double>, std::vector<double>> all_bounds(const Grid& g) {
    const auto n = static_cast<std::size_t>(g.num_cells());
    const auto d = static_cast<std::size_t>(g.ndim());
    std::vector<double> lo(n * d);
    std::vector<double> hi(n * d);
    g.collect_cell_bounds(span2d<double>(lo.data(), n, d), span2d<double>(hi.data(), n, d));
    return {lo, hi};
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// The constructor rejects a malformed factor, on both counts.
void test_construction_validates() {
    bool threw = false;
    try {
        const Ints wrong_length{2, 2};
        const Grid g(root_1d(), wrong_length);
        (void)g;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    PANTR_CHECK_MSG(threw, "a factor of the wrong length must be refused");

    threw = false;
    try {
        const Ints zero{0};
        const Grid g(root_1d(), zero);
        (void)g;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    PANTR_CHECK_MSG(threw, "a factor entry below 1 must be refused");

    // 1 is legal and means "no subdivision on that axis", which is not the same as 0.
    const Ints one{1};
    const Grid g(root_1d(), one);
    PANTR_CHECK(g.num_cells() == 4);
}

/// An unrefined hierarchy is its root, cell for cell and corner for corner.
///
/// Bitwise on the corners: level 0 divides the root cell's width by `factor ** 0 == 1`
/// and adds an offset of zero, so every operation is exact and any difference at all
/// would be a defect rather than a rounding.
void test_the_unrefined_grid_is_its_root() {
    const Grid g = flat_1d();
    const TensorProductGrid<double>& root = g.root();

    PANTR_CHECK(g.ndim() == 1);
    PANTR_CHECK(g.num_cells() == root.num_cells());
    PANTR_CHECK(g.max_level() == 0);

    const auto [lo, hi] = g.active_blocks(0);
    PANTR_CHECK(lo.extent(0) == 1);
    PANTR_CHECK(lo(0, 0) == 0);
    PANTR_CHECK(hi(0, 0) == 4);

    for (std::int64_t cid = 0; cid < g.num_cells(); ++cid) {
        std::vector<double> rlo(1);
        std::vector<double> rhi(1);
        root.cell_bounds(cid, rlo, rhi);
        const auto got = bounds(g, cid);
        PANTR_CHECK_MSG(got.first == rlo && got.second == rhi,
                        "cell " + std::to_string(cid) + " must be the root's own cell");
        PANTR_CHECK(g.cell_level(cid) == 0);
    }
}

/// The refined fixtures are what their comments claim.
void test_the_fixtures_are_what_they_claim() {
    const Grid g = refined_1d();
    PANTR_CHECK(g.num_cells() == 6);
    PANTR_CHECK(g.max_level() == 1);
    // The two coarse cells come first: level 0 is packed before level 1.
    PANTR_CHECK(g.cell_level(0) == 0);
    PANTR_CHECK(g.cell_level(1) == 0);
    PANTR_CHECK(g.cell_level(2) == 1);
    PANTR_CHECK(g.cell_level(5) == 1);
    PANTR_CHECK(bounds(g, 0).first == std::vector<double>({2.0}));
    PANTR_CHECK(bounds(g, 2).first == std::vector<double>({0.0}));
    PANTR_CHECK(bounds(g, 2).second == std::vector<double>({0.5}));

    const Grid h = refined_2d();
    PANTR_CHECK(h.num_cells() == 7);
    PANTR_CHECK(h.max_level() == 1);
    PANTR_CHECK(h.ndim() == 2);
}

// ---------------------------------------------------------------------------
// Addressing
// ---------------------------------------------------------------------------

/// `cell_id` inverts `cell_level` and `cell_multi_index`, for every cell.
void test_the_address_round_trips(const Grid& g, const char* what) {
    const auto d = static_cast<std::size_t>(g.ndim());
    for (std::int64_t cid = 0; cid < g.num_cells(); ++cid) {
        std::vector<std::int64_t> midx(d);
        g.cell_multi_index(cid, midx);
        const std::int64_t lvl = g.cell_level(cid);
        const std::optional<std::int64_t> back = g.cell_id(lvl, midx);
        PANTR_CHECK_MSG(back.has_value() && *back == cid,
                        std::string(what) + ": cell " + std::to_string(cid)
                            + " did not round-trip through (level, midx)");
        PANTR_CHECK(g.is_active_leaf(lvl, midx));
    }
}

/// A position that is not an active leaf answers empty rather than guessing.
void test_inactive_positions_have_no_id() {
    const Grid g = refined_1d();
    // Level 0 cells 0 and 1 were refined away.
    PANTR_CHECK(!g.cell_id(0, Ints{0}).has_value());
    PANTR_CHECK(!g.cell_id(0, Ints{1}).has_value());
    PANTR_CHECK(g.cell_id(0, Ints{2}).has_value());
    // Out of range on every axis of failure.
    PANTR_CHECK(!g.cell_id(0, Ints{-1}).has_value());
    PANTR_CHECK(!g.cell_id(0, Ints{4}).has_value());
    PANTR_CHECK(!g.cell_id(-1, Ints{0}).has_value());
    PANTR_CHECK(!g.cell_id(2, Ints{0}).has_value());
    PANTR_CHECK(!g.is_active_leaf(0, Ints{0}));
}

/// An out-of-range id is an error, not a wrong answer.
void test_cell_id_range_is_checked() {
    const Grid g = refined_1d();
    bool threw = false;
    try {
        (void)g.cell_level(6);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    PANTR_CHECK(threw);
}

// ---------------------------------------------------------------------------
// Geometry: two unrelated paths, and closed form
// ---------------------------------------------------------------------------

/// `cell_bounds` and `collect_cell_bounds` agree bitwise on every cell.
///
/// One decodes a flat id by binary search and evaluates one cell; the other walks blocks
/// with an odometer and evaluates all of them. They share no code. Bitwise rather than
/// close, because they evaluate the same expression in the same order on the same
/// inputs, so any difference is a defect and not a rounding.
void test_the_two_bound_paths_agree(const Grid& g, const char* what) {
    const auto d = static_cast<std::size_t>(g.ndim());
    const auto [lo, hi] = all_bounds(g);
    for (std::int64_t cid = 0; cid < g.num_cells(); ++cid) {
        const auto one = bounds(g, cid);
        for (std::size_t k = 0; k < d; ++k) {
            const std::size_t at = static_cast<std::size_t>(cid) * d + k;
            PANTR_CHECK_MSG(one.first[k] == lo[at] && one.second[k] == hi[at],
                            std::string(what) + ": cell " + std::to_string(cid) + " axis "
                                + std::to_string(k) + " differs between the two paths");
        }
    }
}

/// Locating each cell's own midpoint returns that cell.
///
/// `locate` descends levels and truncates a quotient at each one; `cell_bounds` decodes
/// and evaluates. Neither can satisfy this by mirroring the other.
void test_locate_finds_each_cell(const Grid& g, const char* what) {
    const auto d = static_cast<std::size_t>(g.ndim());
    std::vector<double> mid(d);
    for (std::int64_t cid = 0; cid < g.num_cells(); ++cid) {
        const auto one = bounds(g, cid);
        for (std::size_t k = 0; k < d; ++k) {
            mid[k] = 0.5 * (one.first[k] + one.second[k]);
        }
        const std::optional<std::int64_t> got = g.locate(mid);
        PANTR_CHECK_MSG(got.has_value() && *got == cid,
                        std::string(what) + ": the midpoint of cell " + std::to_string(cid)
                            + " located to "
                            + (got ? std::to_string(*got) : std::string("nothing")));
    }
}

/// The scalar descent and the batch kernel return the same ids.
///
/// Two implementations of one descent -- a member walking `std::optional`s and a kernel
/// over the packed descriptor -- so this is a real cross-check rather than a restatement.
/// Points on cell boundaries are included deliberately: those are where a truncation
/// disagreement would first show, and Rule 11 says no tolerance bounds a changed id.
void test_scalar_and_batch_locate_agree() {
    const Grid g = refined_1d();
    const std::vector<double> xs{-0.5,  0.0, 0.25, 0.5, 0.75, 1.0, 1.25,
                                 1.5,   2.0, 2.5,  3.0, 3.5,  4.0, 4.5};
    const std::vector<std::int64_t> batch =
        g.locate_many(span2d<const double>(xs.data(), xs.size(), 1));
    PANTR_CHECK(batch.size() == xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const std::optional<std::int64_t> one = g.locate(std::span<const double>(&xs[i], 1));
        const std::int64_t as_batch = one ? *one : -1;
        PANTR_CHECK_MSG(batch[i] == as_batch,
                        "x = " + std::to_string(xs[i]) + ": scalar gave "
                            + std::to_string(as_batch) + ", batch gave "
                            + std::to_string(batch[i]));
    }
}

/// Sorted by lower corner, the 1-D cells abut exactly and span the whole domain.
///
/// Exact on a dyadic fixture: a shared corner is `root_lo + j * size` on one side and
/// `root_lo + (j-1) * size + size` on the other, and with dyadic `size` both are exact.
/// That is why this assertion is confined to the dyadic fixtures; the non-dyadic one is
/// checked against its closed form with a derived bound instead.
void test_the_cells_tile_the_domain(const Grid& g, const char* what) {
    const auto [lo, hi] = all_bounds(g);
    std::vector<std::size_t> order(lo.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(),
              [&lo](std::size_t a, std::size_t b) { return lo[a] < lo[b]; });
    const std::span<const double> bp = g.root().breakpoints(0);
    PANTR_CHECK_MSG(lo[order.front()] == bp.front(),
                    std::string(what) + ": the first cell must start at the domain");
    PANTR_CHECK_MSG(hi[order.back()] == bp.back(),
                    std::string(what) + ": the last cell must end at the domain");
    for (std::size_t i = 1; i < order.size(); ++i) {
        PANTR_CHECK_MSG(hi[order[i - 1]] == lo[order[i]],
                        std::string(what) + ": cells must abut exactly");
    }
}

/// **Closed form.** On a dyadic hierarchy every corner is the exact binary rational.
///
/// A level-`l` cell `k` of the unit interval spans `[k / 2**l, (k+1) / 2**l]`, and every
/// one of those is representable, so the computed value must equal it **bitwise**. This
/// is an oracle-free statement: it compares the code against the mathematics, which is
/// what `design/backend_parity.md` asks of an accuracy check.
void test_dyadic_bounds_are_exact() {
    for (std::int64_t depth = 0; depth <= 10; ++depth) {
        const Grid g = dyadic_unit(depth);
        const std::int64_t n = std::int64_t{1} << depth;
        PANTR_CHECK(g.num_cells() == n);
        const double step = 1.0 / static_cast<double>(n);  // exact: a power of two
        for (std::int64_t cid = 0; cid < n; ++cid) {
            const auto one = bounds(g, cid);
            const double want_lo = static_cast<double>(cid) * step;
            const double want_hi = static_cast<double>(cid + 1) * step;
            PANTR_CHECK_MSG(one.first[0] == want_lo && one.second[0] == want_hi,
                            "depth " + std::to_string(depth) + " cell " + std::to_string(cid)
                                + ": got [" + std::to_string(one.first[0]) + ", "
                                + std::to_string(one.second[0]) + "]");
        }
    }
}

/// **The same check where the arithmetic is not exact**, both halves of it.
///
/// Root `[0, 0.1]`, factor 3, level 2: cell `k` should span `[0.1*k/9, 0.1*(k+1)/9]`, and
/// none of those is representable. The bound is derived rather than fitted:
///
///   * `w = fl(0.1) - 0.0` is exact (a subtraction of representable operands, and `0.1`
///     is itself the input, not a computed value);
///   * `size = w / 9` commits one rounding, so `size = (w/9)(1 + d1)`, `|d1| <= u`;
///   * `k * size` commits a second, and `lo + (k*size)` a third, giving
///     `|lo_computed - w*k/9| <= 3u * |w*k/9| + O(u^2)`, and `w*k/9 <= 0.1`.
///
/// With `u = 2^-53` that is `4 * u * 0.1` after rounding the constant up, which is what
/// is asserted. The reference value `0.1 * k / 9` is itself computed in `double` and
/// carries two roundings of its own, so the comparison is against something within `2u`
/// of the exact rational -- absorbed by taking the factor 4 rather than 3.
///
/// **And the error is asserted to be nonzero somewhere**, because a bound checked only
/// on inputs that happen to be exact has not been checked at all: without that half,
/// deleting the arithmetic and returning the reference would pass.
void test_non_dyadic_bounds_are_bounded_and_not_exact() {
    const Grid g = non_dyadic();
    PANTR_CHECK(g.num_cells() == 9);

    constexpr double kUnitRoundoff = 0.5 * std::numeric_limits<double>::epsilon();
    const double bound = 4.0 * kUnitRoundoff * 0.1;

    double worst = 0.0;
    for (std::int64_t cid = 0; cid < 9; ++cid) {
        const auto one = bounds(g, cid);
        const double want = 0.1 * static_cast<double>(cid) / 9.0;
        const double err = std::abs(one.first[0] - want);
        worst = std::max(worst, err);
        PANTR_CHECK_MSG(err <= bound, "cell " + std::to_string(cid) + " lo is off by "
                                          + std::to_string(err));
    }
    PANTR_CHECK_MSG(worst > 0.0,
                    "every non-dyadic corner came out exact, so this bound tested nothing");
}

// ---------------------------------------------------------------------------
// The hooks, each against the default it hides
// ---------------------------------------------------------------------------

/// `cell_level` reports real levels where the generic default always says zero.
///
/// The differential that proves name hiding routed the call: if the hook were not picked
/// up, the two would agree everywhere and this would pass silently, so the disagreement
/// is asserted rather than the agreement.
void test_cell_level_hook_replaces_the_default() {
    const Grid g = refined_1d();
    bool differs = false;
    for (std::int64_t cid = 0; cid < g.num_cells(); ++cid) {
        const std::int64_t generic = g.pantr::grid::GridBase<Grid>::cell_level(cid);
        PANTR_CHECK(generic == 0);
        differs = differs || (g.cell_level(cid) != generic);
    }
    PANTR_CHECK_MSG(differs, "the cell_level hook never differed from the default");
}

/// The batch locate hook agrees with the mixin default, which loops the primitive.
void test_locate_many_agrees_with_the_default() {
    const Grid g = refined_1d();
    const std::vector<double> xs{0.0, 0.3, 0.6, 1.2, 1.9, 2.4, 3.7, 4.0, -1.0, 9.0};
    const span2d<const double> view(xs.data(), xs.size(), 1);
    PANTR_CHECK(g.locate_many(view) == g.pantr::grid::GridBase<Grid>::locate_many(view));
}

/// The bulk bounds hook agrees with the mixin default bitwise.
///
/// The default calls `cell_bounds` once per cell; the hook runs the block odometer. This
/// is the same statement as `test_the_two_bound_paths_agree` reached through the
/// dispatch mechanism, and it is worth both: one says the two algorithms agree, this one
/// says the specialisation is the thing being called.
void test_collect_cell_bounds_agrees_with_the_default() {
    const Grid g = refined_2d();
    const auto n = static_cast<std::size_t>(g.num_cells());
    const auto d = static_cast<std::size_t>(g.ndim());
    std::vector<double> hook_lo(n * d);
    std::vector<double> hook_hi(n * d);
    std::vector<double> base_lo(n * d);
    std::vector<double> base_hi(n * d);
    g.collect_cell_bounds(span2d<double>(hook_lo.data(), n, d),
                          span2d<double>(hook_hi.data(), n, d));
    g.pantr::grid::GridBase<Grid>::collect_cell_bounds(span2d<double>(base_lo.data(), n, d),
                                      span2d<double>(base_hi.data(), n, d));
    PANTR_CHECK(hook_lo == base_lo);
    PANTR_CHECK(hook_hi == base_hi);
}

/// Boundary facets, hand-computed, and against the generic neighbour loop.
///
/// On `refined_1d` the domain is `[0, 4]`. Its outer boundary is two facets: the low face
/// of the cell starting at 0 (flat id 2, the first level-1 cell, `lfid` 0) and the high
/// face of the cell ending at 4 (flat id 1, `lfid` 1). Everything else is an interface.
void test_boundary_facets() {
    const Grid g = refined_1d();
    const std::vector<std::int64_t> got = g.boundary_facets();
    PANTR_CHECK(got == std::vector<std::int64_t>({1, 1, 2, 0}));

    // The generic default asks `neighbor_across_facet` per facet, which is an unrelated
    // route to the same answer -- and it is the route that would notice if the block-face
    // shortcut were wrong about which faces are outer.
    PANTR_CHECK(got == g.pantr::grid::GridBase<Grid>::boundary_facets());
    const Grid h = refined_2d();
    PANTR_CHECK(h.boundary_facets() == h.pantr::grid::GridBase<Grid>::boundary_facets());
    const Grid f = flat_1d();
    PANTR_CHECK(f.boundary_facets() == f.pantr::grid::GridBase<Grid>::boundary_facets());
}

/// Neighbours across a hanging interface, in both directions, hand-computed.
///
/// On `refined_1d`: flat id 0 is the coarse cell `[2,3]`, and across its low facet lies
/// the fine cell `[1.5,2]`, which is flat id 5. From the fine side, cell 5's high facet
/// leads back to the coarse cell 0. Both are level-crossing, which is the case the
/// generic default cannot handle.
void test_neighbours_across_a_hanging_interface() {
    const Grid g = refined_1d();
    PANTR_CHECK(g.neighbor_across_facet(0, 0) == std::optional<std::int64_t>(5));
    PANTR_CHECK(g.neighbor_across_facet(5, 1) == std::optional<std::int64_t>(0));
    // Conforming, within one level.
    PANTR_CHECK(g.neighbor_across_facet(0, 1) == std::optional<std::int64_t>(1));
    PANTR_CHECK(g.neighbor_across_facet(2, 1) == std::optional<std::int64_t>(3));
    // The outer boundary.
    PANTR_CHECK(!g.neighbor_across_facet(2, 0).has_value());
    PANTR_CHECK(!g.neighbor_across_facet(1, 1).has_value());
}

/// `hanging_neighbors` returns every fine neighbour, and its first is the primitive's.
///
/// In 2-D the coarse cell `(1,0)` of `refined_2d` abuts the two fine children of root
/// cell `(0,0)` along its low-`x` face, so this is the case where the generic default --
/// which wraps the single answer from `neighbor_across_facet` -- is genuinely wrong, and
/// the disagreement is asserted.
void test_hanging_neighbors_collect_the_fine_side() {
    const Grid g = refined_2d();
    // Flat ids: level 0 blocks are `[0,1)x[1,2)` (id 0) then `[1,2)x[0,2)` (ids 1, 2);
    // level 1 block `[0,2)x[0,2)` gives ids 3..6 in C-order over (i, j).
    PANTR_CHECK(g.num_cells() == 7);
    const std::vector<std::int64_t> fine = g.hanging_neighbors(1, 0);
    PANTR_CHECK_MSG(fine.size() == 2, "the coarse cell's low-x face abuts two fine cells, got "
                                          + std::to_string(fine.size()));
    PANTR_CHECK(g.neighbor_across_facet(1, 0) == std::optional<std::int64_t>(fine.front()));
    PANTR_CHECK_MSG(fine != g.pantr::grid::GridBase<Grid>::hanging_neighbors(1, 0),
                    "the hook must differ from the default on a hanging interface");

    // Every id it returns is an active leaf whose level is finer than the queried cell's.
    for (const std::int64_t nbr : fine) {
        PANTR_CHECK(nbr >= 0 && nbr < g.num_cells());
        PANTR_CHECK(g.cell_level(nbr) > g.cell_level(1));
    }
}

/// A neighbour query is symmetric where the interface is conforming.
///
/// Not asserted across a hanging interface, where it is false by construction: the
/// coarse side names one of the fine cells and the others name the coarse one back.
void test_conforming_neighbours_are_symmetric() {
    const Grid g = refined_2d();
    for (std::int64_t cid = 0; cid < g.num_cells(); ++cid) {
        for (std::int64_t lfid = 0; lfid < 2 * g.ndim(); ++lfid) {
            const std::optional<std::int64_t> nbr = g.neighbor_across_facet(cid, lfid);
            if (!nbr || g.cell_level(*nbr) != g.cell_level(cid)) {
                continue;
            }
            const std::int64_t back = lfid ^ 1;  // the opposite face on the same axis
            PANTR_CHECK_MSG(g.neighbor_across_facet(*nbr, back)
                                == std::optional<std::int64_t>(cid),
                            "cell " + std::to_string(cid) + " facet " + std::to_string(lfid)
                                + " is not symmetric");
        }
    }
}

// ---------------------------------------------------------------------------
// The active set
// ---------------------------------------------------------------------------

/// The two masks, hand-computed, plus the invariant that ties them to `num_cells`.
void test_masks() {
    const Grid g = refined_1d();

    // Level 0: cells 2 and 3 are active leaves; 0 and 1 were refined away.
    PANTR_CHECK(g.active_leaf_mask(0) == std::vector<std::uint8_t>({0, 0, 1, 1}));
    // Level 1: all eight positions exist, and the first four are the active children.
    PANTR_CHECK(g.active_leaf_mask(1) == std::vector<std::uint8_t>({1, 1, 1, 1, 0, 0, 0, 0}));

    // Omega_0 is everything; Omega_1 excludes what a coarser leaf already covers, which
    // is root cells 2 and 3, i.e. level-1 positions 4..7.
    PANTR_CHECK(g.subdomain_mask(0) == std::vector<std::uint8_t>({1, 1, 1, 1}));
    PANTR_CHECK(g.subdomain_mask(1) == std::vector<std::uint8_t>({1, 1, 1, 1, 0, 0, 0, 0}));

    // The invariant: the active leaves of every level, counted together, are the cells.
    std::int64_t total = 0;
    for (std::int64_t l = 0; l <= g.max_level(); ++l) {
        const std::vector<std::uint8_t> mask = g.active_leaf_mask(l);
        total += std::count(mask.begin(), mask.end(), std::uint8_t{1});
    }
    PANTR_CHECK(total == g.num_cells());

    // An active leaf is always inside its own level's subdomain; the converse is false,
    // which is what makes the two masks different objects.
    for (std::int64_t l = 0; l <= g.max_level(); ++l) {
        const std::vector<std::uint8_t> leaves = g.active_leaf_mask(l);
        const std::vector<std::uint8_t> omega = g.subdomain_mask(l);
        PANTR_CHECK(leaves.size() == omega.size());
        for (std::size_t i = 0; i < leaves.size(); ++i) {
            PANTR_CHECK(leaves[i] == 0 || omega[i] == 1);
        }
    }
}

/// `level_cells_per_axis` is a formula and answers above `max_level` too.
void test_level_cells_per_axis() {
    const Grid g = refined_1d();
    PANTR_CHECK(g.level_cells_per_axis(0, 0) == 4);
    PANTR_CHECK(g.level_cells_per_axis(1, 0) == 8);
    // Deliberately above max_level: this one is a pure formula, unlike active_blocks.
    PANTR_CHECK(g.level_cells_per_axis(5, 0) == 128);

    bool threw = false;
    try {
        (void)g.level_cells_per_axis(-1, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    PANTR_CHECK(threw);

    threw = false;
    try {
        (void)g.level_cells_per_axis(0, 1);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    PANTR_CHECK(threw);
}

/// A level count past `int64` is refused rather than wrapping into a positive lie.
///
/// The oracle counts in Python integers and cannot reach this, so it is a deliberate
/// divergence -- the same one `TensorProductGrid` already makes for its own cell-count
/// product. Both sides of the boundary are asserted, so a guard that rejected every large
/// level would not satisfy this.
void test_level_count_overflow_is_reported() {
    const Grid g = flat_1d();  // 4 root cells, factor 2

    // 4 * 2**60 = 2**62, which fits.
    PANTR_CHECK(g.level_cells_per_axis(60, 0) == (std::int64_t{1} << 62));

    // 4 * 2**61 = 2**63, which does not.
    bool threw = false;
    try {
        (void)g.level_cells_per_axis(61, 0);
    } catch (const std::overflow_error&) {
        threw = true;
    }
    PANTR_CHECK_MSG(threw, "a level whose cell count exceeds int64 must be refused");
}

/// `active_blocks` rejects a level outside the hierarchy, on both sides.
void test_active_blocks_range() {
    const Grid g = refined_1d();
    const auto [lo, hi] = g.active_blocks(1);
    PANTR_CHECK(lo.extent(0) == 1 && lo(0, 0) == 0 && hi(0, 0) == 4);

    for (const std::int64_t bad : {std::int64_t{-1}, std::int64_t{2}}) {
        bool threw = false;
        try {
            (void)g.active_blocks(bad);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        PANTR_CHECK(threw);
    }
}

// ---------------------------------------------------------------------------
// from_blocks
// ---------------------------------------------------------------------------

/// Declaring a clean level clean changes nothing, which is what the fast path claims.
///
/// The property `refine` and `coarsen` will rest on: a level holding a previous call's
/// output normalizes to itself, so skipping the re-run returns the same blocks in the
/// same order -- and therefore the same flat cell ids. Checked by building one grid both
/// ways and comparing every id's geometry, not just the count, because a re-partition
/// preserves the count and moves the ids.
void test_naming_a_clean_level_changes_nothing() {
    const Grid reference = refined_1d();

    // Rebuild from the reference's own (already normalized) blocks, once declaring every
    // level clean and once normalizing everything.
    std::vector<BlockList> blocks;
    for (std::int64_t l = 0; l <= reference.max_level(); ++l) {
        const auto [lo, hi] = reference.active_blocks(l);
        BlockList list(reference.ndim());
        for (std::size_t b = 0; b < lo.extent(0); ++b) {
            list.push_back(BlockView{std::span<const std::int64_t>(&lo(b, 0), lo.extent(1)),
                                     std::span<const std::int64_t>(&hi(b, 0), hi.extent(1))});
        }
        blocks.push_back(std::move(list));
    }
    std::vector<BlockList> copy = blocks;

    const Ints factor{2};
    const Ints none{};
    const Grid clean =
        Grid::from_blocks(root_1d(), factor, std::move(blocks), std::span<const std::int64_t>(none));
    const Grid renormalized =
        Grid::from_blocks(root_1d(), factor, std::move(copy), std::nullopt);

    PANTR_CHECK(clean.num_cells() == renormalized.num_cells());
    for (std::int64_t cid = 0; cid < clean.num_cells(); ++cid) {
        PANTR_CHECK_MSG(bounds(clean, cid) == bounds(renormalized, cid),
                        "cell " + std::to_string(cid)
                            + " moved when a clean level was declared clean");
    }
}

/// Trailing empty levels are dropped, so `max_level` means what it says.
void test_trailing_empty_levels_are_dropped() {
    const Ints factor{2};
    std::vector<BlockList> blocks;
    blocks.push_back(level(1, {{0, 4}}));
    blocks.push_back(level(1, {}));
    blocks.push_back(level(1, {}));
    const Grid g = Grid::from_blocks(root_1d(), factor, std::move(blocks), std::nullopt);
    PANTR_CHECK(g.max_level() == 0);
    PANTR_CHECK(g.num_cells() == 4);
}

/// The representation is the oracle's, character for character.
void test_to_string() {
    PANTR_CHECK_MSG(refined_1d().to_string()
                        == "HierarchicalGrid(ndim=1, root_cells=(4,), factor=(2,), "
                           "num_cells=6, max_level=1)",
                    refined_1d().to_string());
    PANTR_CHECK_MSG(refined_2d().to_string()
                        == "HierarchicalGrid(ndim=2, root_cells=(2, 2), factor=(2, 2), "
                           "num_cells=7, max_level=1)",
                    refined_2d().to_string());
}

/// `restrict` is not specialised yet, so the mixin's throwing default must still stand.
void test_restrict_still_throws() {
    const Grid g = flat_1d();
    bool threw = false;
    try {
        const Ints ids{0};
        (void)g.restrict(ids);
    } catch (const std::logic_error&) {
        threw = true;
    }
    PANTR_CHECK_MSG(threw, "restrict must still be the mixin's throwing default");
}

}  // namespace

// The census, at both scalars. `float` is a compile-time device only: it forces every
// default body at a scalar no binding registers, which is what catches a hook that
// hard-codes `double`. It is never bound and never compared, so it opens no parity claim.
PANTR_GRID_CENSUS(pantr::grid::HierarchicalGrid<double>);
PANTR_GRID_CENSUS(pantr::grid::HierarchicalGrid<float>);

int main() {
    test_construction_validates();
    test_the_unrefined_grid_is_its_root();
    test_the_fixtures_are_what_they_claim();

    test_the_address_round_trips(flat_1d(), "the flat grid");
    test_the_address_round_trips(refined_1d(), "the refined 1-D grid");
    test_the_address_round_trips(refined_2d(), "the refined 2-D grid");
    test_inactive_positions_have_no_id();
    test_cell_id_range_is_checked();

    test_the_two_bound_paths_agree(flat_1d(), "the flat grid");
    test_the_two_bound_paths_agree(refined_1d(), "the refined 1-D grid");
    test_the_two_bound_paths_agree(refined_2d(), "the refined 2-D grid");
    test_the_two_bound_paths_agree(non_dyadic(), "the non-dyadic grid");
    test_locate_finds_each_cell(flat_1d(), "the flat grid");
    test_locate_finds_each_cell(refined_1d(), "the refined 1-D grid");
    test_locate_finds_each_cell(refined_2d(), "the refined 2-D grid");
    test_scalar_and_batch_locate_agree();
    test_the_cells_tile_the_domain(flat_1d(), "the flat grid");
    test_the_cells_tile_the_domain(refined_1d(), "the refined 1-D grid");
    test_the_cells_tile_the_domain(dyadic_unit(6), "the dyadic unit grid");
    test_dyadic_bounds_are_exact();
    test_non_dyadic_bounds_are_bounded_and_not_exact();

    test_cell_level_hook_replaces_the_default();
    test_locate_many_agrees_with_the_default();
    test_collect_cell_bounds_agrees_with_the_default();
    test_boundary_facets();
    test_neighbours_across_a_hanging_interface();
    test_hanging_neighbors_collect_the_fine_side();
    test_conforming_neighbours_are_symmetric();

    test_masks();
    test_level_cells_per_axis();
    test_level_count_overflow_is_reported();
    test_active_blocks_range();

    test_naming_a_clean_level_changes_nothing();
    test_trailing_empty_levels_are_dropped();
    test_to_string();
    test_restrict_still_throws();
    return pantr::test::summary("test_grid_hierarchical_type");
}
