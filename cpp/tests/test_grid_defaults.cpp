/// \file
/// The generic grid layer's defaults, checked against hand-computed values on a
/// synthetic grid that declares no specialisations at all.
///
/// Two fixtures, and the split is the point:
///
///   * `BoxGrid<T>` declares `Hook::none`, so every one of the fourteen replaceable
///     defaults RUNS here, and the census asserts the negative. Comparing the defaults
///     against a real specialisation belongs to the `TensorProductGrid` ticket, which
///     is where the specialisations first exist.
///   * `HookedGrid<T>` declares two hooks, and exists to pin the two halves of the
///     dispatch mechanism the next ticket depends on: name hiding reaches the hook, and
///     the qualified call `g.Base::name()` still reaches the hidden default. That
///     qualified call is the C++ analogue of `Grid.boundary_facets(g)` in
///     `tests/test_grid_hierarchical.py`, and it is what a differential test needs.
///
/// Both are censused at `float` AND at `double`. The `float` instantiation is a
/// COMPILE-TIME census device and nothing else: it forces every default body at a
/// scalar no binding registers, which is what catches a hook that hard-codes `double`.
/// It is never bound and never compared, so it does not open a parity claim --
/// `design/backend_parity.md` Rule 8 forbids one where there is no oracle behind it,
/// and `src/pantr/grid/_grid_backend.py` records that `pantr.grid`'s oracle is
/// `float64`-only.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "check.hpp"
#include "pantr/grid/grid.hpp"

namespace {

using pantr::span2d;
using pantr::geometry::AABB;
using pantr::grid::GridBase;
using pantr::grid::GridLike;
using pantr::grid::GridRestriction;
using pantr::grid::Hook;

// ---------------------------------------------------------------------------
// The zero-hook fixture
// ---------------------------------------------------------------------------

template <class T>
class BoxGrid;

}  // namespace

template <class T>
struct pantr::grid::grid_traits<BoxGrid<T>> {
    using scalar_type = T;
    static constexpr Hook hooks = Hook::none;
};

namespace {

/// A `nx` by `ny` grid of unit cells with its lo corner at the origin.
///
/// Row-major cell ids: `cid = i * ny + j`, cell `(i, j)` spanning
/// `[i, i + 1] x [j, j + 1]`. Everything about it is computable by hand, which is the
/// only property it needs.
template <class T>
class BoxGrid : public GridBase<BoxGrid<T>> {
  public:
    /// The mixin, named publicly so a differential test can reach the hidden defaults.
    using Base = GridBase<BoxGrid<T>>;

    BoxGrid(std::int64_t nx, std::int64_t ny) : Base(2, nx * ny), nx_(nx), ny_(ny) {}

    void cell_bounds(std::int64_t cid, std::span<T> lo, std::span<T> hi) const {
        this->check_cid(cid);
        ++cell_bounds_calls_;
        const std::int64_t i = cid / ny_;
        const std::int64_t j = cid % ny_;
        lo[0] = static_cast<T>(i);
        lo[1] = static_cast<T>(j);
        hi[0] = static_cast<T>(i + 1);
        hi[1] = static_cast<T>(j + 1);
    }

    [[nodiscard]] std::optional<std::int64_t> locate(std::span<const T> pt) const {
        const std::optional<std::int64_t> i = axis_index(pt[0], nx_);
        const std::optional<std::int64_t> j = axis_index(pt[1], ny_);
        if (!i || !j) {
            return std::nullopt;
        }
        return (*i * ny_) + *j;
    }

    [[nodiscard]] std::optional<std::int64_t> neighbor_across_facet(std::int64_t cid,
                                                                    std::int64_t lfid) const {
        this->check_lfid(cid, lfid);
        std::int64_t i = cid / ny_;
        std::int64_t j = cid % ny_;
        const std::int64_t axis = lfid / 2;
        const std::int64_t step = (lfid % 2 == 0) ? -1 : 1;
        (axis == 0 ? i : j) += step;
        if (i < 0 || i >= nx_ || j < 0 || j >= ny_) {
            return std::nullopt;
        }
        return (i * ny_) + j;
    }

    /// How many times the primitive ran; the only way to see the BVH cache from outside.
    [[nodiscard]] std::int64_t cell_bounds_calls() const noexcept { return cell_bounds_calls_; }

  private:
    /// The cell index containing `x` along an axis of `n` unit cells, closed at the top.
    [[nodiscard]] static std::optional<std::int64_t> axis_index(T x, std::int64_t n) {
        using pantr::value_of;
        const auto v = value_of(x);
        if (v < 0 || v > static_cast<decltype(v)>(n)) {
            return std::nullopt;
        }
        std::int64_t k = 0;
        while (k + 1 < n && v >= static_cast<decltype(v)>(k + 1)) {
            ++k;
        }
        return k;
    }

    std::int64_t nx_;
    std::int64_t ny_;
    mutable std::int64_t cell_bounds_calls_ = 0;
};

// ---------------------------------------------------------------------------
// A minimal grid, for the base's own construction checks
// ---------------------------------------------------------------------------

class RawGrid;

}  // namespace

template <>
struct pantr::grid::grid_traits<RawGrid> {
    using scalar_type = double;
    static constexpr Hook hooks = Hook::none;
};

namespace {

/// The smallest thing that is a grid: it forwards `(ndim, num_cells)` to the mixin
/// unchanged, which is what lets the base's own construction checks be reached.
class RawGrid : public GridBase<RawGrid> {
  public:
    RawGrid(std::int64_t ndim, std::int64_t num_cells) : GridBase<RawGrid>(ndim, num_cells) {}

    void cell_bounds(std::int64_t cid, std::span<double> lo, std::span<double> hi) const {
        this->check_cid(cid);
        std::fill(lo.begin(), lo.end(), 0.0);
        std::fill(hi.begin(), hi.end(), 1.0);
    }

    [[nodiscard]] std::optional<std::int64_t> locate(std::span<const double>) const {
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::int64_t> neighbor_across_facet(std::int64_t cid,
                                                                    std::int64_t lfid) const {
        this->check_lfid(cid, lfid);
        return std::nullopt;
    }
};

// ---------------------------------------------------------------------------
// The two-hook fixture
// ---------------------------------------------------------------------------

template <class T>
class HookedGrid;

}  // namespace

template <class T>
struct pantr::grid::grid_traits<HookedGrid<T>> {
    using scalar_type = T;
    static constexpr Hook hooks = Hook::boundary_facets | Hook::locate_many;
};

namespace {

/// The same geometry, with two defaults replaced by deliberately WRONG answers.
///
/// Wrong on purpose: a hook agreeing with its default would prove nothing about which
/// of the two ran.
template <class T>
class HookedGrid : public GridBase<HookedGrid<T>> {
  public:
    /// The mixin, named publicly so the qualified call below can reach it.
    using Base = GridBase<HookedGrid<T>>;

    HookedGrid(std::int64_t nx, std::int64_t ny) : Base(2, nx * ny), inner_(nx, ny) {}

    void cell_bounds(std::int64_t cid, std::span<T> lo, std::span<T> hi) const {
        inner_.cell_bounds(cid, lo, hi);
    }

    [[nodiscard]] std::optional<std::int64_t> locate(std::span<const T> pt) const {
        return inner_.locate(pt);
    }

    [[nodiscard]] std::optional<std::int64_t> neighbor_across_facet(std::int64_t cid,
                                                                    std::int64_t lfid) const {
        return inner_.neighbor_across_facet(cid, lfid);
    }

    [[nodiscard]] std::vector<std::int64_t> boundary_facets() const { return {-1, -1}; }

    [[nodiscard]] std::vector<std::int64_t> locate_many(span2d<const T> points) const {
        return std::vector<std::int64_t>(points.extent(0), -7);
    }

  private:
    BoxGrid<T> inner_;
};

}  // namespace

// ---------------------------------------------------------------------------
// The census (AC7): both fixtures, both scalars
// ---------------------------------------------------------------------------

PANTR_GRID_CENSUS(RawGrid);
PANTR_GRID_CENSUS(BoxGrid<double>);
PANTR_GRID_CENSUS(BoxGrid<float>);
PANTR_GRID_CENSUS(HookedGrid<double>);
PANTR_GRID_CENSUS(HookedGrid<float>);

namespace {

// The concept is a claim about what a grid IS, so it is worth pinning positively as
// well as through the census, and pinning what it excludes.
static_assert(GridLike<BoxGrid<double>>);
static_assert(GridLike<HookedGrid<float>>);
static_assert(!GridLike<int>);

/// A grid whose `cell_bounds` cannot write its output.
///
/// It satisfies a callability-based concept -- `std::span<T>` converts implicitly to
/// `std::span<const T>` -- and compiles, and returns whatever was already in the
/// caller's buffer. Measured accepted by g++ 14.4, g++ 10.5, clang++ 18.1 and
/// clang++ 10.0 alike. The exact member-pointer form of `GridLike` is what rejects it,
/// and this is the assertion that says so.
class UnwritableBoundsGrid;

}  // namespace

template <>
struct pantr::grid::grid_traits<UnwritableBoundsGrid> {
    using scalar_type = double;
    static constexpr Hook hooks = Hook::none;
};

namespace {

class UnwritableBoundsGrid : public GridBase<UnwritableBoundsGrid> {
  public:
    UnwritableBoundsGrid() : GridBase<UnwritableBoundsGrid>(1, 1) {}

    // NOT `std::span<double>`: this is the trap.
    void cell_bounds(std::int64_t, std::span<const double>, std::span<const double>) const {}

    [[nodiscard]] std::optional<std::int64_t> locate(std::span<const double>) const {
        return std::nullopt;
    }

    [[nodiscard]] std::optional<std::int64_t> neighbor_across_facet(std::int64_t,
                                                                    std::int64_t) const {
        return std::nullopt;
    }
};

static_assert(!GridLike<UnwritableBoundsGrid>,
              "a cell_bounds that cannot write its output must not satisfy GridLike");

// The detector itself, in both directions, on the two fixtures.
static_assert(!pantr::grid::detail::redeclares_boundary_facets<BoxGrid<double>>());
static_assert(pantr::grid::detail::redeclares_boundary_facets<HookedGrid<double>>());
static_assert(!pantr::grid::detail::redeclares_locate_many<BoxGrid<double>>());
static_assert(pantr::grid::detail::redeclares_locate_many<HookedGrid<double>>());
static_assert(!pantr::grid::detail::redeclares_num_local_facets<BoxGrid<double>>());

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

using Grid2 = BoxGrid<double>;

/// `(axis, side)`, aliased because the comma in the template argument list would
/// otherwise be read as a second macro argument by `PANTR_CHECK`.
using AxisSide = std::pair<std::int64_t, std::int64_t>;

/// The cell corners as a pair of vectors, for comparison against literals.
std::pair<std::vector<double>, std::vector<double>> bounds_of(const Grid2& g, std::int64_t cid) {
    std::vector<double> lo(2);
    std::vector<double> hi(2);
    g.cell_bounds(cid, lo, hi);
    return {lo, hi};
}

bool spans_equal(std::span<const double> a, std::initializer_list<double> b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

// ---------------------------------------------------------------------------
// The defaults, against hand-computed values
// ---------------------------------------------------------------------------

/// Whether `GridBase` refuses these sizes with `std::invalid_argument`.
bool rejects_construction(std::int64_t ndim, std::int64_t num_cells) {
    try {
        const RawGrid bad(ndim, num_cells);
        static_cast<void>(bad);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void test_state() {
    const Grid2 g(3, 2);
    PANTR_CHECK(g.ndim() == 2);
    PANTR_CHECK(g.num_cells() == 6);

    // The mixin's own construction checks, reached through the grid that forwards
    // `(ndim, num_cells)` unchanged. Both must be rejected BEFORE the tag registries
    // see them, or the message blames `facets_per_cell` for a bad `ndim`.
    PANTR_CHECK(rejects_construction(0, 4));
    PANTR_CHECK(rejects_construction(-1, 4));
    PANTR_CHECK(rejects_construction(2, -1));

    const RawGrid empty(3, 0);
    PANTR_CHECK_MSG(empty.num_cells() == 0, "a grid with no cells is well formed");
    PANTR_CHECK(empty.boundary_facets().empty());
    PANTR_CHECK(empty.facet_tags().facets_per_cell() == 6);
}

void test_cell_accessors() {
    const Grid2 g(3, 2);

    // cid 3 is (i, j) = (1, 1), spanning [1, 2] x [1, 2].
    const auto [lo, hi] = bounds_of(g, 3);
    PANTR_CHECK(lo[0] == 1.0 && lo[1] == 1.0 && hi[0] == 2.0 && hi[1] == 2.0);

    const AABB<double> box = g.cell_aabb(3);
    PANTR_CHECK(spans_equal(box.lo(), {1.0, 1.0}));
    PANTR_CHECK(spans_equal(box.hi(), {2.0, 2.0}));

    PANTR_CHECK(g.cell_level(3) == 0);
    PANTR_CHECK(g.child_cells(3).empty());

    // reference_map is diag(hi - lo) u + lo, so the unit cube's hi corner is the cell's.
    const auto map = g.reference_map(3);
    PANTR_CHECK(map.dim() == 2);
    PANTR_CHECK(spans_equal(map.offset(), {1.0, 1.0}));
    PANTR_CHECK(pantr::at(map.matrix(), 0, 0) == 1.0);
    PANTR_CHECK(pantr::at(map.matrix(), 1, 1) == 1.0);
    PANTR_CHECK(pantr::at(map.matrix(), 0, 1) == 0.0);
    PANTR_CHECK(pantr::at(map.matrix(), 1, 0) == 0.0);
}

void test_neighbors() {
    const Grid2 g(3, 2);
    // Cell 0 = (0, 0): a corner, so two neighbours -- (1, 0) = 2 and (0, 1) = 1.
    PANTR_CHECK(g.neighbors(0) == std::vector<std::int64_t>({2, 1}));
    // Cell 2 = (1, 0): an edge cell, three neighbours -- (0, 0), (2, 0), (1, 1).
    PANTR_CHECK(g.neighbors(2) == std::vector<std::int64_t>({0, 4, 3}));
    // hanging_neighbors wraps the same answer, and is empty on the boundary.
    PANTR_CHECK(g.hanging_neighbors(0, 1) == std::vector<std::int64_t>({2}));
    PANTR_CHECK(g.hanging_neighbors(0, 0).empty());
}

void test_facets() {
    const Grid2 g(3, 2);
    PANTR_CHECK(g.num_local_facets(0) == 4);
    PANTR_CHECK(g.local_facet_axis_side(0, 3) == AxisSide(1, 1));
    PANTR_CHECK(g.local_facet_axis_side(0, 0) == AxisSide(0, 0));

    // Facet 0 of cell 0 is the low face on axis 0: degenerate there, full extent on axis 1.
    std::vector<double> lo(2);
    std::vector<double> hi(2);
    g.local_facet_bounds(0, 0, lo, hi);
    PANTR_CHECK(lo[0] == 0.0 && hi[0] == 0.0);
    PANTR_CHECK(lo[1] == 0.0 && hi[1] == 1.0);

    g.local_facet_bounds(0, 3, lo, hi);
    PANTR_CHECK(lo[1] == 1.0 && hi[1] == 1.0);
    PANTR_CHECK(lo[0] == 0.0 && hi[0] == 1.0);

    PANTR_CHECK(g.is_mesh_boundary_facet(0, 0));
    PANTR_CHECK(!g.is_mesh_boundary_facet(0, 1));
}

void test_boundary_facets() {
    const Grid2 g(3, 2);
    const std::vector<std::int64_t> rows = g.boundary_facets();
    // Every cell of a 3 x 2 grid touches the boundary. Interior facet count is
    // 2 * (2 * 2 + 3 * 1) = 14 of the 24, leaving 10 boundary facets.
    PANTR_CHECK(rows.size() == 20);
    const std::vector<std::int64_t> expected = {
        0, 0, 0, 2,   // cell (0,0): low-x and low-y
        1, 0, 1, 3,   // cell (0,1): low-x and high-y
        2, 2,         // cell (1,0): low-y
        3, 3,         // cell (1,1): high-y
        4, 1, 4, 2,   // cell (2,0): high-x and low-y
        5, 1, 5, 3};  // cell (2,1): high-x and high-y
    PANTR_CHECK(rows == expected);
}

void test_locate_many() {
    const Grid2 g(3, 2);
    const std::vector<double> pts = {0.5, 0.5, 2.5, 1.5, -1.0, 0.5, 0.5, 9.0};
    const std::vector<std::int64_t> got =
        g.locate_many(span2d<const double>(pts.data(), 4, 2));
    PANTR_CHECK(got == std::vector<std::int64_t>({0, 5, -1, -1}));
}

void test_collect_cell_bounds() {
    const Grid2 g(3, 2);
    std::vector<double> lo(12);
    std::vector<double> hi(12);
    g.collect_cell_bounds(span2d<double>(lo.data(), 6, 2), span2d<double>(hi.data(), 6, 2));
    const std::vector<double> expected_lo = {0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1};
    const std::vector<double> expected_hi = {1, 1, 1, 2, 2, 1, 2, 2, 3, 1, 3, 2};
    PANTR_CHECK(lo == expected_lo);
    PANTR_CHECK(hi == expected_hi);
}

/// AC3, restated: the cache builds on first use and is REUSED thereafter.
///
/// The ticket's third clause -- "and rebuilds after invalidation" -- has nothing to
/// assert here, because this ticket ships no `invalidate_caches()`. It was asked for so
/// that `HierarchicalGrid::_rebuild` could drop its base's caches, and
/// FELIGN/pantr#378 makes refinement return a NEW grid instead, so by the time the
/// hierarchical port runs there is nothing to invalidate: the new grid's caches are
/// empty and the old grid's are still valid. A protected mutator nothing calls is the
/// kind of seam that becomes load-bearing by accident.
void test_lazy_bvh_cache() {
    const Grid2 g(3, 2);
    PANTR_CHECK_MSG(g.cell_bounds_calls() == 0, "the cache must not be built at construction");

    const pantr::grid::BVH<double>& first = g.cell_bvh();
    const std::int64_t after_build = g.cell_bounds_calls();
    PANTR_CHECK_MSG(after_build == 6, "building the cache visits every cell exactly once");
    PANTR_CHECK(first.n_cells() == 6);

    const pantr::grid::BVH<double>& second = g.cell_bvh();
    PANTR_CHECK_MSG(&first == &second, "cell_bvh must hand back the same tree");
    PANTR_CHECK_MSG(g.cell_bounds_calls() == after_build, "a reused cache rebuilds nothing");

    // query_aabb rides on the same cache.
    std::vector<std::int64_t> hits = g.query_aabb(AABB<double>(std::vector<double>{1.5, 0.5},
                                                               std::vector<double>{1.6, 0.6}));
    PANTR_CHECK(hits == std::vector<std::int64_t>({2}));
    PANTR_CHECK(g.cell_bounds_calls() == after_build);
}

void test_tags() {
    Grid2 g(3, 2);
    PANTR_CHECK(g.cell_tags().num_cells() == 6);
    PANTR_CHECK(g.facet_tags().num_cells() == 6);
    PANTR_CHECK_MSG(g.facet_tags().facets_per_cell() == 4,
                    "facet_tags is sized 2 * ndim facets per cell");

    const std::vector<std::int64_t> ids = {0, 2};
    const std::vector<std::int64_t> values = {7, 9};
    g.cell_tags().set("a", ids, values);

    // The const overload reads the very same registry the non-const one mutated.
    const Grid2& cg = g;
    PANTR_CHECK(cg.cell_tags().contains("a"));
    PANTR_CHECK(&cg.cell_tags() == &g.cell_tags());
}

void test_validation_messages() {
    const Grid2 g(3, 2);
    std::string cid_message;
    try {
        static_cast<void>(g.cell_level(6));
    } catch (const std::out_of_range& e) {
        cid_message = e.what();
    }
    PANTR_CHECK_MSG(cid_message == "cell id 6 is out of range [0, 6).", cid_message);

    std::string lfid_message;
    try {
        static_cast<void>(g.local_facet_axis_side(0, 4));
    } catch (const std::out_of_range& e) {
        lfid_message = e.what();
    }
    PANTR_CHECK_MSG(lfid_message == "local facet id 4 is out of range [0, 4).", lfid_message);

    bool negative_threw = false;
    try {
        static_cast<void>(g.num_local_facets(-1));
    } catch (const std::out_of_range&) {
        negative_threw = true;
    }
    PANTR_CHECK(negative_threw);
}

void test_restrict_default_throws() {
    const Grid2 g(3, 2);
    const std::vector<std::int64_t> ids = {0, 1};
    bool threw = false;
    try {
        static_cast<void>(g.restrict(ids));
    } catch (const std::logic_error& e) {
        threw = std::string(e.what()).find("restrict()") != std::string::npos;
    }
    PANTR_CHECK_MSG(threw, "the default restrict must throw, as the Python contract does");
}

void test_make_restriction() {
    Grid2 sub(1, 2);
    const GridRestriction<Grid2> r =
        pantr::grid::make_restriction(sub, std::vector<std::int64_t>{4, 5},
                                      std::vector<std::uint8_t>{1, 0});
    PANTR_CHECK(r.grid.num_cells() == 2);
    PANTR_CHECK(r.local_to_global_cell == std::vector<std::int64_t>({4, 5}));
    PANTR_CHECK(r.in_subset == std::vector<std::uint8_t>({1, 0}));

    bool threw = false;
    try {
        static_cast<void>(pantr::grid::make_restriction(Grid2(1, 2),
                                                        std::vector<std::int64_t>{4},
                                                        std::vector<std::uint8_t>{1, 0}));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    PANTR_CHECK_MSG(threw, "make_restriction must reject index arrays of the wrong length");
}

// ---------------------------------------------------------------------------
// Dispatch: name hiding reaches the hook, the qualified call reaches the default
// ---------------------------------------------------------------------------

void test_name_hiding_dispatch() {
    const HookedGrid<double> g(3, 2);
    const BoxGrid<double> plain(3, 2);

    PANTR_CHECK_MSG(g.boundary_facets() == std::vector<std::int64_t>({-1, -1}),
                    "an unqualified call must reach the hook");
    PANTR_CHECK_MSG(g.Base::boundary_facets() == plain.boundary_facets(),
                    "the qualified call must reach the hidden default");

    const std::vector<double> pts = {0.5, 0.5, 2.5, 1.5};
    const span2d<const double> view(pts.data(), 2, 2);
    PANTR_CHECK(g.locate_many(view) == std::vector<std::int64_t>({-7, -7}));
    PANTR_CHECK(g.Base::locate_many(view) == plain.locate_many(view));

    // A default that calls another default through `self()` sees the hook too:
    // `neighbors` is generic here, and it reaches the grid's own primitives.
    PANTR_CHECK(g.neighbors(0) == plain.neighbors(0));
}

/// The `float` census device, exercised once at run time so it is not merely compiled.
void test_float_instantiation() {
    const BoxGrid<float> g(2, 2);
    std::vector<float> lo(2);
    std::vector<float> hi(2);
    g.cell_bounds(3, lo, hi);
    PANTR_CHECK(lo[0] == 1.0F && hi[1] == 2.0F);
    PANTR_CHECK(g.boundary_facets().size() == 16);
}

}  // namespace

int main() {
    test_state();
    test_cell_accessors();
    test_neighbors();
    test_facets();
    test_boundary_facets();
    test_locate_many();
    test_collect_cell_bounds();
    test_lazy_bvh_cache();
    test_tags();
    test_validation_messages();
    test_restrict_default_throws();
    test_make_restriction();
    test_name_hiding_dispatch();
    test_float_instantiation();
    return pantr::test::summary("test_grid_defaults");
}
