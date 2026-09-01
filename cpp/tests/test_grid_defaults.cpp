/// \file
/// The generic grid layer's defaults, checked against hand-computed values on a
/// synthetic grid that declares no specialisations at all.
///
/// Two fixtures, and the split is the point:
///
///   * `BoxGrid<T>` declares `Hook::none`, so every one of the fourteen replaceable
///     defaults RUNS here, and the census asserts the negative.
///   * `HookedGrid<T>` declares two hooks, and exists to pin the two halves of the
///     dispatch mechanism the next ticket depends on: name hiding reaches the hook, and
///     the qualified call `g.Base::name()` still reaches the hidden default. That
///     qualified call is the C++ analogue of `Grid.boundary_facets(g)` in
///     `tests/test_grid_hierarchical.py`, and it is what a differential test needs.
///   * A third section spends that mechanism on the first real specialisation:
///     `TensorProductGrid`'s three hooks against the three defaults they hide, on the
///     same grid and the same input. It lives here rather than in
///     `test_grid_tensor_product.cpp` because what it exercises is the dispatch, and
///     because the code it compares against is the mixin's rather than the grid's.
///
/// Both are censused at `float` AND at `double`. The `float` instantiation is a
/// COMPILE-TIME census device and nothing else: it forces every default body at a
/// scalar no binding registers, which is what catches a hook that hard-codes `double`.
/// It is never bound and never compared, so it does not open a parity claim --
/// `design/backend_parity.md` Rule 8 forbids one where there is no oracle behind it,
/// and `src/pantr/grid/_grid_backend.py` records that `pantr.grid`'s oracle is
/// `float64`-only.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "check.hpp"
#include "pantr/grid/grid.hpp"
#include "pantr/grid/tensor_product_grid.hpp"

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

// Pins that the fixture above differs from a good grid in its PRIMITIVE and in nothing
// else -- and, incidentally, keeps clang from reporting `hooks` unused, which it does
// for the traits of any grid in an anonymous namespace that the census never reads.
static_assert(pantr::grid::grid_traits<UnwritableBoundsGrid>::hooks == Hook::none);

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

/// Whether `fn` throws `std::invalid_argument`.
template <class F>
bool throws_invalid_argument(F fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

/// Whether `fn` throws `std::out_of_range`.
template <class F>
bool throws_out_of_range(F fn) {
    try {
        fn();
    } catch (const std::out_of_range&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
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

/// Threads arriving together get one tree, built once.
///
/// The cache used to be a bare `mutable std::optional` and filling it concurrently was
/// undefined behaviour; it is now a `LazySlot`, whose own file carries the mechanism.
/// Two things about what this establishes, because they are not the same thing.
///
/// The **value** half is real and is asserted here: `cell_bounds_calls()` counts the
/// build, so `== 6` after the storm says it ran once and not once per thread. That
/// counter is a plain integer on purpose -- it is written only inside the build, so
/// counting with it *is* the assertion that the build was serialised.
///
/// The **race** half is settled by no value at all, and this file does not pretend
/// otherwise. `cpp/tests/test_lazy.cpp` carries that argument and a sanitizer build is
/// what runs it -- including the part that matters most, that its threads read the
/// memoised value rather than only taking its address. What is checked here is that the
/// grid reaches the slot through a path that has the guarantee, which is the half a
/// memo used wrongly would fail; this case deliberately does not restate the race
/// argument, so it takes an address and is not evidence about ordering.
void test_concurrent_bvh_build() {
    constexpr int kThreads = 8;
    const Grid2 g(3, 2);

    std::atomic<int> waiting{0};
    std::vector<const pantr::grid::BVH<double>*> seen(kThreads, nullptr);
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            waiting.fetch_add(1, std::memory_order_acq_rel);
            while (waiting.load(std::memory_order_acquire) < kThreads) {
                std::this_thread::yield();
            }
            seen[static_cast<std::size_t>(t)] = &g.cell_bvh();
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }

    PANTR_CHECK_MSG(g.cell_bounds_calls() == 6, "the tree must be built exactly once");
    for (int t = 0; t < kThreads; ++t) {
        PANTR_CHECK(seen[static_cast<std::size_t>(t)] == seen[0]);
    }
    PANTR_CHECK(seen[0]->n_cells() == 6);
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

/// The out-parameter shape checks, which must survive a release build.
///
/// They are ordinary `if`/`throw`, not `PANTR_PRECONDITION`: that macro is `assert` and
/// vanishes under `NDEBUG`, so a release build would index a short span out of bounds
/// instead of reporting it. This test runs in every configuration, which is the point.
void test_output_span_validation() {
    const Grid2 g(3, 2);
    std::vector<double> two(2);
    std::vector<double> one(1);

    PANTR_CHECK(throws_invalid_argument([&] { g.local_facet_bounds(0, 0, one, two); }));
    PANTR_CHECK(throws_invalid_argument([&] { g.local_facet_bounds(0, 0, two, one); }));

    std::vector<double> small(4);
    const span2d<double> short_view(small.data(), 2, 2);
    PANTR_CHECK(
        throws_invalid_argument([&] { g.collect_cell_bounds(short_view, short_view); }));

    const std::vector<double> pts = {0.5, 0.5, 1.5};
    PANTR_CHECK(throws_invalid_argument(
        [&] { static_cast<void>(g.locate_many(span2d<const double>(pts.data(), 1, 3))); }));
}

/// A one-dimensional grid: the facet count, the encoding and the tag registry follow it.
///
/// Every default is written against `2 * ndim`, and `ndim == 1` is the smallest value
/// that exercises the arithmetic without the two axes masking a mixed-up index.
void test_one_dimensional_grid() {
    const RawGrid g(1, 3);
    PANTR_CHECK(g.ndim() == 1);
    PANTR_CHECK(g.num_local_facets(0) == 2);
    PANTR_CHECK(g.facet_tags().facets_per_cell() == 2);
    PANTR_CHECK(g.local_facet_axis_side(0, 0) == AxisSide(0, 0));
    PANTR_CHECK(g.local_facet_axis_side(0, 1) == AxisSide(0, 1));
    PANTR_CHECK(throws_out_of_range([&] { static_cast<void>(g.local_facet_axis_side(0, 2)); }));
    // RawGrid has no neighbours at all, so every one of the six facets is a boundary.
    PANTR_CHECK(g.boundary_facets().size() == 12);
}

/// A single-cell grid: no neighbours, every facet on the boundary.
void test_single_cell_grid() {
    const Grid2 g(1, 1);
    PANTR_CHECK(g.num_cells() == 1);
    PANTR_CHECK(g.neighbors(0).empty());
    PANTR_CHECK(g.boundary_facets() == std::vector<std::int64_t>({0, 0, 0, 1, 0, 2, 0, 3}));
    PANTR_CHECK(g.cell_bvh().n_cells() == 1);
    PANTR_CHECK(g.hanging_neighbors(0, 0).empty());
}

/// A point exactly on an interior cell face lands in one cell, and always the same one.
///
/// The tie cannot be avoided -- a face belongs to both cells geometrically -- so what is
/// asserted is that the fixture's rule (a face belongs to the cell above it) is applied
/// consistently by `locate` and by the generic `locate_many` built on it.
void test_locate_on_a_cell_boundary() {
    const Grid2 g(3, 2);
    const std::vector<double> pts = {1.0, 0.5, 0.0, 0.0, 3.0, 2.0};
    const std::vector<std::int64_t> got = g.locate_many(span2d<const double>(pts.data(), 3, 2));
    // (1.0, 0.5) is on the face between cells 0 and 2, and goes to the upper one.
    // (0.0, 0.0) is the grid's lo corner; (3.0, 2.0) is its hi corner, and the top of the
    // range is closed, so it lands in the last cell rather than outside.
    PANTR_CHECK(got == std::vector<std::int64_t>({2, 0, 5}));
    for (std::size_t i = 0; i < 3; ++i) {
        const std::span<const double> pt(&pts[i * 2], 2);
        PANTR_CHECK(g.locate(pt).value_or(-1) == got[i]);
    }
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

    // `neighbors` is generic on BOTH grids, so this says only that hiding two unrelated
    // names left it alone. That is the claim worth making here -- name hiding is
    // per-name, and a hook must not perturb a default that does not mention it.
    PANTR_CHECK(g.neighbors(0) == plain.neighbors(0));
}

// ---------------------------------------------------------------------------
// The first real specialisation against the defaults it hides (FELIGN/pantr#387)
// ---------------------------------------------------------------------------

/// `TensorProductGrid`'s three hooks against the three generic defaults they replace.
///
/// The qualified call `g.Base::name(...)` reaches the hidden default, so both sides run
/// on ONE grid and one input; nothing here builds a second object that could differ for
/// a reason other than the specialisation.
///
/// **What agreement is claimed, per hook.**
///
///  - `locate_many` returns cell ids, which are `std::int64_t`. There is no bound to
///    state: a cell id is a VERDICT, and `design/backend_parity.md` Rule 11 is explicit
///    that no tolerance bounds one. Exact equality is the only claim available and it
///    is the right one. The two paths reach the same per-axis search, so what this
///    compares is the composition around it -- the strides, the output order and the
///    `-1` convention -- which is where an error would be.
///  - `collect_cell_bounds` writes coordinates, so a bound is meaningful, and the
///    derived bound is EXACTLY ZERO. Every value either side writes is a copy of a
///    stored breakpoint: the default reads `bp[i]` through `cell_bounds`, the hook
///    reads the same `bp[i]` through the flat buffer, and neither performs an
///    arithmetic operation on it. With no operation there is no rounding, so the
///    difference is not merely small but bit-identical, and the comparison is `==`.
///    A tolerance here would be looser than the derivation supports and would hide a
///    transposed stride, which is the defect this is aimed at.
///  - `restrict` has NO comparison to make, and saying so is more honest than
///    inventing one. The default does not compute a value; it throws. Rule 8 -- a
///    parity claim is only defined where the quantity has digits -- applies directly:
///    what is asserted instead is that the default is still reachable and still
///    throws, and that the hook does not. That is the whole content available.
void test_tensor_product_hooks_against_defaults() {
    using TPG = pantr::grid::TensorProductGrid<double>;
    // Deliberately NON-uniform on both axes and unequal in length, so a transposed
    // stride or a swapped axis produces a different answer rather than the same one.
    const TPG g({{0.0, 1.0, 3.0, 6.0, 10.0}, {-2.0, 0.5, 4.0}});

    // locate_many: interior, both breakpoint faces, both outer boundaries, outside on
    // each axis, and a non-finite coordinate.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> pts = {0.5, 0.0, 3.0,  0.5, 10.0, 4.0, 0.0, -2.0,
                                     -0.1, 0.0, 5.0, 9.0, nan,  0.0};
    const span2d<const double> view(pts.data(), 7, 2);
    const std::vector<std::int64_t> hooked = g.locate_many(view);
    const std::vector<std::int64_t> generic = g.TPG::Base::locate_many(view);
    PANTR_CHECK_MSG(hooked == generic,
                    "locate_many must agree with the default exactly: a cell id is a "
                    "verdict, not a displaced value");
    // Non-vacuous: the batch must contain both an interior hit and an outside miss, or
    // an equality of two all-minus-one vectors would pass while proving nothing.
    PANTR_CHECK(hooked.front() >= 0 && hooked.back() == -1);

    // collect_cell_bounds: bit-identical, per the derivation above.
    const auto rows = static_cast<std::size_t>(g.num_cells());
    const auto cols = static_cast<std::size_t>(g.ndim());
    std::vector<double> hook_lo(rows * cols);
    std::vector<double> hook_hi(rows * cols);
    std::vector<double> base_lo(rows * cols);
    std::vector<double> base_hi(rows * cols);
    g.collect_cell_bounds(span2d<double>(hook_lo.data(), rows, cols),
                          span2d<double>(hook_hi.data(), rows, cols));
    g.TPG::Base::collect_cell_bounds(span2d<double>(base_lo.data(), rows, cols),
                                     span2d<double>(base_hi.data(), rows, cols));
    PANTR_CHECK_MSG(hook_lo == base_lo && hook_hi == base_hi,
                    "collect_cell_bounds must agree with the default bit for bit: every "
                    "value either side writes is a copy of a stored breakpoint");
    // Non-vacuous: the arrays must hold the grid, not zeros left over from allocation.
    PANTR_CHECK(hook_lo.front() == 0.0 && hook_hi.back() == 4.0 && base_hi[1] == 0.5);

    // restrict: the hook returns, the default throws. No digits, no bound.
    const std::vector<std::int64_t> ids = {0, 7};
    PANTR_CHECK(g.restrict(ids).grid.num_cells() == 8);
    bool default_threw = false;
    try {
        (void)g.TPG::Base::restrict(ids);
    } catch (const std::logic_error&) {
        default_threw = true;
    }
    PANTR_CHECK_MSG(default_threw,
                    "the hidden default must still be reachable, and must still throw");
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
    test_concurrent_bvh_build();
    test_tags();
    test_validation_messages();
    test_restrict_default_throws();
    test_make_restriction();
    test_output_span_validation();
    test_one_dimensional_grid();
    test_single_cell_grid();
    test_locate_on_a_cell_boundary();
    test_name_hiding_dispatch();
    test_tensor_product_hooks_against_defaults();
    test_float_instantiation();
    return pantr::test::summary("test_grid_defaults");
}
