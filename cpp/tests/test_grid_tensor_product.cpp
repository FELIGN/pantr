/// \file
/// `pantr::grid::TensorProductGrid` and its `uniform_grid` factory.
///
/// Its behaviour on its own; the comparison of each of its three hooks against the
/// generic default it hides lives in `test_grid_defaults.cpp`, beside the dispatch
/// mechanism that comparison depends on.
///
/// The census is expanded here at BOTH scalars. The `float` instantiation is a
/// compile-time device: it forces every default body at a scalar no binding registers,
/// which is what catches a hook that hard-codes `double`. It opens no parity claim --
/// `design/backend_parity.md` Rule 8 -- and it is exercised once at run time below so
/// that it is not merely compiled.

#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/grid/tensor_product_grid.hpp"

namespace {

using pantr::span2d;
using pantr::grid::GridRestriction;
using pantr::grid::TensorProductGrid;

/// The grid type at the scalar the bindings register.
using Grid = TensorProductGrid<double>;

/// A 3-by-2 grid on `[0, 3] x [0, 2]` with unit cells, computable by hand.
///
/// \return The fixture grid.
[[nodiscard]] Grid unit_grid() {
    return Grid({{0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 2.0}});
}

/// Whether a call throws the expected exception carrying `needle` in its message.
///
/// \tparam E The exception type expected.
/// \tparam F The callable.
/// \param f The call to make.
/// \param needle A substring the message must contain.
/// \return `true` iff `f` threw an `E` whose `what()` contains `needle`.
template <class E, class F>
[[nodiscard]] bool throws_with(F&& f, const std::string& needle) {
    try {
        f();
    } catch (const E& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    } catch (...) {
        return false;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Construction and metadata
// ---------------------------------------------------------------------------

void test_sizes_and_layout() {
    const Grid g = unit_grid();
    PANTR_CHECK(g.ndim() == 2);
    PANTR_CHECK(g.num_cells() == 6);
    PANTR_CHECK(g.cells_per_axis()[0] == 3 && g.cells_per_axis()[1] == 2);
    // C order: the last axis varies fastest.
    PANTR_CHECK(g.strides()[0] == 2 && g.strides()[1] == 1);
    PANTR_CHECK(g.breakpoints(0).size() == 4 && g.breakpoints(1).size() == 3);
    PANTR_CHECK(g.breakpoints(1)[2] == 2.0);
    PANTR_CHECK(throws_with<std::out_of_range>([&g] { (void)g.breakpoints(2); }, "axis 2"));
}

void test_construction_is_validated() {
    PANTR_CHECK(throws_with<std::invalid_argument>([] { (void)Grid({}); }, "at least one axis"));
    PANTR_CHECK(
        throws_with<std::invalid_argument>([] { (void)Grid({{0.0}}); }, "at least 2 entries"));
    PANTR_CHECK(throws_with<std::invalid_argument>([] { (void)Grid({{0.0, 1.0, 1.0}}); },
                                                   "strictly increasing"));
    PANTR_CHECK(throws_with<std::invalid_argument>(
        [] { (void)Grid({{0.0, std::numeric_limits<double>::infinity()}}); }, "finite"));
    PANTR_CHECK(throws_with<std::invalid_argument>(
        [] { (void)Grid({{0.0, std::numeric_limits<double>::quiet_NaN(), 1.0}}); }, "finite"));
}

void test_index_helpers() {
    const Grid g = unit_grid();
    std::vector<std::int64_t> multi(2);
    g.cell_multi_index(5, multi);
    PANTR_CHECK(multi[0] == 2 && multi[1] == 1);
    PANTR_CHECK(g.flat_cell_index(multi) == 5);
    PANTR_CHECK(throws_with<std::out_of_range>([&g, &multi] { g.cell_multi_index(6, multi); },
                                               "cell id 6 is out of range [0, 6)."));
    const std::vector<std::int64_t> bad = {3, 0};
    PANTR_CHECK(throws_with<std::out_of_range>([&g, &bad] { (void)g.flat_cell_index(bad); },
                                               "cell index 3 on axis 0"));
}

void test_repr_matches_the_python_form() {
    PANTR_CHECK(unit_grid().to_string()
                == "TensorProductGrid(ndim=2, cells_per_axis=(3, 2), uniform=True)");
    // A one-tuple carries Python's trailing comma; a non-uniform axis flips the flag.
    PANTR_CHECK(Grid({{0.0, 1.0, 3.0}}).to_string()
                == "TensorProductGrid(ndim=1, cells_per_axis=(2,), uniform=False)");
}

// ---------------------------------------------------------------------------
// The primitives
// ---------------------------------------------------------------------------

void test_cell_bounds() {
    const Grid g = unit_grid();
    std::vector<double> lo(2);
    std::vector<double> hi(2);
    g.cell_bounds(5, lo, hi);
    PANTR_CHECK(lo[0] == 2.0 && lo[1] == 1.0 && hi[0] == 3.0 && hi[1] == 2.0);
    std::vector<double> short_span(1);
    PANTR_CHECK(throws_with<std::invalid_argument>(
        [&g, &short_span, &hi] { g.cell_bounds(0, short_span, hi); },
        "lo must have length ndim"));
}

void test_locate() {
    const Grid g = unit_grid();
    const std::vector<double> inside = {0.5, 0.5};
    PANTR_CHECK(g.locate(inside) == std::optional<std::int64_t>(0));
    // An interior breakpoint belongs to the LOWER cell sharing that face.
    const std::vector<double> face = {1.0, 1.0};
    PANTR_CHECK(g.locate(face) == std::optional<std::int64_t>(0));
    // The outer boundary belongs to the adjacent boundary cell, on both ends.
    const std::vector<double> upper = {3.0, 2.0};
    PANTR_CHECK(g.locate(upper) == std::optional<std::int64_t>(5));
    const std::vector<double> lower = {0.0, 0.0};
    PANTR_CHECK(g.locate(lower) == std::optional<std::int64_t>(0));
    const std::vector<double> outside = {3.5, 0.5};
    PANTR_CHECK(!g.locate(outside).has_value());
    // Every non-finite coordinate is outside every cell, with no second pass.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    PANTR_CHECK(!g.locate(std::vector<double>{nan, 0.5}).has_value());
    PANTR_CHECK(!g.locate(std::vector<double>{0.5, inf}).has_value());
    PANTR_CHECK(!g.locate(std::vector<double>{0.5, -inf}).has_value());
}

void test_neighbor_across_facet() {
    const Grid g = unit_grid();
    // cid 2 is (1, 0): low-x neighbour is (0, 0) = 0, high-x is (2, 0) = 4.
    PANTR_CHECK(g.neighbor_across_facet(2, 0) == std::optional<std::int64_t>(0));
    PANTR_CHECK(g.neighbor_across_facet(2, 1) == std::optional<std::int64_t>(4));
    PANTR_CHECK(!g.neighbor_across_facet(0, 0).has_value());
    PANTR_CHECK(!g.neighbor_across_facet(0, 2).has_value());
    PANTR_CHECK(g.neighbor_across_facet(0, 3) == std::optional<std::int64_t>(1));
    PANTR_CHECK(throws_with<std::out_of_range>([&g] { (void)g.neighbor_across_facet(0, 4); },
                                               "local facet id 4 is out of range [0, 4)."));
}

// ---------------------------------------------------------------------------
// The three hooks, on their own terms
// ---------------------------------------------------------------------------

void test_locate_many() {
    const Grid g = unit_grid();
    const std::vector<double> pts = {0.5, 0.5, 2.5, 1.5, 9.0, 9.0};
    const span2d<const double> view(pts.data(), 3, 2);
    PANTR_CHECK(g.locate_many(view) == std::vector<std::int64_t>({0, 5, -1}));
    const std::vector<double> wrong = {0.5, 0.5, 0.5};
    const span2d<const double> bad(wrong.data(), 1, 3);
    PANTR_CHECK(throws_with<std::invalid_argument>([&g, &bad] { (void)g.locate_many(bad); },
                                                   "points must have 2 columns"));
}

/// The flat layout is packed at construction, not per call -- FELIGN/pantr#387 AC9.
///
/// The address of the packed buffer is what makes that observable from outside: a
/// `locate_many` that rebuilt it would have to allocate, and the new storage could not
/// be at the address the constructor produced. The address is read through
/// `breakpoints(0)`, which is a view into the single flat buffer rather than a copy.
void test_flat_layout_is_packed_once() {
    const Grid g = unit_grid();
    const double* before = g.breakpoints(0).data();
    const std::vector<double> pts = {0.5, 0.5, 2.5, 1.5};
    const span2d<const double> view(pts.data(), 2, 2);
    (void)g.locate_many(view);
    (void)g.locate_many(view);
    PANTR_CHECK_MSG(g.breakpoints(0).data() == before,
                    "locate_many must not repack the breakpoints");
    // The two axes share one allocation, which is what "flat" means here.
    PANTR_CHECK(g.breakpoints(1).data() == before + 4);
}

void test_collect_cell_bounds() {
    const Grid g = unit_grid();
    std::vector<double> lo(12);
    std::vector<double> hi(12);
    g.collect_cell_bounds(span2d<double>(lo.data(), 6, 2), span2d<double>(hi.data(), 6, 2));
    // Cell 3 is (1, 1): [1, 2] x [1, 2].
    PANTR_CHECK(lo[6] == 1.0 && lo[7] == 1.0 && hi[6] == 2.0 && hi[7] == 2.0);
    PANTR_CHECK(lo[0] == 0.0 && hi[11] == 2.0);
    std::vector<double> small(4);
    PANTR_CHECK(throws_with<std::invalid_argument>(
        [&g, &small, &hi] {
            g.collect_cell_bounds(span2d<double>(small.data(), 2, 2),
                                  span2d<double>(hi.data(), 6, 2));
        },
        "cell_lo must have shape (6, 2)"));
}

void test_restrict() {
    const Grid g = unit_grid();
    // Cells 0 = (0,0) and 5 = (2,1): the window is the whole grid, and the two
    // requested cells are flagged while the four fill cells are not.
    const std::vector<std::int64_t> ids = {0, 5};
    const GridRestriction<Grid> r = g.restrict(ids);
    PANTR_CHECK(r.grid.num_cells() == 6);
    PANTR_CHECK(r.local_to_global_cell == std::vector<std::int64_t>({0, 1, 2, 3, 4, 5}));
    PANTR_CHECK(r.in_subset == std::vector<std::uint8_t>({1, 0, 0, 0, 0, 1}));

    // A pure slice of the breakpoints: the sub-grid's cells coincide with this grid's.
    const std::vector<std::int64_t> corner = {3};
    const GridRestriction<Grid> one = g.restrict(corner);
    PANTR_CHECK(one.grid.num_cells() == 1);
    PANTR_CHECK(one.grid.breakpoints(0)[0] == 1.0 && one.grid.breakpoints(0)[1] == 2.0);
    PANTR_CHECK(one.grid.breakpoints(1)[0] == 1.0 && one.grid.breakpoints(1)[1] == 2.0);
    PANTR_CHECK(one.local_to_global_cell == std::vector<std::int64_t>({3}));
    PANTR_CHECK(one.in_subset == std::vector<std::uint8_t>({1}));

    // Duplicates are ignored rather than rejected.
    const std::vector<std::int64_t> dupes = {3, 3, 3};
    PANTR_CHECK(g.restrict(dupes).grid.num_cells() == 1);

    PANTR_CHECK(throws_with<std::invalid_argument>([&g] { (void)g.restrict({}); }, "non-empty"));
    const std::vector<std::int64_t> off = {0, 6};
    PANTR_CHECK(throws_with<std::out_of_range>(
        [&g, &off] { (void)g.restrict(off); },
        "restrict: cell id out of range [0, 6); got [0, 6]."));
    const std::vector<std::int64_t> negative = {-1};
    PANTR_CHECK(throws_with<std::out_of_range>([&g, &negative] { (void)g.restrict(negative); },
                                               "got [-1, -1]."));
}

// ---------------------------------------------------------------------------
// Inherited defaults still work on this grid
// ---------------------------------------------------------------------------

void test_inherited_defaults() {
    const Grid g = unit_grid();
    PANTR_CHECK(g.cell_level(0) == 0);
    PANTR_CHECK(g.child_cells(0).empty());
    PANTR_CHECK(g.num_local_facets(0) == 4);
    PANTR_CHECK(g.cell_aabb(5).lo()[0] == 2.0 && g.cell_aabb(5).hi()[1] == 2.0);
    PANTR_CHECK(g.neighbors(0) == std::vector<std::int64_t>({2, 1}));
    // A 3-by-2 grid has 2 * (3 + 2) = 10 outer facets, emitted as (cid, lfid) pairs.
    PANTR_CHECK(g.boundary_facets().size() == 20);
    // The BVH is built from `collect_cell_bounds`, so it exercises the hook through a
    // default; the same reference comes back on a second call.
    PANTR_CHECK(&g.cell_bvh() == &g.cell_bvh());
    PANTR_CHECK(g.cell_bvh().n_cells() == 6);
}

void test_tags_are_sized_from_the_grid() {
    Grid g = unit_grid();
    PANTR_CHECK(g.cell_tags().num_cells() == 6);
    PANTR_CHECK(g.facet_tags().facets_per_cell() == 4);
    const std::vector<std::int64_t> ids = {0, 5};
    const std::vector<std::int64_t> values = {7, 7};
    g.cell_tags().set("cut", ids, values);
    PANTR_CHECK(g.cell_tags().contains("cut"));
}

// ---------------------------------------------------------------------------
// `uniform_grid`
// ---------------------------------------------------------------------------

void test_uniform_grid() {
    const std::vector<double> box = {0.0, 2.0, 0.0, 4.0};
    const std::vector<std::int64_t> cells = {2, 4};
    const Grid g = pantr::grid::uniform_grid(span2d<const double>(box.data(), 2, 2), cells);
    PANTR_CHECK(g.cells_per_axis()[0] == 2 && g.cells_per_axis()[1] == 4);
    PANTR_CHECK(g.breakpoints(0)[0] == 0.0 && g.breakpoints(0)[1] == 1.0);
    PANTR_CHECK(g.breakpoints(1)[4] == 4.0);
    PANTR_CHECK(g.is_uniform());

    const std::vector<std::int64_t> wrong_length = {2};
    PANTR_CHECK(throws_with<std::invalid_argument>(
        [&box, &wrong_length] {
            (void)pantr::grid::uniform_grid(span2d<const double>(box.data(), 2, 2),
                                            wrong_length);
        },
        "length-2 sequence"));
    const std::vector<std::int64_t> zero = {0, 1};
    PANTR_CHECK(throws_with<std::invalid_argument>(
        [&box, &zero] {
            (void)pantr::grid::uniform_grid(span2d<const double>(box.data(), 2, 2), zero);
        },
        "must be >= 1"));
    const std::vector<double> inverted = {2.0, 0.0};
    PANTR_CHECK(throws_with<std::invalid_argument>(
        [&inverted, &cells] {
            (void)pantr::grid::uniform_grid(span2d<const double>(inverted.data(), 1, 2),
                                            std::span<const std::int64_t>(cells.data(), 1));
        },
        "lo < hi"));
}

/// The final breakpoint is the requested `stop` EXACTLY, not the accumulated product.
///
/// numpy's `linspace` assigns it rather than computing it, and the grid's upper bound
/// is what `locate` compares a boundary point against: an upper breakpoint short by one
/// ulp would put `locate(stop)` outside the grid. The case is chosen so the computed
/// value genuinely differs -- the first check asserts that discrepancy exists before
/// the second asserts it was repaired.
void test_uniform_grid_pins_the_last_breakpoint() {
    const std::vector<double> box = {0.0, 2.9};
    const std::vector<std::int64_t> cells = {9};
    const Grid g = pantr::grid::uniform_grid(span2d<const double>(box.data(), 1, 2), cells);
    const double step = 2.9 / 9.0;
    const double computed_product = 9.0 * step;
    const double computed = computed_product + 0.0;
    // 2.8999999999999995 here: SHORT of `stop`, so without the assignment the query
    // below would fall outside the grid rather than in its last cell.
    PANTR_CHECK_MSG(computed < 2.9, "this case must be one where the product misses `stop`");
    PANTR_CHECK(g.breakpoints(0)[9] == 2.9);
    const std::vector<double> at_stop = {2.9};
    PANTR_CHECK(g.locate(at_stop) == std::optional<std::int64_t>(8));
}

/// `is_uniform` is scale-free: the same grid shape is uniform at every magnitude.
///
/// The absolute constant this replaced failed exactly here -- an exact `linspace` grid
/// on `[1e6, 1e6 + 1]` reported non-uniform, because its round-off is proportional to
/// the coordinate magnitude and the constant was not.
void test_is_uniform_is_scale_free() {
    const std::vector<std::int64_t> cells = {100};
    for (const double magnitude : {1e-12, 1.0, 1e6, 1e12}) {
        const std::vector<double> box = {magnitude, magnitude + (magnitude * 0.5)};
        const Grid g = pantr::grid::uniform_grid(span2d<const double>(box.data(), 1, 2), cells);
        PANTR_CHECK_MSG(g.is_uniform(), "a linspace axis is uniform at every magnitude");
    }
    // The offset case the absolute constant got wrong: a unit extent a million from
    // the origin.
    const std::vector<double> offset = {1e6, 1e6 + 1.0};
    const Grid g = pantr::grid::uniform_grid(span2d<const double>(offset.data(), 1, 2), cells);
    PANTR_CHECK(g.is_uniform());
    // And a genuinely non-uniform axis is still rejected, at the same magnitude.
    PANTR_CHECK(!Grid({{1e6, 1e6 + 0.25, 1e6 + 1.0}}).is_uniform());
}

// ---------------------------------------------------------------------------
// Degenerate shapes
// ---------------------------------------------------------------------------

void test_one_dimensional_and_single_cell() {
    const Grid line({{0.0, 1.0, 2.0}});
    PANTR_CHECK(line.ndim() == 1 && line.num_cells() == 2);
    PANTR_CHECK(line.strides()[0] == 1);
    PANTR_CHECK(line.locate(std::vector<double>{1.5}) == std::optional<std::int64_t>(1));
    PANTR_CHECK(line.boundary_facets() == std::vector<std::int64_t>({0, 0, 1, 1}));

    const Grid cell({{0.0, 1.0}, {0.0, 1.0}});
    PANTR_CHECK(cell.num_cells() == 1);
    PANTR_CHECK(cell.is_uniform());
    PANTR_CHECK(cell.neighbors(0).empty());
    std::vector<double> lo(2);
    std::vector<double> hi(2);
    cell.collect_cell_bounds(span2d<double>(lo.data(), 1, 2), span2d<double>(hi.data(), 1, 2));
    PANTR_CHECK(lo[0] == 0.0 && hi[1] == 1.0);
}

/// The `float` census device, exercised once at run time so it is not merely compiled.
void test_float_instantiation() {
    const TensorProductGrid<float> g({{0.0F, 1.0F, 2.0F}, {0.0F, 1.0F}});
    PANTR_CHECK(g.num_cells() == 2);
    std::vector<float> lo(2);
    std::vector<float> hi(2);
    g.cell_bounds(1, lo, hi);
    PANTR_CHECK(lo[0] == 1.0F && hi[0] == 2.0F);
    PANTR_CHECK(g.is_uniform());
    const std::vector<float> pts = {0.5F, 0.5F};
    PANTR_CHECK(g.locate_many(span2d<const float>(pts.data(), 1, 2))
                == std::vector<std::int64_t>({0}));
}

}  // namespace

PANTR_GRID_CENSUS(TensorProductGrid<double>);
PANTR_GRID_CENSUS(TensorProductGrid<float>);

int main() {
    test_sizes_and_layout();
    test_construction_is_validated();
    test_index_helpers();
    test_repr_matches_the_python_form();
    test_cell_bounds();
    test_locate();
    test_neighbor_across_facet();
    test_locate_many();
    test_flat_layout_is_packed_once();
    test_collect_cell_bounds();
    test_restrict();
    test_inherited_defaults();
    test_tags_are_sized_from_the_grid();
    test_uniform_grid();
    test_uniform_grid_pins_the_last_breakpoint();
    test_is_uniform_is_scale_free();
    test_one_dimensional_and_single_cell();
    test_float_instantiation();
    return pantr::test::summary("test_grid_tensor_product");
}
