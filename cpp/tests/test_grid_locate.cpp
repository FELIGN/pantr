/// \file
/// Point location on a tensor-product grid, checked against properties the
/// mathematics fixes rather than against the oracle.
///
/// Nothing here compares the two backends: that is what `tests/parity/` is for,
/// and a transliteration faithful to a wrong algorithm passes a parity test
/// perfectly. What this file checks is that the kernel says what the grid means.
///
/// The tie contract is the interesting part and it gets most of the cases. A point
/// exactly on an interior breakpoint belongs to the LOWER cell sharing that face;
/// a point on either outer boundary belongs to the adjacent boundary cell; a point
/// outside maps to -1. Those are exact claims about integers, so they carry no
/// tolerance -- and they are decided by comparisons on coordinates that were never
/// arithmetic operands, which is why the port can claim them at all.

#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include "check.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/grid/locate.hpp"

namespace {

using pantr::span2d;
using pantr::grid::locate_points;

/// Locate points on a 1-D grid with the given breakpoints.
std::vector<std::int64_t> locate_1d(const std::vector<double>& knots,
                                    const std::vector<double>& xs) {
    const auto ncells = static_cast<std::int64_t>(knots.size()) - 1;
    std::vector<std::int64_t> starts{0};
    std::vector<std::int64_t> cells{ncells};
    std::vector<std::int64_t> strides{1};
    std::vector<std::int64_t> out(xs.size(), -99);

    locate_points<double>(span2d<const double>(xs.data(), xs.size(), 1),
                          std::span<const double>(knots), std::span<const std::int64_t>(starts),
                          std::span<const std::int64_t>(cells),
                          std::span<const std::int64_t>(strides), std::span<std::int64_t>(out));
    return out;
}

/// A point strictly inside cell k lands in cell k.
void interior_points_land_in_their_own_cell() {
    const std::vector<double> knots{0.0, 1.0, 2.0, 3.0};
    const auto got = locate_1d(knots, {0.5, 1.5, 2.5});
    PANTR_CHECK(got[0] == 0);
    PANTR_CHECK(got[1] == 1);
    PANTR_CHECK(got[2] == 2);
}

/// An interior breakpoint belongs to the lower of the two cells sharing it.
void interior_breakpoints_go_to_the_lower_cell() {
    const std::vector<double> knots{0.0, 1.0, 2.0, 3.0};
    const auto got = locate_1d(knots, {1.0, 2.0});
    PANTR_CHECK_MSG(got[0] == 0, "x = 1.0 must take cell 0, not cell 1");
    PANTR_CHECK_MSG(got[1] == 1, "x = 2.0 must take cell 1, not cell 2");
}

/// Either outer boundary belongs to the adjacent boundary cell, not to -1.
void outer_boundaries_are_inside() {
    const std::vector<double> knots{0.0, 1.0, 2.0, 3.0};
    const auto got = locate_1d(knots, {0.0, 3.0});
    PANTR_CHECK_MSG(got[0] == 0, "the lower boundary belongs to the first cell");
    PANTR_CHECK_MSG(got[1] == 2, "the upper boundary belongs to the last cell");
}

/// Anything outside the closed domain is -1.
void outside_maps_to_minus_one() {
    const std::vector<double> knots{0.0, 1.0, 2.0, 3.0};
    const auto got = locate_1d(knots, {-1e-300, 3.0 + 1e-15, -5.0, 100.0});
    for (std::size_t i = 0; i < 4; ++i) {
        PANTR_CHECK(got[i] == -1);
    }
}

/// One ulp inside the boundary is inside; one ulp outside is not.
///
/// The frontier of the domain test, which is where an off-by-one in the
/// comparison would hide. Uses nextafter rather than a comfortable epsilon so the
/// case cannot pass by being far from the edge.
void the_domain_frontier_is_exact() {
    const std::vector<double> knots{0.0, 1.0};
    const double below = std::nextafter(0.0, -1.0);
    const double above = std::nextafter(1.0, 2.0);
    const auto got = locate_1d(knots, {below, 0.0, 1.0, above});
    PANTR_CHECK_MSG(got[0] == -1, "one ulp below the domain is outside");
    PANTR_CHECK_MSG(got[1] == 0, "the lower corner is inside");
    PANTR_CHECK_MSG(got[2] == 0, "the upper corner is inside, in the last cell");
    PANTR_CHECK_MSG(got[3] == -1, "one ulp above the domain is outside");
}

/// A degenerate one-cell grid still resolves both corners and the interior.
void a_single_cell_grid_works() {
    const auto got = locate_1d({-1.3, 2.7}, {-1.3, 0.0, 2.7});
    PANTR_CHECK(got[0] == 0);
    PANTR_CHECK(got[1] == 0);
    PANTR_CHECK(got[2] == 0);
}

/// Non-uniform breakpoints: the search must not assume a uniform spacing.
void non_uniform_breakpoints_resolve() {
    const std::vector<double> knots{0.0, 0.001, 0.002, 100.0};
    const auto got = locate_1d(knots, {0.0005, 0.0015, 50.0, 0.001, 0.002});
    PANTR_CHECK(got[0] == 0);
    PANTR_CHECK(got[1] == 1);
    PANTR_CHECK(got[2] == 2);
    PANTR_CHECK_MSG(got[3] == 0, "a breakpoint of a tiny cell still takes the lower one");
    PANTR_CHECK_MSG(got[4] == 1, "and so does the next");
}

/// The 2-D flat id is the row-major combination of the per-axis indices.
///
/// Checked against the definition rather than against a table, so the test says
/// what C-order means instead of restating the kernel's own multiply.
void two_dimensional_ids_are_row_major() {
    // axis 0: [0,1,2,3] -> 3 cells;  axis 1: [0,1,2] -> 2 cells.  strides (2, 1).
    const std::vector<double> knots{0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0};
    const std::vector<std::int64_t> starts{0, 4};
    const std::vector<std::int64_t> cells{3, 2};
    const std::vector<std::int64_t> strides{2, 1};
    const std::vector<double> pts{0.5, 0.5, 2.5, 1.5, 1.0, 1.0, 3.0, 2.0};
    std::vector<std::int64_t> out(4, -99);

    locate_points<double>(span2d<const double>(pts.data(), 4, 2), std::span<const double>(knots),
                          std::span<const std::int64_t>(starts),
                          std::span<const std::int64_t>(cells),
                          std::span<const std::int64_t>(strides), std::span<std::int64_t>(out));

    PANTR_CHECK(out[0] == 0 * 2 + 0);
    PANTR_CHECK(out[1] == 2 * 2 + 1);
    PANTR_CHECK_MSG(out[2] == 0 * 2 + 0, "a breakpoint on both axes takes the lower cell on both");
    PANTR_CHECK_MSG(out[3] == 2 * 2 + 1, "the far corner is the last cell, not -1");
}

/// A point outside on ONE axis is outside, whatever the other axes say.
void outside_on_any_axis_is_outside() {
    const std::vector<double> knots{0.0, 1.0, 0.0, 1.0};
    const std::vector<std::int64_t> starts{0, 2};
    const std::vector<std::int64_t> cells{1, 1};
    const std::vector<std::int64_t> strides{1, 1};
    const std::vector<double> pts{0.5, 2.0, 2.0, 0.5, 0.5, 0.5};
    std::vector<std::int64_t> out(3, -99);

    locate_points<double>(span2d<const double>(pts.data(), 3, 2), std::span<const double>(knots),
                          std::span<const std::int64_t>(starts),
                          std::span<const std::int64_t>(cells),
                          std::span<const std::int64_t>(strides), std::span<std::int64_t>(out));

    PANTR_CHECK_MSG(out[0] == -1, "outside on axis 1 only");
    PANTR_CHECK_MSG(out[1] == -1, "outside on axis 0 only");
    PANTR_CHECK(out[2] == 0);
}

}  // namespace

int main() {
    interior_points_land_in_their_own_cell();
    interior_breakpoints_go_to_the_lower_cell();
    outer_boundaries_are_inside();
    outside_maps_to_minus_one();
    the_domain_frontier_is_exact();
    a_single_cell_grid_works();
    non_uniform_breakpoints_resolve();
    two_dimensional_ids_are_row_major();
    outside_on_any_axis_is_outside();
    return pantr::test::summary("test_grid_locate");
}
