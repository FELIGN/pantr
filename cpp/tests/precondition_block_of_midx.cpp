/// \file
/// FELIGN/pantr#359's third site, which the first pass at that ticket left unmarked.
///
/// The ticket named three grid sites and the fix addressed two, substituting a newly
/// found one for the third without saying so. This is the one that was missed:
/// `block_of_midx` indexes `level_block_start` with a cast `level`, and a negative level
/// reads out of bounds. Measured at `hierarchical.hpp:162`.

#include <cstdint>
#include <span>
#include <vector>

#include "expect_precondition.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/grid/hierarchical.hpp"

int main() {
    pantr::test::expect_a_precondition_to_fire();
    std::vector<std::int64_t> midx{0}, blo{0}, bhi{10}, base{0}, lbs{0, 1};
    return static_cast<int>(pantr::grid::block_of_midx(
        -1, std::span<const std::int64_t>(midx),
        pantr::span2d<const std::int64_t>(blo.data(), 1, 1),
        pantr::span2d<const std::int64_t>(bhi.data(), 1, 1), std::span<const std::int64_t>(base),
        std::span<const std::int64_t>(lbs)));
}
