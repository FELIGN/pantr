/// \file
/// FELIGN/pantr#359, site 1: a flat id below the first block's base.
///
/// The upper-bound search leaves `lo_b == 0`, so the block index is `-1` cast to
/// `std::size_t`. Before the precondition this was a heap-buffer-overflow READ,
/// reported by AddressSanitizer at `hierarchical.hpp:213`. The Numba oracle indexes
/// backwards from the end instead and returns a wrong answer: `0`, with `midx == [5]`.
///
/// The process is expected to abort. ctest matches the assertion's own text, not
/// merely a non-zero exit, because a non-zero exit is also what the *unfixed* header
/// produces -- ASan exits non-zero too. Matching the message is what makes this test
/// fail against the code it was written for.

#include <cstdint>
#include <span>
#include <vector>

#include "expect_precondition.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/grid/hierarchical.hpp"

int main() {
    pantr::test::expect_a_precondition_to_fire();
    std::vector<std::int64_t> blo{0}, bhi{10}, base{5}, lbs{0, 0, 1}, midx(1);
    return static_cast<int>(pantr::grid::decode_flat_id(
        0,  // below base[0] = 5, which the header names as a precondition
        pantr::span2d<const std::int64_t>(blo.data(), 1, 1),
        pantr::span2d<const std::int64_t>(bhi.data(), 1, 1), std::span<const std::int64_t>(base),
        std::span<const std::int64_t>(lbs), std::span<std::int64_t>(midx)));
}
