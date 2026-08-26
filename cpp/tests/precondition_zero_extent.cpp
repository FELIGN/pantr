/// \file
/// A block with a zero extent, which is arithmetic UB rather than a bad access.
///
/// The class the other grid test does not cover: `decode_flat_id` expands a C-order
/// offset with `%` and `/` per axis, and a block whose `hi` equals its `lo` makes both
/// a division by zero, which traps in hardware. The oracle is `@nb_jit` compiled, where
/// int64 division by zero raises `ZeroDivisionError`.
///
/// Worth its own executable rather than folding into `precondition_decode_flat_id`,
/// because that one violates a different precondition in the same function and the
/// first assertion to fire would hide this one.

#include <cstdint>
#include <span>
#include <vector>

#include "expect_precondition.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/grid/hierarchical.hpp"

int main() {
    pantr::test::expect_a_precondition_to_fire();
    std::vector<std::int64_t> blo{0}, bhi{0}, base{0}, lbs{0, 0, 1}, midx(1);
    return static_cast<int>(pantr::grid::decode_flat_id(
        0, pantr::span2d<const std::int64_t>(blo.data(), 1, 1),
        pantr::span2d<const std::int64_t>(bhi.data(), 1, 1), std::span<const std::int64_t>(base),
        std::span<const std::int64_t>(lbs), std::span<std::int64_t>(midx)));
}
