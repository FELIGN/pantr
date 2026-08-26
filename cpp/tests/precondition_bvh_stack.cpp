/// \file
/// FELIGN/pantr#359, site 2: a BVH deeper than the traversal stack.
///
/// Both query kernels push two children per interior node onto a fixed
/// `std::array<std::int64_t, kBvhStackDepth>`. Before the precondition this was a
/// stack-buffer-overflow WRITE at `bvh.hpp:290`, which is memory corruption rather
/// than a wrong number.
///
/// The RIGHT spine is what overflows, and that is the detail worth keeping: the
/// traversal pops the right child first, so a left-leaning chain is consumed without
/// the stack ever growing. The obvious harness builds the left-leaning tree, returns a
/// clean answer, and proves nothing. That cost two attempts when the bug was found.
///
/// `pantr.grid.BVH.__init__` walks the tree once at construction and refuses a deeper
/// one, so no sanctioned caller reaches this; a C++ caller including the header does.

#include <cstdint>
#include <span>
#include <vector>

#include "expect_precondition.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/grid/bvh.hpp"

int main() {
    pantr::test::expect_a_precondition_to_fire();
    const int depth = 200;  // > kBvhStackDepth
    const int n = 2 * depth + 1;
    std::vector<double> nlo(static_cast<std::size_t>(n), 0.0), nhi(static_cast<std::size_t>(n), 1.0);
    std::vector<std::int64_t> left(static_cast<std::size_t>(n), -1),
        right(static_cast<std::size_t>(n), -1), cell(static_cast<std::size_t>(n), -1);
    for (int i = 0; i < depth; ++i) {
        right[static_cast<std::size_t>(i)] = i + 1;
        left[static_cast<std::size_t>(i)] = depth + 1 + i;
    }
    for (int j = depth; j < n; ++j) {
        cell[static_cast<std::size_t>(j)] = j - depth;
    }
    std::vector<double> qlo{0.0}, qhi{1.0};
    return static_cast<int>(pantr::grid::bvh_query_count<double>(
        std::span<const double>(qlo), std::span<const double>(qhi),
        pantr::span2d<const double>(nlo.data(), static_cast<std::size_t>(n), 1),
        pantr::span2d<const double>(nhi.data(), static_cast<std::size_t>(n), 1),
        std::span<const std::int64_t>(left), std::span<const std::int64_t>(right),
        std::span<const std::int64_t>(cell)));
}
