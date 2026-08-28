/// \file
/// A negative derivative order, which the docstring named and nothing enforced.
///
/// `evaluate_bezier_deriv_1d` documents two memory-safety obligations, a non-empty
/// control net and a non-negative `n_deriv`. The first was asserted and the second was
/// not, which a review found by reading the sentence against the code. Measured as a
/// heap-buffer-overflow WRITE in the derivative table of `evaluate_bezier_deriv_1d`,
/// where `n_deriv` widens to `std::size_t` and a negative order becomes an enormous
/// one. Named by function rather than by line: the line moved with the header's
/// rename and would move again with the next paragraph added above it.
///
/// This also covers the `bezier` module's shape-inferred degree class, which had five
/// sites and no test: the two obligations sit in the same function.

#include <span>
#include <vector>

#include "expect_precondition.hpp"
#include "pantr/bezier/kernels_1d.hpp"
#include "pantr/core/mdspan.hpp"

int main() {
    pantr::test::expect_a_precondition_to_fire();
    std::vector<double> ctrl{0.0, 1.0}, points{0.5}, out(8, 0.0);
    pantr::bezier::evaluate_bezier_deriv_1d<double>(pantr::span2d<const double>(ctrl.data(), 2, 1),
                                                    std::span<const double>(points), -1,
                                                    pantr::span_nd<double, 3>(out.data(), 1, 1, 1));
    return 0;
}
