/// \file
/// The same class outside `grid`, which is what made #359 an architectural question.
///
/// `tabulate_bernstein_1d` takes `int degree` and guarded only the zero case. At
/// `degree = -1` the cast to `std::size_t` sized the loops from a wrapped value and the
/// kernel wrote past `out`: measured as a heap-buffer-overflow WRITE at
/// `bernstein.hpp:149`, against an oracle that returns `[2, 0, 0, 0]` and corrupts
/// nothing. `cpp/bindings/basis.cpp` takes `unsigned degree` deliberately, so no Python
/// caller reaches it; the header was reachable directly.
///
/// One of the five modules is enough here. The point this pins is that the contract is
/// the port's and not `grid`'s, and the other four carry the same macro at the sites
/// the ticket's survey found.

#include <span>
#include <vector>

#include "pantr/basis/bernstein.hpp"
#include "expect_precondition.hpp"
#include "pantr/core/mdspan.hpp"

int main() {
    pantr::test::expect_a_precondition_to_fire();
    std::vector<double> points{0.5};
    std::vector<double> out(4, 0.0);
    pantr::tabulate_bernstein_1d<double>(-1, std::span<const double>(points),
                                         pantr::span2d<double>(out.data(), 4, 1));
    return 0;
}
