// A consumer of the INSTALLED pantr: no Python, no pantr source tree, only the
// package that `cmake --install` produced. Compiled and run by the `consumer`
// section of scripts/ci_local.sh against a throwaway prefix.
//
// It includes four headers on purpose. `bernstein.hpp` and `mdspan.hpp` cover
// the ordinary path; `change_basis.hpp` is the one header that pulls in Eigen,
// so leaving it out would let the package ship without a usable Eigen and still
// pass here.
//
// `bezier/bezier.hpp` is the fourth, and it is here rather than in the tests
// because it is a compatibility forwarder for an installed consumer: the kernels
// moved to `bezier/kernels_1d.hpp` and this path stays behind. In-tree code
// includes the real header, so this build is the only thing that would notice
// the forwarder being deleted -- which is the point of including it from the one
// program that compiles against the INSTALLED package.

#include <pantr/basis/bernstein.hpp>
#include <pantr/bezier/bezier.hpp>
#include <pantr/change_basis/change_basis.hpp>
#include <pantr/core/mdspan.hpp>

#include <cmath>
#include <cstdio>
#include <span>
#include <vector>

int main() {
    const int degree = 3;
    const std::vector<double> pts{0.0, 0.25, 0.5, 1.0};
    std::vector<double> tab(pts.size() * static_cast<std::size_t>(degree + 1), -99.0);
    pantr::span2d<double> view(tab.data(), pts.size(), static_cast<std::size_t>(degree + 1));

    pantr::tabulate_bernstein_1d<double>(degree, std::span<const double>(pts), view);

    // Partition of unity: the cheapest property that is actually true of the
    // result, so the check fails if the package links but computes nothing.
    // Degree 3 is four positive terms summing to one, so a few ulp is the whole
    // error budget; 8 eps is slack, not a derivation, and does not need to be
    // one for a smoke test whose job is "did this run at all".
    constexpr double kEps = 2.220446049250313e-16;
    for (std::size_t i = 0; i < pts.size(); ++i) {
        double s = 0.0;
        for (int j = 0; j <= degree; ++j) {
            s += pantr::at(view, i, static_cast<std::size_t>(j));
        }
        if (std::fabs(s - 1.0) > 8 * kEps) {
            std::printf("FAIL: partition of unity at point %zu is %.17g\n", i, s);
            return 1;
        }
    }

    // Through the forwarder, and under the new namespace: the old PATH still
    // resolves, the old spelling of the names does not. A constant curve is the
    // cheapest call with a knowable answer, and it is the partition of unity
    // again -- the same identity, reached through the Bezier recurrence rather
    // than the tabulation -- so it carries the same slack for the same reason.
    const std::vector<double> ctrl{1.0, 1.0, 1.0, 1.0};
    std::vector<double> curve(pts.size(), -99.0);
    pantr::span2d<const double> ctrl_view(ctrl.data(), ctrl.size(), 1);
    pantr::span2d<double> curve_view(curve.data(), pts.size(), 1);
    pantr::bezier::evaluate_bezier_1d<double>(ctrl_view, std::span<const double>(pts), curve_view);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        if (std::fabs(curve[i] - 1.0) > 8 * kEps) {
            std::printf("FAIL: constant Bezier at point %zu is %.17g\n", i, curve[i]);
            return 1;
        }
    }

    std::printf("consumer: partition of unity holds at %zu points\n", pts.size());
    std::printf("consumer: mdspan branch is %s\n",
#if defined(PANTR_HAS_STD_MDSPAN)
                "std::mdspan");
#else
                "Kokkos");
#endif
    return 0;
}
