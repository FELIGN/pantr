/// \file
/// Properties of the Bernstein ratio-recurrence tabulation.
///
/// ## The forward-error bound
///
/// On `[0, 1]` every Bernstein value is non-negative and the row sums to one, so
/// the row is a probability vector and the error analysis is the same shape as
/// the cardinal B-spline one: absolute errors add across the row without a growth
/// factor.
///
/// Per step the roundings are: one forming `const_factor` (a subtract that is
/// exact for integer operands, then a divide), one multiplying by it, and one
/// multiplying by the ratio -- plus the store, which at float64 is not a rounding
/// since `Acc` is `double` there. Call it three roundings per step, each
/// perturbing a quantity by at most `eps` of its own magnitude, and the errors
/// compound multiplicatively along the row because each term is built from the
/// previous one. After `i` steps the relative error of `B_i` is therefore at most
/// about `3 * i * eps`, and summing over a row whose entries sum to one gives
///
///     |sum_i B_i(u) - 1|  <=  3 * n * eps  +  n * eps                      (1)
///
/// with the second term the summation itself. The constant used below is `8`,
/// generous against the `4` of (1), because the seed `pow((1-u), n)` carries a
/// libm error this argument does not bound: the C standard does not require `pow`
/// to be correctly rounded, and glibc's is documented at up to 1 ulp. One ulp on
/// the seed propagates as one more relative `eps` through every later term, which
/// (1) would have to carry as `4 * n * eps`. Rounding that to the next power of
/// two is the whole of the safety factor.
///
/// ## What each test would catch
///
/// The mirrored and unmirrored kernels are separately exercised by degree: the
/// dispatch happens at degree 20 in float64 and 6 in float32, so a test at degree
/// 24 runs code a test at degree 8 does not.
///
/// ## The recurrence coefficient is mutation-checked, not merely covered
///
/// Coverage says these assertions run; it does not say they would notice a wrong
/// recurrence. They do. Perturbing `const_factor` in
/// `basis/bernstein.hpp::tabulate_bernstein_1d` from `(n - i + 1) / i` to
/// `(n - i + 2) / i`, at both of its sites, makes this file fail while the rest of
/// the C++ suite still passes. That is the check worth recording: the failure is
/// attributable to this file rather than to the suite at large.
///
/// Redo it by making that edit and running `ctest`; nothing automates it, because
/// a mutation harness would have to keep a copy of the kernel and that copy is the
/// thing that goes stale.

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/basis/bernstein.hpp"
#include "pantr/core/mdspan.hpp"

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

/// The bound of equation (1) in this file's header, with its safety factor.
constexpr double partition_bound(int degree) {
    return 8.0 * static_cast<double>(degree) * kEps;
}

std::vector<double> sample_points() {
    std::vector<double> pts;
    constexpr int n_pts = 65;
    pts.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        pts.push_back(static_cast<double>(i) / static_cast<double>(n_pts - 1));
    }
    return pts;
}

template <class T>
std::vector<T> tabulate(int degree, const std::vector<T>& pts) {
    std::vector<T> out(pts.size() * static_cast<std::size_t>(degree + 1), T(0));
    const pantr::span2d<T> view(out.data(), pts.size(), static_cast<std::size_t>(degree + 1));
    pantr::tabulate_bernstein_1d<T>(degree, std::span<const T>(pts), view);
    return out;
}

/// The safe-degree threshold is the one the oracle tabulates, recomputed here
/// from the format. If this drifts, every degree-based claim below picks the
/// wrong kernel and stops testing what it says it tests.
void safe_degree_matches_the_oracle() {
    PANTR_CHECK(pantr::bernstein_max_safe_degree_no_mirror<double>() == 20);
    PANTR_CHECK(pantr::bernstein_max_safe_degree_no_mirror<float>() == 6);
}

/// Partition of unity, across both kernels.
void partition_of_unity() {
    const std::vector<double> pts = sample_points();
    for (int degree : {0, 1, 3, 8, 20, 21, 24, 40}) {
        const std::vector<double> vals = tabulate(degree, pts);
        const auto width = static_cast<std::size_t>(degree + 1);
        for (std::size_t j = 0; j < pts.size(); ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < width; ++i) {
                sum += vals[j * width + i];
            }
            PANTR_CHECK_MSG(std::abs(sum - 1.0) <= partition_bound(degree),
                            "degree " + std::to_string(degree) + " at point index " +
                                std::to_string(j));
        }
    }
}

/// Non-negativity on `[0, 1]`. It is what makes the error argument above valid,
/// so a failure here invalidates every tolerance in this file rather than merely
/// failing one claim.
void values_are_non_negative() {
    const std::vector<double> pts = sample_points();
    for (int degree : {1, 5, 20, 24}) {
        const std::vector<double> vals = tabulate(degree, pts);
        for (double v : vals) {
            PANTR_CHECK(v >= 0.0);
        }
    }
}

/// `B_i,n(u) = B_{n-i},n(1-u)`. The mirrored kernel is built on this identity, so
/// checking it at a degree above the threshold tests the mirror rather than
/// restating the definition.
void basis_is_symmetric() {
    for (int degree : {3, 24}) {
        const std::vector<double> pts = sample_points();
        std::vector<double> reflected(pts.size());
        for (std::size_t j = 0; j < pts.size(); ++j) {
            reflected[j] = 1.0 - pts[pts.size() - 1 - j];
        }
        const std::vector<double> a = tabulate(degree, pts);
        const std::vector<double> b = tabulate(degree, reflected);
        const auto width = static_cast<std::size_t>(degree + 1);
        for (std::size_t j = 0; j < pts.size(); ++j) {
            for (std::size_t i = 0; i < width; ++i) {
                const double lhs = a[j * width + i];
                const double rhs = b[(pts.size() - 1 - j) * width + (width - 1 - i)];
                PANTR_CHECK(std::abs(lhs - rhs) <= partition_bound(degree));
            }
        }
    }
}

/// The endpoints are exact: `B_0(0) = 1` and `B_n(1) = 1`, everything else zero.
/// `u == 1` takes a dedicated branch in the unmirrored kernel and the ordinary
/// path in the mirrored one, so both are covered.
void endpoints_are_exact() {
    for (int degree : {1, 5, 20, 24}) {
        const std::vector<double> pts{0.0, 1.0};
        const std::vector<double> vals = tabulate(degree, pts);
        const auto width = static_cast<std::size_t>(degree + 1);
        for (std::size_t i = 0; i < width; ++i) {
            PANTR_CHECK(vals[i] == (i == 0 ? 1.0 : 0.0));
            PANTR_CHECK(vals[width + i] == (i == width - 1 ? 1.0 : 0.0));
        }
    }
}

/// Degree 2 against its closed form. A recurrence can satisfy partition of unity,
/// non-negativity and symmetry and still be the wrong basis; this is the test
/// that pins which polynomials these are.
void low_degrees_match_closed_forms() {
    const std::vector<double> pts{0.0, 0.25, 0.5, 0.75, 1.0};
    const std::vector<double> vals = tabulate(2, pts);
    for (std::size_t j = 0; j < pts.size(); ++j) {
        const double u = pts[j];
        const double expected[3] = {(1 - u) * (1 - u), 2 * u * (1 - u), u * u};
        for (std::size_t i = 0; i < 3; ++i) {
            PANTR_CHECK(std::abs(vals[j * 3 + i] - expected[i]) <= 8 * kEps);
        }
    }
}

/// The underflow the mirror exists to prevent. At a degree past the threshold and
/// a point one ulp below 1, the unmirrored seed `(1-u)^n` is exactly zero, which
/// would zero the whole row and take partition of unity to 0 rather than 1. This
/// is the frontier case: it fails against the kernel without the mirror.
void the_mirror_prevents_the_underflow_it_was_added_for() {
    const double u = std::nextafter(1.0, 0.0);
    PANTR_CHECK(std::pow(1.0 - u, 40.0) == 0.0);

    const std::vector<double> pts{u};
    const int degree = 40;
    const std::vector<double> vals = tabulate(degree, pts);
    double sum = 0.0;
    for (double v : vals) {
        sum += v;
    }
    PANTR_CHECK_MSG(std::abs(sum - 1.0) <= partition_bound(degree), "mirrored seed underflowed");
    PANTR_CHECK(vals[static_cast<std::size_t>(degree)] > 0.99);
}

/// float32 holds partition of unity too, at its own epsilon and its own
/// threshold: degree 8 is mirrored in float32 while it is not in float64.
void float32_instantiation_holds_partition_of_unity() {
    constexpr auto kEps32 = static_cast<double>(std::numeric_limits<float>::epsilon());
    const std::vector<float> pts{0.0F, 0.125F, 0.5F, 0.875F, 1.0F};
    for (int degree : {3, 6, 8, 20}) {
        const std::vector<float> vals = tabulate(degree, pts);
        const auto width = static_cast<std::size_t>(degree + 1);
        for (std::size_t j = 0; j < pts.size(); ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < width; ++i) {
                sum += static_cast<double>(vals[j * width + i]);
            }
            PANTR_CHECK(std::abs(sum - 1.0) <= 8.0 * static_cast<double>(degree) * kEps32);
        }
    }
}

}  // namespace

int main() {
    safe_degree_matches_the_oracle();
    partition_of_unity();
    values_are_non_negative();
    basis_is_symmetric();
    endpoints_are_exact();
    low_degrees_match_closed_forms();
    the_mirror_prevents_the_underflow_it_was_added_for();
    float32_instantiation_holds_partition_of_unity();
    return pantr::test::summary("test_bernstein");
}
