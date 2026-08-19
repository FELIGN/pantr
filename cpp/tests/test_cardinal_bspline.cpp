/// \file
/// Properties of the Cox-de Boor cardinal B-spline tabulation.
///
/// Every tolerance below is derived from the same forward-error argument, stated
/// once here and referred to by name afterwards.
///
/// ## The forward-error bound
///
/// Write `u` for the unit roundoff (`eps / 2`, but `eps` is used throughout
/// below, which only makes the bounds looser). For evaluation points in
/// `[0, 1]`, `term = (r + 1 - u) / k` lies in `[0, 1]`, so every stage of the
/// recurrence forms a convex combination: the stage map has non-negative
/// coefficients that sum to one, i.e. it is a stochastic matrix, and a
/// stochastic matrix does not amplify an absolute error vector in the 1-norm.
///
/// So the error accumulated over the stages is the *sum* of what each stage
/// introduces, with no growth factor. Per stage, and per index `r`, the
/// roundings are: two forming `term` (one add, one multiply), at most two
/// forming `saved + nr_old * term` (one multiply, one add -- one if the compiler
/// fuses them), and two forming `nr_old * (1 - term)` (one subtract, one
/// multiply). Six roundings, each contributing at most `eps` times the magnitude
/// of the quantity it perturbs. Because the values at any stage are
/// non-negative and sum to one, those magnitudes sum to one across `r`, so a
/// whole stage introduces at most `6 * eps` of absolute error in the 1-norm.
///
/// Over `n` stages:
///
///     sum_r |N_computed[r] - N_exact[r]|  <=  6 * n * eps                  (1)
///
/// and in particular each component obeys the same bound. Second-order terms are
/// `O(n^2 eps^2)`, which at `n = 20` in float64 is about `5e-30`, so they are
/// dropped rather than carried.
///
/// Partition of unity additionally sums `n + 1` computed values, costing at most
/// `n * eps` more, hence `7 * n * eps`. The constant used below is `8`, which is
/// (1) rounded up to the next power of two -- an acknowledged safety factor of
/// `8 / 7`, and the only fudge in this file.

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/basis/cardinal_bspline.hpp"
#include "pantr/core/mdspan.hpp"

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

/// The bound of equation (1) in this file's header, plus the summation term.
constexpr double partition_bound(int degree) {
    return 8.0 * static_cast<double>(degree) * kEps;
}

/// Evaluation points, chosen to include both endpoints and to avoid landing only
/// on values whose binary representation is exact.
std::vector<double> sample_points() {
    std::vector<double> pts;
    constexpr int n_pts = 65;
    pts.reserve(n_pts);
    for (int i = 0; i < n_pts; ++i) {
        pts.push_back(static_cast<double>(i) / static_cast<double>(n_pts - 1));
    }
    return pts;
}

std::vector<double> tabulate(int degree, const std::vector<double>& pts) {
    std::vector<double> out(pts.size() * static_cast<std::size_t>(degree + 1), 0.0);
    const pantr::span2d<double> view(out.data(), pts.size(),
                                     static_cast<std::size_t>(degree + 1));
    pantr::tabulate_cardinal_bspline_1d<double>(degree, std::span<const double>(pts), view);
    return out;
}

/// Partition of unity. The defining property of a B-spline basis, and the one
/// that fails loudly if the recurrence is wrong anywhere.
void partition_of_unity() {
    const auto pts = sample_points();
    for (int degree = 0; degree <= 20; ++degree) {
        const auto values = tabulate(degree, pts);
        const auto stride = static_cast<std::size_t>(degree + 1);
        const double bound = partition_bound(degree) + kEps;  // the +eps covers degree 0

        for (std::size_t j = 0; j < pts.size(); ++j) {
            double sum = 0.0;
            for (std::size_t r = 0; r < stride; ++r) {
                sum += values[j * stride + r];
            }
            PANTR_CHECK_MSG(std::abs(sum - 1.0) <= bound,
                            "degree " + std::to_string(degree) + " at u=" +
                                std::to_string(pts[j]) + ": |sum-1| = " +
                                std::to_string(std::abs(sum - 1.0)) + " > " +
                                std::to_string(bound));
        }
    }
}

/// Non-negativity. Not a rounding statement: every value is a product or a sum of
/// non-negative quantities, so a negative entry means the recurrence took a
/// branch it should not have, not that it lost precision.
void values_are_non_negative() {
    const auto pts = sample_points();
    for (int degree = 0; degree <= 20; ++degree) {
        const auto values = tabulate(degree, pts);
        for (const double v : values) {
            PANTR_CHECK_MSG(v >= 0.0, "degree " + std::to_string(degree) +
                                          ": negative value " + std::to_string(v));
        }
    }
}

/// Symmetry: the cardinal basis on the central span satisfies
/// `N_r(u) = N_{n-r}(1-u)`, because the knot vector is uniform and the span is
/// central. An independent structural property -- it constrains the answer
/// without recomputing it.
void basis_is_symmetric() {
    const auto pts = sample_points();
    std::vector<double> mirrored(pts.size());
    for (std::size_t j = 0; j < pts.size(); ++j) {
        mirrored[j] = 1.0 - pts[pts.size() - 1 - j];
    }

    for (int degree = 0; degree <= 12; ++degree) {
        const auto direct = tabulate(degree, pts);
        const auto reflected = tabulate(degree, mirrored);
        const auto stride = static_cast<std::size_t>(degree + 1);
        // Two independent evaluations, so twice the bound of equation (1).
        const double bound = 2.0 * partition_bound(degree) + kEps;

        for (std::size_t j = 0; j < pts.size(); ++j) {
            for (std::size_t r = 0; r < stride; ++r) {
                const double a = direct[j * stride + r];
                const double b = reflected[(pts.size() - 1 - j) * stride + (stride - 1 - r)];
                PANTR_CHECK_MSG(std::abs(a - b) <= bound,
                                "degree " + std::to_string(degree) + " r=" +
                                    std::to_string(r) + ": " + std::to_string(a) + " vs " +
                                    std::to_string(b));
            }
        }
    }
}

/// Closed forms for the low degrees, written out independently of the
/// recurrence. This is the only check here that is a genuine oracle rather than
/// a property: it compares against an expression derived on paper, so it would
/// catch a recurrence that is self-consistent and wrong.
///
/// Bound: equation (1) for the kernel, plus a handful of roundings for the
/// closed form itself. At these degrees that is well under `16 * eps`, which is
/// what is used.
void low_degrees_match_closed_forms() {
    const auto pts = sample_points();
    const double bound = 16.0 * kEps;

    {  // degree 1: N_0 = 1 - u, N_1 = u
        const auto v = tabulate(1, pts);
        for (std::size_t j = 0; j < pts.size(); ++j) {
            const double t = pts[j];
            PANTR_CHECK(std::abs(v[2 * j + 0] - (1.0 - t)) <= bound);
            PANTR_CHECK(std::abs(v[2 * j + 1] - t) <= bound);
        }
    }

    {  // degree 2: (1-u)^2/2, (1 + 2u - 2u^2)/2, u^2/2
        const auto v = tabulate(2, pts);
        for (std::size_t j = 0; j < pts.size(); ++j) {
            const double t = pts[j];
            PANTR_CHECK(std::abs(v[3 * j + 0] - (1.0 - t) * (1.0 - t) / 2.0) <= bound);
            PANTR_CHECK(std::abs(v[3 * j + 1] - (1.0 + 2.0 * t - 2.0 * t * t) / 2.0) <= bound);
            PANTR_CHECK(std::abs(v[3 * j + 2] - t * t / 2.0) <= bound);
        }
    }

    {  // degree 3: (1-u)^3/6, (3u^3 - 6u^2 + 4)/6, (-3u^3 + 3u^2 + 3u + 1)/6, u^3/6
        const auto v = tabulate(3, pts);
        for (std::size_t j = 0; j < pts.size(); ++j) {
            const double t = pts[j];
            const double t2 = t * t;
            const double t3 = t2 * t;
            PANTR_CHECK(std::abs(v[4 * j + 0] - (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0) <= bound);
            PANTR_CHECK(std::abs(v[4 * j + 1] - (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0) <= bound);
            PANTR_CHECK(std::abs(v[4 * j + 2] -
                                 (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0) <= bound);
            PANTR_CHECK(std::abs(v[4 * j + 3] - t3 / 6.0) <= bound);
        }
    }
}

/// The float32 instantiation: it compiles, it runs, and it satisfies partition of
/// unity to the float32 STORAGE bound rather than to a float32 forward bound.
///
/// **This test is not the check that `accumulator_t` is wired up**, though an
/// earlier comment here said it was. It cannot be: partition of unity is
/// dominated by the float32 stores, and the accumulator's precision barely shows
/// through them. Measured, by building this same function against a mutant trait
/// answering `float` for `float` -- the change that removes the entire point of
/// `accumulator_t` -- it **passes at every degree from 0 to 20**, with an
/// observed error between 4.8% and 10% of the bound. It never comes close to
/// failing, so no re-derivation of the constant would rescue the claim; the
/// invariant simply does not see the accumulator.
///
/// The checks that do are exact and live elsewhere:
/// `static_assert(std::same_as<pantr::accumulator_t<float>, double>)` in
/// cpp/tests/test_scalar_generic.cpp, which is a build failure rather than a
/// tolerance, and `test_float32_intermediates_are_float64` in
/// tests/test_cpp_parity.py, which compares against an explicit model of the
/// arithmetic contract in both directions. This one keeps its own smaller job:
/// the `float` instantiation exists, runs, and is not grossly wrong.
void float32_instantiation_holds_partition_of_unity() {
    constexpr float kEps32 = std::numeric_limits<float>::epsilon();
    std::vector<float> pts;
    for (int i = 0; i <= 32; ++i) {
        pts.push_back(static_cast<float>(i) / 32.0F);
    }

    for (int degree = 0; degree <= 10; ++degree) {
        const auto stride = static_cast<std::size_t>(degree + 1);
        std::vector<float> out(pts.size() * stride, 0.0F);
        const pantr::span2d<float> view(out.data(), pts.size(), stride);
        pantr::tabulate_cardinal_bspline_1d<float>(degree, std::span<const float>(pts), view);

        // The bound, derived the way equation (1) above is, and NOT by counting
        // stores. Stage `k` stores `k + 1` values, so the kernel commits
        // `n * (n + 3) / 2` storage roundings -- 65 at degree 10, not 11, which is
        // what an earlier comment here claimed.
        //
        // The count is not what the bound rests on. At every stage the stored
        // values are non-negative and sum to one, so however many of them there
        // are their roundings add at most `eps32 / 2` in the 1-norm; and the next
        // stage's map is stochastic, so it does not amplify what came before.
        // That gives `n * eps32 / 2` over the stages. The summation below adds
        // `n` float32 additions on partial sums bounded by one, for `n * eps32 / 2`
        // more. Total `n * eps32`. The recurrence itself runs in float64 and
        // contributes `6 * n * eps64 ~ 1e-15`, negligible against `eps32 ~ 1e-7`.
        //
        // `2 * (degree + 1)` rather than `degree` is about a factor of two of
        // acknowledged margin, and covers degree 0, where the derived bound is 0
        // and the answer is exact anyway.
        //
        // Correct, and loose in a way worth recording: measured, the error is FLAT
        // in degree (1 eps32 from degree 4 through 20, 1.5 at degree 16) because
        // the per-stage roundings have mixed signs and cancel, while this bound
        // grows linearly. It is not a tight statement about this kernel.
        const float bound = 2.0F * static_cast<float>(degree + 1) * kEps32;
        for (std::size_t j = 0; j < pts.size(); ++j) {
            float sum = 0.0F;
            for (std::size_t r = 0; r < stride; ++r) {
                sum += out[j * stride + r];
            }
            PANTR_CHECK_MSG(std::abs(sum - 1.0F) <= bound,
                            "float32 degree " + std::to_string(degree) + ": |sum-1| = " +
                                std::to_string(std::abs(sum - 1.0F)));
        }
    }
}

}  // namespace

int main() {
    partition_of_unity();
    values_are_non_negative();
    basis_is_symmetric();
    low_degrees_match_closed_forms();
    float32_instantiation_holds_partition_of_unity();
    return pantr::test::summary("test_cardinal_bspline");
}
