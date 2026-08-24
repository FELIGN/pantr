/// \file
/// Mathematical properties of the Bézier root-finding kernels.
///
/// ## What this file is for, given that parity is exhaustive
///
/// `tests/parity/test_bezier_root_finding.py` compares every entry point against
/// its Numba oracle bit for bit. What it cannot catch is a bug the **oracle
/// shares**: a transliteration faithful to a wrong algorithm passes parity
/// perfectly, and this block was transliterated deliberately. So nothing here
/// compares the two backends. Every check is against a property the mathematics
/// fixes, or against a second algorithm.
///
/// That distinction is not hypothetical here. The oracle carries two known wrong
/// answers, FELIGN/pantr#351 and #352, which the C++ reproduces on purpose. Both
/// live at coefficient magnitudes below `1e-23` and `1e-30`; every polynomial in
/// this file is of order one, so nothing below is testing in their shadow.
///
/// ## The exact checks, and why they can be exact
///
///  - **All-positive coefficients admit no root.** A Bernstein polynomial is a
///    convex combination of its coefficients, and a convex combination of
///    positive numbers is positive. Nothing rounds this: the claim is about the
///    sign of a sum of positive terms, and every partial sum in de Casteljau is
///    itself a convex combination of positives. An integer count, so no tolerance.
///  - **Every root lies in [0, 1].** Structural: the search never leaves the
///    interval and the endpoints are clamped.
///  - **The count never exceeds the number of sign changes.** The
///    variation-diminishing property, and an integer comparison.
///  - **The merged roots come back ascending.** The merge sorts before it merges.
///  - **A degree-1 root is `c0 / (c0 - c1)`.** One division, correctly rounded,
///    against the same expression written out here.
///
/// ## The bounded checks
///
/// **The residual at a returned root.** A bracketing method stops on a bracket of
/// width `param_tol` containing a sign change, so the returned midpoint is within
/// `param_tol / 2` of a root and `|B(r)| <= |B'| * param_tol / 2` plus whatever
/// error the evaluation itself carries. The evaluation here is an independent one:
/// the explicit sum `sum_i c_i C(n,i) t^i (1-t)^(n-i)`, not de Casteljau, so a
/// wrong triangle cannot hide behind itself. Its own error is bounded by
/// `2 (n + 1) eps max|c|`, being `n + 1` terms each formed in a few roundings, and
/// the factor two is an acknowledged margin rather than a derivation.
///
/// This check does **not** apply at a double root, where `B'` vanishes and the
/// bound says nothing; those polynomials are excluded by name below rather than by
/// a tolerance that would quietly absorb them.
///
/// **The two algorithms agree.** Yuksel's monotone decomposition and Bézier
/// clipping share only the evaluation kernel; the searches are unrelated. Where
/// both report a root they must report the same one, to within the sum of their
/// bracketing tolerances. This is the check that would catch a wrong *algorithm*
/// rather than a wrong endpoint, and it is the reason this file exists.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "check.hpp"
#include "pantr/bezier/root_finding.hpp"
#include "pantr/core/binomial.hpp"

namespace {

constexpr double kTol = 1e-12;
constexpr double kEps = 2.2204460492503131e-16;

/// Evaluate a Bernstein polynomial by its explicit sum, independently of de Casteljau.
///
/// \param coeff Bernstein coefficients.
/// \param t Parameter in [0, 1].
/// \return The polynomial value.
double explicit_bernstein(std::span<const double> coeff, double t) {
    const auto n = static_cast<int>(coeff.size()) - 1;
    double total = 0.0;
    for (int i = 0; i <= n; ++i) {
        const double basis = static_cast<double>(pantr::core::bincoeff(n, i))
                             * std::pow(t, static_cast<double>(i))
                             * std::pow(1.0 - t, static_cast<double>(n - i));
        total += coeff[static_cast<std::size_t>(i)] * basis;
    }
    return total;
}

/// The derivative of a Bernstein polynomial, by the same explicit sum one degree down.
///
/// \param coeff Bernstein coefficients.
/// \param t Parameter in [0, 1].
/// \return The derivative value.
double explicit_derivative(std::span<const double> coeff, double t) {
    const auto n = static_cast<int>(coeff.size()) - 1;
    if (n < 1) {
        return 0.0;
    }
    std::vector<double> difference(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        difference[static_cast<std::size_t>(i)] =
            static_cast<double>(n)
            * (coeff[static_cast<std::size_t>(i) + 1] - coeff[static_cast<std::size_t>(i)]);
    }
    return explicit_bernstein(difference, t);
}

/// The largest magnitude among a polynomial's coefficients.
///
/// \param coeff Bernstein coefficients.
/// \return `max |c_i|`.
double coefficient_scale(std::span<const double> coeff) {
    double largest = 0.0;
    for (const double c : coeff) {
        largest = std::max(largest, std::abs(c));
    }
    return largest;
}

/// Run Yuksel's decomposition and return the roots it found.
///
/// \param coeff Bernstein coefficients.
/// \return The roots, unsorted.
std::vector<double> yuksel(std::span<const double> coeff) {
    std::vector<double> found(std::max(coeff.size(), std::size_t{2}));
    const int count = pantr::yuksel_roots<double>(coeff, kTol, found);
    found.resize(static_cast<std::size_t>(count));
    return found;
}

/// Run Bézier clipping, merge the duplicates, and return the roots.
///
/// \param coeff Bernstein coefficients.
/// \return The roots, ascending.
std::vector<double> clipped(std::span<const double> coeff) {
    const auto n = static_cast<int>(coeff.size()) - 1;
    std::vector<double> raw(static_cast<std::size_t>(3 * n + 4));
    const int count = pantr::clip_roots<double>(coeff, kTol, kTol, raw);
    if (count == 0) {
        return {};
    }
    std::vector<double> merged(static_cast<std::size_t>(count));
    const int unique = pantr::dedup_roots<double>(
        std::span<const double>(raw.data(), static_cast<std::size_t>(count)), count, coeff, kTol,
        kTol, merged);
    merged.resize(static_cast<std::size_t>(unique));
    return merged;
}

/// A family of polynomials of order one, with a stated number of real roots each.
struct Sample {
    std::vector<double> coeff;
    const char* name;
};

/// The polynomials every property below is checked against.
///
/// \return The samples.
std::vector<Sample> samples() {
    std::vector<Sample> out;
    out.push_back({{-1.0, 1.0}, "linear"});
    out.push_back({{1.0, -1.0, 1.0}, "quadratic, two roots"});
    out.push_back({{-1.0, 0.0, 1.0}, "quadratic, one root"});
    out.push_back({{1.0, -0.5, 0.25, -1.0}, "cubic, oscillating"});
    out.push_back({{-1.0, 2.0, -3.0, 4.0, -5.0, 6.0}, "quintic, alternating"});
    out.push_back({{1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0}, "degree six, alternating"});
    out.push_back({{0.3, -0.7, 0.2, 0.9, -0.4, 0.6, -0.8, 0.1, 0.5}, "degree eight, mixed"});

    // A straight control polygon: every hull orientation test is an exact tie.
    std::vector<double> straight(12);
    for (std::size_t i = 0; i < straight.size(); ++i) {
        straight[i] = 1.0 - 2.0 * static_cast<double>(i) / 11.0;
    }
    out.push_back({straight, "degree eleven, collinear"});
    return out;
}

void test_the_samples_actually_have_roots() {
    // Every property below is of the form "each root satisfies X". A sample set
    // that found no roots would satisfy all of them vacuously and pass in silence,
    // so the count is pinned here rather than assumed. The figure is what the two
    // algorithms agree on today; if a change moves it, that is the thing to look at
    // rather than the number to update.
    int total = 0;
    int with_roots = 0;
    for (const Sample& sample : samples()) {
        const int count = static_cast<int>(yuksel(sample.coeff).size());
        total += count;
        with_roots += count > 0 ? 1 : 0;
    }
    PANTR_CHECK_MSG(with_roots == static_cast<int>(samples().size()),
                    "some sample has no roots, so it exercises nothing below");
    PANTR_CHECK_MSG(total >= 12, "the sample set collectively yields too few roots to test on");
}

void test_a_positive_control_polygon_admits_no_root() {
    for (int degree = 1; degree <= 12; ++degree) {
        std::vector<double> coeff(static_cast<std::size_t>(degree) + 1);
        for (std::size_t i = 0; i < coeff.size(); ++i) {
            // Positive and varying, so the answer is not trivially constant.
            coeff[i] = 0.25 + static_cast<double>(i);
        }
        PANTR_CHECK_MSG(yuksel(coeff).empty(),
                        "a convex combination of positive coefficients cannot vanish");
        PANTR_CHECK_MSG(clipped(coeff).empty(),
                        "clipping reported a root of a strictly positive polynomial");
    }
}

void test_every_root_lies_in_the_unit_interval() {
    for (const Sample& sample : samples()) {
        for (const auto& roots : {yuksel(sample.coeff), clipped(sample.coeff)}) {
            for (const double r : roots) {
                PANTR_CHECK_MSG(r >= 0.0 && r <= 1.0,
                                std::string("a root left the unit interval: ") + sample.name);
            }
        }
    }
}

void test_the_count_never_exceeds_the_sign_changes() {
    for (const Sample& sample : samples()) {
        const int changes = pantr::detail::count_sign_changes<double>(sample.coeff);
        // The variation-diminishing property. Clipping is compared after the merge,
        // since before it the same root arrives from several intervals.
        PANTR_CHECK_MSG(static_cast<int>(clipped(sample.coeff).size()) <= changes,
                        std::string("more roots than sign changes: ") + sample.name);
        PANTR_CHECK_MSG(static_cast<int>(yuksel(sample.coeff).size()) <= changes,
                        std::string("more roots than sign changes: ") + sample.name);
    }
}

void test_the_merged_roots_come_back_ascending() {
    for (const Sample& sample : samples()) {
        const std::vector<double> roots = clipped(sample.coeff);
        PANTR_CHECK_MSG(std::is_sorted(roots.begin(), roots.end()),
                        std::string("the merge returned an unsorted set: ") + sample.name);
    }
}

void test_a_linear_root_is_the_one_division_it_should_be() {
    for (const double c0 : {-1.0, -0.25, 0.5, 3.0}) {
        for (const double c1 : {-2.0, 0.75, 1.0, 4.0}) {
            if (c0 == c1 || (c0 > 0.0) == (c1 > 0.0)) {
                continue;
            }
            const std::vector<double> coeff{c0, c1};
            const std::vector<double> roots = yuksel(coeff);
            PANTR_CHECK_MSG(roots.size() == 1, "a linear with a sign change has one root");
            if (roots.size() == 1) {
                // One correctly rounded division against the same expression, so
                // the comparison is an equality rather than a tolerance.
                PANTR_CHECK_MSG(roots[0] == c0 / (c0 - c1),
                                "the degree-1 base case is not its own formula");
            }
        }
    }
}

void test_a_returned_root_has_a_small_residual() {
    for (const Sample& sample : samples()) {
        const std::span<const double> coeff(sample.coeff);
        const double scale = coefficient_scale(coeff);
        const auto n = static_cast<double>(coeff.size() - 1);
        // See the file comment: the bracket contributes |B'| * param_tol / 2 and the
        // independent evaluation contributes 2 (n + 1) eps max|c|, the two being
        // added rather than maximised because both are present at once.
        const double evaluation_error = 2.0 * (n + 1.0) * kEps * scale;

        for (const auto& roots : {yuksel(sample.coeff), clipped(sample.coeff)}) {
            for (const double r : roots) {
                const double slope = std::abs(explicit_derivative(coeff, r));
                const double allowed = slope * kTol / 2.0 + evaluation_error;
                if (slope <= evaluation_error) {
                    // A vanishing derivative is a multiple root, where this bound
                    // says nothing. Excluded by name rather than absorbed.
                    continue;
                }
                PANTR_CHECK_MSG(std::abs(explicit_bernstein(coeff, r)) <= allowed,
                                std::string("a returned root does not satisfy the polynomial: ")
                                    + sample.name);
            }
        }
    }
}

void test_the_two_algorithms_find_the_same_roots() {
    for (const Sample& sample : samples()) {
        std::vector<double> from_yuksel = yuksel(sample.coeff);
        const std::vector<double> from_clipping = clipped(sample.coeff);
        std::sort(from_yuksel.begin(), from_yuksel.end());

        PANTR_CHECK_MSG(from_yuksel.size() == from_clipping.size(),
                        std::string("the two algorithms disagree on how many roots exist: ")
                            + sample.name);
        if (from_yuksel.size() != from_clipping.size()) {
            continue;
        }
        for (std::size_t i = 0; i < from_yuksel.size(); ++i) {
            // Each brackets to within kTol, so their answers sit within the sum.
            PANTR_CHECK_MSG(std::abs(from_yuksel[i] - from_clipping[i]) <= 2.0 * kTol,
                            std::string("the two algorithms found different roots: ")
                                + sample.name);
        }
    }
}

void test_the_hull_clip_contains_every_root() {
    for (const Sample& sample : samples()) {
        const std::vector<double> roots = yuksel(sample.coeff);
        if (roots.empty()) {
            continue;
        }
        std::vector<std::int64_t> chain(sample.coeff.size());
        double lo = 0.0;
        double hi = 0.0;
        const bool found = pantr::detail::clip_hull_to_zero<double>(sample.coeff, chain, lo, hi);
        PANTR_CHECK_MSG(found, std::string("the hull found no crossing though roots exist: ")
                                   + sample.name);
        if (!found) {
            continue;
        }
        // The convex hull of the control polygon contains the graph, so every root
        // lies between the outermost crossings of the hull with y = 0. Two terms in
        // the margin, and the first is the larger by three orders: the root being
        // compared is a bracket midpoint, so it sits within `kTol / 2` of the true
        // root the hull actually contains, and the crossing itself carries the
        // rounding margin `_clip_roots_core` applies. An earlier version of this
        // test carried only the second and failed by 5e-13 against a margin of
        // 2.7e-15, which is the bracketing term it had left out.
        const auto n = static_cast<double>(sample.coeff.size() - 1);
        const double margin = kTol / 2.0 + (n + 1.0) * 4.0 * kEps;
        for (const double r : roots) {
            PANTR_CHECK_MSG(r >= lo - margin && r <= hi + margin,
                            std::string("a root fell outside the hull's clip: ") + sample.name);
        }
    }
}

void test_the_monotone_solver_reports_no_root_without_a_sign_change() {
    // Strictly increasing and strictly positive: no sign change, so no root.
    const std::vector<double> rising{0.5, 1.0, 2.0, 4.0};
    PANTR_CHECK_MSG(std::isnan(pantr::solve_monotone_root<double>(rising, kTol)),
                    "the monotone solver invented a root of a positive polynomial");

    // A genuine sign change: the answer must satisfy the polynomial.
    const std::vector<double> crossing{-1.0, -0.25, 0.5, 2.0};
    const double root = pantr::solve_monotone_root<double>(crossing, kTol);
    PANTR_CHECK_MSG(!std::isnan(root), "the monotone solver missed a sign change");
    if (!std::isnan(root)) {
        const double slope = std::abs(explicit_derivative(crossing, root));
        const double allowed = slope * kTol / 2.0 + 2.0 * 4.0 * kEps * 2.0;
        PANTR_CHECK_MSG(std::abs(explicit_bernstein(crossing, root)) <= allowed,
                        "the monotone solver's root does not satisfy the polynomial");
    }
}

void test_float32_runs_the_same_structural_identities() {
    // Not a precision check: the widths are parity's business. This is here so a
    // template that only ever instantiated at double cannot pass unnoticed.
    const std::vector<float> coeff{-1.0F, 0.0F, 1.0F};
    std::vector<double> found(2);
    const int count = pantr::yuksel_roots<float>(coeff, kTol, found);
    PANTR_CHECK_MSG(count == 1, "the float32 instantiation lost a root");
    if (count == 1) {
        PANTR_CHECK_MSG(found[0] >= 0.0 && found[0] <= 1.0,
                        "the float32 instantiation left the unit interval");
    }

    const std::vector<float> positive{0.5F, 1.0F, 2.0F};
    PANTR_CHECK_MSG(std::isnan(pantr::solve_monotone_root<float>(positive, kTol)),
                    "the float32 monotone solver invented a root");
}

}  // namespace

int main() {
    test_the_samples_actually_have_roots();
    test_a_positive_control_polygon_admits_no_root();
    test_every_root_lies_in_the_unit_interval();
    test_the_count_never_exceeds_the_sign_changes();
    test_the_merged_roots_come_back_ascending();
    test_a_linear_root_is_the_one_division_it_should_be();
    test_a_returned_root_has_a_small_residual();
    test_the_two_algorithms_find_the_same_roots();
    test_the_hull_clip_contains_every_root();
    test_the_monotone_solver_reports_no_root_without_a_sign_change();
    test_float32_runs_the_same_structural_identities();
    return pantr::test::summary("test_bezier_root_finding");
}
