/// \file
/// Properties of `pantr/bspline/tabulate.hpp` and the new `tabulate_bernstein_deriv_1d`
/// checkable **without the Python oracle** -- that is `tests/parity/` job, not this
/// file's. Every check here is one a wrong implementation would fail, not a
/// recomputation of the recurrence it exercises.
///
/// ## Where the tolerances below come from
///
/// - **Partition of unity** (`sum_i N_i(u) == 1`): the recurrence commits at most
///   `degree` stages of a convex combination plus the final `degree + 1`-term sum, so
///   `(2 * degree + 1) * (eps / 2)` bounds the relative error -- the same shape as
///   `test_bernstein.cpp`'s `partition_bound`, derived independently here because the
///   two recurrences (Cox-de Boor vs. the ratio recurrence) are different algorithms.
/// - **Derivative sums** (`sum_i N_i^(k)(u) == 0`, `k >= 1`): the target is exactly
///   zero, so a *relative* bound is meaningless -- design/backend_parity.md Rule 2 says
///   use an absolute one, scaled by the row's own largest magnitude instead of by 1.
///   The A2.3 recursion builds each row from `order` terms through the same kind of
///   division-then-multiply-add step the partition-of-unity bound counts, so the same
///   `C * order * eps` shape applies, now against the row's own scale. `C = 8` mirrors
///   `test_scalar_generic.cpp`'s identical identity for the cardinal B-spline kernel;
///   it is not independently re-derived here and is flagged as a margin, not a proof.
/// - **Cross-algorithm agreement** (the Bézier-like fast path against the general
///   Cox-de Boor recurrence, and `tabulate_bernstein_deriv_1d` against a closed form
///   built from `tabulate_bernstein_1d` one degree down): two different algorithms for
///   the same polynomial, so "agree" means "agree to rounding" and the bound combines
///   each algorithm's own partition-style error. Where the combination is a rough
///   estimate rather than a tight derivation, the comment at the site says so.
///
/// Exact (no-tolerance) claims: non-negativity (a sum of non-negative convex-
/// combination terms), derivative rows above the degree (the k-th derivative of a
/// degree-p polynomial vanishes identically for k > p), degree 0 (one basis function,
/// value 1, every derivative row 0), and row 0 of `basis_derivs_1d` against
/// `basis_funcs_1d` (both build the same `ndu` upper triangle by the same operations in
/// the same order).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "check.hpp"
#include "pantr/basis/bernstein.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/bspline/tabulate.hpp"
#include "pantr/core/binomial.hpp"
#include "pantr/core/mdspan.hpp"

namespace {

using pantr::at;
using pantr::span2d;
using pantr::span_nd;
using pantr::bspline::BsplineSpace1D;
using pantr::bspline::KnotSnapping;

// --------------------------------------------------------------------------
// Knot vectors and sample points
// --------------------------------------------------------------------------

/// A clamped, open knot vector over `num_elems` uniform elements.
template <class T>
std::vector<T> clamped_uniform_knots(std::int64_t degree, int num_elems) {
    std::vector<T> knots(static_cast<std::size_t>(degree) + 1, T(0));
    for (int i = 1; i < num_elems; ++i) {
        knots.push_back(static_cast<T>(i));
    }
    for (std::int64_t i = 0; i <= degree; ++i) {
        knots.push_back(static_cast<T>(num_elems));
    }
    return knots;
}

/// An unclamped knot vector: a plain integer run with no repeats anywhere, so the
/// domain's right end (`knots[size - degree - 1]`) is not the vector's last knot for
/// `degree >= 1` -- the case `find_span_and_first_basis`'s upper clamp exists for.
template <class T>
std::vector<T> unclamped_uniform_knots(std::int64_t degree) {
    std::vector<T> knots;
    const auto len = static_cast<std::size_t>(2 * degree + 4);
    knots.reserve(len);
    for (std::size_t i = 0; i < len; ++i) {
        knots.push_back(static_cast<T>(i));
    }
    return knots;
}

/// Clamped ends, one interior knot repeated twice: reduced continuity, no empty span.
template <class T>
std::vector<T> interior_repeat_knots(std::int64_t degree) {
    std::vector<T> knots(static_cast<std::size_t>(degree) + 1, T(0));
    knots.push_back(T(1));
    knots.push_back(T(1));
    for (std::int64_t i = 0; i <= degree; ++i) {
        knots.push_back(T(2));
    }
    return knots;
}

/// Clamped ends, an interior knot repeated `degree + 1` times: a genuine empty span
/// (a zero-width parametric interval among the repeats) and a C^-1 discontinuity.
template <class T>
std::vector<T> full_multiplicity_interior_knots(std::int64_t degree) {
    std::vector<T> knots(static_cast<std::size_t>(degree) + 1, T(0));
    for (std::int64_t i = 0; i <= degree; ++i) {
        knots.push_back(T(1));
    }
    for (std::int64_t i = 0; i <= degree; ++i) {
        knots.push_back(T(2));
    }
    return knots;
}

/// The four knot-vector families the properties below are checked over. The two
/// interior-repeat families need `degree >= 1` to be distinct from the clamped case.
template <class T>
std::vector<std::vector<T>> knot_vector_family(std::int64_t degree) {
    std::vector<std::vector<T>> vectors;
    vectors.push_back(clamped_uniform_knots<T>(degree, 3));
    vectors.push_back(unclamped_uniform_knots<T>(degree));
    if (degree >= 1) {
        vectors.push_back(interior_repeat_knots<T>(degree));
        vectors.push_back(full_multiplicity_interior_knots<T>(degree));
    }
    return vectors;
}

/// `n` points evenly spaced over `[lo, hi]`, endpoints included.
template <class T>
std::vector<T> sample_domain(T lo, T hi, int n) {
    std::vector<T> pts;
    pts.reserve(static_cast<std::size_t>(n));
    const double a = static_cast<double>(lo);
    const double b = static_cast<double>(hi);
    for (int i = 0; i < n; ++i) {
        const double frac = static_cast<double>(i) / static_cast<double>(n - 1);
        pts.push_back(static_cast<T>(a + frac * (b - a)));
    }
    return pts;
}

// --------------------------------------------------------------------------
// Tabulation helpers
// --------------------------------------------------------------------------

template <class T>
std::vector<T> tabulate_general_basis(std::span<const T> knots, std::int64_t degree,
                                      bool periodic, const std::vector<T>& pts,
                                      std::vector<std::int64_t>& first_basis) {
    const auto order = static_cast<std::size_t>(degree) + 1;
    std::vector<T> out(pts.size() * order, T(0));
    first_basis.assign(pts.size(), 0);
    const span2d<T> view(out.data(), pts.size(), order);
    pantr::bspline::basis_funcs_1d<T>(knots, degree, periodic, std::span<const T>(pts), view,
                                      std::span<std::int64_t>(first_basis));
    return out;
}

template <class T>
std::vector<T> tabulate_general_derivs(std::span<const T> knots, std::int64_t degree,
                                       bool periodic, std::int64_t n_deriv,
                                       const std::vector<T>& pts,
                                       std::vector<std::int64_t>& first_basis) {
    const auto order = static_cast<std::size_t>(degree) + 1;
    const auto rows = static_cast<std::size_t>(n_deriv) + 1;
    std::vector<T> out(pts.size() * rows * order, T(0));
    first_basis.assign(pts.size(), 0);
    const span_nd<T, 3> view(out.data(), pts.size(), rows, order);
    pantr::bspline::basis_derivs_1d<T>(knots, degree, periodic, n_deriv, std::span<const T>(pts),
                                       view, std::span<std::int64_t>(first_basis));
    return out;
}

constexpr double eps_of(float /*unused*/) {
    return static_cast<double>(std::numeric_limits<float>::epsilon());
}
constexpr double eps_of(double /*unused*/) {
    return std::numeric_limits<double>::epsilon();
}

// --------------------------------------------------------------------------
// (1) Partition of unity, general knots
// --------------------------------------------------------------------------

template <class T>
void check_partition_of_unity_general_knots() {
    const double eps = eps_of(T(0));
    for (std::int64_t degree = 0; degree <= 5; ++degree) {
        const auto order = static_cast<std::size_t>(degree) + 1;
        const double bound = (2.0 * static_cast<double>(degree) + 1.0) * (eps / 2.0);
        for (const auto& knots : knot_vector_family<T>(degree)) {
            const T domain_lo = knots[static_cast<std::size_t>(degree)];
            const T domain_hi = knots[knots.size() - static_cast<std::size_t>(degree) - 1];
            const auto pts = sample_domain<T>(domain_lo, domain_hi, 33);
            std::vector<std::int64_t> first_basis;
            const auto vals =
                tabulate_general_basis<T>(std::span<const T>(knots), degree, false, pts, first_basis);
            for (std::size_t j = 0; j < pts.size(); ++j) {
                double sum = 0.0;
                for (std::size_t i = 0; i < order; ++i) {
                    sum += static_cast<double>(vals[j * order + i]);
                }
                PANTR_CHECK_MSG(std::abs(sum - 1.0) <= bound,
                                "degree " + std::to_string(degree) + " point idx " +
                                    std::to_string(j) + ": sum " + std::to_string(sum));
            }
        }
    }
}

// --------------------------------------------------------------------------
// (2) Non-negativity, exact
// --------------------------------------------------------------------------

template <class T>
void check_non_negativity() {
    for (std::int64_t degree = 0; degree <= 5; ++degree) {
        for (const auto& knots : knot_vector_family<T>(degree)) {
            const T domain_lo = knots[static_cast<std::size_t>(degree)];
            const T domain_hi = knots[knots.size() - static_cast<std::size_t>(degree) - 1];
            const auto pts = sample_domain<T>(domain_lo, domain_hi, 33);
            std::vector<std::int64_t> first_basis;
            const auto vals =
                tabulate_general_basis<T>(std::span<const T>(knots), degree, false, pts, first_basis);
            for (const T v : vals) {
                PANTR_CHECK(v >= T(0));
            }
        }
    }
}

// --------------------------------------------------------------------------
// (3) Derivative sums vanish, absolute and row-scaled
// --------------------------------------------------------------------------

template <class T>
void check_derivative_sums_vanish() {
    const double eps = eps_of(T(0));
    for (std::int64_t degree = 0; degree <= 5; ++degree) {
        const auto order = static_cast<std::size_t>(degree) + 1;
        const std::int64_t n_deriv = degree + 2;
        const auto rows = static_cast<std::size_t>(n_deriv) + 1;
        for (const auto& knots : knot_vector_family<T>(degree)) {
            const T domain_lo = knots[static_cast<std::size_t>(degree)];
            const T domain_hi = knots[knots.size() - static_cast<std::size_t>(degree) - 1];
            const auto pts = sample_domain<T>(domain_lo, domain_hi, 17);
            std::vector<std::int64_t> first_basis;
            const auto vals = tabulate_general_derivs<T>(std::span<const T>(knots), degree, false,
                                                         n_deriv, pts, first_basis);
            for (std::size_t j = 0; j < pts.size(); ++j) {
                for (std::int64_t k = 1; k <= n_deriv; ++k) {
                    const auto kk = static_cast<std::size_t>(k);
                    double scale = 0.0;
                    double sum = 0.0;
                    for (std::size_t i = 0; i < order; ++i) {
                        const double v = static_cast<double>(vals[(j * rows + kk) * order + i]);
                        sum += v;
                        scale = std::max(scale, std::abs(v));
                    }
                    const double bound = 8.0 * static_cast<double>(order) * eps * std::max(scale, 1.0);
                    PANTR_CHECK_MSG(std::abs(sum) <= bound,
                                    "degree " + std::to_string(degree) + " k=" + std::to_string(k) +
                                        " point idx " + std::to_string(j) + ": sum " +
                                        std::to_string(sum) + " scale " + std::to_string(scale));
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// (4) Rows above the degree are exactly zero
// --------------------------------------------------------------------------

template <class T>
void check_rows_above_degree_are_zero() {
    for (std::int64_t degree = 0; degree <= 5; ++degree) {
        const auto order = static_cast<std::size_t>(degree) + 1;
        const std::int64_t n_deriv = degree + 3;
        const auto rows = static_cast<std::size_t>(n_deriv) + 1;
        for (const auto& knots : knot_vector_family<T>(degree)) {
            const T domain_lo = knots[static_cast<std::size_t>(degree)];
            const T domain_hi = knots[knots.size() - static_cast<std::size_t>(degree) - 1];
            const auto pts = sample_domain<T>(domain_lo, domain_hi, 17);
            std::vector<std::int64_t> first_basis;
            const auto vals = tabulate_general_derivs<T>(std::span<const T>(knots), degree, false,
                                                         n_deriv, pts, first_basis);
            for (std::size_t j = 0; j < pts.size(); ++j) {
                for (std::size_t k = static_cast<std::size_t>(degree) + 1; k < rows; ++k) {
                    for (std::size_t i = 0; i < order; ++i) {
                        PANTR_CHECK(vals[(j * rows + k) * order + i] == T(0));
                    }
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// (5) Row 0 of basis_derivs_1d equals basis_funcs_1d, bit for bit
// --------------------------------------------------------------------------

template <class T>
void check_row0_matches_basis_funcs_bitwise() {
    for (std::int64_t degree = 0; degree <= 5; ++degree) {
        const auto order = static_cast<std::size_t>(degree) + 1;
        constexpr std::int64_t n_deriv = 3;
        const auto rows = static_cast<std::size_t>(n_deriv) + 1;
        for (const auto& knots : knot_vector_family<T>(degree)) {
            const T domain_lo = knots[static_cast<std::size_t>(degree)];
            const T domain_hi = knots[knots.size() - static_cast<std::size_t>(degree) - 1];
            const auto pts = sample_domain<T>(domain_lo, domain_hi, 17);

            std::vector<std::int64_t> first_funcs;
            const auto funcs =
                tabulate_general_basis<T>(std::span<const T>(knots), degree, false, pts, first_funcs);
            std::vector<std::int64_t> first_derivs;
            const auto derivs = tabulate_general_derivs<T>(std::span<const T>(knots), degree, false,
                                                           n_deriv, pts, first_derivs);

            for (std::size_t j = 0; j < pts.size(); ++j) {
                PANTR_CHECK(first_funcs[j] == first_derivs[j]);
                for (std::size_t i = 0; i < order; ++i) {
                    PANTR_CHECK_MSG(
                        funcs[j * order + i] == derivs[(j * rows + 0) * order + i],
                        "degree " + std::to_string(degree) + " point " + std::to_string(j) +
                            " basis " + std::to_string(i) +
                            ": row 0 of basis_derivs_1d must equal basis_funcs_1d bit for bit");
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// (6) find_span_and_first_basis on the boundaries
// --------------------------------------------------------------------------

template <class T>
void check_find_span_boundaries() {
    using pantr::bspline::find_span_and_first_basis;

    for (std::int64_t degree = 0; degree <= 5; ++degree) {
        for (const bool use_unclamped : {false, true}) {
            const std::vector<T> knots =
                use_unclamped ? unclamped_uniform_knots<T>(degree) : clamped_uniform_knots<T>(degree, 3);
            const auto size = static_cast<std::int64_t>(knots.size());
            const T domain_lo = knots[static_cast<std::size_t>(degree)];
            const T domain_hi = knots[static_cast<std::size_t>(size - degree - 1)];
            const std::int64_t last_span = size - degree - 2;
            const std::int64_t num_basis = size - degree - 1;
            const std::string tag =
                "degree " + std::to_string(degree) + (use_unclamped ? " (unclamped)" : " (clamped)");

            const auto check_point = [&](T point) {
                const auto located =
                    find_span_and_first_basis<T>(std::span<const T>(knots), degree, false, point);
                PANTR_CHECK_MSG(located.first_basis + degree + 1 <= num_basis,
                                tag + ": first_basis + degree + 1 <= num_basis");
                return located;
            };

            PANTR_CHECK_MSG(check_point(domain_lo).span == degree, tag + ": left endpoint");
            PANTR_CHECK_MSG(check_point(domain_hi).span == last_span,
                            tag + ": right endpoint must land in the last in-domain span");
            PANTR_CHECK_MSG(check_point(domain_lo - T(1)).span == degree,
                            tag + ": below-domain point clamps low");
            PANTR_CHECK_MSG(check_point(domain_hi + T(1)).span == last_span,
                            tag + ": above-domain point clamps high");
        }
    }
}

// --------------------------------------------------------------------------
// (7) tabulate_basis_1d's Bézier-like path against the general-knot path
// --------------------------------------------------------------------------

template <class T>
void check_tabulate_basis_1d_matches_general_knot_path() {
    const double eps = eps_of(T(0));
    constexpr std::int64_t degree = 2;
    const std::vector<std::pair<std::vector<T>, std::string>> cases = {
        {std::vector<T>{T(0), T(0), T(0), T(1), T(1), T(1)}, "unit domain"},
        {std::vector<T>{T(2), T(2), T(2), T(5), T(5), T(5)}, "non-unit domain"},
    };

    for (const auto& [knots, label] : cases) {
        const BsplineSpace1D<T> space(std::span<const T>(knots), degree, false,
                                      KnotSnapping::merge_near_duplicates);
        PANTR_CHECK_MSG(space.has_bezier_like_knots(), label);

        const std::array<T, 2> domain = space.domain();
        const auto pts = sample_domain<T>(domain[0], domain[1], 41);
        const auto order = static_cast<std::size_t>(degree) + 1;

        std::vector<T> bezier_path(pts.size() * order, T(0));
        std::vector<std::int64_t> first_bezier(pts.size(), 0);
        const span2d<T> bezier_view(bezier_path.data(), pts.size(), order);
        pantr::bspline::tabulate_basis_1d<T>(space, std::span<const T>(pts), bezier_view,
                                             std::span<std::int64_t>(first_bezier));

        std::vector<T> general_path(pts.size() * order, T(0));
        std::vector<std::int64_t> first_general(pts.size(), 0);
        const span2d<T> general_view(general_path.data(), pts.size(), order);
        pantr::bspline::basis_funcs_1d<T>(space.knots(), degree, space.periodic(),
                                          std::span<const T>(pts), general_view,
                                          std::span<std::int64_t>(first_general));

        // Two roads to the same degree-2 polynomials: each carries its own
        // O(degree * eps) forward error (test_bernstein.cpp's partition_bound
        // shape), and the Bézier path additionally forms the change of variable
        // (point - a) / (b - a), perturbing u by a relative eps that the
        // O(degree)-bounded polynomial derivative turns into another O(degree *
        // eps) term. The constant below is a safety margin over that estimate,
        // not a tight derivation.
        const double bound = 24.0 * static_cast<double>(degree + 1) * eps;

        for (std::size_t j = 0; j < pts.size(); ++j) {
            PANTR_CHECK_MSG(first_bezier[j] == first_general[j], label);
            double sum = 0.0;
            for (std::size_t i = 0; i < order; ++i) {
                const double a = static_cast<double>(bezier_path[j * order + i]);
                const double b = static_cast<double>(general_path[j * order + i]);
                sum += a;
                PANTR_CHECK_MSG(std::abs(a - b) <= bound,
                                label + " point idx " + std::to_string(j) + " basis " +
                                    std::to_string(i) + ": bezier " + std::to_string(a) +
                                    " general " + std::to_string(b));
            }
            PANTR_CHECK_MSG(std::abs(sum - 1.0) <= bound, label + " partition of unity");
        }
    }
}

// --------------------------------------------------------------------------
// (8) The chain rule on a non-unit domain
// --------------------------------------------------------------------------

template <class T>
void check_chain_rule_on_non_unit_domain() {
    const double eps = eps_of(T(0));
    constexpr std::int64_t degree = 2;
    constexpr std::int64_t n_deriv = 2;
    const std::vector<T> knots{T(2), T(2), T(2), T(5), T(5), T(5)};
    const BsplineSpace1D<T> space(std::span<const T>(knots), degree, false,
                                  KnotSnapping::merge_near_duplicates);
    PANTR_CHECK(space.has_bezier_like_knots());

    const std::array<T, 2> domain = space.domain();
    const auto pts = sample_domain<T>(domain[0], domain[1], 25);
    const auto order = static_cast<std::size_t>(degree) + 1;
    const auto rows = static_cast<std::size_t>(n_deriv) + 1;

    // The chain-rule path: the Bézier fast path, scaled by 1/(b-a)^k.
    std::vector<T> scaled(pts.size() * rows * order, T(0));
    std::vector<std::int64_t> first_scaled(pts.size(), 0);
    const span_nd<T, 3> scaled_view(scaled.data(), pts.size(), rows, order);
    pantr::bspline::tabulate_basis_derivatives_1d<T>(space, n_deriv, std::span<const T>(pts),
                                                     scaled_view, std::span<std::int64_t>(first_scaled));

    // The general-knot path: Cox-de Boor directly on the space's own non-unit knots,
    // where the scale is intrinsic to the knot differences rather than applied after
    // the fact.
    std::vector<T> general(pts.size() * rows * order, T(0));
    std::vector<std::int64_t> first_general(pts.size(), 0);
    const span_nd<T, 3> general_view(general.data(), pts.size(), rows, order);
    pantr::bspline::basis_derivs_1d<T>(space.knots(), degree, space.periodic(), n_deriv,
                                       std::span<const T>(pts), general_view,
                                       std::span<std::int64_t>(first_general));

    // The unscaled path: what the Bézier fast path computes BEFORE the chain-rule
    // factor is applied, built exactly as tabulate_basis_derivatives_1d's own body
    // does it, so as to have something to be wrong against.
    std::vector<T> reference(pts.size());
    const T span = domain[1] - domain[0];
    for (std::size_t i = 0; i < pts.size(); ++i) {
        reference[i] = (pts[i] - domain[0]) / span;
    }
    std::vector<T> unscaled(pts.size() * rows * order, T(0));
    const span_nd<T, 3> unscaled_view(unscaled.data(), pts.size(), rows, order);
    pantr::tabulate_bernstein_deriv_1d<T>(static_cast<int>(degree), static_cast<int>(n_deriv),
                                         std::span<const T>(reference), unscaled_view);

    for (std::int64_t k = 1; k <= n_deriv; ++k) {
        const auto kk = static_cast<std::size_t>(k);
        double scale = 0.0;
        for (std::size_t j = 0; j < pts.size(); ++j) {
            for (std::size_t i = 0; i < order; ++i) {
                scale = std::max(scale, std::abs(static_cast<double>(at(general_view, j, kk, i))));
            }
        }
        // Same shape as the derivative-sum bound above: each row is built by the
        // same A2.3 triangular recursion, so its forward error is O(order * eps)
        // relative to the row's own largest magnitude (Rule 2 -- an absolute
        // output needs an absolute reference, not 1).
        const double bound = 16.0 * static_cast<double>(order) * eps * std::max(scale, 1.0);

        double max_unscaled_diff = 0.0;
        for (std::size_t j = 0; j < pts.size(); ++j) {
            for (std::size_t i = 0; i < order; ++i) {
                const double s = static_cast<double>(at(scaled_view, j, kk, i));
                const double g = static_cast<double>(at(general_view, j, kk, i));
                const double u = static_cast<double>(at(unscaled_view, j, kk, i));
                PANTR_CHECK_MSG(std::abs(s - g) <= bound,
                                "k=" + std::to_string(k) + " point idx " + std::to_string(j) +
                                    " basis " + std::to_string(i) + ": scaled " + std::to_string(s) +
                                    " general " + std::to_string(g));
                max_unscaled_diff = std::max(max_unscaled_diff, std::abs(u - g));
            }
        }
        // Non-vacuous: a missing or inverted 1/(b-a)^k factor would leave the
        // unscaled values matching the general path instead of differing by 3^k (or
        // 3^-k) -- assert the gap is clearly larger than the matching tolerance.
        PANTR_CHECK_MSG(max_unscaled_diff > 10.0 * bound,
                        "k=" + std::to_string(k) +
                            ": unscaled and scaled results must differ for this test to be "
                            "non-vacuous, got gap " + std::to_string(max_unscaled_diff));
    }
}

// --------------------------------------------------------------------------
// (9) tabulate_bernstein_deriv_1d against closed forms one degree down
// --------------------------------------------------------------------------

template <class T>
void check_bernstein_deriv_against_closed_forms() {
    const double eps = eps_of(T(0));
    const auto pts = sample_domain<T>(T(0), T(1), 33);

    for (int n = 1; n <= 10; ++n) {
        const auto order = static_cast<std::size_t>(n) + 1;
        std::vector<T> deriv(pts.size() * 3 * order, T(0));
        const span_nd<T, 3> deriv_view(deriv.data(), pts.size(), std::size_t{3}, order);
        pantr::tabulate_bernstein_deriv_1d<T>(n, 2, std::span<const T>(pts), deriv_view);

        std::vector<T> lower(pts.size() * static_cast<std::size_t>(n));
        const span2d<T> lower_view(lower.data(), pts.size(), static_cast<std::size_t>(n));
        pantr::tabulate_bernstein_1d<T>(n - 1, std::span<const T>(pts), lower_view);

        // Partition of unity, and the derivative rows summing to zero -- the same
        // shapes as test_bernstein.cpp and check_derivative_sums_vanish above,
        // checked here against tabulate_bernstein_deriv_1d's own row 0, which is
        // the Cox-de Boor triangle rather than the ratio recurrence.
        for (std::size_t j = 0; j < pts.size(); ++j) {
            double sum0 = 0.0;
            for (std::size_t i = 0; i < order; ++i) {
                sum0 += static_cast<double>(at(deriv_view, j, std::size_t{0}, i));
            }
            PANTR_CHECK(std::abs(sum0 - 1.0) <= 8.0 * static_cast<double>(n) * eps);
        }
        for (const int k : {1, 2}) {
            const auto kk = static_cast<std::size_t>(k);
            double scale = 0.0;
            for (std::size_t j = 0; j < pts.size(); ++j) {
                for (std::size_t i = 0; i < order; ++i) {
                    scale = std::max(scale, std::abs(static_cast<double>(at(deriv_view, j, kk, i))));
                }
            }
            const double bound_sum = 16.0 * static_cast<double>(order) * eps * std::max(scale, 1.0);
            for (std::size_t j = 0; j < pts.size(); ++j) {
                double sum = 0.0;
                for (std::size_t i = 0; i < order; ++i) {
                    sum += static_cast<double>(at(deriv_view, j, kk, i));
                }
                PANTR_CHECK(std::abs(sum) <= bound_sum);
            }
        }

        // First derivative: d/du B_{i,n} = n * (B_{i-1,n-1} - B_{i,n-1}), computed
        // independently via tabulate_bernstein_1d at degree n-1 -- the ratio
        // recurrence, a different algorithm from the A2.3 specialisation under test.
        {
            double scale = 0.0;
            for (std::size_t j = 0; j < pts.size(); ++j) {
                for (std::size_t i = 0; i < order; ++i) {
                    scale =
                        std::max(scale, std::abs(static_cast<double>(at(deriv_view, j, std::size_t{1}, i))));
                }
            }
            const double bound = 16.0 * static_cast<double>(order) * eps * std::max(scale, 1.0);
            for (std::size_t j = 0; j < pts.size(); ++j) {
                for (std::size_t i = 0; i < order; ++i) {
                    const double b_im1 =
                        (i >= 1) ? static_cast<double>(at(lower_view, j, i - 1)) : 0.0;
                    const double b_i = (i <= static_cast<std::size_t>(n - 1))
                                          ? static_cast<double>(at(lower_view, j, i))
                                          : 0.0;
                    const double expected = static_cast<double>(n) * (b_im1 - b_i);
                    const double actual = static_cast<double>(at(deriv_view, j, std::size_t{1}, i));
                    PANTR_CHECK_MSG(std::abs(actual - expected) <= bound,
                                    "n=" + std::to_string(n) + " point " + std::to_string(j) +
                                        " basis " + std::to_string(i));
                }
            }
        }

        // Second derivative, one level further down:
        // d^2/du^2 B_{i,n} = n*(n-1) * (B_{i-2,n-2} - 2*B_{i-1,n-2} + B_{i,n-2}).
        if (n >= 2) {
            std::vector<T> lower2(pts.size() * static_cast<std::size_t>(n - 1));
            const span2d<T> lower2_view(lower2.data(), pts.size(), static_cast<std::size_t>(n - 1));
            pantr::tabulate_bernstein_1d<T>(n - 2, std::span<const T>(pts), lower2_view);

            double scale = 0.0;
            for (std::size_t j = 0; j < pts.size(); ++j) {
                for (std::size_t i = 0; i < order; ++i) {
                    scale =
                        std::max(scale, std::abs(static_cast<double>(at(deriv_view, j, std::size_t{2}, i))));
                }
            }
            const double bound = 24.0 * static_cast<double>(order) * eps * std::max(scale, 1.0);

            for (std::size_t j = 0; j < pts.size(); ++j) {
                for (std::size_t i = 0; i < order; ++i) {
                    const auto in_range = [&](std::int64_t idx) { return idx >= 0 && idx <= n - 2; };
                    const auto ii = static_cast<std::int64_t>(i);
                    const double b_im2 =
                        in_range(ii - 2)
                            ? static_cast<double>(at(lower2_view, j, static_cast<std::size_t>(ii - 2)))
                            : 0.0;
                    const double b_im1 =
                        in_range(ii - 1)
                            ? static_cast<double>(at(lower2_view, j, static_cast<std::size_t>(ii - 1)))
                            : 0.0;
                    const double b_i =
                        in_range(ii) ? static_cast<double>(at(lower2_view, j, static_cast<std::size_t>(ii)))
                                     : 0.0;
                    const double expected = static_cast<double>(n) * static_cast<double>(n - 1) *
                                           (b_im2 - 2.0 * b_im1 + b_i);
                    const double actual = static_cast<double>(at(deriv_view, j, std::size_t{2}, i));
                    PANTR_CHECK_MSG(std::abs(actual - expected) <= bound,
                                    "n=" + std::to_string(n) + " point " + std::to_string(j) +
                                        " basis " + std::to_string(i));
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// (10) Degree 0
// --------------------------------------------------------------------------

template <class T>
void check_degree_zero() {
    constexpr std::int64_t degree = 0;
    const std::vector<T> knots{T(0), T(1)};
    const std::vector<T> pts{T(0.25), T(0.75)};

    std::vector<std::int64_t> first_basis;
    const auto vals = tabulate_general_basis<T>(std::span<const T>(knots), degree, false, pts, first_basis);
    for (const T v : vals) {
        PANTR_CHECK(v == T(1));
    }

    constexpr std::int64_t n_deriv = 3;
    const auto rows = static_cast<std::size_t>(n_deriv) + 1;
    const auto deriv =
        tabulate_general_derivs<T>(std::span<const T>(knots), degree, false, n_deriv, pts, first_basis);
    for (std::size_t j = 0; j < pts.size(); ++j) {
        PANTR_CHECK(deriv[j * rows + 0] == T(1));
        for (std::size_t k = 1; k < rows; ++k) {
            PANTR_CHECK(deriv[j * rows + k] == T(0));
        }
    }
}

// --------------------------------------------------------------------------
// (11) pantr::core::wrapping_mul
// --------------------------------------------------------------------------

void check_wrapping_mul_agrees_where_no_overflow() {
    PANTR_CHECK(pantr::core::wrapping_mul(3, 4) == 12);
    PANTR_CHECK(pantr::core::wrapping_mul(-3, 4) == -12);
    PANTR_CHECK(pantr::core::wrapping_mul(0, 123456789) == 0);
    constexpr std::int64_t big = 1'000'000'000;
    PANTR_CHECK(pantr::core::wrapping_mul(big, big) == big * big);
}

/// A 128-bit unsigned value as two 64-bit words, `hi * 2^64 + lo`.
///
/// `__int128` would be the obvious spelling and is not used: `-Wpedantic` (part of
/// this project's warning set) rejects it as a non-standard extension, so the
/// independent computation below is built from portable `std::uint64_t` operations
/// instead -- a schoolbook 32-bit-limb widening multiply, which is a different
/// route to the same 128-bit product than anything `wrapping_mul` does.
struct Wide128 {
    std::uint64_t hi = 0;
    std::uint64_t lo = 0;
};

/// The full 128-bit product of two 64-bit values, via 32-bit limb decomposition.
[[nodiscard]] Wide128 widening_mul(std::uint64_t a, std::uint64_t b) {
    const std::uint64_t a_lo = a & 0xFFFFFFFFULL;
    const std::uint64_t a_hi = a >> 32;
    const std::uint64_t b_lo = b & 0xFFFFFFFFULL;
    const std::uint64_t b_hi = b >> 32;

    const std::uint64_t t0 = a_lo * b_lo;
    const std::uint64_t t1 = a_hi * b_lo;
    const std::uint64_t t2 = a_lo * b_hi;
    const std::uint64_t t3 = a_hi * b_hi;

    const std::uint64_t mid = (t0 >> 32) + (t1 & 0xFFFFFFFFULL) + (t2 & 0xFFFFFFFFULL);
    Wide128 result;
    result.lo = (t0 & 0xFFFFFFFFULL) | (mid << 32);
    result.hi = t3 + (t1 >> 32) + (t2 >> 32) + (mid >> 32);
    return result;
}

/// The specific wrap `tabulate.hpp`'s factorial scaling produces at degree 21,
/// derivative order 19: `fac` starts at `degree` and is repeatedly updated by
/// `wrapping_mul(fac, degree - k)`, so the value used to scale row 19 is
/// `degree! / (degree - 19)! = 21! / 2!`, which exceeds `2^64` and wraps.
///
/// The expected wrap is computed here independently of `wrapping_mul`, by carrying
/// the exact product of `3 * 4 * ... * 21` in a 128-bit accumulator that never
/// overflows (the true value is under `2^67`) and then reading its low 64 bits --
/// which is exactly what "reduction modulo 2^64" means, reached by a completely
/// different route (32-bit limb multiplication) than `wrapping_mul`'s single 64x64
/// multiply relies on the hardware to truncate.
void check_wrapping_mul_reproduces_the_measured_wrap() {
    Wide128 exact{0, 1};
    for (std::uint64_t i = 3; i <= 21; ++i) {
        const Wide128 term = widening_mul(exact.lo, i);
        // `exact.hi` stays 0 or 1 throughout this particular product (it is under
        // 2^67), and `i <= 21`, so `exact.hi * i` cannot itself overflow 64 bits.
        exact.hi = term.hi + exact.hi * i;
        exact.lo = term.lo;
    }
    const auto expected = static_cast<std::int64_t>(exact.lo);

    constexpr std::int64_t degree = 21;
    std::int64_t fac = degree;
    for (std::int64_t k = 1; k <= 18; ++k) {
        fac = pantr::core::wrapping_mul(fac, degree - k);
    }
    PANTR_CHECK_MSG(fac == expected,
                    "wrapping_mul disagrees with the independently computed wrap of 21!/2!: got " +
                        std::to_string(fac) + " want " + std::to_string(expected));
}

// --------------------------------------------------------------------------
// (12) A periodic space: the other arm of the first-basis branch
// --------------------------------------------------------------------------

/// A periodic space is the only input reaching `find_span_and_first_basis`'s other
/// branch, which keeps the **unclamped** first-basis index -- the evaluation loop wraps
/// it modulo the control points -- where the non-periodic arm caps it at
/// `num_basis - order`. Nothing else in this file or in `tests/parity/` constructed
/// one, so that branch shipped unexercised, and a port that applied the non-periodic
/// clamp regardless would have passed every other check here.
///
/// The properties asserted are the ones that do not depend on periodicity: Cox-de Boor
/// is unchanged, only the reported index differs, so the `degree + 1` local values
/// still form a convex partition of unity and the derivative rows still sum to zero.
/// The last assertion is the discriminating one -- it fails if the index was clamped.
template <class T>
void check_periodic_space() {
    using pantr::bspline::basis_derivs_1d;
    using pantr::bspline::basis_funcs_1d;

    const double eps = eps_of(T(0));
    constexpr std::int64_t degree = 2;
    const std::vector<T> knots = {T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7)};
    const BsplineSpace1D<T> space(std::span<const T>(knots), degree, true,
                                  KnotSnapping::merge_near_duplicates);
    PANTR_CHECK(space.periodic());

    const std::array<T, 2> domain = space.domain();
    const auto pts = sample_domain<T>(domain[0], domain[1], 17);
    const auto order = static_cast<std::size_t>(degree) + 1;
    const std::int64_t num_basis = space.num_basis();

    std::vector<T> values(pts.size() * order);
    std::vector<std::int64_t> first_basis(pts.size());
    basis_funcs_1d<T>(std::span<const T>(knots), degree, true, std::span<const T>(pts),
                      span2d<T>(values.data(), pts.size(), order),
                      std::span<std::int64_t>(first_basis));

    const double bound = (2.0 * static_cast<double>(degree) + 1.0) * (eps / 2.0);
    std::int64_t largest_index = -1;
    for (std::size_t j = 0; j < pts.size(); ++j) {
        double sum = 0.0;
        for (std::size_t i = 0; i < order; ++i) {
            const double value = static_cast<double>(values[j * order + i]);
            PANTR_CHECK_MSG(value >= 0.0, "periodic: non-negativity");
            sum += value;
        }
        PANTR_CHECK_MSG(std::abs(sum - 1.0) <= bound, "periodic: partition of unity");
        largest_index = std::max(largest_index, first_basis[j]);
    }

    // Derivatives on the same space: the rows still sum to zero.
    constexpr std::int64_t n_deriv = 3;
    const auto rows = static_cast<std::size_t>(n_deriv) + 1;
    std::vector<T> block(pts.size() * rows * order);
    std::vector<std::int64_t> deriv_first(pts.size());
    basis_derivs_1d<T>(std::span<const T>(knots), degree, true, n_deriv,
                       std::span<const T>(pts), span_nd<T, 3>(block.data(), pts.size(), rows, order),
                       std::span<std::int64_t>(deriv_first));

    for (std::size_t j = 0; j < pts.size(); ++j) {
        PANTR_CHECK_MSG(deriv_first[j] == first_basis[j],
                        "periodic: the two kernels must report the same index");
        for (std::size_t k = 1; k < rows; ++k) {
            double sum = 0.0;
            double scale = 0.0;
            for (std::size_t i = 0; i < order; ++i) {
                const double value = static_cast<double>(block[(j * rows + k) * order + i]);
                sum += value;
                scale = std::max(scale, std::abs(value));
            }
            const double row_bound = 8.0 * static_cast<double>(order) * eps * std::max(scale, 1.0);
            PANTR_CHECK_MSG(std::abs(sum) <= row_bound, "periodic: derivative row sums to zero");
        }
    }

    // The discriminating assertion. A non-periodic space caps `first_basis` at
    // `num_basis - order`; a periodic one must not, or the caller's modulo has nothing
    // to wrap. If this fails, the periodic branch is not being taken.
    PANTR_CHECK_MSG(largest_index > num_basis - static_cast<std::int64_t>(order),
                    "periodic: the first-basis index must exceed the non-periodic clamp");
}


}  // namespace

int main() {
    check_partition_of_unity_general_knots<double>();
    check_partition_of_unity_general_knots<float>();
    check_non_negativity<double>();
    check_non_negativity<float>();
    check_derivative_sums_vanish<double>();
    check_derivative_sums_vanish<float>();
    check_rows_above_degree_are_zero<double>();
    check_rows_above_degree_are_zero<float>();
    check_row0_matches_basis_funcs_bitwise<double>();
    check_row0_matches_basis_funcs_bitwise<float>();
    check_find_span_boundaries<double>();
    check_find_span_boundaries<float>();
    check_tabulate_basis_1d_matches_general_knot_path<double>();
    check_tabulate_basis_1d_matches_general_knot_path<float>();
    check_chain_rule_on_non_unit_domain<double>();
    check_chain_rule_on_non_unit_domain<float>();
    check_bernstein_deriv_against_closed_forms<double>();
    check_bernstein_deriv_against_closed_forms<float>();
    check_degree_zero<double>();
    check_degree_zero<float>();
    check_periodic_space<double>();
    check_periodic_space<float>();
    check_wrapping_mul_agrees_where_no_overflow();
    check_wrapping_mul_reproduces_the_measured_wrap();
    return pantr::test::summary("test_bspline_tabulation");
}
