/// \file
/// Mathematical properties of the Bézier arithmetic kernels.
///
/// ## What this file is for, given that parity is already exhaustive
///
/// `tests/parity/test_bezier_arithmetic.py` compares every kernel here against
/// its Numba oracle bit for bit, over 519 cases. What it cannot catch is a bug
/// the **oracle shares**: a port faithful to a wrong formula passes parity
/// perfectly. So nothing here compares the two backends. Every check below is
/// against a property the mathematics fixes, or against an oracle computed a
/// different way.
///
/// ## The exact cases, and why they are exact
///
/// Most of these assert bit equality rather than a tolerance, and each has a
/// reason it can:
///
///  - **The endpoints.** `B(0) = c_0` and `B(1) = c_p` are structural, not
///    approximate. At `u = 0` the seed `(1-u)^p` is exactly 1 and the ratio
///    `u/(1-u)` is exactly 0, so every later term vanishes; `u = 1` is
///    short-circuited. Elevation carries `bezalfs[0][0] = bezalfs[p][ph] = 1`
///    exactly, and a split's first left point and last right point are copies.
///  - **`restrict` over the full domain, on finite input.** With `lower = 0` and
///    `upper = 1` the branch picks the left pass at `tau = 1`, whose step is
///    `d[j]*1 + d[j-1]*0`, then the right pass at `tau2 = 0/1 = 0`, whose step is
///    `d[j]*1 + d[j+1]*0`. Both return their input exactly for every finite value,
///    subnormals included. **They do not for a signed zero or an infinity**, since
///    `-0.0 + 0.0` is `+0.0` and `inf * 0` is a NaN that then poisons the column.
///    An earlier draft of this file claimed the identity held for those too; the
///    test refuted it, the oracle was measured to behave identically, and both
///    exceptions are now pinned rather than excluded.
///  - **de Casteljau at one half, on integers.** Each step is
///    `0.5*d[i] + 0.5*d[i+1]`, which for integer operands is `(d[i]+d[i+1])/2`, a
///    dyadic rational held exactly while the numerator stays under `2^53`. The
///    independent oracle is `sum_i C(p,i) c_i / 2^p`, assembled in exact `int64`
///    and divided by a power of two, which is also exact. Two different
///    algorithms, both exact, so the comparison is an equality with no tolerance
///    to argue about. This is the one check here that would catch a wrong
///    *formula* rather than a wrong endpoint.
///
/// ## The bounded cases, and two claims a review refuted
///
/// `split_reconstructs` and `restrict_agrees` carry real bounds, `8p` and `16p`
/// times `eps * max|c|`, measured at 31-45x margin to degree 55 across 600
/// decades. One caveat their derivation omitted: they are computed in `eps` where
/// the argument is written in `u = eps/2`, an unacknowledged further factor of two.
/// `cpp/tests/test_cardinal_bspline.cpp` states that substitution explicitly and
/// this file now does too.
///
/// **Two other tolerances here were not bounds at all, and both are gone.**
///
/// `product_with_unit` claimed to be "exact only when the binomial is a power of
/// two". False: exactness holds whenever `a_k * C(p,k)` is representable, which for
/// integers in `[-9, 9]` is every `p <= 53`. All five swept degrees gave error
/// exactly zero at binomials 3, 45, 252 and 184756, none a power of two, so the
/// tolerance had never been consumed. It is an equality now, with its envelope
/// stated.
///
/// `derivative_at_the_endpoints` justified `p * eps * |want|` by Sterbenz exactness
/// of `c_1 - c_0`. **The kernel never subtracts two control points** -- `ctrl`
/// enters only as `+= weight * ctrl` -- so the mechanism does not exist. The bound
/// was also vacuous on its own data (error exactly zero at every swept degree) and
/// false off it: with `c = 1e6 + i/3` at unit scale it is exceeded 1.7e5 times at
/// `p = 5`. The scaling was the wrong quantity, because A2.3's roundings act on
/// contraction terms of size `sum_j |B'_{j,p}| * max|c| <= 2p * max|c|`, not on
/// `|want| = p * |dc|`. It is now an equality on the exact case and a derived
/// `2 p^2 eps max|c|` on a case that actually rounds.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/bezier/kernels_1d.hpp"
#include "pantr/core/binomial.hpp"
#include "pantr/core/mdspan.hpp"

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

/// A control net laid out row-major as `(degree + 1, rank)`.
struct Net {
    std::vector<double> data;
    std::size_t rows = 0;
    std::size_t cols = 0;

    [[nodiscard]] pantr::span2d<const double> view() const {
        return pantr::span2d<const double>(data.data(), rows, cols);
    }
};

/// Build a net from a flat row-major list.
Net net(std::vector<double> values, std::size_t rows, std::size_t cols) {
    return Net{std::move(values), rows, cols};
}

/// A deterministic net of small integers, so exact arithmetic stays available.
Net integer_net(std::size_t rows, std::size_t cols) {
    std::vector<double> values(rows * cols);
    // A fixed, non-monotone pattern in [-9, 9]: monotone data hides an index slip
    // in a de Casteljau triangle, because neighbouring values are then close.
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto k = static_cast<std::int64_t>(i);
        values[i] = static_cast<double>(((k * 7 + 3) % 19) - 9);
    }
    return net(std::move(values), rows, cols);
}

/// `sum_i C(p, i) c_i / 2^p`, the value of a Bézier at one half, exactly.
///
/// Assembled in `int64` from integer control points, so it is an independent
/// oracle rather than the same recurrence run twice.
double value_at_half_exact(const Net& n, std::size_t col) {
    const auto p = static_cast<int>(n.rows) - 1;
    std::int64_t total = 0;
    for (int i = 0; i <= p; ++i) {
        const auto c = static_cast<std::int64_t>(
            n.data[static_cast<std::size_t>(i) * n.cols + col]);
        total += static_cast<std::int64_t>(pantr::core::bincoeff(p, i)) * c;
    }
    return std::ldexp(static_cast<double>(total), -p);
}

void test_slice_at_one_half_matches_an_exact_binomial_sum() {
    // Degree 40 keeps `sum |C(40, i) * 9|` near 9 * 2^40, far under 2^53, so both
    // sides stay in exact integer arithmetic.
    for (std::size_t degree : {std::size_t{0}, std::size_t{1}, std::size_t{2},
                               std::size_t{5}, std::size_t{13}, std::size_t{40}}) {
        const Net n = integer_net(degree + 1, 3);
        std::vector<double> got(3);
        pantr::bezier::slice_bezier_1d<double>(n.view(), 0.5, std::span<double>(got));
        for (std::size_t col = 0; col < 3; ++col) {
            const double want = value_at_half_exact(n, col);
            PANTR_CHECK_MSG(got[col] == want,
                            "degree " + std::to_string(degree) + " column " +
                                std::to_string(col) + ": " + std::to_string(got[col]) +
                                " against the exact " + std::to_string(want));
        }
    }
}

void test_evaluate_reproduces_the_end_control_points() {
    for (std::size_t degree : {std::size_t{0}, std::size_t{1}, std::size_t{4},
                               std::size_t{17}, std::size_t{30}}) {
        const Net n = integer_net(degree + 1, 2);
        const std::vector<double> pts{0.0, 1.0};
        std::vector<double> got(4);
        pantr::bezier::evaluate_bezier_1d<double>(n.view(), std::span<const double>(pts),
                                                  pantr::span2d<double>(got.data(), 2, 2));
        for (std::size_t col = 0; col < 2; ++col) {
            PANTR_CHECK_MSG(got[col] == n.data[col],
                            "degree " + std::to_string(degree) + ": B(0) is not c_0");
            PANTR_CHECK_MSG(got[2 + col] == n.data[degree * 2 + col],
                            "degree " + std::to_string(degree) + ": B(1) is not c_p");
        }
    }
}

void test_slice_reproduces_the_end_control_points() {
    for (std::size_t degree : {std::size_t{0}, std::size_t{1}, std::size_t{9}}) {
        const Net n = integer_net(degree + 1, 2);
        std::vector<double> at_zero(2);
        std::vector<double> at_one(2);
        pantr::bezier::slice_bezier_1d<double>(n.view(), 0.0, std::span<double>(at_zero));
        pantr::bezier::slice_bezier_1d<double>(n.view(), 1.0, std::span<double>(at_one));
        for (std::size_t col = 0; col < 2; ++col) {
            PANTR_CHECK(at_zero[col] == n.data[col]);
            PANTR_CHECK(at_one[col] == n.data[degree * 2 + col]);
        }
    }
}

void test_split_preserves_the_endpoints_and_shares_a_point() {
    // The shared point is what makes a split a partition rather than two
    // approximations: the last left control point and the first right one are the
    // same de Casteljau vertex, so they must be identical bit for bit.
    for (std::size_t degree : {std::size_t{1}, std::size_t{3}, std::size_t{11}}) {
        for (double u : {0.125, 0.5, 0.9}) {
            const Net n = integer_net(degree + 1, 2);
            std::vector<double> left((degree + 1) * 2);
            std::vector<double> right((degree + 1) * 2);
            pantr::bezier::split_bezier_1d<double>(
                n.view(), u, pantr::span2d<double>(left.data(), degree + 1, 2),
                pantr::span2d<double>(right.data(), degree + 1, 2));
            for (std::size_t col = 0; col < 2; ++col) {
                PANTR_CHECK_MSG(left[col] == n.data[col], "left half moved the start point");
                PANTR_CHECK_MSG(right[degree * 2 + col] == n.data[degree * 2 + col],
                                "right half moved the end point");
                PANTR_CHECK_MSG(left[degree * 2 + col] == right[col],
                                "the two halves do not share the split point at degree " +
                                    std::to_string(degree));
            }
        }
    }
}

void test_elevate_preserves_the_endpoints() {
    for (int degree : {0, 1, 4, 12}) {
        for (int inc : {1, 2, 7}) {
            const Net n = integer_net(static_cast<std::size_t>(degree) + 1, 2);
            const auto rows = static_cast<std::size_t>(degree + inc + 1);
            std::vector<double> out(rows * 2);
            pantr::bezier::degree_elevate_bezier_1d<double>(
                degree, n.view(), inc, pantr::span2d<double>(out.data(), rows, 2));
            for (std::size_t col = 0; col < 2; ++col) {
                PANTR_CHECK_MSG(out[col] == n.data[col], "elevation moved the start point");
                PANTR_CHECK_MSG(out[(rows - 1) * 2 + col] ==
                                    n.data[static_cast<std::size_t>(degree) * 2 + col],
                                "elevation moved the end point");
            }
        }
    }
}

void test_elevating_a_linear_gives_the_midpoint() {
    // Degree 1 to 2 has one interior coefficient, `(a + b) / 2`, and for integer
    // endpoints that is exact. A wrong `bezalfs` entry shows up here as a value
    // that is not the midpoint at all, rather than as a last-bit difference.
    const Net n = net({4.0, -2.0, 10.0, 6.0}, 2, 2);
    std::vector<double> out(6);
    pantr::bezier::degree_elevate_bezier_1d<double>(1, n.view(), 1,
                                                    pantr::span2d<double>(out.data(), 3, 2));
    PANTR_CHECK(out[2] == 7.0);
    PANTR_CHECK(out[3] == 2.0);
}

void test_restrict_to_the_full_domain_is_the_identity_on_finite_input() {
    // Exact on finite input, and the qualifier is load-bearing. Both passes reduce
    // to `d[j] * 1 + d[neighbour] * 0`, which returns `d[j]` unchanged for every
    // finite value including subnormals -- but `inf * 0` is a NaN, so one infinity
    // anywhere in a column poisons the whole column, and `-0.0 + 0.0` is `+0.0`, so
    // a signed zero does not survive either.
    //
    // Neither of those is a defect of the port: measured, the numba oracle does
    // exactly the same, and the parity suite covers the agreement. They are
    // recorded below rather than excluded, because "restrict over the full domain
    // returns its input" is the kind of sentence that ends up in a docstring, and
    // it is false as stated.
    Net n = integer_net(8, 3);
    n.data[2] = std::numeric_limits<double>::denorm_min();
    n.data[5] = std::numeric_limits<double>::max();
    std::vector<double> out(8 * 3);
    pantr::bezier::restrict_bezier_1d<double>(n.view(), 0.0, 1.0,
                                              pantr::span2d<double>(out.data(), 8, 3));
    for (std::size_t i = 0; i < out.size(); ++i) {
        PANTR_CHECK_MSG(out[i] == n.data[i],
                        "restrict to [0, 1] changed entry " + std::to_string(i));
    }
}

void test_restrict_over_the_full_domain_loses_a_signed_zero_and_an_infinity() {
    // The two documented exceptions to the identity above, pinned so that a later
    // rewrite of the pass cannot change them silently in one backend only.
    Net signed_zero = integer_net(4, 1);
    signed_zero.data[1] = -0.0;
    std::vector<double> out(4);
    pantr::bezier::restrict_bezier_1d<double>(signed_zero.view(), 0.0, 1.0,
                                              pantr::span2d<double>(out.data(), 4, 1));
    PANTR_CHECK_MSG(out[1] == 0.0 && !std::signbit(out[1]),
                    "a signed zero is expected to come back positive, since the pass "
                    "forms -0.0 + 0.0");

    Net infinite = integer_net(4, 1);
    infinite.data[0] = std::numeric_limits<double>::infinity();
    pantr::bezier::restrict_bezier_1d<double>(infinite.view(), 0.0, 1.0,
                                              pantr::span2d<double>(out.data(), 4, 1));
    bool any_nan = false;
    for (double v : out) {
        any_nan = any_nan || std::isnan(v);
    }
    PANTR_CHECK_MSG(any_nan, "an infinity is expected to reach the output as a NaN, "
                             "since the pass forms inf * 0");
}

void test_product_with_the_unit_polynomial_reproduces_its_input() {
    // `c_k = (a_k * C(p,k)) / C(p,k)`, and this is an EQUALITY, not a tolerance.
    // Both operations are exact whenever `a_k * C(p,k)` is representable: the
    // multiply lands on an integer below 2^53 and the divide undoes it exactly. For
    // `|a| <= 9` that holds through `p = 53`, five orders past the largest swept
    // degree (1.3e6 against 9.0e15). An earlier version asserted `2 * eps * |want|`
    // and justified it by the binomial being a power of two, which is false and
    // which made the assertion vacuous: the error is zero at binomials 3, 45, 252
    // and 184756. A wrong scaling now fails outright rather than inside a tolerance
    // nothing consumed.
    for (int p : {0, 1, 3, 10, 20}) {
        const Net a = integer_net(static_cast<std::size_t>(p) + 1, 1);
        const std::vector<double> unit{1.0};
        std::vector<double> out(static_cast<std::size_t>(p) + 1);
        pantr::bezier::scalar_bernstein_product_1d<double>(
            std::span<const double>(a.data), std::span<const double>(unit),
            std::span<double>(out));
        for (std::size_t k = 0; k < out.size(); ++k) {
            PANTR_CHECK_MSG(out[k] == a.data[k],
                            "degree " + std::to_string(p) + " coefficient " +
                                std::to_string(k) + ": " + std::to_string(out[k]) +
                                " against the exact " + std::to_string(a.data[k]));
        }
    }
}

void test_the_first_derivative_at_the_endpoints_is_exact_on_integers() {
    // `B'(0) = p (c_1 - c_0)` and `B'(1) = p (c_p - c_{p-1})`, and on this data it
    // is an EQUALITY.
    //
    // At `s = 0` and `s = 1` every `ndu` entry collapses to exactly 0 or 1, A2.3's
    // `a`-recursion forms differences of exact integers, and the falling factorial
    // is exact integer arithmetic, so the contraction is a sum of exactly
    // representable integers while `2^{n_deriv} p (p+1) max|c| < 2^53` -- at most
    // 1080 for the degrees below.
    //
    // The version this replaces asserted `p * eps * |want|` and justified it by
    // Sterbenz exactness of `c_1 - c_0`. That mechanism does not exist: the kernel
    // never subtracts two control points, only accumulates `weight * ctrl`. The
    // assertion was also never exercised, since the error was exactly zero at every
    // swept degree, so the wrong bound sat behind a passing test. See
    // `test_the_first_derivative_under_cancellation` below for the case that does
    // round, and the bound that actually holds there.
    for (int p : {1, 2, 5, 9}) {
        const auto rows = static_cast<std::size_t>(p) + 1;
        std::vector<double> values(rows);
        for (std::size_t i = 0; i < rows; ++i) {
            values[i] = 3.0 + static_cast<double>(i);
        }
        const Net n = net(values, rows, 1);
        const std::vector<double> pts{0.0, 1.0};
        std::vector<double> out(2 * 2 * 1);
        pantr::bezier::evaluate_bezier_deriv_1d<double>(
            n.view(), std::span<const double>(pts), 1,
            pantr::span_nd<double, 3>(out.data(), 2, 2, 1));
        const double want = static_cast<double>(p) * 1.0;  // every difference is 1
        PANTR_CHECK_MSG(out[1] == want, "degree " + std::to_string(p) + ": B'(0) is " +
                                            std::to_string(out[1]) + ", expected exactly " +
                                            std::to_string(want));
        PANTR_CHECK_MSG(out[3] == want, "degree " + std::to_string(p) + ": B'(1) is " +
                                            std::to_string(out[3]) + ", expected exactly " +
                                            std::to_string(want));
    }
}

void test_the_first_derivative_under_cancellation() {
    // The case the exact test above cannot reach, and the bound that holds there.
    //
    // `c_i = 1e6 + i/3` makes the answer `p/3` while every control point is 1e6, so
    // the derivative is a difference of large numbers and the roundings act on the
    // contraction terms rather than on the result. A2.3's derivative basis satisfies
    // `sum_j |B'_{j,p}(s)| <= 2p`, and the contraction commits at most one rounding
    // per stage of the recursion, so the absolute error is bounded by
    //
    //     |err|  <=  2 * p^2 * eps * max|c|                                    (1)
    //
    // which is what the tolerance below is. The bound it replaces, `p * eps *
    // |want|`, is scaled by the RESULT rather than by the data, and this data is
    // exactly where those differ: it is exceeded 1.7e5 times at `p = 5`.
    for (int p : {2, 5, 9, 25}) {
        const auto rows = static_cast<std::size_t>(p) + 1;
        std::vector<double> values(rows);
        double worst_c = 0.0;
        for (std::size_t i = 0; i < rows; ++i) {
            values[i] = 1.0e6 + static_cast<double>(i) / 3.0;
            worst_c = std::max(worst_c, std::abs(values[i]));
        }
        const Net n = net(values, rows, 1);
        const std::vector<double> pts{0.0, 1.0};
        std::vector<double> out(2 * 2 * 1);
        pantr::bezier::evaluate_bezier_deriv_1d<double>(
            n.view(), std::span<const double>(pts), 1,
            pantr::span_nd<double, 3>(out.data(), 2, 2, 1));
        const double want = static_cast<double>(p) / 3.0;
        const double pd = static_cast<double>(p);
        const double slack = 2.0 * pd * pd * kEps * worst_c;
        PANTR_CHECK_MSG(std::abs(out[1] - want) <= slack,
                        "degree " + std::to_string(p) + ": B'(0) error " +
                            std::to_string(std::abs(out[1] - want)) + " exceeds " +
                            std::to_string(slack));
        PANTR_CHECK_MSG(std::abs(out[3] - want) <= slack,
                        "degree " + std::to_string(p) + ": B'(1) error " +
                            std::to_string(std::abs(out[3] - want)) + " exceeds " +
                            std::to_string(slack));
    }
}

void test_a_split_reconstructs_the_original_curve() {
    // The left half at parameter `s` is the whole curve at `u * s`. This is the one
    // check that ties the split to the evaluation, so a triangle that is internally
    // consistent but wrong fails here rather than nowhere.
    //
    // The bound: de Casteljau on `[0, 1]` is a sequence of convex combinations, so
    // after `p` stages the accumulated error is at most `p` roundings of a quantity
    // bounded by `max |c_i|`. Evaluation adds its own `3p` roundings of the same
    // scale, per the Bernstein recurrence's three multiplications per step, and the
    // constant below rounds `4p` up to `8p` to cover the `pow` seed's libm error.
    //
    // Written in `eps` while the argument is in `u = eps/2`, so the coded constant
    // is a further factor of two loose. Stated rather than removed, as
    // `cpp/tests/test_cardinal_bspline.cpp` states it: measured margin over the
    // observed worst is 31-45x to degree 55 across 600 decades, so the slack is not
    // what makes the test pass.
    const std::size_t degree = 9;
    const Net n = integer_net(degree + 1, 1);
    double worst_c = 0.0;
    for (double v : n.data) {
        worst_c = std::max(worst_c, std::abs(v));
    }

    for (double u : {0.25, 0.5, 0.75}) {
        std::vector<double> left(degree + 1);
        std::vector<double> right(degree + 1);
        pantr::bezier::split_bezier_1d<double>(n.view(), u,
                                               pantr::span2d<double>(left.data(), degree + 1, 1),
                                               pantr::span2d<double>(right.data(), degree + 1, 1));
        const pantr::span2d<const double> left_view(left.data(), degree + 1, 1);

        for (double s : {0.0, 0.3, 0.6, 1.0}) {
            const std::vector<double> half_pt{s};
            const std::vector<double> whole_pt{u * s};
            double from_half = 0.0;
            double from_whole = 0.0;
            pantr::bezier::evaluate_bezier_1d<double>(left_view, std::span<const double>(half_pt),
                                                      pantr::span2d<double>(&from_half, 1, 1));
            pantr::bezier::evaluate_bezier_1d<double>(n.view(), std::span<const double>(whole_pt),
                                                      pantr::span2d<double>(&from_whole, 1, 1));
            const double slack = 8.0 * static_cast<double>(degree) * kEps * worst_c;
            PANTR_CHECK_MSG(std::abs(from_half - from_whole) <= slack,
                            "split at " + std::to_string(u) + ", left half at " +
                                std::to_string(s) + ": " + std::to_string(from_half) +
                                " against " + std::to_string(from_whole));
        }
    }
}

void test_restrict_agrees_with_evaluation_on_the_subinterval() {
    // The restricted curve at `s` is the original at `lower + s * (upper - lower)`.
    // Both branches of the pass-ordering test are exercised, which is what makes
    // this more than a repeat of the split check above.
    const std::size_t degree = 7;
    const Net n = integer_net(degree + 1, 1);
    double worst_c = 0.0;
    for (double v : n.data) {
        worst_c = std::max(worst_c, std::abs(v));
    }

    const double bounds[][2] = {{0.1, 0.9}, {0.0, 0.4}, {0.6, 1.0}, {0.25, 0.75}};
    for (const auto& b : bounds) {
        std::vector<double> sub(degree + 1);
        pantr::bezier::restrict_bezier_1d<double>(n.view(), b[0], b[1],
                                                  pantr::span2d<double>(sub.data(), degree + 1, 1));
        const pantr::span2d<const double> sub_view(sub.data(), degree + 1, 1);

        for (double s : {0.0, 0.5, 1.0}) {
            const std::vector<double> sub_pt{s};
            const std::vector<double> whole_pt{b[0] + s * (b[1] - b[0])};
            double from_sub = 0.0;
            double from_whole = 0.0;
            pantr::bezier::evaluate_bezier_1d<double>(sub_view, std::span<const double>(sub_pt),
                                                      pantr::span2d<double>(&from_sub, 1, 1));
            pantr::bezier::evaluate_bezier_1d<double>(n.view(), std::span<const double>(whole_pt),
                                                      pantr::span2d<double>(&from_whole, 1, 1));
            // Two de Casteljau passes rather than one, so twice the split's budget.
            const double slack = 16.0 * static_cast<double>(degree) * kEps * worst_c;
            PANTR_CHECK_MSG(std::abs(from_sub - from_whole) <= slack,
                            "restrict to [" + std::to_string(b[0]) + ", " +
                                std::to_string(b[1]) + "] at " + std::to_string(s) + ": " +
                                std::to_string(from_sub) + " against " +
                                std::to_string(from_whole));
        }
    }
}

void test_float32_runs_the_same_structural_identities() {
    // The exact endpoint identities hold at every width, so they are the cheapest
    // way to instantiate the kernels on `float` and find out that they do.
    const std::vector<float> values{4.0F, -2.0F, 10.0F, 6.0F, 1.0F, 0.0F};
    const pantr::span2d<const float> view(values.data(), 3, 2);
    std::vector<float> got(2);
    pantr::bezier::slice_bezier_1d<float>(view, 0.0, std::span<float>(got));
    PANTR_CHECK(got[0] == 4.0F && got[1] == -2.0F);
    pantr::bezier::slice_bezier_1d<float>(view, 1.0, std::span<float>(got));
    PANTR_CHECK(got[0] == 1.0F && got[1] == 0.0F);

    std::vector<float> elevated(4 * 2);
    pantr::bezier::degree_elevate_bezier_1d<float>(2, view, 1,
                                                   pantr::span2d<float>(elevated.data(), 4, 2));
    PANTR_CHECK(elevated[0] == 4.0F && elevated[1] == -2.0F);
    PANTR_CHECK(elevated[6] == 1.0F && elevated[7] == 0.0F);
}

}  // namespace

int main() {
    test_slice_at_one_half_matches_an_exact_binomial_sum();
    test_evaluate_reproduces_the_end_control_points();
    test_slice_reproduces_the_end_control_points();
    test_split_preserves_the_endpoints_and_shares_a_point();
    test_elevate_preserves_the_endpoints();
    test_elevating_a_linear_gives_the_midpoint();
    test_restrict_to_the_full_domain_is_the_identity_on_finite_input();
    test_restrict_over_the_full_domain_loses_a_signed_zero_and_an_infinity();
    test_product_with_the_unit_polynomial_reproduces_its_input();
    test_the_first_derivative_at_the_endpoints_is_exact_on_integers();
    test_the_first_derivative_under_cancellation();
    test_a_split_reconstructs_the_original_curve();
    test_restrict_agrees_with_evaluation_on_the_subinterval();
    test_float32_runs_the_same_structural_identities();
    return pantr::test::summary("test_bezier");
}
