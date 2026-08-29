/// \file
/// Properties of the reference quadrature rule on the unit cube.
///
/// ## Why these cases and not others
///
/// The type does three things, and each fails in its own way:
///
///  - **Validation.** It is the C++ counterpart of Layer 2, so a caller with no
///    Python is protected by these throws and nothing else. The messages are
///    checked verbatim rather than by exception type, because
///    `design/backend_parity.md` treats what the library *says* as part of the
///    contract, and a message that drifts here makes `PANTR_BACKEND` change it.
///  - **Ordering.** The tensor product's whole observable content is which node
///    lands at which row. A transposed or reversed enumeration produces an array
///    of the right shape holding the right multiset of numbers, which no shape
///    check and no sum can see. `test_tensor_product_orders_last_axis_fastest`
///    pins it against two axes whose node sets are disjoint.
///  - **Arithmetic.** Only two places compute anything: the weight product, and
///    the Gauss-Legendre generation behind the second factory. Both are checked
///    against an oracle independent of this file -- the exact rational moment of
///    a monomial, and the exact product of two binary-exact weights.
///
/// ## The tolerance, and where every term comes from
///
/// Two assertions below compare a floating-point sum against an exact rational
/// value, so both need a bound. It is the same bound, built once in
/// `moment_bound` and derived rather than fitted.
///
/// Write `u = eps/2` for the unit roundoff, `N` for the point count, `ndim` for
/// the number of axes, `n_d` for the count on axis `d`, and `M` for the total
/// monomial degree `sum_d m_d`. Every term of `sum_i w_i prod_d x_id^{m_d}` is
/// **non-negative** -- Gauss-Legendre weights are positive and the cube lies in
/// the first orthant -- so the sum is perfectly conditioned, and every relative
/// bound below becomes an absolute one on multiplying by the exact value, which
/// is at most 1.
///
///  1. **Summation**: `N - 1` additions of a positive sum, `(N - 1) u`.
///  2. **Evaluating the monomial**: `M` multiplications to form the powers and
///     `ndim` to multiply them into the weight, `(M + ndim) u`.
///  3. **Displaced nodes**: `design/backend_parity.md` Rule 4 bounds the
///     Gauss-Legendre node displacement at `4.5 u` absolute on `[-1, 1]`, flat in
///     `n`. The map onto `[0, 1]` halves it and adds two roundings of its own, so
///     `4.5 u` covers it in that frame too. `|d/dx_d prod_d x^{m_d}| <= m_d` on
///     the cube, so the monomial moves by at most `M * 4.5 u`.
///  4. **Perturbed weights**: Rule 4 again, `|dw_i| <= u (4.5 A_i |w_i| + 5 |w_i|)`
///     with `A_i = 2|x_i| / (1 - x_i^2)`, and `max_i A_i |w_i| <= 2.6` uniformly
///     in `n` -- SUPPORTED there by a sweep to n = 2000, not proved. Summing over
///     one axis gives `u (11.7 n_d + 5)`. A tensor weight is a product of `ndim`
///     of them, whose relative errors add, and the remaining axes' weights sum to
///     1, so the axes' contributions add: `u sum_d (11.7 n_d + 5)`. Plus `ndim - 1`
///     roundings for forming the product itself.
///
/// The sum of the four is what `moment_bound` returns. It is conservative -- the
/// Python class's docstring records a *measured* worst case of 4 ulp for the
/// weight sum in 3-D -- and it is still twelve orders below the values compared,
/// so it asserts something rather than admitting anything. Rule 3's vacuity trap
/// is guarded from the other side too: `test_gauss_legendre_integrates_monomials`
/// checks that one degree above the rule's claim the same bound is *exceeded*.

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/quad/rule.hpp"

namespace {

using pantr::at;
using pantr::quad::QuadratureRule;
using pantr::quad::Rule1D;

constexpr double kUnitRoundoff = std::numeric_limits<double>::epsilon() / 2.0;
constexpr double kNodeDisplacementUnits = 4.5;  ///< Rule 4, absolute, flat in n.
constexpr double kWeightAmplification = 11.7;   ///< 4.5 * 2.6, from Rule 4's sup.
constexpr double kWeightFormulaRoundings = 5.0;

/// Build a rule from a flat row-major point table, for brevity below.
///
/// \param points Row-major `(num_points, ndim)` values.
/// \param ndim Number of axes.
/// \param weights One weight per point.
/// \return The rule.
QuadratureRule rule_of(const std::vector<double>& points, std::size_t ndim,
                       const std::vector<double>& weights) {
    return QuadratureRule(pantr::span2d<const double>(points.data(), weights.size(), ndim),
                          std::span<const double>(weights));
}

/// The message of the `std::invalid_argument` a call raises, or a marker.
///
/// \param fn The call to attempt.
/// \return The exception's `what()`, or a string naming what happened instead.
template <class F>
std::string message_of(F&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument& exc) {
        return exc.what();
    } catch (...) {
        return "<a different exception>";
    }
    return "<no exception>";
}

/// The derived absolute bound on a weighted monomial moment; see the file comment.
///
/// \param counts Points per axis.
/// \param degrees Monomial degree per axis, same length as `counts`.
/// \return The bound, absolute.
double moment_bound(std::span<const int> counts, std::span<const int> degrees) {
    const auto ndim = static_cast<double>(counts.size());
    double num_points = 1.0;
    double total_degree = 0.0;
    double weight_terms = 0.0;
    for (std::size_t d = 0; d < counts.size(); ++d) {
        num_points *= static_cast<double>(counts[d]);
        total_degree += static_cast<double>(degrees[d]);
        weight_terms +=
            kWeightAmplification * static_cast<double>(counts[d]) + kWeightFormulaRoundings;
    }
    const double summation = num_points - 1.0;
    const double evaluation = total_degree + ndim;
    const double displaced_nodes = total_degree * kNodeDisplacementUnits;
    const double product = ndim - 1.0;
    return kUnitRoundoff * (summation + evaluation + displaced_nodes + weight_terms + product);
}

/// Evaluate `sum_i w_i prod_d x_id^{m_d}` over a rule.
///
/// \param rule The rule.
/// \param degrees One exponent per axis.
/// \return The quadrature approximation of the monomial's integral.
double moment(const QuadratureRule& rule, std::span<const int> degrees) {
    const auto points = rule.points();
    double total = 0.0;
    for (std::size_t i = 0; i < rule.num_points(); ++i) {
        double term = rule.weights()[i];
        for (std::size_t d = 0; d < rule.ndim(); ++d) {
            for (int k = 0; k < degrees[d]; ++k) {
                term *= at(points, i, d);
            }
        }
        total += term;
    }
    return total;
}

void test_construction_validates() {
    const std::vector<double> one_point{0.5};
    const std::vector<double> one_weight{1.0};

    PANTR_CHECK(message_of([] {
        const std::vector<double> empty{};
        (void)QuadratureRule(pantr::span2d<const double>(empty.data(), 0, 1),
                             std::span<const double>(empty));
    }) == "points must be non-empty; got shape (0, 1).");

    PANTR_CHECK(message_of([&one_point] {
        (void)QuadratureRule(pantr::span2d<const double>(one_point.data(), 1, 0),
                             std::span<const double>(one_point));
    }) == "points must be non-empty; got shape (1, 0).");

    PANTR_CHECK(message_of([] {
        const std::vector<double> points{0.25, 0.75};
        const std::vector<double> weights{1.0};
        (void)QuadratureRule(pantr::span2d<const double>(points.data(), 2, 1),
                             std::span<const double>(weights));
    }) == "weights length 1 must match the number of points 2.");

    PANTR_CHECK(message_of([&one_weight] {
        const std::vector<double> points{std::numeric_limits<double>::infinity()};
        (void)rule_of(points, 1, one_weight);
    }) == "points must contain only finite values.");

    PANTR_CHECK(message_of([&one_point] {
        const std::vector<double> weights{std::numeric_limits<double>::quiet_NaN()};
        (void)rule_of(one_point, 1, weights);
    }) == "weights must contain only finite values.");

    PANTR_CHECK(message_of([&one_weight] {
        const std::vector<double> points{-0.25};
        (void)rule_of(points, 1, one_weight);
    }) == "points must lie in the unit cube [0, 1]^ndim.");

    PANTR_CHECK(message_of([&one_weight] {
        const std::vector<double> points{1.5};
        (void)rule_of(points, 1, one_weight);
    }) == "points must lie in the unit cube [0, 1]^ndim.");

    // A NaN coordinate is reported as non-finite rather than as out of range:
    // `NaN < 0.0` and `NaN > 1.0` are both false, so the ORDER of the two checks
    // decides the message, and the oracle checks finiteness first.
    PANTR_CHECK(message_of([&one_weight] {
        const std::vector<double> points{std::numeric_limits<double>::quiet_NaN()};
        (void)rule_of(points, 1, one_weight);
    }) == "points must contain only finite values.");
}

void test_the_boundary_and_a_negative_weight_are_legal() {
    // Lobatto-style endpoints, and a negative weight: neither is an error. The
    // class documents no constraint on weight sign or sum, and Newton-Cotes past
    // degree 8 and moment-fitted rules both carry negative weights.
    const std::vector<double> points{0.0, 1.0};
    const std::vector<double> weights{-1.0, 2.0};
    const QuadratureRule rule = rule_of(points, 1, weights);
    PANTR_CHECK(rule.ndim() == 1);
    PANTR_CHECK(rule.num_points() == 2);
    PANTR_CHECK(rule.weights()[0] == -1.0);
}

void test_the_rule_copies_its_arguments() {
    std::vector<double> points{0.5};
    const std::vector<double> weights{1.0};
    const QuadratureRule rule = rule_of(points, 1, weights);
    points[0] = 0.25;
    PANTR_CHECK_MSG(at(rule.points(), 0, 0) == 0.5, "the rule aliased its caller's array");
}

void test_tensor_product_validates() {
    PANTR_CHECK(message_of([] { (void)QuadratureRule::tensor_product({}); })
                == "tensor_product_quadrature: rules must have at least one axis.");

    PANTR_CHECK(message_of([] {
        const std::vector<double> nodes{0.25, 0.75};
        const std::vector<double> weights{1.0};
        const std::vector<Rule1D> rules{Rule1D{nodes, weights}};
        (void)QuadratureRule::tensor_product(rules);
    }) == "tensor_product_quadrature: axis 0 needs matching non-empty (nodes, weights); "
          "got shapes (2,) and (1,).");

    PANTR_CHECK(message_of([] {
        const std::vector<double> good{0.5};
        const std::vector<double> none{};
        const std::vector<Rule1D> rules{Rule1D{good, good}, Rule1D{none, none}};
        (void)QuadratureRule::tensor_product(rules);
    }) == "tensor_product_quadrature: axis 1 needs matching non-empty (nodes, weights); "
          "got shapes (0,) and (0,).");

    // A node outside the cube is the constructor's to reject, and this message
    // proves the factory delegates rather than re-checking.
    PANTR_CHECK(message_of([] {
        const std::vector<double> nodes{2.0};
        const std::vector<double> weights{1.0};
        const std::vector<Rule1D> rules{Rule1D{nodes, weights}};
        (void)QuadratureRule::tensor_product(rules);
    }) == "points must lie in the unit cube [0, 1]^ndim.");
}

void test_tensor_product_orders_last_axis_fastest() {
    // The two axes carry disjoint node sets, so a transposed or reversed
    // enumeration cannot reproduce this table by accident.
    const std::vector<double> nodes_0{0.1, 0.2};
    const std::vector<double> weights_0{0.25, 0.75};
    const std::vector<double> nodes_1{0.7, 0.8, 0.9};
    const std::vector<double> weights_1{0.5, 0.25, 0.25};
    const std::vector<Rule1D> rules{Rule1D{nodes_0, weights_0}, Rule1D{nodes_1, weights_1}};

    const QuadratureRule rule = QuadratureRule::tensor_product(rules);
    PANTR_CHECK(rule.ndim() == 2);
    PANTR_CHECK(rule.num_points() == 6);

    const auto points = rule.points();
    for (std::size_t i = 0; i < 6; ++i) {
        const std::size_t i0 = i / 3;
        const std::size_t i1 = i % 3;
        PANTR_CHECK_MSG(at(points, i, 0) == nodes_0[i0], "axis 0 node at row " + std::to_string(i));
        PANTR_CHECK_MSG(at(points, i, 1) == nodes_1[i1], "axis 1 node at row " + std::to_string(i));
        // Both factors are exact in binary, so their product is exact and this is
        // an equality rather than a bound.
        PANTR_CHECK_MSG(rule.weights()[i] == weights_0[i0] * weights_1[i1],
                        "weight at row " + std::to_string(i));
    }
}

void test_gauss_legendre_validates() {
    PANTR_CHECK(message_of([] { (void)QuadratureRule::gauss_legendre({}); })
                == "gauss_legendre_quadrature: ndim must be >= 1; got 0.");

    PANTR_CHECK(message_of([] {
        const std::vector<int> npts{0};
        (void)QuadratureRule::gauss_legendre(npts);
    }) == "gauss_legendre_quadrature: every npts entry must be >= 1; got (0,).");

    PANTR_CHECK(message_of([] {
        const std::vector<int> npts{2, -1};
        (void)QuadratureRule::gauss_legendre(npts);
    }) == "gauss_legendre_quadrature: every npts entry must be >= 1; got (2, -1).");
}

void test_gauss_legendre_one_point_is_the_midpoint_rule() {
    // The one rule whose values are exactly writable down: the single node is the
    // exact centre and the weight is the exact measure of the cube, so this is an
    // equality rather than a bound.
    const std::vector<int> npts{1, 1};
    const QuadratureRule rule = QuadratureRule::gauss_legendre(npts);
    PANTR_CHECK(rule.num_points() == 1);
    PANTR_CHECK(at(rule.points(), 0, 0) == 0.5);
    PANTR_CHECK(at(rule.points(), 0, 1) == 0.5);
    PANTR_CHECK(rule.weights()[0] == 1.0);
}

void test_gauss_legendre_integrates_monomials() {
    // n-point Gauss-Legendre is exact to degree 2n - 1 per axis. The oracle is
    // the exact rational moment of x^m on [0, 1], namely 1/(m + 1), which owes
    // nothing to this file.
    const std::vector<int> npts{4, 3};
    const QuadratureRule rule = QuadratureRule::gauss_legendre(npts);
    PANTR_CHECK(rule.num_points() == 12);

    for (int m0 = 0; m0 <= 7; ++m0) {
        for (int m1 = 0; m1 <= 5; ++m1) {
            const std::vector<int> degrees{m0, m1};
            const double exact =
                1.0 / static_cast<double>(m0 + 1) / static_cast<double>(m1 + 1);
            const double got = moment(rule, degrees);
            const double bound = moment_bound(npts, degrees);
            PANTR_CHECK_MSG(std::abs(got - exact) <= bound,
                            "degree (" + std::to_string(m0) + ", " + std::to_string(m1)
                                + "): " + std::to_string(got) + " against "
                                + std::to_string(exact) + ", bound " + std::to_string(bound));
        }
    }

    // One degree above what the rule claims, the quadrature must NOT reproduce
    // the moment. Without this the loop above would also pass for a bound so
    // loose that it admits a genuinely different answer.
    const std::vector<int> too_high{8, 0};
    PANTR_CHECK_MSG(std::abs(moment(rule, too_high) - 1.0 / 9.0) > moment_bound(npts, too_high),
                    "degree 8 is beyond a 4-point rule and must not be reproduced");
}

void test_gauss_legendre_weights_sum_to_the_measure_of_the_cube() {
    // Degree zero on every axis, so the monomial is the constant 1 and its exact
    // integral is the measure of the unit cube.
    const std::vector<int> degrees{0, 0, 0};
    for (const int n : {1, 2, 5, 17}) {
        const std::vector<int> npts{n, n, n};
        const QuadratureRule rule = QuadratureRule::gauss_legendre(npts);
        double total = 0.0;
        for (const double w : rule.weights()) {
            total += w;
        }
        PANTR_CHECK_MSG(std::abs(total - 1.0) <= moment_bound(npts, degrees),
                        "weight sum at n = " + std::to_string(n) + ": " + std::to_string(total));
    }
}

void test_gauss_legendre_maps_a_large_rule_into_the_open_cube() {
    // The constructor would have thrown had a node left the cube, so this says
    // the factory reaches it, and it is the one place a 200-point rule is built.
    const std::vector<int> npts{200};
    const QuadratureRule rule = QuadratureRule::gauss_legendre(npts);
    PANTR_CHECK(rule.num_points() == 200);
    const auto points = rule.points();
    bool ascending = true;
    for (std::size_t i = 1; i < rule.num_points(); ++i) {
        ascending = ascending && at(points, i - 1, 0) < at(points, i, 0);
    }
    PANTR_CHECK_MSG(ascending, "the mapped nodes must stay strictly ascending");
    PANTR_CHECK(at(points, 0, 0) > 0.0);
    PANTR_CHECK(at(points, rule.num_points() - 1, 0) < 1.0);
}

void test_to_string_names_both_counts() {
    const std::vector<int> npts{2, 3};
    PANTR_CHECK(QuadratureRule::gauss_legendre(npts).to_string()
                == "QuadratureRule(ndim=2, num_points=6)");
}

}  // namespace

int main() {
    test_construction_validates();
    test_the_boundary_and_a_negative_weight_are_legal();
    test_the_rule_copies_its_arguments();
    test_tensor_product_validates();
    test_tensor_product_orders_last_axis_fastest();
    test_gauss_legendre_validates();
    test_gauss_legendre_one_point_is_the_midpoint_rule();
    test_gauss_legendre_integrates_monomials();
    test_gauss_legendre_weights_sum_to_the_measure_of_the_cube();
    test_gauss_legendre_maps_a_large_rule_into_the_open_cube();
    test_to_string_names_both_counts();
    return pantr::test::summary("test_quad_rule");
}
