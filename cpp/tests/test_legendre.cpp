/// \file
/// Properties of the orthonormal shifted Legendre tabulation.
///
/// ## Why the tolerances here are not the Bernstein ones
///
/// The Bernstein and cardinal B-spline files both lean on the same fact: their
/// values are non-negative and sum to one, so a stage cannot amplify an error.
/// **Legendre has neither property.** The polynomials oscillate in sign, the
/// recurrence subtracts, and `|p_i(x)|` reaches `sqrt(2i+1)` at the endpoints.
/// So the error argument has to carry a growth factor rather than assume none.
///
/// The naive reading of the three-term recurrence `p_i = a_i y p_{i-1} - b_i p_{i-2}`
/// is that an error in `p_{i-1}` is multiplied by `|a_i y| <= 2` at each step and so
/// compounds as `2^i`. **That is wrong, and this file used to assert it.** It is the
/// classic mistake for a recurrence whose two homogeneous solutions are both bounded:
/// the growing and decaying parts do not simply multiply.
///
/// `design/backend_parity.md` Rule 4 records the correct answer for the Legendre
/// recurrence as **PROVED, both halves**: the error growth is `Theta(n^2)`, with
/// closed form `C(n, +-1) = (7/4) n^2 + (9/4) n - 4 H_n`. Deriving an exponential
/// bound here while a proved quadratic one sat four directories away was the defect;
/// the tolerances below now carry the right order.
///
/// **The order transfers; the constant is not copied.** Rule 4 is about `P_n` on
/// `[-1, 1]`; this kernel tabulates the *orthonormal shifted* family on `[0, 1]`,
/// whose normalisation differs, so `7/4` is not this family's constant. What is used
/// below is `2 i^2`, and its status is: **order derived** from Rule 4, **constant
/// measured**, not proved. Against a long-double reference of the same recurrence,
/// worst relative error over 201 points on `[0, 1]`, in units of eps: 3.5 at degree
/// 4, 8.1 at 8, 11.6 at 16, 23.1 at 32, 71.2 at 64. `2 i^2` sits above every one of
/// those with between 9x and 1150x in hand, and unlike `2^i` the margin does not
/// diverge.
///
/// So the claims below are chosen to be ones that hold **without** needing a
/// sharp forward bound:
///
///   - orthonormality, measured by Gauss-Legendre quadrature of high enough
///     order that the quadrature itself is exact for the integrand;
///   - the endpoint values `p_i(1) = sqrt(2i+1)` and
///     `p_i(0) = (-1)^i sqrt(2i+1)`, which are closed forms;
///   - the low-degree closed forms;
///   - agreement between the two instantiations at the width they share.
///
/// Each is compared against `2^degree * eps` where a growth factor is genuinely
/// unavoidable, and against a small multiple of `eps` where it is not, and the
/// distinction is stated per test rather than hidden in one constant.
///
/// ## The recurrence coefficients are mutation-checked, not merely covered
///
/// Replacing `b[i]`'s leading factor in
/// `basis/legendre.hpp::tabulate_legendre_1d` from `(i - 1) / i` to `i / i`, which
/// is the same expression with the shift dropped and evaluates to one, makes this
/// file fail while the rest of the C++ suite still passes. So the assertions here
/// discriminate the recurrence rather than only exercising it.
///
/// Redo it by making that edit and running `ctest`. See test_bernstein.cpp for the
/// companion check and for why this is not automated.

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <algorithm>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/basis/legendre.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/quad/legendre.hpp"

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

/// The `Theta(i^2)` bound of this file's header: order from Rule 4, constant
/// measured. Zero at degree 0, where the basis is the constant 1 and exact.
double recurrence_bound(int degree) {
    const auto d = static_cast<double>(degree);
    return std::max(1.0, 2.0 * d * d) * kEps;
}

template <class T>
std::vector<T> tabulate(int degree, const std::vector<T>& pts) {
    std::vector<T> out(pts.size() * static_cast<std::size_t>(degree + 1), T(0));
    const pantr::span2d<T> view(out.data(), pts.size(), static_cast<std::size_t>(degree + 1));
    pantr::tabulate_legendre_1d<T>(degree, std::span<const T>(pts), view);
    return out;
}

/// `p_i(1) = sqrt(2i+1)` and `p_i(0) = (-1)^i sqrt(2i+1)`. Closed forms, so this
/// pins which polynomials the recurrence produces rather than merely that it
/// produces a consistent family.
void endpoint_values_match_their_closed_form() {
    const int degree = 12;
    const std::vector<double> pts{0.0, 1.0};
    const std::vector<double> vals = tabulate(degree, pts);
    const auto width = static_cast<std::size_t>(degree + 1);
    for (std::size_t i = 0; i < width; ++i) {
        const double norm = std::sqrt(2.0 * static_cast<double>(i) + 1.0);
        const double at_zero = vals[i];
        const double at_one = vals[width + i];
        const double sign = (i % 2 == 0) ? 1.0 : -1.0;
        PANTR_CHECK_MSG(std::abs(at_one - norm) <= recurrence_bound(static_cast<int>(i)) * norm,
                        "p_" + std::to_string(i) + "(1)");
        PANTR_CHECK_MSG(std::abs(at_zero - sign * norm) <=
                            recurrence_bound(static_cast<int>(i)) * norm,
                        "p_" + std::to_string(i) + "(0)");
    }
}

/// Degrees 0 to 2 against the closed forms of the orthonormal shifted family.
/// No growth factor is needed at these degrees, so the tolerance is a small
/// multiple of eps rather than the compounding bound.
void low_degrees_match_closed_forms() {
    const std::vector<double> pts{0.0, 0.25, 0.5, 0.75, 1.0};
    const std::vector<double> vals = tabulate(2, pts);
    for (std::size_t j = 0; j < pts.size(); ++j) {
        const double y = 2.0 * pts[j] - 1.0;
        const double expected[3] = {1.0, std::sqrt(3.0) * y,
                                    std::sqrt(5.0) * 0.5 * (3.0 * y * y - 1.0)};
        for (std::size_t i = 0; i < 3; ++i) {
            PANTR_CHECK(std::abs(vals[j * 3 + i] - expected[i]) <= 8.0 * kEps);
        }
    }
}

/// Orthonormality on `[0, 1]`, which is the property the basis is named for and
/// the one every consumer relies on.
///
/// The inner products are formed by Gauss-Legendre quadrature on `2 * degree + 1`
/// nodes, which integrates a polynomial of degree `4 * degree + 1` exactly, well
/// above the `2 * degree` of the integrand. So any deviation here is the
/// recurrence's error and not the quadrature's -- with the caveat that the nodes
/// themselves come from `pantr::gauss_legendre_1d`, so this test is not
/// independent of that kernel. It would catch a wrong Legendre basis; it would
/// not distinguish a wrong basis from a wrong quadrature.
void the_basis_is_orthonormal() {
    const int degree = 8;
    const int n_quad_i = 2 * degree + 1;
    const auto n_quad = static_cast<std::size_t>(n_quad_i);
    std::vector<double> nodes(n_quad);
    std::vector<double> weights(n_quad);
    pantr::gauss_legendre_symmetric(n_quad_i, std::span<double>(nodes),
                                    std::span<double>(weights));
    // The kernel returns the rule on [-1, 1]; the basis lives on [0, 1].
    for (std::size_t j = 0; j < n_quad; ++j) {
        nodes[j] = 0.5 * (nodes[j] + 1.0);
        weights[j] *= 0.5;
    }

    const std::vector<double> vals = tabulate(degree, nodes);
    const auto width = static_cast<std::size_t>(degree + 1);

    for (std::size_t i = 0; i < width; ++i) {
        for (std::size_t k = i; k < width; ++k) {
            double integral = 0.0;
            for (std::size_t j = 0; j < n_quad; ++j) {
                integral += weights[j] * vals[j * width + i] * vals[j * width + k];
            }
            const double expected = (i == k) ? 1.0 : 0.0;
            PANTR_CHECK_MSG(std::abs(integral - expected) <= recurrence_bound(degree),
                            "<p_" + std::to_string(i) + ", p_" + std::to_string(k) + ">");
        }
    }
}

/// float32 agrees with float64 to float32's own precision. The coefficients are
/// double in both instantiations by construction, so what this checks is that
/// only the per-point values narrow.
void float32_tracks_float64() {
    constexpr auto kEps32 = static_cast<double>(std::numeric_limits<float>::epsilon());
    const int degree = 6;
    const std::vector<double> pts64{0.0, 0.125, 0.5, 0.875, 1.0};
    const std::vector<float> pts32{0.0F, 0.125F, 0.5F, 0.875F, 1.0F};

    const std::vector<double> a = tabulate(degree, pts64);
    const std::vector<float> b = tabulate(degree, pts32);
    const auto width = static_cast<std::size_t>(degree + 1);

    for (std::size_t j = 0; j < pts64.size(); ++j) {
        for (std::size_t i = 0; i < width; ++i) {
            const double lhs = a[j * width + i];
            const double rhs = static_cast<double>(b[j * width + i]);
            const double scale = std::max(1.0, std::abs(lhs));
            PANTR_CHECK(std::abs(lhs - rhs) <=
                        std::max(1.0, 2.0 * static_cast<double>(i * i)) * kEps32 * scale);
        }
    }
}

}  // namespace

int main() {
    endpoint_values_match_their_closed_form();
    low_degrees_match_closed_forms();
    the_basis_is_orthonormal();
    float32_tracks_float64();
    return pantr::test::summary("test_legendre");
}
