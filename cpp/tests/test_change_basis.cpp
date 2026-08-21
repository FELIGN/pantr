/// \file
/// Properties of the change-of-basis builders that hold without an oracle.
///
/// The Python parity tests are where these are compared against
/// `pantr.change_basis`, and where the sharp `kappa_inf` values from exact
/// rational arithmetic are available. This file asks the complementary question:
/// is what C++ produces a change-of-basis matrix at all?
///
/// ## The defining property, and why it is the test that matters
///
/// A builder named `A_to_B` returns `M` with `M [A values](x) = [B values](x)`
/// at every `x`. That single identity pins the matrix uniquely, so a builder can
/// pass every round trip, every symmetry and every closed form below and still be
/// wrong -- but it cannot pass this one and be wrong. It is checked directly for
/// three of the builders, at points that are not the ones any of them was built
/// from.
///
/// ## Tolerances
///
/// Every claim here is a linear-solve claim, so every tolerance is
/// `kappa_inf(M) * eps` times a stated factor, with `kappa_inf` computed from the
/// matrices in hand rather than assumed. That estimate is known to be unreliable
/// close to the domain boundary -- `pantr.change_basis`'s module docstring
/// measures it 8.6x too high at degree 13 and 4.9x too low at degree 15 for the
/// Bernstein/cardinal pair -- so **the degrees here stay well inside the
/// domain**, where the formed matrix and the exact one agree, and the boundary
/// claims are left to the Python side that has the exact numbers.
///
/// The factor is 16 throughout: a solve, a product against it and a subtraction,
/// each contributing a small multiple of `n eps`, rounded up to a power of two.
/// It is a safety factor and is named as one.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/basis/bernstein.hpp"
#include "pantr/basis/cardinal_bspline.hpp"
#include "pantr/basis/legendre.hpp"
#include "pantr/change_basis/change_basis.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/quad/legendre.hpp"

#include <Eigen/Core>
#include <Eigen/LU>

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

/// Acknowledged safety factor; see this file's header.
constexpr double kFactor = 16.0;

using Matrix = std::vector<double>;

/// Gauss-Legendre on `[0, 1]`, the rule the projected builders integrate with.
struct Rule {
    std::vector<double> points;
    std::vector<double> weights;
};

Rule unit_gauss_legendre(int n) {
    Rule rule{std::vector<double>(static_cast<std::size_t>(n)),
              std::vector<double>(static_cast<std::size_t>(n))};
    pantr::gauss_legendre_symmetric(n, std::span<double>(rule.points),
                                    std::span<double>(rule.weights));
    for (std::size_t j = 0; j < rule.points.size(); ++j) {
        rule.points[j] = 0.5 * (rule.points[j] + 1.0);
        rule.weights[j] *= 0.5;
    }
    return rule;
}

/// `kappa_inf(M)` estimated from `M` and its computed inverse.
double kappa_inf(const Matrix& m, const Matrix& inverse, std::size_t n) {
    double norm_m = 0.0;
    double norm_inv = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double row_m = 0.0;
        double row_inv = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            row_m += std::abs(m[i * n + j]);
            row_inv += std::abs(inverse[i * n + j]);
        }
        norm_m = std::max(norm_m, row_m);
        norm_inv = std::max(norm_inv, row_inv);
    }
    return norm_m * norm_inv;
}


/// `kappa_inf` of the Gram matrix a projected builder actually solves with.
///
/// **This, and not `kappa_inf(M)`, is what the bound needs.** An earlier version of
/// this file sized every tolerance with the condition number of the *result*, which
/// is a different matrix: measured at degree 8 that is 3.8e8 times too large for
/// `legendre_to_cardinal`, whose Gram is the identity and whose true kappa is one,
/// and 3.4e4 times too *small* for `cardinal_to_bernstein` -- a latent false failure
/// rather than merely a dormant test.
///
/// \param basis The new basis tabulated at the quadrature nodes, `(n_quad, n)`.
/// \param weights The quadrature weights.
/// \param n The basis size.
/// \return `||G||_inf ||G^-1||_inf`.
double gram_kappa(const Matrix& basis, const std::vector<double>& weights, std::size_t n) {
    using Dense = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    const auto rows = static_cast<Eigen::Index>(weights.size());
    const auto cols = static_cast<Eigen::Index>(n);

    Dense b(rows, cols);
    for (Eigen::Index j = 0; j < rows; ++j) {
        for (Eigen::Index i = 0; i < cols; ++i) {
            b(j, i) = basis[static_cast<std::size_t>(j) * n + static_cast<std::size_t>(i)];
        }
    }
    Dense weighted = b.transpose();
    for (Eigen::Index j = 0; j < rows; ++j) {
        weighted.col(j) *= weights[static_cast<std::size_t>(j)];
    }
    const Dense gram = weighted * b;
    const Dense inverse =
        Eigen::PartialPivLU<Dense>(gram).solve(Dense::Identity(cols, cols));

    Matrix g(n * n);
    Matrix gi(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            g[i * n + j] = gram(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
            gi[i * n + j] = inverse(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        }
    }
    return kappa_inf(g, gi, n);
}

/// Tabulate one basis at the quadrature nodes, flat and row-major.
///
/// \param which `'b'` for Bernstein, `'l'` for Legendre.
/// \param degree Polynomial degree.
/// \param points Quadrature nodes.
/// \return The `(points.size(), degree + 1)` values.
Matrix tabulated_at(char which, int degree, const std::vector<double>& points) {
    const auto n = static_cast<std::size_t>(degree + 1);
    Matrix values(points.size() * n, 0.0);
    const pantr::span2d<double> view(values.data(), points.size(), n);
    if (which == 'b') {
        pantr::tabulate_bernstein_1d<double>(degree, std::span<const double>(points), view);
    } else {
        pantr::tabulate_legendre_1d<double>(degree, std::span<const double>(points), view);
    }
    return values;
}

/// `max_ij |A B - I|`.
double identity_residual(const Matrix& a, const Matrix& b, std::size_t n) {
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                acc += a[i * n + k] * b[k * n + j];
            }
            worst = std::max(worst, std::abs(acc - (i == j ? 1.0 : 0.0)));
        }
    }
    return worst;
}

Matrix square(int degree) {
    const auto n = static_cast<std::size_t>(degree + 1);
    return Matrix(n * n, 0.0);
}

pantr::span2d<double> view_of(Matrix& m, int degree) {
    const auto n = static_cast<std::size_t>(degree + 1);
    return pantr::span2d<double>(m.data(), n, n);
}

/// Each inverse pair multiplies to the identity, to `kappa_inf * eps`.
void inverse_pairs_round_trip() {
    for (int degree : {1, 3, 5, 8}) {
        const auto n = static_cast<std::size_t>(degree + 1);
        const Rule rule = unit_gauss_legendre(degree + 1);
        const std::span<const double> pts(rule.points);
        const std::span<const double> wts(rule.weights);

        // Lagrange/Bernstein, at equispaced nodes built here rather than
        // imported, so this file does not depend on the node helper.
        std::vector<double> equispaced(n);
        for (std::size_t i = 0; i < n; ++i) {
            equispaced[i] = static_cast<double>(i) / static_cast<double>(degree);
        }
        Matrix l2b = square(degree);
        Matrix b2l = square(degree);
        pantr::lagrange_to_bernstein_1d<double>(degree, std::span<const double>(equispaced),
                                                view_of(l2b, degree));
        pantr::bernstein_to_lagrange_1d<double>(degree, std::span<const double>(equispaced),
                                                view_of(b2l, degree));
        PANTR_CHECK_MSG(identity_residual(l2b, b2l, n) <=
                            kFactor * kappa_inf(l2b, b2l, n) * kEps,
                        "lagrange/bernstein at degree " + std::to_string(degree));

        Matrix b2c = square(degree);
        Matrix c2b = square(degree);
        pantr::bernstein_to_cardinal_1d<double>(degree, pts, wts, view_of(b2c, degree));
        pantr::cardinal_to_bernstein_1d<double>(degree, pts, wts, view_of(c2b, degree));
        PANTR_CHECK_MSG(identity_residual(b2c, c2b, n) <=
                            kFactor * kappa_inf(b2c, c2b, n) * kEps,
                        "bernstein/cardinal at degree " + std::to_string(degree));

        Matrix l2c = square(degree);
        Matrix c2l = square(degree);
        pantr::legendre_to_cardinal_1d<double>(degree, pts, wts, view_of(l2c, degree));
        pantr::cardinal_to_legendre_1d<double>(degree, pts, wts, view_of(c2l, degree));
        PANTR_CHECK_MSG(identity_residual(l2c, c2l, n) <=
                            kFactor * kappa_inf(l2c, c2l, n) * kEps,
                        "legendre/cardinal at degree " + std::to_string(degree));
    }
}

/// `M [A values](x) = [B values](x)` at points none of the builders saw.
///
/// This is the identity that defines each matrix; the round trips above would
/// pass for a consistent but wrong pair, and this would not.
void the_matrices_actually_change_the_basis() {
    const std::vector<double> probe{0.05, 0.31, 0.5, 0.77, 0.99};

    for (int degree : {1, 3, 5, 8}) {
        const auto n = static_cast<std::size_t>(degree + 1);
        const Rule rule = unit_gauss_legendre(degree + 1);
        const std::span<const double> pts(rule.points);
        const std::span<const double> wts(rule.weights);

        std::vector<double> bern(probe.size() * n);
        std::vector<double> card(probe.size() * n);
        std::vector<double> leg(probe.size() * n);
        pantr::tabulate_bernstein_1d<double>(degree, std::span<const double>(probe),
                                             pantr::span2d<double>(bern.data(), probe.size(), n));
        pantr::tabulate_cardinal_bspline_1d<double>(
            degree, std::span<const double>(probe),
            pantr::span2d<double>(card.data(), probe.size(), n));
        pantr::tabulate_legendre_1d<double>(degree, std::span<const double>(probe),
                                            pantr::span2d<double>(leg.data(), probe.size(), n));

        Matrix b2c = square(degree);
        Matrix c2b = square(degree);
        Matrix l2c = square(degree);
        pantr::bernstein_to_cardinal_1d<double>(degree, pts, wts, view_of(b2c, degree));
        pantr::cardinal_to_bernstein_1d<double>(degree, pts, wts, view_of(c2b, degree));
        Matrix c2l = square(degree);
        pantr::legendre_to_cardinal_1d<double>(degree, pts, wts, view_of(l2c, degree));
        pantr::cardinal_to_legendre_1d<double>(degree, pts, wts, view_of(c2l, degree));

        // `M @ [A values] = [B values]`, one probe point at a time.
        //
        // The tolerance is `||M||_inf * max|A values| * kappa_inf(M) * eps`, and
        // each factor is there for a reason rather than for room: `M` carries a
        // relative error of order `kappa eps` from its own solve, the product
        // then scales that by the row sums of `M` and the size of the vector, and
        // the summation itself contributes `n eps`, absorbed into the factor of
        // 16. **`kappa` is the condition number of the matrix that builder's
        // SOLVE used**, passed in per builder, not the condition number of the
        // result -- those differ by up to eight orders of magnitude and in one
        // direction the substitution makes the test too strict rather than too
        // loose.
        const auto check = [&](const Matrix& m, double kappa, const std::vector<double>& from,
                               const std::vector<double>& to, const char* name) {
            double norm_m = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                double row = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    row += std::abs(m[i * n + j]);
                }
                norm_m = std::max(norm_m, row);
            }
            for (std::size_t p = 0; p < probe.size(); ++p) {
                double max_from = 0.0;
                for (std::size_t k = 0; k < n; ++k) {
                    max_from = std::max(max_from, std::abs(from[p * n + k]));
                }
                const double tol = kFactor * norm_m * max_from * kappa * kEps;
                for (std::size_t i = 0; i < n; ++i) {
                    double acc = 0.0;
                    for (std::size_t k = 0; k < n; ++k) {
                        acc += m[i * n + k] * from[p * n + k];
                    }
                    PANTR_CHECK_MSG(std::abs(acc - to[p * n + i]) <= tol,
                                    std::string(name) + " at degree " +
                                        std::to_string(degree));
                }
            }
        };

        // The kappa each builder's own solve carries, not the result's. See
        // `gram_kappa`.
        const Matrix bern_at_nodes = tabulated_at('b', degree, rule.points);
        const Matrix leg_at_nodes = tabulated_at('l', degree, rule.points);
        const double kappa_gram_bern = gram_kappa(bern_at_nodes, rule.weights, n);
        const double kappa_gram_leg = gram_kappa(leg_at_nodes, rule.weights, n);
        const double kappa_b2c_forward = kappa_inf(b2c, c2b, n);
        const double kappa_l2c_forward = kappa_inf(l2c, c2l, n);

        check(b2c, kappa_gram_bern, bern, card, "bernstein_to_cardinal");
        check(c2b, kappa_b2c_forward * kappa_gram_bern, card, bern, "cardinal_to_bernstein");
        check(l2c, kappa_gram_leg, leg, card, "legendre_to_cardinal");
        check(c2l, kappa_l2c_forward, card, leg, "cardinal_to_legendre");
    }
}

/// The dual coefficients are the cardinal-to-Legendre matrix transposed, exactly.
/// Nothing is recomputed between the two, so this is bit-equality or a bug.
void dual_coefficients_are_the_transpose() {
    for (int degree : {0, 1, 4, 7}) {
        const auto n = static_cast<std::size_t>(degree + 1);
        const Rule rule = unit_gauss_legendre(degree + 1);
        Matrix c2l = square(degree);
        Matrix dual = square(degree);
        pantr::cardinal_to_legendre_1d<double>(degree, std::span<const double>(rule.points),
                                               std::span<const double>(rule.weights),
                                               view_of(c2l, degree));
        pantr::cardinal_dual_legendre_coeffs_1d<double>(
            degree, std::span<const double>(rule.points),
            std::span<const double>(rule.weights), view_of(dual, degree));
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                PANTR_CHECK(dual[i * n + j] == c2l[j * n + i]);
            }
        }
    }
}

/// Monomial-to-Bernstein: lower triangular, unit first column, and the closed
/// form at degree 3. The entries are exact rationals with small denominators, so
/// unlike everything else here this is an equality rather than a tolerance.
void monomial_to_bernstein_matches_its_closed_form() {
    const int degree = 3;
    Matrix m = square(degree);
    pantr::monomial_to_bernstein_1d<double>(degree, view_of(m, degree));

    // C(i,j)/C(3,j): rows 1, [1 1/3], [1 2/3 1/3], [1 1 1 1].
    const double expected[16] = {1, 0,          0,          0,  //
                                 1, 1.0 / 3.0,  0,          0,  //
                                 1, 2.0 / 3.0,  1.0 / 3.0,  0,  //
                                 1, 1,          1,          1};
    for (std::size_t k = 0; k < 16; ++k) {
        PANTR_CHECK(m[k] == expected[k]);
    }

    // Every column of a higher-degree matrix ends at 1, since C(n,j)/C(n,j) = 1.
    for (int d : {5, 12, 30}) {
        const auto n = static_cast<std::size_t>(d + 1);
        Matrix big = square(d);
        pantr::monomial_to_bernstein_1d<double>(d, view_of(big, d));
        for (std::size_t j = 0; j < n; ++j) {
            PANTR_CHECK(big[(n - 1) * n + j] == 1.0);
        }
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                PANTR_CHECK(big[i * n + j] == 0.0);
            }
        }
    }
}

/// The float32 instantiation runs and produces a matrix, at its own precision.
/// It exists to catch a template that only compiles for double.
void float32_instantiation_round_trips() {
    constexpr auto kEps32 = static_cast<double>(std::numeric_limits<float>::epsilon());
    const int degree = 4;
    const auto n = static_cast<std::size_t>(degree + 1);
    const Rule rule64 = unit_gauss_legendre(degree + 1);
    const std::vector<float> pts(rule64.points.begin(), rule64.points.end());
    const std::vector<float> wts(rule64.weights.begin(), rule64.weights.end());

    std::vector<float> a(n * n, 0.0F);
    std::vector<float> b(n * n, 0.0F);
    pantr::legendre_to_cardinal_1d<float>(degree, std::span<const float>(pts),
                                          std::span<const float>(wts),
                                          pantr::span2d<float>(a.data(), n, n));
    pantr::cardinal_to_legendre_1d<float>(degree, std::span<const float>(pts),
                                          std::span<const float>(wts),
                                          pantr::span2d<float>(b.data(), n, n));
    const Matrix a64(a.begin(), a.end());
    const Matrix b64(b.begin(), b.end());
    PANTR_CHECK(identity_residual(a64, b64, n) <= kFactor * kappa_inf(a64, b64, n) * kEps32);
}

}  // namespace

int main() {
    inverse_pairs_round_trip();
    the_matrices_actually_change_the_basis();
    dual_coefficients_are_the_transpose();
    monomial_to_bernstein_matches_its_closed_form();
    float32_instantiation_round_trips();
    return pantr::test::summary("test_change_basis");
}
