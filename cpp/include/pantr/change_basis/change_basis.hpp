#pragma once

/// \file
/// The eight change-of-basis builders of `src/pantr/change_basis.py`.
///
/// Every builder returns the `(degree+1, degree+1)` matrix `M` with
/// `M [A values](x) = [B values](x)`, and the Python module stays as the parity
/// oracle.
///
/// ## Nodes come in; they are not computed here
///
/// `compute_lagrange_to_bernstein_1d` picks its nodes from one of five families,
/// and two of them -- Gauss-Lobatto-Legendre and Chebyshev of the first kind --
/// are ones `src/pantr/_backend.py` says are **deliberately never dispatched**
/// and stay on `numpy.polynomial`. A kernel that computed its own nodes would
/// therefore have to either contradict that decision or carry a second, divergent
/// implementation of two rules.
///
/// So it takes the nodes as an argument, and so does every builder that needs a
/// quadrature rule. Layer 2 in Python fetches them from whichever path is right
/// for the requested variant and hands over an array. The `LagrangeVariant` enum
/// never crosses the boundary, which is also what design/cross_backend_types.md
/// asks for: types are Python-owned and only arrays and scalars cross.
///
/// ## Arithmetic width is the output dtype, not the accumulator
///
/// Unlike the tabulation kernels, these do **not** widen. The oracle forms its
/// Gram matrix, its mixed matrix and its solve in the requested dtype throughout
/// -- `np.linalg.solve` on a float32 pair calls LAPACK `sgesv`, not `dgesv`. A
/// C++ port that solved in double would be more accurate and would not be the
/// same function, and the difference would show up as a parity failure attributed
/// to the wrong cause. `Acc` appears below only where the oracle itself computes
/// in float64: the binomial table, which numpy builds from Python ints.
///
/// ## Where parity stops being bit-exact, and why that is not a defect
///
/// The tabulations this builds on are bit-exact against their oracles. These are
/// not, and cannot be:
///
///   - **the Gram and mixed products.** `new.T @ diag(w) @ new` is a `dgemm` in
///     numpy and a blocked product in Eigen. Both are correct; they sum the same
///     terms in different orders. The `diag(w)` factor is not a source of
///     disagreement -- multiplying by a diagonal adds only exact zeros to each
///     dot product -- but the `n_quad`-long contraction that follows it is.
///   - **the solve.** `np.linalg.solve` is LAPACK `gesv`; this is
///     `Eigen::PartialPivLU`. Both are LU with partial pivoting and both are
///     backward stable, so the two computed solutions differ by roughly
///     `2 c n rho kappa_inf(A) eps`. That is a derivable bound rather than a
///     measured one, and the `kappa_inf` it needs is already tabulated per degree
///     in the oracle's own docstrings, from exact rational arithmetic.
///
/// tests/parity/test_change_basis.py carries the bound and its derivation.
///
/// ## One fewer allocation than the oracle, and it changes nothing
///
/// The oracle materialises `np.diag(weights)`, an `n_quad` by `n_quad` matrix
/// that is zero off the diagonal, and multiplies through it. Scaling the rows
/// directly, as below, produces the identical matrix: entry `(i, j)` of
/// `A.T @ diag(w)` is `sum_k A[k, i] w_k [k == j]`, whose only non-zero term is
/// `A[j, i] w_j`, and adding exact zeros to a float sum does not perturb it. The
/// saving is `n_quad^2` stores, not a different answer.

#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "pantr/basis/bernstein.hpp"
#include "pantr/basis/cardinal_bspline.hpp"
#include "pantr/basis/legendre.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr {

namespace detail {

/// Row-major Eigen view over a `span2d`, so no copy is needed to reach Eigen.
template <std::floating_point T>
using eigen_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// Tabulate a basis at `points` into a fresh row-major matrix.
///
/// \tparam T Scalar type.
/// \tparam Tabulate Callable with the signature of the Layer 3 tabulators.
/// \param tabulate The tabulator to run.
/// \param degree Degree passed through to the tabulator.
/// \param points Evaluation points.
/// \return A `(points.size(), degree + 1)` matrix of basis values.
template <std::floating_point T, class Tabulate>
[[nodiscard]] eigen_matrix<T> tabulated(Tabulate tabulate, int degree,
                                        std::span<const T> points) {
    const auto rows = static_cast<Eigen::Index>(points.size());
    const auto cols = static_cast<Eigen::Index>(degree + 1);
    eigen_matrix<T> values(rows, cols);
    const span2d<T> view(values.data(), points.size(), static_cast<std::size_t>(degree + 1));
    tabulate(degree, points, view);
    return values;
}

/// Copy an Eigen matrix into a `span2d` output view.
///
/// \param source Matrix to copy from.
/// \param out Destination view, of matching shape.
template <std::floating_point T>
void write_out(const eigen_matrix<T>& source, span2d<T> out) {
    for (Eigen::Index i = 0; i < source.rows(); ++i) {
        for (Eigen::Index j = 0; j < source.cols(); ++j) {
            at(out, static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = source(i, j);
        }
    }
}

/// Invert `forward` by one LU solve against the identity.
///
/// **This computes an inverse, and saying otherwise was wrong.** The oracle writes
/// `np.linalg.solve(forward, np.eye(n))`, and an earlier version of this comment
/// called that "a solve with `n` right-hand sides rather than an explicit inverse".
/// It is precisely Du Croz and Higham's *Method A* for matrix inversion (IMA J.
/// Numer. Anal. **12** (1992) 1-19, section 3.1), the distinction their paper and
/// Higham's ASNA chapter 14 exist to draw, and in numpy 2.4.6 it is bitwise
/// identical to `numpy.linalg.inv`.
///
/// The consequence is a one-sided guarantee, and callers should know which side.
/// Method A bounds the **right** residual `A X - I` and says nothing about the
/// left. Measured on the Legendre pair in float64: at degree 6 `||AX-I||` is
/// 2.8e-15 against `||XA-I||` 4.8e-13; at degree 12, the last degree the module
/// accepts, 2.7e-13 against **1.2e-4**. The round-trip examples in the Python
/// docstrings exhibit `A @ B` for that reason, not by preference.
///
/// \param forward The matrix to invert. Must be square.
/// \return Its inverse.
template <std::floating_point T>
[[nodiscard]] eigen_matrix<T> solve_against_identity(const eigen_matrix<T>& forward) {
    const eigen_matrix<T> identity =
        eigen_matrix<T>::Identity(forward.rows(), forward.cols());
    return Eigen::PartialPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(forward)
        .solve(identity);
}

/// The Gram projection shared by the two projected builders.
///
/// Solves `G M^T = C` for `M`, with `G` the Gram matrix of the new basis under
/// the given quadrature and `C` the mixed matrix between the two bases. Row `i`
/// of the result holds the `i`-th old basis function expanded in the new basis.
///
/// \param new_basis New-basis values, `(n_quad, n_new)`.
/// \param old_basis Old-basis values, `(n_quad, n_old)`.
/// \param weights Quadrature weights, length `n_quad`.
/// \return The `(n_old, n_new)` change-of-basis matrix.
template <std::floating_point T>
[[nodiscard]] eigen_matrix<T> gram_projection(const eigen_matrix<T>& new_basis,
                                              const eigen_matrix<T>& old_basis,
                                              std::span<const T> weights) {
    const Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> w(
        weights.data(), static_cast<Eigen::Index>(weights.size()));

    // `new.T @ diag(w)`, formed by scaling rather than by a dense product. See
    // the file comment for why the two agree exactly.
    const eigen_matrix<T> weighted = new_basis.transpose() * w.asDiagonal();

    const eigen_matrix<T> gram = weighted * new_basis;
    const eigen_matrix<T> mixed = weighted * old_basis;

    return eigen_matrix<T>(
        Eigen::PartialPivLU<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(gram)
            .solve(mixed)
            .transpose());
}

}  // namespace detail

/// Lagrange-to-Bernstein: one tabulation, no solve.
///
/// `C[j, k] = B_j(x_k)` because the Lagrange basis is cardinal at its own nodes,
/// so the matrix is the Bernstein basis tabulated at the Lagrange points and
/// transposed. The oracle expresses that by handing `out.T` to the tabulator; the
/// same thing is done here by writing the transposed indices.
///
/// \param degree Polynomial degree. Must be at least 1.
/// \param lagrange_points The `degree + 1` nodes of the Lagrange basis.
/// \param out Output view of shape `(degree + 1, degree + 1)`.
///
/// \note No input validation is performed. Layer 3; see CLAUDE.md. For general
///       use call `pantr.change_basis.compute_lagrange_to_bernstein_1d`.
template <std::floating_point T>
void lagrange_to_bernstein_1d(int degree, std::span<const T> lagrange_points, span2d<T> out) {
    const auto values =
        detail::tabulated<T>([](int d, std::span<const T> p,
                                span2d<T> o) { tabulate_bernstein_1d<T>(d, p, o); },
                             degree, lagrange_points);
    const auto n = static_cast<std::size_t>(degree + 1);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            at(out, i, j) = values(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i));
        }
    }
}

/// Bernstein-to-Lagrange: the inverse of the above, by one LU solve.
///
/// \param degree Polynomial degree. Must be at least 1.
/// \param lagrange_points The `degree + 1` nodes of the Lagrange basis.
/// \param out Output view of shape `(degree + 1, degree + 1)`.
///
/// \note No input validation is performed, and in particular the degree domain
///       that keeps this solve non-singular is Layer 2's to enforce. Layer 3; see
///       CLAUDE.md. For general use call
///       `pantr.change_basis.compute_bernstein_to_lagrange_1d`.
template <std::floating_point T>
void bernstein_to_lagrange_1d(int degree, std::span<const T> lagrange_points, span2d<T> out) {
    const auto n = static_cast<std::size_t>(degree + 1);
    detail::eigen_matrix<T> forward(static_cast<Eigen::Index>(n),
                                    static_cast<Eigen::Index>(n));
    const span2d<T> forward_view(forward.data(), n, n);
    lagrange_to_bernstein_1d<T>(degree, lagrange_points, forward_view);
    detail::write_out<T>(detail::solve_against_identity<T>(forward), out);
}

/// Bernstein-to-cardinal: a Gram projection.
///
/// \param degree Polynomial degree. Must be non-negative.
/// \param quad_points Quadrature nodes on `[0, 1]`.
/// \param quad_weights Quadrature weights, same length as `quad_points`.
/// \param out Output view of shape `(degree + 1, degree + 1)`.
///
/// \note No input validation is performed, and the degree domain is Layer 2's.
///       This builder has one -- 26 in float64, 12 in float32 -- and was the only
///       one of the eight whose note omitted to say so. Layer 3; see CLAUDE.md. For
///       general use call `pantr.change_basis.compute_bernstein_to_cardinal_1d`.
template <std::floating_point T>
void bernstein_to_cardinal_1d(int degree, std::span<const T> quad_points,
                              std::span<const T> quad_weights, span2d<T> out) {
    const auto bern = detail::tabulated<T>(
        [](int d, std::span<const T> p, span2d<T> o) { tabulate_bernstein_1d<T>(d, p, o); },
        degree, quad_points);
    const auto card = detail::tabulated<T>(
        [](int d, std::span<const T> p, span2d<T> o) {
            tabulate_cardinal_bspline_1d<T>(d, p, o);
        },
        degree, quad_points);
    detail::write_out<T>(detail::gram_projection<T>(bern, card, quad_weights), out);
}

/// Cardinal-to-Bernstein: the inverse of the above, by one LU solve.
///
/// \param degree Polynomial degree. Must be non-negative.
/// \param quad_points Quadrature nodes on `[0, 1]`.
/// \param quad_weights Quadrature weights, same length as `quad_points`.
/// \param out Output view of shape `(degree + 1, degree + 1)`.
///
/// \note No input validation is performed, and the degree domain is Layer 2's.
///       Layer 3; see CLAUDE.md. For general use call
///       `pantr.change_basis.compute_cardinal_to_bernstein_1d`.
template <std::floating_point T>
void cardinal_to_bernstein_1d(int degree, std::span<const T> quad_points,
                              std::span<const T> quad_weights, span2d<T> out) {
    const auto n = static_cast<std::size_t>(degree + 1);
    detail::eigen_matrix<T> forward(static_cast<Eigen::Index>(n),
                                    static_cast<Eigen::Index>(n));
    const span2d<T> forward_view(forward.data(), n, n);
    bernstein_to_cardinal_1d<T>(degree, quad_points, quad_weights, forward_view);
    detail::write_out<T>(detail::solve_against_identity<T>(forward), out);
}

/// Legendre-to-cardinal: a Gram projection whose Gram matrix is the identity up
/// to round-off, since the new basis is orthonormal and the quadrature is exact
/// for the products formed.
///
/// \param degree Polynomial degree. Must be non-negative.
/// \param quad_points Quadrature nodes on `[0, 1]`.
/// \param quad_weights Quadrature weights, same length as `quad_points`.
/// \param out Output view of shape `(degree + 1, degree + 1)`.
///
/// \note No input validation is performed. Layer 3; see CLAUDE.md. For general
///       use call `pantr.change_basis.compute_legendre_to_cardinal_1d`.
template <std::floating_point T>
void legendre_to_cardinal_1d(int degree, std::span<const T> quad_points,
                             std::span<const T> quad_weights, span2d<T> out) {
    const auto leg = detail::tabulated<T>(
        [](int d, std::span<const T> p, span2d<T> o) { tabulate_legendre_1d<T>(d, p, o); },
        degree, quad_points);
    const auto card = detail::tabulated<T>(
        [](int d, std::span<const T> p, span2d<T> o) {
            tabulate_cardinal_bspline_1d<T>(d, p, o);
        },
        degree, quad_points);
    detail::write_out<T>(detail::gram_projection<T>(leg, card, quad_weights), out);
}

/// Cardinal-to-Legendre: the inverse of the above, by one LU solve.
///
/// The oracle notes that the obvious alternative -- a second Gram projection with
/// the evaluators swapped -- is equivalent in exact arithmetic and measurably
/// worse in floating point, because the cardinal Gram matrix has the square of
/// this one's condition number. The port keeps the solve for the same reason.
///
/// \param degree Polynomial degree. Must be non-negative.
/// \param quad_points Quadrature nodes on `[0, 1]`.
/// \param quad_weights Quadrature weights, same length as `quad_points`.
/// \param out Output view of shape `(degree + 1, degree + 1)`.
///
/// \note No input validation is performed, and the degree domain is Layer 2's.
///       Layer 3; see CLAUDE.md. For general use call
///       `pantr.change_basis.compute_cardinal_to_legendre_1d`.
template <std::floating_point T>
void cardinal_to_legendre_1d(int degree, std::span<const T> quad_points,
                             std::span<const T> quad_weights, span2d<T> out) {
    const auto n = static_cast<std::size_t>(degree + 1);
    detail::eigen_matrix<T> forward(static_cast<Eigen::Index>(n),
                                    static_cast<Eigen::Index>(n));
    const span2d<T> forward_view(forward.data(), n, n);
    legendre_to_cardinal_1d<T>(degree, quad_points, quad_weights, forward_view);
    detail::write_out<T>(detail::solve_against_identity<T>(forward), out);
}

/// The cardinal-dual L2 functionals in the Legendre basis: the transpose of
/// `cardinal_to_legendre_1d`.
///
/// \param degree Polynomial degree. Must be non-negative.
/// \param quad_points Quadrature nodes on `[0, 1]`.
/// \param quad_weights Quadrature weights, same length as `quad_points`.
/// \param out Output view of shape `(degree + 1, degree + 1)`.
///
/// \note No input validation is performed, and the degree domain is Layer 2's.
///       Layer 3; see CLAUDE.md. For general use call
///       `pantr.change_basis.compute_cardinal_dual_legendre_coeffs_1d`.
template <std::floating_point T>
void cardinal_dual_legendre_coeffs_1d(int degree, std::span<const T> quad_points,
                                      std::span<const T> quad_weights, span2d<T> out) {
    const auto n = static_cast<std::size_t>(degree + 1);
    detail::eigen_matrix<T> straight(static_cast<Eigen::Index>(n),
                                     static_cast<Eigen::Index>(n));
    const span2d<T> straight_view(straight.data(), n, n);
    cardinal_to_legendre_1d<T>(degree, quad_points, quad_weights, straight_view);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            at(out, i, j) = straight(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i));
        }
    }
}

/// Monomial-to-Bernstein on `[0, 1]`: `M[i, j] = C(i, j) / C(degree, j)` for
/// `j <= i`, and zero above the diagonal.
///
/// The binomials are built as a Pascal triangle by **addition**, not by the
/// multiplicative recurrence. Addition keeps every entry an exact integer while
/// the values stay at or below `2^53`, so the quotient is then the correctly
/// rounded value of an exact rational -- which is precisely what the oracle
/// computes, since Python's `math.comb` is an exact integer and its `/` rounds
/// the exact quotient once. The multiplicative form would overflow that range in
/// an intermediate even where the result does not.
///
/// The exactness holds while `C(degree, degree/2) <= 2^53`, i.e. through degree
/// 56. Past it both backends round, and they may round differently; the module
/// places no domain limit on this builder, so that is stated rather than
/// enforced here.
///
/// \param degree Polynomial degree. Must be non-negative.
/// \param out Output view of shape `(degree + 1, degree + 1)`, written in full.
///
/// \note No input validation is performed. Layer 3; see CLAUDE.md. For general
///       use call `pantr.change_basis.compute_monomial_to_bernstein_1d`.
template <std::floating_point T>
void monomial_to_bernstein_1d(int degree, span2d<T> out) {
    using Acc = double;

    const auto n = static_cast<std::size_t>(degree + 1);
    std::vector<Acc> triangle(n * n, Acc(0));
    for (std::size_t i = 0; i < n; ++i) {
        triangle[i * n] = Acc(1);
        for (std::size_t j = 1; j <= i; ++j) {
            triangle[i * n + j] = triangle[(i - 1) * n + j - 1] +
                                  (j <= i - 1 ? triangle[(i - 1) * n + j] : Acc(0));
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            at(out, i, j) =
                j <= i ? static_cast<T>(triangle[i * n + j] / triangle[(n - 1) * n + j]) : T(0);
        }
    }
}

}  // namespace pantr
