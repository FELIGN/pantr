#pragma once

/// \file
/// The Bézier extraction operators of a 1D B-spline space, and the mask that says
/// which of them are already the identity.
///
/// The C++ twin of `pantr.bspline._bspline_extraction`'s Bézier and Lagrange
/// halves. `design/extraction_port.md` calls this slice **S3**, minus the one
/// target it still leaves for later: **cardinal** needs the cardinal-interval scan,
/// which is not ported and which `bspline/space_1d.hpp` deliberately excludes from
/// the type.
///
/// The Lagrange operator is the Bézier one post-multiplied by the
/// Lagrange-to-Bernstein matrix. That matrix is an **argument** here rather than
/// something this file builds: `change_basis.hpp` already owns it, the variant that
/// selects its nodes is resolved on the Python side and never crosses the seam
/// (`design/cross_backend_types.md`), and passing the finished matrix is what makes
/// it common mode between the two backends so its own parity is
/// `tests/parity/test_change_basis.py`'s question and not this file's.
///
/// ## What an operator is
///
/// For interval `e` the operator `C_e` of shape `(p+1, p+1)` satisfies
///
///     N_e(x) = C_e @ B(xi)
///
/// where `N_e` are the `p + 1` B-spline functions supported on `e`, `B` the
/// Bernstein basis of degree `p` on the reference interval `[0, 1]`, and `xi` the
/// local coordinate. It is built by starting from the identity and running Boehm
/// knot insertion until each interval's knots are `p + 1`-fold, which makes every
/// entry a product of convex combinations: the entries are non-negative and each
/// **column** sums to one. Columns and not rows -- the identity follows from
/// `sum_i N_i = 1` and `sum_j B_j = 1` plus the linear independence of `B`, so it
/// says `ones^T C_e = ones^T`. `tests/parity/test_bspline_bezier_extraction.py`
/// uses it as one of its two independent accuracy oracles.
///
/// The **Lagrange** operator `A_e = C_e L` satisfies `N_e(x) = A_e @ Lag(xi)` for
/// the Lagrange basis of the same degree on the reference interval, `L` being
/// `L[j, k] = B_j(x_k)`. Two properties follow and both are used as oracles:
/// because the Lagrange basis is cardinal at its own nodes, **column `k` of `A_e`
/// is the vector of B-spline functions evaluated at node `k`** mapped into the
/// interval; and because every node lies in `[0, 1]` where the Bernstein basis is
/// non-negative and sums to one, `L` is column-stochastic, so `A_e` is a product of
/// two column-stochastic matrices and is itself column-stochastic. The second one
/// contradicts `design/extraction_port.md`, which said the Lagrange-to-Bernstein
/// matrix has negative entries; it does not, and the doc carries the correction.
///
/// ## The accumulator is `T`, and that is a promise rather than a detail
///
/// Every arithmetic step below is in `T`: the insertion weight, its complement,
/// and the two products that combine two columns. The oracle is a numba kernel
/// over a `T` array whose only scalar literal is spelled `knots.dtype.type(1.0)`
/// precisely so that it does **not** promote -- a bare `1.0` would make numba
/// compute the whole combination in `float64` at `float32` storage.
/// `design/backend_parity.md` Rule 9 is the rule this obeys; `T(1.0)` below is
/// where it is obeyed, and widening it would be more accurate and would not be the
/// same function.
///
/// ## The two knot windows, and why one of them is not the class multiplicity
///
/// The builder reads two counts off the knot vector, and they are different
/// quantities computed by different rules:
///
///  - the **boundary count**, `multiplicity_of_first_knot_in_domain`, which
///    decides whether the first interval needs the insertions a non-clamped left
///    end omits. It counts only among the first `degree + 1` knots.
///  - the **per-interval multiplicities**, the in-domain subrange of
///    `unique_knots_and_multiplicity`, which advance the sliding knot window.
///
/// `design/extraction_port.md`'s 2026-09-01 amendment says the first is the front
/// entry of the second and needs no port of its own. **That is false**, and a
/// counterexample is in `multiplicity_of_first_knot_in_domain`'s own comment: the
/// two disagree whenever the first in-domain knot is repeated. The doc carries an
/// amendment recording it.
///
/// ## The window never runs off the end, and that is derived rather than assumed
///
/// The second loop reads `knots[w .. w + degree]` with
/// `w = degree + sum of the multiplicities of in-domain classes 1..e`. Writing
/// `b0` for the last knot index of the class holding `knots[degree]`, that sum
/// telescopes to `w = l_e + (degree - b0)` where `l_e` is the last knot index of
/// in-domain class `e`. Since `degree <= b0` by construction, `w <= l_e`, and for
/// `e <= n_intervals - 1` the class `e + 1` still exists, so
/// `l_e <= a_last - 1 <= n - degree - 2` where `a_last` is the first index of the
/// last in-domain class, which holds `n - degree - 1`. Hence
/// `w + degree <= n - 2 < n`. The window fits for every interval, contracting or
/// skipped.
///
/// ## Two inputs the oracle does not defend against, and what happens here
///
/// Both are reachable only by calling the oracle's Layer 2 helper directly with a
/// knot vector `pantr.bspline.BsplineSpace1D` would refuse, or -- for the second --
/// by a vector the space accepts and the oracle then mishandles.
///
///  - **A knot vector spanning no in-domain interval.** The oracle allocates an
///    empty `(0, p+1, p+1)` result and then indexes `out[0]` if the boundary count
///    is short, which is out of bounds on an empty array -- measured, not inferred:
///    interpreted, where numpy bounds checks, that line raises. Here the operator
///    count is a precondition, and `cpp/bindings/bspline_extraction_operators.cpp`
///    refuses the vector with `check_space_has_an_interval`'s message rather than
///    reaching it.
///  - **A first in-domain knot whose multiplicity exceeds one.** The window
///    derivation above is exact, but `w = l_e + (degree - b0)` is then strictly
///    below `l_e`, so the window starts inside the repeated knot and the oracle
///    divides zero by zero. This code reproduces that, because reproducing the
///    oracle is what it is for; the divergence is a defect in the shared algorithm
///    rather than in either backend, and the parity sweep excludes such vectors and
///    says so.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pantr/bspline/knots.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/precondition.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::bspline {

/// Build the Bézier extraction operator of every in-domain interval.
///
/// \tparam T Scalar type of the knots and of the operators.
/// \param knots A non-decreasing knot vector of at least `2 * degree + 2` entries.
/// \param degree The polynomial degree, non-negative.
/// \param tol The absolute parametric tolerance, from `knot_tolerance` or from a
///        space's `tolerance()`.
/// \param out The operators, shape `(n_intervals, degree + 1, degree + 1)`, where
///        `n_intervals` is `classify_knots(knots, degree, tol).num_intervals()`.
///        Overwritten in full.
///
/// \note Inputs are assumed to be correct (no validation performed). For general
///       use, call `pantr.bspline._bspline_extraction`'s
///       `_tabulate_Bspline_Bezier_1D_extraction_impl`, whose C++ half is
///       `cpp/bindings/bspline_extraction_operators.cpp`.
template <Real T>
void bezier_extraction_1d(std::span<const T> knots, std::int64_t degree, double tol,
                          span_nd<T, 3> out) {
    const auto p = static_cast<std::size_t>(degree);
    const KnotClasses<T> classes = unique_knots_and_multiplicity(knots, degree, tol);
    const std::span<const std::int64_t> multiplicity =
        std::span<const std::int64_t>(classes.multiplicity)
            .subspan(classes.domain_begin, classes.domain_end - classes.domain_begin);
    const std::size_t n_intervals = multiplicity.size() - 1;

    PANTR_PRECONDITION(n_intervals >= 1, "the knot vector must span at least one interval");
    PANTR_PRECONDITION(out.extent(0) == n_intervals, "out needs one operator per interval");
    PANTR_PRECONDITION(out.extent(1) == p + 1 && out.extent(2) == p + 1,
                       "each operator is (degree + 1) square");

    const T one(1.0);
    const T zero(0.0);
    for (std::size_t e = 0; e < n_intervals; ++e) {
        for (std::size_t i = 0; i <= p; ++i) {
            for (std::size_t j = 0; j <= p; ++j) {
                at(out, e, i, j) = (i == j) ? one : zero;
            }
        }
    }

    // A left end the knot vector does not clamp leaves the first interval short of
    // the insertions that make it a Bézier patch. `reg` of them are owed, one per
    // missing knot.
    const std::int64_t boundary = multiplicity_of_first_knot_in_domain<T>(knots, degree, tol);
    if (boundary < degree + 1) {
        const auto reg = static_cast<std::size_t>(degree - boundary);
        const T t = knots[p];
        for (std::size_t r = 0; r < reg; ++r) {
            // The oracle slices `lcl_knots = knots[r:]`, so its `lcl_knots[k]` is
            // `knots[r + k]` and its `lcl_knots[k + degree - r]` is `knots[k + degree]`.
            for (std::size_t k = 1; k + r < p; ++k) {
                const T alpha = (t - knots[r + k]) / (knots[k + p] - knots[r + k]);
                const T beta = one - alpha;
                for (std::size_t i = 0; i <= p; ++i) {
                    at(out, 0, i, k - 1) =
                        alpha * at(out, 0, i, k) + beta * at(out, 0, i, k - 1);
                }
            }
        }
    }

    // One insertion weight per knot still owed by the interval being closed.
    // `degree - 1` is the largest count any interval can owe, since an in-domain
    // interior class has multiplicity at least one and a class of `degree` or more
    // needs no insertion at all.
    std::vector<T> alphas(p > 0 ? p - 1 : 0, zero);

    std::size_t window = p;
    std::int64_t mult = 0;
    for (std::size_t e = 0; e < n_intervals; ++e) {
        window += static_cast<std::size_t>(mult);
        mult = multiplicity[e + 1];

        if (mult >= degree) {
            continue;  // already a Bézier patch on the right: nothing to insert
        }
        const auto m = static_cast<std::size_t>(mult);
        const T first_gap = knots[window + 1] - knots[window];
        for (std::size_t i = 0; i + m < p; ++i) {
            alphas[i] = first_gap / (knots[window + m + 1 + i] - knots[window]);
        }

        const std::size_t reg = p - m;
        for (std::size_t r = 1; r <= reg; ++r) {
            const std::size_t s = m + r;
            // Descending, so column `k - 1` is still the value this stage reads.
            for (std::size_t k = p; k + 1 > s; --k) {
                const T alpha = alphas[k - s];
                const T beta = one - alpha;
                for (std::size_t i = 0; i <= p; ++i) {
                    at(out, e, i, k) = alpha * at(out, e, i, k) + beta * at(out, e, i, k - 1);
                }
            }

            // The closed columns of this interval seed the next one's, which is a
            // copy and commits no arithmetic.
            if (e + 1 < n_intervals) {
                for (std::size_t i = 0; i <= r; ++i) {
                    at(out, e + 1, reg - r + i, reg - r) = at(out, e, p - r + i, p);
                }
            }
        }
    }
}

/// Build the Lagrange extraction operator of every in-domain interval.
///
/// `A_e = C_e L`, with `C_e` the Bézier operator of interval `e` and `L` the
/// Lagrange-to-Bernstein matrix of the same degree, so that `N_e(x) = A_e @ Lag(xi)`.
/// The product is formed in place, one row at a time, over a scratch copy of the
/// row being overwritten.
///
/// **The accumulator is `T`.** The oracle contracts this product with
/// :func:`numpy.matmul`, whose BLAS reaches `sgemm` at `float32` and accumulates in
/// the storage format, and `design/backend_parity.md` Rule 9 makes reproducing that
/// width part of the contract rather than an implementation detail. The *summation
/// order* is not part of it: numpy's is unspecified and blocked, so the two backends
/// are bounded rather than bitwise here, which is the one place the Lagrange target
/// differs in kind from the Bézier one.
///
/// **The width is checked here rather than by the parity suite, and no bound could do
/// it.** The parity bound is derived from each backend's forward error against the exact
/// product. A `double` accumulator has a *smaller* forward error than a `T` one, so it
/// sits inside the same bound by construction: any claim built that way admits a more
/// accurate backend, which is what `design/backend_parity.md` Rule 8 records as
/// something a bound deliberately does not license. Nothing about the constant would
/// change it. Measured alongside the argument, so the two agree: mutating this line to a
/// `double` accumulator leaves the whole Python parity file green, and the gap it opens
/// at `float32` is one unit in the last place, the same size as the two backends' own
/// disagreement. `cpp/tests/test_bspline_extraction.cpp`'s
/// `check_the_accumulator_is_the_storage_type` is the test that fails instead.
///
/// \tparam T Scalar type of the knots, of the matrix and of the operators.
/// \param knots A non-decreasing knot vector of at least `2 * degree + 2` entries.
/// \param degree The polynomial degree, non-negative.
/// \param tol The absolute parametric tolerance, from `knot_tolerance` or from a
///        space's `tolerance()`.
/// \param lagrange_to_bernstein The `(degree + 1, degree + 1)` matrix
///        `L[j, k] = B_j(x_k)`, from `pantr::lagrange_to_bernstein_1d`. Must not
///        overlap `out`: the product is formed in place over a scratch copy of the
///        row being overwritten, which makes `out` safe against itself and nothing
///        else. Both callers pass a cached matrix that owns its own storage.
/// \param out The operators, shape `(n_intervals, degree + 1, degree + 1)`.
///        Overwritten in full.
///
/// \note Inputs are assumed to be correct (no validation performed). For general
///       use, call `pantr.bspline._bspline_extraction`'s
///       `_tabulate_Bspline_Lagrange_1D_extraction_impl`, whose C++ half is
///       `cpp/bindings/bspline_extraction_operators.cpp`.
template <Real T>
void lagrange_extraction_1d(std::span<const T> knots, std::int64_t degree, double tol,
                            span2d<const T> lagrange_to_bernstein, span_nd<T, 3> out) {
    const auto p = static_cast<std::size_t>(degree);
    PANTR_PRECONDITION(lagrange_to_bernstein.extent(0) == p + 1
                           && lagrange_to_bernstein.extent(1) == p + 1,
                       "the change-of-basis matrix is (degree + 1) square");

    bezier_extraction_1d<T>(knots, degree, tol, out);

    std::vector<T> row(p + 1, T(0.0));
    for (std::size_t e = 0; e < out.extent(0); ++e) {
        for (std::size_t i = 0; i <= p; ++i) {
            for (std::size_t k = 0; k <= p; ++k) {
                row[k] = at(out, e, i, k);
            }
            for (std::size_t j = 0; j <= p; ++j) {
                T acc(0.0);
                for (std::size_t k = 0; k <= p; ++k) {
                    acc += row[k] * at(lagrange_to_bernstein, k, j);
                }
                at(out, e, i, j) = acc;
            }
        }
    }
}

/// Mark the intervals whose Bézier extraction operator is the identity.
///
/// Interval `e` is one exactly when both of its bounding in-domain knots have
/// multiplicity at least `degree + 1`: the interval is then already a Bézier patch,
/// decoupled from both neighbours, and no insertion touches it.
///
/// Integers throughout, so there is no tolerance here and no scalar type to be
/// generic in. The tolerance already did its work upstream, in the class scan that
/// produced `multiplicity`.
///
/// \param multiplicity The in-domain knot multiplicities,
///        `BsplineSpace1D::multiplicity_in_domain()`, of length `n_intervals + 1`.
/// \param degree The polynomial degree.
/// \param out One flag per interval, length `multiplicity.size() - 1`.
///
/// \note Inputs are assumed to be correct (no validation performed). For general
///       use, call `pantr.bspline.spanwise_element_extraction`'s
///       `_bezier_structural_identity_mask`.
inline void bezier_structural_identity_mask(std::span<const std::int64_t> multiplicity,
                                            std::int64_t degree, std::span<bool> out) {
    PANTR_PRECONDITION(!multiplicity.empty(), "multiplicity must hold at least one class");
    const std::int64_t threshold = degree + 1;
    const std::size_t n_intervals = multiplicity.size() - 1;
    PANTR_PRECONDITION(out.size() == n_intervals, "out needs one flag per interval");
    for (std::size_t e = 0; e < n_intervals; ++e) {
        out[e] = multiplicity[e] >= threshold && multiplicity[e + 1] >= threshold;
    }
}


/// Mark the intervals whose Lagrange extraction operator is the identity.
///
/// The predicate is the Bézier mask when `L` is the identity, and all-false
/// otherwise, which is the oracle's rule. **The all-false half is a claim and it is
/// provable.** `A_e = C_e L = I` needs `C_e = L^{-1}`; `C_e` is entrywise
/// non-negative, so `L^{-1}` would have to be too, and a non-negative matrix with a
/// non-negative inverse is monomial. Both `C_e` and `L` are column-stochastic, so
/// the monomial matrix is a permutation. `L[j, k] = B_j(x_k)` is a permutation only
/// if every node makes one Bernstein function `1` and the rest `0`, which on `[0, 1]`
/// only `x = 0` and `x = 1` do; so `degree <= 1`, and with the ascending nodes every
/// variant here produces, that permutation is the identity. Hence `A_e = I` forces
/// `L = I`, and the else-branch is sound. Measured, `L` is the identity for the
/// equispaced, Gauss-Lobatto-Legendre and second-kind Chebyshev families at degree 1
/// and for no other family or degree tried.
///
/// The comparison against the identity is **exact**: `L`'s entries are the ones the
/// caller holds, the identity's are `0` and `1`, and a matrix that misses the
/// identity by an ulp yields a non-identity operator, which is what the mask is
/// asked about. `design/backend_parity.md` Rule 11's distinction -- a verdict is not
/// a displaced value, and there is no tolerance to apply to it.
///
/// **Degree 0 does not reach here.** `compute_lagrange_to_bernstein_1d` refuses a
/// degree below 1, so there is no matrix to pass and the oracle's Layer 2 answers
/// all-true without one; this function is therefore never called at `degree == 0`,
/// and reproducing that short-circuit here would be inventing a matrix nobody built.
///
/// \tparam T Scalar type of the change-of-basis matrix.
/// \param multiplicity The in-domain knot multiplicities,
///        `BsplineSpace1D::multiplicity_in_domain()`, of length `n_intervals + 1`.
/// \param degree The polynomial degree.
/// \param lagrange_to_bernstein The `(degree + 1, degree + 1)` matrix.
/// \param out One flag per interval, length `multiplicity.size() - 1`.
///
/// \note Inputs are assumed to be correct (no validation performed). For general
///       use, call `pantr.bspline.spanwise_element_extraction`'s
///       `_lagrange_structural_identity_mask`.
template <Real T>
void lagrange_structural_identity_mask(std::span<const std::int64_t> multiplicity,
                                       std::int64_t degree,
                                       span2d<const T> lagrange_to_bernstein,
                                       std::span<bool> out) {
    const auto p = static_cast<std::size_t>(degree);
    PANTR_PRECONDITION(lagrange_to_bernstein.extent(0) == p + 1
                           && lagrange_to_bernstein.extent(1) == p + 1,
                       "the change-of-basis matrix is (degree + 1) square");
    PANTR_PRECONDITION(!multiplicity.empty(), "multiplicity must hold at least one class");
    PANTR_PRECONDITION(out.size() == multiplicity.size() - 1,
                       "out needs one flag per interval");

    bool is_identity = true;
    for (std::size_t j = 0; j <= p && is_identity; ++j) {
        for (std::size_t k = 0; k <= p; ++k) {
            const T want = (j == k) ? T(1.0) : T(0.0);
            if (at(lagrange_to_bernstein, j, k) != want) {
                is_identity = false;
                break;
            }
        }
    }

    if (is_identity) {
        bezier_structural_identity_mask(multiplicity, degree, out);
        return;
    }
    for (std::size_t e = 0; e < out.size(); ++e) {
        out[e] = false;
    }
}

}  // namespace pantr::bspline
