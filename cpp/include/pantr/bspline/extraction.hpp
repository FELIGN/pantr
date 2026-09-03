#pragma once

/// \file
/// The Bézier extraction operators of a 1D B-spline space, and the mask that says
/// which of them are already the identity.
///
/// The C++ twin of `pantr.bspline._bspline_extraction`'s Bézier half.
/// `design/extraction_port.md` calls this slice **S3**, minus the two targets it
/// leaves for later: **Lagrange** is this operator post-multiplied by
/// `lagrange_to_bernstein_1d`, which `change_basis.hpp` already provides, and
/// **cardinal** additionally needs the cardinal-interval scan, which is not ported
/// and which `bspline/space_1d.hpp` deliberately excludes from the type.
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

}  // namespace pantr::bspline
