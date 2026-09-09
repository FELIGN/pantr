#pragma once

/// \file
/// Tabulation of the 1D B-spline basis and its derivatives: Piegl & Tiller A2.2
/// and A2.3 over a general knot vector.
///
/// Port of `_basis_funcs_point`, `_basis_derivs_point`,
/// `_find_span_and_first_basis_point` and the four batch kernels that dispatch to
/// them in `src/pantr/bspline/_bspline_basis_core.py`, which stays as the parity
/// oracle.
///
/// ## One of the free-function ports `space_1d.hpp` names
///
/// `pantr/bspline/space_1d.hpp` states that `BsplineSpace1D` owns no *operations*
/// and that basis tabulation is a separate port over free functions taking a
/// `const BsplineSpace1D&`. This is that header, and it is two layers rather than
/// one, mirroring the oracle:
///
///  - `basis_funcs_1d` and `basis_derivs_1d` are **Layer 3**. They take the knot
///    vector and the degree, validate nothing, and are the general-knot recurrence
///    and nothing else. `cpp/bindings/bspline_basis.cpp` calls these, because the
///    dispatch above them is a decision the *oracle* makes and parity is a claim
///    about reproducing it.
///  - `tabulate_basis_1d` and `tabulate_basis_derivatives_1d` take the space and
///    are the **Layer 2 equivalent**: they choose between the Bézier-like fast
///    path and the general recurrence exactly as
///    `_tabulate_Bspline_basis_1D_impl` does. They are the entry point for a C++
///    caller with no interpreter, and they are deliberately *not* bound to
///    Python -- see "Why the dispatch stays on the Python side" below.
///
/// ## What is reused rather than rewritten
///
/// The Bézier-like fast path is Bernstein evaluation plus a change of variable, so
/// it calls `pantr/basis/bernstein.hpp` -- `tabulate_bernstein_1d` for the values
/// and `tabulate_bernstein_deriv_1d` for the derivatives -- and there is no second
/// Bernstein recurrence anywhere in this file. `num_basis` comes from
/// `pantr/bspline/knots.hpp`. Nothing here recomputes a quantity the space
/// already holds.
///
/// ## The accumulation width, which was measured and not read
///
/// `design/backend_parity.md` Rule 9: an oracle's accumulation width is a
/// per-kernel fact and cannot be read off the source. **These kernels do not use
/// `accumulator_t`**, and that is the finding rather than an oversight.
/// `_basis_funcs_point` opens with `dtype = knots.dtype` and allocates `left`,
/// `right` and its zero/one constants in it, so at `float32` every intermediate is
/// `float32` -- unlike `cardinal_bspline.hpp`, where numba's literal typing
/// promotes and `pantr/core/scalar.hpp`'s policy note says a port must widen to
/// match. Widening here is the same error in the other direction.
///
/// `scripts/measure_bspline_tabulation_widths.py` measures every site
/// behaviourally, two rival transliterations against the kernel, and reports how
/// often the two models disagree so a match cannot come from a check that could
/// not fail. What it found, at `float32`:
///
/// | site | narrow model | wide model | models differ |
/// |---|---|---|---|
/// | A2.2, all intermediates | reproduces every value | fails ~60% | yes, on 61% |
/// | A2.3, all intermediates | reproduces every value | fails ~65% | yes |
/// | A2.3 step 3, factorial scaling | fails on 11% | **reproduces every value** | yes, on 11% |
///
/// So the one site that widens is the factorial scaling, and it widens because
/// `float32 * int64` is `float64` under numba's promotion: the product is formed in
/// `double` and rounded once on the store. Written out below as
/// `T(double(value) * double(fac))` rather than `value * T(fac)`, which is the
/// natural C++ and disagrees.
///
/// Two of those three rows are invisible at `float64`, where all the models
/// coincide.
///
/// **What the width discipline is actually about, which a mutation test settled.**
/// Widening a *single* operation is unobservable: double rounding from binary64 to
/// binary32 is innocuous for `+`, `-`, `*`, `/` and `sqrt`, since binary64 carries
/// more than `2p + 2` bits of binary32's significand, so
/// `static_cast<float>(double(a) / double(b))` is bit-for-bit `a / b` in `float`.
/// Mutating the division that way left every parity case passing. What moves values is
/// a **carried** wide accumulator: holding `saved` in a `double` across the inner loop
/// failed 10 of 15 `float32` value cases. So the rule is about the width the running
/// state is *stored* at between steps, not about the width each operation is evaluated
/// at -- which is why `bernstein.hpp` reads its recurrence back out of `out` on
/// purpose, and why that is the same rule rather than a different one.
///
/// ## The factorial accumulator wraps, in both backends
///
/// Step 3 of A2.3 scales the k-th derivative row by `degree!/(degree-k)!`,
/// accumulated in an integer. numba's is an int64 and wraps; a C++ `std::int64_t`
/// would be *undefined behaviour* instead, so `pantr::core::wrapping_mul` is used --
/// see `pantr/core/binomial.hpp`, which explains why that file carries both this
/// answer and the opposite one.
///
/// Measured: the lowest wrapping degree is **21**, first at derivative order 19.
/// Above it the scaling is a different number in both backends and the derivative
/// values are wrong in both, identically -- which is why parity is claimed there
/// and *accuracy* is not. `tests/parity/test_bspline_basis_tabulation.py` names the
/// excluded degrees, as `design/backend_parity.md` Rule 8 requires.
///
/// ## Where the two backends can disagree
///
/// **Both recurrences contain `a * b + c` sites** -- `N[r] = saved + right[r+1] *
/// temp` in A2.2, `d += a[s2,j] * ndu[...]` in A2.3 -- so unlike
/// `bernstein.hpp` these are fusable, and `-ffp-contract=on` on a target with a
/// fused multiply-add moves values. Rule 10 gives the budget; the parity test
/// carries the conditional claim and takes its bitwise arm on the shipped build,
/// whose target ISA has no FMA.
///
/// **No transcendental appears here.** The general-knot path is `+`, `-`, `*`, `/`
/// and an integer comparison, so IEEE 754 pins every result and the interpreted
/// oracle of Rule 12 reproduces it by guarantee. The Bézier-like path inherits
/// `pow` from `bernstein.hpp` and the falling factorial from `bernstein_deriv`, and
/// both of those need the Rule 12 gates.
///
/// ## Why the dispatch stays on the Python side
///
/// `tabulate_basis_1d(space, ...)` would be the obvious thing for
/// `cpp/bindings/bspline_basis.cpp` to call, and it is not what it calls. Building a
/// `BsplineSpace1D` per tabulation call would re-validate the knot vector, copy it
/// and fill the derived block, which is `O(num_knots)` work in front of a kernel a
/// per-cell FEM assembly calls with two or three points -- exactly the batch size
/// the oracle keeps a serial twin for. Worse, the constructor *snaps* by default,
/// so the C++ side could evaluate on a different knot vector than the oracle did.
///
/// So the binding crosses plain data, the Python adapter in
/// `pantr.bspline._basis_backend` keeps the oracle's own dispatch, and the
/// space-level functions here serve the C++ caller. That leaves the dispatch
/// written twice, which is a real cost and is recorded rather than hidden;
/// `cpp/tests/test_bspline_tabulation.cpp` checks the C++ copy against the same
/// invariants the parity suite checks the Python copy against.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pantr/basis/bernstein.hpp"
#include "pantr/core/binomial.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/precondition.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::bspline {

/// A point's knot span and the global index of the first basis function over it.
///
/// Attributes are the two quantities `_find_span_and_first_basis_point` returns as
/// a tuple. A named pair rather than a `std::pair`, because `.first` and `.second`
/// at the call site say nothing about which is which.
struct SpanAndFirstBasis {
    /// The clamped index of the last knot less than or equal to the point.
    std::int64_t span;
    /// The global index of the first basis function supported at the point.
    std::int64_t first_basis;
};

/// Locate a point's knot span and its first supported basis function.
///
/// Port of `_find_span_and_first_basis_point`. Tier B throughout: the search and the
/// clamps are functions of the point's *value*, so every comparison passes through
/// `value_of`.
///
/// The two clamps on the *span* are the oracle's and both are load-bearing. The upper
/// one holds the span at the last in-domain one, because for a knot vector that is not
/// clamped at the right end the domain's last knot is not the vector's last knot, and
/// an unclamped search can place a point in a span whose Cox-de Boor window reads past
/// the vector. The lower one holds it at `degree`, which is what keeps `span + 1 - j`
/// non-negative for every `j` in the recurrence.
///
/// **The third clamp, on the non-periodic first-basis index, is unreachable, and the
/// oracle's comment claiming otherwise is wrong.** It is kept anyway; see the note on
/// the `std::min` below for both halves of that.
///
/// \tparam T Scalar type of the knots and the point.
/// \param knots The knot vector, non-decreasing.
/// \param degree The polynomial degree.
/// \param periodic Whether the space is periodic. A periodic space keeps the
///        unclamped first-basis index, which its evaluation loop wraps modulo the
///        number of control points.
/// \param point The evaluation point.
/// \return The clamped span and the first-basis index.
///
/// \note No input validation is performed. `knots.size() >= 2 * degree + 2` and
///       `degree >= 0` are **memory-safety** obligations and carry the assertion
///       below; the answer for a point outside the domain is the clamped span, which
///       is the oracle's answer too. A non-finite point yields an unspecified span
///       within the clamped range, never an out-of-range one.
template <Real T>
[[nodiscard]] SpanAndFirstBasis find_span_and_first_basis(std::span<const T> knots,
                                                          std::int64_t degree, bool periodic,
                                                          const T& point) {
    PANTR_PRECONDITION(degree >= 0, "degree must be non-negative");
    PANTR_PRECONDITION(static_cast<std::int64_t>(knots.size()) >= 2 * degree + 2,
                       "knots must have at least 2*degree+2 elements");
    using pantr::value_of;

    const auto value = value_of(point);
    const auto* const upper =
        std::upper_bound(knots.data(), knots.data() + knots.size(), value,
                         [](auto probe, const T& knot) { return probe < value_of(knot); });
    auto span = static_cast<std::int64_t>(upper - knots.data()) - 1;

    span = std::min(span, static_cast<std::int64_t>(knots.size()) - degree - 2);
    span = std::max(span, degree);

    if (periodic) {
        return {span, span - degree};
    }
    // Non-periodic. `num_basis` is `knots.size() - degree - 1` and needs no tolerance,
    // which is why this header takes none: the oracle threads one through for interface
    // consistency and its own docstrings record that it goes unused.
    //
    // **This `std::min` can never bind, and the oracle's comment says it must.** The
    // oracle's `_find_spans_and_first_basis` explains it as clamping "so the final
    // evaluation point always addresses the last degree + 1 active basis functions",
    // but the span clamp above has already made the two bounds equal:
    //
    //     span - degree      <= (n - degree - 2) - degree = n - 2*degree - 2
    //     num_basis - order   = (n - degree - 1) - (degree + 1) = n - 2*degree - 2
    //
    // so the first is always at most the second, with equality at the right endpoint.
    // Checked as well as derived: over 2163 (degree, knot count, point) combinations
    // the min changed the answer zero times.
    //
    // It is written out rather than dropped, for two reasons. It keeps the two
    // backends structurally identical, so a future change to the span clamp -- which
    // is what makes this redundant -- cannot silently make one of them wrong while the
    // other stays right. And deleting a guard whose stated reason is false, in a port
    // whose job is to reproduce, would be a behaviour decision taken in the wrong
    // place: the oracle is where it should be removed, and it is reported rather than
    // done here.
    const std::int64_t num_basis = static_cast<std::int64_t>(knots.size()) - degree - 1;
    return {span, std::min(span - degree, num_basis - degree - 1)};
}

/// Evaluate the non-zero B-spline basis functions at one point (A2.2's body).
///
/// \tparam T Scalar type of the knots, the point and the output.
/// \param knots The knot vector, non-decreasing.
/// \param degree The polynomial degree.
/// \param span The point's clamped knot span, from `find_span_and_first_basis`.
/// \param point The evaluation point.
/// \param left Scratch of at least `degree + 1` entries. Entry 0 is never read.
/// \param right Scratch of at least `degree + 1` entries. Entry 0 is never read.
/// \param out The `degree + 1` basis values, written in full.
///
/// \note No input validation is performed. This is a Layer 3 kernel. The
///       denominator guard is the textbook one and needs no tolerance: `denom` is a
///       sum of two non-negative knot differences and the two terms it then divides
///       are those same differences, so a small denominator cancels against an
///       equally small numerator; `denom == 0` happens exactly for an empty knot
///       span. The exact-zero test is scale-invariant, so the guard is unaffected by
///       the knot vector's parametric span.
///       For general use call `pantr.bspline.BsplineSpace1D.tabulate_basis`.
template <Real T>
void basis_funcs_point(std::span<const T> knots, std::int64_t degree, std::int64_t span,
                       const T& point, std::span<T> left, std::span<T> right,
                       std::span<T> out) {
    PANTR_PRECONDITION(degree >= 0, "degree must be non-negative");
    PANTR_PRECONDITION(static_cast<std::int64_t>(out.size()) >= degree + 1,
                       "out must hold degree+1 values");
    PANTR_PRECONDITION(static_cast<std::int64_t>(left.size()) >= degree + 1 &&
                           static_cast<std::int64_t>(right.size()) >= degree + 1,
                       "the scratch spans must hold degree+1 values");
    using pantr::value_of;

    const auto order = static_cast<std::size_t>(degree) + 1;
    const T zero(0.0);
    out[0] = T(1.0);

    for (std::size_t j = 1; j < order; ++j) {
        const auto jj = static_cast<std::int64_t>(j);
        left[j] = point - knots[static_cast<std::size_t>(span + 1 - jj)];
        right[j] = knots[static_cast<std::size_t>(span + jj)] - point;
        T saved = zero;

        for (std::size_t r = 0; r < j; ++r) {
            // Non-negative for a non-decreasing knot vector; see the note above.
            const T denom = right[r + 1] + left[j - r];
            const T temp = value_of(denom) == value_of(zero) ? zero : out[r] / denom;
            out[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }

        out[j] = saved;
    }
}

/// Evaluate the non-zero B-spline basis derivatives at one point (A2.3's body).
///
/// \tparam T Scalar type of the knots, the point and the output.
/// \param knots The knot vector, non-decreasing.
/// \param degree The polynomial degree.
/// \param n_deriv Highest derivative order to compute. Rows above `degree` come out
///        identically zero, which is correct: the k-th derivative of a degree-p
///        polynomial vanishes for `k > p`.
/// \param span The point's clamped knot span.
/// \param point The evaluation point.
/// \param ndu Scratch of `(degree + 1) * (degree + 1)` entries, row-major.
/// \param left Scratch of at least `degree + 1` entries.
/// \param right Scratch of at least `degree + 1` entries.
/// \param a Scratch of `2 * (n_deriv + 1)` entries, row-major.
/// \param out Shape `(n_deriv + 1, degree + 1)`, written in full.
///
/// \note No input validation is performed. This is a Layer 3 kernel. The `a` table
///       is zeroed once per point and then **carried across the loop over `r`**,
///       which is the oracle's structure and is reproduced rather than tidied: only
///       `a[0, 0]` is reset per `r`, so zeroing the rest per `r` instead would be a
///       different computation whenever a stale entry is read.
///       For general use call
///       `pantr.bspline.BsplineSpace1D.tabulate_basis_derivatives`.
template <Real T>
void basis_derivs_point(std::span<const T> knots, std::int64_t degree,  // NOLINT
                        std::int64_t n_deriv, std::int64_t span, const T& point, span2d<T> ndu,
                        std::span<T> left, std::span<T> right, span2d<T> a, span2d<T> out) {
    PANTR_PRECONDITION(degree >= 0 && n_deriv >= 0, "degree and n_deriv must be non-negative");
    PANTR_PRECONDITION(out.extent(0) == static_cast<std::size_t>(n_deriv) + 1 &&
                           out.extent(1) == static_cast<std::size_t>(degree) + 1,
                       "out must have shape (n_deriv+1, degree+1)");
    PANTR_PRECONDITION(ndu.extent(0) == static_cast<std::size_t>(degree) + 1 &&
                           ndu.extent(1) == static_cast<std::size_t>(degree) + 1,
                       "ndu must have shape (degree+1, degree+1)");
    PANTR_PRECONDITION(a.extent(0) == 2 && a.extent(1) == static_cast<std::size_t>(n_deriv) + 1,
                       "a must have shape (2, n_deriv+1)");
    using pantr::value_of;

    const auto order = static_cast<std::size_t>(degree) + 1;
    const T zero(0.0);

    // The oracle allocates `ndu` and `a` with `np.zeros` per point. `ndu` is written
    // in full over the range it reads, but `a` is not: see the note above.
    for (std::size_t i = 0; i < order; ++i) {
        for (std::size_t j = 0; j < order; ++j) {
            at(ndu, i, j) = zero;
        }
    }
    for (std::size_t j = 0; j <= static_cast<std::size_t>(n_deriv); ++j) {
        at(a, 0, j) = zero;
        at(a, 1, j) = zero;
    }

    // --- Step 1: the ndu table, A2.2 retaining its intermediates ---
    at(ndu, 0, 0) = T(1.0);
    for (std::size_t j = 1; j < order; ++j) {
        const auto jj = static_cast<std::int64_t>(j);
        left[j] = point - knots[static_cast<std::size_t>(span + 1 - jj)];
        right[j] = knots[static_cast<std::size_t>(span + jj)] - point;
        T saved = zero;
        for (std::size_t r = 0; r < j; ++r) {
            at(ndu, j, r) = right[r + 1] + left[j - r];  // knot differences, lower triangle
            const T denom = at(ndu, j, r);
            const T temp =
                value_of(denom) == value_of(zero) ? zero : at(ndu, r, j - 1) / denom;
            at(ndu, r, j) = saved + right[r + 1] * temp;  // basis values, upper triangle
            saved = left[j - r] * temp;
        }
        at(ndu, j, j) = saved;
    }

    for (std::size_t j = 0; j < order; ++j) {
        at(out, 0, j) = at(ndu, j, static_cast<std::size_t>(degree));
    }

    // --- Step 2: the k-th derivatives, by the triangular recursion ---
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(order); ++r) {
        std::size_t s1 = 0;
        std::size_t s2 = 1;
        at(a, 0, 0) = T(1.0);

        for (std::int64_t k = 1; k <= n_deriv; ++k) {
            T d = zero;
            const std::int64_t rk = r - k;
            const std::int64_t pk = degree - k;

            if (r >= k) {
                at(a, s2, 0) = at(a, s1, 0) / at(ndu, pk + 1, rk);
                d = at(a, s2, 0) * at(ndu, rk, pk);
            }

            const std::int64_t j1 = rk >= -1 ? 1 : -rk;
            const std::int64_t j2 = (r - 1) <= pk ? k - 1 : degree - r;

            for (std::int64_t j = j1; j <= j2; ++j) {
                at(a, s2, j) = (at(a, s1, j) - at(a, s1, j - 1)) / at(ndu, pk + 1, rk + j);
                d = d + at(a, s2, j) * at(ndu, rk + j, pk);
            }

            if (r <= pk) {
                at(a, s2, k) = -at(a, s1, k - 1) / at(ndu, pk + 1, r);
                d = d + at(a, s2, k) * at(ndu, r, pk);
            }

            at(out, k, r) = d;
            std::swap(s1, s2);
        }
    }

    // --- Step 3: the factorial scaling, degree!/(degree-k)! ---
    //
    // The one site in this file whose arithmetic is wider than `T`: numba promotes
    // `float32 * int64` to `float64`, so the product is formed in `double` and rounded
    // once on the store. `value * T(fac)` is the natural C++ and disagrees on 11% of
    // float32 values -- see the file comment's table. `wrapping_mul` reproduces the
    // int64 wrap, which first bites at degree 21.
    std::int64_t fac = degree;
    for (std::int64_t k = 1; k <= n_deriv; ++k) {
        const auto fac_wide = static_cast<double>(fac);
        for (std::size_t j = 0; j < order; ++j) {
            at(out, k, j) = T(static_cast<double>(value_of(at(out, k, j))) * fac_wide);
        }
        fac = core::wrapping_mul(fac, degree - k);
    }
}

/// Tabulate the B-spline basis over a general knot vector (A2.2, batched).
///
/// Port of `_compute_basis_nurbs_book_impl` and its serial twin, which are
/// bit-identical to each other: measured over 369 824 values at both widths, none
/// differs. `design/backend_parity.md` Rule 7 -- the oracle's
/// `_PARALLEL_MIN_NUM_PTS` dispatch is a property of the host it was measured on, so
/// it is **inherited, not re-derived here**, and this side has one kernel that runs
/// on the calling thread at every batch size. That is the same shape
/// `pantr.basis._basis_backend`'s cardinal-B-spline entry already has.
///
/// \tparam T Scalar type of the knots, the points and the output.
/// \param knots The knot vector, non-decreasing.
/// \param degree The polynomial degree.
/// \param periodic Whether the space is periodic.
/// \param points The evaluation points.
/// \param out_basis Shape `(points.size(), degree + 1)`, written in full.
/// \param out_first_basis One entry per point, written in full.
///
/// \note No input validation is performed. This is a Layer 3 kernel. Two obligations
///       are for memory safety and carry assertions: the two output extents, and
///       `knots.size() >= 2 * degree + 2`. Points outside the domain are evaluated in
///       the clamped span rather than refused, as the oracle does with
///       `validate=False`.
///       For general use call `pantr.bspline.BsplineSpace1D.tabulate_basis`.
template <Real T>
void basis_funcs_1d(std::span<const T> knots, std::int64_t degree, bool periodic,
                    std::span<const T> points, span2d<T> out_basis,
                    std::span<std::int64_t> out_first_basis) {
    PANTR_PRECONDITION(degree >= 0, "degree must be non-negative");
    PANTR_PRECONDITION(out_basis.extent(0) == points.size() &&
                           out_basis.extent(1) == static_cast<std::size_t>(degree) + 1,
                       "out_basis must have shape (points.size(), degree+1)");
    PANTR_PRECONDITION(out_first_basis.size() == points.size(),
                       "out_first_basis must hold one entry per point");

    const auto order = static_cast<std::size_t>(degree) + 1;
    std::vector<T> left(order, T(0.0));
    std::vector<T> right(order, T(0.0));

    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto located = find_span_and_first_basis<T>(knots, degree, periodic, points[i]);
        out_first_basis[i] = located.first_basis;
        const std::span<T> row(&at(out_basis, i, 0), order);
        basis_funcs_point<T>(knots, degree, located.span, points[i], left, right, row);
    }
}

/// Tabulate the B-spline basis derivatives over a general knot vector (A2.3, batched).
///
/// Port of `_compute_basis_deriv_nurbs_book_impl` and its serial twin. Row 0 of each
/// point's block holds the plain basis values and is identical to `basis_funcs_1d`'s
/// output for the same arguments.
///
/// \tparam T Scalar type of the knots, the points and the output.
/// \param knots The knot vector, non-decreasing.
/// \param degree The polynomial degree.
/// \param periodic Whether the space is periodic.
/// \param n_deriv Highest derivative order.
/// \param points The evaluation points.
/// \param out_deriv Shape `(points.size(), n_deriv + 1, degree + 1)`, written in full.
/// \param out_first_basis One entry per point, written in full.
///
/// \note No input validation is performed. This is a Layer 3 kernel; the obligations
///       are `basis_funcs_1d`'s plus `n_deriv >= 0`. **The values are wrong in both
///       backends for `degree >= 21`** once `n_deriv` reaches the wrapping order, by
///       the factorial accumulator described in the file comment; the two agree
///       there, which is what parity claims, and neither is right.
///       For general use call
///       `pantr.bspline.BsplineSpace1D.tabulate_basis_derivatives`.
template <Real T>
void basis_derivs_1d(std::span<const T> knots, std::int64_t degree,  // NOLINT
                     bool periodic, std::int64_t n_deriv, std::span<const T> points,
                     span_nd<T, 3> out_deriv, std::span<std::int64_t> out_first_basis) {
    PANTR_PRECONDITION(degree >= 0 && n_deriv >= 0, "degree and n_deriv must be non-negative");
    PANTR_PRECONDITION(out_deriv.extent(0) == points.size() &&
                           out_deriv.extent(1) == static_cast<std::size_t>(n_deriv) + 1 &&
                           out_deriv.extent(2) == static_cast<std::size_t>(degree) + 1,
                       "out_deriv must have shape (points.size(), n_deriv+1, degree+1)");
    PANTR_PRECONDITION(out_first_basis.size() == points.size(),
                       "out_first_basis must hold one entry per point");

    const auto order = static_cast<std::size_t>(degree) + 1;
    const auto rows = static_cast<std::size_t>(n_deriv) + 1;
    std::vector<T> left(order, T(0.0));
    std::vector<T> right(order, T(0.0));
    std::vector<T> ndu_storage(order * order, T(0.0));
    std::vector<T> a_storage(2 * rows, T(0.0));
    const span2d<T> ndu(ndu_storage.data(), order, order);
    const span2d<T> a(a_storage.data(), std::size_t{2}, rows);

    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto located = find_span_and_first_basis<T>(knots, degree, periodic, points[i]);
        out_first_basis[i] = located.first_basis;
        const span2d<T> block(&at(out_deriv, i, 0, 0), rows, order);
        basis_derivs_point<T>(knots, degree, n_deriv, located.span, points[i], ndu, left, right,
                              a, block);
    }
}

/// Tabulate the basis of a space, taking its Bézier-like fast path where it has one.
///
/// The Layer 2 equivalent of `_tabulate_Bspline_basis_1D_impl`'s dispatch, and the
/// entry point for a C++ caller. A space whose knots describe a single Bézier segment
/// has a basis that *is* the Bernstein basis of the same degree on the domain, so the
/// points are mapped to `[0, 1]` and `tabulate_bernstein_1d` does the work; the first
/// basis function is then function 0 everywhere.
///
/// \tparam T Scalar type of the space, the points and the output.
/// \param space The space whose basis is tabulated.
/// \param points The evaluation points. Not checked against the domain -- a point
///        outside it is evaluated in the clamped span, which is what the oracle does
///        with `validate=False`.
/// \param out_basis Shape `(points.size(), space.degree() + 1)`, written in full.
/// \param out_first_basis One entry per point, written in full.
///
/// \note The change of variable `(x - a) / (b - a)` is committed here in `T`, exactly
///       as the oracle's numpy expression is. It is *not* a shared common-mode map:
///       the Python adapter performs its own, on the same values in the same order,
///       so the two agree bitwise.
template <Real T>
void tabulate_basis_1d(const BsplineSpace1D<T>& space, std::span<const T> points,
                       span2d<T> out_basis, std::span<std::int64_t> out_first_basis) {
    const std::int64_t degree = space.degree();

    if (!space.has_bezier_like_knots()) {
        basis_funcs_1d<T>(space.knots(), degree, space.periodic(), points, out_basis,
                          out_first_basis);
        return;
    }

    const std::array<T, 2> ends = space.domain();
    std::vector<T> reference(points.size());
    const T span = ends[1] - ends[0];
    for (std::size_t i = 0; i < points.size(); ++i) {
        reference[i] = (points[i] - ends[0]) / span;
    }

    tabulate_bernstein_1d<T>(static_cast<int>(degree), reference, out_basis);
    std::fill(out_first_basis.begin(), out_first_basis.end(), std::int64_t{0});
}

/// Tabulate the basis derivatives of a space, taking its Bézier-like fast path.
///
/// The Layer 2 equivalent of `_tabulate_Bspline_basis_deriv_1D_impl`'s dispatch. On
/// the Bézier-like path the derivatives come from `tabulate_bernstein_deriv_1d` on
/// `[0, 1]` and are then scaled by the chain rule: the k-th row carries
/// `(1 / (b - a))^k`, accumulated by repeated multiplication rather than by `pow`,
/// which is what the oracle does.
///
/// \tparam T Scalar type of the space, the points and the output.
/// \param space The space whose derivatives are tabulated.
/// \param n_deriv Highest derivative order.
/// \param points The evaluation points.
/// \param out_deriv Shape `(points.size(), n_deriv + 1, space.degree() + 1)`, written
///        in full.
/// \param out_first_basis One entry per point, written in full.
///
/// \note **The factor is accumulated wide and applied narrow**, and both halves were
///       measured rather than read. The oracle forms `1.0 / float(b - a)` as a Python
///       `float`, so the accumulation `scale *= inverse_span` is genuinely `double`;
///       but `row * scale` sends a *weak* Python scalar into a `float32` array, and
///       NEP 50 casts it to the array's dtype before multiplying. Measured over
///       120 000 values at six scales: applying `T(scale)` reproduces numpy on every
///       one, multiplying in `double` reproduces 94 298, and the two models differ on
///       25 702. Row 0 is left alone, its factor being one.
template <Real T>
void tabulate_basis_derivatives_1d(const BsplineSpace1D<T>& space, std::int64_t n_deriv,
                                   std::span<const T> points, span_nd<T, 3> out_deriv,
                                   std::span<std::int64_t> out_first_basis) {
    using pantr::value_of;
    const std::int64_t degree = space.degree();

    if (!space.has_bezier_like_knots()) {
        basis_derivs_1d<T>(space.knots(), degree, space.periodic(), n_deriv, points, out_deriv,
                           out_first_basis);
        return;
    }

    const std::array<T, 2> ends = space.domain();
    std::vector<T> reference(points.size());
    const T span = ends[1] - ends[0];
    for (std::size_t i = 0; i < points.size(); ++i) {
        reference[i] = (points[i] - ends[0]) / span;
    }

    tabulate_bernstein_deriv_1d<T>(static_cast<int>(degree), static_cast<int>(n_deriv),
                                   reference, out_deriv);

    const double inverse_span = 1.0 / static_cast<double>(value_of(span));
    double scale = inverse_span;
    const auto order = static_cast<std::size_t>(degree) + 1;
    for (std::int64_t k = 1; k <= n_deriv; ++k) {
        // Narrowed before the multiply, and by a cast rather than by
        // direct-initialisation: `const T factor(scale)` is a declaration, so it
        // trips -Wfloat-conversion, while the cast says the narrowing is meant.
        const T factor = static_cast<T>(scale);
        for (std::size_t i = 0; i < points.size(); ++i) {
            for (std::size_t j = 0; j < order; ++j) {
                at(out_deriv, i, static_cast<std::size_t>(k), j) =
                    at(out_deriv, i, static_cast<std::size_t>(k), j) * factor;
            }
        }
        scale *= inverse_span;
    }

    std::fill(out_first_basis.begin(), out_first_basis.end(), std::int64_t{0});
}

}  // namespace pantr::bspline
