#pragma once

/// \file
/// Structural operations on a B-spline field: the open conversion, splitting,
/// slicing and the boundary.
///
/// Ports `pantr.bspline.Bspline.to_open_bspline`, `.split`, `.slice` and
/// `.boundary`, whose substance is `_bspline_knot_insertion.py`'s
/// `_to_open_bspline_1d_impl` and `_to_open_bspline_impl`, `_bspline_split.py` and
/// `_bspline_slice.py`. The Python side stays as the parity oracle.
///
/// ## What this file adds, and what it only calls
///
/// The open conversion and the split are **knot insertion with a different
/// bookkeeping around it**, so neither carries any new numerics:
///
///  - `to_open` raises each end's multiplicity to `degree + 1` and drops the knots
///    that fall outside the domain,
///  - `split` raises the split value's multiplicity to `degree + 1` and cuts the
///    knot vector and the net in two.
///
/// Both reach `pantr/bspline/knot_insertion.hpp`'s `inserted_knot_vector` and
/// `oslo_bands_1d` and `pantr/bspline/refinement.hpp`'s `refine_along_axis` for the
/// whole of that. **There is no second Oslo recurrence here, no second merge and no
/// second two-scale sweep**; a reader looking for the recurrence should read the
/// first of those files, and a reader looking for what a *field* does with it the
/// second.
///
/// **The one new numerical kernel is `corner_cut_along_axis`**, the de Boor corner
/// cut the slice is built on (Piegl and Tiller, *The NURBS Book*, A5.1
/// `CurvePntByCornerCut`). It is genuinely new work rather than a rearrangement of
/// something present: **no header under `pantr/bspline/` evaluates a basis function**,
/// because basis tabulation is its own port. (`pantr/basis/cardinal_bspline.hpp` does
/// tabulate a B-spline basis, but only the *cardinal* one, over uniform integer knots
/// and not over a `BsplineSpace1D` -- so it is no help here.) The corner cut needs
/// none of it: it reads control points and knots and never touches a basis function,
/// which is why the slice is portable ahead of `Bspline::evaluate`.
///
/// The two remaining pieces of bookkeeping are `select_rows_along_axis`, which is one
/// primitive standing in for three row selections the oracle spells three ways (a
/// periodic net's modulo wrap, the trim of an open conversion, and the two halves of
/// a split), and the wrapper-level argument checks.
///
/// ## Free functions, not methods
///
/// `pantr/bspline/bspline.hpp` lists "the boundary and periodic conversions, ...
/// splitting, slicing" among the computations *over* a field rather than properties
/// *of* one, and states that each is "a separate port over free functions taking a
/// `const Bspline&`". `pantr/bspline/space_1d.hpp:6-20` and
/// `pantr/bspline/space_nd.hpp` draw the same line for a space, and
/// `pantr/bspline/refinement.hpp` already follows it. So every entry point below
/// takes a `const Bspline<T>&` and returns a new field; the type has no mutator and
/// this file adds none.
///
/// The names and the shapes are `pantr/bezier/shape.hpp`'s, deliberately: that file
/// already ported the same four operations for a Bézier, and `split` returning a
/// `std::pair`, `slice` refusing a one-dimensional field, `slice_point` serving that
/// case instead and `boundary` being defined as a `slice` are its four decisions
/// rather than four fresh ones. A caller who has learned one package's spelling has
/// learned the other's.
///
/// ## Why a one-dimensional slice is a second function
///
/// The oracle returns a `Bspline` for `dim >= 2` and a numpy array for `dim == 1`,
/// and C++ cannot return one type or the other. `slice` therefore refuses a
/// one-dimensional field and `slice_point` serves it, which is exactly what
/// `pantr/bezier/shape.hpp` does and for the same reason. **The projection of a
/// rational result is the caller's**, as it is there: `slice_point` hands back the
/// homogeneous components with the weight column still on, and the oracle's
/// `result[:-1] / result[-1]` stays in the wrapper, where it is one numpy division
/// on both backends instead of two spellings of one.
///
/// ## The floating-point discipline, quantity by quantity
///
/// `design/backend_parity.md` Rule 9 is that an oracle's arithmetic width is a
/// per-kernel fact rather than a module convention, and this file is where that has
/// the most teeth in the port so far: the oracle mixes Python scalars and numpy
/// arrays inside single expressions, and the two round differently at `float32`.
/// Each site below was settled by **executing** the oracle rather than by reading
/// it.
///
/// **The corner cut computes its weights in `double` and applies them in `T`.**
/// `_slice_bspline_1d` forms `left_knot`, `right_knot`, `denom` and `alpha` through
/// `float(...)`, so all four are Python floats and the arithmetic is `double`
/// whatever the knots are stored in. It then writes
/// `R[i] = alpha * R[i + 1] + (1.0 - alpha) * R[i]`, where `R` is a `T` array and
/// `alpha` a Python float -- and under NEP 50 a Python float is *weak*, so both
/// weights are rounded to `T` and the whole update runs in `T`. Computing the weights
/// in `T`, or the update in `double`, would each be a divergence, in opposite
/// directions, and both are visible at `float32` in the last bit while neither is at
/// `float64`.
///
/// The claim is **pinned by a test rather than by this comment**: replacing the update
/// with a `double` one fails `tests/parity/test_bspline_structural.py`'s `float32`
/// slice cases and none of its `float64` ones. No count is quoted here, because a
/// count is pinned to the fixture list it was taken over and nothing would re-take
/// it.
///
/// **The span search and the multiplicity count are `double`.** `_find_span` and
/// `_count_multiplicity` reach every knot through `float(knots[i])`, so both
/// comparisons widen first. `np.searchsorted` also compares in `double` against a
/// `double` needle -- measured, and not what NEP 50 would suggest, because
/// `searchsorted` promotes rather than weakening its needle. So the span search is
/// `std::upper_bound`/`std::lower_bound` over the knots read as `double`.
///
/// **The end multiplicities of the open conversion, and the split's multiplicity
/// count, are `T`.** These are the sites that go the other way.
/// `np.sum(np.abs(knots[: p + 1] - a) <= tol)` is a numpy array expression, so the
/// Python float `a` is weakened to `T`, the difference is formed in `T`, **and the
/// `double` `tol` is weakened to `T` before the comparison** -- measured: at
/// `float32`, `x <= tol` and `x <= float32(tol)` genuinely disagree for a `tol` that
/// is not representable, and `numpy` gives the second. `split`'s
/// `np.sum(np.abs(knots - value) <= tol)` is the same shape. So `same_knot_in_storage`
/// below rounds the tolerance rather than widening the difference.
///
/// That is the opposite convention from `inserted_knot_vector`'s domain check in
/// `pantr/bspline/knot_insertion.hpp`, which widens the difference to `double`, **and
/// both are right**, which is worth stating because the pair looks like a
/// contradiction and is not. The predicate is written the same way in both oracles;
/// what differs is that one of them is *compiled*. `_is_in_domain_impl` is
/// `@nb_jit(nopython=True)`, and numba unifies a `float32` array against a `float64`
/// scalar by widening the array -- so its comparison is in `double`. The three
/// expressions this file transcribes sit in plain Layer 2 Python, where NEP 50
/// *weakens* the Python float to the array's format instead. Measured on one
/// constructed input at `float32`: the same distance and the same tolerance give
/// `False` from the jitted oracle, `True` from the numpy spelling.
///
/// So `design/backend_parity.md` Rule 9's "per-kernel fact" is sharper than per-kernel
/// here -- the width follows the **compilation** of the expression, not the module it
/// lives in -- and a transcription that reads the predicate without checking which
/// side of that line it is on will be wrong half the time.
///
/// **The refined knot vectors and the two-scale sweep are unchanged.**
/// `inserted_knot_vector` and `refine_along_axis` carry their own derivations, and
/// nothing here re-rounds their output.
///
/// **The strided sweep replaces the oracle's transpose, and cannot change a bit.**
/// The oracle calls `_flatten_along_axis`, which is `np.moveaxis` plus
/// `np.ascontiguousarray`: a permutation of values, not an arithmetic step. Every
/// primitive below takes the axis's `outer` and `inner` extents instead, which is
/// `refine_along_axis`'s own shape. The claim that this is free is worth stating
/// precisely rather than by analogy: in `corner_cut_along_axis`, output coefficient
/// `(o, k)` is a function of `values(o, ., k)` alone, and the sequence of operations
/// on those operands is fixed by `j` and `i`; so the loops over `o` and over `k` are
/// loops over independent destinations, and permuting them changes no operation and
/// no operand. `select_rows_along_axis` performs no arithmetic at all.
///
/// Under this build -- `-ffp-contract=on`, no `-march`, so baseline x86-64 with no
/// FMA to fuse into, `cpp/README.md` -- the two backends therefore execute the same
/// IEEE-754 operations in the same order and every coefficient is bit-identical.
/// That is a property of the *host* rather than of the code, which is Rule 7: a
/// target with an FMA fuses the corner cut's `wa * R[i + 1] + wb * R[i]` and the
/// claim becomes Rule 10's budget. `pantr._pantr_cpp.__fp_contract__` is the gate
/// that tells them apart.
///
/// ## Why the parameter is a `double` and not an `accumulator_t<T>`
///
/// `pantr/bezier/shape.hpp` types the analogous parameter of its `split`, `slice` and
/// `slice_point` as `accumulator_t<T>`, so that a differentiable scalar keeps its own
/// type through the operation. Every entry point here takes a plain `double` instead,
/// and the reason is `pantr/core/scalar.hpp`'s own rule 4: **a parameter that changes
/// discrete structure is value-only.** A Bézier's parameter changes nothing discrete --
/// de Casteljau runs the same number of passes at every value. A B-spline's decides
/// which knot span it falls in, what multiplicity it has there, and how many knots an
/// insertion adds, so it selects the computation and not merely its operands. That is
/// the same standing this rule gives the degree.
///
/// It costs nothing today, since `accumulator_t<float>` and `accumulator_t<double>` are
/// both `double`, and it is stated here because the deviation from the sibling header is
/// otherwise invisible.
///
/// ## What is not ported, and it is a declared boundary
///
/// **`to_periodic` is not here, and neither is `remove_knots`.** Both decide
/// *whether an operation is admissible* by a tolerance on a **geometric** quantity
/// rather than by locating a knot: `_to_periodic_bspline_1d_impl` refuses a field
/// whose endpoints differ by more than `max(tol, 100 * eps * max(|ctrl|, 1))` and
/// again whose least-squares residual exceeds the same bound, and it reaches
/// `numpy.linalg.qr` to get that residual at all. A verdict of that kind is not the
/// same kind of work as the four operations here, whose every tolerance only ever
/// answers "is this the same knot"; and a QR whose factorization is LAPACK's on one
/// side and Eigen's on the other cannot be claimed bit-identical, so its parity
/// statement is a derivation this file does not have. It gets its own port, with
/// `remove_knots`.
///
/// Nothing here needs it: `split` converts a periodic direction to the open form and
/// returns two non-periodic halves, `slice` expands a periodic net by its own modulo
/// wrap, and `to_open` is the conversion. **So, unlike
/// `pantr/bspline/refinement.hpp`, no entry point below refuses a periodic
/// direction.**
///
/// ## Validating rather than asserting
///
/// This is the C++ counterpart of Layer 2, so it validates and throws in a release
/// build as much as in a debug one, with the oracle's messages character for
/// character. `pantr/core/error.hpp` sets the split: value and range checks here,
/// type-kind checks in the Python wrapper.
///
/// Several of the refusals below are the oracle's **Layer 1** checks -- the
/// direction and axis ranges, the strictly-inside-the-domain test, the in-domain
/// test and the side. They are near-vacuous on the Python path, where
/// `pantr.bspline.Bspline` has already made them, and load-bearing for a C++ caller
/// with no wrapper in front of it. That is the same reason
/// `pantr/bspline/refinement.hpp` restates three of `insert_knots`' Layer 1 checks
/// and `pantr/bspline/bspline.hpp` re-checks the coefficient count its own wrapper
/// checked.
///
/// **One of those texts cannot be reproduced at `float32`, and no Python caller can
/// reach it.** `Bspline.slice`'s out-of-domain message interpolates
/// `space.domain[0]`, which is a *numpy scalar* of the storage format rather than a
/// Python float, so `numpy` renders it at `float32` precision -- `0.1` where
/// `pantr::detail::format_scalar` gives `0.10000000149011612`, because that function
/// widens to `double` first and every other message in the port needs it to. The
/// wrapper refuses an out-of-domain value before dispatching, so the divergence is
/// unreachable from Python and `tests/parity/test_bspline_structural.py` pins the
/// wrapper's text on both backends instead. Changing `format_scalar` to render at
/// the storage width would fix this message and break every message
/// `pantr/bspline/knot_insertion.hpp` already matches, whose oracle *does* go
/// through `float(...)`.
///
/// ## Thread safety
///
/// Every entry point reads its argument and allocates its result, and every type
/// they touch is immutable. They are safe to call concurrently on the same field
/// with no external locking, which is the contract
/// `design/bspline_derived_caches.md` states for this package.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pantr/bezier/control_net.hpp"
#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/knot_insertion.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/bspline/refinement.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/bspline/space_nd.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::bspline {

/// Whether two knots are the same knot, in the storage format.
///
/// `|a - b| <= T(tol)`, with the difference formed in `T` and the tolerance
/// **rounded to `T`** rather than the difference widened to `double`. That is the
/// predicate `numpy` evaluates for `np.abs(knots - x) <= tol`, whose Python float
/// `tol` NEP 50 weakens to the array's format; the file comment records the
/// measurement, and the two spellings genuinely disagree at `float32` for a
/// tolerance that format cannot represent.
///
/// It is *not* the predicate `pantr/bspline/knot_insertion.hpp`'s domain check uses,
/// which transcribes a different oracle expression; see the file comment.
///
/// \param a One knot.
/// \param b The other.
/// \param tol The absolute parametric tolerance, from `BsplineSpace1D::tolerance()`.
/// \return `true` when the two are within `tol` of one another.
template <Real T>
[[nodiscard]] bool same_knot_in_storage(const T& a, const T& b, double tol) {
    using std::abs;
    return abs(a - b) <= static_cast<T>(tol);
}

/// Re-index one axis of a row-major array by an explicit list of rows.
///
/// The array is read as `(outer, num_cols, inner)` and written as
/// `(outer, rows.size(), inner)`, with
/// `out(o, i, k) = values(o, rows[i], k)`. For a control net of shape
/// `(e_0, ..., e_{d-1}, num_components)` re-indexed along direction `d`, `outer` is
/// the product of the extents before `d` and `inner` the product of those after it,
/// the component axis included -- `refine_along_axis`'s own convention.
///
/// One primitive rather than three: a periodic net's modulo wrap
/// (`rows[i] = i % num_cols`), the trim of an open conversion (a contiguous run) and
/// the two halves of a split (two contiguous runs) are the same operation with
/// different row lists, and the oracle spells them as three different numpy
/// indexings.
///
/// No arithmetic is performed, so nothing here can round.
///
/// \param values The coefficients, row-major under `(outer, num_cols, inner)`.
/// \param num_cols The axis's extent in `values`.
/// \param rows The rows to take, in output order. May repeat a row.
/// \param outer The product of the extents ahead of the axis; 1 when it is the first.
/// \param inner The product of the extents behind it, component axis included.
/// \return `outer * rows.size() * inner` coefficients, row-major.
/// \throws std::invalid_argument If any extent is negative, if `values.size()` is not
///         `outer * num_cols * inner`, or if a row is outside `[0, num_cols)`.
template <Real T>
[[nodiscard]] std::vector<T> select_rows_along_axis(std::span<const T> values,
                                                    std::int64_t num_cols,
                                                    std::span<const std::int64_t> rows,
                                                    std::int64_t outer, std::int64_t inner) {
    if (num_cols < 0 || outer < 0 || inner < 0) {
        throw std::invalid_argument("select_rows_along_axis: the extents are counts, so none "
                                    "of them can be negative.");
    }
    const std::int64_t expected = outer * num_cols * inner;
    if (static_cast<std::int64_t>(values.size()) != expected) {
        throw std::invalid_argument(
            "select_rows_along_axis: the buffer holds " + std::to_string(values.size())
            + " coefficients and the shape (" + std::to_string(outer) + ", "
            + std::to_string(num_cols) + ", " + std::to_string(inner) + ") needs "
            + std::to_string(expected) + ".");
    }
    for (const std::int64_t row : rows) {
        if (row < 0 || row >= num_cols) {
            throw std::invalid_argument("select_rows_along_axis: row " + std::to_string(row)
                                        + " is outside [0, " + std::to_string(num_cols) + ").");
        }
    }

    const auto num_rows = static_cast<std::int64_t>(rows.size());
    std::vector<T> out(static_cast<std::size_t>(outer * num_rows * inner));
    for (std::int64_t o = 0; o < outer; ++o) {
        const T* const in_block = values.data() + o * num_cols * inner;
        T* const out_block = out.data() + o * num_rows * inner;
        for (std::int64_t i = 0; i < num_rows; ++i) {
            const T* const source = in_block + rows[static_cast<std::size_t>(i)] * inner;
            std::copy_n(source, inner, out_block + i * inner);
        }
    }
    return out;
}

/// Evaluate one axis of a row-major array at a parameter, by de Boor corner cutting.
///
/// The array is read as `(outer, num_basis, inner)` with
/// `num_basis = knots.size() - degree - 1`, and written as `(outer, inner)`: the axis
/// is gone. This is `CurvePntByCornerCut` (Piegl and Tiller, *The NURBS Book*, A5.1),
/// which reaches the value from the `degree - s + 1` control points local to the span
/// in `degree - s` triangular passes, where `s` is the multiplicity of the parameter
/// as a knot. At a knot of multiplicity `degree` or more only one basis function is
/// non-zero, so the answer is one control point and no pass runs at all.
///
/// The weights are computed in `double` and applied in `T`, which is the oracle's own
/// mixture and not a choice; the file comment carries the measurement that
/// distinguishes it from the two nearby alternatives.
///
/// \param knots The axis's knot vector, non-decreasing, at least `2 * degree + 2`
///        entries.
/// \param degree The polynomial degree, non-negative.
/// \param tol The absolute parametric tolerance, from `BsplineSpace1D::tolerance()`.
/// \param values The coefficients, row-major under `(outer, num_basis, inner)`.
/// \param value The parameter, inside the axis's domain to within `tol`. A value
///        within `tol` of an end is snapped onto it, as the oracle's `_find_span`
///        snaps it.
/// \param outer The product of the extents ahead of the axis; 1 when it is the first.
/// \param inner The product of the extents behind it, component axis included.
/// \return `outer * inner` coefficients, row-major.
/// \throws std::invalid_argument If the degree is negative, the knot vector is too
///         short, any extent is negative, or `values.size()` is not
///         `outer * num_basis * inner`.
///
/// \note No check that `value` is in the domain is performed, and it does **not**
///       extrapolate: `_find_span`'s two guards snap a parameter below the domain
///       onto its start and one above it onto its end, so this returns the nearest
///       endpoint's value rather than a polynomial continued outside its span. The
///       domain is enforced by `slice`, `slice_point` and the Python wrapper, whose
///       messages a caller should see instead of a silently clamped answer.
template <Real T>
[[nodiscard]] std::vector<T> corner_cut_along_axis(std::span<const T> knots, std::int64_t degree,
                                                   double tol, std::span<const T> values,
                                                   double value, std::int64_t outer,
                                                   std::int64_t inner) {
    if (degree < 0) {
        throw std::invalid_argument("corner_cut_along_axis: degree must be non-negative; got "
                                    + std::to_string(degree) + ".");
    }
    const auto knot_count = static_cast<std::int64_t>(knots.size());
    if (knot_count < 2 * degree + 2) {
        throw std::invalid_argument(
            "corner_cut_along_axis: a knot vector of degree " + std::to_string(degree)
            + " needs at least " + std::to_string(2 * degree + 2) + " entries; got "
            + std::to_string(knot_count) + ".");
    }
    if (outer < 0 || inner < 0) {
        throw std::invalid_argument("corner_cut_along_axis: the extents are counts, so neither "
                                    "of them can be negative.");
    }
    const std::int64_t num_basis = knot_count - degree - 1;
    const std::int64_t expected = outer * num_basis * inner;
    if (static_cast<std::int64_t>(values.size()) != expected) {
        throw std::invalid_argument(
            "corner_cut_along_axis: the buffer holds " + std::to_string(values.size())
            + " coefficients and the shape (" + std::to_string(outer) + ", "
            + std::to_string(num_basis) + ", " + std::to_string(inner) + ") needs "
            + std::to_string(expected) + ".");
    }

    using std::abs;
    const std::int64_t p = degree;
    // `_find_span`, in `double` throughout: the oracle reaches every knot through
    // `float(...)`, and `np.searchsorted` promotes its needle rather than weakening
    // it. See the file comment.
    double u = value;
    const double u_left = detail::as_double(knots[static_cast<std::size_t>(p)]);
    const double u_right = detail::as_double(knots[static_cast<std::size_t>(num_basis)]);
    std::int64_t k = 0;
    if (u <= u_left + tol) {
        k = p;
        u = u_left;
    } else if (u >= u_right - tol) {
        k = num_basis - 1;
        u = u_right;
    } else {
        // `np.searchsorted(knots, u, side="right") - 1`. The predicate is on the
        // widened knot, so the comparator widens rather than narrowing `u`.
        const auto upper = std::upper_bound(
            knots.begin(), knots.end(), u,
            [](double needle, const T& knot) { return needle < detail::as_double(knot); });
        k = static_cast<std::int64_t>(upper - knots.begin()) - 1;
    }

    // `_count_multiplicity`: knots `k`, `k - 1`, ... while they match `u`, stopping
    // after at most `degree + 1` of them. The oracle's `range(k, max(k - p - 1, -1),
    // -1)` has inclusive lower bound `max(k - p, 0)`.
    std::int64_t s = 0;
    for (std::int64_t i = k; i >= std::max(k - p, std::int64_t{0}); --i) {
        if (abs(detail::as_double(knots[static_cast<std::size_t>(i)]) - u) <= tol) {
            ++s;
        } else {
            break;
        }
    }

    // At a knot of multiplicity `degree` or more the answer is one control point, so
    // the whole cut is a row selection. Sharing `select_rows_along_axis` for it is
    // what keeps this branch from being a second copy of the strided copy loop.
    if (s >= p) {
        const std::array<std::int64_t, 1> row{k - p};
        return select_rows_along_axis<T>(values, num_basis, std::span<const std::int64_t>(row),
                                         outer, inner);
    }

    const std::int64_t r = p - s;
    // The oracle's `_zero_denom_tol`, transcribed. It stands in for an exact-zero test
    // on a knot-span width rather than deriving a magnitude from anything, so it is
    // reproduced rather than re-derived, and it is deliberately not one of this
    // project's tolerances: it denotes no bound that `BsplineSpace1D::tolerance()`
    // denotes.
    constexpr double kZeroDenominator = 1e-300;

    // One working triangle of `r + 1` rows, refilled per outer block. The oracle
    // allocates `R` once because its `outer` is always 1 -- it moves the axis to the
    // front -- and reusing one buffer here is the same arithmetic on each block.
    std::vector<T> work(static_cast<std::size_t>((r + 1) * inner));
    std::vector<T> out(static_cast<std::size_t>(outer * inner));

    for (std::int64_t o = 0; o < outer; ++o) {
        const T* const in_block = values.data() + o * num_basis * inner;
        std::copy_n(in_block + (k - p) * inner, (r + 1) * inner, work.data());

        for (std::int64_t j = 1; j <= r; ++j) {
            // Ascending `i`, in place: at `i` the row `i + 1` still holds the previous
            // pass's value, because this pass writes rows `0 .. r - j` in order. That
            // is the oracle's loop and the standard corner cut; descending would read
            // a row this pass had already overwritten.
            for (std::int64_t i = 0; i <= r - j; ++i) {
                const double left =
                    detail::as_double(knots[static_cast<std::size_t>(k - p + j + i)]);
                const double right =
                    detail::as_double(knots[static_cast<std::size_t>(i + k + 1)]);
                const double denom = right - left;
                const double alpha =
                    abs(denom) < kZeroDenominator ? 0.0 : (u - left) / denom;
                // Rounded to `T` here and only here: the weights are `double`
                // expressions and the update is a `T` one. See the file comment.
                const T wa = static_cast<T>(alpha);
                const T wb = static_cast<T>(1.0 - alpha);
                T* const row = work.data() + i * inner;
                const T* const next = work.data() + (i + 1) * inner;
                for (std::int64_t c = 0; c < inner; ++c) {
                    // `alpha * R[i + 1] + (1.0 - alpha) * R[i]`, operand for operand
                    // and in that order: the two products and the sum are three
                    // roundings, and swapping the addends is a different number.
                    row[c] = wa * next[c] + wb * row[c];
                }
            }
        }
        std::copy_n(work.data(), inner, out.data() + o * inner);
    }
    return out;
}

/// One direction's open (clamped) form: its knot vector and the re-expressed net.
///
/// \tparam T The scalar type the knots and the coefficients share.
template <Real T>
struct OpenAlongAxis {
    /// The clamped knot vector, of multiplicity `degree + 1` at both ends.
    std::vector<T> knots;

    /// The coefficients, row-major under `(outer, knots.size() - degree - 1, inner)`.
    std::vector<T> values;
};

/// Re-express one axis of a row-major array over a clamped knot vector.
///
/// Raises each end of the knot vector to multiplicity `degree + 1` by inserting the
/// boundary knot as many times as it is short, then drops the knots that fall outside
/// the domain and the coefficients they carried. For a periodic axis the
/// `knots.size() - degree - 1` coefficients of the full non-periodic net are
/// reconstructed first by wrapping the stored ones, `full[i] = stored[i % n_stored]`,
/// which is the representation the ghost knots describe.
///
/// The geometry does not move: the insertion goes through `inserted_knot_vector` and
/// `refine_along_axis`, so the coefficients are the two-scale image of the originals.
///
/// \param knots The axis's knot vector, ghost knots included when `periodic`.
/// \param degree The polynomial degree, non-negative.
/// \param periodic Whether the axis is periodic, which is what makes the stored net
///        shorter than the knot vector implies.
/// \param tol The absolute parametric tolerance, from `BsplineSpace1D::tolerance()`.
/// \param values The coefficients, row-major under `(outer, n_stored, inner)` where
///        `n_stored` is the axis's basis count.
/// \param n_stored The axis's extent in `values`.
/// \param outer The product of the extents ahead of the axis; 1 when it is the first.
/// \param inner The product of the extents behind it, component axis included.
/// \return The clamped knot vector and the coefficients over it.
/// \throws std::invalid_argument If the axis is already clamped at both ends and not
///         periodic -- with the oracle's message, since there is nothing to do and
///         the oracle refuses rather than returning a copy -- or if any argument is
///         one the primitives above refuse.
template <Real T>
[[nodiscard]] OpenAlongAxis<T> open_along_axis(std::span<const T> knots, std::int64_t degree,
                                               bool periodic, double tol,
                                               std::span<const T> values, std::int64_t n_stored,
                                               std::int64_t outer, std::int64_t inner) {
    if (degree < 0) {
        throw std::invalid_argument("open_along_axis: degree must be non-negative; got "
                                    + std::to_string(degree) + ".");
    }
    const auto knot_count = static_cast<std::int64_t>(knots.size());
    if (knot_count < 2 * degree + 2) {
        throw std::invalid_argument(
            "open_along_axis: a knot vector of degree " + std::to_string(degree)
            + " needs at least " + std::to_string(2 * degree + 2) + " entries; got "
            + std::to_string(knot_count) + ".");
    }

    const std::int64_t p = degree;
    const T a = knots[static_cast<std::size_t>(p)];
    const T b = knots[static_cast<std::size_t>(knot_count - p - 1)];

    // The end multiplicities, in `T` and against a `T`-rounded tolerance: these two
    // are numpy array expressions in the oracle. See `same_knot_in_storage`.
    std::int64_t m_left = 0;
    for (std::int64_t i = 0; i <= p; ++i) {
        if (same_knot_in_storage<T>(knots[static_cast<std::size_t>(i)], a, tol)) {
            ++m_left;
        }
    }
    std::int64_t m_right = 0;
    for (std::int64_t i = knot_count - p - 1; i < knot_count; ++i) {
        if (same_knot_in_storage<T>(knots[static_cast<std::size_t>(i)], b, tol)) {
            ++m_right;
        }
    }

    if (m_left == p + 1 && m_right == p + 1 && !periodic) {
        throw std::invalid_argument(
            "B-spline is already open (both boundary knots have multiplicity degree + 1).");
    }

    // The full non-periodic net the knot vector describes. A non-periodic axis already
    // stores it; a periodic one stores one period of it.
    const std::int64_t n_full = knot_count - p - 1;
    std::vector<T> full;
    if (periodic) {
        std::vector<std::int64_t> wrapped(static_cast<std::size_t>(n_full));
        for (std::int64_t i = 0; i < n_full; ++i) {
            wrapped[static_cast<std::size_t>(i)] = i % n_stored;
        }
        full = select_rows_along_axis<T>(values, n_stored,
                                         std::span<const std::int64_t>(wrapped), outer, inner);
    } else {
        if (n_stored != n_full) {
            throw std::invalid_argument(
                "open_along_axis: a non-periodic axis stores one coefficient per basis "
                "function, so the extent must be "
                + std::to_string(n_full) + "; got " + std::to_string(n_stored) + ".");
        }
        full.assign(values.begin(), values.end());
    }

    const std::int64_t n_trim_left = p + 1 - m_left;
    const std::int64_t n_trim_right = p + 1 - m_right;

    std::vector<T> to_insert;
    to_insert.reserve(static_cast<std::size_t>(n_trim_left + n_trim_right));
    to_insert.insert(to_insert.end(), static_cast<std::size_t>(n_trim_left), a);
    to_insert.insert(to_insert.end(), static_cast<std::size_t>(n_trim_right), b);

    std::vector<T> merged;
    std::int64_t num_rows = 0;
    if (to_insert.empty()) {
        merged.assign(knots.begin(), knots.end());
        num_rows = n_full;
    } else {
        merged = inserted_knot_vector<T>(knots, p, std::span<const T>(to_insert), tol);
        const OsloBands<T> bands = oslo_bands_1d<T>(p, knots, std::span<const T>(merged));
        full = refine_along_axis<T>(bands, n_full, std::span<const T>(full), outer, inner);
        num_rows = bands.num_rows;
    }

    // The trim. Each side drops `p + 1 - m` entries, counted with *its own* boundary
    // multiplicity: the two may differ, and an earlier reading of the oracle that used
    // one count for both would be wrong on a vector clamped at one end only.
    OpenAlongAxis<T> out;
    out.knots.assign(merged.begin() + n_trim_left, merged.end() - n_trim_right);
    const auto n_open = static_cast<std::int64_t>(out.knots.size()) - p - 1;
    std::vector<std::int64_t> kept(static_cast<std::size_t>(n_open));
    for (std::int64_t i = 0; i < n_open; ++i) {
        kept[static_cast<std::size_t>(i)] = n_trim_left + i;
    }
    out.values = select_rows_along_axis<T>(std::span<const T>(full), num_rows,
                                           std::span<const std::int64_t>(kept), outer, inner);
    return out;
}

namespace detail {

/// The product of a net's extents ahead of one axis.
///
/// \param shape The net's shape, component axis included.
/// \param axis The axis.
/// \return 1 when `axis` is the first, the product of the earlier extents otherwise.
[[nodiscard]] inline std::int64_t extents_before(std::span<const std::size_t> shape,
                                                 std::int64_t axis) {
    std::int64_t product = 1;
    for (std::int64_t d = 0; d < axis; ++d) {
        product *= static_cast<std::int64_t>(shape[static_cast<std::size_t>(d)]);
    }
    return product;
}

/// The product of a net's extents behind one axis, component axis included.
///
/// \param shape The net's shape, component axis included.
/// \param axis The axis.
/// \return The product of the later extents, which is at least the component count.
[[nodiscard]] inline std::int64_t extents_after(std::span<const std::size_t> shape,
                                                std::int64_t axis) {
    std::int64_t product = 1;
    for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < shape.size(); ++d) {
        product *= static_cast<std::int64_t>(shape[d]);
    }
    return product;
}

/// Refuse an axis index the field does not have, with the oracle's message.
///
/// \param name The oracle's own name for the parameter, `"axis"` or `"direction"`.
/// \param axis The index to check.
/// \param dim The field's number of parametric directions.
/// \throws std::invalid_argument If `axis` is outside `[0, dim)`.
inline void require_axis(const char* name, std::int64_t axis, std::int64_t dim) {
    if (axis < 0 || axis >= dim) {
        throw std::invalid_argument(std::string(name) + " must be in [0, " + std::to_string(dim)
                                    + "), got " + std::to_string(axis) + ".");
    }
}

/// Assemble a field from per-direction spaces, a net shape and its coefficients.
///
/// \tparam T The scalar type the field stores.
/// \param directions One space handle per direction, in axis order.
/// \param values The coefficients, row-major under `shape`.
/// \param shape The net's shape, component axis last.
/// \param is_rational Whether the last component of each coefficient is a weight.
/// \return The field.
/// \throws std::invalid_argument If `Bspline`'s or `BsplineSpace`'s constructor
///         refuses the pieces.
template <Real T>
[[nodiscard]] Bspline<T>
assemble(std::vector<std::shared_ptr<const BsplineSpace1D<T>>> directions,
         const std::vector<T>& values, const std::vector<std::size_t>& shape, bool is_rational) {
    return Bspline<T>(std::make_shared<const BsplineSpace<T>>(std::move(directions)),
                      typename Bspline<T>::net_type(std::span<const T>(values),
                                                    std::span<const std::size_t>(shape)),
                      is_rational);
}

}  // namespace detail

/// The field re-expressed over clamped, non-periodic knot vectors in every direction.
///
/// Each direction that is periodic or unclamped is converted by `open_along_axis`;
/// one that is already open is left exactly alone, its space handle carried into the
/// result rather than rebuilt, so `opened.space.spaces[d] is field.space.spaces[d]`
/// holds for it on the Python side. The geometry is unchanged up to the rounding the
/// file comment bounds.
///
/// \param field The field to convert.
/// \return An open, non-periodic field over the same geometry.
/// \throws std::invalid_argument If every direction is already open and non-periodic,
///         with the oracle's message, since there would be nothing to do.
template <Real T>
[[nodiscard]] Bspline<T> to_open(const Bspline<T>& field) {
    const BsplineSpace<T>& space = field.space_ref();
    const std::int64_t dim = space.dim();

    // Checked before any conversion runs, which is the oracle's order.
    bool anything_to_do = false;
    for (std::int64_t d = 0; d < dim; ++d) {
        const BsplineSpace1D<T>& direction = space.space_ref(d);
        anything_to_do =
            anything_to_do || direction.periodic() || !direction.has_open_knots();
    }
    if (!anything_to_do) {
        throw std::invalid_argument("B-spline is already open in every direction.");
    }

    std::vector<std::shared_ptr<const BsplineSpace1D<T>>> directions(space.spaces().begin(),
                                                                     space.spaces().end());
    std::vector<std::size_t> shape(field.net().shape().begin(), field.net().shape().end());
    std::vector<T> values(field.net().values().begin(), field.net().values().end());

    for (std::int64_t d = 0; d < dim; ++d) {
        const BsplineSpace1D<T>& direction = *directions[static_cast<std::size_t>(d)];
        if (direction.has_open_knots() && !direction.periodic()) {
            continue;
        }
        const std::int64_t degree = direction.degree();
        const OpenAlongAxis<T> opened = open_along_axis<T>(
            direction.knots(), degree, direction.periodic(), direction.tolerance(),
            std::span<const T>(values),
            static_cast<std::int64_t>(shape[static_cast<std::size_t>(d)]),
            detail::extents_before(std::span<const std::size_t>(shape), d),
            detail::extents_after(std::span<const std::size_t>(shape), d));

        values = opened.values;
        shape[static_cast<std::size_t>(d)] =
            static_cast<std::size_t>(opened.knots.size()) - static_cast<std::size_t>(degree) - 1;
        // Snapping on, which is the oracle's `BsplineSpace1D(open_knots, degree,
        // periodic=False)` taking its default -- and *not* the split's, which passes
        // `snap_knots=False`. The two are different calls in the oracle and stay
        // different here.
        directions[static_cast<std::size_t>(d)] = std::make_shared<const BsplineSpace1D<T>>(
            std::span<const T>(opened.knots), degree, false,
            KnotSnapping::merge_near_duplicates);
    }

    return detail::assemble<T>(std::move(directions), values, shape, field.is_rational());
}

/// The field cut in two at a parameter of one direction.
///
/// The split value's multiplicity is raised to `degree + 1`, which makes the knot
/// vector separable there, and the vector and the net are then partitioned. The left
/// half spans `[domain_start, value]` and the right `[value, domain_end]`; every
/// other direction is untouched and its space handle is shared by both halves and by
/// `field`. A periodic split direction is converted to its open form first, so both
/// halves are non-periodic in it whatever `field` was.
///
/// \param field The field to split.
/// \param direction The direction to split, in `[0, field.dim())`.
/// \param value The parameter to split at, strictly inside that direction's domain by
///        more than its tolerance.
/// \return The left and right halves.
/// \throws std::invalid_argument If `direction` is out of range or `value` is not
///         strictly inside the domain, both with the oracle's Layer 1 messages.
template <Real T>
[[nodiscard]] std::pair<Bspline<T>, Bspline<T>> split(const Bspline<T>& field,
                                                      std::int64_t direction, double value) {
    const BsplineSpace<T>& space = field.space_ref();
    const std::int64_t dim = space.dim();
    detail::require_axis("direction", direction, dim);

    const BsplineSpace1D<T>& along = space.space_ref(direction);
    const std::array<T, 2> domain = along.domain();
    const double lo = pantr::bspline::detail::as_double(domain[0]);
    const double hi = pantr::bspline::detail::as_double(domain[1]);
    const double tol = along.tolerance();
    if (value <= lo + tol || value >= hi - tol) {
        throw std::invalid_argument("value must be strictly inside the domain ("
                                    + pantr::detail::format_repr(lo) + ", "
                                    + pantr::detail::format_repr(hi) + "), got "
                                    + pantr::detail::format_repr(value) + ".");
    }

    const std::int64_t p = along.degree();
    std::vector<std::size_t> shape(field.net().shape().begin(), field.net().shape().end());
    const std::int64_t outer = detail::extents_before(std::span<const std::size_t>(shape),
                                                      direction);
    const std::int64_t inner = detail::extents_after(std::span<const std::size_t>(shape),
                                                     direction);

    std::vector<T> knots;
    std::vector<T> values;
    if (along.periodic()) {
        OpenAlongAxis<T> opened = open_along_axis<T>(
            along.knots(), p, true, tol, field.net().values(),
            static_cast<std::int64_t>(shape[static_cast<std::size_t>(direction)]), outer, inner);
        knots = std::move(opened.knots);
        values = std::move(opened.values);
    } else {
        knots.assign(along.knots().begin(), along.knots().end());
        values.assign(field.net().values().begin(), field.net().values().end());
    }

    // The split value's current multiplicity, in `T` against a `T`-rounded tolerance:
    // the oracle's `np.sum(np.abs(knots - value) <= tol)` is a numpy array expression.
    const T value_in_T = static_cast<T>(value);
    std::int64_t multiplicity = 0;
    for (const T& knot : knots) {
        if (same_knot_in_storage<T>(knot, value_in_T, tol)) {
            ++multiplicity;
        }
    }

    const std::int64_t deficit = p + 1 - multiplicity;
    if (deficit > 0) {
        const std::int64_t num_cols = static_cast<std::int64_t>(knots.size()) - p - 1;
        const std::vector<T> to_insert(static_cast<std::size_t>(deficit), value_in_T);
        std::vector<T> merged = inserted_knot_vector<T>(
            std::span<const T>(knots), p, std::span<const T>(to_insert), tol);
        const OsloBands<T> bands =
            oslo_bands_1d<T>(p, std::span<const T>(knots), std::span<const T>(merged));
        values = refine_along_axis<T>(bands, num_cols, std::span<const T>(values), outer, inner);
        knots = std::move(merged);
    }

    // `np.searchsorted(knots, value - tol)`, whose default side is `"left"` and whose
    // comparison is in `double`; the needle is a `double` expression on both sides.
    const double needle = value - tol;
    const auto lower = std::lower_bound(
        knots.begin(), knots.end(), needle,
        [](const T& knot, double target) {
            return pantr::bspline::detail::as_double(knot) < target;
        });
    const auto i_split = static_cast<std::int64_t>(lower - knots.begin());

    const auto knot_count = static_cast<std::int64_t>(knots.size());
    const std::int64_t num_cols = knot_count - p - 1;

    std::vector<std::int64_t> left_rows(static_cast<std::size_t>(i_split));
    for (std::int64_t i = 0; i < i_split; ++i) {
        left_rows[static_cast<std::size_t>(i)] = i;
    }
    std::vector<std::int64_t> right_rows(static_cast<std::size_t>(num_cols - i_split));
    for (std::int64_t i = i_split; i < num_cols; ++i) {
        right_rows[static_cast<std::size_t>(i - i_split)] = i;
    }

    const std::vector<T> left_values = select_rows_along_axis<T>(
        std::span<const T>(values), num_cols, std::span<const std::int64_t>(left_rows), outer,
        inner);
    const std::vector<T> right_values = select_rows_along_axis<T>(
        std::span<const T>(values), num_cols, std::span<const std::int64_t>(right_rows), outer,
        inner);

    const std::vector<T> left_knots(knots.begin(), knots.begin() + i_split + p + 1);
    const std::vector<T> right_knots(knots.begin() + i_split, knots.end());

    // Snapping **off**, which is the oracle's `snap_knots=False` at both halves and
    // not the default `to_open` takes: a half's knot vector is a subrange of one that
    // was already snapped, so snapping it again could only merge a pair the parent
    // kept apart.
    auto direction_space = [p](const std::vector<T>& vector) {
        return std::make_shared<const BsplineSpace1D<T>>(std::span<const T>(vector), p, false,
                                                         KnotSnapping::as_given);
    };

    std::vector<std::shared_ptr<const BsplineSpace1D<T>>> left_directions(
        space.spaces().begin(), space.spaces().end());
    std::vector<std::shared_ptr<const BsplineSpace1D<T>>> right_directions(
        space.spaces().begin(), space.spaces().end());
    left_directions[static_cast<std::size_t>(direction)] = direction_space(left_knots);
    right_directions[static_cast<std::size_t>(direction)] = direction_space(right_knots);

    std::vector<std::size_t> left_shape = shape;
    std::vector<std::size_t> right_shape = shape;
    left_shape[static_cast<std::size_t>(direction)] = static_cast<std::size_t>(i_split);
    right_shape[static_cast<std::size_t>(direction)] =
        static_cast<std::size_t>(num_cols - i_split);

    return {detail::assemble<T>(std::move(left_directions), left_values, left_shape,
                                field.is_rational()),
            detail::assemble<T>(std::move(right_directions), right_values, right_shape,
                                field.is_rational())};
}

namespace detail {

/// Refuse a parameter outside one direction's domain, with the oracle's message.
///
/// \tparam T The scalar type the space stores.
/// \param space The direction whose domain is the bound.
/// \param axis The direction's index, which the message names.
/// \param value The parameter.
/// \throws std::invalid_argument If `value` leaves the domain by more than the
///         direction's tolerance.
///
/// \note The two rendered bounds are `format_scalar` of the knots, which is Python's
///       `repr` of the *widened* value. The oracle interpolates a numpy scalar of the
///       storage format instead, so at `float32` the two texts can differ; the file
///       comment records why that is unreachable from Python and why the fix belongs
///       elsewhere.
template <Real T>
void require_in_domain(const BsplineSpace1D<T>& space, std::int64_t axis, double value) {
    const std::array<T, 2> domain = space.domain();
    const double tol = space.tolerance();
    if (value < as_double(domain[0]) - tol || value > as_double(domain[1]) + tol) {
        throw std::invalid_argument("value " + pantr::detail::format_repr(value)
                                    + " is outside the domain ["
                                    + pantr::detail::format_scalar(domain[0]) + ", "
                                    + pantr::detail::format_scalar(domain[1])
                                    + "] of direction " + std::to_string(axis) + ".");
    }
}

/// One direction's net, expanded to the full non-periodic form when it is periodic.
///
/// The oracle's `np.take(ctrl, np.arange(n_full) % n_periodic, axis=axis)`, which is
/// what makes the corner cut's span indices address a coefficient: a periodic axis
/// stores one period and the knot vector describes the whole ghost-extended net.
///
/// \tparam T The scalar type the field stores.
/// \param space The direction.
/// \param values The coefficients, row-major under `(outer, n_stored, inner)`.
/// \param n_stored The direction's extent in `values`.
/// \param outer The product of the extents ahead of the axis.
/// \param inner The product of the extents behind it, component axis included.
/// \return The coefficients over `knots.size() - degree - 1` rows; a copy of `values`
///         when the direction is not periodic.
template <Real T>
[[nodiscard]] std::vector<T> unwrapped_net(const BsplineSpace1D<T>& space,
                                           std::span<const T> values, std::int64_t n_stored,
                                           std::int64_t outer, std::int64_t inner) {
    if (!space.periodic()) {
        return std::vector<T>(values.begin(), values.end());
    }
    const std::int64_t n_full =
        static_cast<std::int64_t>(space.knots().size()) - space.degree() - 1;
    std::vector<std::int64_t> wrapped(static_cast<std::size_t>(n_full));
    for (std::int64_t i = 0; i < n_full; ++i) {
        wrapped[static_cast<std::size_t>(i)] = i % n_stored;
    }
    return select_rows_along_axis<T>(values, n_stored, std::span<const std::int64_t>(wrapped),
                                     outer, inner);
}

}  // namespace detail

/// The field with one parametric direction fixed at a value, so one dimension fewer.
///
/// A volume becomes a surface and a surface a curve. The surviving directions are
/// untouched and their space handles are shared with `field`, so
/// `sliced.space.spaces[j]` is the wrapper `field.space.spaces[i]` already had for
/// the direction `i` that `j` came from. A periodic sliced direction is expanded to
/// its full non-periodic net first; the *surviving* directions keep whatever they
/// were, periodic included.
///
/// \param field The field to slice, of dimension at least two.
/// \param axis The direction to fix, in `[0, field.dim())`.
/// \param value The parameter, inside that direction's domain to within its
///        tolerance.
/// \return The sliced field, of dimension `field.dim() - 1`.
/// \throws std::invalid_argument If the field is one-dimensional -- use `slice_point`
///         -- if `axis` is out of range, or if `value` leaves the domain. The last two
///         carry the oracle's Layer 1 messages.
template <Real T>
[[nodiscard]] Bspline<T> slice(const Bspline<T>& field, std::int64_t axis, double value) {
    const BsplineSpace<T>& space = field.space_ref();
    const std::int64_t dim = space.dim();
    if (dim < 2) {
        throw std::invalid_argument(
            "slice needs a B-spline of dimension at least two; a one-dimensional one slices "
            "to a point, which slice_point returns.");
    }
    detail::require_axis("axis", axis, dim);
    const BsplineSpace1D<T>& along = space.space_ref(axis);
    detail::require_in_domain<T>(along, axis, value);

    const std::span<const std::size_t> shape = field.net().shape();
    const std::int64_t outer = detail::extents_before(shape, axis);
    const std::int64_t inner = detail::extents_after(shape, axis);
    const auto n_stored = static_cast<std::int64_t>(shape[static_cast<std::size_t>(axis)]);

    const std::vector<T> full =
        detail::unwrapped_net<T>(along, field.net().values(), n_stored, outer, inner);
    const std::vector<T> values =
        corner_cut_along_axis<T>(along.knots(), along.degree(), along.tolerance(),
                                 std::span<const T>(full), value, outer, inner);

    // The axis is dropped and the others keep their order, which is what the oracle's
    // `moveaxis`-plus-reshape produces and what the strided sweep already wrote.
    std::vector<std::shared_ptr<const BsplineSpace1D<T>>> directions;
    directions.reserve(static_cast<std::size_t>(dim) - 1);
    std::vector<std::size_t> reduced;
    reduced.reserve(shape.size() - 1);
    for (std::int64_t d = 0; d < dim; ++d) {
        if (d != axis) {
            directions.push_back(space.space(d));
            reduced.push_back(shape[static_cast<std::size_t>(d)]);
        }
    }
    reduced.push_back(shape.back());

    return detail::assemble<T>(std::move(directions), values, reduced, field.is_rational());
}

/// The point a one-dimensional field takes at a parameter, in homogeneous components.
///
/// The `dim == 1` case of the oracle's `slice`, a separate function because its result
/// is a point rather than a field and C++ cannot return one type or the other. It is
/// the de Boor corner cut and nothing else, so it is also the single-point evaluation
/// of a B-spline curve -- reachable ahead of the basis-tabulation port because the
/// corner cut needs no basis function.
///
/// **The projection of a rational result is the caller's**, exactly as
/// `pantr/bezier/shape.hpp`'s is: the weight column is still on, and the oracle's
/// `point[:-1] / point[-1]` is one numpy division in the wrapper rather than a second
/// spelling here.
///
/// \param field A one-dimensional field.
/// \param value The parameter, inside the domain to within the space's tolerance.
/// \return The `field.net().num_components()` components at that parameter, weight
///         column included.
/// \throws std::invalid_argument If the field is not one-dimensional, or if `value`
///         leaves the domain -- the latter with the oracle's Layer 1 message.
template <Real T>
[[nodiscard]] std::vector<T> slice_point(const Bspline<T>& field, double value) {
    const BsplineSpace<T>& space = field.space_ref();
    if (space.dim() != 1) {
        throw std::invalid_argument("slice_point needs a one-dimensional B-spline, got "
                                    "dimension "
                                    + std::to_string(space.dim()) + ".");
    }
    const BsplineSpace1D<T>& along = space.space_ref(0);
    detail::require_in_domain<T>(along, 0, value);

    const std::span<const std::size_t> shape = field.net().shape();
    const auto components = static_cast<std::int64_t>(field.net().num_components());
    const auto n_stored = static_cast<std::int64_t>(shape[0]);

    const std::vector<T> full =
        detail::unwrapped_net<T>(along, field.net().values(), n_stored, 1, components);
    return corner_cut_along_axis<T>(along.knots(), along.degree(), along.tolerance(),
                                    std::span<const T>(full), value, 1, components);
}

/// One face of the parametric domain.
///
/// `boundary(f, axis, side)` is `slice(f, axis, domain[side])`, which is the oracle's
/// own definition rather than a reimplementation of it: `Bspline.boundary` computes
/// the endpoint and calls `Bspline.slice`.
///
/// **It is that composition over a narrower domain than the oracle's, and this is the
/// one place the two differ.** `Bspline.boundary` works at `dim == 1` too, where its
/// `slice` returns a point; this forwards to the `slice` below, which refuses a
/// one-dimensional field, and there is no `boundary_point` beside it. That is
/// `pantr/bezier/shape.hpp`'s `boundary` unchanged, and it is reachable by no caller:
/// this function is not bound, so the Python `boundary` at `dim == 1` composes its own
/// `slice` into `slice_point` and never arrives here. A C++ caller wanting the point
/// spells it `slice_point(f, f.space_ref().space_ref(0).domain()[side])`.
///
/// It is deliberately **not bound**. The Python `Bspline.boundary` reaches C++ through
/// its own `slice`, so a binding would be public surface with no caller --
/// `pantr._pantr_cpp.bezier_boundary` is exactly that today, bound and stubbed and
/// used by nothing, and one such precedent is enough. This exists for the consumer
/// with no interpreter, which is what the port is for.
///
/// \param field The field, of dimension at least two.
/// \param axis The direction whose face is wanted, in `[0, field.dim())`.
/// \param side 0 for the face at the domain's start, 1 for the one at its end.
/// \return The face, of dimension `field.dim() - 1`.
/// \throws std::invalid_argument If `side` is neither 0 nor 1 -- with the oracle's
///         message -- or as `slice` throws.
template <Real T>
[[nodiscard]] Bspline<T> boundary(const Bspline<T>& field, std::int64_t axis, int side) {
    // The oracle checks the side before the axis, so a call that is wrong in both ways
    // reports the side. Only the order decides which message a caller reads.
    if (side != 0 && side != 1) {
        throw std::invalid_argument("side must be 0 or 1, got " + std::to_string(side) + ".");
    }
    const BsplineSpace<T>& space = field.space_ref();
    detail::require_axis("axis", axis, space.dim());
    const std::array<T, 2> domain = space.space_ref(axis).domain();
    const double value = pantr::bspline::detail::as_double(domain[static_cast<std::size_t>(side)]);
    return slice<T>(field, axis, value);
}

}  // namespace pantr::bspline
