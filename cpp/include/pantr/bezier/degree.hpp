#pragma once

/// \file
/// Degree elevation and reduction of an n-dimensional Bézier, and the `L2` error a
/// reduction would introduce.
///
/// Free functions over `Bezier<T>`, as `bezier.hpp` says: the type owns the value and
/// no operations, and this is one of the four operation ports over it. It composes
/// over `degree_elevate_bezier_1d` and `core::apply_reduction_operator` rather than
/// absorbing them.
///
/// ## What crosses as data, and why that is not a shortcut
///
/// `reduce_degree` and `degree_reduction_error` take their **reduction operators and
/// Bernstein Gram matrices as arguments** rather than assembling them. That is the
/// ruling `core/reduction_operator.hpp` already records for the operator, applied
/// unchanged to the Gram matrix because the two are the same kind of object:
/// assembled once per degree from exact arithmetic, rounded to `double` once, not a
/// kernel, and an array by the time anything computes with it.
///
/// It is worth stating what porting the assembly would cost, because "not ported
/// yet" and "deliberately not ported" read the same in a header. The reduction
/// operator is the solution of an exact rational normal-equation system; measured on
/// this tree, the solution's numerators and denominators reach **156 bits** at the
/// module's own maximum degree of 61, before the elimination intermediates that
/// produce them, so `__int128` is not enough and a faithful port needs arbitrary
/// precision. `core/reduction_operator.hpp` also records that solving the same system
/// in `double` loses eleven digits, so computing it natively is not an alternative
/// either. The Gram matrix is milder but has the same shape of problem: its entries
/// are `C(n,i) C(n,j) / ((2n+1) C(2n,i+j))`, and `C(2n, i+j)` leaves
/// `core::kBincoeffMaxN` at degree 31, less than half the range its callers accept.
///
/// **The consequence, stated plainly:** a C++ caller with no Python can elevate a
/// degree unaided, and cannot reduce one without supplying an operator it obtained
/// elsewhere. That is a real gap in the self-sufficiency
/// `design/cross_backend_types.md` asks for, it is recorded here rather than in a
/// commit message, and closing it is a decision about arbitrary-precision arithmetic
/// rather than an omission to tidy up.
///
/// ### `minimize_degree` is absent, and this is why
///
/// `pantr.bezier.Bezier.minimize_degree` has no counterpart here, which is worth
/// naming rather than leaving a reader to infer from the paragraph above. Its greedy
/// search lowers a degree by one at a time and grades each trial, so it needs a fresh
/// reduction operator and Gram matrix **per degree it visits**, discovered as it goes.
/// Supplying those from outside is possible -- the set is enumerable in advance, since
/// the degrees visited run from each direction's own degree down to one -- so this is
/// a decision rather than an impossibility. What it is not is a decision this port
/// made unilaterally: it would fix the C++ signature of a search around a table its
/// caller must precompute, which is a worse shape than the two functions above and is
/// worth settling deliberately.
///
/// Its rational branch is a second, independent cost: that one grades trials by the
/// deviation of the *projected* mapping on a Gauss-Legendre grid, which needs the
/// collocation and sampling machinery of `_bezier_degree.py` as well.
///
/// ## Which arithmetic each part runs in
///
/// Three widths appear here and they are not the same, which is
/// `design/backend_parity.md` Rule 9 in its ordinary form:
///
///  - **elevation and reduction** run at the kernels' own widths, which are
///    `accumulator_t<T>` -- `double` even for `float` storage -- because those
///    kernels' oracles are Numba and Numba promotes a `float64` scalar against a
///    `float32` array. Unchanged from `kernels_1d.hpp`;
///  - **the difference** `restored - original` is taken at the **storage** width,
///    because the oracle subtracts two arrays of the Bézier's own dtype before
///    anything widens it;
///  - **the norm** is `double` throughout whatever the Bézier stores, because the
///    oracle opens with `coeffs.astype(np.float64)`. So a `float32` Bézier's
///    reduction error is computed in `double` from a `float32` difference, and the
///    only narrow operation in the whole chain is that subtraction.
///
/// ## The norm is exact as a formula and bounded as a computation
///
/// `degree_reduction_error` reduces, elevates the result back -- an exact operation
/// on polynomials -- and takes the Bernstein-Gram norm of the coefficient difference,
/// so the value it returns is the true `||f - g||` over the domain rather than a
/// sample of it. That is a statement about the formula. **The computation carries the
/// roundings of an elevation, a reduction and a quadratic form**, and the oracle's
/// quadratic form goes through `np.tensordot`, which reaches BLAS, and `np.sum`,
/// whose pairwise order is not reproducible. So the parity claim is bounded, and
/// `tests/parity/test_bezier_degree.py` carries the derivation.
///
/// For a **rational** Bézier the norm is taken over the homogeneous coefficients,
/// weight column included, which is not the error of the projected mapping. The
/// oracle documents that at its public surface and this reproduces it; it is not a
/// defect to be fixed here.
///
/// ## Validating rather than asserting
///
/// Like `bezier.hpp` and `evaluate.hpp`, and unlike `kernels_1d.hpp`: these are
/// operations on a domain type rather than Layer 3 kernels, so they validate and
/// throw `std::invalid_argument` in a release build as much as a debug one. A caller
/// with no Python cannot be protected by `cpp/bindings/`.

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "pantr/bezier/axis_layout.hpp"
#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/kernels_1d.hpp"
#include "pantr/core/binomial.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/reduction_operator.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::bezier {

namespace detail {

/// Refuse a degree the exact-integer binomial recurrence cannot reach.
///
/// The message is the oracle's, character for character
/// (`_bspline_degree_core._check_bincoeff_envelope`), because
/// `tests/parity/test_bezier_degree.py` compares it.
///
/// \param n Largest upper index the elevation will need.
/// \param what Description of the operation, opening the message.
/// \throws std::invalid_argument If `n` exceeds `core::kBincoeffMaxN`.
inline void require_bincoeff_envelope(std::size_t n, const std::string& what) {
    if (n > static_cast<std::size_t>(core::kBincoeffMaxN)) {
        throw std::invalid_argument(
            what + " needs binomial coefficients up to C(" + std::to_string(n)
            + ", k), beyond the largest upper index " + std::to_string(core::kBincoeffMaxN)
            + " that pantr's exact-integer binomial kernel can compute without an int64 "
              "overflow. Past that the coefficients wrap silently and the result is "
              "corrupted rather than merely inaccurate.");
    }
}

/// Refuse a per-direction argument whose length is not the parametric dimension.
///
/// \param count The length given.
/// \param dim The dimension required.
/// \param what The argument's name, for the message.
/// \throws std::invalid_argument If the two differ.
inline void require_per_direction(std::size_t count, std::size_t dim, const char* what) {
    if (count != dim) {
        throw std::invalid_argument(std::string(what) + " must have one entry per parametric "
                                    "direction (" + std::to_string(dim) + "), got "
                                    + std::to_string(count) + ".");
    }
}

}  // namespace detail

/// Degree-elevate a Bézier in one or more parametric directions.
///
/// Exact: the elevated Bézier is the same mapping, written at a higher degree.
///
/// \param bezier The Bézier to elevate.
/// \param increments Degrees to add per direction, `bezier.dim()` of them. A zero
///        leaves its direction untouched.
/// \return The elevated Bézier, rational exactly when `bezier` is.
/// \throws std::invalid_argument If `increments` has the wrong length, or if an
///         elevated degree would leave the exact-integer binomial envelope.
template <Real T>
[[nodiscard]] Bezier<T> elevate_degree(const Bezier<T>& bezier,
                                       std::span<const std::size_t> increments) {
    const std::size_t dim = bezier.dim();
    detail::require_per_direction(increments.size(), dim, "increments");

    // Every envelope check runs before any elevation, which is the oracle's order:
    // two directions out of range must produce the first one's message on both sides.
    for (std::size_t d = 0; d < dim; ++d) {
        if (increments[d] > 0) {
            const std::size_t elevated = bezier.degree(d) + increments[d];
            detail::require_bincoeff_envelope(elevated, "Degree elevation to degree "
                                                            + std::to_string(elevated)
                                                            + " in direction "
                                                            + std::to_string(d));
        }
    }

    const ControlNet<T>& net = bezier.net();
    std::vector<std::size_t> shape(net.shape().begin(), net.shape().end());
    std::vector<T> values(net.values().begin(), net.values().end());

    for (std::size_t d = 0; d < dim; ++d) {
        if (increments[d] == 0) {
            continue;
        }
        const std::size_t along = shape[d];
        const std::size_t trailing = detail::extent_product(shape, 0) / along;
        const std::size_t elevated = along + increments[d];

        std::vector<T> moved(along * trailing);
        detail::gather_axis_to_front<T>(values, shape, d, moved);

        std::vector<T> raised(elevated * trailing);
        degree_elevate_bezier_1d<T>(static_cast<int>(along) - 1,
                                    span2d<const T>(moved.data(), along, trailing),
                                    static_cast<int>(increments[d]),
                                    span2d<T>(raised.data(), elevated, trailing));

        values.assign(elevated * trailing, T(0));
        detail::scatter_axis_from_front<T>(raised, shape, d, elevated, values);
        shape[d] = elevated;
    }

    return Bezier<T>(ControlNet<T>(std::span<const T>(values), std::span<const std::size_t>(shape)),
                     bezier.is_rational());
}

/// Degree-reduce a Bézier in one or more parametric directions.
///
/// An approximation in general, exact at the boundary of the parametric domain: the
/// operators the oracle supplies interpolate the endpoints.
///
/// \param bezier The Bézier to reduce.
/// \param decrements Degrees to drop per direction, `bezier.dim()` of them. A zero
///        leaves its direction untouched.
/// \param operators One reduction operator per direction, `bezier.dim()` of them,
///        each `(degree - decrement + 1, degree + 1)`. Entries for directions with a
///        zero decrement are not read. See the file comment for why these are
///        supplied rather than assembled.
/// \return The reduced Bézier, rational exactly when `bezier` is.
/// \throws std::invalid_argument If a length is wrong, if a decrement exceeds its
///         direction's degree, or if an operator's shape does not match its
///         direction.
template <Real T>
[[nodiscard]] Bezier<T> reduce_degree(const Bezier<T>& bezier,
                                      std::span<const std::size_t> decrements,
                                      std::span<const span2d<const double>> operators) {
    const std::size_t dim = bezier.dim();
    detail::require_per_direction(decrements.size(), dim, "decrements");
    detail::require_per_direction(operators.size(), dim, "operators");

    for (std::size_t d = 0; d < dim; ++d) {
        if (decrements[d] == 0) {
            continue;
        }
        if (decrements[d] > bezier.degree(d)) {
            throw std::invalid_argument("Degree decrement " + std::to_string(decrements[d])
                                        + " exceeds the degree " + std::to_string(bezier.degree(d))
                                        + " in direction " + std::to_string(d) + ".");
        }
        const std::size_t rows = bezier.degree(d) - decrements[d] + 1;
        const std::size_t cols = bezier.degree(d) + 1;
        if (operators[d].extent(0) != rows || operators[d].extent(1) != cols) {
            throw std::invalid_argument(
                "The reduction operator for direction " + std::to_string(d) + " must have shape ("
                + std::to_string(rows) + ", " + std::to_string(cols) + ").");
        }
    }

    const ControlNet<T>& net = bezier.net();
    std::vector<std::size_t> shape(net.shape().begin(), net.shape().end());
    std::vector<T> values(net.values().begin(), net.values().end());

    for (std::size_t d = 0; d < dim; ++d) {
        if (decrements[d] == 0) {
            continue;
        }
        const std::size_t along = shape[d];
        const std::size_t trailing = detail::extent_product(shape, 0) / along;
        const std::size_t reduced = operators[d].extent(0);

        std::vector<T> moved(along * trailing);
        detail::gather_axis_to_front<T>(values, shape, d, moved);

        std::vector<T> shrunk(reduced * trailing);
        core::apply_reduction_operator<T>(operators[d],
                                          span2d<const T>(moved.data(), along, trailing),
                                          span2d<T>(shrunk.data(), reduced, trailing));

        values.assign(reduced * trailing, T(0));
        detail::scatter_axis_from_front<T>(shrunk, shape, d, reduced, values);
        shape[d] = reduced;
    }

    return Bezier<T>(ControlNet<T>(std::span<const T>(values), std::span<const std::size_t>(shape)),
                     bezier.is_rational());
}

/// The squared Bernstein-Gram `L2` norm of one component's coefficient tensor.
///
/// `||p||^2 = c^T G c` with `G` the tensor-product Bernstein mass matrix, applied one
/// axis at a time so the Kronecker product is never formed. Everything is `double`,
/// whatever the Bézier stores, because the oracle opens by casting to `float64`.
///
/// \param coefficients The coefficients, row-major with extents `shape`.
/// \param shape The parametric extents, without a component axis.
/// \param grams One Bernstein Gram matrix per axis, each `(shape[a], shape[a])`.
/// \return The squared norm, non-negative: the analytic value is, and the absolute
///         value guards a small negative from round-off, as the oracle does.
[[nodiscard]] inline double squared_l2_norm(std::span<const double> coefficients,
                                            std::span<const std::size_t> shape,
                                            std::span<const span2d<const double>> grams) {
    const std::size_t total = detail::extent_product(shape, 0);
    std::vector<double> front(coefficients.begin(), coefficients.end());
    std::vector<double> back(total);

    for (std::size_t a = 0; a < shape.size(); ++a) {
        const std::size_t outer = detail::extent_product(shape.subspan(0, a), 0);
        const std::size_t along = shape[a];
        const std::size_t inner = detail::extent_product(shape, a + 1);

        for (std::size_t o = 0; o < outer; ++o) {
            for (std::size_t i = 0; i < along; ++i) {
                for (std::size_t n = 0; n < inner; ++n) {
                    double acc = 0.0;
                    for (std::size_t j = 0; j < along; ++j) {
                        acc += at(grams[a], i, j) * front[(((o * along) + j) * inner) + n];
                    }
                    back[(((o * along) + i) * inner) + n] = acc;
                }
            }
        }
        front.swap(back);
    }

    using std::abs;

    double total_sum = 0.0;
    for (std::size_t i = 0; i < total; ++i) {
        total_sum += coefficients[i] * front[i];
    }
    return abs(total_sum);
}

/// The `L2` error a degree reduction would introduce.
///
/// Reduces, elevates the result back to the original degrees, and takes the
/// Bernstein-Gram norm of the coefficient difference, so the value is
/// `||f - g||` over the domain rather than a sample of it -- **as a formula**; see the
/// file comment for what the computation of it carries. Components are combined in
/// the Euclidean sense, and for a rational Bézier the weight column is one of them.
///
/// \param bezier The Bézier that would be reduced.
/// \param decrements Degrees to drop per direction.
/// \param operators One reduction operator per direction, as `reduce_degree` takes.
/// \param grams One Bernstein Gram matrix per direction, each square of the
///        direction's original order. Supplied rather than assembled; see the file
///        comment.
/// \return The `L2` norm of the error, in the units of the control points.
/// \throws std::invalid_argument If any length or shape is wrong.
template <Real T>
[[nodiscard]] double degree_reduction_error(const Bezier<T>& bezier,
                                            std::span<const std::size_t> decrements,
                                            std::span<const span2d<const double>> operators,
                                            std::span<const span2d<const double>> grams) {
    const std::size_t dim = bezier.dim();
    detail::require_per_direction(grams.size(), dim, "grams");
    for (std::size_t d = 0; d < dim; ++d) {
        const std::size_t order = bezier.degree(d) + 1;
        if (grams[d].extent(0) != order || grams[d].extent(1) != order) {
            throw std::invalid_argument("The Gram matrix for direction " + std::to_string(d)
                                        + " must have shape (" + std::to_string(order) + ", "
                                        + std::to_string(order) + ").");
        }
    }

    const Bezier<T> reduced = reduce_degree(bezier, decrements, operators);
    const Bezier<T> restored = elevate_degree(reduced, decrements);

    const std::span<const T> original = bezier.net().values();
    const std::span<const T> back = restored.net().values();
    const std::size_t components = bezier.net().num_components();
    const std::size_t coefficients = original.size() / components;

    std::vector<std::size_t> parametric(bezier.net().shape().begin(),
                                        bezier.net().shape().end() - 1);
    std::vector<double> component(coefficients);

    using std::sqrt;

    double total = 0.0;
    for (std::size_t r = 0; r < components; ++r) {
        for (std::size_t i = 0; i < coefficients; ++i) {
            // The subtraction is at the STORAGE width, which is the only narrow
            // operation in this function: the oracle differences two arrays of the
            // Bézier's own dtype and casts to float64 only inside the norm.
            component[i] = static_cast<double>(
                static_cast<T>(back[(i * components) + r] - original[(i * components) + r]));
        }
        total += squared_l2_norm(component, parametric, grams);
    }
    return sqrt(total);
}

}  // namespace pantr::bezier
