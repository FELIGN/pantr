#pragma once

/// \file
/// The shape-changing operations on an n-dimensional Bézier: reversal, permutation,
/// affine transformation, restriction, splitting, slicing, boundaries and collapse.
///
/// Free functions over `Bezier<T>`, as `bezier.hpp` says. Each composes over an
/// existing 1-D kernel or over `axis_layout.hpp`'s permutation, and none reimplements
/// one.
///
/// ## Three kinds of operation, and they carry three kinds of claim
///
/// **Exact rearrangements.** `reverse` and `permute_directions` move values and
/// compute nothing -- the oracle spells them `np.flip` and `np.transpose` -- so the
/// two backends agree bit for bit by construction, at any dtype and on any build.
/// There is no arithmetic for a fused multiply-add to change.
///
/// **Compositions over a 1-D kernel.** `restrict`, `split` and `slice` reduce to
/// `restrict_bezier_1d`, `split_bezier_1d` and `slice_bezier_1d` applied along one
/// axis, with `axis_layout.hpp` moving the axis and moving it back. They inherit
/// those kernels' claims exactly: bitwise where the build cannot fuse, and Rule 10's
/// budget where it can. `boundary` is `slice` at a parameter of 0 or 1, and inherits
/// through it.
///
/// **A contraction.** `collapse_along_axis` evaluates the Bernstein basis in every
/// direction but one and contracts, which the oracle does with `np.tensordot` --
/// reaching BLAS, whose summation order is not reproducible. So it carries a bounded
/// claim and no bitwise arm, the same situation `evaluate.hpp` records for its own
/// lattice entry point and for the same reason. `transform` is in this family too:
/// the oracle writes it `cp @ A.T + b`, which is a matrix product.
///
/// ## `transform` never converts an affine map between backends
///
/// It takes the **matrix and the offset**, not an `AffineTransform`. That is
/// deliberate and it is what the wrapper does too: `Bezier.transform` reads
/// `affine.matrix` and `affine.offset`, which are numpy arrays on either backend, so
/// no `AffineTransform` implementation is ever converted into the other -- the shape
/// `design/cross_backend_types.md` forbids. Only arrays cross, and the map's own
/// arithmetic stays on whichever side built it.
///
/// The matrix is `double` whatever the Bézier stores, because that is what an
/// `AffineTransform` holds, and it is **cast to the storage format before any
/// multiplication** -- the oracle's `matrix.astype(dtype)`. Casting after would
/// compute the product in `double` and change the answer at `float32`.
///
/// ## The `in_place` flag is not here
///
/// The oracle's `reverse`, `permute_directions` and `transform` each take
/// `in_place=`, returning `None` when it is set. That flag is a property of the
/// *wrapper*, which owns the mutation and rebuilds its implementation through
/// `Bezier._mutate_control_points`; nothing about it needs to cross. These functions
/// return a new value, and the wrapper decides what to do with it. Porting the flag
/// would mean a C++ function whose return type depends on a runtime bool, which is
/// not a thing, and would import a known API defect into a language that never had
/// it.
///
/// ## Validating rather than asserting
///
/// Like `bezier.hpp`, `evaluate.hpp` and `degree.hpp`: operations on a domain type
/// validate and throw `std::invalid_argument` in a release build as much as a debug
/// one. A caller with no Python cannot be protected by `cpp/bindings/`.

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pantr/basis/bernstein.hpp"
#include "pantr/bezier/axis_layout.hpp"
#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/kernels_1d.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::bezier {

namespace detail {

/// Refuse a direction index outside `[0, dim)`.
///
/// The message is the oracle's, character for character, because
/// `tests/parity/test_bezier_shape.py` compares it.
///
/// \param name The parameter's name, opening the message.
/// \param value The index given.
/// \param dim The parametric dimension.
/// \throws std::invalid_argument If the index is out of range.
inline void require_direction(const char* name, std::size_t value, std::size_t dim) {
    if (value >= dim) {
        throw std::invalid_argument(std::string(name) + " must be in [0, " + std::to_string(dim)
                                    + "), got " + std::to_string(value) + ".");
    }
}

/// Build a Bézier from a flat value array and its extents.
///
/// \param values The control points, row-major.
/// \param shape The extents, components last.
/// \param is_rational Whether the last component is a homogeneous weight.
/// \return The Bézier.
template <Real T>
[[nodiscard]] Bezier<T> from_values(std::span<const T> values,
                                    std::span<const std::size_t> shape, bool is_rational) {
    return Bezier<T>(ControlNet<T>(values, shape), is_rational);
}

}  // namespace detail

/// Reverse the orientation of one parametric direction.
///
/// A pure rearrangement: the coefficients along `direction` are read back to front and
/// nothing is computed, so this is bit-exact on both backends and on any build.
///
/// \param bezier The Bézier.
/// \param direction The direction to reverse, in `[0, bezier.dim())`.
/// \return The reversed Bézier.
/// \throws std::invalid_argument If `direction` is out of range.
template <Real T>
[[nodiscard]] Bezier<T> reverse(const Bezier<T>& bezier, std::size_t direction) {
    detail::require_direction("direction", direction, bezier.dim());

    const ControlNet<T>& net = bezier.net();
    const std::span<const std::size_t> shape = net.shape();
    const std::span<const T> values = net.values();

    const std::size_t outer = detail::extent_product(shape.subspan(0, direction), 0);
    const std::size_t along = shape[direction];
    const std::size_t inner = detail::extent_product(shape, direction + 1);

    std::vector<T> flipped(values.size());
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t i = 0; i < along; ++i) {
            const std::size_t mirrored = along - 1 - i;
            for (std::size_t n = 0; n < inner; ++n) {
                flipped[(((o * along) + i) * inner) + n] =
                    values[(((o * along) + mirrored) * inner) + n];
            }
        }
    }
    return detail::from_values<T>(flipped, shape, bezier.is_rational());
}

/// Reorder the parametric directions.
///
/// New direction `k` is old direction `permutation[k]`, which is the oracle's
/// convention. A pure rearrangement, so bit-exact for the same reason as `reverse`.
///
/// \param bezier The Bézier.
/// \param permutation A permutation of `[0, bezier.dim())`.
/// \return The permuted Bézier.
/// \throws std::invalid_argument If `permutation` is not one.
template <Real T>
[[nodiscard]] Bezier<T> permute_directions(const Bezier<T>& bezier,
                                           std::span<const std::size_t> permutation) {
    const std::size_t dim = bezier.dim();
    std::vector<bool> seen(dim, false);
    bool valid = permutation.size() == dim;
    for (const std::size_t entry : permutation) {
        if (entry >= dim || seen[entry]) {
            valid = false;
            break;
        }
        seen[entry] = true;
    }
    if (!valid) {
        std::string given;
        for (std::size_t k = 0; k < permutation.size(); ++k) {
            given += (k ? ", " : "") + std::to_string(permutation[k]);
        }
        throw std::invalid_argument("permutation must be a permutation of range("
                                    + std::to_string(dim) + "), got [" + given + "].");
    }

    const ControlNet<T>& net = bezier.net();
    const std::span<const std::size_t> shape = net.shape();
    const std::span<const T> values = net.values();
    const std::size_t components = net.num_components();

    std::vector<std::size_t> permuted(dim + 1);
    for (std::size_t k = 0; k < dim; ++k) {
        permuted[k] = shape[permutation[k]];
    }
    permuted[dim] = components;

    // Strides of the SOURCE layout, so a destination walked in order reads the right
    // source element: the destination's axis `k` steps by the source's stride for
    // axis `permutation[k]`.
    //
    // These are strides in COEFFICIENTS, not in values, which is why the component
    // axis is dropped before the product is taken. Leaving it in makes each stride a
    // factor of `components` too large, and the `* components` at the read below then
    // counts it twice -- which does not crash, does not change the shape, and simply
    // returns a differently transposed surface. Measured before the fix: a relative
    // difference of 3.4e4 against the oracle.
    const std::span<const std::size_t> parametric = shape.subspan(0, dim);
    std::vector<std::size_t> source_stride(dim);
    for (std::size_t d = 0; d < dim; ++d) {
        source_stride[d] = detail::extent_product(parametric, d + 1);
    }

    std::vector<T> out(values.size());
    std::vector<std::size_t> index(dim, 0);
    const std::size_t coefficients = values.size() / components;
    for (std::size_t flat = 0; flat < coefficients; ++flat) {
        std::size_t source = 0;
        for (std::size_t k = 0; k < dim; ++k) {
            source += index[k] * source_stride[permutation[k]];
        }
        for (std::size_t c = 0; c < components; ++c) {
            out[(flat * components) + c] = values[(source * components) + c];
        }
        for (std::size_t k = dim; k-- > 0;) {
            if (++index[k] < permuted[k]) {
                break;
            }
            index[k] = 0;
        }
    }
    return detail::from_values<T>(out, permuted, bezier.is_rational());
}

/// Apply an affine map to the geometric coordinates.
///
/// For a rational Bézier the weighted coordinates transform as
/// `w (A x + b) = A (w x) + w b`, so the weight column is left alone. See the file
/// comment for why the map arrives as a matrix and an offset rather than as an
/// `AffineTransform`.
///
/// \param bezier The Bézier.
/// \param matrix The linear part, `(n, n)` with `n` the geometric rank -- the
///        component count, less one when rational. Always `double`.
/// \param offset The translation, `n` values. Always `double`.
/// \return The transformed Bézier.
/// \throws std::invalid_argument If either shape does not match the geometric rank.
template <Real T>
[[nodiscard]] Bezier<T> transform(const Bezier<T>& bezier, span2d<const double> matrix,
                                  std::span<const double> offset) {
    const ControlNet<T>& net = bezier.net();
    const std::size_t components = net.num_components();
    const std::size_t n = bezier.is_rational() ? components - 1 : components;

    if (matrix.extent(0) != n || matrix.extent(1) != n) {
        throw std::invalid_argument("Transform dimension (" + std::to_string(matrix.extent(0))
                                    + ") does not match the geometric rank ("
                                    + std::to_string(n) + ") of the control points.");
    }
    if (offset.size() != n) {
        throw std::invalid_argument("The translation must have " + std::to_string(n)
                                    + " entries.");
    }

    // Cast to the storage format BEFORE multiplying, which is the oracle's
    // `matrix.astype(dtype)`. Casting after would run the product in double and move
    // the answer at float32.
    std::vector<T> linear(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            linear[(i * n) + j] = static_cast<T>(at(matrix, i, j));
        }
    }
    std::vector<T> shift(n);
    for (std::size_t i = 0; i < n; ++i) {
        shift[i] = static_cast<T>(offset[i]);
    }

    const std::span<const T> values = net.values();
    const std::size_t coefficients = values.size() / components;
    std::vector<T> out(values.size());

    for (std::size_t k = 0; k < coefficients; ++k) {
        const std::size_t base = k * components;
        const T weight = bezier.is_rational() ? values[base + n] : T(1);
        for (std::size_t i = 0; i < n; ++i) {
            T acc = T(0);
            for (std::size_t j = 0; j < n; ++j) {
                acc = static_cast<T>(acc + values[base + j] * linear[(i * n) + j]);
            }
            out[base + i] = static_cast<T>(acc + (weight * shift[i]));
        }
        if (bezier.is_rational()) {
            out[base + n] = values[base + n];
        }
    }
    return detail::from_values<T>(out, net.shape(), bezier.is_rational());
}

namespace detail {

/// Apply a one-directional kernel along one axis of a net, and rebuild the net.
///
/// The body every axis-wise shape operation shares: move the axis to the front, run
/// the kernel over the `(along, trailing)` matrix, move it back. The kernel is handed
/// as a callable so that restriction, splitting and slicing do not each grow their own
/// copy of the permutation.
///
/// \param values The control points, row-major with extents `shape`.
/// \param shape The extents.
/// \param axis The axis to act along.
/// \param produced The extent the kernel leaves along that axis.
/// \param run Called with the gathered `(along, trailing)` input and the
///        `(produced, trailing)` destination.
/// \param out The result, in the original layout with `shape[axis]` replaced.
template <Real T, class Kernel>
void along_axis(std::span<const T> values, std::span<const std::size_t> shape, std::size_t axis,
                std::size_t produced, Kernel run, std::span<T> out) {
    const std::size_t along = shape[axis];
    const std::size_t trailing = extent_product(shape, 0) / along;

    std::vector<T> moved(along * trailing);
    gather_axis_to_front<T>(values, shape, axis, moved);

    std::vector<T> result(produced * trailing);
    run(span2d<const T>(moved.data(), along, trailing),
        span2d<T>(result.data(), produced, trailing));

    scatter_axis_from_front<T>(result, shape, axis, produced, out);
}

}  // namespace detail

/// Restrict a Bézier to a sub-box of its parametric domain.
///
/// A direction whose bounds are the full `[0, 1]` is left untouched, which is the
/// oracle's own short-circuit and not an optimisation: running the two-pass
/// restriction with those bounds would commit roundings the oracle does not.
///
/// \param bezier The Bézier.
/// \param lower Lower bound per direction, `bezier.dim()` of them.
/// \param upper Upper bound per direction.
/// \return The restricted Bézier, on `[0, 1]^dim` again.
/// \throws std::invalid_argument If a length is wrong, if a bound leaves `[0, 1]` or
///         is inverted, or if every direction is the full domain.
template <Real T>
[[nodiscard]] Bezier<T> restrict(const Bezier<T>& bezier, std::span<const double> lower,
                                 std::span<const double> upper) {
    const std::size_t dim = bezier.dim();
    if (lower.size() != dim || upper.size() != dim) {
        throw std::invalid_argument("lower and upper must each have one entry per parametric "
                                    "direction (" + std::to_string(dim) + ").");
    }

    const ControlNet<T>& net = bezier.net();
    std::vector<std::size_t> shape(net.shape().begin(), net.shape().end());
    std::vector<T> values(net.values().begin(), net.values().end());

    bool restricted = false;
    for (std::size_t d = 0; d < dim; ++d) {
        if (!(lower[d] >= 0.0 && upper[d] <= 1.0 && lower[d] <= upper[d])) {
            throw std::invalid_argument("The bounds of direction " + std::to_string(d)
                                        + " must satisfy 0 <= lower <= upper <= 1.");
        }
        if (lower[d] == 0.0 && upper[d] == 1.0) {
            continue;
        }
        restricted = true;
        std::vector<T> out(values.size());
        detail::along_axis<T>(
            values, shape, d, shape[d],
            [&](span2d<const T> in, span2d<T> dest) {
                restrict_bezier_1d<T>(in, static_cast<T>(lower[d]), static_cast<T>(upper[d]),
                                      dest);
            },
            out);
        values.swap(out);
    }
    if (!restricted) {
        throw std::invalid_argument(
            "Bounds match the full domain; at least one direction must be restricted.");
    }
    return detail::from_values<T>(values, shape, bezier.is_rational());
}

/// Split a Bézier in two along one direction.
///
/// \param bezier The Bézier.
/// \param direction The direction to split, in `[0, bezier.dim())`.
/// \param value The parameter to split at, in `[0, 1]`.
/// \return The left and right halves, each reparametrised onto `[0, 1]`.
/// \throws std::invalid_argument If `direction` is out of range or `value` leaves
///         `[0, 1]`.
template <Real T>
[[nodiscard]] std::pair<Bezier<T>, Bezier<T>> split(const Bezier<T>& bezier,
                                                    std::size_t direction, T value) {
    detail::require_direction("direction", direction, bezier.dim());
    if (!(value >= T(0) && value <= T(1))) {
        throw std::invalid_argument("value must be in [0, 1].");
    }

    const ControlNet<T>& net = bezier.net();
    const std::span<const std::size_t> shape = net.shape();
    const std::span<const T> values = net.values();

    std::vector<T> left(values.size());
    std::vector<T> right(values.size());
    const std::size_t along = shape[direction];
    const std::size_t trailing = detail::extent_product(shape, 0) / along;

    std::vector<T> moved(along * trailing);
    detail::gather_axis_to_front<T>(values, shape, direction, moved);
    std::vector<T> out_left(along * trailing);
    std::vector<T> out_right(along * trailing);
    split_bezier_1d<T>(span2d<const T>(moved.data(), along, trailing), value,
                       span2d<T>(out_left.data(), along, trailing),
                       span2d<T>(out_right.data(), along, trailing));
    detail::scatter_axis_from_front<T>(out_left, shape, direction, along, left);
    detail::scatter_axis_from_front<T>(out_right, shape, direction, along, right);

    return {detail::from_values<T>(left, shape, bezier.is_rational()),
            detail::from_values<T>(right, shape, bezier.is_rational())};
}

/// Evaluate a one-dimensional Bézier at one parameter, in homogeneous components.
///
/// The `dim == 1` case of the oracle's `slice`, split out because its result is a
/// point rather than a Bézier and C++ cannot return one type or the other. The
/// projection of a rational result is the caller's, exactly as the oracle's is.
///
/// \param bezier A one-dimensional Bézier.
/// \param value The parameter, in `[0, 1]`.
/// \return The raw components at that parameter, weight column included.
/// \throws std::invalid_argument If the Bézier is not one-dimensional or `value`
///         leaves `[0, 1]`.
template <Real T>
[[nodiscard]] std::vector<T> slice_point(const Bezier<T>& bezier, T value) {
    if (bezier.dim() != 1) {
        throw std::invalid_argument("slice_point needs a one-dimensional Bézier, got dimension "
                                    + std::to_string(bezier.dim()) + ".");
    }
    if (!(value >= T(0) && value <= T(1))) {
        throw std::invalid_argument("value must be in [0, 1].");
    }

    const ControlNet<T>& net = bezier.net();
    const std::size_t components = net.num_components();
    std::vector<T> point(components);
    slice_bezier_1d<T>(span2d<const T>(net.values().data(), net.extent(0), components), value,
                       std::span<T>(point));
    return point;
}

/// Fix one parametric direction at a value, dropping it.
///
/// \param bezier The Bézier, of dimension at least two.
/// \param axis The direction to fix, in `[0, bezier.dim())`.
/// \param value The parameter, in `[0, 1]`.
/// \return The sliced Bézier, of dimension `bezier.dim() - 1`.
/// \throws std::invalid_argument If the Bézier is one-dimensional -- use
///         `slice_point` -- if `axis` is out of range, or if `value` leaves `[0, 1]`.
template <Real T>
[[nodiscard]] Bezier<T> slice(const Bezier<T>& bezier, std::size_t axis, T value) {
    if (bezier.dim() < 2) {
        throw std::invalid_argument(
            "slice needs a Bézier of dimension at least two; a one-dimensional one slices to a "
            "point, which slice_point returns.");
    }
    detail::require_direction("axis", axis, bezier.dim());
    if (!(value >= T(0) && value <= T(1))) {
        throw std::invalid_argument("value must be in [0, 1].");
    }

    const ControlNet<T>& net = bezier.net();
    const std::span<const std::size_t> shape = net.shape();
    const std::size_t along = shape[axis];
    const std::size_t trailing = detail::extent_product(shape, 0) / along;

    std::vector<T> moved(along * trailing);
    detail::gather_axis_to_front<T>(net.values(), shape, axis, moved);
    std::vector<T> flat(trailing);
    slice_bezier_1d<T>(span2d<const T>(moved.data(), along, trailing), value,
                       std::span<T>(flat));

    // The oracle reshapes the flat result to the trailing shape directly, so the axes
    // keep their order with `axis` removed -- which is what the gather produced.
    std::vector<std::size_t> reduced;
    reduced.reserve(shape.size() - 1);
    for (std::size_t d = 0; d < shape.size(); ++d) {
        if (d != axis) {
            reduced.push_back(shape[d]);
        }
    }
    return detail::from_values<T>(flat, reduced, bezier.is_rational());
}

/// One face of the parametric domain.
///
/// `boundary(b, axis, side)` is `slice(b, axis, side ? 1 : 0)`, which is the oracle's
/// own definition rather than a reimplementation of it.
///
/// \param bezier The Bézier, of dimension at least two.
/// \param axis The direction whose face is wanted.
/// \param side 0 for the face at parameter 0, 1 for the face at parameter 1.
/// \return The face, of dimension `bezier.dim() - 1`.
/// \throws std::invalid_argument If `side` is neither 0 nor 1, or as `slice` throws.
template <Real T>
[[nodiscard]] Bezier<T> boundary(const Bezier<T>& bezier, std::size_t axis, int side) {
    if (side != 0 && side != 1) {
        throw std::invalid_argument("side must be 0 or 1, got " + std::to_string(side) + ".");
    }
    detail::require_direction("axis", axis, bezier.dim());
    return slice<T>(bezier, axis, side == 0 ? T(0) : T(1));
}

/// Collapse to a univariate Bézier by fixing every direction but one.
///
/// The oracle contracts the directions from highest to lowest, skipping `axis`, so
/// that each direction's index still equals its original one when its turn comes.
/// This reproduces that order, which matters: the contractions are not associative in
/// floating point and a different order is a different answer.
///
/// \param bezier The Bézier, of dimension at least two.
/// \param axis The direction to keep.
/// \param values One parameter per collapsed direction, `bezier.dim() - 1` of them.
///        Entry `i` is direction `i` for `i < axis` and direction `i + 1` above it.
/// \return A one-dimensional Bézier along `axis`.
/// \throws std::invalid_argument If `axis` is out of range, if `values` has the wrong
///         length, or if a value leaves `[0, 1]`.
template <Real T>
[[nodiscard]] Bezier<T> collapse_along_axis(const Bezier<T>& bezier, std::size_t axis,
                                            std::span<const T> values) {
    const std::size_t dim = bezier.dim();
    if (dim < 2) {
        throw std::invalid_argument("collapse_along_axis needs a Bézier of dimension at least "
                                    "two, got " + std::to_string(dim) + ".");
    }
    detail::require_direction("axis", axis, dim);
    if (values.size() != dim - 1) {
        throw std::invalid_argument("values must have length dim - 1 = "
                                    + std::to_string(dim - 1) + ", got "
                                    + std::to_string(values.size()) + ".");
    }
    for (const T value : values) {
        if (!(value >= T(0) && value <= T(1))) {
            throw std::invalid_argument("All values must be in [0, 1].");
        }
    }

    const ControlNet<T>& net = bezier.net();
    std::vector<std::size_t> shape(net.shape().begin(), net.shape().end());
    std::vector<T> current(net.values().begin(), net.values().end());

    for (std::size_t d = dim; d-- > 0;) {
        if (d == axis) {
            continue;
        }
        const std::size_t value_index = d < axis ? d : d - 1;
        const std::size_t along = shape[d];

        std::vector<T> basis(along);
        tabulate_bernstein_1d<T>(static_cast<int>(along) - 1,
                                 std::span<const T>(&values[value_index], 1),
                                 span2d<T>(basis.data(), 1, along));

        const std::size_t outer = detail::extent_product(
            std::span<const std::size_t>(shape).subspan(0, d), 0);
        const std::size_t inner =
            detail::extent_product(std::span<const std::size_t>(shape), d + 1);

        std::vector<T> reduced(outer * inner);
        for (std::size_t o = 0; o < outer; ++o) {
            for (std::size_t n = 0; n < inner; ++n) {
                T acc = T(0);
                for (std::size_t i = 0; i < along; ++i) {
                    acc = static_cast<T>(acc + basis[i] * current[(((o * along) + i) * inner) + n]);
                }
                reduced[(o * inner) + n] = acc;
            }
        }
        current.swap(reduced);
        shape.erase(shape.begin() + static_cast<std::ptrdiff_t>(d));
    }
    return detail::from_values<T>(current, shape, bezier.is_rational());
}

}  // namespace pantr::bezier
