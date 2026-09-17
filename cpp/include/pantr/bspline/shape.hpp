#pragma once

/// \file
/// The shape operations on a B-spline field: reversal of one direction, permutation
/// of the directions, and the affine transform of the control points.
///
/// Free functions over `Bspline<T>`, as `bspline.hpp` says, and the B-spline twins of
/// `pantr/bezier/shape.hpp`'s first three. A Bézier has no knot vector and no space,
/// so `reverse` and `permute_directions` here carry the part that has no Bézier
/// counterpart: reseating the field's space.
///
/// ## Two kinds of operation, and they carry two kinds of claim
///
/// **Exact rearrangements.** The *control points* of all three operations are moved
/// or copied and never arithmetically combined -- the oracle spells them `np.flip`,
/// `np.roll`, `np.transpose` and, for `transform`, a copy of the weight column -- so
/// the two backends agree bit for bit by construction, at any storage format and on
/// any build. `permute_directions` computes nothing at all: it reorders the space
/// handles and the net's axes, and a differing knot could only be a lost entry.
///
/// **`reverse`'s reflected knot vector is arithmetic, and it is still exact across
/// the two backends.** The oracle writes `new_knots = (a + b) - knots[::-1]`, with
/// `a` and `b` this direction's domain ends, and this header transcribes that
/// expression operand for operand and in that order. Both operands are storage-format
/// values that the two backends already agree on bitwise, IEEE-754 addition and
/// subtraction are correctly rounded, and there is **no multiplication for a fused
/// multiply-add to absorb** -- so no `-ffp-contract` setting and no target ISA can
/// separate the two results. That argument is why the parity suite claims this vector
/// bitwise rather than within a rounding budget; it does **not** say the expression is
/// error-free. `(a + b) - k` cancels near either domain end, so the reflected value's
/// error against the exact reflection is bounded relative to `|a| + |b|` and not
/// relative to the possibly-tiny result. Both backends make the same error, which is
/// what makes the *parity* claim exact while the *accuracy* claim is not.
///
/// A knot vector reflected this way is still non-decreasing and still clamped, so the
/// space it builds needs no repair; snapping stays on, which is the oracle's
/// `BsplineSpace1D(new_knots, degree, periodic=...)` taking its default.
///
/// **A contraction.** `transform` is `cp @ A.T + b` in the oracle, a matrix product
/// that reaches BLAS, whose summation order is not reproducible. It carries a bounded
/// claim and no bitwise arm, exactly as `pantr/bezier/shape.hpp`'s `transform` does
/// and for the same reason. The weight column of a rational field is copied, not
/// computed, and stays in the exact regime.
///
/// ## `transform` never converts an affine map between backends
///
/// It takes the **matrix and the offset**, not an `AffineTransform`, for the reason
/// `pantr/bezier/shape.hpp` states at length: only arrays cross, so no
/// `AffineTransform` implementation is ever converted into the other, which is the
/// shape `design/cross_backend_types.md` forbids. The matrix is `double` whatever the
/// field stores, because that is what an `AffineTransform` holds, and it is **cast to
/// the storage format before any multiplication** -- the oracle's
/// `matrix.astype(dtype)`. Casting after would run the product in `double` and move
/// the answer at `float32`.
///
/// ## The space is reseated, never rebuilt
///
/// `permute_directions` reorders the space's own `shared_ptr` handles and builds no
/// univariate space at all; `reverse` builds exactly one, for the direction it
/// reflects, and carries every other handle through; `transform` passes the whole
/// space handle through untouched. That is what makes
/// `design/bspline_ownership_lifetime.md`'s F6 identity contract hold under this
/// backend: a direction the operation left alone comes back as the object that went
/// in, so the Python wrapper's `_wrap_over` can reuse its wrapper rather than
/// rebuilding an equal-valued one.
///
/// ## The `in_place` flag is not here
///
/// The oracle's `reverse`, `permute_directions` and `transform` each take `in_place=`,
/// returning `None` when it is set. `bspline.hpp` rules that this type offers no
/// mutator of any kind; the flag is a property of the *wrapper*, which owns the
/// mutation and reseats its implementation through `Bspline._mutate`. These functions
/// return a new field and the wrapper decides what to do with it.
///
/// ## Validating rather than asserting
///
/// Like `bspline.hpp`, `structural.hpp` and `degree.hpp`: operations on a domain type
/// validate and throw `std::invalid_argument` in a release build as much as a debug
/// one. A caller with no Python cannot be protected by `cpp/bindings/`.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "pantr/bezier/axis_layout.hpp"
#include "pantr/bezier/control_net.hpp"
#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/bspline/space_nd.hpp"
#include "pantr/bspline/structural.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::bspline {

namespace detail {

/// Reflect a knot vector about the midpoint of its own domain.
///
/// The oracle's `new_knots = (a + b) - knots[::-1]`, transcribed operand for operand:
/// `a + b` is formed once, in the storage format, and subtracted from each knot read
/// back to front. See the file comment for why that order is what makes the result
/// bitwise reproducible, and for what it does *not* claim about accuracy.
///
/// \tparam T The scalar type the knots are stored in.
/// \param knots The knot vector, non-decreasing.
/// \param degree The polynomial degree, which is what picks the domain ends out.
/// \return The reflected vector, non-decreasing and of the same length.
template <Real T>
[[nodiscard]] std::vector<T> reflected_knots(std::span<const T> knots, std::int64_t degree) {
    const std::size_t first = static_cast<std::size_t>(degree);
    const std::size_t last = knots.size() - first - 1;
    const T sum = static_cast<T>(knots[first] + knots[last]);

    std::vector<T> reflected(knots.size());
    for (std::size_t i = 0; i < knots.size(); ++i) {
        reflected[i] = static_cast<T>(sum - knots[knots.size() - 1 - i]);
    }
    return reflected;
}

/// Refuse a permutation that is not one, with the oracle's message.
///
/// The text is `Bspline.permute_directions`' own, character for character, because
/// `tests/parity/test_bspline_shape.py` compares it. The oracle renders the argument
/// as a Python list, which is why the brackets and the `", "` separator are here.
///
/// \param permutation The candidate permutation.
/// \param dim The field's number of parametric directions.
/// \throws std::invalid_argument If it is not a permutation of `[0, dim)`.
inline void require_permutation(std::span<const std::int64_t> permutation, std::int64_t dim) {
    const std::size_t width = static_cast<std::size_t>(dim);
    std::vector<bool> seen(width, false);
    bool valid = permutation.size() == width;
    if (valid) {
        for (const std::int64_t entry : permutation) {
            if (entry < 0 || entry >= dim || seen[static_cast<std::size_t>(entry)]) {
                valid = false;
                break;
            }
            seen[static_cast<std::size_t>(entry)] = true;
        }
    }
    if (!valid) {
        std::string given;
        for (std::size_t k = 0; k < permutation.size(); ++k) {
            given += (k != 0 ? ", " : "") + std::to_string(permutation[k]);
        }
        throw std::invalid_argument("permutation must be a permutation of range("
                                    + std::to_string(dim) + "), got [" + given + "].");
    }
}

}  // namespace detail

/// The field with one parametric direction reversed.
///
/// The control points are read back to front along `direction` and that direction's
/// knot vector is reflected about its own domain midpoint, so the geometry is
/// unchanged and only the parametrization runs the other way. A periodic direction
/// needs a cyclic shift on top of the flip: its stored net expands as
/// `full[j] = stored[j % n_stored]`, so reversing the *full* sequence is a flip
/// followed by a roll of `n_full - n_stored`, the ghost count. Every other
/// direction's space handle is carried into the result.
///
/// \param field The field to reverse.
/// \param direction The direction to reverse, in `[0, field.dim())`.
/// \return The reversed field.
/// \throws std::invalid_argument If `direction` is out of range, with the oracle's
///         Layer 1 message.
template <Real T>
[[nodiscard]] Bspline<T> reverse(const Bspline<T>& field, std::int64_t direction) {
    detail::require_axis("direction", direction, field.dim());

    const BsplineSpace<T>& space = field.space_ref();
    const std::size_t axis = static_cast<std::size_t>(direction);
    const BsplineSpace1D<T>& reversed_direction = *space.spaces()[axis];

    std::vector<std::shared_ptr<const BsplineSpace1D<T>>> directions(space.spaces().begin(),
                                                                    space.spaces().end());
    const std::vector<T> reflected =
        detail::reflected_knots<T>(reversed_direction.knots(), reversed_direction.degree());
    directions[axis] = std::make_shared<const BsplineSpace1D<T>>(
        std::span<const T>(reflected), reversed_direction.degree(),
        reversed_direction.periodic(), KnotSnapping::merge_near_duplicates);

    const typename Bspline<T>::net_type& net = field.net();
    const std::span<const std::size_t> shape = net.shape();
    const std::span<const T> values = net.values();

    const std::size_t outer = pantr::bezier::detail::extent_product(shape.subspan(0, axis), 0);
    const std::size_t along = shape[axis];
    const std::size_t inner = pantr::bezier::detail::extent_product(shape, axis + 1);

    // A non-periodic direction is a plain flip; a periodic one is that flip rolled by
    // its ghost count, which is the oracle's `np.roll(flipped, shift, axis)` and
    // therefore reads `flipped[(i - shift) mod along]`.
    std::size_t shift = 0;
    if (reversed_direction.periodic()) {
        const std::size_t order = static_cast<std::size_t>(reversed_direction.degree()) + 1;
        const std::size_t full = reversed_direction.knots().size() - order;
        shift = (full - along) % along;
    }

    std::vector<T> out(values.size());
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t i = 0; i < along; ++i) {
            const std::size_t rolled = (i + along - shift) % along;
            const std::size_t source = along - 1 - rolled;
            for (std::size_t n = 0; n < inner; ++n) {
                out[(((o * along) + i) * inner) + n] =
                    values[(((o * along) + source) * inner) + n];
            }
        }
    }

    return detail::assemble<T>(std::move(directions), out,
                               std::vector<std::size_t>(shape.begin(), shape.end()),
                               field.is_rational());
}

/// The field with its parametric directions reordered.
///
/// New direction `k` is old direction `permutation[k]`, which is the oracle's
/// convention. Nothing is computed: the space handles are reordered and the net's
/// axes transposed, so every direction comes back as the object that went in.
///
/// \param field The field to permute.
/// \param permutation A permutation of `[0, field.dim())`.
/// \return The permuted field.
/// \throws std::invalid_argument If `permutation` is not one, with the oracle's
///         Layer 1 message.
template <Real T>
[[nodiscard]] Bspline<T> permute_directions(const Bspline<T>& field,
                                            std::span<const std::int64_t> permutation) {
    const std::int64_t dim = field.dim();
    detail::require_permutation(permutation, dim);

    const std::span<const std::shared_ptr<const BsplineSpace1D<T>>> old_directions =
        field.space_ref().spaces();
    const std::size_t width = static_cast<std::size_t>(dim);

    std::vector<std::shared_ptr<const BsplineSpace1D<T>>> directions(width);
    for (std::size_t k = 0; k < width; ++k) {
        directions[k] = old_directions[static_cast<std::size_t>(permutation[k])];
    }

    const typename Bspline<T>::net_type& net = field.net();
    const std::span<const std::size_t> shape = net.shape();
    const std::span<const T> values = net.values();
    const std::size_t components = net.num_components();

    std::vector<std::size_t> permuted(width + 1);
    for (std::size_t k = 0; k < width; ++k) {
        permuted[k] = shape[static_cast<std::size_t>(permutation[k])];
    }
    permuted[width] = components;

    // Strides of the SOURCE layout, in COEFFICIENTS rather than in values -- which is
    // why the component axis is dropped before the product is taken, and why the read
    // below multiplies by `components` exactly once.
    const std::span<const std::size_t> parametric = shape.subspan(0, width);
    std::vector<std::size_t> source_stride(width);
    for (std::size_t d = 0; d < width; ++d) {
        source_stride[d] = pantr::bezier::detail::extent_product(parametric, d + 1);
    }

    std::vector<T> out(values.size());
    std::vector<std::size_t> index(width, 0);
    const std::size_t coefficients = values.size() / components;
    for (std::size_t flat = 0; flat < coefficients; ++flat) {
        std::size_t source = 0;
        for (std::size_t k = 0; k < width; ++k) {
            source += index[k] * source_stride[static_cast<std::size_t>(permutation[k])];
        }
        for (std::size_t c = 0; c < components; ++c) {
            out[(flat * components) + c] = values[(source * components) + c];
        }
        for (std::size_t k = width; k-- > 0;) {
            if (++index[k] < permuted[k]) {
                break;
            }
            index[k] = 0;
        }
    }

    return detail::assemble<T>(std::move(directions), out, permuted, field.is_rational());
}

/// The field with an affine map applied to its geometric coordinates.
///
/// For a rational field the weighted coordinates transform as
/// `w (A x + b) = A (w x) + w b`, so the weight column is copied rather than computed.
/// The space is untouched and its handle is passed through, which is what makes
/// `s.transform(t).space is s.space` hold on this backend as it does on the oracle.
/// See the file comment for why the map arrives as a matrix and an offset rather than
/// as an `AffineTransform`, and why it is cast before the product rather than after.
///
/// \param field The field to transform.
/// \param matrix The linear part, `(n, n)` with `n` the field's rank. Always `double`.
/// \param offset The translation, `n` values. Always `double`.
/// \return The transformed field.
/// \throws std::invalid_argument If either shape does not match the field's rank, the
///         first with the oracle's own message.
template <Real T>
[[nodiscard]] Bspline<T> transform(const Bspline<T>& field, span2d<const double> matrix,
                                   std::span<const double> offset) {
    const typename Bspline<T>::net_type& net = field.net();
    const std::size_t components = net.num_components();
    const std::size_t n = static_cast<std::size_t>(field.rank());

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
        const T weight = field.is_rational() ? values[base + n] : T(1);
        for (std::size_t i = 0; i < n; ++i) {
            T acc = T(0);
            for (std::size_t j = 0; j < n; ++j) {
                acc = static_cast<T>(acc + values[base + j] * linear[(i * n) + j]);
            }
            out[base + i] = static_cast<T>(acc + (weight * shift[i]));
        }
        if (field.is_rational()) {
            out[base + n] = values[base + n];
        }
    }

    return Bspline<T>(field.space(),
                      typename Bspline<T>::net_type(std::span<const T>(out), net.shape()),
                      field.is_rational());
}

}  // namespace pantr::bspline
