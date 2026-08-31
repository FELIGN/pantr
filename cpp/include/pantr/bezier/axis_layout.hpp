#pragma once

/// \file
/// Moving one axis of a row-major control net to the front, and back again.
///
/// Every operation that acts along a single parametric direction -- degree elevation
/// and reduction, restriction, splitting, slicing -- wants its axis contiguous and
/// leading, because the 1-D kernels take a `(n_along, trailing)` matrix. The Python
/// oracle spells that `_flatten_along_axis` / `_unflatten_along_axis`
/// (`src/pantr/_array_utils.py`); this is the same permutation in C++.
///
/// **Shared rather than copied, and that is the point of the file.** Three headers
/// need it -- `degree.hpp`, `shape.hpp` and, for its extent products, `evaluate.hpp`
/// -- and three copies of an index permutation is three places for an off-by-one to
/// hide, in code where an off-by-one silently transposes a surface rather than
/// crashing. `binomial.hpp` states the same reasoning for the same reason.
///
/// **Nothing here computes.** These functions move values and multiply extents; they
/// commit no floating-point operation at all, so they sit outside every parity claim
/// in the package. That is what lets an operation built on them inherit its kernel's
/// claim unchanged.

#include <cstddef>
#include <span>

#include "pantr/core/scalar.hpp"

namespace pantr::bezier::detail {

/// The number of values spanned by `shape[from]` through `shape.back()`.
///
/// \param shape The extents, innermost last.
/// \param from The first extent to count, so a partially contracted block can be
///        sized without rebuilding its shape.
/// \return The product, and 1 when the range is empty.
[[nodiscard]] inline std::size_t extent_product(std::span<const std::size_t> shape,
                                                std::size_t from) noexcept {
    std::size_t size = 1;
    for (std::size_t d = from; d < shape.size(); ++d) {
        size *= shape[d];
    }
    return size;
}

/// Move one axis to the front, as `_flatten_along_axis` does.
///
/// The oracle reshapes to `(shape[axis], everything else)` before handing a kernel a
/// two-dimensional view, and the trailing block it produces runs over the axes before
/// `axis` and then the axes after it, in that order. This reproduces that layout.
///
/// \param values The array, row-major with extents `shape`.
/// \param shape The extents.
/// \param axis The axis to bring to the front.
/// \param out The permuted array, `(shape[axis], trailing)` row-major.
template <Real T>
void gather_axis_to_front(std::span<const T> values, std::span<const std::size_t> shape,
                          std::size_t axis, std::span<T> out) {
    const std::size_t outer = extent_product(shape.subspan(0, axis), 0);
    const std::size_t along = shape[axis];
    const std::size_t inner = extent_product(shape, axis + 1);
    const std::size_t trailing = outer * inner;

    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t i = 0; i < along; ++i) {
            for (std::size_t n = 0; n < inner; ++n) {
                out[(i * trailing) + (o * inner) + n] = values[(((o * along) + i) * inner) + n];
            }
        }
    }
}

/// Invert `gather_axis_to_front`, for a possibly different extent along the axis.
///
/// The extent may change, which is the whole point: elevation, reduction, restriction
/// and splitting all return a different number of coefficients along the axis they
/// act on.
///
/// \param moved The permuted array, `(along, trailing)` row-major.
/// \param shape The original extents; `shape[axis]` is ignored.
/// \param axis The axis that was brought to the front.
/// \param along The new extent along that axis.
/// \param out The array in the original layout, with `shape[axis]` replaced by
///        `along`.
template <Real T>
void scatter_axis_from_front(std::span<const T> moved, std::span<const std::size_t> shape,
                             std::size_t axis, std::size_t along, std::span<T> out) {
    const std::size_t outer = extent_product(shape.subspan(0, axis), 0);
    const std::size_t inner = extent_product(shape, axis + 1);
    const std::size_t trailing = outer * inner;

    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t i = 0; i < along; ++i) {
            for (std::size_t n = 0; n < inner; ++n) {
                out[(((o * along) + i) * inner) + n] = moved[(i * trailing) + (o * inner) + n];
            }
        }
    }
}

}  // namespace pantr::bezier::detail
