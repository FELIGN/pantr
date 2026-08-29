#pragma once

/// \file
/// The control-point array of a tensor-product geometry, and the shape questions
/// it answers.
///
/// ## Why this is a type and not just a vector plus a shape
///
/// A control net is `(*num_basis, num_components)`: one coefficient per
/// tensor-product index, each of `num_components` scalars, laid out row-major so
/// that the last axis is contiguous. Every consumer needs the same four answers
/// from it -- how many parametric directions, how long each one is, how many
/// components, and how many coefficients in total -- and every one of them is a
/// subtraction or a product away from being wrong in a way that walks off the
/// allocation. Answering them once, beside the storage, is the whole of this
/// type's job.
///
/// It is deliberately *not* a Bézier. Nothing here knows what the extents mean:
/// for `pantr::bezier::Bezier` an extent is `degree + 1`, for a B-spline it is a
/// basis count, and neither reading belongs to the array. `degree()` therefore
/// lives on `Bezier` and `extent()` lives here.
///
/// ## Parity notes for the Python oracle
///
/// Two conveniences of `pantr.bezier.Bezier.__init__` are reproduced here rather
/// than in the Python wrapper, so that both languages meet one definition of what
/// a well-formed control net is:
///
///  - **A rank-1 shape `(n,)` is read as `(n, 1)`.** The oracle spells this
///    `cp[:, np.newaxis]` and documents it as "a 1D input of shape `(n,)` is
///    reshaped to `(n, 1)` (scalar field)". A C++ caller building a scalar curve
///    from a flat vector of coefficients wants exactly the same reading.
///  - **The two rejections carry the oracle's messages verbatim.** A rank-0 shape
///    and a zero-length parametric direction are the two the oracle raises, and
///    `tests/parity/test_bezier_type.py` asserts the text of both matches
///    character for character. Rewording one of them is a parity failure, not a
///    style change.
///
/// The oracle has no counterpart for the values-against-shape consistency check
/// below, and cannot: a numpy array's data and shape agree by construction. It is
/// here for the caller that has no numpy.

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "pantr/core/scalar.hpp"

namespace pantr::bezier {

/// The control points of a tensor-product geometry, owned by value.
///
/// Stores `size()` scalars in row-major order under the shape
/// `(extent(0), ..., extent(dim() - 1), num_components())`. Instances are
/// immutable: nothing here changes a net once it is built, and every derived net
/// is a new object.
///
/// The stored copy is the point. A net built from a caller's buffer does not
/// alias it, so no later write through that buffer can change a geometry that has
/// already been validated.
template <Real T>
class ControlNet {
  public:
    /// Build a net from a flat coefficient buffer and its shape, validating both.
    ///
    /// \param values The coefficients, row-major, `prod(shape)` of them.
    /// \param shape The extents; the last is the component count. A rank-1 shape
    ///        `(n,)` is read as `(n, 1)`.
    /// \throws std::invalid_argument If `shape` is empty, if any parametric
    ///         extent is zero, or if `values` does not have `prod(shape)` entries.
    ControlNet(std::span<const T> values, std::span<const std::size_t> shape)
        : values_(values.begin(), values.end()), shape_(shape.begin(), shape.end()) {
        if (shape_.empty()) {
            throw std::invalid_argument("Control points must be at least 1D.");
        }
        if (shape_.size() == 1) {
            shape_.push_back(1);
        }
        for (std::size_t d = 0; d + 1 < shape_.size(); ++d) {
            if (shape_[d] < 1) {
                throw std::invalid_argument(
                    "Control points must have at least 1 entry in parametric direction "
                    + std::to_string(d) + ", got " + std::to_string(shape_[d]) + ".");
            }
        }
        std::size_t expected = 1;
        for (const std::size_t extent : shape_) {
            expected *= extent;
        }
        if (values_.size() != expected) {
            throw std::invalid_argument("Control points hold " + std::to_string(values_.size())
                                        + " values, but the shape asks for "
                                        + std::to_string(expected) + ".");
        }
    }

    /// The coefficients, row-major.
    ///
    /// \return A view of the stored values, valid while the net lives.
    [[nodiscard]] std::span<const T> values() const noexcept { return values_; }

    /// The shape, `(extent(0), ..., extent(dim() - 1), num_components())`.
    ///
    /// \return A view of the stored shape, valid while the net lives. Always at
    ///         least two entries long.
    [[nodiscard]] std::span<const std::size_t> shape() const noexcept { return shape_; }

    /// The number of parametric directions.
    ///
    /// \return `shape().size() - 1`, at least 1.
    [[nodiscard]] std::size_t dim() const noexcept { return shape_.size() - 1; }

    /// The extent along one parametric direction.
    ///
    /// \param d The direction, in `[0, dim())`.
    /// \return The number of coefficients along `d`.
    /// \throws std::out_of_range If `d >= dim()`.
    [[nodiscard]] std::size_t extent(std::size_t d) const {
        if (d >= dim()) {
            throw std::out_of_range("ControlNet.extent: direction " + std::to_string(d)
                                    + " is out of range for dim " + std::to_string(dim()) + ".");
        }
        return shape_[d];
    }

    /// The length of the component axis, weights included.
    ///
    /// This is the *stored* width. `pantr::bezier::Bezier::rank` is the width a
    /// caller sees, which is one less for a rational geometry.
    ///
    /// \return `shape().back()`.
    [[nodiscard]] std::size_t num_components() const noexcept { return shape_.back(); }

    /// The total number of stored scalars.
    ///
    /// \return `values().size()`, which equals the product of the shape.
    [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }

  private:
    std::vector<T> values_;             ///< Coefficients, row-major.
    std::vector<std::size_t> shape_;    ///< Extents, at least two of them.
};

}  // namespace pantr::bezier
