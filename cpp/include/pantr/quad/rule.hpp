#pragma once

/// \file
/// The reference quadrature rule on the unit cube, and the two factories that
/// build one.
///
/// ## Why this header contains a class
///
/// `design/cross_backend_types.md`'s 2026-08-27 supersession names
/// `QuadratureRule` among the domain types a C++ program has to be able to build
/// with no interpreter present. `AABB` and `AffineTransform` moved first and are
/// the working precedent; this is the same move, applied to `pantr.quad`.
///
/// Like those two, and unlike every kernel header in this directory, the type
/// **validates and throws** rather than asserting. It is the C++ counterpart of
/// Layer 2, not a Layer 3 kernel, so its checks stand in a release build; a
/// caller with no Python cannot be protected by `cpp/bindings/`.
///
/// ## `double` only, and not templated on the scalar
///
/// Two independent reasons, and the second is enforced.
///
///  1. **It is faithful.** `src/pantr/quad/_rule_nd.py` casts points and weights
///     to `float64` unconditionally, so there is no `float32` oracle a templated
///     instantiation could be parity against, and `design/backend_parity.md`
///     Rule 8 forbids a surface with no oracle behind it.
///  2. **`scripts/ci_local.sh` asserts it.** Exactly one `template <` may appear
///     under `cpp/include/pantr/quad`, and it must be the one in
///     `simple_rules.hpp`. The guard's recorded reason is measured: instantiating
///     Newton at `float` gave a 1.46e-3 relative weight error at n = 200, while a
///     double-then-narrow port of the Chebyshev nodes differed on 17% of
///     `float32` arguments. So this file declares no template of its own -- it
///     *uses* `span2d`, which is declared in `pantr/core/mdspan.hpp` and is not
///     under this directory.
///
/// ## `Rule1D`, and the defect class it closes
///
/// The tensor-product factory needs one 1-D rule per axis. Written as two
/// parallel sequences of spans it would admit a transposed call that pairs every
/// axis's nodes with the wrong axis's weights -- silently, with no type error and
/// no shape mismatch whenever the counts happen to agree. That is the defect
/// recorded by FELIGN/pantr#358 for `dedup_roots`, met a second time here, so the
/// pair is a named struct instead. It does not exist in the Python either; it is
/// new, and it exists to make the transposition unspellable rather than to tidy.
///
/// ## Parity notes for the Python oracle
///
/// Three places where reproducing `src/pantr/quad/_rule_nd.py` exactly takes care.
///
///  - **The weight of a tensor-product point is accumulated left to right.** The
///    oracle forms it as `np.prod(..., axis=0)` over an `(ndim, num_points)`
///    block, which reduces axis 0 in order, so `((w0 * w1) * w2)`. Any other
///    association differs in the last bits from three axes up.
///  - **Points are enumerated in C order, last axis fastest**, matching
///    `np.meshgrid(..., indexing="ij")` followed by `ravel()`, and matching
///    `pantr.grid.TensorProductGrid` cell ids. Each coordinate is a copy of a
///    node, so that half of the rule is exact by construction.
///  - **The map from `[-1, 1]` onto `[0, 1]` is `(x + 1) * 0.5` and `w * 0.5`**,
///    the same two operations in the same order as
///    `pantr.quad._rules._scale_and_cast_nodes_and_weights`. Both are elementwise
///    and both are pinned by IEEE 754, so the map is common mode and cancels
///    exactly; `design/backend_parity.md` Rule 1 is why that matters and why the
///    bound would otherwise have to be transported.
///
/// ## The messages are the oracle's, verbatim
///
/// A caller that catches on a message must not see `PANTR_BACKEND` change what
/// the library says. So the text below names the *Python* entry points
/// (`tensor_product_quadrature`, `gauss_legendre_quadrature`) even where the C++
/// spelling differs, which is the same decision `AABB` took for `union` /`merge`.

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "pantr/core/error.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/quad/legendre.hpp"

namespace pantr::quad {

namespace detail {

/// Render a length the way Python renders a 1-D shape tuple.
///
/// \param n The length.
/// \return `"(n,)"`.
[[nodiscard]] inline std::string format_shape(std::size_t n) {
    return "(" + std::to_string(n) + ",)";
}

/// Render two extents the way Python renders a 2-D shape tuple.
///
/// \param rows The first extent.
/// \param cols The second extent.
/// \return `"(rows, cols)"`.
[[nodiscard]] inline std::string format_shape(std::size_t rows, std::size_t cols) {
    return "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")";
}

/// Render point counts the way Python's `repr` renders a tuple of ints.
///
/// The trailing comma of a one-element tuple is part of that spelling, and the
/// oracle interpolates the tuple with `!r`, so it has to be reproduced.
///
/// \param counts The per-axis counts.
/// \return `"(2, 3)"`, or `"(2,)"` for a single axis.
[[nodiscard]] inline std::string format_counts(std::span<const int> counts) {
    std::string text = "(";
    for (std::size_t d = 0; d < counts.size(); ++d) {
        if (d > 0) {
            text += ", ";
        }
        text += std::to_string(counts[d]);
    }
    if (counts.size() == 1) {
        text += ",";
    }
    return text + ")";
}

}  // namespace detail

/// One axis's 1-D rule: nodes and weights that must not be transposed.
///
/// A non-owning view of two arrays the caller keeps alive for the duration of
/// the call. See this file's comment for why the pair is named rather than
/// passed as two adjacent same-typed spans.
struct Rule1D {
    std::span<const double> nodes;    ///< The axis's nodes, on `[0, 1]`.
    std::span<const double> weights;  ///< The matching weights, of equal length.
};

/// An immutable quadrature rule on the unit cube `[0, 1]^ndim`.
///
/// Bundles quadrature points and weights as the *reference* rule that
/// `pantr.grid.cell_quadrature` affinely maps onto each cell of a grid. Points
/// lie in the closed unit cube; the rules the two factories build have weights
/// summing to `1`, the measure of the unit cube, so the rule integrates the
/// constant `1` to within the rounding of that sum and not exactly.
///
/// Instances are immutable: there is no operation that changes an existing rule,
/// and both factories return by value.
class QuadratureRule {
  public:
    /// Build a rule from its points and weights, validating them.
    ///
    /// \param points `(num_points, ndim)` table of points, each coordinate in
    ///        `[0, 1]`.
    /// \param weights `(num_points,)` weights. Any finite value is legal,
    ///        negative ones included: several legitimate rules (Newton-Cotes past
    ///        degree 8, moment-fitted rules) carry them, and the oracle checks
    ///        neither sign nor sum.
    /// \throws std::invalid_argument If either extent of `points` is zero, the
    ///         two lengths disagree, any value is not finite, or any coordinate
    ///         lies outside `[0, 1]`.
    QuadratureRule(span2d<const double> points, std::span<const double> weights)
        : points_(), weights_(weights.begin(), weights.end()), num_points_(points.extent(0)),
          ndim_(points.extent(1)) {
        if (num_points_ == 0 || ndim_ == 0) {
            throw std::invalid_argument("points must be non-empty; got shape "
                                        + detail::format_shape(num_points_, ndim_) + ".");
        }
        if (weights_.size() != num_points_) {
            throw std::invalid_argument("weights length " + std::to_string(weights_.size())
                                        + " must match the number of points "
                                        + std::to_string(num_points_) + ".");
        }
        points_.resize(num_points_ * ndim_);
        for (std::size_t i = 0; i < num_points_; ++i) {
            for (std::size_t d = 0; d < ndim_; ++d) {
                points_[i * ndim_ + d] = at(points, i, d);
            }
        }
        // The three loops are separate and in this order because the oracle
        // reports the first failure of each kind in exactly this order, and the
        // message a caller sees is part of the contract.
        for (const double value : points_) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("points must contain only finite values.");
            }
        }
        for (const double value : weights_) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("weights must contain only finite values.");
            }
        }
        for (const double value : points_) {
            if (value < 0.0 || value > 1.0) {
                throw std::invalid_argument("points must lie in the unit cube [0, 1]^ndim.");
            }
        }
    }

    /// Build the tensor product of one 1-D rule per axis.
    ///
    /// Points are enumerated in row-major order -- the last axis varies fastest,
    /// matching `pantr.grid.TensorProductGrid` cell ids -- and each weight is the
    /// product of the corresponding per-axis weights, accumulated from axis 0
    /// upwards.
    ///
    /// \param rules One 1-D rule per axis, at least one.
    /// \return The `rules.size()`-dimensional rule on `[0, 1]^ndim`.
    /// \throws std::invalid_argument If `rules` is empty, if any axis carries an
    ///         empty or mismatched `(nodes, weights)` pair, or -- through the
    ///         constructor -- if any node lies outside `[0, 1]` or is not finite.
    /// \throws CapacityError If the point count, or the point table it implies,
    ///         does not fit `std::size_t`. The oracle instead dies inside numpy's
    ///         allocator, so the two backends diverge in a regime neither can
    ///         serve; this is the limit of *this* build rather than a defect in
    ///         the argument, which is why it is not `std::invalid_argument`.
    [[nodiscard]] static QuadratureRule tensor_product(std::span<const Rule1D> rules) {
        if (rules.empty()) {
            throw std::invalid_argument(
                "tensor_product_quadrature: rules must have at least one axis.");
        }
        const std::size_t ndim = rules.size();
        std::size_t num_points = 1;
        for (std::size_t d = 0; d < ndim; ++d) {
            const std::size_t count = rules[d].nodes.size();
            if (count == 0 || count != rules[d].weights.size()) {
                throw std::invalid_argument(
                    "tensor_product_quadrature: axis " + std::to_string(d)
                    + " needs matching non-empty (nodes, weights); got shapes "
                    + detail::format_shape(count) + " and "
                    + detail::format_shape(rules[d].weights.size()) + ".");
            }
            require_product_fits(num_points, count);
            num_points *= count;
        }
        require_product_fits(num_points, ndim);

        std::vector<double> points(num_points * ndim);
        std::vector<double> weights(num_points);
        std::vector<std::size_t> index(ndim, 0);
        for (std::size_t i = 0; i < num_points; ++i) {
            // Left to right, matching `np.prod(..., axis=0)` over the oracle's
            // (ndim, num_points) block. See this file's comment.
            double weight = rules[0].weights[index[0]];
            points[i * ndim] = rules[0].nodes[index[0]];
            for (std::size_t d = 1; d < ndim; ++d) {
                points[i * ndim + d] = rules[d].nodes[index[d]];
                weight *= rules[d].weights[index[d]];
            }
            weights[i] = weight;
            // Odometer over the axes, last one fastest.
            for (std::size_t d = ndim; d-- > 0;) {
                if (++index[d] < rules[d].nodes.size()) {
                    break;
                }
                index[d] = 0;
            }
        }
        return QuadratureRule(span2d<const double>(points.data(), num_points, ndim), weights);
    }

    /// Build the tensor product of per-axis Gauss-Legendre rules.
    ///
    /// Exact for tensor-product polynomials of per-axis degree `2 * npts[d] - 1`;
    /// the weights sum to `1` up to the rounding of that sum.
    ///
    /// \param npts Points per axis; `npts.size()` is the spatial dimension.
    /// \return The tensor-product Gauss-Legendre rule on `[0, 1]^ndim`.
    /// \throws std::invalid_argument If `npts` is empty or any entry is below 1.
    /// \throws CapacityError As for `tensor_product`.
    [[nodiscard]] static QuadratureRule gauss_legendre(std::span<const int> npts) {
        if (npts.empty()) {
            throw std::invalid_argument("gauss_legendre_quadrature: ndim must be >= 1; got 0.");
        }
        for (const int count : npts) {
            if (count < 1) {
                throw std::invalid_argument(
                    "gauss_legendre_quadrature: every npts entry must be >= 1; got "
                    + detail::format_counts(npts) + ".");
            }
        }
        const std::size_t ndim = npts.size();
        std::vector<std::vector<double>> nodes(ndim);
        std::vector<std::vector<double>> weights(ndim);
        std::vector<Rule1D> rules(ndim);
        for (std::size_t d = 0; d < ndim; ++d) {
            const auto count = static_cast<std::size_t>(npts[d]);
            nodes[d].resize(count);
            weights[d].resize(count);
            gauss_legendre_symmetric(npts[d], nodes[d], weights[d]);
            // The oracle's map onto [0, 1], operation for operation. Elementwise
            // and pinned by IEEE 754, so it is common mode across the backends.
            for (std::size_t i = 0; i < count; ++i) {
                nodes[d][i] = (nodes[d][i] + 1.0) * 0.5;
                weights[d][i] = weights[d][i] * 0.5;
            }
            rules[d] = Rule1D{nodes[d], weights[d]};
        }
        return tensor_product(rules);
    }

    /// The spatial dimension.
    ///
    /// \return The number of axes, `>= 1`.
    [[nodiscard]] std::size_t ndim() const noexcept { return ndim_; }

    /// The number of quadrature points.
    ///
    /// \return The point count, `>= 1`.
    [[nodiscard]] std::size_t num_points() const noexcept { return num_points_; }

    /// The quadrature points.
    ///
    /// \return A `(num_points, ndim)` view of the stored points, valid while the
    ///         rule lives.
    [[nodiscard]] span2d<const double> points() const noexcept {
        return span2d<const double>(points_.data(), num_points_, ndim_);
    }

    /// The quadrature weights.
    ///
    /// \return A view of the stored weights, valid while the rule lives.
    [[nodiscard]] std::span<const double> weights() const noexcept { return weights_; }

    /// A compact representation, matching the oracle's `__repr__`.
    ///
    /// Only two integers, so there is none of the float-formatting difficulty
    /// `AABB` had to solve. The Python wrapper formats its own `__repr__`
    /// regardless, so this is what a C++ caller sees.
    ///
    /// \return `"QuadratureRule(ndim=..., num_points=...)"`.
    [[nodiscard]] std::string to_string() const {
        return "QuadratureRule(ndim=" + std::to_string(ndim_)
               + ", num_points=" + std::to_string(num_points_) + ")";
    }

  private:
    /// Reject a product of sizes that would wrap around.
    ///
    /// \param value The running product.
    /// \param factor What it is about to be multiplied by; must be non-zero.
    /// \throws CapacityError If the product would exceed `SIZE_MAX`.
    static void require_product_fits(std::size_t value, std::size_t factor) {
        if (value > std::numeric_limits<std::size_t>::max() / factor) {
            throw CapacityError("tensor_product_quadrature: the point count does not fit "
                                "this platform's size type.");
        }
    }

    std::vector<double> points_;   ///< Points, row-major, `num_points * ndim` long.
    std::vector<double> weights_;  ///< Weights, `num_points` long.
    std::size_t num_points_;       ///< Number of points, `>= 1`.
    std::size_t ndim_;             ///< Number of axes, `>= 1`.
};

}  // namespace pantr::quad
