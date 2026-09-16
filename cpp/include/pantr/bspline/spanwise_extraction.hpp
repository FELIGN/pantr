#pragma once

/// \file
/// The tensor-product spanwise element extraction: the per-direction 1D extraction
/// operators of a `BsplineSpace`, stored compactly, and the quantities that storage
/// fixes.
///
/// ## What this type owns, and what it does not
///
/// It owns the *value* -- the space it extracts from, the target basis, the Lagrange
/// point distribution, and the three per-direction array bundles the oracle keeps in
/// `_compact_ops_1d`, `_idx_maps_1d` and `_is_identity_mask_1d` -- plus the two
/// quantities `design/bspline_derived_caches.md` assigns here: the dense `ops_1d`
/// block, lazily decompressed, and the fully-identity element count, eager.
///
/// It owns no *operations*. Applying an operator to a vector, materialising a
/// Kronecker product, normalising a cell index and iterating the grid are
/// computations *over* an extraction rather than properties *of* one, so they stay
/// on the Python wrapper over `pantr/bspline/extraction_kernels.hpp` -- which is
/// already a separate port with its own parity claim. That is the same line
/// `space_nd.hpp` draws, and `design/extraction_port.md`'s S4 is exactly this
/// boundary: "the class-H `space` accessor, the two memos ..., the wrapper,
/// `__reduce__`".
///
/// ## The operators arrive built, and that is the port's shape rather than a gap
///
/// This type does **not** build the per-direction operators. They arrive as
/// constructor arguments, already dense and already accompanied by their identity
/// masks, and what this type does with them is the *compaction*: keep the
/// non-identity rows, index the rest.
///
/// Three things decide that, and none of them is convenience.
///
///  - **The builders are their own port, and it has landed.**
///    `pantr::bspline::bezier_extraction_1d` and `lagrange_extraction_1d` in
///    `extraction.hpp` are `design/extraction_port.md`'s slice S3, dispatched from
///    `pantr.bspline._extraction_backend` and carrying their own parity claims.
///    Calling them from in here would put the dispatch in two places.
///  - **The cardinal target has no C++ builder** and is not scheduled to get one in
///    this slice: it needs the cardinal-interval scan, which `space_1d.hpp`
///    deliberately keeps off the 1D type. A constructor that built its own operators
///    could therefore not serve `ExtractionTarget::cardinal` at all, and the oracle
///    serves it today.
///  - **The Lagrange builder takes its change-of-basis matrix as an argument**, for
///    the reasons `_extraction_backend.py` states -- one implementation of the
///    tabulation, one cache in front of it, and `pantr.basis.LagrangeVariant` (a
///    `StrEnum`, which numba accepts and then silently mis-compares) kept off the
///    seam. A constructor that built its own operators would have to take that
///    matrix too, so the argument list grows rather than shrinks.
///
/// So the seam is the same one `BsplineSpace` uses: the directions arrive built, and
/// this type owns what the collection determines.
///
/// ## The target and the variant cross as a plain integer and a plain string
///
/// `ExtractionTarget` below mirrors `pantr.bspline.ExtractionTarget`, a Python
/// `IntEnum`, member for member and value for value. It is a C++ `enum class` with an
/// explicit `std::int64_t` base so the two agree by construction, and the binding
/// takes and returns the integer rather than a `nb::enum_` -- there is no bound enum
/// anywhere in this extension and this is not the place to introduce the first.
///
/// The Lagrange variant is carried as its **own string**, which is what
/// `pantr.basis.LagrangeVariant` *is*: a `StrEnum` whose member value is that string,
/// so `LagrangeVariant(stored)` round-trips exactly. Nothing here interprets it, and
/// that is the point -- `design/extraction_port.md` F5 records that the one thing
/// which must never happen to this tag is a kernel branching on it, and a value this
/// type stores and hands back unread cannot. A C++ enumeration duplicating the five
/// members would be a second definition of a set Python owns, to be kept in step by
/// hand, for a value no C++ code reads.
///
/// ## Validating rather than asserting
///
/// This is the C++ counterpart of Layer 2, so it validates and throws in a release
/// build as much as in a debug one. `pantr/core/error.hpp` sets the split: value and
/// range checks live here, type-kind checks stay in the Python wrapper.
///
/// Three refusals the oracle makes are **not** here, and each for the same reason as
/// its counterpart in `space_nd.hpp`:
///
///  - the mixed-dtype `ValueError` over the per-direction operators, because
///    `SpanwiseElementExtraction<T>` can hold only `T` and a mixed collection is not
///    representable;
///  - the `NotImplementedError` for a periodic direction, because
///    `pantr/core/error.hpp` records that nanobind has no path from a
///    `std::exception` to that Python class;
///  - the `ValueError` naming an unrecognised target spelling, because the legacy
///    string spelling is a Python compatibility boundary and the enum is what crosses.
///
/// All three stay in `pantr.bspline.spanwise_element_extraction`, which is where a
/// type-kind check belongs.
///
/// ## Thread safety
///
/// Every accessor is safe to call concurrently on the same object with no external
/// locking. The one memo -- the dense `ops_1d` block -- is behind
/// `pantr/core/lazy.hpp`'s `LazySlot`, filled at most once and published atomically;
/// everything else is frozen at construction. In a sweep, hoist `ops_1d(d)` into a
/// local before the loop rather than calling it per element: the span is free, the
/// double-checked load is not.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pantr/bspline/space_nd.hpp"
#include "pantr/core/lazy.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::bspline {

/// The element-local basis a spanwise extraction maps the B-spline basis onto.
///
/// Mirrors `pantr.bspline.ExtractionTarget`, member for member and value for value.
/// The base type is fixed so the two cannot drift: the Python side is an `IntEnum`
/// and the integer is what crosses the binding.
enum class ExtractionTarget : std::int64_t {
    bezier = 0,    ///< Bernstein (Bézier) basis on each element.
    lagrange = 1,  ///< Lagrange basis on each element, at the chosen nodes.
    cardinal = 2,  ///< Cardinal B-spline basis on each element.
};

/// Whether `value` names a member of `ExtractionTarget`.
///
/// The binding's range check, spelled once. A cast of an out-of-range integer to a
/// scoped enumeration with a fixed underlying type is well defined but produces a
/// value no branch here handles, so it is refused rather than admitted.
///
/// \param value The integer a caller supplied.
/// \return `true` if `value` is one of the three members.
[[nodiscard]] inline constexpr bool is_extraction_target(std::int64_t value) noexcept {
    return value >= static_cast<std::int64_t>(ExtractionTarget::bezier)
           && value <= static_cast<std::int64_t>(ExtractionTarget::cardinal);
}

/// A tensor-product change-of-basis operator across a B-spline space's elements.
///
/// Holds the per-direction 1D extraction operators in *compact* form: only the rows
/// that are not the identity are stored, which is what makes an identity-heavy space
/// -- a cardinal space on a uniform mesh, a spline already in Bézier form -- cost
/// almost nothing. The dense layout is reconstructed on demand and memoised.
///
/// Instances are immutable: no operation changes one, and the memo is a function of
/// state frozen at construction.
///
/// **Move-only.** The three per-direction bundles include an identity mask held as
/// `std::unique_ptr<bool[]>`, because `std::vector<bool>` is a bitset with no `bool`
/// storage to hand out and the mask goes to Python as a `numpy.bool_` array viewing
/// this object's own memory. Nothing copies an extraction: the binding constructs in
/// place, the wrapper holds one handle, and a derived extraction is a fresh object.
///
/// \tparam T The scalar type the operators are stored in.
template <Real T>
class SpanwiseElementExtraction {
  public:
    /// The scalar type the operators are stored in.
    using scalar_type = T;

    /// One direction's dense operators and identity mask, as the builders produce them.
    ///
    /// The constructor's argument type. Neither span is retained: the operators are
    /// compacted into this object's own storage and the mask is copied.
    struct DirectionInput {
        /// The dense operators, shape `(n_elements, n_out, n_in)`.
        span_nd<const T, 3> operators;
        /// The per-element identity flags, length `n_elements`.
        std::span<const bool> is_identity;
    };

    /// Compact the given per-direction operators and hold them.
    ///
    /// \param space The space being extracted from. Shared, not copied, so that the
    ///        wrapper's `extraction.space is space` contract survives; see
    ///        `design/bspline_ownership_lifetime.md`'s class **H**.
    /// \param target The element-local basis the operators map onto. Stored and handed
    ///        back; nothing here branches on it.
    /// \param lagrange_variant The `pantr.basis.LagrangeVariant` member's own string,
    ///        meaningful only for `ExtractionTarget::lagrange`. Stored and handed back
    ///        unread; see the file comment.
    /// \param directions One entry per direction of `space`, in axis order.
    /// \throws std::invalid_argument If `space` is null, if `directions` does not have
    ///         one entry per direction, if a direction's operator count or mask length
    ///         does not match that direction's element count, or if a direction's
    ///         operators have no rows or columns.
    SpanwiseElementExtraction(std::shared_ptr<const BsplineSpace<T>> space,
                              ExtractionTarget target, std::string lagrange_variant,
                              std::span<const DirectionInput> directions)
        : space_(std::move(space)),
          target_(target),
          lagrange_variant_(std::move(lagrange_variant)) {
        if (space_ == nullptr) {
            throw std::invalid_argument("the B-spline space is null");
        }
        const std::span<const std::int64_t> num_intervals = space_->num_intervals();
        if (directions.size() != num_intervals.size()) {
            throw std::invalid_argument(
                "expected one operator bundle per direction; got "
                + std::to_string(directions.size()) + " for a space of dimension "
                + std::to_string(num_intervals.size()));
        }
        directions_.reserve(directions.size());
        for (std::size_t d = 0; d < directions.size(); ++d) {
            directions_.push_back(compact(directions[d], num_intervals[d], d));
        }
        fill_derived();
    }

    /// Share the space this extraction was built over.
    ///
    /// The returned handle keeps its value alive independently of this extraction, so
    /// a caller may outlive the owner.
    ///
    /// \return A handle on the space.
    [[nodiscard]] std::shared_ptr<const BsplineSpace<T>> space() const { return space_; }

    /// Borrow the space this extraction was built over.
    ///
    /// Valid while `*this` is, and **not bound**: an inner loop must not pay an atomic
    /// pair per access. See the `_ref` rule in
    /// `design/bspline_ownership_lifetime.md`.
    ///
    /// \return A reference to the space.
    [[nodiscard]] const BsplineSpace<T>& space_ref() const noexcept { return *space_; }

    /// The element-local basis this extraction maps onto.
    ///
    /// \return The target, as supplied at construction.
    [[nodiscard]] ExtractionTarget target() const noexcept { return target_; }

    /// The Lagrange point distribution's own name.
    ///
    /// Meaningful only for `ExtractionTarget::lagrange`, and never interpreted here.
    ///
    /// \return The string supplied at construction.
    [[nodiscard]] const std::string& lagrange_variant() const noexcept {
        return lagrange_variant_;
    }

    /// The number of tensor-product directions.
    ///
    /// \return The dimension of the space.
    [[nodiscard]] std::int64_t dim() const noexcept { return space_->dim(); }

    /// The per-direction number of elements.
    ///
    /// \return A view of `dim()` counts, in axis order.
    [[nodiscard]] std::span<const std::int64_t> num_intervals() const noexcept {
        return space_->num_intervals();
    }

    /// The total number of elements on the tensor-product grid.
    ///
    /// \return The product of `num_intervals()`.
    [[nodiscard]] std::int64_t num_total_intervals() const noexcept {
        return space_->num_total_intervals();
    }

    /// Direction `d`'s compact operators: the non-identity rows only.
    ///
    /// \param d The direction, in `[0, dim())`.
    /// \return A view of shape `(n_compact, n_out, n_in)`, with `n_compact` at least
    ///         one -- a direction whose every element is the identity carries one row
    ///         of zeros, so that an index computed by a numba kernel is always in
    ///         range. That sentinel is the oracle's, and it is never read.
    /// \throws std::out_of_range If `d` is not a direction of this extraction.
    [[nodiscard]] span_nd<const T, 3> compact_ops(std::int64_t d) const {
        const Direction& dir = at_direction(d);
        return span_nd<const T, 3>(dir.compact.data(), static_cast<std::size_t>(dir.n_compact),
                                   static_cast<std::size_t>(dir.n_out),
                                   static_cast<std::size_t>(dir.n_in));
    }

    /// Direction `d`'s compact index map.
    ///
    /// \param d The direction, in `[0, dim())`.
    /// \return A view of `n_elements` row indices into `compact_ops(d)`. The entry of
    ///         an identity element is zero and is unused: a caller short-circuits on
    ///         `is_identity_mask(d)` first.
    /// \throws std::out_of_range If `d` is not a direction of this extraction.
    [[nodiscard]] std::span<const std::int64_t> idx_map(std::int64_t d) const {
        return std::span<const std::int64_t>(at_direction(d).idx_map);
    }

    /// Direction `d`'s per-element identity flags.
    ///
    /// \param d The direction, in `[0, dim())`.
    /// \return A view of `n_elements` flags; entry `i` is `true` iff element `i`'s
    ///         operator in this direction is the identity.
    /// \throws std::out_of_range If `d` is not a direction of this extraction.
    [[nodiscard]] std::span<const bool> is_identity_mask(std::int64_t d) const {
        const Direction& dir = at_direction(d);
        return std::span<const bool>(dir.mask.get(), static_cast<std::size_t>(dir.n_elements));
    }

    /// Direction `d`'s dense operators, one per element.
    ///
    /// Decompressed from the compact storage on first use and memoised; the view is of
    /// the memo, never of a copy. Identity elements read as the `(n_out, n_in)`
    /// rectangular identity.
    ///
    /// The whole block is built on the first call for **any** direction, because the
    /// oracle's `ops_1d` is one `cached_property` over every direction and a per
    /// direction memo would be a second shape for the same quantity.
    ///
    /// \param d The direction, in `[0, dim())`.
    /// \return A view of shape `(n_elements, n_out, n_in)`, valid while this object is.
    /// \throws std::out_of_range If `d` is not a direction of this extraction.
    [[nodiscard]] span_nd<const T, 3> ops_1d(std::int64_t d) const {
        const Direction& dir = at_direction(d);
        const std::vector<std::vector<T>>& dense = dense_.get([this] { return build_dense(); });
        return span_nd<const T, 3>(dense[static_cast<std::size_t>(d)].data(),
                                   static_cast<std::size_t>(dir.n_elements),
                                   static_cast<std::size_t>(dir.n_out),
                                   static_cast<std::size_t>(dir.n_in));
    }

    /// The per-direction input size of each element's operator.
    ///
    /// \return A view of `dim()` sizes `(n_in_0, ..., n_in_{dim-1})`.
    [[nodiscard]] std::span<const std::int64_t> input_shape_per_dir() const noexcept {
        return std::span<const std::int64_t>(input_shape_);
    }

    /// The per-direction output size of each element's operator.
    ///
    /// \return A view of `dim()` sizes `(n_out_0, ..., n_out_{dim-1})`.
    [[nodiscard]] std::span<const std::int64_t> output_shape_per_dir() const noexcept {
        return std::span<const std::int64_t>(output_shape_);
    }

    /// The number of elements whose operator is the identity in every direction.
    ///
    /// An eager field: the product of the per-direction identity counts, which is one
    /// pass over the masks at construction. `design/bspline_derived_caches.md` assigns
    /// it that way -- the oracle memoises it only because a Python attribute read is
    /// dearer than the count.
    ///
    /// \return The count; `1` for a dimensionless space, matching the empty product.
    [[nodiscard]] std::int64_t num_identity_elements() const noexcept {
        return num_identity_elements_;
    }

    /// Whether every element on the grid has an identity operator.
    ///
    /// \return `true` iff every per-direction mask is all-`true`.
    [[nodiscard]] bool is_identity() const noexcept { return is_identity_; }

  private:
    /// One direction's compacted storage.
    struct Direction {
        std::vector<T> compact;             ///< Row-major `(n_compact, n_out, n_in)`.
        std::vector<std::int64_t> idx_map;  ///< `n_elements` rows into `compact`.
        std::unique_ptr<bool[]> mask;       ///< `n_elements` identity flags.
        std::int64_t n_elements = 0;        ///< Elements in this direction.
        std::int64_t n_compact = 0;         ///< Rows of `compact`; at least one.
        std::int64_t n_out = 0;             ///< Operator rows.
        std::int64_t n_in = 0;              ///< Operator columns.
        std::int64_t n_identity = 0;        ///< `true` entries of `mask`.
    };

    /// Compact one direction's operators, keeping only the non-identity rows.
    ///
    /// Reproduces the oracle's construction exactly, sentinel included: a direction
    /// with no non-identity element still gets one row, of zeros, so that
    /// `compact[idx_map[e]]` is in range for every `e` even where no caller reads it.
    ///
    /// \param input The direction's dense operators and identity mask.
    /// \param n_elements The direction's element count, from the space.
    /// \param d The direction, for the messages.
    /// \return The compacted storage.
    /// \throws std::invalid_argument If the shapes disagree with `n_elements`, or the
    ///         operators have no rows or no columns.
    [[nodiscard]] static Direction compact(const DirectionInput& input, std::int64_t n_elements,
                                           std::size_t d) {
        const auto expected = static_cast<std::size_t>(n_elements);
        if (input.operators.extent(0) != expected) {
            throw std::invalid_argument(
                "direction " + std::to_string(d) + " has "
                + std::to_string(input.operators.extent(0)) + " operators for "
                + std::to_string(n_elements) + " elements");
        }
        if (input.is_identity.size() != expected) {
            throw std::invalid_argument("direction " + std::to_string(d) + " has an identity mask "
                                        "of length " + std::to_string(input.is_identity.size())
                                        + " for " + std::to_string(n_elements) + " elements");
        }
        // The oracle reads `n_out` and `n_in` off the operator array and would raise an
        // index error rather than a diagnosis on a degenerate one. Refusing here is
        // cheaper than the sentinel row of a zero-column operator, which is a shape no
        // builder produces and no kernel could use.
        if (input.operators.extent(1) == 0 || input.operators.extent(2) == 0) {
            throw std::invalid_argument("direction " + std::to_string(d)
                                        + " has operators with no rows or no columns");
        }

        Direction dir;
        dir.n_elements = n_elements;
        dir.n_out = static_cast<std::int64_t>(input.operators.extent(1));
        dir.n_in = static_cast<std::int64_t>(input.operators.extent(2));

        dir.mask = std::make_unique<bool[]>(expected);
        dir.idx_map.assign(expected, 0);
        std::int64_t n_non_identity = 0;
        for (std::size_t e = 0; e < expected; ++e) {
            dir.mask[e] = input.is_identity[e];
            if (!dir.mask[e]) {
                dir.idx_map[e] = n_non_identity;
                ++n_non_identity;
            }
        }
        dir.n_identity = n_elements - n_non_identity;

        const auto rows = static_cast<std::size_t>(dir.n_out);
        const auto cols = static_cast<std::size_t>(dir.n_in);
        dir.n_compact = n_non_identity > 0 ? n_non_identity : 1;
        dir.compact.assign(static_cast<std::size_t>(dir.n_compact) * rows * cols, T{0});
        for (std::size_t e = 0; e < expected; ++e) {
            if (dir.mask[e]) {
                continue;
            }
            const auto row = static_cast<std::size_t>(dir.idx_map[e]);
            T* dst = dir.compact.data() + ((row * rows) * cols);
            for (std::size_t i = 0; i < rows; ++i) {
                for (std::size_t j = 0; j < cols; ++j) {
                    dst[(i * cols) + j] = pantr::at(input.operators, e, i, j);
                }
            }
        }
        return dir;
    }

    /// Set the eager derived fields from the compacted directions.
    void fill_derived() {
        input_shape_.reserve(directions_.size());
        output_shape_.reserve(directions_.size());
        for (const Direction& dir : directions_) {
            input_shape_.push_back(dir.n_in);
            output_shape_.push_back(dir.n_out);
            // A product over `dim() <= 3` per-direction counts, each at most the
            // direction's element count, so it is bounded by `num_total_intervals()`,
            // which `BsplineSpace` has already refused to let overflow.
            num_identity_elements_ *= dir.n_identity;
            is_identity_ = is_identity_ && dir.n_identity == dir.n_elements;
        }
    }

    /// Decompress every direction's operators into the dense layout.
    ///
    /// \return One row-major `(n_elements, n_out, n_in)` block per direction.
    [[nodiscard]] std::vector<std::vector<T>> build_dense() const {
        std::vector<std::vector<T>> dense;
        dense.reserve(directions_.size());
        for (const Direction& dir : directions_) {
            const auto rows = static_cast<std::size_t>(dir.n_out);
            const auto cols = static_cast<std::size_t>(dir.n_in);
            const auto n_elements = static_cast<std::size_t>(dir.n_elements);
            std::vector<T> block(n_elements * rows * cols, T{0});
            for (std::size_t e = 0; e < n_elements; ++e) {
                T* dst = block.data() + ((e * rows) * cols);
                if (dir.mask[e]) {
                    // The rectangular identity, which is what `numpy.eye(n_out, n_in)`
                    // is: ones on the main diagonal, zeros elsewhere, whatever the two
                    // sizes are. The block is already zeroed.
                    for (std::size_t i = 0; i < rows && i < cols; ++i) {
                        dst[(i * cols) + i] = T{1};
                    }
                } else {
                    const T* src =
                        dir.compact.data() + ((static_cast<std::size_t>(dir.idx_map[e]) * rows) * cols);
                    for (std::size_t k = 0; k < rows * cols; ++k) {
                        dst[k] = src[k];
                    }
                }
            }
            dense.push_back(std::move(block));
        }
        return dense;
    }

    /// Direction `d`'s storage, refusing an out-of-range direction.
    ///
    /// \param d The direction.
    /// \return Its storage.
    /// \throws std::out_of_range If `d` is not a direction of this extraction.
    [[nodiscard]] const Direction& at_direction(std::int64_t d) const {
        if (d < 0 || d >= dim()) {
            throw std::out_of_range("direction must lie in [0, " + std::to_string(dim())
                                    + "); got " + std::to_string(d));
        }
        return directions_[static_cast<std::size_t>(d)];
    }

    std::shared_ptr<const BsplineSpace<T>> space_;  ///< The space, shared.
    ExtractionTarget target_;                       ///< The element-local basis.
    std::string lagrange_variant_;                  ///< The node distribution's own name.
    std::vector<Direction> directions_;             ///< One per direction, in axis order.
    std::vector<std::int64_t> input_shape_;         ///< One `n_in` per direction.
    std::vector<std::int64_t> output_shape_;        ///< One `n_out` per direction.
    std::int64_t num_identity_elements_ = 1;        ///< Product of the identity counts.
    bool is_identity_ = true;                       ///< Whether every mask is all-`true`.
    LazySlot<std::vector<std::vector<T>>> dense_;   ///< The dense `ops_1d` block.
};

}  // namespace pantr::bspline
