#pragma once

/// \file
/// The tensor-product B-spline space: one `BsplineSpace1D` per direction, and the
/// quantities the collection fixes.
///
/// ## What this type owns, and what it does not
///
/// It owns the *value* -- the univariate spaces, in axis order -- plus every
/// quantity that is a reduction over them: the per-direction degrees, basis counts
/// and interval counts, their two products, the parametric tolerance, the
/// per-direction domain, and whether every direction is a single Bézier segment.
///
/// It owns no *operations*, exactly as `pantr/bspline/space_1d.hpp` owns none.
/// Basis tabulation, the per-cell control-point support, the boundary slab and the
/// windowed restriction are computations *over* a space rather than properties
/// *of* one, so they are separate ports over free functions taking a
/// `const BsplineSpace&`. That is the same line `space_1d.hpp` draws to keep
/// `get_cardinal_intervals` out of the 1D type.
///
/// ## The directions are shared, not copied, and the identity contract is why
///
/// `spaces_` is a `std::vector<std::shared_ptr<const BsplineSpace1D<T>>>`, per
/// `design/bspline_ownership_lifetime.md`'s class **H**: an accessor that hands out
/// a subobject the owner keeps returns a copy of the handle, so the *value* is
/// shared and the owner's death is irrelevant. `rv_policy::reference_internal`
/// would put that guarantee in the binding, and the binding is scheduled for
/// deletion; a `shared_ptr<const T>` puts it in the type, where a C++ consumer with
/// no interpreter present gets it too.
///
/// Sharing is not a performance choice here. `tests/test_bspline_space.py:89`
/// asserts `space.spaces[0] is space_1d`, so the Python wrapper hands back the
/// object it was built from; a C++ space that *copied* its directions would present
/// two Python objects agreeing on identity over two different C++ objects. That is
/// F6 of the ownership note, and it is what fixes the constructor's signature.
///
/// Sharing is safe because a `BsplineSpace1D` is immutable after construction. A
/// `shared_ptr<const T>` over an immutable `T` is a value with a cheap copy.
///
/// The borrowing twin, `space_ref`, is **not bound**: copying the handle costs an
/// uncontended atomic increment/decrement pair per access, which an inner loop must
/// not pay for a value it does not keep. `design/bspline_ownership_lifetime.md`
/// carries the measurement and the machine it was taken on.
/// `tests/parity/test_bspline_binding_contract.py` asserts that no bound method name
/// ends in `_ref`.
///
/// ## No memo, and that is a decision rather than an omission
///
/// `design/bspline_derived_caches.md` F1 counts the oracle's seven
/// `functools.cached_property` sites on this class and concludes that **none of them
/// becomes a memo**: every one is an O(dim) reduction over the directions, with
/// `dim` at most 3 everywhere in the tree, and they are memoised in Python only
/// because an attribute read that walks three objects costs more than a `__dict__`
/// hit. So there is no `LazySlot` here. Six are eager fields, set once in the
/// constructor; `has_bezier_like_knots` is a three-iteration accessor, because the
/// oracle's counterpart is a method and computes on call.
///
/// ## Validating rather than asserting
///
/// This is the C++ counterpart of Layer 2, so it validates and throws in a release
/// build as much as in a debug one. `pantr/core/error.hpp` sets the split: value and
/// range checks live here, type-kind checks stay in the Python wrapper.
///
/// **One check that does not live here, and cannot.** The oracle refuses directions
/// of differing dtype with *"All B-spline spaces must have the same data type."*
/// `BsplineSpace<T>` can only hold `BsplineSpace1D<T>`, so a mixed collection is not
/// representable and there is nothing to check: the refusal is the Python wrapper's,
/// which is where a dtype -- a type-kind fact -- belongs anyway.
///
/// ## Parity notes for the Python oracle
///
/// `pantr.bspline.BsplineSpace` is the oracle. Three things this type reproduces, and
/// **one** place it deliberately differs:
///
///  - **Every reduction is over the directions in axis order**, and the order is
///    load-bearing rather than incidental: `degrees`, `num_basis`, `num_intervals`
///    and `domain` are per-direction sequences, and nothing about their *values*
///    would reveal a transposition on a space whose directions happen to agree.
///    `tests/parity/test_bspline_space_nd.py` therefore keeps its case table
///    asymmetric in each of them, and asserts of the table that it does.
///  - **A dimensionless space is constructible**, with `dim() == 0`. The oracle
///    admits `BsplineSpace([])` -- pinned by
///    `tests/test_bspline_space.py::test_empty_spaces_list` -- so this does too. The
///    empty products are 1, which is the empty tensor product's own convention and
///    what the oracle's own `math.prod(())` returns.
///  - **`tolerance()` refuses a dimensionless space with the oracle's own message**,
///    character for character, as every other refusal in this port does. That is
///    worth a sentence because the oracle's message was *made* deliberate for it:
///    left to `max()` over an empty sequence it would have been CPython's own text,
///    which moved between 3.11 and 3.12, so reproducing it here would have broken
///    parity on one leg of the test matrix and held on the others.
///    `src/pantr/bspline/_bspline_space_nd.py` states the same string and says why,
///    and `tests/parity/test_bspline_space_nd.py` compares the two texts rather than
///    just the exception type.
///  - **The one difference: the two products refuse an overflow instead of
///    wrapping.** `numpy.prod` over an `int64` tuple wraps silently; signed overflow
///    in C++ is undefined, so the UBSan leg would abort rather than wrap. Neither is
///    a good answer, and the input is reachable: three directions of 2.1e6 basis
///    functions each need about 50 MB of knots and overflow `int64`. So the products
///    throw `pantr::CapacityError`, which `pantr/core/error.hpp` reserves for a limit
///    of the implementation rather than a defect in the argument. The oracle uses
///    `math.prod` rather than `numpy.prod` and so returns the exact integer instead,
///    which is where the two part company -- above the `int64` range, and nowhere
///    below it.
///
/// ## Thread safety
///
/// Every accessor below is safe to call concurrently on the same object with no
/// external locking, and here that needs no mechanism at all: there is no memo, so
/// every accessor reads state frozen at construction. In a sweep, hoist
/// `num_basis()` or `degrees()` into a local span before the loop rather than
/// calling it per element -- the span is free, but the call is not.

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pantr/bspline/space_1d.hpp"
#include "pantr/core/error.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::bspline {

namespace detail {

/// The product of a sequence of non-negative counts, refusing an overflow.
///
/// A tensor-product total is a product of per-direction counts. The oracle forms it
/// with `math.prod`, whose Python integers are arbitrary-precision, so it returns the
/// exact value and never wraps. Wrapping is not available here either -- signed
/// overflow is undefined behaviour, and the `gcc-debug` preset carries
/// `-fno-sanitize-recover=undefined` -- so this refuses instead, which is the one
/// deliberate divergence from the oracle. The input is reachable rather than
/// hypothetical: see the parity note in the file comment.
///
/// The empty product is 1, matching `math.prod(())`.
///
/// **Non-negative counts are a trusted precondition, not a checked one**, which is
/// the one place this header departs from its own "validate rather than assert"
/// rule -- and it departs deliberately, because the precondition is not a caller's
/// to violate. Both call sites pass a `BsplineSpace1D`'s `num_basis` or
/// `num_intervals`, and that type's constructor refuses a space with neither, so a
/// negative count is unreachable rather than merely unlikely. A check here would
/// need a message describing a state the type cannot be in.
///
/// \param counts The per-direction counts; each must be non-negative, which every
///        call site guarantees.
/// \param what The quantity being formed, for the message.
/// \return The product.
/// \throws pantr::CapacityError If the product exceeds `std::int64_t`.
[[nodiscard]] inline std::int64_t checked_product(std::span<const std::int64_t> counts,
                                                  const char* what) {
    constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
    std::int64_t total = 1;
    for (const std::int64_t count : counts) {
        if (count == 0) {
            return 0;
        }
        // Divided rather than multiplied, so the check itself cannot overflow.
        // Both operands are positive here: a count is non-negative by the
        // constructor's own invariants and the zero case returned above.
        if (total > kMax / count) {
            throw pantr::CapacityError(std::string(what)
                                       + " exceeds the range of a 64-bit integer");
        }
        total *= count;
    }
    return total;
}

}  // namespace detail

/// A tensor-product B-spline space: one univariate space per direction.
///
/// Instances are immutable: no operation changes an existing space, and every
/// derived one is returned by value. Unlike `BsplineSpace1D` there is not even a
/// derived block to fill lazily, so construct-then-freeze holds here in its purest
/// form, with no exception to reason about.
///
/// A dimensionless space -- no directions at all -- is legal; see the file comment.
///
/// \tparam T The scalar type the directions store their knots in.
template <Real T>
class BsplineSpace {
  public:
    /// The scalar type the directions store their knots in.
    using scalar_type = T;

    /// Share the given univariate spaces.
    ///
    /// This is the constructor the binding calls, and sharing is what preserves the
    /// wrapper's identity contract; see the file comment.
    ///
    /// \param spaces One handle per direction, in axis order. None may be null.
    /// \throws std::invalid_argument If any handle is null.
    /// \throws pantr::CapacityError If a tensor-product total overflows.
    explicit BsplineSpace(std::vector<std::shared_ptr<const BsplineSpace1D<T>>> spaces)
        : spaces_(std::move(spaces)) {
        for (std::size_t d = 0; d < spaces_.size(); ++d) {
            if (spaces_[d] == nullptr) {
                throw std::invalid_argument("direction " + std::to_string(d)
                                            + " is a null B-spline space");
            }
        }
        fill_derived();
    }

    /// Copy the given univariate spaces, for a caller holding values rather than
    /// handles.
    ///
    /// The copies are what this space shares afterwards, so a later write through
    /// the caller's own objects -- there is none to make, they are immutable -- could
    /// not reach it either way. Provided because a C++ caller with no reason to
    /// allocate should not have to.
    ///
    /// \param spaces One space per direction, in axis order.
    /// \throws pantr::CapacityError If a tensor-product total overflows.
    explicit BsplineSpace(std::span<const BsplineSpace1D<T>> spaces) {
        spaces_.reserve(spaces.size());
        for (const BsplineSpace1D<T>& space : spaces) {
            spaces_.push_back(std::make_shared<const BsplineSpace1D<T>>(space));
        }
        fill_derived();
    }

    /// The number of directions.
    ///
    /// \return The dimension of the parametric domain; zero for a dimensionless
    ///         space.
    [[nodiscard]] std::int64_t dim() const noexcept {
        return static_cast<std::int64_t>(spaces_.size());
    }

    /// Share direction `d`'s univariate space.
    ///
    /// The returned handle keeps its value alive independently of this space, so a
    /// caller may outlive the owner.
    ///
    /// \param d The direction, in `[0, dim())`.
    /// \return A handle on that direction's space.
    /// \throws std::out_of_range If `d` is not a direction of this space.
    [[nodiscard]] std::shared_ptr<const BsplineSpace1D<T>> space(std::int64_t d) const {
        check_direction(d);
        return spaces_[static_cast<std::size_t>(d)];
    }

    /// Borrow direction `d`'s univariate space.
    ///
    /// Valid while `*this` is, and **not bound**: an inner loop must not pay an
    /// atomic pair per access. See the `_ref` rule in the file comment.
    ///
    /// \param d The direction, in `[0, dim())`.
    /// \return A reference to that direction's space.
    /// \throws std::out_of_range If `d` is not a direction of this space.
    [[nodiscard]] const BsplineSpace1D<T>& space_ref(std::int64_t d) const {
        check_direction(d);
        return *spaces_[static_cast<std::size_t>(d)];
    }

    /// Share every direction, in axis order.
    ///
    /// \return A view of this space's own handles, valid while it lives. Copying a
    ///         handle out of it is what extends a direction's lifetime.
    [[nodiscard]] std::span<const std::shared_ptr<const BsplineSpace1D<T>>>
    spaces() const noexcept {
        return std::span<const std::shared_ptr<const BsplineSpace1D<T>>>(spaces_);
    }

    /// The polynomial degree of each direction.
    ///
    /// \return A view of `dim()` non-negative degrees, in axis order.
    [[nodiscard]] std::span<const std::int64_t> degrees() const noexcept {
        return std::span<const std::int64_t>(degrees_);
    }

    /// The absolute tolerance for parametric comparisons on this space.
    ///
    /// The largest of the directions' tolerances, each of which already carries its
    /// own direction's knot magnitude. Taking the largest is the conservative choice
    /// when the directions are scaled differently: it is the only one of the three
    /// obvious reductions that never under-states a gap in any direction.
    ///
    /// A *selection*, not an arithmetic combination, which is why the two backends
    /// agree on it bit for bit: the result is one of the directions' own tolerances,
    /// unmodified.
    ///
    /// Being an absolute *parametric* length, it is not the factor to scale a
    /// physical coordinate or a spline coefficient by.
    ///
    /// \return The tolerance, a `double` at every knot width. See
    ///         `pantr/bspline/knots.hpp` for why.
    /// \throws std::invalid_argument If the space has no directions. The oracle
    ///         refuses this too; see the file comment for why the message differs.
    [[nodiscard]] double tolerance() const {
        if (spaces_.empty()) {
            throw std::invalid_argument(
                "tolerance: a B-spline space with no directions has no tolerance");
        }
        return tol_;
    }

    /// The number of basis functions in each direction.
    ///
    /// \return A view of `dim()` counts, in axis order.
    [[nodiscard]] std::span<const std::int64_t> num_basis() const noexcept {
        return std::span<const std::int64_t>(num_basis_);
    }

    /// The total number of basis functions.
    ///
    /// \return The product of `num_basis()`; 1 for a dimensionless space, which is
    ///         the empty tensor product's own convention.
    [[nodiscard]] std::int64_t num_total_basis() const noexcept { return num_total_basis_; }

    /// The number of knot intervals in each direction.
    ///
    /// \return A view of `dim()` counts, in axis order, each at least 1.
    [[nodiscard]] std::span<const std::int64_t> num_intervals() const noexcept {
        return std::span<const std::int64_t>(num_intervals_);
    }

    /// The total number of cells.
    ///
    /// \return The product of `num_intervals()`; 1 for a dimensionless space.
    [[nodiscard]] std::int64_t num_total_intervals() const noexcept {
        return num_total_intervals_;
    }

    /// The per-direction domain, as one row-major `(dim(), 2)` block.
    ///
    /// Row `d` is direction `d`'s own `domain()`, copied unchanged, so the two
    /// backends agree on it bit for bit. Flat rather than a vector of pairs because
    /// the binding hands it to `numpy` as a two-dimensional view of exactly this
    /// storage.
    ///
    /// \return A view of `2 * dim()` values: `{lo_0, hi_0, lo_1, hi_1, ...}`.
    [[nodiscard]] std::span<const T> domain_flat() const noexcept {
        return std::span<const T>(domain_);
    }

    /// Whether every direction describes a single Bézier segment.
    ///
    /// A three-iteration accessor rather than a field, because the oracle's
    /// counterpart is a method and computes on call.
    ///
    /// \return `true` if every direction is non-periodic, clamped at both ends and
    ///         has exactly `degree + 1` basis functions. `true` for a dimensionless
    ///         space, which is what `all(())` gives.
    [[nodiscard]] bool has_bezier_like_knots() const noexcept {
        for (const auto& space : spaces_) {
            if (!space->has_bezier_like_knots()) {
                return false;
            }
        }
        return true;
    }

  private:
    /// Refuse a direction index that is not one of this space's.
    ///
    /// \param d The requested direction.
    /// \throws std::out_of_range If `d` is not in `[0, dim())`.
    void check_direction(std::int64_t d) const {
        if (d < 0 || d >= dim()) {
            throw std::out_of_range("direction must lie in [0, " + std::to_string(dim())
                                    + "); got " + std::to_string(d));
        }
    }

    /// Set every derived field from the directions, which are already stored.
    ///
    /// One pass over the directions for the four per-direction sequences, then the
    /// two products and the tolerance. Called by both constructors, which is the
    /// whole reason it exists.
    ///
    /// \throws pantr::CapacityError If a tensor-product total overflows.
    void fill_derived() {
        const std::size_t n = spaces_.size();
        degrees_.reserve(n);
        num_basis_.reserve(n);
        num_intervals_.reserve(n);
        domain_.reserve(2 * n);
        // Indexed rather than range-based, so that the running maximum can be
        // seeded by the first direction rather than by a sentinel. Two directions
        // are very often the *same* handle -- `BsplineSpace([s, s])` is the common
        // spelling in the suite -- so "is this the first one?" cannot be asked of
        // the handle, only of the index.
        for (std::size_t d = 0; d < n; ++d) {
            const BsplineSpace1D<T>& space = *spaces_[d];
            degrees_.push_back(space.degree());
            num_basis_.push_back(space.num_basis());
            num_intervals_.push_back(space.num_intervals());
            const std::array<T, 2> ends = space.domain();
            domain_.push_back(ends[0]);
            domain_.push_back(ends[1]);
            tol_ = (d == 0) ? space.tolerance() : std::max(tol_, space.tolerance());
        }
        num_total_basis_ = detail::checked_product(num_basis(), "num_total_basis");
        num_total_intervals_ = detail::checked_product(num_intervals(), "num_total_intervals");
    }

    std::vector<std::shared_ptr<const BsplineSpace1D<T>>> spaces_;  ///< One per direction.
    std::vector<std::int64_t> degrees_;                             ///< One per direction.
    std::vector<std::int64_t> num_basis_;                           ///< One per direction.
    std::vector<std::int64_t> num_intervals_;                       ///< One per direction.
    std::vector<T> domain_;              ///< Row-major `(dim, 2)`.
    double tol_ = 0.0;                   ///< The largest direction tolerance.
    std::int64_t num_total_basis_ = 1;   ///< The product of `num_basis_`.
    std::int64_t num_total_intervals_ = 1;  ///< The product of `num_intervals_`.
};

}  // namespace pantr::bspline
