#pragma once

/// \file
/// The B-spline field: a tensor-product space, a control net over it, and the
/// quantities the pair fixes.
///
/// ## What this type owns, and what it does not
///
/// It owns the *value* -- a handle on the space, the control points, and the
/// rationality flag -- plus the quantities they determine: `dim`, `degree` and
/// `rank`.
///
/// It owns no *operations*, exactly as `pantr/bspline/space_nd.hpp` and
/// `pantr/bspline/space_1d.hpp` own none. Evaluation, the derivative, degree
/// elevation and reduction, knot insertion and removal, the boundary and periodic
/// conversions, restriction, splitting, slicing, the Bézier decomposition, the
/// product, reversal, direction permutation and the affine transform are
/// computations *over* a field rather than properties *of* one, so they are
/// separate ports over free functions taking a `const Bspline&`. That is what lets
/// them proceed independently of one another once this type exists, and it is the
/// same line those two headers draw.
///
/// **Two of them cannot be ported yet, and the reason is a declared boundary
/// rather than an omission.** `pantr.bspline.Bspline.evaluate`,
/// `.evaluate_derivatives` and `.to_beziers` reach
/// `BsplineSpace1D.tabulate_basis` or `.tabulate_basis_derivatives`, and
/// `space_1d.hpp` states that basis tabulation is a separate port over free
/// functions which no ticket in this milestone covers. So those operations wait on
/// that port, not on this one.
///
/// ## The space is shared, not copied, and the identity contract is why
///
/// `space_` is a `std::shared_ptr<const BsplineSpace<T>>`, per
/// `design/bspline_ownership_lifetime.md`'s class **H**: an accessor that hands out
/// a subobject the owner keeps returns a copy of the handle, so the *value* is
/// shared and the owner's death is irrelevant. `rv_policy::reference_internal`
/// would put that guarantee in the binding, and the binding is scheduled for
/// deletion; a `shared_ptr<const T>` puts it in the type, where a C++ consumer with
/// no interpreter present gets it too.
///
/// This is the type that note's reason 3 was measured on. Of three storage shapes
/// it tabulates, only "store the handle, return a copy of the handle" reproduces
/// today's Python: storing the space by value and returning a reference makes an
/// escaped space silently start reporting the *new* space after an in-place
/// reseat, and storing the handle while returning a reference is a use-after-free
/// that reads back the correct value.
///
/// The borrowing twin, `space_ref`, is **not bound**: copying the handle costs an
/// uncontended atomic increment/decrement pair per access, which an inner loop must
/// not pay for a value it does not keep.
/// `tests/parity/test_bspline_binding_contract.py` asserts that no bound method
/// name ends in `_ref`.
///
/// ## No mutation, and therefore no `in_place`
///
/// `pantr.bspline.Bspline` is the one type in this milestone whose Python surface
/// mutates: `reverse`, `permute_directions` and `transform` each take
/// `in_place=True`, and the first two *reseat* the field's space.
/// `design/bspline_derived_caches.md` calls it the type where construct-then-freeze
/// does not hold.
///
/// **It holds here.** This type offers no mutator of any kind, and that is a
/// decision with three reasons rather than an oversight:
///
///  - The project's rule is that a mutating sweep earns its place only when the
///    mutation is **unobservable**. `in_place=True` is observable by construction
///    -- it is chosen precisely so that a caller's handle sees the new value -- so
///    it cannot be earned. The two ways the rule does admit are to take the object
///    by value or to mutate a private duplicate, and both of those are just
///    "return a new field".
///  - `design/bspline_derived_caches.md` states as a **contract** that every
///    non-mutating public accessor on a C++-owned pantr domain type is safe to call
///    concurrently on one object with no external locking, with exactly one
///    exception (a grid's tag registries, which are the reasoned
///    accumulating-container case). A reseating `Bspline` would be a second
///    exception with no such reason: a reseat of `space_` races every reader of
///    `space()`, and no memo discipline helps, because the racing write is on the
///    base state rather than on a memo.
///  - The interpreter-free consumer gains nothing from it. `b = reversed(b, 0);`
///    costs one move of a handle and a net. The Python flag exists to keep a numpy
///    array's *identity* stable across the mutation, and this type's storage is not
///    a numpy array a caller can hold in the first place.
///
/// So the Python `in_place=True` surface survives -- it is published API and
/// removing it is not this port's to do -- and it is implemented one level up, by
/// `pantr.bspline.Bspline` replacing its whole implementation in a single
/// assignment. That is also the repair `design/bspline_derived_caches.md` asks
/// this ticket for: the oracle's three separate cache-invalidation sites become one
/// assignment that replaces the derived block wholesale, so there is no way to
/// reseat one part of it without the other.
///
/// ## Where the derived caches are, and why not here
///
/// The oracle memoises two derived quantities on a field: the Bézier decomposition
/// and the point-inversion context. Neither is here, and the reason is the
/// operation boundary above rather than a disagreement with
/// `design/bspline_derived_caches.md`: both are *produced by operations that have
/// not been ported*, one of which cannot be until basis tabulation is. A memo can
/// only live beside the computation that fills it. They move here with the
/// operations that make them, and until then the wrapper holds them in one block
/// it replaces wholesale.
///
/// So there is no `LazySlot` in this type and nothing `mutable`.
///
/// ## The control net is a copy, and that is the one deliberate divergence
///
/// `ControlNet<T>` stores its own buffer, so this type does not alias the array it
/// was built from and hands out a read-only view of its own storage. The oracle
/// does neither: `pantr.bspline.Bspline` stores the caller's array and returns that
/// same object from `control_points`, so a caller can mutate a constructed field
/// through either end -- and doing so desynchronises both of the memos above with
/// nothing raising. `design/bspline_ownership_lifetime.md` records that as a defect
/// in today's Python which the port removes, and this is the half of the removal
/// that this ticket owns. It is the same divergence `pantr/bezier/bezier.hpp`
/// already carries for the same reason, in the same direction.
///
/// ## Reusing `ControlNet`, across the package boundary
///
/// `net_` is a `pantr::bezier::ControlNet<T>`, which is the first include of
/// another package's header from `pantr/bspline/`. It is deliberate:
/// `pantr/bezier/control_net.hpp` says of itself that it is *not* a Bézier and that
/// "for `pantr::bezier::Bezier` an extent is `degree + 1`, for a B-spline it is a
/// basis count, and neither reading belongs to the array". This is that second
/// reading. Writing a second owning `(shape, values)` pair here would be two
/// implementations of one idea, which is what this port exists to remove.
///
/// The dependency direction matches the Python layer's, where
/// `pantr.bspline._bspline_to_beziers` imports `pantr.bezier`. **The type's home is
/// still wrong**: `pantr::core` is where a shape-plus-storage array belongs, and
/// moving it is a rename across `pantr/bezier/bezier.hpp`, its binding, its tests
/// and the *installed* header set, which `pantr/bezier/bezier.hpp` records as a
/// promise. That is a separate change and is flagged rather than taken here.
///
/// ## Validating rather than asserting
///
/// This is the C++ counterpart of Layer 2, so it validates and throws in a release
/// build as much as in a debug one. `pantr/core/error.hpp` sets the split: value
/// and range checks live here, type-kind checks stay in the Python wrapper.
///
/// ## Parity notes for the Python oracle
///
/// `pantr.bspline.Bspline` is the oracle. What this type reproduces, and where it
/// deliberately parts company:
///
///  - **The refusal messages are the oracle's, character for character**, as every
///    other refusal in this port is. `tests/parity/test_bspline_type.py` compares
///    the two texts rather than just the exception type.
///  - **Two refusals of the oracle's are not reachable here, and stay the
///    wrapper's.** The dtype mismatch between the control points and the space is a
///    type-kind fact and `Bspline<T>` cannot hold a space of another width, so
///    there is nothing to check; and the oracle raises `numpy`'s own reshape error
///    for a control-point buffer that is empty while the space is not, whose text
///    is numpy's rather than pantr's. The wrapper does the reshape, so that text is
///    common mode and is not restated here.
///  - **The order of the checks is the oracle's.** The count check comes before
///    the rank check, so a field that is bad in both ways reports the count. Only
///    the order decides which message a caller reads, and
///    `cpp/tests/test_bspline_type.cpp` asserts the C++ half against a literal
///    while the parity file asserts it against the live oracle.
///  - **A field over a space with no directions is refused.** The oracle refuses
///    it too, but by accident and with a different exception: it reads
///    `space.dtype`, which raises `IndexError` on a dimensionless space. Since no
///    control net can have zero parametric axes -- `ControlNet` reads a rank-1
///    shape as `(n, 1)` -- there is no shape this constructor could accept, so it
///    says so instead of reporting a shape mismatch against an empty basis count.
///    The Python path never reaches it: the wrapper raises the oracle's
///    `IndexError` first.
///
/// ## Thread safety
///
/// Every accessor below is safe to call concurrently on the same object with no
/// external locking, and here that needs no mechanism at all: there is no memo and
/// nothing `mutable`, so every accessor reads state frozen at construction. In a
/// sweep, hoist `space_ref()` and `net()` into locals before the loop rather than
/// calling them per element -- the references are free, but the calls are not, and
/// `space()` costs an atomic pair.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pantr/bezier/control_net.hpp"
#include "pantr/bspline/space_nd.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr::bspline {

/// A B-spline curve, surface, volume or field over a tensor-product space.
///
/// Holds a handle on the space, a control net whose parametric extents are the
/// space's basis counts, and a flag. For a rational field the last component of
/// every coefficient is a homogeneous weight and the coordinates are stored
/// weighted, so `rank()` is one less than the net's component count.
///
/// Instances are immutable: nothing here changes a field once it is built. See the
/// file comment for why that holds even though the Python surface has three
/// `in_place=True` methods.
///
/// \tparam T The scalar type the control points and the space's knots share.
template <Real T>
class Bspline {
  public:
    /// The scalar type the control points and the space's knots share.
    using scalar_type = T;

    /// The control-net type this field stores.
    ///
    /// Named so that a caller building one does not have to know it comes from
    /// another package; see the file comment on why it does.
    using net_type = pantr::bezier::ControlNet<T>;

    /// Build a field from a space and a control net.
    ///
    /// The primary constructor, and the one the binding calls. Taking the net
    /// rather than a flat buffer is what makes the extents checkable: a transposed
    /// net has the same number of coefficients as a correct one, so the flat
    /// overload below cannot tell them apart and this one can.
    ///
    /// \param space The space, shared. Sharing rather than copying is what
    ///        preserves the wrapper's identity contract; see the file comment.
    /// \param net The control points, moved in. Its parametric extents must be the
    ///        space's basis counts, in axis order, and its last axis is the
    ///        component axis.
    /// \param is_rational Whether the last component of each coefficient is a
    ///        homogeneous weight.
    /// \throws std::invalid_argument If `space` is null, if it has no directions,
    ///         if the net's parametric shape is not the space's basis counts, or if
    ///         the resulting rank is not at least 1.
    Bspline(std::shared_ptr<const BsplineSpace<T>> space, net_type net, bool is_rational)
        : space_(std::move(space)), net_(std::move(net)), is_rational_(is_rational) {
        check_space(space_);
        check_net_matches_space();
        check_rank();
    }

    /// Build a field from a space and a flat coefficient buffer.
    ///
    /// The convenience the Python surface offers, for a C++ caller that has the
    /// coefficients and not the shape: the component count is derived as
    /// `values.size() / space->num_total_basis()`, and the parametric extents come
    /// from the space. It carries the oracle's own message for a buffer that is not
    /// a whole number of coefficients.
    ///
    /// \param space The space, shared. None may be null.
    /// \param values The coefficients, row-major under
    ///        `(*num_basis, num_components)`. Its size must be a multiple of
    ///        `space->num_total_basis()`.
    /// \param is_rational Whether the last component of each coefficient is a
    ///        homogeneous weight.
    /// \throws std::invalid_argument If `space` is null, if it has no directions,
    ///         if `values` is not a whole number of coefficients, or if the
    ///         resulting rank is not at least 1. An empty buffer is refused by the
    ///         rank check rather than by the count check, since zero *is* a
    ///         multiple; the message then says `rank 0`.
    Bspline(std::shared_ptr<const BsplineSpace<T>> space, std::span<const T> values,
            bool is_rational)
        : space_(std::move(space)), net_(net_from_flat(space_, values)),
          is_rational_(is_rational) {
        // `space_` is declared before `net_`, so it is initialized first and
        // `net_from_flat` can read it. The space's own validation happens there,
        // because the shape cannot be derived without it; repeating it here would
        // report one fault twice.
        check_rank();
    }

    /// Share this field's space.
    ///
    /// The returned handle keeps its value alive independently of this field, so a
    /// caller may outlive the owner.
    ///
    /// \return A handle on the space.
    [[nodiscard]] std::shared_ptr<const BsplineSpace<T>> space() const noexcept { return space_; }

    /// Borrow this field's space.
    ///
    /// Valid while `*this` is, and **not bound**: an inner loop must not pay an
    /// atomic pair per access. See the `_ref` rule in the file comment.
    ///
    /// \return A reference to the space.
    [[nodiscard]] const BsplineSpace<T>& space_ref() const noexcept { return *space_; }

    /// The control points.
    ///
    /// \return The stored net, valid while the field lives.
    [[nodiscard]] const net_type& net() const noexcept { return net_; }

    /// Whether the last component of each coefficient is a homogeneous weight.
    ///
    /// \return `true` for a rational field (a NURBS).
    [[nodiscard]] bool is_rational() const noexcept { return is_rational_; }

    /// The number of parametric directions.
    ///
    /// Forwarded from the space rather than read off the net, because the space is
    /// what defines it; the constructor is what makes the two agree.
    ///
    /// \return `space_ref().dim()`, at least 1.
    [[nodiscard]] std::int64_t dim() const noexcept { return space_->dim(); }

    /// The polynomial degree of each parametric direction.
    ///
    /// \return A view of `dim()` non-negative degrees, in axis order, owned by the
    ///         space and valid while it lives -- which is at least as long as this
    ///         field.
    [[nodiscard]] std::span<const std::int64_t> degree() const noexcept {
        return space_->degrees();
    }

    /// The number of value components a caller sees.
    ///
    /// The weight column of a rational field is not one of them: a rational surface
    /// in space stores four components and has rank 3.
    ///
    /// \return `net().num_components()`, less one when rational. At least 1.
    [[nodiscard]] std::int64_t rank() const noexcept { return signed_rank(); }

  private:
    /// The rank, before it is known to be positive.
    ///
    /// Split out so that the constructor's check and the public accessor cannot
    /// drift apart, which is `pantr/bezier/bezier.hpp`'s reason for the same split.
    /// Signed because the oracle reports a negative rank for a rational net with no
    /// coordinates at all, and reproducing its message means reproducing its
    /// arithmetic.
    ///
    /// \return The component count less the weight column, possibly zero or
    ///         negative.
    [[nodiscard]] std::int64_t signed_rank() const noexcept {
        return static_cast<std::int64_t>(net_.num_components())
               - (is_rational_ ? std::int64_t{1} : std::int64_t{0});
    }

    /// Refuse a space this field cannot be built over.
    ///
    /// Static and taking its argument, so that both constructors can run it: the
    /// flat one needs it inside its member-initializer list, before `net_` exists,
    /// where a non-static member function cannot go. One function rather than two
    /// copies of two message literals, which is the drift this shape removes.
    ///
    /// \param space The handle to check.
    /// \throws std::invalid_argument If the handle is null or the space has no
    ///         directions.
    static void check_space(const std::shared_ptr<const BsplineSpace<T>>& space) {
        if (space == nullptr) {
            throw std::invalid_argument("the B-spline space is a null handle");
        }
        if (space->dim() == 0) {
            throw std::invalid_argument(
                "a B-spline over a space with no directions has no control net");
        }
    }

    /// Refuse a net whose parametric shape is not the space's basis counts.
    ///
    /// The check the flat overload cannot make. It is near-vacuous on the Python
    /// path, where the wrapper derives the shape from the same `num_basis` this
    /// compares against, and load-bearing for a C++ caller, where a transposed net
    /// has exactly as many coefficients as a correct one.
    ///
    /// \throws std::invalid_argument If the net's dimension or any extent differs
    ///         from the space's.
    void check_net_matches_space() const {
        const std::span<const std::int64_t> counts = space_->num_basis();
        if (static_cast<std::int64_t>(net_.dim()) != space_->dim()) {
            throw std::invalid_argument(
                "the control net has " + std::to_string(net_.dim())
                + " parametric direction(s) and the space has " + std::to_string(space_->dim()));
        }
        for (std::size_t d = 0; d < net_.dim(); ++d) {
            const auto expected = static_cast<std::size_t>(counts[d]);
            if (net_.extent(d) != expected) {
                throw std::invalid_argument(
                    "the control net has " + std::to_string(net_.extent(d))
                    + " coefficient(s) along direction " + std::to_string(d)
                    + " and the space has " + std::to_string(expected) + " basis function(s)");
            }
        }
    }

    /// Refuse a field with no value components left after the weight column.
    ///
    /// \throws std::invalid_argument If the rank is not at least 1, with the
    ///         oracle's message.
    void check_rank() const {
        const std::int64_t rank = signed_rank();
        if (rank <= 0) {
            throw std::invalid_argument("The B-spline must have at least rank one. Got rank "
                                        + std::to_string(rank));
        }
    }

    /// The net a flat coefficient buffer forms over `space`.
    ///
    /// A static helper rather than a member, because it runs inside the member
    /// initializer list of the flat constructor, before `net_` exists. It validates
    /// the space itself, since the shape cannot be derived without one.
    ///
    /// \param space The space, already moved into `space_`.
    /// \param values The coefficients.
    /// \return A net of shape `(*num_basis, values.size() / num_total_basis)`.
    /// \throws std::invalid_argument If `space` is unusable, or if `values.size()`
    ///         is not a multiple of `space->num_total_basis()`, with the oracle's
    ///         message for the latter.
    [[nodiscard]] static net_type
    net_from_flat(const std::shared_ptr<const BsplineSpace<T>>& space,
                  std::span<const T> values) {
        check_space(space);
        const auto total = static_cast<std::size_t>(space->num_total_basis());
        if (values.size() % total != 0) {
            // The oracle's text, and the missing space after the full stop is the
            // oracle's too: it concatenates two f-strings without a separator.
            // Reproducing it is what makes a caller's `pytest.raises(match=...)`
            // backend-independent, so it is not a typo to tidy here.
            throw std::invalid_argument(
                "The number of control points must be a multiple of the number of basis "
                "functions.Got "
                + std::to_string(values.size()) + " control points and " + std::to_string(total)
                + " basis functions.");
        }
        std::vector<std::size_t> shape;
        shape.reserve(static_cast<std::size_t>(space->dim()) + 1);
        for (const std::int64_t count : space->num_basis()) {
            shape.push_back(static_cast<std::size_t>(count));
        }
        shape.push_back(values.size() / total);
        return net_type(values, std::span<const std::size_t>(shape));
    }

    std::shared_ptr<const BsplineSpace<T>> space_;  ///< The space, shared.
    net_type net_;                                  ///< The control points, owned.
    bool is_rational_;  ///< Whether the last component is a homogeneous weight.
};

}  // namespace pantr::bspline
