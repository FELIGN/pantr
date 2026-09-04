/// \file
/// The B-spline field: what it holds, what it shares, and what it refuses.
///
/// ## What is asserted, and against what
///
/// Nothing here carries a numerical tolerance, and that is a property of the type
/// rather than a choice about the test. A field stores a handle, a buffer and a
/// flag; `dim`, `degree` and `rank` are integer reads and one subtraction. There is
/// no arithmetic on a coefficient anywhere in this file's subject, so `==` is the
/// only comparison that says anything -- a tolerance would be hiding a
/// transcription error rather than allowing for rounding. The coefficients are
/// therefore asserted against the exact values they were built from.
///
/// ## The case set is asymmetric where it can be
///
/// A tensor-product type is where a symmetric case set proves nothing: if two
/// directions agree on their basis count then a transposed net, an off-by-one in
/// the axis index and a shape check that compares sums all pass.
/// `check_a_surface` and `check_a_volume` therefore have no two directions alike in
/// the degree, the basis count or the domain, and a component count that is none of
/// the basis counts -- so no permutation of a net's shape is another admissible
/// shape for its space. Those two read a valid field's properties; what they buy is
/// that a per-direction forward cannot be right by coincidence.
///
/// **Refusing a wrong shape is a different property and a different case.**
/// `check_the_shape_is_the_spaces_basis_counts` is where it is tested, on a space of
/// its own, and it needs three nets rather than one: transposed, mis-ranked, and one
/// whose *later* direction is wrong -- the last because a transposed net has
/// direction 0 wrong too, so a check that compares the first extent and stops
/// refuses it anyway.
///
/// ## The five cases that exist because getting them wrong is silent
///
/// **`check_the_space_is_shared_not_copied`.** The constructor takes a handle and
/// must store it, not a copy of what it points at. If it copied, every value
/// assertion here would still pass, and the Python contract
/// `Bspline(space, cp).space is space` would present two Python objects agreeing on
/// identity over two different C++ objects -- `design/bspline_ownership_lifetime.md`
/// F6. Comparing addresses is what distinguishes the two.
///
/// **`check_the_space_outlives_the_field`.** The reason class H stores a
/// `shared_ptr<const T>` rather than taking `reference_internal` on the binding is
/// that the guarantee has to live in the type, where a consumer with no interpreter
/// gets it too. So the space is taken out, the field destroyed, and the space read.
/// F4 of that note is why the assertion is on a *count* rather than on the pointer
/// being non-null: a scalar read after free returns the correct value often enough
/// that the obvious version of this test passes on a broken design.
///
/// **`check_the_net_is_copied_not_aliased`.** This is the one place the port
/// deliberately disagrees with its oracle, and it disagrees in the direction that
/// looks like nothing happening: the oracle stores the caller's array, so a write
/// through it moves an already-validated field, and this type stores a copy so it
/// does not. Asserted by mutating the source buffer afterwards, which is the only
/// way to tell a copy from a view when both read back the same values.
///
/// **`check_a_periodic_direction`.** The net's parametric extents are the space's
/// **basis counts**, which for a periodic direction are not `len(knots) - degree - 1`.
/// A field built with the non-periodic count has the wrong number of coefficients
/// and must be refused; a field built with the periodic count must be accepted. The
/// expected counts are stated as literals derived from
/// `pantr/bspline/knots.hpp`'s own formula rather than read off the space, so this
/// case cannot agree with the implementation by taking its answer from it.
///
/// **`check_concurrent_reads`.** `bspline.hpp` claims every accessor is safe to
/// call concurrently with no external locking. Here that should hold structurally
/// -- there is no memo and nothing `mutable` -- but "should" is what
/// `design/bspline_derived_caches.md` F3 refutes one level down, where a bare
/// `mutable std::optional` gave 60 correct answers in 60 unsanitized runs and four
/// ThreadSanitizer reports. So the claim gets a gate rather than a reader's trust,
/// and the gate is a sanitizer build rather than an assertion on a value.
///
/// ## The refusal messages are compared against literals
///
/// `check_refusals` hardcodes each message. The parity file compares the same
/// messages against the *live* oracle; this half is what makes a reworded message a
/// failure even when nobody runs Python. Two of the seven have no oracle
/// counterpart and are C++-only, and the file comment of `bspline.hpp` says which
/// and why.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "check.hpp"
#include "pantr/bspline/bspline.hpp"

namespace {

using pantr::bezier::ControlNet;
using pantr::bspline::Bspline;
using pantr::bspline::BsplineSpace;
using pantr::bspline::BsplineSpace1D;
using pantr::bspline::KnotSnapping;

/// Build a shared univariate space from a knot vector and a degree.
///
/// \tparam T The scalar type.
/// \param knots The knot vector.
/// \param degree The polynomial degree.
/// \param periodic Whether the space wraps.
/// \return A handle on the space, snapping near-duplicates as the oracle's default
///         does.
template <class T>
std::shared_ptr<const BsplineSpace1D<T>> one_d(const std::vector<T>& knots, std::int64_t degree,
                                               bool periodic = false) {
    return std::make_shared<const BsplineSpace1D<T>>(std::span<const T>(knots), degree, periodic,
                                                     KnotSnapping::merge_near_duplicates);
}

/// Build a shared tensor-product space from univariate handles.
///
/// \tparam T The scalar type.
/// \param spaces One handle per direction, in axis order.
/// \return A handle on the tensor-product space.
template <class T>
std::shared_ptr<const BsplineSpace<T>>
nd(std::vector<std::shared_ptr<const BsplineSpace1D<T>>> spaces) {
    return std::make_shared<const BsplineSpace<T>>(std::move(spaces));
}

/// A control net over the given shape, holding `1, 2, 3, ...`.
///
/// Consecutive integers rather than a constant, so that a transposed or
/// wrongly-strided copy reads back different values instead of the same ones.
///
/// \tparam T The scalar type.
/// \param shape The extents, the last being the component count.
/// \return The net.
template <class T>
ControlNet<T> ramp(const std::vector<std::size_t>& shape) {
    std::size_t size = 1;
    for (const std::size_t extent : shape) {
        size *= extent;
    }
    std::vector<T> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        values[i] = static_cast<T>(i + 1);
    }
    return ControlNet<T>(std::span<const T>(values), std::span<const std::size_t>(shape));
}

/// Whether a span holds exactly the given values, in order.
///
/// \tparam T The element type.
/// \param actual The span.
/// \param expected The values.
/// \return `true` if the two agree elementwise.
template <class T>
bool equals(std::span<const T> actual, const std::vector<T>& expected) {
    if (actual.size() != expected.size()) {
        return false;
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (!(actual[i] == expected[i])) {
            return false;
        }
    }
    return true;
}

/// The message of the `std::invalid_argument` a call raises, or a marker.
///
/// \tparam F The callable.
/// \param build A no-argument call expected to throw.
/// \return The message, or a marker saying what happened instead, so that the
///         caller's assertion is the one that reports.
template <class F>
std::string refusal_of(F build) {
    try {
        build();
    } catch (const std::invalid_argument& error) {
        return error.what();
    } catch (const std::exception& error) {
        return std::string("<other exception: ") + error.what() + ">";
    }
    return "<did not throw>";
}

/// A vector-valued surface whose every direction differs from every other.
///
/// Direction 0 is a clamped quadratic over three intervals on `[0, 3]`, so five
/// basis functions; direction 1 is a clamped linear over two intervals on
/// `[10, 12]`, so three. The component count is 4, which is neither basis count, so
/// no permutation of `(5, 3, 4)` is another admissible shape for this space.
void check_a_surface() {
    const std::vector<double> first{0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0};
    const std::vector<double> second{10.0, 10.0, 11.0, 12.0, 12.0};
    const auto space = nd<double>({one_d(first, 2), one_d(second, 1)});
    const Bspline<double> field(space, ramp<double>({5, 3, 4}), false);

    PANTR_CHECK(field.dim() == 2);
    PANTR_CHECK(equals(field.degree(), std::vector<std::int64_t>{2, 1}));
    PANTR_CHECK_MSG(field.rank() == 4, "non-rational: the rank is the whole component axis");
    PANTR_CHECK(!field.is_rational());
    PANTR_CHECK(equals(field.net().shape(), std::vector<std::size_t>{5, 3, 4}));
    PANTR_CHECK(field.net().size() == 60);
    PANTR_CHECK_MSG(field.net().values()[0] == 1.0 && field.net().values()[59] == 60.0,
                    "the coefficients arrived in order and none was lost at either end");
    PANTR_CHECK_MSG(field.space().get() == space.get(), "the space is the handle it was given");
    PANTR_CHECK(&field.space_ref() == space.get());
}

/// A three-direction field, no two directions alike in anything.
///
/// A third direction is what distinguishes a per-direction forward from one that
/// happens to be right for two: `degree()` and the net's shape are both sequences,
/// and a two-entry sequence read backwards is still a two-entry sequence.
///
/// Degrees 2, 1 and 3; basis counts 5, 3 and 7; domains `[0, 3]`, `[10, 12]` and
/// `[100, 104]`, three different scales. The component axis is 2, which is none of
/// the basis counts, so no permutation of `(5, 3, 7, 2)` is another admissible shape
/// for this space.
void check_a_volume() {
    const std::vector<double> first{0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0};
    const std::vector<double> second{10.0, 10.0, 11.0, 12.0, 12.0};
    const std::vector<double> third{100.0, 100.0, 100.0, 100.0, 101.0,
                                    102.0, 103.0, 104.0, 104.0, 104.0, 104.0};
    const auto space = nd<double>({one_d(first, 2), one_d(second, 1), one_d(third, 3)});
    const Bspline<double> field(space, ramp<double>({5, 3, 7, 2}), false);

    PANTR_CHECK(field.dim() == 3);
    PANTR_CHECK(equals(field.degree(), std::vector<std::int64_t>{2, 1, 3}));
    PANTR_CHECK(equals(field.space()->num_basis(), std::vector<std::int64_t>{5, 3, 7}));
    PANTR_CHECK(field.rank() == 2);
    PANTR_CHECK(equals(field.net().shape(), std::vector<std::size_t>{5, 3, 7, 2}));
    PANTR_CHECK(field.net().size() == 210);
    PANTR_CHECK_MSG(field.net().values()[209] == 210.0,
                    "the last coefficient of the largest net arrived");
}

/// A rational curve: the weight column is stored and is not part of the rank.
void check_a_rational_curve() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    const auto space = nd<double>({one_d(knots, 2)});
    const Bspline<double> field(space, ramp<double>({3, 3}), true);

    PANTR_CHECK(field.dim() == 1);
    PANTR_CHECK(field.is_rational());
    PANTR_CHECK_MSG(field.rank() == 2, "three stored components, one of them the weight");
    PANTR_CHECK_MSG(field.net().num_components() == 3, "and the storage still holds all three");

    const Bspline<double> polynomial(space, ramp<double>({3, 3}), false);
    PANTR_CHECK_MSG(polynomial.rank() == 3,
                    "the same net read as non-rational has one more component of rank, "
                    "so the flag is what the rank depends on and not the shape alone");
}

/// A scalar field, which is the shape the projection and interpolation layers build.
void check_a_scalar_field() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0};
    const auto space = nd<double>({one_d(knots, 2)});
    const Bspline<double> field(space, ramp<double>({4, 1}), false);

    PANTR_CHECK(field.rank() == 1);
    PANTR_CHECK(equals(field.net().shape(), std::vector<std::size_t>{4, 1}));
}

/// The net's parametric extents are the space's basis counts, in axis order.
///
/// The vacuity guard for the whole case set: a transposed net has exactly as many
/// coefficients as a correct one, so nothing about a *size* can refuse it.
///
/// Three refusals rather than one, and the middle one is the reason: a transposed
/// net has direction 0 wrong, so it is caught by a check that compares the first
/// extent and stops. `later_direction` has direction 0 *right*, which is what makes
/// the loop's bound observable.
void check_the_shape_is_the_spaces_basis_counts() {
    const std::vector<double> first{0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0};  // 5 basis
    const std::vector<double> second{10.0, 10.0, 11.0, 12.0, 12.0};           // 3 basis
    const auto space = nd<double>({one_d(first, 2), one_d(second, 1)});

    const std::string transposed = refusal_of([&] {
        return Bspline<double>(space, ramp<double>({3, 5, 4}), false);
    });
    PANTR_CHECK_MSG(transposed
                        == "the control net has 3 coefficient(s) along direction 0 and the "
                           "space has 5 basis function(s)",
                    "message was: " + transposed);

    // Direction 0 correct and direction 1 wrong, which is the case that distinguishes
    // a loop over every direction from one that compares the first and stops. Measured:
    // without it, narrowing the check to `d < 1` left this file green.
    const std::string later_direction = refusal_of([&] {
        return Bspline<double>(space, ramp<double>({5, 4, 2}), false);
    });
    PANTR_CHECK_MSG(later_direction
                        == "the control net has 4 coefficient(s) along direction 1 and the "
                           "space has 3 basis function(s)",
                    "message was: " + later_direction);

    const std::string mis_ranked = refusal_of([&] {
        return Bspline<double>(space, ramp<double>({15, 4}), false);
    });
    PANTR_CHECK_MSG(mis_ranked
                        == "the control net has 1 parametric direction(s) and the space has 2",
                    "message was: " + mis_ranked);

    // Bad in both ways at once, which is what pins the ORDER of the net
    // constructor's two checks: transposed AND, being rational with one stored
    // component, of rank 0. Every other case here violates one rule and has one
    // possible message, so a reordering would be invisible.
    const std::string both = refusal_of([&] {
        return Bspline<double>(space, ramp<double>({3, 5, 1}), true);
    });
    PANTR_CHECK_MSG(both == transposed,
                    "the shape check runs before the rank check, so a net that is both "
                    "transposed and rank-zero reports the shape: "
                        + both);
}

/// The space is stored as the handle it was given, not as a copy of its value.
void check_the_space_is_shared_not_copied() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    const auto one = one_d<double>(knots, 2);
    const auto space = nd<double>({one});
    const auto before = space.use_count();
    const Bspline<double> field(space, ramp<double>({3, 2}), false);

    PANTR_CHECK_MSG(field.space().get() == space.get(),
                    "a copying constructor would give a different address here and pass "
                    "every value assertion in this file");
    PANTR_CHECK_MSG(space.use_count() > before, "and the field really took a reference");
    PANTR_CHECK_MSG(field.space()->space(0).get() == one.get(),
                    "sharing composes: the field's space shares its directions too");
}

/// A space taken out of a field stays valid after the field is destroyed.
///
/// Asserted on a count rather than on a non-null pointer, because
/// `design/bspline_ownership_lifetime.md` F4 measured that a scalar read after free
/// returns the correct value often enough for the obvious test to pass on a broken
/// design. Under a `shared_ptr` this is not a read after free at all, which is the
/// point; under a raw reference the sanitizer leg reports one.
void check_the_space_outlives_the_field() {
    const std::vector<double> first{0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0};
    const std::vector<double> second{10.0, 10.0, 11.0, 12.0, 12.0};
    std::shared_ptr<const BsplineSpace<double>> escaped;
    {
        const Bspline<double> field(nd<double>({one_d(first, 2), one_d(second, 1)}),
                                    ramp<double>({5, 3, 4}), false);
        escaped = field.space();
        PANTR_CHECK(escaped.use_count() >= 2);
    }
    PANTR_CHECK_MSG(escaped.use_count() == 1, "the field released its reference");
    PANTR_CHECK_MSG(escaped->num_total_basis() == 15, "and the value survived it");
    PANTR_CHECK(equals(escaped->degrees(), std::vector<std::int64_t>{2, 1}));
}

/// The field copies the coefficients it is built from and does not alias them.
///
/// The one deliberate divergence from the oracle, and the one that looks like
/// nothing happening: the oracle stores the caller's array, so a later write moves
/// an already-validated field. Mutating the source afterwards is the only way to
/// tell a copy from a view when both read back the same values at first.
void check_the_net_is_copied_not_aliased() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    const auto space = nd<double>({one_d(knots, 2)});
    std::vector<double> values{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const std::vector<std::size_t> shape{3, 2};
    const Bspline<double> field(
        space, ControlNet<double>(std::span<const double>(values),
                                  std::span<const std::size_t>(shape)),
        false);

    PANTR_CHECK(field.net().values()[0] == 1.0);
    values[0] = 99.0;
    PANTR_CHECK_MSG(field.net().values()[0] == 1.0,
                    "a write through the source buffer must not reach a built field");
    PANTR_CHECK_MSG(field.net().values().data() != values.data(),
                    "and the storage is genuinely the field's own");
}

/// The flat constructor derives the shape the net constructor is handed.
///
/// Both spellings must produce the same field, and the flat one must derive the
/// component count from the space rather than guess it: the same 24 coefficients
/// are rank 2 over a 12-basis space and rank 4 over a 6-basis one.
void check_the_flat_constructor_derives_the_shape() {
    const std::vector<double> first{0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0};  // 5 basis
    const std::vector<double> second{10.0, 10.0, 11.0, 12.0, 12.0};           // 3 basis
    const auto space = nd<double>({one_d(first, 2), one_d(second, 1)});
    std::vector<double> values(60);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<double>(i + 1);
    }

    const Bspline<double> flat(space, std::span<const double>(values), false);
    const Bspline<double> shaped(space, ramp<double>({5, 3, 4}), false);
    PANTR_CHECK(equals(flat.net().shape(), std::vector<std::size_t>{5, 3, 4}));
    PANTR_CHECK(equals(flat.net().values(), values));
    PANTR_CHECK(equals(flat.net().values(), std::vector<double>(shaped.net().values().begin(),
                                                               shaped.net().values().end())));
    PANTR_CHECK(flat.rank() == shaped.rank());
    PANTR_CHECK(flat.rank() == 4);

    // The same buffer over a space with a different total is a different rank, so
    // the derivation reads the space rather than assuming a component count.
    const std::vector<double> narrow{0.0, 0.0, 1.0, 1.0};  // 2 basis, degree 1
    const auto small = nd<double>({one_d(narrow, 1), one_d(narrow, 1)});  // 4 total
    const Bspline<double> wide(small, std::span<const double>(values), false);
    PANTR_CHECK_MSG(wide.rank() == 15, "60 coefficients over 4 basis functions is rank 15");
}

/// A periodic direction stores its own basis count, not `len(knots) - degree - 1`.
///
/// The expected counts are literals derived from `pantr/bspline/knots.hpp`'s
/// formula, not read off the space: for a periodic direction the count is
/// `len(knots) - degree - 1 - (degree - multiplicity_of_first_in_domain) - 1`.
/// This knot vector has 9 entries at degree 2 and the first in-domain knot is
/// simple, so the count is `9 - 2 - 1 - (2 - 1) - 1 = 4`, against `6` if the
/// direction were read as non-periodic.
void check_a_periodic_direction() {
    const std::vector<double> knots{-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const auto direction = one_d<double>(knots, 2, true);
    PANTR_CHECK_MSG(direction->num_basis() == 4,
                    "the case is only meaningful if the space agrees with the hand "
                    "derivation, and it is the hand derivation that is the oracle here");
    const auto space = nd<double>({direction});

    const Bspline<double> field(space, ramp<double>({4, 3}), false);
    PANTR_CHECK(field.rank() == 3);
    PANTR_CHECK(equals(field.net().shape(), std::vector<std::size_t>{4, 3}));

    const std::string refused = refusal_of([&] {
        return Bspline<double>(space, ramp<double>({6, 3}), false);
    });
    PANTR_CHECK_MSG(refused
                        == "the control net has 6 coefficient(s) along direction 0 and the "
                           "space has 4 basis function(s)",
                    "message was: " + refused);
}

/// Every construction the type must refuse, and the exact text of each.
///
/// The order of the two checks the flat constructor makes is pinned by the last
/// case, which is bad in both ways at once: every other entry violates one rule and
/// has one possible message, so a reordering would be invisible.
void check_refusals() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 1.0, 1.0, 1.0};  // 3 basis, degree 2
    const auto space = nd<double>({one_d(knots, 2)});
    const std::vector<double> three{1.0, 2.0, 3.0};
    const std::vector<double> four{1.0, 2.0, 3.0, 4.0};

    const std::string null_space = refusal_of([&] {
        return Bspline<double>(nullptr, ramp<double>({3, 2}), false);
    });
    PANTR_CHECK_MSG(null_space == "the B-spline space is a null handle",
                    "message was: " + null_space);

    const std::string null_flat = refusal_of([&] {
        return Bspline<double>(nullptr, std::span<const double>(three), false);
    });
    PANTR_CHECK_MSG(null_flat == "the B-spline space is a null handle",
                    "the flat overload validates the space too: " + null_flat);

    const auto dimensionless =
        nd<double>(std::vector<std::shared_ptr<const BsplineSpace1D<double>>>{});
    const std::string no_directions = refusal_of([&] {
        return Bspline<double>(dimensionless, ramp<double>({1, 1}), false);
    });
    PANTR_CHECK_MSG(no_directions == "a B-spline over a space with no directions has no control net",
                    "message was: " + no_directions);

    const std::string rank_zero = refusal_of([&] {
        return Bspline<double>(space, ramp<double>({3, 1}), true);
    });
    PANTR_CHECK_MSG(rank_zero == "The B-spline must have at least rank one. Got rank 0",
                    "message was: " + rank_zero);

    const std::string rank_negative = refusal_of([&] {
        return Bspline<double>(space, ramp<double>({3, 0}), true);
    });
    PANTR_CHECK_MSG(rank_negative == "The B-spline must have at least rank one. Got rank -1",
                    "a rational net with no components at all is why the arithmetic is "
                    "signed; an unsigned one would report a number near 2^64: "
                        + rank_negative);

    const std::string not_a_multiple = refusal_of([&] {
        return Bspline<double>(space, std::span<const double>(four), false);
    });
    PANTR_CHECK_MSG(not_a_multiple
                        == "The number of control points must be a multiple of the number of "
                           "basis functions.Got 4 control points and 3 basis functions.",
                    "the oracle's own text, missing space included: " + not_a_multiple);

    // Bad in both ways: 4 is not a multiple of 3, AND a rational field of three
    // coefficients would have rank 0. Only the order decides which is reported.
    const std::string both = refusal_of([&] {
        return Bspline<double>(space, std::span<const double>(four), true);
    });
    PANTR_CHECK_MSG(both == not_a_multiple,
                    "the count check runs before the rank check, as the oracle's does: "
                        + both);

    // An empty buffer is the one case the count check does NOT refuse, because zero
    // is a multiple of everything; the rank check is what catches it, and the header
    // says so. Asserted because a documented route through a check is a claim like
    // any other. Unreachable from Python, where numpy's reshape refuses it first.
    const std::vector<double> none;
    const std::string empty = refusal_of([&] {
        return Bspline<double>(space, std::span<const double>(none), false);
    });
    PANTR_CHECK_MSG(empty == "The B-spline must have at least rank one. Got rank 0",
                    "an empty flat buffer must be refused by the rank check: " + empty);
}

/// The `float32` field, which is the half of the matrix nobody reads first.
void check_float_storage() {
    const std::vector<float> knots{0.0F, 0.0F, 0.0F, 1.0F, 2.0F, 2.0F, 2.0F};
    const auto space = nd<float>({one_d(knots, 2)});
    const Bspline<float> field(space, ramp<float>({4, 3}), true);

    PANTR_CHECK(field.dim() == 1);
    PANTR_CHECK(field.rank() == 2);
    PANTR_CHECK(equals(field.net().shape(), std::vector<std::size_t>{4, 3}));
    PANTR_CHECK(field.net().values()[11] == 12.0F);
}

/// A field is copyable and movable, and a copy shares the space rather than cloning it.
void check_copy_and_move() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    const auto space = nd<double>({one_d(knots, 2)});
    const Bspline<double> field(space, ramp<double>({3, 2}), false);

    const Bspline<double> copied = field;
    PANTR_CHECK_MSG(copied.space().get() == space.get(),
                    "a copy shares the space; there is nothing to deep-copy behind an "
                    "immutable handle");
    PANTR_CHECK(copied.rank() == 2);
    PANTR_CHECK_MSG(copied.net().values().data() != field.net().values().data(),
                    "but it owns its own coefficients");

    Bspline<double> source(space, ramp<double>({3, 2}), false);
    const Bspline<double> moved = std::move(source);
    PANTR_CHECK(moved.space().get() == space.get());
    PANTR_CHECK(equals(moved.net().values(), std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
}

/// Every accessor is safe to call concurrently on one field, with no locking.
///
/// `space()` is included deliberately and is the only accessor here with machinery
/// behind it: it copies a `shared_ptr`, so eight threads hammering it contend on one
/// atomic control block. The threads are released together off an atomic flag so
/// that the reads genuinely overlap; a run in which each finished before the next
/// began would report clean for the wrong reason.
///
/// **Nothing here asserts that the overlap happened, and nothing can**: that is the
/// sanitizer's to detect, and `design/bspline_derived_caches.md` F3 is the reason --
/// the shape one level down gave 60 correct answers in 60 unsanitized runs and four
/// ThreadSanitizer reports. What the assertions do is stop the loop becoming dead
/// code, which would make the sanitizer's silence meaningless. So the total is
/// compared against a hand-computed literal rather than only against the other
/// threads' totals: eight threads agreeing on zero is agreement.
///
/// An earlier version ended on `field.space().use_count() >= 2` and called that the
/// vacuity guard. It was not one. `space` is a named local held for the whole
/// function, so the local plus the field's own member already make the count 2 the
/// instant the field is built -- before any thread runs, and whether or not the loop
/// body calls anything at all.
void check_concurrent_reads() {
    const std::vector<double> first{0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0};
    const std::vector<double> second{10.0, 10.0, 11.0, 12.0, 12.0};
    const auto space = nd<double>({one_d(first, 2), one_d(second, 1)});
    const Bspline<double> field(space, ramp<double>({5, 3, 4}), false);

    constexpr int num_threads = 8;
    std::atomic<bool> go{false};
    std::vector<double> sums(num_threads, 0.0);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&field, &go, &sums, t] {
            while (!go.load(std::memory_order_acquire)) {
                // Spin until every thread is up, so the reads overlap.
            }
            double sum = 0.0;
            for (int i = 0; i < 2000; ++i) {
                sum += static_cast<double>(field.dim() + field.rank());
                sum += static_cast<double>(field.degree()[0] + field.degree()[1]);
                sum += field.is_rational() ? 1.0 : 0.0;
                sum += field.net().values()[0] + field.net().values()[59];
                sum += static_cast<double>(field.net().num_components());
                // The one accessor with an atomic in it.
                sum += static_cast<double>(field.space()->num_total_basis());
                sum += static_cast<double>(field.space_ref().dim());
            }
            sums[static_cast<std::size_t>(t)] = sum;
        });
    }
    go.store(true, std::memory_order_release);
    for (std::thread& thread : threads) {
        thread.join();
    }

    // Per iteration: dim + rank (2 + 4), the two degrees (2 + 1), the flag (0), the
    // first and last coefficients (1 + 60), the component count (4), the space's
    // total basis count (5 * 3), and the borrowed space's dim (2). That is 91, over
    // 2000 iterations. Hand-computed from the case above, so it dies if any accessor
    // stops being called, starts returning something else, or the loop goes dead.
    constexpr double expected = 91.0 * 2000.0;
    for (int t = 0; t < num_threads; ++t) {
        PANTR_CHECK_MSG(sums[static_cast<std::size_t>(t)] == sums[0],
                        "every thread must read one state");
        PANTR_CHECK_MSG(sums[static_cast<std::size_t>(t)] == expected,
                        "a thread's total is not the hand-computed one, so the loop body "
                        "is not reading what this test says it reads");
    }
}

}  // namespace

int main() {
    check_a_surface();
    check_a_volume();
    check_a_rational_curve();
    check_a_scalar_field();
    check_the_shape_is_the_spaces_basis_counts();
    check_the_space_is_shared_not_copied();
    check_the_space_outlives_the_field();
    check_the_net_is_copied_not_aliased();
    check_the_flat_constructor_derives_the_shape();
    check_a_periodic_direction();
    check_refusals();
    check_float_storage();
    check_copy_and_move();
    check_concurrent_reads();
    return pantr::test::summary("test_bspline_type");
}
