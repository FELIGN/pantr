/// \file
/// The tensor-product B-spline space: what it aggregates, and what it refuses.
///
/// ## What is asserted, and against what
///
/// Every quantity here is a reduction over the directions, so each case states the
/// per-direction values by hand and the expected reduction alongside them. Two of
/// the cases are the shipped docstring examples of
/// `pantr.bspline.BsplineSpace.cell_supports` and `.boundary_dofs`, so their basis
/// counts are checkable against the library's own published contract rather than
/// against a run of it.
///
/// Nothing here carries a numerical tolerance. The counts are integers, the domain
/// values are reproduced bit for bit from the directions, `has_bezier_like_knots`
/// is a boolean, and the tolerance is a *selection* -- one of the directions' own
/// tolerances, unmodified -- so `==` is the right comparison for it too.
///
/// ## The case set is asymmetric where it can be, and isolating where it cannot
///
/// A tensor-product type is the place where a symmetric test set proves nothing. If
/// two directions agree on their degree, their counts and their domain, then a
/// transposition, an off-by-one in the axis index, a `min` written for a `max` and a
/// reduction that returns its first argument all produce the right answer.
///
/// So the case set is in two parts, and the split is deliberate rather than untidy.
/// `check_two_directions` and `check_three_directions` are **fully asymmetric**:
/// their directions differ in the degree, the basis count, the interval count, the
/// domain and the scale at once, and `check_the_reductions_see_each_direction`
/// asserts exactly that of them -- a test of the test set, which fails if a later
/// case stops carrying its weight. The rest are **isolating**: they vary one
/// quantity and hold the others fixed, which is what lets a failure name the
/// reduction that moved. `check_the_tolerance_is_the_largest` uses three directions
/// that differ only in scale, and puts the argmax first, last and in the middle over
/// the same three values; `check_bezier_like_needs_every_direction` repeats one
/// direction and puts the single exception in each of three positions in turn.
/// Neither could be fully asymmetric without confounding what it isolates.
///
/// ## The four cases that exist because getting them wrong is silent
///
/// **`check_the_directions_are_shared_not_copied`.** The constructor takes handles
/// and must store them, not copies of what they point at. If it copied, every value
/// assertion in this file would still pass, and the Python identity contract
/// `space.spaces[0] is space_1d` -- `tests/test_bspline_space.py:89`, and
/// `design/bspline_ownership_lifetime.md` F6 -- would present two Python objects
/// agreeing on identity over two different C++ objects. Comparing addresses is what
/// distinguishes the two.
///
/// **`check_a_direction_outlives_the_space`.** The whole reason class H stores a
/// `shared_ptr<const T>` rather than taking `reference_internal` on the binding is
/// that the guarantee has to live in the type, where a consumer with no interpreter
/// gets it too. So a direction is taken out, the space is destroyed, and the
/// direction is read. Under the design this passes; under a raw reference the
/// sanitizer leg reports the use-after-free. `design/bspline_ownership_lifetime.md`
/// F4 is why this is asserted on a *count* the destroyed space would have
/// overwritten rather than on the pointer being non-null: a scalar read after free
/// returns the correct value often enough that the obvious version of this test
/// passes on a broken design.
///
/// **`check_the_totals_refuse_an_overflow`.** `numpy.prod` over an `int64` tuple
/// wraps; signed overflow in C++ is undefined, and the `gcc-debug` preset carries
/// `-fno-sanitize-recover=undefined`, so a wrap here would abort the sanitizer leg
/// on an input that is reachable rather than hypothetical. The guard is asserted
/// through `detail::checked_product` directly, because reaching it through a space
/// would need three directions of 2.1e6 basis functions and about 50 MB of knots.
///
/// **`check_the_tolerance_stays_a_double_at_float_storage`.** The tolerance is a
/// `double` at every storage width by design, and on a knot vector whose scale is
/// dyadic that is *unobservable*: the tolerance is `8 * eps * scale`, the factor is a
/// power of two, so a space storing it in `T` would agree bit for bit. Measured --
/// narrowing the member passed every other case here and the whole Python parity
/// suite. That case is the one where the two spellings differ, and it carries its own
/// vacuity guard saying so.
///
/// **`check_concurrent_reads`.** The header claims every accessor is safe to call
/// concurrently with no external locking. Here that should hold structurally, since
/// there is no memo -- but `design/bspline_derived_caches.md` F3 measured the shape
/// one level down giving 60 correct answers in 60 unsanitized runs and four
/// ThreadSanitizer reports, so the claim gets a gate rather than a reader's trust,
/// and the gate is a sanitizer build rather than an assertion on a value.

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "check.hpp"
#include "pantr/bspline/space_nd.hpp"

namespace {

using pantr::bspline::BsplineSpace;
using pantr::bspline::BsplineSpace1D;
using pantr::bspline::KnotSnapping;

/// Build a shared univariate space from a knot vector and a degree.
///
/// \tparam T The scalar type.
/// \param knots The knot vector.
/// \param degree The polynomial degree.
/// \return A handle on the space, snapping near-duplicates as the oracle's default
///         does.
template <class T>
std::shared_ptr<const BsplineSpace1D<T>> one_d(const std::vector<T>& knots,
                                               std::int64_t degree) {
    return std::make_shared<const BsplineSpace1D<T>>(std::span<const T>(knots), degree, false,
                                                     KnotSnapping::merge_near_duplicates);
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

/// A two-direction space whose directions differ in every reduced quantity.
///
/// Direction 0 is a clamped quadratic over three intervals on `[0, 3]`; direction 1
/// is a clamped linear over two intervals on `[10, 12]`. So the degrees differ (2
/// against 1), the basis counts differ (5 against 3), the interval counts differ (3
/// against 2), the domains differ in both location and width, and the tolerances
/// differ because the scales do (3 against 12). Nothing about this space is
/// symmetric.
void check_two_directions() {
    const std::vector<double> first{0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0};
    const std::vector<double> second{10.0, 10.0, 11.0, 12.0, 12.0};
    const auto a = one_d(first, 2);
    const auto b = one_d(second, 1);
    const BsplineSpace<double> s({a, b});

    PANTR_CHECK(s.dim() == 2);
    PANTR_CHECK(equals(s.degrees(), std::vector<std::int64_t>{2, 1}));
    PANTR_CHECK(equals(s.num_basis(), std::vector<std::int64_t>{5, 3}));
    PANTR_CHECK(equals(s.num_intervals(), std::vector<std::int64_t>{3, 2}));
    PANTR_CHECK_MSG(s.num_total_basis() == 15, "5 * 3, which is neither 5 + 3 nor either factor");
    PANTR_CHECK_MSG(s.num_total_intervals() == 6, "3 * 2, likewise");
    PANTR_CHECK(equals(s.domain_flat(), std::vector<double>{0.0, 3.0, 10.0, 12.0}));
    PANTR_CHECK_MSG(s.tolerance() == b->tolerance(),
                    "direction 1 has the larger scale, so it sets the tolerance");
    PANTR_CHECK_MSG(s.tolerance() > a->tolerance(), "and the two are genuinely different");
    PANTR_CHECK_MSG(!s.has_bezier_like_knots(), "neither direction is a single span");
}

/// A three-direction space, all three directions distinct.
///
/// The docstring example of `BsplineSpace.cell_supports` is the first two
/// directions; the third makes the totals a product of three unequal factors, which
/// a two-direction case cannot distinguish from a pairwise reduction applied twice
/// in the wrong order.
void check_three_directions() {
    const std::vector<double> first{0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0};
    const std::vector<double> second{0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    const std::vector<double> third{0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0};
    const BsplineSpace<double> s({one_d(first, 2), one_d(second, 2), one_d(third, 1)});

    PANTR_CHECK(s.dim() == 3);
    PANTR_CHECK(equals(s.degrees(), std::vector<std::int64_t>{2, 2, 1}));
    PANTR_CHECK(equals(s.num_basis(), std::vector<std::int64_t>{5, 3, 5}));
    PANTR_CHECK(equals(s.num_intervals(), std::vector<std::int64_t>{3, 1, 4}));
    PANTR_CHECK_MSG(s.num_total_basis() == 75, "5 * 3 * 5");
    PANTR_CHECK_MSG(s.num_total_intervals() == 12, "3 * 1 * 4");
    PANTR_CHECK(
        equals(s.domain_flat(), std::vector<double>{0.0, 3.0, 0.0, 1.0, 0.0, 4.0}));
}

/// One direction, which is the case an nD reduction can get right by accident.
///
/// Kept anyway, because it is what the whole `pantr.cad` layer builds for a curve
/// and because the empty and single cases are where an off-by-one in a loop bound
/// shows.
void check_one_direction() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    const auto a = one_d(knots, 2);
    const BsplineSpace<double> s({a});

    PANTR_CHECK(s.dim() == 1);
    PANTR_CHECK(equals(s.degrees(), std::vector<std::int64_t>{2}));
    PANTR_CHECK(equals(s.num_basis(), std::vector<std::int64_t>{3}));
    PANTR_CHECK(equals(s.num_intervals(), std::vector<std::int64_t>{1}));
    PANTR_CHECK(s.num_total_basis() == 3);
    PANTR_CHECK(s.num_total_intervals() == 1);
    PANTR_CHECK(s.tolerance() == a->tolerance());
    PANTR_CHECK_MSG(s.has_bezier_like_knots(), "a clamped single span at degree 2");
}

/// A space with no directions, which the oracle admits and so does this.
///
/// `tests/test_bspline_space.py::TestBsplineSpaceEdgeCases::test_empty_spaces_list`
/// pins it on the Python side. The empty products are 1, which is the empty tensor
/// product's own convention and what `numpy.prod(())` returns; `all(())` is `true`.
/// The tolerance is the one thing with no answer, so it refuses.
void check_no_directions() {
    const BsplineSpace<double> s(std::vector<std::shared_ptr<const BsplineSpace1D<double>>>{});

    PANTR_CHECK(s.dim() == 0);
    PANTR_CHECK(s.degrees().empty());
    PANTR_CHECK(s.num_basis().empty());
    PANTR_CHECK(s.num_intervals().empty());
    PANTR_CHECK(s.domain_flat().empty());
    PANTR_CHECK_MSG(s.num_total_basis() == 1, "the empty product");
    PANTR_CHECK_MSG(s.num_total_intervals() == 1, "likewise");
    PANTR_CHECK_MSG(s.has_bezier_like_knots(), "all(()) is true");

    bool refused = false;
    try {
        (void) s.tolerance();
    } catch (const std::invalid_argument& error) {
        refused = true;
        PANTR_CHECK_MSG(std::string(error.what())
                            == "tolerance: a B-spline space with no directions has no tolerance",
                        std::string("message was: ") + error.what());
    }
    PANTR_CHECK_MSG(refused, "a dimensionless space has no tolerance to report");
}

/// The tolerance is the largest of the directions', with the argmax in each place.
///
/// Two spaces built from the same three directions in different orders. If the
/// reduction returned its first argument, its last, or the smallest, exactly one of
/// the two assertions below would still pass -- which is why one case would not be
/// enough.
void check_the_tolerance_is_the_largest() {
    // Scales 1, 1000 and 0.001, so the three tolerances span six orders and the
    // largest is unambiguous at either dtype.
    const std::vector<double> unit{0.0, 0.0, 0.5, 1.0, 1.0};
    const std::vector<double> big{0.0, 0.0, 500.0, 1000.0, 1000.0};
    const std::vector<double> small{0.0, 0.0, 5e-4, 1e-3, 1e-3};
    const auto a = one_d(unit, 1);
    const auto b = one_d(big, 1);
    const auto c = one_d(small, 1);

    PANTR_CHECK_MSG(b->tolerance() > a->tolerance() && a->tolerance() > c->tolerance(),
                    "the three directions must have three different tolerances");

    const BsplineSpace<double> argmax_last({c, a, b});
    PANTR_CHECK_MSG(argmax_last.tolerance() == b->tolerance(),
                    "the largest is the last direction here");

    const BsplineSpace<double> argmax_first({b, a, c});
    PANTR_CHECK_MSG(argmax_first.tolerance() == b->tolerance(),
                    "and the first direction there, over the same three values");

    const BsplineSpace<double> argmax_middle({c, b, a});
    PANTR_CHECK_MSG(argmax_middle.tolerance() == b->tolerance(), "and the middle one there");
}

/// Bézier-likeness needs *every* direction, with the exception in each place.
///
/// Three spaces, each with exactly one non-Bézier direction, at position 0, 1 and 2.
/// A reduction written as `any`, or as "ask the first direction", passes one of
/// these and fails the other two.
void check_bezier_like_needs_every_direction() {
    const std::vector<double> single_span{0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    const std::vector<double> two_spans{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
    const auto bez = one_d(single_span, 2);
    const auto not_bez = one_d(two_spans, 2);

    PANTR_CHECK(bez->has_bezier_like_knots());
    PANTR_CHECK(!not_bez->has_bezier_like_knots());

    const BsplineSpace<double> all_bezier({bez, bez, bez});
    PANTR_CHECK_MSG(all_bezier.has_bezier_like_knots(), "every direction is a single span");

    for (int position = 0; position < 3; ++position) {
        std::vector<std::shared_ptr<const BsplineSpace1D<double>>> directions{bez, bez, bez};
        directions[static_cast<std::size_t>(position)] = not_bez;
        const BsplineSpace<double> mixed(std::move(directions));
        PANTR_CHECK_MSG(!mixed.has_bezier_like_knots(),
                        std::string("one non-Bezier direction at position ")
                            + std::to_string(position) + " must decide the whole space");
    }
}

/// Every reduced quantity differs between the directions of every case above.
///
/// A test of the test set rather than of the type: it is what stops a later case
/// from being added with two identical directions, which would make a transposition
/// or an axis-index error invisible. See the file comment.
void check_the_reductions_see_each_direction() {
    const std::vector<double> first{0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0};
    const std::vector<double> second{10.0, 10.0, 11.0, 12.0, 12.0};
    const std::vector<double> third{0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0};
    const BsplineSpace<double> s({one_d(first, 2), one_d(second, 1), one_d(third, 1)});

    // Pairwise distinct in the four per-direction sequences. The degrees are the one
    // exception and are checked as "not all equal" instead: with three directions
    // and the degrees restricted to what the other constraints allow, two of them
    // coincide, and requiring otherwise would over-constrain the case rather than
    // strengthen it.
    const auto degrees = s.degrees();
    PANTR_CHECK_MSG(!(degrees[0] == degrees[1] && degrees[1] == degrees[2]),
                    "the degrees must not all agree");
    for (std::int64_t i = 0; i < s.dim(); ++i) {
        for (std::int64_t j = i + 1; j < s.dim(); ++j) {
            const auto a = static_cast<std::size_t>(i);
            const auto b = static_cast<std::size_t>(j);
            PANTR_CHECK_MSG(s.num_intervals()[a] != s.num_intervals()[b],
                            "two directions with equal interval counts hide a transposition");
            PANTR_CHECK_MSG(s.domain_flat()[2 * a] != s.domain_flat()[2 * b]
                                || s.domain_flat()[2 * a + 1] != s.domain_flat()[2 * b + 1],
                            "two directions with equal domains hide a row swap");
        }
    }
    PANTR_CHECK_MSG(s.num_basis()[0] != s.num_basis()[1],
                    "the first two basis counts must differ");
}

/// The constructor stores the handles it is given, not copies of their pointees.
///
/// Addresses, because values would agree under a copying constructor. See the file
/// comment for what a copy would break at the Python level.
void check_the_directions_are_shared_not_copied() {
    const std::vector<double> first{0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0};
    const std::vector<double> second{0.0, 0.0, 1.0, 1.0};
    const auto a = one_d(first, 2);
    const auto b = one_d(second, 1);
    const BsplineSpace<double> s({a, b});

    PANTR_CHECK_MSG(s.space(0).get() == a.get(), "direction 0 must be the handle given");
    PANTR_CHECK_MSG(s.space(1).get() == b.get(), "direction 1 must be the handle given");
    PANTR_CHECK_MSG(&s.space_ref(0) == a.get(), "and the borrowing accessor must agree");
    PANTR_CHECK_MSG(s.spaces()[0].get() == a.get(), "and so must the whole-collection form");
    PANTR_CHECK_MSG(a.use_count() >= 2, "the space holds a reference of its own");

    // The value-taking constructor is the other half of the contract: it exists for
    // a caller holding values rather than handles, and it must NOT share.
    const std::array<BsplineSpace1D<double>, 2> values{*a, *b};
    // The span is named rather than built in the constructor call: a temporary there
    // is the most vexing parse, and `copied` becomes a function declaration.
    const std::span<const BsplineSpace1D<double>> as_values(values);
    const BsplineSpace<double> copied(as_values);
    PANTR_CHECK_MSG(copied.space(0).get() != a.get(),
                    "the value-taking constructor owns its own directions");
    PANTR_CHECK(equals(copied.num_basis(), std::vector<std::int64_t>{4, 2}));
    PANTR_CHECK(copied.tolerance() == s.tolerance());
}

/// A direction taken out of a space outlives it.
///
/// The point of storing `shared_ptr<const T>` rather than annotating the binding.
/// Asserted on a value the destroyed space's own storage would have been reused for
/// -- a fresh space is built in the freed region first -- because
/// `design/bspline_ownership_lifetime.md` F4 measured a scalar read after free
/// returning the correct value, which is exactly the shape of test that passes on a
/// broken design.
void check_a_direction_outlives_the_space() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0};
    std::shared_ptr<const BsplineSpace1D<double>> escapee;
    {
        const std::vector<double> other{0.0, 0.0, 1.0, 1.0};
        const BsplineSpace<double> s({one_d(knots, 2), one_d(other, 1)});
        escapee = s.space(0);
    }
    // Churn the freed storage, so a dangling read is unlikely to find its old bytes.
    for (int i = 0; i < 64; ++i) {
        const std::vector<double> filler{0.0, 0.0, 1.0, 2.0, 3.0, 3.0};
        const BsplineSpace<double> scratch({one_d(filler, 1)});
        PANTR_CHECK(scratch.num_total_basis() == 4);
    }
    PANTR_CHECK_MSG(escapee->num_basis() == 5, "the escaped direction still knows its own state");
    PANTR_CHECK(escapee->degree() == 2);
    PANTR_CHECK(escapee->domain()[1] == 3.0);
}

/// A copy or a move of a space carries the same directions, and the same values.
///
/// There is no memo here to go cold -- `design/bspline_derived_caches.md` F1 -- so
/// unlike `BsplineSpace1D` a copy is a plain value copy, and what is worth asserting
/// is that the *sharing* survives it: a copied space points at the same directions.
void check_copy_and_move() {
    const std::vector<double> first{0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0};
    const std::vector<double> second{5.0, 5.0, 6.0, 7.0, 7.0};
    const auto a = one_d(first, 2);
    const auto b = one_d(second, 1);
    BsplineSpace<double> original({a, b});

    const BsplineSpace<double> copied(original);
    PANTR_CHECK(copied.dim() == 2);
    PANTR_CHECK(equals(copied.num_basis(), std::vector<std::int64_t>{4, 3}));
    PANTR_CHECK(copied.num_total_basis() == 12);
    PANTR_CHECK(copied.tolerance() == original.tolerance());
    PANTR_CHECK_MSG(copied.space(0).get() == a.get(), "a copy shares the same directions");

    const BsplineSpace<double> moved(std::move(original));
    PANTR_CHECK(moved.dim() == 2);
    PANTR_CHECK(equals(moved.num_intervals(), std::vector<std::int64_t>{2, 2}));
    PANTR_CHECK(moved.space(1).get() == b.get());
}

/// `float` storage, so the template is instantiated at both widths.
///
/// The domain values come out in the storage format and the tolerance stays a
/// `double`, which is `pantr/bspline/knots.hpp`'s rule rather than this type's.
void check_float_storage() {
    const std::vector<float> first{0.0F, 0.0F, 0.0F, 1.0F, 2.0F, 2.0F, 2.0F};
    const std::vector<float> second{0.0F, 0.0F, 1.0F, 1.0F};
    const auto a = one_d(first, 2);
    const auto b = one_d(second, 1);
    const BsplineSpace<float> s({a, b});

    PANTR_CHECK(equals(s.domain_flat(), std::vector<float>{0.0F, 2.0F, 0.0F, 1.0F}));
    PANTR_CHECK(equals(s.num_basis(), std::vector<std::int64_t>{4, 2}));
    PANTR_CHECK(s.num_total_basis() == 8);
    PANTR_CHECK_MSG(s.tolerance() == a->tolerance(),
                    "the wider-scaled direction sets it, and it is a double either way");
    PANTR_CHECK_MSG(s.tolerance() > b->tolerance(), "the two directions differ here too");
}

/// The tolerance stays a `double` at `float` storage, which is not free to observe.
///
/// `pantr/bspline/knots.hpp` records that the tolerance is a `double` at every
/// storage width, deliberately. That is invisible on almost every knot vector: the
/// tolerance is `8 * eps * scale`, the factor is a power of two, so wherever the
/// winning scale is exactly representable in `float` the tolerance is too, and a
/// space storing it in `T` would agree bit for bit. Measured: narrowing the member to
/// `T` passed every other case in this file and the whole Python parity suite.
///
/// This is the case where the two spellings differ. The scale is `hi - lo` across
/// zero, so it is a *sum* of two `float`s and needs one bit more than `float` has --
/// and the vacuity guard below is what says so, rather than leaving it to be believed.
void check_the_tolerance_stays_a_double_at_float_storage() {
    // From -1e-7f to 1.0f, so the scale is the span rather than either coordinate.
    const std::vector<float> straddling{-1e-7F, -1e-7F, -1e-7F, 0.5F, 1.0F, 1.0F, 1.0F};
    // Scale exactly 1, so this direction's own tolerance IS a float value and it
    // loses the reduction by one part in 1e8.
    const std::vector<float> unit{0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};
    const auto wide = one_d(straddling, 2);
    const auto narrow = one_d(unit, 2);

    PANTR_CHECK_MSG(static_cast<double>(static_cast<float>(wide->tolerance()))
                        != wide->tolerance(),
                    "the vacuity guard: this direction's tolerance must not be a float "
                    "value, or narrowing the space's member would be undetectable");
    PANTR_CHECK_MSG(static_cast<double>(static_cast<float>(narrow->tolerance()))
                        == narrow->tolerance(),
                    "and the other one's must be, so the two are told apart by width "
                    "rather than by magnitude alone");
    PANTR_CHECK_MSG(wide->tolerance() > narrow->tolerance(),
                    "the non-representable one must win the reduction");

    for (const auto& order : {std::vector{wide, narrow}, std::vector{narrow, wide}}) {
        const BsplineSpace<float> s(order);
        PANTR_CHECK_MSG(s.tolerance() == wide->tolerance(),
                        "the space must carry the double, not its float rounding");
    }
}

/// A null handle and an out-of-range direction are both refused.
void check_refusals() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    const auto a = one_d(knots, 2);

    bool refused_null = false;
    try {
        const BsplineSpace<double> s({a, nullptr});
        (void) s.dim();
    } catch (const std::invalid_argument& error) {
        refused_null = true;
        PANTR_CHECK_MSG(std::string(error.what()) == "direction 1 is a null B-spline space",
                        std::string("message was: ") + error.what());
    }
    PANTR_CHECK_MSG(refused_null, "a null direction cannot be aggregated");

    const BsplineSpace<double> s({a});
    for (const std::int64_t bad : {-1L, 1L, 7L}) {
        bool refused = false;
        try {
            (void) s.space(bad);
        } catch (const std::out_of_range&) {
            refused = true;
        }
        PANTR_CHECK_MSG(refused, "a direction outside [0, dim) must be refused");

        refused = false;
        try {
            (void) s.space_ref(bad);
        } catch (const std::out_of_range&) {
            refused = true;
        }
        PANTR_CHECK_MSG(refused, "and the borrowing accessor must refuse it too");
    }
}

/// A tensor-product total that would not fit in an `int64` is refused, not wrapped.
///
/// Asserted through the helper rather than through a space; see the file comment for
/// why, and for why wrapping is not an option here even though the oracle wraps.
void check_the_totals_refuse_an_overflow() {
    using pantr::bspline::detail::checked_product;
    constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();

    const std::vector<std::int64_t> empty{};
    PANTR_CHECK_MSG(checked_product(empty, "x") == 1, "the empty product is 1");

    const std::vector<std::int64_t> ordinary{5, 3, 7};
    PANTR_CHECK(checked_product(ordinary, "x") == 105);

    // A zero short-circuits, so a zero beside a value that would otherwise overflow
    // is still zero rather than a refusal. That matters: a direction cannot have
    // zero basis functions today, and the branch is what keeps the guard from
    // depending on that staying true.
    const std::vector<std::int64_t> with_zero{kMax, 0, kMax};
    PANTR_CHECK_MSG(checked_product(with_zero, "x") == 0, "a zero factor short-circuits");

    // Either side of the boundary, on the tightest pair available. `root` is
    // `floor(sqrt(kMax))`, so `root * root` fits and so does `root * (root + 1)` --
    // checked, and the reason this pair is spelled out rather than guessed: an
    // earlier version used `root + 1` as the overflowing factor and it does not
    // overflow. `root + 2` is the first that does.
    const std::int64_t root = 3037000499;
    const std::vector<std::int64_t> fits{root, root + 1};
    PANTR_CHECK_MSG(checked_product(fits, "x") == root * (root + 1), "this one still fits");

    const std::vector<std::int64_t> just_over{root, root + 2};
    bool refused = false;
    try {
        (void) checked_product(just_over, "num_total_basis");
    } catch (const pantr::CapacityError& error) {
        refused = true;
        PANTR_CHECK_MSG(
            std::string(error.what()) == "num_total_basis exceeds the range of a 64-bit integer",
            std::string("message was: ") + error.what());
    }
    PANTR_CHECK_MSG(refused, "a product past the int64 range must refuse rather than wrap");

    const std::vector<std::int64_t> three_way{2097152, 2097152, 2097152};
    refused = false;
    try {
        (void) checked_product(three_way, "num_total_basis");
    } catch (const pantr::CapacityError&) {
        refused = true;
    }
    PANTR_CHECK_MSG(refused, "and three directions of 2^21 are what makes it reachable");
}

/// Every accessor is safe to call concurrently on one space, with no locking.
///
/// `space_nd.hpp` claims that, and the claim needs a gate rather than a reader's
/// trust. Here it should hold for a structural reason -- there is no memo, so every
/// accessor reads state frozen at construction -- but "should" is exactly what
/// `design/bspline_derived_caches.md` F3 refutes for the shape one level down: a
/// bare `mutable std::optional` memo produced **60 correct answers in 60
/// unsanitized runs** and four ThreadSanitizer reports, so no assertion on a value
/// can stand in for the sanitizer.
///
/// `space(d)` is included deliberately and is the only accessor here with any
/// machinery behind it: it copies a `shared_ptr`, so eight threads hammering it
/// contend on one atomic control block. The rest are span and scalar reads.
///
/// The threads are released together off an atomic flag rather than started in
/// sequence, so that they genuinely overlap; a run in which each finished before the
/// next began would report clean for the wrong reason.
void check_concurrent_reads() {
    const std::vector<double> first{0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0};
    const std::vector<double> second{10.0, 10.0, 11.0, 12.0, 12.0};
    const BsplineSpace<double> s({one_d(first, 2), one_d(second, 1)});

    constexpr int num_threads = 8;
    std::atomic<bool> go{false};
    std::vector<std::int64_t> totals(num_threads, 0);
    std::vector<double> tolerances(num_threads, 0.0);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&s, &go, &totals, &tolerances, t] {
            while (!go.load(std::memory_order_acquire)) {
                // Spin until every thread is up, so the reads overlap.
            }
            std::int64_t sum = 0;
            double tol = 0.0;
            for (int i = 0; i < 2000; ++i) {
                for (std::int64_t d = 0; d < s.dim(); ++d) {
                    sum += s.degrees()[static_cast<std::size_t>(d)];
                    sum += s.num_basis()[static_cast<std::size_t>(d)];
                    sum += s.num_intervals()[static_cast<std::size_t>(d)];
                    sum += static_cast<std::int64_t>(s.domain_flat()[static_cast<std::size_t>(
                        2 * d)]);
                    // The one accessor with an atomic in it.
                    sum += s.space(d)->num_basis();
                    sum += s.space_ref(d).degree();
                }
                sum += s.num_total_basis() + s.num_total_intervals();
                sum += s.has_bezier_like_knots() ? 1 : 0;
                tol = s.tolerance();
            }
            totals[static_cast<std::size_t>(t)] = sum;
            tolerances[static_cast<std::size_t>(t)] = tol;
        });
    }
    go.store(true, std::memory_order_release);
    for (std::thread& thread : threads) {
        thread.join();
    }

    for (int t = 0; t < num_threads; ++t) {
        const auto index = static_cast<std::size_t>(t);
        PANTR_CHECK_MSG(totals[index] == totals[0], "every thread must read one state");
        PANTR_CHECK_MSG(tolerances[index] == s.tolerance(),
                        "and one tolerance, exactly the single-threaded one");
    }
    PANTR_CHECK_MSG(s.space(0).use_count() >= 2,
                    "the vacuity guard: the shared handle survived the contention, so "
                    "the atomic path really was exercised");
}

}  // namespace

int main() {
    check_two_directions();
    check_three_directions();
    check_one_direction();
    check_no_directions();
    check_the_tolerance_is_the_largest();
    check_bezier_like_needs_every_direction();
    check_the_reductions_see_each_direction();
    check_the_directions_are_shared_not_copied();
    check_a_direction_outlives_the_space();
    check_copy_and_move();
    check_float_storage();
    check_the_tolerance_stays_a_double_at_float_storage();
    check_refusals();
    check_the_totals_refuse_an_overflow();
    check_concurrent_reads();
    return pantr::test::summary("test_bspline_space_nd");
}
