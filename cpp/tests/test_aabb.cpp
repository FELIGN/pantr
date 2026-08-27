/// \file
/// Properties of the axis-aligned bounding box.
///
/// ## Why these cases and not others
///
/// The arithmetic here is min, max and comparison, so there is no forward-error
/// budget to bound and no tolerance to derive: every assertion below is exact,
/// and any tolerance would be hiding something rather than allowing for it. The
/// one place arithmetic happens is `transform`, and the cases chosen there are
/// the ones where IEEE special values decide the answer rather than the
/// magnitudes.
///
/// What each group would catch:
///
///  - **Validation.** The type is the C++ counterpart of Layer 2, so a caller
///    with no Python is protected by these throws and nothing else. A missing
///    one is invisible until a consumer hands the box a NaN.
///  - **Emptiness.** Empty boxes are the corner every method has to answer for,
///    and the answers differ per method: `contains_point` falls out of the
///    comparison, `overlaps` and `intersect` short-circuit, `merge` treats it as
///    neutral, `transform` maps it to a canonical empty.
///  - **Signed zero.** `operator==` says `-0.0 == 0.0`; a hash over raw values
///    would disagree, and every unordered container would then lose boxes. The
///    two are only consistent because the hash normalizes, and nothing but a
///    test says so.
///  - **Zero times infinity.** `transform` must let a zero matrix entry
///    contribute nothing against an infinite bound. Multiplying first gives NaN
///    and poisons an axis the map projects out -- a wrong answer that looks like
///    an arithmetic accident rather than a missing branch.

#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/geometry/aabb.hpp"

namespace {

using pantr::geometry::AABB;

constexpr double kInf = std::numeric_limits<double>::infinity();
const double kNaN = std::numeric_limits<double>::quiet_NaN();

/// Build a box from two brace lists, for brevity below.
///
/// \param lo Lower corner.
/// \param hi Upper corner.
/// \return The box.
AABB<double> box(std::vector<double> lo, std::vector<double> hi) {
    return AABB<double>(std::span<const double>(lo), std::span<const double>(hi));
}

/// Whether calling `fn` throws `std::invalid_argument`.
///
/// \param fn The call to attempt.
/// \return `true` when it threw.
template <class F>
bool throws_invalid(F&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

void test_construction_validates() {
    PANTR_CHECK_MSG(throws_invalid([] { (void)box({0.0, 1.0}, {2.0}); }), "mismatched corner lengths");
    PANTR_CHECK_MSG(throws_invalid([] { (void)box({}, {}); }), "ndim must be >= 1");
    PANTR_CHECK_MSG(throws_invalid([] { (void)box({kNaN}, {1.0}); }), "NaN in lo");
    PANTR_CHECK_MSG(throws_invalid([] { (void)box({0.0}, {kNaN}); }), "NaN in hi");
    // Infinities are bounds, not errors: an unbounded axis is the whole point.
    PANTR_CHECK(box({-kInf}, {kInf}).ndim() == 1);
}

void test_emptiness() {
    PANTR_CHECK(AABB<double>::empty(3).is_empty());
    PANTR_CHECK(!AABB<double>::unbounded(3).is_empty());
    // Empty on one axis is enough, even when every other axis is well ordered.
    PANTR_CHECK(box({0.0, 5.0}, {1.0, 3.0}).is_empty());
    PANTR_CHECK(!box({0.0, 0.0}, {1.0, 1.0}).is_empty());
    // A degenerate box -- lo == hi -- is NOT empty; it contains exactly one point.
    PANTR_CHECK(!box({1.0}, {1.0}).is_empty());
    PANTR_CHECK(box({1.0}, {1.0}).contains_point(std::vector<double>{1.0}));
}

void test_contains_point_is_inclusive() {
    const auto b = box({0.0, 0.0}, {1.0, 1.0});
    PANTR_CHECK_MSG(b.contains_point(std::vector<double>{0.0, 0.0}), "the lo corner is inside");
    PANTR_CHECK_MSG(b.contains_point(std::vector<double>{1.0, 1.0}), "the hi corner is inside");
    const double just_over = std::nextafter(1.0, 2.0);
    PANTR_CHECK_MSG(!b.contains_point(std::vector<double>{just_over, 0.5}),
                    "one ulp past the face is outside");
    PANTR_CHECK_MSG(!AABB<double>::empty(2).contains_point(std::vector<double>{0.0, 0.0}),
                    "an empty box contains nothing");
    PANTR_CHECK(throws_invalid([&b] { (void)b.contains_point(std::vector<double>{0.0}); }));
    PANTR_CHECK(throws_invalid([&b] { (void)b.contains_point(std::vector<double>{kNaN, 0.0}); }));
}

void test_overlaps_is_inclusive_and_empty_overlaps_nothing() {
    // Face contact counts as an overlap. This is the tie contract, and it is the
    // one property of this predicate that is not obviously either way.
    PANTR_CHECK(box({0.0}, {1.0}).overlaps(box({1.0}, {2.0})));
    PANTR_CHECK(!box({0.0}, {1.0}).overlaps(box({std::nextafter(1.0, 2.0)}, {2.0})));

    // An empty box overlaps nothing, INCLUDING a box that contains its reversed
    // interval. This is exactly where pantr/grid/bvh.hpp's predicate disagrees:
    // it tests separating axes only and reports an overlap here. The
    // disagreement is recorded in the aabb.hpp file comment and is deliberately
    // not reconciled; this assertion pins which side this type is on, so that a
    // later reconciliation has to change a test rather than pass silently.
    const auto empty_interval = box({5.0}, {3.0});
    PANTR_CHECK(empty_interval.is_empty());
    PANTR_CHECK_MSG(!empty_interval.overlaps(box({0.0}, {10.0})),
                    "an empty box overlaps nothing, even inside a containing box");
    PANTR_CHECK(!AABB<double>::unbounded(1).overlaps(AABB<double>::empty(1)));
    PANTR_CHECK(throws_invalid([] { (void)AABB<double>::empty(1).overlaps(AABB<double>::empty(2)); }));
}

void test_merge_treats_empty_as_neutral() {
    const auto b = box({0.0, 0.0}, {1.0, 2.0});
    PANTR_CHECK(AABB<double>::empty(2).merge(b) == b);
    PANTR_CHECK(b.merge(AABB<double>::empty(2)) == b);
    PANTR_CHECK(b.merge(box({-1.0, 1.0}, {0.5, 3.0})) == box({-1.0, 0.0}, {1.0, 3.0}));

    // Equality is by value, not by geometry: two boxes that both contain no
    // point are still different boxes when their corners differ. Merging one
    // empty into another therefore returns an operand unchanged rather than a
    // canonical empty, and this pins which one.
    const auto other_empty = box({5.0}, {3.0});
    PANTR_CHECK(!(other_empty == AABB<double>::empty(1)));
    PANTR_CHECK(AABB<double>::empty(1).merge(other_empty) == other_empty);
}

void test_intersect() {
    const auto a = box({0.0, 0.0}, {2.0, 2.0});
    const auto b = box({1.0, 1.0}, {3.0, 3.0});
    const auto hit = a.intersect(b);
    PANTR_CHECK(hit.has_value());
    PANTR_CHECK(*hit == box({1.0, 1.0}, {2.0, 2.0}));

    // Face contact intersects in a degenerate, non-empty box -- consistent with
    // overlaps() reporting the same pair as overlapping.
    const auto touching = box({0.0}, {1.0}).intersect(box({1.0}, {2.0}));
    PANTR_CHECK(touching.has_value());
    PANTR_CHECK(!touching->is_empty());

    PANTR_CHECK(!a.intersect(box({3.0, 3.0}, {4.0, 4.0})).has_value());
    PANTR_CHECK(!a.intersect(AABB<double>::empty(2)).has_value());
}

void test_pad() {
    PANTR_CHECK(box({0.0}, {1.0}).pad(0.5) == box({-0.5}, {1.5}));
    // A negative radius may empty the box, which is allowed rather than an error.
    PANTR_CHECK(box({0.0}, {1.0}).pad(-1.0).is_empty());
    // An infinite bound stays infinite: inf + finite == inf, no special case needed.
    const auto padded = AABB<double>::unbounded(1).pad(1.0);
    PANTR_CHECK(std::isinf(padded.lo()[0]) && padded.lo()[0] < 0.0);
    PANTR_CHECK(std::isinf(padded.hi()[0]) && padded.hi()[0] > 0.0);
    PANTR_CHECK(throws_invalid([] { (void)box({0.0}, {1.0}).pad(kInf); }));
    PANTR_CHECK(throws_invalid([] { (void)box({0.0}, {1.0}).pad(kNaN); }));
}

void test_transform_zero_entry_beats_an_infinite_bound() {
    // Axis 1 is unbounded, and the map projects it out. Multiplying first would
    // give 0 * inf == NaN and poison output axis 0; masking the zero entry keeps
    // the wrap finite and correct.
    const std::vector<double> a{1.0, 0.0, 0.0, 1.0};
    const std::vector<double> b{0.0, 0.0};
    const auto matrix = pantr::span2d<const double>(a.data(), 2, 2);
    const auto src = box({0.0, -kInf}, {2.0, kInf});
    const auto out = src.transform(matrix, std::span<const double>(b));
    PANTR_CHECK_MSG(out.lo()[0] == 0.0 && out.hi()[0] == 2.0, "the bounded axis is exact");
    PANTR_CHECK(std::isinf(out.lo()[1]) && std::isinf(out.hi()[1]));

    // A rotation by a right angle swaps the axes exactly; no tolerance needed
    // because every entry is 0 or +/-1 and the products are exact.
    const std::vector<double> rot{0.0, -1.0, 1.0, 0.0};
    const auto rotated = box({0.0, 0.0}, {1.0, 2.0})
                             .transform(pantr::span2d<const double>(rot.data(), 2, 2),
                                        std::span<const double>(b));
    PANTR_CHECK(rotated == box({-2.0, 0.0}, {0.0, 1.0}));

    // An empty box maps to the canonical empty regardless of the map.
    PANTR_CHECK(AABB<double>::empty(2)
                    .transform(matrix, std::span<const double>(b))
                    .is_empty());
}

void test_transform_rejects_opposing_infinities() {
    // Row 0 receives -inf from axis 0 and +inf from axis 1, so its lower
    // accumulator computes (-inf) + (+inf) == NaN. That is not a box, and it is
    // reported rather than returned.
    //
    // Getting here takes both axes degenerate AT an infinity, not merely
    // unbounded: an axis with lo = hi = -inf contributes -inf to both
    // accumulators, and a finite axis contributes a finite amount, so neither on
    // its own can produce the cancellation.
    const std::vector<double> a{1.0, 1.0, 0.0, 1.0};
    const std::vector<double> b{0.0, 0.0};
    const auto matrix = pantr::span2d<const double>(a.data(), 2, 2);
    const auto src = box({-kInf, kInf}, {-kInf, kInf});
    PANTR_CHECK_MSG(!src.is_empty(), "the source box is degenerate but not empty");
    PANTR_CHECK(throws_invalid(
        [&] { (void)src.transform(matrix, std::span<const double>(b)); }));
}

void test_bounds_round_trip() {
    // from_bounds and as_bounds are duals, so the round trip is the test: it
    // would catch a transposed index, which is the one mistake an (ndim, 2)
    // table invites and which no shape check can see.
    const auto b = box({0.0, -1.0, 2.0}, {1.0, 3.0, 2.0});
    std::vector<double> table(3 * 2);
    b.as_bounds(pantr::span2d<double>(table.data(), 3, 2));
    PANTR_CHECK(table[0] == 0.0 && table[1] == 1.0);
    PANTR_CHECK(table[2] == -1.0 && table[3] == 3.0);
    PANTR_CHECK(AABB<double>::from_bounds(pantr::span2d<const double>(table.data(), 3, 2)) == b);

    // A (2, ndim) table is a plausible transposition and must be refused rather
    // than silently read as a 2-dimensional box.
    PANTR_CHECK(throws_invalid([&table] {
        (void)AABB<double>::from_bounds(pantr::span2d<const double>(table.data(), 2, 3));
    }));
    PANTR_CHECK(throws_invalid([&b, &table] {
        b.as_bounds(pantr::span2d<double>(table.data(), 2, 3));
    }));
}

void test_to_string_names_both_corners() {
    // Not a formatting test: it pins that both corners reach the string, which is
    // what makes a repr useful in a failing assertion.
    const auto text = box({0.0}, {1.0}).to_string();
    PANTR_CHECK(text.find("lo=[") != std::string::npos);
    PANTR_CHECK(text.find("hi=[") != std::string::npos);
}

void test_equality_and_hash_agree_on_signed_zero() {
    const auto neg = box({-0.0}, {1.0});
    const auto pos = box({0.0}, {1.0});
    PANTR_CHECK_MSG(neg == pos, "-0.0 and +0.0 compare equal");
    PANTR_CHECK_MSG(std::hash<AABB<double>>{}(neg) == std::hash<AABB<double>>{}(pos),
                    "and therefore must hash equal");
    // The stored sign is not normalized away; only the hash is.
    PANTR_CHECK(std::signbit(neg.lo()[0]));
    PANTR_CHECK(!std::signbit(pos.lo()[0]));
    // Different dimensions are never equal, and the ndim seed keeps them apart.
    PANTR_CHECK(!(AABB<double>::empty(1) == AABB<double>::empty(2)));
}

}  // namespace

int main() {
    test_construction_validates();
    test_emptiness();
    test_contains_point_is_inclusive();
    test_overlaps_is_inclusive_and_empty_overlaps_nothing();
    test_merge_treats_empty_as_neutral();
    test_intersect();
    test_pad();
    test_transform_zero_entry_beats_an_infinite_bound();
    test_transform_rejects_opposing_infinities();
    test_bounds_round_trip();
    test_to_string_names_both_corners();
    test_equality_and_hash_agree_on_signed_zero();
    return pantr::test::summary("test_aabb");
}
