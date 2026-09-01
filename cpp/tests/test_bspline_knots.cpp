/// \file
/// The knot vector's tolerance, its classes, and the refusals a space owes.
///
/// ## Where the expected values come from
///
/// Three different provenances, kept apart on purpose because they carry
/// different weight.
///
///  - **Derived here.** The tolerance is asserted against `8 * eps * scale`
///    recomputed in the test from `std::numeric_limits`, not against a literal
///    copied from a run. A literal would pass if both sides drifted together.
///  - **Hand-checkable structure.** The class counts, multiplicities and interval
///    counts are read off the knot vectors by eye; each case says what it is for.
///  - **The oracle's text, verbatim.** The five messages are what
///    `pantr.bspline.BsplineSpace1D` prints for the same arguments, captured by
///    running it. They are asserted character for character because they are also
///    the messages a Python caller sees under either backend, and a reworded one
///    here is a parity failure that an assertion on the exception *type* would
///    not notice.
///
/// Nothing here has a numerical tolerance of its own. Every quantity is a count,
/// an index, a knot value reproduced exactly, or a string.
///
/// ## The two cases that would not be written without having got them wrong
///
/// **Chaining.** Classes are grouped by gap, so `n` knots each within `tol` of the
/// next collapse into one class however far apart the ends are. A vector whose
/// four leading knots step by `0.56 * tol` is one class even though its ends are
/// `1.7 * tol` apart, and the test says so.
///
/// **Snapping decides its boundaries on the original vector.** Rewriting knots in
/// place while scanning changes the predicate for every later step: the same
/// four-knot chain splits into two classes if `knots[i-1]` has already been moved
/// onto its representative. That is a silent wrong answer -- a different interval
/// count on a legal space -- and `check_snap_reads_the_original_vector` is the only
/// thing in the suite that distinguishes the two implementations.

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/bspline/knots.hpp"

namespace {

using pantr::bspline::check_snapping_kept_an_interval;
using pantr::bspline::check_space_has_an_interval;
using pantr::bspline::classify_knots;
using pantr::bspline::knot_scale;
using pantr::bspline::knot_tolerance;
using pantr::bspline::KnotClassRange;
using pantr::bspline::num_basis;
using pantr::bspline::snap_knots;
using pantr::bspline::unique_knots_and_multiplicity;

/// A span over a vector, for brevity below.
template <class T>
std::span<const T> view(const std::vector<T>& v) {
    return std::span<const T>(v);
}

/// The message of the `std::invalid_argument` that `fn` throws.
///
/// Returns a marker rather than asserting, so the caller's own check reports and a
/// failure names the case it came from.
template <class F>
std::string message_of(F&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument& e) {
        return e.what();
    } catch (...) {
        return "<threw something else>";
    }
    return "<did not throw>";
}

/// Check a string against the oracle's, reporting both on failure.
void same(const std::string& actual, const std::string& expected, const char* what) {
    PANTR_CHECK_MSG(actual == expected,
                    std::string(what) + ":\n  got  '" + actual + "'\n  want '" + expected + "'");
}

/// The scale is the largest of the span and the two coordinate magnitudes.
///
/// The offset cases are the point: on a knot vector based at 1e6 a knot difference
/// is formed from two coordinates of that size, so the span alone would understate
/// the magnitude a comparison is relative to by six orders.
void check_knot_scale() {
    PANTR_CHECK(knot_scale<double>(view(std::vector<double>{0.0, 1.0})) == 1.0);
    PANTR_CHECK(knot_scale<double>(view(std::vector<double>{1e6, 1e6 + 1.0})) == 1e6 + 1.0);
    PANTR_CHECK(knot_scale<double>(view(std::vector<double>{-5.0, -1.0})) == 5.0);
    PANTR_CHECK(knot_scale<double>(view(std::vector<double>{-2.0, 3.0})) == 5.0);
    // A domain smaller than one unit keeps its own scale: no floor is applied, so
    // the tolerance stays covariant under a change of parametric unit.
    PANTR_CHECK(knot_scale<double>(view(std::vector<double>{0.0, 1e-6})) == 1e-6);
    PANTR_CHECK(knot_scale<double>(view(std::vector<double>{0.0, 0.0})) == 0.0);
}

/// The tolerance is `8 * eps(T) * scale`, at both widths.
///
/// Recomputed from `numeric_limits` rather than compared against a literal: the
/// factor is what is being asserted, and a literal would agree with a wrong factor
/// as readily as with the right one. The exactness claim is separate and is checked
/// too -- 8 and eps are both powers of two, so the product is exact and the only
/// rounding is the final multiply, which is what lets the two backends agree on the
/// threshold to the last bit rather than merely to within it.
void check_knot_tolerance() {
    constexpr double eps64 = std::numeric_limits<double>::epsilon();
    constexpr double eps32 = static_cast<double>(std::numeric_limits<float>::epsilon());

    const std::vector<double> unit{0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0};
    PANTR_CHECK(knot_tolerance(view(unit)) == 8.0 * eps64 * 2.0);

    const std::vector<float> unit32{0.0F, 0.0F, 0.0F, 0.25F, 0.7F, 0.7F, 1.0F, 1.0F, 1.0F};
    PANTR_CHECK(knot_tolerance(view(unit32)) == 8.0 * eps32 * 1.0);

    // Scale covariance: doubling every knot doubles the tolerance exactly, because
    // the scale is linear and the multiply by a power of two is exact.
    const std::vector<double> doubled{0.0, 0.0, 0.0, 2.0, 4.0, 4.0, 4.0};
    PANTR_CHECK(knot_tolerance(view(doubled)) == 2.0 * knot_tolerance(view(unit)));
}

/// Classes, multiplicities, and the in-domain range.
void check_classes() {
    // Clamped, degree 2, two unit intervals. Read off the vector: three classes
    // with multiplicities 3, 1, 3, all of them in the domain.
    const std::vector<double> clamped{0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0};
    const double tol = knot_tolerance(view(clamped));
    const KnotClassRange range = classify_knots(view(clamped), 2, tol);
    PANTR_CHECK(range.count == 3);
    PANTR_CHECK(range.domain_begin == 0);
    PANTR_CHECK(range.domain_end == 3);
    PANTR_CHECK(range.num_intervals() == 2);

    const auto classes = unique_knots_and_multiplicity(view(clamped), 2, tol);
    PANTR_CHECK(classes.unique == std::vector<double>({0.0, 1.0, 2.0}));
    PANTR_CHECK(classes.multiplicity == std::vector<std::int64_t>({3, 1, 3}));
    PANTR_CHECK(classes.domain_begin == 0);
    PANTR_CHECK(classes.domain_end == 3);

    // The multiplicities sum to the whole vector, which is what makes
    // `repeat(unique, multiplicity)` reproduce it.
    std::int64_t total = 0;
    for (const std::int64_t m : classes.multiplicity) {
        total += m;
    }
    PANTR_CHECK(total == static_cast<std::int64_t>(clamped.size()));

    // Unclamped, degree 3: the domain is the interior, so the in-domain range is a
    // strict subrange and the boundary classes outside it are still counted.
    const std::vector<double> open{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0};
    const KnotClassRange interior = classify_knots(view(open), 3, knot_tolerance(view(open)));
    PANTR_CHECK(interior.count == 9);
    PANTR_CHECK(interior.domain_begin == 3);
    PANTR_CHECK(interior.domain_end == 6);
    PANTR_CHECK(interior.num_intervals() == 2);
}

/// Knots chain into one class through the steps between them.
///
/// Four knots each `0.56 * tol` from the next are one class, although the first and
/// the last are `1.7 * tol` apart. Grouping by gap is what makes the answer depend
/// on the knots rather than on where they sit relative to an origin, and chaining
/// is the accepted cost of it: capping a class's width would break idempotence,
/// since a class split by the cap leaves two representatives within `tol` of each
/// other that would merge on the next pass.
void check_chaining() {
    const std::vector<double> chained{0.0, 1e-15, 2e-15, 3e-15, 0.5, 1.0};
    const double tol = knot_tolerance(view(chained));
    PANTR_CHECK_MSG(1e-15 < tol, "the step must be inside the tolerance for this to be the case");
    PANTR_CHECK_MSG(3e-15 > tol, "and the ends outside it, or nothing is being shown");

    const auto classes = unique_knots_and_multiplicity(view(chained), 1, tol);
    PANTR_CHECK(classes.unique == std::vector<double>({0.0, 0.5, 1.0}));
    PANTR_CHECK(classes.multiplicity == std::vector<std::int64_t>({4, 1, 1}));
}

/// A step of exactly the tolerance is inside it, not outside.
///
/// The predicate is "the same knot unless they differ by MORE than `tol`", so the
/// boundary is closed on the merging side. Nothing else in this file pins it: every
/// other case has steps that are orders of magnitude clear of the threshold, so
/// flipping `>` to `>=` left the whole suite green when it was tried. The case has
/// to be constructed, because a step lands exactly on the threshold only if it is
/// built from it.
///
/// The oracle agrees: `_get_unique_knots_and_multiplicity_impl` on this vector
/// reports classes `[0, 0.25, 0.5, 0.75, 1]` with multiplicities `[2, 1, 1, 1, 1]`.
void check_the_boundary_step_merges() {
    constexpr double eps = std::numeric_limits<double>::epsilon();
    const std::vector<double> knots{0.0, 8.0 * eps, 0.25, 0.5, 0.75, 1.0};
    const double tol = knot_tolerance(view(knots));
    PANTR_CHECK_MSG(tol == 8.0 * eps, "the vector's scale must be exactly 1 for this to be exact");
    PANTR_CHECK_MSG(knots[1] - knots[0] == tol, "the step must be exactly the tolerance");

    const auto classes = unique_knots_and_multiplicity(view(knots), 1, tol);
    PANTR_CHECK_MSG(classes.multiplicity == std::vector<std::int64_t>({2, 1, 1, 1, 1}),
                    "a step of exactly the tolerance is a merge, not a split");
    PANTR_CHECK(classes.unique == std::vector<double>({0.0, 0.25, 0.5, 0.75, 1.0}));
}

/// Snapping takes the class's FIRST knot, and is idempotent.
///
/// Chosen rather than averaged, so the values returned are knots the vector
/// actually contains. An average would be a fresh rounding, and over `degree + 1`
/// identical copies at large magnitude it is not even the identity, so it moved the
/// reported domain.
void check_snap_takes_the_first_knot() {
    const std::vector<double> raw{0.0, 0.0, 0.0, 0.5, 0.5 + 2e-16, 1.0, 1.0, 1.0};
    const double tol = knot_tolerance(view(raw));
    PANTR_CHECK_MSG(raw[3] != raw[4], "the pair must really be distinct before snapping");

    const std::vector<double> snapped = snap_knots(view(raw), tol);
    PANTR_CHECK(snapped == std::vector<double>({0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0}));
    PANTR_CHECK_MSG(snapped[4] == raw[3], "the representative is the class's first knot");

    const std::vector<double> again = snap_knots(view(snapped), tol);
    PANTR_CHECK_MSG(again == snapped, "snapping is idempotent");
}

/// Snapping reads the original vector, never the one it is writing.
///
/// This is the whole content of the test. Rewriting in place would compare
/// `knots[i]` against a `knots[i-1]` already moved onto its representative, which
/// splits the chain above into two classes and yields a *legal space with a
/// different interval count* -- a wrong answer with nothing raising. The two
/// implementations agree on every vector without a chain, which is why this case
/// has to be constructed rather than stumbled on.
void check_snap_reads_the_original_vector() {
    const std::vector<double> chained{0.0, 1e-15, 2e-15, 3e-15, 0.5, 1.0};
    const double tol = knot_tolerance(view(chained));
    const std::vector<double> snapped = snap_knots(view(chained), tol);
    PANTR_CHECK(snapped == std::vector<double>({0.0, 0.0, 0.0, 0.0, 0.5, 1.0}));

    // What the in-place spelling would produce, computed here so the difference is
    // visible rather than asserted in the abstract.
    std::vector<double> in_place(chained);
    for (std::size_t i = 1; i < in_place.size(); ++i) {
        if (in_place[i] - in_place[i - 1] > tol) {
            continue;
        }
        in_place[i] = in_place[i - 1];
    }
    PANTR_CHECK_MSG(in_place != snapped,
                    "the in-place spelling must differ here, or this case proves nothing");
}

/// The basis count, non-periodic and periodic.
void check_num_basis() {
    const std::vector<double> clamped{0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0};
    // Seven knots, degree 2: `7 - 2 - 1`.
    PANTR_CHECK(num_basis(view(clamped), 2, false, knot_tolerance(view(clamped))) == 4);

    // Ten uniform knots, degree 2, periodic. The first in-domain knot has
    // multiplicity 1, so the regularity is `2 - 1 = 1` and the count is
    // `(10 - 2 - 1) - 1 - 1 = 5`.
    const std::vector<double> uniform{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    const double tol = knot_tolerance(view(uniform));
    PANTR_CHECK(num_basis(view(uniform), 2, false, tol) == 7);
    PANTR_CHECK(num_basis(view(uniform), 2, true, tol) == 5);
}

/// The snapping refusal, at both widths, with the oracle's text.
///
/// Both cases are reachable rather than contrived: a mesh of `n` intervals over a
/// span `s` at offset `x` survives only while `s / n > 8 * eps * max(s, |x|)`, so
/// four unit-quarter intervals stop being resolvable past `|x| / s` of about
/// `5.2e5 / n` at `float32` and `5.6e14 / n` at `float64`. The two widths differ in
/// their remedy line, which is why both are here: widening the format is only a
/// remedy when there is a wider one to move to.
void check_snapping_refusal() {
    const std::vector<float> raw32{1000000.0F,      1000000.0F,      1000000.0F,
                                   1000000.25F,     1000000.5F,      1000000.75F,
                                   1000001.0F,      1000001.0F,      1000001.0F};
    const double tol32 = knot_tolerance(view(raw32));
    const std::vector<float> snapped32 = snap_knots(view(raw32), tol32);
    same(message_of([&] { check_snapping_kept_an_interval<float>(raw32, snapped32, tol32); }),
         "knot snapping collapsed every knot onto 1000000.0: in float32 at |coordinate| ~ 1e+06 "
         "two knots are the same knot unless they differ by more than 0.954 (15 ulp there), and "
         "the closest pair in [1000000.0, 1000001.0] is 0.25 apart. This mesh is finer than "
         "float32 resolves at that magnitude. Use float64, move the domain nearer the origin, or "
         "coarsen the mesh.",
         "snapping refusal at float32");

    const std::vector<double> raw64{2e14,        2e14,        2e14,        2e14 + 0.25,
                                    2e14 + 0.5,  2e14 + 0.75, 2e14 + 1.0,  2e14 + 1.0,
                                    2e14 + 1.0};
    const double tol64 = knot_tolerance(view(raw64));
    const std::vector<double> snapped64 = snap_knots(view(raw64), tol64);
    same(message_of([&] { check_snapping_kept_an_interval<double>(raw64, snapped64, tol64); }),
         "knot snapping collapsed every knot onto 200000000000000.0: in float64 at |coordinate| ~ "
         "2e+14 two knots are the same knot unless they differ by more than 0.355 (11 ulp there), "
         "and the closest pair in [200000000000000.0, 200000000000001.0] is 0.25 apart. This mesh "
         "is finer than float64 resolves at that magnitude. Move the domain nearer the origin, or "
         "coarsen the mesh.",
         "snapping refusal at float64");

    // A vector that arrived flat is NOT this refusal's case, and passes through it
    // untouched so that the interval check downstream owns the consequence and can
    // give the message that is true of it.
    const std::vector<double> flat{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const double flat_tol = knot_tolerance(view(flat));
    same(message_of([&] {
             check_snapping_kept_an_interval<double>(flat, snap_knots(view(flat), flat_tol),
                                                     flat_tol);
         }),
         "<did not throw>", "an already-flat vector is not the snapping case");
}

/// The no-interval refusal, with the oracle's text.
///
/// The vector holds three distinct values and still has a one-point domain, because
/// the domain runs from `knots[degree]` to `knots[n - degree - 1]` and an interior
/// knot of high enough multiplicity swallows it whole. Counting intervals catches
/// that and the flat case together, and introduces no threshold of its own.
void check_no_interval_refusal() {
    const std::vector<double> swallowed{0.0, 1.0, 1.0, 1.0, 2.0};
    const double tol = knot_tolerance(view(swallowed));
    const KnotClassRange range = classify_knots(view(swallowed), 1, tol);
    PANTR_CHECK_MSG(range.num_intervals() == 0, "this vector must really span no interval");

    same(message_of([&] {
             check_space_has_an_interval<double>(swallowed, 1, range.num_intervals(), tol);
         }),
         "knot vector spans no interval: at degree 1 the domain runs from knots[1] = 1.0 to "
         "knots[3] = 1.0, and every step between them is at most this vector's tolerance of "
         "3.55e-15, so its in-domain knots are all one knot, the space has no cell, and nothing "
         "can be evaluated, tabulated or located on it. The domain needs two consecutive knots "
         "more than 3.55e-15 apart.",
         "no-interval refusal");

    same(message_of([&] { check_space_has_an_interval<double>(swallowed, 1, 1, tol); }),
         "<did not throw>", "a space with an interval passes");
}

}  // namespace

int main() {
    check_knot_scale();
    check_knot_tolerance();
    check_classes();
    check_chaining();
    check_the_boundary_step_merges();
    check_snap_takes_the_first_knot();
    check_snap_reads_the_original_vector();
    check_num_basis();
    check_snapping_refusal();
    check_no_interval_refusal();
    return pantr::test::summary("test_bspline_knots");
}
