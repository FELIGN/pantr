/// \file
/// The 1D B-spline space: what it stores, what it derives, and what it refuses.
///
/// ## What is asserted, and against what
///
/// The structural quantities -- basis counts, interval counts, class
/// multiplicities, per-interval first-basis indices -- are read off each knot
/// vector by hand, and every case below says which feature of the vector it is
/// there for. Three of them are the shipped docstring examples of
/// `pantr.bspline.BsplineSpace1D`, so they are checkable against the library's own
/// published contract rather than against a run of it.
///
/// Nothing here carries a numerical tolerance. A space stores knots and answers
/// counting questions about them; every assertion is an integer, a knot value
/// reproduced bit for bit from the input, a boolean, or a string.
///
/// ## The three cases that exist because getting them wrong is silent
///
/// **`check_stored_basis_count_comes_from_the_snapped_vector`.** The oracle
/// computes the basis count *twice* from different vectors: once from the knots as
/// supplied, purely to refuse a vector that cannot support the degree, and once
/// from the snapped vector, which is what it stores. Snapping can change the
/// multiplicity of the first in-domain knot, so for a periodic space the two
/// genuinely differ -- 3 against 4 on the vector used here. A port that stored the
/// validation count would be wrong by one basis function with nothing raising.
///
/// **`check_the_derived_block_is_shared_not_rebuilt`.** The derived arrays are
/// handed out as views of storage the space owns. If an accessor rebuilt or copied
/// them, every value assertion in this file would still pass, a caller's span would
/// dangle, and a loop over intervals would recompute an `O(n)` scan per iteration.
/// Comparing addresses is what distinguishes the two.
///
/// **`check_end_openness_is_absolute`.** Clampedness is an absolute comparison
/// against the space's own tolerance, in `double`. A relative one -- `np.isclose`'s
/// default `rtol` is `1e-5` -- would read a gap of `1e-5 * |knots[degree]|` as
/// clamped, which on a knot vector based at 1e6 is half the domain.

#include <atomic>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "check.hpp"
#include "pantr/bspline/space_1d.hpp"

namespace {

using pantr::bspline::BsplineSpace1D;
using pantr::bspline::KnotSnapping;

/// Build a space from a knot list, snapping, for brevity below.
template <class T>
BsplineSpace1D<T> space(const std::vector<T>& knots, std::int64_t degree, bool periodic = false) {
    return BsplineSpace1D<T>(std::span<const T>(knots), degree, periodic,
                             KnotSnapping::merge_near_duplicates);
}

/// The message of the `std::invalid_argument` that `fn` throws.
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

/// A span compared against a vector, elementwise.
template <class T>
bool equals(std::span<const T> actual, const std::vector<T>& expected) {
    return std::vector<T>(actual.begin(), actual.end()) == expected;
}

/// A clamped quadratic over two unit intervals: the docstring's own example.
void check_clamped_quadratic() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0};
    const auto s = space(knots, 2);

    PANTR_CHECK(equals(s.knots(), knots));
    PANTR_CHECK(s.degree() == 2);
    PANTR_CHECK(!s.periodic());
    // Seven knots, degree 2, so `7 - 2 - 1` functions over two distinct interior
    // steps. `space.num_intervals` is 2 in the shipped docstring.
    PANTR_CHECK(s.num_basis() == 4);
    PANTR_CHECK(s.num_intervals() == 2);
    PANTR_CHECK(s.domain()[0] == 0.0);
    PANTR_CHECK(s.domain()[1] == 2.0);
    PANTR_CHECK(s.has_left_end_open());
    PANTR_CHECK(s.has_right_end_open());
    PANTR_CHECK(s.has_open_knots());
    // Not Bézier-like: four basis functions against `degree + 1 == 3`, so there is
    // more than one non-zero span.
    PANTR_CHECK(!s.has_bezier_like_knots());

    PANTR_CHECK(equals(s.unique_knots(), std::vector<double>{0.0, 1.0, 2.0}));
    PANTR_CHECK(equals(s.multiplicity(), std::vector<std::int64_t>{3, 1, 3}));
    PANTR_CHECK(equals(s.unique_knots_in_domain(), std::vector<double>{0.0, 1.0, 2.0}));
    PANTR_CHECK(equals(s.multiplicity_in_domain(), std::vector<std::int64_t>{3, 1, 3}));
    PANTR_CHECK(equals(s.first_basis_per_interval(), std::vector<std::int64_t>{0, 1}));
}

/// An interior knot of multiplicity 2: the docstring's `[0, 1, 3]` example.
///
/// The successive differences of the result are the interior multiplicities, which
/// is what the jump from 1 to 3 shows and what a per-interval loop that assumed
/// consecutive indices would get wrong.
void check_repeated_interior_knot() {
    const auto s = space(std::vector<double>{0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0}, 2);
    PANTR_CHECK(s.num_basis() == 6);
    PANTR_CHECK(s.num_intervals() == 3);
    PANTR_CHECK(equals(s.multiplicity(), std::vector<std::int64_t>{3, 1, 2, 3}));
    PANTR_CHECK(equals(s.first_basis_per_interval(), std::vector<std::int64_t>{0, 1, 3}));
}

/// A single Bézier segment, and degree zero.
void check_bezier_like_and_degree_zero() {
    // `[1, 1, 1, 3, 3, 3]` at degree 2: clamped, one span, `degree + 1` functions.
    const auto bez = space(std::vector<double>{1.0, 1.0, 1.0, 3.0, 3.0, 3.0}, 2);
    PANTR_CHECK(bez.num_basis() == 3);
    PANTR_CHECK(bez.num_intervals() == 1);
    PANTR_CHECK(bez.has_bezier_like_knots());
    PANTR_CHECK(equals(bez.first_basis_per_interval(), std::vector<std::int64_t>{0}));

    // Degree 0: one function per interval, and the ends are trivially "clamped"
    // because the first and last `degree + 1 == 1` knots are one knot.
    const auto flat = space(std::vector<double>{0.0, 1.0, 2.0, 3.0}, 0);
    PANTR_CHECK(flat.num_basis() == 3);
    PANTR_CHECK(flat.num_intervals() == 3);
    PANTR_CHECK(flat.has_open_knots());
    PANTR_CHECK(!flat.has_bezier_like_knots());
    PANTR_CHECK(equals(flat.first_basis_per_interval(), std::vector<std::int64_t>{0, 1, 2}));
}

/// An unclamped vector, where the domain is a strict interior.
void check_unclamped() {
    const auto s = space(std::vector<double>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0}, 3);
    PANTR_CHECK(s.num_basis() == 5);
    PANTR_CHECK(s.num_intervals() == 2);
    PANTR_CHECK(s.domain()[0] == 3.0);
    PANTR_CHECK(s.domain()[1] == 5.0);
    PANTR_CHECK(!s.has_left_end_open());
    PANTR_CHECK(!s.has_right_end_open());
    PANTR_CHECK(equals(s.unique_knots_in_domain(), std::vector<double>{3.0, 4.0, 5.0}));
    // The first interval is no different from the rest: selecting the classes whose
    // last position lies in `[degree, num_basis)` needs no special case for a
    // non-clamped vector.
    PANTR_CHECK(equals(s.first_basis_per_interval(), std::vector<std::int64_t>{0, 1}));
}

/// A periodic space: no clamped end whatever the knots, and no per-interval index.
void check_periodic() {
    const auto s = space(
        std::vector<double>{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}, 2, true);
    PANTR_CHECK(s.periodic());
    // `(10 - 2 - 1)` less the regularity `2 - 1` less one.
    PANTR_CHECK(s.num_basis() == 5);
    PANTR_CHECK(s.num_intervals() == 5);
    PANTR_CHECK(s.domain()[0] == 2.0);
    PANTR_CHECK(s.domain()[1] == 7.0);
    PANTR_CHECK_MSG(!s.has_left_end_open(), "a periodic space has no clamped end by definition");
    PANTR_CHECK(!s.has_right_end_open());
    PANTR_CHECK(!s.has_bezier_like_knots());
    PANTR_CHECK(equals(s.unique_knots_in_domain(),
                       std::vector<double>{2.0, 3.0, 4.0, 5.0, 6.0, 7.0}));

    same(message_of([&] { (void)s.first_basis_per_interval(); }),
         "first_basis_per_interval: periodic B-spline spaces are not supported.",
         "first_basis_per_interval on a periodic space");
}

/// The stored basis count comes from the snapped vector, not the validated one.
///
/// See the file comment. `knots[0..2]` step by `0.6 * tol` each, so they chain into
/// one class although the outer two are `1.2 * tol` apart: the multiplicity of the
/// first in-domain knot is 2 before snapping and 3 after, which moves the periodic
/// regularity and so the count. The oracle reports 4.
void check_stored_basis_count_comes_from_the_snapped_vector() {
    constexpr double tol = 8.0 * 2.220446049250313e-16;  // the tolerance of a unit domain
    const std::vector<double> knots{0.0, 0.6 * tol, 1.2 * tol, 0.25, 0.5, 0.75, 1.0};
    const auto s = space(knots, 2, true);

    PANTR_CHECK(equals(s.knots(), std::vector<double>{0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0}));
    PANTR_CHECK_MSG(s.num_basis() == 4,
                    "the count is 3 from the vector as supplied and 4 from the snapped one");
    PANTR_CHECK(s.num_intervals() == 2);
}

/// `KnotSnapping::as_given` stores the vector untouched.
///
/// The interval requirement still applies; only the merging and the snapping
/// diagnosis are skipped. The class scan then reports the same classes either way,
/// because grouping is by gap and the gaps are unchanged -- what differs is the
/// values stored, which is what a caller asking for `as_given` wants.
void check_as_given() {
    const std::vector<double> raw{0.0, 0.0, 0.0, 0.5, 0.5 + 2e-16, 1.0, 1.0, 1.0};
    const BsplineSpace1D<double> kept(std::span<const double>(raw), 2, false,
                                      KnotSnapping::as_given);
    PANTR_CHECK(equals(kept.knots(), raw));
    PANTR_CHECK(kept.num_intervals() == 2);
    PANTR_CHECK(equals(kept.unique_knots(), std::vector<double>{0.0, 0.5, 1.0}));
    PANTR_CHECK(equals(kept.multiplicity(), std::vector<std::int64_t>{3, 2, 3}));

    const auto merged = space(raw, 2);
    PANTR_CHECK(merged.knots()[4] == 0.5);
    PANTR_CHECK_MSG(kept.knots()[4] != merged.knots()[4],
                    "the two spellings must really store different knots here");
}

/// Clampedness is absolute, and in `double`.
///
/// A relative comparison at `numpy`'s default `rtol` of `1e-5` would admit a gap of
/// `1e-5 * 1e6 == 10` at the left end of this vector, which is ten times its whole
/// domain. The gap here is 1, so an absolute test rejects it and a relative one
/// would not.
void check_end_openness_is_absolute() {
    const auto s = space(std::vector<double>{1000000.0, 1000000.0, 1000001.0, 1000002.0,
                                             1000003.0, 1000003.0},
                         2);
    PANTR_CHECK_MSG(s.tolerance() < 1.0, "the tolerance must be far below the gap being tested");
    PANTR_CHECK_MSG(!s.has_left_end_open(),
                    "knots[0] and knots[degree] differ by 1, far more than the tolerance");
    PANTR_CHECK(!s.has_right_end_open());
}

/// A clamping gap of exactly the tolerance counts as clamped.
///
/// The comparison is `<= tol`, so the boundary is closed on the clamped side, and
/// nothing else here pins that: every other case is orders clear of the threshold.
/// Snapping is off, because with it on the two knots would merge first -- for the
/// same reason, the boundary being closed -- and the gap under test would be zero.
///
/// The oracle agrees: the same vector reports a clamped left end and an unclamped
/// right one.
void check_the_boundary_clamping_gap_is_open() {
    constexpr double eps = std::numeric_limits<double>::epsilon();
    const std::vector<double> knots{0.0, 8.0 * eps, 0.5, 1.0};
    const BsplineSpace1D<double> s(std::span<const double>(knots), 1, false,
                                   KnotSnapping::as_given);
    PANTR_CHECK_MSG(s.tolerance() == 8.0 * eps, "the scale must be exactly 1 here");
    PANTR_CHECK_MSG(knots[1] - knots[0] == s.tolerance(),
                    "the gap must be exactly the tolerance");
    PANTR_CHECK_MSG(s.has_left_end_open(), "a gap of exactly the tolerance is clamped");
    PANTR_CHECK_MSG(!s.has_right_end_open(), "and the far end is not, which keeps this honest");
}

/// The derived arrays are the space's own storage, shared rather than rebuilt.
///
/// Addresses, because values would agree under a copying implementation. The
/// in-domain forms are asserted to be *subspans of the same buffer*, which is what
/// makes them one memo rather than two.
void check_the_derived_block_is_shared_not_rebuilt() {
    const auto s = space(std::vector<double>{0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0}, 2);

    PANTR_CHECK(s.unique_knots().data() == s.unique_knots().data());
    const auto* first = s.unique_knots().data();
    const auto* again = s.unique_knots().data();
    PANTR_CHECK_MSG(first == again, "two reads must return the same storage");
    PANTR_CHECK_MSG(s.multiplicity().data() == s.multiplicity().data(),
                    "and so must the multiplicities");
    PANTR_CHECK_MSG(s.first_basis_per_interval().data() == s.first_basis_per_interval().data(),
                    "and so must the per-interval indices");

    const auto whole = s.unique_knots();
    const auto in_domain = s.unique_knots_in_domain();
    PANTR_CHECK_MSG(in_domain.data() >= whole.data()
                        && in_domain.data() + in_domain.size() <= whole.data() + whole.size(),
                    "the in-domain form is a subspan of the whole, not a second array");
}

/// A copy carries the value and not the memo; a move leaves the target usable.
///
/// Copying the memo would be correct here and buys nothing; what must not happen is
/// the opposite error, a copy that keeps a memo describing the source. The values
/// are asserted after the copy so that "cold" does not mean "wrong".
void check_copy_and_move() {
    const std::vector<double> knots{0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0};
    auto original = space(knots, 2);
    PANTR_CHECK(equals(original.unique_knots(), std::vector<double>{0.0, 1.0, 2.0}));

    const BsplineSpace1D<double> copied(original);
    PANTR_CHECK(copied.num_basis() == 4);
    PANTR_CHECK(copied.num_intervals() == 2);
    PANTR_CHECK(copied.tolerance() == original.tolerance());
    PANTR_CHECK(equals(copied.unique_knots(), std::vector<double>{0.0, 1.0, 2.0}));
    PANTR_CHECK_MSG(copied.unique_knots().data() != original.unique_knots().data(),
                    "the copy owns its own derived block");

    const BsplineSpace1D<double> moved(std::move(original));
    PANTR_CHECK(moved.num_basis() == 4);
    PANTR_CHECK(equals(moved.first_basis_per_interval(), std::vector<std::int64_t>{0, 1}));
}

/// The same space at `float`, where the stored knots are the rounded ones.
///
/// `float32` is a supported storage format for a pantr space, so the type is
/// exercised at both widths. `0.7F` is not 0.7, and the space stores and reports
/// the `float` value; the tolerance is `8 * eps32` rather than `8 * eps64`, which is
/// four orders wider and is why a `float32` mesh stops resolving so much sooner.
void check_float_storage() {
    const std::vector<float> knots{0.0F, 0.0F, 0.0F, 0.25F, 0.7F, 0.7F, 1.0F, 1.0F, 1.0F};
    const auto s = space(knots, 2);
    PANTR_CHECK(s.num_basis() == 6);
    PANTR_CHECK(s.num_intervals() == 3);
    PANTR_CHECK(s.domain()[0] == 0.0F);
    PANTR_CHECK(s.domain()[1] == 1.0F);
    PANTR_CHECK(equals(s.unique_knots(), std::vector<float>{0.0F, 0.25F, 0.7F, 1.0F}));
    PANTR_CHECK(equals(s.multiplicity(), std::vector<std::int64_t>{3, 1, 2, 3}));
    PANTR_CHECK(equals(s.first_basis_per_interval(), std::vector<std::int64_t>{0, 1, 3}));
    PANTR_CHECK(s.tolerance() == 8.0 * static_cast<double>(1.1920928955078125e-07));
}

/// The four argument refusals, in the oracle's order and with its text.
///
/// The order is load-bearing rather than incidental: two simultaneously bad
/// arguments must produce the same message on both sides, and only the order
/// decides which one they get. The last case has a legal length and a legal
/// ordering and still cannot support its degree.
void check_argument_refusals() {
    const std::vector<double> fine{0.0, 0.0, 1.0, 1.0};
    same(message_of([&] {
             (void)BsplineSpace1D<double>(std::span<const double>(fine), -1, false,
                                          KnotSnapping::merge_near_duplicates);
         }),
         "degree must be non-negative", "negative degree");

    const std::vector<double> tooshort{0.0, 0.0, 1.0};
    same(message_of([&] { (void)space(tooshort, 2); }),
         "knots must have at least 2*degree+2 elements", "too few knots");

    const std::vector<double> unsorted{0.0, 0.0, 1.0, 0.5, 1.0, 1.0};
    same(message_of([&] { (void)space(unsorted, 2); }), "knots must be non-decreasing",
         "a descending step");

    const std::vector<double> uniform{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    same(message_of([&] { (void)space(uniform, 2, true); }),
         "Not enough knots for the specified degree", "a periodic space with too few functions");

    // Each adjacent pair of checks, exercised by an input that fails BOTH, so that
    // only the order decides the message. Without these, swapping two checks that
    // are each individually correct leaves the whole suite green -- which is what a
    // review found, with only the first of the three transitions covered. Every
    // expected message here is the one the oracle produces for the same input.
    same(message_of([&] {
             (void)BsplineSpace1D<double>(std::span<const double>(tooshort), -1, false,
                                          KnotSnapping::merge_near_duplicates);
         }),
         "degree must be non-negative", "degree before length");

    const std::vector<double> short_and_descending{1.0, 0.0, 1.0};
    same(message_of([&] { (void)space(short_and_descending, 2); }),
         "knots must have at least 2*degree+2 elements", "length before monotonicity");

    const std::vector<double> descending_and_too_few{0.0, 1.0, 2.0, 3.0, 2.0, 5.0, 6.0};
    same(message_of([&] { (void)space(descending_and_too_few, 2, true); }),
         "knots must be non-decreasing", "monotonicity before the basis count");
}

/// The two constructor refusals that are properties of the knots, not the arguments.
///
/// Snapping is diagnosed before the interval requirement, and that ordering is what
/// lets each message be true: "this mesh is finer than float32 resolves here" names
/// a remedy and is false of a vector the caller supplied flat, which the second
/// check owns instead.
void check_knot_refusals() {
    const std::vector<float> unresolvable{1000000.0F,  1000000.0F,  1000000.0F,
                                          1000000.25F, 1000000.5F,  1000000.75F,
                                          1000001.0F,  1000001.0F,  1000001.0F};
    const std::string collapsed = message_of([&] { (void)space(unresolvable, 2); });
    PANTR_CHECK_MSG(collapsed.rfind("knot snapping collapsed every knot onto 1000000.0:", 0) == 0,
                    "an unresolvable mesh is diagnosed as snapping, not as a flat domain: got '"
                        + collapsed + "'");

    // The same vector with snapping off reaches the *other* refusal, which is what
    // shows the two are separate rules rather than one with two spellings.
    const std::string as_given = message_of([&] {
        (void)BsplineSpace1D<float>(std::span<const float>(unresolvable), 2, false,
                                    KnotSnapping::as_given);
    });
    PANTR_CHECK_MSG(as_given.rfind("knot vector spans no interval:", 0) == 0,
                    "with snapping off the interval rule owns the same symptom: got '" + as_given
                        + "'");

    const std::vector<double> swallowed{0.0, 1.0, 1.0, 1.0, 2.0};
    const std::string no_interval = message_of([&] { (void)space(swallowed, 1); });
    PANTR_CHECK_MSG(no_interval.rfind("knot vector spans no interval: at degree 1", 0) == 0,
                    "a domain swallowed by an interior knot: got '" + no_interval + "'");
}

/// Eight threads first-touching one space's derived block. The sanitizer's target.
///
/// It asserts what it can -- one buffer, the right values from every thread -- but
/// its real job is to put concurrent first touches through the memo so that a
/// thread-sanitizer build has something to look at. A value assertion cannot see
/// the failure this guards: the unsynchronised spelling of the same memo produced
/// 60 correct answers in 60 runs and 4 sanitizer reports.
void check_concurrent_first_touch() {
    constexpr int num_threads = 8;
    const auto s = space(std::vector<double>{0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0}, 2);

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<const double*> seen(num_threads, nullptr);
    std::vector<std::int64_t> sums(num_threads, -1);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t] {
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {
            }
            const auto index = static_cast<std::size_t>(t);
            seen[index] = s.unique_knots().data();
            std::int64_t total = 0;
            for (const std::int64_t m : s.multiplicity()) {
                total += m;
            }
            sums[index] = total;
        });
    }
    while (ready.load(std::memory_order_acquire) < num_threads) {
    }
    go.store(true, std::memory_order_release);
    for (std::thread& thread : threads) {
        thread.join();
    }

    for (int t = 0; t < num_threads; ++t) {
        const auto index = static_cast<std::size_t>(t);
        PANTR_CHECK_MSG(seen[index] == seen[0], "every thread must see one buffer");
        PANTR_CHECK_MSG(sums[index] == 9, "the multiplicities sum to the knot count");
    }
}

}  // namespace

int main() {
    check_clamped_quadratic();
    check_repeated_interior_knot();
    check_bezier_like_and_degree_zero();
    check_unclamped();
    check_periodic();
    check_stored_basis_count_comes_from_the_snapped_vector();
    check_as_given();
    check_end_openness_is_absolute();
    check_the_boundary_clamping_gap_is_open();
    check_the_derived_block_is_shared_not_rebuilt();
    check_copy_and_move();
    check_float_storage();
    check_argument_refusals();
    check_knot_refusals();
    check_concurrent_first_touch();
    return pantr::test::summary("test_bspline_space_1d");
}
