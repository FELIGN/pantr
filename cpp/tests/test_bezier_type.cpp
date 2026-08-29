/// \file
/// Properties of the Bézier value type and its control net.
///
/// ## Why these cases and not others
///
/// Nothing here computes: the type stores coefficients and answers a handful of
/// questions about their shape. So there is no forward-error budget and no
/// tolerance -- every assertion below is exact, and a tolerance would be hiding
/// something rather than allowing for it. The cases are chosen for what they would
/// catch:
///
///  - **Validation, with its message.** The type is the C++ counterpart of Layer
///    2, so a caller with no Python is protected by these throws and nothing else.
///    The messages are asserted verbatim rather than merely "it threw", because
///    they are also the Python oracle's messages and
///    `tests/parity/test_bezier_type.py` compares the two character for character.
///    A reworded message here is a parity failure that a test asserting only the
///    exception *type* would not notice.
///  - **The rank arithmetic around the weight column.** Three shapes give a rank
///    of zero or less, each reaching it differently: no components at all, one
///    component consumed by the weight, and a rational net with no components,
///    which the oracle reports as `rank -1`. That last one is the whole reason the
///    subtraction is signed; an unsigned one would still reject the input while
///    reporting a number near 2^64, and only an assertion on the text catches a
///    wrong message on a correct verdict.
///  - **The rank-1 shape promotion.** `(n,)` means the scalar field `(n, 1)`, and
///    getting it wrong turns a degree-`n-1` curve into a degree-0 one in silence.
///  - **The copy at construction.** This is the defect of the oracle that the port
///    does not reproduce, so it is asserted rather than assumed: writing through
///    the caller's buffer after construction must not move the stored net.
///  - **Both scalar types.** `float32` is a supported storage format for a pantr
///    Bézier, unlike for `AABB`, so the type is exercised at `float` too.

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "pantr/bezier/bezier.hpp"

namespace {

using pantr::bezier::Bezier;
using pantr::bezier::ControlNet;

/// Build a control net from a coefficient list and a shape, for brevity below.
///
/// \param values The coefficients, row-major.
/// \param shape The extents.
/// \return The net.
template <class T>
ControlNet<T> net(std::vector<T> values, std::vector<std::size_t> shape) {
    return ControlNet<T>(std::span<const T>(values), std::span<const std::size_t>(shape));
}

/// The message of the `std::invalid_argument` that `fn` throws.
///
/// Returns a marker rather than asserting, so that the caller's own check is the
/// one that reports and a failure names the case it came from.
///
/// \param fn The call to attempt.
/// \return The exception's `what()`, or a marker saying what happened instead.
template <class F>
std::string message_of(F&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument& e) {
        return e.what();
    } catch (...) {
        return "<threw something other than std::invalid_argument>";
    }
    return "<did not throw>";
}

/// Whether calling `fn` throws `std::out_of_range`.
///
/// \param fn The call to attempt.
/// \return `true` when it threw.
template <class F>
bool throws_out_of_range(F&& fn) {
    try {
        fn();
    } catch (const std::out_of_range&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

void check_shape_validation() {
    PANTR_CHECK(message_of([] { (void)net<double>({}, {}); })
                == "Control points must be at least 1D.");

    // The parametric directions are checked; the component axis is not, and must
    // not be -- a zero component count is the rank check's to report, with a
    // message of its own.
    PANTR_CHECK(message_of([] { (void)net<double>({}, {0, 2}); })
                == "Control points must have at least 1 entry in parametric direction 0, got 0.");
    PANTR_CHECK(message_of([] { (void)net<double>({}, {2, 0, 3}); })
                == "Control points must have at least 1 entry in parametric direction 1, got 0.");

    // No oracle counterpart: a numpy array's data and shape agree by
    // construction, so only a caller without numpy can reach this one.
    PANTR_CHECK(message_of([] { (void)net<double>({1.0, 2.0}, {3, 1}); })
                == "Control points hold 2 values, but the shape asks for 3.");
}

void check_rank_validation() {
    PANTR_CHECK(message_of([] { (void)Bezier<double>(net<double>({}, {3, 0}), false); })
                == "The Bézier must have at least rank one. Got rank 0.");
    PANTR_CHECK(message_of(
                    [] { (void)Bezier<double>(net<double>({1.0, 2.0, 3.0}, {3, 1}), true); })
                == "The Bézier must have at least rank one. Got rank 0.");
    PANTR_CHECK(message_of([] { (void)Bezier<double>(net<double>({}, {3, 0}), true); })
                == "The Bézier must have at least rank one. Got rank -1.");

    // The shape check runs first, so a net that is bad in both ways reports the
    // shape. Reordering the two would silently change what a caller reads and
    // would diverge from the oracle.
    PANTR_CHECK(message_of([] { (void)Bezier<double>(net<double>({}, {0, 0}), true); })
                == "Control points must have at least 1 entry in parametric direction 0, got 0.");
}

void check_derived_quantities() {
    const std::vector<double> cube{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    const Bezier<double> surface(net<double>(cube, {2, 2, 2}), false);
    PANTR_CHECK(surface.dim() == 2);
    PANTR_CHECK(surface.rank() == 2);
    PANTR_CHECK(surface.degree() == std::vector<std::size_t>({1, 1}));
    PANTR_CHECK(surface.degree(0) == 1);
    PANTR_CHECK(!surface.is_rational());
    PANTR_CHECK(surface.net().num_components() == 2);
    PANTR_CHECK(surface.net().size() == 8);
    PANTR_CHECK(surface.net().extent(1) == 2);
    PANTR_CHECK(throws_out_of_range([&surface] { (void)surface.net().extent(2); }));
    PANTR_CHECK(throws_out_of_range([&surface] { (void)surface.degree(2); }));

    // A rational planar curve: three stored components, rank 2.
    const Bezier<double> curve(net<double>({0.0, 0.0, 1.0, 1.0, 1.0, 2.0}, {2, 3}), true);
    PANTR_CHECK(curve.dim() == 1);
    PANTR_CHECK(curve.rank() == 2);
    PANTR_CHECK(curve.degree() == std::vector<std::size_t>({1}));
    PANTR_CHECK(curve.is_rational());

    // Degree 0 in one direction is legal: one coefficient is a constant, not an
    // empty direction.
    const Bezier<double> ribbon(net<double>({1.0, 2.0, 3.0}, {1, 3, 1}), false);
    PANTR_CHECK(ribbon.degree() == std::vector<std::size_t>({0, 2}));
}

void check_rank_one_shape_is_promoted() {
    const Bezier<double> scalar_curve(net<double>({0.0, 1.0, 4.0}, {3}), false);
    PANTR_CHECK(scalar_curve.dim() == 1);
    PANTR_CHECK(scalar_curve.rank() == 1);
    PANTR_CHECK(scalar_curve.degree() == std::vector<std::size_t>({2}));
    PANTR_CHECK(scalar_curve.net().shape().size() == 2);
    PANTR_CHECK(scalar_curve.net().shape()[1] == 1);
}

void check_the_net_copies_its_input() {
    std::vector<double> values{0.0, 1.0, 2.0, 3.0};
    const std::vector<std::size_t> shape{4, 1};
    const Bezier<double> curve(
        ControlNet<double>(std::span<const double>(values), std::span<const std::size_t>(shape)),
        false);

    values[0] = 99.0;
    PANTR_CHECK_MSG(curve.net().values()[0] == 0.0,
                    "the net aliased the caller's buffer instead of copying it");
}

void check_float_storage() {
    const Bezier<float> curve(net<float>({0.0F, 1.0F, 2.0F, 3.0F}, {2, 2}), true);
    PANTR_CHECK(curve.dim() == 1);
    PANTR_CHECK(curve.rank() == 1);
    PANTR_CHECK(curve.degree() == std::vector<std::size_t>({1}));
    PANTR_CHECK(curve.net().values()[3] == 3.0F);
}

}  // namespace

int main() {
    check_shape_validation();
    check_rank_validation();
    check_derived_quantities();
    check_rank_one_shape_is_promoted();
    check_the_net_copies_its_input();
    check_float_storage();
    return pantr::test::summary("test_bezier_type");
}
