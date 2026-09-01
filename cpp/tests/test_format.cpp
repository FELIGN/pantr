/// \file
/// The three Python format specifiers `pantr/core/format.hpp` reproduces.
///
/// ## Why this file exists separately from `test_aabb.cpp`
///
/// `format_repr` was checked until now only through the AABB messages that use
/// it, which meant it was checked on the values those messages happened to carry.
/// That is exactly how its notation bug survived: fixed and scientific notation
/// coincide on typical magnitudes and diverge on round numbers, so a sweep over
/// realistic box corners never reached the divergence. Now that a second type
/// depends on the same rule, the rule gets its own cases, chosen for where the two
/// languages *could* part company rather than for where they are exercised.
///
/// ## Where each expectation comes from
///
/// Every string below is what CPython prints, and each was obtained by running the
/// corresponding expression rather than by reasoning about it. The three that
/// matter most are the round numbers -- `1e5`, `1e16`, `1e-5` -- because those are
/// the ones where `std::to_chars`'s "shortest text" rule and Python's positional
/// rule give different answers, and where a formatter that merely round-trips
/// would still be wrong.
///
/// Nothing here has a tolerance. A rendering is a string: it is right or it is not.

#include <cmath>
#include <limits>
#include <string>

#include "check.hpp"
#include "pantr/core/format.hpp"

namespace {

using pantr::detail::format_fixed;
using pantr::detail::format_general;
using pantr::detail::format_repr;
using pantr::detail::format_scalar;

/// Check one rendering against the text Python produces.
///
/// \param actual What the header produced.
/// \param expected What CPython prints.
/// \param what The expression, for the failure line.
void same(const std::string& actual, const std::string& expected, const char* what) {
    PANTR_CHECK_MSG(actual == expected,
                    std::string(what) + ": got '" + actual + "', want '" + expected + "'");
}

/// `repr`, on the values where the notation rule decides the answer.
void check_repr() {
    // Ordinary magnitudes, where the two rules agree and a wrong formatter still
    // looks right.
    same(format_repr(0.5), "0.5", "repr(0.5)");
    same(format_repr(-0.5), "-0.5", "repr(-0.5)");
    same(format_repr(0.1), "0.1", "repr(0.1)");
    same(format_repr(2.0), "2.0", "repr(2.0)");
    same(format_repr(0.0), "0.0", "repr(0.0)");
    same(format_repr(-0.0), "-0.0", "repr(-0.0)");

    // The round numbers. `to_chars` alone renders these as `1e+05` and `1e-05`,
    // because the scientific spelling is shorter; Python decides on the decimal
    // exponent and keeps fixed notation until it leaves `[-4, 16)`.
    same(format_repr(100000.0), "100000.0", "repr(1e5)");
    same(format_repr(1e15), "1000000000000000.0", "repr(1e15)");
    same(format_repr(1e16), "1e+16", "repr(1e16)");
    same(format_repr(1e-4), "0.0001", "repr(1e-4)");
    same(format_repr(1e-5), "1e-05", "repr(1e-5)");

    // Shortest round-trip, where a fixed 17-digit rendering would differ.
    same(format_repr(0.30000000000000004), "0.30000000000000004", "repr(0.1 + 0.2)");
    same(format_repr(1000000.25), "1000000.25", "repr(1000000.25)");
    same(format_repr(200000000000000.0), "200000000000000.0", "repr(2e14)");

    // The non-finite values, which Python spells without a sign on nan.
    constexpr double infinity = std::numeric_limits<double>::infinity();
    same(format_repr(infinity), "inf", "repr(inf)");
    same(format_repr(-infinity), "-inf", "repr(-inf)");
    same(format_repr(std::numeric_limits<double>::quiet_NaN()), "nan", "repr(nan)");
}

/// `repr` reached through a scalar, at both storage widths.
///
/// The widening is the point: `repr(float(np.float32(0.7)))` is the `double`
/// rendering of the `float32` value, not `0.7`, and a formatter that rendered the
/// `float` directly would print `0.7` and be wrong by the same rule the oracle
/// follows.
void check_scalar() {
    same(format_scalar(0.7), "0.7", "repr(0.7) as double");
    same(format_scalar(0.7F), "0.699999988079071", "repr(float(np.float32(0.7)))");
    same(format_scalar(1000000.0F), "1000000.0", "repr(float(np.float32(1e6)))");
}

/// `f"{v:.Ng}"`, including the two exponent forms and the trailing-zero rule.
void check_general() {
    same(format_general(1000000.0, 3), "1e+06", "f'{1e6:.3g}'");
    same(format_general(0.9536743, 3), "0.954", "f'{0.9536743:.3g}'");
    // The tolerance of a unit-domain float64 space, which is what the "spans no
    // interval" message interpolates.
    same(format_general(3.552713678800501e-15, 3), "3.55e-15", "f'{3.55e-15:.3g}'");
    same(format_general(0.25, 3), "0.25", "f'{0.25:.3g}'");
    same(format_general(2e14, 3), "2e+14", "f'{2e14:.3g}'");
    same(format_general(1.0, 3), "1", "f'{1.0:.3g}'");
    same(format_general(1234.0, 3), "1.23e+03", "f'{1234.0:.3g}'");
    // Negative values, which no case here reached until a review said so.
    same(format_general(-0.9536743, 3), "-0.954", "f'{-0.9536743:.3g}'");
    same(format_general(-1e6, 3), "-1e+06", "f'{-1e6:.3g}'");
    same(format_general(-0.0, 3), "-0", "f'{-0.0:.3g}'");
}

/// The non-finite values, at every spelling.
///
/// CPython prints a NaN **without a sign, ever**, even for one whose sign bit is
/// set; glibc's `printf` prints `-nan` for that value. Found by an adversarial
/// case rather than by reasoning: a knot vector holding an infinity produces
/// `tol / ulp == inf / nan`, the NaN came out negative, and the two backends'
/// messages differed by one character. Every spelling therefore intercepts these
/// rather than handing them to `snprintf`.
void check_non_finite() {
    constexpr double infinity = std::numeric_limits<double>::infinity();
    const double positive_nan = std::numeric_limits<double>::quiet_NaN();
    const double negative_nan = -positive_nan;
    PANTR_CHECK_MSG(std::signbit(negative_nan), "the negative NaN must really be negative");

    for (const double nan : {positive_nan, negative_nan}) {
        same(format_repr(nan), "nan", "repr(nan)");
        same(format_general(nan, 3), "nan", "f'{nan:.3g}'");
        same(format_fixed(nan, 0), "nan", "f'{nan:.0f}'");
    }
    same(format_general(infinity, 3), "inf", "f'{inf:.3g}'");
    same(format_general(-infinity, 3), "-inf", "f'{-inf:.3g}'");
    same(format_fixed(infinity, 0), "inf", "f'{inf:.0f}'");
    same(format_fixed(-infinity, 0), "-inf", "f'{-inf:.0f}'");
}

/// `f"{v:.Nf}"`, at the precision the snapping message uses.
///
/// The half-way case is the one worth pinning: both CPython and glibc round to
/// even, so 0.5 renders as `0` and 1.5 as `2`. A formatter rounding half away from
/// zero would print `1` and `2`.
void check_fixed() {
    same(format_fixed(15.4, 0), "15", "f'{15.4:.0f}'");
    same(format_fixed(11.2, 0), "11", "f'{11.2:.0f}'");
    same(format_fixed(0.5, 0), "0", "f'{0.5:.0f}'");
    same(format_fixed(1.5, 0), "2", "f'{1.5:.0f}'");
    same(format_fixed(2.5, 0), "2", "f'{2.5:.0f}'");
    same(format_fixed(0.125, 2), "0.12", "f'{0.125:.2f}'");
    same(format_fixed(-15.4, 0), "-15", "f'{-15.4:.0f}'");
    same(format_fixed(-0.5, 0), "-0", "f'{-0.5:.0f}'");
    same(format_fixed(-0.125, 2), "-0.12", "f'{-0.125:.2f}'");
}

}  // namespace

int main() {
    check_repr();
    check_scalar();
    check_general();
    check_fixed();
    check_non_finite();
    return pantr::test::summary("test_format");
}
