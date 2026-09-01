#pragma once

/// \file
/// Rendering a floating-point value the way Python renders it, for the port's
/// exception messages.
///
/// ## Why this is core rather than local to one type
///
/// The port reproduces its oracle's messages character for character, because a
/// parity test compares them (`tests/parity/test_bezier_type.py` is the shipped
/// example) and because the message is part of what a caller is told. Several of
/// those messages interpolate a floating-point number, and each spelling below is
/// a distinct Python format specifier, not a matter of taste:
///
///  - `format_repr`, and `format_scalar` over a scalar, are `repr(x)` and `f"{x!r}"`,
///  - `format_general` is `f"{x:.Ng}"`,
///  - `format_fixed` is `f"{x:.Nf}"`.
///
/// `pantr/geometry/aabb.hpp` needed the first of them and grew it in place; it is
/// here now because `pantr/bspline/knots.hpp` needs all three, and a second copy
/// of a repr rule is exactly how the first one went wrong. The notation bug the
/// comment on `format_repr` records was found once and would have had to be found
/// again.
///
/// ## Where the two languages actually agree
///
/// The *digits* are correctly rounded on both sides -- CPython formats through
/// David Gay's `dtoa` and glibc's `printf` is correctly rounded too -- so
/// `%.3g` and `f"{x:.3g}"` produce the same characters. What does **not** transfer
/// is `std::to_chars`'s choice of notation, which is why `format_repr` picks the
/// notation itself. That difference is the only one this file has to bridge, and
/// it is bridged in exactly one place.
///
/// ## One deliberate change from the code this replaces
///
/// The unreachable formatting-overflow branch used to throw
/// `std::invalid_argument("AABB: could not format a bound.")`. It now throws
/// `std::logic_error`, which is the accurate type: nothing about the caller's
/// argument is wrong, an internal buffer bound was exceeded. It matters at the
/// binding, where `pantr/core/error.hpp` records that nanobind maps
/// `std::invalid_argument` to `ValueError` and everything else to `RuntimeError` --
/// and blaming the caller for this one would be wrong. `aabb.hpp`'s file comment
/// still says the class throws `std::invalid_argument`; that remains true of every
/// reachable path and of all of its actual validation.
///
/// ## The toolchain floor this file carries
///
/// `format_repr` needs the **floating-point** overload of `std::to_chars`, which
/// libstdc++ 10 does not provide -- it defines no `__cpp_lib_to_chars`, and the
/// facility arrived in libstdc++ 11. That is FELIGN/pantr#376, whose specification
/// is a configure-time probe saying so rather than a reimplementation, so the call
/// stays. Collecting the rule here is what turns "some header somewhere needs it"
/// into one file a probe can name, and `format_general` and `format_fixed`
/// deliberately go through `snprintf` instead, which the floor does have.

#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>

#include "pantr/core/scalar.hpp"

namespace pantr::detail {

/// The widest a rendered `double` can be here, with room to spare.
///
/// A shortest-round-trip `double` needs at most 17 significant digits, a sign, a
/// point and a four-character exponent; `%.Nf` of a value near `DBL_MAX` needs
/// about 310 digits before the point. The `errc` and truncation checks below are
/// what make this a bound rather than an assumption.
inline constexpr std::size_t kFormatBufferSize = 512;

/// Python's spelling of a value `snprintf` would spell differently, if this is one.
///
/// \param v The value to render.
/// \return The rendering, or nothing at all when `v` is finite.
[[nodiscard]] inline std::optional<std::string> format_non_finite(double v) {
    // CPython prints a NaN **without a sign, ever**, at every format specifier, even
    // for one whose sign bit is set. glibc's `printf` prints `-nan` for that value.
    // So every spelling below has to intercept the non-finite cases rather than hand
    // them to `snprintf`, and this is not theoretical: a knot vector holding an
    // infinity gives `tol / ulp == inf / nan`, whose NaN came out negative on this
    // platform, and the two backends' messages then differed by one character.
    if (std::isnan(v)) {
        return "nan";
    }
    if (std::isinf(v)) {
        return v < 0.0 ? "-inf" : "inf";
    }
    return std::nullopt;
}

/// Render a `double` the way Python's `repr` renders a float.
///
/// Two rules have to agree, and conflating them is how this went wrong once
/// already. The **digits** are the shortest sequence that round-trips, which is
/// what `std::to_chars` produces and what Python's `repr` uses. The **notation**
/// is a separate decision, and the two languages make it differently: bare
/// `std::to_chars` picks whichever spelling is textually shorter, so `100000.0`
/// comes out as `1e+05`, while Python decides positionally -- fixed notation when
/// the decimal exponent lands in `[-4, 16)`, scientific otherwise. Those rules
/// coincide on typical magnitudes and diverge on round numbers, which is why the
/// first version passed every value the tests happened to carry.
///
/// So the exponent is obtained first, in scientific form, and the notation is then
/// chosen by Python's rule and rendered explicitly.
///
/// \param v The value to render.
/// \return Its Python-style textual form.
/// \throws std::logic_error If the value could not be rendered, which the buffer
///         size makes unreachable and which is checked rather than assumed.
[[nodiscard]] inline std::string format_repr(double v) {
    if (const std::optional<std::string> special = format_non_finite(v)) {
        return *special;
    }

    std::array<char, kFormatBufferSize> buffer{};
    auto written = std::to_chars(buffer.data(), buffer.data() + buffer.size(), v,
                                 std::chars_format::scientific);
    if (written.ec != std::errc{}) {
        throw std::logic_error("pantr: could not render a floating-point value.");
    }
    const std::string sci(buffer.data(), written.ptr);

    // `to_chars`'s scientific form is always `d[.ddd]e[+-]dd`, so the exponent is
    // whatever follows the one `e`.
    const std::size_t e_pos = sci.find('e');
    const int exponent = std::stoi(sci.substr(e_pos + 1));

    // Python: fixed notation while the decimal point sits inside the digits or just
    // before them, scientific once it has moved far enough either way.
    constexpr int kFixedLowerExponent = -5;
    constexpr int kFixedUpperExponent = 16;
    if (exponent <= kFixedLowerExponent || exponent >= kFixedUpperExponent) {
        return sci;
    }

    written = std::to_chars(buffer.data(), buffer.data() + buffer.size(), v,
                            std::chars_format::fixed);
    if (written.ec != std::errc{}) {
        throw std::logic_error("pantr: could not render a floating-point value.");
    }
    std::string text(buffer.data(), written.ptr);
    if (text.find('.') == std::string::npos) {
        text += ".0";
    }
    return text;
}

/// Render a scalar the way Python's `repr` renders a float.
///
/// Tier B: the rendering is a function of the value, so it goes through
/// `value_of` rather than touching the scalar directly.
///
/// \param x The value to render.
/// \return Its Python-style textual form.
template <class T>
[[nodiscard]] inline std::string format_scalar(const T& x) {
    using pantr::value_of;
    return format_repr(static_cast<double>(value_of(x)));
}

/// Render a `double` the way Python's `f"{v:.Ng}"` renders a float.
///
/// \param v The value to render.
/// \param precision The `N` of the format specifier; at least 1, as Python's is.
/// \return The rendered value.
/// \throws std::logic_error If the value could not be rendered.
[[nodiscard]] inline std::string format_general(double v, int precision) {
    if (const std::optional<std::string> special = format_non_finite(v)) {
        return *special;
    }
    std::array<char, kFormatBufferSize> buffer{};
    const int written =
        std::snprintf(buffer.data(), buffer.size(), "%.*g", precision, v);  // NOLINT
    if (written < 0 || static_cast<std::size_t>(written) >= buffer.size()) {
        throw std::logic_error("pantr: could not render a floating-point value.");
    }
    return {buffer.data(), static_cast<std::size_t>(written)};
}

/// Render a `double` the way Python's `f"{v:.Nf}"` renders a float.
///
/// \param v The value to render.
/// \param precision The `N` of the format specifier; may be 0.
/// \return The rendered value.
/// \throws std::logic_error If the value could not be rendered.
[[nodiscard]] inline std::string format_fixed(double v, int precision) {
    if (const std::optional<std::string> special = format_non_finite(v)) {
        return *special;
    }
    std::array<char, kFormatBufferSize> buffer{};
    const int written =
        std::snprintf(buffer.data(), buffer.size(), "%.*f", precision, v);  // NOLINT
    if (written < 0 || static_cast<std::size_t>(written) >= buffer.size()) {
        throw std::logic_error("pantr: could not render a floating-point value.");
    }
    return {buffer.data(), static_cast<std::size_t>(written)};
}

}  // namespace pantr::detail
