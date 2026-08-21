#pragma once

/// \file
/// Exact-integer binomial coefficients, the C++ twin of
/// `pantr.bspline._bspline_degree_core._bincoeff`.
///
/// ## Why this lives in `core/` and not in `bezier/`
///
/// It is `bezier`'s degree elevation and Bernstein product that need it first,
/// but the Python original sits in `bspline` and `bspline`'s own port will want
/// the same function. Two implementations of one recurrence is how the two
/// diverge, and the divergence would be silent: both would agree on every small
/// argument and part company only near the envelope, where nobody looks.
///
/// ## The recurrence, and why it is exact
///
/// `C(m, i) = C(m - 1, i - 1) * m / i` run over the smaller of `k` and `n - k`.
/// The running product after step `i` is exactly `C(n - kk + i, i)`, an integer,
/// so every division is exact and the result carries no rounding at all. The
/// route through `lgamma` that this replaces is off by 380 at `(57, 28)`, since
/// a value above `2^53` cannot be recovered from logarithms, and where it starts
/// failing depends on the `libm` in use. Integer arithmetic makes the envelope a
/// property of the algorithm rather than of the platform.
///
/// ## The envelope, and how it differs from Python's
///
/// The largest intermediate is `C(n, k) * min(k, n - k)`, which fits in `int64`
/// for every `k` up to `n = 61`. That bound is shared with the Python kernel and
/// `kBincoeffMaxN` states it.
///
/// **The consequence of exceeding it is not shared.** Numba wraps on int64
/// overflow and returns a corrupted value; in C++ signed overflow is undefined
/// behaviour, so the same input is worse here than there. The two callers in
/// `bezier` are guarded on the Python side by `_check_bincoeff_envelope`, which
/// runs in Layer 2 before any kernel is entered, and `cpp/bindings/bezier.cpp`
/// re-checks it for a caller that reaches the extension directly. `bincoeff` is the
/// Layer 3 form and validates nothing, as Layer 3 does not.
///
/// A `checked_bincoeff` wrapper lived here briefly and was deleted as dead code. It
/// had no caller, and the binding had independently written the same check with a
/// different message -- which is the divergence this file's first paragraph warns
/// about, one layer up. The check now exists once.
///
/// Independently of the integer limit, the `double` return is lossless only
/// while `C(n, k) <= 2^53`, i.e. up to `n = 56`; `C(57, 28)` is the first past
/// it. Between 57 and 61 the exact integer is computed and then correctly
/// rounded, which is the most a `double` return can carry. Every caller consumes
/// these only inside a floating-point *ratio* of binomials, so a correctly
/// rounded operand is all a ratio can use.

#include <cstdint>

namespace pantr::core {

/// Largest upper index for which the exact-integer recurrence cannot overflow.
///
/// Mirrors `pantr.bspline._bspline_degree_core._BINCOEFF_MAX_N`. Above this the
/// Python kernel wraps silently and this one is undefined; neither is usable.
inline constexpr int kBincoeffMaxN = 61;

/// Largest upper index for which every `C(n, k)` is representable in a `double`.
///
/// Between this and `kBincoeffMaxN` the integer is still exact and the cast
/// rounds once.
inline constexpr int kBincoeffExactDoubleMaxN = 56;

/// Compute `C(n, k)` in exact integer arithmetic, returning `0.0` outside `[0, n]`.
///
/// \param n Upper index. Must satisfy `n <= kBincoeffMaxN`.
/// \param k Lower index.
/// \return `C(n, k)` as a `double`, exact for `n <= kBincoeffExactDoubleMaxN` and
///         correctly rounded above it.
///
/// \note No input validation is performed. Passing `n > kBincoeffMaxN` is
///       undefined behaviour, not a wrong answer. The caller establishes the
///       envelope; `cpp/bindings/bezier.cpp`'s `require_bincoeff_envelope` is where
///       the library does so.
[[nodiscard]] constexpr double bincoeff(int n, int k) noexcept {
    if (k < 0 || k > n) {
        return 0.0;
    }
    const int kk = (k < n - k) ? k : n - k;
    std::int64_t result = 1;
    for (int i = 1; i <= kk; ++i) {
        result = result * static_cast<std::int64_t>(n - kk + i) / static_cast<std::int64_t>(i);
    }
    return static_cast<double>(result);
}

}  // namespace pantr::core
