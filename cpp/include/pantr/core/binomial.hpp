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
/// runs in Layer 2 before any kernel is entered. `checked_bincoeff` below is for
/// a C++ caller that has no such guard above it; `bincoeff` is the Layer 3 form
/// and validates nothing, as Layer 3 does not.
///
/// Independently of the integer limit, the `double` return is lossless only
/// while `C(n, k) <= 2^53`, i.e. up to `n = 56`; `C(57, 28)` is the first past
/// it. Between 57 and 61 the exact integer is computed and then correctly
/// rounded, which is the most a `double` return can carry. Every caller consumes
/// these only inside a floating-point *ratio* of binomials, so a correctly
/// rounded operand is all a ratio can use.

#include <cstdint>
#include <stdexcept>
#include <string>

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
///       undefined behaviour, not a wrong answer. For general use call
///       `checked_bincoeff` instead.
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

/// `bincoeff` with the envelope enforced, for a caller with no Layer 2 above it.
///
/// \param n Upper index.
/// \param k Lower index.
/// \param what Description of the requested operation, used in the message.
/// \return `C(n, k)`, as `bincoeff`.
/// \throws std::invalid_argument If `n` exceeds `kBincoeffMaxN`.
[[nodiscard]] inline double checked_bincoeff(int n, int k, const char* what) {
    if (n > kBincoeffMaxN) {
        throw std::invalid_argument(
            std::string(what) + " needs binomial coefficients up to C(" + std::to_string(n) +
            ", k), beyond the largest upper index " + std::to_string(kBincoeffMaxN) +
            " that pantr's exact-integer binomial kernel can compute without an int64 "
            "overflow.");
    }
    return bincoeff(n, k);
}

}  // namespace pantr::core
