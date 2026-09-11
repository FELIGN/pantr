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
/// behaviour, so the same input is worse here than there. The callers in
/// `bezier` and `bspline` are guarded on the Python side by
/// `_check_bincoeff_envelope`, which runs in Layer 2 before any kernel is entered,
/// and `require_bincoeff_envelope` below re-checks it for a caller that reaches the
/// C++ core directly. `bincoeff` is the Layer 3 form and validates nothing, as Layer
/// 3 does not.
///
/// A `checked_bincoeff` wrapper lived here briefly and was deleted as dead code. It
/// had no caller, and the binding had independently written the same check with a
/// different message -- which is the divergence this file's first paragraph warns
/// about, one layer up. `require_bincoeff_envelope` is that check rewritten with
/// callers: `pantr/bezier/degree.hpp`, `pantr/bezier/product.hpp` and
/// `pantr/bspline/degree.hpp` share it, and it carries the oracle's message verbatim.
/// `cpp/bindings/bezier.cpp` still phrases its own refusal differently for the two
/// Bézier entry points a Python caller reaches directly; that one is a binding
/// concern rather than a second answer to this question.
///
/// ## The other answer to the same question, and why both are here
///
/// `bincoeff` **guards** the envelope rather than reproducing numba's wrap, and that
/// is right for it: its callers can refuse an out-of-envelope degree in Layer 2, where
/// the refusal is shared by both backends. `wrapping_mul` below is the opposite answer,
/// and it exists because one port cannot take the first one. The factorial scaling of
/// Piegl & Tiller A2.3 -- `pantr/basis/bernstein.hpp` and
/// `pantr/bspline/tabulate.hpp` -- accumulates `degree!/(degree-k)!` in an int64 that
/// wraps from degree 21, and *nothing in Layer 2 refuses that call today*. Making C++
/// refuse where the oracle returns a number would make `PANTR_BACKEND` change what the
/// library accepts, which `pantr.basis._basis_backend` states as a rule rather than a
/// preference. So that port reproduces the wrap, and it needs a spelling of it that is
/// not undefined behaviour.
///
/// Both live here for this file's own opening reason: two headers in two packages need
/// the same integer arithmetic, and two copies of it is how they diverge silently.
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
///       undefined behaviour, not a wrong answer. The caller establishes the
///       envelope; `require_bincoeff_envelope` below is where the library does so.
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

/// Multiply two signed 64-bit integers, wrapping rather than overflowing.
///
/// The product is formed in `std::uint64_t`, where the wrap is specified, and converted
/// back -- a conversion C++20 [conv.integral]/3 defines as modular for every value,
/// where signed overflow would be undefined behaviour. That reproduces exactly what a
/// numba int64 accumulator does.
///
/// See the file comment for why one port reproduces the wrap while `bincoeff` guards
/// against it.
///
/// \param a Left operand.
/// \param b Right operand.
/// \return The low 64 bits of the product, read as a signed value.
[[nodiscard]] constexpr std::int64_t wrapping_mul(std::int64_t a, std::int64_t b) noexcept {
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(a) *
                                     static_cast<std::uint64_t>(b));
}

/// Refuse an upper index the exact-integer recurrence cannot reach.
///
/// The message is the oracle's, character for character
/// (`pantr.bspline._bspline_degree_core._check_bincoeff_envelope`), because the parity
/// suites compare it.
///
/// It lives here rather than beside either caller for this file's opening reason: two
/// packages need the same refusal, and the message has to match the oracle's exactly, so
/// two copies of it is how the two come to refuse the same degree in different words.
/// `cpp/bindings/bezier.cpp` still carries a third spelling with its own wording, for the
/// Bézier entry points a Python caller reaches directly.
///
/// \param n Largest upper index the computation will need.
/// \param what Description of the operation, opening the message.
/// \throws std::invalid_argument If `n` exceeds `kBincoeffMaxN`.
inline void require_bincoeff_envelope(std::int64_t n, const std::string& what) {
    if (n > kBincoeffMaxN) {
        throw std::invalid_argument(
            what + " needs binomial coefficients up to C(" + std::to_string(n)
            + ", k), beyond the largest upper index " + std::to_string(kBincoeffMaxN)
            + " that pantr's exact-integer binomial kernel can compute without an int64 "
              "overflow. Past that the coefficients wrap silently and the result is "
              "corrupted rather than merely inaccurate.");
    }
}

}  // namespace pantr::core
