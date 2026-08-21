#pragma once

/// \file
/// Ratio-recurrence tabulation of the Bernstein basis on `[0, 1]`.
///
/// Port of `_bernstein_point`, `_bernstein_point_no_mirror` and
/// `_tabulate_Bernstein_basis_1D_core` in `src/pantr/basis/_basis_core.py`,
/// which stay as the parity oracle. The recurrence, the midpoint branch and the
/// per-batch dispatch between the two point kernels are unchanged.
///
/// ## Two point kernels, dispatched once per batch
///
/// `B_0 = (1-u)^n` then `B_i = B_{i-1} * ((n-i+1)/i) * (u/(1-u))` is the whole
/// method. Its seed underflows once `u` is close enough to 1 at high degree, and
/// because every later term is a positive multiple of the previous one the whole
/// row then stays zero -- including `B_n`, whose true value is near 1. So the
/// general kernel mirrors: for `u > 1/2` it runs the same ascending recurrence on
/// `1-u`, which by `B_i,n(u) = B_{n-i},n(1-u)` fills the row backwards, and
/// reverses it. That bounds the seed below by `2^-n`.
///
/// The mirror costs a branch that most callers can prove they never need, so the
/// oracle keeps an unmirrored twin and picks between them **once per call**. This
/// port keeps that structure, because dispatching per point instead would be a
/// different kernel with the same answer and a different cost profile, and the
/// benchmark would then not be comparing the same thing.
///
/// ## The safe degree is derived from the format, not tabulated
///
/// The oracle hard-codes 20 for float64 and 6 for float32, each with its
/// derivation in the docstring. Here the same two numbers come out of
/// `std::numeric_limits`, so a hypothetical third format cannot silently inherit
/// a limit that was computed for another one.
///
/// The smallest positive `1 - u` for a representable `u < 1` is `2^-digits`
/// (`2^-53`, `2^-24`), and the smallest subnormal is `2^(min_exponent-digits)`
/// (`2^-1074`, `2^-149`). The seed `(1-u)^n` reaches exact zero once
/// `digits * n > digits - min_exponent`, so the largest safe degree is the
/// integer quotient of the two: `1074/53 = 20` and `149/24 = 6`, matching the
/// oracle exactly.
///
/// ## Where the two backends can disagree, and where they cannot
///
/// **No `a * b + c` anywhere.** The inner step is `(prev * const) * ratio`, three
/// multiplications and no addition, so unlike the cardinal B-spline kernel there
/// is no site for `-ffp-contract` to fuse. The ISA ladder of design/simd.md
/// cannot move a single value here.
///
/// **The recurrence rounds through `out`, and that is deliberate.** The oracle
/// reads `out_row[i - 1]` back out of a float32 array before forming the next
/// term, so at float32 every step rounds to float32 and the running value is
/// never carried at full width. Keeping `prev` in an `Acc` register here would be
/// the natural C++ and would disagree with the oracle by a growing multiple of
/// one float32 ulp along the row. The reads below are therefore written through
/// `out` on purpose; this is the one place where a faithful port is the less
/// obvious code.
///
/// **`pow` is the open question.** The seed is the one transcendental call, and
/// numba's `np.power` and the platform libm's `pow` are two implementations of
/// the same function with no shared guarantee. Nothing here claims they agree.
/// tests/parity/test_basis_bernstein.py measures it and states what it found.

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>

#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr {

/// Largest degree at which the unmirrored Bernstein recurrence cannot underflow.
///
/// See the file comment for the derivation. Returns the value for `T`'s
/// underlying floating-point format: 20 for `double`, 6 for `float`.
///
/// \tparam T Scalar type the kernel is instantiated on.
/// \return The largest safe degree, as an `int`.
template <Real T>
[[nodiscard]] constexpr int bernstein_max_safe_degree_no_mirror() noexcept {
    using L = std::numeric_limits<value_type_t<T>>;
    static_assert(L::is_specialized, "Bernstein needs the format's exponent range");
    return (L::digits - L::min_exponent) / L::digits;
}

/// Tabulate the Bernstein basis of `degree` at every point of `points`.
///
/// Evaluates the `degree + 1` polynomials `B_i,n` on `[0, 1]` by the O(n) ratio
/// recurrence, mirroring about `u = 1/2` when the degree is high enough for the
/// unmirrored seed to underflow.
///
/// \param degree Degree of the basis. Must be non-negative.
/// \param points Evaluation points. Values outside `[0, 1]` are evaluated by the
///        same recurrence rather than clamped, matching the Python kernel.
/// \param out Output view of shape `(points.size(), degree + 1)`, written in
///        full.
///
/// \note No input validation is performed. This is a Layer 3 kernel in the
///       layering of CLAUDE.md. Two preconditions are load-bearing for memory
///       safety rather than for the answer: `degree >= 0`, since it is widened to
///       `std::size_t` and a negative value becomes a loop bound of `SIZE_MAX`;
///       and `out` having exactly `points.size()` rows and `degree + 1` columns,
///       since nothing here bounds-checks a write. Non-finite points are not a
///       precondition -- they propagate into non-finite outputs, which is an
///       answer rather than undefined behaviour.
///       For general use call `pantr.basis.tabulate_bernstein_1d`.
template <Real T>
void tabulate_bernstein_1d(int degree, std::span<const T> points, span2d<T> out) {
    using Acc = accumulator_t<T>;
    using std::pow;

    const std::size_t num_pts = points.size();

    if (degree == 0) {
        for (std::size_t j = 0; j < num_pts; ++j) {
            at(out, j, 0) = T(1);
        }
        return;
    }

    const auto n = static_cast<std::size_t>(degree);
    const Acc n_acc = Acc(static_cast<double>(degree));
    const bool mirror = degree > bernstein_max_safe_degree_no_mirror<T>();

    for (std::size_t j = 0; j < num_pts; ++j) {
        const Acc u = static_cast<Acc>(points[j]);

        if (!mirror) {
            // The unmirrored twin. `u == 1` is special-cased rather than
            // mirrored, exactly as the oracle does: the ratio would be a
            // division by zero otherwise.
            if (u == Acc(1)) {
                for (std::size_t i = 0; i <= n; ++i) {
                    at(out, j, i) = T(0);
                }
                at(out, j, n) = T(1);
                continue;
            }
            const Acc one_minus_u = Acc(1) - u;
            at(out, j, 0) = static_cast<T>(pow(one_minus_u, n_acc));
            const Acc ratio = u / one_minus_u;
            for (std::size_t i = 1; i <= n; ++i) {
                const Acc i_acc = Acc(static_cast<double>(i));
                const Acc const_factor = (n_acc - i_acc + Acc(1)) / i_acc;
                // Read back through `out`, not from a register: see the file
                // comment.
                at(out, j, i) = static_cast<T>(static_cast<Acc>(at(out, j, i - 1)) *
                                               const_factor * ratio);
            }
            continue;
        }

        // The mirrored kernel. Above the midpoint the recurrence is run on
        // `1 - u`, which fills the row in reverse, and the row is then reversed
        // back. `u == 1` needs no special case here: the seed is `1^n == 1` and
        // every step multiplies by `(1-u)/u == 0`, so the row comes out as
        // `[1, 0, ..., 0]` and reads back correctly reversed.
        //
        // The two branches are spelled out rather than folded into a common
        // `base`/`ratio` pair. Folding them looks like the same arithmetic and is
        // not: below the midpoint the oracle forms `u / (1 - u)`, and recovering
        // `u` as `1 - (1 - u)` loses it outright for small `u` -- at `u = 1e-20`
        // that round trip returns exactly zero.
        const bool above_midpoint = u > Acc(0.5);
        const Acc one_minus_u = Acc(1) - u;
        const Acc base = above_midpoint ? u : one_minus_u;
        const Acc ratio = above_midpoint ? one_minus_u / u : u / one_minus_u;

        at(out, j, 0) = static_cast<T>(pow(base, n_acc));
        for (std::size_t i = 1; i <= n; ++i) {
            const Acc i_acc = Acc(static_cast<double>(i));
            const Acc const_factor = (n_acc - i_acc + Acc(1)) / i_acc;
            at(out, j, i) =
                static_cast<T>(static_cast<Acc>(at(out, j, i - 1)) * const_factor * ratio);
        }

        if (above_midpoint) {
            std::size_t lo = 0;
            std::size_t hi = n;
            while (lo < hi) {
                const T tmp = at(out, j, lo);
                at(out, j, lo) = at(out, j, hi);
                at(out, j, hi) = tmp;
                ++lo;
                --hi;
            }
        }
    }
}

}  // namespace pantr
