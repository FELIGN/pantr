#pragma once

/// \file
/// Ratio-recurrence tabulation of the Bernstein basis on `[0, 1]`.
///
/// Port of `_bernstein_point`, `_bernstein_point_no_mirror`,
/// `_tabulate_Bernstein_basis_1D_core`, `_bernstein_derivs_point` and
/// `_tabulate_Bernstein_basis_deriv_1D_core` in `src/pantr/basis/_basis_core.py`,
/// which stay as the parity oracle. The recurrence, the midpoint branch and the
/// per-batch dispatch between the two point kernels are unchanged.
///
/// **The values and the derivatives come from two different algorithms here, as they
/// do in the oracle.** `tabulate_bernstein_1d` is the O(n) ratio recurrence;
/// `tabulate_bernstein_deriv_1d` is Piegl & Tiller A2.3 specialised to unit knot
/// spans, and its row 0 is the Cox-de Boor triangle's values rather than the ratio
/// recurrence's. The two agree only to a rounding, and unifying them would answer
/// differently from both oracles.
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
/// tests/parity/test_basis_tabulations.py measures it and states what it found:
/// bitwise on every argument tested, which makes this claim observed rather than
/// derived, unlike the Legendre kernel's.


/// \note **What "no validation" means here, and it is not what it means in the oracle.**
/// These kernels are transliterations of Numba kernels whose docstrings use this same
/// sentence, where a violated precondition yields a *defined wrong answer*: numpy
/// indexes negatively and a Python integer does not overflow. On this side the same
/// violation is undefined behaviour. So the obligations are of two kinds, and the code
/// says which:
///
/// - a **correctness** obligation is documented and not asserted; violating it gives a
///   wrong answer in both backends, which is what the sentence above promises;
/// - a **memory-safety** obligation carries `PANTR_PRECONDITION`, from
///   `pantr/core/precondition.hpp`. Grep for it to see every one in this file.
///
/// The macro is `assert`, so it costs nothing in a release build. The bindings under
/// `cpp/bindings/` refuse all of these before a Python caller can express them; the
/// macro is for the C++ caller who includes this header directly.

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "pantr/core/binomial.hpp"
#include "pantr/core/precondition.hpp"
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
    // Memory safety. A negative `degree` becomes an enormous `std::size_t`
    // below, and the loops then write past `out`: measured, a heap-buffer
    // overflow WRITE, where the oracle returns a wrong answer and no more.
    PANTR_PRECONDITION(degree >= 0, "degree must be non-negative");
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

/// Tabulate the Bernstein basis and its derivatives to order `n_deriv` on `[0, 1]`.
///
/// Piegl & Tiller A2.3 specialised to the Bernstein case, where every knot difference
/// is one: the `ndu` table's lower triangle is all ones, so the three divisions of the
/// general recursion disappear and the `a` table's steps become plain differences.
///
/// ## Widths, measured rather than read
///
/// Every intermediate is `T`; `accumulator_t` is deliberately unused, for the reason
/// `pantr/bspline/tabulate.hpp`'s file comment gives at length. The one wider site is
/// the factorial scaling, where numba's `float32 * int64` promotion forms the product
/// in `double` and rounds once on the store.
/// `scripts/measure_bspline_tabulation_widths.py` measures both, three rival models
/// against the kernel: at `float32` over 50 284 values the narrow-intermediate,
/// wide-scaling model reproduces every one, narrowing the scaling instead fails on
/// 5 599, and running the recursion in `double` fails on 29 707. All three coincide at
/// `float64`, which is why the measurement is at `float32`.
///
/// ## The factorial accumulator wraps
///
/// `degree!/(degree-k)!` is accumulated in an int64 that wraps, first at degree 21 and
/// derivative order 19. Above that the scaling is a different number in both backends
/// and neither is right; `pantr::core::wrapping_mul` is the defined-behaviour spelling
/// of the wrap, and its own comment says why this port reproduces rather than guards.
///
/// ## No `a * b + c`, so nothing here fuses
///
/// Unlike the general-knot A2.3 of `pantr/bspline/tabulate.hpp`, whose `ndu` entries
/// are knot differences, this one's are one -- so `saved + (one - s) * temp` is the
/// only candidate site and `d + a * ndu` reduces to `d + a` where `ndu` is a diagonal
/// one. That is *not* enough to claim the kernel is contraction-free: `ndu[rk, pk]` is
/// a basis value off the diagonal in general, so the accumulation does carry real
/// products. The parity claim therefore takes the same conditional form as the
/// general-knot one, and does not assert a stronger property than
/// `tabulate_bernstein_1d`'s.
///
/// \tparam T Scalar type the kernel is instantiated on.
/// \param degree Degree of the basis. Must be non-negative.
/// \param n_deriv Highest derivative order. Must be non-negative. Rows above `degree`
///        come out identically zero, the k-th derivative of a degree-p polynomial
///        vanishing for `k > p`.
/// \param points Evaluation points. Values outside `[0, 1]` are evaluated by the same
///        recursion rather than clamped, matching the Python kernel.
/// \param out Output view of shape `(points.size(), n_deriv + 1, degree + 1)`, written
///        in full.
///
/// \note No input validation is performed. This is a Layer 3 kernel. `degree >= 0` and
///       `n_deriv >= 0` are memory-safety obligations, both being widened to
///       `std::size_t` below, as are `out`'s three extents.
///       For general use call
///       `pantr.bspline.BsplineSpace1D.tabulate_basis_derivatives` on a space with
///       Bézier-like knots.
template <Real T>
void tabulate_bernstein_deriv_1d(int degree, int n_deriv, std::span<const T> points,
                                 span_nd<T, 3> out) {
    PANTR_PRECONDITION(degree >= 0, "degree must be non-negative");
    PANTR_PRECONDITION(n_deriv >= 0, "n_deriv must be non-negative");
    PANTR_PRECONDITION(out.extent(0) == points.size() &&
                           out.extent(1) == static_cast<std::size_t>(n_deriv) + 1 &&
                           out.extent(2) == static_cast<std::size_t>(degree) + 1,
                       "out must have shape (points.size(), n_deriv+1, degree+1)");
    using pantr::value_of;

    const auto order = static_cast<std::size_t>(degree) + 1;
    const auto rows = static_cast<std::size_t>(n_deriv) + 1;
    const T zero(0.0);
    const T one(1.0);

    std::vector<T> ndu_storage(order * order, zero);
    std::vector<T> a_storage(2 * rows, zero);
    const span2d<T> ndu(ndu_storage.data(), order, order);
    const span2d<T> a(a_storage.data(), std::size_t{2}, rows);

    // The falling factorials, wrapping as the oracle's int64 accumulator does. Hoisted
    // out of the point loop because they depend on the degree alone; the oracle
    // recomputes them per point, which is the same sequence of integer operations and
    // so the same values.
    std::vector<std::int64_t> factorial(rows, 0);
    {
        std::int64_t fac = degree;
        for (std::size_t k = 1; k < rows; ++k) {
            factorial[k] = fac;
            fac = core::wrapping_mul(fac, degree - static_cast<std::int64_t>(k));
        }
    }

    for (std::size_t p = 0; p < points.size(); ++p) {
        const T s = points[p];

        // The oracle zeroes this point's output slice, the `ndu` table and the `a`
        // table on entry. `a` is then carried across the loop over `r` -- only
        // `a[0, 0]` is reset per `r` -- and that is reproduced rather than tidied,
        // since zeroing it per `r` would be a different computation wherever a stale
        // entry is read.
        for (std::size_t k = 0; k < rows; ++k) {
            for (std::size_t i = 0; i < order; ++i) {
                at(out, p, k, i) = zero;
            }
            at(a, 0, k) = zero;
            at(a, 1, k) = zero;
        }
        for (std::size_t i = 0; i < order; ++i) {
            for (std::size_t j = 0; j < order; ++j) {
                at(ndu, i, j) = zero;
            }
        }

        // --- The ndu table, with every knot difference one ---
        at(ndu, 0, 0) = one;
        for (std::size_t j = 1; j < order; ++j) {
            T saved = zero;
            for (std::size_t r = 0; r < j; ++r) {
                at(ndu, j, r) = one;                // the knot difference, one throughout
                const T temp = at(ndu, r, j - 1);   // divided by that one
                at(ndu, r, j) = saved + (one - s) * temp;
                saved = s * temp;
            }
            at(ndu, j, j) = saved;
        }

        for (std::size_t j = 0; j < order; ++j) {
            at(out, p, 0, j) = at(ndu, j, static_cast<std::size_t>(degree));
        }

        // --- The k-th derivatives, by the triangular recursion ---
        for (std::int64_t r = 0; r < static_cast<std::int64_t>(order); ++r) {
            std::size_t s1 = 0;
            std::size_t s2 = 1;
            at(a, 0, 0) = one;

            for (std::int64_t k = 1; k <= n_deriv; ++k) {
                T d = zero;
                const std::int64_t rk = r - k;
                const std::int64_t pk = degree - k;

                if (r >= k) {
                    at(a, s2, 0) = at(a, s1, 0);  // divided by one
                    d = at(a, s2, 0) * at(ndu, rk, pk);
                }

                const std::int64_t j1 = rk >= -1 ? 1 : -rk;
                const std::int64_t j2 = (r - 1) <= pk ? k - 1 : degree - r;

                for (std::int64_t j = j1; j <= j2; ++j) {
                    at(a, s2, j) = at(a, s1, j) - at(a, s1, j - 1);  // divided by one
                    d = d + at(a, s2, j) * at(ndu, rk + j, pk);
                }

                if (r <= pk) {
                    at(a, s2, k) = -at(a, s1, k - 1);  // divided by one
                    d = d + at(a, s2, k) * at(ndu, r, pk);
                }

                at(out, p, static_cast<std::size_t>(k), static_cast<std::size_t>(r)) = d;
                std::swap(s1, s2);
            }
        }

        // --- The factorial scaling, formed in `double` and rounded on the store ---
        for (std::size_t k = 1; k < rows; ++k) {
            const auto fac_wide = static_cast<double>(factorial[k]);
            // **Widened for a plain scalar and not for a differentiable one, and the
            // branch is not an optimisation.** For `float` and `double` the wide form is
            // what parity requires: numba promotes `float32 * int64` to `float64`, so the
            // product is formed in `double` and rounded once on the store. But it reads
            // only the *value* through `value_of` and rebuilds `T` from a raw `double`,
            // which for a differentiable scalar constructs a zero-derivative value and so
            // discards everything steps 1 and 2 carried -- in the one kernel whose whole
            // job is derivatives. `pantr/core/scalar.hpp` says AD compatibility rides
            // along for free with the float32 discipline; here the two genuinely conflict,
            // so each gets the arithmetic it needs. The two branches agree exactly at
            // `double`, where the wide product IS the storage-width product; only `float`
            // separates them, and only `float` has an oracle to be faithful to.
            for (std::size_t j = 0; j < order; ++j) {
                if constexpr (std::same_as<T, value_type_t<T>>) {
                    at(out, p, k, j) =
                        T(static_cast<double>(value_of(at(out, p, k, j))) * fac_wide);
                } else {
                    at(out, p, k, j) = at(out, p, k, j) * T(fac_wide);
                }
            }
        }
    }
}

}  // namespace pantr
