#pragma once

/// \file
/// The sixteen root-finding kernels of `src/pantr/bezier/_root_finding_core.py`,
/// `_clipping_core.py`, `_yuksel_core.py` and `_batch_core.py`, which stay as the
/// parity oracle.
///
/// This is a **transliteration**, branch for branch and expression for expression,
/// and that is a decision rather than laziness. The block is iterative and every
/// verdict it reaches is discrete: how many roots, which intervals survive, which
/// hull vertices are kept. A cleaner reimplementation would have to be graded by a
/// set distance over a discrete verdict, with the Numba side as the only available
/// reference, which is the weakest parity claim in the tree. Reproducing the
/// arithmetic instead buys an equality, and an equality is checked rather than
/// argued.
///
/// ## Numba's `float()` does not widen, and three of five widths follow from that
///
/// `float(coeff[0])` in a `nopython` kernel reads as a promotion and is a no-op: a
/// `float32` stays a `float32` and so does every operation built on it. What widens
/// in Numba is type unification across assignments, so a variable seeded with `0.0`
/// or `float("inf")` and later given a `float32` is `float64` throughout. Six sites
/// in the oracle carry a `float()` a reader would take for a cast; none promotes.
///
/// `scripts/measure_root_finding_widths.py` measures all of this behaviourally, two
/// rival models per site against the kernel, and reports how often the two models
/// disagree so a match cannot come from a check that could not fail. It is the
/// specification this file is written against, and it is the thing to re-run rather
/// than the table below to trust.
///
/// | site | width | what a C++ author writes by default |
/// |---|---|---|
/// | de Casteljau triangle | accumulate `Acc`, **store `T`** | keeps the running value in a register |
/// | `d1 - d0` in the derivative | **`T`** | subtracts two `Acc` locals |
/// | hull orientation predicate | differences in **`T`**, product in `double` | templates the whole expression on `T` |
/// | Yuksel forward difference | subtract in **`T`**, widen on store | subtracts into the `double` destination |
/// | degree-1 base case `c0/(c0-c1)` | **`T`** | divides in `double` |
///
/// The last is the one that matters most. Writing `double root = c0 / (c0 - c1);`
/// breaks parity on **every** `float32` input (the `double` model matched none of
/// 20000 pairs) and the difference reads as round-off rather than as a defect. Every
/// Yuksel recursion bottoms out there.
///
/// And two sites look exactly like those and behave the opposite way, so none of
/// this is a per-kernel or per-module rule: `_batch_core.py:87` divides in `double`
/// and `_yuksel_core.py:309` divides the same shape of expression in `T`. What
/// separates them is whether some other assignment forced the variable wider.
///
/// ## Two library spellings, both measured rather than assumed
///
/// The one call in the whole block that leaves the four arithmetic operations is
/// `zero_tol ** (1/3)`, and it must be `std::pow(x, 1.0 / 3.0)`. `std::cbrt` differs
/// from Numba's `**` on 49659 of 50000 tolerances spanning `1e-300` to `1e3`, while
/// `std::pow` differs on none.
///
/// Minima and maxima must be `std::min` and `std::max`, whose ternary expansion
/// matches Numba's builtins on NaN, on signed zero and on infinities. `std::fmin`
/// and `std::fmax` return the non-NaN operand and would not.
///
/// ## Contraction can change a verdict here, not just a last bit
///
/// The hull orientation predicate is `a * b - c * d`. On an **exactly collinear**
/// control polygon it evaluates to exactly zero, so `cross >= 0` holds and Andrew's
/// monotone chain pops the vertex. With `-ffp-contract=on` and an FMA-capable target
/// the compiler may fuse it, that exact tie becomes a signed residue, and the
/// tie-break turns into a coin toss: which vertices survive changes, and so does the
/// clipped interval. Measured against the kernel, an unfused transliteration agrees
/// on all 4800 collinear cases and a fused one on 3789.
///
/// A collinear control polygon is not exotic. A degree-elevated linear polynomial,
/// an affine segment and a constant all produce one.
///
/// The shipped build carries no `-march`, so the baseline target has no fused
/// multiply-add and nothing to contract into, which is why the parity claim is an
/// equality there and bounded above it. That gate is `__fp_contract__`, read at run
/// time by the parity tests, not assumed here.
///
/// Rule 11 of design/backend_parity.md states all of this, including the bound that
/// applies where contraction is live: it is the algorithm's, not the arithmetic's,
/// and it is the larger of the bracketing tolerance and the interval where the
/// computed residual is indistinguishable from zero.
///
/// ## Two defects are reproduced on purpose
///
/// The oracle has two pre-existing wrong answers in this block, filed as
/// FELIGN/pantr#351 and #352, and this file reproduces both **deliberately**,
/// because the port's contract is parity and not correction. They are marked at
/// their sites. Fixing either means fixing both backends and the parity tests in one
/// change; fixing one backend alone breaks the equality this whole file exists to
/// support.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "pantr/core/mdspan.hpp"
#include "pantr/core/scalar.hpp"

namespace pantr {

/// Machine epsilon for IEEE 754 double precision.
///
/// Spelled as a literal rather than taken from `std::numeric_limits`, and `double`
/// whatever the coefficients are, mirroring `_root_finding_core._DBL_EPSILON`. Both
/// choices are the oracle's: the constant is `float64` there even on the `float32`
/// path, so deriving it from `T` here would change every tolerance in the block.
inline constexpr double kDblEpsilon = 2.2204460492503131e-16;

/// Iteration cap for the Newton/bisection hybrid. Mirrors `_yuksel_core._MAX_NEWTON_ITER`.
inline constexpr int kMaxNewtonIter = 64;

/// Relative interval reduction below which clipping subdivides instead of recursing.
///
/// Mirrors `_clipping_core._CLIP_REDUCTION_THRESHOLD`.
inline constexpr double kClipReductionThreshold = 0.2;

/// Recursion-depth cap for Bézier clipping. Mirrors `_clipping_core._CLIP_MAX_DEPTH`.
inline constexpr int kClipMaxDepth = 64;

/// Interval-stack capacity for Bézier clipping. Mirrors `_clipping_core._MAX_STACK_SIZE`.
inline constexpr int kMaxStackSize = 256;

/// Lowest degree at which Bézier clipping is considered. Mirrors `_batch_core._CLIP_MIN_DEGREE`.
inline constexpr int kClipMinDegree = 6;

/// Coefficient dynamic range above which clipping is declined. Mirrors
/// `_batch_core._CLIP_COEFF_RANGE_LIMIT`.
inline constexpr double kClipCoeffRangeLimit = 1e8;

namespace detail {

/// Evaluate a scalar Bernstein polynomial at one parameter, by de Casteljau.
///
/// The workspace carries the **coefficient** type while the parameter is `double`,
/// so each step computes at accumulator width and rounds once on the store. Hoisting
/// `work[i]` into an accumulator register would be the natural C++ and would drift
/// from the oracle by a growing multiple of one `T` ulp.
///
/// \tparam T Coefficient type.
/// \param coeff Bernstein coefficients, length `n + 1`.
/// \param t Parameter in [0, 1].
/// \param work Scratch of at least `coeff.size()` entries; contents are not read.
/// \return The polynomial value, at coefficient width, as the oracle returns it.
///
/// Mirrors `_root_finding_core._de_casteljau_eval_scalar`. No validation is
/// performed; the Python layer guarantees the shapes.
template <Real T>
T de_casteljau_eval_scalar(std::span<const T> coeff, accumulator_t<T> t, std::span<T> work) {
    const auto n = static_cast<std::ptrdiff_t>(coeff.size()) - 1;
    std::copy(coeff.begin(), coeff.end(), work.begin());
    for (std::ptrdiff_t k = 1; k <= n; ++k) {
        for (std::ptrdiff_t i = 0; i <= n - k; ++i) {
            work[static_cast<std::size_t>(i)] = static_cast<T>(
                (accumulator_t<T>{1} - t) * static_cast<accumulator_t<T>>(work[static_cast<std::size_t>(i)])
                + t * static_cast<accumulator_t<T>>(work[static_cast<std::size_t>(i) + 1]));
        }
    }
    return work[0];
}

/// Evaluate a scalar Bernstein polynomial and its derivative at one parameter.
///
/// Runs the triangle to the penultimate row, whose two entries give the derivative.
/// **`d1 - d0` is a `T` subtraction**, not a `double` one: the two entries are read
/// out of the narrow workspace and subtracted before the degree factor promotes the
/// product. Subtracting them as accumulators instead matched only 1543 of 2400
/// float32 cases.
///
/// \tparam T Coefficient type.
/// \param coeff Bernstein coefficients, length `n + 1`.
/// \param t Parameter in [0, 1].
/// \param work Scratch of at least `coeff.size()` entries; contents are not read.
/// \param out_deriv Receives the derivative.
/// \return The polynomial value.
///
/// Mirrors `_root_finding_core._de_casteljau_eval_and_deriv_scalar`. No validation is
/// performed.
template <Real T>
accumulator_t<T> de_casteljau_eval_and_deriv_scalar(std::span<const T> coeff, accumulator_t<T> t,
                                                    std::span<T> work,
                                                    accumulator_t<T>& out_deriv) {
    using Acc = accumulator_t<T>;
    const auto n = static_cast<std::ptrdiff_t>(coeff.size()) - 1;
    if (n == 0) {
        out_deriv = Acc{0};
        return static_cast<Acc>(coeff[0]);
    }

    const Acc s = Acc{1} - t;
    std::copy(coeff.begin(), coeff.end(), work.begin());
    for (std::ptrdiff_t k = 1; k < n; ++k) {
        for (std::ptrdiff_t i = 0; i <= n - k; ++i) {
            work[static_cast<std::size_t>(i)] =
                static_cast<T>(s * static_cast<Acc>(work[static_cast<std::size_t>(i)])
                               + t * static_cast<Acc>(work[static_cast<std::size_t>(i) + 1]));
        }
    }

    const T d0 = work[0];
    const T d1 = work[1];
    // Narrow on purpose: the oracle subtracts two workspace entries, not two doubles.
    const T narrow_difference = static_cast<T>(d1 - d0);
    out_deriv = static_cast<Acc>(n) * static_cast<Acc>(narrow_difference);
    return s * static_cast<Acc>(d0) + t * static_cast<Acc>(d1);
}

/// Restrict a scalar Bernstein polynomial to `[lower, upper]`, reparametrised to [0, 1].
///
/// Two-pass de Casteljau, the pass order chosen to avoid dividing by a small number.
/// The workspace is `double` whatever the coefficients are, so the widening happens
/// once on the way in and every step after it is a `double` step.
///
/// \tparam T Coefficient type.
/// \param coeff Bernstein coefficients, length `p + 1`.
/// \param lower Left bound in [0, 1).
/// \param upper Right bound in (0, 1].
/// \param out Receives `p + 1` restricted coefficients.
///
/// Mirrors `_root_finding_core._restrict_scalar`. No validation is performed.
template <Real T>
void restrict_scalar(std::span<const T> coeff, double lower, double upper, std::span<double> out) {
    using std::abs;
    const auto p = static_cast<std::ptrdiff_t>(coeff.size()) - 1;
    for (std::ptrdiff_t i = 0; i <= p; ++i) {
        out[static_cast<std::size_t>(i)] = static_cast<double>(coeff[static_cast<std::size_t>(i)]);
    }

    if (abs(upper) >= abs(lower - 1.0)) {
        const double tau = upper;
        for (std::ptrdiff_t step = 1; step <= p; ++step) {
            for (std::ptrdiff_t j = p; j >= step; --j) {
                const auto u = static_cast<std::size_t>(j);
                out[u] = out[u] * tau + out[u - 1] * (1.0 - tau);
            }
        }
        const double tau2 = upper != 0.0 ? lower / upper : 0.0;
        for (std::ptrdiff_t step = 1; step <= p; ++step) {
            for (std::ptrdiff_t j = 0; j <= p - step; ++j) {
                const auto u = static_cast<std::size_t>(j);
                out[u] = out[u] * (1.0 - tau2) + out[u + 1] * tau2;
            }
        }
    } else {
        const double tau = lower;
        for (std::ptrdiff_t step = 1; step <= p; ++step) {
            for (std::ptrdiff_t j = 0; j <= p - step; ++j) {
                const auto u = static_cast<std::size_t>(j);
                out[u] = out[u] * (1.0 - tau) + out[u + 1] * tau;
            }
        }
        const double tau2 = lower != 1.0 ? (upper - lower) / (1.0 - lower) : 0.0;
        for (std::ptrdiff_t step = 1; step <= p; ++step) {
            for (std::ptrdiff_t j = p; j >= step; --j) {
                const auto u = static_cast<std::size_t>(j);
                out[u] = out[u] * tau2 + out[u - 1] * (1.0 - tau2);
            }
        }
    }
}

/// Extract Bernstein coefficients for the sub-interval `[t_min, t_max]`.
///
/// \tparam T Coefficient type.
/// \param coeff Bernstein coefficients on [0, 1].
/// \param t_min Sub-interval start, clamped into [0, 1].
/// \param t_max Sub-interval end, clamped into [0, 1].
/// \param out Receives `coeff.size()` reparametrised coefficients.
///
/// Mirrors `_root_finding_core._subdivide_scalar`. No validation is performed.
template <Real T>
void subdivide_scalar(std::span<const T> coeff, double t_min, double t_max,
                      std::span<double> out) {
    if (t_min <= 0.0 && t_max >= 1.0) {
        for (std::size_t i = 0; i < coeff.size(); ++i) {
            out[i] = static_cast<double>(coeff[i]);
        }
        return;
    }
    // std::min/std::max, never fmin/fmax: the ternary expansion is what Numba's
    // builtins do on NaN and on signed zero.
    t_min = std::min(std::max(t_min, 0.0), 1.0);
    t_max = std::min(std::max(t_max, 0.0), 1.0);
    restrict_scalar<T>(coeff, t_min, t_max, out);
}

/// Count sign changes in a Bernstein coefficient sequence, ignoring exact zeros.
///
/// Comparisons only, so this kernel has no width at all and needs none of the care
/// the rest of the file does.
///
/// \tparam T Coefficient type.
/// \param coeff Bernstein coefficients.
/// \return The number of sign changes, which bounds the root count above by the
///     variation-diminishing property.
///
/// Mirrors `_root_finding_core._count_sign_changes`. No validation is performed.
template <Real T>
int count_sign_changes(std::span<const T> coeff) {
    int changes = 0;
    int previous = 0;
    for (std::size_t i = 0; i < coeff.size(); ++i) {
        const T v = coeff[i];
        int sign = 0;
        if (v > T{0}) {
            sign = 1;
        } else if (v < T{0}) {
            sign = -1;
        } else {
            continue;
        }
        if (previous != 0 && previous != sign) {
            ++changes;
        }
        previous = sign;
    }
    return changes;
}

/// Clip the parameter range using the convex hull of the control polygon.
///
/// Andrew's monotone chain over vertices `(i/n, c_i)`, whose x-coordinates are
/// already ordered so no sort is needed, then every hull edge is tested against
/// `y = 0`.
///
/// Two things about the orientation predicate. Its coefficient **differences run at
/// `T`** and only the integer factor promotes the product, which is numpy's
/// promotion of `int64` against `float32` and the opposite of what a C++ template
/// over `T` gives by default; widening the differences first matched only 3541 of
/// 10000 float32 triples. And on an exactly collinear polygon the predicate is
/// exactly zero, so `cross >= 0` holds and the vertex is popped: that exact tie is
/// what contraction destroys. See this file's opening note.
///
/// \tparam T Coefficient type. Reached with `double` from every call site in the
///     library, since the caller passes the widened sub-interval coefficients; the
///     `float` instantiation exists because the oracle accepts one.
/// \param coeff Bernstein coefficients, length `n + 1`.
/// \param chain Scratch of at least `n + 1` entries for the hull vertex indices.
/// \param t_lo Receives the lowest crossing parameter found, untouched if none is.
/// \param t_hi Receives the highest.
/// \return Whether any zero crossing was detected.
///
/// Mirrors `_root_finding_core._clip_hull_to_zero`. No validation is performed.
template <Real T>
bool clip_hull_to_zero(std::span<const T> coeff, std::span<std::int64_t> chain, double& t_lo,
                       double& t_hi) {
    const auto n = static_cast<std::ptrdiff_t>(coeff.size()) - 1;
    t_lo = 0.0;
    t_hi = 0.0;
    if (n < 1) {
        return false;
    }

    const double inv_n = 1.0 / static_cast<double>(n);
    t_lo = 1.0;
    t_hi = 0.0;
    bool found = false;

    // The oracle builds the upper hull and consumes it, then does the same for the
    // lower one, so a single scratch array serves both passes.
    for (const bool upper : {true, false}) {
        std::ptrdiff_t size = 0;
        for (std::ptrdiff_t i = 0; i <= n; ++i) {
            while (size >= 2) {
                const auto j0 = static_cast<std::ptrdiff_t>(chain[static_cast<std::size_t>(size) - 2]);
                const auto j1 = static_cast<std::ptrdiff_t>(chain[static_cast<std::size_t>(size) - 1]);
                // Differences at T, products at double: numpy promotes int64 against
                // float32 to float64, so only the index factors widen the product.
                const T rise = static_cast<T>(coeff[static_cast<std::size_t>(i)]
                                              - coeff[static_cast<std::size_t>(j0)]);
                const T edge = static_cast<T>(coeff[static_cast<std::size_t>(j1)]
                                              - coeff[static_cast<std::size_t>(j0)]);
                const double cross = static_cast<double>(j1 - j0) * static_cast<double>(rise)
                                     - static_cast<double>(edge) * static_cast<double>(i - j0);
                if (upper ? (cross >= 0.0) : (cross <= 0.0)) {
                    --size;
                } else {
                    break;
                }
            }
            chain[static_cast<std::size_t>(size)] = static_cast<std::int64_t>(i);
            ++size;
        }

        for (std::ptrdiff_t k = 0; k < size - 1; ++k) {
            const auto ia = static_cast<std::ptrdiff_t>(chain[static_cast<std::size_t>(k)]);
            const auto ib = static_cast<std::ptrdiff_t>(chain[static_cast<std::size_t>(k) + 1]);
            const auto da = static_cast<double>(coeff[static_cast<std::size_t>(ia)]);
            const auto db = static_cast<double>(coeff[static_cast<std::size_t>(ib)]);
            if (da * db < 0.0) {
                const double ta = static_cast<double>(ia) * inv_n;
                const double tb = static_cast<double>(ib) * inv_n;
                const double crossing = ta + (-da) / (db - da) * (tb - ta);
                t_lo = std::min(t_lo, crossing);
                t_hi = std::max(t_hi, crossing);
                found = true;
            }
            if (da == 0.0) {
                const double ta = static_cast<double>(ia) * inv_n;
                t_lo = std::min(t_lo, ta);
                t_hi = std::max(t_hi, ta);
                found = true;
            }
        }

        const auto last = static_cast<std::ptrdiff_t>(chain[static_cast<std::size_t>(size) - 1]);
        if (coeff[static_cast<std::size_t>(last)] == T{0}) {
            const double ta = static_cast<double>(last) * inv_n;
            t_lo = std::min(t_lo, ta);
            t_hi = std::max(t_hi, ta);
            found = true;
        }
    }
    return found;
}

/// Polish a root candidate with a single Newton step, accepted only if it improves.
///
/// \tparam T Coefficient type.
/// \param coeff Original Bernstein coefficients on [0, 1].
/// \param mid Initial estimate.
/// \param lo Left bound of the current interval.
/// \param hi Right bound.
/// \param param_tol Width of the acceptance neighbourhood around `[lo, hi]`.
/// \param work Scratch of at least `coeff.size()` entries.
/// \param out_f Receives the residual at the returned parameter.
/// \param out_df Receives the derivative at `mid`, whatever is returned.
/// \return The polished parameter, or `mid` unchanged.
///
/// Mirrors `_root_finding_core._newton_polish_scalar`. No validation is performed.
template <Real T>
accumulator_t<T> newton_polish_scalar(std::span<const T> coeff, accumulator_t<T> mid,
                                      accumulator_t<T> lo, accumulator_t<T> hi,
                                      accumulator_t<T> param_tol, std::span<T> work,
                                      accumulator_t<T>& out_f, accumulator_t<T>& out_df) {
    using std::abs;
    using Acc = accumulator_t<T>;
    Acc df_mid{};
    const Acc f_mid = de_casteljau_eval_and_deriv_scalar<T>(coeff, mid, work, df_mid);
    out_df = df_mid;
    if (abs(df_mid) > static_cast<Acc>(kDblEpsilon)) {
        Acc newton = mid - f_mid / df_mid;
        if (lo - param_tol <= newton && newton <= hi + param_tol) {
            newton = std::max(Acc{0}, std::min(Acc{1}, newton));
            const Acc f_newton = static_cast<Acc>(de_casteljau_eval_scalar<T>(coeff, newton, work));
            if (abs(f_newton) <= abs(f_mid)) {
                out_f = f_newton;
                return newton;
            }
        }
    }
    out_f = f_mid;
    return mid;
}

/// Find the unique root of a monotone Bernstein polynomial on [0, 1].
///
/// Yuksel's clamped Newton/bisection hybrid, seeded by false position clamped away
/// from the boundaries where the derivative may vanish.
///
/// **`f_lo` and `f_hi` are at `T`, not at accumulator width.** The oracle writes
/// `float(coeff[0])`, which does not widen in Numba, so the guard below, the
/// false-position denominator and the initial quotient are all `T` operations.
///
/// \tparam T Coefficient type.
/// \param coeff Bernstein coefficients of a monotone scalar polynomial.
/// \param param_tol Bracket-width termination tolerance.
/// \param work Scratch of at least `coeff.size()` entries.
/// \return The root parameter, or NaN when no sign change is detected.
///
/// Mirrors `_yuksel_core._solve_monotone_root_kernel`. No validation is performed.
template <Real T>
accumulator_t<T> solve_monotone_root_kernel(std::span<const T> coeff, accumulator_t<T> param_tol,
                                            std::span<T> work) {
    using std::abs;
    using Acc = accumulator_t<T>;
    Acc lo{0};
    Acc hi{1};

    const T f_lo = coeff[0];
    const T f_hi = coeff[coeff.size() - 1];

    // FELIGN/pantr#351, reproduced deliberately. This product is at T, so at float32
    // two same-sign coefficients below about 1e-23 multiply to zero, the guard does
    // not fire, and a root is returned for a polynomial that has none. Computing it
    // at Acc would be correct and would break parity with the oracle.
    if (f_lo * f_hi > T{0}) {
        return std::numeric_limits<Acc>::quiet_NaN();
    }

    Acc x{};
    if (abs(static_cast<Acc>(static_cast<T>(f_hi - f_lo))) > Acc{0}) {
        // Divide at T, then promote: `(-f_lo) / (f_hi - f_lo)` binds before the
        // multiplication by the double-valued span.
        const T quotient = static_cast<T>(static_cast<T>(-f_lo) / static_cast<T>(f_hi - f_lo));
        x = lo + static_cast<Acc>(quotient) * (hi - lo);
        const Acc margin = Acc{0.1} * (hi - lo);
        x = std::max(lo + margin, std::min(x, hi - margin));
    } else {
        x = Acc{0.5} * (lo + hi);
    }

    for (int iteration = 0; iteration < kMaxNewtonIter; ++iteration) {
        Acc dfx{};
        const Acc fx = de_casteljau_eval_and_deriv_scalar<T>(coeff, x, work, dfx);

        if (static_cast<Acc>(f_lo) * fx <= Acc{0}) {
            hi = x;
        } else {
            lo = x;
        }

        if ((hi - lo) <= param_tol) {
            break;
        }

        const Acc x_new = abs(dfx) > Acc{0} ? x - fx / dfx : Acc{0.5} * (lo + hi);
        x = (lo < x_new && x_new < hi) ? x_new : Acc{0.5} * (lo + hi);
    }

    return Acc{0.5} * (lo + hi);
}

/// Find one root of a Bernstein polynomial on `[lo, hi]` without reparametrising.
///
/// \tparam T Coefficient type.
/// \param coeff Bernstein coefficients on [0, 1].
/// \param lo Left boundary of the search interval.
/// \param hi Right boundary.
/// \param f_lo Pre-evaluated value at `lo`.
/// \param tol Bracket-width termination tolerance.
/// \param work Scratch of at least `coeff.size()` entries.
/// \return The root parameter in `[lo, hi]`.
///
/// Mirrors `_yuksel_core._solve_on_interval`. No validation is performed.
template <Real T>
accumulator_t<T> solve_on_interval(std::span<const T> coeff, accumulator_t<T> lo,
                                   accumulator_t<T> hi, accumulator_t<T> f_lo,
                                   accumulator_t<T> tol, std::span<T> work) {
    using std::abs;
    using Acc = accumulator_t<T>;
    Acc x = Acc{0.5} * (lo + hi);

    for (int iteration = 0; iteration < kMaxNewtonIter; ++iteration) {
        Acc dfx{};
        const Acc fx = de_casteljau_eval_and_deriv_scalar<T>(coeff, x, work, dfx);

        if (f_lo * fx <= Acc{0}) {
            hi = x;
        } else {
            lo = x;
        }

        if ((hi - lo) <= tol) {
            return Acc{0.5} * (lo + hi);
        }

        const Acc x_new = abs(dfx) > Acc{0} ? x - fx / dfx : Acc{0.5} * (lo + hi);
        x = (lo < x_new && x_new < hi) ? x_new : Acc{0.5} * (lo + hi);
    }

    return Acc{0.5} * (lo + hi);
}

/// Find every root by walking the monotone intervals the critical parameters define.
///
/// **`f_prev` and `f_curr` are at `T`.** They come from `float(coeff[i])`, which does
/// not widen, and from the evaluation kernel, which returns at coefficient width. So
/// the sign-change test below is a `T` product, and so is the scale that sizes
/// `boundary_eps`.
///
/// \tparam T Coefficient type.
/// \param coeff Bernstein coefficients, length `n + 1`.
/// \param crit Sorted critical parameters, of which the first `n_crit` are valid.
/// \param n_crit Number of valid critical parameters.
/// \param tol Bracket-width tolerance.
/// \param roots Receives at most `n` roots.
/// \param work Scratch of at least `coeff.size()` entries.
/// \return How many entries of `roots` are valid.
///
/// Mirrors `_yuksel_core._find_roots_at_level`. No validation is performed.
template <Real T>
int find_roots_at_level(std::span<const T> coeff, std::span<const double> crit, int n_crit,
                        accumulator_t<T> tol, std::span<double> roots, std::span<T> work) {
    using std::abs;
    using Acc = accumulator_t<T>;
    const auto n = static_cast<int>(coeff.size()) - 1;
    int count = 0;

    const T d_min = *std::min_element(coeff.begin(), coeff.end());
    const T d_max = *std::max_element(coeff.begin(), coeff.end());
    // The subtraction is at T: `float()` does not widen, so widening here would size
    // every boundary test differently. Measured, 6000 of 6000.
    const T scale = static_cast<T>(d_max - d_min);
    // FELIGN/pantr#352, reproduced deliberately. The 1e-30 floor is absolute inside
    // an otherwise scale-relative tolerance, so below it every coefficient of the
    // problem reads as zero and the endpoints are returned as roots. Rescaling the
    // same polynomial moves the answer, which is the defect.
    const Acc boundary_eps =
        std::max(static_cast<Acc>(abs(scale)) * static_cast<Acc>(kDblEpsilon) * Acc{8},
                 Acc{1e-30});

    Acc prev_t{0};
    T f_prev = coeff[0];

    for (int k = 0; k <= n_crit; ++k) {
        const Acc curr_t =
            k < n_crit ? static_cast<Acc>(crit[static_cast<std::size_t>(k)]) : Acc{1};

        const T f_at_curr =
            curr_t < Acc{1} ? de_casteljau_eval_scalar<T>(coeff, curr_t, work)
                            : coeff[static_cast<std::size_t>(n)];

        if (curr_t - prev_t < tol) {
            f_prev = f_at_curr;
            prev_t = curr_t;
            continue;
        }

        const T f_curr = f_at_curr;

        if (abs(static_cast<Acc>(f_prev)) <= boundary_eps
            && (count == 0
                || abs(roots[static_cast<std::size_t>(count) - 1] - prev_t) > tol)) {
            if (count < n) {
                roots[static_cast<std::size_t>(count)] = prev_t;
                ++count;
            }
            f_prev = f_curr;
            prev_t = curr_t;
            continue;
        }

        // FELIGN/pantr#351, reproduced deliberately: a T product, so at float32 two
        // opposite-sign values below about 1e-23 multiply to zero, no sign change is
        // seen, and a root that exists is lost.
        if (f_prev * f_curr < T{0}) {
            const Acc root = solve_on_interval<T>(coeff, prev_t, curr_t,
                                                  static_cast<Acc>(f_prev), tol, work);
            if (count < n) {
                roots[static_cast<std::size_t>(count)] = root;
                ++count;
            }
        }

        f_prev = f_curr;
        prev_t = curr_t;
    }

    if (abs(static_cast<Acc>(f_prev)) <= boundary_eps
        && (count == 0 || abs(roots[static_cast<std::size_t>(count) - 1] - Acc{1}) > tol)
        && count < n) {
        roots[static_cast<std::size_t>(count)] = 1.0;
        ++count;
    }

    return count;
}

/// Find every root on [0, 1] by Yuksel's monotone decomposition.
///
/// Differentiates down to degree one, solves there, and walks back up using each
/// level's roots as the next level's monotone-interval boundaries.
///
/// The degree-1 base case is the sharpest width in the block: `c0 / (c0 - c1)` is a
/// **`T` division**, and dividing at accumulator width instead reproduced *none* of
/// 20000 float32 pairs. Every recursion reaches it.
///
/// \tparam T Coefficient type.
/// \param coeff Bernstein coefficients, length `n + 1`.
/// \param tol Bracket-width tolerance.
/// \param roots Receives at most `max(n, 1)` roots, unsorted.
/// \param work Scratch of at least `coeff.size()` entries.
/// \return How many entries of `roots` are valid.
///
/// Mirrors `_yuksel_core._yuksel_roots`. No validation is performed.
template <Real T>
int yuksel_roots(std::span<const T> coeff, accumulator_t<T> tol, std::span<double> roots,
                 std::span<T> work) {
    using std::abs;
    const auto n = static_cast<int>(coeff.size()) - 1;
    if (n <= 0) {
        return 0;
    }

    T d_min = coeff[0];
    T d_max = coeff[0];
    for (int i = 1; i <= n; ++i) {
        d_min = std::min(d_min, coeff[static_cast<std::size_t>(i)]);
        d_max = std::max(d_max, coeff[static_cast<std::size_t>(i)]);
    }
    if (d_min > T{0} || d_max < T{0}) {
        return 0;
    }

    if (n == 1) {
        const T c0 = coeff[0];
        const T c1 = coeff[1];
        if (c0 == c1) {
            return 0;
        }
        // Divide at T. This is the width the whole file is most exposed to.
        const T root = static_cast<T>(c0 / static_cast<T>(c0 - c1));
        if (T{0} <= root && root <= T{1}) {
            roots[0] = static_cast<double>(root);
            return 1;
        }
        return 0;
    }

    // derivs[lev] holds the level-`lev` derivative, of degree `n - 1 - lev`.
    std::vector<double> derivs(static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(n), 0.0);
    const auto row = [&derivs, n](int lev) {
        return std::span<double>(derivs.data() + static_cast<std::size_t>(lev)
                                                     * static_cast<std::size_t>(n),
                                 static_cast<std::size_t>(n));
    };
    for (int i = 0; i < n; ++i) {
        // Subtract at T, widen on the store. The destination array being float64
        // says nothing about the width the subtraction ran at: widening first
        // reproduced only 88 of 4000 float32 vectors.
        row(0)[static_cast<std::size_t>(i)] = static_cast<double>(
            static_cast<T>(coeff[static_cast<std::size_t>(i) + 1] - coeff[static_cast<std::size_t>(i)]));
    }
    for (int lev = 1; lev < n - 1; ++lev) {
        const int size_prev = n - lev;
        for (int i = 0; i < size_prev; ++i) {
            row(lev)[static_cast<std::size_t>(i)] =
                row(lev - 1)[static_cast<std::size_t>(i) + 1] - row(lev - 1)[static_cast<std::size_t>(i)];
        }
    }

    const int deepest = n - 2;
    std::vector<double> crit(static_cast<std::size_t>(n), 0.0);
    std::vector<double> crit_next(static_cast<std::size_t>(n), 0.0);
    std::vector<double> level_work(static_cast<std::size_t>(n), 0.0);
    int n_crit = 0;
    {
        const double c0 = row(deepest)[0];
        const double c1 = row(deepest)[1];
        if (c0 != c1) {
            const double r = c0 / (c0 - c1);
            if (0.0 <= r && r <= 1.0) {
                crit[0] = r;
                n_crit = 1;
            }
        }
    }

    for (int lev = deepest - 1; lev >= 0; --lev) {
        const int deg_lev = n - 1 - lev;
        const std::span<const double> coeff_lev(row(lev).data(),
                                                static_cast<std::size_t>(deg_lev) + 1);

        double lo_val = coeff_lev[0];
        double hi_val = coeff_lev[0];
        for (int i = 1; i <= deg_lev; ++i) {
            lo_val = std::min(lo_val, coeff_lev[static_cast<std::size_t>(i)]);
            hi_val = std::max(hi_val, coeff_lev[static_cast<std::size_t>(i)]);
        }
        if (lo_val > 0.0 || hi_val < 0.0) {
            n_crit = 0;
            continue;
        }

        if (n_crit == 0) {
            const double f_lo = coeff_lev[0];
            const double f_hi = coeff_lev[static_cast<std::size_t>(deg_lev)];
            const double scale = abs(hi_val - lo_val);
            const double boundary_eps = std::max(scale * kDblEpsilon * 8.0, 1e-30);

            if (abs(f_lo) <= boundary_eps) {
                crit[0] = 0.0;
                n_crit = 1;
            } else if (f_lo * f_hi < 0.0) {
                crit[0] = solve_on_interval<double>(coeff_lev, 0.0, 1.0, f_lo, tol, level_work);
                n_crit = 1;
            } else if (abs(f_hi) <= boundary_eps) {
                crit[0] = 1.0;
                n_crit = 1;
            } else {
                n_crit = 0;
            }
            continue;
        }

        n_crit = find_roots_at_level<double>(coeff_lev, std::span<const double>(crit.data(),
                                                                               static_cast<std::size_t>(n_crit)),
                                             n_crit, tol, crit_next, level_work);
        std::swap(crit, crit_next);
    }

    return find_roots_at_level<T>(coeff,
                                  std::span<const double>(crit.data(),
                                                          static_cast<std::size_t>(n_crit)),
                                  n_crit, tol, roots, work);
}

/// Stack-based Bézier clipping root finder.
///
/// Always subdivides from the original coefficients rather than from the previous
/// sub-interval's, so repeated de Casteljau splits cannot accumulate error.
///
/// Note the contrast with `yuksel_roots`: its degree-1 base case divides at `T`,
/// this one divides at `double`, because here the operands come from the widened
/// sub-interval coefficients. The same expression, two widths, and nothing about the
/// kernel says which.
///
/// \tparam T Coefficient type.
/// \param root_coeff Original Bernstein coefficients on [0, 1].
/// \param param_tol Bracket-width termination tolerance.
/// \param geom_tol Geometric tolerance for near-zero detection.
/// \param roots Receives at most `3 * n + 4` unsorted, possibly duplicated roots.
/// \param work Scratch of at least `root_coeff.size()` entries.
/// \return How many entries of `roots` are valid.
///
/// Mirrors `_clipping_core._clip_roots_core`. No validation is performed.
template <Real T>
int clip_roots_core(std::span<const T> root_coeff, accumulator_t<T> param_tol,
                    accumulator_t<T> geom_tol, std::span<double> roots, std::span<T> work) {
    using std::abs;
    using Acc = accumulator_t<T>;
    const auto n = static_cast<int>(root_coeff.size()) - 1;
    const int max_roots = 3 * n + 4;
    int n_roots = 0;

    // Seeded with a double, so unification makes the running maximum double even
    // where the coefficients are float32.
    double coeff_scale = 0.0;
    for (int i = 0; i <= n; ++i) {
        coeff_scale =
            std::max(coeff_scale, static_cast<double>(abs(root_coeff[static_cast<std::size_t>(i)])));
    }
    const double zero_tol =
        std::max(coeff_scale * static_cast<double>(n + 1) * 4.0 * kDblEpsilon, geom_tol);

    if (abs(static_cast<double>(root_coeff[0])) <= zero_tol) {
        roots[static_cast<std::size_t>(n_roots)] = 0.0;
        ++n_roots;
    }
    if (abs(static_cast<double>(root_coeff[static_cast<std::size_t>(n)])) <= zero_tol) {
        roots[static_cast<std::size_t>(n_roots)] = 1.0;
        ++n_roots;
    }

    std::vector<double> stack_lo(static_cast<std::size_t>(kMaxStackSize));
    std::vector<double> stack_hi(static_cast<std::size_t>(kMaxStackSize));
    std::vector<std::int64_t> stack_depth(static_cast<std::size_t>(kMaxStackSize));
    std::vector<double> local(static_cast<std::size_t>(n) + 1);
    std::vector<std::int64_t> chain(static_cast<std::size_t>(n) + 1);
    stack_lo[0] = 0.0;
    stack_hi[0] = 1.0;
    stack_depth[0] = 0;
    int stack_size = 1;

    const auto push = [&](double lo_value, double hi_value, std::int64_t depth_value) {
        stack_lo[static_cast<std::size_t>(stack_size)] = lo_value;
        stack_hi[static_cast<std::size_t>(stack_size)] = hi_value;
        stack_depth[static_cast<std::size_t>(stack_size)] = depth_value;
        ++stack_size;
    };

    while (stack_size > 0) {
        --stack_size;
        const double lo = stack_lo[static_cast<std::size_t>(stack_size)];
        const double hi = stack_hi[static_cast<std::size_t>(stack_size)];
        const std::int64_t depth = stack_depth[static_cast<std::size_t>(stack_size)];
        const double span = hi - lo;

        // Step 1: convergence.
        if (span <= param_tol || depth > kClipMaxDepth) {
            double mid = 0.5 * (lo + hi);
            if (lo <= 0.0 && hi <= param_tol) {
                mid = 0.0;
            } else if (lo >= 1.0 - param_tol && hi >= 1.0) {
                mid = 1.0;
            }

            Acc f_mid{};
            Acc df_mid{};
            mid = newton_polish_scalar<T>(root_coeff, mid, lo, hi, param_tol, work, f_mid, df_mid);

            if (abs(f_mid) <= zero_tol) {
                if (abs(df_mid) <= static_cast<Acc>(kDblEpsilon)) {
                    // Double root: the derivative is useless, so bisect.
                    double a = lo;
                    double b = hi;
                    T fa = de_casteljau_eval_scalar<T>(root_coeff, a, work);
                    for (int bisect = 0; bisect < 10; ++bisect) {
                        const double m = 0.5 * (a + b);
                        const T fm = de_casteljau_eval_scalar<T>(root_coeff, m, work);
                        // FELIGN/pantr#351, third site: a T product, so it can
                        // underflow at float32. Reproduced deliberately; unlike the
                        // other two this one has no verified reproduction, only the
                        // same mechanism.
                        if (abs(fm) < abs(fa)) {
                            if (fa * fm <= T{0}) {
                                b = m;
                            } else {
                                a = m;
                                fa = fm;
                            }
                        } else if (fa * fm <= T{0}) {
                            b = m;
                        } else {
                            a = m;
                            fa = fm;
                        }
                        if (b - a <= kDblEpsilon) {
                            break;
                        }
                    }
                    mid = 0.5 * (a + b);
                }

                const T f_final = de_casteljau_eval_scalar<T>(root_coeff, mid, work);
                if (abs(static_cast<Acc>(f_final)) <= zero_tol && n_roots < max_roots) {
                    roots[static_cast<std::size_t>(n_roots)] = mid;
                    ++n_roots;
                }
            }
            continue;
        }

        // Step 2: local coefficients, widened once and double from here on.
        subdivide_scalar<T>(root_coeff, lo, hi, local);
        double local_scale = 0.0;
        for (int i = 0; i <= n; ++i) {
            local_scale = std::max(local_scale, abs(local[static_cast<std::size_t>(i)]));
        }
        const double local_zero_tol =
            std::max(local_scale * static_cast<double>(n + 1) * 4.0 * kDblEpsilon, geom_tol);

        // Step 3: coefficient range.
        double c_min = local[0];
        double c_max = local[0];
        for (int i = 1; i <= n; ++i) {
            c_min = std::min(c_min, local[static_cast<std::size_t>(i)]);
            c_max = std::max(c_max, local[static_cast<std::size_t>(i)]);
        }

        // Step 4: quick rejection when every coefficient shares a sign.
        if (c_min > local_zero_tol || c_max < -local_zero_tol) {
            const double rejection_margin = c_min > local_zero_tol ? c_min : -c_max;
            if (rejection_margin <= zero_tol) {
                Acc f_mid{};
                Acc df_mid{};
                const double mid = newton_polish_scalar<T>(root_coeff, 0.5 * (lo + hi), lo, hi,
                                                           Acc{0}, work, f_mid, df_mid);
                if (abs(f_mid) <= zero_tol && n_roots < max_roots) {
                    roots[static_cast<std::size_t>(n_roots)] = mid;
                    ++n_roots;
                }
            }
            continue;
        }

        // Step 5: flat within noise.
        if (c_max - c_min <= geom_tol) {
            if (abs(c_min) <= local_zero_tol || abs(c_max) <= local_zero_tol
                || c_min * c_max < 0.0) {
                const double mid = 0.5 * (lo + hi);
                const T f_mid = de_casteljau_eval_scalar<T>(root_coeff, mid, work);
                if (abs(static_cast<Acc>(f_mid)) <= zero_tol && n_roots < max_roots) {
                    roots[static_cast<std::size_t>(n_roots)] = mid;
                    ++n_roots;
                }
            }
            continue;
        }

        // Step 6: endpoint roots.
        if (abs(local[0]) <= local_zero_tol && n_roots < max_roots) {
            roots[static_cast<std::size_t>(n_roots)] = lo;
            ++n_roots;
        }
        if (abs(local[static_cast<std::size_t>(n)]) <= local_zero_tol && n_roots < max_roots) {
            roots[static_cast<std::size_t>(n_roots)] = hi;
            ++n_roots;
        }

        // Step 7: variation-diminishing bound.
        const int n_sc = count_sign_changes<double>(local);
        if (n_sc == 0) {
            continue;
        }

        // Step 8: linear base case, at double here.
        if (n == 1) {
            const double c0 = local[0];
            const double c1 = local[1];
            if (c0 != c1) {
                const double r = c0 / (c0 - c1);
                if (0.0 <= r && r <= 1.0 && n_roots < max_roots) {
                    roots[static_cast<std::size_t>(n_roots)] = lo + r * span;
                    ++n_roots;
                }
            }
            continue;
        }

        // Step 9: convex-hull clipping.
        double t_lo_clip = 0.0;
        double t_hi_clip = 0.0;
        const bool clip_found =
            clip_hull_to_zero<double>(local, chain, t_lo_clip, t_hi_clip);
        if (!clip_found) {
            if (n_sc > 0 && stack_size + 1 < kMaxStackSize) {
                const double mid_param = 0.5 * (lo + hi);
                push(lo, mid_param, depth + 1);
                push(mid_param, hi, depth + 1);
            }
            continue;
        }

        const double margin = static_cast<double>(n + 1) * 4.0 * kDblEpsilon;
        const double t_lo_safe = std::max(t_lo_clip - margin, 0.0);
        const double t_hi_safe = std::min(t_hi_clip + margin, 1.0);

        const double new_lo = lo + t_lo_safe * span;
        const double new_hi = lo + t_hi_safe * span;
        const double new_span = new_hi - new_lo;

        if (new_span <= param_tol) {
            Acc f_mid{};
            Acc df_mid{};
            const double mid = newton_polish_scalar<T>(root_coeff, 0.5 * (new_lo + new_hi), new_lo,
                                                       new_hi, param_tol, work, f_mid, df_mid);
            if (abs(f_mid) <= zero_tol && n_roots < max_roots) {
                roots[static_cast<std::size_t>(n_roots)] = mid;
                ++n_roots;
            }
            continue;
        }

        // Step 10: recurse, split, or give up on the clip and split the original.
        const double reduction = span > 0.0 ? 1.0 - (new_span / span) : 0.0;

        if (n_sc == 1) {
            if (reduction >= kClipReductionThreshold) {
                if (stack_size < kMaxStackSize) {
                    push(new_lo, new_hi, depth + 1);
                }
            } else if (stack_size + 1 < kMaxStackSize) {
                const double mid = 0.5 * (new_lo + new_hi);
                push(new_lo, mid, depth + 1);
                push(mid, new_hi, depth + 1);
            }
        } else if (reduction >= kClipReductionThreshold) {
            if (stack_size + 1 < kMaxStackSize) {
                const double mid = 0.5 * (new_lo + new_hi);
                push(new_lo, mid, depth + 1);
                push(mid, new_hi, depth + 1);
            }
        } else if (stack_size + 1 < kMaxStackSize) {
            const double mid = 0.5 * (lo + hi);
            push(lo, mid, depth + 1);
            push(mid, hi, depth + 1);
        }
    }

    return n_roots;
}

/// Sort root candidates and merge the duplicates, with a derivative-aware radius.
///
/// The same root reaches this from several converging intervals. The gap between
/// duplicates is `O(zero_tol / |f'|)`, so the merge radius is computed per candidate
/// and capped at `zero_tol^(1/3)`, an upper bound on the cluster width around a root
/// of multiplicity up to three. Without the cap an exact multiple root, where the
/// derivative vanishes, would swallow every later candidate.
///
/// The cap **must** be `std::pow(x, 1.0 / 3.0)`. `std::cbrt` is the natural spelling
/// and differs from Numba's `**` on 49659 of 50000 tolerances.
///
/// The insertion sort is transliterated rather than replaced by `std::sort`, which
/// is not stable and would need an argument about ties this way avoids needing.
///
/// \tparam T Coefficient type.
/// \param raw_roots Candidates, of which the first `n_roots` are valid.
/// \param n_roots Number of valid candidates.
/// \param coeff Original Bernstein coefficients, for the derivative.
/// \param param_tol Parametric tolerance.
/// \param geom_tol Geometric tolerance.
/// \param out Receives at most `n_roots` roots, sorted ascending.
/// \param work Scratch of at least `coeff.size()` entries.
/// \return How many entries of `out` are valid.
///
/// Mirrors `_clipping_core._dedup_roots_core`. No validation is performed.
template <Real T>
int dedup_roots_core(std::span<const double> raw_roots, int n_roots, std::span<const T> coeff,
                     accumulator_t<T> param_tol, accumulator_t<T> geom_tol,
                     std::span<double> out, std::span<T> work) {
    using std::abs;
    using std::pow;
    using Acc = accumulator_t<T>;
    if (n_roots == 0) {
        return 0;
    }

    const auto n = static_cast<int>(coeff.size()) - 1;
    double coeff_scale = 0.0;
    for (int i = 0; i <= n; ++i) {
        coeff_scale =
            std::max(coeff_scale, abs(static_cast<double>(coeff[static_cast<std::size_t>(i)])));
    }
    const double zero_tol =
        std::max(coeff_scale * static_cast<double>(n + 1) * 4.0 * kDblEpsilon, geom_tol);
    const double base_dedup = std::max(param_tol * 2.0, zero_tol * 4.0);
    const double radius_cap = pow(zero_tol, 1.0 / 3.0);

    for (int i = 0; i < n_roots; ++i) {
        out[static_cast<std::size_t>(i)] = raw_roots[static_cast<std::size_t>(i)];
    }
    for (int i = 1; i < n_roots; ++i) {
        const double key = out[static_cast<std::size_t>(i)];
        int j = i - 1;
        while (j >= 0 && out[static_cast<std::size_t>(j)] > key) {
            out[static_cast<std::size_t>(j) + 1] = out[static_cast<std::size_t>(j)];
            --j;
        }
        out[static_cast<std::size_t>(j) + 1] = key;
    }

    int count = 1;
    for (int i = 1; i < n_roots; ++i) {
        const double gap =
            out[static_cast<std::size_t>(i)] - out[static_cast<std::size_t>(count) - 1];
        if (gap <= base_dedup) {
            continue;
        }
        Acc df{};
        de_casteljau_eval_and_deriv_scalar<T>(
            coeff, out[static_cast<std::size_t>(count) - 1], work, df);
        const double local_tol = zero_tol / std::max(abs(df), static_cast<Acc>(kDblEpsilon));
        if (gap <= std::max(base_dedup, std::min(local_tol * 4.0, radius_cap))) {
            continue;
        }
        out[static_cast<std::size_t>(count)] = out[static_cast<std::size_t>(i)];
        ++count;
    }

    return count;
}

/// Choose between Yuksel and clipping for one polynomial, then deduplicate.
///
/// `c_max` and `c_min_nonzero` are seeded with a `double` and with infinity, so
/// unification makes the dynamic-range ratio a `double` division even at float32.
/// This is the sibling of `yuksel_roots`'s base case, which divides at `T`, and the
/// pair is why the widths in this file cannot be applied by rule.
///
/// \tparam T Coefficient type.
/// \param coeff Bernstein coefficients.
/// \param param_tol Parametric tolerance.
/// \param geom_tol Geometric tolerance.
/// \param out Receives the sorted, deduplicated roots.
/// \return How many entries of `out` are valid.
///
/// Mirrors `_batch_core._dispatch_and_find`. No validation is performed.
template <Real T>
int dispatch_and_find(std::span<const T> coeff, accumulator_t<T> param_tol,
                      accumulator_t<T> geom_tol, std::span<double> out) {
    using std::abs;
    const auto n = static_cast<int>(coeff.size()) - 1;
    if (n < 1) {
        return 0;
    }

    bool all_zero = true;
    for (int i = 0; i <= n; ++i) {
        if (abs(static_cast<accumulator_t<T>>(coeff[static_cast<std::size_t>(i)]))
            > geom_tol) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) {
        return 0;
    }

    bool use_clipping = false;
    if (n >= kClipMinDegree) {
        double c_max = 0.0;
        double c_min_nonzero = std::numeric_limits<double>::infinity();
        for (int i = 0; i <= n; ++i) {
            const auto av =
                static_cast<double>(abs(coeff[static_cast<std::size_t>(i)]));
            c_max = std::max(c_max, av);
            if (av > 0.0 && av < c_min_nonzero) {
                c_min_nonzero = av;
            }
        }
        const double coeff_range = c_min_nonzero < std::numeric_limits<double>::infinity()
                                       ? c_max / c_min_nonzero
                                       : std::numeric_limits<double>::infinity();
        if (coeff_range <= kClipCoeffRangeLimit) {
            use_clipping = true;
        }
    }

    std::vector<T> work(coeff.size());
    std::vector<double> raw(static_cast<std::size_t>(3 * n + 4));
    const int n_raw = use_clipping
                          ? clip_roots_core<T>(coeff, param_tol, geom_tol, raw, work)
                          : yuksel_roots<T>(coeff, param_tol, raw, work);
    if (n_raw == 0) {
        return 0;
    }

    return dedup_roots_core<T>(std::span<const double>(raw.data(), static_cast<std::size_t>(n_raw)),
                               n_raw, coeff, param_tol, geom_tol, out, work);
}

}  // namespace detail

/// Find every root on [0, 1] by Yuksel's monotone decomposition.
///
/// The scratch the algorithm needs is allocated here, as the oracle allocates it per
/// call, rather than being threaded through the binding layer.
///
/// \tparam T Coefficient type.
/// \param coeff Bernstein coefficients, length `degree + 1`.
/// \param tol Bracket-width tolerance.
/// \param out Receives at most `degree` roots, unsorted and possibly duplicated.
/// \return How many entries of `out` are valid.
///
/// Mirrors `_yuksel_core._yuksel_roots`. No validation is performed.
template <Real T>
int yuksel_roots(std::span<const T> coeff, accumulator_t<T> tol, std::span<double> out) {
    std::vector<T> work(coeff.size());
    return detail::yuksel_roots<T>(coeff, tol, out, work);
}

/// Find every root on [0, 1] by Bézier clipping.
///
/// \tparam T Coefficient type.
/// \param coeff Bernstein coefficients, length `degree + 1`.
/// \param param_tol Bracket-width termination tolerance.
/// \param geom_tol Geometric tolerance for near-zero detection.
/// \param out Receives at most `3 * degree + 4` roots, unsorted and possibly
///     duplicated.
/// \return How many entries of `out` are valid.
///
/// Mirrors `_clipping_core._clip_roots_core`. No validation is performed.
template <Real T>
int clip_roots(std::span<const T> coeff, accumulator_t<T> param_tol, accumulator_t<T> geom_tol,
               std::span<double> out) {
    std::vector<T> work(coeff.size());
    return detail::clip_roots_core<T>(coeff, param_tol, geom_tol, out, work);
}

/// Sort root candidates and merge the duplicates.
///
/// \tparam T Coefficient type.
/// \param raw_roots Candidates, of which the first `n_roots` are valid.
/// \param n_roots Number of valid candidates.
/// \param coeff Original Bernstein coefficients, for the derivative.
/// \param param_tol Parametric tolerance.
/// \param geom_tol Geometric tolerance.
/// \param out Receives at most `n_roots` roots, sorted ascending.
/// \return How many entries of `out` are valid.
///
/// Mirrors `_clipping_core._dedup_roots_core`. No validation is performed.
template <Real T>
int dedup_roots(std::span<const double> raw_roots, int n_roots, std::span<const T> coeff,
                accumulator_t<T> param_tol, accumulator_t<T> geom_tol, std::span<double> out) {
    std::vector<T> work(coeff.size());
    return detail::dedup_roots_core<T>(raw_roots, n_roots, coeff, param_tol, geom_tol, out, work);
}

/// Find the unique root of a monotone scalar Bernstein polynomial on [0, 1].
///
/// \tparam T Coefficient type.
/// \param coeff Bernstein coefficients of a monotone polynomial.
/// \param tol Bracket-width termination tolerance.
/// \return The root parameter, or NaN when no sign change is detected.
///
/// Mirrors `_yuksel_core._solve_monotone_root_kernel`. No validation is performed.
template <Real T>
accumulator_t<T> solve_monotone_root(std::span<const T> coeff, accumulator_t<T> tol) {
    std::vector<T> work(coeff.size());
    return detail::solve_monotone_root_kernel<T>(coeff, tol, work);
}

/// Find the roots of many same-degree scalar Bernstein polynomials.
///
/// Each polynomial is independent and writes only its own row, so unlike a reduction
/// this kernel's result cannot move with the thread count.
///
/// \tparam T Coefficient type.
/// \param coeffs Batch of coefficients, shape `(n_polys, degree + 1)`.
/// \param param_tol Parametric tolerance.
/// \param geom_tol Geometric tolerance.
/// \param out_roots Shape `(n_polys, degree)`. Entries past each row's count keep
///     whatever the caller left there, which is NaN.
/// \param out_counts Shape `(n_polys,)`, receiving the per-row root count.
///
/// Mirrors `_batch_core._find_roots_batch_core`. No validation is performed.
template <Real T>
void find_roots_batch(span2d<const T> coeffs, accumulator_t<T> param_tol,
                      accumulator_t<T> geom_tol, span2d<double> out_roots,
                      std::span<std::int64_t> out_counts) {
    const auto n_polys = static_cast<std::ptrdiff_t>(coeffs.extent(0));
    const auto width = static_cast<std::ptrdiff_t>(coeffs.extent(1));
    const auto row_capacity = static_cast<int>(out_roots.extent(1));

    std::vector<T> coeff_row(static_cast<std::size_t>(width));
    std::vector<double> found(static_cast<std::size_t>(3 * width + 4));

    for (std::ptrdiff_t i = 0; i < n_polys; ++i) {
        for (std::ptrdiff_t j = 0; j < width; ++j) {
            coeff_row[static_cast<std::size_t>(j)] = at(coeffs, i, j);
        }
        int count = detail::dispatch_and_find<T>(coeff_row, param_tol, geom_tol, found);
        // A degree-n polynomial has at most n roots; clamp as a memory-safety
        // backstop so a dedup artifact can never overflow the row.
        count = std::min(count, row_capacity);
        out_counts[static_cast<std::size_t>(i)] = count;
        for (int j = 0; j < count; ++j) {
            at(out_roots, i, j) = found[static_cast<std::size_t>(j)];
        }
    }
}

/// Solve for the monotone root of many same-degree scalar Bernstein polynomials.
///
/// \tparam T Coefficient type.
/// \param coeffs Batch of coefficients, shape `(n_polys, degree + 1)`.
/// \param param_tol Parameter-space termination tolerance.
/// \param out_roots Shape `(n_polys,)`, pre-filled with NaN by the caller. A row
///     whose polynomial has no root is left untouched.
///
/// Mirrors `_batch_core._solve_monotone_root_batch_core`. No validation is performed.
template <Real T>
void solve_monotone_root_batch(span2d<const T> coeffs, accumulator_t<T> param_tol,
                               std::span<double> out_roots) {
    const auto n_polys = static_cast<std::ptrdiff_t>(coeffs.extent(0));
    const auto width = static_cast<std::ptrdiff_t>(coeffs.extent(1));

    std::vector<T> coeff_row(static_cast<std::size_t>(width));
    std::vector<T> work(static_cast<std::size_t>(width));

    for (std::ptrdiff_t i = 0; i < n_polys; ++i) {
        for (std::ptrdiff_t j = 0; j < width; ++j) {
            coeff_row[static_cast<std::size_t>(j)] = at(coeffs, i, j);
        }
        const auto root = detail::solve_monotone_root_kernel<T>(coeff_row, param_tol, work);
        if (!std::isnan(root)) {
            out_roots[static_cast<std::size_t>(i)] = root;
        }
    }
}

}  // namespace pantr
