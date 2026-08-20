#pragma once

/// \file
/// Tanh-sinh (double-exponential) quadrature nodes and weights on `[-1, 1]`.
///
/// Port of `_generate_tanh_sinh` in `src/pantr/quad/_rules.py`, which stays as
/// the parity oracle. Under `x(t) = tanh((pi/2) sinh t)` the integral over
/// `[-1, 1]` becomes one over the whole line whose integrand decays doubly
/// exponentially, so the trapezoidal rule with uniform step `h` converges fast;
/// the Jacobian gives the weight. Takahasi and Mori (1974), *Publ. RIMS, Kyoto
/// Univ.* 9(3), 721-741.
///
/// ## Two transliteration traps, both measured, both deliberate
///
/// **`pow`, not a multiplication.** The Python writes `np.cosh(omega) ** 2` on a
/// *scalar*, where numpy does not special-case the exponent and calls libm `pow`.
/// Measured: `x ** 2` differs from `x * x` by exactly 1 ulp on 163 of 200000
/// random arguments, and `math.pow` reproduces `x ** 2` exactly, which identifies
/// the callee. (The *array* path does special-case it, which is why this is easy
/// to miss.) So `std::pow(c, 2.0)` below is not clumsiness: writing `c * c` --
/// which is the better numerics, being correctly rounded where `pow` here is not
/// -- would silently move about 0.08% of the weights. The Python is not corrected
/// either, because an external consumer pins golden values off this rule.
///
/// **The rescaling sum is pairwise in numpy and plain here.** `np.sum` uses
/// pairwise summation with a block size of 128; measured to differ from a plain
/// left-to-right accumulation from 8 elements up. Reproducing numpy's blocking
/// would buy nothing, because this rule cannot be bit-exact anyway: `np.exp`,
/// `np.sinh` and `np.cosh` disagree with glibc's by up to 2 ulp, on 26% of
/// arguments for `sinh`. So the sum is written plainly and the parity bound
/// carries both terms -- the transcendental one and the `O(m u)` summation one
/// against numpy's `O(log m . u)`.
///
/// ## The node count is an output, not a function of the inputs
///
/// Generation stops at the last node whose distance to the endpoint is still
/// representable once the rule is mapped onto `[0, 1]` in the caller's storage
/// format, so the effective count may be below `n`. That is why this kernel
/// **returns** its count rather than filling a shape the caller predicted, and it
/// is the one place in `pantr/quad` where the seam signature differs from the
/// others. `design/cross_backend_types.md` records the shape.
///
/// A count that moved between backends would be a genuine parity failure rather
/// than a rounding difference, since `gap < min_gap` is a discrete decision.
/// Measured on the Python original: under a +-2 ulp perturbation of every
/// transcendental the rule moves at most 3.3e-16, 15x inside the golden
/// tolerance, and the count never changes. The tests assert the count, not the
/// measurement.

#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>

#include "pantr/quad/lambert_w.hpp"

namespace pantr {

namespace detail {

/// Decay-rate factor selecting the uniform step `h` in transform space.
///
/// Matches `_TANH_SINH_DECAY_FACTOR`. It is coupled to
/// `lambert_w_halley_steps`: the Halley start is only valid above about 1.61 and
/// this factor sets the argument, smallest at n = 2 where it is 1.885. Neither
/// constant records the coupling on its own, so a test pins it.
inline constexpr double tanh_sinh_decay_factor = 0.6;

}  // namespace detail

/// Generate the tanh-sinh rule on `[-1, 1]` in double precision.
///
/// Nodes come out in the Python's order, which is generation order rather than
/// ascending: the central node first when `n` is odd, then each symmetric pair as
/// it is produced, negative member first. Both members of a pair are written from
/// the same `gap`, so their coordinates are exact negations of each other.
///
/// The weights are finally rescaled onto the measure 2 of `[-1, 1]`. That sum is
/// 2 only up to its own rounding: dividing by a computed sum cannot make a
/// floating-point sum exact.
///
/// \param n Requested number of points. Must be at least 1.
/// \param min_gap Smallest endpoint distance `1 - |x|` a node may carry, in the
///        frame the rule will be returned in. The caller derives it from its
///        storage format; `pantr.quad._rules._tanh_sinh_min_gap` is that
///        derivation and it comes to one machine epsilon.
/// \param out_nodes Output of length at least `n`. Only the first `m` entries are
///        written, where `m` is the return value.
/// \param out_weights Output of length at least `n`, written the same way.
/// \return `m`, the effective node count, which may be below `n`.
///
/// \note No input validation is performed. This is a Layer 3 kernel in the
///       layering of CLAUDE.md. Load-bearing preconditions, established by the
///       Layer 2 caller: `n >= 1`; both spans having length at least `n`, since
///       nothing here bounds-checks a write and the worst case really does reach
///       `n`; and `min_gap > 0`, since a non-positive value never terminates the
///       loop by its own test and would leave the rule to run to `n` with nodes
///       that collapse onto the endpoint when mapped. **`out_nodes` and `out_weights` must not
///       overlap**: each is written independently, so an aliased pair leaves one
///       silently overwriting the other. `cpp/bindings/quad.cpp` refuses that at
///       the boundary and the Python adapters cannot produce it, but a C++ caller
///       establishes it itself. For general use call
///       `pantr.quad.get_tanh_sinh_1d`.
inline std::size_t generate_tanh_sinh(int n, double min_gap, std::span<double> out_nodes,
                                      std::span<double> out_weights) {
    using std::cosh;
    using std::exp;
    using std::pow;
    using std::sinh;

    if (n == 1) {
        out_nodes[0] = 0.0;
        out_weights[0] = 2.0;
        return 1;
    }

    const double half_pi = 0.5 * std::numbers::pi;
    // The argument of W follows from the large-argument truncation balance
    // described on `tanh_sinh_decay_factor`. Association is numpy's.
    const double decay_arg =
        2.0 * detail::tanh_sinh_decay_factor * half_pi * static_cast<double>(n - 1);
    const double h = 2.0 * lambert_w_principal(decay_arg) / static_cast<double>(n);

    std::size_t count = 0;

    const bool odd = (n % 2) != 0;
    if (odd) {
        // Central node at t = 0: x = 0, w = (pi / 2) cosh(0) / cosh(0)^2.
        out_nodes[count] = 0.0;
        out_weights[count] = half_pi;
        ++count;
    }

    for (int i = 0; i < n / 2; ++i) {
        // Odd n samples t = h, 2h, ...; even n offsets by half a step.
        const double t = odd ? (static_cast<double>(i) + 1.0) * h : (static_cast<double>(i) + 0.5) * h;
        const double omega = half_pi * sinh(t);
        // `pow(c, 2.0)`, not `c * c`. See the file comment: the Python's scalar
        // `** 2` is libm `pow`, and the two differ by 1 ulp often enough to matter.
        const double weight = half_pi * cosh(t) / pow(cosh(omega), 2.0);
        // gap = 1 - tanh(omega), the node's distance from the +1 endpoint, via the
        // algebraically equal form 2 / (1 + e^{2 omega}). That keeps gap a small
        // but nonzero double right up to the resolution limit, where
        // `1 - tanh(omega)` saturates to 0 a step early and truncates the rule.
        const double gap = 2.0 / (1.0 + exp(2.0 * omega));

        // gap decreases monotonically in t, so the first node whose distance no
        // longer survives the mapping onto [0, 1] ends the rule.
        if (gap < min_gap) {
            break;
        }

        // Writing both members from `gap` keeps them exact negations.
        out_nodes[count] = -(1.0 - gap);
        out_weights[count] = weight;
        ++count;
        out_nodes[count] = 1.0 - gap;
        out_weights[count] = weight;
        ++count;
    }

    // Rescale onto the measure 2 of [-1, 1]. Plain accumulation, deliberately:
    // see the file comment on numpy's pairwise sum.
    double total = 0.0;
    for (std::size_t j = 0; j < count; ++j) {
        total = total + out_weights[j];
    }
    const double scale = 2.0 / total;
    for (std::size_t j = 0; j < count; ++j) {
        out_weights[j] = out_weights[j] * scale;
    }

    return count;
}

}  // namespace pantr
