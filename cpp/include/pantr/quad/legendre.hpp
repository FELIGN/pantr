#pragma once

/// \file
/// Gauss-Legendre nodes and weights on `[-1, 1]`, by Newton on `P_n`.
///
/// Port of `_gauss_legendre_symmetric` in `src/pantr/quad/_rules.py`, which stays
/// as the parity oracle. `design/quadrature_algorithms.md` records why the method
/// is Newton on the Legendre recurrence rather than an eigensolve on the Jacobi
/// matrix: Newton converges to the root, so the node error is flat in `n` at well
/// under one unit of roundoff, while an eigensolve inherits the accumulated
/// backward error of its sweep.
///
/// ## This is a transliteration, and that is a contract rather than a style
///
/// Every expression below is written in the association the Python uses, because
/// the parity claim on the nodes is **bitwise**, not bounded. The recurrence uses
/// only `+`, `-`, `*` and `/`, each of which IEEE 754 pins to one correctly
/// rounded result, so two implementations that evaluate the same expression tree
/// in the same order agree to the last bit on any conforming machine. Re-associate
/// one line here -- `a * b * c` as `a * (b * c)`, or Horner where numpy wrote the
/// three-term form -- and the nodes move in their last bits, the weights move with
/// them, and the whole bound shape of `design/backend_parity.md` collapses from a
/// bitwise claim to a fitted one.
///
/// The one operation that is **not** pinned by IEEE 754 is `std::cos`, used once
/// per node for Tricomi's starting guess. `tests/parity/` measures it against
/// numpy's rather than assuming: `np.cos` was measured to agree with glibc's
/// exactly, and separately numpy's array and scalar paths were measured to agree
/// with each other on this rule's start (0 of 350 differed at n = 700), so the
/// scalar loop here is not exposed to numpy's SIMD dispatch. Both are properties
/// of a platform, so the test asserts them; it does not trust this comment.
///
/// ## No `Real` genericity, and no accumulator trait
///
/// Deliberate, and narrower than it looks: it applies to *this* kernel, not to
/// every kernel in `pantr/quad`. A rule generator has no differentiable input --
/// `n` is an integer -- so Tier B never reaches it and there is nothing for the
/// `Real` concept to buy. Against that, instantiating Newton at `float` was
/// measured to give a 1.46e-3 relative weight error at n = 200. So the arithmetic
/// is `double` throughout and the caller narrows.
///
/// The map onto `[0, 1]` and the narrowing to the storage format stay in Python,
/// in `_scale_and_cast_nodes_and_weights`, shared by both backends. That is what
/// makes the map common mode: it cancels exactly in a parity comparison instead of
/// contributing its own endpoint conditioning to the bound.

#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>

namespace pantr {

namespace detail {

/// Evaluate `P_n` and `P_n'` at one point by the three-term recurrence.
///
/// \param n Degree. Must be at least 1.
/// \param x The point, strictly inside `(-1, 1)`.
/// \param value Set to `P_n(x)`.
/// \param derivative Set to `P_n'(x)`.
///
/// \note No input validation is performed. This is a Layer 3 kernel in the
///       layering of CLAUDE.md. Two preconditions are load-bearing: `n >= 1`,
///       and `|x| != 1`, because the derivative is formed as
///       `n (x P_n - P_{n-1}) / (x^2 - 1)` and that denominator vanishes at the
///       endpoints. Neither can occur through `gauss_legendre_symmetric`, whose
///       every node is interior with the outermost about `1 - c/n^2` away. For
///       general use call `pantr.quad._rules._legendre_and_derivative`.
inline void legendre_and_derivative(int n, double x, double& value, double& derivative) {
    double previous = 1.0;
    double current = x;

    for (int k = 2; k <= n; ++k) {
        // The association is numpy's: `(2k-1) * x` first, then `* current`. The
        // integer coefficients are exact in double for any n a rule is asked for.
        const double next =
            (static_cast<double>(2 * k - 1) * x * current - static_cast<double>(k - 1) * previous) /
            static_cast<double>(k);
        previous = current;
        current = next;
    }

    value = current;
    derivative = static_cast<double>(n) * (x * current - previous) / (x * x - 1.0);
}

/// Fixed Newton steps taken from Tricomi's start, matching the Python constant.
///
/// Four, and the count is independent of `n` rather than tuned: the start's error
/// is `O(1/n^2)` at the outermost root while Newton's constant `|x| / (1 - x^2)`
/// is `O(n^2)` there, and the two cancel, leaving `C e_0` about 0.02 at every
/// degree. The chain 0.02, 4e-4, 1.6e-7, 2.6e-14, 7e-28 clears the unit roundoff
/// at the fourth step. `_GAUSS_LEGENDRE_NEWTON_STEPS` in `src/pantr/quad/_rules.py`
/// is the same number and the two must move together or parity is lost.
inline constexpr int gauss_legendre_newton_steps = 4;

}  // namespace detail

/// Compute the `n`-point Gauss-Legendre rule on `[-1, 1]` in double precision.
///
/// Only the `ceil(n/2)` non-negative roots are found; each is written together
/// with its exact negation, which keeps the rule symmetric to the last bit rather
/// than to a tolerance. Nodes come out ascending.
///
/// Newton runs independently on each root, so no intermediate array is needed and
/// the kernel allocates nothing: the two mirror positions of a root are filled as
/// soon as it converges. For odd `n` the central root's two positions coincide,
/// and the positive value is written second so it is the one that survives --
/// matching the Python, which splices the centre in from the non-negated half.
///
/// \param n Number of points. Must be at least 1.
/// \param out_nodes Output of length `n`, written in full, ascending in `(-1, 1)`.
/// \param out_weights Output of length `n`, written in full. The weights sum to 2
///        up to the rounding of that sum.
///
/// \note No input validation is performed. This is a Layer 3 kernel in the
///       layering of CLAUDE.md. Two preconditions are load-bearing for memory
///       safety rather than for the answer: `n >= 1`, since it is widened to
///       `std::size_t` and a negative value becomes an enormous loop bound; and
///       both spans having length exactly `n`, since nothing here bounds-checks a
///       write. **`out_nodes` and `out_weights` must not
///       overlap**: each is written independently, so an aliased pair leaves one
///       silently overwriting the other. `cpp/bindings/quad.cpp` refuses that at
///       the boundary and the Python adapters cannot produce it, but a C++ caller
///       establishes it itself. Both are established by the Layer 2 caller. For general use call
///       `pantr.quad.get_gauss_legendre_1d`.
inline void gauss_legendre_symmetric(int n, std::span<double> out_nodes,
                                     std::span<double> out_weights) {
    // Unqualified, after a using-declaration. `std::cos(x)` would name the
    // overload directly and suppress ADL, which is how a user-defined scalar's
    // own overload gets excluded; `scripts/ci_local.sh discipline` enforces the
    // rule across cpp/include and cpp/bindings. It does not bind here, since
    // this kernel is double-only by design, but the guard is repository-wide and
    // an exception in one header is how a rule stops being one.
    using std::cos;

    if (n == 1) {
        out_nodes[0] = 0.0;
        out_weights[0] = 2.0;
        return;
    }

    const auto count = static_cast<std::size_t>(n);
    const std::size_t half = (count + 1) / 2;

    for (std::size_t i = 0; i < half; ++i) {
        // Tricomi's asymptotic approximation to the roots, counted inward from +1.
        // `n + 0.5` is exact, and the index is a small integer, so the only
        // inexactness in the start is the division and the cosine.
        const double index = static_cast<double>(i + 1);
        double x = cos(std::numbers::pi * (index - 0.25) / (static_cast<double>(n) + 0.5));

        double value = 0.0;
        double derivative = 0.0;
        for (int step = 0; step < detail::gauss_legendre_newton_steps; ++step) {
            detail::legendre_and_derivative(n, x, value, derivative);
            x = x - value / derivative;
        }
        detail::legendre_and_derivative(n, x, value, derivative);

        const double weight = 2.0 / ((1.0 - x * x) * derivative * derivative);

        // The negation is exact, so the two halves are exact mirrors. For odd `n`
        // the last root has `i == half - 1` and `count - 1 - i == half - 1` too;
        // the positive write lands second and wins, as in the Python.
        out_nodes[i] = -x;
        out_weights[i] = weight;
        out_nodes[count - 1 - i] = x;
        out_weights[count - 1 - i] = weight;
    }
}

}  // namespace pantr
