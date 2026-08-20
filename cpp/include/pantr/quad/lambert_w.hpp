#pragma once

/// \file
/// The principal branch of Lambert W, by Halley's method.
///
/// Port of `_lambert_w_principal` in `src/pantr/quad/_rules.py`, which stays as
/// the parity oracle. It replaced `scipy.special.lambertw`, the module's only use
/// of scipy; `design/quadrature_algorithms.md` carries the derivation of both the
/// starting guess and the step count.
///
/// ## Parity here is bounded, not bitwise, and the reason is in libm
///
/// Unlike `legendre.hpp`, which uses only IEEE-pinned arithmetic and comes out
/// bit-identical, this function's shape is set by `log` and `exp`. Neither is
/// correctly rounded in any implementation this project targets, and numpy's are
/// not glibc's: `np.exp`, `np.sinh` and `np.cosh` were measured to disagree with
/// glibc by up to 2 ulp, on 26% of arguments for `sinh`. So the comparison is a
/// bounded one and the tests state it as such.
///
/// It is called **once per rule**, not once per node, so nothing here is on a hot
/// path and no accuracy is traded for speed.

#include <cmath>

namespace pantr {

namespace detail {

/// Halley steps taken from the branch-free asymptotic start.
///
/// Four, matching `_LAMBERT_W_HALLEY_STEPS`. Three miss the unit roundoff: with a
/// Lagrange constant `M <= 0.6886` over the argument range the absolute chain is
/// 0.455, 6.5e-2, 1.1e-4, 4.9e-13, 4.5e-38. The argument that actually decides it
/// is the validity threshold -- bisecting on the decay factor at n = 2, the
/// binding case, three steps reach one unit of roundoff down to 0.5932 and four
/// down to 0.5097, so the shipped 0.6 sits 1.1% from failing with three and 18%
/// with four. The two constants must move together or parity is lost.
inline constexpr int lambert_w_halley_steps = 4;

}  // namespace detail

/// Solve `w e^w = x` on the principal branch.
///
/// \param x The argument. Must be at least about 1.61.
/// \return `W(x)`, within about one unit of roundoff of `W`.
///
/// \note No input validation is performed. This is a Layer 3 kernel in the
///       layering of CLAUDE.md, and the precondition on `x` is load-bearing for
///       the *answer* rather than for memory safety: the starting guess is the
///       large-argument asymptotic form and is branch free, so for a small
///       argument it lands negative and no number of Halley steps recovers.
///       Measured on the Python original: an argument corresponding to a decay
///       factor of 0.50 leaves 2.8e4 units of roundoff after four steps, 0.40
///       leaves 2.4e16, and below about 0.31 the inner logarithm is not real. The
///       only caller is `generate_tanh_sinh`, whose argument is smallest at n = 2
///       where it is 1.885. For general use call
///       `pantr.quad._rules._lambert_w_principal`.
inline double lambert_w_principal(double x) {
    using std::exp;
    using std::log;

    const double log_x = log(x);
    const double log_log_x = log(log_x);
    double w = log_x - log_log_x + log_log_x / log_x;

    for (int step = 0; step < detail::lambert_w_halley_steps; ++step) {
        const double exp_w = exp(w);
        const double residual = w * exp_w - x;
        const double derivative = exp_w * (w + 1.0);
        w = w - residual / (derivative - (w + 2.0) * residual / (2.0 * w + 2.0));
    }

    return w;
}

}  // namespace pantr
