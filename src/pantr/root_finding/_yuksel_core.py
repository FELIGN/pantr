"""Yuksel's monotone-decomposition root finder for Bernstein polynomials.

Implements the algorithm from Yuksel (2022) for finding all real roots of a
Bernstein polynomial on [0, 1].

Algorithm overview:

1. **Differentiate.** Bernstein derivative coefficients are ``d_i = c_{i+1} - c_i``.
2. **Recurse.** Find all roots of the derivative (degree ``n - 1``).
3. **Monotone intervals.** The derivative roots partition [0, 1] into sub-intervals
   on which the polynomial is monotone.
4. **Sign-change filter.** On each monotone interval, evaluate at the endpoints; a
   root exists iff the signs differ.
5. **Solve.** Apply a clamped Newton/bisection hybrid on each interval that contains
   a root.

The recursion bottoms out at degree 1 (linear direct formula).

Main exports (all Numba-compiled):

- :func:`_solve_monotone_root_kernel` -- Newton/bisection hybrid for a single root
  on a known-monotone interval.
- :func:`_yuksel_roots` -- general multi-root finder implementing the full algorithm.

References:
    Cem Yuksel (2022), *High-Performance Polynomial Root Finding for Graphics*.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from pantr._numba_compat import nb_jit
from pantr.root_finding._root_finding_core import (
    _DBL_EPSILON,
    _de_casteljau_eval_and_deriv_scalar,
    _de_casteljau_eval_scalar,
)

_MAX_NEWTON_ITER: int = 64
"""Safety cap on Newton/bisection iterations.

For double-precision arithmetic on a [0, 1] bracket, pure bisection converges
in at most 53 steps (``log2(1 / eps)``). With Newton acceleration it is
typically far fewer. This constant is a generous upper bound that is never
reached in practice.
"""


@nb_jit(nopython=True, cache=True)
def _solve_monotone_root_kernel(
    coeff: npt.NDArray[np.float32 | np.float64],
    param_tol: float,
) -> float:
    """Find the unique root of a monotone Bernstein polynomial on [0, 1].

    Implements the Yuksel (2022) clamped Newton-bisection hybrid. Maintains a
    valid bracket ``[lo, hi]`` guaranteed to contain the root. At each
    iteration:

    1. Evaluate ``f(x)`` and ``f'(x)`` in a single de Casteljau pass.
    2. Shrink the bracket based on the sign of ``f(x)`` relative to ``f(lo)``.
    3. Attempt a Newton step ``x_new = x - f(x) / f'(x)``.
    4. If the Newton step falls outside the bracket, fall back to bisection.

    Uses false-position as the initial guess (better convergence than midpoint
    for skewed roots), clamped to the interior to avoid boundaries where
    ``f'`` may vanish.

    Returns ``NaN`` if no sign change is detected across [0, 1].

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): Bernstein coefficients of the
            monotone scalar polynomial.
        param_tol (float): Parameter-space termination tolerance.

    Returns:
        float: Root parameter in [0, 1], or ``NaN`` if no root exists.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.root_finding.solve_monotone_root`
        instead.
    """
    lo = 0.0
    hi = 1.0

    f_lo = float(coeff[0])
    f_hi = float(coeff[-1])

    # No sign change: no root.
    if f_lo * f_hi > 0.0:
        return np.nan

    # False-position initial guess.
    if abs(f_hi - f_lo) > 0.0:
        x = lo + (-f_lo) / (f_hi - f_lo) * (hi - lo)
        # Clamp to interior to avoid boundaries where f' = 0.
        margin = 0.1 * (hi - lo)
        x = max(lo + margin, min(x, hi - margin))
    else:
        x = 0.5 * (lo + hi)

    for _ in range(_MAX_NEWTON_ITER):
        fx, dfx = _de_casteljau_eval_and_deriv_scalar(coeff, x)

        # Shrink bracket.
        if f_lo * fx <= 0.0:
            hi = x
        else:
            lo = x

        # Termination: bracket width.
        if (hi - lo) <= param_tol:
            break

        # Newton step.
        x_new = x - fx / dfx if abs(dfx) > 0.0 else 0.5 * (lo + hi)

        # Validate: must land strictly inside bracket.
        x = x_new if lo < x_new < hi else 0.5 * (lo + hi)

    return 0.5 * (lo + hi)


@nb_jit(nopython=True, cache=True)
def _solve_on_interval(
    coeff: npt.NDArray[np.float32 | np.float64],
    lo: float,
    hi: float,
    f_lo: float,
    tol: float,
) -> float:
    """Find one root of a Bernstein polynomial on ``[lo, hi]``.

    Same Newton/bisection hybrid as :func:`_solve_monotone_root_kernel`, but
    operating on an arbitrary sub-interval of [0, 1] without reparametrizing
    the coefficients.

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): Bernstein coefficients on [0, 1].
        lo (float): Left boundary of the search interval.
        hi (float): Right boundary of the search interval.
        f_lo (float): Pre-evaluated ``P(lo)``.
        tol (float): Bracket-width termination tolerance.

    Returns:
        float: Approximate root parameter in ``[lo, hi]``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.root_finding.find_roots` instead.
    """
    x = 0.5 * (lo + hi)

    for _ in range(_MAX_NEWTON_ITER):
        fx, dfx = _de_casteljau_eval_and_deriv_scalar(coeff, x)

        if f_lo * fx <= 0.0:
            hi = x
        else:
            lo = x

        if (hi - lo) <= tol:
            return 0.5 * (lo + hi)

        x_new = x - fx / dfx if abs(dfx) > 0.0 else 0.5 * (lo + hi)
        x = x_new if lo < x_new < hi else 0.5 * (lo + hi)

    return 0.5 * (lo + hi)


@nb_jit(nopython=True, cache=True)
def _find_roots_at_level(
    coeff: npt.NDArray[np.float32 | np.float64],
    crit: npt.NDArray[np.float32 | np.float64],
    n_crit: int,
    tol: float,
) -> tuple[npt.NDArray[np.float64], int]:
    """Find all roots by walking monotone intervals defined by critical params.

    Given Bernstein coefficients and their sorted critical parameters (roots
    of the derivative), walks the monotone intervals, detects sign changes,
    and solves each interval containing a root.

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): Bernstein coefficients (degree =
            ``len(coeff) - 1``).
        crit (npt.NDArray[np.float32 | np.float64]): Sorted critical parameters (roots of
            the derivative).
        n_crit (int): Number of valid entries in ``crit``.
        tol (float): Bracket-width tolerance.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], int]: ``(roots_array, count)`` where
            only the first ``count`` entries are valid.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.root_finding.find_roots` instead.
    """
    n = len(coeff) - 1
    roots = np.empty(n, dtype=np.float64)
    count = 0

    # Scale-relative boundary tolerance for detecting near-zero values.
    d_min = float(np.min(coeff))
    d_max = float(np.max(coeff))
    scale = abs(d_max - d_min)
    boundary_eps = max(scale * _DBL_EPSILON * 8.0, 1e-30)

    prev_t = 0.0
    f_prev = float(coeff[0])

    for k in range(n_crit + 1):
        curr_t = crit[k] if k < n_crit else 1.0

        # Skip degenerate (zero-width) intervals.
        if curr_t - prev_t < tol:
            f_prev = _de_casteljau_eval_scalar(coeff, curr_t) if curr_t < 1.0 else float(coeff[n])
            prev_t = curr_t
            continue

        # Evaluate at right boundary.
        f_curr = _de_casteljau_eval_scalar(coeff, curr_t) if curr_t < 1.0 else float(coeff[n])

        # Check for exact (or near-exact) endpoint zeros.
        if abs(f_prev) <= boundary_eps:  # noqa: SIM102
            if count == 0 or abs(roots[count - 1] - prev_t) > tol:
                if count < n:
                    roots[count] = prev_t
                    count += 1
                f_prev = f_curr
                prev_t = curr_t
                continue

        # Check for sign change -> guaranteed interior root.
        if f_prev * f_curr < 0.0:
            root = _solve_on_interval(coeff, prev_t, curr_t, f_prev, tol)
            if count < n:
                roots[count] = root
                count += 1

        f_prev = f_curr
        prev_t = curr_t

    # Check the last endpoint (t = 1).
    if abs(f_prev) <= boundary_eps and (  # noqa: SIM102
        count == 0 or abs(roots[count - 1] - 1.0) > tol
    ):
        if count < n:
            roots[count] = 1.0
            count += 1

    return roots, count


@nb_jit(nopython=True, cache=True)
def _yuksel_roots(  # noqa: PLR0912, PLR0915
    coeff: npt.NDArray[np.float32 | np.float64],
    tol: float,
) -> tuple[npt.NDArray[np.float64], int]:
    """Find all roots of a Bernstein polynomial on [0, 1].

    Implements Yuksel's (2022) monotone-decomposition algorithm in an iterative
    bottom-up form (avoids Numba recursion issues):

    1. Build derivative chain: ``coeff -> deriv_1 -> deriv_2 -> ...`` down to
       degree 1.
    2. Solve the linear (degree 1) polynomial directly.
    3. Walk back up the chain: at each level, the roots of the lower level are
       the critical parameters that partition [0, 1] into monotone intervals.

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): Bernstein coefficients of the
            polynomial.
        tol (float): Parametric tolerance.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], int]: ``(roots_array, count)`` where
            only the first ``count`` entries are valid.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.root_finding.find_roots` instead.
    """
    n = len(coeff) - 1

    if n <= 0:
        return np.empty(0, dtype=np.float64), 0

    # Quick rejection via convex-hull property.
    d_min = coeff[0]
    d_max = coeff[0]
    for i in range(1, n + 1):
        d_min = min(d_min, coeff[i])
        d_max = max(d_max, coeff[i])
    if d_min > 0.0 or d_max < 0.0:
        return np.empty(n, dtype=np.float64), 0

    # Base case: degree 1 (linear).
    if n == 1:
        roots = np.empty(1, dtype=np.float64)
        c0 = coeff[0]
        c1 = coeff[1]
        if c0 == c1:
            return roots, 0
        root = c0 / (c0 - c1)
        if 0.0 <= root <= 1.0:
            roots[0] = root
            return roots, 1
        return roots, 0

    # Build derivative chain in a padded 2-D array.
    # derivs[k] has degree n - 1 - k, i.e. n - k coefficients.
    derivs = np.zeros((n - 1, n), dtype=np.float64)
    for i in range(n):
        derivs[0, i] = coeff[i + 1] - coeff[i]
    for lev in range(1, n - 1):
        sz_prev = n - lev
        for i in range(sz_prev):
            derivs[lev, i] = derivs[lev - 1, i + 1] - derivs[lev - 1, i]

    # Bottom-up: solve degree-1 at the deepest derivative level.
    deepest = n - 2
    c0 = derivs[deepest, 0]
    c1 = derivs[deepest, 1]
    crit = np.empty(1, dtype=np.float64)
    n_crit = 0
    if c0 != c1:
        r = c0 / (c0 - c1)
        if 0.0 <= r <= 1.0:
            crit[0] = r
            n_crit = 1

    # Walk back up from level n-3 to level 0.
    for lev in range(deepest - 1, -1, -1):
        deg_lev = n - 1 - lev
        coeff_lev = derivs[lev, : deg_lev + 1].copy()

        # Quick rejection.
        lo_val = coeff_lev[0]
        hi_val = coeff_lev[0]
        for i in range(1, deg_lev + 1):
            lo_val = min(lo_val, coeff_lev[i])
            hi_val = max(hi_val, coeff_lev[i])
        if lo_val > 0.0 or hi_val < 0.0:
            crit = np.empty(deg_lev, dtype=np.float64)
            n_crit = 0
            continue

        if n_crit == 0:
            # Polynomial at this level is monotone -- at most 1 root.
            f_lo = coeff_lev[0]
            f_hi = coeff_lev[deg_lev]

            scale = abs(hi_val - lo_val)
            boundary_eps = max(scale * _DBL_EPSILON * 8.0, 1e-30)

            if abs(f_lo) <= boundary_eps:
                crit = np.empty(1, dtype=np.float64)
                crit[0] = 0.0
                n_crit = 1
            elif f_lo * f_hi < 0.0:
                root = _solve_on_interval(coeff_lev, 0.0, 1.0, f_lo, tol)
                crit = np.empty(1, dtype=np.float64)
                crit[0] = root
                n_crit = 1
            elif abs(f_hi) <= boundary_eps:
                crit = np.empty(1, dtype=np.float64)
                crit[0] = 1.0
                n_crit = 1
            else:
                crit = np.empty(deg_lev, dtype=np.float64)
                n_crit = 0
            continue

        new_crit, new_n = _find_roots_at_level(coeff_lev, crit, n_crit, tol)
        crit = new_crit
        n_crit = new_n

    # Solve the original polynomial using its critical parameters.
    return _find_roots_at_level(coeff, crit, n_crit, tol)


def _warmup_numba_functions() -> None:
    """Trigger Numba compilation of all kernels in this module.

    Called from the background warmup thread in ``pantr.__init__``.
    """
    dummy = np.array([1.0, -1.0, 0.5], dtype=np.float64)
    _solve_monotone_root_kernel(dummy, 1e-12)
    _solve_on_interval(dummy, 0.0, 1.0, 1.0, 1e-12)
    crit = np.array([0.5], dtype=np.float64)
    _find_roots_at_level(dummy, crit, 1, 1e-12)
    _yuksel_roots(dummy, 1e-12)
