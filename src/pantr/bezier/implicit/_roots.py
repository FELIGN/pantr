"""1D Bernstein polynomial root finding for implicit quadrature.

Uses the fast subdivision + Newton method from algoim (Saye 2022,
Supplementary material Section D): recursive de Casteljau subdivision
with sign-change counting, followed by Newton's method (safeguarded by
bisection) on monotone intervals.

Main exports:

- :func:`find_roots` -- find all real roots of a 1D Bernstein polynomial in [0, 1].

Note:
    Inputs are assumed to be correct (no validation performed).
    These are Layer 3 kernels for the implicit quadrature module.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from pantr._numba_compat import nb_jit
from pantr.bezier._root_finding_core import (
    _de_casteljau_eval_and_deriv_scalar,
    _de_casteljau_eval_scalar,
    _restrict_scalar,
)

_ROOT_TOL: float = 1e-14
"""Tolerance for root parameter convergence."""

_MAX_NEWTON: int = 50
"""Maximum Newton iterations per root."""

_MAX_SUBDIV_DEPTH: int = 6
"""Maximum recursion depth for subdivision (2^6 = 64 sub-intervals)."""


# ---------------------------------------------------------------------------
# Section A: Newton-bisection solver on a monotone interval
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _newton_bisection(
    coeffs: npt.NDArray[np.float64],
    lo: float,
    hi: float,
) -> float:
    """Find a root of a Bernstein polynomial on [lo, hi] via Newton + bisection.

    Assumes there is exactly one root in the interval (sign change verified
    by caller). Falls back to bisection if Newton steps leave the bracket.

    Args:
        coeffs (npt.NDArray[np.float64]): Original Bernstein coefficients on [0, 1].
        lo (float): Left bound.
        hi (float): Right bound.

    Returns:
        float: Root parameter, or NaN if convergence fails.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    f_lo = _de_casteljau_eval_scalar(coeffs, lo)
    f_hi = _de_casteljau_eval_scalar(coeffs, hi)

    if abs(f_lo) < _ROOT_TOL * 0.1:
        return lo
    if abs(f_hi) < _ROOT_TOL * 0.1:
        return hi

    # Track which side is negative: neg_side=lo means f(lo)<0.
    # Keep lo < hi always.
    if f_lo > 0.0:
        # f is positive at lo, negative at hi. Track with a flag.
        f_neg_at_lo = False
    else:
        f_neg_at_lo = True

    # Initial guess by false position.
    mid = lo + abs(f_lo) / (abs(f_lo) + abs(f_hi)) * (hi - lo)

    for _ in range(_MAX_NEWTON):
        f_mid, df_mid = _de_casteljau_eval_and_deriv_scalar(coeffs, mid)

        if abs(f_mid) < _ROOT_TOL:
            return mid

        # Update bracket based on sign.
        if (f_mid < 0.0) == f_neg_at_lo:
            lo = mid
        else:
            hi = mid

        if hi - lo < _ROOT_TOL:
            return 0.5 * (lo + hi)

        # Newton step.
        if abs(df_mid) > 1e-300:
            newton = mid - f_mid / df_mid
            if lo < newton < hi:
                mid = newton
                continue

        # Bisection fallback.
        mid = 0.5 * (lo + hi)

    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Section B: Sign-change counting
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _count_sign_changes(coeffs: npt.NDArray[np.float64]) -> int:
    """Count sign changes in a Bernstein coefficient sequence.

    Args:
        coeffs (npt.NDArray[np.float64]): 1D coefficient array.

    Returns:
        int: Number of sign changes (ignoring zero coefficients).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    changes = 0
    prev_sign = 0
    for i in range(len(coeffs)):
        v = coeffs[i]
        if v > 0.0:
            s = 1
        elif v < 0.0:
            s = -1
        else:
            continue
        if prev_sign != 0 and prev_sign != s:
            changes += 1
        prev_sign = s
    return changes


# ---------------------------------------------------------------------------
# Section C: Recursive subdivision root finder
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def find_roots(
    coeffs: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], int]:
    """Find all real roots of a 1D Bernstein polynomial in (0, 1).

    Uses iterative stack-based subdivision with sign-change counting,
    followed by Newton-bisection on intervals with exactly one sign change.
    Maximum subdivision depth is 4 (16 sub-intervals of [0,1]).

    Args:
        coeffs (npt.NDArray[np.float64]): 1D Bernstein coefficients of
            length ``n + 1`` (degree n >= 0).

    Returns:
        tuple[npt.NDArray[np.float64], int]: (roots_buffer, count) where
            only the first *count* entries are valid roots, sorted in
            ascending order.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = len(coeffs) - 1
    max_roots = max(n, 1)
    roots = np.empty(max_roots, dtype=np.float64)
    count = 0

    if n <= 0:
        return roots, 0

    # Quick rejection: uniform sign.
    all_pos = True
    all_neg = True
    for i in range(len(coeffs)):
        if coeffs[i] <= 0.0:
            all_pos = False
        if coeffs[i] >= 0.0:
            all_neg = False
    if all_pos or all_neg:
        return roots, 0

    # Degree 1: linear.
    if n == 1:
        c0, c1 = coeffs[0], coeffs[1]
        denom = c1 - c0
        if abs(denom) > 1e-300:
            t = -c0 / denom
            if _ROOT_TOL < t < 1.0 - _ROOT_TOL:
                roots[0] = t
                return roots, 1
        return roots, 0

    # Stack-based subdivision.
    # Stack entries: (lo, hi, depth)
    stack_lo = np.empty(256, dtype=np.float64)
    stack_hi = np.empty(256, dtype=np.float64)
    stack_depth = np.empty(256, dtype=np.int64)
    sp = 0

    # Push initial interval.
    stack_lo[0] = 0.0
    stack_hi[0] = 1.0
    stack_depth[0] = 0
    sp = 1

    while sp > 0:
        sp -= 1
        lo = stack_lo[sp]
        hi = stack_hi[sp]
        depth = stack_depth[sp]

        # Restrict polynomial to [lo, hi].
        sub = _restrict_scalar(coeffs, lo, hi)

        # Check for near-zero coefficients.
        max_abs = 0.0
        for i in range(len(sub)):
            a = abs(sub[i])
            max_abs = max(max_abs, a)

        # Very small polynomial: skip (no significant root).
        if max_abs < _ROOT_TOL * 10.0:
            continue

        # Count sign changes.
        sc = _count_sign_changes(sub)

        if sc == 0:
            # No roots in this interval.
            continue

        if sc == 1:
            # Exactly one root: solve via Newton-bisection.
            root = _newton_bisection(coeffs, lo, hi)
            if _ROOT_TOL < root < 1.0 - _ROOT_TOL and count < max_roots:
                # Check not duplicate.
                is_dup = False
                for j in range(count):
                    if abs(root - roots[j]) < _ROOT_TOL * 100.0:
                        is_dup = True
                        break
                if not is_dup:
                    roots[count] = root
                    count += 1
            continue

        # Multiple sign changes: subdivide if depth allows.
        if depth < _MAX_SUBDIV_DEPTH:
            mid = 0.5 * (lo + hi)
            if sp + 2 <= 256:
                stack_lo[sp] = lo
                stack_hi[sp] = mid
                stack_depth[sp] = depth + 1
                sp += 1
                stack_lo[sp] = mid
                stack_hi[sp] = hi
                stack_depth[sp] = depth + 1
                sp += 1
        else:
            # Max depth reached: try Newton-bisection anyway.
            root = _newton_bisection(coeffs, lo, hi)
            if _ROOT_TOL < root < 1.0 - _ROOT_TOL and count < max_roots:
                is_dup = False
                for j in range(count):
                    if abs(root - roots[j]) < _ROOT_TOL * 100.0:
                        is_dup = True
                        break
                if not is_dup:
                    roots[count] = root
                    count += 1

    # Sort roots.
    for i in range(count - 1):
        for j in range(i + 1, count):
            if roots[j] < roots[i]:
                roots[i], roots[j] = roots[j], roots[i]

    return roots, count
