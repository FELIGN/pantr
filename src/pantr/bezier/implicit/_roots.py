"""Self-contained 1D Bernstein polynomial root finding for implicit quadrature.

Duplicates the Yuksel monotone-decomposition and Bezier clipping algorithms
from ``pantr.bezier`` as standalone Numba nopython functions, so the implicit
quadrature module has no runtime dependency on the rest of pantr's root-finding
infrastructure.

Dispatch heuristic (same as ``pantr.bezier._batch_core._dispatch_and_find``):

- Degree < 6: Yuksel (lower overhead for small polynomials).
- Degree >= 6 with coefficient range <= 1e8: Bezier clipping (superlinear).
- Otherwise: Yuksel.

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

_DBL_EPSILON: float = 2.2204460492503131e-16
"""Machine epsilon for IEEE 754 double precision."""

_MAX_NEWTON_ITER: int = 64
"""Safety cap on Newton/bisection iterations."""

_CLIP_REDUCTION_THRESHOLD: float = 0.2
"""Minimum parameter reduction to accept a clip without subdivision."""

_CLIP_MAX_DEPTH: int = 64
"""Maximum recursion depth for the clipping stack."""

_MAX_STACK_SIZE: int = 4096
"""Maximum stack size for the iterative clipping loop.

Sized to handle all practical cases. If the stack is exhausted, the overflow
is reported via the returned boolean flag so callers can warn the user.
"""

_CLIP_MIN_DEGREE: int = 6
"""Minimum polynomial degree for Bezier clipping dispatch."""

_CLIP_COEFF_RANGE_LIMIT: float = 1e8
"""Maximum coefficient dynamic range for Bezier clipping dispatch."""

_ROOT_TOL: float = 1e-15
"""Default parametric tolerance for root finding."""


# ---------------------------------------------------------------------------
# Section A: Scalar de Casteljau evaluation (self-contained)
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _eval_scalar(coeff: npt.NDArray[np.float64], t: float) -> float:
    """Evaluate a scalar Bernstein polynomial at *t* via de Casteljau.

    Args:
        coeff (npt.NDArray[np.float64]): 1D Bernstein coefficients.
        t (float): Parameter in [0, 1].

    Returns:
        float: Polynomial value.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    work = coeff.copy()
    n = len(work) - 1
    for k in range(1, n + 1):
        for i in range(n - k + 1):
            work[i] = (1.0 - t) * work[i] + t * work[i + 1]
    return float(work[0])


@nb_jit(nopython=True, cache=True)
def _eval_and_deriv(coeff: npt.NDArray[np.float64], t: float) -> tuple[float, float]:
    """Evaluate a scalar Bernstein polynomial and its derivative at *t*.

    Args:
        coeff (npt.NDArray[np.float64]): 1D Bernstein coefficients.
        t (float): Parameter in [0, 1].

    Returns:
        tuple[float, float]: (f(t), f'(t)).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = len(coeff) - 1
    if n == 0:
        return float(coeff[0]), 0.0
    s = 1.0 - t
    row = coeff.copy()
    for k in range(1, n):
        for i in range(n - k + 1):
            row[i] = s * row[i] + t * row[i + 1]
    d0 = row[0]
    d1 = row[1]
    f = s * d0 + t * d1
    deriv = float(n) * (d1 - d0)
    return f, deriv


@nb_jit(nopython=True, cache=True)
def _restrict_scalar(
    coeff: npt.NDArray[np.float64], lower: float, upper: float
) -> npt.NDArray[np.float64]:
    """Restrict a scalar Bernstein polynomial to [lower, upper].

    Args:
        coeff (npt.NDArray[np.float64]): 1D Bernstein coefficients.
        lower (float): Left bound in [0, 1].
        upper (float): Right bound in [0, 1].

    Returns:
        npt.NDArray[np.float64]: Restricted coefficients on [0, 1].

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    p = len(coeff) - 1
    d = np.empty(p + 1, dtype=np.float64)
    for i in range(p + 1):
        d[i] = float(coeff[i])

    if abs(upper) >= abs(lower - 1.0):
        tau = upper
        for _step in range(1, p + 1):
            for j in range(p, _step - 1, -1):
                d[j] = d[j] * tau + d[j - 1] * (1.0 - tau)
        tau2 = lower / upper if upper != 0.0 else 0.0
        for _step in range(1, p + 1):
            for j in range(p - _step + 1):
                d[j] = d[j] * (1.0 - tau2) + d[j + 1] * tau2
    else:
        tau = lower
        for _step in range(1, p + 1):
            for j in range(p - _step + 1):
                d[j] = d[j] * (1.0 - tau) + d[j + 1] * tau
        tau2 = (upper - lower) / (1.0 - lower) if lower != 1.0 else 0.0
        for _step in range(1, p + 1):
            for j in range(p, _step - 1, -1):
                d[j] = d[j] * tau2 + d[j - 1] * (1.0 - tau2)
    return d


@nb_jit(nopython=True, cache=True)
def _count_sign_changes(coeff: npt.NDArray[np.float64]) -> int:
    """Count sign changes in a Bernstein coefficient sequence.

    Args:
        coeff (npt.NDArray[np.float64]): 1D coefficient array.

    Returns:
        int: Number of sign changes (ignoring zeros).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    changes = 0
    prev_sign = 0
    for i in range(len(coeff)):
        v = coeff[i]
        if v > 0.0:
            s = 1
        elif v < 0.0:
            s = -1
        else:
            continue
        if prev_sign != 0 and prev_sign != s:  # noqa: PLR1714
            changes += 1
        prev_sign = s
    return changes


@nb_jit(nopython=True, cache=True)
def _clip_hull_to_zero(  # noqa: PLR0912, PLR0915
    coeff: npt.NDArray[np.float64],
) -> tuple[float, float, bool]:
    """Clip parameter range using the convex hull of the control polygon vs y=0.

    Args:
        coeff (npt.NDArray[np.float64]): Bernstein coefficients of shape ``(n+1,)``.

    Returns:
        tuple[float, float, bool]: (t_lo, t_hi, found).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = len(coeff) - 1
    if n < 1:
        return 0.0, 0.0, False

    inv_n = 1.0 / n
    t_lo = 1.0
    t_hi = 0.0
    found = False

    # Upper hull.
    upper = np.empty(n + 1, dtype=np.int64)
    u_size = 0
    for i in range(n + 1):
        while u_size >= 2:  # noqa: PLR2004
            j0 = upper[u_size - 2]
            j1 = upper[u_size - 1]
            cross = (j1 - j0) * (coeff[i] - coeff[j0]) - (coeff[j1] - coeff[j0]) * (i - j0)
            if cross >= 0.0:
                u_size -= 1
            else:
                break
        upper[u_size] = i
        u_size += 1

    for k in range(u_size - 1):
        ia = upper[k]
        ib = upper[k + 1]
        da = coeff[ia]
        db = coeff[ib]
        if da * db < 0.0:
            ta = ia * inv_n
            tb = ib * inv_n
            t_cross = ta + (-da) / (db - da) * (tb - ta)
            t_lo = min(t_lo, t_cross)
            t_hi = max(t_hi, t_cross)
            found = True
        if da == 0.0:
            t_lo = min(t_lo, ia * inv_n)
            t_hi = max(t_hi, ia * inv_n)
            found = True
    last_u = upper[u_size - 1]
    if coeff[last_u] == 0.0:
        t_lo = min(t_lo, last_u * inv_n)
        t_hi = max(t_hi, last_u * inv_n)
        found = True

    # Lower hull.
    lower = np.empty(n + 1, dtype=np.int64)
    l_size = 0
    for i in range(n + 1):
        while l_size >= 2:  # noqa: PLR2004
            j0 = lower[l_size - 2]
            j1 = lower[l_size - 1]
            cross = (j1 - j0) * (coeff[i] - coeff[j0]) - (coeff[j1] - coeff[j0]) * (i - j0)
            if cross <= 0.0:
                l_size -= 1
            else:
                break
        lower[l_size] = i
        l_size += 1

    for k in range(l_size - 1):
        ia = lower[k]
        ib = lower[k + 1]
        da = coeff[ia]
        db = coeff[ib]
        if da * db < 0.0:
            ta = ia * inv_n
            tb = ib * inv_n
            t_cross = ta + (-da) / (db - da) * (tb - ta)
            t_lo = min(t_lo, t_cross)
            t_hi = max(t_hi, t_cross)
            found = True
        if da == 0.0:
            t_lo = min(t_lo, ia * inv_n)
            t_hi = max(t_hi, ia * inv_n)
            found = True
    last_l = lower[l_size - 1]
    if coeff[last_l] == 0.0:
        t_lo = min(t_lo, last_l * inv_n)
        t_hi = max(t_hi, last_l * inv_n)
        found = True

    return t_lo, t_hi, found


@nb_jit(nopython=True, cache=True)
def _newton_polish(
    coeff: npt.NDArray[np.float64],
    mid: float,
    lo: float,
    hi: float,
    param_tol: float,
) -> tuple[float, float, float]:
    """Polish a root candidate with a single Newton step.

    Args:
        coeff (npt.NDArray[np.float64]): Bernstein coefficients on [0, 1].
        mid (float): Initial root estimate.
        lo (float): Left bound.
        hi (float): Right bound.
        param_tol (float): Acceptance tolerance.

    Returns:
        tuple[float, float, float]: (polished_t, f_value, df_value).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    f_mid, df_mid = _eval_and_deriv(coeff, mid)
    if abs(df_mid) > _DBL_EPSILON:
        newton = mid - f_mid / df_mid
        if lo - param_tol <= newton <= hi + param_tol:
            newton = max(0.0, min(1.0, newton))
            f_newton = _eval_scalar(coeff, newton)
            if abs(f_newton) <= abs(f_mid):
                return newton, f_newton, df_mid
    return mid, f_mid, df_mid


# ---------------------------------------------------------------------------
# Section B: Yuksel monotone-decomposition root finder
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _solve_on_interval(
    coeff: npt.NDArray[np.float64],
    lo: float,
    hi: float,
    f_lo: float,
    tol: float,
) -> float:
    """Find one root on [lo, hi] via Newton/bisection hybrid.

    Args:
        coeff (npt.NDArray[np.float64]): Bernstein coefficients on [0, 1].
        lo (float): Left boundary.
        hi (float): Right boundary.
        f_lo (float): Pre-evaluated P(lo).
        tol (float): Bracket-width tolerance.

    Returns:
        float: Approximate root.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    x = 0.5 * (lo + hi)
    for _ in range(_MAX_NEWTON_ITER):
        fx, dfx = _eval_and_deriv(coeff, x)
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
    coeff: npt.NDArray[np.float64],
    crit: npt.NDArray[np.float64],
    n_crit: int,
    tol: float,
) -> tuple[npt.NDArray[np.float64], int]:
    """Find all roots by walking monotone intervals defined by critical params.

    Args:
        coeff (npt.NDArray[np.float64]): Bernstein coefficients.
        crit (npt.NDArray[np.float64]): Sorted critical parameters.
        n_crit (int): Number of valid entries in *crit*.
        tol (float): Bracket-width tolerance.

    Returns:
        tuple[npt.NDArray[np.float64], int]: (roots_array, count).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = len(coeff) - 1
    roots = np.empty(n, dtype=np.float64)
    count = 0

    d_min = float(np.min(coeff))
    d_max = float(np.max(coeff))
    scale = abs(d_max - d_min)
    boundary_eps = max(scale * _DBL_EPSILON * 8.0, 1e-30)

    prev_t = 0.0
    f_prev = float(coeff[0])

    for k in range(n_crit + 1):
        curr_t = crit[k] if k < n_crit else 1.0
        if curr_t - prev_t < tol:
            f_prev = _eval_scalar(coeff, curr_t) if curr_t < 1.0 else float(coeff[n])
            prev_t = curr_t
            continue

        f_curr = _eval_scalar(coeff, curr_t) if curr_t < 1.0 else float(coeff[n])

        if abs(f_prev) <= boundary_eps:  # noqa: SIM102
            if count == 0 or abs(roots[count - 1] - prev_t) > tol:
                if count < n:
                    roots[count] = prev_t
                    count += 1
                f_prev = f_curr
                prev_t = curr_t
                continue

        if f_prev * f_curr < 0.0:
            root = _solve_on_interval(coeff, prev_t, curr_t, f_prev, tol)
            if count < n:
                roots[count] = root
                count += 1

        f_prev = f_curr
        prev_t = curr_t

    if abs(f_prev) <= boundary_eps and (count == 0 or abs(roots[count - 1] - 1.0) > tol):  # noqa: SIM102
        if count < n:
            roots[count] = 1.0
            count += 1

    return roots, count


@nb_jit(nopython=True, cache=True)
def _yuksel_roots(  # noqa: PLR0912, PLR0915
    coeff: npt.NDArray[np.float64],
    tol: float,
) -> tuple[npt.NDArray[np.float64], int]:
    """Find all roots using Yuksel's monotone-decomposition algorithm.

    Args:
        coeff (npt.NDArray[np.float64]): Bernstein coefficients.
        tol (float): Parametric tolerance.

    Returns:
        tuple[npt.NDArray[np.float64], int]: (roots_array, count).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = len(coeff) - 1
    if n <= 0:
        return np.empty(0, dtype=np.float64), 0

    d_min = coeff[0]
    d_max = coeff[0]
    for i in range(1, n + 1):
        d_min = min(d_min, coeff[i])
        d_max = max(d_max, coeff[i])
    if d_min > 0.0 or d_max < 0.0:
        return np.empty(n, dtype=np.float64), 0

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

    # Build derivative chain.
    derivs = np.zeros((n - 1, n), dtype=np.float64)
    for i in range(n):
        derivs[0, i] = coeff[i + 1] - coeff[i]
    for lev in range(1, n - 1):
        sz_prev = n - lev
        for i in range(sz_prev):
            derivs[lev, i] = derivs[lev - 1, i + 1] - derivs[lev - 1, i]

    # Bottom-up: solve degree-1 at the deepest level.
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

    # Walk back up.
    for lev in range(deepest - 1, -1, -1):
        deg_lev = n - 1 - lev
        coeff_lev = derivs[lev, : deg_lev + 1].copy()

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

    return _find_roots_at_level(coeff, crit, n_crit, tol)


# ---------------------------------------------------------------------------
# Section C: Bezier clipping root finder
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _clip_roots_core(  # noqa: PLR0912, PLR0915
    root_coeff: npt.NDArray[np.float64],
    param_tol: float,
    geom_tol: float,
) -> tuple[npt.NDArray[np.float64], int, bool]:
    """Stack-based Bezier clipping root finder.

    Args:
        root_coeff (npt.NDArray[np.float64]): Bernstein coefficients on [0, 1].
        param_tol (float): Parametric tolerance.
        geom_tol (float): Geometric tolerance for near-zero detection.

    Returns:
        tuple[npt.NDArray[np.float64], int, bool]: (roots_array, count, overflowed)
            where *overflowed* is True if the clipping stack was exhausted.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = len(root_coeff) - 1
    max_roots = 3 * n + 4
    roots = np.empty(max_roots, dtype=np.float64)
    n_roots = 0
    overflowed = False

    coeff_scale = 0.0
    for i in range(n + 1):
        coeff_scale = max(coeff_scale, abs(root_coeff[i]))
    zero_tol = max(coeff_scale * (n + 1) * 4.0 * _DBL_EPSILON, geom_tol)

    if abs(root_coeff[0]) <= zero_tol:
        roots[n_roots] = 0.0
        n_roots += 1
    if abs(root_coeff[n]) <= zero_tol:
        roots[n_roots] = 1.0
        n_roots += 1

    stack_lo = np.empty(_MAX_STACK_SIZE, dtype=np.float64)
    stack_hi = np.empty(_MAX_STACK_SIZE, dtype=np.float64)
    stack_depth = np.empty(_MAX_STACK_SIZE, dtype=np.int64)
    stack_lo[0] = 0.0
    stack_hi[0] = 1.0
    stack_depth[0] = 0
    stack_size = 1

    while stack_size > 0:
        stack_size -= 1
        lo = stack_lo[stack_size]
        hi = stack_hi[stack_size]
        depth = stack_depth[stack_size]
        span = hi - lo

        if span <= param_tol or depth > _CLIP_MAX_DEPTH:
            mid = 0.5 * (lo + hi)
            if lo <= 0.0 and hi <= param_tol:
                mid = 0.0
            elif lo >= 1.0 - param_tol and hi >= 1.0:
                mid = 1.0
            mid, f_mid, df_mid = _newton_polish(root_coeff, mid, lo, hi, param_tol)
            if abs(f_mid) <= zero_tol:
                if abs(df_mid) <= _DBL_EPSILON:
                    a, b = lo, hi
                    fa = _eval_scalar(root_coeff, a)
                    for _bi in range(10):
                        m = 0.5 * (a + b)
                        fm = _eval_scalar(root_coeff, m)
                        if abs(fm) < abs(fa):
                            if fa * fm <= 0.0:
                                b = m
                            else:
                                a = m
                                fa = fm
                        elif fa * fm <= 0.0:
                            b = m
                        else:
                            a = m
                            fa = fm
                        if b - a <= _DBL_EPSILON:
                            break
                    mid = 0.5 * (a + b)
                f_final = _eval_scalar(root_coeff, mid)
                if abs(f_final) <= zero_tol and n_roots < max_roots:
                    roots[n_roots] = mid
                    n_roots += 1
            continue

        local = _restrict_scalar(root_coeff, lo, hi)
        local_scale = 0.0
        for i in range(n + 1):
            local_scale = max(local_scale, abs(local[i]))
        local_zero_tol = max(local_scale * (n + 1) * 4.0 * _DBL_EPSILON, geom_tol)

        c_min = local[0]
        c_max = local[0]
        for i in range(1, n + 1):
            c_min = min(c_min, local[i])
            c_max = max(c_max, local[i])

        if c_min > local_zero_tol or c_max < -local_zero_tol:
            rejection_margin = c_min if c_min > local_zero_tol else -c_max
            if rejection_margin <= zero_tol:
                mid = 0.5 * (lo + hi)
                mid, f_mid, _ = _newton_polish(root_coeff, mid, lo, hi, 0.0)
                if abs(f_mid) <= zero_tol and n_roots < max_roots:
                    roots[n_roots] = mid
                    n_roots += 1
            continue

        if c_max - c_min <= geom_tol:
            if abs(c_min) <= local_zero_tol or abs(c_max) <= local_zero_tol or c_min * c_max < 0.0:
                mid = 0.5 * (lo + hi)
                f_mid = _eval_scalar(root_coeff, mid)
                if abs(f_mid) <= zero_tol and n_roots < max_roots:
                    roots[n_roots] = mid
                    n_roots += 1
            continue

        if abs(local[0]) <= local_zero_tol and n_roots < max_roots:
            roots[n_roots] = lo
            n_roots += 1
        if abs(local[n]) <= local_zero_tol and n_roots < max_roots:
            roots[n_roots] = hi
            n_roots += 1

        n_sc = _count_sign_changes(local)
        if n_sc == 0:
            continue

        if n == 1:
            c0 = local[0]
            c1 = local[1]
            if c0 != c1:
                r = c0 / (c0 - c1)
                if 0.0 <= r <= 1.0 and n_roots < max_roots:
                    roots[n_roots] = lo + r * span
                    n_roots += 1
            continue

        t_lo_clip, t_hi_clip, clip_found = _clip_hull_to_zero(local)
        if not clip_found:
            if n_sc > 0 and stack_size + 1 < _MAX_STACK_SIZE:
                mid_param = 0.5 * (lo + hi)
                stack_lo[stack_size] = lo
                stack_hi[stack_size] = mid_param
                stack_depth[stack_size] = depth + 1
                stack_size += 1
                stack_lo[stack_size] = mid_param
                stack_hi[stack_size] = hi
                stack_depth[stack_size] = depth + 1
                stack_size += 1
            elif n_sc > 0:
                overflowed = True
            continue

        margin = (n + 1) * 4.0 * _DBL_EPSILON
        t_lo_safe = max(t_lo_clip - margin, 0.0)
        t_hi_safe = min(t_hi_clip + margin, 1.0)
        new_lo = lo + t_lo_safe * span
        new_hi = lo + t_hi_safe * span
        new_span = new_hi - new_lo

        if new_span <= param_tol:
            mid = 0.5 * (new_lo + new_hi)
            mid, f_mid, _ = _newton_polish(root_coeff, mid, new_lo, new_hi, param_tol)
            if abs(f_mid) <= zero_tol and n_roots < max_roots:
                roots[n_roots] = mid
                n_roots += 1
            continue

        reduction = 1.0 - (new_span / span) if span > 0.0 else 0.0

        if n_sc == 1:
            if reduction >= _CLIP_REDUCTION_THRESHOLD:
                if stack_size < _MAX_STACK_SIZE:
                    stack_lo[stack_size] = new_lo
                    stack_hi[stack_size] = new_hi
                    stack_depth[stack_size] = depth + 1
                    stack_size += 1
                else:
                    overflowed = True
            else:
                mid = 0.5 * (new_lo + new_hi)
                if stack_size + 1 < _MAX_STACK_SIZE:
                    stack_lo[stack_size] = new_lo
                    stack_hi[stack_size] = mid
                    stack_depth[stack_size] = depth + 1
                    stack_size += 1
                    stack_lo[stack_size] = mid
                    stack_hi[stack_size] = new_hi
                    stack_depth[stack_size] = depth + 1
                    stack_size += 1
                else:
                    overflowed = True
        elif reduction >= _CLIP_REDUCTION_THRESHOLD:
            mid = 0.5 * (new_lo + new_hi)
            if stack_size + 1 < _MAX_STACK_SIZE:
                stack_lo[stack_size] = new_lo
                stack_hi[stack_size] = mid
                stack_depth[stack_size] = depth + 1
                stack_size += 1
                stack_lo[stack_size] = mid
                stack_hi[stack_size] = new_hi
                stack_depth[stack_size] = depth + 1
                stack_size += 1
            else:
                overflowed = True
        else:
            mid = 0.5 * (lo + hi)
            if stack_size + 1 < _MAX_STACK_SIZE:
                stack_lo[stack_size] = lo
                stack_hi[stack_size] = mid
                stack_depth[stack_size] = depth + 1
                stack_size += 1
                stack_lo[stack_size] = mid
                stack_hi[stack_size] = hi
                stack_depth[stack_size] = depth + 1
                stack_size += 1
            else:
                overflowed = True

    return roots, n_roots, overflowed


# ---------------------------------------------------------------------------
# Section D: Deduplication (Numba-compiled)
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _dedup_roots(
    raw_roots: npt.NDArray[np.float64],
    n_roots: int,
    coeff: npt.NDArray[np.float64],
    param_tol: float,
    geom_tol: float,
) -> tuple[npt.NDArray[np.float64], int]:
    """Sort and deduplicate raw root candidates.

    Args:
        raw_roots (npt.NDArray[np.float64]): Unsorted root candidates.
        n_roots (int): Number of valid entries.
        coeff (npt.NDArray[np.float64]): Original Bernstein coefficients.
        param_tol (float): Parametric tolerance.
        geom_tol (float): Geometric tolerance.

    Returns:
        tuple[npt.NDArray[np.float64], int]: (unique_roots, count).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    if n_roots == 0:
        return raw_roots, 0

    # Insertion sort.
    for i in range(1, n_roots):
        key = raw_roots[i]
        j = i - 1
        while j >= 0 and raw_roots[j] > key:
            raw_roots[j + 1] = raw_roots[j]
            j -= 1
        raw_roots[j + 1] = key

    n = len(coeff) - 1
    coeff_scale = 0.0
    for i in range(n + 1):
        coeff_scale = max(coeff_scale, abs(coeff[i]))
    zero_tol = max(coeff_scale * (n + 1) * 4.0 * _DBL_EPSILON, geom_tol)
    base_dedup = max(param_tol * 2.0, zero_tol * 4.0)

    unique = np.empty(n_roots, dtype=np.float64)
    unique[0] = raw_roots[0]
    count = 1

    for i in range(1, n_roots):
        gap = raw_roots[i] - unique[count - 1]
        if gap <= base_dedup:
            continue
        _, df_val = _eval_and_deriv(coeff, unique[count - 1])
        local_tol = zero_tol / max(abs(df_val), _DBL_EPSILON)
        if gap <= max(base_dedup, local_tol * 4.0):
            continue
        unique[count] = raw_roots[i]
        count += 1

    return unique, count


# ---------------------------------------------------------------------------
# Section E: Dispatch and main entry point
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def find_roots(  # noqa: PLR0911, PLR0912, PLR0915
    coeffs: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], int, bool]:
    """Find all real roots of a 1D Bernstein polynomial in (0, 1).

    Dispatches between Yuksel and Bezier clipping based on degree and
    coefficient conditioning:

    - Degree < 6: Yuksel (lower overhead).
    - Degree >= 6 with coefficient range <= 1e8: Bezier clipping.
    - Otherwise: Yuksel.

    Args:
        coeffs (npt.NDArray[np.float64]): 1D Bernstein coefficients of
            length ``n + 1`` (degree n >= 0).

    Returns:
        tuple[npt.NDArray[np.float64], int, bool]: (roots_buffer, count,
            overflowed) where only the first *count* entries are valid
            roots, sorted in ascending order, and *overflowed* indicates
            whether the clipping stack was exhausted.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = len(coeffs) - 1
    if n <= 0:
        return np.empty(0, dtype=np.float64), 0, False

    # Quick rejection: uniform sign.
    d_min = coeffs[0]
    d_max = coeffs[0]
    for i in range(1, n + 1):
        d_min = min(d_min, coeffs[i])
        d_max = max(d_max, coeffs[i])
    if d_min > 0.0 or d_max < 0.0:
        return np.empty(0, dtype=np.float64), 0, False

    # All-zero check.
    coeff_scale = max(abs(d_min), abs(d_max))
    if coeff_scale <= _DBL_EPSILON:
        return np.empty(0, dtype=np.float64), 0, False

    # --- Fast path for degree 1 (linear) ---
    # Root: alpha[0] / (alpha[0] - alpha[1]), keep if in (0, 1).
    if n == 1:
        if coeffs[0] == coeffs[1]:
            return np.empty(0, dtype=np.float64), 0, False
        x = coeffs[0] / (coeffs[0] - coeffs[1])
        if x <= 0.0 or x >= 1.0:
            return np.empty(0, dtype=np.float64), 0, False
        roots = np.empty(1, dtype=np.float64)
        roots[0] = x
        return roots, 1, False

    # --- Fast path for degree 2 (quadratic) ---
    # Convert Bernstein to standard form: a*t^2 + b*t + c = 0.
    #   a = alpha[0] - 2*alpha[1] + alpha[2]
    #   b = 2*(alpha[1] - alpha[0])
    #   c = alpha[0]
    # Uses numerically stable quadratic formula (cf. algoim bernstein.hpp).
    if n == 2:  # noqa: PLR2004
        a = coeffs[0] - 2.0 * coeffs[1] + coeffs[2]
        b = 2.0 * (coeffs[1] - coeffs[0])
        c = coeffs[0]
        delta = b * b - 4.0 * a * c
        tol_delta = coeff_scale * 1.0e4 * _DBL_EPSILON
        if abs(delta) < tol_delta:
            delta = 0.0
        if delta < 0.0:
            return np.empty(0, dtype=np.float64), 0, False
        roots = np.empty(2, dtype=np.float64)
        count = 0
        if abs(a) < _DBL_EPSILON * coeff_scale:
            # Degenerate: linear equation b*t + c = 0.
            if abs(b) > _DBL_EPSILON * coeff_scale:
                x = -c / b
                if 0.0 < x < 1.0:
                    roots[0] = x
                    count = 1
        else:
            sqrt_delta = np.sqrt(delta)
            q_val = -0.5 * (b + sqrt_delta) if b >= 0.0 else -0.5 * (b - sqrt_delta)
            r1 = q_val / a
            r2 = c / q_val if abs(q_val) > 0.0 else -1.0
            if 0.0 < r1 < 1.0:
                roots[count] = r1
                count += 1
            if 0.0 < r2 < 1.0 and abs(r2 - r1) > _ROOT_TOL:
                roots[count] = r2
                count += 1
            # Sort if two roots found.
            if count == 2 and roots[0] > roots[1]:  # noqa: PLR2004
                roots[0], roots[1] = roots[1], roots[0]
        return roots, count, False

    tol = _ROOT_TOL

    if n < _CLIP_MIN_DEGREE:
        roots, count = _yuksel_roots(coeffs, tol)
        return roots, count, False

    # Check coefficient dynamic range for clipping dispatch.
    # Use relative threshold so mixed-scale polynomials are handled correctly.
    threshold = coeff_scale * _DBL_EPSILON
    c_min_nonzero = coeff_scale
    for i in range(n + 1):
        a = abs(coeffs[i])
        if a > threshold and a < c_min_nonzero:
            c_min_nonzero = a
    coeff_range = coeff_scale / c_min_nonzero

    if coeff_range <= _CLIP_COEFF_RANGE_LIMIT:
        raw, n_raw, clip_overflowed = _clip_roots_core(coeffs, tol, tol)
        unique, n_unique = _dedup_roots(raw, n_raw, coeffs, tol, tol)
        return unique, n_unique, clip_overflowed

    roots, count = _yuksel_roots(coeffs, tol)
    return roots, count, False


@nb_jit(nopython=True, cache=True)
def find_roots_into(  # noqa: PLR0911, PLR0912
    coeffs: npt.NDArray[np.float64],
    roots_buf: npt.NDArray[np.float64],
) -> tuple[int, bool]:
    """Find all real roots of a 1D Bernstein polynomial, writing into a pre-allocated buffer.

    Same algorithm as :func:`find_roots` but avoids allocating the roots array.
    For degree 1 and 2 (the fast paths in the construction hot loop), this
    eliminates 1-2 heap allocations per call.

    Args:
        coeffs (npt.NDArray[np.float64]): 1D Bernstein coefficients.
        roots_buf (npt.NDArray[np.float64]): Pre-allocated buffer for roots
            (must have length >= degree).

    Returns:
        tuple[int, bool]: (count, overflowed).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = len(coeffs) - 1
    if n <= 0:
        return 0, False

    d_min = coeffs[0]
    d_max = coeffs[0]
    for i in range(1, n + 1):
        d_min = min(d_min, coeffs[i])
        d_max = max(d_max, coeffs[i])
    if d_min > 0.0 or d_max < 0.0:
        return 0, False

    coeff_scale = max(abs(d_min), abs(d_max))
    if coeff_scale <= _DBL_EPSILON:
        return 0, False

    if n == 1:
        if coeffs[0] == coeffs[1]:
            return 0, False
        x = coeffs[0] / (coeffs[0] - coeffs[1])
        if x <= 0.0 or x >= 1.0:
            return 0, False
        roots_buf[0] = x
        return 1, False

    if n == 2:  # noqa: PLR2004
        a = coeffs[0] - 2.0 * coeffs[1] + coeffs[2]
        b = 2.0 * (coeffs[1] - coeffs[0])
        c = coeffs[0]
        delta = b * b - 4.0 * a * c
        tol_delta = coeff_scale * 1.0e4 * _DBL_EPSILON
        if abs(delta) < tol_delta:
            delta = 0.0
        if delta < 0.0:
            return 0, False
        count = 0
        if abs(a) < _DBL_EPSILON * coeff_scale:
            if abs(b) > _DBL_EPSILON * coeff_scale:
                x = -c / b
                if 0.0 < x < 1.0:
                    roots_buf[0] = x
                    count = 1
        else:
            sqrt_delta = np.sqrt(delta)
            q_val = -0.5 * (b + sqrt_delta) if b >= 0.0 else -0.5 * (b - sqrt_delta)
            r1 = q_val / a
            r2 = c / q_val if abs(q_val) > 0.0 else -1.0
            if 0.0 < r1 < 1.0:
                roots_buf[count] = r1
                count += 1
            if 0.0 < r2 < 1.0 and abs(r2 - r1) > _ROOT_TOL:
                roots_buf[count] = r2
                count += 1
            if count == 2 and roots_buf[0] > roots_buf[1]:  # noqa: PLR2004
                roots_buf[0], roots_buf[1] = roots_buf[1], roots_buf[0]
        return count, False

    # For higher degrees, fall back to the allocating version.
    roots, count, overflowed = find_roots(coeffs)
    for i in range(count):
        roots_buf[i] = roots[i]
    return count, overflowed
