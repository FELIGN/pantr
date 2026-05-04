"""Shared Numba-compiled helpers for Bernstein polynomial root finding.

Provides scalar de Casteljau evaluation, subdivision, sign-change counting,
convex-hull clipping, and Newton polishing -- all operating on 1-D Bernstein
coefficient arrays. These are inner building blocks called from within the
Yuksel and Bezier clipping algorithm kernels.

Main exports (all Numba-compiled, no ``parallel=True``):

- :func:`_de_casteljau_eval_scalar` -- evaluate at a single parameter.
- :func:`_de_casteljau_eval_and_deriv_scalar` -- evaluate value and derivative.
- :func:`_subdivide_scalar` -- extract coefficients for a sub-interval.
- :func:`_count_sign_changes` -- Variation Diminishing Property sign-change count.
- :func:`_clip_hull_to_zero` -- O(n) convex-hull clipping against y = 0.
- :func:`_newton_polish_scalar` -- single Newton refinement step.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from pantr._numba_compat import nb_jit

_DBL_EPSILON: float = 2.2204460492503131e-16
"""Machine epsilon for IEEE 754 double precision."""


@nb_jit(nopython=True, cache=True)
def _de_casteljau_eval_scalar(
    coeff: npt.NDArray[np.float32 | np.float64],
    t: float,
) -> float:
    """Evaluate a scalar Bernstein polynomial at parameter *t*.

    Computes B(t) = sum_i c_i * B_i^n(t) using the numerically stable
    de Casteljau triangle.

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): 1-D Bernstein coefficients of length
            ``n + 1``.
        t (float): Parameter value in [0, 1].

    Returns:
        float: Polynomial value B(t).

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer 2 helpers in ``_find_roots`` instead.
    """
    work = coeff.copy()
    n = len(work) - 1
    for k in range(1, n + 1):
        for i in range(n - k + 1):
            work[i] = (1.0 - t) * work[i] + t * work[i + 1]
    return float(work[0])


@nb_jit(nopython=True, cache=True)
def _restrict_scalar(
    coeff: npt.NDArray[np.float32 | np.float64],
    lower: float,
    upper: float,
) -> npt.NDArray[np.float64]:
    r"""Restrict a scalar Bernstein polynomial to ``[lower, upper]``.

    Uses a numerically stable two-pass de Casteljau strategy, choosing
    the pass order to avoid dividing by a small number.

    - If ``|upper| >= |lower - 1|``: left pass at ``upper``, then right
      pass at ``lower / upper``.
    - Otherwise: right pass at ``lower``, then left pass at
      ``(upper - lower) / (1 - lower)``.

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): 1-D Bernstein
            coefficients of length ``n + 1``.
        lower (float): Left bound of the sub-interval in ``[0, 1)``.
        upper (float): Right bound of the sub-interval in ``(0, 1]``.

    Returns:
        npt.NDArray[np.float64]: Restricted Bernstein coefficients
            reparametrized to [0, 1].

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer 2 helpers in ``_find_roots`` instead.
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
def _de_casteljau_eval_and_deriv_scalar(
    coeff: npt.NDArray[np.float32 | np.float64],
    t: float,
) -> tuple[float, float]:
    """Evaluate a scalar Bernstein polynomial and its derivative at *t*.

    Runs the de Casteljau triangle to the penultimate row (length 2) to obtain
    ``f'(t) = n * (row[1] - row[0])``, then one final reduction gives ``f(t)``.
    For degree 0, the derivative is identically 0.

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): 1-D Bernstein coefficients of length
            ``n + 1``.
        t (float): Parameter value in [0, 1].

    Returns:
        tuple[float, float]: ``(f(t), f'(t))``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer 2 helpers in ``_find_roots`` instead.
    """
    n = len(coeff) - 1

    if n == 0:
        return float(coeff[0]), 0.0

    s = 1.0 - t
    row = coeff.copy()

    for k in range(1, n):
        for i in range(n - k + 1):
            row[i] = s * row[i] + t * row[i + 1]

    # Penultimate row has length 2: row[0] and row[1].
    d0 = row[0]
    d1 = row[1]
    f = s * d0 + t * d1
    deriv = float(n) * (d1 - d0)

    return f, deriv


@nb_jit(nopython=True, cache=True)
def _subdivide_scalar(
    coeff: npt.NDArray[np.float32 | np.float64],
    t_min: float,
    t_max: float,
) -> npt.NDArray[np.float64]:
    """Extract Bernstein coefficients for the sub-interval ``[t_min, t_max]``.

    Thin adapter around :func:`_restrict_scalar`, which uses a numerically
    stable two-pass de Casteljau strategy (pass ordering chosen to avoid
    dividing by a small number).

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): Original 1-D Bernstein coefficients
            on [0, 1].
        t_min (float): Sub-interval start.
        t_max (float): Sub-interval end.

    Returns:
        npt.NDArray[np.float64]: Bernstein coefficients reparametrized to [0, 1].

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer 2 helpers in ``_find_roots`` instead.
    """
    if t_min <= 0.0 and t_max >= 1.0:
        return np.asarray(coeff, dtype=np.float64).copy()  # type: ignore[no-any-return]

    t_min = max(t_min, 0.0)
    t_min = min(t_min, 1.0)
    t_max = max(t_max, 0.0)
    t_max = min(t_max, 1.0)

    return _restrict_scalar(coeff, t_min, t_max)


@nb_jit(nopython=True, cache=True)
def _count_sign_changes(
    coeff: npt.NDArray[np.float32 | np.float64],
) -> int:
    """Count sign changes in a Bernstein coefficient sequence.

    By the Variation Diminishing Property, the number of real roots of a
    Bernstein polynomial on [0, 1] is at most the number of sign changes
    in its coefficient sequence, and has the same parity.

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): 1-D Bernstein coefficient array.

    Returns:
        int: Number of sign changes (ignoring zero coefficients).

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer 2 helpers in ``_find_roots`` instead.
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
        if prev_sign not in (0, s):
            changes += 1
        prev_sign = s
    return changes


@nb_jit(nopython=True, cache=True)
def _clip_hull_to_zero(  # noqa: PLR0912, PLR0915
    coeff: npt.NDArray[np.float32 | np.float64],
) -> tuple[float, float, bool]:
    """Clip the parameter range using the convex hull of the control polygon.

    The control polygon has vertices ``D_i = (i/n, c_i)`` with uniformly
    spaced x-coordinates. The upper and lower convex hulls are computed in
    O(n) time using Andrew's monotone chain algorithm (no sort needed since
    x-coordinates are already ordered), then all hull edges are checked for
    crossings with ``y = 0``.

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): Bernstein coefficients of shape
            ``(n + 1,)``.

    Returns:
        tuple[float, float, bool]: ``(t_lo, t_hi, found)`` where ``found``
            indicates whether any zero crossing was detected.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer 2 helpers in ``_find_roots`` instead.
    """
    n = len(coeff) - 1
    if n < 1:
        return 0.0, 0.0, False

    inv_n = 1.0 / n
    t_lo = 1.0
    t_hi = 0.0
    found = False

    # --- Upper hull (reject non-clockwise turns: cross >= 0) -----------
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

    # Check upper hull edges for y=0 crossings.
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
            ta = ia * inv_n
            t_lo = min(t_lo, ta)
            t_hi = max(t_hi, ta)
            found = True
    # Last vertex of upper hull.
    last_u = upper[u_size - 1]
    if coeff[last_u] == 0.0:
        ta = last_u * inv_n
        t_lo = min(t_lo, ta)
        t_hi = max(t_hi, ta)
        found = True

    # --- Lower hull (reject non-counter-clockwise turns: cross <= 0) ---
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

    # Check lower hull edges for y=0 crossings.
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
            ta = ia * inv_n
            t_lo = min(t_lo, ta)
            t_hi = max(t_hi, ta)
            found = True
    # Last vertex of lower hull.
    last_l = lower[l_size - 1]
    if coeff[last_l] == 0.0:
        ta = last_l * inv_n
        t_lo = min(t_lo, ta)
        t_hi = max(t_hi, ta)
        found = True

    return t_lo, t_hi, found


@nb_jit(nopython=True, cache=True)
def _newton_polish_scalar(
    coeff: npt.NDArray[np.float32 | np.float64],
    mid: float,
    lo: float,
    hi: float,
    param_tol: float,
) -> tuple[float, float, float]:
    """Polish a root candidate with a single Newton step.

    Evaluates ``f(mid)`` and ``f'(mid)`` via de Casteljau, then attempts one
    Newton iteration ``mid - f/f'``. The candidate is accepted only if it lies
    within ``[lo - param_tol, hi + param_tol]`` and reduces the absolute
    residual.

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): Original Bernstein coefficients on
            [0, 1].
        mid (float): Initial root estimate.
        lo (float): Left bound of the current parameter interval.
        hi (float): Right bound of the current parameter interval.
        param_tol (float): Parametric tolerance controlling the acceptance
            neighborhood around ``[lo, hi]``.

    Returns:
        tuple[float, float, float]: ``(polished_t, f_value, df_value)`` where
            ``polished_t`` is the (possibly improved) parameter, ``f_value``
            is its residual, and ``df_value`` is the derivative at the
            original ``mid``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer 2 helpers in ``_find_roots`` instead.
    """
    f_mid, df_mid = _de_casteljau_eval_and_deriv_scalar(coeff, mid)
    if abs(df_mid) > _DBL_EPSILON:
        newton = mid - f_mid / df_mid
        if lo - param_tol <= newton <= hi + param_tol:
            newton = max(0.0, min(1.0, newton))
            f_newton = _de_casteljau_eval_scalar(coeff, newton)
            if abs(f_newton) <= abs(f_mid):
                return newton, f_newton, df_mid
    return mid, f_mid, df_mid


def _warmup_numba_functions() -> None:
    """Trigger Numba compilation of all kernels in this module.

    Called from the background warmup thread in ``pantr.__init__``.
    """
    dummy = np.array([1.0, -1.0, 0.5], dtype=np.float64)
    _de_casteljau_eval_scalar(dummy, 0.5)
    _de_casteljau_eval_and_deriv_scalar(dummy, 0.5)
    _subdivide_scalar(dummy, 0.25, 0.75)
    _count_sign_changes(dummy)
    _clip_hull_to_zero(dummy)
    _newton_polish_scalar(dummy, 0.5, 0.0, 1.0, 1e-15)
