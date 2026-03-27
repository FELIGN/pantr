"""Root finding via Bezier clipping for polynomials in the Bernstein basis.

Implements Bezier clipping, a root-finding algorithm that iteratively clips
the parameter domain by exploiting the convex hull property of the Bernstein
control polygon. Converges at a super-linear rate and avoids the derivative
chain, making it more stable for high-degree polynomials (e.g. degree-26
projection polynomials arising from rational curves with d=9).

Main exports:

- :func:`_clip_roots_core` -- stack-based iterative clipping loop (Numba-compiled).
- :func:`_dedup_roots` -- sort and deduplicate raw root candidates with
  derivative-aware merge radius (plain Python).

References:
    T. Nishita, T. W. Sederberg, M. Kakimoto (1990), *Ray Tracing Trimmed
    Rational Surface Patches*.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from pantr._numba_compat import nb_jit
from pantr.root_finding._root_finding_core import (
    _DBL_EPSILON,
    _clip_hull_to_zero,
    _count_sign_changes,
    _de_casteljau_eval_and_deriv_scalar,
    _de_casteljau_eval_scalar,
    _newton_polish_scalar,
    _subdivide_scalar,
)

_CLIP_REDUCTION_THRESHOLD: float = 0.2
"""Minimum parameter reduction to accept a clip without subdivision."""

_CLIP_MAX_DEPTH: int = 64
"""Maximum recursion depth for the clipping stack."""

_MAX_STACK_SIZE: int = 256
"""Maximum stack size for the iterative clipping loop."""


@nb_jit(nopython=True, cache=True)
def _clip_roots_core(  # noqa: PLR0912, PLR0915
    root_coeff: npt.NDArray[np.float32 | np.float64],
    param_tol: float,
    geom_tol: float,
) -> tuple[npt.NDArray[np.float64], int]:
    """Stack-based Bezier clipping root finder (Numba-compiled core).

    Always subdivides from the original root coefficients to avoid accumulated
    floating-point error from repeated de Casteljau splits.

    Args:
        root_coeff (npt.NDArray[np.float32 | np.float64]): Original Bernstein coefficients
            on [0, 1].
        param_tol (float): Parametric tolerance (bracket-width termination).
        geom_tol (float): Geometric tolerance for near-zero detection.

    Returns:
        tuple[npt.NDArray[np.float64], int]: ``(roots_array, count)`` where
            only the first ``count`` entries are valid. Roots are unsorted and
            may contain duplicates.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.root_finding.find_roots` instead.
    """
    n = len(root_coeff) - 1

    # Pre-allocate output. A degree-n polynomial has at most n roots,
    # but endpoint checks may produce duplicates before dedup.
    max_roots = 3 * n + 4
    roots = np.empty(max_roots, dtype=np.float64)
    n_roots = 0

    # Global coefficient scale for tolerance computation.
    coeff_scale = 0.0
    for i in range(n + 1):
        coeff_scale = max(coeff_scale, abs(root_coeff[i]))

    # Zero tolerance: de Casteljau evaluation error bound with 4x safety.
    zero_tol = max(coeff_scale * (n + 1) * 4.0 * _DBL_EPSILON, geom_tol)

    # Check boundary roots.
    if abs(root_coeff[0]) <= zero_tol:
        roots[n_roots] = 0.0
        n_roots += 1
    if abs(root_coeff[n]) <= zero_tol:
        roots[n_roots] = 1.0
        n_roots += 1

    # Stack: three parallel arrays for (lo, hi, depth).
    stack_lo = np.empty(_MAX_STACK_SIZE, dtype=np.float64)
    stack_hi = np.empty(_MAX_STACK_SIZE, dtype=np.float64)
    stack_depth = np.empty(_MAX_STACK_SIZE, dtype=np.int64)
    stack_lo[0] = 0.0
    stack_hi[0] = 1.0
    stack_depth[0] = 0
    stack_size = 1

    while stack_size > 0:
        # Pop.
        stack_size -= 1
        lo = stack_lo[stack_size]
        hi = stack_hi[stack_size]
        depth = stack_depth[stack_size]
        span = hi - lo

        # ---- Step 1: convergence ----------------------------------------
        if span <= param_tol or depth > _CLIP_MAX_DEPTH:
            mid = 0.5 * (lo + hi)
            # Snap to domain boundary when straddling.
            if lo <= 0.0 and hi <= param_tol:
                mid = 0.0
            elif lo >= 1.0 - param_tol and hi >= 1.0:
                mid = 1.0

            # Newton polish on the ORIGINAL polynomial.
            mid, f_mid, df_mid = _newton_polish_scalar(root_coeff, mid, lo, hi, param_tol)

            if abs(f_mid) <= zero_tol:
                if abs(df_mid) <= _DBL_EPSILON:
                    # f' near zero (double root) -- bisection refinement.
                    a, b = lo, hi
                    fa = _de_casteljau_eval_scalar(root_coeff, a)
                    for _bisect_iter in range(10):
                        m = 0.5 * (a + b)
                        fm = _de_casteljau_eval_scalar(root_coeff, m)
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

                # Final residual check.
                f_final = _de_casteljau_eval_scalar(root_coeff, mid)
                if abs(f_final) <= zero_tol and n_roots < max_roots:
                    roots[n_roots] = mid
                    n_roots += 1
            continue

        # ---- Step 2: extract local coefficients -------------------------
        local = _subdivide_scalar(root_coeff, lo, hi)

        # Local coefficient scale.
        local_scale = 0.0
        for i in range(n + 1):
            v = abs(local[i])
            local_scale = max(local_scale, v)
        local_zero_tol = max(local_scale * (n + 1) * 4.0 * _DBL_EPSILON, geom_tol)

        # ---- Step 3: coefficient range ----------------------------------
        c_min = local[0]
        c_max = local[0]
        for i in range(1, n + 1):
            c_min = min(c_min, local[i])
            c_max = max(c_max, local[i])

        # ---- Step 4: quick rejection (all same sign) --------------------
        if c_min > local_zero_tol or c_max < -local_zero_tol:
            rejection_margin = c_min if c_min > local_zero_tol else -c_max
            if rejection_margin <= zero_tol:
                mid = 0.5 * (lo + hi)
                mid, f_mid, _ = _newton_polish_scalar(root_coeff, mid, lo, hi, 0.0)
                if abs(f_mid) <= zero_tol and n_roots < max_roots:
                    roots[n_roots] = mid
                    n_roots += 1
            continue

        # ---- Step 5: near-zero polynomial (flat within noise) -----------
        if c_max - c_min <= geom_tol:
            if abs(c_min) <= local_zero_tol or abs(c_max) <= local_zero_tol or c_min * c_max < 0.0:
                mid = 0.5 * (lo + hi)
                f_mid = _de_casteljau_eval_scalar(root_coeff, mid)
                if abs(f_mid) <= zero_tol and n_roots < max_roots:
                    roots[n_roots] = mid
                    n_roots += 1
            continue

        # ---- Step 6: endpoint root detection ----------------------------
        if abs(local[0]) <= local_zero_tol and n_roots < max_roots:
            roots[n_roots] = lo
            n_roots += 1
        if abs(local[n]) <= local_zero_tol and n_roots < max_roots:
            roots[n_roots] = hi
            n_roots += 1

        # ---- Step 7: sign changes (VDP) ---------------------------------
        n_sc = _count_sign_changes(local)
        if n_sc == 0:
            continue

        # ---- Step 8: linear base case -----------------------------------
        if n == 1:
            c0 = local[0]
            c1 = local[1]
            if c0 != c1:
                r = c0 / (c0 - c1)
                if 0.0 <= r <= 1.0 and n_roots < max_roots:
                    roots[n_roots] = lo + r * span
                    n_roots += 1
            continue

        # ---- Step 9: convex hull clipping -------------------------------
        t_lo_clip, t_hi_clip, clip_found = _clip_hull_to_zero(local)
        if not clip_found:
            # Fall back to uniform subdivision.
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
            continue

        # Small safety margin against rounding in hull crossing.
        margin = (n + 1) * 4.0 * _DBL_EPSILON
        t_lo_clip_safe = t_lo_clip - margin
        t_lo_clip_safe = max(t_lo_clip_safe, 0.0)
        t_hi_clip_safe = t_hi_clip + margin
        t_hi_clip_safe = min(t_hi_clip_safe, 1.0)

        # Map to global coordinates.
        new_lo = lo + t_lo_clip_safe * span
        new_hi = lo + t_hi_clip_safe * span
        new_span = new_hi - new_lo

        # Narrow clipped interval: treat as converged.
        if new_span <= param_tol:
            mid = 0.5 * (new_lo + new_hi)
            mid, f_mid, _ = _newton_polish_scalar(root_coeff, mid, new_lo, new_hi, param_tol)
            if abs(f_mid) <= zero_tol and n_roots < max_roots:
                roots[n_roots] = mid
                n_roots += 1
            continue

        # ---- Step 10: reduction heuristic -------------------------------
        reduction = 1.0 - (new_span / span) if span > 0.0 else 0.0

        if n_sc == 1:
            if reduction >= _CLIP_REDUCTION_THRESHOLD:
                if stack_size < _MAX_STACK_SIZE:
                    stack_lo[stack_size] = new_lo
                    stack_hi[stack_size] = new_hi
                    stack_depth[stack_size] = depth + 1
                    stack_size += 1
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

    return roots, n_roots


def _dedup_roots(
    raw_roots: npt.NDArray[np.float32 | np.float64],
    n_roots: int,
    coeff: npt.NDArray[np.float32 | np.float64],
    param_tol: float,
    geom_tol: float,
) -> npt.NDArray[np.float64]:
    """Sort and deduplicate raw root candidates with derivative-aware merging.

    The same root can be reported from multiple converging intervals. The
    maximum gap between duplicates is ``O(zero_tol / |f'(root)|)``, so the
    dedup threshold is adaptive: for each candidate, the derivative is
    evaluated and the local merge radius is computed.

    Args:
        raw_roots (npt.NDArray[np.float32 | np.float64]): Unsorted array of root
            candidates. Only the first ``n_roots`` entries are valid.
        n_roots (int): Number of valid entries in ``raw_roots``.
        coeff (npt.NDArray[np.float32 | np.float64]): Original Bernstein coefficients
            (used for derivative evaluation during dedup).
        param_tol (float): Parametric tolerance.
        geom_tol (float): Geometric tolerance.

    Returns:
        npt.NDArray[np.float64]: Sorted, deduplicated array of unique roots.
    """
    if n_roots == 0:
        return np.empty(0, dtype=np.float64)

    n = len(coeff) - 1
    coeff_scale = float(np.max(np.abs(coeff)))
    zero_tol = max(coeff_scale * (n + 1) * 4.0 * _DBL_EPSILON, geom_tol)
    base_dedup = max(param_tol * 2.0, zero_tol * 4.0)

    result = sorted(float(raw_roots[i]) for i in range(n_roots))
    unique: list[float] = [result[0]]
    for r in result[1:]:
        gap = r - unique[-1]
        if gap <= base_dedup:
            continue
        # Derivative-aware merge.
        _, df = _de_casteljau_eval_and_deriv_scalar(coeff, unique[-1])
        local_tol = zero_tol / max(abs(df), _DBL_EPSILON)
        if gap <= max(base_dedup, local_tol * 4.0):
            continue
        unique.append(r)

    return np.array(unique, dtype=np.float64)


def _warmup_numba_functions() -> None:
    """Trigger Numba compilation of all kernels in this module.

    Called from the background warmup thread in ``pantr.__init__``.
    """
    dummy = np.array([1.0, -1.0, 0.5], dtype=np.float64)
    _clip_roots_core(dummy, 1e-12, 1e-12)
