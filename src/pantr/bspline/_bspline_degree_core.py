"""Numba kernels for B-spline degree elevation and reduction.

These kernels implement the algorithms from The NURBS Book / MATLAB NURBS
Toolbox layer and the bidiagonal least-squares reduction from algoim.

Note:
    Inputs are assumed to be correct (no validation performed).
    For general use, call ``_degree_elevate_bspline`` or
    ``_degree_reduce_bspline`` instead.
"""

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit


@nb_jit(nopython=True, cache=True)
def _bincoeff(n: int, k: int) -> float:
    """Compute binomial coefficient (n choose k)."""
    if k < 0 or k > n:
        return 0.0
    if k in (0, n):
        return 1.0
    return math.floor(
        0.5 + math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))
    )


@nb_jit(nopython=True, cache=True)
def _degree_elevate_1d_core(  # noqa: PLR0912, PLR0915
    degree: int,
    ctrl: npt.NDArray[Any],
    knots: npt.NDArray[Any],
    degree_increment: int,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Degree elevate a B-spline curve by degree_increment.

    Implements a robust version of Piegl & Tiller Algorithm A5.9
    (Degree elevate a B-spline curve).

    Args:
        degree (int): Original degree.
        ctrl (np.ndarray): Control points of shape (n_pts, rank).
        knots (np.ndarray): Knot vector of shape (n_knots,).
        degree_increment (int): How much to increase the degree.

    Returns:
        tuple[np.ndarray, np.ndarray]: Expanded control points and new knot vector.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer 2 helper instead.
    """
    n_pts = ctrl.shape[0]
    rank = ctrl.shape[1]
    d = degree
    t = degree_increment

    n = n_pts - 1
    ph = d + t
    ph2 = ph // 2

    bezalfs = np.zeros((d + 1, ph + 1), dtype=np.float64)
    bpts = np.zeros((d + 1, rank), dtype=ctrl.dtype)
    ebpts = np.zeros((ph + 1, rank), dtype=ctrl.dtype)
    Nextbpts = np.zeros((d + 1, rank), dtype=ctrl.dtype)
    alfs = np.zeros(d, dtype=np.float64)

    m = n + d + 1

    # compute bezier degree elevation coefficients
    bezalfs[0, 0] = 1.0
    bezalfs[d, ph] = 1.0

    for i in range(1, ph2 + 1):
        inv = 1.0 / _bincoeff(ph, i)
        mpi = min(d, i)
        for j in range(max(0, i - t), mpi + 1):
            bezalfs[j, i] = inv * _bincoeff(d, j) * _bincoeff(t, i - j)

    for i in range(ph2 + 1, ph):
        mpi = min(d, i)
        for j in range(max(0, i - t), mpi + 1):
            bezalfs[j, i] = bezalfs[d - j, ph - i]

    kind = ph + 1
    r = -1
    a = d
    b = d + 1
    cind = 1
    ua = knots[0]

    # We allocate more than enough space for the new arrays
    max_new_knots = len(knots) + t * len(knots)
    ik = np.zeros(max_new_knots, dtype=np.float64)
    ic = np.zeros((n_pts + t * len(knots), rank), dtype=ctrl.dtype)

    for ii in range(rank):
        ic[0, ii] = ctrl[0, ii]

    for i in range(ph + 1):
        ik[i] = ua

    for i in range(d + 1):
        for ii in range(rank):
            bpts[i, ii] = ctrl[i, ii]

    while b < m:
        i = b
        while b < m and knots[b] == knots[b + 1]:
            b += 1

        mul = b - i + 1
        ub = knots[b]
        oldr = r
        r = d - mul

        lbz = (oldr + 2) // 2 if oldr > 0 else 1

        rbz = ph - (r + 1) // 2 if r > 0 else ph

        if r > 0:
            numer = ub - ua
            for q in range(d, mul, -1):
                alfs[q - mul - 1] = numer / (knots[a + q] - ua)

            for j in range(1, r + 1):
                save = r - j
                s = mul + j

                for q in range(d, s - 1, -1):
                    for ii in range(rank):
                        tmp1 = alfs[q - s] * bpts[q, ii]
                        tmp2 = (1.0 - alfs[q - s]) * bpts[q - 1, ii]
                        bpts[q, ii] = tmp1 + tmp2

                for ii in range(rank):
                    Nextbpts[save, ii] = bpts[d, ii]

        for i in range(lbz, ph + 1):
            for ii in range(rank):
                ebpts[i, ii] = 0.0

            mpi = min(d, i)
            for j in range(max(0, i - t), mpi + 1):
                for ii in range(rank):
                    tmp2 = bezalfs[j, i] * bpts[j, ii]
                    ebpts[i, ii] += tmp2

        if oldr > 1:
            first = kind - 2
            last = kind
            den = ub - ua
            bet = (ub - ik[kind - 1]) / den

            for tr in range(1, oldr):
                i = first
                j = last
                kj = j - kind + 1
                while j - i > tr:
                    if i < cind:
                        alf = (ub - ik[i]) / (ua - ik[i])
                        for ii in range(rank):
                            tmp1 = alf * ic[i, ii]
                            tmp2 = (1.0 - alf) * ic[i - 1, ii]
                            ic[i, ii] = tmp1 + tmp2

                    if j >= lbz:
                        if j - tr <= kind - ph + oldr:
                            gam = (ub - ik[j - tr]) / den
                            for ii in range(rank):
                                tmp1 = gam * ebpts[kj, ii]
                                tmp2 = (1.0 - gam) * ebpts[kj + 1, ii]
                                ebpts[kj, ii] = tmp1 + tmp2
                        else:
                            for ii in range(rank):
                                tmp1 = bet * ebpts[kj, ii]
                                tmp2 = (1.0 - bet) * ebpts[kj + 1, ii]
                                ebpts[kj, ii] = tmp1 + tmp2

                    i += 1
                    j -= 1
                    kj -= 1

                first -= 1
                last += 1

        if a != d:
            for _i in range(ph - oldr):
                ik[kind] = ua
                kind += 1

        for j in range(lbz, rbz + 1):
            for ii in range(rank):
                ic[cind, ii] = ebpts[j, ii]
            cind += 1

        if b < m:
            for j in range(r):
                for ii in range(rank):
                    bpts[j, ii] = Nextbpts[j, ii]
            for j in range(r, d + 1):
                for ii in range(rank):
                    bpts[j, ii] = ctrl[b - d + j, ii]
            a = b
            b += 1
            ua = ub
        else:
            for i in range(ph + 1):
                ik[kind + i] = ub

    return ic[:cind].copy(), ik[: kind + ph + 1].copy()


@nb_jit(nopython=True, cache=True)
def _reduce_bezier_segment(  # noqa: PLR0912
    degree: int,
    bpts: npt.NDArray[Any],
    degree_decrement: int,
    out: npt.NDArray[Any],
) -> None:
    """Degree-reduce a single Bézier segment into a pre-allocated output.

    Implements the same bidiagonal least-squares algorithm as
    :func:`~pantr.bezier._bezier_core._degree_reduce_bezier_1d_core`.
    The code is duplicated here because a circular import prevents
    calling the Bézier kernel directly (``_bezier_core`` already
    imports ``_bincoeff`` from this module).

    Args:
        degree (int): Current polynomial degree (``p >= 1``).
        bpts (npt.NDArray[Any]): Input Bézier control points of shape
            ``(p + 1, rank)``.
        degree_decrement (int): Number of degrees to reduce (``1 <= t <= p``).
        out (npt.NDArray[Any]): Pre-allocated output of shape
            ``(p - t + 1, rank)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_degree_reduce_bspline` instead.
    """
    rank = bpts.shape[1]

    # Pre-allocate scratch arrays sized for the largest step (initial degree).
    diag = np.empty(degree, dtype=np.float64)
    sup = np.empty(degree, dtype=np.float64)
    rhs = np.empty(degree + 1, dtype=np.float64)
    # Store Givens cosines/sines to avoid recomputing per rank component.
    cs_arr = np.empty(degree, dtype=np.float64)
    sn_arr = np.empty(degree, dtype=np.float64)

    cur = np.empty((degree + 1, rank), dtype=bpts.dtype)
    for i in range(degree + 1):
        for r in range(rank):
            cur[i, r] = bpts[i, r]

    cur_deg = degree

    for _step in range(degree_decrement):
        p = cur_deg
        n_new = p

        nxt = np.empty((n_new, rank), dtype=bpts.dtype)

        # Compute Givens rotation coefficients (independent of rank).
        for k in range(p):
            diag[k] = 1.0 - np.float64(k) / np.float64(p)
        for k in range(p):
            sup[k] = 0.0

        for k in range(p):
            bk = np.float64(k + 1) / np.float64(p)
            ak = diag[k]
            if bk == 0.0:
                cs = 1.0
                sn = 0.0
            elif abs(bk) > abs(ak):
                tmp = ak / bk
                sn = 1.0 / np.sqrt(1.0 + tmp * tmp)
                cs = tmp * sn
            else:
                tmp = bk / ak
                cs = 1.0 / np.sqrt(1.0 + tmp * tmp)
                sn = tmp * cs

            cs_arr[k] = cs
            sn_arr[k] = sn
            diag[k] = cs * ak + sn * bk
            if k + 1 < p:
                sup[k] = sn * diag[k + 1]
                diag[k + 1] = cs * diag[k + 1]

        # Apply rotations and back-substitute for each rank component.
        for r in range(rank):
            for k in range(p + 1):
                rhs[k] = np.float64(cur[k, r])

            for k in range(p):
                cs = cs_arr[k]
                sn = sn_arr[k]
                rk = rhs[k]
                rhs[k] = cs * rk + sn * rhs[k + 1]
                rhs[k + 1] = -sn * rk + cs * rhs[k + 1]

            nxt[p - 1, r] = bpts.dtype.type(rhs[p - 1] / diag[p - 1])
            for k in range(p - 2, -1, -1):
                nxt[k, r] = bpts.dtype.type((rhs[k] - sup[k] * np.float64(nxt[k + 1, r])) / diag[k])

        cur = nxt
        cur_deg = p - 1

    for i in range(out.shape[0]):
        for r in range(rank):
            out[i, r] = cur[i, r]


@nb_jit(nopython=True, cache=True)
def _degree_reduce_1d_core(  # noqa: PLR0912, PLR0915
    degree: int,
    ctrl: npt.NDArray[Any],
    knots: npt.NDArray[Any],
    degree_decrement: int,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Degree reduce a B-spline curve by ``degree_decrement``.

    Decomposes the B-spline into Bézier segments by iterating through knot
    spans (same alpha-blending as the elevation kernel), reduces each segment
    via bidiagonal least-squares, and stitches the results into a B-spline in
    Bézier form (C0 at every interior breakpoint).

    Args:
        degree (int): Original degree.
        ctrl (npt.NDArray[Any]): Control points of shape ``(n_pts, rank)``.
        knots (npt.NDArray[Any]): Knot vector of shape ``(n_knots,)``.
        degree_decrement (int): Number of degrees to reduce (``1 <= t <= degree``).

    Returns:
        tuple[npt.NDArray[Any], npt.NDArray[Any]]: ``(reduced_ctrl, reduced_knots)``
        in Bézier form — all interior breakpoints have multiplicity ``new_degree``
        (C0 continuity).

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_degree_reduce_bspline` instead.
    """
    n_pts = ctrl.shape[0]
    rank = ctrl.shape[1]
    d = degree
    t = degree_decrement
    new_deg = d - t

    n = n_pts - 1
    m = n + d + 1  # last knot index

    bpts = np.zeros((d + 1, rank), dtype=ctrl.dtype)
    Nextbpts = np.zeros((d + 1, rank), dtype=ctrl.dtype)
    alfs = np.zeros(d, dtype=np.float64)
    rbpts = np.empty((new_deg + 1, rank), dtype=ctrl.dtype)

    # Over-allocate output arrays (same pattern as elevation kernel).
    max_new_knots = len(knots) + len(knots)
    oc = np.empty((n_pts + len(knots), rank), dtype=ctrl.dtype)
    ok = np.empty(max_new_knots, dtype=np.float64)

    # Initialise: first boundary knots and first Bézier segment.
    ua = knots[0]
    for i in range(new_deg + 1):
        ok[i] = ua
    kind = new_deg + 1

    for i in range(d + 1):
        for ii in range(rank):
            bpts[i, ii] = ctrl[i, ii]

    cind = 0  # output control point index
    a = d
    b = d + 1
    r = -1

    while b < m:
        # Find next distinct knot.
        i = b
        while b < m and knots[b] == knots[b + 1]:
            b += 1
        mul = b - i + 1
        ub = knots[b]
        oldr = r
        r = d - mul

        # Extract Bézier control points for the current span.
        if r > 0:
            numer = ub - ua
            for q in range(d, mul, -1):
                alfs[q - mul - 1] = numer / (knots[a + q] - ua)

            for j in range(1, r + 1):
                save = r - j
                s = mul + j
                for q in range(d, s - 1, -1):
                    for ii in range(rank):
                        bpts[q, ii] = (
                            alfs[q - s] * bpts[q, ii] + (1.0 - alfs[q - s]) * bpts[q - 1, ii]
                        )
                for ii in range(rank):
                    Nextbpts[save, ii] = bpts[d, ii]

        # --- Reduce the current Bézier segment ---
        _reduce_bezier_segment(d, bpts, t, rbpts)

        # If this is the very first segment, write all control points.
        # Otherwise, average the shared boundary point with the previous
        # segment's last point, then write the interior + end points.
        if cind == 0:
            # First segment: write all new_deg + 1 points.
            for j in range(new_deg + 1):
                for ii in range(rank):
                    oc[cind, ii] = rbpts[j, ii]
                cind += 1
        else:
            # Average the shared boundary point.
            for ii in range(rank):
                oc[cind - 1, ii] = (oc[cind - 1, ii] + rbpts[0, ii]) * 0.5
            # Write interior and end points.
            for j in range(1, new_deg + 1):
                for ii in range(rank):
                    oc[cind, ii] = rbpts[j, ii]
                cind += 1

        # Write interior breakpoint knots (multiplicity = new_deg for C0).
        if a != d:
            for _i in range(new_deg):
                ok[kind] = ua
                kind += 1

        # Prepare for next span.
        if b < m:
            for j in range(r):
                for ii in range(rank):
                    bpts[j, ii] = Nextbpts[j, ii]
            for j in range(r, d + 1):
                for ii in range(rank):
                    bpts[j, ii] = ctrl[b - d + j, ii]
            a = b
            b += 1
            ua = ub
        else:
            # Write closing knots.
            for i in range(new_deg + 1):
                ok[kind + i] = ub

    return oc[:cind].copy(), ok[: kind + new_deg + 1].copy()
