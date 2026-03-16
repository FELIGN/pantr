"""Numba kernels for B-spline degree elevation.

These kernels implement the algorithms from The NURBS Book / MATLAB NURBS Toolbox layer.

Note:
    Inputs are assumed to be correct (no validation performed).
    For general use, call _degree_elevate_bspline instead.
"""

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from ._numba_compat import nb_jit


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
