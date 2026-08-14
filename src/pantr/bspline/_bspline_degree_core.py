"""Numba kernels for B-spline degree elevation and reduction.

Elevation follows The NURBS Book (Piegl & Tiller, Algorithm A5.9) / MATLAB
NURBS Toolbox layer.  Reduction decomposes the spline into Bézier segments and
applies a precomputed reduction operator to each; the operator itself is
assembled in :mod:`pantr.bezier._bezier_degree`.

Note:
    Inputs are assumed to be correct (no validation performed).
    For general use, call ``_degree_elevate_bspline`` or
    ``_degree_reduce_bspline`` instead.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit

_BINCOEFF_MAX_N: int = 61
"""Largest ``n`` for which :func:`_bincoeff` honours its contract.

The exact-integer recurrence is safe while its largest intermediate,
``C(n, k) * min(k, n - k)``, fits in ``int64``: ``6.98e18`` at ``n = 61`` against an
``int64`` ceiling of ``9.22e18``, and ``1.44e19`` at ``n = 62``. Numba does not trap
integer overflow, so from ``n = 62`` an intermediate wraps and the returned value is
silently wrong (negative, at ``(62, 31)``).

This is the *integer* limit, deliberately, and not the tighter ``n = 56`` beyond which
the ``float`` return can no longer hold ``C(n, k)`` exactly. Every caller in the
library consumes ``_bincoeff`` only inside a floating-point **ratio** of binomial
coefficients -- ``C(d, j) * C(t, i - j) / C(ph, i)`` in degree elevation,
``C(p, i) * C(q, j) / C(r, k)`` in the Bernstein product -- so those ratios are formed
in floating point regardless and a correctly rounded operand is all they can use. For
``57 <= n <= 61`` that is exactly what :func:`_bincoeff` delivers: the integer is exact
and the cast rounds once. Capping at 56 instead would reject usable degrees for no
gain in the accuracy of any result the library actually computes.
"""


def _check_bincoeff_envelope(n: int, what: str) -> None:
    """Raise if ``n`` exceeds the exactness envelope of :func:`_bincoeff`.

    A Layer-2 validator: it lives beside the kernel whose contract it enforces so the
    two cannot drift apart, but it is plain Python and is never called from inside a
    kernel. Layer-3 kernels here still validate nothing.

    Args:
        n (int): Largest upper index the caller will pass to :func:`_bincoeff`.
        what (str): Description of the requested operation, used in the error message.

    Raises:
        ValueError: If ``n`` exceeds :data:`_BINCOEFF_MAX_N`.
    """
    if n > _BINCOEFF_MAX_N:
        raise ValueError(
            f"{what} needs binomial coefficients up to C({n}, k), beyond the largest "
            f"upper index {_BINCOEFF_MAX_N} that pantr's exact-integer binomial kernel "
            f"can compute without an int64 overflow. Past that the coefficients wrap "
            f"silently and the result is corrupted rather than merely inaccurate."
        )


@nb_jit(nopython=True, cache=True)
def _bincoeff(n: int, k: int) -> float:
    """Compute the binomial coefficient ``C(n, k)`` in exact integer arithmetic.

    Runs the multiplicative recurrence ``C(m, i) = C(m - 1, i - 1) * m // i`` over the
    smaller of ``k`` and ``n - k``. The running product after step ``i`` is exactly
    ``C(n - kk + i, i)``, an integer, so every division is exact and the result carries
    no rounding at all.

    Args:
        n (int): Upper index.
        k (int): Lower index.

    Returns:
        float: ``C(n, k)``, or ``0.0`` when ``k`` lies outside ``[0, n]``.

    Note:
        Inputs are assumed to be correct (no validation performed).

        Exactness envelope, measured against :func:`math.comb`. The largest intermediate
        is ``C(n, k) * min(k, n - k)``, so the integer arithmetic is exact while that
        fits in ``int64``: every ``k`` up to ``n = 61``. From ``n = 62`` an intermediate
        wraps silently (Numba does not trap integer overflow) and the result is outside
        the contract. Independently, the ``float`` return is lossless only while
        ``C(n, k) <= 2**53``, i.e. every ``k`` up to ``n = 56`` (``C(57, 28)`` is the
        first coefficient past it); between 57 and 61 the exact integer is computed and
        then correctly rounded, which is the most a float return can carry. Call sites
        pass ``p + t`` (degree elevation) and ``p + q`` (Bernstein product), so the
        envelope covers every numerically meaningful degree.

        The previous route, ``floor(0.5 + exp(lgamma(n + 1) - lgamma(k + 1) -
        lgamma(n - k + 1)))``, was silently wrong well inside both limits: it is off by
        380 at ``(57, 28)`` and by 5600 at ``(60, 30)``, since a value above ``2**53``
        cannot be recovered from logarithms. Where exactly it starts failing depends on
        the ``libm`` in use -- compiled here it already differed from ``(48, 25)`` on,
        while CPython's own ``lgamma`` still agreed there -- which is the other reason to
        prefer integer arithmetic: the envelope becomes a property of the algorithm
        rather than of the platform.
    """
    if k < 0 or k > n:
        return 0.0
    kk = min(k, n - k)
    result = np.int64(1)
    for i in range(1, kk + 1):
        result = result * np.int64(n - kk + i) // np.int64(i)
    return float(result)


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
def _apply_reduction_operator(
    operator: npt.NDArray[np.float64],
    ctrl: npt.NDArray[Any],
    out: npt.NDArray[Any],
) -> None:
    r"""Apply a degree-reduction operator to one set of Bézier control points.

    Computes :math:`\hat q = R\, c` for the dense operator ``R`` assembled by
    :func:`~pantr.bezier._bezier_degree._interpolating_reduction_operator` (or
    its unconstrained sibling), accumulating in ``float64`` regardless of the
    control-point dtype and rounding once on the write.

    Rows of ``R`` that pin an endpoint are unit vectors, so the corresponding
    output values reproduce their inputs bit for bit.

    This kernel lives here rather than in ``pantr.bezier`` because
    ``_bezier_core`` already imports ``_bincoeff`` from this module; the reverse
    import would close a cycle.

    Args:
        operator (npt.NDArray[np.float64]): Reduction operator of shape
            ``(q + 1, p + 1)``.
        ctrl (npt.NDArray[Any]): Control points of shape ``(p + 1, rank)``.
        out (npt.NDArray[Any]): Pre-allocated output of shape ``(q + 1, rank)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_degree_reduce_bspline` or
        :func:`~pantr.bezier._bezier_degree._degree_reduce_bezier` instead.
    """
    n_out = operator.shape[0]
    n_in = operator.shape[1]
    rank = ctrl.shape[1]

    for i in range(n_out):
        for r in range(rank):
            acc = 0.0
            for j in range(n_in):
                acc += operator[i, j] * np.float64(ctrl[j, r])
            out[i, r] = out.dtype.type(acc)


@nb_jit(nopython=True, cache=True)
def _degree_reduce_1d_core(  # noqa: PLR0912, PLR0915
    degree: int,
    ctrl: npt.NDArray[Any],
    knots: npt.NDArray[Any],
    degree_decrement: int,
    reduction_operator: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Degree reduce a B-spline curve by ``degree_decrement``.

    Decomposes the B-spline into Bézier segments by iterating through knot
    spans (same alpha-blending as the elevation kernel), applies the reduction
    operator to each segment, and stitches the results into a B-spline in Bézier
    form (C0 at every interior breakpoint).

    Because the operator interpolates the segment endpoints, the last control
    point of one reduced segment and the first of the next are both the shared
    breakpoint value of the *original* spline, bit for bit.  The junction is
    therefore written once and the stitch is exactly C0; no averaging of the two
    sides is involved, and neither segment is perturbed away from its own
    optimum.

    Args:
        degree (int): Original degree.
        ctrl (npt.NDArray[Any]): Control points of shape ``(n_pts, rank)``.
        knots (npt.NDArray[Any]): Knot vector of shape ``(n_knots,)``.
        degree_decrement (int): Number of degrees to reduce (``1 <= t <= degree``).
        reduction_operator (npt.NDArray[np.float64]): Operator of shape
            ``(degree - t + 1, degree + 1)`` for this degree pair.

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

    # Size the output arrays from the number of Bézier segments the sweep below
    # will produce, counted with that sweep's own advance rule.  The reduced
    # spline is in Bézier form, so it has exactly ``n_seg * new_deg + 1`` control
    # points and ``n_seg * new_deg + new_deg + 2`` knots.  (A bound expressed in
    # terms of ``len(knots)`` alone is not an upper bound: it fails from 8
    # elements on at degree 5, and the kernel then writes past the end.)
    n_seg = 0
    scan = d + 1
    while scan < m:
        while scan < m and knots[scan] == knots[scan + 1]:
            scan += 1
        n_seg += 1
        scan += 1

    oc = np.empty((n_seg * new_deg + 1, rank), dtype=ctrl.dtype)
    ok = np.empty(n_seg * new_deg + new_deg + 2, dtype=np.float64)

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
        oldr = r  # noqa: F841
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
        _apply_reduction_operator(reduction_operator, bpts, rbpts)

        # The first segment contributes all of its control points; every later
        # one skips its first, which the previous segment already wrote with the
        # identical value.
        start = 0 if cind == 0 else 1
        for j in range(start, new_deg + 1):
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
