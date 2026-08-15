"""Numba kernels for spline root finding by the Mørken-Reimers method.

Computes the zeros of a scalar B-spline directly on its own knot vector, by
repeatedly inserting the zero of the control polygon as a new knot
:cite:p:`morken2007zeros`. Every arithmetic operation is a convex combination
(Boehm knot insertion), the iteration needs no starting value, and it converges
for any spline: the control polygon of a spline with a zero always has a sign
change, and the variation-diminishing property keeps the sign-change count from
growing under refinement.

The kernels refine one working copy of the whole knot vector in place. Reporting
a zero splits the spline there, raising the zero to knot multiplicity
``degree + 1``, and everything to its left is then dropped: what is left is again
an open (clamped) knot vector, this time on ``[zero, end]``. That is what bounds
the memory, since the insertions spent on one zero are discarded when the next
one is reported, and it is what keeps every span search inside the buffer.

Nothing here is reported on the strength of the branch that produced it. The
paper's certificates -- a repeated iterate, an iterate at the end of its Greville
interval -- are theorems of exact arithmetic, and a floating-point iteration can
satisfy their conclusions' *tests* without being anywhere near a zero. So
:func:`_track_zero` hands back ``|f(x)|`` on every exit, its status says only how
the iteration stopped, and :func:`_morken_reimers_roots` reports an iterate if and
only if that residual is within ``root_tol``.

Note:
    Inputs are assumed to be correct (no validation performed).
    For general use, call :func:`pantr.bspline.find_roots` instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit

_STATUS_STAGNATED: int = 1
"""The iteration reached a point it will not leave.

Covers the three stopping rules: an iterate repeated exactly, the last ``degree``
iterates spanned less than ``tol``, and an iterate reached the right end of its
Greville interval, where every value of ``lam`` gives the same point.
"""

_STATUS_VANISHED: int = 2
"""The tracked sign change disappeared under refinement.

Either a tangential zero, where the two sign changes bracketing it collapse, or a
false warning of the variation-diminishing bound.
"""

_STATUS_BUDGET: int = 3
"""The insertion budget ran out before the iteration stopped."""

_COROLLARY_14_DEGREE: int = 2
"""Lowest degree at which Corollary 14 of Mørken-Reimers asks for a collapsed knot run.

The corollary needs ``degree - 1`` active knots collapsed onto the iterate, so at
degree 1 it asks for none and a repeated iterate satisfies it on its own.
"""


@nb_jit(nopython=True, cache=True)
def _knot_average(knots: npt.NDArray[Any], degree: int, index: int) -> float:
    """Compute the Greville abscissa ``(t[index+1] + ... + t[index+degree]) / degree``.

    Args:
        knots (npt.NDArray[Any]): Knot vector.
        degree (int): Polynomial degree, at least 1.
        index (int): Coefficient index whose knot average is requested.

    Returns:
        float: The knot average (Greville abscissa) of coefficient ``index``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.bspline.find_roots` instead.
    """
    total = 0.0
    for j in range(index + 1, index + degree + 1):
        total += knots[j]
    return total / degree


@nb_jit(nopython=True, cache=True)
def _zero_index(coeffs: npt.NDArray[Any], num_coeffs: int, start: int) -> int:
    """Find the smallest zero index of the control polygon at or after ``start``.

    Following Mørken and Reimers, ``a`` is a zero index when ``coeffs[a - 1]`` is
    non-zero and ``coeffs[a - 1] * coeffs[a] <= 0``; the non-zero requirement is
    what keeps the secant denominator ``coeffs[a] - coeffs[a - 1]`` away from
    zero across a run of vanishing coefficients.

    Args:
        coeffs (npt.NDArray[Any]): B-spline coefficients.
        num_coeffs (int): Number of valid entries in ``coeffs``.
        start (int): First index to test; clamped to 1 from below.

    Returns:
        int: The zero index, or ``-1`` if the polygon has no sign change left.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.bspline.find_roots` instead.
    """
    index = start if start > 1 else 1
    while index < num_coeffs:
        previous = coeffs[index - 1]
        if previous != 0.0 and previous * coeffs[index] <= 0.0:
            return index
        index += 1
    return -1


@nb_jit(nopython=True, cache=True)
def _is_zero_index(coeffs: npt.NDArray[Any], num_coeffs: int, index: int) -> bool:
    """Check whether ``index`` is a zero index of the control polygon.

    Args:
        coeffs (npt.NDArray[Any]): B-spline coefficients.
        num_coeffs (int): Number of valid entries in ``coeffs``.
        index (int): Index to test.

    Returns:
        bool: True if ``coeffs[index - 1]`` is non-zero and the product
        ``coeffs[index - 1] * coeffs[index]`` is non-positive.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.bspline.find_roots` instead.
    """
    if index < 1 or index >= num_coeffs:
        return False
    previous = coeffs[index - 1]
    return bool(previous != 0.0 and previous * coeffs[index] <= 0.0)


@nb_jit(nopython=True, cache=True)
def _deboor_point(  # noqa: PLR0913
    knots: npt.NDArray[Any],
    coeffs: npt.NDArray[Any],
    degree: int,
    point: float,
    span: int,
    work: npt.NDArray[Any],
) -> float:
    """Evaluate a scalar spline at one point with de Boor's algorithm.

    Args:
        knots (npt.NDArray[Any]): Knot vector.
        coeffs (npt.NDArray[Any]): B-spline coefficients.
        degree (int): Polynomial degree.
        point (float): Evaluation point, inside ``[knots[span], knots[span+1]]``.
        span (int): Knot span index of ``point``.
        work (npt.NDArray[Any]): Scratch buffer of at least ``degree + 1`` entries.

    Returns:
        float: The value of the spline at ``point``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.bspline.find_roots` instead.
    """
    for i in range(degree + 1):
        work[i] = coeffs[span - degree + i]
    for level in range(1, degree + 1):
        for i in range(degree, level - 1, -1):
            left = span - degree + i
            denominator = knots[left + degree - level + 1] - knots[left]
            alpha = (point - knots[left]) / denominator if denominator > 0.0 else 0.0
            work[i] = (1.0 - alpha) * work[i - 1] + alpha * work[i]
    return float(work[degree])


@nb_jit(nopython=True, cache=True)
def _span_at(
    knots: npt.NDArray[Any],
    num_coeffs: int,
    degree: int,
    point: float,
) -> int:
    """Locate the knot span of ``point`` in an open (clamped) knot window.

    The search is bounded above by ``num_coeffs - 1``, the last span of an open
    knot vector, so that a point at the end of the domain is handled from the
    left instead of walking off the end of the buffer.

    Args:
        knots (npt.NDArray[Any]): Open (clamped) knot window.
        num_coeffs (int): Number of valid coefficients in the window.
        degree (int): Polynomial degree.
        point (float): Point inside the window's domain.

    Returns:
        int: Span index in ``[degree, num_coeffs - 1]``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.bspline.find_roots` instead.
    """
    span = degree
    while span + 1 < num_coeffs and point >= knots[span + 1]:
        span += 1
    return span


@nb_jit(nopython=True, cache=True)
def _residual_at(  # noqa: PLR0913
    knots: npt.NDArray[Any],
    coeffs: npt.NDArray[Any],
    num_coeffs: int,
    degree: int,
    point: float,
    work: npt.NDArray[Any],
) -> float:
    """Evaluate ``|f(point)|`` on a refined window.

    This is the certificate every reported zero has to produce, so it is computed
    from the window as it stands rather than assumed from the branch that reached
    it.

    Args:
        knots (npt.NDArray[Any]): Open (clamped) knot window.
        coeffs (npt.NDArray[Any]): Coefficient window.
        num_coeffs (int): Number of valid entries in ``coeffs``.
        degree (int): Polynomial degree.
        point (float): Evaluation point, inside the window's domain.
        work (npt.NDArray[Any]): Scratch buffer of at least ``degree + 1`` entries.

    Returns:
        float: ``|f(point)|``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.bspline.find_roots` instead.
    """
    span = _span_at(knots, num_coeffs, degree, point)
    return abs(_deboor_point(knots, coeffs, degree, point, span, work))


@nb_jit(nopython=True, cache=True)
def _insert_knot(  # noqa: PLR0913
    knots: npt.NDArray[Any],
    coeffs: npt.NDArray[Any],
    num_coeffs: int,
    degree: int,
    point: float,
    span: int,
) -> None:
    """Insert one knot in place by Boehm's algorithm, growing both buffers by one.

    Args:
        knots (npt.NDArray[Any]): Knot buffer holding ``num_coeffs + degree + 1``
            valid entries, with room for one more.
        coeffs (npt.NDArray[Any]): Coefficient buffer holding ``num_coeffs``
            valid entries, with room for one more.
        num_coeffs (int): Number of valid coefficients before the insertion.
        degree (int): Polynomial degree.
        point (float): Knot value to insert.
        span (int): Knot span index of ``point``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.bspline.find_roots` instead.
    """
    for i in range(num_coeffs, span, -1):
        coeffs[i] = coeffs[i - 1]
    for i in range(span, span - degree, -1):
        alpha = (point - knots[i]) / (knots[i + degree] - knots[i])
        coeffs[i] = (1.0 - alpha) * coeffs[i - 1] + alpha * coeffs[i]

    for i in range(num_coeffs + degree + 1, span + 1, -1):
        knots[i] = knots[i - 1]
    knots[span + 1] = point


@nb_jit(nopython=True, cache=True)
def _split_at_root(  # noqa: PLR0913
    knots: npt.NDArray[Any],
    coeffs: npt.NDArray[Any],
    num_coeffs: int,
    degree: int,
    point: float,
    zero_tol: float,
) -> tuple[int, int]:
    """Raise ``point`` to multiplicity ``degree + 1`` and pin its coefficient to zero.

    Splitting the spline at a zero that has already been reported is what keeps
    the next sign change from drifting back onto it: a tracked sequence converges
    to *a* zero of the spline inside ``[t[a], t[a+degree]]``, not necessarily to
    the one its own Greville interval brackets, so an already-reported zero has
    to be removed from that bracket first. Mørken and Reimers prescribe exactly
    this, splitting the spline at each zero before looking for the next one.

    Multiplicity ``degree + 1`` is what makes the split a genuine one: the two
    sides then share no basis function, and dropping the left side leaves an
    open (clamped) knot vector on ``[point, end]``, so every later span search
    and every later Boehm insertion sees the same structure as the original
    input. Multiplicity ``degree`` would leave one knot of the old spline
    hanging to the left of the new domain, and a span search starting from a
    coefficient index would then be free to run off the front of the buffer.

    The first coefficient of that clamped remainder is ``f(point)``, which is
    zero because ``point`` is a zero of the spline, so pinning it to exactly zero
    removes the rounding left by the insertions and makes it a barrier:
    :func:`_zero_index` never reports an index whose left neighbour is exactly
    zero.

    Args:
        knots (npt.NDArray[Any]): Knot vector, refined in place.
        coeffs (npt.NDArray[Any]): Coefficients, refined in place.
        num_coeffs (int): Number of valid entries in ``coeffs``.
        degree (int): Polynomial degree.
        point (float): Zero of the spline, below the end of the domain.
        zero_tol (float): Residual below which a coefficient counts as zero.

    Returns:
        tuple[int, int]: ``(num_coeffs, offset)`` with the updated coefficient
        count and the index at which the remainder begins, that is the first
        index of the knot run carrying ``point``. Both arrays are to be shifted
        down by ``offset``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.bspline.find_roots` instead.
    """
    # Count only the run that ends at this span: the same value may well appear
    # elsewhere in the knot vector, and it is this run that has to be raised.
    span = 0
    while point >= knots[span + 1]:
        span += 1
    multiplicity = 0
    index = span
    while index >= 0 and knots[index] == point:
        multiplicity += 1
        index -= 1

    while multiplicity <= degree and num_coeffs < coeffs.shape[0]:
        _insert_knot(knots, coeffs, num_coeffs, degree, point, span)
        num_coeffs += 1
        multiplicity += 1
        span += 1

    offset = span
    while offset > 0 and knots[offset - 1] == point:
        offset -= 1

    coeffs[offset] = 0.0
    # The coefficients just right of the barrier average knots that are all the
    # split point, so they carry f(point) = 0 up to rounding. Left as they are,
    # their arbitrary sign seeds a sign change that tracks straight back to the
    # split point; pinning them extends the barrier.
    i = offset + 1
    while i < num_coeffs and abs(coeffs[i]) <= zero_tol:
        coeffs[i] = 0.0
        i += 1
    return num_coeffs, offset


@nb_jit(nopython=True, cache=True)
def _drop_window_head(
    knots: npt.NDArray[Any],
    coeffs: npt.NDArray[Any],
    num_coeffs: int,
    degree: int,
    offset: int,
) -> int:
    """Discard the first ``offset`` coefficients of the window, shifting the rest down.

    Args:
        knots (npt.NDArray[Any]): Knot window, shifted in place.
        coeffs (npt.NDArray[Any]): Coefficient window, shifted in place.
        num_coeffs (int): Number of valid entries in ``coeffs`` before the shift.
        degree (int): Polynomial degree.
        offset (int): Number of leading coefficients to drop.

    Returns:
        int: The coefficient count after the shift, ``num_coeffs - offset``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.bspline.find_roots` instead.
    """
    remaining = num_coeffs - offset
    for i in range(remaining):
        coeffs[i] = coeffs[i + offset]
    for i in range(remaining + degree + 1):
        knots[i] = knots[i + offset]
    return remaining


@nb_jit(nopython=True, cache=True)
def _track_zero(  # noqa: PLR0913
    knots: npt.NDArray[Any],
    coeffs: npt.NDArray[Any],
    num_coeffs: int,
    degree: int,
    index: int,
    tol: float,
    domain_length: float,
    max_insertions: int,
    iterates: npt.NDArray[Any],
    work: npt.NDArray[Any],
) -> tuple[float, int, float, int, int]:
    """Track one sign change of the control polygon to a zero of the spline.

    Runs Algorithm 2 of Mørken and Reimers on the window buffers: the zero of
    the control polygon is inserted as a new knot, and the tracked index follows
    the same sign change, which after the insertion sits at ``index`` or at
    ``index + 1`` and nowhere else. Both buffers are refined in place.

    The stopping rule is the paper's: the spread of the last ``degree`` iterates,
    measured relative to ``max(|knots[index]|, |knots[index+degree]|,
    domain_length)``. That window spans exactly one quadratic step, since the
    error is quadratic per ``degree - 1`` insertions and not per insertion.

    Args:
        knots (npt.NDArray[Any]): Knot window, refined in place.
        coeffs (npt.NDArray[Any]): Coefficient window, refined in place.
        num_coeffs (int): Number of valid entries in ``coeffs``.
        degree (int): Polynomial degree, at least 1.
        index (int): Zero index to track, inside the window. The caller
            guarantees that :func:`_is_zero_index` holds for it.
        tol (float): Relative stagnation tolerance.
        domain_length (float): Length of the spline's parametric domain, used as
            a floor for the tolerance scale.
        max_insertions (int): Insertion budget for this zero.
        iterates (npt.NDArray[Any]): Ring buffer of at least ``degree`` entries.
        work (npt.NDArray[Any]): Scratch buffer of at least ``degree + 1`` entries.

    Returns:
        tuple[float, int, float, int, int]: ``(x, status, residual, num_coeffs,
        index)``. ``x`` is the last polygon zero computed and ``residual`` is
        ``|f(x)|``, evaluated on the refined window on *every* exit: the status
        records how the iteration stopped, never whether ``x`` may be reported,
        which is the caller's residual test alone. ``status`` is one of
        ``_STATUS_STAGNATED``, ``_STATUS_VANISHED`` or ``_STATUS_BUDGET``, and
        the last two are the coefficient count and tracked index left behind in
        the refined window.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.bspline.find_roots` instead.
    """
    num_iterates = 0
    previous_x = 0.0
    x = 0.0

    # Never write past the working buffers: the budget is whichever runs out
    # first, the insertions allowed for this zero or the room left in the
    # arrays, keeping `degree + 1` slots for the split that follows a reported
    # zero.
    budget = min(max_insertions, coeffs.shape[0] - degree - 1 - num_coeffs)
    inserted = 0

    while True:
        left_value = coeffs[index - 1]
        right_value = coeffs[index]
        left_abscissa = _knot_average(knots, degree, index - 1)
        right_abscissa = _knot_average(knots, degree, index)

        # Zero of the segment joining (left_abscissa, left_value) and
        # (right_abscissa, right_value), as a clamped convex combination: the
        # two values straddle zero, so the parameter is in [0, 1] up to rounding.
        lam = -left_value / (right_value - left_value)
        lam = min(max(lam, 0.0), 1.0)
        x = left_abscissa + lam * (right_abscissa - left_abscissa)
        x = min(max(x, left_abscissa), right_abscissa)

        # A knot average is a sum of `degree` terms divided by `degree`, so it
        # carries a relative error of a few ulp and can land below its own
        # smallest term once the knots have piled up to within that much. The
        # theory places every iterate in `[t[index], t[index+degree]]`, and the
        # span search below only ever walks right from `index`, so restore the
        # left end rather than let the search return a span that does not
        # bracket x.
        x = max(x, knots[index])

        # A repeated iterate is a fixed point of the iteration, hence a zero of
        # the spline (Morken-Reimers, Lemma 13 and Corollary 14). The corollary
        # is a statement about exact arithmetic, so what it licenses is stopping
        # here, not the value being a zero: the residual is what says that.
        #
        # Its hypothesis is that the tracked sign change has `degree - 1` of its
        # active knots collapsed onto the iterate, `t[index + 1] = ... =
        # t[index + degree - 1] = x`, and the knots being ascending the two ends
        # of that run decide it. Repetition alone is a much weaker signal: it is
        # also what a secant that is nearly horizontal in the control polygon
        # produces, once the correction it asks for falls below an ulp of the
        # Greville abscissa, and that happens at a shallow non-zero minimum just
        # as readily as at a zero. Testing repetition alone stops the iteration
        # early, wherever precision runs out first, which at degree 3 is after
        # one collapsed knot of the two the corollary needs.
        #
        # Falling through instead inserts x again, which collapses one more knot
        # onto it, so the hypothesis is reached in at most `degree - 1` further
        # insertions and the budget bounds the rest. For degree 1 the run is
        # empty and the corollary asks for nothing.
        if num_iterates > 0 and x == previous_x:
            collapsed = degree < _COROLLARY_14_DEGREE or (
                knots[index + 1] == x and knots[index + degree - 1] == x
            )
            if collapsed:
                residual = _residual_at(knots, coeffs, num_coeffs, degree, x, work)
                return x, _STATUS_STAGNATED, residual, num_coeffs, index

        # Their Lemma 3: an iterate that reaches the right end of its Greville
        # interval carries `coeffs[index] = 0`, hence f(x) = 0.
        #
        # The lemma places x in `(knot_average(index - 1), knot_average(index)]`,
        # and that interval is empty exactly when the two abscissae coincide,
        # which happens exactly when `knots[index] == knots[index + degree]`,
        # that is when the knot run at x carries multiplicity `degree + 1` and
        # the spline is C^-1 there. In that one excluded case the secant through
        # the two coefficients is vertical, its zero is x for every `lam`, and
        # nothing forces `coeffs[index]` to vanish; the sign change is then the
        # spline jumping across the axis rather than meeting it.
        #
        # Either way the conclusion is tested rather than assumed, so the two
        # cases need no separating: reaching this branch means the knot run at x
        # has multiplicity at least `degree`, and `coeffs[index]` is then exactly
        # the value of the spline immediately to the right of the run, with no
        # positional error to inflate it. Refining instead would insert a knot
        # into a run the method keeps at multiplicity at most `degree`, and
        # divide by a zero-length span.
        if x >= knots[index + degree]:
            return x, _STATUS_STAGNATED, abs(coeffs[index]), num_coeffs, index

        iterates[num_iterates % degree] = x
        previous_x = x
        num_iterates += 1

        if num_iterates >= degree:
            lowest = iterates[0]
            highest = iterates[0]
            for i in range(1, degree):
                lowest = min(lowest, iterates[i])
                highest = max(highest, iterates[i])
            scale = max(abs(knots[index]), abs(knots[index + degree]))
            scale = max(scale, domain_length)
            if highest - lowest <= tol * scale:
                residual = _residual_at(knots, coeffs, num_coeffs, degree, x, work)
                return x, _STATUS_STAGNATED, residual, num_coeffs, index

        if inserted >= budget:
            residual = _residual_at(knots, coeffs, num_coeffs, degree, x, work)
            return x, _STATUS_BUDGET, residual, num_coeffs, index

        span = index
        while x >= knots[span + 1]:
            span += 1
        _insert_knot(knots, coeffs, num_coeffs, degree, x, span)
        num_coeffs += 1
        inserted += 1

        # Their Algorithm 2, step 3: the tracked sign change is now at `index`
        # or at `index + 1`; if it is at neither, it has vanished.
        if not _is_zero_index(coeffs, num_coeffs, index):
            index += 1
            if not _is_zero_index(coeffs, num_coeffs, index):
                residual = _residual_at(knots, coeffs, num_coeffs, degree, x, work)
                return x, _STATUS_VANISHED, residual, num_coeffs, index


@nb_jit(nopython=True, cache=True)
def _morken_reimers_roots(  # noqa: PLR0912, PLR0913, PLR0915
    knots: npt.NDArray[Any],
    degree: int,
    coeffs: npt.NDArray[Any],
    tol: float,
    zero_tol: float,
    root_tol: float,
    max_insertions: int,
) -> tuple[npt.NDArray[np.float64], int, int, int]:
    """Compute every zero of a scalar spline by the Mørken-Reimers method.

    Scans the control polygon left to right and tracks each sign change. What
    the polygon supplies is a *place to look*, never a certificate: the
    tracking stops for three different reasons, none of which implies that the
    point it stopped at is a zero, so every iterate is reported only when
    ``|f(x)| <= root_tol``. It is what separates a tangential zero, where the
    two sign changes bracketing it collapse, from a false warning of the
    variation-diminishing bound; what rejects a sign change across a knot of
    multiplicity ``degree + 1``, where the spline is C^-1 and jumps across the
    axis without reaching it; and what rejects an iteration that stopped moving
    somewhere other than a zero.

    The two domain endpoints are decided by ``zero_tol`` instead, on the
    coefficient an open knot vector interpolates there: that is the value of the
    spline, carrying no location error at all, so only the evaluation term
    applies.

    Args:
        knots (npt.NDArray[Any]): Open (clamped) knot vector of shape
            ``(num_coeffs + degree + 1,)``.
        degree (int): Polynomial degree, at least 1.
        coeffs (npt.NDArray[Any]): B-spline coefficients of the scalar spline.
        tol (float): Relative stagnation tolerance for the iterates.
        zero_tol (float): Absolute residual below which an exactly located value
            counts as zero, that is the error of evaluating the spline.
        root_tol (float): Absolute residual below which a *tracked* iterate
            counts as a zero. Larger than ``zero_tol`` by what the parametric
            tolerance the iteration stops at leaves behind, and derived in
            :func:`pantr.bspline._bspline_roots._find_roots_impl`.
        max_insertions (int): Insertion budget per zero.

    Returns:
        tuple[npt.NDArray[np.float64], int, int, int]: ``(roots, count,
        truncated, abandoned)`` where only the first ``count`` entries of
        ``roots`` are valid, sorted ascending. ``truncated`` counts the reported
        zeros whose insertion budget ran out first, so that they sit at the last
        iterate reached rather than at a stagnated one; ``abandoned`` counts the
        sign changes whose budget ran out without the residual ever certifying
        them, each of which may be a zero this call does not report.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.bspline.find_roots` instead.
    """
    num_coeffs = coeffs.shape[0]
    # A connected spline has at most `num_coeffs - 1` sign changes in its control
    # polygon and therefore at most that many interior zeros, plus the two
    # endpoints. The extra slot is what lets the loop below stop on a full array
    # instead of writing past it, which nothing would catch inside a kernel.
    roots = np.empty(num_coeffs + 2, dtype=np.float64)
    count = 0
    truncated = 0
    abandoned = 0

    start = float(knots[degree])
    end = float(knots[num_coeffs])
    domain_length = end - start

    # The working arrays hold the spline from the zero reported last to the end
    # of the domain, refined in place. They only ever grow by insertions between
    # two reported zeros, because reporting one compacts everything behind it.
    capacity = num_coeffs + degree + 2 + 4 * max_insertions
    live_coeffs = np.empty(capacity, dtype=np.float64)
    live_knots = np.empty(capacity + degree + 1, dtype=np.float64)
    iterates = np.empty(max(degree, 1), dtype=np.float64)
    work = np.empty(degree + 1, dtype=np.float64)

    for i in range(num_coeffs):
        live_coeffs[i] = coeffs[i]
    for i in range(num_coeffs + degree + 1):
        live_knots[i] = knots[i]
    num_live = num_coeffs

    # An open knot vector interpolates its first and last coefficient, so an
    # endpoint zero is read straight off the coefficients (their pseudo code).
    previous_root = start - domain_length - 1.0
    if abs(coeffs[0]) <= zero_tol:
        roots[count] = start
        count += 1
        previous_root = start

    scan_from = 1
    while count + 1 < roots.shape[0]:
        index = _zero_index(live_coeffs, num_live, scan_from)
        if index < 0:
            break

        # Stop while a full tracking and the split that may follow it still fit.
        # Reaching this means the insertions between two reported zeros outran
        # the room the compaction reclaims, which the capacity below is sized to
        # prevent; the count of unfinished zeros is what tells the caller. This
        # sign change is never tracked at all, so it is abandoned rather than
        # truncated.
        if num_live + max_insertions + 2 * (degree + 1) > capacity:
            abandoned += 1
            break

        x, status, residual, num_live, index = _track_zero(
            live_knots,
            live_coeffs,
            num_live,
            degree,
            index,
            tol,
            domain_length,
            max_insertions,
            iterates,
            work,
        )

        # The residual, and nothing else, is what makes an iterate a zero. The
        # three ways the tracking can stop are three provenances, not three
        # entitlements: the polygon lost its sign change, the iteration stopped
        # moving, or the budget ran out, and each of them can stop at a point
        # where the spline does not vanish. The theory that says otherwise holds
        # in exact arithmetic, where a coefficient reaches zero only near a zero
        # of the spline; in floating point it can reach zero for reasons of its
        # own, so the conclusion is tested rather than inherited.
        #
        # `root_tol` rather than `zero_tol`, because the iteration stops at a
        # *parametric* resolution: a zero located to that much leaves `|f'|`
        # times it behind however exactly f is then evaluated, and testing the
        # evaluation error alone would reject the genuine zeros of a steep
        # spline.
        certified = residual <= root_tol

        if status == _STATUS_BUDGET:
            if certified:
                truncated += 1
            else:
                abandoned += 1

        # The sweep is strictly left to right, so an iterate that did not pass
        # the zero reported last is that same zero seen again through another
        # sign change of the polygon, not a new one.
        if certified and previous_root < x <= end:
            roots[count] = x
            count += 1
            previous_root = x
            scan_from = index + 1
            if x < live_knots[num_live]:
                # Split at the zero just reported so that the next sign change
                # cannot drift back onto it, then drop everything left of the
                # split: it is behind the sweep for good, and dropping it is
                # what keeps the working arrays from growing with each zero and
                # what leaves an open knot vector on the rest of the domain.
                num_live, offset = _split_at_root(
                    live_knots, live_coeffs, num_live, degree, x, zero_tol
                )
                num_live = _drop_window_head(live_knots, live_coeffs, num_live, degree, offset)
                scan_from = 1
        else:
            # One index past the sign change just rejected, and no further: the
            # refinement left by the tracking can leave a second sign change of
            # its own immediately to the right, and that one may well bracket
            # the next zero. Stepping over the knot run instead would skip it.
            # This is what makes the sweep terminate, the index returned by
            # `_track_zero` being at least the one the scan handed it.
            scan_from = index + 1

    if abs(coeffs[num_coeffs - 1]) <= zero_tol:
        roots[count] = end
        count += 1

    return roots, count, truncated, abandoned


@nb_jit(nopython=True, cache=True)
def _merge_roots(
    roots: npt.NDArray[np.float64],
    radii: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], int]:
    """Merge runs of ascending roots that lie within their own merge radii.

    The scan reports one root per sign change of the control polygon, and a zero
    of even multiplicity is bracketed by two of them, so consecutive reports have
    to be collapsed. Two neighbours join the same run when their separation does
    not exceed the larger of the two radii; each run is replaced by its midpoint,
    which for a bracketed tangential zero is the better estimate of the two.

    Args:
        roots (npt.NDArray[np.float64]): Ascending root candidates.
        radii (npt.NDArray[np.float64]): Per-root merge radius, same length.

    Returns:
        tuple[npt.NDArray[np.float64], int]: ``(merged, n_merged)`` where only
        the first ``n_merged`` entries are valid.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`pantr.bspline.find_roots` instead.
    """
    count = roots.shape[0]
    merged = np.empty(max(count, 1), dtype=np.float64)
    if count == 0:
        return merged, 0

    n_merged = 0
    run_start = roots[0]
    run_end = roots[0]
    for i in range(1, count):
        if roots[i] - run_end <= max(radii[i], radii[i - 1]):
            run_end = roots[i]
        else:
            merged[n_merged] = 0.5 * (run_start + run_end)
            n_merged += 1
            run_start = roots[i]
            run_end = roots[i]
    merged[n_merged] = 0.5 * (run_start + run_end)
    n_merged += 1
    return merged, n_merged


def _warmup_numba_functions() -> None:
    """Precompile the root-finding kernels with float64 signatures.

    Compiles the whole call tree through :func:`_morken_reimers_roots` on a
    cubic spline with one interior knot and a single sign change.
    """
    knots = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    coeffs = np.array([-1.0, -0.5, 0.5, 1.0, 2.0], dtype=np.float64)
    roots, count, _, _ = _morken_reimers_roots(knots, 3, coeffs, 1e-15, 1e-14, 1e-13, 64)
    _merge_roots(roots[:count], np.full(count, 1e-15, dtype=np.float64))


__all__ = [
    "_deboor_point",
    "_drop_window_head",
    "_insert_knot",
    "_is_zero_index",
    "_knot_average",
    "_merge_roots",
    "_morken_reimers_roots",
    "_residual_at",
    "_span_at",
    "_split_at_root",
    "_track_zero",
    "_zero_index",
]
