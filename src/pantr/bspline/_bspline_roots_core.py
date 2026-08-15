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

Note:
    Inputs are assumed to be correct (no validation performed).
    For general use, call :func:`pantr.bspline.find_roots` instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit

_STATUS_CONVERGED: int = 1
"""The iterates stagnated, or repeated exactly, at a zero of the spline."""

_STATUS_CANDIDATE: int = 2
"""The polygon cannot certify the last iterate; its residual decides.

Reached when the tracked sign change vanished under refinement, which is either a
tangential zero or a false warning of the variation-diminishing bound, and when the
iterate reached a knot of multiplicity ``degree + 1``, where the spline jumps and the
sign change need not bracket a zero at all.
"""

_STATUS_BUDGET: int = 3
"""The insertion budget ran out before the iterates stagnated."""


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
def _track_zero(  # noqa: PLR0913
    knots: npt.NDArray[Any],
    coeffs: npt.NDArray[Any],
    num_coeffs: int,
    degree: int,
    index: int,
    tol: float,
    zero_tol: float,
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
        zero_tol (float): Absolute residual below which a coefficient counts as
            zero.
        domain_length (float): Length of the spline's parametric domain, used as
            a floor for the tolerance scale.
        max_insertions (int): Insertion budget for this zero.
        iterates (npt.NDArray[Any]): Ring buffer of at least ``degree`` entries.
        work (npt.NDArray[Any]): Scratch buffer of at least ``degree + 1`` entries.

    Returns:
        tuple[float, int, float, int, int]: ``(x, status, residual, num_coeffs,
        index)``. ``status`` is one of ``_STATUS_CONVERGED``,
        ``_STATUS_CANDIDATE`` or ``_STATUS_BUDGET``, ``x`` is the last polygon
        zero computed, ``residual`` is ``|f(x)|`` when the status is
        ``_STATUS_CANDIDATE`` and ``0.0`` otherwise, and the last two are the
        coefficient count and tracked index left behind in the refined window.

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
        # the spline (Morken-Reimers, Lemma 13 and Corollary 14).
        if num_iterates > 0 and x == previous_x:
            return x, _STATUS_CONVERGED, 0.0, num_coeffs, index

        # Their Lemma 3: an iterate that reaches the right end of its Greville
        # interval carries `coeffs[index] = 0`, hence f(x) = 0.
        #
        # The lemma places x in `(knot_average(index - 1), knot_average(index)]`,
        # and that interval is empty exactly when the two abscissae coincide,
        # which happens exactly when `knots[index] == knots[index + degree]`,
        # that is when the knot run at x carries multiplicity `degree + 1` and
        # the spline is C^-1 there. The hypothesis is therefore a property of the
        # knot vector alone: the test below is structural and needs no tolerance,
        # and it leaves every other spline on the path it already took.
        #
        # In that one excluded case the secant through the two coefficients is
        # vertical, its zero is x for every `lam`, and nothing forces
        # `coeffs[index]` to vanish; the sign change is then the spline jumping
        # across the axis rather than meeting it. So test the lemma's conclusion
        # instead of assuming it. At an exact C^-1 knot `coeffs[index]` is the
        # value of the spline immediately to the right of the run, with no
        # positional error to inflate it, so comparing it against the residual
        # threshold is dimensionally sound.
        if x >= knots[index + degree]:
            if knots[index] < knots[index + degree] or abs(coeffs[index]) <= zero_tol:
                return x, _STATUS_CONVERGED, 0.0, num_coeffs, index
            # Hand the jump to the caller's residual test, which rejects it.
            # Refining instead would insert a knot into a run the method keeps at
            # multiplicity at most `degree`, and divide by a zero-length span.
            return x, _STATUS_CANDIDATE, abs(coeffs[index]), num_coeffs, index

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
                return x, _STATUS_CONVERGED, 0.0, num_coeffs, index

        if inserted >= budget:
            return x, _STATUS_BUDGET, 0.0, num_coeffs, index

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
                span = 0
                while x >= knots[span + 1]:
                    span += 1
                residual = _deboor_point(knots, coeffs, degree, x, span, work)
                return x, _STATUS_CANDIDATE, abs(residual), num_coeffs, index


@nb_jit(nopython=True, cache=True)
def _morken_reimers_roots(  # noqa: PLR0912, PLR0913
    knots: npt.NDArray[Any],
    degree: int,
    coeffs: npt.NDArray[Any],
    tol: float,
    zero_tol: float,
    max_insertions: int,
) -> tuple[npt.NDArray[np.float64], int, int]:
    """Compute every zero of a scalar spline by the Mørken-Reimers method.

    Scans the control polygon left to right and tracks each sign change to a
    zero of the spline. A sign change that disappears under refinement is a
    false warning of the variation-diminishing bound, unless the spline is
    tangent to the axis there: those are separated by testing the residual
    ``|f(x)| <= zero_tol``, which is also how the two domain endpoints are
    tested. Zeros of even multiplicity have no sign change in the limit and
    cannot be certified by the polygon alone, so they are reported through that
    residual test only. A sign change across a knot of multiplicity
    ``degree + 1``, where the spline is C^-1 and jumps across the axis without
    reaching it, is rejected by that same test.

    Args:
        knots (npt.NDArray[Any]): Open (clamped) knot vector of shape
            ``(num_coeffs + degree + 1,)``.
        degree (int): Polynomial degree, at least 1.
        coeffs (npt.NDArray[Any]): B-spline coefficients of the scalar spline.
        tol (float): Relative stagnation tolerance for the iterates.
        zero_tol (float): Absolute residual below which a value counts as zero.
        max_insertions (int): Insertion budget per zero.

    Returns:
        tuple[npt.NDArray[np.float64], int, int]: ``(roots, count, truncated)``
        where only the first ``count`` entries of ``roots`` are valid, sorted
        ascending, and ``truncated`` counts the zeros whose insertion budget ran
        out before the iterates stagnated.

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
        # prevent; the count of unfinished zeros is what tells the caller.
        if num_live + max_insertions + 2 * (degree + 1) > capacity:
            truncated += 1
            break

        x, status, residual, num_live, index = _track_zero(
            live_knots,
            live_coeffs,
            num_live,
            degree,
            index,
            tol,
            zero_tol,
            domain_length,
            max_insertions,
            iterates,
            work,
        )

        if status == _STATUS_BUDGET:
            truncated += 1

        # A polygon that lost its sign change is at a zero of even multiplicity if
        # the spline vanishes there, and at a false warning of the
        # variation-diminishing bound otherwise. A sign change straddling a knot
        # of multiplicity `degree + 1` arrives here as well, and the same test
        # rejects it: the spline jumps across the axis without ever reaching it.
        accepted = residual <= zero_tol if status == _STATUS_CANDIDATE else True

        # The sweep is strictly left to right, so an iterate that did not pass
        # the zero reported last is that same zero seen again through another
        # sign change of the polygon, not a new one.
        if accepted and previous_root < x <= end:
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
                for i in range(num_live - offset):
                    live_coeffs[i] = live_coeffs[i + offset]
                for i in range(num_live - offset + degree + 1):
                    live_knots[i] = live_knots[i + offset]
                num_live -= offset
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

    return roots, count, truncated


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
    roots, count, _ = _morken_reimers_roots(knots, 3, coeffs, 1e-15, 1e-14, 64)
    _merge_roots(roots[:count], np.full(count, 1e-15, dtype=np.float64))


__all__ = [
    "_deboor_point",
    "_insert_knot",
    "_is_zero_index",
    "_knot_average",
    "_merge_roots",
    "_morken_reimers_roots",
    "_split_at_root",
    "_track_zero",
    "_zero_index",
]
