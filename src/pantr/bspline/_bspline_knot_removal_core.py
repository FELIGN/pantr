"""Numba kernel for B-spline knot removal (Algorithm A5.8, Piegl & Tiller).

Implements single-knot removal with tolerance-based acceptance: given a knot
value, its index, and current multiplicity, attempts up to ``num`` removals,
accepting each only when the geometric deviation stays within the specified
tolerance.

Note:
    Inputs are assumed to be correct (no validation performed).
    For general use, call the Layer 2 helpers in ``_bspline_knot_removal`` instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit


@nb_jit(nopython=True, cache=True)
def _remove_knot_1d_core(  # noqa: PLR0912, PLR0913, PLR0915
    degree: int,
    knots: npt.NDArray[Any],
    ctrl: npt.NDArray[Any],
    knot_value: float,
    knot_index: int,
    multiplicity: int,
    num: int,
    tol: float,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], int]:
    """Remove a single knot value up to *num* times from a 1D B-spline.

    Implements Algorithm A5.8 from *The NURBS Book* (Piegl & Tiller, 1997).
    Each removal is accepted only if the resulting geometric deviation does not
    exceed *tol*.

    Args:
        degree (int): Polynomial degree.
        knots (npt.NDArray[Any]): Knot vector of shape ``(n + degree + 2,)``.
        ctrl (npt.NDArray[Any]): Control points of shape ``(n + 1, rank)``.
        knot_value (float): The knot value to remove.
        knot_index (int): Index *r* such that ``knots[r] == knot_value`` and
            ``knots[r] != knots[r + 1]`` (i.e. the last occurrence).
        multiplicity (int): Current multiplicity *s* of *knot_value*.
        num (int): Maximum number of removals to attempt.
        tol (float): Maximum allowed Euclidean distance between the two
            intermediate control-point approximations.

    Returns:
        tuple[npt.NDArray[Any], npt.NDArray[Any], int]: ``(new_knots, new_ctrl,
        removals)`` where *removals* is the number of knots actually removed
        (``0 <= removals <= num``).

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call ``_remove_knot_bspline_1d_impl`` instead.
    """
    n = ctrl.shape[0] - 1  # last control point index
    rank = ctrl.shape[1]
    m = n + degree + 1  # last knot index
    order = degree + 1
    r = knot_index
    s = multiplicity

    # Work on copies so the originals are not modified.
    work_knots = knots.copy()
    work_ctrl = ctrl.copy()

    # Temporary control point buffer (maximum size needed: 2*degree+1 rows).
    temp = np.empty((2 * degree + 1, rank), dtype=ctrl.dtype)

    fout = (2 * r - s - degree) // 2  # first output row that changes
    first = r - degree
    last = r - s

    removals = 0
    for t in range(num):
        off = first - 1
        # Seed the temp buffer.
        for d in range(rank):
            temp[0, d] = work_ctrl[off, d]
            temp[last + 1 - off, d] = work_ctrl[last + 1, d]

        # Compute new control points from both sides.
        i = first
        ii = 1
        j = last
        jj = last - off

        while j - i > t:
            denom_i = work_knots[i + order + t] - work_knots[i]
            alpha_i = (knot_value - work_knots[i]) / denom_i if denom_i != 0.0 else 0.0

            denom_j = work_knots[j + order] - work_knots[j - t]
            alpha_j = (knot_value - work_knots[j - t]) / denom_j if denom_j != 0.0 else 0.0

            for d in range(rank):
                temp[ii, d] = (work_ctrl[i, d] - (1.0 - alpha_i) * temp[ii - 1, d]) / alpha_i
                temp[jj, d] = (work_ctrl[j, d] - alpha_j * temp[jj + 1, d]) / (1.0 - alpha_j)

            i += 1
            ii += 1
            j -= 1
            jj -= 1

        # Check deviation.
        accepted = False
        if j - i < t:
            # Odd case: check distance between the two meeting points.
            dist = 0.0
            for d in range(rank):
                diff = temp[ii - 1, d] - temp[jj + 1, d]
                dist += diff * diff
            dist = np.sqrt(dist)
            if dist <= tol:
                accepted = True
        else:
            # Even case: check blended point against both-side approximation.
            denom_i = work_knots[i + order + t] - work_knots[i]
            alpha_i = (knot_value - work_knots[i]) / denom_i if denom_i != 0.0 else 0.0
            dist = 0.0
            for d in range(rank):
                blended = alpha_i * temp[ii + t + 1, d] + (1.0 - alpha_i) * temp[ii - 1, d]
                diff = work_ctrl[i, d] - blended
                dist += diff * diff
            dist = np.sqrt(dist)
            if dist <= tol:
                accepted = True

        if not accepted:
            break

        # Accept the removal: write temp back into work_ctrl.
        i = first
        j = last
        while j - i > t:
            for d in range(rank):
                work_ctrl[i, d] = temp[i - off, d]
                work_ctrl[j, d] = temp[j - off, d]
            i += 1
            j -= 1

        first -= 1
        last += 1
        removals += 1

    if removals == 0:
        return knots.copy(), ctrl.copy(), 0

    # Compact the knot vector: shift entries after the removed knots.
    for k in range(r + 1, m + 1):
        work_knots[k - removals] = work_knots[k]

    new_n_knots = knots.shape[0] - removals
    new_knots = work_knots[:new_n_knots].copy()

    # Compact the control points: shift entries to fill removed rows.
    # Determine the gap boundaries after the alternating shift.
    j_out = fout
    i_out = j_out
    for k in range(1, removals):
        if k % 2 == 1:
            i_out += 1
        else:
            j_out -= 1

    # In-place shift on work_ctrl: copy entries from i_out+1..n into j_out..
    dest = j_out
    for k in range(i_out + 1, n + 1):
        for d in range(rank):
            work_ctrl[dest, d] = work_ctrl[k, d]
        dest += 1

    new_n_ctrl = n + 1 - removals
    new_ctrl = work_ctrl[:new_n_ctrl].copy()

    return new_knots, new_ctrl, removals


def _warmup_numba_functions() -> None:
    """Precompile numba functions with float64 signatures for faster first call.

    This function triggers compilation of the numba-decorated functions
    with float64 arrays, ensuring they are cached and ready for use.
    """
    knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
    ctrl = np.array([[0.0, 0.0], [0.25, 1.0], [0.5, 0.5], [1.0, 0.0]], dtype=np.float64)
    _remove_knot_1d_core(2, knots, ctrl, 0.5, 3, 1, 1, 1e-10)


__all__ = [
    "_remove_knot_1d_core",
]
