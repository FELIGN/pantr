"""Numba-compiled kernels for Bézier evaluation and degree elevation.

Provides fused evaluation kernels that compute Bernstein basis values and
contract them with control points in a single pass, plus a simplified
single-element degree elevation kernel.

Note:
    Inputs are assumed to be correct (no validation performed).
    For general use, call the Layer 2 helpers in ``_bezier_eval`` and
    ``_bezier_degree`` instead.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._numba_compat import nb_jit, nb_prange
from .bspline._bspline_degree_core import _bincoeff


@nb_jit(
    nopython=True,
    cache=True,
    parallel=True,
)
def _evaluate_bezier_1d_core(
    ctrl: npt.NDArray[np.float32 | np.float64],
    pts: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate a 1D Bézier curve at the given points.

    Fuses Bernstein basis evaluation with control point contraction: for each
    evaluation point the ``(p+1)`` Bernstein basis values are computed via the
    recurrence relation and immediately multiplied with the corresponding
    control points, avoiding allocation of a full ``(n_pts, p+1)`` basis matrix.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(p+1, rank)``.
        pts (npt.NDArray[np.float32 | np.float64]): 1D evaluation points of
            shape ``(n_pts,)``.
        out (npt.NDArray[np.float32 | np.float64]): Pre-allocated output array
            of shape ``(n_pts, rank)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_bezier_eval._evaluate_bezier` instead.
    """
    p = ctrl.shape[0] - 1
    rank = ctrl.shape[1]
    n_pts = pts.shape[0]

    for pt_id in nb_prange(n_pts):
        u = pts[pt_id]

        if p == 0:
            for r in range(rank):
                out[pt_id, r] = ctrl[0, r]
        elif u == 1.0:
            for r in range(rank):
                out[pt_id, r] = ctrl[p, r]
        else:
            one_minus_u = 1.0 - u
            # B_0 = (1 - u)^p
            b_prev = np.power(one_minus_u, p)

            # Accumulate: start with basis[0] * ctrl[0]
            for r in range(rank):
                out[pt_id, r] = b_prev * ctrl[0, r]

            if p > 0:
                t_over_1mt = u / one_minus_u
                for i in range(1, p + 1):
                    b_curr = b_prev * ((p - i + 1.0) / i) * t_over_1mt
                    for r in range(rank):
                        out[pt_id, r] += b_curr * ctrl[i, r]
                    b_prev = b_curr


@nb_jit(
    nopython=True,
    cache=True,
    parallel=True,
)
def _evaluate_bezier_deriv_1d_core(  # noqa: PLR0912
    ctrl: npt.NDArray[np.float32 | np.float64],
    pts: npt.NDArray[np.float32 | np.float64],
    n_deriv: int,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate derivatives of a 1D Bézier curve at the given points.

    Fuses Bernstein derivative basis evaluation (Algorithm A2.3 from Piegl &
    Tiller specialised for Bernstein polynomials) with control point contraction.
    For each evaluation point, computes the ``ndu`` table and derivative
    coefficients, then contracts each derivative order with control points.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(p+1, rank)``.
        pts (npt.NDArray[np.float32 | np.float64]): 1D evaluation points of
            shape ``(n_pts,)``.
        n_deriv (int): Maximum derivative order to compute (>= 0).
        out (npt.NDArray[np.float32 | np.float64]): Pre-allocated output array
            of shape ``(n_pts, n_deriv+1, rank)``. ``out[i, k, :]`` is the
            k-th derivative of the Bézier at ``pts[i]``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_bezier_eval._evaluate_bezier_deriv` instead.
    """
    p = ctrl.shape[0] - 1
    rank = ctrl.shape[1]
    n_pts = pts.shape[0]
    order = p + 1
    dtype = pts.dtype
    zero = dtype.type(0.0)
    one = dtype.type(1.0)

    for pt_id in nb_prange(n_pts):
        s = pts[pt_id]

        # Zero out output for this point
        for k in range(n_deriv + 1):
            for r in range(rank):
                out[pt_id, k, r] = zero

        # --- Build ndu table for uniform Bernstein knots ---
        ndu = np.zeros((order, order), dtype=dtype)
        a_arr = np.zeros((2, n_deriv + 1), dtype=dtype)

        ndu[0, 0] = one
        for j in range(1, order):
            saved = zero
            for rr in range(j):
                ndu[j, rr] = one  # knot difference = 1
                temp = ndu[rr, j - 1]
                ndu[rr, j] = saved + (one - s) * temp
                saved = s * temp
            ndu[j, j] = saved

        # 0th derivative: contract basis values with control points
        for j in range(order):
            basis_val = ndu[j, p]
            for r in range(rank):
                out[pt_id, 0, r] += basis_val * ctrl[j, r]

        # k-th derivatives via triangular recursion (A2.3, unit knot spans)
        # First compute basis derivatives, then contract
        basis_derivs = np.zeros((n_deriv + 1, order), dtype=dtype)
        for j in range(order):
            basis_derivs[0, j] = ndu[j, p]

        for rr in range(order):
            s1 = 0
            s2 = 1
            a_arr[0, 0] = one

            for k in range(1, n_deriv + 1):
                d = zero
                rk = rr - k
                pk = p - k

                if rr >= k:
                    a_arr[s2, 0] = a_arr[s1, 0]
                    d = a_arr[s2, 0] * ndu[rk, pk]

                j1 = 1 if rk >= -1 else -rk
                j2 = k - 1 if (rr - 1) <= pk else p - rr

                for jj in range(j1, j2 + 1):
                    a_arr[s2, jj] = a_arr[s1, jj] - a_arr[s1, jj - 1]
                    d += a_arr[s2, jj] * ndu[rk + jj, pk]

                if rr <= pk:
                    a_arr[s2, k] = -a_arr[s1, k - 1]
                    d += a_arr[s2, k] * ndu[rr, pk]

                basis_derivs[k, rr] = d

                tmp_s = s1
                s1 = s2
                s2 = tmp_s

        # Factorial scaling and contraction
        fac = p
        for k in range(1, n_deriv + 1):
            for j in range(order):
                scaled = basis_derivs[k, j] * fac
                for r in range(rank):
                    out[pt_id, k, r] += scaled * ctrl[j, r]
            fac *= p - k


@nb_jit(nopython=True, cache=True)
def _degree_elevate_bezier_1d_core(
    degree: int,
    ctrl: npt.NDArray[np.float32 | np.float64],
    degree_increment: int,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Degree-elevate a single Bézier segment.

    Computes the ``bezalfs`` matrix of Bézier degree elevation coefficients
    and applies it to the control points:

    .. math::

        Q_i = \sum_{j=\max(0,i-t)}^{\min(p,i)}
              \frac{\binom{p}{j}\,\binom{t}{i-j}}{\binom{p+t}{i}}\, P_j

    where ``p`` is the original degree and ``t`` the increment.

    Args:
        degree (int): Original polynomial degree (``p >= 0``).
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(p+1, rank)``.
        degree_increment (int): Number of degrees to add (``t >= 1``).

    Returns:
        npt.NDArray[np.float32 | np.float64]: Elevated control points of shape
        ``(p+t+1, rank)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_bezier_degree._degree_elevate_bezier` instead.
    """
    p = degree
    t = degree_increment
    ph = p + t
    rank = ctrl.shape[1]
    ph2 = ph // 2

    # Compute bezalfs coefficients
    bezalfs = np.zeros((p + 1, ph + 1), dtype=np.float64)
    bezalfs[0, 0] = 1.0
    bezalfs[p, ph] = 1.0

    for i in range(1, ph2 + 1):
        inv = 1.0 / _bincoeff(ph, i)
        mpi = min(p, i)
        for j in range(max(0, i - t), mpi + 1):
            bezalfs[j, i] = inv * _bincoeff(p, j) * _bincoeff(t, i - j)

    # Symmetry
    for i in range(ph2 + 1, ph):
        mpi = min(p, i)
        for j in range(max(0, i - t), mpi + 1):
            bezalfs[j, i] = bezalfs[p - j, ph - i]

    # Apply elevation: new_ctrl[i] = sum_j bezalfs[j, i] * ctrl[j]
    new_ctrl = np.zeros((ph + 1, rank), dtype=ctrl.dtype)
    for i in range(ph + 1):
        mpi = min(p, i)
        for j in range(max(0, i - t), mpi + 1):
            coeff = bezalfs[j, i]
            for r in range(rank):
                new_ctrl[i, r] += coeff * ctrl[j, r]

    return new_ctrl
