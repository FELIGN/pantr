"""Numba-compiled kernels for Bézier evaluation, degree elevation, and degree reduction.

Provides fused evaluation kernels that compute Bernstein basis values and
contract them with control points in a single pass, plus degree elevation
and degree reduction kernels.

Note:
    Inputs are assumed to be correct (no validation performed).
    For general use, call the Layer 2 helpers in ``_bezier_eval`` and
    ``_bezier_degree`` instead.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit, nb_prange
from ..bspline._bspline_degree_core import _bincoeff


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


@nb_jit(nopython=True, cache=True)
def _degree_reduce_bezier_1d_core(  # noqa: PLR0912
    degree: int,
    ctrl: npt.NDArray[np.float32 | np.float64],
    degree_decrement: int,
) -> npt.NDArray[np.float32 | np.float64]:
    r"""Degree-reduce a single Bézier segment via least-squares approximation.

    The degree elevation matrix from degree ``P-1`` to ``P`` is a
    ``(P+1) \times P`` lower bidiagonal matrix with diagonal
    ``a_k = 1 - k/P`` and sub-diagonal ``b_k = (k+1)/P``.  Degree reduction
    solves the overdetermined system ``M x = c`` in the least-squares sense
    using QR factorisation with Givens rotations (complexity ``O(P)`` per
    rank component).

    When ``degree_decrement > 1``, the single-step reduction is applied
    iteratively.

    Args:
        degree (int): Current polynomial degree (``p >= 1``).
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of shape
            ``(p+1, rank)``.
        degree_decrement (int): Number of degrees to remove (``1 <= t <= p``).

    Returns:
        npt.NDArray[np.float32 | np.float64]: Reduced control points of shape
        ``(p - t + 1, rank)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_bezier_degree._degree_reduce_bezier`
        instead.
    """
    rank = ctrl.shape[1]
    cur = np.empty((ctrl.shape[0], rank), dtype=ctrl.dtype)
    for i in range(ctrl.shape[0]):
        for r in range(rank):
            cur[i, r] = ctrl[i, r]

    cur_deg = degree

    for _step in range(degree_decrement):
        p = cur_deg
        n_new = p  # reduced degree = p - 1, so p control points

        nxt = np.empty((n_new, rank), dtype=ctrl.dtype)

        # Build diagonal and sub-diagonal of the (P+1) x P elevation matrix.
        # diagonal:     a[k] = 1 - k/P   for k = 0 .. P-1
        # sub-diagonal: b[k] = (k+1)/P   for k = 0 .. P-1
        #
        # Givens QR: zero each sub-diagonal entry, producing an upper
        # bidiagonal R and transforming the RHS.  Then back-substitute.
        diag = np.empty(p, dtype=np.float64)
        sup = np.empty(p, dtype=np.float64)  # super-diagonal created by Givens
        rhs = np.empty(p + 1, dtype=np.float64)

        for r in range(rank):
            # Initialise diagonal entries and RHS for this rank component.
            for k in range(p):
                diag[k] = 1.0 - np.float64(k) / np.float64(p)
            for k in range(p):
                sup[k] = 0.0
            for k in range(p + 1):
                rhs[k] = np.float64(cur[k, r])

            # Forward Givens pass
            for k in range(p):
                bk = np.float64(k + 1) / np.float64(p)  # sub-diagonal value
                ak = diag[k]

                # Givens rotation to zero bk
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

                diag[k] = cs * ak + sn * bk  # = hypot(ak, bk)

                # Rotation creates a super-diagonal entry at (k, k+1)
                if k + 1 < p:
                    sup[k] = sn * diag[k + 1]
                    diag[k + 1] = cs * diag[k + 1]

                # Rotate RHS rows k and k+1
                rk = rhs[k]
                rhs[k] = cs * rk + sn * rhs[k + 1]
                rhs[k + 1] = -sn * rk + cs * rhs[k + 1]

            # Back-substitution on upper bidiagonal R
            nxt[p - 1, r] = ctrl.dtype.type(rhs[p - 1] / diag[p - 1])
            for k in range(p - 2, -1, -1):
                nxt[k, r] = ctrl.dtype.type((rhs[k] - sup[k] * np.float64(nxt[k + 1, r])) / diag[k])

        cur = nxt
        cur_deg = p - 1

    return cur


@nb_jit(
    nopython=True,
    cache=True,
    parallel=True,
)
def _slice_bezier_1d_core(
    ctrl: npt.NDArray[np.float32 | np.float64],
    value: float,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Evaluate a 1D Bézier at a single parameter value via de Casteljau.

    Performs the de Casteljau triangular reduction on each column of the
    control point array independently, parallelized across columns with
    ``prange``.  At the boundary values ``0`` and ``1``, the first or
    last control point is returned directly without iteration.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of
            shape ``(p + 1, n_cols)``, where ``p`` is the polynomial
            degree and ``n_cols`` is the number of independent columns.
        value (float): Parameter value in ``[0, 1]``.
        out (npt.NDArray[np.float32 | np.float64]): Pre-allocated output
            array of shape ``(n_cols,)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_slice_bezier` in ``_bezier_slice``
        instead.
    """
    p = ctrl.shape[0] - 1
    n_cols = ctrl.shape[1]
    u = value
    one_minus_u = 1.0 - u

    # Boundary shortcuts: O(1) for endpoints.
    if u == 0.0:
        for col in nb_prange(n_cols):
            out[col] = ctrl[0, col]
        return
    if u == 1.0:
        for col in nb_prange(n_cols):
            out[col] = ctrl[p, col]
        return

    for col in nb_prange(n_cols):
        # Local workspace for de Casteljau on this column.
        d = np.empty(p + 1, dtype=ctrl.dtype)
        for i in range(p + 1):
            d[i] = ctrl[i, col]

        for r in range(1, p + 1):
            for i in range(p - r + 1):
                d[i] = one_minus_u * d[i] + u * d[i + 1]

        out[col] = d[0]


@nb_jit(
    nopython=True,
    cache=True,
    parallel=True,
)
def _split_bezier_1d_core(
    ctrl: npt.NDArray[np.float32 | np.float64],
    value: float,
    out_left: npt.NDArray[np.float32 | np.float64],
    out_right: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Split a 1D Bézier at a parameter value via de Casteljau.

    Performs a single forward de Casteljau pass that simultaneously produces
    the control points for both the left ``[0, value]`` and right
    ``[value, 1]`` halves (reparametrized to ``[0, 1]``).

    After level ``r`` of the forward iteration, the workspace naturally
    contains ``[d_0^r, d_1^{r-1}, ..., d_p^0]``.  The left-half control
    points are ``d[0]`` at each level; the right-half control points are
    the final workspace state.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of
            shape ``(p + 1, n_cols)``, where ``p`` is the polynomial
            degree and ``n_cols`` is the number of independent columns.
        value (float): Parameter value in ``(0, 1)`` at which to split.
        out_left (npt.NDArray[np.float32 | np.float64]): Pre-allocated
            output array of shape ``(p + 1, n_cols)`` for the left half.
        out_right (npt.NDArray[np.float32 | np.float64]): Pre-allocated
            output array of shape ``(p + 1, n_cols)`` for the right half.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_split_bezier` in ``_bezier_split``
        instead.
    """
    p = ctrl.shape[0] - 1
    n_cols = ctrl.shape[1]
    u = value
    one_minus_u = 1.0 - u

    for col in nb_prange(n_cols):
        # Local workspace for de Casteljau on this column.
        d = np.empty(p + 1, dtype=ctrl.dtype)
        for i in range(p + 1):
            d[i] = ctrl[i, col]

        # First left control point is the original first control point.
        out_left[0, col] = d[0]

        for r in range(1, p + 1):
            for i in range(p - r + 1):
                d[i] = one_minus_u * d[i] + u * d[i + 1]
            # After level r, d[0] = d_0^r → left-half control point.
            out_left[r, col] = d[0]

        # After all iterations, d = [d_0^p, d_1^{p-1}, ..., d_p^0] = right half.
        for i in range(p + 1):
            out_right[i, col] = d[i]


@nb_jit(
    nopython=True,
    cache=True,
    parallel=True,
)
def _restrict_bezier_1d_core(  # noqa: PLR0912
    ctrl: npt.NDArray[np.float32 | np.float64],
    lower: float,
    upper: float,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    r"""Restrict a 1D Bézier to a sub-interval via two de Casteljau passes.

    Computes the Bernstein coefficients of the polynomial restricted to
    ``[lower, upper]`` and reparametrized to ``[0, 1]``.  Uses the
    numerically stable two-pass strategy: the order of the left/right
    passes is chosen to avoid dividing by a small number.

    - If ``|upper| >= |lower - 1|``: left pass at ``upper``, then right
      pass at ``lower / upper``.
    - Otherwise: right pass at ``lower``, then left pass at
      ``(upper - lower) / (1 - lower)``.

    Args:
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points of
            shape ``(p + 1, n_cols)``.
        lower (float): Left bound of the sub-interval in ``[0, 1)``.
        upper (float): Right bound of the sub-interval in ``(0, 1]``.
        out (npt.NDArray[np.float32 | np.float64]): Pre-allocated output
            array of shape ``(p + 1, n_cols)`` for the restricted
            coefficients.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_restrict_bezier` in
        ``_bezier_restrict`` instead.
    """
    p = ctrl.shape[0] - 1
    n_cols = ctrl.shape[1]

    for col in nb_prange(n_cols):
        d = np.empty(p + 1, dtype=ctrl.dtype)
        for i in range(p + 1):
            d[i] = ctrl[i, col]

        if abs(upper) >= abs(lower - 1.0):
            # deCasteljauLeft(upper), then deCasteljauRight(lower / upper)
            tau = upper
            for step in range(1, p + 1):
                for j in range(p, step - 1, -1):
                    d[j] = d[j] * tau + d[j - 1] * (1.0 - tau)

            tau2 = lower / upper
            for step in range(1, p + 1):
                for j in range(p - step + 1):
                    d[j] = d[j] * (1.0 - tau2) + d[j + 1] * tau2
        else:
            # deCasteljauRight(lower), then deCasteljauLeft((upper-lower)/(1-lower))
            tau = lower
            for step in range(1, p + 1):
                for j in range(p - step + 1):
                    d[j] = d[j] * (1.0 - tau) + d[j + 1] * tau

            tau2 = (upper - lower) / (1.0 - lower)
            for step in range(1, p + 1):
                for j in range(p, step - 1, -1):
                    d[j] = d[j] * tau2 + d[j - 1] * (1.0 - tau2)

        for i in range(p + 1):
            out[i, col] = d[i]


def _warmup_numba_functions() -> None:
    """Precompile numba functions with float64 signatures for faster first call.

    This function triggers compilation of the numba-decorated functions
    with float64 arrays, ensuring they are cached and ready for use.
    """
    ctrl_dummy = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]], dtype=np.float64)
    pts_dummy = np.array([0.5], dtype=np.float64)
    out_eval_dummy = np.empty((1, 2), dtype=np.float64)
    out_deriv_dummy = np.empty((1, 1, 2), dtype=np.float64)
    out_slice_dummy = np.empty(2, dtype=np.float64)

    out_split_dummy = np.empty((3, 2), dtype=np.float64)

    _evaluate_bezier_1d_core(ctrl_dummy, pts_dummy, out_eval_dummy)
    _evaluate_bezier_deriv_1d_core(ctrl_dummy, pts_dummy, 0, out_deriv_dummy)
    _degree_elevate_bezier_1d_core(2, ctrl_dummy, 1)
    _degree_reduce_bezier_1d_core(2, ctrl_dummy, 1)
    _slice_bezier_1d_core(ctrl_dummy, 0.5, out_slice_dummy)
    _split_bezier_1d_core(ctrl_dummy, 0.5, out_split_dummy, out_split_dummy.copy())
    _restrict_bezier_1d_core(ctrl_dummy, 0.2, 0.8, out_split_dummy)
