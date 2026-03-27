"""Numba-compiled kernels for mask operations on Bernstein polynomial subcell grids.

Provides point/line query kernels, nonzero mask construction via recursive
subdivision with uniform sign detection, intersection mask construction via
orthant tests, and scalar de Casteljau restriction helpers.

Masks are N-dimensional boolean arrays of shape ``(M,)*N`` that divide the
reference cube ``[0,1]^N`` into a regular grid of ``M^N`` subcells.  A True
entry indicates the associated polynomial may have zeros in that subcell; False
guarantees the polynomial is nonzero there.

This is a translation of the masking logic from the algoim library
(R. I. Saye, *J. Comput. Phys.* 448, 110720, 2022).

Note:
    Inputs are assumed to be correct (no validation performed).
    For general use, call the Layer 2 helpers in ``_mask`` instead.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit
from ._bezier_core import _uniform_sign_core

# ---------------------------------------------------------------------------
# Point / line query kernels
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _point_within_mask_core(
    mask_flat: npt.NDArray[np.bool_],
    x: npt.NDArray[np.float64],
    M: int,
    N: int,
) -> bool:
    """Test if point ``x`` falls in a True subcell of a flattened mask.

    Discretizes each coordinate to a cell index via ``floor(x[d] * M)``,
    clamped to ``[0, M-1]``, then furls the N-D index to a flat offset.

    Args:
        mask_flat (npt.NDArray[np.bool_]): Flattened mask of length ``M^N``.
        x (npt.NDArray[np.float64]): Point in ``[0,1]^N``, shape ``(N,)``.
        M (int): Grid resolution per axis.
        N (int): Number of parametric dimensions.

    Returns:
        bool: True if the subcell containing ``x`` is marked True.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_mask._point_within_mask` instead.
    """
    idx = 0
    for d in range(N):
        cell = int(math.floor(x[d] * M))  # noqa: RUF046
        if cell < 0:
            cell = 0
        elif cell >= M:
            cell = M - 1
        idx = idx * M + cell
    return bool(mask_flat[idx])


@nb_jit(nopython=True, cache=True)
def _line_intersects_mask_core(
    mask_flat: npt.NDArray[np.bool_],
    x: npt.NDArray[np.float64],
    k: int,
    M: int,
    N: int,
) -> bool:
    """Test if line ``{x + alpha e_k : alpha in [0,1]}`` hits a True subcell.

    The point ``x`` has ``N-1`` components (axis ``k`` excluded).  For each
    of the ``M`` cells along axis ``k``, the full N-D cell index is assembled
    and checked against the mask.

    Args:
        mask_flat (npt.NDArray[np.bool_]): Flattened mask of length ``M^N``.
        x (npt.NDArray[np.float64]): Point in ``[0,1]^{N-1}``, shape
            ``(N-1,)``.  Coordinate ``d < k`` maps to axis ``d``; coordinate
            ``d >= k`` maps to axis ``d + 1``.
        k (int): Axis along which to scan.
        M (int): Grid resolution per axis.
        N (int): Number of parametric dimensions of the *full* mask.

    Returns:
        bool: True if any subcell along the line is marked True.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_mask._line_intersects_mask` instead.
    """
    if N == 1:
        for i in range(M):  # noqa: SIM110
            if mask_flat[i]:
                return True
        return False

    # Precompute cell indices for the N-1 fixed dimensions.
    cells = np.empty(N, dtype=np.int64)
    for d in range(N):
        if d < k:
            cell = int(math.floor(x[d] * M))  # noqa: RUF046
        elif d > k:
            cell = int(math.floor(x[d - 1] * M))  # noqa: RUF046
        else:
            cell = 0  # placeholder for the scanning axis
        if cell < 0:
            cell = 0
        elif cell >= M:
            cell = M - 1
        cells[d] = cell

    # Scan along axis k.
    for i in range(M):
        cells[k] = i
        idx = 0
        for d in range(N):
            idx = idx * M + cells[d]
        if mask_flat[idx]:
            return True
    return False


# ---------------------------------------------------------------------------
# Scalar de Casteljau restriction (1D, no column loop)
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _restrict_scalar_1d(
    coeffs: npt.NDArray[np.float64],
    lower: float,
    upper: float,
    out: npt.NDArray[np.float64],
) -> None:
    """Restrict 1D scalar Bernstein coefficients to ``[lower, upper]``.

    Two-pass de Casteljau with the numerically stable ordering from
    ``_restrict_bezier_1d_core``, specialized for scalar coefficients
    (no column loop).

    Args:
        coeffs (npt.NDArray[np.float64]): Input coefficients, shape ``(p+1,)``.
        lower (float): Left bound of the sub-interval.
        upper (float): Right bound of the sub-interval.
        out (npt.NDArray[np.float64]): Output coefficients, shape ``(p+1,)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_mask._nonzero_mask` instead.
    """
    p = coeffs.shape[0] - 1
    for i in range(p + 1):
        out[i] = coeffs[i]

    if abs(upper) >= abs(lower - 1.0):
        tau = upper
        for _step in range(1, p + 1):
            for j in range(p, _step - 1, -1):
                out[j] = out[j] * tau + out[j - 1] * (1.0 - tau)
        tau2 = lower / upper if upper != 0.0 else 0.0
        for _step in range(1, p + 1):
            for j in range(p - _step + 1):
                out[j] = out[j] * (1.0 - tau2) + out[j + 1] * tau2
    else:
        tau = lower
        for _step in range(1, p + 1):
            for j in range(p - _step + 1):
                out[j] = out[j] * (1.0 - tau) + out[j + 1] * tau
        denom = 1.0 - lower
        tau2 = (upper - lower) / denom if denom != 0.0 else 0.0
        for _step in range(1, p + 1):
            for j in range(p, _step - 1, -1):
                out[j] = out[j] * tau2 + out[j - 1] * (1.0 - tau2)


# ---------------------------------------------------------------------------
# Orthant test kernel
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _orthant_test_base_core(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    sign: int,
) -> bool:
    """Check if there exists alpha such that ``sign*x[i] + alpha*y[i] > 0`` for all i.

    This is the core feasibility test from algoim's ``orthantTestBase``.

    Args:
        x (npt.NDArray[np.float64]): First coefficient array (flat).
        y (npt.NDArray[np.float64]): Second coefficient array (flat), same length.
        sign (int): +1 or -1.

    Returns:
        bool: True if the linear combination is feasible.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_mask._intersection_mask` instead.
    """
    alpha_max = np.inf
    alpha_min = -np.inf
    n = x.shape[0]
    for i in range(n):
        xi = x[i] * sign
        yi = y[i]
        if yi == 0.0 and xi <= 0.0:
            return False
        if yi > 0.0:
            alpha_min = max(alpha_min, -xi / yi)
        elif yi < 0.0:
            alpha_max = min(alpha_max, -xi / yi)

    if np.isinf(alpha_min) or np.isinf(alpha_max):
        return True
    if alpha_max - alpha_min > 1.0e5 * np.finfo(np.float64).eps * max(  # noqa: SIM103
        abs(alpha_min), abs(alpha_max)
    ):
        return True
    return False


@nb_jit(nopython=True, cache=True)
def _orthant_test_core(
    f: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
) -> bool:
    """Test if polynomials f and g provably do not share a common zero.

    Checks if there exist scalars alpha, beta such that
    ``alpha*f[i] + beta*g[i] > 0`` for every Bernstein coefficient.
    Both arrays must be flat and have the same length (degree-elevate
    beforehand if needed).

    Args:
        f (npt.NDArray[np.float64]): Flat Bernstein coefficients of f.
        g (npt.NDArray[np.float64]): Flat Bernstein coefficients of g.

    Returns:
        bool: True if f and g provably share no common zero.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_mask._intersection_mask` instead.
    """
    return _orthant_test_base_core(f, g, -1) or _orthant_test_base_core(f, g, 1)


# ---------------------------------------------------------------------------
# Scalar Bernstein degree elevation (1D, for orthant test pre-processing)
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _elevate_scalar_1d(
    coeffs: npt.NDArray[np.float64],
    target_len: int,
    out: npt.NDArray[np.float64],
) -> None:
    """Degree-elevate 1D scalar Bernstein coefficients from degree p to degree q.

    Uses the single-step formula for elevation by 1, applied iteratively.

    Args:
        coeffs (npt.NDArray[np.float64]): Input coefficients, shape ``(p+1,)``.
        target_len (int): Target length ``q+1`` where ``q >= p``.
        out (npt.NDArray[np.float64]): Output array of shape ``(target_len,)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    p = coeffs.shape[0] - 1

    # Copy input to a workspace.
    cur = np.empty(target_len, dtype=np.float64)
    for i in range(p + 1):
        cur[i] = coeffs[i]

    cur_deg = p
    while cur_deg < target_len - 1:
        n = cur_deg
        # Elevate by 1: new[k] = k/(n+1) * old[k-1] + (1 - k/(n+1)) * old[k]
        prev_val = cur[0]
        cur_last = cur[n]
        for kk in range(n, 0, -1):
            ratio = float(kk) / float(n + 1)
            cur[kk] = ratio * cur[kk - 1] + (1.0 - ratio) * cur[kk]
        cur[0] = prev_val  # k=0: ratio=0, so cur[0] stays
        cur[n + 1] = cur_last  # endpoint
        cur_deg += 1

    for i in range(target_len):
        out[i] = cur[i]


# ---------------------------------------------------------------------------
# Nonzero mask: 1D
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _nonzero_mask_1d_core(
    coeffs: npt.NDArray[np.float64],
    fmask: npt.NDArray[np.bool_],
    out: npt.NDArray[np.bool_],
    M: int,
) -> None:
    """Compute conservative nonzero mask for a 1D scalar Bernstein polynomial.

    Uses recursive bisection with de Casteljau restriction and uniform sign
    detection.  Matches the ``mask_driver`` algorithm from algoim.

    Args:
        coeffs (npt.NDArray[np.float64]): Bernstein coefficients, shape ``(p+1,)``.
        fmask (npt.NDArray[np.bool_]): Input mask, shape ``(M,)``.
        out (npt.NDArray[np.bool_]): Output mask, shape ``(M,)``, initialized to
            False by the caller.
        M (int): Grid resolution.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_mask._nonzero_mask` instead.
    """
    p = coeffs.shape[0] - 1
    eps = 0.015625 / M  # padding = 1 / (64 * M)
    work = np.empty(p + 1, dtype=np.float64)

    _nz_mask_1d_recurse(coeffs, fmask, out, M, eps, work, 0, M)


@nb_jit(nopython=True, cache=True)
def _nz_mask_1d_recurse(  # noqa: PLR0913
    coeffs: npt.NDArray[np.float64],
    fmask: npt.NDArray[np.bool_],
    out: npt.NDArray[np.bool_],
    M: int,
    eps: float,
    work: npt.NDArray[np.float64],
    a: int,
    b: int,
) -> None:
    """Recursive helper for 1D nonzero mask construction.

    Args:
        coeffs (npt.NDArray[np.float64]): Bernstein coefficients, shape ``(p+1,)``.
        fmask (npt.NDArray[np.bool_]): Input mask, shape ``(M,)``.
        out (npt.NDArray[np.bool_]): Output mask, shape ``(M,)``.
        M (int): Grid resolution.
        eps (float): Padding for subcell restriction.
        work (npt.NDArray[np.float64]): Workspace, shape ``(p+1,)``.
        a (int): Start of current range (inclusive).
        b (int): End of current range (exclusive).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    # Check overlap with input mask.
    overlap = False
    for i in range(a, b):
        if fmask[i]:
            overlap = True
            break
    if not overlap:
        return

    # Restrict polynomial to padded subcell and check uniform sign.
    xa = float(a) / M - eps
    xb = float(b) / M + eps
    _restrict_scalar_1d(coeffs, xa, xb, work)
    if _uniform_sign_core(work) != 0:
        return

    # Base case: single subcell.
    if b - a == 1:
        out[a] = True
        return

    # Recurse on two halves.
    mid = (a + b) // 2
    _nz_mask_1d_recurse(coeffs, fmask, out, M, eps, work, a, mid)
    _nz_mask_1d_recurse(coeffs, fmask, out, M, eps, work, mid, b)


# ---------------------------------------------------------------------------
# Nonzero mask: 2D
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _nonzero_mask_2d_core(
    coeffs: npt.NDArray[np.float64],
    fmask: npt.NDArray[np.bool_],
    out: npt.NDArray[np.bool_],
    M: int,
) -> None:
    """Compute conservative nonzero mask for a 2D scalar Bernstein polynomial.

    Args:
        coeffs (npt.NDArray[np.float64]): Bernstein coefficients, shape
            ``(p0+1, p1+1)``.
        fmask (npt.NDArray[np.bool_]): Input mask, shape ``(M, M)``.
        out (npt.NDArray[np.bool_]): Output mask, shape ``(M, M)``, initialized
            to False by the caller.
        M (int): Grid resolution.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_mask._nonzero_mask` instead.
    """
    n0 = coeffs.shape[0]
    n1 = coeffs.shape[1]
    eps = 0.015625 / M  # padding = 1 / (64 * M)
    # Workspace for restriction.
    work0 = np.empty((n0, n1), dtype=np.float64)
    work1 = np.empty((n0, n1), dtype=np.float64)
    flat_work = np.empty(n0 * n1, dtype=np.float64)
    col_work = np.empty(n0, dtype=np.float64)
    col_out_work = np.empty(n0, dtype=np.float64)
    row_work = np.empty(n1, dtype=np.float64)

    _nz_mask_2d_recurse(
        coeffs,
        fmask,
        out,
        M,
        eps,
        work0,
        work1,
        flat_work,
        col_work,
        col_out_work,
        row_work,
        0,
        M,
        0,
        M,
    )


@nb_jit(nopython=True, cache=True)
def _restrict_scalar_2d(  # noqa: PLR0913
    coeffs: npt.NDArray[np.float64],
    xa0: float,
    xb0: float,
    xa1: float,
    xb1: float,
    tmp: npt.NDArray[np.float64],
    out: npt.NDArray[np.float64],
    col_work: npt.NDArray[np.float64],
    col_out_work: npt.NDArray[np.float64],
    row_work: npt.NDArray[np.float64],
) -> None:
    """Restrict 2D scalar Bernstein coefficients to a sub-rectangle.

    Applies de Casteljau restriction along axis 0, then axis 1.

    Args:
        coeffs (npt.NDArray[np.float64]): Input, shape ``(n0, n1)``.
        xa0 (float): Lower bound for axis 0.
        xb0 (float): Upper bound for axis 0.
        xa1 (float): Lower bound for axis 1.
        xb1 (float): Upper bound for axis 1.
        tmp (npt.NDArray[np.float64]): Workspace, shape ``(n0, n1)``.
        out (npt.NDArray[np.float64]): Output, shape ``(n0, n1)``.
        col_work (npt.NDArray[np.float64]): Workspace, shape ``(n0,)``.
        col_out_work (npt.NDArray[np.float64]): Workspace, shape ``(n0,)``.
        row_work (npt.NDArray[np.float64]): Workspace, shape ``(n1,)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n0 = coeffs.shape[0]
    n1 = coeffs.shape[1]

    # Restrict along axis 0: for each column j, restrict coeffs[:, j].
    for j in range(n1):
        for i in range(n0):
            col_work[i] = coeffs[i, j]
        _restrict_scalar_1d(col_work, xa0, xb0, col_out_work)
        for i in range(n0):
            tmp[i, j] = col_out_work[i]

    # Restrict along axis 1: for each row i, restrict tmp[i, :].
    for i in range(n0):
        _restrict_scalar_1d(tmp[i, :], xa1, xb1, row_work)
        for j in range(n1):
            out[i, j] = row_work[j]


@nb_jit(nopython=True, cache=True)
def _nz_mask_2d_recurse(  # noqa: PLR0913
    coeffs: npt.NDArray[np.float64],
    fmask: npt.NDArray[np.bool_],
    out: npt.NDArray[np.bool_],
    M: int,
    eps: float,
    work0: npt.NDArray[np.float64],
    work1: npt.NDArray[np.float64],
    flat_work: npt.NDArray[np.float64],
    col_work: npt.NDArray[np.float64],
    col_out_work: npt.NDArray[np.float64],
    row_work: npt.NDArray[np.float64],
    a0: int,
    b0: int,
    a1: int,
    b1: int,
) -> None:
    """Recursive helper for 2D nonzero mask construction.

    Args:
        coeffs (npt.NDArray[np.float64]): Bernstein coefficients, shape
            ``(n0, n1)``.
        fmask (npt.NDArray[np.bool_]): Input mask, shape ``(M, M)``.
        out (npt.NDArray[np.bool_]): Output mask, shape ``(M, M)``.
        M (int): Grid resolution.
        eps (float): Padding for subcell restriction.
        work0 (npt.NDArray[np.float64]): Workspace, shape ``(n0, n1)``.
        work1 (npt.NDArray[np.float64]): Workspace, shape ``(n0, n1)``.
        flat_work (npt.NDArray[np.float64]): Workspace, shape ``(n0*n1,)``.
        col_work (npt.NDArray[np.float64]): Workspace, shape ``(n0,)``.
        col_out_work (npt.NDArray[np.float64]): Workspace, shape ``(n0,)``.
        row_work (npt.NDArray[np.float64]): Workspace, shape ``(n1,)``.
        a0 (int): Start of range along axis 0 (inclusive).
        b0 (int): End of range along axis 0 (exclusive).
        a1 (int): Start of range along axis 1 (inclusive).
        b1 (int): End of range along axis 1 (exclusive).

    Note:
        Inputs are assumed to be correct (no validation performed).
        ``work0``, ``work1``, ``flat_work``, ``col_work``, ``col_out_work``,
        and ``row_work`` are mutated and shared across all recursive calls;
        the parent finishes consuming each workspace before recursing.
    """
    # Check overlap with input mask.
    overlap = False
    for i0 in range(a0, b0):
        for i1 in range(a1, b1):
            if fmask[i0, i1]:
                overlap = True
                break
        if overlap:
            break
    if not overlap:
        return

    # Restrict polynomial to padded subcell and check uniform sign.
    xa0 = float(a0) / M - eps
    xb0 = float(b0) / M + eps
    xa1 = float(a1) / M - eps
    xb1 = float(b1) / M + eps
    _restrict_scalar_2d(coeffs, xa0, xb0, xa1, xb1, work0, work1, col_work, col_out_work, row_work)

    n0 = coeffs.shape[0]
    n1 = coeffs.shape[1]
    for i0 in range(n0):
        for i1 in range(n1):
            flat_work[i0 * n1 + i1] = work1[i0, i1]

    if _uniform_sign_core(flat_work) != 0:
        return

    # Base case: single subcell.
    if b0 - a0 == 1 and b1 - a1 == 1:
        out[a0, a1] = True
        return

    # Recurse on 2x2 children.
    mid0 = (a0 + b0) // 2
    mid1 = (a1 + b1) // 2
    for s0 in range(2):
        lo0 = a0 if s0 == 0 else mid0
        hi0 = mid0 if s0 == 0 else b0
        for s1 in range(2):
            lo1 = a1 if s1 == 0 else mid1
            hi1 = mid1 if s1 == 0 else b1
            _nz_mask_2d_recurse(
                coeffs,
                fmask,
                out,
                M,
                eps,
                work0,
                work1,
                flat_work,
                col_work,
                col_out_work,
                row_work,
                lo0,
                hi0,
                lo1,
                hi1,
            )


# ---------------------------------------------------------------------------
# Nonzero mask: 3D
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _restrict_scalar_3d(  # noqa: PLR0913
    coeffs: npt.NDArray[np.float64],
    xa0: float,
    xb0: float,
    xa1: float,
    xb1: float,
    xa2: float,
    xb2: float,
    tmp1: npt.NDArray[np.float64],
    tmp2: npt.NDArray[np.float64],
    out: npt.NDArray[np.float64],
    work1d: npt.NDArray[np.float64],
    work1d_b: npt.NDArray[np.float64],
) -> None:
    """Restrict 3D scalar Bernstein coefficients to a sub-box.

    Applies de Casteljau restriction along axis 0, then axis 1, then axis 2.

    Args:
        coeffs (npt.NDArray[np.float64]): Input, shape ``(n0, n1, n2)``.
        xa0 (float): Lower bound for axis 0.
        xb0 (float): Upper bound for axis 0.
        xa1 (float): Lower bound for axis 1.
        xb1 (float): Upper bound for axis 1.
        xa2 (float): Lower bound for axis 2.
        xb2 (float): Upper bound for axis 2.
        tmp1 (npt.NDArray[np.float64]): Workspace, shape ``(n0, n1, n2)``.
        tmp2 (npt.NDArray[np.float64]): Workspace, shape ``(n0, n1, n2)``.
        out (npt.NDArray[np.float64]): Output, shape ``(n0, n1, n2)``.
        work1d (npt.NDArray[np.float64]): 1D workspace, shape
            ``(max(n0, n1, n2),)``.
        work1d_b (npt.NDArray[np.float64]): Workspace, shape ``(max(n0, n1, n2),)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n0 = coeffs.shape[0]
    n1 = coeffs.shape[1]
    n2 = coeffs.shape[2]

    # Restrict along axis 0.
    for j in range(n1):
        for kk in range(n2):
            for i in range(n0):
                work1d[i] = coeffs[i, j, kk]
            _restrict_scalar_1d(work1d[:n0], xa0, xb0, work1d_b[:n0])
            for i in range(n0):
                tmp1[i, j, kk] = work1d_b[i]

    # Restrict along axis 1.
    for i in range(n0):
        for kk in range(n2):
            for j in range(n1):
                work1d[j] = tmp1[i, j, kk]
            _restrict_scalar_1d(work1d[:n1], xa1, xb1, work1d_b[:n1])
            for j in range(n1):
                tmp2[i, j, kk] = work1d_b[j]

    # Restrict along axis 2.
    for i in range(n0):
        for j in range(n1):
            for kk in range(n2):
                work1d[kk] = tmp2[i, j, kk]
            _restrict_scalar_1d(work1d[:n2], xa2, xb2, work1d_b[:n2])
            for kk in range(n2):
                out[i, j, kk] = work1d_b[kk]


@nb_jit(nopython=True, cache=True)
def _nonzero_mask_3d_core(
    coeffs: npt.NDArray[np.float64],
    fmask: npt.NDArray[np.bool_],
    out: npt.NDArray[np.bool_],
    M: int,
) -> None:
    """Compute conservative nonzero mask for a 3D scalar Bernstein polynomial.

    Args:
        coeffs (npt.NDArray[np.float64]): Bernstein coefficients, shape
            ``(n0, n1, n2)``.
        fmask (npt.NDArray[np.bool_]): Input mask, shape ``(M, M, M)``.
        out (npt.NDArray[np.bool_]): Output mask, shape ``(M, M, M)``,
            initialized to False by the caller.
        M (int): Grid resolution.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_mask._nonzero_mask` instead.
    """
    n0 = coeffs.shape[0]
    n1 = coeffs.shape[1]
    n2 = coeffs.shape[2]
    eps = 0.015625 / M  # padding = 1 / (64 * M)
    tmp1 = np.empty((n0, n1, n2), dtype=np.float64)
    tmp2 = np.empty((n0, n1, n2), dtype=np.float64)
    res = np.empty((n0, n1, n2), dtype=np.float64)
    flat_work = np.empty(n0 * n1 * n2, dtype=np.float64)
    nmax = max(n0, max(n1, n2))  # noqa: PLW3301
    work1d = np.empty(nmax, dtype=np.float64)
    work1d_b = np.empty(nmax, dtype=np.float64)

    _nz_mask_3d_recurse(
        coeffs,
        fmask,
        out,
        M,
        eps,
        tmp1,
        tmp2,
        res,
        flat_work,
        work1d,
        work1d_b,
        0,
        M,
        0,
        M,
        0,
        M,
    )


@nb_jit(nopython=True, cache=True)
def _nz_mask_3d_recurse(  # noqa: PLR0912, PLR0913
    coeffs: npt.NDArray[np.float64],
    fmask: npt.NDArray[np.bool_],
    out: npt.NDArray[np.bool_],
    M: int,
    eps: float,
    tmp1: npt.NDArray[np.float64],
    tmp2: npt.NDArray[np.float64],
    res: npt.NDArray[np.float64],
    flat_work: npt.NDArray[np.float64],
    work1d: npt.NDArray[np.float64],
    work1d_b: npt.NDArray[np.float64],
    a0: int,
    b0: int,
    a1: int,
    b1: int,
    a2: int,
    b2: int,
) -> None:
    """Recursive helper for 3D nonzero mask construction.

    Args:
        coeffs (npt.NDArray[np.float64]): Bernstein coefficients.
        fmask (npt.NDArray[np.bool_]): Input mask.
        out (npt.NDArray[np.bool_]): Output mask.
        M (int): Grid resolution.
        eps (float): Padding for subcell restriction.
        tmp1 (npt.NDArray[np.float64]): Workspace.
        tmp2 (npt.NDArray[np.float64]): Workspace.
        res (npt.NDArray[np.float64]): Workspace for restricted coefficients.
        flat_work (npt.NDArray[np.float64]): Flat workspace for sign test.
        work1d (npt.NDArray[np.float64]): 1D workspace.
        work1d_b (npt.NDArray[np.float64]): 1D workspace.
        a0 (int): Start of range along axis 0.
        b0 (int): End of range along axis 0.
        a1 (int): Start of range along axis 1.
        b1 (int): End of range along axis 1.
        a2 (int): Start of range along axis 2.
        b2 (int): End of range along axis 2.

    Note:
        Inputs are assumed to be correct (no validation performed).
        ``tmp1``, ``tmp2``, ``res``, ``flat_work``, ``work1d``, and
        ``work1d_b`` are mutated and shared across all recursive calls;
        the parent finishes consuming each workspace before recursing.
    """
    # Check overlap with input mask.
    overlap = False
    for i0 in range(a0, b0):
        for i1 in range(a1, b1):
            for i2 in range(a2, b2):
                if fmask[i0, i1, i2]:
                    overlap = True
                    break
            if overlap:
                break
        if overlap:
            break
    if not overlap:
        return

    # Restrict and check uniform sign.
    xa0 = float(a0) / M - eps
    xb0 = float(b0) / M + eps
    xa1 = float(a1) / M - eps
    xb1 = float(b1) / M + eps
    xa2 = float(a2) / M - eps
    xb2 = float(b2) / M + eps
    _restrict_scalar_3d(coeffs, xa0, xb0, xa1, xb1, xa2, xb2, tmp1, tmp2, res, work1d, work1d_b)

    n0 = coeffs.shape[0]
    n1 = coeffs.shape[1]
    n2 = coeffs.shape[2]
    idx = 0
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                flat_work[idx] = res[i0, i1, i2]
                idx += 1

    if _uniform_sign_core(flat_work) != 0:
        return

    # Base case.
    if b0 - a0 == 1 and b1 - a1 == 1 and b2 - a2 == 1:
        out[a0, a1, a2] = True
        return

    # Recurse on 2x2x2 children.
    mid0 = (a0 + b0) // 2
    mid1 = (a1 + b1) // 2
    mid2 = (a2 + b2) // 2
    for s0 in range(2):
        lo0 = a0 if s0 == 0 else mid0
        hi0 = mid0 if s0 == 0 else b0
        for s1 in range(2):
            lo1 = a1 if s1 == 0 else mid1
            hi1 = mid1 if s1 == 0 else b1
            for s2 in range(2):
                lo2 = a2 if s2 == 0 else mid2
                hi2 = mid2 if s2 == 0 else b2
                _nz_mask_3d_recurse(
                    coeffs,
                    fmask,
                    out,
                    M,
                    eps,
                    tmp1,
                    tmp2,
                    res,
                    flat_work,
                    work1d,
                    work1d_b,
                    lo0,
                    hi0,
                    lo1,
                    hi1,
                    lo2,
                    hi2,
                )


# ---------------------------------------------------------------------------
# Intersection mask: 2D
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _intersection_mask_2d_core(  # noqa: PLR0913
    coeffs_f: npt.NDArray[np.float64],
    fmask: npt.NDArray[np.bool_],
    coeffs_g: npt.NDArray[np.float64],
    gmask: npt.NDArray[np.bool_],
    out: npt.NDArray[np.bool_],
    M: int,
) -> None:
    """Compute intersection mask for two 2D scalar Bernstein polynomials.

    Finds subcells where f and g may share a common zero, using recursive
    subdivision with the orthant test.

    Args:
        coeffs_f (npt.NDArray[np.float64]): Coefficients of f, shape
            ``(nf0, nf1)``.
        fmask (npt.NDArray[np.bool_]): Nonzero mask of f, shape ``(M, M)``.
        coeffs_g (npt.NDArray[np.float64]): Coefficients of g, shape
            ``(ng0, ng1)``.
        gmask (npt.NDArray[np.bool_]): Nonzero mask of g, shape ``(M, M)``.
        out (npt.NDArray[np.bool_]): Output intersection mask, shape ``(M, M)``,
            initialized to False.
        M (int): Grid resolution.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_mask._intersection_mask` instead.
    """
    nf0, nf1 = coeffs_f.shape[0], coeffs_f.shape[1]
    ng0, ng1 = coeffs_g.shape[0], coeffs_g.shape[1]
    eps = 0.015625 / M  # padding = 1 / (64 * M)

    # Workspaces for f.
    f_tmp = np.empty((nf0, nf1), dtype=np.float64)
    f_res = np.empty((nf0, nf1), dtype=np.float64)
    f_row = np.empty(nf1, dtype=np.float64)

    # Workspaces for g.
    g_tmp = np.empty((ng0, ng1), dtype=np.float64)
    g_res = np.empty((ng0, ng1), dtype=np.float64)
    g_row = np.empty(ng1, dtype=np.float64)

    # Workspaces for orthant test (flat, degree-elevated to common size).
    max_len = max(nf0, ng0) * max(nf1, ng1)
    f_flat = np.empty(max_len, dtype=np.float64)
    g_flat = np.empty(max_len, dtype=np.float64)

    # 1D elevation workspaces.
    max_n = max(max(nf0, ng0), max(nf1, ng1))  # noqa: PLW3301
    elev_work = np.empty(max_n, dtype=np.float64)

    # Shared column workspace for f and g restriction (calls are sequential).
    max_col = max(nf0, ng0)
    col_work = np.empty(max_col, dtype=np.float64)
    col_out_work = np.empty(max_col, dtype=np.float64)

    _int_mask_2d_recurse(
        coeffs_f,
        fmask,
        coeffs_g,
        gmask,
        out,
        M,
        eps,
        f_tmp,
        f_res,
        col_work,
        col_out_work,
        f_row,
        g_tmp,
        g_res,
        g_row,
        f_flat,
        g_flat,
        elev_work,
        0,
        M,
        0,
        M,
    )


@nb_jit(nopython=True, cache=True)
def _elevate_2d_to_common(  # noqa: PLR0912
    f: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    f_flat: npt.NDArray[np.float64],
    g_flat: npt.NDArray[np.float64],
    elev_work: npt.NDArray[np.float64],
) -> int:
    """Degree-elevate two 2D Bernstein coefficient arrays to common extent and flatten.

    Args:
        f (npt.NDArray[np.float64]): Coefficients of f, shape ``(nf0, nf1)``.
        g (npt.NDArray[np.float64]): Coefficients of g, shape ``(ng0, ng1)``.
        f_flat (npt.NDArray[np.float64]): Output flat elevated f.
        g_flat (npt.NDArray[np.float64]): Output flat elevated g.
        elev_work (npt.NDArray[np.float64]): 1D workspace.

    Returns:
        int: Length of the flat arrays (common_n0 * common_n1).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    nf0, nf1 = f.shape[0], f.shape[1]
    ng0, ng1 = g.shape[0], g.shape[1]
    cn0 = max(nf0, ng0)
    cn1 = max(nf1, ng1)
    total = cn0 * cn1

    # Elevate f: first along axis 0, then axis 1.
    # Intermediate: shape (cn0, nf1).
    f_mid = np.empty((cn0, nf1), dtype=np.float64)
    if nf0 == cn0:
        for i in range(nf0):
            for j in range(nf1):
                f_mid[i, j] = f[i, j]
    else:
        col = np.empty(nf0, dtype=np.float64)
        for j in range(nf1):
            for i in range(nf0):
                col[i] = f[i, j]
            _elevate_scalar_1d(col, cn0, elev_work)
            for i in range(cn0):
                f_mid[i, j] = elev_work[i]

    # Then along axis 1: shape (cn0, cn1).
    if nf1 == cn1:
        idx = 0
        for i in range(cn0):
            for j in range(cn1):
                f_flat[idx] = f_mid[i, j]
                idx += 1
    else:
        row = np.empty(nf1, dtype=np.float64)
        idx = 0
        for i in range(cn0):
            for j in range(nf1):
                row[j] = f_mid[i, j]
            _elevate_scalar_1d(row, cn1, elev_work)
            for j in range(cn1):
                f_flat[idx] = elev_work[j]
                idx += 1

    # Elevate g: first along axis 0, then axis 1.
    g_mid = np.empty((cn0, ng1), dtype=np.float64)
    if ng0 == cn0:
        for i in range(ng0):
            for j in range(ng1):
                g_mid[i, j] = g[i, j]
    else:
        col = np.empty(ng0, dtype=np.float64)
        for j in range(ng1):
            for i in range(ng0):
                col[i] = g[i, j]
            _elevate_scalar_1d(col, cn0, elev_work)
            for i in range(cn0):
                g_mid[i, j] = elev_work[i]

    if ng1 == cn1:
        idx = 0
        for i in range(cn0):
            for j in range(cn1):
                g_flat[idx] = g_mid[i, j]
                idx += 1
    else:
        row = np.empty(ng1, dtype=np.float64)
        idx = 0
        for i in range(cn0):
            for j in range(ng1):
                row[j] = g_mid[i, j]
            _elevate_scalar_1d(row, cn1, elev_work)
            for j in range(cn1):
                g_flat[idx] = elev_work[j]
                idx += 1

    return total


@nb_jit(nopython=True, cache=True)
def _int_mask_2d_recurse(  # noqa: PLR0913
    coeffs_f: npt.NDArray[np.float64],
    fmask: npt.NDArray[np.bool_],
    coeffs_g: npt.NDArray[np.float64],
    gmask: npt.NDArray[np.bool_],
    out: npt.NDArray[np.bool_],
    M: int,
    eps: float,
    f_tmp: npt.NDArray[np.float64],
    f_res: npt.NDArray[np.float64],
    col_work: npt.NDArray[np.float64],
    col_out_work: npt.NDArray[np.float64],
    f_row: npt.NDArray[np.float64],
    g_tmp: npt.NDArray[np.float64],
    g_res: npt.NDArray[np.float64],
    g_row: npt.NDArray[np.float64],
    f_flat: npt.NDArray[np.float64],
    g_flat: npt.NDArray[np.float64],
    elev_work: npt.NDArray[np.float64],
    a0: int,
    b0: int,
    a1: int,
    b1: int,
) -> None:
    """Recursive helper for 2D intersection mask construction.

    Args:
        coeffs_f (npt.NDArray[np.float64]): Coefficients of f.
        fmask (npt.NDArray[np.bool_]): Nonzero mask of f.
        coeffs_g (npt.NDArray[np.float64]): Coefficients of g.
        gmask (npt.NDArray[np.bool_]): Nonzero mask of g.
        out (npt.NDArray[np.bool_]): Output intersection mask.
        M (int): Grid resolution.
        eps (float): Padding.
        f_tmp (npt.NDArray[np.float64]): Workspace for f restriction.
        f_res (npt.NDArray[np.float64]): Workspace for f restriction result.
        col_work (npt.NDArray[np.float64]): Shared column workspace for f and g.
        col_out_work (npt.NDArray[np.float64]): Shared column output workspace for f and g.
        f_row (npt.NDArray[np.float64]): Row workspace for f.
        g_tmp (npt.NDArray[np.float64]): Workspace for g restriction.
        g_res (npt.NDArray[np.float64]): Workspace for g restriction result.
        g_row (npt.NDArray[np.float64]): Row workspace for g.
        f_flat (npt.NDArray[np.float64]): Flat workspace for elevated f.
        g_flat (npt.NDArray[np.float64]): Flat workspace for elevated g.
        elev_work (npt.NDArray[np.float64]): 1D elevation workspace.
        a0 (int): Start of range along axis 0.
        b0 (int): End of range along axis 0.
        a1 (int): Start of range along axis 1.
        b1 (int): End of range along axis 1.

    Note:
        Inputs are assumed to be correct (no validation performed).
        ``f_tmp``, ``f_res``, ``col_work``, ``col_out_work``, ``f_row``,
        ``g_tmp``, ``g_res``, ``g_row``, ``f_flat``, ``g_flat``, and
        ``elev_work`` are mutated and shared across all recursive calls;
        the parent finishes consuming each workspace before recursing.
        ``col_work`` and ``col_out_work`` are also shared between the f
        and g restriction calls within the same frame; both calls are
        sequential and non-overlapping.
    """
    # Check overlap with both input masks.
    overlap = False
    for i0 in range(a0, b0):
        for i1 in range(a1, b1):
            if fmask[i0, i1] and gmask[i0, i1]:
                overlap = True
                break
        if overlap:
            break
    if not overlap:
        return

    # Restrict both polynomials.
    xa0 = float(a0) / M - eps
    xb0 = float(b0) / M + eps
    xa1 = float(a1) / M - eps
    xb1 = float(b1) / M + eps

    _restrict_scalar_2d(coeffs_f, xa0, xb0, xa1, xb1, f_tmp, f_res, col_work, col_out_work, f_row)
    _restrict_scalar_2d(coeffs_g, xa0, xb0, xa1, xb1, g_tmp, g_res, col_work, col_out_work, g_row)

    # Degree-elevate to common extent and run orthant test.
    total = _elevate_2d_to_common(f_res, g_res, f_flat, g_flat, elev_work)
    if _orthant_test_core(f_flat[:total], g_flat[:total]):
        return

    # Base case.
    if b0 - a0 == 1 and b1 - a1 == 1:
        out[a0, a1] = True
        return

    # Recurse on 2x2 children.
    mid0 = (a0 + b0) // 2
    mid1 = (a1 + b1) // 2
    for s0 in range(2):
        lo0 = a0 if s0 == 0 else mid0
        hi0 = mid0 if s0 == 0 else b0
        for s1 in range(2):
            lo1 = a1 if s1 == 0 else mid1
            hi1 = mid1 if s1 == 0 else b1
            _int_mask_2d_recurse(
                coeffs_f,
                fmask,
                coeffs_g,
                gmask,
                out,
                M,
                eps,
                f_tmp,
                f_res,
                col_work,
                col_out_work,
                f_row,
                g_tmp,
                g_res,
                g_row,
                f_flat,
                g_flat,
                elev_work,
                lo0,
                hi0,
                lo1,
                hi1,
            )


# ---------------------------------------------------------------------------
# Intersection mask: 3D
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _intersection_mask_3d_core(  # noqa: PLR0913
    coeffs_f: npt.NDArray[np.float64],
    fmask: npt.NDArray[np.bool_],
    coeffs_g: npt.NDArray[np.float64],
    gmask: npt.NDArray[np.bool_],
    out: npt.NDArray[np.bool_],
    M: int,
) -> None:
    """Compute intersection mask for two 3D scalar Bernstein polynomials.

    Args:
        coeffs_f (npt.NDArray[np.float64]): Coefficients of f, shape
            ``(nf0, nf1, nf2)``.
        fmask (npt.NDArray[np.bool_]): Nonzero mask of f, shape ``(M, M, M)``.
        coeffs_g (npt.NDArray[np.float64]): Coefficients of g, shape
            ``(ng0, ng1, ng2)``.
        gmask (npt.NDArray[np.bool_]): Nonzero mask of g, shape ``(M, M, M)``.
        out (npt.NDArray[np.bool_]): Output intersection mask, shape
            ``(M, M, M)``, initialized to False.
        M (int): Grid resolution.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`_mask._intersection_mask` instead.
    """
    nf0, nf1, nf2 = coeffs_f.shape[0], coeffs_f.shape[1], coeffs_f.shape[2]
    ng0, ng1, ng2 = coeffs_g.shape[0], coeffs_g.shape[1], coeffs_g.shape[2]
    eps = 0.015625 / M  # padding = 1 / (64 * M)

    nmax_f = max(nf0, max(nf1, nf2))  # noqa: PLW3301
    nmax_g = max(ng0, max(ng1, ng2))  # noqa: PLW3301

    f_tmp1 = np.empty((nf0, nf1, nf2), dtype=np.float64)
    f_tmp2 = np.empty((nf0, nf1, nf2), dtype=np.float64)
    f_res = np.empty((nf0, nf1, nf2), dtype=np.float64)
    f_w1d = np.empty(nmax_f, dtype=np.float64)
    f_w1d_b = np.empty(nmax_f, dtype=np.float64)

    g_tmp1 = np.empty((ng0, ng1, ng2), dtype=np.float64)
    g_tmp2 = np.empty((ng0, ng1, ng2), dtype=np.float64)
    g_res = np.empty((ng0, ng1, ng2), dtype=np.float64)
    g_w1d = np.empty(nmax_g, dtype=np.float64)
    g_w1d_b = np.empty(nmax_g, dtype=np.float64)

    max_len = max(nf0, ng0) * max(nf1, ng1) * max(nf2, ng2)
    f_flat = np.empty(max_len, dtype=np.float64)
    g_flat = np.empty(max_len, dtype=np.float64)
    elev_max = max(max(nf0, ng0), max(max(nf1, ng1), max(nf2, ng2)))  # noqa: PLW3301
    elev_work = np.empty(elev_max, dtype=np.float64)

    _int_mask_3d_recurse(
        coeffs_f,
        fmask,
        coeffs_g,
        gmask,
        out,
        M,
        eps,
        f_tmp1,
        f_tmp2,
        f_res,
        f_w1d,
        f_w1d_b,
        g_tmp1,
        g_tmp2,
        g_res,
        g_w1d,
        g_w1d_b,
        f_flat,
        g_flat,
        elev_work,
        0,
        M,
        0,
        M,
        0,
        M,
    )


@nb_jit(nopython=True, cache=True)
def _elevate_3d_to_common(  # noqa: PLR0912, PLR0915
    f: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    f_flat: npt.NDArray[np.float64],
    g_flat: npt.NDArray[np.float64],
    elev_work: npt.NDArray[np.float64],
) -> int:
    """Degree-elevate two 3D Bernstein arrays to common extent and flatten.

    Args:
        f (npt.NDArray[np.float64]): Coefficients of f, shape ``(nf0, nf1, nf2)``.
        g (npt.NDArray[np.float64]): Coefficients of g, shape ``(ng0, ng1, ng2)``.
        f_flat (npt.NDArray[np.float64]): Output flat elevated f.
        g_flat (npt.NDArray[np.float64]): Output flat elevated g.
        elev_work (npt.NDArray[np.float64]): 1D workspace.

    Returns:
        int: Length of the flat arrays.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    nf0, nf1, nf2 = f.shape[0], f.shape[1], f.shape[2]
    ng0, ng1, ng2 = g.shape[0], g.shape[1], g.shape[2]
    cn0 = max(nf0, ng0)
    cn1 = max(nf1, ng1)
    cn2 = max(nf2, ng2)
    total = cn0 * cn1 * cn2

    # Elevate f: axis 0 → axis 1 → axis 2 → flatten.
    f_mid0 = np.empty((cn0, nf1, nf2), dtype=np.float64)
    if nf0 == cn0:
        for i in range(nf0):
            for j in range(nf1):
                for kk in range(nf2):
                    f_mid0[i, j, kk] = f[i, j, kk]
    else:
        col = np.empty(nf0, dtype=np.float64)
        for j in range(nf1):
            for kk in range(nf2):
                for i in range(nf0):
                    col[i] = f[i, j, kk]
                _elevate_scalar_1d(col, cn0, elev_work)
                for i in range(cn0):
                    f_mid0[i, j, kk] = elev_work[i]

    f_mid1 = np.empty((cn0, cn1, nf2), dtype=np.float64)
    if nf1 == cn1:
        for i in range(cn0):
            for j in range(nf1):
                for kk in range(nf2):
                    f_mid1[i, j, kk] = f_mid0[i, j, kk]
    else:
        col = np.empty(nf1, dtype=np.float64)
        for i in range(cn0):
            for kk in range(nf2):
                for j in range(nf1):
                    col[j] = f_mid0[i, j, kk]
                _elevate_scalar_1d(col, cn1, elev_work)
                for j in range(cn1):
                    f_mid1[i, j, kk] = elev_work[j]

    if nf2 == cn2:
        idx = 0
        for i in range(cn0):
            for j in range(cn1):
                for kk in range(cn2):
                    f_flat[idx] = f_mid1[i, j, kk]
                    idx += 1
    else:
        col = np.empty(nf2, dtype=np.float64)
        idx = 0
        for i in range(cn0):
            for j in range(cn1):
                for kk in range(nf2):
                    col[kk] = f_mid1[i, j, kk]
                _elevate_scalar_1d(col, cn2, elev_work)
                for kk in range(cn2):
                    f_flat[idx] = elev_work[kk]
                    idx += 1

    # Elevate g: axis 0 → axis 1 → axis 2 → flatten.
    g_mid0 = np.empty((cn0, ng1, ng2), dtype=np.float64)
    if ng0 == cn0:
        for i in range(ng0):
            for j in range(ng1):
                for kk in range(ng2):
                    g_mid0[i, j, kk] = g[i, j, kk]
    else:
        col = np.empty(ng0, dtype=np.float64)
        for j in range(ng1):
            for kk in range(ng2):
                for i in range(ng0):
                    col[i] = g[i, j, kk]
                _elevate_scalar_1d(col, cn0, elev_work)
                for i in range(cn0):
                    g_mid0[i, j, kk] = elev_work[i]

    g_mid1 = np.empty((cn0, cn1, ng2), dtype=np.float64)
    if ng1 == cn1:
        for i in range(cn0):
            for j in range(ng1):
                for kk in range(ng2):
                    g_mid1[i, j, kk] = g_mid0[i, j, kk]
    else:
        col = np.empty(ng1, dtype=np.float64)
        for i in range(cn0):
            for kk in range(ng2):
                for j in range(ng1):
                    col[j] = g_mid0[i, j, kk]
                _elevate_scalar_1d(col, cn1, elev_work)
                for j in range(cn1):
                    g_mid1[i, j, kk] = elev_work[j]

    if ng2 == cn2:
        idx = 0
        for i in range(cn0):
            for j in range(cn1):
                for kk in range(cn2):
                    g_flat[idx] = g_mid1[i, j, kk]
                    idx += 1
    else:
        col = np.empty(ng2, dtype=np.float64)
        idx = 0
        for i in range(cn0):
            for j in range(cn1):
                for kk in range(ng2):
                    col[kk] = g_mid1[i, j, kk]
                _elevate_scalar_1d(col, cn2, elev_work)
                for kk in range(cn2):
                    g_flat[idx] = elev_work[kk]
                    idx += 1

    return total


@nb_jit(nopython=True, cache=True)
def _int_mask_3d_recurse(  # noqa: PLR0913
    coeffs_f: npt.NDArray[np.float64],
    fmask: npt.NDArray[np.bool_],
    coeffs_g: npt.NDArray[np.float64],
    gmask: npt.NDArray[np.bool_],
    out: npt.NDArray[np.bool_],
    M: int,
    eps: float,
    f_tmp1: npt.NDArray[np.float64],
    f_tmp2: npt.NDArray[np.float64],
    f_res: npt.NDArray[np.float64],
    f_w1d: npt.NDArray[np.float64],
    f_w1d_b: npt.NDArray[np.float64],
    g_tmp1: npt.NDArray[np.float64],
    g_tmp2: npt.NDArray[np.float64],
    g_res: npt.NDArray[np.float64],
    g_w1d: npt.NDArray[np.float64],
    g_w1d_b: npt.NDArray[np.float64],
    f_flat: npt.NDArray[np.float64],
    g_flat: npt.NDArray[np.float64],
    elev_work: npt.NDArray[np.float64],
    a0: int,
    b0: int,
    a1: int,
    b1: int,
    a2: int,
    b2: int,
) -> None:
    """Recursive helper for 3D intersection mask construction.

    Args:
        coeffs_f (npt.NDArray[np.float64]): Coefficients of f.
        fmask (npt.NDArray[np.bool_]): Nonzero mask of f.
        coeffs_g (npt.NDArray[np.float64]): Coefficients of g.
        gmask (npt.NDArray[np.bool_]): Nonzero mask of g.
        out (npt.NDArray[np.bool_]): Output intersection mask.
        M (int): Grid resolution.
        eps (float): Padding.
        f_tmp1 (npt.NDArray[np.float64]): Workspace for f restriction.
        f_tmp2 (npt.NDArray[np.float64]): Workspace for f restriction.
        f_res (npt.NDArray[np.float64]): Restricted f result.
        f_w1d (npt.NDArray[np.float64]): 1D workspace for f.
        f_w1d_b (npt.NDArray[np.float64]): 1D output workspace for f.
        g_tmp1 (npt.NDArray[np.float64]): Workspace for g restriction.
        g_tmp2 (npt.NDArray[np.float64]): Workspace for g restriction.
        g_res (npt.NDArray[np.float64]): Restricted g result.
        g_w1d (npt.NDArray[np.float64]): 1D workspace for g.
        g_w1d_b (npt.NDArray[np.float64]): 1D output workspace for g.
        f_flat (npt.NDArray[np.float64]): Flat workspace for elevated f.
        g_flat (npt.NDArray[np.float64]): Flat workspace for elevated g.
        elev_work (npt.NDArray[np.float64]): 1D elevation workspace.
        a0 (int): Start of range along axis 0.
        b0 (int): End of range along axis 0.
        a1 (int): Start of range along axis 1.
        b1 (int): End of range along axis 1.
        a2 (int): Start of range along axis 2.
        b2 (int): End of range along axis 2.

    Note:
        Inputs are assumed to be correct (no validation performed).
        ``f_tmp1``, ``f_tmp2``, ``f_res``, ``f_w1d``, ``f_w1d_b``,
        ``g_tmp1``, ``g_tmp2``, ``g_res``, ``g_w1d``, ``g_w1d_b``,
        ``f_flat``, ``g_flat``, and ``elev_work`` are mutated and shared
        across all recursive calls; the parent finishes consuming each
        workspace before recursing.
    """
    # Check overlap with both masks.
    overlap = False
    for i0 in range(a0, b0):
        for i1 in range(a1, b1):
            for i2 in range(a2, b2):
                if fmask[i0, i1, i2] and gmask[i0, i1, i2]:
                    overlap = True
                    break
            if overlap:
                break
        if overlap:
            break
    if not overlap:
        return

    # Restrict both polynomials.
    xa0 = float(a0) / M - eps
    xb0 = float(b0) / M + eps
    xa1 = float(a1) / M - eps
    xb1 = float(b1) / M + eps
    xa2 = float(a2) / M - eps
    xb2 = float(b2) / M + eps

    _restrict_scalar_3d(
        coeffs_f, xa0, xb0, xa1, xb1, xa2, xb2, f_tmp1, f_tmp2, f_res, f_w1d, f_w1d_b
    )
    _restrict_scalar_3d(
        coeffs_g, xa0, xb0, xa1, xb1, xa2, xb2, g_tmp1, g_tmp2, g_res, g_w1d, g_w1d_b
    )

    # Elevate to common extent and orthant test.
    total = _elevate_3d_to_common(f_res, g_res, f_flat, g_flat, elev_work)
    if _orthant_test_core(f_flat[:total], g_flat[:total]):
        return

    # Base case.
    if b0 - a0 == 1 and b1 - a1 == 1 and b2 - a2 == 1:
        out[a0, a1, a2] = True
        return

    # Recurse on 2x2x2 children.
    mid0 = (a0 + b0) // 2
    mid1 = (a1 + b1) // 2
    mid2 = (a2 + b2) // 2
    for s0 in range(2):
        lo0 = a0 if s0 == 0 else mid0
        hi0 = mid0 if s0 == 0 else b0
        for s1 in range(2):
            lo1 = a1 if s1 == 0 else mid1
            hi1 = mid1 if s1 == 0 else b1
            for s2 in range(2):
                lo2 = a2 if s2 == 0 else mid2
                hi2 = mid2 if s2 == 0 else b2
                _int_mask_3d_recurse(
                    coeffs_f,
                    fmask,
                    coeffs_g,
                    gmask,
                    out,
                    M,
                    eps,
                    f_tmp1,
                    f_tmp2,
                    f_res,
                    f_w1d,
                    f_w1d_b,
                    g_tmp1,
                    g_tmp2,
                    g_res,
                    g_w1d,
                    g_w1d_b,
                    f_flat,
                    g_flat,
                    elev_work,
                    lo0,
                    hi0,
                    lo1,
                    hi1,
                    lo2,
                    hi2,
                )


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def _warmup_mask_numba_functions() -> None:
    """Precompile mask Numba kernels with small dummy inputs.

    This function triggers compilation of all mask-related Numba functions,
    ensuring they are cached and ready for use.
    """
    # Point/line queries.
    mask_1d = np.array([True, False], dtype=np.bool_)
    x_1d = np.array([0.25], dtype=np.float64)
    _point_within_mask_core(mask_1d, x_1d, 2, 1)
    _line_intersects_mask_core(mask_1d, np.empty(0, dtype=np.float64), 0, 2, 1)

    mask_2d = np.array([True, False, False, True], dtype=np.bool_)
    x_2d = np.array([0.25, 0.75], dtype=np.float64)
    _point_within_mask_core(mask_2d, x_2d, 2, 2)
    x_1d_for_line = np.array([0.25], dtype=np.float64)
    _line_intersects_mask_core(mask_2d, x_1d_for_line, 0, 2, 2)

    # Scalar restriction.
    c1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    out1 = np.empty(3, dtype=np.float64)
    _restrict_scalar_1d(c1, 0.2, 0.8, out1)

    # Elevation.
    e_out = np.empty(4, dtype=np.float64)
    _elevate_scalar_1d(c1, 4, e_out)

    # Orthant test.
    _orthant_test_base_core(c1, c1, 1)
    _orthant_test_core(c1, c1)

    # 1D nonzero mask.
    fm1 = np.ones(2, dtype=np.bool_)
    om1 = np.zeros(2, dtype=np.bool_)
    _nonzero_mask_1d_core(c1, fm1, om1, 2)

    # 2D nonzero mask.
    c2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    fm2 = np.ones((2, 2), dtype=np.bool_)
    om2 = np.zeros((2, 2), dtype=np.bool_)
    _nonzero_mask_2d_core(c2, fm2, om2, 2)

    # 3D nonzero mask.
    c3 = np.ones((2, 2, 2), dtype=np.float64)
    fm3 = np.ones((2, 2, 2), dtype=np.bool_)
    om3 = np.zeros((2, 2, 2), dtype=np.bool_)
    _nonzero_mask_3d_core(c3, fm3, om3, 2)

    # 2D intersection mask.
    _intersection_mask_2d_core(c2, fm2, c2, fm2, om2, 2)

    # 3D intersection mask.
    _intersection_mask_3d_core(c3, fm3, c3, fm3, om3, 2)
