"""Mask operations for implicit quadrature.

Masks are boolean arrays on an ``M x M x ... x M`` grid (``M = 8``) over
``[0,1]^d``. A ``True`` subcell indicates that the associated polynomial may
have a zero there; ``False`` means it is provably nonzero.

Self-contained Numba nopython implementations. Uses iterative subdivision
with de Casteljau restriction and uniform sign detection for mask construction,
and orthant tests for intersection masks.

Main exports:

- :func:`compute_nonzero_mask_1d` / ``_2d`` / ``_3d`` -- build nonzero masks.
- :func:`compute_intersection_mask_2d` / ``_3d`` -- build intersection masks.
- :func:`_mask_is_empty_1d` / ``_2d`` / ``_3d`` -- test if mask is empty.
- :func:`_collapse_mask_2d` / ``_3d`` -- OR-reduce along one axis.
- :func:`_face_restrict_mask_2d` / ``_3d`` -- extract boundary face mask.
- :func:`_point_within_1d` / ``_2d`` / ``_3d`` -- test if point in active subcell.
- :func:`_line_intersects_2d` / ``_3d`` -- test if line hits active subcells.

Note:
    Inputs are assumed to be correct (no validation performed).
    These are Layer 3 kernels for the implicit quadrature module.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from pantr._numba_compat import nb_jit
from pantr.bezier.implicit._roots import _restrict_scalar

M: int = 8
"""Mask grid resolution per axis."""

_MASK_EPS: float = 1.0 / (64.0 * M)
"""Overlap epsilon for subcell restriction (about 1% of subcell width)."""


@nb_jit(nopython=True, cache=True)
def _clamp_to_mask_index(x: float) -> int:
    """Clamp a coordinate in [0, 1] to a mask subcell index in [0, M-1].

    Args:
        x (float): Coordinate value in [0, 1].

    Returns:
        int: Subcell index clamped to [0, M-1].

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    idx = int(x * M)
    if idx >= M:
        idx = M - 1
    return max(idx, 0)


# ---------------------------------------------------------------------------
# Section A: Uniform sign detection
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _has_uniform_sign(coeffs: npt.NDArray[np.float64]) -> bool:
    """Check if all elements of a flat array have the same strict sign.

    Args:
        coeffs (npt.NDArray[np.float64]): Flat coefficient array.

    Returns:
        bool: True if all > 0 or all < 0.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = coeffs.size
    if n == 0:
        return True
    flat = coeffs.ravel()
    all_pos = True
    all_neg = True
    for i in range(n):
        if flat[i] <= 0.0:
            all_pos = False
        if flat[i] >= 0.0:
            all_neg = False
        if not all_pos and not all_neg:
            return False
    return True


# ---------------------------------------------------------------------------
# Section B: 1D scalar de Casteljau restriction (self-contained)
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _restrict_1d(
    coeffs: npt.NDArray[np.float64],
    lower: float,
    upper: float,
) -> npt.NDArray[np.float64]:
    """Restrict 1D Bernstein coefficients to [lower, upper] via de Casteljau.

    Args:
        coeffs (npt.NDArray[np.float64]): 1D Bernstein coefficients.
        lower (float): Left bound in [0, 1].
        upper (float): Right bound in [0, 1].

    Returns:
        npt.NDArray[np.float64]: Restricted coefficients on [0, 1].

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    return _restrict_scalar(coeffs, lower, upper)


# ---------------------------------------------------------------------------
# Section C: Nonzero mask construction
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def compute_nonzero_mask_1d(
    coeffs: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Compute a conservative nonzero mask for a 1D scalar polynomial.

    Subdivides [0,1] into M subcells and checks each for uniform sign using
    de Casteljau restriction.

    Args:
        coeffs (npt.NDArray[np.float64]): 1D Bernstein coefficients.

    Returns:
        npt.NDArray[np.bool_]: Boolean mask of shape ``(M,)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    out = np.zeros(M, dtype=np.bool_)
    eps = _MASK_EPS

    # Quick check: if all coefficients have uniform sign, no zeros at all.
    if _has_uniform_sign(coeffs):
        return out

    # Check each subcell.
    inv_m = 1.0 / M
    for i in range(M):
        lo = max(i * inv_m - eps, 0.0)
        hi = min((i + 1) * inv_m + eps, 1.0)
        sub = _restrict_1d(coeffs, lo, hi)
        if not _has_uniform_sign(sub):
            out[i] = True

    return out


@nb_jit(nopython=True, cache=True)
def _restrict_2d_subcell(
    coeffs: npt.NDArray[np.float64],
    lo0: float,
    hi0: float,
    lo1: float,
    hi1: float,
) -> npt.NDArray[np.float64]:
    """Restrict a 2D TP Bernstein polynomial to a sub-rectangle.

    Applies 1D restriction sequentially along each axis.

    Args:
        coeffs (npt.NDArray[np.float64]): 2D coefficient array.
        lo0 (float): Lower bound along axis 0.
        hi0 (float): Upper bound along axis 0.
        lo1 (float): Lower bound along axis 1.
        hi1 (float): Upper bound along axis 1.

    Returns:
        npt.NDArray[np.float64]: Restricted 2D coefficient array.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    s0, s1 = coeffs.shape
    # Restrict along axis 0.
    tmp = np.empty((s0, s1), dtype=np.float64)
    for j in range(s1):
        col = np.empty(s0, dtype=np.float64)
        for i in range(s0):
            col[i] = coeffs[i, j]
        res = _restrict_1d(col, lo0, hi0)
        for i in range(s0):
            tmp[i, j] = res[i]
    # Restrict along axis 1.
    out = np.empty((s0, s1), dtype=np.float64)
    for i in range(s0):
        row = np.empty(s1, dtype=np.float64)
        for j in range(s1):
            row[j] = tmp[i, j]
        res = _restrict_1d(row, lo1, hi1)
        for j in range(s1):
            out[i, j] = res[j]
    return out


@nb_jit(nopython=True, cache=True)
def compute_nonzero_mask_2d(
    coeffs: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Compute a conservative nonzero mask for a 2D scalar polynomial.

    Args:
        coeffs (npt.NDArray[np.float64]): 2D Bernstein coefficients.

    Returns:
        npt.NDArray[np.bool_]: Boolean mask of shape ``(M, M)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    out = np.zeros((M, M), dtype=np.bool_)
    eps = _MASK_EPS

    if _has_uniform_sign(coeffs):
        return out

    inv_m = 1.0 / M
    for i0 in range(M):
        lo0 = max(i0 * inv_m - eps, 0.0)
        hi0 = min((i0 + 1) * inv_m + eps, 1.0)
        for i1 in range(M):
            lo1 = max(i1 * inv_m - eps, 0.0)
            hi1 = min((i1 + 1) * inv_m + eps, 1.0)
            sub = _restrict_2d_subcell(coeffs, lo0, hi0, lo1, hi1)
            if not _has_uniform_sign(sub):
                out[i0, i1] = True

    return out


@nb_jit(nopython=True, cache=True)
def _restrict_3d_subcell(  # noqa: PLR0913
    coeffs: npt.NDArray[np.float64],
    lo0: float,
    hi0: float,
    lo1: float,
    hi1: float,
    lo2: float,
    hi2: float,
) -> npt.NDArray[np.float64]:
    """Restrict a 3D TP Bernstein polynomial to a sub-box.

    Args:
        coeffs (npt.NDArray[np.float64]): 3D coefficient array.
        lo0 (float): Lower bound along axis 0.
        hi0 (float): Upper bound along axis 0.
        lo1 (float): Lower bound along axis 1.
        hi1 (float): Upper bound along axis 1.
        lo2 (float): Lower bound along axis 2.
        hi2 (float): Upper bound along axis 2.

    Returns:
        npt.NDArray[np.float64]: Restricted 3D coefficient array.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    s0, s1, s2 = coeffs.shape
    # Restrict along axis 0.
    tmp1 = np.empty((s0, s1, s2), dtype=np.float64)
    for j in range(s1):
        for k in range(s2):
            col = np.empty(s0, dtype=np.float64)
            for i in range(s0):
                col[i] = coeffs[i, j, k]
            res = _restrict_1d(col, lo0, hi0)
            for i in range(s0):
                tmp1[i, j, k] = res[i]
    # Restrict along axis 1.
    tmp2 = np.empty((s0, s1, s2), dtype=np.float64)
    for i in range(s0):
        for k in range(s2):
            col = np.empty(s1, dtype=np.float64)
            for j in range(s1):
                col[j] = tmp1[i, j, k]
            res = _restrict_1d(col, lo1, hi1)
            for j in range(s1):
                tmp2[i, j, k] = res[j]
    # Restrict along axis 2.
    out = np.empty((s0, s1, s2), dtype=np.float64)
    for i in range(s0):
        for j in range(s1):
            col = np.empty(s2, dtype=np.float64)
            for k in range(s2):
                col[k] = tmp2[i, j, k]
            res = _restrict_1d(col, lo2, hi2)
            for k in range(s2):
                out[i, j, k] = res[k]
    return out


@nb_jit(nopython=True, cache=True)
def compute_nonzero_mask_3d(
    coeffs: npt.NDArray[np.float64],
) -> npt.NDArray[np.bool_]:
    """Compute a conservative nonzero mask for a 3D scalar polynomial.

    Args:
        coeffs (npt.NDArray[np.float64]): 3D Bernstein coefficients.

    Returns:
        npt.NDArray[np.bool_]: Boolean mask of shape ``(M, M, M)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    out = np.zeros((M, M, M), dtype=np.bool_)
    eps = _MASK_EPS

    if _has_uniform_sign(coeffs):
        return out

    inv_m = 1.0 / M
    for i0 in range(M):
        lo0 = max(i0 * inv_m - eps, 0.0)
        hi0 = min((i0 + 1) * inv_m + eps, 1.0)
        for i1 in range(M):
            lo1 = max(i1 * inv_m - eps, 0.0)
            hi1 = min((i1 + 1) * inv_m + eps, 1.0)
            for i2 in range(M):
                lo2 = max(i2 * inv_m - eps, 0.0)
                hi2 = min((i2 + 1) * inv_m + eps, 1.0)
                sub = _restrict_3d_subcell(coeffs, lo0, hi0, lo1, hi1, lo2, hi2)
                if not _has_uniform_sign(sub):
                    out[i0, i1, i2] = True

    return out


# ---------------------------------------------------------------------------
# Section D: Intersection mask (orthant test)
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _orthant_test(
    f: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
) -> bool:
    """Test if two flat coefficient arrays provably share no common zero.

    Checks whether there exist scalars alpha, beta such that
    ``alpha * f[i] + beta * g[i] > 0`` for all *i*. If so, *f* and *g*
    cannot both be zero at any point in the corresponding subdomain.

    Args:
        f (npt.NDArray[np.float64]): Flat coefficients of first polynomial.
        g (npt.NDArray[np.float64]): Flat coefficients of second polynomial.

    Returns:
        bool: True if *f* and *g* provably share no common zero.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = len(f)
    if n == 0:
        return True

    # Check sign=+1: find alpha range such that f[i] + alpha*g[i] > 0 for all i.
    for sign in (1, -1):
        alpha_lo = -1e300
        alpha_hi = 1e300
        feasible = True
        for i in range(n):
            fi = float(sign) * f[i]
            gi = g[i]
            if gi > 0.0:
                # fi + alpha*gi > 0 iff alpha > -fi/gi
                bound = -fi / gi
                alpha_lo = max(alpha_lo, bound)
            elif gi < 0.0:
                # fi + alpha*gi > 0 iff alpha < -fi/gi
                bound = -fi / gi
                alpha_hi = min(alpha_hi, bound)
            # gi == 0: need fi > 0
            elif fi <= 0.0:
                feasible = False
                break
            if alpha_lo >= alpha_hi:
                feasible = False
                break
        if feasible and alpha_lo < alpha_hi:
            return True

    return False


@nb_jit(nopython=True, cache=True)
def _degree_elevate_1d_inplace(
    coeffs: npt.NDArray[np.float64],
    target_len: int,
) -> npt.NDArray[np.float64]:
    """Degree-elevate 1D Bernstein coefficients to target length.

    Args:
        coeffs (npt.NDArray[np.float64]): Input coefficients of length ``p+1``.
        target_len (int): Target length (>= len(coeffs)).

    Returns:
        npt.NDArray[np.float64]: Elevated coefficients of length *target_len*.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    cur = coeffs.copy()
    while len(cur) < target_len:
        p = len(cur) - 1
        new_p = p + 1
        elev = np.empty(new_p + 1, dtype=np.float64)
        elev[0] = cur[0]
        elev[new_p] = cur[p]
        for i in range(1, new_p):
            alpha = float(i) / float(new_p)
            elev[i] = alpha * cur[i - 1] + (1.0 - alpha) * cur[i]
        cur = elev
    return cur


@nb_jit(nopython=True, cache=True)
def compute_intersection_mask_2d(  # noqa: PLR0912
    coeffs_f: npt.NDArray[np.float64],
    mask_f: npt.NDArray[np.bool_],
    coeffs_g: npt.NDArray[np.float64],
    mask_g: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """Compute intersection mask for two 2D polynomials.

    Uses orthant tests on each subcell to determine where *f* and *g* may
    share a common zero. Polynomials are degree-elevated to a common degree
    before testing.

    Args:
        coeffs_f (npt.NDArray[np.float64]): Coefficients of first polynomial.
        mask_f (npt.NDArray[np.bool_]): Nonzero mask of first polynomial.
        coeffs_g (npt.NDArray[np.float64]): Coefficients of second polynomial.
        mask_g (npt.NDArray[np.bool_]): Nonzero mask of second polynomial.

    Returns:
        npt.NDArray[np.bool_]: Intersection mask of shape ``(M, M)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    out = np.zeros((M, M), dtype=np.bool_)
    eps = _MASK_EPS
    inv_m = 1.0 / M

    for i0 in range(M):
        for i1 in range(M):
            # Skip if either mask is inactive.
            if not mask_f[i0, i1] or not mask_g[i0, i1]:
                continue

            lo0 = max(i0 * inv_m - eps, 0.0)
            hi0 = min((i0 + 1) * inv_m + eps, 1.0)
            lo1 = max(i1 * inv_m - eps, 0.0)
            hi1 = min((i1 + 1) * inv_m + eps, 1.0)

            sub_f = _restrict_2d_subcell(coeffs_f, lo0, hi0, lo1, hi1)
            sub_g = _restrict_2d_subcell(coeffs_g, lo0, hi0, lo1, hi1)

            # Degree-elevate to common degree in each direction.
            sf0, sf1 = sub_f.shape
            sg0, sg1 = sub_g.shape
            max0 = max(sf0, sg0)
            max1 = max(sf1, sg1)

            # Elevate along axis 0 then axis 1, flatten for orthant test.
            f_flat = np.empty(max0 * max1, dtype=np.float64)
            g_flat = np.empty(max0 * max1, dtype=np.float64)

            # Elevate f.
            f_elev = np.empty((max0, max1), dtype=np.float64)
            for j in range(sf1):
                col = np.empty(sf0, dtype=np.float64)
                for i in range(sf0):
                    col[i] = sub_f[i, j]
                elev = _degree_elevate_1d_inplace(col, max0)
                for i in range(max0):
                    f_elev[i, j] = elev[i]
            if sf1 < max1:
                for i in range(max0):
                    row = np.empty(sf1, dtype=np.float64)
                    for j in range(sf1):
                        row[j] = f_elev[i, j]
                    elev = _degree_elevate_1d_inplace(row, max1)
                    for j in range(max1):
                        f_elev[i, j] = elev[j]

            # Elevate g.
            g_elev = np.empty((max0, max1), dtype=np.float64)
            for j in range(sg1):
                col = np.empty(sg0, dtype=np.float64)
                for i in range(sg0):
                    col[i] = sub_g[i, j]
                elev = _degree_elevate_1d_inplace(col, max0)
                for i in range(max0):
                    g_elev[i, j] = elev[i]
            if sg1 < max1:
                for i in range(max0):
                    row = np.empty(sg1, dtype=np.float64)
                    for j in range(sg1):
                        row[j] = g_elev[i, j]
                    elev = _degree_elevate_1d_inplace(row, max1)
                    for j in range(max1):
                        g_elev[i, j] = elev[j]

            # Flatten.
            idx = 0
            for i in range(max0):
                for j in range(max1):
                    f_flat[idx] = f_elev[i, j]
                    g_flat[idx] = g_elev[i, j]
                    idx += 1

            if not _orthant_test(f_flat, g_flat):
                out[i0, i1] = True

    return out


@nb_jit(nopython=True, cache=True)
def _elevate_3d_to_common(
    sub: npt.NDArray[np.float64],
    max0: int,
    max1: int,
    max2: int,
) -> npt.NDArray[np.float64]:
    """Degree-elevate a 3D subcell polynomial to a common shape.

    Applies 1D degree elevation sequentially along each axis.

    Args:
        sub (npt.NDArray[np.float64]): Input 3D subcell coefficients.
        max0 (int): Target size along axis 0.
        max1 (int): Target size along axis 1.
        max2 (int): Target size along axis 2.

    Returns:
        npt.NDArray[np.float64]: Elevated array of shape ``(max0, max1, max2)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    s0, s1, s2 = sub.shape
    # Elevate along axis 0.
    tmp0 = np.empty((max0, s1, s2), dtype=np.float64)
    for j in range(s1):
        for k in range(s2):
            col = np.empty(s0, dtype=np.float64)
            for i in range(s0):
                col[i] = sub[i, j, k]
            elev = _degree_elevate_1d_inplace(col, max0)
            for i in range(max0):
                tmp0[i, j, k] = elev[i]
    # Elevate along axis 1.
    tmp1 = np.empty((max0, max1, s2), dtype=np.float64)
    for i in range(max0):
        for k in range(s2):
            col = np.empty(s1, dtype=np.float64)
            for j in range(s1):
                col[j] = tmp0[i, j, k]
            elev = _degree_elevate_1d_inplace(col, max1)
            for j in range(max1):
                tmp1[i, j, k] = elev[j]
    # Elevate along axis 2.
    out = np.empty((max0, max1, max2), dtype=np.float64)
    for i in range(max0):
        for j in range(max1):
            col = np.empty(s2, dtype=np.float64)
            for k in range(s2):
                col[k] = tmp1[i, j, k]
            elev = _degree_elevate_1d_inplace(col, max2)
            for k in range(max2):
                out[i, j, k] = elev[k]
    return out


@nb_jit(nopython=True, cache=True)
def compute_intersection_mask_3d(
    coeffs_f: npt.NDArray[np.float64],
    mask_f: npt.NDArray[np.bool_],
    coeffs_g: npt.NDArray[np.float64],
    mask_g: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    """Compute intersection mask for two 3D polynomials.

    Uses orthant tests on each subcell to determine where *f* and *g* may
    share a common zero. Polynomials are degree-elevated to a common degree
    before testing.

    Args:
        coeffs_f (npt.NDArray[np.float64]): Coefficients of first polynomial.
        mask_f (npt.NDArray[np.bool_]): Nonzero mask of first polynomial.
        coeffs_g (npt.NDArray[np.float64]): Coefficients of second polynomial.
        mask_g (npt.NDArray[np.bool_]): Nonzero mask of second polynomial.

    Returns:
        npt.NDArray[np.bool_]: Intersection mask of shape ``(M, M, M)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    out = np.zeros((M, M, M), dtype=np.bool_)
    eps = _MASK_EPS
    inv_m = 1.0 / M

    for i0 in range(M):
        for i1 in range(M):
            for i2 in range(M):
                if not mask_f[i0, i1, i2] or not mask_g[i0, i1, i2]:
                    continue

                lo0 = max(i0 * inv_m - eps, 0.0)
                hi0 = min((i0 + 1) * inv_m + eps, 1.0)
                lo1 = max(i1 * inv_m - eps, 0.0)
                hi1 = min((i1 + 1) * inv_m + eps, 1.0)
                lo2 = max(i2 * inv_m - eps, 0.0)
                hi2 = min((i2 + 1) * inv_m + eps, 1.0)

                sub_f = _restrict_3d_subcell(coeffs_f, lo0, hi0, lo1, hi1, lo2, hi2)
                sub_g = _restrict_3d_subcell(coeffs_g, lo0, hi0, lo1, hi1, lo2, hi2)

                # Degree-elevate to common degree in each direction.
                sf0, sf1, sf2 = sub_f.shape
                sg0, sg1, sg2 = sub_g.shape
                max0 = max(sf0, sg0)
                max1 = max(sf1, sg1)
                max2 = max(sf2, sg2)

                f_elev = _elevate_3d_to_common(sub_f, max0, max1, max2)
                g_elev = _elevate_3d_to_common(sub_g, max0, max1, max2)

                # Flatten for orthant test.
                n_total = max0 * max1 * max2
                f_flat = np.empty(n_total, dtype=np.float64)
                g_flat = np.empty(n_total, dtype=np.float64)
                idx = 0
                for a0 in range(max0):
                    for a1 in range(max1):
                        for a2 in range(max2):
                            f_flat[idx] = f_elev[a0, a1, a2]
                            g_flat[idx] = g_elev[a0, a1, a2]
                            idx += 1

                if not _orthant_test(f_flat, g_flat):
                    out[i0, i1, i2] = True

    return out


# ---------------------------------------------------------------------------
# Section E: Mask queries
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _mask_is_empty_1d(mask: npt.NDArray[np.bool_]) -> bool:
    """Test if a 1D mask has no active subcells.

    Args:
        mask (npt.NDArray[np.bool_]): 1D boolean mask.

    Returns:
        bool: True if all entries are False.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    for i in range(mask.shape[0]):  # noqa: SIM110
        if mask[i]:
            return False
    return True


@nb_jit(nopython=True, cache=True)
def _mask_is_empty_2d(mask: npt.NDArray[np.bool_]) -> bool:
    """Test if a 2D mask has no active subcells.

    Args:
        mask (npt.NDArray[np.bool_]): 2D boolean mask.

    Returns:
        bool: True if all entries are False.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    for i0 in range(mask.shape[0]):
        for i1 in range(mask.shape[1]):
            if mask[i0, i1]:
                return False
    return True


@nb_jit(nopython=True, cache=True)
def _mask_is_empty_3d(mask: npt.NDArray[np.bool_]) -> bool:
    """Test if a 3D mask has no active subcells.

    Args:
        mask (npt.NDArray[np.bool_]): 3D boolean mask.

    Returns:
        bool: True if all entries are False.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    for i0 in range(mask.shape[0]):
        for i1 in range(mask.shape[1]):
            for i2 in range(mask.shape[2]):
                if mask[i0, i1, i2]:
                    return False
    return True


# ---------------------------------------------------------------------------
# Section F: Mask collapse (OR-reduce along one axis)
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _collapse_mask_2d(
    mask: npt.NDArray[np.bool_],
    axis: int,
) -> npt.NDArray[np.bool_]:
    """Collapse a 2D mask to 1D by OR-reducing along *axis*.

    Args:
        mask (npt.NDArray[np.bool_]): 2D boolean mask of shape ``(M, M)``.
        axis (int): Axis to reduce (0 or 1).

    Returns:
        npt.NDArray[np.bool_]: 1D mask of shape ``(M,)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    out = np.zeros(M, dtype=np.bool_)
    if axis == 0:
        for i1 in range(M):
            for i0 in range(M):
                if mask[i0, i1]:
                    out[i1] = True
                    break
    else:
        for i0 in range(M):
            for i1 in range(M):
                if mask[i0, i1]:
                    out[i0] = True
                    break
    return out


@nb_jit(nopython=True, cache=True)
def _collapse_mask_3d(  # noqa: PLR0912
    mask: npt.NDArray[np.bool_],
    axis: int,
) -> npt.NDArray[np.bool_]:
    """Collapse a 3D mask to 2D by OR-reducing along *axis*.

    Args:
        mask (npt.NDArray[np.bool_]): 3D boolean mask of shape ``(M, M, M)``.
        axis (int): Axis to reduce (0, 1, or 2).

    Returns:
        npt.NDArray[np.bool_]: 2D mask of shape ``(M, M)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    out = np.zeros((M, M), dtype=np.bool_)
    if axis == 0:
        for i1 in range(M):
            for i2 in range(M):
                for i0 in range(M):
                    if mask[i0, i1, i2]:
                        out[i1, i2] = True
                        break
    elif axis == 1:
        for i0 in range(M):
            for i2 in range(M):
                for i1 in range(M):
                    if mask[i0, i1, i2]:
                        out[i0, i2] = True
                        break
    else:
        for i0 in range(M):
            for i1 in range(M):
                for i2 in range(M):
                    if mask[i0, i1, i2]:
                        out[i0, i1] = True
                        break
    return out


# ---------------------------------------------------------------------------
# Section G: Face restriction of masks
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _face_restrict_mask_2d(
    mask: npt.NDArray[np.bool_],
    axis: int,
    side: int,
) -> npt.NDArray[np.bool_]:
    """Extract a face from a 2D mask.

    Args:
        mask (npt.NDArray[np.bool_]): 2D boolean mask of shape ``(M, M)``.
        axis (int): Axis to restrict (0 or 1).
        side (int): 0 for lower face, 1 for upper face.

    Returns:
        npt.NDArray[np.bool_]: 1D mask of shape ``(M,)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    out = np.empty(M, dtype=np.bool_)
    idx = 0 if side == 0 else M - 1
    if axis == 0:
        for i1 in range(M):
            out[i1] = mask[idx, i1]
    else:
        for i0 in range(M):
            out[i0] = mask[i0, idx]
    return out


@nb_jit(nopython=True, cache=True)
def _face_restrict_mask_3d(
    mask: npt.NDArray[np.bool_],
    axis: int,
    side: int,
) -> npt.NDArray[np.bool_]:
    """Extract a face from a 3D mask.

    Args:
        mask (npt.NDArray[np.bool_]): 3D boolean mask of shape ``(M, M, M)``.
        axis (int): Axis to restrict (0, 1, or 2).
        side (int): 0 for lower face, 1 for upper face.

    Returns:
        npt.NDArray[np.bool_]: 2D mask of shape ``(M, M)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    out = np.empty((M, M), dtype=np.bool_)
    idx = 0 if side == 0 else M - 1
    if axis == 0:
        for i1 in range(M):
            for i2 in range(M):
                out[i1, i2] = mask[idx, i1, i2]
    elif axis == 1:
        for i0 in range(M):
            for i2 in range(M):
                out[i0, i2] = mask[i0, idx, i2]
    else:
        for i0 in range(M):
            for i1 in range(M):
                out[i0, i1] = mask[i0, i1, idx]
    return out


# ---------------------------------------------------------------------------
# Section H: Point / line queries
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _point_within_1d(
    mask: npt.NDArray[np.bool_],
    x: float,
) -> bool:
    """Test if a 1D point falls in an active subcell.

    Args:
        mask (npt.NDArray[np.bool_]): 1D boolean mask of shape ``(M,)``.
        x (float): Parameter in [0, 1].

    Returns:
        bool: True if the subcell containing *x* is active.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    idx = _clamp_to_mask_index(x)
    return bool(mask[idx])


@nb_jit(nopython=True, cache=True)
def _point_within_2d(
    mask: npt.NDArray[np.bool_],
    x: npt.NDArray[np.float64],
) -> bool:
    """Test if a 2D point falls in an active subcell.

    Args:
        mask (npt.NDArray[np.bool_]): 2D boolean mask of shape ``(M, M)``.
        x (npt.NDArray[np.float64]): Point of shape ``(2,)`` in [0,1]^2.

    Returns:
        bool: True if the subcell containing *x* is active.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    i0 = _clamp_to_mask_index(x[0])
    i1 = _clamp_to_mask_index(x[1])
    return bool(mask[i0, i1])


@nb_jit(nopython=True, cache=True)
def _point_within_3d(
    mask: npt.NDArray[np.bool_],
    x: npt.NDArray[np.float64],
) -> bool:
    """Test if a 3D point falls in an active subcell.

    Args:
        mask (npt.NDArray[np.bool_]): 3D boolean mask of shape ``(M, M, M)``.
        x (npt.NDArray[np.float64]): Point of shape ``(3,)`` in [0,1]^3.

    Returns:
        bool: True if the subcell containing *x* is active.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    i0 = _clamp_to_mask_index(x[0])
    i1 = _clamp_to_mask_index(x[1])
    i2 = _clamp_to_mask_index(x[2])
    return bool(mask[i0, i1, i2])


@nb_jit(nopython=True, cache=True)
def _line_intersects_2d(
    mask: npt.NDArray[np.bool_],
    x_base: float,
    k: int,
) -> bool:
    """Test if a vertical line along axis *k* hits any active subcell.

    Args:
        mask (npt.NDArray[np.bool_]): 2D boolean mask of shape ``(M, M)``.
        x_base (float): Position in tangential direction, in [0, 1].
        k (int): Height direction (0 or 1).

    Returns:
        bool: True if any subcell along the line is active.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    idx_tang = _clamp_to_mask_index(x_base)

    if k == 0:
        for i0 in range(M):
            if mask[i0, idx_tang]:
                return True
    else:
        for i1 in range(M):
            if mask[idx_tang, i1]:
                return True
    return False


@nb_jit(nopython=True, cache=True)
def _line_intersects_3d(
    mask: npt.NDArray[np.bool_],
    x_base: npt.NDArray[np.float64],
    k: int,
) -> bool:
    """Test if a vertical line along axis *k* hits any active subcell.

    Args:
        mask (npt.NDArray[np.bool_]): 3D boolean mask of shape ``(M, M, M)``.
        x_base (npt.NDArray[np.float64]): Base point of shape ``(2,)``
            in tangential directions (ordered by increasing axis index,
            skipping *k*).
        k (int): Height direction (0, 1, or 2).

    Returns:
        bool: True if any subcell along the line is active.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    it0 = _clamp_to_mask_index(x_base[0])
    it1 = _clamp_to_mask_index(x_base[1])

    if k == 0:
        for i0 in range(M):
            if mask[i0, it0, it1]:
                return True
    elif k == 1:
        for i1 in range(M):
            if mask[it0, i1, it1]:
                return True
    else:
        for i2 in range(M):
            if mask[it0, it1, i2]:
                return True
    return False
