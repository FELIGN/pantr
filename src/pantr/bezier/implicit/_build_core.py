"""Build phase for the implicit quadrature algorithm.

Implements Algorithm 1 from Saye (2022): given a set of multivariate Bernstein
polynomials, build a recursive dimension-reduction hierarchy. At each level,
the algorithm chooses a height direction, computes face restrictions,
discriminants, and pairwise resultants, then recurses on the reduced problem.

The build result is a flat tuple of typed lists and scalars encoding the
hierarchy for each dimension level.

Main exports:

- :func:`build_2d` -- build hierarchy for 2D polynomials.
- :func:`build_3d` -- build hierarchy for 3D polynomials.

Note:
    Inputs are assumed to be correct (no validation performed).
    These are Layer 3 kernels for the implicit quadrature module.
"""

from __future__ import annotations

import numpy as np
from numba.typed import List as NumbaList

from pantr._numba_compat import nb_jit
from pantr.bezier.implicit._bernstein_core import (
    _face_restrict_2d,
    _face_restrict_3d,
    _normalize_1d,
    _normalize_2d,
)
from pantr.bezier.implicit._mask_core import (
    _collapse_mask_2d,
    _collapse_mask_3d,
    _face_restrict_mask_2d,
    _face_restrict_mask_3d,
    _mask_is_empty_1d,
    _mask_is_empty_2d,
    _mask_is_empty_3d,
    compute_intersection_mask_2d,
    compute_intersection_mask_3d,
    compute_nonzero_mask_1d,
    compute_nonzero_mask_2d,
)
from pantr.bezier.implicit._resultant_core import (
    discriminant_2d,
    discriminant_3d,
    resultant_2d,
    resultant_3d,
)
from pantr.bezier.implicit._score_core import score_estimate_2d, score_estimate_3d

# Integral type constants.
INTEGRAL_INNER: int = 0
INTEGRAL_OUTER_SINGLE: int = 1
INTEGRAL_OUTER_AGGREGATE: int = 2


# ---------------------------------------------------------------------------
# Section A: Axis elimination for 2D -> 1D
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _eliminate_axis_2d(  # noqa: PLR0912
    coeffs_list: NumbaList,
    masks_list: NumbaList,
    k: int,
) -> tuple[NumbaList, NumbaList, bool]:
    """Eliminate axis *k* from 2D polynomials, producing 1D polynomial set.

    For each input polynomial:
    - Extracts lower and upper face restrictions (x_k=0 and x_k=1).
    - Computes pseudo-discriminant if degree >= 2 in direction k.
    For each pair of input polynomials:
    - Computes pairwise resultant along axis k.

    Args:
        coeffs_list (NumbaList): List of 2D coefficient arrays.
        masks_list (NumbaList): List of 2D boolean mask arrays.
        k (int): Axis to eliminate (0 or 1).

    Returns:
        tuple[NumbaList, NumbaList, bool]: (new_coeffs_1d, new_masks_1d,
            has_discriminant) where *has_discriminant* is True if any
            non-empty discriminant or resultant was added (indicating
            branching points that require tanh-sinh quadrature).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    out_coeffs = NumbaList()
    out_masks = NumbaList()

    # Force type for empty list.
    _dummy_c = np.empty(1, dtype=np.float64)
    _dummy_m = np.empty(1, dtype=np.bool_)
    out_coeffs.append(_dummy_c)
    out_masks.append(_dummy_m)
    out_coeffs.pop()
    out_masks.pop()

    n_polys = len(coeffs_list)
    has_disc = False

    # --- Face restrictions ---
    for i in range(n_polys):
        coeffs = coeffs_list[i]
        mask = masks_list[i]
        for side in range(2):
            face_coeffs = _face_restrict_2d(coeffs, k, side)
            face_coeffs = _normalize_1d(face_coeffs)
            face_mask_raw = _face_restrict_mask_2d(mask, k, side)
            face_mask = compute_nonzero_mask_1d(face_coeffs)
            for j in range(len(face_mask)):
                face_mask[j] = face_mask[j] and face_mask_raw[j]
            if not _mask_is_empty_1d(face_mask):
                out_coeffs.append(face_coeffs)
                out_masks.append(face_mask)

    # --- Pseudo-discriminants ---
    for i in range(n_polys):
        coeffs = coeffs_list[i]
        mask = masks_list[i]
        degree_k = coeffs.shape[k] - 1
        if degree_k < 2:  # noqa: PLR2004
            continue
        disc = discriminant_2d(coeffs, k)
        disc = _normalize_1d(disc)
        disc_mask = compute_nonzero_mask_1d(disc)
        collapsed = _collapse_mask_2d(mask, k)
        for j in range(len(disc_mask)):
            disc_mask[j] = disc_mask[j] and collapsed[j]
        if not _mask_is_empty_1d(disc_mask):
            out_coeffs.append(disc)
            out_masks.append(disc_mask)
            has_disc = True

    # --- Pairwise resultants ---
    for i in range(n_polys):
        for j_idx in range(i + 1, n_polys):
            int_mask = compute_intersection_mask_2d(
                coeffs_list[i], masks_list[i], coeffs_list[j_idx], masks_list[j_idx]
            )
            if _mask_is_empty_2d(int_mask):
                continue
            res = resultant_2d(coeffs_list[i], coeffs_list[j_idx], k)
            res = _normalize_1d(res)
            res_mask = compute_nonzero_mask_1d(res)
            collapsed = _collapse_mask_2d(int_mask, k)
            for m_i in range(len(res_mask)):
                res_mask[m_i] = res_mask[m_i] and collapsed[m_i]
            if not _mask_is_empty_1d(res_mask):
                out_coeffs.append(res)
                out_masks.append(res_mask)
                has_disc = True

    return out_coeffs, out_masks, has_disc


# ---------------------------------------------------------------------------
# Section B: Axis elimination for 3D -> 2D
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _eliminate_axis_3d(  # noqa: PLR0912
    coeffs_list: NumbaList,
    masks_list: NumbaList,
    k: int,
) -> tuple[NumbaList, NumbaList, bool]:
    """Eliminate axis *k* from 3D polynomials, producing 2D polynomial set.

    Args:
        coeffs_list (NumbaList): List of 3D coefficient arrays.
        masks_list (NumbaList): List of 3D boolean mask arrays.
        k (int): Axis to eliminate (0, 1, or 2).

    Returns:
        tuple[NumbaList, NumbaList, bool]: (new_coeffs_2d, new_masks_2d,
            has_discriminant).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    out_coeffs = NumbaList()
    out_masks = NumbaList()

    _dummy_c = np.empty((1, 1), dtype=np.float64)
    _dummy_m = np.empty((1, 1), dtype=np.bool_)
    out_coeffs.append(_dummy_c)
    out_masks.append(_dummy_m)
    out_coeffs.pop()
    out_masks.pop()

    n_polys = len(coeffs_list)
    has_disc = False

    # --- Face restrictions ---
    for i in range(n_polys):
        coeffs = coeffs_list[i]
        mask = masks_list[i]
        for side in range(2):
            face_coeffs = _face_restrict_3d(coeffs, k, side)
            face_coeffs = _normalize_2d(face_coeffs)
            face_mask = compute_nonzero_mask_2d(face_coeffs)
            face_mask_raw = _face_restrict_mask_3d(mask, k, side)
            for j0 in range(face_mask.shape[0]):
                for j1 in range(face_mask.shape[1]):
                    face_mask[j0, j1] = face_mask[j0, j1] and face_mask_raw[j0, j1]
            if not _mask_is_empty_2d(face_mask):
                out_coeffs.append(face_coeffs)
                out_masks.append(face_mask)

    # --- Pseudo-discriminants ---
    for i in range(n_polys):
        coeffs = coeffs_list[i]
        mask = masks_list[i]
        degree_k = coeffs.shape[k] - 1
        if degree_k < 2:  # noqa: PLR2004
            continue
        disc = discriminant_3d(coeffs, k)
        disc = _normalize_2d(disc)
        disc_mask = compute_nonzero_mask_2d(disc)
        collapsed = _collapse_mask_3d(mask, k)
        for j0 in range(disc_mask.shape[0]):
            for j1 in range(disc_mask.shape[1]):
                disc_mask[j0, j1] = disc_mask[j0, j1] and collapsed[j0, j1]
        if not _mask_is_empty_2d(disc_mask):
            out_coeffs.append(disc)
            out_masks.append(disc_mask)
            has_disc = True

    # --- Pairwise resultants ---
    for i in range(n_polys):
        for j_idx in range(i + 1, n_polys):
            int_mask = compute_intersection_mask_3d(
                coeffs_list[i], masks_list[i], coeffs_list[j_idx], masks_list[j_idx]
            )
            if _mask_is_empty_3d(int_mask):
                continue
            res = resultant_3d(coeffs_list[i], coeffs_list[j_idx], k)
            res = _normalize_2d(res)
            res_mask = compute_nonzero_mask_2d(res)
            collapsed = _collapse_mask_3d(int_mask, k)
            for m0 in range(res_mask.shape[0]):
                for m1 in range(res_mask.shape[1]):
                    res_mask[m0, m1] = res_mask[m0, m1] and collapsed[m0, m1]
            if not _mask_is_empty_2d(res_mask):
                out_coeffs.append(res)
                out_masks.append(res_mask)
                has_disc = True

    return out_coeffs, out_masks, has_disc


# ---------------------------------------------------------------------------
# Section C: Build hierarchy
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def build_2d(
    coeffs_list: NumbaList,
    masks_list: NumbaList,
) -> tuple[
    NumbaList,  # coeffs_1d (level 0)
    NumbaList,  # masks_1d (level 0)
    int,  # k0
    bool,  # use_ts_0
    int,  # type_0
    NumbaList,  # coeffs_2d (level 1)
    NumbaList,  # masks_2d (level 1)
    int,  # k1
    bool,  # use_ts_1
    int,  # type_1
]:
    """Build the dimension-reduction hierarchy for 2D polynomials.

    Implements Algorithm 1 from Saye (2022):
    1. Choose height direction k1 for the 2D level.
    2. Eliminate axis k1 to produce 1D polynomial set.
    3. The 1D base level uses k0=0 (only one axis).

    Args:
        coeffs_list (NumbaList): List of 2D coefficient arrays.
        masks_list (NumbaList): List of 2D boolean mask arrays.

    Returns:
        tuple: 10-element tuple with per-level data:
            - Level 0 (1D): coeffs_1d, masks_1d, k0, use_ts_0, type_0
            - Level 1 (2D): coeffs_2d, masks_2d, k1, use_ts_1, type_1

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    # Check if all masks are empty (no interfaces).
    all_empty = True
    for i in range(len(masks_list)):
        if not _mask_is_empty_2d(masks_list[i]):
            all_empty = False
            break

    if all_empty:
        # No interfaces: return dummy hierarchy with k=dim (signals TP quad).
        empty_1d = NumbaList()
        empty_1d_m = NumbaList()
        _dc = np.empty(1, dtype=np.float64)
        _dm = np.empty(1, dtype=np.bool_)
        empty_1d.append(_dc)
        empty_1d_m.append(_dm)
        empty_1d.pop()
        empty_1d_m.pop()
        return (
            empty_1d,
            empty_1d_m,
            0,
            False,
            INTEGRAL_INNER,
            coeffs_list,
            masks_list,
            2,
            False,
            INTEGRAL_INNER,  # k=2 signals no interface
        )

    # Score estimation to choose height direction.
    # Directions WITHOUT discriminant features get a +1.0 bonus (prefer GL).
    scores, has_disc = score_estimate_2d(coeffs_list, masks_list)
    max_s = max(abs(scores[0]), abs(scores[1]))
    if max_s > 0:
        scores[0] /= 2 * max_s
        scores[1] /= 2 * max_s
    for d in range(2):
        if not has_disc[d]:
            scores[d] += 1.0

    if scores[0] >= scores[1]:  # noqa: SIM108
        k1 = 0
    else:
        k1 = 1

    # Eliminate axis k1 to produce 1D polynomial set.
    coeffs_1d, masks_1d, _has_disc_1d = _eliminate_axis_2d(coeffs_list, masks_list, k1)

    # Use tanh-sinh when discriminants detected at this level OR
    # when the chosen direction has branching (propagate TS downward).
    use_ts_1 = _has_disc_1d or has_disc[k1]
    type_1 = INTEGRAL_OUTER_SINGLE

    # 1D base level.
    k0 = 0
    use_ts_0 = False
    type_0 = INTEGRAL_INNER

    return (
        coeffs_1d,
        masks_1d,
        k0,
        use_ts_0,
        type_0,
        coeffs_list,
        masks_list,
        k1,
        use_ts_1,
        type_1,
    )


@nb_jit(nopython=True, cache=True)
def build_2d_forced_k(
    coeffs_list: NumbaList,
    masks_list: NumbaList,
    k1: int,
) -> tuple[
    NumbaList,
    NumbaList,
    int,
    bool,
    int,
    NumbaList,
    NumbaList,
    int,
    bool,
    int,
]:
    """Build a 2D hierarchy with a forced height direction.

    Same as :func:`build_2d` but skips score estimation and uses the
    given *k1* directly. Used by the aggregate surface quadrature mode.

    Args:
        coeffs_list (NumbaList): List of 2D coefficient arrays.
        masks_list (NumbaList): List of 2D boolean mask arrays.
        k1 (int): Forced height direction (0 or 1).

    Returns:
        tuple: 10-element tuple (same format as :func:`build_2d`).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    coeffs_1d, masks_1d, _has_disc_1d = _eliminate_axis_2d(coeffs_list, masks_list, k1)
    use_ts_1 = _has_disc_1d
    type_1 = INTEGRAL_OUTER_SINGLE

    return (
        coeffs_1d,
        masks_1d,
        0,
        False,
        INTEGRAL_INNER,
        coeffs_list,
        masks_list,
        k1,
        use_ts_1,
        type_1,
    )


@nb_jit(nopython=True, cache=True)
def build_3d_forced_k(
    coeffs_list: NumbaList,
    masks_list: NumbaList,
    k2: int,
) -> tuple[
    NumbaList,
    NumbaList,
    int,
    bool,
    int,
    NumbaList,
    NumbaList,
    int,
    bool,
    int,
    NumbaList,
    NumbaList,
    int,
    bool,
    int,
]:
    """Build a 3D hierarchy with a forced outermost height direction.

    Args:
        coeffs_list (NumbaList): List of 3D coefficient arrays.
        masks_list (NumbaList): List of 3D boolean mask arrays.
        k2 (int): Forced outermost height direction (0, 1, or 2).

    Returns:
        tuple: 15-element tuple (same format as :func:`build_3d`).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    coeffs_2d, masks_2d, _has_disc_2d = _eliminate_axis_3d(coeffs_list, masks_list, k2)
    use_ts_2 = _has_disc_2d
    type_2 = INTEGRAL_OUTER_SINGLE

    scores_2d, has_disc_2d = score_estimate_2d(coeffs_2d, masks_2d)
    max_s2 = max(abs(scores_2d[0]), abs(scores_2d[1]))
    if max_s2 > 0:
        scores_2d[0] /= 2 * max_s2
        scores_2d[1] /= 2 * max_s2
    for d in range(2):
        if not has_disc_2d[d]:
            scores_2d[d] += 1.0
    k1 = 0 if scores_2d[0] >= scores_2d[1] else 1

    coeffs_1d, masks_1d, _has_disc_1d = _eliminate_axis_2d(coeffs_2d, masks_2d, k1)
    use_ts_1 = _has_disc_1d or use_ts_2 or has_disc_2d[k1]
    type_1 = INTEGRAL_INNER

    return (
        coeffs_1d,
        masks_1d,
        0,
        False,
        INTEGRAL_INNER,
        coeffs_2d,
        masks_2d,
        k1,
        use_ts_1,
        type_1,
        coeffs_list,
        masks_list,
        k2,
        use_ts_2,
        type_2,
    )


@nb_jit(nopython=True, cache=True)
def build_3d(  # noqa: PLR0912, PLR0915
    coeffs_list: NumbaList,
    masks_list: NumbaList,
) -> tuple[
    NumbaList,
    NumbaList,
    int,
    bool,
    int,  # level 0 (1D)
    NumbaList,
    NumbaList,
    int,
    bool,
    int,  # level 1 (2D)
    NumbaList,
    NumbaList,
    int,
    bool,
    int,  # level 2 (3D)
]:
    """Build the dimension-reduction hierarchy for 3D polynomials.

    Implements Algorithm 1 from Saye (2022) for 3D:
    1. Choose height direction k2 for the 3D level.
    2. Eliminate axis k2 to produce 2D polynomial set.
    3. Choose height direction k1 for the 2D level.
    4. Eliminate axis k1 to produce 1D polynomial set.
    5. The 1D base level uses k0=0.

    Args:
        coeffs_list (NumbaList): List of 3D coefficient arrays.
        masks_list (NumbaList): List of 3D boolean mask arrays.

    Returns:
        tuple: 15-element tuple with per-level data (5 per level x 3 levels).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    # Check for empty masks.
    all_empty = True
    for i in range(len(masks_list)):
        if not _mask_is_empty_3d(masks_list[i]):
            all_empty = False
            break

    if all_empty:
        empty_1d = NumbaList()
        empty_2d = NumbaList()
        _dc1 = np.empty(1, dtype=np.float64)
        _dm1 = np.empty(1, dtype=np.bool_)
        _dc2 = np.empty((1, 1), dtype=np.float64)
        _dm2 = np.empty((1, 1), dtype=np.bool_)
        empty_1d.append(_dc1)
        empty_1d.pop()
        empty_2d.append(_dc2)
        empty_2d.pop()
        empty_1d_m = NumbaList()
        empty_2d_m = NumbaList()
        empty_1d_m.append(_dm1)
        empty_1d_m.pop()
        empty_2d_m.append(_dm2)
        empty_2d_m.pop()
        return (
            empty_1d,
            empty_1d_m,
            0,
            False,
            INTEGRAL_INNER,
            empty_2d,
            empty_2d_m,
            0,
            False,
            INTEGRAL_INNER,
            coeffs_list,
            masks_list,
            3,
            False,
            INTEGRAL_INNER,
        )

    # Score for 3D -> choose k2.
    # Directions WITHOUT discriminant features get a +1.0 bonus (prefer GL).
    scores, has_disc_3d = score_estimate_3d(coeffs_list, masks_list)
    max_s = 0.0
    for d in range(3):
        max_s = max(max_s, abs(scores[d]))
    if max_s > 0:
        for d in range(3):
            scores[d] /= 2 * max_s
    for d in range(3):
        if not has_disc_3d[d]:
            scores[d] += 1.0

    k2 = 0
    best = scores[0]
    for d in range(1, 3):
        if scores[d] > best:
            best = scores[d]
            k2 = d

    # Eliminate axis k2 to get 2D polynomial set.
    coeffs_2d, masks_2d, _has_disc_2d = _eliminate_axis_3d(coeffs_list, masks_list, k2)

    # TS is used at this level if branching detected at this level OR
    # the chosen direction has discriminant features.
    use_ts_2 = _has_disc_2d or has_disc_3d[k2]
    type_2 = INTEGRAL_OUTER_SINGLE

    # Score for 2D -> choose k1 (with the same bonus logic).
    scores_2d, has_disc_2d = score_estimate_2d(coeffs_2d, masks_2d)
    max_s2 = max(abs(scores_2d[0]), abs(scores_2d[1]))
    if max_s2 > 0:
        scores_2d[0] /= 2 * max_s2
        scores_2d[1] /= 2 * max_s2
    for d in range(2):
        if not has_disc_2d[d]:
            scores_2d[d] += 1.0

    if scores_2d[0] >= scores_2d[1]:  # noqa: SIM108
        k1 = 0
    else:
        k1 = 1

    # Eliminate axis k1 to get 1D polynomial set.
    coeffs_1d, masks_1d, _has_disc_1d = _eliminate_axis_2d(coeffs_2d, masks_2d, k1)

    # TS propagates: if the outer level uses TS, suggest it for inner levels too.
    use_ts_1 = _has_disc_1d or use_ts_2 or has_disc_2d[k1]
    type_1 = INTEGRAL_INNER

    k0 = 0
    use_ts_0 = False
    type_0 = INTEGRAL_INNER

    return (
        coeffs_1d,
        masks_1d,
        k0,
        use_ts_0,
        type_0,
        coeffs_2d,
        masks_2d,
        k1,
        use_ts_1,
        type_1,
        coeffs_list,
        masks_list,
        k2,
        use_ts_2,
        type_2,
    )
