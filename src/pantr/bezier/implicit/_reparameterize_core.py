"""Reparameterization phase for implicit domain visualization.

Given a pre-built dimension-reduction hierarchy (from the build phase), generate
structured Lagrange cells that tile the implicit domain (volume) or its zero-set
(surface).  Follows the algorithm from Saye / algoim:

- **Node placement** uses modified Chebyshev (Chebyshev-Lobatto) nodes, which
  cluster at interval endpoints and avoid Runge-phenomenon artefacts.
- **Interval matching** ensures the partition topology (root count) at each
  tangential node matches the reference computed at the interval midpoint.
  When the root count changes (near degenerate configurations), the algorithm
  falls back to intermediate-point sampling and ultimately to the reference
  roots themselves.
- **Surface cells** are identified via sign transitions between adjacent
  intervals (XOR logic), and each surface point is placed at the partition
  boundary between an inside and an outside interval.

Main exports:

- :func:`volume_reparam_2d` -- volume Lagrange quads for 2D.
- :func:`volume_reparam_3d` -- volume Lagrange hexes for 3D.
- :func:`surface_reparam_2d` -- surface Lagrange curves for 2D.
- :func:`surface_reparam_3d` -- surface Lagrange quads for 3D.

Note:
    Inputs are assumed to be correct (no validation performed).
    These are Layer 3 kernels for the implicit reparameterization module.
"""

from __future__ import annotations

import numpy as np
from numba.typed import List as NumbaList
from numpy import typing as npt

from pantr._numba_compat import nb_jit
from pantr.bezier.implicit._bernstein_core import (
    _eval_bernstein_2d,
    _eval_bernstein_3d,
)
from pantr.bezier.implicit._construct_core import (
    _collect_and_partition_from_2d_into,
    _collect_and_partition_from_3d_into,
)

_MERGE_TOL: float = 10.0 * 2.2204460492503131e-16
"""Tolerance for merging nearby roots (same as ``_construct_core``)."""

_MAX_REF_BOUNDS: int = 200
"""Maximum number of partition boundaries per reference (generous)."""

# Backup sampling fractions for interval matching (algoim-style).
_BACKUP_TS = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5], dtype=np.float64)


# ---------------------------------------------------------------------------
# Section A: Helpers
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _modified_chebyshev_node(j: int, q: int) -> float:
    """Return the *j*-th modified Chebyshev node for *q* points on [0, 1].

    These are Chebyshev-Lobatto points (including endpoints), optimal for
    polynomial interpolation and used by algoim for reparameterization.

    Args:
        j: Node index (0-based, 0 ≤ j < q).
        q: Total number of nodes (≥ 2).

    Returns:
        float: Node position in [0, 1].

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    return float(0.5 - 0.5 * np.cos(np.pi * j / (q - 1)))


@nb_jit(nopython=True, cache=True)
def _check_signs_2d(
    coeffs_list: NumbaList,
    pt: npt.NDArray[np.float64],
    signs: npt.NDArray[np.int64],
    n_polys: int,
) -> bool:
    """Check whether all polynomials satisfy the sign condition at *pt*.

    Args:
        coeffs_list: List of 2D coefficient arrays.
        pt: Point of shape ``(2,)``.
        signs: Sign conditions (+1, -1, or 0 to skip).
        n_polys: Number of polynomials to check.

    Returns:
        bool: True if all sign conditions are met.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    for i in range(n_polys):
        if signs[i] == 0:
            continue
        val = _eval_bernstein_2d(coeffs_list[i], pt)
        if signs[i] > 0 and val < 0.0:
            return False
        if signs[i] < 0 and val > 0.0:
            return False
    return True


@nb_jit(nopython=True, cache=True)
def _check_signs_3d(
    coeffs_list: NumbaList,
    pt: npt.NDArray[np.float64],
    signs: npt.NDArray[np.int64],
    n_polys: int,
) -> bool:
    """Check whether all polynomials satisfy the sign condition at *pt*.

    Args:
        coeffs_list: List of 3D coefficient arrays.
        pt: Point of shape ``(3,)``.
        signs: Sign conditions (+1, -1, or 0 to skip).
        n_polys: Number of polynomials to check.

    Returns:
        bool: True if all sign conditions are met.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    for i in range(n_polys):
        if signs[i] == 0:
            continue
        val = _eval_bernstein_3d(coeffs_list[i], pt)
        if signs[i] > 0 and val < 0.0:
            return False
        if signs[i] < 0 and val > 0.0:
            return False
    return True


@nb_jit(nopython=True, cache=True)
def _adapt_partition_2d(  # noqa: PLR0913
    coeffs_2d: NumbaList,
    masks_2d: NumbaList,
    k: int,
    x_new: float,
    ref_bounds: npt.NDArray[np.float64],
    ref_nb: int,
    ref_x: float,
    ws_nodes: npt.NDArray[np.float64],
    ws_poly1d: npt.NDArray[np.float64],
    ws_basis: npt.NDArray[np.float64],
    ws_roots: npt.NDArray[np.float64],
) -> tuple[int, bool]:
    """Compute a partition at *x_new* that matches the reference topology.

    Tries a fresh partition first; if the root count differs, samples
    intermediate points between *x_new* and *ref_x*; finally falls back
    to the reference roots.  Result is written to *ws_nodes*.

    Args:
        coeffs_2d: 2D polynomial coefficients.
        masks_2d: 2D masks.
        k: Height direction.
        x_new: New tangential position.
        ref_bounds: Reference partition boundaries.
        ref_nb: Number of reference boundaries.
        ref_x: Reference tangential position (midpoint).
        ws_nodes: Output boundary buffer (also workspace).
        ws_poly1d: 1D polynomial workspace.
        ws_basis: Basis workspace.
        ws_roots: Roots workspace.

    Returns:
        tuple[int, bool]: ``(count, any_overflow)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    any_overflow = False

    # Step 1: Fresh partition at x_new.
    nb, ovf = _collect_and_partition_from_2d_into(
        coeffs_2d,
        masks_2d,
        k,
        x_new,
        ws_nodes,
        ws_poly1d,
        ws_basis,
        ws_roots,
    )
    any_overflow |= ovf
    if nb == ref_nb:
        return nb, any_overflow

    # Step 2: Try intermediate points (algoim backup strategy).
    for ti in range(len(_BACKUP_TS)):
        t = _BACKUP_TS[ti]
        x_try = (1.0 - t) * x_new + t * ref_x
        nb, ovf = _collect_and_partition_from_2d_into(
            coeffs_2d,
            masks_2d,
            k,
            x_try,
            ws_nodes,
            ws_poly1d,
            ws_basis,
            ws_roots,
        )
        any_overflow |= ovf
        if nb == ref_nb:
            return nb, any_overflow

    # Step 3: Final fallback — use reference roots.
    for i in range(ref_nb):
        ws_nodes[i] = ref_bounds[i]
    return ref_nb, any_overflow


@nb_jit(nopython=True, cache=True)
def _adapt_partition_3d(  # noqa: PLR0913
    coeffs_3d: NumbaList,
    masks_3d: NumbaList,
    k: int,
    x_new: npt.NDArray[np.float64],
    ref_bounds: npt.NDArray[np.float64],
    ref_nb: int,
    ref_x: npt.NDArray[np.float64],
    ws_nodes: npt.NDArray[np.float64],
    ws_poly1d: npt.NDArray[np.float64],
    ws_basis0: npt.NDArray[np.float64],
    ws_basis1: npt.NDArray[np.float64],
    ws_roots: npt.NDArray[np.float64],
) -> tuple[int, bool]:
    """3D variant of :func:`_adapt_partition_2d`.

    Args:
        coeffs_3d: 3D polynomial coefficients.
        masks_3d: 3D masks.
        k: Height direction.
        x_new: New tangential position, shape ``(2,)``.
        ref_bounds: Reference partition boundaries.
        ref_nb: Number of reference boundaries.
        ref_x: Reference tangential position, shape ``(2,)``.
        ws_nodes: Output boundary buffer.
        ws_poly1d: 1D polynomial workspace.
        ws_basis0: First basis workspace.
        ws_basis1: Second basis workspace.
        ws_roots: Roots workspace.

    Returns:
        tuple[int, bool]: ``(count, any_overflow)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    any_overflow = False
    x_try = np.empty(2, dtype=np.float64)

    nb, ovf = _collect_and_partition_from_3d_into(
        coeffs_3d,
        masks_3d,
        k,
        x_new,
        ws_nodes,
        ws_poly1d,
        ws_basis0,
        ws_basis1,
        ws_roots,
    )
    any_overflow |= ovf
    if nb == ref_nb:
        return nb, any_overflow

    for ti in range(len(_BACKUP_TS)):
        t = _BACKUP_TS[ti]
        x_try[0] = (1.0 - t) * x_new[0] + t * ref_x[0]
        x_try[1] = (1.0 - t) * x_new[1] + t * ref_x[1]
        nb, ovf = _collect_and_partition_from_3d_into(
            coeffs_3d,
            masks_3d,
            k,
            x_try,
            ws_nodes,
            ws_poly1d,
            ws_basis0,
            ws_basis1,
            ws_roots,
        )
        any_overflow |= ovf
        if nb == ref_nb:
            return nb, any_overflow

    for i in range(ref_nb):
        ws_nodes[i] = ref_bounds[i]
    return ref_nb, any_overflow


# ---------------------------------------------------------------------------
# Section B: Volume reparameterization
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True, fastmath=True)
def volume_reparam_2d(  # noqa: PLR0912, PLR0913, PLR0915
    base_bounds: npt.NDArray[np.float64],
    base_nb: int,
    k0: int,
    use_ts_0: bool,
    type_0: int,
    coeffs_2d: NumbaList,
    masks_2d: NumbaList,
    k1: int,
    use_ts_1: bool,
    type_1: int,
    lagrange_nodes: npt.NDArray[np.float64],
    signs: npt.NDArray[np.int64],
    n_polys: int,
) -> tuple[npt.NDArray[np.float64], int, bool]:
    """Generate volume Lagrange quad cells for 2D implicit domains.

    Args:
        base_bounds: Pre-computed sorted 1D base partition.
        base_nb: Number of entries in *base_bounds*.
        k0: Height direction at base level (unused, for signature consistency).
        use_ts_0: (Unused.)
        type_0: (Unused.)
        coeffs_2d: 2D polynomial coefficients (original inputs).
        masks_2d: 2D masks (original inputs).
        k1: Height direction at top level.
        use_ts_1: (Unused.)
        type_1: (Unused.)
        lagrange_nodes: 1D Lagrange nodes on ``[0, 1]``, shape ``(q,)``.
        signs: Sign condition per polynomial.
        n_polys: Number of input polynomials.

    Returns:
        tuple: ``(points, n_cells, any_overflow)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    q = len(lagrange_nodes)
    tang1 = 1 - k1

    # No interfaces: full TP cell on [0,1]^2 if sign passes.
    if k1 >= 2:  # noqa: PLR2004
        pt = np.array([0.5, 0.5])
        if not _check_signs_2d(coeffs_2d, pt, signs, n_polys):
            return np.empty((0, 2), dtype=np.float64), 0, False
        points = np.empty((q * q, 2), dtype=np.float64)
        idx = 0
        for i in range(q):
            for j in range(q):
                points[idx, 0] = lagrange_nodes[i]
                points[idx, 1] = lagrange_nodes[j]
                idx += 1
        return points, 1, False

    # Pre-allocate workspace.
    max_deg_k1 = 0
    max_deg_tang1 = 0
    max_nodes_2d = 2
    for i in range(len(coeffs_2d)):
        dk = coeffs_2d[i].shape[k1] - 1
        dt = coeffs_2d[i].shape[tang1] - 1
        max_deg_k1 = max(max_deg_k1, dk)
        max_deg_tang1 = max(max_deg_tang1, dt)
        max_nodes_2d += 2 * dk

    ws_nodes = np.empty(max_nodes_2d, dtype=np.float64)
    ws_poly1d = np.empty(max_deg_k1 + 1, dtype=np.float64)
    ws_basis = np.empty(max_deg_tang1 + 1, dtype=np.float64)
    ws_roots = np.empty(max(max_deg_k1, 1), dtype=np.float64)
    ref_bounds = np.empty(_MAX_REF_BOUNDS, dtype=np.float64)
    ref_active = np.empty(_MAX_REF_BOUNDS, dtype=np.bool_)

    any_overflow = False
    max_total = (base_nb - 1) * 20 * q * q
    max_total = max(max_total, q * q)
    points = np.empty((max_total, 2), dtype=np.float64)
    n_pts = 0
    n_cells = 0
    pt = np.empty(2, dtype=np.float64)

    for b_idx in range(base_nb - 1):
        lo_base = base_bounds[b_idx]
        hi_base = base_bounds[b_idx + 1]
        if hi_base - lo_base < _MERGE_TOL:
            continue

        # --- Phase A: Reference topology at midpoint ---
        x_mid = 0.5 * (lo_base + hi_base)
        nb_ref, ovf = _collect_and_partition_from_2d_into(
            coeffs_2d,
            masks_2d,
            k1,
            x_mid,
            ws_nodes,
            ws_poly1d,
            ws_basis,
            ws_roots,
        )
        any_overflow |= ovf

        # Save reference and determine activity.
        for i in range(nb_ref):
            ref_bounds[i] = ws_nodes[i]
        for i in range(nb_ref - 1):
            ilo = ref_bounds[i]
            ihi = ref_bounds[i + 1]
            if ihi - ilo < _MERGE_TOL:
                ref_active[i] = False
                continue
            mid_h = 0.5 * (ilo + ihi)
            pt[tang1] = x_mid
            pt[k1] = mid_h
            ref_active[i] = _check_signs_2d(coeffs_2d, pt, signs, n_polys)

        # Count active intervals.
        n_active = 0
        for i in range(nb_ref - 1):
            if ref_active[i]:
                n_active += 1
        if n_active == 0:
            continue

        # --- Phase B: Generate cells ---
        for ci in range(nb_ref - 1):
            if not ref_active[ci]:
                continue

            for qi in range(q):
                x_tang = lo_base + (hi_base - lo_base) * lagrange_nodes[qi]

                # Adapt inner partition.
                nb_inner, ovf = _adapt_partition_2d(
                    coeffs_2d,
                    masks_2d,
                    k1,
                    x_tang,
                    ref_bounds,
                    nb_ref,
                    x_mid,
                    ws_nodes,
                    ws_poly1d,
                    ws_basis,
                    ws_roots,
                )
                any_overflow |= ovf

                ilo = ws_nodes[ci]
                ihi = ws_nodes[ci + 1]

                for qj in range(q):
                    x_height = ilo + (ihi - ilo) * lagrange_nodes[qj]

                    if n_pts >= len(points):
                        new_cap = len(points) * 2
                        new_p = np.empty((new_cap, 2), dtype=np.float64)
                        for c_i in range(n_pts):
                            new_p[c_i, 0] = points[c_i, 0]
                            new_p[c_i, 1] = points[c_i, 1]
                        points = new_p

                    points[n_pts, tang1] = x_tang
                    points[n_pts, k1] = x_height
                    n_pts += 1

            n_cells += 1

    return points[:n_pts].copy(), n_cells, any_overflow


@nb_jit(nopython=True, cache=True, fastmath=True)
def volume_reparam_3d(  # noqa: PLR0912, PLR0913, PLR0915
    base_bounds: npt.NDArray[np.float64],
    base_nb: int,
    k0: int,
    use_ts_0: bool,
    type_0: int,
    coeffs_2d: NumbaList,
    masks_2d: NumbaList,
    k1: int,
    use_ts_1: bool,
    type_1: int,
    coeffs_3d: NumbaList,
    masks_3d: NumbaList,
    k2: int,
    use_ts_2: bool,
    type_2: int,
    lagrange_nodes: npt.NDArray[np.float64],
    signs: npt.NDArray[np.int64],
    n_polys: int,
) -> tuple[npt.NDArray[np.float64], int, bool]:
    """Generate volume Lagrange hex cells for 3D implicit domains.

    Args:
        base_bounds: Pre-computed sorted 1D base partition.
        base_nb: Number of entries in *base_bounds*.
        k0: Height direction at base level (unused).
        use_ts_0: (Unused.)
        type_0: (Unused.)
        coeffs_2d: 2D polynomial coefficients (middle level, derived).
        masks_2d: 2D masks.
        k1: Height direction at middle level.
        use_ts_1: (Unused.)
        type_1: (Unused.)
        coeffs_3d: 3D polynomial coefficients (original inputs).
        masks_3d: 3D masks.
        k2: Height direction at top level.
        use_ts_2: (Unused.)
        type_2: (Unused.)
        lagrange_nodes: 1D Lagrange nodes on ``[0, 1]``, shape ``(q,)``.
        signs: Sign condition per polynomial.
        n_polys: Number of input polynomials.

    Returns:
        tuple: ``(points, n_cells, any_overflow)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    q = len(lagrange_nodes)

    if k2 >= 3:  # noqa: PLR2004
        pt = np.array([0.5, 0.5, 0.5])
        if not _check_signs_3d(coeffs_3d, pt, signs, n_polys):
            return np.empty((0, 3), dtype=np.float64), 0, False
        n_total = q * q * q
        points = np.empty((n_total, 3), dtype=np.float64)
        idx = 0
        for i in range(q):
            for j in range(q):
                for l in range(q):  # noqa: E741
                    points[idx, 0] = lagrange_nodes[i]
                    points[idx, 1] = lagrange_nodes[j]
                    points[idx, 2] = lagrange_nodes[l]
                    idx += 1
        return points, 1, False

    # Axis mapping (same as volume_quad_3d).
    tang2 = np.empty(2, dtype=np.int64)
    ti = 0
    for d in range(3):
        if d != k2:
            tang2[ti] = d
            ti += 1
    tang1_2d = 1 - k1
    axis_k1_3d = tang2[k1]
    axis_tang1_3d = tang2[tang1_2d]

    # Level 1 workspace.
    max_deg_k1 = 0
    max_deg_tang1_2d = 0
    max_nodes_2d = 2
    for i in range(len(coeffs_2d)):
        dk = coeffs_2d[i].shape[k1] - 1
        dt = coeffs_2d[i].shape[1 - k1] - 1
        max_deg_k1 = max(max_deg_k1, dk)
        max_deg_tang1_2d = max(max_deg_tang1_2d, dt)
        max_nodes_2d += 2 * dk
    ws1_nodes = np.empty(max_nodes_2d, dtype=np.float64)
    ws1_poly1d = np.empty(max_deg_k1 + 1, dtype=np.float64)
    ws1_basis = np.empty(max_deg_tang1_2d + 1, dtype=np.float64)
    ws1_roots = np.empty(max(max_deg_k1, 1), dtype=np.float64)

    # Level 2 workspace.
    max_deg_k2 = 0
    max_deg_tang0_3d = 0
    max_deg_tang1_3d = 0
    max_nodes_3d = 2
    for i in range(len(coeffs_3d)):
        dk = coeffs_3d[i].shape[k2] - 1
        max_deg_k2 = max(max_deg_k2, dk)
        max_nodes_3d += 2 * dk
        ti_loc = 0
        for d in range(3):
            if d != k2:
                dt_loc = coeffs_3d[i].shape[d] - 1
                if ti_loc == 0:
                    max_deg_tang0_3d = max(max_deg_tang0_3d, dt_loc)
                else:
                    max_deg_tang1_3d = max(max_deg_tang1_3d, dt_loc)
                ti_loc += 1
    ws2_nodes = np.empty(max_nodes_3d, dtype=np.float64)
    ws2_poly1d = np.empty(max_deg_k2 + 1, dtype=np.float64)
    ws2_basis0 = np.empty(max_deg_tang0_3d + 1, dtype=np.float64)
    ws2_basis1 = np.empty(max_deg_tang1_3d + 1, dtype=np.float64)
    ws2_roots = np.empty(max(max_deg_k2, 1), dtype=np.float64)

    ref1_bounds = np.empty(_MAX_REF_BOUNDS, dtype=np.float64)
    ref1_active = np.empty(_MAX_REF_BOUNDS, dtype=np.bool_)
    ref2_active = np.empty(_MAX_REF_BOUNDS, dtype=np.bool_)

    # Per-cell reference storage: (l1_index, ref2 bounds snapshot).
    max_combos = _MAX_REF_BOUNDS
    combo_l1_idx = np.empty(max_combos, dtype=np.int64)
    combo_l2_idx = np.empty(max_combos, dtype=np.int64)
    combo_ref2_bounds = np.empty((max_combos, _MAX_REF_BOUNDS), dtype=np.float64)
    combo_ref2_nb = np.empty(max_combos, dtype=np.int64)

    any_overflow = False
    q3 = q * q * q
    max_total = q3 * 8
    points = np.empty((max_total, 3), dtype=np.float64)
    n_pts = 0
    n_cells = 0
    x_base_3d = np.empty(2, dtype=np.float64)
    ref_x_3d = np.empty(2, dtype=np.float64)
    pt = np.empty(3, dtype=np.float64)

    for b0 in range(base_nb - 1):
        lo0 = base_bounds[b0]
        hi0 = base_bounds[b0 + 1]
        if hi0 - lo0 < _MERGE_TOL:
            continue
        base_mid = 0.5 * (lo0 + hi0)

        # --- Phase A: Reference topology ---
        # Level 1 reference at base midpoint.
        nb1_ref, ovf = _collect_and_partition_from_2d_into(
            coeffs_2d,
            masks_2d,
            k1,
            base_mid,
            ws1_nodes,
            ws1_poly1d,
            ws1_basis,
            ws1_roots,
        )
        any_overflow |= ovf
        for i in range(nb1_ref):
            ref1_bounds[i] = ws1_nodes[i]

        # For each level-1 interval, compute level-2 reference + activity.
        n_combos = 0
        for b1 in range(nb1_ref - 1):
            l1_lo = ref1_bounds[b1]
            l1_hi = ref1_bounds[b1 + 1]
            if l1_hi - l1_lo < _MERGE_TOL:
                ref1_active[b1] = False
                continue
            l1_mid = 0.5 * (l1_lo + l1_hi)

            if tang1_2d == 0:
                x_base_3d[0] = base_mid
                x_base_3d[1] = l1_mid
            else:
                x_base_3d[0] = l1_mid
                x_base_3d[1] = base_mid

            nb2_ref, ovf = _collect_and_partition_from_3d_into(
                coeffs_3d,
                masks_3d,
                k2,
                x_base_3d,
                ws2_nodes,
                ws2_poly1d,
                ws2_basis0,
                ws2_basis1,
                ws2_roots,
            )
            any_overflow |= ovf

            has_active = False
            for b2 in range(nb2_ref - 1):
                l2_lo = ws2_nodes[b2]
                l2_hi = ws2_nodes[b2 + 1]
                if l2_hi - l2_lo < _MERGE_TOL:
                    ref2_active[b2] = False
                    continue
                l2_mid = 0.5 * (l2_lo + l2_hi)
                pt[axis_tang1_3d] = base_mid
                pt[axis_k1_3d] = l1_mid
                pt[k2] = l2_mid
                is_inside = _check_signs_3d(coeffs_3d, pt, signs, n_polys)
                ref2_active[b2] = is_inside
                if is_inside:
                    has_active = True

            ref1_active[b1] = has_active

            # Store combos for active level-2 intervals.
            if has_active:
                for b2 in range(nb2_ref - 1):
                    if ref2_active[b2] and n_combos < max_combos:
                        combo_l1_idx[n_combos] = b1
                        combo_l2_idx[n_combos] = b2
                        combo_ref2_nb[n_combos] = nb2_ref
                        for ii in range(nb2_ref):
                            combo_ref2_bounds[n_combos, ii] = ws2_nodes[ii]
                        n_combos += 1

        if n_combos == 0:
            continue

        # --- Phase B: Generate cells ---
        for ci in range(n_combos):
            l1_ci = combo_l1_idx[ci]
            l2_ci = combo_l2_idx[ci]
            l1_mid = 0.5 * (ref1_bounds[l1_ci] + ref1_bounds[l1_ci + 1])

            for q0 in range(q):
                x_1d = lo0 + (hi0 - lo0) * lagrange_nodes[q0]

                nb1, ovf = _adapt_partition_2d(
                    coeffs_2d,
                    masks_2d,
                    k1,
                    x_1d,
                    ref1_bounds,
                    nb1_ref,
                    base_mid,
                    ws1_nodes,
                    ws1_poly1d,
                    ws1_basis,
                    ws1_roots,
                )
                any_overflow |= ovf
                l1_lo = ws1_nodes[l1_ci]
                l1_hi = ws1_nodes[l1_ci + 1]

                for q1 in range(q):
                    x_2d = l1_lo + (l1_hi - l1_lo) * lagrange_nodes[q1]

                    if tang1_2d == 0:
                        x_base_3d[0] = x_1d
                        x_base_3d[1] = x_2d
                        ref_x_3d[0] = base_mid
                        ref_x_3d[1] = l1_mid
                    else:
                        x_base_3d[0] = x_2d
                        x_base_3d[1] = x_1d
                        ref_x_3d[0] = l1_mid
                        ref_x_3d[1] = base_mid

                    local_ref_nb = combo_ref2_nb[ci]
                    nb2, ovf = _adapt_partition_3d(
                        coeffs_3d,
                        masks_3d,
                        k2,
                        x_base_3d,
                        combo_ref2_bounds[ci],
                        local_ref_nb,
                        ref_x_3d,
                        ws2_nodes,
                        ws2_poly1d,
                        ws2_basis0,
                        ws2_basis1,
                        ws2_roots,
                    )
                    any_overflow |= ovf
                    l2_lo = ws2_nodes[l2_ci]
                    l2_hi = ws2_nodes[l2_ci + 1]

                    for q2 in range(q):
                        x_3d = l2_lo + (l2_hi - l2_lo) * lagrange_nodes[q2]

                        if n_pts >= len(points):
                            new_cap = len(points) * 2
                            new_p = np.empty((new_cap, 3), dtype=np.float64)
                            for c_i in range(n_pts):
                                for di in range(3):
                                    new_p[c_i, di] = points[c_i, di]
                            points = new_p

                        points[n_pts, axis_tang1_3d] = x_1d
                        points[n_pts, axis_k1_3d] = x_2d
                        points[n_pts, k2] = x_3d
                        n_pts += 1

            n_cells += 1

    return points[:n_pts].copy(), n_cells, any_overflow


# ---------------------------------------------------------------------------
# Section C: Surface reparameterization
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True, fastmath=True)
def surface_reparam_2d(  # noqa: PLR0912, PLR0913, PLR0915
    base_bounds: npt.NDArray[np.float64],
    base_nb: int,
    k0: int,
    use_ts_0: bool,
    type_0: int,
    coeffs_2d: NumbaList,
    masks_2d: NumbaList,
    k1: int,
    use_ts_1: bool,
    type_1: int,
    poly_idx: int,
    lagrange_nodes: npt.NDArray[np.float64],
    signs: npt.NDArray[np.int64],
    n_polys: int,
) -> tuple[npt.NDArray[np.float64], int, bool]:
    """Generate surface Lagrange curve cells for 2D levelsets.

    Uses sign-transition (XOR) logic: the surface is at boundaries
    between inside and outside intervals.

    Args:
        base_bounds: Pre-computed sorted 1D base partition.
        base_nb: Number of entries in *base_bounds*.
        k0: Height direction at base level (unused).
        use_ts_0: (Unused.)
        type_0: (Unused.)
        coeffs_2d: 2D polynomial coefficients (original inputs).
        masks_2d: 2D masks.
        k1: Height direction at top level.
        use_ts_1: (Unused.)
        type_1: (Unused.)
        poly_idx: Index of the polynomial whose zero set to trace.
        lagrange_nodes: 1D Lagrange nodes on ``[0, 1]``, shape ``(q,)``.
        signs: Sign condition per polynomial (``signs[poly_idx]`` ignored).
        n_polys: Number of input polynomials.

    Returns:
        tuple: ``(points, n_cells, any_overflow)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    q = len(lagrange_nodes)
    tang1 = 1 - k1

    if k1 >= 2:  # noqa: PLR2004
        return np.empty((0, 2), dtype=np.float64), 0, False

    max_deg_k1 = 0
    max_deg_tang1 = 0
    max_nodes_2d = 2
    for i in range(len(coeffs_2d)):
        dk = coeffs_2d[i].shape[k1] - 1
        dt = coeffs_2d[i].shape[tang1] - 1
        max_deg_k1 = max(max_deg_k1, dk)
        max_deg_tang1 = max(max_deg_tang1, dt)
        max_nodes_2d += 2 * dk
    ws_nodes = np.empty(max_nodes_2d, dtype=np.float64)
    ws_poly1d = np.empty(max_deg_k1 + 1, dtype=np.float64)
    ws_basis = np.empty(max_deg_tang1 + 1, dtype=np.float64)
    ws_roots = np.empty(max(max_deg_k1, 1), dtype=np.float64)
    ref_bounds = np.empty(_MAX_REF_BOUNDS, dtype=np.float64)
    ref_active = np.empty(_MAX_REF_BOUNDS, dtype=np.bool_)

    any_overflow = False
    max_total = q * (base_nb - 1) * 10
    max_total = max(max_total, q)
    points = np.empty((max_total, 2), dtype=np.float64)
    n_pts = 0
    n_cells = 0
    pt = np.empty(2, dtype=np.float64)

    for b_idx in range(base_nb - 1):
        lo_base = base_bounds[b_idx]
        hi_base = base_bounds[b_idx + 1]
        if hi_base - lo_base < _MERGE_TOL:
            continue

        # --- Phase A: Reference at midpoint ---
        x_mid = 0.5 * (lo_base + hi_base)
        nb_ref, ovf = _collect_and_partition_from_2d_into(
            coeffs_2d,
            masks_2d,
            k1,
            x_mid,
            ws_nodes,
            ws_poly1d,
            ws_basis,
            ws_roots,
        )
        any_overflow |= ovf
        for i in range(nb_ref):
            ref_bounds[i] = ws_nodes[i]

        # Activity flags (sign condition check).
        for i in range(nb_ref - 1):
            ilo = ref_bounds[i]
            ihi = ref_bounds[i + 1]
            if ihi - ilo < _MERGE_TOL:
                ref_active[i] = False
                continue
            mid_h = 0.5 * (ilo + ihi)
            pt[tang1] = x_mid
            pt[k1] = mid_h
            ref_active[i] = _check_signs_2d(coeffs_2d, pt, signs, n_polys)

        # Sign transitions: surface lies at boundaries between in/out.
        # Transition at interval i means: active[i] != active[i+1].
        # The surface point is at ref_bounds[i+1] (the shared boundary).
        n_transitions = 0
        transition_indices = np.empty(nb_ref, dtype=np.int64)
        for i in range(nb_ref - 2):
            if ref_active[i] != ref_active[i + 1]:
                transition_indices[n_transitions] = i
                n_transitions += 1

        if n_transitions == 0:
            continue

        # --- Phase B: Generate surface cells ---
        for ti in range(n_transitions):
            si = transition_indices[ti]

            for qi in range(q):
                x_tang = lo_base + (hi_base - lo_base) * lagrange_nodes[qi]

                nb_inner, ovf = _adapt_partition_2d(
                    coeffs_2d,
                    masks_2d,
                    k1,
                    x_tang,
                    ref_bounds,
                    nb_ref,
                    x_mid,
                    ws_nodes,
                    ws_poly1d,
                    ws_basis,
                    ws_roots,
                )
                any_overflow |= ovf

                # Surface point at the right endpoint of the transition.
                x_height = ws_nodes[si + 1]

                if n_pts >= len(points):
                    new_cap = len(points) * 2
                    new_p = np.empty((new_cap, 2), dtype=np.float64)
                    for c_i in range(n_pts):
                        new_p[c_i, 0] = points[c_i, 0]
                        new_p[c_i, 1] = points[c_i, 1]
                    points = new_p

                points[n_pts, tang1] = x_tang
                points[n_pts, k1] = x_height
                n_pts += 1

            n_cells += 1

    return points[:n_pts].copy(), n_cells, any_overflow


@nb_jit(nopython=True, cache=True, fastmath=True)
def surface_reparam_3d(  # noqa: PLR0912, PLR0913, PLR0915
    base_bounds: npt.NDArray[np.float64],
    base_nb: int,
    k0: int,
    use_ts_0: bool,
    type_0: int,
    coeffs_2d: NumbaList,
    masks_2d: NumbaList,
    k1: int,
    use_ts_1: bool,
    type_1: int,
    coeffs_3d: NumbaList,
    masks_3d: NumbaList,
    k2: int,
    use_ts_2: bool,
    type_2: int,
    poly_idx: int,
    lagrange_nodes: npt.NDArray[np.float64],
    signs: npt.NDArray[np.int64],
    n_polys: int,
) -> tuple[npt.NDArray[np.float64], int, bool]:
    """Generate surface Lagrange quad cells for 3D levelsets.

    Uses sign-transition logic at the innermost level (k2 direction).

    Args:
        base_bounds: Pre-computed sorted 1D base partition.
        base_nb: Number of entries in *base_bounds*.
        k0: Height direction at base level (unused).
        use_ts_0: (Unused.)
        type_0: (Unused.)
        coeffs_2d: 2D polynomial coefficients (middle level, derived).
        masks_2d: 2D masks.
        k1: Height direction at middle level.
        use_ts_1: (Unused.)
        type_1: (Unused.)
        coeffs_3d: 3D polynomial coefficients (original inputs).
        masks_3d: 3D masks.
        k2: Height direction at top level.
        use_ts_2: (Unused.)
        type_2: (Unused.)
        poly_idx: Index of the polynomial whose zero set to trace.
        lagrange_nodes: 1D Lagrange nodes on ``[0, 1]``, shape ``(q,)``.
        signs: Sign condition per polynomial (``signs[poly_idx]`` ignored).
        n_polys: Number of input polynomials.

    Returns:
        tuple: ``(points, n_cells, any_overflow)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    q = len(lagrange_nodes)

    if k2 >= 3:  # noqa: PLR2004
        return np.empty((0, 3), dtype=np.float64), 0, False

    tang2 = np.empty(2, dtype=np.int64)
    ti = 0
    for d in range(3):
        if d != k2:
            tang2[ti] = d
            ti += 1
    tang1_2d = 1 - k1
    axis_k1_3d = tang2[k1]
    axis_tang1_3d = tang2[tang1_2d]

    # Level 1 workspace.
    max_deg_k1 = 0
    max_deg_tang1_2d = 0
    max_nodes_2d = 2
    for i in range(len(coeffs_2d)):
        dk = coeffs_2d[i].shape[k1] - 1
        dt = coeffs_2d[i].shape[1 - k1] - 1
        max_deg_k1 = max(max_deg_k1, dk)
        max_deg_tang1_2d = max(max_deg_tang1_2d, dt)
        max_nodes_2d += 2 * dk
    ws1_nodes = np.empty(max_nodes_2d, dtype=np.float64)
    ws1_poly1d = np.empty(max_deg_k1 + 1, dtype=np.float64)
    ws1_basis = np.empty(max_deg_tang1_2d + 1, dtype=np.float64)
    ws1_roots = np.empty(max(max_deg_k1, 1), dtype=np.float64)

    # Level 2 workspace.
    max_deg_k2 = 0
    max_deg_tang0_3d = 0
    max_deg_tang1_3d = 0
    max_nodes_3d = 2
    for i in range(len(coeffs_3d)):
        dk = coeffs_3d[i].shape[k2] - 1
        max_deg_k2 = max(max_deg_k2, dk)
        max_nodes_3d += 2 * dk
        ti_loc = 0
        for d in range(3):
            if d != k2:
                dt_loc = coeffs_3d[i].shape[d] - 1
                if ti_loc == 0:
                    max_deg_tang0_3d = max(max_deg_tang0_3d, dt_loc)
                else:
                    max_deg_tang1_3d = max(max_deg_tang1_3d, dt_loc)
                ti_loc += 1
    ws2_nodes = np.empty(max_nodes_3d, dtype=np.float64)
    ws2_poly1d = np.empty(max_deg_k2 + 1, dtype=np.float64)
    ws2_basis0 = np.empty(max_deg_tang0_3d + 1, dtype=np.float64)
    ws2_basis1 = np.empty(max_deg_tang1_3d + 1, dtype=np.float64)
    ws2_roots = np.empty(max(max_deg_k2, 1), dtype=np.float64)

    ref1_bounds = np.empty(_MAX_REF_BOUNDS, dtype=np.float64)
    ref2_active = np.empty(_MAX_REF_BOUNDS, dtype=np.bool_)

    max_combos = _MAX_REF_BOUNDS
    combo_l1_idx = np.empty(max_combos, dtype=np.int64)
    combo_trans_bound_idx = np.empty(max_combos, dtype=np.int64)
    combo_ref2_bounds = np.empty((max_combos, _MAX_REF_BOUNDS), dtype=np.float64)
    combo_ref2_nb = np.empty(max_combos, dtype=np.int64)

    any_overflow = False
    max_total = q * q * (base_nb - 1) * 20
    max_total = max(max_total, q * q)
    points = np.empty((max_total, 3), dtype=np.float64)
    n_pts = 0
    n_cells = 0
    x_base_3d = np.empty(2, dtype=np.float64)
    ref_x_3d = np.empty(2, dtype=np.float64)
    pt = np.empty(3, dtype=np.float64)

    for b0 in range(base_nb - 1):
        lo0 = base_bounds[b0]
        hi0 = base_bounds[b0 + 1]
        if hi0 - lo0 < _MERGE_TOL:
            continue
        base_mid = 0.5 * (lo0 + hi0)

        # --- Phase A: Reference ---
        nb1_ref, ovf = _collect_and_partition_from_2d_into(
            coeffs_2d,
            masks_2d,
            k1,
            base_mid,
            ws1_nodes,
            ws1_poly1d,
            ws1_basis,
            ws1_roots,
        )
        any_overflow |= ovf
        for i in range(nb1_ref):
            ref1_bounds[i] = ws1_nodes[i]

        n_combos = 0
        for b1 in range(nb1_ref - 1):
            l1_lo = ref1_bounds[b1]
            l1_hi = ref1_bounds[b1 + 1]
            if l1_hi - l1_lo < _MERGE_TOL:
                continue
            l1_mid = 0.5 * (l1_lo + l1_hi)

            if tang1_2d == 0:
                x_base_3d[0] = base_mid
                x_base_3d[1] = l1_mid
            else:
                x_base_3d[0] = l1_mid
                x_base_3d[1] = base_mid

            nb2_ref, ovf = _collect_and_partition_from_3d_into(
                coeffs_3d,
                masks_3d,
                k2,
                x_base_3d,
                ws2_nodes,
                ws2_poly1d,
                ws2_basis0,
                ws2_basis1,
                ws2_roots,
            )
            any_overflow |= ovf

            # Activity + sign transitions for level 2.
            for b2 in range(nb2_ref - 1):
                l2_lo = ws2_nodes[b2]
                l2_hi = ws2_nodes[b2 + 1]
                if l2_hi - l2_lo < _MERGE_TOL:
                    ref2_active[b2] = False
                    continue
                l2_mid = 0.5 * (l2_lo + l2_hi)
                pt[axis_tang1_3d] = base_mid
                pt[axis_k1_3d] = l1_mid
                pt[k2] = l2_mid
                ref2_active[b2] = _check_signs_3d(coeffs_3d, pt, signs, n_polys)

            # Sign transitions along k2.
            for b2 in range(nb2_ref - 2):
                if ref2_active[b2] != ref2_active[b2 + 1] and n_combos < max_combos:
                    combo_l1_idx[n_combos] = b1
                    combo_trans_bound_idx[n_combos] = b2 + 1
                    combo_ref2_nb[n_combos] = nb2_ref
                    for ii in range(nb2_ref):
                        combo_ref2_bounds[n_combos, ii] = ws2_nodes[ii]
                    n_combos += 1

        if n_combos == 0:
            continue

        # --- Phase B: Generate surface cells ---
        for ci in range(n_combos):
            l1_ci = combo_l1_idx[ci]
            trans_idx = combo_trans_bound_idx[ci]
            l1_mid = 0.5 * (ref1_bounds[l1_ci] + ref1_bounds[l1_ci + 1])

            for q0 in range(q):
                x_1d = lo0 + (hi0 - lo0) * lagrange_nodes[q0]

                nb1, ovf = _adapt_partition_2d(
                    coeffs_2d,
                    masks_2d,
                    k1,
                    x_1d,
                    ref1_bounds,
                    nb1_ref,
                    base_mid,
                    ws1_nodes,
                    ws1_poly1d,
                    ws1_basis,
                    ws1_roots,
                )
                any_overflow |= ovf
                l1_lo = ws1_nodes[l1_ci]
                l1_hi = ws1_nodes[l1_ci + 1]

                for q1 in range(q):
                    x_2d = l1_lo + (l1_hi - l1_lo) * lagrange_nodes[q1]

                    if tang1_2d == 0:
                        x_base_3d[0] = x_1d
                        x_base_3d[1] = x_2d
                        ref_x_3d[0] = base_mid
                        ref_x_3d[1] = l1_mid
                    else:
                        x_base_3d[0] = x_2d
                        x_base_3d[1] = x_1d
                        ref_x_3d[0] = l1_mid
                        ref_x_3d[1] = base_mid

                    local_ref_nb = combo_ref2_nb[ci]
                    nb2, ovf = _adapt_partition_3d(
                        coeffs_3d,
                        masks_3d,
                        k2,
                        x_base_3d,
                        combo_ref2_bounds[ci],
                        local_ref_nb,
                        ref_x_3d,
                        ws2_nodes,
                        ws2_poly1d,
                        ws2_basis0,
                        ws2_basis1,
                        ws2_roots,
                    )
                    any_overflow |= ovf

                    x_surface = ws2_nodes[trans_idx]

                    if n_pts >= len(points):
                        new_cap = len(points) * 2
                        new_p = np.empty((new_cap, 3), dtype=np.float64)
                        for c_i in range(n_pts):
                            for di in range(3):
                                new_p[c_i, di] = points[c_i, di]
                        points = new_p

                    points[n_pts, axis_tang1_3d] = x_1d
                    points[n_pts, axis_k1_3d] = x_2d
                    points[n_pts, k2] = x_surface
                    n_pts += 1

            n_cells += 1

    return points[:n_pts].copy(), n_cells, any_overflow
