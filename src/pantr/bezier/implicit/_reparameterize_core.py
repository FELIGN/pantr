"""Reparameterization phase for implicit domain visualization.

Given a pre-built dimension-reduction hierarchy (from the build phase), generate
structured Lagrange cells that tile the implicit domain (volume) or its zero-set
(surface).  Uses equispaced or Gauss-Lobatto-Legendre nodes instead of
Gauss-Legendre / tanh-sinh, and returns cell connectivity rather than
flat (points, weights) arrays.

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
    _collapse_2d,
    _collapse_3d,
    _eval_bernstein_2d,
    _eval_bernstein_3d,
)
from pantr.bezier.implicit._construct_core import (
    _collect_and_partition_from_2d_into,
    _collect_and_partition_from_3d_into,
)
from pantr.bezier.implicit._mask_core import (
    _line_intersects_2d,
    _line_intersects_3d,
)
from pantr.bezier.implicit._roots_core import find_roots

_REPARAM_OFFSET: float = 1e-10
"""Relative inset from interval boundaries when placing Lagrange nodes.

Prevents degenerate cells when a levelset intersects the ``[0, 1]^d``
boundary, causing roots to coincide with endpoints.
"""

_REPARAM_MIN_LEN: float = 1e-12
"""Minimum interval length.  Intervals shorter than this are skipped."""

_MAX_REF_CELLS: int = 200
"""Maximum reference cells per base interval (generous pre-allocation)."""


# ---------------------------------------------------------------------------
# Section A: Helpers
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _find_containing_interval_idx(
    bounds: npt.NDArray[np.float64],
    count: int,
    target: float,
) -> int:
    """Find the interval in *bounds* containing *target*.

    Args:
        bounds: Sorted boundary array with *count* valid entries.
        count: Number of valid entries.
        target: Value to locate.

    Returns:
        int: Index ``j`` such that ``bounds[j] <= target <= bounds[j+1]``.
            Falls back to the closest endpoint interval if not found.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    for j in range(count - 1):
        if bounds[j] <= target <= bounds[j + 1]:
            return j
    # Fallback: clamp to valid range.
    if target <= bounds[0]:
        return 0
    return count - 2


@nb_jit(nopython=True, cache=True)
def _find_closest_root(
    roots: npt.NDArray[np.float64],
    n_roots: int,
    target: float,
) -> tuple[float, bool]:
    """Find the root closest to *target*.

    Args:
        roots: Array of root values.
        n_roots: Number of valid entries.
        target: Target value to match.

    Returns:
        tuple[float, bool]: ``(root, found)`` where *found* is False if
            ``n_roots == 0``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    if n_roots == 0:
        return target, False
    best = roots[0]
    best_dist = abs(roots[0] - target)
    for i in range(1, n_roots):
        d = abs(roots[i] - target)
        if d < best_dist:
            best_dist = d
            best = roots[i]
    return best, True


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
def _scale_nodes(
    ref_nodes: npt.NDArray[np.float64],
    lo: float,
    hi: float,
    offset: float,
    out: npt.NDArray[np.float64],
) -> None:
    """Map reference nodes from [0,1] to [lo+d, hi-d] with inset d.

    Args:
        ref_nodes: 1D reference nodes on ``[0, 1]``.
        lo: Left endpoint of the target interval.
        hi: Right endpoint of the target interval.
        offset: Relative inset fraction.
        out: Pre-allocated output array (same length as *ref_nodes*).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    span = hi - lo
    delta = offset * span
    lo_in = lo + delta
    hi_in = hi - delta
    scale = hi_in - lo_in
    for i in range(len(ref_nodes)):
        out[i] = lo_in + ref_nodes[i] * scale


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
    offset: float,
    min_len: float,
) -> tuple[npt.NDArray[np.float64], int, bool]:
    """Generate volume Lagrange quad cells for 2D implicit domains.

    Walks the hierarchy bottom-up (same as ``volume_quad_2d``), using
    Lagrange nodes instead of quadrature nodes and filtering intervals
    by sign condition.

    Args:
        base_bounds: Pre-computed sorted 1D base partition.
        base_nb: Number of entries in *base_bounds*.
        k0: Height direction at base level.
        use_ts_0: (Unused, kept for signature consistency.)
        type_0: (Unused, kept for signature consistency.)
        coeffs_2d: 2D polynomial coefficients (original inputs).
        masks_2d: 2D masks (original inputs).
        k1: Height direction at top level.
        use_ts_1: (Unused.)
        type_1: (Unused.)
        lagrange_nodes: 1D Lagrange nodes on ``[0, 1]``, shape ``(q,)``.
        signs: Sign condition per polynomial (``+1``/``-1``/``0``).
        n_polys: Number of input polynomials.
        offset: Relative inset from interval boundaries.
        min_len: Minimum interval length.

    Returns:
        tuple: ``(points, n_cells, any_overflow)`` with points shape
            ``(n_cells * q * q, 2)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    q = len(lagrange_nodes)
    tang1 = 1 - k1

    # No interfaces: full tensor-product cell on [0,1]^2 if sign passes.
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

    # Pre-allocate workspace for inner partition calls.
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

    any_overflow = False

    # Pre-allocate output and temp arrays.
    max_total = (base_nb - 1) * 20 * q * q
    max_total = max(max_total, q * q)
    points = np.empty((max_total, 2), dtype=np.float64)
    n_pts = 0
    n_cells = 0

    tang_buf = np.empty(q, dtype=np.float64)
    height_buf = np.empty(q, dtype=np.float64)
    pt = np.empty(2, dtype=np.float64)

    # Reference topology buffers.
    ref_mids = np.empty(_MAX_REF_CELLS, dtype=np.float64)

    for b_idx in range(base_nb - 1):
        lo_base = base_bounds[b_idx]
        hi_base = base_bounds[b_idx + 1]
        if hi_base - lo_base < min_len:
            continue

        # --- Phase A: Determine reference topology at midpoint ---
        x_mid = 0.5 * (lo_base + hi_base)
        nb_ref, ovf_ref = _collect_and_partition_from_2d_into(
            coeffs_2d,
            masks_2d,
            k1,
            x_mid,
            ws_nodes,
            ws_poly1d,
            ws_basis,
            ws_roots,
        )
        any_overflow |= ovf_ref

        n_inside = 0
        for ib in range(nb_ref - 1):
            ilo = ws_nodes[ib]
            ihi = ws_nodes[ib + 1]
            if ihi - ilo < min_len:
                continue
            mid_h = 0.5 * (ilo + ihi)
            pt[tang1] = x_mid
            pt[k1] = mid_h
            if _check_signs_2d(coeffs_2d, pt, signs, n_polys) and n_inside < _MAX_REF_CELLS:
                ref_mids[n_inside] = mid_h
                n_inside += 1

        if n_inside == 0:
            continue

        # --- Phase B: Generate cells ---
        _scale_nodes(lagrange_nodes, lo_base, hi_base, offset, tang_buf)

        for ci in range(n_inside):
            ref_mid = ref_mids[ci]

            for qi in range(q):
                x_tang = tang_buf[qi]

                nb_inner, ovf_inner = _collect_and_partition_from_2d_into(
                    coeffs_2d,
                    masks_2d,
                    k1,
                    x_tang,
                    ws_nodes,
                    ws_poly1d,
                    ws_basis,
                    ws_roots,
                )
                any_overflow |= ovf_inner

                j = _find_containing_interval_idx(ws_nodes, nb_inner, ref_mid)
                ilo = ws_nodes[j]
                ihi = ws_nodes[j + 1]
                _scale_nodes(lagrange_nodes, ilo, ihi, offset, height_buf)

                for qj in range(q):
                    # Resize if needed.
                    if n_pts >= len(points):
                        new_cap = len(points) * 2
                        new_p = np.empty((new_cap, 2), dtype=np.float64)
                        for c_i in range(n_pts):
                            new_p[c_i, 0] = points[c_i, 0]
                            new_p[c_i, 1] = points[c_i, 1]
                        points = new_p

                    points[n_pts, tang1] = x_tang
                    points[n_pts, k1] = height_buf[qj]
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
    offset: float,
    min_len: float,
) -> tuple[npt.NDArray[np.float64], int, bool]:
    """Generate volume Lagrange hex cells for 3D implicit domains.

    Three-level hierarchy walk: 1D base -> 2D -> 3D.

    Args:
        base_bounds: Pre-computed sorted 1D base partition.
        base_nb: Number of entries in *base_bounds*.
        k0: Height direction at base level.
        use_ts_0: (Unused.)
        type_0: (Unused.)
        coeffs_2d: 2D polynomial coefficients (middle level, derived).
        masks_2d: 2D masks (middle level).
        k1: Height direction at middle level.
        use_ts_1: (Unused.)
        type_1: (Unused.)
        coeffs_3d: 3D polynomial coefficients (original inputs).
        masks_3d: 3D masks (original inputs).
        k2: Height direction at top level.
        use_ts_2: (Unused.)
        type_2: (Unused.)
        lagrange_nodes: 1D Lagrange nodes on ``[0, 1]``, shape ``(q,)``.
        signs: Sign condition per polynomial.
        n_polys: Number of input polynomials.
        offset: Relative inset from interval boundaries.
        min_len: Minimum interval length.

    Returns:
        tuple: ``(points, n_cells, any_overflow)`` with points shape
            ``(n_cells * q^3, 3)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    q = len(lagrange_nodes)

    # No interfaces: full TP cell.
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

    any_overflow = False

    # Output buffers.
    q3 = q * q * q
    max_total = q3 * 8
    points = np.empty((max_total, 3), dtype=np.float64)
    n_pts = 0
    n_cells = 0

    x_base_3d = np.empty(2, dtype=np.float64)
    pt = np.empty(3, dtype=np.float64)
    base_buf = np.empty(q, dtype=np.float64)
    l1_buf = np.empty(q, dtype=np.float64)
    l2_buf = np.empty(q, dtype=np.float64)

    ref_l1_mids = np.empty(_MAX_REF_CELLS, dtype=np.float64)
    ref_l2_mids = np.empty(_MAX_REF_CELLS, dtype=np.float64)

    for b0 in range(base_nb - 1):
        lo0 = base_bounds[b0]
        hi0 = base_bounds[b0 + 1]
        if hi0 - lo0 < min_len:
            continue

        base_mid = 0.5 * (lo0 + hi0)

        # --- Phase A: Reference topology at base midpoint ---
        nb1_ref, ovf1 = _collect_and_partition_from_2d_into(
            coeffs_2d,
            masks_2d,
            k1,
            base_mid,
            ws1_nodes,
            ws1_poly1d,
            ws1_basis,
            ws1_roots,
        )
        any_overflow |= ovf1

        n_inside = 0
        for b1 in range(nb1_ref - 1):
            l1_lo = ws1_nodes[b1]
            l1_hi = ws1_nodes[b1 + 1]
            if l1_hi - l1_lo < min_len:
                continue
            l1_mid = 0.5 * (l1_lo + l1_hi)

            if tang1_2d == 0:
                x_base_3d[0] = base_mid
                x_base_3d[1] = l1_mid
            else:
                x_base_3d[0] = l1_mid
                x_base_3d[1] = base_mid

            nb2_ref, ovf2 = _collect_and_partition_from_3d_into(
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
            any_overflow |= ovf2

            for b2 in range(nb2_ref - 1):
                l2_lo = ws2_nodes[b2]
                l2_hi = ws2_nodes[b2 + 1]
                if l2_hi - l2_lo < min_len:
                    continue
                l2_mid = 0.5 * (l2_lo + l2_hi)

                pt[axis_tang1_3d] = base_mid
                pt[axis_k1_3d] = l1_mid
                pt[k2] = l2_mid
                if _check_signs_3d(coeffs_3d, pt, signs, n_polys) and n_inside < _MAX_REF_CELLS:
                    ref_l1_mids[n_inside] = l1_mid
                    ref_l2_mids[n_inside] = l2_mid
                    n_inside += 1

        if n_inside == 0:
            continue

        # --- Phase B: Generate cells ---
        _scale_nodes(lagrange_nodes, lo0, hi0, offset, base_buf)

        for ci in range(n_inside):
            ref_l1 = ref_l1_mids[ci]
            ref_l2 = ref_l2_mids[ci]

            for q0 in range(q):
                x_1d = base_buf[q0]

                nb1, ovf1 = _collect_and_partition_from_2d_into(
                    coeffs_2d,
                    masks_2d,
                    k1,
                    x_1d,
                    ws1_nodes,
                    ws1_poly1d,
                    ws1_basis,
                    ws1_roots,
                )
                any_overflow |= ovf1

                j1 = _find_containing_interval_idx(ws1_nodes, nb1, ref_l1)
                l1_lo = ws1_nodes[j1]
                l1_hi = ws1_nodes[j1 + 1]
                _scale_nodes(lagrange_nodes, l1_lo, l1_hi, offset, l1_buf)

                for q1 in range(q):
                    x_2d = l1_buf[q1]

                    if tang1_2d == 0:
                        x_base_3d[0] = x_1d
                        x_base_3d[1] = x_2d
                    else:
                        x_base_3d[0] = x_2d
                        x_base_3d[1] = x_1d

                    nb2, ovf2 = _collect_and_partition_from_3d_into(
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
                    any_overflow |= ovf2

                    j2 = _find_containing_interval_idx(ws2_nodes, nb2, ref_l2)
                    l2_lo = ws2_nodes[j2]
                    l2_hi = ws2_nodes[j2 + 1]
                    _scale_nodes(lagrange_nodes, l2_lo, l2_hi, offset, l2_buf)

                    for q2 in range(q):
                        if n_pts >= len(points):
                            new_cap = len(points) * 2
                            new_p = np.empty((new_cap, 3), dtype=np.float64)
                            for c_i in range(n_pts):
                                for di in range(3):
                                    new_p[c_i, di] = points[c_i, di]
                            points = new_p

                        points[n_pts, axis_tang1_3d] = x_1d
                        points[n_pts, axis_k1_3d] = x_2d
                        points[n_pts, k2] = l2_buf[q2]
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
    offset: float,
    min_len: float,
) -> tuple[npt.NDArray[np.float64], int, bool]:
    """Generate surface Lagrange curve cells for 2D levelsets.

    Traces roots of ``coeffs_2d[poly_idx]`` along the tangential direction,
    producing one Lagrange curve segment per (base interval, root) pair.

    Args:
        base_bounds: Pre-computed sorted 1D base partition.
        base_nb: Number of entries in *base_bounds*.
        k0: Height direction at base level.
        use_ts_0: (Unused.)
        type_0: (Unused.)
        coeffs_2d: 2D polynomial coefficients (original inputs).
        masks_2d: 2D masks (original inputs).
        k1: Height direction at top level.
        use_ts_1: (Unused.)
        type_1: (Unused.)
        poly_idx: Index of the polynomial whose zero set to trace.
        lagrange_nodes: 1D Lagrange nodes on ``[0, 1]``, shape ``(q,)``.
        signs: Sign condition per polynomial (``signs[poly_idx]`` is ignored).
        n_polys: Number of input polynomials.
        offset: Relative inset from interval boundaries.
        min_len: Minimum interval length.

    Returns:
        tuple: ``(points, n_cells, any_overflow)`` with points shape
            ``(n_cells * q, 2)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    q = len(lagrange_nodes)
    tang1 = 1 - k1

    # No interfaces: empty surface.
    if k1 >= 2:  # noqa: PLR2004
        return np.empty((0, 2), dtype=np.float64), 0, False

    any_overflow = False

    max_total = q * (base_nb - 1) * 10
    max_total = max(max_total, q)
    points = np.empty((max_total, 2), dtype=np.float64)
    n_pts = 0
    n_cells = 0

    tang_buf = np.empty(q, dtype=np.float64)
    pt = np.empty(2, dtype=np.float64)
    ref_roots_buf = np.empty(_MAX_REF_CELLS, dtype=np.float64)

    target_poly = coeffs_2d[poly_idx]

    for b_idx in range(base_nb - 1):
        lo_base = base_bounds[b_idx]
        hi_base = base_bounds[b_idx + 1]
        if hi_base - lo_base < min_len:
            continue

        # --- Phase A: Reference roots at midpoint ---
        x_mid = 0.5 * (lo_base + hi_base)

        if not _line_intersects_2d(masks_2d[poly_idx], x_mid, k1):
            continue

        poly_1d = _collapse_2d(target_poly, k1, x_mid)
        raw_roots, n_raw, ovf = find_roots(poly_1d)
        any_overflow |= ovf

        n_ref = 0
        for ri in range(n_raw):
            root = raw_roots[ri]
            pt[tang1] = x_mid
            pt[k1] = root
            if _check_signs_2d(coeffs_2d, pt, signs, n_polys) and n_ref < _MAX_REF_CELLS:
                ref_roots_buf[n_ref] = root
                n_ref += 1

        if n_ref == 0:
            continue

        # --- Phase B: Trace each root across tangential nodes ---
        _scale_nodes(lagrange_nodes, lo_base, hi_base, offset, tang_buf)

        for ri in range(n_ref):
            ref_root = ref_roots_buf[ri]

            for qi in range(q):
                x_tang = tang_buf[qi]

                if not _line_intersects_2d(masks_2d[poly_idx], x_tang, k1):
                    # Fallback: use reference root.
                    matched = ref_root
                else:
                    p1d = _collapse_2d(target_poly, k1, x_tang)
                    roots, n_roots, ovf = find_roots(p1d)
                    any_overflow |= ovf
                    matched, _found = _find_closest_root(roots, n_roots, ref_root)

                if n_pts >= len(points):
                    new_cap = len(points) * 2
                    new_p = np.empty((new_cap, 2), dtype=np.float64)
                    for c_i in range(n_pts):
                        new_p[c_i, 0] = points[c_i, 0]
                        new_p[c_i, 1] = points[c_i, 1]
                    points = new_p

                points[n_pts, tang1] = x_tang
                points[n_pts, k1] = matched
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
    offset: float,
    min_len: float,
) -> tuple[npt.NDArray[np.float64], int, bool]:
    """Generate surface Lagrange quad cells for 3D levelsets.

    Walks the hierarchy (base -> level 1), at each ``(q0, q1)`` position
    collapses the target polynomial along *k2* and finds roots.  Each
    traced root produces one ``q x q`` Lagrange quad.

    Args:
        base_bounds: Pre-computed sorted 1D base partition.
        base_nb: Number of entries in *base_bounds*.
        k0: Height direction at base level.
        use_ts_0: (Unused.)
        type_0: (Unused.)
        coeffs_2d: 2D polynomial coefficients (middle level, derived).
        masks_2d: 2D masks (middle level).
        k1: Height direction at middle level.
        use_ts_1: (Unused.)
        type_1: (Unused.)
        coeffs_3d: 3D polynomial coefficients (original inputs).
        masks_3d: 3D masks (original inputs).
        k2: Height direction at top level.
        use_ts_2: (Unused.)
        type_2: (Unused.)
        poly_idx: Index of the polynomial whose zero set to trace.
        lagrange_nodes: 1D Lagrange nodes on ``[0, 1]``, shape ``(q,)``.
        signs: Sign condition per polynomial (``signs[poly_idx]`` ignored).
        n_polys: Number of input polynomials.
        offset: Relative inset from interval boundaries.
        min_len: Minimum interval length.

    Returns:
        tuple: ``(points, n_cells, any_overflow)`` with points shape
            ``(n_cells * q * q, 3)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    q = len(lagrange_nodes)

    if k2 >= 3:  # noqa: PLR2004
        return np.empty((0, 3), dtype=np.float64), 0, False

    # Axis mapping.
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

    any_overflow = False

    max_total = q * q * (base_nb - 1) * 20
    max_total = max(max_total, q * q)
    points = np.empty((max_total, 3), dtype=np.float64)
    n_pts = 0
    n_cells = 0

    x_base_3d = np.empty(2, dtype=np.float64)
    pt = np.empty(3, dtype=np.float64)
    base_buf = np.empty(q, dtype=np.float64)
    l1_buf = np.empty(q, dtype=np.float64)

    ref_l1_mids = np.empty(_MAX_REF_CELLS, dtype=np.float64)
    ref_roots = np.empty(_MAX_REF_CELLS, dtype=np.float64)

    target_poly = coeffs_3d[poly_idx]

    for b0 in range(base_nb - 1):
        lo0 = base_bounds[b0]
        hi0 = base_bounds[b0 + 1]
        if hi0 - lo0 < min_len:
            continue

        base_mid = 0.5 * (lo0 + hi0)

        # --- Phase A: Reference topology ---
        nb1_ref, ovf1 = _collect_and_partition_from_2d_into(
            coeffs_2d,
            masks_2d,
            k1,
            base_mid,
            ws1_nodes,
            ws1_poly1d,
            ws1_basis,
            ws1_roots,
        )
        any_overflow |= ovf1

        n_ref = 0
        for b1 in range(nb1_ref - 1):
            l1_lo = ws1_nodes[b1]
            l1_hi = ws1_nodes[b1 + 1]
            if l1_hi - l1_lo < min_len:
                continue
            l1_mid = 0.5 * (l1_lo + l1_hi)

            if tang1_2d == 0:
                x_base_3d[0] = base_mid
                x_base_3d[1] = l1_mid
            else:
                x_base_3d[0] = l1_mid
                x_base_3d[1] = base_mid

            if not _line_intersects_3d(masks_3d[poly_idx], x_base_3d, k2):
                continue

            poly_1d = _collapse_3d(target_poly, k2, x_base_3d)
            raw_roots, n_raw, ovf = find_roots(poly_1d)
            any_overflow |= ovf

            for ri in range(n_raw):
                root = raw_roots[ri]
                pt[axis_tang1_3d] = base_mid
                pt[axis_k1_3d] = l1_mid
                pt[k2] = root
                if _check_signs_3d(coeffs_3d, pt, signs, n_polys) and n_ref < _MAX_REF_CELLS:
                    ref_l1_mids[n_ref] = l1_mid
                    ref_roots[n_ref] = root
                    n_ref += 1

        if n_ref == 0:
            continue

        # --- Phase B: Generate cells ---
        _scale_nodes(lagrange_nodes, lo0, hi0, offset, base_buf)

        for ci in range(n_ref):
            ref_l1 = ref_l1_mids[ci]
            ref_root = ref_roots[ci]

            for q0 in range(q):
                x_1d = base_buf[q0]

                nb1, ovf1 = _collect_and_partition_from_2d_into(
                    coeffs_2d,
                    masks_2d,
                    k1,
                    x_1d,
                    ws1_nodes,
                    ws1_poly1d,
                    ws1_basis,
                    ws1_roots,
                )
                any_overflow |= ovf1

                j1 = _find_containing_interval_idx(ws1_nodes, nb1, ref_l1)
                l1_lo = ws1_nodes[j1]
                l1_hi = ws1_nodes[j1 + 1]
                _scale_nodes(lagrange_nodes, l1_lo, l1_hi, offset, l1_buf)

                for q1 in range(q):
                    x_2d = l1_buf[q1]

                    if tang1_2d == 0:
                        x_base_3d[0] = x_1d
                        x_base_3d[1] = x_2d
                    else:
                        x_base_3d[0] = x_2d
                        x_base_3d[1] = x_1d

                    if not _line_intersects_3d(masks_3d[poly_idx], x_base_3d, k2):
                        matched = ref_root
                    else:
                        p1d = _collapse_3d(target_poly, k2, x_base_3d)
                        roots, n_roots, ovf = find_roots(p1d)
                        any_overflow |= ovf
                        matched, _found = _find_closest_root(
                            roots,
                            n_roots,
                            ref_root,
                        )

                    if n_pts >= len(points):
                        new_cap = len(points) * 2
                        new_p = np.empty((new_cap, 3), dtype=np.float64)
                        for c_i in range(n_pts):
                            for di in range(3):
                                new_p[c_i, di] = points[c_i, di]
                        points = new_p

                    points[n_pts, axis_tang1_3d] = x_1d
                    points[n_pts, axis_k1_3d] = x_2d
                    points[n_pts, k2] = matched
                    n_pts += 1

            n_cells += 1

    return points[:n_pts].copy(), n_cells, any_overflow
