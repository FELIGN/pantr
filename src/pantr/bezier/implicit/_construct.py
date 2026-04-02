"""Construction phase for the implicit quadrature algorithm.

Implements Algorithm 2 from Saye (2022): given a pre-built hierarchy, walk
it bottom-up to generate quadrature points and weights. At each level,
collapse polynomials to 1D along the height direction, find roots, partition
[0,1] into sub-intervals, and apply 1D quadrature (Gauss-Legendre or
tanh-sinh) on each sub-interval.

Main exports:

- :func:`volume_quad_2d` -- volume quadrature for 2D.
- :func:`volume_quad_3d` -- volume quadrature for 3D.
- :func:`surface_quad_2d` -- surface quadrature for 2D.

Note:
    Inputs are assumed to be correct (no validation performed).
    These are Layer 3 kernels for the implicit quadrature module.
"""

from __future__ import annotations

import numpy as np
from numba.typed import List as NumbaList
from numpy import typing as npt

from pantr._numba_compat import nb_jit
from pantr.bezier.implicit._bernstein import (
    _collapse_2d,
    _collapse_3d,
    _eval_gradient_2d,
    _eval_gradient_3d,
)
from pantr.bezier.implicit._mask import (
    _line_intersects_2d,
    _line_intersects_3d,
    _point_within_1d,
    _point_within_2d,
)
from pantr.bezier.implicit._roots import find_roots

_MERGE_TOL: float = 10.0 * 2.2204460492503131e-16
"""Tolerance for merging nearby roots with interval boundaries."""


# ---------------------------------------------------------------------------
# Section A: Root collection and interval partitioning
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def _collect_and_partition_1d(
    coeffs_list: NumbaList,
    masks_list: NumbaList,
) -> tuple[npt.NDArray[np.float64], int]:
    """Collect roots from all 1D polynomials and partition [0, 1].

    For the 1D base case: finds roots of each polynomial, filters through
    masks, merges with boundaries {0, 1}, sorts and deduplicates.

    Args:
        coeffs_list (NumbaList): List of 1D coefficient arrays.
        masks_list (NumbaList): List of 1D boolean mask arrays.

    Returns:
        tuple[npt.NDArray[np.float64], int]: (boundaries, count) where
            boundaries are sorted and include 0 and 1.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    # Estimate max roots: sum of degrees + 2 for boundaries.
    max_roots = 2
    for i in range(len(coeffs_list)):
        max_roots += len(coeffs_list[i]) - 1

    nodes = np.empty(max_roots, dtype=np.float64)
    nodes[0] = 0.0
    nodes[1] = 1.0
    count = 2

    for i in range(len(coeffs_list)):
        roots, n_roots = find_roots(coeffs_list[i])
        for r in range(n_roots):
            root = roots[r]
            # Filter through mask.
            if not _point_within_1d(masks_list[i], root):
                continue
            # Check not too close to existing nodes.
            too_close = False
            for j in range(count):
                if abs(root - nodes[j]) < _MERGE_TOL:
                    too_close = True
                    break
            if not too_close:
                nodes[count] = root
                count += 1

    # Sort.
    for i in range(count - 1):
        for j in range(i + 1, count):
            if nodes[j] < nodes[i]:
                nodes[i], nodes[j] = nodes[j], nodes[i]

    return nodes, count


@nb_jit(nopython=True, cache=True)
def _collect_and_partition_from_2d(
    coeffs_list: NumbaList,
    masks_list: NumbaList,
    k: int,
    x_base: float,
) -> tuple[npt.NDArray[np.float64], int]:
    """Collapse 2D polynomials to 1D at *x_base*, find roots, partition [0,1].

    Args:
        coeffs_list (NumbaList): List of 2D coefficient arrays.
        masks_list (NumbaList): List of 2D boolean mask arrays.
        k (int): Height direction.
        x_base (float): Base point in tangential direction.

    Returns:
        tuple[npt.NDArray[np.float64], int]: (boundaries, count).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    max_roots = 2
    for i in range(len(coeffs_list)):
        max_roots += coeffs_list[i].shape[k] - 1

    nodes = np.empty(max_roots, dtype=np.float64)
    nodes[0] = 0.0
    nodes[1] = 1.0
    count = 2

    for i in range(len(coeffs_list)):
        # Check if line intersects mask.
        if not _line_intersects_2d(masks_list[i], x_base, k):
            continue

        # Collapse to 1D along k.
        poly_1d = _collapse_2d(coeffs_list[i], k, x_base)
        roots, n_roots = find_roots(poly_1d)

        for r in range(n_roots):
            root = roots[r]
            # Build 2D point for mask check.
            pt = np.empty(2, dtype=np.float64)
            if k == 0:
                pt[0] = root
                pt[1] = x_base
            else:
                pt[0] = x_base
                pt[1] = root
            if not _point_within_2d(masks_list[i], pt):
                continue
            # Check not too close to existing nodes.
            too_close = False
            for j in range(count):
                if abs(root - nodes[j]) < _MERGE_TOL:
                    too_close = True
                    break
            if not too_close:
                nodes[count] = root
                count += 1

    # Sort.
    for i in range(count - 1):
        for j in range(i + 1, count):
            if nodes[j] < nodes[i]:
                nodes[i], nodes[j] = nodes[j], nodes[i]

    return nodes, count


@nb_jit(nopython=True, cache=True)
def _collect_and_partition_from_3d(
    coeffs_list: NumbaList,
    masks_list: NumbaList,
    k: int,
    x_base: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], int]:
    """Collapse 3D polynomials to 1D at *x_base*, find roots, partition [0,1].

    Args:
        coeffs_list (NumbaList): List of 3D coefficient arrays.
        masks_list (NumbaList): List of 3D boolean mask arrays.
        k (int): Height direction.
        x_base (npt.NDArray[np.float64]): Base point of shape ``(2,)``
            in tangential directions.

    Returns:
        tuple[npt.NDArray[np.float64], int]: (boundaries, count).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    max_roots = 2
    for i in range(len(coeffs_list)):
        max_roots += coeffs_list[i].shape[k] - 1

    nodes = np.empty(max_roots, dtype=np.float64)
    nodes[0] = 0.0
    nodes[1] = 1.0
    count = 2

    for i in range(len(coeffs_list)):
        if not _line_intersects_3d(masks_list[i], x_base, k):
            continue

        poly_1d = _collapse_3d(coeffs_list[i], k, x_base)
        roots, n_roots = find_roots(poly_1d)

        for r in range(n_roots):
            root = roots[r]
            too_close = False
            for j in range(count):
                if abs(root - nodes[j]) < _MERGE_TOL:
                    too_close = True
                    break
            if not too_close:
                nodes[count] = root
                count += 1

    # Sort.
    for i in range(count - 1):
        for j in range(i + 1, count):
            if nodes[j] < nodes[i]:
                nodes[i], nodes[j] = nodes[j], nodes[i]

    return nodes, count


# ---------------------------------------------------------------------------
# Section B: Volume quadrature
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def volume_quad_2d(
    coeffs_1d: NumbaList,
    masks_1d: NumbaList,
    k0: int,
    use_ts_0: bool,
    type_0: int,
    coeffs_2d: NumbaList,
    masks_2d: NumbaList,
    k1: int,
    use_ts_1: bool,
    type_1: int,
    gl_nodes: npt.NDArray[np.float64],
    gl_weights: npt.NDArray[np.float64],
    ts_nodes: npt.NDArray[np.float64],
    ts_weights: npt.NDArray[np.float64],
    strategy: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate volume quadrature points and weights for 2D.

    Walks the hierarchy bottom-up:
    1. Base (1D): partition [0,1] by roots of base polynomials, apply quad.
    2. Level 1 (2D): for each base point, collapse 2D polys to 1D along k1,
       partition [0,1] by roots, apply quad, combine coordinates/weights.

    Args:
        coeffs_1d (NumbaList): 1D polynomial coefficients (base level).
        masks_1d (NumbaList): 1D masks (base level).
        k0 (int): Height direction at base level (always 0 for 1D).
        use_ts_0 (bool): Use tanh-sinh at base level.
        type_0 (int): Integral type at base level.
        coeffs_2d (NumbaList): 2D polynomial coefficients (top level).
        masks_2d (NumbaList): 2D masks (top level).
        k1 (int): Height direction at top level.
        use_ts_1 (bool): Use tanh-sinh at top level.
        type_1 (int): Integral type at top level.
        gl_nodes (npt.NDArray[np.float64]): GL quadrature nodes on [0, 1].
        gl_weights (npt.NDArray[np.float64]): GL quadrature weights on [0, 1].
        ts_nodes (npt.NDArray[np.float64]): Tanh-sinh nodes on [0, 1].
        ts_weights (npt.NDArray[np.float64]): Tanh-sinh weights on [0, 1].
        strategy (int): 0=GL only, 1=TS only, 2=auto mixed.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            (points, weights) with shapes ``(n_pts, 2)`` and ``(n_pts,)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    q = len(gl_nodes)

    # If k1 >= 2, no interfaces -> tensor product quadrature on [0,1]^2.
    if k1 >= 2:
        n_total = q * q
        points = np.empty((n_total, 2), dtype=np.float64)
        weights = np.empty(n_total, dtype=np.float64)
        idx = 0
        for i in range(q):
            for j in range(q):
                points[idx, 0] = gl_nodes[i]
                points[idx, 1] = gl_nodes[j]
                weights[idx] = gl_weights[i] * gl_weights[j]
                idx += 1
        return points, weights

    # Select quadrature rule for each level.
    nodes_outer = gl_nodes
    wts_outer = gl_weights
    nodes_inner = gl_nodes
    wts_inner = gl_weights
    if strategy == 1:
        nodes_outer = ts_nodes
        wts_outer = ts_weights
        nodes_inner = ts_nodes
        wts_inner = ts_weights
    elif strategy == 2:
        if use_ts_1:
            nodes_outer = ts_nodes
            wts_outer = ts_weights

    # Tangential direction index for level 1.
    tang1 = 1 - k1

    # Phase 1: Partition [0,1] by base (1D) polynomials.
    boundaries, n_bounds = _collect_and_partition_1d(coeffs_1d, masks_1d)

    # Estimate max output size.
    max_intervals_base = n_bounds - 1
    max_pts_per_base = q
    # For each base point, we get up to max_intervals_2d * q inner points.
    max_inner_per_base = 20 * q  # generous estimate
    max_total = max_intervals_base * max_pts_per_base * max_inner_per_base
    max_total = max(max_total, 1)

    points = np.empty((max_total, 2), dtype=np.float64)
    weights = np.empty(max_total, dtype=np.float64)
    n_pts = 0

    # Phase 2: Walk hierarchy.
    for b_idx in range(n_bounds - 1):
        lo = boundaries[b_idx]
        hi = boundaries[b_idx + 1]
        if hi - lo < _MERGE_TOL:
            continue
        scale_outer = hi - lo

        for qi in range(q):
            # Map outer quadrature node to [lo, hi].
            x_tang_val = lo + nodes_outer[qi] * scale_outer
            w_tang = wts_outer[qi] * scale_outer

            # Phase 3: Collapse 2D polys to 1D along k1 at x_tang_val.
            inner_bounds, n_inner = _collect_and_partition_from_2d(
                coeffs_2d, masks_2d, k1, x_tang_val
            )

            for ib in range(n_inner - 1):
                ilo = inner_bounds[ib]
                ihi = inner_bounds[ib + 1]
                if ihi - ilo < _MERGE_TOL:
                    continue
                scale_inner = ihi - ilo

                for qj in range(q):
                    x_height = ilo + nodes_inner[qj] * scale_inner
                    w_height = wts_inner[qj] * scale_inner

                    # Resize if needed.
                    if n_pts >= len(weights):
                        new_cap = len(weights) * 2
                        new_pts = np.empty((new_cap, 2), dtype=np.float64)
                        new_wts = np.empty(new_cap, dtype=np.float64)
                        for c_i in range(n_pts):
                            new_pts[c_i, 0] = points[c_i, 0]
                            new_pts[c_i, 1] = points[c_i, 1]
                            new_wts[c_i] = weights[c_i]
                        points = new_pts
                        weights = new_wts

                    # Assemble point: tang1 gets x_tang_val, k1 gets x_height.
                    points[n_pts, tang1] = x_tang_val
                    points[n_pts, k1] = x_height
                    weights[n_pts] = w_tang * w_height
                    n_pts += 1

    return points[:n_pts].copy(), weights[:n_pts].copy()


@nb_jit(nopython=True, cache=True)
def volume_quad_3d(
    coeffs_1d: NumbaList,
    masks_1d: NumbaList,
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
    gl_nodes: npt.NDArray[np.float64],
    gl_weights: npt.NDArray[np.float64],
    ts_nodes: npt.NDArray[np.float64],
    ts_weights: npt.NDArray[np.float64],
    strategy: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate volume quadrature points and weights for 3D.

    Three-level hierarchy walk: 1D base -> 2D -> 3D.

    Args:
        coeffs_1d through type_2: Build result (15 values, 5 per level).
        gl_nodes, gl_weights: GL quadrature on [0, 1].
        ts_nodes, ts_weights: Tanh-sinh quadrature on [0, 1].
        strategy (int): 0=GL only, 1=TS only, 2=auto mixed.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            (points, weights) with shapes ``(n_pts, 3)`` and ``(n_pts,)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    q = len(gl_nodes)

    # No interfaces: TP quadrature.
    if k2 >= 3:
        n_total = q * q * q
        points = np.empty((n_total, 3), dtype=np.float64)
        weights = np.empty(n_total, dtype=np.float64)
        idx = 0
        for i in range(q):
            for j in range(q):
                for l in range(q):
                    points[idx, 0] = gl_nodes[i]
                    points[idx, 1] = gl_nodes[j]
                    points[idx, 2] = gl_nodes[l]
                    weights[idx] = gl_weights[i] * gl_weights[j] * gl_weights[l]
                    idx += 1
        return points, weights

    # Select quadrature rules per level.
    # Level 0 (outermost/base): TS if branching detected (use_ts_1 from 2D level).
    # Level 1 (middle): TS if branching detected (use_ts_2 from 3D level).
    # Level 2 (innermost/height): GL (smooth on each sub-interval).
    nodes_0 = gl_nodes
    wts_0 = gl_weights
    nodes_1 = gl_nodes
    wts_1 = gl_weights
    nodes_2 = gl_nodes
    wts_2 = gl_weights
    if strategy == 1:
        nodes_0 = ts_nodes
        wts_0 = ts_weights
        nodes_1 = ts_nodes
        wts_1 = ts_weights
        nodes_2 = ts_nodes
        wts_2 = ts_weights
    elif strategy == 2:
        if use_ts_1:
            nodes_0 = ts_nodes
            wts_0 = ts_weights
        if use_ts_2:
            nodes_1 = ts_nodes
            wts_1 = ts_weights

    # Determine the 3D tangential directions (the two not equal to k2).
    tang2 = np.empty(2, dtype=np.int64)
    ti = 0
    for d in range(3):
        if d != k2:
            tang2[ti] = d
            ti += 1

    # Determine which of tang2[0], tang2[1] corresponds to the 2D axes 0, 1
    # and which is the k1 direction in the 2D subproblem.
    # The 2D polynomial set lives in the tangential plane of the 3D problem.
    # tang2[0] -> 2D axis 0, tang2[1] -> 2D axis 1.
    # k1 is the height direction in 2D, tang1_2d is the tangential in 2D.
    tang1_2d = 1 - k1  # the 2D tangential axis

    # Map 2D axis indices back to 3D axis indices.
    axis_k1_3d = tang2[k1]  # 3D axis for the 2D height direction
    axis_tang1_3d = tang2[tang1_2d]  # 3D axis for the 2D tangential direction

    # Pre-allocate output.
    max_total = q * q * q * 8  # generous
    points = np.empty((max_total, 3), dtype=np.float64)
    weights = np.empty(max_total, dtype=np.float64)
    n_pts = 0

    # Level 0: 1D base partition.
    bounds_0, nb_0 = _collect_and_partition_1d(coeffs_1d, masks_1d)

    for b0 in range(nb_0 - 1):
        lo0 = bounds_0[b0]
        hi0 = bounds_0[b0 + 1]
        if hi0 - lo0 < _MERGE_TOL:
            continue
        scale0 = hi0 - lo0

        for q0 in range(q):
            x_1d = lo0 + nodes_0[q0] * scale0
            w_1d = wts_0[q0] * scale0

            # Level 1: Collapse 2D polys to 1D along k1 at x_1d.
            bounds_1, nb_1 = _collect_and_partition_from_2d(coeffs_2d, masks_2d, k1, x_1d)

            for b1 in range(nb_1 - 1):
                lo1 = bounds_1[b1]
                hi1 = bounds_1[b1 + 1]
                if hi1 - lo1 < _MERGE_TOL:
                    continue
                scale1 = hi1 - lo1

                for q1 in range(q):
                    x_2d = lo1 + nodes_1[q1] * scale1
                    w_2d = wts_1[q1] * scale1

                    # Level 2: Collapse 3D polys to 1D along k2.
                    # The base point for 3D collapse is a 2D point in the
                    # tangential plane.
                    x_base_3d = np.empty(2, dtype=np.float64)
                    x_base_3d[0] = x_1d  # maps to tang2[tang1_2d] direction...
                    # Actually we need to figure out the ordering.
                    # In the 3D collapse, x_tang has 2 components ordered by
                    # increasing 3D axis index (skipping k2).
                    # tang2[0] corresponds to x_tang[0], tang2[1] to x_tang[1].
                    # x_1d is the coordinate for the 2D tangential direction,
                    # which is tang2[tang1_2d] = axis_tang1_3d.
                    # x_2d is the coordinate for the 2D height direction,
                    # which is tang2[k1] = axis_k1_3d.
                    # In x_tang ordering: tang2[0] and tang2[1].
                    if tang1_2d == 0:
                        # x_tang[0] = x_1d, x_tang[1] = x_2d
                        x_base_3d[0] = x_1d
                        x_base_3d[1] = x_2d
                    else:
                        # x_tang[0] = x_2d, x_tang[1] = x_1d
                        x_base_3d[0] = x_2d
                        x_base_3d[1] = x_1d

                    bounds_2, nb_2 = _collect_and_partition_from_3d(
                        coeffs_3d, masks_3d, k2, x_base_3d
                    )

                    for b2 in range(nb_2 - 1):
                        lo2 = bounds_2[b2]
                        hi2 = bounds_2[b2 + 1]
                        if hi2 - lo2 < _MERGE_TOL:
                            continue
                        scale2 = hi2 - lo2

                        for q2 in range(q):
                            x_3d = lo2 + nodes_2[q2] * scale2
                            w_3d = wts_2[q2] * scale2

                            # Resize if needed.
                            if n_pts >= len(weights):
                                new_cap = len(weights) * 2
                                new_p = np.empty((new_cap, 3), dtype=np.float64)
                                new_w = np.empty(new_cap, dtype=np.float64)
                                for ci in range(n_pts):
                                    new_p[ci, 0] = points[ci, 0]
                                    new_p[ci, 1] = points[ci, 1]
                                    new_p[ci, 2] = points[ci, 2]
                                    new_w[ci] = weights[ci]
                                points = new_p
                                weights = new_w

                            # Assemble 3D point.
                            points[n_pts, axis_tang1_3d] = x_1d
                            points[n_pts, axis_k1_3d] = x_2d
                            points[n_pts, k2] = x_3d
                            weights[n_pts] = w_1d * w_2d * w_3d
                            n_pts += 1

    return points[:n_pts].copy(), weights[:n_pts].copy()


# ---------------------------------------------------------------------------
# Section C: Surface quadrature
# ---------------------------------------------------------------------------


@nb_jit(nopython=True, cache=True)
def surface_quad_2d(
    coeffs_1d: NumbaList,
    masks_1d: NumbaList,
    k0: int,
    use_ts_0: bool,
    type_0: int,
    coeffs_2d: NumbaList,
    masks_2d: NumbaList,
    k1: int,
    use_ts_1: bool,
    type_1: int,
    n_input_polys: int,
    input_coeffs_2d: NumbaList,
    gl_nodes: npt.NDArray[np.float64],
    gl_weights: npt.NDArray[np.float64],
    ts_nodes: npt.NDArray[np.float64],
    ts_weights: npt.NDArray[np.float64],
    strategy: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Generate surface (flux-form) quadrature for 2D.

    The surface integral is computed in flux form: for each point on the
    interface, the weight includes the normal direction. Specifically,
    for each elimination direction k, the algorithm evaluates
    ``f(x) * sign(d_k phi) * w`` at each root.

    Args:
        coeffs_1d through type_1: Build result (10 values).
        n_input_polys (int): Number of original input polynomials.
        input_coeffs_2d (NumbaList): Original 2D polynomial coefficients.
        gl_nodes, gl_weights: GL quadrature on [0, 1].
        ts_nodes, ts_weights: Tanh-sinh quadrature on [0, 1].
        strategy (int): 0=GL only, 1=TS only, 2=auto mixed.

    Returns:
        tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
            (points, scalar_weights, normal_weights) with shapes
            ``(n_pts, 2)``, ``(n_pts,)``, ``(n_pts, 2)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    q = len(gl_nodes)
    tang1 = 1 - k1

    if k1 >= 2:
        # No interface -> empty surface quad.
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
        )

    nodes_outer = gl_nodes
    wts_outer = gl_weights
    if strategy == 1 or (strategy == 2 and use_ts_1):
        nodes_outer = ts_nodes
        wts_outer = ts_weights

    # Partition by base polynomials.
    boundaries, n_bounds = _collect_and_partition_1d(coeffs_1d, masks_1d)

    max_total = q * 100
    points = np.empty((max_total, 2), dtype=np.float64)
    scalar_wts = np.empty(max_total, dtype=np.float64)
    normal_wts = np.empty((max_total, 2), dtype=np.float64)
    n_pts = 0

    for b_idx in range(n_bounds - 1):
        lo = boundaries[b_idx]
        hi = boundaries[b_idx + 1]
        if hi - lo < _MERGE_TOL:
            continue
        scale = hi - lo

        for qi in range(q):
            x_tang_val = lo + nodes_outer[qi] * scale
            w_tang = wts_outer[qi] * scale

            # For each original input polynomial, find roots along k1.
            for p_idx in range(n_input_polys):
                poly_2d = input_coeffs_2d[p_idx]

                if not _line_intersects_2d(masks_2d[p_idx], x_tang_val, k1):
                    continue

                poly_1d = _collapse_2d(poly_2d, k1, x_tang_val)
                roots, n_roots = find_roots(poly_1d)

                for ri in range(n_roots):
                    root = roots[ri]

                    # Build 2D point.
                    pt = np.empty(2, dtype=np.float64)
                    pt[tang1] = x_tang_val
                    pt[k1] = root

                    if not _point_within_2d(masks_2d[p_idx], pt):
                        continue

                    # Compute gradient for surface Jacobian.
                    grad = _eval_gradient_2d(poly_2d, pt)
                    grad_norm = np.sqrt(grad[0] ** 2 + grad[1] ** 2)
                    dk_phi = grad[k1]

                    if abs(dk_phi) < 1e-300:
                        continue

                    # Resize if needed.
                    if n_pts >= len(scalar_wts):
                        new_cap = len(scalar_wts) * 2
                        new_p = np.empty((new_cap, 2), dtype=np.float64)
                        new_s = np.empty(new_cap, dtype=np.float64)
                        new_n = np.empty((new_cap, 2), dtype=np.float64)
                        for ci in range(n_pts):
                            new_p[ci, 0] = points[ci, 0]
                            new_p[ci, 1] = points[ci, 1]
                            new_s[ci] = scalar_wts[ci]
                            new_n[ci, 0] = normal_wts[ci, 0]
                            new_n[ci, 1] = normal_wts[ci, 1]
                        points = new_p
                        scalar_wts = new_s
                        normal_wts = new_n

                    # Flux-form surface weight.
                    alpha = w_tang * grad_norm / abs(dk_phi)
                    points[n_pts, 0] = pt[0]
                    points[n_pts, 1] = pt[1]
                    scalar_wts[n_pts] = alpha
                    # Normal direction (unit normal * alpha).
                    if grad_norm > 1e-300:
                        normal_wts[n_pts, 0] = w_tang * grad[0] / abs(dk_phi)
                        normal_wts[n_pts, 1] = w_tang * grad[1] / abs(dk_phi)
                    else:
                        normal_wts[n_pts, 0] = 0.0
                        normal_wts[n_pts, 1] = 0.0
                    n_pts += 1

    return (
        points[:n_pts].copy(),
        scalar_wts[:n_pts].copy(),
        normal_wts[:n_pts].copy(),
    )


@nb_jit(nopython=True, cache=True)
def surface_quad_3d(  # noqa: PLR0912, PLR0915
    coeffs_1d: NumbaList,
    masks_1d: NumbaList,
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
    n_input_polys: int,
    input_coeffs_3d: NumbaList,
    gl_nodes: npt.NDArray[np.float64],
    gl_weights: npt.NDArray[np.float64],
    ts_nodes: npt.NDArray[np.float64],
    ts_weights: npt.NDArray[np.float64],
    strategy: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Generate surface (flux-form) quadrature for 3D.

    Walks the hierarchy bottom-up (same as volume_quad_3d) but at the
    innermost level evaluates at roots of the input polynomials along k2,
    weighted by the surface Jacobian |grad phi| / |d_k2 phi|.

    Args:
        coeffs_1d through type_2: Build result (15 values, 5 per level).
        n_input_polys (int): Number of original input polynomials.
        input_coeffs_3d (NumbaList): Original 3D polynomial coefficients.
        gl_nodes, gl_weights: GL quadrature on [0, 1].
        ts_nodes, ts_weights: Tanh-sinh quadrature on [0, 1].
        strategy (int): 0=GL only, 1=TS only, 2=auto mixed.

    Returns:
        tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
            (points, scalar_weights, normal_weights) with shapes
            ``(n_pts, 3)``, ``(n_pts,)``, ``(n_pts, 3)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    q = len(gl_nodes)

    if k2 >= 3:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )

    # Per-level quadrature selection.
    nodes_0 = gl_nodes
    wts_0 = gl_weights
    nodes_1 = gl_nodes
    wts_1 = gl_weights
    if strategy == 1:
        nodes_0 = ts_nodes
        wts_0 = ts_weights
        nodes_1 = ts_nodes
        wts_1 = ts_weights
    elif strategy == 2:
        if use_ts_1:
            nodes_0 = ts_nodes
            wts_0 = ts_weights
        if use_ts_2:
            nodes_1 = ts_nodes
            wts_1 = ts_weights

    # 3D tangential directions (skipping k2).
    tang2 = np.empty(2, dtype=np.int64)
    ti = 0
    for d in range(3):
        if d != k2:
            tang2[ti] = d
            ti += 1
    tang1_2d = 1 - k1
    axis_k1_3d = tang2[k1]
    axis_tang1_3d = tang2[tang1_2d]

    max_total = q * q * 50
    points = np.empty((max_total, 3), dtype=np.float64)
    scalar_wts = np.empty(max_total, dtype=np.float64)
    normal_wts = np.empty((max_total, 3), dtype=np.float64)
    n_pts = 0

    # Level 0: 1D base partition.
    bounds_0, nb_0 = _collect_and_partition_1d(coeffs_1d, masks_1d)

    for b0 in range(nb_0 - 1):
        lo0 = bounds_0[b0]
        hi0 = bounds_0[b0 + 1]
        if hi0 - lo0 < _MERGE_TOL:
            continue
        scale0 = hi0 - lo0

        for q0 in range(q):
            x_1d = lo0 + nodes_0[q0] * scale0
            w_1d = wts_0[q0] * scale0

            # Level 1: partition along k1.
            bounds_1, nb_1 = _collect_and_partition_from_2d(coeffs_2d, masks_2d, k1, x_1d)

            for b1 in range(nb_1 - 1):
                lo1 = bounds_1[b1]
                hi1 = bounds_1[b1 + 1]
                if hi1 - lo1 < _MERGE_TOL:
                    continue
                scale1 = hi1 - lo1

                for q1 in range(q):
                    x_2d = lo1 + nodes_1[q1] * scale1
                    w_2d = wts_1[q1] * scale1
                    w_base = w_1d * w_2d

                    # Build 3D base point for collapse.
                    x_base_3d = np.empty(2, dtype=np.float64)
                    if tang1_2d == 0:
                        x_base_3d[0] = x_1d
                        x_base_3d[1] = x_2d
                    else:
                        x_base_3d[0] = x_2d
                        x_base_3d[1] = x_1d

                    # Level 2: find roots of input polynomials along k2.
                    for p_idx in range(n_input_polys):
                        poly_3d = input_coeffs_3d[p_idx]

                        if not _line_intersects_3d(masks_3d[p_idx], x_base_3d, k2):
                            continue

                        poly_1d = _collapse_3d(poly_3d, k2, x_base_3d)
                        roots, n_roots = find_roots(poly_1d)

                        for ri in range(n_roots):
                            root = roots[ri]

                            # Build 3D point.
                            pt = np.empty(3, dtype=np.float64)
                            pt[axis_tang1_3d] = x_1d
                            pt[axis_k1_3d] = x_2d
                            pt[k2] = root

                            # Compute gradient for surface Jacobian.
                            grad = _eval_gradient_3d(poly_3d, pt)
                            grad_norm = np.sqrt(grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2)
                            dk_phi = grad[k2]

                            if abs(dk_phi) < 1e-300:
                                continue

                            # Resize if needed.
                            if n_pts >= len(scalar_wts):
                                new_cap = len(scalar_wts) * 2
                                new_p = np.empty((new_cap, 3), dtype=np.float64)
                                new_s = np.empty(new_cap, dtype=np.float64)
                                new_n = np.empty((new_cap, 3), dtype=np.float64)
                                for ci in range(n_pts):
                                    for di in range(3):
                                        new_p[ci, di] = points[ci, di]
                                        new_n[ci, di] = normal_wts[ci, di]
                                    new_s[ci] = scalar_wts[ci]
                                points = new_p
                                scalar_wts = new_s
                                normal_wts = new_n

                            # Flux-form surface weight.
                            alpha = w_base * grad_norm / abs(dk_phi)
                            for di in range(3):
                                points[n_pts, di] = pt[di]
                            scalar_wts[n_pts] = alpha
                            if grad_norm > 1e-300:
                                for di in range(3):
                                    normal_wts[n_pts, di] = w_base * grad[di] / abs(dk_phi)
                            else:
                                for di in range(3):
                                    normal_wts[n_pts, di] = 0.0
                            n_pts += 1

    return (
        points[:n_pts].copy(),
        scalar_wts[:n_pts].copy(),
        normal_wts[:n_pts].copy(),
    )
