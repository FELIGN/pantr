"""Profile the implicit quadrature build and construction phases.

Measures timing for:
  - Build phase (ImplicitPolyQuadrature constructor) with sub-function breakdown
  - Construction phase (volume_quad / surface_quad) with micro-benchmarks
  - Root finding by polynomial degree
  - Side-by-side comparison with algoim C++ benchmark

Test geometries (from Saye, JCP 2022):
  2D: circle, ellipse, bilinear, deltoid (singular)
  3D: sphere, ellipsoid, trilinear tunnel, dingdong (singular)

Reference C++ timings from Table 1 (Intel Xeon E3-1535m v6, ~4 GHz):
  2D ellipse/cell:   build=0.2 us,  q=1:0.1  q=2:0.2  q=4:0.5  q=8:1.0  q=16:2.5
  3D ellipsoid/cell:  build=1.5 us,  q=1:0.2  q=2:0.8  q=4:2.2  q=8:8.8  q=16:50
  2D random/inst:     build=0.4 us,  q=1:0.2  q=2:0.3  q=4:0.6  q=8:1.2  q=16:2.9
  3D random/inst:     build=6 us,    q=1:0.9  q=2:2.2  q=4:8.0  q=8:25   q=16:120
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from numba.typed import List as NumbaList
from numpy import typing as npt

from pantr.bezier.implicit import ImplicitPolyQuadrature, QuadStrategy
from pantr.bezier.implicit._bernstein import (
    _collapse_2d,
    _collapse_3d,
    _eval_gradient_2d,
    _eval_gradient_3d,
    _face_restrict_2d,
    _face_restrict_3d,
)
from pantr.bezier.implicit._build import build_2d, build_3d
from pantr.bezier.implicit._construct import (
    _collect_and_partition_1d,
    _collect_and_partition_from_2d,
    _collect_and_partition_from_3d,
)
from pantr.bezier.implicit._convert import monomial_to_bernstein_2d, monomial_to_bernstein_3d
from pantr.bezier.implicit._mask import compute_nonzero_mask_2d, compute_nonzero_mask_3d
from pantr.bezier.implicit._resultant import discriminant_2d, discriminant_3d
from pantr.bezier.implicit._roots import find_roots
from pantr.bezier.implicit._score import score_estimate_2d, score_estimate_3d

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Q_VALUES = [1, 2, 4, 8, 16]
N_WARMUP = 3
N_REPEAT = 80


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------


def _mono_to_bernstein_1d(mono: npt.NDArray[np.float64], degree: int) -> npt.NDArray[np.float64]:
    n = degree
    mat = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            mat[i, j] = comb(i, j) / comb(n, j)
    m = np.zeros(n + 1)
    m[: min(len(mono), n + 1)] = mono[: min(len(mono), n + 1)]
    return mat @ m


def make_circle() -> npt.NDArray[np.float64]:
    """Circle: (x-0.5)^2 + (y-0.5)^2 - 0.1 on [0,1]^2. Degree (2,2)."""
    r_sq = 0.1
    c_val = 0.5 - r_sq
    return np.array(
        [
            [c_val, c_val - 0.5, c_val],
            [c_val - 0.5, c_val - 1.0, c_val - 0.5],
            [c_val, c_val - 0.5, c_val],
        ]
    )


def make_ellipse() -> npt.NDArray[np.float64]:
    """Ellipse: x^2 + 4y^2 - 1 on (-1.1, 1.1)^2. Degree (2,2)."""
    lo, hi = -1.1, 1.1
    a, c = lo, lo
    hx, hy = hi - lo, hi - lo
    bern_x = _mono_to_bernstein_1d(np.array([a**2, 2 * a * hx, hx**2]), 2)
    bern_y = _mono_to_bernstein_1d(np.array([4 * c**2, 8 * c * hy, 4 * hy**2]), 2)
    coeffs = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            coeffs[i, j] = bern_x[i] + bern_y[j] - 1.0
    return coeffs


def make_bilinear(eps: float = 0.1) -> npt.NDArray[np.float64]:
    """Bilinear: (x-0.5)(y-0.5) - eps^2 on [0,1]^2. Degree (1,1)."""
    e2 = eps * eps
    return np.array([[0.25 - e2, -0.25 - e2], [-0.25 - e2, 0.25 - e2]])


def make_deltoid() -> npt.NDArray[np.float64]:
    """Deltoid: (x^2+y^2)^2+18(x^2+y^2)-8(x^3-3xy^2)-27 on (-2.5,3.5)x(-3,3). Degree (4,4)."""
    mono = np.zeros((5, 5))
    mono[0, 0] = -27.0
    mono[2, 0] = 18.0
    mono[0, 2] = 18.0
    mono[3, 0] = -8.0
    mono[1, 2] = 24.0
    mono[4, 0] = 1.0
    mono[2, 2] = 2.0
    mono[0, 4] = 1.0
    lo = np.array([-2.5, -3.0])
    hi = np.array([3.5, 3.0])
    return monomial_to_bernstein_2d(mono, (4, 4), lo, hi)


def make_sphere() -> npt.NDArray[np.float64]:
    """Sphere: (x-.5)^2+(y-.5)^2+(z-.5)^2-0.09 on [0,1]^3. Degree (2,2,2)."""
    r_sq = 0.09
    const = 0.75 - r_sq
    bx = np.array([0.0, -0.5, 0.0])
    c = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c[i, j, k] = bx[i] + bx[j] + bx[k] + const
    return c


def make_ellipsoid() -> npt.NDArray[np.float64]:
    """Ellipsoid: x^2+4y^2+9z^2-1 on (-1.1,1.1)^3. Degree (2,2,2)."""
    lo, hi = -1.1, 1.1
    a, c, e = lo, lo, lo
    hx, hy, hz = hi - lo, hi - lo, hi - lo
    bern_x = _mono_to_bernstein_1d(np.array([a**2, 2 * a * hx, hx**2]), 2)
    bern_y = _mono_to_bernstein_1d(np.array([4 * c**2, 8 * c * hy, 4 * hy**2]), 2)
    bern_z = _mono_to_bernstein_1d(np.array([9 * e**2, 18 * e * hz, 9 * hz**2]), 2)
    coeffs = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                coeffs[i, j, k] = bern_x[i] + bern_y[j] + bern_z[k] - 1.0
    return coeffs


def make_trilinear_tunnel() -> npt.NDArray[np.float64]:
    """Trilinear tunnel (paper sec 4.5): degree (1,1,1) on [0,1]^3."""
    c = np.empty((2, 2, 2))
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                x, y, z = float(ix), float(iy), float(iz)
                c[ix, iy, iz] = (
                    0.5
                    - 1.2 * x
                    - 1.3 * y
                    - 1.4 * z
                    + 2.9 * x * y
                    + 3.2 * x * z
                    + 3.3 * y * z
                    - 6.5 * x * y * z
                )
    return c


def make_dingdong() -> npt.NDArray[np.float64]:
    """Ding-dong: x^2+y^2-(1-z)z^2 on (-1,1)^3. Degree (2,2,3). Cusp singularity."""
    mono = np.zeros((3, 3, 4))
    mono[2, 0, 0] = 1.0
    mono[0, 2, 0] = 1.0
    mono[0, 0, 2] = -1.0
    mono[0, 0, 3] = 1.0
    lo = np.array([-1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0])
    return monomial_to_bernstein_3d(mono, (2, 2, 3), lo, hi)


def make_random_2d(seed: int = 42) -> npt.NDArray[np.float64]:
    """Random degree-(2,2) polynomial on (-1,1)^2 (Saye Table 1 class A)."""
    rng = np.random.default_rng(seed)
    alpha = 2.0
    leg_mono = [
        np.array([np.sqrt(0.5)]),
        np.array([0.0, np.sqrt(1.5)]),
        np.array([-np.sqrt(5.0 / 8.0), 0.0, 3.0 * np.sqrt(5.0 / 8.0)]),
    ]
    raw_c = rng.uniform(-1, 1, size=(3, 3))
    mono = np.zeros((3, 3))
    for i0 in range(3):
        for i1 in range(3):
            s = i0 + i1
            lam = 1.0 if s == 0 else float(s) ** (-alpha)
            c = raw_c[i0, i1] * lam
            lx = leg_mono[i0]
            ly = leg_mono[i1]
            for px in range(len(lx)):
                for py in range(len(ly)):
                    mono[px, py] += c * lx[px] * ly[py]
    lo = np.array([-1.0, -1.0])
    hi = np.array([1.0, 1.0])
    return monomial_to_bernstein_2d(mono, (2, 2), lo, hi)


def make_random_3d(seed: int = 42) -> npt.NDArray[np.float64]:
    """Random degree-(2,2,2) polynomial on (-1,1)^3 (Saye Table 1 class A)."""
    rng = np.random.default_rng(seed)
    alpha = 2.0
    leg_mono = [
        np.array([np.sqrt(0.5)]),
        np.array([0.0, np.sqrt(1.5)]),
        np.array([-np.sqrt(5.0 / 8.0), 0.0, 3.0 * np.sqrt(5.0 / 8.0)]),
    ]
    raw_c = rng.uniform(-1, 1, size=(3, 3, 3))
    mono = np.zeros((3, 3, 3))
    for i0 in range(3):
        for i1 in range(3):
            for i2 in range(3):
                s = i0 + i1 + i2
                lam = 1.0 if s == 0 else float(s) ** (-alpha)
                c = raw_c[i0, i1, i2] * lam
                lx = leg_mono[i0]
                ly = leg_mono[i1]
                lz = leg_mono[i2]
                for px in range(len(lx)):
                    for py in range(len(ly)):
                        for pz in range(len(lz)):
                            mono[px, py, pz] += c * lx[px] * ly[py] * lz[pz]
    lo = np.array([-1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0])
    return monomial_to_bernstein_3d(mono, (2, 2, 2), lo, hi)


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


def _time_ns(func: Any, *args: Any, **kwargs: Any) -> tuple[float, Any]:  # noqa: ANN401
    """Return (elapsed_ns, result)."""
    t0 = time.perf_counter_ns()
    result = func(*args, **kwargs)
    t1 = time.perf_counter_ns()
    return float(t1 - t0), result


def _median_time_us(func: Any, n_warmup: int, n_repeat: int, *args: Any) -> float:  # noqa: ANN401
    """Return median time in microseconds."""
    for _ in range(n_warmup):
        func(*args)
    times = []
    for _ in range(n_repeat):
        t, _ = _time_ns(func, *args)
        times.append(t / 1e3)  # ns -> us
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Build-phase sub-function profiling
# ---------------------------------------------------------------------------


def profile_build_breakdown_2d(
    coeffs: npt.NDArray[np.float64], n_repeat: int = N_REPEAT
) -> dict[str, float]:
    """Detailed 2D build breakdown: mask, score, discriminant, face_restrict, total, base_part."""
    results: dict[str, float] = {}

    coeffs_list = NumbaList()
    coeffs_list.append(coeffs)

    # 1. Mask computation
    results["mask"] = _median_time_us(compute_nonzero_mask_2d, N_WARMUP, n_repeat, coeffs)

    masks_list = NumbaList()
    masks_list.append(compute_nonzero_mask_2d(coeffs))

    # 2. Score estimation
    results["score"] = _median_time_us(
        score_estimate_2d, N_WARMUP, n_repeat, coeffs_list, masks_list
    )

    # 3. Get best axis k for sub-function profiling
    scores, _ = score_estimate_2d(coeffs_list, masks_list)
    k = int(np.argmax(scores))

    # 4. Face restriction (2 sides)
    results["face_restrict"] = _median_time_us(_face_restrict_2d, N_WARMUP, n_repeat, coeffs, k, 0)

    # 5. Discriminant (if degree >= 2 in direction k)
    degree_k = coeffs.shape[k] - 1
    if degree_k >= 2:  # noqa: PLR2004
        results["discriminant"] = _median_time_us(discriminant_2d, N_WARMUP, n_repeat, coeffs, k)
    else:
        results["discriminant"] = 0.0

    # 6. Full build_2d (total Numba)
    results["build_total"] = _median_time_us(build_2d, N_WARMUP, n_repeat, coeffs_list, masks_list)

    # 7. Base partition (root finding on 1D resultants)
    br = build_2d(coeffs_list, masks_list)
    coeffs_1d = br[0]
    masks_1d = br[1]
    if len(coeffs_1d) > 0:
        results["base_partition"] = _median_time_us(
            _collect_and_partition_1d, N_WARMUP, n_repeat, coeffs_1d, masks_1d
        )
    else:
        results["base_partition"] = 0.0

    return results


def profile_build_breakdown_3d(
    coeffs: npt.NDArray[np.float64], n_repeat: int = N_REPEAT
) -> dict[str, float]:
    """Detailed 3D build breakdown."""
    results: dict[str, float] = {}

    coeffs_list = NumbaList()
    coeffs_list.append(coeffs)

    # 1. Mask
    results["mask"] = _median_time_us(compute_nonzero_mask_3d, N_WARMUP, n_repeat, coeffs)

    masks_list = NumbaList()
    masks_list.append(compute_nonzero_mask_3d(coeffs))

    # 2. Score (3D level)
    results["score_3d"] = _median_time_us(
        score_estimate_3d, N_WARMUP, n_repeat, coeffs_list, masks_list
    )

    # 3. Get best axis for sub-function profiling
    scores, _ = score_estimate_3d(coeffs_list, masks_list)
    k = int(np.argmax(scores))

    # 4. Face restriction (3D)
    results["face_restrict_3d"] = _median_time_us(
        _face_restrict_3d, N_WARMUP, n_repeat, coeffs, k, 0
    )

    # 5. Discriminant (3D)
    degree_k = coeffs.shape[k] - 1
    if degree_k >= 2:  # noqa: PLR2004
        results["discriminant_3d"] = _median_time_us(discriminant_3d, N_WARMUP, n_repeat, coeffs, k)
    else:
        results["discriminant_3d"] = 0.0

    # 6. Full build_3d
    results["build_total"] = _median_time_us(build_3d, N_WARMUP, n_repeat, coeffs_list, masks_list)

    # 7. Base partition
    br = build_3d(coeffs_list, masks_list)
    coeffs_1d = br[0]
    masks_1d = br[1]
    if len(coeffs_1d) > 0:
        results["base_partition"] = _median_time_us(
            _collect_and_partition_1d, N_WARMUP, n_repeat, coeffs_1d, masks_1d
        )
    else:
        results["base_partition"] = 0.0

    return results


# ---------------------------------------------------------------------------
# Construction-phase sub-function profiling
# ---------------------------------------------------------------------------


def _get_representative_base_point(base_bounds: npt.NDArray[np.float64], base_nb: int) -> float:
    """Pick the midpoint of the first non-trivial partition interval."""
    if base_nb >= 3:  # noqa: PLR2004
        return 0.5 * (base_bounds[0] + base_bounds[1])
    return 0.5


def profile_construct_breakdown_2d(
    coeffs: npt.NDArray[np.float64], n_repeat: int = N_REPEAT
) -> dict[str, Any]:
    """Profile construction sub-functions for 2D."""
    ipq = ImplicitPolyQuadrature(coeffs)
    br = ipq._build_result
    results: dict[str, Any] = {}

    # Extract hierarchy
    # build_2d returns 10-tuple: (c1d, m1d, k0, ts0, t0, c2d, m2d, k1, ts1, t1)
    coeffs_2d = br[5]
    masks_2d = br[6]
    k1 = br[7]
    base_bounds = ipq._base_bounds
    base_nb = ipq._base_nb

    x_tang = _get_representative_base_point(base_bounds, base_nb)

    # 1. _collapse_2d
    if len(coeffs_2d) > 0:
        results["collapse_2d"] = _median_time_us(
            _collapse_2d, N_WARMUP, n_repeat, coeffs_2d[0], k1, x_tang
        )
    else:
        results["collapse_2d"] = 0.0

    # 2. find_roots on collapsed polynomial
    if len(coeffs_2d) > 0:
        poly_1d = _collapse_2d(coeffs_2d[0], k1, x_tang)
        results["find_roots"] = _median_time_us(find_roots, N_WARMUP, n_repeat, poly_1d)
    else:
        results["find_roots"] = 0.0

    # 3. _collect_and_partition_from_2d (combined collapse + roots + partition)
    if len(coeffs_2d) > 0:
        results["collect_partition_2d"] = _median_time_us(
            _collect_and_partition_from_2d,
            N_WARMUP,
            n_repeat,
            coeffs_2d,
            masks_2d,
            k1,
            x_tang,
        )
    else:
        results["collect_partition_2d"] = 0.0

    # 4. _eval_gradient_2d (for surface quad)
    pt = np.zeros(2)
    pt[1 - k1] = x_tang
    if len(coeffs_2d) > 0:
        poly_1d = _collapse_2d(coeffs_2d[0], k1, x_tang)
        roots, n_roots, _ = find_roots(poly_1d)
        if n_roots > 0:
            pt[k1] = roots[0]
        else:
            pt[k1] = 0.5
    results["eval_gradient_2d"] = _median_time_us(_eval_gradient_2d, N_WARMUP, n_repeat, coeffs, pt)

    # 5. Estimate vs actual for volume quad
    n_base = max(base_nb - 1, 1)
    results["n_base_intervals"] = n_base
    for q in Q_VALUES:
        n_calls = n_base * q
        results[f"est_vol_q={q}"] = n_calls * results["collect_partition_2d"]

    return results


def profile_construct_breakdown_3d(
    coeffs: npt.NDArray[np.float64], n_repeat: int = N_REPEAT
) -> dict[str, Any]:
    """Profile construction sub-functions for 3D."""
    ipq = ImplicitPolyQuadrature(coeffs)
    br = ipq._build_result
    results: dict[str, Any] = {}

    # build_3d returns 15-tuple:
    # (c1d, m1d, k0, ts0, t0, c2d, m2d, k1, ts1, t1, c3d, m3d, k2, ts2, t2)
    coeffs_2d = br[5]
    masks_2d = br[6]
    k1 = br[7]
    coeffs_3d = br[10]
    masks_3d = br[11]
    k2 = br[12]
    base_bounds = ipq._base_bounds
    base_nb = ipq._base_nb

    x_tang = _get_representative_base_point(base_bounds, base_nb)

    # 1. _collapse_3d
    if len(coeffs_3d) > 0:
        x_base_2d = np.array([x_tang, 0.5])
        results["collapse_3d"] = _median_time_us(
            _collapse_3d, N_WARMUP, n_repeat, coeffs_3d[0], k2, x_base_2d
        )
    else:
        results["collapse_3d"] = 0.0

    # 2. find_roots on collapsed 3D polynomial
    if len(coeffs_3d) > 0:
        x_base_2d = np.array([x_tang, 0.5])
        poly_1d = _collapse_3d(coeffs_3d[0], k2, x_base_2d)
        results["find_roots"] = _median_time_us(find_roots, N_WARMUP, n_repeat, poly_1d)
    else:
        results["find_roots"] = 0.0

    # 3. _collect_and_partition_from_2d (level 1: 2D -> 1D partitioning)
    if len(coeffs_2d) > 0:
        results["collect_partition_2d"] = _median_time_us(
            _collect_and_partition_from_2d,
            N_WARMUP,
            n_repeat,
            coeffs_2d,
            masks_2d,
            k1,
            x_tang,
        )
    else:
        results["collect_partition_2d"] = 0.0

    # 4. _collect_and_partition_from_3d (level 2: 3D -> 1D partitioning)
    if len(coeffs_3d) > 0:
        x_base_2d = np.array([x_tang, 0.5])
        results["collect_partition_3d"] = _median_time_us(
            _collect_and_partition_from_3d,
            N_WARMUP,
            n_repeat,
            coeffs_3d,
            masks_3d,
            k2,
            x_base_2d,
        )
    else:
        results["collect_partition_3d"] = 0.0

    # 5. _eval_gradient_3d
    pt = np.full(3, 0.5)
    results["eval_gradient_3d"] = _median_time_us(_eval_gradient_3d, N_WARMUP, n_repeat, coeffs, pt)

    # 6. Estimate: for 3D, outer = q * n_base, each calls collect_partition_2d
    #    then inner = q * n_mid, each calls collect_partition_3d
    n_base = max(base_nb - 1, 1)
    results["n_base_intervals"] = n_base

    # Estimate n_mid by calling collect_partition_2d at representative point
    if len(coeffs_2d) > 0:
        mid_bounds, mid_nb = _collect_and_partition_from_2d(coeffs_2d, masks_2d, k1, x_tang)
        n_mid = max(mid_nb - 1, 1)
    else:
        n_mid = 1
    results["n_mid_intervals"] = n_mid

    for q in Q_VALUES:
        n_outer_calls = n_base * q
        n_inner_calls = n_outer_calls * n_mid * q
        est_vol = (
            n_outer_calls * results["collect_partition_2d"]
            + n_inner_calls * results["collect_partition_3d"]
        )
        results[f"est_vol_q={q}"] = est_vol

    return results


# ---------------------------------------------------------------------------
# Root finding by degree
# ---------------------------------------------------------------------------


def profile_root_finding_by_degree(n_repeat: int = N_REPEAT) -> dict[int, float]:
    """Time find_roots on polynomials of increasing degree."""
    results: dict[int, float] = {}
    rng = np.random.default_rng(123)

    for degree in [1, 2, 3, 4, 6, 8, 10, 15, 20]:
        # Create a Bernstein polynomial with 1-3 roots in (0,1).
        # Use a polynomial that changes sign: p(0) = +1, p(1) = -1 with random middle.
        coeffs = rng.standard_normal(degree + 1)
        coeffs[0] = abs(coeffs[0]) + 0.1
        coeffs[-1] = -(abs(coeffs[-1]) + 0.1)
        coeffs = np.ascontiguousarray(coeffs, dtype=np.float64)

        results[degree] = _median_time_us(find_roots, N_WARMUP, n_repeat, coeffs)

    return results


# ---------------------------------------------------------------------------
# High-level profiling
# ---------------------------------------------------------------------------


def profile_geometry(  # noqa: PLR0912
    name: str,
    coeffs: npt.NDArray[np.float64],
    q_values: list[int] | None = None,
    n_warmup: int = N_WARMUP,
    n_repeat: int = N_REPEAT,
) -> dict[str, Any]:
    """Profile build + construction for one geometry."""
    if q_values is None:
        q_values = Q_VALUES
    dim = coeffs.ndim
    results: dict[str, Any] = {"name": name, "dim": dim, "shape": coeffs.shape}

    # --- Build phase ---
    for _ in range(n_warmup):
        ipq = ImplicitPolyQuadrature(coeffs)

    build_times = []
    for _ in range(n_repeat):
        t, ipq = _time_ns(ImplicitPolyQuadrature, coeffs)
        build_times.append(t / 1e3)
    results["build_us"] = float(np.median(build_times))

    # --- Construction phase: volume ---
    vol_times: dict[int, float] = {}
    vol_npts: dict[int, int] = {}
    for q in q_values:
        for _ in range(n_warmup):
            ipq.volume_quad(q, QuadStrategy.AUTO_MIXED)
        times = []
        for _ in range(n_repeat):
            t, (pts, wts) = _time_ns(ipq.volume_quad, q, QuadStrategy.AUTO_MIXED)
            times.append(t / 1e3)
        vol_times[q] = float(np.median(times))
        vol_npts[q] = len(wts)
    results["volume_us"] = vol_times
    results["volume_npts"] = vol_npts

    # --- Construction phase: surface ---
    surf_times: dict[int, float] = {}
    surf_npts: dict[int, int] = {}
    for q in q_values:
        for _ in range(n_warmup):
            ipq.surface_quad(q, QuadStrategy.AUTO_MIXED)
        times = []
        for _ in range(n_repeat):
            t, (pts, sw, nw) = _time_ns(ipq.surface_quad, q, QuadStrategy.AUTO_MIXED)
            times.append(t / 1e3)
        surf_times[q] = float(np.median(times))
        surf_npts[q] = len(sw)
    results["surface_us"] = surf_times
    results["surface_npts"] = surf_npts

    # --- Build sub-function breakdown ---
    if dim == 2:  # noqa: PLR2004
        results["build_breakdown"] = profile_build_breakdown_2d(coeffs, n_repeat)
    else:
        results["build_breakdown"] = profile_build_breakdown_3d(coeffs, n_repeat)

    # --- Construction sub-function breakdown ---
    if dim == 2:  # noqa: PLR2004
        results["construct_breakdown"] = profile_construct_breakdown_2d(coeffs, n_repeat)
    else:
        results["construct_breakdown"] = profile_construct_breakdown_3d(coeffs, n_repeat)

    return results


# ---------------------------------------------------------------------------
# C++ benchmark runner
# ---------------------------------------------------------------------------


def _parse_cpp_output(text: str) -> dict[str, dict[str, Any]]:
    """Parse structured output from bench_algoim binary."""
    results: dict[str, dict[str, Any]] = {}
    current_name: str | None = None
    current_section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("=", "-")):
            if line.startswith("="):
                current_section = None
            continue

        # Geometry header
        m = re.match(r"^([\w\s=.]+\([23]D\))", line)
        if m:
            current_name = m.group(1).strip()
            results[current_name] = {"build_us": 0.0, "volume_us": {}, "surface_us": {}}
            current_section = None
            continue

        if current_name is None:
            continue

        # Build phase
        m = re.match(r"Build phase:\s+([\d.]+)\s+us", line)
        if m:
            results[current_name]["build_us"] = float(m.group(1))
            continue

        # Section header
        if "Volume quadrature" in line:
            current_section = "volume"
            continue
        if "Surface quadrature" in line:
            current_section = "surface"
            continue

        # Data row
        m = re.match(r"(\d+)\s+([\d.]+)\s+(\d+)", line)
        if m and current_section:
            q = int(m.group(1))
            t = float(m.group(2))
            key = f"{current_section}_us"
            results[current_name][key][q] = t

    return results


def _map_cpp_name(py_name: str) -> str | None:
    """Map Python geometry name to C++ output name."""
    mapping = {
        "circle (2D)": "circle (2D)",
        "ellipse (2D)": "ellipse (2D)",
        "bilinear eps=0.1 (2D)": "bilinear eps=0.1 (2D)",
        "sphere (3D)": "sphere (3D)",
        "ellipsoid (3D)": "ellipsoid (3D)",
        "trilinear tunnel (3D)": "trilinear tunnel (3D)",
    }
    return mapping.get(py_name)


def run_cpp_benchmark() -> dict[str, dict[str, Any]] | None:
    """Run the C++ algoim benchmark and parse results."""
    binary = Path(__file__).parent / "bench_algoim"
    if not binary.exists():
        print("  C++ benchmark binary not found. Skipping comparison.")
        return None

    env = os.environ.copy()
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        env["DYLD_LIBRARY_PATH"] = f"{conda_prefix}/lib"
    else:
        env["DYLD_LIBRARY_PATH"] = "/Users/antolin/miniconda3/envs/pantr/lib"

    try:
        result = subprocess.run(
            [str(binary)], check=False, capture_output=True, text=True, env=env, timeout=120
        )
        if result.returncode != 0:
            print(f"  C++ benchmark failed: {result.stderr[:200]}")
            return None
        return _parse_cpp_output(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  C++ benchmark error: {e}")
        return None


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_build_breakdown(results: dict[str, Any]) -> None:
    """Print detailed build-phase breakdown for one geometry."""
    name = results["name"]
    dim = results["dim"]
    bd = results.get("build_breakdown", {})
    total = bd.get("build_total", 1.0)

    print(f"\n  Build phase breakdown ({name}, dim={dim}):")
    print(f"  {'Sub-function':30s} {'time (us)':>10s} {'% of build':>10s}")
    print(f"  {'-' * 52}")

    display_keys = []
    if dim == 2:  # noqa: PLR2004
        display_keys = [
            ("mask", "mask"),
            ("score", "score"),
            ("face_restrict", "face_restrict (x1)"),
            ("discriminant", "discriminant"),
            ("base_partition", "base_partition (roots)"),
            ("build_total", "build_total (Numba)"),
        ]
    else:
        display_keys = [
            ("mask", "mask (3D)"),
            ("score_3d", "score (3D level)"),
            ("face_restrict_3d", "face_restrict_3d (x1)"),
            ("discriminant_3d", "discriminant (3D)"),
            ("base_partition", "base_partition (roots)"),
            ("build_total", "build_total (Numba)"),
        ]

    for key, label in display_keys:
        t = bd.get(key, 0.0)
        pct = t / total * 100 if total > 0 else 0
        if key == "build_total":
            print(f"  {'':30s} {'':>10s} {'':>10s}")
            print(f"  {label:30s} {t:10.1f} {'100%':>10s}")
        else:
            print(f"  {label:30s} {t:10.1f} {pct:9.0f}%")

    # Python constructor total
    print(f"  {'build (Python constructor)':30s} {results['build_us']:10.1f}")


def print_construct_breakdown(results: dict[str, Any]) -> None:
    """Print construction-phase micro-benchmarks."""
    cd = results.get("construct_breakdown", {})
    dim = results["dim"]

    print("\n  Construction micro-benchmarks (per-call, representative point):")
    print(f"  {'Operation':35s} {'time (us)':>10s}")
    print(f"  {'-' * 47}")

    if dim == 2:  # noqa: PLR2004
        for key, label in [
            ("collapse_2d", "collapse_2d"),
            ("find_roots", "find_roots (on collapsed)"),
            ("collect_partition_2d", "collect_partition_from_2d"),
            ("eval_gradient_2d", "eval_gradient_2d"),
        ]:
            t = cd.get(key, 0.0)
            print(f"  {label:35s} {t:10.2f}")
    else:
        for key, label in [
            ("collapse_3d", "collapse_3d"),
            ("find_roots", "find_roots (on collapsed)"),
            ("collect_partition_2d", "collect_partition_from_2d (level 1)"),
            ("collect_partition_3d", "collect_partition_from_3d (level 2)"),
            ("eval_gradient_3d", "eval_gradient_3d"),
        ]:
            t = cd.get(key, 0.0)
            print(f"  {label:35s} {t:10.2f}")

    n_base = cd.get("n_base_intervals", 1)
    n_mid = cd.get("n_mid_intervals", None)
    print(f"\n  Partition info: n_base_intervals={n_base}", end="")
    if n_mid is not None:
        print(f", n_mid_intervals={n_mid}", end="")
    print()

    # Estimate vs actual
    # NOTE: "est" is computed from per-call micro-benchmarks invoked *from Python*.
    # Each such call pays ~0.3-2 us of Python->Numba dispatch overhead.
    # The actual volume_quad kernel runs everything inside a single Numba call,
    # so actual < est means Numba-internal calls are much cheaper (no dispatch).
    # When actual > est, there's real overhead from buffer management, etc.
    print("\n  Estimated vs actual volume quad:")
    print(f"  {'q':>4s} {'est (us)':>10s} {'actual (us)':>12s} {'ratio':>8s}")
    print(f"  {'-' * 40}")
    for q in Q_VALUES:
        est_key = f"est_vol_q={q}"
        if est_key in cd:
            est = cd[est_key]
            actual = results["volume_us"].get(q, 0)
            if actual > 0:
                ratio = est / actual
                print(f"  {q:4d} {est:10.1f} {actual:12.1f} {ratio:7.1f}x")
            else:
                print(f"  {q:4d} {est:10.1f} {'N/A':>12s}")


def print_quad_tables(results: dict[str, Any], cpp_data: dict[str, Any] | None) -> None:
    """Print volume/surface quad tables with optional C++ comparison."""
    name = results["name"]
    cpp_key = _map_cpp_name(name)
    cpp = cpp_data.get(cpp_key) if (cpp_data and cpp_key) else None

    print("\n  Volume quadrature:")
    header = f"  {'q':>4s} {'time (us)':>12s} {'n_pts':>8s}"
    if cpp:
        header += f" {'C++ (us)':>10s} {'ratio':>8s}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for q in sorted(results["volume_us"]):
        t = results["volume_us"][q]
        n = results["volume_npts"][q]
        line = f"  {q:4d} {t:12.1f} {n:8d}"
        if cpp and q in cpp.get("volume_us", {}):
            ct = cpp["volume_us"][q]
            ratio = t / ct if ct > 0 else float("inf")
            line += f" {ct:10.1f} {ratio:7.1f}x"
        print(line)

    print("\n  Surface quadrature:")
    header = f"  {'q':>4s} {'time (us)':>12s} {'n_pts':>8s}"
    if cpp:
        header += f" {'C++ (us)':>10s} {'ratio':>8s}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for q in sorted(results["surface_us"]):
        t = results["surface_us"][q]
        n = results["surface_npts"][q]
        line = f"  {q:4d} {t:12.1f} {n:8d}"
        if cpp and q in cpp.get("surface_us", {}):
            ct = cpp["surface_us"][q]
            ratio = t / ct if ct > 0 else float("inf")
            line += f" {ct:10.1f} {ratio:7.1f}x"
        print(line)


def print_results(results: dict[str, Any], cpp_data: dict[str, Any] | None) -> None:
    """Print full results for one geometry."""
    name = results["name"]
    dim = results["dim"]
    cpp_key = _map_cpp_name(name)
    cpp = cpp_data.get(cpp_key) if (cpp_data and cpp_key) else None

    print(f"\n{'=' * 80}")
    print(f"  {name}  (dim={dim}, shape={results['shape']})")
    print(f"{'=' * 80}")

    build_line = f"\n  Build phase: {results['build_us']:.1f} us"
    if cpp:
        ratio = results["build_us"] / cpp["build_us"] if cpp["build_us"] > 0 else float("inf")
        build_line += f"  (C++: {cpp['build_us']:.1f} us, ratio: {ratio:.1f}x)"
    print(build_line)

    print_build_breakdown(results)
    print_construct_breakdown(results)
    print_quad_tables(results, cpp_data)


def print_root_finding_table(root_times: dict[int, float]) -> None:
    """Print root finding times by degree."""
    print(f"\n{'=' * 80}")
    print("  ROOT FINDING BY POLYNOMIAL DEGREE")
    print(f"{'=' * 80}")
    print(f"  {'degree':>6s} {'time (us)':>10s} {'method':>15s}")
    print(f"  {'-' * 35}")
    for deg in sorted(root_times):
        t = root_times[deg]
        if deg <= 2:  # noqa: PLR2004
            method = "analytic"
        elif deg < 6:  # noqa: PLR2004
            method = "Yuksel"
        else:
            method = "Bezier clip"
        print(f"  {deg:6d} {t:10.2f} {method:>15s}")


def print_summary_table(all_results: list[dict[str, Any]], cpp_data: dict[str, Any] | None) -> None:
    """Print compact summary table."""
    print(f"\n\n{'=' * 120}")
    print("  SUMMARY TABLE (all times in us)")
    print(f"{'=' * 120}")

    # Header
    header = f"  {'Geometry':28s} {'dim':>3s} {'build':>7s}"
    for q in Q_VALUES:
        header += f" {'v q=' + str(q):>8s}"
    for q in Q_VALUES:
        header += f" {'s q=' + str(q):>8s}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for r in all_results:
        line = f"  {r['name']:28s} {r['dim']:3d} {r['build_us']:7.1f}"
        for q in Q_VALUES:
            line += f" {r['volume_us'].get(q, 0):8.1f}"
        for q in Q_VALUES:
            line += f" {r['surface_us'].get(q, 0):8.1f}"
        print(line)

    # C++ reference
    if cpp_data:
        print()
        for cpp_name, cd in cpp_data.items():
            line = f"  {'C++ ' + cpp_name:28s} {'':>3s} {cd.get('build_us', 0):7.1f}"
            for q in Q_VALUES:
                line += f" {cd.get('volume_us', {}).get(q, 0):8.1f}"
            for q in Q_VALUES:
                line += f" {cd.get('surface_us', {}).get(q, 0):8.1f}"
            print(line)


def print_optimization_recommendations(  # noqa: PLR0915
    all_results: list[dict[str, Any]], cpp_data: dict[str, Any] | None
) -> None:
    """Analyze results and print specific optimization recommendations."""
    print(f"\n\n{'=' * 80}")
    print("  OPTIMIZATION OPPORTUNITIES")
    print(f"{'=' * 80}")

    for r in all_results:
        name = r["name"]
        bd = r.get("build_breakdown", {})
        cd = r.get("construct_breakdown", {})
        dim = r["dim"]
        total_build = bd.get("build_total", 0)

        print(f"\n  [{name}]")

        # Build-phase analysis
        disc_key = "discriminant" if dim == 2 else "discriminant_3d"  # noqa: PLR2004
        disc_t = bd.get(disc_key, 0)
        if total_build > 0 and disc_t > 0:
            disc_pct = disc_t / total_build * 100
            print(f"    Build: discriminant = {disc_pct:.0f}% of Numba build ({disc_t:.1f} us)")
            if disc_pct > 40:  # noqa: PLR2004
                print("      -> SVD in _bernstein_interpolate_1d is likely the bottleneck.")
                print("      -> Consider: QR instead of SVD, or direct Bernstein interpolation.")

        # Construction-phase analysis: dispatch overhead ratio
        # est/actual > 1 means Python->Numba dispatch overhead inflates micro-benchmarks.
        # est/actual < 1 means actual construction has real overhead beyond sub-functions.
        est_key = "est_vol_q=16"
        if est_key in cd:
            est = cd[est_key]
            actual = r["volume_us"].get(16, 0)
            if actual > 0 and est > 0:
                ratio = est / actual
                if ratio > 2.0:  # noqa: PLR2004
                    print(
                        f"    Construction: micro-bench est is {ratio:.1f}x > actual (q=16)."
                        f" Python->Numba dispatch overhead dominates sub-function timings."
                    )
                elif ratio < 0.8:  # noqa: PLR2004
                    overhead_pct = (actual - est) / actual * 100
                    print(
                        f"    Construction q=16: {overhead_pct:.0f}% real overhead"
                        f" beyond sub-functions (est={est:.1f}, actual={actual:.1f} us)."
                    )
                    print(
                        "      -> Likely: buffer resizing, point assembly, or"
                        " additional roots from multiple base intervals."
                    )

        # C++ comparison
        cpp_key = _map_cpp_name(name)
        cpp = cpp_data.get(cpp_key) if (cpp_data and cpp_key) else None
        if cpp:
            cpp_build = cpp.get("build_us", 0)
            if cpp_build > 0:
                build_ratio = r["build_us"] / cpp_build
                print(f"    Build: {build_ratio:.1f}x slower than local C++")
            # Construction ratio at q=8
            py_vol_8 = r["volume_us"].get(8, 0)
            cpp_vol_8 = cpp.get("volume_us", {}).get(8, 0)
            if py_vol_8 > 0 and cpp_vol_8 > 0:
                vol_ratio = py_vol_8 / cpp_vol_8
                print(f"    Volume q=8: {vol_ratio:.1f}x slower than local C++")

    # General recommendations
    print("\n  General observations:")
    print("    BUILD PHASE:")
    print("    - For degree-(2,2) polynomials: discriminant ~45-50% of Numba build.")
    print("      Mask (~30%) and score (~40%) are also significant.")
    print("    - For degree-(4,4) (deltoid): discriminant dominates at ~96%.")
    print("    - Python constructor overhead is ~40-90 us above Numba build total,")
    print("      caused by NumbaList creation, _precompute_base_partition, etc.")
    print("    - For random (3D), base_partition (eigvals path) is extremely slow.")
    print()
    print("    CONSTRUCTION PHASE:")
    print("    - Per-call micro-benchmarks from Python overestimate by 1.5-8x due to")
    print("      Python->Numba dispatch overhead (~0.3-2 us per call).")
    print("    - This means intra-Numba function calls are very efficient.")
    print("    - For 3D surface quad at high q, pantr matches or beats C++ (ratio ~1x).")
    print("    - For 2D and low-q 3D, ~3-15x slower than C++ (fixed overhead dominates).")
    print()
    print("    KEY SPEEDUP OPPORTUNITIES:")
    print("    1. Build phase (biggest wins for single-cell use):")
    print("       a. Reduce mask computation cost (currently 20-380 us).")
    print("          The 8x8x8 grid with iterative subdivision is expensive.")
    print("       b. Speed up score_estimate: 400 us for 3D is large.")
    print("          Consider fewer sample points or cheaper gradient estimates.")
    print("       c. Replace SVD with QR in _bernstein_interpolate_1d (small matrices).")
    print("       d. For degree-1 polynomials (bilinear, trilinear): skip discriminant")
    print("          computation entirely (degree_k < 2), but score is still expensive.")
    print("    2. Construction phase:")
    print("       a. Reduce fixed overhead per volume_quad call (~4 us for 2D, ~7 us 3D)")
    print("          that dominates at low q. This is likely Numba kernel entry cost.")
    print("       b. For dingdong/singular cases: construction overhead is real (est<actual)")
    print("          due to more complex partition structures and more root-finding calls.")
    print("    3. Python constructor overhead:")
    print("       a. NumbaList creation and type coercion adds ~40-50 us per call.")
    print("       b. _precompute_base_partition with eigvals path can be very slow (random 3D).")
    print("       c. Consider lazy base partition (defer to first volume_quad call).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0915
    """Run the implicit quadrature profiling suite."""
    print("Implicit quadrature profiling")
    print(f"Python {sys.version}")
    print(f"NumPy {np.__version__}")
    try:
        import numba  # noqa: PLC0415

        print(f"Numba {numba.__version__}")
    except ImportError:
        pass

    print(f"\nN_WARMUP={N_WARMUP}, N_REPEAT={N_REPEAT}")
    print(f"q values: {Q_VALUES}")

    # --- Run C++ benchmark first ---
    print("\nRunning C++ benchmark...", flush=True)
    cpp_data = run_cpp_benchmark()
    if cpp_data:
        print(f"  Got results for {len(cpp_data)} geometries.")

    # --- Geometry definitions ---
    geometries = [
        ("circle (2D)", make_circle()),
        ("ellipse (2D)", make_ellipse()),
        ("bilinear eps=0.1 (2D)", make_bilinear(0.1)),
        ("deltoid (2D)", make_deltoid()),
        ("random (2D)", make_random_2d()),
        ("sphere (3D)", make_sphere()),
        ("ellipsoid (3D)", make_ellipsoid()),
        ("trilinear tunnel (3D)", make_trilinear_tunnel()),
        ("dingdong (3D)", make_dingdong()),
        ("random (3D)", make_random_3d()),
    ]

    # --- JIT warmup ---
    print("\nWarming up JIT...", end="", flush=True)
    for _, coeffs in geometries:
        ipq = ImplicitPolyQuadrature(coeffs)
        ipq.volume_quad(2, QuadStrategy.AUTO_MIXED)
        ipq.surface_quad(2, QuadStrategy.AUTO_MIXED)
    # Also warm up sub-functions
    for _, coeffs in geometries:
        if coeffs.ndim == 2:  # noqa: PLR2004
            compute_nonzero_mask_2d(coeffs)
            cl = NumbaList()
            cl.append(coeffs)
            ml = NumbaList()
            ml.append(compute_nonzero_mask_2d(coeffs))
            score_estimate_2d(cl, ml)
            _face_restrict_2d(coeffs, 0, 0)
            if coeffs.shape[0] - 1 >= 2:  # noqa: PLR2004
                discriminant_2d(coeffs, 0)
        else:
            compute_nonzero_mask_3d(coeffs)
            cl = NumbaList()
            cl.append(coeffs)
            ml = NumbaList()
            ml.append(compute_nonzero_mask_3d(coeffs))
            score_estimate_3d(cl, ml)
            _face_restrict_3d(coeffs, 0, 0)
            if coeffs.shape[0] - 1 >= 2:  # noqa: PLR2004
                discriminant_3d(coeffs, 0)

    # Warm up root finding and collapse
    dummy_poly = np.array([1.0, -0.5, -1.0])
    find_roots(dummy_poly)
    dummy_2d = make_circle()
    _collapse_2d(dummy_2d, 0, 0.5)
    _eval_gradient_2d(dummy_2d, np.array([0.5, 0.5]))
    dummy_3d = make_sphere()
    _collapse_3d(dummy_3d, 0, np.array([0.5, 0.5]))
    _eval_gradient_3d(dummy_3d, np.array([0.5, 0.5, 0.5]))
    print(" done.")

    # --- Profile root finding by degree ---
    print("\nProfiling root finding by degree...", flush=True)
    root_times = profile_root_finding_by_degree(N_REPEAT)
    print_root_finding_table(root_times)

    # --- Profile each geometry ---
    all_results = []
    for name, coeffs in geometries:
        print(f"\nProfiling {name}...", flush=True)
        r = profile_geometry(name, coeffs)
        print_results(r, cpp_data)
        all_results.append(r)

    print_summary_table(all_results, cpp_data)
    print_optimization_recommendations(all_results, cpp_data)


if __name__ == "__main__":
    main()
