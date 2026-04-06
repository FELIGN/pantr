"""Profile the implicit quadrature build and construction phases.

Measures timing for:
  - Build phase (ImplicitPolyQuadrature constructor)
  - Construction phase (volume_quad / surface_quad)
  - Detailed sub-function breakdown

Test geometries (from Saye, JCP 2022):
  2D: circle, ellipse (single cell), bilinear (eps=0.1)
  3D: sphere, ellipsoid (single cell), trilinear tunnel

Reference C++ timings from Table 1 (Intel Xeon E3-1535m v6, ~4 GHz):
  2D ellipse/cell:   build=0.2 µs,  q=1:0.1  q=2:0.2  q=4:0.5  q=8:1.0  q=16:2.5
  3D ellipsoid/cell:  build=1.5 µs,  q=1:0.2  q=2:0.8  q=4:2.2  q=8:8.8  q=16:50
  2D random/inst:     build=0.4 µs,  q=1:0.2  q=2:0.3  q=4:0.6  q=8:1.2  q=16:2.9
  3D random/inst:     build=6 µs,    q=1:0.9  q=2:2.2  q=4:8.0  q=8:25   q=16:120
"""

from __future__ import annotations

import sys
import time
from math import comb
from typing import Any

import numpy as np
from numba.typed import List as NumbaList
from numpy import typing as npt

from pantr.bezier.implicit import ImplicitPolyQuadrature, QuadStrategy
from pantr.bezier.implicit._build import build_2d, build_3d
from pantr.bezier.implicit._construct import (
    _collect_and_partition_1d,
)
from pantr.bezier.implicit._mask import compute_nonzero_mask_2d, compute_nonzero_mask_3d
from pantr.bezier.implicit._score import score_estimate_2d, score_estimate_3d

# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

Q_VALUES = [1, 2, 4, 8, 16]
N_WARMUP = 2
N_REPEAT = 50


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


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------


def _time_ns(func: Any, *args: Any, **kwargs: Any) -> tuple[float, Any]:
    """Return (elapsed_ns, result)."""
    t0 = time.perf_counter_ns()
    result = func(*args, **kwargs)
    t1 = time.perf_counter_ns()
    return float(t1 - t0), result


def _median_time_us(func: Any, n_warmup: int, n_repeat: int, *args: Any, **kwargs: Any) -> float:
    """Return median time in microseconds."""
    for _ in range(n_warmup):
        func(*args, **kwargs)
    times = []
    for _ in range(n_repeat):
        t, _ = _time_ns(func, *args, **kwargs)
        times.append(t / 1e3)  # ns -> µs
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Build-phase sub-function profiling
# ---------------------------------------------------------------------------


def profile_build_breakdown_2d(
    coeffs: npt.NDArray[np.float64], n_repeat: int = N_REPEAT
) -> dict[str, float]:
    """Break down 2D build into: mask, score, eliminate_axis, base_partition."""
    results: dict[str, float] = {}

    coeffs_list = NumbaList()
    coeffs_list.append(coeffs)

    # 1. Mask computation
    times = []
    for _ in range(n_repeat):
        t, _ = _time_ns(compute_nonzero_mask_2d, coeffs)
        times.append(t / 1e3)
    results["mask"] = float(np.median(times))

    masks_list = NumbaList()
    masks_list.append(compute_nonzero_mask_2d(coeffs))

    # 2. Score estimation
    times = []
    for _ in range(n_repeat):
        t, _ = _time_ns(score_estimate_2d, coeffs_list, masks_list)
        times.append(t / 1e3)
    results["score"] = float(np.median(times))

    # 3. Full build (includes all of the above)
    times = []
    for _ in range(n_repeat):
        t, br = _time_ns(build_2d, coeffs_list, masks_list)
        times.append(t / 1e3)
    results["build_2d (total Numba)"] = float(np.median(times))

    # 4. Base partition (root finding on 1D resultants)
    coeffs_1d = br[0]
    masks_1d = br[1]
    if len(coeffs_1d) > 0:
        times = []
        for _ in range(n_repeat):
            t, _ = _time_ns(_collect_and_partition_1d, coeffs_1d, masks_1d)
            times.append(t / 1e3)
        results["base_partition (roots)"] = float(np.median(times))
    else:
        results["base_partition (roots)"] = 0.0

    return results


def profile_build_breakdown_3d(
    coeffs: npt.NDArray[np.float64], n_repeat: int = N_REPEAT
) -> dict[str, float]:
    """Break down 3D build into: mask, score, build, base_partition."""
    results: dict[str, float] = {}

    coeffs_list = NumbaList()
    coeffs_list.append(coeffs)

    # 1. Mask computation
    times = []
    for _ in range(n_repeat):
        t, _ = _time_ns(compute_nonzero_mask_3d, coeffs)
        times.append(t / 1e3)
    results["mask"] = float(np.median(times))

    masks_list = NumbaList()
    masks_list.append(compute_nonzero_mask_3d(coeffs))

    # 2. Score estimation
    times = []
    for _ in range(n_repeat):
        t, _ = _time_ns(score_estimate_3d, coeffs_list, masks_list)
        times.append(t / 1e3)
    results["score"] = float(np.median(times))

    # 3. Full build
    times = []
    for _ in range(n_repeat):
        t, br = _time_ns(build_3d, coeffs_list, masks_list)
        times.append(t / 1e3)
    results["build_3d (total Numba)"] = float(np.median(times))

    # 4. Base partition
    coeffs_1d = br[0]
    masks_1d = br[1]
    if len(coeffs_1d) > 0:
        times = []
        for _ in range(n_repeat):
            t, _ = _time_ns(_collect_and_partition_1d, coeffs_1d, masks_1d)
            times.append(t / 1e3)
        results["base_partition (roots)"] = float(np.median(times))
    else:
        results["base_partition (roots)"] = 0.0

    return results


# ---------------------------------------------------------------------------
# High-level profiling
# ---------------------------------------------------------------------------


def profile_geometry(
    name: str,
    coeffs: npt.NDArray[np.float64],
    q_values: list[int] = Q_VALUES,
    n_warmup: int = N_WARMUP,
    n_repeat: int = N_REPEAT,
) -> dict[str, Any]:
    """Profile build + construction for one geometry."""
    dim = coeffs.ndim
    results: dict[str, Any] = {"name": name, "dim": dim, "shape": coeffs.shape}

    # --- Build phase ---
    # Warmup (triggers JIT compilation)
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
    if dim == 2:
        results["build_breakdown"] = profile_build_breakdown_2d(coeffs, n_repeat)
    else:
        results["build_breakdown"] = profile_build_breakdown_3d(coeffs, n_repeat)

    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

# Reference C++ times from Table 1 (Saye, JCP 2022), in microseconds.
ALGOIM_REF = {
    "ellipse": {"build": 0.2, 1: 0.1, 2: 0.2, 4: 0.5, 8: 1.0, 16: 2.5},
    "ellipsoid": {"build": 1.5, 1: 0.2, 2: 0.8, 4: 2.2, 8: 8.8, 16: 50},
}


def print_results(results: dict[str, Any]) -> None:
    name = results["name"]
    dim = results["dim"]
    print(f"\n{'=' * 80}")
    print(f"  {name}  (dim={dim}, shape={results['shape']})")
    print(f"{'=' * 80}")

    print(f"\n  Build phase: {results['build_us']:.1f} µs")

    # Build breakdown
    bd = results.get("build_breakdown", {})
    if bd:
        print("  Build breakdown:")
        for k, v in bd.items():
            print(f"    {k:30s}: {v:8.1f} µs")

    # Volume quad
    print(f"\n  {'Volume quadrature':40s}")
    print(f"  {'q':>4s} {'time (µs)':>12s} {'n_pts':>8s}", end="")

    ref_key = None
    if "ellipse" in name.lower() and dim == 2:
        ref_key = "ellipse"
    elif "ellipsoid" in name.lower() and dim == 3:
        ref_key = "ellipsoid"
    if ref_key:
        print(f" {'C++ ref (µs)':>14s} {'ratio':>8s}", end="")
    print()

    print(f"  {'-' * 50}")
    for q in sorted(results["volume_us"]):
        t = results["volume_us"][q]
        n = results["volume_npts"][q]
        print(f"  {q:4d} {t:12.1f} {n:8d}", end="")
        if ref_key and q in ALGOIM_REF[ref_key]:
            ref_t = ALGOIM_REF[ref_key][q]
            print(f" {ref_t:14.1f} {t / ref_t:8.1f}x", end="")
        print()

    # Surface quad
    print(f"\n  {'Surface quadrature':40s}")
    print(f"  {'q':>4s} {'time (µs)':>12s} {'n_pts':>8s}")
    print(f"  {'-' * 50}")
    for q in sorted(results["surface_us"]):
        t = results["surface_us"][q]
        n = results["surface_npts"][q]
        print(f"  {q:4d} {t:12.1f} {n:8d}")


def print_summary_table(all_results: list[dict[str, Any]]) -> None:
    print(f"\n\n{'=' * 100}")
    print("  SUMMARY TABLE (all times in µs)")
    print(f"{'=' * 100}")

    # Header
    header = f"  {'Geometry':30s} {'dim':>4s} {'build':>8s}"
    for q in Q_VALUES:
        header += f" {'vol q=' + str(q):>10s}"
    for q in Q_VALUES:
        header += f" {'srf q=' + str(q):>10s}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for r in all_results:
        line = f"  {r['name']:30s} {r['dim']:4d} {r['build_us']:8.1f}"
        for q in Q_VALUES:
            line += f" {r['volume_us'].get(q, 0):10.1f}"
        for q in Q_VALUES:
            line += f" {r['surface_us'].get(q, 0):10.1f}"
        print(line)

    # Print C++ reference
    print(f"\n  {'C++ ref (ellipse)':30s} {'2':>4s} {ALGOIM_REF['ellipse']['build']:8.1f}", end="")
    for q in Q_VALUES:
        print(f" {ALGOIM_REF['ellipse'].get(q, 0):10.1f}", end="")
    print()
    print(
        f"  {'C++ ref (ellipsoid)':30s} {'3':>4s} {ALGOIM_REF['ellipsoid']['build']:8.1f}", end=""
    )
    for q in Q_VALUES:
        print(f" {ALGOIM_REF['ellipsoid'].get(q, 0):10.1f}", end="")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Implicit quadrature profiling")
    print(f"Python {sys.version}")
    print(f"NumPy {np.__version__}")
    try:
        import numba

        print(f"Numba {numba.__version__}")
    except ImportError:
        pass

    print(f"\nN_WARMUP={N_WARMUP}, N_REPEAT={N_REPEAT}")
    print(f"q values: {Q_VALUES}")

    geometries = [
        ("circle (2D)", make_circle()),
        ("ellipse (2D)", make_ellipse()),
        ("bilinear eps=0.1 (2D)", make_bilinear(0.1)),
        ("sphere (3D)", make_sphere()),
        ("ellipsoid (3D)", make_ellipsoid()),
        ("trilinear tunnel (3D)", make_trilinear_tunnel()),
    ]

    # First pass: JIT warmup
    print("\nWarming up JIT...", end="", flush=True)
    for _, coeffs in geometries:
        ipq = ImplicitPolyQuadrature(coeffs)
        ipq.volume_quad(2, QuadStrategy.AUTO_MIXED)
        ipq.surface_quad(2, QuadStrategy.AUTO_MIXED)
    print(" done.")

    # Profile each geometry
    all_results = []
    for name, coeffs in geometries:
        print(f"\nProfiling {name}...", flush=True)
        r = profile_geometry(name, coeffs)
        print_results(r)
        all_results.append(r)

    print_summary_table(all_results)


if __name__ == "__main__":
    main()
