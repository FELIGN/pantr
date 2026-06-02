"""Benchmark: TensorProductGrid.locate_many vs. a Python loop over locate.

Usage::

    conda run -n pantr python benchmarks/bench_grid_locate.py

Builds a 3D uniform grid and times the Numba batch ``locate_many`` kernel against
a Python loop calling ``locate`` per point, for a range of batch sizes. Prints a
table of loop_ms vs. batch_ms with speedup.
"""

from __future__ import annotations

import time

import numpy as np

from pantr.grid import uniform_grid

_N_REPEATS = 5
_BATCH_SIZES = [1, 100, 1000, 10_000, 100_000]


def _time_loop(grid, pts: np.ndarray) -> float:  # noqa: ANN001
    """Time a Python loop over per-point locate; return minimum wall time (ms)."""
    n = pts.shape[0]
    best = float("inf")
    for _ in range(_N_REPEATS):
        t0 = time.perf_counter()
        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            cid = grid.locate(pts[i])
            out[i] = -1 if cid is None else cid
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def _time_batch(grid, pts: np.ndarray) -> float:  # noqa: ANN001
    """Time locate_many; return minimum wall time (ms)."""
    best = float("inf")
    for _ in range(_N_REPEATS):
        t0 = time.perf_counter()
        grid.locate_many(pts)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main() -> None:
    """Run the locate benchmark and print results."""
    grid = uniform_grid([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], 50)
    print(f"Grid: 3D uniform, {grid.cells_per_axis} cells, {grid.num_cells} total")
    rng = np.random.default_rng(0)

    # Warm up the JIT kernel so compile time is excluded from the timings.
    grid.locate_many(rng.uniform(0.0, 1.0, size=(8, 3)))

    print(f"{'n_pts':>9}  {'loop_ms':>10}  {'batch_ms':>10}  {'speedup':>8}")
    print("-" * 44)
    for n_pts in _BATCH_SIZES:
        pts = rng.uniform(-0.1, 1.1, size=(n_pts, 3))
        loop_ms = _time_loop(grid, pts)
        batch_ms = _time_batch(grid, pts)
        speedup = loop_ms / batch_ms if batch_ms > 0 else float("nan")
        print(f"{n_pts:>9}  {loop_ms:>10.3f}  {batch_ms:>10.3f}  {speedup:>8.2f}x")


if __name__ == "__main__":
    main()
