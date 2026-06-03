"""Benchmark: cell_quadrature vs. a Python loop over per-cell reference_map.

Usage::

    conda run -n pantr python benchmarks/bench_grid_cell_quadrature.py

Builds 2D/3D uniform grids and times the vectorized ``cell_quadrature`` bridge
against a Python loop that maps the reference rule with ``Grid.reference_map``
per cell. Prints a table of loop_ms vs. vec_ms with speedup, for a range of grid
sizes.
"""

from __future__ import annotations

import time

import numpy as np

from pantr.grid import cell_quadrature, uniform_grid
from pantr.quad import QuadratureRule, gauss_legendre_quadrature

_N_REPEATS = 5


def _time_loop(grid, rule: QuadratureRule) -> float:  # noqa: ANN001
    """Time a Python loop over per-cell reference_map; return min wall time (ms)."""
    best = float("inf")
    for _ in range(_N_REPEATS):
        t0 = time.perf_counter()
        pts = np.empty((grid.num_cells, rule.num_points, grid.ndim), dtype=np.float64)
        w = np.empty((grid.num_cells, rule.num_points), dtype=np.float64)
        for cid in range(grid.num_cells):
            lo, hi = grid.cell_bounds(cid)
            pts[cid] = grid.reference_map(cid)(rule.points)
            w[cid] = rule.weights * float(np.prod(hi - lo))
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def _time_vectorized(grid, rule: QuadratureRule) -> float:  # noqa: ANN001
    """Time cell_quadrature; return min wall time (ms)."""
    best = float("inf")
    for _ in range(_N_REPEATS):
        t0 = time.perf_counter()
        cell_quadrature(grid, rule)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main() -> None:
    """Run the cell-quadrature benchmark and print results."""
    rule = gauss_legendre_quadrature(2, 3)
    print(f"2D, GL {rule.num_points}-pt rule")
    print(f"{'cells':>9}  {'loop_ms':>10}  {'vec_ms':>10}  {'speedup':>8}")
    print("-" * 44)
    for n in (8, 32, 128, 512):
        grid = uniform_grid([[0.0, 1.0], [0.0, 1.0]], n)
        loop_ms = _time_loop(grid, rule)
        vec_ms = _time_vectorized(grid, rule)
        speedup = loop_ms / vec_ms if vec_ms > 0 else float("nan")
        print(f"{grid.num_cells:>9}  {loop_ms:>10.3f}  {vec_ms:>10.3f}  {speedup:>8.2f}x")

    rule3 = gauss_legendre_quadrature(3, 3)
    print(f"\n3D, GL {rule3.num_points}-pt rule")
    print(f"{'cells':>9}  {'loop_ms':>10}  {'vec_ms':>10}  {'speedup':>8}")
    print("-" * 44)
    for n in (8, 16, 32, 64):
        grid = uniform_grid([[0.0, 1.0]] * 3, n)
        loop_ms = _time_loop(grid, rule3)
        vec_ms = _time_vectorized(grid, rule3)
        speedup = loop_ms / vec_ms if vec_ms > 0 else float("nan")
        print(f"{grid.num_cells:>9}  {loop_ms:>10.3f}  {vec_ms:>10.3f}  {speedup:>8.2f}x")


if __name__ == "__main__":
    main()
