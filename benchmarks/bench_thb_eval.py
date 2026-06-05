"""Benchmark: THBSplineSpace per-cell evaluation (FEM-assembly hot path).

Usage::

    conda run -n pantr python benchmarks/bench_thb_eval.py

Builds refined 2D and 3D truncated-hierarchical spaces and times an assembly-like
sweep — :meth:`THBSplineSpace.tabulate_basis` (and a first-derivative sweep) at a few
points on every active cell — reporting microseconds per cell.  The 1D basis evaluation
is Numba-jitted; this measures the THB combine/orchestration cost targeted by the
``_combine_tp_values`` kernel, the vectorized truncated column, and the
``_cell_contributions`` cache.  Run with ``NUMBA_DISABLE_JIT=1`` to see the pure-Python
cost instead.
"""

from __future__ import annotations

import time

import numpy as np

from pantr.bspline import THBSplineSpace, create_uniform_space
from pantr.grid import hierarchical_grid, uniform_grid

_N_REPEATS = 6


def _build(dim: int, n: int, degree: int) -> THBSplineSpace:
    """Build a two-band refined THB space: ``dim``-D, ``n`` root cells/axis, given degree."""
    root = create_uniform_space([degree] * dim, [n] * dim)
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * dim, n), 2)
    grid.refine(0, [n // 4] * dim, [3 * n // 4] * dim)
    grid.refine(1, [3 * n // 4] * dim, [5 * n // 4] * dim)
    return THBSplineSpace(root, grid)


def _sweep(thb: THBSplineSpace, xi: np.ndarray, orders: tuple[int, ...] | None) -> None:
    """Evaluate ``tabulate_basis`` (or derivatives) at ``xi`` on every active cell."""
    for cid in range(thb.grid.num_cells):
        lo, hi = thb.grid.cell_bounds(cid)
        pts = lo + (hi - lo) * xi
        if orders is None:
            thb.tabulate_basis(cid, pts)
        else:
            thb.tabulate_basis_derivatives(cid, pts, orders)


def _best_ms(thb: THBSplineSpace, xi: np.ndarray, orders: tuple[int, ...] | None) -> float:
    """Minimum wall time (ms) of the sweep over ``_N_REPEATS`` runs."""
    best = float("inf")
    for _ in range(_N_REPEATS):
        t0 = time.perf_counter()
        _sweep(thb, xi, orders)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main() -> None:
    """Run the benchmark and print a per-cell timing table."""
    print(f"{'case':>16}  {'cells':>6}  {'values_us':>10}  {'deriv_us':>10}")
    print("-" * 50)
    for dim, n, degree in [(2, 16, 2), (2, 16, 3), (3, 8, 2)]:
        thb = _build(dim, n, degree)
        nc = thb.grid.num_cells
        xi = np.full((4, dim), 0.3)
        xi[1:] += np.linspace(0.1, 0.4, 3)[:, None]
        deriv_orders = (1,) + (0,) * (dim - 1)
        _sweep(thb, xi, None)  # warm up the JIT / cache
        _sweep(thb, xi, deriv_orders)
        values_us = _best_ms(thb, xi, None) * 1e3 / nc
        deriv_us = _best_ms(thb, xi, deriv_orders) * 1e3 / nc
        print(f"{f'{dim}D deg{degree} n{n}':>16}  {nc:>6}  {values_us:>10.1f}  {deriv_us:>10.1f}")


if __name__ == "__main__":
    main()
