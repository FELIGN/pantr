"""Benchmark: TensorProductGrid memory footprint vs. cell count.

Usage::

    conda run -n pantr python benchmarks/bench_grid_footprint.py

Demonstrates that a :class:`pantr.grid.TensorProductGrid` stores only its
per-axis breakpoint arrays (plus a little metadata) -- a payload proportional to
``sum_k (cells_per_axis[k] + 1)`` -- and never materializes per-cell data until
a spatial query lazily builds the BVH. The reported "grid payload" stays tiny as
the cell count grows by orders of magnitude; for contrast, the table also shows
the size of a densely materialized ``(num_cells, ndim)`` bounds array and the
lazily-built BVH.
"""

from __future__ import annotations

import numpy as np

from pantr.geometry import AABB
from pantr.grid import TensorProductGrid, uniform_grid

_CASES = [
    (2, 10),
    (2, 100),
    (2, 1000),
    (3, 10),
    (3, 50),
    (3, 100),
]


def _grid_payload_bytes(g: TensorProductGrid) -> int:
    """Sum the bytes retained by the grid's stored arrays."""
    total = sum(bp.nbytes for bp in g.breakpoints)
    total += g.bounds.nbytes
    total += g._strides.nbytes
    return total


def _bvh_bytes(g: TensorProductGrid) -> int:
    """Sum the bytes of the (lazily built) BVH node arrays."""
    bvh = g.cell_bvh()
    return int(
        bvh.node_lo.nbytes
        + bvh.node_hi.nbytes
        + bvh.node_left.nbytes
        + bvh.node_right.nbytes
        + bvh.node_cell.nbytes
    )


def main() -> None:
    """Run the footprint benchmark and print a table."""
    print(
        f"{'ndim':>4}  {'cells/axis':>10}  {'num_cells':>11}  "
        f"{'grid B':>9}  {'dense B':>12}  {'bvh B':>12}  {'grid/dense':>10}"
    )
    print("-" * 80)
    for ndim, n in _CASES:
        g = uniform_grid([[0.0, 1.0]] * ndim, n)
        num_cells = g.num_cells
        grid_b = _grid_payload_bytes(g)
        dense_b = num_cells * ndim * 8  # float64 (num_cells, ndim) bounds (one corner)
        assert g._bvh is None
        # Touch a query to force the lazy BVH, then measure it.
        g.query_aabb(AABB(np.zeros(ndim), np.full(ndim, 0.1)))
        bvh_b = _bvh_bytes(g)
        ratio = grid_b / dense_b if dense_b else float("nan")
        print(
            f"{ndim:>4}  {n:>10}  {num_cells:>11}  {grid_b:>9}  "
            f"{dense_b:>12}  {bvh_b:>12}  {ratio:>10.2e}"
        )
    print()
    print("grid payload grows with sum(cells_per_axis), NOT num_cells;")
    print("the O(num_cells) BVH is built only on the first query_aabb.")


if __name__ == "__main__":
    main()
