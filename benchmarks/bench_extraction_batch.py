"""Benchmark: apply_many vs. per-cell apply loop for SpanwiseElementExtraction.

Usage::

    conda run -n pantr python benchmarks/bench_extraction_batch.py

Constructs a 3D B-spline space (degree 3, 20 elements per direction) and
times ``apply_many`` against a Python loop over ``apply`` for a range of
batch sizes. Prints a table of loop_ms vs. batch_ms with speedup.
"""

from __future__ import annotations

import time

import numpy as np

from pantr.bspline import BsplineSpace, BsplineSpace1D, SpanwiseElementExtraction

_N_REPEATS = 5
_BATCH_SIZES = [1, 10, 50, 100, 500, 1000, 5000]


def _build_space() -> BsplineSpace:
    """3D space: degree 3, 20 elements per direction, ~50k DOFs."""
    knots = np.concatenate(
        [
            np.zeros(4),
            np.arange(1, 20),
            np.full(4, 20),
        ]
    ).astype(np.float64)
    sp1 = BsplineSpace1D(knots, 3)
    return BsplineSpace([sp1, sp1, sp1])


def _time_loop(
    ext: SpanwiseElementExtraction,
    v: np.ndarray,
    flat_idx: np.ndarray,
) -> float:
    """Time a Python loop over per-cell apply; return minimum wall time in ms."""
    n_cells = v.shape[0]
    best = float("inf")
    for _ in range(_N_REPEATS):
        t0 = time.perf_counter()
        out = np.empty_like(v[:, : int(np.prod(ext.output_shape_per_dir))])
        for c in range(n_cells):
            out[c] = ext.apply(v[c], int(flat_idx[c]))
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def _time_batch(
    ext: SpanwiseElementExtraction,
    v: np.ndarray,
    flat_idx: np.ndarray,
) -> float:
    """Time apply_many; return minimum wall time in ms."""
    best = float("inf")
    for _ in range(_N_REPEATS):
        t0 = time.perf_counter()
        ext.apply_many(v, flat_idx)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main() -> None:
    """Run the benchmark and print results."""
    sp = _build_space()
    ext = SpanwiseElementExtraction(sp, "bezier")
    n_in = int(np.prod(ext.input_shape_per_dir))
    total = ext.num_total_intervals

    print(f"Space: 3D, degree 3, {ext.num_intervals} intervals, {total} total cells")
    print(f"N_in per cell: {n_in},  N_repeats: {_N_REPEATS}")
    print()
    print(f"{'n_cells':>8}  {'loop_ms':>10}  {'batch_ms':>10}  {'speedup':>8}")
    print("-" * 44)

    rng = np.random.default_rng(0)
    for n_cells in _BATCH_SIZES:
        flat_idx = rng.integers(0, total, size=n_cells)
        v = rng.standard_normal((n_cells, n_in))

        loop_ms = _time_loop(ext, v, flat_idx)
        batch_ms = _time_batch(ext, v, flat_idx)
        speedup = loop_ms / batch_ms if batch_ms > 0 else float("nan")
        print(f"{n_cells:>8}  {loop_ms:>10.3f}  {batch_ms:>10.3f}  {speedup:>8.2f}x")


if __name__ == "__main__":
    main()
