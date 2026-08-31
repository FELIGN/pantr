#!/usr/bin/env python
"""Measure what a hierarchical refinement costs, and what it does *not* scale with.

Run it as::

    python scripts/bench_grid_refine.py
    python scripts/bench_grid_refine.py --repeats 40

Why this script exists
----------------------

Issue #378 changed :meth:`~pantr.grid.HierarchicalGrid.refine` and its three
siblings from mutating the receiver to returning a new grid, and the ticket's
binding invariant was that no refinement sweep may end up copying the whole grid
per step. That is a claim about *scaling*, not about a single number, so this
script measures the two scalings that settle it rather than reporting a timing:

``cells``
    One ``refine`` promoting a single-block region, on grids from 64 to 4096
    cells. The block count is held at 1 while the cell count grows 64-fold. A cost
    that tracked the cell count would show up here as a 64-fold rise.

``blocks``
    One ``refine`` on a grid whose block count has been driven up by a checkerboard
    of prior refinements. Here the block count is what grows.

``deep``
    One ``refine`` at level 0 on a staircase that leaves blocks on *every* level.
    This is the case the value-returning implementation could plausibly regress:
    it hands its whole per-level block list to ``_from_blocks``, which normalizes
    every level, including the ones the call never touched, where the mutating
    version normalized only the one or two it did.

``sweep``
    A whole adaptive loop, N successive local ``refine_cells`` steps: what an
    adaptive algorithm actually pays.

Reading the numbers
-------------------

**CPU time, and the minimum over repeats.** These are sub-millisecond,
single-threaded, pure-Python-plus-small-NumPy measurements. On a shared machine
wall time measures the scheduler, so ``time.process_time`` is used, and the
minimum is reported rather than the mean because the fastest run is the one least
interfered with. A figure that breaks the trend should be read as noise until more
repeats say otherwise.

**The cost is the block-list normalization, and it always was.** The grid stores no
per-cell data at all -- only ``(lo, hi)`` rectangles per level -- so neither the
mutating nor the value-returning call can be ``O(cells)``. What both pay is the
greedy merge in ``_normalize_blocks``, which is superlinear in the number of
blocks at a level. Expect the ``cells`` sweep to be flat and the ``blocks`` sweep
to rise steeply; that shape, not any individual number, is the result.

**To compare against the mutating implementation**, check out the commit before
#378 into a second tree and run this same script against it with ``PYTHONPATH``
pointing there. The script adapts to either API, so the two runs measure the same
work.
"""

from __future__ import annotations

import argparse
import gc
import time
from typing import TYPE_CHECKING, Any, Final, NamedTuple

from pantr.grid import hierarchical_grid, uniform_grid

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pantr.grid import HierarchicalGrid

_DEFAULT_REPEATS: Final[int] = 25
"""
Repeats per configuration, for the minimum to be a stable estimator.

Each repeat rebuilds the fixture, so the cost is a few seconds in total. Twenty-five
is enough for the minimum to stop moving between runs on a loaded 20-CPU host, which
is the regime this was written in; it is not a tuned number.
"""


class Row(NamedTuple):
    """One measured configuration.

    Attributes:
        label (str): What was varied, for the printed table.
        num_cells (int): Active cells in the fixture the operation was applied to.
        total_blocks (int): Active blocks summed over every level of that fixture.
        untouched_blocks (int): Blocks on levels the timed operation does not touch.
        micros (float): Minimum CPU time for one operation, in microseconds.
    """

    label: str
    num_cells: int
    total_blocks: int
    untouched_blocks: int
    micros: float


def _apply(grid: HierarchicalGrid, name: str, args: Sequence[Any]) -> HierarchicalGrid:
    """Apply one hierarchy operation, tolerating either the old or the new API.

    The pre-#378 methods returned ``None`` and mutated the receiver; the current ones
    return a new grid. Accepting both is what lets one script measure the same work in
    a checkout on either side of that change.

    Args:
        grid (HierarchicalGrid): The grid to operate on.
        name (str): Method name: ``refine``, ``refine_cells``, ``coarsen`` or
            ``coarsen_cells``.
        args (Sequence[Any]): Positional arguments for the method.

    Returns:
        HierarchicalGrid: The operation's result, or ``grid`` itself if the method
        mutated it and returned ``None``.
    """
    result = getattr(grid, name)(*args)
    return grid if result is None else result


def _total_blocks(grid: HierarchicalGrid) -> int:
    """Return the number of active blocks summed over every level.

    Args:
        grid (HierarchicalGrid): The grid to count.

    Returns:
        int: Total block count.
    """
    return sum(len(grid.active_blocks(level)) for level in range(grid.max_level + 1))


def _min_micros(
    build: Callable[[], HierarchicalGrid],
    name: str,
    args: Sequence[Any],
    repeats: int,
) -> float:
    """Return the minimum CPU time of one operation, in microseconds.

    Args:
        build (Callable[[], HierarchicalGrid]): Builds a fresh fixture per repeat, so
            no repeat inherits the previous one's caches.
        name (str): Method name to time.
        args (Sequence[Any]): Positional arguments for the method.
        repeats (int): Number of repeats.

    Returns:
        float: Minimum CPU microseconds over the repeats.
    """
    best = float("inf")
    for _ in range(repeats):
        grid = build()
        gc.collect()
        start = time.process_time()
        _apply(grid, name, args)
        best = min(best, time.process_time() - start)
    return best * 1e6


def _plain(cells_per_axis: int) -> HierarchicalGrid:
    """Return an unrefined 2D grid: one block, ``cells_per_axis ** 2`` cells.

    Args:
        cells_per_axis (int): Root cells per axis.

    Returns:
        HierarchicalGrid: The unrefined grid.
    """
    return hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], cells_per_axis), 2)


def _checkerboard(cells_per_axis: int) -> HierarchicalGrid:
    """Return a 2D grid whose block count has been driven up by isolated refinements.

    Refining every other cell forces the peeled remainder into many non-mergeable
    blocks, which is what makes the block count grow with the root size.

    Args:
        cells_per_axis (int): Root cells per axis.

    Returns:
        HierarchicalGrid: The block-rich grid.
    """
    grid = _plain(cells_per_axis)
    for i in range(0, cells_per_axis, 2):
        for j in range(0, cells_per_axis, 2):
            grid = _apply(grid, "refine", (0, (i, j), (i + 1, j + 1)))
    return grid


def _staircase(levels: int) -> HierarchicalGrid:
    """Return a 2D grid leaving blocks behind on every level from 0 to ``levels``.

    The case where normalizing every level rather than only the touched ones could
    cost more than the mutating implementation did.

    Args:
        levels (int): Number of refinement rounds, one per level.

    Returns:
        HierarchicalGrid: The deep grid.
    """
    grid = _plain(8)
    for level in range(levels):
        width = 8 * 2**level
        for i in range(0, min(width, 12), 2):
            grid = _apply(grid, "refine", (level, (i, 0), (i + 1, 1)))
    return grid


def _sweep_micros(steps: int, repeats: int) -> tuple[float, HierarchicalGrid]:
    """Time a whole adaptive loop of ``steps`` local refinements.

    Args:
        steps (int): Number of ``refine_cells`` steps.
        repeats (int): Number of repeats.

    Returns:
        tuple[float, HierarchicalGrid]: Minimum CPU microseconds for the whole loop,
        and the final grid of the last repeat, so the caller can report what was built.
    """
    best = float("inf")
    final = _plain(16)
    for _ in range(repeats):
        gc.collect()
        start = time.process_time()
        grid = _plain(16)
        for _step in range(steps):
            deepest = grid.max_level
            ids = [c for c in range(grid.num_cells) if grid.cell_level(c) == deepest][:4]
            if not ids:
                break
            grid = _apply(grid, "refine_cells", (ids,))
        best = min(best, time.process_time() - start)
        final = grid
    return best * 1e6, final


def _print_table(title: str, note: str, rows: Sequence[Row]) -> None:
    """Print one measured sweep.

    Args:
        title (str): Sweep name.
        note (str): What the reader should conclude from the shape.
        rows (Sequence[Row]): The measured rows.
    """
    print(f"\n{title}")
    print(f"  {note}")
    print(f"  {'fixture':>16} {'cells':>8} {'blocks':>8} {'untouched':>10} {'us/call':>10}")
    for row in rows:
        print(
            f"  {row.label:>16} {row.num_cells:>8} {row.total_blocks:>8} "
            f"{row.untouched_blocks:>10} {row.micros:>10.1f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the four sweeps and print them.

    Args:
        argv (Sequence[str] | None): Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        int: Process exit status, always ``0``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats",
        type=int,
        default=_DEFAULT_REPEATS,
        help=f"repeats per configuration (default {_DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--sweep-steps", type=int, default=24, help="refinement steps in the adaptive loop"
    )
    args = parser.parse_args(argv)

    rows = []
    for n in (8, 16, 32, 64):
        grid = _plain(n)
        rows.append(
            Row(
                f"root {n}x{n}",
                grid.num_cells,
                _total_blocks(grid),
                0,
                _min_micros(lambda n=n: _plain(n), "refine", (0, (0, 0), (n, n)), args.repeats),  # type: ignore[misc]
            )
        )
    _print_table(
        "cells -- one refine of a single-block region",
        "flat means the cost does not track the cell count",
        rows,
    )

    rows = []
    for n in (8, 12, 16, 20):
        grid = _checkerboard(n)
        rows.append(
            Row(
                f"root {n}x{n}",
                grid.num_cells,
                _total_blocks(grid),
                0,
                _min_micros(
                    lambda n=n: _checkerboard(n),  # type: ignore[misc]
                    "refine",
                    (0, (1, 1), (2, 2)),
                    args.repeats,
                ),
            )
        )
    _print_table(
        "blocks -- one refine on a block-rich grid",
        "rises steeply: the greedy merge in _normalize_blocks is what a refine pays",
        rows,
    )

    rows = []
    for levels in (3, 5, 7):
        grid = _staircase(levels)
        per_level = [len(grid.active_blocks(lv)) for lv in range(grid.max_level + 1)]
        rows.append(
            Row(
                f"{levels} levels",
                grid.num_cells,
                sum(per_level),
                sum(per_level[2:]),
                _min_micros(
                    lambda levels=levels: _staircase(levels),  # type: ignore[misc]
                    "refine",
                    (0, (7, 7), (8, 8)),
                    args.repeats,
                ),
            )
        )
    _print_table(
        "deep -- one refine at level 0, with blocks on every level",
        "the untouched-level normalization the value-returning form adds",
        rows,
    )

    micros, final = _sweep_micros(args.sweep_steps, max(args.repeats // 2, 3))
    print(f"\nsweep -- {args.sweep_steps} successive refine_cells steps")
    print(
        f"  {micros / 1e3:.2f} ms for the whole loop; final grid: "
        f"{final.num_cells} cells, max_level {final.max_level}, "
        f"{_total_blocks(final)} blocks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
