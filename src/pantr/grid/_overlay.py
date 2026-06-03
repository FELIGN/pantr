"""Overlay of two tensor-product grids.

:func:`overlay` takes two :class:`TensorProductGrid` instances and returns a
third whose per-axis breakpoints are the union of both inputs' breakpoints,
restricted to the intersection of their domains. The overlay is the coarsest
tensor-product grid that simultaneously refines both inputs: every overlay cell
is contained in exactly one cell of each input.

This is the background-grid bridge for immersed / unfitted quadrature. When a
B-spline (one knot structure) is integrated over an immersion grid (a different
structure), each overlay cell falls inside one polynomial piece of the B-spline
*and* one immersion cell, so a Bernstein patch can be reused without further
subdivision.

The operation is symmetric -- ``overlay(a, b)`` and ``overlay(b, a)`` produce
the same breakpoints up to floating-point noise -- and is defined for any
``ndim >= 1``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..tolerance import get_default
from ._tensor_product_grid import TensorProductGrid

if TYPE_CHECKING:
    import numpy.typing as npt


def overlay(grid_a: TensorProductGrid, grid_b: TensorProductGrid) -> TensorProductGrid:
    """Return the tensor-product overlay of ``grid_a`` and ``grid_b``.

    The overlay's per-axis breakpoints are the sorted union of both inputs'
    breakpoint arrays restricted to the intersection of their domains.
    Breakpoints closer than the default ``float64`` tolerance
    (:func:`pantr.tolerance.get_default`) are merged into one. The result is the
    coarsest :class:`TensorProductGrid` that refines both inputs.

    Args:
        grid_a (TensorProductGrid): First input grid.
        grid_b (TensorProductGrid): Second input grid; must share
            :attr:`~TensorProductGrid.ndim` with ``grid_a`` and have a non-empty
            domain intersection on every axis.

    Returns:
        TensorProductGrid: The overlay grid.

    Raises:
        TypeError: If either argument is not a :class:`TensorProductGrid`.
        ValueError: If the grids have different
            :attr:`~TensorProductGrid.ndim`, or if their domains do not overlap
            on some axis (the per-axis intersection is empty or degenerate).

    Example:
        >>> import numpy.testing as npt
        >>> from pantr.grid import uniform_grid, overlay
        >>> a = uniform_grid([[0.0, 1.0]], 2)
        >>> b = uniform_grid([[0.0, 1.0]], 3)
        >>> npt.assert_allclose(
        ...     overlay(a, b).breakpoints[0],
        ...     [0.0, 1/3, 0.5, 2/3, 1.0],
        ... )
    """
    if not isinstance(grid_a, TensorProductGrid) or not isinstance(grid_b, TensorProductGrid):
        raise TypeError(
            "overlay() requires two TensorProductGrid instances; got "
            f"{type(grid_a).__name__!r} and {type(grid_b).__name__!r}."
        )
    if grid_a.ndim != grid_b.ndim:
        raise ValueError(f"overlay(): grids must share ndim; got {grid_a.ndim} vs {grid_b.ndim}.")
    atol = get_default(np.float64)
    merged: list[npt.NDArray[np.float64]] = []
    for d in range(grid_a.ndim):
        ba = grid_a.breakpoints[d]
        bb = grid_b.breakpoints[d]
        lo = max(float(ba[0]), float(bb[0]))
        hi = min(float(ba[-1]), float(bb[-1]))
        if hi - lo <= atol:
            raise ValueError(
                f"overlay(): domains do not overlap on axis {d}; grid_a extent "
                f"[{ba[0]}, {ba[-1]}] vs grid_b extent [{bb[0]}, {bb[-1]}]."
            )
        merged.append(_merge_axis_breakpoints(ba, bb, lo, hi, atol))
    return TensorProductGrid(merged)


def _merge_axis_breakpoints(
    ba: npt.NDArray[np.float64],
    bb: npt.NDArray[np.float64],
    lo: float,
    hi: float,
    atol: float,
) -> npt.NDArray[np.float64]:
    """Merge two 1-D breakpoint arrays into their sorted, deduplicated union.

    Only breakpoints strictly inside ``(lo, hi)`` participate; the ``lo`` / ``hi``
    intersection bounds are always emitted as the first and last entries.
    Breakpoints closer than ``atol`` are folded into a single entry.

    Args:
        ba (npt.NDArray[np.float64]): First input's 1-D breakpoints.
        bb (npt.NDArray[np.float64]): Second input's 1-D breakpoints.
        lo (float): Lower intersection bound; always emitted first.
        hi (float): Upper intersection bound; always emitted last.
        atol (float): Absolute tolerance for merging near-coincident entries.

    Returns:
        npt.NDArray[np.float64]: Strictly increasing ``float64`` array starting
        with ``lo`` and ending with ``hi`` (at least two entries).
    """
    candidates = np.concatenate(
        [
            np.asarray([lo, hi], dtype=np.float64),
            ba[(ba > lo + atol) & (ba < hi - atol)],
            bb[(bb > lo + atol) & (bb < hi - atol)],
        ]
    )
    candidates.sort(kind="stable")
    keep = np.ones(candidates.shape[0], dtype=bool)
    keep[1:] = np.diff(candidates) > atol
    merged = candidates[keep]
    # Belt-and-suspenders: re-pin lo/hi so any float conversion noise in
    # the sort/concatenate path doesn't alter the caller-supplied bounds.
    merged[0] = lo
    merged[-1] = hi
    return np.ascontiguousarray(merged, dtype=np.float64)
