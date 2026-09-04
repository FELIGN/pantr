"""Measure the footprint of the THB contribution table the C++ port fills eagerly.

``design/bspline_derived_caches.md`` rules that ``THBSplineSpace._contrib_cache`` --
today a ``dict[int, list[tuple]]`` filled one cell at a time -- becomes **one flat CSR
table filled for every cell behind a single double-checked flag**, and it makes that
ruling contingent:

    #397 owes a footprint measurement on a realistic adaptive hierarchy before it
    commits, because a lazy-all-cells fill trades memory for the dict's incrementality;
    the fallback if it is too large is per-level rather than per-cell granularity, not a
    return to per-cell.

This script is that measurement. It answers two questions.

**How large is the flat table, in bytes, on a hierarchy of realistic size?** The table is
``num_cells + 1`` offsets plus, per entry, one global dof, one level and ``dim``
function indices, all ``int64``:

    bytes = 8 * (num_cells + 1) + 8 * (2 + dim) * num_entries

There is nothing to measure in that formula; what has to be measured is
``num_entries``, which is the sum over cells of how many active functions each one
carries and is a property of the hierarchy rather than of the implementation.

**How does it compare with what the oracle's dict costs once every cell has been
touched?** That is the honest comparison, because ``max_active_per_cell()`` already
forces every cell in both implementations -- and any consumer that assembles over the
mesh touches them all too. The dict's cost is measured rather than derived: a Python
``int``, ``tuple`` and ``list`` each carry a header, and reasoning about them from
memory is how a factor of two gets lost. ``_deep_size`` walks the structure with
``sys.getsizeof``, counting each object once.

The cases below are adaptive rather than uniform: a corner refinement chain and a
diagonal band, in 2D and 3D, at the degrees this library is actually used at. A uniformly
refined hierarchy would be the easy case for both representations and would say nothing
about the one the ruling is about.

Run:
    python scripts/measure_thb_contribution_table.py
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Final, NamedTuple

import numpy as np

from pantr.bspline import BsplineSpace, BsplineSpace1D, THBSplineSpace
from pantr.grid import hierarchical_grid, uniform_grid

if TYPE_CHECKING:
    import numpy.typing as npt

_INT64_BYTES: Final = 8
"""Width of every field of the flat table. It is `int64` throughout, as the C++ is."""


class _Footprint(NamedTuple):
    """What one hierarchy costs in each representation.

    Attributes:
        label (str): What was built.
        dim (int): Parametric dimension.
        num_cells (int): Active cells in the hierarchy.
        num_entries (int): Total contribution entries over every cell.
        widest (int): The largest per-cell count, which is `max_active_per_cell`.
        flat_bytes (int): The C++ table's size, from the formula in the module docstring.
        dict_bytes (int): The oracle's fully populated dict, measured.
    """

    label: str
    dim: int
    num_cells: int
    num_entries: int
    widest: int
    flat_bytes: int
    dict_bytes: int


def _deep_size(obj: object, seen: set[int] | None = None) -> int:
    """The transitive size of a Python object, counting each object once.

    ``sys.getsizeof`` reports one object's own storage and not what it refers to, so a
    dict of lists of tuples reports as a dict of pointers. Walking it is what makes the
    comparison honest. Identity-keyed, because small integers and the level values repeat
    heavily and counting them once each is what the interpreter actually pays.

    Args:
        obj (object): The object to size.
        seen (set[int] | None): Object ids already counted, for the recursion.

    Returns:
        int: Bytes, transitively.
    """
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    total = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for key, value in obj.items():
            total += _deep_size(key, seen) + _deep_size(value, seen)
    elif isinstance(obj, list | tuple | set | frozenset):
        for item in obj:
            total += _deep_size(item, seen)
    return total


def _open_knots(degree: int, num_elements: int) -> npt.NDArray[np.float64]:
    """A clamped uniform knot vector on the unit interval.

    Args:
        degree (int): Polynomial degree.
        num_elements (int): Number of equal spans.

    Returns:
        npt.NDArray[np.float64]: The knot vector.
    """
    inner = np.linspace(0.0, 1.0, num_elements + 1)
    return np.concatenate([np.zeros(degree), inner, np.ones(degree)])


def _corner_chain(dim: int, degree: int, num_elements: int, levels: int) -> THBSplineSpace:
    """A hierarchy refined repeatedly into one corner.

    The shape an adaptive solver produces around a re-entrant corner, and the one where
    the per-level function counts are most uneven.

    Args:
        dim (int): Parametric dimension.
        degree (int): Polynomial degree, the same in every direction.
        num_elements (int): Root elements per direction.
        levels (int): How many successive corner refinements to apply.

    Returns:
        THBSplineSpace: The space.
    """
    direction = BsplineSpace1D(_open_knots(degree, num_elements), degree)
    root = BsplineSpace([direction] * dim)
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * dim, [num_elements] * dim), 2)
    extent = num_elements
    for level in range(levels):
        half = max(1, extent // 2)
        grid = grid.refine(level, [0] * dim, [half] * dim)
        extent = half * 2
    return THBSplineSpace(root, grid)


def _diagonal_band(dim: int, degree: int, num_elements: int) -> THBSplineSpace:
    """A hierarchy refined along the diagonal, cell by cell.

    A band rather than a box, so the active set is not a single rectangle and the
    grid's block partition is genuinely fragmented -- which is the case where a
    per-cell dict looks best relative to a flat table.

    Args:
        dim (int): Parametric dimension.
        degree (int): Polynomial degree, the same in every direction.
        num_elements (int): Root elements per direction.

    Returns:
        THBSplineSpace: The space.
    """
    direction = BsplineSpace1D(_open_knots(degree, num_elements), degree)
    root = BsplineSpace([direction] * dim)
    grid = hierarchical_grid(uniform_grid([[0.0, 1.0]] * dim, [num_elements] * dim), 2)
    for index in range(num_elements):
        grid = grid.refine(0, [index] * dim, [index + 1] * dim)
    return THBSplineSpace(root, grid)


def _measure(label: str, space: THBSplineSpace) -> _Footprint:
    """Fill both representations for one space and size them.

    Args:
        label (str): What was built.
        space (THBSplineSpace): The space to measure.

    Returns:
        _Footprint: The two sizes and the counts behind them.
    """
    widest = space.max_active_per_cell()  # forces every cell in both implementations
    cache = space._contrib_cache
    entries = sum(len(contributions) for contributions in cache.values())
    dim = space.dim
    flat = _INT64_BYTES * (space.grid.num_cells + 1) + _INT64_BYTES * (2 + dim) * entries
    return _Footprint(
        label=label,
        dim=dim,
        num_cells=space.grid.num_cells,
        num_entries=entries,
        widest=widest,
        flat_bytes=flat,
        dict_bytes=_deep_size(cache),
    )


def main() -> None:
    """Measure every case and print the table."""
    cases = [
        ("2D deg 2, 32 root elements, 4 corner levels", _corner_chain(2, 2, 32, 4)),
        ("2D deg 3, 64 root elements, 5 corner levels", _corner_chain(2, 3, 64, 5)),
        ("2D deg 2, 32 root elements, diagonal band", _diagonal_band(2, 2, 32)),
        ("3D deg 2, 8 root elements, 3 corner levels", _corner_chain(3, 2, 8, 3)),
        ("3D deg 3, 12 root elements, 3 corner levels", _corner_chain(3, 3, 12, 3)),
    ]
    print(
        f"{'case':46s} {'cells':>7s} {'entries':>9s} {'widest':>7s} "
        f"{'flat kB':>9s} {'dict kB':>9s} {'ratio':>6s}"
    )
    for label, space in cases:
        report = _measure(label, space)
        print(
            f"{report.label:46s} {report.num_cells:7d} {report.num_entries:9d} "
            f"{report.widest:7d} {report.flat_bytes / 1024:9.1f} "
            f"{report.dict_bytes / 1024:9.1f} "
            f"{report.dict_bytes / report.flat_bytes:6.1f}x"
        )
    print()
    print(
        "flat bytes = 8 * (cells + 1) + 8 * (2 + dim) * entries; dict bytes measured "
        "with sys.getsizeof, transitively, counting each object once."
    )


if __name__ == "__main__":
    main()
