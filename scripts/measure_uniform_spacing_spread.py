"""Measure the spacing spread `is_uniform` compares against, and where its bound dies.

`TensorProductGrid.is_uniform` asks whether an axis's breakpoint spacings are constant,
and it compares `ptp(diff(breakpoints))` against `16 * eps * (|first| + |last|)`. The
derivation for that constant is written out at `_UNIFORM_SPACING_EPS_FACTOR` in
`src/pantr/grid/_tensor_product_grid.py` and again in
`cpp/include/pantr/grid/tensor_product_grid.hpp`. This script is the evidence for the
two checkable numbers those derivations quote, so that neither is taken on trust.

It answers three questions.

**Does the spread grow with the cell count?** The constant it replaced was justified by
cell count -- "chosen so a 1e6-cell uniform grid still registers as uniform" -- and the
derivation says there is no `n` in the bound at all. One of the two is wrong.

**How much margin does 16 carry over what is observed?** The derivation gives
`ptp(d) <= 9 eps S`; 16 is the next power of two above that, so the margin over the
derivation is exact arithmetic and needs no measurement. The margin over what is
*observed* is not quoted anywhere in the code, deliberately -- a measured number written
into a comment is pinned to a machine and a seed and nothing will re-measure it. This
script prints it, and asserts the observed ratio stays under the derivation's bound,
which is the part that has to keep holding.

**Where does the tolerance itself underflow?** Below some scale `16 * eps * S` rounds to
zero and the test degenerates to exact equality. The docstrings quote that scale too,
and a first version of them was wrong about it by seventeen orders of magnitude -- which
is why it is measured here rather than reasoned about again.

Everything here is pure numpy. The C++ factory reproduces `numpy.linspace` bit for bit
(asserted by `tests/parity/test_grid_types.py`), so the spreads below are the spreads
both backends see, and no build is needed.

Run:
    python scripts/measure_uniform_spacing_spread.py
"""

from __future__ import annotations

import numpy as np

EPS = float(np.finfo(np.float64).eps)
"""Double-precision machine epsilon; the unit every figure below is stated in."""

DERIVED_FACTOR = 9.0
"""The derivation's own bound on `ptp(diff(linspace)) / (eps * S)`.

Four roundings on the breakpoint plus one on the spacing gives `9 u S` per spacing, and
the spread of the spacings is at most twice that: `18 u S = 9 eps S`."""

SHIPPED_FACTOR = 16
"""The constant the code compares against, mirroring `_UNIFORM_SPACING_EPS_FACTOR`.

Kept as a literal rather than imported so this script needs no `pantr` on the path --
and asserted against the module below, so the two cannot drift apart silently."""

TRIALS = 200_000
"""Random axes drawn for the margin sweep, matching the figure the docstrings quote."""

SEED = 20260831
"""Fixed, so the table is reproducible rather than merely repeatable."""

MIN_SPACINGS_TO_COMPARE = 2
"""A spread needs two spacings; a one-cell axis is uniform by definition."""

CELL_COUNTS = (1, 2, 3, 7, 10, 100, 1_000, 10_000, 100_000, 1_000_000)
"""Cell counts for the growth question, spanning six orders of magnitude."""

DOMAINS = (
    (0.0, 1.0),
    (-1.0, 1.0),
    (0.0, 1e-12),
    (0.0, 1e12),
    (1e6, 1e6 + 1.0),
    (0.3, 0.7),
    (1e-30, 1e-29),
)
"""Domains for the growth question: unit, straddling zero, tiny, huge, offset, interior."""


def spread_ratio(lo: float, hi: float, cells: int) -> float:
    """The spacing spread of a `linspace` axis, in units of `eps * (|lo| + |hi|)`.

    Args:
        lo (float): Axis lower bound.
        hi (float): Axis upper bound.
        cells (int): Number of cells.

    Returns:
        float: `ptp(diff(bp)) / (eps * S)`, or `0.0` when the axis has one cell and so
        has no two spacings to compare.
    """
    bp = np.linspace(lo, hi, cells + 1, dtype=np.float64)
    spacings = np.diff(bp)
    if spacings.size < MIN_SPACINGS_TO_COMPARE:
        return 0.0
    scale = abs(lo) + abs(hi)
    return float(np.ptp(spacings)) / (EPS * scale)


def report_growth() -> None:
    """Print the spread against cell count, to settle whether it grows with it."""
    print("The spread does not grow with the cell count. Rows are domains, columns cells;")
    print("every entry is ptp(diff(linspace)) / (eps * S), S = |lo| + |hi|.\n")
    header = "domain".ljust(22) + "".join(f"{n:>10}" for n in CELL_COUNTS)
    print(header)
    print("-" * len(header))
    for lo, hi in DOMAINS:
        label = f"[{lo:g}, {hi:g}]".ljust(22)
        cells_row = "".join(f"{spread_ratio(lo, hi, n):>10.2f}" for n in CELL_COUNTS)
        print(label + cells_row)
    print()
    per_domain = {(lo, hi): [spread_ratio(lo, hi, n) for n in CELL_COUNTS] for lo, hi in DOMAINS}
    worst_growth = max(max(row) - row[0] if row else 0.0 for row in per_domain.values())
    print(f"Largest increase from the smallest cell count to any larger one: {worst_growth:.2f}")
    print("The old constant's rationale claimed this grows like the cell count. It does not:")
    print("the number of cells enters only through the SPACING, which shrinks, while the")
    print("breakpoint's own error stays proportional to the coordinate magnitude.\n")


def report_margin() -> None:
    """Print the largest observed ratio over a random sweep, against the constant."""
    rng = np.random.default_rng(SEED)
    worst = 0.0
    worst_case: tuple[float, float, int] | None = None
    checked = 0
    while checked < TRIALS:
        exponent = int(rng.integers(-20, 20))
        lo = float(rng.normal()) * 10.0**exponent
        hi = lo + abs(float(rng.normal())) * 10.0 ** int(rng.integers(-20, 20))
        cells = int(rng.integers(2, 2_000))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        bp = np.linspace(lo, hi, cells + 1, dtype=np.float64)
        if not np.all(np.isfinite(bp)) or not np.all(np.diff(bp) > 0.0):
            continue
        scale = abs(lo) + abs(hi)
        if scale == 0.0:
            continue
        checked += 1
        ratio = float(np.ptp(np.diff(bp))) / (EPS * scale)
        if ratio > worst:
            worst = ratio
            worst_case = (lo, hi, cells)
    print(f"Largest observed ptp(d) / (eps * S) over {checked} random linspace axes: {worst:.2f}")
    print(f"  at {worst_case}")
    print(f"  derivation's bound: {DERIVED_FACTOR:.0f}    shipped constant: {SHIPPED_FACTOR}")
    print(f"  margin of the constant over the derivation: {SHIPPED_FACTOR / DERIVED_FACTOR:.2f}x")
    print(f"  margin of the constant over what was observed: {SHIPPED_FACTOR / worst:.2f}x")
    assert worst < DERIVED_FACTOR, (
        f"an observed ratio of {worst:.2f} exceeds the derivation's own bound of "
        f"{DERIVED_FACTOR:.0f}, so the derivation is wrong and the constant with it"
    )
    print()


def report_underflow() -> None:
    """Print the coordinate scale below which the tolerance rounds to zero."""
    tolerance = SHIPPED_FACTOR * EPS
    print(f"{SHIPPED_FACTOR} * eps = {tolerance:g}; the smallest positive subnormal is")
    print(f"{np.nextafter(0.0, 1.0):g}, so the product rounds to exactly zero once S is")
    print("small enough. Bisected:\n")
    # Bisect the EXPONENT, not the value. A geometric midpoint `(low * high) ** 0.5`
    # multiplies two subnormals first, which underflows to zero, pins `low` at zero and
    # leaves `high` where it started -- so the first version of this printed the interval
    # it was given rather than the boundary. The script found its own bug.
    low_exp, high_exp = -320.0, -290.0
    for _ in range(200):
        middle_exp = 0.5 * (low_exp + high_exp)
        if tolerance * (10.0**middle_exp) == 0.0:
            low_exp = middle_exp
        else:
            high_exp = middle_exp
    print(f"  16 * eps * S rounds to exactly 0.0 for S below about {10.0**high_exp:.2g}")
    for scale in (1e-292, 1e-300, 1e-305, 1e-309, 1e-310):
        print(f"    S = {scale:<8g} -> {tolerance * scale:g}")
    print()
    print("Below that scale the comparison becomes exact equality of the spacings, which")
    print("is the honest answer there rather than a failure: a tolerance below the")
    print("smallest subnormal has no representation to compare against. A first version")
    print("of the docstrings put this boundary at 1e-292, which is where the product")
    print("stops being NORMAL, not where it reaches zero -- wrong by seventeen orders of")
    print("magnitude, and the reason this section exists.")


def main() -> None:
    """Run the three measurements and check the shipped constant has not drifted."""
    try:
        from pantr.grid._tensor_product_grid import (  # noqa: PLC0415  (optional)
            _UNIFORM_SPACING_EPS_FACTOR,
        )
    except ImportError:
        print("(pantr not importable; skipping the constant-drift check)\n")
    else:
        assert _UNIFORM_SPACING_EPS_FACTOR == SHIPPED_FACTOR, (
            f"this script measures a margin for {SHIPPED_FACTOR} but the module now ships "
            f"{_UNIFORM_SPACING_EPS_FACTOR}; the figures below would be about the wrong "
            f"constant"
        )
    report_growth()
    report_margin()
    report_underflow()


if __name__ == "__main__":
    main()
