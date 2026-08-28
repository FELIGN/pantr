"""Locate the dimension at which `AABB.transform` stops being bit-exact.

`cpp/include/pantr/geometry/aabb.hpp`'s `transform` accumulates the per-axis
contributions in a sequential loop and adds the offset last. The oracle,
`_AABBPython.transform` in `src/pantr/geometry.py`, computes
``np.sum(contributions, axis=1) + b``. The two agree bit for bit only while
numpy's own summation is sequential, and `np.sum` blocks pairwise above a fixed
block size. This script is the evidence for where that boundary is.

It also records the defect that made the script necessary. The header used to
claim the boundary was 8 and to name numpy's blocking as the only hazard. Both
were wrong: the boundary is 7, and the accumulation ALSO diverged at ndim = 3
because the C++ loop seeded its accumulator with the offset instead of adding it
last, which reorders the sum. The second measurement below reproduces that,
because a bound whose refutation is not reproducible is a claim again.

Everything here is pure numpy: the C++ side's order is a sequential loop, which
Python reproduces exactly, so no build is needed to establish the threshold.

Run:
    python scripts/measure_aabb_transform_summation.py
"""

from __future__ import annotations

import numpy as np

TRIALS = 20_000
"""Random draws per dimension. Large enough that a ~50% divergence rate cannot
read as zero, and small enough to run in seconds."""

MAX_NDIM = 13
"""Highest dimension probed. numpy's blocking is already unambiguous well before
this, and pantr builds no box beyond a handful of axes."""

SEED = 20260827
"""Fixed, so the table is reproducible rather than merely repeatable."""

FIRST_DIVERGENT_NDIM = 8
"""The dimension at which np.sum stops matching a sequential loop.

The header's equality claim is stated as `ndim <= 7`, which is this minus one.
The two numbers are the same fact and must move together, so the assertion below
names this constant rather than a literal."""


def _sequential(values: np.ndarray) -> np.float64:
    """Sum left to right, the order a C++ loop uses.

    Args:
        values (np.ndarray): The terms.

    Returns:
        np.float64: Their sequential sum.
    """
    total = np.float64(0.0)
    for value in values:
        total = total + value
    return total


def main() -> None:
    """Print the divergence table and assert the threshold the header states."""
    rng = np.random.default_rng(SEED)

    print("Sequential summation against np.sum, per dimension:")
    print(f"{'ndim':>5}  {'np.sum differs':>15}  {'+ offset last differs':>22}")
    first_divergent = None
    for ndim in range(2, MAX_NDIM + 1):
        plain = 0
        with_offset = 0
        for _ in range(TRIALS):
            terms = rng.normal(size=ndim)
            offset = rng.normal()
            seq = _sequential(terms)
            if np.sum(terms).tobytes() != np.float64(seq).tobytes():
                plain += 1
            if (np.sum(terms) + offset).tobytes() != np.float64(seq + offset).tobytes():
                with_offset += 1
        print(f"{ndim:>5}  {100 * plain / TRIALS:>14.2f}%  {100 * with_offset / TRIALS:>21.2f}%")
        if plain and first_divergent is None:
            first_divergent = ndim

    print(f"\nFirst dimension where np.sum stops matching a sequential loop: {first_divergent}")
    assert first_divergent == FIRST_DIVERGENT_NDIM, (
        f"the header claims equality holds while ndim <= 7, which requires the first "
        f"divergent dimension to be 8; measured {first_divergent} with numpy {np.__version__}"
    )

    print("\nSeeding the accumulator with the offset instead of adding it last:")
    for ndim in (2, 3, 4):
        differ = 0
        for _ in range(TRIALS):
            terms = rng.normal(size=ndim)
            offset = rng.normal()
            correct = _sequential(terms) + offset
            seeded = _sequential(np.concatenate(([offset], terms)))
            if np.float64(correct).tobytes() != np.float64(seeded).tobytes():
                differ += 1
        print(f"  ndim={ndim}: {100 * differ / TRIALS:.1f}% of draws differ")
    print(
        "\nThat is the defect the parity suite missed, because every matrix in its\n"
        "original cases was exactly representable and no rounding could show."
    )


if __name__ == "__main__":
    main()
