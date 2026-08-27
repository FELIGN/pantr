"""Compare the two ways of bounding a balanced BVH's height.

`cpp/bindings/grid.cpp`'s `bind_bvh_build` needs `ceil(log2(n))` for an integer
`n`. It used to compute that through a `double`; it now uses `bit_width(n - 1)`,
which is the same quantity in exact integer arithmetic. This script is the
evidence for the claim that the change is unobservable on any reachable input,
and for the claim that where the two differ the integer form is the correct one.

The floating-point form is evaluated with :func:`math.log2`, which is the same
libm call C++'s ``std::log2`` makes on the same host, over the same IEEE-754
binary64. The verdicts therefore transfer; the C++ side was separately confirmed
to agree on GCC 14, Clang 18, GCC 10 and Clang 10.

Run:
    python scripts/measure_bvh_depth_arithmetic.py
"""

from __future__ import annotations

import math

# Exhaustive up to here, then structural cases only. The interesting inputs are
# powers of two and their immediate neighbours, because that is where `ceil`
# changes value and where a `double` first stops separating `n` from `2**k`.
EXHAUSTIVE_LIMIT = 20_000_000
MAX_EXPONENT = 62

# The binding takes the `n_cells > 1` branch from here up; one cell has depth 1 and
# never reaches either formula.
MIN_CELLS = 2


def depth_via_double(n: int) -> int:
    """Bound the height through binary64, as the binding used to.

    Args:
        n (int): Cell count, at least 2.

    Returns:
        int: ``ceil(log2(n))`` as evaluated in binary64.
    """
    return math.ceil(math.log2(float(n)))


def depth_via_integer(n: int) -> int:
    """Bound the height exactly, as the binding now does.

    Args:
        n (int): Cell count, at least 2.

    Returns:
        int: ``ceil(log2(n))``, computed as the bit width of ``n - 1``.
    """
    return (n - 1).bit_length()


def _candidates() -> list[int]:
    """Enumerate the inputs to compare.

    Returns:
        list[int]: Every integer up to the exhaustive limit, then each power of
            two up to ``2**MAX_EXPONENT`` with its two neighbours either side.
    """
    xs = list(range(MIN_CELLS, EXHAUSTIVE_LIMIT + 1))
    for e in range(1, MAX_EXPONENT + 1):
        for d in (-2, -1, 0, 1, 2):
            n = (1 << e) + d
            if n >= MIN_CELLS:
                xs.append(n)
    return xs


def main() -> None:
    """Report where the two forms disagree, and which one is right."""
    disagreements = []
    checked = 0
    for n in _candidates():
        checked += 1
        fp, exact = depth_via_double(n), depth_via_integer(n)
        if fp != exact:
            disagreements.append((n, fp, exact))

    print(f"checked      {checked}")
    print(f"disagree     {len(disagreements)}")
    if disagreements:
        smallest = min(n for n, _, _ in disagreements)
        print(
            f"smallest n   {smallest} = 2**{smallest.bit_length() - 1} + "
            f"{smallest - (1 << (smallest.bit_length() - 1))}"
        )
        print("first few    (n, via double, exact)")
        for row in disagreements[:5]:
            print(f"             {row}")
        # 2 * 8 bytes per cell for one coordinate column of lo and hi.
        print(
            f"that needs   {smallest * 16 / 2**50:.1f} PiB for one axis of "
            f"cell_lo and cell_hi, so it is not reachable"
        )
        assert all(exact > fp for _, fp, exact in disagreements), (
            "expected the double form to UNDER-report the height wherever they differ"
        )
        print(
            "direction    the double form under-reports every time, so the "
            "integer form is the correct one"
        )


if __name__ == "__main__":
    main()
