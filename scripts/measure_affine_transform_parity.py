"""Measure where the C++ affine map can and cannot reproduce its Python oracle.

`cpp/include/pantr/transform/affine.hpp` cites this script for every number it
states. The port's claims are not uniform -- some operations agree bit for bit
and some only within a bound -- and the point of measuring rather than asserting
is that four plausible explanations for one of the gaps turned out to be wrong.

What is measured
----------------

1. **The norm, by target ISA.** `np.linalg.norm` routes through OpenBLAS's
   `ddot`, which fuses its multiply-add. A C++ loop fuses only where the target
   has FMA, so the same source gives different answers at `x86-64` and
   `x86-64-v3`. This is why the parity claim is a bound.
2. **The sum of squares, by length.** `ddot` changes code path with the vector
   length, so the agreement is not uniform in the dimension either.
3. **The trigonometry.** The extension's `cos` and the interpreter's `math.cos`
   disagree by one ulp on a small fraction of arguments, which no discipline on
   the port's side removes.

What was ruled out, and why it is recorded
------------------------------------------

Each of these looked like the explanation and was refuted by measurement:

- **numpy's scalar `cos`.** It agrees with `math.cos`, so numpy is not the
  difference.
- **`sincos`.** GCC does fuse the `cos`/`sin` pair into one `sincos` call, and
  the extension imports it -- but at run time `sincos` and a separate `cos` agree
  with each other, so the fusion is not the difference either.
- **The contraction setting.** `-ffp-contract=off` changes nothing where the
  target has no FMA, because there is nothing to contract.

Recording refuted explanations is the point: without them a reader re-derives
each one, and the third is convincing enough to be believed without checking.

Run:
    python scripts/measure_affine_transform_parity.py
"""

from __future__ import annotations

import math

import numpy as np

TRIALS = 5000
"""Draws per configuration. Enough that a one-in-a-thousand rate is not noise."""

SEED = 20260828
"""Fixed, so the tables are reproducible rather than merely repeatable."""

MAX_LENGTH = 7
"""Longest vector probed. `ddot`'s path changes well below this."""

HAS_FMA = hasattr(math, "fma")
"""Whether this interpreter can build a fused reference.

``math.fma`` arrived in Python 3.13 and this project supports 3.11, so the fused
column below is gated rather than assumed. The unfused column, which is what a
baseline-target build actually computes, runs on every supported version.
"""


def _fused_sum_of_squares(values: np.ndarray) -> float:
    """Sum squares the way a C++ loop does on a target with FMA.

    Args:
        values (np.ndarray): The vector.

    Returns:
        float: The fused sum.
    """
    if not HAS_FMA:
        raise RuntimeError("math.fma is Python 3.13+; the caller must gate on HAS_FMA")
    fma = math.fma  # type: ignore[attr-defined]  # 3.13+, gated just above
    total = float(values[0]) * float(values[0])
    for value in values[1:]:
        total = fma(float(value), float(value), total)
    return total


def _plain_sum_of_squares(values: np.ndarray) -> float:
    """Sum squares the way a C++ loop does on a target without FMA.

    Args:
        values (np.ndarray): The vector.

    Returns:
        float: The unfused sum.
    """
    total = 0.0
    for value in values:
        total = total + float(value) * float(value)
    return total


def _norms_by_length(rng: np.random.Generator) -> None:
    """Print how each summation order compares with `np.linalg.norm`, by length."""
    print("The norm against np.linalg.norm, by vector length:")
    print(f"{'n':>3}  {'fused differs':>14}  {'unfused differs':>16}")
    for n in range(1, MAX_LENGTH + 1):
        fused_bad = 0
        plain_bad = 0
        for _ in range(TRIALS):
            v = rng.normal(size=n) * 10.0 ** rng.integers(-6, 6)
            reference = np.float64(np.linalg.norm(v)).tobytes()
            if HAS_FMA:
                fused_bad += np.float64(math.sqrt(_fused_sum_of_squares(v))).tobytes() != reference
            plain_bad += np.float64(math.sqrt(_plain_sum_of_squares(v))).tobytes() != reference
        fused_column = f"{100 * fused_bad / TRIALS:>13.2f}%" if HAS_FMA else f"{'n/a':>14}"
        print(f"{n:>3}  {fused_column}  {100 * plain_bad / TRIALS:>15.2f}%")
    print(
        "\nThe fused column is what a build WITH FMA computes and the unfused column\n"
        "what a build without it computes. Neither is uniformly zero, which is the\n"
        "whole reason the parity claim is a bound.\n"
    )


def _trigonometry(rng: np.random.Generator) -> None:
    """Print how far numpy's scalar trigonometry is from the interpreter's."""
    differing = 0
    for _ in range(TRIALS * 10):
        angle = float(rng.normal() * 10.0)
        differing += np.float64(np.cos(angle)).tobytes() != np.float64(math.cos(angle)).tobytes()
    print(f"numpy's scalar cos against math.cos: {differing}/{TRIALS * 10} differ")
    print(
        "Zero is the expected answer, and it is what rules numpy out as the source\n"
        "of the rotation_2d gap. The gap is between the EXTENSION's cos and the\n"
        "interpreter's, which this script cannot reach from Python -- it is measured\n"
        "in tests/parity/test_transform_affine.py, against the built extension.\n"
    )


def main() -> None:
    """Print every table this script is cited for."""
    rng = np.random.default_rng(SEED)
    _norms_by_length(rng)
    _trigonometry(rng)
    print(f"numpy {np.__version__}")


if __name__ == "__main__":
    main()
