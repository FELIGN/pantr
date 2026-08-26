"""Measure the three ways an interpreted numba oracle differs from a compiled one.

``make coverage`` runs the suite with ``NUMBA_DISABLE_JIT=1``, because coverage.py
cannot trace machine code numba compiled. ``design/backend_parity.md`` Rule 12 states
that exactly three things then differ, and quotes a figure for each. This script is
where those figures come from, so they can be re-measured rather than re-read.

Run it from the repository root, inside the project environment::

    PYTHONPATH="$(pwd)/src" python scripts/measure_interpreted_divergences.py

Every number it prints is a property of numba, numpy and the platform libm on the
machine it runs on, not of pantr. Expect the ``pow`` disagreement count in particular
to move with a numpy or numba upgrade; what must not move is that it is non-zero,
because a bitwise parity claim seeded with ``pow`` rests on it being zero for the
compiled kernel alone.
"""

from __future__ import annotations

import numpy as np
from numba import njit

_INT64_MODULUS = 2**64
"""Where a numba integer accumulator wraps, and a Python one does not."""

_INT64_CEILING = 2**63
"""Above this a wrapped value reads as negative in two's complement."""


@njit(cache=False)  # type: ignore[untyped-decorator]
def _compiled_pow(u: float, p: int) -> float:
    """Seed a Bernstein ratio recurrence the way three kernels in `pantr` do.

    Compiled with ``cache=False`` deliberately: a cached kernel is loaded from disk
    rather than compiled here, and the point is to exercise numba's own code
    generation.

    Args:
        u (float): The base, in ``[0, 1]``.
        p (int): The exponent.

    Returns:
        float: ``u ** p`` as the compiled kernel computes it.
    """
    return float(np.power(u, p))


def measure_pow_disagreement(degrees: int = 30, per_degree: int = 2000) -> tuple[int, int]:
    """Count how often numba's ``np.power`` differs from numpy's own.

    Args:
        degrees (int): Exponents swept, from 0 upward. Defaults to 30.
        per_degree (int): Random bases per exponent. Defaults to 2000.

    Returns:
        tuple[int, int]: ``(disagreements, comparisons)``.
    """
    _compiled_pow(0.5, 2)
    rng = np.random.default_rng(0)
    disagreements = 0
    comparisons = 0
    for exponent in range(degrees):
        for base in rng.random(per_degree):
            comparisons += 1
            if _compiled_pow(float(base), exponent) != np.power(base, exponent):
                disagreements += 1
    return disagreements, comparisons


def measure_factorial_wrap(degree: int = 25, order: int = 16) -> tuple[int, int, float]:
    """Compute a falling factorial exactly and as int64 would hold it.

    ``_evaluate_bezier_deriv_1d_core`` and ``_bernstein_derivs_point`` accumulate this
    product. Compiled, the accumulator is an int64 and wraps; interpreted, it is a
    Python integer and grows, so past the overflow the two paths differ by a factor
    rather than by a rounding.

    Args:
        degree (int): The polynomial degree. Defaults to 25.
        order (int): The derivative order. Defaults to 16.

    Returns:
        tuple[int, int, float]: ``(exact, wrapped, ratio)``.
    """
    exact = degree
    for k in range(1, order):
        exact *= degree - k
    residue = exact % _INT64_MODULUS
    wrapped = residue - _INT64_MODULUS if residue >= _INT64_CEILING else residue
    return exact, wrapped, exact / wrapped


def measure_float32_underflow(scale: float = 1e-23) -> tuple[float, float]:
    """Show a product that underflows at ``float32`` and does not once widened.

    This is the mechanism behind FELIGN/pantr#351, and the reason its regression test
    cannot exercise the defect under ``NUMBA_DISABLE_JIT=1``: numba's ``float()`` does
    not widen a ``float32`` while CPython's does, so interpreted, the product never
    reaches the subnormal range where the defect lives.

    Args:
        scale (float): Coefficient magnitude. Defaults to 1e-23, the frontier.

    Returns:
        tuple[float, float]: ``(narrow_product, widened_product)``.
    """
    lo = np.float32(scale)
    hi = np.float32(-scale)
    return float(lo * hi), float(lo) * float(hi)


def main() -> None:
    """Print all three measurements with the labels Rule 12 quotes them under."""
    disagreements, comparisons = measure_pow_disagreement()
    print(f"pow disagreement:  {disagreements} of {comparisons}")

    exact, wrapped, ratio = measure_factorial_wrap()
    print(f"factorial wrap:    exact {exact}, wrapped {wrapped}, ratio {ratio:.4f}")

    narrow, widened = measure_float32_underflow()
    print(f"float32 underflow: narrow {narrow}, widened {widened:.3e}")


if __name__ == "__main__":
    main()
