#!/usr/bin/env python
"""Reproduce the contraction bound of the Bézier kernels, and the slack it carries.

The shipped build sets ``-ffp-contract=on`` and no ``-march``, so the target is
baseline x86-64, which has no fused multiply-add and nothing to contract into. Every
parity claim in ``tests/parity/test_bezier_arithmetic.py`` therefore takes its
**bitwise** branch, and the **bounded** branch, which is the one this script is
about, ships unevaluated.

Run it against the shipped build and it reports zeros and says so. Run it against a
build that fuses and it reports, per kernel and per dtype, how many values move and
how much of the derived bound the worst one uses.

    PYTHONPATH="$(pwd)/src:$(pwd)" .venv/bin/python scripts/measure_bezier_fma_bound.py

Building one that fuses, and the two traps in doing so
------------------------------------------------------

**``-DCMAKE_CXX_FLAGS=-ffp-contract=off`` does nothing.** ``PantrCompileOptions.cmake``
adds ``-ffp-contract=on`` as a target option, which lands *after* ``CMAKE_CXX_FLAGS``
on the command line, and the last one wins. A build meant to isolate contraction from
vectorisation has to append the flag some other way -- a compiler wrapper script is
the shortest. Check ``compile_commands.json`` rather than trusting the cache.

**Eigen does not compile under ``-march=native`` with ``PANTR_WERROR=ON``.** Its
AVX512 ``TrsmKernel.h`` trips ``-Wmaybe-uninitialized`` in its own code. Measurement
builds pass ``-DPANTR_WERROR=OFF``; the ISA ladder of ``design/simd.md`` will have to
decide what the shipped build does about it.

    cmake --preset gcc -B build/fma-native -DPANTR_BUILD_PYTHON=ON \
        -DPANTR_BUILD_TESTS=OFF -DPANTR_BUILD_BENCHMARK=OFF -DPANTR_WERROR=OFF \
        -DPython_EXECUTABLE="$(pwd)/.venv/bin/python" -DCMAKE_CXX_FLAGS=-march=native
    cmake --build build/fma-native
    cp build/fma-native/cpp/bindings/_pantr_cpp*.so \
       .venv/lib/python3*/site-packages/pantr/

Restore the shipped extension afterwards with ``pip install -e ".[dev]"``.

What was measured when this was written
---------------------------------------

With ``-march=native`` on gcc 14.4, fourteen sites fuse and every one was predicted by
reading the source first: two in ``evaluate`` (the two accumulation branches), five in
``evaluate_deriv``, two in ``restrict`` (one per pass) and one each in
``degree_elevate``, ``slice``, ``split``, ``scalar_bernstein_product`` and the
reduction apply. The baseline build contains none, in the whole shared object.

Contraction is the only mechanism: ``-march=native`` with contraction genuinely off
restores bit-identity exactly, 0 of 1260 and 0 of 3616, while the vectorisation stays
(191 packed instructions against 34). And the oracle never fuses -- numba targets this
host's ISA and still emits no FMA without ``fastmath``, which
``test_the_oracle_does_not_contract_a_multiply_add`` pins.

Slack of the derived bounds, worst case over this sweep, on that build: 6.8x for the
two de Casteljau kernels, 70x for ``evaluate``, and between those for the rest. At
float32 only ``evaluate_deriv`` and ``restrict`` move at all -- the other six contract
in a float64 accumulator and the narrowing store absorbs it -- so ``restrict``'s
float32 slack reads 883x, which is the bound being honest about a straddle that almost
never happens rather than the bound being wrong.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from math import comb, prod
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from pantr._backend import Backend, available_backends, use_backend
from pantr.bezier import _bezier_backend as backend
from pantr.bezier._bezier_degree import _interpolating_reduction_operator
from tests._parity_harness import (
    absolute_tolerance,
    build_provenance,
    contraction_may_fuse,
    unit_roundoff,
)
from tests.parity.test_bezier_arithmetic import (
    _Budget,
    _derivative_scale,
    _fused_claim,
    _mixed_control_points,
    _net_magnitude,
    _reduction_amplification,
)

DEGREES: Final = (1, 2, 3, 5, 8, 13, 17, 25, 34)
"""Degrees swept. 34 is past anything pantr builds and is where the reduction operator
grows enough for its amplification to matter."""

PARAMS: Final = (0.0, 1e-8, 0.25, 0.5, 0.75, 1.0 - 1e-8, 1.0)
"""Parameters swept, straddling the mirror threshold and both endpoints."""

MAX_PRODUCT_DEGREE: Final = 56
"""Largest sum of degrees the binomial table stays exact for.

``core::kBincoeffExactDoubleMaxN``: past it ``C(n, k)`` no longer fits a double
exactly, and the kernel's Layer 2 refuses the call rather than rounding it."""


def _run_both(
    fill: Callable[[Backend, npt.NDArray[Any]], None],
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Drive one kernel under each backend at its own entry point.

    Args:
        fill (Callable): Takes a backend and an output array, and fills it.
        shape (tuple[int, ...]): Output shape.
        dtype (npt.DTypeLike): Output dtype.

    Returns:
        tuple: The Numba result and the C++ result, in that order.
    """
    reference = np.zeros(shape, dtype=dtype)
    actual = np.zeros(shape, dtype=dtype)
    with use_backend(Backend.PYTHON):
        fill(Backend.PYTHON, reference)
    with use_backend(Backend.CPP):
        fill(Backend.CPP, actual)
    return reference, actual


def _product_amplification(
    left: npt.NDArray[Any], right: npt.NDArray[Any]
) -> npt.NDArray[np.float64]:
    """Convex amplification of the Bernstein product, per output coefficient.

    Args:
        left (npt.NDArray[Any]): First operand.
        right (npt.NDArray[Any]): Second operand.

    Returns:
        npt.NDArray[np.float64]: The weighted sum of ``|a_i b_j|``.
    """
    p = left.size - 1
    q = right.size - 1
    total = p + q
    return np.array(
        [
            sum(
                comb(p, i) * comb(q, k - i) * abs(float(left[i])) * abs(float(right[k - i]))
                for i in range(max(0, k - q), min(p, k) + 1)
            )
            / comb(total, k)
            for k in range(total + 1)
        ],
        dtype=np.float64,
    )


def _cases(degree: int, dtype: npt.DTypeLike, rng: np.random.Generator) -> list[Any]:
    """Build one case per kernel at one degree.

    Args:
        degree (int): The degree.
        dtype (npt.DTypeLike): Storage format.
        rng (np.random.Generator): Unused; the nets come from the test module's
            deterministic generator so the figures match the suite's.

    Returns:
        list: Name, driver, shape, budget and accumulator, one entry per kernel.
    """
    del rng
    ctrl = _mixed_control_points((degree + 1, 3), dtype)
    points = np.ascontiguousarray(PARAMS, dtype=dtype)
    magnitude = _net_magnitude(ctrl)
    rank = 3

    left = _mixed_control_points((degree + 1,), dtype, seed=20260822)
    right = _mixed_control_points((min(degree, 8) + 1,), dtype, seed=20260823)
    operator = _interpolating_reduction_operator(degree, 1)

    cases = [
        (
            "evaluate",
            lambda b, o: backend.evaluate_kernel(b)(ctrl, points, o),
            (len(PARAMS), rank),
            _Budget(degree, np.full((len(PARAMS), rank), magnitude, dtype=np.float64)),
            np.float64,
        ),
        (
            "evaluate_deriv",
            lambda b, o: backend.evaluate_deriv_kernel(b)(ctrl, points, 2, o),
            (len(PARAMS), 3, rank),
            _Budget(
                degree + 1,
                np.stack(
                    [
                        np.full((len(PARAMS), rank), _derivative_scale(degree, k, magnitude))
                        for k in range(3)
                    ],
                    axis=1,
                ),
            ),
            dtype,
        ),
        (
            "degree_elevate",
            lambda b, o: backend.degree_kernels(b).elevate(degree, ctrl, 2, o),
            (degree + 3, rank),
            _Budget(degree + 1, np.full((degree + 3, rank), magnitude, dtype=np.float64)),
            np.float64,
        ),
        (
            "slice",
            lambda b, o: backend.slice_kernel(b)(ctrl, 0.25, o),
            (rank,),
            _Budget(degree, np.full((rank,), magnitude, dtype=np.float64)),
            np.float64,
        ),
        (
            "split",
            lambda b, o: backend.split_kernel(b)(ctrl, 0.25, o, np.zeros_like(o)),
            (degree + 1, rank),
            _Budget(degree, np.full((degree + 1, rank), magnitude, dtype=np.float64)),
            np.float64,
        ),
        (
            "restrict",
            lambda b, o: backend.restrict_kernel(b)(ctrl, 0.1, 0.9, o),
            (degree + 1, rank),
            _Budget(2 * degree, np.full((degree + 1, rank), magnitude, dtype=np.float64)),
            np.float64,
        ),
        (
            "reduce_apply",
            lambda b, o: backend.degree_kernels(b).reduce_apply(operator, ctrl, o),
            (degree, rank),
            _Budget(degree + 1, _reduction_amplification(ctrl, degree, 1)),
            np.float64,
        ),
    ]
    if left.size - 1 + right.size - 1 <= MAX_PRODUCT_DEGREE:
        cases.append(
            (
                "scalar_product",
                lambda b, o: backend.product_kernel(b)(left, right, o),
                (left.size + right.size - 1,),
                _Budget(min(left.size, right.size), _product_amplification(left, right)),
                np.float64,
            )
        )
    return cases


def measure() -> dict[tuple[str, str], tuple[int, int, float]]:
    """Measure movement and slack for every kernel at both dtypes.

    Returns:
        dict: Keyed by kernel and dtype name; values are elements moved, elements
            compared, and the worst ratio of the observed difference to the bound.
    """
    results: dict[tuple[str, str], tuple[int, int, float]] = {}
    for dtype in (np.float64, np.float32):
        name = str(np.dtype(dtype).name)
        rng = np.random.default_rng(20260822)
        for degree in DEGREES:
            for kernel, fill, shape, budget, accumulator in _cases(degree, dtype, rng):
                reference, actual = _run_both(fill, shape, dtype)
                claim = _fused_claim(
                    fused_why=f"{kernel} at degree {degree}",
                    budget=budget,
                    storage=dtype,
                    accumulator=accumulator,
                )
                tolerance = np.broadcast_to(absolute_tolerance(claim), np.shape(reference))
                difference = np.abs(np.float64(actual) - np.float64(reference))
                usable = np.isfinite(difference) & (tolerance > 0.0)
                moved, seen, worst = results.get((kernel, name), (0, 0, 0.0))
                moved += int((difference[usable] > 0.0).sum())
                seen += int(usable.sum())
                if usable.any():
                    worst = max(worst, float((difference[usable] / tolerance[usable]).max()))
                results[(kernel, name)] = (moved, seen, worst)
    return results


def derivative_amplification_is_sharp() -> float:
    """Measure how much of the derivative amplification the basis actually reaches.

    The order-``k`` block of the bound is ``p!/(p-k)! * 2^k * max|c|``, from
    ``B^(k) = p!/(p-k)! sum_j (Delta^k c)_j B_{j,p-k}`` and ``||Delta^k||_inf <= 2``.
    Driving the kernel with the identity net returns the basis derivatives themselves,
    so ``max_s sum_j |B^(k)_{j,p}(s)|`` divided by that factor says how tight it is. A
    value below 1 would mean the amplification is loose; above 1 would refute it.

    Returns:
        float: The largest ratio over the sweep. Measured 1.0000 when written.
    """
    sampled = np.linspace(0.0, 1.0, 401)
    worst = 0.0
    for degree in DEGREES:
        orders = min(4, degree)
        identity = np.ascontiguousarray(np.eye(degree + 1))
        out = np.zeros((sampled.size, orders + 1, degree + 1))
        with use_backend(Backend.PYTHON):
            backend.evaluate_deriv_kernel(Backend.PYTHON)(
                identity, np.ascontiguousarray(sampled), orders, out
            )
        for order in range(orders + 1):
            falling = prod(range(degree - order + 1, degree + 1)) if order else 1
            reached = float(np.abs(out[:, order, :]).sum(axis=1).max())
            worst = max(worst, reached / (falling * 2.0**order))
    return worst


def main() -> int:
    """Report the bound's slack against whatever extension is installed.

    Returns:
        int: 0 if every bound held, 1 if the C++ backend is missing or one was
            exceeded.
    """
    if Backend.CPP not in available_backends():
        print("the C++ backend is not built; nothing to compare")
        return 1

    provenance = build_provenance()
    fusing = contraction_may_fuse()
    print(f"compiler            {provenance.compiler}")
    print(f"__fp_contract__     {provenance.fp_contract}")
    print(f"unit roundoff       float64 {unit_roundoff(np.float64):.3e}")
    print(f"                    float32 {unit_roundoff(np.float32):.3e}")
    if not fusing:
        print(
            "\nThis build's target ISA has no fused multiply-add, so the suite takes its\n"
            "bitwise branch and every count below will be zero. That is the expected\n"
            "result here and not a measurement of the bound. Build with -march=native to\n"
            "exercise it; the header of this file says how, and names the two traps."
        )

    print(f"\n{'kernel':<18}{'dtype':<10}{'moved':>8}{'seen':>8}{'worst/bound':>14}{'slack':>10}")
    exceeded = False
    for (kernel, dtype), (moved, seen, worst) in sorted(measure().items()):
        slack = f"{1 / worst:8.1f}x" if worst > 0.0 else "     n/a"
        print(f"{kernel:<18}{dtype:<10}{moved:>8}{seen:>8}{worst:>14.5f}{slack:>10}")
        exceeded = exceeded or worst > 1.0

    sharpness = derivative_amplification_is_sharp()
    print(
        f"\nderivative amplification, reached / derived: {sharpness:.4f}\n"
        f"  1.0000 means the bound p!/(p-k)! * 2^k is attained rather than conservative."
    )

    if exceeded:
        print("\nAt least one bound was EXCEEDED. The derivation is wrong, not the build.")
    return 1 if exceeded else 0


if __name__ == "__main__":
    sys.exit(main())
