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
:func:`report_the_disassembly` re-derives all of that, and the packed-instruction
counts beside it, rather than quoting them: a review found both had been written into
permanent artifacts with nothing in the tree able to reproduce either.

Contraction is the only mechanism: ``-march=native`` with contraction genuinely off
restores bit-identity exactly, 0 of 1260 and 0 of 3616, while the vectorisation stays
(191 packed instructions against 34). And the oracle never fuses -- numba targets this
host's ISA and still emits no FMA without ``fastmath``, which
``test_the_oracle_does_not_contract_a_multiply_add`` pins.

Slack of the derived bounds, worst case over this sweep, on that build: 6.8x for the
two de Casteljau kernels, 14.5x for the product, 17.9x for elevation, 18.3x and 43.2x
for the derivative, 26.2x for the reduction apply, 35.8x for ``restrict`` at float64
and 70x for ``evaluate``. At float32 only ``evaluate_deriv`` and ``restrict`` move at
all, because the other six contract in a float64 accumulator and the narrowing store
absorbs it.

**Two things the slack column does not tell you, and both were review findings.** It
is ``1 / max(diff/tolerance)``, the *tightest* point in the array, so a block whose
bound is orders of magnitude too loose can only lower that maximum and is structurally
invisible here. And it is not a fixed safety factor: at float32 it is roughly
``2 * stages * (amplification / |value|)``, so it grows with the degree by
construction. ``restrict`` at float32 reads 883x for exactly that reason, one whole
storage ulp charged at each of ``2p`` stages while a real run straddles a rounding
boundary once or twice, and not because a straddle is rare in some fixed proportion.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from math import comb
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from pantr._backend import Backend, available_backends, use_backend
from pantr.bezier import Bezier
from pantr.bezier import _bezier_backend as backend
from pantr.bezier._bezier_degree import _interpolating_reduction_operator
from tests._parity_harness import (
    absolute_tolerance,
    build_provenance,
    contraction_may_fuse,
    unit_roundoff,
)
from tests.parity.test_bezier_arithmetic import (
    _a23_absolute_rows,
    _Budget,
    _derivative_amplification,
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
                2 * degree + 4,
                np.stack([_derivative_amplification(ctrl, points, k) for k in range(3)], axis=1),
            ),
            dtype,
        ),
        (
            "degree_elevate",
            lambda b, o: backend.degree_kernels(b).elevate(degree, ctrl, 2, o),
            (degree + 3, rank),
            _Budget(min(degree, 2) + 1, np.full((degree + 3, rank), magnitude, dtype=np.float64)),
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
            _Budget(degree + 1, _reduction_amplification(ctrl, degree, 1), narrowing_stores=1),
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


_PROBE_SOURCE: Final = """
// Instantiate every Bezier kernel at both widths behind a noinline wrapper, so a
// disassembler can attribute each fused multiply-add to one kernel and one line.
#include "pantr/bezier/kernels_1d.hpp"
#include "pantr/core/reduction_operator.hpp"
using pantr::span2d;
using pantr::span_nd;
#define K __attribute__((noinline))
template <class T> K void k1(span2d<const T> c, std::span<const T> p, span2d<T> o) {
    pantr::bezier::evaluate_bezier_1d<T>(c, p, o); }
template <class T> K void k2(span2d<const T> c, std::span<const T> p, int n, span_nd<T,3> o) {
    pantr::bezier::evaluate_bezier_deriv_1d<T>(c, p, n, o); }
template <class T> K void k3(int d, span2d<const T> c, int t, span2d<T> o) {
    pantr::bezier::degree_elevate_bezier_1d<T>(d, c, t, o); }
template <class T> K void k4(span2d<const T> c, pantr::accumulator_t<T> v, std::span<T> o) {
    pantr::bezier::slice_bezier_1d<T>(c, v, o); }
template <class T> K void k5(span2d<const T> c, pantr::accumulator_t<T> v, span2d<T> l,
                             span2d<T> r) { pantr::bezier::split_bezier_1d<T>(c, v, l, r); }
template <class T> K void k6(span2d<const T> c, pantr::accumulator_t<T> a,
                             pantr::accumulator_t<T> b, span2d<T> o) {
    pantr::bezier::restrict_bezier_1d<T>(c, a, b, o); }
template <class T> K void k7(std::span<const T> a, std::span<const T> b, std::span<T> o) {
    pantr::bezier::scalar_bernstein_product_1d<T>(a, b, o); }
template <class T> K void k8(span2d<const double> op, span2d<const T> c, span2d<T> o) {
    pantr::core::apply_reduction_operator<T>(op, c, o); }
#define INST(T) \\
  template void k1<T>(span2d<const T>, std::span<const T>, span2d<T>); \\
  template void k2<T>(span2d<const T>, std::span<const T>, int, span_nd<T,3>); \\
  template void k3<T>(int, span2d<const T>, int, span2d<T>); \\
  template void k4<T>(span2d<const T>, double, std::span<T>); \\
  template void k5<T>(span2d<const T>, double, span2d<T>, span2d<T>); \\
  template void k6<T>(span2d<const T>, double, double, span2d<T>); \\
  template void k7<T>(std::span<const T>, std::span<const T>, std::span<T>); \\
  template void k8<T>(span2d<const double>, span2d<const T>, span2d<T>);
INST(float)
INST(double)
"""
"""A translation unit that instantiates all eight kernels, for the disassembly."""

_FUSED_MNEMONIC: Final = re.compile(r"\bvf(n?)m(add|sub)[a-z0-9]*\b")
"""x86 fused multiply-add mnemonics, in all their add/subtract/negate spellings."""

_PACKED_ARITHMETIC: Final = re.compile(r"\bv(add|sub|mul|fmadd|fnmadd)[a-z]*p[sd]\b")
"""Packed floating-point arithmetic, which is how vectorisation shows up.

Named explicitly because a count of "SIMD instructions" is meaningless without it:
a broader pattern that also caught moves and shuffles gives a number twice as large
and says nothing about whether the arithmetic was vectorised.
"""


def enumerate_fused_sites(extra_flags: list[str]) -> dict[tuple[str, int], int] | None:
    """Compile the probe and count fused multiply-adds per source line.

    This is the artifact behind "fourteen sites fuse". A review found that claim, and
    the packed-instruction counts beside it, quoted from a scratch directory with
    nothing in the tree to re-derive them -- the same failure Rule 9 of
    design/backend_parity.md records against itself.

    Args:
        extra_flags (list[str]): Compiler flags to add, e.g. ``["-march=native"]``.

    Returns:
        dict[tuple[str, int], int] | None: Fused instruction count per (file, line),
            or None if the toolchain is not available.
    """
    root = Path(__file__).resolve().parent.parent
    mdspan = root / "build" / "gcc" / "_deps" / "mdspan-src" / "include"
    if not shutil.which("g++") or not shutil.which("objdump") or not mdspan.is_dir():
        return None

    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / "probe.cpp"
        obj = Path(scratch) / "probe.o"
        source.write_text(_PROBE_SOURCE)
        compile_command = [
            "g++", "-std=c++20", "-O3", "-g", "-ffp-contract=on",
            f"-I{root / 'cpp' / 'include'}", f"-I{mdspan}",
            *extra_flags, "-c", str(source), "-o", str(obj),
        ]  # fmt: skip
        if subprocess.run(compile_command, capture_output=True, check=False).returncode:
            return None
        listing = subprocess.run(
            ["objdump", "-dl", "-C", "--no-show-raw-insn", str(obj)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    sites: dict[tuple[str, int], int] = {}
    location: tuple[str, int] | None = None
    for line in listing.splitlines():
        source_line = re.match(r"^(/\S+):(\d+)", line)
        if source_line:
            location = (Path(source_line.group(1)).name, int(source_line.group(2)))
            continue
        mnemonic = re.match(r"^\s+[0-9a-f]+:\s+(\S+)", line)
        if mnemonic and _FUSED_MNEMONIC.match(mnemonic.group(1)) and location is not None:
            sites[location] = sites.get(location, 0) + 1
    return sites


def count_packed_arithmetic(extra_flags: list[str]) -> int | None:
    """Count packed floating-point arithmetic instructions in the probe.

    Used to separate vectorisation from contraction: turning contraction off must not
    turn vectorisation off, or the isolation argument compares two different things.

    Args:
        extra_flags (list[str]): Compiler flags to add.

    Returns:
        int | None: The count, or None if the toolchain is not available.
    """
    root = Path(__file__).resolve().parent.parent
    mdspan = root / "build" / "gcc" / "_deps" / "mdspan-src" / "include"
    if not shutil.which("g++") or not shutil.which("objdump") or not mdspan.is_dir():
        return None
    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / "probe.cpp"
        obj = Path(scratch) / "probe.o"
        source.write_text(_PROBE_SOURCE)
        command = [
            "g++", "-std=c++20", "-O3", "-ffp-contract=on",
            f"-I{root / 'cpp' / 'include'}", f"-I{mdspan}",
            *extra_flags, "-c", str(source), "-o", str(obj),
        ]  # fmt: skip
        if subprocess.run(command, capture_output=True, check=False).returncode:
            return None
        listing = subprocess.run(
            ["objdump", "-d", str(obj)], capture_output=True, text=True, check=True
        ).stdout
    return len(_PACKED_ARITHMETIC.findall(listing))


def report_the_disassembly() -> None:
    """Print the fused-site enumeration and the vectorisation counts."""
    baseline = enumerate_fused_sites([])
    fusing = enumerate_fused_sites(["-march=native"])
    unfused = enumerate_fused_sites(["-march=native", "-ffp-contract=off"])
    if baseline is None or fusing is None or unfused is None:
        print("\nno g++/objdump or no fetched mdspan, so the disassembly is not reported")
        return

    print(f"\nfused sites, baseline target       {sum(baseline.values()):>6}")
    print(f"fused sites, -march=native         {len(fusing):>6} distinct source lines")
    print(f"fused sites, and contraction off   {sum(unfused.values()):>6}")
    for (name, line), count in sorted(fusing.items()):
        print(f"    {name}:{line:<5} x{count}")

    packed_on = count_packed_arithmetic(["-march=native"])
    packed_off = count_packed_arithmetic(["-march=native", "-ffp-contract=off"])
    print(
        f"\npacked FP arithmetic, -march=native: {packed_on} with contraction, "
        f"{packed_off} without.\n"
        f"  Both non-zero, which is the point: turning contraction off leaves the "
        f"vectorisation\n  in place, so the bit-identity it restores isolates "
        f"contraction and nothing else.\n"
        f"  The second number is the larger because each fused instruction becomes two."
    )


def the_a23_majorant_holds() -> tuple[int, int, float]:
    """Check that the A2.3 majorant bounds the signed basis derivatives it replaces.

    ``tests.parity.test_bezier_arithmetic._a23_absolute_rows`` runs A2.3 with every
    coefficient replaced by its modulus, which is the standard majorant for a linear
    recursion with signed coefficients. This is the check that it is one: a violation
    would mean the derivative kernel's amplification is unsound, and a maximum ratio
    far below 1 would mean it is loose.

    Returns:
        tuple[int, int, float]: Entries checked, violations, and the largest ratio of
            the signed value to its majorant. Measured 0 violations and 1.000000 when
            written, so the majorant is attained rather than conservative.
    """
    sampled = np.linspace(0.0, 1.0, 97)
    checked = violations = 0
    worst = 0.0
    for degree in DEGREES:
        identity = np.ascontiguousarray(np.eye(degree + 1))
        for order in range(min(degree, 6) + 1):
            majorant = _a23_absolute_rows(degree, sampled, order)
            with use_backend(Backend.PYTHON):
                driven = Bezier(identity).evaluate_derivatives(sampled, order)
            signed = np.abs(np.asarray(driven).reshape(sampled.size, degree + 1))
            checked += signed.size
            violations += int((signed > majorant * (1.0 + 1e-12)).sum())
            live = majorant > 0.0
            worst = max(worst, float(np.max(signed[live] / majorant[live])))
    return checked, violations, worst


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

    checked, violations, worst_ratio = the_a23_majorant_holds()
    print(
        f"\nA2.3 majorant: {violations} violations over {checked} (point, basis) entries, "
        f"largest |signed| / majorant {worst_ratio:.6f}\n"
        f"  Must be at most 1. Exactly 1 means the majorant is attained rather than "
        f"conservative;\n  a violation would mean the derivative kernel's amplification "
        f"is unsound."
    )
    if violations:
        exceeded = True

    report_the_disassembly()

    if exceeded:
        print("\nAt least one bound was EXCEEDED. The derivation is wrong, not the build.")
    return 1 if exceeded else 0


if __name__ == "__main__":
    sys.exit(main())
