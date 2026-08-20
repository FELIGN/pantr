#!/usr/bin/env python
"""Time the C++ quadrature kernels against the Python ones they are ported from.

Run it as::

    python scripts/bench_quad.py
    python scripts/bench_quad.py --counts 17 200 1000

Two levels, for the reason ``scripts/bench_parity.py`` records
--------------------------------------------------------------

That script's first version compared a kernel called directly against a kernel
called through the public entry point and concluded the port was three times
slower, having charged one side for the binding, the validation and the numpy
round trip and the other side for none of them. The same two levels are measured
here, each with both backends on the same footing:

``kernel``
    The compiled kernel alone, with ``out`` preallocated on both sides, so neither
    allocates and neither validates. **The port's own figure.**

``entry``
    The public function, once per backend. Identical validation, allocation and
    dispatch on both sides; the only difference is which kernel runs underneath.
    **What a caller sees.**

Reading the numbers
-------------------

**These are microsecond-scale measurements and they are noisy.** A rule generator
builds an ``n``-point rule once and the result is then used many times, so there
is no large batch to average over and no hot loop to warm. The minimum over
repeats is reported rather than the mean, and a single figure that breaks the
trend in ``n`` should be read as noise rather than as a finding: an earlier run of
this comparison reported the C++ Chebyshev kernel as *slower* at one count, with
the C++ time non-monotone in ``n``, and it did not survive more repeats.

**Gauss-Legendre has a real crossover and it is worth understanding rather than
reporting.** The Python builds the whole rule with numpy, vectorised over the
``n/2`` non-negative roots, so its inner recurrence runs on arrays and gets the
machine's SIMD throughput for free. The C++ runs a scalar loop over one root at a
time. Both are the same ``O(n^2)`` work, so at small ``n`` the C++ wins by the
whole of numpy's per-operation overhead and at large ``n`` numpy's vectorisation
catches up. That is the gap ``design/simd.md``'s stage 2 exists to close, and it
is a property of this kernel's shape rather than of the port.
"""

from __future__ import annotations

import argparse
import functools
import timeit
from collections.abc import Callable, Sequence
from typing import Final

import numpy as np

import pantr._pantr_cpp as extension
from pantr._backend import Backend, use_backend
from pantr.quad import (
    get_gauss_legendre_1d,
    get_modified_chebyshev_nodes_1d,
    get_tanh_sinh_1d,
    get_trapezoidal_1d,
)
from pantr.quad._rules import _tanh_sinh_min_gap
from pantr.quad._rules_core import (
    _gauss_legendre_symmetric_core,
    _generate_tanh_sinh_core,
    _modified_chebyshev_nodes_core,
    _trapezoidal_core,
)

DEFAULT_COUNTS: Final = (17, 64, 200, 1000)
"""Point counts timed by default, spanning the Gauss-Legendre crossover."""

_REPEATS: Final = 15
"""Batches timed per measurement; the minimum of them is reported."""

_MIN_BATCH_MICROSECONDS: Final = 2000.0
"""Target duration of one timed batch.

Below roughly a millisecond the timer's own resolution and the scheduler start to
show, so the batch size is chosen to reach this rather than fixed. That is what
turned an apparent slowdown into the noise it was.
"""


def _microseconds(callable_under_test: Callable[[], object]) -> float:
    """Time one call, in microseconds, as the minimum over repeated batches.

    Args:
        callable_under_test (Callable[[], object]): The call to time.

    Returns:
        float: Microseconds per call.
    """
    once = min(timeit.repeat(callable_under_test, number=1, repeat=3))
    batch = max(1, int(_MIN_BATCH_MICROSECONDS * 1e-6 / max(once, 1e-9)))
    timings = timeit.repeat(callable_under_test, number=batch, repeat=_REPEATS)
    return min(timings) / batch * 1e6


def _report(level: str, rule: str, count: int, python: float, cpp: float) -> None:
    """Print one measured row.

    Args:
        level (str): ``kernel`` or ``entry``.
        rule (str): The rule's name.
        count (int): The point count.
        python (float): Microseconds under the Python backend.
        cpp (float): Microseconds under the C++ backend.
    """
    print(f"{level:<8}{rule:<20}{count:>6}{python:>12.2f}{cpp:>12.2f}{python / cpp:>9.2f}x")


def _bench_kernels(counts: Sequence[int]) -> None:
    """Time each compiled kernel against its Python counterpart.

    Args:
        counts (Sequence[int]): Point counts to time.
    """
    min_gap = _tanh_sinh_min_gap(np.float64)
    for count in counts:
        nodes = np.empty(count, dtype=np.float64)
        weights = np.empty(count, dtype=np.float64)
        for rule, python_call, cpp_call in (
            (
                "gauss_legendre",
                lambda n=count: _gauss_legendre_symmetric_core(n),
                lambda n=count, x=nodes, w=weights: extension.gauss_legendre_symmetric(n, x, w),
            ),
            (
                "trapezoidal",
                lambda n=count: _trapezoidal_core(n),
                lambda n=count, x=nodes, w=weights: extension.trapezoidal(n, x, w),
            ),
            (
                "chebyshev_nodes",
                lambda n=count: _modified_chebyshev_nodes_core(n, np.float64),
                lambda n=count, x=nodes: extension.modified_chebyshev_nodes(n, x),
            ),
            (
                "tanh_sinh",
                lambda n=count: _generate_tanh_sinh_core(n, min_gap),
                lambda n=count, x=nodes, w=weights: extension.generate_tanh_sinh(n, min_gap, x, w),
            ),
        ):
            _report("kernel", rule, count, _microseconds(python_call), _microseconds(cpp_call))


def _bench_entry_points(counts: Sequence[int]) -> None:
    """Time each public entry point under both backends.

    Args:
        counts (Sequence[int]): Point counts to time.
    """
    entries: tuple[tuple[str, Callable[[int], object]], ...] = (
        ("gauss_legendre", lambda n: get_gauss_legendre_1d(n, np.float64)),
        ("trapezoidal", lambda n: get_trapezoidal_1d(n, np.float64)),
        ("chebyshev_nodes", lambda n: get_modified_chebyshev_nodes_1d(n, np.float64)),
        ("tanh_sinh", lambda n: get_tanh_sinh_1d(n, np.float64)),
    )
    for count in counts:
        for rule, entry in entries:
            call = functools.partial(entry, count)
            with use_backend(Backend.PYTHON):
                python = _microseconds(call)
            with use_backend(Backend.CPP):
                cpp = _microseconds(call)
            _report("entry", rule, count, python, cpp)


def main() -> None:
    """Parse the arguments and run both levels.

    Raises:
        SystemExit: If the C++ extension is not built into this installation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_COUNTS),
        help="point counts to time",
    )
    arguments = parser.parse_args()

    from pantr._backend import available_backends  # noqa: PLC0415  (after the parse)

    if Backend.CPP not in available_backends():
        raise SystemExit(
            "the C++ extension is not built into this installation, so there is "
            "nothing to compare against.\n  Fix: pip install -e ."
        )

    print(f"compiler {extension.__compiler__}, fp_contract {extension.__fp_contract__}")
    print(f"{'level':<8}{'rule':<20}{'n':>6}{'python us':>12}{'cpp us':>12}{'speedup':>10}")
    _bench_kernels(arguments.counts)
    _bench_entry_points(arguments.counts)


if __name__ == "__main__":
    main()
