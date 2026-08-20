#!/usr/bin/env python
"""Time the C++ cardinal B-spline kernel against the numba one it is ported from.

The number this prototype exists to produce. Run it as::

    python scripts/bench_parity.py
    python scripts/bench_parity.py --degrees 3 --points 1000 1000000

Two levels, because one level cannot answer the question
---------------------------------------------------------

The first version of this script compared the numba kernel called *directly*
against the C++ kernel called *through the public Python entry point*, and
concluded the port was three times slower. That comparison was wrong, and wrong
in the direction that would have killed the prototype: it charged the C++ side
for the binding, the Layer 2 validation and the numpy round trip while charging
the numba side for none of them.

So both levels are measured, each with both backends on the same footing:

``kernel``
    The compiled kernel alone. numba's ``_tabulate_cardinal_Bspline_basis_1D_core``
    against the raw nanobind function ``pantr._pantr_cpp.tabulate_cardinal_bspline_1d``.
    Both take ``(degree, points, out)`` with ``out`` preallocated, so neither
    allocates and neither validates. **This is the port's own figure.**

``entry``
    :func:`pantr.basis.tabulate_cardinal_bspline_1d`, once per backend. Identical
    Layer 2 validation, identical allocation, identical dispatch on both sides;
    the only difference is which kernel runs underneath. **This is what a caller
    experiences.**

The gap between the two columns is the fixed cost of the Python entry point, and
it is paid by both backends. Reporting only ``entry`` hides how good the kernel
is; reporting only ``kernel`` hides that most callers will never see it.

Threads
-------

The C++ kernel is single-threaded. The numba kernel is ``parallel=True``
*unconditionally* -- unlike the Bernstein kernels it has no serial twin and so
never dispatches to one, paying a fork/join even for three points. It is
therefore timed twice: on every thread, and on one thread via
``numba.set_num_threads(1)``, which lowers the count around a single compiled
kernel. **The one-thread column is the like-for-like comparison.**

Two things that would make the numbers lie, and how they are avoided
--------------------------------------------------------------------

**JIT compilation.** A numba kernel's first call includes compiling it, of the
order of a second, which would swamp everything. Every timed callable is run
once untimed first, and ``wait_for_jit_warmup`` is awaited before anything --
the background warmup thread ``pantr/__init__.py`` starts is not safe against a
concurrent ``parallel=True`` call from another thread, and the failure mode is
an abort rather than an exception.

**The clock.** The reported figure is the *minimum* over repeats, not the mean.
Timing noise on a shared machine is one-sided: interference can only make a run
slower. The minimum is the best estimate of the true cost; the mean measures how
busy the server was.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt

from pantr._backend import Backend, available_backends, use_backend
from pantr._numba_compat import wait_for_jit_warmup
from pantr.basis import tabulate_cardinal_bspline_1d
from pantr.basis._basis_core import _tabulate_cardinal_Bspline_basis_1D_core

DEFAULT_DEGREES: Final[tuple[int, ...]] = (1, 2, 3, 5, 8)
"""Degrees timed by default: the range isogeometric analysis actually uses."""

DEFAULT_POINT_COUNTS: Final[tuple[int, ...]] = (100, 4_096, 100_000, 1_000_000)
"""Point counts timed by default.

Spanning four orders of magnitude, because the answer changes across them: at
100 points a numba launch costs more than the arithmetic it distributes, and at
10^6 both kernels are streaming memory and the gap should close. A measurement
taken at one size reports whichever of those regimes it landed in.

4096 is ``_PARALLEL_MIN_NUM_PTS``, the count at which the *Bernstein* kernels
switch to their serial twin. This kernel has no twin and does not switch, but
the threshold is the library's own estimate of where a parallel launch starts
paying for itself, which makes it a meaningful place to look.
"""

DEFAULT_REPEATS: Final[int] = 7
"""Repeats per case. The reported figure is the minimum over these."""


class Timing(NamedTuple):
    """One case's measurements, in seconds.

    A named record rather than a dict, per the project's convention that a fixed
    set of named values is never carried in a mapping.

    Attributes:
        degree (int): Degree of the basis.
        num_pts (int): Number of evaluation points.
        numba_kernel_par (float): numba kernel alone, every thread.
        numba_kernel_1thr (float): numba kernel alone, one thread.
        cpp_kernel (float): C++ kernel alone, through the raw binding.
        numba_entry (float): Public entry point on the numba backend.
        cpp_entry (float): Public entry point on the C++ backend.
    """

    degree: int
    num_pts: int
    numba_kernel_par: float
    numba_kernel_1thr: float
    cpp_kernel: float
    numba_entry: float
    cpp_entry: float

    @property
    def kernel_speedup(self) -> float:
        """Speedup of the C++ kernel over the one-thread numba kernel.

        The like-for-like figure, and the port's own: both single-threaded, both
        called directly, neither allocating or validating.

        Returns:
            float: Ratio of one-thread numba kernel time to C++ kernel time.
        """
        return self.numba_kernel_1thr / self.cpp_kernel

    @property
    def entry_speedup(self) -> float:
        """Speedup at the public entry point, backend against backend.

        Returns:
            float: Ratio of numba entry-point time to C++ entry-point time.
        """
        return self.numba_entry / self.cpp_entry

    @property
    def entry_overhead(self) -> float:
        """Seconds the Python entry point adds on top of the C++ kernel.

        Paid by both backends, so it is neither backend's fault; reported so the
        two columns can be read against each other.

        Returns:
            float: Difference between the C++ entry-point and kernel times.
        """
        return self.cpp_entry - self.cpp_kernel


def _best_of(call: Callable[[], object], repeats: int) -> float:
    """Time a zero-argument callable, returning the minimum elapsed time.

    Args:
        call (Callable[[], object]): The callable to time.
        repeats (int): Number of timed repetitions.

    Returns:
        float: The smallest elapsed time, in seconds.
    """
    call()  # Warm: JIT compilation, first-touch page faults, cache.
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return best


def _time_case(degree: int, num_pts: int, repeats: int) -> Timing:
    """Time both kernels and both entry points for one case.

    Args:
        degree (int): Degree of the basis.
        num_pts (int): Number of evaluation points.
        repeats (int): Repetitions per measurement.

    Returns:
        Timing: The five measurements.
    """
    import numba  # noqa: PLC0415  (imported here so --help works without it)

    from pantr import _pantr_cpp  # noqa: PLC0415  (absent unless the port is built)

    pts: npt.NDArray[np.float64] = np.linspace(0.0, 1.0, num_pts, dtype=np.float64)
    out: npt.NDArray[np.float64] = np.empty((num_pts, degree + 1), dtype=np.float64)
    n = np.int32(degree)

    max_threads = numba.get_num_threads()

    numba.set_num_threads(max_threads)
    numba_kernel_par = _best_of(
        lambda: _tabulate_cardinal_Bspline_basis_1D_core(n, pts, out), repeats
    )

    # set_num_threads can only LOWER the count -- NUMBA_NUM_THREADS is read once
    # at import and cannot be raised afterwards -- which is the direction needed.
    numba.set_num_threads(1)
    try:
        numba_kernel_1thr = _best_of(
            lambda: _tabulate_cardinal_Bspline_basis_1D_core(n, pts, out), repeats
        )
    finally:
        numba.set_num_threads(max_threads)

    # The raw binding, not the entry point: same contract as the numba kernel
    # above, so the two are comparable.
    cpp_kernel = _best_of(
        lambda: _pantr_cpp.tabulate_cardinal_bspline_1d(degree, pts, out), repeats
    )

    with use_backend(Backend.PYTHON):
        numba_entry = _best_of(lambda: tabulate_cardinal_bspline_1d(degree, pts, out=out), repeats)
    with use_backend(Backend.CPP):
        cpp_entry = _best_of(lambda: tabulate_cardinal_bspline_1d(degree, pts, out=out), repeats)

    return Timing(
        degree=degree,
        num_pts=num_pts,
        numba_kernel_par=numba_kernel_par,
        numba_kernel_1thr=numba_kernel_1thr,
        cpp_kernel=cpp_kernel,
        numba_entry=numba_entry,
        cpp_entry=cpp_entry,
    )


def _format_table(timings: list[Timing]) -> str:
    """Render the measurements as a fixed-width table, times in milliseconds.

    Args:
        timings (list[Timing]): The measurements to render.

    Returns:
        str: The rendered table.
    """
    header = (
        f"{'deg':>4} {'points':>9} | {'nb par':>9} {'nb 1thr':>9} {'cpp':>9} {'kernel':>8}"
        f" | {'nb entry':>9} {'cpp entry':>9} {'entry':>8}"
    )
    lines = [header, "-" * len(header)]
    for t in timings:
        lines.append(
            f"{t.degree:>4} {t.num_pts:>9} | "
            f"{t.numba_kernel_par * 1e3:>9.3f} {t.numba_kernel_1thr * 1e3:>9.3f} "
            f"{t.cpp_kernel * 1e3:>9.3f} {t.kernel_speedup:>7.2f}x | "
            f"{t.numba_entry * 1e3:>9.3f} {t.cpp_entry * 1e3:>9.3f} {t.entry_speedup:>7.2f}x"
        )
    return "\n".join(lines)


def main() -> int:
    """Run the benchmark and print the table.

    Returns:
        int: Process exit status; 1 if the C++ backend is not available.
    """
    parser = argparse.ArgumentParser(description="C++ against numba, kernel and entry point")
    parser.add_argument("--degrees", type=int, nargs="+", default=list(DEFAULT_DEGREES))
    parser.add_argument("--points", type=int, nargs="+", default=list(DEFAULT_POINT_COUNTS))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()

    if Backend.CPP not in available_backends():
        print(
            "The C++ backend is not available in this installation, so there is\n"
            "nothing to compare against.\n"
            "  Fix: pip install -e . --no-build-isolation"
        )
        return 1

    # numba's background warmup thread is not safe against a concurrent
    # parallel=True call: the process aborts rather than raising. See CLAUDE.md.
    wait_for_jit_warmup()

    import numba  # noqa: PLC0415

    from pantr import _pantr_cpp  # noqa: PLC0415

    print(f"numba threads:  {numba.get_num_threads()}")
    print(f"C++ built by:   {_pantr_cpp.__compiler__}")
    print(f"FMA on target:  {_pantr_cpp.__fp_contract__}")
    print("this kernel is parallel=True unconditionally; it has no serial twin")
    print()

    timings = [
        _time_case(degree, num_pts, args.repeats)
        for degree in args.degrees
        for num_pts in args.points
    ]
    print(_format_table(timings))
    print(
        "\nTimes in ms, minimum over repeats.\n"
        "  'kernel' -- C++ against ONE numba thread, both called directly. The port's figure.\n"
        "  'entry'  -- both through the public entry point. What a caller sees.\n"
        "The gap between the two is the Python entry point's fixed cost, paid by both."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
