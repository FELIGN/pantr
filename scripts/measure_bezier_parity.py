#!/usr/bin/env python
"""Reproduce the parity figures the Bézier port's prose quotes.

A deep review found that the headline measurements in
``design/backend_parity.md`` Rule 9, ``cpp/include/pantr/bezier/bezier.hpp`` and
``tests/parity/test_bezier_arithmetic.py`` had no reproducible artifact anywhere in
the tree: the scripts that produced them lived in a scratch directory and are gone.
``cpp/include/pantr/basis/bernstein.hpp`` does the opposite for its own ``pow``
claim, pointing at a committed test, and that is the convention. This script is the
missing half.

It is **not** a test. The parity suite asserts; this reports, because the numbers in
the prose are counts over specific grids and a reader deciding whether to believe
them wants the grid, not a green tick.

Run it as::

    PYTHONPATH="$(pwd)/src" .venv/bin/python scripts/measure_bezier_parity.py

Two figures it cannot reproduce, and says so rather than omitting them: the
``-march=native`` movement counts need a second build of the extension with that
flag, which is a rebuild rather than a run. The command is printed at the end.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import partial
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from pantr._backend import Backend, available_backends, use_backend
from pantr.bezier import Bezier

DEGREES: Final = (0, 1, 2, 3, 4, 7, 11, 17, 25)
"""The degree sweep the whole-kernel figure is taken over."""

SLICE_DEGREES: Final = (0, 1, 2, 3, 5, 8, 13, 21, 34)
"""The degree sweep the de Casteljau figure is taken over."""

PARAMS: Final = (
    0.0,
    1e-300,
    1e-20,
    1e-8,
    0.25,
    0.5,
    0.5 + 2**-52,
    0.75,
    1.0 - 1e-8,
    float(np.nextafter(1.0, 0.0)),
    1.0,
)
"""Adversarial parameters: both endpoints, either side of the mirror threshold, and
values chosen so that ``1 - (1 - u)`` loses them."""


def _mixed(
    rng: np.random.Generator, shape: tuple[int, ...], dtype: npt.DTypeLike
) -> npt.NDArray[np.floating[Any]]:
    """Random control points spanning twelve decades, so cancellation has work to do.

    Args:
        rng (np.random.Generator): Source of randomness.
        shape (tuple[int, ...]): Shape of the net.
        dtype (npt.DTypeLike): Storage format.

    Returns:
        npt.NDArray[np.floating[Any]]: The control points, C-contiguous.
    """
    values = rng.standard_normal(shape) * 10.0 ** rng.integers(-6, 7, shape)
    return np.ascontiguousarray(values, dtype=dtype)


def _bit_differences(
    actual: npt.NDArray[np.floating[Any]], reference: npt.NDArray[np.floating[Any]]
) -> int:
    """Count elements whose bit patterns differ, NaN pairs excluded.

    Args:
        actual (npt.NDArray[np.floating[Any]]): The C++ backend's result.
        reference (npt.NDArray[np.floating[Any]]): The Numba oracle's result.

    Returns:
        int: How many elements differ.
    """
    unsigned = np.uint64 if actual.dtype == np.float64 else np.uint32
    differ = actual.view(unsigned) != reference.view(unsigned)
    both_nan = np.isnan(actual) & np.isnan(reference)
    return int((differ & ~both_nan).sum())


def _as_array(result: object) -> npt.NDArray[np.floating[Any]]:
    """Coerce a public-API return to the array to compare.

    :meth:`~pantr.bezier.Bezier.slice` returns a bare array when the Bézier is
    univariate and a :class:`~pantr.bezier.Bezier` otherwise, so the caller cannot
    know which without branching.

    Args:
        result (object): Whatever the method returned.

    Returns:
        npt.NDArray[np.floating[Any]]: The values, as an array.
    """
    control_points = getattr(result, "control_points", result)
    return np.asarray(control_points)


def _sliced(ctrl: npt.NDArray[np.floating[Any]], value: float) -> npt.NDArray[np.floating[Any]]:
    """Evaluate a univariate Bézier at one parameter.

    Args:
        ctrl (npt.NDArray[np.floating[Any]]): Control points.
        value (float): Parameter in ``[0, 1]``.

    Returns:
        npt.NDArray[np.floating[Any]]: The point, one entry per column.
    """
    return _as_array(Bezier(ctrl).slice(0, value))


def _evaluated(
    ctrl: npt.NDArray[np.floating[Any]], points: npt.NDArray[np.floating[Any]]
) -> npt.NDArray[np.floating[Any]]:
    """Evaluate a Bézier at every point.

    Args:
        ctrl (npt.NDArray[np.floating[Any]]): Control points.
        points (npt.NDArray[np.floating[Any]]): Evaluation points.

    Returns:
        npt.NDArray[np.floating[Any]]: The values.
    """
    return _as_array(Bezier(ctrl).evaluate(points))


def _differentiated(
    ctrl: npt.NDArray[np.floating[Any]], points: npt.NDArray[np.floating[Any]]
) -> npt.NDArray[np.floating[Any]]:
    """Evaluate a Bézier's derivatives up to order two.

    Args:
        ctrl (npt.NDArray[np.floating[Any]]): Control points.
        points (npt.NDArray[np.floating[Any]]): Evaluation points.

    Returns:
        npt.NDArray[np.floating[Any]]: The derivative values.
    """
    return _as_array(Bezier(ctrl).evaluate_derivatives(points, 2))


def _elevated(ctrl: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
    """Raise a Bézier's degree by two.

    Args:
        ctrl (npt.NDArray[np.floating[Any]]): Control points.

    Returns:
        npt.NDArray[np.floating[Any]]: The elevated control points.
    """
    return _as_array(Bezier(ctrl).elevate_degree(2))


def _restricted(ctrl: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
    """Restrict a Bézier to a sub-interval.

    Args:
        ctrl (npt.NDArray[np.floating[Any]]): Control points.

    Returns:
        npt.NDArray[np.floating[Any]]: The restricted control points.
    """
    return _as_array(Bezier(ctrl).restrict((0.1, 0.9)))


def _left_half(ctrl: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.floating[Any]]:
    """Split a Bézier and return the left half's control points.

    Args:
        ctrl (npt.NDArray[np.floating[Any]]): Control points.

    Returns:
        npt.NDArray[np.floating[Any]]: The left half.
    """
    return _as_array(Bezier(ctrl).split(0, 0.25)[0])


def _both(
    build: Callable[[], npt.NDArray[np.floating[Any]]],
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """Run ``build`` under each backend and return the two results.

    Args:
        build (Callable[[], npt.NDArray[np.floating[Any]]]): Produces the array to
            compare, using whichever backend is in effect.

    Returns:
        tuple: The Numba result and the C++ result, in that order.
    """
    with use_backend(Backend.PYTHON):
        reference = build()
    with use_backend(Backend.CPP):
        actual = build()
    return reference, actual


def measure_de_casteljau() -> tuple[int, int]:
    """Count values and bit differences on the de Casteljau grid.

    The grid that produces the "630 values" figure: nine degrees, ten parameters,
    seven columns. The endpoints are excluded because ``slice`` short-circuits them,
    which is what makes the count 630 rather than 693.

    Returns:
        tuple[int, int]: Values compared, and bit differences, summed over dtypes.
    """
    total = 0
    bad = 0
    for dtype in (np.float64, np.float32):
        rng = np.random.default_rng(20260821)
        for degree in SLICE_DEGREES:
            ctrl = _mixed(rng, (degree + 1, 7), dtype)
            for value in PARAMS[:-1]:
                reference, actual = _both(partial(_sliced, ctrl, value))
                total += reference.size
                bad += _bit_differences(actual, reference)
    return total, bad


def measure_every_kernel() -> tuple[int, int]:
    """Count values and bit differences across all seven arithmetic kernels.

    The sweep behind the "3616 values" figure. Each kernel is driven through the
    public :class:`~pantr.bezier.Bezier` surface, so what is compared is what a
    caller gets rather than what a binding returns.

    Returns:
        tuple[int, int]: Values compared, and bit differences, summed over dtypes.
    """
    total = 0
    bad = 0
    for dtype in (np.float64, np.float32):
        rng = np.random.default_rng(20260821)
        for degree in DEGREES:
            for rank in (1, 3):
                ctrl = _mixed(rng, (degree + 1, rank), dtype)
                points: npt.NDArray[np.floating[Any]] = np.ascontiguousarray(PARAMS, dtype=dtype)
                cases: list[Callable[[], npt.NDArray[np.floating[Any]]]] = [
                    partial(_evaluated, ctrl, points),
                    partial(_differentiated, ctrl, points),
                    partial(_elevated, ctrl),
                    partial(_restricted, ctrl),
                ]
                if degree >= 1:
                    cases.append(partial(_left_half, ctrl))
                for build in cases:
                    reference, actual = _both(build)
                    total += reference.size
                    bad += _bit_differences(actual, reference)
    return total, bad


def measure_pow_agreement() -> tuple[int, int]:
    """Count agreements between numba's ``np.power`` and the platform ``powf``.

    The evaluation kernel's mirrored branch seeds its recurrence with ``u`` raised
    at **storage width**, so at ``float32`` the C++ calls ``powf``. The comparison
    has to call the same symbol, which is what the ``ctypes`` handle below does; an
    earlier draft of this script computed in ``double`` and narrowed instead, and
    reported 1305 differences out of 1280256. That number is not a refutation of the
    parity claim, it is a measurement of the mistake this port made once and fixed:
    computing the mirrored seed at the wrong width.

    Returns:
        tuple[int, int]: Pairs compared, and pairs differing.
    """
    import ctypes  # noqa: PLC0415

    from numba import njit  # noqa: PLC0415

    libm = ctypes.CDLL("libm.so.6")
    libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
    libm.powf.restype = ctypes.c_float

    @njit(cache=False)  # type: ignore[untyped-decorator]  # numba's is untyped
    def seeds(bases: npt.NDArray[np.float32], power: int, out: npt.NDArray[np.float32]) -> None:
        for i in range(bases.size):
            out[i] = np.power(bases[i], power)

    rng = np.random.default_rng(7)
    # The mirrored branch only ever raises a base in (0.5, 1], plus the exact
    # endpoints, so that is the whole domain this claim has to cover.
    bases: npt.NDArray[np.float32] = np.concatenate(
        [rng.uniform(0.5, 1.0, 20000), np.array([0.5, 0.75, 1.0, 0.9999999])]
    ).astype(np.float32)
    total = 0
    bad = 0
    reference = np.empty(bases.size, dtype=np.float32)
    for power in range(1, 65):
        seeds(bases, power, reference)
        got = np.array([libm.powf(float(b), float(power)) for b in bases], dtype=np.float32)
        total += bases.size
        bad += int((reference.view(np.uint32) != got.view(np.uint32)).sum())
    return total, bad


def main() -> int:
    """Report every figure this script can reproduce without a rebuild.

    Returns:
        int: 0 if every count agrees bit for bit, 1 otherwise.
    """
    if Backend.CPP not in available_backends():
        print("the C++ backend is not built; nothing to compare")
        return 1

    rows = [
        ("de Casteljau grid (slice)", *measure_de_casteljau()),
        ("all seven kernels", *measure_every_kernel()),
        ("pow, numba against libm", *measure_pow_agreement()),
    ]
    print(f"{'measurement':<28} {'values':>10} {'differing':>10}")
    for label, total, bad in rows:
        print(f"{label:<28} {total:>10} {bad:>10}")

    print(
        "\nNot reproducible from a run: the -march=native movement counts, which need a\n"
        "second build. To take them:\n"
        "  cmake --preset gcc -B build/native -DCMAKE_CXX_FLAGS=-march=native\n"
        "  cmake --build build/native && ctest --test-dir build/native\n"
        "then re-run this script against an extension built the same way."
    )
    return 0 if all(bad == 0 for _, _, bad in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
