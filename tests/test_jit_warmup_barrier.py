"""Every Layer 2 entry point over a ``parallel=True`` kernel waits for the JIT warmup.

``pantr/__init__.py`` compiles (and runs) Numba kernels on a background thread at import,
and Numba's default workqueue threading layer is not safe against a concurrent
``parallel=True`` call from another thread: the process *aborts* rather than raising, which
takes the whole session with it.  ``CLAUDE.md`` states the rule that follows -- a Layer 2
entry point over such kernels calls :func:`pantr._numba_compat.wait_for_jit_warmup` first --
and FELIGN/pantr#418 records the gap this module closes.

**These tests assert the call, not the absence of a crash, and that is deliberate.**  The
barrier is a once-per-process event, so by the time any in-process test runs the warmup is
long finished and nothing in-process can observe the race; a test that merely ran the entry
point and checked it did not abort would pass whether or not the call is there.  Each case
below patches the barrier *in the entry point's own module* together with a sentinel on the
way to the kernel, so deleting the call from any one entry point fails exactly that case.
The same reasoning, and the measurements behind it, are in
``tests/test_bspline_locate.py::TestNumbaWarmup``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bezier import Bezier, fit_bezier
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D

# ---------------------------------------------------------------------------
# Objects the drivers exercise
# ---------------------------------------------------------------------------


def _bezier_1d() -> Bezier:
    """Build a planar quadratic Bézier curve.

    Returns:
        ~pantr.bezier.Bezier: A ``dim == 1``, ``rank == 2`` Bézier.
    """
    return Bezier(np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]]))


def _bezier_2d() -> Bezier:
    """Build a bilinear Bézier surface in 3D.

    Returns:
        ~pantr.bezier.Bezier: A ``dim == 2``, ``rank == 3`` Bézier.
    """
    return Bezier(np.arange(12.0).reshape(2, 2, 3))


def _general_space_1d() -> BsplineSpace1D:
    """Build a 1D space whose knots are *not* Bézier-like.

    The interior knot matters: with Bézier-like knots the tabulation takes its Bernstein
    branch and never reaches the kernel accessor the corresponding case uses as a sentinel.

    Returns:
        ~pantr.bspline.BsplineSpace1D: A degree-2 space over ``[0, 1]`` with one interior
        knot.
    """
    return BsplineSpace1D(np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]), 2)


def _bspline_1d() -> Bspline:
    """Build a scalar 1D B-spline over :func:`_general_space_1d`.

    Returns:
        ~pantr.bspline.Bspline: A ``dim == 1`` B-spline.
    """
    space = BsplineSpace([_general_space_1d()])
    return Bspline(space, np.arange(1.0, space.num_total_basis + 1.0))


_PTS: npt.NDArray[np.float64] = np.linspace(0.0, 1.0, 5)
"""Parametric points shared by the evaluation drivers."""


# ---------------------------------------------------------------------------
# The census: one case per Layer 2 entry point
# ---------------------------------------------------------------------------


class _BarrierCase(NamedTuple):
    """One Layer 2 entry point over ``parallel=True`` kernels, and how to drive it.

    Attributes:
        entry_point (str): ``module.function`` of the entry point, used as the test id.
        module (str): Module whose ``wait_for_jit_warmup`` name is patched. This is the
            entry point's own module, so the case fails if that one call is deleted even
            when a caller elsewhere still has one.
        sentinel (str): Attribute of ``module`` that the entry point calls on its way to a
            ``parallel=True`` kernel. The barrier must precede it.
        drive (Callable[[], Any]): Public-API call that reaches the entry point.
    """

    entry_point: str
    module: str
    sentinel: str
    drive: Callable[[], Any]


_CASES: tuple[_BarrierCase, ...] = (
    # -- pantr.bezier ------------------------------------------------------
    _BarrierCase(
        "_bezier_utils._tabulate_bernstein_1d_fast",
        "pantr.bezier._bezier_utils",
        "_tabulate_Bernstein_basis_1D_core",
        # Driven through `fit_bezier` rather than `interpolate_bezier`: the scattered path
        # builds its Vandermonde uncached, so the helper is reached whatever ran before.
        lambda: fit_bezier(np.zeros(4), np.array([[0.1], [0.4], [0.6], [0.9]]), degree=1, tol=None),
    ),
    _BarrierCase(
        "_bezier_eval._evaluate_bezier",
        "pantr.bezier._bezier_eval",
        "_evaluate_bezier_1d",
        lambda: _bezier_1d().evaluate(_PTS),
    ),
    _BarrierCase(
        "_bezier_eval._evaluate_bezier_deriv",
        "pantr.bezier._bezier_eval",
        "_evaluate_bezier_deriv_1d",
        lambda: _bezier_1d().evaluate_derivatives(_PTS, [1]),
    ),
    _BarrierCase(
        "_bezier_slice._slice_bezier",
        "pantr.bezier._bezier_slice",
        "slice_nd_kernel",
        lambda: _bezier_2d().slice(0, 0.5),
    ),
    _BarrierCase(
        "_bezier_split._split_bezier",
        "pantr.bezier._bezier_split",
        "split_nd_kernel",
        lambda: _bezier_2d().split(0, 0.5),
    ),
    _BarrierCase(
        "_bezier_restrict._restrict_bezier",
        "pantr.bezier._bezier_restrict",
        "restrict_nd_kernel",
        lambda: _bezier_2d().restrict([(0.25, 0.75), None]),
    ),
    _BarrierCase(
        "_bezier_compose._compose_bezier",
        "pantr.bezier._bezier_compose",
        "compose_kernel",
        lambda: _bezier_1d().compose(Bezier(np.array([[0.2], [0.8]]))),
    ),
    _BarrierCase(
        "_bezier_collapse._collapse_along_axis",
        "pantr.bezier._bezier_collapse",
        "collapse_kernel",
        lambda: _bezier_2d().collapse_along_axis(0, [0.5]),
    ),
    # -- pantr.bspline -----------------------------------------------------
    _BarrierCase(
        "_bspline_basis_core._tabulate_Bspline_basis_1D_impl",
        "pantr.bspline._bspline_basis_core",
        "bspline_basis_core",
        lambda: _general_space_1d().tabulate_basis(_PTS),
    ),
    _BarrierCase(
        "_bspline_basis_core._tabulate_Bspline_basis_deriv_1D_impl",
        "pantr.bspline._bspline_basis_core",
        "bspline_basis_deriv_core",
        lambda: _general_space_1d().tabulate_basis_derivatives(_PTS, 1),
    ),
    _BarrierCase(
        "_bspline_eval._evaluate_Bspline_1D",
        "pantr.bspline._bspline_eval",
        "_evaluate_Bspline_basis_combine_1D",
        lambda: _bspline_1d().evaluate(_PTS),
    ),
    _BarrierCase(
        "_bspline_eval._evaluate_Bspline_deriv_1D",
        "pantr.bspline._bspline_eval",
        "_evaluate_Bspline_deriv_1D_non_rational",
        lambda: _bspline_1d().evaluate_derivatives(_PTS, [1]),
    ),
    _BarrierCase(
        "_bspline_to_beziers._to_beziers_impl",
        "pantr.bspline._bspline_to_beziers",
        "_apply_bezier_extraction_1d_core",
        lambda: _bspline_1d().to_beziers(),
    ),
)
"""The Layer 2 entry points whose barrier this module pins, one case each."""


class TestNumbaWarmupBarrier:
    """Each Layer 2 entry point over parallel kernels reaches the barrier first."""

    @pytest.mark.parametrize("case", _CASES, ids=lambda case: case.entry_point)
    def test_the_barrier_runs_before_the_kernel_is_reached(
        self, case: _BarrierCase, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The entry point calls the barrier, and calls it before it reaches a kernel.

        Both names are patched in the entry point's *own* module namespace, so a case can
        only pass on the call that module makes: deleting ``wait_for_jit_warmup()`` from one
        entry point fails that entry point's case and no other.

        Args:
            case (_BarrierCase): The entry point under test.
            monkeypatch (pytest.MonkeyPatch): Patches the two module-level names.
        """
        calls: list[str] = []
        module = importlib.import_module(case.module)
        real_sentinel = getattr(module, case.sentinel)

        def record_barrier() -> None:
            calls.append("barrier")

        def record_sentinel(*args: Any, **kwargs: Any) -> Any:
            calls.append("kernel")
            return real_sentinel(*args, **kwargs)

        monkeypatch.setattr(module, "wait_for_jit_warmup", record_barrier)
        monkeypatch.setattr(module, case.sentinel, record_sentinel)

        case.drive()

        assert "barrier" in calls, f"{case.entry_point} must wait for the JIT warmup"
        assert "kernel" in calls, (
            f"{case.sentinel} was never reached, so this case proves nothing about "
            f"{case.entry_point}; the driver no longer takes the intended branch"
        )
        assert calls.index("barrier") < calls.index("kernel"), (
            f"the barrier must precede the first kernel call; got {calls[:3]}"
        )

    def test_every_case_names_a_distinct_entry_point(self) -> None:
        """No two cases cover the same entry point, so the count is the coverage.

        A copy-paste that left two cases pointing at one function would look like coverage
        of two entry points while leaving one unpinned.
        """
        names = [case.entry_point for case in _CASES]
        assert len(set(names)) == len(names), f"duplicate entry points in the census: {names}"
