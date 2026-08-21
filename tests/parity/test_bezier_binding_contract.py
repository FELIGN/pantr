"""What the eight Bézier bindings refuse, and why each refusal has to be a test.

`cpp/bindings/bezier.cpp` is the C++ half of Layer 2: the kernels behind it validate
nothing, so every precondition they rely on is established here. That is only true if
the checks exist, and three classes of them did not until a deep review found them.
Each is pinned below with the failure it prevents, because each failure is silent or
worse.

**A degree inferred from a shape has a floor.** Five kernels take no ``degree``
argument and compute ``ctrl.extent(0) - 1`` in ``std::size_t``. A zero-row ``ctrl``
underflowed to ``SIZE_MAX`` and the kernel walked off the allocation: measured,
SIGSEGV with no exception, on all five. `basis.cpp` avoids the class by taking
``degree`` as an ``unsigned`` the caster refuses to make negative; there is no such
argument here.

**Two same-shaped output buffers can be the same buffer.** ``split_bezier_1d``
returned the right half under both names when handed one array twice.
``nb::kw_only()`` closes transposition and nothing closed aliasing.
`cpp/bindings/quad.cpp` had carried the guard for its own two-output rules since the
quad port; it simply was not brought across.

**A documented domain is a promise.** ``value``, ``lower`` and ``upper`` were
documented as living in the unit interval and never checked, so ``slice`` at 2.5
extrapolated silently and ``restrict`` with its bounds transposed returned a
different, plausible Bézier.

These are reachable by importing :mod:`pantr._pantr_cpp`, not through
:class:`~pantr.bezier.Bezier`, whose Layer 1 refuses all of them first. That is
exactly the surface `basis.cpp` documents as real: the extension is importable and
every name here is a public attribute of a public module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

_CTRL = np.ascontiguousarray(np.arange(8.0).reshape(4, 2))
"""A well-formed degree-3, rank-2 control net, so only the argument under test is bad."""


def _bindings() -> Any:
    """Import the extension.

    Returns:
        Any: The :mod:`pantr._pantr_cpp` module.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    return _pantr_cpp


@pytest.mark.parametrize(
    "call",
    [
        "evaluate_bezier_1d",
        "evaluate_bezier_deriv_1d",
        "slice_bezier_1d",
        "split_bezier_1d",
        "restrict_bezier_1d",
    ],
)
def test_an_empty_control_net_raises_rather_than_crashing(cpp_backend: None, call: str) -> None:
    """The five kernels that infer a degree from a shape refuse a zero-row ``ctrl``.

    Before the check, every one of these exited with SIGSEGV. A test cannot observe a
    segfault in its own process, so what it observes instead is the exception that now
    stands in its place: if the check is removed, this test does not fail, it takes the
    whole run down, which is a louder signal than a red.
    """
    del cpp_backend
    empty = np.empty((0, 2))
    bindings = _bindings()
    invocations = {
        "evaluate_bezier_1d": lambda: bindings.evaluate_bezier_1d(
            empty, np.array([0.5]), out=np.empty((1, 2))
        ),
        "evaluate_bezier_deriv_1d": lambda: bindings.evaluate_bezier_deriv_1d(
            empty, np.array([0.5]), 1, out=np.empty((1, 2, 2))
        ),
        "slice_bezier_1d": lambda: bindings.slice_bezier_1d(empty, 0.5, out=np.empty(2)),
        "split_bezier_1d": lambda: bindings.split_bezier_1d(
            empty, 0.5, out_left=np.empty((0, 2)), out_right=np.empty((0, 2))
        ),
        "restrict_bezier_1d": lambda: bindings.restrict_bezier_1d(
            empty, 0.2, 0.8, out=np.empty((0, 2))
        ),
    }
    with pytest.raises(ValueError, match="no rows"):
        invocations[call]()


def test_split_refuses_two_outputs_that_are_the_same_array(cpp_backend: None) -> None:
    """Aliased halves are refused rather than silently collapsed into one.

    Measured before the guard: the call returned the right half under both names, with
    no exception. Two views onto one buffer alias just as destructively, so the guard
    tests for overlap rather than identity, and this checks both.
    """
    del cpp_backend
    bindings = _bindings()

    same = np.zeros((4, 2))
    with pytest.raises(ValueError, match="overlap in memory"):
        bindings.split_bezier_1d(_CTRL, 0.5, out_left=same, out_right=same)

    # Two halves of one allocation: different objects, overlapping storage.
    holder = np.zeros((8, 2))
    with pytest.raises(ValueError, match="overlap in memory"):
        bindings.split_bezier_1d(_CTRL, 0.5, out_left=holder[:4], out_right=holder[3:7])

    # Adjacent but disjoint is legal, which is what makes the guard about overlap and
    # not about sharing a base array.
    bindings.split_bezier_1d(_CTRL, 0.5, out_left=holder[:4], out_right=holder[4:])
    assert np.any(holder != 0.0)


@pytest.mark.parametrize("value", [-0.5, 1.5, 2.5, 5.0, float("nan"), float("inf")])
def test_a_parameter_outside_the_unit_interval_is_refused(cpp_backend: None, value: float) -> None:
    """``slice`` and ``split`` refuse a parameter their docstring says is in ``[0, 1]``.

    NaN is in the list deliberately: the check is written as a conjunction of two
    comparisons rather than a negated range, because a NaN compares false against
    everything and a negated range would have admitted it.
    """
    del cpp_backend
    bindings = _bindings()

    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        bindings.slice_bezier_1d(_CTRL, value, out=np.empty(2))

    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        bindings.split_bezier_1d(
            _CTRL, value, out_left=np.empty((4, 2)), out_right=np.empty((4, 2))
        )


def test_restrict_refuses_transposed_bounds(cpp_backend: None) -> None:
    """``lower`` and ``upper`` are ordered, which is what closes the transposition trap.

    They are adjacent same-typed positional parameters, so nothing in the type system
    separates them, and a transposed call used to return a different and entirely
    plausible restriction. Ordering them makes that call an error instead.
    """
    del cpp_backend
    bindings = _bindings()

    with pytest.raises(ValueError, match="must not exceed upper"):
        bindings.restrict_bezier_1d(_CTRL, 0.8, 0.2, out=np.empty((4, 2)))

    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
            bindings.restrict_bezier_1d(_CTRL, bad, 1.0, out=np.empty((4, 2)))
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
            bindings.restrict_bezier_1d(_CTRL, 0.0, bad, out=np.empty((4, 2)))

    # A zero-width interval is admitted: the kernel is well defined there, and Layer 1
    # is the layer that decides whether it is a useful thing to ask for.
    bindings.restrict_bezier_1d(_CTRL, 0.5, 0.5, out=np.empty((4, 2)))
