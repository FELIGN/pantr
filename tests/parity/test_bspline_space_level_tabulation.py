"""The C++ space-level dispatch, compared against the oracle's rather than itself.

``cpp/include/pantr/bspline/tabulate.hpp`` offers two pairs of free functions. The
raw-knot pair is what :mod:`pantr.bspline._basis_backend` calls, and that will not
change: building a space per call would re-validate and copy the knot vector in front
of a kernel the oracle calls with two or three points, and the constructor snaps by
default, so it could silently evaluate on a different vector than the oracle did.

The *space-level* pair -- ``tabulate_basis_1d`` and ``tabulate_basis_derivatives_1d``
-- is the Layer 2 equivalent for a C++ caller with no interpreter, and it is the code
a C++ consumer writes. It chooses between the general recurrence and the Bézier-like
fast path, and on that path it commits its own change of variable and its own
chain-rule scaling. **Until these bindings existed, ``ctest`` was the only thing that
ran it**, which tests it against itself: a `ctest` case cannot notice the two sides
disagreeing about which path a space takes, or about the arithmetic on the way into
it.

The claim is **bitwise**, and it is available rather than generous. The C++ dispatch
and the oracle's reach the same kernels through the same decisions, so where the
values are not identical something has genuinely diverged, and a bound would hide it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pantr._backend import Backend, use_backend
from pantr.bspline import BsplineSpace1D
from tests._parity_harness import demand_the_compiled_kernel

_CASES: dict[str, tuple[list[float], int]] = {
    # General knots: the recurrence path.
    "clamped-uniform-p2": ([0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0], 2),
    "clamped-repeat-p2": ([0.0, 0.0, 0.0, 0.25, 0.75, 0.75, 1.0, 1.0, 1.0], 2),
    "unclamped-p2": ([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5], 2),
    # No interior knot: both sides must pick the Bézier-like path, and the domain is
    # neither `[0, 1]` nor dyadic, so the change of variable is not a no-op and a
    # divergence in it cannot cancel.
    "bezier-like-p2": ([2.0, 2.0, 2.0, 6.0, 6.0, 6.0], 2),
    "bezier-like-p3": ([0.1, 0.1, 0.1, 0.1, 2.2, 2.2, 2.2, 2.2], 3),
    # Degree 0, where the recurrence has no step at all.
    "degree-zero": ([0.0, 0.25, 0.5, 1.0], 0),
}
"""One space per path and per decision the dispatch makes."""

_PERIODIC_CASES: dict[str, tuple[list[float], int]] = {
    "periodic-uniform-p2": ([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], 2),
    "periodic-uniform-p3": ([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], 3),
}
"""Periodic spaces, kept apart because the dispatch forwards a flag for them.

``tabulate.hpp`` calls the periodic arm of ``find_span_and_first_basis`` a live path
rather than dead code, and the raw-knot parity file reaches it by passing ``periodic``
explicitly to the Layer 3 kernel. Nothing was checking that the *space-level* dispatch
forwards ``space.periodic()`` at all: an edit that dropped or hardcoded the flag would
have left this file green.
"""

_POINTS_BY_CASE = {
    name: np.linspace(knots[degree], knots[len(knots) - degree - 1], 7)
    for name, (knots, degree) in (_CASES | _PERIODIC_CASES).items()
}
"""Seven points spanning each space's own domain, endpoints included."""


def _cpp_space(knots: list[float], degree: int, dtype: Any, *, periodic: bool = False) -> Any:
    """Build the bound C++ space for these knots.

    Args:
        knots (list[float]): The knot vector.
        degree (int): The polynomial degree.
        dtype (Any): ``np.float32`` or ``np.float64``.
        periodic (bool): Whether the space is periodic. Defaults to False.

    Returns:
        Any: The space the bindings take, reached through the public wrapper so that
        the snapping and validation the oracle applied are the ones applied here.
    """
    with use_backend(Backend.CPP):
        return BsplineSpace1D(np.array(knots, dtype=dtype), degree, periodic=periodic)._impl


def _gate_the_interpreted_oracle(case: str, dtype: Any, *, n_deriv: int | None) -> None:
    """Skip the cases whose bitwise claim is about the compiled oracle only.

    Rule 12: under ``NUMBA_DISABLE_JIT=1`` the oracle is a different object, so a
    bitwise claim against it is a claim about something this project does not ship.

    **The gate is scoped to what was measured, which is narrower than the rule.** All
    24 instances were run against the interpreted oracle: only the two Bézier-like
    ``float32`` *value* cases diverge, by up to 6e-8. The four non-Bézier value cases
    and all eighteen derivative cases -- Bézier included, since row 0 of the
    derivative kernel does not show what the standalone value kernel does -- are
    identical with the JIT on or off. Gating all of them, which a first version did,
    would have taken 22 instances out of the coverage run for no measured reason and
    left a future regression in any of them silently skipped instead of caught.

    Args:
        case (str): Key into :data:`_CASES`.
        dtype (Any): The storage dtype under test.
        n_deriv (int | None): The derivative order, or ``None`` for the values.
    """
    if n_deriv is None and dtype is np.float32 and case.startswith("bezier-like"):
        demand_the_compiled_kernel(dtype)


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_space_level_basis_matches_the_oracle_bitwise(
    cpp_backend: None, case: str, dtype: Any
) -> None:
    del cpp_backend
    _gate_the_interpreted_oracle(case, dtype, n_deriv=None)
    from pantr import _pantr_cpp  # noqa: PLC0415  (only this module needs it)

    knots, degree = _CASES[case]
    points = _POINTS_BY_CASE[case].astype(dtype)
    space = _cpp_space(knots, degree, dtype)

    out_basis = np.zeros((points.size, degree + 1), dtype=dtype)
    out_first = np.zeros(points.size, dtype=np.int64)
    _pantr_cpp.tabulate_bspline_space_basis_1d(
        space, points, out_basis=out_basis, out_first_basis=out_first
    )

    with use_backend(Backend.PYTHON):
        expected, expected_first = BsplineSpace1D(
            np.array(knots, dtype=dtype), degree
        ).tabulate_basis(points)

    np.testing.assert_array_equal(out_basis, expected)
    np.testing.assert_array_equal(out_first, expected_first)


@pytest.mark.parametrize("n_deriv", [0, 1, 2])
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_space_level_derivatives_match_the_oracle_bitwise(
    cpp_backend: None, case: str, dtype: Any, n_deriv: int
) -> None:
    del cpp_backend
    _gate_the_interpreted_oracle(case, dtype, n_deriv=n_deriv)
    from pantr import _pantr_cpp  # noqa: PLC0415  (only this module needs it)

    knots, degree = _CASES[case]
    points = _POINTS_BY_CASE[case].astype(dtype)
    space = _cpp_space(knots, degree, dtype)

    out_deriv = np.zeros((points.size, n_deriv + 1, degree + 1), dtype=dtype)
    out_first = np.zeros(points.size, dtype=np.int64)
    _pantr_cpp.tabulate_bspline_space_basis_derivatives_1d(
        space, n_deriv, points, out_deriv=out_deriv, out_first_basis=out_first
    )

    with use_backend(Backend.PYTHON):
        expected, expected_first = BsplineSpace1D(
            np.array(knots, dtype=dtype), degree
        ).tabulate_basis_derivatives(points, n_deriv)

    np.testing.assert_array_equal(out_deriv, expected)
    np.testing.assert_array_equal(out_first, expected_first)


@pytest.mark.parametrize("n_deriv", [None, 1], ids=["values", "d1"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("case", sorted(_PERIODIC_CASES))
def test_a_periodic_space_matches_the_oracle_bitwise(
    cpp_backend: None, case: str, dtype: Any, n_deriv: int | None
) -> None:
    """The flag the dispatch has to forward, which nothing else here would catch."""
    del cpp_backend
    from pantr import _pantr_cpp  # noqa: PLC0415  (only this module needs it)

    knots, degree = _PERIODIC_CASES[case]
    points = _POINTS_BY_CASE[case].astype(dtype)
    space = _cpp_space(knots, degree, dtype, periodic=True)
    out_first = np.zeros(points.size, dtype=np.int64)

    with use_backend(Backend.PYTHON):
        oracle = BsplineSpace1D(np.array(knots, dtype=dtype), degree, periodic=True)
        if n_deriv is None:
            expected, expected_first = oracle.tabulate_basis(points)
        else:
            expected, expected_first = oracle.tabulate_basis_derivatives(points, n_deriv)

    if n_deriv is None:
        got = np.zeros((points.size, degree + 1), dtype=dtype)
        _pantr_cpp.tabulate_bspline_space_basis_1d(
            space, points, out_basis=got, out_first_basis=out_first
        )
    else:
        got = np.zeros((points.size, n_deriv + 1, degree + 1), dtype=dtype)
        _pantr_cpp.tabulate_bspline_space_basis_derivatives_1d(
            space, n_deriv, points, out_deriv=got, out_first_basis=out_first
        )

    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(out_first, expected_first)


def test_the_periodic_cases_really_are_periodic(cpp_backend: None) -> None:
    """Non-vacuity for the test above: a flag that is never set proves nothing."""
    del cpp_backend
    for name, (knots, degree) in _PERIODIC_CASES.items():
        space = _cpp_space(knots, degree, np.float64, periodic=True)
        assert space.periodic, name
        assert not _cpp_space(knots, degree, np.float64).periodic, name


@pytest.mark.parametrize(
    ("wrong", "message"),
    [
        ("deriv_points", "out_deriv has shape"),
        ("deriv_rows", "out_deriv has shape"),
        ("first", "out_first_basis has"),
    ],
)
def test_a_wrong_derivative_output_shape_is_refused(
    cpp_backend: None, wrong: str, message: str
) -> None:
    """The derivative binding's own checks, which the value test does not reach."""
    del cpp_backend
    from pantr import _pantr_cpp  # noqa: PLC0415  (only this module needs it)

    knots, degree = _CASES["clamped-uniform-p2"]
    points = _POINTS_BY_CASE["clamped-uniform-p2"]
    space = _cpp_space(knots, degree, np.float64)
    n_deriv = 2

    rows = points.size + 1 if wrong == "deriv_points" else points.size
    orders = n_deriv if wrong == "deriv_rows" else n_deriv + 1
    out_deriv = np.zeros((rows, orders, degree + 1))
    out_first = np.zeros(points.size + (1 if wrong == "first" else 0), dtype=np.int64)

    with pytest.raises(ValueError, match=message):
        _pantr_cpp.tabulate_bspline_space_basis_derivatives_1d(
            space, n_deriv, points, out_deriv=out_deriv, out_first_basis=out_first
        )


def test_a_negative_derivative_order_is_refused(cpp_backend: None) -> None:
    """``n_deriv`` is ``unsigned`` in the binding, so nanobind refuses it first.

    A ``TypeError`` rather than a ``ValueError``, and that is the contract the
    raw-knot pair already states: the caster rejects a negative value before the
    function body runs, while the kernels' own parameters stay ``std::int64_t``.
    """
    del cpp_backend
    from pantr import _pantr_cpp  # noqa: PLC0415  (only this module needs it)

    knots, degree = _CASES["clamped-uniform-p2"]
    points = _POINTS_BY_CASE["clamped-uniform-p2"]
    space = _cpp_space(knots, degree, np.float64)
    out_deriv = np.zeros((points.size, 1, degree + 1))
    out_first = np.zeros(points.size, dtype=np.int64)

    with pytest.raises(TypeError):
        _pantr_cpp.tabulate_bspline_space_basis_derivatives_1d(
            space, -1, points, out_deriv=out_deriv, out_first_basis=out_first
        )


def test_a_dtype_mismatch_is_refused_rather_than_silently_converted(
    cpp_backend: None,
) -> None:
    """``.noconvert()`` on every array, and why it matters here.

    A converting cast would fill a discarded temporary and hand the caller back an
    untouched array with no exception anywhere -- and on the input side it would
    change the accumulation width, so the result would disagree with the oracle for a
    reason no caller could see.
    """
    del cpp_backend
    from pantr import _pantr_cpp  # noqa: PLC0415  (only this module needs it)

    knots, degree = _CASES["clamped-uniform-p2"]
    space = _cpp_space(knots, degree, np.float64)
    points_32 = _POINTS_BY_CASE["clamped-uniform-p2"].astype(np.float32)
    out_first = np.zeros(points_32.size, dtype=np.int64)

    with pytest.raises(TypeError):
        _pantr_cpp.tabulate_bspline_space_basis_1d(
            space,
            points_32,
            out_basis=np.zeros((points_32.size, degree + 1)),
            out_first_basis=out_first,
        )


def test_the_bezier_cases_really_take_the_fast_path(cpp_backend: None) -> None:
    """Non-vacuity: without this the two paths could both be the recurrence.

    The whole reason for binding the space-level pair is that it *chooses*, so a
    parametrization in which nothing chooses the Bézier route would compare the
    recurrence against itself and the file would be green while saying nothing about
    the change of variable or the chain-rule scaling.
    """
    del cpp_backend
    taken = {
        name: BsplineSpace1D(np.array(knots), degree).has_Bezier_like_knots()
        for name, (knots, degree) in _CASES.items()
    }

    assert taken["bezier-like-p2"], taken
    assert taken["bezier-like-p3"], taken
    assert not taken["clamped-uniform-p2"], taken
    assert not taken["unclamped-p2"], taken


@pytest.mark.parametrize(
    ("shape_kind", "message"),
    [("basis", "out_basis has shape"), ("first", "out_first_basis has")],
)
def test_a_wrong_output_shape_is_refused(cpp_backend: None, shape_kind: str, message: str) -> None:
    """The checks the space cannot make for the caller.

    The space validated its own knots at construction, so the knot-length check the
    raw-knot binding performs has no counterpart here. The output shapes remain the
    caller's to get wrong, and a binding that filled a too-small buffer would corrupt
    the heap rather than raise -- which the raw-knot binding's own file comment
    records as measured, not hypothetical.
    """
    del cpp_backend
    from pantr import _pantr_cpp  # noqa: PLC0415  (only this module needs it)

    knots, degree = _CASES["clamped-uniform-p2"]
    points = _POINTS_BY_CASE["clamped-uniform-p2"]
    space = _cpp_space(knots, degree, np.float64)

    basis_rows = points.size if shape_kind != "basis" else points.size + 1
    first_size = points.size if shape_kind != "first" else points.size + 1
    out_basis = np.zeros((basis_rows, degree + 1))
    out_first = np.zeros(first_size, dtype=np.int64)

    with pytest.raises(ValueError, match=message):
        _pantr_cpp.tabulate_bspline_space_basis_1d(
            space, points, out_basis=out_basis, out_first_basis=out_first
        )
