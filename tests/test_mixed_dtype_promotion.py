"""A mixed-width call computes and returns at the wider of its two dtypes.

A space stores its knots at one width and may be evaluated at points of another. The
library used to answer that by running the kernels with scratch in the *knots'* dtype
while reading the points at their own, so a ``float32`` space at ``float64`` points
truncated every intermediate to ``float32`` and returned an array that was ``float64``
in name only. The C++ kernels are templated on one scalar type and cannot express that
shape at all, so the backend seam had to fall back to Numba for it -- the only
documented fallback in the port.

The contract now says such a call computes at the **wider** of the two widths and
returns it. The promotion is applied above the backend seam, so it is the same
computation under both, and the fallback is gone.

**The property that pins it is value equality with the fully widened call**, not the
result's dtype. Every one of these assertions would also have passed under the old
behaviour if it only checked the dtype, which is exactly why the five pre-existing
tests that exercise a mixed call did not notice the change.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pantr._numba_compat import wait_for_jit_warmup
from pantr.bspline import BsplineSpace1D

wait_for_jit_warmup()


_GENERAL = ([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0], 2)
"""A clamped quadratic with interior knots: takes the general-knot kernels."""

_BEZIER_LIKE = ([0.0, 0.0, 0.0, 3.0, 3.0, 3.0], 2)
"""No interior knot, so the space takes the Bernstein fast path instead."""

_POINTS = [0.0, 0.3, 1.7, 2.9, 3.0]
"""Endpoints included, since those are where the recurrences special-case."""


def _space_and_points(
    knots: list[float], degree: int, knot_dtype: Any, point_dtype: Any
) -> tuple[BsplineSpace1D, Any]:
    """Build a space and a point array at the requested dtypes.

    Args:
        knots (list[float]): The knot vector.
        degree (int): The polynomial degree.
        knot_dtype (Any): Storage dtype for the space.
        point_dtype (Any): Dtype for the evaluation points.

    Returns:
        tuple[BsplineSpace1D, Any]: The space, and the points.
    """
    return (
        BsplineSpace1D(np.array(knots, dtype=knot_dtype), degree),
        np.array(_POINTS, dtype=point_dtype),
    )


@pytest.mark.parametrize(("knots", "degree"), [_GENERAL, _BEZIER_LIKE], ids=["general", "bezier"])
@pytest.mark.parametrize(
    ("knot_dtype", "point_dtype"),
    [(np.float32, np.float64), (np.float64, np.float32)],
    ids=["f32-knots", "f64-knots"],
)
@pytest.mark.parametrize("n_deriv", [None, 0, 2], ids=["values", "d0", "d2"])
def test_a_mixed_call_equals_the_fully_widened_one(
    knots: list[float], degree: int, knot_dtype: Any, point_dtype: Any, n_deriv: int | None
) -> None:
    promoted = np.promote_types(knot_dtype, point_dtype)
    space, points = _space_and_points(knots, degree, knot_dtype, point_dtype)
    wide_space, _ = _space_and_points(knots, degree, promoted, promoted)
    # Widened *from the narrow array*, never parsed afresh at the wide width: the two
    # calls have to be given the same values, or this compares two different inputs.
    wide_points = points.astype(promoted)

    call = (
        (lambda s, p: s.tabulate_basis(p))
        if n_deriv is None
        else (lambda s, p: s.tabulate_basis_derivatives(p, n_deriv))
    )
    mixed, mixed_first = call(space, points)
    wide, wide_first = call(wide_space, wide_points)

    assert mixed.dtype == promoted
    np.testing.assert_array_equal(mixed, wide)
    np.testing.assert_array_equal(mixed_first, wide_first)


@pytest.mark.parametrize(("knots", "degree"), [_GENERAL, _BEZIER_LIKE], ids=["general", "bezier"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_a_same_width_call_is_untouched(knots: list[float], degree: int, dtype: Any) -> None:
    # The promotion must fire on mixed widths only: a `float32` space at `float32`
    # points still computes and returns `float32`, which is the whole of what the
    # narrow storage is for.
    space, points = _space_and_points(knots, degree, dtype, dtype)
    values, _ = space.tabulate_basis(points)
    derivs, _ = space.tabulate_basis_derivatives(points, 1)

    assert values.dtype == np.dtype(dtype)
    assert derivs.dtype == np.dtype(dtype)


def test_the_mixed_result_differs_from_the_narrow_computation() -> None:
    """Non-vacuity: the contract change moved values, not only the output dtype.

    Without this the rest of the file could pass on a library that still truncated
    every intermediate to the knots' width and merely cast the result on the way out.
    """
    space, points = _space_and_points(*_GENERAL, np.float32, np.float64)
    narrow_space, narrow_points = _space_and_points(*_GENERAL, np.float32, np.float32)

    wide, _ = space.tabulate_basis(points)
    narrow, _ = narrow_space.tabulate_basis(narrow_points)

    assert np.any(wide != narrow.astype(np.float64)), (
        "the mixed call agrees bit for bit with the all-float32 one, so either the "
        "points chosen cannot tell the two widths apart or the promotion is not "
        "reaching the kernels"
    )
