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

That equality is about the *arithmetic*, on the points a call accepts. It is not
equality of everything: the space keeps its own tolerance, which is derived from its
storage width, so a narrow space's domain-membership gate stays looser than its widened
twin's. Promoting the array adds no resolution to knots that were already rounded.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pantr._numba_compat import wait_for_jit_warmup
from pantr.bspline import BsplineSpace, BsplineSpace1D

wait_for_jit_warmup()


_GENERAL = ([0.0, 0.0, 0.0, 0.7, 1.3, 2.0, 2.0, 2.0], 2, [0.0, 0.3, 1.0, 1.9, 2.0])
"""A clamped quadratic with interior knots: takes the general-knot kernels.

Its **interior** knots are not representable in ``float32``, so the narrow and the
widened space really do differ in the knots the kernel reads, while the domain bounds
are exact at either width and the endpoints can therefore be evaluated.
"""

_BEZIER_LIKE = ([0.1, 0.1, 0.1, 2.2, 2.2, 2.2], 2, [0.15, 0.3, 1.0, 2.1])
"""No interior knot, so the space takes the Bernstein fast path instead.

**The bounds are deliberately not representable in ``float32``**, which is the only way
to exercise this path at all: its knots *are* its bounds. It maps the points with
``(pts - k0) / (k1 - k0)``, and ``k1 - k0`` is scalar against scalar, so read at the
space's storage width it carries the narrow span's rounding into every point, while
``pts - k0`` widens on its own and hides nothing. A first version of this file used
``0.0`` and ``3.0``, exact at either width, and the subtraction then had no rounding to
carry: the assertion passed however the code was written. Measured before the span was
read at the promoted width: 8.4e-8 between the mixed call and the fully widened one,
and 0 after.

The points stay strictly inside, because ``float32(2.2)`` lies outside the widened
space's domain and that space's tolerance is eight orders tighter.
"""


def _space_and_points(
    fixture: tuple[list[float], int, list[float]], knot_dtype: Any, point_dtype: Any
) -> tuple[BsplineSpace1D, Any]:
    """Build a space and a point array at the requested dtypes.

    Args:
        fixture (tuple[list[float], int, list[float]]): Knots, degree and points.
        knot_dtype (Any): Storage dtype for the space.
        point_dtype (Any): Dtype for the evaluation points.

    Returns:
        tuple[BsplineSpace1D, Any]: The space, and the points.
    """
    knots, degree, points = fixture
    return (
        BsplineSpace1D(np.array(knots, dtype=knot_dtype), degree),
        np.array(points, dtype=point_dtype),
    )


@pytest.mark.parametrize("fixture", [_GENERAL, _BEZIER_LIKE], ids=["general", "bezier"])
@pytest.mark.parametrize(
    ("knot_dtype", "point_dtype"),
    [(np.float32, np.float64), (np.float64, np.float32)],
    ids=["f32-knots", "f64-knots"],
)
@pytest.mark.parametrize("n_deriv", [None, 0, 2], ids=["values", "d0", "d2"])
def test_a_mixed_call_equals_the_fully_widened_one(
    fixture: tuple[list[float], int, list[float]],
    knot_dtype: Any,
    point_dtype: Any,
    n_deriv: int | None,
) -> None:
    promoted = np.promote_types(knot_dtype, point_dtype)
    degree = fixture[1]
    space, points = _space_and_points(fixture, knot_dtype, point_dtype)
    # Both widened *from the narrow arrays*, never parsed afresh at the wide width.
    # `float32(0.1)` and `float64(0.1)` are different numbers, so building the wide
    # space from the literals would compare two different problems and the difference
    # would read as the promotion failing.
    wide_space = BsplineSpace1D(np.asarray(space.knots).astype(promoted), degree)
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


@pytest.mark.parametrize("fixture", [_GENERAL, _BEZIER_LIKE], ids=["general", "bezier"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_a_same_width_call_is_untouched(
    fixture: tuple[list[float], int, list[float]], dtype: Any
) -> None:
    # The promotion must fire on mixed widths only: a `float32` space at `float32`
    # points still computes and returns `float32`, which is the whole of what the
    # narrow storage is for.
    space, points = _space_and_points(fixture, dtype, dtype)
    values, _ = space.tabulate_basis(points)
    derivs, _ = space.tabulate_basis_derivatives(points, 1)

    assert values.dtype == np.dtype(dtype)
    assert derivs.dtype == np.dtype(dtype)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_a_multi_dimensional_space_obeys_the_same_rule(dim: int) -> None:
    # The tensor-product path combines the per-direction results with
    # `np.multiply(..., out=out_basis)`, whose default casting downcasts silently, so
    # sizing that buffer from the space's own dtype narrowed a promoted computation
    # on the way out: float32 back, and 2.4e-8 from the widened answer. No mixed-width
    # case reached this module before.
    knots, degree, points = _GENERAL
    narrow = np.array(knots, dtype=np.float32)

    mixed = BsplineSpace([BsplineSpace1D(narrow, degree) for _ in range(dim)])
    widened = BsplineSpace([BsplineSpace1D(narrow.astype(np.float64), degree) for _ in range(dim)])
    pts = np.array([[value] * dim for value in points], dtype=np.float64)

    mixed_basis, mixed_first = mixed.tabulate_basis(pts)
    wide_basis, wide_first = widened.tabulate_basis(pts)

    assert mixed_basis.dtype == np.float64
    np.testing.assert_array_equal(mixed_basis, wide_basis)
    np.testing.assert_array_equal(mixed_first, wide_first)


def test_a_multi_dimensional_same_width_call_is_untouched() -> None:
    # The promotion must not widen a space that was asked for in `float32` throughout.
    knots, degree, points = _GENERAL
    space = BsplineSpace([BsplineSpace1D(np.array(knots, dtype=np.float32), degree)] * 2)
    pts = np.array([[value, value] for value in points], dtype=np.float32)

    basis, _ = space.tabulate_basis(pts)

    assert basis.dtype == np.float32


def test_the_mixed_result_differs_from_the_narrow_computation() -> None:
    """Non-vacuity: the contract change moved values, not only the output dtype.

    Without this the rest of the file could pass on a library that still truncated
    every intermediate to the knots' width and merely cast the result on the way out.
    """
    space, points = _space_and_points(_GENERAL, np.float32, np.float64)
    narrow_space, narrow_points = _space_and_points(_GENERAL, np.float32, np.float32)

    wide, _ = space.tabulate_basis(points)
    narrow, _ = narrow_space.tabulate_basis(narrow_points)

    assert np.any(wide != narrow.astype(np.float64)), (
        "the mixed call agrees bit for bit with the all-float32 one, so either the "
        "points chosen cannot tell the two widths apart or the promotion is not "
        "reaching the kernels"
    )
