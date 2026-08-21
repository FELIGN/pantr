"""The ``out`` contract of the two Bézier kernels that used to allocate their result.

:func:`~pantr.bezier._bezier_core._degree_elevate_bezier_1d_core` and
:func:`~pantr.bezier._bezier_core._scalar_bernstein_product_1d_core` both accumulate
into their destination, so they must zero it first. While they allocated with
:func:`numpy.zeros` that was free; now that the caller owns the buffer it is a promise,
and a kernel that forgets it reads as correct against a freshly allocated array and
wrong against a reused one.

That is the failure these tests exist to catch, and it is the one a C++ port is most
likely to reintroduce, since the natural translation of ``np.zeros`` is a destination
the caller already zeroed.
"""

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_array_equal

from pantr.bezier._bezier_core import (
    _degree_elevate_bezier_1d_core,
    _scalar_bernstein_product_1d_core,
)

# A value no correct result can contain, so a surviving one is unmistakable.
_POISON = -12345.0


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(("degree", "increment", "rank"), [(0, 1, 1), (2, 1, 3), (5, 4, 2)])
def test_degree_elevate_zeros_its_destination(
    degree: int, increment: int, rank: int, dtype: type[np.floating]
) -> None:
    """Elevation into a poisoned buffer matches elevation into a zeroed one."""
    rng = np.random.default_rng(20260821)
    ctrl: npt.NDArray[np.floating] = rng.random((degree + 1, rank)).astype(dtype)
    shape = (degree + increment + 1, rank)

    clean = np.zeros(shape, dtype=dtype)
    _degree_elevate_bezier_1d_core(degree, ctrl, increment, clean)

    poisoned = np.full(shape, _POISON, dtype=dtype)
    _degree_elevate_bezier_1d_core(degree, ctrl, increment, poisoned)

    assert_array_equal(poisoned, clean)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(("p", "q"), [(0, 0), (1, 1), (3, 5)])
def test_bernstein_product_zeros_its_destination(p: int, q: int, dtype: type[np.floating]) -> None:
    """The Bernstein product into a poisoned buffer matches the product into a zeroed one."""
    rng = np.random.default_rng(20260821)
    a: npt.NDArray[np.floating] = rng.random(p + 1).astype(dtype)
    b: npt.NDArray[np.floating] = rng.random(q + 1).astype(dtype)

    clean = np.zeros(p + q + 1, dtype=dtype)
    _scalar_bernstein_product_1d_core(a, b, clean)

    poisoned = np.full(p + q + 1, _POISON, dtype=dtype)
    _scalar_bernstein_product_1d_core(a, b, poisoned)

    assert_array_equal(poisoned, clean)


def test_bernstein_product_of_two_linears_is_exact() -> None:
    """``(1-t)*(1-t)`` in Bernstein form has control points ``(1, 0, 0)`` exactly.

    A known-answer case, so it fails on a wrong binomial scaling rather than only on a
    forgotten zeroing. In Bernstein form ``1 - t`` is ``(1, 0)``, and the square of a
    polynomial whose only nonzero coefficient sits at index 0 keeps that shape.
    """
    linear = np.array([1.0, 0.0])
    out = np.empty(3)
    _scalar_bernstein_product_1d_core(linear, linear, out)
    assert_array_equal(out, np.array([1.0, 0.0, 0.0]))


@pytest.mark.parametrize(
    "method",
    ["evaluate", "evaluate_derivatives"],
)
def test_the_public_surface_refuses_a_mismatched_out_dtype_on_both_backends(
    method: str,
) -> None:
    """``PANTR_BACKEND`` must not change which ``out`` the library accepts.

    The two kernels above lost a structural guarantee when they stopped allocating
    their own result: while they called ``np.zeros(..., dtype=a.dtype)`` a
    mismatched accumulation width was unrepresentable, and now it is the caller's
    to get right. Numba accepts a wider ``out`` and accumulates at that width; the
    C++ binding refuses it with :class:`TypeError`. That asymmetry is confined to a
    direct Layer 3 call, where nothing validates anything by design.

    What must not differ is the **library's** surface, and this pins it: Layer 2
    refuses the mismatch with the same exception under either backend. If a future
    refactor lets one backend through, this fails rather than the difference being
    discovered as a wrong number at float32.
    """
    from pantr._backend import Backend, available_backends, use_backend  # noqa: PLC0415
    from pantr.bezier import Bezier  # noqa: PLC0415

    ctrl = np.ascontiguousarray(np.linspace(0.0, 1.0, 8).reshape(4, 2), dtype=np.float32)
    points = np.array([0.25, 0.5], dtype=np.float32)

    messages = {}
    for backend in available_backends():
        wide = np.empty((2, 2) if method == "evaluate" else (2, 2, 2), dtype=np.float64)
        args = (points,) if method == "evaluate" else (points, 1)
        with use_backend(backend), pytest.raises(ValueError) as caught:
            getattr(Bezier(ctrl), method)(*args, out=wide)
        messages[backend] = str(caught.value)

    if Backend.CPP in messages:
        assert messages[Backend.PYTHON] == messages[Backend.CPP], (
            "the two backends refuse a mismatched out with different messages, so the "
            "library's surface depends on PANTR_BACKEND"
        )
