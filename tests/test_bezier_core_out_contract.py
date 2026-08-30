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

The two n-dimensional evaluation entry points are here for the same reason and are
worse placed to get away with it: they accumulate into one destination element per
output value, over ``sum_d (degree_d + 1)`` terms, and a forgotten zeroing there adds
whatever the buffer held to a result that is otherwise correct. Layer 2 allocates that
buffer with :func:`numpy.empty`, so in production it is never zeroed for them.
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


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("degrees", [(2, 3), (1, 2, 2)])
@pytest.mark.parametrize("rank", [1, 3])
def test_the_nd_entry_points_write_every_element_on_both_backends(
    degrees: tuple[int, ...], rank: int, dtype: type[np.floating]
) -> None:
    """Neither n-d schedule leaves anything of a reused destination behind.

    The kernels are reached through their accessors rather than through
    :meth:`~pantr.bezier.Bezier.evaluate`, and that is the whole point: Layer 2
    allocates the buffer itself and then copies into the caller's ``out``, so a
    poisoned ``out`` at the public surface is overwritten by the copy and would pass
    whatever the kernel did. Poisoning the buffer the kernel is handed is what
    reaches the obligation.

    Both entry points are checked, and separately: they are two different
    contraction schedules over two different buffers, so a zeroing forgotten in one
    says nothing about the other.

    Neither backend can fail the poison half today -- both overwrite the destination
    unconditionally -- so that half is a regression guard rather than a live check,
    and it is written down as one. The **strided** half below is live: the C++
    binding refuses a non-contiguous destination outright, and only
    :func:`~pantr.bezier._bezier_backend._fill` makes the two backends accept the
    same arrays. Bypassing it would raise :class:`TypeError` here rather than
    silently returning a wrong number.
    """
    from pantr._backend import available_backends, use_backend  # noqa: PLC0415
    from pantr.bezier import Bezier  # noqa: PLC0415
    from pantr.bezier._bezier_backend import (  # noqa: PLC0415
        evaluate_nd_kernel,
        evaluate_nd_lattice_kernel,
    )

    rng = np.random.default_rng(20260830)
    shape = (*(degree + 1 for degree in degrees), rank)
    ctrl: npt.NDArray[np.floating] = rng.standard_normal(shape).astype(dtype)
    dim = len(degrees)
    points = np.ascontiguousarray(rng.random((5, dim)), dtype=dtype)
    columns = [np.asarray(rng.random(3), dtype=dtype) for _ in range(dim)]

    for backend in available_backends():
        with use_backend(backend):
            bezier = Bezier(ctrl)

            clean = np.zeros((5, rank), dtype=dtype)
            evaluate_nd_kernel()(bezier, points, clean)
            poisoned = np.full((5, rank), _POISON, dtype=dtype)
            evaluate_nd_kernel()(bezier, points, poisoned)
            assert_array_equal(poisoned, clean)

            lattice_shape = ((3,) * dim) + (rank,)
            clean_lattice = np.zeros(lattice_shape, dtype=dtype)
            evaluate_nd_lattice_kernel()(bezier, columns, clean_lattice)
            poisoned_lattice = np.full(lattice_shape, _POISON, dtype=dtype)
            evaluate_nd_lattice_kernel()(bezier, columns, poisoned_lattice)
            assert_array_equal(poisoned_lattice, clean_lattice)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_the_nd_entry_points_accept_a_strided_destination_on_both_backends(
    dtype: type[np.floating],
) -> None:
    """``PANTR_BACKEND`` must not change which ``out`` the n-d schedules accept.

    The numba kernels fill a strided ``out`` in place; the C++ bindings require
    C-contiguous memory and refuse anything else, because ``.noconvert()`` is what
    stops nanobind from filling a temporary and discarding it. A strided destination
    therefore only works because Layer 2 absorbs it, and this is what says so.
    """
    from pantr._backend import available_backends, use_backend  # noqa: PLC0415
    from pantr.bezier import Bezier  # noqa: PLC0415
    from pantr.bezier._bezier_backend import (  # noqa: PLC0415
        evaluate_nd_kernel,
        evaluate_nd_lattice_kernel,
    )

    rng = np.random.default_rng(20260831)
    ctrl: npt.NDArray[np.floating] = rng.standard_normal((3, 4, 2)).astype(dtype)
    points = np.ascontiguousarray(rng.random((5, 2)), dtype=dtype)
    columns = [np.asarray(rng.random(3), dtype=dtype) for _ in range(2)]

    for backend in available_backends():
        with use_backend(backend):
            bezier = Bezier(ctrl)

            contiguous = np.empty((5, 2), dtype=dtype)
            evaluate_nd_kernel()(bezier, points, contiguous)
            strided = np.empty((5, 4), dtype=dtype)[:, ::2]
            assert not strided.flags["C_CONTIGUOUS"]
            evaluate_nd_kernel()(bezier, points, strided)
            assert_array_equal(strided, contiguous)

            contiguous_lattice = np.empty((3, 3, 2), dtype=dtype)
            evaluate_nd_lattice_kernel()(bezier, columns, contiguous_lattice)
            strided_lattice = np.empty((3, 3, 4), dtype=dtype)[..., ::2]
            assert not strided_lattice.flags["C_CONTIGUOUS"]
            evaluate_nd_lattice_kernel()(bezier, columns, strided_lattice)
            assert_array_equal(strided_lattice, contiguous_lattice)
