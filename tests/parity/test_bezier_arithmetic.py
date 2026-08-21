r"""Parity of the eight Bézier arithmetic kernels against their Numba oracles.

`cpp/include/pantr/bezier/bezier.hpp` and
`cpp/include/pantr/core/reduction_operator.hpp` name this file as the place their
parity claims are measured.

Every claim here is **bitwise**, and that is a first for this port. `quad` could
not claim it because Golub-Welsch and a companion-matrix eigensolver are different
algorithms; `change_basis` could not because a dense solve sits in the middle. Here
both backends run the same expressions in the same order over `+`, `-`, `*` and
`/`, each of which IEEE 754 pins to one correctly rounded result, so the two agree
to the last bit.

Where exactness was not free
----------------------------

Three things had to be got right, and each one is invisible at float64.

**The accumulation widths are not uniform, and are not this port's to choose.** The
four de Casteljau kernels compute each step in double and round once on the store,
because numba promotes their float64 scalars against a float32 workspace. The
derivative kernel is the exception: it opens with ``dtype = pts.dtype`` and
allocates every workspace in it, so at float32 the whole recursion is float32.
Elevation and the product mix, their coefficient tables being float64
unconditionally. Accumulating narrow where the oracle accumulates wide moved 125 of
630 values in the measurement that found it.

**The evaluation kernel's two branches seed from bases of different width.** Above
the mirror threshold the oracle raises ``u``, which is the point array's own dtype;
below it it raises ``1 - u``, which the literal ``1.0`` has already promoted to
float64. One value in 16224 caught that, at degree 17 and ``u = 0.75``, where
``0.75^17 = 3^17 / 2^34`` needs 27 significand bits.

**The seed is a ``pow``, so its claim is observed rather than derived.** Neither C
nor IEEE 754 requires ``pow`` to be correctly rounded. Measured separately over
1280384 pairs, degrees 1 to 64 across the whole mirrored range: the platform
``powf`` and numba's ``np.power`` agree on every one. `bernstein.hpp` records the
same open question with the same answer.

What this file does NOT cover, and it is a real gap
---------------------------------------------------

Unlike the Bernstein tabulation, these kernels contain ``a * b + c * d`` sites, so
``-ffp-contract`` has something to fuse and the exactness above is a property of
the **build**, not of the code. Measured with ``-march=native``: 125 of 630 float64
de Casteljau values move, and 237 of 970 for the reduction-operator apply. None of
the float32 ones do, in either kernel, because the wide accumulator absorbs the
difference before the narrowing store.

**No bound is derived here for a fusing build.** The tests skip rather than weaken,
and the skip is the honest report: a bound written for a branch no host in this
project can execute would ship untested, which is how a wrong bound gets believed.
Deriving it is a prerequisite for the ISA ladder of design/simd.md, not for this
port, and Rule 7 of design/backend_parity.md is the reason it can be deferred at
all: a bound is a property of the code and whether it is approached is a property
of the host.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.bezier import Bezier
from tests._parity_harness import (
    assert_parity,
    bitwise_parity,
    contraction_may_fuse,
    demand_the_compiled_kernel,
)

DTYPES: Final = (np.float64, np.float32)
"""The two storage formats the Bézier layer accepts."""

DEGREES: Final = (0, 1, 2, 3, 5, 8, 13, 17, 25)
"""Degrees swept by the univariate tests.

0 and 1 are the two branches the evaluation kernel short-circuits; 17 is the degree
at which the mirrored seed's width was caught, and is kept for that reason; 25 is
past anything pantr's own builders reach.
"""

_DE_CASTELJAU_WHY = (
    "the triangle is +, -, * and / only, each pinned by IEEE 754 to one correctly "
    "rounded result, evaluated in the oracle's order and rounded through the "
    "workspace at the oracle's width rather than carried in a register. No fused "
    "multiply-add is available on this build, so the one operation that could "
    "differ cannot occur"
)

_EVALUATE_WHY = (
    "every operation but the seed is +, -, * or /, evaluated in the oracle's order, "
    "with the running term carried in a register exactly as the oracle carries it. "
    "The seed is pow, which neither C nor IEEE 754 requires to be correctly rounded, "
    "so this claim is observed rather than derived: measured over 1280384 pairs, the "
    "platform pow and numba's np.power agree on every argument these degrees form, "
    "at both widths the two branches seed at. No fused multiply-add on this build"
)

_BINOMIAL_WHY = (
    "the coefficient tables are built from an exact-integer binomial recurrence that "
    "is the same recurrence on both sides, and every later operation is +, -, * or / "
    "in the oracle's order. No fused multiply-add on this build"
)

_REDUCTION_WHY = (
    "the operator is assembled once in exact rational arithmetic on the Python side "
    "and crosses as float64, so both backends multiply the same matrix; the apply "
    "accumulates in float64 in the same order on both sides and rounds once on the "
    "write. No fused multiply-add on this build"
)

_NO_BOUND_FOR_A_FUSING_BUILD = (
    "this build can fuse a multiply-add, and no parity bound has been derived for "
    "the Bezier kernels in that regime. Measured with -march=native: 125 of 630 "
    "float64 de Casteljau values move and 237 of 970 for the reduction apply. "
    "Deriving the bound is a prerequisite for the ISA ladder of design/simd.md; see "
    "this file's docstring."
)


def demand_a_non_fusing_build() -> None:
    """Skip when the build can fuse, since no bound covers that case yet.

    Raises:
        Skipped: Always, on a build whose target ISA offers a fused multiply-add.
    """
    if contraction_may_fuse():
        pytest.skip(_NO_BOUND_FOR_A_FUSING_BUILD)


def _mixed_control_points(
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    seed: int = 20260821,
    exponents: tuple[int, int] = (-6, 7),
) -> npt.NDArray[np.float32 | np.float64]:
    """Control points spanning many magnitudes, so the triangle has cancellation to do.

    A net of uniform magnitude is the easy case: every partial sum is the size of
    the answer and nothing cancels. Scaling each entry by a random power of ten
    between 1e-6 and 1e6 is what makes a difference in accumulation width or in
    operation order actually reach the output.

    Args:
        shape (tuple[int, ...]): Shape of the control net, rank last.
        dtype (npt.DTypeLike): Storage format.
        seed (int): Generator seed. Defaults to 20260821.
        exponents (tuple[int, int]): Half-open range of decimal exponents to draw
            from. The default spans twelve orders of magnitude, which is right for
            an operation whose output is the size of its input. An operation that
            raises its input to a power needs a narrower range or it overflows
            float32 before any kernel is at fault. Defaults to (-6, 7).

    Returns:
        npt.NDArray[np.float32 | np.float64]: The control points.
    """
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(shape) * 10.0 ** rng.integers(*exponents, shape)
    return np.ascontiguousarray(values, dtype=dtype)


def _adversarial_parameters(dtype: npt.DTypeLike) -> list[float]:
    """Parameters reaching the branches a uniform sweep does not.

    Both endpoints, either side of the mirror threshold, a value small enough that
    ``1 - (1 - u)`` loses it outright, and both neighbours of one, where the
    unmirrored seed underflows at high degree.

    Args:
        dtype (npt.DTypeLike): Storage format, which sets what "next to one" means.

    Returns:
        list[float]: The parameters, ascending.
    """
    one = np.array(1.0, dtype=dtype)
    half = np.array(0.5, dtype=dtype)
    return [
        0.0,
        1e-20,
        1e-8,
        0.25,
        0.5,
        float(np.nextafter(half, one)),
        0.75,
        1.0 - 1e-8,
        float(np.nextafter(one, np.array(0.0, dtype=dtype))),
        1.0,
    ]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("rank", [1, 3])
def test_evaluate_is_bitwise(
    cpp_backend: None, degree: int, rank: int, dtype: npt.DTypeLike
) -> None:
    """The two backends evaluate a curve identically at every adversarial parameter."""
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_non_fusing_build()

    ctrl = _mixed_control_points((degree + 1, rank), dtype)
    points = np.array(_adversarial_parameters(dtype), dtype=dtype)

    with use_backend(Backend.PYTHON):
        reference = Bezier(ctrl).evaluate(points)
    with use_backend(Backend.CPP):
        actual = Bezier(ctrl).evaluate(points)

    assert_parity(
        actual,
        reference,
        bitwise_parity(why=_EVALUATE_WHY),
        context=f"evaluate degree {degree} rank {rank} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", [0, 1, 3, 8, 17])
@pytest.mark.parametrize("order", [0, 1, 2, 4])
def test_evaluate_derivatives_is_bitwise(
    cpp_backend: None, degree: int, order: int, dtype: npt.DTypeLike
) -> None:
    """The two backends agree on every derivative order, including past the degree.

    ``order`` runs above ``degree`` on purpose: A2.3's index bounds go negative
    there and the recursion takes branches a well-matched order never reaches.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_non_fusing_build()

    ctrl = _mixed_control_points((degree + 1, 2), dtype)
    points = np.array(_adversarial_parameters(dtype), dtype=dtype)

    with use_backend(Backend.PYTHON):
        reference = Bezier(ctrl).evaluate_derivatives(points, order)
    with use_backend(Backend.CPP):
        actual = Bezier(ctrl).evaluate_derivatives(points, order)

    assert_parity(
        actual,
        reference,
        bitwise_parity(why=_DE_CASTELJAU_WHY),
        context=f"evaluate_derivatives degree {degree} order {order} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("increment", [1, 2, 5])
def test_elevate_degree_is_bitwise(
    cpp_backend: None, degree: int, increment: int, dtype: npt.DTypeLike
) -> None:
    """The two backends elevate identically, coefficient table included."""
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_non_fusing_build()

    ctrl = _mixed_control_points((degree + 1, 3), dtype)

    with use_backend(Backend.PYTHON):
        reference = Bezier(ctrl).elevate_degree(increment).control_points
    with use_backend(Backend.CPP):
        actual = Bezier(ctrl).elevate_degree(increment).control_points

    assert_parity(
        actual,
        reference,
        bitwise_parity(why=_BINOMIAL_WHY),
        context=f"elevate_degree {degree} by {increment} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", [2, 3, 5, 8, 13, 20])
@pytest.mark.parametrize("decrement", [1, 2])
def test_reduce_degree_is_bitwise(
    cpp_backend: None, degree: int, decrement: int, dtype: npt.DTypeLike
) -> None:
    """The two backends apply the same reduction operator to the same result."""
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_non_fusing_build()

    ctrl = _mixed_control_points((degree + 1, 3), dtype)

    with use_backend(Backend.PYTHON):
        reference = Bezier(ctrl).reduce_degree(decrement).control_points
    with use_backend(Backend.CPP):
        actual = Bezier(ctrl).reduce_degree(decrement).control_points

    assert_parity(
        actual,
        reference,
        bitwise_parity(why=_REDUCTION_WHY),
        context=f"reduce_degree {degree} by {decrement} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", [3, 4, 7, 12])
def test_minimize_degree_is_bitwise(cpp_backend: None, degree: int, dtype: npt.DTypeLike) -> None:
    """The greedy degree search takes the same decisions on both backends.

    This is the one consumer that needs two kernels within a single call, and so
    the reason `pantr.bezier._bezier_backend` hands out a record for them rather
    than two callables. It is also the only test here whose output is *discrete*:
    the search accepts or rejects each trial by comparing a round-trip error
    against a tolerance, so a one-ulp disagreement in either kernel can change the
    resulting degree rather than the last bit of a coefficient. Bit-exact control
    points therefore prove more here than elsewhere -- they prove the two backends
    took the same path, not only that they landed nearby.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_non_fusing_build()

    # A net that is genuinely reducible, so the search has something to find: a
    # quadratic elevated to `degree`, which is exactly recoverable. The lowest
    # degree swept is 3 because elevating by zero is refused.
    base = _mixed_control_points((3, 2), dtype)
    ctrl = Bezier(base).elevate_degree(degree - 2).control_points

    with use_backend(Backend.PYTHON):
        reference = Bezier(ctrl).minimize_degree()
    with use_backend(Backend.CPP):
        actual = Bezier(ctrl).minimize_degree()

    assert actual.degree == reference.degree, (
        f"minimize_degree from {degree} in {np.dtype(dtype).name}: the backends "
        f"stopped at different degrees, {actual.degree} against {reference.degree}"
    )
    assert_parity(
        actual.control_points,
        reference.control_points,
        bitwise_parity(why=_REDUCTION_WHY),
        context=f"minimize_degree from {degree} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("value", [0.0, 1e-20, 0.25, 0.5, 0.75, 1.0])
def test_slice_is_bitwise(
    cpp_backend: None, degree: int, value: float, dtype: npt.DTypeLike
) -> None:
    """The two backends run the same de Casteljau triangle to the same last bit."""
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_non_fusing_build()

    ctrl = _mixed_control_points((degree + 1, degree + 1, 2), dtype)

    with use_backend(Backend.PYTHON):
        reference = Bezier(ctrl).slice(0, value)
    with use_backend(Backend.CPP):
        actual = Bezier(ctrl).slice(0, value)

    assert isinstance(actual, Bezier) and isinstance(reference, Bezier)
    assert_parity(
        actual.control_points,
        reference.control_points,
        bitwise_parity(why=_DE_CASTELJAU_WHY),
        context=f"slice degree {degree} at {value!r} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("value", [1e-20, 1e-8, 0.25, 0.5, 0.75, 1.0 - 1e-8])
def test_split_is_bitwise(
    cpp_backend: None, degree: int, value: float, dtype: npt.DTypeLike
) -> None:
    """Both halves of a split agree bit for bit.

    The endpoints are absent because Layer 1 refuses them: :meth:`Bezier.split`
    requires a value strictly inside ``(0, 1)``. The kernel itself has no such
    shortcut and would run the full triangle at either end, unlike
    :meth:`~pantr.bezier.Bezier.slice`, so 1e-20 and ``1 - 1e-8`` are as close to
    the ends as this test can legitimately get.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_non_fusing_build()

    ctrl = _mixed_control_points((degree + 1, 3), dtype)

    with use_backend(Backend.PYTHON):
        ref_left, ref_right = Bezier(ctrl).split(0, value)
    with use_backend(Backend.CPP):
        got_left, got_right = Bezier(ctrl).split(0, value)

    for name, actual, reference in (
        ("left", got_left, ref_left),
        ("right", got_right, ref_right),
    ):
        assert_parity(
            actual.control_points,
            reference.control_points,
            bitwise_parity(why=_DE_CASTELJAU_WHY),
            context=f"split {name} degree {degree} at {value!r} {np.dtype(dtype).name}",
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize(
    "bounds",
    [(0.1, 0.9), (0.0, 1e-8), (1.0 - 1e-8, 1.0), (0.25, 0.75), (0.9, 1.0), (0.0, 0.1)],
)
def test_restrict_is_bitwise(
    cpp_backend: None, degree: int, bounds: tuple[float, float], dtype: npt.DTypeLike
) -> None:
    """Both orderings of the two-pass restriction agree bit for bit.

    The bounds list straddles the ``|upper| >= |lower - 1|`` test that chooses
    which pass runs first, so both branches are exercised rather than whichever
    one a symmetric interval happens to pick.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_non_fusing_build()

    ctrl = _mixed_control_points((degree + 1, 3), dtype)

    with use_backend(Backend.PYTHON):
        reference = Bezier(ctrl).restrict(bounds).control_points
    with use_backend(Backend.CPP):
        actual = Bezier(ctrl).restrict(bounds).control_points

    assert_parity(
        actual,
        reference,
        bitwise_parity(why=_DE_CASTELJAU_WHY),
        context=f"restrict degree {degree} to {bounds} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("outer_degree", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("inner_degree", [1, 2, 4])
def test_compose_is_bitwise(
    cpp_backend: None, outer_degree: int, inner_degree: int, dtype: npt.DTypeLike
) -> None:
    """The Bernstein product agrees bit for bit, binomial scaling included.

    Driven through :meth:`Bezier.compose` and **not** through
    :meth:`Bezier.multiply`, which is the route a first draft of this test took and
    which exercises none of the ported code. ``multiply`` goes to
    ``_bernstein_product_coefficients_nd``, a pure-numpy helper that is not
    dispatched at all; the scalar 1D product kernel is reached only from
    ``compose``, and only when the inner map is univariate. The mistake was caught
    by mutation: reassociating the kernel's accumulation left the ``multiply``
    version passing.

    A composition runs the kernel many times over -- once per Bernstein basis power
    of the inner map, then again for each tensor term -- so a single case here
    carries far more products than a single multiply would have.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_non_fusing_build()

    # `inner.rank` must equal `outer.dim`, so a univariate outer map takes a
    # rank-1 inner one. That is exactly the case `use_1d_kernel` selects.
    # A composition of degree `outer_degree` raises the inner map to that power, so
    # the twelve-decade default range overflows float32 well before any kernel is
    # at fault: measured, 1e6 to the eighth is 1e48 against a float32 ceiling near
    # 3.4e38. Three decades still spans enough scale for cancellation to bite.
    spread = (-1, 2)
    outer = Bezier(_mixed_control_points((outer_degree + 1, 2), dtype, 11, spread))
    inner = Bezier(_mixed_control_points((inner_degree + 1, 1), dtype, 22, spread))

    with use_backend(Backend.PYTHON):
        reference = outer.compose(inner).control_points
    with use_backend(Backend.CPP):
        actual = outer.compose(inner).control_points

    assert_parity(
        actual,
        reference,
        bitwise_parity(why=_BINOMIAL_WHY),
        context=(f"compose degree {outer_degree} with {inner_degree} {np.dtype(dtype).name}"),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_strided_out_reaches_the_callers_array(cpp_backend: None, dtype: npt.DTypeLike) -> None:
    """A non-contiguous ``out`` is filled, and filled identically, on both backends.

    The C++ binding refuses a strided array, because ``.noconvert()`` is what stops
    nanobind from filling a temporary and discarding it, and the Python adapter
    absorbs that by buffering and copying back. An adapter that dropped the copy
    would return the right answer and leave the caller's array untouched, with no
    exception anywhere, which is the worst failure shape available.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    demand_a_non_fusing_build()

    ctrl = _mixed_control_points((6, 3), dtype)
    points = np.linspace(0.0, 1.0, 9, dtype=dtype)

    results = {}
    for backend in (Backend.PYTHON, Backend.CPP):
        holder = np.zeros((3, points.size), dtype=dtype)
        view = holder.T
        assert not view.flags["C_CONTIGUOUS"]
        with use_backend(backend):
            Bezier(ctrl).evaluate(points, out=view)
        assert np.any(holder != 0.0), f"{backend.name}: the caller's array was not written"
        results[backend] = holder.copy()

    assert_parity(
        results[Backend.CPP],
        results[Backend.PYTHON],
        bitwise_parity(why=f"{_EVALUATE_WHY}; buffering a strided out adds no arithmetic"),
        context=f"strided out, {np.dtype(dtype).name}",
    )


def test_the_split_binding_refuses_its_outputs_positionally(cpp_backend: None) -> None:
    """``out_left`` and ``out_right`` are keyword-only, so they cannot be exchanged.

    They share a dtype and a shape, so nothing in the type system separates them
    and a positional call would silently return the two halves the wrong way round.
    This asserts the guard exists rather than trusting that it was written.
    """
    del cpp_backend
    from pantr import _pantr_cpp  # noqa: PLC0415

    ctrl = np.ascontiguousarray(np.linspace(0.0, 1.0, 8).reshape(4, 2))
    left = np.empty((4, 2))
    right = np.empty((4, 2))

    with pytest.raises(TypeError):
        _pantr_cpp.split_bezier_1d(ctrl, 0.5, left, right)  # type: ignore[misc]

    _pantr_cpp.split_bezier_1d(ctrl, 0.5, out_left=left, out_right=right)
    assert not np.array_equal(left, right), (
        "a split at the midpoint of a non-symmetric net must give two different halves"
    )
