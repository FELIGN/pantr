r"""Parity of the Bernstein and Legendre tabulations against their Numba oracles.

`cpp/include/pantr/basis/bernstein.hpp` and its Legendre sibling name this file as
the place their parity claims are measured, and
`tests/parity/test_change_basis.py` leans on the result: its bound treats the
tabulated inputs as contributing nothing, which is only true while the two
backends agree bit for bit.

**This file exists because that citation was made before it did.** Three places
cited a parity test for these kernels that had never been written. The
bit-exactness was true, measured once by hand and then asserted from memory; what
was missing was anything that would notice if it stopped being true. If the
Legendre tabulation ever drifts by an ulp, the change-of-basis bound silently
loses a term and no other test in the suite fails.

What is claimed, and how strong each claim is
---------------------------------------------

**Legendre: bitwise, and derivable.** The recurrence uses only `+`, `-`, `*`, `/`
and `sqrt`, every one of which IEEE 754 pins to a single correctly rounded result.
Two implementations evaluating the same expressions in the same order therefore
agree to the last bit, and the port was written to evaluate them in the same
order -- including reading each previous term back out of the output array rather
than carrying it in a register, which at float32 is a rounding the oracle commits
and a register would not.

**Bernstein: bitwise, and measured rather than derived.** Everything above holds
except the seed, `pow((1-u), n)` or `pow(u, n)`. Neither C nor IEEE 754 requires
`pow` to be correctly rounded, so bit-exactness here rests on numba's `np.power`
and the platform libm agreeing on exactly these arguments. They do, on every
argument tested. The claim is therefore **BITWISE with an observed rather than
proved justification**, and that distinction is the reason the two kernels get
separate reasons below rather than one shared one.

Both claims are gated on the compiled kernel: with `NUMBA_DISABLE_JIT=1` the
oracle runs interpreted and its float32 intermediates widen, which is a property
of how the oracle was run rather than of either implementation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.basis import tabulate_bernstein_1d, tabulate_legendre_1d
from tests._parity_harness import (
    assert_parity,
    bitwise_parity,
    demand_the_compiled_kernel,
)

_BERNSTEIN_WHY = (
    "every operation but the seed is +, -, * or /, each pinned by IEEE 754 to one "
    "correctly rounded result, and the port evaluates them in the oracle's order, "
    "including reading each previous term back out of the output array. The seed is "
    "pow, which neither C nor IEEE 754 requires to be correctly rounded, so this "
    "claim is observed rather than derived: numba's np.power and the platform libm "
    "agree on every argument these degrees form"
)

_LEGENDRE_WHY = (
    "the recurrence uses only +, -, *, / and sqrt, all correctly rounded by IEEE "
    "754, evaluated in the oracle's order, with each previous term read back out of "
    "the output array rather than carried in a register. No pow, so unlike Bernstein "
    "this is derivable rather than merely observed"
)


def _adversarial_points(dtype: npt.DTypeLike) -> npt.NDArray[np.float32 | np.float64]:
    """Points chosen to reach the branches a uniform sweep does not.

    Includes both endpoints, the midpoint the mirror branches on, the largest
    representable value below one (where the unmirrored Bernstein seed underflows
    at high degree), a value small enough that ``1 - (1 - u)`` loses it entirely,
    and a subnormal.

    Args:
        dtype (npt.DTypeLike): The dtype to build them in.

    Returns:
        npt.NDArray: The points, ascending.
    """
    one = np.array(1.0, dtype=dtype)
    zero = np.array(0.0, dtype=dtype)
    rng = np.random.default_rng(20260821)
    special = np.array(
        [
            0.0,
            float(np.nextafter(zero, one)),
            1e-20,
            0.25,
            0.5,
            float(np.nextafter(np.array(0.5, dtype=dtype), one)),
            0.75,
            float(np.nextafter(one, zero)),
            1.0,
        ],
        dtype=dtype,
    )
    return np.concatenate([special, rng.random(24).astype(dtype)])


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("degree", [0, 1, 3, 6, 7, 8, 20, 21, 24, 40])
def test_bernstein_tabulation_is_bitwise(
    cpp_backend: None, degree: int, dtype: npt.DTypeLike
) -> None:
    """The two backends tabulate the Bernstein basis identically.

    The degree list straddles both dispatch thresholds on purpose: the unmirrored
    kernel runs at or below degree 20 in float64 and 6 in float32, so 7, 8 and 21
    are the first degrees on the other side of each, and 24 and 40 exercise the
    mirrored branch well past them.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    points = _adversarial_points(dtype)

    with use_backend(Backend.PYTHON):
        reference = tabulate_bernstein_1d(degree, points)
    with use_backend(Backend.CPP):
        actual = tabulate_bernstein_1d(degree, points)

    assert_parity(
        actual,
        reference,
        bitwise_parity(why=_BERNSTEIN_WHY),
        context=f"tabulate_bernstein_1d degree {degree} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("degree", [0, 1, 2, 5, 12, 20, 32])
def test_legendre_tabulation_is_bitwise(
    cpp_backend: None, degree: int, dtype: npt.DTypeLike
) -> None:
    """The two backends tabulate the orthonormal shifted Legendre basis identically."""
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    points = _adversarial_points(dtype)

    with use_backend(Backend.PYTHON):
        reference = tabulate_legendre_1d(degree, points)
    with use_backend(Backend.CPP):
        actual = tabulate_legendre_1d(degree, points)

    assert_parity(
        actual,
        reference,
        bitwise_parity(why=_LEGENDRE_WHY),
        context=f"tabulate_legendre_1d degree {degree} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_a_strided_out_reaches_the_callers_array(cpp_backend: None, dtype: npt.DTypeLike) -> None:
    """A non-contiguous ``out`` is filled, and filled identically, on both backends.

    The C++ binding refuses a strided array and the Python adapter absorbs that by
    buffering and copying back. An adapter that dropped the copy would return the
    right answer and leave the caller's array untouched, which is the worst failure
    shape available and is why this is checked rather than assumed. The
    change-of-basis Lagrange builder passes exactly such a view, ``out.T``.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    degree = 5
    points = np.linspace(0.0, 1.0, 9, dtype=dtype)

    results = {}
    for backend in (Backend.PYTHON, Backend.CPP):
        holder = np.zeros((degree + 1, points.size), dtype=dtype)
        view = holder.T
        assert not view.flags["C_CONTIGUOUS"]
        with use_backend(backend):
            tabulate_bernstein_1d(degree, points, out=view)
        assert np.any(holder != 0.0), f"{backend.name}: the caller's array was not written"
        results[backend] = holder.copy()

    assert_parity(
        results[Backend.CPP],
        results[Backend.PYTHON],
        bitwise_parity(why=f"{_BERNSTEIN_WHY}; buffering a strided out adds no arithmetic"),
        context=f"strided out, {np.dtype(dtype).name}",
    )
