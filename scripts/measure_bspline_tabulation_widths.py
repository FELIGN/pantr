#!/usr/bin/env python
"""Measure the accumulation width of every expression in the 1D B-spline tabulation kernels.

``design/backend_parity.md`` Rule 9: an oracle's accumulation width is a per-kernel
fact and **cannot be read off the source**. ``float()`` does not widen a ``float32``
in numba's ``nopython`` mode, what widens is type unification across assignments, and
the destination's dtype says nothing about the width the operation ran at. So the C++
transliteration in ``cpp/include/pantr/bspline/tabulate.hpp`` is written against this
script rather than against a reading of
``src/pantr/bspline/_bspline_basis_core.py``.

Three things are measured, and the second is the one that matters:

1. **numba's own inferred types**, expression by expression, read off
   ``nopython_signatures`` rather than off a returned value. Reading a returned value
   measures the boxing to a Python ``float`` and reports ``float64`` for everything.

2. **Behaviourally, two rival transliterations per kernel against the kernel itself**,
   at ``float32`` where the widths separate. Each site is reported with how often the
   two models *disagree with each other*, so a match cannot come from a check that
   could not fail.

3. **Where the falling-factorial accumulator wraps.** ``_basis_derivs_point`` and
   ``_bernstein_derivs_point`` both accumulate ``degree!/(degree-k)!`` in an integer
   that wraps at int64 when compiled. Past the wrap the *scaling* of the k-th
   derivative row is a different number, in both backends, and it is not a rounding
   difference. The wrap is reproduced by the port and bounds the degree at which an
   independent accuracy claim can be made.

Run it from the repository root inside the ``pantr`` environment::

    conda run -n pantr python scripts/measure_bspline_tabulation_widths.py
"""

from __future__ import annotations

import sys
from typing import Any, TypeAlias, cast

import numba as nb
import numpy as np
import numpy.typing as npt
from numba import float32, int64

from pantr._numba_compat import nb_jit
from pantr.basis._basis_core import _tabulate_Bernstein_basis_deriv_1D_serial_core
from pantr.bspline._bspline_basis_core import (
    _compute_basis_deriv_nurbs_book_impl,
    _compute_basis_deriv_nurbs_book_serial_impl,
    _compute_basis_nurbs_book_impl,
    _compute_basis_nurbs_book_serial_impl,
)

_Storage: TypeAlias = type[np.float32] | type[np.float64]
"""A storage format, as the scalar type numpy names it."""

_Scalar: TypeAlias = np.float32 | np.float64
"""One value in a storage format."""

_Array: TypeAlias = npt.NDArray[np.float32 | np.float64]
"""An array in either storage format."""

_F32_1D = nb.types.Array(float32, 1, "C")
"""The numba type of a 1D C-contiguous float32 array."""

_RNG_SEED = 20260909
"""Fixed so the sweep is the same sweep on every re-run."""

_DEGREES = (0, 1, 2, 3, 5, 12, 20, 21, 25)
"""Degrees swept.

The low ones cover the ordinary cases and the empty-span guard. The high ones are
there because the two factorial-scaling models are *provably identical* while
``degree!/(degree-k)!`` is exactly representable in the storage format: the exact
product of two ``float32`` values has at most 48 significand bits, so it is exact in
``float64`` and rounding it once to ``float32`` is the same single rounding a
``float32`` multiply commits. The models can only separate once ``fac`` needs more
than 24 bits, which first happens around degree 12, and 21 and 25 are where the int64
accumulator itself wraps.
"""


# ---------------------------------------------------------------------------
# 1. numba's inferred types, per expression
# ---------------------------------------------------------------------------


def _return_type(fn: Any, *argtypes: Any) -> str:  # noqa: ANN401 -- see Args
    """Compile ``fn`` for ``argtypes`` and report the return type numba inferred.

    Args:
        fn (Any): A numba dispatcher. ``numba.core.registry.CPUDispatcher`` is not
            statically typed and is private, so there is no narrower annotation.
        argtypes (Any): numba type objects to compile for, likewise untyped.

    Returns:
        str: The inferred return type, as numba spells it.
    """
    fn.compile(tuple(argtypes))
    return str(fn.nopython_signatures[-1].return_type)


@nb_jit(nopython=True)
def _expr_sub(a: _Scalar, b: _Scalar) -> _Scalar:
    """``left[j] = pt - knots[...]`` and ``right[j] = knots[...] - pt``."""
    return a - b


@nb_jit(nopython=True)
def _expr_div(a: _Scalar, b: _Scalar) -> _Scalar:
    """``N[r] / denom``."""
    return a / b


@nb_jit(nopython=True)
def _expr_mul_int(a: _Scalar, k: int) -> _Scalar:
    """``out_pt[k, j] *= fac`` with ``fac`` an integer."""
    return a * k


@nb_jit(nopython=True)
def _expr_temp(n: npt.NDArray[np.float32], denom: _Scalar) -> _Scalar:
    """``temp = zero if denom == zero else N[r] / denom``, with ``zero = dtype.type(0.0)``."""
    dtype = n.dtype
    zero = dtype.type(0.0)
    return zero if denom == zero else n[0] / denom


@nb_jit(nopython=True)
def _expr_saved(a: _Scalar, b: _Scalar) -> _Scalar:
    """``saved = zero`` then ``saved = left[j - r] * temp``: unification across assignments."""
    saved: _Scalar = np.float32(0.0)
    saved = a * b
    return saved


def report_inferred_types() -> None:
    """Print the width numba gives each expression the two kernels are built from."""
    print("== 1. numba's inferred types, at float32 storage ==")
    print("   read off nopython_signatures; a returned value would report the boxing instead")
    rows = [
        ("pt - knots[i]                  ", _return_type(_expr_sub, float32, float32)),
        ("N[r] / denom                   ", _return_type(_expr_div, float32, float32)),
        ("temp = zero if .. else N[r]/d  ", _return_type(_expr_temp, _F32_1D, float32)),
        ("saved = 0.0; saved = a * temp  ", _return_type(_expr_saved, float32, float32)),
        ("out_pt[k, j] * fac  (fac int64)", _return_type(_expr_mul_int, float32, int64)),
    ]
    for name, width in rows:
        print(f"   {name} -> {width}")
    print("   so: every Cox-de Boor intermediate stays float32, and the ONE site that")
    print("   widens is the factorial scaling, which multiplies in float64 and stores narrow.")
    print()


# ---------------------------------------------------------------------------
# 2. Rival transliterations, measured against the kernel
# ---------------------------------------------------------------------------


def _span_and_first_basis(
    knots: _Array, degree: int, num_basis: int, pt: _Scalar
) -> tuple[int, int]:
    """Reproduce ``_find_span_and_first_basis_point`` for a non-periodic space.

    Args:
        knots (_Array): The knot vector.
        degree (int): The degree.
        num_basis (int): The number of basis functions.
        pt (_Scalar): The evaluation point.

    Returns:
        tuple[int, int]: ``(knot_id, first_basis)``.
    """
    knot_id = int(np.searchsorted(knots, pt, side="right")) - 1
    knot_id = min(knot_id, knots.size - degree - 2)
    knot_id = max(knot_id, degree)
    return knot_id, min(knot_id - degree, num_basis - degree - 1)


def _model_basis_funcs(  # noqa: PLR0913
    knots: _Array,
    degree: int,
    knot_id: int,
    pt: _Scalar,
    arithmetic: _Storage,
    storage: _Storage,
) -> _Array:
    """Transliterate A2.2 with every intermediate in ``arithmetic``, storing in ``storage``.

    Args:
        knots (_Array): The knot vector.
        degree (int): The degree.
        knot_id (int): The clamped knot span of ``pt``.
        pt (_Scalar): The evaluation point.
        arithmetic (_Storage): Width every intermediate is held at.
        storage (_Storage): Width the result is stored at.

    Returns:
        _Array: The ``degree + 1`` basis values, in ``storage``.
    """
    order = degree + 1
    dt = arithmetic
    values = np.zeros(order, dtype=dt)
    left = np.zeros(order, dtype=dt)
    right = np.zeros(order, dtype=dt)
    values[0] = dt(1.0)

    for j in range(1, order):
        left[j] = dt(pt) - dt(knots[knot_id + 1 - j])
        right[j] = dt(knots[knot_id + j]) - dt(pt)
        saved = dt(0.0)
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            temp = dt(0.0) if denom == dt(0.0) else values[r] / denom
            values[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        values[j] = saved

    return cast("_Array", values.astype(storage))


def _model_basis_derivs(  # noqa: PLR0913
    knots: _Array,
    degree: int,
    n_deriv: int,
    knot_id: int,
    pt: _Scalar,
    arithmetic: _Storage,
    storage: _Storage,
    *,
    widen_the_factorial: bool,
) -> _Array:
    """Transliterate A2.3, with the factorial scaling either widened or not.

    Args:
        knots (_Array): The knot vector.
        degree (int): The degree.
        n_deriv (int): Highest derivative order.
        knot_id (int): The clamped knot span of ``pt``.
        pt (_Scalar): The evaluation point.
        arithmetic (_Storage): Width every A2.3 intermediate is held at.
        storage (_Storage): Width the result is stored at.
        widen_the_factorial (bool): Whether the step-3 scaling multiplies in float64
            and rounds on the store, as ``float32 * int64 -> float64`` implies, or
            narrows the factorial first and multiplies in ``storage``.

    Returns:
        _Array: Shape ``(n_deriv + 1, degree + 1)``, in ``storage``.
    """
    order = degree + 1
    dt = arithmetic
    ndu = np.zeros((order, order), dtype=dt)
    left = np.zeros(order, dtype=dt)
    right = np.zeros(order, dtype=dt)
    a = np.zeros((2, n_deriv + 1), dtype=dt)
    out = np.zeros((n_deriv + 1, order), dtype=dt)

    ndu[0, 0] = dt(1.0)
    for j in range(1, order):
        left[j] = dt(pt) - dt(knots[knot_id + 1 - j])
        right[j] = dt(knots[knot_id + j]) - dt(pt)
        saved = dt(0.0)
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            denom = ndu[j, r]
            temp = dt(0.0) if denom == dt(0.0) else ndu[r, j - 1] / denom
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved

    for j in range(order):
        out[0, j] = ndu[j, degree]

    for r in range(order):
        s1, s2 = 0, 1
        a[0, 0] = dt(1.0)
        for k in range(1, n_deriv + 1):
            d = dt(0.0)
            rk = r - k
            pk = degree - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if (r - 1) <= pk else degree - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d = d + a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d = d + a[s2, k] * ndu[r, pk]
            out[k, r] = d
            s1, s2 = s2, s1

    result = out.astype(storage)
    fac = falling_factorials(degree, n_deriv)
    for k in range(1, n_deriv + 1):
        for j in range(order):
            if widen_the_factorial:
                result[k, j] = storage(np.float64(result[k, j]) * np.float64(fac[k]))
            else:
                result[k, j] = storage(result[k, j] * storage(fac[k]))
    return cast("_Array", result)


def falling_factorials(degree: int, n_deriv: int) -> list[int]:
    """The step-3 scaling factors, accumulated exactly as the kernel accumulates them.

    ``fac`` starts at ``degree`` and is multiplied by ``degree - k``, so entry ``k`` is
    ``degree!/(degree-k)!`` while that fits an int64 and its two's-complement wrap
    afterwards. Entry 0 is unused and is 0.

    Args:
        degree (int): The degree.
        n_deriv (int): Highest derivative order.

    Returns:
        list[int]: ``n_deriv + 1`` factors, wrapped to signed 64-bit.
    """
    mask = (1 << 64) - 1
    factors = [0]
    fac = degree
    for k in range(1, n_deriv + 1):
        factors.append(fac)
        fac = (fac * (degree - k)) & mask
        if fac >= 1 << 63:
            fac -= 1 << 64
    return factors


def _knot_vectors(rng: np.random.Generator, dtype: _Storage) -> list[_Array]:
    """Knot vectors chosen to be adversarial about representability, not magnitude.

    Rule 9's sharpest case was found by a case list built around exactly-representable
    ratios rather than round numbers, so the interior knots here are dyadic fractions,
    near-duplicates a few ulp apart, and an exactly repeated knot (an empty span, which
    is the only input the ``denom == 0`` guard exists for).

    Args:
        rng (np.random.Generator): Source of the random interior knots.
        dtype (_Storage): The storage dtype.

    Returns:
        list[_Array]: One knot vector per case, unclamped and clamped.
    """
    vectors = []
    for interior in (
        [],
        [0.5],
        [0.25, 0.5, 0.75],
        [1.0 / 3.0, 2.0 / 3.0],
        [0.5, 0.5],  # an empty span: the denom == 0 guard
        [0.5, np.nextafter(np.float32(0.5), np.float32(1.0)).item()],
        sorted(rng.random(5).tolist()),
    ):
        for degree in _DEGREES:
            head = [0.0] * (degree + 1)
            tail = [1.0] * (degree + 1)
            vectors.append(cast("_Array", np.array(head + list(interior) + tail, dtype=dtype)))
    return vectors


def _points(rng: np.random.Generator, dtype: _Storage) -> _Array:
    """Evaluation points: both endpoints, every knot, dyadic values, and random ones."""
    fixed = [0.0, 1.0, 0.5, 0.25, 0.75, 1.0 / 3.0]
    fixed += [np.nextafter(np.float32(0.5), np.float32(0.0)).item()]
    fixed += [np.nextafter(np.float32(1.0), np.float32(0.0)).item()]
    return cast("_Array", np.array(fixed + rng.random(18).tolist(), dtype=dtype))


def measure_a2_2_width(dtype: _Storage = np.float32) -> int:
    """Compare a narrow and a wide A2.2 transliteration against the kernel.

    Args:
        dtype (_Storage): The storage dtype to measure at. Defaults to ``np.float32``.

    Returns:
        int: 0 when exactly one model reproduces the kernel on every value.
    """
    rng = np.random.default_rng(_RNG_SEED)
    points = _points(rng, dtype)
    narrow_hits = wide_hits = models_differ = total = 0

    for knots in _knot_vectors(rng, dtype):
        degree = int(np.count_nonzero(knots == knots[0]) - 1)
        num_basis = knots.size - degree - 1
        if num_basis < degree + 1:
            continue
        actual = np.empty((points.size, degree + 1), dtype=dtype)
        first = np.empty(points.size, dtype=np.int_)
        _compute_basis_nurbs_book_serial_impl(knots, degree, False, 1e-10, points, actual, first)
        for j, pt in enumerate(points):
            knot_id, _ = _span_and_first_basis(knots, degree, num_basis, pt)
            narrow = _model_basis_funcs(knots, degree, knot_id, pt, dtype, dtype)
            wide = _model_basis_funcs(knots, degree, knot_id, pt, np.float64, dtype)
            total += actual.shape[1]
            narrow_hits += int(np.count_nonzero(narrow == actual[j]))
            wide_hits += int(np.count_nonzero(wide == actual[j]))
            models_differ += int(np.count_nonzero(narrow != wide))

    name = np.dtype(dtype).name
    print(f"== 2a. A2.2 (_basis_funcs_point) at {name}: {total} values ==")
    print(f"   all-{name} intermediates   reproduce the kernel: {narrow_hits}/{total}")
    print(f"   float64 intermediates      reproduce the kernel: {wide_hits}/{total}")
    print(f"   the two models disagree on {models_differ}/{total} values")
    print()
    return 0 if narrow_hits == total and (dtype is np.float64 or models_differ > 0) else 1


def measure_a2_3_width(dtype: _Storage = np.float32) -> int:
    """Compare narrow/wide A2.3 and both factorial-scaling models against the kernel.

    Args:
        dtype (_Storage): The storage dtype to measure at. Defaults to ``np.float32``.

    Returns:
        int: 0 when exactly one combination reproduces the kernel on every value.
    """
    rng = np.random.default_rng(_RNG_SEED)
    points = _points(rng, dtype)
    hits = {("narrow", "wide-fac"): 0, ("narrow", "narrow-fac"): 0, ("wide", "wide-fac"): 0}
    fac_models_differ = total = 0

    for knots in _knot_vectors(rng, dtype):
        degree = int(np.count_nonzero(knots == knots[0]) - 1)
        num_basis = knots.size - degree - 1
        if num_basis < degree + 1:
            continue
        n_deriv = degree + 1
        actual = np.empty((points.size, n_deriv + 1, degree + 1), dtype=dtype)
        first = np.empty(points.size, dtype=np.int_)
        _compute_basis_deriv_nurbs_book_serial_impl(
            knots, degree, False, 1e-10, n_deriv, points, actual, first
        )
        for j, pt in enumerate(points):
            knot_id, _ = _span_and_first_basis(knots, degree, num_basis, pt)
            models = {
                ("narrow", "wide-fac"): _model_basis_derivs(
                    knots,
                    degree,
                    n_deriv,
                    knot_id,
                    pt,
                    dtype,
                    dtype,
                    widen_the_factorial=True,
                ),
                ("narrow", "narrow-fac"): _model_basis_derivs(
                    knots,
                    degree,
                    n_deriv,
                    knot_id,
                    pt,
                    dtype,
                    dtype,
                    widen_the_factorial=False,
                ),
                ("wide", "wide-fac"): _model_basis_derivs(
                    knots,
                    degree,
                    n_deriv,
                    knot_id,
                    pt,
                    np.float64,
                    dtype,
                    widen_the_factorial=True,
                ),
            }
            total += actual[j].size
            for key, model in models.items():
                hits[key] += int(np.count_nonzero(model == actual[j]))
            fac_models_differ += int(
                np.count_nonzero(models[("narrow", "wide-fac")] != models[("narrow", "narrow-fac")])
            )

    name = np.dtype(dtype).name
    print(f"== 2b. A2.3 (_basis_derivs_point) at {name}: {total} values ==")
    for (arith, fac), count in hits.items():
        print(f"   {arith:6s} intermediates + {fac:11s} scaling reproduces: {count}/{total}")
    print(f"   the two factorial-scaling models disagree on {fac_models_differ}/{total} values")
    print()
    ok = hits[("narrow", "wide-fac")] == total
    return 0 if ok and (dtype is np.float64 or fac_models_differ > 0) else 1


def _model_bernstein_derivs(  # noqa: PLR0913
    degree: int,
    pt: _Scalar,
    n_deriv: int,
    arithmetic: _Storage,
    storage: _Storage,
    *,
    widen_the_factorial: bool,
) -> _Array:
    """Transliterate ``_bernstein_derivs_point``: A2.3 with every knot difference one.

    Args:
        degree (int): The degree.
        pt (_Scalar): The evaluation point, on ``[0, 1]``.
        n_deriv (int): Highest derivative order.
        arithmetic (_Storage): Width every intermediate is held at.
        storage (_Storage): Width the result is stored at.
        widen_the_factorial (bool): Whether the step-3 scaling widens, as in
            :func:`_model_basis_derivs`.

    Returns:
        _Array: Shape ``(n_deriv + 1, degree + 1)``, in ``storage``.
    """
    order = degree + 1
    dt = arithmetic
    ndu = np.zeros((order, order), dtype=dt)
    a = np.zeros((2, n_deriv + 1), dtype=dt)
    out = np.zeros((n_deriv + 1, order), dtype=dt)
    s = dt(pt)

    ndu[0, 0] = dt(1.0)
    for j in range(1, order):
        saved = dt(0.0)
        for r in range(j):
            ndu[j, r] = dt(1.0)
            temp = ndu[r, j - 1]
            ndu[r, j] = saved + (dt(1.0) - s) * temp
            saved = s * temp
        ndu[j, j] = saved

    for j in range(order):
        out[0, j] = ndu[j, degree]

    for r in range(order):
        s1, s2 = 0, 1
        a[0, 0] = dt(1.0)
        for k in range(1, n_deriv + 1):
            d = dt(0.0)
            rk = r - k
            pk = degree - k
            if r >= k:
                a[s2, 0] = a[s1, 0]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if (r - 1) <= pk else degree - r
            for j in range(j1, j2 + 1):
                a[s2, j] = a[s1, j] - a[s1, j - 1]
                d = d + a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1]
                d = d + a[s2, k] * ndu[r, pk]
            out[k, r] = d
            s1, s2 = s2, s1

    result = out.astype(storage)
    fac = falling_factorials(degree, n_deriv)
    for k in range(1, n_deriv + 1):
        for j in range(order):
            if widen_the_factorial:
                result[k, j] = storage(np.float64(result[k, j]) * np.float64(fac[k]))
            else:
                result[k, j] = storage(result[k, j] * storage(fac[k]))
    return cast("_Array", result)


def measure_bernstein_deriv_width(dtype: _Storage = np.float32) -> int:
    """Compare rival transliterations of ``_bernstein_derivs_point`` against the kernel.

    Measured separately from A2.3 rather than inherited from it. Rule 9: a width cannot
    be inferred per module or from the kernel next door, and these two kernels differ in
    exactly the place a width surprise hides -- this one forms ``one - s`` against a
    ``dtype.type(1.0)`` while A2.3 reads both operands out of arrays.

    Args:
        dtype (_Storage): The storage dtype. Defaults to ``np.float32``.

    Returns:
        int: 0 when exactly one combination reproduces the kernel on every value.
    """
    rng = np.random.default_rng(_RNG_SEED)
    points = _points(rng, dtype)
    hits = {("narrow", "wide-fac"): 0, ("narrow", "narrow-fac"): 0, ("wide", "wide-fac"): 0}
    fac_models_differ = total = 0

    for degree in _DEGREES:
        n_deriv = degree + 1
        actual = np.empty((points.size, n_deriv + 1, degree + 1), dtype=dtype)
        _tabulate_Bernstein_basis_deriv_1D_serial_core(np.int32(degree), points, n_deriv, actual)
        for j, pt in enumerate(points):
            models = {
                ("narrow", "wide-fac"): _model_bernstein_derivs(
                    degree, pt, n_deriv, dtype, dtype, widen_the_factorial=True
                ),
                ("narrow", "narrow-fac"): _model_bernstein_derivs(
                    degree, pt, n_deriv, dtype, dtype, widen_the_factorial=False
                ),
                ("wide", "wide-fac"): _model_bernstein_derivs(
                    degree, pt, n_deriv, np.float64, dtype, widen_the_factorial=True
                ),
            }
            total += actual[j].size
            for key, model in models.items():
                hits[key] += int(np.count_nonzero(model == actual[j]))
            fac_models_differ += int(
                np.count_nonzero(models[("narrow", "wide-fac")] != models[("narrow", "narrow-fac")])
            )

    name = np.dtype(dtype).name
    print(f"== 2d. _bernstein_derivs_point at {name}: {total} values ==")
    for (arith, fac), count in hits.items():
        print(f"   {arith:6s} intermediates + {fac:11s} scaling reproduces: {count}/{total}")
    print(f"   the two factorial-scaling models disagree on {fac_models_differ}/{total} values")
    print()
    ok = hits[("narrow", "wide-fac")] == total
    return 0 if ok and (dtype is np.float64 or fac_models_differ > 0) else 1


def measure_the_two_twins_agree(dtype: _Storage = np.float32) -> int:
    """The parallel and serial numba kernels must agree bit for bit.

    The C++ backend has one kernel at every batch size, so the oracle's
    ``_PARALLEL_MIN_NUM_PTS`` dispatch selects which of two kernels a parity claim is
    compared against. That is only sound if the two are bit-identical, which is
    asserted here rather than assumed: they differ in how the span search is spelled
    (vectorised ``searchsorted`` versus a scalar one per ``prange`` iteration), and
    both are exact integer results, so nothing should move.

    Args:
        dtype (_Storage): The storage dtype. Defaults to ``np.float32``.

    Returns:
        int: 0 when every value agrees.
    """
    rng = np.random.default_rng(_RNG_SEED)
    points = _points(rng, dtype)
    differing = total = 0

    for knots in _knot_vectors(rng, dtype):
        degree = int(np.count_nonzero(knots == knots[0]) - 1)
        if knots.size - degree - 1 < degree + 1:
            continue
        n_deriv = degree + 1
        shapes: list[tuple[Any, Any, tuple[int, ...]]] = [
            (
                _compute_basis_nurbs_book_impl,
                _compute_basis_nurbs_book_serial_impl,
                (points.size, degree + 1),
            ),
        ]
        for parallel, serial, shape in shapes:
            got_p = np.empty(shape, dtype=dtype)
            got_s = np.empty(shape, dtype=dtype)
            first_p = np.empty(points.size, dtype=np.int_)
            first_s = np.empty(points.size, dtype=np.int_)
            parallel(knots, degree, False, 1e-10, points, got_p, first_p)
            serial(knots, degree, False, 1e-10, points, got_s, first_s)
            total += got_p.size
            bits = np.uint32 if dtype is np.float32 else np.uint64
            differing += int(np.count_nonzero(got_p.view(bits) != got_s.view(bits)))
            differing += int(np.count_nonzero(first_p != first_s))

        deriv_shape = (points.size, n_deriv + 1, degree + 1)
        got_p = np.empty(deriv_shape, dtype=dtype)
        got_s = np.empty(deriv_shape, dtype=dtype)
        first_p = np.empty(points.size, dtype=np.int_)
        first_s = np.empty(points.size, dtype=np.int_)
        _compute_basis_deriv_nurbs_book_impl(
            knots, degree, False, 1e-10, n_deriv, points, got_p, first_p
        )
        _compute_basis_deriv_nurbs_book_serial_impl(
            knots, degree, False, 1e-10, n_deriv, points, got_s, first_s
        )
        total += got_p.size
        view = np.uint32 if dtype is np.float32 else np.uint64
        differing += int(np.count_nonzero(got_p.view(view) != got_s.view(view)))
        differing += int(np.count_nonzero(first_p != first_s))

    print(f"== 2c. parallel versus serial twin at {np.dtype(dtype).name} ==")
    print(f"   {differing} of {total} values differ")
    print()
    return 0 if differing == 0 else 1


# ---------------------------------------------------------------------------
# 3. Where the falling factorial wraps
# ---------------------------------------------------------------------------


@nb_jit(nopython=True)
def _kernel_falling_factorials(degree: int, n_deriv: int) -> npt.NDArray[np.int64]:
    """The kernel's own accumulator, extracted so its wrap can be observed."""
    out = np.zeros(n_deriv + 1, dtype=np.int64)
    fac = degree
    for k in range(1, n_deriv + 1):
        out[k] = fac
        fac *= degree - k
    return out


def report_the_factorial_wrap() -> int:
    """Print the lowest degree at which the step-3 scaling wraps, and check the model.

    Returns:
        int: 0 when :func:`falling_factorials` reproduces the compiled accumulator at
        every degree swept.
    """
    print("== 3. the step-3 falling factorial, degree!/(degree-k)! in an int64 ==")
    mismatches = 0
    first_wrapping_degree = None
    for degree in range(41):
        compiled = _kernel_falling_factorials(degree, degree + 2)
        modelled = falling_factorials(degree, degree + 2)
        if [int(v) for v in compiled] != modelled:
            mismatches += 1
        exact = [0]
        value = 1
        for k in range(1, degree + 3):
            exact.append(value * degree if k == 1 else exact[k - 1] * (degree - k + 1))
        wrapped = [k for k in range(1, degree + 1) if int(compiled[k]) != exact[k]]
        if wrapped and first_wrapping_degree is None:
            first_wrapping_degree = (degree, wrapped[0])
    print(
        f"   the python model reproduces the compiled accumulator: "
        f"{'yes' if mismatches == 0 else f'NO, {mismatches} degrees differ'}"
    )
    if first_wrapping_degree is None:
        print("   no wrap up to degree 40")
    else:
        degree, order = first_wrapping_degree
        print(f"   lowest wrapping degree: {degree}, first at derivative order {order}")
        print("   so an independent accuracy claim on the k-th derivative row is")
        print(f"   defined only for degree < {degree}; parity is asserted above it too,")
        print("   because both backends wrap identically.")
    print()
    return 0 if mismatches == 0 else 1


def main() -> int:
    """Run every measurement and return a non-zero status if any model failed.

    Returns:
        int: 0 when every reported model reproduced the kernel exactly.
    """
    report_inferred_types()
    status = 0
    status |= measure_a2_2_width(np.float32)
    status |= measure_a2_2_width(np.float64)
    status |= measure_a2_3_width(np.float32)
    status |= measure_a2_3_width(np.float64)
    status |= measure_bernstein_deriv_width(np.float32)
    status |= measure_bernstein_deriv_width(np.float64)
    status |= measure_the_two_twins_agree(np.float32)
    status |= measure_the_two_twins_agree(np.float64)
    status |= report_the_factorial_wrap()
    print("all models reproduced the kernel" if status == 0 else "A MODEL FAILED, see above")
    return status


if __name__ == "__main__":
    sys.exit(main())
