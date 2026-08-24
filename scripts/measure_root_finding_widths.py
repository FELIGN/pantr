#!/usr/bin/env python
"""Measure the arithmetic width of every operation in the root-finding kernels.

``design/backend_parity.md`` Rule 9 says an oracle's accumulation width is a
**per-kernel fact, not a module convention**, and that getting it wrong is invisible
at float64. This script is the measurement for ``pantr.bezier``'s root-finding block,
and it is what the C++ transliteration in ``cpp/include/pantr/bezier/root_finding.hpp``
is written against.

The method is behavioural rather than a reading of Numba's type inference: for each
site, two rival models are run against the kernel and compared **bit for bit**. A
model that matches proves nothing on its own, so every site also reports how often
the two models *disagree*. A hypothesis confirmed by a check that cannot fail is not
confirmed.

    PYTHONPATH="$(pwd)/src:$(pwd)" .venv/bin/python scripts/measure_root_finding_widths.py

What it found when it was written
---------------------------------

**Numba's ``float()`` does not widen a float32.** That is the finding the rest rests
on, and it is the opposite of what the source reads like: ``float(coeff[0])`` is a
no-op, and so is ``float(np.min(coeff))``. What widens is type unification across
assignments, so a variable seeded with ``0.0`` or ``float("inf")`` and later given a
float32 is float64 throughout. Six sites in the block carry a ``float()`` a reader
would take for a promotion; none of them promotes.

Five widths follow, three of them not what a C++ author would write by default:

* the de Casteljau triangle accumulates in ``float64``, because the parameter is a
  float64 argument, and **narrows on every store** into the ``coeff.copy()`` buffer;
* the derivative kernel does the same and then takes ``d1 - d0`` **in float32**;
* the hull predicate subtracts ``coeff[i] - coeff[j0]`` **in float32** before the
  integer factor promotes the product;
* the Yuksel forward difference subtracts **in float32** and widens on the store into
  a ``float64`` array;
* the degree-1 base case divides ``c0 / (c0 - c1)`` **entirely in float32**. This is
  the worst of the five: the float64 model matched *none* of 20000 pairs, so a C++
  ``double root = c0 / (c0 - c1);`` breaks parity on every float32 input, and it reads
  as round-off rather than as a defect.

And two sites that look exactly like those and are not, which is why the table cannot
be read as a per-kernel or per-module rule. ``_batch_core.py:87`` divides in float64
and ``_yuksel_core.py:309`` divides the same shape of expression in float32; what
separates them is whether some other assignment forced the variable wider. The C++ has
to be transliterated expression by expression.

Two of the widths are measured but unreachable through the public entry points:
``_clip_hull_to_zero`` and ``_count_sign_changes`` are only ever called on the float64
array that ``_subdivide_scalar`` returns. They are here because the kernels accept
float32 and a direct caller can reach them.

Plus two library facts and one hazard that is not about width at all. See
:func:`measure_the_cube_root`, :func:`report_minimum_and_maximum` and
:func:`measure_the_hull_under_contraction`.

"""

from __future__ import annotations

import sys
from fractions import Fraction
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt

from pantr._numba_compat import nb_jit
from pantr.bezier._root_finding_core import (
    _clip_hull_to_zero,
    _de_casteljau_eval_and_deriv_scalar,
    _de_casteljau_eval_scalar,
)

_SEED: Final = 20260824
"""Fixed so a rerun reproduces the table rather than a similar one.

Every measurement gets its **own** generator derived from this, rather than all of
them drawing from one. Sharing a generator makes each number depend on how many
samples the measurements above it happened to take, so adding a site silently
moves every figure below it, and a number quoted from this script goes stale
without anything changing about what it measures."""

_F32: Final = np.float32
"""The narrow dtype, spelled once."""

_DEGREES: Final = (1, 2, 3, 5, 8, 13, 21, 34)
"""A spread rather than a sweep: the widths do not depend on degree, but the number
of roundings that can accumulate before the two models separate does."""

_FloatArray = npt.NDArray[np.float32 | np.float64]
"""A coefficient array at either width."""

_COLLINEAR_SHARE: Final = 0.5
"""Fraction of hull samples drawn as a near-straight control polygon."""

_HULL_MIN_STACK: Final = 2
"""Andrew's monotone chain needs two vertices on the stack before it can turn."""


class Verdict(NamedTuple):
    """One width hypothesis, measured against the kernel.

    Attributes:
        site (str): Where the operation lives, as ``file:line`` or a kernel name.
        hypothesis (str): The width this model assumes.
        matched (int): Cases in which the model reproduced the oracle bit for bit.
        total (int): Cases run.
        rivals_differ (int): Cases in which this model and its rival disagree. A
            hypothesis whose rival never disagrees has not been discriminated, and
            the match count means nothing.
    """

    site: str
    hypothesis: str
    matched: int
    total: int
    rivals_differ: int


def bits(value: float) -> int:
    """Reinterpret a float64 as its integer bit pattern, so equality is bitwise.

    Args:
        value (float): The value to reinterpret. ``nan`` compares equal to itself
            here, which ``==`` does not.

    Returns:
        int: The IEEE 754 binary64 bit pattern as a signed integer.
    """
    return int(np.float64(value).view(np.int64))


def _round_to_double(exact: Fraction) -> float:
    """Round an exact rational to the nearest float64.

    Used to build a fused reference without ``math.fma``, which arrived in Python
    3.13 while this project supports 3.11. ``float(Fraction)`` divides two integers,
    and CPython's integer true division is correctly rounded, so this is exact.

    Args:
        exact (Fraction): The exact value.

    Returns:
        float: The nearest float64, ties to even.
    """
    return float(exact)


def fused_product_difference(a: float, b: float, c: float, d: float) -> float:
    """Compute ``a*b - c*d`` the way a contracting compiler may emit it.

    With ``-ffp-contract=on`` a compiler is free to turn ``a*b - c*d`` into
    ``fma(a, b, -(c*d))``: the first product stays exact and only the subtraction
    rounds, where the unfused form rounds both products and then the subtraction.

    Args:
        a (float): First factor of the first product.
        b (float): Second factor of the first product.
        c (float): First factor of the second product.
        d (float): Second factor of the second product.

    Returns:
        float: One correctly rounded result of ``a*b - fl(c*d)``.
    """
    return _round_to_double(Fraction(a) * Fraction(b) - Fraction(c * d))


@nb_jit(nopython=True, cache=False)
def _forward_difference(coeff: _FloatArray, n: int) -> npt.NDArray[np.float64]:
    """Reproduce the Yuksel derivative-chain seed, ``_yuksel_core.py:318-319``.

    Args:
        coeff (_FloatArray): Bernstein coefficients of length ``n + 1``.
        n (int): The degree.

    Returns:
        npt.NDArray[np.float64]: The first forward difference, in a float64 array.

    Note:
        Inputs are assumed to be correct (no validation performed). This is a probe,
        not a library function.
    """
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        out[i] = coeff[i + 1] - coeff[i]
    return out


@nb_jit(nopython=True, cache=False)
def _linear_root(coeff: _FloatArray) -> float:
    """Reproduce the degree-1 base case, ``_yuksel_core.py:309``.

    Args:
        coeff (_FloatArray): Two Bernstein coefficients.

    Returns:
        float: ``c0 / (c0 - c1)``, at whatever width Numba chooses.

    Note:
        Inputs are assumed to be correct (no validation performed). The caller
        guarantees ``coeff[0] != coeff[1]``.
    """
    c0 = coeff[0]
    c1 = coeff[1]
    return float(c0 / (c0 - c1))


@nb_jit(nopython=True, cache=False)
def _hull_cross(coeff: _FloatArray, i: int, j0: int, j1: int) -> float:
    """Reproduce the orientation predicate, ``_root_finding_core.py:274``.

    Args:
        coeff (_FloatArray): Bernstein coefficients.
        i (int): Index of the candidate vertex.
        j0 (int): Index of the earlier stack vertex.
        j1 (int): Index of the later stack vertex.

    Returns:
        float: The signed area of the triple, at whatever width Numba chooses.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    return float((j1 - j0) * (coeff[i] - coeff[j0]) - (coeff[j1] - coeff[j0]) * (i - j0))


def measure_de_casteljau(rng: np.random.Generator) -> list[Verdict]:
    """Measure the width of the scalar de Casteljau triangle.

    The working buffer is ``coeff.copy()`` (``_root_finding_core.py:51``), so it
    carries the input dtype while the parameter is a Python float. Every store
    therefore narrows.

    Args:
        rng (np.random.Generator): Source of the coefficient samples.

    Returns:
        list[Verdict]: The wide model and the narrow model, in that order.
    """

    def wide(coeff: _FloatArray, t: float) -> float:
        work = coeff.copy()
        n = len(work) - 1
        for k in range(1, n + 1):
            for i in range(n - k + 1):
                work[i] = (1.0 - t) * float(work[i]) + t * float(work[i + 1])
        return float(work[0])

    def narrow(coeff: _FloatArray, t: float) -> float:
        work = coeff.copy()
        n = len(work) - 1
        s, tt = _F32(1.0) - _F32(t), _F32(t)
        for k in range(1, n + 1):
            for i in range(n - k + 1):
                work[i] = _F32(_F32(s * work[i]) + _F32(tt * work[i + 1]))
        return float(work[0])

    hits = [0, 0]
    differ = total = 0
    for degree in _DEGREES:
        for _trial in range(400):
            coeff = rng.standard_normal(degree + 1).astype(_F32)
            t = float(rng.uniform(0.0, 1.0))
            oracle = bits(_de_casteljau_eval_scalar(coeff, t))
            models = (bits(wide(coeff, t)), bits(narrow(coeff, t)))
            total += 1
            hits = [h + (m == oracle) for h, m in zip(hits, models, strict=True)]
            differ += models[0] != models[1]

    site = "_root_finding_core.py:51"
    return [
        Verdict(site, "accumulate float64, narrow on store", hits[0], total, differ),
        Verdict(site, "every operation in float32", hits[1], total, differ),
    ]


def measure_de_casteljau_derivative(rng: np.random.Generator) -> list[Verdict]:
    """Measure the width of ``d1 - d0`` in the value-and-derivative kernel.

    The triangle itself behaves as :func:`measure_de_casteljau` found, so this
    isolates the one operation that differs: the penultimate row is read out of the
    narrow buffer and subtracted **before** the degree factor promotes the product.

    Args:
        rng (np.random.Generator): Source of the coefficient samples.

    Returns:
        list[Verdict]: The float32 subtraction and the widened subtraction.
    """

    def model(coeff: _FloatArray, t: float, *, narrow_diff: bool) -> float:
        n = len(coeff) - 1
        if n == 0:
            return 0.0
        s = 1.0 - t
        row = coeff.copy()
        for k in range(1, n):
            for i in range(n - k + 1):
                row[i] = s * float(row[i]) + t * float(row[i + 1])
        d0, d1 = row[0], row[1]
        diff = float(_F32(d1 - d0)) if narrow_diff else float(d1) - float(d0)
        return float(n) * diff

    hits = [0, 0]
    differ = total = 0
    for degree in _DEGREES:
        for _trial in range(300):
            coeff = rng.standard_normal(degree + 1).astype(_F32)
            t = float(rng.uniform(0.0, 1.0))
            oracle = bits(_de_casteljau_eval_and_deriv_scalar(coeff, t)[1])
            models = (
                bits(model(coeff, t, narrow_diff=True)),
                bits(model(coeff, t, narrow_diff=False)),
            )
            total += 1
            hits = [h + (m == oracle) for h, m in zip(hits, models, strict=True)]
            differ += models[0] != models[1]

    site = "_root_finding_core.py:157"
    return [
        Verdict(site, "d1 - d0 in float32", hits[0], total, differ),
        Verdict(site, "d1 - d0 widened to float64", hits[1], total, differ),
    ]


def measure_hull_predicate(rng: np.random.Generator) -> list[Verdict]:
    """Measure the width of the convex-hull orientation predicate.

    The index factors are ``int64``, and numpy promotes ``int64`` with ``float32`` to
    ``float64``, so the *products* are float64 at both dtypes. The question is the
    coefficient differences, which happen first.

    A C++ template over the scalar type gets this wrong in the opposite direction:
    ``int64 * float`` is ``float`` in C++ but ``float64`` in numpy, so the integer
    factor needs an explicit cast to ``double``.

    Args:
        rng (np.random.Generator): Source of the coefficient samples.

    Returns:
        list[Verdict]: The in-dtype model and the widen-first model.
    """

    def in_dtype(coeff: _FloatArray, i: int, j0: int, j1: int) -> float:
        d1, d2 = coeff[i] - coeff[j0], coeff[j1] - coeff[j0]
        return float(j1 - j0) * float(d1) - float(d2) * float(i - j0)

    def widen_first(coeff: _FloatArray, i: int, j0: int, j1: int) -> float:
        return float(j1 - j0) * (float(coeff[i]) - float(coeff[j0])) - (
            float(coeff[j1]) - float(coeff[j0])
        ) * float(i - j0)

    hits = [0, 0]
    differ = total = 0
    for degree in (3, 6, 10, 17, 25):
        for _trial in range(2000):
            coeff = _sample_coefficients(rng, degree, _F32)
            j0, j1, i = (int(v) for v in sorted(rng.choice(degree + 1, 3, replace=False)))
            oracle = bits(_hull_cross(coeff, i, j0, j1))
            models = (bits(in_dtype(coeff, i, j0, j1)), bits(widen_first(coeff, i, j0, j1)))
            total += 1
            hits = [h + (m == oracle) for h, m in zip(hits, models, strict=True)]
            differ += models[0] != models[1]

    site = "_root_finding_core.py:274"
    return [
        Verdict(site, "coefficient differences in float32", hits[0], total, differ),
        Verdict(site, "coefficient differences widened first", hits[1], total, differ),
    ]


def measure_forward_difference(rng: np.random.Generator) -> list[Verdict]:
    """Measure the width of the Yuksel derivative-chain seed.

    The destination array is ``float64``, which is what makes this one easy to get
    wrong: the array's dtype says nothing about the width the subtraction ran at.

    Args:
        rng (np.random.Generator): Source of the coefficient samples.

    Returns:
        list[Verdict]: The float32 subtraction and the widened subtraction.
    """
    hits = [0, 0]
    differ = total = 0
    for _trial in range(4000):
        n = int(rng.integers(2, 30))
        coeff = rng.standard_normal(n + 1).astype(_F32)
        oracle = [bits(v) for v in _forward_difference(coeff, n)]
        narrow = [bits(float(_F32(coeff[i + 1] - coeff[i]))) for i in range(n)]
        wide = [bits(float(coeff[i + 1]) - float(coeff[i])) for i in range(n)]
        total += 1
        hits = [h + (m == oracle) for h, m in zip(hits, (narrow, wide), strict=True)]
        differ += narrow != wide

    site = "_yuksel_core.py:319"
    return [
        Verdict(site, "subtract in float32, widen on store", hits[0], total, differ),
        Verdict(site, "widen, then subtract in float64", hits[1], total, differ),
    ]


def measure_linear_base_case(rng: np.random.Generator) -> list[Verdict]:
    """Measure the width of the degree-1 base case ``c0 / (c0 - c1)``.

    Every Yuksel recursion bottoms out here, and the result is stored into a
    ``float64`` array, so the narrow division is invisible downstream.

    Args:
        rng (np.random.Generator): Source of the coefficient samples.

    Returns:
        list[Verdict]: The float32 division and the float64 division.
    """
    hits = [0, 0]
    differ = total = 0
    for _trial in range(20000):
        coeff = rng.standard_normal(2).astype(_F32)
        if coeff[0] == coeff[1]:
            continue
        oracle = bits(_linear_root(coeff))
        models = (
            bits(float(_F32(coeff[0] / _F32(coeff[0] - coeff[1])))),
            bits(float(coeff[0]) / (float(coeff[0]) - float(coeff[1]))),
        )
        total += 1
        hits = [h + (m == oracle) for h, m in zip(hits, models, strict=True)]
        differ += models[0] != models[1]

    site = "_yuksel_core.py:309"
    return [
        Verdict(site, "divide in float32", hits[0], total, differ),
        Verdict(site, "divide in float64", hits[1], total, differ),
    ]


def measure_the_near_misses(rng: np.random.Generator) -> list[Verdict]:
    """Measure two sites that look like the traps above and are not.

    Both read out of a float32 array and both look like they then work in float64,
    one because an explicit ``float()`` stands in the way and the other because the
    variable is seeded with ``float("inf")``. Only the second actually widens, and
    that is the whole point of measuring rather than reading: see
    :func:`report_the_float_cast`.

    They are here so the table cannot be read as a per-kernel or per-module rule.
    ``_batch_core.py:87`` divides in float64 and ``_yuksel_core.py:309`` divides the
    same shape of expression in float32, and what separates them is whether some
    other assignment to the same variable forced a wider type.

    Args:
        rng (np.random.Generator): Source of the coefficient samples.

    Returns:
        list[Verdict]: Each site's correct model first, its plausible rival second.
    """

    @nb_jit(nopython=True, cache=False)
    def level_scale(coeff: _FloatArray) -> float:
        """Reproduce ``_yuksel_core.py:206-207``.

        Args:
            coeff (_FloatArray): Bernstein coefficients.

        Returns:
            float: ``abs(d_max - d_min)`` with both ends widened before subtracting.

        Note:
            Inputs are assumed to be correct (no validation performed).
        """
        d_min = float(np.min(coeff))
        d_max = float(np.max(coeff))
        return float(abs(d_max - d_min))

    @nb_jit(nopython=True, cache=False)
    def coefficient_range(coeff: _FloatArray) -> float:
        """Reproduce ``_batch_core.py:79-87``.

        Args:
            coeff (_FloatArray): Bernstein coefficients, none of them zero.

        Returns:
            float: The ratio of the largest magnitude to the smallest non-zero one.

        Note:
            Inputs are assumed to be correct (no validation performed).
        """
        c_max = 0.0
        c_min_nonzero = float("inf")
        for i in range(len(coeff)):
            av = abs(coeff[i])
            c_max = max(c_max, av)
            if av > 0.0 and av < c_min_nonzero:
                c_min_nonzero = av
        return float(c_max / c_min_nonzero)

    scale_hits = [0, 0]
    range_hits = [0, 0]
    scale_differ = range_differ = total = 0
    for _trial in range(6000):
        degree = int(rng.integers(2, 30))
        coeff = rng.standard_normal(degree + 1).astype(_F32)
        low, high = _F32(np.min(coeff)), _F32(np.max(coeff))
        wide_scale = bits(abs(float(high) - float(low)))
        narrow_scale = bits(float(abs(_F32(high - low))))
        oracle_scale = bits(level_scale(coeff))

        magnitudes = np.abs(coeff)
        biggest, smallest = _F32(magnitudes.max()), _F32(magnitudes.min())
        if smallest == 0.0:
            continue
        wide_range = bits(float(biggest) / float(smallest))
        narrow_range = bits(float(_F32(biggest / smallest)))
        oracle_range = bits(coefficient_range(coeff))

        total += 1
        scale_hits = [
            h + (m == oracle_scale)
            for h, m in zip(scale_hits, (wide_scale, narrow_scale), strict=True)
        ]
        range_hits = [
            h + (m == oracle_range)
            for h, m in zip(range_hits, (wide_range, narrow_range), strict=True)
        ]
        scale_differ += wide_scale != narrow_scale
        range_differ += wide_range != narrow_range

    return [
        Verdict(
            "_yuksel_core.py:207",
            "subtract in float32; float() is a no-op",
            scale_hits[1],
            total,
            scale_differ,
        ),
        Verdict("_yuksel_core.py:207", "subtract in float64", scale_hits[0], total, scale_differ),
        Verdict("_batch_core.py:87", "divide in float64", range_hits[0], total, range_differ),
        Verdict("_batch_core.py:87", "divide in float32", range_hits[1], total, range_differ),
    ]


def _sample_coefficients(
    rng: np.random.Generator, degree: int, dtype: type[np.floating]
) -> _FloatArray:
    """Draw coefficients, half of them a near-collinear control polygon.

    The hull predicate is fragile exactly where the control polygon is close to
    straight, and independent normal samples almost never land there.

    Args:
        rng (np.random.Generator): Source of randomness.
        degree (int): Polynomial degree; the array has ``degree + 1`` entries.
        dtype (type[np.floating]): The width to return.

    Returns:
        _FloatArray: The coefficients.
    """
    if rng.random() < _COLLINEAR_SHARE:
        index = np.arange(degree + 1, dtype=np.float64)
        line = rng.standard_normal() + rng.standard_normal() * index
        return np.asarray(line + rng.standard_normal(degree + 1) * 1e-7, dtype=dtype)
    return np.asarray(rng.standard_normal(degree + 1), dtype=dtype)


def measure_the_cube_root(rng: np.random.Generator) -> None:
    """Report which C++ spelling reproduces ``zero_tol ** (1/3)``.

    ``_clipping_core.py:349`` is the only call in the whole block that leaves the
    four arithmetic operations, so it is the only place a library can disagree.

    Args:
        rng (np.random.Generator): Source of the sample exponents.
    """

    @nb_jit(nopython=True, cache=False)
    def kernel(x: float) -> float:
        """Reproduce the one non-arithmetic call in the block.

        Args:
            x (float): The tolerance to take the cube root of.

        Returns:
            float: ``x ** (1/3)``.

        Note:
            Inputs are assumed to be correct (no validation performed).
        """
        return float(x ** (1.0 / 3.0))

    import math  # noqa: PLC0415 -- only this function needs it

    mismatch_pow = mismatch_cbrt = total = 0
    for _trial in range(50000):
        x = float(10.0 ** rng.uniform(-300.0, 3.0)) * float(rng.uniform(0.5, 2.0))
        reference = bits(kernel(x))
        total += 1
        mismatch_pow += reference != bits(math.pow(x, 1.0 / 3.0))
        mismatch_cbrt += reference != bits(math.cbrt(x))

    print(f"\nThe one library call, {total} tolerances spanning 1e-300 to 1e3")
    print(f"  std::pow(x, 1.0/3.0)   differs on {mismatch_pow:>6} -- this is the spelling to use")
    print(f"  std::cbrt(x)           differs on {mismatch_cbrt:>6} -- do not transliterate to this")


def report_minimum_and_maximum() -> None:
    """Report whether the C++ ``std::min``/``std::max`` ternary matches Numba.

    ``std::fmin`` and ``std::fmax`` do not: they return the non-NaN operand, where
    the ternary and Numba's builtins propagate whichever the comparison selects.
    The block compares tolerances against values that can be NaN, so this decides a
    branch rather than a last bit.
    """

    @nb_jit(nopython=True, cache=False)
    def kernel(a: float, b: float) -> tuple[float, float]:
        """Return Numba's ``min`` and ``max`` of two floats.

        Args:
            a (float): First operand.
            b (float): Second operand.

        Returns:
            tuple[float, float]: ``(min(a, b), max(a, b))``.

        Note:
            Inputs are assumed to be correct (no validation performed).
        """
        return min(a, b), max(a, b)

    nan, inf = float("nan"), float("inf")
    print("\nmin/max, against the C++ ternary that std::min and std::max expand to")
    for a, b in ((nan, 1.0), (1.0, nan), (-0.0, 0.0), (0.0, -0.0), (-inf, inf)):
        low, high = kernel(a, b)
        agrees = bits(low) == bits(b if b < a else a) and bits(high) == bits(b if a < b else a)
        print(
            f"  min({a!r:>6}, {b!r:>6}) = {low!r:>6}   max = {high!r:>6}   ternary agrees: {agrees}"
        )


def report_the_float_cast(rng: np.random.Generator) -> bool:
    """Show that Numba's ``float()`` does not widen a float32, and what does.

    This is the finding the rest of the table rests on, and it is the opposite of
    what the source reads like. ``float(coeff[0])`` looks like a widening cast and
    is a no-op: the value stays float32 and so does every operation built on it.
    What *does* widen is type unification across assignments, so a variable seeded
    with ``0.0`` or ``float("inf")`` and later given a float32 is float64 throughout.

    Six sites in this block carry an explicit ``float()`` that a reader would take
    for a promotion. None of them promotes, and that is why three of the five widths
    above are narrower than the source suggests.

    Measured behaviourally rather than read off a compiled signature, so it stands on
    the same evidence as everything else here.

    Args:
        rng (np.random.Generator): Source of the coefficient samples.

    Returns:
        bool: True when ``float()`` was observed to widen, which would invalidate the
            widths above and is therefore a failure.
    """

    @nb_jit(nopython=True, cache=False)
    def through_a_cast(coeff: _FloatArray) -> float:
        """Subtract two array elements that were passed through ``float()``.

        Args:
            coeff (_FloatArray): Two Bernstein coefficients.

        Returns:
            float: Their difference, at whatever width Numba chooses.

        Note:
            Inputs are assumed to be correct (no validation performed).
        """
        low = float(coeff[0])
        high = float(coeff[1])
        return float(high - low)

    @nb_jit(nopython=True, cache=False)
    def through_a_seed(coeff: _FloatArray) -> float:
        """Subtract two array elements held in variables seeded with a float64.

        Args:
            coeff (_FloatArray): Two Bernstein coefficients.

        Returns:
            float: Their difference, at whatever width Numba chooses.

        Note:
            Inputs are assumed to be correct (no validation performed).
        """
        low = 0.0
        high = 0.0
        for i in range(len(coeff)):
            if i == 0:
                low = coeff[i]
            else:
                high = coeff[i]
        return float(high - low)

    cast_narrow = seed_wide = total = 0
    for _trial in range(6000):
        pair = rng.standard_normal(2).astype(_F32)
        narrow = bits(float(_F32(pair[1] - pair[0])))
        wide = bits(float(pair[1]) - float(pair[0]))
        if narrow == wide:
            continue
        total += 1
        cast_narrow += bits(through_a_cast(pair)) == narrow
        seed_wide += bits(through_a_seed(pair)) == wide

    print(f"\nWhat float() does to a float32, {total} discriminating pairs")
    print(f"  float(c[1]) - float(c[0])          stayed narrow in {cast_narrow}/{total}")
    print(f"  the same, via variables seeded 0.0  widened     in {seed_wide}/{total}")
    widens = cast_narrow != total
    print(f"  float() is a no-op on a float32: {not widens}; unification is what widens")
    return widens


def transliterate_hull(coeff: npt.NDArray[np.float64], *, fuse: bool) -> tuple[float, float, bool]:
    """Reimplement ``_clip_hull_to_zero`` with the orientation predicate switchable.

    Args:
        coeff (npt.NDArray[np.float64]): Bernstein coefficients.
        fuse (bool): Evaluate the predicate as a contracting compiler would.

    Returns:
        tuple[float, float, bool]: ``(t_lo, t_hi, found)``, as the kernel returns.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    n = len(coeff) - 1
    if n < 1:
        return 0.0, 0.0, False
    inv_n = 1.0 / n
    t_lo, t_hi, found = 1.0, 0.0, False

    for upper in (True, False):
        chain = np.empty(n + 1, dtype=np.int64)
        size = 0
        for i in range(n + 1):
            while size >= _HULL_MIN_STACK:
                j0, j1 = int(chain[size - 2]), int(chain[size - 1])
                a, b = float(j1 - j0), float(coeff[i]) - float(coeff[j0])
                c, d = float(coeff[j1]) - float(coeff[j0]), float(i - j0)
                cross = fused_product_difference(a, b, c, d) if fuse else a * b - c * d
                if (cross >= 0.0) if upper else (cross <= 0.0):
                    size -= 1
                else:
                    break
            chain[size] = i
            size += 1
        for k in range(size - 1):
            ia, ib = int(chain[k]), int(chain[k + 1])
            da, db = float(coeff[ia]), float(coeff[ib])
            if da * db < 0.0:
                ta, tb = ia * inv_n, ib * inv_n
                crossing = ta + (-da) / (db - da) * (tb - ta)
                t_lo, t_hi, found = min(t_lo, crossing), max(t_hi, crossing), True
            if da == 0.0:
                ta = ia * inv_n
                t_lo, t_hi, found = min(t_lo, ta), max(t_hi, ta), True
        last = int(chain[size - 1])
        if coeff[last] == 0.0:
            ta = last * inv_n
            t_lo, t_hi, found = min(t_lo, ta), max(t_hi, ta), True
    return t_lo, t_hi, found


def measure_the_hull_under_contraction(rng: np.random.Generator) -> None:
    """Report whether contraction can change the hull, and on what input.

    This is not a width question and it is the one hazard in the block that a
    tolerance cannot absorb. The predicate is ``a*b - c*d``, and on an exactly
    collinear control polygon it evaluates to **exactly zero**, so ``cross >= 0``
    holds and the vertex is popped. Fusing turns that exact tie into a signed
    residue, and the tie-break becomes a coin toss.

    A collinear control polygon is not an exotic input: a degree-elevated linear
    polynomial, an affine segment and a constant all produce one.

    Args:
        rng (np.random.Generator): Source of the coefficient samples.
    """
    random = [rng.standard_normal(d + 1) for d in (3, 6, 10, 17, 25) for _ in range(600)]
    collinear = [
        rng.standard_normal() + rng.standard_normal() * np.arange(d + 1, dtype=np.float64)
        for d in range(2, 26)
        for _ in range(200)
    ]

    print("\nThe hull predicate under contraction, against the kernel")
    for name, cases in (("random", random), ("collinear", collinear)):
        for fuse in (False, True):
            same = sum(transliterate_hull(c, fuse=fuse) == _clip_hull_to_zero(c) for c in cases)
            label = "fused  " if fuse else "unfused"
            print(f"  {label} on {name:>9}: {same:>4}/{len(cases)} identical")


def report(verdicts: list[Verdict]) -> bool:
    """Print the width table and say whether every site was discriminated.

    Args:
        verdicts (list[Verdict]): The measurements, rivals adjacent and the expected
            winner first.

    Returns:
        bool: True when every site has exactly one model matching every case and a
            rival that disagrees somewhere, which is what makes the match meaningful.
    """
    print(f"{'site':<32} {'hypothesis':<40} {'matched':>12} {'discriminated':>14}")
    sound = True
    for index, verdict in enumerate(verdicts):
        share = f"{verdict.matched}/{verdict.total}"
        print(
            f"{verdict.site:<32} {verdict.hypothesis:<40} {share:>12} {verdict.rivals_differ:>14}"
        )
        expected_winner = index % 2 == 0
        won = verdict.matched == verdict.total
        if won != expected_winner or verdict.rivals_differ == 0:
            sound = False
    return sound


def main() -> int:
    """Run every measurement and report.

    Returns:
        int: 0 when every width was confirmed and discriminated, 1 otherwise.
    """
    seeds = np.random.SeedSequence(_SEED).spawn(9)
    verdicts: list[Verdict] = []
    measurements = (
        measure_de_casteljau,
        measure_de_casteljau_derivative,
        measure_hull_predicate,
        measure_forward_difference,
        measure_linear_base_case,
        measure_the_near_misses,
    )
    for seed, measure in zip(seeds, measurements, strict=False):
        verdicts.extend(measure(np.random.default_rng(seed)))
    sound = report(verdicts)

    widened = report_the_float_cast(np.random.default_rng(seeds[6]))
    measure_the_cube_root(np.random.default_rng(seeds[7]))
    report_minimum_and_maximum()
    measure_the_hull_under_contraction(np.random.default_rng(seeds[8]))

    if widened:
        print("\nfloat() widened after all, so every width above needs re-deriving.")
        return 1
    if not sound:
        print("\nA width was not confirmed, or its check could not have failed.")
        return 1
    print("\nEvery width confirmed, and every check discriminates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
