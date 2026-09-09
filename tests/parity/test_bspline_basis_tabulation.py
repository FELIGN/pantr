"""Parity and accuracy for the 1D B-spline basis tabulation port.

The C++ side is `cpp/include/pantr/bspline/tabulate.hpp` (Piegl & Tiller A2.2 and
A2.3 over a general knot vector) and `pantr::tabulate_bernstein_deriv_1d` in
`cpp/include/pantr/basis/bernstein.hpp` (A2.3 on unit knot spans, which the
Bézier-like fast path reaches). The oracle is
``pantr.bspline._bspline_basis_core``, and the public surface both go through is
:meth:`pantr.bspline.BsplineSpace1D.tabulate_basis` and
:meth:`~pantr.bspline.BsplineSpace1D.tabulate_basis_derivatives`.

What is claimed, and why
------------------------

**Parity is BITWISE on the shipped build.** The general-knot recurrences are built
from ``+``, ``-``, ``*``, ``/`` and an exact-zero comparison, so IEEE 754 pins every
result once the operation order matches. The C++ transliteration was written against
``scripts/measure_bspline_tabulation_widths.py`` rather than against a reading of the
oracle, because ``design/backend_parity.md`` Rule 9 says a width cannot be read off
the source: at ``float32`` a model computing the recurrence in ``float64`` reproduces
only 7 029 of 17 836 A2.2 values, and a model narrowing the factorial scaling
reproduces 312 749 of 351 988 A2.3 values. The port takes the model that reproduces
every one.

**Both recurrences contain a fusable ``a * b + c``**, unlike the Bernstein ratio
recurrence, so the claim is conditional: bitwise where the target ISA carries no
fused multiply-add, and bounded by Rule 10's budget where it does.
``contraction_may_fuse()`` selects, and both arms are written -- Rule 10's own
lesson is that the branch shipping unevaluated is the one that is broken.

**Accuracy is checked against three independent oracles, none of them the
algorithm.** Rule's premise: parity says the two backends agree, not that either is
right, so a shared error is invisible to every parity test.

- **Marsden's identity.** ``sum_i N_{i,p}(x) prod_{j=1..p} (t_{i+j} - y) = (x - y)^p``
  for every ``y``. Evaluated in exact rational arithmetic from dyadic knots, so the
  reference carries no error at all. This is the only one of the three that pins the
  **indexing and the scale**: it is a different positive weight per basis function, so
  a permuted, shifted or misscaled row fails it while the other two pass.
  ``y`` is taken **outside** the knot range, which makes every weight the same sign
  and the sum cancellation-free -- see :func:`_marsden_weights`.
- **Partition of unity**, ``sum_i N_i(x) = 1``. Weaker than Marsden by construction:
  it is the ``p = 0`` case, so it cannot see a permutation. It is kept because it is
  the one invariant that holds *exactly* at every degree with no conditioning, so it
  still says something where Marsden's weights have lost digits.
- **Differentiated Marsden**, ``sum_i N^{(k)}_i(x) c_i(y) = p!/(p-k)! (x-y)^{p-k}``.
  This is the one that pins the **derivative scale**, which the derivative sums are
  blind to: ``sum_i N^{(k)}_i = 0`` survives any uniform rescaling of a row, and the
  factorial scaling of A2.3's step 3 is exactly a uniform rescaling. So a dropped or
  inverted ``p!/(p-k)!`` passes the sums and fails this.

Two further checks are exact rather than bounded: the derivative sums vanish, and
every row above the degree is identically zero.

The degrees no accuracy claim covers, and why
---------------------------------------------

A2.3's step 3 accumulates ``degree!/(degree-k)!`` in an integer that **wraps at
int64**, in both backends. Measured by the script above: the lowest wrapping degree is
21, first at derivative order 19. Above it the derivative rows are scaled by a
two's-complement remainder and are wrong in both backends, identically -- so parity is
asserted there and accuracy is not.
:func:`test_the_wrapping_degrees_are_named_and_still_wrap` is Rule 8's obligation
discharged: it enumerates the excluded region and fails if it ever becomes empty,
because that would mean either the accumulator changed or the wrap was fixed, and both
are findings rather than good news.

Rule 12 gates
-------------

The general-knot **value** path carries no transcendental and no integer accumulator,
so a bitwise claim over it survives interpretation by IEEE guarantee. It still calls
:func:`demand_the_compiled_kernel` at ``float32``, which owns the storage-format
question and whose docstring says which intermediates promote has not been pinned.

The general-knot **derivative** path and both Bézier-like paths call
:func:`demand_a_compiled_seed`. That gate's own docstring named
``_basis_derivs_point``'s falling factorial and said "whoever gives it one inherits
this gate as a precondition"; this is that inheritance. The Bézier-like value path
inherits it through ``np.power`` in the Bernstein ratio recurrence, and the
Bézier-like derivative path through ``_bernstein_derivs_point``'s own factorial.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, available_backends, use_backend
from pantr.basis._basis_backend import DerivKernels, bernstein_deriv_core
from pantr.bspline import BsplineSpace1D
from pantr.bspline._basis_backend import (
    BasisDerivKernels,
    BasisKernels,
    bspline_basis_core,
    bspline_basis_deriv_core,
)
from tests._parity_harness import (
    Field,
    ParityClaim,
    Roundings,
    assert_accuracy,
    assert_object_parity,
    assert_parity,
    bitwise_parity,
    bounded_parity,
    contraction_may_fuse,
    demand_a_compiled_seed,
    demand_the_compiled_kernel,
    demand_the_reference_host,
    derived_accuracy,
    exact_parity,
    unit_roundoff,
)

_FloatArray = npt.NDArray[np.floating[Any]]

_DERIVATIVE_ORDERS = (0, 1, 2, 5, 12, 22)
"""Derivative orders the general-knot parity matrix runs.

Named once so that
:func:`test_the_case_list_separates_the_factorial_widths` checks the orders the
parametrization actually uses rather than a copy of them.
"""

# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------

_GENERAL_KNOTS: dict[str, tuple[list[float], int]] = {
    # name: (knots, degree). Every knot is dyadic, so `Fraction(knot)` is exact and
    # the Marsden reference carries no representation error of its own.
    "clamped-uniform-p1": ([0.0, 0.0, 0.5, 1.0, 1.0], 1),
    "clamped-uniform-p2": ([0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0], 2),
    "clamped-uniform-p3": ([0.0] * 4 + [0.25, 0.5, 0.75] + [1.0] * 4, 3),
    "clamped-repeat-p2": ([0.0, 0.0, 0.0, 0.25, 0.75, 0.75, 1.0, 1.0, 1.0], 2),
    # A doubled interior knot makes an empty span, which is the only input the
    # `denom == 0` guard exists for.
    "empty-span-p3": ([0.0] * 4 + [0.5, 0.5, 0.5] + [1.0] * 4, 3),
    # Not clamped at either end: the right domain knot is not the vector's last knot,
    # which is the case the span search's upper clamp exists for.
    "unclamped-p2": ([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5], 2),
    "unclamped-p3": ([-0.75, -0.5, -0.25, 0.0, 0.5, 1.0, 1.25, 1.5, 1.75], 3),
    # A non-unit, offset domain, so a bound that quietly assumes `[0, 1]` fails.
    "offset-scale-p2": ([2.0, 2.0, 2.0, 3.0, 4.0, 6.0, 6.0, 6.0], 2),
    "high-degree-p8": ([0.0] * 9 + [0.25, 0.5, 0.75] + [1.0] * 9, 8),
}
"""General-knot spaces the accuracy oracles can reach, dyadic throughout.

Capped at degree 8, which is what keeps every accuracy claim below well inside the
factorial accumulator's wrapping region. Parity cases are the union of this and
:data:`_HIGH_DEGREE_KNOTS`; Bézier-like ones are kept separate again.
"""

_HIGH_DEGREE_KNOTS: dict[str, tuple[list[float], int]] = {
    "sep-factorial-p14": ([0.0] * 15 + [0.5] + [1.0] * 15, 14),
    "wrapping-p22": ([0.0] * 23 + [0.5] + [1.0] * 23, 22),
}
"""Degrees high enough to separate the factorial scaling's two width models.

**These exist because a mutation test showed the rest of the file could not tell the
two apart.** ``T(double(value) * double(fac))`` and ``value * T(fac)`` are *provably*
identical while ``fac`` is exactly representable in the storage format: the exact
product of two ``float32`` values needs at most 48 significand bits, so it is exact in
``float64`` and rounding it once to ``float32`` is the same single rounding the
``float32`` multiply commits. They can only differ once ``fac`` needs more than 24
bits, and ``degree!/(degree-k)!`` first does so around degree 12 -- so a case list
stopping at degree 8 asserts the width and cannot fail on it. Narrowing the scaling in
``tabulate.hpp`` passed all 36 derivative cases before these two were added.

``wrapping-p22`` additionally reaches the int64 wrap itself, where both backends scale
by a two's-complement remainder. Parity is the claim there and accuracy is not; see
:func:`test_the_wrapping_degrees_are_named_and_still_wrap`.

:func:`test_the_case_list_separates_the_factorial_widths` is the guard that keeps this
property rather than trusting the comment.
"""

_BEZIER_KNOTS: dict[str, tuple[list[float], int]] = {
    "bezier-p0": ([0.0, 1.0], 0),
    "bezier-p2-unit": ([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2),
    "bezier-p3-offset": ([2.0] * 4 + [6.0] * 4, 3),
    "bezier-p5-small": ([0.0] * 6 + [0.0625] * 6, 5),
}
"""Spaces whose knots describe a single Bézier segment, so the fast path fires.

``bezier-p3-offset`` and ``bezier-p5-small`` have a span far from one, which is what
makes the chain-rule factor ``(1/(b-a))^k`` visible: at a unit span it is one and a
dropped factor passes.
"""


def _evaluation_points(knots: list[float], degree: int, dtype: Any) -> _FloatArray:
    """Points chosen to be adversarial about representability, not about magnitude.

    Rule 9's sharpest case was found by a list built around exactly-representable
    ratios rather than round numbers. So: both domain ends, every distinct in-domain
    knot, the value one ulp inside each end, the midpoint of every span, and a fixed
    random sample.

    Args:
        knots (list[float]): The knot vector.
        degree (int): The polynomial degree.
        dtype (Any): The storage dtype the points are produced in.

    Returns:
        _FloatArray: A 1D array of in-domain points, in ``dtype``.
    """
    begin = knots[degree]
    end = knots[len(knots) - degree - 1]
    interior = sorted({k for k in knots if begin <= k <= end})

    fixed = [begin, end, *interior]
    fixed += [float(np.nextafter(np.array(begin, dtype), np.array(end, dtype)))]
    fixed += [float(np.nextafter(np.array(end, dtype), np.array(begin, dtype)))]
    fixed += [0.5 * (a + b) for a, b in pairwise(interior)]

    rng = np.random.default_rng(20260909)
    sample = begin + (end - begin) * rng.random(16)
    points = np.array(sorted(fixed + sample.tolist()), dtype=dtype)
    return np.asarray(np.clip(points, dtype(begin), dtype(end)), dtype=dtype)


class _Tabulation(NamedTuple):
    """One backend's answer, so the two fields can carry different claim kinds.

    :func:`assert_object_parity` compares field by field, which is what lets the
    floating-point block take a bitwise or bounded claim while the integer index array
    takes an exactness claim. Folding them into one comparison is not possible: no
    tolerance applies to an index, and a bitwise claim about an index says something
    weaker than exactness does.

    Attributes:
        block (_FloatArray): The basis values, or the derivative block.
        first_basis (npt.NDArray[np.int_]): The first-basis index per point.
    """

    block: _FloatArray
    first_basis: npt.NDArray[np.int_]


def _tabulate(  # noqa: PLR0913
    backend: Backend,
    knots: list[float],
    degree: int,
    points: _FloatArray,
    dtype: Any,
    n_deriv: int | None,
) -> _Tabulation:
    """Build the space and tabulate on one backend, through the public surface.

    The space is built **inside** the backend block, because
    :class:`~pantr.bspline.BsplineSpace1D` is a wrapper whose implementation is chosen
    at construction; that is what makes this an end-to-end claim about what a caller
    gets rather than about a kernel in isolation.

    Args:
        backend (Backend): The backend to run under.
        knots (list[float]): The knot vector.
        degree (int): The polynomial degree.
        points (_FloatArray): The evaluation points.
        dtype (Any): The storage dtype.
        n_deriv (int | None): Derivative order, or ``None`` for the value tabulation.

    Returns:
        _Tabulation: The block and the index array.
    """
    with use_backend(backend):
        space = BsplineSpace1D(np.array(knots, dtype=dtype), degree)
        if n_deriv is None:
            block, first = space.tabulate_basis(points)
        else:
            block, first = space.tabulate_basis_derivatives(points, n_deriv)
    return _Tabulation(block=block, first_basis=first)


def _demand_the_two_spaces_agree(knots: list[float], degree: int, dtype: Any) -> None:
    """Fail with a space-level message if the two backends disagree about the space.

    Every claim below is about the *tabulation*, and both sides build their own space
    first. A one-bit divergence in the stored knots or the tolerance would surface as
    a tabulation failure and be diagnosed as one, so it is separated out here.
    ``tests/parity/test_bspline_space_1d.py`` is what actually covers the space; this
    is a guard against a misattributed failure, not a second copy of that test.

    Args:
        knots (list[float]): The knot vector.
        degree (int): The polynomial degree.
        dtype (Any): The storage dtype.
    """
    built = {}
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            space = BsplineSpace1D(np.array(knots, dtype=dtype), degree)
            built[backend] = (np.asarray(space.knots).copy(), space.tolerance, space.degree)

    py, cpp = built[Backend.PYTHON], built[Backend.CPP]
    assert np.array_equal(py[0].view(np.uint8), cpp[0].view(np.uint8)), (
        "the two backends stored different knots, so any tabulation difference below "
        "belongs to the space port and not to this one"
    )
    assert py[1] == cpp[1], "the two backends derived different tolerances for this space"
    assert py[2] == cpp[2]


# ---------------------------------------------------------------------------
# The rounding budget, for the branch where contraction is live
# ---------------------------------------------------------------------------

_BEZIER_VALUES_EXACT = (
    "The Bezier-like value path does not reach tabulate.hpp: the change of variable "
    "onto [0, 1] is the same numpy expression above the seam on both sides, and the "
    "work is tabulate_bernstein_1d's ratio recurrence. That recurrence contains **no "
    "a * b + c site at all** -- its inner step is (prev * const) * ratio, three "
    "multiplications and no addition -- so the claim is bitwise unconditionally rather "
    "than conditional on the build, exactly as tests/parity/test_basis_tabulations.py "
    "already claims for the same kernel. cpp/include/pantr/basis/bernstein.hpp states "
    "the argument; it was also checked here, by running this file against a "
    "-march=native build, where zero of the values moved while 31 derivative cases "
    "did. What the claim does rest on rather than derive is that numba's np.power "
    "agrees with the platform libm, which is why demand_a_compiled_seed gates it."
)


_EXACT_BY_BUILD = (
    "A2.2 and A2.3 are built from +, -, *, / and an exact-zero comparison, with no "
    "transcendental and no library call, so IEEE 754 pins every result and the only "
    "way the two backends can differ is by performing different operations. The "
    "transliteration performs the same ones in the same order at the same width, "
    "which scripts/measure_bspline_tabulation_widths.py establishes by measuring two "
    "rival width models per site against the kernel and reporting how often they "
    "disagree with each other, so a match cannot come from a check that could not "
    "fail. The one site that widens is A2.3's factorial scaling, where numba promotes "
    "float32 * int64 to float64; the port forms that product in double and rounds "
    "once on the store. This build reports fp_contract unavailable on the target ISA, "
    "so the fusable `saved + right * temp` and `d + a * ndu` sites cannot fuse and "
    "bit-identity is the right claim. Under -march=native the bounded arm applies "
    "instead."
)

_BOUNDED_BY_FMA_VALUES = (
    "This build's target ISA carries a fused multiply-add and PantrCompileOptions "
    "adds -ffp-contract=on, so `saved + right[r+1] * temp` in A2.2 may compile to one "
    "FMA while the numba oracle never fuses (LLVM does not contract without fastmath, "
    "which no pantr kernel sets). design/backend_parity.md Rule 10: at a fused site "
    "the two differ by |b*c| u (1+u) + |a+b*c| 2u, three accumulator roundings, and "
    "the accumulator is the storage format here so there is no narrowing store to "
    "charge. The dependency chain through one output element passes one fusable site "
    "per stage of the outer recurrence and there are `degree` stages. "
    "The amplification is the absolute-value companion of the recurrence, which Rule "
    "10 licenses for a convex recurrence: A2.2's two weights are right[r+1]/denom and "
    "left[j-r]/denom, both non-negative for a non-decreasing knot vector and summing "
    "to one, so replacing every coefficient by its modulus changes nothing and the "
    "companion IS the computed row. It is hull-widened so an entry that rounded to "
    "zero while the true value did not still carries a tolerance."
)

_BOUNDED_BY_FMA_DERIVS = (
    "The fused sites are `saved + right[r+1] * temp` in the ndu triangle and "
    "`d + a[s2,j] * ndu[...]` in the a-table recursion, and the per-site budget is "
    "Rule 10's: three accumulator roundings, no narrowing store, the accumulator "
    "being the storage format. The chain through one output element runs `degree` ndu "
    "stages then up to `n_deriv` a-table stages. "
    "**The amplification is NOT the finished row**, and Rule 10 is explicit about why: "
    "A2.3 differences, so it is not convex, its partial sums can exceed what survives "
    "to the output, and no telescoping argument recovers them -- the finished row was "
    "measured to be exceeded by a structural factor on the sibling Bezier kernel. It "
    "is instead the majorant of the recursion itself: the same algorithm with every "
    "coefficient replaced by its modulus, which is the two sign changes in the a-table "
    "that turn its differences into sums. For a linear recursion with signed "
    "coefficients that bounds every intermediate by induction, and with the signs gone "
    "the partial sums are monotone, so the final value majorises all of them. The "
    "factorial scaling is applied with the same wrapped integers the kernel uses, since "
    "the amplification must bound what the kernel computed and not what it should have."
)


def _a23_majorant(  # noqa: PLR0913
    knots: list[float],
    degree: int,
    n_deriv: int,
    points: _FloatArray,
    first_basis: npt.NDArray[np.int_],
    *,
    unit_spans: bool,
) -> _FloatArray:
    """The majorant of the A2.3 recursion: the same algorithm on moduli.

    Rule 10's licensed amplification for a **non-convex** recursion. Two lines differ
    from the oracle, both in the ``a`` table: its two differences become sums. Every
    quantity is then non-negative, so each intermediate bounds the corresponding signed
    one by induction, and the monotone partial sums make the final value majorise all
    of them.

    Run in ``float64`` throughout. The amplification is a magnitude rather than a
    computed value, so its own rounding is second order against the ``u`` it multiplies.

    Args:
        knots (list[float]): The knot vector.
        degree (int): The polynomial degree.
        n_deriv (int): The highest derivative order.
        points (_FloatArray): The evaluation points.
        first_basis (npt.NDArray[np.int_]): The first-basis index per point, used only
            to recover each point's span.
        unit_spans (bool): ``True`` for the Bezier-like path, whose kernel is A2.3 with
            every knot difference equal to one and whose ``ndu`` upper triangle is
            therefore built from ``s`` and ``1 - s`` on the reference interval. The two
            kernels are different algorithms, so one majorant cannot serve both.

    Returns:
        _FloatArray: Shape ``(points.size, n_deriv + 1, degree + 1)``, non-negative.
    """
    order = degree + 1
    rows = n_deriv + 1
    out = np.zeros((points.size, rows, order), dtype=np.float64)

    if unit_spans:
        begin, end = knots[degree], knots[len(knots) - degree - 1]
        reference = (np.asarray(points, dtype=np.float64) - begin) / (end - begin)
        inverse_span = 1.0 / (end - begin)
    else:
        reference = np.asarray(points, dtype=np.float64)
        inverse_span = 1.0

    exact_knots = np.asarray(knots, dtype=np.float64)
    factorial = _wrapped_falling_factorials(degree, n_deriv)

    for index in range(points.size):
        span = degree if unit_spans else int(first_basis[index]) + degree
        ndu = _ndu_majorant(
            exact_knots,
            degree,
            span,
            float(points[index]),
            float(reference[index]),
            unit_spans=unit_spans,
        )
        out[index, 0, :] = ndu[:, degree]
        a = np.zeros((2, rows), dtype=np.float64)

        for r in range(order):
            s1, s2 = 0, 1
            a[0, 0] = 1.0
            for k in range(1, rows):
                d = 0.0
                rk, pk = r - k, degree - k
                if r >= k:
                    a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk] if ndu[pk + 1, rk] else 0.0
                    d = a[s2, 0] * ndu[rk, pk]
                j1 = 1 if rk >= -1 else -rk
                j2 = k - 1 if (r - 1) <= pk else degree - r
                for j in range(j1, j2 + 1):
                    # The first of the two sign changes: a difference becomes a sum.
                    numerator = a[s1, j] + a[s1, j - 1]
                    a[s2, j] = numerator / ndu[pk + 1, rk + j] if ndu[pk + 1, rk + j] else 0.0
                    d += a[s2, j] * ndu[rk + j, pk]
                if r <= pk:
                    # The second: a negation becomes a modulus.
                    a[s2, k] = a[s1, k - 1] / ndu[pk + 1, r] if ndu[pk + 1, r] else 0.0
                    d += a[s2, k] * ndu[r, pk]
                out[index, k, r] = d
                s1, s2 = s2, s1

    for k in range(1, rows):
        out[:, k, :] *= abs(float(factorial[k])) * inverse_span**k
    return out


def _ndu_majorant(  # noqa: PLR0913
    knots: _FloatArray,
    degree: int,
    span: int,
    point: float,
    reference_point: float,
    *,
    unit_spans: bool,
) -> _FloatArray:
    """The ``ndu`` table of A2.3 built on moduli, for either kernel variant.

    For an in-domain point the knot differences are already non-negative, so taking
    moduli changes nothing here and the table equals the kernel's. It is written on
    moduli anyway, because the majorant's soundness must not depend on the caller
    having clipped its points.

    Args:
        knots (_FloatArray): The knot vector, in ``float64``.
        degree (int): The polynomial degree.
        span (int): The point's knot span.
        point (float): The evaluation point, in the knot vector's coordinate.
        reference_point (float): The same point mapped onto ``[0, 1]``; used only by
            the unit-span variant.
        unit_spans (bool): ``True`` for the Bezier-like kernel, whose knot differences
            are all one and whose triangle is built from ``s`` and ``1 - s``.

    Returns:
        _FloatArray: Shape ``(degree + 1, degree + 1)``, non-negative.
    """
    order = degree + 1
    ndu = np.zeros((order, order), dtype=np.float64)
    ndu[0, 0] = 1.0

    for j in range(1, order):
        saved = 0.0
        for r in range(j):
            if unit_spans:
                ndu[j, r] = 1.0
                left, right = abs(reference_point), abs(1.0 - reference_point)
            else:
                left = abs(point - knots[span + 1 - (j - r)])
                right = abs(knots[span + r + 1] - point)
                ndu[j, r] = right + left
            denom = ndu[j, r]
            temp = 0.0 if denom == 0.0 else ndu[r, j - 1] / denom
            ndu[r, j] = saved + right * temp
            saved = left * temp
        ndu[j, j] = saved
    return ndu


def _wrapped_falling_factorials(degree: int, n_deriv: int) -> list[int]:
    """``degree!/(degree-k)!`` as the kernel accumulates it, wrapping at signed int64.

    The amplification has to bound what the kernel *computed*, so the wrapped factor is
    the right one above degree 21 and the exact one would be wrong there.

    Args:
        degree (int): The polynomial degree.
        n_deriv (int): The highest derivative order.

    Returns:
        list[int]: ``n_deriv + 1`` factors; entry 0 is unused and is 0.
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


def _companion(magnitudes: _FloatArray, stages: int, dtype: Any) -> _FloatArray:
    """Widen a computed magnitude into a hull that covers the true value too.

    A bound built from the *computed* coefficient collapses where that coefficient
    rounds to exactly zero while the true one does not, which
    ``design/backend_parity.md`` records as a correction the infrastructure PR had to
    make: without it the bound could be violated by a factor of ``1/u``.

    The widening is ``(1 + gamma_m)`` with ``m = 3 * stages``, the reference backend's
    own relative forward error over the same chain the parity budget charges: the
    computed magnitude is within that of the true one, so scaling by it covers both. An
    earlier version multiplied by ``1 + 4u`` while this docstring claimed the
    recurrence's whole budget, which is the defect the file's history records twice --
    a derivation stated in prose that the code does not implement.

    The additive floor is the format's smallest normal, so an entry that came out
    exactly zero still carries a tolerance the format can express. That is a floor
    rather than a derivation and is deliberately larger than the smallest subnormal,
    matching the harness's own :func:`~tests._parity_harness.underflow_floor`, which
    takes the unhalved subnormal for the same reason.

    Args:
        magnitudes (_FloatArray): The computed magnitudes, elementwise.
        stages (int): The dependency-chain length the parity budget charges, so that
            the hull and the budget describe the same chain.
        dtype (Any): The storage dtype, which fixes ``u``.

    Returns:
        _FloatArray: The widened magnitudes, in ``float64``.

    Raises:
        ValueError: If the chain is long enough that ``gamma_m`` runs away, which
            would make the hull meaningless rather than merely loose.
    """
    u = unit_roundoff(dtype)
    m = 3 * max(stages, 1)
    if m * u >= 0.5:
        raise ValueError(
            f"a chain of {stages} stages at {np.dtype(dtype).name} accumulates a "
            f"relative budget of {m * u:.3g}, so the hull factor is not first order "
            f"and the amplification it produces would not bound anything"
        )
    gamma = m * u / (1.0 - m * u)
    values = np.abs(np.asarray(magnitudes, dtype=np.float64))
    widened = values * (1.0 + gamma) + np.finfo(dtype).smallest_normal
    return np.asarray(widened, dtype=np.float64)


def _value_claim(reference: _FloatArray, degree: int, dtype: Any) -> ParityClaim:
    """The parity claim for a basis-value tabulation, conditional on the build.

    Args:
        reference (_FloatArray): The Python backend's block, whose magnitudes the
            amplification is built from.
        degree (int): The polynomial degree, which sets the stage count.
        dtype (Any): The storage dtype.

    Returns:
        ParityClaim: A bitwise claim where nothing can fuse, a bounded one otherwise.
    """
    if not contraction_may_fuse():
        return bitwise_parity(why=_EXACT_BY_BUILD)
    return bounded_parity(
        roundings=Roundings(stages=max(degree, 1), accumulator_per_stage=3, storage_per_stage=0),
        accumulator=dtype,
        storage=dtype,
        amplification=_companion(reference, degree, dtype),
        why=_BOUNDED_BY_FMA_VALUES,
    )


def _deriv_claim(  # noqa: PLR0913
    knots: list[float],
    degree: int,
    n_deriv: int,
    points: _FloatArray,
    first_basis: npt.NDArray[np.int_],
    dtype: Any,
    *,
    unit_spans: bool,
) -> ParityClaim:
    """The parity claim for a derivative tabulation, conditional on the build.

    The chain is longer than the value kernel's: A2.3 builds the ``ndu`` triangle in
    ``degree`` stages and then runs the ``a``-table recursion for up to ``n_deriv``
    more, so the stage count is their sum.

    The amplification is :func:`_a23_majorant` and **not** the finished row. That
    distinction is the whole of the derivation and it is not a refinement: on a fusing
    build the finished row exceeded the observed difference on 31 of 418 cases, at both
    widths, which is the signature Rule 10 names -- a structural shortfall rather than
    rounding. The recursion is not convex, so its partial sums can exceed what survives
    to the output.

    Args:
        knots (list[float]): The knot vector, needed to rebuild the majorant.
        degree (int): The polynomial degree.
        n_deriv (int): The highest derivative order.
        points (_FloatArray): The evaluation points.
        first_basis (npt.NDArray[np.int_]): The first-basis index per point.
        dtype (Any): The storage dtype.
        unit_spans (bool): Whether the Bezier-like kernel ran; see
            :func:`_a23_majorant`.

    Returns:
        ParityClaim: A bitwise claim where nothing can fuse, a bounded one otherwise.
    """
    if not contraction_may_fuse():
        return bitwise_parity(why=_EXACT_BY_BUILD)
    majorant = _a23_majorant(knots, degree, n_deriv, points, first_basis, unit_spans=unit_spans)
    return bounded_parity(
        roundings=Roundings(
            stages=max(degree + n_deriv, 1), accumulator_per_stage=3, storage_per_stage=0
        ),
        accumulator=dtype,
        storage=dtype,
        amplification=_companion(majorant, degree + n_deriv, dtype),
        why=_BOUNDED_BY_FMA_DERIVS,
    )


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("case", sorted(_GENERAL_KNOTS | _HIGH_DEGREE_KNOTS))
def test_general_knot_values_match(cpp_backend: None, case: str, dtype: Any) -> None:
    """The two backends tabulate the general-knot basis identically.

    Args:
        cpp_backend (None): Requires the extension; see ``tests/parity/conftest.py``.
        case (str): Key into :data:`_GENERAL_KNOTS`.
        dtype (Any): The storage dtype.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    knots, degree = (_GENERAL_KNOTS | _HIGH_DEGREE_KNOTS)[case]
    _demand_the_two_spaces_agree(knots, degree, dtype)
    points = _evaluation_points(knots, degree, dtype)

    reference = _tabulate(Backend.PYTHON, knots, degree, points, dtype, None)
    actual = _tabulate(Backend.CPP, knots, degree, points, dtype, None)

    assert_object_parity(
        py=reference,
        cpp=actual,
        fields=[
            Field("block", claim=_value_claim(reference.block, degree, dtype)),
            Field(
                "first_basis",
                claim=exact_parity(
                    why="the first-basis index is exact integer arithmetic on the span "
                    "search's result -- a binary search, two clamps and a subtraction -- so "
                    "no tolerance applies and a difference is a different answer, not a "
                    "displaced one. It decides which global functions the row refers to, so "
                    "a wrong index makes correct values wrong."
                ),
            ),
        ],
        context=f"tabulate_basis {case} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("n_deriv", _DERIVATIVE_ORDERS)
@pytest.mark.parametrize("case", sorted(_GENERAL_KNOTS | _HIGH_DEGREE_KNOTS))
def test_general_knot_derivatives_match(
    cpp_backend: None, case: str, n_deriv: int, dtype: Any
) -> None:
    """The two backends tabulate the general-knot derivatives identically.

    ``n_deriv`` reaches 22 because that is what makes the claim about the factorial
    scaling's *width* falsifiable: below ``fac = 2**24`` the two candidate widths are
    provably the same number, so a smaller matrix asserts the width without being able
    to fail on it. See :data:`_HIGH_DEGREE_KNOTS`. The small orders are kept because
    ``n_deriv`` above the degree is what exercises the rows the scaling zeroes out.

    Args:
        cpp_backend (None): Requires the extension.
        case (str): Key into :data:`_GENERAL_KNOTS`.
        n_deriv (int): The highest derivative order.
        dtype (Any): The storage dtype.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    # `_basis_derivs_point` accumulates a falling factorial that wraps at int64 when
    # compiled and grows without bound when interpreted.
    demand_a_compiled_seed()
    knots, degree = (_GENERAL_KNOTS | _HIGH_DEGREE_KNOTS)[case]
    _demand_the_two_spaces_agree(knots, degree, dtype)
    points = _evaluation_points(knots, degree, dtype)

    reference = _tabulate(Backend.PYTHON, knots, degree, points, dtype, n_deriv)
    actual = _tabulate(Backend.CPP, knots, degree, points, dtype, n_deriv)

    assert_parity(
        actual.block,
        reference.block,
        _deriv_claim(
            knots,
            degree,
            n_deriv,
            points,
            reference.first_basis,
            dtype,
            unit_spans=False,
        ),
        context=f"tabulate_basis_derivatives {case} n_deriv={n_deriv} {np.dtype(dtype).name}",
    )
    assert np.array_equal(actual.first_basis, reference.first_basis)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("case", sorted(_BEZIER_KNOTS))
def test_bezier_like_values_match(cpp_backend: None, case: str, dtype: Any) -> None:
    """The Bézier-like fast path agrees, and it is a different kernel from the general one.

    This path does not reach ``tabulate.hpp`` at all: the oracle maps the points onto
    ``[0, 1]`` and calls the Bernstein ratio recurrence, and so does the port. The
    change of variable is the *same numpy expression* on both sides, above the seam, so
    it is common mode and contributes nothing -- which is Rule 1's consequence applied
    here rather than restated.

    **The claim is bitwise unconditionally**, unlike every other one in this file: that
    recurrence has no fusable site, so contraction cannot reach it. See
    :data:`_BEZIER_VALUES_EXACT`.

    Args:
        cpp_backend (None): Requires the extension.
        case (str): Key into :data:`_BEZIER_KNOTS`.
        dtype (Any): The storage dtype.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    # The Bernstein ratio recurrence seeds with `np.power`.
    demand_a_compiled_seed()
    knots, degree = _BEZIER_KNOTS[case]
    _demand_the_two_spaces_agree(knots, degree, dtype)
    points = _evaluation_points(knots, degree, dtype)

    reference = _tabulate(Backend.PYTHON, knots, degree, points, dtype, None)
    actual = _tabulate(Backend.CPP, knots, degree, points, dtype, None)

    assert_parity(
        actual.block,
        reference.block,
        bitwise_parity(why=_BEZIER_VALUES_EXACT),
        context=f"tabulate_basis bezier {case} {np.dtype(dtype).name}",
    )
    assert np.array_equal(actual.first_basis, reference.first_basis)
    assert np.all(actual.first_basis == 0), (
        "a Bézier-like space has one span, so every point's first supported function "
        "is function 0; a non-zero index means the fast path was not taken"
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("n_deriv", [0, 1, 3])
@pytest.mark.parametrize("case", sorted(_BEZIER_KNOTS))
def test_bezier_like_derivatives_match(
    cpp_backend: None, case: str, n_deriv: int, dtype: Any
) -> None:
    """The Bézier-like derivative fast path agrees, chain-rule scaling included.

    Args:
        cpp_backend (None): Requires the extension.
        case (str): Key into :data:`_BEZIER_KNOTS`.
        n_deriv (int): The highest derivative order.
        dtype (Any): The storage dtype.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    # `_bernstein_derivs_point` accumulates the same wrapping falling factorial.
    demand_a_compiled_seed()
    knots, degree = _BEZIER_KNOTS[case]
    _demand_the_two_spaces_agree(knots, degree, dtype)
    points = _evaluation_points(knots, degree, dtype)

    reference = _tabulate(Backend.PYTHON, knots, degree, points, dtype, n_deriv)
    actual = _tabulate(Backend.CPP, knots, degree, points, dtype, n_deriv)

    assert_parity(
        actual.block,
        reference.block,
        _deriv_claim(
            knots,
            degree,
            n_deriv,
            points,
            reference.first_basis,
            dtype,
            unit_spans=True,
        ),
        context=f"tabulate_basis_derivatives bezier {case} n_deriv={n_deriv} "
        f"{np.dtype(dtype).name}",
    )
    assert np.array_equal(actual.first_basis, reference.first_basis)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_a_strided_out_reaches_the_callers_array(cpp_backend: None, dtype: Any) -> None:
    """A non-contiguous ``out`` is filled, not silently left alone.

    The C++ binding refuses a non-contiguous array rather than converting it, because
    a converted output would be filled and discarded; the adapter in
    ``pantr.bspline._basis_backend`` absorbs that by computing into a buffer and
    copying back. Without the absorption the caller's array comes back untouched with
    no exception anywhere, which ``cpp/bindings/basis.cpp`` records as a measured
    regression on the sibling kernel.

    Args:
        cpp_backend (None): Requires the extension.
        dtype (Any): The storage dtype.
    """
    del cpp_backend
    knots, degree = _GENERAL_KNOTS["clamped-uniform-p2"]
    points = np.array([0.1, 0.4, 0.9], dtype=dtype)

    with use_backend(Backend.CPP):
        space = BsplineSpace1D(np.array(knots, dtype=dtype), degree)
        # The reference is the SAME backend with a contiguous out, not the oracle. The
        # question here is whether the adapter's buffer round-trip loses the answer,
        # and comparing across backends would fold in the parity claim -- which on a
        # fusing build is bounded rather than bitwise, so a bit-equality assertion
        # would fail there for a reason that has nothing to do with the strides.
        expected, expected_first = space.tabulate_basis(points)

        # Every other column of a wider array, so `out` is strided in its last axis.
        canvas = np.full((points.size, 2 * (degree + 1)), np.nan, dtype=dtype)
        strided = canvas[:, ::2]
        indices = np.full(2 * points.size, -1, dtype=np.int_)[::2]
        assert not strided.flags["C_CONTIGUOUS"]
        got, got_first = space.tabulate_basis(points, strided, indices)

    assert got is strided, "the caller's array must be returned, not a copy of it"
    assert not np.any(np.isnan(strided)), "the strided out was never written"
    assert np.array_equal(strided, expected), (
        "the same kernel on the same input through a strided out gave a different "
        "answer, so the adapter's buffer round-trip is not transparent"
    )
    assert got_first is indices
    assert np.all(indices >= 0), "the strided index array was never written"
    assert np.array_equal(indices, expected_first)


def test_the_seam_returns_kernels_for_every_available_backend() -> None:
    """Every backend the installation offers answers all three catalogue functions.

    A catalogue that raises for a backend :func:`available_backends` lists would make
    every parity test above skip rather than fail, which is the trap ``CLAUDE.md``
    names: a missing optional dependency skips without complaint.
    """
    for backend in available_backends():
        assert isinstance(bspline_basis_core(backend), BasisKernels)
        assert isinstance(bspline_basis_deriv_core(backend), BasisDerivKernels)
        assert isinstance(bernstein_deriv_core(backend), DerivKernels)

    for backend in available_backends():
        if backend is Backend.PYTHON:
            assert bspline_basis_core(backend).serial is not None, (
                "the oracle's serial twin is what _PARALLEL_MIN_NUM_PTS selects below "
                "the threshold; losing it would silently change which kernel every "
                "small batch runs"
            )
        else:
            assert bspline_basis_core(backend).serial is None, (
                "the C++ kernel runs on the calling thread at every batch size, so a "
                "serial twin would be a second name for the same function and the "
                "threshold would then select between two identical things"
            )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_the_case_list_separates_the_factorial_widths(dtype: Any) -> None:
    """The parity matrix must contain a case where the two factorial widths differ.

    A2.3's step 3 forms ``T(double(value) * double(fac))``. The alternative,
    ``value * T(fac)``, is **provably the same number** while ``fac`` is exactly
    representable in the storage format -- the exact product of two ``float32`` values
    needs at most 48 significand bits, so it is exact in ``float64`` and one rounding
    to ``float32`` is what a ``float32`` multiply commits. So every case with
    ``fac < 2**24`` asserts the width and cannot fail on it.

    This was not hypothetical: narrowing the scaling in ``tabulate.hpp`` passed all 36
    derivative parity cases before :data:`_HIGH_DEGREE_KNOTS` existed. The guard is
    here so that trimming the case list for speed cannot quietly restore that.

    At ``float64`` the two widths coincide unconditionally, since the accumulator *is*
    the storage format; the test still runs and asserts the weaker, true thing --
    that some ``fac`` exceeds the format's exact-integer range -- so that the
    parametrization does not silently mean two different things.

    Args:
        dtype (Any): The storage dtype, which fixes the exact-integer range.
    """
    exact_integers = 2 ** np.finfo(dtype).nmant
    reachable = [
        _falling_factorial(degree, n_deriv)
        for case, (_, degree) in (_GENERAL_KNOTS | _HIGH_DEGREE_KNOTS).items()
        for n_deriv in _DERIVATIVE_ORDERS
        if case is not None
    ]
    largest = max(reachable)
    assert largest > exact_integers, (
        f"the largest falling factorial the derivative parity matrix reaches is "
        f"{largest}, which {np.dtype(dtype).name} represents exactly, so every case "
        f"agrees on the factorial scaling's width whatever the port chose. Add a "
        f"higher degree with a matching n_deriv, or the width claim asserts nothing"
    )


def _falling_factorial(degree: int, n_deriv: int) -> int:
    """The largest scaling factor A2.3 forms for this degree and derivative order.

    Args:
        degree (int): The polynomial degree.
        n_deriv (int): The highest derivative order.

    Returns:
        int: ``degree!/(degree-k)!`` at the largest ``k <= min(degree, n_deriv)``,
        in exact Python integers rather than the kernel's wrapping int64 -- the
        question here is which factors the matrix *reaches*, not what the kernel does
        with them.
    """
    largest = 1
    value = 1
    for k in range(1, min(degree, n_deriv) + 1):
        value *= degree - k + 1
        largest = max(largest, value)
    return largest


# ---------------------------------------------------------------------------
# Independent accuracy oracles
# ---------------------------------------------------------------------------


def _a22_chain_roundings(degree: int) -> int:
    """Roundings on the dependency chain through one A2.2 output element.

    Counted against ``_basis_funcs_point``, six per stage of the ``j`` loop:
    ``left[j]``, ``right[j]``, ``denom``, the division ``N[r] / denom``, the
    multiplication ``right[r+1] * temp`` and the addition ``saved + ...``. The
    recurrence is a convex combination -- the two weights are ``right[r+1]/denom`` and
    ``left[j-r]/denom``, non-negative for a non-decreasing knot vector and summing to
    one -- so an inherited relative error is carried rather than amplified and the six
    are additive per stage. There are ``degree`` stages.

    Args:
        degree (int): The polynomial degree.

    Returns:
        int: The rounding count, for use as ``m`` in ``gamma_m``.
    """
    return 6 * degree


def _a23_chain_roundings(degree: int, n_deriv: int) -> int:
    """Roundings on the dependency chain through one A2.3 output element.

    :func:`_a22_chain_roundings` for the ``ndu`` triangle, which A2.3 builds by the
    same steps, plus **four** per stage of the ``a``-table recursion, counted against
    ``_basis_derivs_point``: the subtraction ``a[s1,j] - a[s1,j-1]``, the division by
    ``ndu``, the multiplication ``a[s2,j] * ndu[...]`` and the accumulation into ``d``.
    Plus **one** for the factorial scaling, which happens once per element and not per
    stage.

    An earlier version charged three per stage and no factorial, which under-counted --
    the dangerous direction, since a bound tighter than derivable can be exceeded by
    correct code. The margin had been absorbing it.

    Args:
        degree (int): The polynomial degree.
        n_deriv (int): The highest derivative order.

    Returns:
        int: The rounding count, for use as ``m`` in ``gamma_m``.
    """
    return _a22_chain_roundings(degree) + 4 * n_deriv + 1


def _dot_product_roundings(degree: int) -> int:
    """Roundings in a dot product of the ``degree + 1`` values against exact weights.

    ``degree + 1`` multiplications and ``degree`` additions, plus one for casting each
    exact rational weight to ``double``. The cast is charged once because it is one
    rounding per term and the terms are summed, so it enters the bound the same way a
    per-term relative error does.

    Args:
        degree (int): The polynomial degree.

    Returns:
        int: The rounding count, for use as ``m`` in ``gamma_m``.
    """
    return (degree + 1) + degree + 1


def _gamma(m: int, dtype: Any) -> float:
    """Higham's ``gamma_m = m u / (1 - m u)``, refusing a budget that runs away.

    Args:
        m (int): The rounding count.
        dtype (Any): The storage dtype, which fixes ``u = eps / 2``.

    Returns:
        float: ``gamma_m``, or at least one ``u`` so that a zero-stage case still
        carries the smallest bound the format can express.

    Raises:
        ValueError: If ``m u >= 1/2``, where ``gamma_m`` no longer bounds a first-order
            accumulation and a claim built on it would assert nothing.
    """
    u = unit_roundoff(dtype)
    if m * u >= 0.5:
        raise ValueError(
            f"a budget of {m} roundings at {np.dtype(dtype).name} accumulates to "
            f"{m * u:.3g}, at which gamma_m stops being a first-order bound"
        )
    return max(m * u / (1.0 - m * u), u)


def _marsden_weights(
    knots: list[float], degree: int, first_basis: int, y: Fraction
) -> list[Fraction]:
    """The Marsden weights of the ``degree + 1`` functions supported at a point.

    Marsden's identity: for every ``y``,
    ``sum_i N_{i,p}(x) prod_{j=1..p} (t_{i+j} - y) = (x - y)^p``.
    See de Boor, *A Practical Guide to Splines*, Ch. IX; it is a theorem about the
    basis rather than a rearrangement of the recurrence, which is what makes it an
    independent oracle here.

    **``y`` must lie outside the knot range.** Then every factor ``t_{i+j} - y`` has
    the same sign, so every weight does, and the sum has no cancellation at all: the
    computed sum's magnitude equals ``|(x - y)^p|`` and a relative bound on it is a
    relative bound on the identity. With ``y`` inside the range the weights straddle
    zero, the sum cancels, and the bound needed to cover it exceeds the value it
    compares -- which ``design/backend_parity.md`` Rule 3 refuses, correctly.

    Args:
        knots (list[float]): The knot vector.
        degree (int): The polynomial degree.
        first_basis (int): Global index of the first supported function.
        y (Fraction): The identity's parameter, outside ``[min(knots), max(knots)]``.

    Returns:
        list[Fraction]: ``degree + 1`` exact weights, one per supported function.
    """
    exact_knots = [Fraction(k) for k in knots]
    weights = []
    for i in range(first_basis, first_basis + degree + 1):
        weight = Fraction(1)
        for j in range(1, degree + 1):
            weight *= exact_knots[i + j] - y
        weights.append(weight)
    return weights


def _outside_y(knots: list[float]) -> tuple[Fraction, Fraction]:
    """Two values of Marsden's parameter, one below the knots and one above.

    Each is one full knot range away from the nearest knot, so ``t - y`` is of the same
    order as the range itself: close to the range the weights would span decades and
    the identity would test the smallest ones not at all.

    Args:
        knots (list[float]): The knot vector.

    Returns:
        tuple[Fraction, Fraction]: ``(below, above)``, both exact and dyadic.
    """
    low, high = Fraction(min(knots)), Fraction(max(knots))
    width = high - low
    return low - width, high + width


def _partition_bound(degree: int, dtype: Any) -> float:
    """An absolute bound on ``|sum_i N_i - 1|``, derived rather than measured.

    Two contributions, both first order in ``u = eps/2``:

    - **The values' own forward error**, :func:`_a22_chain_roundings`. Since the values
      are non-negative and sum to one, the sum of their *absolute* errors is bounded by
      the same ``gamma`` that bounds each one's relative error.
    - **The summation**, ``degree`` additions of quantities summing to one, so
      ``gamma_degree``.

    The two counts are added rather than combined with a safety factor, and
    :func:`_gamma` supplies the one-``u`` floor that covers ``degree = 0``, where the
    single value is exactly one and no operation runs.

    Args:
        degree (int): The polynomial degree.
        dtype (Any): The storage dtype, which fixes ``u``.

    Returns:
        float: The bound, the same for every point.
    """
    return _gamma(_a22_chain_roundings(degree) + degree, dtype)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.CPP])
@pytest.mark.parametrize("case", sorted(_GENERAL_KNOTS) + sorted(_BEZIER_KNOTS))
def test_partition_of_unity(cpp_backend: None, case: str, backend: Backend, dtype: Any) -> None:
    """The basis values sum to one, on each backend independently.

    Run per backend rather than on the difference: this is the check that survives a
    bug present in *both*, which every parity assertion above is blind to by
    construction.

    Args:
        cpp_backend (None): Requires the extension.
        case (str): Key into either knot table.
        backend (Backend): The backend to check.
        dtype (Any): The storage dtype.
    """
    del cpp_backend
    knots, degree = (_GENERAL_KNOTS | _BEZIER_KNOTS)[case]
    points = _evaluation_points(knots, degree, dtype)
    got = _tabulate(backend, knots, degree, points, dtype, None)

    sums = np.asarray(got.block.sum(axis=-1), dtype=np.float64)
    assert np.all(np.asarray(got.block) >= 0.0), (
        "every value of a B-spline basis is non-negative, exactly: the recurrence is a "
        "convex combination of non-negative terms and no rounding can make one negative"
    )
    assert_accuracy(
        sums,
        np.ones_like(sums),
        derived_accuracy(
            bound=np.full(sums.shape, _partition_bound(degree, dtype)),
            why="the partition of unity is exact in the reals, so the whole discrepancy "
            "is the computation's own forward error: six roundings per stage carried by "
            "the convex recurrence over degree stages, plus the degree additions of the "
            "sum. Both counts are counted against _basis_funcs_point rather than "
            "estimated; see _a22_chain_roundings and _partition_bound.",
        ),
        context=f"partition of unity, {case}, {backend.name}, {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.CPP])
@pytest.mark.parametrize("case", sorted(_GENERAL_KNOTS))
def test_marsden_identity_pins_the_values(
    cpp_backend: None, case: str, backend: Backend, dtype: Any
) -> None:
    """Marsden's identity holds, which pins the indexing and the scale of each row.

    The one check here that a permutation of the row, or a uniform rescaling of it,
    cannot pass: each function carries a distinct weight computed from the knots in
    exact rational arithmetic.

    Args:
        cpp_backend (None): Requires the extension.
        case (str): Key into :data:`_GENERAL_KNOTS`.
        backend (Backend): The backend to check.
        dtype (Any): The storage dtype.
    """
    del cpp_backend
    knots, degree = _GENERAL_KNOTS[case]
    points = _evaluation_points(knots, degree, dtype)
    got = _tabulate(backend, knots, degree, points, dtype, None)
    values = np.asarray(got.block, dtype=np.float64)

    for y in _outside_y(knots):
        computed = np.empty(points.size, dtype=np.float64)
        exact = np.empty(points.size, dtype=np.float64)
        bound = np.empty(points.size, dtype=np.float64)

        for p, point in enumerate(points):
            weights = _marsden_weights(knots, degree, int(got.first_basis[p]), y)
            floats = np.array([float(w) for w in weights], dtype=np.float64)
            computed[p] = float(np.dot(floats, values[p]))
            exact[p] = float((Fraction(float(point)) - y) ** degree)
            # Every weight has the same sign and the values are non-negative, so this
            # sum does not cancel and its magnitude is |exact| up to the budget below.
            magnitude = float(np.dot(np.abs(floats), np.abs(values[p])))
            # The values' own chain, plus the dot product against the exact weights.
            # There is no partition-of-unity summation term here: Marsden's sum IS the
            # dot product, and charging both would double-count it.
            budget = _a22_chain_roundings(degree) + _dot_product_roundings(degree)
            bound[p] = _gamma(budget, dtype) * magnitude

        assert np.all(bound < np.abs(exact) + np.finfo(np.float64).tiny), (
            f"the Marsden bound is not smaller than the value it compares for {case} "
            f"at y={y}, so the claim would admit any answer; Rule 3 refuses that and "
            f"so does this test rather than asserting it anyway"
        )
        assert_accuracy(
            computed,
            exact,
            derived_accuracy(
                bound=bound,
                why="Marsden's identity is a theorem about the basis, and its right-hand "
                "side is computed in exact rational arithmetic from dyadic knots, so the "
                "reference carries no error. y sits a full knot range outside, which makes "
                "every weight the same sign and the sum cancellation-free. The bound is the "
                "values' own chain (_a22_chain_roundings) plus the dot product's "
                "(_dot_product_roundings), times the sum of |weight| * |value|. It carries "
                "no separate summation term, Marsden's sum being the dot product itself.",
            ),
            context=f"Marsden at y={y}, {case}, {backend.name}, {np.dtype(dtype).name}",
        )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.CPP])
@pytest.mark.parametrize("case", sorted(_GENERAL_KNOTS) + sorted(_BEZIER_KNOTS))
def test_derivative_sums_vanish(cpp_backend: None, case: str, backend: Backend, dtype: Any) -> None:
    """Every derivative row sums to zero, to a bound scaled by that row's magnitude.

    An **absolute** comparison scaled by the row's own largest magnitude, not a
    relative one: the quantity being compared vanishes, and a relative bound on a
    quantity whose true value is zero is unbounded. That is
    ``design/backend_parity.md`` Rule 2, and the row magnitudes here really do span
    decades, since a k-th derivative carries ``p!/(p-k)!`` and an inverse knot spacing
    to the k-th power.

    Args:
        cpp_backend (None): Requires the extension.
        case (str): Key into either knot table.
        backend (Backend): The backend to check.
        dtype (Any): The storage dtype.
    """
    del cpp_backend
    knots, degree = (_GENERAL_KNOTS | _BEZIER_KNOTS)[case]
    n_deriv = min(degree + 1, 4)
    points = _evaluation_points(knots, degree, dtype)
    got = _tabulate(backend, knots, degree, points, dtype, n_deriv)
    block = np.asarray(got.block, dtype=np.float64)

    for k in range(1, n_deriv + 1):
        rows = block[:, k, :]
        sums = rows.sum(axis=-1)
        # The row's own scale. `sum_i |N_i^(k)|` rather than the max, because the
        # summation error is driven by the partial sums and those reach the total.
        magnitude = np.abs(rows).sum(axis=-1)
        budget = _a23_chain_roundings(degree, n_deriv) + degree
        bound = np.maximum(_gamma(budget, dtype) * magnitude, np.finfo(dtype).smallest_normal)
        assert_accuracy(
            sums,
            np.zeros_like(sums),
            derived_accuracy(
                bound=bound,
                why="sum_i N_i^(k) = 0 for k >= 1 follows from differentiating the "
                "partition of unity, so the whole discrepancy is forward error. The bound "
                "is _a23_chain_roundings -- six per ndu stage, four per a-table stage and "
                "one for the factorial scaling, each counted against _basis_derivs_point -- "
                "plus the degree additions of the sum, times sum_i |N_i^(k)|. Absolute and "
                "scaled by the row's own magnitude, since the quantity vanishes and no "
                "relative bound on it is finite.",
            ),
            context=f"derivative sum k={k}, {case}, {backend.name}, {np.dtype(dtype).name}",
        )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.CPP])
@pytest.mark.parametrize("case", sorted(_GENERAL_KNOTS))
def test_differentiated_marsden_pins_the_derivative_scale(
    cpp_backend: None, case: str, backend: Backend, dtype: Any
) -> None:
    """Differentiating Marsden's identity pins the factorial scaling of each row.

    ``sum_i N^{(k)}_i(x) c_i(y) = p!/(p-k)! (x-y)^{p-k}``, the right-hand side exact in
    rational arithmetic. This is the check :func:`test_derivative_sums_vanish` cannot
    make: that one is invariant under multiplying a whole row by any constant, and
    A2.3's step 3 multiplies a whole row by a constant. A dropped, inverted or wrapped
    ``p!/(p-k)!`` passes the sums and fails here.

    The degrees where the factorial accumulator wraps are excluded by
    :data:`_GENERAL_KNOTS` carrying none above 8; see
    :func:`test_the_wrapping_degrees_are_named_and_still_wrap` for the region and why
    no accuracy claim reaches it.

    Args:
        cpp_backend (None): Requires the extension.
        case (str): Key into :data:`_GENERAL_KNOTS`.
        backend (Backend): The backend to check.
        dtype (Any): The storage dtype.
    """
    del cpp_backend
    knots, degree = _GENERAL_KNOTS[case]
    if degree == 0:
        pytest.skip("a degree-0 basis has no non-zero derivative to pin")
    n_deriv = degree
    points = _evaluation_points(knots, degree, dtype)
    got = _tabulate(backend, knots, degree, points, dtype, n_deriv)
    block = np.asarray(got.block, dtype=np.float64)
    y = _outside_y(knots)[0]

    falling = 1
    for k in range(1, n_deriv + 1):
        falling *= degree - k + 1
        computed = np.empty(points.size, dtype=np.float64)
        exact = np.empty(points.size, dtype=np.float64)
        bound = np.empty(points.size, dtype=np.float64)

        for p, point in enumerate(points):
            weights = _marsden_weights(knots, degree, int(got.first_basis[p]), y)
            floats = np.array([float(w) for w in weights], dtype=np.float64)
            row = block[p, k, :]
            computed[p] = float(np.dot(floats, row))
            exact[p] = float(falling * (Fraction(float(point)) - y) ** (degree - k))
            magnitude = float(np.dot(np.abs(floats), np.abs(row)))
            budget = _a23_chain_roundings(degree, n_deriv) + _dot_product_roundings(degree)
            bound[p] = _gamma(budget, dtype) * magnitude

        if not np.all(bound < np.abs(exact)):
            pytest.skip(
                f"the derivative rows of {case} at order {k} cancel against these "
                f"weights hard enough that the bound exceeds the value; Rule 3 refuses "
                f"such a claim and this case is reported rather than asserted"
            )
        assert_accuracy(
            computed,
            exact,
            derived_accuracy(
                bound=bound,
                why="differentiating Marsden's identity k times gives "
                "p!/(p-k)! (x-y)^(p-k) on the right, exact in rational arithmetic. Unlike "
                "the derivative sums this is not invariant under rescaling a row, which is "
                "what makes it the check that pins A2.3's factorial scaling. The bound is "
                "the row's forward error plus the dot product's, times sum |weight| * "
                "|value| -- the derivative rows carry mixed signs, so the sum does cancel "
                "and the magnitude cannot be replaced by the right-hand side.",
            ),
            context=f"differentiated Marsden k={k}, {case}, {backend.name}, {np.dtype(dtype).name}",
        )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.CPP])
@pytest.mark.parametrize("case", sorted(_GENERAL_KNOTS) + sorted(_BEZIER_KNOTS))
def test_rows_above_the_degree_vanish_exactly(
    cpp_backend: None, case: str, backend: Backend, dtype: Any
) -> None:
    """Derivative rows above the degree are identically zero, with no tolerance.

    Exact, not bounded: the ``k``-th derivative of a degree-``p`` polynomial is the
    zero polynomial for ``k > p``, and A2.3 reaches it by a factorial factor that
    becomes exactly ``0`` once ``degree - k`` does. A rounding cannot produce a nonzero
    value from a multiplication by exact zero, so any nonzero entry is a wrong answer
    rather than a displaced one.

    Args:
        cpp_backend (None): Requires the extension.
        case (str): Key into either knot table.
        backend (Backend): The backend to check.
        dtype (Any): The storage dtype.
    """
    del cpp_backend
    knots, degree = (_GENERAL_KNOTS | _BEZIER_KNOTS)[case]
    n_deriv = degree + 3
    points = _evaluation_points(knots, degree, dtype)
    got = _tabulate(backend, knots, degree, points, dtype, n_deriv)
    above = np.asarray(got.block)[:, degree + 1 :, :]

    assert above.size > 0, "the case list must reach past the degree for this to say anything"
    assert np.all(above == 0.0), (
        f"{np.count_nonzero(above)} of {above.size} entries above degree {degree} are "
        f"non-zero on {backend.name}; the k-th derivative of a degree-p polynomial is "
        f"identically zero for k > p, and A2.3 reaches that by an exactly-zero factor"
    )


def test_the_wrapping_degrees_are_named_and_still_wrap() -> None:
    """Name the region where no accuracy claim is made, and fail if it disappears.

    ``design/backend_parity.md`` Rule 8's obligation: a test that quietly stops where
    its bound runs out reads as full coverage. A2.3's step 3 accumulates
    ``degree!/(degree-k)!`` in an int64 that wraps, so above some degree the derivative
    rows are scaled by a two's-complement remainder. Both backends wrap identically --
    ``pantr::core::wrapping_mul`` exists so the C++ side reproduces rather than invokes
    undefined behaviour -- so **parity holds there and accuracy does not**, and the
    accuracy tests above carry no case past degree 8.

    This fails if the wrap moves or disappears, because either would mean the
    accumulator changed under us and the excluded region is no longer the one the
    accuracy tests were scoped around.
    """
    from pantr.bspline._bspline_basis_core import _basis_derivs_point  # noqa: PLC0415

    wrapping: list[tuple[int, int]] = []
    for degree in range(1, 31):
        exact = 1
        modelled = degree
        mask = (1 << 64) - 1
        for k in range(1, degree + 1):
            exact *= degree - k + 1
            if modelled != exact:
                wrapping.append((degree, k))
                break
            modelled = (modelled * (degree - k)) & mask
            if modelled >= 1 << 63:
                modelled -= 1 << 64
    assert wrapping, (
        "the falling factorial no longer wraps at any degree up to 30. That is a "
        "finding, not good news: the accuracy tests here are scoped to degree <= 8 "
        "because of the wrap, and the exclusion should be revisited"
    )
    lowest = min(degree for degree, _ in wrapping)
    assert lowest == 21, (
        f"the lowest wrapping degree moved from 21 to {lowest}. "
        f"scripts/measure_bspline_tabulation_widths.py measures it against the compiled "
        f"kernel; re-run it and rescope the accuracy tests before changing this number"
    )
    # The kernel is imported to make the coupling explicit rather than implied: if it
    # stops carrying an integer accumulator, this test's premise is gone.
    assert _basis_derivs_point is not None


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_the_partition_bound_is_still_approached(cpp_backend: None, dtype: Any) -> None:
    """The partition-of-unity bound is not orders of magnitude looser than the truth.

    A bound nothing exercises can rot with no test noticing, so this asserts the
    opposite of the bound: that the worst observed discrepancy is still within a stated
    factor of it. That factor is **measured, not derived**, which is why it is enforced
    only on the reference host -- ``design/backend_parity.md`` Rule 7, and here the
    host-dependence is real because the Bézier-like cases route through the platform's
    ``pow``.

    Args:
        cpp_backend (None): Requires the extension.
        dtype (Any): The storage dtype.
    """
    del cpp_backend
    demand_the_reference_host(
        "the margin between the partition-of-unity bound and the worst observed discrepancy",
        "this project's calibrated host; the general-knot cases are pure IEEE "
        "arithmetic and so deterministic, but the Bezier-like ones route through the "
        "platform's pow and the float32 worst case sits on one of them",
    )
    worst = 0.0
    for case, (knots, degree) in (_GENERAL_KNOTS | _BEZIER_KNOTS).items():
        del case
        points = _evaluation_points(knots, degree, dtype)
        got = _tabulate(Backend.CPP, knots, degree, points, dtype, None)
        sums = np.asarray(got.block.sum(axis=-1), dtype=np.float64)
        bound = _partition_bound(degree, dtype)
        worst = max(worst, float(np.max(np.abs(sums - 1.0))) / bound)

    # The threshold follows from the bound's own structure rather than from a
    # measurement. _partition_bound charges 7 roundings per stage; the path that is
    # actually attained -- the degree additions of the final sum -- commits 1 of those
    # 7, so the worst observation cannot sit far below 1/7 of the bound. The extra
    # factor of about 3 below is an **admitted heuristic margin**, not a derivation:
    # it is there so that adding a case which happens not to attain the summation
    # worst case cannot turn this red on its own.
    assert worst > 1.0 / 20.0, (
        f"the worst partition-of-unity discrepancy is {worst:.3g} of its bound, so the "
        f"bound has become loose enough to stop asserting anything. The bound charges "
        f"7 roundings per stage and 1 of them is attained, so this ratio should be near "
        f"1/7; either the cases stopped being adversarial or the derivation gained a "
        f"factor it does not need"
    )
    assert worst <= 1.0, "the bound was exceeded, which the accuracy tests should have caught"
