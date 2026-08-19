r"""Parity and correctness of the C++ cardinal B-spline kernel against its Numba oracle.

`cpp/include/pantr/basis/cardinal_bspline.hpp` names this file as the place its
parity tolerance is derived. This is that derivation, and the tests that would
notice if it stopped holding. The shared, kernel-agnostic machinery lives in
``tests/_parity_harness.py``; only what is specific to this kernel is here.

The kernel
----------

Both backends run the same specialisation of Cox-de Boor to span index 0. Writing
:math:`N^{(k)}_r` for the value of output column ``r`` after stage ``k``, and with
:math:`u` the evaluation point:

.. math::

    t^{(k)}_r = \frac{r + (1 - u)}{k}, \qquad s^{(k)}_r = 1 - t^{(k)}_r,

    N^{(k)}_r = N^{(k-1)}_{r-1}\, s^{(k)}_{r-1} + N^{(k-1)}_r\, t^{(k)}_r,
    \qquad N^{(0)}_0 = 1 .

There are ``degree`` stages, and the dependency chain from an input to an output
element has length ``degree`` -- not ``degree * (degree + 1) / 2``, which is the
operation count. Errors compound along the chain, not along the operation count.

What the two backends share, bit for bit
----------------------------------------

This is the observation the whole parity analysis rests on, and it is what makes a
parity bound a different object from an accuracy bound.

``one_minus_u``, ``inv_k``, ``term`` and ``1 - term`` are computed by the identical
expressions in both backends, from identical inputs, and neither expression admits
contraction (``(r + one_minus_u) * inv_k`` is an add feeding a multiply, which is
the wrong way round to fuse). So the *coefficients* :math:`\hat t` and
:math:`\hat s` are bit-identical in the two backends, rounding errors included.

Every error they carry is therefore **common mode**: it cancels exactly in the
difference between the backends, and it does not cancel at all in the difference
between either backend and the mathematics. Hence:

* the **parity** bound is derived against a reference recurrence that uses the
  *computed* coefficients, where the convexity argument below applies;
* the **accuracy** bound is derived against the exact spline, where it does not.

Where the convexity argument holds, and where it does not
---------------------------------------------------------

For :math:`u \in [0, 1]` and :math:`0 \le r \le k-1`, :math:`t^{(k)}_r \in [0,1]`,
so :math:`s` and :math:`t` are both non-negative and each stage is a convex
combination of non-negative values. The lemma that matters is elementary and is
the reason the bound is linear rather than exponential in the degree: if
:math:`x, y \ge 0` and each is known to within a relative :math:`\varepsilon`,
then :math:`x + y` is known to within the same relative :math:`\varepsilon`. A
non-cancelling sum **inherits** its summands' relative error instead of
amplifying it, so per-stage errors add across stages.

Three corrections to the argument as originally stated:

1. **It is not "one rounding per stage".** Once the two backends' values differ at
   all, every later rounding differs too -- ``fl(a*s)`` and ``fl(b*s)`` for
   :math:`a \ne b` commit unrelated errors of size up to :math:`u` each, which is
   a first-order effect, not a second-order one. There is no cancellation to
   exploit, so the honest route is
   :math:`|A - B| \le |A - \text{exact}| + |\text{exact} - B|`: **twice** a
   one-sided forward-error bound. Per stage each backend commits one multiply
   rounding on the dominant summand (the two summands' multiplies count once
   between them, by the lemma) plus one add rounding, so the one-sided growth is
   :math:`2u` per stage and the parity bound is :math:`4\,n\,u` relative, not
   :math:`n\,u/2`. Measured against a deliberately FMA-enabled build (see below),
   the observed worst case reaches 0.29 of that bound, so it is neither wrong nor
   idle.

2. **The recurrence does cancel, just not where the argument looked.** ``1 - term``
   is a subtraction of two nearby quantities whenever ``term`` is near 1, which
   happens at :math:`r = k-1` for small :math:`u`, where
   :math:`1 - t = u/k`. Its *relative* error is
   :math:`\sim u_{\text{roundoff}} \cdot k / u`, unbounded as :math:`u \to 0`.
   Measured: at degree 1 and :math:`u = 1/400`, ``1 - fl(1 - u)`` is
   ``0.0024999999999999467`` against an exact ``0.0025``, wrong by **192 unit
   roundoffs** relative, against a claimed 2. (That is 96 *epsilons*; this file
   counts in :math:`u = \varepsilon/2` throughout, and the two were once
   conflated here.) The claim "nothing here can cancel" in the header's comment is
   false as an accuracy claim. It survives as a *parity* claim only because those
   coefficients are common mode.

   Pushed far enough the cancellation is total: once :math:`u \le \varepsilon/4`,
   ``1 - u`` rounds to exactly 1 and ``1 - term`` is exactly 0, so the last column
   comes back as 0 while the true value is :math:`u^n/n! > 0`. That is a complete
   loss of *relative* accuracy at an entry whose absolute size is negligible, and
   :func:`_companion_bounds` has to bound it rather than deny it -- see the hull
   there, and the two regression points at the end of :func:`_oracle_points`.

3. **Outside :math:`[0,1]` the convexity argument dies**, as expected: ``term``
   leaves :math:`[0,1]`, the stage weights take opposite signs, and contributions
   cancel. The domain is **not excluded** here, because it does not have to be.
   Replacing the recurrence's coefficients by their absolute values gives a
   companion recurrence :math:`X` (:func:`_companion_bounds`) that bounds the sum
   of the magnitudes of all contributions to each output element, and the error
   bound is relative to :math:`X` rather than to the value. Inside
   :math:`[0,1]` every coefficient is already non-negative, so :math:`X` equals
   the value and the bound degenerates exactly to the relative one. One bound,
   one implementation, valid on both sides of the span ends. What *is* excluded is
   :math:`|u|` large enough for the recurrence to overflow, where :math:`X` is
   infinite and no comparison means anything; the harness rejects an infinite
   amplification rather than silently passing.

Which claim holds in this build
-------------------------------

The two backends differ at exactly one site: ``saved + nr_old * term``, which the
C++ may fuse under ``-ffp-contract=on`` and Numba emits unfused. Whether it
*does* fuse is a property of the build, and the extension reports it
(``pantr._pantr_cpp.__fp_contract__``):

* ``cmake/PantrCompileOptions.cmake`` sets ``-ffp-contract=on`` but no ``-march``,
  so the target is baseline x86-64, which has **no FMA instruction**. Nothing can
  fuse, the two backends execute the identical sequence of IEEE-754 operations in
  the identical order, and parity is **bit-exact**. Asserting a
  ``degree * eps`` tolerance here would pass while asserting nothing.
* ``design/simd.md`` schedules ``-march=x86-64-v3`` as a later stage. Then the
  site fuses, and the bounded claim above becomes the live one.

So the claim is selected from the build's own provenance, and both branches are
written and both are exercised -- the bounded branch by
:func:`test_bounded_branch_admits_a_perturbation_at_the_bound` and its sibling,
which drive the harness directly rather than waiting for the rebuild.

float32
-------

Both backends accumulate in float64 and round only on the store into ``out``
(``accumulator_t`` in the header; verified for Numba in the header's own note, and
re-verified here by :func:`test_float32_intermediates_are_float64`). The parity
bound then carries one float32 store rounding per stage, giving
:math:`2\,n\,u_{32}` relative -- that is :math:`n` ulps of float32, not one.
"One ulp" is the per-site statement, and it is not the per-result statement:
measured on an FMA-enabled build at degree 11, two entries of one row differ by
**2 ulps** of float32, so the tighter claim is already refuted at degrees the
library uses.

That bound is also **too loose to catch the mistake it looks like it guards**: a
port that accumulated in ``float`` throughout differs from this one by about
1 ulp of float32, which :math:`n\,u_{32}` waves through for any :math:`n > 2`.
That bug gets its own test against an explicit model of the arithmetic contract,
not a tolerance.

And it is **too tight below the smallest normal**, which is the direction that
actually broke. Every bound above is proportional to a magnitude, so it goes to
zero with the value while the true error does not: a float32 store commits an
absolute error of up to one subnormal however small the result is. On this file's
own :func:`_oracle_points`, float32 at degree 16, that made the accuracy bound
too small by a factor of :math:`10^6` -- for two backends that agree perfectly and
a kernel that is right. Every bound here therefore carries an absolute floor as
well as a relative term; see :func:`~tests._parity_harness.underflow_floor`.

Ground truth
------------

Parity is a consistency check. Both backends running the same wrong recurrence
agree perfectly, so this file also carries an oracle that is not the code: the
truncated-power form of the cardinal B-spline evaluated in exact rational
arithmetic (:func:`_exact_rows`), plus two analytic invariants (partition of
unity, and the reflection :math:`N_r(u) = N_{n-r}(1-u)`). Those are what would
catch a recurrence that is wrong in both backends at once.
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest

from pantr import _numba_compat
from pantr._backend import Backend, active_backend, use_backend
from pantr.basis import tabulate_cardinal_bspline_1d
from pantr.basis._basis_backend import cardinal_bspline_core
from tests._parity_harness import (
    AccuracyClaim,
    FloatArray,
    ParityClaim,
    ParityKind,
    Roundings,
    absolute_tolerance,
    assert_accuracy,
    assert_parity,
    bitwise_parity,
    bounded_parity,
    build_provenance,
    contraction_may_fuse,
    cpp_backend_available,
    demand_cpp_backend,
    derived_accuracy,
    underflow_floor,
    unit_roundoff,
)


@pytest.fixture(scope="session")
def cpp_backend() -> None:
    """Require the compiled extension for the test that asks for it.

    Requested explicitly rather than applied module-wide, so the harness self-tests
    and the Numba-side invariants below still run in an installation without the
    extension -- which is the common local configuration. The skip-or-fail decision
    itself lives in the harness, so the next ported kernel inherits it instead of
    reaching for a plain ``skipif`` that would skip silently.
    """
    demand_cpp_backend()


DTYPES: Final = (np.float64, np.float32)
"""The two storage formats the B-spline layer accepts."""

DEGREES: Final = (0, 1, 2, 3, 4, 5, 7, 11, 16)
"""Degrees swept by the main parity tests.

0 and 1 are the two branches that are provably exact; 2 is the smallest degree at
which the backends can diverge at all; 16 is past any degree pantr's own builders
reach, so the linear-in-degree bound is exercised where it is loosest.
"""

OUTSIDE_DEGREES: Final = (1, 2, 3, 5, 8, 12, 16)
"""Degrees swept off the central span, reaching the same top degree as :data:`DEGREES`.

This list used to stop at 12, on the stated grounds that the extrapolated
polynomial grows like ``|u|**degree`` until the companion recurrence overflows.
**That reason is false**, and in the harmless direction: the recurrence divides by
``k`` at every stage, so the amplification behaves like ``|u|**n / n!`` and turns
over once ``n`` passes ``|u|``. Measured on :func:`_outside_points`, whose largest
magnitude is 4, the maximum amplification rises to 49.9 at degree 12 and then
*falls* -- 47.1 at degree 16, 27.4 at degree 60 -- and is finite at every degree
tried. Nothing here approaches overflow, so nothing had to be capped.

There would still be a real ceiling for a point set reaching much further out,
where ``|u|**n`` outruns ``n!`` for the degrees of interest. The harness refuses
an infinite amplification rather than passing vacuously, so that case reports
itself.
"""


def _span_points(dtype: npt.DTypeLike) -> FloatArray:
    """Build the in-span evaluation set on a dyadic grid.

    A grid of 1/64 rather than an arbitrary linspace so that ``1 - u`` is exact in
    both formats, which the reflection test needs: an inexact mirror point would
    put a rounding of the *argument* inside a comparison of *values*.

    Args:
        dtype (npt.DTypeLike): Storage format.

    Returns:
        FloatArray: Points covering [0, 1] inclusive.
    """
    return np.linspace(0.0, 1.0, 65, dtype=dtype)


def _boundary_points(dtype: npt.DTypeLike) -> FloatArray:
    """Build the set of points at and immediately around the span ends.

    Args:
        dtype (npt.DTypeLike): Storage format.

    Returns:
        FloatArray: The two span ends, their nearest neighbours on both sides, and
            both signed zeros.
    """
    one = np.array(1.0, dtype=dtype)
    zero = np.array(0.0, dtype=dtype)
    return np.array(
        [
            0.0,
            -0.0,
            float(np.nextafter(zero, np.array(-1.0, dtype=dtype))),
            float(np.nextafter(zero, one)),
            float(np.nextafter(one, zero)),
            1.0,
            float(np.nextafter(one, np.array(2.0, dtype=dtype))),
        ],
        dtype=dtype,
    )


def _outside_points(dtype: npt.DTypeLike) -> FloatArray:
    """Build the off-span evaluation set.

    Args:
        dtype (npt.DTypeLike): Storage format.

    Returns:
        FloatArray: Points strictly outside [0, 1] on both sides.
    """
    return np.array([-3.0, -2.0, -1.0, -0.5, -0.25, 1.25, 1.5, 2.0, 3.0, 4.0], dtype=dtype)


def _oracle_points(dtype: npt.DTypeLike) -> FloatArray:
    """Build the small mixed set the exact rational oracle is evaluated on.

    Kept short because the oracle costs one exact rational power per term, and
    deliberately **not** dyadic throughout: on a grid of 1/64 the whole recurrence
    is exact at low degree, the observed error is identically zero, and a test that
    the bound is not idle would have nothing to measure. Thirds, sevenths and 0.1
    are what make the roundings happen; 0.001 and 0.999 are what make the
    cancellation in ``1 - term`` bite.

    The last two are the regression points for the two ways a bound built only
    from relative terms fails, and both are the *smallest* points that reach
    their case rather than arbitrary small ones:

    * ``nextafter(0, 1)`` is the smallest positive float64, where every product
      of it underflows and the error stops being proportional to anything;
    * ``5e-17`` is below ``eps/4``, which is the largest point at which
      ``1 - u`` still returns exactly 1. That is what makes the computed
      ``1 - term`` exactly zero at ``r = k-1`` and collapses a companion built on
      the computed coefficients.

    In float32 both narrow to values already covered (0 and a normal), which is
    fine: the float32 case of the same family is already reached by ``0.999``,
    whose leading entry shrinks past what float32 can hold as the degree rises.
    Measured there: at degree 11 it is ``2.5e-41`` and still stored as a normal;
    at degree 12, ``2.1e-45``, stored as the smallest subnormal; from degree 13
    the store flushes it to exactly zero, and by degree 16 the true value is
    ``4.8e-62`` against a stored 0. The relative-only bound is first violated at
    degree 11 -- before any flush to zero, because a subnormal store already
    commits an absolute error no relative term can express.

    Args:
        dtype (npt.DTypeLike): Storage format.

    Returns:
        FloatArray: Points inside, at, and outside the span.
    """
    return np.array(
        [
            -1.5,
            -1.0 / 3.0,
            -0.25,
            0.0,
            1.0 / 1024.0,
            0.001,
            0.1,
            0.125,
            1.0 / 3.0,
            0.375,
            0.5,
            0.625,
            1.0 / 7.0,
            0.9375,
            0.999,
            1.0,
            1.25,
            1.7,
            2.5,
            float(np.nextafter(0.0, 1.0)),
            5e-17,
        ],
        dtype=dtype,
    )


def _rounding_points(dtype: npt.DTypeLike) -> FloatArray:
    """Build a dense set of points on which the recurrence actually rounds.

    A grid of 1/200 rather than 1/64: dyadic points make the low-degree recurrence
    exact in *any* accumulator format, which would leave the accumulator check
    comparing two models that happen to agree.

    Args:
        dtype (npt.DTypeLike): Storage format.

    Returns:
        FloatArray: Points on [0, 1] plus a few adversarial ones.
    """
    return np.concatenate(
        [
            np.linspace(0.0, 1.0, 201),
            np.array([1.0 / 3.0, 1.0 / 7.0, 0.1, 1.0e-4, 1.0 - 1.0e-4]),
        ]
    ).astype(dtype)


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------


def _exact_rows(degree: int, points: FloatArray) -> npt.NDArray[np.float64]:
    r"""Evaluate the tabulation exactly, in rational arithmetic, without a recurrence.

    Independent of the code under test: it uses the truncated-power form of the
    cardinal B-spline rather than Cox-de Boor, so a wrong recurrence in *both*
    backends does not hide behind it.

    With :math:`M_n` the cardinal B-spline supported on :math:`[0, n+1]`,

    .. math::

        M_n(x) = \frac{1}{n!} \sum_{i=0}^{n+1} (-1)^i \binom{n+1}{i} (x - i)_+^n ,

    the ``degree + 1`` functions the kernel returns are
    :math:`\text{out}[j, r] = M_n(u_j + n - r)`, whose leading term is the one with
    :math:`i \le n - r`. Dropping the truncation and keeping exactly those terms
    gives the **polynomial piece** on the central span, analytically continued --
    which is what the kernel computes off the span too, since its arithmetic is
    polynomial in ``u`` and branches on nothing.

    Args:
        degree (int): Degree of the basis.
        points (FloatArray): Evaluation points; each is converted to the rational
            it exactly is.

    Returns:
        npt.NDArray[np.float64]: Exact values, rounded once to float64 on return.
    """
    factorial = math.factorial(degree)
    out = np.zeros((points.size, degree + 1), dtype=np.float64)
    for j, point in enumerate(points):
        base = Fraction(float(point))
        for r in range(degree + 1):
            x = base + (degree - r)
            total = Fraction(0)
            for i in range(degree - r + 1):
                total += (-1) ** i * math.comb(degree + 1, i) * (x - i) ** degree
            out[j, r] = float(total / factorial)
    return out


def _companion_bounds(
    degree: int, points: FloatArray, storage: npt.DTypeLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Run the amplification and forward-error companions of the kernel's recurrence.

    Two arrays, both elementwise, both produced by walking the kernel's own
    recurrence with the kernel's own computed coefficients:

    ``X`` -- the same recurrence with every coefficient replaced by an upper bound
    on its magnitude and started at 1. It bounds the sum of the magnitudes of all
    contributions to each output element, so it is the factor by which a relative
    perturbation of the intermediates can appear in the result. Inside
    :math:`[0,1]` all coefficients are already non-negative and ``X`` coincides
    with the value itself to within a rounding.

    The bound used is the **hull** :math:`|\hat c| + \delta_c` -- the computed
    coefficient widened by its own error -- and not :math:`|\hat c|` alone. The
    difference is invisible almost everywhere, inflating ``X`` by about
    :math:`3nu`, which is :math:`5 \times 10^{-15}` at degree 16. In one place it
    is the entire bound. A computed coefficient can be **exactly zero** while the
    true one is not: for :math:`u \lesssim \varepsilon/4` the subtraction
    ``1 - u`` returns exactly 1, so ``1 - term`` at :math:`r = k-1` is exactly 0,
    and a companion built on :math:`|\hat c|` has its last column collapse to 0
    at that stage and every later one, after which it bounds nothing. The true
    value there is not zero -- at :math:`u = 5\times 10^{-17}`, degree 3, it is
    :math:`2.1\times 10^{-50}` where the kernel returns exactly 0 -- so the
    bound was exceeded by a factor of :math:`1/u = 2^{53}`. That is a *correct*
    kernel losing all relative accuracy on an entry of negligible absolute size,
    which is the case a companion has to survive rather than deny.

    ``E`` -- an absolute forward-error bound, propagated by the same recurrence and
    fed at each step by the roundings that step commits:

    * the coefficient ``term`` carries three accumulator roundings (``1/k``, the
      add, the multiply) plus the propagated error of ``1 - u``;
    * ``1 - term`` carries one more, **plus the absolute error of** ``term``, which
      is where the cancellation enters: near ``term = 1`` that absolute error is
      large compared with ``1 - term`` itself;
    * each stage commits one multiply and one add in the accumulator, and one
      store into the storage format.

    Every one of those roundings is charged **twice over**, once relatively and
    once absolutely, per Higham's model ``fl(x op y) = (x op y)(1 + d) + eta``.
    The absolute part is :func:`~tests._parity_harness.underflow_floor` of the
    format the rounding happens in, and omitting it is not a small error: a
    relative-only bound shrinks with the value while the true error does not, so
    it goes to zero exactly where the recurrence underflows. Measured on
    :func:`_oracle_points` with the relative-only version, float32 at degree 16
    exceeded its own bound by a factor of ``1.0e6`` -- at ``u = 0.999``, where
    the leading basis function is ``4.8e-62`` and the float32 store flushes it to
    zero. That store commits an absolute error of ``4.8e-62``, which no multiple
    of ``u * 4.8e-62`` can cover, and which one float32 subnormal
    (``1.4e-45``) covers with room to spare.

    ``E`` is the accuracy bound (against the exact spline). The parity bound uses
    only ``X`` and the harness's own floor, because the coefficient errors are
    common to the two backends.

    Args:
        degree (int): Degree of the basis.
        points (FloatArray): Evaluation points.
        storage (npt.DTypeLike): Format the output array holds.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ``(X, E)``, both
            of shape ``(points.size, degree + 1)``.
    """
    narrow_store = np.dtype(storage) != np.float64
    u_acc = unit_roundoff(np.float64)
    u_store = unit_roundoff(storage) if narrow_store else 0.0
    # A store into the accumulator's own format rounds nothing, so it carries
    # neither a relative cost nor a floor.
    eta_acc = underflow_floor(np.float64)
    eta_store = underflow_floor(storage) if narrow_store else 0.0

    amplification = np.zeros((points.size, degree + 1), dtype=np.float64)
    error = np.zeros_like(amplification)
    amplification[:, 0] = 1.0
    if degree == 0:
        return amplification, error

    one_minus_u = 1.0 - points.astype(np.float64)
    d_one_minus_u = u_acc * np.abs(one_minus_u) + eta_acc

    for k in range(1, degree + 1):
        inv_k = 1.0 / k
        saved_x = np.zeros(points.size, dtype=np.float64)
        saved_e = np.zeros(points.size, dtype=np.float64)
        for r in range(k):
            x_old = amplification[:, r].copy()
            e_old = error[:, r].copy()

            term = (r + one_minus_u) * inv_k
            complement = 1.0 - term
            d_term = 3.0 * u_acc * np.abs(term) + 3.0 * eta_acc + inv_k * d_one_minus_u
            d_complement = u_acc * np.abs(complement) + eta_acc + d_term

            # The hulls: each bounds BOTH the computed coefficient and the exact
            # one, which is what lets X bound the exact value as well as the
            # computed one. See the note above on the column that collapses.
            term_hull = np.abs(term) + d_term
            complement_hull = np.abs(complement) + d_complement

            new_x = saved_x + x_old * term_hull
            amplification[:, r] = new_x
            error[:, r] = (
                saved_e
                + term_hull * e_old
                + x_old * d_term
                + u_acc * x_old * np.abs(term)
                + eta_acc
                + u_acc * new_x
                + eta_acc
                + u_store * new_x
                + eta_store
            )
            saved_e = (
                complement_hull * e_old
                + x_old * d_complement
                + u_acc * x_old * np.abs(complement)
                + eta_acc
            )
            saved_x = x_old * complement_hull
        amplification[:, k] = saved_x
        error[:, k] = saved_e + u_store * saved_x + eta_store

    return amplification, error


def _arithmetic_contract_model(
    degree: int, points: FloatArray, accumulator: npt.DTypeLike
) -> FloatArray:
    """Recompute the kernel in NumPy with a *chosen* accumulator format.

    This is deliberately a mirror of the kernel's arithmetic and is worth nothing
    as a correctness oracle -- it recomputes the same algorithm. Its one job is to
    pin the *arithmetic contract*: which format the intermediates live in. Run with
    ``accumulator=np.float64`` it must reproduce both backends bit for bit; run
    with ``accumulator=np.float32`` it must not. The pair is what makes the check
    discriminating, and neither half means anything alone.

    Args:
        degree (int): Degree of the basis.
        points (FloatArray): Evaluation points, in the storage format.
        accumulator (npt.DTypeLike): Format the intermediates are held in.

    Returns:
        FloatArray: Values in ``points.dtype``, stored exactly as the kernel does.
    """
    # Untyped on purpose: these are scalar constructors chosen at run time, which
    # is the whole point of the model.
    acc: Any = np.dtype(accumulator).type
    store: Any = points.dtype.type
    out = np.zeros((points.size, degree + 1), dtype=points.dtype)
    if degree == 0:
        out[:, 0] = store(1.0)
        return out
    for j in range(points.size):
        out[j, 0] = store(1.0)
        u = acc(points[j])
        one_minus_u = acc(1.0) - u
        for k in range(1, degree + 1):
            inv_k = acc(1.0) / acc(k)
            saved = acc(0.0)
            for r in range(k):
                nr_old = acc(out[j, r])
                term = (acc(r) + one_minus_u) * inv_k
                out[j, r] = store(saved + nr_old * term)
                saved = nr_old * (acc(1.0) - term)
            out[j, k] = store(saved)
    return out


# ---------------------------------------------------------------------------
# Running the kernels, and stating the claim
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _jit_warmup_barrier() -> None:
    """Block until the background Numba warmup finished, before any kernel call.

    Session-scoped because the barrier is a once-per-process event. CLAUDE.md
    records the failure it prevents: a Layer 3 ``parallel=True`` kernel called from
    the main thread while ``pantr.__init__`` is still compiling on a background
    thread **aborts** the interpreter rather than raising, and a kernel-heavy file
    reaching kernels early is exactly the pattern that triggers it. This file calls
    the Layer 3 kernel directly, bypassing the Layer 2 entry points that carry the
    barrier themselves, so it must take the barrier itself.
    """
    _numba_compat.wait_for_jit_warmup()


def _tabulate(backend: Backend, degree: int, points: FloatArray) -> FloatArray:
    """Run one backend's Layer 3 kernel into a freshly poisoned output array.

    The output is filled with NaN rather than zeros, so an element the kernel
    fails to write is a NaN in the result rather than a plausible zero. Every
    comparison in this file therefore also checks that the kernel wrote everything
    it promised to write.

    Args:
        backend (Backend): Which implementation to call.
        degree (int): Degree of the basis.
        points (FloatArray): Contiguous 1D evaluation points.

    Returns:
        FloatArray: Shape ``(points.size, degree + 1)`` in ``points.dtype``.
    """
    out = np.full((points.size, degree + 1), np.nan, dtype=points.dtype)
    cardinal_bspline_core(backend)(np.int32(degree), points, out)
    return out


def _both_backends(degree: int, points: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Run both backends on the same input.

    Args:
        degree (int): Degree of the basis.
        points (FloatArray): Contiguous 1D evaluation points.

    Returns:
        tuple[FloatArray, FloatArray]: ``(cpp, numba)``.
    """
    return (
        _tabulate(Backend.CPP, degree, points),
        _tabulate(Backend.NUMBA, degree, points),
    )


_EXACT_BY_STRUCTURE: Final[str] = (
    "degree {degree} has no site at which the backends can diverge. Degree 0 writes "
    "the literal 1 and returns. Degree 1 runs the recurrence once, at k=1, r=0, where "
    "`saved` is exactly 0 and `nr_old` is exactly 1: the site `saved + nr_old * term` "
    "is `0 + 1 * term`, whose product is exact, so fusing and not fusing round the same "
    "real number the same once. This holds in every build, FMA or not."
)

_EXACT_BY_BUILD: Final[str] = (
    "this build cannot fuse: pantr._pantr_cpp.__fp_contract__ is "
    "'unavailable-on-target-isa', because cmake/PantrCompileOptions.cmake sets "
    "-ffp-contract=on but no -march, leaving the target at baseline x86-64 which has "
    "no FMA instruction. Numba emits the same site unfused. The two backends therefore "
    "execute the identical sequence of IEEE-754 operations in the identical order. "
    "Adding -march=x86-64-v3 (design/simd.md, stage 2) flips this to the bounded claim; "
    "a tolerance asserted here instead would be satisfied by construction."
)

_BOUNDED_BY_FMA: Final[str] = (
    "the backends differ at one site, `saved + nr_old * term`, which this build may "
    "fuse and Numba does not. Neither is the exact answer, so the parity bound is twice "
    "a one-sided forward-error bound. Per stage each backend commits one multiply "
    "rounding on the dominant summand -- the two summands are non-negative inside the "
    "span, so a non-cancelling sum inherits the larger of their relative errors rather "
    "than the sum, and the two multiplies count once between them -- plus one add, plus "
    "one store when the storage format is narrower than the float64 accumulator. Over "
    "`degree` stages that gives 2*gamma, and the elementwise amplification is the "
    "companion recurrence run on |coefficients|, which equals the value itself inside "
    "[0, 1] and is strictly larger outside, where the stage weights change sign and the "
    "convexity argument no longer applies. The coefficients themselves are bit-identical "
    "in the two backends and their (large, cancelling) errors are common mode."
)


def _parity_claim(degree: int, points: FloatArray) -> ParityClaim:
    """Select the parity claim this build supports for a given case.

    Args:
        degree (int): Degree of the basis.
        points (FloatArray): The evaluation points, which fix the amplification.

    Returns:
        ParityClaim: The strongest claim this build supports for this case.
    """
    if degree <= 1:
        return bitwise_parity(why=_EXACT_BY_STRUCTURE.format(degree=degree))
    if not contraction_may_fuse():
        return bitwise_parity(why=_EXACT_BY_BUILD)
    amplification, _ = _companion_bounds(degree, points, points.dtype)
    return bounded_parity(
        roundings=Roundings(stages=degree, accumulator_per_stage=2, storage_per_stage=1),
        accumulator=np.float64,
        storage=points.dtype,
        amplification=amplification,
        why=_BOUNDED_BY_FMA,
    )


def _accuracy_claim(
    degree: int, points: FloatArray, exact: npt.NDArray[np.float64]
) -> AccuracyClaim:
    """Build the accuracy claim against the exact rational oracle.

    Args:
        degree (int): Degree of the basis.
        points (FloatArray): Evaluation points.
        exact (npt.NDArray[np.float64]): The oracle's values.

    Returns:
        AccuracyClaim: The elementwise bound and its derivation.
    """
    _, error = _companion_bounds(degree, points, points.dtype)
    bound = error + unit_roundoff(np.float64) * np.abs(exact)
    return derived_accuracy(
        bound=bound,
        why=(
            "the forward-error companion of the recurrence (see _companion_bounds), "
            "fed per stage by three accumulator roundings on `term`, one more on "
            "`1 - term` carrying `term`'s absolute error -- which is the cancellation "
            "the convexity argument misses -- one multiply and one add on the values, "
            "and one store when the storage format is narrower than the accumulator. "
            "Plus one float64 rounding for converting the exact rational oracle to a "
            "float. This bound is NOT the parity bound: the coefficient errors it is "
            "dominated by are common to both backends and cancel in parity."
        ),
    )


# ---------------------------------------------------------------------------
# Availability, and making a skip audible
# ---------------------------------------------------------------------------


def test_cpp_extension_presence_is_declared() -> None:
    """Report the extension's presence as a test outcome rather than as silence.

    CLAUDE.md: "A missing optional dependency skips tests silently", and a local
    green built on such a skip has let real failures through here before. This test
    exists so that state is always in the report -- a pass when the extension is
    there, one visible skip with a build hint when it is not, and a failure when
    ``PANTR_REQUIRE_CPP`` says it had to be.
    """
    demand_cpp_backend()
    assert cpp_backend_available()


def test_build_provenance_is_reported(cpp_backend: None) -> None:
    """Pin the three attributes the parity claim is selected from.

    The claim in :func:`_parity_claim` is conditional on ``__fp_contract__``. A
    binding that stopped reporting it, or reported an unrecognised value, would send
    every parity test down whichever branch a default happened to pick, silently.
    """
    provenance = build_provenance()
    assert provenance.compiler, "the extension must name the compiler that built it"
    assert provenance.fp_contract in {"available", "unavailable-on-target-isa"}, (
        f"unrecognised __fp_contract__ {provenance.fp_contract!r}: the parity claim is "
        f"selected from this value and cannot be selected from a value it does not know"
    )
    assert isinstance(provenance.has_std_mdspan, bool)


def test_jit_warmup_barrier_was_taken() -> None:
    """Assert the session barrier ran before any kernel call.

    No in-process test can observe the race it prevents (the barrier is a
    once-per-process event and the failure is an abort, not an exception), so what
    is testable is that the barrier was taken. CLAUDE.md says exactly this.
    """
    assert _numba_compat._warmup_done, (
        "the session-scoped warmup barrier did not run before the kernel tests; a "
        "parallel=True kernel called during background compilation aborts the process"
    )


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", DEGREES)
def test_parity_on_the_span(degree: int, dtype: npt.DTypeLike, cpp_backend: None) -> None:
    """The two backends agree on [0, 1], as far as this build's claim allows.

    The broad sweep. It catches anything that changes what the port computes: a
    shifted index, a swapped ``term`` and ``1 - term``, a wrong loop bound, a
    ``float`` accumulator, a row written in reverse. The output array is
    NaN-poisoned before the call, so an element neither backend writes fails here
    too rather than comparing equal at zero.
    """
    points = _span_points(dtype)
    cpp, numba = _both_backends(degree, points)
    assert np.all(np.isfinite(cpp)), "the C++ kernel left an element unwritten"
    assert_parity(
        cpp,
        numba,
        _parity_claim(degree, points),
        context=f"cardinal B-spline, degree {degree}, {np.dtype(dtype).name}, on [0, 1]",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", DEGREES)
def test_parity_at_the_span_ends(degree: int, dtype: npt.DTypeLike, cpp_backend: None) -> None:
    """The two backends agree at 0, at 1, and one ulp either side of each.

    The span ends are where a well-meaning guard gets added: an ``if (u < 0)``, a
    ``std::clamp``, a ``std::max(T(0), ...)``. Any of them changes the value at
    ``nextafter(0, -1)`` and at ``nextafter(1, 2)`` while leaving the interior
    untouched, so the sweep above would not see it. Both signed zeros are included
    because the kernel's first stage subtracts the point from 1.
    """
    points = _boundary_points(dtype)
    cpp, numba = _both_backends(degree, points)
    assert_parity(
        cpp,
        numba,
        _parity_claim(degree, points),
        context=f"cardinal B-spline, degree {degree}, {np.dtype(dtype).name}, at the span ends",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", OUTSIDE_DEGREES)
def test_parity_outside_the_span(degree: int, dtype: npt.DTypeLike, cpp_backend: None) -> None:
    """The two backends agree off the span, where the convexity argument does not hold.

    Both the docstring of the C++ kernel and the Python one promise that points
    outside [0, 1] are *evaluated*, not clamped: the recurrence is polynomial in
    ``u`` and branches on nothing, so it returns the analytic continuation of the
    central polynomial piece. This is the case a clamp would break, and it is also
    the case that exercises the amplification array -- inside the span the
    amplification equals the value and the bound is a plain relative one, outside it
    is strictly larger.
    """
    points = _outside_points(dtype)
    cpp, numba = _both_backends(degree, points)
    assert np.all(np.isfinite(cpp)), "the C++ kernel produced a non-finite value off the span"
    assert_parity(
        cpp,
        numba,
        _parity_claim(degree, points),
        context=f"cardinal B-spline, degree {degree}, {np.dtype(dtype).name}, outside [0, 1]",
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_empty_input_writes_nothing_and_returns(dtype: npt.DTypeLike, cpp_backend: None) -> None:
    """Zero evaluation points is a no-op, in both backends.

    ``num_pts`` is a ``std::size_t`` in the C++ kernel, so a loop written
    ``j <= num_pts - 1`` would wrap to 2**64 iterations and walk off the array
    instead of doing nothing. That class of bug is invisible at every other size.
    """
    points = np.zeros(0, dtype=dtype)
    for degree in (0, 1, 3):
        cpp, numba = _both_backends(degree, points)
        assert cpp.shape == (0, degree + 1)
        assert_parity(
            cpp,
            numba,
            bitwise_parity(why="no element is computed, so there is nothing to round"),
            context=f"cardinal B-spline, degree {degree}, empty input",
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_degree_zero_writes_exactly_one(dtype: npt.DTypeLike, cpp_backend: None) -> None:
    """Degree 0 is the constant 1, written into every element.

    Its own branch in both kernels, reached by no other test's arithmetic. The
    NaN poisoning is what makes this catch the real failure -- a branch that returns
    without writing leaves the caller's array untouched, which a zero-filled output
    would hide.
    """
    points = np.concatenate([_span_points(dtype), _outside_points(dtype)])
    cpp = _tabulate(Backend.CPP, 0, points)
    assert cpp.shape == (points.size, 1)
    assert np.array_equal(cpp, np.ones_like(cpp)), (
        "degree 0 must be exactly 1 everywhere, including off the span: the constant "
        "basis function is not truncated by this kernel any more than the others are"
    )


def test_degree_one_is_bitwise_exact_in_every_build(cpp_backend: None) -> None:
    """Degree 1 agrees bit for bit whatever the build does about contraction.

    Proved rather than measured: at degree 1 the only site is ``k=1, r=0``, where
    ``saved`` is exactly 0 and ``nr_old`` is exactly 1, so ``saved + nr_old * term``
    is ``0 + 1 * term`` and the product is exact -- fused and unfused round the same
    real number once.

    This is the sharpest bit-level assertion in the file, and it catches something
    no tolerance can: a port that rearranged the *expression*, computing
    ``(r + 1 - u) / k`` instead of ``(r + one_minus_u) * inv_k``. That rearrangement
    is algebraically identical, numerically different in the last bits, and would be
    absorbed silently by any bounded claim.
    """
    for dtype in DTYPES:
        for points in (_span_points(dtype), _boundary_points(dtype), _outside_points(dtype)):
            cpp, numba = _both_backends(1, points)
            assert_parity(
                cpp,
                numba,
                bitwise_parity(why=_EXACT_BY_STRUCTURE.format(degree=1)),
                context=f"cardinal B-spline, degree 1, {np.dtype(dtype).name}",
            )


def test_float32_intermediates_are_float64(cpp_backend: None) -> None:
    """The float32 path accumulates in float64, and the check can tell the difference.

    ``accumulator_t`` in the C++ header states this decision, and
    ``design/large_data_fitting.md`` gives its reason. It cannot be checked by the
    float32 parity tolerance: a port accumulating in ``float`` throughout differs by
    about one ulp of float32, and the derived float32 parity bound is
    ``degree`` ulps, so it would wave the bug through for any degree above 2.

    So it is checked against an explicit model of the arithmetic contract, in both
    directions. The second assertion is the one that gives the first its meaning: if
    the two models did not differ, agreeing with one of them would say nothing.
    """
    points = _rounding_points(np.float32)
    for degree in (1, 2, 3, 6, 11):
        cpp = _tabulate(Backend.CPP, degree, points)
        wide = _arithmetic_contract_model(degree, points, np.float64)
        narrow = _arithmetic_contract_model(degree, points, np.float32)
        assert np.array_equal(cpp, wide), (
            f"degree {degree}: the float32 kernel does not match a float64-accumulator "
            f"model bit for bit, so its intermediates are not float64"
        )
        assert not np.array_equal(wide, narrow), (
            f"degree {degree}: the float64- and float32-accumulator models agree here, "
            f"so the assertion above is vacuous and this case proves nothing"
        )


# ---------------------------------------------------------------------------
# Correctness against an oracle that is not the code
# ---------------------------------------------------------------------------


def test_exact_oracle_reproduces_the_published_degree_two_values() -> None:
    """Check the oracle itself against values published in the library's own docs.

    The rational oracle is the only ground truth in this file that is not the code,
    so nothing else would catch it being wrong. These four rows are the worked
    example in :func:`pantr.basis.tabulate_cardinal_bspline_1d`'s docstring, and
    every entry is a dyadic rational, so the comparison is exact.
    """
    points = np.array([0.0, 0.5, 0.75, 1.0])
    expected = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.125, 0.75, 0.125],
            [0.03125, 0.6875, 0.28125],
            [0.0, 0.5, 0.5],
        ]
    )
    assert np.array_equal(_exact_rows(2, points), expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", (0, 1, 2, 3, 5, 8, 12))
def test_matches_the_exact_rational_oracle(
    degree: int, dtype: npt.DTypeLike, cpp_backend: None
) -> None:
    """Both backends compute the right function, not merely the same function.

    Parity is a consistency check: two backends running the same wrong recurrence
    agree perfectly. This is the test that would notice, and it is the reason the
    suite's ground truth is not the code under test. The bound is the forward-error
    companion of the recurrence, which is far larger than the parity bound because
    the cancellation in ``1 - term`` is real -- it just happens to be common mode
    between the backends.

    Degree 12 is here because the degree list used to stop at 8 while
    :data:`DEGREES` ran to 16, and the gap was load-bearing: on float32 this test
    is violated from degree 11 upwards by an error model with no underflow floor,
    and its own sweep never reached one. A degree list shorter than the one the
    parity tests use is a place for exactly that to hide.
    """
    points = _oracle_points(dtype)
    exact = _exact_rows(degree, points)
    claim = _accuracy_claim(degree, points, exact)
    for backend in (Backend.CPP, Backend.NUMBA):
        assert_accuracy(
            _tabulate(backend, degree, points),
            exact,
            claim,
            context=f"{backend.name} cardinal B-spline, degree {degree}, {np.dtype(dtype).name}",
        )


@pytest.mark.parametrize("degree", (0, 1, 2, 3, 5, 8))
def test_forward_error_bound_is_neither_violated_nor_idle(degree: int, cpp_backend: None) -> None:
    """The derived accuracy bound is both correct and tight enough to bite.

    A bound nothing approaches is a bound that would not notice the kernel getting
    worse. This asserts the observed error reaches at least 1% of it somewhere,
    which is the check that would have caught the original derivation: that one
    claimed ``2 * degree`` unit roundoffs relative and the true error reaches
    several hundred, so it is violated rather than idle.

    **Where the 1% comes from.** The bound sums roughly ``6 * degree`` worst cases,
    each of which assumes its rounding took the largest value of the right sign.
    Real roundings have mixed signs, so a realization behaves like a random walk
    and reaches about ``1 / sqrt(N)`` of a sum of ``N`` worst cases -- between
    0.41 and 0.12 over the degrees swept here. That is a heuristic and not a
    bound, so the threshold is set an order of magnitude below it. What the test
    then detects is a bound more than ten times looser than any realization could
    plausibly approach, which is the shape of the defect that actually occurred.
    Measured, for calibration rather than as the criterion: the observed ratio
    runs from 0.136 at degree 1 to 0.213 at degree 12, so the assertion holds with
    more than a factor of ten to spare and is nowhere near being a fitted number.
    """
    points = _oracle_points(np.float64)
    exact = _exact_rows(degree, points)
    claim = _accuracy_claim(degree, points, exact)
    deviation = assert_accuracy(
        _tabulate(Backend.CPP, degree, points),
        exact,
        claim,
        context=f"cardinal B-spline, degree {degree}, tightness probe",
    )
    if degree == 0:
        assert deviation.max_absolute == 0.0, "degree 0 is exact"
        return
    assert deviation.max_ratio_to_bound > 0.01, (
        f"degree {degree}: the observed error reaches only "
        f"{deviation.max_ratio_to_bound:.3g} of the derived bound, which is loose "
        f"enough that it would accept a substantially worse kernel without complaint"
    )


def test_the_bounds_slack_does_not_shrink_as_the_degree_grows(cpp_backend: None) -> None:
    """Falsify the bound's degree dependence rather than assume it.

    The bound grows linearly in ``degree``. If the true error grew faster, the
    ratio of the two would climb with degree, and a bound that is correct at
    degree 2 and wrong in its degree dependence would still pass an "is it under
    the bound" test at small degrees while failing this one.

    **What this cannot see, despite an earlier claim that it could.** It cannot
    distinguish errors that *add* across stages from errors that *compound*. Those
    differ by ``(1 + 2u)^n - 1`` against ``2nu``, whose ratio is ``1 + (n-1)u`` --
    at degree 12 in float64 that is ``1 + 1.2e-15``, some fourteen orders of
    magnitude below the ratio's own scatter. No measurement in this format can
    resolve it, so no test can, and one claiming to would be asserting a check
    nobody performs.

    **What it can see, and where the threshold comes from.** A bound short by a
    factor growing like the degree -- the realistic way to get the shape wrong,
    and what a missing stage or a mis-stated stage count produces. Such a bound
    would make the ratio grow by ``12 / 3 = 4`` between the two ends of the sweep,
    while a correct one keeps it flat, i.e. a growth of 1. The threshold is the
    geometric mean of the two hypotheses, ``sqrt(1 * 4) = 2``, which is the choice
    that does not favour either. Measured, as corroboration rather than as the
    derivation: the correct bound gives a growth of 0.82 (it drifts slightly
    *down*), and simulating the linear shortfall on top of that gives about 3.2 --
    3.26 if the ratio profile is taken as unchanged and simply scaled by 4, 3.24 if
    each degree's ratio is scaled by its own degree. So the threshold rejects the
    alternative by a factor of about 1.6 and accepts the truth by about 2.4, and
    the gap does not depend on which of the two readings is used.

    The shipped threshold was 4, which accepts 3.2 and so admitted precisely the
    alternative the docstring named.
    """
    low_degrees = (1, 2, 3)
    high_degrees = (10, 11, 12)
    # A bound short by one power of the degree would inflate the ratio by this
    # much across the sweep; a correct one leaves it flat. Split the difference.
    growth_if_short_by_one_power = max(high_degrees) / max(low_degrees)
    max_growth = math.sqrt(growth_if_short_by_one_power)

    points = _oracle_points(np.float64)
    ratios: dict[int, float] = {}
    for degree in range(1, max(high_degrees) + 1):
        exact = _exact_rows(degree, points)
        deviation = assert_accuracy(
            _tabulate(Backend.CPP, degree, points),
            exact,
            _accuracy_claim(degree, points, exact),
            context=f"cardinal B-spline, degree {degree}, scaling probe",
        )
        ratios[degree] = deviation.max_ratio_to_bound

    low = max(ratios[degree] for degree in low_degrees)
    high = max(ratios[degree] for degree in high_degrees)
    assert high <= max_growth * low, (
        f"the observed error is outgrowing the bound's degree dependence: the "
        f"error-to-bound ratio rises from {low:.4g} at low degree to {high:.4g} at "
        f"high degree, a growth of {high / low:.4g} against the {max_growth:.4g} "
        f"that separates a correct linear bound from one short by a power of the "
        f"degree. A linear-in-degree bound is then the wrong shape, and it will "
        f"fail on correct code at some degree beyond this sweep. Ratios: {ratios}"
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", DEGREES)
def test_partition_of_unity(degree: int, dtype: npt.DTypeLike, cpp_backend: None) -> None:
    """The basis sums to 1 at every point of the span, in both backends.

    An analytic invariant, so it does not care which backend is running or whether
    they agree. It catches a recurrence that lost or duplicated a term while keeping
    every value plausible. The tolerance is the sum of the elementwise forward-error
    bounds plus the roundings of the summation itself, both derived.
    """
    points = _span_points(dtype)
    amplification, error = _companion_bounds(degree, points, dtype)
    # The summation's own roundings, bounded by the magnitudes being summed, which
    # is what the amplification array already is.
    tolerance = error.sum(axis=1) + degree * unit_roundoff(np.float64) * amplification.sum(axis=1)
    for backend in (Backend.CPP, Backend.NUMBA):
        sums = _tabulate(backend, degree, points).astype(np.float64).sum(axis=1)
        assert np.all(np.abs(sums - 1.0) <= tolerance), (
            f"{backend.name}, degree {degree}, {np.dtype(dtype).name}: partition of "
            f"unity is violated by up to {np.max(np.abs(sums - 1.0)):.3e} against a "
            f"derived bound of {np.max(tolerance):.3e}"
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", (1, 2, 3, 5, 8))
def test_reflection_symmetry(degree: int, dtype: npt.DTypeLike, cpp_backend: None) -> None:
    """``N_r(u)`` equals ``N_{degree-r}(1-u)``, in both backends.

    A second analytic invariant, and one that partition of unity cannot see: a
    kernel that wrote its output row reversed still sums to 1. The mirror points are
    exact because the grid is dyadic, so this compares values and not a rounding of
    the argument.
    """
    points = _span_points(dtype)
    mirrored = (1.0 - points.astype(np.float64)).astype(dtype)
    _, error = _companion_bounds(degree, points, dtype)
    _, error_mirrored = _companion_bounds(degree, mirrored, dtype)
    tolerance = error + error_mirrored[:, ::-1]
    for backend in (Backend.CPP, Backend.NUMBA):
        direct = _tabulate(backend, degree, points).astype(np.float64)
        reflected = _tabulate(backend, degree, mirrored).astype(np.float64)[:, ::-1]
        assert np.all(np.abs(direct - reflected) <= tolerance), (
            f"{backend.name}, degree {degree}, {np.dtype(dtype).name}: the basis is not "
            f"symmetric under u -> 1-u, by up to "
            f"{np.max(np.abs(direct - reflected)):.3e} against {np.max(tolerance):.3e}"
        )


# ---------------------------------------------------------------------------
# The seam: Layer 1/2 over the backend selector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_public_api_agrees_across_backends_for_every_input_shape(
    dtype: npt.DTypeLike, cpp_backend: None
) -> None:
    """The seam passes the same data to both backends, whatever shape the caller used.

    The kernels agreeing says nothing about the adapter between them: shape
    flattening and restoration, the dtype dispatch that picks one of the two
    nanobind overloads, and the ``out`` convention all live in
    ``pantr.basis._basis_1D`` and ``pantr.basis._basis_backend``, and none of them is
    exercised
    by calling the Layer 3 kernel directly. This is the only test here that runs the
    path a user runs.
    """
    degree = 3
    inputs: tuple[Any, ...] = (
        np.asarray(0.375, dtype=dtype),
        [0.0, 0.25, 0.5, 1.0],
        np.linspace(0.0, 1.0, 9, dtype=dtype),
        np.linspace(-0.5, 1.5, 12, dtype=dtype).reshape(3, 4),
        np.array([0, 1], dtype=np.int64),
    )
    for points in inputs:
        with use_backend(Backend.CPP):
            from_cpp = tabulate_cardinal_bspline_1d(degree, points)
        with use_backend(Backend.NUMBA):
            from_numba = tabulate_cardinal_bspline_1d(degree, points)
        assert from_cpp.shape == from_numba.shape
        assert from_cpp.dtype == from_numba.dtype

        # The claim needs the points as the kernel saw them: flattened, and in the
        # dtype Layer 2 normalised them to (integers become float64).
        flat = np.ascontiguousarray(np.asarray(points).reshape(-1), dtype=from_cpp.dtype)
        assert flat.size * (degree + 1) == from_cpp.size
        assert_parity(
            from_cpp.reshape(-1, degree + 1),
            from_numba.reshape(-1, degree + 1),
            _parity_claim(degree, flat),
            context=f"public API, degree {degree}, input {np.shape(points)}",
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_out_argument_is_filled_in_place_by_both_backends(
    dtype: npt.DTypeLike, cpp_backend: None
) -> None:
    """A caller-supplied contiguous ``out`` is written, not replaced.

    The ``out`` convention is the project's, not NumPy's doing: Layer 2 validates
    the array and the kernel writes into it. A binding that converted its argument
    would write into a temporary, return normally, and leave the caller's array
    exactly as it was -- a silent wrong answer with no exception anywhere.
    """
    points = np.linspace(0.0, 1.0, 7, dtype=dtype)
    degree = 4
    results = []
    for backend in (Backend.CPP, Backend.NUMBA):
        out = np.full((points.size, degree + 1), np.nan, dtype=dtype)
        with use_backend(backend):
            returned = tabulate_cardinal_bspline_1d(degree, points, out=out)
        assert returned is out, "the out convention returns the caller's own array"
        assert np.all(np.isfinite(out)), f"{backend.name} did not write into the caller's array"
        results.append(out)
    assert_parity(
        results[0],
        results[1],
        _parity_claim(degree, points),
        context=f"public API with out=, degree {degree}, {np.dtype(dtype).name}",
    )


def test_a_non_contiguous_out_is_filled_identically_by_both_backends(
    cpp_backend: None,
) -> None:
    """A strided ``out`` is written, with the same values, whichever backend runs.

    This is the regression test for B1, and it pins **one** of the two halves of
    that fix: the buffering in
    ``pantr.basis._basis_backend._cpp_cardinal_bspline_core``,
    which computes into a contiguous temporary and copies back so that the public
    API accepts the same inputs under either ``PANTR_BACKEND``.

    It does **not** pin the other half. Measured: stripping
    ``nb::arg("out").noconvert()`` from the binding leaves this test passing,
    because the buffering above intercepts every non-contiguous ``out`` before
    nanobind ever sees one. That half is pinned by
    :func:`test_the_binding_refuses_a_non_contiguous_out`, which calls the
    binding directly.

    Layer 2 validates shape, dtype and writability but not contiguity, so a
    strided ``out`` does reach ``_cpp_cardinal_bspline_core`` and this path is
    live.
    """
    points = np.linspace(0.0, 1.0, 5)
    results = []
    for backend in (Backend.NUMBA, Backend.CPP):
        out = np.zeros((5, 6))[:, ::2]
        assert not out.flags["C_CONTIGUOUS"]
        with use_backend(backend):
            returned = tabulate_cardinal_bspline_1d(2, points, out=out)
        assert returned is out, f"{backend.name} did not return the caller's out array"
        assert np.any(out != 0.0), (
            f"the {backend.name} backend returned normally and wrote nothing into "
            f"the caller's non-contiguous out array"
        )
        results.append(out.copy())
    np.testing.assert_array_equal(
        results[0], results[1], err_msg="the two backends disagree on a strided out"
    )


@pytest.mark.parametrize("argument", ("points", "out"))
def test_a_masked_array_is_refused_identically_by_both_backends(
    argument: str, cpp_backend: None
) -> None:
    """A masked array raises on both backends rather than being honoured by neither.

    ``pantr._backend`` promises that selecting a backend changes how fast the
    library is and never what it accepts. Measured before the fix, a masked array
    broke that promise in the worst available direction: Numba raised
    ``NumbaTypeError``, while the C++ backend accepted the array through the
    buffer protocol, ignored the mask and returned ordinary-looking numbers for
    entries the caller had marked invalid. With ``out`` masked it went further
    and wrote through the mask into the caller's own array.

    The fix refuses the type in Layer 2, so neither kernel is reached. Numba's
    behaviour is the correct one of the two, so following it costs nothing.
    """
    masked_points = np.ma.MaskedArray([0.0, 0.25, 0.5, 1.0], mask=[False, True, False, False])
    degree = 2

    for backend in (Backend.CPP, Backend.NUMBA):
        with pytest.raises(TypeError, match="MaskedArray"), use_backend(backend):
            if argument == "points":
                tabulate_cardinal_bspline_1d(degree, masked_points)
            else:
                tabulate_cardinal_bspline_1d(
                    degree,
                    np.asarray(masked_points),
                    out=np.ma.MaskedArray(np.zeros((masked_points.size, degree + 1))),
                )

    # What is refused is the mask, not the data: the escape the message names
    # works, and the two backends still agree on it.
    results = []
    for backend in (Backend.CPP, Backend.NUMBA):
        with use_backend(backend):
            results.append(tabulate_cardinal_bspline_1d(degree, masked_points.filled(0.0)))
    np.testing.assert_array_equal(results[0], results[1])


def test_an_unavailable_backend_request_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit request for a missing backend raises rather than running the other.

    ``pantr._backend`` states this as the reason the selector exists: a silent
    downgrade makes every A/B measurement a measurement of the wrong thing, and the
    measurement is what the override is for. It is untestable as written on a
    machine where both backends exist, so the absence is simulated: the extension
    handle is removed and the two entry points are asked for it anyway.
    """
    from pantr import _backend  # noqa: PLC0415

    monkeypatch.setattr(_backend, "_CPP_AVAILABLE", False)

    assert _backend.available_backends() == (Backend.NUMBA,)
    with pytest.raises(RuntimeError, match="not available"):
        cardinal_bspline_core(Backend.CPP)
    with pytest.raises(RuntimeError, match="not available"), use_backend(Backend.CPP):
        pytest.fail("use_backend must refuse an unavailable backend before yielding")

    # And the Numba backend is still reachable, i.e. the guard rejects rather than
    # disabling the selector.
    assert cardinal_bspline_core(Backend.NUMBA) is not None


def test_overlapping_use_backend_blocks_in_two_threads_do_not_leak(cpp_backend: None) -> None:
    """Two threads whose ``use_backend`` blocks overlap both restore what they found.

    The lost update, forced deterministically with events rather than sleeps: A
    enters, B enters while A is inside, A exits, B exits. Against a plain module
    global, B's saved "previous" is A's override rather than the process default,
    so B restores the process to A's value on the way out and it stays there --
    for the rest of the process, with nothing left in scope to put it back.

    That is not a race in the sense of a rare interleaving. It is the *only*
    outcome of this ordering, which is why the test is deterministic.

    The backend the two threads select is whichever one the process is *not*
    already on, and the assertion is against the ambient value rather than a named
    one. Both matter: this file is run under ``PANTR_BACKEND=numba`` and under
    ``PANTR_BACKEND=cpp``, and a fixed backend makes the test assert nothing under
    one of the two.
    """
    ambient = active_backend()
    other = Backend.NUMBA if ambient is Backend.CPP else Backend.CPP
    entered_a = threading.Event()
    entered_b = threading.Event()
    exited_a = threading.Event()
    timeout = 10.0

    def first() -> None:
        with use_backend(other):
            entered_a.set()
            assert entered_b.wait(timeout), "the second thread never entered its block"
        exited_a.set()

    def second() -> None:
        assert entered_a.wait(timeout), "the first thread never entered its block"
        with use_backend(other):
            entered_b.set()
            assert exited_a.wait(timeout), "the first thread never left its block"

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=timeout)
        assert not thread.is_alive(), "a thread did not finish; the events deadlocked"

    assert active_backend() is ambient, (
        f"two use_backend blocks that overlapped have left the process on "
        f"{active_backend().name} rather than the {ambient.name} it started on, so "
        f"every later call in this process runs a kernel nobody asked for"
    )


def test_use_backend_does_not_reach_into_another_thread(cpp_backend: None) -> None:
    """A thread started inside a ``use_backend`` block runs on the process default.

    The other half, and the one that decides an A/B measurement. The binding
    releases the GIL precisely so a caller can thread at the Python level, so
    "one thread is inside an override while another is working" is the shape this
    module invites rather than a pathological one. Against a module global the
    worker silently switches backend mid-flight, which is exactly the silent
    downgrade ``pantr._backend`` exists to prevent -- only in the other
    direction, and without an exception to notice it by.

    The direction asserted here is the deliberate one: an override is scoped to
    the thread (and the task) that took it, so a thread inherits the process
    default rather than its spawner's override. :func:`use_backend` says so.

    The override is whichever backend the process is *not* already on, so that
    the two values being compared always differ. Naming a fixed backend would
    make this assert nothing whenever ``PANTR_BACKEND`` happened to select it.
    """
    ambient = active_backend()
    other = Backend.NUMBA if ambient is Backend.CPP else Backend.CPP
    observed: list[Backend] = []

    with use_backend(other):
        assert active_backend() is other, "the override did not take in its own thread"
        worker = threading.Thread(target=lambda: observed.append(active_backend()))
        worker.start()
        worker.join(timeout=10.0)
        assert not worker.is_alive(), "the observing thread did not finish"

    assert observed == [ambient], (
        f"a thread started inside a use_backend({other.name}) block observed "
        f"{[b.name for b in observed]} rather than the process default "
        f"{ambient.name}, so the selection is process-wide rather than scoped to "
        f"the block that took it"
    )


def test_kernel_is_reentrant_across_threads(cpp_backend: None) -> None:
    """Concurrent calls on disjoint outputs match the serial results.

    The binding releases the GIL around the kernel, which is what makes this
    reachable at all. It catches shared mutable state in the C++ -- a static
    scratch buffer, a cached span index -- which is invisible to every single-
    threaded test and produces a wrong answer only under load.
    """
    degree = 6
    batches = [np.linspace(start, start + 1.0, 97) for start in (0.0, 0.25, 0.5, 0.75)]
    serial = [_tabulate(Backend.CPP, degree, points) for points in batches]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(lambda p: _tabulate(Backend.CPP, degree, p), batches))
    for index, (one, many) in enumerate(zip(serial, concurrent, strict=True)):
        assert np.array_equal(one, many), (
            f"batch {index}: the C++ kernel gave a different answer when four threads "
            f"ran it at once, so it holds state across calls"
        )


# ---------------------------------------------------------------------------
# The binding's own contract, exercised where the public API cannot reach it
# ---------------------------------------------------------------------------
#
# Every test above goes through `pantr.basis` or through `cardinal_bspline_core`,
# so all of them are separated from the binding by Layer 2 -- which normalises
# the degree, allocates `out`, and buffers a strided one. That is the right thing
# for a user and the wrong thing for a test of the binding: it means none of the
# tests above can observe what the binding does with an argument Layer 2 would
# never pass it. `pantr._pantr_cpp` is an importable module with a public
# attribute, so those arguments are one line of Python away, and two of them
# reached undefined behaviour before these tests existed.


def test_the_binding_refuses_a_negative_degree(cpp_backend: None) -> None:
    """A negative degree is refused before the kernel runs.

    The regression test for a crash rather than for a wrong answer, so its
    failure mode before the fix was not an assertion but the death of the
    process: the kernel does ``static_cast<std::size_t>(degree)``, which turns
    ``-1`` into ``SIZE_MAX``, and then indexes ``out`` that many times. Measured
    before the fix: ``SIGSEGV``, from exactly the call below.

    The exception is a ``TypeError`` rather than a ``ValueError`` because the C++
    parameter is ``unsigned``: nanobind's own caster rejects the value, so no
    pantr code runs at all. That is the reason to spell it that way -- a check
    inside the body can be forgotten, a type cannot.

    The Numba kernel is *not* the model to copy here. Called at Layer 3 with the
    same degree it returns normally, having written nothing, which is only safe
    because NumPy bounds-checks. Both backends agree at the public API, where
    Layer 2 refuses a negative degree before either is reached.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (absent in a Numba-only install)

    points = np.linspace(0.0, 1.0, 4)

    # Python ints and NumPy integer scalars of every width, because the Layer 2
    # caller passes `int(n)` but a direct caller need not: `np.int32(-1)` is the
    # shape a kernel-level caller would actually reach for, and it goes through a
    # different path in nanobind's caster than a Python int does.
    negatives: tuple[Any, ...] = (
        -1,
        -2,
        -(2**31),
        np.int8(-1),
        np.int32(-1),
        np.int64(-1),
    )
    for degree in negatives:
        with pytest.raises(TypeError):
            _pantr_cpp.tabulate_cardinal_bspline_1d(degree, points, np.zeros((4, 1)))

    # A non-integer degree is refused by the same mechanism rather than truncated.
    # The type: ignore is the point of the assertion rather than a workaround:
    # `src/pantr/_pantr_cpp.pyi` declares `degree: int`, and this checks that the
    # binding enforces at run time what the stub only promises statically.
    with pytest.raises(TypeError):
        _pantr_cpp.tabulate_cardinal_bspline_1d(3.0, points, np.zeros((4, 4)))  # type: ignore[arg-type]

    # The public API is where a user meets this, and there it is a ValueError
    # from Layer 2, identically on both backends.
    for backend in (Backend.CPP, Backend.NUMBA):
        with pytest.raises(ValueError, match="degree must be non-negative"), use_backend(backend):
            tabulate_cardinal_bspline_1d(-1, points)


def test_the_binding_refuses_an_out_of_the_wrong_extent(cpp_backend: None) -> None:
    """An ``out`` that does not match ``(points.size, degree + 1)`` is refused.

    The other half of the same crash. nanobind's typed signature pins the dtype,
    the rank and the contiguity of ``out``, but a relation *between* two
    arguments is not something a type can state, so an ``out`` of the right rank
    and the wrong extent passed every check and the kernel wrote past the end of
    it. Measured before the fix: heap corruption, reported by glibc as
    ``corrupted size vs. prev_size`` and an abort.

    The last case is the one a plausible caller hits: ``out`` correct in every
    respect but one column short.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (absent in a Numba-only install)

    degree = 3
    points = np.linspace(0.0, 1.0, 4)
    wrong = (
        np.zeros((2, 2)),  # smaller in both extents
        np.zeros((5, 4)),  # too many rows
        np.zeros((4, 5)),  # too many columns
        np.zeros((4, 3)),  # one column short
    )
    for out in wrong:
        with pytest.raises(ValueError, match="out has shape"):
            _pantr_cpp.tabulate_cardinal_bspline_1d(degree, points, out)
        assert not out.any(), "the binding wrote into an array it went on to reject"

    # The positive control. Without it the four assertions above are consistent
    # with a binding that refuses everything.
    accepted = np.zeros((points.size, degree + 1))
    _pantr_cpp.tabulate_cardinal_bspline_1d(degree, points, accepted)
    assert accepted.any(), "the correctly shaped call was refused too"


def test_the_binding_refuses_a_non_contiguous_out(cpp_backend: None) -> None:
    """A strided ``out`` reaches a ``TypeError``, not a discarded temporary.

    This is the half of B1 that lives in the binding, and it needs a direct call
    to be tested at all: the adapter in ``pantr.basis._basis_backend`` buffers a
    non-contiguous ``out`` before nanobind sees it, so the public-API test cannot
    tell whether ``nb::arg("out").noconvert()`` is present. Verified by removing
    it from a copy of the tree: the rest of the file still passed, and this test
    failed.

    Without ``.noconvert()`` nanobind satisfies the ``c_contig`` constraint by
    CONVERTING rather than by rejecting: the kernel fills a contiguous temporary,
    the temporary is discarded, and the caller's array comes back untouched --
    with no exception anywhere. The second assertion is what distinguishes the
    two: a refusal leaves ``out`` as it was, and so does the silent conversion,
    but only the refusal raises.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (absent in a Numba-only install)

    points = np.linspace(0.0, 1.0, 5)
    out = np.zeros((5, 6))[:, ::2]
    assert not out.flags["C_CONTIGUOUS"], "the case this test exists for is not being built"

    with pytest.raises(TypeError):
        _pantr_cpp.tabulate_cardinal_bspline_1d(2, points, out)
    assert not out.any(), "the binding wrote into a strided out it should have refused"


# ---------------------------------------------------------------------------
# The harness itself: can these assertions fail?
# ---------------------------------------------------------------------------


def test_point_sets_reach_the_cases_they_are_named_for() -> None:
    """The constructed point sets really contain the boundary and outside cases.

    A generator is only worth what it actually produces, and these are edited by
    hand. Without this, a later edit that dropped the span ends would leave every
    boundary test passing on interior points.
    """
    for dtype in DTYPES:
        boundary = _boundary_points(dtype)
        assert np.any(boundary == 0.0) and np.any(boundary == 1.0)
        assert np.any(boundary < 0.0), "no point below the span"
        assert np.any(boundary > 1.0), "no point above the span"
        outside = _outside_points(dtype)
        assert np.all((outside < 0.0) | (outside > 1.0))
        span = _span_points(dtype)
        assert span[0] == 0.0 and span[-1] == 1.0
        oracle = _oracle_points(dtype)
        assert np.any(oracle < 0.0) and np.any(oracle > 1.0), "oracle set stays on the span"
        mirrored = 1.0 - span.astype(np.float64)
        assert np.array_equal(np.sort(mirrored), np.sort(span.astype(np.float64))), (
            "the span grid is not closed under u -> 1-u, so the reflection test would "
            "be comparing values at points that are not exact mirrors"
        )


def test_the_rounding_grid_actually_rounds() -> None:
    """The dense grid really distinguishes a float64 accumulator from a float32 one.

    :func:`test_float32_intermediates_are_float64` is only as good as its point
    set: on a dyadic grid the two accumulator models agree at low degree and the
    check silently proves nothing. This pins the property the grid was chosen for,
    separately from the test that relies on it, so a later edit to the grid fails
    here rather than quietly hollowing out that test.
    """
    points = _rounding_points(np.float32)
    for degree in (1, 2, 3):
        wide = _arithmetic_contract_model(degree, points, np.float64)
        narrow = _arithmetic_contract_model(degree, points, np.float32)
        differing = int(np.count_nonzero(wide != narrow))
        assert differing > 10, (
            f"degree {degree}: only {differing} of {wide.size} entries distinguish a "
            f"float64 accumulator from a float32 one on this grid"
        )


def test_bitwise_claim_detects_a_one_ulp_difference() -> None:
    """A bitwise claim fails on the smallest possible disagreement.

    The sensitivity probe for the strongest assertion in the file: if this passed,
    every bitwise test above would be certifying nothing.
    """
    points = _span_points(np.float64)
    reference = _tabulate(Backend.NUMBA, 3, points)
    perturbed = reference.copy()
    perturbed[2, 1] = np.nextafter(perturbed[2, 1], np.float64(np.inf))
    with pytest.raises(AssertionError, match="bitwise parity claimed and violated"):
        assert_parity(
            perturbed,
            reference,
            bitwise_parity(why="probe"),
            context="sensitivity probe",
        )


def test_bounded_branch_admits_a_perturbation_at_the_bound() -> None:
    """The bounded branch accepts a difference just inside the bound it derives.

    The FMA branch of :func:`_parity_claim` is not taken on a baseline build, so
    without this it would ship unexecuted and its first run would be the day
    ``-march=x86-64-v3`` lands. Driving the harness directly exercises it now.
    """
    points = _span_points(np.float64)
    reference = _tabulate(Backend.NUMBA, 5, points)
    amplification, _ = _companion_bounds(5, points, np.float64)
    claim = bounded_parity(
        roundings=Roundings(stages=5, accumulator_per_stage=2, storage_per_stage=1),
        accumulator=np.float64,
        storage=np.float64,
        amplification=amplification,
        why="probe: the same budget the FMA branch of _parity_claim builds",
    )
    tolerance = absolute_tolerance(claim)
    assert claim.kind is ParityKind.BOUNDED
    assert np.max(tolerance) > 0.0, "the bounded branch derived a zero tolerance"
    perturbed = reference + 0.75 * tolerance
    assert_parity(perturbed, reference, claim, context="sensitivity probe, inside the bound")


def test_bounded_branch_rejects_a_perturbation_past_the_bound() -> None:
    """The bounded branch fails on a difference just outside the bound it derives.

    The other half of the probe. A tolerance nothing can exceed is not a tolerance.
    """
    points = _span_points(np.float64)
    reference = _tabulate(Backend.NUMBA, 5, points)
    amplification, _ = _companion_bounds(5, points, np.float64)
    claim = bounded_parity(
        roundings=Roundings(stages=5, accumulator_per_stage=2, storage_per_stage=1),
        accumulator=np.float64,
        storage=np.float64,
        amplification=amplification,
        why="probe: the same budget the FMA branch of _parity_claim builds",
    )
    perturbed = reference + 1.5 * absolute_tolerance(claim)
    with pytest.raises(AssertionError, match="more than the derived bound"):
        assert_parity(perturbed, reference, claim, context="sensitivity probe, past the bound")


def test_a_tolerance_cannot_be_stated_without_a_derivation() -> None:
    """Every way of building a claim without a derivation is refused.

    The harness's whole purpose is that the next five ported modules cannot express
    an underived tolerance. That property is a property of the API, so it is tested
    like any other.
    """
    with pytest.raises(ValueError, match="why"):
        bitwise_parity(why="   ")
    with pytest.raises(ValueError, match="derivation"):
        bounded_parity(
            roundings=Roundings(stages=3, accumulator_per_stage=2, storage_per_stage=0),
            accumulator=np.float64,
            storage=np.float64,
            amplification=np.ones((2, 2)),
            why="",
        )
    with pytest.raises(ValueError, match="bitwise_parity"):
        bounded_parity(
            roundings=Roundings(stages=0, accumulator_per_stage=2, storage_per_stage=0),
            accumulator=np.float64,
            storage=np.float64,
            amplification=np.ones((2, 2)),
            why="zero stages is not a bound",
        )
    with pytest.raises(ValueError, match="finite"):
        bounded_parity(
            roundings=Roundings(stages=3, accumulator_per_stage=2, storage_per_stage=0),
            accumulator=np.float64,
            storage=np.float64,
            amplification=np.array([[np.inf]]),
            why="an overflowed amplification makes the comparison vacuous",
        )


def test_a_budget_that_exhausts_the_format_is_refused() -> None:
    """A bound that accepts every finite result is reported, not returned.

    The failure mode a "derived" tolerance still permits: derive it for a degree so
    large, or a format so narrow, that the bound exceeds 1 in relative terms and the
    comparison stops meaning anything. That is worth an error rather than a pass.
    """
    claim = bounded_parity(
        roundings=Roundings(stages=10**8, accumulator_per_stage=2, storage_per_stage=1),
        accumulator=np.float64,
        storage=np.float32,
        amplification=np.ones((2, 2)),
        why="probe: a budget large enough to exhaust float32",
    )
    with pytest.raises(ValueError, match="vacuous"):
        absolute_tolerance(claim)
