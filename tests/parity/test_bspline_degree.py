"""Parity of the two degree operations: `Bspline.derivative` and `.elevate_degree`.

Like :mod:`tests.parity.test_bspline_refinement` and unlike
:mod:`tests.parity.test_bspline_type`, what this file compares is a **computation**: a
control net is pushed through a difference quotient or through Piegl and Tiller A5.9, so
there is real arithmetic on every coefficient and the criterion has to be argued rather
than assumed. It is argued per quantity in :func:`_fields`, not once in bulk.

## The criterion, and the one claim here that does not depend on the host

**Bitwise, for the coefficients and the knots, on both operations.** The two backends run
the same IEEE-754 operations in the same order; ``cpp/include/pantr/bspline/degree.hpp``
carries the argument beside the code and it differs between the two.

**The hodograph's claim is a property of the code rather than of the host, and that is
worth saying because both sibling ports say the opposite of their own kernels.**
``_derivative_ctrl_1d`` is plain numpy rather than a numba kernel, so NEP 50 decides its
widths: the knot and coefficient differences are array expressions in the storage format,
the Python ``int`` degree is *weak* and is converted to that format rather than promoting
it, and ``np.divide``'s output is allocated at the control points' dtype. Every rounding
is therefore in ``T``, and the expression is a difference, a scaling and a division --
**no sum of a product, so no site in it can contract**. ``design/backend_parity.md``
Rule 7 is what makes that remark necessary rather than decorative: a bitwise claim is
normally about the build, and this one is not.

**A5.9's claim is about this build, per Rule 7.** Its inner statements are
``tmp1 + tmp2`` with both terms products, and ``ebpts[i, ii] += bezalfs[j, i] *
bpts[j, ii]``; a target with a fused multiply-add would contract them. On the baseline
x86-64 this project compiles for there is none.
:func:`~tests._parity_harness.contraction_may_fuse` is the gate, and where it reports a
fusing build the claim is skipped rather than weakened -- Rule 10's budget would apply
and no host here can execute that branch to check it.

**The widths inside A5.9 are three, they vary within the kernel, and Rule 9 is sharper
here than it is anywhere else in the port.** Numba types each SSA version of a variable
separately, so ``tmp1``'s four assignments do not agree: ``alfs[q - s] * bpts[q, ii]``
reads a ``float64`` table and is ``float64``, while ``alf * ic[i, ii]``,
``gam * ebpts[kj, ii]`` and ``bet * ebpts[kj, ii]`` read knot quotients that are
themselves the storage format and are ``float32`` on a ``float32`` net. ``ebpts``
narrows on **every term** of its accumulation rather than once at the end, and the Boehm
quotient divides narrow and widens only on the store into a ``float64`` table.
``scripts/measure_bspline_degree_widths.py`` prints numba's own inferred type per site
and runs a rival model per site against the kernel; none of it is visible at ``float64``,
which is Rule 9's own warning.

**The interpreted oracle differs for the elevation and not for the hodograph, and the
mechanism is Rule 12's first divergence reached by a route that rule does not name.**
Rule 12 lists three things that change under ``NUMBA_DISABLE_JIT=1`` -- a storage width,
``pow``, and an integer accumulator that wraps when compiled -- and attributes the width
one to ``float()``, which numba's ``nopython`` mode does not treat as a promotion and
CPython does. A5.9 contains no ``float()`` call, and its width moves anyway:
``tmp2 = (1.0 - alf) * ic[i - 1, ii]`` carries a **Python float literal** against an
``alf`` that is the storage format, and numba types that literal ``float64`` while NEP 50
makes it *weak* and lets the ``float32`` operand decide. So the product is ``float64``
compiled and ``float32`` interpreted, and the two answers are different numbers. The same
applies to the `ebpts` blend's companion.

Measured, with the JIT disabled: every ``float64`` elevation case in this file's
field-by-field comparison still agrees bit for bit, and five of the nine ``float32`` ones
no longer do. The four that survive are the four whose knot vectors never make
``oldr > 1`` -- a single Bézier span, a ``C^0`` breakpoint, a ``C^-1`` breakpoint and a
degree-2 direction -- so they never execute the block the divergent literal is in. That
the split falls exactly along that line, and along the storage format rather than along
the algorithm, is what identifies the cause as a width.
:func:`~tests._parity_harness.demand_the_compiled_kernel` is the gate that owns exactly
that question, and it is applied to the elevation.

The hodograph needs no gate and that is a guarantee rather than an observation:
``_derivative_ctrl_1d`` is plain numpy and is **not compiled in either configuration**,
so it is the same object under both. Its own weak operand -- the Python ``int`` degree --
is weak to numba too, the kernels being absent, so nothing about it moves. Neither kernel
carries ``pow``, and the only integer accumulator in reach is ``_bincoeff``'s, which casts
each step to ``np.int64`` explicitly and so wraps identically either way; Rule 12 names
that kernel as its deliberate counter-example.

## The independent check, and what it covers

``design/backend_parity.md`` opens with the fact everything else follows from: parity
says the two backends agree and not that either is right, so a shared error is invisible
to every comparison above.

**Marsden's identity.** For a knot vector ``t`` and degree ``p``,

    u^r = sum_i [ e_r(t_{i+1}, ..., t_{i+p}) / C(p, r) ] N_{i,p}(u),

with ``e_r`` the elementary symmetric polynomial, so the B-spline coefficients of a
monomial are a closed form in the knots alone. A field carrying those numbers *is* the map
``u -> u^r``, and each operation has to move it to the right place:

- **elevation** must reproduce the same closed form on the **elevated** vector at the
  **elevated** degree, since it does not change the map;
- **the hodograph** must reproduce ``r u^{r-1}``'s closed form on ``t[1:-1]`` at degree
  ``p - 1``. That one is an identity rather than a statement about the answer, provable
  from ``e_r(S + {x}) = e_r(S) + x e_{r-1}(S)`` in two lines --
  ``cpp/tests/test_bspline_degree.cpp``'s file comment writes them out -- and it holds
  for any knot vector, clamped or not.

Taking every ``r`` in ``[0, p]`` pins each row completely rather than up to an affine map;
``r = 1`` alone is the Greville abscissa and ``r = 0`` alone is the partition of unity.
Nothing in :func:`_marsden_column` consults either backend.

## The accuracy bounds, and why the elevation's is a majorant

**The hodograph.** ``gamma_K`` times an elementwise amplification, with ``K`` counted:
``2p + 1`` for the input window's ``e_r`` recurrence and ``2p - 1`` for the derived one at
degree ``p - 1``, one for the cast of the closed form into the field's storage, four for
the formula itself -- the coefficient difference, the knot difference, the scaling by the
degree and the division -- one for the multiply by ``r`` that turns the derived window's
closed form into ``r u^(r-1)``'s, and one for the store. ``K = 4p + 7``, which is what
:func:`_hodograph_accuracy` computes. The magnitude is **not** the result's own:
``A_{i+1} - A_i`` can cancel to nothing while its operands do not, so the relative budget
is charged against what was subtracted,

    amplification_i = p (|A_i| + |A_{i+1}|) / |t_{i+p+1} - t_{i+1}|,

which majorises the computed value and every partial result that fed it, there being only
one subtraction, one scaling and one division.

**The elevation, and here the obvious amplification is wrong rather than loose.** The
finished elevation operator is non-negative with rows summing to one -- degree elevation
writes each ``N_{i,p}`` as a non-negative combination of the elevated basis -- so every
*output* coefficient is a convex combination and ``max |c|`` bounds it. A rounding budget
is charged against the **intermediates**, though, and A5.9's are not convex combinations
of anything: its knot-removal step blends with ``alf = (ub - ik[i]) / (ua - ik[i])``,
whose numerator exceeds its denominator whenever ``ik[i] < ua < ub``, so ``alf > 1`` and
``1 - alf < 0``. That is ``design/backend_parity.md`` Rule 10's recorded shortfall met on
a second kernel, and its answer is what :func:`_elevate_majorant` uses: **run the
recursion on the moduli of the coefficients and of the weights**. For a linear recursion
with signed coefficients that bounds every intermediate by induction, and with the signs
gone the partial sums are monotone, so the final value majorises all of them. It computes
no value this file compares against -- those come from Marsden -- only a magnitude.

``K`` for the elevation: the two ``e_r`` recurrences at degree ``p`` and ``p + t``, the
cast in and the store out, and the kernel's own chain, whose **three blocks do not cost
the same** and are counted separately. At most ``p - 1`` Boehm insertion passes, each
``bpts[q] = alf * bpts[q] + (1 - alf) * bpts[q-1]`` -- a subtraction, two products, an
addition and a narrowing store, so five; then ``min(p, t) + 1`` terms accumulated into one
elevated Bézier coefficient, each a product, an addition and a narrowing store, so three;
then at most ``p - 2`` knot-removal passes, the same shape as a Boehm pass, so five. The
middle count is Rule 10's for the Bézier elevation and it is right here for the same
reason -- the accumulation into ``ebpts[i]`` runs ``j`` from ``max(0, i - t)`` to
``min(p, i)`` -- and charging ``p + 1`` instead is the over-count that rule records.
Charging one flat number across all three blocks under-charges the two blend blocks, which
is a defect a review found in an earlier version of this file.

Both bounds add ``K`` underflow floors, the absolute half of Higham's model, because a
closed form can be an exact zero and a relative bound of zero asserts bit-identity
nothing here has grounds for.

## The four places the backends do not meet

Four, counted as :mod:`pantr.bspline._degree_backend` counts them, and all four are
recorded there and pinned here rather than left to be met. Two are boundaries of the
port, one is a defect in the oracle that the port deliberately does not inherit and
deliberately does not change, and one is an asymmetry between the two sides' own argument
checking.

- **`keep_degree=True` and a rational derivative run the oracle**, because the oracle's
  own answers on those paths are under review and there is nothing stable to be at parity
  with. ``keep_degree`` is absent from the C++ signature outright; a rational field is
  refused by the binding, which
  :func:`test_a_rational_derivative_is_refused_by_the_binding_and_served_by_the_oracle`
  pins.
- **A periodic direction being elevated runs the oracle**, because elevating one
  round-trips through ``_to_periodic_bspline_1d_impl``, which
  ``cpp/include/pantr/bspline/structural.hpp`` declares as its own boundary. The
  derivative does **not** share that boundary and serves a periodic direction in C++;
  :func:`test_the_periodic_hodograph_reaches_the_cpp_path` is the other half of that rule,
  which an over-broad condition would get wrong.
- **An unclamped direction being elevated runs the oracle, and the oracle is wrong
  there.** A5.9 walks segments until a run of equal knots reaches the last index of the
  vector; an unclamped vector has none, so the walk reads past the control array. Numba
  does not bounds check in ``nopython`` mode, so it returns whatever the read found. The
  C++ half refuses rather than reproducing undefined behaviour, and the routing sends the
  call to the oracle anyway so that what the library accepts does not change with
  ``PANTR_BACKEND``.
  :func:`test_an_unclamped_elevation_fails_the_same_way_under_both_backends` pins the
  oracle's current behaviour on purpose, so that fixing it is a visible change rather than
  a silent one -- including that the symptom depends on the configuration: compiled, the
  walk returns and the field's constructor refuses the coefficient count; interpreted,
  numpy bounds checks and the read itself raises ``IndexError``.
- **An increment tuple with nothing to elevate runs the oracle.** The C++ half refuses an
  all-zero or negative one with :meth:`~pantr.bspline.Bspline.elevate_degree`'s own Layer 1
  message while ``_degree_elevate_bspline`` never checks and returns the field unchanged.
  No public caller reaches it, the method refusing such an argument first;
  :func:`test_an_argument_with_nothing_to_elevate_goes_to_the_oracle` pins the private
  entry point, which a downstream consumer of this package's private symbols could call.
"""

from __future__ import annotations

import math
from typing import Any, Final, NamedTuple

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.bspline._bspline_degree_core import _bincoeff
from pantr.bspline._degree_backend import elevate_field_degree
from tests._parity_harness import (
    AccuracyClaim,
    Field,
    assert_accuracy,
    assert_object_parity,
    bitwise_parity,
    contraction_may_fuse,
    demand_the_compiled_kernel,
    derived_accuracy,
    exact_parity,
    the_jit_is_disabled,
    underflow_floor,
    unit_roundoff,
)

pytestmark = pytest.mark.usefixtures("cpp_backend")

DTYPES: Final = (np.float64, np.float32)
"""Both storage formats: a field stores `float32` too, so the C++ side has two classes."""

_FUSING: Final = (
    "this build's target ISA has a fused multiply-add, so A5.9's `tmp1 + tmp2` and its "
    "`ebpts[i, ii] += bezalfs[j, i] * bpts[j, ii]` may contract and the bitwise claim "
    "does not hold. design/backend_parity.md Rule 10's budget would apply; no host in "
    "this project can execute that branch to check it, so it is skipped rather than "
    "written blind."
)
"""Skip reason for the elevation's bitwise claims on a build that can fuse."""


# ---------------------------------------------------------------------------
# The independent oracle: Marsden's identity
# ---------------------------------------------------------------------------


def _marsden_column(
    knots: npt.NDArray[np.float32 | np.float64], degree: int, power: int
) -> npt.NDArray[np.float64]:
    """The B-spline coefficients of ``u ** power`` over one knot vector.

    ``A_i = e_power(knots[i+1], ..., knots[i+degree]) / C(degree, power)``; see the
    module docstring. Formed in ``float64`` whatever the knots are stored in, by the
    elementary-symmetric recurrence, so nothing here re-runs either backend.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): The knot vector.
        degree (int): The polynomial degree.
        power (int): The monomial power, in ``[0, degree]``.

    Returns:
        npt.NDArray[np.float64]: One coefficient per basis function,
        ``len(knots) - degree - 1`` of them.
    """
    count = len(knots) - degree - 1
    out = np.zeros(count, dtype=np.float64)
    for i in range(count):
        symmetric = np.zeros(degree + 1, dtype=np.float64)
        symmetric[0] = 1.0
        for k in range(1, degree + 1):
            knot = float(knots[i + k])
            for j in range(k, 0, -1):
                symmetric[j] += knot * symmetric[j - 1]
        out[i] = symmetric[power] / float(math.comb(degree, power))
    return out


def _elevate_majorant(  # noqa: PLR0912, PLR0915
    degree: int,
    magnitudes: npt.NDArray[np.float64],
    knots: npt.NDArray[np.float64],
    increment: int,
) -> npt.NDArray[np.float64]:
    """Run A5.9 on the moduli of its coefficients and of its blending weights.

    The amplification the elevation's bound multiplies. A linear recursion with signed
    coefficients is bounded term by term by the same recursion on the moduli, by
    induction; with the signs gone every partial sum is monotone, so the value returned
    for an output coefficient majorises every intermediate that fed it. See the module
    docstring for why a companion built from the finished operator does not.

    It computes no value this file compares against, only a magnitude.

    Args:
        degree (int): The original degree.
        magnitudes (npt.NDArray[np.float64]): One non-negative magnitude per
            coefficient.
        knots (npt.NDArray[np.float64]): The knot vector.
        increment (int): Degrees to add.

    Returns:
        npt.NDArray[np.float64]: One majorant per elevated coefficient.
    """
    mag = np.abs(np.asarray(magnitudes, dtype=np.float64))
    kn = np.asarray(knots, dtype=np.float64)
    num_rows = mag.shape[0]
    d, t = degree, increment
    ph, ph2 = d + t, (d + t) // 2
    m = (num_rows - 1) + d + 1

    bezalfs = np.zeros((d + 1, ph + 1))
    alfs = np.zeros(max(d, 1))
    bpts = np.zeros(d + 1)
    ebpts = np.zeros(ph + 1)
    nextbpts = np.zeros(d + 1)

    bezalfs[0, 0] = 1.0
    bezalfs[d, ph] = 1.0
    for i in range(1, ph2 + 1):
        inv = 1.0 / _bincoeff(ph, i)
        for j in range(max(0, i - t), min(d, i) + 1):
            bezalfs[j, i] = inv * _bincoeff(d, j) * _bincoeff(t, i - j)
    for i in range(ph2 + 1, ph):
        for j in range(max(0, i - t), min(d, i) + 1):
            bezalfs[j, i] = bezalfs[d - j, ph - i]

    kind, r, a, b, cind = ph + 1, -1, d, d + 1, 1
    ua = kn[0]
    ik = np.zeros(len(kn) + t * len(kn))
    ic = np.zeros(num_rows + t * len(kn))
    ic[0] = mag[0]
    ik[: ph + 1] = ua
    bpts[: d + 1] = mag[: d + 1]

    while b <= m:
        run_start = b
        while b < m and kn[b] == kn[b + 1]:
            b += 1
        mul = b - run_start + 1
        ub = kn[b]
        oldr, r = r, d - mul
        if oldr > 0:
            lbz = (oldr + 2) // 2
        elif oldr < 0 and a != d:
            lbz = 0
        else:
            lbz = 1
        rbz = ph - (r + 1) // 2 if r > 0 else ph

        if r > 0:
            numer = ub - ua
            for q in range(d, mul, -1):
                alfs[q - mul - 1] = numer / (kn[a + q] - ua)
            for j in range(1, r + 1):
                save, s = r - j, mul + j
                for q in range(d, s - 1, -1):
                    bpts[q] = abs(alfs[q - s]) * bpts[q] + abs(1.0 - alfs[q - s]) * bpts[q - 1]
                nextbpts[save] = bpts[d]

        for i in range(lbz, ph + 1):
            ebpts[i] = 0.0
            for j in range(max(0, i - t), min(d, i) + 1):
                ebpts[i] += abs(bezalfs[j, i]) * bpts[j]

        if oldr > 1:
            first, last = kind - 2, kind
            den = ub - ua
            bet = (ub - ik[kind - 1]) / den
            for tr in range(1, oldr):
                i, j = first, last
                kj = j - kind + 1
                while j - i > tr:
                    if i < cind:
                        alf = (ub - ik[i]) / (ua - ik[i])
                        ic[i] = abs(alf) * ic[i] + abs(1.0 - alf) * ic[i - 1]
                    if j >= lbz:
                        weight = (ub - ik[j - tr]) / den if j - tr <= kind - ph + oldr else bet
                        ebpts[kj] = abs(weight) * ebpts[kj] + abs(1.0 - weight) * ebpts[kj + 1]
                    i, j, kj = i + 1, j - 1, kj - 1
                first, last = first - 1, last + 1

        if a != d:
            for _ in range(ph - oldr):
                ik[kind] = ua
                kind += 1
        for j in range(lbz, rbz + 1):
            ic[cind] = ebpts[j]
            cind += 1
        if b < m:
            bpts[:r] = nextbpts[:r]
            for j in range(max(0, r), d + 1):
                bpts[j] = mag[b - d + j]
            a, b, ua = b, b + 1, ub
        else:
            ik[kind : kind + ph + 1] = ub
            break
    return ic[:cind].copy()


# ---------------------------------------------------------------------------
# The case table
# ---------------------------------------------------------------------------


class _Case(NamedTuple):
    """One field to operate on, and what to do to it.

    Attributes:
        label (str): The id a parametrized test reports.
        knots (tuple[tuple[float, ...], ...]): One knot vector per direction.
        degrees (tuple[int, ...]): One degree per direction.
        periodic (tuple[bool, ...]): Whether each direction wraps.
        increments (tuple[int, ...]): Degrees to add per direction, for the elevation.
        direction (int): The direction to differentiate.
        is_rational (bool): Whether the last component is a weight.
    """

    label: str
    knots: tuple[tuple[float, ...], ...]
    degrees: tuple[int, ...]
    periodic: tuple[bool, ...]
    increments: tuple[int, ...]
    direction: int
    is_rational: bool


CASES: Final = (
    _Case(
        "cubic curve, four simple knots",
        ((0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.7, 1.0, 1.0, 1.0, 1.0),),
        (3,),
        (False,),
        (2,),
        0,
        False,
    ),
    _Case(
        "quadratic curve, a C^0 breakpoint",
        ((0.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0, 1.0, 1.0),),
        (2,),
        (False,),
        (1,),
        0,
        False,
    ),
    _Case(
        "cubic curve, a C^-1 breakpoint",
        ((0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0),),
        (3,),
        (False,),
        (1,),
        0,
        False,
    ),
    _Case(
        "one quartic Bezier span",
        ((0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0),),
        (4,),
        (False,),
        (3,),
        0,
        False,
    ),
    _Case(
        "surface, direction 1 differentiated and elevated",
        (
            (0.0, 0.0, 0.0, 0.4, 1.0, 1.0, 1.0),
            (-1.0, -1.0, -1.0, -1.0, 1.0 / 7.0, 2.0, 2.0, 2.0, 2.0),
        ),
        (2, 3),
        (False, False),
        (0, 2),
        1,
        False,
    ),
    _Case(
        "rational surface, direction 0 only",
        (
            (0.0, 0.0, 0.0, 0.0, 0.3, 1.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 1.0 / 7.0, 1.0, 1.0),
        ),
        (3, 1),
        (False, False),
        (1, 0),
        0,
        True,
    ),
    _Case(
        "volume, the middle direction",
        (
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 0.3, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 0.6, 1.0, 1.0, 1.0, 1.0),
        ),
        (1, 2, 3),
        (False, False, False),
        (0, 1, 0),
        1,
        False,
    ),
    _Case(
        "quintic curve, three simple interior knots",
        ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 0.6, 0.82, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),),
        (5,),
        (False,),
        (1,),
        0,
        False,
    ),
    _Case(
        "cubic curve, a span 500 times narrower than its neighbour",
        ((0.0, 0.0, 0.0, 0.0, 0.001, 0.5, 1.0, 1.0, 1.0, 1.0),),
        (3,),
        (False,),
        (1,),
        0,
        False,
    ),
    _Case(
        "periodic curve",
        ((-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),),
        (2,),
        (True,),
        (1,),
        0,
        False,
    ),
    _Case(
        "unclamped curve",
        ((-0.3, -0.2, -0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.1, 1.2, 1.3),),
        (3,),
        (False,),
        (1,),
        0,
        False,
    ),
)
"""The shapes the two operations are compared over.

Two of them are here for A5.9's knot-removal block, which a table of cubics does not
reach even though it runs, and the reasons are worth writing down because they are the
difference between a bitwise claim that pins a width and one that cannot.

**The quintic is the only case whose `ebpts` blends survive to the output.** That block
modifies `ebpts[kj]` with `kj` running 1, 2, 3, ... over its `tr` passes, while the
segment's coefficients are emitted from `lbz = (oldr + 2) // 2` upwards. At degree 3 with
simple interior knots `oldr` is 2 and `lbz` is 2, so the only write the block makes --
`kj = 1` -- is to a coefficient never emitted, and a variant of
`cpp/include/pantr/bspline/degree.hpp` computing either of that blend's two terms at the
wrong width passed this whole file. `kj` first reaches `lbz` at **degree 4** (`oldr = 3`,
`lbz = 2`, and the `tr = 2` pass writes `kj = 2`); the quintic is here because no other
case in this table is a curve with several simple interior knots above degree 3, not
because degree 4 could not have served.

**The narrow span is for the bound rather than for a width**: it takes A5.9's blending
weights far from one, which is where the majorant the accuracy check uses has to earn its
place against a convex companion. It is *not* what makes the `ic` blend's own companion
observable, and nothing in reach is: that one forms `1 - alf` with
`alf = (ub - ik[i]) / (ua - ik[i])`, and `ik[i] <= ua < ub` makes `alf >= 1`, for which the
subtraction is exact in `float32` **while `alf < 2^24`** -- above that the exact difference
needs a bit `float32` no longer has, and `float32(1) - 16777218` is off by one. The weights
these knot vectors produce top out near 500, so the hypothesis holds by a wide margin here;
it is stated rather than assumed because the claim is otherwise false. Computing that
companion in the storage format is therefore a choice with no consequence, which is why no
case in this table tries to catch it.
"""


def _spaces(case: _Case, dtype: npt.DTypeLike) -> list[BsplineSpace1D]:
    """The univariate spaces of one case.

    Args:
        case (_Case): The shape to build.
        dtype (npt.DTypeLike): The storage format.

    Returns:
        list[BsplineSpace1D]: One space per direction, in axis order.
    """
    return [
        BsplineSpace1D(np.array(knots, dtype=dtype), degree, periodic=periodic)
        for knots, degree, periodic in zip(case.knots, case.degrees, case.periodic, strict=True)
    ]


def _make_field(case: _Case, dtype: npt.DTypeLike) -> Bspline:
    """Build one case's field, with control points that are not a symmetric pattern.

    The coefficients are a fixed deterministic sequence rather than the Marsden net: the
    field-by-field comparison wants data that would expose a transposition or a dropped
    row, and the accuracy check builds its own field from the closed form.

    Args:
        case (_Case): The shape to build.
        dtype (npt.DTypeLike): The storage format.

    Returns:
        ~pantr.bspline.Bspline: The field.
    """
    spaces = _spaces(case, dtype)
    space = BsplineSpace(spaces)
    rank = 2 + int(case.is_rational)
    shape = (*(s.num_basis for s in spaces), rank)
    count = math.prod(shape)
    # A sequence with no repeats and no symmetry, so a permuted or dropped coefficient
    # shows up as a value rather than only as a count -- and **not dyadic**, which is
    # load-bearing rather than decorative. Coefficients of the form `k / 16` are exact in
    # `float32`, so their differences are exact too and the hodograph's quotient is then
    # the one case where computing it in `double` and narrowing once agrees with
    # computing it narrow. A bitwise claim over such data is blind to the width it exists
    # to pin: measured, a `double` variant of `derivative_along_axis` passed the whole of
    # this file before the divisor below was 97 rather than 16.
    # `test_the_case_table_earns_its_keep` asserts the values really do round.
    values = np.array([(7.0 + ((i * 37) % 101)) / 109.0 for i in range(count)], dtype=dtype)
    control_points = values.reshape(shape)
    if case.is_rational:
        # Weights must be positive, and distinct from each other.
        control_points = control_points.copy()
        control_points[..., -1] = np.abs(control_points[..., -1]) + 1.0
    return Bspline(space, control_points, is_rational=case.is_rational)


def _elevatable(case: _Case) -> bool:
    """Whether every direction this case elevates is one the C++ half will take.

    Args:
        case (_Case): The shape.

    Returns:
        bool: ``True`` when no elevated direction is periodic or unclamped.
    """
    spaces = _spaces(case, np.float64)
    return all(
        increment == 0 or (not space.periodic and space.has_open_knots())
        for increment, space in zip(case.increments, spaces, strict=True)
    )


# ---------------------------------------------------------------------------
# The field-by-field comparison
# ---------------------------------------------------------------------------


def _fields(dim: int, *, elevated: bool) -> tuple[Field, ...]:
    """Every piece of a result field's state, one field each, with its own argument.

    Args:
        dim (int): The field's number of parametric directions.
        elevated (bool): Whether the result came from the elevation rather than the
            derivative, which changes the coefficients' argument and nothing else.

    Returns:
        tuple[Field, ...]: The field list for
        :func:`~tests._parity_harness.assert_object_parity`.
    """
    coefficients_why = (
        (
            "the elevated coefficients. The two backends run A5.9's operations in the "
            "same order and at the same widths: the Bezier coefficient table entirely "
            "in `float64`, the Boehm quotient divided in the storage format and widened "
            "only on its store into that table, the segment blend's two products both "
            "`float64` and narrowed on the store, the elevated-Bezier accumulation "
            "narrowed on *every* term rather than once, and the knot-removal blend's "
            "first product in the storage format against a `float64` companion. See "
            "the module docstring for why those are read off numba's own type inference "
            "rather than off the source, and for the build this claim depends on."
        )
        if elevated
        else (
            "the hodograph's coefficients, and the only quantity here arithmetic "
            "touches. `p * (c[i+1] - c[i]) / (k[i+p+1] - k[i+1])`, three roundings, all "
            "in the storage format because NEP 50 converts the weak Python `int` degree "
            "to the array's dtype rather than promoting it and `np.divide`'s output is "
            "allocated at the control points'. A row whose denominator vanishes keeps "
            "the exact zero it was allocated with on both sides, the quotient never "
            "being formed. There is no sum of a product anywhere in it, so unlike every "
            "other bitwise claim in this port it does not depend on the target ISA."
        )
    )
    per_direction: list[Field] = []
    for direction in range(dim):
        per_direction.append(
            Field(
                f"space.spaces[{direction}].knots",
                bitwise_parity(
                    why=(
                        "the result's knot vector, and no arithmetic touches a knot in "
                        "either operation: the hodograph drops the first and last entry "
                        "of the input vector, and A5.9 writes copies of `ua` and `ub`, "
                        "which are elements of it. A difference could only be a lost "
                        "value, a miscount or a narrowing cast. The space then snaps "
                        "near duplicates, identically on both sides, from identical "
                        "input."
                    )
                ),
                read=lambda field, d=direction: field.space.spaces[d].knots,  # type: ignore[misc]
            )
        )
        per_direction.append(
            Field(
                f"space.spaces[{direction}].tolerance",
                bitwise_parity(
                    why=(
                        "the result's space derives its own tolerance from its knots, "
                        "so it is a new quantity of the result rather than one carried "
                        "over, and it is what every later domain and multiplicity "
                        "comparison on that space will use. The knots above agree bit "
                        "for bit and `knot_tolerance` is the same reduction over them "
                        "on both sides. A difference would mean the two backends "
                        "disagree about which knots are the same knot, which no "
                        "comparison of the knots themselves would reveal."
                    )
                ),
                read=lambda field, d=direction: field.space.spaces[d].tolerance,  # type: ignore[misc]
            )
        )
        per_direction.append(
            Field(
                f"space.spaces[{direction}].periodic",
                exact_parity(
                    why=(
                        "a flag, and the two operations treat it oppositely: the "
                        "hodograph keeps a periodic direction periodic, while the "
                        "elevation refuses one outright and never sees it. So a "
                        "direction that came back with the wrong wrap would mean a "
                        "conversion happened that neither operation performs, and "
                        "nothing in the coefficients or the knots would say so."
                    )
                ),
                read=lambda field, d=direction: field.space.spaces[d].periodic,  # type: ignore[misc]
            )
        )
    return (
        Field("control_points", bitwise_parity(why=coefficients_why)),
        *per_direction,
        Field(
            "space.num_basis",
            exact_parity(
                why=(
                    "the per-direction basis counts the result's net is laid out on. "
                    "Exact integer counts, and the field's business rather than the "
                    "space's: this is the tuple the C++ constructor checks the net's "
                    "extents against, so a disagreement means the two backends laid one "
                    "result out on two different grids. A5.9 sizes its output by "
                    "counting as it writes rather than from a closed form, so this is "
                    "where a segment walk that took a wrong turn surfaces."
                )
            ),
        ),
        Field(
            "degree",
            exact_parity(
                why=(
                    "one integer per direction, in axis order, and it is what each "
                    "operation is *for* -- one less in the differentiated direction, "
                    "one more per increment in an elevated one, unchanged everywhere "
                    "else. The order is the load-bearing part: nothing about a degree "
                    "reveals a transposition on a field whose directions happen to "
                    "agree, which is why every multi-direction case here differs in it."
                )
            ),
        ),
        Field(
            "rank",
            exact_parity(
                why=(
                    "the stored component count less the weight column. Neither "
                    "operation touches a component axis, so a rank that moved means the "
                    "sweep consumed or produced an axis -- and it is the one field that "
                    "folds the rationality flag against the shape, so it catches a "
                    "result that preserved the coefficient count while moving which "
                    "axis carries the components."
                )
            ),
        ),
        Field(
            "is_rational",
            exact_parity(
                why=(
                    "the flag is carried through the elevation and is `False` by "
                    "construction on every hodograph the C++ half serves, since it "
                    "refuses a rational field. It is named as its own field rather than "
                    "left implicit because it is what `rank` subtracts, so a flag "
                    "dropped shows up here as well as one field away, and reading only "
                    "`rank` could not say which moved."
                )
            ),
        ),
        Field(
            "dtype",
            exact_parity(
                why=(
                    "the storage format the result reports, carried by the class of the "
                    "C++ handle. A disagreement means the operation changed precision, "
                    "which for a coefficient representable in both formats -- most of "
                    "them -- the bitwise claim above would not see."
                )
            ),
        ),
    )


def _both_derivatives(case: _Case, dtype: npt.DTypeLike) -> tuple[Bspline, Bspline]:
    """The hodograph under each backend.

    Args:
        case (_Case): The shape to build.
        dtype (npt.DTypeLike): The storage format.

    Returns:
        tuple[~pantr.bspline.Bspline, ~pantr.bspline.Bspline]: The Python and C++
        results, in that order.
    """
    with use_backend(Backend.PYTHON):
        py = _make_field(case, dtype).derivative(case.direction)
    with use_backend(Backend.CPP):
        cpp = _make_field(case, dtype).derivative(case.direction)
    return py, cpp


def _both_elevations(case: _Case, dtype: npt.DTypeLike) -> tuple[Bspline, Bspline]:
    """The elevated field under each backend.

    Args:
        case (_Case): The shape to build.
        dtype (npt.DTypeLike): The storage format.

    Returns:
        tuple[~pantr.bspline.Bspline, ~pantr.bspline.Bspline]: The Python and C++
        results, in that order.
    """
    with use_backend(Backend.PYTHON):
        py = _make_field(case, dtype).elevate_degree(case.increments)
    with use_backend(Backend.CPP):
        cpp = _make_field(case, dtype).elevate_degree(case.increments)
    return py, cpp


_DERIVATIVE_CASES: Final = tuple(case for case in CASES if not case.is_rational)
"""Every case the C++ hodograph serves; a rational field routes to the oracle."""

_ELEVATION_CASES: Final = tuple(case for case in CASES if any(case.increments))
"""Every case with something to elevate."""


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", _DERIVATIVE_CASES, ids=lambda case: case.label)
def test_the_hodograph_agrees_field_by_field(case: _Case, dtype: npt.DTypeLike) -> None:
    """Every piece of the hodograph's state agrees, under its own claim.

    No contraction gate: the difference quotient carries no sum of a product, so the
    bitwise claim holds on a fusing build too. See the module docstring.

    Args:
        case (_Case): The shape to build.
        dtype (npt.DTypeLike): The storage format.
    """
    py, cpp = _both_derivatives(case, dtype)
    assert_object_parity(
        py=py,
        cpp=cpp,
        fields=_fields(len(case.degrees), elevated=False),
        context=f"derivative(direction={case.direction}) on {case.label} at {dtype}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", _ELEVATION_CASES, ids=lambda case: case.label)
def test_the_elevation_agrees_field_by_field(case: _Case, dtype: npt.DTypeLike) -> None:
    """Every piece of the elevated field's state agrees, under its own claim.

    Args:
        case (_Case): The shape to build.
        dtype (npt.DTypeLike): The storage format.
    """
    if contraction_may_fuse():
        pytest.skip(_FUSING)
    # A5.9's `(1.0 - alf)` is `float64` compiled and `float32` interpreted, so the
    # oracle is a different computation there; see the module docstring.
    demand_the_compiled_kernel(dtype)
    if not _elevatable(case):
        pytest.skip("this case elevates a direction the C++ half refuses; see its own test")
    py, cpp = _both_elevations(case, dtype)
    assert_object_parity(
        py=py,
        cpp=cpp,
        fields=_fields(len(case.degrees), elevated=True),
        context=f"elevate_degree({case.increments}) on {case.label} at {dtype}",
    )


# ---------------------------------------------------------------------------
# The independent accuracy check
# ---------------------------------------------------------------------------


def _monomial_curve(
    knots: npt.NDArray[np.float32 | np.float64], degree: int, power: int
) -> tuple[Bspline, npt.NDArray[np.float64]]:
    """A curve whose control points are Marsden's coefficients of ``u ** power``.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): The knot vector.
        degree (int): The polynomial degree.
        power (int): The monomial power.

    Returns:
        tuple[~pantr.bspline.Bspline, npt.NDArray[np.float64]]: The field, and the
        closed form in ``float64`` before the cast into the field's storage.
    """
    closed_form = _marsden_column(knots, degree, power)
    space = BsplineSpace([BsplineSpace1D(knots, degree)])
    control_points = closed_form.astype(knots.dtype).reshape(-1, 1)
    return Bspline(space, control_points, is_rational=False), closed_form


def _hodograph_accuracy(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    closed_form: npt.NDArray[np.float64],
    live: npt.NDArray[np.bool_],
) -> AccuracyClaim:
    """The elementwise bound between a hodograph and ``r u^(r-1)``'s closed form.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): The original knot vector.
        degree (int): The original degree.
        closed_form (npt.NDArray[np.float64]): The input coefficients, in ``float64``.
        live (npt.NDArray[np.bool_]): Which rows have a non-vanishing denominator. The
            bound is returned over those rows alone, since a row whose span collapsed is
            not compared at all and a placeholder in the bound array would have to be
            invented for it.

    Returns:
        AccuracyClaim: The bound over the live rows, and its derivation.
    """
    denominator = np.asarray(knots[degree + 1 : len(closed_form) + degree], dtype=np.float64) - (
        np.asarray(knots[1 : len(closed_form)], dtype=np.float64)
    )
    magnitude = np.abs(closed_form[1:]) + np.abs(closed_form[:-1])
    # The vanishing rows are excluded from the comparison; a positive placeholder keeps
    # the bound array finite and the harness's own guards meaningful.
    amplification = degree * magnitude[live] / np.abs(denominator[live])

    # `2p + 1` for the input window's `e_r` recurrence, `2p - 1` for the derived window's
    # at degree `p - 1`, one for the cast into the field's storage, four for the formula,
    # one for the multiply by `r` on the expected side and one for the store -- charged
    # although the quotient is already in the storage format, which is the conservative
    # direction and keeps this count the same shape as the elevation's.
    roundings = 4 * degree + 7
    unit = unit_roundoff(knots.dtype)
    relative = roundings * unit / (1.0 - roundings * unit)
    return derived_accuracy(
        bound=relative * amplification + roundings * underflow_floor(knots.dtype),
        why=(
            f"Marsden's identity gives the coefficients of u**r in closed form, and the "
            f"derivative formula applied to them is exactly the closed form of "
            f"r*u**(r-1) on the derived knot vector -- an identity, proved in two lines "
            f"in cpp/tests/test_bspline_degree.cpp's file comment from "
            f"e_r(S + x) = e_r(S) + x e_(r-1)(S). gamma_{roundings} at unit roundoff "
            f"{unit:.3e}, times p*(|A_i| + |A_(i+1)|)/|t_(i+p+1) - t_(i+1)| elementwise: "
            f"{2 * degree + 1} roundings form the input window's closed form and "
            f"{2 * degree - 1} the derived window's, one is the cast into the field's "
            f"storage, four are the formula itself -- the coefficient difference, the "
            f"knot difference, the scaling by the degree and the division -- one is the "
            f"multiply by r on the expected side, and one is the store. The magnitude is "
            f"what was subtracted rather than the result, because that difference can "
            f"cancel to nothing while its operands do not. "
            f"{roundings} underflow floors are added for the rows whose closed form is "
            f"an exact zero, where a relative bound asserts bit-identity it has no "
            f"grounds for."
        ),
    )


def _elevation_accuracy(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    increment: int,
    stored: npt.NDArray[np.float64],
) -> AccuracyClaim:
    """The elementwise bound between an elevated curve and Marsden's closed form.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): The original knot vector.
        degree (int): The original degree.
        increment (int): Degrees added.
        stored (npt.NDArray[np.float64]): The input coefficients as the field stores
            them, already cast, which is what the kernel actually reads.

    Returns:
        AccuracyClaim: The bound and its derivation.
    """
    amplification = _elevate_majorant(
        degree, np.abs(stored), np.asarray(knots, dtype=np.float64), increment
    )
    # A5.9's chain has three blocks and they do not cost the same. At most `p - 1` Boehm
    # insertion passes, each `alf * bpts[q] + (1 - alf) * bpts[q-1]`: a subtraction, two
    # products, an addition and a narrowing store, so five. Then `min(p, t) + 1` terms
    # accumulated into one elevated Bezier coefficient -- Rule 10's own count -- each a
    # product, an addition and a narrowing store, so three. Then at most `p - 2`
    # knot-removal passes, the same shape as a Boehm pass, so five.
    boehm = 5 * max(0, degree - 1)
    accumulation = 3 * (min(degree, increment) + 1)
    removal = 5 * max(0, degree - 2)
    chain = max(1, boehm + accumulation + removal)
    roundings = (2 * degree + 1) + (2 * (degree + increment) + 1) + 2 + chain
    unit = unit_roundoff(knots.dtype)
    relative = roundings * unit / (1.0 - roundings * unit)
    return derived_accuracy(
        bound=relative * amplification + roundings * underflow_floor(knots.dtype),
        why=(
            f"Marsden's identity gives the coefficients of u**r in closed form, so an "
            f"elevation that does not move the map must reproduce it on the elevated "
            f"knots at the elevated degree. gamma_{roundings} at unit roundoff "
            f"{unit:.3e}, times the majorant of A5.9 run on the moduli of its "
            f"coefficients and of its blending weights: {2 * degree + 1} roundings form "
            f"the input closed form and {2 * (degree + increment) + 1} the elevated "
            f"one, one is the cast into the field's storage, {chain} are the kernel's "
            f"own chain -- {boehm} for at most p - 1 Boehm insertion passes at five "
            f"roundings each, {accumulation} for min(p, t) + 1 accumulated terms at "
            f"three, {removal} for at most p - 2 knot-removal passes at five -- and one "
            f"is the store. The amplification is the majorant "
            f"rather than a convex companion because A5.9's knot-removal step blends "
            f"with a weight above one, so its intermediates are not convex combinations "
            f"and a bound built from the finished operator would be exceeded rather "
            f"than merely loose -- design/backend_parity.md Rule 10. "
            f"{roundings} underflow floors are added for the coefficients whose closed "
            f"form is an exact zero."
        ),
    )


_ACCURACY_KNOTS: Final = (
    ((0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.7, 1.0, 1.0, 1.0, 1.0), 3),
    ((0.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0, 1.0, 1.0), 2),
    ((0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0), 3),
    ((0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 7.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0), 4),
    (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 0.6, 0.82, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        5,
    ),
)
"""Knot vectors for the independent check, each with a knot no binary format represents.

A dyadic vector rounds nothing and would report agreement without asking the bound a
question, which is why every entry carries a third or a seventh.

The degree-5 entry is here for the elevation budget rather than for the oracle. That
budget's Boehm-insertion and knot-removal terms grow with the degree while its
accumulation term is capped by the increment, so the three-block count above is only
tested where the first two dominate -- which needs a degree past 4 and several simple
interior knots. A review found the budget under-charging those two blocks and could not
say whether it mattered, because no accuracy case reached them.
"""


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.CPP], ids=["python", "cpp"])
@pytest.mark.parametrize("knots_and_degree", _ACCURACY_KNOTS, ids=lambda pair: f"degree{pair[1]}")
def test_the_hodograph_is_the_analytic_derivative(
    knots_and_degree: tuple[tuple[float, ...], int],
    backend: Backend,
    dtype: npt.DTypeLike,
) -> None:
    """Each backend's hodograph is ``r u^(r-1)``, against a closed form neither computes.

    Args:
        knots_and_degree (tuple[tuple[float, ...], int]): The knot vector and degree.
        backend (Backend): Which backend to exercise.
        dtype (npt.DTypeLike): The storage format.
    """
    raw, degree = knots_and_degree
    knots = np.array(raw, dtype=dtype)
    derived = knots[1:-1]
    checked = 0
    for power in range(degree + 1):
        with use_backend(backend):
            field, closed_form = _monomial_curve(knots, degree, power)
            hodograph = field.derivative()
        denominator = np.asarray(
            knots[degree + 1 : len(closed_form) + degree], dtype=np.float64
        ) - np.asarray(knots[1 : len(closed_form)], dtype=np.float64)
        live = denominator != 0.0

        expected = np.zeros(len(closed_form) - 1, dtype=np.float64)
        if power >= 1:
            expected = power * _marsden_column(derived, degree - 1, power - 1)

        got = hodograph.control_points.reshape(-1).astype(np.float64)
        assert np.all(got[~live] == 0.0), "a zero-width span did not give an exact zero"
        checked += int(np.count_nonzero(live))
        assert_accuracy(
            computed=got[live],
            exact=expected[live],
            claim=_hodograph_accuracy(knots, degree, closed_form, live),
            context=f"the hodograph of u**{power} at degree {degree}, {dtype}, {backend.name}",
        )
    assert checked > 0, "every row was excluded, so the check compared nothing"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.CPP], ids=["python", "cpp"])
@pytest.mark.parametrize("increment", [1, 2, 3])
@pytest.mark.parametrize("knots_and_degree", _ACCURACY_KNOTS, ids=lambda pair: f"degree{pair[1]}")
def test_the_elevation_reproduces_the_polynomial(
    knots_and_degree: tuple[tuple[float, ...], int],
    increment: int,
    backend: Backend,
    dtype: npt.DTypeLike,
) -> None:
    """Each backend's elevation is the same map, against a closed form neither computes.

    Args:
        knots_and_degree (tuple[tuple[float, ...], int]): The knot vector and degree.
        increment (int): Degrees to add.
        backend (Backend): Which backend to exercise.
        dtype (npt.DTypeLike): The storage format.
    """
    raw, degree = knots_and_degree
    knots = np.array(raw, dtype=dtype)
    for power in range(degree + 1):
        with use_backend(backend):
            field, closed_form = _monomial_curve(knots, degree, power)
            raised = field.elevate_degree(increment)
        stored = field.control_points.reshape(-1).astype(np.float64)
        space = raised.space.spaces[0]
        assert space.degree == degree + increment
        expected = _marsden_column(space.knots, space.degree, power)
        got = raised.control_points.reshape(-1).astype(np.float64)
        assert got.size == expected.size, (
            f"the elevation returned {got.size} coefficients and the elevated vector "
            f"calls for {expected.size}"
        )
        assert_accuracy(
            computed=got,
            exact=expected,
            claim=_elevation_accuracy(knots, degree, increment, stored),
            context=(
                f"elevating u**{power} at degree {degree} by {increment}, {dtype}, {backend.name}"
            ),
        )


def _worst_margin(dtype: npt.DTypeLike) -> float:
    """The largest ratio to its own bound either accuracy check reaches at one format.

    Args:
        dtype (npt.DTypeLike): The storage format.

    Returns:
        float: The worst ``|deviation| / bound`` over the sweep, which must not exceed 1
        and must not be so far below it that the bound asserts nothing.
    """
    worst = 0.0
    for raw, degree in _ACCURACY_KNOTS:
        knots = np.array(raw, dtype=dtype)
        derived = knots[1:-1]
        for power in range(degree + 1):
            with use_backend(Backend.CPP):
                field, closed_form = _monomial_curve(knots, degree, power)
                hodograph = field.derivative()
            denominator = np.asarray(
                knots[degree + 1 : len(closed_form) + degree], dtype=np.float64
            ) - np.asarray(knots[1 : len(closed_form)], dtype=np.float64)
            live = denominator != 0.0
            expected = (
                np.zeros(len(closed_form) - 1, dtype=np.float64)
                if power == 0
                else power * _marsden_column(derived, degree - 1, power - 1)
            )
            got = hodograph.control_points.reshape(-1).astype(np.float64)
            worst = max(
                worst,
                assert_accuracy(
                    computed=got[live],
                    exact=expected[live],
                    claim=_hodograph_accuracy(knots, degree, closed_form, live),
                    context="the margin sweep",
                ).max_ratio_to_bound,
            )
            for increment in (1, 2, 3):
                with use_backend(Backend.CPP):
                    source, _ = _monomial_curve(knots, degree, power)
                    raised = source.elevate_degree(increment)
                stored = source.control_points.reshape(-1).astype(np.float64)
                space = raised.space.spaces[0]
                worst = max(
                    worst,
                    assert_accuracy(
                        computed=raised.control_points.reshape(-1).astype(np.float64),
                        exact=_marsden_column(space.knots, space.degree, power),
                        claim=_elevation_accuracy(knots, degree, increment, stored),
                        context="the margin sweep",
                    ).max_ratio_to_bound,
                )
    return worst


def test_the_accuracy_bounds_are_approached_rather_than_merely_satisfied() -> None:
    """Both bounds come within an order of magnitude of being violated.

    A bound nothing approaches asserts nothing, and the two above are built from counted
    roundings times an amplification, either of which could be over-stated without any
    test noticing. This runs the same sweep the two accuracy tests do and asserts the
    worst ratio to the bound clears a floor.

    **The floor is a hundredth, and it is an acknowledged slack rather than a
    derivation.** What it has to absorb is the one deliberate over-statement in both
    budgets: every rounding is charged at the *storage* format's unit roundoff, the
    oracle's own ``e_r`` recurrences included, although those run in ``float64`` whatever
    the field stores. At ``float32`` that alone over-states roughly ``4p + 2`` of the
    counted roundings by ``2**29``. A tighter floor would be pinning the size of that
    slack, which is not a quantity anything here derives.
    """
    worst = max(_worst_margin(dtype) for dtype in DTYPES)
    assert worst > 0.01, (
        f"the accuracy bounds were never approached, worst ratio {worst:.3g}; either "
        f"nothing in the sweep rounded or one of the two bounds is vacuous"
    )


# ---------------------------------------------------------------------------
# The four places the backends do not meet
# ---------------------------------------------------------------------------


def _binding() -> Any:
    """The compiled extension, imported late so a missing one skips rather than errors.

    Returns:
        Any: The :mod:`pantr._pantr_cpp` module.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    return _pantr_cpp


def _case(label: str) -> _Case:
    """Look one case up by its label.

    Args:
        label (str): The case's label.

    Returns:
        _Case: The case.
    """
    return next(case for case in CASES if case.label == label)


def test_a_periodic_direction_is_refused_by_the_binding_and_served_by_the_oracle() -> None:
    """The C++ half refuses to elevate a periodic direction; the public API still does it.

    The two assertions together are what say the routing happened. Neither alone would:
    a result is a result whichever path produced it, and a refusal proves only that the
    binding has the boundary, not that the wrapper routes around it.
    """
    with use_backend(Backend.CPP):
        field = _make_field(_case("periodic curve"), np.float64)
        with pytest.raises(ValueError, match="open-to-periodic conversion"):
            _binding().elevate_bspline_degree(field._impl, [1])
        raised = field.elevate_degree(1)

    assert raised.space.spaces[0].periodic, "the periodic direction lost its wrap"
    assert raised.space.spaces[0].degree == field.space.spaces[0].degree + 1


def test_an_unclamped_direction_is_refused_by_the_binding() -> None:
    """The C++ half refuses to elevate an unclamped direction, rather than reading past.

    A5.9 walks segments until a run of equal knots reaches the last index of the knot
    vector, and an unclamped vector has none. The oracle performs the read; in C++ that
    would be undefined behaviour, so the boundary is declared and the routing sends the
    call to the oracle -- which is the subject of
    :func:`test_an_unclamped_elevation_fails_the_same_way_under_both_backends`.
    """
    with use_backend(Backend.CPP):
        field = _make_field(_case("unclamped curve"), np.float64)
        with pytest.raises(ValueError, match="not clamped"):
            _binding().elevate_bspline_degree(field._impl, [1])


@pytest.mark.parametrize("dtype", DTYPES)
def test_an_unclamped_elevation_fails_the_same_way_under_both_backends(
    dtype: npt.DTypeLike,
) -> None:
    """An unclamped elevation raises the oracle's own error under either backend.

    **This pins a defect on purpose.** A5.9 needs a clamped vector and the oracle does
    not check for one; what a caller currently sees is the count mismatch its
    out-of-bounds walk produces, raised by ``Bspline``'s own constructor. The C++ half
    refuses such a direction outright and :mod:`pantr.bspline._degree_backend` routes it
    to the oracle, so what the library accepts does not change with ``PANTR_BACKEND`` --
    and the day the oracle is fixed, this test fails and says so rather than the change
    landing silently.

    Args:
        dtype (npt.DTypeLike): The storage format.
    """
    # The defect's *symptom* depends on the configuration, which is worth pinning rather
    # than gating away: numba does not bounds check, so compiled the walk returns and the
    # field's own constructor refuses the coefficient count; interpreted, numpy checks and
    # the read itself raises. Both are the same out-of-bounds walk.
    if the_jit_is_disabled():
        expected_type: type[Exception] = IndexError
        fragment = "is out of bounds for axis 0"
    else:
        expected_type = ValueError
        fragment = "must be a multiple of the number of basis functions"

    messages = []
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            field = _make_field(_case("unclamped curve"), dtype)
            with pytest.raises(expected_type) as raised:
                field.elevate_degree(1)
            messages.append(str(raised.value))
    assert messages[0] == messages[1], (
        "the two backends refuse an unclamped elevation differently, so the routing is "
        "not sending it to the oracle"
    )
    assert fragment in messages[0], (
        f"the oracle's unclamped elevation no longer fails with {fragment!r}; if it now "
        f"refuses the vector properly, cpp/include/pantr/bspline/degree.hpp's boundary "
        f"and pantr.bspline._degree_backend's routing should both be revisited"
    )


def test_a_rational_derivative_is_refused_by_the_binding_and_served_by_the_oracle() -> None:
    """The C++ half refuses a rational hodograph; the public API still computes one.

    The oracle's quotient-rule path is under review, so there is nothing stable for a
    C++ side to be at parity with and it is not ported.
    """
    with use_backend(Backend.CPP):
        field = _make_field(_case("rational surface, direction 0 only"), np.float64)
        with pytest.raises(ValueError, match="rational B-spline is not part of"):
            _binding().differentiate_bspline(field._impl, 0)
        hodograph = field.derivative(0)

    assert hodograph.is_rational, "the rational hodograph came back non-rational"


@pytest.mark.parametrize("dtype", DTYPES)
def test_keep_degree_runs_the_oracle_under_both_backends(dtype: npt.DTypeLike) -> None:
    """``keep_degree=True`` is absent from the C++ signature, so both backends agree.

    **Common mode**, said out loud rather than counted as evidence: one implementation
    ran twice. What this pins is that the routing produces a complete field under the
    C++ backend and that no half-ported path leaks into it.

    Args:
        dtype (npt.DTypeLike): The storage format.
    """
    case = _case("cubic curve, four simple knots")
    with use_backend(Backend.PYTHON):
        py = _make_field(case, dtype).derivative(keep_degree=True)
    with use_backend(Backend.CPP):
        cpp = _make_field(case, dtype).derivative(keep_degree=True)
    assert cpp.degree == py.degree == case.degrees
    assert np.array_equal(cpp.control_points.view(np.uint8), py.control_points.view(np.uint8)), (
        "the two backends disagree on a path neither of them ports"
    )


def test_the_periodic_hodograph_reaches_the_cpp_path() -> None:
    """A periodic direction does not push the *derivative* onto the oracle.

    The other half of the routing rule, and the one an over-broad condition would get
    wrong. ``_derivative_nonrational_nd`` handles a periodic direction by expanding the
    net with its own modulo wrap and trimming afterwards, reaching no boundary or
    periodic conversion, so the C++ half serves it. Pinned by the binding accepting the
    same call the public method makes.
    """
    with use_backend(Backend.CPP):
        field = _make_field(_case("periodic curve"), np.float64)
        through_the_binding = _binding().differentiate_bspline(field._impl, 0)
        hodograph = field.derivative()

    assert np.array_equal(hodograph.control_points, through_the_binding.control_points)
    assert hodograph.space.spaces[0].periodic, "the periodic direction lost its wrap"


def test_the_unclamped_hodograph_reaches_the_cpp_path() -> None:
    """An unclamped direction does not push the *derivative* onto the oracle either.

    The elevation's clamp requirement is A5.9's, not the difference quotient's: the
    hodograph formula is elementwise and assumes nothing about the ends. A condition
    written once for both operations would send this to the oracle for no reason.
    """
    with use_backend(Backend.CPP):
        field = _make_field(_case("unclamped curve"), np.float64)
        through_the_binding = _binding().differentiate_bspline(field._impl, 0)
        hodograph = field.derivative()

    assert np.array_equal(hodograph.control_points, through_the_binding.control_points)
    assert not hodograph.space.spaces[0].has_open_knots(), (
        "the unclamped direction came back clamped"
    )


def test_an_argument_with_nothing_to_elevate_goes_to_the_oracle() -> None:
    """A degenerate increment tuple is served by the oracle, not refused by the binding.

    The public method refuses an all-zero or negative increment before either backend is
    reached, so this is about the private entry point: `elevate_field_degree` is an
    importable symbol of a package whose private symbols a downstream consumer already
    imports, and the two sides disagree about such an argument -- the C++ half raises
    `Bspline.elevate_degree`'s own Layer 1 message while `_degree_elevate_bspline` never
    checks and returns the field unchanged. Found by a review of the routing predicate,
    which said yes vacuously because `all()` over an empty filter is true.
    """
    case = _case("surface, direction 1 differentiated and elevated")
    for increments in ((0, 0), (-1, -1)):
        results = []
        for backend in (Backend.PYTHON, Backend.CPP):
            with use_backend(backend):
                results.append(elevate_field_degree(_make_field(case, np.float64), increments))
        assert results[0].degree == results[1].degree == case.degrees, (
            f"{increments} changed a degree on some backend"
        )
        assert np.array_equal(results[0].control_points, results[1].control_points), (
            f"the two backends disagree about the increment tuple {increments}"
        )


def test_an_untouched_direction_keeps_its_wrapper() -> None:
    """A direction neither operation changed is carried over rather than rebuilt.

    ``result.space.spaces[d] is field.space.spaces[d]`` is the contract ``_wrap_over``
    exists for, and it is what says the C++ half reported which directions it left alone
    rather than the wrapper rebuilding them all.
    """
    with use_backend(Backend.CPP):
        case = _case("surface, direction 1 differentiated and elevated")
        field = _make_field(case, np.float64)
        hodograph = field.derivative(case.direction)
        raised = field.elevate_degree(case.increments)

    assert hodograph.space.spaces[0] is field.space.spaces[0]
    assert raised.space.spaces[0] is field.space.spaces[0]
    assert hodograph.space.spaces[1] is not field.space.spaces[1]
    assert raised.space.spaces[1] is not field.space.spaces[1]


# ---------------------------------------------------------------------------
# The case table
# ---------------------------------------------------------------------------


def test_the_case_table_earns_its_keep() -> None:
    """Each case is asymmetric where a symmetric one would hide a defect.

    Six properties the module docstring and ``CASES`` rely on, asserted rather than
    believed: every multi-direction case differs between its directions in the degree and
    the basis count; some case is rational, some periodic, some unclamped; some carries a
    knot no binary format represents; and both operations really do change the field.
    """
    multi = [case for case in CASES if len(case.degrees) > 1]
    assert multi, "no multi-direction case, so nothing here could see a transposition"
    for case in multi:
        assert len(set(case.degrees)) == len(case.degrees), f"{case.label}: repeated degree"
        counts = [
            len(knots) - degree - 1 for knots, degree in zip(case.knots, case.degrees, strict=True)
        ]
        assert len(set(counts)) == len(counts), f"{case.label}: repeated basis count"

    assert any(case.is_rational for case in CASES), "no rational case"
    assert any(any(case.periodic) for case in CASES), "no periodic case"
    assert any(not _elevatable(case) for case in CASES), "no case the C++ elevation refuses"
    assert any(
        any(
            knot not in (0.0, 0.25, 0.5, 0.75, 1.0, -1.0, 2.0)
            for knots in case.knots
            for knot in knots
        )
        for case in CASES
    ), "every knot in the table is dyadic, so no operation here has to round"

    # And the coefficients round too. A field of exactly representable values makes the
    # hodograph's subtraction and quotient exact, and a bitwise claim over it cannot tell
    # the storage format's arithmetic from `double` narrowed once -- which is the very
    # thing the claim is for. See :func:`_make_field`.
    for case in CASES:
        narrow = _make_field(case, np.float32).control_points
        wide = _make_field(case, np.float64).control_points
        # Every coefficient, not merely some: one exactly representable value in the net
        # is one fibre over which the bitwise claim cannot see the arithmetic width.
        assert not np.any(narrow.astype(np.float64) == wide), (
            f"{case.label}: a control point is exactly representable in float32, so the "
            f"bitwise claims over its fibre are blind to the arithmetic width"
        )

    with use_backend(Backend.PYTHON):
        for case in _DERIVATIVE_CASES:
            field = _make_field(case, np.float64)
            hodograph = field.derivative(case.direction)
            assert hodograph.degree[case.direction] == case.degrees[case.direction] - 1, (
                f"{case.label}: the hodograph did not lower a degree"
            )
        for case in _ELEVATION_CASES:
            if not _elevatable(case):
                continue
            field = _make_field(case, np.float64)
            raised = field.elevate_degree(case.increments)
            assert raised.control_points.size > field.control_points.size, (
                f"{case.label}: the elevation produced no new coefficients"
            )
