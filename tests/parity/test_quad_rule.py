r"""Parity of the C++ `QuadratureRule` against the pure-Python oracle it was ported from.

`cpp/include/pantr/quad/rule.hpp` is the third *type* in the C++ core rather than
a kernel, after `AABB` and `AffineTransform`, and it is the first whose state is
two arrays of different shape plus two counts. That is what it costs to use
`assert_object_parity`, and the answer is: four `Field`s and no special case.

What the claims are, and why each is the strongest available
------------------------------------------------------------

**Constructing a rule from given points and weights is BITWISE, unconditionally.**
Neither backend computes anything: both copy the caller's arrays, check them and
store them. There is no arithmetic for a tolerance to allow for, so a tolerance
here would hide a defect rather than absorb one.

**The tensor product is BITWISE, unconditionally, and that is the claim with a
premise worth stating.** Each coordinate of a tensor-product point is a *copy* of
a 1-D node, so that half is exact by construction. Each weight is a chain of
``ndim - 1`` multiplications, and the two backends perform them in the same order:
the oracle's ``np.prod(..., axis=0)`` over an ``(ndim, num_points)`` block reduces
axis 0 in sequence, and the C++ loop accumulates from axis 0 upwards. **A chain of
multiplications contains no addition, so a fused multiply-add has nothing to
fuse** -- which is why this claim, unlike the Gauss-Legendre one below, does not
consult ``__fp_contract__`` at all.

Its premise is numpy's reduction order, which is not ours. `np.sum` blocks
pairwise from 8 elements up; `np.multiply.reduce` does not, and nothing obliges it
to stay that way. Measured before these assertions were written: 480 random rules
of 1 to 8 axes, and zero of the 480 differ in either array. If numpy ever blocks
its multiplicative reduction the way it blocks its additive one, this is the file
that will say so, and the fix is a bound rather than a looser equality.

**The Gauss-Legendre rule is BITWISE on a build that cannot fuse, and BOUNDED on
one that can.** Everything this ticket adds on top of the 1-D kernel is exact or
common mode; the whole difference comes from `gauss_legendre_symmetric`, whose
node and weight bounds are derived in `tests/parity/test_quad_gauss_legendre.py`
and in `design/backend_parity.md` Rule 4. The constants are **imported from that
module rather than restated**: they denote the same quantities, and two spellings
of one bound drift.

Composing the 1-D bound onto the rule, which is this file's own derivation
--------------------------------------------------------------------------

Write ``u`` for the unit roundoff and ``c_x = 4.5`` for the one-sided
Gauss-Legendre node displacement in units of ``u``, absolute and flat in ``n``
(Rule 4). The harness doubles every amplification, so everything below is
*one-sided*, per backend, against the exact rule.

**Points.** A coordinate is a copy of ``t = fl(fl(x + 1) * 0.5)`` where ``x`` is a
1-D node on ``[-1, 1]``. Multiplying by ``0.5`` is exact, so the map halves the
inherited displacement and adds one rounding of its own at the ``x + 1``:

.. math::

    |\delta t| \le \tfrac{1}{2} c_x u + u\,|x + 1| = \left(\tfrac{c_x}{2}
                 + 2|t|\right) u .

Both terms are needed and `design/backend_parity.md` Rule 1 says why the first one
has to be *transported* rather than rewritten: an absolute floor re-expressed with
the mapped magnitude is eleven orders too small at one end of the array.

**Weights.** A tensor weight is ``W = prod_d w_d``, each ``w_d`` a 1-D weight
already halved by the map -- exactly, so its *relative* error is unchanged. The
relative errors of a product add, and forming the product costs ``ndim - 1``
roundings:

.. math::

    \frac{|\delta W|}{|W|} \le \sum_d \frac{|\delta w_d|}{|w_d|} + (ndim - 1) u ,
    \qquad
    \frac{|\delta w_d|}{|w_d|} \le \left(c_x A_d + 5\right) u ,

with ``A_d = 2|x_d| / (1 - x_d^2)`` the weight's logarithmic sensitivity at a root
of ``P_n``, and 5 the roundings of ``2 / ((1 - x*x) d * d)``. Both come from Rule 4.

**``A_d`` is evaluated at the node on ``[-1, 1]``, never at the mapped one, and
that is a precondition rather than a preference.** ``A`` is even in ``x`` while its
naive rewrite in ``t`` is not, and the map sends the two singular ends to ``t = 0``
and ``t = 1`` where only the second is singular -- so half the array would lose its
singularity, by a factor measured at 1.9e12 by ``n = 2000``. `_weight_claim` below
therefore takes the unmapped nodes and refuses an array with no negative entry.

**The composition step is the only new content above, and it is checked rather
than asserted.** `test_the_composed_weight_bound_survives_a_perturbed_1d_rule`
perturbs a 1-D rule by exactly its own claimed budget, pushes both through the
tensor product, and requires the ND difference to stay inside the composed bound.
It runs on any build, so the BOUNDED branch is not shipped unexercised even though
this host cannot fuse. Verified offline over a sweep 20x the one that ships; the
margin is reported in that test's docstring.

What these tests would catch that a numeric comparison would not
----------------------------------------------------------------

That the two implementations agree on **which exception** they raise and,
verbatim, on **what it says**; that the wrapper's ``__repr__`` and pickle round
trip do not move with ``PANTR_BACKEND``; and -- the one a parity suite can get
wrong about itself -- that the two objects really are two implementations, which
`test_the_two_backends_hold_different_implementations` asserts directly.
"""

from __future__ import annotations

import functools
import pickle
from collections.abc import Callable, Sequence
from typing import Any, Final, TypeVar

import numpy as np
import pytest

from pantr._backend import Backend, use_backend
from pantr.quad import QuadratureRule, gauss_legendre_quadrature, tensor_product_quadrature
from pantr.quad._rule_nd import _QuadratureRulePython
from pantr.quad._rules import _gauss_legendre_symmetric
from tests._parity_harness import (
    ExactClaim,
    Field,
    FloatArray,
    ParityClaim,
    Roundings,
    absolute_tolerance,
    assert_object_parity,
    bitwise_parity,
    bounded_parity,
    contraction_may_fuse,
    exact_parity,
)
from tests.parity.test_quad_gauss_legendre import (
    NODE_DISPLACEMENT_UNITS,
    WEIGHT_FORMULA_ROUNDINGS,
)

pytestmark = pytest.mark.usefixtures("cpp_backend")

_Result = TypeVar("_Result")

_AxisRule = tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]
"""One axis's ``(nodes, weights)`` pair, both 1-D and ``float64``."""

_PRODUCT_ROUNDINGS: Final = Roundings(stages=1, accumulator_per_stage=1, storage_per_stage=1)
"""The budget every bounded claim here carries.

One stage of one rounding, which reduces :class:`~tests._parity_harness.Roundings`
to a way of spelling ``u``: the derivation lives in the amplification array, as
``design/backend_parity.md`` records is the case for any bound that is a derived
ratio rather than a dependency chain. ``storage_per_stage`` is 1 rather than 0
because the rule narrows on the way out in principle; it costs nothing here, since
storage and accumulator are both ``float64`` and the harness charges a store into
the accumulator's own format at zero.
"""

_COUNTS_ARE_STRUCTURE: Final = exact_parity(
    why=(
        "ndim and num_points are shapes, not quantities: they are the extents of "
        "the two arrays compared beside them. No rounding can move one, and a "
        "difference is a differently shaped rule rather than a displaced value, "
        "which no tolerance could absorb. design/backend_parity.md Rule 11."
    )
)

_NOTHING_IS_COMPUTED: Final = bitwise_parity(
    why=(
        "constructing a rule from given points and weights performs no arithmetic "
        "at all: both backends copy the caller's arrays, validate them and store "
        "them. There is no operation for a fused multiply-add to change and none "
        "for a tolerance to allow for."
    )
)

_THE_SAME_MULTIPLICATIONS: Final = bitwise_parity(
    why=(
        "a tensor-product coordinate is a copy of a 1-D node, and a tensor-product "
        "weight is a chain of ndim - 1 multiplications performed in the same order "
        "on both sides: the oracle's np.prod(..., axis=0) reduces axis 0 of an "
        "(ndim, num_points) block in sequence, and the C++ loop accumulates from "
        "axis 0 upwards. A chain of multiplications contains no addition, so "
        "contraction has nothing to fuse and this claim does not depend on the "
        "build. Its premise is numpy's reduction order, which is not ours: "
        "np.sum blocks pairwise from 8 elements up and np.multiply.reduce does "
        "not. Measured before this was asserted: 480 random rules of 1 to 8 axes, "
        "zero differing values. If numpy ever blocks the multiplicative reduction "
        "too, this assertion is where it will say so."
    )
)

_GAUSS_LEGENDRE_IS_EXACT_BY_BUILD: Final = bitwise_parity(
    why=(
        "the whole difference would come from gauss_legendre_symmetric, which is "
        "an operation-for-operation transliteration using only +, -, * and /, "
        "each pinned by IEEE 754 -- and this build emits no fused multiply-add. "
        "tests/parity/test_quad_gauss_legendre.py carries the measurement and "
        "names cos as the only unpinned operation. Everything this rule adds on "
        "top is exact (the copy of a node, the multiplication by 0.5) or common "
        "mode (the x + 1 and the weight product, identical operations in an "
        "identical order)."
    )
)

_BOUNDED_BY_FMA: Final = (
    "this build fuses a multiply-add, so the two backends commit different "
    "numbers of roundings inside the Legendre recurrence behind each 1-D rule. "
    "See this module's docstring for how the 1-D bound of "
    "design/backend_parity.md Rule 4 composes onto the tensor-product rule, and "
    "tests/parity/test_quad_gauss_legendre.py for the 1-D bound itself."
)


def _both(build: Callable[[], _Result]) -> tuple[_Result, _Result]:
    """Run a builder once under each backend.

    Args:
        build (Callable[[], _Result]): A zero-argument callable returning the rule.

    Returns:
        tuple[_Result, _Result]: The Python result then the C++ result.
    """
    with use_backend(Backend.PYTHON):
        python = build()
    with use_backend(Backend.CPP):
        cpp = build()
    return python, cpp


def _fields(claim: ParityClaim | ExactClaim, weight_claim: ParityClaim | ExactClaim) -> list[Field]:
    """Name the four pieces of state a rule is, and the claim governing each.

    A rule's whole observable content is two counts and two arrays. Nothing is
    derived and nothing is lazily built, so there is no field this list may
    reasonably leave out -- unlike a grid, where the harness's docstring records
    that an expensive accessor is the caller's to reach for.

    Args:
        claim (ParityClaim | ExactClaim): The claim for ``points``.
        weight_claim (ParityClaim | ExactClaim): The claim for ``weights``,
            separate because the two carry different sensitivities under a build
            that fuses.

    Returns:
        list[Field]: The four fields, in the order a failure should report them.
    """
    return [
        Field("ndim", _COUNTS_ARE_STRUCTURE),
        Field("num_points", _COUNTS_ARE_STRUCTURE),
        Field("points", claim),
        Field("weights", weight_claim),
    ]


def _sensitivity(reference_nodes: FloatArray) -> FloatArray:
    """The weight's logarithmic sensitivity ``2|x| / (1 - x^2)`` at a Gauss node.

    From the Legendre differential equation collapsing to ``P'' = 2 x P' / (1 - x^2)``
    at a root of ``P_n``; see ``design/backend_parity.md`` Rule 4.

    Args:
        reference_nodes (FloatArray): The nodes **on ``[-1, 1]``**, not the mapped
            ones. See this module's docstring for why the frame is a precondition.

    Returns:
        FloatArray: The sensitivity, zero where ``|x| == 1`` (which a Gauss rule
        never produces, and which keeps the expression finite for a caller that
        passes a Lobatto rule).

    Raises:
        ValueError: If the array has no negative entry while holding more than one
            node, which means it is the rule already mapped onto ``[0, 1]``.
    """
    x = np.asarray(reference_nodes, dtype=np.float64)
    if x.size > 1 and float(np.min(x)) >= 0.0:
        raise ValueError(
            "the weight sensitivity is derived at a root of P_n, so it needs the "
            "nodes on [-1, 1]; this array has no negative entry, so it is the rule "
            "mapped onto [0, 1]. Un-map it before building the claim"
        )
    magnitude = np.abs(x)
    interior = magnitude < 1.0
    out = np.zeros_like(magnitude)
    out[interior] = 2.0 * magnitude[interior] / (1.0 - magnitude[interior] * magnitude[interior])
    return out


def _point_claim(points: FloatArray) -> ParityClaim:
    """State the claim for the mapped Gauss-Legendre points.

    Args:
        points (FloatArray): The rule's points, on ``[0, 1]^ndim``, whose
            magnitude the map's own rounding is proportional to.

    Returns:
        ParityClaim: BITWISE where the build cannot fuse, otherwise BOUNDED.
    """
    if not contraction_may_fuse():
        return _GAUSS_LEGENDRE_IS_EXACT_BY_BUILD
    amplification = NODE_DISPLACEMENT_UNITS / 2.0 + 2.0 * np.abs(np.asarray(points))
    return bounded_parity(
        roundings=_PRODUCT_ROUNDINGS,
        accumulator=np.float64,
        storage=np.float64,
        amplification=amplification,
        why=(
            f"{_BOUNDED_BY_FMA} A coordinate is a copy of "
            f"t = fl(fl(x + 1) * 0.5). The multiplication by 0.5 is exact, so the "
            f"map halves the inherited {NODE_DISPLACEMENT_UNITS} u node "
            f"displacement and adds one rounding of its own at the x + 1, worth "
            f"u|x + 1| = 2u|t|. The first term is TRANSPORTED rather than "
            f"rewritten in the mapped magnitude, per Rule 1: rewriting it "
            f"evaluates an absolute floor at the wrong end of an array spanning "
            f"six decades and fails by 37.8x on correct code."
        ),
    )


def _weight_claim(
    reference_nodes_per_axis: Sequence[FloatArray], weights: FloatArray
) -> ParityClaim:
    """State the claim for the tensor-product Gauss-Legendre weights.

    Args:
        reference_nodes_per_axis (Sequence[FloatArray]): Each axis's nodes **on
            ``[-1, 1]``**, in axis order. The sensitivity is a function of that
            coordinate; see this module's docstring.
        weights (FloatArray): The tensor-product weights, which carry the
            magnitude. Frame-free, since the relative perturbation is.

    Returns:
        ParityClaim: BITWISE where the build cannot fuse, otherwise BOUNDED.
    """
    if not contraction_may_fuse():
        return _GAUSS_LEGENDRE_IS_EXACT_BY_BUILD
    return bounded_parity(
        roundings=_PRODUCT_ROUNDINGS,
        accumulator=np.float64,
        storage=np.float64,
        amplification=_composed_weight_amplification(reference_nodes_per_axis, weights),
        why=(
            f"{_BOUNDED_BY_FMA} A tensor weight is a product of ndim 1-D weights, "
            f"each already halved by the map -- exactly, so its relative error is "
            f"unchanged. Relative errors of a product add, and forming the product "
            f"costs ndim - 1 roundings. Each factor carries "
            f"({NODE_DISPLACEMENT_UNITS} A_d + {WEIGHT_FORMULA_ROUNDINGS}) u with "
            f"A_d = 2|x_d|/(1 - x_d^2) evaluated at the node on [-1, 1]: that "
            f"factor is Theta(n^2) at the outermost node, so a per-element "
            f"relative bound with a CONSTANT is refuted, and the absolute one "
            f"survives only because the endpoint weight decays like 1/n^2."
        ),
    )


def _compose_relative_budgets(
    per_axis_units: Sequence[FloatArray], weights: FloatArray
) -> FloatArray:
    """Compose per-axis relative weight budgets onto the tensor-product weights.

    **This function is the composition claim**, and it is what
    :func:`test_the_composed_weight_bound_survives_a_perturbed_1d_rule` checks: the
    relative errors of a product add, and forming the product costs ``ndim - 1``
    further roundings. Everything else in the weight bound is the 1-D module's.

    Args:
        per_axis_units (Sequence[FloatArray]): Each axis's relative budget, in
            units of ``u``, one entry per node on that axis.
        weights (FloatArray): The tensor-product weights, which carry the
            magnitude that turns a relative budget into an absolute one.

    Returns:
        FloatArray: One amplification per weight, in units of ``u``.
    """
    ndim = len(per_axis_units)
    # Same enumeration as the rule itself: last axis fastest.
    mesh = np.meshgrid(*per_axis_units, indexing="ij")
    total = np.sum(np.stack([m.ravel() for m in mesh], axis=0), axis=0) + (ndim - 1)
    return np.asarray(np.abs(np.asarray(weights, dtype=np.float64)) * total, dtype=np.float64)


def _composed_weight_amplification(
    reference_nodes_per_axis: Sequence[FloatArray], weights: FloatArray
) -> FloatArray:
    """Build the elementwise amplification of the tensor-product weights.

    Args:
        reference_nodes_per_axis (Sequence[FloatArray]): Each axis's nodes on
            ``[-1, 1]``, in axis order.
        weights (FloatArray): The tensor-product weights.

    Returns:
        FloatArray: One amplification per weight, in units of ``u`` times the
        weight's own magnitude.
    """
    return _compose_relative_budgets(
        [
            NODE_DISPLACEMENT_UNITS * _sensitivity(nodes) + WEIGHT_FORMULA_ROUNDINGS
            for nodes in reference_nodes_per_axis
        ],
        weights,
    )


def _reference_nodes(npts: Sequence[int]) -> list[FloatArray]:
    """The per-axis Gauss-Legendre nodes on ``[-1, 1]``, from the oracle.

    Taken unmapped rather than reconstructed from the rule's own ``[0, 1]``
    points, so the sensitivity is evaluated in the coordinate it was derived in
    with no inverse map to be approximately right about.

    Args:
        npts (Sequence[int]): Points per axis.

    Returns:
        list[FloatArray]: One node array per axis.
    """
    with use_backend(Backend.PYTHON):
        return [np.asarray(_gauss_legendre_symmetric(n)[0], dtype=np.float64) for n in npts]


# Corner cases chosen for the phenomena rather than for coverage: an ordinary
# rule, a single point, the closed-cube boundary that Lobatto rules land on, a
# negative weight (legal, and several real rules carry one), a one-dimensional
# rule, and a four-dimensional one where the weight product is a longer chain.
_RULES: Final = [
    ([[0.25, 0.75], [0.5, 0.5]], [0.4, 0.6]),
    ([[0.5]], [1.0]),
    ([[0.0, 0.0], [1.0, 1.0]], [0.5, 0.5]),
    ([[0.5]], [-1.0]),
    ([[0.0], [0.5], [1.0]], [0.25, 0.5, 0.25]),
    ([[0.1, 0.2, 0.3, 0.4]], [0.125]),
]


@pytest.mark.parametrize(("points", "weights"), _RULES)
def test_the_constructed_rule_agrees_field_by_field(points: Any, weights: Any) -> None:
    """Every piece of a directly built rule agrees, and the arrays bit for bit."""
    py, cpp = _both(lambda: QuadratureRule(points, weights))
    assert_object_parity(
        py=py,
        cpp=cpp,
        fields=_fields(_NOTHING_IS_COMPUTED, _NOTHING_IS_COMPUTED),
        context=f"QuadratureRule(points={points!r})",
    )


@pytest.mark.parametrize(("points", "weights"), _RULES)
def test_the_two_backends_hold_different_implementations(points: Any, weights: Any) -> None:
    """The comparison above is between two implementations, not one with itself.

    The failure this guards against is the one a parity suite cannot see in its own
    results: if the C++ branch silently fell back to the oracle, every assertion in
    this file would pass and none of them would mean anything.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    py, cpp = _both(lambda: QuadratureRule(points, weights))
    assert isinstance(py._impl, _QuadratureRulePython)
    assert isinstance(cpp._impl, _pantr_cpp.QuadratureRule)


@pytest.mark.parametrize("stored", ["points", "weights"])
def test_both_backends_hand_out_read_only_arrays(stored: str) -> None:
    """Neither backend lets a caller write through the arrays it hands back."""
    py, cpp = _both(lambda: QuadratureRule([[0.25, 0.75]], [1.0]))
    assert not getattr(py, stored).flags.writeable
    assert not getattr(cpp, stored).flags.writeable


def _random_axis_rules(rng: np.random.Generator, ndim: int) -> list[_AxisRule]:
    """Draw one random ``(nodes, weights)`` pair per axis.

    Nodes are sorted only so the rule reads like a rule; nothing in the tensor
    product depends on the order. Weights are drawn on ``(0, 1)`` so a product of
    up to eight of them stays well clear of the subnormal range, where the
    comparison would be about the format rather than about the port.

    Args:
        rng (np.random.Generator): The source of the draw.
        ndim (int): Number of axes.

    Returns:
        list[_AxisRule]: One pair per axis.
    """
    rules: list[_AxisRule] = []
    for _ in range(ndim):
        count = int(rng.integers(1, 5))
        nodes = np.sort(rng.random(count))
        rules.append((nodes, rng.random(count)))
    return rules


@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 8])
def test_the_tensor_product_agrees_field_by_field(ndim: int) -> None:
    """The tensor product agrees bit for bit, out to eight axes.

    Eight rather than three, because the claim at risk is the *order* the weight
    product accumulates in, and a two-axis product has only one association. Eight
    is also past where `np.sum` starts blocking pairwise, so a reduction that
    behaved like the additive one would show here and nowhere below it.
    """
    rng = np.random.default_rng(20260829 + ndim)
    for _ in range(20):
        rules = _random_axis_rules(rng, ndim)
        py, cpp = _both(functools.partial(tensor_product_quadrature, rules))
        assert_object_parity(
            py=py,
            cpp=cpp,
            fields=_fields(_THE_SAME_MULTIPLICATIONS, _THE_SAME_MULTIPLICATIONS),
            context=f"tensor_product_quadrature, ndim={ndim}",
        )


@pytest.mark.parametrize("npts", [(1,), (2,), (5,), (17,), (200,), (2, 3), (4, 4), (2, 3, 4)])
def test_the_gauss_legendre_rule_agrees_field_by_field(npts: tuple[int, ...]) -> None:
    """The Gauss-Legendre rule agrees, bit for bit on a build that cannot fuse."""
    ndim = len(npts)
    py, cpp = _both(lambda: gauss_legendre_quadrature(ndim, list(npts)))
    assert_object_parity(
        py=py,
        cpp=cpp,
        fields=_fields(_point_claim(py.points), _weight_claim(_reference_nodes(npts), py.weights)),
        context=f"gauss_legendre_quadrature{npts}",
    )


def test_the_gauss_legendre_rule_is_the_tensor_product_of_its_axes() -> None:
    """The factory equals the composition it claims to be, under both backends.

    This is what lets the bound above be *composed* from the 1-D one rather than
    re-derived: if the C++ factory built its rule by some other route, the 1-D
    kernel's bound would not be the thing it inherits.
    """
    from pantr.quad import get_gauss_legendre_1d  # noqa: PLC0415

    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            direct = gauss_legendre_quadrature(2, [3, 4])
            composed = tensor_product_quadrature(
                [get_gauss_legendre_1d(3), get_gauss_legendre_1d(4)]
            )
            assert direct.points.tobytes() == composed.points.tobytes(), backend
            assert direct.weights.tobytes() == composed.weights.tobytes(), backend


def test_the_composed_weight_bound_survives_a_perturbed_1d_rule() -> None:
    """The composition step of the weight bound holds, on a build that cannot fuse.

    The BOUNDED branch above is unreachable on this host, so the piece of it this
    ticket actually contributes -- how a 1-D weight bound composes onto a
    tensor-product weight -- would otherwise ship unexercised. This exercises it
    directly and without an FMA: perturb each 1-D weight by its own claimed
    one-sided budget with a random sign, push the perturbed and unperturbed rules
    through the *same* tensor product, and require the difference to stay inside
    the composed bound the claim would carry.

    Two stand-in backends are displaced in opposite directions by their own
    one-sided budgets, which is exactly the worst case the harness's two-sided
    tolerance admits, so the observed ratio approaches 1 from below and must never
    reach it.

    Twenty draws per dimension ship. Verified offline at 400 per dimension, 20x
    what ships: over 2000 rules of 1 to 5 axes no draw exceeded the bound, and the
    worst ratio was 0.99999999999999934 at ndim = 1, 0.9974 at 2, 0.9969 at 3,
    0.9952 at 4 and 0.9945 at 5. The bound is therefore tight rather than
    generous, which is what makes a violation of it informative; the widening
    slack with dimension is the ``ndim - 1`` product roundings, which the rules
    drawn here do not fully spend.
    """
    rng = np.random.default_rng(20260829)
    worst = _worst_composed_weight_ratio(rng, dimensions=(1, 2, 3, 4, 5), draws=20)
    # The bound must be approached, or it would be admitting anything. The
    # threshold is 0.9 rather than the observed 0.99999 because the observation is
    # a property of this host's rounding and the claim being made is only that the
    # bound is of the right order.
    assert worst > 0.9, f"the composed bound was never approached; worst ratio {worst:.6f}"


def _worst_composed_weight_ratio(
    rng: np.random.Generator, *, dimensions: Sequence[int], draws: int
) -> float:
    """Perturb 1-D rules by their own budget and report the worst ratio to the bound.

    Asserts as it goes, so a violation names the rule that produced it; the return
    value is what lets the caller check the bound is also *approached*. Separated
    from the test so the offline sweep the docstring quotes runs the same code.

    Args:
        rng (np.random.Generator): The source of the point counts and the signs.
        dimensions (Sequence[int]): Spatial dimensions to sweep.
        draws (int): Rules per dimension.

    Returns:
        float: The largest observed ratio of difference to the one-sided bound.

    Raises:
        AssertionError: If any draw exceeds the composed bound.
    """
    from pantr.quad import get_gauss_legendre_1d  # noqa: PLC0415

    unit_roundoff = float(np.finfo(np.float64).eps) / 2.0
    worst = 0.0
    for ndim in dimensions:
        for _ in range(draws):
            npts = [int(rng.integers(2, 12)) for _ in range(ndim)]
            reference = _reference_nodes(npts)
            with use_backend(Backend.PYTHON):
                axes = [get_gauss_legendre_1d(n, dtype=np.float64) for n in npts]
                budgets = [
                    unit_roundoff
                    * (NODE_DISPLACEMENT_UNITS * _sensitivity(x) + WEIGHT_FORMULA_ROUNDINGS)
                    for x in reference
                ]
                # Two stand-in backends, each displaced by its own one-sided
                # budget in the opposite direction, which is the worst case the
                # two-sided parity tolerance admits. Perturbing only one of them
                # and comparing against half the tolerance would leave the second
                # product's own rounding unaccounted for.
                shaken = [
                    [
                        (nodes, weights * (1.0 + sign * budget))
                        for (nodes, weights), budget in zip(axes, budgets, strict=True)
                    ]
                    for sign in (1.0, -1.0)
                ]
                low, high = (tensor_product_quadrature(rules) for rules in shaken)

            # The budget each axis ACTUALLY received, not the one aimed for: the
            # injection is itself a floating-point multiply and overshoots its
            # target by up to one rounding. Measuring what landed keeps this a test
            # of the composition rather than of the test's own arithmetic.
            landed = [
                np.maximum(np.abs(plus_weights - weights), np.abs(minus_weights - weights))
                / np.abs(weights)
                / unit_roundoff
                for (_, weights), (_, plus_weights), (_, minus_weights) in zip(
                    axes, shaken[0], shaken[1], strict=True
                )
            ]
            claim = bounded_parity(
                roundings=_PRODUCT_ROUNDINGS,
                accumulator=np.float64,
                storage=np.float64,
                amplification=_compose_relative_budgets(landed, low.weights),
                why="the composed bound under test; see this module's docstring",
            )
            tolerance = absolute_tolerance(claim)
            ratio = float(np.max(np.abs(high.weights - low.weights) / tolerance))
            worst = max(worst, ratio)
            assert ratio <= 1.0, (
                f"ndim={ndim}, npts={npts}: the composed weight bound was exceeded by "
                f"a factor of {ratio:.4f} by a perturbation at exactly its own budget"
            )
    return worst


def test_repr_is_the_same_string_under_both_backends() -> None:
    """The wrapper formats its own repr, so the backend cannot change what it says.

    ``tests/test_quad.py`` pins the exact string for one rule; this pins that the
    two backends produce the same one, which that test cannot see because it runs
    under one backend at a time.
    """
    py, cpp = _both(lambda: QuadratureRule([[0.5, 0.5]], [1.0]))
    assert repr(py) == "QuadratureRule(ndim=2, num_points=1)"
    assert repr(cpp) == repr(py)


@pytest.mark.parametrize("reader", [Backend.PYTHON, Backend.CPP])
@pytest.mark.parametrize("writer", [Backend.PYTHON, Backend.CPP])
def test_pickle_round_trips_across_every_backend_pair(writer: Backend, reader: Backend) -> None:
    """A pickle written under one backend loads under the other, all four ways.

    ``__reduce__`` stores points and weights rather than the implementation
    handle, precisely so the backend switch cannot become a data-format switch.
    ``pantr.mpi`` sends these across collective calls, so a rule pickled by one
    rank and unpickled by another must not depend on how each was configured.
    """
    with use_backend(writer):
        original = gauss_legendre_quadrature(2, 3)
        blob = pickle.dumps(original)
    with use_backend(reader):
        restored = pickle.loads(blob)

    assert type(restored) is QuadratureRule
    assert restored.__module__ == "pantr.quad"
    assert restored.points.tobytes() == original.points.tobytes()
    assert restored.weights.tobytes() == original.weights.tobytes()
    assert restored.ndim == original.ndim
    assert restored.num_points == original.num_points


def _message_of(fn: Callable[[], object]) -> str:
    """Run ``fn`` and return the text of the ``ValueError`` it raises.

    Args:
        fn (Callable[[], object]): A zero-argument call expected to raise.

    Returns:
        str: The exception's message.

    Raises:
        AssertionError: If ``fn`` did not raise ``ValueError``.
    """
    try:
        fn()
    except ValueError as exc:
        return str(exc)
    raise AssertionError("expected a ValueError and got none")


@pytest.mark.parametrize(
    ("build", "what"),
    [
        (lambda: QuadratureRule([0.5, 0.5], [1.0]), "points not 2D"),
        (lambda: QuadratureRule([[0.5]], [[1.0]]), "weights not 1D"),
        (lambda: QuadratureRule([[0.5], [0.5]], [1.0]), "length mismatch"),
        (lambda: QuadratureRule(np.empty((0, 2)), []), "no points"),
        (lambda: QuadratureRule(np.empty((1, 0)), [1.0]), "no axes"),
        (lambda: QuadratureRule([[np.inf, 0.5]], [1.0]), "non-finite point"),
        (lambda: QuadratureRule([[np.nan, 0.5]], [1.0]), "NaN point"),
        (lambda: QuadratureRule([[0.5, 0.5]], [np.nan]), "non-finite weight"),
        (lambda: QuadratureRule([[-0.1, 0.5]], [1.0]), "below the cube"),
        (lambda: QuadratureRule([[0.5, 1.5]], [1.0]), "above the cube"),
        (lambda: tensor_product_quadrature([]), "no axes at all"),
        (
            lambda: tensor_product_quadrature([(np.array([0.5, 0.5]), np.array([1.0]))]),
            "axis lengths disagree",
        ),
        (
            lambda: tensor_product_quadrature([(np.zeros(0), np.zeros(0))]),
            "an empty axis",
        ),
        (
            lambda: tensor_product_quadrature([(np.array([[0.5]]), np.array([1.0]))]),
            "axis nodes not 1D",
        ),
        (
            lambda: tensor_product_quadrature([(np.array([2.0]), np.array([1.0]))]),
            "an axis node outside the cube",
        ),
        (lambda: gauss_legendre_quadrature(0, 3), "ndim zero"),
        (lambda: gauss_legendre_quadrature(-1, 3), "ndim negative"),
        (lambda: gauss_legendre_quadrature(2, [2]), "npts of the wrong length"),
        (lambda: gauss_legendre_quadrature(2, [2, 0]), "an npts entry below one"),
        (lambda: gauss_legendre_quadrature(1, 0), "a scalar npts below one"),
    ],
)
def test_error_messages_agree_verbatim(build: Callable[[], object], what: str) -> None:
    """Both backends raise ``ValueError`` and say **exactly** the same thing.

    Verbatim, not a substring. ``tests/parity/test_geometry_aabb.py`` records what
    a substring match cost there: it passed while five of six messages differed.
    The message is part of the claim rather than decoration, because a caller that
    catches on it would otherwise see ``PANTR_BACKEND`` change what the library
    says rather than only how fast it is.

    The list crosses the two layers this port splits its checks across: the ranks
    and the ``(ndim, npts)`` convention, which the wrapper owns because no C++
    exception can reach Python as a ``TypeError``, and everything else, which the
    C++ type owns.
    """
    with use_backend(Backend.PYTHON):
        oracle = _message_of(build)
    with use_backend(Backend.CPP):
        ported = _message_of(build)
    assert oracle == ported, f"{what}: oracle said {oracle!r}, C++ said {ported!r}"
