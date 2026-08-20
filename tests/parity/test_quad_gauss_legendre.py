r"""Parity of the Gauss-Legendre rule against its Python oracle.

`cpp/include/pantr/quad/legendre.hpp` names this file as the place its parity
claims are derived. `design/backend_parity.md` Rule 4 carries the analysis the
bounded branch rests on, and this is where that analysis becomes an assertion.

On the shipped build the claim is BITWISE, and that is measured
-----------------------------------------------------------------

The kernel is an operation-for-operation transliteration and uses only `+`, `-`,
`*` and `/`, each of which IEEE 754 pins to one correctly rounded result. The one
operation the standard does not pin is `cos`, called once per node for Tricomi's
starting guess -- so if a platform's `cos` disagrees with numpy's, this file's
BITWISE assertions are what will say so, and `cos` is the only candidate.

Measured on this build before the assertions were written: 2143 nodes and 2143
weights over n from 1 to 700, **zero differ**, in float64 and in float32 alike.
`objdump -d` finds no `vfmadd` in the extension, which is why: baseline x86-64 has
no fused multiply-add, so `-ffp-contract=on` has nothing to fuse.

**The refutation succeeds where it should.** Rebuilt with `-march=native` the
compiler emits six FMA instructions and 1994 of 2143 values move. Adding
`-ffp-contract=off` on top restores bit-identity exactly, which isolates the
mechanism to contraction alone -- not vectorisation, and not a different cosine.
So the claim below is a property of the BUILD, selected from
``__fp_contract__`` at run time, exactly as the cardinal B-spline's is.

The bounded branch, for the build that fuses
--------------------------------------------

Unreachable today and derived anyway, because the day `design/simd.md`'s stage 2
turns on the ISA ladder it becomes the live branch and a bound derived under
pressure is a bound fitted to whatever the measurement said.

**Nodes.** Rule 4: writing `C(n, x)` for a bound on `|fl(P_n(x)) - P_n(x)| / u`,
the supremum of `C` over `x` is `Theta(n^2)` with closed form
`C(n, +-1) = (7/4) n^2 + (9/4) n - 4 H_n`. What the node displacement inherits is
not that numerator but the **ratio** `C(n, x_i) / |P'_n(x_i)|` at the Gauss nodes,
which is bounded by `7/2` uniformly in `n` -- both blow up at the same rate in the
same place, each carrying `sqrt(n) (1 - x^2)^{-3/4}`, and the cancellation is why
the displacement is flat in `n` rather than growing. Adding one rounding for the
Newton update's own subtraction gives `9/2` units of roundoff, absolute, at every
degree.

**Weights.** `w = 2 / ((1 - x^2) d^2)` with `d = P'_n(x)`, so

.. math::

    \frac{\mathrm{d}\log w}{\mathrm{d}x}
      = \frac{2x}{1 - x^2} - \frac{2 d'}{d} .

At a Gauss node `P_n` vanishes, and the Legendre differential equation
`(1 - x^2) P'' - 2x P' + n(n+1) P = 0` therefore collapses to `P'' = 2x P' / (1 - x^2)`,
so `d'/d = 2x / (1 - x^2)` and the two terms leave

.. math::

    \left| \frac{\delta w}{w} \right| = \frac{2|x|}{1 - x^2} \, |\delta x| .

That factor is `Theta(n^2)` at the outermost node, where `1 - x^2` is about
`2c/n^2`. **A per-element relative bound with a constant is therefore refuted**,
and the absolute bound is what survives -- because the endpoint weight decays like
`1/n^2` and cancels it. Measured, and it is the load-bearing claim of the weight
bound: `max_i |w_i| \cdot 2|x_i|/(1 - x_i^2)` is monotone in `n` and converges,
reaching 1.732 at n = 2, 2.5663 at n = 512 and 2.566322 at n = 2000. **SUPPORTED**
by that sweep, not proved. :func:`test_the_weight_amplification_stays_bounded`
re-measures it, so a degree where it stops holding fails there rather than inside
a bound.

Adding the weight formula's own five roundings -- `x*x`, `1 - x*x`, two multiplies
by `d`, and the final division -- the elementwise absolute bound is

.. math::

    |\delta w_i| \le u \left( \frac{9}{2} \cdot \frac{2|x_i|}{1 - x_i^2} |w_i|
                              + 5 |w_i| \right) ,

doubled by the harness because neither backend is the exact rule.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Final, TypeVar

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.quad import get_gauss_legendre_1d
from pantr.quad._rules import _gauss_legendre_symmetric
from tests._parity_harness import (
    FloatArray,
    ParityClaim,
    Roundings,
    absolute_tolerance,
    assert_parity,
    bitwise_parity,
    bounded_parity,
    contraction_may_fuse,
)

DTYPES: Final = (np.float64, np.float32)
"""The two storage formats the quadrature layer accepts."""

N_PTS: Final = (1, 2, 3, 4, 5, 17, 64, 200, 1000)
"""Point counts swept, the legal corners of the adversarial sweep's ``_N_PTS_FULL``."""

NEWTON_RATIO_BOUND: Final = 3.5
"""``sup_i C(n, x_i) / |P'_n(x_i)|`` at the Gauss nodes, uniform in ``n``.

``design/backend_parity.md`` Rule 4, SUPPORTED there by an exhaustive sweep over
every ``n`` from 2 to 600 and selectively to 2048, worst value 3.4999940516,
monotone and always attained at the outermost node.
"""

NODE_DISPLACEMENT_UNITS: Final = NEWTON_RATIO_BOUND + 1.0
"""Units of roundoff a node may move, absolutely, on a build that fuses.

The ratio above, plus one rounding for the Newton update's own subtraction.
Flat in ``n``: that is the point of inheriting a ratio rather than a numerator.
"""

WEIGHT_FORMULA_ROUNDINGS: Final = 5
"""Roundings in ``2 / ((1 - x*x) * d * d)``: the square, the subtraction, two
multiplies and the division."""

WEIGHT_AMPLIFICATION_SUP: Final = 2.6
"""``sup_i |w_i| * 2|x_i| / (1 - x_i^2)``, uniform in ``n``.

The cancellation that makes the weight bound absolute rather than growing:
``2|x|/(1 - x^2)`` is ``Theta(n^2)`` at the outermost node while the weight there
decays like ``1/n^2``. Measured monotone and convergent -- 1.732 at n = 2, 2.5663
at n = 512, 2.566322 at n = 2000 -- and rounded up to 2.6.
**SUPPORTED by that sweep, not proved.**
"""

_Result = TypeVar("_Result")


def _both_backends(build: Callable[[], _Result]) -> tuple[_Result, _Result]:
    """Run a rule builder once under each backend.

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


_EXACT_BY_BUILD: Final[str] = (
    "the kernel is an operation-for-operation transliteration using only +, -, * "
    "and / , each pinned by IEEE 754 to one correctly rounded result, and this "
    "build emits no fused multiply-add: objdump finds no vfmadd in the extension, "
    "because baseline x86-64 has no FMA for -ffp-contract=on to use. Measured: "
    "2143 nodes and 2143 weights over n from 1 to 700, zero differ. The only "
    "operation IEEE 754 does not pin is cos, called once per node for Tricomi's "
    "start; if this assertion fails on some platform, that is the candidate."
)

_BOUNDED_BY_FMA: Final[str] = (
    "this build fuses a multiply-add, so the two backends commit different "
    "numbers of roundings inside the Legendre recurrence. Rebuilt with "
    "-march=native the compiler emits six FMA instructions and 1994 of 2143 "
    "values move; -ffp-contract=off on top restores bit-identity exactly, which "
    "isolates the mechanism to contraction rather than to vectorisation or to a "
    "different cosine."
)


def _fused_node_claim(nodes: FloatArray) -> ParityClaim:
    """State the parity claim for the Gauss-Legendre nodes.

    Args:
        nodes (FloatArray): The reference nodes, for their shape.

    Returns:
        ParityClaim: The BOUNDED claim for a build that fuses.
    """
    amplification = np.full(np.shape(nodes), NODE_DISPLACEMENT_UNITS, dtype=np.float64)
    return bounded_parity(
        roundings=Roundings(stages=1, accumulator_per_stage=1, storage_per_stage=0),
        accumulator=np.float64,
        storage=np.float64,
        amplification=amplification,
        why=(
            f"{_BOUNDED_BY_FMA} A node inherits the RATIO C(n, x_i) / |P'_n(x_i)|, "
            f"bounded by {NEWTON_RATIO_BOUND} uniformly in n, not the Theta(n^2) "
            f"numerator: both blow up at the same rate in the same place and cancel. "
            f"Plus one rounding for the Newton update's own subtraction gives "
            f"{NODE_DISPLACEMENT_UNITS} units of roundoff, absolute and flat in n. "
            f"Feeding the x-uniform (7/4) n^2 into the same division instead would "
            f"give 7e4 u at n = 1000 against a measured 0.5 u, so dropping the "
            f"(1 - x^2)^(-3/4) does not give a conservative answer."
        ),
    )


def _fused_weight_claim(nodes: FloatArray, weights: FloatArray) -> ParityClaim:
    """State the parity claim for the Gauss-Legendre weights.

    Args:
        nodes (FloatArray): The reference nodes, which set the amplification.
        weights (FloatArray): The reference weights, which set the magnitude.

    Returns:
        ParityClaim: The BOUNDED claim for a build that fuses.
    """
    x = np.abs(np.asarray(nodes, dtype=np.float64))
    w = np.abs(np.asarray(weights, dtype=np.float64))
    # d(log w)/dx = -2x/(1 - x^2) at a Gauss node, from the Legendre ODE. The
    # amplification array carries magnitude times dimensionless factor, because
    # the comparison happens in an absolute frame on a quantity spanning decades.
    sensitivity = np.zeros_like(w)
    interior = x < 1.0
    sensitivity[interior] = 2.0 * x[interior] / (1.0 - x[interior] * x[interior])
    amplification = w * (NODE_DISPLACEMENT_UNITS * sensitivity + WEIGHT_FORMULA_ROUNDINGS)
    return bounded_parity(
        roundings=Roundings(stages=1, accumulator_per_stage=1, storage_per_stage=0),
        accumulator=np.float64,
        storage=np.float64,
        amplification=amplification,
        why=(
            f"{_BOUNDED_BY_FMA} A displaced node moves its weight by "
            f"|dw/w| = 2|x|/(1 - x^2) |dx|, from the Legendre ODE collapsing to "
            f"P'' = 2x P'/(1 - x^2) at a root. That factor is Theta(n^2) at the "
            f"outermost node, so a per-element RELATIVE bound with a constant is "
            f"refuted; the absolute one survives because the endpoint weight decays "
            f"like 1/n^2 and cancels it. Plus the weight formula's own "
            f"{WEIGHT_FORMULA_ROUNDINGS} roundings."
        ),
    )


def _node_claim(nodes: FloatArray) -> ParityClaim:
    """Select the node claim this build supports.

    Args:
        nodes (FloatArray): The reference nodes.

    Returns:
        ParityClaim: BITWISE where the build cannot fuse, otherwise BOUNDED.
    """
    if not contraction_may_fuse():
        return bitwise_parity(why=_EXACT_BY_BUILD)
    return _fused_node_claim(nodes)


def _weight_claim(nodes: FloatArray, weights: FloatArray) -> ParityClaim:
    """Select the weight claim this build supports.

    Args:
        nodes (FloatArray): The reference nodes.
        weights (FloatArray): The reference weights.

    Returns:
        ParityClaim: BITWISE where the build cannot fuse, otherwise BOUNDED.
    """
    if not contraction_may_fuse():
        return bitwise_parity(why=_EXACT_BY_BUILD)
    return _fused_weight_claim(nodes, weights)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_pts", N_PTS)
def test_the_public_rule_agrees_across_backends(
    n_pts: int, dtype: npt.DTypeLike, cpp_backend: None
) -> None:
    """The rule a caller gets agrees, through the mapping and the narrowing.

    Args:
        n_pts (int): Number of points.
        dtype (npt.DTypeLike): Storage format.
        cpp_backend (None): Requires the compiled extension.
    """
    (py_nodes, py_weights), (cpp_nodes, cpp_weights) = _both_backends(
        functools.partial(get_gauss_legendre_1d, n_pts, dtype)
    )
    assert_parity(
        cpp_nodes,
        py_nodes,
        _node_claim(py_nodes),
        context=f"Gauss-Legendre nodes on [0, 1], n={n_pts}, {dtype}",
    )
    assert_parity(
        cpp_weights,
        py_weights,
        _weight_claim(py_nodes, py_weights),
        context=f"Gauss-Legendre weights on [0, 1], n={n_pts}, {dtype}",
    )


@pytest.mark.parametrize("n_pts", N_PTS)
def test_the_unmapped_rule_agrees_across_backends(n_pts: int, cpp_backend: None) -> None:
    """The kernel's own output agrees, before the shared map onto [0, 1].

    Compared separately from the public rule and not instead of it. The map and
    the narrowing are Python and shared, so they are common mode and cancel; that
    is why they are kept out of the C++ in the first place. Testing only the
    mapped rule would therefore leave the claim resting on that cancellation
    rather than on the kernel, and testing only the kernel would not exercise the
    path a caller takes.

    Args:
        n_pts (int): Number of points.
        cpp_backend (None): Requires the compiled extension.
    """
    (py_nodes, py_weights), (cpp_nodes, cpp_weights) = _both_backends(
        functools.partial(_gauss_legendre_symmetric, n_pts)
    )
    assert_parity(
        cpp_nodes,
        py_nodes,
        _node_claim(py_nodes),
        context=f"Gauss-Legendre nodes on [-1, 1], n={n_pts}",
    )
    assert_parity(
        cpp_weights,
        py_weights,
        _weight_claim(py_nodes, py_weights),
        context=f"Gauss-Legendre weights on [-1, 1], n={n_pts}",
    )


@pytest.mark.parametrize("n_pts", [n for n in N_PTS if n > 1])
def test_the_rule_is_symmetric_to_the_last_bit_in_both_backends(
    n_pts: int, cpp_backend: None
) -> None:
    """Each node is the exact negation of its partner, in both implementations.

    Not a parity assertion and not covered by one: two backends could agree with
    each other and both be asymmetric. The kernels claim exactness rather than a
    tolerance here, because each pair is written from one root by negation, which
    is a sign flip and cannot round.

    Args:
        n_pts (int): Number of points.
        cpp_backend (None): Requires the compiled extension.
    """
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            nodes, weights = _gauss_legendre_symmetric(n_pts)
        assert np.array_equal(nodes, -nodes[::-1]), (
            f"{backend.name}: the nodes are not exact negations of each other at "
            f"n={n_pts}, so the pair was not written from a single root"
        )
        assert np.array_equal(weights, weights[::-1]), (
            f"{backend.name}: the weights are not mirror-equal at n={n_pts}"
        )


def test_the_weight_amplification_stays_bounded(cpp_backend: None) -> None:
    """Re-measure the cancellation the weight bound rests on.

    ``|w_i| * 2|x_i| / (1 - x_i^2)`` is a product of a factor that grows like
    ``n^2`` and a weight that decays like ``n^-2``, and the whole weight bound is
    absolute rather than growing only because the two cancel. That is SUPPORTED by
    a sweep, not proved, so a degree where it stops holding must fail here -- with
    this message -- rather than inside a bound that silently stopped bounding.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    worst = 0.0
    for n_pts in (2, 3, 5, 17, 64, 200, 512, 1000):
        with use_backend(Backend.PYTHON):
            nodes, weights = _gauss_legendre_symmetric(n_pts)
        x = np.abs(nodes)
        amplification = np.abs(weights) * 2.0 * x / (1.0 - x * x)
        worst = max(worst, float(amplification.max()))

    assert worst <= WEIGHT_AMPLIFICATION_SUP, (
        f"the weight amplification reached {worst:.6f}, past the "
        f"{WEIGHT_AMPLIFICATION_SUP} the bound is derived from. The two Theta(n^2) "
        f"factors have stopped cancelling, so the weight bound is no longer absolute "
        f"and re-deriving it is the fix, not widening this constant."
    )
    assert worst > 0.5 * WEIGHT_AMPLIFICATION_SUP, (
        f"the weight amplification only reached {worst:.6f} against a stated "
        f"supremum of {WEIGHT_AMPLIFICATION_SUP}, so the sweep is no longer "
        f"visiting the regime the constant was measured in"
    )


@pytest.mark.parametrize("n_pts", (5, 64, 1000))
def test_the_bounded_branch_admits_a_perturbation_inside_its_own_bound(n_pts: int) -> None:
    """The FMA-build bound accepts a displacement it says it accepts.

    The bounded branch is unreachable on a build with no fused multiply-add, so
    without this it would ship unexecuted and its first run would be the day the
    ISA ladder lands. Driving the claim directly exercises it now.

    Three quarters of the tolerance rather than all of it, following the same
    probe in ``test_basis_cardinal_bspline.py``: the addition that applies the
    perturbation rounds, so a displacement of exactly the bound can land a hair
    outside it and the probe would be testing its own arithmetic.

    Needs no C++ backend: what is under test is the claim's own arithmetic.

    Args:
        n_pts (int): Number of points.
    """
    with use_backend(Backend.PYTHON):
        nodes, weights = _gauss_legendre_symmetric(n_pts)

    for name, reference, claim in (
        ("nodes", nodes, _fused_node_claim(nodes)),
        ("weights", weights, _fused_weight_claim(nodes, weights)),
    ):
        tolerance = absolute_tolerance(claim)
        assert float(np.max(tolerance)) > 0.0, (
            f"{name}: the bounded branch derived a zero tolerance at n={n_pts}"
        )
        deviation = assert_parity(
            reference + 0.75 * tolerance,
            reference,
            claim,
            context=f"{name} sensitivity probe, inside the bound, n={n_pts}",
        )
        assert deviation.max_ratio_to_bound > 0.5, (
            f"{name}: a displacement of three quarters of the bound registered as "
            f"{deviation.max_ratio_to_bound:.3g} of it, so the perturbation did not "
            f"survive its own rounding and this probe tested nothing"
        )


@pytest.mark.parametrize("n_pts", (5, 64, 1000))
def test_the_bounded_branch_rejects_a_perturbation_past_its_own_bound(n_pts: int) -> None:
    """The FMA-build bound refuses a displacement past what it allows.

    The other half, and the one that decides whether the bound is a bound at all:
    a tolerance nothing can exceed would pass the probe above unchanged.

    Args:
        n_pts (int): Number of points.
    """
    with use_backend(Backend.PYTHON):
        nodes, weights = _gauss_legendre_symmetric(n_pts)

    for name, reference, claim in (
        ("nodes", nodes, _fused_node_claim(nodes)),
        ("weights", weights, _fused_weight_claim(nodes, weights)),
    ):
        perturbed = reference + 1.5 * absolute_tolerance(claim)
        with pytest.raises(AssertionError, match="more than the derived bound"):
            assert_parity(
                perturbed,
                reference,
                claim,
                context=f"{name} sensitivity probe, past the bound, n={n_pts}",
            )
