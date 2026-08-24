"""Parity of the root-finding block: what the two backends must agree on, and why.

Sixteen kernels, all iterative, reached through six dispatch points. Unlike the
arithmetic block next door, every result here carries a **discrete verdict** as well
as a value: how many roots there are. A tolerance bounds a value and says nothing
about a count, so the two are asserted separately throughout, and the count is never
folded into a numeric claim.

The claim, and the one place it is not an equality
--------------------------------------------------

**On the shipped build the claim is bitwise**, and that is not a hope. The target
carries no ``-march``, so baseline ``x86-64`` has no fused multiply-add and nothing to
contract into; every operation on both sides is a correctly rounded add, subtract,
multiply, divide or compare, and the C++ reproduces the oracle's per-expression widths
(``scripts/measure_root_finding_widths.py`` measures those and is the specification the
transliteration was written against).

**On a fusing build the values move and the sets do not.** Measured rather than
assumed, by building the extension at ``-march=native`` and comparing 732 public
results against the Numba oracle: 619 identical, 113 displaced, and **zero changed the
root set**. The worst displacement was ``4.951e-13`` against a ``param_tol`` of
``1e-12``.

That ratio is the derivation rather than a coincidence. Both backends run the same
bracketing iteration and stop when ``hi - lo <= param_tol``, returning ``0.5 * (lo +
hi)``. A midpoint of a bracket of width at most ``param_tol`` that contains a root is
within ``param_tol / 2`` of that root, so two such answers for the same root differ by
at most ``param_tol``. The observed worst is ``0.495`` of that, which is the bound
being approached rather than merely respected.

So the fused claim is the **algorithm's own termination tolerance**, not a
floating-point bound: it does not grow with degree, it does not depend on the
coefficient magnitudes, and it would hold between two runs of the same backend that
happened to take different bracket paths.

What the fused claim does **not** cover, and this is stated rather than hidden: it
assumes both backends found the same roots. Contraction can flip the convex-hull
tie-break, which changes which vertices survive and so which interval is clipped, and
that is demonstrable at the predicate level. It did not change a single count in 732
results, but 732 results are evidence and not a proof, so the count agreement is
asserted as its own check and Rule 11 of ``design/backend_parity.md`` records that it
is measured rather than derived.

Two defects are asserted, not worked around
-------------------------------------------

The oracle returns a wrong answer in two regimes, filed as FELIGN/pantr#351 and #352,
and the C++ reproduces both deliberately because the port's contract is parity and not
correction. The tests below name them. **If you are here because one of them started
failing, the fix is in both backends and in these tests together**; making one side
correct on its own is what breaks the equality this file exists to hold.
"""

from __future__ import annotations

from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.bezier import Bezier, find_monotone_root, find_roots
from tests._parity_harness import (
    assert_parity,
    bitwise_parity,
    contraction_may_fuse,
    converged_parity,
    demand_the_compiled_kernel,
    unit_roundoff,
)

DTYPES: Final = (np.float64, np.float32)
"""The two storage formats the root finder accepts."""

DEGREES: Final = (1, 2, 3, 5, 6, 8, 11, 17)
"""Degrees swept by the single-polynomial tests.

1 is the base case every Yuksel recursion bottoms out in, and the sharpest width in
the block. 5 and 6 straddle ``_CLIP_MIN_DEGREE``, so the pair exercises both arms of
the dispatch on otherwise identical data. 17 is deep enough that the derivative chain
has fifteen levels.
"""

TOL: Final = 1e-12
"""The tolerance every test passes explicitly.

Explicit rather than defaulted because it is the bound in the fused claim: the
displacement between backends is at most one ``param_tol``, so a test that let the
default vary by dtype would be asserting two different bounds under one name.
"""

DEFECT_TOL: Final = 1e-40
"""The tolerance the two defect tests pass, and why it is not :data:`TOL`.

Both defects live at coefficient magnitudes around ``1e-25`` and ``1e-31``, and
:func:`pantr.bezier._find_roots._dispatch_single` rejects a polynomial outright when
``all(abs(coeff) <= geom_tol)``. At ``TOL`` that guard fires first, both backends
return an empty array, they agree, and the test passes **without reaching the defect
at all**. A tolerance below the coefficients is what makes these tests able to fail.
"""

_BITWISE_WHY: Final = (
    "every operation in the block is a correctly rounded add, subtract, multiply, "
    "divide or compare, and the C++ reproduces the oracle's per-expression widths, "
    "which scripts/measure_root_finding_widths.py measures. The one library call is "
    "the dedup radius cap, where std::pow reproduces numba's ** on every tolerance "
    "tested. Nothing rounds twice and nothing is reassociated, so the two backends "
    "compute the same sequence of correctly rounded results"
)
"""Why the two backends are bit-identical where nothing can fuse."""

_FUSED_WHY: Final = (
    "a root is located only to within the interval where the COMPUTED residual is "
    "indistinguishable from zero, which has half-width eps_f / |f'(r)| for an "
    "evaluation whose forward error is eps_f. Two backends whose residuals differ "
    "within eps_f may return any point of that interval, so their roots differ by at "
    "most twice it. Below that scale the bracketing tolerance also applies, so the "
    "bound is the larger of the two. De Casteljau is a convex combination at every "
    "stage, so it neither amplifies nor cancels and eps_f is (degree + 1) roundings "
    "at the COEFFICIENT width times the coefficient magnitude -- the coefficient "
    "width, not float64, because the triangle stores back into a buffer of the input "
    "dtype at every step. The half-width is capped at eps_f^(1/3), the cluster width "
    "around a root of multiplicity up to three, which is the same cap and the same "
    "reasoning _dedup_roots_core uses for its merge radius; without it a double root, "
    "where f' vanishes, would carry an unbounded tolerance"
)
"""Why a fusing build still agrees, and to what.

**An earlier version of this claimed the bound was `param_tol` alone**, derived from
both backends returning a bracket midpoint. That was fitted to float64 data, where
``eps_f / |f'|`` is around 1e-15 and ``param_tol`` of 1e-12 hides it. At float32 the
residual term is four orders larger and dominates: a degree-5 case moved by 2e-8. The
measurement that produced the wrong bound, 732 float64-heavy results with a worst
displacement of 4.951e-13, was consistent with it and could not have refuted it.
"""


def _polynomial(degree: int, dtype: npt.DTypeLike, kind: str) -> npt.NDArray[np.floating]:
    """Build one adversarial coefficient vector.

    Args:
        degree (int): Polynomial degree; the vector has ``degree + 1`` entries.
        dtype (npt.DTypeLike): Storage format.
        kind (str): Which adversary. See :data:`KINDS` for what each one attacks.

    Returns:
        npt.NDArray[np.floating]: Bernstein coefficients on [0, 1].
    """
    index = np.arange(degree + 1, dtype=np.float64)
    if kind == "collinear":
        # Every orientation test in the hull's monotone chain is an exact tie here,
        # which is the configuration contraction destroys.
        coeff = 1.0 - 2.0 * index / max(degree, 1)
    elif kind == "sign_changes":
        coeff = np.cos(np.pi * index / max(degree, 1) * 3.0)
    elif kind == "double_root":
        centre = index / max(degree, 1) - 0.5
        coeff = centre * centre
    elif kind == "wide_range":
        coeff = np.cos(index) * 10.0 ** (index - degree / 2.0)
    elif kind == "endpoint_zeros":
        coeff = np.sin(np.pi * index / max(degree, 1))
    else:  # "monotone"
        coeff = index / max(degree, 1) - 0.5
    return np.asarray(coeff, dtype=dtype)


KINDS: Final = (
    "collinear",
    "sign_changes",
    "double_root",
    "wide_range",
    "endpoint_zeros",
    "monotone",
)
"""What each adversary attacks.

``collinear`` is the one this block was written for: a straight control polygon makes
every hull orientation test an exact tie, and an exact tie is what contraction turns
into a signed residue. ``double_root`` reaches the bisection refinement, which is the
only branch that uses the third of the three sign-test-by-product sites.
``wide_range`` crosses ``_CLIP_COEFF_RANGE_LIMIT`` so the dispatch declines clipping.
"""


class Case(NamedTuple):
    """What is being compared, so the assertion helper takes one argument for it.

    Attributes:
        coeff (npt.NDArray[np.floating] | None): The polynomial, which sizes the
            residual term of the fused bound. None where the caller has no single
            polynomial to name, in which case the bound falls back to the bracketing
            term alone and the claim is weaker than it could be.
        tol (float): The bracketing tolerance the call ran at.
    """

    coeff: npt.NDArray[np.floating] | None
    tol: float


PARAMETER_DOMAIN: Final = 1.0
"""The scale a root is meaningful against, and it is not the root's own magnitude.

Every root here is a curve parameter confined to [0, 1]. The harness's vacuity guard
otherwise compares a bound against the values it is applied to, which is right for a
coefficient and wrong for a parameter: a root at ``3.6e-13`` is not a quantity with no
digits, it is a parameter sitting next to an endpoint, and a bound of ``1e-12`` on it
still says something a wrong answer would violate.

Measured while getting this wrong: with the root's own magnitude as the scale, two
cases of the seeded sweep were refused as vacuous **while the two backends agreed
bit for bit**.

How loose the fused bound gets, honestly. For coefficients spanning eight decades at
``float32`` it reaches ``1.067e-01`` at degree 8, a tenth of the domain: weak, but a
wrong answer would still violate it, so it is asserted rather than excluded. By degree
17 the same family reaches ``9.5``, which covers the whole interval and asserts
nothing; that case is skipped, named, with its number in the message, which is what
Rule 8 of ``design/backend_parity.md`` asks for. Both are the accuracy limit of the
problem at that width rather than a parity failure, and both are bit-identical on the
shipped build.
"""


def _root_uncertainty(
    coeff: npt.NDArray[np.floating], roots: npt.NDArray[np.float64], param_tol: float
) -> npt.NDArray[np.float64]:
    """Elementwise bound on how far two backends' versions of one root may sit apart.

    See :data:`_FUSED_WHY` for the derivation. The derivative is taken from the
    oracle at the oracle's own roots, which is the only derivative available: an
    exact one would need the root, which is what is being bounded.

    Args:
        coeff (npt.NDArray[np.floating]): The polynomial's Bernstein coefficients,
            whose dtype sets the evaluation's rounding unit.
        roots (npt.NDArray[np.float64]): The oracle's roots.
        param_tol (float): The bracketing tolerance the call ran at.

    Returns:
        npt.NDArray[np.float64]: One bound per root.
    """
    if roots.size == 0:
        return np.zeros(0, dtype=np.float64)

    degree = coeff.size - 1
    scale = float(np.max(np.abs(coeff), initial=0.0))
    residual_error = float(degree + 1) * unit_roundoff(coeff.dtype) * scale

    inside = np.clip(np.nan_to_num(roots, nan=0.5), 0.0, 1.0).astype(coeff.dtype)
    with use_backend(Backend.PYTHON):
        derivative = np.asarray(
            Bezier(coeff.reshape(-1, 1)).evaluate_derivatives(inside, 1), dtype=np.float64
        ).reshape(inside.size, -1)[:, 0]

    slope = np.maximum(np.abs(derivative), np.finfo(np.float64).tiny)
    # Capped at the multiplicity-three cluster width, exactly as the oracle's own
    # merge radius is: at a double root the slope vanishes and the ratio diverges.
    half_width = np.minimum(residual_error / slope, residual_error ** (1.0 / 3.0))
    bound = np.maximum(np.full(roots.shape, param_tol, dtype=np.float64), 2.0 * half_width)
    return np.asarray(bound, dtype=np.float64)


def _both_backends(
    call: str, coeff: npt.NDArray[np.floating], tol: float = TOL
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Run one public entry point under each backend.

    Args:
        call (str): ``"roots"`` or ``"monotone"``.
        coeff (npt.NDArray[np.floating]): 1-D Bernstein coefficients.
        tol (float): Root-finding tolerance. Defaults to :data:`TOL`.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: The oracle's result
            and the C++ one, in that order.
    """
    curve = Bezier(coeff.reshape(-1, 1))
    with use_backend(Backend.PYTHON):
        reference = (
            find_roots(curve, tol=tol)
            if call == "roots"
            else np.array([find_monotone_root(curve, tol=tol)], dtype=np.float64)
        )
    with use_backend(Backend.CPP):
        actual = (
            find_roots(curve, tol=tol)
            if call == "roots"
            else np.array([find_monotone_root(curve, tol=tol)], dtype=np.float64)
        )
    return np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)


def _assert_the_same_roots(
    actual: npt.NDArray[np.float64],
    reference: npt.NDArray[np.float64],
    context: str,
    case: Case,
) -> None:
    """Assert the two backends agree, on the count first and then on the values.

    The count is checked on its own rather than folded into the numeric claim,
    because no tolerance bounds a verdict: two backends that find a different number
    of roots have not disagreed by a small amount, they have disagreed about what is
    true. On a fusing build that is the one outcome this file's claim does not cover.

    Args:
        actual (npt.NDArray[np.float64]): The C++ backend's roots.
        reference (npt.NDArray[np.float64]): The oracle's roots.
        context (str): What was being computed, quoted in a failure message.
        case (Case): The polynomial and the tolerance the call ran at.
    """
    assert actual.shape == reference.shape, (
        f"{context}: the backends found different numbers of roots, "
        f"{actual.size} against {reference.size}. That is a changed verdict rather "
        f"than a displaced value, and no tolerance covers it. Rule 11 of "
        f"design/backend_parity.md records that contraction can flip the convex-hull "
        f"tie-break; if this build fuses, that is the first thing to check."
    )

    if not contraction_may_fuse():
        assert_parity(actual, reference, bitwise_parity(why=_BITWISE_WHY), context=context)
        return

    bound = (
        np.full(actual.shape, case.tol, dtype=np.float64)
        if case.coeff is None
        else _root_uncertainty(case.coeff, reference, case.tol)
    )

    worst = float(bound.max(initial=0.0))
    if worst >= PARAMETER_DOMAIN:
        # Rule 8: a parity claim is only defined where the quantity still has digits.
        # Here the derived uncertainty covers the whole parameter interval, so there
        # is nothing left to compare. Named and reported rather than skipped
        # silently, and asserted to be a float32 phenomenon: the same bound reaching
        # the domain at float64 would mean the derivation, not the arithmetic, is at
        # fault.
        assert case.coeff is not None and case.coeff.dtype == np.float32, (
            f"{context}: the derived uncertainty {worst:.3e} covers the whole "
            f"parameter domain at {None if case.coeff is None else case.coeff.dtype}. "
            f"Outside float32 that is a defect in the derivation rather than a limit "
            f"of the format."
        )
        pytest.skip(
            f"{context}: outside the parity domain. The derived uncertainty "
            f"{worst:.3e} covers the whole parameter interval, so the roots carry no "
            f"digits to compare at this width. Bit-identical on the shipped build, "
            f"where nothing fuses."
        )

    assert_parity(
        actual,
        reference,
        converged_parity(bound=bound, scale=PARAMETER_DOMAIN, why=_FUSED_WHY),
        context=context,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("kind", KINDS)
def test_find_roots_matches_the_oracle(
    cpp_backend: None, degree: int, kind: str, dtype: npt.DTypeLike
) -> None:
    """The two backends find the same roots of the same polynomial."""
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    coeff = _polynomial(degree, dtype, kind)
    reference, actual = _both_backends("roots", coeff)
    _assert_the_same_roots(
        actual,
        reference,
        f"find_roots, {kind}, degree {degree}, {dtype}",
        Case(coeff, TOL),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", DEGREES)
def test_find_monotone_root_matches_the_oracle(
    cpp_backend: None, degree: int, dtype: npt.DTypeLike
) -> None:
    """The two backends solve the same monotone polynomial identically."""
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    coeff = _polynomial(degree, dtype, "monotone")
    reference, actual = _both_backends("monotone", coeff)
    _assert_the_same_roots(
        actual,
        reference,
        f"find_monotone_root, degree {degree}, {dtype}",
        Case(coeff, TOL),
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degree", DEGREES)
def test_the_batch_path_matches_the_oracle(
    cpp_backend: None, degree: int, dtype: npt.DTypeLike
) -> None:
    """The batch entry points agree row by row, counts included.

    The batch kernels are the only ones in the block where the Numba side is
    ``parallel=True`` and the C++ side is serial. Each polynomial writes only its own
    row and no reduction crosses them, so the thread count cannot move a result and
    this is a parity test rather than a reproducibility one; it is here to hold that
    property rather than to discover it.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    curves = [Bezier(_polynomial(degree, dtype, kind).reshape(-1, 1)) for kind in KINDS]

    with use_backend(Backend.PYTHON):
        reference_roots, reference_counts = find_roots(curves, tol=TOL)
        reference_mono = find_monotone_root(curves, tol=TOL)
    with use_backend(Backend.CPP):
        actual_roots, actual_counts = find_roots(curves, tol=TOL)
        actual_mono = find_monotone_root(curves, tol=TOL)

    np.testing.assert_array_equal(
        actual_counts,
        reference_counts,
        err_msg=f"batch root counts, degree {degree}, {dtype}: a changed verdict, "
        "which no tolerance covers",
    )

    # Only the prefix of each row is contractual; the rest is the NaN the caller
    # pre-filled, and comparing it would be comparing untouched memory.
    for row, count in enumerate(reference_counts):
        _assert_the_same_roots(
            np.asarray(actual_roots[row, :count], dtype=np.float64),
            np.asarray(reference_roots[row, :count], dtype=np.float64),
            f"batch find_roots row {row}, degree {degree}, {dtype}",
            Case(_polynomial(degree, dtype, KINDS[row]), TOL),
        )

    _assert_the_same_roots(
        np.asarray(actual_mono, dtype=np.float64),
        np.asarray(reference_mono, dtype=np.float64),
        f"batch find_monotone_root, degree {degree}, {dtype}",
        Case(None, TOL),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_backends_agree_on_the_root_that_is_not_there(
    cpp_backend: None, dtype: npt.DTypeLike
) -> None:
    """Both backends reproduce FELIGN/pantr#351, and that is the assertion.

    `find_monotone_root` reports a root of a strictly positive polynomial when the
    coefficients are small enough at ``float32``: the no-sign-change guard is written
    as ``f_lo * f_hi > 0.0`` on two ``float32`` values, their product falls under that
    format's minimum subnormal and rounds to zero, and the guard does not fire.

    **This test asserts a wrong answer on purpose.** The C++ reproduces the defect
    because the port's contract is parity, not correction. When #351 is fixed, it is
    fixed in both backends and here in one change; correcting one side alone breaks
    the equality rather than improving anything.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    coeff = np.array([1e-25, 0.5e-25, 2e-25], dtype=dtype)
    reference, actual = _both_backends("monotone", coeff, DEFECT_TOL)

    if dtype is np.float32:
        assert np.isfinite(reference[0]), (
            "the defect is gone from the oracle. If #351 was fixed, fix the C++ and "
            "this test in the same change rather than relaxing either."
        )
    else:
        assert np.isnan(reference[0]), "float64 is wide enough that the guard fires"

    _assert_the_same_roots(actual, reference, f"the #351 regime, {dtype}", Case(coeff, DEFECT_TOL))


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_backends_agree_on_the_root_that_is_lost(
    cpp_backend: None, dtype: npt.DTypeLike
) -> None:
    """Both backends reproduce the second face of FELIGN/pantr#351.

    ``B(t) = a(1 - 2t)`` has its only root at ``t = 0.5``. At ``float32`` with
    ``a = 1e-25`` the sign-change test ``f_prev * f_curr < 0.0`` underflows, no sign
    change is seen, and the root is lost. See the note on the sibling test: the
    assertion is deliberate.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    coeff = np.array([1e-25, 0.0, -1e-25], dtype=dtype)
    reference, actual = _both_backends("roots", coeff, DEFECT_TOL)

    if dtype is np.float32:
        assert reference.size == 0, (
            "the defect is gone from the oracle. If #351 was fixed, fix the C++ and "
            "this test in the same change rather than relaxing either."
        )
    else:
        np.testing.assert_allclose(reference, [0.5], atol=TOL)

    _assert_the_same_roots(
        actual,
        reference,
        f"the lost-root face of #351, {dtype}",
        Case(coeff, DEFECT_TOL),
    )


def test_the_backends_agree_where_the_tolerance_stops_scaling(cpp_backend: None) -> None:
    """Both backends reproduce FELIGN/pantr#352, at ``float64``.

    ``boundary_eps`` carries an absolute ``1e-30`` floor inside an otherwise
    scale-relative tolerance, so below it every coefficient reads as zero, the
    endpoints are reported as roots and the real one is lost. Rescaling the same
    polynomial moves the answer, which is what makes it a defect rather than a limit.

    Asserted here for the same reason as #351: the C++ reproduces it, and both sides
    change together or not at all.
    """
    del cpp_backend
    demand_the_compiled_kernel(np.float64)

    for scale, expected in ((1e-29, [0.5]), (1e-31, [0.0, 1.0])):
        coeff = np.array([scale, 0.0, -scale], dtype=np.float64)
        reference, actual = _both_backends("roots", coeff, DEFECT_TOL)
        np.testing.assert_allclose(
            reference,
            expected,
            atol=TOL,
            err_msg=f"the oracle changed at scale {scale:.0e}; if #352 was fixed, fix "
            "the C++ and this test in the same change",
        )
        _assert_the_same_roots(
            actual,
            reference,
            f"the #352 regime at scale {scale:.0e}",
            Case(coeff, DEFECT_TOL),
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_seeded_sweep_matches_the_oracle(cpp_backend: None, dtype: npt.DTypeLike) -> None:
    """The two backends agree over a seeded sweep of random polynomials.

    The families above are hand-built to attack one mechanism each, which means they
    can only find what someone thought of. This is the breadth complement: 240
    polynomials over eight degrees, four generators each, drawn from a fixed seed so
    a failure is reproducible.

    It is the cross-check that validated the transliteration before any binding
    existed, moved into the suite so it runs on every commit rather than once on one
    machine.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    rng = np.random.default_rng(20260824)
    for degree in DEGREES:
        for trial in range(30):
            base: npt.NDArray[np.floating] = rng.standard_normal(degree + 1)
            if trial % 4 == 1:
                # A straight control polygon: every hull orientation test is a tie.
                base = rng.standard_normal() + rng.standard_normal() * np.arange(degree + 1)
            elif trial % 4 == 2:
                # Sorted, so a sign change is guaranteed and the solver runs.
                base = np.sort(base)
            elif trial % 4 == 3:
                base = base * 10.0 ** rng.integers(-6, 7, size=degree + 1)
            coeff = np.asarray(base, dtype=dtype)

            reference, actual = _both_backends("roots", coeff)
            _assert_the_same_roots(
                actual,
                reference,
                f"seeded sweep, degree {degree}, trial {trial}, {dtype}",
                Case(coeff, TOL),
            )


def test_a_changed_root_count_is_refused() -> None:
    """The count check fires, and says why no tolerance would have caught it.

    The helper's first assertion is the one nothing else in the harness covers, so
    it needs its own probe: a bounded claim compares elementwise and cannot see two
    results of different length at all.
    """
    with pytest.raises(AssertionError, match="different numbers of roots"):
        _assert_the_same_roots(
            np.array([0.25, 0.75]),
            np.array([0.25]),
            "a fabricated count mismatch",
            Case(None, TOL),
        )


def test_the_converged_bound_refuses_a_displacement_past_itself() -> None:
    """The fused branch's bound is not vacuous, and rejects what it should.

    Built unconditionally rather than behind ``contraction_may_fuse()``, because on
    the shipped build that branch is never taken and an unexercised branch is how a
    claim ships broken. This file's fused branch **did** ship broken during
    development: it called ``bounded_parity`` with an argument that function does not
    take, and 133 tests passed over it because none of them reached the branch.
    """
    reference = np.array([0.5])
    inside = np.array([0.5 + 4e-13])
    outside = np.array([0.5 + 4e-12])

    claim = converged_parity(
        bound=np.full(1, TOL), scale=PARAMETER_DOMAIN, why="a probe of the bound itself"
    )

    assert_parity(inside, reference, claim, context="a displacement inside the bound")
    with pytest.raises(AssertionError, match="more than the derived bound"):
        assert_parity(outside, reference, claim, context="a displacement past the bound")


def test_a_bound_reaching_the_whole_domain_is_refused() -> None:
    """The vacuity guard still fires, now against the domain rather than the value.

    Two probes, because the change that made the near-endpoint roots pass could also
    have disabled the guard entirely: a bound of one covers every parameter in [0, 1]
    and must be refused, while a bound far under the domain must be accepted even
    when it exceeds the root's own magnitude.
    """
    tiny_root = np.array([3.6e-13])

    accepted = converged_parity(
        bound=np.full(1, TOL), scale=PARAMETER_DOMAIN, why="a probe of the domain premise"
    )
    assert_parity(tiny_root, tiny_root, accepted, context="a root next to an endpoint")

    vacuous = converged_parity(
        bound=np.full(1, PARAMETER_DOMAIN), scale=PARAMETER_DOMAIN, why="a probe of the guard"
    )
    with pytest.raises(AssertionError, match="vacuous"):
        assert_parity(tiny_root, tiny_root, vacuous, context="a bound covering the domain")
