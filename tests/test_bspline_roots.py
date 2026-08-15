"""Tests for Mørken-Reimers B-spline root finding (``pantr.bspline.find_roots``)."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bezier import find_roots as bezier_find_roots
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D, _bspline_roots, find_roots
from pantr.bspline._bspline_roots import _max_insertions
from pantr.bspline._bspline_roots_core import (
    _insert_knot,
    _is_zero_index,
    _knot_average,
    _merge_roots,
    _morken_reimers_roots,
    _residual_at,
    _span_at,
    _split_at_root,
    _zero_index,
)
from pantr.tolerance import get_strict

_EPS: float = float(np.finfo(np.float64).eps)
"""Machine epsilon of ``float64``, the precision every tolerance below derives from."""

_STRICT_TOL: float = get_strict(np.float64)
"""
The default relative parametric tolerance of :func:`find_roots`, ``1e-15``, or 4.5 ulp.

It is what sets the accuracy of a reported root, so it is what the residual bound scales
with; machine epsilon alone understates that by the factor between the two.
"""

_ROOT_ULPS: float = 64.0
"""
Accuracy of a simple root, in ulp of the coordinate scale.

A root is a parametric coordinate, so its accuracy floor is one ulp of its own
magnitude; the iteration adds the evaluation error of the control polygon, which is
``degree + 1`` convex combinations. The worst error measured over the analytic cases
below is ``2.3e-13`` at degree 5 on a domain around ``1e3``, that is two ulp of the
coordinate, so 64 ulp leaves a factor of 32.
"""

_MULTIPLE_ROOT_SAFETY: float = 2.0
"""
Safety factor on the half-width of the interval a multiple root cannot be located inside.

That half-width is the physical limit, not an estimate, so asserting against it bare leaves
no margin at all and the assertion turns on where the iteration happens to stop. It stops
somewhere different on every build: the same double root lands at ``2.3e-8`` under one numpy
and at ``3.1e-8`` under another, a 40 % spread with no bug behind it. Two absorbs that, and
still fails on anything an order of magnitude out. Measured margins with it: 16 times at
multiplicities two and three, 9.5 at four, 3.3 for a double root beside a simple one.
"""

_PARITY_RTOL: float = 1e-10
"""
Agreement required between this method and extraction plus Bernstein root finding.

The assertion is that the two routes report the *same set* of zeros, not that either is
accurate: the Bézier route maps each segment root back with ``lo + t * (hi - lo)`` and so
carries the segment's own rounding, which is not the object under test. Accuracy is
asserted separately, against analytic polynomials and against the residual bound. The
worst disagreement measured over 700 random splines is ``5.6e-16`` on a unit domain, so
this leaves five decades of margin.
"""


def _open_knots(
    degree: int,
    breaks: npt.ArrayLike,
    multiplicities: npt.NDArray[np.int_] | None = None,
) -> npt.NDArray[np.float64]:
    """Build an open (clamped) knot vector from breakpoints and interior multiplicities."""
    inner = np.asarray(breaks, dtype=np.float64)
    interior = inner[1:-1] if multiplicities is None else np.repeat(inner[1:-1], multiplicities)
    return np.concatenate(([inner[0]] * (degree + 1), interior, [inner[-1]] * (degree + 1)))


def _spline(
    knots: npt.NDArray[np.float64],
    coeffs: npt.ArrayLike,
    degree: int,
    dtype: npt.DTypeLike = np.float64,
) -> Bspline:
    """Build the scalar univariate B-spline with the given knots and coefficients."""
    space = BsplineSpace([BsplineSpace1D(knots.astype(dtype), degree, snap_knots=False)])
    return Bspline(space, np.asarray(coeffs, dtype=dtype).reshape(-1, 1))


def _polynomial_coefficients(
    knots: npt.NDArray[np.float64],
    degree: int,
    roots: tuple[float, ...],
) -> npt.NDArray[np.float64]:
    """Return the B-spline coefficients of ``prod(x - r)`` through its blossom.

    The blossom of ``x ** k`` at ``u_1, ..., u_p`` is ``e_k(u) / C(p, k)``, with ``e_k``
    the elementary symmetric polynomial, so the coefficient ``c_j``, which is the blossom
    evaluated on the knot window ``t[j+1 .. j+p]``, follows from the power-basis
    coefficients of the polynomial. This is an oracle independent of any root finder: the
    roots are put in, the coefficients come out.

    Build the polynomial on a domain around the origin and move it by scaling the *knots*
    afterwards. Building it directly on a domain around ``1e3`` instead loses it entirely,
    the power basis then cancelling terms of order ``1e15`` down to a result of order
    ``1e-5``.
    """
    power = np.poly(np.asarray(roots, dtype=np.float64))[::-1]
    binomials = np.array([math.comb(degree, k) for k in range(degree + 1)], dtype=np.float64)
    out = np.empty(len(knots) - degree - 1, dtype=np.float64)
    for j in range(out.shape[0]):
        window = np.asarray(knots[j + 1 : j + degree + 1], dtype=np.float64)
        # np.poly(window)[i] is the coefficient of x ** (p - i) in prod(x - u), which is
        # (-1) ** i * e_i, so the sign alternation recovers e_i itself.
        signs = np.array([(-1.0) ** i for i in range(degree + 1)])
        out[j] = float(np.dot(power, np.poly(window) * signs / binomials))
    return out


def _bezier_reference_roots(
    spline: Bspline,
    knots: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the zeros found by extracting Bézier segments and solving each of them.

    An independent route to the same answer: it converts to the Bernstein basis and runs
    the clipping solver of :func:`pantr.bezier.find_roots` on every segment, then stitches
    and deduplicates. Nothing of the knot-insertion method is reused.
    """
    unique = np.unique(knots)
    found: list[float] = []
    for segment, bezier in enumerate(spline.to_beziers().ravel()):
        low, high = unique[segment], unique[segment + 1]
        found.extend(low + bezier_find_roots(bezier) * (high - low))
    out = np.sort(np.asarray(found, dtype=np.float64))
    if out.size:
        length = unique[-1] - unique[0]
        out = out[np.concatenate(([True], np.diff(out) > 1e-9 * length))]
    return out


def _hodograph_slope_bound(
    coeffs: npt.NDArray[np.float64],
    knots: npt.NDArray[np.float64],
    degree: int,
) -> float:
    """Bound ``|f'|`` over the whole domain from the hodograph coefficients.

    The derivative of a spline is the spline whose coefficients are
    ``degree * (c[i] - c[i-1]) / (t[i+degree] - t[i])``, and a spline never leaves the
    range of its own coefficients, so the largest such difference quotient bounds ``|f'|``
    everywhere. A zero-length denominator is a C^-1 knot, where the spline jumps and no
    finite slope describes it, and is skipped.
    """
    gaps = knots[degree + 1 : coeffs.shape[0] + degree] - knots[1 : coeffs.shape[0]]
    quotients = np.abs(np.diff(coeffs))[gaps > 0.0] / gaps[gaps > 0.0]
    return float(degree * quotients.max()) if quotients.size else 0.0


def _zero_tol(coeffs: npt.NDArray[np.float64], degree: int) -> float:
    """Return the residual threshold ``find_roots`` derives for these coefficients."""
    return float(np.abs(coeffs).max()) * max((degree + 1) * 4.0 * _EPS, 1e-15)


def _root_tol(
    coeffs: npt.NDArray[np.float64],
    knots: npt.NDArray[np.float64],
    degree: int,
) -> float:
    """Return the residual threshold ``find_roots`` allows a *tracked* iterate.

    Larger than :func:`_zero_tol` by the value the spline reaches at the parametric
    resolution the iteration stops at, ``|f'| * 2 * tol * scale``.
    """
    num_coeffs = coeffs.shape[0]
    scale = max(abs(float(knots[degree])), abs(float(knots[num_coeffs])))
    scale = max(scale, float(knots[num_coeffs] - knots[degree]))
    slope = _hodograph_slope_bound(coeffs, knots, degree)
    return _zero_tol(coeffs, degree) + 2.0 * slope * _STRICT_TOL * scale


# --- Analytic cases -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("degree", "roots"),
    [
        (1, (0.5,)),
        (2, (0.25, 0.8)),
        (3, (0.2, 0.5, 0.9)),
        (4, (0.15, 0.4, 0.6, 0.85)),
        (5, (0.1, 0.3, 0.55, 0.8, 0.95)),
    ],
)
def test_polynomial_roots_are_found_to_a_few_ulp(degree: int, roots: tuple[float, ...]) -> None:
    """A polynomial in B-spline form gives back exactly the roots it was built from."""
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 6))
    coeffs = _polynomial_coefficients(knots, degree, roots)

    found = find_roots(_spline(knots, coeffs, degree))

    assert found.shape == (len(roots),)
    np.testing.assert_allclose(found, np.array(roots), rtol=0.0, atol=_ROOT_ULPS * _EPS)


@pytest.mark.parametrize(
    ("origin", "span"),
    [(0.0, 1.0), (1.0e3, 1.0), (1.0e6, 1.0), (-5.0, 7.0), (0.0, 1.0e-3)],
)
def test_roots_follow_an_affine_reparametrization(origin: float, span: float) -> None:
    """Moving and scaling the knots moves and scales the roots, to the coordinate's ulp.

    A B-spline whose knots are mapped affinely, with the coefficients left alone, *is* the
    affinely reparametrized function, so the exact answer is known at every scale. This is
    what shows the tolerances are scale-covariant rather than tuned to the unit interval.
    """
    degree = 3
    unit_knots = _open_knots(degree, np.linspace(0.0, 1.0, 6))
    unit_roots = (0.2, 0.5, 0.9)
    coeffs = _polynomial_coefficients(unit_knots, degree, unit_roots)

    found = find_roots(_spline(origin + span * unit_knots, coeffs, degree))

    expected = np.array([origin + span * r for r in unit_roots])
    scale = max(abs(origin), span)
    np.testing.assert_allclose(found, expected, rtol=0.0, atol=_ROOT_ULPS * _EPS * scale)


@pytest.mark.parametrize("multiplicity", [2, 3, 4])
def test_a_multiple_root_is_reported_once(multiplicity: int) -> None:
    """A zero of multiplicity ``m`` collapses to one root, accurate to ``zero_tol ** (1/m)``.

    That exponent is not a safety factor but the accuracy the problem allows: around a zero
    of multiplicity ``m`` the spline only reaches ``zero_tol`` at a distance
    ``(m! * zero_tol / |f^(m)|) ** (1 / m)``, and for ``(x - 1/2) ** m`` the ``m``-th
    derivative is exactly ``m!``. A zero of even multiplicity has no sign change at all, so
    it is reported through the residual test rather than through the control polygon.
    """
    knots = _open_knots(multiplicity, np.linspace(0.0, 1.0, 6))
    coeffs = _polynomial_coefficients(knots, multiplicity, (0.5,) * multiplicity)

    found = find_roots(_spline(knots, coeffs, multiplicity))

    half_width = _zero_tol(coeffs, multiplicity) ** (1.0 / multiplicity)
    assert found.shape == (1,)
    assert abs(found[0] - 0.5) <= _MULTIPLE_ROOT_SAFETY * half_width


def test_a_simple_and_a_double_root_stay_separate() -> None:
    """Merging collapses a repeated report of one zero, never two distinct zeros.

    The double root's accuracy comes from its own curvature, not from a generic one: for
    ``(x - 1/4) (x - 1/2) ** 2`` the second derivative at ``1/2`` is ``2 * (1/2 - 1/4)``,
    so the interval where the spline is indistinguishable from zero has half-width
    ``sqrt(2 * zero_tol / 0.5)``, four times wider than for ``(x - 1/2) ** 2``.
    """
    degree = 3
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 6))
    coeffs = _polynomial_coefficients(knots, degree, (0.25, 0.5, 0.5))

    found = find_roots(_spline(knots, coeffs, degree))

    second_derivative = 2.0 * (0.5 - 0.25)
    half_width = math.sqrt(2.0 * _zero_tol(coeffs, degree) / second_derivative)
    assert found.shape == (2,)
    assert abs(found[0] - 0.25) <= _ROOT_ULPS * _EPS
    assert abs(found[1] - 0.5) <= _MULTIPLE_ROOT_SAFETY * half_width


def test_a_root_sitting_exactly_on_an_interior_knot() -> None:
    """A zero at a knot needs no special case: it is where the method inserts anyway."""
    degree = 3
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 6))
    coeffs = _polynomial_coefficients(knots, degree, (0.2, 0.4, 0.8))

    found = find_roots(_spline(knots, coeffs, degree))

    np.testing.assert_allclose(found, [0.2, 0.4, 0.8], rtol=0.0, atol=_ROOT_ULPS * _EPS)


def test_roots_at_both_domain_endpoints() -> None:
    """An open knot vector interpolates its end coefficients, so an endpoint zero shows up."""
    degree = 2
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 5))
    coeffs = _polynomial_coefficients(knots, degree, (0.0, 1.0))

    found = find_roots(_spline(knots, coeffs, degree))

    np.testing.assert_allclose(found, [0.0, 1.0], rtol=0.0, atol=_ROOT_ULPS * _EPS)


def test_no_roots_when_the_spline_is_strictly_positive() -> None:
    """A control polygon without a sign change means no zeros at all."""
    degree = 3
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 6))
    coefficients = np.full(len(knots) - degree - 1, 2.0)

    found = find_roots(_spline(knots, coefficients, degree))

    assert found.shape == (0,)


# --- Agreement with an independent route ----------------------------------------------


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5, 6, 7])
def test_matches_the_bezier_route_on_random_splines(degree: int) -> None:
    """Random splines with repeated interior knots give the same zero set as extraction.

    Repeated interior knots are where the two defects this module was fixed for both live:
    with simple knots the two routes agreed on every one of 160 trials, while the failures
    appeared at about one per cent once multiplicities were drawn as well.
    """
    rng = np.random.default_rng(20260814 + degree)
    for _ in range(50):
        n_elements = int(rng.integers(2, 20))
        breaks = np.unique(np.concatenate(([0.0, 1.0], rng.uniform(0.0, 1.0, n_elements - 1))))
        multiplicities = rng.integers(1, degree + 1, size=max(breaks.size - 2, 0))
        knots = _open_knots(degree, breaks, multiplicities)
        coeffs = rng.normal(size=len(knots) - degree - 1)
        spline = _spline(knots, coeffs, degree)

        found = find_roots(spline)
        expected = _bezier_reference_roots(spline, knots)

        assert found.shape == expected.shape
        if found.size:
            np.testing.assert_allclose(found, expected, rtol=0.0, atol=_PARITY_RTOL)


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5, 6, 7])
def test_sign_alternating_splines_agree_with_the_bezier_route(degree: int) -> None:
    """Coefficients alternating ``+1, -1`` give the same zero set as extraction.

    The family the August 2026 sweep found the certificate defect with, and the one the
    residual test has to be careful not to over-reject: one sign change per interval, so
    every zero the polygon can bracket is present, and a slope steep enough that the
    evaluation error alone is not what a correctly located zero leaves behind.

    Two intervals are excluded. There the outer zeros are near-tangential and both routes
    place them only to ``zero_tol ** (1 / m)``, which at degree 7 is ``1.8e-3``: a real
    accuracy limit of the problem, asserted by ``test_a_multiple_root_is_reported_once``
    on data built for it, and not what this test is about.
    """
    for n_intervals in (3, 4, 5, 8):
        knots = _open_knots(degree, np.linspace(0.0, 1.0, n_intervals + 1))
        coeffs = np.where(np.arange(len(knots) - degree - 1) % 2, -1.0, 1.0)
        spline = _spline(knots, coeffs, degree)

        found = find_roots(spline)
        expected = _bezier_reference_roots(spline, knots)

        assert found.shape == expected.shape, f"{n_intervals} intervals"
        np.testing.assert_allclose(found, expected, rtol=0.0, atol=_PARITY_RTOL)


@pytest.mark.parametrize(
    ("origin", "span"),
    [(0.0, 1.0), (1.0e3, 1.0), (1.0e6, 1.0), (0.0, 1.0e-3)],
)
def test_residual_at_every_root_is_within_the_evaluation_bound(origin: float, span: float) -> None:
    """``|f(x)|`` at a reported root stays under the error of evaluating ``f`` there.

    The bound has two terms and needs both. The first is the de Boor evaluation error,
    ``coeff_scale * (degree + 1) * 4 * eps``. The second is ``|f'| * 2 * tol * max(|x|, L)``,
    the value the spline takes at the parametric resolution the iteration actually stops at:
    the rule is a spread of the last ``degree`` iterates below ``tol * max(|x|, L)``, so the
    root carries that much slack and one further step. Writing ``eps`` there instead of
    ``tol`` understates it by the factor between them, which is 4.5 for the default
    ``get_strict(float64)``; that goes unnoticed on the unit domain, where the first term
    dominates, and fails by 6.8 times on a domain around ``1e6``, where the second does.
    """
    rng = np.random.default_rng(613)
    worst = 0.0
    for degree in (2, 3, 5, 7):
        for _ in range(15):
            n_elements = int(rng.integers(2, 15))
            breaks = np.unique(np.concatenate(([0.0, 1.0], rng.uniform(0.0, 1.0, n_elements - 1))))
            knots = origin + span * _open_knots(degree, breaks)
            coeffs = rng.normal(size=len(knots) - degree - 1)
            spline = _spline(knots, coeffs, degree)

            roots = find_roots(spline)
            if roots.size == 0:
                continue
            values = np.abs(np.asarray(spline.evaluate(roots), dtype=np.float64).reshape(-1))
            slopes = np.abs(
                np.asarray(spline.evaluate_derivatives(roots, [1]), dtype=np.float64).reshape(-1)
            )
            bound = _zero_tol(coeffs, degree) + slopes * 2.0 * _STRICT_TOL * np.maximum(
                np.abs(roots), span
            )
            worst = max(worst, float((values / bound).max()))
    assert worst <= 1.0


# --- Regression cases, with the data that triggered them -------------------------------

_DIVISION_DEGREE = 7
"""Degree of the spline that used to make the Boehm insertion divide by zero."""

_DIVISION_BREAKS = np.array(
    [
        0.024892404472518503,
        0.03354892843448798,
        0.10136056366541846,
        0.1122190513796808,
        0.1464272974472447,
        0.4624100974100952,
        0.5109112536781879,
        0.6235078061995366,
        0.6378198461841913,
        0.729584735215864,
        0.7697545067745368,
        0.7715103674467213,
        0.8530564246522064,
        0.9409084808000586,
        0.9485109253452616,
    ]
)
"""Interior breakpoints of the division-by-zero case."""

_DIVISION_MULTS = np.array([4, 6, 1, 5, 6, 2, 6, 7, 1, 2, 6, 3, 7, 6, 3])
"""Interior knot multiplicities of the division-by-zero case."""

_DIVISION_COEFFS = np.array(
    [
        0.9437459682550059,
        -1.7619232704479926,
        -1.769325337044876,
        -0.18024728004294954,
        0.041247632747591106,
        -0.3994308994247353,
        -0.516022669178932,
        0.18100580447818726,
        0.4993511564441196,
        0.5321707345411638,
        0.4660851995749392,
        0.7410879736380998,
        -0.38089957480056474,
        1.3631888996881332,
        -0.5494494669227896,
        -1.0452973037922395,
        0.3203141999627478,
        0.1757283130970403,
        -0.5007916818951128,
        0.11060288310608378,
        -1.5937131347903954,
        -0.8313734056944073,
        0.24133882168833876,
        0.7983299714590487,
        0.8765505820977094,
        -0.12259583863913351,
        -0.009193873407048046,
        1.5711360452267944,
        -1.9321387632151084,
        -1.3190257696414207,
        -0.2844767117704297,
        -1.1990518322779973,
        0.51671830294839,
        0.2624975354984742,
        -0.9803343820777106,
        1.5654425190301766,
        0.015287705067431509,
        0.05394948182127942,
        1.3959184688042012,
        -0.7235266585844332,
        -0.8978583025676149,
        2.828565592337862,
        -1.886413481161058,
        0.2974819653205911,
        -0.08995063875416641,
        -0.9172722203772358,
        -0.9587210338847852,
        -0.5818281363389338,
        1.5514175764178706,
        -1.195103342222694,
        -0.14838726266626737,
        -0.384981200717065,
        1.6790678865844157,
        0.6536032708161893,
        0.09219390150258462,
        -0.6454490385078762,
        1.0916575980237546,
        1.0695505549977262,
        1.130009070519936,
        -0.28799879294347763,
        -0.7013417360350601,
        -0.9074097956738512,
        -0.1007157586601924,
        -1.2357574762970038,
        0.9824577648202112,
        2.1521157661220105,
        -0.26550142696010753,
        -1.36185989740396,
        0.3919810001338171,
        -0.481357556584237,
        -0.0985054408304594,
        0.09376462814118997,
        1.1837644094770001,
    ]
)
"""Coefficients of the division-by-zero case."""

_MISSED_DEGREE = 3
"""Degree of the spline that used to lose a root."""

_MISSED_BREAKS = np.array(
    [
        0.05686473281359683,
        0.08537928894568714,
        0.13343296683579242,
        0.2689055644096665,
        0.389738489622555,
        0.4083831147882375,
        0.5487495352562979,
        0.5596814072327488,
        0.5877034964862079,
        0.5948584020096005,
        0.6785788110950385,
        0.7229176228671199,
        0.7299050172882676,
        0.7351085418280764,
        0.898802239121968,
        0.93445666115023,
        0.9393359179619315,
    ]
)
"""Interior breakpoints of the lost-root case."""

_MISSED_MULTS = np.array([1, 1, 3, 2, 1, 1, 1, 2, 3, 1, 3, 3, 1, 3, 2, 1, 3])
"""Interior knot multiplicities of the lost-root case."""

_MISSED_COEFFS = np.array(
    [
        -1.0006117751615216,
        0.10262535130347963,
        -0.48647873649939866,
        0.46005458592952936,
        0.4281128136333478,
        -0.8192377984112155,
        0.13140957712401866,
        0.18713602388592093,
        -0.12555552301405867,
        0.2942570898257234,
        1.3492740850774352,
        0.5886224136013747,
        -0.2619988438521604,
        0.23047441055342363,
        1.9599383087338076,
        -0.6097700812684572,
        1.1006657532543598,
        -0.6559691195441099,
        -1.5714860195424816,
        -0.6787185365612848,
        0.332880561443392,
        -0.23357475164202018,
        -0.6903722447753305,
        -0.7720728746968911,
        -0.9478203174394234,
        -1.0478317575078717,
        -0.6009049657319299,
        -2.4532962394326048,
        0.4157017527144879,
        0.2762701458252833,
        0.08991282869754379,
        0.8858012047756563,
        -0.5933233317599033,
        -0.01809771231938426,
        -2.566961781300906,
        0.1730469906173836,
    ]
)
"""Coefficients of the lost-root case."""

_MISSED_ROOT = 0.611347358111781
"""The zero the lost-root case used to drop, located by the Bézier route."""

_JUMP_KNOTS = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
"""Knot vector of the C^-1 case: ``BsplineSpace1D(_open_knots(2, [0, 1])).subdivide(2, -1)``.

The interior knot carries multiplicity ``degree + 1``, which is the only multiplicity at
which the two Greville abscissae ``t[index-1]`` and ``t[index]`` coincide.
"""

_JUMP_COEFFS = np.array([-1.0, -0.5, 0.3, -0.4, 0.5, 1.0])
"""Coefficients of the C^-1 case: the pair straddling the break changes sign, 0.3 to -0.4."""

_JUMP_ROOTS = np.array(
    [0.5 * (math.sqrt(2.2) - 1.0) / 0.6, 0.5 + 0.5 * (9.0 - math.sqrt(65.0)) / 4.0]
)
"""The two zeros of the C^-1 case in closed form, 0.40269975 and 0.61721778.

Each Bézier piece is a quadratic that can be solved by hand, which is what makes this an
independent oracle rather than a second run of the method. Left piece, in ``u = 2t``:
``0.3u^2 + u - 1``, so ``u = (sqrt(2.2) - 1) / 0.6``. Right piece, in ``v = 2t - 1``:
``-0.4v^2 + 1.8v - 0.4``, so ``v = (9 - sqrt(65)) / 4``. Both are the root inside ``[0, 1]``.
"""


def test_regression_repeated_knots_do_not_divide_by_zero() -> None:
    """Knots piled to within an ulp used to send a span search off the front of the buffer.

    Splitting at a reported zero raised it to multiplicity ``degree`` and dropped everything
    strictly left of the run, which left one knot of the old spline hanging below the new
    domain: the remainder was no longer an open knot vector. A Greville abscissa is an
    average of ``degree`` knots and can round below its own smallest term once they have
    piled up, and the span search, which walks right from the tracked coefficient index,
    then returned a span smaller than ``degree``. Boehm's loop indexed backwards past zero
    from there, read uninitialized memory, and divided by a difference of two equal knots.
    """
    knots = _open_knots(
        _DIVISION_DEGREE,
        np.concatenate(([0.0], _DIVISION_BREAKS, [1.0])),
        _DIVISION_MULTS,
    )
    spline = _spline(knots, _DIVISION_COEFFS, _DIVISION_DEGREE)

    found = find_roots(spline)

    expected = _bezier_reference_roots(spline, knots)
    assert found.shape == expected.shape
    np.testing.assert_allclose(found, expected, rtol=0.0, atol=_PARITY_RTOL)


def test_regression_no_root_is_lost_after_a_rejected_sign_change() -> None:
    """A sign change rejected as already reported used to take its right neighbour with it.

    Splitting leaves the coefficient at the zero pinned, but the next one along is only zero
    to within the rounding of a hundred knot insertions, which is coarser than the residual
    threshold; so a spurious sign change survives beside the barrier and tracks straight back
    to the zero just reported. Rejecting it used to advance the scan past the whole knot run,
    and the genuine sign change bracketing the next zero sits inside that run.
    """
    knots = _open_knots(
        _MISSED_DEGREE, np.concatenate(([0.0], _MISSED_BREAKS, [1.0])), _MISSED_MULTS
    )
    spline = _spline(knots, _MISSED_COEFFS, _MISSED_DEGREE)

    found = find_roots(spline)

    assert float(np.abs(found - _MISSED_ROOT).min()) <= _PARITY_RTOL
    expected = _bezier_reference_roots(spline, knots)
    assert found.shape == expected.shape
    np.testing.assert_allclose(found, expected, rtol=0.0, atol=_PARITY_RTOL)


def test_regression_a_jump_across_the_axis_is_not_a_root() -> None:
    """A C^-1 knot used to be reported as a zero, and to swallow the next genuine one.

    The tracking stopped and declared ``f(x) = 0`` whenever an iterate reached the right end
    of its Greville interval, citing Mørken-Reimers Lemma 3. The lemma places its ``x`` in
    the half-open interval ``(t[index-1], t[index]]``, which is empty when the two abscissae
    coincide, and they coincide exactly when the knot run at ``x`` has multiplicity
    ``degree + 1``. There the secant through the two coefficients is vertical, its zero is
    ``x`` for every ``lambda``, and nothing forces the coefficient to vanish.

    So on the spline below the break at 0.5 was reported as a root although the spline jumps
    from ``+0.3`` to ``-0.4`` there. Reporting it then split the spline at 0.5 and pinned the
    coefficient ``-0.4`` to zero as the split barrier, which destroyed the sign change
    bracketing the genuine zero at 0.61721778 and lost it. One fabricated root and one lost
    root, from one cause, which is why both halves are asserted here.
    """
    space = BsplineSpace1D(_JUMP_KNOTS, 2)
    spline = _spline(_JUMP_KNOTS, _JUMP_COEFFS, 2)

    # Precondition: the interior knot is C^-1, the only case the lemma does not cover.
    unique, mults = space.get_unique_knots_and_multiplicity(in_domain=True)
    assert int(mults[np.searchsorted(unique, 0.5)]) == space.degree + 1

    # Precondition: the spline really does jump across the axis at the break.
    jump = spline.evaluate(np.array([[0.5 - 1e-9], [0.5 + 1e-9]])).reshape(-1)
    assert jump[0] > 0.0 > jump[1]

    found = find_roots(spline)

    np.testing.assert_allclose(found, _JUMP_ROOTS, rtol=0.0, atol=_ROOT_ULPS * _EPS)

    # The Bézier route as well, as the two regression tests above do: it solves each
    # segment on its own, so a C^-1 break is simply where one segment ends, and it has no
    # opportunity to invent a root there.
    expected = _bezier_reference_roots(spline, _JUMP_KNOTS)
    assert found.shape == expected.shape
    np.testing.assert_allclose(found, expected, rtol=0.0, atol=_PARITY_RTOL)


# --- Kernels --------------------------------------------------------------------------


def test_knot_average_is_the_greville_abscissa() -> None:
    """``_knot_average`` averages the ``degree`` knots that follow its coefficient index."""
    degree = 3
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 5))

    for index in range(len(knots) - degree - 1):
        expected = float(np.mean(knots[index + 1 : index + degree + 1]))
        assert _knot_average(knots, degree, index) == pytest.approx(expected, abs=_EPS)


def test_span_at_stays_inside_the_last_span_of_the_domain() -> None:
    """The span search brackets every point, and treats the end of the domain from the left."""
    degree = 2
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 5))
    num_coeffs = len(knots) - degree - 1

    for point, expected in ((0.0, 2), (0.1, 2), (0.25, 3), (0.6, 4), (0.99, 5)):
        assert _span_at(knots, num_coeffs, degree, point) == expected
        assert knots[expected] <= point < knots[expected + 1]

    # The one point no bracket contains: an open knot vector's last span is half open, so
    # the search stops at it rather than walking into the clamped tail.
    assert _span_at(knots, num_coeffs, degree, 1.0) == num_coeffs - 1


def test_residual_at_evaluates_the_spline_it_is_given() -> None:
    """``_residual_at`` is ``|f|``, and agrees with the public evaluator that certifies it."""
    degree = 3
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 5))
    coeffs = _polynomial_coefficients(knots, degree, (0.2, 0.5, 0.9))
    num_coeffs = coeffs.shape[0]
    work = np.empty(degree + 1, dtype=np.float64)
    points = np.array([0.0, 0.2, 0.37, 0.5, 0.75, 0.9, 1.0])

    expected = np.abs(np.asarray(_spline(knots, coeffs, degree).evaluate(points)).reshape(-1))
    got = [_residual_at(knots, coeffs, num_coeffs, degree, float(p), work) for p in points]

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=(degree + 1) * _EPS)


def test_zero_index_needs_a_non_zero_left_neighbour() -> None:
    """A pinned coefficient is a barrier: no sign change is reported across it."""
    coeffs = np.array([0.0, -1.0, 1.0, 1.0, -1.0])

    assert _zero_index(coeffs, coeffs.shape[0], 1) == 2
    assert _zero_index(coeffs, coeffs.shape[0], 3) == 4
    assert _zero_index(np.array([0.0, 0.0, 1.0]), 3, 1) == -1


def test_insert_knot_leaves_the_spline_unchanged() -> None:
    """Boehm insertion refines the representation without moving the function."""
    degree = 3
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 5))
    num_coeffs = len(knots) - degree - 1
    coeffs = np.linspace(-1.0, 2.0, num_coeffs) ** 2 - 0.5
    sample = np.linspace(0.0, 1.0, 37)
    before = np.asarray(_spline(knots, coeffs, degree).evaluate(sample)).reshape(-1)

    buffer_knots = np.empty(knots.shape[0] + 1)
    buffer_knots[: knots.shape[0]] = knots
    buffer_coeffs = np.empty(num_coeffs + 1)
    buffer_coeffs[:num_coeffs] = coeffs
    point = 0.375
    span = int(np.searchsorted(knots, point, side="right") - 1)
    _insert_knot(buffer_knots, buffer_coeffs, num_coeffs, degree, point, span)

    after = np.asarray(
        _spline(buffer_knots, buffer_coeffs, degree).evaluate(sample), dtype=np.float64
    ).reshape(-1)
    np.testing.assert_allclose(after, before, rtol=0.0, atol=16 * _EPS)


def test_split_at_root_leaves_an_open_knot_vector() -> None:
    """The split raises the zero to multiplicity ``degree + 1`` and clamps the remainder.

    This is the invariant that keeps every later span search inside the buffer: after the
    caller shifts both arrays down by ``offset``, the knot vector is open again, this time
    on ``[root, end]``, exactly as the original input was.
    """
    degree = 3
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 6))
    coeffs = _polynomial_coefficients(knots, degree, (0.2, 0.5, 0.9))
    num_coeffs = coeffs.shape[0]
    capacity = num_coeffs + 4 * (degree + 1)
    buffer_coeffs = np.empty(capacity)
    buffer_coeffs[:num_coeffs] = coeffs
    buffer_knots = np.empty(capacity + degree + 1)
    buffer_knots[: knots.shape[0]] = knots

    root = 0.5
    num_live, offset = _split_at_root(
        buffer_knots, buffer_coeffs, num_coeffs, degree, root, _zero_tol(coeffs, degree)
    )

    assert np.all(buffer_knots[offset : offset + degree + 1] == root)
    assert buffer_knots[offset - 1] < root
    assert buffer_coeffs[offset] == 0.0
    assert num_live == num_coeffs + degree + 1  # 0.5 is not a breakpoint here


def test_is_zero_index_rejects_an_index_outside_the_polygon() -> None:
    """The tracker steps its index past the end, so the bound check has to hold there."""
    coeffs = np.array([-1.0, 1.0, 2.0])

    assert _is_zero_index(coeffs, 3, 1)
    assert not _is_zero_index(coeffs, 3, 0)
    assert not _is_zero_index(coeffs, 3, 3)


def test_merge_roots_on_an_empty_input() -> None:
    """No roots in, no roots out, and a buffer that is still safe to index."""
    merged, count = _merge_roots(np.empty(0), np.empty(0))

    assert count == 0
    assert merged.shape == (1,)


def test_merge_roots_collapses_a_run_but_not_a_gap() -> None:
    """Neighbours within the larger of their two radii become one root, at the midpoint."""
    roots = np.array([0.1, 0.1 + 1e-9, 0.5, 0.9])
    radii = np.array([1e-8, 1e-8, 1e-12, 1e-12])

    merged, count = _merge_roots(roots, radii)

    assert count == 3
    np.testing.assert_allclose(merged[:count], [0.1 + 0.5e-9, 0.5, 0.9], rtol=0.0, atol=1e-15)


def test_kernel_reports_a_zero_whose_budget_ran_out() -> None:
    """Cutting the insertion budget to one is reported rather than silently accepted."""
    degree = 3
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 6))
    coeffs = _polynomial_coefficients(knots, degree, (0.2, 0.5, 0.9))
    zero_tol = _zero_tol(coeffs, degree)

    _, _, truncated, abandoned = _morken_reimers_roots(
        knots, degree, coeffs, 1e-15, zero_tol, _root_tol(coeffs, knots, degree), 1
    )

    assert truncated + abandoned > 0


def test_budget_exhaustion_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero reported at an unconverged iterate says so: it may be less accurate than tol.

    A budget of eight is enough for the residual to certify all three zeros of this
    spline but not for the iterates of the first to stagnate, which is the case the
    warning is for. It has to be measured rather than reasoned: the budget at which the
    two stop coinciding is a property of this spline.
    """
    degree = 3
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 6))
    coeffs = _polynomial_coefficients(knots, degree, (0.2, 0.5, 0.9))
    monkeypatch.setattr(_bspline_roots, "_max_insertions", lambda _degree: 8)

    with pytest.warns(RuntimeWarning, match="reported at the last iterate reached"):
        found = find_roots(_spline(knots, coeffs, degree))

    assert found.shape == (3,)


def test_a_sign_change_the_budget_never_certified_is_not_silently_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejecting an uncertified iterate loses a zero, and losing one silently is not allowed.

    Uniform residual testing is what keeps a non-root out of the result, and the price is
    that a sign change whose tracking never converges is dropped rather than reported at
    whatever it reached. That is the sound direction, but it is a *missing* root, so it is
    the one rejection the caller is told about. A budget of one exhausts before any of the
    three zeros below is located at all.
    """
    degree = 3
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 6))
    coeffs = _polynomial_coefficients(knots, degree, (0.2, 0.5, 0.9))
    monkeypatch.setattr(_bspline_roots, "_max_insertions", lambda _degree: 1)

    with pytest.warns(RuntimeWarning, match="may be missing a root"):
        found = find_roots(_spline(knots, coeffs, degree))

    assert found.shape == (0,)


def test_insertion_budget_grows_with_degree() -> None:
    """The budget is degree-aware: convergence is quadratic per ``degree - 1`` insertions."""
    assert _max_insertions(1) == 64
    assert _max_insertions(25) > 4 * _max_insertions(1) // 2
    assert _max_insertions(5) > _max_insertions(3) > _max_insertions(1)


# --- Public surface -------------------------------------------------------------------


def test_returned_roots_are_sorted_read_only_float64() -> None:
    """The result follows the same contract as every other array pantr hands out."""
    degree = 3
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 6))
    coeffs = _polynomial_coefficients(knots, degree, (0.2, 0.5, 0.9))

    found = find_roots(_spline(knots, coeffs, degree))

    assert found.dtype == np.float64
    assert not found.flags.writeable
    assert np.all(np.diff(found) > 0.0)


def test_rational_spline_reports_the_zeros_of_its_numerator() -> None:
    """With positive weights the zeros of the mapping are the zeros of the numerator."""
    degree = 2
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 5))
    numerator = _polynomial_coefficients(knots, degree, (0.3, 0.7))
    weights = np.linspace(1.0, 3.0, numerator.shape[0])
    # Control points are homogeneous, so the first slot holds the numerator spline itself:
    # the mapping is that spline over the weight spline, and the weights being positive its
    # zeros are the numerator's.
    control = np.stack((numerator, weights), axis=1)
    space = BsplineSpace([BsplineSpace1D(knots, degree, snap_knots=False)])

    found = find_roots(Bspline(space, control, is_rational=True))

    np.testing.assert_allclose(found, [0.3, 0.7], rtol=0.0, atol=_ROOT_ULPS * _EPS)


def test_float32_spline_is_solved_in_float64() -> None:
    """A float32 spline is promoted exactly; only the tolerances stay at float32 level."""
    degree = 3
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 6))
    coeffs = _polynomial_coefficients(knots, degree, (0.25, 0.5, 0.75))

    found = find_roots(_spline(knots, coeffs, degree, dtype=np.float32))

    assert found.dtype == np.float64
    np.testing.assert_allclose(found, [0.25, 0.5, 0.75], rtol=0.0, atol=1e-6)


def test_a_looser_tolerance_still_finds_every_root() -> None:
    """``tol`` trades parametric accuracy for insertions, never a root."""
    degree = 3
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 6))
    coeffs = _polynomial_coefficients(knots, degree, (0.2, 0.5, 0.9))

    found = find_roots(_spline(knots, coeffs, degree), tol=1e-8)

    np.testing.assert_allclose(found, [0.2, 0.5, 0.9], rtol=0.0, atol=1e-7)


def test_unclamped_spline_is_converted_before_solving() -> None:
    """The method needs an open knot vector; a uniform one is converted, not rejected."""
    degree = 2
    knots = np.linspace(-0.4, 1.4, 10)
    coeffs = np.array([1.0, 0.6, -0.4, -0.8, 0.2, 1.0, 1.4])
    spline = _spline(knots, coeffs, degree)

    found = find_roots(spline)

    values = np.abs(np.asarray(spline.evaluate(found), dtype=np.float64).reshape(-1))
    assert found.size > 0
    assert float(values.max()) <= 1e-12


# --- Validation -----------------------------------------------------------------------


def test_rejects_a_non_bspline() -> None:
    """The argument has to be a B-spline, and the message says what arrived instead."""
    with pytest.raises(TypeError, match="Expected a Bspline"):
        find_roots(np.array([1.0, -1.0]))  # type: ignore[arg-type]


def test_rejects_a_multivariate_spline() -> None:
    """Root finding is one-dimensional: a surface has a zero set, not a set of roots."""
    space = BsplineSpace([BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0]), 1)] * 2)
    spline = Bspline(space, np.arange(4.0).reshape(2, 2, 1) - 1.5)

    with pytest.raises(ValueError, match="univariate"):
        find_roots(spline)


def test_rejects_a_vector_valued_spline() -> None:
    """A curve in the plane has no zeros in this sense; the caller must pick a component."""
    space = BsplineSpace([BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0]), 1)])
    spline = Bspline(space, np.array([[-1.0, 1.0], [1.0, 2.0]]))

    with pytest.raises(ValueError, match="scalar valued"):
        find_roots(spline)


def test_rejects_a_degree_zero_spline() -> None:
    """A piecewise constant vanishes on whole intervals or nowhere, never at points."""
    space = BsplineSpace([BsplineSpace1D(np.array([0.0, 0.5, 1.0]), 0)])
    spline = Bspline(space, np.array([[-1.0], [1.0]]))

    with pytest.raises(ValueError, match="degree >= 1"):
        find_roots(spline)


def test_rejects_a_non_positive_weight() -> None:
    """Only positive weights make the numerator's zeros the mapping's zeros."""
    degree = 1
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 3))
    control = np.array([[-1.0, 1.0], [0.5, 0.0], [1.0, 2.0]])
    space = BsplineSpace([BsplineSpace1D(knots, degree, snap_knots=False)])

    with pytest.raises(ValueError, match="strictly positive weights"):
        find_roots(Bspline(space, control, is_rational=True))


def test_rejects_an_identically_zero_spline() -> None:
    """Every point of the domain would be a root, so there is nothing to report."""
    degree = 2
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 4))

    with pytest.raises(ValueError, match="identically zero"):
        find_roots(_spline(knots, np.zeros(len(knots) - degree - 1), degree))


def test_rejects_a_spline_that_vanishes_on_a_knot_interval() -> None:
    """A disconnected spline breaks the variation-diminishing bound the method rests on."""
    degree = 2
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 8))
    coeffs = np.zeros(len(knots) - degree - 1)
    coeffs[-2:] = [1.0, 2.0]

    with pytest.raises(ValueError, match="vanishes identically"):
        find_roots(_spline(knots, coeffs, degree))


@pytest.mark.parametrize("tol", [0.0, -1e-12])
def test_rejects_a_non_positive_tolerance(tol: float) -> None:
    """A tolerance that is not positive would make the stopping rule unreachable."""
    degree = 2
    knots = _open_knots(degree, np.linspace(0.0, 1.0, 4))
    coeffs = _polynomial_coefficients(knots, degree, (0.3, 0.7))

    with pytest.raises(ValueError, match="tol must be positive"):
        find_roots(_spline(knots, coeffs, degree), tol=tol)
