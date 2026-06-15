"""Correctness of :meth:`Bspline.to_beziers` across knot multiplicities.

The governing property is **exact reproduction**: each Bézier patch, evaluated
over its local ``[0, 1]^d``, must equal the parent B-spline evaluated over the
corresponding knot span. The historical bug selected each element's local
control points by element index (valid only for multiplicity-1 interior knots),
so B-splines with repeated interior knots (multiplicity ``>= 2`` — as used by
NURBS circles/disks) produced geometrically wrong patches.

The suite exercises that property across degree, interior-knot multiplicity
(including mixed and per-direction), dimension, rank, and rational/non-rational,
and pins it down with evaluate-independent analytic oracles (the radius of a
NURBS circle, hand-computed control points of a straight line) so a shared bug
in ``evaluate`` cannot mask a bug in the extraction.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bezier import Bezier
from pantr.bspline import (
    Bspline,
    BsplineSpace,
    BsplineSpace1D,
    create_uniform_periodic_knots,
)
from pantr.cad import create_circle

# ---------------------------------------------------------------------------
# Builders and helpers
# ---------------------------------------------------------------------------


def _open_knots(degree: int, n_elem: int, interior_mult: int = 1) -> list[float]:
    """Open (clamped) knot vector with uniform interior multiplicity.

    Interior knots ``1 .. n_elem-1`` each appear ``interior_mult`` times
    (``1 <= interior_mult <= degree``).
    """
    knots = [0.0] * (degree + 1)
    for i in range(1, n_elem):
        knots += [float(i)] * interior_mult
    knots += [float(n_elem)] * (degree + 1)
    return knots


def _make_bspline(  # noqa: PLR0913 -- distinct structural knobs, all needed
    degrees: tuple[int, ...],
    mults: tuple[int, ...],
    n_elems: tuple[int, ...],
    *,
    rank: int,
    rational: bool,
    seed: int = 0,
) -> Bspline:
    """Build a tensor-product B-spline with the given per-direction structure.

    Control points are deterministic pseudo-random; for rational splines the
    last coordinate is a strictly-positive weight column.
    """
    spaces = [
        BsplineSpace1D(_open_knots(p, n, m), p)
        for p, m, n in zip(degrees, mults, n_elems, strict=True)
    ]
    space = BsplineSpace(spaces)
    n_basis = space.num_basis  # tuple
    n_coords = rank + 1 if rational else rank
    rng = np.random.default_rng(seed)
    cp = rng.standard_normal((*n_basis, n_coords))
    if rational:
        cp[..., -1] = rng.uniform(0.5, 1.5, size=n_basis)  # positive weights
    return Bspline(space, cp, is_rational=rational)


def _domain_knots(sp1d: BsplineSpace1D) -> npt.NDArray[np.float64]:
    uk, _ = sp1d.get_unique_knots_and_multiplicity(in_domain=True)
    return np.asarray(uk, dtype=np.float64)


def _sample_local(dim: int, k: int = 4) -> npt.NDArray[np.float64]:
    """Tensor grid of *interior* local samples in ``[0, 1]^dim``, shape ``(n, dim)``."""
    s = np.linspace(0.13, 0.87, k)
    grids = np.meshgrid(*([s] * dim), indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=-1)


def _ev(obj: Bspline | Bezier, pts_nd: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Evaluate a Bspline/Bezier at ``(n, dim)`` points, handling the 1D signature."""
    if obj.dim == 1:
        return np.asarray(obj.evaluate(pts_nd[:, 0]))
    return np.asarray(obj.evaluate(pts_nd))


def assert_reproduces(bsp: Bspline, *, atol: float = 1e-10) -> None:
    """Assert every Bézier patch reproduces the parent over its element span."""
    beziers = bsp.to_beziers()
    uks = [_domain_knots(sp) for sp in bsp.space.spaces]
    dim = bsp.dim
    loc = _sample_local(dim)
    for idx in np.ndindex(*bsp.space.num_intervals):
        patch = cast(Bezier, beziers[idx])
        glob = np.empty_like(loc)
        for d in range(dim):
            t0, t1 = uks[d][idx[d]], uks[d][idx[d] + 1]
            glob[:, d] = t0 + loc[:, d] * (t1 - t0)
        pp = _ev(patch, loc)
        pg = _ev(bsp, glob)
        np.testing.assert_allclose(
            pp, pg, atol=atol, rtol=0.0, err_msg=f"patch {idx} does not reproduce parent"
        )


# ---------------------------------------------------------------------------
# 1. Exact-reproduction property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("rational", [False, True])
@pytest.mark.parametrize("rank", [2, 3])
def test_reproduction_1d(degree: int, rational: bool, rank: int) -> None:
    for mult in range(1, degree + 1):  # multiplicity 1 .. degree (C^{p-1} .. C^0)
        bsp = _make_bspline((degree,), (mult,), (4,), rank=rank, rational=rational)
        assert_reproduces(bsp)


@pytest.mark.parametrize(
    ("degrees", "mults", "n_elems"),
    [
        ((2, 2), (1, 1), (3, 3)),
        ((2, 2), (2, 1), (3, 3)),  # repeated knots in u only
        ((2, 2), (1, 2), (3, 3)),  # repeated knots in v only
        ((2, 2), (2, 2), (3, 3)),  # repeated in both
        ((3, 1), (2, 1), (4, 3)),  # anisotropic degree + repeated u
        ((2, 3), (2, 3), (3, 3)),  # different multiplicity per direction
    ],
)
@pytest.mark.parametrize("rational", [False, True])
def test_reproduction_2d(
    degrees: tuple[int, int], mults: tuple[int, int], n_elems: tuple[int, int], rational: bool
) -> None:
    bsp = _make_bspline(degrees, mults, n_elems, rank=3, rational=rational)
    assert_reproduces(bsp)


@pytest.mark.parametrize(
    ("degrees", "mults"),
    [((2, 2, 2), (2, 1, 2)), ((2, 1, 2), (1, 1, 2))],
)
@pytest.mark.parametrize("rational", [False, True])
def test_reproduction_3d(
    degrees: tuple[int, int, int], mults: tuple[int, int, int], rational: bool
) -> None:
    bsp = _make_bspline(degrees, mults, (3, 2, 3), rank=3, rational=rational)
    assert_reproduces(bsp)


@pytest.mark.parametrize("rational", [False, True])
def test_reproduction_mixed_multiplicities_1d(rational: bool) -> None:
    # degree 2; interior knots 1 (m=1), 2 (m=2), 3 (m=1): mixed multiplicities.
    knots = [0, 0, 0, 1, 2, 2, 3, 4, 4, 4]
    space = BsplineSpace([BsplineSpace1D(knots, 2)])
    n = space.num_total_basis
    rng = np.random.default_rng(1)
    cp = rng.standard_normal((n, 3 if rational else 2))
    if rational:
        cp[:, -1] = rng.uniform(0.5, 1.5, size=n)
    assert_reproduces(Bspline(space, cp, is_rational=rational))


# ---------------------------------------------------------------------------
# 2. Independent analytic oracles (no parent.evaluate)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "angle",
    [(0.0, 2.0 * np.pi), (0.0, 0.5 * np.pi), (0.25 * np.pi, 1.3 * np.pi)],
)
def test_circle_patches_have_unit_radius(angle: tuple[float, float]) -> None:
    radius = 1.0
    circ = create_circle(radius=radius, angle=angle)
    s = np.linspace(0.0, 1.0, 11)
    for i, patch in enumerate(circ.to_beziers().flat):
        pts = np.asarray(cast(Bezier, patch).evaluate(s))
        r = np.hypot(pts[:, 0], pts[:, 1])
        np.testing.assert_allclose(
            r, radius, atol=1e-10, rtol=0.0, err_msg=f"circle patch {i} off the circle"
        )


def test_extracted_control_points_match_expected_window() -> None:
    # Evaluate-independent control-point oracle. With a C^0 interior knot
    # (multiplicity == degree) and clamped ends, every element is already a Bézier
    # segment, so extraction is the identity and each patch's control points are
    # exactly the parent's support window: patch 0 == cp[0:3], patch 1 == cp[2:5].
    # The historical element-indexed bug takes cp[1:4] for element 1, so this
    # catches it directly without calling evaluate. The control points are
    # intentionally non-collinear (collinear points stay collinear under any
    # windowing, so a line cannot detect a wrong-window bug).
    knots = [0, 0, 0, 1, 1, 2, 2, 2]  # degree 2, 2 elements, interior mult == degree
    space = BsplineSpace([BsplineSpace1D(knots, 2)])
    cp = np.array([[0.0, 0.0], [1.0, 3.0], [2.0, -1.0], [3.0, 2.0], [4.0, 0.0]])
    patches = Bspline(space, cp).to_beziers()
    p0 = cast(Bezier, patches.flat[0])
    p1 = cast(Bezier, patches.flat[1])
    np.testing.assert_allclose(np.asarray(p0.control_points), cp[0:3], atol=1e-12)
    np.testing.assert_allclose(np.asarray(p1.control_points), cp[2:5], atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Targeted regression (fail on the buggy code)
# ---------------------------------------------------------------------------


def test_regression_nonrational_multiplicity_two() -> None:
    knots = [0, 0, 0, 1, 1, 2, 2, 2]  # mult-2 interior knot, 2 elements
    space = BsplineSpace([BsplineSpace1D(knots, 2)])
    n = space.num_total_basis
    cp = np.column_stack([np.linspace(0, 1, n), np.linspace(0, 1, n) ** 2])
    assert_reproduces(Bspline(space, cp))


def test_regression_full_circle_alternating_patches() -> None:
    circ = create_circle(radius=1.0, angle=(0.0, 2.0 * np.pi))
    s = np.linspace(0.0, 1.0, 9)
    for i, patch in enumerate(circ.to_beziers().flat):
        pts = np.asarray(cast(Bezier, patch).evaluate(s))
        r = np.hypot(pts[:, 0], pts[:, 1])
        np.testing.assert_allclose(r, 1.0, atol=1e-10, err_msg=f"patch {i} not on unit circle")


# ---------------------------------------------------------------------------
# 4. Structural invariants
# ---------------------------------------------------------------------------


def test_result_shape_matches_num_intervals() -> None:
    bsp = _make_bspline((2, 3), (2, 2), (3, 4), rank=3, rational=True)
    assert bsp.to_beziers().shape == bsp.space.num_intervals


def test_patch_degrees_match_parent() -> None:
    bsp = _make_bspline((2, 3), (2, 1), (3, 3), rank=3, rational=False)
    for patch in bsp.to_beziers().flat:
        assert cast(Bezier, patch).degree == bsp.degree


def test_patch_control_points_readonly() -> None:
    bsp = _make_bspline((2,), (2,), (3,), rank=2, rational=False)
    for patch in bsp.to_beziers().flat:
        assert not cast(Bezier, patch).control_points.flags.writeable


def test_periodic_converts_then_reproduces() -> None:
    knots = create_uniform_periodic_knots(5, 2)
    sp1d = BsplineSpace1D(knots, 2, periodic=True)
    space = BsplineSpace([sp1d])
    n = space.num_total_basis
    rng = np.random.default_rng(2)
    bsp = Bspline(space, rng.standard_normal((n, 2)))
    # to_beziers converts to open form; patches must reproduce that open spline.
    opened = bsp.to_open_bspline()
    beziers = bsp.to_beziers()
    uks = _domain_knots(opened.space.spaces[0])
    s = np.linspace(0.13, 0.87, 5)
    for i, patch in enumerate(beziers.flat):
        glob = uks[i] + s * (uks[i + 1] - uks[i])
        np.testing.assert_allclose(
            np.asarray(cast(Bezier, patch).evaluate(s)),
            np.asarray(opened.evaluate(glob)),
            atol=1e-10,
            err_msg=f"periodic patch {i} mismatch",
        )


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


def test_single_element() -> None:
    bsp = _make_bspline((3,), (1,), (1,), rank=2, rational=False)
    assert bsp.to_beziers().shape == (1,)
    assert_reproduces(bsp)


def test_interior_knot_full_multiplicity() -> None:
    # Interior multiplicity == degree => C^0 joints; elements already near-Bézier.
    for degree in (2, 3):
        bsp = _make_bspline((degree,), (degree,), (3,), rank=2, rational=True)
        assert_reproduces(bsp)


def test_degree_one() -> None:
    bsp = _make_bspline((1,), (1,), (4,), rank=2, rational=False)
    assert_reproduces(bsp)


def test_adjacent_repeated_knots_first_and_last_elements() -> None:
    # Two interior repeated knots; checks first and last elements explicitly.
    knots = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3]  # degree 2, 3 elements, all interior mult 2
    space = BsplineSpace([BsplineSpace1D(knots, 2)])
    n = space.num_total_basis
    rng = np.random.default_rng(3)
    cp = np.column_stack([np.linspace(0, 1, n), rng.standard_normal(n)])
    assert_reproduces(Bspline(space, cp))
