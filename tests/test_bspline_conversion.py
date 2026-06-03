"""Tests for Bspline conversion methods (to_open, to_periodic, to/from Bezier)."""

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bezier import Bezier
from pantr.bspline import (
    Bspline,
    BsplineSpace,
    BsplineSpace1D,
    create_from_bezier,
    create_uniform_periodic_knots,
)
from pantr.bspline._bspline_basis_core import _compute_basis_nurbs_book_impl
from pantr.bspline.spanwise_element_extraction import SpanwiseElementExtraction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_periodic_bspline(
    num_intervals: int,
    degree: int,
    dtype: type = np.float64,
    continuity: int | None = None,
) -> Bspline:
    """Create a simple periodic B-spline with sequential integer control points.

    Args:
        num_intervals (int): Number of intervals.
        degree (int): B-spline degree.
        dtype (type): Data type. Defaults to np.float64.
        continuity (int | None): Continuity level at interior knots. None uses degree-1
            (maximum continuity). Defaults to None.

    Returns:
        Bspline: A 1D periodic scalar B-spline.
    """
    knots = create_uniform_periodic_knots(num_intervals, degree, continuity=continuity, dtype=dtype)
    space_1d = BsplineSpace1D(knots, degree, periodic=True)
    space = BsplineSpace([space_1d])
    n = space.num_total_basis
    ctrl: npt.NDArray[np.float64] = np.arange(1, n + 1, dtype=dtype)
    return Bspline(space, ctrl)


def _eval_periodic_correct(f: Bspline, pts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Evaluate a periodic B-spline using the mathematically correct algorithm.

    Uses the unclamped ``first_basis = knot_id - degree`` index with modulo-wrapped
    control point lookup, which is the standard mathematical definition of a periodic
    B-spline. This differs from ``f.evaluate()`` which uses a clamped first_basis.

    Args:
        f (Bspline): A 1D periodic B-spline.
        pts (np.ndarray): Interior evaluation points (must lie strictly inside domain).

    Returns:
        np.ndarray: Evaluated values at the given points.
    """
    space_1d = f.space.spaces[0]
    knots = space_1d.knots
    p = space_1d.degree
    tol = float(space_1d.tolerance)
    n_stored = f.space.num_total_basis
    ctrl = f._control_points  # shape (n_stored, rank)

    # Compute knot spans using the non-periodic (unclamped) algorithm
    basis_out = np.zeros((len(pts), p + 1), dtype=np.float64)
    first_basis_arr = np.zeros(len(pts), dtype=np.int64)
    _compute_basis_nurbs_book_impl(knots, p, False, tol, pts, basis_out, first_basis_arr)

    rank = ctrl.shape[1]
    result = np.zeros((len(pts), rank), dtype=np.float64)
    for i in range(len(pts)):
        s = int(first_basis_arr[i])
        for j in range(p + 1):
            idx = (s + j) % n_stored
            result[i] += basis_out[i, j] * ctrl[idx]

    return result.squeeze()


def _make_unclamped_bspline(dtype: type = np.float64) -> Bspline:
    """Create a non-open non-periodic B-spline (uniform, no repeated boundary knots)."""
    # Cardinal-style knot vector: no repeated boundary knots, uniform spacing.
    # Degree 2, domain [2, 5] (interior of a larger uniform grid).
    knots: npt.NDArray[np.float64] = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=dtype)
    space_1d = BsplineSpace1D(knots, 2)
    space = BsplineSpace([space_1d])
    n = space.num_total_basis
    ctrl: npt.NDArray[np.float64] = np.arange(1, n + 1, dtype=dtype)
    return Bspline(space, ctrl)


def _make_non_open_bspline_varying_bdry(degree: int, boundary_mult: int) -> Bspline:
    """Create a non-open non-periodic B-spline with given boundary multiplicity.

    Args:
        degree (int): B-spline degree.
        boundary_mult (int): Knot multiplicity at domain endpoints (< degree + 1).

    Returns:
        Bspline: A 1D non-open scalar B-spline with sequential integer control points.
    """
    n_int = max(3, 2 * degree - 2 * boundary_mult + 3)
    interior = np.linspace(0.0, 1.0, n_int + 1)[1:-1]
    knots = np.concatenate([[0.0] * boundary_mult, interior, [1.0] * boundary_mult])
    space_1d = BsplineSpace1D(knots, degree)
    space = BsplineSpace([space_1d])
    n = space.num_total_basis
    ctrl: npt.NDArray[np.float64] = np.arange(1, n + 1, dtype=np.float64)
    return Bspline(space, ctrl)


def _evaluate_bezier_on_subdomain(
    bezier: Bezier,
    domain: npt.NDArray[np.float64],
    num_pts: int = 5,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    """Evaluate a Bezier on its original sub-domain by mapping to [0, 1].

    Args:
        bezier: Bezier patch.
        domain: Array of shape (dim, 2) with sub-domain bounds.
        num_pts: Number of evaluation points per direction.

    Returns:
        Tuple of (physical_pts, values) where physical_pts are in the original
        domain and values are the Bezier evaluations.
    """
    dim = bezier.dim
    if dim == 1:
        xi = np.linspace(0.0, 1.0, num_pts, dtype=np.float64)
        phys = domain[0, 0] + xi * (domain[0, 1] - domain[0, 0])
        vals = bezier.evaluate(xi)
        return phys, vals
    else:
        grids = []
        phys_grids = []
        for d in range(dim):
            xi_d = np.linspace(0.0, 1.0, num_pts, dtype=np.float64)
            phys_d = domain[d, 0] + xi_d * (domain[d, 1] - domain[d, 0])
            grids.append(xi_d)
            phys_grids.append(phys_d)
        mesh_xi = np.array(np.meshgrid(*grids, indexing="ij")).reshape(dim, -1).T
        mesh_phys = np.array(np.meshgrid(*phys_grids, indexing="ij")).reshape(dim, -1).T
        vals = bezier.evaluate(mesh_xi)
        return mesh_phys, vals


# ---------------------------------------------------------------------------
# to_open_bspline tests
# ---------------------------------------------------------------------------


class TestToOpenBspline:
    """Tests for Bspline.to_open_bspline()."""

    def test_periodic_to_open_is_non_periodic(self) -> None:
        """to_open_bspline on a periodic spline returns a non-periodic spline."""
        f = _make_periodic_bspline(3, 2)
        f_open = f.to_open_bspline()

        assert not f_open.space.spaces[0].periodic
        assert f_open.space.spaces[0].has_open_knots()

    def test_periodic_to_open_correctness(self) -> None:
        """Open B-spline agrees with the correct mathematical periodic evaluation."""
        f = _make_periodic_bspline(4, 2)
        f_open = f.to_open_bspline()

        a, b = f_open.space.spaces[0].domain
        # Exclude endpoints to avoid boundary-matching edge cases.
        pts = np.linspace(float(a), float(b), 51, dtype=np.float64)[1:-1]

        vals_correct = _eval_periodic_correct(f, pts)
        vals_open = f_open.evaluate(pts)

        np.testing.assert_allclose(vals_open, vals_correct, atol=1e-12)

    def test_periodic_to_open_degree3(self) -> None:
        """Works for degree-3 periodic splines."""
        f = _make_periodic_bspline(4, 3)
        f_open = f.to_open_bspline()

        assert f_open.space.spaces[0].has_open_knots()
        assert f_open.space.spaces[0].degree == 3

        a, b = f_open.space.spaces[0].domain
        pts = np.linspace(float(a), float(b), 51, dtype=np.float64)[1:-1]
        np.testing.assert_allclose(f_open.evaluate(pts), _eval_periodic_correct(f, pts), atol=1e-12)

    def test_non_open_non_periodic_to_open(self) -> None:
        """to_open_bspline on an unclamped non-periodic spline clamps it correctly."""
        f = _make_unclamped_bspline()
        assert not f.space.spaces[0].has_open_knots()
        assert not f.space.spaces[0].periodic

        f_open = f.to_open_bspline()

        assert f_open.space.spaces[0].has_open_knots()
        assert not f_open.space.spaces[0].periodic

        a, b = f.space.spaces[0].domain
        pts = np.linspace(float(a), float(b), 51, dtype=np.float64)[1:-1]
        np.testing.assert_allclose(f_open.evaluate(pts), f.evaluate(pts), atol=1e-12)

    def test_already_open_raises(self) -> None:
        """to_open_bspline on an already-open spline raises ValueError."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        space_1d = BsplineSpace1D(knots, 2)
        space = BsplineSpace([space_1d])
        f = Bspline(space, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))

        with pytest.raises(ValueError, match="already open"):
            f.to_open_bspline()

    def test_multidim_periodic_to_open(self) -> None:
        """to_open_bspline on a 2D spline with one periodic and one open direction."""
        # Direction 0: periodic degree-2, Direction 1: open degree-1
        knots_per = create_uniform_periodic_knots(4, 2, dtype=np.float64)
        knots_open = np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float64)
        space_per = BsplineSpace1D(knots_per, 2, periodic=True)
        space_open = BsplineSpace1D(knots_open, 1)
        space = BsplineSpace([space_per, space_open])

        n0 = space_per.num_basis
        n1 = space_open.num_basis
        rng = np.random.default_rng(0)
        ctrl = rng.random((n0 * n1,), dtype=np.float64)
        f = Bspline(space, ctrl)
        f_open = f.to_open_bspline()

        # Direction 0 must become open; direction 1 must remain unchanged.
        assert f_open.space.spaces[0].has_open_knots()
        assert not f_open.space.spaces[0].periodic
        assert f_open.space.spaces[1].has_open_knots()
        assert not f_open.space.spaces[1].periodic

    def test_multidim_already_open_raises(self) -> None:
        """to_open_bspline raises ValueError when all directions are already open."""
        knots1 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        knots2 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        space = BsplineSpace([BsplineSpace1D(knots1, 2), BsplineSpace1D(knots2, 1)])
        f = Bspline(space, np.ones((3 * 2,), dtype=np.float64))

        with pytest.raises(ValueError, match="already open"):
            f.to_open_bspline()

    def test_rational_periodic_to_open(self) -> None:
        """to_open_bspline works on rational periodic B-splines."""
        knots = create_uniform_periodic_knots(3, 2, dtype=np.float64)
        space_1d = BsplineSpace1D(knots, 2, periodic=True)
        space = BsplineSpace([space_1d])
        n = space.num_total_basis
        # rational: last coordinate is homogeneous weight (all weights = 1, so NURBS == B-spline)
        ctrl = np.column_stack(
            [np.arange(1, n + 1, dtype=np.float64), np.ones(n, dtype=np.float64)]
        )
        f = Bspline(space, ctrl, is_rational=True)
        f_open = f.to_open_bspline()

        assert f_open.is_rational
        assert not f_open.space.spaces[0].periodic
        assert f_open.space.spaces[0].has_open_knots()

        # With all weights = 1, NURBS reduces to polynomial B-spline.
        # Evaluate the scalar component (x*w / w = x) and compare against correct periodic eval.
        # Build a non-rational version of f with only the x-coordinates for comparison.
        f_scalar = Bspline(space, ctrl[:, 0], is_rational=False)
        a, b = f_open.space.spaces[0].domain
        pts = np.linspace(float(a), float(b), 51, dtype=np.float64)[1:-1]
        vals_correct = _eval_periodic_correct(f_scalar, pts)
        np.testing.assert_allclose(f_open.evaluate(pts), vals_correct, atol=1e-12)

    @pytest.mark.parametrize(
        "degree,boundary_mult",
        [
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 2),
            (4, 4),
        ],
    )
    def test_non_open_varying_bdry_to_open_correctness(
        self, degree: int, boundary_mult: int
    ) -> None:
        """to_open_bspline().evaluate() matches evaluate() for non-open varying boundary mult."""
        f = _make_non_open_bspline_varying_bdry(degree, boundary_mult)
        assert not f.space.spaces[0].has_open_knots()
        f_open = f.to_open_bspline()
        assert f_open.space.spaces[0].has_open_knots()

        a, b = f.space.spaces[0].domain
        pts = np.linspace(float(a), float(b), 31, dtype=np.float64)[1:-1]
        np.testing.assert_allclose(f_open.evaluate(pts), f.evaluate(pts), atol=1e-12)

    @pytest.mark.parametrize(
        "degree,boundary_mult",
        [
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 2),
            (4, 4),
        ],
    )
    def test_non_open_varying_bdry_evaluate_derivatives_order_0(
        self, degree: int, boundary_mult: int
    ) -> None:
        """evaluate_derivatives(pts, [0]) matches evaluate(pts) for non-open varying bdry mult."""
        f = _make_non_open_bspline_varying_bdry(degree, boundary_mult)
        a, b = f.space.spaces[0].domain
        pts = np.linspace(float(a), float(b), 21, dtype=np.float64)[1:-1]
        vals = f.evaluate(pts)
        derivs = f.evaluate_derivatives(pts, [0])
        np.testing.assert_allclose(derivs, vals, atol=1e-14)


# ---------------------------------------------------------------------------
# to_periodic tests
# ---------------------------------------------------------------------------


class TestToPeriodic:
    """Tests for Bspline.to_periodic()."""

    @pytest.mark.parametrize(
        "num_intervals, degree, continuity",
        [
            (2, 1, None),
            (3, 2, None),
            (4, 3, None),
            (5, 3, None),
            (4, 2, 1),
            (5, 3, 2),
        ],
    )
    def test_round_trip_max_and_high_regularity(
        self, num_intervals: int, degree: int, continuity: int | None
    ) -> None:
        """Periodic -> open -> periodic round-trip preserves evaluation."""
        f = _make_periodic_bspline(num_intervals, degree, continuity=continuity)
        f_open = f.to_open_bspline()
        c = continuity if continuity is not None else degree - 1
        f_per = f_open.to_periodic(continuity=c)

        assert f_per.space.spaces[0].periodic
        assert f_per.space.spaces[0].num_basis == f.space.spaces[0].num_basis

        a, b = f.space.spaces[0].domain
        pts = np.linspace(float(a), float(b), 51, dtype=np.float64)[1:-1]
        np.testing.assert_allclose(f_per.evaluate(pts), f.evaluate(pts), atol=1e-11)

    def test_round_trip_default_continuity(self) -> None:
        """to_periodic() with no continuity arg uses max regularity."""
        f = _make_periodic_bspline(4, 3)
        f_open = f.to_open_bspline()
        f_per = f_open.to_periodic()

        assert f_per.space.spaces[0].periodic
        a, b = f.space.spaces[0].domain
        pts = np.linspace(float(a), float(b), 51, dtype=np.float64)[1:-1]
        np.testing.assert_allclose(f_per.evaluate(pts), f.evaluate(pts), atol=1e-11)

    def test_non_periodic_raises(self) -> None:
        """to_periodic on a non-periodic open B-spline raises ValueError."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        ctrl = np.array([1.0, 2.0, 5.0, 3.0])  # P_0 != P_{n-1}
        f = Bspline(space, ctrl)
        with pytest.raises(ValueError, match="not periodic"):
            f.to_periodic()

    def test_already_periodic_raises(self) -> None:
        """to_periodic on an already-periodic B-spline raises ValueError."""
        f = _make_periodic_bspline(3, 2)
        with pytest.raises(ValueError, match="already periodic"):
            f.to_periodic()

    def test_continuity_out_of_range_raises(self) -> None:
        """to_periodic with invalid continuity raises ValueError."""
        f = _make_periodic_bspline(3, 2)
        f_open = f.to_open_bspline()
        with pytest.raises(ValueError, match="continuity"):
            f_open.to_periodic(continuity=2)  # degree=2, max cont=1

    def test_rational_round_trip(self) -> None:
        """Rational periodic -> open -> periodic preserves evaluation."""
        knots = create_uniform_periodic_knots(3, 2)
        space = BsplineSpace([BsplineSpace1D(knots, 2, periodic=True)])
        # 2D rational control points (x, y, w)
        n = space.spaces[0].num_basis
        ctrl = np.column_stack(
            [
                np.cos(np.linspace(0, 2 * np.pi, n, endpoint=False)),
                np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False)),
                np.ones(n),
            ]
        )
        f = Bspline(space, ctrl, is_rational=True)
        f_open = f.to_open_bspline()
        f_per = f_open.to_periodic()

        assert f_per.is_rational
        assert f_per.space.spaces[0].periodic
        a, b = f.space.spaces[0].domain
        pts = np.linspace(float(a), float(b), 41, dtype=np.float64)[1:-1]
        np.testing.assert_allclose(f_per.evaluate(pts), f.evaluate(pts), atol=1e-11)

    def test_multidim_round_trip(self) -> None:
        """2D B-spline with both directions periodic round-trips correctly."""
        from pantr.quad import PointsLattice  # noqa: PLC0415

        knots_u = create_uniform_periodic_knots(3, 2)
        knots_v = create_uniform_periodic_knots(4, 2)
        space = BsplineSpace(
            [
                BsplineSpace1D(knots_u, 2, periodic=True),
                BsplineSpace1D(knots_v, 2, periodic=True),
            ]
        )
        nu = space.spaces[0].num_basis
        nv = space.spaces[1].num_basis
        ctrl = np.arange(1, nu * nv + 1, dtype=np.float64).reshape(nu, nv, 1)
        f = Bspline(space, ctrl)
        f_open = f.to_open_bspline()
        f_per = f_open.to_periodic()

        assert all(s.periodic for s in f_per.space.spaces)
        assert f_per.space.spaces[0].num_basis == nu
        assert f_per.space.spaces[1].num_basis == nv

        au, bu = f.space.spaces[0].domain
        av, bv = f.space.spaces[1].domain
        pts_u = np.linspace(float(au), float(bu), 11, dtype=np.float64)[1:-1]
        pts_v = np.linspace(float(av), float(bv), 11, dtype=np.float64)[1:-1]
        lattice = PointsLattice([pts_u, pts_v])
        np.testing.assert_allclose(f_per.evaluate(lattice), f.evaluate(lattice), atol=1e-10)

    def test_selective_direction_conversion(self) -> None:
        """to_periodic with tuple continuity converts only selected directions."""
        from pantr.quad import PointsLattice  # noqa: PLC0415

        # 2D: direction 0 is periodic (open after to_open), direction 1 is open (non-periodic).
        knots_per = create_uniform_periodic_knots(3, 2)
        knots_open = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        space = BsplineSpace(
            [
                BsplineSpace1D(knots_per, 2, periodic=True),
                BsplineSpace1D(knots_open, 2),
            ]
        )
        n0 = space.spaces[0].num_basis
        n1 = space.spaces[1].num_basis
        rng = np.random.default_rng(42)
        ctrl = rng.random((n0, n1, 1))
        f = Bspline(space, ctrl)

        # Convert to open in all directions.
        f_open = f.to_open_bspline()
        assert not any(s.periodic for s in f_open.space.spaces)

        # Convert only direction 0 back to periodic, skip direction 1.
        f_mixed = f_open.to_periodic(continuity=(1, None))
        assert f_mixed.space.spaces[0].periodic
        assert not f_mixed.space.spaces[1].periodic

        # Evaluation should match on the original domain.
        a0, b0 = f.space.spaces[0].domain
        a1, b1 = f.space.spaces[1].domain
        pts_0 = np.linspace(float(a0), float(b0), 11, dtype=np.float64)[1:-1]
        pts_1 = np.linspace(float(a1), float(b1), 11, dtype=np.float64)[1:-1]
        lattice = PointsLattice([pts_0, pts_1])
        np.testing.assert_allclose(f_mixed.evaluate(lattice), f.evaluate(lattice), atol=1e-10)

    def test_all_skipped_raises(self) -> None:
        """to_periodic raises when all directions are skipped or already periodic."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        f = Bspline(space, np.array([1.0, 2.0, 3.0, 4.0]))
        with pytest.raises(ValueError, match="No direction to convert"):
            f.to_periodic(continuity=(None,))


# ---------------------------------------------------------------------------
# To / from Bezier
# ---------------------------------------------------------------------------


class TestToBezier:
    """Test Bspline.to_bezier conversion."""

    def test_bezier_like_1d(self) -> None:
        """Test to_bezier for a B-spline with Bézier-like knots."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        cp = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        bs = Bspline(space, cp)
        bez = bs.to_bezier()
        assert isinstance(bez, Bezier)
        assert bez.degree == (2,)
        np.testing.assert_array_equal(bez.control_points, cp)

    def test_bezier_like_2d(self) -> None:
        """Test to_bezier for a 2D B-spline with Bézier-like knots."""
        s0 = BsplineSpace1D([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2)
        s1 = BsplineSpace1D([0.0, 0.0, 1.0, 1.0], 1)
        space = BsplineSpace([s0, s1])
        cp = np.ones((3, 2, 1), dtype=np.float64)
        bs = Bspline(space, cp)
        bez = bs.to_bezier()
        assert isinstance(bez, Bezier)
        assert bez.degree == (2, 1)
        np.testing.assert_array_equal(bez.control_points, cp)

    def test_non_open_single_element(self) -> None:
        """Test to_bezier for a non-open (unclamped) B-spline with one element."""
        # Degree 1, knots [0, 0.5, 1, 1.5] → 2 basis fns, domain [0.5, 1.0].
        # Not open, but a single span — opening produces Bézier-like knots.
        knots = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
        space = BsplineSpace([BsplineSpace1D(knots, 1)])
        cp = np.array([[1.0], [3.0]], dtype=np.float64)
        bs = Bspline(space, cp)
        bez = bs.to_bezier()
        # The open form has domain [0.5, 1.0] mapped to Bézier on [0, 1].
        # Compare via the open B-spline intermediate.
        bs_open = bs.to_open_bspline()
        np.testing.assert_array_equal(bez.control_points, bs_open.control_points)

    def test_periodic_multi_element_raises(self) -> None:
        """Test that to_bezier raises for a periodic B-spline with multiple elements."""
        f = _make_periodic_bspline(3, 2)
        with pytest.raises(ValueError, match="more than one element"):
            f.to_bezier()

    def test_multi_element_raises(self) -> None:
        """Test that to_bezier raises for multi-element B-splines."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        cp = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        bs = Bspline(space, cp)
        with pytest.raises(ValueError, match="more than one element"):
            bs.to_bezier()

    def test_copy_true(self) -> None:
        """Test that to_bezier with copy=True creates independent arrays."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        cp = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        bs = Bspline(space, cp)
        bez = bs.to_bezier(copy=True)
        assert not np.shares_memory(bs.control_points, bez.control_points)

    def test_copy_false(self) -> None:
        """Test that to_bezier with copy=False shares the control point array."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        cp = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        bs = Bspline(space, cp)
        bez = bs.to_bezier(copy=False)
        assert np.shares_memory(bs.control_points, bez.control_points)

    def test_default_copies(self) -> None:
        """Test that to_bezier copies by default."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        cp = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        bs = Bspline(space, cp)
        bez = bs.to_bezier()
        assert not np.shares_memory(bs.control_points, bez.control_points)

    def test_rational(self) -> None:
        """Test to_bezier preserves rationality."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        cp = np.array([[1.0, 0.5], [2.0, 1.0], [3.0, 0.5]], dtype=np.float64)
        bs = Bspline(space, cp, is_rational=True)
        bez = bs.to_bezier()
        assert bez.is_rational
        np.testing.assert_array_equal(bez.control_points, cp)

    def test_roundtrip(self) -> None:
        """Test to_bezier -> from_bezier roundtrip."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        cp = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        bs = Bspline(space, cp)
        bs2 = create_from_bezier(bs.to_bezier())
        np.testing.assert_array_equal(bs2.control_points, bs.control_points)
        assert bs2.degree == bs.degree


class TestFromBezier:
    """Test create_from_bezier conversion."""

    def test_from_bezier_1d(self) -> None:
        """Test from_bezier creates a B-spline with Bézier-like knots."""
        cp = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        bez = Bezier(cp)
        bs = create_from_bezier(bez)
        assert bs.space.has_Bezier_like_knots()
        assert bs.degree == (2,)
        np.testing.assert_array_equal(bs.control_points, cp)

    def test_from_bezier_copy_true(self) -> None:
        """Test that from_bezier with copy=True creates independent arrays."""
        cp = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        bez = Bezier(cp)
        bs = create_from_bezier(bez, copy=True)
        assert not np.shares_memory(bez.control_points, bs.control_points)

    def test_from_bezier_copy_false(self) -> None:
        """Test that from_bezier with copy=False shares the control point array."""
        cp = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        bez = Bezier(cp)
        bs = create_from_bezier(bez, copy=False)
        assert np.shares_memory(bez.control_points, bs.control_points)

    def test_from_bezier_rational(self) -> None:
        """Test from_bezier preserves rationality."""
        cp = np.array([[1.0, 0.5], [2.0, 1.0], [3.0, 0.5]], dtype=np.float64)
        bez = Bezier(cp, is_rational=True)
        bs = create_from_bezier(bez)
        assert bs.is_rational
        np.testing.assert_array_equal(bs.control_points, cp)

    def test_from_bezier_2d(self) -> None:
        """Test from_bezier for a 2D Bézier."""
        cp = np.ones((3, 2, 1), dtype=np.float64)
        bez = Bezier(cp)
        bs = create_from_bezier(bez)
        assert bs.space.has_Bezier_like_knots()
        assert bs.degree == (2, 1)
        np.testing.assert_array_equal(bs.control_points, cp)


# ---------------------------------------------------------------------------
# to_beziers (Bezier decomposition)
# ---------------------------------------------------------------------------


class TestToBeziers:
    """Test Bspline.to_beziers decomposition."""

    def test_1d_single_element(self) -> None:
        """Test to_beziers for a single-element 1D B-spline (Bezier-like)."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        cp = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        bs = Bspline(space, cp)

        beziers = bs.to_beziers()
        assert beziers.shape == (1,)
        assert beziers[0].degree == (2,)
        np.testing.assert_allclose(beziers[0].control_points, cp)

    def test_1d_multi_element(self) -> None:
        """Test to_beziers for a multi-element 1D B-spline."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        cp = np.array([[0.0, 0.0], [0.5, 1.0], [0.5, 0.5], [1.0, 0.0]], dtype=np.float64)
        bs = Bspline(space, cp)

        beziers = bs.to_beziers()
        assert beziers.shape == (2,)
        for i in range(2):
            assert beziers[i].degree == (2,)
            assert isinstance(beziers[i], Bezier)

    def test_1d_evaluation_consistency(self) -> None:
        """Test that each 1D Bezier patch evaluates identically to the original B-spline."""
        knots = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0], dtype=np.float64
        )
        space = BsplineSpace([BsplineSpace1D(knots, 3)])
        rng = np.random.default_rng(42)
        cp = rng.standard_normal((7, 2))
        bs = Bspline(space, cp)

        beziers = bs.to_beziers()
        assert beziers.shape == (4,)

        unique_knots, _ = space.spaces[0].get_unique_knots_and_multiplicity(in_domain=True)
        for i in range(4):
            domain = np.array([[unique_knots[i], unique_knots[i + 1]]], dtype=np.float64)
            phys_pts, bez_vals = _evaluate_bezier_on_subdomain(beziers[i], domain, num_pts=7)
            bs_vals = bs.evaluate(phys_pts)
            np.testing.assert_allclose(bez_vals, bs_vals, atol=1e-12)

    def test_2d_tensor_product(self) -> None:
        """Test to_beziers for a 2D tensor-product B-spline."""
        s0 = BsplineSpace1D([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], 2)
        s1 = BsplineSpace1D([0.0, 0.0, 0.5, 1.0, 1.0], 1)
        space = BsplineSpace([s0, s1])
        rng = np.random.default_rng(123)
        cp = rng.standard_normal((*space.num_basis, 3))
        bs = Bspline(space, cp)

        beziers = bs.to_beziers()
        assert beziers.shape == (2, 2)
        for idx in np.ndindex(2, 2):
            bez = cast(Bezier, beziers[idx])
            assert bez.degree == (2, 1)
            assert isinstance(bez, Bezier)

    def test_2d_evaluation_consistency(self) -> None:
        """Test evaluation consistency for a 2D B-spline decomposition."""
        s0 = BsplineSpace1D([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], 2)
        s1 = BsplineSpace1D([0.0, 0.0, 0.5, 1.0, 1.0], 1)
        space = BsplineSpace([s0, s1])
        rng = np.random.default_rng(456)
        cp = rng.standard_normal((*space.num_basis, 2))
        bs = Bspline(space, cp)

        beziers = bs.to_beziers()
        uknots_0, _ = s0.get_unique_knots_and_multiplicity(in_domain=True)
        uknots_1, _ = s1.get_unique_knots_and_multiplicity(in_domain=True)

        for i0 in range(2):
            for i1 in range(2):
                domain = np.array(
                    [
                        [uknots_0[i0], uknots_0[i0 + 1]],
                        [uknots_1[i1], uknots_1[i1 + 1]],
                    ],
                    dtype=np.float64,
                )
                phys_pts, bez_vals = _evaluate_bezier_on_subdomain(
                    beziers[i0, i1], domain, num_pts=4
                )
                bs_vals = bs.evaluate(phys_pts)
                np.testing.assert_allclose(bez_vals, bs_vals, atol=1e-12)

    def test_3d_shape(self) -> None:
        """Test to_beziers for a 3D trivariate B-spline."""
        s0 = BsplineSpace1D([0.0, 0.0, 0.5, 1.0, 1.0], 1)
        s1 = BsplineSpace1D([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], 2)
        s2 = BsplineSpace1D([0.0, 0.0, 0.0, 1.0 / 3, 2.0 / 3, 1.0, 1.0, 1.0], 2)
        space = BsplineSpace([s0, s1, s2])
        rng = np.random.default_rng(789)
        cp = rng.standard_normal((*space.num_basis, 1))
        bs = Bspline(space, cp)

        beziers = bs.to_beziers()
        assert beziers.shape == (2, 2, 3)
        for idx in np.ndindex(2, 2, 3):
            assert cast(Bezier, beziers[idx]).degree == (1, 2, 2)

    def test_periodic(self) -> None:
        """Test to_beziers for a periodic B-spline."""
        bs_per = _make_periodic_bspline(3, 2)
        beziers = bs_per.to_beziers()
        assert beziers.shape == (3,)

        # Compare to decomposition of the open equivalent.
        bs_open = bs_per.to_open_bspline()
        beziers_open = bs_open.to_beziers()
        for i in range(3):
            np.testing.assert_allclose(
                beziers[i].control_points,
                beziers_open[i].control_points,
                atol=1e-12,
            )

    def test_rational(self) -> None:
        """Test to_beziers preserves rationality on each Bezier."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        cp = np.array(
            [[0.0, 1.0], [0.5, 1.0], [0.5, 0.5], [1.0, 1.0]],
            dtype=np.float64,
        )
        bs = Bspline(space, cp, is_rational=True)
        beziers = bs.to_beziers()
        assert beziers.shape == (2,)
        for i in range(2):
            assert beziers[i].is_rational

    def test_caching(self) -> None:
        """Test that to_beziers caches the result."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        cp = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
        bs = Bspline(space, cp)

        result1 = bs.to_beziers()
        result2 = bs.to_beziers()
        assert result1 is result2

    def test_cache_invalidation_reverse(self) -> None:
        """Test that in-place reverse invalidates the Bezier cache."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        cp = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
        bs = Bspline(space, cp)

        result_before = bs.to_beziers()
        bs.reverse(in_place=True)
        result_after = bs.to_beziers()
        assert result_before is not result_after

    def test_cache_invalidation_transform(self) -> None:
        """Test that in-place transform invalidates the Bezier cache."""
        from pantr.transform import AffineTransform  # noqa: PLC0415

        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        cp = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
        bs = Bspline(space, cp)

        result_before = bs.to_beziers()
        bs.transform(AffineTransform(np.eye(1), np.array([1.0])), in_place=True)
        result_after = bs.to_beziers()
        assert result_before is not result_after

    def test_cache_invalidation_permute(self) -> None:
        """Test that in-place permute_directions invalidates the Bezier cache."""
        s0 = BsplineSpace1D([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], 2)
        s1 = BsplineSpace1D([0.0, 0.0, 0.5, 1.0, 1.0], 1)
        space = BsplineSpace([s0, s1])
        rng = np.random.default_rng(99)
        cp = rng.standard_normal((*space.num_basis, 2))
        bs = Bspline(space, cp)

        result_before = bs.to_beziers()
        bs.permute_directions([1, 0], in_place=True)
        result_after = bs.to_beziers()
        assert result_before is not result_after

    def test_read_only_control_points(self) -> None:
        """Test that Bezier control points from to_beziers are read-only."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        cp = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
        bs = Bspline(space, cp)

        beziers = bs.to_beziers()
        for i in range(2):
            assert not beziers[i].control_points.flags.writeable

    def test_degree_preserved(self) -> None:
        """Test that each Bezier has the same degree as the original B-spline."""
        s0 = BsplineSpace1D([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], 2)
        s1 = BsplineSpace1D([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0], 3)
        space = BsplineSpace([s0, s1])
        rng = np.random.default_rng(11)
        cp = rng.standard_normal((*space.num_basis, 1))
        bs = Bspline(space, cp)

        beziers = bs.to_beziers()
        for idx in np.ndindex(*beziers.shape):
            assert cast(Bezier, beziers[idx]).degree == bs.degree

    def test_matches_to_bezier_single_element(self) -> None:
        """Test that to_beziers matches to_bezier for single-element B-splines."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        cp = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        bs = Bspline(space, cp)

        single_bez = bs.to_bezier()
        beziers = bs.to_beziers()
        np.testing.assert_allclose(beziers[0].control_points, single_bez.control_points, atol=1e-14)


# ---------------------------------------------------------------------------
# Regression parity: to_beziers() vs SpanwiseElementExtraction
# ---------------------------------------------------------------------------


class TestToBeziersSpanwiseParity:
    """Parity: to_beziers() control points match ``operator(idx).T @ ctrl_local``.

    These tests catch drift between _to_beziers_impl and the extraction operators
    cached by SpanwiseElementExtraction. The identity verified is::

        beziers[idx].control_points.reshape(N, rank) == M.T @ ctrl_local.reshape(N, rank)

    where ``M = extraction.operator(idx) = kron(C_0[i0], ..., C_{d-1}[i_{d-1}])`` and
    ``ctrl_local = ctrl[i0:i0+order_0, ..., i_{d-1}:i_{d-1}+order_{d-1}, :]``.

    The ``.T`` arises because ``_apply_bezier_extraction_1d_core`` computes
    ``C_i^T @ P_local`` (column-convention operators matching the kernel at
    ``_bspline_to_beziers.py:62``).

    Periodic spaces are excluded: ``SpanwiseElementExtraction`` raises
    ``NotImplementedError`` for them; periodic coverage lives in ``TestToBeziers``.
    """

    @pytest.mark.parametrize(
        "spaces,rng_seed",
        [
            ([BsplineSpace1D([0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0], 2)], 10),
            (
                [
                    BsplineSpace1D([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], 2),
                    BsplineSpace1D([0.0, 0.0, 0.5, 1.0, 1.0], 1),
                ],
                20,
            ),
            (
                [
                    BsplineSpace1D([0.0, 0.0, 0.5, 1.0, 1.0], 1),
                    BsplineSpace1D([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], 2),
                    BsplineSpace1D([0.0, 0.0, 0.0, 1.0 / 3, 2.0 / 3, 1.0, 1.0, 1.0], 2),
                ],
                30,
            ),
            # Single-element space: Bezier extraction is the identity everywhere,
            # so expected == ctrl_local. Exercises the identity short-circuit path.
            ([BsplineSpace1D([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2)], 40),
            # Degree-3 space: exercises order=4 in the Numba kernel and higher-degree
            # extraction operators with non-trivial entries.
            ([BsplineSpace1D([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0], 3)], 50),
        ],
        ids=["1d", "2d", "3d", "1d_single", "1d_deg3"],
    )
    def test_element_cp_matches_operator_transpose(
        self, spaces: list[BsplineSpace1D], rng_seed: int
    ) -> None:
        """Bezier control points equal SpanwiseElementExtraction.operator(idx).T @ ctrl_local."""
        space = BsplineSpace(spaces)
        rng = np.random.default_rng(rng_seed)
        cp = rng.standard_normal((*space.num_basis, 3))
        bs = Bspline(space, cp)
        beziers = bs.to_beziers()

        extraction = SpanwiseElementExtraction(space, target="bezier")
        degrees = bs.degree
        orders = tuple(p + 1 for p in degrees)

        for idx in np.ndindex(*space.num_intervals):
            # Gather local B-spline control points for this element.
            slices: tuple[slice | int, ...] = tuple(
                slice(i, i + orders[d]) for d, i in enumerate(idx)
            )
            ctrl_local = bs.control_points[(*slices, slice(None))]
            n_in = int(np.prod(orders))
            rank = ctrl_local.shape[-1]

            # Full Kronecker operator from SpanwiseElementExtraction.
            M = extraction.operator(idx)
            expected = (M.T @ ctrl_local.reshape(n_in, rank)).reshape(*orders, rank)

            np.testing.assert_allclose(
                cast(Bezier, beziers[idx]).control_points,
                expected,
                atol=1e-12,
                err_msg=f"Mismatch at element {idx}",
            )
