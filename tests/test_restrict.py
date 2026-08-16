"""Tests for Bspline.restrict() and Bezier.restrict()."""

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bezier import Bezier
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D, create_uniform_periodic_knots
from pantr.quad import PointsLattice

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_open_bspline_1d(
    degree: int = 2,
    n_intervals: int = 4,
    rank: int = 2,
    dtype: type = np.float64,
) -> Bspline:
    """Create a 1D open B-spline with random control points."""
    interior: npt.NDArray[np.float64] = np.linspace(0.0, 1.0, n_intervals + 1, dtype=dtype)[1:-1]
    knots: npt.NDArray[np.float64] = np.concatenate(
        [[0.0] * (degree + 1), interior, [1.0] * (degree + 1)]
    ).astype(dtype)
    space_1d = BsplineSpace1D(knots, degree)
    space = BsplineSpace([space_1d])
    rng = np.random.default_rng(42)
    ctrl: npt.NDArray[np.float64] = rng.random((space.num_total_basis, rank)).astype(dtype)
    return Bspline(space, ctrl)


def _make_open_bspline_2d(
    degrees: tuple[int, int] = (2, 1),
    n_intervals: tuple[int, int] = (4, 3),
    rank: int = 3,
    dtype: type = np.float64,
) -> Bspline:
    """Create a 2D open tensor-product B-spline with random control points."""
    spaces: list[BsplineSpace1D] = []
    for p, n_int in zip(degrees, n_intervals, strict=True):
        interior: npt.NDArray[np.float64] = np.linspace(0.0, 1.0, n_int + 1, dtype=dtype)[1:-1]
        knots: npt.NDArray[np.float64] = np.concatenate(
            [[0.0] * (p + 1), interior, [1.0] * (p + 1)]
        ).astype(dtype)
        spaces.append(BsplineSpace1D(knots, p))
    space = BsplineSpace(spaces)
    rng = np.random.default_rng(42)
    ctrl: npt.NDArray[np.float64] = rng.random((space.num_total_basis, rank)).astype(dtype)
    return Bspline(space, ctrl)


def _make_periodic_bspline(
    num_intervals: int = 4,
    degree: int = 2,
    rank: int = 2,
    dtype: type = np.float64,
) -> Bspline:
    """Create a 1D periodic B-spline with random control points."""
    knots = create_uniform_periodic_knots(num_intervals, degree, dtype=dtype)
    space_1d = BsplineSpace1D(knots, degree, periodic=True)
    space = BsplineSpace([space_1d])
    rng = np.random.default_rng(42)
    ctrl: npt.NDArray[np.float64] = rng.random((space.num_total_basis, rank)).astype(dtype)
    return Bspline(space, ctrl)


def _make_rational_bspline_1d(
    degree: int = 2,
    n_intervals: int = 3,
    dtype: type = np.float64,
) -> Bspline:
    """Create a 1D rational B-spline (NURBS) with random control points and weights."""
    interior: npt.NDArray[np.float64] = np.linspace(0.0, 1.0, n_intervals + 1, dtype=dtype)[1:-1]
    knots: npt.NDArray[np.float64] = np.concatenate(
        [[0.0] * (degree + 1), interior, [1.0] * (degree + 1)]
    ).astype(dtype)
    space_1d = BsplineSpace1D(knots, degree)
    space = BsplineSpace([space_1d])
    rng = np.random.default_rng(42)
    n = space.num_total_basis
    # rank 3: (x, y, w) in homogeneous coordinates
    ctrl: npt.NDArray[np.float64] = rng.random((n, 3)).astype(dtype)
    ctrl[:, 2] = rng.uniform(0.5, 2.0, size=n).astype(dtype)  # positive weights
    return Bspline(space, ctrl, is_rational=True)


# ---------------------------------------------------------------------------
# Bspline.restrict — 1D
# ---------------------------------------------------------------------------


class TestBsplineRestrict1D:
    """Tests for Bspline.restrict() on 1D B-splines."""

    def test_interior_subinterval(self) -> None:
        """Restrict to an interior sub-interval and verify evaluation agreement."""
        f = _make_open_bspline_1d()
        r = f.restrict((0.25, 0.75))

        pts = np.linspace(0.25, 0.75, 100)
        np.testing.assert_allclose(r.evaluate(pts), f.evaluate(pts), atol=1e-14)

    def test_left_at_domain(self) -> None:
        """Restrict with left bound at domain start (skip left insertion)."""
        f = _make_open_bspline_1d()
        r = f.restrict((0.0, 0.5))

        pts = np.linspace(0.0, 0.5, 100)
        np.testing.assert_allclose(r.evaluate(pts), f.evaluate(pts), atol=1e-14)

    def test_right_at_domain(self) -> None:
        """Restrict with right bound at domain end (skip right insertion)."""
        f = _make_open_bspline_1d()
        r = f.restrict((0.5, 1.0))

        pts = np.linspace(0.5, 1.0, 100)
        np.testing.assert_allclose(r.evaluate(pts), f.evaluate(pts), atol=1e-14)

    def test_single_span(self) -> None:
        """Restrict to a single knot span produces a Bézier-like result."""
        f = _make_open_bspline_1d(degree=2, n_intervals=4)
        r = f.restrict((0.25, 0.5))

        assert r.space.has_Bezier_like_knots()
        pts = np.linspace(0.25, 0.5, 50)
        np.testing.assert_allclose(r.evaluate(pts), f.evaluate(pts), atol=1e-14)

    def test_at_existing_knots(self) -> None:
        """Restrict at values that are already knots (less insertion needed)."""
        f = _make_open_bspline_1d(degree=2, n_intervals=4)
        # Knots are at 0, 0.25, 0.5, 0.75, 1.0
        r = f.restrict((0.25, 0.75))

        pts = np.linspace(0.25, 0.75, 100)
        np.testing.assert_allclose(r.evaluate(pts), f.evaluate(pts), atol=1e-14)

    def test_higher_degree(self) -> None:
        """Restrict works with higher degree B-splines."""
        f = _make_open_bspline_1d(degree=4, n_intervals=6)
        r = f.restrict((0.2, 0.8))

        pts = np.linspace(0.2, 0.8, 100)
        np.testing.assert_allclose(r.evaluate(pts), f.evaluate(pts), atol=1e-13)

    def test_periodic_auto_convert(self) -> None:
        """Restrict on a periodic B-spline auto-converts to open form."""
        f = _make_periodic_bspline(num_intervals=4, degree=2)
        f_open = f.to_open_bspline()
        r = f.restrict((0.25, 0.75))

        assert not r.space.spaces[0].periodic
        assert r.space.spaces[0].has_open_knots()
        pts = np.linspace(0.25, 0.75, 100)
        np.testing.assert_allclose(r.evaluate(pts), f_open.evaluate(pts), atol=1e-13)

    def test_rational(self) -> None:
        """Restrict preserves rationality and evaluates correctly."""
        f = _make_rational_bspline_1d()
        r = f.restrict((0.2, 0.8))

        assert r.is_rational
        pts = np.linspace(0.2, 0.8, 100)
        np.testing.assert_allclose(r.evaluate(pts), f.evaluate(pts), atol=1e-13)

    def test_domain_is_correct(self) -> None:
        """Restricted B-spline has the correct domain."""
        f = _make_open_bspline_1d()
        r = f.restrict((0.3, 0.7))

        domain = r.space.domain
        np.testing.assert_allclose(domain[0, 0], 0.3, atol=1e-15)
        np.testing.assert_allclose(domain[0, 1], 0.7, atol=1e-15)

    def test_float32(self) -> None:
        """Restrict works with float32 B-splines."""
        f = _make_open_bspline_1d(dtype=np.float32)
        r = f.restrict((0.25, 0.75))

        assert r.dtype == np.float32
        pts = np.linspace(0.25, 0.75, 50, dtype=np.float32)
        np.testing.assert_allclose(r.evaluate(pts), f.evaluate(pts), atol=1e-5)


# ---------------------------------------------------------------------------
# Bspline.restrict — error cases
# ---------------------------------------------------------------------------


class TestBsplineRestrictErrors:
    """Tests for Bspline.restrict() error handling."""

    def test_full_domain_raises(self) -> None:
        """Restrict to the full domain raises ValueError."""
        f = _make_open_bspline_1d()
        with pytest.raises(ValueError, match="full domain"):
            f.restrict((0.0, 1.0))

    def test_lower_ge_upper_raises(self) -> None:
        """Lower bound >= upper bound raises ValueError."""
        f = _make_open_bspline_1d()
        with pytest.raises(ValueError, match="strictly less"):
            f.restrict((0.5, 0.5))
        with pytest.raises(ValueError, match="strictly less"):
            f.restrict((0.7, 0.3))

    def test_out_of_domain_lower_raises(self) -> None:
        """Lower bound below domain raises ValueError."""
        f = _make_open_bspline_1d()
        with pytest.raises(ValueError, match="below the domain"):
            f.restrict((-0.1, 0.5))

    def test_out_of_domain_upper_raises(self) -> None:
        """Upper bound above domain raises ValueError."""
        f = _make_open_bspline_1d()
        with pytest.raises(ValueError, match="above the domain"):
            f.restrict((0.5, 1.1))

    def test_wrong_dim_raises(self) -> None:
        """Wrong sequence length for nD raises ValueError."""
        f = _make_open_bspline_2d()
        with pytest.raises(ValueError, match="must match dim"):
            f.restrict([(0.1, 0.9)])  # only 1 direction for a 2D spline

    def test_all_none_raises(self) -> None:
        """All directions None raises ValueError."""
        f = _make_open_bspline_2d()
        with pytest.raises(ValueError, match="non-None bounds"):
            f.restrict([None, None])


# ---------------------------------------------------------------------------
# Bspline.restrict — nD
# ---------------------------------------------------------------------------


class TestBsplineRestrictND:
    """Tests for Bspline.restrict() on multi-dimensional B-splines."""

    def test_restrict_one_direction(self) -> None:
        """Restrict 2D B-spline in one direction only."""
        f = _make_open_bspline_2d()
        r = f.restrict([(0.25, 0.75), None])

        # Direction 0 is restricted, direction 1 unchanged.
        domain = r.space.domain
        np.testing.assert_allclose(domain[0], [0.25, 0.75], atol=1e-15)
        np.testing.assert_allclose(domain[1], [0.0, 1.0], atol=1e-15)

    def test_restrict_both_directions(self) -> None:
        """Restrict 2D B-spline in both directions."""
        f = _make_open_bspline_2d()
        r = f.restrict([(0.25, 0.75), (0.33, 0.67)])

        domain = r.space.domain
        np.testing.assert_allclose(domain[0], [0.25, 0.75], atol=1e-15)
        np.testing.assert_allclose(domain[1], [0.33, 0.67], atol=1e-15)

    def test_restrict_2d_evaluation(self) -> None:
        """Restricted 2D B-spline evaluates the same as original on subdomain."""
        f = _make_open_bspline_2d()
        r = f.restrict([(0.25, 0.75), (0.33, 0.67)])

        pts_u = np.linspace(0.25, 0.75, 20)
        pts_v = np.linspace(0.33, 0.67, 15)
        lattice = PointsLattice([pts_u, pts_v])

        np.testing.assert_allclose(r.evaluate(lattice), f.evaluate(lattice), atol=1e-13)


class TestBsplineRestrictScaleCovariance:
    """The verdicts restrict reaches must not depend on where the domain sits.

    Restriction adds no tolerance of its own: every comparison it makes is a knot
    identity question answered by ``BsplineSpace1D.tolerance``, an absolute
    parametric length derived from the knot vector's own magnitude. These tests pin
    the property that follows: send ``x -> lambda * x`` and nothing changes but the
    coordinates.
    """

    @staticmethod
    def _shifted(lo: float, span: float, degree: int = 3, n_intervals: int = 5) -> Bspline:
        """Build an open 1D B-spline on ``[lo, lo + span]`` with a fixed control polygon."""
        interior = np.linspace(lo, lo + span, n_intervals + 1)[1:-1]
        knots = np.concatenate([[lo] * (degree + 1), interior, [lo + span] * (degree + 1)])
        space = BsplineSpace([BsplineSpace1D(knots, degree)])
        rng = np.random.default_rng(11)
        return Bspline(space, rng.random((space.num_total_basis, 2)))

    @pytest.mark.parametrize("lam", [1.0e-6, 1.0, 1.0e6, 1.0e9])
    def test_restriction_keeps_the_whole_boundary_group(self, lam: float) -> None:
        """The extraction must not truncate the clamped end at any coordinate magnitude.

        ``searchsorted(refined_knots, b_new + tol)`` is what cuts the sub-vector, and
        with the previous fixed ``1e-12`` the offset was absorbed by ``b_new`` itself
        once one ulp of the coordinate exceeded it (at ``lam = 1e6`` an ulp is about
        ``1.2e-10``), so the cut fell before the first copy of ``b_new`` and the last
        ``degree + 1`` knots were dropped. A tolerance of ``8 * eps * scale`` is at
        least four ulp of any coordinate present, so the offset always bites.
        """
        f = self._shifted(0.0, lam)
        degree = f.degree[0]
        r = f.restrict((0.2 * lam, 0.6 * lam))

        lo, hi = r.space.domain[0]
        assert lo == pytest.approx(0.2 * lam, rel=1.0e-14)
        assert hi == pytest.approx(0.6 * lam, rel=1.0e-14)
        # Clamped at both ends: the boundary group survived the cut intact.  Counted
        # through the space's own notion of knot identity, not bitwise: a boundary that
        # coincides with an existing knot keeps that knot's stored representative, which
        # may differ from the requested bound in the last bit.
        _, mult = r.space.spaces[0].get_unique_knots_and_multiplicity()
        assert mult[0] == degree + 1
        assert mult[-1] == degree + 1
        assert r.space.spaces[0].knots.size == mult.sum()

    @pytest.mark.parametrize("lam", [1.0e-6, 1.0, 1.0e6, 1.0e9])
    def test_the_restricted_map_still_agrees_with_the_original(self, lam: float) -> None:
        """Restriction is exact geometry, so the two evaluate alike at every scale."""
        f = self._shifted(0.0, lam)
        r = f.restrict((0.2 * lam, 0.6 * lam))
        pts = np.linspace(0.2 * lam, 0.6 * lam, 50)
        np.testing.assert_allclose(r.evaluate(pts), f.evaluate(pts), atol=0.0, rtol=1.0e-12)

    @pytest.mark.parametrize("lam", [1.0e-6, 1.0, 1.0e6, 1.0e9])
    def test_a_bound_one_ulp_outside_the_domain_is_still_the_domain(self, lam: float) -> None:
        """One ulp of slack is forgiven wherever the domain sits.

        The bound is snapped onto the endpoint rather than rejected, so the right
        insertion is skipped exactly as it is for the endpoint itself.
        """
        f = self._shifted(0.0, lam)
        just_outside = float(np.nextafter(lam, np.inf))
        r = f.restrict((0.4 * lam, just_outside))
        assert r.space.domain[0][1] == pytest.approx(lam, rel=1.0e-15)

    def test_an_offset_domain_reaches_the_same_verdicts_as_a_centered_one(self) -> None:
        """Translating the knot vector changes the arithmetic, not what restrict decides.

        Only the *verdicts* are covariant.  The control points are not expected to
        agree bitwise: at an offset of ``1e6`` the insertion weights are formed from
        coordinates of that magnitude, so they carry a relative error of about
        ``eps * 1e6 / span = 2.2e-10``, and the control points inherit it.  Measured
        agreement here is ``2.9e-10`` relative, which is that figure and not a defect;
        what must be identical is the knot structure the two runs produce.
        """
        base = self._shifted(0.0, 1.0)
        moved = self._shifted(1.0e6, 1.0)
        r_base = base.restrict((0.2, 0.6))
        r_moved = moved.restrict((1.0e6 + 0.2, 1.0e6 + 0.6))

        _, mult_base = r_base.space.spaces[0].get_unique_knots_and_multiplicity()
        _, mult_moved = r_moved.space.spaces[0].get_unique_knots_and_multiplicity()
        np.testing.assert_array_equal(mult_moved, mult_base)
        assert r_moved.control_points.shape == r_base.control_points.shape
        # The insertion-weight error above, with a factor of four for the chain.
        weight_error = 4.0 * np.finfo(np.float64).eps * 1.0e6
        np.testing.assert_allclose(
            r_moved.control_points, r_base.control_points, rtol=weight_error, atol=0.0
        )


# ---------------------------------------------------------------------------
# Bezier.restrict
# ---------------------------------------------------------------------------


class TestBezierRestrict:
    """Tests for Bezier.restrict()."""

    def test_1d_restrict(self) -> None:
        """Restrict a 1D Bézier to a sub-interval and verify reparametrization."""
        ctrl = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
        bez = Bezier(ctrl)
        r = bez.restrict((0.2, 0.8))

        assert r.degree == (2,)
        # r evaluates on [0,1], maps to bez on [0.2, 0.8]
        t_restr = np.linspace(0.0, 1.0, 100)
        t_mapped = 0.2 + 0.6 * t_restr
        np.testing.assert_allclose(r.evaluate(t_restr), bez.evaluate(t_mapped), atol=1e-14)

    def test_1d_restrict_higher_degree(self) -> None:
        """Restrict a cubic Bézier."""
        rng = np.random.default_rng(42)
        ctrl = rng.random((4, 3))
        bez = Bezier(ctrl)
        r = bez.restrict((0.1, 0.9))

        t_restr = np.linspace(0.0, 1.0, 100)
        t_mapped = 0.1 + 0.8 * t_restr
        np.testing.assert_allclose(r.evaluate(t_restr), bez.evaluate(t_mapped), atol=1e-13)

    def test_nd_restrict_one_direction(self) -> None:
        """Restrict a 2D Bézier in one direction."""
        rng = np.random.default_rng(42)
        ctrl = rng.random((3, 4, 2))
        bez = Bezier(ctrl)
        r = bez.restrict([(0.2, 0.8), None])

        assert r.degree == (2, 3)

    def test_rational(self) -> None:
        """Restrict a rational Bézier preserves rationality."""
        rng = np.random.default_rng(42)
        ctrl = rng.random((3, 3))
        ctrl[:, 2] = rng.uniform(0.5, 2.0, size=3)
        bez = Bezier(ctrl, is_rational=True)
        r = bez.restrict((0.2, 0.8))

        assert r.is_rational
        t_restr = np.linspace(0.0, 1.0, 100)
        t_mapped = 0.2 + 0.6 * t_restr  # [0,1] -> [0.2, 0.8]
        np.testing.assert_allclose(r.evaluate(t_restr), bez.evaluate(t_mapped), atol=1e-13)

    def test_full_domain_raises(self) -> None:
        """Restrict to full [0, 1] raises ValueError."""
        ctrl = np.array([[0.0], [1.0], [2.0]])
        bez = Bezier(ctrl)
        with pytest.raises(ValueError, match="full domain"):
            bez.restrict((0.0, 1.0))
