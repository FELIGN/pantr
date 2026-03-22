"""Tests for Bspline.restrict() and Bezier.restrict()."""

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bezier import Bezier
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D, create_uniform_periodic
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
    knots = create_uniform_periodic(num_intervals, degree, dtype=dtype)
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
