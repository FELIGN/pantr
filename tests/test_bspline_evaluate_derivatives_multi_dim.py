"""Tests for multi-dimensional B-spline derivative evaluation with per-direction orders."""

import numpy as np
import pytest

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D, create_uniform_periodic
from pantr.quad import PointsLattice


def _make_2d_space(
    knots1: list[float],
    degree1: int,
    knots2: list[float],
    degree2: int,
    dtype: type = float,
) -> BsplineSpace:
    """Build a 2D BsplineSpace from two knot vectors."""
    dt = np.float64 if dtype is float else np.float32
    s1 = BsplineSpace1D(np.array(knots1, dtype=dt), degree1)
    s2 = BsplineSpace1D(np.array(knots2, dtype=dt), degree2)
    return BsplineSpace([s1, s2])


class TestMultiDimDerivOutputShape:
    """Verify output shapes for all combinations of input type and rank."""

    def test_pts_array_scalar_non_rational_zeroth(self) -> None:
        """orders=[0,0] for scalar non-rational returns shape (n_pts,)."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.ones(6, dtype=np.float64)
        bspline = Bspline(space, cp)

        pts = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=np.float64)
        result = bspline.evaluate_derivatives(pts, [0, 0])

        assert result.shape == (2,)

    def test_pts_array_scalar_non_rational_mixed(self) -> None:
        """orders=[1,1] for scalar non-rational returns shape (n_pts,)."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.ones(6, dtype=np.float64)
        bspline = Bspline(space, cp)

        pts = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=np.float64)
        result = bspline.evaluate_derivatives(pts, [1, 1])

        assert result.shape == (2,)

    def test_pts_array_vector_non_rational(self) -> None:
        """orders=[1,0] for vector non-rational returns shape (n_pts, rank)."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        # 6 basis x 3 columns → rank 3
        cp = np.arange(18, dtype=np.float64)
        bspline = Bspline(space, cp)

        pts = np.array([[0.5, 0.5]], dtype=np.float64)
        result = bspline.evaluate_derivatives(pts, [1, 0])

        assert result.shape == (1, 3)

    def test_lattice_scalar_non_rational(self) -> None:
        """orders=[1,0] for lattice scalar non-rational returns (*grid_shape,)."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.ones(6, dtype=np.float64)
        bspline = Bspline(space, cp)

        pts1 = np.linspace(0.0, 1.0, 4, dtype=np.float64)
        pts2 = np.linspace(0.0, 1.0, 3, dtype=np.float64)
        lattice = PointsLattice([pts1, pts2])
        result = bspline.evaluate_derivatives(lattice, [1, 0])

        assert result.shape == (4, 3)

    def test_lattice_vector_non_rational(self) -> None:
        """orders=[0,1] for lattice vector non-rational returns (*grid_shape, rank)."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.arange(12, dtype=np.float64)
        bspline = Bspline(space, cp)

        pts1 = np.linspace(0.0, 1.0, 3, dtype=np.float64)
        pts2 = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        lattice = PointsLattice([pts1, pts2])
        result = bspline.evaluate_derivatives(lattice, [0, 1])

        assert result.shape == (3, 5, 2)

    def test_pts_array_rational_scalar(self) -> None:
        """orders=[1,0] for rational scalar (rank=1) returns shape (n_pts,)."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        # 6 basis x 2 columns → cp_size=2, rational rank=1
        cp = np.tile([1.0, 1.0], (6, 1)).astype(np.float64)
        bspline = Bspline(space, cp.ravel(), is_rational=True)

        pts = np.array([[0.5, 0.5]], dtype=np.float64)
        result = bspline.evaluate_derivatives(pts, [1, 0])

        assert result.shape == (1,)

    def test_pts_array_rational_vector(self) -> None:
        """orders=[1,0] for rational vector returns shape (n_pts, rank_value)."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        # 6 basis x 3 columns → cp_size=3, rational rank_value=2
        cp = np.tile([1.0, 0.0, 1.0], (6, 1)).astype(np.float64)
        bspline = Bspline(space, cp.ravel(), is_rational=True)

        pts = np.array([[0.5, 0.5]], dtype=np.float64)
        result = bspline.evaluate_derivatives(pts, [1, 0])

        assert result.shape == (1, 2)

    def test_orders_zero_shape(self) -> None:
        """orders=[0,0] produces shape (n_pts,) for scalar."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.ones(6, dtype=np.float64)
        bspline = Bspline(space, cp)

        pts = np.array([[0.5, 0.5]], dtype=np.float64)
        result = bspline.evaluate_derivatives(pts, [0, 0])

        assert result.shape == (1,)


class TestMultiDimDerivCorrectness:
    """Verify correctness of multi-dimensional derivative values."""

    def test_orders_zero_matches_evaluate_scalar(self) -> None:
        """orders=[0,0] must equal evaluate() for scalar."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        bspline = Bspline(space, cp)

        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 1, (20, 2)).astype(np.float64)

        result_d = bspline.evaluate_derivatives(pts, [0, 0])  # (20,)
        result_v = bspline.evaluate(pts)  # (20,)

        np.testing.assert_allclose(result_d, result_v, atol=1e-13)

    def test_orders_zero_matches_evaluate_vector(self) -> None:
        """orders=[0,0] matches evaluate() for vector-valued spline."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.arange(12, dtype=np.float64)
        bspline = Bspline(space, cp)

        rng = np.random.default_rng(1)
        pts = rng.uniform(0, 1, (15, 2)).astype(np.float64)

        result_d = bspline.evaluate_derivatives(pts, [0, 0])  # (15, 2)
        result_v = bspline.evaluate(pts)  # (15, 2)

        np.testing.assert_allclose(result_d, result_v, atol=1e-13)

    def test_linear_spline_constant_first_derivative(self) -> None:
        """Degree-(1,1) bilinear spline f(u,v)=u+v: df/du=1, df/dv=1 everywhere."""
        s1 = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), 1)
        s2 = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), 1)
        space = BsplineSpace([s1, s2])
        cp = np.array([0.0, 1.0, 1.0, 2.0], dtype=np.float64)
        bspline = Bspline(space, cp)

        pts = np.array([[0.0, 0.0], [0.5, 0.5], [0.75, 0.25]], dtype=np.float64)

        # f(u,v) = u + v
        np.testing.assert_allclose(
            bspline.evaluate_derivatives(pts, [0, 0]),
            pts[:, 0] + pts[:, 1],
            atol=1e-13,
        )
        # df/du = 1
        np.testing.assert_allclose(
            bspline.evaluate_derivatives(pts, [1, 0]),
            np.ones(3),
            atol=1e-13,
        )
        # df/dv = 1
        np.testing.assert_allclose(
            bspline.evaluate_derivatives(pts, [0, 1]),
            np.ones(3),
            atol=1e-13,
        )

    def test_bilinear_separable_derivative(self) -> None:
        """f(u,v)=u*v (separable): df/du=v, df/dv=u, d²f/dudv=1."""
        s1 = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), 1)
        s2 = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), 1)
        space = BsplineSpace([s1, s2])
        cp = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        bspline = Bspline(space, cp)

        pts = np.array([[0.25, 0.75], [0.5, 0.5]], dtype=np.float64)
        us = pts[:, 0]
        vs = pts[:, 1]

        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [1, 0]), vs, atol=1e-13)
        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [0, 1]), us, atol=1e-13)
        # Mixed derivative d²f/dudv = 1
        np.testing.assert_allclose(
            bspline.evaluate_derivatives(pts, [1, 1]), np.ones(2), atol=1e-13
        )

    def test_finite_difference_2d_scalar(self) -> None:
        """Central-difference validation of first derivatives for 2D scalar spline."""
        s1 = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64), 2)
        s2 = BsplineSpace1D(np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float64), 1)
        space = BsplineSpace([s1, s2])
        rng = np.random.default_rng(42)
        cp = rng.standard_normal(space.num_total_basis).astype(np.float64)
        bspline = Bspline(space, cp)

        h = 1e-5
        pts = np.array([[0.3, 0.4]], dtype=np.float64)
        du = np.array([[h, 0.0]], dtype=np.float64)
        dv = np.array([[0.0, h]], dtype=np.float64)
        fd_u = (bspline.evaluate(pts + du) - bspline.evaluate(pts - du)) / (2 * h)
        fd_v = (bspline.evaluate(pts + dv) - bspline.evaluate(pts - dv)) / (2 * h)

        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [1, 0]), fd_u, atol=1e-9)
        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [0, 1]), fd_v, atol=1e-9)

    def test_finite_difference_2d_vector(self) -> None:
        """Central-difference validation for 2D vector-valued spline."""
        s1 = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64), 2)
        s2 = BsplineSpace1D(np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float64), 1)
        space = BsplineSpace([s1, s2])
        n = space.num_total_basis
        rng = np.random.default_rng(7)
        cp = rng.standard_normal((n, 3)).astype(np.float64)
        bspline = Bspline(space, cp)

        h = 1e-5
        pts = np.array([[0.2, 0.6]], dtype=np.float64)
        du = np.array([[h, 0.0]], dtype=np.float64)
        dv = np.array([[0.0, h]], dtype=np.float64)
        fd_u = (bspline.evaluate(pts + du) - bspline.evaluate(pts - du)) / (2 * h)
        fd_v = (bspline.evaluate(pts + dv) - bspline.evaluate(pts - dv)) / (2 * h)

        # evaluate() squeezes (1, 3) → (3,); derivatives return (1, 3) for n_pts=1
        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [1, 0])[0], fd_u, atol=1e-9)
        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [0, 1])[0], fd_v, atol=1e-9)

    def test_finite_difference_2d_second_mixed(self) -> None:
        """Central-difference validation of ∂²f/∂u∂v for a 2D quadratic spline."""
        s1 = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64), 2)
        s2 = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64), 2)
        space = BsplineSpace([s1, s2])
        rng = np.random.default_rng(99)
        cp = rng.standard_normal(space.num_total_basis).astype(np.float64)
        bspline = Bspline(space, cp)

        h = 1e-4
        pts = np.array([[0.3, 0.4]], dtype=np.float64)

        # 4-point central FD stencil for ∂²f/∂u∂v, truncation error O(h²)
        pp = pts + np.array([[+h, +h]])
        pm = pts + np.array([[+h, -h]])
        mp = pts + np.array([[-h, +h]])
        mm = pts + np.array([[-h, -h]])
        fd = (
            bspline.evaluate(pp)
            - bspline.evaluate(pm)
            - bspline.evaluate(mp)
            + bspline.evaluate(mm)
        ) / (4.0 * h**2)

        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [1, 1]), fd, atol=1e-7)

    def test_lattice_matches_pts_array(self) -> None:
        """Lattice and pts-array evaluation give the same results on a grid."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        rng = np.random.default_rng(99)
        cp = rng.standard_normal(6).astype(np.float64)
        bspline = Bspline(space, cp)

        pts1 = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        pts2 = np.linspace(0.0, 1.0, 4, dtype=np.float64)

        g0, g1 = np.meshgrid(pts1, pts2, indexing="ij")
        pts_arr = np.stack([g0.ravel(), g1.ravel()], axis=1)

        for ords in ([0, 0], [1, 0], [0, 1], [1, 1]):
            result_arr = bspline.evaluate_derivatives(pts_arr, ords)  # (20,)
            result_arr_grid = result_arr.reshape(5, 4)

            lattice = PointsLattice([pts1, pts2])
            result_lat = bspline.evaluate_derivatives(lattice, ords)  # (5, 4)

            np.testing.assert_allclose(result_lat, result_arr_grid, atol=1e-13)

    def test_3d_bspline_derivatives(self) -> None:
        """evaluate_derivatives works for dim=3 B-splines."""
        s1 = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), 1)
        s2 = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), 1)
        s3 = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), 1)
        space = BsplineSpace([s1, s2, s3])
        rng = np.random.default_rng(3)
        cp = rng.standard_normal(space.num_total_basis).astype(np.float64)
        bspline = Bspline(space, cp)

        pts = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)

        # scalar 3D: partial (1,0,1)
        result = bspline.evaluate_derivatives(pts, [1, 0, 1])
        assert result.shape == (1,)

        # zeroth order matches evaluate
        result0 = bspline.evaluate_derivatives(pts, [0, 0, 0])
        np.testing.assert_allclose(result0, bspline.evaluate(pts), atol=1e-13)


class TestMultiDimDerivRational:
    """Verify rational B-spline derivative evaluation."""

    def test_rational_orders_zero_matches_evaluate(self) -> None:
        """orders=[0,0] of rational derivatives matches evaluate()."""
        s1 = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64), 2)
        s2 = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), 1)
        space = BsplineSpace([s1, s2])
        rng = np.random.default_rng(5)
        n = space.num_total_basis
        cp = np.column_stack([rng.standard_normal(n), np.ones(n)]).astype(np.float64)
        bspline = Bspline(space, cp.ravel(), is_rational=True)

        rng2 = np.random.default_rng(6)
        pts = rng2.uniform(0, 1, (10, 2)).astype(np.float64)

        result_d = bspline.evaluate_derivatives(pts, [0, 0])  # (10,)
        result_v = bspline.evaluate(pts)

        np.testing.assert_allclose(result_d, result_v, atol=1e-13)

    def test_rational_unit_weights_matches_non_rational(self) -> None:
        """Rational with unit weights gives same first derivatives as non-rational."""
        s1 = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64), 2)
        s2 = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), 1)
        space = BsplineSpace([s1, s2])
        rng = np.random.default_rng(10)
        n = space.num_total_basis
        values = rng.standard_normal(n).astype(np.float64)

        bspline_nr = Bspline(space, values)
        cp_rational = np.column_stack([values, np.ones(n)]).astype(np.float64)
        bspline_r = Bspline(space, cp_rational.ravel(), is_rational=True)

        pts = np.array([[0.3, 0.7], [0.6, 0.2]], dtype=np.float64)
        for ords in ([0, 0], [1, 0], [0, 1]):
            result_nr = bspline_nr.evaluate_derivatives(pts, ords)
            result_r = bspline_r.evaluate_derivatives(pts, ords)
            np.testing.assert_allclose(result_r, result_nr, atol=1e-11)

    def test_rational_finite_difference(self) -> None:
        """Central-difference validation of rational 2D first derivative."""
        s1 = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64), 2)
        s2 = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), 1)
        space = BsplineSpace([s1, s2])
        n = space.num_total_basis
        rng = np.random.default_rng(20)
        values = rng.standard_normal(n).astype(np.float64)
        weights = rng.uniform(0.5, 1.5, n).astype(np.float64)
        cp = np.column_stack([values * weights, weights]).astype(np.float64)
        bspline = Bspline(space, cp.ravel(), is_rational=True)

        h = 1e-5
        pts = np.array([[0.4, 0.6]], dtype=np.float64)
        du = np.array([[h, 0.0]], dtype=np.float64)
        dv = np.array([[0.0, h]], dtype=np.float64)
        fd_u = (bspline.evaluate(pts + du) - bspline.evaluate(pts - du)) / (2 * h)
        fd_v = (bspline.evaluate(pts + dv) - bspline.evaluate(pts - dv)) / (2 * h)

        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [1, 0]), fd_u, atol=1e-9)
        np.testing.assert_allclose(bspline.evaluate_derivatives(pts, [0, 1]), fd_v, atol=1e-9)


class TestMultiDimDerivValidation:
    """Test input validation for multi-dimensional derivatives."""

    def test_wrong_orders_length(self) -> None:
        """len(orders) != dim raises ValueError."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.ones(6, dtype=np.float64)
        bspline = Bspline(space, cp)
        pts = np.array([[0.5, 0.5]], dtype=np.float64)

        with pytest.raises(ValueError, match="len\\(orders\\)"):
            bspline.evaluate_derivatives(pts, [1])

    def test_negative_order(self) -> None:
        """Negative order raises ValueError."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.ones(6, dtype=np.float64)
        bspline = Bspline(space, cp)
        pts = np.array([[0.5, 0.5]], dtype=np.float64)

        with pytest.raises(ValueError, match="orders\\["):
            bspline.evaluate_derivatives(pts, [-1, 0])

    def test_wrong_pts_shape(self) -> None:
        """Pts array with wrong number of columns raises ValueError."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.ones(6, dtype=np.float64)
        bspline = Bspline(space, cp)
        pts = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)  # 3 cols, need 2

        with pytest.raises(ValueError):
            bspline.evaluate_derivatives(pts, [1, 0])

    def test_wrong_pts_dtype(self) -> None:
        """Pts array with wrong dtype raises ValueError."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.ones(6, dtype=np.float64)
        bspline = Bspline(space, cp)
        pts = np.array([[0.5, 0.5]], dtype=np.float32)

        with pytest.raises(ValueError):
            bspline.evaluate_derivatives(pts, [1, 0])

    def test_wrong_lattice_dim(self) -> None:
        """PointsLattice with wrong dimension raises ValueError."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.ones(6, dtype=np.float64)
        bspline = Bspline(space, cp)
        lattice = PointsLattice([np.linspace(0, 1, 5, dtype=np.float64)])  # 1D lattice

        with pytest.raises(ValueError):
            bspline.evaluate_derivatives(lattice, [1, 0])

    def test_out_array_reuse_scalar(self) -> None:
        """Pre-allocated out (scalar, shape (n_pts,)) is filled in-place."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.ones(6, dtype=np.float64)
        bspline = Bspline(space, cp)
        pts = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float64)

        out = np.zeros(2, dtype=np.float64)
        result = bspline.evaluate_derivatives(pts, [1, 0], out=out)

        np.testing.assert_array_equal(result, out)
        assert result is out

    def test_out_array_reuse_vector(self) -> None:
        """Pre-allocated out (vector, shape (n_pts, rank)) is filled in-place."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.arange(12, dtype=np.float64)
        bspline = Bspline(space, cp)
        pts = np.array([[0.5, 0.5]], dtype=np.float64)

        out = np.zeros((1, 2), dtype=np.float64)
        result = bspline.evaluate_derivatives(pts, [1, 0], out=out)

        np.testing.assert_array_equal(result, out)
        assert result is out


# ---------------------------------------------------------------------------
# Periodic multi-dimensional derivative evaluation
# ---------------------------------------------------------------------------


class TestPeriodicMultiDimDerivEvaluation:
    """Test multi-dimensional evaluate_derivatives with periodic directions.

    Direct comparison of ``f.evaluate_derivatives()`` vs ``f_open.evaluate_derivatives()``
    is only performed for reduced-continuity (C^0, C^1) periodic directions.
    Max-continuity periodic correctness is covered via ``to_open_bspline()`` constant-field
    tests in test_bspline_evaluate_multi_dim.py.
    """

    @pytest.mark.parametrize(
        "orders",
        [
            [0, 0],
            [1, 0],
            [0, 1],
        ],
    )
    def test_2d_one_periodic_C0_derivatives_match_open(self, orders: list[int]) -> None:
        """2-D (periodic C^0 x open) evaluate_derivatives agrees with open form."""
        knots_per = create_uniform_periodic(4, 2, continuity=0, dtype=np.float64)
        knots_open = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        s_per = BsplineSpace1D(knots_per, 2, periodic=True)
        s_open = BsplineSpace1D(knots_open, 2)
        space = BsplineSpace([s_per, s_open])
        n = space.num_total_basis
        ctrl = np.arange(1.0, n + 1.0, dtype=np.float64)
        f = Bspline(space, ctrl)
        f_open = f.to_open_bspline()

        a0, b0 = f_open.space.spaces[0].domain
        a1, b1 = f_open.space.spaces[1].domain
        us = np.linspace(float(a0), float(b0), 6, dtype=np.float64)[1:-1]
        vs = np.linspace(float(a1), float(b1), 6, dtype=np.float64)[1:-1]
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        pts = np.column_stack([uu.ravel(), vv.ravel()])

        np.testing.assert_allclose(
            f.evaluate_derivatives(pts, orders),
            f_open.evaluate_derivatives(pts, orders),
            atol=1e-9,
        )

    @pytest.mark.parametrize(
        "orders",
        [
            [0, 0],
            [1, 0],
            [0, 1],
        ],
    )
    def test_2d_both_periodic_C0_derivatives_match_open(self, orders: list[int]) -> None:
        """2-D (periodic C^0 x periodic C^0) evaluate_derivatives agrees with open form."""
        knots0 = create_uniform_periodic(4, 2, continuity=0, dtype=np.float64)
        knots1 = create_uniform_periodic(4, 2, continuity=0, dtype=np.float64)
        s0 = BsplineSpace1D(knots0, 2, periodic=True)
        s1 = BsplineSpace1D(knots1, 2, periodic=True)
        space = BsplineSpace([s0, s1])
        n = space.num_total_basis
        ctrl = np.arange(1.0, n + 1.0, dtype=np.float64)
        f = Bspline(space, ctrl)
        f_open = f.to_open_bspline()

        a0, b0 = f_open.space.spaces[0].domain
        a1, b1 = f_open.space.spaces[1].domain
        us = np.linspace(float(a0), float(b0), 5, dtype=np.float64)[1:-1]
        vs = np.linspace(float(a1), float(b1), 5, dtype=np.float64)[1:-1]
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        pts = np.column_stack([uu.ravel(), vv.ravel()])

        np.testing.assert_allclose(
            f.evaluate_derivatives(pts, orders),
            f_open.evaluate_derivatives(pts, orders),
            atol=1e-9,
        )

    def test_2d_periodic_degree3_C1_derivatives_match_open(self) -> None:
        """2-D (periodic degree-3 C^1 x open) evaluate_derivatives agrees with open form."""
        knots_per = create_uniform_periodic(5, 3, continuity=1, dtype=np.float64)
        knots_open = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        s_per = BsplineSpace1D(knots_per, 3, periodic=True)
        s_open = BsplineSpace1D(knots_open, 2)
        space = BsplineSpace([s_per, s_open])
        n = space.num_total_basis
        ctrl = np.arange(1.0, n + 1.0, dtype=np.float64)
        f = Bspline(space, ctrl)
        f_open = f.to_open_bspline()

        a0, b0 = f_open.space.spaces[0].domain
        a1, b1 = f_open.space.spaces[1].domain
        us = np.linspace(float(a0), float(b0), 6, dtype=np.float64)[1:-1]
        vs = np.linspace(float(a1), float(b1), 6, dtype=np.float64)[1:-1]
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        pts = np.column_stack([uu.ravel(), vv.ravel()])

        for orders in [[0, 0], [1, 0], [0, 1]]:
            np.testing.assert_allclose(
                f.evaluate_derivatives(pts, orders),
                f_open.evaluate_derivatives(pts, orders),
                atol=1e-9,
                err_msg=f"Mismatch at orders={orders}",
            )
