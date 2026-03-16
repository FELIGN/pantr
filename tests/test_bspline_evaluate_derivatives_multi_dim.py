"""Tests for multi-dimensional B-spline derivative evaluation."""

import numpy as np
import pytest

from pantr.bspline import Bspline
from pantr.bspline_space_1D import BsplineSpace1D
from pantr.bspline_space_nd import BsplineSpace
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

    def test_pts_array_scalar_non_rational(self) -> None:
        """Shape (n_pts, n_deriv+1, dim) for scalar non-rational."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.ones(6, dtype=np.float64)
        bspline = Bspline(space, cp)

        pts = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=np.float64)
        result = bspline.evaluate_derivatives(pts, n_deriv=2)

        assert result.shape == (2, 3, 2)

    def test_pts_array_vector_non_rational(self) -> None:
        """Shape (n_pts, n_deriv+1, dim, rank) for vector non-rational."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        # 6 basis x 3 columns → rank 3
        cp = np.arange(18, dtype=np.float64)
        bspline = Bspline(space, cp)

        pts = np.array([[0.5, 0.5]], dtype=np.float64)
        result = bspline.evaluate_derivatives(pts, n_deriv=1)

        assert result.shape == (1, 2, 2, 3)

    def test_lattice_scalar_non_rational(self) -> None:
        """Shape (*pts_grid_shape, n_deriv+1, dim) for lattice scalar non-rational."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.ones(6, dtype=np.float64)
        bspline = Bspline(space, cp)

        pts1 = np.linspace(0.0, 1.0, 4, dtype=np.float64)
        pts2 = np.linspace(0.0, 1.0, 3, dtype=np.float64)
        lattice = PointsLattice([pts1, pts2])
        result = bspline.evaluate_derivatives(lattice, n_deriv=1)

        assert result.shape == (4, 3, 2, 2)

    def test_pts_array_rational_scalar(self) -> None:
        """Shape (n_pts, n_deriv+1, dim) for rational scalar (rank_value=1)."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        # 6 basis x 2 columns → cp_size=2, rational rank_value=1
        cp = np.tile([1.0, 1.0], (6, 1)).astype(np.float64)
        bspline = Bspline(space, cp.ravel(), is_rational=True)

        pts = np.array([[0.5, 0.5]], dtype=np.float64)
        result = bspline.evaluate_derivatives(pts, n_deriv=1)

        assert result.shape == (1, 2, 2)

    def test_pts_array_rational_vector(self) -> None:
        """Shape (n_pts, n_deriv+1, dim, rank_value) for rational vector."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        # 6 basis x 3 columns → cp_size=3, rational rank_value=2
        cp = np.tile([1.0, 0.0, 1.0], (6, 1)).astype(np.float64)
        bspline = Bspline(space, cp.ravel(), is_rational=True)

        pts = np.array([[0.5, 0.5]], dtype=np.float64)
        result = bspline.evaluate_derivatives(pts, n_deriv=1)

        assert result.shape == (1, 2, 2, 2)

    def test_n_deriv_0_shape(self) -> None:
        """n_deriv=0 produces shape (*pts_shape, 1, dim)."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.ones(6, dtype=np.float64)
        bspline = Bspline(space, cp)

        pts = np.array([[0.5, 0.5]], dtype=np.float64)
        result = bspline.evaluate_derivatives(pts, n_deriv=0)

        assert result.shape == (1, 1, 2)


class TestMultiDimDerivCorrectness:
    """Verify correctness of multi-dimensional derivative values."""

    def test_n_deriv_0_matches_evaluate_scalar(self) -> None:
        """k=0 slice of derivative output must equal evaluate() for scalar."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        bspline = Bspline(space, cp)

        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 1, (20, 2)).astype(np.float64)

        result_d = bspline.evaluate_derivatives(pts, n_deriv=0)  # (20, 1, 2)
        result_v = bspline.evaluate(pts)  # (20,)

        # All directions must give the same value (k=0 is the function value)
        np.testing.assert_allclose(result_d[:, 0, 0], result_v, atol=1e-13)
        np.testing.assert_allclose(result_d[:, 0, 1], result_v, atol=1e-13)

    def test_n_deriv_0_matches_evaluate_vector(self) -> None:
        """k=0 slice matches evaluate() for vector-valued spline."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.arange(12, dtype=np.float64)
        bspline = Bspline(space, cp)

        rng = np.random.default_rng(1)
        pts = rng.uniform(0, 1, (15, 2)).astype(np.float64)

        result_d = bspline.evaluate_derivatives(pts, n_deriv=0)  # (15, 1, 2, 2)
        result_v = bspline.evaluate(pts)  # (15, 2)

        np.testing.assert_allclose(result_d[:, 0, 0, :], result_v, atol=1e-13)
        np.testing.assert_allclose(result_d[:, 0, 1, :], result_v, atol=1e-13)

    def test_linear_spline_constant_first_derivative(self) -> None:
        """Degree-(1,1) bilinear spline f(u,v)=u+v: df/du=1, df/dv=1 everywhere."""
        # f(u, v) = u + v encoded as linear CPs: (0,0)->0, (1,0)->1, (0,1)->1, (1,1)->2
        # Knots: degree 1 in each direction
        s1 = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), 1)
        s2 = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), 1)
        space = BsplineSpace([s1, s2])
        # CP layout (num_basis_0=2, num_basis_1=2): [f(0,0), f(0,1), f(1,0), f(1,1)]
        cp = np.array([0.0, 1.0, 1.0, 2.0], dtype=np.float64)
        bspline = Bspline(space, cp)

        # Exclude the domain endpoints: the kernel's endpoint shortcut only fills
        # the zeroth-derivative slot; higher-order derivatives are left zero there.
        pts = np.array([[0.0, 0.0], [0.5, 0.5], [0.75, 0.25]], dtype=np.float64)
        result = bspline.evaluate_derivatives(pts, n_deriv=1)
        # shape (3, 2, 2): [pts, k=0..1, dir=0..1]

        # k=0 should be f(u,v) = u + v
        np.testing.assert_allclose(result[:, 0, 0], [0.0, 1.0, 1.0], atol=1e-13)
        # k=1, dir=0: df/du = 1.0
        np.testing.assert_allclose(result[:, 1, 0], [1.0, 1.0, 1.0], atol=1e-13)
        # k=1, dir=1: df/dv = 1.0
        np.testing.assert_allclose(result[:, 1, 1], [1.0, 1.0, 1.0], atol=1e-13)

    def test_bilinear_separable_derivative(self) -> None:
        """f(u,v)=u*v (separable): df/du=v, df/dv=u."""
        s1 = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), 1)
        s2 = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), 1)
        space = BsplineSpace([s1, s2])
        # CP layout: f(0,0)=0, f(0,1)=0, f(1,0)=0, f(1,1)=1
        cp = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        bspline = Bspline(space, cp)

        # Exclude domain endpoints (derivative not computed there).
        pts = np.array([[0.25, 0.75], [0.5, 0.5]], dtype=np.float64)
        result = bspline.evaluate_derivatives(pts, n_deriv=1)

        us = pts[:, 0]
        vs = pts[:, 1]
        # k=1, dir=0: df/du = v
        np.testing.assert_allclose(result[:, 1, 0], vs, atol=1e-13)
        # k=1, dir=1: df/dv = u
        np.testing.assert_allclose(result[:, 1, 1], us, atol=1e-13)

    def test_finite_difference_2d_scalar(self) -> None:
        """Finite-difference validation of first derivatives for 2D scalar spline."""
        s1 = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64), 2)
        s2 = BsplineSpace1D(np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float64), 1)
        space = BsplineSpace([s1, s2])
        rng = np.random.default_rng(42)
        cp = rng.standard_normal(space.num_total_basis).astype(np.float64)
        bspline = Bspline(space, cp)

        h = 1e-5
        pts = np.array([[0.3, 0.4]], dtype=np.float64)
        pts_du = np.array([[0.3 + h, 0.4]], dtype=np.float64)
        pts_dv = np.array([[0.3, 0.4 + h]], dtype=np.float64)

        result = bspline.evaluate_derivatives(pts, n_deriv=1)
        fd_u = (bspline.evaluate(pts_du) - bspline.evaluate(pts)) / h
        fd_v = (bspline.evaluate(pts_dv) - bspline.evaluate(pts)) / h

        np.testing.assert_allclose(result[0, 1, 0], fd_u, rtol=1e-3)
        np.testing.assert_allclose(result[0, 1, 1], fd_v, rtol=1e-3)

    def test_finite_difference_2d_vector(self) -> None:
        """Finite-difference validation for 2D vector-valued spline."""
        s1 = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64), 2)
        s2 = BsplineSpace1D(np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float64), 1)
        space = BsplineSpace([s1, s2])
        n = space.num_total_basis
        rng = np.random.default_rng(7)
        cp = rng.standard_normal((n, 3)).astype(np.float64)
        bspline = Bspline(space, cp)

        h = 1e-5
        pts = np.array([[0.2, 0.6]], dtype=np.float64)
        pts_du = pts + np.array([[h, 0.0]])
        pts_dv = pts + np.array([[0.0, h]])

        result = bspline.evaluate_derivatives(pts, n_deriv=1)
        fd_u = (bspline.evaluate(pts_du) - bspline.evaluate(pts)) / h
        fd_v = (bspline.evaluate(pts_dv) - bspline.evaluate(pts)) / h

        np.testing.assert_allclose(result[0, 1, 0, :], fd_u, atol=1e-4)
        np.testing.assert_allclose(result[0, 1, 1, :], fd_v, atol=1e-4)

    def test_lattice_matches_pts_array(self) -> None:
        """Lattice and pts-array evaluation give the same results on a grid."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        rng = np.random.default_rng(99)
        cp = rng.standard_normal(6).astype(np.float64)
        bspline = Bspline(space, cp)

        pts1 = np.linspace(0.0, 1.0, 5, dtype=np.float64)
        pts2 = np.linspace(0.0, 1.0, 4, dtype=np.float64)

        # Build pts array from grid
        g0, g1 = np.meshgrid(pts1, pts2, indexing="ij")
        pts_arr = np.stack([g0.ravel(), g1.ravel()], axis=1)

        result_arr = bspline.evaluate_derivatives(pts_arr, n_deriv=1)
        # shape (20, 2, 2) → reshape to (5, 4, 2, 2)
        result_arr_grid = result_arr.reshape(5, 4, 2, 2)

        lattice = PointsLattice([pts1, pts2])
        result_lat = bspline.evaluate_derivatives(lattice, n_deriv=1)
        # shape (5, 4, 2, 2)

        np.testing.assert_allclose(result_lat, result_arr_grid, atol=1e-13)


class TestMultiDimDerivRational:
    """Verify rational B-spline derivative evaluation."""

    def test_rational_n_deriv_0_matches_evaluate(self) -> None:
        """k=0 slice of rational derivatives matches evaluate()."""
        s1 = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64), 2)
        s2 = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64), 1)
        space = BsplineSpace([s1, s2])
        rng = np.random.default_rng(5)
        # cp_size=2: [numerator, weight], all weights=1 so rational == non-rational
        n = space.num_total_basis
        cp = np.column_stack([rng.standard_normal(n), np.ones(n)]).astype(np.float64)
        bspline = Bspline(space, cp.ravel(), is_rational=True)

        rng2 = np.random.default_rng(6)
        pts = rng2.uniform(0, 1, (10, 2)).astype(np.float64)

        result_d = bspline.evaluate_derivatives(pts, n_deriv=0)  # (10, 1, 2)
        result_v = bspline.evaluate(pts)  # (10,) scalar because rank_value=1

        np.testing.assert_allclose(result_d[:, 0, 0], result_v, atol=1e-13)
        np.testing.assert_allclose(result_d[:, 0, 1], result_v, atol=1e-13)

    def test_rational_unit_weights_matches_non_rational(self) -> None:
        """Rational with unit weights gives same derivatives as non-rational."""
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
        result_nr = bspline_nr.evaluate_derivatives(pts, n_deriv=2)
        result_r = bspline_r.evaluate_derivatives(pts, n_deriv=2)

        np.testing.assert_allclose(result_r, result_nr, atol=1e-11)

    def test_rational_finite_difference(self) -> None:
        """Finite-difference validation of rational 2D derivative."""
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
        pts_du = pts + np.array([[h, 0.0]])
        pts_dv = pts + np.array([[0.0, h]])

        result = bspline.evaluate_derivatives(pts, n_deriv=1)  # (1, 2, 2)
        fd_u = (bspline.evaluate(pts_du) - bspline.evaluate(pts)) / h
        fd_v = (bspline.evaluate(pts_dv) - bspline.evaluate(pts)) / h

        np.testing.assert_allclose(result[0, 1, 0], fd_u, atol=1e-5)
        np.testing.assert_allclose(result[0, 1, 1], fd_v, atol=1e-5)


class TestMultiDimDerivValidation:
    """Test input validation for multi-dimensional derivatives."""

    def test_invalid_n_deriv(self) -> None:
        """n_deriv < 0 raises ValueError."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.ones(6, dtype=np.float64)
        bspline = Bspline(space, cp)
        pts = np.array([[0.5, 0.5]], dtype=np.float64)

        with pytest.raises(ValueError, match="n_deriv"):
            bspline.evaluate_derivatives(pts, n_deriv=-1)

    def test_wrong_pts_shape(self) -> None:
        """Pts array with wrong number of columns raises ValueError."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.ones(6, dtype=np.float64)
        bspline = Bspline(space, cp)
        pts = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)  # 3 cols, need 2

        with pytest.raises(ValueError):
            bspline.evaluate_derivatives(pts, n_deriv=1)

    def test_wrong_pts_dtype(self) -> None:
        """Pts array with wrong dtype raises ValueError."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.ones(6, dtype=np.float64)
        bspline = Bspline(space, cp)
        pts = np.array([[0.5, 0.5]], dtype=np.float32)

        with pytest.raises(ValueError):
            bspline.evaluate_derivatives(pts, n_deriv=1)

    def test_wrong_lattice_dim(self) -> None:
        """PointsLattice with wrong dimension raises ValueError."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.ones(6, dtype=np.float64)
        bspline = Bspline(space, cp)
        lattice = PointsLattice([np.linspace(0, 1, 5, dtype=np.float64)])  # 1D lattice

        with pytest.raises(ValueError):
            bspline.evaluate_derivatives(lattice, n_deriv=1)

    def test_out_array_reuse(self) -> None:
        """Pre-allocated out array is filled in-place and result is a view of it."""
        space = _make_2d_space([0, 0, 0, 1, 1, 1], 2, [0, 0, 1, 1], 1)
        cp = np.ones(6, dtype=np.float64)
        bspline = Bspline(space, cp)
        pts = np.array([[0.5, 0.5]], dtype=np.float64)

        # For scalar non-rational, out must include the trailing cp_size=1 dimension.
        out = np.zeros((1, 2, 2, 1), dtype=np.float64)
        result = bspline.evaluate_derivatives(pts, n_deriv=1, out=out)

        # result is out[..., 0] (squeezed view)
        np.testing.assert_array_equal(result, out[..., 0])
        assert not np.all(out == 0.0)

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
        result = bspline.evaluate_derivatives(pts, n_deriv=1)

        # scalar 3D: (1, 2, 3)
        assert result.shape == (1, 2, 3)
