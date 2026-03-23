"""Tests for multi-dimensional B-spline evaluation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D, create_uniform_periodic
from pantr.quad import PointsLattice

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bilinear_space(dtype: type = float) -> BsplineSpace:
    """Return a degree-(1,1) space on [0,1]^2."""
    knots: npt.NDArray[np.float32 | np.float64] = np.array([0.0, 0.0, 1.0, 1.0], dtype=dtype)
    s = BsplineSpace1D(knots, 1)
    return BsplineSpace([s, s])


def _biquadratic_space(dtype: type = float) -> BsplineSpace:
    """Return a degree-(2,2) space on [0,1]^2."""
    knots: npt.NDArray[np.float32 | np.float64] = np.array(
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=dtype
    )
    s = BsplineSpace1D(knots, 2)
    return BsplineSpace([s, s])


def _make_2d(space: BsplineSpace, cp_array: npt.NDArray[np.float32 | np.float64]) -> Bspline:
    """Wrap space + control-point array into a Bspline."""
    return Bspline(space, cp_array)


# ---------------------------------------------------------------------------
# Correctness: points array
# ---------------------------------------------------------------------------


class TestEvaluateMultiDimPointsArray:
    """Test multi-dim evaluation with a 2-D points array (n_pts, dim)."""

    def test_constant_field(self) -> None:
        """All-ones control points give f(u,v) = 1 everywhere."""
        space = _bilinear_space()
        bsp = _make_2d(space, np.ones(4, dtype=np.float64))
        pts = np.array([[0.0, 0.0], [0.3, 0.7], [1.0, 1.0]], dtype=np.float64)
        result = bsp.evaluate(pts)
        np.testing.assert_allclose(result, 1.0, atol=1e-14)

    def test_bilinear_product_uv(self) -> None:
        """Bilinear B-spline with cp[1,1]=1, rest=0 gives f(u,v)=u*v."""
        space = _bilinear_space()
        cp = np.zeros((2, 2, 1), dtype=np.float64)
        cp[1, 1, 0] = 1.0
        bsp = Bspline(space, cp)
        pts = np.array([[0.3, 0.4], [0.5, 0.7], [0.8, 0.2]], dtype=np.float64)
        result = bsp.evaluate(pts)
        np.testing.assert_allclose(result, pts[:, 0] * pts[:, 1], atol=1e-14)

    def test_biquadratic_u_squared(self) -> None:
        """Biquadratic B-spline encoding f(u,v) = u^2 (constant in v)."""
        # Bernstein quadratic CPs [0,0,1] -> f(u)=u^2
        # f(u,v) = u^2 * 1  -> cp[i,j] = cp_u[i] * 1 for all j
        knots2 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        knots1 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        s_quad = BsplineSpace1D(knots2, 2)
        s_lin = BsplineSpace1D(knots1, 1)
        space = BsplineSpace([s_quad, s_lin])
        # cp_u = [0, 0, 1], cp_v = [1, 1]
        # cp[i, j] = cp_u[i] (constant in v)
        cp = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        bsp = Bspline(space, cp)
        pts = np.array([[0.0, 0.5], [0.5, 0.3], [0.9, 0.8]], dtype=np.float64)
        result = bsp.evaluate(pts)
        np.testing.assert_allclose(result, pts[:, 0] ** 2, atol=1e-13)

    def test_biquadratic_product_u2_v(self) -> None:
        """Biquadratic x linear encoding f(u,v)=u^2*v."""
        knots_quad = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots_lin = [0.0, 0.0, 1.0, 1.0]
        s_quad = BsplineSpace1D(np.array(knots_quad, dtype=np.float64), 2)
        s_lin = BsplineSpace1D(np.array(knots_lin, dtype=np.float64), 1)
        space = BsplineSpace([s_quad, s_lin])
        # cp[i,j] = cp_u[i] * cp_v[j], cp_u=[0,0,1], cp_v=[0,1]
        cp = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        bsp = Bspline(space, cp)
        pts = np.array([[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]], dtype=np.float64)
        result = bsp.evaluate(pts)
        np.testing.assert_allclose(result, pts[:, 0] ** 2 * pts[:, 1], atol=1e-13)

    def test_vector_valued(self) -> None:
        """2D space with 2-column control points produces (n_pts, 2) output."""
        space = _bilinear_space()
        # x-component: u*v, y-component: 1-u*v
        cp = np.zeros((2, 2, 2), dtype=np.float64)
        cp[1, 1, 0] = 1.0  # x-component
        cp[0, 0, 1] = 1.0  # y-component contribution: (1-u)*(1-v)
        bsp = Bspline(space, cp)
        pts = np.array([[0.5, 0.5], [0.2, 0.8]], dtype=np.float64)
        result = bsp.evaluate(pts)
        assert result.shape == (2, 2)

    def test_3d_space(self) -> None:
        """3D trilinear B-spline with cp[1,1,1]=1 gives f(u,v,w)=u*v*w."""
        knots = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        s = BsplineSpace1D(knots, 1)
        space = BsplineSpace([s, s, s])
        cp = np.zeros((2, 2, 2, 1), dtype=np.float64)
        cp[1, 1, 1, 0] = 1.0
        bsp = Bspline(space, cp)
        pts = np.array([[0.3, 0.4, 0.5], [0.7, 0.2, 0.9]], dtype=np.float64)
        result = bsp.evaluate(pts)
        np.testing.assert_allclose(result, pts[:, 0] * pts[:, 1] * pts[:, 2], atol=1e-13)

    def test_float32(self) -> None:
        """Float32 control points and points produce float32 output."""
        space = _bilinear_space(dtype=np.float32)
        cp = np.ones(4, dtype=np.float32)
        bsp = Bspline(space, cp)
        pts = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float32)
        result = bsp.evaluate(pts)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Correctness: PointsLattice
# ---------------------------------------------------------------------------


class TestEvaluateMultiDimLattice:
    """Test multi-dim evaluation with a PointsLattice."""

    def test_constant_field_lattice(self) -> None:
        """All-ones CPs give f=1 on every lattice point."""
        space = _bilinear_space()
        bsp = _make_2d(space, np.ones(4, dtype=np.float64))
        lattice = PointsLattice([np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 4)])
        result = bsp.evaluate(lattice)
        assert result.shape == (5, 4)
        np.testing.assert_allclose(result, 1.0, atol=1e-14)

    def test_bilinear_product_uv_lattice(self) -> None:
        """Lattice evaluation of f(u,v)=u*v agrees with meshgrid."""
        space = _bilinear_space()
        cp = np.zeros((2, 2, 1), dtype=np.float64)
        cp[1, 1, 0] = 1.0
        bsp = Bspline(space, cp)
        u_pts = np.linspace(0.0, 1.0, 6)
        v_pts = np.linspace(0.0, 1.0, 5)
        lattice = PointsLattice([u_pts, v_pts])
        result = bsp.evaluate(lattice)
        U, V = np.meshgrid(u_pts, v_pts, indexing="ij")
        np.testing.assert_allclose(result, U * V, atol=1e-14)

    def test_biquadratic_lattice(self) -> None:
        """Biquadratic lattice evaluation of f(u,v)=u^2*v agrees with exact formula."""
        knots_quad = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
        knots_lin = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        s_quad = BsplineSpace1D(knots_quad, 2)
        s_lin = BsplineSpace1D(knots_lin, 1)
        space = BsplineSpace([s_quad, s_lin])
        cp = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        bsp = Bspline(space, cp)
        u_pts = np.linspace(0.0, 1.0, 7)
        v_pts = np.linspace(0.0, 1.0, 5)
        lattice = PointsLattice([u_pts, v_pts])
        result = bsp.evaluate(lattice)
        U, V = np.meshgrid(u_pts, v_pts, indexing="ij")
        np.testing.assert_allclose(result, U**2 * V, atol=1e-13)

    def test_lattice_output_shape_3d(self) -> None:
        """3D lattice evaluation produces the correct grid shape."""
        knots = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        s = BsplineSpace1D(knots, 1)
        space = BsplineSpace([s, s, s])
        cp = np.ones((2, 2, 2, 1), dtype=np.float64)
        bsp = Bspline(space, cp)
        lattice = PointsLattice(
            [np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 5)]
        )
        result = bsp.evaluate(lattice)
        assert result.shape == (3, 4, 5)
        np.testing.assert_allclose(result, 1.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Consistency: points array vs PointsLattice
# ---------------------------------------------------------------------------


class TestEvaluateMultiDimConsistency:
    """Points-array and PointsLattice evaluations must agree on grid points."""

    def test_array_vs_lattice(self) -> None:
        """Evaluate on a grid via both APIs; results must match."""
        space = _biquadratic_space()
        rng = np.random.default_rng(0)
        cp = rng.standard_normal((3, 3, 2)).astype(np.float64)
        bsp = Bspline(space, cp)

        u_pts = np.array([0.1, 0.5, 0.9], dtype=np.float64)
        v_pts = np.array([0.2, 0.6], dtype=np.float64)

        lattice = PointsLattice([u_pts, v_pts])
        result_lattice = bsp.evaluate(lattice)  # shape (3, 2, 2)

        # Build corresponding points array
        U, V = np.meshgrid(u_pts, v_pts, indexing="ij")
        pts_arr = np.stack([U.ravel(), V.ravel()], axis=1)
        result_arr = bsp.evaluate(pts_arr)  # shape (6, 2)

        np.testing.assert_allclose(result_lattice.reshape(-1, 2), result_arr, atol=1e-13)

    def test_single_point_both_apis(self) -> None:
        """Single evaluation point: PointsLattice and array give identical result."""
        space = _bilinear_space()
        cp = np.zeros((2, 2, 1), dtype=np.float64)
        cp[1, 1, 0] = 1.0
        bsp = Bspline(space, cp)

        u_val = np.array([0.6], dtype=np.float64)
        v_val = np.array([0.4], dtype=np.float64)

        result_lattice = bsp.evaluate(PointsLattice([u_val, v_val]))  # shape (1,1) -> scalar
        result_arr = bsp.evaluate(np.array([[0.6, 0.4]], dtype=np.float64))

        np.testing.assert_allclose(float(result_lattice), float(result_arr), atol=1e-14)


# ---------------------------------------------------------------------------
# out parameter
# ---------------------------------------------------------------------------


class TestEvaluateMultiDimOut:
    """Test the optional pre-allocated output buffer."""

    def test_out_array_filled_inplace_points(self) -> None:
        """Pre-allocated out is filled in-place for points array input."""
        space = _bilinear_space()
        bsp = _make_2d(space, np.ones(4, dtype=np.float64))
        pts = np.array([[0.2, 0.3], [0.7, 0.8]], dtype=np.float64)
        out = np.zeros((2, 1), dtype=np.float64)
        result = bsp.evaluate(pts, out=out)
        np.testing.assert_array_equal(result, out[:, 0])
        np.testing.assert_allclose(out[:, 0], 1.0, atol=1e-14)

    def test_out_array_filled_inplace_lattice(self) -> None:
        """Pre-allocated out is filled in-place for PointsLattice input."""
        space = _bilinear_space()
        bsp = _make_2d(space, np.ones(4, dtype=np.float64))
        lattice = PointsLattice([np.array([0.2, 0.8]), np.array([0.3, 0.7])])
        out = np.zeros((2, 2, 1), dtype=np.float64)
        bsp.evaluate(lattice, out=out)
        np.testing.assert_allclose(out[..., 0], 1.0, atol=1e-14)

    def test_out_wrong_shape_raises(self) -> None:
        """Providing out with wrong shape raises ValueError."""
        space = _bilinear_space()
        bsp = _make_2d(space, np.ones(4, dtype=np.float64))
        pts = np.array([[0.5, 0.5]], dtype=np.float64)
        out = np.zeros((2, 1), dtype=np.float64)  # wrong n_pts
        with pytest.raises(ValueError):
            bsp.evaluate(pts, out=out)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestEvaluateMultiDimErrors:
    """Error-handling for multi-dim evaluate."""

    def test_wrong_dim_lattice(self) -> None:
        """PointsLattice dimension mismatch raises ValueError."""
        space = _bilinear_space()
        bsp = _make_2d(space, np.ones(4, dtype=np.float64))
        lattice = PointsLattice([np.array([0.5], dtype=np.float64)])  # 1-D lattice
        with pytest.raises(ValueError, match="dimension"):
            bsp.evaluate(lattice)

    def test_wrong_dim_array(self) -> None:
        """Points array with wrong number of columns raises ValueError."""
        space = _bilinear_space()
        bsp = _make_2d(space, np.ones(4, dtype=np.float64))
        pts = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)  # 3 columns for 2D space
        with pytest.raises(ValueError, match="2 columns"):
            bsp.evaluate(pts)

    def test_wrong_dtype_array(self) -> None:
        """Points dtype mismatch raises ValueError."""
        space = _bilinear_space()
        bsp = _make_2d(space, np.ones(4, dtype=np.float64))
        pts = np.array([[0.5, 0.5]], dtype=np.float32)  # float32 vs float64
        with pytest.raises(ValueError, match="dtype"):
            bsp.evaluate(pts)

    def test_wrong_dtype_lattice(self) -> None:
        """PointsLattice dtype mismatch raises ValueError."""
        space = _bilinear_space()
        bsp = _make_2d(space, np.ones(4, dtype=np.float64))
        lattice = PointsLattice([np.array([0.5], dtype=np.float32)] * 2)
        with pytest.raises(ValueError, match="dtype"):
            bsp.evaluate(lattice)

    def test_point_outside_domain_raises(self) -> None:
        """Point outside the knot domain raises ValueError."""
        space = _bilinear_space()
        bsp = _make_2d(space, np.ones(4, dtype=np.float64))
        pts = np.array([[1.5, 0.5]], dtype=np.float64)  # u=1.5 outside [0,1]
        with pytest.raises(ValueError):
            bsp.evaluate(pts)


# ---------------------------------------------------------------------------
# Periodic multi-dimensional evaluation
# ---------------------------------------------------------------------------


class TestPeriodicMultiDimEvaluation:
    """Test multi-dimensional evaluation with one or both periodic directions."""

    @staticmethod
    def _make_periodic_open_2d(
        n_per: int,
        deg_per: int,
        continuity: int | None,
        n_open: int,
        deg_open: int,
    ) -> Bspline:
        """Build a 2-D B-spline (periodic x open).

        Args:
            n_per (int): Number of intervals for the periodic direction.
            deg_per (int): Degree for the periodic direction.
            continuity (int | None): Continuity for the periodic direction.
            n_open (int): Number of intervals for the open direction.
            deg_open (int): Degree for the open direction.

        Returns:
            Bspline: A 2-D B-spline with one periodic and one open direction.
        """
        knots_per = create_uniform_periodic(n_per, deg_per, continuity=continuity, dtype=np.float64)
        knots_open = np.array(
            [0.0] * (deg_open + 1)
            + list(np.linspace(0, 1, n_open + 1)[1:-1])
            + [1.0] * (deg_open + 1),
            dtype=np.float64,
        )
        s_per = BsplineSpace1D(knots_per, deg_per, periodic=True)
        s_open = BsplineSpace1D(knots_open, deg_open)
        space = BsplineSpace([s_per, s_open])
        n = space.num_total_basis
        ctrl = np.arange(1.0, n + 1.0, dtype=np.float64)
        return Bspline(space, ctrl)

    def test_2d_one_periodic_C0_evaluate_consistent(self) -> None:
        """2-D (periodic C^0 x open) evaluate is consistent with evaluate_derivatives."""
        f = self._make_periodic_open_2d(4, 2, 0, 3, 2)

        a0, b0 = f.space.spaces[0].domain
        a1, b1 = f.space.spaces[1].domain
        us = np.linspace(float(a0), float(b0), 7, dtype=np.float64)[1:-1]
        vs = np.linspace(float(a1), float(b1), 7, dtype=np.float64)[1:-1]
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        pts = np.column_stack([uu.ravel(), vv.ravel()])

        np.testing.assert_allclose(f.evaluate(pts), f.evaluate_derivatives(pts, [0, 0]), atol=1e-13)

    def test_2d_one_periodic_degree3_C1_evaluate_consistent(self) -> None:
        """2-D (periodic degree-3 C^1 x open) evaluate is consistent with evaluate_derivatives."""
        f = self._make_periodic_open_2d(5, 3, 1, 3, 2)

        a0, b0 = f.space.spaces[0].domain
        a1, b1 = f.space.spaces[1].domain
        us = np.linspace(float(a0), float(b0), 7, dtype=np.float64)[1:-1]
        vs = np.linspace(float(a1), float(b1), 7, dtype=np.float64)[1:-1]
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        pts = np.column_stack([uu.ravel(), vv.ravel()])

        np.testing.assert_allclose(f.evaluate(pts), f.evaluate_derivatives(pts, [0, 0]), atol=1e-13)

    def test_2d_both_periodic_C0_evaluate_consistent(self) -> None:
        """2-D (periodic C^0 x periodic C^0) evaluate is consistent with evaluate_derivatives."""
        knots0 = create_uniform_periodic(4, 2, continuity=0, dtype=np.float64)
        knots1 = create_uniform_periodic(4, 2, continuity=0, dtype=np.float64)
        s0 = BsplineSpace1D(knots0, 2, periodic=True)
        s1 = BsplineSpace1D(knots1, 2, periodic=True)
        space = BsplineSpace([s0, s1])
        n = space.num_total_basis
        ctrl = np.arange(1.0, n + 1.0, dtype=np.float64)
        f = Bspline(space, ctrl)

        a0, b0 = f.space.spaces[0].domain
        a1, b1 = f.space.spaces[1].domain
        us = np.linspace(float(a0), float(b0), 6, dtype=np.float64)[1:-1]
        vs = np.linspace(float(a1), float(b1), 6, dtype=np.float64)[1:-1]
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        pts = np.column_stack([uu.ravel(), vv.ravel()])

        np.testing.assert_allclose(f.evaluate(pts), f.evaluate_derivatives(pts, [0, 0]), atol=1e-13)

    def test_2d_one_periodic_max_continuity_constant_field(self) -> None:
        """Periodic x open spline with all-ones ctrl evaluates to f=1 (partition of unity)."""
        knots_per = create_uniform_periodic(4, 2, dtype=np.float64)
        knots_open = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64)
        s_per = BsplineSpace1D(knots_per, 2, periodic=True)
        s_open = BsplineSpace1D(knots_open, 2)
        space = BsplineSpace([s_per, s_open])
        n = space.num_total_basis
        ctrl = np.ones(n, dtype=np.float64)
        f = Bspline(space, ctrl)

        a0, b0 = f.space.spaces[0].domain
        a1, b1 = f.space.spaces[1].domain
        us = np.linspace(float(a0), float(b0), 6, dtype=np.float64)[1:-1]
        vs = np.linspace(float(a1), float(b1), 6, dtype=np.float64)[1:-1]
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        pts = np.column_stack([uu.ravel(), vv.ravel()])

        np.testing.assert_allclose(f.evaluate(pts), 1.0, atol=1e-12)

    def test_2d_both_periodic_max_continuity_constant_field(self) -> None:
        """Periodic x periodic spline with all-ones ctrl evaluates to f=1 (partition of unity)."""
        knots0 = create_uniform_periodic(4, 2, dtype=np.float64)
        knots1 = create_uniform_periodic(4, 2, dtype=np.float64)
        s0 = BsplineSpace1D(knots0, 2, periodic=True)
        s1 = BsplineSpace1D(knots1, 2, periodic=True)
        space = BsplineSpace([s0, s1])
        n = space.num_total_basis
        ctrl = np.ones(n, dtype=np.float64)
        f = Bspline(space, ctrl)

        a0, b0 = f.space.spaces[0].domain
        a1, b1 = f.space.spaces[1].domain
        us = np.linspace(float(a0), float(b0), 6, dtype=np.float64)[1:-1]
        vs = np.linspace(float(a1), float(b1), 6, dtype=np.float64)[1:-1]
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        pts = np.column_stack([uu.ravel(), vv.ravel()])

        np.testing.assert_allclose(f.evaluate(pts), 1.0, atol=1e-12)
