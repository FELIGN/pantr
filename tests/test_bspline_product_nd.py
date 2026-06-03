"""Tests for N-dimensional B-spline pointwise multiplication."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bezier._bezier_product import (
    _bernstein_product_coefficients,
    _bernstein_product_coefficients_nd,
)
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D, create_uniform_periodic_knots
from pantr.bspline._bspline_knots import _get_unique_knots_and_multiplicity_impl
from pantr.quad import PointsLattice

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_space_1d(
    knots: list[float], degree: int, dtype: type = np.float64, periodic: bool = False
) -> BsplineSpace1D:
    """Create a 1D B-spline space."""
    return BsplineSpace1D(np.array(knots, dtype=dtype), degree, periodic=periodic)


def _make_2d_bspline(
    space_u: BsplineSpace1D,
    space_v: BsplineSpace1D,
    ctrl: npt.NDArray[np.float64],
    is_rational: bool = False,
) -> Bspline:
    """Create a 2D B-spline from two 1D spaces and control points."""
    space = BsplineSpace([space_u, space_v])
    return Bspline(space, ctrl, is_rational=is_rational)


def _make_3d_bspline(
    space_u: BsplineSpace1D,
    space_v: BsplineSpace1D,
    space_w: BsplineSpace1D,
    ctrl: npt.NDArray[np.float64],
    is_rational: bool = False,
) -> Bspline:
    """Create a 3D B-spline from three 1D spaces and control points."""
    space = BsplineSpace([space_u, space_v, space_w])
    return Bspline(space, ctrl, is_rational=is_rational)


def _eval_lattice_pts(n: int = 21, a: float = 0.0, b: float = 1.0) -> npt.NDArray[np.float64]:
    """Return evenly-spaced 1D evaluation points in [a, b]."""
    return np.linspace(a, b, n, dtype=np.float64)


def _eval_2d_product(f: Bspline, g: Bspline, h: Bspline, n: int = 21, atol: float = 1e-10) -> None:
    """Assert h == f*g by evaluating on a lattice at interior points.

    Uses strictly interior points to avoid boundary evaluation issues with
    periodic and non-open B-splines.
    """
    dom_u, dom_v = f.space.spaces[0].domain, f.space.spaces[1].domain
    pts_u = _eval_lattice_pts(n, float(dom_u[0]), float(dom_u[1]))[1:-1]
    pts_v = _eval_lattice_pts(n, float(dom_v[0]), float(dom_v[1]))[1:-1]
    lattice = PointsLattice([pts_u, pts_v])

    f_vals = f.evaluate(lattice)
    g_vals = g.evaluate(lattice)
    h_vals = h.evaluate(lattice)
    np.testing.assert_allclose(h_vals, f_vals * g_vals, atol=atol)


def _eval_3d_product(f: Bspline, g: Bspline, h: Bspline, n: int = 11, atol: float = 1e-10) -> None:
    """Assert h == f*g by evaluating on a 3D lattice at interior points.

    Uses strictly interior points to avoid boundary evaluation issues with
    periodic and non-open B-splines.
    """
    dom_u = f.space.spaces[0].domain
    dom_v = f.space.spaces[1].domain
    dom_w = f.space.spaces[2].domain
    pts_u = _eval_lattice_pts(n, float(dom_u[0]), float(dom_u[1]))[1:-1]
    pts_v = _eval_lattice_pts(n, float(dom_v[0]), float(dom_v[1]))[1:-1]
    pts_w = _eval_lattice_pts(n, float(dom_w[0]), float(dom_w[1]))[1:-1]
    lattice = PointsLattice([pts_u, pts_v, pts_w])

    f_vals = f.evaluate(lattice)
    g_vals = g.evaluate(lattice)
    h_vals = h.evaluate(lattice)
    np.testing.assert_allclose(h_vals, f_vals * g_vals, atol=atol)


# ---------------------------------------------------------------------------
# nD Bernstein product unit tests
# ---------------------------------------------------------------------------


class TestBernsteinProductND:
    """Unit tests for the nD Bernstein product formula."""

    def test_1d_matches_existing(self) -> None:
        """1D nD product matches the existing 1D implementation."""
        rng = np.random.default_rng(42)
        b_f = rng.random((4, 2))  # degree 3, rank 2
        b_g = rng.random((3, 2))  # degree 2, rank 2

        result_1d = _bernstein_product_coefficients(b_f, b_g)
        result_nd = _bernstein_product_coefficients_nd(b_f, b_g)
        np.testing.assert_allclose(result_nd, result_1d, atol=1e-13)

    def test_2d_constant(self) -> None:
        """Product of two constant Bezier patches is constant."""
        # Degree (1,1) patches, all ctrl = 2.0 and 3.0.
        b_f = np.full((2, 2, 1), 2.0)
        b_g = np.full((2, 2, 1), 3.0)
        result = _bernstein_product_coefficients_nd(b_f, b_g)
        assert result.shape == (3, 3, 1)
        np.testing.assert_allclose(result, 6.0, atol=1e-14)

    def test_2d_separable(self) -> None:
        """Product of separable fields: f(u,v)=f_u(u)*f_v(v), g(u,v)=g_u(u)*g_v(v).

        Product should be (f_u*g_u)(u) * (f_v*g_v)(v).
        """
        rng = np.random.default_rng(123)
        fu = rng.random(3)  # degree 2 in u
        fv = rng.random(4)  # degree 3 in v
        gu = rng.random(2)  # degree 1 in u
        gv = rng.random(3)  # degree 2 in v

        # Build 2D control points as outer products (rank=1).
        b_f = np.outer(fu, fv)[:, :, np.newaxis]
        b_g = np.outer(gu, gv)[:, :, np.newaxis]

        result = _bernstein_product_coefficients_nd(b_f, b_g)

        # Expected: outer product of 1D products.
        prod_u = _bernstein_product_coefficients(fu[:, np.newaxis], gu[:, np.newaxis])[:, 0]
        prod_v = _bernstein_product_coefficients(fv[:, np.newaxis], gv[:, np.newaxis])[:, 0]
        expected = np.outer(prod_u, prod_v)[:, :, np.newaxis]

        np.testing.assert_allclose(result, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 2D non-rational product tests
# ---------------------------------------------------------------------------


class TestNonRationalProduct2D:
    """Correctness tests for 2D non-rational B-spline multiplication."""

    def test_same_spaces_both_directions(self) -> None:
        """Product of two biquadratic B-splines on the same mesh."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        s = _make_space_1d(knots, 2)

        rng = np.random.default_rng(0)
        n = 4  # num_basis per direction
        f = _make_2d_bspline(s, s, rng.random((n, n, 1)))
        g = _make_2d_bspline(s, s, rng.random((n, n, 1)))

        h = f * g

        assert h.degree == (4, 4)
        _eval_2d_product(f, g, h)

    def test_different_knots_per_direction(self) -> None:
        """Product with different knot vectors in u and v."""
        knots_u = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        knots_v = [0.0, 0.0, 0.0, 0.25, 0.75, 1.0, 1.0, 1.0]
        s_u = _make_space_1d(knots_u, 2)
        s_v = _make_space_1d(knots_v, 2)

        rng = np.random.default_rng(1)
        n_u, n_v = 4, 5
        f = _make_2d_bspline(s_u, s_v, rng.random((n_u, n_v, 1)))
        g = _make_2d_bspline(s_u, s_v, rng.random((n_u, n_v, 1)))

        h = f * g
        _eval_2d_product(f, g, h)

    def test_different_degrees(self) -> None:
        """Product with different degrees in u (2) and v (3)."""
        knots_u = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        knots_v = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
        s_u = _make_space_1d(knots_u, 2)
        s_v = _make_space_1d(knots_v, 3)

        rng = np.random.default_rng(2)
        n_u_f, n_v_f = 4, 5
        f = _make_2d_bspline(s_u, s_v, rng.random((n_u_f, n_v_f, 1)))

        # g has different degrees: degree 3 in u, degree 1 in v.
        knots_gu = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
        knots_gv = [0.0, 0.0, 0.5, 1.0, 1.0]
        s_gu = _make_space_1d(knots_gu, 3)
        s_gv = _make_space_1d(knots_gv, 1)
        n_u_g, n_v_g = 5, 3
        g = _make_2d_bspline(s_gu, s_gv, rng.random((n_u_g, n_v_g, 1)))

        h = f * g
        assert h.degree == (5, 4)
        _eval_2d_product(f, g, h)

    def test_vector_field_rank2(self) -> None:
        """Product of rank-2 B-spline fields."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        s = _make_space_1d(knots, 2)

        rng = np.random.default_rng(3)
        n = 3
        f = _make_2d_bspline(s, s, rng.random((n, n, 2)))
        g = _make_2d_bspline(s, s, rng.random((n, n, 2)))

        h = f * g
        _eval_2d_product(f, g, h)

    def test_single_bezier_element(self) -> None:
        """Product of single Bezier patches (no interior breakpoints)."""
        knots_u = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots_v = [0.0, 0.0, 1.0, 1.0]
        s_u = _make_space_1d(knots_u, 2)
        s_v = _make_space_1d(knots_v, 1)

        rng = np.random.default_rng(4)
        f = _make_2d_bspline(s_u, s_v, rng.random((3, 2, 1)))
        g = _make_2d_bspline(s_u, s_v, rng.random((3, 2, 1)))

        h = f * g
        assert h.degree == (4, 2)
        _eval_2d_product(f, g, h)

    def test_different_meshes_per_operand(self) -> None:
        """Operands have non-matching interior breakpoints in both directions."""
        knots_f_u = [0.0, 0.0, 0.0, 0.3, 1.0, 1.0, 1.0]
        knots_g_u = [0.0, 0.0, 0.0, 0.7, 1.0, 1.0, 1.0]
        knots_f_v = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        knots_g_v = [0.0, 0.0, 0.0, 0.25, 0.75, 1.0, 1.0, 1.0]

        s_fu = _make_space_1d(knots_f_u, 2)
        s_gu = _make_space_1d(knots_g_u, 2)
        s_fv = _make_space_1d(knots_f_v, 2)
        s_gv = _make_space_1d(knots_g_v, 2)

        rng = np.random.default_rng(5)
        f = _make_2d_bspline(s_fu, s_fv, rng.random((4, 4, 1)))
        g = _make_2d_bspline(s_gu, s_gv, rng.random((4, 5, 1)))

        h = f * g
        _eval_2d_product(f, g, h)

    def test_multiply_by_constant(self) -> None:
        """Multiplying by a constant field scales the spline."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        s = _make_space_1d(knots, 2)

        rng = np.random.default_rng(6)
        f = _make_2d_bspline(s, s, rng.random((4, 4, 1)))

        # Constant g: degree 0 Bezier in both directions.
        knots_const = [0.0, 1.0]
        s_const = _make_space_1d(knots_const, 0)
        g = _make_2d_bspline(s_const, s_const, np.array([[[3.0]]]))

        h = f * g
        _eval_2d_product(f, g, h)


# ---------------------------------------------------------------------------
# 3D non-rational product test
# ---------------------------------------------------------------------------


class TestNonRationalProduct3D:
    """Correctness test for 3D B-spline multiplication."""

    def test_trilinear_product(self) -> None:
        """Product of two trilinear B-splines."""
        knots = [0.0, 0.0, 1.0, 1.0]
        s = _make_space_1d(knots, 1)

        rng = np.random.default_rng(10)
        n = 2
        f = _make_3d_bspline(s, s, s, rng.random((n, n, n, 1)))
        g = _make_3d_bspline(s, s, s, rng.random((n, n, n, 1)))

        h = f * g
        assert h.degree == (2, 2, 2)
        _eval_3d_product(f, g, h)

    def test_mixed_degrees_3d(self) -> None:
        """3D product with different degrees per direction."""
        knots_1 = [0.0, 0.0, 1.0, 1.0]
        knots_2 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        knots_3 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

        s1 = _make_space_1d(knots_1, 1)
        s2 = _make_space_1d(knots_2, 2)
        s3 = _make_space_1d(knots_3, 3)

        rng = np.random.default_rng(11)
        f = _make_3d_bspline(s1, s2, s3, rng.random((2, 3, 4, 1)))
        g = _make_3d_bspline(s2, s1, s2, rng.random((3, 2, 3, 1)))

        h = f * g
        assert h.degree == (3, 3, 5)
        _eval_3d_product(f, g, h)


# ---------------------------------------------------------------------------
# Optimal continuity tests
# ---------------------------------------------------------------------------


def _get_interior_mults(
    space_1d: BsplineSpace1D,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Return interior breakpoints and their multiplicities for a 1D space."""
    unique, mults = _get_unique_knots_and_multiplicity_impl(
        space_1d.knots, space_1d.degree, float(space_1d.tolerance), in_domain=True
    )
    return unique[1:-1], mults[1:-1]


class TestOptimalContinuity2D:
    """Verify that the 2D product has optimal interior multiplicities per direction."""

    def test_same_mesh_c1_both_directions(self) -> None:
        """Both operands C^1 at 0.5 in both dirs -> product interior mult = max(1+2,1+2) = 3.

        Full-Bezier would give mult 4. Optimal gives mult 3 (C^1).
        """
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        s = _make_space_1d(knots, 2)
        n = s.num_basis

        rng = np.random.default_rng(50)
        f = _make_2d_bspline(s, s, rng.random((n, n, 1)))
        g = _make_2d_bspline(s, s, rng.random((n, n, 1)))

        h = f * g
        assert h.degree == (4, 4)

        for d in range(2):
            bp, mults = _get_interior_mults(h.space.spaces[d])
            assert len(bp) == 1
            assert int(mults[0]) == 3
            np.testing.assert_allclose(bp[0], 0.5, atol=1e-12)

        # Fewer basis than full-Bezier: optimal has 8 per dir, full-Bezier has 9.
        assert h.space.num_basis == (8, 8)
        _eval_2d_product(f, g, h)

    def test_c0_in_one_direction(self) -> None:
        """C^0 in u (mult=2) and C^1 in v (mult=1) -> different optimal mults per dir.

        u-direction: max(2+2, 1+2) = 4 (full-Bezier = C^0)
        v-direction: max(1+2, 1+2) = 3 (C^1)
        """
        knots_u_f = [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0]  # mult 2 at 0.5
        knots_u_g = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]  # mult 1 at 0.5
        knots_v = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]  # mult 1 at 0.5

        s_u_f = _make_space_1d(knots_u_f, 2)
        s_u_g = _make_space_1d(knots_u_g, 2)
        s_v = _make_space_1d(knots_v, 2)

        rng = np.random.default_rng(51)
        f = _make_2d_bspline(s_u_f, s_v, rng.random((5, 4, 1)))
        g = _make_2d_bspline(s_u_g, s_v, rng.random((4, 4, 1)))

        h = f * g
        assert h.degree == (4, 4)

        # u-direction: mult = max(2+2, 1+2) = 4 (C^0)
        bp_u, mults_u = _get_interior_mults(h.space.spaces[0])
        assert len(bp_u) == 1
        assert int(mults_u[0]) == 4

        # v-direction: mult = max(1+2, 1+2) = 3 (C^1)
        bp_v, mults_v = _get_interior_mults(h.space.spaces[1])
        assert len(bp_v) == 1
        assert int(mults_v[0]) == 3

        _eval_2d_product(f, g, h)

    def test_different_degrees_shared_breakpoint(self) -> None:
        """Mixed degrees with shared breakpoint: p=(2,3), q=(3,2), product=(5,5).

        Both operands share breakpoint 0.5 in both directions.
        u-direction: f mult 1 (deg 2), g mult 1 (deg 3) -> max(1+3, 1+2) = 4
        v-direction: f mult 1 (deg 3), g mult 1 (deg 2) -> max(1+2, 1+3) = 4
        Both are less than full-Bezier (5).
        """
        knots_u_f = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]  # degree 2, mult 1
        knots_v_f = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]  # degree 3, mult 1
        knots_u_g = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]  # degree 3, mult 1
        knots_v_g = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]  # degree 2, mult 1

        s_u_f = _make_space_1d(knots_u_f, 2)
        s_v_f = _make_space_1d(knots_v_f, 3)
        s_u_g = _make_space_1d(knots_u_g, 3)
        s_v_g = _make_space_1d(knots_v_g, 2)

        rng = np.random.default_rng(52)
        f = _make_2d_bspline(s_u_f, s_v_f, rng.random((4, 5, 1)))
        g = _make_2d_bspline(s_u_g, s_v_g, rng.random((5, 4, 1)))

        h = f * g
        assert h.degree == (5, 5)

        # u-direction: max(1+3, 1+2) = 4, less than full-Bezier = 5
        bp_u, mults_u = _get_interior_mults(h.space.spaces[0])
        assert len(bp_u) == 1
        assert int(mults_u[0]) == 4

        # v-direction: max(1+2, 1+3) = 4, less than full-Bezier = 5
        bp_v, mults_v = _get_interior_mults(h.space.spaces[1])
        assert len(bp_v) == 1
        assert int(mults_v[0]) == 4

        _eval_2d_product(f, g, h)

    def test_multiple_shared_breakpoints(self) -> None:
        """Multiple shared breakpoints with different multiplicities per direction.

        u-direction: shared breakpoints at 1/3 and 2/3, both mult 1.
           max(1+2, 1+2) = 3.
        v-direction: shared breakpoints at 0.25, 0.5, 0.75 with mult 1.
           max(1+2, 1+2) = 3.
        """
        knots_u = [0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0]
        knots_v = [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0]

        s_u = _make_space_1d(knots_u, 2)
        s_v = _make_space_1d(knots_v, 2)

        rng = np.random.default_rng(53)
        f = _make_2d_bspline(s_u, s_v, rng.random((5, 6, 1)))
        g = _make_2d_bspline(s_u, s_v, rng.random((5, 6, 1)))

        h = f * g
        assert h.degree == (4, 4)

        # u-direction: 2 breakpoints, each with optimal mult 3 < full-Bezier 4.
        bp_u, mults_u = _get_interior_mults(h.space.spaces[0])
        assert len(bp_u) == 2
        np.testing.assert_array_equal(mults_u, [3, 3])

        # v-direction: 3 breakpoints, each with optimal mult 3 < full-Bezier 4.
        bp_v, mults_v = _get_interior_mults(h.space.spaces[1])
        assert len(bp_v) == 3
        np.testing.assert_array_equal(mults_v, [3, 3, 3])

        _eval_2d_product(f, g, h)

    def test_fewer_basis_than_full_bezier_2d(self) -> None:
        """Product has fewer total basis functions than full-Bezier in 2D.

        Both operands: degree 2, interior knot 0.5 with mult 1, same in both dirs.
        Full-Bezier: 9 x 9 = 81 basis.
        Optimal: 8 x 8 = 64 basis.
        """
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        s = _make_space_1d(knots, 2)
        n = s.num_basis

        rng = np.random.default_rng(54)
        f = _make_2d_bspline(s, s, rng.random((n, n, 1)))
        g = _make_2d_bspline(s, s, rng.random((n, n, 1)))

        h = f * g

        # Full-Bezier: degree 4, 2 elements per dir -> 9 basis per dir.
        # Optimal: interior mult 3 -> 8 basis per dir.
        assert h.space.num_total_basis == 64
        assert h.space.num_basis == (8, 8)

        _eval_2d_product(f, g, h)


# ---------------------------------------------------------------------------
# Rational product tests
# ---------------------------------------------------------------------------


class TestRationalProduct2D:
    """Tests for rational (NURBS) 2D multiplication."""

    def test_rational_times_rational(self) -> None:
        """Product of two rational 2D B-splines."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        s = _make_space_1d(knots, 2)

        rng = np.random.default_rng(20)
        n = 3
        # Rational: last column is weights (must be > 0).
        ctrl_f = rng.random((n, n, 2))
        ctrl_f[..., -1] = np.abs(ctrl_f[..., -1]) + 0.1
        ctrl_g = rng.random((n, n, 2))
        ctrl_g[..., -1] = np.abs(ctrl_g[..., -1]) + 0.1

        f = _make_2d_bspline(s, s, ctrl_f, is_rational=True)
        g = _make_2d_bspline(s, s, ctrl_g, is_rational=True)

        h = f * g
        assert h.is_rational
        _eval_2d_product(f, g, h, atol=1e-9)

    def test_mixed_rational_nonrational(self) -> None:
        """Product of rational x non-rational 2D B-splines."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        s = _make_space_1d(knots, 2)

        rng = np.random.default_rng(21)
        n = 3
        ctrl_f = rng.random((n, n, 2))
        ctrl_f[..., -1] = np.abs(ctrl_f[..., -1]) + 0.1
        f = _make_2d_bspline(s, s, ctrl_f, is_rational=True)
        g = _make_2d_bspline(s, s, rng.random((n, n, 1)))

        h = f * g
        assert h.is_rational
        _eval_2d_product(f, g, h, atol=1e-9)


# ---------------------------------------------------------------------------
# Boundary type tests
# ---------------------------------------------------------------------------


class TestBoundaryTypes2D:
    """Tests for per-direction boundary type handling in 2D products."""

    def test_both_open(self) -> None:
        """Both directions open should produce open result."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        s = _make_space_1d(knots, 2)

        rng = np.random.default_rng(30)
        f = _make_2d_bspline(s, s, rng.random((4, 4, 1)))
        g = _make_2d_bspline(s, s, rng.random((4, 4, 1)))

        h = f * g
        for d in range(2):
            assert h.space.spaces[d].has_open_knots()
            assert not h.space.spaces[d].periodic
        _eval_2d_product(f, g, h)

    def test_both_periodic(self) -> None:
        """Both directions periodic should produce periodic result.

        Verification: convert operands to open, compute open product, and compare
        against periodic product converted to open at interior points.
        """
        for degree in [1, 2, 3]:
            n_spans = 4
            knots = create_uniform_periodic_knots(degree, n_spans)
            s = BsplineSpace1D(knots, degree, periodic=True)
            n_per = s.num_basis

            rng = np.random.default_rng(31 + degree)
            f = _make_2d_bspline(s, s, rng.random((n_per, n_per, 1)))
            g = _make_2d_bspline(s, s, rng.random((n_per, n_per, 1)))

            h = f * g
            for d in range(2):
                assert h.space.spaces[d].periodic

            # Verify correctness: product must reproduce pointwise product.
            dom_u = h.space.spaces[0].domain
            dom_v = h.space.spaces[1].domain
            pts_u = _eval_lattice_pts(21, float(dom_u[0]), float(dom_u[1]))[1:-1]
            pts_v = _eval_lattice_pts(21, float(dom_v[0]), float(dom_v[1]))[1:-1]
            lattice = PointsLattice([pts_u, pts_v])
            np.testing.assert_allclose(
                h.evaluate(lattice), f.evaluate(lattice) * g.evaluate(lattice), atol=1e-10
            )

    def test_mixed_periodic_open(self) -> None:
        """One direction periodic, one open -> periodic in u, open in v."""
        degree = 2
        knots_per = create_uniform_periodic_knots(degree, 4)
        s_per = BsplineSpace1D(knots_per, degree, periodic=True)
        knots_open = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        s_open = _make_space_1d(knots_open, 2)

        rng = np.random.default_rng(32)
        # f: periodic in u, open in v
        f = _make_2d_bspline(s_per, s_open, rng.random((s_per.num_basis, 4, 1)))
        # g: same structure
        g = _make_2d_bspline(s_per, s_open, rng.random((s_per.num_basis, 4, 1)))

        h = f * g
        # u-direction: periodic x periodic -> periodic
        assert h.space.spaces[0].periodic
        # v-direction: open x open -> open
        assert h.space.spaces[1].has_open_knots()
        assert not h.space.spaces[1].periodic

        # Verify correctness: product must reproduce pointwise product.
        dom_u = h.space.spaces[0].domain
        dom_v = h.space.spaces[1].domain
        pts_u = _eval_lattice_pts(21, float(dom_u[0]), float(dom_u[1]))[1:-1]
        pts_v = _eval_lattice_pts(21, float(dom_v[0]), float(dom_v[1]))[1:-1]
        lattice = PointsLattice([pts_u, pts_v])
        np.testing.assert_allclose(
            h.evaluate(lattice), f.evaluate(lattice) * g.evaluate(lattice), atol=1e-10
        )

    def test_nonopen_both_directions(self) -> None:
        """Both directions non-open should produce non-open result."""
        # Non-open knot vector: boundary mult (2) < degree+1 (3).
        knots_no = np.array([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0], dtype=np.float64)
        s_no = BsplineSpace1D(knots_no, 2)

        rng = np.random.default_rng(33)
        n = s_no.num_basis
        f = _make_2d_bspline(s_no, s_no, rng.random((n, n, 1)))
        g = _make_2d_bspline(s_no, s_no, rng.random((n, n, 1)))

        h = f * g
        for d in range(2):
            assert not h.space.spaces[d].has_open_knots()
            assert not h.space.spaces[d].periodic
        _eval_2d_product(f, g, h)


# ---------------------------------------------------------------------------
# Validation error tests
# ---------------------------------------------------------------------------


class TestValidationErrors:
    """Test that proper errors are raised for invalid inputs."""

    def test_different_dims(self) -> None:
        """Multiplying 1D x 2D should raise ValueError."""
        knots = [0.0, 0.0, 1.0, 1.0]
        s = _make_space_1d(knots, 1)

        f_1d = Bspline(BsplineSpace([s]), np.array([[1.0], [2.0]]))
        f_2d = _make_2d_bspline(s, s, np.ones((2, 2, 1)))

        with pytest.raises(ValueError, match="same parametric dimension"):
            f_2d * f_1d

    def test_different_dtypes(self) -> None:
        """Multiplying splines with different dtypes should raise."""
        knots_64 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
        knots_32 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        s64 = BsplineSpace1D(knots_64, 1)
        s32 = BsplineSpace1D(knots_32, 1)

        f = _make_2d_bspline(s64, s64, np.ones((2, 2, 1), dtype=np.float64))
        g = _make_2d_bspline(s32, s32, np.ones((2, 2, 1), dtype=np.float32))

        with pytest.raises(ValueError, match="same dtype"):
            f * g

    def test_different_ranks(self) -> None:
        """Multiplying rank-1 x rank-2 should raise."""
        knots = [0.0, 0.0, 1.0, 1.0]
        s = _make_space_1d(knots, 1)

        f = _make_2d_bspline(s, s, np.ones((2, 2, 1)))
        g = _make_2d_bspline(s, s, np.ones((2, 2, 2)))

        with pytest.raises(ValueError, match="same rank"):
            f * g

    def test_different_domains(self) -> None:
        """Multiplying splines on different domains should raise."""
        s1 = _make_space_1d([0.0, 0.0, 1.0, 1.0], 1)
        s2 = _make_space_1d([0.0, 0.0, 2.0, 2.0], 1)

        f = _make_2d_bspline(s1, s1, np.ones((2, 2, 1)))
        g = _make_2d_bspline(s1, s2, np.ones((2, 2, 1)))

        with pytest.raises(ValueError, match="same parametric domain"):
            f * g
