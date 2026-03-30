"""Tests for interpolate_bspline, fit_bspline, l2_project_bspline, and support functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pytest

from pantr.bspline import (
    BsplineSpace1D,
    create_greville_lattice,
    create_uniform_open_knots,
    create_uniform_periodic_knots,
    create_uniform_space,
    fit_bspline,
    get_greville_abscissae,
    interpolate_bspline,
    l2_project_bspline,
)
from pantr.quad import PointsLattice

# ---------------------------------------------------------------------------
# Greville abscissae
# ---------------------------------------------------------------------------


class TestGrevilleAbscissae:
    """Tests for get_greville_abscissae."""

    def test_open_degree2(self) -> None:
        """Greville points for open degree-2 space with 2 intervals."""
        knots = create_uniform_open_knots(2, 2)
        space = BsplineSpace1D(knots, 2)
        g = get_greville_abscissae(space)
        assert g.shape == (space.num_basis,)
        nptest.assert_allclose(g, [0.0, 0.25, 0.75, 1.0], atol=1e-14)

    def test_open_degree3(self) -> None:
        """Greville points for open degree-3 space with 4 intervals."""
        knots = create_uniform_open_knots(4, 3)
        space = BsplineSpace1D(knots, 3)
        g = get_greville_abscissae(space)
        assert g.shape == (space.num_basis,)
        # Endpoints should be exactly at domain boundaries.
        assert g[0] == pytest.approx(0.0, abs=1e-14)
        assert g[-1] == pytest.approx(1.0, abs=1e-14)

    def test_periodic(self) -> None:
        """Greville points for periodic space are sorted and inside domain."""
        knots = create_uniform_periodic_knots(4, 3)
        space = BsplineSpace1D(knots, 3, periodic=True)
        g = get_greville_abscissae(space)
        assert g.shape == (space.num_basis,)
        a, b = space.domain
        assert np.all(g >= a - 1e-14)
        assert np.all(g < b + 1e-14)
        # Should be sorted.
        assert np.all(np.diff(g) >= 0)

    def test_degree0(self) -> None:
        """Greville points for degree-0 space are midpoints."""
        knots = create_uniform_open_knots(3, 0)
        space = BsplineSpace1D(knots, 0)
        g = get_greville_abscissae(space)
        expected = np.array([1.0 / 6, 0.5, 5.0 / 6])
        nptest.assert_allclose(g, expected, atol=1e-14)

    def test_type_error(self) -> None:
        """Passing wrong type raises TypeError."""
        with pytest.raises(TypeError, match="Expected BsplineSpace1D"):
            get_greville_abscissae("not a space")  # type: ignore[arg-type]


class TestGrevilleLattice:
    """Tests for create_greville_lattice."""

    def test_2d(self) -> None:
        """Greville lattice for a 2D space has correct structure."""
        space = create_uniform_space([2, 3], [3, 4])
        lat = create_greville_lattice(space)
        assert isinstance(lat, PointsLattice)
        assert lat.dim == 2  # noqa: PLR2004
        assert lat.pts_per_dir[0].shape[0] == space.num_basis[0]
        assert lat.pts_per_dir[1].shape[0] == space.num_basis[1]


# ---------------------------------------------------------------------------
# create_uniform_space
# ---------------------------------------------------------------------------


class TestCreateUniformSpace:
    """Tests for create_uniform_space."""

    def test_1d_scalar_args(self) -> None:
        """Scalar arguments create a 1D space."""
        space = create_uniform_space(3, 4)
        assert space.dim == 1
        assert space.degrees == (3,)

    def test_2d_list_args(self) -> None:
        """Sequence arguments create an nD space."""
        space = create_uniform_space([2, 3], [4, 5])
        assert space.dim == 2  # noqa: PLR2004
        assert space.degrees == (2, 3)

    def test_periodic(self) -> None:
        """Periodic flag creates periodic spaces."""
        space = create_uniform_space(3, 4, periodic=True)
        assert space.spaces[0].periodic is True

    def test_mixed_periodic(self) -> None:
        """Per-direction periodic flags work."""
        space = create_uniform_space([2, 2], [3, 3], periodic=[True, False])
        assert space.spaces[0].periodic is True
        assert space.spaces[1].periodic is False

    def test_custom_domain(self) -> None:
        """Custom domain boundaries are applied."""
        space = create_uniform_space(2, 3, domain=(0.0, 2.0))
        a, b = space.spaces[0].domain
        assert a == pytest.approx(0.0)
        assert b == pytest.approx(2.0)

    def test_periodic_tuple_infers_ndim(self) -> None:
        """A tuple periodic=(True, False) correctly infers ndim=2."""
        space = create_uniform_space([2, 2], [3, 3], periodic=(True, False))
        assert space.dim == 2  # noqa: PLR2004
        assert space.spaces[0].periodic is True
        assert space.spaces[1].periodic is False

    def test_tuple_degree_infers_ndim(self) -> None:
        """A tuple degree=(3, 4) with scalar num_intervals infers ndim=2."""
        space = create_uniform_space(degree=(3, 4), num_intervals=4)
        assert space.dim == 2  # noqa: PLR2004
        assert space.degrees == (3, 4)
        assert space.spaces[0].degree == 3  # noqa: PLR2004
        assert space.spaces[1].degree == 4  # noqa: PLR2004

    def test_inconsistent_lengths_raises(self) -> None:
        """Inconsistent sequence lengths raise ValueError."""
        with pytest.raises(ValueError, match="Inconsistent"):
            create_uniform_space([2, 3], [4, 5, 6])


# ---------------------------------------------------------------------------
# interpolate_bspline — 1D scalar
# ---------------------------------------------------------------------------


class TestInterpolate1DScalar:
    """Tests for 1D scalar interpolation."""

    def test_polynomial_reproduction_quadratic(self) -> None:
        """Degree-3 interpolation reproduces x^2 exactly."""
        space = create_uniform_space(3, 4)
        b = interpolate_bspline(
            lambda lat: lat.get_all_points()[:, 0] ** 2,
            space,
        )
        pts = np.linspace(0, 1, 11)
        nptest.assert_allclose(b.evaluate(pts), pts**2, atol=1e-13)

    def test_polynomial_reproduction_cubic(self) -> None:
        """Degree-3 interpolation reproduces x^3 exactly."""
        space = create_uniform_space(3, 4)
        b = interpolate_bspline(
            lambda lat: lat.get_all_points()[:, 0] ** 3,
            space,
        )
        pts = np.linspace(0, 1, 11)
        nptest.assert_allclose(b.evaluate(pts), pts**3, atol=1e-12)

    def test_custom_domain(self) -> None:
        """Interpolation on a non-unit domain."""
        space = create_uniform_space(3, 4, domain=(2.0, 5.0))
        b = interpolate_bspline(
            lambda lat: lat.get_all_points()[:, 0] ** 2,
            space,
        )
        pts = np.linspace(2, 5, 11)
        nptest.assert_allclose(b.evaluate(pts), pts**2, atol=1e-10)

    def test_custom_nodes(self) -> None:
        """Interpolation with user-provided nodes."""
        space = create_uniform_space(3, 4)
        nodes = np.linspace(0, 1, space.num_basis[0])
        b = interpolate_bspline(
            lambda lat: lat.get_all_points()[:, 0] ** 2,
            space,
            nodes=[nodes],
        )
        pts = np.linspace(0, 1, 11)
        nptest.assert_allclose(b.evaluate(pts), pts**2, atol=1e-12)


# ---------------------------------------------------------------------------
# interpolate_bspline — 1D vector-valued
# ---------------------------------------------------------------------------


class TestInterpolate1DVector:
    """Tests for 1D vector-valued interpolation (parametric curves)."""

    def test_quarter_circle(self) -> None:
        """Interpolate a quarter circle arc."""
        space = create_uniform_space(3, 8, domain=(0.0, np.pi / 2))

        def quarter_circle(lat: PointsLattice) -> npt.NDArray[Any]:
            t = lat.get_all_points()[:, 0]
            return np.stack([np.cos(t), np.sin(t)], axis=-1)

        b = interpolate_bspline(quarter_circle, space)
        assert b.rank == 2  # noqa: PLR2004

        pts = np.linspace(0, np.pi / 2, 21)
        vals = b.evaluate(pts)
        expected = np.stack([np.cos(pts), np.sin(pts)], axis=-1)
        nptest.assert_allclose(vals, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# interpolate_bspline — periodic
# ---------------------------------------------------------------------------


class TestInterpolatePeriodic:
    """Tests for periodic B-spline interpolation."""

    def test_sin_periodic(self) -> None:
        """Interpolate sin(x) on a periodic space over [0, 2pi]."""
        space = create_uniform_space(3, 16, periodic=True, domain=(0.0, 2 * np.pi))
        b = interpolate_bspline(
            lambda lat: np.sin(lat.get_all_points()[:, 0]),
            space,
        )
        pts = np.linspace(0, 2 * np.pi, 31)[:-1]  # Exclude endpoint for periodic.
        vals = b.evaluate(pts)
        nptest.assert_allclose(vals, np.sin(pts), atol=5e-5)


# ---------------------------------------------------------------------------
# interpolate_bspline — 2D
# ---------------------------------------------------------------------------


class TestInterpolate2D:
    """Tests for 2D tensor-product interpolation."""

    def test_bilinear(self) -> None:
        """Interpolate x + y exactly with degree-2 space."""
        space = create_uniform_space([2, 2], [3, 3])

        b = interpolate_bspline(
            lambda lat: lat.get_all_points()[:, 0] + lat.get_all_points()[:, 1],
            space,
        )
        assert b.degree == (2, 2)

        lat = PointsLattice([np.linspace(0, 1, 5), np.linspace(0, 1, 5)])
        vals = b.evaluate(lat)
        pts = lat.get_all_points()
        expected = (pts[:, 0] + pts[:, 1]).reshape(5, 5)
        nptest.assert_allclose(vals.reshape(5, 5), expected, atol=1e-12)

    def test_biquadratic(self) -> None:
        """Interpolate x^2 + y^2 exactly with degree-3 space."""
        space = create_uniform_space([3, 3], [4, 4])

        b = interpolate_bspline(
            lambda lat: lat.get_all_points()[:, 0] ** 2 + lat.get_all_points()[:, 1] ** 2,
            space,
        )

        lat = PointsLattice([np.linspace(0, 1, 5), np.linspace(0, 1, 5)])
        vals = b.evaluate(lat)
        pts = lat.get_all_points()
        expected = (pts[:, 0] ** 2 + pts[:, 1] ** 2).reshape(5, 5)
        nptest.assert_allclose(vals.reshape(5, 5), expected, atol=1e-11)


# ---------------------------------------------------------------------------
# fit_bspline — tensor-product
# ---------------------------------------------------------------------------


class TestFitTensorProduct:
    """Tests for fit_bspline with tensor-product nodes."""

    def test_exact_fit_1d(self) -> None:
        """Exact fit with n_nodes = n_basis recovers polynomial."""
        space = create_uniform_space(3, 4)
        nodes = get_greville_abscissae(space.spaces[0])
        vals = nodes**2
        b = fit_bspline(vals, [nodes], space)
        pts = np.linspace(0, 1, 11)
        nptest.assert_allclose(b.evaluate(pts), pts**2, atol=1e-12)

    def test_overdetermined_1d(self) -> None:
        """Least-squares fit with more nodes than basis functions."""
        space = create_uniform_space(3, 4)
        nodes = np.linspace(0, 1, 20)
        vals = nodes**2
        b = fit_bspline(vals, [nodes], space)
        pts = np.linspace(0, 1, 11)
        nptest.assert_allclose(b.evaluate(pts), pts**2, atol=1e-12)

    def test_2d_fit(self) -> None:
        """2D tensor-product fit."""
        space = create_uniform_space([2, 2], [3, 3])
        nodes_x = get_greville_abscissae(space.spaces[0])
        nodes_y = get_greville_abscissae(space.spaces[1])

        xx, yy = np.meshgrid(nodes_x, nodes_y, indexing="ij")
        vals = xx + yy
        b = fit_bspline(vals, [nodes_x, nodes_y], space)

        lat = PointsLattice([np.linspace(0, 1, 5), np.linspace(0, 1, 5)])
        result = b.evaluate(lat)
        pts = lat.get_all_points()
        expected = (pts[:, 0] + pts[:, 1]).reshape(5, 5)
        nptest.assert_allclose(result.reshape(5, 5), expected, atol=1e-12)

    def test_vector_valued_fit(self) -> None:
        """Fit vector-valued function (rank > 1)."""
        space = create_uniform_space(3, 4)
        nodes = get_greville_abscissae(space.spaces[0])
        vals = np.stack([nodes**2, nodes**3], axis=-1)
        b = fit_bspline(vals, [nodes], space)
        assert b.rank == 2  # noqa: PLR2004

        pts = np.linspace(0, 1, 11)
        result = b.evaluate(pts)
        nptest.assert_allclose(result[:, 0], pts**2, atol=1e-12)
        nptest.assert_allclose(result[:, 1], pts**3, atol=1e-12)


# ---------------------------------------------------------------------------
# fit_bspline — scattered
# ---------------------------------------------------------------------------


class TestFitScattered:
    """Tests for fit_bspline with scattered (non-tensor-product) nodes."""

    def test_scattered_1d(self) -> None:
        """Scattered fit in 1D."""
        space = create_uniform_space(3, 4)
        pts = np.linspace(0, 1, 20).reshape(-1, 1)
        vals = pts[:, 0] ** 2
        b = fit_bspline(vals, pts, space)
        test_pts = np.linspace(0, 1, 11)
        nptest.assert_allclose(b.evaluate(test_pts), test_pts**2, atol=1e-11)

    def test_underdetermined_raises(self) -> None:
        """Underdetermined system raises ValueError."""
        space = create_uniform_space(3, 4)
        pts = np.linspace(0, 1, 3).reshape(-1, 1)
        vals = pts[:, 0] ** 2
        with pytest.raises(ValueError, match="Underdetermined"):
            fit_bspline(vals, pts, space)


# ---------------------------------------------------------------------------
# l2_project_bspline
# ---------------------------------------------------------------------------


class TestL2Project:
    """Tests for L2 projection."""

    def test_polynomial_reproduction(self) -> None:
        """L2 projection of x^2 onto degree-3 space recovers it exactly."""
        space = create_uniform_space(3, 4)
        b = l2_project_bspline(
            lambda lat: lat.get_all_points()[:, 0] ** 2,
            space,
        )
        pts = np.linspace(0, 1, 11)
        nptest.assert_allclose(b.evaluate(pts), pts**2, atol=1e-12)

    def test_boundary_interpolation(self) -> None:
        """L2 projection with boundary interpolation matches function at endpoints."""
        space = create_uniform_space(3, 4)

        def func(lat: PointsLattice) -> npt.NDArray[Any]:
            result: npt.NDArray[Any] = np.sin(np.pi * lat.get_all_points()[:, 0])
            return result

        b = l2_project_bspline(func, space, boundary_interpolation=True)
        # sin(0) = 0, sin(pi) ≈ 0 — check boundary values.
        endpoints = np.array([0.0, 1.0])
        vals = b.evaluate(endpoints)
        expected = np.sin(np.pi * endpoints)
        nptest.assert_allclose(vals, expected, atol=1e-10)

    def test_boundary_interpolation_vector_valued(self) -> None:
        """L2 projection with boundary interpolation for vector-valued function."""
        space = create_uniform_space(3, 4)

        def func(lat: PointsLattice) -> npt.NDArray[Any]:
            x = lat.get_all_points()[:, 0]
            return np.stack([x**2, x**3], axis=-1)

        b = l2_project_bspline(func, space, boundary_interpolation=True)
        assert b.rank == 2  # noqa: PLR2004
        endpoints = np.array([0.0, 1.0])
        vals = b.evaluate(endpoints)
        expected = np.array([[0.0, 0.0], [1.0, 1.0]])
        nptest.assert_allclose(vals, expected, atol=1e-10)

    def test_gauss_lobatto_quadrature(self) -> None:
        """L2 projection with Gauss-Lobatto quadrature."""
        space = create_uniform_space(3, 4)
        b = l2_project_bspline(
            lambda lat: lat.get_all_points()[:, 0] ** 2,
            space,
            quadrature="gauss-lobatto",
        )
        pts = np.linspace(0, 1, 11)
        nptest.assert_allclose(b.evaluate(pts), pts**2, atol=1e-11)

    def test_2d_projection(self) -> None:
        """L2 projection in 2D."""
        space = create_uniform_space([2, 2], [3, 3])
        b = l2_project_bspline(
            lambda lat: lat.get_all_points()[:, 0] + lat.get_all_points()[:, 1],
            space,
        )

        lat = PointsLattice([np.linspace(0, 1, 5), np.linspace(0, 1, 5)])
        vals = b.evaluate(lat)
        pts = lat.get_all_points()
        expected = (pts[:, 0] + pts[:, 1]).reshape(5, 5)
        nptest.assert_allclose(vals.reshape(5, 5), expected, atol=1e-11)

    def test_periodic_projection(self) -> None:
        """L2 projection on a periodic space."""
        space = create_uniform_space(3, 8, periodic=True, domain=(0.0, 2 * np.pi))
        b = l2_project_bspline(
            lambda lat: np.sin(lat.get_all_points()[:, 0]),
            space,
        )
        pts = np.linspace(0, 2 * np.pi, 31)[:-1]
        vals = b.evaluate(pts)
        nptest.assert_allclose(vals, np.sin(pts), atol=1e-3)


# ---------------------------------------------------------------------------
# Boundary derivative constraints
# ---------------------------------------------------------------------------


class TestBoundaryDerivatives:
    """Tests for interpolation with boundary derivative constraints."""

    def test_zero_first_derivative(self) -> None:
        """Boundary derivative constraints set derivatives to zero at endpoints."""
        space = create_uniform_space(3, 8)
        b = interpolate_bspline(
            lambda lat: np.sin(np.pi * lat.get_all_points()[:, 0]),
            space,
            boundary_derivatives=[(1, 1)],
        )
        assert b.degree == (3,)
        # Verify that the first derivative is close to zero at both endpoints.
        d_left = b.evaluate_derivatives(np.array([0.0]), orders=1)
        d_right = b.evaluate_derivatives(np.array([1.0]), orders=1)
        nptest.assert_allclose(d_left, 0.0, atol=1e-10)
        nptest.assert_allclose(d_right, 0.0, atol=1e-10)

    def test_periodic_ignores_boundary_derivs(self) -> None:
        """Boundary derivatives are ignored for periodic directions."""
        space = create_uniform_space(3, 8, periodic=True, domain=(0.0, 2 * np.pi))
        # Should not raise even with boundary_derivatives set.
        b = interpolate_bspline(
            lambda lat: np.sin(lat.get_all_points()[:, 0]),
            space,
            boundary_derivatives=[(1, 1)],
        )
        assert b is not None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    """Tests for error handling in interpolation/fit/projection."""

    def test_interpolate_wrong_space_type(self) -> None:
        """Passing wrong type for space raises TypeError."""
        with pytest.raises(TypeError, match="Expected BsplineSpace"):
            interpolate_bspline(lambda lat: lat.get_all_points()[:, 0], "not a space")  # type: ignore[arg-type]

    def test_fit_wrong_space_type(self) -> None:
        """Passing wrong type for space raises TypeError."""
        with pytest.raises(TypeError, match="Expected BsplineSpace"):
            fit_bspline(np.array([1.0]), [np.array([0.5])], "not a space")  # type: ignore[arg-type]

    def test_l2_wrong_space_type(self) -> None:
        """Passing wrong type for space raises TypeError."""
        with pytest.raises(TypeError, match="Expected BsplineSpace"):
            l2_project_bspline(lambda lat: lat.get_all_points()[:, 0], "not a space")  # type: ignore[arg-type]

    def test_wrong_func_shape(self) -> None:
        """Function returning wrong shape raises ValueError."""
        space = create_uniform_space(3, 4)
        with pytest.raises(ValueError, match="Function returned shape"):
            interpolate_bspline(lambda lat: np.array([1.0, 2.0]), space)
