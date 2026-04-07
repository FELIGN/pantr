"""Tests for the implicit domain reparameterization module.

Validates volume and surface reparameterization with Lagrange cells for
circles (2D), spheres (3D), hyperplanes (boundary intersection), and
multi-polynomial cases.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy import typing as npt

from pantr.bezier.implicit import ImplicitQuadrature, ReparamResult
from pantr.bezier.implicit._bernstein_core import _eval_bernstein_2d, _eval_bernstein_3d

# ---------------------------------------------------------------------------
# Test geometries
# ---------------------------------------------------------------------------


def _make_circle_coeffs(r_sq: float = 0.1) -> npt.NDArray[np.float64]:
    """Bernstein degree-(2,2) coefficients for (x-0.5)^2 + (y-0.5)^2 - r_sq."""
    c_val = 0.5 - r_sq
    return np.array(
        [
            [c_val, c_val - 0.5, c_val],
            [c_val - 0.5, c_val - 1.0, c_val - 0.5],
            [c_val, c_val - 0.5, c_val],
        ],
    )


def _make_sphere_coeffs(r_sq: float = 0.09) -> npt.NDArray[np.float64]:
    """Bernstein degree-(2,2,2) for (x-.5)^2+(y-.5)^2+(z-.5)^2-r_sq."""
    const = 0.75 - r_sq
    bx = np.array([0.0, -0.5, 0.0])
    c = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c[i, j, k] = bx[i] + bx[j] + bx[k] + const
    return c


def _make_hyperplane_2d() -> npt.NDArray[np.float64]:
    """Bernstein degree-(1,1) coefficients for phi(x,y) = x + y - 0.5.

    This levelset intersects the [0,1]^2 boundary, triggering the
    degenerate-interval edge case.
    """
    return np.array([[-0.5, 0.5], [0.5, 1.5]])


# ---------------------------------------------------------------------------
# 2D volume reparameterization
# ---------------------------------------------------------------------------


class TestVolumeReparam2D:
    """Tests for 2D volume reparameterization (Lagrange quads)."""

    @pytest.fixture(scope="class")
    def circle_ipq(self) -> ImplicitQuadrature:
        return ImplicitQuadrature(_make_circle_coeffs())

    def test_basic_structure(self, circle_ipq: ImplicitQuadrature) -> None:
        """Result has correct structure and non-zero cells."""
        result = circle_ipq.volume_reparam(q=3, signs=[-1])
        assert isinstance(result, ReparamResult)
        assert result.dim == 2  # noqa: PLR2004
        assert result.cell_dim == 2  # noqa: PLR2004
        assert result.q == 3  # noqa: PLR2004
        assert result.pts_per_cell == 9  # noqa: PLR2004
        assert result.n_cells > 0
        assert result.points.shape == (result.n_cells * 9, 2)

    def test_nodes_inside_domain(self, circle_ipq: ImplicitQuadrature) -> None:
        """All volume cell nodes should satisfy phi < 0 (with tolerance)."""
        result = circle_ipq.volume_reparam(q=4, signs=[-1])
        coeffs = _make_circle_coeffs()
        values = np.array(
            [_eval_bernstein_2d(coeffs, result.points[i]) for i in range(len(result.points))]
        )
        # Allow small positive values at cell boundaries.
        assert np.all(values < 1e-6)  # noqa: PLR2004

    def test_nodes_in_unit_square(self, circle_ipq: ImplicitQuadrature) -> None:
        """All nodes should be within [0,1]^2."""
        result = circle_ipq.volume_reparam(q=4, signs=[-1])
        assert np.all(result.points >= -1e-15)  # noqa: PLR2004
        assert np.all(result.points <= 1.0 + 1e-15)

    def test_positive_side(self, circle_ipq: ImplicitQuadrature) -> None:
        """signs=[+1] should select the exterior."""
        result = circle_ipq.volume_reparam(q=3, signs=[+1])
        assert result.n_cells > 0
        coeffs = _make_circle_coeffs()
        values = np.array(
            [_eval_bernstein_2d(coeffs, result.points[i]) for i in range(len(result.points))]
        )
        assert np.all(values > -1e-6)  # noqa: PLR2004

    def test_gll_nodes(self, circle_ipq: ImplicitQuadrature) -> None:
        """GLL node type should also work."""
        result = circle_ipq.volume_reparam(q=3, signs=[-1], node_type="gll")
        assert result.n_cells > 0

    def test_q2_minimal(self, circle_ipq: ImplicitQuadrature) -> None:
        """q=2 is the minimum allowed."""
        result = circle_ipq.volume_reparam(q=2, signs=[-1])
        assert result.n_cells > 0
        assert result.pts_per_cell == 4  # noqa: PLR2004

    def test_q1_raises(self, circle_ipq: ImplicitQuadrature) -> None:
        """q=1 should be rejected."""
        with pytest.raises(ValueError, match="q must be >= 2"):
            circle_ipq.volume_reparam(q=1, signs=[-1])

    def test_sign_zero_includes_all(self) -> None:
        """signs=[0] should include the entire [0,1]^2."""
        iq = ImplicitQuadrature(_make_circle_coeffs())
        result = iq.volume_reparam(q=3, signs=[0])
        assert result.n_cells > 0
        # With sign=0, both interior and exterior intervals are kept.
        # Should have more cells than interior-only.
        result_neg = iq.volume_reparam(q=3, signs=[-1])
        assert result.n_cells >= result_neg.n_cells


# ---------------------------------------------------------------------------
# 2D surface reparameterization
# ---------------------------------------------------------------------------


class TestSurfaceReparam2D:
    """Tests for 2D surface reparameterization (Lagrange curves)."""

    @pytest.fixture(scope="class")
    def circle_ipq(self) -> ImplicitQuadrature:
        return ImplicitQuadrature(_make_circle_coeffs())

    def test_basic_structure(self, circle_ipq: ImplicitQuadrature) -> None:
        result = circle_ipq.surface_reparam(q=5, poly_idx=0)
        assert result.dim == 2  # noqa: PLR2004
        assert result.cell_dim == 1
        assert result.q == 5  # noqa: PLR2004
        assert result.pts_per_cell == 5  # noqa: PLR2004
        assert result.n_cells > 0
        assert result.points.shape == (result.n_cells * 5, 2)

    def test_nodes_on_levelset(self, circle_ipq: ImplicitQuadrature) -> None:
        """Surface nodes should satisfy |phi| < tol."""
        result = circle_ipq.surface_reparam(q=6, poly_idx=0)
        coeffs = _make_circle_coeffs()
        values = np.array(
            [_eval_bernstein_2d(coeffs, result.points[i]) for i in range(len(result.points))]
        )
        assert np.all(np.abs(values) < 1e-5)  # noqa: PLR2004

    def test_nodes_in_unit_square(self, circle_ipq: ImplicitQuadrature) -> None:
        result = circle_ipq.surface_reparam(q=4, poly_idx=0)
        assert np.all(result.points >= -1e-15)  # noqa: PLR2004
        assert np.all(result.points <= 1.0 + 1e-15)


# ---------------------------------------------------------------------------
# Boundary intersection (degenerate interval regression)
# ---------------------------------------------------------------------------


class TestDegenerateInterval:
    """Test geometries where the levelset intersects the [0,1]^2 boundary."""

    def test_hyperplane_volume_no_crash(self) -> None:
        """Hyperplane crossing boundary should not crash."""
        coeffs = _make_hyperplane_2d()
        iq = ImplicitQuadrature(coeffs)
        result = iq.volume_reparam(q=3, signs=[-1])
        assert result.n_cells >= 0
        # All nodes should be in [0,1]^2.
        if result.n_cells > 0:
            assert np.all(result.points >= -1e-12)  # noqa: PLR2004
            assert np.all(result.points <= 1.0 + 1e-12)

    def test_hyperplane_surface_no_crash(self) -> None:
        """Hyperplane surface reparameterization should not crash."""
        coeffs = _make_hyperplane_2d()
        iq = ImplicitQuadrature(coeffs)
        result = iq.surface_reparam(q=4, poly_idx=0)
        assert result.n_cells >= 0

    def test_hyperplane_surface_on_levelset(self) -> None:
        """Surface nodes should lie on the hyperplane."""
        coeffs = _make_hyperplane_2d()
        iq = ImplicitQuadrature(coeffs)
        result = iq.surface_reparam(q=6, poly_idx=0)
        if result.n_cells > 0:
            values = np.array(
                [_eval_bernstein_2d(coeffs, result.points[i]) for i in range(len(result.points))]
            )
            assert np.all(np.abs(values) < 1e-5)  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Multi-polynomial 2D
# ---------------------------------------------------------------------------


class TestMultiPolynomial2D:
    """Tests with two polynomials and combined sign conditions."""

    @pytest.fixture(scope="class")
    def two_poly_ipq(self) -> ImplicitQuadrature:
        """Two circles: one centered at (0.3, 0.5), one at (0.7, 0.5)."""
        from pantr.bezier.implicit import monomial_to_bernstein_2d  # noqa: PLC0415

        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        # (x-0.3)^2 + (y-0.5)^2 - 0.04
        mono1 = np.array(
            [
                [0.3**2 + 0.5**2 - 0.04, -1.0, 1.0],
                [-0.6, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        # (x-0.7)^2 + (y-0.5)^2 - 0.04
        mono2 = np.array(
            [
                [0.7**2 + 0.5**2 - 0.04, -1.0, 1.0],
                [-1.4, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        c1 = monomial_to_bernstein_2d(mono1, (2, 2), lo, hi)
        c2 = monomial_to_bernstein_2d(mono2, (2, 2), lo, hi)
        return ImplicitQuadrature(c1, c2)

    def test_both_negative(self, two_poly_ipq: ImplicitQuadrature) -> None:
        """Interior of both circles (intersection)."""
        result = two_poly_ipq.volume_reparam(q=3, signs=[-1, -1])
        # The two circles are non-overlapping, so intersection should be empty.
        assert result.n_cells == 0

    def test_first_negative_second_positive(self, two_poly_ipq: ImplicitQuadrature) -> None:
        """Inside first circle, outside second."""
        result = two_poly_ipq.volume_reparam(q=3, signs=[-1, +1])
        assert result.n_cells > 0

    def test_surface_with_sign_filter(self, two_poly_ipq: ImplicitQuadrature) -> None:
        """Surface of first circle, restricted to outside second."""
        result = two_poly_ipq.surface_reparam(q=4, poly_idx=0, signs=[0, +1])
        assert result.n_cells > 0


# ---------------------------------------------------------------------------
# 3D volume reparameterization
# ---------------------------------------------------------------------------


class TestVolumeReparam3D:
    """Tests for 3D volume reparameterization (Lagrange hexes)."""

    @pytest.fixture(scope="class")
    def sphere_ipq(self) -> ImplicitQuadrature:
        return ImplicitQuadrature(_make_sphere_coeffs())

    def test_basic_structure(self, sphere_ipq: ImplicitQuadrature) -> None:
        result = sphere_ipq.volume_reparam(q=3, signs=[-1])
        assert result.dim == 3  # noqa: PLR2004
        assert result.cell_dim == 3  # noqa: PLR2004
        assert result.q == 3  # noqa: PLR2004
        assert result.pts_per_cell == 27  # noqa: PLR2004
        assert result.n_cells > 0
        assert result.points.shape == (result.n_cells * 27, 3)

    def test_cell_midpoints_inside_domain(self, sphere_ipq: ImplicitQuadrature) -> None:
        """Cell midpoints (not boundary nodes) should be inside the domain.

        Individual nodes near cell boundaries may extend slightly outside
        the true curved domain, especially at low q.
        """
        result = sphere_ipq.volume_reparam(q=3, signs=[-1])
        coeffs = _make_sphere_coeffs()
        # Check the center node of each cell (index q^3 // 2 within cell).
        center_idx = result.pts_per_cell // 2
        for c in range(result.n_cells):
            idx = c * result.pts_per_cell + center_idx
            val = _eval_bernstein_3d(coeffs, result.points[idx])
            assert val < 0.0, f"Cell {c} center has phi={val}"

    def test_nodes_in_unit_cube(self, sphere_ipq: ImplicitQuadrature) -> None:
        result = sphere_ipq.volume_reparam(q=3, signs=[-1])
        assert np.all(result.points >= -1e-15)  # noqa: PLR2004
        assert np.all(result.points <= 1.0 + 1e-15)


# ---------------------------------------------------------------------------
# 3D surface reparameterization
# ---------------------------------------------------------------------------


class TestSurfaceReparam3D:
    """Tests for 3D surface reparameterization (Lagrange quads)."""

    @pytest.fixture(scope="class")
    def sphere_ipq(self) -> ImplicitQuadrature:
        return ImplicitQuadrature(_make_sphere_coeffs())

    def test_basic_structure(self, sphere_ipq: ImplicitQuadrature) -> None:
        result = sphere_ipq.surface_reparam(q=4, poly_idx=0)
        assert result.dim == 3  # noqa: PLR2004
        assert result.cell_dim == 2  # noqa: PLR2004
        assert result.q == 4  # noqa: PLR2004
        assert result.pts_per_cell == 16  # noqa: PLR2004
        assert result.n_cells > 0
        assert result.points.shape == (result.n_cells * 16, 3)

    def test_nodes_on_levelset(self, sphere_ipq: ImplicitQuadrature) -> None:
        result = sphere_ipq.surface_reparam(q=4, poly_idx=0)
        coeffs = _make_sphere_coeffs()
        values = np.array(
            [_eval_bernstein_3d(coeffs, result.points[i]) for i in range(len(result.points))]
        )
        assert np.all(np.abs(values) < 1e-4)  # noqa: PLR2004


# ---------------------------------------------------------------------------
# pyvista integration
# ---------------------------------------------------------------------------


class TestPyvistaIntegration:
    """Tests for the pyvista conversion."""

    @pytest.fixture(scope="class")
    def circle_ipq(self) -> ImplicitQuadrature:
        return ImplicitQuadrature(_make_circle_coeffs())

    def test_volume_to_pyvista(self, circle_ipq: ImplicitQuadrature) -> None:
        pv = pytest.importorskip("pyvista")  # noqa: F841
        from pantr.viz import implicit_to_pyvista  # noqa: PLC0415

        result = circle_ipq.volume_reparam(q=3, signs=[-1])
        grid = implicit_to_pyvista(result)
        assert grid.n_cells == result.n_cells
        assert grid.n_points == result.n_cells * result.pts_per_cell

    def test_surface_to_pyvista(self, circle_ipq: ImplicitQuadrature) -> None:
        pv = pytest.importorskip("pyvista")  # noqa: F841
        from pantr.viz import implicit_to_pyvista  # noqa: PLC0415

        result = circle_ipq.surface_reparam(q=4, poly_idx=0)
        grid = implicit_to_pyvista(result)
        assert grid.n_cells == result.n_cells
        assert grid.n_points == result.n_cells * result.pts_per_cell

    def test_3d_volume_to_pyvista(self) -> None:
        pv = pytest.importorskip("pyvista")  # noqa: F841
        from pantr.viz import implicit_to_pyvista  # noqa: PLC0415

        iq = ImplicitQuadrature(_make_sphere_coeffs())
        result = iq.volume_reparam(q=2, signs=[-1])
        grid = implicit_to_pyvista(result)
        assert grid.n_cells == result.n_cells

    def test_3d_surface_to_pyvista(self) -> None:
        pv = pytest.importorskip("pyvista")  # noqa: F841
        from pantr.viz import implicit_to_pyvista  # noqa: PLC0415

        iq = ImplicitQuadrature(_make_sphere_coeffs())
        result = iq.surface_reparam(q=3, poly_idx=0)
        grid = implicit_to_pyvista(result)
        assert grid.n_cells == result.n_cells


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for input validation."""

    def test_bad_signs_length(self) -> None:
        iq = ImplicitQuadrature(_make_circle_coeffs())
        with pytest.raises(ValueError, match="signs must have length"):
            iq.volume_reparam(q=3, signs=[-1, +1])

    def test_bad_signs_value(self) -> None:
        iq = ImplicitQuadrature(_make_circle_coeffs())
        with pytest.raises(ValueError, match="signs entries must be"):
            iq.volume_reparam(q=3, signs=[2])

    def test_bad_poly_idx(self) -> None:
        iq = ImplicitQuadrature(_make_circle_coeffs())
        with pytest.raises(IndexError, match="poly_idx"):
            iq.surface_reparam(q=3, poly_idx=5)

    def test_bad_node_type(self) -> None:
        iq = ImplicitQuadrature(_make_circle_coeffs())
        with pytest.raises(ValueError, match="Unknown node_type"):
            iq.volume_reparam(q=3, signs=[-1], node_type="bad")  # type: ignore[arg-type]
