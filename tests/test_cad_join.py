"""Tests for the join operation."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pantr.cad import create_bilinear, create_line, join


class TestJoinCurves:
    """Test joining 1D B-spline curves."""

    def test_join_two_line_segments(self) -> None:
        """Test joining two collinear line segments."""
        c1 = create_line([0, 0, 0], [1, 0, 0])
        c2 = create_line([1, 0, 0], [2, 0, 0])
        result = join(c1, c2, axis=0)
        assert result.dim == 1
        # Evaluate over the full domain
        domain = result.space.spaces[0].domain
        t = np.linspace(float(domain[0]), float(domain[1]), 11)
        pts = result.evaluate(t)
        expected_x = np.linspace(0, 2, 11)
        assert_allclose(pts[:, 0], expected_x, atol=1e-13)
        assert_allclose(pts[:, 1], 0.0, atol=1e-14)
        assert_allclose(pts[:, 2], 0.0, atol=1e-14)

    def test_join_preserves_geometry(self) -> None:
        """Test that join preserves the geometry of both segments."""
        c1 = create_line([0, 0, 0], [1, 1, 0])
        c2 = create_line([1, 1, 0], [3, 0, 0])
        result = join(c1, c2, axis=0)
        domain = result.space.spaces[0].domain
        pt_start = result.evaluate(np.array([float(domain[0])]))
        pt_end = result.evaluate(np.array([float(domain[1])]))
        assert_allclose(pt_start, [0, 0, 0], atol=1e-14)
        assert_allclose(pt_end, [3, 0, 0], atol=1e-14)

    def test_join_c0_at_junction(self) -> None:
        """Test C0 continuity at the junction point."""
        c1 = create_line([0, 0, 0], [1, 0, 0])
        c2 = create_line([1, 0, 0], [1, 1, 0])
        result = join(c1, c2, axis=0)
        # The junction should be at parameter u=1 (end of c1's domain)
        pt = result.evaluate(np.array([1.0]))
        assert_allclose(pt, [1, 0, 0], atol=1e-12)


class TestJoinSurfaces:
    """Test joining 2D B-spline surfaces."""

    def test_join_two_bilinear_patches(self) -> None:
        """Test joining two bilinear patches along axis 0."""
        corners1 = np.array([[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]]], dtype=np.float64)
        corners2 = np.array([[[1, 0, 0], [1, 1, 0]], [[2, 0, 0], [2, 1, 0]]], dtype=np.float64)
        s1 = create_bilinear(corners1)
        s2 = create_bilinear(corners2)
        result = join(s1, s2, axis=0)
        assert result.dim == 2
        domain_u = result.space.spaces[0].domain
        u_end = float(domain_u[1])
        pts = result.evaluate(
            np.array(
                [
                    [0.0, 0.0],
                    [u_end, 0.0],
                    [0.0, 1.0],
                    [u_end, 1.0],
                ]
            )
        )
        assert_allclose(pts[0], [0, 0, 0], atol=1e-13)
        assert_allclose(pts[1], [2, 0, 0], atol=1e-13)
        assert_allclose(pts[2], [0, 1, 0], atol=1e-13)
        assert_allclose(pts[3], [2, 1, 0], atol=1e-13)

    def test_join_along_axis_1(self) -> None:
        """Test joining two patches along the second axis."""
        corners1 = np.array([[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]]], dtype=np.float64)
        corners2 = np.array([[[0, 1, 0], [0, 2, 0]], [[1, 1, 0], [1, 2, 0]]], dtype=np.float64)
        s1 = create_bilinear(corners1)
        s2 = create_bilinear(corners2)
        result = join(s1, s2, axis=1)
        domain_v = result.space.spaces[1].domain
        v_end = float(domain_v[1])
        pt = result.evaluate(np.array([[0.5, v_end]]))
        assert_allclose(pt, [0.5, 2, 0], atol=1e-13)


class TestJoinErrors:
    """Test error handling in join."""

    def test_different_dim_raises(self) -> None:
        """Test that different dimensions raises ValueError."""
        crv = create_line([0, 0, 0], [1, 0, 0])
        srf = create_bilinear()
        with pytest.raises(ValueError, match="same dim"):
            join(crv, srf, axis=0)

    def test_axis_out_of_range_raises(self) -> None:
        """Test that out-of-range axis raises ValueError."""
        c1 = create_line([0, 0, 0], [1, 0, 0])
        c2 = create_line([1, 0, 0], [2, 0, 0])
        with pytest.raises(ValueError, match="axis"):
            join(c1, c2, axis=1)
