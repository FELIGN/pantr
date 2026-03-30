"""Tests for Coons surface and volume constructions."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pantr.bspline import Bspline
from pantr.cad import create_bilinear, create_coons_surface, create_coons_volume, create_line

_RANK_3D = 3


class TestCoonsSurface:
    """Test the coons_surface function."""

    def test_four_straight_lines_gives_bilinear(self) -> None:
        """Test Coons from 4 straight lines matches bilinear."""
        c_u0 = create_line([0, 0, 0], [1, 0, 0])
        c_u1 = create_line([0, 1, 0], [1, 1, 0])
        c_v0 = create_line([0, 0, 0], [0, 1, 0])
        c_v1 = create_line([1, 0, 0], [1, 1, 0])

        srf = create_coons_surface(((c_v0, c_v1), (c_u0, c_u1)))
        assert srf.dim == 2  # noqa: PLR2004
        assert srf.rank == _RANK_3D

        # Should match bilinear
        corners = np.array(
            [[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]]],
            dtype=np.float64,
        )
        ref = create_bilinear(corners)
        t = np.linspace(0, 1, 10)
        params = np.array([[u, v] for u in t for v in t])
        assert_allclose(srf.evaluate(params), ref.evaluate(params), atol=1e-13)

    def test_evaluate_corners(self) -> None:
        """Test that Coons surface evaluates correctly at corners."""
        c_u0 = create_line([0, 0, 0], [3, 0, 0])
        c_u1 = create_line([0, 2, 0], [3, 2, 0])
        c_v0 = create_line([0, 0, 0], [0, 2, 0])
        c_v1 = create_line([3, 0, 0], [3, 2, 0])

        srf = create_coons_surface(((c_v0, c_v1), (c_u0, c_u1)))
        pts = srf.evaluate(
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ]
            )
        )
        assert_allclose(pts[0], [0, 0, 0], atol=1e-14)
        assert_allclose(pts[1], [3, 0, 0], atol=1e-14)
        assert_allclose(pts[2], [0, 2, 0], atol=1e-14)
        assert_allclose(pts[3], [3, 2, 0], atol=1e-14)

    def test_non_planar_coons(self) -> None:
        """Test Coons with non-planar boundaries."""
        c_u0 = create_line([0, 0, 0], [1, 0, 0])
        c_u1 = create_line([0, 0, 1], [1, 0, 1])
        c_v0 = create_line([0, 0, 0], [0, 0, 1])
        c_v1 = create_line([1, 0, 0], [1, 0, 1])

        srf = create_coons_surface(((c_v0, c_v1), (c_u0, c_u1)))
        # Should be a flat rectangle in the xz plane
        pt = srf.evaluate(np.array([[0.5, 0.5]]))
        assert_allclose(pt, [0.5, 0, 0.5], atol=1e-13)

    def test_non_1d_curve_raises(self) -> None:
        """Test that a surface as input raises ValueError."""
        srf = create_bilinear()
        crv = create_line([0, 0, 0], [1, 0, 0])
        with pytest.raises(ValueError, match="1D"):
            create_coons_surface(((crv, crv), (srf, crv)))

    def test_corner_mismatch_raises(self) -> None:
        """Test that inconsistent corners raise ValueError."""
        c_u0 = create_line([0, 0, 0], [1, 0, 0])
        c_u1 = create_line([0, 1, 0], [1, 1, 0])
        c_v0 = create_line([0, 0, 0], [0, 1, 0])
        c_v1 = create_line([2, 0, 0], [2, 1, 0])  # wrong: should start at (1,0,0)

        with pytest.raises(ValueError, match="mismatch"):
            create_coons_surface(((c_v0, c_v1), (c_u0, c_u1)))


class TestCoonsVolume:
    """Test the coons_volume function."""

    def _make_cube_faces(
        self,
    ) -> tuple[
        tuple[Bspline, Bspline],
        tuple[Bspline, Bspline],
        tuple[Bspline, Bspline],
    ]:
        """Build 6 planar faces of the unit cube [0,1]^3."""
        # face_u0: u=0 plane, parameterized by (v, w)
        face_u0 = create_bilinear(
            np.array([[[0, 0, 0], [0, 0, 1]], [[0, 1, 0], [0, 1, 1]]], dtype=np.float64)
        )
        # face_u1: u=1 plane
        face_u1 = create_bilinear(
            np.array([[[1, 0, 0], [1, 0, 1]], [[1, 1, 0], [1, 1, 1]]], dtype=np.float64)
        )
        # face_v0: v=0 plane, parameterized by (u, w)
        face_v0 = create_bilinear(
            np.array([[[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [1, 0, 1]]], dtype=np.float64)
        )
        # face_v1: v=1 plane
        face_v1 = create_bilinear(
            np.array([[[0, 1, 0], [0, 1, 1]], [[1, 1, 0], [1, 1, 1]]], dtype=np.float64)
        )
        # face_w0: w=0 plane, parameterized by (u, v)
        face_w0 = create_bilinear(
            np.array([[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]]], dtype=np.float64)
        )
        # face_w1: w=1 plane
        face_w1 = create_bilinear(
            np.array([[[0, 0, 1], [0, 1, 1]], [[1, 0, 1], [1, 1, 1]]], dtype=np.float64)
        )
        return (face_u0, face_u1), (face_v0, face_v1), (face_w0, face_w1)

    def test_cube_properties(self) -> None:
        """Test that Coons volume from cube faces has correct properties."""
        faces = self._make_cube_faces()
        vol = create_coons_volume(faces)
        assert vol.dim == _RANK_3D
        assert vol.rank == _RANK_3D

    def test_cube_evaluate_center(self) -> None:
        """Test evaluation at the center of the unit cube."""
        faces = self._make_cube_faces()
        vol = create_coons_volume(faces)
        pt = vol.evaluate(np.array([[0.5, 0.5, 0.5]]))
        assert_allclose(pt, [0.5, 0.5, 0.5], atol=1e-13)

    def test_cube_evaluate_corners(self) -> None:
        """Test evaluation at all 8 corners of the unit cube."""
        faces = self._make_cube_faces()
        vol = create_coons_volume(faces)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    pt = vol.evaluate(np.array([[float(i), float(j), float(k)]]))
                    assert_allclose(pt, [float(i), float(j), float(k)], atol=1e-13)

    def test_cube_boundaries_match_faces(self) -> None:
        """Test that volume boundaries match the input faces."""
        faces = self._make_cube_faces()
        vol = create_coons_volume(faces)
        # Evaluate on the u=0 face
        t = np.linspace(0, 1, 5)
        params_face_u0 = np.array([[0.0, v, w] for v in t for w in t])
        pts_vol = vol.evaluate(params_face_u0)
        # Compare with face_u0 evaluation
        params_face = np.array([[v, w] for v in t for w in t])
        (face_u0, _), _, _ = faces
        pts_face = face_u0.evaluate(params_face)
        assert_allclose(pts_vol, pts_face, atol=1e-12)

    def test_non_2d_face_raises(self) -> None:
        """Test that a 1D input raises ValueError."""
        crv = create_line([0, 0, 0], [1, 0, 0])
        srf = create_bilinear()
        with pytest.raises(ValueError, match="2D"):
            create_coons_volume(((crv, crv), (srf, srf), (srf, srf)))
