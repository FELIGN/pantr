"""Tests for VTK Bézier point ordering (pure NumPy, no pyvista required)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy import typing as npt

from pantr.viz._vtk_ordering import (
    vtk_ordering,
    vtk_ordering_curve,
    vtk_ordering_hex,
    vtk_ordering_quad,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_valid_permutation(perm: npt.NDArray[np.intp], n: int) -> bool:
    """Check that *perm* is a valid permutation of 0..n-1."""
    return perm.shape == (n,) and set(perm.tolist()) == set(range(n))


# ---------------------------------------------------------------------------
# 1D — Curves
# ---------------------------------------------------------------------------


class TestVtkOrderingCurve:
    """Tests for vtk_ordering_curve."""

    def test_degree_1(self) -> None:
        perm = vtk_ordering_curve(1)
        n_pts = 2
        assert _is_valid_permutation(perm, n_pts)
        np.testing.assert_array_equal(perm[:n_pts], [0, 1])

    def test_degree_2(self) -> None:
        perm = vtk_ordering_curve(2)
        n_pts = 3
        assert _is_valid_permutation(perm, n_pts)
        np.testing.assert_array_equal(perm, [0, 2, 1])

    def test_degree_3(self) -> None:
        perm = vtk_ordering_curve(3)
        n_pts = 4
        assert _is_valid_permutation(perm, n_pts)
        np.testing.assert_array_equal(perm, [0, 3, 1, 2])

    def test_degree_5(self) -> None:
        degree = 5
        perm = vtk_ordering_curve(degree)
        n_pts = degree + 1
        assert _is_valid_permutation(perm, n_pts)
        assert perm[0] == 0
        assert perm[1] == degree
        np.testing.assert_array_equal(perm[2:], [1, 2, 3, 4])

    def test_corners_first(self) -> None:
        for p in range(1, 8):
            perm = vtk_ordering_curve(p)
            assert perm[0] == 0, f"degree={p}: first corner should be 0"
            assert perm[1] == p, f"degree={p}: second corner should be {p}"


# ---------------------------------------------------------------------------
# 2D — Quads
# ---------------------------------------------------------------------------


class TestVtkOrderingQuad:
    """Tests for vtk_ordering_quad."""

    def test_is_valid_permutation(self) -> None:
        for pu in range(1, 5):
            for pv in range(1, 5):
                perm = vtk_ordering_quad(pu, pv)
                n = (pu + 1) * (pv + 1)
                assert _is_valid_permutation(perm, n), f"degree=({pu},{pv})"

    def test_degree_1x1(self) -> None:
        perm = vtk_ordering_quad(1, 1)
        n_corners = 4
        assert len(perm) == n_corners
        # Corners: (0,0), (1,0), (1,1), (0,1) in flat indices with shape (2,2)
        # (0,0)=0, (1,0)=2, (1,1)=3, (0,1)=1
        np.testing.assert_array_equal(perm, [0, 2, 3, 1])

    def test_corners_degree_2x2(self) -> None:
        perm = vtk_ordering_quad(2, 2)
        n_pts = 9
        assert len(perm) == n_pts
        # Corners: (0,0), (2,0), (2,2), (0,2) in shape (3,3)
        # (0,0)=0, (2,0)=6, (2,2)=8, (0,2)=2
        np.testing.assert_array_equal(perm[:4], [0, 6, 8, 2])

    def test_corners_are_first_four(self) -> None:
        """First 4 entries must be the 4 corner indices."""
        for pu in range(1, 5):
            for pv in range(1, 5):
                perm = vtk_ordering_quad(pu, pv)
                expected_corners = {
                    0,  # (0, 0)
                    pu * (pv + 1),  # (pu, 0)
                    pu * (pv + 1) + pv,  # (pu, pv)
                    pv,  # (0, pv)
                }
                actual_corners = set(perm[:4].tolist())
                assert actual_corners == expected_corners, f"degree=({pu},{pv})"

    def test_edge_count(self) -> None:
        """Check the number of edge interior points."""
        for pu in range(1, 5):
            for pv in range(1, 5):
                n_edge_interior = 2 * (pu - 1) + 2 * (pv - 1)
                n_face_interior = (pu - 1) * (pv - 1)
                n_total = (pu + 1) * (pv + 1)
                n_corners = 4
                assert n_corners + n_edge_interior + n_face_interior == n_total


# ---------------------------------------------------------------------------
# 3D — Hexahedra
# ---------------------------------------------------------------------------


class TestVtkOrderingHex:
    """Tests for vtk_ordering_hex."""

    def test_is_valid_permutation(self) -> None:
        for pu in range(1, 4):
            for pv in range(1, 4):
                for pw in range(1, 4):
                    perm = vtk_ordering_hex(pu, pv, pw)
                    n = (pu + 1) * (pv + 1) * (pw + 1)
                    assert _is_valid_permutation(perm, n), f"degree=({pu},{pv},{pw})"

    def test_degree_1x1x1(self) -> None:
        n_corners = 8
        perm = vtk_ordering_hex(1, 1, 1)
        assert len(perm) == n_corners
        # shape (2, 2, 2): flat index = i*4 + j*2 + k
        # VTK corners: (0,0,0)=0, (1,0,0)=4, (1,1,0)=6, (0,1,0)=2,
        #              (0,0,1)=1, (1,0,1)=5, (1,1,1)=7, (0,1,1)=3
        np.testing.assert_array_equal(perm, [0, 4, 6, 2, 1, 5, 7, 3])

    def test_corners_are_first_eight(self) -> None:
        """First 8 entries must be the 8 corner indices."""
        for pu in range(1, 4):
            for pv in range(1, 4):
                for pw in range(1, 4):
                    perm = vtk_ordering_hex(pu, pv, pw)
                    nv = pv + 1
                    nw = pw + 1
                    expected_corners = set()
                    for i in (0, pu):
                        for j in (0, pv):
                            for k in (0, pw):
                                expected_corners.add(i * nv * nw + j * nw + k)
                    actual_corners = set(perm[:8].tolist())
                    assert actual_corners == expected_corners, f"degree=({pu},{pv},{pw})"

    def test_total_count(self) -> None:
        """Verify count decomposition: corners + edges + faces + interior."""
        n_corners = 8
        for pu in range(1, 4):
            for pv in range(1, 4):
                for pw in range(1, 4):
                    n_edges = 4 * (pu - 1) + 4 * (pv - 1) + 4 * (pw - 1)
                    n_faces = (
                        2 * (pu - 1) * (pv - 1) + 2 * (pu - 1) * (pw - 1) + 2 * (pv - 1) * (pw - 1)
                    )
                    n_interior = (pu - 1) * (pv - 1) * (pw - 1)
                    n_total = (pu + 1) * (pv + 1) * (pw + 1)
                    assert n_corners + n_edges + n_faces + n_interior == n_total


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestVtkOrdering:
    """Tests for the vtk_ordering dispatch function."""

    def test_1d(self) -> None:
        np.testing.assert_array_equal(vtk_ordering((3,)), vtk_ordering_curve(3))

    def test_2d(self) -> None:
        np.testing.assert_array_equal(vtk_ordering((2, 3)), vtk_ordering_quad(2, 3))

    def test_3d(self) -> None:
        np.testing.assert_array_equal(vtk_ordering((2, 2, 2)), vtk_ordering_hex(2, 2, 2))

    def test_invalid_dim(self) -> None:
        with pytest.raises(ValueError, match="Unsupported parametric dimension"):
            vtk_ordering((1, 1, 1, 1))
