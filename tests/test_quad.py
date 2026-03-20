"""Tests for 1D quadrature rules in pantr.quad."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pytest
from numpy.polynomial import chebyshev

from pantr.basis import LagrangeVariant
from pantr.quad import (
    PointsLattice,
    create_lagrange_points_lattice,
    get_chebyshev_gauss_1st_kind_1d,
    get_chebyshev_gauss_2nd_kind_1d,
    get_gauss_legendre_1d,
    get_gauss_lobatto_legendre_1d,
    get_trapezoidal_1d,
)
from pantr.tolerance import get_conservative, get_default, get_strict


def _integrate_polynomial_on_unit_interval(
    power: int,
    nodes: npt.NDArray[np.floating[Any]],
    weights: npt.NDArray[np.floating[Any]],
) -> np.floating[Any]:
    vals = nodes**power
    result = np.sum(weights * vals, dtype=np.result_type(nodes.dtype, weights.dtype))
    return cast(np.floating[Any], result)


class TestTrapezoidal:
    """Tests for get_trapezoidal_1d."""

    def test_invalid_n_pts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            get_trapezoidal_1d(0)

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            get_trapezoidal_1d(2, np.int32)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_npts_one_midpoint_and_unit_weight(self, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_trapezoidal_1d(1, dtype)
        nptest.assert_allclose(nodes, np.array([0.5], dtype=dtype))
        nptest.assert_allclose(weights, np.array([1.0], dtype=dtype))
        assert nodes.dtype == np.dtype(dtype)
        assert weights.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("n_pts", [2, 5, 11])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_partition_and_end_weights(self, n_pts: int, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_trapezoidal_1d(n_pts, dtype)
        # nodes in [0, 1]
        assert np.all((nodes >= 0.0) & (nodes <= 1.0))
        # weights sum to 1
        nptest.assert_allclose(np.sum(weights), np.array(1.0, dtype=dtype))
        if n_pts > 1:
            h = np.array(1.0 / (n_pts - 1), dtype=dtype)
            nptest.assert_allclose(weights[1:-1], h)
            nptest.assert_allclose(weights[[0, -1]], 0.5 * h)


class TestGaussLegendre:
    """Tests for get_gauss_legendre_1d."""

    def test_invalid_n_pts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            get_gauss_legendre_1d(0)

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            get_gauss_legendre_1d(2, np.int32)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_basic_properties(self, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_gauss_legendre_1d(4, dtype)
        assert nodes.dtype == np.dtype(dtype)
        assert weights.dtype == np.dtype(dtype)
        assert np.all((nodes >= 0.0) & (nodes <= 1.0))
        assert np.all(weights > 0.0)
        nptest.assert_allclose(np.sum(weights, dtype=np.float64), 1.0, rtol=get_strict(dtype))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_polynomial_exactness(self, dtype: npt.DTypeLike) -> None:
        # n points should integrate polynomials up to degree 2n-1 exactly
        n = 4
        nodes, weights = get_gauss_legendre_1d(n, dtype)
        rtol = get_default(dtype)
        for p in range(2 * n):  # inclusive upper bound 2n-1
            approx = _integrate_polynomial_on_unit_interval(p, nodes, weights)
            exact = 1.0 / (p + 1)
            nptest.assert_allclose(approx, np.array(exact, dtype=dtype), rtol=rtol, atol=0.0)


class TestGaussLobattoLegendre:
    """Tests for get_gauss_lobatto_legendre_1d."""

    def test_invalid_n_pts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            get_gauss_lobatto_legendre_1d(1)

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            get_gauss_lobatto_legendre_1d(2, np.int32)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_endpoints_and_sum_weights(self, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_gauss_lobatto_legendre_1d(4, dtype)
        # Endpoints included
        nptest.assert_allclose(nodes[0], np.array(0.0, dtype=dtype))
        nptest.assert_allclose(nodes[-1], np.array(1.0, dtype=dtype))
        # weights positive and sum to 1
        assert np.all(weights > 0.0)
        nptest.assert_allclose(np.sum(weights, dtype=np.float64), 1.0, rtol=get_strict(dtype))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_polynomial_exactness(self, dtype: npt.DTypeLike) -> None:
        # Degree of exactness: 2n-3
        n = 5
        nodes, weights = get_gauss_lobatto_legendre_1d(n, dtype)
        rtol = get_conservative(dtype)
        for p in range(2 * n - 2):  # inclusive upper bound 2n-3
            approx = _integrate_polynomial_on_unit_interval(p, nodes, weights)
            exact = 1.0 / (p + 1)
            nptest.assert_allclose(approx, np.array(exact, dtype=dtype), rtol=rtol, atol=0.0)


class TestChebyshevGaussFirstKind:
    """Tests for get_chebyshev_gauss_1st_kind_1d."""

    def test_invalid_n_pts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            get_chebyshev_gauss_1st_kind_1d(0)

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            get_chebyshev_gauss_1st_kind_1d(2, np.int32)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_npts_one_midpoint_and_weight_sum(self, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_chebyshev_gauss_1st_kind_1d(1, dtype)
        # cheb1 at n=1 returns node 0 on [-1,1] which maps to 0.5
        nptest.assert_allclose(nodes, np.array([0.5], dtype=dtype))
        # Sum of weights equals integral of 1/sqrt(1-x^2) over [0,1] = pi/2
        nptest.assert_allclose(
            np.sum(weights), np.array(np.pi / 2.0, dtype=dtype), rtol=get_strict(dtype)
        )
        assert nodes.dtype == np.dtype(dtype)
        assert weights.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("n_pts", [2, 5, 10])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_nodes_and_total_weight(self, n_pts: int, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_chebyshev_gauss_1st_kind_1d(n_pts, dtype)
        # nodes are mapped chebpts1
        cheb1_t = cast(Callable[[int], npt.NDArray[np.float64]], chebyshev.chebpts1)
        mapped = ((cheb1_t(n_pts) + 1.0) * 0.5).astype(dtype)
        nptest.assert_allclose(nodes, mapped, rtol=get_strict(dtype))
        # weights sum to pi/2 after scaling to [0,1]
        nptest.assert_allclose(
            np.sum(weights), np.array(np.pi / 2.0, dtype=dtype), rtol=get_strict(dtype)
        )
        assert np.all((nodes >= 0.0) & (nodes <= 1.0))


class TestChebyshevGaussSecondKind:
    """Tests for get_chebyshev_gauss_2nd_kind_1d."""

    def test_invalid_n_pts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            get_chebyshev_gauss_2nd_kind_1d(1)

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            get_chebyshev_gauss_2nd_kind_1d(2, np.int32)

    @pytest.mark.parametrize("n_pts", [2, 5, 9])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_nodes_endpoints_and_total_weight(self, n_pts: int, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_chebyshev_gauss_2nd_kind_1d(n_pts, dtype)
        # endpoints included for n>=2
        nptest.assert_allclose(nodes[0], np.array(0.0, dtype=dtype))
        nptest.assert_allclose(nodes[-1], np.array(1.0, dtype=dtype))
        # weights sum to (integral of sqrt(1-x^2) over [0,1]) = pi/4 after scaling
        nptest.assert_allclose(
            np.sum(weights), np.array(np.pi / 4.0, dtype=dtype), rtol=get_strict(dtype)
        )
        assert np.all((nodes >= 0.0) & (nodes <= 1.0))


class TestPointsLattice:
    """Tests for PointsLattice class."""

    def test_empty_iterable_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1 dimension"):
            PointsLattice([])

    def test_different_dtypes_raises(self) -> None:
        pts1 = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        pts2 = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="same dtype"):
            PointsLattice([pts1, pts2])

    def test_non_1d_points_raises(self) -> None:
        pts1 = np.array([0.0, 0.5, 1.0])
        pts2 = np.array([[0.0, 0.5], [1.0, 1.5]])
        with pytest.raises(ValueError, match="must be 1D"):
            PointsLattice([pts1, pts2])

    def test_empty_points_raises(self) -> None:
        pts1 = np.array([0.0, 0.5, 1.0])
        pts2 = np.array([])
        with pytest.raises(ValueError, match="at least 1 point"):
            PointsLattice([pts1, pts2])

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_1d_lattice_properties(self, dtype: npt.DTypeLike) -> None:
        pts = np.array([0.0, 0.5, 1.0], dtype=dtype)
        lattice = PointsLattice([pts])
        assert lattice.dim == 1
        assert lattice.dtype == np.dtype(dtype)
        assert len(lattice.pts_per_dir) == 1
        nptest.assert_array_equal(lattice.pts_per_dir[0], pts)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_2d_lattice_properties(self, dtype: npt.DTypeLike) -> None:
        pts_x = np.array([0.0, 0.5, 1.0], dtype=dtype)
        pts_y = np.array([0.0, 1.0], dtype=dtype)
        pts_dir = [pts_x, pts_y]
        lattice = PointsLattice(pts_dir)
        assert lattice.dim == len(pts_dir)
        assert lattice.dtype == np.dtype(dtype)
        assert len(lattice.pts_per_dir) == len(pts_dir)
        nptest.assert_array_equal(lattice.pts_per_dir[0], pts_x)
        nptest.assert_array_equal(lattice.pts_per_dir[1], pts_y)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_3d_lattice_properties(self, dtype: npt.DTypeLike) -> None:
        pts_x = np.array([0.0, 1.0], dtype=dtype)
        pts_y = np.array([0.0, 0.5, 1.0], dtype=dtype)
        pts_z = np.array([0.0, 1.0], dtype=dtype)
        pts_dir = [pts_x, pts_y, pts_z]
        lattice = PointsLattice(pts_dir)
        assert lattice.dim == len(pts_dir)
        assert lattice.dtype == np.dtype(dtype)
        assert len(lattice.pts_per_dir) == len(pts_dir)
        nptest.assert_array_equal(lattice.pts_per_dir[0], pts_x)
        nptest.assert_array_equal(lattice.pts_per_dir[1], pts_y)
        nptest.assert_array_equal(lattice.pts_per_dir[2], pts_z)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_get_all_points_1d_c_order(self, dtype: npt.DTypeLike) -> None:
        pts = np.array([0.0, 0.5, 1.0], dtype=dtype)
        lattice = PointsLattice([pts])
        all_pts = lattice.get_all_points(order="C")
        assert all_pts.shape == (3, 1)
        nptest.assert_array_equal(all_pts[:, 0], pts)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_get_all_points_1d_f_order(self, dtype: npt.DTypeLike) -> None:
        pts = np.array([0.0, 0.5, 1.0], dtype=dtype)
        lattice = PointsLattice([pts])
        all_pts = lattice.get_all_points(order="F")
        assert all_pts.shape == (3, 1)
        nptest.assert_array_equal(all_pts[:, 0], pts)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_get_all_points_2d_c_order(self, dtype: npt.DTypeLike) -> None:
        pts_x = np.array([0.0, 1.0], dtype=dtype)
        pts_y = np.array([0.0, 0.5, 1.0], dtype=dtype)
        lattice = PointsLattice([pts_x, pts_y])
        all_pts = lattice.get_all_points(order="C")
        assert all_pts.shape == (6, 2)
        # C order: last index varies fastest
        expected = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.5],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.5],
                [1.0, 1.0],
            ],
            dtype=dtype,
        )
        nptest.assert_array_equal(all_pts, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_get_all_points_2d_f_order(self, dtype: npt.DTypeLike) -> None:
        pts_x = np.array([0.0, 1.0], dtype=dtype)
        pts_y = np.array([0.0, 0.5, 1.0], dtype=dtype)
        lattice = PointsLattice([pts_x, pts_y])
        all_pts = lattice.get_all_points(order="F")
        assert all_pts.shape == (6, 2)
        # F order: first index varies fastest
        expected = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.5],
                [1.0, 0.5],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=dtype,
        )
        nptest.assert_array_equal(all_pts, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_get_all_points_3d_c_order(self, dtype: npt.DTypeLike) -> None:
        pts_x = np.array([0.0, 1.0], dtype=dtype)
        pts_y = np.array([0.0, 1.0], dtype=dtype)
        pts_z = np.array([0.0, 1.0], dtype=dtype)
        lattice = PointsLattice([pts_x, pts_y, pts_z])
        all_pts = lattice.get_all_points(order="C")
        assert all_pts.shape == (8, 3)
        # C order: last index (z) varies fastest
        assert np.allclose(all_pts[0], [0.0, 0.0, 0.0])
        assert np.allclose(all_pts[1], [0.0, 0.0, 1.0])
        assert np.allclose(all_pts[2], [0.0, 1.0, 0.0])
        assert np.allclose(all_pts[3], [0.0, 1.0, 1.0])
        assert np.allclose(all_pts[4], [1.0, 0.0, 0.0])
        assert np.allclose(all_pts[5], [1.0, 0.0, 1.0])
        assert np.allclose(all_pts[6], [1.0, 1.0, 0.0])
        assert np.allclose(all_pts[7], [1.0, 1.0, 1.0])


class TestCreateLagrangePointsLattice:
    """Tests for create_lagrange_points_lattice function."""

    def test_invalid_n_pts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            create_lagrange_points_lattice(LagrangeVariant.EQUISPACES, [0])

    def test_invalid_n_pts_in_list_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            create_lagrange_points_lattice(LagrangeVariant.EQUISPACES, [2, 0, 3])

    @pytest.mark.parametrize("variant", list(LagrangeVariant))
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_1d_lattice_creation(self, variant: LagrangeVariant, dtype: npt.DTypeLike) -> None:
        n = [3]
        lattice = create_lagrange_points_lattice(variant, n, dtype)
        assert lattice.dim == len(n)
        assert lattice.dtype == np.dtype(dtype)
        assert len(lattice.pts_per_dir[0]) == n[0]
        # Points should be in [0, 1]
        assert np.all((lattice.pts_per_dir[0] >= 0.0) & (lattice.pts_per_dir[0] <= 1.0))

    @pytest.mark.parametrize("variant", list(LagrangeVariant))
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_2d_lattice_creation(self, variant: LagrangeVariant, dtype: npt.DTypeLike) -> None:
        n = [3, 4]
        lattice = create_lagrange_points_lattice(variant, n, dtype)
        assert lattice.dim == len(n)
        assert lattice.dtype == np.dtype(dtype)
        assert len(lattice.pts_per_dir[0]) == n[0]
        assert len(lattice.pts_per_dir[1]) == n[1]
        # Points should be in [0, 1]
        assert np.all((lattice.pts_per_dir[0] >= 0.0) & (lattice.pts_per_dir[0] <= 1.0))
        assert np.all((lattice.pts_per_dir[1] >= 0.0) & (lattice.pts_per_dir[1] <= 1.0))

    @pytest.mark.parametrize("variant", list(LagrangeVariant))
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_3d_lattice_creation(self, variant: LagrangeVariant, dtype: npt.DTypeLike) -> None:
        n = [2, 3, 4]
        lattice = create_lagrange_points_lattice(variant, n, dtype)
        assert lattice.dim == len(n)
        assert lattice.dtype == np.dtype(dtype)
        assert len(lattice.pts_per_dir[0]) == n[0]
        assert len(lattice.pts_per_dir[1]) == n[1]
        assert len(lattice.pts_per_dir[2]) == n[2]

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_equispaced_points(self, dtype: npt.DTypeLike) -> None:
        lattice = create_lagrange_points_lattice(LagrangeVariant.EQUISPACES, [4], dtype)
        expected = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=dtype)
        nptest.assert_allclose(lattice.pts_per_dir[0], expected, rtol=get_strict(dtype))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_gauss_lobatto_legendre_endpoints(self, dtype: npt.DTypeLike) -> None:
        lattice = create_lagrange_points_lattice(LagrangeVariant.GAUSS_LOBATTO_LEGENDRE, [4], dtype)
        pts = lattice.pts_per_dir[0]
        # GLL should include endpoints
        nptest.assert_allclose(pts[0], np.array(0.0, dtype=dtype), rtol=get_strict(dtype))
        nptest.assert_allclose(pts[-1], np.array(1.0, dtype=dtype), rtol=get_strict(dtype))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_get_all_points_from_lattice(self, dtype: npt.DTypeLike) -> None:
        lattice = create_lagrange_points_lattice(LagrangeVariant.EQUISPACES, [2, 3], dtype)
        all_pts = lattice.get_all_points(order="C")
        assert all_pts.shape == (6, 2)
        # Verify all points are in [0, 1]
        assert np.all((all_pts >= 0.0) & (all_pts <= 1.0))
