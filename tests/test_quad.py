"""Tests for 1D quadrature rules in pantr.quad."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from math import gamma
from typing import Any, cast

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pytest
from numpy.polynomial import chebyshev

from pantr.basis import LagrangeVariant
from pantr.quad import (
    PointsLattice,
    QuadratureRule,
    create_lagrange_points_lattice,
    gauss_legendre_quadrature,
    get_chebyshev_gauss_1st_kind_1d,
    get_chebyshev_gauss_2nd_kind_1d,
    get_gauss_legendre_1d,
    get_gauss_lobatto_legendre_1d,
    get_modified_chebyshev_nodes_1d,
    get_trapezoidal_1d,
    tensor_product_quadrature,
)
from pantr.tolerance import get_conservative, get_default, get_machine_epsilon, get_strict


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

    @pytest.mark.slow
    @pytest.mark.parametrize("n_pts", [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 400])
    def test_agrees_with_an_independent_implementation(self, n_pts: int) -> None:
        """The rule stays close to ``numpy.polynomial.legendre.leggauss``.

        This test exists because the implementation stopped calling that
        function. Removing it removed the only independent Gauss-Legendre in the
        process, and the C++ backend is an operation-for-operation
        transliteration of the Python one, so a parity comparison between the two
        backends cannot see an error they would both make. Something has to
        compare against a rule this repository did not write.

        numpy's is genuinely independent in its arithmetic while agreeing in
        method: it takes the eigenvalues of the same symmetric tridiagonal matrix,
        applies one Newton polish, and evaluates the weights from a different but
        algebraically equal expression.

        The bounds are measurements of the two implementations against each
        other, not derivations, and they are stated in the frame each quantity
        lives in. The nodes agree absolutely, at 0.5 units of roundoff measured
        over every n from 1 to 1000; 8 leaves a factor 16. The weights agree
        absolutely and **not** relatively: their smallest entry decays like
        ``n^-2`` while the difference between two algebraically equal formulas
        does not, so the relative disagreement grows like ``n^2.7`` and reaches
        2.5e6 units of roundoff at n = 400. Measured absolute worst is 278 units
        of roundoff to n = 1000, and 1024 leaves a factor 3.7. See
        ``design/quadrature_algorithms.md`` for why that difference is the weight
        formula's own rounding rather than the nodes or the root finder.

        Args:
            n_pts (int): Number of quadrature points.
        """
        from numpy.polynomial import legendre  # noqa: PLC0415

        nodes, weights = get_gauss_legendre_1d(n_pts, np.float64)
        reference_nodes, reference_weights = legendre.leggauss(n_pts)
        # Both mapped onto [0, 1], the frame the public rule returns.
        reference_nodes = (reference_nodes + 1.0) * 0.5
        reference_weights = reference_weights * 0.5

        unit_roundoff = get_machine_epsilon(np.float64) / 2.0
        assert nodes.shape == reference_nodes.shape
        assert np.max(np.abs(nodes - reference_nodes)) <= 8.0 * unit_roundoff
        assert np.max(np.abs(weights - reference_weights)) <= 1024.0 * unit_roundoff

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

    def test_n_pts_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            get_gauss_lobatto_legendre_1d(0)

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

    def test_n_pts_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            get_chebyshev_gauss_2nd_kind_1d(0)

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            get_chebyshev_gauss_2nd_kind_1d(2, np.int32)

    @pytest.mark.parametrize("n_pts", [2, 5, 9])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_nodes_interior_and_total_weight(self, n_pts: int, dtype: npt.DTypeLike) -> None:
        nodes, weights = get_chebyshev_gauss_2nd_kind_1d(n_pts, dtype)
        # Gauss nodes are the mapped roots of U_{n_pts}: all interior, ascending.
        assert np.all((nodes > 0.0) & (nodes < 1.0))
        assert np.all(np.diff(nodes) > 0.0)
        # weights sum to (integral of sqrt(1-(2x-1)^2) over [0,1]) = pi/4
        nptest.assert_allclose(
            np.sum(weights), np.array(np.pi / 4.0, dtype=dtype), rtol=get_strict(dtype)
        )

    @pytest.mark.parametrize("n_pts", [2, 5, 9])
    def test_weighted_polynomial_exactness(self, n_pts: int) -> None:
        """The rule must be exact for x^d * sqrt(1-(2x-1)^2) up to d = 2*n_pts - 1."""
        nodes, weights = get_chebyshev_gauss_2nd_kind_1d(n_pts, np.float64)
        u = 2.0 * np.asarray(nodes, dtype=np.float64) - 1.0
        w = np.asarray(weights, dtype=np.float64)
        # int_{-1}^{1} u^d sqrt(1-u^2) du = sqrt(pi)/2 * gamma((d+1)/2) / gamma(d/2+2)
        # for even d (0 for odd d); the [0,1] mapping halves the value.
        for d in range(2 * n_pts):
            exact = (
                0.5 * np.sqrt(np.pi) / 2.0 * gamma((d + 1) / 2.0) / gamma(d / 2.0 + 2.0)
                if d % 2 == 0
                else 0.0
            )
            nptest.assert_allclose(np.sum(w * u**d), exact, atol=1e-14)


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

    # Immutability. Three separate holes, and freezing alone closes only two of
    # them: numpy's writeable flag governs the data and not the metadata, so the
    # in-place reshape below needs the property to hand out views instead.
    # QuadratureRule in the same package has made all three promises since it was
    # written, which is why these read as catching up rather than as new policy.

    def test_the_caller_s_array_is_not_aliased(self) -> None:
        source = np.array([0.0, 0.5, 1.0])
        lattice = PointsLattice([source])
        source[1] = 999.0
        nptest.assert_array_equal(lattice.pts_per_dir[0], [0.0, 0.5, 1.0])
        nptest.assert_array_equal(lattice.get_all_points().ravel(), [0.0, 0.5, 1.0])

    def test_the_exposed_arrays_are_read_only(self) -> None:
        lattice = PointsLattice([np.array([0.0, 0.5, 1.0])])
        assert not lattice.pts_per_dir[0].flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            lattice.pts_per_dir[0][0] = -7.0

    def test_reshaping_what_the_property_returned_does_not_reach_the_lattice(self) -> None:
        lattice = PointsLattice([np.array([0.0, 0.5, 1.0])])
        borrowed = lattice.pts_per_dir[0]
        borrowed.shape = (3, 1)
        assert lattice.pts_per_dir[0].ndim == 1
        assert lattice.dim == 1
        nptest.assert_array_equal(lattice.get_all_points().ravel(), [0.0, 0.5, 1.0])


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


class TestModifiedChebyshevNodes:
    """Tests for get_modified_chebyshev_nodes_1d."""

    def test_invalid_n_pts_raises(self) -> None:
        """n_pts < 2 must raise."""
        with pytest.raises(ValueError, match="at least 2"):
            get_modified_chebyshev_nodes_1d(1)

    def test_n_pts_zero_raises(self) -> None:
        """n_pts=0 must also raise with the min_pts=2 message."""
        with pytest.raises(ValueError, match="at least 2"):
            get_modified_chebyshev_nodes_1d(0)

    def test_invalid_dtype_raises(self) -> None:
        """Non-floating dtype must raise."""
        with pytest.raises(ValueError, match="float32 or float64"):
            get_modified_chebyshev_nodes_1d(3, np.int32)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_endpoints(self, dtype: npt.DTypeLike) -> None:
        """First node is 0, last node is 1."""
        nodes = get_modified_chebyshev_nodes_1d(5, dtype)
        nptest.assert_allclose(nodes[0], 0.0, atol=1e-15)
        nptest.assert_allclose(nodes[-1], 1.0, atol=1e-15)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_shape_and_dtype(self, dtype: npt.DTypeLike) -> None:
        """Output shape and dtype are correct."""
        n = 7
        nodes = get_modified_chebyshev_nodes_1d(n, dtype)
        assert nodes.shape == (n,)
        assert nodes.dtype == np.dtype(dtype)

    def test_two_points(self) -> None:
        """n_pts=2 gives [0, 1]."""
        nodes = get_modified_chebyshev_nodes_1d(2)
        nptest.assert_allclose(nodes, [0.0, 1.0], atol=1e-15)

    def test_three_points(self) -> None:
        """n_pts=3 gives [0, 0.5, 1]."""
        nodes = get_modified_chebyshev_nodes_1d(3)
        nptest.assert_allclose(nodes, [0.0, 0.5, 1.0], atol=1e-15)

    def test_symmetry(self) -> None:
        """Nodes are symmetric about 0.5: nodes[i] + nodes[n-1-i] == 1."""
        nodes = get_modified_chebyshev_nodes_1d(8)
        nptest.assert_allclose(nodes + nodes[::-1], 1.0, atol=1e-15)

    def test_monotonicity(self) -> None:
        """Nodes are strictly increasing."""
        nodes = get_modified_chebyshev_nodes_1d(10)
        assert np.all(np.diff(nodes) > 0)


class TestQuadratureRule:
    """Tests for the QuadratureRule value type."""

    def test_basic_properties(self) -> None:
        rule = QuadratureRule(points=[[0.25, 0.75], [0.5, 0.5]], weights=[0.4, 0.6])
        assert rule.ndim == 2
        assert rule.num_points == 2
        assert rule.points.shape == (2, 2)
        assert rule.weights.shape == (2,)

    def test_arrays_are_read_only(self) -> None:
        rule = QuadratureRule(points=[[0.5]], weights=[1.0])
        assert not rule.points.flags.writeable
        assert not rule.weights.flags.writeable

    def test_does_not_alias_input(self) -> None:
        pts = np.array([[0.5, 0.5]])
        rule = QuadratureRule(points=pts, weights=[1.0])
        pts[0, 0] = 0.1  # mutating the source must not affect the rule
        nptest.assert_array_equal(rule.points, [[0.5, 0.5]])

    def test_repr(self) -> None:
        assert repr(QuadratureRule([[0.5, 0.5]], [1.0])) == "QuadratureRule(ndim=2, num_points=1)"

    @pytest.mark.parametrize(
        ("points", "weights", "match"),
        [
            ([0.5, 0.5], [1.0], "points must be 2D"),
            ([[0.5]], [[1.0]], "weights must be 1D"),
            ([[0.5], [0.5]], [1.0], "must match the number of points"),
            (np.empty((0, 2)), [], "non-empty"),
            ([[np.inf, 0.5]], [1.0], "finite"),
            ([[0.5, 0.5]], [np.nan], "finite"),
            ([[-0.1, 0.5]], [1.0], "unit cube"),
            ([[0.5, 1.5]], [1.0], "unit cube"),
        ],
    )
    def test_validation(
        self,
        points: npt.ArrayLike,
        weights: npt.ArrayLike,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            QuadratureRule(points, weights)

    def test_endpoints_allowed(self) -> None:
        # Points exactly on the unit-cube boundary are valid (e.g. Lobatto).
        rule = QuadratureRule(points=[[0.0, 0.0], [1.0, 1.0]], weights=[0.5, 0.5])
        assert rule.num_points == 2

    def test_pickle_round_trip_reports_the_public_module(self) -> None:
        # `pantr.mpi` pickles rules for collective calls, and `__reduce__` names
        # `type(self)`, so the `__module__` rebinding in `pantr/quad/__init__.py`
        # is what keeps the pickle loadable. Under the C++ backend this also says
        # the wrapper pickles its arrays rather than its unpicklable handle;
        # `tests/parity/test_quad_rule.py` crosses the two backends.
        rule = gauss_legendre_quadrature(2, 3)
        restored = pickle.loads(pickle.dumps(rule))
        assert type(restored) is QuadratureRule
        assert restored.__module__ == "pantr.quad"
        nptest.assert_array_equal(restored.points, rule.points)
        nptest.assert_array_equal(restored.weights, rule.weights)

    def test_the_stored_arrays_are_the_same_object_on_every_read(self) -> None:
        # Not cosmetic: the C++ binding allocates a fresh copy per access, so an
        # uncached property would copy the whole table on every `rule.points[i]`.
        rule = gauss_legendre_quadrature(2, 3)
        assert rule.points is rule.points
        assert rule.weights is rule.weights


class TestTensorProductQuadrature:
    """Tests for tensor_product_quadrature."""

    def test_shape_and_count(self) -> None:
        rule = tensor_product_quadrature([get_gauss_legendre_1d(2), get_gauss_legendre_1d(3)])
        assert rule.ndim == 2
        assert rule.num_points == 6
        assert rule.points.shape == (6, 2)

    def test_c_order_last_axis_fastest(self) -> None:
        # Both axes need more than one node, or the claim is unobservable: this
        # test carried a single node on axis 1 and passed against a tensor product
        # whose odometer ran the other way round, which a mutation of the C++
        # factory exposed. Every value here is exact in binary, so the comparison
        # is an equality rather than a tolerance.
        rule = tensor_product_quadrature(
            [
                (np.array([0.25, 0.75]), np.array([0.25, 0.75])),
                (np.array([0.125, 0.5, 0.875]), np.array([0.5, 0.25, 0.25])),
            ]
        )
        nptest.assert_array_equal(
            rule.points,
            [
                [0.25, 0.125],
                [0.25, 0.5],
                [0.25, 0.875],
                [0.75, 0.125],
                [0.75, 0.5],
                [0.75, 0.875],
            ],
        )
        nptest.assert_array_equal(rule.weights, [0.125, 0.0625, 0.0625, 0.375, 0.1875, 0.1875])

    def test_weights_are_outer_product(self) -> None:
        rule = tensor_product_quadrature(
            [(np.array([0.5]), np.array([0.3])), (np.array([0.5]), np.array([0.4]))]
        )
        nptest.assert_allclose(rule.weights, [0.12])

    def test_single_axis(self) -> None:
        nodes, weights = get_gauss_legendre_1d(4)
        rule = tensor_product_quadrature([(nodes, weights)])
        assert rule.ndim == 1
        assert rule.num_points == 4
        nptest.assert_allclose(rule.points[:, 0], nodes)

    def test_integrates_polynomial(self) -> None:
        # int over [0,1]^2 of x^3 y = (1/4)(1/2) = 1/8.
        rule = tensor_product_quadrature([get_gauss_legendre_1d(2), get_gauss_legendre_1d(2)])
        val = float((rule.weights * rule.points[:, 0] ** 3 * rule.points[:, 1]).sum())
        nptest.assert_allclose(val, 1.0 / 8.0, rtol=1e-13)

    def test_empty_rules(self) -> None:
        with pytest.raises(ValueError, match="at least one axis"):
            tensor_product_quadrature([])

    def test_mismatched_nodes_weights(self) -> None:
        with pytest.raises(ValueError, match="matching non-empty"):
            tensor_product_quadrature([(np.array([0.5, 0.5]), np.array([1.0]))])


class TestGaussLegendreQuadrature:
    """Tests for gauss_legendre_quadrature."""

    def test_isotropic(self) -> None:
        rule = gauss_legendre_quadrature(3, 2)
        assert rule.ndim == 3
        assert rule.num_points == 8
        nptest.assert_allclose(rule.weights.sum(), 1.0, atol=1e-14)

    def test_anisotropic(self) -> None:
        rule = gauss_legendre_quadrature(2, [2, 4])
        assert rule.num_points == 8

    def test_exactness_degree(self) -> None:
        # n-point GL is exact to degree 2n-1; n=3 -> degree 5 per axis.
        rule = gauss_legendre_quadrature(2, 3)
        # int over [0,1]^2 of x^5 y^5 = (1/6)^2.
        val = float((rule.weights * rule.points[:, 0] ** 5 * rule.points[:, 1] ** 5).sum())
        nptest.assert_allclose(val, (1.0 / 6.0) ** 2, rtol=1e-13)

    def test_matches_manual_tensor_product(self) -> None:
        rule = gauss_legendre_quadrature(2, [2, 3])
        manual = tensor_product_quadrature([get_gauss_legendre_1d(2), get_gauss_legendre_1d(3)])
        nptest.assert_allclose(rule.points, manual.points)
        nptest.assert_allclose(rule.weights, manual.weights)

    @pytest.mark.parametrize(
        ("ndim", "npts", "match"),
        [
            (0, 2, "ndim must be >= 1"),
            (2, [2], "length-2 sequence"),
            (2, [2, 0], "must be >= 1"),
            (1, 0, "must be >= 1"),
        ],
    )
    def test_validation(self, ndim: int, npts: int | list[int], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            gauss_legendre_quadrature(ndim, npts)
