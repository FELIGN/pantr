"""Tests for cardinal B-spline basis on the central unit span."""

from __future__ import annotations

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pytest

from pantr.basis import tabulate_cardinal_bspline_1d
from pantr.tolerance import get_default


class TestCardinalBspline:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_degree_two_doc_example(self, dtype: npt.DTypeLike) -> None:
        pts = np.array([0.0, 0.5, 0.75, 1.0], dtype=dtype)
        res = tabulate_cardinal_bspline_1d(2, pts)
        exp = np.array(
            [
                [0.5, 0.5, 0.0],
                [0.125, 0.75, 0.125],
                [0.03125, 0.6875, 0.28125],
                [0.0, 0.5, 0.5],
            ],
            dtype=dtype,
        )
        rtol = get_default(dtype)
        nptest.assert_allclose(res, exp, rtol=rtol, atol=0.0)

    @pytest.mark.parametrize("degree", [0, 1, 2, 3, 6])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_partition_of_unity_on_span(self, degree: int, dtype: npt.DTypeLike) -> None:
        pts = np.linspace(0.0, 1.0, 21, dtype=dtype)
        res = tabulate_cardinal_bspline_1d(degree, pts)
        sums = np.sum(res, axis=-1)
        rtol = get_default(dtype)
        nptest.assert_allclose(sums, 1.0, rtol=rtol, atol=0.0)

    def test_nonnegativity(self) -> None:
        pts = np.linspace(0.0, 1.0, 51)
        res = tabulate_cardinal_bspline_1d(5, pts)
        assert np.all(res >= -get_default(res.dtype))

    def test_outside_span(self) -> None:
        tol = get_default(np.float64)
        pts = np.array(
            [
                -0.5,
                -tol,
                1.0 + tol,
                2.0,
            ],
            dtype=np.float64,
        )
        res = tabulate_cardinal_bspline_1d(3, pts)
        exp = np.array(
            [
                [5.62500000e-01, 3.54166667e-01, 1.04166667e-01, -2.08333333e-02],
                [1.66666667e-01, 6.66666667e-01, 1.66666667e-01, -1.66674107e-37],
                [-1.66711121e-37, 1.66666667e-01, 6.66666667e-01, 1.66666667e-01],
                [-1.66666667e-01, 6.66666667e-01, -8.33333333e-01, 1.33333333e00],
            ],
            dtype=np.float64,
        )
        nptest.assert_allclose(res, exp)

    def test_dtype_preservation(self) -> None:
        pts32 = np.array([0.0, 0.25, 0.5], dtype=np.float32)
        pts64 = np.array([0.0, 0.25, 0.5], dtype=np.float64)
        assert tabulate_cardinal_bspline_1d(3, pts32).dtype == np.float32
        assert tabulate_cardinal_bspline_1d(3, pts64).dtype == np.float64

    def test_2d_ndarray_shape_preservation(self) -> None:
        degree = 2
        pts = np.array([[0.0, 0.5], [0.25, 0.75]], dtype=np.float64)
        res = tabulate_cardinal_bspline_1d(degree, pts)
        # Original shape + basis dimension
        assert res.shape == (2, 2, degree + 1)
        sums = np.sum(res, axis=-1)
        np.testing.assert_allclose(sums, 1.0)

    def test_list_input(self) -> None:
        degree = 3
        pts = [[0.0, 0.25], [0.5, 0.75]]
        res = tabulate_cardinal_bspline_1d(degree, pts)
        assert res.shape == (2, 2, degree + 1)
        sums = np.sum(res, axis=-1)
        np.testing.assert_allclose(sums, 1.0)

    def test_tuple_input(self) -> None:
        degree = 2
        pts = (0.0, 0.5, 1.0)
        res = tabulate_cardinal_bspline_1d(degree, pts)
        assert res.shape == (3, degree + 1)
        sums = np.sum(res, axis=-1)
        np.testing.assert_allclose(sums, 1.0)

    def test_scalar_input(self) -> None:
        # Degree-0 should be 1 on [0, 1]
        res = tabulate_cardinal_bspline_1d(0, 0.5)
        np.testing.assert_allclose(res, np.array([1.0]))

    def test_negative_degree_raises(self) -> None:
        with pytest.raises(ValueError, match="degree must be non-negative"):
            tabulate_cardinal_bspline_1d(-1, [0.0, 0.5])
