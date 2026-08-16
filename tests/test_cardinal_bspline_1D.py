"""Tests for cardinal B-spline basis on the central unit span."""

from __future__ import annotations

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pytest

from pantr.basis import tabulate_cardinal_bspline_1d
from pantr.tolerance import get_default, get_strict


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
        """Points off the central span extrapolate the cubic polynomials, they are not clamped.

        Checked against the closed form of the four uniform cubic B-splines on the
        central span rather than against frozen digits, which is an independent
        oracle (the kernel runs a de Boor recurrence) and does not move when a
        tolerance preset does. Two of the sampled points sit ``get_default`` outside
        the span -- dimensionally sound here because the span is ``[0, 1]``, so the
        dimensionless preset is already in the units of the parameter.
        """
        tol = get_default(np.float64)
        pts = np.array([-0.5, -tol, 1.0 + tol, 2.0], dtype=np.float64)
        res = tabulate_cardinal_bspline_1d(3, pts)

        u = pts
        exp = (
            np.stack(
                [
                    (1.0 - u) ** 3,
                    3.0 * u**3 - 6.0 * u**2 + 4.0,
                    -3.0 * u**3 + 3.0 * u**2 + 3.0 * u + 1.0,
                    u**3,
                ],
                axis=-1,
            )
            / 6.0
        )

        # The basis values are O(1) and reach the output through `degree + 1` levels
        # of convex combination, so the strict preset -- four roundings -- times that
        # magnitude is the bound. It is an absolute bound because the two entries
        # nearest the span ends are ~1e-43, far below the noise of an O(1)
        # computation, and pinning their digits would be pinning noise.
        nptest.assert_allclose(res, exp, rtol=get_strict(np.float64), atol=get_strict(np.float64))

        # That `atol` is twenty-one orders above the two entries at the span ends, so
        # it says nothing about them. They are the whole reason for sampling at
        # `+/- tol`: the point is *outside*, so the outermost basis function must be
        # small, negative -- and not clamped to zero, which is what treating the point
        # as the boundary would produce. Checked relatively, against `-tol**3 / 6`,
        # which both of them are: at `-tol` the last basis function is `u**3 / 6` and
        # at `1 + tol` the first is `(1 - u)**3 / 6`.
        #
        # 5% is as tight as this can be, and the reason is worth stating: a value of
        # order `tol**3` is being produced by a recurrence whose inputs are O(1), so
        # its relative accuracy is limited by how exactly those inputs reproduce `u`,
        # not by the recurrence. The measured deviation is 1/64 at `-tol` and nothing
        # at `1 + tol`. That is still four orders of margin over what this is for --
        # separating a cubic extrapolation from a clamp to zero.
        for row, col in ((1, 3), (2, 0)):
            outside = float(res[row, col])
            assert outside < 0.0, f"row {row} was clamped to {outside!r} instead of extrapolating"
            nptest.assert_allclose(outside, -(tol**3) / 6.0, rtol=0.05)

        # Extrapolation, not clamping: outside the span the polynomials leave [0, 1].
        assert res[0].min() < 0.0
        assert res[3].max() > 1.0

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
