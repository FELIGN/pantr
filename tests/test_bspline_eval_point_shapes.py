"""No illegal point argument may escape Layer 2 into a Numba kernel.

The 1D ``Bspline`` evaluation path coerced nothing and checked no rank, so two
classes of bad input got past validation entirely:

* a ``(n_pts, 1)`` array reached the Cox-de Boor kernel and died inside it, with
  ``numba.core.errors.TypingError: No implementation of function Function(<class
  'int'>) found for signature: >>> int(array(int64, 1d, C))`` raised from
  ``_bspline_basis_core.py`` where the kernel does
  ``int(np.searchsorted(knots, pt, side="right")) - 1`` on what it assumes is a
  scalar. With the JIT disabled the same input surfaced as ``TypeError: only
  0-dimensional arrays can be converted to Python scalars``;
* an array-like such as a plain Python list failed even earlier, on
  ``AttributeError: 'list' object has no attribute 'dtype'`` (1D) or ``'ndim'``
  (nD) -- before reaching any validation at all, so not even a bad-input message.

``(n_pts, 1)`` is *accepted*, not rejected: it is what the general ``(n_pts, dim)``
convention degenerates to at ``dim == 1``, ``Bspline.locate`` and ``Grid`` already
take it, and the library's own documented example in
``_bspline_quasi_interpolation`` passes it. ``Bezier`` was the sibling that already
got this right, calling ``np.asarray`` at all four of its entry points.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy import typing as npt

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D

# A quadratic arc, so evaluation at an interior point is not degenerate.
_KNOTS = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
_PARAMS = np.array([0.3, 0.5, 0.7])


def _curve_1d(dtype: npt.DTypeLike = np.float64) -> Bspline:
    """Degree-2, rank-1 B-spline on ``[0, 1]``."""
    space = BsplineSpace([BsplineSpace1D(_KNOTS.astype(dtype), 2)])
    return Bspline(space, np.array([[0.0], [1.0], [0.0]], dtype=dtype))


def _curve_1d_rank2() -> Bspline:
    """Degree-2, rank-2 B-spline, to cover the vector-valued output shape."""
    space = BsplineSpace([BsplineSpace1D(_KNOTS, 2)])
    return Bspline(space, np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))


def _surface_2d() -> Bspline:
    """Degree-(2, 2), rank-1 B-spline on ``[0, 1]^2``."""
    space = BsplineSpace([BsplineSpace1D(_KNOTS, 2), BsplineSpace1D(_KNOTS, 2)])
    return Bspline(space, np.arange(9, dtype=np.float64).reshape(3, 3, 1))


class TestColumnShapeAccepted:
    """``(n_pts, 1)`` is the same request as ``(n_pts,)`` for a 1D B-spline."""

    @pytest.mark.parametrize("n_pts", [1, 3])
    def test_evaluate_column_matches_flat(self, n_pts: int) -> None:
        """A column of points must give bit-identical values to the flat form.

        Not merely close: the normalization is a reshape, so the kernel sees the
        same buffer and must produce the same floats.
        """
        curve = _curve_1d()
        flat = _PARAMS[:n_pts]

        from_flat = np.asarray(curve.evaluate(flat))
        from_column = np.asarray(curve.evaluate(flat.reshape(-1, 1)))

        assert from_column.shape == from_flat.shape
        np.testing.assert_array_equal(from_column, from_flat)

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_evaluate_derivatives_column_matches_flat(self, order: int) -> None:
        """The derivative sibling had the identical gap and must behave identically."""
        curve = _curve_1d()

        from_flat = np.asarray(curve.evaluate_derivatives(_PARAMS, order))
        from_column = np.asarray(curve.evaluate_derivatives(_PARAMS.reshape(-1, 1), order))

        assert from_column.shape == from_flat.shape
        np.testing.assert_array_equal(from_column, from_flat)

    def test_column_works_for_vector_valued_output(self) -> None:
        """Rank > 1 keeps its trailing axis; only the point axis is normalized."""
        curve = _curve_1d_rank2()

        from_flat = np.asarray(curve.evaluate(_PARAMS))
        from_column = np.asarray(curve.evaluate(_PARAMS.reshape(-1, 1)))

        assert from_column.shape == (3, 2)
        np.testing.assert_array_equal(from_column, from_flat)

    def test_the_documented_quasi_interpolation_example_runs(self) -> None:
        """The library's own docstring example passes ``(1, 1)`` and used to fail.

        ``_bspline_quasi_interpolation`` documents ``qi.evaluate(np.array([[0.3]]))``,
        which is the shape this fix accepts; the doctest raised the TypingError above.
        """
        from pantr.bspline import (  # noqa: PLC0415
            create_uniform_space,
            quasi_interpolate_bspline,
        )

        qi = quasi_interpolate_bspline(lambda p: p[:, 0] ** 2, create_uniform_space([2], [4]))

        assert float(np.asarray(qi.evaluate(np.array([[0.3]])))) == pytest.approx(0.09)


class TestArrayLikesCoerced:
    """An array-like must be converted, not met with an ``AttributeError``."""

    @pytest.mark.parametrize(
        "pts",
        [
            [0.3, 0.5, 0.7],
            [[0.3], [0.5], [0.7]],
            (0.3, 0.5, 0.7),
        ],
    )
    def test_evaluate_accepts_array_likes(self, pts: Any) -> None:
        """Lists and tuples, flat or as a column, give the ndarray answer."""
        curve = _curve_1d()

        np.testing.assert_array_equal(
            np.asarray(curve.evaluate(pts)), np.asarray(curve.evaluate(_PARAMS))
        )

    def test_evaluate_derivatives_accepts_array_likes(self) -> None:
        """The derivative path coerces too."""
        curve = _curve_1d()

        np.testing.assert_array_equal(
            np.asarray(curve.evaluate_derivatives([0.3, 0.5, 0.7], 1)),
            np.asarray(curve.evaluate_derivatives(_PARAMS, 1)),
        )

    def test_nd_evaluate_accepts_array_likes(self) -> None:
        """The nD path raised ``AttributeError: 'list' object has no attribute 'ndim'``."""
        surface = _surface_2d()
        pts = [[0.3, 0.4], [0.5, 0.6]]

        np.testing.assert_array_equal(
            np.asarray(surface.evaluate(pts)),
            np.asarray(surface.evaluate(np.asarray(pts, dtype=np.float64))),
        )

    def test_a_list_of_the_wrong_rank_gives_a_shape_error(self) -> None:
        """Coercion must not swallow a genuinely wrong shape."""
        surface = _surface_2d()

        with pytest.raises(ValueError, match="must be a 2D array with 2 columns"):
            surface.evaluate([0.3, 0.4])


class TestIllegalShapesRejectedInLayer2:
    """Every rejection must be a ``ValueError`` from Layer 2, never a kernel error."""

    @pytest.mark.parametrize(
        "pts",
        [
            np.array([[0.3, 0.4], [0.5, 0.6]]),  # (2, 2): trailing axis is not 1
            np.array([[[0.3]]]),  # (1, 1, 1): rank 3
            np.array(0.3),  # 0-d
        ],
    )
    @pytest.mark.parametrize("method", ["evaluate", "evaluate_derivatives"])
    def test_1d_rejects_unusable_shapes(self, pts: npt.NDArray[np.float64], method: str) -> None:
        """These are exactly the shapes that used to reach the kernel and fail there."""
        curve = _curve_1d()
        args = (1,) if method == "evaluate_derivatives" else ()

        with pytest.raises(ValueError, match=r"shape \(n_pts,\) or \(n_pts, 1\)"):
            getattr(curve, method)(pts, *args)

    @pytest.mark.parametrize("method", ["evaluate", "evaluate_derivatives"])
    def test_dtype_mismatch_is_still_rejected(self, method: str) -> None:
        """Coercion must not silently promote a mismatched dtype.

        Points are validated against the B-spline's dtype rather than cast, so a
        float32 array against a float64 spline stays an error. Widening the
        *accepted shapes* must not widen the accepted dtypes.
        """
        curve = _curve_1d()
        args = (1,) if method == "evaluate_derivatives" else ()

        with pytest.raises(ValueError, match="Points dtype must match B-spline dtype"):
            getattr(curve, method)(_PARAMS.astype(np.float32), *args)

    def test_integer_points_are_still_rejected(self) -> None:
        """An integer array is a dtype mismatch, before and after this change."""
        curve = _curve_1d()

        with pytest.raises(ValueError, match="Points dtype must match B-spline dtype"):
            curve.evaluate(np.array([0, 1]))

    def test_float32_spline_takes_float32_points_in_both_shapes(self) -> None:
        """The normalization is dtype-agnostic: a float32 space still works."""
        curve = _curve_1d(np.float32)
        pts = _PARAMS.astype(np.float32)

        np.testing.assert_array_equal(
            np.asarray(curve.evaluate(pts.reshape(-1, 1))), np.asarray(curve.evaluate(pts))
        )


class TestNoKernelErrorEscapes:
    """The acceptance criterion, stated directly."""

    @pytest.mark.parametrize(
        "pts",
        [
            [0.3],
            (0.3,),
            np.array([0.3]),
            np.array([[0.3]]),
            np.array([[0.3], [0.5]]),
            np.array([0]),
            np.array([0.3], dtype=np.float32),
            np.array(0.3),
            np.array([[0.3, 0.4]]),
            np.array([[[0.3]]]),
            np.empty((0,)),
            np.empty((0, 1)),
        ],
    )
    def test_no_input_produces_a_numba_or_attribute_error(self, pts: Any) -> None:
        """Whatever the input, the outcome is a result or a ``ValueError``.

        In particular never a ``numba.core.errors.TypingError``, never the
        ``TypeError: only 0-dimensional arrays can be converted to Python scalars``
        it degrades to with the JIT disabled, and never an ``AttributeError`` from
        Layer 2 poking at an attribute the input does not have.
        """
        curve = _curve_1d()

        try:
            curve.evaluate(pts)
        except ValueError:
            pass  # a validated rejection is the acceptable failure
        except Exception as exc:
            pytest.fail(f"{type(exc).__name__} escaped Layer 2 for pts={pts!r}: {exc}")
