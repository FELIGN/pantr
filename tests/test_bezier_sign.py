"""Tests for uniform sign detection of scalar Bézier polynomials."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bezier import Bezier
from pantr.bezier._bezier_sign import UniformSign, _uniform_sign

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scalar_bezier(
    ctrl: Sequence[float] | npt.NDArray[np.floating[Any]],
    dtype: type = np.float64,
) -> Bezier:
    """Create a 1D scalar Bézier from a flat list of coefficients."""
    return Bezier(np.array(ctrl, dtype=dtype))


# ---------------------------------------------------------------------------
# Tests — valid inputs
# ---------------------------------------------------------------------------


class TestUniformSign1D:
    """Uniform sign for 1D (univariate) scalar Béziers."""

    def test_all_positive(self) -> None:
        assert _uniform_sign(_scalar_bezier([1.0, 2.0, 3.0])) is UniformSign.POSITIVE

    def test_all_negative(self) -> None:
        assert _uniform_sign(_scalar_bezier([-1.0, -2.0, -3.0])) is UniformSign.NEGATIVE

    def test_mixed_signs(self) -> None:
        assert _uniform_sign(_scalar_bezier([-1.0, 2.0, -3.0])) is UniformSign.MIXED

    def test_zero_coefficient(self) -> None:
        assert _uniform_sign(_scalar_bezier([1.0, 0.0, 3.0])) is UniformSign.MIXED

    def test_all_zeros(self) -> None:
        assert _uniform_sign(_scalar_bezier([0.0, 0.0])) is UniformSign.MIXED

    def test_single_positive(self) -> None:
        assert _uniform_sign(_scalar_bezier([5.0])) is UniformSign.POSITIVE

    def test_single_negative(self) -> None:
        assert _uniform_sign(_scalar_bezier([-5.0])) is UniformSign.NEGATIVE

    def test_single_zero(self) -> None:
        assert _uniform_sign(_scalar_bezier([0.0])) is UniformSign.MIXED

    def test_high_degree(self) -> None:
        coeffs = list(range(1, 21))  # degree 19, all positive
        assert _uniform_sign(_scalar_bezier(coeffs)) is UniformSign.POSITIVE

    def test_float32(self) -> None:
        b = _scalar_bezier([1.0, 2.0, 3.0], dtype=np.float32)
        assert _uniform_sign(b) is UniformSign.POSITIVE

    def test_negative_near_zero(self) -> None:
        """Tiny negative coefficient breaks uniform positive sign."""
        assert _uniform_sign(_scalar_bezier([1.0, 1e-15, 1.0])) is UniformSign.POSITIVE
        assert _uniform_sign(_scalar_bezier([1.0, -1e-15, 1.0])) is UniformSign.MIXED

    def test_int_enum_comparison(self) -> None:
        """UniformSign values compare equal to their integer counterparts."""
        assert _uniform_sign(_scalar_bezier([1.0, 2.0])) == 1
        assert _uniform_sign(_scalar_bezier([-1.0, -2.0])) == -1
        assert _uniform_sign(_scalar_bezier([-1.0, 2.0])) == 0


class TestUniformSignND:
    """Uniform sign for multi-dimensional scalar Béziers."""

    def test_2d_all_positive(self) -> None:
        ctrl = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
        assert _uniform_sign(Bezier(ctrl)) is UniformSign.POSITIVE

    def test_2d_all_negative(self) -> None:
        ctrl = np.array([[[-1.0], [-2.0]], [[-3.0], [-4.0]]])
        assert _uniform_sign(Bezier(ctrl)) is UniformSign.NEGATIVE

    def test_2d_mixed(self) -> None:
        ctrl = np.array([[[1.0], [-2.0]], [[3.0], [4.0]]])
        assert _uniform_sign(Bezier(ctrl)) is UniformSign.MIXED

    def test_3d_all_positive(self) -> None:
        ctrl = np.ones((2, 3, 4, 1), dtype=np.float64)
        assert _uniform_sign(Bezier(ctrl)) is UniformSign.POSITIVE


# ---------------------------------------------------------------------------
# Tests — invalid inputs
# ---------------------------------------------------------------------------


class TestUniformSignRational:
    """Uniform sign for rational scalar Béziers."""

    def test_positive_numer_positive_weights(self) -> None:
        """Positive numerator and positive weights → POSITIVE."""
        b = Bezier(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), is_rational=True)
        assert _uniform_sign(b) is UniformSign.POSITIVE

    def test_negative_numer_positive_weights(self) -> None:
        """Negative numerator and positive weights → NEGATIVE."""
        b = Bezier(np.array([[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]]), is_rational=True)
        assert _uniform_sign(b) is UniformSign.NEGATIVE

    def test_positive_numer_negative_weights(self) -> None:
        """Positive numerator and negative weights → NEGATIVE (pos/neg)."""
        b = Bezier(np.array([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]]), is_rational=True)
        assert _uniform_sign(b) is UniformSign.NEGATIVE

    def test_negative_numer_negative_weights(self) -> None:
        """Negative numerator and negative weights → POSITIVE (neg/neg)."""
        b = Bezier(np.array([[-1.0, -1.0], [-2.0, -2.0], [-3.0, -3.0]]), is_rational=True)
        assert _uniform_sign(b) is UniformSign.POSITIVE

    def test_mixed_numer(self) -> None:
        """Mixed numerator → MIXED regardless of weights."""
        b = Bezier(np.array([[-1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]), is_rational=True)
        assert _uniform_sign(b) is UniformSign.MIXED

    def test_mixed_weights(self) -> None:
        """Uniform numerator but mixed weights → MIXED."""
        b = Bezier(np.array([[1.0, 1.0], [2.0, -1.0], [3.0, 1.0]]), is_rational=True)
        assert _uniform_sign(b) is UniformSign.MIXED

    def test_zero_weight(self) -> None:
        """A zero weight makes the denominator MIXED → MIXED."""
        b = Bezier(np.array([[1.0, 1.0], [2.0, 0.0], [3.0, 1.0]]), is_rational=True)
        assert _uniform_sign(b) is UniformSign.MIXED

    def test_zero_numer(self) -> None:
        """A zero numerator coefficient → MIXED."""
        b = Bezier(np.array([[0.0, 1.0], [2.0, 1.0], [3.0, 1.0]]), is_rational=True)
        assert _uniform_sign(b) is UniformSign.MIXED


class TestUniformSignErrors:
    """Error handling for invalid Bézier inputs."""

    def test_vector_raises(self) -> None:
        b = Bezier(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        with pytest.raises(ValueError, match="rank == 1"):
            _uniform_sign(b)
