"""Tests for uniform sign detection of scalar non-rational Bézier polynomials."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bezier import Bezier
from pantr.bezier._bezier_sign import _uniform_sign

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
        assert _uniform_sign(_scalar_bezier([1.0, 2.0, 3.0])) == 1

    def test_all_negative(self) -> None:
        assert _uniform_sign(_scalar_bezier([-1.0, -2.0, -3.0])) == -1

    def test_mixed_signs(self) -> None:
        assert _uniform_sign(_scalar_bezier([-1.0, 2.0, -3.0])) == 0

    def test_zero_coefficient(self) -> None:
        assert _uniform_sign(_scalar_bezier([1.0, 0.0, 3.0])) == 0

    def test_all_zeros(self) -> None:
        assert _uniform_sign(_scalar_bezier([0.0, 0.0])) == 0

    def test_single_positive(self) -> None:
        assert _uniform_sign(_scalar_bezier([5.0])) == 1

    def test_single_negative(self) -> None:
        assert _uniform_sign(_scalar_bezier([-5.0])) == -1

    def test_single_zero(self) -> None:
        assert _uniform_sign(_scalar_bezier([0.0])) == 0

    def test_high_degree(self) -> None:
        coeffs = list(range(1, 21))  # degree 19, all positive
        assert _uniform_sign(_scalar_bezier(coeffs)) == 1

    def test_float32(self) -> None:
        b = _scalar_bezier([1.0, 2.0, 3.0], dtype=np.float32)
        assert _uniform_sign(b) == 1

    def test_negative_near_zero(self) -> None:
        """Tiny negative coefficient breaks uniform positive sign."""
        assert _uniform_sign(_scalar_bezier([1.0, 1e-15, 1.0])) == 1
        assert _uniform_sign(_scalar_bezier([1.0, -1e-15, 1.0])) == 0


class TestUniformSignND:
    """Uniform sign for multi-dimensional scalar Béziers."""

    def test_2d_all_positive(self) -> None:
        ctrl = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
        assert _uniform_sign(Bezier(ctrl)) == 1

    def test_2d_all_negative(self) -> None:
        ctrl = np.array([[[-1.0], [-2.0]], [[-3.0], [-4.0]]])
        assert _uniform_sign(Bezier(ctrl)) == -1

    def test_2d_mixed(self) -> None:
        ctrl = np.array([[[1.0], [-2.0]], [[3.0], [4.0]]])
        assert _uniform_sign(Bezier(ctrl)) == 0

    def test_3d_all_positive(self) -> None:
        ctrl = np.ones((2, 3, 4, 1), dtype=np.float64)
        assert _uniform_sign(Bezier(ctrl)) == 1


# ---------------------------------------------------------------------------
# Tests — invalid inputs
# ---------------------------------------------------------------------------


class TestUniformSignErrors:
    """Error handling for invalid Bézier inputs."""

    def test_rational_raises(self) -> None:
        b = Bezier(np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]), is_rational=True)
        with pytest.raises(TypeError, match="non-rational"):
            _uniform_sign(b)

    def test_vector_raises(self) -> None:
        b = Bezier(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        with pytest.raises(ValueError, match="rank == 1"):
            _uniform_sign(b)
