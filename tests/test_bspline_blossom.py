"""Tests for B-spline blossom (polar form) evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from pantr._bspline_blossom import _evaluate_blossom_1d
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval_blossom(
    knots: list[float],
    degree: int,
    ctrl: list[float] | list[list[float]],
    u_values: list[float],
) -> np.ndarray:  # type: ignore[type-arg]
    """Thin wrapper: build numpy arrays and call _evaluate_blossom_1d."""
    kv = np.array(knots, dtype=np.float64)
    cp = np.array(ctrl, dtype=np.float64)
    if cp.ndim == 1:
        cp = cp[:, np.newaxis]
    u = np.array(u_values, dtype=np.float64)
    tol = float(np.finfo(np.float64).eps * max(1.0, float(np.abs(kv).max())) * 64)
    return _evaluate_blossom_1d(kv, degree, cp, u, tol)


def _eval_spline(
    knots: list[float],
    degree: int,
    ctrl: list[float],
    t: float,
) -> float:
    """Evaluate a 1D B-spline at a single point via de Boor."""
    kv = np.array(knots, dtype=np.float64)
    cp = np.array(ctrl, dtype=np.float64)[:, np.newaxis]
    space = BsplineSpace([BsplineSpace1D(kv, degree)])
    f = Bspline(space, cp)
    return float(f.evaluate(np.array([t])))


# ---------------------------------------------------------------------------
# Diagonal property: blossom(t, t, ..., t) == f(t)
# ---------------------------------------------------------------------------


class TestBlossomDiagonal:
    """Test that the diagonal of the blossom equals the B-spline evaluation."""

    def test_degree2_single_element(self) -> None:
        """Diagonal property for a degree-2 single-element B-spline."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        ctrl = [1.0, 2.0, 3.0]

        for t in np.linspace(0.0, 1.0, 9):
            blossom_val = _eval_blossom(knots, 2, ctrl, [t, t])
            direct_val = _eval_spline(knots, 2, ctrl, t)
            np.testing.assert_allclose(blossom_val[0], direct_val, atol=1e-12, err_msg=f"t={t}")

    def test_degree3_two_elements(self) -> None:
        """Diagonal property for a cubic B-spline with one interior knot."""
        knots = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
        ctrl = [1.0, 0.5, 2.0, 1.5, 0.5]

        for t in np.linspace(0.0, 1.0, 11):
            blossom_val = _eval_blossom(knots, 3, ctrl, [t, t, t])
            direct_val = _eval_spline(knots, 3, ctrl, t)
            np.testing.assert_allclose(blossom_val[0], direct_val, atol=1e-12, err_msg=f"t={t}")

    def test_degree1_linear(self) -> None:
        """Diagonal property for a degree-1 B-spline (piecewise linear)."""
        knots = [0.0, 0.0, 0.5, 1.0, 1.0]
        ctrl = [1.0, 3.0, 2.0]

        for t in np.linspace(0.0, 1.0, 9):
            blossom_val = _eval_blossom(knots, 1, ctrl, [t])
            direct_val = _eval_spline(knots, 1, ctrl, t)
            np.testing.assert_allclose(blossom_val[0], direct_val, atol=1e-12, err_msg=f"t={t}")

    def test_vector_rank(self) -> None:
        """Diagonal property holds for each component of a vector-valued spline."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        ctrl = [[1.0, 2.0], [3.0, 0.5], [2.0, 1.5]]

        for t in np.linspace(0.0, 1.0, 7):
            blossom_val = _eval_blossom(knots, 2, ctrl, [t, t])
            # Evaluate each component separately
            val0 = _eval_spline(knots, 2, [row[0] for row in ctrl], t)
            val1 = _eval_spline(knots, 2, [row[1] for row in ctrl], t)
            np.testing.assert_allclose(blossom_val[0], val0, atol=1e-12)
            np.testing.assert_allclose(blossom_val[1], val1, atol=1e-12)


# ---------------------------------------------------------------------------
# Symmetry: blossom(u1, u2) == blossom(u2, u1)
# ---------------------------------------------------------------------------


class TestBlossomSymmetry:
    """Test that the blossom is symmetric in its arguments."""

    def test_degree2_symmetry(self) -> None:
        """Blossom is symmetric for degree-2 spline."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        ctrl = [1.0, 2.0, 0.5, 3.0]

        for u1, u2 in [(0.1, 0.4), (0.2, 0.8), (0.5, 0.5), (0.0, 1.0)]:
            b12 = _eval_blossom(knots, 2, ctrl, [u1, u2])
            b21 = _eval_blossom(knots, 2, ctrl, [u2, u1])
            np.testing.assert_allclose(b12, b21, atol=1e-12)

    def test_degree3_symmetry(self) -> None:
        """Blossom is symmetric for degree-3 spline with two elements."""
        knots = [0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0, 1.0]
        ctrl = [1.0, 0.5, 2.0, 1.5, 3.0, 0.5]

        u1, u2, u3 = 0.1, 0.5, 0.8
        orderings = [
            [u1, u2, u3],
            [u1, u3, u2],
            [u2, u1, u3],
            [u2, u3, u1],
            [u3, u1, u2],
            [u3, u2, u1],
        ]
        results = [_eval_blossom(knots, 3, ctrl, perm) for perm in orderings]
        for res in results[1:]:
            np.testing.assert_allclose(res, results[0], atol=1e-12)


# ---------------------------------------------------------------------------
# Control point recovery: blossom(t_{i+1}, ..., t_{i+p}) == P_i
# ---------------------------------------------------------------------------


class TestBlossomControlPointRecovery:
    """Test that the blossom at consecutive interior knots recovers control points."""

    def test_degree2_knot_values(self) -> None:
        """blossom(t_{i+1}, t_{i+2}) == P[i] for degree-2 B-spline."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        ctrl = [1.0, 2.0, 0.5, 3.0]
        n = len(ctrl)
        p = 2

        for i in range(n):
            u_vals = [float(knots[i + r]) for r in range(1, p + 1)]
            blossom_val = _eval_blossom(knots, p, ctrl, u_vals)
            np.testing.assert_allclose(blossom_val[0], ctrl[i], atol=1e-12, err_msg=f"i={i}")

    def test_degree3_knot_values(self) -> None:
        """blossom(t_{i+1}, t_{i+2}, t_{i+3}) == P[i] for degree-3 B-spline."""
        knots = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
        ctrl = [1.0, 0.5, 2.0, 1.5, 0.5]
        n = len(ctrl)
        p = 3

        for i in range(n):
            u_vals = [float(knots[i + r]) for r in range(1, p + 1)]
            blossom_val = _eval_blossom(knots, p, ctrl, u_vals)
            np.testing.assert_allclose(blossom_val[0], ctrl[i], atol=1e-12, err_msg=f"i={i}")


# ---------------------------------------------------------------------------
# Input validation errors
# ---------------------------------------------------------------------------


class TestBlossomValidation:
    """Tests for input validation in _evaluate_blossom_1d."""

    def test_wrong_u_values_length(self) -> None:
        """Raises ValueError when len(u_values) != degree."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        ctrl = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        u = np.array([0.5], dtype=np.float64)  # degree=2 requires 2 values
        with pytest.raises(ValueError, match="length"):
            _evaluate_blossom_1d(
                np.array(knots, dtype=np.float64),
                2,
                ctrl,
                u,
                1e-12,
            )

    def test_u_values_out_of_domain(self) -> None:
        """Raises ValueError when a u value lies outside the domain."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        ctrl = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        u = np.array([0.5, 1.5], dtype=np.float64)  # 1.5 > domain [0, 1]
        with pytest.raises(ValueError, match="outside domain"):
            _evaluate_blossom_1d(
                np.array(knots, dtype=np.float64),
                2,
                ctrl,
                u,
                1e-12,
            )
