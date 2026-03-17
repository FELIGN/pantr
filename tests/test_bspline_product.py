"""Tests for B-spline pointwise multiplication."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import Bspline
from pantr.bspline_space_1D import BsplineSpace1D
from pantr.bspline_space_nd import BsplineSpace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_bspline(
    knots: list[float],
    degree: int,
    ctrl: list[float] | list[list[float]],
    is_rational: bool = False,
    dtype: type = np.float64,
) -> Bspline:
    """Create a 1D B-spline from plain Python lists."""
    space_1d = BsplineSpace1D(np.array(knots, dtype=dtype), degree)
    space = BsplineSpace([space_1d])
    cp: npt.NDArray[np.float32 | np.float64] = np.array(ctrl, dtype=dtype)
    return Bspline(space, cp, is_rational=is_rational)


def eval_pts(a: float = 0.0, b: float = 1.0, n: int = 201) -> np.ndarray:  # type: ignore[type-arg]
    """Return n evenly-spaced evaluation points in [a, b]."""
    return np.linspace(a, b, n, dtype=np.float64)


# ---------------------------------------------------------------------------
# Non-rational tests
# ---------------------------------------------------------------------------


class TestNonRationalProduct:
    """Correctness tests for non-rational B-spline multiplication."""

    def test_same_knot_vector_degree2(self) -> None:
        """Product of two quadratic B-splines on the same mesh."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        f = make_bspline(knots, 2, [1.0, 2.0, 1.5, 3.0])
        g = make_bspline(knots, 2, [0.5, 1.0, 2.0, 0.5])

        h = f.multiply(g)

        pts = eval_pts()
        f_vals = f.evaluate(pts)
        g_vals = g.evaluate(pts)
        h_vals = h.evaluate(pts)

        np.testing.assert_allclose(h_vals, f_vals * g_vals, atol=1e-11)

    def test_different_knot_vectors(self) -> None:
        """Product with non-matching interior breakpoints (union rule)."""
        knots_f = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        knots_g = [0.0, 0.0, 0.0, 0.75, 1.0, 1.0, 1.0]
        f = make_bspline(knots_f, 2, [1.0, 2.0, 0.5, 3.0])
        g = make_bspline(knots_g, 2, [2.0, 1.0, 3.0, 0.5])

        h = f.multiply(g)

        pts = eval_pts()
        np.testing.assert_allclose(h.evaluate(pts), f.evaluate(pts) * g.evaluate(pts), atol=1e-11)

    def test_different_degrees(self) -> None:
        """Product of splines with different degrees (p=2, q=3)."""
        knots_f = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        knots_g = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
        f = make_bspline(knots_f, 2, [1.0, 2.0, 0.5, 3.0])
        g = make_bspline(knots_g, 3, [1.0, 0.5, 2.0, 1.5, 0.5])

        h = f.multiply(g)

        assert h.degree == (5,)
        pts = eval_pts()
        np.testing.assert_allclose(h.evaluate(pts), f.evaluate(pts) * g.evaluate(pts), atol=1e-11)

    def test_additive_multiplicity_at_shared_knot(self) -> None:
        """Interior multiplicity in h equals sum of multiplicities in f and g."""
        # f: degree 2, interior knot 0.5 with mult 2 (C^0)
        knots_f = [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0]
        # g: degree 2, interior knot 0.5 with mult 1 (C^1)
        knots_g = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        f = make_bspline(knots_f, 2, [1.0, 2.0, 3.0, 1.5, 2.0])
        g = make_bspline(knots_g, 2, [0.5, 1.5, 2.5, 1.0])

        h = f.multiply(g)

        # Product degree = 4, interior mult = 2+1 = 3
        assert h.degree == (4,)
        pts = eval_pts()
        np.testing.assert_allclose(h.evaluate(pts), f.evaluate(pts) * g.evaluate(pts), atol=1e-11)

    def test_vector_rank(self) -> None:
        """Component-wise product for rank-2 splines."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        f = make_bspline(knots, 2, [[1.0, 2.0], [0.5, 1.0], [2.0, 3.0], [1.5, 0.5]])
        g = make_bspline(knots, 2, [[2.0, 0.5], [1.0, 3.0], [0.5, 1.5], [3.0, 2.0]])

        h = f.multiply(g)

        assert h.rank == 2  # noqa: PLR2004
        pts = eval_pts()
        np.testing.assert_allclose(h.evaluate(pts), f.evaluate(pts) * g.evaluate(pts), atol=1e-11)

    def test_three_element_mesh(self) -> None:
        """Product on a mesh with two interior breakpoints."""
        # degree=3, knots give 10-3-1=6 basis functions for each spline
        knots = [0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0, 1.0]
        f = make_bspline(knots, 3, [1.0, 0.5, 2.0, 1.5, 3.0, 0.5])
        g = make_bspline(knots, 3, [2.0, 1.0, 0.5, 3.0, 1.5, 2.0])

        h = f.multiply(g)

        assert h.degree == (6,)
        pts = eval_pts()
        np.testing.assert_allclose(h.evaluate(pts), f.evaluate(pts) * g.evaluate(pts), atol=1e-11)

    def test_mul_operator(self) -> None:
        """``f * g`` is equivalent to ``f.multiply(g)``."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        f = make_bspline(knots, 2, [1.0, 2.0, 3.0])
        g = make_bspline(knots, 2, [2.0, 1.0, 0.5])

        h_mul = f.multiply(g)
        h_op = f * g

        pts = eval_pts()
        np.testing.assert_allclose(h_mul.evaluate(pts), h_op.evaluate(pts), atol=1e-15)


# ---------------------------------------------------------------------------
# Rational tests
# ---------------------------------------------------------------------------


class TestRationalProduct:
    """Tests for products involving rational (NURBS) B-splines."""

    def test_mixed_nonrational_times_rational(self) -> None:
        """Non-rational x rational: result should be rational."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        f = make_bspline(knots, 2, [1.0, 2.0, 3.0])
        # Rational with non-unit weights: CPs store [w*P, w]
        g = make_bspline(knots, 2, [[1.0, 1.0], [4.0, 2.0], [3.0, 1.0]], is_rational=True)

        h = f.multiply(g)

        assert h.is_rational
        pts = eval_pts()
        np.testing.assert_allclose(h.evaluate(pts), f.evaluate(pts) * g.evaluate(pts), atol=1e-11)

    def test_rational_times_rational(self) -> None:
        """Rational x rational with non-unit weights."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        # f: w*P stored in col 0, w in col 1
        f = make_bspline(
            knots, 2, [[1.0, 1.0], [4.0, 2.0], [1.5, 1.0], [3.0, 1.5]], is_rational=True
        )
        g = make_bspline(
            knots, 2, [[2.0, 2.0], [1.0, 1.0], [3.0, 1.5], [0.5, 0.5]], is_rational=True
        )

        h = f.multiply(g)

        assert h.is_rational
        pts = eval_pts()
        np.testing.assert_allclose(h.evaluate(pts), f.evaluate(pts) * g.evaluate(pts), atol=1e-11)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case tests for B-spline multiplication."""

    def test_single_bezier_element(self) -> None:
        """Single-element Bezier x Bezier (no interior breakpoints)."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        f = make_bspline(knots, 2, [1.0, 2.0, 3.0])
        g = make_bspline(knots, 2, [3.0, 1.0, 2.0])

        h = f.multiply(g)

        assert h.degree == (4,)
        pts = eval_pts()
        np.testing.assert_allclose(h.evaluate(pts), f.evaluate(pts) * g.evaluate(pts), atol=1e-11)

    def test_multiply_by_constant(self) -> None:
        """Multiplying by a degree-0 constant spline scales the result."""
        knots_f = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        f = make_bspline(knots_f, 2, [1.0, 2.0, 0.5, 3.0])
        # Degree-0 constant: B-spline of value 2 everywhere
        knots_c = [0.0, 1.0]
        c = make_bspline(knots_c, 0, [2.0])

        h = f.multiply(c)

        pts = eval_pts()
        np.testing.assert_allclose(h.evaluate(pts), f.evaluate(pts) * 2.0, atol=1e-11)

    def test_multiply_by_one(self) -> None:
        """Multiplying by the constant spline 1 is the identity (up to degree)."""
        knots_f = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        f = make_bspline(knots_f, 2, [1.0, 2.0, 0.5, 3.0])
        knots_one = [0.0, 1.0]
        one = make_bspline(knots_one, 0, [1.0])

        h = f.multiply(one)

        pts = eval_pts()
        np.testing.assert_allclose(h.evaluate(pts), f.evaluate(pts), atol=1e-11)

    def test_error_dim_not_1_f(self) -> None:
        """Raises ValueError when f has dim != 1."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        space_1d = BsplineSpace1D(knots, 2)
        space_2d = BsplineSpace([space_1d, space_1d])
        f2d = Bspline(space_2d, np.ones((3, 3, 1), dtype=np.float64))

        knots_1d = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        g = make_bspline(knots_1d, 2, [1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="dim=2"):
            f2d.multiply(g)

    def test_error_dtype_mismatch(self) -> None:
        """Raises ValueError when dtypes differ."""
        knots64 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        f = make_bspline(knots64, 2, [1.0, 2.0, 3.0], dtype=np.float64)
        g = make_bspline(knots64, 2, [1.0, 1.0, 1.0], dtype=np.float32)

        with pytest.raises(ValueError, match="dtype"):
            f.multiply(g)

    def test_error_rank_mismatch(self) -> None:
        """Raises ValueError when ranks differ."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        f = make_bspline(knots, 2, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # rank 2
        g = make_bspline(knots, 2, [1.0, 1.0, 1.0])  # rank 1

        with pytest.raises(ValueError, match="rank"):
            f.multiply(g)

    def test_error_domain_mismatch(self) -> None:
        """Raises ValueError when domains differ."""
        f = make_bspline([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2, [1.0, 2.0, 3.0])
        g = make_bspline([0.0, 0.0, 0.0, 2.0, 2.0, 2.0], 2, [1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="domain"):
            f.multiply(g)

    def test_error_periodic(self) -> None:
        """Raises NotImplementedError for periodic B-splines."""
        knots = [0.0, 0.0, 0.5, 1.0, 1.0]
        f = make_bspline(knots, 1, [1.0, 2.0, 3.0])
        knots_p = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
        space_p_1d = BsplineSpace1D(knots_p, 1, periodic=True)
        space_p = BsplineSpace([space_p_1d])
        n_basis = space_p.num_total_basis
        g_p = Bspline(space_p, np.ones(n_basis, dtype=np.float64))
        with pytest.raises(NotImplementedError):
            f.multiply(g_p)
