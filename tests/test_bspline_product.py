"""Tests for B-spline pointwise multiplication."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr._bspline_basis_core import _compute_basis_nurbs_book_impl
from pantr._bspline_knots import _get_unique_knots_and_multiplicity_impl
from pantr._bspline_space_factory import create_uniform_periodic
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


def eval_pts(a: float = 0.0, b: float = 1.0, n: int = 201) -> npt.NDArray[np.float64]:
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

        # Product degree = 4, interior mult = max(2+2, 1+2) = 4
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

    def test_error_periodic_was_removed(self) -> None:
        """Periodic B-splines no longer raise NotImplementedError; they are converted."""
        # Periodic degree-1 spline over domain [0, 1]: knots [-0.5, 0, 0.5, 1, 1.5].
        knots_p = create_uniform_periodic(2, 1, domain=(0.0, 1.0))
        space_p_1d = BsplineSpace1D(knots_p, 1, periodic=True)
        space_p = BsplineSpace([space_p_1d])
        n_basis = space_p.num_total_basis
        g_p = Bspline(space_p, np.ones(n_basis, dtype=np.float64))
        f = make_bspline([0.0, 0.0, 0.5, 1.0, 1.0], 1, [1.0, 2.0, 3.0])
        # Should not raise — periodic operand is converted to open form.
        h = f.multiply(g_p)
        assert not h.space.spaces[0].periodic


# ---------------------------------------------------------------------------
# Optimal continuity tests
# ---------------------------------------------------------------------------


class TestOptimalContinuity:
    """Tests verifying that the product has optimal continuity in the knot vector."""

    def test_optimal_multiplicity_at_shared_knot(self) -> None:
        """Interior multiplicity equals max(m_f+q, m_g+p).

        f: degree 2, interior knot 0.5 with mult 2 → C^0
        g: degree 2, interior knot 0.5 with mult 1 → C^1
        Product continuity = C^{min(p-m_f, q-m_g)} = C^{min(0,1)} = C^0.
        Product degree 4, interior mult = max(2+2, 1+2) = 4 = full-Bezier.
        """
        knots_f = [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0]
        knots_g = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        f = make_bspline(knots_f, 2, [1.0, 2.0, 3.0, 1.5, 2.0])
        g = make_bspline(knots_g, 2, [0.5, 1.5, 2.5, 1.0])

        h = f.multiply(g)

        # Product degree must be 4.
        assert h.degree == (4,)

        # Interior multiplicity of 0.5 in the product: max(2+2, 1+2) = 4.
        h_space = h.space.spaces[0]
        unique, mults = _get_unique_knots_and_multiplicity_impl(
            h_space.knots, 4, float(h_space.tolerance), in_domain=True
        )
        # unique[0]=0.0, unique[1]=0.5, unique[2]=1.0
        assert unique.shape[0] == 3  # noqa: PLR2004
        np.testing.assert_allclose(unique[1], 0.5, atol=1e-12)
        assert int(mults[1]) == 4  # noqa: PLR2004

        # Correctness check.
        pts = eval_pts()
        np.testing.assert_allclose(h.evaluate(pts), f.evaluate(pts) * g.evaluate(pts), atol=1e-11)

    def test_fewer_basis_than_full_bezier(self) -> None:
        """Product has fewer basis functions than full-Bezier when interior mults < p+q.

        Both f and g have degree 2 with interior knot 0.5 at mult 1 (C^1).
        Full-Bezier would give interior mult 4 → 9 basis functions.
        Optimal: interior mult = max(1+2, 1+2) = 3 → 5+3+5 = 13 knots → 8 basis.
        """
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        f = make_bspline(knots, 2, [1.0, 2.0, 0.5, 3.0])
        g = make_bspline(knots, 2, [0.5, 1.5, 2.5, 1.0])

        h = f.multiply(g)

        # Full-Bezier product: degree 4, 2 elements → 4*2+1 = 9 basis functions.
        # Optimal: interior mult = max(1+2, 1+2) = 3 → [0]*5+[0.5]*3+[1]*5 = 13 knots → 8 basis.
        n_h = h.space.num_total_basis
        assert n_h == 8  # noqa: PLR2004

        pts = eval_pts()
        np.testing.assert_allclose(h.evaluate(pts), f.evaluate(pts) * g.evaluate(pts), atol=1e-11)

    def test_single_element_no_interior_knots(self) -> None:
        """Single Bezier element: optimal and full-Bezier coincide (no interior knots)."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        f = make_bspline(knots, 2, [1.0, 2.0, 3.0])
        g = make_bspline(knots, 2, [3.0, 1.0, 2.0])

        h = f.multiply(g)

        # Single element, degree 4: 5 basis functions.
        assert h.space.num_total_basis == 5  # noqa: PLR2004
        pts = eval_pts()
        np.testing.assert_allclose(h.evaluate(pts), f.evaluate(pts) * g.evaluate(pts), atol=1e-11)


# ---------------------------------------------------------------------------
# Periodic product tests
# ---------------------------------------------------------------------------


def _make_periodic(
    num_intervals: int, degree: int, domain: tuple[float, float] = (0.0, 1.0)
) -> Bspline:
    """Create a periodic B-spline with simple linear control points."""
    knots = create_uniform_periodic(num_intervals, degree, domain=domain)
    space_1d = BsplineSpace1D(knots, degree, periodic=True)
    space = BsplineSpace([space_1d])
    n = space.num_total_basis
    ctrl = np.linspace(1.0, 2.0, n, dtype=np.float64)
    return Bspline(space, ctrl)


def _eval_periodic_correct(f: Bspline, pts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Evaluate a periodic B-spline using the mathematically correct algorithm.

    Uses the unclamped ``first_basis = knot_id - degree`` index with modulo-wrapped
    control point lookup, which is the standard mathematical definition of a periodic
    B-spline. This differs from ``f.evaluate()`` which uses a clamped first_basis.

    Args:
        f (Bspline): A 1D periodic B-spline.
        pts (npt.NDArray[np.float64]): Interior evaluation points (must lie strictly inside
            domain).

    Returns:
        npt.NDArray[np.float64]: Evaluated values at the given points.
    """
    space_1d = f.space.spaces[0]
    knots = space_1d.knots
    p = space_1d.degree
    tol = float(space_1d.tolerance)
    n_stored = f.space.num_total_basis
    ctrl = f._control_points  # shape (n_stored, rank)

    basis_out = np.zeros((len(pts), p + 1), dtype=np.float64)
    first_basis_arr = np.zeros(len(pts), dtype=np.int64)
    _compute_basis_nurbs_book_impl(knots, p, False, tol, pts, basis_out, first_basis_arr)

    rank = ctrl.shape[1]
    result = np.zeros((len(pts), rank), dtype=np.float64)
    for i in range(len(pts)):
        s = int(first_basis_arr[i])
        for j in range(p + 1):
            idx = (s + j) % n_stored
            result[i] += basis_out[i, j] * ctrl[idx]

    return result.squeeze()


class TestPeriodicProduct:
    """Correctness tests for multiplication involving periodic B-splines."""

    def test_periodic_times_open_correctness(self) -> None:
        """Product of periodic and open B-splines equals pointwise product at interior pts."""
        f_per = _make_periodic(4, 2)
        g_open = make_bspline(
            [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0],
            2,
            [1.0, 1.5, 2.0, 1.5, 1.0, 1.5],
        )
        h = f_per.multiply(g_open)

        pts = eval_pts()[1:-1]  # interior only
        expected = _eval_periodic_correct(f_per, pts) * g_open.evaluate(pts)
        np.testing.assert_allclose(h.evaluate(pts), expected, atol=1e-11)

    def test_open_times_periodic_correctness(self) -> None:
        """Commutativity: open * periodic gives same values as periodic * open."""
        f_per = _make_periodic(4, 2)
        g_open = make_bspline(
            [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0],
            2,
            [1.0, 1.5, 2.0, 1.5, 1.0, 1.5],
        )
        h1 = f_per.multiply(g_open)
        h2 = g_open.multiply(f_per)

        pts = eval_pts()[1:-1]
        np.testing.assert_allclose(h1.evaluate(pts), h2.evaluate(pts), atol=1e-11)

    def test_periodic_times_periodic_correctness(self) -> None:
        """Product of two periodic B-splines equals pointwise product at interior pts."""
        f_per = _make_periodic(3, 2)
        g_per = _make_periodic(4, 2)
        h = f_per.multiply(g_per)

        pts = eval_pts()[1:-1]
        expected = _eval_periodic_correct(f_per, pts) * _eval_periodic_correct(g_per, pts)
        np.testing.assert_allclose(_eval_periodic_correct(h, pts), expected, atol=1e-11)

    def test_periodic_times_periodic_is_periodic(self) -> None:
        """Product of two periodic operands is periodic."""
        f_per = _make_periodic(3, 2)
        g_per = _make_periodic(3, 1)
        h = f_per.multiply(g_per)
        assert h.space.spaces[0].periodic
        assert not h.space.spaces[0].has_open_knots()

    def test_periodic_degree3_correctness(self) -> None:
        """Works for degree-3 periodic splines."""
        f_per = _make_periodic(4, 3)
        g_open = make_bspline(
            [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
            3,
            [1.0, 1.2, 0.8, 1.5, 1.0],
        )
        h = f_per.multiply(g_open)

        pts = eval_pts()[1:-1]
        expected = _eval_periodic_correct(f_per, pts) * g_open.evaluate(pts)
        np.testing.assert_allclose(h.evaluate(pts), expected, atol=1e-11)

    def test_periodic_product_different_degrees(self) -> None:
        """Product of periodic splines with different degrees is periodic and correct."""
        f_per = _make_periodic(3, 2)
        g_per = _make_periodic(4, 3)
        h = f_per.multiply(g_per)

        assert h.space.spaces[0].periodic
        assert h.degree == (5,)
        pts = eval_pts()[1:-1]
        expected = _eval_periodic_correct(f_per, pts) * _eval_periodic_correct(g_per, pts)
        np.testing.assert_allclose(_eval_periodic_correct(h, pts), expected, atol=1e-11)

    def test_periodic_product_few_elements(self) -> None:
        """Periodic product with minimal number of elements."""
        f_per = _make_periodic(3, 2)
        g_per = _make_periodic(3, 2)
        h = f_per.multiply(g_per)

        assert h.space.spaces[0].periodic
        pts = eval_pts()[1:-1]
        expected = _eval_periodic_correct(f_per, pts) * _eval_periodic_correct(g_per, pts)
        np.testing.assert_allclose(_eval_periodic_correct(h, pts), expected, atol=1e-11)

    def test_periodic_product_different_meshes(self) -> None:
        """Product of periodic splines on different meshes is periodic and correct."""
        f_per = _make_periodic(3, 2)
        g_per = _make_periodic(5, 2)
        h = f_per.multiply(g_per)

        assert h.space.spaces[0].periodic
        pts = eval_pts()[1:-1]
        expected = _eval_periodic_correct(f_per, pts) * _eval_periodic_correct(g_per, pts)
        np.testing.assert_allclose(_eval_periodic_correct(h, pts), expected, atol=1e-11)

    def test_periodic_times_open_stays_open(self) -> None:
        """When only one operand is periodic, result is open."""
        f_per = _make_periodic(3, 2)
        g_open = make_bspline(
            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            2,
            [1.0, 2.0, 1.5, 3.0],
        )
        h = f_per.multiply(g_open)
        assert not h.space.spaces[0].periodic
        assert h.space.spaces[0].has_open_knots()


# ---------------------------------------------------------------------------
# Non-open product tests
# ---------------------------------------------------------------------------


def _make_nonopen(
    num_intervals: int, degree: int, domain: tuple[float, float] = (0.0, 1.0)
) -> Bspline:
    """Create a non-open, non-periodic B-spline (unclamped boundary knots).

    Uses a periodic knot vector but with ``periodic=False``, giving an unclamped
    spline with boundary multiplicity < degree + 1.
    """
    knots = create_uniform_periodic(num_intervals, degree, domain=domain)
    space_1d = BsplineSpace1D(knots, degree, periodic=False)
    space = BsplineSpace([space_1d])
    n = space.num_total_basis
    ctrl = np.linspace(1.0, 3.0, n, dtype=np.float64)
    return Bspline(space, ctrl)


class TestNonOpenProduct:
    """Correctness tests for multiplication involving non-open B-splines."""

    def test_nonopen_times_nonopen_correctness(self) -> None:
        """Product of two non-open splines is non-open and correct."""
        f = _make_nonopen(3, 2)
        g = _make_nonopen(4, 2)
        h = f.multiply(g)

        assert not h.space.spaces[0].has_open_knots()
        assert not h.space.spaces[0].periodic

        pts = eval_pts()[1:-1]
        expected = f.evaluate(pts) * g.evaluate(pts)
        np.testing.assert_allclose(h.evaluate(pts), expected, atol=1e-11)

    def test_nonopen_times_periodic_is_nonopen(self) -> None:
        """Product of non-open and periodic is non-open (not periodic)."""
        f_no = _make_nonopen(3, 2)
        g_per = _make_periodic(3, 2)
        h = f_no.multiply(g_per)

        assert not h.space.spaces[0].periodic
        assert not h.space.spaces[0].has_open_knots()

        pts = eval_pts()[1:-1]
        expected = f_no.evaluate(pts) * _eval_periodic_correct(g_per, pts)
        np.testing.assert_allclose(h.evaluate(pts), expected, atol=1e-11)

    def test_nonopen_times_open_stays_open(self) -> None:
        """When one operand is open and the other non-open, result is open."""
        f_no = _make_nonopen(3, 2)
        g_open = make_bspline(
            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            2,
            [1.0, 2.0, 1.5, 3.0],
        )
        h = f_no.multiply(g_open)
        assert h.space.spaces[0].has_open_knots()
        assert not h.space.spaces[0].periodic

    def test_nonopen_different_degrees(self) -> None:
        """Product of non-open splines with different degrees."""
        f = _make_nonopen(3, 2)
        g = _make_nonopen(3, 3)
        h = f.multiply(g)

        assert not h.space.spaces[0].has_open_knots()
        assert h.degree == (5,)

        pts = eval_pts()[1:-1]
        expected = f.evaluate(pts) * g.evaluate(pts)
        np.testing.assert_allclose(h.evaluate(pts), expected, atol=1e-11)
