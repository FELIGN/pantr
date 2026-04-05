"""Tests for interpolate_bezier and fit_bezier (including scattered fitting)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pytest

from pantr.bezier import fit_bezier, interpolate_bezier
from pantr.quad import PointsLattice

# ---------------------------------------------------------------------------
# 1D scalar interpolation
# ---------------------------------------------------------------------------


class TestInterpolate1DScalar:
    """Tests for 1D scalar-valued interpolation."""

    def test_linear(self) -> None:
        """Interpolating a linear function with 2 points recovers it exactly."""
        b = interpolate_bezier(lambda lat: 2.0 * lat.pts_per_dir[0] + 1.0, 2)
        assert b.degree == (1,)
        assert b.rank == 1
        pts = np.array([0.0, 0.5, 1.0])
        vals = b.evaluate(pts)
        nptest.assert_allclose(vals, [1.0, 2.0, 3.0], atol=1e-14)

    def test_quadratic(self) -> None:
        """Interpolating x^2 with 3 points recovers it exactly."""
        b = interpolate_bezier(lambda lat: lat.pts_per_dir[0] ** 2, 3)
        assert b.degree == (2,)
        pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        expected = pts**2
        nptest.assert_allclose(b.evaluate(pts), expected, atol=1e-13)

    def test_cubic(self) -> None:
        """Interpolating x^3 with 4 points recovers it exactly."""
        b = interpolate_bezier(lambda lat: lat.pts_per_dir[0] ** 3, 4)
        assert b.degree == (3,)
        pts = np.linspace(0, 1, 10)
        nptest.assert_allclose(b.evaluate(pts), pts**3, atol=1e-12)

    def test_single_point(self) -> None:
        """A single interpolation point gives a degree-0 (constant) Bezier."""
        b = interpolate_bezier(lambda lat: np.full(1, 7.0), 1)
        assert b.degree == (0,)
        nptest.assert_allclose(b.evaluate(np.array([0.5])), [7.0], atol=1e-14)


# ---------------------------------------------------------------------------
# 1D vector-valued interpolation
# ---------------------------------------------------------------------------


class TestInterpolate1DVector:
    """Tests for 1D vector-valued interpolation (parametric curves)."""

    def test_circle_arc(self) -> None:
        """Interpolate a quarter-circle parametric curve."""

        def quarter_circle(lat: PointsLattice) -> npt.NDArray[np.floating[Any]]:
            t = lat.pts_per_dir[0]
            theta = t * (np.pi / 2.0)
            return np.stack([np.cos(theta), np.sin(theta)], axis=-1)

        b = interpolate_bezier(quarter_circle, 8)
        assert b.degree == (7,)
        nptest.assert_equal(b.rank, 2)

        # Evaluate at midpoint
        pts = np.array([0.5])
        result = b.evaluate(pts)
        theta = 0.5 * np.pi / 2.0
        nptest.assert_allclose(result, [[np.cos(theta), np.sin(theta)]], atol=1e-6)

    def test_linear_curve(self) -> None:
        """A linear vector-valued function is recovered exactly."""

        def line(lat: PointsLattice) -> npt.NDArray[np.floating[Any]]:
            t = lat.pts_per_dir[0]
            return np.stack([t, 2.0 * t + 1.0, -t + 3.0], axis=-1)

        b = interpolate_bezier(line, 2)
        assert b.degree == (1,)
        nptest.assert_equal(b.rank, 3)
        pts = np.array([0.0, 0.5, 1.0])
        expected = np.array([[0.0, 1.0, 3.0], [0.5, 2.0, 2.5], [1.0, 3.0, 2.0]])
        nptest.assert_allclose(b.evaluate(pts), expected, atol=1e-14)


# ---------------------------------------------------------------------------
# 2D scalar interpolation
# ---------------------------------------------------------------------------


class TestInterpolate2DScalar:
    """Tests for 2D scalar-valued interpolation."""

    def test_bilinear(self) -> None:
        """Interpolating a bilinear function with (2,2) points recovers it."""

        def bilinear(lat: PointsLattice) -> npt.NDArray[np.floating[Any]]:
            pts = lat.get_all_points()
            return pts[:, 0] + pts[:, 1]

        b = interpolate_bezier(bilinear, [2, 2])
        assert b.degree == (1, 1)
        assert b.rank == 1
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
        expected = pts[:, 0] + pts[:, 1]
        nptest.assert_allclose(b.evaluate(pts), expected, atol=1e-13)

    def test_biquadratic(self) -> None:
        """Interpolating x^2 + y^2 with (3,3) points."""

        def biquad(lat: PointsLattice) -> npt.NDArray[np.floating[Any]]:
            pts = lat.get_all_points()
            return pts[:, 0] ** 2 + pts[:, 1] ** 2

        b = interpolate_bezier(biquad, [3, 3])
        assert b.degree == (2, 2)
        pts = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        expected = pts[:, 0] ** 2 + pts[:, 1] ** 2
        nptest.assert_allclose(b.evaluate(pts), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 2D vector-valued interpolation
# ---------------------------------------------------------------------------


class TestInterpolate2DVector:
    """Tests for 2D vector-valued interpolation (parametric surfaces)."""

    def test_planar_surface(self) -> None:
        """A linear vector-valued 2D function is recovered exactly."""

        def plane(lat: PointsLattice) -> npt.NDArray[np.floating[Any]]:
            pts = lat.get_all_points()
            x, y = pts[:, 0], pts[:, 1]
            return np.stack([x, y, x + y], axis=-1)

        b = interpolate_bezier(plane, [2, 2])
        assert b.degree == (1, 1)
        nptest.assert_equal(b.rank, 3)
        pts = np.array([[0.5, 0.5], [0.0, 1.0], [1.0, 0.0]])
        expected = np.array([[0.5, 0.5, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
        nptest.assert_allclose(b.evaluate(pts), expected, atol=1e-13)


# ---------------------------------------------------------------------------
# Node selection
# ---------------------------------------------------------------------------


class TestNodeSelection:
    """Tests for different node selection strategies."""

    def test_chebyshev_default(self) -> None:
        """Default (Chebyshev) nodes recover a quadratic exactly."""
        b = interpolate_bezier(lambda lat: lat.pts_per_dir[0] ** 2, 3, nodes=None)
        pts = np.array([0.0, 0.5, 1.0])
        nptest.assert_allclose(b.evaluate(pts), pts**2, atol=1e-13)

    def test_chebyshev_explicit(self) -> None:
        """Explicitly requesting 'chebyshev' is the same as default."""
        b = interpolate_bezier(lambda lat: lat.pts_per_dir[0] ** 2, 3, nodes="chebyshev")
        pts = np.array([0.0, 0.5, 1.0])
        nptest.assert_allclose(b.evaluate(pts), pts**2, atol=1e-13)

    def test_uniform_nodes(self) -> None:
        """Uniform nodes can also recover polynomials (less stable for high degree)."""
        b = interpolate_bezier(lambda lat: lat.pts_per_dir[0] ** 2, 3, nodes="uniform")
        pts = np.array([0.0, 0.5, 1.0])
        nptest.assert_allclose(b.evaluate(pts), pts**2, atol=1e-12)

    def test_custom_nodes_array(self) -> None:
        """User-provided custom nodes as a single array."""
        custom = np.array([0.0, 0.5, 1.0])
        b = interpolate_bezier(lambda lat: lat.pts_per_dir[0] ** 2, 3, nodes=custom)
        pts = np.array([0.0, 0.5, 1.0])
        nptest.assert_allclose(b.evaluate(pts), pts**2, atol=1e-12)

    def test_custom_nodes_sequence(self) -> None:
        """User-provided custom nodes as a sequence of arrays (2D)."""
        nodes_x = np.array([0.0, 0.5, 1.0])
        nodes_y = np.array([0.0, 1.0])

        def func(lat: PointsLattice) -> npt.NDArray[np.floating[Any]]:
            pts = lat.get_all_points()
            return pts[:, 0] + pts[:, 1]

        b = interpolate_bezier(func, [3, 2], nodes=[nodes_x, nodes_y])
        assert b.degree == (2, 1)
        pts = np.array([[0.5, 0.5]])
        nptest.assert_allclose(b.evaluate(pts), [1.0], atol=1e-12)


# ---------------------------------------------------------------------------
# Callable receives PointsLattice
# ---------------------------------------------------------------------------


class TestInterpolateCallableLattice:
    """Tests verifying that the callable receives a PointsLattice."""

    def test_1d_receives_lattice(self) -> None:
        """1D interpolation passes a PointsLattice to the callable."""

        def func(lat: PointsLattice) -> npt.NDArray[np.floating[Any]]:
            assert isinstance(lat, PointsLattice)
            assert lat.dim == 1
            return lat.pts_per_dir[0] ** 2

        b = interpolate_bezier(func, 3)
        assert b.degree == (2,)

    def test_2d_receives_lattice(self) -> None:
        """2D interpolation passes a PointsLattice to the callable."""

        def func(lat: PointsLattice) -> npt.NDArray[np.floating[Any]]:
            assert isinstance(lat, PointsLattice)
            assert lat.dim == 2  # noqa: PLR2004
            pts = lat.get_all_points()
            return pts[:, 0] + pts[:, 1]

        b = interpolate_bezier(func, [3, 3])
        assert b.degree == (2, 2)

    def test_lattice_has_correct_nodes(self) -> None:
        """The PointsLattice contains the correct node arrays."""
        custom_nodes = np.array([0.0, 0.5, 1.0])

        def func(lat: PointsLattice) -> npt.NDArray[np.floating[Any]]:
            nptest.assert_allclose(lat.pts_per_dir[0], custom_nodes)
            return lat.pts_per_dir[0]

        interpolate_bezier(func, 3, nodes=custom_nodes)

    def test_points_lattice_as_nodes(self) -> None:
        """Passing a PointsLattice as the nodes parameter works."""
        lattice = PointsLattice([np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0])])

        def func(lat: PointsLattice) -> npt.NDArray[np.floating[Any]]:
            pts = lat.get_all_points()
            return pts[:, 0] + pts[:, 1]

        b = interpolate_bezier(func, [3, 2], nodes=lattice)
        assert b.degree == (2, 1)
        pts = np.array([[0.5, 0.5]])
        nptest.assert_allclose(b.evaluate(pts), [1.0], atol=1e-12)


# ---------------------------------------------------------------------------
# dtype handling
# ---------------------------------------------------------------------------


class TestDtype:
    """Tests for dtype inference."""

    def test_float64_default(self) -> None:
        """Default dtype is float64 when func returns float64."""
        b = interpolate_bezier(lambda lat: lat.pts_per_dir[0], 2)
        assert b.dtype == np.float64

    def test_int_return_promoted_to_float64(self) -> None:
        """Integer return from func is promoted to float64."""
        b = interpolate_bezier(lambda lat: np.ones(len(lat.pts_per_dir[0]), dtype=int), 2)
        assert b.dtype == np.float64

    def test_fit_infers_from_values(self) -> None:
        """fit_bezier infers dtype from the values array."""
        nodes = np.array([0.0, 1.0], dtype=np.float32)
        vals = np.array([1.0, 2.0], dtype=np.float32)
        b = fit_bezier(vals, nodes)
        assert b.dtype == np.float32


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestInterpolateValidation:
    """Input validation tests."""

    def test_n_pts_too_small(self) -> None:
        """n_pts < 1 raises ValueError."""
        with pytest.raises(ValueError, match=r"n_pts.*>= 1"):
            interpolate_bezier(lambda lat: lat.pts_per_dir[0], 0)

    def test_mismatched_nodes_n_pts(self) -> None:
        """Custom nodes with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="does not match"):
            interpolate_bezier(lambda lat: lat.pts_per_dir[0], 3, nodes=np.array([0.0, 1.0]))

    def test_wrong_number_of_node_arrays(self) -> None:
        """Wrong number of node arrays for 2D raises ValueError."""
        with pytest.raises(ValueError, match="Expected 2"):
            interpolate_bezier(
                lambda lat: lat.get_all_points()[:, 0],
                [3, 3],
                nodes=[np.array([0.0, 0.5, 1.0])],
            )

    def test_bad_function_output_shape(self) -> None:
        """Function returning wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="Function returned shape"):
            interpolate_bezier(lambda lat: np.ones((2, 3)), 3)


# ---------------------------------------------------------------------------
# Degree parameter (least-squares approximation)
# ---------------------------------------------------------------------------


class TestInterpolateDegree:
    """Tests for the optional degree parameter on interpolate_bezier."""

    def test_exact_when_degree_equals_n_pts_minus_1(self) -> None:
        """Explicit degree = n_pts - 1 gives the same result as default."""
        b = interpolate_bezier(lambda lat: lat.pts_per_dir[0] ** 2, 5, degree=4)
        assert b.degree == (4,)
        pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        nptest.assert_allclose(b.evaluate(pts), pts**2, atol=1e-13)

    def test_least_squares_quadratic(self) -> None:
        """Fitting x^2 with degree=2 from 5 sample points recovers it."""
        b = interpolate_bezier(lambda lat: lat.pts_per_dir[0] ** 2, 5, degree=2)
        assert b.degree == (2,)
        pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        nptest.assert_allclose(b.evaluate(pts), pts**2, atol=1e-12)

    def test_least_squares_linear(self) -> None:
        """Fitting a linear function with degree=1 from 10 points."""
        b = interpolate_bezier(lambda lat: 3.0 * lat.pts_per_dir[0] + 1.0, 10, degree=1)
        assert b.degree == (1,)
        pts = np.array([0.0, 0.5, 1.0])
        nptest.assert_allclose(b.evaluate(pts), [1.0, 2.5, 4.0], atol=1e-12)

    def test_2d_least_squares(self) -> None:
        """Fitting x+y with degree=(1,1) from (3,3) samples."""

        def func(lat: PointsLattice) -> npt.NDArray[np.floating[Any]]:
            pts = lat.get_all_points()
            return pts[:, 0] + pts[:, 1]

        b = interpolate_bezier(func, [3, 3], degree=[1, 1])
        assert b.degree == (1, 1)
        pts = np.array([[0.5, 0.5], [0.0, 1.0], [1.0, 0.0]])
        expected = pts[:, 0] + pts[:, 1]
        nptest.assert_allclose(b.evaluate(pts), expected, atol=1e-12)

    def test_degree_too_large_raises(self) -> None:
        """Degree >= n_pts raises ValueError."""
        with pytest.raises(ValueError, match="must be < n_pts"):
            interpolate_bezier(lambda lat: lat.pts_per_dir[0], 3, degree=3)

    def test_degree_negative_raises(self) -> None:
        """Negative degree raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            interpolate_bezier(lambda lat: lat.pts_per_dir[0], 3, degree=-1)

    def test_degree_length_mismatch_raises(self) -> None:
        """Degree sequence length != n_pts sequence length raises ValueError."""
        with pytest.raises(ValueError, match="entries"):
            interpolate_bezier(lambda lat: lat.get_all_points()[:, 0], [3, 3], degree=[1])


# ---------------------------------------------------------------------------
# fit_bezier — 1D scalar
# ---------------------------------------------------------------------------


class TestFit1DScalar:
    """Tests for 1D scalar-valued fitting from pre-evaluated values."""

    def test_exact_linear(self) -> None:
        """Fitting 2 values at nodes recovers a linear Bezier."""
        nodes = np.array([0.0, 1.0])
        vals = 2.0 * nodes + 1.0
        b = fit_bezier(vals, nodes)
        assert b.degree == (1,)
        pts = np.array([0.0, 0.5, 1.0])
        nptest.assert_allclose(b.evaluate(pts), [1.0, 2.0, 3.0], atol=1e-13)

    def test_exact_quadratic(self) -> None:
        """Fitting 3 values recovers a quadratic."""
        nodes = np.array([0.0, 0.5, 1.0])
        vals = nodes**2
        b = fit_bezier(vals, nodes)
        assert b.degree == (2,)
        pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        nptest.assert_allclose(b.evaluate(pts), pts**2, atol=1e-12)

    def test_single_value(self) -> None:
        """Fitting a single value gives a degree-0 Bezier."""
        b = fit_bezier(np.array([7.0]), np.array([0.5]))
        assert b.degree == (0,)
        nptest.assert_allclose(b.evaluate(np.array([0.5])), [7.0], atol=1e-14)


# ---------------------------------------------------------------------------
# fit_bezier — 1D vector
# ---------------------------------------------------------------------------


class TestFit1DVector:
    """Tests for 1D vector-valued fitting."""

    def test_linear_vector(self) -> None:
        """Fitting a linear vector-valued function."""
        nodes = np.array([0.0, 1.0])
        vals = np.stack([nodes, 2.0 * nodes + 1.0], axis=-1)
        b = fit_bezier(vals, nodes)
        assert b.degree == (1,)
        nptest.assert_equal(b.rank, 2)
        pts = np.array([0.0, 0.5, 1.0])
        expected = np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]])
        nptest.assert_allclose(b.evaluate(pts), expected, atol=1e-13)


# ---------------------------------------------------------------------------
# fit_bezier — 2D tensor-product
# ---------------------------------------------------------------------------


class TestFit2D:
    """Tests for 2D fitting from pre-evaluated values."""

    def test_bilinear(self) -> None:
        """Fitting bilinear values on a 2x2 grid."""
        nodes_x = np.array([0.0, 1.0])
        nodes_y = np.array([0.0, 1.0])
        xx, yy = np.meshgrid(nodes_x, nodes_y, indexing="ij")
        vals = xx + yy
        b = fit_bezier(vals, [nodes_x, nodes_y])
        assert b.degree == (1, 1)
        pts = np.array([[0.5, 0.5], [0.0, 1.0]])
        expected = pts[:, 0] + pts[:, 1]
        nptest.assert_allclose(b.evaluate(pts), expected, atol=1e-12)

    def test_with_points_lattice(self) -> None:
        """Fitting bilinear values using a PointsLattice for nodes."""
        nodes_x = np.array([0.0, 1.0])
        nodes_y = np.array([0.0, 1.0])
        lattice = PointsLattice([nodes_x, nodes_y])
        xx, yy = np.meshgrid(nodes_x, nodes_y, indexing="ij")
        vals = xx + yy
        b = fit_bezier(vals, lattice)
        assert b.degree == (1, 1)
        pts = np.array([[0.5, 0.5]])
        nptest.assert_allclose(b.evaluate(pts), [1.0], atol=1e-12)


# ---------------------------------------------------------------------------
# fit_bezier — degree (least-squares)
# ---------------------------------------------------------------------------


class TestFitDegree:
    """Tests for the degree parameter on fit_bezier."""

    def test_least_squares_quadratic(self) -> None:
        """Fitting x^2 from 5 points with degree=2."""
        nodes = np.linspace(0.0, 1.0, 5)
        vals = nodes**2
        b = fit_bezier(vals, nodes, degree=2)
        assert b.degree == (2,)
        pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        nptest.assert_allclose(b.evaluate(pts), pts**2, atol=1e-12)

    def test_least_squares_constant(self) -> None:
        """Fitting a constant from 5 points with degree=0."""
        nodes = np.linspace(0.0, 1.0, 5)
        vals = np.full(5, 3.14)
        b = fit_bezier(vals, nodes, degree=0)
        assert b.degree == (0,)
        nptest.assert_allclose(b.evaluate(np.array([0.5])), [3.14], atol=1e-12)


# ---------------------------------------------------------------------------
# fit_bezier — scattered (non-tensor-product) points
# ---------------------------------------------------------------------------


class TestFitScattered1D:
    """Tests for 1D scattered fitting."""

    def test_linear_scattered(self) -> None:
        """Fit a linear from scattered 1D points (presented as 2D array)."""
        pts = np.array([[0.0], [0.3], [0.7], [1.0]])
        vals = 2.0 * pts[:, 0] + 1.0
        b = fit_bezier(vals, pts, degree=1)
        assert b.degree == (1,)
        eval_pts = np.array([0.0, 0.5, 1.0])
        nptest.assert_allclose(b.evaluate(eval_pts), [1.0, 2.0, 3.0], atol=1e-12)

    def test_quadratic_scattered(self) -> None:
        """Fit x^2 from scattered 1D points."""
        pts = np.array([[0.0], [0.2], [0.5], [0.8], [1.0]])
        vals = pts[:, 0] ** 2
        b = fit_bezier(vals, pts, degree=2)
        assert b.degree == (2,)
        eval_pts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        nptest.assert_allclose(b.evaluate(eval_pts), eval_pts**2, atol=1e-12)


class TestFitScattered2D:
    """Tests for 2D scattered fitting."""

    def test_bilinear_scattered(self) -> None:
        """Fit x + y from scattered 2D points."""
        rng = np.random.default_rng(42)
        pts = rng.random((20, 2))
        vals = pts[:, 0] + pts[:, 1]
        b = fit_bezier(vals, pts, degree=[1, 1])
        assert b.degree == (1, 1)
        eval_pts = np.array([[0.5, 0.5], [0.0, 1.0], [1.0, 0.0]])
        expected = eval_pts[:, 0] + eval_pts[:, 1]
        nptest.assert_allclose(b.evaluate(eval_pts), expected, atol=1e-10)

    def test_biquadratic_scattered(self) -> None:
        """Fit x^2 + y^2 from scattered 2D points."""
        rng = np.random.default_rng(123)
        pts = rng.random((30, 2))
        vals = pts[:, 0] ** 2 + pts[:, 1] ** 2
        b = fit_bezier(vals, pts, degree=[2, 2])
        assert b.degree == (2, 2)
        eval_pts = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        expected = eval_pts[:, 0] ** 2 + eval_pts[:, 1] ** 2
        nptest.assert_allclose(b.evaluate(eval_pts), expected, atol=1e-10)

    def test_vector_scattered(self) -> None:
        """Fit a vector-valued function from scattered 2D points."""
        rng = np.random.default_rng(7)
        pts = rng.random((20, 2))
        vals = np.stack([pts[:, 0], pts[:, 1], pts[:, 0] + pts[:, 1]], axis=-1)
        b = fit_bezier(vals, pts, degree=[1, 1])
        assert b.degree == (1, 1)
        nptest.assert_equal(b.rank, 3)
        eval_pts = np.array([[0.5, 0.5]])
        expected = np.array([[0.5, 0.5, 1.0]])
        nptest.assert_allclose(b.evaluate(eval_pts), expected, atol=1e-10)


class TestFitScatteredValidation:
    """Validation tests for scattered fitting."""

    def test_degree_required(self) -> None:
        """Scattered fitting without degree raises ValueError."""
        pts = np.array([[0.0], [0.5], [1.0]])
        with pytest.raises(ValueError, match="degree is required"):
            fit_bezier(np.array([1.0, 2.0, 3.0]), pts)

    def test_underdetermined(self) -> None:
        """Too few points for the requested degree raises ValueError."""
        pts = np.array([[0.0, 0.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="Underdetermined"):
            fit_bezier(np.array([1.0, 2.0]), pts, degree=[2, 2])

    def test_values_pts_mismatch(self) -> None:
        """Mismatched number of values and points raises ValueError."""
        pts = np.array([[0.0], [0.5], [1.0]])
        with pytest.raises(ValueError, match="entries"):
            fit_bezier(np.array([1.0, 2.0]), pts, degree=1)

    def test_degree_dim_mismatch(self) -> None:
        """Degree length != number of point columns raises ValueError."""
        pts = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        with pytest.raises(ValueError, match="columns"):
            fit_bezier(np.array([1.0, 2.0, 3.0]), pts, degree=[1])


# ---------------------------------------------------------------------------
# fit_bezier — validation (tensor-product)
# ---------------------------------------------------------------------------


class TestFitValidation:
    """Input validation tests for fit_bezier."""

    def test_mismatched_nodes_values(self) -> None:
        """Node array length != values length raises ValueError."""
        with pytest.raises(ValueError, match="does not match"):
            fit_bezier(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0]))

    def test_degree_too_large(self) -> None:
        """Degree >= n_pts raises ValueError."""
        with pytest.raises(ValueError, match="must be < n_pts"):
            fit_bezier(np.array([1.0, 2.0]), np.array([0.0, 1.0]), degree=2)

    def test_node_arrays_ndim_mismatch(self) -> None:
        """Node arrays implying different ndim than values raises ValueError."""
        vals = np.ones((3, 3))
        nodes_1d = np.array([0.0, 0.5, 1.0])
        # 3 node arrays implies 3D, but values is 2D
        with pytest.raises(ValueError, match="dimensions"):
            fit_bezier(vals, [nodes_1d, nodes_1d, nodes_1d])

    def test_values_wrong_ndim(self) -> None:
        """Values with wrong number of dims raises ValueError."""
        with pytest.raises(ValueError, match="dimensions"):
            fit_bezier(np.ones((2, 3, 4)), np.array([0.0, 1.0]))
