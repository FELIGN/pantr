"""Tests for Bézier composition."""

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose

from pantr.bezier import Bezier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _random_bezier(
    degree: tuple[int, ...],
    rank: int,
    *,
    rng: np.random.Generator = RNG,
) -> Bezier:
    """Create a Bezier with random control points in [0, 1]."""
    shape = (*tuple(d + 1 for d in degree), rank)
    ctrl: npt.NDArray[np.float64] = rng.random(shape)
    return Bezier(ctrl)


def _identity_bezier(dim: int) -> Bezier:
    """Create an identity Bezier mapping (degree 1 in each direction).

    The identity maps [0,1]^dim -> [0,1]^dim such that f(x) = x.
    """
    if dim == 1:
        return Bezier(np.array([[0.0], [1.0]]))
    if dim == 2:  # noqa: PLR2004
        ctrl: npt.NDArray[np.float64] = np.zeros((2, 2, 2))
        for i in range(2):
            for j in range(2):
                ctrl[i, j, 0] = float(i)
                ctrl[i, j, 1] = float(j)
        return Bezier(ctrl)
    if dim == 3:  # noqa: PLR2004
        ctrl = np.zeros((2, 2, 2, 3))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    ctrl[i, j, k, 0] = float(i)
                    ctrl[i, j, k, 1] = float(j)
                    ctrl[i, j, k, 2] = float(k)
        return Bezier(ctrl)
    raise NotImplementedError


def _verify_composition(
    outer: Bezier,
    inner: Bezier,
    composed: Bezier,
    n_pts: int = 50,
    *,
    atol: float = 1e-12,
) -> None:
    """Verify that composed(t) == outer(inner(t)) at random points."""
    rng = np.random.default_rng(123)
    dtype = inner.dtype
    if inner.dim == 1:
        pts = rng.random(n_pts).astype(dtype)
    else:
        pts = rng.random((n_pts, inner.dim)).astype(dtype)

    inner_vals = inner.evaluate(pts)
    expected = outer.evaluate(inner_vals)
    actual = composed.evaluate(pts)
    assert_allclose(actual, expected, atol=atol)


# ---------------------------------------------------------------------------
# 1D -> 1D
# ---------------------------------------------------------------------------


class TestCompose1Dto1D:
    """Test composition of 1D Beziers with 1D inner."""

    def test_linear_reparametrization(self) -> None:
        """Test composing a quadratic curve with a linear reparametrization."""
        f = Bezier(np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]]))
        g = Bezier(np.array([[0.2], [0.8]]))
        h = f.compose(g)

        assert h.dim == 1
        assert h.degree == (2,)
        assert h.rank == 2  # noqa: PLR2004
        _verify_composition(f, g, h)

    def test_quadratic_with_quadratic(self) -> None:
        """Test composing two quadratic curves."""
        f = _random_bezier((3,), 2, rng=np.random.default_rng(1))
        g = _random_bezier((2,), 1, rng=np.random.default_rng(2))
        h = f.compose(g)

        assert h.degree == (6,)
        _verify_composition(f, g, h)

    def test_high_degree(self) -> None:
        """Test composition with higher-degree curves."""
        f = _random_bezier((5,), 3, rng=np.random.default_rng(3))
        g = _random_bezier((4,), 1, rng=np.random.default_rng(4))
        h = f.compose(g)

        assert h.degree == (20,)
        _verify_composition(f, g, h)

    def test_scalar_rank(self) -> None:
        """Test composition with rank-1 (scalar) outer."""
        f = _random_bezier((3,), 1, rng=np.random.default_rng(5))
        g = _random_bezier((2,), 1, rng=np.random.default_rng(6))
        h = f.compose(g)

        assert h.rank == 1
        _verify_composition(f, g, h)


# ---------------------------------------------------------------------------
# nD -> 1D
# ---------------------------------------------------------------------------


class TestComposeNDto1D:
    """Test composition of nD Beziers with 1D inner (curve on surface/volume)."""

    def test_surface_with_curve(self) -> None:
        """Test composing a 2D surface with a 1D curve."""
        surface = _random_bezier((2, 3), 2, rng=np.random.default_rng(10))
        curve = _random_bezier((2,), 2, rng=np.random.default_rng(11))
        h = surface.compose(curve)

        assert h.dim == 1
        assert h.degree == (10,)  # (2+3) * 2
        assert h.rank == 2  # noqa: PLR2004
        _verify_composition(surface, curve, h)

    def test_volume_with_curve(self) -> None:
        """Test composing a 3D volume with a 1D curve."""
        volume = _random_bezier((2, 2, 2), 2, rng=np.random.default_rng(20))
        curve = _random_bezier((2,), 3, rng=np.random.default_rng(21))
        h = volume.compose(curve)

        assert h.dim == 1
        assert h.degree == (12,)  # (2+2+2) * 2
        assert h.rank == 2  # noqa: PLR2004
        _verify_composition(volume, curve, h)

    def test_surface_with_linear_curve(self) -> None:
        """Test composing a surface with a linear curve (degree 1)."""
        surface = _random_bezier((3, 2), 3, rng=np.random.default_rng(30))
        curve = _random_bezier((1,), 2, rng=np.random.default_rng(31))
        h = surface.compose(curve)

        assert h.degree == (5,)  # (3+2) * 1
        _verify_composition(surface, curve, h)


# ---------------------------------------------------------------------------
# nD -> 2D
# ---------------------------------------------------------------------------


class TestComposeNDto2D:
    """Test composition of nD Beziers with 2D inner (surface reparametrization)."""

    def test_surface_with_surface(self) -> None:
        """Test composing a 2D surface with a 2D surface reparametrization."""
        surface = _random_bezier((2, 3), 2, rng=np.random.default_rng(40))
        inner = _random_bezier((2, 2), 2, rng=np.random.default_rng(41))
        h = surface.compose(inner)

        assert h.dim == 2  # noqa: PLR2004
        assert h.degree == (10, 10)  # (2+3)*2, (2+3)*2
        assert h.rank == 2  # noqa: PLR2004
        _verify_composition(surface, inner, h)

    def test_volume_with_surface(self) -> None:
        """Test composing a 3D volume with a 2D surface."""
        volume = _random_bezier((2, 2, 2), 2, rng=np.random.default_rng(50))
        inner = _random_bezier((2, 2), 3, rng=np.random.default_rng(51))
        h = volume.compose(inner)

        assert h.dim == 2  # noqa: PLR2004
        assert h.degree == (12, 12)  # (2+2+2)*2, (2+2+2)*2
        assert h.rank == 2  # noqa: PLR2004
        _verify_composition(volume, inner, h)


# ---------------------------------------------------------------------------
# nD -> 3D (including 3D -> 3D)
# ---------------------------------------------------------------------------


class TestComposeNDto3D:
    """Test composition with 3D inner Beziers."""

    def test_volume_with_volume(self) -> None:
        """Test composing a 3D volume with a 3D volume reparametrization."""
        volume = _random_bezier((2, 2, 2), 2, rng=np.random.default_rng(60))
        inner = _random_bezier((1, 1, 1), 3, rng=np.random.default_rng(61))
        h = volume.compose(inner)

        assert h.dim == 3  # noqa: PLR2004
        assert h.degree == (6, 6, 6)  # (2+2+2)*1
        assert h.rank == 2  # noqa: PLR2004
        _verify_composition(volume, inner, h)


# ---------------------------------------------------------------------------
# Identity composition
# ---------------------------------------------------------------------------


class TestComposeIdentity:
    """Test that composing with the identity Bezier returns an equivalent result."""

    def test_identity_1d(self) -> None:
        """Test composing a 1D curve with the 1D identity."""
        f = _random_bezier((3,), 2, rng=np.random.default_rng(70))
        identity = _identity_bezier(1)
        h = f.compose(identity)

        assert h.degree == (3,)
        _verify_composition(f, identity, h)

    def test_identity_2d(self) -> None:
        """Test composing a 2D surface with the 2D identity."""
        f = _random_bezier((2, 3), 2, rng=np.random.default_rng(71))
        identity = _identity_bezier(2)
        h = f.compose(identity)

        assert h.degree == (5, 5)  # (2+3)*1
        _verify_composition(f, identity, h)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestComposeEdgeCases:
    """Test edge cases for Bezier composition."""

    def test_degree_zero_outer(self) -> None:
        """Test composing a degree-0 (constant) outer with a curve."""
        f = Bezier(np.array([[3.0, 7.0]]))  # constant, degree 0
        g = _random_bezier((3,), 1, rng=np.random.default_rng(80))
        h = f.compose(g)

        assert h.degree == (0,)
        assert_allclose(h.control_points, f.control_points)

    def test_degree_zero_inner(self) -> None:
        """Test composing with a degree-0 (constant) inner."""
        f = _random_bezier((3,), 2, rng=np.random.default_rng(81))
        g = Bezier(np.array([[0.5]]))  # constant, degree 0
        h = f.compose(g)

        assert h.degree == (0,)
        # h should be f(0.5) as a constant Bezier
        expected = f.evaluate(np.array([0.5]))
        assert_allclose(h.control_points, expected, atol=1e-14)

    def test_degree_zero_outer_2d(self) -> None:
        """Test composing a degree-(0,0) constant surface with an inner."""
        f = Bezier(np.array([[[1.0, 2.0]]]))  # degree (0,0), rank 2
        g = _random_bezier((2,), 2, rng=np.random.default_rng(82))
        h = f.compose(g)

        assert h.degree == (0,)

    def test_mixed_degrees(self) -> None:
        """Test outer with different degrees per direction."""
        surface = _random_bezier((1, 4), 2, rng=np.random.default_rng(83))
        curve = _random_bezier((3,), 2, rng=np.random.default_rng(84))
        h = surface.compose(curve)

        assert h.degree == (15,)  # (1+4)*3
        _verify_composition(surface, curve, h)

    def test_float32(self) -> None:
        """Test composition with float32 operands (1D → 1D)."""
        f_ctrl = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        g_ctrl = np.array([[0.2], [0.8]], dtype=np.float32)
        f = Bezier(f_ctrl)
        g = Bezier(g_ctrl)
        h = f.compose(g)

        assert h.dtype == np.float32
        _verify_composition(f, g, h, atol=1e-5)

    def test_float32_nd(self) -> None:
        """Test composition with float32 operands (2D surface → 1D curve)."""
        surface = _random_bezier((2, 3), 2, rng=np.random.default_rng(85))
        curve = _random_bezier((2,), 2, rng=np.random.default_rng(86))
        # Cast to float32
        surface_f32 = Bezier(surface.control_points.astype(np.float32))
        curve_f32 = Bezier(curve.control_points.astype(np.float32))
        h = surface_f32.compose(curve_f32)

        assert h.dtype == np.float32
        assert h.dim == 1
        _verify_composition(surface_f32, curve_f32, h, atol=1e-4)

    def test_degree_zero_inner_one_direction(self) -> None:
        """Test composing a 2D surface with an inner that is constant in one direction."""
        surface = _random_bezier((2, 3), 2, rng=np.random.default_rng(87))
        # Inner has degree (2, 0): linear in s, constant in t — maps to a curve
        inner = Bezier(np.array([[[0.3, 0.5]], [[0.7, 0.5]]], dtype=np.float64))
        assert inner.degree == (1, 0)
        h = surface.compose(inner)

        assert h.dim == 2  # noqa: PLR2004
        assert h.degree == (5, 0)  # (2+3)*1, (2+3)*0
        _verify_composition(surface, inner, h)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestComposeErrors:
    """Test error handling for Bezier composition."""

    def test_rational_outer(self) -> None:
        """Test that rational outer raises TypeError."""
        f = Bezier(np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]), is_rational=True)
        g = Bezier(np.array([[0.0], [1.0]]))
        with pytest.raises(TypeError, match="outer is rational"):
            f.compose(g)

    def test_rational_inner(self) -> None:
        """Test that rational inner raises TypeError."""
        f = Bezier(np.array([[0.0], [1.0]]))
        g = Bezier(np.array([[0.0, 1.0], [1.0, 1.0]]), is_rational=True)
        with pytest.raises(TypeError, match="inner is rational"):
            f.compose(g)

    def test_rank_dim_mismatch(self) -> None:
        """Test that rank/dim mismatch raises ValueError."""
        f = Bezier(np.array([[[0.0], [1.0]], [[2.0], [3.0]]]))  # dim=2
        g = Bezier(np.array([[0.0], [1.0]]))  # rank=1, but outer.dim=2
        with pytest.raises(ValueError, match=r"rank.*must equal.*dimension"):
            f.compose(g)

    def test_dtype_mismatch(self) -> None:
        """Test that dtype mismatch raises ValueError."""
        f = Bezier(np.array([[0.0], [1.0]], dtype=np.float64))
        g = Bezier(np.array([[0.0], [1.0]], dtype=np.float32))
        with pytest.raises(ValueError, match="dtype"):
            f.compose(g)
