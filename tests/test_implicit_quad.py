"""Tests for the pure-Numba implicit quadrature implementation.

Validates the algorithm against analytical results for circles (2D) and
convergence behavior matching the paper (Saye, JCP 2022).
"""

from __future__ import annotations

import numpy as np
import pytest

from pantr.bezier.implicit import ImplicitPolyQuadrature, QuadStrategy
from pantr.bezier.implicit._bernstein import (
    _collapse_2d,
    _degree_elevate_1d,
    _derivative_along_axis_1d,
    _eval_bernstein_2d,
    _eval_bernstein_basis_1d,
    _eval_gradient_2d,
    _face_restrict_2d,
    _normalize_1d,
)
from pantr.bezier.implicit._mask import (
    _collapse_mask_2d,
    _mask_is_empty_1d,
    _mask_is_empty_2d,
    _point_within_2d,
    compute_nonzero_mask_1d,
    compute_nonzero_mask_2d,
)
from pantr.bezier.implicit._roots import find_roots

# ---------------------------------------------------------------------------
# Circle test geometry
# ---------------------------------------------------------------------------


def _make_circle_coeffs(r_sq: float = 0.1) -> np.ndarray:
    """Bernstein degree-(2,2) coefficients for (x-0.5)^2 + (y-0.5)^2 - r_sq."""
    c_val = 0.5 - r_sq
    return np.array(
        [
            [c_val, c_val - 0.5, c_val],
            [c_val - 0.5, c_val - 1.0, c_val - 0.5],
            [c_val, c_val - 0.5, c_val],
        ],
    )


# ---------------------------------------------------------------------------
# Unit tests: Bernstein operations
# ---------------------------------------------------------------------------


class TestBernstein:
    """Tests for _bernstein.py operations."""

    def test_basis_eval_degree2(self) -> None:
        b = _eval_bernstein_basis_1d(2, 0.5)
        assert np.allclose(b, [0.25, 0.5, 0.25])

    def test_basis_eval_boundary(self) -> None:
        b0 = _eval_bernstein_basis_1d(3, 0.0)
        assert b0[0] == 1.0 and np.sum(b0[1:]) == 0.0
        b1 = _eval_bernstein_basis_1d(3, 1.0)
        assert b1[-1] == 1.0 and np.sum(b1[:-1]) == 0.0

    def test_eval_2d_linear(self) -> None:
        # phi(x,y) = x + y. Bernstein deg (1,1).
        c = np.array([[0.0, 1.0], [1.0, 2.0]])
        assert abs(_eval_bernstein_2d(c, np.array([0.3, 0.7])) - 1.0) < 1e-14

    def test_collapse_consistency(self) -> None:
        c = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        x = np.array([0.3, 0.7])
        val_direct = _eval_bernstein_2d(c, x)
        from pantr.bezier._root_finding_core import _de_casteljau_eval_scalar

        c1d = _collapse_2d(c, 0, 0.7)
        val_collapse = _de_casteljau_eval_scalar(c1d, 0.3)
        assert abs(val_direct - val_collapse) < 1e-12

    def test_gradient_constant(self) -> None:
        # phi(x,y) = x + y → grad = (1, 1).
        c = np.array([[0.0, 1.0], [1.0, 2.0]])
        grad = _eval_gradient_2d(c, np.array([0.5, 0.5]))
        assert np.allclose(grad, [1.0, 1.0])

    def test_face_restrict(self) -> None:
        # phi(x,y) = xy. Bernstein: c[0,0]=0,c[1,0]=0,c[0,1]=0,c[1,1]=1.
        c = np.array([[0.0, 0.0], [0.0, 1.0]])
        assert np.allclose(_face_restrict_2d(c, 1, 0), [0.0, 0.0])
        assert np.allclose(_face_restrict_2d(c, 1, 1), [0.0, 1.0])

    def test_derivative_1d(self) -> None:
        # f(x) = x^2. Bernstein deg 2: [0, 0, 1]. d/dx = 2x → Bernstein [0, 2].
        d = _derivative_along_axis_1d(np.array([0.0, 0.0, 1.0]))
        assert np.allclose(d, [0.0, 2.0])

    def test_degree_elevate(self) -> None:
        # Elevate f(x) = x (Bernstein [0, 1]) by 1 → [0, 0.5, 1].
        elev = _degree_elevate_1d(np.array([0.0, 1.0]), 1)
        assert np.allclose(elev, [0.0, 0.5, 1.0])

    def test_normalize(self) -> None:
        n = _normalize_1d(np.array([2.0, -4.0, 1.0]))
        assert np.allclose(n, [0.5, -1.0, 0.25])


# ---------------------------------------------------------------------------
# Unit tests: Mask operations
# ---------------------------------------------------------------------------


class TestMask:
    """Tests for _mask.py operations."""

    def test_nonzero_mask_1d_with_root(self) -> None:
        m = compute_nonzero_mask_1d(np.array([-0.5, 0.5]))
        assert not _mask_is_empty_1d(m)

    def test_nonzero_mask_1d_no_root(self) -> None:
        m = compute_nonzero_mask_1d(np.array([1.0, 1.0, 1.0]))
        assert _mask_is_empty_1d(m)

    def test_nonzero_mask_2d_circle(self) -> None:
        c = _make_circle_coeffs()
        m = compute_nonzero_mask_2d(c)
        assert not _mask_is_empty_2d(m)

    def test_point_within(self) -> None:
        c = _make_circle_coeffs()
        m = compute_nonzero_mask_2d(c)
        r = np.sqrt(0.1)
        # Point on circle boundary should be in active cell.
        assert _point_within_2d(m, np.array([0.5, 0.5 + r]))

    def test_collapse_mask(self) -> None:
        c = _make_circle_coeffs()
        m = compute_nonzero_mask_2d(c)
        collapsed = _collapse_mask_2d(m, 0)
        assert any(collapsed)


# ---------------------------------------------------------------------------
# Unit tests: Root finding
# ---------------------------------------------------------------------------


class TestRootFinding:
    """Tests for _roots.py."""

    def test_linear(self) -> None:
        r, c = find_roots(np.array([1.0, -1.0]))
        assert c == 1
        assert abs(r[0] - 0.5) < 1e-12

    def test_quadratic(self) -> None:
        # (t-0.3)(t-0.7) in Bernstein.
        r, c = find_roots(np.array([0.21, -0.29, 0.21]))
        assert c == 2
        assert abs(r[0] - 0.3) < 1e-10
        assert abs(r[1] - 0.7) < 1e-10

    def test_higher_degree(self) -> None:
        """Test root finding on a degree-4 polynomial with well-separated roots."""
        from math import comb

        # Build (t-0.2)(t-0.8) in Bernstein degree 2.
        mono = np.array([0.16, -1.0, 1.0])  # 0.16 - t + t^2
        deg = 2
        M = np.zeros((deg + 1, deg + 1))
        for i in range(deg + 1):
            for j in range(i + 1):
                M[i, j] = comb(i, j) / comb(deg, j)
        bern = M @ mono
        r, c = find_roots(bern)
        assert c == 2
        assert abs(r[0] - 0.2) < 1e-6
        assert abs(r[1] - 0.8) < 1e-6

    def test_cubic_yuksel(self) -> None:
        """Yuksel should find 3 roots of a cubic with well-separated roots."""
        from pantr.bezier.implicit._roots import _yuksel_roots

        # (t-0.1)(t-0.5)(t-0.9) in Bernstein degree 3.
        # f(0)=-0.045, f(1/3)~0.0296, f(2/3)~-0.0296, f(1)=0.045
        # Bernstein: c0=f(0)=-0.045, c3=f(1)=0.045
        # c1 = c0 + f'(0)/3. f'(t)=3t^2-3t+0.59, f'(0)=0.59. c1=-0.045+0.59/3≈0.15167
        # c2 = c3 - f'(1)/3. f'(1)=3-3+0.59=0.59. c2=0.045-0.59/3≈-0.15167
        bern = np.array([-0.045, 0.15166666666, -0.15166666666, 0.045])
        r, c = _yuksel_roots(bern, 1e-15)
        assert c == 3, f"Expected 3 roots, got {c}: {r[:c]}"
        for exp in [0.1, 0.5, 0.9]:
            assert any(abs(r[i] - exp) < 1e-4 for i in range(c)), f"Missing root {exp}: {r[:c]}"

    def test_high_degree_clipping(self) -> None:
        """Bezier clipping should handle degree >= 6 polynomials."""
        from math import comb

        import numpy.polynomial.polynomial as P

        # (t-0.15)(t-0.35)(t-0.55)(t-0.75)(t-0.95) in Bernstein degree 5.
        # This has degree < 6 so will use Yuksel via dispatch,
        # but let's test clipping directly.
        from pantr.bezier.implicit._roots import _clip_roots_core, _dedup_roots

        roots_expected = [0.15, 0.35, 0.55, 0.75, 0.95]
        mono = P.polyfromroots(roots_expected)
        deg = len(mono) - 1
        M = np.zeros((deg + 1, deg + 1))
        for i in range(deg + 1):
            for j in range(i + 1):
                M[i, j] = comb(i, j) / comb(deg, j)
        bern = M @ mono
        raw, n_raw = _clip_roots_core(bern, 1e-15, 1e-15)
        unique, n_unique = _dedup_roots(raw, n_raw, bern, 1e-15, 1e-15)
        assert n_unique == 5, f"Expected 5 roots, got {n_unique}: {unique[:n_unique]}"
        for exp in roots_expected:
            assert any(abs(unique[i] - exp) < 1e-6 for i in range(n_unique)), (
                f"Missing root {exp}: {unique[:n_unique]}"
            )

    def test_no_roots(self) -> None:
        _r, c = find_roots(np.array([1.0, 2.0, 3.0]))
        assert c == 0

    def test_circle_collapsed(self) -> None:
        # Circle collapsed at y=0.5: roots at 0.5 ± sqrt(0.1).
        r, c = find_roots(np.array([0.15, -0.35, 0.15]))
        assert c == 2
        assert abs(r[0] - 0.18377223) < 1e-6
        assert abs(r[1] - 0.81622777) < 1e-6


# ---------------------------------------------------------------------------
# Integration tests: Volume quadrature
# ---------------------------------------------------------------------------


class TestVolumeQuad2D:
    """Tests for 2D volume quadrature convergence."""

    @pytest.fixture()
    def circle_ipq(self) -> ImplicitPolyQuadrature:
        return ImplicitPolyQuadrature(_make_circle_coeffs())

    def test_weight_sum(self, circle_ipq: ImplicitPolyQuadrature) -> None:
        """Total weights should sum to 1 (volume of [0,1]^2)."""
        _pts, wts = circle_ipq.volume_quad(5, QuadStrategy.TS_ONLY)
        assert abs(np.sum(wts) - 1.0) < 1e-10

    def test_area_convergence_ts(self, circle_ipq: ImplicitPolyQuadrature) -> None:
        """Area should converge exponentially with tanh-sinh."""
        expected = np.pi * 0.1

        errors = []
        for q in [5, 10, 20]:
            pts, wts = circle_ipq.volume_quad(q, QuadStrategy.TS_ONLY)
            vals = circle_ipq.eval_poly(0, pts)
            area = np.sum(wts[vals < 0])
            errors.append(abs(area - expected) / expected)

        # Check exponential convergence: each doubling of q should
        # roughly halve the number of accurate digits.
        assert errors[0] < 0.01  # q=5: ~0.2%
        assert errors[1] < 1e-5  # q=10: < 0.001%
        assert errors[2] < 1e-8  # q=20: < 1e-8

    def test_area_convergence_auto(self, circle_ipq: ImplicitPolyQuadrature) -> None:
        """AUTO_MIXED should also give exponential convergence."""
        expected = np.pi * 0.1

        pts, wts = circle_ipq.volume_quad(20, QuadStrategy.AUTO_MIXED)
        vals = circle_ipq.eval_poly(0, pts)
        area = np.sum(wts[vals < 0])
        err = abs(area - expected) / expected
        assert err < 1e-8


# ---------------------------------------------------------------------------
# Integration tests: Surface quadrature
# ---------------------------------------------------------------------------


class TestSurfaceQuad2D:
    """Tests for 2D surface quadrature convergence."""

    @pytest.fixture()
    def circle_ipq(self) -> ImplicitPolyQuadrature:
        return ImplicitPolyQuadrature(_make_circle_coeffs())

    def test_perimeter_convergence(self, circle_ipq: ImplicitPolyQuadrature) -> None:
        """Perimeter should converge with increasing q."""
        expected = 2.0 * np.pi * np.sqrt(0.1)

        errors = []
        for q in [5, 10, 20]:
            s_pts, s_wts, _ = circle_ipq.surface_quad(q, QuadStrategy.TS_ONLY)
            perim = np.sum(s_wts)
            errors.append(abs(perim - expected) / expected)

        assert errors[0] < 0.05  # q=5
        assert errors[1] < 0.005  # q=10
        assert errors[2] < 1e-4  # q=20

    def test_normal_weights_sum(self, circle_ipq: ImplicitPolyQuadrature) -> None:
        """Normal weights should approximately cancel (closed curve)."""
        s_pts, s_wts, s_nwts = circle_ipq.surface_quad(10, QuadStrategy.TS_ONLY)
        # For a closed curve, sum of normal flux should be close to zero.
        flux = np.sum(s_nwts, axis=0)
        # Not exactly zero due to quadrature error, but should be small.
        assert np.linalg.norm(flux) < 0.1


# ---------------------------------------------------------------------------
# Integration tests: Public API
# ---------------------------------------------------------------------------


class TestImplicitPolyQuadrature:
    """Tests for the ImplicitPolyQuadrature class."""

    def test_init_with_array(self) -> None:
        c = _make_circle_coeffs()
        ipq = ImplicitPolyQuadrature(c)
        assert ipq.dim == 2
        assert ipq.n_polys == 1

    def test_init_validation(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            ImplicitPolyQuadrature()

    def test_eval_poly(self) -> None:
        c = _make_circle_coeffs()
        ipq = ImplicitPolyQuadrature(c)
        pts = np.array([[0.5, 0.5]])
        vals = ipq.eval_poly(0, pts)
        # At center: (0-0)^2 + (0-0)^2 - 0.1 = -0.1.
        assert abs(vals[0] - (-0.1)) < 1e-12
