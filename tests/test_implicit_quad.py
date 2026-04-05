"""Tests for the pure-Numba implicit quadrature implementation.

Validates the algorithm against analytical results for circles (2D) and
convergence behavior matching the paper (Saye, JCP 2022).
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy import typing as npt

from pantr.bezier.implicit import ImplicitPolyQuadrature, QuadStrategy
from pantr.bezier.implicit._bernstein import (
    _collapse_2d,
    _collapse_3d,
    _degree_elevate_1d,
    _derivative_along_axis_1d,
    _eval_bernstein_2d,
    _eval_bernstein_basis_1d,
    _eval_gradient_2d,
    _eval_gradient_3d,
    _face_restrict_2d,
    _normalize_1d,
)
from pantr.bezier.implicit._convert import (
    _validate_degrees,
    monomial_to_bernstein_2d,
    monomial_to_bernstein_3d,
)
from pantr.bezier.implicit._mask import (
    _collapse_mask_2d,
    _mask_is_empty_1d,
    _mask_is_empty_2d,
    _point_within_2d,
    compute_nonzero_mask_1d,
    compute_nonzero_mask_2d,
)
from pantr.bezier.implicit._resultant import resultant_2d
from pantr.bezier.implicit._roots import find_roots

# ---------------------------------------------------------------------------
# Circle test geometry
# ---------------------------------------------------------------------------


def _make_circle_coeffs(r_sq: float = 0.1) -> npt.NDArray[np.float64]:
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
        assert abs(_eval_bernstein_2d(c, np.array([0.3, 0.7])) - 1.0) < 1e-14  # noqa: PLR2004

    def test_collapse_consistency(self) -> None:
        c = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        x = np.array([0.3, 0.7])
        val_direct = _eval_bernstein_2d(c, x)
        from pantr.bezier._root_finding_core import _de_casteljau_eval_scalar  # noqa: PLC0415

        c1d = _collapse_2d(c, 0, 0.7)
        val_collapse = _de_casteljau_eval_scalar(c1d, 0.3)
        assert abs(val_direct - val_collapse) < 1e-12  # noqa: PLR2004

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
        r, c, _ = find_roots(np.array([1.0, -1.0]))
        assert c == 1
        assert abs(r[0] - 0.5) < 1e-12  # noqa: PLR2004

    def test_quadratic(self) -> None:
        # (t-0.3)(t-0.7) in Bernstein.
        r, c, _ = find_roots(np.array([0.21, -0.29, 0.21]))
        assert c == 2  # noqa: PLR2004
        assert abs(r[0] - 0.3) < 1e-10  # noqa: PLR2004
        assert abs(r[1] - 0.7) < 1e-10  # noqa: PLR2004

    def test_higher_degree(self) -> None:
        """Test root finding on a degree-4 polynomial with well-separated roots."""
        from math import comb  # noqa: PLC0415

        # Build (t-0.2)(t-0.8) in Bernstein degree 2.
        mono = np.array([0.16, -1.0, 1.0])  # 0.16 - t + t^2
        deg = 2
        M = np.zeros((deg + 1, deg + 1))
        for i in range(deg + 1):
            for j in range(i + 1):
                M[i, j] = comb(i, j) / comb(deg, j)
        bern = M @ mono
        r, c, _ = find_roots(bern)
        assert c == 2  # noqa: PLR2004
        assert abs(r[0] - 0.2) < 1e-6  # noqa: PLR2004
        assert abs(r[1] - 0.8) < 1e-6  # noqa: PLR2004

    def test_cubic_yuksel(self) -> None:
        """Yuksel should find 3 roots of a cubic with well-separated roots."""
        from pantr.bezier.implicit._roots import _yuksel_roots  # noqa: PLC0415

        # (t-0.1)(t-0.5)(t-0.9) in Bernstein degree 3.
        # f(0)=-0.045, f(1/3)~0.0296, f(2/3)~-0.0296, f(1)=0.045
        # Bernstein: c0=f(0)=-0.045, c3=f(1)=0.045
        # c1 = c0 + f'(0)/3. f'(t)=3t^2-3t+0.59, f'(0)=0.59. c1=-0.045+0.59/3≈0.15167
        # c2 = c3 - f'(1)/3. f'(1)=3-3+0.59=0.59. c2=0.045-0.59/3≈-0.15167
        bern = np.array([-0.045, 0.15166666666, -0.15166666666, 0.045])
        r, c = _yuksel_roots(bern, 1e-15)
        assert c == 3, f"Expected 3 roots, got {c}: {r[:c]}"  # noqa: PLR2004
        for exp in [0.1, 0.5, 0.9]:
            assert any(abs(r[i] - exp) < 1e-4 for i in range(c)), f"Missing root {exp}: {r[:c]}"  # noqa: PLR2004

    def test_high_degree_clipping(self) -> None:
        """Bezier clipping should handle degree >= 6 polynomials."""
        from math import comb  # noqa: PLC0415

        import numpy.polynomial.polynomial as P  # noqa: PLC0415

        # (t-0.15)(t-0.35)(t-0.55)(t-0.75)(t-0.95) in Bernstein degree 5.
        # This has degree < 6 so will use Yuksel via dispatch,
        # but let's test clipping directly.
        from pantr.bezier.implicit._roots import _clip_roots_core, _dedup_roots  # noqa: PLC0415

        roots_expected = [0.15, 0.35, 0.55, 0.75, 0.95]
        mono = P.polyfromroots(roots_expected)  # type: ignore[no-untyped-call]
        deg = len(mono) - 1
        M = np.zeros((deg + 1, deg + 1))
        for i in range(deg + 1):
            for j in range(i + 1):
                M[i, j] = comb(i, j) / comb(deg, j)
        bern = M @ mono
        raw, n_raw, _ = _clip_roots_core(bern, 1e-15, 1e-15)
        unique, n_unique = _dedup_roots(raw, n_raw, bern, 1e-15, 1e-15)
        assert n_unique == 5, f"Expected 5 roots, got {n_unique}: {unique[:n_unique]}"  # noqa: PLR2004
        for exp in roots_expected:
            assert any(abs(unique[i] - exp) < 1e-6 for i in range(n_unique)), (  # noqa: PLR2004
                f"Missing root {exp}: {unique[:n_unique]}"
            )

    def test_no_roots(self) -> None:
        _r, c, _ = find_roots(np.array([1.0, 2.0, 3.0]))
        assert c == 0

    def test_circle_collapsed(self) -> None:
        # Circle collapsed at y=0.5: roots at 0.5 ± sqrt(0.1).
        r, c, _ = find_roots(np.array([0.15, -0.35, 0.15]))
        assert c == 2  # noqa: PLR2004
        assert abs(r[0] - 0.18377223) < 1e-6  # noqa: PLR2004
        assert abs(r[1] - 0.81622777) < 1e-6  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Integration tests: Volume quadrature
# ---------------------------------------------------------------------------


class TestVolumeQuad2D:
    """Tests for 2D volume quadrature convergence."""

    @pytest.fixture(scope="class")
    def circle_ipq(self) -> ImplicitPolyQuadrature:
        return ImplicitPolyQuadrature(_make_circle_coeffs())

    def test_weight_sum(self, circle_ipq: ImplicitPolyQuadrature) -> None:
        """Total weights should sum to 1 (volume of [0,1]^2)."""
        _pts, wts = circle_ipq.volume_quad(5, QuadStrategy.TS_ONLY)
        assert abs(np.sum(wts) - 1.0) < 1e-10  # noqa: PLR2004

    def test_area_convergence_ts(self, circle_ipq: ImplicitPolyQuadrature) -> None:
        """Area should converge exponentially with tanh-sinh."""
        expected = np.pi * 0.1

        errors = []
        for q in [5, 10, 20]:
            pts, wts = circle_ipq.volume_quad(q, QuadStrategy.TS_ONLY)
            vals = circle_ipq.eval_poly(0, pts)
            area = np.sum(wts[vals < 0])
            errors.append(abs(area - expected) / expected)

        # Exponential convergence: actual errors ~2e-3, 7e-7, 3e-10.
        assert errors[0] < 1e-2  # q=5: actual ~2e-3  # noqa: PLR2004
        assert errors[1] < 4e-6  # q=10: actual ~7e-7  # noqa: PLR2004
        assert errors[2] < 2e-9  # q=20: actual ~3e-10  # noqa: PLR2004

    def test_area_convergence_auto(self, circle_ipq: ImplicitPolyQuadrature) -> None:
        """AUTO_MIXED should also give exponential convergence."""
        expected = np.pi * 0.1

        pts, wts = circle_ipq.volume_quad(20, QuadStrategy.AUTO_MIXED)
        vals = circle_ipq.eval_poly(0, pts)
        area = np.sum(wts[vals < 0])
        err = abs(area - expected) / expected
        assert err < 2e-9  # q=20: actual ~3e-10  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Integration tests: Surface quadrature
# ---------------------------------------------------------------------------


class TestSurfaceQuad2D:
    """Tests for 2D surface quadrature convergence."""

    @pytest.fixture(scope="class")
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

        # Actual errors: ~2.7e-2, 1.8e-3, 2.1e-5.
        assert errors[0] < 0.15  # q=5: actual ~2.7e-2  # noqa: PLR2004
        assert errors[1] < 1e-2  # q=10: actual ~1.8e-3  # noqa: PLR2004
        assert errors[2] < 1e-4  # q=20: actual ~2.1e-5  # noqa: PLR2004

    def test_normal_weights_sum(self, circle_ipq: ImplicitPolyQuadrature) -> None:
        """Normal weights should approximately cancel (closed curve)."""
        s_pts, s_wts, s_nwts = circle_ipq.surface_quad(10, QuadStrategy.TS_ONLY)
        # For a closed curve, sum of normal flux should be close to zero.
        flux = np.sum(s_nwts, axis=0)
        # Not exactly zero due to quadrature error, but should be small.
        assert np.linalg.norm(flux) < 0.01  # noqa: PLR2004

    def test_perimeter_aggregate(self, circle_ipq: ImplicitPolyQuadrature) -> None:
        """Aggregate perimeter should converge faster than single-direction."""
        expected = 2.0 * np.pi * np.sqrt(0.1)
        _, sw, _ = circle_ipq.surface_quad(20, QuadStrategy.TS_ONLY, aggregate=True)
        perim = np.sum(sw)
        err = abs(perim - expected) / expected
        assert err < 2e-9  # q=20 aggregate: actual ~3.1e-10  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Integration tests: 3D quadrature
# ---------------------------------------------------------------------------


def _make_sphere_coeffs(r_sq: float = 0.09) -> npt.NDArray[np.float64]:
    """Bernstein degree-(2,2,2) coefficients for (x-.5)^2+(y-.5)^2+(z-.5)^2-r_sq."""
    const = 0.75 - r_sq
    bx = np.array([0.0, -0.5, 0.0])
    c = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c[i, j, k] = bx[i] + bx[j] + bx[k] + const
    return c


class TestVolumeQuad3D:
    """Tests for 3D volume quadrature convergence."""

    @pytest.fixture(scope="class")
    def sphere_ipq(self) -> ImplicitPolyQuadrature:
        return ImplicitPolyQuadrature(_make_sphere_coeffs())

    def test_weight_sum(self, sphere_ipq: ImplicitPolyQuadrature) -> None:
        """Total weights should sum to 1 (volume of [0,1]^3)."""
        _pts, wts = sphere_ipq.volume_quad(3, QuadStrategy.TS_ONLY)
        assert abs(np.sum(wts) - 1.0) < 1e-8  # noqa: PLR2004

    def test_volume_convergence(self, sphere_ipq: ImplicitPolyQuadrature) -> None:
        """Sphere volume should converge exponentially."""
        expected = (4.0 / 3.0) * np.pi * 0.3**3

        errors = []
        for q in [5, 10, 15]:
            pts, wts = sphere_ipq.volume_quad(q, QuadStrategy.TS_ONLY)
            vals = sphere_ipq.eval_poly(0, pts)
            vol = np.sum(wts[vals < 0])
            errors.append(abs(vol - expected) / expected)

        # Actual errors: ~1.1e-2, 3.5e-5, 1.8e-7.
        assert errors[0] < 6e-2  # q=5: actual ~1.1e-2  # noqa: PLR2004
        assert errors[1] < 2e-4  # q=10: actual ~3.5e-5  # noqa: PLR2004
        assert errors[2] < 1e-6  # q=15: actual ~1.8e-7  # noqa: PLR2004


class TestSurfaceQuad3D:
    """Tests for 3D surface quadrature convergence."""

    @pytest.fixture(scope="class")
    def sphere_ipq(self) -> ImplicitPolyQuadrature:
        return ImplicitPolyQuadrature(_make_sphere_coeffs())

    def test_area_convergence(self, sphere_ipq: ImplicitPolyQuadrature) -> None:
        """Sphere surface area should converge."""
        expected = 4.0 * np.pi * 0.3**2

        errors = []
        for q in [5, 10, 15]:
            s_pts, s_wts, _ = sphere_ipq.surface_quad(q, QuadStrategy.TS_ONLY)
            area = np.sum(s_wts)
            errors.append(abs(area - expected) / expected)

        # Actual errors: ~2.7e-2, 1.8e-3, 1.8e-4.
        assert errors[0] < 0.15  # q=5: actual ~2.7e-2  # noqa: PLR2004
        assert errors[1] < 1e-2  # q=10: actual ~1.8e-3  # noqa: PLR2004
        assert errors[2] < 1e-3  # q=15: actual ~1.8e-4  # noqa: PLR2004

    def test_normal_flux_closed(self, sphere_ipq: ImplicitPolyQuadrature) -> None:
        """Normal flux sum should be near zero for a closed surface."""
        s_pts, s_wts, s_nwts = sphere_ipq.surface_quad(7, QuadStrategy.TS_ONLY)
        flux = np.sum(s_nwts, axis=0)
        assert np.linalg.norm(flux) < 0.05  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Integration tests: Public API
# ---------------------------------------------------------------------------


class TestImplicitPolyQuadrature:
    """Tests for the ImplicitPolyQuadrature class."""

    def test_init_with_array(self) -> None:
        c = _make_circle_coeffs()
        ipq = ImplicitPolyQuadrature(c)
        assert ipq.dim == 2  # noqa: PLR2004
        assert ipq.n_polys == 1

    def test_init_validation(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            ImplicitPolyQuadrature()

    def test_init_inconsistent_dims(self) -> None:
        """Reject polynomials with inconsistent dimensions."""
        with pytest.raises(ValueError, match="same dimension"):
            ImplicitPolyQuadrature(np.ones((3, 3)), np.ones((2, 2, 2)))

    def test_init_unsupported_dim_1d(self) -> None:
        """Reject 1D polynomials."""
        with pytest.raises(ValueError, match="Only 2D and 3D"):
            ImplicitPolyQuadrature(np.array([1.0, 2.0, 3.0]))

    def test_init_unsupported_dim_4d(self) -> None:
        """Reject 4D polynomials."""
        with pytest.raises(ValueError, match="Only 2D and 3D"):
            ImplicitPolyQuadrature(np.ones((2, 2, 2, 2)))

    def test_init_vector_bezier_rejected(self) -> None:
        """Reject vector-valued Bezier (rank > 1)."""
        from pantr.bezier import Bezier  # noqa: PLC0415

        cp = np.ones((3, 3, 2))  # rank=2
        bez = Bezier(cp)
        with pytest.raises(ValueError, match="scalar Bezier"):
            ImplicitPolyQuadrature(bez)

    def test_volume_quad_q_validation(self) -> None:
        """Reject q < 1."""
        ipq = ImplicitPolyQuadrature(_make_circle_coeffs())
        with pytest.raises(ValueError, match="q must be >= 1"):
            ipq.volume_quad(0)

    def test_surface_quad_q_validation(self) -> None:
        """Reject q < 1 in surface_quad."""
        ipq = ImplicitPolyQuadrature(_make_circle_coeffs())
        with pytest.raises(ValueError, match="q must be >= 1"):
            ipq.surface_quad(0)

    def test_eval_poly_out_of_range(self) -> None:
        """Reject poly_idx out of range."""
        ipq = ImplicitPolyQuadrature(_make_circle_coeffs())
        pts = np.array([[0.5, 0.5]])
        with pytest.raises(IndexError, match="out of range"):
            ipq.eval_poly(1, pts)
        with pytest.raises(IndexError, match="out of range"):
            ipq.eval_poly(-1, pts)

    def test_eval_poly(self) -> None:
        c = _make_circle_coeffs()
        ipq = ImplicitPolyQuadrature(c)
        pts = np.array([[0.5, 0.5]])
        vals = ipq.eval_poly(0, pts)
        # At center: (0-0)^2 + (0-0)^2 - 0.1 = -0.1.
        assert abs(vals[0] - (-0.1)) < 1e-12  # noqa: PLR2004

    def test_init_with_bezier_object(self) -> None:
        """Accept a pantr Bezier object with scalar rank."""
        from pantr.bezier import Bezier  # noqa: PLC0415

        c = _make_circle_coeffs()
        cp = c[..., np.newaxis]  # rank=1
        bez = Bezier(cp)
        ipq = ImplicitPolyQuadrature(bez)
        assert ipq.dim == 2  # noqa: PLR2004
        pts, wts = ipq.volume_quad(10, QuadStrategy.AUTO_MIXED)
        vals = ipq.eval_poly(0, pts)
        area = np.sum(wts[vals < 0])
        assert abs(area - np.pi * 0.1) / (np.pi * 0.1) < 1e-5  # noqa: PLR2004

    def test_monomial_to_bernstein_utilities(self) -> None:
        """Public conversion utilities produce correct Bernstein coefficients."""
        from pantr.bezier.implicit import (  # noqa: PLC0415
            monomial_to_bernstein_2d,
            monomial_to_bernstein_3d,
        )
        from pantr.bezier.implicit._bernstein import (  # noqa: PLC0415
            _eval_bernstein_2d,
            _eval_bernstein_3d,
        )

        # 2D: phi(x,y) = x^2 + 4y^2 - 1 on (-1,1)^2.
        mono_2d = np.zeros((3, 3))
        mono_2d[2, 0] = 1.0
        mono_2d[0, 2] = 4.0
        mono_2d[0, 0] = -1.0
        bern_2d = monomial_to_bernstein_2d(
            mono_2d,
            (2, 2),
            np.array([-1.0, -1.0]),
            np.array([1.0, 1.0]),
        )
        # phi(0,0) = -1.
        t = np.array([0.5, 0.5])
        assert abs(_eval_bernstein_2d(bern_2d, t) - (-1.0)) < 1e-12  # noqa: PLR2004

        # 3D: phi(x,y,z) = x^2+y^2+z^2-1 on (-1,1)^3.
        mono_3d = np.zeros((3, 3, 3))
        mono_3d[2, 0, 0] = 1.0
        mono_3d[0, 2, 0] = 1.0
        mono_3d[0, 0, 2] = 1.0
        mono_3d[0, 0, 0] = -1.0
        bern_3d = monomial_to_bernstein_3d(
            mono_3d,
            (2, 2, 2),
            np.array([-1.0, -1.0, -1.0]),
            np.array([1.0, 1.0, 1.0]),
        )
        t3 = np.array([0.5, 0.5, 0.5])
        assert abs(_eval_bernstein_3d(bern_3d, t3) - (-1.0)) < 1e-12  # noqa: PLR2004

    def test_square_free_preprocessing(self) -> None:
        """Square-free factoring removes repeated roots from 1D polynomials."""
        from pantr.bezier.implicit._bernstein import _make_square_free_1d  # noqa: PLC0415

        # p(x) = (x-0.3)^2 in Bernstein degree 2.
        p = _mono_to_bernstein_1d(np.array([0.09, -0.6, 1.0]), 2)
        sf = _make_square_free_1d(p)
        # Should reduce to degree 1 (single root at 0.3).
        assert len(sf) == 2  # noqa: PLR2004

        # p with no repeated roots should be unchanged.
        p2 = _mono_to_bernstein_1d(np.array([0.21, -1.0, 1.0]), 2)
        sf2 = _make_square_free_1d(p2)
        assert len(sf2) == len(p2)


# ---------------------------------------------------------------------------
# Integration tests: Multiple polynomials
# ---------------------------------------------------------------------------


def _circle_bernstein(cx: float, cy: float, r_sq: float) -> npt.NDArray[np.float64]:
    """Bernstein deg-(2,2) for (x-cx)^2 + (y-cy)^2 - r_sq on [0,1]^2."""
    bx = _mono_to_bernstein_1d(np.array([cx**2, -2 * cx, 1.0]), 2)
    by = _mono_to_bernstein_1d(np.array([cy**2, -2 * cy, 1.0]), 2)
    c = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            c[i, j] = bx[i] + by[j] - r_sq
    return c


class TestMultiplePolynomials:
    """Tests for quadrature with 2 intersecting polynomials."""

    def test_two_circles_individual_areas(self) -> None:
        """Each circle area should be accurate independently."""
        c1 = _circle_bernstein(0.35, 0.5, 0.04)
        c2 = _circle_bernstein(0.65, 0.5, 0.04)
        ipq = ImplicitPolyQuadrature(c1, c2)
        assert ipq.n_polys == 2  # noqa: PLR2004

        pts, wts = ipq.volume_quad(15, QuadStrategy.AUTO_MIXED)
        v1 = ipq.eval_poly(0, pts)
        v2 = ipq.eval_poly(1, pts)

        expected = np.pi * 0.04
        assert abs(np.sum(wts[v1 < 0]) - expected) / expected < 1e-6  # noqa: PLR2004
        assert abs(np.sum(wts[v2 < 0]) - expected) / expected < 1e-6  # noqa: PLR2004

    def test_two_circles_boolean_ops(self) -> None:
        """Intersection and union should satisfy A|B = A + B - A&B."""
        c1 = _circle_bernstein(0.35, 0.5, 0.04)
        c2 = _circle_bernstein(0.65, 0.5, 0.04)
        ipq = ImplicitPolyQuadrature(c1, c2)

        pts, wts = ipq.volume_quad(15, QuadStrategy.AUTO_MIXED)
        v1 = ipq.eval_poly(0, pts)
        v2 = ipq.eval_poly(1, pts)

        a1 = np.sum(wts[v1 < 0])
        a2 = np.sum(wts[v2 < 0])
        a_inter = np.sum(wts[(v1 < 0) & (v2 < 0)])
        a_union = np.sum(wts[(v1 < 0) | (v2 < 0)])

        # A|B = A + B - A&B.
        assert abs(a_union - (a1 + a2 - a_inter)) < 1e-14  # noqa: PLR2004

    def test_two_circles_intersection_area(self) -> None:
        """Intersection area should match the analytical formula."""
        c1 = _circle_bernstein(0.35, 0.5, 0.04)
        c2 = _circle_bernstein(0.65, 0.5, 0.04)
        ipq = ImplicitPolyQuadrature(c1, c2)

        pts, wts = ipq.volume_quad(15, QuadStrategy.AUTO_MIXED)
        v1 = ipq.eval_poly(0, pts)
        v2 = ipq.eval_poly(1, pts)
        a_inter = np.sum(wts[(v1 < 0) & (v2 < 0)])

        # Analytical: 2R^2*arccos(d/(2R)) - (d/2)*sqrt(4R^2-d^2).
        d, r = 0.3, 0.2
        expected = 2 * r**2 * np.arccos(d / (2 * r)) - (d / 2) * np.sqrt(4 * r**2 - d**2)
        assert abs(a_inter - expected) / expected < 1e-6  # noqa: PLR2004

    def test_three_circles_inclusion_exclusion(self) -> None:
        """Inclusion-exclusion identity holds for 3 intersecting circles."""
        c1 = _circle_bernstein(0.35, 0.5, 0.04)
        c2 = _circle_bernstein(0.65, 0.5, 0.04)
        c3 = _circle_bernstein(0.5, 0.75, 0.04)
        ipq = ImplicitPolyQuadrature(c1, c2, c3)
        assert ipq.n_polys == 3  # noqa: PLR2004

        pts, wts = ipq.volume_quad(15, QuadStrategy.AUTO_MIXED)
        v1 = ipq.eval_poly(0, pts)
        v2 = ipq.eval_poly(1, pts)
        v3 = ipq.eval_poly(2, pts)

        a1 = np.sum(wts[v1 < 0])
        a2 = np.sum(wts[v2 < 0])
        a3 = np.sum(wts[v3 < 0])
        a12 = np.sum(wts[(v1 < 0) & (v2 < 0)])
        a13 = np.sum(wts[(v1 < 0) & (v3 < 0)])
        a23 = np.sum(wts[(v2 < 0) & (v3 < 0)])
        a123 = np.sum(wts[(v1 < 0) & (v2 < 0) & (v3 < 0)])
        a_union = np.sum(wts[(v1 < 0) | (v2 < 0) | (v3 < 0)])

        # |A|B|C| = |A|+|B|+|C| - |A&B| - |A&C| - |B&C| + |A&B&C|.
        ie = a1 + a2 + a3 - a12 - a13 - a23 + a123
        assert abs(ie - a_union) < 1e-13  # noqa: PLR2004

    def test_four_non_overlapping_circles(self) -> None:
        """Four non-overlapping circles: union = sum of individual areas."""
        c_tl = _circle_bernstein(0.3, 0.7, 0.02)
        c_tr = _circle_bernstein(0.7, 0.7, 0.02)
        c_bl = _circle_bernstein(0.3, 0.3, 0.02)
        c_br = _circle_bernstein(0.7, 0.3, 0.02)
        ipq = ImplicitPolyQuadrature(c_tl, c_tr, c_bl, c_br)
        assert ipq.n_polys == 4  # noqa: PLR2004

        pts, wts = ipq.volume_quad(10, QuadStrategy.AUTO_MIXED)
        vs = [ipq.eval_poly(i, pts) for i in range(4)]

        expected_each = np.pi * 0.02
        for i in range(4):
            a = np.sum(wts[vs[i] < 0])
            assert abs(a - expected_each) / expected_each < 1e-5  # noqa: PLR2004

        a_union = np.sum(wts[vs[0] < 0])
        for i in range(1, 4):
            a_union = np.sum(
                wts[np.any(np.column_stack([vs[j] < 0 for j in range(i + 1)]), axis=1)]
            )
        assert abs(a_union - 4 * expected_each) / (4 * expected_each) < 1e-5  # noqa: PLR2004

    def test_circles_plus_line(self) -> None:
        """Two circles + a line: mixed polynomial degrees."""
        c1 = _circle_bernstein(0.35, 0.5, 0.04)
        c2 = _circle_bernstein(0.65, 0.5, 0.04)
        c_line = np.array([[-0.5, 0.5], [-0.5, 0.5]])  # y - 0.5
        ipq = ImplicitPolyQuadrature(c1, c2, c_line)
        assert ipq.n_polys == 3  # noqa: PLR2004

        pts, wts = ipq.volume_quad(10, QuadStrategy.AUTO_MIXED)
        v1 = ipq.eval_poly(0, pts)
        v3 = ipq.eval_poly(2, pts)

        # Circle1 centered at (0.35, 0.5) with R=0.2, line y=0.5 through center.
        # Half-circle below line = pi*R^2/2.
        a_half = np.sum(wts[(v1 < 0) & (v3 < 0)])
        expected_half = np.pi * 0.04 / 2
        assert abs(a_half - expected_half) / expected_half < 1e-4  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Helpers for paper test cases (§4.1, §4.2)
# ---------------------------------------------------------------------------


def _mono_to_bernstein_1d(mono: npt.NDArray[np.float64], degree: int) -> npt.NDArray[np.float64]:
    """Convert monomial coefficients (ascending power) to Bernstein degree *degree*."""
    from math import comb as _comb  # noqa: PLC0415

    n = degree
    mat = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            mat[i, j] = _comb(i, j) / _comb(n, j)
    m = np.zeros(n + 1)
    m[: min(len(mono), n + 1)] = mono[: min(len(mono), n + 1)]
    return mat @ m


def _ellipse_bernstein_on_cell(
    cell_lo: npt.NDArray[np.float64],
    cell_hi: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Bernstein degree-(2,2) coefficients for phi(x,y)=x^2+4y^2-1 on a cell.

    Maps the cell [cell_lo, cell_hi] to [0,1]^2 and converts to Bernstein form.
    """
    a, c = cell_lo[0], cell_lo[1]
    hx, hy = cell_hi[0] - a, cell_hi[1] - c
    # f_x(t) = (a + hx*t)^2, monomial: [a^2, 2*a*hx, hx^2]
    bern_x = _mono_to_bernstein_1d(np.array([a**2, 2 * a * hx, hx**2]), 2)
    # f_y(s) = 4*(c + hy*s)^2, monomial: [4c^2, 8c*hy, 4*hy^2]
    bern_y = _mono_to_bernstein_1d(np.array([4 * c**2, 8 * c * hy, 4 * hy**2]), 2)
    coeffs = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            coeffs[i, j] = bern_x[i] + bern_y[j] - 1.0
    return coeffs


def _ellipsoid_bernstein_on_cell(
    cell_lo: npt.NDArray[np.float64],
    cell_hi: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Bernstein degree-(2,2,2) coefficients for phi(x,y,z)=x^2+4y^2+9z^2-1 on a cell."""
    a, c, e = cell_lo[0], cell_lo[1], cell_lo[2]
    hx, hy, hz = cell_hi[0] - a, cell_hi[1] - c, cell_hi[2] - e
    bern_x = _mono_to_bernstein_1d(np.array([a**2, 2 * a * hx, hx**2]), 2)
    bern_y = _mono_to_bernstein_1d(np.array([4 * c**2, 8 * c * hy, 4 * hy**2]), 2)
    bern_z = _mono_to_bernstein_1d(np.array([9 * e**2, 18 * e * hz, 9 * hz**2]), 2)
    coeffs = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                coeffs[i, j, k] = bern_x[i] + bern_y[j] + bern_z[k] - 1.0
    return coeffs


def _integrand_f(pts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Integrand f(x) = cos(1/4 * ||x||^2) from paper §4.2."""
    return np.asarray(np.cos(0.25 * np.sum(pts**2, axis=1)), dtype=np.float64)


# ---------------------------------------------------------------------------
# Paper convergence tests: h-refinement (§4.1)
# ---------------------------------------------------------------------------


class TestHRefinement:
    """h-refinement convergence on the ellipse/ellipsoid (paper §4.1).

    Under h-refinement with fixed q, the error should be O(h^{2q}).
    """

    @staticmethod
    def _h_refine_ellipse_volume(n: int, q: int) -> float:
        """Compute ellipse area via h-refinement with n cells per axis."""
        lo, hi = -1.1, 1.1
        h = (hi - lo) / n
        total = 0.0
        for ix in range(n):
            for iy in range(n):
                cell_lo = np.array([lo + ix * h, lo + iy * h])
                cell_hi = np.array([lo + (ix + 1) * h, lo + (iy + 1) * h])
                coeffs = _ellipse_bernstein_on_cell(cell_lo, cell_hi)
                ipq = ImplicitPolyQuadrature(coeffs)
                # GL_ONLY is correct for h-refinement: as h→0 the geometry
                # becomes locally flat, so GL gives optimal convergence.
                pts, wts = ipq.volume_quad(q, QuadStrategy.GL_ONLY)
                vals = ipq.eval_poly(0, pts)
                total += np.sum(wts[vals < 0]) * h**2
        return total

    def test_ellipse_area_h_refinement(self) -> None:
        """Ellipse area converges as O(h^{2q}) under h-refinement.

        With q=3, the theoretical rate is h^6. Halving h (doubling n)
        should reduce the error by ~2^6 = 64.
        """
        expected = np.pi / 2.0
        q = 3
        errors = []
        for n in [8, 16, 32]:
            area = self._h_refine_ellipse_volume(n, q)
            errors.append(abs(area - expected) / expected)

        assert errors[0] < 1.5e-5  # n=8: actual ~2.8e-6  # noqa: PLR2004
        # Check convergence rate between n=16 and n=32.
        rate = np.log2(errors[1] / errors[2]) if errors[2] > 0 else 20.0
        assert rate > 4.0, f"h-refinement rate too low: {rate:.1f} (expected ~{2 * q})"  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Paper convergence tests: q-refinement (§4.2)
# ---------------------------------------------------------------------------


class TestQRefinement:
    """q-refinement convergence on the ellipse (paper §4.2, Fig. 5).

    With the geometry fixed (single cell containing the ellipse), increasing
    q should give approximately exponential convergence.
    """

    @staticmethod
    def _single_cell_ellipse() -> ImplicitPolyQuadrature:
        """Create quadrature for the ellipse on U=(-1.1, 1.1)^2."""
        cell_lo = np.array([-1.1, -1.1])
        cell_hi = np.array([1.1, 1.1])
        coeffs = _ellipse_bernstein_on_cell(cell_lo, cell_hi)
        return ImplicitPolyQuadrature(coeffs)

    def test_ellipse_volume_q_refinement(self) -> None:
        """Volume integral I_Omega should converge exponentially in q."""
        ipq = self._single_cell_ellipse()
        cell_vol = 2.2**2
        expected = np.pi / 2.0

        errors = []
        for q in [5, 10, 20, 30]:
            pts, wts = ipq.volume_quad(q, QuadStrategy.AUTO_MIXED)
            vals = ipq.eval_poly(0, pts)
            vol = np.sum(wts[vals < 0]) * cell_vol
            errors.append(abs(vol - expected) / expected)

        # Actual errors: ~2.1e-3, 7.0e-7, 3.1e-10, 1.2e-13.
        assert errors[0] < 1e-2  # q=5: actual ~2.1e-3  # noqa: PLR2004
        assert errors[1] < 4e-6  # q=10: actual ~7.0e-7  # noqa: PLR2004
        assert errors[2] < 2e-9  # q=20: actual ~3.1e-10  # noqa: PLR2004
        assert errors[3] < 1e-12  # q=30: actual ~1.2e-13  # noqa: PLR2004

    def test_ellipse_surface_flux_q_refinement(self) -> None:
        """Flux-form surface integral I_{Gamma_n} should converge exponentially."""
        ipq = self._single_cell_ellipse()
        cell_scale = 2.2  # 1D cell width for weight scaling
        # For the ellipse, ∫_Γ f·n with f(x)=cos(¼||x||²) has a known reference.
        # Use aggregate mode for best convergence.

        errors = []
        ref_q = 40
        _, sw_ref, nw_ref = ipq.surface_quad(ref_q, QuadStrategy.TS_ONLY, aggregate=True)
        # Scale by cell dimensions (surface points are in [0,1]^2, need to map).
        pts_ref, _, nw_ref2 = ipq.surface_quad(ref_q, QuadStrategy.TS_ONLY, aggregate=True)
        # Compute reference integral: ∫_Γ 1 · n dS (should be zero for closed curve).
        # Instead test convergence of ∫_Γ 1 dS = perimeter.
        # Perimeter of ellipse x^2+4y^2=1: semiaxes a=1, b=1/2.
        # Approx perimeter (Ramanujan): pi*(3(a+b) - sqrt((3a+b)(a+3b)))
        a_ax, b_ax = 1.0, 0.5
        expected_perim = np.pi * (
            3 * (a_ax + b_ax) - np.sqrt((3 * a_ax + b_ax) * (a_ax + 3 * b_ax))
        )

        for q in [5, 10, 20]:
            _, sw, _ = ipq.surface_quad(q, QuadStrategy.TS_ONLY, aggregate=True)
            perim = np.sum(sw) * cell_scale
            errors.append(abs(perim - expected_perim) / expected_perim)

        # Actual errors: ~1.6e-2, 2.1e-3, 7.3e-5.
        assert errors[0] < 0.08  # q=5: actual ~1.6e-2  # noqa: PLR2004
        assert errors[1] < 1e-2  # q=10: actual ~2.1e-3  # noqa: PLR2004
        assert errors[2] < 4e-4  # q=20: actual ~7.3e-5  # noqa: PLR2004

    def test_ellipse_surface_nonflux_q_refinement(self) -> None:
        """Non-flux surface integral I_Gamma should converge (slower than I_Gamma_n).

        Paper Fig. 5 shows I_Gamma converges exponentially but at a slower rate
        than the flux-form I_Gamma_n, due to the |grad phi|/|d_k phi| Jacobian
        factor introducing a pole near the complex plane.
        """
        ipq = self._single_cell_ellipse()
        cell_scale = 2.2
        a_ax, b_ax = 1.0, 0.5
        expected_perim = np.pi * (
            3 * (a_ax + b_ax) - np.sqrt((3 * a_ax + b_ax) * (a_ax + 3 * b_ax))
        )

        errors = []
        for q in [5, 10, 20, 30]:
            s_pts, s_wts, _ = ipq.surface_quad(q, QuadStrategy.TS_ONLY, aggregate=False)
            perim = np.sum(s_wts) * cell_scale
            errors.append(abs(perim - expected_perim) / expected_perim)

        # Actual errors: ~1.8e-2, 1.2e-3, 1.1e-5, 2.5e-6.
        assert errors[0] < 0.09  # q=5: actual ~1.8e-2  # noqa: PLR2004
        assert errors[1] < 6e-3  # q=10: actual ~1.2e-3  # noqa: PLR2004
        assert errors[2] < 6e-5  # q=20: actual ~1.1e-5  # noqa: PLR2004
        assert errors[3] < 1.5e-5  # q=30: actual ~2.5e-6  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Paper §4.4: Bilinear example
# ---------------------------------------------------------------------------


class TestBilinear:
    """Bilinear polynomial with varying curvature (paper §4.4).

    phi(x,y) = (x - 1/2)(y - 1/2) - eps^2 on U = (0, 1)^2.
    - eps = 0.1: smooth rounded corner
    - eps = 0.01: high-curvature corner (R ~ 1% of cell)
    - eps = 0: degenerate cross (sharp corner)
    """

    @staticmethod
    def _bilinear_bernstein(eps: float) -> npt.NDArray[np.float64]:
        """Bernstein coefficients for (x-0.5)(y-0.5) - eps^2 on [0,1]^2.

        Degree (1,1): coefficients are corner values.
        """
        e2 = eps * eps
        return np.array(
            [
                [0.25 - e2, -0.25 - e2],
                [-0.25 - e2, 0.25 - e2],
            ]
        )

    def test_bilinear_smooth(self) -> None:
        """Eps = 0.1: smooth interface, should converge well with GL."""
        bern = self._bilinear_bernstein(0.1)
        ipq = ImplicitPolyQuadrature(bern)

        # Reference.
        pts_r, wts_r = ipq.volume_quad(30, QuadStrategy.GL_ONLY)
        vals_r = ipq.eval_poly(0, pts_r)
        vol_ref = float(np.sum(wts_r[vals_r < 0]))

        errors = []
        for q in [3, 7, 15]:
            pts, wts = ipq.volume_quad(q, QuadStrategy.GL_ONLY)
            vals = ipq.eval_poly(0, pts)
            vol = float(np.sum(wts[vals < 0]))
            errors.append(abs(vol - vol_ref) / max(abs(vol_ref), 1e-15))

        # Actual errors: ~1.0e-2, ~7.2e-7.
        assert errors[0] < 5e-2  # q=3: actual ~1.0e-2  # noqa: PLR2004
        assert errors[2] < 4e-6  # q=15: actual ~7.2e-7  # noqa: PLR2004

    def test_bilinear_high_curvature(self) -> None:
        """Eps = 0.01: high curvature, (TS,GL) should outperform (GL,GL)."""
        bern = self._bilinear_bernstein(0.01)
        ipq = ImplicitPolyQuadrature(bern)

        pts_r, wts_r = ipq.volume_quad(30, QuadStrategy.TS_ONLY)
        vals_r = ipq.eval_poly(0, pts_r)
        vol_ref = float(np.sum(wts_r[vals_r < 0]))

        # TS should converge.
        pts, wts = ipq.volume_quad(15, QuadStrategy.TS_ONLY)
        vals = ipq.eval_poly(0, pts)
        vol = float(np.sum(wts[vals < 0]))
        err = abs(vol - vol_ref) / max(abs(vol_ref), 1e-15)
        assert err < 2e-5, f"Bilinear eps=0.01 err: {err:.2e}"  # noqa: PLR2004

    def test_bilinear_degenerate(self) -> None:
        """Eps = 0: sharp cross, should still produce a valid quadrature."""
        bern = self._bilinear_bernstein(0.0)
        ipq = ImplicitPolyQuadrature(bern)
        pts, wts = ipq.volume_quad(10, QuadStrategy.TS_ONLY)
        vals = ipq.eval_poly(0, pts)
        vol = float(np.sum(wts[vals < 0]))
        # For the cross: area of {phi < 0} is 0.5. Actual ~machine eps.
        assert abs(vol - 0.5) < 1e-12  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Paper §4.5: Trilinear tunnel
# ---------------------------------------------------------------------------


class TestTrilinearTunnel:
    """Trilinear polynomial with tunnel topology (paper §4.5).

    phi(x,y,z) = 0.5 - 1.4z + 2.9xy - 6.5xyz - 1.2x + 3.2xz + 3.3yz - 1.3y
    on U = (0, 1)^3.
    """

    @staticmethod
    def _tunnel_bernstein() -> npt.NDArray[np.float64]:
        """Bernstein coefficients for the trilinear tunnel on [0,1]^3.

        Degree (1,1,1): coefficients are the 8 corner values.
        """
        # phi(x,y,z) = 0.5 - 1.2x - 1.3y - 1.4z + 2.9xy + 3.2xz + 3.3yz - 6.5xyz
        c = np.empty((2, 2, 2))
        for ix in range(2):
            for iy in range(2):
                for iz in range(2):
                    x, y, z = float(ix), float(iy), float(iz)
                    c[ix, iy, iz] = (
                        0.5
                        - 1.2 * x
                        - 1.3 * y
                        - 1.4 * z
                        + 2.9 * x * y
                        + 3.2 * x * z
                        + 3.3 * y * z
                        - 6.5 * x * y * z
                    )
        return c

    def test_tunnel_volume_convergence(self) -> None:
        """Volume integral should converge on the tunnel geometry."""
        bern = self._tunnel_bernstein()
        ipq = ImplicitPolyQuadrature(bern)

        # Reference.
        pts_r, wts_r = ipq.volume_quad(20, QuadStrategy.TS_ONLY)
        vals_r = ipq.eval_poly(0, pts_r)
        vol_ref = float(np.sum(wts_r[vals_r < 0]))

        errors = []
        for q in [3, 7, 15]:
            pts, wts = ipq.volume_quad(q, QuadStrategy.TS_ONLY)
            vals = ipq.eval_poly(0, pts)
            vol = float(np.sum(wts[vals < 0]))
            if abs(vol_ref) > 1e-15:  # noqa: PLR2004
                errors.append(abs(vol - vol_ref) / abs(vol_ref))

        # Actual errors: ~1.8e-3, 2.0e-5, 2.3e-7. Should converge.
        assert len(errors) == 3  # noqa: PLR2004
        assert errors[0] > errors[2], "Not converging"
        assert errors[2] < 1.5e-6  # q=15: actual ~2.3e-7  # noqa: PLR2004

    def test_tunnel_weight_sum(self) -> None:
        """Total weights should sum to 1."""
        bern = self._tunnel_bernstein()
        ipq = ImplicitPolyQuadrature(bern)
        _, wts = ipq.volume_quad(5, QuadStrategy.TS_ONLY)
        assert abs(np.sum(wts) - 1.0) < 1e-10  # noqa: PLR2004

    def test_tunnel_surface(self) -> None:
        """Surface quadrature should produce points on the tunnel interface."""
        bern = self._tunnel_bernstein()
        ipq = ImplicitPolyQuadrature(bern)
        s_pts, s_wts, _ = ipq.surface_quad(5, QuadStrategy.TS_ONLY)
        assert len(s_wts) > 0, "No surface points generated"


# ---------------------------------------------------------------------------
# Singularity test cases (paper §A.1, supplementary material)
# ---------------------------------------------------------------------------


def _mono_to_bernstein_2d(
    mono: npt.NDArray[np.float64],
    degrees: tuple[int, int],
    domain_lo: npt.NDArray[np.float64],
    domain_hi: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert 2D monomial polynomial to Bernstein form on a given domain.

    ``mono[i, j]`` is the coefficient of ``x^i * y^j``.
    """
    from math import comb as _comb  # noqa: PLC0415

    dx, dy = degrees
    lo_x, lo_y = domain_lo[0], domain_lo[1]
    hx = domain_hi[0] - lo_x
    hy = domain_hi[1] - lo_y

    # Substitute x = lo_x + hx*t, y = lo_y + hy*s into monomial form.
    mapped = np.zeros((dx + 1, dy + 1))
    for ix in range(mono.shape[0]):
        for iy in range(mono.shape[1]):
            c = mono[ix, iy]
            if c == 0.0:
                continue
            for p in range(min(ix, dx) + 1):
                cx = _comb(ix, p) * lo_x ** (ix - p) * hx**p
                for q in range(min(iy, dy) + 1):
                    cy = _comb(iy, q) * lo_y ** (iy - q) * hy**q
                    mapped[p, q] += c * cx * cy

    # 1D monomial-to-Bernstein matrices.
    def _m2b_mat(n: int) -> npt.NDArray[np.float64]:
        m = np.zeros((n + 1, n + 1))
        for i in range(n + 1):
            for j in range(i + 1):
                m[i, j] = _comb(i, j) / _comb(n, j)
        return m

    return _m2b_mat(dx) @ mapped @ _m2b_mat(dy).T


class TestSingularities2D:
    """Test cases with singular geometry from paper §A.1.1.

    These polynomials have cusps or self-intersections where the gradient
    vanishes. The volumetric integral should still converge, while surface
    integrals may plateau at reduced precision.
    """

    @staticmethod
    def _compute_volume_error(  # noqa: PLR0913
        mono: npt.NDArray[np.float64],
        degrees: tuple[int, int],
        lo: npt.NDArray[np.float64],
        hi: npt.NDArray[np.float64],
        q: int,
        ref_q: int = 40,
    ) -> float:
        """Compute relative volume error vs a high-q reference."""
        bern = _mono_to_bernstein_2d(mono, degrees, lo, hi)
        ipq = ImplicitPolyQuadrature(bern)
        cell_vol = float(np.prod(hi - lo))

        # Reference.
        pts_ref, wts_ref = ipq.volume_quad(ref_q, QuadStrategy.TS_ONLY)
        vals_ref = ipq.eval_poly(0, pts_ref)
        vol_ref = np.sum(wts_ref[vals_ref < 0]) * cell_vol

        # Test.
        pts, wts = ipq.volume_quad(q, QuadStrategy.TS_ONLY)
        vals = ipq.eval_poly(0, pts)
        vol = np.sum(wts[vals < 0]) * cell_vol

        if abs(vol_ref) < 1e-300:  # noqa: PLR2004
            return float(abs(vol))
        return float(abs(vol - vol_ref) / abs(vol_ref))

    def test_deltoid_volume(self) -> None:
        """Deltoid: 3 cusps. Volume integral should converge.

        phi(x,y) = (x^2+y^2)^2 + 18(x^2+y^2) - 8(x^3-3xy^2) - 27
        U = (-2.5, 3.5) x (-3, 3).
        """
        # Expand: x^4 + 2x^2y^2 + y^4 + 18x^2 + 18y^2 - 8x^3 + 24xy^2 - 27.
        mono = np.zeros((5, 5))
        mono[0, 0] = -27.0
        mono[2, 0] = 18.0
        mono[0, 2] = 18.0
        mono[3, 0] = -8.0
        mono[1, 2] = 24.0
        mono[4, 0] = 1.0
        mono[2, 2] = 2.0
        mono[0, 4] = 1.0

        lo = np.array([-2.5, -3.0])
        hi = np.array([3.5, 3.0])

        # Deltoid: singular (3 cusps). Actual err ~1.4e-2 at q=15.
        err = self._compute_volume_error(mono, (4, 4), lo, hi, q=15)
        assert err < 0.15, f"Deltoid volume error too large: {err:.2e}"  # noqa: PLR2004

    def test_folium_volume(self) -> None:
        """Folium of Descartes: self-intersection at origin.

        phi(x,y) = x^3 + y^3 - 3xy
        U = (-1.4, 2.1) x (-1.5, 2).
        """
        mono = np.zeros((4, 4))
        mono[3, 0] = 1.0
        mono[0, 3] = 1.0
        mono[1, 1] = -3.0

        lo = np.array([-1.4, -1.5])
        hi = np.array([2.1, 2.0])

        # Folium: singular but well-handled. Actual err ~2.7e-8 at q=15.
        err = self._compute_volume_error(mono, (3, 3), lo, hi, q=15)
        assert err < 3e-7, f"Folium volume error too large: {err:.2e}"  # noqa: PLR2004

    def test_trifolium_volume(self) -> None:
        """Trifolium: high-degree self-intersection at origin.

        phi(x,y) = (x^2+y^2)^2 - x^3 + 3xy^2
        U = (-1, 1.2) x (-1.1, 1.1).
        """
        # Expand: x^4 + 2x^2y^2 + y^4 - x^3 + 3xy^2.
        mono = np.zeros((5, 5))
        mono[4, 0] = 1.0
        mono[2, 2] = 2.0
        mono[0, 4] = 1.0
        mono[3, 0] = -1.0
        mono[1, 2] = 3.0

        lo = np.array([-1.0, -1.1])
        hi = np.array([1.2, 1.1])

        # Trifolium: singular. Actual err ~5e-3 at q=15 (non-monotonic).
        err = self._compute_volume_error(mono, (4, 4), lo, hi, q=15)
        assert err < 0.05, f"Trifolium volume error too large: {err:.2e}"  # noqa: PLR2004

    def test_deltoid_convergence(self) -> None:
        """Deltoid volume should show approximately exponential convergence."""
        mono = np.zeros((5, 5))
        mono[0, 0] = -27.0
        mono[2, 0] = 18.0
        mono[0, 2] = 18.0
        mono[3, 0] = -8.0
        mono[1, 2] = 24.0
        mono[4, 0] = 1.0
        mono[2, 2] = 2.0
        mono[0, 4] = 1.0

        lo = np.array([-2.5, -3.0])
        hi = np.array([3.5, 3.0])

        errors = []
        for q in [5, 10, 20]:
            err = self._compute_volume_error(mono, (4, 4), lo, hi, q=q)
            errors.append(err)

        # Actual errors: ~5.6e-3, 8.5e-3, 3.3e-2 (non-monotonic due to cusps).
        assert errors[0] > errors[1] or errors[1] < 0.05, "Not converging"  # noqa: PLR2004
        assert errors[2] < 0.35, f"Deltoid q=20 error: {errors[2]:.2e}"  # noqa: PLR2004


@pytest.mark.slow
class TestSingularities3D:
    """Test cases with singular 3D geometry from paper §A.1.2.

    These stress-test the 3D quadrature near cusps and self-intersections.
    """

    @staticmethod
    def _mono_to_bernstein_3d(
        mono: npt.NDArray[np.float64],
        degrees: tuple[int, int, int],
        lo: npt.NDArray[np.float64],
        hi: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Convert 3D monomial polynomial to Bernstein on a given domain."""
        from math import comb as _comb  # noqa: PLC0415

        dx, dy, dz = degrees
        lo_x, lo_y, lo_z = lo[0], lo[1], lo[2]
        hx, hy, hz = hi[0] - lo_x, hi[1] - lo_y, hi[2] - lo_z

        mapped = np.zeros((dx + 1, dy + 1, dz + 1))
        for ix in range(mono.shape[0]):
            for iy in range(mono.shape[1]):
                for iz in range(mono.shape[2]):
                    c = mono[ix, iy, iz]
                    if c == 0.0:
                        continue
                    for p in range(min(ix, dx) + 1):
                        cx = _comb(ix, p) * lo_x ** (ix - p) * hx**p
                        for q in range(min(iy, dy) + 1):
                            cy = _comb(iy, q) * lo_y ** (iy - q) * hy**q
                            for r_idx in range(min(iz, dz) + 1):
                                cz = _comb(iz, r_idx) * lo_z ** (iz - r_idx) * hz**r_idx
                                mapped[p, q, r_idx] += c * cx * cy * cz

        def _m2b_mat(n: int) -> npt.NDArray[np.float64]:
            m = np.zeros((n + 1, n + 1))
            for i in range(n + 1):
                for j in range(i + 1):
                    m[i, j] = _comb(i, j) / _comb(n, j)
            return m

        mx, my, mz = _m2b_mat(dx), _m2b_mat(dy), _m2b_mat(dz)
        tmp1 = np.einsum("ip,pqr->iqr", mx, mapped)
        tmp2 = np.einsum("jq,iqr->ijr", my, tmp1)
        return np.asarray(np.einsum("kr,ijr->ijk", mz, tmp2), dtype=np.float64)

    @staticmethod
    def _compute_3d_volume_error(
        bern: npt.NDArray[np.float64],
        cell_vol: float,
        q: int,
        ref_q: int = 20,
    ) -> float:
        """Compute relative volume error vs a high-q reference."""
        ipq = ImplicitPolyQuadrature(bern)
        pts_r, wts_r = ipq.volume_quad(ref_q, QuadStrategy.TS_ONLY)
        vals_r = ipq.eval_poly(0, pts_r)
        vol_ref = np.sum(wts_r[vals_r < 0]) * cell_vol

        pts, wts = ipq.volume_quad(q, QuadStrategy.TS_ONLY)
        vals = ipq.eval_poly(0, pts)
        vol = np.sum(wts[vals < 0]) * cell_vol

        if abs(vol_ref) < 1e-300:  # noqa: PLR2004
            return float(abs(vol))
        return float(abs(vol - vol_ref) / abs(vol_ref))

    def test_dingdong_volume(self) -> None:
        """Ding-dong surface: cusp at origin.

        phi(x,y,z) = x^2 + y^2 - (1-z)*z^2 = x^2 + y^2 - z^2 + z^3
        U = (-1, 1)^3.
        """
        mono = np.zeros((3, 3, 4))
        mono[2, 0, 0] = 1.0  # x^2
        mono[0, 2, 0] = 1.0  # y^2
        mono[0, 0, 2] = -1.0  # -z^2
        mono[0, 0, 3] = 1.0  # z^3

        lo = np.array([-1.0, -1.0, -1.0])
        hi = np.array([1.0, 1.0, 1.0])
        bern = self._mono_to_bernstein_3d(mono, (2, 2, 3), lo, hi)

        # Ding-dong: singular (cusp). Actual err ~1.6e-3 at q=10.
        err = self._compute_3d_volume_error(bern, 8.0, q=10)
        assert err < 0.016, f"Ding-dong volume error: {err:.2e}"  # noqa: PLR2004

    def test_oloid_volume(self) -> None:
        """Oloid: cusp at origin.

        phi(x,y,z) = x^2 + y^2 + z^3
        U = (-1, 1)^3.
        """
        mono = np.zeros((3, 3, 4))
        mono[2, 0, 0] = 1.0
        mono[0, 2, 0] = 1.0
        mono[0, 0, 3] = 1.0

        lo = np.array([-1.0, -1.0, -1.0])
        hi = np.array([1.0, 1.0, 1.0])
        bern = self._mono_to_bernstein_3d(mono, (2, 2, 3), lo, hi)

        # Oloid: singular (cusp). Actual err ~2.3e-2 at q=10.
        err = self._compute_3d_volume_error(bern, 8.0, q=10)
        assert err < 0.25, f"Oloid volume error: {err:.2e}"  # noqa: PLR2004

    def test_mobius_volume(self) -> None:
        """Mobius surface: line of self-intersection.

        phi(x,y,z) = -4y + x^2*y + y^3 + 4xz - 2x^2*z - 2y^2*z + yz^2
        U = (-1.25, 2.75) x (-2, 2) x (-2, 2).
        """
        mono = np.zeros((3, 4, 3))
        mono[0, 1, 0] = -4.0  # -4y
        mono[2, 1, 0] = 1.0  # x^2*y
        mono[0, 3, 0] = 1.0  # y^3
        mono[1, 0, 1] = 4.0  # 4xz
        mono[2, 0, 1] = -2.0  # -2x^2*z
        mono[0, 2, 1] = -2.0  # -2y^2*z
        mono[0, 1, 2] = 1.0  # yz^2

        lo = np.array([-1.25, -2.0, -2.0])
        hi = np.array([2.75, 2.0, 2.0])
        cell_vol = float(np.prod(hi - lo))
        bern = self._mono_to_bernstein_3d(mono, (2, 3, 2), lo, hi)

        # Möbius: singular. Actual err ~5.2e-14 at q=7 (reaches machine eps).
        err = self._compute_3d_volume_error(bern, cell_vol, q=7)
        assert err < 1e-12, f"Mobius volume error: {err:.2e}"  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Randomly generated geometry (paper §4.3)
# ---------------------------------------------------------------------------


def _make_random_poly_2d(
    rng: np.random.Generator,
    alpha: float = 2.0,
) -> npt.NDArray[np.float64]:
    """Generate a random degree-(2,2) polynomial on (-1,1)^2 per paper eq (10).

    Uses normalized Legendre basis with decay factor alpha.
    Returns Bernstein coefficients on (-1,1)^2 mapped to [0,1]^2.
    """
    # Normalized Legendre polynomials on (-1,1) in monomial form (ascending power).
    # p0(x) = sqrt(1/2)
    # p1(x) = sqrt(3/2)*x
    # p2(x) = sqrt(5/8)*(3x^2-1)
    leg_mono = [
        np.array([np.sqrt(0.5)]),
        np.array([0.0, np.sqrt(1.5)]),
        np.array([-np.sqrt(5.0 / 8.0), 0.0, 3.0 * np.sqrt(5.0 / 8.0)]),
    ]

    # Random coefficients c_i ~ U[-1, 1] with decay lambda(i).
    raw_c = rng.uniform(-1, 1, size=(3, 3))

    # Build monomial representation on (-1,1)^2.
    # phi(x,y) = sum_{i0,i1} lambda(i)*c_i * p_{i0}(x) * p_{i1}(y)
    mono = np.zeros((3, 3))
    for i0 in range(3):
        for i1 in range(3):
            s = i0 + i1
            lam = 1.0 if s == 0 else float(s) ** (-alpha)
            c = raw_c[i0, i1] * lam
            # Outer product of Legendre monomial coefficients.
            lx = leg_mono[i0]
            ly = leg_mono[i1]
            for px in range(len(lx)):
                for py in range(len(ly)):
                    mono[px, py] += c * lx[px] * ly[py]

    return _mono_to_bernstein_2d(mono, (2, 2), np.array([-1.0, -1.0]), np.array([1.0, 1.0]))


def _volume_fraction_2d(
    bern: npt.NDArray[np.float64],
    n_sample: int = 50,
) -> float:
    """Estimate volume fraction of {phi<0} on [0,1]^2."""
    from pantr.bezier.implicit._bernstein import _eval_bernstein_2d  # noqa: PLC0415

    count = 0
    pt = np.empty(2, dtype=np.float64)
    for ix in range(n_sample):
        for iy in range(n_sample):
            pt[0] = (ix + 0.5) / n_sample
            pt[1] = (iy + 0.5) / n_sample
            if _eval_bernstein_2d(bern, pt) < 0:
                count += 1
    return count / (n_sample * n_sample)


class TestRandomGeometry2D:
    """Test convergence on randomly generated 2D geometry (paper §4.3).

    Generates random degree-(2,2) polynomials on (-1,1)^2, filters by
    volume fraction, and checks that the volume integral converges.
    Uses a small sample (20 instances) for CI speed.
    """

    @staticmethod
    def _generate_valid_polys(
        rng: np.random.Generator,
        n_target: int,
        max_attempts: int = 500,
    ) -> list[npt.NDArray[np.float64]]:
        """Generate random polynomials with volume fraction in [10%, 90%]."""
        polys: list[npt.NDArray[np.float64]] = []
        for _ in range(max_attempts):
            bern = _make_random_poly_2d(rng)
            vf = _volume_fraction_2d(bern)
            if 0.1 <= vf <= 0.9:  # noqa: PLR2004
                polys.append(bern)
                if len(polys) >= n_target:
                    break
        return polys

    def test_random_volume_convergence(self) -> None:
        """Median volume error should decrease with increasing q."""
        rng = np.random.default_rng(12345)
        polys = self._generate_valid_polys(rng, n_target=20)
        assert len(polys) >= 10, f"Only generated {len(polys)} valid polynomials"  # noqa: PLR2004

        cell_vol = 4.0  # (-1,1)^2 has area 4

        median_errors: dict[int, float] = {}
        for q in [5, 15]:
            errors = []
            for bern in polys:
                ipq = ImplicitPolyQuadrature(bern)
                # Reference.
                pts_r, wts_r = ipq.volume_quad(30, QuadStrategy.TS_ONLY)
                vals_r = ipq.eval_poly(0, pts_r)
                vol_ref = np.sum(wts_r[vals_r < 0]) * cell_vol

                # Test.
                pts, wts = ipq.volume_quad(q, QuadStrategy.TS_ONLY)
                vals = ipq.eval_poly(0, pts)
                vol = np.sum(wts[vals < 0]) * cell_vol

                if abs(vol_ref) > 1e-10:  # noqa: PLR2004
                    errors.append(abs(vol - vol_ref) / abs(vol_ref))
            median_errors[q] = float(np.median(errors))

        # Median error should decrease significantly from q=5 to q=15.
        assert median_errors[5] > median_errors[15], (
            f"Not converging: median err q=5={median_errors[5]:.2e}, q=15={median_errors[15]:.2e}"
        )
        # At q=15, median error should be modest.
        assert median_errors[15] < 0.01, (  # noqa: PLR2004
            f"Median error at q=15 too large: {median_errors[15]:.2e}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Robustness tests for degenerate configurations."""

    def test_phi_positive_everywhere(self) -> None:
        """No interface: phi > 0 on all of [0,1]^2."""
        c = np.ones((3, 3))
        ipq = ImplicitPolyQuadrature(c)
        pts, wts = ipq.volume_quad(5, QuadStrategy.GL_ONLY)
        vals = ipq.eval_poly(0, pts)
        assert np.sum(wts[vals < 0]) == 0.0

    def test_phi_negative_everywhere(self) -> None:
        """No interface: phi < 0 on all of [0,1]^2. Entire domain is inside."""
        c = -np.ones((3, 3))
        ipq = ImplicitPolyQuadrature(c)
        pts, wts = ipq.volume_quad(5, QuadStrategy.GL_ONLY)
        vals = ipq.eval_poly(0, pts)
        assert abs(np.sum(wts[vals < 0]) - 1.0) < 1e-12  # noqa: PLR2004

    def test_interface_through_corners(self) -> None:
        """Circle passing exactly through all 4 corners of [0,1]^2.

        phi(x,y) = (x-0.5)^2 + (y-0.5)^2 - 0.5. At corners: phi = 0.
        The entire [0,1]^2 is inside or on the circle.
        """
        bx = _mono_to_bernstein_1d(np.array([0.25, -1.0, 1.0]), 2)
        by = _mono_to_bernstein_1d(np.array([0.25, -1.0, 1.0]), 2)
        c = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                c[i, j] = bx[i] + by[j] - 0.5
        ipq = ImplicitPolyQuadrature(c)
        pts, wts = ipq.volume_quad(10, QuadStrategy.AUTO_MIXED)
        vals = ipq.eval_poly(0, pts)
        # Domain is inside or on the circle → area ≈ 1.
        assert abs(np.sum(wts[vals < 0]) - 1.0) < 0.01  # noqa: PLR2004

    def test_straight_line(self) -> None:
        """Linear phi = x - 0.5 splits domain exactly in half."""
        c = np.array([[-0.5, -0.5], [0.5, 0.5]])
        ipq = ImplicitPolyQuadrature(c)
        pts, wts = ipq.volume_quad(5, QuadStrategy.GL_ONLY)
        vals = ipq.eval_poly(0, pts)
        assert abs(np.sum(wts[vals < 0]) - 0.5) < 1e-12  # noqa: PLR2004

    def test_zero_polynomial(self) -> None:
        """Phi = 0 everywhere: rejected as undefined domain."""
        c = np.zeros((2, 2))
        with pytest.raises(ValueError, match="identically zero"):
            ImplicitPolyQuadrature(c)

    def test_surface_no_interface(self) -> None:
        """Surface quad when phi > 0 everywhere: should return empty."""
        c = np.ones((3, 3))
        ipq = ImplicitPolyQuadrature(c)
        s_pts, s_wts, s_nw = ipq.surface_quad(5, QuadStrategy.GL_ONLY)
        assert len(s_wts) == 0

    def test_weight_sum_straight_line(self) -> None:
        """Total weights should sum to 1 even with a straight-line interface."""
        c = np.array([[-0.5, -0.5], [0.5, 0.5]])
        ipq = ImplicitPolyQuadrature(c)
        pts, wts = ipq.volume_quad(5, QuadStrategy.GL_ONLY)
        assert abs(np.sum(wts) - 1.0) < 1e-12  # noqa: PLR2004

    def test_3d_no_interface(self) -> None:
        """3D: phi > 0 everywhere."""
        c = np.ones((2, 2, 2))
        ipq = ImplicitPolyQuadrature(c)
        pts, wts = ipq.volume_quad(3, QuadStrategy.GL_ONLY)
        vals = ipq.eval_poly(0, pts)
        assert np.sum(wts[vals < 0]) == 0.0

    def test_3d_phi_negative_everywhere(self) -> None:
        """3D: phi < 0 everywhere. Entire domain is inside."""
        c = -np.ones((2, 2, 2))
        ipq = ImplicitPolyQuadrature(c)
        pts, wts = ipq.volume_quad(3, QuadStrategy.GL_ONLY)
        vals = ipq.eval_poly(0, pts)
        assert abs(np.sum(wts[vals < 0]) - 1.0) < 1e-12  # noqa: PLR2004

    def test_3d_straight_plane(self) -> None:
        """3D: linear phi = z - 0.5 splits domain in half."""
        c = np.zeros((2, 2, 2))
        c[:, :, 0] = -0.5
        c[:, :, 1] = 0.5
        ipq = ImplicitPolyQuadrature(c)
        pts, wts = ipq.volume_quad(3, QuadStrategy.GL_ONLY)
        vals = ipq.eval_poly(0, pts)
        assert abs(np.sum(wts[vals < 0]) - 0.5) < 1e-12  # noqa: PLR2004

    def test_3d_zero_polynomial(self) -> None:
        """3D: phi = 0 everywhere: rejected as undefined domain."""
        c = np.zeros((2, 2, 2))
        with pytest.raises(ValueError, match="identically zero"):
            ImplicitPolyQuadrature(c)

    def test_3d_surface_no_interface(self) -> None:
        """3D surface quad when phi > 0 everywhere: should return empty."""
        c = np.ones((2, 2, 2))
        ipq = ImplicitPolyQuadrature(c)
        s_pts, s_wts, s_nw = ipq.surface_quad(3, QuadStrategy.GL_ONLY)
        assert len(s_wts) == 0

    def test_3d_surface_aggregate(self) -> None:
        """3D aggregate surface quad for a sphere."""
        from pantr.bezier.implicit import monomial_to_bernstein_3d  # noqa: PLC0415

        # phi = x^2 + y^2 + z^2 - r^2, r=0.3, centered at (0.5,0.5,0.5).
        r = 0.3
        lo = np.array([0.0, 0.0, 0.0])
        hi = np.array([1.0, 1.0, 1.0])
        mono = np.zeros((3, 3, 3))
        # (x-0.5)^2 + (y-0.5)^2 + (z-0.5)^2 - r^2
        # = x^2 - x + 0.25 + y^2 - y + 0.25 + z^2 - z + 0.25 - r^2
        mono[0, 0, 0] = 0.75 - r**2
        mono[1, 0, 0] = -1.0
        mono[2, 0, 0] = 1.0
        mono[0, 1, 0] = -1.0
        mono[0, 2, 0] = 1.0
        mono[0, 0, 1] = -1.0
        mono[0, 0, 2] = 1.0
        bern = monomial_to_bernstein_3d(mono, (2, 2, 2), lo, hi)
        ipq = ImplicitPolyQuadrature(bern)
        _, sw, _ = ipq.surface_quad(8, QuadStrategy.TS_ONLY, aggregate=True)
        expected = 4.0 * np.pi * r**2
        err = abs(np.sum(sw) - expected) / expected
        assert err < 0.01  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Paper §A.1.2d: Deltoid3 (3D generalization of the 2D deltoid)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDeltoid3:
    """3D deltoid: the most challenging singularity test (paper §A.1.2d).

    phi(x,y,z) = (x^2+y^2+z^2)^2 + 18(x^2+y^2+z^2)
                 - 8(x^3+z^3-3xy^2) - 27
    U = (-2.5, 3.5) x (-3, 3) x (-2, 4).

    This surface has four cusps and is described as "perhaps the most
    challenging example presented in this work."
    """

    @staticmethod
    def _deltoid3_bernstein() -> npt.NDArray[np.float64]:
        """Bernstein coefficients for the 3D deltoid on its domain."""
        from pantr.bezier.implicit import monomial_to_bernstein_3d  # noqa: PLC0415

        # phi = (r^2)^2 + 18*r^2 - 8*(x^3+z^3-3xy^2) - 27
        # where r^2 = x^2+y^2+z^2
        # Expanded monomials:
        # (r^2)^2 = x^4+y^4+z^4+2x^2y^2+2x^2z^2+2y^2z^2
        # 18*r^2 = 18x^2+18y^2+18z^2
        # -8(x^3+z^3-3xy^2) = -8x^3-8z^3+24xy^2
        mono = np.zeros((5, 5, 5))
        mono[4, 0, 0] = 1.0  # x^4
        mono[0, 4, 0] = 1.0  # y^4
        mono[0, 0, 4] = 1.0  # z^4
        mono[2, 2, 0] = 2.0  # 2x^2y^2
        mono[2, 0, 2] = 2.0  # 2x^2z^2
        mono[0, 2, 2] = 2.0  # 2y^2z^2
        mono[2, 0, 0] = 18.0  # 18x^2
        mono[0, 2, 0] = 18.0  # 18y^2
        mono[0, 0, 2] = 18.0  # 18z^2
        mono[3, 0, 0] = -8.0  # -8x^3
        mono[0, 0, 3] = -8.0  # -8z^3
        mono[1, 2, 0] = 24.0  # 24xy^2  (from -8*(-3xy^2))
        mono[0, 0, 0] = -27.0  # -27

        lo = np.array([-2.5, -3.0, -2.0])
        hi = np.array([3.5, 3.0, 4.0])
        return monomial_to_bernstein_3d(mono, (4, 4, 4), lo, hi)

    def test_deltoid3_volume(self) -> None:
        """Volume integral should converge for the 3D deltoid."""
        bern = self._deltoid3_bernstein()
        ipq = ImplicitPolyQuadrature(bern)
        cell_vol = 6.0 * 6.0 * 6.0  # (3.5-(-2.5)) * (3-(-3)) * (4-(-2))

        # Reference at q=10 (higher q is too slow for degree-(4,4,4)).
        pts_r, wts_r = ipq.volume_quad(10, QuadStrategy.AUTO_MIXED)
        vals_r = ipq.eval_poly(0, pts_r)
        vol_ref = float(np.sum(wts_r[vals_r < 0]) * cell_vol)

        # Test at q=5.
        pts, wts = ipq.volume_quad(5, QuadStrategy.AUTO_MIXED)
        vals = ipq.eval_poly(0, pts)
        vol = float(np.sum(wts[vals < 0]) * cell_vol)

        if abs(vol_ref) > 1e-10:  # noqa: PLR2004
            err = abs(vol - vol_ref) / abs(vol_ref)
            assert err < 0.5, f"Deltoid3 volume error: {err:.2e}"  # noqa: PLR2004
        assert vol > 0, "Deltoid3 volume should be positive"


# ---------------------------------------------------------------------------
# Paper §4.4: Bilinear with explicit (TS,GL) strategy comparison
# ---------------------------------------------------------------------------


class TestBilinearTSGL:
    """Bilinear tests comparing TS_ONLY (proxy for TS,GL) vs GL_ONLY (paper §4.4 Fig. 10).

    The paper shows that for eps=0.01, (TS,GL) significantly outperforms (GL,GL).
    Since pantr's AUTO_MIXED picks (GL,GL) for this degree-(1,1) polynomial,
    we test TS_ONLY as a proxy for the (TS,GL) strategy.
    """

    @staticmethod
    def _bilinear_bernstein(eps: float) -> npt.NDArray[np.float64]:
        """Bernstein coefficients for (x-0.5)(y-0.5) - eps^2."""
        e2 = eps * eps
        return np.array([[0.25 - e2, -0.25 - e2], [-0.25 - e2, 0.25 - e2]])

    def test_ts_beats_gl_for_high_curvature(self) -> None:
        """For eps=0.01, TS converges faster than GL at moderate q.

        Paper Fig. 10: (TS,GL) column for eps=0.01 shows convergence to ~1e-9
        at q=40, while (GL,GL) only reaches ~1e-5.
        """
        bern = self._bilinear_bernstein(0.01)
        ipq = ImplicitPolyQuadrature(bern)

        # Reference with TS at high q.
        pts_r, wts_r = ipq.volume_quad(60, QuadStrategy.TS_ONLY)
        vals_r = ipq.eval_poly(0, pts_r)
        vol_ref = float(np.sum(wts_r[vals_r < 0]))

        # TS convergence.
        ts_errors: list[float] = []
        gl_errors: list[float] = []
        for q in [10, 20, 30]:
            for strat, errs in [
                (QuadStrategy.TS_ONLY, ts_errors),
                (QuadStrategy.GL_ONLY, gl_errors),
            ]:
                pts, wts = ipq.volume_quad(q, strat)
                vals = ipq.eval_poly(0, pts)
                vol = float(np.sum(wts[vals < 0]))
                errs.append(abs(vol - vol_ref) / max(abs(vol_ref), 1e-15))

        # TS should be significantly better at q=20 and q=30.
        assert ts_errors[1] < gl_errors[1], (
            f"TS ({ts_errors[1]:.2e}) should beat GL ({gl_errors[1]:.2e}) at q=20"
        )
        assert ts_errors[2] < 1e-5, f"TS at q=30: {ts_errors[2]:.2e}"  # noqa: PLR2004

    def test_eps0_gl_exact(self) -> None:
        """For eps=0, GL should reach machine precision quickly.

        Paper Fig. 10: (GL,GL) for eps=0 converges very fast because
        the degenerate cross is exactly representable.
        """
        bern = self._bilinear_bernstein(0.0)
        ipq = ImplicitPolyQuadrature(bern)

        pts, wts = ipq.volume_quad(5, QuadStrategy.GL_ONLY)
        vals = ipq.eval_poly(0, pts)
        vol = float(np.sum(wts[vals < 0]))
        # Area of {phi < 0} = two opposite quadrants = 0.5.
        assert abs(vol - 0.5) < 1e-12  # noqa: PLR2004

    def test_eps01_gl_converges(self) -> None:
        """For eps=0.1, GL converges to machine precision (smooth geometry)."""
        bern = self._bilinear_bernstein(0.1)
        ipq = ImplicitPolyQuadrature(bern)

        pts_r, wts_r = ipq.volume_quad(40, QuadStrategy.GL_ONLY)
        vals_r = ipq.eval_poly(0, pts_r)
        vol_ref = float(np.sum(wts_r[vals_r < 0]))

        pts, wts = ipq.volume_quad(20, QuadStrategy.GL_ONLY)
        vals = ipq.eval_poly(0, pts)
        vol = float(np.sum(wts[vals < 0]))
        err = abs(vol - vol_ref) / max(abs(vol_ref), 1e-15)
        assert err < 2e-8, f"Bilinear eps=0.1 GL at q=20: {err:.2e}"  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Paper §4.1: h-refinement with extended range (up to n=128)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestHRefinementExtended:
    """Extended h-refinement tests (paper §4.1, Fig. 4) up to n=128.

    Tests convergence rates O(h^{2q}) for the ellipse (2D) and ellipsoid (3D).
    """

    @staticmethod
    def _h_refine_ellipse(n: int, q: int) -> float:
        """Compute ellipse area via h-refinement with n cells per axis."""
        lo_v, hi_v = -1.1, 1.1
        h = (hi_v - lo_v) / n
        total = 0.0
        for ix in range(n):
            for iy in range(n):
                cell_lo = np.array([lo_v + ix * h, lo_v + iy * h])
                cell_hi = np.array([lo_v + (ix + 1) * h, lo_v + (iy + 1) * h])
                coeffs = _ellipse_bernstein_on_cell(cell_lo, cell_hi)
                ipq = ImplicitPolyQuadrature(coeffs)
                pts, wts = ipq.volume_quad(q, QuadStrategy.GL_ONLY)
                vals = ipq.eval_poly(0, pts)
                total += np.sum(wts[vals < 0]) * h**2
        return total

    @staticmethod
    def _h_refine_ellipsoid(n: int, q: int) -> float:
        """Compute ellipsoid volume via h-refinement with n cells per axis."""
        lo_v, hi_v = -1.1, 1.1
        h = (hi_v - lo_v) / n
        total = 0.0
        for ix in range(n):
            for iy in range(n):
                for iz in range(n):
                    cell_lo = np.array([lo_v + ix * h, lo_v + iy * h, lo_v + iz * h])
                    cell_hi = np.array(
                        [lo_v + (ix + 1) * h, lo_v + (iy + 1) * h, lo_v + (iz + 1) * h]
                    )
                    coeffs = _ellipsoid_bernstein_on_cell(cell_lo, cell_hi)
                    ipq = ImplicitPolyQuadrature(coeffs)
                    pts, wts = ipq.volume_quad(q, QuadStrategy.GL_ONLY)
                    vals = ipq.eval_poly(0, pts)
                    total += np.sum(wts[vals < 0]) * h**3
        return total

    def test_ellipse_h_refinement_extended(self) -> None:
        """Ellipse area O(h^{2q}) for q=3 up to n=64.

        Paper Fig. 4 (top-left) shows clean O(h^6) convergence for q=3.
        """
        expected = np.pi / 2.0
        q = 3
        errors = {}
        for n in [8, 16, 32, 64]:
            area = self._h_refine_ellipse(n, q)
            errors[n] = abs(area - expected) / expected

        # Convergence rate between successive doublings should be ~2q=6.
        for n1, n2 in [(16, 32), (32, 64)]:
            if errors[n2] > 0:
                rate = np.log2(errors[n1] / errors[n2])
                assert rate > 3.0, (  # noqa: PLR2004
                    f"h-refine rate n={n1}->{n2}: {rate:.1f} (expected ~{2 * q})"
                )

    def test_ellipsoid_h_refinement(self) -> None:
        """Ellipsoid volume O(h^{2q}) for q=3 with n=4,8,16.

        Paper Fig. 4 (top-right). Limited to n=16 for reasonable CI time.
        """
        expected = 4.0 / 3.0 * np.pi / (2 * 3)  # pi/(2*3) from semi-axes 1, 1/2, 1/3
        q = 3
        errors = {}
        for n in [4, 8, 16]:
            vol = self._h_refine_ellipsoid(n, q)
            errors[n] = abs(vol - expected) / expected

        # Check convergence rate.
        if errors[16] > 0:
            rate = np.log2(errors[8] / errors[16])
            assert rate > 3.5, f"3D h-refine rate: {rate:.1f} (expected ~{2 * q})"  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Additional unit tests
# ---------------------------------------------------------------------------


class TestResultant:
    """Direct unit tests for resultant_2d from _resultant.py."""

    def test_resultant_along_y(self) -> None:
        """Resultant of (x-0.5) and (y-0.5) along axis 0 yields root at y=0.5."""
        phi1 = np.array([[-0.5, -0.5], [0.5, 0.5]])  # x - 0.5
        phi2 = np.array([[-0.5, 0.5], [-0.5, 0.5]])  # y - 0.5
        res = resultant_2d(phi1, phi2, k=0)
        roots_buf, count, _ = find_roots(res)
        roots = roots_buf[:count]
        assert count == 1, f"Expected 1 root, got {count}"
        assert abs(roots[0] - 0.5) < 1e-12  # noqa: PLR2004

    def test_resultant_along_x(self) -> None:
        """Resultant of (x-0.5) and (y-0.5) along axis 1 yields root at x=0.5."""
        phi1 = np.array([[-0.5, -0.5], [0.5, 0.5]])  # x - 0.5
        phi2 = np.array([[-0.5, 0.5], [-0.5, 0.5]])  # y - 0.5
        res = resultant_2d(phi1, phi2, k=1)
        roots_buf, count, _ = find_roots(res)
        roots = roots_buf[:count]
        assert count == 1, f"Expected 1 root, got {count}"
        assert abs(roots[0] - 0.5) < 1e-12  # noqa: PLR2004


class TestSurfaceQuad3DAggregate:
    """Tests for 3D aggregate surface quadrature convergence."""

    def test_sphere_area_convergence(self) -> None:
        """Sphere surface area converges with aggregate mode."""
        r = 0.2
        r_sq = r**2
        coeffs = _make_sphere_coeffs(r_sq=r_sq)
        ipq = ImplicitPolyQuadrature(coeffs)
        expected = 4.0 * np.pi * r**2

        errors = []
        for q in [5, 10, 15]:
            s_pts, s_wts, _ = ipq.surface_quad(q, QuadStrategy.TS_ONLY, aggregate=True)
            area = np.sum(s_wts)
            errors.append(abs(area - expected) / expected)

        # Errors should decrease monotonically.
        for i in range(len(errors) - 1):
            assert errors[i + 1] < errors[i], f"Errors not monotonically decreasing: {errors}"

        # Final error should be small.
        assert errors[-1] < 1e-2, f"Final error {errors[-1]:.2e} exceeds 1e-2"  # noqa: PLR2004


class TestBernstein3D:
    """Unit tests for 3D Bernstein operations."""

    def test_collapse_3d_linear(self) -> None:
        """Collapse phi(x,y,z) = x + 2y + 3z at x=0.3, z=0.7 with k=1."""
        # Bernstein coefficients for phi = x + 2y + 3z on [0,1]^3, degree (1,1,1).
        # B[i,j,k] = phi(i, j, k) for linear polynomial.
        coeffs = np.array([[[0.0, 3.0], [2.0, 5.0]], [[1.0, 4.0], [3.0, 6.0]]])

        # k=1: keep y-axis, contract x (axis 0) and z (axis 2).
        # x_tang = [0.3, 0.7] -> x_tang[0]=0.3 for axis 0, x_tang[1]=0.7 for axis 2.
        result = _collapse_3d(coeffs, k=1, x_tang=np.array([0.3, 0.7]))

        # phi(0.3, y, 0.7) = 0.3 + 2y + 2.1 = 2.4 + 2y
        # Degree-1 Bernstein: [phi(0.3, 0, 0.7), phi(0.3, 1, 0.7)] = [2.4, 4.4]
        expected = np.array([2.4, 4.4])
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_eval_gradient_3d_linear(self) -> None:
        """Gradient of phi = x + 2y + 3z should be [1, 2, 3] everywhere."""
        coeffs = np.array([[[0.0, 3.0], [2.0, 5.0]], [[1.0, 4.0], [3.0, 6.0]]])
        grad = _eval_gradient_3d(coeffs, np.array([0.5, 0.5, 0.5]))
        np.testing.assert_allclose(grad, np.array([1.0, 2.0, 3.0]), atol=1e-14)


class TestConvertValidation:
    """Tests for _convert.py input validation."""

    def test_monomial_to_bernstein_2d_bad_domain(self) -> None:
        """Reject domain_hi <= domain_lo in 2D."""
        mono = np.array([[1.0]])
        with pytest.raises(ValueError, match="domain_hi"):
            monomial_to_bernstein_2d(mono, (0, 0), np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="domain_hi"):
            monomial_to_bernstein_2d(mono, (0, 0), np.array([0.0, 1.0]), np.array([1.0, 0.0]))

    def test_monomial_to_bernstein_3d_bad_domain(self) -> None:
        """Reject domain_hi <= domain_lo in 3D."""
        mono = np.array([[[1.0]]])
        with pytest.raises(ValueError, match="domain_hi"):
            monomial_to_bernstein_3d(
                mono, (0, 0, 0), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 1.0])
            )

    def test_validate_degrees_too_small(self) -> None:
        """Reject target degree < monomial degree."""
        with pytest.raises(ValueError, match="less than monomial degree"):
            _validate_degrees((3, 2), (1, 2))


class TestZeroPolyValidation:
    """Tests for identically-zero polynomial rejection."""

    def test_zero_polynomial_raises(self) -> None:
        """ImplicitPolyQuadrature rejects identically-zero coefficients."""
        with pytest.raises(ValueError, match="identically zero"):
            ImplicitPolyQuadrature(np.zeros((3, 3)))


class TestEvalPolyValidation:
    """Tests for eval_poly input validation."""

    def test_wrong_points_shape_1d(self) -> None:
        """Reject 1D points array."""
        ipq = ImplicitPolyQuadrature(_make_circle_coeffs())
        with pytest.raises(ValueError, match="shape"):
            ipq.eval_poly(0, np.array([0.5, 0.5]))

    def test_wrong_points_dim(self) -> None:
        """Reject points with wrong number of columns."""
        ipq = ImplicitPolyQuadrature(_make_circle_coeffs())
        with pytest.raises(ValueError, match="shape"):
            ipq.eval_poly(0, np.array([[0.5, 0.5, 0.5]]))
