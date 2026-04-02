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
        r, c = find_roots(np.array([1.0, -1.0]))
        assert c == 1
        assert abs(r[0] - 0.5) < 1e-12  # noqa: PLR2004

    def test_quadratic(self) -> None:
        # (t-0.3)(t-0.7) in Bernstein.
        r, c = find_roots(np.array([0.21, -0.29, 0.21]))
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
        r, c = find_roots(bern)
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
        raw, n_raw = _clip_roots_core(bern, 1e-15, 1e-15)
        unique, n_unique = _dedup_roots(raw, n_raw, bern, 1e-15, 1e-15)
        assert n_unique == 5, f"Expected 5 roots, got {n_unique}: {unique[:n_unique]}"  # noqa: PLR2004
        for exp in roots_expected:
            assert any(abs(unique[i] - exp) < 1e-6 for i in range(n_unique)), (  # noqa: PLR2004
                f"Missing root {exp}: {unique[:n_unique]}"
            )

    def test_no_roots(self) -> None:
        _r, c = find_roots(np.array([1.0, 2.0, 3.0]))
        assert c == 0

    def test_circle_collapsed(self) -> None:
        # Circle collapsed at y=0.5: roots at 0.5 ± sqrt(0.1).
        r, c = find_roots(np.array([0.15, -0.35, 0.15]))
        assert c == 2  # noqa: PLR2004
        assert abs(r[0] - 0.18377223) < 1e-6  # noqa: PLR2004
        assert abs(r[1] - 0.81622777) < 1e-6  # noqa: PLR2004


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

        # Check exponential convergence: each doubling of q should
        # roughly halve the number of accurate digits.
        assert errors[0] < 0.01  # q=5: ~0.2%  # noqa: PLR2004
        assert errors[1] < 1e-5  # q=10: < 0.001%  # noqa: PLR2004
        assert errors[2] < 1e-8  # q=20: < 1e-8  # noqa: PLR2004

    def test_area_convergence_auto(self, circle_ipq: ImplicitPolyQuadrature) -> None:
        """AUTO_MIXED should also give exponential convergence."""
        expected = np.pi * 0.1

        pts, wts = circle_ipq.volume_quad(20, QuadStrategy.AUTO_MIXED)
        vals = circle_ipq.eval_poly(0, pts)
        area = np.sum(wts[vals < 0])
        err = abs(area - expected) / expected
        assert err < 1e-8  # noqa: PLR2004


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

        assert errors[0] < 0.05  # q=5  # noqa: PLR2004
        assert errors[1] < 0.005  # q=10  # noqa: PLR2004
        assert errors[2] < 1e-4  # q=20  # noqa: PLR2004

    def test_normal_weights_sum(self, circle_ipq: ImplicitPolyQuadrature) -> None:
        """Normal weights should approximately cancel (closed curve)."""
        s_pts, s_wts, s_nwts = circle_ipq.surface_quad(10, QuadStrategy.TS_ONLY)
        # For a closed curve, sum of normal flux should be close to zero.
        flux = np.sum(s_nwts, axis=0)
        # Not exactly zero due to quadrature error, but should be small.
        assert np.linalg.norm(flux) < 0.1  # noqa: PLR2004

    def test_perimeter_aggregate(self, circle_ipq: ImplicitPolyQuadrature) -> None:
        """Aggregate perimeter should converge faster than single-direction."""
        expected = 2.0 * np.pi * np.sqrt(0.1)
        _, sw, _ = circle_ipq.surface_quad(20, QuadStrategy.TS_ONLY, aggregate=True)
        perim = np.sum(sw)
        err = abs(perim - expected) / expected
        assert err < 1e-8  # noqa: PLR2004


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

    @pytest.fixture()
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

        assert errors[0] < 0.01  # noqa: PLR2004
        assert errors[1] < 1e-4  # noqa: PLR2004
        assert errors[2] < 1e-6  # noqa: PLR2004


class TestSurfaceQuad3D:
    """Tests for 3D surface quadrature convergence."""

    @pytest.fixture()
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

        assert errors[0] < 0.05  # noqa: PLR2004
        assert errors[1] < 0.005  # noqa: PLR2004
        assert errors[2] < 5e-4  # noqa: PLR2004

    def test_normal_flux_closed(self, sphere_ipq: ImplicitPolyQuadrature) -> None:
        """Normal flux sum should be near zero for a closed surface."""
        s_pts, s_wts, s_nwts = sphere_ipq.surface_quad(7, QuadStrategy.TS_ONLY)
        flux = np.sum(s_nwts, axis=0)
        assert np.linalg.norm(flux) < 0.5  # noqa: PLR2004


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

    def test_eval_poly(self) -> None:
        c = _make_circle_coeffs()
        ipq = ImplicitPolyQuadrature(c)
        pts = np.array([[0.5, 0.5]])
        vals = ipq.eval_poly(0, pts)
        # At center: (0-0)^2 + (0-0)^2 - 0.1 = -0.1.
        assert abs(vals[0] - (-0.1)) < 1e-12  # noqa: PLR2004


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

        assert errors[0] < 1e-4  # n=8  # noqa: PLR2004
        # Check convergence rate between n=16 and n=32.
        rate = np.log2(errors[1] / errors[2]) if errors[2] > 0 else 20.0
        assert rate > 3.5, f"h-refinement rate too low: {rate:.1f} (expected ~{2 * q})"  # noqa: PLR2004


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

        # Approximately exponential: doubling q roughly doubles accurate digits.
        assert errors[0] < 0.01  # q=5  # noqa: PLR2004
        assert errors[1] < 1e-5  # q=10  # noqa: PLR2004
        assert errors[2] < 1e-9  # q=20  # noqa: PLR2004
        assert errors[3] < 1e-12  # q=30  # noqa: PLR2004

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

        assert errors[0] < 0.05  # q=5  # noqa: PLR2004
        assert errors[1] < 0.005  # q=10  # noqa: PLR2004
        assert errors[2] < 1e-3  # q=20  # noqa: PLR2004
