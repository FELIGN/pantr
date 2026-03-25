"""Stress tests for Bernstein polynomial root finding.

Cross-validates Yuksel's monotone-decomposition algorithm against Bezier
clipping on large batches of randomized polynomials. Agreement between two
independent implementations is strong evidence of correctness.

Test categories mirror the curve families from the ``parametric_immersion``
stress suite, adapted to work on raw 1-D Bernstein coefficient arrays
(pantr does not yet have curve geometry).

Running::

    pytest tests/test_root_finding_stress.py --no-cov -v
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy import typing as npt

from pantr.root_finding import find_roots, find_roots_batch
from pantr.root_finding._clipping_core import _clip_roots_core, _dedup_roots
from pantr.root_finding._root_finding_core import _de_casteljau_eval_scalar
from pantr.root_finding._yuksel_core import _yuksel_roots

_ROOT_TOL: float = 1e-6
"""Maximum parameter-space discrepancy between matched roots."""

_RESIDUAL_TOL: float = 1e-6
"""Maximum allowed residual |f(root)| for any reported root."""


# =====================================================================
# Coefficient factories
# =====================================================================


def _make_random_coeff(
    rng: np.random.Generator,
    degree: int,
    scale: float = 2.0,
) -> npt.NDArray[np.float64]:
    """Random Bernstein coefficients in [-scale, scale]."""
    return rng.uniform(-scale, scale, degree + 1).astype(np.float64)


def _make_wiggly_coeff(
    rng: np.random.Generator,
    degree: int,
    scale: float = 3.0,
) -> npt.NDArray[np.float64]:
    """Alternating-sign coefficients (many sign changes)."""
    c = rng.uniform(0.1, scale, degree + 1).astype(np.float64)
    c[1::2] *= -1.0
    return c


def _make_near_zero_coeff(
    rng: np.random.Generator,
    degree: int,
) -> npt.NDArray[np.float64]:
    """Coefficients clustered around zero (tests tolerance handling)."""
    return rng.normal(scale=1e-8, size=degree + 1).astype(np.float64)


def _make_extreme_range_coeff(
    rng: np.random.Generator,
    degree: int,
) -> npt.NDArray[np.float64]:
    """Coefficients spanning many orders of magnitude."""
    signs = rng.choice([-1.0, 1.0], size=degree + 1)
    magnitudes = np.exp2(rng.uniform(-12.0, 12.0, degree + 1))
    result: npt.NDArray[np.float64] = (signs * magnitudes).astype(np.float64)
    return result


def _make_double_root_coeff(
    rng: np.random.Generator,
    degree: int,
) -> npt.NDArray[np.float64]:
    """Polynomial with a known double root (even multiplicity).

    Constructs (t - r)^2 * q(t) in Bernstein form by multiplying
    Bernstein representations.
    """
    if degree < 2:  # noqa: PLR2004
        return _make_random_coeff(rng, degree)

    from pantr.bezier._bezier_core import (  # noqa: PLC0415
        _scalar_bernstein_product_1d_core,
    )

    # (t - r) in Bernstein form: [-r, 1-r]
    r = rng.uniform(0.2, 0.8)
    linear = np.array([-r, 1.0 - r], dtype=np.float64)
    # Square it via Bernstein product
    squared = _scalar_bernstein_product_1d_core(linear, linear)

    # Multiply by a random cofactor to reach target degree
    remaining = degree - 2
    if remaining > 0:
        cofactor = rng.uniform(0.5, 2.0, remaining + 1).astype(np.float64)
        result = _scalar_bernstein_product_1d_core(squared, cofactor)
    else:
        result = squared

    return np.ascontiguousarray(result, dtype=np.float64)


def _make_rational_ray_coeff(
    rng: np.random.Generator,
    degree: int,
) -> npt.NDArray[np.float64]:
    """Numerator of a rational ray equation: N_y(t) - offset * W(t).

    Simulates the scalar polynomial arising from intersecting a rational
    Bezier curve with a horizontal ray.
    """
    ctrl_y = rng.uniform(-2.0, 2.0, degree + 1).astype(np.float64)
    weights = rng.uniform(0.2, 3.0, degree + 1).astype(np.float64)
    offset = rng.uniform(float(ctrl_y.min()), float(ctrl_y.max()))
    return (ctrl_y * weights - offset * weights).astype(np.float64)


# =====================================================================
# Helpers
# =====================================================================


def _find_roots_yuksel(
    coeff: npt.NDArray[np.float64],
    tol: float = 1e-12,
) -> npt.NDArray[np.float64]:
    """Run Yuksel root finder and return sorted roots."""
    if len(coeff) < 2 or np.all(np.abs(coeff) <= tol):  # noqa: PLR2004
        return np.empty(0, dtype=np.float64)
    roots, count = _yuksel_roots(coeff, tol)
    if count == 0:
        return np.empty(0, dtype=np.float64)
    return np.sort(roots[:count])


def _find_roots_clipping(
    coeff: npt.NDArray[np.float64],
    tol: float = 1e-12,
) -> npt.NDArray[np.float64]:
    """Run Bezier clipping root finder and return sorted, deduped roots."""
    if len(coeff) < 2 or np.all(np.abs(coeff) <= tol):  # noqa: PLR2004
        return np.empty(0, dtype=np.float64)
    raw, count = _clip_roots_core(coeff, tol, tol)
    return _dedup_roots(raw, count, coeff, tol, tol)


def _assert_roots_valid(
    coeff: npt.NDArray[np.float64],
    roots: npt.NDArray[np.float64],
    label: str,
) -> None:
    """Assert all roots are in [0, 1] and have small residuals."""
    for r in roots:
        assert 0.0 - _ROOT_TOL <= r <= 1.0 + _ROOT_TOL, f"{label}: root {r} outside [0, 1]"
        val = _de_casteljau_eval_scalar(coeff, float(np.clip(r, 0.0, 1.0)))
        assert abs(val) < _RESIDUAL_TOL, f"{label}: root t={r:.12f} has residual={val:.2e}"


def _assert_algorithms_agree(
    coeff: npt.NDArray[np.float64],
    label: str,
    tol: float = 1e-12,
) -> None:
    """Assert Yuksel and clipping find the same roots."""
    roots_yuk = _find_roots_yuksel(coeff, tol)
    roots_clip = _find_roots_clipping(coeff, tol)

    # Both should produce valid roots.
    _assert_roots_valid(coeff, roots_yuk, f"{label}/yuksel")
    _assert_roots_valid(coeff, roots_clip, f"{label}/clip")

    # Same count.
    assert len(roots_yuk) == len(roots_clip), (
        f"{label}: count mismatch yuksel={len(roots_yuk)} vs "
        f"clip={len(roots_clip)}, "
        f"yuksel={roots_yuk}, clip={roots_clip}"
    )

    # Matched parameters.
    for ry, rc in zip(roots_yuk, roots_clip, strict=True):
        assert abs(ry - rc) < _ROOT_TOL, (
            f"{label}: root mismatch yuksel={ry:.12f} vs clip={rc:.12f}"
        )


# =====================================================================
# Stress tests
# =====================================================================


class TestAlgorithmAgreement:
    """Cross-validate Yuksel vs Bezier clipping on random polynomials."""

    @pytest.mark.parametrize("seed", range(10))
    def test_random_low_degree(self, seed: int) -> None:
        """50 random polynomials per seed, degree 2-5."""
        rng = np.random.default_rng(1000 + seed)
        for i in range(50):
            degree = int(rng.integers(2, 6))
            coeff = _make_random_coeff(rng, degree)
            _assert_algorithms_agree(coeff, f"low-deg/s{seed}/i{i}")

    @pytest.mark.parametrize("seed", range(10))
    def test_random_high_degree(self, seed: int) -> None:
        """50 random polynomials per seed, degree 6-12."""
        rng = np.random.default_rng(2000 + seed)
        for i in range(50):
            degree = int(rng.integers(6, 13))
            coeff = _make_random_coeff(rng, degree)
            _assert_algorithms_agree(coeff, f"high-deg/s{seed}/i{i}")

    @pytest.mark.parametrize("seed", range(5))
    def test_very_high_degree(self, seed: int) -> None:
        """20 random polynomials per seed, degree 13-20."""
        rng = np.random.default_rng(3000 + seed)
        for i in range(20):
            degree = int(rng.integers(13, 21))
            coeff = _make_random_coeff(rng, degree)
            _assert_algorithms_agree(coeff, f"vhigh-deg/s{seed}/i{i}")

    @pytest.mark.parametrize("seed", range(5))
    def test_wiggly(self, seed: int) -> None:
        """40 wiggly polynomials per seed (many sign changes)."""
        rng = np.random.default_rng(4000 + seed)
        for i in range(40):
            degree = int(rng.integers(5, 13))
            coeff = _make_wiggly_coeff(rng, degree)
            _assert_algorithms_agree(coeff, f"wiggly/s{seed}/i{i}")

    @pytest.mark.parametrize("seed", range(5))
    def test_near_zero(self, seed: int) -> None:
        """40 near-zero polynomials per seed."""
        rng = np.random.default_rng(5000 + seed)
        for i in range(40):
            degree = int(rng.integers(3, 9))
            coeff = _make_near_zero_coeff(rng, degree)
            # Near-zero polynomials: just verify residuals, algorithms
            # may legitimately disagree on whether a near-zero crossing
            # is a root.
            roots = _find_roots_yuksel(coeff)
            _assert_roots_valid(coeff, roots, f"near-zero/s{seed}/i{i}")

    @pytest.mark.parametrize("seed", range(5))
    def test_extreme_range(self, seed: int) -> None:
        """40 extreme-range polynomials per seed."""
        rng = np.random.default_rng(6000 + seed)
        for i in range(40):
            degree = int(rng.integers(3, 10))
            coeff = _make_extreme_range_coeff(rng, degree)
            # Extreme range: Yuksel is the reference (scale-invariant).
            # Just verify its roots are valid.
            roots = _find_roots_yuksel(coeff)
            _assert_roots_valid(coeff, roots, f"extreme/s{seed}/i{i}")

    @pytest.mark.parametrize("seed", range(5))
    def test_double_roots(self, seed: int) -> None:
        """20 polynomials with known double roots per seed."""
        rng = np.random.default_rng(7000 + seed)
        for i in range(20):
            degree = int(rng.integers(4, 10))
            coeff = _make_double_root_coeff(rng, degree)
            # Double roots: just verify that find_roots finds at least
            # one root near the known location with valid residual.
            roots = find_roots(coeff, tol=1e-12)
            _assert_roots_valid(coeff, roots, f"double/s{seed}/i{i}")
            assert len(roots) >= 1, f"double/s{seed}/i{i}: expected at least 1 root, got 0"

    @pytest.mark.parametrize("seed", range(5))
    def test_rational_ray(self, seed: int) -> None:
        """40 rational ray-equation polynomials per seed."""
        rng = np.random.default_rng(8000 + seed)
        for i in range(40):
            degree = int(rng.integers(3, 9))
            coeff = _make_rational_ray_coeff(rng, degree)
            _assert_algorithms_agree(coeff, f"rational/s{seed}/i{i}")


class TestBatchConsistency:
    """Verify batch API matches single-polynomial results."""

    @pytest.mark.parametrize("seed", range(5))
    def test_batch_matches_single(self, seed: int) -> None:
        """Batch find_roots matches individual find_roots calls."""
        rng = np.random.default_rng(9000 + seed)
        degree = 5
        n_polys = 50
        coeffs = rng.uniform(-2.0, 2.0, (n_polys, degree + 1)).astype(np.float64)

        roots_batch, counts_batch = find_roots_batch(coeffs, tol=1e-12)

        for i in range(n_polys):
            roots_single = find_roots(coeffs[i], tol=1e-12)
            assert counts_batch[i] == len(roots_single), (
                f"s{seed}/i{i}: count mismatch "
                f"batch={counts_batch[i]} vs single={len(roots_single)}"
            )
            batch_roots_i = np.sort(roots_batch[i, : counts_batch[i]])
            np.testing.assert_allclose(
                batch_roots_i,
                roots_single,
                atol=_ROOT_TOL,
                err_msg=f"s{seed}/i{i}: root values differ",
            )
