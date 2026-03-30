"""Tests for tanh-sinh quadrature in pantr.quad."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pytest

from pantr.quad import get_tanh_sinh_1d
from pantr.tolerance import get_conservative


class TestTanhSinhValidation:
    """Input validation tests for get_tanh_sinh_1d."""

    def test_invalid_n_pts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            get_tanh_sinh_1d(0)

    def test_negative_n_pts_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            get_tanh_sinh_1d(-1)

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="float32 or float64"):
            get_tanh_sinh_1d(5, np.int32)


class TestTanhSinhBasicProperties:
    """Basic structural tests for tanh-sinh quadrature."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_n1_midpoint_rule(self, dtype: npt.DTypeLike) -> None:
        """n=1 returns the midpoint rule: node=0.5, weight=1."""
        nodes, weights = get_tanh_sinh_1d(1, dtype)
        assert nodes.shape == (1,)
        assert weights.shape == (1,)
        nptest.assert_allclose(nodes, [0.5], atol=1e-15)
        nptest.assert_allclose(weights, [1.0], atol=1e-15)
        assert nodes.dtype == np.dtype(dtype)
        assert weights.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("n_pts", [2, 3, 5, 10, 20])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_weights_sum_to_one(self, n_pts: int, dtype: npt.DTypeLike) -> None:
        """Weights on [0,1] sum to 1."""
        _, weights = get_tanh_sinh_1d(n_pts, dtype)
        nptest.assert_allclose(np.sum(weights, dtype=np.float64), 1.0, rtol=get_conservative(dtype))

    @pytest.mark.parametrize("n_pts", [2, 5, 10, 20])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_nodes_in_unit_interval(self, n_pts: int, dtype: npt.DTypeLike) -> None:
        """All nodes lie in [0, 1]."""
        nodes, _ = get_tanh_sinh_1d(n_pts, dtype)
        assert np.all(nodes >= 0.0)
        assert np.all(nodes <= 1.0)

    @pytest.mark.parametrize("n_pts", [2, 5, 10, 20])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_positive_weights(self, n_pts: int, dtype: npt.DTypeLike) -> None:
        """All weights are strictly positive."""
        _, weights = get_tanh_sinh_1d(n_pts, dtype)
        assert np.all(weights > 0.0)

    @pytest.mark.parametrize("n_pts", [2, 5, 10, 20])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_output_dtype(self, n_pts: int, dtype: npt.DTypeLike) -> None:
        """Output arrays have the requested dtype."""
        nodes, weights = get_tanh_sinh_1d(n_pts, dtype)
        assert nodes.dtype == np.dtype(dtype)
        assert weights.dtype == np.dtype(dtype)

    @pytest.mark.parametrize("n_pts", [2, 3, 5, 10])
    def test_effective_nodes_le_n(self, n_pts: int) -> None:
        """Effective number of nodes is at most n_pts."""
        nodes, _ = get_tanh_sinh_1d(n_pts)
        assert len(nodes) <= n_pts


class TestTanhSinhEndpointSnapping:
    """Tests for endpoint snapping behavior at large n."""

    def test_snapping_reduces_node_count(self) -> None:
        """For large n, endpoint snapping reduces effective node count."""
        n_requested = 100
        nodes, _ = get_tanh_sinh_1d(n_requested)
        assert len(nodes) < n_requested

    def test_snapped_nodes_include_endpoints(self) -> None:
        """After snapping, 0 and 1 appear as nodes."""
        nodes, _ = get_tanh_sinh_1d(100)
        assert np.isclose(nodes.min(), 0.0, atol=1e-15)
        assert np.isclose(nodes.max(), 1.0, atol=1e-15)

    def test_snapped_weights_still_sum_to_one(self) -> None:
        """Weights sum to 1 even after endpoint snapping."""
        _, weights = get_tanh_sinh_1d(100)
        nptest.assert_allclose(np.sum(weights), 1.0, rtol=1e-14)


class TestTanhSinhSymmetry:
    """Tests for symmetry of the tanh-sinh scheme."""

    @pytest.mark.parametrize("n_pts", [2, 5, 10, 20])
    def test_node_symmetry_about_half(self, n_pts: int) -> None:
        """Nodes are symmetric about 0.5: for each node x, 1-x also exists."""
        nodes, _ = get_tanh_sinh_1d(n_pts)
        sorted_nodes = np.sort(nodes)
        reversed_nodes = 1.0 - np.sort(nodes)[::-1]
        nptest.assert_allclose(sorted_nodes, reversed_nodes, atol=1e-14)

    @pytest.mark.parametrize("n_pts", [2, 5, 10, 20])
    def test_weight_symmetry(self, n_pts: int) -> None:
        """Symmetric node pairs have equal weights."""
        nodes, weights = get_tanh_sinh_1d(n_pts)
        order = np.argsort(nodes)
        sorted_w = weights[order]
        nptest.assert_allclose(sorted_w, sorted_w[::-1], atol=1e-14)


class TestTanhSinhIntegration:
    """Integration accuracy tests for tanh-sinh quadrature."""

    @staticmethod
    def _integrate(n_pts: int, f: Any, dtype: npt.DTypeLike = np.float64) -> np.floating[Any]:
        """Integrate f on [0,1] using n-point tanh-sinh."""
        nodes, weights = get_tanh_sinh_1d(n_pts, dtype)
        result = np.sum(weights * f(nodes))
        return cast(np.floating[Any], result)

    @pytest.mark.parametrize("power", [0, 1, 2, 3])
    def test_polynomial_integration(self, power: int) -> None:
        """Tanh-sinh integrates low-degree polynomials accurately with 30 points."""
        approx = self._integrate(30, lambda x: x**power)
        exact = 1.0 / (power + 1)
        nptest.assert_allclose(approx, exact, rtol=1e-10)

    @pytest.mark.parametrize("n_pts", [10, 20, 50])
    def test_smooth_function_convergence(self, n_pts: int) -> None:
        """Integration of exp(x) on [0,1] converges with increasing n."""
        approx = self._integrate(n_pts, np.exp)
        exact = np.e - 1.0
        if n_pts >= 50:  # noqa: PLR2004
            nptest.assert_allclose(approx, exact, rtol=1e-14)
        elif n_pts >= 20:  # noqa: PLR2004
            nptest.assert_allclose(approx, exact, rtol=1e-9)
        else:
            nptest.assert_allclose(approx, exact, rtol=1e-5)

    def test_endpoint_singular_integrand(self) -> None:
        """Tanh-sinh handles sqrt(x) (endpoint singularity) better than GL at same n."""
        from pantr.quad import get_gauss_legendre_1d  # noqa: PLC0415

        n = 30
        exact = 2.0 / 3.0  # integral of sqrt(x) on [0,1]

        ts_nodes, ts_weights = get_tanh_sinh_1d(n)
        ts_approx = float(np.sum(ts_weights * np.sqrt(ts_nodes)))

        gl_nodes, gl_weights = get_gauss_legendre_1d(n)
        gl_approx = float(np.sum(gl_weights * np.sqrt(gl_nodes)))

        ts_err = abs(ts_approx - exact)
        gl_err = abs(gl_approx - exact)

        # Tanh-sinh should outperform GL for endpoint-singular integrands
        assert ts_err < gl_err

    def test_constant_function(self) -> None:
        """Integral of 1 on [0,1] = 1, exact for any n."""
        for n in [1, 2, 5, 10]:
            approx = self._integrate(n, lambda x: np.ones_like(x))
            nptest.assert_allclose(approx, 1.0, rtol=1e-14)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_integration_dtype_consistency(self, dtype: npt.DTypeLike) -> None:
        """Integration result respects output dtype precision."""
        approx = self._integrate(30, lambda x: x**2, dtype)
        rtol = get_conservative(dtype)
        nptest.assert_allclose(float(approx), 1.0 / 3.0, rtol=rtol)


class TestTanhSinhOddEven:
    """Tests for odd vs even n_pts behavior."""

    def test_odd_n_has_midpoint(self) -> None:
        """Odd n includes a node at 0.5."""
        nodes, _ = get_tanh_sinh_1d(5)
        assert np.any(np.isclose(nodes, 0.5, atol=1e-14))

    def test_even_n_no_midpoint(self) -> None:
        """Even n does not include a node at 0.5."""
        nodes, _ = get_tanh_sinh_1d(6)
        assert not np.any(np.isclose(nodes, 0.5, atol=1e-14))
