"""Tests for tanh-sinh quadrature in pantr.quad."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
import pytest

from pantr.quad import (
    _TANH_SINH_DECAY_FACTOR,
    _generate_tanh_sinh,
    get_tanh_sinh_1d,
)
from pantr.quad._rules import _lambert_w_principal
from pantr.tolerance import get_conservative, get_machine_epsilon

# Golden node/weight values for ``get_tanh_sinh_1d``. Provenance: captured from
# the pre-refactor implementation on ``main`` (commit 71ede9a, the original
# algoim-derived rule) by calling ``get_tanh_sinh_1d(n)`` for each ``n`` below.
# Regenerate by checking out that commit and re-running the same calls. The
# public rule is consumed verbatim by a downstream consumer, so this guard pins
# the values the
# clean-room reimplementation must reproduce — it is NOT generated from the new
# code, so it genuinely tests backward compatibility.
_GOLDEN_PATH = Path(__file__).parent / "data" / "tanh_sinh_golden.npz"
_GOLDEN_N_PTS: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 50, 100, 200)
_UNIT_ROUNDOFF: float = get_machine_epsilon(np.float64) / 2.0
"""Half an eps, the spacing either side of a rounded float64 result."""


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
        if n_pts >= 50:
            nptest.assert_allclose(approx, exact, rtol=1e-14)
        elif n_pts >= 20:
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
            approx = self._integrate(n, np.ones_like)
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


def _resolvable(golden_nodes: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Select the golden nodes whose distance to an endpoint survives float64.

    The reference rule moved a node onto the boundary once its endpoint gap
    underflowed, and the affine map onto ``[0, 1]`` collapsed a few more onto
    ``1.0`` on its own. Those are exactly the nodes the current rule declines to
    emit, so this predicate turns the golden arrays into the expected answer.

    Args:
        golden_nodes (npt.NDArray[np.float64]): Golden nodes on ``[0, 1]``.

    Returns:
        npt.NDArray[np.bool_]: True where ``min(x, 1 - x)`` is at least half an
        ``eps``, the smallest gap ``1 - x`` can carry and stay below ``1``.
    """
    gap = np.minimum(golden_nodes, 1.0 - golden_nodes)
    return np.asarray(gap >= 0.5 * get_machine_epsilon(np.float64))


class TestTanhSinhGoldenValues:
    """Golden-value regression guarding a downstream consumer contract.

    ``pantr.quad.get_tanh_sinh_1d`` is imported by a downstream consumer, which
    feeds the returned nodes/weights straight into its implicit-quadrature
    kernels. The values must therefore stay numerically identical across
    refactors. These tests pin the node/weight arrays and the effective node
    count for a representative range of ``n_pts``.

    One deliberate divergence from the reference rule is asserted rather than
    excused. The reference snapped a node onto ``0`` or ``1`` once its endpoint
    gap underflowed and kept it, with a nonzero weight and sometimes duplicated,
    which made ``1/sqrt(x)`` -- the integrand the rule exists for -- come back as
    ``inf`` from ``n_pts = 45`` on. The rule now stops there instead. Everything
    the reference placed legitimately is still reproduced, and
    :func:`_resolvable` says exactly which entries went: at the fourteen point
    counts below, the current nodes and weights equal the golden arrays
    restricted by that predicate, to 2.2e-16 and 1.1e-16 respectively.

    All fourteen counts are unaffected by a separate limit that starts well
    above them: from ``n_pts = 544`` in float64 two consecutive samples round
    onto one representable coordinate, so the returned nodes stop being
    pairwise distinct. That is the format running out of coordinates, not a node
    reaching an endpoint, and it is out of this test's range.
    """

    @pytest.mark.parametrize("n_pts", _GOLDEN_N_PTS)
    def test_nodes_weights_match_golden(self, n_pts: int) -> None:
        """Nodes and weights reproduce the resolvable part of the golden arrays."""
        golden = np.load(_GOLDEN_PATH)
        nodes, weights = get_tanh_sinh_1d(n_pts)
        keep = _resolvable(golden[f"nodes_{n_pts}"])
        golden_nodes = golden[f"nodes_{n_pts}"][keep]
        golden_weights = golden[f"weights_{n_pts}"][keep]
        assert nodes.shape == golden_nodes.shape
        assert weights.shape == golden_weights.shape
        # Two comparisons, because one cannot guard this rule. The absolute bound
        # is what a caller integrating with these weights sees, and it is set to
        # 1e-15 against a measured worst deviation of 2.2e-16 over these fourteen
        # counts, a factor 4.5. It was 5e-15, which is 23x the measurement and,
        # more to the point, larger than 10 of the 382 node distances and 8 of the
        # 382 weights it is meant to guard: a flat absolute tolerance says nothing
        # about a quantity whose whole content is that it is small, and the
        # endpoint cluster being double-exponentially small is the property this
        # rule exists for. The relative bound covers exactly those, at 1e-13
        # against a measured 1.3e-14, a factor 8.
        #
        # Both are measurements of this implementation against the reference one,
        # not derivations, and they are deliberately not tightened to the
        # measurement: the golden file is a compatibility guard for a downstream
        # consumer, so it must not go red on a libm or numpy version that moves
        # the last bits. See design/backend_parity.md for what a derived bound on
        # this rule looks like.
        nptest.assert_allclose(nodes, golden_nodes, rtol=0.0, atol=1e-15)
        nptest.assert_allclose(weights, golden_weights, rtol=0.0, atol=1e-15)
        endpoint_distance = np.minimum(golden_nodes, 1.0 - golden_nodes)
        nptest.assert_allclose(
            np.minimum(nodes, 1.0 - nodes), endpoint_distance, rtol=1e-13, atol=0.0
        )
        nptest.assert_allclose(weights, golden_weights, rtol=1e-13, atol=0.0)

    def test_an_absolute_tolerance_alone_cannot_guard_the_endpoint_cluster(self) -> None:
        """A one-ulp move of the outermost node is invisible to any absolute bound.

        This is why the golden comparison carries a relative assertion as well,
        and it is the whole argument for it, so it is pinned rather than
        explained. The outermost node sits one ulp below 1, so its distance to
        the endpoint is 2.2e-16, the smallest a float64 can carry. Moving that
        node by a single ulp, the smallest change that exists, **doubles** that
        distance. The absolute deviation is 1.1e-16, which is below any absolute
        tolerance this test could sensibly use: 5e-15 as it was, 1e-15 as it now
        is, both blind to it.

        So an absolute bound permits a 50% error in the quantity the rule exists
        for. A double-exponential rule is what it is because the endpoint cluster
        is resolved; guarding it with a flat tolerance guards everything except
        that.
        """
        golden = np.load(_GOLDEN_PATH)
        keep = _resolvable(golden["nodes_200"])
        nodes = golden["nodes_200"][keep]

        outermost = int(np.argmin(np.minimum(nodes, 1.0 - nodes)))
        moved = nodes.copy()
        moved[outermost] = np.nextafter(moved[outermost], 0.0)

        distance = np.minimum(nodes, 1.0 - nodes)
        assert np.minimum(moved, 1.0 - moved)[outermost] == pytest.approx(
            2.0 * distance[outermost]
        ), "one ulp should double the endpoint distance of the outermost node"

        nptest.assert_allclose(moved, nodes, rtol=0.0, atol=1e-15)
        with pytest.raises(AssertionError):
            nptest.assert_allclose(np.minimum(moved, 1.0 - moved), distance, rtol=1e-13, atol=0.0)

    @pytest.mark.parametrize("n_pts", _GOLDEN_N_PTS)
    def test_effective_node_count_matches_golden(self, n_pts: int) -> None:
        """Truncation drops the reference's unresolvable nodes and nothing else."""
        golden = np.load(_GOLDEN_PATH)
        nodes, _ = get_tanh_sinh_1d(n_pts)
        golden_nodes = golden[f"nodes_{n_pts}"]
        assert nodes.shape[0] == int(np.count_nonzero(_resolvable(golden_nodes)))
        # 2, 2 and 4 entries go at n_pts 50, 100 and 200; none at any smaller count.
        assert nodes.shape[0] == golden_nodes.shape[0] or n_pts in (50, 100, 200)


class TestLambertWCoupling:
    """The tanh-sinh decay factor is a precondition of the Lambert W kernel.

    Neither constant records the other, and the failure is silent, so it is
    pinned here. ``_lambert_w_principal`` starts from the large-argument
    asymptotic form with no branch, which is what keeps it free of a comparison
    on the scalar. The price is a domain: for a small enough argument the guess
    lands off the principal branch and no number of Halley steps recovers.

    ``_TANH_SINH_DECAY_FACTOR`` is what sets the smallest argument the kernel
    ever sees, through ``2 * factor * (pi/2) * (n - 1)`` at ``n = 2``. So a
    change to that factor, made for a better discretization error, would produce
    a silently wrong step size and therefore a silently wrong rule.
    """

    def test_the_shipped_decay_factor_keeps_the_kernel_in_its_domain(self) -> None:
        """Four Halley steps reach one unit of roundoff at the smallest argument.

        The binding case is ``n = 2``, where the argument is smallest. Measured
        by bisection, four steps hold down to a decay factor of 0.5097 and three
        only to 0.5932, so the shipped 0.6 sits 18% clear with four and 1.1%
        clear with three. This asserts the shipped value is inside the domain,
        which is the part a future edit can break.
        """
        smallest_argument = 2.0 * _TANH_SINH_DECAY_FACTOR * (np.pi / 2.0) * (2 - 1)
        assert smallest_argument > 1.61, (
            "the decay factor puts the Lambert W kernel's smallest argument below "
            "its validity threshold; the asymptotic start leaves the principal branch"
        )

        w = _lambert_w_principal(smallest_argument)
        assert w * np.exp(w) == pytest.approx(smallest_argument, rel=4.0 * _UNIT_ROUNDOFF)

    @pytest.mark.parametrize("n_pts", [2, 3, 5, 17, 100, 400, 1000])
    def test_the_solver_satisfies_its_own_equation(self, n_pts: int) -> None:
        """``W e^W = x`` holds to its own conditioning, across the argument range.

        A residual check rather than a comparison against another implementation,
        so it stays an independent oracle rather than a second opinion. The
        conditioning of ``w e^w`` at the root is ``(1 + w)``, so a residual of a
        few units of roundoff is what a correctly rounded root gives.

        Args:
            n_pts (int): Point count, which sets the argument through the decay
                factor.
        """
        argument = 2.0 * _TANH_SINH_DECAY_FACTOR * (np.pi / 2.0) * (n_pts - 1)
        w = _lambert_w_principal(argument)
        residual = abs(w * np.exp(w) - argument)
        assert residual <= 4.0 * (1.0 + w) * _UNIT_ROUNDOFF * argument


class TestGenerateTanhSinhContract:
    """``_generate_tanh_sinh`` is private and its shape is frozen anyway.

    A separate, not-yet-public consumer imports pantr's private symbols, and this
    is the one private name in ``quad`` that returns something the public API
    cannot give: the rule on ``[-1, 1]`` in float64 together with the effective
    node count. This repository's CI cannot see breakage there, so the shape is
    asserted here rather than left to a reviewer to notice.
    """

    @pytest.mark.parametrize("n_pts", [1, 2, 5, 17, 100])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_shape_and_frame_are_unchanged(self, n_pts: int, dtype: npt.DTypeLike) -> None:
        """The return is ``(data, count)`` with ``data`` of shape ``(count, 2)`` on ``[-1, 1]``.

        Args:
            n_pts (int): Requested number of points.
            dtype (npt.DTypeLike): The dtype that sets the truncation point.
        """
        data, count = _generate_tanh_sinh(n_pts, dtype)
        assert isinstance(count, int)
        assert data.dtype == np.float64, "the data is float64 whatever dtype selects"
        assert data.shape == (count, 2), "columns are [node, weight]"
        nodes, weights = data[:, 0], data[:, 1]
        assert np.all(np.abs(nodes) < 1.0), "the frame is [-1, 1], open at both ends"
        assert np.all(weights > 0.0)
        assert weights.sum() == pytest.approx(2.0, rel=64.0 * _UNIT_ROUNDOFF), (
            "the weights carry the measure of [-1, 1], not of [0, 1]"
        )
