"""Tests for B-spline blossom (polar form) evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.bspline._bspline_blossom import _evaluate_blossom_1d
from pantr.bspline._bspline_blossom_core import _blossom_span

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval_blossom(
    knots: list[float],
    degree: int,
    ctrl: list[float] | list[list[float]],
    u_values: list[float],
) -> npt.NDArray[Any]:
    """Thin wrapper: build numpy arrays and call _evaluate_blossom_1d."""
    kv = np.array(knots, dtype=np.float64)
    cp = np.array(ctrl, dtype=np.float64)
    if cp.ndim == 1:
        cp = cp[:, np.newaxis]
    u = np.array(u_values, dtype=np.float64)
    tol = float(np.finfo(np.float64).eps * max(1.0, float(np.abs(kv).max())) * 64)
    return _evaluate_blossom_1d(kv, degree, cp, u, tol)


def _eval_spline(
    knots: list[float],
    degree: int,
    ctrl: list[float],
    t: float,
) -> float:
    """Evaluate a 1D B-spline at a single point via de Boor."""
    kv = np.array(knots, dtype=np.float64)
    cp = np.array(ctrl, dtype=np.float64)[:, np.newaxis]
    space = BsplineSpace([BsplineSpace1D(kv, degree)])
    f = Bspline(space, cp)
    return float(f.evaluate(np.array([t])))


# ---------------------------------------------------------------------------
# Diagonal property: blossom(t, t, ..., t) == f(t)
# ---------------------------------------------------------------------------


class TestBlossomDiagonal:
    """Test that the diagonal of the blossom equals the B-spline evaluation."""

    def test_degree2_single_element(self) -> None:
        """Diagonal property for a degree-2 single-element B-spline."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        ctrl = [1.0, 2.0, 3.0]

        for t in np.linspace(0.0, 1.0, 9):
            blossom_val = _eval_blossom(knots, 2, ctrl, [t, t])
            direct_val = _eval_spline(knots, 2, ctrl, t)
            np.testing.assert_allclose(blossom_val[0], direct_val, atol=1e-12, err_msg=f"t={t}")

    def test_degree3_two_elements(self) -> None:
        """Diagonal property for a cubic B-spline with one interior knot."""
        knots = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
        ctrl = [1.0, 0.5, 2.0, 1.5, 0.5]

        for t in np.linspace(0.0, 1.0, 11):
            blossom_val = _eval_blossom(knots, 3, ctrl, [t, t, t])
            direct_val = _eval_spline(knots, 3, ctrl, t)
            np.testing.assert_allclose(blossom_val[0], direct_val, atol=1e-12, err_msg=f"t={t}")

    def test_degree1_linear(self) -> None:
        """Diagonal property for a degree-1 B-spline (piecewise linear)."""
        knots = [0.0, 0.0, 0.5, 1.0, 1.0]
        ctrl = [1.0, 3.0, 2.0]

        for t in np.linspace(0.0, 1.0, 9):
            blossom_val = _eval_blossom(knots, 1, ctrl, [t])
            direct_val = _eval_spline(knots, 1, ctrl, t)
            np.testing.assert_allclose(blossom_val[0], direct_val, atol=1e-12, err_msg=f"t={t}")

    def test_vector_rank(self) -> None:
        """Diagonal property holds for each component of a vector-valued spline."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        ctrl = [[1.0, 2.0], [3.0, 0.5], [2.0, 1.5]]

        for t in np.linspace(0.0, 1.0, 7):
            blossom_val = _eval_blossom(knots, 2, ctrl, [t, t])
            # Evaluate each component separately
            val0 = _eval_spline(knots, 2, [row[0] for row in ctrl], t)
            val1 = _eval_spline(knots, 2, [row[1] for row in ctrl], t)
            np.testing.assert_allclose(blossom_val[0], val0, atol=1e-12)
            np.testing.assert_allclose(blossom_val[1], val1, atol=1e-12)


# ---------------------------------------------------------------------------
# Symmetry: blossom(u1, u2) == blossom(u2, u1)
# ---------------------------------------------------------------------------


class TestBlossomSymmetry:
    """Test that the blossom is symmetric in its arguments."""

    def test_degree2_symmetry(self) -> None:
        """Blossom is symmetric for degree-2 spline."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        ctrl = [1.0, 2.0, 0.5, 3.0]

        for u1, u2 in [(0.1, 0.4), (0.2, 0.8), (0.5, 0.5), (0.0, 1.0)]:
            b12 = _eval_blossom(knots, 2, ctrl, [u1, u2])
            b21 = _eval_blossom(knots, 2, ctrl, [u2, u1])
            np.testing.assert_allclose(b12, b21, atol=1e-12)

    def test_degree3_symmetry(self) -> None:
        """Blossom is symmetric for degree-3 spline with two elements."""
        knots = [0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0, 1.0]
        ctrl = [1.0, 0.5, 2.0, 1.5, 3.0, 0.5]

        u1, u2, u3 = 0.1, 0.5, 0.8
        orderings = [
            [u1, u2, u3],
            [u1, u3, u2],
            [u2, u1, u3],
            [u2, u3, u1],
            [u3, u1, u2],
            [u3, u2, u1],
        ]
        results = [_eval_blossom(knots, 3, ctrl, perm) for perm in orderings]
        for res in results[1:]:
            np.testing.assert_allclose(res, results[0], atol=1e-12)


# ---------------------------------------------------------------------------
# Control point recovery: blossom(t_{i+1}, ..., t_{i+p}) == P_i
# ---------------------------------------------------------------------------


class TestBlossomControlPointRecovery:
    """Test that the blossom at consecutive interior knots recovers control points."""

    def test_degree2_knot_values(self) -> None:
        """blossom(t_{i+1}, t_{i+2}) == P[i] for degree-2 B-spline."""
        knots = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
        ctrl = [1.0, 2.0, 0.5, 3.0]
        n = len(ctrl)
        p = 2

        for i in range(n):
            u_vals = [float(knots[i + r]) for r in range(1, p + 1)]
            blossom_val = _eval_blossom(knots, p, ctrl, u_vals)
            np.testing.assert_allclose(blossom_val[0], ctrl[i], atol=1e-12, err_msg=f"i={i}")

    def test_degree3_knot_values(self) -> None:
        """blossom(t_{i+1}, t_{i+2}, t_{i+3}) == P[i] for degree-3 B-spline."""
        knots = [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
        ctrl = [1.0, 0.5, 2.0, 1.5, 0.5]
        n = len(ctrl)
        p = 3

        for i in range(n):
            u_vals = [float(knots[i + r]) for r in range(1, p + 1)]
            blossom_val = _eval_blossom(knots, p, ctrl, u_vals)
            np.testing.assert_allclose(blossom_val[0], ctrl[i], atol=1e-12, err_msg=f"i={i}")


# ---------------------------------------------------------------------------
# Input validation errors
# ---------------------------------------------------------------------------


class TestBlossomValidation:
    """Tests for input validation in _evaluate_blossom_1d."""

    def test_wrong_u_values_length(self) -> None:
        """Raises ValueError when len(u_values) != degree."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        ctrl = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        u = np.array([0.5], dtype=np.float64)  # degree=2 requires 2 values
        with pytest.raises(ValueError, match="length"):
            _evaluate_blossom_1d(
                np.array(knots, dtype=np.float64),
                2,
                ctrl,
                u,
                1e-12,
            )

    def test_u_values_out_of_domain(self) -> None:
        """Raises ValueError when a u value lies outside the domain."""
        knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        ctrl = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
        u = np.array([0.5, 1.5], dtype=np.float64)  # 1.5 > domain [0, 1]
        with pytest.raises(ValueError, match="outside domain"):
            _evaluate_blossom_1d(
                np.array(knots, dtype=np.float64),
                2,
                ctrl,
                u,
                1e-12,
            )


# ---------------------------------------------------------------------------
# Knot-span search: the binary search must reproduce the linear scan
# ---------------------------------------------------------------------------


def _linear_scan_span(knots: npt.NDArray[Any], u_last: float, n: int, degree: int) -> int:
    """The ``O(n)`` scan ``_blossom_span`` replaced, kept here as the reference oracle.

    Verbatim from the kernel before the change, including its ``k = n`` fall-through when
    no half-open span contains ``u_last``.
    """
    k = n
    for idx in range(n + degree):
        if knots[idx] <= u_last < knots[idx + 1]:
            k = idx
            break
    return k


def _random_open_knots(
    degree: int, n_interior: int, seed: int, *, repeat_first: bool = False
) -> npt.NDArray[np.float64]:
    """Return a clamped knot vector on ``[0, 1]`` with random interior knots."""
    rng = np.random.default_rng(seed)
    interior = np.sort(rng.uniform(0.0, 1.0, size=n_interior))
    if repeat_first and n_interior > 1:
        interior[1] = interior[0]
    return np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)]).astype(np.float64)


class TestBlossomSpan:
    """The binary span search agrees with the linear scan on every valid input."""

    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("n_interior", [0, 1, 2, 5])
    @pytest.mark.parametrize("repeat_first", [False, True])
    def test_matches_the_linear_scan(
        self, degree: int, n_interior: int, repeat_first: bool
    ) -> None:
        """Identical span indices, interior multiplicities and endpoints included.

        The probe set covers the interior, both domain endpoints, every interior knot
        exactly, and the band just outside the domain that Layer 2 still accepts within
        its tolerance.
        """
        knots = _random_open_knots(
            degree, n_interior, seed=100 * degree + n_interior, repeat_first=repeat_first
        )
        n = knots.shape[0] - degree - 2
        interior = knots[degree + 1 : knots.shape[0] - degree - 1]
        rng = np.random.default_rng(7)
        probes = [
            *rng.uniform(0.0, 1.0, size=30).tolist(),
            0.0,
            1.0,
            *interior.tolist(),
            -1e-12,
            1.0 + 1e-12,
        ]
        for u in probes:
            assert int(_blossom_span(knots, float(u), n)) == _linear_scan_span(
                knots, float(u), n, degree
            ), f"degree={degree} u={u!r} knots={knots.tolist()}"

    def test_below_the_first_knot_gives_the_rightmost_span(self) -> None:
        """Preserved behaviour: an under-range parameter falls through to span ``n``.

        Reachable through Layer 2, which accepts parameters up to its tolerance below the
        domain start. The scan returned ``n`` here because no span matched; the binary
        search would give ``-1`` without the guard.
        """
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        n = knots.shape[0] - 4
        assert int(_blossom_span(knots, -1e-9, n)) == n
        assert _linear_scan_span(knots, -1e-9, n, 2) == n

    def test_at_and_beyond_the_last_knot_clamps(self) -> None:
        """The right domain endpoint and anything past it both give span ``n``."""
        knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
        n = knots.shape[0] - 4
        assert int(_blossom_span(knots, 1.0, n)) == n
        assert int(_blossom_span(knots, 2.0, n)) == n

    def test_non_clamped_knots_stay_in_range(self) -> None:
        """On a non-clamped vector a parameter past the domain still yields a valid index.

        Here the linear scan matched a span above ``n`` and the caller then read past the
        control points; the clamp removes that. Inputs like this are outside the Layer 2
        contract either way.
        """
        knots = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        n = knots.shape[0] - 4  # degree 2 => 3 control points
        assert int(_blossom_span(knots, 4.5, n)) == n
        assert _linear_scan_span(knots, 4.5, n, 2) > n

    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
    def test_diagonal_property_with_interior_multiplicities(self, degree: int) -> None:
        """``f[t, ..., t] == f(t)`` on knots with a repeated interior knot.

        The end-to-end guard on the span change: degrees 1 to 5, a non-uniform knot vector
        carrying an interior multiplicity, and ``t`` at both endpoints, exactly on every
        interior knot, and in between.
        """
        knots = _random_open_knots(degree, 4, seed=degree, repeat_first=True)
        space = BsplineSpace1D(knots, degree)
        rng = np.random.default_rng(degree + 50)
        ctrl = rng.uniform(-1.0, 1.0, size=(space.num_basis, 2))
        curve = Bspline(BsplineSpace([space]), np.ascontiguousarray(ctrl))
        interior = knots[degree + 1 : knots.shape[0] - degree - 1]
        tol = 64.0 * float(np.finfo(np.float64).eps)

        for t in [0.0, 1.0, *interior.tolist(), *rng.uniform(0.0, 1.0, size=10).tolist()]:
            blossom = _evaluate_blossom_1d(knots, degree, ctrl, np.full(degree, float(t)), tol)
            direct = np.asarray(curve.evaluate(np.array([float(t)]))).reshape(2)
            np.testing.assert_allclose(
                np.asarray(blossom).reshape(2), direct, atol=tol, rtol=0.0, err_msg=f"t={t}"
            )
