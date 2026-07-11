"""Regression tests for Bernstein ratio-recurrence underflow near u=1 (issue #258).

The O(p) ratio recurrence in ``_bernstein_point`` and the fused kernel
``_evaluate_bezier_1d_core`` now branches on ``u > 0.5`` and runs from
whichever endpoint keeps the seed term (``(1-u)^p`` or ``u^p``) bounded below
by ``0.5^p``, using the symmetry ``B_i,p(u) = B_{p-i},p(1-u)``. Before the
fix, the forward-only recurrence seeded from ``(1-u)^p`` underflowed to exact
zero for ``u`` close enough to 1 at high degree, and every subsequent term (a
positive multiple of the previous one) stayed zero: ``sum_i B_i(u)`` collapsed
to 0 instead of 1.

This module covers:
    - Partition of unity across degrees and points spanning both recurrence
      branches and their shared midpoint (float32 and float64).
    - Agreement with an independent O(p^2) reference (corner-cutting
      "AllBernstein", Piegl & Tiller A1.3), which never underflows because
      every intermediate value is a convex combination weighted by ``u`` and
      ``1-u`` rather than a ratio that can blow up.
    - Symmetry between the forward branch (evaluated at ``u <= 0.5``) and the
      mirrored branch (evaluated at ``1-u > 0.5``).
    - Consistency between the 0th row of the derivative kernel
      (``_bernstein_derivs_point``, an unaffected O(p^2) ``ndu``-table
      recurrence) and the fixed value kernel.
    - An end-to-end guard for the fused Bezier-evaluation kernel
      (``_evaluate_bezier_1d_core``) against de Casteljau evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr.basis._basis_core import _bernstein_derivs_point, _bernstein_point
from pantr.bezier._bezier_core import _evaluate_bezier_1d_core, _slice_bezier_1d_core

DEGREES = [2, 5, 10, 20, 30, 64]

_EPS64 = float(np.finfo(np.float64).eps)
_EPS32 = float(np.finfo(np.float32).eps)

# Points spanning both recurrence branches and their shared midpoint, plus the
# near-1 region where the pre-fix forward recurrence underflows.
_U64 = [0.0, 1e-17, 0.25, 0.5 - _EPS64, 0.5, 0.5 + _EPS64, 0.75, 1.0 - 1e-8, 1.0 - 1e-16, 1.0]
_U32 = [0.0, 1e-7, 0.25, 0.5 - _EPS32, 0.5, 0.5 + _EPS32, 0.75, 1.0 - 1e-4, 1.0 - _EPS32, 1.0]


def _eval_bernstein_row(n: int, u: float, dtype: npt.DTypeLike) -> npt.NDArray[np.floating[Any]]:
    """Evaluate the Layer-3 kernel `_bernstein_point` at one point."""
    dt = np.dtype(dtype)
    out_row = np.empty(n + 1, dtype=dt)
    _bernstein_point(np.int32(n), dt.type(u), out_row)
    return out_row


def _all_bernstein_reference(
    n: int, u: float, dtype: npt.DTypeLike = np.float64
) -> npt.NDArray[np.floating[Any]]:
    """Evaluate all degree-n Bernstein basis functions via corner-cutting (A1.3).

    Independent O(n^2) reference (Piegl & Tiller, "AllBernstein"): every
    intermediate value is a convex combination of already-computed entries
    weighted by ``u`` and ``1 - u``, both in ``[0, 1]``, so there is no ratio
    blow-up and hence no risk of the total-flush failure mode this module
    guards against.
    """
    dt = np.dtype(dtype)
    one = dt.type(1.0)
    uu = dt.type(u)
    u1 = one - uu

    row = np.zeros(n + 1, dtype=dt)
    row[0] = one
    for j in range(1, n + 1):
        saved = dt.type(0.0)
        for k in range(j):
            temp = row[k]
            row[k] = saved + u1 * temp
            saved = uu * temp
        row[j] = saved
    return row


class TestPartitionOfUnity:
    """Partition of unity must hold across both recurrence branches."""

    @pytest.mark.parametrize("u", _U64, ids=lambda u: f"{u:.17g}")
    @pytest.mark.parametrize("degree", DEGREES)
    def test_float64(self, degree: int, u: float) -> None:
        """Sum of Bernstein basis values stays within 8*p*eps of 1 (float64).

        This is the fixed version of the regression covered by issue #258:
        the (degree=30, u=1-1e-16) cell used to fail before the mirrored
        recurrence was introduced.
        """
        row = _eval_bernstein_row(degree, u, np.float64)
        assert abs(float(row.sum()) - 1.0) <= 8 * degree * _EPS64

    @pytest.mark.parametrize("u", _U32, ids=lambda u: f"{u:.9g}")
    @pytest.mark.parametrize("degree", DEGREES)
    def test_float32(self, degree: int, u: float) -> None:
        """Sum of Bernstein basis values stays within 8*p*eps of 1 (float32)."""
        row = _eval_bernstein_row(degree, u, np.float32)
        assert abs(float(row.sum()) - 1.0) <= 8 * degree * _EPS32


class TestReferenceCrossCheck:
    """Compare against the independent O(p^2) AllBernstein reference."""

    @pytest.mark.parametrize("degree", DEGREES)
    def test_matches_all_bernstein_reference(self, degree: int) -> None:
        """Kernel values match the reference over a grid biased toward u=1.

        The grid mixes uniform random points with a geometric sequence
        approaching both endpoints, since uniform sampling in [0, 1] almost
        never lands within the underflow-prone region near u=1 that this fix
        targets.
        """
        rng = np.random.default_rng(1000 + degree)
        uniform_pts = rng.uniform(0.0, 1.0, size=64)
        near_boundary = np.concatenate(
            [1.0 - np.geomspace(1e-16, 1e-2, 15), np.geomspace(1e-16, 1e-2, 15)]
        )
        pts = np.concatenate([uniform_pts, near_boundary, [0.0, 0.5, 1.0]])

        for u in pts:
            row = _eval_bernstein_row(degree, float(u), np.float64)
            ref = _all_bernstein_reference(degree, float(u))
            np.testing.assert_allclose(row, ref, rtol=1e-14, atol=1e-14)


class TestForwardMirroredSymmetry:
    """Forward branch at u must match the mirrored branch at 1-u, reversed."""

    @pytest.mark.parametrize("degree", DEGREES)
    def test_symmetry(self, degree: int) -> None:
        """B_i(u) (forward branch) equals B_{p-i}(1-u) (mirrored branch).

        Values are compared within ``p*eps``, not bitwise: the two branches
        accumulate the same mathematical quantity through different sequences
        of floating-point multiplications, so rounding differs by up to a few
        ULPs per step.
        """
        rng = np.random.default_rng(2000 + degree)
        # u < 0.5 strictly, so `u` takes the forward branch and `1 - u` (> 0.5)
        # takes the mirrored branch.
        us = np.concatenate([rng.uniform(0.0, 0.5, size=20), [1e-17, 0.25]])
        for u in us:
            u = float(u)
            row_forward = _eval_bernstein_row(degree, u, np.float64)
            row_mirrored = _eval_bernstein_row(degree, 1.0 - u, np.float64)
            np.testing.assert_allclose(
                row_forward,
                row_mirrored[::-1],
                rtol=degree * _EPS64,
                atol=degree * _EPS64,
            )


class TestDerivativeConsistency:
    """The unaffected O(p^2) derivative kernel must agree with the fixed kernel."""

    @pytest.mark.parametrize("u", _U64, ids=lambda u: f"{u:.17g}")
    @pytest.mark.parametrize("degree", DEGREES)
    def test_row0_matches_value_kernel(self, degree: int, u: float) -> None:
        """Row 0 of `_bernstein_derivs_point` (ndu table) matches `_bernstein_point`."""
        row = _eval_bernstein_row(degree, u, np.float64)

        n_deriv = 2
        out_pt = np.empty((n_deriv + 1, degree + 1), dtype=np.float64)
        _bernstein_derivs_point(np.int32(degree), np.float64(u), n_deriv, out_pt)

        np.testing.assert_allclose(out_pt[0, :], row, rtol=1e-13, atol=1e-13)


class TestFusedBezierKernel:
    """End-to-end guard for the fused evaluation kernel `_evaluate_bezier_1d_core`."""

    def test_degree_30_near_u1_matches_de_casteljau(self) -> None:
        """Random-control-point degree-30 Bezier at u=1-1e-16 matches de Casteljau.

        `_slice_bezier_1d_core` is a stable de Casteljau triangular reduction
        (no ratio recurrence, hence no underflow risk) used here as the
        reference for the fused kernel under test.
        """
        degree = 30
        rank = 4
        u = 1.0 - 1e-16
        rng = np.random.default_rng(42)
        ctrl = rng.normal(size=(degree + 1, rank))

        out_fused = np.empty((1, rank), dtype=np.float64)
        _evaluate_bezier_1d_core(ctrl, np.array([u]), out_fused)

        out_ref = np.empty(rank, dtype=np.float64)
        _slice_bezier_1d_core(ctrl, u, out_ref)

        np.testing.assert_allclose(out_fused[0], out_ref, rtol=1e-10, atol=1e-10)
