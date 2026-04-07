"""Height direction scoring for the implicit quadrature dimension reduction.

Selects the best coordinate axis for the height direction by evaluating
polynomial gradients at subcell midpoints and accumulating a score per
direction. Directions with detected discriminant-type intersections are
penalized.

Main exports:

- :func:`score_estimate_2d` -- score all directions for 2D polynomials.
- :func:`score_estimate_3d` -- score all directions for 3D polynomials.

Note:
    Inputs are assumed to be correct (no validation performed).
    These are Layer 3 kernels for the implicit quadrature module.
"""

from __future__ import annotations

import numpy as np
from numba.typed import List as NumbaList
from numpy import typing as npt

from pantr._numba_compat import nb_jit
from pantr.bezier.implicit._bernstein import (
    _elevated_derivative_along_axis_2d,
    _elevated_derivative_along_axis_3d,
    _eval_bernstein_basis_1d_into,
)
from pantr.bezier.implicit._mask import (
    M,
    has_intersection_2d,
    has_intersection_3d,
)

_NEAR_ZERO: float = 1e-300
"""Guard against division by zero (well below subnormal range)."""


@nb_jit(nopython=True, cache=True)
def _deriv_basis_val(
    n: int,
    fn: float,
    bm: npt.NDArray[np.float64],
    ii: int,
) -> float:
    """Evaluate the *ii*-th derivative basis function value from a lower-degree basis.

    Computes ``n * (B_{ii-1,n-1}(x) - B_{ii,n-1}(x))`` given the pre-evaluated
    lower-degree basis ``bm = B_{*,n-1}(x)``.

    Args:
        n (int): Polynomial degree.
        fn (float): Float cast of *n*.
        bm (npt.NDArray[np.float64]): Lower-degree basis values of shape ``(n,)``.
        ii (int): Basis function index in ``[0, n]``.

    Returns:
        float: Derivative basis value at the pre-evaluated point.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    if n == 0:
        return 0.0
    if ii == 0:
        return float(-fn * bm[0])
    if ii == n:
        return float(fn * bm[n - 1])
    return float(fn * (bm[ii - 1] - bm[ii]))


_MAX_SCORE_SAMPLES_3D: int = 24
"""Maximum number of subcell gradient evaluations in 3D score estimation.

The score is a heuristic for axis selection; sampling a subset of active
subcells is sufficient for a reliable estimate while significantly reducing
the cost for polynomials with many active mask cells.
"""


@nb_jit(nopython=True, cache=True)
def score_estimate_2d(
    coeffs_list: NumbaList,
    masks_list: NumbaList,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Estimate scores for each height direction in 2D.

    For each polynomial, samples the gradient at the midpoint of every active
    mask subcell and accumulates ``|d_k phi| / ||grad phi||_1`` per direction k.
    The direction with the highest score is the best elimination axis.

    Pre-computes derivative coefficient arrays once per polynomial to avoid
    redundant allocation and differentiation inside the subcell loop.

    Args:
        coeffs_list (NumbaList): List of 2D coefficient arrays.
        masks_list (NumbaList): List of 2D boolean mask arrays.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
            ``(scores, has_disc)`` where *scores* has shape ``(2,)`` and
            *has_disc* has shape ``(2,)`` indicating directions with
            non-empty discriminant-like features (used in the build phase
            to apply a score bonus and select tanh-sinh quadrature).

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    scores = np.zeros(2, dtype=np.float64)
    has_disc = np.zeros(2, dtype=np.bool_)
    inv_m = 1.0 / M

    n_polys = len(coeffs_list)
    for p in range(n_polys):
        coeffs = coeffs_list[p]
        mask = masks_list[p]
        n0 = coeffs.shape[0] - 1
        n1 = coeffs.shape[1] - 1

        # Pre-allocate basis buffers outside the subcell loop.
        b0 = np.empty(n0 + 1, dtype=np.float64)
        b1 = np.empty(n1 + 1, dtype=np.float64)
        # Derivative basis: reuse lower-degree basis arrays.
        b0m = np.empty(max(n0, 1), dtype=np.float64)  # degree n0-1
        b1m = np.empty(max(n1, 1), dtype=np.float64)  # degree n1-1

        fn0 = float(n0)
        fn1 = float(n1)

        # Accumulate gradient-based score with inlined gradient computation
        # (no per-call allocations).
        for i0 in range(M):
            x0 = (i0 + 0.5) * inv_m
            _eval_bernstein_basis_1d_into(n0, x0, b0)
            if n0 > 0:
                _eval_bernstein_basis_1d_into(n0 - 1, x0, b0m)
            for i1 in range(M):
                if not mask[i0, i1]:
                    continue
                x1 = (i1 + 0.5) * inv_m
                _eval_bernstein_basis_1d_into(n1, x1, b1)
                if n1 > 0:
                    _eval_bernstein_basis_1d_into(n1 - 1, x1, b1m)

                # Inline fused gradient computation.
                g0 = 0.0
                g1 = 0.0
                for ii0 in range(n0 + 1):
                    p0_val = _deriv_basis_val(n0, fn0, b0m, ii0)
                    acc_b1 = 0.0
                    acc_p1 = 0.0
                    for ii1 in range(n1 + 1):
                        c = coeffs[ii0, ii1]
                        acc_b1 += c * b1[ii1]
                        acc_p1 += c * _deriv_basis_val(n1, fn1, b1m, ii1)

                    g0 += p0_val * acc_b1
                    g1 += b0[ii0] * acc_p1

                norm1 = abs(g0) + abs(g1)
                if norm1 > _NEAR_ZERO:
                    scores[0] += abs(g0) / norm1
                    scores[1] += abs(g1) / norm1

        # Check for discriminant features.
        for k in range(2):
            if has_disc[k]:
                continue
            ed = _elevated_derivative_along_axis_2d(coeffs, k)
            if has_intersection_2d(coeffs, mask, ed, mask):
                has_disc[k] = True

    return scores, has_disc


@nb_jit(nopython=True, cache=True)
def score_estimate_3d(  # noqa: PLR0912, PLR0915
    coeffs_list: NumbaList,
    masks_list: NumbaList,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Estimate scores for each height direction in 3D.

    Pre-computes derivative coefficient arrays once per polynomial to avoid
    redundant allocation and differentiation inside the subcell loop.

    Args:
        coeffs_list (NumbaList): List of 3D coefficient arrays.
        masks_list (NumbaList): List of 3D boolean mask arrays.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
            ``(scores, has_disc)`` with shapes ``(3,)`` each.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    scores = np.zeros(3, dtype=np.float64)
    has_disc = np.zeros(3, dtype=np.bool_)
    inv_m = 1.0 / M

    n_polys = len(coeffs_list)
    for p in range(n_polys):
        coeffs = coeffs_list[p]
        mask = masks_list[p]
        n0 = coeffs.shape[0] - 1
        n1 = coeffs.shape[1] - 1
        n2 = coeffs.shape[2] - 1

        # Pre-allocate basis buffers outside the subcell loop.
        b0 = np.empty(n0 + 1, dtype=np.float64)
        b1 = np.empty(n1 + 1, dtype=np.float64)
        b2 = np.empty(n2 + 1, dtype=np.float64)
        b0m = np.empty(max(n0, 1), dtype=np.float64)
        b1m = np.empty(max(n1, 1), dtype=np.float64)
        b2m = np.empty(max(n2, 1), dtype=np.float64)

        fn0 = float(n0)
        fn1 = float(n1)
        fn2 = float(n2)

        # Deterministic subsampling: use a prime stride over the linear mask
        # index to reduce alignment artifacts.  A prime stride ensures the
        # sampling pattern is not aligned with any regular mask structure.
        total_cells = M * M * M
        raw_stride = max(total_cells // _MAX_SCORE_SAMPLES_3D, 1)
        # Round up to the next odd value to avoid powers-of-two alignment;
        # for M=4 this gives stride 3 (prime), sampling ~21 of 64 cells.
        skip_stride = raw_stride + 1 if raw_stride % 2 == 0 and raw_stride > 1 else raw_stride
        cell_idx = 0

        # Accumulate gradient-based score with inlined gradient computation.
        # Factored loop: reuse axis-0 basis across i1, i2 iterations.
        for i0 in range(M):
            x0 = (i0 + 0.5) * inv_m
            _eval_bernstein_basis_1d_into(n0, x0, b0)
            if n0 > 0:
                _eval_bernstein_basis_1d_into(n0 - 1, x0, b0m)
            for i1 in range(M):
                x1 = (i1 + 0.5) * inv_m
                _eval_bernstein_basis_1d_into(n1, x1, b1)
                if n1 > 0:
                    _eval_bernstein_basis_1d_into(n1 - 1, x1, b1m)
                for i2 in range(M):
                    cur_idx = cell_idx
                    cell_idx += 1
                    if not mask[i0, i1, i2]:
                        continue
                    # Subsample: only evaluate every skip_stride-th active cell.
                    if cur_idx % skip_stride != 0:
                        continue

                    x2 = (i2 + 0.5) * inv_m
                    _eval_bernstein_basis_1d_into(n2, x2, b2)
                    if n2 > 0:
                        _eval_bernstein_basis_1d_into(n2 - 1, x2, b2m)

                    # Inline gradient computation.
                    g0 = 0.0
                    g1 = 0.0
                    g2 = 0.0
                    for ii0 in range(n0 + 1):
                        p0_val = _deriv_basis_val(n0, fn0, b0m, ii0)
                        for ii1 in range(n1 + 1):
                            p1_val = _deriv_basis_val(n1, fn1, b1m, ii1)
                            v01 = b0[ii0] * b1[ii1]
                            p0_b1 = p0_val * b1[ii1]
                            b0_p1 = b0[ii0] * p1_val
                            for ii2 in range(n2 + 1):
                                c = coeffs[ii0, ii1, ii2]
                                g0 += c * p0_b1 * b2[ii2]
                                g1 += c * b0_p1 * b2[ii2]
                                g2 += c * v01 * _deriv_basis_val(n2, fn2, b2m, ii2)

                    norm1 = abs(g0) + abs(g1) + abs(g2)
                    if norm1 > _NEAR_ZERO:
                        scores[0] += abs(g0) / norm1
                        scores[1] += abs(g1) / norm1
                        scores[2] += abs(g2) / norm1

        # Check for discriminant features.
        for k in range(3):
            if has_disc[k]:
                continue
            ed = _elevated_derivative_along_axis_3d(coeffs, k)
            if has_intersection_3d(coeffs, mask, ed, mask):
                has_disc[k] = True

    return scores, has_disc
