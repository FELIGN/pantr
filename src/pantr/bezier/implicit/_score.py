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
    _eval_gradient_fused_2d,
    _eval_gradient_fused_3d,
)
from pantr.bezier.implicit._mask import (
    M,
    has_intersection_2d,
    has_intersection_3d,
)

_NEAR_ZERO: float = 1e-300
"""Guard against division by zero (well below subnormal range)."""


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
    x = np.empty(2, dtype=np.float64)
    inv_m = 1.0 / M

    n_polys = len(coeffs_list)
    for p in range(n_polys):
        coeffs = coeffs_list[p]
        mask = masks_list[p]

        # Accumulate gradient-based score at each active subcell midpoint
        # using the fused gradient evaluation (single pass over coefficients).
        for i0 in range(M):
            for i1 in range(M):
                if not mask[i0, i1]:
                    continue
                x[0] = (i0 + 0.5) * inv_m
                x[1] = (i1 + 0.5) * inv_m
                grad = _eval_gradient_fused_2d(coeffs, x)
                norm1 = abs(grad[0]) + abs(grad[1])
                if norm1 > _NEAR_ZERO:
                    scores[0] += abs(grad[0]) / norm1
                    scores[1] += abs(grad[1]) / norm1

        # Check for discriminant features: does the intersection mask of
        # phi and elevated_derivative(phi, k) contain any active subcells?
        # This detects vertical tangents / branching points.
        for k in range(2):
            if has_disc[k]:
                continue
            ed = _elevated_derivative_along_axis_2d(coeffs, k)
            if has_intersection_2d(coeffs, mask, ed, mask):
                has_disc[k] = True

    return scores, has_disc


@nb_jit(nopython=True, cache=True)
def score_estimate_3d(
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
    x = np.empty(3, dtype=np.float64)
    inv_m = 1.0 / M

    n_polys = len(coeffs_list)
    for p in range(n_polys):
        coeffs = coeffs_list[p]
        mask = masks_list[p]

        # Accumulate gradient-based score using fused gradient evaluation
        # (single pass over coefficients, shared basis evaluations).
        for i0 in range(M):
            for i1 in range(M):
                for i2 in range(M):
                    if not mask[i0, i1, i2]:
                        continue
                    x[0] = (i0 + 0.5) * inv_m
                    x[1] = (i1 + 0.5) * inv_m
                    x[2] = (i2 + 0.5) * inv_m
                    grad = _eval_gradient_fused_3d(coeffs, x)
                    norm1 = abs(grad[0]) + abs(grad[1]) + abs(grad[2])
                    if norm1 > _NEAR_ZERO:
                        scores[0] += abs(grad[0]) / norm1
                        scores[1] += abs(grad[1]) / norm1
                        scores[2] += abs(grad[2]) / norm1

        # Check for discriminant features.
        for k in range(3):
            if has_disc[k]:
                continue
            ed = _elevated_derivative_along_axis_3d(coeffs, k)
            if has_intersection_3d(coeffs, mask, ed, mask):
                has_disc[k] = True

    return scores, has_disc
