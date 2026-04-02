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
from pantr.bezier.implicit._bernstein import _eval_gradient_2d, _eval_gradient_3d
from pantr.bezier.implicit._mask import M


@nb_jit(nopython=True, cache=True)
def score_estimate_2d(
    coeffs_list: NumbaList,
    masks_list: NumbaList,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Estimate scores for each height direction in 2D.

    For each polynomial, samples the gradient at the midpoint of every active
    mask subcell and accumulates ``|d_k phi| / ||grad phi||_1`` per direction k.
    The direction with the highest score is the best elimination axis.

    Args:
        coeffs_list (NumbaList): List of 2D coefficient arrays.
        masks_list (NumbaList): List of 2D boolean mask arrays.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
            ``(scores, has_disc)`` where *scores* has shape ``(2,)`` and
            *has_disc* has shape ``(2,)`` indicating directions with
            non-empty discriminant-like features (not used in initial
            implementation but reserved for future aggregate mode).

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
        for i0 in range(M):
            for i1 in range(M):
                if not mask[i0, i1]:
                    continue
                x[0] = (i0 + 0.5) * inv_m
                x[1] = (i1 + 0.5) * inv_m
                grad = _eval_gradient_2d(coeffs, x)
                norm1 = abs(grad[0]) + abs(grad[1])
                if norm1 > 1e-300:  # noqa: PLR2004
                    for d in range(2):
                        scores[d] += abs(grad[d]) / norm1

    return scores, has_disc


@nb_jit(nopython=True, cache=True)
def score_estimate_3d(
    coeffs_list: NumbaList,
    masks_list: NumbaList,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Estimate scores for each height direction in 3D.

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
        for i0 in range(M):
            for i1 in range(M):
                for i2 in range(M):
                    if not mask[i0, i1, i2]:
                        continue
                    x[0] = (i0 + 0.5) * inv_m
                    x[1] = (i1 + 0.5) * inv_m
                    x[2] = (i2 + 0.5) * inv_m
                    grad = _eval_gradient_3d(coeffs, x)
                    norm1 = abs(grad[0]) + abs(grad[1]) + abs(grad[2])
                    if norm1 > 1e-300:  # noqa: PLR2004
                        for d in range(3):
                            scores[d] += abs(grad[d]) / norm1

    return scores, has_disc
