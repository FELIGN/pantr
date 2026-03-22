"""Shared Layer 2 helper for applying affine transforms to control points.

Handles both non-rational and rational (NURBS / rational Bézier) control
point arrays.  Used by :meth:`Bspline.transform` and :meth:`Bezier.transform`.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt


def _apply_affine_to_control_points(
    control_points: npt.NDArray[np.float32 | np.float64],
    is_rational: bool,
    matrix: npt.NDArray[np.float64],
    translation: npt.NDArray[np.float64],
    *,
    in_place: bool = False,
) -> npt.NDArray[np.float32 | np.float64]:
    """Apply an affine map ``T(x) = A x + b`` to every control point.

    For non-rational geometry the last axis is the full coordinate vector and
    the transform is applied directly.  For rational geometry the stored
    control points are in weighted homogeneous form ``(w·x, w)``; the linear
    part operates on the weighted coordinates and the translation is scaled
    by each point's weight.

    Args:
        control_points (npt.NDArray[np.float32 | np.float64]): Control point
            array with shape ``(..., rank_full)`` where *rank_full* includes
            the weight column for rational geometry.
        is_rational (bool): Whether the geometry is rational.
        matrix (npt.NDArray[np.float64]): ``(n, n)`` linear part, where
            ``n`` equals the geometric rank (excluding weight).
        translation (npt.NDArray[np.float64]): ``(n,)`` translation vector.
        in_place (bool): If ``True``, write the result directly into
            *control_points* and return it.  If ``False`` (default), allocate
            and return a new array.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Transformed control point
        array with the same shape and dtype as *control_points*.  When
        *in_place* is ``True`` this is the same object as *control_points*.

    Raises:
        ValueError: If the transform dimension does not match the geometric
            rank of the control points.
    """
    cp = control_points
    dtype = cp.dtype
    rank_full = cp.shape[-1]

    n = rank_full - 1 if is_rational else rank_full

    if matrix.shape != (n, n):
        raise ValueError(
            f"Transform dimension ({matrix.shape[0]}) does not match the "
            f"geometric rank ({n}) of the control points."
        )

    # Cast matrix and translation to control-point dtype for computation.
    A = matrix.astype(dtype, copy=False)  # noqa: N806
    b = translation.astype(dtype, copy=False)

    if is_rational:
        out = cp if in_place else cp.copy()
        # Weighted coordinates: cp[..., :n] = w * x_i
        # After T(x) = A x + b  →  w * (A x + b) = A (w x) + w b
        out[..., :n] = cp[..., :n] @ A.T + cp[..., n : n + 1] * b
        # Weight out[..., n] stays unchanged.
        return out

    if in_place:
        cp[...] = cp @ A.T + b
        return cp
    return (cp @ A.T + b).astype(dtype, copy=False)
