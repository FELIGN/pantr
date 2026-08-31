"""Bézier restriction to a sub-interval via de Casteljau.

This module provides :func:`_restrict_bezier`, which restricts a Bézier to a
sub-region of ``[0, 1]^dim`` and reparametrizes the result back to
``[0, 1]^dim``.

The algorithm uses two de Casteljau passes per direction (with numerically
stable pass ordering), avoiding the previous Bézier → B-spline → restrict →
Bézier round-trip.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .._array_utils import _flatten_along_axis, _unflatten_along_axis
from ._bezier_backend import restrict_kernel, restrict_nd_kernel

if TYPE_CHECKING:
    from . import Bezier


def _restrict_bezier(
    bezier: Bezier,
    bounds_per_dim: list[tuple[float, float] | None],
) -> Bezier:
    """Restrict a Bézier to a sub-region of ``[0, 1]^dim``.

    Applies the backend's restriction kernel per parametric direction via the
    shared :func:`_flatten_along_axis` / :func:`_unflatten_along_axis` helpers.
    Directions with ``None`` bounds are left unchanged.

    Args:
        bezier (~pantr.bezier.Bezier): Input Bézier.
        bounds_per_dim: Per-direction bounds as ``(lower, upper)`` or ``None``
            to skip. Must have length ``dim``.

    Returns:
        ~pantr.bezier.Bezier: New Bézier on ``[0, 1]^dim`` representing the
        restriction.

    Raises:
        ValueError: If every direction is ``None`` or matches the full
            ``[0, 1]`` domain.
    """
    lower = [0.0 if bounds is None else float(bounds[0]) for bounds in bounds_per_dim]
    upper = [1.0 if bounds is None else float(bounds[1]) for bounds in bounds_per_dim]

    # Checked above the backend branch, so both raise the same message for the same
    # argument. A direction whose bounds are the full domain is skipped rather than
    # restricted, which is not an optimisation: the two-pass restriction over [0, 1]
    # commits roundings that leaving the direction alone does not.
    if not any(lo != 0.0 or up != 1.0 for lo, up in zip(lower, upper, strict=True)):
        raise ValueError("Bounds match the full domain; at least one direction must be restricted.")

    return restrict_nd_kernel()(bezier, lower, upper)


def _restrict_python(
    bezier: Bezier,
    lower: Sequence[float],
    upper: Sequence[float],
) -> Bezier:
    """Restrict with NumPy and the Numba kernel: the oracle for the port.

    Args:
        bezier (~pantr.bezier.Bezier): The Bézier to restrict.
        lower (Sequence[float]): Lower bound per parametric direction.
        upper (Sequence[float]): Upper bound per parametric direction.

    Returns:
        ~pantr.bezier.Bezier: The restricted Bézier.

    Note:
        No input validation is performed here; Layer 2 did it above the branch.
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    ctrl = bezier.control_points
    restrict = restrict_kernel()

    for i, (low, high) in enumerate(zip(lower, upper, strict=True)):
        if low == 0.0 and high == 1.0:
            continue

        pts_2d, trailing_shape = _flatten_along_axis(ctrl, i)
        out = np.empty_like(pts_2d)
        restrict(pts_2d, low, high, out)
        ctrl = _unflatten_along_axis(out, trailing_shape, i)

    return BezierCls(ctrl, is_rational=bezier.is_rational)
