"""Bézier restriction to a sub-interval via de Casteljau.

This module provides :func:`_restrict_bezier`, which restricts a Bézier to a
sub-region of ``[0, 1]^dim`` and reparametrizes the result back to
``[0, 1]^dim``.

The algorithm uses two de Casteljau passes per direction (with numerically
stable pass ordering), avoiding the previous Bézier → B-spline → restrict →
Bézier round-trip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ._bezier_core import _restrict_bezier_1d_core

if TYPE_CHECKING:
    from . import Bezier


def _restrict_bezier(
    bezier: Bezier,
    bounds_per_dim: list[tuple[float, float] | None],
) -> Bezier:
    """Restrict a Bézier to a sub-region of ``[0, 1]^dim``.

    Applies :func:`_restrict_bezier_1d_core` per parametric direction using the
    standard moveaxis pattern. Directions with ``None`` bounds are left
    unchanged.

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
    from . import Bezier as BezierCls  # noqa: PLC0415

    ctrl = bezier.control_points
    any_restricted = False

    for i, bounds in enumerate(bounds_per_dim):
        if bounds is None:
            continue

        lower, upper = bounds

        # Skip full-domain bounds.
        if lower == 0.0 and upper == 1.0:
            continue

        any_restricted = True

        # Move target axis to position 0, flatten the rest.
        moved = np.moveaxis(ctrl, i, 0)
        orig_shape = moved.shape
        pts_2d: npt.NDArray[np.floating[Any]] = moved.reshape(orig_shape[0], -1)

        pts_2d = np.ascontiguousarray(pts_2d)

        out = np.empty_like(pts_2d)
        _restrict_bezier_1d_core(pts_2d, lower, upper, out)

        # Restore shape and move axis back.
        ctrl = np.moveaxis(out.reshape(orig_shape), 0, i)

    if not any_restricted:
        raise ValueError("Bounds match the full domain; at least one direction must be restricted.")

    return BezierCls(ctrl, is_rational=bezier.is_rational)
