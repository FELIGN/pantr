"""Uniform sign detection for scalar non-rational Bézier polynomials.

Provides :func:`_uniform_sign`, which determines whether all Bernstein
coefficients of a scalar Bézier share the same strict sign.  By the convex
hull property of Bernstein polynomials, a uniform sign on the coefficients
guarantees the polynomial does not change sign over its domain.

This is the ``uniformSign`` utility from the algoim library
(R. I. Saye, *J. Comput. Phys.* 448, 110720, 2022).
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from . import Bezier


class UniformSign(enum.IntEnum):
    """Result of a uniform sign test on Bernstein coefficients.

    Attributes:
        NEGATIVE: All coefficients are strictly negative.
        MIXED: Coefficients have mixed signs or at least one is zero.
        POSITIVE: All coefficients are strictly positive.
    """

    NEGATIVE = -1
    """All coefficients are strictly negative."""

    MIXED = 0
    """Coefficients have mixed signs or at least one is zero."""

    POSITIVE = 1
    """All coefficients are strictly positive."""


def _uniform_sign(bezier: Bezier) -> UniformSign:
    """Check whether a scalar non-rational Bézier has uniform sign.

    Uses the convex hull property of Bernstein polynomials: if all coefficients
    share the same strict sign, the polynomial cannot change sign over its
    domain.

    Args:
        bezier (~pantr.bezier.Bezier): A scalar (``rank == 1``),
            non-rational Bézier of any parametric dimension.

    Returns:
        UniformSign: The uniform sign of the Bernstein coefficients.

    Raises:
        TypeError: If ``bezier`` is rational.
        ValueError: If ``bezier`` is not scalar (``rank != 1``).
    """
    if bezier.is_rational:
        raise TypeError("uniform_sign is only defined for non-rational Bézier polynomials.")
    if bezier.rank != 1:
        raise ValueError(
            f"uniform_sign requires a scalar Bézier (rank == 1), got rank {bezier.rank}."
        )

    coeffs: npt.NDArray[np.floating[Any]] = bezier.control_points[..., 0].ravel()
    # Ensure contiguous for the kernel (int arrays are already cast in Bezier constructor).
    coeffs = np.ascontiguousarray(coeffs, dtype=bezier.dtype)

    if np.all(coeffs > 0):
        return UniformSign.POSITIVE
    if np.all(coeffs < 0):
        return UniformSign.NEGATIVE
    return UniformSign.MIXED
