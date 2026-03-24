"""Uniform sign detection for scalar non-rational Bézier polynomials.

Provides :func:`_uniform_sign`, which determines whether all Bernstein
coefficients of a scalar Bézier share the same strict sign.  By the convex
hull property of Bernstein polynomials, a uniform sign on the coefficients
guarantees the polynomial does not change sign over its domain.

This is the ``uniformSign`` utility from the algoim library
(R. I. Saye, *J. Comput. Phys.* 448, 110720, 2022).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ._bezier_core import _uniform_sign_core

if TYPE_CHECKING:
    from . import Bezier


def _uniform_sign(bezier: Bezier) -> int:
    """Check whether a scalar non-rational Bézier has uniform sign.

    Returns ``+1`` if every Bernstein coefficient is strictly positive,
    ``-1`` if every coefficient is strictly negative, and ``0`` otherwise
    (mixed signs or at least one zero coefficient).

    Args:
        bezier (~pantr.bezier.Bezier): A scalar (``rank == 1``),
            non-rational Bézier of any parametric dimension.

    Returns:
        int: ``+1``, ``-1``, or ``0``.

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

    return int(_uniform_sign_core(coeffs))
