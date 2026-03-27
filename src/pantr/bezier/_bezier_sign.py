"""Uniform sign detection for scalar Bézier polynomials.

Provides :func:`_uniform_sign`, which determines whether a scalar Bézier
polynomial has a definite sign over its domain.

For non-rational Béziers, this checks that all Bernstein coefficients share
the same strict sign.  For rational Béziers, both the numerator and
denominator (weight) coefficients must independently have uniform strict
sign; the sign of the rational function is then the product of the two signs.

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


def _coeff_sign(coeffs: npt.NDArray[np.floating[Any]]) -> UniformSign:
    """Return the uniform sign of a flat coefficient array.

    Args:
        coeffs (npt.NDArray[np.floating]): Flat coefficient array.

    Returns:
        UniformSign: ``POSITIVE`` if all > 0, ``NEGATIVE`` if all < 0,
        ``MIXED`` otherwise.
    """
    if np.all(coeffs > 0):
        return UniformSign.POSITIVE
    if np.all(coeffs < 0):
        return UniformSign.NEGATIVE
    return UniformSign.MIXED


def _uniform_sign(bezier: Bezier) -> UniformSign:
    """Check whether a scalar Bézier has uniform sign.

    For non-rational Béziers, checks that all Bernstein coefficients share
    the same strict sign.  For rational Béziers, both the numerator
    (``w_i * c_i``) and denominator (``w_i``) coefficients must independently
    have uniform strict sign; the sign of the rational function is the
    product of the two.

    Args:
        bezier (~pantr.bezier.Bezier): A scalar (``rank == 1``) Bézier of
            any parametric dimension, rational or non-rational.

    Returns:
        UniformSign: The uniform sign of the polynomial.

    Raises:
        ValueError: If ``bezier`` is not scalar (``rank != 1``).
    """
    if bezier.rank != 1:
        raise ValueError(
            f"uniform_sign requires a scalar Bézier (rank == 1), got rank {bezier.rank}."
        )

    if bezier.is_rational:
        weights = bezier.control_points[..., -1].ravel()
        sign_w = _coeff_sign(weights)
        if sign_w is UniformSign.MIXED:
            return UniformSign.MIXED
        # _extract_scalar_coeffs won't raise: rank==1 checked, weights same-sign.
        sign_n = _coeff_sign(_extract_scalar_coeffs(bezier).ravel())
        if sign_n is UniformSign.MIXED:
            return UniformSign.MIXED
        # sign(n/w) = sign(n) * sign(w)
        return UniformSign(sign_n * sign_w)

    return _coeff_sign(_extract_scalar_coeffs(bezier).ravel())


def _extract_scalar_coeffs(
    bezier: Bezier,
) -> npt.NDArray[np.float64]:
    """Extract scalar Bernstein coefficients from a Bezier as a contiguous N-D float64 array.

    For non-rational Beziers, returns the scalar component ``control_points[..., 0]``.

    For rational Beziers with all weights sharing the same strict sign (all
    positive or all negative), returns the numerator coefficients
    ``control_points[..., 0]`` (which are ``w_i * c_i`` in homogeneous form).
    The zeros of the rational function coincide with the zeros of the
    numerator when all weights have the same sign.

    Args:
        bezier (~pantr.bezier.Bezier): A scalar Bezier (``rank == 1``).
            Rational Beziers are accepted when all weights share the same
            strict sign.

    Returns:
        npt.NDArray[np.float64]: Contiguous coefficient array of shape
        ``(p0+1, p1+1, ...)``.

    Raises:
        TypeError: If ``bezier`` is rational and the weights do not all share
            the same strict sign (i.e., some are positive and some negative,
            or any are zero).
        ValueError: If ``bezier`` is not scalar (``rank != 1``).
    """
    if bezier.rank != 1:
        raise ValueError(
            f"_extract_scalar_coeffs requires a scalar Bézier (rank == 1), got rank {bezier.rank}."
        )

    if bezier.is_rational:
        weights = bezier.control_points[..., -1]
        if _coeff_sign(weights.ravel()) is UniformSign.MIXED:
            raise TypeError(
                "Rational Bézier operations require all weights to share "
                "the same strict sign (all positive or all negative)."
            )

    coeffs: npt.NDArray[np.float64] = np.ascontiguousarray(
        bezier.control_points[..., 0], dtype=np.float64
    )
    return coeffs
