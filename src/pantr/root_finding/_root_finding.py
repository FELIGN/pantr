"""Public API for Bernstein polynomial root finding.

This module contains the Layer 1 public functions for finding roots of
scalar Bernstein polynomials on [0, 1]. Each function performs lightweight
validation and delegates to Layer 2 implementations in :mod:`_find_roots`.

- :func:`find_roots` -- find all roots (single polynomial, auto-dispatch).
- :func:`find_roots_batch` -- find roots of many same-degree polynomials.
- :func:`solve_monotone_root` -- fast solver for a single monotone polynomial.
- :func:`solve_monotone_root_batch` -- batch-parallel monotone solver.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from pantr.root_finding._find_roots import (
    _find_roots_batch_impl,
    _find_roots_impl,
    _solve_monotone_root_batch_impl,
    _solve_monotone_root_impl,
)


def find_roots(
    coeff: npt.NDArray[np.float32 | np.float64],
    *,
    tol: float | None = None,
) -> npt.NDArray[np.float64]:
    """Find all roots of a Bernstein polynomial in [0, 1].

    Auto-selects between Yuksel's monotone-decomposition algorithm and Bezier
    clipping based on polynomial degree and coefficient dynamic range.

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): Bernstein coefficients
            of the scalar polynomial. Must be a 1-D array with at least 1
            element. Both float32 and float64 are accepted; float32 is not
            recommended for polynomials of degree > 5 due to limited
            significand precision.
        tol (float | None): Root-finding tolerance (bracket-width
            termination). Defaults to ``tolerance.get_strict(coeff.dtype)``.

    Returns:
        npt.NDArray[np.float64]: Sorted array of root parameters in [0, 1].
            Empty if no roots exist. Always float64 regardless of input dtype.

    Raises:
        TypeError: If ``coeff`` is not a float32 or float64 ndarray.
        ValueError: If ``coeff`` is not 1-D, is empty, or ``tol`` is not
            positive.

    Example:
        >>> import numpy as np
        >>> from pantr.root_finding import find_roots
        >>> find_roots(np.array([1.0, -1.0]))
        array([0.5])
    """
    return _find_roots_impl(coeff, tol=tol)


def find_roots_batch(
    coeffs: npt.NDArray[np.float32 | np.float64],
    *,
    tol: float | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
    """Find roots of multiple same-degree Bernstein polynomials in parallel.

    All polynomials in the batch must have the same degree.

    Args:
        coeffs (npt.NDArray[np.float32 | np.float64]): Batch of Bernstein
            coefficients with shape ``(n_polys, degree + 1)``. Both float32
            and float64 are accepted; float32 is not recommended for
            polynomials of degree > 5.
        tol (float | None): Root-finding tolerance. Defaults to
            ``tolerance.get_strict(coeffs.dtype)``.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
            - ``roots``: padded array of shape ``(n_polys, degree)`` where
              only the first ``counts[i]`` entries per row are valid.
              Always float64.
            - ``counts``: 1-D array of shape ``(n_polys,)`` with the number
              of valid roots per polynomial.

    Raises:
        TypeError: If ``coeffs`` is not a float32 or float64 ndarray.
        ValueError: If ``coeffs`` is not 2-D, has fewer than 1 column, or
            ``tol`` is not positive.
    """
    return _find_roots_batch_impl(coeffs, tol=tol)


def solve_monotone_root(
    coeff: npt.NDArray[np.float32 | np.float64],
    *,
    tol: float | None = None,
) -> float:
    """Find the unique root of a monotone Bernstein polynomial in [0, 1].

    Uses a Newton/bisection hybrid with false-position initialization. The
    polynomial must be monotone on [0, 1] (i.e. its derivative does not
    change sign).

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): Bernstein coefficients
            of the monotone scalar polynomial. Must be a 1-D array. Both
            float32 and float64 are accepted; float32 is not recommended for
            polynomials of degree > 5.
        tol (float | None): Parameter-space termination tolerance. Defaults
            to ``tolerance.get_strict(coeff.dtype)``.

    Returns:
        float: Root parameter in [0, 1], or ``NaN`` if no root exists (no
            sign change across the interval).

    Raises:
        TypeError: If ``coeff`` is not a float32 or float64 ndarray.
        ValueError: If ``coeff`` is not 1-D, is empty, or ``tol`` is not
            positive.
    """
    return _solve_monotone_root_impl(coeff, tol=tol)


def solve_monotone_root_batch(
    coeffs: npt.NDArray[np.float32 | np.float64],
    *,
    tol: float | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Solve for roots on multiple monotone Bernstein polynomials in parallel.

    Each polynomial must be monotone on [0, 1]. The batch must have uniform
    degree.

    Args:
        coeffs (npt.NDArray[np.float32 | np.float64]): Batch of Bernstein
            coefficients with shape ``(n_polys, degree + 1)``. Both float32
            and float64 are accepted; float32 is not recommended for
            polynomials of degree > 5.
        tol (float | None): Parameter-space termination tolerance. Defaults
            to ``tolerance.get_strict(coeffs.dtype)``.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
            - ``roots``: 1-D array of shape ``(n_polys,)`` with root values.
              Contains ``NaN`` where no root exists. Always float64.
            - ``found``: boolean mask of shape ``(n_polys,)`` indicating
              which polynomials had a root.

    Raises:
        TypeError: If ``coeffs`` is not a float32 or float64 ndarray.
        ValueError: If ``coeffs`` is not 2-D, has fewer than 1 column, or
            ``tol`` is not positive.
    """
    return _solve_monotone_root_batch_impl(coeffs, tol=tol)
