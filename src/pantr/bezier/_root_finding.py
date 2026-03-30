"""Public API for Bernstein polynomial root finding.

This module contains the Layer 1 public functions for finding roots of
scalar Bernstein polynomials on [0, 1]. Each function performs lightweight
validation and delegates to Layer 2 implementations in :mod:`_find_roots`.

- :func:`find_roots` -- find all roots (single Bezier, auto-dispatch).
- :func:`find_roots_batch` -- find roots of many same-degree Beziers.
- :func:`solve_monotone_root` -- fast solver for a single monotone Bezier.
- :func:`solve_monotone_root_batch` -- batch-parallel monotone solver.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

from pantr.bezier._find_roots import (
    _find_roots_batch_impl,
    _find_roots_impl,
    _solve_monotone_root_batch_impl,
    _solve_monotone_root_impl,
)

if TYPE_CHECKING:
    from pantr.bezier._bezier import Bezier


def find_roots(
    bezier: Bezier,
    *,
    tol: float | None = None,
) -> npt.NDArray[np.float64]:
    """Find all roots of a scalar Bezier curve in [0, 1].

    Auto-selects between Yuksel's monotone-decomposition algorithm and Bezier
    clipping based on polynomial degree and coefficient dynamic range.

    Args:
        bezier (Bezier): A 1-D (``dim == 1``) scalar (``rank == 1``) Bezier
            curve. For rational Beziers, roots are found on the numerator
            polynomial (first homogeneous component).
        tol (float | None): Root-finding tolerance (bracket-width
            termination). Defaults to ``tolerance.get_strict(bezier.dtype)``.

    Returns:
        npt.NDArray[np.float64]: Sorted array of root parameters in [0, 1].
            Empty if no roots exist. Always float64 regardless of input dtype.

    Raises:
        TypeError: If ``bezier`` is not a :class:`Bezier` instance.
        ValueError: If ``bezier.dim != 1``, ``bezier.rank != 1``, or ``tol``
            is not positive.

    Example:
        >>> import numpy as np
        >>> from pantr.bezier import Bezier, find_roots
        >>> find_roots(Bezier([1.0, -1.0]))
        array([0.5])
    """
    return _find_roots_impl(bezier, tol=tol)


def find_roots_batch(
    beziers: Sequence[Bezier],
    *,
    tol: float | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
    """Find roots of multiple same-degree scalar Bezier curves in parallel.

    All Beziers in the batch must have the same degree.

    Args:
        beziers (Sequence[Bezier]): Sequence of 1-D (``dim == 1``) scalar
            (``rank == 1``) Bezier curves, all with the same degree. For
            rational Beziers, roots are found on the numerator polynomial.
        tol (float | None): Root-finding tolerance. Defaults to
            ``tolerance.get_strict(dtype)`` of the first Bezier.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
            - ``roots``: padded array of shape ``(n_polys, degree)`` where
              only the first ``counts[i]`` entries per row are valid.
              Always float64.
            - ``counts``: 1-D array of shape ``(n_polys,)`` with the number
              of valid roots per polynomial.

    Raises:
        TypeError: If any element is not a :class:`Bezier` instance.
        ValueError: If any Bezier has ``dim != 1`` or ``rank != 1``, or if
            degrees are not uniform, or ``tol`` is not positive.

    Note:
        The batch path uses a simpler fixed-threshold dedup (no
        derivative-aware merge radius) compared to :func:`find_roots`.
        In rare edge cases involving near-duplicate roots, the two
        functions may report different root counts.
    """
    return _find_roots_batch_impl(beziers, tol=tol)


def solve_monotone_root(
    bezier: Bezier,
    *,
    tol: float | None = None,
) -> float:
    """Find the unique root of a monotone scalar Bezier curve in [0, 1].

    Uses a Newton/bisection hybrid with false-position initialization. The
    Bezier must be monotone on [0, 1] (i.e. its derivative does not
    change sign).

    Args:
        bezier (Bezier): A 1-D (``dim == 1``) scalar (``rank == 1``) Bezier
            curve. For rational Beziers, roots are found on the numerator
            polynomial.
        tol (float | None): Parameter-space termination tolerance. Defaults
            to ``tolerance.get_strict(bezier.dtype)``.

    Returns:
        float: Root parameter in [0, 1], or ``NaN`` if no root exists (no
            sign change across the interval).

    Raises:
        TypeError: If ``bezier`` is not a :class:`Bezier` instance.
        ValueError: If ``bezier.dim != 1``, ``bezier.rank != 1``, or ``tol``
            is not positive.
    """
    return _solve_monotone_root_impl(bezier, tol=tol)


def solve_monotone_root_batch(
    beziers: Sequence[Bezier],
    *,
    tol: float | None = None,
) -> npt.NDArray[np.float64]:
    """Solve for roots on multiple monotone scalar Bezier curves in parallel.

    Each Bezier must be monotone on [0, 1]. The batch must have uniform
    degree.

    Args:
        beziers (Sequence[Bezier]): Sequence of 1-D (``dim == 1``) scalar
            (``rank == 1``) Bezier curves, all with the same degree. For
            rational Beziers, roots are found on the numerator polynomial.
        tol (float | None): Parameter-space termination tolerance. Defaults
            to ``tolerance.get_strict(dtype)`` of the first Bezier.

    Returns:
        npt.NDArray[np.float64]: 1-D array of shape ``(n_polys,)`` with root
            values. Contains ``NaN`` where no root exists. Always float64.

    Raises:
        TypeError: If any element is not a :class:`Bezier` instance.
        ValueError: If any Bezier has ``dim != 1`` or ``rank != 1``, or if
            degrees are not uniform, or ``tol`` is not positive.
    """
    return _solve_monotone_root_batch_impl(beziers, tol=tol)
