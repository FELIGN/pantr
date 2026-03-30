"""Public API for Bernstein polynomial root finding.

This module contains the Layer 1 public functions for finding roots of
scalar Bernstein polynomials on [0, 1]. Each function performs lightweight
validation and delegates to Layer 2 implementations in :mod:`_find_roots`.

- :func:`find_roots` -- find all roots (single or batch, auto-dispatch).
- :func:`find_monotone_root` -- fast solver for monotone Beziers (single or batch).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, overload

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


@overload
def find_roots(
    bezier: Bezier,
    *,
    tol: float | None = ...,
) -> npt.NDArray[np.float64]: ...


@overload
def find_roots(
    bezier: Sequence[Bezier],
    *,
    tol: float | None = ...,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]: ...


def find_roots(
    bezier: Bezier | Sequence[Bezier],
    *,
    tol: float | None = None,
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
    """Find all roots of one or more scalar Bezier curves in [0, 1].

    Auto-selects between Yuksel's monotone-decomposition algorithm and Bezier
    clipping based on polynomial degree and coefficient dynamic range.

    When called with a single :class:`Bezier`, returns a sorted array of root
    parameters. When called with a sequence of Beziers (batch mode), all curves
    must have the same degree and the function returns a ``(roots, counts)``
    tuple.

    Args:
        bezier (Bezier | Sequence[Bezier]): A single 1-D (``dim == 1``) scalar
            (``rank == 1``) Bezier curve, or a sequence of such curves (all with
            the same degree). For rational Beziers, roots are found on the
            numerator polynomial (first homogeneous component).
        tol (float | None): Root-finding tolerance (bracket-width
            termination). Defaults to ``tolerance.get_strict(bezier.dtype)``.

    Returns:
        npt.NDArray[np.float64]: *(single mode)* Sorted array of root
            parameters in [0, 1]. Empty if no roots exist. Always float64
            regardless of input dtype.
        tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]: *(batch mode)*
            - ``roots``: padded array of shape ``(n_polys, degree)`` where
              only the first ``counts[i]`` entries per row are valid.
              Always float64.
            - ``counts``: 1-D array of shape ``(n_polys,)`` with the number
              of valid roots per polynomial.

    Raises:
        TypeError: If ``bezier`` is not a :class:`Bezier` instance (or sequence
            thereof).
        ValueError: If any Bezier has ``dim != 1`` or ``rank != 1``, or ``tol``
            is not positive. In batch mode, also raised if degrees are not
            uniform.

    Note:
        In batch mode, a simpler fixed-threshold dedup (no derivative-aware
        merge radius) is used compared to single mode. In rare edge cases
        involving near-duplicate roots, the two modes may report different
        root counts.

    Example:
        >>> import numpy as np
        >>> from pantr.bezier import Bezier, find_roots
        >>> find_roots(Bezier([1.0, -1.0]))
        array([0.5])
    """
    from pantr.bezier._bezier import Bezier as BezierCls  # noqa: PLC0415

    if isinstance(bezier, BezierCls):
        return _find_roots_impl(bezier, tol=tol)
    return _find_roots_batch_impl(bezier, tol=tol)


@overload
def find_monotone_root(
    bezier: Bezier,
    *,
    tol: float | None = ...,
) -> float: ...


@overload
def find_monotone_root(
    bezier: Sequence[Bezier],
    *,
    tol: float | None = ...,
) -> npt.NDArray[np.float64]: ...


def find_monotone_root(
    bezier: Bezier | Sequence[Bezier],
    *,
    tol: float | None = None,
) -> float | npt.NDArray[np.float64]:
    """Find the unique root of one or more monotone scalar Bezier curves in [0, 1].

    Uses a Newton/bisection hybrid with false-position initialization. Each
    Bezier must be monotone on [0, 1] (i.e. its derivative does not change
    sign).

    When called with a single :class:`Bezier`, returns a float. When called
    with a sequence of Beziers (batch mode), all curves must have the same
    degree and the function returns an array.

    Args:
        bezier (Bezier | Sequence[Bezier]): A single 1-D (``dim == 1``) scalar
            (``rank == 1``) Bezier curve, or a sequence of such curves (all with
            the same degree). For rational Beziers, roots are found on the
            numerator polynomial.
        tol (float | None): Parameter-space termination tolerance. Defaults
            to ``tolerance.get_strict(bezier.dtype)``.

    Returns:
        float: *(single mode)* Root parameter in [0, 1], or ``NaN`` if no root
            exists (no sign change across the interval).
        npt.NDArray[np.float64]: *(batch mode)* 1-D array of shape
            ``(n_polys,)`` with root values. Contains ``NaN`` where no root
            exists. Always float64.

    Raises:
        TypeError: If ``bezier`` is not a :class:`Bezier` instance (or sequence
            thereof).
        ValueError: If any Bezier has ``dim != 1`` or ``rank != 1``, or ``tol``
            is not positive. In batch mode, also raised if degrees are not
            uniform.
    """
    from pantr.bezier._bezier import Bezier as BezierCls  # noqa: PLC0415

    if isinstance(bezier, BezierCls):
        return _solve_monotone_root_impl(bezier, tol=tol)
    return _solve_monotone_root_batch_impl(bezier, tol=tol)
