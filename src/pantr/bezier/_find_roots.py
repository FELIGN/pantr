"""Layer 2 implementation for Bernstein polynomial root finding.

Handles input validation, tolerance resolution, algorithm dispatch, and output
formatting. Delegates all computation to Layer 3 Numba kernels in
:mod:`_yuksel_core`, :mod:`_clipping_core`, and :mod:`_batch_core`.

This module is not part of the public API. Users should call the functions
exported from :mod:`pantr.bezier`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy import typing as npt

from pantr._numba_compat import wait_for_jit_warmup
from pantr.bezier._bezier import Bezier
from pantr.bezier._clipping_core import _clip_roots_core, _dedup_roots
from pantr.bezier._yuksel_core import (
    _solve_monotone_root_kernel,
    _yuksel_roots,
)
from pantr.tolerance import get_strict

_CLIP_MIN_DEGREE: int = 6
"""Minimum polynomial degree for which Bezier clipping is considered.

For degree <= 5 the Yuksel derivative chain is at most 4 levels deep and has
negligible rounding accumulation, while its per-call overhead is lower than
clipping's convex hull construction and subdivision.

Duplicated in ``_batch_core.py`` (Numba kernels cannot import Python-level
module constants at call time) -- keep in sync.
"""

_CLIP_COEFF_RANGE_LIMIT: float = 1e8
"""Maximum coefficient dynamic range for which Bezier clipping is used.

Empirically determined via a 27 000-trial sweep comparing clipping against
Yuksel across coefficient ranges from 1e0 to 1e20.

Duplicated in ``_batch_core.py`` -- keep in sync.
"""

_SUPPORTED_DTYPES = (np.float32, np.float64)
"""Floating-point dtypes accepted by the root-finding API."""


def _extract_coeff(bezier: Bezier) -> npt.NDArray[np.float32 | np.float64]:
    """Extract a 1-D coefficient array from a scalar Bezier curve.

    For non-rational Beziers, returns ``control_points[:, 0]``. For rational
    Beziers, returns the first homogeneous component (numerator ``x * w``),
    whose roots coincide with those of the rational function ``x`` since
    weights are positive.

    Args:
        bezier (Bezier): A validated 1-D scalar Bezier curve.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Contiguous 1-D coefficient
            array of length ``degree + 1``.
    """
    coeff = bezier.control_points[:, 0]
    if not coeff.flags.c_contiguous:
        return np.ascontiguousarray(coeff)
    return coeff


def _validate_bezier_for_roots(bezier: object) -> Bezier:
    """Validate that an object is a scalar 1-D Bezier suitable for root finding.

    Args:
        bezier (object): Input to validate.

    Returns:
        Bezier: The validated Bezier.

    Raises:
        TypeError: If ``bezier`` is not a :class:`Bezier`.
        ValueError: If ``bezier.dim != 1`` or ``bezier.rank != 1``.
    """
    if not isinstance(bezier, Bezier):
        msg = f"Expected a Bezier instance, got {type(bezier).__name__}"
        raise TypeError(msg)
    if bezier.dim != 1:
        msg = f"Bezier must be 1-D (dim == 1), got dim={bezier.dim}"
        raise ValueError(msg)
    if bezier.rank != 1:
        msg = f"Bezier must be scalar (rank == 1), got rank={bezier.rank}"
        raise ValueError(msg)
    return bezier


def _extract_batch_coeffs(
    beziers: Sequence[Bezier],
) -> npt.NDArray[np.float32 | np.float64]:
    """Validate a sequence of Beziers and assemble a 2-D coefficient array.

    All Beziers must be 1-D, scalar, and have the same degree. The returned
    array has shape ``(n_polys, degree + 1)`` with coefficients copied from
    each Bezier's control points.

    Args:
        beziers (Sequence[Bezier]): Sequence of Bezier curves.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Contiguous 2-D coefficient
            array of shape ``(n_polys, degree + 1)``.

    Raises:
        TypeError: If any element is not a :class:`Bezier`.
        ValueError: If any Bezier has ``dim != 1`` or ``rank != 1``, or if
            degrees are not uniform.
    """
    n_polys = len(beziers)
    if n_polys == 0:
        return np.empty((0, 1), dtype=np.float64)

    first = _validate_bezier_for_roots(beziers[0])
    degree = first.degree[0]
    dtype = first.dtype

    coeffs = np.empty((n_polys, degree + 1), dtype=dtype)
    coeffs[0] = _extract_coeff(first)

    for i in range(1, n_polys):
        bez = _validate_bezier_for_roots(beziers[i])
        if bez.degree[0] != degree:
            msg = (
                f"All Beziers must have the same degree. "
                f"Bezier 0 has degree {degree}, Bezier {i} has degree {bez.degree[0]}"
            )
            raise ValueError(msg)
        coeffs[i] = _extract_coeff(bez)

    return coeffs


def _resolve_tol(
    coeff: npt.NDArray[np.float32 | np.float64],
    tol: float | None,
) -> float:
    """Resolve tolerance from user input or dtype default.

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): Coefficient array
            (used for dtype).
        tol (float | None): User-provided tolerance, or ``None`` for default.

    Returns:
        float: Resolved tolerance value.

    Raises:
        ValueError: If ``tol`` is not positive.
    """
    if tol is None:
        return get_strict(coeff.dtype)
    if tol <= 0.0:
        msg = f"tol must be positive, got {tol}"
        raise ValueError(msg)
    return tol


def _validate_coeff_1d(
    coeff: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Validate a single coefficient array.

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): Bernstein coefficients.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Contiguous 1-D array.

    Raises:
        TypeError: If ``coeff`` is not a float32/float64 ndarray.
        ValueError: If the array has fewer than 1 element or is not 1-D.
    """
    if not isinstance(coeff, np.ndarray) or coeff.dtype not in _SUPPORTED_DTYPES:
        dtype = getattr(coeff, "dtype", "N/A")
        msg = (
            f"coeff must be a float32 or float64 ndarray, "
            f"got {type(coeff).__name__} with dtype {dtype}"
        )
        raise TypeError(msg)
    if coeff.ndim != 1:
        msg = f"coeff must be 1-D, got shape {coeff.shape}"
        raise ValueError(msg)
    if coeff.size < 1:
        msg = "coeff must have at least 1 element"
        raise ValueError(msg)
    if not coeff.flags.c_contiguous:
        return np.ascontiguousarray(coeff)
    return coeff


def _validate_coeffs_batch(
    coeffs: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Validate a batch of coefficient arrays.

    Args:
        coeffs (npt.NDArray[np.float32 | np.float64]): Batch of Bernstein
            coefficients, shape ``(n_polys, degree + 1)``.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Contiguous 2-D array.

    Raises:
        TypeError: If ``coeffs`` is not a float32/float64 ndarray.
        ValueError: If the array is not 2-D or has fewer than 1 column.
    """
    if not isinstance(coeffs, np.ndarray) or coeffs.dtype not in _SUPPORTED_DTYPES:
        dtype = getattr(coeffs, "dtype", "N/A")
        msg = (
            f"coeffs must be a float32 or float64 ndarray, "
            f"got {type(coeffs).__name__} with dtype {dtype}"
        )
        raise TypeError(msg)
    if coeffs.ndim != 2:  # noqa: PLR2004
        msg = f"coeffs must be 2-D, got shape {coeffs.shape}"
        raise ValueError(msg)
    if coeffs.shape[1] < 1:
        msg = "coeffs must have at least 1 column (degree + 1)"
        raise ValueError(msg)
    if not coeffs.flags.c_contiguous:
        return np.ascontiguousarray(coeffs)
    return coeffs


def _dispatch_single(  # noqa: PLR0911
    coeff: npt.NDArray[np.float32 | np.float64],
    param_tol: float,
    geom_tol: float,
) -> npt.NDArray[np.float64]:
    """Find roots of a single Bernstein polynomial with auto-dispatch.

    Args:
        coeff (npt.NDArray[np.float32 | np.float64]): Validated 1-D
            coefficient array.
        param_tol (float): Parametric tolerance.
        geom_tol (float): Geometric tolerance.

    Returns:
        npt.NDArray[np.float64]: Sorted array of roots in [0, 1].
    """
    n = len(coeff) - 1
    if n < 1:
        return np.empty(0, dtype=np.float64)

    # All-zero polynomial: every point is a root -- return empty.
    if np.all(np.abs(coeff) <= geom_tol):
        return np.empty(0, dtype=np.float64)

    # Low degree: Yuksel is cheaper and equally robust.
    if n < _CLIP_MIN_DEGREE:
        roots_arr, n_roots = _yuksel_roots(coeff, param_tol)
        if n_roots == 0:
            return np.empty(0, dtype=np.float64)
        result = np.sort(roots_arr[:n_roots])
        return result

    # High degree: check coefficient dynamic range.
    abs_coeff = np.abs(coeff)
    c_max = float(abs_coeff.max())
    nonzero = abs_coeff[abs_coeff > 0.0]
    c_min = float(nonzero.min()) if nonzero.size > 0 else 0.0
    coeff_range = c_max / c_min if c_min > 0.0 else float("inf")

    if coeff_range <= _CLIP_COEFF_RANGE_LIMIT:
        # Well-conditioned: use Bezier clipping.
        raw_roots, n_roots = _clip_roots_core(coeff, param_tol, geom_tol)
        return _dedup_roots(raw_roots, n_roots, coeff, param_tol, geom_tol)

    # Extreme range: fall back to Yuksel.
    roots_arr, n_roots = _yuksel_roots(coeff, param_tol)
    if n_roots == 0:
        return np.empty(0, dtype=np.float64)
    return np.sort(roots_arr[:n_roots])


def _find_roots_impl(
    bezier: Bezier,
    *,
    tol: float | None = None,
) -> npt.NDArray[np.float64]:
    """L2 implementation for :func:`pantr.bezier.find_roots`."""
    _validate_bezier_for_roots(bezier)
    coeff = _extract_coeff(bezier)
    wait_for_jit_warmup()
    arr = _validate_coeff_1d(coeff)
    resolved_tol = _resolve_tol(arr, tol)
    return _dispatch_single(arr, resolved_tol, resolved_tol)


def _find_roots_batch_impl(
    beziers: Sequence[Bezier],
    *,
    tol: float | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
    """L2 implementation for :func:`pantr.bezier.find_roots_batch`."""
    from pantr.bezier._batch_core import (  # noqa: PLC0415
        _find_roots_batch_core,
    )

    coeffs = _extract_batch_coeffs(beziers)
    wait_for_jit_warmup()
    arr = _validate_coeffs_batch(coeffs)

    n_polys = arr.shape[0]
    degree = arr.shape[1] - 1

    if n_polys == 0:
        return np.empty((0, max(degree, 1)), dtype=np.float64), np.empty(0, dtype=np.intp)

    resolved_tol = _resolve_tol(arr[0], tol)

    out_roots = np.full((n_polys, max(degree, 1)), np.nan, dtype=np.float64)
    out_counts = np.zeros(n_polys, dtype=np.intp)

    if degree >= 1:
        _find_roots_batch_core(arr, resolved_tol, resolved_tol, out_roots, out_counts)

    return out_roots, out_counts


def _solve_monotone_root_impl(
    bezier: Bezier,
    *,
    tol: float | None = None,
) -> float:
    """L2 implementation for :func:`pantr.bezier.find_monotone_root`."""
    _validate_bezier_for_roots(bezier)
    coeff = _extract_coeff(bezier)
    wait_for_jit_warmup()
    arr = _validate_coeff_1d(coeff)
    resolved_tol = _resolve_tol(arr, tol)
    return float(_solve_monotone_root_kernel(arr, resolved_tol))


def _solve_monotone_root_batch_impl(
    beziers: Sequence[Bezier],
    *,
    tol: float | None = None,
) -> npt.NDArray[np.float64]:
    """L2 implementation for :func:`pantr.bezier.solve_monotone_root_batch`."""
    from pantr.bezier._batch_core import (  # noqa: PLC0415
        _solve_monotone_root_batch_core,
    )

    coeffs = _extract_batch_coeffs(beziers)
    wait_for_jit_warmup()
    arr = _validate_coeffs_batch(coeffs)

    n_polys = arr.shape[0]

    if n_polys == 0:
        return np.empty(0, dtype=np.float64)

    resolved_tol = _resolve_tol(arr[0], tol)
    out_roots = np.full(n_polys, np.nan, dtype=np.float64)

    _solve_monotone_root_batch_core(arr, resolved_tol, out_roots)

    return out_roots
