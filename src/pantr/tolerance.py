"""Tolerance utilities for floating-point comparisons in IGA applications.

The presets returned here (:func:`get_default`, :func:`get_strict`,
:func:`get_conservative`) are **absolute** tolerances: a fixed value per dtype,
independent of the magnitude of the values being compared. This is appropriate for
comparisons made directly against parametric coordinates on an assumed-reasonable
(O(1)) domain, such as knot-multiplicity detection.

Kernels that compare a *difference of two knots* against tolerance (e.g. the
Cox-de Boor recurrence denominator guard in
:mod:`pantr.bspline._bspline_basis_core`) additionally scale the absolute preset by
the knot vector's parametric span before comparing, so the guard stays invariant
under affine reparametrization (shift and scale) of the knots. See
``_scaled_denom_tol`` in that module for the exact rule.
"""

from __future__ import annotations

from functools import cache
from typing import Any, NamedTuple, TypedDict

import numpy as np
from numpy import typing as npt


@cache
def _ensure_float_dtype_by_name(name: str) -> np.dtype[np.floating[Any]]:
    """Cached validator returning a floating dtype from its canonical name.

    Args:
        name (str): Canonical NumPy dtype name (e.g., "float64").

    Returns:
        np.dtype[np.floating[Any]]: Validated floating-point dtype.

    Raises:
        ValueError: If dtype is not a supported floating-point type.
    """
    dtype_obj = np.dtype(name)
    if dtype_obj.type not in (np.float16, np.float32, np.float64, np.longdouble):
        raise ValueError(f"Unsupported dtype: {name}")
    return dtype_obj


def _ensure_float_dtype(dtype: npt.DTypeLike) -> np.dtype[np.floating[Any]]:
    """Normalize and validate a dtype-like value into a supported floating dtype.

    Args:
        dtype (npt.DTypeLike): Any dtype-like value (e.g., ``np.float32``,
            ``"float64"``, or a ``np.dtype`` instance).

    Returns:
        np.dtype[np.floating[Any]]: Validated floating-point dtype.

    Raises:
        ValueError: If ``dtype`` is not one of the supported floating-point
            types (float16, float32, float64, longdouble).
    """
    dtype_obj = np.dtype(dtype)
    return _ensure_float_dtype_by_name(dtype_obj.name)


class _TolerancePreset(NamedTuple):
    """A named tuple to hold tolerance values for different floating-point types."""

    float16: float
    float32: float
    float64: float
    longdouble: float


# This is platform dependent.
if np.dtype(np.longdouble) == np.dtype(np.float64):
    _TOLERANCE_PRESETS = {
        "default": _TolerancePreset(1e-3, 1e-6, 1e-12, 1e-12),
        "strict": _TolerancePreset(1e-4, 1e-7, 1e-15, 1e-15),
        "conservative": _TolerancePreset(1e-2, 1e-5, 1e-10, 1e-10),
    }
else:
    _TOLERANCE_PRESETS = {
        "default": _TolerancePreset(1e-3, 1e-6, 1e-12, 1e-15),
        "strict": _TolerancePreset(1e-4, 1e-7, 1e-15, 1e-18),
        "conservative": _TolerancePreset(1e-2, 1e-5, 1e-10, 1e-12),
    }


def _get_tolerance(
    dtype: npt.DTypeLike,
    preset: _TolerancePreset,
) -> float:
    """Get the tolerance value for a specific dtype from a preset.

    Args:
        dtype (npt.DTypeLike): NumPy floating-point data type.
        preset (_TolerancePreset): A named tuple containing tolerance values.

    Returns:
        float: Tolerance value for the given dtype.

    Raises:
        ValueError: If dtype is not a supported floating-point type.
    """
    dtype_obj = _ensure_float_dtype(dtype)

    # Respect an explicit request for np.longdouble even on platforms where it
    # aliases float64 (e.g., macOS, Windows). The tests expect semantic intent,
    # not platform aliasing.
    if dtype is np.longdouble or (
        isinstance(dtype, str) and dtype.lower().replace(" ", "") == "longdouble"
    ):
        return preset.longdouble

    if dtype_obj.type == np.float16:
        return preset.float16
    elif dtype_obj.type == np.float32:
        return preset.float32
    elif dtype_obj.type == np.float64:
        return preset.float64
    else:  # if dtype_obj.type == np.longdouble:
        return preset.longdouble


def get_default(dtype: npt.DTypeLike) -> float:
    """Get a reasonable default tolerance for floating-point comparisons.

    Args:
        dtype (npt.DTypeLike): NumPy floating-point data type or numpy scalar
            type.

    Returns:
        float: Recommended tolerance value for the given dtype.

    Raises:
        ValueError: If dtype is not a supported floating-point type.

    Example:
        >>> get_default(np.float32)
        1e-06
        >>> get_default("float64")
        1e-12
    """
    return _get_tolerance(dtype, _TOLERANCE_PRESETS["default"])


def get_strict(dtype: npt.DTypeLike) -> float:
    """Get a strict tolerance for high-precision floating-point comparisons.

    Args:
        dtype (npt.DTypeLike): NumPy floating-point data type.

    Returns:
        float: Strict tolerance value for the given dtype. Typically used for
            parametric coordinates requiring high precision.

    Raises:
        ValueError: If dtype is not a supported floating-point type.
    """
    return _get_tolerance(dtype, _TOLERANCE_PRESETS["strict"])


def get_conservative(dtype: npt.DTypeLike) -> float:
    """Get a conservative tolerance for robust floating-point comparisons.

    Args:
        dtype (npt.DTypeLike): NumPy floating-point data type.

    Returns:
        float: Conservative tolerance value for the given dtype. Used when
            robustness is more important than precision.

    Raises:
        ValueError: If dtype is not a supported floating-point type.
    """
    return _get_tolerance(dtype, _TOLERANCE_PRESETS["conservative"])


def get_machine_epsilon(dtype: npt.DTypeLike) -> float:
    """Get machine epsilon for a given floating-point dtype.

    Machine epsilon is the smallest positive number that, when added to 1.0,
    produces a result different from 1.0. It represents the relative error
    in floating-point arithmetic for the given precision.

    Args:
        dtype (npt.DTypeLike): NumPy floating-point data type.

    Returns:
        float: Machine epsilon for the given dtype.

    Raises:
        ValueError: If dtype is not a supported floating-point type.
    """
    _ensure_float_dtype(dtype)

    return float(np.finfo(_ensure_float_dtype(dtype)).eps)


class ToleranceInfo(TypedDict):
    """A TypedDict holding comprehensive tolerance and precision information."""

    dtype: npt.DTypeLike
    machine_epsilon: float
    default_tolerance: float
    strict_tolerance: float
    conservative_tolerance: float
    precision_bits: int
    precision_decimals: int
    resolution: float
    max_value: float
    min_value: float


def get_info(
    dtype: npt.DTypeLike,
) -> ToleranceInfo:
    """Get comprehensive tolerance information for a dtype.

    Args:
        dtype (npt.DTypeLike): NumPy floating-point data type.

    Returns:
        ToleranceInfo: Dictionary containing tolerance information including
            machine epsilon, default/strict/conservative tolerances, precision
            bits, and min/max values for the dtype.

    Raises:
        ValueError: If dtype is not a supported floating-point type.
    """
    dt = _ensure_float_dtype(dtype)
    finfo = np.finfo(dt)

    return {
        "dtype": dtype,  # preserve original representation
        "machine_epsilon": get_machine_epsilon(dt),
        "default_tolerance": get_default(dt),
        "strict_tolerance": get_strict(dt),
        "conservative_tolerance": get_conservative(dt),
        "precision_bits": finfo.precision,
        "precision_decimals": finfo.precision,
        "resolution": float(finfo.resolution),
        "max_value": float(finfo.max),
        "min_value": float(finfo.tiny),
    }


__all__ = [
    "ToleranceInfo",
    "get_conservative",
    "get_default",
    "get_info",
    "get_machine_epsilon",
    "get_strict",
]
