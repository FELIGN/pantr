"""Relative tolerances for floating-point comparisons in IGA applications.

The presets returned here (:func:`get_strict`, :func:`get_default`,
:func:`get_conservative`) are **dimensionless relative** tolerances: a number of
machine epsilons, the *same* number for every dtype. They carry no units and no
scale of their own, so none of them is usable as a comparison on its own.

A call site turns one into a comparison by multiplying it by the magnitude of the
quantity it is actually comparing, and by naming that magnitude where it does so:
``max(|a|, |b|)`` for two values, ``knots[-1] - knots[0]`` for a knot gap, the
bounding-box diagonal for a physical coordinate, ``max|c|`` for a spline
coefficient. The floor near zero is that same product -- a relative test can never
accept ``a = 0`` against a tiny ``b``, so a floor is genuinely needed, but the
problem supplies its scale, never the dtype.

Each preset is ``K * eps(dtype)``. ``K`` is an acknowledged safety factor, stated
rather than tuned, and a power of two so that it is exact in every format and
carries no decimal literal to mistranscribe:

======================  ======  =============================================
preset                  ``K``   what it pays for
======================  ======  =============================================
``get_strict``               4  a handful of roundings: one subtraction and
                                one convex combination, then a difference
``get_default``             64  a short algorithm (about 16 operations) plus
                                build slack: FMA contraction, vector width,
                                libm
``get_conservative``      4096  a long accumulation, or a condition number
                                up to about 1e3, with headroom
======================  ======  =============================================

``K`` being constant across dtypes is what makes the three presets mean the same
thing in every precision: ``get_strict`` is four roundings whether the roundings
are float32's or float64's.

The one place that uniformity has a visible edge is ``float16``, which carries 11
significant bits and therefore cannot absorb 4096 roundings at all:
``get_conservative(np.float16)`` is 4.0, a relative tolerance above one, which
accepts everything. That is the honest answer -- a long accumulation in half
precision retains no digits -- and it is reported rather than clipped. Nothing in
pantr reaches it: the B-spline layer accepts only float32 and float64
(``BsplineSpace1D`` rejects the rest at construction).

Kernels that compare a *difference of two knots* against zero -- the Cox-de Boor
recurrence denominator guard in ``pantr.bspline._bspline_basis_core``
(``_basis_funcs_point``, ``_basis_derivs_point``) -- take no tolerance at all and
use an exact ``denom == 0.0`` test. What makes that sound is the shape of the
recurrence, not the knot vector: ``denom`` is the sum of the two non-negative
terms the step then divides by, so the two weights ``right / denom`` and
``left / denom`` are non-negative and sum to one up to a single rounding. A
denominator small enough to make the quotient large is multiplied straight back by
factors no larger than itself, and the term stays bounded by the basis value it
came from. Near-duplicate knots therefore cost accuracy in that ratio, not
boundedness, and the guard needs no tolerance and no bitwise-identical knots.
"""

from __future__ import annotations

from functools import cache
from typing import Any, Final, TypedDict

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


_STRICT_EPS_FACTOR: Final[float] = 4.0
"""Machine epsilons a :func:`get_strict` comparison allows for.

A handful of roundings and nothing more. Four is the count for the shape this tier
is meant for: a value reached by one convex combination -- a subtract, a multiply
and an add, three roundings -- and then differenced against another such value,
which is the fourth. (The comparison itself is exact in IEEE-754 and costs
nothing.) Four is also already a power of two, so nothing is rounded up here.
"""

_DEFAULT_EPS_FACTOR: Final[float] = 64.0
"""Machine epsilons a :func:`get_default` comparison allows for.

A short algorithm -- on the order of 16 operations, whose worst-case accumulation
is ``16 * eps`` -- plus build slack: an FMA contraction, a different vector width
and a different libm each move the last bits without changing the algorithm.
Four times the accumulation bound covers that, and 64 is the power of two it
lands on.
"""

_CONSERVATIVE_EPS_FACTOR: Final[float] = 4096.0
"""Machine epsilons a :func:`get_conservative` comparison allows for.

A long accumulation, or a problem whose condition number reaches about 1e3: the
error is then ``cond * eps`` rather than an operation count times ``eps``, and
4096 leaves a factor of four over that on top of the same build slack the default
tier allows. Past this the tolerance stops being a safety factor and starts
hiding the answer, so a site that needs more needs a derivation of its own.
"""


def _relative_tolerance(dtype: npt.DTypeLike, eps_factor: float) -> float:
    """Scale a dimensionless safety factor by a dtype's machine epsilon.

    Args:
        dtype (npt.DTypeLike): NumPy floating-point data type.
        eps_factor (float): Number of machine epsilons the tolerance allows for.

    Returns:
        float: The dimensionless relative tolerance ``eps_factor * eps(dtype)``.

    Raises:
        ValueError: If dtype is not a supported floating-point type.
    """
    return eps_factor * float(np.finfo(_ensure_float_dtype(dtype)).eps)


def get_default(dtype: npt.DTypeLike) -> float:
    """Get the default relative tolerance for floating-point comparisons.

    ``_DEFAULT_EPS_FACTOR * eps(dtype)``: the tier for a short algorithm plus build
    slack. Dimensionless -- multiply it by the magnitude of the quantity being
    compared, and name that magnitude at the call site.

    Args:
        dtype (npt.DTypeLike): NumPy floating-point data type or numpy scalar
            type.

    Returns:
        float: Relative tolerance for the given dtype.

    Raises:
        ValueError: If dtype is not a supported floating-point type.

    Example:
        >>> get_default("float64")
        1.4210854715202004e-14
        >>> get_default(np.float32) / float(np.finfo(np.float32).eps)
        64.0
    """
    return _relative_tolerance(dtype, _DEFAULT_EPS_FACTOR)


def get_strict(dtype: npt.DTypeLike) -> float:
    """Get the strict relative tolerance for high-precision comparisons.

    ``_STRICT_EPS_FACTOR * eps(dtype)``: the tier for a handful of roundings, used
    where a quantity is expected to agree to within the format's own noise.
    Dimensionless -- multiply it by the magnitude of the quantity being compared,
    and name that magnitude at the call site.

    Args:
        dtype (npt.DTypeLike): NumPy floating-point data type.

    Returns:
        float: Relative tolerance for the given dtype.

    Raises:
        ValueError: If dtype is not a supported floating-point type.

    Example:
        >>> get_strict("float64")
        8.881784197001252e-16
        >>> get_strict(np.float32) / float(np.finfo(np.float32).eps)
        4.0
    """
    return _relative_tolerance(dtype, _STRICT_EPS_FACTOR)


def get_conservative(dtype: npt.DTypeLike) -> float:
    """Get the conservative relative tolerance for robust comparisons.

    ``_CONSERVATIVE_EPS_FACTOR * eps(dtype)``: the tier for a long accumulation or
    a moderately ill-conditioned step, used where robustness matters more than
    resolving power. Dimensionless -- multiply it by the magnitude of the quantity
    being compared, and name that magnitude at the call site.

    Args:
        dtype (npt.DTypeLike): NumPy floating-point data type.

    Returns:
        float: Relative tolerance for the given dtype.

    Raises:
        ValueError: If dtype is not a supported floating-point type.

    Example:
        >>> get_conservative("float64")
        9.094947017729282e-13
        >>> get_conservative(np.float32) / float(np.finfo(np.float32).eps)
        4096.0
    """
    return _relative_tolerance(dtype, _CONSERVATIVE_EPS_FACTOR)


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
    """A TypedDict holding comprehensive tolerance and precision information.

    The three tolerance entries are the **dimensionless relative** presets, not
    absolute bounds; see the module docstring.
    """

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
            machine epsilon, the default/strict/conservative *relative*
            tolerances, precision bits, and min/max values for the dtype.

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
