"""B-spline knot vector utilities and analysis.

This module provides functions for validating, analyzing, and querying
B-spline knot vectors including multiplicity computation, domain checks,
cardinal interval identification, and knot vector generation utilities.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit
from ..basis._basis_utils import _validate_float_dtype, _validate_out_array


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _check_spline_info(knots: npt.NDArray[np.float32 | np.float64], degree: int) -> None:
    """Validate basic constraints on B-spline knot vector and degree.

    Performs core checks on a B-spline knot vector and polynomial degree,
    raising an ValueError if any validation fails.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): 1D array representing the B-spline
            knot vector to check.
        degree (int): Non-negative polynomial degree for the B-spline.

    Raises:
        TypeError: If `knots` is not 1-dimensional.
        ValueError: If `degree` is negative,
            if there are fewer than `2*degree+2` knots, or if the knot vector
            is not non-decreasing.
    """
    if knots.ndim != 1:
        raise TypeError("knots must be a 1D array")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if knots.size < (2 * degree + 2):
        raise ValueError("knots must have at least 2*degree+2 elements")
    if not np.all(np.diff(knots) >= knots.dtype.type(0.0)):
        raise ValueError("knots must be non-decreasing")


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _get_multiplicity_of_first_knot_in_domain_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
) -> int:
    """Get the multiplicity of the first knot in the domain (i.e., the `degree`-th knot).

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.

    Returns:
        int: Multiplicity of the first knot in the domain.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    first_knot = knots[degree]
    return int(np.sum(np.isclose(knots[: degree + 1], first_knot, atol=tol)))


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _get_unique_knots_and_multiplicity_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    in_domain: bool = False,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Get unique knots and their multiplicities.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.
        in_domain (bool): If True, only consider knots in the domain.
            Defaults to False.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]: Tuple of
            (unique_knots, multiplicities) where unique_knots contains the distinct knot values
            and multiplicities contains the corresponding multiplicity counts. Both arrays have
            the same length.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    # Round to tolerance precision for grouping
    dtype = knots.dtype
    scale = dtype.type(1.0 / tol)
    rounded_knots = np.round(knots * scale) / scale

    n = knots.size
    # unique_rounded_knots = np.empty(n, dtype=rounded_knots.dtype)
    unique_rounded_knots_ids = np.empty(n, dtype=np.int_)
    mult = np.zeros(n, dtype=np.int_)

    if in_domain:
        rknot_0, rknot_1 = rounded_knots[degree], rounded_knots[-degree - 1]
    else:
        rknot_0, rknot_1 = rounded_knots[0], rounded_knots[-1]

    j = -1
    last_rknot = np.nan

    for i, rknot in enumerate(rounded_knots):
        if rknot < rknot_0:
            continue
        elif rknot > rknot_1:
            break

        if rknot == last_rknot:
            mult[j] += 1
        else:
            j += 1
            last_rknot = rknot
            unique_rounded_knots_ids[j] = i
            mult[j] = 1

    unique_knots = rounded_knots[unique_rounded_knots_ids[: j + 1]]
    mults = mult[: j + 1]
    return unique_knots, mults


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _is_in_domain_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    pts: npt.NDArray[np.float32 | np.float64],
    tol: float,
) -> npt.NDArray[np.bool_]:
    """Check if points are within the B-spline domain (up to tolerance).

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        pts (npt.NDArray[np.float32 | np.float64]): Points to check.
        tol (float): Tolerance for numerical comparisons.

    Returns:
        npt.NDArray[np.bool_]: Boolean array where True indicates points
            are within the domain. It has the same length as the number of points.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    knot_begin, knot_end = knots[degree], knots[-degree - 1]
    return np.logical_and(  # type: ignore[no-any-return]
        (knot_begin < pts) | np.isclose(knot_begin, pts, atol=tol),
        (pts < knot_end) | np.isclose(pts, knot_end, atol=tol),
    )


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _get_last_knot_smaller_equal_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    pts: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.int_]:
    """Get the index of the last knot which is less than or equal to each point in pts.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector (must be non-decreasing).
        pts (npt.NDArray[np.float32 | np.float64]): Points (1D array) to find knot indices for.

    Returns:
        npt.NDArray[np.int_]: Array of computed indices, one for each point in pts.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    return np.searchsorted(knots, pts, side="right") - 1


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _get_Bspline_num_basis_1D_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    periodic: bool,
    tol: float,
) -> int:
    """Compute the number of basis functions.

    In the non-periodic case, the number of basis functions is given by
    the number of knots minus the degree minus 1.

    In the periodic case, the number of basis functions is computed as
    before, minus the regularity at the domain's beginning/end minus 1.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        periodic (bool): Whether the B-spline is periodic.
        tol (np.float32 | np.float64): Tolerance for numerical comparisons.

    Returns:
        int: Number of basis functions.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    num_basis = int(len(knots) - degree - 1)

    if periodic:
        # Determining the number of extra basis required in the periodic case.
        # This depends on the regularity of the knot vector at domain's
        # begining.
        regularity = degree - _get_multiplicity_of_first_knot_in_domain_impl(knots, degree, tol)
        num_basis -= regularity + 1

    return num_basis


@nb_jit(
    nopython=True,
    cache=True,
    parallel=False,
)
def _get_Bspline_cardinal_intervals_1D_core(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    out: npt.NDArray[np.bool_],
) -> None:
    """Core implementation to compute cardinal intervals, writing to output array.

    An interval is cardinal if it has the same length as the degree-1
    previous and the degree-1 next intervals.

    In the case of open knot vectors, this definition automatically
    discards the first degree-1 and the last degree-1 intervals.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.
        out (npt.NDArray[np.bool_]): Output array where results will be written.
            Must have the correct shape (no validation performed inside this
            numba-compiled function).

    Note:
        This is a Numba-compiled function optimized for performance. It
        expects pre-validated inputs and assumes the output array has the
        correct shape. Inputs are assumed to be correct (no validation performed).
        For general use, call _get_Bspline_cardinal_intervals_1D_impl instead.
    """
    _, mult = _get_unique_knots_and_multiplicity_impl(knots, degree, tol, in_domain=True)
    num_intervals = len(mult) - 1

    out.fill(np.False_)

    if np.all(mult > 1):
        return

    knot_id = degree

    # Note: this loop could be shortened by only looking at those
    # intervals for which the multiplicity of the first knot is 1.
    # This would require to compute knot_id differently.
    for elem_id in range(num_intervals):
        if mult[elem_id] == 1 and mult[elem_id + 1] == 1:
            local_knots = knots[knot_id - degree + 1 : knot_id + degree + 1]
            lengths = np.diff(local_knots)
            if np.all(np.isclose(lengths, lengths[degree - 1], atol=tol)):
                out[elem_id] = np.True_

        knot_id += mult[elem_id + 1]


def _get_Bspline_cardinal_intervals_1D_impl(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    tol: float,
    out: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.bool_]:
    """Get boolean array indicating whether intervals are cardinal.

    An interval is cardinal if it has the same length as the degree-1
    previous and the degree-1 next intervals.

    In the case of open knot vectors, this definition automatically
    discards the first degree-1 and the last degree-1 intervals.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): B-spline degree.
        tol (float): Tolerance for numerical comparisons.
        out (npt.NDArray[np.bool_] | None): Optional output array where the result will be
            stored. If None, a new array is allocated. Must have the correct shape and dtype
            if provided. This follows NumPy's style for output arrays. Defaults to None.

    Returns:
        npt.NDArray[np.bool_]: Boolean array where True indicates cardinal intervals.
            It has length equal to the number of intervals. If `out` was provided,
            returns the same array.

    Raises:
        ValueError: If the knot vector or degree fails basic validation or if tol is negative.
        ValueError: If `out` is provided and has incorrect shape or dtype.
    """
    _, mult = _get_unique_knots_and_multiplicity_impl(knots, degree, tol, in_domain=True)
    num_intervals = len(mult) - 1

    if out is None:
        out = np.empty(num_intervals, dtype=np.bool_)
    else:
        _validate_out_array(out, (num_intervals,), np.bool_)

    _get_Bspline_cardinal_intervals_1D_core(knots, degree, tol, out)

    return out


def _validate_knot_input(
    num_intervals: int,
    degree: int,
    continuity: int,
    domain: tuple[np.float32 | np.float64, np.float32 | np.float64],
    dtype: npt.DTypeLike,
) -> None:
    """Validate input parameters for knot vector generation.

    Args:
        num_intervals (int_): Number of intervals in the domain.
        degree (int): B-spline degree.
        continuity (int): Continuity level at interior knots.
        domain (tuple[np.float32 | np.float64, np.float32 | np.float64]):
            Domain boundaries as (start, end).
        dtype (np.dtype): Data type for the knot vector.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if domain[0] >= domain[1]:
        raise ValueError("domain[0] must be less than domain[1]")

    if num_intervals < 0:
        raise ValueError("num_intervals must be non-negative")

    if degree < 0:
        raise ValueError("degree must be non-negative")

    if continuity < -1 or continuity >= degree:
        raise ValueError(f"Continuity must be between -1 and {degree - 1} for degree {degree}.")

    _validate_float_dtype(dtype)


def _ensure_scalar_arrays(
    values: dict[str, float | int | np.floating[Any] | None],
) -> dict[str, npt.NDArray[np.generic]]:
    """Convert scalar inputs to zero-dimensional arrays.

    Args:
        values (dict[str, Optional[float | int | np.floating]]): Mapping of parameter
            names to scalar values.

    Returns:
        dict[str, npt.NDArray[np.generic]]: Mapping of provided names to 0-D arrays.

    Raises:
        ValueError: If any value is not scalar.
    """
    arrays: dict[str, npt.NDArray[np.generic]] = {}
    for name, value in values.items():
        if value is None:
            continue
        array_value = np.array(value)
        if array_value.ndim != 0:
            raise ValueError(f"{name} must be a scalar value")
        arrays[name] = array_value
    return arrays


def _resolve_dtype_from_arrays(
    arrays: dict[str, npt.NDArray[np.generic]],
    requested_dtype: npt.DTypeLike | None,
) -> np.dtype[np.floating[Any]]:
    """Resolve the floating dtype to use for knot endpoints.

    Args:
        arrays (dict[str, npt.NDArray[np.generic]]): Scalar arrays for each value.
        requested_dtype (Optional[npt.DTypeLike]): Explicit dtype request.

    Returns:
        np.dtype[np.floating[Any]]: Resolved floating-point dtype.

    Raises:
        ValueError: If the dtype is invalid or inconsistent across values.
    """
    if requested_dtype is not None:
        dtype_obj = np.dtype(requested_dtype)
        if dtype_obj.kind != "f":
            raise ValueError("dtype must be a floating-point type")
        for name, array_value in arrays.items():
            if array_value.dtype != dtype_obj:
                raise ValueError(f"{name} must be of type dtype {dtype_obj}")
        return cast(np.dtype[np.floating[Any]], dtype_obj)

    inferred_dtype: np.dtype[np.floating[Any]] | None = None
    for array_value in arrays.values():
        candidate = (
            cast(np.dtype[np.floating[Any]], np.dtype(array_value.dtype))
            if array_value.dtype.kind == "f"
            else cast(np.dtype[np.floating[Any]], np.dtype(np.float64))
        )
        if inferred_dtype is None:
            inferred_dtype = candidate
        elif candidate != inferred_dtype:
            raise ValueError("start and end must have the same dtype")

    return (
        inferred_dtype
        if inferred_dtype is not None
        else cast(np.dtype[np.floating[Any]], np.dtype(np.float64))
    )


def _coerce_scalar(
    array_value: npt.NDArray[np.generic] | None,
    dtype_obj: np.dtype[np.floating[Any]],
    default: float,
) -> np.floating[Any]:
    """Convert a scalar array to the target dtype or use the default value.

    Args:
        array_value (Optional[npt.NDArray[np.generic]]): Scalar array to convert.
        dtype_obj (np.dtype[np.floating[Any]]): Target floating dtype.
        default (float): Default value when the array is None.

    Returns:
        np.floating[Any]: Value converted to the requested dtype.
    """
    if array_value is None:
        return dtype_obj.type(default)
    return dtype_obj.type(array_value.astype(dtype_obj, copy=False).item())


def _get_knots_ends_and_dtype(
    start: float | int | np.floating[Any] | None = None,
    end: float | int | np.floating[Any] | None = None,
    dtype: npt.DTypeLike | None = None,
) -> tuple[np.floating[Any], np.floating[Any], np.dtype[np.floating[Any]]]:
    """Get the start, end, and dtype for a knot vector.

    Args:
        start (Optional[float | int | np.floating]): Start value of the domain.
            Defaults to 0.0 if not provided.
        end (Optional[float | int | np.floating]): End value of the domain.
            Defaults to 1.0 if not provided.
        dtype (Optional[npt.DTypeLike]): Data type for the knot vector.
            If ``None``, inferred from start/end or defaults to ``np.float64``.

    Returns:
        tuple[np.floating, np.floating, np.dtype]: Tuple of (start, end, dtype).

    Raises:
        ValueError: If inputs are non-scalar, have incompatible dtypes, or if end <= start.
    """
    arrays = _ensure_scalar_arrays({"start": start, "end": end})
    dtype_obj = _resolve_dtype_from_arrays(arrays, dtype)
    start_value = _coerce_scalar(arrays.get("start"), dtype_obj, 0.0)
    end_value = _coerce_scalar(arrays.get("end"), dtype_obj, 1.0)

    if end_value <= start_value:
        raise ValueError("end must be greater than start")

    return start_value, end_value, dtype_obj


def _find_span(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    n_basis: int,
    value: float,
    tol: float,
) -> tuple[int, float]:
    """Find the knot span index for a parameter value (Piegl-Tiller convention).

    Returns the span index ``k`` such that ``knots[k] <= u < knots[k+1]``,
    with the convention that at the right boundary ``k = n_basis - 1``.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): Polynomial degree.
        n_basis (int): Number of basis functions.
        value (float): Parameter value (must be in the domain).
        tol (float): Tolerance for boundary clamping.

    Returns:
        tuple[int, float]: ``(k, u)`` where ``k`` is the span index and
        ``u`` is the (possibly clamped) parameter value.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    u = float(value)
    u_left = float(knots[degree])
    u_right = float(knots[n_basis])

    # Clamp to domain boundaries.
    if u <= u_left + tol:
        return degree, u_left
    if u >= u_right - tol:
        return n_basis - 1, u_right

    # Interior: binary search via searchsorted.
    k = int(np.searchsorted(knots, u, side="right")) - 1
    return k, u


def _count_multiplicity(
    knots: npt.NDArray[np.float32 | np.float64],
    k: int,
    degree: int,
    u: float,
    tol: float,
) -> int:
    """Count the multiplicity of ``u`` among the local knots around span ``k``.

    Scans knots ``k``, ``k-1``, ... going left while they match ``u``
    (within tolerance).  This gives the number of consecutive knots equal
    to ``u`` ending at position ``k``.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        k (int): Span index from :func:`_find_span`.
        degree (int): Polynomial degree.
        u (float): Parameter value.
        tol (float): Tolerance for knot coincidence tests.

    Returns:
        int: Multiplicity of ``u`` in the local knot neighborhood.

    Note:
        Inputs are assumed to be correct (no validation performed).
    """
    s = 0
    for i in range(k, max(k - degree - 1, -1), -1):
        if abs(float(knots[i]) - u) <= tol:
            s += 1
        else:
            break
    return s


def _find_knot_index_and_multiplicity(
    knots: npt.NDArray[np.float32 | np.float64],
    degree: int,
    knot_value: float,
    tol: float,
) -> tuple[int, int]:
    """Find the last index and multiplicity of a knot value in the knot vector.

    Searches the knot vector for the last occurrence of *knot_value* (within
    tolerance) and returns its index *r* together with the multiplicity *s*.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): B-spline knot vector.
        degree (int): Polynomial degree.
        knot_value (float): The knot value to locate.
        tol (float): Tolerance for numerical comparisons.

    Returns:
        tuple[int, int]: ``(r, s)`` where *r* is the index of the last
        occurrence and *s* is the multiplicity.

    Raises:
        ValueError: If *knot_value* is not found in the knot vector.
    """
    unique_knots, mults = _get_unique_knots_and_multiplicity_impl(
        knots, degree, tol, in_domain=False
    )

    # Find the matching unique knot.
    matches = np.where(np.isclose(unique_knots, knot_value, atol=tol))[0]
    if len(matches) == 0:
        raise ValueError(f"Knot value {knot_value} not found in knot vector (tolerance={tol}).")

    idx = int(matches[0])
    s = int(mults[idx])

    # Compute the last index r of this knot value in the full knot vector.
    # Sum multiplicities up to and including this knot.
    r = int(np.sum(mults[: idx + 1])) - 1

    return r, s


def _warmup_numba_functions() -> None:
    """Precompile numba functions with float64 signatures for faster first call.

    This function triggers compilation of the numba-decorated functions
    with float64 arrays, ensuring they are cached and ready for use.
    """
    # Small dummy arrays for warmup
    knots_dummy = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    pts_dummy = np.array([0.5], dtype=np.float64)
    tol_dummy = 1e-10
    degree_dummy = 2

    # Warmup each numba function with float64
    _check_spline_info(knots_dummy, degree_dummy)
    _get_multiplicity_of_first_knot_in_domain_impl(knots_dummy, degree_dummy, tol_dummy)
    _get_unique_knots_and_multiplicity_impl(knots_dummy, degree_dummy, tol_dummy, False)
    _is_in_domain_impl(knots_dummy, degree_dummy, pts_dummy, tol_dummy)
    _get_Bspline_num_basis_1D_impl(knots_dummy, degree_dummy, False, tol_dummy)
    _get_last_knot_smaller_equal_impl(knots_dummy, pts_dummy)
    # For cardinal intervals: with knots [0,0,0,1,1,1] and degree 2, we have 1 interval
    out_cardinal_dummy = np.empty(1, dtype=np.bool_)
    _get_Bspline_cardinal_intervals_1D_core(
        knots_dummy, degree_dummy, tol_dummy, out_cardinal_dummy
    )


# Precompile numba functions on module import
# (Moved to central thread in __init__.py)


__all__ = [
    "_check_spline_info",
    "_count_multiplicity",
    "_find_knot_index_and_multiplicity",
    "_find_span",
    "_get_Bspline_cardinal_intervals_1D_impl",
    "_get_Bspline_num_basis_1D_impl",
    "_get_knots_ends_and_dtype",
    "_get_last_knot_smaller_equal_impl",
    "_get_multiplicity_of_first_knot_in_domain_impl",
    "_get_unique_knots_and_multiplicity_impl",
    "_is_in_domain_impl",
    "_validate_knot_input",
]
