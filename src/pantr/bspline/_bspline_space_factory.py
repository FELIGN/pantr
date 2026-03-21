"""Knot vector construction functions for B-splines."""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from ._bspline_knots import (
    _get_knots_ends_and_dtype,
    _validate_knot_input,
)


def create_uniform_open(
    num_intervals: int,
    degree: int,
    continuity: int | None = None,
    domain: tuple[np.float32 | np.float64 | float, np.float32 | np.float64 | float] | None = None,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create a uniform open knot vector.

    An open knot vector has the first and last knots repeated (degree+1) times,
    ensuring the B-spline interpolates the first and last control points.

    Args:
        num_intervals (int): Number of intervals in the domain. Must be non-negative.
        degree (int): B-spline degree. Must be non-negative.
        continuity (Optional[int]): Continuity level at interior knots.
            Must be between -1 and degree-1. Defaults to degree-1 (maximum continuity).
        domain (Optional[tuple[np.float32 | np.float64 | float, np.float32 | np.float64 | float]]):
            Domain boundaries as (start, end). Defaults to (0.0, 1.0) if not provided.
        dtype (Optional[np.dtype]): Data type for the knot vector.
            If None, inferred from start/end or defaults to float64.

    Returns:
        npt.NDArray[np.floating]: Open knot vector with uniform spacing.

    Raises:
        ValueError: If any parameter is invalid.

    Example:
        >>> create_uniform_open(2, 2, domain=(0.0, 1.0))
        array([0., 0., 0., 0.5, 1., 1., 1.])
    """
    start_value: np.float32 | np.float64 | None
    end_value: np.float32 | np.float64 | None
    if domain is None:
        start_value = None
        end_value = None
    else:
        start_raw, end_raw = domain
        start_value = start_raw if isinstance(start_raw, np.floating) else np.float64(start_raw)
        end_value = end_raw if isinstance(end_raw, np.floating) else np.float64(end_raw)

    start, end, dtype = _get_knots_ends_and_dtype(start_value, end_value, dtype)

    continuity = degree - 1 if continuity is None else continuity

    _validate_knot_input(
        num_intervals,
        degree,
        continuity,
        (start, end),
        dtype,
    )

    unique_knots = np.linspace(start, end, num_intervals + 1, dtype=dtype)
    knots = np.array([start] * (degree + 1), dtype)

    interior_multiplicity = degree - continuity
    for knot in unique_knots[1:-1]:
        knots = np.append(knots, [knot] * interior_multiplicity)

    knots = np.append(knots, [end] * (degree + 1))

    return knots


def create_uniform_periodic(
    num_intervals: int,
    degree: int,
    continuity: int | None = None,
    domain: tuple[np.float32 | np.float64 | float, np.float32 | np.float64 | float] | None = None,
    dtype: npt.DTypeLike | None = np.float64,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create a uniform periodic knot vector.

    A periodic knot vector extends beyond the domain boundaries to ensure
    periodicity of the B-spline basis functions.

    Args:
        num_intervals (int): Number of intervals in the domain. Must be non-negative.
        degree (int): B-spline degree. Must be non-negative.
        continuity (Optional[int]): Continuity level at interior knots.
            Must be between -1 and degree-1. Defaults to degree-1 (maximum continuity).
        domain (Optional[tuple[np.float32 | np.float64 | float, np.float32 | np.float64 | float]]):
            Domain boundaries as (start, end). Defaults to (0.0, 1.0) if not provided.
        dtype (Optional[np.dtype]): Data type for the knot vector.
            If None, inferred from start/end or defaults to float64.

    Returns:
        npt.NDArray[np.floating]: Periodic knot vector with uniform spacing.

    Raises:
        ValueError: If any parameter is invalid.

    Example:
        >>> create_uniform_periodic(2, 2, domain=(0.0, 1.0))
        array([-1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ])
    """
    start_value: np.float32 | np.float64 | None
    end_value: np.float32 | np.float64 | None
    if domain is None:
        start_value = None
        end_value = None
    else:
        start_raw, end_raw = domain
        start_value = start_raw if isinstance(start_raw, np.floating) else np.float64(start_raw)
        end_value = end_raw if isinstance(end_raw, np.floating) else np.float64(end_raw)

    start, end, dtype = _get_knots_ends_and_dtype(start_value, end_value, dtype)
    continuity = degree - 1 if continuity is None else continuity

    _validate_knot_input(
        num_intervals,
        degree,
        continuity,
        (start, end),
        dtype,
    )

    # Create uniform spacing for unique interior knots
    unique_knots = np.linspace(start, end, num_intervals + 1, dtype=dtype)

    # Build knot vector with repetitions
    knots = np.array([], dtype=dtype)

    multiplicity = degree - continuity

    # Starting periodic knots.
    length = (end - start) / num_intervals
    knots = np.linspace(
        start - length * (degree - multiplicity + 1),
        start,
        degree + 2 - multiplicity,
        dtype=dtype,
    )[:-1]

    # Interior knots with specified multiplicity
    for knot in unique_knots:
        knots = np.append(knots, [knot] * multiplicity)

    # End periodic knots.
    knots = np.append(
        knots,
        np.linspace(
            end,
            end + length * (degree - multiplicity + 1),
            degree + 2 - multiplicity,
            dtype=dtype,
        )[1:],
    )

    return knots


def create_cardinal(
    num_intervals: int,
    degree: int,
    dtype: npt.DTypeLike = np.float64,
) -> npt.NDArray[np.float32 | np.float64]:
    """Create a knot vector for cardinal B-spline basis functions.

    Cardinal B-splines are B-splines defined on uniform knot vectors with
    maximum continuity, where the basis functions in the central region
    have the same shape and are translated versions of each other.

    Args:
        num_intervals (int): Number of intervals in the domain. Must be at least 1.
        degree (int): B-spline degree. Must be non-negative.
        dtype (npt.DTypeLike): Data type for the knot vector.
            It must be either float32 or float64. Defaults to np.float64.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Cardinal B-spline knot vector
            with uniform spacing.

    Raises:
        ValueError: If num_intervals < 1, degree < 0, or dtype is not float32/float64.

    Example:
        >>> create_cardinal(2, 2)
        array([-2., -1.,  0.,  1.,  2.,  3., 4.])
    """
    if num_intervals < 1:
        raise ValueError("num_intervals must be at least 1")

    if degree < 0:
        raise ValueError("degree must be non-negative")

    dtype_obj = np.dtype(dtype)
    if dtype_obj not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")

    start_value: np.float32 | np.float64
    end_value: np.float32 | np.float64
    if dtype_obj == np.dtype(np.float64):
        start_value = np.float64(0)
        end_value = np.float64(num_intervals)
    else:
        start_value = np.float32(0)
        end_value = np.float32(num_intervals)

    return create_uniform_periodic(
        num_intervals,
        degree,
        continuity=degree - 1,
        domain=(start_value, end_value),
        dtype=dtype_obj,
    )
