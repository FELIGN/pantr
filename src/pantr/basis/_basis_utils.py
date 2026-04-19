"""Utility functions for basis function evaluation.

This module provides shared helpers used across Layer 2 (implementation
helpers) of PaNTr:

- Point normalization: convert arbitrary array-likes to 1D float arrays.
- Output shape computation: determine the expected array shape given input
  dimensions and the number of basis functions.
- Output array validation: check shape, dtype, and writability of pre-allocated
  ``out`` arrays before calling Layer 3 kernels.
"""

import numpy as np
from numpy import typing as npt


def _normalize_points_1D(pts: npt.ArrayLike) -> npt.NDArray[np.float32 | np.float64]:
    """Normalize points to a 1D float array for basis function evaluation.

    Converts input points (scalar, list, or numpy array) to a 1D numpy array
    with floating point dtype. Types different from float32 or float64 are
    automatically converted to float64.
    Zero-dimensional arrays (scalars) are converted to 1D arrays with a single
    element. Multi-dimensional arrays will be flattened to 1D.

    Args:
        pts (npt.ArrayLike): Evaluation points. Can be a scalar, list, or numpy
            array of any floating-point or integer dtype.

    Returns:
        npt.NDArray[np.float32 | np.float64]: A 1D numpy array with floating
        point dtype. The dtype is preserved from the input if it's already a
        floating point type (float32/float64), otherwise converted to np.float64.
        The array is guaranteed to have exactly one dimension (ndim == 1).
    """
    if not isinstance(pts, np.ndarray):
        pts = np.array(pts)

    if pts.dtype not in (np.float32, np.float64):
        pts = pts.astype(np.float64)

    if pts.ndim == 0:
        pts = np.array([pts], dtype=pts.dtype)
    elif pts.ndim > 1:
        pts = pts.ravel()

    return pts


def _compute_final_output_shape_1D(input_shape: tuple[int, ...], n_basis: int) -> tuple[int, ...]:
    """Compute the final output shape for 1D basis functions.

    Args:
        input_shape (tuple[int, ...]): The shape of the input points (before normalization).
        n_basis (int): The number of basis functions (degree + 1).

    Returns:
        tuple[int, ...]: The final output shape.
    """
    if len(input_shape) == 0:
        # Scalar input: output shape is (n_basis,)
        return (n_basis,)
    else:
        # Non-scalar input: output shape is (*input_shape, n_basis)
        return (*input_shape, n_basis)


def _compute_final_output_shape_1D_deriv(
    input_shape: tuple[int, ...],
    n_deriv: int,
    n_basis: int,
) -> tuple[int, ...]:
    """Compute the final output shape for 1D B-spline derivative arrays.

    Args:
        input_shape (tuple[int, ...]): The shape of the input points (before normalization).
        n_deriv (int): Maximum derivative order.
        n_basis (int): The number of local basis functions (degree + 1).

    Returns:
        tuple[int, ...]: Output shape with two trailing axes (n_deriv+1, n_basis).
    """
    if len(input_shape) == 0:
        return (n_deriv + 1, n_basis)
    return (*input_shape, n_deriv + 1, n_basis)


def _validate_out_array(
    out: np.ndarray,  # type: ignore[type-arg]
    expected_shape: tuple[int, ...],
    expected_dtype: npt.DTypeLike,
) -> None:
    """Validate that an ``out`` array has the expected shape, dtype, and is writeable.

    Single shared validator used across Layer 2 for any ``out`` array — float,
    bool, or integer — passed in the NumPy ``out=`` style.

    Args:
        out (np.ndarray): The output array to validate.
        expected_shape (tuple[int, ...]): The expected shape.
        expected_dtype (npt.DTypeLike): The expected dtype (e.g. ``np.float32``,
            ``np.float64``, ``np.bool_``, ``np.int_``).

    Raises:
        ValueError: If the array shape, dtype, or writability does not match.
    """
    if out.shape != expected_shape:
        raise ValueError(f"Output array has shape {out.shape}, but expected shape {expected_shape}")
    if out.dtype != np.dtype(expected_dtype):
        raise ValueError(f"Output array has dtype {out.dtype}, but expected dtype {expected_dtype}")
    if not out.flags.writeable:
        raise ValueError("Output array is not writeable")


def _compute_output_shape_multidimensional(
    n_points: int,
    n_basis_functions: int,
) -> tuple[int, int]:
    """Compute the expected output shape for multidimensional basis functions.

    Args:
        n_points (int): The number of points at which to evaluate.
        n_basis_functions (int): The total number of basis functions.

    Returns:
        tuple[int, int]: The expected output shape (n_points, n_basis_functions).
    """
    return (n_points, n_basis_functions)


def _allocate_or_validate_out(
    out: npt.NDArray[np.float32 | np.float64] | None,
    expected_shape: tuple[int, ...],
    expected_dtype: npt.DTypeLike,
) -> npt.NDArray[np.float32 | np.float64]:
    """Allocate a fresh ``out`` array or validate the user-provided one.

    Encapsulates the NumPy ``out=``-style pattern used throughout Layer 2: if
    no array is provided, allocate one with the expected shape and dtype;
    otherwise validate that the provided array matches.

    Args:
        out (npt.NDArray[np.float32 | np.float64] | None): Caller-provided
            output array, or ``None`` to allocate a new one.
        expected_shape (tuple[int, ...]): Required shape.
        expected_dtype (npt.DTypeLike): Required dtype.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The validated or freshly
        allocated output array.

    Raises:
        ValueError: If ``out`` is provided with a mismatching shape, dtype, or
            is not writeable.
    """
    if out is None:
        return np.empty(expected_shape, dtype=expected_dtype)
    _validate_out_array(out, expected_shape, expected_dtype)
    return out
