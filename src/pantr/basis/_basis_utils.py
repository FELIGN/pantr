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


def _validate_out_array_1D(
    out: npt.NDArray[np.float32 | np.float64],
    expected_shape: tuple[int, ...],
    expected_dtype: npt.DTypeLike,
) -> None:
    """Validate that the output array has the correct shape and dtype.

    This function follows NumPy's style for output array validation.
    Checks that the array has the expected shape and dtype.

    Args:
        out (npt.NDArray[np.float32 | np.float64]): The output array to validate.
        expected_shape (tuple[int, ...]): The expected shape of the output array.
        expected_dtype (npt.DTypeLike): The expected dtype (should be np.float32 or np.float64).

    Raises:
        ValueError: If the array shape or dtype does not match expectations.
    """
    if out.shape != expected_shape:
        raise ValueError(f"Output array has shape {out.shape}, but expected shape {expected_shape}")
    if out.dtype != expected_dtype:
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


def _validate_out_array_multidimensional(
    out: npt.NDArray[np.float32 | np.float64],
    expected_shape: tuple[int, int],
    expected_dtype: npt.DTypeLike,
) -> None:
    """Validate output array shape and dtype for multidimensional basis functions.

    This function follows NumPy's style for output array validation.
    Checks that the array has the expected shape and dtype.

    Args:
        out (npt.NDArray[np.float32 | np.float64]): The output array to validate.
        expected_shape (tuple[int, int]): The expected shape of the output array
            (n_points, n_basis_functions).
        expected_dtype (npt.DTypeLike): The expected dtype (should be np.float32 or np.float64).

    Raises:
        ValueError: If the array shape or dtype does not match expectations.
    """
    if out.shape != expected_shape:
        raise ValueError(f"Output array has shape {out.shape}, but expected shape {expected_shape}")
    if out.dtype != expected_dtype:
        raise ValueError(f"Output array has dtype {out.dtype}, but expected dtype {expected_dtype}")
    if not out.flags.writeable:
        raise ValueError("Output array is not writeable")


def _validate_out_array_bool(
    out: npt.NDArray[np.bool_],
    expected_shape: tuple[int, ...],
) -> None:
    """Validate that the output array has the correct shape and bool dtype.

    This function follows NumPy's style for output array validation.
    Checks that the array has the expected shape and dtype.

    Args:
        out (npt.NDArray[np.bool_]): The output array to validate.
        expected_shape (tuple[int, ...]): The expected shape of the output array.

    Raises:
        ValueError: If the array shape or dtype does not match expectations.
    """
    if out.shape != expected_shape:
        raise ValueError(f"Output array has shape {out.shape}, but expected shape {expected_shape}")
    if out.dtype != np.bool_:
        raise ValueError(f"Output array has dtype {out.dtype}, but expected dtype bool_")
    if not out.flags.writeable:
        raise ValueError("Output array is not writeable")


def _validate_out_array_3d_float(
    out: npt.NDArray[np.float32 | np.float64],
    expected_shape: tuple[int, int, int],
    expected_dtype: npt.DTypeLike,
) -> None:
    """Validate output array shape and dtype for 3D float arrays.

    This function follows NumPy's style for output array validation.
    Checks that the array has the expected shape and dtype.

    Args:
        out (npt.NDArray[np.float32 | np.float64]): The output array to validate.
        expected_shape (tuple[int, int, int]): The expected shape of the output array.
        expected_dtype (npt.DTypeLike): The expected dtype (should be np.float32 or np.float64).

    Raises:
        ValueError: If the array shape or dtype does not match expectations.
    """
    if out.shape != expected_shape:
        raise ValueError(f"Output array has shape {out.shape}, but expected shape {expected_shape}")
    if out.dtype != expected_dtype:
        raise ValueError(f"Output array has dtype {out.dtype}, but expected dtype {expected_dtype}")
    if not out.flags.writeable:
        raise ValueError("Output array is not writeable")


def _validate_out_array_deriv_1D(
    out: npt.NDArray[np.float32 | np.float64],
    expected_shape: tuple[int, ...],
    expected_dtype: npt.DTypeLike,
) -> None:
    """Validate output array shape and dtype for 1D B-spline derivative results.

    The expected shape is ``(*pts_shape, n_deriv+1, degree+1)``.

    Args:
        out (npt.NDArray[np.float32 | np.float64]): The output array to validate.
        expected_shape (tuple[int, ...]): The expected shape of the output array.
        expected_dtype (npt.DTypeLike): The expected dtype (should be np.float32 or np.float64).

    Raises:
        ValueError: If the array shape, dtype, or writability does not match expectations.
    """
    if out.shape != expected_shape:
        raise ValueError(f"Output array has shape {out.shape}, but expected shape {expected_shape}")
    if out.dtype != expected_dtype:
        raise ValueError(f"Output array has dtype {out.dtype}, but expected dtype {expected_dtype}")
    if not out.flags.writeable:
        raise ValueError("Output array is not writeable")


def _validate_out_array_first_basis(
    out: npt.NDArray[np.int_],
    expected_shape: tuple[int, ...],
) -> None:
    """Validate output array shape and dtype for first basis indices.

    This function follows NumPy's style for output array validation.
    Checks that the array has the expected shape and dtype.

    Args:
        out (npt.NDArray[np.int_]): The output array to validate.
        expected_shape (tuple[int, ...]): The expected shape of the output array.

    Raises:
        ValueError: If the array shape or dtype does not match expectations.
    """
    if out.shape != expected_shape:
        raise ValueError(f"Output array has shape {out.shape}, but expected shape {expected_shape}")
    if out.dtype != np.int_:
        raise ValueError(f"Output array has dtype {out.dtype}, but expected dtype int_")
    if not out.flags.writeable:
        raise ValueError("Output array is not writeable")
