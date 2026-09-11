"""Utility functions for basis function evaluation.

This module provides shared helpers used across Layer 2 (implementation
helpers) of PaNTr:

- Point normalization: convert arbitrary array-likes to 1D float arrays.
- Output shape computation: determine the expected array shape given input
  dimensions and the number of basis functions.
- Output array validation: check shape, dtype, and writability of pre-allocated
  ``out`` arrays before calling Layer 3 kernels.
- Output array reshaping: hand a kernel the shape it expects, and say whether the
  caller's array still has to be written from the result.
"""

from typing import Any

import numpy as np
from numpy import typing as npt

from .._array_utils import _validate_float_dtype

__all__ = ["_validate_float_dtype"]


def _reject_masked_array(array: npt.ArrayLike, role: str) -> None:
    """Refuse a masked array, which no kernel below this layer can honour.

    A :class:`numpy.ma.MaskedArray` carries per-element validity that every
    kernel would silently drop -- and the two backends do not even drop it the
    same way. Measured on ``tabulate_cardinal_bspline_1d``: Numba refuses the
    type outright, while the C++ binding accepts it through the buffer protocol,
    computes on the underlying data and returns plausible numbers for the entries
    the caller marked invalid. That is precisely the divergence
    ``pantr._backend`` promises cannot happen: selecting a backend must change
    how fast the library is, never what it accepts.

    So the mask is refused here, in the layer that owns validation, and the
    refusal follows Numba, which is the correct behaviour of the two. Other
    ``ndarray`` subclasses are deliberately not checked: this is the one whose
    *semantics* are lost, rather than merely its subclass identity.

    Args:
        array (npt.ArrayLike): The argument to inspect. Anything that is not a
            masked array passes.
        role (str): What the argument is, named in the message (e.g. ``points``).

    Raises:
        TypeError: If ``array`` is a :class:`numpy.ma.MaskedArray`.
    """
    if isinstance(array, np.ma.MaskedArray):
        raise TypeError(
            f"{role} is a numpy.ma.MaskedArray, which pantr does not accept: the mask "
            f"would be discarded and the masked entries computed on anyway. Pass "
            f"`{role}.filled(fill_value)` to choose what those entries hold, or "
            f"`np.asarray({role})` to drop the mask deliberately."
        )


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

    Raises:
        TypeError: If ``pts`` is a masked array; see :func:`_reject_masked_array`.
    """
    _reject_masked_array(pts, "points")

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
    out: npt.NDArray[Any],
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
        TypeError: If ``out`` is a masked array; see :func:`_reject_masked_array`.
            An ``out`` is the worse half of that divergence -- the C++ backend
            writes through the mask into the caller's own array.
        ValueError: If the array shape, dtype, or writability does not match.
    """
    _reject_masked_array(out, "out")

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

    This helper is intentionally restricted to float arrays (``float32`` or
    ``float64``).  For non-float output arrays (``bool_``, ``int_``), use
    :func:`_validate_out_array` directly together with ``np.empty``.

    Args:
        out (npt.NDArray[np.float32 | np.float64] | None): Caller-provided
            output array, or ``None`` to allocate a new one.
        expected_shape (tuple[int, ...]): Required shape.
        expected_dtype (npt.DTypeLike): Required dtype (``float32`` or
            ``float64``).

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


def _reshaped_out(
    out: npt.NDArray[Any],
    shape: tuple[int, ...],
) -> tuple[npt.NDArray[Any], bool]:
    """Reshape a caller's ``out`` for a kernel, and say whether to copy back.

    Layer 2 normalises an ``out`` array's shape before handing it to a kernel that
    writes into it. ``reshape`` returns a *view* whenever the strides allow one, and
    a **copy** otherwise -- and a kernel writing into a copy leaves the caller's
    array untouched, so the result is silently discarded and the caller reads back
    whatever it allocated: a ``float64`` tabulation with an ``order="F"`` ``out`` and
    multi-dimensional points used to return the caller's array untouched, because the
    kernel had filled a copy of it.

    **The test is whether the reshape shared memory, not whether the array was
    C-contiguous**, and the difference is not academic: a strided slice such as
    ``big[..., ::2]`` is *not* C-contiguous and *does* reshape to a view, because
    merging the leading axes is expressible in strides. Refusing or copying it would
    slow down a case that works correctly today.

    The counterpart on the C++ side is
    :func:`pantr.bspline._extraction_backend._contiguous_out`, which absorbs the same
    situation for a different reason -- the bindings declare their writable arguments
    ``c_contig`` -- and states the shared policy: a strided ``out`` is rare and legal,
    so it is absorbed rather than refused.

    Args:
        out (npt.NDArray[Any]): The caller's output array, already validated.
        shape (tuple[int, ...]): The shape the kernel expects to write into.

    Returns:
        tuple[npt.NDArray[Any], bool]: The array to hand the kernel, and whether the
        caller's ``out`` still has to be written from it once the kernel returns.

    Raises:
        ValueError: If ``shape`` holds a different number of elements than ``out``.
            Call sites validate the shape first, so this cannot be reached from them.

    Example:
        >>> import numpy as np
        >>> out = np.zeros((2, 2, 3), order="F")
        >>> buf, copy_back = _reshaped_out(out, (4, 3))
        >>> copy_back
        True
        >>> buf[:] = 1.0
        >>> bool(out.any())
        False
        >>> out[...] = buf.reshape(out.shape)
        >>> bool(out.all())
        True
    """
    buffer = out.reshape(shape)
    return buffer, not np.shares_memory(buffer, out)
