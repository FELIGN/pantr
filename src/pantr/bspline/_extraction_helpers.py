"""Layer-2 helpers for tensor-product change-of-basis extraction.

Provides shape/dtype/writability validation, scratch sizing, ``out`` and
``scratch`` allocation, and kernel dispatch for the Layer-3 kernels in
:mod:`pantr.bspline._extraction_kernels`.

Four operation kinds are supported (see :data:`OP_KINDS`):

- ``"apply"``:    ``out = M @ v``           (input shape prod -> output shape prod)
- ``"apply_T"``:  ``out = M^T @ v``         (output shape prod -> input shape prod)
- ``"MT_K_M"``:   ``out = M^T @ K @ M``     (output x output -> input x input)
- ``"M_K_MT"``:   ``out = M @ K @ M^T``     (input x input -> output x output)

with ``M = kron(M_0, M_1, ..., M_{d-1})``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast, get_args

import numpy as np
import numpy.typing as npt

from ..basis._basis_utils import _allocate_or_validate_out
from ._extraction_kernels import (
    apply_kron_1d,
    apply_kron_2d,
    apply_kron_3d,
    apply_kron_apply_many_1d,
    apply_kron_apply_many_2d,
    apply_kron_apply_many_3d,
    apply_kron_apply_T_many_1d,
    apply_kron_apply_T_many_2d,
    apply_kron_apply_T_many_3d,
    apply_kron_M_K_MT_1d,
    apply_kron_M_K_MT_2d,
    apply_kron_M_K_MT_3d,
    apply_kron_M_K_MT_many_1d,
    apply_kron_M_K_MT_many_2d,
    apply_kron_M_K_MT_many_3d,
    apply_kron_MT_K_M_1d,
    apply_kron_MT_K_M_2d,
    apply_kron_MT_K_M_3d,
    apply_kron_MT_K_M_many_1d,
    apply_kron_MT_K_M_many_2d,
    apply_kron_MT_K_M_many_3d,
    apply_kron_T_1d,
    apply_kron_T_2d,
    apply_kron_T_3d,
)

OpKind = Literal["apply", "apply_T", "MT_K_M", "M_K_MT"]
"""Tag distinguishing the four apply variants.

See :mod:`pantr.bspline._extraction_helpers` for the meaning of each tag.
"""


MAX_SUPPORTED_DIM = 3
"""Highest tensor-product dimension for which specialized kernels exist."""


_KERNELS: dict[tuple[OpKind, int], Callable[..., None]] = {
    ("apply", 1): apply_kron_1d,
    ("apply", 2): apply_kron_2d,
    ("apply", 3): apply_kron_3d,
    ("apply_T", 1): apply_kron_T_1d,
    ("apply_T", 2): apply_kron_T_2d,
    ("apply_T", 3): apply_kron_T_3d,
    ("MT_K_M", 1): apply_kron_MT_K_M_1d,
    ("MT_K_M", 2): apply_kron_MT_K_M_2d,
    ("MT_K_M", 3): apply_kron_MT_K_M_3d,
    ("M_K_MT", 1): apply_kron_M_K_MT_1d,
    ("M_K_MT", 2): apply_kron_M_K_MT_2d,
    ("M_K_MT", 3): apply_kron_M_K_MT_3d,
}

_KERNELS_MANY: dict[tuple[OpKind, int], Callable[..., None]] = {
    ("apply", 1): apply_kron_apply_many_1d,
    ("apply", 2): apply_kron_apply_many_2d,
    ("apply", 3): apply_kron_apply_many_3d,
    ("apply_T", 1): apply_kron_apply_T_many_1d,
    ("apply_T", 2): apply_kron_apply_T_many_2d,
    ("apply_T", 3): apply_kron_apply_T_many_3d,
    ("MT_K_M", 1): apply_kron_MT_K_M_many_1d,
    ("MT_K_M", 2): apply_kron_MT_K_M_many_2d,
    ("MT_K_M", 3): apply_kron_MT_K_M_many_3d,
    ("M_K_MT", 1): apply_kron_M_K_MT_many_1d,
    ("M_K_MT", 2): apply_kron_M_K_MT_many_2d,
    ("M_K_MT", 3): apply_kron_M_K_MT_many_3d,
}


def _prod(shape: tuple[int, ...]) -> int:
    """Return the product of a shape tuple (1 for an empty tuple)."""
    result = 1
    for n in shape:
        result *= n
    return result


def _validate_op_kind(op_kind: str) -> OpKind:
    """Validate an operation-kind tag.

    Args:
        op_kind (str): Tag to validate.

    Returns:
        OpKind: The validated tag (narrowed to :data:`OpKind`).

    Raises:
        ValueError: If ``op_kind`` is not one of :data:`OpKind`'s literals.
    """
    if op_kind not in get_args(OpKind):
        valid = ", ".join(repr(v) for v in get_args(OpKind))
        raise ValueError(f"Unknown op_kind {op_kind!r}; expected one of {valid}")
    return cast(OpKind, op_kind)


def _apply_scratch_size(
    n_per_dir_in: tuple[int, ...],
    n_per_dir_out: tuple[int, ...],
) -> int:
    """Scratch size for an ``apply``-style vector kernel.

    Simulates the mode-contraction stages from ``n_per_dir_in`` to
    ``n_per_dir_out`` and returns twice the largest intermediate buffer size
    (one "half" per ping-pong buffer; in d=1 or d=2 only one half is used).

    Args:
        n_per_dir_in (tuple[int, ...]): Starting per-direction sizes.
        n_per_dir_out (tuple[int, ...]): Ending per-direction sizes.

    Returns:
        int: Required scratch size in elements (0 for d<=1).
    """
    d = len(n_per_dir_in)
    if d <= 1:
        return 0
    max_sz = 0
    for k in range(d - 1):
        sz = 1
        for j in range(k + 1):
            sz *= n_per_dir_out[j]
        for j in range(k + 1, d):
            sz *= n_per_dir_in[j]
        max_sz = max(max_sz, sz)
    return 2 * max_sz


def _bilateral_scratch_size(
    n_per_dir_initial: tuple[int, ...],
    n_per_dir_final: tuple[int, ...],
) -> int:
    """Scratch size for a bilateral matrix kernel.

    Simulates the interleaved (row, column) mode-contraction stages on the
    2d-axis reshape of the input matrix, going from per-direction sizes
    ``n_per_dir_initial`` to ``n_per_dir_final``. Returns twice the largest
    intermediate buffer size to cover a two-buffer ping-pong.

    Args:
        n_per_dir_initial (tuple[int, ...]): Starting per-direction sizes.
            The initial matrix is ``(prod, prod)``.
        n_per_dir_final (tuple[int, ...]): Ending per-direction sizes.

    Returns:
        int: Required scratch size in elements (0 for d<1).
    """
    d = len(n_per_dir_initial)
    if d < 1:
        return 0
    shape = list(n_per_dir_initial) + list(n_per_dir_initial)
    max_sz = 0
    total_stages = 2 * d
    for stage in range(total_stages):
        k = stage // 2
        axis = k if stage % 2 == 0 else d + k
        shape[axis] = n_per_dir_final[k]
        if stage == total_stages - 1:
            break  # final stage writes to out, not scratch
        sz = 1
        for s in shape:
            sz *= s
        max_sz = max(max_sz, sz)
    return 2 * max_sz


def _required_scratch_size(
    input_shape_per_dir: tuple[int, ...],
    output_shape_per_dir: tuple[int, ...],
    op_kind: OpKind,
) -> int:
    """Return the scratch buffer size required by the kernel for a single call.

    Computed by simulating the kernel's mode-contraction stages and taking
    twice the largest intermediate buffer (to hold two ping-pong halves).
    The returned size is sufficient for any identity-flag pattern.

    For ``d = 1`` and kind ``apply``/``apply_T`` the kernel does not use
    scratch and the returned size is 0.

    Args:
        input_shape_per_dir (tuple[int, ...]): Per-direction input sizes
            (``n_in_k``).
        output_shape_per_dir (tuple[int, ...]): Per-direction output sizes
            (``n_out_k``); must have the same length as ``input_shape_per_dir``.
        op_kind (OpKind): Operation kind.

    Returns:
        int: Required scratch size in array elements.

    Raises:
        ValueError: If the two shape tuples have different lengths or
            ``op_kind`` is invalid.
    """
    _validate_op_kind(op_kind)
    if len(input_shape_per_dir) != len(output_shape_per_dir):
        raise ValueError(
            "input_shape_per_dir and output_shape_per_dir must have the same length; "
            f"got {len(input_shape_per_dir)} and {len(output_shape_per_dir)}"
        )
    if op_kind == "apply":
        return _apply_scratch_size(input_shape_per_dir, output_shape_per_dir)
    if op_kind == "apply_T":
        return _apply_scratch_size(output_shape_per_dir, input_shape_per_dir)
    if op_kind == "MT_K_M":
        return _bilateral_scratch_size(output_shape_per_dir, input_shape_per_dir)
    # op_kind == "M_K_MT"
    return _bilateral_scratch_size(input_shape_per_dir, output_shape_per_dir)


def _dispatch_apply_many(d: int, op_kind: OpKind) -> Callable[..., None]:
    """Return the batch Layer-3 kernel for a given dimension and operation kind.

    Args:
        d (int): Number of tensor-product directions.
        op_kind (OpKind): Operation kind.

    Returns:
        Callable[..., None]: The corresponding ``@njit(parallel=True)`` kernel.

    Raises:
        NotImplementedError: If ``d`` is not in ``{1, 2, 3}``.
        ValueError: If ``op_kind`` is invalid.
    """
    _validate_op_kind(op_kind)
    if d < 1 or d > MAX_SUPPORTED_DIM:
        raise NotImplementedError(
            f"Extraction kernels are specialized for d in {{1, 2, 3}}; got d={d}. "
            "Add a generic-d kernel in pantr.bspline._extraction_kernels to support "
            "higher dimensions."
        )
    return _KERNELS_MANY[(op_kind, d)]


def _dispatch_apply(d: int, op_kind: OpKind) -> Callable[..., None]:
    """Return the Layer-3 kernel for a given dimension and operation kind.

    Args:
        d (int): Number of tensor-product directions.
        op_kind (OpKind): Operation kind.

    Returns:
        Callable[..., None]: The corresponding ``@njit`` kernel.

    Raises:
        NotImplementedError: If ``d`` is not in ``{1, 2, 3}``.
        ValueError: If ``op_kind`` is invalid.
    """
    _validate_op_kind(op_kind)
    if d < 1 or d > MAX_SUPPORTED_DIM:
        raise NotImplementedError(
            f"Extraction kernels are specialized for d in {{1, 2, 3}}; got d={d}. "
            "Add a generic-d kernel in pantr.bspline._extraction_kernels to support "
            "higher dimensions."
        )
    return _KERNELS[(op_kind, d)]


def _operation_shapes(
    input_shape_per_dir: tuple[int, ...],
    output_shape_per_dir: tuple[int, ...],
    op_kind: OpKind,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(input_shape, output_shape)`` for a given operation kind.

    For vector apply variants the shapes are 1D ``(N,)`` tuples. For bilateral
    variants they are 2D ``(N, N)`` tuples. ``N`` is the product of the
    corresponding per-direction sizes.

    Args:
        input_shape_per_dir (tuple[int, ...]): Per-direction input sizes.
        output_shape_per_dir (tuple[int, ...]): Per-direction output sizes.
        op_kind (OpKind): Operation kind.

    Returns:
        tuple[tuple[int, ...], tuple[int, ...]]: ``(input_shape, output_shape)``
        of the operand and result arrays.
    """
    _validate_op_kind(op_kind)
    n_in = _prod(input_shape_per_dir)
    n_out = _prod(output_shape_per_dir)
    if op_kind == "apply":
        return (n_in,), (n_out,)
    if op_kind == "apply_T":
        return (n_out,), (n_in,)
    if op_kind == "MT_K_M":
        return (n_out, n_out), (n_in, n_in)
    # op_kind == "M_K_MT"
    return (n_in, n_in), (n_out, n_out)


def _validate_operand(
    operand: np.ndarray,  # type: ignore[type-arg]
    expected_shape: tuple[int, ...],
    dtype: npt.DTypeLike,
) -> None:
    """Validate shape and dtype of an input operand.

    Args:
        operand (np.ndarray): Input array to check.
        expected_shape (tuple[int, ...]): Required shape.
        dtype (npt.DTypeLike): Required dtype.

    Raises:
        ValueError: If the operand shape or dtype does not match.
    """
    if operand.shape != expected_shape:
        raise ValueError(f"Operand has shape {operand.shape}, but expected shape {expected_shape}")
    if operand.dtype != np.dtype(dtype):
        raise ValueError(f"Operand has dtype {operand.dtype}, but expected dtype {np.dtype(dtype)}")


def _allocate_or_validate_scratch(
    scratch: npt.NDArray[np.float32 | np.float64] | None,
    required_size: int,
    dtype: npt.DTypeLike,
) -> npt.NDArray[np.float32 | np.float64]:
    """Allocate a fresh scratch buffer or validate the user-provided one.

    A user-provided buffer is accepted if it is at least ``required_size``
    elements long (it may be larger; only the first ``required_size`` elements
    are used), has the expected dtype, is writeable, and 1D.

    Args:
        scratch (npt.NDArray[np.float32 | np.float64] | None): Caller-provided
            scratch buffer, or ``None`` to allocate.
        required_size (int): Minimum number of elements required.
        dtype (npt.DTypeLike): Required dtype.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The validated or freshly
        allocated scratch buffer.

    Raises:
        ValueError: If ``scratch`` is provided with wrong dtype, wrong ndim,
            is too small, or is not writeable.
    """
    if scratch is None:
        return np.empty(required_size, dtype=dtype)
    if scratch.ndim != 1:
        raise ValueError(f"Scratch must be 1D; got ndim={scratch.ndim}")
    if scratch.dtype != np.dtype(dtype):
        raise ValueError(f"Scratch has dtype {scratch.dtype}, but expected dtype {np.dtype(dtype)}")
    if scratch.size < required_size:
        raise ValueError(f"Scratch size {scratch.size} is smaller than required {required_size}")
    if not scratch.flags.writeable:
        raise ValueError("Scratch array is not writeable")
    return scratch


def _prepare_apply_call(  # noqa: PLR0913 -- each arg reflects a distinct kernel input.
    ops_1d_per_cell: tuple[npt.NDArray[np.float32 | np.float64], ...],
    is_identity_per_dir: tuple[bool, ...],
    operand: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64] | None,
    scratch: npt.NDArray[np.float32 | np.float64] | None,
    op_kind: OpKind,
) -> tuple[
    Callable[..., None],
    tuple[Any, ...],
    npt.NDArray[np.float32 | np.float64],
]:
    """Validate inputs and build the kernel call for a single cell.

    Args:
        ops_1d_per_cell (tuple[npt.NDArray[np.float32 | np.float64], ...]):
            Per-direction 2D operators ``M_k`` already selected for the cell,
            each of shape ``(n_out_k, n_in_k)``. Length must equal the number
            of directions ``d``.
        is_identity_per_dir (tuple[bool, ...]): Per-direction identity flags.
            Must have length ``d``.
        operand (npt.NDArray[np.float32 | np.float64]): Input vector (for
            ``apply``/``apply_T``) or matrix (for ``MT_K_M``/``M_K_MT``).
        out (npt.NDArray[np.float32 | np.float64] | None): Output array, or
            ``None`` to allocate. Same dtype as the operators.
        scratch (npt.NDArray[np.float32 | np.float64] | None): Scratch
            buffer, or ``None`` to allocate.
        op_kind (OpKind): Operation kind.

    Returns:
        tuple[Callable, tuple, npt.NDArray]: A 3-tuple ``(kernel, args, out)``
        where ``kernel(*args)`` performs the requested operation, writing into
        ``out``. ``out`` is the same array the caller can return.

    Raises:
        ValueError: If shapes, dtypes, writability, or length invariants fail.
        NotImplementedError: If ``d > 3``.
    """
    _validate_op_kind(op_kind)
    d = len(ops_1d_per_cell)
    if len(is_identity_per_dir) != d:
        raise ValueError(
            f"is_identity_per_dir has length {len(is_identity_per_dir)}, "
            f"expected {d} to match ops_1d_per_cell"
        )
    if d < 1:
        raise ValueError("At least one direction is required")
    for k, M_k in enumerate(ops_1d_per_cell):
        if M_k.ndim != 2:  # noqa: PLR2004
            raise ValueError(f"ops_1d_per_cell[{k}] must be 2D; got ndim={M_k.ndim}")
    dtype = ops_1d_per_cell[0].dtype
    for k, M_k in enumerate(ops_1d_per_cell[1:], start=1):
        if M_k.dtype != dtype:
            raise ValueError(
                f"ops_1d_per_cell[{k}] dtype {M_k.dtype} differs from ops_1d_per_cell[0] "
                f"dtype {dtype}"
            )

    input_shape_per_dir = tuple(int(M_k.shape[1]) for M_k in ops_1d_per_cell)
    output_shape_per_dir = tuple(int(M_k.shape[0]) for M_k in ops_1d_per_cell)
    expected_in_shape, expected_out_shape = _operation_shapes(
        input_shape_per_dir, output_shape_per_dir, op_kind
    )
    _validate_operand(operand, expected_in_shape, dtype)
    out = _allocate_or_validate_out(out, expected_out_shape, dtype)

    scratch_size = _required_scratch_size(input_shape_per_dir, output_shape_per_dir, op_kind)
    scratch = _allocate_or_validate_scratch(scratch, scratch_size, dtype)

    kernel = _dispatch_apply(d, op_kind)

    # Kernel signature: (M_0, [M_1, [M_2,]], is_id_0, [is_id_1, [is_id_2,]], operand, out, scratch)
    args: tuple[Any, ...] = (
        *ops_1d_per_cell,
        *is_identity_per_dir,
        operand,
        out,
        scratch,
    )
    return kernel, args, out


def _allocate_or_validate_scratch_many(
    scratch: npt.NDArray[np.float32 | np.float64] | None,
    n_cells: int,
    scratch_size_per_cell: int,
    dtype: npt.DTypeLike,
) -> npt.NDArray[np.float32 | np.float64]:
    """Allocate or validate the per-cell scratch buffer for a batch call.

    The scratch array has shape ``(n_cells, max(scratch_size_per_cell, 1))``
    so that ``scratch[cell]`` is always a non-empty 1D view (even when the
    per-cell kernel does not use scratch).

    A user-provided buffer is accepted when it has the correct dtype, is
    writeable, is 2D, has ``shape[0] == n_cells``, and
    ``shape[1] >= scratch_size_per_cell``.

    Args:
        scratch (npt.NDArray[np.float32 | np.float64] | None): Caller-provided
            scratch array, or ``None`` to allocate.
        n_cells (int): Number of cells in the batch.
        scratch_size_per_cell (int): Minimum scratch elements required per cell
            (may be 0 for d=1 vector kernels).
        dtype (npt.DTypeLike): Required dtype.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The validated or freshly
        allocated 2D scratch array.

    Raises:
        ValueError: If a provided ``scratch`` has wrong dtype, wrong ndim,
            wrong ``shape[0]``, is too narrow, or is not writeable.
    """
    alloc_width = max(scratch_size_per_cell, 1)
    if scratch is None:
        return np.empty((n_cells, alloc_width), dtype=dtype)
    if scratch.ndim != 2:  # noqa: PLR2004
        raise ValueError(f"Batch scratch must be 2D; got ndim={scratch.ndim}")
    if scratch.dtype != np.dtype(dtype):
        raise ValueError(f"Batch scratch has dtype {scratch.dtype}, but expected {np.dtype(dtype)}")
    if scratch.shape[0] != n_cells:
        raise ValueError(
            f"Batch scratch shape[0]={scratch.shape[0]} does not match n_cells={n_cells}"
        )
    if scratch.shape[1] < scratch_size_per_cell:
        raise ValueError(
            f"Batch scratch shape[1]={scratch.shape[1]} is smaller than "
            f"required scratch_size_per_cell={scratch_size_per_cell}"
        )
    if not scratch.flags.writeable:
        raise ValueError("Batch scratch array is not writeable")
    return scratch  # type: ignore[return-value]


def _prepare_apply_many_call(  # noqa: PLR0913
    ops_1d: tuple[npt.NDArray[np.float32 | np.float64], ...],
    is_identity_masks: tuple[npt.NDArray[np.bool_], ...],
    cell_indices: npt.NDArray[np.intp],
    operand: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64] | None,
    scratch: npt.NDArray[np.float32 | np.float64] | None,
    op_kind: OpKind,
) -> tuple[
    Callable[..., None],
    tuple[Any, ...],
    npt.NDArray[np.float32 | np.float64],
]:
    """Validate inputs and build the kernel call for a batch of cells.

    Args:
        ops_1d (tuple[npt.NDArray[np.float32 | np.float64], ...]): Full per-direction
            3D operator arrays ``(n_el_k, n_out_k, n_in_k)``. Length must equal ``d``.
        is_identity_masks (tuple[npt.NDArray[np.bool_], ...]): Full per-direction
            identity mask arrays of shape ``(n_el_k,)``. Length must equal ``d``.
        cell_indices (npt.NDArray[np.intp]): Per-direction element indices,
            shape ``(n_cells, d)``. Values must be non-negative and within the
            per-direction element counts from ``ops_1d``.
        operand (npt.NDArray[np.float32 | np.float64]): Batch input array.
            Shape ``(n_cells, N_in)`` for vector kinds or
            ``(n_cells, N, N)`` for bilateral kinds.
        out (npt.NDArray[np.float32 | np.float64] | None): Batch output array,
            or ``None`` to allocate.
        scratch (npt.NDArray[np.float32 | np.float64] | None): Per-cell scratch
            array of shape ``(n_cells, s)`` with ``s >= scratch_size_per_cell``,
            or ``None`` to allocate.
        op_kind (OpKind): Operation kind.

    Returns:
        tuple[Callable, tuple, npt.NDArray]: ``(kernel, args, out)`` where
        ``kernel(*args)`` runs the batch operation, writing into ``out``.

    Raises:
        ValueError: If shapes, dtypes, writability, or index-range invariants fail.
        NotImplementedError: If ``d > 3``.
    """
    _validate_op_kind(op_kind)
    d = len(ops_1d)
    if len(is_identity_masks) != d:
        raise ValueError(f"is_identity_masks has length {len(is_identity_masks)}, expected {d}")
    if d < 1:
        raise ValueError("At least one direction is required")
    for k, op in enumerate(ops_1d):
        if op.ndim != 3:  # noqa: PLR2004
            raise ValueError(f"ops_1d[{k}] must be 3D; got ndim={op.ndim}")
    dtype = ops_1d[0].dtype
    for k, op in enumerate(ops_1d[1:], start=1):
        if op.dtype != dtype:
            raise ValueError(f"ops_1d[{k}] dtype {op.dtype} differs from ops_1d[0] dtype {dtype}")

    if cell_indices.ndim != 2:  # noqa: PLR2004
        raise ValueError(f"cell_indices must be 2D; got ndim={cell_indices.ndim}")
    if cell_indices.shape[1] != d:
        raise ValueError(f"cell_indices.shape[1]={cell_indices.shape[1]} does not match d={d}")
    n_cells = cell_indices.shape[0]
    if n_cells > 0:
        for k, op in enumerate(ops_1d):
            n_el_k = op.shape[0]
            col = cell_indices[:, k]
            if int(col.min()) < 0 or int(col.max()) >= n_el_k:
                raise IndexError(f"cell_indices[:, {k}] contains values outside [0, {n_el_k})")

    input_shape_per_dir = tuple(int(op.shape[2]) for op in ops_1d)
    output_shape_per_dir = tuple(int(op.shape[1]) for op in ops_1d)
    per_cell_in, per_cell_out = _operation_shapes(
        input_shape_per_dir, output_shape_per_dir, op_kind
    )
    expected_operand = (n_cells, *per_cell_in)
    expected_out_shape = (n_cells, *per_cell_out)
    _validate_operand(operand, expected_operand, dtype)
    out = _allocate_or_validate_out(out, expected_out_shape, dtype)

    scratch_size = _required_scratch_size(input_shape_per_dir, output_shape_per_dir, op_kind)
    scratch = _allocate_or_validate_scratch_many(scratch, n_cells, scratch_size, dtype)

    kernel = _dispatch_apply_many(d, op_kind)
    args: tuple[Any, ...] = (*ops_1d, *is_identity_masks, cell_indices, operand, out, scratch)
    return kernel, args, out


__all__ = [
    "MAX_SUPPORTED_DIM",
    "OpKind",
    "_allocate_or_validate_scratch",
    "_allocate_or_validate_scratch_many",
    "_dispatch_apply",
    "_dispatch_apply_many",
    "_operation_shapes",
    "_prepare_apply_call",
    "_prepare_apply_many_call",
    "_required_scratch_size",
    "_validate_op_kind",
    "_validate_operand",
]
