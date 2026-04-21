"""Numba-callable Kronecker contraction kernels for change-of-basis extraction.

This module provides Layer-3 (Numba) primitives that apply tensor-product
change-of-basis operators ``M = kron(M_0, M_1, ..., M_{d-1})`` to vectors and
matrices without ever materializing the full Kronecker product.

**Per-cell kernels** (single cell, no ``parallel``; callable from outer prange):

- ``apply_kron_{d}d``:        ``out = M @ v``
- ``apply_kron_T_{d}d``:      ``out = M^T @ v``
- ``apply_kron_MT_K_M_{d}d``: ``out = M^T @ K @ M``
- ``apply_kron_M_K_MT_{d}d``: ``out = M @ K @ M^T``

**Batch kernels** (``parallel=True``; ``prange`` over cells):

- ``apply_kron_apply_many_{d}d``:    ``out[c] = M_c @ v[c]``
- ``apply_kron_apply_T_many_{d}d``:  ``out[c] = M_c^T @ v[c]``
- ``apply_kron_MT_K_M_many_{d}d``:   ``out[c] = M_c^T @ K[c] @ M_c``
- ``apply_kron_M_K_MT_many_{d}d``:   ``out[c] = M_c @ K[c] @ M_c^T``

All kernels are specialized for ``d in {1, 2, 3}``.

Identity short-circuit: when ``is_id_k`` is True, the mode-k contraction is
skipped (the axis is passed through unchanged, and ``M_k`` is not read). When
all directions are identity, the kernel degenerates to a direct copy
``out[:] = input[:]``, which is aliasing-safe -- the caller may pass the same
array as input and output.

Per-cell kernels are designed to be callable from other ``@njit`` code: they are
module-level free functions with plain NumPy-array arguments, no optional
parameters, and ``cache=True``. The batch kernels use ``parallel=True`` and
dispatch to the per-cell kernels inside the ``prange`` body.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .._numba_compat import nb_jit, nb_prange


@nb_jit(nopython=True, cache=True)
def apply_kron_1d(
    M_0: npt.NDArray[np.float32 | np.float64],
    is_id_0: bool,
    v: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
    scratch: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Compute ``out = M_0 @ v`` with identity short-circuit.

    Args:
        M_0 (npt.NDArray[np.float32 | np.float64]): Operator, shape
            ``(n_out_0, n_in_0)``.
        is_id_0 (bool): If True, ``M_0`` is treated as identity and not read;
            ``out`` is set equal to ``v`` (aliasing-safe).
        v (npt.NDArray[np.float32 | np.float64]): Input vector, shape
            ``(n_in_0,)``.
        out (npt.NDArray[np.float32 | np.float64]): Output vector, shape
            ``(n_out_0,)``.
        scratch (npt.NDArray[np.float32 | np.float64]): Unused for d=1; pass a
            zero-size buffer or a dummy array.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    _ = scratch  # unused for d=1; kept for signature uniformity across d.
    n_out_0 = out.shape[0]
    if is_id_0:
        for i in range(n_out_0):
            out[i] = v[i]
        return

    n_in_0 = v.shape[0]
    zero = M_0.dtype.type(0.0)
    for i in range(n_out_0):
        s = zero
        for k in range(n_in_0):
            s += M_0[i, k] * v[k]
        out[i] = s


@nb_jit(nopython=True, cache=True)
def apply_kron_2d(  # noqa: PLR0913, PLR0912 -- kernel fan-in and identity branching are intentional.
    M_0: npt.NDArray[np.float32 | np.float64],
    M_1: npt.NDArray[np.float32 | np.float64],
    is_id_0: bool,
    is_id_1: bool,
    v: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
    scratch: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Compute ``out = kron(M_0, M_1) @ v`` via mode-wise contractions.

    Args:
        M_0 (npt.NDArray[np.float32 | np.float64]): Direction-0 operator,
            shape ``(n_out_0, n_in_0)``.
        M_1 (npt.NDArray[np.float32 | np.float64]): Direction-1 operator,
            shape ``(n_out_1, n_in_1)``.
        is_id_0 (bool): Identity flag for direction 0.
        is_id_1 (bool): Identity flag for direction 1.
        v (npt.NDArray[np.float32 | np.float64]): Input vector, shape
            ``(n_in_0 * n_in_1,)``.
        out (npt.NDArray[np.float32 | np.float64]): Output vector, shape
            ``(n_out_0 * n_out_1,)``.
        scratch (npt.NDArray[np.float32 | np.float64]): Work buffer of size at
            least ``n_out_0 * n_in_1`` when neither direction is identity;
            unused when any direction is identity.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    n_out_0 = M_0.shape[0]
    n_in_0 = M_0.shape[1]
    n_out_1 = M_1.shape[0]
    n_in_1 = M_1.shape[1]

    if is_id_0 and is_id_1:
        total = n_out_0 * n_out_1
        for i in range(total):
            out[i] = v[i]
        return

    zero = M_0.dtype.type(0.0)

    if is_id_0:
        v_view = v.reshape(n_out_0, n_in_1)
        out_view = out.reshape(n_out_0, n_out_1)
        for i in range(n_out_0):
            for j in range(n_out_1):
                s = zero
                for k in range(n_in_1):
                    s += M_1[j, k] * v_view[i, k]
                out_view[i, j] = s
        return

    if is_id_1:
        v_view = v.reshape(n_in_0, n_out_1)
        out_view = out.reshape(n_out_0, n_out_1)
        for i in range(n_out_0):
            for j in range(n_out_1):
                s = zero
                for k in range(n_in_0):
                    s += M_0[i, k] * v_view[k, j]
                out_view[i, j] = s
        return

    v_view = v.reshape(n_in_0, n_in_1)
    s_view = scratch[: n_out_0 * n_in_1].reshape(n_out_0, n_in_1)
    out_view = out.reshape(n_out_0, n_out_1)
    for i in range(n_out_0):
        for j in range(n_in_1):
            s = zero
            for k in range(n_in_0):
                s += M_0[i, k] * v_view[k, j]
            s_view[i, j] = s
    for i in range(n_out_0):
        for j in range(n_out_1):
            s = zero
            for k in range(n_in_1):
                s += s_view[i, k] * M_1[j, k]
            out_view[i, j] = s


@nb_jit(nopython=True, cache=True)
def apply_kron_3d(  # noqa: PLR0913, PLR0912 -- kernel fan-in and identity branching are intentional.
    M_0: npt.NDArray[np.float32 | np.float64],
    M_1: npt.NDArray[np.float32 | np.float64],
    M_2: npt.NDArray[np.float32 | np.float64],
    is_id_0: bool,
    is_id_1: bool,
    is_id_2: bool,
    v: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
    scratch: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Compute ``out = kron(M_0, M_1, M_2) @ v`` via mode-wise contractions.

    Args:
        M_0 (npt.NDArray[np.float32 | np.float64]): Direction-0 operator,
            shape ``(n_out_0, n_in_0)``.
        M_1 (npt.NDArray[np.float32 | np.float64]): Direction-1 operator,
            shape ``(n_out_1, n_in_1)``.
        M_2 (npt.NDArray[np.float32 | np.float64]): Direction-2 operator,
            shape ``(n_out_2, n_in_2)``.
        is_id_0 (bool): Identity flag for direction 0.
        is_id_1 (bool): Identity flag for direction 1.
        is_id_2 (bool): Identity flag for direction 2.
        v (npt.NDArray[np.float32 | np.float64]): Input vector, shape
            ``(n_in_0 * n_in_1 * n_in_2,)``.
        out (npt.NDArray[np.float32 | np.float64]): Output vector, shape
            ``(n_out_0 * n_out_1 * n_out_2,)``.
        scratch (npt.NDArray[np.float32 | np.float64]): Work buffer. Must be
            sized per :func:`pantr.bspline._extraction_helpers._required_scratch_size`
            for the ``"apply"`` kind; pessimistic size is
            ``2 * max(prod(input_shape), prod(output_shape))``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    n_out_0 = M_0.shape[0]
    n_in_0 = M_0.shape[1]
    n_out_1 = M_1.shape[0]
    n_in_1 = M_1.shape[1]
    n_out_2 = M_2.shape[0]
    n_in_2 = M_2.shape[1]

    # Active axis sizes after each stage: identity directions pass through with
    # n_in_k == n_out_k, so we use n_in_k for non-identity and n_out_k for
    # identity (picking the value that is actually present along that axis).
    d1_in = n_in_1 if not is_id_1 else n_out_1
    d2_in = n_in_2 if not is_id_2 else n_out_2

    zero = M_0.dtype.type(0.0)

    if is_id_0 and is_id_1 and is_id_2:
        total = n_out_0 * n_out_1 * n_out_2
        for i in range(total):
            out[i] = v[i]
        return

    # Ping-pong between two scratch halves. Each half sized at the pessimistic
    # bound 2 * max_intermediate; we only ever need one "current" and one
    # "next" intermediate buffer.
    half = scratch.size // 2
    buf_a = scratch[:half]
    buf_b = scratch[half:]

    # Stage 0: v -> cur, applying M_0 on axis 0 if not identity.
    # After stage 0, the tensor has shape (d0_cur, d1_in, d2_in) where
    # d0_cur = n_out_0 if we contracted, else n_in_0 = n_out_0 (still the same).
    # In both cases the output axis-0 size is n_out_0.
    if is_id_0:
        # No contraction; cur view is just v.
        cur = v.reshape(n_out_0, d1_in, d2_in)
    else:
        v_view = v.reshape(n_in_0, d1_in, d2_in)
        cur = buf_a[: n_out_0 * d1_in * d2_in].reshape(n_out_0, d1_in, d2_in)
        for i in range(n_out_0):
            for j in range(d1_in):
                for k in range(d2_in):
                    s = zero
                    for m in range(n_in_0):
                        s += M_0[i, m] * v_view[m, j, k]
                    cur[i, j, k] = s

    # Stage 1: apply M_1 on axis 1 if not identity; shape becomes
    # (n_out_0, n_out_1, d2_in).
    if is_id_1:
        # Pass-through; cur shape already (n_out_0, n_out_1, d2_in).
        pass
    else:
        nxt = buf_b[: n_out_0 * n_out_1 * d2_in].reshape(n_out_0, n_out_1, d2_in)
        for i in range(n_out_0):
            for j in range(n_out_1):
                for k in range(d2_in):
                    s = zero
                    for m in range(n_in_1):
                        s += M_1[j, m] * cur[i, m, k]
                    nxt[i, j, k] = s
        cur = nxt

    # Stage 2: apply M_2 on axis 2 (write directly into out) if not identity;
    # otherwise copy cur into out.
    out_view = out.reshape(n_out_0, n_out_1, n_out_2)
    if is_id_2:
        for i in range(n_out_0):
            for j in range(n_out_1):
                for k in range(n_out_2):
                    out_view[i, j, k] = cur[i, j, k]
    else:
        for i in range(n_out_0):
            for j in range(n_out_1):
                for k in range(n_out_2):
                    s = zero
                    for m in range(n_in_2):
                        s += M_2[k, m] * cur[i, j, m]
                    out_view[i, j, k] = s


@nb_jit(nopython=True, cache=True)
def apply_kron_T_1d(
    M_0: npt.NDArray[np.float32 | np.float64],
    is_id_0: bool,
    v: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
    scratch: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Compute ``out = M_0^T @ v`` with identity short-circuit.

    Args:
        M_0 (npt.NDArray[np.float32 | np.float64]): Operator, shape
            ``(n_out_0, n_in_0)``.
        is_id_0 (bool): If True, ``out`` is set equal to ``v``.
        v (npt.NDArray[np.float32 | np.float64]): Input vector, shape
            ``(n_out_0,)``.
        out (npt.NDArray[np.float32 | np.float64]): Output vector, shape
            ``(n_in_0,)``.
        scratch (npt.NDArray[np.float32 | np.float64]): Unused for d=1.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    _ = scratch  # unused for d=1; kept for signature uniformity across d.
    n_out_0 = M_0.shape[0]
    n_in_0 = M_0.shape[1]
    if is_id_0:
        for i in range(n_in_0):
            out[i] = v[i]
        return

    zero = M_0.dtype.type(0.0)
    for j in range(n_in_0):
        s = zero
        for k in range(n_out_0):
            s += M_0[k, j] * v[k]
        out[j] = s


@nb_jit(nopython=True, cache=True)
def apply_kron_T_2d(  # noqa: PLR0913, PLR0912 -- kernel fan-in and identity branching are intentional.
    M_0: npt.NDArray[np.float32 | np.float64],
    M_1: npt.NDArray[np.float32 | np.float64],
    is_id_0: bool,
    is_id_1: bool,
    v: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
    scratch: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Compute ``out = kron(M_0, M_1)^T @ v`` via mode-wise contractions.

    Args:
        M_0 (npt.NDArray[np.float32 | np.float64]): Direction-0 operator,
            shape ``(n_out_0, n_in_0)``.
        M_1 (npt.NDArray[np.float32 | np.float64]): Direction-1 operator,
            shape ``(n_out_1, n_in_1)``.
        is_id_0 (bool): Identity flag for direction 0.
        is_id_1 (bool): Identity flag for direction 1.
        v (npt.NDArray[np.float32 | np.float64]): Input vector, shape
            ``(n_out_0 * n_out_1,)``.
        out (npt.NDArray[np.float32 | np.float64]): Output vector, shape
            ``(n_in_0 * n_in_1,)``.
        scratch (npt.NDArray[np.float32 | np.float64]): Work buffer of size at
            least ``n_in_0 * n_out_1`` when neither direction is identity.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    n_out_0 = M_0.shape[0]
    n_in_0 = M_0.shape[1]
    n_out_1 = M_1.shape[0]
    n_in_1 = M_1.shape[1]

    if is_id_0 and is_id_1:
        total = n_in_0 * n_in_1
        for i in range(total):
            out[i] = v[i]
        return

    zero = M_0.dtype.type(0.0)

    if is_id_0:
        v_view = v.reshape(n_in_0, n_out_1)
        out_view = out.reshape(n_in_0, n_in_1)
        for i in range(n_in_0):
            for j in range(n_in_1):
                s = zero
                for k in range(n_out_1):
                    s += M_1[k, j] * v_view[i, k]
                out_view[i, j] = s
        return

    if is_id_1:
        v_view = v.reshape(n_out_0, n_in_1)
        out_view = out.reshape(n_in_0, n_in_1)
        for i in range(n_in_0):
            for j in range(n_in_1):
                s = zero
                for k in range(n_out_0):
                    s += M_0[k, i] * v_view[k, j]
                out_view[i, j] = s
        return

    v_view = v.reshape(n_out_0, n_out_1)
    s_view = scratch[: n_in_0 * n_out_1].reshape(n_in_0, n_out_1)
    out_view = out.reshape(n_in_0, n_in_1)
    for i in range(n_in_0):
        for j in range(n_out_1):
            s = zero
            for k in range(n_out_0):
                s += M_0[k, i] * v_view[k, j]
            s_view[i, j] = s
    for i in range(n_in_0):
        for j in range(n_in_1):
            s = zero
            for k in range(n_out_1):
                s += s_view[i, k] * M_1[k, j]
            out_view[i, j] = s


@nb_jit(nopython=True, cache=True)
def apply_kron_T_3d(  # noqa: PLR0913, PLR0912 -- kernel fan-in and identity branching are intentional.
    M_0: npt.NDArray[np.float32 | np.float64],
    M_1: npt.NDArray[np.float32 | np.float64],
    M_2: npt.NDArray[np.float32 | np.float64],
    is_id_0: bool,
    is_id_1: bool,
    is_id_2: bool,
    v: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
    scratch: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Compute ``out = kron(M_0, M_1, M_2)^T @ v`` via mode-wise contractions.

    Args:
        M_0 (npt.NDArray[np.float32 | np.float64]): Direction-0 operator,
            shape ``(n_out_0, n_in_0)``.
        M_1 (npt.NDArray[np.float32 | np.float64]): Direction-1 operator,
            shape ``(n_out_1, n_in_1)``.
        M_2 (npt.NDArray[np.float32 | np.float64]): Direction-2 operator,
            shape ``(n_out_2, n_in_2)``.
        is_id_0 (bool): Identity flag for direction 0.
        is_id_1 (bool): Identity flag for direction 1.
        is_id_2 (bool): Identity flag for direction 2.
        v (npt.NDArray[np.float32 | np.float64]): Input vector, shape
            ``(n_out_0 * n_out_1 * n_out_2,)``.
        out (npt.NDArray[np.float32 | np.float64]): Output vector, shape
            ``(n_in_0 * n_in_1 * n_in_2,)``.
        scratch (npt.NDArray[np.float32 | np.float64]): Work buffer; see
            :func:`pantr.bspline._extraction_helpers._required_scratch_size`
            for the ``"apply_T"`` kind.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    n_out_0 = M_0.shape[0]
    n_in_0 = M_0.shape[1]
    n_out_1 = M_1.shape[0]
    n_in_1 = M_1.shape[1]
    n_out_2 = M_2.shape[0]
    n_in_2 = M_2.shape[1]

    d1_out = n_out_1 if not is_id_1 else n_in_1
    d2_out = n_out_2 if not is_id_2 else n_in_2

    zero = M_0.dtype.type(0.0)

    if is_id_0 and is_id_1 and is_id_2:
        total = n_in_0 * n_in_1 * n_in_2
        for i in range(total):
            out[i] = v[i]
        return

    half = scratch.size // 2
    buf_a = scratch[:half]
    buf_b = scratch[half:]

    # Stage 0: apply M_0^T on axis 0. Axis-0 size becomes n_in_0 = n_out_0 for identity.
    if is_id_0:
        cur = v.reshape(n_in_0, d1_out, d2_out)
    else:
        v_view = v.reshape(n_out_0, d1_out, d2_out)
        cur = buf_a[: n_in_0 * d1_out * d2_out].reshape(n_in_0, d1_out, d2_out)
        for i in range(n_in_0):
            for j in range(d1_out):
                for k in range(d2_out):
                    s = zero
                    for m in range(n_out_0):
                        s += M_0[m, i] * v_view[m, j, k]
                    cur[i, j, k] = s

    # Stage 1: apply M_1^T on axis 1.
    if is_id_1:
        pass
    else:
        nxt = buf_b[: n_in_0 * n_in_1 * d2_out].reshape(n_in_0, n_in_1, d2_out)
        for i in range(n_in_0):
            for j in range(n_in_1):
                for k in range(d2_out):
                    s = zero
                    for m in range(n_out_1):
                        s += M_1[m, j] * cur[i, m, k]
                    nxt[i, j, k] = s
        cur = nxt

    # Stage 2: apply M_2^T on axis 2, write to out.
    out_view = out.reshape(n_in_0, n_in_1, n_in_2)
    if is_id_2:
        for i in range(n_in_0):
            for j in range(n_in_1):
                for k in range(n_in_2):
                    out_view[i, j, k] = cur[i, j, k]
    else:
        for i in range(n_in_0):
            for j in range(n_in_1):
                for k in range(n_in_2):
                    s = zero
                    for m in range(n_out_2):
                        s += M_2[m, k] * cur[i, j, m]
                    out_view[i, j, k] = s


@nb_jit(nopython=True, cache=True)
def apply_kron_MT_K_M_1d(
    M_0: npt.NDArray[np.float32 | np.float64],
    is_id_0: bool,
    K: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
    scratch: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Compute ``out = M_0^T @ K @ M_0`` with identity short-circuit.

    Args:
        M_0 (npt.NDArray[np.float32 | np.float64]): Operator, shape
            ``(n_out_0, n_in_0)``.
        is_id_0 (bool): If True, ``out`` is set equal to ``K``.
        K (npt.NDArray[np.float32 | np.float64]): Input matrix, shape
            ``(n_out_0, n_out_0)``.
        out (npt.NDArray[np.float32 | np.float64]): Output matrix, shape
            ``(n_in_0, n_in_0)``.
        scratch (npt.NDArray[np.float32 | np.float64]): Work buffer of size at
            least ``n_in_0 * n_out_0`` when not identity.

    Note:
        Inputs are assumed to be correct (no validation performed).
        ``out`` must not alias ``K`` except in the all-identity case.
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    n_out_0 = M_0.shape[0]
    n_in_0 = M_0.shape[1]

    if is_id_0:
        for i in range(n_out_0):
            for j in range(n_out_0):
                out[i, j] = K[i, j]
        return

    zero = M_0.dtype.type(0.0)
    # Stage 1: tmp = M_0^T @ K, shape (n_in_0, n_out_0).
    tmp = scratch[: n_in_0 * n_out_0].reshape(n_in_0, n_out_0)
    for i in range(n_in_0):
        for j in range(n_out_0):
            s = zero
            for k in range(n_out_0):
                s += M_0[k, i] * K[k, j]
            tmp[i, j] = s
    # Stage 2: out = tmp @ M_0, shape (n_in_0, n_in_0).
    for i in range(n_in_0):
        for j in range(n_in_0):
            s = zero
            for k in range(n_out_0):
                s += tmp[i, k] * M_0[k, j]
            out[i, j] = s


@nb_jit(nopython=True, cache=True)
def apply_kron_M_K_MT_1d(
    M_0: npt.NDArray[np.float32 | np.float64],
    is_id_0: bool,
    K: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
    scratch: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Compute ``out = M_0 @ K @ M_0^T`` with identity short-circuit.

    Args:
        M_0 (npt.NDArray[np.float32 | np.float64]): Operator, shape
            ``(n_out_0, n_in_0)``.
        is_id_0 (bool): If True, ``out`` is set equal to ``K``.
        K (npt.NDArray[np.float32 | np.float64]): Input matrix, shape
            ``(n_in_0, n_in_0)``.
        out (npt.NDArray[np.float32 | np.float64]): Output matrix, shape
            ``(n_out_0, n_out_0)``.
        scratch (npt.NDArray[np.float32 | np.float64]): Work buffer of size at
            least ``n_out_0 * n_in_0`` when not identity.

    Note:
        Inputs are assumed to be correct (no validation performed).
        ``out`` must not alias ``K`` except in the all-identity case.
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    n_out_0 = M_0.shape[0]
    n_in_0 = M_0.shape[1]

    if is_id_0:
        for i in range(n_in_0):
            for j in range(n_in_0):
                out[i, j] = K[i, j]
        return

    zero = M_0.dtype.type(0.0)
    tmp = scratch[: n_out_0 * n_in_0].reshape(n_out_0, n_in_0)
    for i in range(n_out_0):
        for j in range(n_in_0):
            s = zero
            for k in range(n_in_0):
                s += M_0[i, k] * K[k, j]
            tmp[i, j] = s
    for i in range(n_out_0):
        for j in range(n_out_0):
            s = zero
            for k in range(n_in_0):
                s += tmp[i, k] * M_0[j, k]
            out[i, j] = s


@nb_jit(nopython=True, cache=True)
def apply_kron_MT_K_M_2d(  # noqa: PLR0913, PLR0912 -- kernel fan-in and identity branching are intentional.
    M_0: npt.NDArray[np.float32 | np.float64],
    M_1: npt.NDArray[np.float32 | np.float64],
    is_id_0: bool,
    is_id_1: bool,
    K: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
    scratch: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Compute ``out = kron(M_0, M_1)^T @ K @ kron(M_0, M_1)``.

    Uses mode-wise contractions: for each direction, apply ``M_k^T`` to the
    corresponding row axis and ``M_k`` to the corresponding column axis of
    the 4-axis reshape of K.

    Args:
        M_0 (npt.NDArray[np.float32 | np.float64]): Direction-0 operator,
            shape ``(n_out_0, n_in_0)``.
        M_1 (npt.NDArray[np.float32 | np.float64]): Direction-1 operator,
            shape ``(n_out_1, n_in_1)``.
        is_id_0 (bool): Identity flag for direction 0.
        is_id_1 (bool): Identity flag for direction 1.
        K (npt.NDArray[np.float32 | np.float64]): Input matrix, shape
            ``(n_out_0 * n_out_1, n_out_0 * n_out_1)``.
        out (npt.NDArray[np.float32 | np.float64]): Output matrix, shape
            ``(n_in_0 * n_in_1, n_in_0 * n_in_1)``.
        scratch (npt.NDArray[np.float32 | np.float64]): Work buffer sized per
            :func:`pantr.bspline._extraction_helpers._required_scratch_size`
            for the ``"MT_K_M"`` kind.

    Note:
        Inputs are assumed to be correct (no validation performed).
        ``out`` must not alias ``K`` except in the all-identity case.
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    n_out_0 = M_0.shape[0]
    n_in_0 = M_0.shape[1]
    n_out_1 = M_1.shape[0]
    n_in_1 = M_1.shape[1]

    if is_id_0 and is_id_1:
        N_out = n_out_0 * n_out_1
        for i in range(N_out):
            for j in range(N_out):
                out[i, j] = K[i, j]
        return

    zero = M_0.dtype.type(0.0)

    half = scratch.size // 2
    buf_a = scratch[:half]
    buf_b = scratch[half:]

    # Reshape K as 4D: (n_out_0, n_out_1, n_out_0, n_out_1).
    K_view = K.reshape(n_out_0, n_out_1, n_out_0, n_out_1)

    # Stage 0: apply M_0^T on axis 0. Shape -> (n_in_0, n_out_1, n_out_0, n_out_1).
    d0 = n_in_0 if not is_id_0 else n_out_0
    if is_id_0:
        cur = K_view  # (n_out_0, n_out_1, n_out_0, n_out_1)
    else:
        cur = buf_a[: d0 * n_out_1 * n_out_0 * n_out_1].reshape(d0, n_out_1, n_out_0, n_out_1)
        for i in range(d0):
            for j in range(n_out_1):
                for k in range(n_out_0):
                    for m in range(n_out_1):
                        s = zero
                        for r in range(n_out_0):
                            s += M_0[r, i] * K_view[r, j, k, m]
                        cur[i, j, k, m] = s

    # Stage 1: apply M_0 on axis 2 (the col-side counterpart of axis 0).
    # Shape -> (d0, n_out_1, n_in_0, n_out_1).
    if is_id_0:
        pass
    else:
        nxt = buf_b[: d0 * n_out_1 * d0 * n_out_1].reshape(d0, n_out_1, d0, n_out_1)
        for i in range(d0):
            for j in range(n_out_1):
                for k in range(d0):
                    for m in range(n_out_1):
                        s = zero
                        for r in range(n_out_0):
                            s += cur[i, j, r, m] * M_0[r, k]
                        nxt[i, j, k, m] = s
        cur = nxt

    # Stage 2: apply M_1^T on axis 1. Shape -> (d0, n_in_1, d0, n_out_1).
    d1 = n_in_1 if not is_id_1 else n_out_1
    if is_id_1:
        pass
    else:
        # Use buf_a as next (cur currently in buf_b if we did stage 1; else in K_view).
        # Size needed: d0 * d1 * d0 * n_out_1.
        nxt = buf_a[: d0 * d1 * d0 * n_out_1].reshape(d0, d1, d0, n_out_1)
        for i in range(d0):
            for j in range(d1):
                for k in range(d0):
                    for m in range(n_out_1):
                        s = zero
                        for r in range(n_out_1):
                            s += M_1[r, j] * cur[i, r, k, m]
                        nxt[i, j, k, m] = s
        cur = nxt

    # Stage 3: apply M_1 on axis 3. Shape -> (d0, d1, d0, d1) = (n_in_0, n_in_1, n_in_0, n_in_1).
    out_view = out.reshape(d0, d1, d0, d1)
    if is_id_1:
        for i in range(d0):
            for j in range(d1):
                for k in range(d0):
                    for m in range(d1):
                        out_view[i, j, k, m] = cur[i, j, k, m]
    else:
        for i in range(d0):
            for j in range(d1):
                for k in range(d0):
                    for m in range(d1):
                        s = zero
                        for r in range(n_out_1):
                            s += cur[i, j, k, r] * M_1[r, m]
                        out_view[i, j, k, m] = s


@nb_jit(nopython=True, cache=True)
def apply_kron_M_K_MT_2d(  # noqa: PLR0913, PLR0912 -- kernel fan-in and identity branching are intentional.
    M_0: npt.NDArray[np.float32 | np.float64],
    M_1: npt.NDArray[np.float32 | np.float64],
    is_id_0: bool,
    is_id_1: bool,
    K: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
    scratch: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Compute ``out = kron(M_0, M_1) @ K @ kron(M_0, M_1)^T``.

    Mirror of :func:`apply_kron_MT_K_M_2d` with the roles of ``M_k`` and
    ``M_k^T`` swapped.

    Args:
        M_0 (npt.NDArray[np.float32 | np.float64]): Direction-0 operator,
            shape ``(n_out_0, n_in_0)``.
        M_1 (npt.NDArray[np.float32 | np.float64]): Direction-1 operator,
            shape ``(n_out_1, n_in_1)``.
        is_id_0 (bool): Identity flag for direction 0.
        is_id_1 (bool): Identity flag for direction 1.
        K (npt.NDArray[np.float32 | np.float64]): Input matrix, shape
            ``(n_in_0 * n_in_1, n_in_0 * n_in_1)``.
        out (npt.NDArray[np.float32 | np.float64]): Output matrix, shape
            ``(n_out_0 * n_out_1, n_out_0 * n_out_1)``.
        scratch (npt.NDArray[np.float32 | np.float64]): Work buffer sized per
            :func:`pantr.bspline._extraction_helpers._required_scratch_size`
            for the ``"M_K_MT"`` kind.

    Note:
        Inputs are assumed to be correct (no validation performed).
        ``out`` must not alias ``K`` except in the all-identity case.
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    n_out_0 = M_0.shape[0]
    n_in_0 = M_0.shape[1]
    n_out_1 = M_1.shape[0]
    n_in_1 = M_1.shape[1]

    if is_id_0 and is_id_1:
        N_in = n_in_0 * n_in_1
        for i in range(N_in):
            for j in range(N_in):
                out[i, j] = K[i, j]
        return

    zero = M_0.dtype.type(0.0)

    half = scratch.size // 2
    buf_a = scratch[:half]
    buf_b = scratch[half:]

    K_view = K.reshape(n_in_0, n_in_1, n_in_0, n_in_1)

    # Stage 0: apply M_0 on axis 0.
    d0 = n_out_0 if not is_id_0 else n_in_0
    if is_id_0:
        cur = K_view
    else:
        cur = buf_a[: d0 * n_in_1 * n_in_0 * n_in_1].reshape(d0, n_in_1, n_in_0, n_in_1)
        for i in range(d0):
            for j in range(n_in_1):
                for k in range(n_in_0):
                    for m in range(n_in_1):
                        s = zero
                        for r in range(n_in_0):
                            s += M_0[i, r] * K_view[r, j, k, m]
                        cur[i, j, k, m] = s

    # Stage 1: apply M_0^T on axis 2.
    if is_id_0:
        pass
    else:
        nxt = buf_b[: d0 * n_in_1 * d0 * n_in_1].reshape(d0, n_in_1, d0, n_in_1)
        for i in range(d0):
            for j in range(n_in_1):
                for k in range(d0):
                    for m in range(n_in_1):
                        s = zero
                        for r in range(n_in_0):
                            s += cur[i, j, r, m] * M_0[k, r]
                        nxt[i, j, k, m] = s
        cur = nxt

    # Stage 2: apply M_1 on axis 1.
    d1 = n_out_1 if not is_id_1 else n_in_1
    if is_id_1:
        pass
    else:
        nxt = buf_a[: d0 * d1 * d0 * n_in_1].reshape(d0, d1, d0, n_in_1)
        for i in range(d0):
            for j in range(d1):
                for k in range(d0):
                    for m in range(n_in_1):
                        s = zero
                        for r in range(n_in_1):
                            s += M_1[j, r] * cur[i, r, k, m]
                        nxt[i, j, k, m] = s
        cur = nxt

    # Stage 3: apply M_1^T on axis 3.
    out_view = out.reshape(d0, d1, d0, d1)
    if is_id_1:
        for i in range(d0):
            for j in range(d1):
                for k in range(d0):
                    for m in range(d1):
                        out_view[i, j, k, m] = cur[i, j, k, m]
    else:
        for i in range(d0):
            for j in range(d1):
                for k in range(d0):
                    for m in range(d1):
                        s = zero
                        for r in range(n_in_1):
                            s += cur[i, j, k, r] * M_1[m, r]
                        out_view[i, j, k, m] = s


@nb_jit(nopython=True, cache=True)
def apply_kron_MT_K_M_3d(  # noqa: PLR0913, PLR0912, PLR0915 -- kernel fan-in and identity branching are intentional.
    M_0: npt.NDArray[np.float32 | np.float64],
    M_1: npt.NDArray[np.float32 | np.float64],
    M_2: npt.NDArray[np.float32 | np.float64],
    is_id_0: bool,
    is_id_1: bool,
    is_id_2: bool,
    K: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
    scratch: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Compute ``out = kron(M_0, M_1, M_2)^T @ K @ kron(M_0, M_1, M_2)``.

    Uses mode-wise contractions in the 6-axis reshape of K.

    Args:
        M_0 (npt.NDArray[np.float32 | np.float64]): Direction-0 operator,
            shape ``(n_out_0, n_in_0)``.
        M_1 (npt.NDArray[np.float32 | np.float64]): Direction-1 operator,
            shape ``(n_out_1, n_in_1)``.
        M_2 (npt.NDArray[np.float32 | np.float64]): Direction-2 operator,
            shape ``(n_out_2, n_in_2)``.
        is_id_0 (bool): Identity flag for direction 0.
        is_id_1 (bool): Identity flag for direction 1.
        is_id_2 (bool): Identity flag for direction 2.
        K (npt.NDArray[np.float32 | np.float64]): Input matrix, shape
            ``(N_out, N_out)`` with ``N_out = n_out_0 * n_out_1 * n_out_2``.
        out (npt.NDArray[np.float32 | np.float64]): Output matrix, shape
            ``(N_in, N_in)`` with ``N_in = n_in_0 * n_in_1 * n_in_2``.
        scratch (npt.NDArray[np.float32 | np.float64]): Work buffer sized per
            :func:`pantr.bspline._extraction_helpers._required_scratch_size`
            for the ``"MT_K_M"`` kind.

    Note:
        Inputs are assumed to be correct (no validation performed).
        ``out`` must not alias ``K`` except in the all-identity case.
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    n_out_0 = M_0.shape[0]
    n_in_0 = M_0.shape[1]
    n_out_1 = M_1.shape[0]
    n_in_1 = M_1.shape[1]
    n_out_2 = M_2.shape[0]
    n_in_2 = M_2.shape[1]

    if is_id_0 and is_id_1 and is_id_2:
        N_out = n_out_0 * n_out_1 * n_out_2
        for i in range(N_out):
            for j in range(N_out):
                out[i, j] = K[i, j]
        return

    zero = M_0.dtype.type(0.0)

    half = scratch.size // 2
    buf_a = scratch[:half]
    buf_b = scratch[half:]

    # Track current shape axes (row0, row1, row2, col0, col1, col2).
    d0_r = n_out_0
    d1_r = n_out_1
    d2_r = n_out_2
    d0_c = n_out_0
    d1_c = n_out_1
    d2_c = n_out_2

    # cur starts as K reshaped into 6D.
    # Ping-pong buffers: use buf_a and buf_b alternately.
    # We'll track "cur" by reassigning after each stage.
    cur = K.reshape(d0_r, d1_r, d2_r, d0_c, d1_c, d2_c)

    # Stage: apply M_0^T on axis 0 (row side).
    if not is_id_0:
        d0_r = n_in_0
        nxt = buf_a[: d0_r * d1_r * d2_r * d0_c * d1_c * d2_c].reshape(
            d0_r, d1_r, d2_r, d0_c, d1_c, d2_c
        )
        for i in range(d0_r):
            for j in range(d1_r):
                for k in range(d2_r):
                    for a in range(d0_c):
                        for b in range(d1_c):
                            for c in range(d2_c):
                                s = zero
                                for r in range(n_out_0):
                                    s += M_0[r, i] * cur[r, j, k, a, b, c]
                                nxt[i, j, k, a, b, c] = s
        cur = nxt

    # Stage: apply M_0 on axis 3 (col side).
    if not is_id_0:
        d0_c = n_in_0
        nxt = buf_b[: d0_r * d1_r * d2_r * d0_c * d1_c * d2_c].reshape(
            d0_r, d1_r, d2_r, d0_c, d1_c, d2_c
        )
        for i in range(d0_r):
            for j in range(d1_r):
                for k in range(d2_r):
                    for a in range(d0_c):
                        for b in range(d1_c):
                            for c in range(d2_c):
                                s = zero
                                for r in range(n_out_0):
                                    s += cur[i, j, k, r, b, c] * M_0[r, a]
                                nxt[i, j, k, a, b, c] = s
        cur = nxt

    # Stage: apply M_1^T on axis 1.
    if not is_id_1:
        d1_r = n_in_1
        nxt = buf_a[: d0_r * d1_r * d2_r * d0_c * d1_c * d2_c].reshape(
            d0_r, d1_r, d2_r, d0_c, d1_c, d2_c
        )
        for i in range(d0_r):
            for j in range(d1_r):
                for k in range(d2_r):
                    for a in range(d0_c):
                        for b in range(d1_c):
                            for c in range(d2_c):
                                s = zero
                                for r in range(n_out_1):
                                    s += M_1[r, j] * cur[i, r, k, a, b, c]
                                nxt[i, j, k, a, b, c] = s
        cur = nxt

    # Stage: apply M_1 on axis 4.
    if not is_id_1:
        d1_c = n_in_1
        nxt = buf_b[: d0_r * d1_r * d2_r * d0_c * d1_c * d2_c].reshape(
            d0_r, d1_r, d2_r, d0_c, d1_c, d2_c
        )
        for i in range(d0_r):
            for j in range(d1_r):
                for k in range(d2_r):
                    for a in range(d0_c):
                        for b in range(d1_c):
                            for c in range(d2_c):
                                s = zero
                                for r in range(n_out_1):
                                    s += cur[i, j, k, a, r, c] * M_1[r, b]
                                nxt[i, j, k, a, b, c] = s
        cur = nxt

    # Stage: apply M_2^T on axis 2.
    if not is_id_2:
        d2_r = n_in_2
        nxt = buf_a[: d0_r * d1_r * d2_r * d0_c * d1_c * d2_c].reshape(
            d0_r, d1_r, d2_r, d0_c, d1_c, d2_c
        )
        for i in range(d0_r):
            for j in range(d1_r):
                for k in range(d2_r):
                    for a in range(d0_c):
                        for b in range(d1_c):
                            for c in range(d2_c):
                                s = zero
                                for r in range(n_out_2):
                                    s += M_2[r, k] * cur[i, j, r, a, b, c]
                                nxt[i, j, k, a, b, c] = s
        cur = nxt

    # Stage: apply M_2 on axis 5, writing directly to out.
    d2_c_final = n_in_2 if not is_id_2 else d2_c
    out_view = out.reshape(d0_r, d1_r, d2_r, d0_c, d1_c, d2_c_final)
    if not is_id_2:
        for i in range(d0_r):
            for j in range(d1_r):
                for k in range(d2_r):
                    for a in range(d0_c):
                        for b in range(d1_c):
                            for c in range(d2_c_final):
                                s = zero
                                for r in range(n_out_2):
                                    s += cur[i, j, k, a, b, r] * M_2[r, c]
                                out_view[i, j, k, a, b, c] = s
    else:
        for i in range(d0_r):
            for j in range(d1_r):
                for k in range(d2_r):
                    for a in range(d0_c):
                        for b in range(d1_c):
                            for c in range(d2_c_final):
                                out_view[i, j, k, a, b, c] = cur[i, j, k, a, b, c]


@nb_jit(nopython=True, cache=True)
def apply_kron_M_K_MT_3d(  # noqa: PLR0913, PLR0912, PLR0915 -- kernel fan-in and identity branching are intentional.
    M_0: npt.NDArray[np.float32 | np.float64],
    M_1: npt.NDArray[np.float32 | np.float64],
    M_2: npt.NDArray[np.float32 | np.float64],
    is_id_0: bool,
    is_id_1: bool,
    is_id_2: bool,
    K: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
    scratch: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Compute ``out = kron(M_0, M_1, M_2) @ K @ kron(M_0, M_1, M_2)^T``.

    Mirror of :func:`apply_kron_MT_K_M_3d` with the roles of ``M_k`` and
    ``M_k^T`` swapped.

    Args:
        M_0 (npt.NDArray[np.float32 | np.float64]): Direction-0 operator,
            shape ``(n_out_0, n_in_0)``.
        M_1 (npt.NDArray[np.float32 | np.float64]): Direction-1 operator,
            shape ``(n_out_1, n_in_1)``.
        M_2 (npt.NDArray[np.float32 | np.float64]): Direction-2 operator,
            shape ``(n_out_2, n_in_2)``.
        is_id_0 (bool): Identity flag for direction 0.
        is_id_1 (bool): Identity flag for direction 1.
        is_id_2 (bool): Identity flag for direction 2.
        K (npt.NDArray[np.float32 | np.float64]): Input matrix, shape
            ``(N_in, N_in)`` with ``N_in = n_in_0 * n_in_1 * n_in_2``.
        out (npt.NDArray[np.float32 | np.float64]): Output matrix, shape
            ``(N_out, N_out)`` with ``N_out = n_out_0 * n_out_1 * n_out_2``.
        scratch (npt.NDArray[np.float32 | np.float64]): Work buffer sized per
            :func:`pantr.bspline._extraction_helpers._required_scratch_size`
            for the ``"M_K_MT"`` kind.

    Note:
        Inputs are assumed to be correct (no validation performed).
        ``out`` must not alias ``K`` except in the all-identity case.
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    n_out_0 = M_0.shape[0]
    n_in_0 = M_0.shape[1]
    n_out_1 = M_1.shape[0]
    n_in_1 = M_1.shape[1]
    n_out_2 = M_2.shape[0]
    n_in_2 = M_2.shape[1]

    if is_id_0 and is_id_1 and is_id_2:
        N_in = n_in_0 * n_in_1 * n_in_2
        for i in range(N_in):
            for j in range(N_in):
                out[i, j] = K[i, j]
        return

    zero = M_0.dtype.type(0.0)

    half = scratch.size // 2
    buf_a = scratch[:half]
    buf_b = scratch[half:]

    d0_r = n_in_0
    d1_r = n_in_1
    d2_r = n_in_2
    d0_c = n_in_0
    d1_c = n_in_1
    d2_c = n_in_2

    cur = K.reshape(d0_r, d1_r, d2_r, d0_c, d1_c, d2_c)

    if not is_id_0:
        d0_r = n_out_0
        nxt = buf_a[: d0_r * d1_r * d2_r * d0_c * d1_c * d2_c].reshape(
            d0_r, d1_r, d2_r, d0_c, d1_c, d2_c
        )
        for i in range(d0_r):
            for j in range(d1_r):
                for k in range(d2_r):
                    for a in range(d0_c):
                        for b in range(d1_c):
                            for c in range(d2_c):
                                s = zero
                                for r in range(n_in_0):
                                    s += M_0[i, r] * cur[r, j, k, a, b, c]
                                nxt[i, j, k, a, b, c] = s
        cur = nxt

    if not is_id_0:
        d0_c = n_out_0
        nxt = buf_b[: d0_r * d1_r * d2_r * d0_c * d1_c * d2_c].reshape(
            d0_r, d1_r, d2_r, d0_c, d1_c, d2_c
        )
        for i in range(d0_r):
            for j in range(d1_r):
                for k in range(d2_r):
                    for a in range(d0_c):
                        for b in range(d1_c):
                            for c in range(d2_c):
                                s = zero
                                for r in range(n_in_0):
                                    s += cur[i, j, k, r, b, c] * M_0[a, r]
                                nxt[i, j, k, a, b, c] = s
        cur = nxt

    if not is_id_1:
        d1_r = n_out_1
        nxt = buf_a[: d0_r * d1_r * d2_r * d0_c * d1_c * d2_c].reshape(
            d0_r, d1_r, d2_r, d0_c, d1_c, d2_c
        )
        for i in range(d0_r):
            for j in range(d1_r):
                for k in range(d2_r):
                    for a in range(d0_c):
                        for b in range(d1_c):
                            for c in range(d2_c):
                                s = zero
                                for r in range(n_in_1):
                                    s += M_1[j, r] * cur[i, r, k, a, b, c]
                                nxt[i, j, k, a, b, c] = s
        cur = nxt

    if not is_id_1:
        d1_c = n_out_1
        nxt = buf_b[: d0_r * d1_r * d2_r * d0_c * d1_c * d2_c].reshape(
            d0_r, d1_r, d2_r, d0_c, d1_c, d2_c
        )
        for i in range(d0_r):
            for j in range(d1_r):
                for k in range(d2_r):
                    for a in range(d0_c):
                        for b in range(d1_c):
                            for c in range(d2_c):
                                s = zero
                                for r in range(n_in_1):
                                    s += cur[i, j, k, a, r, c] * M_1[b, r]
                                nxt[i, j, k, a, b, c] = s
        cur = nxt

    if not is_id_2:
        d2_r = n_out_2
        nxt = buf_a[: d0_r * d1_r * d2_r * d0_c * d1_c * d2_c].reshape(
            d0_r, d1_r, d2_r, d0_c, d1_c, d2_c
        )
        for i in range(d0_r):
            for j in range(d1_r):
                for k in range(d2_r):
                    for a in range(d0_c):
                        for b in range(d1_c):
                            for c in range(d2_c):
                                s = zero
                                for r in range(n_in_2):
                                    s += M_2[k, r] * cur[i, j, r, a, b, c]
                                nxt[i, j, k, a, b, c] = s
        cur = nxt

    d2_c_final = n_out_2 if not is_id_2 else d2_c
    out_view = out.reshape(d0_r, d1_r, d2_r, d0_c, d1_c, d2_c_final)
    if not is_id_2:
        for i in range(d0_r):
            for j in range(d1_r):
                for k in range(d2_r):
                    for a in range(d0_c):
                        for b in range(d1_c):
                            for c in range(d2_c_final):
                                s = zero
                                for r in range(n_in_2):
                                    s += cur[i, j, k, a, b, r] * M_2[c, r]
                                out_view[i, j, k, a, b, c] = s
    else:
        for i in range(d0_r):
            for j in range(d1_r):
                for k in range(d2_r):
                    for a in range(d0_c):
                        for b in range(d1_c):
                            for c in range(d2_c_final):
                                out_view[i, j, k, a, b, c] = cur[i, j, k, a, b, c]


# ---------------------------------------------------------------- batch kernels
# Each batch kernel wraps the corresponding per-cell kernel in a prange loop
# over cells. Scratch is 2D: shape (n_cells, scratch_size_per_cell). Indexing
# scratch[cell] gives the 1D per-cell slice that the per-cell kernels expect.


@nb_jit(nopython=True, cache=True, parallel=True)
def apply_kron_apply_many_1d(  # noqa: PLR0913 -- kernel fan-in is intentional.
    ops_0: npt.NDArray[Any],
    is_id_0: npt.NDArray[Any],
    cell_indices: npt.NDArray[Any],
    v: npt.NDArray[Any],
    out: npt.NDArray[Any],
    scratch: npt.NDArray[Any],
) -> None:
    """Apply ``M_0 @ v[c]`` for every cell ``c`` in the batch (d=1).

    Args:
        ops_0 (npt.NDArray[Any]): All element operators for direction 0,
            shape ``(n_el_0, n_out_0, n_in_0)``.
        is_id_0 (npt.NDArray[Any]): Per-element identity flags for direction 0,
            shape ``(n_el_0,)`` boolean.
        cell_indices (npt.NDArray[Any]): Per-direction element indices, shape
            ``(n_cells, 1)`` integer.
        v (npt.NDArray[Any]): Batch input vectors, shape ``(n_cells, N_in)``.
        out (npt.NDArray[Any]): Batch output vectors, shape ``(n_cells, N_out)``.
        scratch (npt.NDArray[Any]): Per-cell scratch buffers, shape
            ``(n_cells, scratch_size)``; unused for d=1.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    for cell in nb_prange(cell_indices.shape[0]):
        i0 = cell_indices[cell, 0]
        apply_kron_1d(ops_0[i0], is_id_0[i0], v[cell], out[cell], scratch[cell])


@nb_jit(nopython=True, cache=True, parallel=True)
def apply_kron_apply_many_2d(  # noqa: PLR0913
    ops_0: npt.NDArray[Any],
    ops_1: npt.NDArray[Any],
    is_id_0: npt.NDArray[Any],
    is_id_1: npt.NDArray[Any],
    cell_indices: npt.NDArray[Any],
    v: npt.NDArray[Any],
    out: npt.NDArray[Any],
    scratch: npt.NDArray[Any],
) -> None:
    """Apply ``kron(M_0, M_1) @ v[c]`` for every cell ``c`` in the batch (d=2).

    Args:
        ops_0 (npt.NDArray[Any]): All element operators for direction 0,
            shape ``(n_el_0, n_out_0, n_in_0)``.
        ops_1 (npt.NDArray[Any]): All element operators for direction 1,
            shape ``(n_el_1, n_out_1, n_in_1)``.
        is_id_0 (npt.NDArray[Any]): Per-element identity flags for direction 0,
            shape ``(n_el_0,)`` boolean.
        is_id_1 (npt.NDArray[Any]): Per-element identity flags for direction 1,
            shape ``(n_el_1,)`` boolean.
        cell_indices (npt.NDArray[Any]): Per-direction element indices, shape
            ``(n_cells, 2)`` integer.
        v (npt.NDArray[Any]): Batch input vectors, shape ``(n_cells, N_in)``.
        out (npt.NDArray[Any]): Batch output vectors, shape ``(n_cells, N_out)``.
        scratch (npt.NDArray[Any]): Per-cell scratch buffers, shape
            ``(n_cells, scratch_size)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    for cell in nb_prange(cell_indices.shape[0]):
        i0 = cell_indices[cell, 0]
        i1 = cell_indices[cell, 1]
        apply_kron_2d(
            ops_0[i0], ops_1[i1], is_id_0[i0], is_id_1[i1], v[cell], out[cell], scratch[cell]
        )


@nb_jit(nopython=True, cache=True, parallel=True)
def apply_kron_apply_many_3d(  # noqa: PLR0913
    ops_0: npt.NDArray[Any],
    ops_1: npt.NDArray[Any],
    ops_2: npt.NDArray[Any],
    is_id_0: npt.NDArray[Any],
    is_id_1: npt.NDArray[Any],
    is_id_2: npt.NDArray[Any],
    cell_indices: npt.NDArray[Any],
    v: npt.NDArray[Any],
    out: npt.NDArray[Any],
    scratch: npt.NDArray[Any],
) -> None:
    """Apply ``kron(M_0, M_1, M_2) @ v[c]`` for every cell ``c`` in the batch (d=3).

    Args:
        ops_0 (npt.NDArray[Any]): All element operators for direction 0,
            shape ``(n_el_0, n_out_0, n_in_0)``.
        ops_1 (npt.NDArray[Any]): All element operators for direction 1,
            shape ``(n_el_1, n_out_1, n_in_1)``.
        ops_2 (npt.NDArray[Any]): All element operators for direction 2,
            shape ``(n_el_2, n_out_2, n_in_2)``.
        is_id_0 (npt.NDArray[Any]): Per-element identity flags for direction 0,
            shape ``(n_el_0,)`` boolean.
        is_id_1 (npt.NDArray[Any]): Per-element identity flags for direction 1,
            shape ``(n_el_1,)`` boolean.
        is_id_2 (npt.NDArray[Any]): Per-element identity flags for direction 2,
            shape ``(n_el_2,)`` boolean.
        cell_indices (npt.NDArray[Any]): Per-direction element indices, shape
            ``(n_cells, 3)`` integer.
        v (npt.NDArray[Any]): Batch input vectors, shape ``(n_cells, N_in)``.
        out (npt.NDArray[Any]): Batch output vectors, shape ``(n_cells, N_out)``.
        scratch (npt.NDArray[Any]): Per-cell scratch buffers, shape
            ``(n_cells, scratch_size)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    for cell in nb_prange(cell_indices.shape[0]):
        i0 = cell_indices[cell, 0]
        i1 = cell_indices[cell, 1]
        i2 = cell_indices[cell, 2]
        apply_kron_3d(
            ops_0[i0],
            ops_1[i1],
            ops_2[i2],
            is_id_0[i0],
            is_id_1[i1],
            is_id_2[i2],
            v[cell],
            out[cell],
            scratch[cell],
        )


@nb_jit(nopython=True, cache=True, parallel=True)
def apply_kron_apply_T_many_1d(  # noqa: PLR0913 -- kernel fan-in is intentional.
    ops_0: npt.NDArray[Any],
    is_id_0: npt.NDArray[Any],
    cell_indices: npt.NDArray[Any],
    v: npt.NDArray[Any],
    out: npt.NDArray[Any],
    scratch: npt.NDArray[Any],
) -> None:
    """Apply ``M_0^T @ v[c]`` for every cell ``c`` in the batch (d=1).

    Args:
        ops_0 (npt.NDArray[Any]): All element operators for direction 0,
            shape ``(n_el_0, n_out_0, n_in_0)``.
        is_id_0 (npt.NDArray[Any]): Per-element identity flags, shape
            ``(n_el_0,)`` boolean.
        cell_indices (npt.NDArray[Any]): Per-direction element indices, shape
            ``(n_cells, 1)`` integer.
        v (npt.NDArray[Any]): Batch input vectors, shape ``(n_cells, N_out)``.
        out (npt.NDArray[Any]): Batch output vectors, shape ``(n_cells, N_in)``.
        scratch (npt.NDArray[Any]): Per-cell scratch buffers, shape
            ``(n_cells, scratch_size)``; unused for d=1.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    for cell in nb_prange(cell_indices.shape[0]):
        i0 = cell_indices[cell, 0]
        apply_kron_T_1d(ops_0[i0], is_id_0[i0], v[cell], out[cell], scratch[cell])


@nb_jit(nopython=True, cache=True, parallel=True)
def apply_kron_apply_T_many_2d(  # noqa: PLR0913
    ops_0: npt.NDArray[Any],
    ops_1: npt.NDArray[Any],
    is_id_0: npt.NDArray[Any],
    is_id_1: npt.NDArray[Any],
    cell_indices: npt.NDArray[Any],
    v: npt.NDArray[Any],
    out: npt.NDArray[Any],
    scratch: npt.NDArray[Any],
) -> None:
    """Apply ``kron(M_0, M_1)^T @ v[c]`` for every cell ``c`` in the batch (d=2).

    Args:
        ops_0 (npt.NDArray[Any]): All element operators for direction 0,
            shape ``(n_el_0, n_out_0, n_in_0)``.
        ops_1 (npt.NDArray[Any]): All element operators for direction 1,
            shape ``(n_el_1, n_out_1, n_in_1)``.
        is_id_0 (npt.NDArray[Any]): Per-element identity flags for direction 0,
            shape ``(n_el_0,)`` boolean.
        is_id_1 (npt.NDArray[Any]): Per-element identity flags for direction 1,
            shape ``(n_el_1,)`` boolean.
        cell_indices (npt.NDArray[Any]): Per-direction element indices, shape
            ``(n_cells, 2)`` integer.
        v (npt.NDArray[Any]): Batch input vectors, shape ``(n_cells, N_out)``.
        out (npt.NDArray[Any]): Batch output vectors, shape ``(n_cells, N_in)``.
        scratch (npt.NDArray[Any]): Per-cell scratch buffers, shape
            ``(n_cells, scratch_size)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    for cell in nb_prange(cell_indices.shape[0]):
        i0 = cell_indices[cell, 0]
        i1 = cell_indices[cell, 1]
        apply_kron_T_2d(
            ops_0[i0], ops_1[i1], is_id_0[i0], is_id_1[i1], v[cell], out[cell], scratch[cell]
        )


@nb_jit(nopython=True, cache=True, parallel=True)
def apply_kron_apply_T_many_3d(  # noqa: PLR0913
    ops_0: npt.NDArray[Any],
    ops_1: npt.NDArray[Any],
    ops_2: npt.NDArray[Any],
    is_id_0: npt.NDArray[Any],
    is_id_1: npt.NDArray[Any],
    is_id_2: npt.NDArray[Any],
    cell_indices: npt.NDArray[Any],
    v: npt.NDArray[Any],
    out: npt.NDArray[Any],
    scratch: npt.NDArray[Any],
) -> None:
    """Apply ``kron(M_0, M_1, M_2)^T @ v[c]`` for every cell ``c`` in the batch (d=3).

    Args:
        ops_0 (npt.NDArray[Any]): All element operators for direction 0,
            shape ``(n_el_0, n_out_0, n_in_0)``.
        ops_1 (npt.NDArray[Any]): All element operators for direction 1,
            shape ``(n_el_1, n_out_1, n_in_1)``.
        ops_2 (npt.NDArray[Any]): All element operators for direction 2,
            shape ``(n_el_2, n_out_2, n_in_2)``.
        is_id_0 (npt.NDArray[Any]): Per-element identity flags for direction 0,
            shape ``(n_el_0,)`` boolean.
        is_id_1 (npt.NDArray[Any]): Per-element identity flags for direction 1,
            shape ``(n_el_1,)`` boolean.
        is_id_2 (npt.NDArray[Any]): Per-element identity flags for direction 2,
            shape ``(n_el_2,)`` boolean.
        cell_indices (npt.NDArray[Any]): Per-direction element indices, shape
            ``(n_cells, 3)`` integer.
        v (npt.NDArray[Any]): Batch input vectors, shape ``(n_cells, N_out)``.
        out (npt.NDArray[Any]): Batch output vectors, shape ``(n_cells, N_in)``.
        scratch (npt.NDArray[Any]): Per-cell scratch buffers, shape
            ``(n_cells, scratch_size)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    for cell in nb_prange(cell_indices.shape[0]):
        i0 = cell_indices[cell, 0]
        i1 = cell_indices[cell, 1]
        i2 = cell_indices[cell, 2]
        apply_kron_T_3d(
            ops_0[i0],
            ops_1[i1],
            ops_2[i2],
            is_id_0[i0],
            is_id_1[i1],
            is_id_2[i2],
            v[cell],
            out[cell],
            scratch[cell],
        )


@nb_jit(nopython=True, cache=True, parallel=True)
def apply_kron_MT_K_M_many_1d(  # noqa: PLR0913 -- kernel fan-in is intentional.
    ops_0: npt.NDArray[Any],
    is_id_0: npt.NDArray[Any],
    cell_indices: npt.NDArray[Any],
    K: npt.NDArray[Any],
    out: npt.NDArray[Any],
    scratch: npt.NDArray[Any],
) -> None:
    """Compute ``M_0^T @ K[c] @ M_0`` for every cell ``c`` in the batch (d=1).

    Args:
        ops_0 (npt.NDArray[Any]): All element operators for direction 0,
            shape ``(n_el_0, n_out_0, n_in_0)``.
        is_id_0 (npt.NDArray[Any]): Per-element identity flags, shape
            ``(n_el_0,)`` boolean.
        cell_indices (npt.NDArray[Any]): Per-direction element indices, shape
            ``(n_cells, 1)`` integer.
        K (npt.NDArray[Any]): Batch input matrices, shape
            ``(n_cells, N_out, N_out)``.
        out (npt.NDArray[Any]): Batch output matrices, shape
            ``(n_cells, N_in, N_in)``.
        scratch (npt.NDArray[Any]): Per-cell scratch buffers, shape
            ``(n_cells, scratch_size)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        ``out[c]`` must not alias ``K[c]`` except in the all-identity case.
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    for cell in nb_prange(cell_indices.shape[0]):
        i0 = cell_indices[cell, 0]
        apply_kron_MT_K_M_1d(ops_0[i0], is_id_0[i0], K[cell], out[cell], scratch[cell])


@nb_jit(nopython=True, cache=True, parallel=True)
def apply_kron_MT_K_M_many_2d(  # noqa: PLR0913
    ops_0: npt.NDArray[Any],
    ops_1: npt.NDArray[Any],
    is_id_0: npt.NDArray[Any],
    is_id_1: npt.NDArray[Any],
    cell_indices: npt.NDArray[Any],
    K: npt.NDArray[Any],
    out: npt.NDArray[Any],
    scratch: npt.NDArray[Any],
) -> None:
    """Compute ``kron(M_0,M_1)^T @ K[c] @ kron(M_0,M_1)`` for every cell (d=2).

    Args:
        ops_0 (npt.NDArray[Any]): All element operators for direction 0,
            shape ``(n_el_0, n_out_0, n_in_0)``.
        ops_1 (npt.NDArray[Any]): All element operators for direction 1,
            shape ``(n_el_1, n_out_1, n_in_1)``.
        is_id_0 (npt.NDArray[Any]): Per-element identity flags for direction 0,
            shape ``(n_el_0,)`` boolean.
        is_id_1 (npt.NDArray[Any]): Per-element identity flags for direction 1,
            shape ``(n_el_1,)`` boolean.
        cell_indices (npt.NDArray[Any]): Per-direction element indices, shape
            ``(n_cells, 2)`` integer.
        K (npt.NDArray[Any]): Batch input matrices, shape
            ``(n_cells, N_out, N_out)``.
        out (npt.NDArray[Any]): Batch output matrices, shape
            ``(n_cells, N_in, N_in)``.
        scratch (npt.NDArray[Any]): Per-cell scratch buffers, shape
            ``(n_cells, scratch_size)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        ``out[c]`` must not alias ``K[c]`` except in the all-identity case.
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    for cell in nb_prange(cell_indices.shape[0]):
        i0 = cell_indices[cell, 0]
        i1 = cell_indices[cell, 1]
        apply_kron_MT_K_M_2d(
            ops_0[i0], ops_1[i1], is_id_0[i0], is_id_1[i1], K[cell], out[cell], scratch[cell]
        )


@nb_jit(nopython=True, cache=True, parallel=True)
def apply_kron_MT_K_M_many_3d(  # noqa: PLR0913
    ops_0: npt.NDArray[Any],
    ops_1: npt.NDArray[Any],
    ops_2: npt.NDArray[Any],
    is_id_0: npt.NDArray[Any],
    is_id_1: npt.NDArray[Any],
    is_id_2: npt.NDArray[Any],
    cell_indices: npt.NDArray[Any],
    K: npt.NDArray[Any],
    out: npt.NDArray[Any],
    scratch: npt.NDArray[Any],
) -> None:
    """Compute ``kron(M_0,M_1,M_2)^T @ K[c] @ kron(M_0,M_1,M_2)`` for every cell (d=3).

    Args:
        ops_0 (npt.NDArray[Any]): All element operators for direction 0,
            shape ``(n_el_0, n_out_0, n_in_0)``.
        ops_1 (npt.NDArray[Any]): All element operators for direction 1,
            shape ``(n_el_1, n_out_1, n_in_1)``.
        ops_2 (npt.NDArray[Any]): All element operators for direction 2,
            shape ``(n_el_2, n_out_2, n_in_2)``.
        is_id_0 (npt.NDArray[Any]): Per-element identity flags for direction 0,
            shape ``(n_el_0,)`` boolean.
        is_id_1 (npt.NDArray[Any]): Per-element identity flags for direction 1,
            shape ``(n_el_1,)`` boolean.
        is_id_2 (npt.NDArray[Any]): Per-element identity flags for direction 2,
            shape ``(n_el_2,)`` boolean.
        cell_indices (npt.NDArray[Any]): Per-direction element indices, shape
            ``(n_cells, 3)`` integer.
        K (npt.NDArray[Any]): Batch input matrices, shape
            ``(n_cells, N_out, N_out)``.
        out (npt.NDArray[Any]): Batch output matrices, shape
            ``(n_cells, N_in, N_in)``.
        scratch (npt.NDArray[Any]): Per-cell scratch buffers, shape
            ``(n_cells, scratch_size)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        ``out[c]`` must not alias ``K[c]`` except in the all-identity case.
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    for cell in nb_prange(cell_indices.shape[0]):
        i0 = cell_indices[cell, 0]
        i1 = cell_indices[cell, 1]
        i2 = cell_indices[cell, 2]
        apply_kron_MT_K_M_3d(
            ops_0[i0],
            ops_1[i1],
            ops_2[i2],
            is_id_0[i0],
            is_id_1[i1],
            is_id_2[i2],
            K[cell],
            out[cell],
            scratch[cell],
        )


@nb_jit(nopython=True, cache=True, parallel=True)
def apply_kron_M_K_MT_many_1d(  # noqa: PLR0913 -- kernel fan-in is intentional.
    ops_0: npt.NDArray[Any],
    is_id_0: npt.NDArray[Any],
    cell_indices: npt.NDArray[Any],
    K: npt.NDArray[Any],
    out: npt.NDArray[Any],
    scratch: npt.NDArray[Any],
) -> None:
    """Compute ``M_0 @ K[c] @ M_0^T`` for every cell ``c`` in the batch (d=1).

    Args:
        ops_0 (npt.NDArray[Any]): All element operators for direction 0,
            shape ``(n_el_0, n_out_0, n_in_0)``.
        is_id_0 (npt.NDArray[Any]): Per-element identity flags, shape
            ``(n_el_0,)`` boolean.
        cell_indices (npt.NDArray[Any]): Per-direction element indices, shape
            ``(n_cells, 1)`` integer.
        K (npt.NDArray[Any]): Batch input matrices, shape
            ``(n_cells, N_in, N_in)``.
        out (npt.NDArray[Any]): Batch output matrices, shape
            ``(n_cells, N_out, N_out)``.
        scratch (npt.NDArray[Any]): Per-cell scratch buffers, shape
            ``(n_cells, scratch_size)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        ``out[c]`` must not alias ``K[c]`` except in the all-identity case.
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    for cell in nb_prange(cell_indices.shape[0]):
        i0 = cell_indices[cell, 0]
        apply_kron_M_K_MT_1d(ops_0[i0], is_id_0[i0], K[cell], out[cell], scratch[cell])


@nb_jit(nopython=True, cache=True, parallel=True)
def apply_kron_M_K_MT_many_2d(  # noqa: PLR0913
    ops_0: npt.NDArray[Any],
    ops_1: npt.NDArray[Any],
    is_id_0: npt.NDArray[Any],
    is_id_1: npt.NDArray[Any],
    cell_indices: npt.NDArray[Any],
    K: npt.NDArray[Any],
    out: npt.NDArray[Any],
    scratch: npt.NDArray[Any],
) -> None:
    """Compute ``kron(M_0,M_1) @ K[c] @ kron(M_0,M_1)^T`` for every cell (d=2).

    Args:
        ops_0 (npt.NDArray[Any]): All element operators for direction 0,
            shape ``(n_el_0, n_out_0, n_in_0)``.
        ops_1 (npt.NDArray[Any]): All element operators for direction 1,
            shape ``(n_el_1, n_out_1, n_in_1)``.
        is_id_0 (npt.NDArray[Any]): Per-element identity flags for direction 0,
            shape ``(n_el_0,)`` boolean.
        is_id_1 (npt.NDArray[Any]): Per-element identity flags for direction 1,
            shape ``(n_el_1,)`` boolean.
        cell_indices (npt.NDArray[Any]): Per-direction element indices, shape
            ``(n_cells, 2)`` integer.
        K (npt.NDArray[Any]): Batch input matrices, shape
            ``(n_cells, N_in, N_in)``.
        out (npt.NDArray[Any]): Batch output matrices, shape
            ``(n_cells, N_out, N_out)``.
        scratch (npt.NDArray[Any]): Per-cell scratch buffers, shape
            ``(n_cells, scratch_size)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        ``out[c]`` must not alias ``K[c]`` except in the all-identity case.
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    for cell in nb_prange(cell_indices.shape[0]):
        i0 = cell_indices[cell, 0]
        i1 = cell_indices[cell, 1]
        apply_kron_M_K_MT_2d(
            ops_0[i0], ops_1[i1], is_id_0[i0], is_id_1[i1], K[cell], out[cell], scratch[cell]
        )


@nb_jit(nopython=True, cache=True, parallel=True)
def apply_kron_M_K_MT_many_3d(  # noqa: PLR0913
    ops_0: npt.NDArray[Any],
    ops_1: npt.NDArray[Any],
    ops_2: npt.NDArray[Any],
    is_id_0: npt.NDArray[Any],
    is_id_1: npt.NDArray[Any],
    is_id_2: npt.NDArray[Any],
    cell_indices: npt.NDArray[Any],
    K: npt.NDArray[Any],
    out: npt.NDArray[Any],
    scratch: npt.NDArray[Any],
) -> None:
    """Compute ``kron(M_0,M_1,M_2) @ K[c] @ kron(M_0,M_1,M_2)^T`` for every cell (d=3).

    Args:
        ops_0 (npt.NDArray[Any]): All element operators for direction 0,
            shape ``(n_el_0, n_out_0, n_in_0)``.
        ops_1 (npt.NDArray[Any]): All element operators for direction 1,
            shape ``(n_el_1, n_out_1, n_in_1)``.
        ops_2 (npt.NDArray[Any]): All element operators for direction 2,
            shape ``(n_el_2, n_out_2, n_in_2)``.
        is_id_0 (npt.NDArray[Any]): Per-element identity flags for direction 0,
            shape ``(n_el_0,)`` boolean.
        is_id_1 (npt.NDArray[Any]): Per-element identity flags for direction 1,
            shape ``(n_el_1,)`` boolean.
        is_id_2 (npt.NDArray[Any]): Per-element identity flags for direction 2,
            shape ``(n_el_2,)`` boolean.
        cell_indices (npt.NDArray[Any]): Per-direction element indices, shape
            ``(n_cells, 3)`` integer.
        K (npt.NDArray[Any]): Batch input matrices, shape
            ``(n_cells, N_in, N_in)``.
        out (npt.NDArray[Any]): Batch output matrices, shape
            ``(n_cells, N_out, N_out)``.
        scratch (npt.NDArray[Any]): Per-cell scratch buffers, shape
            ``(n_cells, scratch_size)``.

    Note:
        Inputs are assumed to be correct (no validation performed).
        ``out[c]`` must not alias ``K[c]`` except in the all-identity case.
        For general use, call the Layer-2 dispatcher in
        :mod:`pantr.bspline._extraction_helpers` instead.
    """
    for cell in nb_prange(cell_indices.shape[0]):
        i0 = cell_indices[cell, 0]
        i1 = cell_indices[cell, 1]
        i2 = cell_indices[cell, 2]
        apply_kron_M_K_MT_3d(
            ops_0[i0],
            ops_1[i1],
            ops_2[i2],
            is_id_0[i0],
            is_id_1[i1],
            is_id_2[i2],
            K[cell],
            out[cell],
            scratch[cell],
        )


__all__ = [
    "apply_kron_1d",
    "apply_kron_2d",
    "apply_kron_3d",
    "apply_kron_MT_K_M_1d",
    "apply_kron_MT_K_M_2d",
    "apply_kron_MT_K_M_3d",
    "apply_kron_MT_K_M_many_1d",
    "apply_kron_MT_K_M_many_2d",
    "apply_kron_MT_K_M_many_3d",
    "apply_kron_M_K_MT_1d",
    "apply_kron_M_K_MT_2d",
    "apply_kron_M_K_MT_3d",
    "apply_kron_M_K_MT_many_1d",
    "apply_kron_M_K_MT_many_2d",
    "apply_kron_M_K_MT_many_3d",
    "apply_kron_T_1d",
    "apply_kron_T_2d",
    "apply_kron_T_3d",
    "apply_kron_apply_T_many_1d",
    "apply_kron_apply_T_many_2d",
    "apply_kron_apply_T_many_3d",
    "apply_kron_apply_many_1d",
    "apply_kron_apply_many_2d",
    "apply_kron_apply_many_3d",
]
