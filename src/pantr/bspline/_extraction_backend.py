"""Which implementation of an extraction kernel runs: the Numba one or C++.

:mod:`pantr._backend` owns the **policy**. This module owns the **catalogue** for
the extraction kernels, exactly as :mod:`pantr.change_basis._change_basis_backend`
and :mod:`pantr.basis._basis_backend` do for theirs, and for the same reason: a
catalogue imports the kernels it hands out, so keeping each one next to its own
kernels is what stops the policy module from importing the library it must stay
independent of.

Two accessors, not a record
---------------------------

By the rule ``design/cross_backend_types.md`` states -- a record when the consumer
needs more than one kernel at once, a bare callable when it does not -- these are
bare callables. :func:`pantr.bspline._extraction_helpers._prepare_apply_call`
selects exactly one kernel per call, keyed by ``(op_kind, dim)``, and never needs
a second one in the same call.

What crosses the boundary
-------------------------

Arrays and scalars only. The identity flags cross as ``bool``, the cell indices
as an ``intp`` array, and the operators as the caller's own arrays. ``OpKind``
does **not** cross: each ``(kind, dimension)`` pair has its own entry point, so
the selection is resolved on this side and the kernel never sees a tag. That is
the same shape as :class:`pantr.basis.LagrangeVariant` staying Python-side in
``change_basis`` -- a dispatch decision another module owns does not travel.

Why the C++ kernels are not simply the same functions
-----------------------------------------------------

The twelve per-cell Numba kernels are specialised per dimension because a
``nopython`` function cannot loop over a variable-length tuple of arrays. The C++
side is four dimension-generic functions behind twenty-four flat entry points with
the same names, arities and argument order, so this catalogue selects between two
functions with one signature rather than between two conventions.

Measured, and it is stronger than the port needed: on a build whose target ISA has
no fused multiply-add the two agree **bit for bit**, because the C++ reproduces the
oracle's stage order and its within-stage summation order exactly. Where the
contraction can fuse they differ by ``design/backend_parity.md`` Rule 10's budget.
``tests/parity/test_extraction_kernels.py`` carries the claim and its gate.

The batch kernels are the one place the two differ in kind rather than in bits: the
oracle's are ``parallel=True`` over cells and the C++ loops. Each cell writes only
its own rows and there is no reduction, so the answer cannot move with the thread
count either way; only the speed does, and nothing here has been profiled. Same
argument and same precedent as ``pantr/grid/locate.hpp``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final, TypeVar

import numpy as np
import numpy.typing as npt

from .._backend import Backend, active_backend, available_backends
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

if TYPE_CHECKING:
    from ._extraction_helpers import OpKind

_K = TypeVar("_K", bound=Callable[..., None])
"""One kernel's signature, so :func:`_select` returns the type it was given."""

_Kernel = Callable[..., None]
"""Every kernel here fills the caller's buffers and returns ``None``."""


_KERNELS: Final[dict[tuple[str, int], _Kernel]] = {
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
"""The single-cell Numba kernels, keyed by ``(op_kind, dimension)``.

Defined here rather than in :mod:`pantr.bspline._extraction_helpers`, which
re-exports it under its original name: a catalogue that selects between two
implementations has to hold the Python one, and holding it twice is how the two
tables drift.
"""

_KERNELS_MANY: Final[dict[tuple[str, int], _Kernel]] = {
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
"""The batch Numba kernels, keyed by ``(op_kind, dimension)``."""


_CPP_NAMES: Final[dict[tuple[str, int], str]] = (
    {("apply", d): f"apply_kron_{d}d" for d in (1, 2, 3)}
    | {("apply_T", d): f"apply_kron_T_{d}d" for d in (1, 2, 3)}
    | {("MT_K_M", d): f"apply_kron_MT_K_M_{d}d" for d in (1, 2, 3)}
    | {("M_K_MT", d): f"apply_kron_M_K_MT_{d}d" for d in (1, 2, 3)}
)
"""The C++ binding names, which are the Numba names. Kept explicit rather than
derived at the call site, so a rename shows up here as a mismatch."""

_CPP_NAMES_MANY: Final[dict[tuple[str, int], str]] = (
    {("apply", d): f"apply_kron_apply_many_{d}d" for d in (1, 2, 3)}
    | {("apply_T", d): f"apply_kron_apply_T_many_{d}d" for d in (1, 2, 3)}
    | {("MT_K_M", d): f"apply_kron_MT_K_M_many_{d}d" for d in (1, 2, 3)}
    | {("M_K_MT", d): f"apply_kron_M_K_MT_many_{d}d" for d in (1, 2, 3)}
)
"""The batch C++ binding names."""


def _select(backend: Backend | None, python_kernel: _K, cpp_kernel: _K) -> _K:
    """Pick one kernel's implementation for the requested backend.

    The one place the never-fall-back rule is applied for this package, so the two
    accessors below cannot drift from each other on it.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect, per :func:`pantr._backend.active_backend`.
        python_kernel (_K): The Numba implementation, always available.
        cpp_kernel (_K): The C++ implementation, available only when the extension
            was built.

    Returns:
        _K: The kernel of the chosen backend.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    chosen = active_backend() if backend is None else backend

    if chosen is Backend.PYTHON:
        return python_kernel

    if chosen not in available_backends():
        raise RuntimeError(f"the {chosen.name} backend is not available in this installation")
    return cpp_kernel


def _contiguous_out(
    array: npt.NDArray[np.float32 | np.float64],
) -> tuple[npt.NDArray[np.float32 | np.float64], bool]:
    """Give the binding a C-contiguous buffer to fill, copying back if needed.

    The bindings declare their writable arguments ``c_contig``, while Layer 2 will
    hand through whatever ``out`` the caller supplied. A strided one is rare and
    legal, so it is absorbed here rather than refused.

    Args:
        array (npt.NDArray[np.float32 | np.float64]): The caller's array.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], bool]: The buffer to pass, and
        whether the result has to be copied back into ``array`` afterwards.
    """
    if array.flags["C_CONTIGUOUS"]:
        return array, False
    return np.empty_like(array, order="C"), True


def _cpp_adapter(binding_name: str, n_directions: int, *, batch: bool) -> _Kernel:
    """Build an adapter from a C++ binding to the Numba kernel's calling convention.

    The binding's argument order, arity and names already mirror the Numba kernel's,
    so the adapter exists only to absorb what a ``c_contig`` annotation cannot: a
    strided operand, output or scratch buffer.

    Args:
        binding_name (str): Attribute of :mod:`pantr._pantr_cpp` to call.
        n_directions (int): How many leading operator arguments the kernel takes.
        batch (bool): Whether this is a batch kernel, which carries index maps and
            masks between the operators and the operand.

    Returns:
        _Kernel: The adapter, callable exactly as the Numba kernel is.
    """
    # Per direction the per-cell kernels take one operator and one flag; the batch
    # kernels take a stack, an index map and a mask, plus the cell-index block.
    n_leading = 3 * n_directions + 1 if batch else 2 * n_directions

    def adapter(*args: object) -> None:
        """Call the binding, absorbing non-contiguous buffers.

        Args:
            *args: The Numba kernel's positional arguments, unchanged.

        Note:
            No input validation is performed here; the binding checks dtype, rank,
            contiguity and the relations between shapes, and Layer 2 established
            the rest.
        """
        from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

        kernel = getattr(_pantr_cpp, binding_name)
        leading = list(args[:n_leading])
        operand, out, scratch = args[n_leading:]

        operand = np.ascontiguousarray(operand)
        out_buffer, copy_out = _contiguous_out(out)  # type: ignore[arg-type]
        scratch_buffer, _ = _contiguous_out(scratch)  # type: ignore[arg-type]

        kernel(*leading, operand, out_buffer, scratch_buffer)
        if copy_out:
            out[...] = out_buffer  # type: ignore[index]

    return adapter


def apply_kernel(op_kind: OpKind, dim: int, backend: Backend | None = None) -> _Kernel:
    """Get the single-cell kernel for one operation kind and dimension.

    Args:
        op_kind (OpKind): Which apply variant.
        dim (int): Number of tensor-product directions, in ``{1, 2, 3}``.
        backend (Backend | None): The backend to use, or ``None`` for the one
            currently in effect. Defaults to None.

    Returns:
        _Kernel: The kernel, callable as
        ``(M_0, …, is_id_0, …, operand, out, scratch) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    key = (op_kind, dim)
    return _select(backend, _KERNELS[key], _cpp_adapter(_CPP_NAMES[key], dim, batch=False))


def apply_many_kernel(op_kind: OpKind, dim: int, backend: Backend | None = None) -> _Kernel:
    """Get the batch kernel for one operation kind and dimension.

    Args:
        op_kind (OpKind): Which apply variant.
        dim (int): Number of tensor-product directions, in ``{1, 2, 3}``.
        backend (Backend | None): The backend to use, or ``None`` for the one
            currently in effect. Defaults to None.

    Returns:
        _Kernel: The kernel, callable as
        ``(ops_0, …, idx_map_0, …, is_id_0, …, cell_indices, operand, out, scratch)
        -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    key = (op_kind, dim)
    return _select(backend, _KERNELS_MANY[key], _cpp_adapter(_CPP_NAMES_MANY[key], dim, batch=True))
