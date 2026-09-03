"""Which implementation of an extraction kernel runs: the Numba one or C++.

:mod:`pantr._backend` owns the **policy**. This module owns the **catalogue** for
the extraction kernels, exactly as :mod:`pantr.change_basis._change_basis_backend`
and :mod:`pantr.basis._basis_backend` do for theirs, and for the same reason: a
catalogue imports the kernels it hands out, so keeping each one next to its own
kernels is what stops the policy module from importing the library it must stay
independent of.

Four accessors, not a record
----------------------------

By the rule ``design/cross_backend_types.md`` states -- a record when the consumer
needs more than one kernel at once, a bare callable when it does not -- these are
bare callables. :func:`pantr.bspline._extraction_helpers._prepare_apply_call`
selects exactly one kernel per call, keyed by ``(op_kind, dim)``, and never needs
a second one in the same call; and neither of the two builders below needs the
other.

Two of the four *apply* an operator and two *build* one, which is why the C++ side
is two registrations rather than one -- ``bspline_extraction.cpp`` for the
tensor-product apply kernels, ``bspline_extraction_operators.cpp`` for the Bézier
builder and its mask. They are separate ports with separate parity claims.

Only the **Bézier** target has a builder here. Lagrange is that operator
post-multiplied by ``lagrange_to_bernstein_1d``, which
:mod:`pantr.change_basis._change_basis_backend` already dispatches, so it inherits
the backend rather than needing an entry; the cardinal target additionally needs
the cardinal-interval scan, which is not ported and which
``cpp/include/pantr/bspline/space_1d.hpp`` deliberately keeps off the type.

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
``tests/parity/test_extraction_kernels.py`` carries the claim and its gate, and
measures it over **both** halves of the surface -- the twelve single-cell entry
points and the twelve batch ones are twelve different C++ functions each, so a
claim measured on one says nothing about the other.

Where the two backends differ, and it is one input
-------------------------------------------------

The Bézier builder's C++ half **refuses a knot vector spanning no in-domain
interval**, with the message :class:`pantr.bspline.BsplineSpace1D` raises for the
same vector. The Numba half accepts it: Layer 2 allocates an empty
``(0, degree+1, degree+1)`` result and the kernel then indexes ``out[0]`` on it
whenever the boundary multiplicity is short of ``degree + 1``, which is out of
bounds on an empty array. Refusing is the only thing the C++ side can do that is not
undefined behaviour, so the divergence is deliberate and recorded here rather than
left to be met. Such a vector is unreachable through the public API, since
:class:`~pantr.bspline.BsplineSpace1D` refuses it at construction; only the private
Layer 2 helper accepts one.

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
from ._bspline_extraction_core import (
    _bezier_structural_identity_mask_core,
    _tabulate_Bspline_Bezier_1D_extraction_core,
)
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

_Array = npt.NDArray[np.float32 | np.float64]
"""A float32 or float64 array, the two dtypes these kernels handle."""

_BezierBuilder = Callable[[_Array, int, float, _Array], None]
"""Signature of the Bézier operator builder: ``(knots, degree, tol, out) -> None``."""

_IdentityMask = Callable[[npt.NDArray[np.intp], int, npt.NDArray[np.bool_]], None]
"""Signature of the structural identity mask: ``(multiplicities, degree, out) -> None``."""


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


def _cpp_bezier_extraction(knots: _Array, degree: int, tol: float, out: _Array) -> None:
    """Build the Bézier extraction operators through the C++ binding.

    Args:
        knots (_Array): The knot vector.
        degree (int): The polynomial degree.
        tol (float): The absolute parametric tolerance.
        out (_Array): Output of shape ``(n_intervals, degree + 1, degree + 1)``.

    Note:
        No input validation is performed here. The binding is the C++ half of
        Layer 2 and re-checks dtype, rank, contiguity, the oracle's three
        knot-vector refusals and the shape of ``out``; Layer 2 in Python
        established the rest.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    knot_array = np.ascontiguousarray(knots)
    buffer, copy_back = _contiguous_out(out)
    _pantr_cpp.bezier_extraction_1d(knot_array, int(degree), float(tol), buffer)
    if copy_back:
        out[...] = buffer


def _cpp_bezier_identity_mask(
    multiplicities: npt.NDArray[np.intp], degree: int, out: npt.NDArray[np.bool_]
) -> None:
    """Mark the identity elements through the C++ binding.

    The multiplicities are normalized to :data:`numpy.intp` rather than passed
    through: the oracle's knot scan returns :data:`numpy.int_`, which is ``int64``
    on the platforms this is built for but ``int32`` on Windows, and the binding
    takes ``int64`` under ``.noconvert()``. The cast is a no-op wherever the two
    already agree.

    Args:
        multiplicities (npt.NDArray[np.intp]): In-domain knot multiplicities.
        degree (int): The polynomial degree.
        out (npt.NDArray[np.bool_]): One flag per element.

    Note:
        No input validation is performed here; the binding re-checks dtype, rank,
        contiguity and the length relation, and Layer 2 established the rest.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    counts = np.ascontiguousarray(multiplicities, dtype=np.intp)
    if out.flags["C_CONTIGUOUS"]:
        _pantr_cpp.bezier_structural_identity_mask(counts, int(degree), out)
        return
    buffer = np.empty_like(out, order="C")
    _pantr_cpp.bezier_structural_identity_mask(counts, int(degree), buffer)
    out[...] = buffer


def bezier_extraction_kernel(backend: Backend | None = None) -> _BezierBuilder:
    """Get the Bézier extraction operator builder of the requested backend.

    Args:
        backend (Backend | None): The backend to use, or ``None`` for the one
            currently in effect. Defaults to None.

    Returns:
        _BezierBuilder: The kernel, ``(knots, degree, tol, out) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _tabulate_Bspline_Bezier_1D_extraction_core, _cpp_bezier_extraction)


def bezier_identity_mask_kernel(backend: Backend | None = None) -> _IdentityMask:
    """Get the structural identity mask kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use, or ``None`` for the one
            currently in effect. Defaults to None.

    Returns:
        _IdentityMask: The kernel, ``(multiplicities, degree, out) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _bezier_structural_identity_mask_core, _cpp_bezier_identity_mask)
