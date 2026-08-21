"""Which implementation of a Bézier arithmetic kernel runs: the Numba one or C++.

:mod:`pantr._backend` owns the **policy**. This module owns the **catalogue** for
the bezier package, exactly as :mod:`pantr.basis._basis_backend`,
:mod:`pantr.quad._quad_backend` and
:mod:`pantr.change_basis._change_basis_backend` do for theirs, and for the same
reason: a catalogue imports the kernels it hands out, so keeping each one beside
its own kernels is what stops the policy module from importing the library it has
to stay independent of.

Seven accessors over eight kernels
----------------------------------

Six are bare callables and one is a record, which is the rule
``design/cross_backend_types.md`` states: a record when a consumer needs more
than one kernel at once, a bare callable when it does not.
:func:`~pantr.bezier._bezier_degree._minimize_degree_bezier` is the consumer that
needs two. Inside one call it reduces a degree and then re-elevates the result so
the trial can be compared against the original, so it reaches for the reduction
apply and the elevation in the same breath. It therefore resolves once and takes both.

**Not because two resolutions could disagree.** An earlier version of this docstring
said a scoped switch on another thread could land between them; that is measurably
false, because a :class:`~contextvars.ContextVar` gives each thread its own context
and a task runs in a copy, so no sibling's ``set`` is ever observable here. What
resolving once buys is that the backend is a property of the call rather than of
each lookup, and that the dependency is stated in a signature rather than read from
ambient state. :class:`DegreeKernels` is the convenient shape for two kernels always
fetched together, not a safety mechanism.

The other six call sites use exactly one kernel each, so they get one callable.

The accessors are named ``*_kernel`` rather than ``*_core``, following
:mod:`pantr.quad._quad_backend` and :mod:`pantr.change_basis._change_basis_backend`.
:mod:`pantr.basis._basis_backend` uses ``*_core``, and it is the oldest of the four;
the two written since converged on ``_kernel`` and this is the third to do so.

What crosses the boundary
-------------------------

Arrays and scalars only. The reduction operator is the case worth naming, because
it looks like a type and is not: it is assembled in exact rational arithmetic by
:func:`~pantr.bezier._bezier_degree._l2_reduction_operator` and its interpolating
sibling, cached per degree pair, and converted to ``float64`` before it ever
reaches a kernel. What crosses is that finished matrix.

Every kernel here fills the caller's buffer and returns ``None``, which is
:mod:`pantr.basis`'s convention rather than :mod:`pantr.quad`'s. It follows from
the public surface: :meth:`pantr.bezier.Bezier.evaluate` and
:meth:`~pantr.bezier.Bezier.evaluate_derivatives` take ``out``, and the methods
that do not still hand Layer 2 an array to fill before wrapping it in a
:class:`~pantr.bezier.Bezier`.

- :class:`DegreeKernels`: the elevation and reduction-apply kernels of one backend.
- :func:`evaluate_kernel`, :func:`evaluate_deriv_kernel`, :func:`slice_kernel`,
  :func:`split_kernel`, :func:`restrict_kernel`, :func:`product_kernel`,
  :func:`degree_kernels`: the accessors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, NamedTuple, TypeVar

import numpy as np
import numpy.typing as npt

from .._backend import Backend, active_backend, available_backends
from ..bspline._bspline_degree_core import _apply_reduction_operator
from ._bezier_core import (
    _degree_elevate_bezier_1d_core,
    _evaluate_bezier_1d_core,
    _evaluate_bezier_deriv_1d_core,
    _restrict_bezier_1d_core,
    _scalar_bernstein_product_1d_core,
    _slice_bezier_1d_core,
    _split_bezier_1d_core,
)

_K = TypeVar("_K")
"""One kernel's signature, so :func:`_select` returns the type it was given."""

_Array = npt.NDArray[np.float32 | np.float64]
"""A float32 or float64 array, the only two dtypes these kernels handle."""

_EvaluateFunc = Callable[[_Array, _Array, _Array], None]
"""Signature of the fused evaluation kernel: ``(ctrl, points, out) -> None``."""

_EvaluateDerivFunc = Callable[[_Array, _Array, int, _Array], None]
"""Signature of the derivative kernel: ``(ctrl, points, n_deriv, out) -> None``."""

_ElevateFunc = Callable[[int, _Array, int, _Array], None]
"""Signature of degree elevation: ``(degree, ctrl, degree_increment, out) -> None``."""

_SliceFunc = Callable[[_Array, float, _Array], None]
"""Signature of the single-parameter de Casteljau: ``(ctrl, value, out) -> None``."""

_SplitFunc = Callable[[_Array, float, _Array, _Array], None]
"""Signature of the split: ``(ctrl, value, out_left, out_right) -> None``."""

_RestrictFunc = Callable[[_Array, float, float, _Array], None]
"""Signature of the restriction: ``(ctrl, lower, upper, out) -> None``."""

_ProductFunc = Callable[[_Array, _Array, _Array], None]
"""Signature of the Bernstein product: ``(a, b, out) -> None``."""

_ReduceApplyFunc = Callable[[npt.NDArray[np.float64], _Array, _Array], None]
"""Signature of the reduction-operator apply: ``(operator, ctrl, out) -> None``.

The operator is ``float64`` whatever the control points are, in both backends.
"""


class DegreeKernels(NamedTuple):
    """The two kernels a degree round trip needs, from one backend.

    Attributes:
        elevate (_ElevateFunc): Degree elevation.
        reduce_apply (_ReduceApplyFunc): Application of a reduction operator.
    """

    elevate: _ElevateFunc
    reduce_apply: _ReduceApplyFunc


def _select(backend: Backend | None, python_kernel: _K, cpp_kernel: _K) -> _K:
    """Pick one kernel's implementation for the requested backend.

    The one place the never-fall-back rule is applied for this package, so the
    accessors below cannot drift from each other on it.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect, per :func:`pantr._backend.active_backend`.
        python_kernel (_K): The Python implementation, always available.
        cpp_kernel (_K): The C++ implementation, available only when the
            extension was built.

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


def _fill(out: _Array, run: Callable[[_Array], None]) -> None:
    """Run ``run`` against ``out``, absorbing a non-contiguous destination.

    The numba kernels fill a strided ``out`` in place; the C++ bindings require
    C-contiguous memory and refuse anything else, because ``.noconvert()`` is what
    stops nanobind from filling a temporary and discarding it. So a strided ``out``
    is computed into a contiguous buffer and copied back, and the caller sees the
    numba behaviour on either backend.

    Refusing instead would make ``PANTR_BACKEND`` change what the library accepts
    rather than only how fast it is.

    Args:
        out (_Array): The caller's destination. Need not be contiguous.
        run (Callable[[_Array], None]): Calls the binding with the array it is
            given as the destination.
    """
    if out.flags["C_CONTIGUOUS"]:
        run(out)
        return

    # The uncommon path. The test above is a flag read, so a contiguous caller
    # pays essentially nothing; only one that passes a strided view pays the copy.
    buffer = np.empty_like(out, order="C")
    run(buffer)
    out[...] = buffer


def _cpp_evaluate(ctrl: _Array, points: _Array, out: _Array) -> None:
    """Evaluate through the C++ binding, absorbing non-contiguous arrays.

    Args:
        ctrl (_Array): Control points of shape ``(degree + 1, rank)``.
        points (_Array): 1D evaluation points.
        out (_Array): Output of shape ``(points.size, rank)``.

    Note:
        No input validation is performed here. Shape and dtype were established
        by Layer 2; the binding re-checks dtype, rank and contiguity.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    ctrl_c = np.ascontiguousarray(ctrl)
    pts_c = np.ascontiguousarray(points)
    _fill(out, lambda dest: _pantr_cpp.evaluate_bezier_1d(ctrl_c, pts_c, out=dest))


def _cpp_evaluate_deriv(ctrl: _Array, points: _Array, n_deriv: int, out: _Array) -> None:
    """Evaluate derivatives through the C++ binding, absorbing non-contiguous arrays.

    Args:
        ctrl (_Array): Control points of shape ``(degree + 1, rank)``.
        points (_Array): 1D evaluation points.
        n_deriv (int): Highest derivative order.
        out (_Array): Output of shape ``(points.size, n_deriv + 1, rank)``.

    Note:
        No input validation is performed here; see :func:`_cpp_evaluate`.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    ctrl_c = np.ascontiguousarray(ctrl)
    pts_c = np.ascontiguousarray(points)
    _fill(
        out,
        lambda dest: _pantr_cpp.evaluate_bezier_deriv_1d(ctrl_c, pts_c, int(n_deriv), out=dest),
    )


def _cpp_degree_elevate(degree: int, ctrl: _Array, degree_increment: int, out: _Array) -> None:
    """Degree-elevate through the C++ binding, absorbing non-contiguous arrays.

    Args:
        degree (int): Original degree.
        ctrl (_Array): Control points of shape ``(degree + 1, rank)``.
        degree_increment (int): Degrees to add.
        out (_Array): Output of shape ``(degree + degree_increment + 1, rank)``.

    Note:
        No input validation is performed here; see :func:`_cpp_evaluate`.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    ctrl_c = np.ascontiguousarray(ctrl)
    _fill(
        out,
        lambda dest: _pantr_cpp.degree_elevate_bezier_1d(
            int(degree), ctrl_c, int(degree_increment), out=dest
        ),
    )


def _cpp_slice(ctrl: _Array, value: float, out: _Array) -> None:
    """Evaluate at one parameter through the C++ binding.

    Args:
        ctrl (_Array): Control points of shape ``(degree + 1, n_cols)``.
        value (float): Parameter in ``[0, 1]``.
        out (_Array): Output of shape ``(n_cols,)``.

    Note:
        No input validation is performed here; see :func:`_cpp_evaluate`.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    ctrl_c = np.ascontiguousarray(ctrl)
    _fill(out, lambda dest: _pantr_cpp.slice_bezier_1d(ctrl_c, float(value), out=dest))


def _cpp_split(ctrl: _Array, value: float, out_left: _Array, out_right: _Array) -> None:
    """Split through the C++ binding, absorbing non-contiguous destinations.

    Both halves are absorbed independently, so a caller may pass one contiguous
    and one strided.

    Args:
        ctrl (_Array): Control points of shape ``(degree + 1, n_cols)``.
        value (float): Parameter in ``[0, 1]``.
        out_left (_Array): Left half, shape ``(degree + 1, n_cols)``.
        out_right (_Array): Right half, same shape.

    Note:
        No input validation is performed here; see :func:`_cpp_evaluate`.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    ctrl_c = np.ascontiguousarray(ctrl)
    left_c = out_left if out_left.flags["C_CONTIGUOUS"] else np.empty_like(out_left, order="C")
    right_c = out_right if out_right.flags["C_CONTIGUOUS"] else np.empty_like(out_right, order="C")
    _pantr_cpp.split_bezier_1d(ctrl_c, float(value), out_left=left_c, out_right=right_c)
    if left_c is not out_left:
        out_left[...] = left_c
    if right_c is not out_right:
        out_right[...] = right_c


def _cpp_restrict(ctrl: _Array, lower: float, upper: float, out: _Array) -> None:
    """Restrict through the C++ binding, absorbing non-contiguous arrays.

    Args:
        ctrl (_Array): Control points of shape ``(degree + 1, n_cols)``.
        lower (float): Left bound.
        upper (float): Right bound.
        out (_Array): Output of shape ``(degree + 1, n_cols)``.

    Note:
        No input validation is performed here; see :func:`_cpp_evaluate`.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    ctrl_c = np.ascontiguousarray(ctrl)
    _fill(
        out,
        lambda dest: _pantr_cpp.restrict_bezier_1d(ctrl_c, float(lower), float(upper), out=dest),
    )


def _cpp_product(a: _Array, b: _Array, out: _Array) -> None:
    """Multiply two scalar Béziers through the C++ binding.

    Args:
        a (_Array): Control points of the first curve.
        b (_Array): Control points of the second curve.
        out (_Array): Product control points, shape ``(a.size + b.size - 1,)``.

    Note:
        No input validation is performed here; see :func:`_cpp_evaluate`.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    a_c = np.ascontiguousarray(a)
    b_c = np.ascontiguousarray(b)
    _fill(out, lambda dest: _pantr_cpp.scalar_bernstein_product_1d(a_c, b_c, out=dest))


def _cpp_reduce_apply(operator: npt.NDArray[np.float64], ctrl: _Array, out: _Array) -> None:
    """Apply a reduction operator through the C++ binding.

    Args:
        operator (npt.NDArray[np.float64]): Reduction operator, shape
            ``(q + 1, p + 1)``.
        ctrl (_Array): Control points of shape ``(p + 1, rank)``.
        out (_Array): Output of shape ``(q + 1, rank)``.

    Note:
        No input validation is performed here; see :func:`_cpp_evaluate`.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    op_c = np.ascontiguousarray(operator, dtype=np.float64)
    ctrl_c = np.ascontiguousarray(ctrl)
    _fill(out, lambda dest: _pantr_cpp.apply_reduction_operator(op_c, ctrl_c, out=dest))


# Hoisted to module level rather than rebuilt inside each accessor. An accessor
# that closed over a fresh function object would return a different callable on
# every call, which defeats identity comparison in the tests and re-creates a
# closure in front of a kernel measured in microseconds. The change_basis port
# shipped that bug once.
_CPP_EVALUATE: Final[_EvaluateFunc] = _cpp_evaluate
"""The fused evaluation kernel of the C++ backend."""

_CPP_EVALUATE_DERIV: Final[_EvaluateDerivFunc] = _cpp_evaluate_deriv
"""The derivative-evaluation kernel of the C++ backend."""

_CPP_DEGREE_KERNELS: Final[DegreeKernels] = DegreeKernels(
    elevate=_cpp_degree_elevate, reduce_apply=_cpp_reduce_apply
)
"""The degree round-trip kernels of the C++ backend."""

_PYTHON_DEGREE_KERNELS: Final[DegreeKernels] = DegreeKernels(
    elevate=_degree_elevate_bezier_1d_core, reduce_apply=_apply_reduction_operator
)
"""The degree round-trip kernels of the Numba backend."""

_CPP_SLICE: Final[_SliceFunc] = _cpp_slice
"""The single-parameter de Casteljau of the C++ backend."""

_CPP_SPLIT: Final[_SplitFunc] = _cpp_split
"""The split kernel of the C++ backend."""

_CPP_RESTRICT: Final[_RestrictFunc] = _cpp_restrict
"""The restriction kernel of the C++ backend."""

_CPP_PRODUCT: Final[_ProductFunc] = _cpp_product
"""The Bernstein product of the C++ backend."""


def evaluate_kernel(backend: Backend | None = None) -> _EvaluateFunc:
    """Return the fused evaluation kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        _EvaluateFunc: Callable as ``(ctrl, points, out) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _evaluate_bezier_1d_core, _CPP_EVALUATE)


def evaluate_deriv_kernel(backend: Backend | None = None) -> _EvaluateDerivFunc:
    """Return the derivative-evaluation kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        _EvaluateDerivFunc: Callable as ``(ctrl, points, n_deriv, out) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _evaluate_bezier_deriv_1d_core, _CPP_EVALUATE_DERIV)


def degree_kernels(backend: Backend | None = None) -> DegreeKernels:
    """Return the elevation and reduction-apply kernels of the requested backend.

    The two travel together because
    :func:`~pantr.bezier._bezier_degree._minimize_degree_bezier` reduces and
    re-elevates within a single call; see the module docstring.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        DegreeKernels: Both kernels, from the same backend.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _PYTHON_DEGREE_KERNELS, _CPP_DEGREE_KERNELS)


def slice_kernel(backend: Backend | None = None) -> _SliceFunc:
    """Return the single-parameter de Casteljau kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        _SliceFunc: Callable as ``(ctrl, value, out) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _slice_bezier_1d_core, _CPP_SLICE)


def split_kernel(backend: Backend | None = None) -> _SplitFunc:
    """Return the split kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        _SplitFunc: Callable as ``(ctrl, value, out_left, out_right) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _split_bezier_1d_core, _CPP_SPLIT)


def restrict_kernel(backend: Backend | None = None) -> _RestrictFunc:
    """Return the restriction kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        _RestrictFunc: Callable as ``(ctrl, lower, upper, out) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _restrict_bezier_1d_core, _CPP_RESTRICT)


def product_kernel(backend: Backend | None = None) -> _ProductFunc:
    """Return the Bernstein product kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        _ProductFunc: Callable as ``(a, b, out) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _scalar_bernstein_product_1d_core, _CPP_PRODUCT)
