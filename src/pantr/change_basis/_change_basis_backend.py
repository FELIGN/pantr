"""Which implementation of a change-of-basis builder runs: the Python one or C++.

:mod:`pantr._backend` owns the **policy**. This module owns the **catalogue** for
the change_basis package, exactly as :mod:`pantr.quad._quad_backend` and
:mod:`pantr.basis._basis_backend` do for theirs, and for the same reason: a
catalogue imports the kernels it hands out, so keeping each one next to its own
kernels is what stops the policy module from importing the library it must stay
independent of.

Eight accessors, not a record
-----------------------------

By the rule ``design/cross_backend_types.md`` states -- a record when the consumer
needs more than one kernel at once, a bare callable when it does not -- these are
bare callables. No builder in :mod:`pantr.change_basis._builders` uses two kernels
in one call: the inverse builders invert their forward matrix *inside* the kernel,
on the same side of the boundary, precisely so that a C++ inverse is not a Python
forward matrix inverted in C++.

That last point is the one worth stating, because the alternative is tempting and
wrong. ``compute_cardinal_to_bernstein_1d`` could have called
``compute_bernstein_to_cardinal_1d`` and then solved, which is how the Python
module was written before the port. Under a mixed backend that composes two
different implementations into a result that is neither, and no parity claim
covers it.

What crosses the boundary
-------------------------

Arrays and scalars only. In particular
:class:`~pantr.basis.LagrangeVariant` does **not**: the builders resolve a variant
to its nodes on this side and pass the array. That is not only the type rule --
:mod:`pantr._backend` records that two of the five node families are deliberately
never dispatched and stay on :mod:`numpy.polynomial`, so a kernel that resolved
the variant itself would have to reach back through the dispatch to honour that.

What a catalogue entry looks like
---------------------------------

Every public builder in this package takes ``out``, so by the mirroring rule every
kernel here fills the caller's buffer and returns ``None``. That is
:mod:`pantr.basis`'s convention rather than :mod:`pantr.quad`'s, and it follows
from the public surface rather than from either package's habit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, TypeVar

import numpy as np
import numpy.typing as npt

from .._backend import Backend, active_backend, available_backends
from ._change_basis_core import (
    _bernstein_to_cardinal_core,
    _bernstein_to_lagrange_core,
    _cardinal_dual_legendre_coeffs_core,
    _cardinal_to_bernstein_core,
    _cardinal_to_legendre_core,
    _lagrange_to_bernstein_core,
    _legendre_to_cardinal_core,
    _monomial_to_bernstein_core,
)

_K = TypeVar("_K")
"""One kernel's signature, so :func:`_select` returns the type it was given."""

_Array = npt.NDArray[np.float32 | np.float64]
"""A float32 or float64 array, the only two dtypes these kernels handle."""

_FromNodesFunc = Callable[[int, _Array, _Array], None]
"""Signature of a builder driven by nodes alone: ``(degree, nodes, out) -> None``."""

_FromRuleFunc = Callable[[int, _Array, _Array, _Array], None]
"""Signature of a builder driven by a quadrature rule.

``(degree, nodes, weights, out) -> None``. The nodes and weights stay two arrays
rather than one ``(n, 2)`` block because that is the shape
:func:`pantr.quad.get_gauss_legendre_1d` returns, and re-packing them would put a
copy in front of every call.
"""

_ClosedFormFunc = Callable[[int, _Array], None]
"""Signature of the one builder that needs nothing but a degree: ``(degree, out) -> None``."""


def _select(backend: Backend | None, python_kernel: _K, cpp_kernel: _K) -> _K:
    """Pick one kernel's implementation for the requested backend.

    The one place the never-fall-back rule is applied for this package, so the
    eight accessors below cannot drift from each other on it.

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


def _cpp_from_nodes(binding_name: str) -> _FromNodesFunc:
    """Build an adapter from a node-driven binding to :data:`_FromNodesFunc`.

    Args:
        binding_name (str): Attribute of :mod:`pantr._pantr_cpp` to call.

    Returns:
        _FromNodesFunc: The adapter, callable as ``(degree, nodes, out) -> None``.
    """

    def adapter(degree: int, nodes: _Array, out: _Array) -> None:
        """Build the matrix through the C++ binding, absorbing a strided ``out``.

        Args:
            degree (int): Polynomial degree.
            nodes (_Array): The ``degree + 1`` nodes.
            out (_Array): Output of shape ``(degree + 1, degree + 1)``.

        Note:
            No input validation is performed here; the binding re-checks dtype,
            rank and contiguity, and Layer 2 established the rest.
        """
        from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

        kernel = getattr(_pantr_cpp, binding_name)
        node_array = np.ascontiguousarray(nodes)
        if out.flags["C_CONTIGUOUS"]:
            kernel(int(degree), nodes=node_array, out=out)
            return
        buffer = np.empty_like(out, order="C")
        kernel(int(degree), nodes=node_array, out=buffer)
        out[...] = buffer

    return adapter


def _cpp_from_rule(binding_name: str) -> _FromRuleFunc:
    """Build an adapter from a rule-driven binding to :data:`_FromRuleFunc`.

    Args:
        binding_name (str): Attribute of :mod:`pantr._pantr_cpp` to call.

    Returns:
        _FromRuleFunc: The adapter, ``(degree, nodes, weights, out) -> None``.
    """

    def adapter(degree: int, nodes: _Array, weights: _Array, out: _Array) -> None:
        """Build the matrix through the C++ binding, absorbing a strided ``out``.

        Args:
            degree (int): Polynomial degree.
            nodes (_Array): Quadrature nodes, ``degree + 1`` of them.
            weights (_Array): The matching weights.
            out (_Array): Output of shape ``(degree + 1, degree + 1)``.

        Note:
            No input validation is performed here; the binding re-checks dtype,
            rank and contiguity, and Layer 2 established the rest.
        """
        from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

        kernel = getattr(_pantr_cpp, binding_name)
        node_array = np.ascontiguousarray(nodes)
        weight_array = np.ascontiguousarray(weights)
        if out.flags["C_CONTIGUOUS"]:
            kernel(int(degree), nodes=node_array, weights=weight_array, out=out)
            return
        buffer = np.empty_like(out, order="C")
        kernel(int(degree), nodes=node_array, weights=weight_array, out=buffer)
        out[...] = buffer

    return adapter


def _cpp_monomial_to_bernstein(degree: int, out: _Array) -> None:
    """Build the monomial-to-Bernstein matrix through the C++ binding.

    Args:
        degree (int): Polynomial degree.
        out (_Array): Output of shape ``(degree + 1, degree + 1)``.

    Note:
        No input validation is performed here; the binding re-checks dtype, rank
        and contiguity, and Layer 2 established the rest.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    if out.flags["C_CONTIGUOUS"]:
        _pantr_cpp.monomial_to_bernstein_1d(int(degree), out=out)
        return
    buffer = np.empty_like(out, order="C")
    _pantr_cpp.monomial_to_bernstein_1d(int(degree), out=buffer)
    out[...] = buffer


_cpp_lagrange_to_bernstein: Final[_FromNodesFunc] = _cpp_from_nodes("lagrange_to_bernstein_1d")
"""The Lagrange-to-Bernstein builder of the C++ backend."""

_cpp_bernstein_to_lagrange: Final[_FromNodesFunc] = _cpp_from_nodes("bernstein_to_lagrange_1d")
"""The Bernstein-to-Lagrange builder of the C++ backend."""

_cpp_bernstein_to_cardinal: Final[_FromRuleFunc] = _cpp_from_rule("bernstein_to_cardinal_1d")
"""The Bernstein-to-cardinal builder of the C++ backend."""

_cpp_cardinal_to_bernstein: Final[_FromRuleFunc] = _cpp_from_rule("cardinal_to_bernstein_1d")
"""The cardinal-to-Bernstein builder of the C++ backend."""

_cpp_legendre_to_cardinal: Final[_FromRuleFunc] = _cpp_from_rule("legendre_to_cardinal_1d")
"""The Legendre-to-cardinal builder of the C++ backend."""

_cpp_cardinal_to_legendre: Final[_FromRuleFunc] = _cpp_from_rule("cardinal_to_legendre_1d")
"""The cardinal-to-Legendre builder of the C++ backend."""

_cpp_cardinal_dual_legendre_coeffs: Final[_FromRuleFunc] = _cpp_from_rule(
    "cardinal_dual_legendre_coeffs_1d"
)
"""The cardinal-dual Legendre builder of the C++ backend."""


def lagrange_to_bernstein_kernel(backend: Backend | None = None) -> _FromNodesFunc:
    """Get the Lagrange-to-Bernstein kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        _FromNodesFunc: The kernel, ``(degree, nodes, out) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _lagrange_to_bernstein_core, _cpp_lagrange_to_bernstein)


def bernstein_to_lagrange_kernel(backend: Backend | None = None) -> _FromNodesFunc:
    """Get the Bernstein-to-Lagrange kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        _FromNodesFunc: The kernel, ``(degree, nodes, out) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _bernstein_to_lagrange_core, _cpp_bernstein_to_lagrange)


def bernstein_to_cardinal_kernel(backend: Backend | None = None) -> _FromRuleFunc:
    """Get the Bernstein-to-cardinal kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        _FromRuleFunc: The kernel, ``(degree, nodes, weights, out) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _bernstein_to_cardinal_core, _cpp_bernstein_to_cardinal)


def cardinal_to_bernstein_kernel(backend: Backend | None = None) -> _FromRuleFunc:
    """Get the cardinal-to-Bernstein kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        _FromRuleFunc: The kernel, ``(degree, nodes, weights, out) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _cardinal_to_bernstein_core, _cpp_cardinal_to_bernstein)


def legendre_to_cardinal_kernel(backend: Backend | None = None) -> _FromRuleFunc:
    """Get the Legendre-to-cardinal kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        _FromRuleFunc: The kernel, ``(degree, nodes, weights, out) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _legendre_to_cardinal_core, _cpp_legendre_to_cardinal)


def cardinal_to_legendre_kernel(backend: Backend | None = None) -> _FromRuleFunc:
    """Get the cardinal-to-Legendre kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        _FromRuleFunc: The kernel, ``(degree, nodes, weights, out) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _cardinal_to_legendre_core, _cpp_cardinal_to_legendre)


def cardinal_dual_legendre_coeffs_kernel(backend: Backend | None = None) -> _FromRuleFunc:
    """Get the cardinal-dual Legendre coefficients kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        _FromRuleFunc: The kernel, ``(degree, nodes, weights, out) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(
        backend,
        _cardinal_dual_legendre_coeffs_core,
        _cpp_cardinal_dual_legendre_coeffs,
    )


def monomial_to_bernstein_kernel(backend: Backend | None = None) -> _ClosedFormFunc:
    """Get the monomial-to-Bernstein kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        _ClosedFormFunc: The kernel, ``(degree, out) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _monomial_to_bernstein_core, _cpp_monomial_to_bernstein)
