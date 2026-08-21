"""Which implementation of a quadrature rule runs: the Python one or the C++ port.

:mod:`pantr._backend` owns the **policy** -- which backends exist, which one is
selected, and the rule that an explicit request never falls back. This module
owns the **catalogue** for the quad package, exactly as
:mod:`pantr.basis._basis_backend` does for its own kernels, and for the same
reason: a catalogue has to import the kernels it hands out, so keeping each
catalogue next to its own kernels is what stops the policy module from importing
the library it is supposed to be independent of. An import-linter contract in
``pyproject.toml`` holds that line.

What crosses the boundary
-------------------------

Only arrays and scalars, per ``design/cross_backend_types.md``. There is no
``pantr::PointsLattice`` and no C++ counterpart of any pantr type: the dispatch
is defined for functions, and two backends cannot own two different classes that
are then passed between them. Every record in this module -- :class:`QuadKernels`
and the arrays it hands back -- is Python-owned and stays on this side.

One accessor per kernel, and when a record is right instead
-----------------------------------------------------------

**The rule, which every module ported after this one inherits: a catalogue
returns a record when the consumer needs more than one kernel at once, and a
bare callable when it does not.**

:mod:`pantr.basis._basis_backend` needs the record. Its
:class:`~pantr.basis._basis_backend.CoreKernels` carries a parallel kernel and
its optional serial twin, and
:func:`pantr.basis._basis_1D._tabulate_basis_1D_impl_helper` chooses between
them per call on ``_PARALLEL_MIN_NUM_PTS``. Two kernels reach one consumer, so
one object carries both.

Quadrature has no such split. A rule generator builds an ``n``-point rule once
and the result is cached, so there is no threshold to dispatch on and no twin to
carry, and every consumer in :mod:`pantr.quad._rules` takes exactly one kernel.
Five accessors say that; a five-field record says the opposite, and the two
shapes then mean nothing, because the same structure would stand for "these
belong together" in one package and "these happen to live nearby" in the other.

An earlier version of this module argued the record made it impossible to
express a rule whose nodes came from one backend and whose weights from another.
That argument does not hold, and it is recorded here so it is not reinvented:
every rule kernel returns the pair from a single call, so its **signature**
already forbids the mix. Nor did the record buy atomicity across kernels. The
catalogue reads :func:`pantr._backend.active_backend` when it is called, so five
calls behave exactly as five accessors do, and no consumer ever held two fields
at once for a backend switch to land between.

What a catalogue entry looks like
---------------------------------

**The entry mirrors its module's public surface**, and the criterion is that
neither shape introduces a copy between Layer 2 and Layer 3.

:mod:`pantr.basis` takes ``out`` on its public functions, so its kernels take
the caller's buffer and fill it. No public function in :mod:`pantr.quad` takes
``out``: every rule getter constructs its arrays and returns them, because a
rule is built once and cached and there is no caller-owned buffer to fill. So a
quad kernel allocates and returns. The two conventions differ because the two
public surfaces differ, not because either drifted.

- :func:`gauss_legendre_kernel`, :func:`lambert_w_kernel`,
  :func:`tanh_sinh_kernel`, :func:`trapezoidal_kernel`,
  :func:`chebyshev_nodes_kernel`: one rule kernel each, from the requested
  backend.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .._backend import Backend, active_backend, available_backends
from ._rules_core import (
    _gauss_legendre_symmetric_core,
    _generate_tanh_sinh_core,
    _lambert_w_principal_core,
    _modified_chebyshev_nodes_core,
    _trapezoidal_core,
)

_K = TypeVar("_K")
"""One kernel's signature, so :func:`_select` returns the type it was given."""

_GaussLegendreFunc = Callable[[int], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]
"""Signature of the Gauss-Legendre kernel: ``(n) -> (nodes, weights)`` on ``[-1, 1]``."""

_LambertWFunc = Callable[[float], float]
"""Signature of the Lambert W kernel: ``(x) -> W(x)`` on the principal branch."""

_TanhSinhFunc = Callable[[int, float], tuple[npt.NDArray[np.float64], int]]
"""Signature of the tanh-sinh kernel: ``(n, min_gap) -> (data, count)``.

The count is returned rather than implied because the output length is not a
function of the inputs: generation stops at the last node whose distance to an
endpoint is still representable, so it may be below ``n``. This is the one seam
in the package whose shape differs from the others, and
``design/cross_backend_types.md`` records it.
"""

_TrapezoidalFunc = Callable[[int], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]
"""Signature of the trapezoidal kernel: ``(n) -> (nodes, weights)`` on ``[0, 1]``."""

_ChebyshevNodesFunc = Callable[[int, npt.DTypeLike], npt.NDArray[np.float32 | np.float64]]
"""Signature of the modified Chebyshev kernel: ``(n, dtype) -> nodes`` on ``[0, 1]``.

The only kernel in the package that takes a dtype, and it is a genuine argument
rather than a formatting one: the Python computes in the storage format, so a
float32 request is float32 *arithmetic* and not a narrowed float64 result. The
two were measured to differ on 62% of arguments.
"""


def _cpp_gauss_legendre(n: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Build the Gauss-Legendre rule through the C++ kernel.

    Args:
        n (int): Number of points. Must be at least 1.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Nodes ascending
            in ``(-1, 1)`` and their weights.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (lazy: the extension is optional)

    nodes = np.empty(n, dtype=np.float64)
    weights = np.empty(n, dtype=np.float64)
    _pantr_cpp.gauss_legendre_symmetric(n, out_nodes=nodes, out_weights=weights)
    return nodes, weights


def _cpp_lambert_w(x: float) -> float:
    """Solve ``w e^w = x`` through the C++ kernel.

    Args:
        x (float): The argument. Must be at least about 1.61.

    Returns:
        float: ``W(x)`` on the principal branch.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (lazy: the extension is optional)

    return float(_pantr_cpp.lambert_w_principal(float(x)))


def _cpp_tanh_sinh(n: int, min_gap: float) -> tuple[npt.NDArray[np.float64], int]:
    """Build the tanh-sinh rule through the C++ kernel.

    The buffers are sized for the worst case, ``n`` nodes, and sliced to the
    count the kernel reports. The copy is deliberate rather than a slice view:
    the caller keeps the result, and a view would hold the whole worst-case
    allocation alive behind it.

    Args:
        n (int): Requested number of points. Must be at least 1.
        min_gap (float): Smallest endpoint distance a node may carry, in the
            frame the rule will be returned in.

    Returns:
        tuple[npt.NDArray[np.float64], int]: A pair ``(data, m)`` where *data*
            has shape ``(m, 2)`` with columns ``[node, weight]`` on ``[-1, 1]``,
            and *m* is the effective node count.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (lazy: the extension is optional)

    nodes = np.empty(n, dtype=np.float64)
    weights = np.empty(n, dtype=np.float64)
    count = int(
        _pantr_cpp.generate_tanh_sinh(n, float(min_gap), out_nodes=nodes, out_weights=weights)
    )

    data = np.empty((count, 2), dtype=np.float64)
    data[:, 0] = nodes[:count]
    data[:, 1] = weights[:count]
    return data, count


def _cpp_trapezoidal(n: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Build the trapezoidal rule through the C++ kernel.

    Args:
        n (int): Number of points. Must be at least 1.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Nodes ascending
            from 0 to 1, and their weights.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (lazy: the extension is optional)

    nodes = np.empty(n, dtype=np.float64)
    weights = np.empty(n, dtype=np.float64)
    _pantr_cpp.trapezoidal(n, out_nodes=nodes, out_weights=weights)
    return nodes, weights


def _cpp_chebyshev_nodes(n: int, dtype: npt.DTypeLike) -> npt.NDArray[np.float32 | np.float64]:
    """Build the modified Chebyshev nodes through the C++ kernel.

    The output array's dtype selects the binding overload, and with it the
    arithmetic the kernel computes in. That is the point: a float32 request is
    float32 arithmetic on both sides, not a narrowed float64 result.

    Args:
        n (int): Number of nodes. Must be at least 2.
        dtype (npt.DTypeLike): Floating dtype; float32 or float64.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Nodes from 0 to 1.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (lazy: the extension is optional)

    nodes: npt.NDArray[np.float32 | np.float64] = np.empty(n, dtype=dtype)
    _pantr_cpp.modified_chebyshev_nodes(n, out=nodes)
    return nodes


def _select(backend: Backend | None, python_kernel: _K, cpp_kernel: _K) -> _K:
    """Pick one kernel's implementation for the requested backend.

    The one place the never-fall-back rule is applied for this package, so the
    five accessors below cannot drift from each other on it.

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


def gauss_legendre_kernel(backend: Backend | None = None) -> _GaussLegendreFunc:
    """Get the Gauss-Legendre kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        _GaussLegendreFunc: The kernel, ``(n) -> (nodes, weights)`` on ``[-1, 1]``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _gauss_legendre_symmetric_core, _cpp_gauss_legendre)


def lambert_w_kernel(backend: Backend | None = None) -> _LambertWFunc:
    """Get the Lambert W kernel of the requested backend.

    Exposed for its own sake rather than for a consumer: the tanh-sinh kernels
    each solve their step size internally, so nothing in the library calls this
    one. It is dispatched so that the two implementations of a function the
    rules depend on can be compared directly, instead of only through the rule
    that hides it.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        _LambertWFunc: The kernel, ``(x) -> W(x)`` on the principal branch.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _lambert_w_principal_core, _cpp_lambert_w)


def tanh_sinh_kernel(backend: Backend | None = None) -> _TanhSinhFunc:
    """Get the tanh-sinh kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        _TanhSinhFunc: The kernel, ``(n, min_gap) -> (data, count)``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _generate_tanh_sinh_core, _cpp_tanh_sinh)


def trapezoidal_kernel(backend: Backend | None = None) -> _TrapezoidalFunc:
    """Get the trapezoidal kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        _TrapezoidalFunc: The kernel, ``(n) -> (nodes, weights)`` on ``[0, 1]``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _trapezoidal_core, _cpp_trapezoidal)


def chebyshev_nodes_kernel(backend: Backend | None = None) -> _ChebyshevNodesFunc:
    """Get the modified Chebyshev nodes kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect. Defaults to None.

    Returns:
        _ChebyshevNodesFunc: The kernel, ``(n, dtype) -> nodes`` on ``[0, 1]``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    return _select(backend, _modified_chebyshev_nodes_core, _cpp_chebyshev_nodes)
