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

Why one record rather than five catalogue functions
---------------------------------------------------

:mod:`pantr.basis._basis_backend` publishes one catalogue function per
tabulation, because each returns a :class:`~pantr.basis._basis_backend.CoreKernels`
carrying a parallel kernel and an optional serial twin, and which of the two runs
is a per-call decision. Quadrature has no such split: a rule generator builds an
``n``-point rule once and is then cached, so there is nothing to dispatch on
below a threshold and no twin to carry.

What the five *do* share is that they are ported and shipped as a unit and are
always taken from the same backend -- a rule whose nodes came from one
implementation and whose weights came from another would be a bug no test looks
for. One record makes that impossible to express, and makes "which backend built
this rule" a single question rather than five.

- :class:`QuadKernels`: the five rule kernels of one backend.
- :func:`quad_kernels`: the kernels of the requested backend.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

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
two were measured to differ on 17% of arguments.
"""


class QuadKernels(NamedTuple):
    """The five quadrature rule kernels, in one backend.

    Attributes:
        gauss_legendre (_GaussLegendreFunc): Gauss-Legendre nodes and weights on
            ``[-1, 1]``, in float64 whatever the caller will narrow to.
        lambert_w (_LambertWFunc): The principal branch of Lambert W, which the
            tanh-sinh step size is solved from.
        tanh_sinh (_TanhSinhFunc): The tanh-sinh rule on ``[-1, 1]``, returning
            its own effective node count.
        trapezoidal (_TrapezoidalFunc): The trapezoidal rule, already on
            ``[0, 1]`` and so needing no mapping.
        chebyshev_nodes (_ChebyshevNodesFunc): The modified Chebyshev
            interpolation nodes on ``[0, 1]``, computed in the storage format.
    """

    gauss_legendre: _GaussLegendreFunc
    lambert_w: _LambertWFunc
    tanh_sinh: _TanhSinhFunc
    trapezoidal: _TrapezoidalFunc
    chebyshev_nodes: _ChebyshevNodesFunc


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
    _pantr_cpp.gauss_legendre_symmetric(n, nodes, weights)
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
    count = int(_pantr_cpp.generate_tanh_sinh(n, float(min_gap), nodes, weights))

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
    _pantr_cpp.trapezoidal(n, nodes, weights)
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
    _pantr_cpp.modified_chebyshev_nodes(n, nodes)
    return nodes


def quad_kernels(backend: Backend | None = None) -> QuadKernels:
    """Return the quadrature rule kernels of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect, per :func:`pantr._backend.active_backend`.
            Defaults to None.

    Returns:
        QuadKernels: The five kernels of that backend.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    chosen = active_backend() if backend is None else backend

    if chosen is Backend.PYTHON:
        return QuadKernels(
            gauss_legendre=_gauss_legendre_symmetric_core,
            lambert_w=_lambert_w_principal_core,
            tanh_sinh=_generate_tanh_sinh_core,
            trapezoidal=_trapezoidal_core,
            chebyshev_nodes=_modified_chebyshev_nodes_core,
        )

    if chosen not in available_backends():
        raise RuntimeError(f"the {chosen.name} backend is not available in this installation")
    return QuadKernels(
        gauss_legendre=_cpp_gauss_legendre,
        lambert_w=_cpp_lambert_w,
        tanh_sinh=_cpp_tanh_sinh,
        trapezoidal=_cpp_trapezoidal,
        chebyshev_nodes=_cpp_chebyshev_nodes,
    )
