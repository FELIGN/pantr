"""Which implementation of a basis kernel runs: the Numba one or the C++ port.

:mod:`pantr._backend` owns the **policy** -- which backends exist, which one is
selected, and the rule that an explicit request never falls back. This module
owns the **catalogue** for the basis package: it maps that selection onto the
callables themselves, and it is where the C++ adapter lives.

The split is what keeps the dependency one-directional. A catalogue has to
import the kernels it hands out, so a catalogue living inside the policy module
makes :mod:`pantr._backend` import a subpackage of :mod:`pantr` -- while
:mod:`pantr.basis._basis_1D` imports the policy to ask which kernel to call.
That cycle used to be held open by a lazy import and a ``TYPE_CHECKING`` one,
and it would have grown by one lazy import per kernel ported. Keeping each
catalogue next to its own kernels removes it instead: the policy module imports
nothing from the library, and an import-linter contract in ``pyproject.toml``
keeps it that way.

- :data:`_BasisCoreFunc`: the signature every 1D tabulation kernel has.
- :func:`cardinal_bspline_core`: the cardinal B-spline kernel of a backend.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from .._backend import Backend, active_backend, available_backends
from ._basis_core import _tabulate_cardinal_Bspline_basis_1D_core

_BasisCoreFunc = Callable[
    [np.int32, npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]],
    None,
]
"""Signature of a 1D basis tabulation core kernel: ``(degree, pts, out) -> None``."""


def _cpp_cardinal_bspline_core(
    n: np.int32,
    t: npt.NDArray[np.float32 | np.float64],
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Adapt the C++ kernel to the :data:`_BasisCoreFunc` signature.

    The adapter's whole job is that **the public API must behave identically on
    both backends**, because otherwise an A/B measurement is comparing two
    contracts rather than two kernels. One difference has to be absorbed here:
    the numba kernel accepts a non-contiguous ``out`` and fills it, while the
    C++ binding requires C-contiguous memory and refuses anything else. So a
    non-contiguous ``out`` is computed into a contiguous buffer and copied back,
    and the caller sees the numba behaviour either way.

    Refusing instead would be the wrong fix: it would make ``PANTR_BACKEND``
    change what the library accepts, not just how fast it is.

    Args:
        n (np.int32): Degree of the basis. Assumed non-negative.
        t (npt.NDArray[np.float32 | np.float64]): 1D evaluation points.
        out (npt.NDArray[np.float32 | np.float64]): Output of shape
            ``(t.size, n + 1)`` and matching dtype. Need not be contiguous.

    Note:
        No input validation is performed. Shape and dtype are established by the
        Layer 2 caller; dtype, rank and contiguity are re-checked by nanobind's
        typed signature, which raises :class:`TypeError` before the kernel runs
        rather than silently converting -- see the ``.noconvert()`` in
        ``cpp/bindings/pantr_cpp.cpp`` and why it is a correctness requirement.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    points = np.ascontiguousarray(t)
    if out.flags["C_CONTIGUOUS"]:
        _pantr_cpp.tabulate_cardinal_bspline_1d(int(n), points, out)
        return

    # The uncommon path. The check above is a flag read, so the common case pays
    # essentially nothing for it; only a caller that actually passes a strided
    # view pays for the buffer and the copy back.
    buffer = np.empty_like(out, order="C")
    _pantr_cpp.tabulate_cardinal_bspline_1d(int(n), points, buffer)
    out[...] = buffer


def cardinal_bspline_core(backend: Backend | None = None) -> _BasisCoreFunc:
    """Return the cardinal B-spline tabulation kernel of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect, per :func:`pantr._backend.active_backend`.
            Defaults to None.

    Returns:
        _BasisCoreFunc: The kernel, callable as ``(n, t, out) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    chosen = active_backend() if backend is None else backend

    if chosen is Backend.NUMBA:
        return _tabulate_cardinal_Bspline_basis_1D_core

    if chosen not in available_backends():
        raise RuntimeError(f"the {chosen.name} backend is not available in this installation")
    return _cpp_cardinal_bspline_core
