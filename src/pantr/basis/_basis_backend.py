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
- :class:`CoreKernels`: a tabulation's kernels in one backend, parallel and
  serial.
- :func:`cardinal_bspline_core`: the cardinal B-spline kernels of a backend.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from .._backend import Backend, active_backend, available_backends
from ._basis_core import _tabulate_cardinal_Bspline_basis_1D_core

_BasisCoreFunc = Callable[
    [np.int32, npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]],
    None,
]
"""Signature of a 1D basis tabulation core kernel: ``(degree, pts, out) -> None``."""


class CoreKernels(NamedTuple):
    """The kernels implementing one tabulation, in one backend.

    What a catalogue function returns, and it is a record rather than a bare
    callable because the consumer already needs two of them:
    :func:`pantr.basis._basis_1D._tabulate_basis_1D_impl_helper` dispatches on
    ``_PARALLEL_MIN_NUM_PTS``, calling the serial twin below that many points to
    avoid a parallel launch that costs more than the work. Bernstein is the only
    1D basis tabulation that has such a twin today -- cardinal B-spline, Lagrange
    and Legendre have none -- so a bare callable happens to fit the one kernel
    ported so far and would stop fitting at Bernstein, which is the obvious next
    one. The same concept would then have two return shapes, and every consumer
    would have to know which it got.

    Attributes:
        parallel (_BasisCoreFunc): The kernel used for batches of
            ``_PARALLEL_MIN_NUM_PTS`` points or more.
        serial (_BasisCoreFunc | None): The serial twin, used below that
            threshold. ``None`` when this tabulation has none, in which case
            ``parallel`` runs at every batch size. Defaults to None.
    """

    parallel: _BasisCoreFunc
    serial: _BasisCoreFunc | None = None


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


def cardinal_bspline_core(backend: Backend | None = None) -> CoreKernels:
    """Return the cardinal B-spline tabulation kernels of the requested backend.

    Neither backend has a serial twin for this tabulation, so ``serial`` is
    ``None`` in both. The Numba kernel is ``parallel=True`` and pays the launch
    overhead at every batch size; the C++ one runs on the calling thread.
    Whether a twin is worth adding is a performance question, and
    ``design/simd.md`` open question 4 already asks whether the threshold that
    would select it is derived or measured.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect, per :func:`pantr._backend.active_backend`.
            Defaults to None.

    Returns:
        CoreKernels: The kernels, each callable as ``(n, t, out) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    chosen = active_backend() if backend is None else backend

    if chosen is Backend.PYTHON:
        return CoreKernels(parallel=_tabulate_cardinal_Bspline_basis_1D_core)

    if chosen not in available_backends():
        raise RuntimeError(f"the {chosen.name} backend is not available in this installation")
    return CoreKernels(parallel=_cpp_cardinal_bspline_core)
