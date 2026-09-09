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
- :data:`_BasisDerivCoreFunc`: the signature every 1D derivative tabulation has.
- :class:`DerivKernels`: a derivative tabulation's kernels in one backend.
- :func:`bernstein_deriv_core`: the Bernstein derivative kernels of a backend.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt

from .._backend import Backend, active_backend, available_backends
from ._basis_core import (
    _tabulate_Bernstein_basis_1D_core,
    _tabulate_Bernstein_basis_1D_serial_core,
    _tabulate_Bernstein_basis_deriv_1D_core,
    _tabulate_Bernstein_basis_deriv_1D_serial_core,
    _tabulate_cardinal_Bspline_basis_1D_core,
    _tabulate_Legendre_basis_1D_core,
)

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
        _pantr_cpp.tabulate_cardinal_bspline_1d(int(n), points, out=out)
        return

    # The uncommon path. The check above is a flag read, so the common case pays
    # essentially nothing for it; only a caller that actually passes a strided
    # view pays for the buffer and the copy back.
    buffer = np.empty_like(out, order="C")
    _pantr_cpp.tabulate_cardinal_bspline_1d(int(n), points, out=buffer)
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


def _cpp_tabulation_core(binding_name: str) -> _BasisCoreFunc:
    """Build an adapter from a binding name to the :data:`_BasisCoreFunc` signature.

    The three tabulations share one adapter body because they share one contract:
    same arguments, same output shape, same absorption of a non-contiguous ``out``.
    Generating them keeps a single copy of that contract; three near-identical
    functions is how one of them silently stops absorbing a strided ``out``.

    Args:
        binding_name (str): Attribute of :mod:`pantr._pantr_cpp` to call.

    Returns:
        _BasisCoreFunc: The adapter, callable as ``(n, t, out) -> None``.
    """

    def adapter(
        n: np.int32,
        t: npt.NDArray[np.float32 | np.float64],
        out: npt.NDArray[np.float32 | np.float64],
    ) -> None:
        """Tabulate through the C++ binding, absorbing a non-contiguous ``out``.

        Args:
            n (np.int32): Degree of the basis. Assumed non-negative.
            t (npt.NDArray[np.float32 | np.float64]): 1D evaluation points.
            out (npt.NDArray[np.float32 | np.float64]): Output of shape
                ``(t.size, n + 1)`` and matching dtype. Need not be contiguous.

        Note:
            No input validation is performed. The contract is
            :func:`_cpp_cardinal_bspline_core`'s, in full, including why a
            non-contiguous ``out`` is absorbed rather than refused.
        """
        from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

        kernel = getattr(_pantr_cpp, binding_name)
        points = np.ascontiguousarray(t)
        if out.flags["C_CONTIGUOUS"]:
            kernel(int(n), points, out=out)
            return

        buffer = np.empty_like(out, order="C")
        kernel(int(n), points, out=buffer)
        out[...] = buffer

    return adapter


_cpp_bernstein_core: Final[_BasisCoreFunc] = _cpp_tabulation_core("tabulate_bernstein_1d")
"""The Bernstein tabulation of the C++ backend."""

_cpp_legendre_core: Final[_BasisCoreFunc] = _cpp_tabulation_core("tabulate_legendre_1d")
"""The Legendre tabulation of the C++ backend."""


def bernstein_core(backend: Backend | None = None) -> CoreKernels:
    """Return the Bernstein tabulation kernels of the requested backend.

    **This is the tabulation :class:`CoreKernels` was shaped for.** That class's
    docstring named Bernstein as the one 1D basis carrying a serial twin and as
    the obvious next port, and so it is: the Python backend returns both kernels
    and :func:`pantr.basis._basis_1D._tabulate_basis_1D_impl_helper` picks between
    them per call on ``_PARALLEL_MIN_NUM_PTS``.

    The C++ backend returns no twin, and that is not an omission. Its kernel runs
    on the calling thread at every batch size, so there is no parallel launch to
    avoid below a threshold and nothing for a second entry to select.

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
        return CoreKernels(
            parallel=_tabulate_Bernstein_basis_1D_core,
            serial=_tabulate_Bernstein_basis_1D_serial_core,
        )

    if chosen not in available_backends():
        raise RuntimeError(f"the {chosen.name} backend is not available in this installation")
    return CoreKernels(parallel=_cpp_bernstein_core)


def legendre_core(backend: Backend | None = None) -> CoreKernels:
    """Return the Legendre tabulation kernels of the requested backend.

    Neither backend carries a serial twin, for the reasons
    :func:`cardinal_bspline_core` gives.

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
        return CoreKernels(parallel=_tabulate_Legendre_basis_1D_core)

    if chosen not in available_backends():
        raise RuntimeError(f"the {chosen.name} backend is not available in this installation")
    return CoreKernels(parallel=_cpp_legendre_core)


_BasisDerivCoreFunc = Callable[
    [np.int32, npt.NDArray[np.float32 | np.float64], int, npt.NDArray[np.float32 | np.float64]],
    None,
]
"""Signature of a 1D derivative tabulation kernel: ``(degree, pts, n_deriv, out) -> None``."""


class DerivKernels(NamedTuple):
    """The kernels tabulating one basis's derivatives, in one backend.

    :class:`CoreKernels`'s counterpart for a derivative tabulation. A separate record
    rather than a reuse, because the signature carries ``n_deriv`` and the output has
    rank 3; one record holding both would have to be read as one or the other at every
    call site.

    Attributes:
        parallel (_BasisDerivCoreFunc): The kernel used for batches of
            ``_PARALLEL_MIN_NUM_PTS`` points or more.
        serial (_BasisDerivCoreFunc | None): The serial twin, used below that
            threshold. ``None`` when this backend has none, in which case ``parallel``
            runs at every batch size. Defaults to None.
    """

    parallel: _BasisDerivCoreFunc
    serial: _BasisDerivCoreFunc | None = None


def _cpp_bernstein_deriv_core(
    n: np.int32,
    t: npt.NDArray[np.float32 | np.float64],
    n_deriv: int,
    out: npt.NDArray[np.float32 | np.float64],
) -> None:
    """Adapt the C++ Bernstein derivative kernel to the :data:`_BasisDerivCoreFunc` signature.

    The contract is :func:`_cpp_cardinal_bspline_core`'s, in full, including why a
    non-contiguous ``out`` is absorbed rather than refused.

    Args:
        n (np.int32): Degree of the basis. Assumed non-negative.
        t (npt.NDArray[np.float32 | np.float64]): 1D evaluation points on ``[0, 1]``.
        n_deriv (int): Highest derivative order. Assumed non-negative.
        out (npt.NDArray[np.float32 | np.float64]): Output of shape
            ``(t.size, n_deriv + 1, n + 1)`` and matching dtype. Need not be
            contiguous.

    Note:
        No input validation is performed. Shape and dtype are established by the
        Layer 2 caller; dtype, rank and contiguity are re-checked by nanobind's typed
        signature, which raises :class:`TypeError` before the kernel runs rather than
        silently converting.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    points = np.ascontiguousarray(t)
    if out.flags["C_CONTIGUOUS"]:
        _pantr_cpp.tabulate_bernstein_deriv_1d(int(n), int(n_deriv), points, out=out)
        return

    buffer = np.empty_like(out, order="C")
    _pantr_cpp.tabulate_bernstein_deriv_1d(int(n), int(n_deriv), points, out=buffer)
    out[...] = buffer


def bernstein_deriv_core(backend: Backend | None = None) -> DerivKernels:
    """Return the Bernstein derivative tabulation kernels of the requested backend.

    Reached from :mod:`pantr.bspline._bspline_basis_core`'s Bézier-like fast path,
    which is the only consumer today: a B-spline space whose knots describe a single
    Bézier segment has the Bernstein basis of the same degree, so its derivatives come
    from here and are then scaled by the chain rule *above* this seam, in the same
    numpy expression under both backends.

    The Python backend returns both kernels, and the dispatch between them on
    ``_PARALLEL_MIN_NUM_PTS`` stays where the oracle put it. The C++ backend returns no
    twin, for the reason :func:`cardinal_bspline_core` gives: its kernel runs on the
    calling thread at every batch size.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect, per :func:`pantr._backend.active_backend`.
            Defaults to None.

    Returns:
        DerivKernels: The kernels, each callable as ``(n, t, n_deriv, out) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    chosen = active_backend() if backend is None else backend

    if chosen is Backend.PYTHON:
        return DerivKernels(
            parallel=_tabulate_Bernstein_basis_deriv_1D_core,
            serial=_tabulate_Bernstein_basis_deriv_1D_serial_core,
        )

    if chosen not in available_backends():
        raise RuntimeError(f"the {chosen.name} backend is not available in this installation")
    return DerivKernels(parallel=_cpp_bernstein_deriv_core)
