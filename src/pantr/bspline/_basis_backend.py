"""Which implementation of the 1D B-spline basis tabulation runs: Numba or the C++ port.

:mod:`pantr._backend` owns the **policy** -- which backends exist, which one is
selected, and the rule that an explicit request never falls back. This module owns
the **catalogue** for the general-knot basis tabulation of :mod:`pantr.bspline`, and
it is where the C++ adapters live. :mod:`pantr.bspline._extraction_backend` and
:mod:`pantr.bspline._refinement_backend` are its siblings; the split by area is what
keeps :mod:`pantr._backend` free of any import from the library.

- :class:`BasisKernels`: the value-tabulation kernels of one backend.
- :class:`BasisDerivKernels`: the derivative-tabulation kernels of one backend.
- :func:`bspline_basis_core`: the value kernels of a backend.
- :func:`bspline_basis_deriv_core`: the derivative kernels of a backend.

Two things about the seam are worth stating here rather than leaving to be
rediscovered.

**Only the general-knot path is dispatched here.** A space with Bézier-like knots
takes the Bernstein fast path instead, and that path was already backend-dispatched
before this module existed: the values go through
:func:`pantr.basis._basis_1D._tabulate_Bernstein_basis_1D_impl`, which asks
:func:`pantr.basis._basis_backend.bernstein_core`, and the derivatives now go through
:func:`pantr.basis._basis_backend.bernstein_deriv_core`. The change of variable onto
``[0, 1]`` and the chain-rule scaling stay in
:mod:`pantr.bspline._bspline_basis_core`, above the seam, so they are the *same
numpy expressions* under both backends and cannot contribute a difference. That is
``design/backend_parity.md`` Rule 1's consequence applied here: keep a shared map on
the common side and it cancels exactly.

**One call shape falls back to Numba**, and it is the only one:
:func:`_the_cpp_kernels_cannot_serve` names it and says at length why falling back is
the least-bad of the three available answers. Everything else reaches C++.

**The C++ side has one kernel at every batch size, and that is not an omission.**
The oracle keeps a ``parallel=True`` kernel and a serial twin and picks between them
on ``_PARALLEL_MIN_NUM_PTS``; the C++ kernel runs on the calling thread, so there is
no parallel launch to avoid below a threshold and nothing for a second entry to
select. Rule 7 -- a liveness threshold is a property of the host it was measured on
-- so that constant is **inherited, not re-derived**, and nothing here measures a new
one. ``scripts/measure_bspline_tabulation_widths.py`` checks the twins are bitwise
identical to each other (0 of 369 824 values differ, at both widths), which is what
makes it sound to compare one C++ kernel against whichever of the two the oracle's
dispatch happened to choose.

**The two Numba kernel imports are deferred, and the reason is structural.** Every
other catalogue in this package imports its kernels at module scope, which
:mod:`pantr.bspline._extraction_backend` can do because
:mod:`pantr.bspline._bspline_extraction_core` holds the kernels and nothing else --
its module docstring says it exists for exactly that. The general-knot basis kernels
are **not** split that way: :mod:`pantr.bspline._bspline_basis_core` holds both them
and the Layer 2 entry points that have to ask this module which to call, so a
module-scope import here closes a cycle.

Deferring it inside the two catalogue functions is the smaller of the two repairs.
The larger one is to split that module the way the extraction port is split, and it
is the better long-term shape -- but it moves seven hundred lines of private symbols,
and ``CLAUDE.md`` records a downstream consumer whose largest exposure is
``pantr.bspline``. Recorded here as a follow-up rather than done in passing. Note what
:mod:`pantr.basis._basis_backend`'s docstring objects to about lazy imports: that the
count *grows by one per kernel ported*. Here it is two and cannot grow, since every
future kernel is reached through one of the same two functions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from .._backend import Backend, active_backend, available_backends

_FloatArray = npt.NDArray[np.float32 | np.float64]

_BasisFunc = Callable[
    [_FloatArray, int, bool, float, _FloatArray, _FloatArray, npt.NDArray[np.int_]],
    None,
]
"""Signature of a general-knot basis kernel.

``(knots, degree, periodic, tol, pts, out_basis, out_first_basis) -> None``.
"""

_BasisDerivFunc = Callable[
    [_FloatArray, int, bool, float, int, _FloatArray, _FloatArray, npt.NDArray[np.int_]],
    None,
]
"""Signature of a general-knot basis derivative kernel.

``(knots, degree, periodic, tol, n_deriv, pts, out_deriv, out_first_basis) -> None``.
"""


class BasisKernels(NamedTuple):
    """The kernels tabulating the general-knot basis values, in one backend.

    Attributes:
        parallel (_BasisFunc): The kernel used for batches of
            ``_PARALLEL_MIN_NUM_PTS`` points or more.
        serial (_BasisFunc | None): The serial twin, used below that threshold.
            ``None`` when this backend has none, in which case ``parallel`` runs at
            every batch size. Defaults to None.
    """

    parallel: _BasisFunc
    serial: _BasisFunc | None = None


class BasisDerivKernels(NamedTuple):
    """The kernels tabulating the general-knot basis derivatives, in one backend.

    A second record rather than a reuse of :class:`BasisKernels`, because the
    signatures differ by the ``n_deriv`` argument and a single record holding both
    would have to be read as one or the other at every call site.

    Attributes:
        parallel (_BasisDerivFunc): The kernel used for batches of
            ``_PARALLEL_MIN_NUM_PTS`` points or more.
        serial (_BasisDerivFunc | None): The serial twin, used below that threshold.
            ``None`` when this backend has none. Defaults to None.
    """

    parallel: _BasisDerivFunc
    serial: _BasisDerivFunc | None = None


def _contiguous_int64(out: npt.NDArray[np.int_]) -> npt.NDArray[np.int64] | None:
    """Return a contiguous int64 buffer for ``out``, or ``None`` if ``out`` will serve.

    The binding's typed signature demands a C-contiguous ``int64`` array and refuses
    anything else rather than converting it, because a converted output would be
    filled and discarded. The numba kernels accept any writable integer array, so the
    difference has to be absorbed on this side.

    Args:
        out (npt.NDArray[np.int_]): The caller's first-basis-index array.

    Returns:
        npt.NDArray[np.int64] | None: A fresh buffer to compute into and copy back
        from, or ``None`` when ``out`` can be passed straight through.
    """
    if out.flags["C_CONTIGUOUS"] and out.dtype == np.int64:
        return None
    return np.empty(out.shape, dtype=np.int64)


def _the_cpp_kernels_cannot_serve(knots: _FloatArray, pts: _FloatArray) -> bool:
    """Whether this call's dtypes are outside what the C++ kernels can express.

    **The library deliberately supports a space whose knots are ``float32`` evaluated
    at ``float64`` points**, and returns the points' dtype;
    ``tests/test_mpi_thb_qi.py::test_result_is_float64_for_float32_space`` and
    ``tests/test_thb_spline_space.py::TestCellMembershipTolerance
    ::test_a_float32_root_space_is_still_graded_in_float64`` assert it by name. The
    Numba kernels serve it by opening with ``dtype = knots.dtype`` for their scratch
    while reading points at the points' own width, so the computation runs in a mixed
    width that narrows at every array store.

    The C++ kernels are templated on one scalar type and have no mixed overload, so
    they cannot express that call at all. Three ways to reconcile that, and only one
    of them keeps both halves of the contract:

    - **Refuse it.** Tried, and wrong: it makes the selected backend change what the
      library accepts, and it fails the five tests above.
    - **Promote both sides and compute in ``float64``.** Keeps the output dtype but
      changes the *values*, since the oracle truncates every intermediate to the knots'
      width. A silent value divergence between backends is worse than a visible
      fallback.
    - **Run the Numba kernel for this call.** What this does. It is exactly the
      behaviour the mixed case had before the C++ path existed, so nothing regresses,
      and the values stay identical to the oracle's because they *are* the oracle's.

    The cost is real and is why this is a narrow, named predicate rather than a
    ``try``/``except``: an A/B measurement over mixed-dtype input measures Numba on
    both sides, which is what ``pantr._backend``'s never-fall-back rule exists to
    prevent. It is confined to a call the C++ side cannot express at all, and
    ``tests/parity/test_bspline_basis_tabulation.py`` asserts both that the two
    backends agree there and that the C++ binding is genuinely not reached, so the
    fallback cannot widen unnoticed.

    Which of the three is right in the long run is a contract decision rather than a
    port decision: it turns on whether a mixed-width answer is wanted at all. This
    takes the option that decides nothing.

    Args:
        knots (_FloatArray): The space's knot vector.
        pts (_FloatArray): The evaluation points.

    Returns:
        bool: True when the dtypes differ, so the C++ kernels cannot be used.
    """
    return knots.dtype != pts.dtype


def _cpp_basis(  # noqa: PLR0913
    knots: _FloatArray,
    degree: int,
    periodic: bool,
    tol: float,
    pts: _FloatArray,
    out_basis: _FloatArray,
    out_first_basis: npt.NDArray[np.int_],
) -> None:
    """Adapt the C++ kernel to the :data:`_BasisFunc` signature.

    The adapter's whole job is that **the public API must behave identically on both
    backends**, because otherwise an A/B measurement compares two contracts rather
    than two kernels. Two differences are absorbed here:

    - the numba kernels accept a non-contiguous ``out`` and fill it, while the binding
      requires C-contiguous memory and refuses anything else;
    - ``tol`` is accepted and **not forwarded on the C++ path**, because the C++ kernel
      has no such parameter (the fallback path below does forward it, to the kernel that
      declares it). That is not a dropped argument: the oracle threads ``tol`` through to
      :func:`pantr.bspline._bspline_knots._get_Bspline_num_basis_1D_impl` purely for
      interface consistency, and its own docstrings record that it goes unused --
      the non-periodic basis count is ``len(knots) - degree - 1`` and involves no
      tolerance, while the periodic path short-circuits the count entirely.

    Args:
        knots (_FloatArray): The knot vector, non-decreasing.
        degree (int): The polynomial degree. Assumed non-negative.
        periodic (bool): Whether the space is periodic.
        tol (float): Accepted for signature compatibility and unused; see above.
        pts (_FloatArray): 1D evaluation points.
        out_basis (_FloatArray): Output of shape ``(pts.size, degree + 1)`` and
            matching dtype. Need not be contiguous.
        out_first_basis (npt.NDArray[np.int_]): Output of shape ``(pts.size,)``.
            Need not be contiguous.

    Note:
        No input validation is performed. Shape and dtype are established by the
        Layer 2 caller in :mod:`pantr.bspline._bspline_basis_core`, whose checks run
        before either backend and so cannot diverge between them.
    """
    if _the_cpp_kernels_cannot_serve(knots, pts):
        from ._bspline_basis_core import (  # noqa: PLC0415  (see the module docstring)
            _compute_basis_nurbs_book_impl,
        )

        _compute_basis_nurbs_book_impl(
            knots, degree, periodic, tol, pts, out_basis, out_first_basis
        )
        return

    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    del tol
    knot_view = np.ascontiguousarray(knots)
    point_view = np.ascontiguousarray(pts)
    index_buffer = _contiguous_int64(out_first_basis)
    indices = out_first_basis if index_buffer is None else index_buffer

    if out_basis.flags["C_CONTIGUOUS"]:
        _pantr_cpp.tabulate_bspline_basis_1d(
            knot_view,
            int(degree),
            bool(periodic),
            point_view,
            out_basis=out_basis,
            out_first_basis=indices,
        )
    else:
        # The uncommon path. The check above is a flag read, so the common case pays
        # essentially nothing for it; only a caller that actually passes a strided
        # view pays for the buffer and the copy back.
        buffer = np.empty_like(out_basis, order="C")
        _pantr_cpp.tabulate_bspline_basis_1d(
            knot_view,
            int(degree),
            bool(periodic),
            point_view,
            out_basis=buffer,
            out_first_basis=indices,
        )
        out_basis[...] = buffer

    if index_buffer is not None:
        out_first_basis[...] = index_buffer


def _cpp_basis_derivatives(  # noqa: PLR0913
    knots: _FloatArray,
    degree: int,
    periodic: bool,
    tol: float,
    n_deriv: int,
    pts: _FloatArray,
    out_deriv: _FloatArray,
    out_first_basis: npt.NDArray[np.int_],
) -> None:
    """Adapt the C++ derivative kernel to the :data:`_BasisDerivFunc` signature.

    The contract is :func:`_cpp_basis`'s, in full, including why a non-contiguous
    output is absorbed rather than refused and why ``tol`` is not forwarded.

    Args:
        knots (_FloatArray): The knot vector, non-decreasing.
        degree (int): The polynomial degree. Assumed non-negative.
        periodic (bool): Whether the space is periodic.
        tol (float): Accepted for signature compatibility and unused.
        n_deriv (int): Highest derivative order. Assumed non-negative.
        pts (_FloatArray): 1D evaluation points.
        out_deriv (_FloatArray): Output of shape ``(pts.size, n_deriv + 1,
            degree + 1)`` and matching dtype. Need not be contiguous.
        out_first_basis (npt.NDArray[np.int_]): Output of shape ``(pts.size,)``.

    Note:
        No input validation is performed; see :func:`_cpp_basis`.
    """
    if _the_cpp_kernels_cannot_serve(knots, pts):
        from ._bspline_basis_core import (  # noqa: PLC0415  (see the module docstring)
            _compute_basis_deriv_nurbs_book_impl,
        )

        _compute_basis_deriv_nurbs_book_impl(
            knots, degree, periodic, tol, n_deriv, pts, out_deriv, out_first_basis
        )
        return

    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    del tol
    knot_view = np.ascontiguousarray(knots)
    point_view = np.ascontiguousarray(pts)
    index_buffer = _contiguous_int64(out_first_basis)
    indices = out_first_basis if index_buffer is None else index_buffer

    target = out_deriv if out_deriv.flags["C_CONTIGUOUS"] else np.empty_like(out_deriv, order="C")
    _pantr_cpp.tabulate_bspline_basis_derivatives_1d(
        knot_view,
        int(degree),
        bool(periodic),
        int(n_deriv),
        point_view,
        out_deriv=target,
        out_first_basis=indices,
    )
    if target is not out_deriv:
        out_deriv[...] = target
    if index_buffer is not None:
        out_first_basis[...] = index_buffer


def bspline_basis_core(backend: Backend | None = None) -> BasisKernels:
    """Return the general-knot basis tabulation kernels of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect, per :func:`pantr._backend.active_backend`.
            Defaults to None.

    Returns:
        BasisKernels: The kernels, each callable as
        ``(knots, degree, periodic, tol, pts, out_basis, out_first_basis) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    chosen = active_backend() if backend is None else backend

    if chosen is Backend.PYTHON:
        from ._bspline_basis_core import (  # noqa: PLC0415  (breaks a cycle; see the module docstring)
            _compute_basis_nurbs_book_impl,
            _compute_basis_nurbs_book_serial_impl,
        )

        return BasisKernels(
            parallel=_compute_basis_nurbs_book_impl,
            serial=_compute_basis_nurbs_book_serial_impl,
        )

    if chosen not in available_backends():
        raise RuntimeError(f"the {chosen.name} backend is not available in this installation")
    return BasisKernels(parallel=_cpp_basis)


def bspline_basis_deriv_core(backend: Backend | None = None) -> BasisDerivKernels:
    """Return the general-knot basis derivative kernels of the requested backend.

    Args:
        backend (Backend | None): The backend to use. ``None`` means the backend
            currently in effect, per :func:`pantr._backend.active_backend`.
            Defaults to None.

    Returns:
        BasisDerivKernels: The kernels, each callable as ``(knots, degree, periodic,
        tol, n_deriv, pts, out_deriv, out_first_basis) -> None``.

    Raises:
        RuntimeError: If ``backend`` is given and is not available.
    """
    chosen = active_backend() if backend is None else backend

    if chosen is Backend.PYTHON:
        from ._bspline_basis_core import (  # noqa: PLC0415  (breaks a cycle; see the module docstring)
            _compute_basis_deriv_nurbs_book_impl,
            _compute_basis_deriv_nurbs_book_serial_impl,
        )

        return BasisDerivKernels(
            parallel=_compute_basis_deriv_nurbs_book_impl,
            serial=_compute_basis_deriv_nurbs_book_serial_impl,
        )

    if chosen not in available_backends():
        raise RuntimeError(f"the {chosen.name} backend is not available in this installation")
    return BasisDerivKernels(parallel=_cpp_basis_derivatives)
