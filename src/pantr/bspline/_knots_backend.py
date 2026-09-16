"""Which implementation runs a knot computation over a space: Numba or the C++ port.

:mod:`pantr._backend` owns the **policy** -- which backends exist, which one is
selected, and the rule that an explicit request never falls back. This module owns the
**catalogue** for the knot computations of :mod:`pantr.bspline`, exactly as
:mod:`pantr.bspline._structural_backend` does for the three structural operations and
for the same reason: a catalogue imports the implementations it hands out, so keeping
each one beside its own is what stops the policy module from importing the library it
must stay independent of.

- :func:`cardinal_intervals`: which of a space's intervals are cardinal.

One entry point today, and it is the whole of the area. Every other function in
``cpp/include/pantr/bspline/knots.hpp`` is called from the space's own constructor and
has no Python caller to dispatch.

An entry point that *does* the work rather than an accessor returning a callable, for
:mod:`pantr.bspline._refinement_backend`'s reason: what crosses the seam is a
:class:`~pantr.bspline.BsplineSpace1D`, and there is exactly one consumer.

Why a catalogue and not the type
--------------------------------

Because the scan is an **operation** and not a property of the knots.
``cpp/include/pantr/bspline/space_1d.hpp`` says that type owns no operations and names
this scan as the reason the line is drawn where it is, so it reaches C++ as a free
function this module selects -- the shape every ported operation in
:mod:`pantr.bspline` already has. ``cpp/bindings/bspline_types.cpp`` still binds only
genuine members of the type.

What crosses the boundary
-------------------------

**The space, not its arrays**, which is what ``design/cross_backend_types.md``'s
2026-08-27 amendment requires of an operation on a domain type C++ owns: the adapter
hands the binding the handle the wrapper already holds. It matters here beyond the
rule, because the tolerance the scan compares knot spans against **is** the space's
own :attr:`~pantr.bspline.BsplineSpace1D.tolerance`, derived once at construction from
the knots as supplied and never recomputed. Handing over a knot vector alone would
invite the other side to re-derive it.

The result crosses back as an ``out=`` buffer this module allocates or validates, which
is where the rest of ``pantr``'s ``out=`` convention puts the allocation.

Where the two backends differ, and there is one place
-----------------------------------------------------

**Nothing here refuses any space either backend takes**, and the two answer the same
flags for every one of them. The C++ scan reproduces the oracle's index arithmetic
knot for knot, including the one place that arithmetic is off by one;
``cpp/include/pantr/bspline/knots.hpp`` carries which place and why it is reproduced
rather than corrected.

Which refusals each backend makes, and where
--------------------------------------------

``out`` is validated **above** the branch, by the oracle's own
:func:`pantr.basis._basis_utils._validate_out_array`, so both backends refuse the same
array with the same message and neither path can differ in order. The binding checks
the length again for a caller holding the handle with no wrapper in front of it, which
is ``cpp/bindings/bspline_knots.cpp``'s own reason for restating it; no Python caller
can reach that text.

Cross-backend spaces
--------------------

:func:`_cpp_handle` refuses a space built under the other backend rather than
converting it, which is what ``design/cross_backend_types.md`` forbids. The refusal is
a property of *taking the C++ route*, so it fires under the C++ backend and not under
the Python one, where the oracle runs happily over a C++ space's read-only knots. That
asymmetry is :func:`pantr.bspline._structural_backend._cpp_handle`'s, unchanged.

**It is also the one way this differs from the accessors beside it on the wrapper.**
:attr:`~pantr.bspline.BsplineSpace1D.num_intervals` and its siblings are answered by
whichever implementation the space holds, so they never notice the active backend;
this is an operation rather than a property, so the catalogue decides, and a space
built under the other backend is refused rather than served. Stated here because the
two sit next to each other on one class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from .._backend import Backend, active_backend, available_backends
from ..basis._basis_utils import _validate_out_array
from ._bspline_knots import _get_Bspline_cardinal_intervals_1D_impl

if TYPE_CHECKING:
    import numpy.typing as npt

    from .._pantr_cpp import BsplineSpace1D32 as _CppSpace32
    from .._pantr_cpp import BsplineSpace1D64 as _CppSpace64
    from ._bspline_space_1d import BsplineSpace1D

    _CppHandle: TypeAlias = "_CppSpace32 | _CppSpace64"
    """A space's C++ implementation, of either storage format."""

_Flags: TypeAlias = "npt.NDArray[np.bool_]"
"""One boolean per in-domain interval, which is what the scan returns."""


def _cpp_handle(space: BsplineSpace1D) -> _CppHandle:
    """The C++ implementation a space holds, refusing a Python one.

    Args:
        space (~pantr.bspline.BsplineSpace1D): The space whose handle is wanted.

    Returns:
        _CppHandle: The ``BsplineSpace1D32`` or ``BsplineSpace1D64`` handle.

    Raises:
        TypeError: If the space holds the Python implementation. That means the active
            backend changed after it was built, and converting one implementation into
            the other is what ``design/cross_backend_types.md`` forbids.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    impl = space._impl  # same package; the wrapper exposes no public handle
    if not isinstance(impl, _pantr_cpp.BsplineSpace1D32 | _pantr_cpp.BsplineSpace1D64):
        raise TypeError(
            f"BsplineSpace1D: cannot scan a space built under a different backend "
            f"({type(impl).__name__} against the active C++ one); the backend is "
            f"chosen per process, so this means the active one changed after this "
            f"BsplineSpace1D was built."
        )
    return impl


def _the_cpp_backend_can_take_it() -> bool:
    """Report whether the C++ path covers a call into this catalogue.

    One condition: the C++ backend has to be the active one. There is no periodicity
    question here, unlike refinement's -- the scan reads the knot vector as it stands
    and needs no conversion to reach a periodic space's intervals.

    Returns:
        bool: True when the ``_cpp_*`` functions below may run.

    Raises:
        RuntimeError: If the C++ backend is the active one and is not available. That
            cannot happen through :func:`pantr._backend.active_backend`, which refuses
            an unavailable backend at import; the check is here so that this module
            states the never-fall-back rule rather than relying on where it was
            enforced.
    """
    if active_backend() is Backend.PYTHON:
        return False
    if Backend.CPP not in available_backends():
        raise RuntimeError("the CPP backend is not available in this installation")
    return True


def _cpp_cardinal_intervals(space: BsplineSpace1D, out: _Flags) -> _Flags:
    """Scan a space's intervals through the C++ binding.

    Args:
        space (~pantr.bspline.BsplineSpace1D): The space, holding a C++ handle.
        out (npt.NDArray[np.bool_]): The destination, already validated by
            :func:`cardinal_intervals`. Need not be contiguous.

    Returns:
        npt.NDArray[np.bool_]: ``out``, filled.

    Raises:
        TypeError: If the space holds the Python implementation; see
            :func:`_cpp_handle`.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    handle = _cpp_handle(space)
    if out.flags["C_CONTIGUOUS"]:
        _pantr_cpp.bspline_space_cardinal_intervals_1d(handle, out=out)
    else:
        # The uncommon path. The binding's typed signature demands contiguous memory
        # and refuses anything else rather than converting it, because a converted
        # output would be filled and discarded; the numba kernel writes into any
        # writable boolean array, so the difference is absorbed here. The check above
        # is a flag read, so only a caller that actually passes a strided view pays.
        buffer: _Flags = np.empty(out.shape, dtype=np.bool_)
        _pantr_cpp.bspline_space_cardinal_intervals_1d(handle, out=buffer)
        out[...] = buffer
    return out


def cardinal_intervals(space: BsplineSpace1D, out: _Flags | None = None) -> _Flags:
    """Report which of a space's intervals are cardinal, on whichever backend is active.

    Args:
        space (~pantr.bspline.BsplineSpace1D): The space to scan.
        out (npt.NDArray[np.bool_] | None): Optional destination of shape
            ``(space.num_intervals,)`` and dtype ``bool``. Allocated here if None.
            Defaults to None.

    Returns:
        npt.NDArray[np.bool_]: Boolean array where True marks a cardinal interval, one
        entry per interval. ``out`` itself when it was provided.

    Raises:
        TypeError: If ``out`` is a masked array, or if the C++ backend is active and
            the space was built under the other one.
        ValueError: If ``out`` has the wrong shape or dtype, or is not writeable.
    """
    if not _the_cpp_backend_can_take_it():
        return _get_Bspline_cardinal_intervals_1D_impl(
            space.knots, space.degree, space.tolerance, out=out
        )

    # Allocated or validated on this side of the branch, so that the message a caller
    # sees is the oracle's under either backend; the module docstring says why.
    num_intervals = space.num_intervals
    if out is None:
        out = np.empty(num_intervals, dtype=np.bool_)
    else:
        _validate_out_array(out, (num_intervals,), np.bool_)
    return _cpp_cardinal_intervals(space, out)
