"""Which implementation performs a structural operation on a field: Numba or C++.

:mod:`pantr._backend` owns the **policy**. This module owns the **catalogue** for the
three structural operations, exactly as :mod:`pantr.bspline._refinement_backend` does
for the two refinement ones, and for the same reason: a catalogue imports the
implementations it hands out, so keeping each one beside its own is what stops the
policy module from importing the library it must stay independent of.

Three entry points rather than accessors
----------------------------------------

:func:`to_open_field`, :func:`split_field` and :func:`slice_field` are functions that
*do* the work, not accessors that return a callable, for the reason
:mod:`pantr.bspline._refinement_backend` gives in full: what crosses the seam is a
:class:`~pantr.bspline.Bspline`, and there is exactly one consumer per entry point.

:meth:`pantr.bspline.Bspline.boundary` has no entry point here because it needs none.
It is defined as :meth:`~pantr.bspline.Bspline.slice` at a domain endpoint, so it
reaches C++ through :func:`slice_field` and a fourth spelling would be a second
definition of one operation. ``pantr._pantr_cpp`` binds no ``bspline_boundary``
either, for the same reason.

What crosses the boundary
-------------------------

**The field, not its arrays**, which is what
``design/cross_backend_types.md``'s 2026-08-27 amendment requires of an operation on a
domain type C++ owns: the adapter hands the binding the handle the wrapper already
holds and wraps the handle that comes back. The *arguments* cross as plain scalars --
an axis, a direction, a parameter -- with no invariant to lose.

The one exception is a one-dimensional slice, whose result is a point rather than a
field. It crosses as an ``out=`` buffer this module allocates in the field's own
format, which is where the rest of ``pantr``'s ``out=`` convention puts the allocation,
and which is how ``slice_bezier_point`` already crosses.

Where the two backends differ, and there is one place
-----------------------------------------------------

**Nothing here refuses a periodic direction**, which is the one asymmetry with
refinement worth stating rather than leaving to be noticed.
:mod:`~pantr.bspline._refinement_backend` keeps a periodic direction on the oracle
because refining one needs the open-to-periodic conversion, which is not ported. None
of these three needs it: :func:`split_field` converts a periodic direction to its open
form and returns two non-periodic halves, :func:`slice_field` expands a periodic net by
its own modulo wrap, and :func:`to_open_field` *is* the conversion in the direction
that is ported. So the C++ path takes every field either backend takes.

**The one divergence that does exist is unreachable**, and it is recorded here so that
a later reader does not have to rediscover it.
:meth:`pantr.bspline.Bspline.slice`'s out-of-domain message interpolates
``space.domain[0]``, a *numpy scalar* of the storage format, and the C++ half renders
the widened value instead -- ``0.1`` against ``0.10000000149011612`` at ``float32``.
The wrapper makes that check before dispatching, so no caller can reach the C++ text;
``cpp/include/pantr/bspline/structural.hpp`` carries why the fix belongs to
``pantr/core/format.hpp`` and not here. ``tests/parity/test_bspline_structural.py``
pins the wrapper's own text under both backends.

Which refusals each backend makes, and where
--------------------------------------------

Every Layer 1 check of :meth:`~pantr.bspline.Bspline.split`,
:meth:`~pantr.bspline.Bspline.slice` and :meth:`~pantr.bspline.Bspline.boundary` runs
in the wrapper, above the branch, so both backends refuse the same argument with the
same message and neither path can differ in order. The C++ header restates those
checks for a caller with no wrapper in front of it, which is
``cpp/include/pantr/bspline/refinement.hpp``'s own reason for restating three of
``insert_knots``'.

:meth:`~pantr.bspline.Bspline.to_open_bspline` is the exception: it makes **no** Layer 1
check, and "already open in every direction" is raised by whichever implementation
runs. Both texts are the oracle's and both predicates are the oracle's, negated the
same way.

Cross-backend fields
--------------------

:func:`_cpp_handle` refuses a field built under the other backend rather than
converting it, which is what ``design/cross_backend_types.md`` forbids. The refusal is
a property of *taking the C++ route*, so it fires under the C++ backend and not under
the Python one, where the oracle runs happily over a C++ field's read-only control
points. That asymmetry is
:func:`pantr.bspline._refinement_backend._cpp_handle`'s, unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, cast

import numpy as np

from .._backend import Backend, active_backend, available_backends
from ._bspline_knot_insertion import _to_open_bspline_impl
from ._bspline_slice import _slice_bspline
from ._bspline_split import _split_bspline_impl

if TYPE_CHECKING:
    import numpy.typing as npt

    from .._pantr_cpp import Bspline32 as _CppBspline32
    from .._pantr_cpp import Bspline64 as _CppBspline64
    from ._bspline import Bspline

    _CppHandle: TypeAlias = "_CppBspline32 | _CppBspline64"
    """A field's C++ implementation, of either storage format."""

_Point: TypeAlias = "npt.NDArray[np.float32 | np.float64]"
"""The point a one-dimensional slice returns, in the field's storage format."""


def _cpp_handle(bspline: Bspline) -> _CppHandle:
    """The C++ implementation a field holds, refusing a Python one.

    Args:
        bspline (~pantr.bspline.Bspline): The field whose handle is wanted.

    Returns:
        _CppHandle: The ``Bspline32`` or ``Bspline64`` handle.

    Raises:
        TypeError: If the field holds the Python implementation. That means the active
            backend changed after it was built, and converting one implementation into
            the other is what ``design/cross_backend_types.md`` forbids.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    impl = bspline._impl  # same package; the wrapper exposes no public handle
    if not isinstance(impl, _pantr_cpp.Bspline32 | _pantr_cpp.Bspline64):
        raise TypeError(
            f"Bspline: cannot restructure a Bspline built under a different backend "
            f"({type(impl).__name__} against the active C++ one); the backend is "
            f"chosen per process, so this means the active one changed after this "
            f"Bspline was built."
        )
    return impl


def _the_cpp_backend_can_take_it() -> bool:
    """Report whether the C++ path covers a structural call.

    One condition, unlike refinement's three: the C++ backend has to be the active one.
    There is no periodicity question here -- the module docstring says why -- and no
    "at least one direction" question either, since none of the three takes a
    per-direction argument that could be empty.

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


def _cpp_to_open(bspline: Bspline) -> Bspline:
    """Convert every direction to open form through the C++ binding.

    Args:
        bspline (~pantr.bspline.Bspline): The field, holding a C++ handle.

    Returns:
        ~pantr.bspline.Bspline: The open field, wrapping a C++ handle.

    Raises:
        ValueError: If every direction is already open and non-periodic, with the
            oracle's message.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    from ._bspline import Bspline as BsplineCls  # noqa: PLC0415  (cycle)

    handle = _cpp_handle(bspline)
    return BsplineCls._wrap_over(_pantr_cpp.open_bspline(handle), bspline.space.spaces)


def _cpp_split(bspline: Bspline, direction: int, value: float) -> tuple[Bspline, Bspline]:
    """Split a field through the C++ binding.

    Both halves are wrapped over the *same* prior direction list, because both keep
    every direction but the split one: an untouched direction's wrapper is therefore
    shared by the two halves and by ``bspline``.

    Args:
        bspline (~pantr.bspline.Bspline): The field, holding a C++ handle.
        direction (int): The direction to split, already checked by the wrapper.
        value (float): The parameter to split at, already checked by the wrapper.

    Returns:
        tuple[~pantr.bspline.Bspline, ~pantr.bspline.Bspline]: The left and right
        halves, each wrapping a C++ handle.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    from ._bspline import Bspline as BsplineCls  # noqa: PLC0415  (cycle)

    handle = _cpp_handle(bspline)
    prior = bspline.space.spaces
    left, right = _pantr_cpp.split_bspline(handle, direction, float(value))
    return BsplineCls._wrap_over(left, prior), BsplineCls._wrap_over(right, prior)


def _cpp_slice(bspline: Bspline, axis: int, value: float) -> Bspline:
    """Slice a field of dimension at least two through the C++ binding.

    Args:
        bspline (~pantr.bspline.Bspline): The field, holding a C++ handle, of dimension
            at least two.
        axis (int): The direction to fix, already checked by the wrapper.
        value (float): The parameter, already checked by the wrapper.

    Returns:
        ~pantr.bspline.Bspline: The sliced field, of one dimension fewer, wrapping a
        C++ handle.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    from ._bspline import Bspline as BsplineCls  # noqa: PLC0415  (cycle)

    handle = _cpp_handle(bspline)
    # The sliced axis is dropped from the prior list too: the reuse is positional
    # against the *result's* directions. See :meth:`Bspline._wrap_over`.
    prior = [space for index, space in enumerate(bspline.space.spaces) if index != axis]
    return BsplineCls._wrap_over(_pantr_cpp.slice_bspline(handle, axis, float(value)), prior)


def _cpp_slice_point(bspline: Bspline, value: float) -> _Point:
    """Evaluate a one-dimensional field at a parameter through the C++ binding.

    The binding hands back the homogeneous components with the weight column still on,
    so the projection of a rational field happens here -- as one numpy division, which
    is the oracle's own expression rather than a second spelling of it in C++.

    Args:
        bspline (~pantr.bspline.Bspline): A one-dimensional field, holding a C++ handle.
        value (float): The parameter, already checked by the wrapper.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The point, of length ``bspline.rank``.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    handle = _cpp_handle(bspline)
    components = bspline.control_points.shape[-1]
    point: _Point = np.empty(components, dtype=bspline.dtype)
    _pantr_cpp.slice_bspline_point(handle, float(value), out=point)
    if bspline.is_rational:
        # `cast` rather than `np.asarray`: the division of a `T` array by a `T` scalar
        # is already a `T` array, and the oracle's own
        # :func:`~pantr.bspline._bspline_slice._slice_bspline` casts the identical
        # expression for the identical reason -- the numpy stubs type `/` as `Any`.
        return cast("_Point", point[:-1] / point[-1])
    return point


def to_open_field(bspline: Bspline) -> Bspline:
    """Convert a field to open, non-periodic form, on whichever backend covers the call.

    Args:
        bspline (~pantr.bspline.Bspline): The field to convert.

    Returns:
        ~pantr.bspline.Bspline: An open, non-periodic field over the same geometry.

    Raises:
        TypeError: If the C++ backend is active and the field was built under the other
            one.
        ValueError: If every direction is already open and non-periodic. Both backends
            raise it, with the same text; see the module docstring on why this is the
            one refusal the wrapper does not make first.
    """
    if _the_cpp_backend_can_take_it():
        return _cpp_to_open(bspline)
    return _to_open_bspline_impl(bspline)


def split_field(bspline: Bspline, direction: int, value: float) -> tuple[Bspline, Bspline]:
    """Split a field in two, on whichever backend covers the call.

    The direction and the parameter have already been validated by
    :meth:`pantr.bspline.Bspline.split`, so neither branch re-derives which arguments
    are legal.

    Args:
        bspline (~pantr.bspline.Bspline): The field to split.
        direction (int): The direction to split, in ``[0, dim)``.
        value (float): The parameter to split at, strictly inside that direction's
            domain.

    Returns:
        tuple[~pantr.bspline.Bspline, ~pantr.bspline.Bspline]: The left and right
        halves, non-periodic in the split direction whatever ``bspline`` was.

    Raises:
        TypeError: If the C++ backend is active and the field was built under the other
            one.
    """
    if _the_cpp_backend_can_take_it():
        return _cpp_split(bspline, direction, value)
    return _split_bspline_impl(bspline, direction, value)


def slice_field(bspline: Bspline, axis: int, value: float) -> Bspline | _Point:
    """Fix one direction of a field at a value, on whichever backend covers the call.

    The axis and the parameter have already been validated by
    :meth:`pantr.bspline.Bspline.slice`, so neither branch re-derives which arguments
    are legal.

    Args:
        bspline (~pantr.bspline.Bspline): The field to slice.
        axis (int): The direction to fix, in ``[0, dim)``.
        value (float): The parameter, inside that direction's domain.

    Returns:
        ~pantr.bspline.Bspline | npt.NDArray[np.float32 | np.float64]: A field of one
        dimension fewer when ``bspline.dim >= 2``, or the point itself when it is 1 --
        projected to physical coordinates for a rational field, as the oracle projects
        it.

    Raises:
        TypeError: If the C++ backend is active and the field was built under the other
            one.
    """
    if _the_cpp_backend_can_take_it():
        # Two bindings rather than one, because C++ cannot return a field or an array;
        # `cpp/include/pantr/bspline/structural.hpp` argues the split in full and the
        # branch is the oracle's own `bspline.dim == 1`.
        if bspline.dim == 1:
            return _cpp_slice_point(bspline, value)
        return _cpp_slice(bspline, axis, value)
    return _slice_bspline(bspline, axis, value)
