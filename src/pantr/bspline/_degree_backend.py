"""Which implementation changes a B-spline field's degree: Numba, or C++.

:mod:`pantr._backend` owns the **policy**. This module owns the **catalogue** for the
two degree-changing operations, exactly as :mod:`pantr.bspline._refinement_backend`
and :mod:`pantr.bspline._structural_backend` do for theirs, and for the same reason: a
catalogue imports the implementations it hands out, so keeping each one beside its own
is what stops the policy module from importing the library it must stay independent of.

Two entry points rather than accessors
--------------------------------------

:func:`derivative_of_field` and :func:`elevate_field_degree` are functions that *do*
the work, not accessors that return a callable, for the reason
:mod:`pantr.bspline._refinement_backend` gives: these are not kernels selected by a
caller that then invokes them. What crosses the seam is a :class:`~pantr.bspline.Bspline`,
the branch is decided by more than the active backend, and there is exactly one
consumer per entry point.

What crosses the boundary
-------------------------

**The field, not its arrays**, for the reason
:mod:`pantr.bspline._refinement_backend` gives: these are operations on a domain type
C++ has owned since the 2026-08-27 amendment to ``design/cross_backend_types.md``, so
the adapter hands the binding the handle the wrapper already holds and wraps the
handle that comes back. The *arguments* cross as a scalar direction and a tuple of
increments, neither of which carries an invariant to lose.

Where the two backends differ, and there are four places
--------------------------------------------------------

**An argument with nothing to elevate runs the oracle**, which is the one divergence
here that is neither a boundary nor a defect but an asymmetry between the two sides'
own argument checking: ``cpp/include/pantr/bspline/degree.hpp`` refuses an all-zero or
negative increment with :meth:`pantr.bspline.Bspline.elevate_degree`'s own Layer 1
message, while ``_degree_elevate_bspline`` never checks and returns the field unchanged.
The public method refuses such an argument before either is reached, so this closes a gap
only a direct caller of :func:`elevate_field_degree` could open -- which is the same
reason :mod:`pantr.bspline._refinement_backend` closes its own.

**A rational field, or a ``keep_degree=True`` request, always runs the oracle.**
``cpp/include/pantr/bspline/degree.hpp`` gives ``derivative`` no ``keep_degree``
parameter at all and refuses a rational field outright, and says why: both of the
oracle's corresponding paths currently have an open defect --
``derivative(keep_degree=True)`` on an unclamped, non-periodic direction returns a
wrong function, and the rational derivative raises whenever the differentiated
direction is periodic -- so there is nothing stable for a C++ side to be at parity
with yet. :func:`_the_cpp_backend_can_differentiate` keeps both cases on the Python
path rather than letting a caller meet that refusal or, worse, a silently absent
parameter.

**A periodic direction to be elevated always runs the oracle.** Elevating one
round-trips through ``_to_periodic_bspline_1d_impl``, which
``cpp/include/pantr/bspline/structural.hpp`` declares as its own boundary and gives its
reasons for; ``cpp/include/pantr/bspline/degree.hpp`` refuses such a direction rather
than porting that conversion under cover of this one. It is the same declared boundary
:mod:`pantr.bspline._refinement_backend` draws around a periodic direction that receives
knots. A direction with a zero increment is not
affected: C++ carries its space handle into the result untouched.

**An unclamped direction to be elevated also always runs the oracle, and that one is
not a missing port -- it is a defect in the oracle that this port deliberately does
not reproduce.** A5.9 walks past the end of the coefficient array on an unclamped
knot vector, and the C++ core refuses the read rather than perform it, because in C++
an out-of-bounds read is undefined behaviour rather than a wrong number; the same
header explains the read is silently wrong in the oracle itself, under numba, and
raises ``IndexError`` only when JIT is disabled. **What the library accepts must not
change with ``PANTR_BACKEND``, so an unclamped direction is routed to the oracle here
rather than refused**, and whatever the oracle does with it is what a caller gets on
either backend, unchanged. On the shapes exercised so far that is a ``ValueError`` from
``Bspline``'s own constructor about the coefficient count, which is the out-of-bounds
walk surfacing one step later rather than a diagnosis of it;
``tests/parity/test_bspline_degree.py`` pins that behaviour on purpose, so that fixing
the oracle is a visible change rather than a silent one.

Cross-backend fields
--------------------

:func:`_cpp_handle` refuses a field built under the other backend rather than
converting it, exactly as :func:`pantr.bspline._refinement_backend._cpp_handle` does
and for the same reason: :meth:`pantr.bspline.Bspline._mutate` already refuses the
same conversion for the three ``in_place=`` methods, and
``design/cross_backend_types.md`` forbids it generally. The refusal is a property of
*taking the C++ route*, so it fires under the C++ backend and not under the Python
one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from .._backend import Backend, active_backend, available_backends
from ._bspline_degree import _degree_elevate_bspline
from ._bspline_derivative import _derivative_bspline

if TYPE_CHECKING:
    from .._pantr_cpp import Bspline32 as _CppBspline32
    from .._pantr_cpp import Bspline64 as _CppBspline64
    from ._bspline import Bspline

    _CppHandle: TypeAlias = "_CppBspline32 | _CppBspline64"
    """A field's C++ implementation, of either storage format."""


def _cpp_handle(bspline: Bspline, operation: str) -> _CppHandle:
    """The C++ implementation a field holds, refusing a Python one.

    Args:
        bspline (~pantr.bspline.Bspline): The field whose handle is wanted.
        operation (str): The verb naming the operation, for the error message.

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
            f"Bspline: cannot {operation} a Bspline built under a different backend "
            f"({type(impl).__name__} against the active C++ one); the backend is "
            f"chosen per process, so this means the active one changed after this "
            f"Bspline was built."
        )
    return impl


def _the_cpp_backend_can_differentiate(bspline: Bspline, *, keep_degree: bool) -> bool:
    """Report whether the C++ path covers a :func:`derivative_of_field` call.

    Args:
        bspline (~pantr.bspline.Bspline): The field to differentiate.
        keep_degree (bool): Whether the caller asked for the degree-preserving path.

    Returns:
        bool: True when :func:`_cpp_derivative` may run.

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
    return not keep_degree and not bspline.is_rational


def _the_cpp_backend_can_elevate(bspline: Bspline, degree_increments: tuple[int, ...]) -> bool:
    """Report whether the C++ path covers a :func:`elevate_field_degree` call.

    Args:
        bspline (~pantr.bspline.Bspline): The field to elevate.
        degree_increments (tuple[int, ...]): The increments requested, in axis order.

    Returns:
        bool: True when :func:`_cpp_elevate_degree` may run.

    Raises:
        RuntimeError: If the C++ backend is the active one and is not available; see
            :func:`_the_cpp_backend_can_differentiate` for why the check is here.
    """
    if active_backend() is Backend.PYTHON:
        return False
    if Backend.CPP not in available_backends():
        raise RuntimeError("the CPP backend is not available in this installation")
    # An argument with nothing to elevate goes to the oracle, which is the guard
    # `_refinement_backend._the_cpp_backend_can_take_it` makes for the same reason: the
    # two sides disagree about it. `Bspline.elevate_degree` refuses such an argument above
    # this branch, so no public caller reaches it -- but these entry points are importable
    # private symbols of a package whose private symbols a downstream consumer already
    # imports, and the C++ half refuses what the oracle silently returns unchanged. The
    # `all()` below would say yes to it vacuously.
    if not any(increment > 0 for increment in degree_increments):
        return False
    spaces = bspline.space.spaces
    return all(
        not spaces[direction].periodic and spaces[direction].has_open_knots()
        for direction, increment in enumerate(degree_increments)
        if increment > 0
    )


def _cpp_derivative(bspline: Bspline, direction: int) -> Bspline:
    """Differentiate through the C++ binding.

    Args:
        bspline (~pantr.bspline.Bspline): The field, holding a C++ handle.
        direction (int): The direction to differentiate.

    Returns:
        ~pantr.bspline.Bspline: The hodograph, wrapping a C++ handle.

    Raises:
        ValueError: If ``direction`` is out of range, if the degree in that direction
            is 0, or if the field is rational. Every message is the oracle's, except
            the rational one, which names a boundary
            :func:`_the_cpp_backend_can_differentiate` never lets a caller reach.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    from ._bspline import Bspline as BsplineCls  # noqa: PLC0415  (cycle)

    handle = _cpp_handle(bspline, "differentiate")
    return BsplineCls._wrap_over(
        _pantr_cpp.differentiate_bspline(handle, direction), bspline.space.spaces
    )


def _cpp_elevate_degree(bspline: Bspline, degree_increments: tuple[int, ...]) -> Bspline:
    """Elevate degree through the C++ binding.

    Args:
        bspline (~pantr.bspline.Bspline): The field, holding a C++ handle.
        degree_increments (tuple[int, ...]): Increments for each dimension.

    Returns:
        ~pantr.bspline.Bspline: The elevated field, wrapping a C++ handle.

    Raises:
        ValueError: If an elevated degree would exceed the exactness envelope of the
            binomial-coefficient kernel. The length, sign and all-zero refusals were
            made by :meth:`pantr.bspline.Bspline.elevate_degree` above the branch, so
            both backends refuse the same argument with the same message.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    from ._bspline import Bspline as BsplineCls  # noqa: PLC0415  (cycle)

    handle = _cpp_handle(bspline, "elevate the degree of")
    return BsplineCls._wrap_over(
        _pantr_cpp.elevate_bspline_degree(handle, list(degree_increments)),
        bspline.space.spaces,
    )


def derivative_of_field(bspline: Bspline, direction: int, *, keep_degree: bool) -> Bspline:
    """Differentiate a B-spline field, on whichever backend covers the call.

    Args:
        bspline (~pantr.bspline.Bspline): The field to differentiate.
        direction (int): Parametric direction for differentiation.
        keep_degree (bool): Whether the result should preserve the original degree by
            re-elevating after differentiation. Forces the Python path; see the module
            docstring.

    Returns:
        ~pantr.bspline.Bspline: The hodograph.

    Raises:
        TypeError: If the C++ backend is active and the field was built under the
            other one.
        ValueError: If ``direction`` is out of range, if the degree in that direction
            is 0, or if the degree the result has to be built at exceeds the
            exactness envelope of the binomial-coefficient kernel.
    """
    if _the_cpp_backend_can_differentiate(bspline, keep_degree=keep_degree):
        return _cpp_derivative(bspline, direction)
    return _derivative_bspline(bspline, direction, keep_degree=keep_degree)


def elevate_field_degree(bspline: Bspline, degree_increments: tuple[int, ...]) -> Bspline:
    """Elevate a B-spline field's degree, on whichever backend covers the call.

    Args:
        bspline (~pantr.bspline.Bspline): The field to elevate.
        degree_increments (tuple[int, ...]): Increments for each dimension, already
            validated by :meth:`pantr.bspline.Bspline.elevate_degree`.

    Returns:
        ~pantr.bspline.Bspline: The elevated field.

    Raises:
        TypeError: If the C++ backend is active and the field was built under the
            other one.
        ValueError: If an elevated degree would exceed the exactness envelope of the
            binomial-coefficient kernel.
    """
    if _the_cpp_backend_can_elevate(bspline, degree_increments):
        return _cpp_elevate_degree(bspline, degree_increments)
    return _degree_elevate_bspline(bspline, degree_increments)
