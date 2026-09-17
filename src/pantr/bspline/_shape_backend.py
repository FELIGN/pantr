"""Which implementation performs a shape operation on a field: Numba or C++.

:mod:`pantr._backend` owns the **policy**. This module owns the **catalogue** for the
three shape operations, exactly as :mod:`pantr.bspline._structural_backend` does for the
structural ones, and for the same reason: a catalogue imports the implementations it
hands out, so keeping each one beside its own is what stops the policy module from
importing the library it must stay independent of.

Three entry points rather than accessors
----------------------------------------

:func:`reverse_field`, :func:`permute_field_directions` and :func:`transform_field` are
functions that *do* the work, not accessors that return a callable, for the reason
:mod:`pantr.bspline._refinement_backend` gives in full: what crosses the seam is a
:class:`~pantr.bspline.Bspline`, and there is exactly one consumer per entry point.

The space is reseated by two of the three, and each needs a different wrapper
---------------------------------------------------------------------------

This is what separates this catalogue from the three before it, and it is the only
place where following their shape blindly would be wrong.

- :func:`reverse_field` reseats one direction and keeps the rest, so its result wraps
  through :meth:`pantr.bspline.Bspline._wrap_over` with the field's own direction list.
  A direction the operation left alone keeps the wrapper it already had.
- :func:`permute_field_directions` keeps every direction but **moves** them, so it
  wraps through the same method with the list **permuted**. ``_wrap_over`` reuses
  positionally against the *result's* directions, and handing it the unpermuted list
  would return a wrapper for a different direction while every value comparison agreed.
  That case is named in ``_wrap_over``'s own docstring and this is its first caller.
- :func:`transform_field` touches no space at all, so it wraps through
  :meth:`pantr.bspline.Bspline._wrap_over_the_same_space` and presents the field's own
  :class:`~pantr.bspline.BsplineSpace` object. ``_wrap_over`` would build a new one and
  break ``field.transform(t).space is field.space``, which ``tests/test_transform.py``
  asserts and which ``design/bspline_ownership_lifetime.md``'s **F6** names as the
  stronger of its two identity assertions.

What crosses the boundary
-------------------------

**The field, not its arrays**, which is what ``design/cross_backend_types.md``'s
2026-08-27 amendment requires of an operation on a domain type C++ owns.

The one thing that is not a plain scalar is the affine map, and it crosses as
``affine.matrix`` and ``affine.offset`` rather than as an
:class:`~pantr.transform.AffineTransform`. That is deliberate and it is what
``Bezier.transform`` already does: those two attributes are numpy arrays on either
backend, so no affine implementation is ever converted into the other, which is the
shape ``design/cross_backend_types.md`` forbids. They are ``float64`` whatever the
field stores; ``cpp/include/pantr/bspline/shape.hpp`` casts them to the storage format
**before** multiplying, which is the oracle's ``matrix.astype(dtype)``.

``in_place=True`` is not routed here, and that is the ticket's own scope
-----------------------------------------------------------------------

FELIGN/pantr#495 asks for the three methods to dispatch **in their value-returning
form**. The mutating form stays on :meth:`pantr.bspline.Bspline._mutate`, which is the
single writer and whose ``rebuild`` callable is defined in terms of *a control-point
array and a space* -- not a field, which is what a C++ shape function returns. Routing
it would mean changing that contract, which the ticket's invariants forbid.

The consequence worth stating: under the C++ backend ``f.transform(t)`` runs the C++
loop while ``f.transform(t, in_place=True)`` runs the oracle's ``cp @ A.T + b``, so the
two may differ in the last bits. They cannot differ for :meth:`reverse` or
:meth:`permute_directions`, whose control points are rearranged rather than computed,
and whose reflected knot vector is the same two IEEE operations on either side.
``tests/parity/test_bspline_shape.py`` pins the value each mutator leaves behind
against the value-returning form under both backends, which is where that would show.

Where the two backends differ, and there is nowhere
---------------------------------------------------

**Nothing here refuses anything the oracle takes.** A periodic direction is handled:
:func:`reverse_field` reflects its knot vector and rolls its net by the ghost count,
which is the oracle's ``np.roll``, and the other two never look at periodicity. A
rational field is handled by all three -- the weight column is carried through
unchanged by the two rearrangements and copied by the transform, which is
``w (A x + b) = A (w x) + w b``.

Which refusals each backend makes, and where
--------------------------------------------

Every Layer 1 check of :meth:`~pantr.bspline.Bspline.reverse`,
:meth:`~pantr.bspline.Bspline.permute_directions` and
:meth:`~pantr.bspline.Bspline.transform` runs in the wrapper, above the branch, so both
backends refuse the same argument with the same message and neither path can differ in
order. ``transform``'s rank check is the one that does not look like the others: it
lives in :func:`pantr._transform_control_points._apply_affine_to_control_points`, which
this module calls **before** branching for exactly that reason -- see
:func:`transform_field`. The C++ header restates all three for a caller with no wrapper
in front of it, and ``cpp/tests/test_bspline_shape.cpp`` pins them there.

Cross-backend fields
--------------------

:func:`_cpp_handle` refuses a field built under the other backend rather than converting
it, which is what ``design/cross_backend_types.md`` forbids. The refusal is a property
of *taking the C++ route*, so it fires under the C++ backend and not under the Python
one, where the oracle runs happily over a C++ field's read-only control points. That
asymmetry is :func:`pantr.bspline._structural_backend._cpp_handle`'s, unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from .._backend import Backend, active_backend, available_backends
from .._transform_control_points import _check_affine_rank, _geometric_rank

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .._pantr_cpp import Bspline32 as _CppBspline32
    from .._pantr_cpp import Bspline64 as _CppBspline64
    from ..transform import AffineTransform
    from ._bspline import Bspline

    _CppHandle: TypeAlias = "_CppBspline32 | _CppBspline64"
    """A field's C++ implementation, of either storage format."""


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
            f"Bspline: cannot reshape a Bspline built under a different backend "
            f"({type(impl).__name__} against the active C++ one); the backend is "
            f"chosen per process, so this means the active one changed after this "
            f"Bspline was built."
        )
    return impl


def _the_cpp_backend_can_take_it() -> bool:
    """Report whether the C++ path covers a shape call.

    One condition, as for the structural operations: the C++ backend has to be the
    active one. The module docstring says why there is no periodicity question and no
    rationality question here.

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


def _cpp_reverse(bspline: Bspline, direction: int) -> Bspline:
    """Reverse one direction of a field through the C++ binding.

    Wrapped over the field's own direction list: the C++ half rebuilds exactly one
    univariate space, so every other direction's wrapper is reused and
    ``reversed.space.spaces[d] is field.space.spaces[d]`` holds for them.

    Args:
        bspline (~pantr.bspline.Bspline): The field, holding a C++ handle.
        direction (int): The direction to reverse, already checked by the wrapper.

    Returns:
        ~pantr.bspline.Bspline: The reversed field, wrapping a C++ handle.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    from ._bspline import Bspline as BsplineCls  # noqa: PLC0415  (cycle)

    handle = _cpp_handle(bspline)
    return BsplineCls._wrap_over(
        _pantr_cpp.reverse_bspline(handle, direction), bspline.space.spaces
    )


def _cpp_permute(bspline: Bspline, permutation: Sequence[int]) -> Bspline:
    """Reorder a field's directions through the C++ binding.

    The prior list is **permuted the same way**, because
    :meth:`pantr.bspline.Bspline._wrap_over` reuses positionally against the result's
    directions rather than the caller's. Handing it the unpermuted list would give
    direction ``k`` of the result the wrapper of direction ``k`` of the source, which
    is a different direction, and no value comparison would report it.

    Args:
        bspline (~pantr.bspline.Bspline): The field, holding a C++ handle.
        permutation (Sequence[int]): A permutation of ``range(dim)``, already checked
            by the wrapper.

    Returns:
        ~pantr.bspline.Bspline: The permuted field, wrapping a C++ handle.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    from ._bspline import Bspline as BsplineCls  # noqa: PLC0415  (cycle)

    handle = _cpp_handle(bspline)
    prior = bspline.space.spaces
    return BsplineCls._wrap_over(
        _pantr_cpp.permute_bspline_directions(handle, list(permutation)),
        [prior[i] for i in permutation],
    )


def _cpp_transform(bspline: Bspline, affine: AffineTransform) -> Bspline:
    """Map a field's control points through the C++ binding.

    Wrapped over the field's **own space object**, not a rebuild of it: an affine map
    moves the control points and not the parametrization, so the C++ half passes the
    space handle straight through and this wrapper must present the wrapper that was
    already in front of it. See
    :meth:`pantr.bspline.Bspline._wrap_over_the_same_space` for why ``_wrap_over``
    cannot serve here.

    Args:
        bspline (~pantr.bspline.Bspline): The field, holding a C++ handle.
        affine (~pantr.transform.AffineTransform): The map to apply, whose rank the
            caller has already checked against the field's.

    Returns:
        ~pantr.bspline.Bspline: The transformed field, wrapping a C++ handle.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    from ._bspline import Bspline as BsplineCls  # noqa: PLC0415  (cycle)

    handle = _cpp_handle(bspline)
    return BsplineCls._wrap_over_the_same_space(
        _pantr_cpp.transform_bspline(handle, affine.matrix, affine.offset), bspline.space
    )


def reverse_field(bspline: Bspline, direction: int) -> Bspline:
    """Reverse one parametric direction, on whichever backend covers the call.

    Args:
        bspline (~pantr.bspline.Bspline): The field to reverse.
        direction (int): The direction to reverse, already checked by the caller.

    Returns:
        ~pantr.bspline.Bspline: The reversed field.
    """
    from ._bspline import Bspline as BsplineCls  # noqa: PLC0415  (cycle)

    if _the_cpp_backend_can_take_it():
        return _cpp_reverse(bspline, direction)
    new_cp, new_space = bspline._reversed(direction, bspline.control_points, write_into=False)
    return BsplineCls(new_space, new_cp, is_rational=bspline.is_rational)


def permute_field_directions(bspline: Bspline, permutation: Sequence[int]) -> Bspline:
    """Reorder the parametric directions, on whichever backend covers the call.

    Args:
        bspline (~pantr.bspline.Bspline): The field to permute.
        permutation (Sequence[int]): A permutation of ``range(dim)``, already checked by
            the caller.

    Returns:
        ~pantr.bspline.Bspline: The permuted field.
    """
    from ._bspline import Bspline as BsplineCls  # noqa: PLC0415  (cycle)

    if _the_cpp_backend_can_take_it():
        return _cpp_permute(bspline, permutation)
    new_cp, new_space = bspline._permuted(list(permutation), bspline.control_points)
    return BsplineCls(new_space, new_cp, is_rational=bspline.is_rational)


def transform_field(bspline: Bspline, affine: AffineTransform) -> Bspline:
    """Apply an affine map to the control points, on whichever backend covers the call.

    The rank check is the oracle's own, and it runs **before** the branch: it lives
    inside :func:`pantr._transform_control_points._apply_affine_to_control_points`
    rather than in :meth:`pantr.bspline.Bspline.transform`, so unlike the other two
    methods' checks it is not already behind the caller. Reaching it by running the
    oracle's helper on the Python path only would put the refusal behind the backend,
    which is precisely what every other entry point in these catalogues avoids -- so it
    is made here, and the C++ path is entered with the rank already agreed.

    It is the **same** check rather than a second spelling of it.
    :func:`pantr._transform_control_points._check_affine_rank` and
    :func:`~pantr._transform_control_points._geometric_rank` were split out of that
    helper so that the message and the rank expression live in one place; the oracle
    path then runs the check twice, which is harmless and is the price of it being one
    check rather than two that must be kept in step.

    Args:
        bspline (~pantr.bspline.Bspline): The field to transform.
        affine (~pantr.transform.AffineTransform): The map to apply.

    Returns:
        ~pantr.bspline.Bspline: The transformed field.

    Raises:
        ValueError: If the transform's dimension does not match the field's geometric
            rank, with the oracle's message.
    """
    from ._bspline import Bspline as BsplineCls  # noqa: PLC0415  (cycle)

    _check_affine_rank(
        affine.matrix,
        _geometric_rank(bspline.control_points.shape[-1], bspline.is_rational),
    )

    if _the_cpp_backend_can_take_it():
        return _cpp_transform(bspline, affine)
    new_cp, new_space = bspline._transformed(affine, bspline.control_points, write_into=False)
    return BsplineCls(new_space, new_cp, is_rational=bspline.is_rational)
