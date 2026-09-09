"""Which implementation refines a B-spline field: the Numba one or C++.

:mod:`pantr._backend` owns the **policy**. This module owns the **catalogue** for the
two field-refinement operations, exactly as :mod:`pantr.bspline._extraction_backend`
and :mod:`pantr.bezier._bezier_backend` do for theirs, and for the same reason: a
catalogue imports the implementations it hands out, so keeping each one beside its own
is what stops the policy module from importing the library it must stay independent of.

Two entry points rather than accessors
--------------------------------------

:func:`insert_knots_into_field` and :func:`subdivide_field` are functions that *do* the
work, not accessors that return a callable. The rule
``design/cross_backend_types.md`` states -- a record when the consumer needs more than
one implementation at once, a bare callable when it does not -- is about kernels
selected by a caller that then invokes them. These are not kernels: what crosses the
seam is a :class:`~pantr.bspline.Bspline`, the branch is decided by more than the
active backend, and there is exactly one consumer per entry point. A selector returning
one of two callables would have to hand its caller the periodicity question as well.

What crosses the boundary
-------------------------

**The field, not its arrays.** These are operations on a domain type that C++ has owned
since the 2026-08-27 amendment to ``design/cross_backend_types.md``, so the adapter
hands the binding the handle the wrapper already holds and wraps the handle that comes
back. Unpacking a field into a knot vector and a control net on one side and
reassembling it on the other is the shape that amendment forbids, and the rationality
flag is exactly the invariant a reassembly drops. :mod:`pantr.bezier._bezier_backend`
says the same of its own n-dimensional entry points.

The *arguments* cross as arrays and scalars: one 1D knot array per direction, and the
per-direction subdivision counts. ``None`` does not cross in either. An unrefined
direction is an **empty array** for the insertion and a count of **1** for the
subdivision, which are the two spellings the oracle already treats as identical to
``None`` -- every branch of :meth:`pantr.bspline.Bspline.subdivide` reads ``None`` and
``1`` the same way, and :func:`~pantr.bspline._bspline_knot_insertion._insert_knots_bspline`
reads ``None`` and an empty array the same way.

Where the two backends differ, and there are two places
-------------------------------------------------------

**A periodic direction that receives knots runs the oracle.** The oracle refines one by
a round trip through the open representation --
``_to_open_bspline_1d_impl``, insert, ``_to_periodic_bspline_1d_impl`` -- and those two
are the *boundary and periodic conversions*, which
``cpp/include/pantr/bspline/bspline.hpp`` lists as their own port and which no ticket in
this milestone covers. ``cpp/include/pantr/bspline/refinement.hpp`` therefore refuses a
periodic direction outright, and :func:`_the_cpp_backend_can_take_it` keeps the Python
path for such a field rather than letting a caller meet that refusal. It is the same
declared boundary ``cpp/include/pantr/bspline/space_1d.hpp`` draws around
``get_cardinal_intervals``, and the same shape as the cardinal extraction builder having
no C++ half in :mod:`pantr.bspline._extraction_backend`.

A periodic direction that receives **no** knots is not affected: C++ carries its space
handle into the result untouched, exactly as the oracle carries its wrapper.

**The order of two refusals differs, and only their order.** The oracle checks each
insertion array's *rank* inside the per-direction loop, interleaved with that
direction's domain and multiplicity checks, so its order is ``rank(0), domain(0),
multiplicity(0), rank(1), ...``. A ``std::span`` has no rank, so the C++ path cannot
make that check and :func:`_flat_insertions` makes it here -- for every direction,
before the call.

What survives of that is doubly-bad input where an *earlier* direction fails a check
the oracle makes **after** rank, and a *later* direction fails rank. The oracle reports
the earlier direction's failure, the C++ path reports the later direction's rank. Both
texts are the oracle's, both refusals are ``ValueError``, and
``tests/parity/test_bspline_refinement.py`` pins them rather than leaving the
divergence to be met.

**Exactly which shapes those are, closed by enumeration.**
``_compute_inserted_knot_vector_1d`` makes four checks, in order: rank, empty, domain,
multiplicity. Rank is the check the C++ path hoists, so it cannot diverge. Empty cannot
either: the caller skips a zero-size array before the per-direction loop is reached, so
the later rank failure surfaces on both sides -- measured. That leaves **domain and
multiplicity, and both diverge.**

**This said "a single shape" and named only the domain one.** The multiplicity shape
diverges identically and no test constructed it, so nothing in the suite could have
distinguished the claim from the truth. Two reviewers found it independently; the count
above is now settled by enumerating the oracle's checks rather than by inspection.

**A third shape existed until a review round, and was a behaviour divergence rather
than an order one.** A *zero-size* non-1D array -- shape ``(0, 3)`` -- is skipped by the
oracle before its rank is ever looked at, and
:func:`_flat_insertions` refused it: the port raising where the oracle returns, which
is a divergence in behaviour rather than in a message and which no comparison of
refusal texts could have found. It now skips on size first, in the oracle's own order.

Cross-backend fields
--------------------

:func:`_cpp_handle` refuses a field built under the other backend rather than
converting it, which is what ``design/cross_backend_types.md`` forbids and what
:meth:`pantr.bspline.Bspline._mutate` already refuses for the three ``in_place=``
methods. The refusal is a property of *taking the C++ route*, so it fires under the C++
backend and not under the Python one, where the oracle runs happily over a C++ field's
read-only control points. That asymmetry is
:func:`pantr.bezier._bezier_backend._cpp_handle`'s, unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from .._backend import Backend, active_backend, available_backends
from ._bspline_knot_insertion import (
    _compute_uniform_subdivision_knots,
    _insert_knots_bspline,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from .._pantr_cpp import Bspline32 as _CppBspline32
    from .._pantr_cpp import Bspline64 as _CppBspline64
    from ._bspline import Bspline

    _CppHandle: TypeAlias = "_CppBspline32 | _CppBspline64"
    """A field's C++ implementation, of either storage format."""

_Knots: TypeAlias = "npt.NDArray[np.float32 | np.float64]"
"""A knot array in one of the two storage formats a B-spline may use."""


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
            f"Bspline: cannot refine a Bspline built under a different backend "
            f"({type(impl).__name__} against the active C++ one); the backend is "
            f"chosen per process, so this means the active one changed after this "
            f"Bspline was built."
        )
    return impl


def _the_cpp_backend_can_take_it(bspline: Bspline, refined: Sequence[bool]) -> bool:
    """Report whether the C++ path covers this call.

    Three conditions. The C++ backend has to be the active one; no direction being
    refined may be **periodic**, which is the declared boundary the module docstring
    argues; and at least one direction has to be refined at all.

    That third one looks redundant and is not. The two C++ entry points restate
    :class:`pantr.bspline.Bspline`'s own "at least one direction" refusal for the
    benefit of a C++ caller with no wrapper in front of them, while the oracle path
    returns an unrefined copy for the same argument. Both are right where they sit and
    they disagree, so a *dispatcher* that could reach either would be the one thing
    that must not differ between the backends. Nothing above it can reach the case --
    :meth:`~pantr.bspline.Bspline.insert_knots` and
    :meth:`~pantr.bspline.Bspline.subdivide` both refuse it first -- but this module's
    two entry points are private symbols in a package whose private symbols a
    downstream consumer already imports, so the asymmetry is closed here rather than
    left resting on the one caller staying the only one.

    Args:
        bspline (~pantr.bspline.Bspline): The field to refine.
        refined (Sequence[bool]): Whether each direction receives knots, in axis order.

    Returns:
        bool: True when :func:`_cpp_insert_knots` or :func:`_cpp_subdivide` may run.

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
    if not any(refined):
        return False
    spaces = bspline.space.spaces
    return not any(
        touched and spaces[direction].periodic for direction, touched in enumerate(refined)
    )


def _flat_insertions(new_knots_per_dim: Sequence[_Knots | None], dtype: Any) -> list[_Knots]:  # noqa: ANN401
    """Normalize the per-direction insertions to the 1D arrays the binding takes.

    ``None`` becomes an empty array, which is how the binding spells "skip this
    direction"; see the module docstring on why no ``Optional`` crosses.

    **A zero-size array is skipped before its rank is looked at**, and the order of
    those two tests is the whole of what makes this function faithful. The oracle
    skips on ``nk.size == 0`` in
    :func:`~pantr.bspline._bspline_knot_insertion._insert_knots_bspline`, one level
    above the rank check in ``_compute_inserted_knot_vector_1d``, so an array of shape
    ``(0, 3)`` never reaches that check and the refinement succeeds. Testing the rank
    first here refused it instead -- the port raising where the oracle returns, which is
    the worse direction of divergence and the one a message comparison would never
    surface.

    Args:
        new_knots_per_dim (Sequence[npt.NDArray | None]): One array of knots to insert
            per direction, or ``None``.
        dtype (Any): The field's storage format, which the arrays already carry.

    Returns:
        list[npt.NDArray[np.float32 | np.float64]]: One contiguous 1D array per
        direction.

    Raises:
        ValueError: If a non-empty array is not 1D, with the oracle's message. The C++
            path cannot make this check and the order it lands in differs from the
            oracle's; the module docstring says how and pins where.
    """
    flat: list[_Knots] = []
    for new_knots in new_knots_per_dim:
        if new_knots is None or new_knots.size == 0:
            flat.append(np.empty(0, dtype=dtype))
            continue
        if new_knots.ndim != 1:
            raise ValueError(f"new_knots must be a 1D array-like, got shape {new_knots.shape}")
        flat.append(np.ascontiguousarray(new_knots))
    return flat


def _cpp_insert_knots(bspline: Bspline, new_knots_per_dim: Sequence[_Knots | None]) -> Bspline:
    """Insert knots through the C++ binding.

    Args:
        bspline (~pantr.bspline.Bspline): The field, holding a C++ handle.
        new_knots_per_dim (Sequence[npt.NDArray | None]): One array of knots to insert
            per direction, already cast to the field's dtype.

    Returns:
        ~pantr.bspline.Bspline: The refined field, wrapping a C++ handle.

    Raises:
        ValueError: If a non-empty array is not 1D, if a knot lies outside its
            direction's domain, or if a merge would exceed the maximum multiplicity.
            Every message is the oracle's. "Every direction empty" is not among them:
            :func:`_the_cpp_backend_can_take_it` keeps that case on the oracle path,
            where it returns an unrefined copy.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    from ._bspline import Bspline as BsplineCls  # noqa: PLC0415  (cycle)

    handle = _cpp_handle(bspline)
    flat = _flat_insertions(new_knots_per_dim, bspline.dtype)
    return BsplineCls._wrap_over(_pantr_cpp.insert_bspline_knots(handle, flat), bspline.space)


def _cpp_subdivide(
    bspline: Bspline, counts: Sequence[int | None], regularity: int | None
) -> Bspline:
    """Subdivide through the C++ binding.

    Args:
        bspline (~pantr.bspline.Bspline): The field, holding a C++ handle.
        counts (Sequence[int | None]): Equal sub-spans per existing span, one per
            direction; ``None`` and 1 both skip.
        regularity (int | None): The continuity at each inserted knot, or ``None`` for
            ``degree - 1`` per direction.

    Returns:
        ~pantr.bspline.Bspline: The refined field, wrapping a C++ handle.

    Raises:
        ValueError: If a merge would exceed the maximum multiplicity. The count and
            regularity refusals were made by
            :meth:`pantr.bspline.Bspline.subdivide` above the branch, so both backends
            refuse the same argument with the same message.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (resolved against the .pyi stub)

    from ._bspline import Bspline as BsplineCls  # noqa: PLC0415  (cycle)

    handle = _cpp_handle(bspline)
    return BsplineCls._wrap_over(
        _pantr_cpp.subdivide_bspline(
            handle, [1 if count is None else int(count) for count in counts], regularity
        ),
        bspline.space,
    )


def insert_knots_into_field(
    bspline: Bspline, new_knots_per_dim: Sequence[_Knots | None]
) -> Bspline:
    """Insert knots into a B-spline field, on whichever backend covers the call.

    Args:
        bspline (~pantr.bspline.Bspline): The field to refine.
        new_knots_per_dim (Sequence[npt.NDArray | None]): One array of knot values to
            insert per direction, in axis order, already cast to the field's dtype.
            ``None`` or an empty array skips that direction.

    Returns:
        ~pantr.bspline.Bspline: A field over the same geometry, with refined knot
        vectors.

    Raises:
        TypeError: If the C++ backend is active and the field was built under the
            other one.
        ValueError: If a non-empty array is not 1D, if a knot lies outside its
            direction's domain, or if a merge would exceed the maximum multiplicity.
            An all-empty argument is not refused here; it returns an unrefined copy, as
            the oracle does. :meth:`~pantr.bspline.Bspline.insert_knots` is what refuses
            it.
    """
    refined = [nk is not None and nk.size > 0 for nk in new_knots_per_dim]
    if _the_cpp_backend_can_take_it(bspline, refined):
        return _cpp_insert_knots(bspline, new_knots_per_dim)
    return _insert_knots_bspline(bspline, list(new_knots_per_dim))


def subdivide_field(
    bspline: Bspline, counts: Sequence[int | None], regularity: int | None
) -> Bspline:
    """Uniformly subdivide a B-spline field, on whichever backend covers the call.

    The counts and the regularity have already been validated by
    :meth:`pantr.bspline.Bspline.subdivide`, so neither branch here re-derives which
    arguments are legal.

    Args:
        bspline (~pantr.bspline.Bspline): The field to refine.
        counts (Sequence[int | None]): Equal sub-spans per existing knot span, one per
            direction. ``None`` and 1 both skip that direction.
        regularity (int | None): The continuity at each inserted knot, applied to every
            subdivided direction. ``None`` uses ``degree - 1`` per direction.

    Returns:
        ~pantr.bspline.Bspline: A field over the same geometry, with uniformly refined
        knot vectors.

    Raises:
        TypeError: If the C++ backend is active and the field was built under the
            other one.
        ValueError: If a merge would exceed the maximum multiplicity.
    """
    refined = [count is not None and count > 1 for count in counts]
    if _the_cpp_backend_can_take_it(bspline, refined):
        return _cpp_subdivide(bspline, counts, regularity)

    # The oracle's own path: turn each active count into the knots it stands for, then
    # insert them. `_compute_uniform_subdivision_knots` cannot raise here -- the
    # regularity is in range and the count is at least 2, both established above the
    # branch -- which is what lets the validation live there instead of interleaved
    # with this loop, as it was before the C++ half existed.
    dtype = bspline.dtype
    new_knots_per_dim: list[_Knots | None] = []
    for direction, count in enumerate(counts):
        if count is None or count == 1:
            new_knots_per_dim.append(None)
            continue
        space_1d = bspline.space.spaces[direction]
        effective = space_1d.degree - 1 if regularity is None else regularity
        new_knots_per_dim.append(
            _compute_uniform_subdivision_knots(
                space_1d.knots, space_1d.degree, space_1d.tolerance, count, effective
            ).astype(dtype, copy=False)
        )
    return _insert_knots_bspline(bspline, new_knots_per_dim)
