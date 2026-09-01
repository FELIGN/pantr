"""1D B-spline space: knot vector, degree, and basis evaluation.

This module exposes :class:`BsplineSpace1D`, which combines a knot vector and
polynomial degree to define a 1D B-spline space. It provides methods for
basis evaluation, Bézier/Lagrange/cardinal extraction operators, and several
geometric properties (domain, intervals, cardinality).

Since ``design/cross_backend_types.md`` the *value* is owned by C++
(``cpp/include/pantr/bspline/space_1d.hpp``) and :class:`BsplineSpace1D` is a
wrapper holding one implementation of it. There are two implementations and they
are not two spaces: :class:`_BsplineSpace1DPython` is the oracle the port is
checked against, and the C++ handle is the thing being checked.
:func:`_impl_class` picks between them, per process and per dtype.

Only the *state and what it determines* moved. The operations -- basis
tabulation, the three extraction families, knot insertion, subdivision and
restriction -- are unchanged, still run on numba kernels, and now live on the
wrapper. That mixed dispatch is the temporary seam this front introduces; a
cleanup ticket removes it once the whole front lands.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Final, TypeAlias, cast

import numpy as np
from numpy import typing as npt

from .._backend import Backend, active_backend, available_backends
from ..basis import LagrangeVariant
from ._bspline_basis_core import (
    _tabulate_Bspline_basis_1D_impl,
    _tabulate_Bspline_basis_deriv_1D_impl,
)
from ._bspline_extraction import (
    _tabulate_Bspline_Bezier_1D_extraction_impl,
    _tabulate_Bspline_cardinal_1D_extraction_impl,
    _tabulate_Bspline_Lagrange_1D_extraction_impl,
)
from ._bspline_knot_insertion import (
    _compute_inserted_knot_vector_1d,
    _compute_uniform_subdivision_knots,
)
from ._bspline_knots import (
    _check_snapping_kept_an_interval,
    _check_space_has_an_interval,
    _get_Bspline_cardinal_intervals_1D_impl,
    _get_Bspline_num_basis_1D_impl,
    _get_unique_knots_and_multiplicity_impl,
    _knot_tolerance,
)

if TYPE_CHECKING:
    from .._pantr_cpp import BsplineSpace1D32 as _CppSpace32
    from .._pantr_cpp import BsplineSpace1D64 as _CppSpace64

    _Impl: TypeAlias = "_BsplineSpace1DPython | _CppSpace32 | _CppSpace64"
    """The implementation a :class:`BsplineSpace1D` holds: the oracle, or a handle.

    Type-checking only. The three are unrelated nominal types that happen to offer
    the same surface, which is the port's whole claim; naming the union here is
    what lets the checker verify it instead of taking it on trust.
    """

_Knots: TypeAlias = "npt.NDArray[np.float32 | np.float64]"
"""A knot vector in one of the two storage formats a space may hold."""

_SUPPORTED_DTYPES: Final = (np.dtype(np.float32), np.dtype(np.float64))
"""The two storage formats a space may hold, after integer input has been cast.

Rejecting anything else is the wrapper\'s job rather than either implementation\'s:
a dtype is a type-kind check, and ``pantr/core/error.hpp`` records that nanobind
has no path that produces a :class:`TypeError`.
"""


class _BsplineSpace1DPython:
    """The 1D space as the Python backend stores it: the port's oracle.

    Holds the knot vector, the degree, the periodicity flag and the tolerance
    those imply, and answers the questions they determine. Everything the class
    used to do beyond that is on :class:`BsplineSpace1D`, which is the only class
    a caller ever holds; this one is reachable only through it. When the C++ core
    stops being optional this class goes and :class:`BsplineSpace1D` collapses onto
    the handle.

    It receives a knot vector the wrapper has already normalized: a one-dimensional
    :class:`numpy.ndarray` of ``float32`` or ``float64``. The value checks below are
    still its own, because they are also the C++ type's, and the two must refuse the
    same inputs with the same messages.

    Attributes:
        _tol (float): Absolute tolerance for parametric comparisons on this space,
            in the units of its knots. Derived once at construction, from the knots
            **as supplied**, and never recomputed -- snapping can move the last knot
            onto its class's first one, which would move the scale.
        _knots (npt.NDArray[np.float32 | np.float64]): Knot vector, read-only.
        _degree (int): Polynomial degree of the B-spline.
        _periodic (bool): Whether the B-spline is periodic.
        _unique_all (tuple | None): Memo for
            :meth:`get_unique_knots_and_multiplicity` over the whole vector.
        _unique_in_domain (tuple | None): The same, restricted to the domain.
    """

    _tol: float
    _knots: _Knots
    _degree: int
    _periodic: bool
    _unique_all: tuple[_Knots, npt.NDArray[np.int_]] | None
    _unique_in_domain: tuple[_Knots, npt.NDArray[np.int_]] | None

    def __init__(
        self,
        knots: _Knots,
        degree: int,
        periodic: bool = False,
        snap_knots: bool | None = True,
    ) -> None:
        """Initialize the oracle's B-spline space.

        Args:
            knots (npt.NDArray[np.float32 | np.float64]): Knot vector, already
                normalized by the wrapper: 1D and of a supported float dtype.
            degree (int): Polynomial degree of the B-spline. Must be non-negative.
            periodic (bool): Whether the B-spline is periodic. Defaults to False.
            snap_knots (bool | None): Whether to snap nearby knots to avoid
                numerical issues. Defaults to True.

        Raises:
            ValueError: If degree is negative, knots are insufficient, or knots are
                not non-decreasing; if the resulting space would span no interval,
                i.e. every step between ``knots[degree]`` and ``knots[-degree-1]`` is
                within :attr:`tolerance`, so the domain is one knot and the space has
                no cell; or if ``snap_knots`` is set and snapping is what collapsed a
                knot vector that had more than one distinct knot onto that single
                point, which means the requested spacing is finer than the dtype
                resolves at that coordinate magnitude. ``snap_knots=False`` bypasses
                the merging and the snapping diagnosis, but not the interval
                requirement.
        """
        _BsplineSpace1DPython._validate_input(knots, degree, periodic)

        # Always own the storage: the vector is frozen read-only below, and
        # freezing a caller-supplied array in place would mutate caller state.
        knots_arr = knots.copy()
        self._knots = knots_arr

        # Derived from the knots, not from the dtype alone: the presets in
        # ``pantr.tolerance`` are dimensionless, and the magnitude a parametric
        # comparison on this space is relative to is the knot vector's own extent.
        self._tol = _knot_tolerance(self._knots)

        self._degree = int(degree)
        self._periodic = bool(periodic)
        self._unique_all = None
        self._unique_in_domain = None

        if snap_knots:
            self._snap_knots()
            # Runs first: when snapping is what destroyed the mesh, its diagnosis is
            # the specific one and names the remedy. The check below owns the same
            # symptom from any other cause, including ``snap_knots=False``.
            _check_snapping_kept_an_interval(knots_arr, self._knots, self._tol)

        _check_space_has_an_interval(self._knots, self._degree, self.num_intervals, self._tol)

        self._knots.flags.writeable = False

    @staticmethod
    def _validate_input(
        knots: _Knots,
        degree: int,
        periodic: bool = False,
    ) -> None:
        """Validate the value half of the B-spline input parameters.

        The type-kind half -- is this an array at all, is it one-dimensional, is its
        dtype one a space can store -- belongs to :func:`_stored_knots` on the
        wrapper, because it raises :class:`TypeError` and nanobind has no path that
        produces one. What is left here is what the C++ type also checks, in the
        same order and with the same messages.

        Args:
            knots (npt.NDArray[np.float32 | np.float64]): Knot vector to validate.
            degree (int): Degree to validate.
            periodic (bool): Whether the B-spline is periodic.

        Raises:
            ValueError: If degree is negative, knots are insufficient, or
                knots are not non-decreasing.
        """
        if degree < 0:
            raise ValueError("degree must be non-negative")

        if knots.size < (2 * degree + 2):
            raise ValueError("knots must have at least 2*degree+2 elements")

        if not np.all(np.diff(knots) >= 0):
            raise ValueError("knots must be non-decreasing")

        # Derived only here, after the shape and ordering checks: the tolerance
        # reads the endpoints, so it needs a vector that has them.
        tol = _knot_tolerance(knots)
        num_basis = _get_Bspline_num_basis_1D_impl(knots, degree, periodic, tol)
        if num_basis < (degree + 1):
            raise ValueError("Not enough knots for the specified degree")

    def _snap_knots(self) -> None:
        """Collapse each group of knots meant to be one knot onto a single stored value.

        The grouping is delegated to :func:`_get_unique_knots_and_multiplicity_impl`
        and the vector rebuilt from the classes it reports, so the space and its own
        accessor cannot disagree about which knots are the same knot -- two
        implementations of one idea is how they came to disagree before.

        The representative is the class's *first* knot, chosen and not averaged, so
        snapping returns values the caller supplied and is idempotent. Averaging is
        a fresh rounding: over ``degree + 1`` identical copies at large magnitude
        ``np.mean`` is not even the identity, so it moved the reported domain.

        It replaces :attr:`_knots` with the collapsed vector, of the same length,
        dtype and ordering. Called once from ``__init__``, before the array is
        frozen read-only.
        """
        unique, mult = _get_unique_knots_and_multiplicity_impl(
            self._knots, self._degree, self._tol, False
        )
        self._knots = np.repeat(unique, mult)

    @property
    def degree(self) -> int:
        """Get the polynomial degree of the B-spline.

        Returns:
            int: The degree.
        """
        return self._degree

    @property
    def knots(self) -> _Knots:
        """Get the knot vector.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The read-only knot vector.
        """
        return self._knots

    @property
    def periodic(self) -> bool:
        """Check if the B-spline is periodic.

        Returns:
            bool: True if periodic, False otherwise.
        """
        return self._periodic

    @property
    def tolerance(self) -> float:
        """Get the absolute tolerance used for parametric comparisons on this space.

        Returns:
            float: The absolute parametric tolerance; see
            :attr:`BsplineSpace1D.tolerance` for what it is derived from.
        """
        return self._tol

    @functools.cached_property
    def num_basis(self) -> int:
        """Get the number of basis functions.

        Returns:
            int: Number of basis functions.
        """
        return _get_Bspline_num_basis_1D_impl(self._knots, self._degree, self._periodic, self._tol)

    def get_unique_knots_and_multiplicity(
        self,
        in_domain: bool = False,
    ) -> tuple[_Knots, npt.NDArray[np.int_]]:
        """Get unique knots and their multiplicities.

        Memoized **per space**, in two slots rather than on a key. That is what the
        process-global cache this port removed was replaced by: the derived knot
        classes belong to the space, so there is nothing to key on and nothing a
        cache key could omit. The C++ implementation memoizes the same two arrays
        behind one flag, for the same reason.

        Args:
            in_domain (bool): If True, only consider knots in the domain.
                Defaults to False.

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[numpy.intp]]:
            The distinct knot values and their multiplicity counts, both read-only.
        """
        memo = self._unique_in_domain if in_domain else self._unique_all
        if memo is not None:
            return memo

        # ``tol`` goes in unrounded, exactly as :meth:`_snap_knots` passes it, so the
        # space and its own accessor apply one predicate rather than two that differ
        # in the last bits of the threshold.
        unique, mults = _get_unique_knots_and_multiplicity_impl(
            self._knots, self._degree, self._tol, in_domain
        )
        # Frozen because the same arrays are handed to every later caller.
        unique.flags.writeable = False
        mults.flags.writeable = False
        memo = (unique, mults)
        if in_domain:
            self._unique_in_domain = memo
        else:
            self._unique_all = memo
        return memo

    @functools.cached_property
    def num_intervals(self) -> int:
        """Get the number of intervals in the domain.

        Returns:
            int: Number of intervals.
        """
        unique_knots, _ = self.get_unique_knots_and_multiplicity(in_domain=True)
        return len(unique_knots) - 1

    @functools.cached_property
    def _first_basis_per_interval_cached(self) -> npt.NDArray[np.int64]:
        """Compute and freeze the first supported basis index of each interval.

        Cached because the knot vector is immutable, and frozen because the cached
        array is handed out directly.

        The index is exact integer arithmetic, no basis evaluation: the functions
        non-zero on the span starting at knot index ``k`` are ``k - degree`` through
        ``k``, so the first one is ``k - degree``, where ``k`` is the *last* position
        at which that unique knot occurs. Selecting the unique knots whose last
        position lies in ``[degree, num_basis)`` picks out exactly the knots that
        start an in-domain interval, which makes the first interval no different from
        the rest and needs no special case for a non-clamped knot vector.

        Returns:
            npt.NDArray[np.int64]: Read-only array of shape ``(num_intervals,)``.
        """
        _, multiplicities = self.get_unique_knots_and_multiplicity(in_domain=False)
        last_position = np.cumsum(np.asarray(multiplicities, dtype=np.int64)) - 1
        starts_an_interval = (last_position >= self._degree) & (last_position < self.num_basis)
        first_basis = (last_position[starts_an_interval] - self._degree).astype(np.int64)
        first_basis.flags.writeable = False
        return first_basis

    def first_basis_per_interval(self) -> npt.NDArray[np.int64]:
        """Get the index of the first B-spline function supported on each interval.

        Returns:
            npt.NDArray[np.int64]: Read-only ``int64`` array of shape
            ``(num_intervals,)``; see :meth:`BsplineSpace1D.first_basis_per_interval`.

        Raises:
            ValueError: If the space is periodic.
        """
        if self._periodic:
            raise ValueError(
                "first_basis_per_interval: periodic B-spline spaces are not supported."
            )
        return self._first_basis_per_interval_cached

    def _get_domain_indices(self) -> tuple[int, int]:
        """Get the domain boundary indices of the knot vector.

        Returns:
            tuple[int, int]: Tuple of (start_index, end_index) defining the domain.
        """
        return (self._degree, self._knots.size - self._degree - 1)

    @functools.cached_property
    def domain(self) -> tuple[np.float32 | np.float64, np.float32 | np.float64]:
        """Get the knot vector domain.

        Returns:
            tuple[np.float32 | np.float64, np.float32 | np.float64]: Tuple of
            (start_value, end_value) defining the domain.
        """
        i0, i1 = self._get_domain_indices()
        return (self._knots[i0], self._knots[i1])

    @functools.cached_property
    def _left_end_open(self) -> bool:
        """Whether the left end is open (cached; knots are immutable).

        Returns:
            bool: True if the first ``degree + 1`` knots are equal; always
            False for periodic splines regardless of knot values.
        """
        if self.periodic:
            return False

        # Check if the first degree+1 knots are equal
        # (we know that they are non-decreasing).  The comparison is absolute, as
        # ``_snap_knots`` already notes: ``np.isclose`` keeps its default
        # ``rtol=1e-5``, which would read a gap of up to ``1e-5 * |knots[degree]|``
        # as clamped (half the domain, on a knot vector based at 1e6).
        return bool(abs(float(self._knots[0]) - float(self._knots[self._degree])) <= self._tol)

    @functools.cached_property
    def _right_end_open(self) -> bool:
        """Whether the right end is open (cached; knots are immutable).

        Returns:
            bool: True if the last ``degree + 1`` knots are equal; always
            False for periodic splines regardless of knot values.
        """
        if self.periodic:
            return False

        # Check if the last degree+1 knots are equal
        # (we know that they are non-decreasing).  Absolute comparison, as in
        # :attr:`_left_end_open`.
        return bool(
            abs(float(self._knots[-self._degree - 1]) - float(self._knots[-1])) <= self._tol
        )

    @functools.cached_property
    def _bezier_like_knots(self) -> bool:
        """Whether the knots are a Bézier-like configuration (cached; knots are immutable).

        Returns:
            bool: True iff the spline is non-periodic, open on both ends, and
            ``num_basis == degree + 1`` (i.e., exactly one non-zero span).
        """
        return (
            (not self._periodic)
            and self._left_end_open
            and self._right_end_open
            and self.num_basis == (self._degree + 1)
        )

    def has_left_end_open(self) -> bool:
        """Check if the left end of the B-spline is open.

        Returns:
            bool: True if the left end is open, False otherwise.
        """
        return self._left_end_open

    def has_right_end_open(self) -> bool:
        """Check if the right end of the B-spline is open.

        Returns:
            bool: True if the right end is open, False otherwise.
        """
        return self._right_end_open

    def has_open_knots(self) -> bool:
        """Check if the B-spline has open ends.

        Returns:
            bool: True if both ends are open, False otherwise.
        """
        return self._left_end_open and self._right_end_open

    def has_Bezier_like_knots(self) -> bool:
        """Check if the knot vector represents a Bézier-like configuration.

        Returns:
            bool: True if knots have open ends and only one span.
        """
        return self._bezier_like_knots


def _stored_knots(knots: npt.ArrayLike) -> _Knots:
    """Normalize an array-like to the knot vector a space would be built from.

    The type-kind half of what :meth:`BsplineSpace1D.__init__` owes, in the order
    the oracle has always applied it, because two simultaneously bad arguments must
    produce the same message they always did.

    Args:
        knots (npt.ArrayLike): The knot vector as supplied.

    Returns:
        npt.NDArray[np.float32 | np.float64]: A one-dimensional array of a dtype a
        space can store. Integer input is cast to ``float64``; everything else keeps
        its dtype, which is how ``float32`` survives.

    Raises:
        TypeError: If the input is not a list or a one-dimensional array.
        ValueError: If the dtype is not one a space can store.
    """
    if isinstance(knots, list):
        arr = np.array(knots)
    elif not isinstance(knots, np.ndarray):
        raise TypeError("knots must be a 1D numpy array or Python list")
    else:
        arr = knots

    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float64)

    if arr.ndim != 1:
        raise TypeError("knots must be a 1D numpy array or Python list")

    if arr.dtype not in _SUPPORTED_DTYPES:
        raise ValueError("knots type must be float (32 or 64 bits)")

    return cast("_Knots", arr)


def _impl_class(dtype: np.dtype[Any]) -> type[_BsplineSpace1DPython] | type[Any]:
    """The implementation class the active backend and the dtype select.

    The backend is per process rather than per instance, deliberately, and for the
    reason ``design/cross_backend_types.md`` gives: two spaces built under different
    backends could otherwise meet in one :class:`~pantr.bspline.BsplineSpace`, and
    reconciling them would mean converting one implementation into the other, which
    is the shape that note forbids.

    The dtype, by contrast, *is* per instance. It has to be: ``float32`` is a
    supported storage format for a knot vector, and the C++ side carries the format
    in the class name because the class of the handle is the only thing left to
    carry it.

    Args:
        dtype (np.dtype[Any]): The dtype the knots will be stored in.

    Returns:
        type: The oracle under the Python backend, and the C++ class for that
        storage format otherwise.

    Raises:
        RuntimeError: If the C++ backend is requested and is not available.
    """
    if active_backend() is Backend.PYTHON:
        return _BsplineSpace1DPython
    if Backend.CPP not in available_backends():
        raise RuntimeError("the CPP backend is not available in this installation")
    from pantr import _pantr_cpp  # noqa: PLC0415  (optional, imported only when selected)

    if dtype == np.float32:
        return _pantr_cpp.BsplineSpace1D32
    return _pantr_cpp.BsplineSpace1D64


def _new_impl(
    knots: _Knots,
    degree: int,
    periodic: bool,
    snap_knots: bool | None,
) -> _Impl:
    """Build a 1D space in whichever implementation the active backend selects.

    Args:
        knots (npt.NDArray[np.float32 | np.float64]): Knots, already normalized by
            :func:`_stored_knots`.
        degree (int): Polynomial degree.
        periodic (bool): Whether the space is periodic.
        snap_knots (bool | None): Whether to merge knots that are the same knot.

    Returns:
        _Impl: The implementation object; an oracle instance or a C++ handle.

    Raises:
        ValueError: Whatever the implementation refuses the arguments with.
        RuntimeError: If the C++ backend is requested and is not available.
    """
    cls = _impl_class(knots.dtype)
    if cls is _BsplineSpace1DPython:
        return _BsplineSpace1DPython(knots, degree, periodic, snap_knots)
    # Each C++ class accepts only its own storage format and refuses rather than
    # casts, so the array's dtype and the class have to agree. They do: the class
    # was chosen from that dtype one line above. That is a correlation between a
    # value and a type, which the checker cannot state, and the cast is what stands
    # in for it.
    return cast(
        "_Impl",
        cls(np.ascontiguousarray(knots), int(degree), bool(periodic), bool(snap_knots)),
    )


class BsplineSpace1D:
    """A class representing a 1D B-spline with configurable degree and knot vector.

    This class provides methods to analyze B-spline properties, validate input
    parameters, compute various geometric characteristics of the spline,
    and access various properties of the B-spline.

    An instance always has :attr:`num_intervals` at least 1: a knot vector whose
    in-domain knots all fall in one class is refused at construction, since such a
    space has no cell and nothing can be evaluated, tabulated or located on it.

    **This class is a wrapper.** The value -- the knots, the degree, the
    periodicity and everything they fix -- is owned by an implementation chosen by
    :func:`_impl_class`, which is the C++ type
    (``cpp/include/pantr/bspline/space_1d.hpp``) or the oracle
    :class:`_BsplineSpace1DPython`. The operations below are still Python over
    numba kernels and are unchanged; only the state moved.

    Instances are immutable, and ``__slots__`` is what says so: there is no
    ``__dict__``, so nothing can be attached to a space after it is built.

    Attributes:
        _impl (_Impl): The implementation this wrapper holds; see
            :func:`_impl_class`.
    """

    __slots__ = ("_impl",)

    _impl: _Impl
    """The implementation this wrapper holds; see :func:`_impl_class`."""

    def __init__(
        self,
        knots: npt.ArrayLike,
        degree: int,
        periodic: bool = False,
        snap_knots: bool | None = True,
    ) -> None:
        """Initialize a B-spline 1D object.

        Args:
            knots (npt.ArrayLike): Knot vector defining the B-spline. Must be non-decreasing
                and have at least 2*degree+2 elements.
            degree (int): Polynomial degree of the B-spline. Must be non-negative.
            periodic (bool): Whether the B-spline is periodic. Defaults to False.
            snap_knots (bool | None): Whether to snap nearby knots to avoid numerical issues.
                Defaults to True.

        Raises:
            ValueError: If degree is negative, knots are insufficient, or knots are
                not non-decreasing; if the resulting space would span no interval,
                i.e. every step between ``knots[degree]`` and ``knots[-degree-1]`` is
                within :attr:`tolerance`, so the domain is one knot and the space has
                no cell; or if
                ``snap_knots`` is set and snapping is what collapsed a knot vector
                that had more than one distinct knot onto that single point, which
                means the requested spacing is finer than the dtype resolves at that
                coordinate magnitude. ``snap_knots=False`` bypasses the merging and
                the snapping diagnosis, but not the interval requirement.
            TypeError: If knots cannot be converted to a numpy array.
        """
        # The degree is checked here as well as in the implementation, and the
        # duplication is the point: the oracle has always refused a negative degree
        # *before* looking at the knots at all, so a call with both a bad degree and
        # a bad knot type has always reported the degree. Normalizing the knots
        # first would silently reverse that, and the seam these two checks sit
        # either side of did not exist when the order was chosen.
        if degree < 0:
            raise ValueError("degree must be non-negative")
        self._impl = _new_impl(_stored_knots(knots), degree, periodic, snap_knots)

    def __reduce__(
        self,
    ) -> tuple[type[BsplineSpace1D], tuple[_Knots, int, bool, bool]]:
        """Pickle by the constructor's arguments rather than by implementation.

        The C++ handle is not picklable and must not become part of the wire format:
        a pickle written under the C++ backend has to load under the Python one and
        the other way round, or the backend switch would silently become a
        data-format switch.

        The knots are copied out of the read-only view, because pickle preserves the
        flag and the oracle's constructor would then store a read-only array where
        it expects to own a writable one. Snapping is off on the way back in: the
        stored vector is already snapped, so re-snapping is a no-op by idempotence,
        and skipping it makes the reconstruction independent of a scan.

        Note:
            The tolerance is *recomputed* from the stored knots rather than carried,
            which is what keeps the wire format the constructor's arguments and
            nothing else. Where snapping moved the last knot onto its class's first
            one, the scale moves with it and the reconstructed tolerance differs
            from the original's. The bound is

            ``|tol' - tol| / tol <= (m - 1) * 8 * eps``,

            where ``m`` is the multiplicity of the **last** knot class. Snapping
            replaces each class by its *first* knot, so the first knot of the vector
            never moves and only the last one does; every step inside a class is at
            most one tolerance, so it moves by at most ``(m - 1) * tol``. The scale
            is a maximum over three arguments and so is 1-Lipschitz in each, so it
            moves by no more than that, and the tolerance is a fixed multiple of the
            scale. Recomputing the scale and the tolerance adds a further relative
            ``eps`` or so, which the measured margin absorbs.

            The ``m - 1`` is not decoration. A first version of this note claimed a
            flat ``8 * eps``, reasoning that the scale moves by at most one
            tolerance; that ignores chaining, and a sweep built to chain the final
            class **exceeded it by a factor of 4.4**. ``tests/parity`` pins the
            corrected form, checks that the drift is actually non-zero, and checks
            that the flat version fails.

        Returns:
            tuple: The class, and the knots, degree, periodicity and snapping flag
            to rebuild it from.
        """
        return (type(self), (np.array(self.knots), self.degree, self.periodic, False))

    @property
    def degree(self) -> int:
        """Get the polynomial degree of the B-spline.

        Returns:
            int: The degree.
        """
        return int(self._impl.degree)

    @property
    def knots(self) -> _Knots:
        """Get the knot vector.

        Read-only under both backends, and under the C++ one it is a view of the
        space's own storage rather than a copy, so it stays valid after the space is
        dropped and costs nothing per access.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The knot vector.
        """
        return self._impl.knots

    @property
    def periodic(self) -> bool:
        """Check if the B-spline is periodic.

        Returns:
            bool: True if periodic, False otherwise.
        """
        return bool(self._impl.periodic)

    @property
    def tolerance(self) -> float:
        """Get the absolute tolerance used for parametric comparisons on this space.

        It is in the units of the knot vector, being the dimensionless strict
        preset of :mod:`pantr.tolerance` scaled by the knot vector's own magnitude
        (``max(span, |knots[0]|, |knots[-1]|)``) and doubled for one extra rounding
        upstream. Two knots closer than this are the same knot, a point within this
        of an end is inside the domain, and two knot intervals differing by less
        than this are the same length.

        Being an absolute parametric length, it is *not* the factor to scale a
        physical coordinate or a spline coefficient by; take the dimensionless
        preset from :mod:`pantr.tolerance` for those and scale it by the magnitude
        that applies.

        It is derived from the knots **as supplied**, before snapping, and never
        recomputed: snapping can move the last knot onto its class's first one,
        which would move the scale.

        Returns:
            float: The absolute parametric tolerance.
        """
        return float(self._impl.tolerance)

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the data type of the knot vector (and used in computations).

        Read off the knots rather than delegated: the C++ handle carries its storage
        format in its class name, not as a numpy dtype it could hand back.

        Returns:
            npt.DTypeLike: The numpy data type of the knots.
        """
        return self.knots.dtype

    @property
    def num_basis(self) -> int:
        """Get the number of basis functions.

        This depends on the knot vector length and the degree, but
        also on whether the B-spline is periodic.

        Returns:
            int: Number of basis functions.
        """
        return int(self._impl.num_basis)

    @property
    def num_intervals(self) -> int:
        """Get the number of intervals in the domain.

        Returns:
            int: Number of intervals.

        Example:
            >>> space = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
            >>> space.num_intervals
            2
        """
        return int(self._impl.num_intervals)

    @property
    def domain(self) -> tuple[np.float32 | np.float64, np.float32 | np.float64]:
        """Get the knot vector domain.

        Returns:
            tuple[np.float32 | np.float64, np.float32 | np.float64]: Tuple of
            (start_value, end_value) defining the domain.

        Example:
            >>> bspline = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
            >>> start, end = bspline.domain
            >>> float(start), float(end)
            (0.0, 2.0)
        """
        # Presented as numpy scalars of the space's own dtype under either backend:
        # the oracle indexes its array and gets one already, the C++ handle returns
        # a Python float because that is what nanobind casts a `T` to, and a caller
        # must not be able to tell which one built the space. Routed through a
        # two-element array rather than the domain indices, so that the formula for
        # where the domain starts and ends lives in the implementation only.
        ends = np.asarray(self._impl.domain, dtype=self.dtype)
        return (ends[0], ends[1])

    def get_unique_knots_and_multiplicity(
        self,
        in_domain: bool = False,
    ) -> tuple[_Knots, npt.NDArray[np.int_]]:
        """Get unique knots and their multiplicities.

        Computed once per space, under both backends. There is no cache key, and
        deliberately so: the derived knot classes belong to the space.

        **Hoist it out of a loop.** What is memoized is the *values*; the numpy
        arrays presenting them are built per call, and under the C++ backend that
        construction dominates the access and costs more than the whole call does
        under the oracle. ``design/bspline_derived_caches.md`` rejects memoizing the
        presentation as well, because two memos over one buffer are correct only
        until something rebuilds it -- so the accessor looks free and is not, and the
        fix is a local rather than a mechanism. The C++ side carries the same warning
        in ``pantr/core/lazy.hpp``.

        Args:
            in_domain (bool): If True, only consider knots in the domain.
                Defaults to False.

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[numpy.intp]]: Tuple of
            (unique_knots, multiplicities) where unique_knots contains the distinct knot values
            and multiplicities contains the corresponding multiplicity counts. Both arrays
            are read-only. With ``in_domain=False`` the multiplicities sum to
            ``knots.size``, so ``np.repeat(unique_knots, multiplicities)`` rebuilds
            the whole vector with each class collapsed onto its representative.
        """
        return self._impl.get_unique_knots_and_multiplicity(in_domain)

    def first_basis_per_interval(self) -> npt.NDArray[np.int64]:
        """Get the index of the first B-spline function supported on each interval.

        Entry ``j`` is the smallest global function index ``i`` whose function is
        non-zero on the open interval between in-domain unique knots ``j`` and
        ``j + 1``. The functions non-zero there are exactly ``i`` through
        ``i + degree``, so this one index describes the whole support of the interval.

        The result is computed once per space and read-only. Repeated calls view
        the same memory; they do **not** necessarily return the same numpy object.
        The oracle's ``cached_property`` does return one object, and this was
        documented as such until the port, but the C++ backend memoizes the buffer
        and builds a view over it per access -- so a caller keying on ``id()`` would
        see the two backends differ. ``np.shares_memory`` is what holds under both.

        Returns:
            npt.NDArray[np.int64]: Read-only ``int64`` array of shape
            ``(num_intervals,)``, non-decreasing, whose successive differences are
            the interior knot multiplicities.

        Raises:
            ValueError: If the space is periodic.

        Example:
            >>> space = BsplineSpace1D([0, 0, 0, 1, 2, 2, 3, 3, 3], 2)
            >>> space.first_basis_per_interval().tolist()
            [0, 1, 3]
        """
        return self._impl.first_basis_per_interval()

    def has_left_end_open(self) -> bool:
        """Check if the left end of the B-spline is open.

        A left end is open if the first degree+1 knots are equal.

        Returns:
            bool: True if the left end is open, False otherwise.
        """
        return bool(self._impl.has_left_end_open())

    def has_right_end_open(self) -> bool:
        """Check if the right end of the B-spline is open.

        A right end is open if the last degree+1 knots are equal.

        Returns:
            bool: True if the right end is open, False otherwise.
        """
        return bool(self._impl.has_right_end_open())

    def has_open_knots(self) -> bool:
        """Check if the B-spline has open ends.

        Returns:
            bool: True if both ends are open, False otherwise.
        """
        return bool(self._impl.has_open_knots())

    def has_Bezier_like_knots(self) -> bool:
        """Check if the knot vector represents a Bézier-like configuration.

        A Bézier-like configuration has open ends and only one non-zero span.

        Returns:
            bool: True if knots have open ends and only one span.

        Example:
            >>> bspline = BsplineSpace1D([1, 1, 1, 3, 3, 3], 2)
            >>> bspline.has_Bezier_like_knots()
            True
        """
        return bool(self._impl.has_Bezier_like_knots())

    def get_cardinal_intervals(
        self, out: npt.NDArray[np.bool_] | None = None
    ) -> npt.NDArray[np.bool_]:
        """Get boolean array indicating whether the intervals (non-zero spans) are cardinal or not.

        An interval is cardinal if has the same length as the degree-1
        previous and the degree-1 next intervals.

        Concretely, an interval ``[u_k, u_(k+1)]`` is cardinal when both conditions
        hold: neither bounding knot is repeated, and the ``2 * degree - 1`` knot spans
        of the window ``u_(k-degree+1), ..., u_(k+degree)`` -- the ``degree - 1`` spans
        before the interval, the interval itself, and the ``degree - 1`` spans after it
        -- all have the interval's own length, to within the space tolerance.

        Those ``2 * degree`` knots are exactly the ones an evaluation on that span
        reads. The standard basis-function recurrence works up from level ``1`` to
        ``degree``, and at level ``j`` it references only ``u_(k+1-j)`` and ``u_(k+j)``;
        across all levels those indices sweep ``k-degree+1`` through ``k+degree`` and no
        further. So the window holds every knot the ``degree + 1`` functions restricted
        to that interval depend on, and equal spacing across it is what makes those
        restrictions coincide with the cardinal (uniform-knot) ones.

        In the case of open knot vectors, this definition automatically
        discards the first degree-1 and the last degree-1 intervals: the repeated
        end knots put zero-length spans inside their window.

        At ``degree == 0`` the equal-length condition ranges over an empty set of
        neighbouring intervals, so it holds vacuously and the knot spacing does not
        affect the answer. What remains is the multiplicity gate that applies at every
        degree: an interval is reported cardinal unless one of the two knots bounding
        it is repeated. Either way this is consistent with the geometry, since a
        degree-0 space carries one basis function per interval and its cardinal
        extraction operator is the 1x1 identity on every interval regardless of the
        flag.

        Args:
            out (npt.NDArray[bool] | None): Optional output array where the result will be
                stored. If None, a new array is allocated. Must have the correct shape and dtype
                if provided. This follows NumPy's style for output arrays. Defaults to None.

        Returns:
            npt.NDArray[bool]: Boolean array where True indicates cardinal intervals.
                It has length equal to the number of intervals. If `out` was provided,
                returns the same array.

        Raises:
            ValueError: If `out` is provided and has incorrect shape or dtype.

        Example:
            Six unit intervals, degree 2: the window is one span either side, so
            only the two end intervals see a zero-length span from the repeated
            end knots.

            >>> bspline = BsplineSpace1D([0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6], 2)
            >>> bspline.get_cardinal_intervals().tolist()
            [False, True, True, True, True, False]

            Repeating the interior knot 5 disqualifies the two intervals it bounds:

            >>> bspline = BsplineSpace1D([0, 0, 0, 1, 2, 3, 4, 5, 5, 6, 6, 6], 2)
            >>> bspline.get_cardinal_intervals().tolist()
            [False, True, True, True, False, False]

            Degree 3 over the domain ``[3, 5]``: the window reaches two spans past
            each interval, so the irregular span ``[7, 10]`` falls outside both and
            neither interval is affected.

            >>> bspline = BsplineSpace1D([0, 1, 2, 3, 4, 5, 6, 7, 10], 3)
            >>> bspline.get_cardinal_intervals().tolist()
            [True, True]

        References:
            The recurrence whose knot footprint sets this window is Algorithm A2.2
            of :cite:p:`piegl1997nurbs`.
        """
        return _get_Bspline_cardinal_intervals_1D_impl(
            self.knots, self.degree, self.tolerance, out=out
        )

    def tabulate_basis(
        self,
        pts: npt.ArrayLike,
        out_basis: npt.NDArray[np.float32 | np.float64] | None = None,
        out_first_basis: npt.NDArray[np.int_] | None = None,
        *,
        validate: bool = True,
    ) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
        """Evaluate the B-spline basis functions at the given points.

        Args:
            pts (npt.ArrayLike): Evaluation points.
            out_basis (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
                basis values will be stored. If None, a new array is allocated. Must have the
                correct shape and dtype if provided. This follows NumPy's style for output arrays.
                Defaults to None.
            out_first_basis (npt.NDArray[numpy.intp] | None): Optional output array where the
                first basis indices will be stored. If None, a new array is allocated. Must have
                the correct shape and dtype numpy.intp if provided. This follows NumPy's style for
                output arrays. Defaults to None.
            validate (bool): If True (default), check that every point lies inside
                the spline domain. Pass False only when the caller guarantees
                in-domain points; out-of-domain points are then undefined behavior.
                Defaults to True.

        Returns:
            tuple[
                npt.NDArray[np.float32] | npt.NDArray[np.float64],
                npt.NDArray[numpy.intp]
            ]: Tuple containing:
                - basis_values: (npt.NDArray[np.float32] | npt.NDArray[np.float64])
                  Array of shape matching `pts` with the last dimension length (degree+1),
                  containing the basis function values evaluated at each point.
                  If `out_basis` was provided, returns the same array.
                - first_basis_indices: (npt.NDArray[numpy.intp])
                  1D integer array indicating the index of the first nonzero basis function
                  for each evaluation point. The length is the same as the number
                  of evaluation points. If `out_first_basis` was provided, returns the same array.

        Raises:
            ValueError: If ``validate`` is True and any evaluation point is outside the
                B-spline domain, or if `out_basis` or `out_first_basis` is provided and
                has incorrect shape or dtype.

        Example:
            >>> import numpy as np
            >>> bspline = BsplineSpace1D([0, 0, 0, 0.25, 0.7, 0.7, 1, 1, 1], 2)
            >>> values, first = bspline.tabulate_basis([0.0, 0.5, 0.75, 1.0])
            >>> np.allclose(
            ...     values,
            ...     [
            ...         [1.0, 0.0, 0.0],
            ...         [0.12698413, 0.5643739, 0.30864198],
            ...         [0.69444444, 0.27777778, 0.02777778],
            ...         [0.0, 0.0, 1.0],
            ...     ],
            ... )
            True
            >>> first.tolist()
            [0, 1, 3, 3]

        References:
            Basis values are computed with the stable Cox-de Boor recurrence
            :cite:p:`deboor2001splines`.
        """
        return _tabulate_Bspline_basis_1D_impl(
            self, pts, out_basis=out_basis, out_first_basis=out_first_basis, validate=validate
        )

    def tabulate_basis_derivatives(
        self,
        pts: npt.ArrayLike,
        n_deriv: int,
        out_deriv: npt.NDArray[np.float32 | np.float64] | None = None,
        out_first_basis: npt.NDArray[np.int_] | None = None,
        *,
        validate: bool = True,
    ) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
        """Evaluate B-spline basis function derivatives at the given points.

        Implements Algorithm A2.3 (DerBasisFuncs) from Piegl & Tiller.
        The 0th slice ``deriv_values[..., 0, :]`` is identical to the result
        of :meth:`tabulate_basis`.  For ``n_deriv > degree`` all rows beyond
        ``degree`` are identically zero.

        Args:
            pts (npt.ArrayLike): Evaluation points.
            n_deriv (int): Maximum derivative order to compute (>= 0).
            out_deriv (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array for derivative values. If None, a new array is allocated. Must
                have shape ``(*pts_shape, n_deriv+1, degree+1)`` and dtype matching
                ``pts`` if provided. Defaults to None.
            out_first_basis (npt.NDArray[numpy.intp] | None): Optional output array
                for first basis indices. If None, a new array is allocated. Must have
                shape ``pts_shape`` and dtype ``numpy.intp`` if provided.
                Defaults to None.
            validate (bool): If True (default), check that every point lies inside
                the spline domain. Pass False only when the caller guarantees
                in-domain points; out-of-domain points are then undefined behavior.
                Defaults to True.

        Returns:
            tuple[
                npt.NDArray[np.float32] | npt.NDArray[np.float64],
                npt.NDArray[numpy.intp]
            ]: Tuple containing:
                - deriv_values: Array of shape ``(*pts_shape, n_deriv+1, degree+1)``.
                  ``deriv_values[..., k, i]`` is the k-th derivative of the i-th
                  local B-spline basis function at each point.
                - first_basis_indices: Integer array of shape ``pts_shape`` giving the
                  global index of the first nonzero basis function for each point.

        Raises:
            ValueError: If ``n_deriv < 0``, if ``validate`` is True and any evaluation
                point is outside the domain, or ``out_deriv`` / ``out_first_basis``
                has incorrect shape or dtype.

        Example:
            >>> import numpy as np
            >>> bspline = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
            >>> d, first = bspline.tabulate_basis_derivatives([0.5], n_deriv=1)
            >>> d.shape
            (1, 2, 3)
            >>> # first derivatives at x=0.5: B0'=-1, B1'=0, B2'=1
            >>> np.allclose(d[0, 1, :], [-1.0, 0.0, 1.0])
            True
        """
        return _tabulate_Bspline_basis_deriv_1D_impl(
            self,
            pts,
            n_deriv,
            out_deriv=out_deriv,
            out_first_basis=out_first_basis,
            validate=validate,
        )

    def tabulate_Bezier_extraction_operators(
        self, out: npt.NDArray[np.float32 | np.float64] | None = None
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Create Bézier extraction operators of the B-spline.

        Args:
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
                result will be stored. If None, a new array is allocated. Must have the correct
                shape and dtype if provided. This follows NumPy's style for output arrays.
                Defaults to None.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Array of extraction matrices with shape
                (n_intervals, degree+1, degree+1) where each matrix transforms
                Bernstein basis functions to B-spline basis functions for that interval.

                Each matrix C[i, :, :] transforms Bernstein basis functions
                to B-spline basis functions for the i-th interval as

                    C[i, :, :] @ [Bernstein values] = [B-spline values in interval].

                If `out` was provided, returns the same array.

        Raises:
            ValueError: If `out` is provided and has incorrect shape or dtype.
        """
        return _tabulate_Bspline_Bezier_1D_extraction_impl(
            self.knots, self.degree, self.tolerance, out=out
        )

    def tabulate_Lagrange_extraction_operators(
        self,
        lagrange_variant: LagrangeVariant = LagrangeVariant.EQUISPACES,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Create Lagrange extraction operators of the B-spline.

        Args:
            lagrange_variant (LagrangeVariant): Lagrange point distribution to use
                (e.g., equispaced, Gauss-Lobatto-Legendre, etc).
                Defaults to `LagrangeVariant.EQUISPACES`.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
                result will be stored. If None, a new array is allocated. Must have the correct
                shape and dtype if provided. This follows NumPy's style for output arrays.
                Defaults to None.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Array of extraction matrices with shape
                (n_intervals, degree+1, degree+1) where each matrix transforms
                Lagrange basis functions to B-spline basis functions for that interval.

                Each matrix C[i, :, :] transforms Lagrange basis functions
                to B-spline basis functions for the i-th interval as

                    C[i, :, :] @ [Lagrange values] = [B-spline values in interval].

                If `out` was provided, returns the same array.

        Raises:
            ValueError: If `out` is provided and has incorrect shape or dtype.
        """
        return _tabulate_Bspline_Lagrange_1D_extraction_impl(
            self.knots, self.degree, self.tolerance, lagrange_variant=lagrange_variant, out=out
        )

    def tabulate_cardinal_extraction_operators(
        self, out: npt.NDArray[np.float32 | np.float64] | None = None
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Create cardinal B-spline extraction operators of the B-spline.

        Args:
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
                result will be stored. If None, a new array is allocated. Must have the correct
                shape and dtype if provided. This follows NumPy's style for output arrays.
                Defaults to None.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Array of extraction matrices with shape
                (n_intervals, degree+1, degree+1) where each matrix transforms
                cardinal spline basis functions to B-spline basis functions for that interval.

                Each matrix C[i, :, :] transforms cardinal spline basis functions
                to B-spline basis functions for the i-th interval as

                    C[i, :, :] @ [cardinal values] = [B-spline values in interval].

                If `out` was provided, returns the same array.

        Raises:
            ValueError: If `out` is provided and has incorrect shape or dtype.
        """
        return _tabulate_Bspline_cardinal_1D_extraction_impl(
            self.knots, self.degree, self.tolerance, out=out
        )

    def insert_knots(self, new_knots: npt.ArrayLike) -> BsplineSpace1D:
        """Return a new BsplineSpace1D with additional knots inserted.

        Args:
            new_knots (npt.ArrayLike): 1D array-like of knot values to insert.
                Must be non-empty.  Values must lie in
                ``[knots[degree], knots[-degree-1]]``.  Inserting a value
                already present increases its multiplicity; multiplicity cannot
                exceed ``degree + 1``.  Repeated values in ``new_knots`` insert
                that knot multiple times in one call.

        Returns:
            BsplineSpace1D: New space with the inserted knots.

        Raises:
            ValueError: If ``new_knots`` is empty.
            ValueError: If ``new_knots`` is not 1D.
            ValueError: If any new knot lies outside the domain.
            ValueError: If inserting a knot would exceed maximum multiplicity
                ``degree + 1``.
        """
        arr = np.asarray(new_knots, dtype=self.dtype)
        if arr.size == 0:
            raise ValueError("new_knots must not be empty.")
        merged = _compute_inserted_knot_vector_1d(self.knots, self.degree, arr, self.tolerance)
        return BsplineSpace1D(merged, self.degree)

    def subdivide(self, n_subdivisions: int, regularity: int | None = None) -> BsplineSpace1D:
        """Return a new BsplineSpace1D with each knot span split into equal sub-spans.

        For every non-zero knot interval ``[t_i, t_{i+1})``, inserts
        ``n_subdivisions - 1`` uniformly spaced knot values.  Each value is
        repeated ``degree - regularity`` times so that the resulting B-spline
        has ``C^regularity`` continuity at every inserted knot.

        Args:
            n_subdivisions (int): Number of equal sub-spans per existing knot
                interval.  Must be >= 2.
            regularity (int | None): Continuity order at every inserted knot.
                Must be in ``[-1, degree - 1]``.  ``None`` (default) uses
                ``degree - 1``, giving maximum continuity (multiplicity 1).

        Returns:
            BsplineSpace1D: New space with uniformly refined knot vector.

        Raises:
            ValueError: If ``n_subdivisions < 2``.
            ValueError: If ``regularity`` is outside ``[-1, degree - 1]``.
        """
        if n_subdivisions < 2:  # noqa: PLR2004
            raise ValueError(f"n_subdivisions must be >= 2, got {n_subdivisions}")
        eff_regularity = self.degree - 1 if regularity is None else regularity
        if eff_regularity < -1 or eff_regularity > self.degree - 1:
            raise ValueError(
                f"regularity must be in [-1, degree - 1] = [-1, {self.degree - 1}], "
                f"got {eff_regularity}"
            )
        new_knots = _compute_uniform_subdivision_knots(
            self.knots, self.degree, self.tolerance, n_subdivisions, eff_regularity
        )
        return self.insert_knots(new_knots)

    def restrict(
        self, interval_lo: int, interval_hi: int
    ) -> tuple[BsplineSpace1D, npt.NDArray[np.int64]]:
        """Return the windowed sub-space over the intervals ``[interval_lo, interval_hi)``.

        The windowed knot vector is a pure slice of this space's knots (never
        re-clamped), chosen so the windowed basis functions are exactly the global
        basis functions supported on the windowed intervals; they therefore equal
        the global basis pointwise over those intervals.

        Args:
            interval_lo (int): First interval (knot span) of the window, inclusive.
            interval_hi (int): One past the last interval of the window, exclusive.
                Must satisfy ``0 <= interval_lo < interval_hi <= num_intervals``.

        Returns:
            tuple[BsplineSpace1D, npt.NDArray[np.int64]]: The windowed space and a
            read-only ``local_to_global_dof`` array of shape
            ``(windowed_space.num_basis,)`` mapping each windowed basis index to its
            index in this space.

        Raises:
            ValueError: If the space is periodic, or the interval range is invalid.
        """
        if self.periodic:
            raise ValueError("restrict: periodic B-spline spaces are not supported.")
        n_int = self.num_intervals
        lo, hi = int(interval_lo), int(interval_hi)
        if not 0 <= lo < hi <= n_int:
            raise ValueError(
                f"restrict: require 0 <= interval_lo < interval_hi <= {n_int}; got [{lo}, {hi})."
            )
        unique_knots, _ = self.get_unique_knots_and_multiplicity(in_domain=True)
        mids = np.array(
            [
                0.5 * (unique_knots[lo] + unique_knots[lo + 1]),
                0.5 * (unique_knots[hi - 1] + unique_knots[hi]),
            ],
            dtype=self.knots.dtype,
        )
        _, first_basis = self.tabulate_basis(mids)
        j_lo = int(first_basis[0])
        j_hi = int(first_basis[1]) + self.degree
        windowed_knots = self.knots[j_lo : j_hi + self.degree + 2]
        windowed = BsplineSpace1D(windowed_knots, self.degree, periodic=False, snap_knots=False)
        local_to_global_dof = np.arange(j_lo, j_hi + 1, dtype=np.int64)
        local_to_global_dof.flags.writeable = False
        return windowed, local_to_global_dof
