"""Multi-dimensional B-spline spaces using tensor products.

This module defines :class:`BsplineSpace`, which aggregates multiple
:class:`~pantr.bspline.BsplineSpace1D` objects to represent
multi-dimensional parameter domains. It handles tensor-product basis evaluation
by combining the 1D components.

Since ``design/cross_backend_types.md`` the *value* is owned by C++
(``cpp/include/pantr/bspline/space_nd.hpp``) and :class:`BsplineSpace` is a wrapper
holding one implementation of it, exactly as
:mod:`pantr.bspline._bspline_space_1d` does for the univariate type. There are two
implementations and they are not two spaces: :class:`_BsplineSpaceNDPython` is the
oracle the port is checked against, and the C++ handle is the thing being checked.
:func:`_impl_class` picks between them, per process and per dtype.

Only the *state and what it determines* moved. The operations --
:meth:`BsplineSpace.tabulate_basis`, :meth:`BsplineSpace.cell_supports`,
:meth:`BsplineSpace.boundary_dofs` and :meth:`BsplineSpace.restrict` -- are
computations *over* a space rather than properties *of* one, so they are unchanged,
still run on numba kernels and numpy, and live on the wrapper. That is the same line
``space_1d.hpp`` draws to keep ``get_cardinal_intervals`` out of the 1D type, and the
mixed dispatch it produces is the temporary seam this front introduces; a cleanup
ticket removes it once the whole front lands.

The wrapper keeps the univariate wrappers it was built from, in ``_spaces``, so that
``space.spaces[0] is space_1d`` holds. ``design/bspline_ownership_lifetime.md`` F6
records why that is an identity contract rather than a convenience, and it is what
requires the C++ constructor to *share* its directions rather than copy them.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple, NoReturn, TypeAlias, cast

import numpy as np
from numpy import typing as npt

from .._backend import Backend, active_backend, available_backends
from ._bspline_basis_multidim import _tabulate_Bspline_basis_impl
from ._bspline_cell_supports import _cell_supports_impl
from ._bspline_space_1d import _impl_class as _one_d_impl_class

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .._pantr_cpp import BsplineSpace32 as _CppSpace32
    from .._pantr_cpp import BsplineSpace64 as _CppSpace64
    from ..quad import PointsLattice
    from ._bspline_space_1d import BsplineSpace1D

    _Impl: TypeAlias = "_BsplineSpaceNDPython | _CppSpace32 | _CppSpace64"
    """The implementation a :class:`BsplineSpace` holds: the oracle, or a handle.

    Type-checking only, and the same alias :mod:`pantr.bspline._bspline_space_1d`
    declares for the univariate case: the three are unrelated nominal types that
    happen to offer the same surface, which is the port's whole claim.
    """

    _OneDImpl: TypeAlias = "Any"
    """One direction's implementation, as either backend holds it.

    Deliberately opaque. The univariate module owns the union of the three concrete
    types, and restating it here would be a second place to keep in step; nothing in
    this module does anything with a direction's implementation except hand it to
    the matching nD implementation.
    """


class _BsplineSpaceNDPython:
    """The pure-Python tensor-product B-spline space: the port's parity oracle.

    Holds one univariate *implementation* per direction, not one wrapper, which is
    what makes it the exact counterpart of ``pantr::bspline::BsplineSpace<T>``: that
    type holds ``BsplineSpace1D<T>`` handles, and this one holds whatever
    :class:`~pantr.bspline.BsplineSpace1D` selected for the same backend.

    Every derived quantity here is an ``O(dim)`` reduction over the directions, with
    ``dim`` at most 3 everywhere in the tree, and **none of them is memoised**.
    ``design/bspline_derived_caches.md`` F1 is what settles that: the seven
    ``functools.cached_property`` sites this class used to carry were memos only
    because a Python attribute read that walks three objects costs more than a
    ``__dict__`` hit, which is a fact about CPython rather than about the
    mathematics. Removing them is also what removes the defect that note records
    under its third prohibition -- ``domain`` handing out a cached array a caller
    could write through, corrupting it for the object's whole life.

    Attributes:
        _spaces (tuple[Any, ...]): One univariate implementation per direction, in
            axis order.
        _dtype (np.dtype[Any]): The storage format the directions share. Carried
            rather than derived, because it is the value the C++ counterpart carries
            as its template parameter; deriving it here would need a special case
            for a space with no directions.
    """

    __slots__ = ("_dtype", "_spaces")

    def __init__(
        self,
        spaces: Sequence[_OneDImpl],
        dtype: np.dtype[Any],
    ) -> None:
        """Hold the given univariate implementations, in axis order.

        Args:
            spaces (Sequence[Any]): One univariate implementation per direction.
                Already validated by :class:`BsplineSpace`, which owns every
                type-kind check.
            dtype (np.dtype[Any]): The storage format the directions share.
        """
        self._spaces = tuple(spaces)
        self._dtype = dtype

    @property
    def dim(self) -> int:
        """Get the number of directions.

        Returns:
            int: The dimension of the parametric domain.
        """
        return len(self._spaces)

    @property
    def spaces(self) -> tuple[_OneDImpl, ...]:
        """Get the univariate implementations, in axis order.

        Returns:
            tuple[Any, ...]: One implementation per direction.
        """
        return self._spaces

    @property
    def degrees(self) -> tuple[int, ...]:
        """Get the polynomial degree of each direction.

        Returns:
            tuple[int, ...]: The degree for each dimension.
        """
        return tuple(int(space.degree) for space in self._spaces)

    @property
    def tolerance(self) -> float:
        """Get the absolute tolerance used for parametric comparisons on this space.

        Returns:
            float: The largest of the directions' tolerances.

        Raises:
            ValueError: If the space has no directions.
        """
        if not self._spaces:
            # Stated rather than inherited from `max()`. CPython's own message for
            # an empty `max()` changed between 3.11 and 3.12, so leaving it to the
            # builtin would make the two backends disagree on one leg of the test
            # matrix and agree on the others -- a parity gap created by the standard
            # library. `space_nd.hpp` raises this same text.
            raise ValueError("tolerance: a B-spline space with no directions has no tolerance")
        return max(float(space.tolerance) for space in self._spaces)

    @property
    def num_basis(self) -> tuple[int, ...]:
        """Get the number of basis functions for each direction.

        Returns:
            tuple[int, ...]: The number of basis functions for each dimension.
        """
        return tuple(int(space.num_basis) for space in self._spaces)

    @property
    def num_total_basis(self) -> int:
        """Get the total number of basis functions.

        ``math.prod`` rather than ``numpy.prod``: the counts are Python integers and
        the product of three of them can exceed ``int64`` on a knot vector this
        machine can hold, which ``numpy.prod`` wraps silently and this does not. The
        C++ counterpart cannot follow -- signed overflow there is undefined -- and
        refuses instead, which is the one input on which the two disagree; see
        ``space_nd.hpp``.

        Returns:
            int: The product of :attr:`num_basis`; 1 for a space with no directions,
            which is the empty tensor product's own convention.
        """
        return math.prod(self.num_basis)

    @property
    def num_intervals(self) -> tuple[int, ...]:
        """Get the number of intervals for each direction.

        Returns:
            tuple[int, ...]: The number of intervals for each dimension.
        """
        return tuple(int(space.num_intervals) for space in self._spaces)

    @property
    def num_total_intervals(self) -> int:
        """Get the total number of cells.

        Returns:
            int: The product of :attr:`num_intervals`; 1 for a space with no
            directions.
        """
        return math.prod(self.num_intervals)

    @property
    def domain(self) -> npt.NDArray[np.float32 | np.float64]:
        """Get the per-direction domain.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Shape ``(dim, 2)``, row ``d``
            holding direction ``d``'s own domain ends.
        """
        domain = np.empty((self.dim, 2), dtype=self._dtype)
        for i, space in enumerate(self._spaces):
            domain[i, :] = space.domain
        return domain

    def has_Bezier_like_knots(self) -> bool:
        """Check whether every direction describes a single Bézier segment.

        Returns:
            bool: True if every direction has open ends and only one span; True for
            a space with no directions, which is what ``all(())`` gives.
        """
        return all(space.has_Bezier_like_knots() for space in self._spaces)


def _validated_spaces(spaces: Iterable[BsplineSpace1D]) -> tuple[BsplineSpace1D, ...]:
    """Refuse a collection of directions that cannot form a tensor-product space.

    The dtype check is this function's and cannot be either implementation's:
    ``BsplineSpace<T>`` can hold only ``BsplineSpace1D<T>``, so a mixed collection is
    not representable in C++ and there is nothing there to check. It is also the
    right side of the seam on its own terms -- a dtype is a type-kind fact, and
    ``pantr/core/error.hpp`` records that nanobind has no path producing a
    :class:`TypeError`, so the whole port keeps type-kind checks in the wrapper.

    Args:
        spaces (Iterable[BsplineSpace1D]): The directions, in axis order.

    Returns:
        tuple[BsplineSpace1D, ...]: The directions as a tuple.

    Raises:
        ValueError: If the directions do not all share one storage format.
    """
    validated = tuple(spaces)
    if any(space.dtype != validated[0].dtype for space in validated):
        raise ValueError("All B-spline spaces must have the same data type.")
    return validated


def _stored_dtype(spaces: tuple[BsplineSpace1D, ...]) -> np.dtype[Any]:
    """The storage format a tensor-product space over these directions would have.

    Args:
        spaces (tuple[BsplineSpace1D, ...]): The directions, already validated to
            share one dtype.

    Returns:
        np.dtype[Any]: That dtype, or ``float64`` when there are no directions. A
        space with no directions has nothing dtype-dependent reachable on it -- both
        :attr:`BsplineSpace.dtype` and :attr:`BsplineSpace.domain` raise
        :class:`IndexError` -- so the choice has no observable consequence, and
        making one here is what keeps the dimensionless case out of both
        implementations.
    """
    if not spaces:
        return np.dtype(np.float64)
    return np.dtype(spaces[0].dtype)


def _impl_class(dtype: np.dtype[Any]) -> type[_BsplineSpaceNDPython] | type[Any]:
    """The implementation class the active backend and the dtype select.

    The backend is per process rather than per instance, for the reason
    :func:`pantr.bspline._bspline_space_1d._impl_class` gives and which bites here
    first: two spaces built under different backends meeting in one
    :class:`BsplineSpace` is the concrete hazard that rule was written for, and
    :func:`_new_impl` refuses it rather than reconciling it.

    Args:
        dtype (np.dtype[Any]): The storage format the directions share.

    Returns:
        type: The oracle under the Python backend, and the C++ class for that
        storage format otherwise.

    Raises:
        RuntimeError: If the C++ backend is requested and is not available.
    """
    if active_backend() is Backend.PYTHON:
        return _BsplineSpaceNDPython
    if Backend.CPP not in available_backends():
        raise RuntimeError("the CPP backend is not available in this installation")
    from pantr import _pantr_cpp  # noqa: PLC0415  (optional, imported only when selected)

    if dtype == np.float32:
        return _pantr_cpp.BsplineSpace32
    return _pantr_cpp.BsplineSpace64


def _new_impl(
    spaces: tuple[BsplineSpace1D, ...],
    dtype: np.dtype[Any],
) -> _Impl:
    """Build a tensor-product space in whichever implementation the backend selects.

    Args:
        spaces (tuple[BsplineSpace1D, ...]): The directions, already validated by
            :func:`_validated_spaces`.
        dtype (np.dtype[Any]): The storage format they share.

    Returns:
        _Impl: The implementation object; an oracle instance or a C++ handle.

    Raises:
        ValueError: If any direction was built under a different backend.
        RuntimeError: If the C++ backend is requested and is not available.
    """
    cls = _impl_class(dtype)
    # Both implementations aggregate the directions' *implementations*, so a
    # direction built under the other backend cannot be aggregated -- and the two
    # ways that fails are both bad. Handing a Python oracle to a C++ class raises a
    # nanobind `TypeError` naming C++ types, which is loud but unreadable; handing a
    # C++ handle to the oracle SUCCEEDS and yields a hybrid whose reductions run in
    # Python over C++ values, which no parity claim covers and nothing announces.
    # `design/cross_backend_types.md` forbids exactly that second shape.
    one_d_cls = _one_d_impl_class(dtype)
    impls = []
    for direction, space in enumerate(spaces):
        impl = space._impl
        if not isinstance(impl, one_d_cls):
            raise ValueError(
                f"All B-spline spaces must come from the active backend; direction "
                f"{direction} was built under a different one."
            )
        impls.append(impl)
    if cls is _BsplineSpaceNDPython:
        return _BsplineSpaceNDPython(impls, dtype)
    # `cls` is one of the C++ classes here. The checker cannot narrow an identity
    # test on a `type[...]` union, so it still admits the oracle and then reports its
    # extra `dtype` parameter as missing; the univariate module's counterpart needs
    # no such step only because its two constructors happen to share an arity.
    cpp_cls: Any = cls
    return cast("_Impl", cpp_cls(impls))


class BsplineSpace:
    """A class representing a multi-dimensional B-spline space.

    This space is defined by a set of B-spline spaces, one for each dimension.

    This class provides methods to analyze B-spline properties, validate input
    parameters, compute various geometric characteristics of the spline,
    and access various properties of the B-spline.

    **This class is a wrapper.** The value -- the univariate spaces and every
    reduction over them -- is owned by an implementation chosen by
    :func:`_impl_class`, which is the C++ type
    (``cpp/include/pantr/bspline/space_nd.hpp``) or the oracle
    :class:`_BsplineSpaceNDPython`. The four operations below
    (:meth:`tabulate_basis`, :meth:`cell_supports`, :meth:`boundary_dofs`,
    :meth:`restrict`) are still Python over numba kernels and numpy, and are
    unchanged; only the state moved.

    Instances are immutable, and that is enforced rather than documented:
    ``__slots__`` means there is no ``__dict__`` to attach anything to, and
    :meth:`__setattr__` refuses even a rebinding of the two slots. The wrapper fills
    them through ``object.__setattr__``, which is the pattern
    ``design/bspline_derived_caches.md`` asks for and
    ``src/pantr/grid/_tensor_product_grid.py`` already ships.

    Attributes:
        _impl (_Impl): The implementation this wrapper holds; see
            :func:`_impl_class`.
        _spaces (tuple[BsplineSpace1D, ...]): The univariate wrappers this space was
            built from, so that ``space.spaces[0] is space_1d`` holds. It is a
            *presentation* memo, never a second truth about a value: the counts, the
            tolerance and the domain all come from :attr:`_impl` on every access.
    """

    __slots__ = ("_impl", "_spaces")

    _impl: _Impl
    """The implementation this wrapper holds; see :func:`_impl_class`."""

    _spaces: tuple[BsplineSpace1D, ...]
    """The univariate wrappers this space was built from; see the class docstring."""

    def __init__(
        self,
        spaces: Iterable[BsplineSpace1D],
    ) -> None:
        """Initialize a B-spline space object.

        Args:
            spaces (Iterable[BsplineSpace1D]): List of B-spline spaces, one for each
                dimension.

        Raises:
            ValueError: If the B-spline spaces have different data types, or if any
                of them was built under a different backend.
            RuntimeError: If the C++ backend is requested and is not available.
        """
        validated = _validated_spaces(spaces)
        impl = _new_impl(validated, _stored_dtype(validated))
        object.__setattr__(self, "_impl", impl)
        object.__setattr__(self, "_spaces", validated)

    def __setattr__(self, name: str, value: object) -> NoReturn:
        """Refuse to set an attribute, because a space is immutable.

        Args:
            name (str): The attribute a caller tried to set.
            value (object): The value it tried to set.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError(f"{type(self).__name__} is immutable; cannot set {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        """Refuse to delete an attribute, because a space is immutable.

        Args:
            name (str): The attribute a caller tried to delete.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError(f"{type(self).__name__} is immutable; cannot delete {name!r}")

    def __reduce__(
        self,
    ) -> tuple[type[BsplineSpace], tuple[tuple[BsplineSpace1D, ...]]]:
        """Pickle by the univariate wrappers rather than by implementation.

        The C++ handle is not picklable and must not become part of the wire format:
        a pickle written under the C++ backend has to load under the Python one and
        the other way round, or the backend switch would silently become a
        data-format switch.

        The directions go out as **wrappers**, not as their implementations, which is
        what carries their own ``__reduce__`` -- and with it the tolerance-drift
        bound ``design/bspline_pickle_tolerance.md`` derives for a univariate space --
        into this one's round trip. It is also what makes sharing survive a single
        pickle for free: ``pickle`` memoises, so dumping ``(space, space.spaces[0])``
        restores a pair that still satisfies ``space.spaces[0] is one_d``. Sharing
        does not survive two independent ``dumps`` calls, which is also true today.

        Returns:
            tuple: The class, and the univariate spaces to rebuild it from.
        """
        return (type(self), (self.spaces,))

    @property
    def dim(self) -> int:
        """Get the dimension of the B-spline space.

        Returns:
            int: The dimension of the B-spline space.
        """
        return int(self._impl.dim)

    @property
    def spaces(self) -> tuple[BsplineSpace1D, ...]:
        """Get the B-spline spaces.

        The objects a caller passed to the constructor, not copies and not
        re-wrappings of what the implementation holds, so
        ``space.spaces[0] is space_1d`` holds under both backends.
        ``design/bspline_ownership_lifetime.md`` F6 records why: no C++ object can
        supply the constructor argument's own Python object, and only the wrapper
        can, by keeping what it was built from.

        Returns:
            tuple[BsplineSpace1D, ...]: The B-spline spaces.
        """
        return self._spaces

    @property
    def degrees(self) -> tuple[int, ...]:
        """Get the polynomial degree of the B-spline.

        Returns:
            tuple[int, ...]: The degree for each dimension.
        """
        return tuple(self._impl.degrees)

    @property
    def tolerance(self) -> float:
        """Get the absolute tolerance used for parametric comparisons on this space.

        The largest of the univariate spaces' tolerances, each of which already
        carries its own direction's knot magnitude (see
        :attr:`~pantr.bspline.BsplineSpace1D.tolerance`). Taking the largest is the
        conservative choice when the directions are scaled differently: of the
        obvious reductions it is the only one that never under-states a gap in any
        direction.

        A *selection* rather than an arithmetic combination, so the result is one of
        the directions' own tolerances unmodified, and the two backends agree on it
        bit for bit.

        Returns:
            float: The absolute parametric tolerance.

        Raises:
            ValueError: If the space has no directions.
        """
        return float(self._impl.tolerance)

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the data type of the B-spline space.

        Read off the first direction rather than delegated, for the reason
        :attr:`~pantr.bspline.BsplineSpace1D.dtype` is: the C++ handle carries its
        storage format in its class name, not as a numpy dtype it could hand back.

        Returns:
            npt.DTypeLike: The numpy data type of the B-spline space.

        Raises:
            IndexError: If the space has no directions.
        """
        return self.spaces[0].dtype

    @property
    def num_basis(self) -> tuple[int, ...]:
        """Get the number of basis functions for each dimension.

        Returns:
            tuple[int, ...]: The number of basis functions for each dimension.
        """
        return tuple(self._impl.num_basis)

    @property
    def num_total_basis(self) -> int:
        """Get the total number of basis functions.

        Returns:
            int: The total number of basis functions; 1 for a space with no
            directions, which is the empty tensor product's own convention.
        """
        return int(self._impl.num_total_basis)

    @property
    def num_intervals(self) -> tuple[int, ...]:
        """Get the number of intervals for each dimension.

        Returns:
            tuple[int, ...]: The number of intervals for each dimension.
        """
        return tuple(self._impl.num_intervals)

    @property
    def num_total_intervals(self) -> int:
        """Get the total number of intervals.

        Returns:
            int: The total number of intervals; 1 for a space with no directions.
        """
        return int(self._impl.num_total_intervals)

    @property
    def domain(self) -> npt.NDArray[np.float32 | np.float64]:
        """Get the domain of the B-spline space.

        A fresh, writable array per call under both backends, and a copy of what the
        implementation owns rather than a view of it. That is what makes the two
        backends indistinguishable here: the C++ side hands out a read-only view of
        its own storage and the oracle builds an array, and a caller must not be able
        to tell which built the space.

        It also retires a defect. Until this port ``domain`` was a
        ``functools.cached_property`` handing out its cached array **unfrozen**, so
        ``d = space.domain; d[0, 0] = 999.0`` corrupted the cache for the object's
        whole life -- the third prohibition of ``design/bspline_derived_caches.md``,
        reproduced there. There is now no cache on either side to corrupt, so a write
        reaches the caller's own copy and nothing else sees it.

        ``space.domain is space.domain`` was ``True`` and is now ``False``. Nothing
        promised that identity and nothing in the suite relied on it; recorded
        because the sibling weakening on
        :attr:`~pantr.bspline.BsplineSpace1D.domain` is recorded, and one of the two
        going unsaid is how the pair becomes folklore.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The domain of the B-spline space.
            The shape is (dim, 2), where the last dimension contains the start
            and end values of the domain.

        Raises:
            IndexError: If the space has no directions.
        """
        # `dtype` first, and on its own line, because it is what raises on a space
        # with no directions -- the behaviour before the port, pinned by
        # `tests/test_bspline_space.py::TestBsplineSpaceEdgeCases`. Reading the
        # implementation instead would hand back an empty `(0, 2)` array.
        dtype = self.dtype
        return np.array(self._impl.domain, dtype=dtype)

    def has_Bezier_like_knots(self) -> bool:
        """Check if the knot vector represents a Bézier-like configuration.

        A Bézier-like configuration has open ends and only one non-zero span
        for each dimension.

        Returns:
            bool: True if knots have open ends and only one span.

        Example:
            >>> from pantr.bspline import BsplineSpace1D
            >>> bspline_1D = BsplineSpace1D([1, 1, 1, 3, 3, 3], 2)
            >>> bspline_2D = BsplineSpace([bspline_1D, bspline_1D])
            >>> bspline_2D.has_Bezier_like_knots()
            True
        """
        return bool(self._impl.has_Bezier_like_knots())

    def tabulate_basis(
        self,
        pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
        out_basis: npt.NDArray[np.float32 | np.float64] | None = None,
        out_first_basis: npt.NDArray[np.int_] | None = None,
    ) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
        """Tabulate the B-spline basis functions at the given points.

        Args:
            pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): The points
               at which to tabulate the basis functions.
               It can be a 2D array with shape (num_pts, dim) or a PointsLattice object.
            out_basis (npt.NDArray[np.float32 | np.float64] | None): Optional output array where the
                basis values will be stored. If None, a new array is allocated. Must have the
                correct shape and dtype if provided. This follows NumPy's style for output arrays.
                Defaults to None.
            out_first_basis (npt.NDArray[numpy.intp] | None): Optional output array where the
                first basis indices will be stored. If None, a new array is allocated. Must have
                the correct shape and dtype numpy.intp if provided. This follows NumPy's style for
                output arrays. Defaults to None.

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[numpy.intp]]: The basis
            function values and the first basis function indices.

            In the case pts is a 2D array, the shape of the basis function values array
            is (num_pts, order[0], order[1], ..., order[d-1]), where d is the dimension
            of the B-spline space and num_pts is the number of points.
            In the case pts is a PointsLattice object, the shape of the
            basis function values array is
            (num_pts_0, num_pts_1, ..., num_pts_d, order[0], order[1], ..., order[d-1]),
            where num_pts_i is the number of points in the i-th dimension.

            The shape of the first basis function indices array is (num_pts, dim),
            if pts is a 2D array, or (num_pts_0, num_pts_1, ..., num_pts_d, dim),
            if pts is a PointsLattice object.

            If `out_basis` or `out_first_basis` was provided, the corresponding element of the tuple
            is the same array.

        Raises:
            ValueError: If pts is not a 2D array or a PointsLattice object.
            ValueError: If the pts dimension does not match the dimension of the B-spline space.
            ValueError: If one or more points are outside the domain of the B-spline space, or if
                `out_basis` or `out_first_basis` is provided and has incorrect shape or dtype.
        """
        return _tabulate_Bspline_basis_impl(
            self, pts, out_basis=out_basis, out_first_basis=out_first_basis
        )

    def cell_supports(self, cell_ids: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Return the control points supported on each of the given cells.

        Every cell of a tensor-product space is supported by exactly
        ``prod(degree + 1)`` control points, contiguous per direction, so the result
        is a dense ``(n, W)`` table rather than a ragged structure. Column ``k``
        always means the same local offset: the one whose C-order index within
        ``degrees + 1`` is ``k``.

        Args:
            cell_ids (npt.ArrayLike): Flat knot-span cell ids in C-order over
                :attr:`num_intervals`, the same convention as
                :func:`pantr.grid.tensor_product_grid` and
                :class:`SpanwiseElementExtraction`. Each must satisfy
                ``0 <= cid < num_total_intervals``.

        Returns:
            npt.NDArray[np.int64]: Array of shape ``(n, prod(degree + 1))`` of flat
            C-order control-point ids. Empty input gives shape ``(0, W)``.

        Raises:
            ValueError: If any direction is periodic, ``cell_ids`` is not integral or
                one-dimensional, or an id is out of range.

        Example:
            >>> from pantr.bspline import BsplineSpace, BsplineSpace1D
            >>> first = BsplineSpace1D([0, 0, 0, 1, 2, 3, 3, 3], 2)
            >>> second = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
            >>> space = BsplineSpace([first, second])
            >>> space.num_basis
            (5, 3)
            >>> space.cell_supports([0]).tolist()
            [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
        """
        return _cell_supports_impl(self, cell_ids)

    def boundary_dofs(
        self,
        direction: int,
        side: int,
        layers: int = 1,
    ) -> npt.NDArray[np.int64]:
        """Return the control points of a boundary slab, as flat C-order ids.

        The slab holds the control points whose index along ``direction`` lies in the
        first (``side == 0``) or last (``side == 1``) ``layers`` positions, and every
        index on the remaining axes. For an open knot vector, ``layers == 1`` selects
        exactly the basis functions with non-zero trace on that face, which is what a
        strong Dirichlet condition needs; a thicker slab serves a clamped or
        :math:`C^1` condition.

        The result holds **control-point** ids over :attr:`num_basis`, never cell ids
        over :attr:`num_intervals`. Interior knot multiplicities do not affect it: a
        boundary slab is a notion in index space.

        Args:
            direction (int): Axis of the face, in ``[0, dim)``.
            side (int): ``0`` for the low face, ``1`` for the high face. Together with
                ``direction`` this is the ``lfid = 2 * direction + side`` encoding of
                :meth:`pantr.grid.Grid.local_facet_axis_side`.
            layers (int): Slab thickness in control-point indices, in
                ``[1, num_basis[direction]]``. Defaults to 1.

        Returns:
            npt.NDArray[np.int64]: Read-only, strictly ascending flat C-order
            control-point ids, of shape
            ``(layers * num_total_basis // num_basis[direction],)``.

        Raises:
            ValueError: If ``direction``, ``side`` or ``layers`` is out of range, or if
                any direction is periodic -- a periodic direction has no boundary.

        Example:
            >>> from pantr.bspline import BsplineSpace, BsplineSpace1D
            >>> first = BsplineSpace1D([0, 0, 0, 1, 2, 3, 3, 3], 2)
            >>> second = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
            >>> space = BsplineSpace([first, second])
            >>> space.num_basis
            (5, 4)
            >>> space.boundary_dofs(0, 0).tolist()
            [0, 1, 2, 3]
            >>> space.boundary_dofs(1, 0).tolist()
            [0, 4, 8, 12, 16]
            >>> space.boundary_dofs(1, 1, layers=2).tolist()
            [2, 3, 6, 7, 10, 11, 14, 15, 18, 19]
        """
        if any(space.periodic for space in self.spaces):
            raise ValueError("boundary_dofs: periodic B-spline spaces are not supported.")
        if not 0 <= direction < self.dim:
            raise ValueError(
                f"boundary_dofs: direction must lie in [0, {self.dim}); got {direction}."
            )
        if side not in (0, 1):
            raise ValueError(f"boundary_dofs: side must be 0 (low) or 1 (high); got {side}.")
        num_basis = self.num_basis
        num_basis_dir = num_basis[direction]
        if not 1 <= layers <= num_basis_dir:
            raise ValueError(
                f"boundary_dofs: layers must lie in [1, {num_basis_dir}]; got {layers}."
            )

        first = 0 if side == 0 else num_basis_dir - layers
        axes = [np.arange(n, dtype=np.int64) for n in num_basis]
        axes[direction] = np.arange(first, first + layers, dtype=np.int64)

        # Each axis range is ascending and the C-order flat id is monotone in the
        # lexicographic order of the multi-index, so the slab comes out sorted and
        # unique without an explicit sort.
        mesh = np.meshgrid(*axes, indexing="ij")
        dofs = np.ravel_multi_index(tuple(m.ravel() for m in mesh), num_basis).astype(np.int64)
        dofs.flags.writeable = False
        return dofs

    def restrict(self, cell_ids: npt.ArrayLike) -> BsplineSpaceRestriction:
        """Return the bounding-box windowed sub-space spanning ``cell_ids``.

        The window is the per-axis multi-index bounding box of the requested
        knot-span cells (flat ids in C-order over :attr:`num_intervals`, the same
        convention as :func:`pantr.grid.tensor_product_grid` and
        :class:`SpanwiseElementExtraction`). Each axis is windowed by slicing its
        knot vector (never re-clamped), so the windowed basis equals this space's
        basis pointwise over the windowed cells.

        Args:
            cell_ids (npt.ArrayLike): Flat knot-span cell ids to span; duplicates
                are ignored. Each must satisfy ``0 <= cid < num_total_intervals``.

        Returns:
            BsplineSpaceRestriction: The windowed :class:`BsplineSpace` and the
            read-only ``local_to_global_dof`` map of shape
            ``(windowed_space.num_total_basis,)``.

        Raises:
            ValueError: If ``cell_ids`` is empty or any axis is periodic.
            IndexError: If any cell id is out of range ``[0, num_total_intervals)``.
            TypeError: If ``cell_ids`` is not integer-valued.
        """
        spaces = self.spaces
        if any(space.periodic for space in spaces):
            raise ValueError("restrict: periodic B-spline spaces are not supported.")
        ids = np.asarray(cell_ids).ravel()
        if ids.size == 0:
            raise ValueError("restrict: cell_ids must be non-empty.")
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(f"restrict: cell_ids must be integer-valued; got dtype {ids.dtype}.")
        ids = ids.astype(np.int64, copy=False)
        n_int = self.num_total_intervals
        lo_id, hi_id = int(ids.min()), int(ids.max())
        if lo_id < 0 or hi_id >= n_int:
            raise IndexError(
                f"restrict: cell id out of range [0, {n_int}); got [{lo_id}, {hi_id}]."
            )

        multi = np.unravel_index(ids, self.num_intervals)
        windowed_1d: list[BsplineSpace1D] = []
        dof_axes: list[npt.NDArray[np.int64]] = []
        for d, space in enumerate(spaces):
            w_space, dof_d = space.restrict(int(multi[d].min()), int(multi[d].max()) + 1)
            windowed_1d.append(w_space)
            dof_axes.append(dof_d)

        mesh = np.meshgrid(*dof_axes, indexing="ij")
        local_to_global_dof = np.ravel_multi_index(
            tuple(m.ravel() for m in mesh), self.num_basis
        ).astype(np.int64)
        local_to_global_dof.flags.writeable = False
        return BsplineSpaceRestriction(BsplineSpace(windowed_1d), local_to_global_dof)


class BsplineSpaceRestriction(NamedTuple):
    """Result of :meth:`BsplineSpace.restrict`: a windowed space and its DOF map.

    - ``space`` -- the windowed :class:`BsplineSpace`: per axis a pure knot-vector
      slice of the parent (never re-clamped), so its basis equals the parent's
      pointwise over the windowed cells.
    - ``local_to_global_dof`` -- read-only, shape ``(space.num_total_basis,)``,
      mapping each windowed DOF (flat, C-order over the windowed per-axis basis
      counts) to its flat index in the parent space.

    Unlike :class:`pantr.grid.GridRestriction` there is no ``in_subset`` mask: every
    windowed DOF is a genuine parent DOF (a windowed space spans a box of cells, so
    there are no fill DOFs).

    ``space`` is a freshly built wrapper rather than a subobject of the space that
    produced it -- ``design/bspline_ownership_lifetime.md``'s class **V** -- so this
    stays a plain :class:`typing.NamedTuple` with nothing to decide about lifetime,
    and it pickles through the default protocol over its two picklable fields.
    """

    space: BsplineSpace
    local_to_global_dof: npt.NDArray[np.int64]
