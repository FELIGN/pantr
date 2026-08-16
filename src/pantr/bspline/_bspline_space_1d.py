"""1D B-spline space: knot vector, degree, and basis evaluation.

This module exposes :class:`BsplineSpace1D`, which combines a knot vector and
polynomial degree to define a 1D B-spline space. It provides methods for
basis evaluation, Bézier/Lagrange/cardinal extraction operators, and several
geometric properties (domain, intervals, cardinality).
"""

from __future__ import annotations

import functools

import numpy as np
from numpy import typing as npt

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
    _get_Bspline_cardinal_intervals_1D_impl,
    _get_Bspline_num_basis_1D_impl,
    _get_unique_knots_and_multiplicity_impl,
    _knot_tolerance,
)


@functools.lru_cache(maxsize=128)
def _cached_unique_knots_and_multiplicity(
    knots_repr: tuple[bytes, str, int],
    degree: int,
    tol: float,
    in_domain: bool = False,
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
    """Compute unique knots and multiplicities with memoization.

    The cache is bounded to prevent unbounded memory growth in long-running
    applications that construct many distinct spline spaces.  128 entries is
    generous for real-world use (an adaptive simulation rarely needs more than
    a few dozen distinct knot vectors) while still eliminating the risk of a
    memory leak.

    Args:
        knots_repr (tuple[bytes, str, int]): Serialized knot vector bytes, dtype string, and size.
        degree (int): B-spline degree.
        tol (float): Tolerance value.
        in_domain (bool): Whether to restrict to the spline domain.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
            Unique knots and corresponding multiplicities.
    """
    knots_bytes, dtype_str, size = knots_repr
    dtype = np.dtype(dtype_str)
    knots = np.frombuffer(knots_bytes, dtype=dtype, count=size).copy()
    # ``tol`` goes in unrounded, exactly as ``BsplineSpace1D._snap_knots`` passes it, so
    # the space and its own accessor apply one predicate rather than two that differ in
    # the last bits of the threshold.
    unique, mults = _get_unique_knots_and_multiplicity_impl(knots, degree, tol, in_domain)
    # The same arrays are returned to every caller; freeze them so a caller
    # mutation cannot poison the cache.
    unique.flags.writeable = False
    mults.flags.writeable = False
    return unique, mults


class BsplineSpace1D:
    """A class representing a 1D B-spline with configurable degree and knot vector.

    This class provides methods to analyze B-spline properties, validate input
    parameters, compute various geometric characteristics of the spline,
    and access various properties of the B-spline.

    Attributes:
        _tol (float): Absolute tolerance for parametric comparisons on this space,
            in the units of its knots. Derived once at construction by
            ``_bspline_knots._knot_tolerance`` as
            ``8 * eps(dtype) * max(span, |knots[0]|, |knots[-1]|)``, so it tracks
            the knot vector's own magnitude instead of being a per-dtype constant.
        _knots (npt.NDArray[np.float32 | np.float64]): Knot vector defining the B-spline.
        _degree (int): Polynomial degree of the B-spline.
        _periodic (bool): Whether the B-spline is periodic.
    """

    _tol: float
    _knots: npt.NDArray[np.float32 | np.float64]
    _degree: int
    _periodic: bool

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
            ValueError: If degree is negative, knots are insufficient, or
                knots are not non-decreasing.
            TypeError: If knots cannot be converted to a numpy array.
        """
        BsplineSpace1D._validate_input(knots, degree, periodic)

        knots_arr = np.asarray(knots)
        if np.issubdtype(knots_arr.dtype, np.integer):
            knots_arr = knots_arr.astype(np.float64)
        else:
            # Always own the storage: the vector is frozen read-only below, and
            # freezing a caller-supplied array in place would mutate caller state.
            knots_arr = knots_arr.copy()
        self._knots = knots_arr

        # Derived from the knots, not from the dtype alone: the presets in
        # ``pantr.tolerance`` are dimensionless, and the magnitude a parametric
        # comparison on this space is relative to is the knot vector's own extent.
        self._tol = _knot_tolerance(self._knots)

        self._degree = int(degree)
        self._periodic = bool(periodic)

        if snap_knots:
            self._snap_knots()

        self._knots.flags.writeable = False

    @staticmethod
    def _validate_input(
        knots: npt.ArrayLike,
        degree: int,
        periodic: bool = False,
    ) -> None:
        """Validate the B-spline input parameters.

        Args:
            knots (npt.ArrayLike): Knot vector to validate.
            degree (int): Degree to validate.
            periodic (bool): Whether the B-spline is periodic.

        Raises:
            ValueError: If degree is negative, knots are insufficient, or
                knots are not non-decreasing.
            TypeError: If knots cannot be converted to a numpy array.
        """
        if degree < 0:
            raise ValueError("degree must be non-negative")

        if isinstance(knots, list):
            knots = np.array(knots)
        elif not isinstance(knots, np.ndarray):
            raise TypeError("knots must be a 1D numpy array or Python list")

        if np.issubdtype(knots.dtype, np.integer):
            knots = knots.astype(np.float64)

        if not isinstance(knots, np.ndarray) or knots.ndim != 1:
            raise TypeError("knots must be a 1D numpy array or Python list")

        if knots.dtype not in (np.float32, np.float64):
            raise ValueError("knots type must be float (32 or 64 bits)")

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
    def knots(self) -> npt.NDArray[np.float32 | np.float64]:
        """Get the knot vector.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The knot vector.
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

        Returns:
            float: The absolute parametric tolerance.
        """
        return self._tol

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the data type of the knot vector (and used in computations).

        Returns:
            npt.DTypeLike: The numpy data type of the knots.
        """
        return self._knots.dtype

    @functools.cached_property
    def num_basis(self) -> int:
        """Get the number of basis functions.

        This depends on the knot vector length and the degree, but
        also on whether the B-spline is periodic.

        Returns:
            int: Number of basis functions.
        """
        return _get_Bspline_num_basis_1D_impl(self._knots, self._degree, self._periodic, self._tol)

    def get_unique_knots_and_multiplicity(
        self,
        in_domain: bool = False,
    ) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.int_]]:
        """Get unique knots and their multiplicities.

        Args:
            in_domain (bool): If True, only consider knots in the domain.
                Defaults to False.

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[numpy.intp]]: Tuple of
            (unique_knots, multiplicities) where unique_knots contains the distinct knot values
            and multiplicities contains the corresponding multiplicity counts.
        """
        knots_repr = (self._knots.tobytes(), self._knots.dtype.str, int(self._knots.size))
        degree = int(self._degree)
        tol = float(self._tol)
        return _cached_unique_knots_and_multiplicity(knots_repr, degree, tol, in_domain)

    @functools.cached_property
    def num_intervals(self) -> int:
        """Get the number of intervals in the domain.

        Returns:
            int: Number of intervals.

        Example:
            >>> space = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
            >>> space.num_intervals
            2
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

        Entry ``j`` is the smallest global function index ``i`` whose function is
        non-zero on the open interval between in-domain unique knots ``j`` and
        ``j + 1``. The functions non-zero there are exactly ``i`` through
        ``i + degree``, so this one index describes the whole support of the interval.

        The result is cached and read-only: repeated calls return the same array.

        Returns:
            npt.NDArray[np.int64]: Read-only ``int64`` array of shape
            ``(num_intervals,)``, non-decreasing, whose successive differences are
            the interior knot multiplicities.

        Raises:
            ValueError: If the space is periodic.

        Example:
            >>> space = BsplineSpace1D([0, 0, 0, 1, 2, 2, 3, 3, 3], 2)
            >>> space.first_basis_per_interval()
            array([0, 1, 3])
        """
        if self._periodic:
            raise ValueError(
                "first_basis_per_interval: periodic B-spline spaces are not supported."
            )
        return self._first_basis_per_interval_cached

    def _get_domain_indices(self) -> tuple[int, int]:
        """Get the domain boundary indices of the knot vector.

        I.e., the indices of the knot vector that define the domain.

        Returns:
            tuple[int, int]: Tuple of (start_index, end_index) defining the domain.
        """
        return (self._degree, self._knots.size - self._degree - 1)

    @functools.cached_property
    def domain(self) -> tuple[np.float32 | np.float64, np.float32 | np.float64]:
        """Get the knot vector domain.

        Returns:
            tuple[np.float32 | np.float64, np.float64]: Tuple of
            (start_value, end_value) defining the domain.

        Example:
            >>> bspline = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
            >>> bspline.domain
            (0.0, 2.0)
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

        A left end is open if the first degree+1 knots are equal.

        Returns:
            bool: True if the left end is open, False otherwise.
        """
        return self._left_end_open

    def has_right_end_open(self) -> bool:
        """Check if the right end of the B-spline is open.

        A right end is open if the last degree+1 knots are equal.

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

        A Bézier-like configuration has open ends and only one non-zero span.

        Returns:
            bool: True if knots have open ends and only one span.

        Example:
            >>> bspline = BsplineSpace1D([1, 1, 1, 3, 3, 3], 2)
            >>> bspline.has_Bezier_like_knots()
            True
        """
        return self._bezier_like_knots

    def get_cardinal_intervals(
        self, out: npt.NDArray[np.bool_] | None = None
    ) -> npt.NDArray[np.bool_]:
        """Get boolean array indicating whether the intervals (non-zero spans) are cardinal or not.

        An interval is cardinal if has the same length as the degree-1
        previous and the degree-1 next intervals.

        In the case of open knot vectors, this definition automatically
        discards the first degree-1 and the last degree-1 intervals.

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
            >>> bspline = BsplineSpace1D([0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6], 2)
            >>> bspline.get_cardinal_intervals()
            array([False, False, True, True, False, False])

            >>> bspline = BsplineSpace1D([0, 0, 0, 1, 2, 3, 4, 5, 5, 6, 6, 6], 2)
            >>> bspline.get_cardinal_intervals()
            array([False, False, True, False, False, False])

            >>> bspline = BsplineSpace1D([0, 1, 2, 3, 4, 5, 6, 7, 10], 3)
            >>> bspline.get_cardinal_intervals()
            array([True, False])
        """
        return _get_Bspline_cardinal_intervals_1D_impl(
            self._knots, self._degree, self._tol, out=out
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
            >>> bspline = BsplineSpace1D([0, 0, 0, 0.25, 0.7, 0.7, 1, 1, 1], 2)
            >>> bspline.tabulate_basis([0.0, 0.5, 0.75, 1.0])
            (array([[1.        , 0.        , 0.        ],
                    [0.12698413, 0.5643739 , 0.30864198],
                    [0.69444444, 0.27777778, 0.02777778],
                    [0.        , 0.        , 1.        ]]),
             array([0, 1, 3, 3]))

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
            >>> bspline = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
            >>> d, first = bspline.tabulate_basis_derivatives([0.5], n_deriv=1)
            >>> d.shape
            (1, 2, 3)
            >>> d[0, 1, :]   # first derivatives at x=0.5: B0'=-1, B1'=0, B2'=1
            array([-1.,  0.,  1.])
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
        merged = _compute_inserted_knot_vector_1d(self._knots, self._degree, arr, self._tol)
        return BsplineSpace1D(merged, self._degree)

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
        eff_regularity = self._degree - 1 if regularity is None else regularity
        if eff_regularity < -1 or eff_regularity > self._degree - 1:
            raise ValueError(
                f"regularity must be in [-1, degree - 1] = [-1, {self._degree - 1}], "
                f"got {eff_regularity}"
            )
        new_knots = _compute_uniform_subdivision_knots(
            self._knots, self._degree, self._tol, n_subdivisions, eff_regularity
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
        if self._periodic:
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
            dtype=self._knots.dtype,
        )
        _, first_basis = self.tabulate_basis(mids)
        j_lo = int(first_basis[0])
        j_hi = int(first_basis[1]) + self._degree
        windowed_knots = self._knots[j_lo : j_hi + self._degree + 2]
        windowed = BsplineSpace1D(windowed_knots, self._degree, periodic=False, snap_knots=False)
        local_to_global_dof = np.arange(j_lo, j_hi + 1, dtype=np.int64)
        local_to_global_dof.flags.writeable = False
        return windowed, local_to_global_dof
