"""BsplineSpace1D class for 1D B-spline spaces."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

from ._bspline_basis_core import _tabulate_Bspline_basis_1D_impl
from ._bspline_extraction import (
    _tabulate_Bspline_Bezier_1D_extraction_impl,
    _tabulate_Bspline_cardinal_1D_extraction_impl,
    _tabulate_Bspline_Lagrange_1D_extraction_impl,
)
from ._bspline_knots import (
    _get_Bspline_cardinal_intervals_1D_impl,
    _get_Bspline_num_basis_1D_impl,
    _get_unique_knots_and_multiplicity_impl,
)
from .tolerance import get_strict_tolerance

if TYPE_CHECKING:
    pass


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
    tol_value = dtype.type(tol)
    return _get_unique_knots_and_multiplicity_impl(knots, degree, tol_value, in_domain)


class BsplineSpace1D:
    """A class representing a 1D B-spline with configurable degree and knot vector.

    This class provides methods to analyze B-spline properties, validate input
    parameters, compute various geometric characteristics of the spline,
    and access various properties of the B-spline.

    Attributes:
        _tol (np.float32 | np.float64): Tolerance value for numerical comparisons.
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

        self._knots = np.asarray(knots)
        if np.issubdtype(self._knots.dtype, np.integer):
            self._knots = self._knots.astype(np.float64)

        self._tol = BsplineSpace1D._get_tolerance(self.dtype)

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

        dtype = knots.dtype
        tol = BsplineSpace1D._get_tolerance(dtype)

        if not isinstance(knots, np.ndarray) or knots.ndim != 1:
            raise TypeError("knots must be a 1D numpy array or Python list")

        if knots.dtype not in (np.float32, np.float64):
            raise ValueError("knots type must be float (32 or 64 bits)")

        if knots.size < (2 * degree + 2):
            raise ValueError("knots must have at least 2*degree+2 elements")

        if not np.all(np.diff(knots) >= 0):
            raise ValueError("knots must be non-decreasing")

        num_basis = _get_Bspline_num_basis_1D_impl(knots, degree, periodic, tol)
        if num_basis < (degree + 1):
            raise ValueError("Not enough knots for the specified degree")

    @staticmethod
    def _get_tolerance(dtype: npt.DTypeLike) -> float:
        """Create tolerance value based on data type.

        Right now, strict tolerance is used.

        Args:
            dtype (np.dtype): NumPy data type.

        Returns:
            float: Tolerance value appropriate for the given data type.
        """
        return float(get_strict_tolerance(dtype))

    def _snap_knots(self) -> None:
        """Snap knots within tolerance to avoid numerical precision issues.

        This method rounds knots to a precision determined by the stored tolerance
        and then averages any knots that are close together.

        It modifies the knot vector in place.
        """
        scale = 1.0 / self._tol
        rounded = np.round(self._knots * scale) / scale
        unique_vals = np.unique(rounded)

        snapped_knots = self._knots.copy()
        for val in unique_vals:
            mask = np.isclose(rounded, val, atol=0)
            snapped_knots[mask] = np.mean(self._knots[mask], dtype=self.dtype)
        self._knots = snapped_knots

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
        """Get the tolerance value used for numerical comparisons.

        Returns:
            float: The tolerance value.
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
            >>> bspline = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
            >>> bspline.get_num_intervals
            2
        """
        unique_knots, _ = self.get_unique_knots_and_multiplicity(in_domain=True)
        return len(unique_knots) - 1

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

    def has_left_end_open(self) -> bool:
        """Check if the left end of the B-spline is open.

        A left end is open if the first degree+1 knots are equal.

        Returns:
            bool: True if the left end is open, False otherwise.
        """
        if self.periodic:
            return False

        # Check if the first degree+1 knots are equal
        # (we know that they are non-decreasing).
        return bool(np.isclose(self._knots[0], self._knots[self._degree], atol=self._tol))

    def has_right_end_open(self) -> bool:
        """Check if the right end of the B-spline is open.

        A right end is open if the last degree+1 knots are equal.

        Returns:
            bool: True if the right end is open, False otherwise.
        """
        if self.periodic:
            return False

        # Check if the last degree+1 knots are equal
        # (we know that they are non-decreasing).
        return bool(np.isclose(self._knots[-self._degree - 1], self._knots[-1], atol=self._tol))

    def has_open_knots(self) -> bool:
        """Check if the B-spline has open ends.

        Returns:
            bool: True if both ends are open, False otherwise.
        """
        return self.has_left_end_open() and self.has_right_end_open()

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
        return (
            (not self._periodic) and self.has_open_knots() and self.num_basis == (self._degree + 1)
        )

    def get_cardinal_intervals(
        self, out: npt.NDArray[np.bool_] | None = None
    ) -> npt.NDArray[np.bool_]:
        """Get boolean array indicating whether the intervals (non-zero spans) are cardinal or not.

        An interval is cardinal if has the same length as the degree-1
        previous and the degree-1 next intervals.

        In the case of open knot vectors, this definition automatically
        discards the first degree-1 and the last degree-1 intervals.

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
            ValueError: If any evaluation points are outside the B-spline domain, or if `out_basis`
                or `out_first_basis` is provided and has incorrect shape or dtype.

        Example:
            >>> bspline = BsplineSpace1D([0, 0, 0, 0.25, 0.7, 0.7, 1, 1, 1], 2)
            >>> bspline.tabulate_basis([0.0, 0.5, 0.75, 1.0])
            (array([[1.        , 0.        , 0.        ],
                    [0.12698413, 0.5643739 , 0.30864198],
                    [0.69444444, 0.27777778, 0.02777778],
                    [0.        , 0.        , 1.        ]]),
             array([0, 1, 3, 3]))
        """
        return _tabulate_Bspline_basis_1D_impl(
            self, pts, out_basis=out_basis, out_first_basis=out_first_basis
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
        self, out: npt.NDArray[np.float32 | np.float64] | None = None
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Create Lagrange extraction operators of the B-spline.

        Args:
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
            self.knots, self.degree, self.tolerance, out=out
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
