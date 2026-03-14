"""Multi-dimensional B-spline spaces using tensor products.

This module defines :class:`BsplineSpace`, which aggregates multiple
:class:`BsplineSpace1D` objects to represent multi-dimensional parameter
domains. It handles tensor-product basis evaluation by combining the 1D
components.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

from ._bspline_basis_multidim import _tabulate_Bspline_basis_impl

if TYPE_CHECKING:
    from .bspline_space_1D import BsplineSpace1D
    from .quad import PointsLattice


class BsplineSpace:
    """A class representing a multi-dimensional B-spline space.

    This space is defined by a set of B-spline spaces, one for each dimension.

    This class provides methods to analyze B-spline properties, validate input
    parameters, compute various geometric characteristics of the spline,
    and access various properties of the B-spline.

    Attributes:
        _spaces (Iterable[BsplineSpace1D]): List of B-spline spaces, one for each dimension.
    """

    _spaces: tuple[BsplineSpace1D, ...]

    def __init__(
        self,
        spaces: Iterable[BsplineSpace1D],
    ) -> None:
        """Initialize a B-spline space object.

        Args:
            spaces (Iterable[BsplineSpace1D]): List of B-spline spaces, one for each dimension.

        Raises:
            ValueError: If the B-spline spaces have different data types.
        """
        self._spaces = tuple(spaces)

        if not all(space.dtype == self.dtype for space in self._spaces):
            raise ValueError("All B-spline spaces must have the same data type.")

    @property
    def dim(self) -> int:
        """Get the dimension of the B-spline space.

        Returns:
            int: The dimension of the B-spline space.
        """
        return len(self._spaces)

    @property
    def spaces(self) -> tuple[BsplineSpace1D, ...]:
        """Get the B-spline spaces.

        Returns:
            tuple[BsplineSpace1D, ...]: The B-spline spaces.
        """
        return self._spaces

    @functools.cached_property
    def degrees(self) -> tuple[int, ...]:
        """Get the polynomial degree of the B-spline.

        Returns:
            tuple[int, ...]: The degree for each dimension.
        """
        return tuple(space.degree for space in self._spaces)

    @functools.cached_property
    def tolerance(self) -> float:
        """Get the tolerance value used for numerical comparisons.

        It is the maximum tolerance of the B-spline spaces.

        Returns:
            float: The tolerance value.
        """
        return max(space.tolerance for space in self._spaces)

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the data type of the B-spline space.

        Returns:
            npt.DTypeLike: The numpy data type of the B-spline space.
        """
        return self._spaces[0].dtype

    @functools.cached_property
    def num_basis(self) -> tuple[int, ...]:
        """Get the number of basis functions for each dimension.

        Returns:
            tuple[int, ...]: The number of basis functions for each dimension.
        """
        return tuple(space.num_basis for space in self._spaces)

    @functools.cached_property
    def num_total_basis(self) -> int:
        """Get the total number of basis functions.

        Returns:
            int: The total number of basis functions.
        """
        return int(np.prod(self.num_basis))

    @functools.cached_property
    def num_intervals(self) -> tuple[int, ...]:
        """Get the number of intervals for each dimension.

        Returns:
            tuple[int, ...]: The number of intervals for each dimension.
        """
        return tuple(space.num_intervals for space in self._spaces)

    @functools.cached_property
    def num_total_intervals(self) -> int:
        """Get the total number of intervals.

        Returns:
            int: The total number of intervals.
        """
        return int(np.prod(self.num_intervals))

    @functools.cached_property
    def domain(self) -> npt.NDArray[np.float32 | np.float64]:
        """Get the domain of the B-spline space.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The domain of the B-spline space.
            The shape is (dim, 2), where the last dimension contains the start
            and end values of the domain.
        """
        domain = np.empty((self.dim, 2), dtype=self.dtype)
        for i, space in enumerate(self._spaces):
            domain[i, :] = space.domain
        return domain

    def has_Bezier_like_knots(self) -> bool:
        """Check if the knot vector represents a Bézier-like configuration.

        A Bézier-like configuration has open ends and only one non-zero span
        for each dimension.

        Returns:
            bool: True if knots have open ends and only one span.

        Example:
            >>> bspline_1D = BsplineSpace1D([1, 1, 1, 3, 3, 3], 2)
            >>> bspline_2D = BsplineSpace([bspline_1D, bspline_1D])
            >>> bspline_2D.has_Bezier_like_knots()
            True
        """
        return all(space.has_Bezier_like_knots() for space in self._spaces)

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
