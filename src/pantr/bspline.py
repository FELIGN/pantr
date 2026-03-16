"""B-spline geometric objects: the Bspline class and evaluation helpers.

This module provides :class:`Bspline`, which pairs a
:class:`~pantr.bspline_space_nd.BsplineSpace` with control points to represent a
parametric B-spline curve, surface, or volume. Evaluation at arbitrary points
is dispatched to the de Boor algorithm implemented in ``_bspline_eval``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

from ._bspline_degree import _degree_elevate_bspline
from ._bspline_eval import _evaluate_Bspline, _evaluate_Bspline_deriv

if TYPE_CHECKING:
    from .bspline_space_nd import BsplineSpace
    from .quad import PointsLattice


class Bspline:
    """A parametric B-spline curve/surface defined by a space and control points.

    Combines a :class:`~pantr.bspline_space_nd.BsplineSpace` (knot vectors, degrees)
    with a set of control points to represent a B-spline mapping.

    Attributes:
        _space (pantr.bspline_space_nd.BsplineSpace): The multi-dimensional
            B-spline space.
        _control_points (npt.NDArray[np.float32 | np.float64]): Control point
            array reshaped to ``(*num_basis, rank)``.
        _is_rational (bool): Whether the B-spline is rational (NURBS).
    """

    def __init__(
        self, space: BsplineSpace, control_points: npt.ArrayLike, is_rational: bool = False
    ) -> None:
        """Initialize a B-spline.

        Args:
            space (~pantr.bspline_space_nd.BsplineSpace): The B-spline space.
            control_points (npt.ArrayLike): The control points.
            is_rational (bool): Whether the B-spline is rational.

        Raises:
            ValueError: If the number of control points is not a multiple
                of the number of basis functions.
            ValueError: If the B-spline has rank smaller than 1.
        """
        self._space = space

        control_points = np.asarray(control_points)
        num_basis = space.num_total_basis
        if control_points.size % num_basis != 0:
            raise ValueError(
                f"The number of control points must be a multiple of the number of basis functions."
                f"Got {control_points.size} control points and {num_basis} basis functions."
            )

        self._control_points = control_points.reshape([*space.num_basis, -1])

        if self._control_points.dtype != self._space.dtype:
            raise ValueError(
                f"The control points must have the same dtype as the B-spline space."
                f"Got {self._control_points.dtype} control points and {self._space.dtype} "
                "B-spline space."
            )

        self._is_rational = is_rational

        if self.rank <= 0:
            raise ValueError(f"The B-spline must have at least rank one. Got rank {self.rank}")

    @property
    def dim(self) -> int:
        """Get the geometric dimension of the B-spline.

        Returns:
            int: Number of parametric dimensions (equals the dimension of the
            underlying B-spline space).
        """
        return self._space.dim

    @property
    def degree(self) -> tuple[int, ...]:
        """Get the B-spline degrees per parametric direction.

        Returns:
            tuple[int, ...]: Polynomial degree for each parametric dimension.
        """
        return self._space.degrees

    @property
    def space(self) -> BsplineSpace:
        """Get the underlying B-spline space.

        Returns:
            ~pantr.bspline_space_nd.BsplineSpace: The multi-dimensional
            B-spline space defining the knot vectors and polynomial degrees.
        """
        return self._space

    @property
    def control_points(self) -> npt.NDArray[np.float32 | np.float64]:
        """Get the control points of the B-spline.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Control point array with
            shape ``(*num_basis, rank)``.
        """
        return self._control_points

    @property
    def is_rational(self) -> bool:
        """Check whether the B-spline is rational (NURBS).

        Returns:
            bool: True if the B-spline is rational (i.e., the last control point
            coordinate is a homogeneous weight), False otherwise.
        """
        return self._is_rational

    @property
    def rank(self) -> int:
        """Get the output rank of the B-spline.

        The rank is the number of value dimensions produced by the mapping.
        For a scalar field it is 1; for a 3D curve it is 3.  For rational
        B-splines the weight coordinate is excluded.

        Returns:
            int: Output rank of the B-spline.
        """
        rk = int(self._control_points.shape[-1])
        return rk - 1 if self.is_rational else rk

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the floating-point dtype of the B-spline.

        Returns:
            npt.DTypeLike: The numpy dtype (float32 or float64) of the control
            point array.
        """
        return self._control_points.dtype

    def evaluate(
        self,
        pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Evaluate the B-spline at the given points.

        Args:
            pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): The
                parametric points at which to evaluate the B-spline.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array where the result will be stored. If None, a new array is
                allocated. This follows NumPy's style for output arrays.
                Defaults to None.

        Returns:
            npt.NDArray[np.float32 | np.float64]: B-spline values at the given
            points.

        Raises:
            ValueError: If the points dtype does not match the B-spline dtype,
                or if `out` has incorrect shape or dtype.
        """
        return _evaluate_Bspline(self, pts, out)

    def evaluate_derivatives(
        self,
        pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
        orders: int | Sequence[int],
        out: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Evaluate a specific partial derivative of the B-spline.

        Computes the single partial derivative specified by ``orders``,
        where ``orders[d]`` is the derivative order in parametric direction ``d``.
        For rational B-splines the generalised quotient rule is applied so that
        the returned values are derivatives of the projected mapping.

        Args:
            pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): The
                parametric points at which to evaluate. For 1D B-splines, must
                be a 1D array of shape ``(n_pts,)`` or a 1D
                :class:`~pantr.quad.PointsLattice`. For multi-dimensional
                B-splines, must be a 2D array of shape ``(n_pts, dim)`` or a
                :class:`~pantr.quad.PointsLattice` with matching dimension.
                The dtype must match the B-spline dtype.
            orders (int | Sequence[int]): Derivative order(s). A single
                ``int`` is broadcast to all ``self.dim`` directions. A sequence
                must contain one non-negative integer per parametric direction
                (``len(orders) == self.dim``). Pass ``0`` (or ``[0, ..., 0]``)
                to obtain the function value (equivalent to :meth:`evaluate`).
            out (npt.NDArray[np.float32 | np.float64] | None): Optional
                pre-allocated output array with the same shape and dtype as the
                returned array (see below). Filled in-place and returned.
                Defaults to None.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Mixed partial derivative
            values. Shape is ``(*pts_base_shape,)`` for scalar output or
            ``(*pts_base_shape, rank)`` for vector-valued output, where
            ``pts_base_shape`` is ``(n_pts,)`` for a points array or
            ``(*pts_grid_shape,)`` for a :class:`~pantr.quad.PointsLattice`.
            For rational B-splines the weight column is divided out and not
            included in the output.

        Raises:
            ValueError: If ``len(orders) != self.dim``, if any order is
                negative, if the points dtype does not match the B-spline dtype,
                or if ``out`` has incorrect shape or dtype.

        Example:
            >>> # 1D: second derivative (int shorthand)
            >>> result = spline.evaluate_derivatives(pts, 2)
            >>> # 1D: second derivative (sequence form)
            >>> result = spline.evaluate_derivatives(pts, [2])
            >>> # 2D: partial derivative ∂³f/∂u ∂v²
            >>> result = spline.evaluate_derivatives(pts, [1, 2])
        """
        orders_seq: Sequence[int] = [orders] * self.dim if isinstance(orders, int) else orders
        return _evaluate_Bspline_deriv(self, pts, orders_seq, out)

    def elevate_degree(self, degree_increments: int | Sequence[int]) -> Bspline:
        """Elevate the polynomial degree of the B-spline.

        Creates a new B-spline that represents the same mapping as the original
        one but with higher-order polynomial basis functions. This is achieved
        by increasing the degree in each parametric direction and adjusting the
        control points and knot vectors accordingly.

        Args:
            degree_increments (int | Sequence[int]): Number of degrees to
                increase. If an integer, the same increment is applied to all
                parametric directions. If a sequence, must have length equal
                to the B-spline dimension.

        Returns:
            Bspline: A new B-spline with elevated degrees.

        Raises:
            ValueError: If any degree increment is negative.
            ValueError: If the number of increments does not match the dimension.
        """
        if isinstance(degree_increments, int):
            increments = (degree_increments,) * self.dim
        else:
            increments = tuple(degree_increments)

        if len(increments) != self.dim:
            raise ValueError(
                f"Number of degree increments ({len(increments)}) "
                f"must match dimension ({self.dim})."
            )

        if any(inc < 0 for inc in increments):
            raise ValueError("Degree increments must be non-negative.")

        # If all increments are zero, return self
        if all(inc == 0 for inc in increments):
            return self

        return _degree_elevate_bspline(self, increments)
