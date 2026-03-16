"""B-spline geometric objects: the Bspline class and evaluation helpers.

This module provides :class:`Bspline`, which pairs a
:class:`~pantr.bspline_space_nd.BsplineSpace` with control points to represent a
parametric B-spline curve, surface, or volume. Evaluation at arbitrary points
is dispatched to the de Boor algorithm implemented in ``_bspline_eval``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

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
        n_deriv: int,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Evaluate B-spline derivatives up to order ``n_deriv`` at the given points.

        Computes all derivatives from order 0 (the value itself) through order
        ``n_deriv`` at each evaluation point. Derivatives of order higher than
        the polynomial degree are identically zero.

        For rational B-splines the quotient rule (Algorithm A4.2 from Piegl &
        Tiller, *The NURBS Book*) is applied so that the returned values are
        derivatives of the geometrically projected curve, not of the homogeneous
        representation.

        Args:
            pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): The
                parametric points at which to evaluate. For 1D B-splines, must
                be a 1D array of shape ``(n_pts,)`` or a 1D
                :class:`~pantr.quad.PointsLattice`. For multi-dimensional
                B-splines, must be a 2D array of shape ``(n_pts, dim)`` or a
                :class:`~pantr.quad.PointsLattice` with matching dimension.
                The dtype must match the B-spline dtype.
            n_deriv (int): Maximum derivative order to compute. Must be >= 0.
                Pass 0 to obtain just the values (equivalent to
                :meth:`evaluate`).
            out (npt.NDArray[np.float32 | np.float64] | None): Optional
                pre-allocated output array with the same shape and dtype as the
                returned array (see below). Filled in-place and returned.
                Defaults to None.

        Returns:
            npt.NDArray[np.float32 | np.float64]: For 1D B-splines, shape
            ``(n_pts, n_deriv+1)`` for scalar output or
            ``(n_pts, n_deriv+1, rank)`` for vector-valued output. For
            multi-dimensional B-splines, shape
            ``(*pts_shape, n_deriv+1, dim)`` for scalar output or
            ``(*pts_shape, n_deriv+1, dim, rank)`` for vector-valued output,
            where ``pts_shape`` is ``(n_pts,)`` for a points array or
            ``(*pts_grid_shape,)`` for a :class:`~pantr.quad.PointsLattice`.
            Axis ``-2`` indexes derivative order (0 = value, 1 = first
            derivative, …) and for multi-dimensional B-splines axis ``-1``
            (before the optional rank axis) indexes parametric direction.
            For rational B-splines the weight column is divided out and not
            included in the output.

        Raises:
            ValueError: If ``n_deriv < 0``, if the points dtype does not match
                the B-spline dtype, or if ``out`` has incorrect shape or dtype.
        """
        return _evaluate_Bspline_deriv(self, pts, n_deriv, out)
