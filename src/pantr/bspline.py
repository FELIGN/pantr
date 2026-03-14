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

from ._bspline_eval import _evaluate_Bspline

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
        rk = self._control_points.ndim - self.dim
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
