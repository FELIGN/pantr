"""Bézier geometric objects: the Bezier class.

This module provides :class:`Bezier`, which stores control points to represent
a parametric Bézier curve, surface, or volume. Degree is derived from the
control point array shape. Evaluation and manipulation methods use direct
Bernstein algorithms implemented in ``_bezier_core``, ``_bezier_eval``,
``_bezier_derivative``, ``_bezier_degree``, and ``_bezier_product``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

from ._bezier_degree import _degree_elevate_bezier
from ._bezier_derivative import _derivative_bezier
from ._bezier_eval import _evaluate_bezier, _evaluate_bezier_deriv
from ._bezier_product import _multiply_bezier

if TYPE_CHECKING:
    from .bspline import Bspline
    from .quad import PointsLattice


class Bezier:
    """A parametric Bézier curve, surface, or volume defined by control points.

    Stores only control points and an ``is_rational`` flag. The polynomial
    degree in each parametric direction is inferred from the control point
    array shape: ``degree[d] = control_points.shape[d] - 1``.

    Attributes:
        _control_points (npt.NDArray[np.float32 | np.float64]): Control point
            array with shape ``(*degrees_plus_1, rank)``, where the last axis
            is the output rank (including the homogeneous weight for rational).
        _is_rational (bool): Whether the Bézier is rational (last control
            point coordinate is a homogeneous weight).
    """

    def __init__(
        self,
        control_points: npt.ArrayLike,
        is_rational: bool = False,
    ) -> None:
        """Initialize a Bézier from control points.

        Args:
            control_points (npt.ArrayLike): Control points. Shape must be at
                least 2D: ``(*degrees_plus_1, rank)``. A 1D input of shape
                ``(n,)`` is reshaped to ``(n, 1)`` (scalar field). Integer
                arrays are cast to ``float64``.
            is_rational (bool): Whether the Bézier is rational. Defaults to
                False.

        Raises:
            ValueError: If the control points have fewer than 1 entry in any
                parametric direction.
            ValueError: If the Bézier has rank smaller than 1.
        """
        cp = np.asarray(control_points)
        if np.issubdtype(cp.dtype, np.integer):
            cp = cp.astype(np.float64)
        if cp.ndim < 1:
            raise ValueError("Control points must be at least 1D.")
        if cp.ndim == 1:
            cp = cp[:, np.newaxis]

        if cp.ndim < 2:  # pragma: no cover - guarded by the reshape above  # noqa: PLR2004
            raise ValueError("Control points must be at least 2D after reshape.")

        for d in range(cp.ndim - 1):
            if cp.shape[d] < 1:
                raise ValueError(
                    f"Control points must have at least 1 entry in parametric "
                    f"direction {d}, got {cp.shape[d]}."
                )

        self._control_points: npt.NDArray[np.float32 | np.float64] = cp
        self._is_rational = is_rational

        if self.rank <= 0:
            raise ValueError(f"The Bézier must have at least rank one. Got rank {self.rank}.")

    @property
    def dim(self) -> int:
        """Get the parametric dimension of the Bézier.

        Returns:
            int: Number of parametric dimensions.
        """
        return self._control_points.ndim - 1

    @property
    def degree(self) -> tuple[int, ...]:
        """Get the polynomial degrees per parametric direction.

        Returns:
            tuple[int, ...]: Polynomial degree for each parametric dimension,
            computed as ``shape[d] - 1``.
        """
        return tuple(s - 1 for s in self._control_points.shape[:-1])

    @property
    def control_points(self) -> npt.NDArray[np.float32 | np.float64]:
        """Get the control points of the Bézier.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Control point array with
            shape ``(*degrees_plus_1, rank)``.
        """
        return self._control_points

    @property
    def is_rational(self) -> bool:
        """Check whether the Bézier is rational.

        Returns:
            bool: True if the Bézier is rational (i.e., the last control point
            coordinate is a homogeneous weight), False otherwise.
        """
        return self._is_rational

    @property
    def rank(self) -> int:
        """Get the output rank of the Bézier.

        The rank is the number of value dimensions produced by the mapping.
        For a scalar field it is 1; for a 3D curve it is 3. For rational
        Béziers the weight coordinate is excluded.

        Returns:
            int: Output rank of the Bézier.
        """
        rk = int(self._control_points.shape[-1])
        return rk - 1 if self.is_rational else rk

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the floating-point dtype of the Bézier.

        Returns:
            npt.DTypeLike: The numpy dtype (float32 or float64) of the control
            point array.
        """
        return self._control_points.dtype

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Evaluate the Bézier at the given parametric points.

        Args:
            pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): The
                parametric points at which to evaluate. For 1D Bézier, must be
                a 1D array of shape ``(n_pts,)``. For multi-dimensional Bézier,
                must be a 2D array of shape ``(n_pts, dim)`` or a
                :class:`~pantr.quad.PointsLattice`.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array. Defaults to None.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Bézier values at the given
            points.

        Raises:
            ValueError: If the points dtype does not match the Bézier dtype,
                or if ``out`` has incorrect shape or dtype.
        """
        return _evaluate_bezier(self, pts, out)

    def evaluate_derivatives(
        self,
        pts: npt.NDArray[np.float32 | np.float64] | PointsLattice,
        orders: int | Sequence[int],
        out: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Evaluate a specific partial derivative of the Bézier.

        Computes the single partial derivative specified by ``orders``,
        where ``orders[d]`` is the derivative order in parametric direction
        ``d``. For rational Bézier the generalised quotient rule is applied.

        Args:
            pts (npt.NDArray[np.float32 | np.float64] | PointsLattice): The
                parametric points at which to evaluate.
            orders (int | Sequence[int]): Derivative order(s). A single
                ``int`` is broadcast to all ``self.dim`` directions. A sequence
                must contain one non-negative integer per parametric direction.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional
                pre-allocated output array. Defaults to None.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Mixed partial derivative
            values.

        Raises:
            ValueError: If ``len(orders) != self.dim``, if any order is
                negative, or if the points dtype does not match the Bézier dtype.

        Example:
            >>> result = bezier.evaluate_derivatives(pts, 2)
            >>> result = bezier.evaluate_derivatives(pts, [1, 2])
        """
        orders_seq: Sequence[int] = [orders] * self.dim if isinstance(orders, int) else orders
        return _evaluate_bezier_deriv(self, pts, orders_seq, out)

    # ------------------------------------------------------------------
    # Derivative (returns new Bezier)
    # ------------------------------------------------------------------

    def derivative(self, direction: int = 0) -> Bezier:
        """Return a Bézier representing the first derivative in the given direction.

        Computes the hodograph: a new Bézier whose value at every parametric
        point equals the partial derivative of this Bézier with respect to
        parametric direction ``direction``.

        For non-rational Bézier of degree ``p`` in direction ``d``, the
        result has degree ``p - 1``.

        For rational Bézier, the quotient rule is applied, producing a
        rational Bézier of degree ``2p`` in direction ``d``.

        Args:
            direction (int): Parametric direction for differentiation.
                Must be in ``[0, dim)``. Defaults to 0.

        Returns:
            Bezier: A new Bézier representing the derivative.

        Raises:
            ValueError: If ``direction`` is out of range ``[0, dim)``.
            ValueError: If the degree in the given direction is 0.

        Example:
            >>> f_prime = f.derivative()
            >>> df_dv = surface.derivative(direction=1)
        """
        if direction < 0 or direction >= self.dim:
            raise ValueError(f"direction must be in [0, {self.dim}), got {direction}.")
        if self.degree[direction] < 1:
            raise ValueError("Derivative of a degree-0 Bézier is not defined.")
        return _derivative_bezier(self, direction)

    # ------------------------------------------------------------------
    # Degree elevation
    # ------------------------------------------------------------------

    def elevate_degree(self, degree_increments: int | Sequence[int]) -> Bezier:
        """Elevate the polynomial degree of the Bézier.

        Creates a new Bézier that represents the same mapping as the original
        but with higher polynomial degree.

        Args:
            degree_increments (int | Sequence[int]): Number of degrees to
                increase. If an integer, the same increment is applied to all
                parametric directions. If a sequence, must have length equal
                to ``self.dim``.

        Returns:
            Bezier: A new Bézier with elevated degrees.

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

        if all(inc == 0 for inc in increments):
            return self

        return _degree_elevate_bezier(self, increments)

    # ------------------------------------------------------------------
    # Multiply
    # ------------------------------------------------------------------

    def multiply(self, other: Bezier) -> Bezier:
        """Return the exact pointwise product of this Bézier and another.

        Given Bézier ``self`` and ``other`` over the same parametric domain
        ``[0, 1]^dim``, returns a new Bézier ``h`` such that
        ``h(t) = self(t) * other(t)``. The result has degree ``p_d + q_d``
        per direction.

        Args:
            other (Bezier): The second Bézier operand. Must have the same
                dimension, dtype, and rank as ``self``.

        Returns:
            Bezier: A new Bézier representing ``self * other``.

        Raises:
            ValueError: If the operands have different dimensions, dtypes,
                or ranks.

        Example:
            >>> h = f.multiply(g)
            >>> h2 = f * g
        """
        return _multiply_bezier(self, other)

    __mul__ = multiply

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_bspline(self) -> Bspline:
        """Convert to an equivalent B-spline with Bézier knot vectors.

        Creates a :class:`~pantr.bspline.Bspline` with open knot vectors
        ``[0]*(p+1) + [1]*(p+1)`` in each parametric direction.

        Returns:
            ~pantr.bspline.Bspline: Equivalent B-spline representation.
        """
        from .bspline import Bspline as BsplineCls  # noqa: PLC0415
        from .bspline_space_1D import BsplineSpace1D  # noqa: PLC0415
        from .bspline_space_nd import BsplineSpace  # noqa: PLC0415

        dtype = self.dtype
        spaces: list[BsplineSpace1D] = []
        for p in self.degree:
            knots = np.zeros(2 * (p + 1), dtype=dtype)
            knots[p + 1 :] = 1.0
            spaces.append(BsplineSpace1D(knots, p))

        return BsplineCls(BsplineSpace(spaces), self._control_points, self._is_rational)

    @classmethod
    def from_bspline(cls, bspline: Bspline) -> Bezier:
        """Create a Bézier from a B-spline with Bézier-like knot vectors.

        Validates that the B-spline has Bézier-like knots (open knots with
        ``num_basis == degree + 1`` in each direction) and extracts the
        control points.

        Args:
            bspline (~pantr.bspline.Bspline): A B-spline with Bézier-like
                knot structure.

        Returns:
            Bezier: The equivalent Bézier.

        Raises:
            ValueError: If the B-spline does not have Bézier-like knots.
        """
        if not bspline.space.has_Bezier_like_knots():
            raise ValueError("B-spline does not have Bézier-like knots. Cannot convert to Bézier.")
        return cls(bspline.control_points, is_rational=bspline.is_rational)
