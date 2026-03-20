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
from ._bspline_knot_insertion import (
    _compute_uniform_subdivision_knots,
    _insert_knots_bspline,
    _to_open_bspline_impl,
)

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

    def insert_knots(
        self,
        new_knots: npt.ArrayLike | Sequence[npt.ArrayLike | None],
    ) -> Bspline:
        """Return a geometrically equivalent B-spline with additional knots inserted.

        Args:
            new_knots (npt.ArrayLike | Sequence[npt.ArrayLike | None]):
                For a 1D B-spline, a flat non-empty 1D array-like of knot values
                to insert.  For multi-dimensional B-splines, a sequence of length
                ``dim`` where each element is a 1D array-like of knots to insert
                in that direction, or ``None`` to skip that direction.  At least
                one direction must have a non-empty array of knots to insert.
                Repeated values in an array insert the same knot multiple times.

        Returns:
            Bspline: New B-spline with the same geometry and refined knot vectors.

        Raises:
            ValueError: If the sequence length does not match ``dim`` (multi-dim case).
            ValueError: If all directions have empty or ``None`` knot arrays.
            ValueError: If any knot lies outside its direction's domain.
            ValueError: If any insertion would exceed maximum multiplicity.
        """
        dtype = self.dtype

        if self.dim == 1:
            arr = np.asarray(new_knots, dtype=dtype)
            new_knots_per_dim: list[npt.NDArray[np.float32 | np.float64] | None] = [arr]
        else:
            seq = list(new_knots)  # type: ignore[arg-type]
            if len(seq) != self.dim:
                raise ValueError(
                    f"new_knots sequence length ({len(seq)}) must match dim ({self.dim})."
                )
            new_knots_per_dim = [None if nk is None else np.asarray(nk, dtype=dtype) for nk in seq]

        # Require at least one non-empty direction.
        if all(nk is None or nk.size == 0 for nk in new_knots_per_dim):
            raise ValueError(
                "At least one direction must have a non-empty array of knots to insert."
            )

        return _insert_knots_bspline(self, new_knots_per_dim)

    def to_open_bspline(self) -> Bspline:
        """Return an open (clamped) non-periodic B-spline equivalent to this one.

        Converts each parametric direction to an open representation by inserting knots
        at the domain boundaries until each has multiplicity ``degree + 1``, then
        trimming any ghost knots outside the domain.  Works for 1D and multi-dimensional
        B-splines, and handles periodic, unclamped non-periodic, and mixed cases.

        For periodic splines the ``n_full = len(knots) - degree - 1`` control points are
        reconstructed by modulo-wrapping the ``n_periodic`` stored control points
        (``ctrl_full[i] = ctrl[i % n_periodic]``), and the Oslo algorithm is applied to
        this full set. The resulting open B-spline represents the mathematical periodic
        function defined by the periodic knot vector and the wrapped control points.

        Returns:
            Bspline: Open, non-periodic B-spline with clamped knot vectors.

        Raises:
            ValueError: If the B-spline is already open in every direction.
        """
        return _to_open_bspline_impl(self)

    def multiply(self, other: Bspline) -> Bspline:
        """Return the exact pointwise product of this B-spline and another.

        Given B-splines ``self`` and ``other`` over the same parametric
        domain, returns a new B-spline ``h`` such that ``h(t) = self(t) *
        other(t)`` for all ``t`` in the domain.  The result lives in the product
        space of degree ``p_d + q_d`` per direction where ``p_d`` and ``q_d``
        are the degrees of the two operands in direction *d*.

        Both non-rational and rational (NURBS) operands are supported.  A
        non-rational operand is promoted to rational (unit weights) when the
        other is rational.

        Args:
            other (Bspline): The second B-spline operand. Must have the
                same dimension, dtype, rank, and parametric domain as ``self``.

        Returns:
            Bspline: A new B-spline representing ``self * other``.

        Raises:
            ValueError: If the operands have different dimensions.
            ValueError: If the operands have different dtypes.
            ValueError: If the operands have different ranks.
            ValueError: If the operands have different parametric domains.

        Note:
            The boundary structure of the operands is preserved per direction:
            both periodic → periodic, both non-open → non-open, either open → open.

        Example:
            >>> h = f.multiply(g)
            >>> h2 = f * g  # equivalent via __mul__
        """
        if self.dim == 1:
            from ._bspline_product import _multiply_bspline_1d  # noqa: PLC0415

            return _multiply_bspline_1d(self, other)
        from ._bspline_product_nd import _multiply_bspline_nd  # noqa: PLC0415

        return _multiply_bspline_nd(self, other)

    __mul__ = multiply

    def subdivide(
        self,
        n_subdivisions: int | Sequence[int | None],
        regularity: int | None = None,
    ) -> Bspline:
        """Return a geometrically equivalent B-spline with uniformly refined knot vectors.

        For every non-zero knot span in each active parametric direction,
        inserts ``n_subdivisions - 1`` uniformly spaced knot values.  Each
        value is repeated ``degree - regularity`` times so that the B-spline
        has ``C^regularity`` continuity at every inserted knot.

        Args:
            n_subdivisions (int | Sequence[int | None]): Number of equal
                sub-spans per existing interval.  A single ``int`` is applied
                to all directions; must be >= 2.  A sequence of length ``dim``
                provides per-direction counts; use ``None`` to skip a direction.
                At least one direction must have a count >= 2.
            regularity (int | None): Continuity order at every inserted knot.
                Applied uniformly across all active directions.  Must be in
                ``[-1, degree - 1]`` for each active direction.  ``None``
                (default) uses ``degree - 1`` per direction.

        Returns:
            Bspline: New B-spline with refined knot vectors and same geometry.

        Raises:
            ValueError: If the sequence length does not match ``dim``.
            ValueError: If any subdivision count is < 1.
            ValueError: If no direction has a count >= 2.
            ValueError: If ``regularity`` is outside the valid range for any
                active direction.
        """
        if isinstance(n_subdivisions, int):
            counts: list[int | None] = [n_subdivisions] * self.dim
        else:
            counts = list(n_subdivisions)
            if len(counts) != self.dim:
                raise ValueError(
                    f"n_subdivisions sequence length ({len(counts)}) must match dim ({self.dim})."
                )

        # Validate counts.
        for c in counts:
            if c is not None and c < 1:
                raise ValueError(f"n_subdivisions must be >= 1, got {c}")

        # Require at least one active direction with count >= 2.
        if all(c is None or c == 1 for c in counts):
            raise ValueError("At least one direction must have n_subdivisions >= 2.")

        # Validate regularity per active direction and compute per-direction new knots.
        dtype = self.dtype
        new_knots_per_dim: list[npt.NDArray[np.float32 | np.float64] | None] = []
        for i, c in enumerate(counts):
            if c is None or c == 1:
                new_knots_per_dim.append(None)
            else:
                space_1d = self.space.spaces[i]
                deg = space_1d.degree
                eff_regularity = deg - 1 if regularity is None else regularity
                if eff_regularity < -1 or eff_regularity > deg - 1:
                    raise ValueError(
                        f"regularity must be in [-1, degree - 1] = [-1, {deg - 1}] "
                        f"for direction {i}, got {eff_regularity}"
                    )
                nk = _compute_uniform_subdivision_knots(
                    space_1d.knots, space_1d.degree, space_1d.tolerance, c, eff_regularity
                ).astype(dtype, copy=False)
                new_knots_per_dim.append(nk)

        return _insert_knots_bspline(self, new_knots_per_dim)
