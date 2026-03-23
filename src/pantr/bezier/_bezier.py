"""Bezier geometric object: the Bezier class."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from numpy import typing as npt

from .._transform_control_points import _apply_affine_to_control_points
from ._bezier_degree import _degree_elevate_bezier
from ._bezier_derivative import _derivative_bezier
from ._bezier_eval import _evaluate_bezier, _evaluate_bezier_deriv
from ._bezier_product import _multiply_bezier
from ._bezier_slice import _slice_bezier

if TYPE_CHECKING:
    from ..bspline import Bspline
    from ..quad import PointsLattice
    from ..transform import AffineTransform


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

    def derivative(self, direction: int = 0, *, keep_degree: bool = False) -> Bezier:
        """Return a Bézier representing the first derivative in the given direction.

        Computes the hodograph: a new Bézier whose value at every parametric
        point equals the partial derivative of this Bézier with respect to
        parametric direction ``direction``.

        For non-rational Bézier of degree ``p`` in direction ``d``, the
        result has degree ``p - 1`` (or ``p`` when ``keep_degree=True``).

        For rational Bézier, the quotient rule is applied, producing a
        rational Bézier of degree ``2p`` in direction ``d`` (or the original
        degree when ``keep_degree=True``).

        Args:
            direction (int): Parametric direction for differentiation.
                Must be in ``[0, dim)``. Defaults to 0.
            keep_degree (bool): If ``True``, the result preserves the same
                degree as the original Bézier by fusing derivative and degree
                elevation. This is useful, for instance, when computing
                derivatives of rational polynomials (in the numerator).
                Defaults to ``False``.

        Returns:
            Bezier: A new Bézier representing the derivative.

        Raises:
            ValueError: If ``direction`` is out of range ``[0, dim)``.
            ValueError: If the degree in the given direction is 0.

        Example:
            >>> f_prime = f.derivative()
            >>> df_dv = surface.derivative(direction=1)
            >>> f_prime_same_deg = f.derivative(keep_degree=True)
        """
        if direction < 0 or direction >= self.dim:
            raise ValueError(f"direction must be in [0, {self.dim}), got {direction}.")
        if self.degree[direction] < 1:
            raise ValueError("Derivative of a degree-0 Bézier is not defined.")
        return _derivative_bezier(self, direction, keep_degree=keep_degree)

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
            ValueError: If all degree increments are zero.
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
            raise ValueError("At least one degree increment must be positive.")

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
    # Reverse and permute
    # ------------------------------------------------------------------

    @overload
    def reverse(self, direction: int = ..., *, in_place: Literal[False] = ...) -> Bezier: ...

    @overload
    def reverse(self, direction: int = ..., *, in_place: Literal[True]) -> None: ...

    def reverse(self, direction: int = 0, *, in_place: bool = False) -> Bezier | None:
        """Reverse the orientation of one parametric direction.

        Flips the control points along the given parametric axis so that the
        mapping is reparametrised in the opposite sense along that direction.

        Args:
            direction (int): Parametric direction to reverse. Must be in
                ``[0, dim)``. Defaults to 0.
            in_place (bool): If ``True``, modify this Bézier in place and
                return ``None``. If ``False`` (default), return a new Bézier.

        Returns:
            Bezier | None: The reversed Bézier, or ``None`` when
            ``in_place=True``.

        Raises:
            ValueError: If ``direction`` is out of range ``[0, dim)``.

        Example:
            >>> rev = bezier.reverse(direction=0)
            >>> bezier.reverse(direction=1, in_place=True)
        """
        if direction < 0 or direction >= self.dim:
            raise ValueError(f"direction must be in [0, {self.dim}), got {direction}.")

        from .._control_points_utils import _reverse_control_points  # noqa: PLC0415

        new_cp = _reverse_control_points(self._control_points, direction, in_place=in_place)

        if in_place:
            return None
        return Bezier(new_cp, is_rational=self._is_rational)

    @overload
    def permute_directions(
        self, permutation: Sequence[int], *, in_place: Literal[False] = ...
    ) -> Bezier: ...

    @overload
    def permute_directions(
        self, permutation: Sequence[int], *, in_place: Literal[True]
    ) -> None: ...

    def permute_directions(
        self, permutation: Sequence[int], *, in_place: bool = False
    ) -> Bezier | None:
        """Reorder the parametric directions according to a permutation.

        Given a permutation ``[i_0, i_1, …]``, the new direction ``k`` is
        the old direction ``permutation[k]``. For example, ``[1, 2, 0]`` on
        a 3D volume maps old direction 1 → new 0, old 2 → new 1, old 0 → new 2.

        Args:
            permutation (Sequence[int]): A permutation of ``range(dim)``.
            in_place (bool): If ``True``, modify this Bézier in place and
                return ``None``. If ``False`` (default), return a new Bézier.

        Returns:
            Bezier | None: The permuted Bézier, or ``None`` when
            ``in_place=True``.

        Raises:
            ValueError: If ``permutation`` is not a valid permutation of
                ``range(dim)``.

        Example:
            >>> surface.permute_directions([1, 0])  # swap u ↔ v
        """
        perm = list(permutation)
        if sorted(perm) != list(range(self.dim)):
            raise ValueError(f"permutation must be a permutation of range({self.dim}), got {perm}.")

        from .._control_points_utils import _permute_control_points  # noqa: PLC0415

        new_cp = _permute_control_points(self._control_points, perm, self.dim, in_place=in_place)

        if in_place:
            self._control_points = new_cp
            return None
        return Bezier(new_cp, is_rational=self._is_rational)

    # ------------------------------------------------------------------
    # Affine transformation
    # ------------------------------------------------------------------

    @overload
    def transform(self, affine: AffineTransform, *, in_place: Literal[False] = ...) -> Bezier: ...

    @overload
    def transform(self, affine: AffineTransform, *, in_place: Literal[True]) -> None: ...

    def transform(
        self,
        affine: AffineTransform,
        *,
        in_place: bool = False,
    ) -> Bezier | None:
        """Apply an affine transformation to the control points.

        For non-rational Bézier, every control point ``P`` is mapped to
        ``A @ P + b``.  For rational Bézier the weighted homogeneous
        coordinates are updated so that the projected geometry undergoes the
        same affine map while the weights are preserved.

        Args:
            affine (~pantr.transform.AffineTransform): The affine
                transformation to apply.
            in_place (bool): If ``True``, the control points are modified in
                place and ``None`` is returned.  If ``False`` (default), a
                new :class:`Bezier` is returned.

        Returns:
            Bezier | None: The transformed Bézier, or ``None`` when
            ``in_place=True``.

        Raises:
            ValueError: If the transform dimension does not match the
                geometric rank of the Bézier.

        Example:
            >>> from pantr.transform import AffineTransform
            >>> T = AffineTransform.translation([1.0, 2.0])
            >>> shifted = bezier.transform(T)
        """
        new_cp = _apply_affine_to_control_points(
            self._control_points,
            self._is_rational,
            affine.matrix,
            affine.offset,
            in_place=in_place,
        )
        if in_place:
            return None
        return Bezier(new_cp, is_rational=self._is_rational)

    # ------------------------------------------------------------------
    # Restrict
    # ------------------------------------------------------------------

    def restrict(
        self,
        bounds: tuple[float, float] | Sequence[tuple[float, float] | None],
    ) -> Bezier:
        """Return a Bézier restricted to a sub-region of ``[0, 1]^dim``.

        Extracts the portion of the Bézier defined on the given sub-interval
        and reparametrizes the result back to ``[0, 1]^dim``.  The returned
        Bézier has the same degree but different control points that encode
        the restricted mapping.

        Internally converts to a B-spline, restricts, and converts back.

        Args:
            bounds (tuple[float, float] | Sequence[tuple[float, float] | None]):
                For a 1D Bézier, a ``(lower, upper)`` tuple within ``[0, 1]``.
                For multi-dimensional Bézier, a sequence of length ``dim``
                where each element is a ``(lower, upper)`` tuple for that
                direction, or ``None`` to keep the full ``[0, 1]`` range.
                At least one direction must have non-``None`` bounds that
                restrict the domain.

        Returns:
            Bezier: New Bézier on ``[0, 1]^dim`` representing the restriction.

        Raises:
            ValueError: If the sequence length does not match ``dim``.
            ValueError: If all directions are ``None`` or match the full domain.
            ValueError: If any bound lies outside ``[0, 1]``.
            ValueError: If ``lower >= upper`` in any direction.
        """
        bspline = self.to_bspline()
        restricted = bspline.restrict(bounds)
        return Bezier.from_bspline(restricted)

    # ------------------------------------------------------------------
    # Slice and boundary
    # ------------------------------------------------------------------

    def slice(self, axis: int, value: float) -> Bezier | npt.NDArray[np.float32 | np.float64]:
        """Slice the Bézier by fixing one parametric direction at a given value.

        Reduces the parametric dimension by one using the de Casteljau
        algorithm on the control points.  A surface becomes a curve, a curve
        becomes a point (returned as a NumPy array).

        At the boundary values ``0`` and ``1`` the result is obtained in
        O(1) by direct control point lookup.

        Args:
            axis (int): Parametric direction to fix (0-indexed).
                Must be in ``[0, dim)``.
            value (float): Parameter value at which to slice.  Must lie
                within ``[0, 1]``.

        Returns:
            Bezier | npt.NDArray[np.float32 | np.float64]:
            A Bézier with ``dim - 1`` dimensions when ``dim >= 2``,
            or a NumPy array of shape ``(rank,)`` when ``dim == 1``.
            Rational Béziers preserve the rational structure when ``dim >= 2``;
            for ``dim == 1`` the result is projected to physical coordinates.

        Raises:
            ValueError: If ``axis`` is out of range ``[0, dim)``.
            ValueError: If ``value`` is outside ``[0, 1]``.

        Example:
            >>> # Slice a surface at v=0.5 to get a curve
            >>> curve = surface.slice(1, 0.5)
            >>> # Composable: surface -> curve -> point
            >>> pt = surface.slice(1, 0.5).slice(0, 0.2)
        """
        if axis < 0 or axis >= self.dim:
            raise ValueError(f"axis must be in [0, {self.dim}), got {axis}.")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"value must be in [0, 1], got {value}.")

        return _slice_bezier(self, axis, value)

    def boundary(self, axis: int, side: int) -> Bezier | npt.NDArray[np.float32 | np.float64]:
        """Extract the boundary of the Bézier along one parametric direction.

        Returns the restriction of the Bézier to one end of the ``[0, 1]``
        domain in the given direction.

        Args:
            axis (int): Parametric direction (0-indexed).
                Must be in ``[0, dim)``.
            side (int): Which end of the domain: ``0`` for the start,
                ``1`` for the end.

        Returns:
            Bezier | npt.NDArray[np.float32 | np.float64]:
            A Bézier with ``dim - 1`` dimensions when ``dim >= 2``,
            or a NumPy array of shape ``(rank,)`` when ``dim == 1``.

        Raises:
            ValueError: If ``axis`` is out of range ``[0, dim)``.
            ValueError: If ``side`` is not 0 or 1.

        Example:
            >>> # Extract left boundary of a surface along direction 0
            >>> left_curve = surface.boundary(0, 0)
        """
        if side not in (0, 1):
            raise ValueError(f"side must be 0 or 1, got {side}.")
        if axis < 0 or axis >= self.dim:
            raise ValueError(f"axis must be in [0, {self.dim}), got {axis}.")

        value = 0.0 if side == 0 else 1.0
        return self.slice(axis, value)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_bspline(self, *, copy: bool = True) -> Bspline:
        """Convert to an equivalent B-spline with Bézier knot vectors.

        Creates a :class:`~pantr.bspline.Bspline` with open knot vectors
        ``[0]*(p+1) + [1]*(p+1)`` in each parametric direction.

        Args:
            copy (bool): If ``True`` (default), the control points are
                deep-copied into the new B-spline. If ``False``, the
                B-spline shares the same underlying control point array.

        Returns:
            ~pantr.bspline.Bspline: Equivalent B-spline representation.
        """
        from ..bspline import Bspline as BsplineCls  # noqa: PLC0415
        from ..bspline import BsplineSpace, BsplineSpace1D  # noqa: PLC0415

        dtype = self.dtype
        spaces: list[BsplineSpace1D] = []
        for p in self.degree:
            knots = np.zeros(2 * (p + 1), dtype=dtype)
            knots[p + 1 :] = 1.0
            spaces.append(BsplineSpace1D(knots, p))

        cp = self._control_points.copy() if copy else self._control_points
        return BsplineCls(BsplineSpace(spaces), cp, self._is_rational)

    @classmethod
    def from_bspline(cls, bspline: Bspline, *, copy: bool = True) -> Bezier:
        """Create a Bézier from a B-spline with Bézier-like knot vectors.

        Validates that the B-spline has Bézier-like knots (open knots with
        ``num_basis == degree + 1`` in each direction) and extracts the
        control points.

        Args:
            bspline (~pantr.bspline.Bspline): A B-spline with Bézier-like
                knot structure.
            copy (bool): If ``True`` (default), the control points are
                deep-copied into the new Bézier. If ``False``, the Bézier
                shares the same underlying control point array.

        Returns:
            Bezier: The equivalent Bézier.

        Raises:
            ValueError: If the B-spline does not have Bézier-like knots.
        """
        if not bspline.space.has_Bezier_like_knots():
            raise ValueError("B-spline does not have Bézier-like knots. Cannot convert to Bézier.")
        cp = bspline.control_points.copy() if copy else bspline.control_points
        return cls(cp, is_rational=bspline.is_rational)
