"""Bezier geometric object: the Bezier class."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from numpy import typing as npt

from .._transform_control_points import _apply_affine_to_control_points
from ._bezier_collapse import _collapse_along_axis
from ._bezier_compose import _compose_bezier
from ._bezier_degree import _degree_elevate_bezier, _degree_reduce_bezier
from ._bezier_derivative import _derivative_bezier
from ._bezier_eval import _evaluate_bezier, _evaluate_bezier_deriv
from ._bezier_interpolate import _fit_bezier, _interpolate_bezier
from ._bezier_product import _multiply_bezier
from ._bezier_restrict import _restrict_bezier
from ._bezier_slice import _slice_bezier
from ._bezier_split import _split_bezier

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
    # Degree reduction
    # ------------------------------------------------------------------

    def reduce_degree(self, degree_decrements: int | Sequence[int]) -> Bezier:
        """Reduce the polynomial degree of the Bézier via least-squares approximation.

        Creates a new Bézier whose degree is lower by the requested amount in
        each parametric direction.  The reduction minimises the squared error
        under the Bernstein degree-elevation matrix using QR factorisation with
        Givens rotations.

        Unlike :meth:`elevate_degree`, this operation is **not exact** in
        general: the result is an approximation of the original mapping.

        Args:
            degree_decrements (int | Sequence[int]): Number of degrees to
                reduce. If an integer, the same decrement is applied to all
                parametric directions. If a sequence, must have length equal
                to ``self.dim``.

        Returns:
            Bezier: A new Bézier with reduced degrees.

        Raises:
            ValueError: If any degree decrement is negative.
            ValueError: If all degree decrements are zero.
            ValueError: If the number of decrements does not match the dimension.
            ValueError: If any decrement exceeds the current degree in that
                direction.
        """
        if isinstance(degree_decrements, int):
            decrements = (degree_decrements,) * self.dim
        else:
            decrements = tuple(degree_decrements)

        if len(decrements) != self.dim:
            raise ValueError(
                f"Number of degree decrements ({len(decrements)}) "
                f"must match dimension ({self.dim})."
            )

        if any(dec < 0 for dec in decrements):
            raise ValueError("Degree decrements must be non-negative.")

        if all(dec == 0 for dec in decrements):
            raise ValueError("At least one degree decrement must be positive.")

        for d, dec in enumerate(decrements):
            if dec > self.degree[d]:
                raise ValueError(
                    f"Degree decrement ({dec}) in direction {d} exceeds "
                    f"current degree ({self.degree[d]})."
                )

        return _degree_reduce_bezier(self, decrements)

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
    # Compose
    # ------------------------------------------------------------------

    def compose(self, inner: Bezier) -> Bezier:
        """Compose this Bézier with another: ``result(t) = self(inner(t))``.

        Computes the exact composition of two non-rational Bézier objects.
        The result is a new Bézier with parametric dimension equal to
        ``inner.dim``, rank equal to ``self.rank``, and degree
        ``sum(self.degree) * inner.degree[s]`` in each direction ``s``.

        Args:
            inner (Bezier): The inner Bézier (reparametrization). Must be
                non-rational and satisfy ``inner.rank == self.dim``.

        Returns:
            Bezier: A new Bézier representing ``self(inner(t))``.

        Raises:
            TypeError: If either Bézier is rational.
            ValueError: If ``inner.rank != self.dim``.
            ValueError: If the operands have different dtypes.

        Example:
            >>> f = Bezier(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]))
            >>> g = Bezier(np.array([[0.2], [0.8]]))
            >>> h = f.compose(g)
        """
        return _compose_bezier(self, inner)

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

        Uses two de Casteljau passes per direction for direct Bernstein
        coefficient computation without B-spline conversion. The order
        of the passes is chosen for numerical stability.

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
        if self.dim == 1:
            bounds_per_dim: list[tuple[float, float] | None] = [
                bounds  # type: ignore[list-item]
            ]
        else:
            seq = list(bounds)  # type: ignore[arg-type,unused-ignore]
            if len(seq) != self.dim:
                raise ValueError(
                    f"bounds sequence length ({len(seq)}) must match dim ({self.dim})."
                )
            bounds_per_dim = seq  # type: ignore[assignment]

        # Validate bounds.
        for i, b in enumerate(bounds_per_dim):
            if b is None:
                continue
            lower, upper = b
            if lower >= upper:
                raise ValueError(
                    f"Lower bound ({lower}) must be strictly less than upper bound ({upper}) "
                    f"in direction {i}."
                )
            if lower < 0.0 or upper > 1.0:
                raise ValueError(
                    f"Bounds ({lower}, {upper}) must lie within [0, 1] in direction {i}."
                )

        return _restrict_bezier(self, bounds_per_dim)

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def split(self, direction: int, value: float) -> tuple[Bezier, Bezier]:
        """Split the Bézier into two at a parameter value in one direction.

        Uses the de Casteljau algorithm to subdivide the Bézier into a
        left half (representing the original on ``[0, value]``) and a right
        half (representing the original on ``[value, 1]``), both
        reparametrized to ``[0, 1]``.

        Args:
            direction (int): Parametric direction along which to split.
                Must be in ``[0, dim)``.
            value (float): Parameter value at which to split.  Must lie
                strictly inside ``(0, 1)``.

        Returns:
            tuple[Bezier, Bezier]: A pair ``(left, right)`` of Béziers on
            ``[0, 1]^dim``.

        Raises:
            ValueError: If ``direction`` is out of range ``[0, dim)``.
            ValueError: If ``value`` is not strictly inside ``(0, 1)``.

        Example:
            >>> left, right = curve.split(0, 0.5)
            >>> left, right = surface.split(1, 0.3)
        """
        if direction < 0 or direction >= self.dim:
            raise ValueError(f"direction must be in [0, {self.dim}), got {direction}.")
        if value <= 0.0 or value >= 1.0:
            raise ValueError(f"value must be strictly inside (0, 1), got {value}.")

        return _split_bezier(self, direction, value)

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
    # Collapse along axis
    # ------------------------------------------------------------------

    def collapse_along_axis(
        self,
        axis: int,
        values: npt.ArrayLike,
    ) -> Bezier:
        """Collapse to a univariate Bézier along one parametric direction.

        Fixes all parametric directions except ``axis`` at the given parameter
        values, producing a 1D Bézier whose control points are the Bernstein
        coefficients along ``axis``.  This is a tensor contraction: for each
        collapsed direction, the Bernstein basis is evaluated at the given
        value and contracted with the control point array.

        Args:
            axis (int): Parametric direction to keep (0-indexed).
                Must be in ``[0, dim)``.
            values (npt.ArrayLike): Parameter values for all directions
                except ``axis``.  Must have length ``dim - 1`` with all
                values in ``[0, 1]``.  ``values[i]`` corresponds to
                direction ``i`` for ``i < axis``, and direction ``i + 1``
                for ``i >= axis``.

        Returns:
            Bezier: A 1D Bézier with degree ``self.degree[axis]`` and
            the same rank and rationality as the input.

        Raises:
            ValueError: If ``dim < 2`` (nothing to collapse).
            ValueError: If ``axis`` is out of range ``[0, dim)``.
            ValueError: If ``values`` does not have length ``dim - 1``.
            ValueError: If any value is outside ``[0, 1]``.

        Example:
            >>> # Collapse a 3D volume along axis 1 at (u=0.3, w=0.7)
            >>> curve = volume.collapse_along_axis(1, [0.3, 0.7])
        """
        if self.dim < 2:  # noqa: PLR2004
            raise ValueError("collapse_along_axis requires dim >= 2.")
        if axis < 0 or axis >= self.dim:
            raise ValueError(f"axis must be in [0, {self.dim}), got {axis}.")

        return _collapse_along_axis(self, axis, values)

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
    def interpolate(  # noqa: PLR0913
        cls,
        func: Callable[..., npt.ArrayLike],
        n_pts: int | Sequence[int],
        *,
        degree: int | Sequence[int] | None = None,
        nodes: (
            Literal["chebyshev", "uniform"]
            | npt.NDArray[np.floating[Any]]
            | Sequence[npt.NDArray[np.floating[Any]]]
            | None
        ) = None,
        dtype: npt.DTypeLike = np.float64,
        tol: float | None = None,
    ) -> Bezier:
        """Interpolate a callable function into a Bézier in Bernstein form.

        Evaluates ``func`` on a tensor-product grid of interpolation nodes
        and recovers the Bernstein coefficients via truncated SVD.

        The parametric dimension is determined by the length of ``n_pts``
        (when given as a sequence) or defaults to 1 (when a scalar ``int``).

        Args:
            func (Callable[..., npt.ArrayLike]): Function to interpolate.
                Called as ``func(x0, x1, ...)`` where each ``xi`` has shape
                ``(*grid_shape)`` (meshgrid broadcasting).  Must return an
                array of shape ``(*grid_shape)`` for a scalar-valued function
                or ``(*grid_shape, rank)`` for a vector-valued function.
            n_pts (int | Sequence[int]): Number of sample points per
                parametric direction.  A single ``int`` gives a 1D Bézier.
            degree (int | Sequence[int] | None): Polynomial degree per
                direction.  If *None* (default), ``degree = n_pts - 1``
                (exact interpolation).  If provided, must satisfy
                ``degree < n_pts`` in each direction; the result is a
                least-squares approximation.
            nodes: Interpolation node selection.

                - ``None`` or ``"chebyshev"`` (default): modified
                  Chebyshev-Lobatto nodes on [0, 1].
                - ``"uniform"``: equispaced nodes on [0, 1].
                - A 1D ``ndarray``: custom nodes broadcast to all directions.
                - A sequence of 1D ``ndarray`` values: per-direction custom nodes.
            dtype (npt.DTypeLike): Floating dtype. Defaults to ``float64``.
            tol (float | None): SVD truncation tolerance. If *None*, uses a
                default based on machine epsilon.

        Returns:
            Bezier: A non-rational Bézier whose evaluation approximates
            ``func``.

        Raises:
            ValueError: If ``n_pts`` values are < 1, *degree* >= *n_pts*,
                or *nodes* is inconsistent with *n_pts*.
            ValueError: If the callable returns an unexpected shape.

        Example:
            >>> import numpy as np
            >>> b = Bezier.interpolate(lambda x: x**2, 5)
            >>> b.degree
            (4,)
        """
        return _interpolate_bezier(func, n_pts, degree=degree, nodes=nodes, dtype=dtype, tol=tol)

    @classmethod
    def fit(
        cls,
        values: npt.ArrayLike,
        nodes: npt.NDArray[np.floating[Any]] | Sequence[npt.NDArray[np.floating[Any]]],
        *,
        degree: int | Sequence[int] | None = None,
        dtype: npt.DTypeLike = np.float64,
        tol: float | None = None,
    ) -> Bezier:
        """Construct a Bézier from pre-evaluated sample values at known nodes.

        Given function values on a tensor-product grid of nodes, recovers the
        Bernstein coefficients via truncated SVD.

        Args:
            values (npt.ArrayLike): Sample values on the tensor-product grid.
                Shape ``(*n_pts_per_dir)`` for scalar or
                ``(*n_pts_per_dir, rank)`` for vector-valued.
            nodes (npt.NDArray | Sequence[npt.NDArray]): Interpolation nodes.
                A single 1D array for 1D fitting, or a sequence of 1D arrays
                (one per parametric direction) for N-D.
            degree (int | Sequence[int] | None): Polynomial degree per
                direction.  If *None* (default), ``degree = n_pts - 1``
                (exact interpolation).  If provided, must satisfy
                ``degree < n_pts`` in each direction; the result is a
                least-squares approximation.
            dtype (npt.DTypeLike): Floating dtype. Defaults to ``float64``.
            tol (float | None): SVD truncation tolerance. If *None*, uses a
                default based on machine epsilon.

        Returns:
            Bezier: A non-rational Bézier.

        Raises:
            ValueError: If *nodes* lengths are inconsistent with *values*
                shape, or *degree* >= *n_pts* in any direction.

        Example:
            >>> import numpy as np
            >>> nodes = np.array([0.0, 0.5, 1.0])
            >>> vals = nodes**2
            >>> b = Bezier.fit(vals, nodes)
            >>> b.degree
            (2,)
        """
        return _fit_bezier(values, nodes, degree=degree, dtype=dtype, tol=tol)

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

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(
        self,
        *,
        color: str | None = None,
        show_control_polygon: bool = False,
        **plotter_kwargs: Any,  # noqa: ANN401
    ) -> object:
        """Quick interactive visualization of this Bézier (requires pyvista).

        For finer control, use ``pantr.viz.Scene`` directly.

        Args:
            color: Surface color.
            show_control_polygon: Render control polygon (points and wireframe).
            **plotter_kwargs: Additional keyword arguments for ``pv.Plotter()``.

        Returns:
            object: The pyvista ``Plotter`` after showing.

        Raises:
            ImportError: If pyvista is not installed.
        """
        from ..viz import plot as _plot  # noqa: PLC0415

        return _plot(
            self,
            color=color,
            show_control_polygon=show_control_polygon,
            **plotter_kwargs,
        )
