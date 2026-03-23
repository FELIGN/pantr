"""B-spline geometric objects: the Bspline class and evaluation helpers.

This module provides :class:`Bspline`, which pairs a
:class:`~pantr.bspline.BsplineSpace` with control points to represent a
parametric B-spline curve, surface, or volume. Evaluation at arbitrary points
is dispatched to the de Boor algorithm implemented in ``_bspline_eval``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from numpy import typing as npt

from .._transform_control_points import _apply_affine_to_control_points
from ._bspline_degree import _degree_elevate_bspline
from ._bspline_derivative import _derivative_bspline
from ._bspline_eval import _evaluate_Bspline, _evaluate_Bspline_deriv
from ._bspline_knot_insertion import (
    _compute_uniform_subdivision_knots,
    _insert_knots_bspline,
    _to_open_bspline_impl,
    _to_periodic_bspline_impl,
)
from ._bspline_knot_removal import _remove_knots_bspline
from ._bspline_restrict import _restrict_bspline_impl
from ._bspline_slice import _slice_bspline
from ._bspline_split import _split_bspline_impl
from ._bspline_to_beziers import _to_beziers_impl

if TYPE_CHECKING:
    from ..bezier import Bezier
    from ..quad import PointsLattice
    from ..transform import AffineTransform
    from ._bspline_space_nd import BsplineSpace


class Bspline:
    """A parametric B-spline curve/surface defined by a space and control points.

    Combines a :class:`~pantr.bspline.BsplineSpace` (knot vectors, degrees)
    with a set of control points to represent a B-spline mapping.

    Attributes:
        _space (pantr.bspline.BsplineSpace): The multi-dimensional
            B-spline space.
        _control_points (npt.NDArray[np.float32 | np.float64]): Control point
            array reshaped to ``(*num_basis, rank)``.
        _is_rational (bool): Whether the B-spline is rational (NURBS).
        _beziers_cache (``npt.NDArray[np.object_] | None``): Cached Bézier
            decomposition, or ``None`` if not yet computed.
    """

    def __init__(
        self, space: BsplineSpace, control_points: npt.ArrayLike, is_rational: bool = False
    ) -> None:
        """Initialize a B-spline.

        Args:
            space (~pantr.bspline.BsplineSpace): The B-spline space.
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
        self._beziers_cache: npt.NDArray[np.object_] | None = None

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
            ~pantr.bspline.BsplineSpace: The multi-dimensional
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

    def derivative(self, direction: int = 0, *, keep_degree: bool = False) -> Bspline:
        """Return a B-spline representing the first derivative in the given direction.

        Computes the hodograph: a new B-spline whose value at every parametric
        point equals the partial derivative of this B-spline with respect to
        parametric direction ``direction``.

        For non-rational B-splines of degree ``p`` in direction ``d``, the
        result has degree ``p - 1`` in direction ``d`` and the same degree in
        all other directions (or ``p`` when ``keep_degree=True``).

        For rational B-splines (NURBS), the quotient rule is applied, producing
        a rational B-spline of degree ``2p`` in direction ``d`` (or the
        original degree when ``keep_degree=True``).

        Args:
            direction (int): Parametric direction for differentiation.
                Must be in ``[0, dim)``. Defaults to 0.
            keep_degree (bool): If ``True``, the result preserves the same
                degree as the original B-spline by applying degree elevation
                after differentiation. This is useful, for instance, when
                computing derivatives of rational polynomials (in the
                numerator). Defaults to ``False``.

        Returns:
            Bspline: A new B-spline representing the derivative.

        Raises:
            ValueError: If ``direction`` is out of range ``[0, dim)``.
            ValueError: If the degree in the given direction is 0.

        Example:
            >>> # First derivative of a 1D curve
            >>> f_prime = f.derivative()
            >>> # Partial derivative of a surface with respect to direction 1
            >>> df_dv = surface.derivative(direction=1)
            >>> # Second derivative (composable)
            >>> f_double_prime = f.derivative().derivative()
            >>> # Derivative preserving degree
            >>> f_prime_same_deg = f.derivative(keep_degree=True)
        """
        if direction < 0 or direction >= self.dim:
            raise ValueError(f"direction must be in [0, {self.dim}), got {direction}.")
        if self.space.spaces[direction].degree < 1:
            raise ValueError("Derivative of a degree-0 B-spline is not defined.")
        return _derivative_bspline(self, direction, keep_degree=keep_degree)

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

    def remove_knots(
        self,
        knot_values: float | npt.ArrayLike | Sequence[npt.ArrayLike | None],
        *,
        num: int | None = None,
        tol: float | None = None,
    ) -> Bspline:
        """Return a B-spline with specified interior knots removed.

        Each listed knot value is removed up to *num* times (or as many as
        possible when ``num=None``), provided the geometric deviation stays
        within *tol*.

        Args:
            knot_values (float | npt.ArrayLike | Sequence[npt.ArrayLike | None]):
                For a 1D B-spline, a single float or a 1D array-like of
                distinct interior knot values to remove. For multi-dimensional
                B-splines, a sequence of length ``dim`` where each element is a
                1D array-like of knot values to remove in that direction, or
                ``None`` to skip that direction. At least one direction must
                have a non-empty array of knot values.
            num (int | None): Maximum number of removals per distinct knot
                value. ``None`` (default) removes as many as possible (up to
                the current multiplicity, capped at the degree).
            tol (float | None): Maximum allowed geometric deviation for each
                removal step. ``None`` (default) uses ``1e-10``.

        Returns:
            Bspline: New B-spline with the same geometry (within tolerance)
            and reduced knot vectors.

        Raises:
            ValueError: If the B-spline is periodic in any direction.
            ValueError: If the sequence length does not match ``dim``
                (multi-dim case).
            ValueError: If all directions have empty or ``None`` knot arrays.
            ValueError: If any knot value is not found or is a boundary knot.
        """
        # Periodic splines are not supported.
        for i, sp in enumerate(self._space.spaces):
            if sp.periodic:
                raise ValueError(
                    f"Knot removal is not supported for periodic B-splines "
                    f"(direction {i} is periodic)."
                )

        dtype = self.dtype

        if self.dim == 1:
            arr = np.atleast_1d(np.asarray(knot_values, dtype=dtype)).ravel()
            kv_per_dim: list[npt.NDArray[np.float32 | np.float64] | None] = [arr]
        else:
            seq = list(knot_values)  # type: ignore[arg-type]
            if len(seq) != self.dim:
                raise ValueError(
                    f"knot_values sequence length ({len(seq)}) must match dim ({self.dim})."
                )
            kv_per_dim = [
                None if kv is None else np.atleast_1d(np.asarray(kv, dtype=dtype)).ravel()
                for kv in seq
            ]

        # Require at least one non-empty direction.
        if all(kv is None or kv.size == 0 for kv in kv_per_dim):
            raise ValueError(
                "At least one direction must have a non-empty array of knot values to remove."
            )

        return _remove_knots_bspline(self, kv_per_dim, num, tol)

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

    def restrict(
        self,
        bounds: tuple[float, float] | Sequence[tuple[float, float] | None],
    ) -> Bspline:
        """Return a B-spline restricted to a sub-region of the parametric domain.

        Extracts the portion of the B-spline defined on the given sub-domain by
        inserting knots at the new boundaries until they reach multiplicity
        ``degree + 1``, then extracting the relevant knot sub-vector and control
        points.  Skips insertion when a bound already coincides with a clamped
        domain endpoint.  Periodic directions are automatically converted to
        open form before restricting.

        Args:
            bounds (tuple[float, float] | Sequence[tuple[float, float] | None]):
                For a 1D B-spline, a ``(lower, upper)`` tuple defining the new
                domain.  For multi-dimensional B-splines, a sequence of length
                ``dim`` where each element is a ``(lower, upper)`` tuple for
                that direction, or ``None`` to keep the full domain in that
                direction.  At least one direction must have non-``None`` bounds
                that actually restrict the domain.

        Returns:
            Bspline: New B-spline defined on the restricted domain.

        Raises:
            ValueError: If the sequence length does not match ``dim`` (multi-dim).
            ValueError: If all directions are ``None`` or match the full domain.
            ValueError: If any bound lies outside its direction's domain.
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

        return _restrict_bspline_impl(self, bounds_per_dim)

    def split(self, direction: int, value: float) -> tuple[Bspline, Bspline]:
        """Split the B-spline into two at a parameter value in one direction.

        Inserts knots at ``value`` until the multiplicity reaches
        ``degree + 1``, then extracts the left and right sub-splines.
        Periodic directions are automatically converted to open form before
        splitting.

        Args:
            direction (int): Parametric direction along which to split.
                Must be in ``[0, dim)``.
            value (float): Parameter value at which to split.  Must lie
                strictly inside the domain of the specified direction.

        Returns:
            tuple[Bspline, Bspline]: A pair ``(left, right)`` of B-splines.
            The left sub-spline has domain ``[domain_start, value]`` and
            the right has domain ``[value, domain_end]`` in the split
            direction. Other directions are unchanged.

        Raises:
            ValueError: If ``direction`` is out of range ``[0, dim)``.
            ValueError: If ``value`` is not strictly inside the domain.

        Example:
            >>> left, right = spline.split(0, 0.5)
            >>> left, right = surface.split(1, 0.3)
        """
        if direction < 0 or direction >= self.dim:
            raise ValueError(f"direction must be in [0, {self.dim}), got {direction}.")

        space_1d = self.space.spaces[direction]
        domain = space_1d.domain
        a, b = float(domain[0]), float(domain[1])
        tol = float(space_1d.tolerance)

        if value <= a + tol or value >= b - tol:
            raise ValueError(f"value must be strictly inside the domain ({a}, {b}), got {value}.")

        return _split_bspline_impl(self, direction, value)

    def to_periodic(self, continuity: int | tuple[int | None, ...] | None = None) -> Bspline:
        """Return a periodic B-spline equivalent to this one.

        Converts each non-periodic parametric direction to a periodic
        representation with the requested continuity at the seam (the domain
        boundary where the periodic function wraps around).  Directions that
        are already periodic are left unchanged.

        The conversion performs an exact change of basis: the Oslo algorithm
        defines the linear map from periodic control points (with modulo
        wrapping) to open control points, and inverting this map via QR
        factorization recovers the periodic representation.

        Warning:
            This method assumes the B-spline represents an inherently periodic
            function — i.e., the function values and derivatives match at the
            domain boundaries up to the requested continuity order.  If the
            function is not periodic, the result will **not** represent the
            same geometry.  Gross violations raise ``ValueError`` (via a
            residual check), but small deviations may pass silently.  Use
            :meth:`to_open_bspline` on the result to verify round-trip
            fidelity if in doubt.

        Args:
            continuity (int | tuple[int | None, ...] | None): Target continuity
                at the seam per direction.  ``None`` (default) requests maximum
                regularity ``C^{p-1}`` in every non-periodic direction.  An
                integer applies to all non-periodic directions; a tuple specifies
                per-direction values where ``None`` entries skip that direction
                (leave it unchanged).  Integer values must satisfy
                ``0 <= continuity <= degree - 1``.

        Returns:
            Bspline: B-spline with the requested directions made periodic.

        Raises:
            ValueError: If no direction would be converted (all already periodic
                or all skipped via ``None`` in the tuple).
            ValueError: If the function is not periodic (endpoint mismatch
                or residual exceeds tolerance).
            ValueError: If ``continuity`` is out of range.
        """
        return _to_periodic_bspline_impl(self, continuity)

    def to_bezier(self, *, copy: bool = True) -> Bezier:
        """Convert to an equivalent Bézier.

        Extracts a :class:`~pantr.bezier.Bezier` from this B-spline when
        it represents a single polynomial patch (one element per direction).
        Periodic and non-open B-splines are automatically converted to open
        form before the check.

        Args:
            copy (bool): If ``True`` (default), the control points are
                deep-copied into the new Bézier.  If ``False``, the Bézier
                shares the same underlying control point array when possible
                (direct extraction) or owns the freshly allocated array
                produced by the open-form conversion.

        Returns:
            ~pantr.bezier.Bezier: Equivalent Bézier representation.

        Raises:
            ValueError: If the B-spline has more than one element per
                direction and cannot be represented as a single Bézier.
        """
        from ..bezier import Bezier as BezierCls  # noqa: PLC0415

        if self._space.has_Bezier_like_knots():
            cp = self._control_points.copy() if copy else self._control_points
            return BezierCls(cp, is_rational=self._is_rational)

        # Already open but not Bezier-like → multi-element, cannot convert.
        spaces = self._space.spaces
        if all(s.has_open_knots() and not s.periodic for s in spaces):
            raise ValueError(
                "B-spline has more than one element per direction "
                "and cannot be represented as a single Bézier."
            )

        # Periodic / non-open: convert to open form first.
        open_bspline = self.to_open_bspline()
        if not open_bspline.space.has_Bezier_like_knots():
            raise ValueError(
                "B-spline has more than one element per direction "
                "and cannot be represented as a single Bézier."
            )
        cp = open_bspline.control_points.copy() if copy else open_bspline.control_points
        return BezierCls(cp, is_rational=self._is_rational)

    def to_beziers(self) -> npt.NDArray[np.object_]:
        """Decompose into Bézier patches.

        Decomposes the B-spline into its constituent Bézier patches by applying
        Bézier extraction operators direction by direction.  Periodic directions
        are automatically converted to open form first.

        The result is cached: the first call computes and stores the
        decomposition; subsequent calls return the cached array.  In-place
        mutations (``reverse``, ``permute_directions``, ``transform`` with
        ``in_place=True``) invalidate the cache.

        Returns:
            ``npt.NDArray[np.object_]``: Array of :class:`~pantr.bezier.Bezier`
            objects with shape ``(*num_intervals)`` following the tensor-product
            interval structure.  For a 1D B-spline with 3 intervals the shape
            is ``(3,)``.  For a 2D surface with 3x2 intervals the shape is
            ``(3, 2)``.  The Bézier control points are read-only.

        Example:
            >>> beziers = spline.to_beziers()
            >>> beziers.shape
            (3,)
            >>> beziers[0].degree
            (2,)
        """
        if self._beziers_cache is None:
            self._beziers_cache = _to_beziers_impl(self)
        return self._beziers_cache

    @classmethod
    def from_bezier(cls, bezier: Bezier, *, copy: bool = True) -> Bspline:
        """Create a B-spline from a Bézier.

        Builds a :class:`Bspline` with Bézier-like knot vectors
        (``[0]*(p+1) + [1]*(p+1)`` per direction) whose control points
        are taken from the given Bézier.

        Args:
            bezier (~pantr.bezier.Bezier): The source Bézier.
            copy (bool): If ``True`` (default), the control points are
                deep-copied into the new B-spline.  If ``False``, the
                B-spline shares the same underlying control point array.

        Returns:
            Bspline: Equivalent B-spline with Bézier-like knots.
        """
        return bezier.to_bspline(copy=copy)

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

    # ------------------------------------------------------------------
    # Reverse and permute
    # ------------------------------------------------------------------

    @overload
    def reverse(self, direction: int = ..., *, in_place: Literal[False] = ...) -> Bspline: ...

    @overload
    def reverse(self, direction: int = ..., *, in_place: Literal[True]) -> None: ...

    def reverse(self, direction: int = 0, *, in_place: bool = False) -> Bspline | None:
        """Reverse the orientation of one parametric direction.

        Flips the control points along the given parametric axis and reflects
        the corresponding knot vector so that the mapping is reparametrised in
        the opposite sense along that direction.

        Args:
            direction (int): Parametric direction to reverse. Must be in
                ``[0, dim)``. Defaults to 0.
            in_place (bool): If ``True``, modify this B-spline in place and
                return ``None``. If ``False`` (default), return a new B-spline.

        Returns:
            Bspline | None: The reversed B-spline, or ``None`` when
            ``in_place=True``.

        Raises:
            ValueError: If ``direction`` is out of range ``[0, dim)``.

        Example:
            >>> rev = spline.reverse(direction=0)
            >>> spline.reverse(direction=1, in_place=True)
        """
        if direction < 0 or direction >= self.dim:
            raise ValueError(f"direction must be in [0, {self.dim}), got {direction}.")

        from .._control_points_utils import _reverse_control_points  # noqa: PLC0415
        from ._bspline_space_1d import BsplineSpace1D  # noqa: PLC0415
        from ._bspline_space_nd import BsplineSpace  # noqa: PLC0415

        # Reflect the knot vector: knots_new = a + b - knots[::-1].
        old_space = self._space.spaces[direction]
        knots = old_space.knots
        a, b = old_space.domain
        new_knots = (a + b) - knots[::-1]
        new_space_1d = BsplineSpace1D(new_knots, old_space.degree, periodic=old_space.periodic)

        new_spaces = list(self._space.spaces)
        new_spaces[direction] = new_space_1d
        new_space = BsplineSpace(new_spaces)

        new_cp = _reverse_control_points(self._control_points, direction, in_place=in_place)

        if in_place:
            self._space = new_space
            self._beziers_cache = None
            return None
        return Bspline(new_space, new_cp, is_rational=self._is_rational)

    @overload
    def permute_directions(
        self, permutation: Sequence[int], *, in_place: Literal[False] = ...
    ) -> Bspline: ...

    @overload
    def permute_directions(
        self, permutation: Sequence[int], *, in_place: Literal[True]
    ) -> None: ...

    def permute_directions(
        self, permutation: Sequence[int], *, in_place: bool = False
    ) -> Bspline | None:
        """Reorder the parametric directions according to a permutation.

        Given a permutation ``[i_0, i_1, …]``, the new direction ``k`` is
        the old direction ``permutation[k]``. For example, ``[1, 2, 0]`` on
        a 3D volume maps old direction 1 → new 0, old 2 → new 1, old 0 → new 2.

        Args:
            permutation (Sequence[int]): A permutation of ``range(dim)``.
            in_place (bool): If ``True``, modify this B-spline in place and
                return ``None``. If ``False`` (default), return a new B-spline.

        Returns:
            Bspline | None: The permuted B-spline, or ``None`` when
            ``in_place=True``.

        Raises:
            ValueError: If ``permutation`` is not a valid permutation of
                ``range(dim)``.

        Example:
            >>> surface.permute_directions([1, 0])  # swap u ↔ v
        """
        from .._control_points_utils import _permute_control_points  # noqa: PLC0415
        from ._bspline_space_nd import BsplineSpace  # noqa: PLC0415

        perm = list(permutation)
        if sorted(perm) != list(range(self.dim)):
            raise ValueError(f"permutation must be a permutation of range({self.dim}), got {perm}.")

        new_cp = _permute_control_points(self._control_points, perm, self.dim, in_place=in_place)

        # Reorder 1D spaces.
        old_spaces = self._space.spaces
        new_spaces = tuple(old_spaces[i] for i in perm)
        new_space = BsplineSpace(new_spaces)

        if in_place:
            self._control_points = new_cp
            self._space = new_space
            self._beziers_cache = None
            return None
        return Bspline(new_space, new_cp, is_rational=self._is_rational)

    # ------------------------------------------------------------------
    # Affine transformation
    # ------------------------------------------------------------------

    @overload
    def transform(self, affine: AffineTransform, *, in_place: Literal[False] = ...) -> Bspline: ...

    @overload
    def transform(self, affine: AffineTransform, *, in_place: Literal[True]) -> None: ...

    def transform(
        self,
        affine: AffineTransform,
        *,
        in_place: bool = False,
    ) -> Bspline | None:
        """Apply an affine transformation to the control points.

        For non-rational B-splines, every control point ``P`` is mapped to
        ``A @ P + b``.  For rational B-splines (NURBS) the weighted
        homogeneous coordinates are updated so that the projected geometry
        undergoes the same affine map while the weights are preserved.

        Args:
            affine (~pantr.transform.AffineTransform): The affine
                transformation to apply.
            in_place (bool): If ``True``, the control points are modified in
                place and ``None`` is returned.  If ``False`` (default), a
                new :class:`Bspline` is returned.

        Returns:
            Bspline | None: The transformed B-spline, or ``None`` when
            ``in_place=True``.

        Raises:
            ValueError: If the transform dimension does not match the
                geometric rank of the B-spline.

        Example:
            >>> from pantr.transform import AffineTransform
            >>> T = AffineTransform.translation([1.0, 2.0])
            >>> shifted = spline.transform(T)
        """
        new_cp = _apply_affine_to_control_points(
            self._control_points,
            self._is_rational,
            affine.matrix,
            affine.offset,
            in_place=in_place,
        )
        if in_place:
            self._beziers_cache = None
            return None
        return Bspline(self._space, new_cp, is_rational=self._is_rational)

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

    def slice(self, axis: int, value: float) -> Bspline | npt.NDArray[np.float32 | np.float64]:
        """Slice the B-spline by fixing one parametric direction at a given value.

        Reduces the parametric dimension by one using de Boor corner cutting
        on the control points.  A volume becomes a surface, a surface becomes
        a curve, and a curve becomes a point (returned as a NumPy array).

        When the parameter value coincides with a knot of multiplicity ``s``,
        only ``p - s`` de Boor iterations are needed.  At a C0 knot
        (``s >= p``) the result is obtained in O(1) by direct control point
        lookup.

        Args:
            axis (int): Parametric direction to fix (0-indexed).
                Must be in ``[0, dim)``.
            value (float): Parameter value at which to slice.  Must lie
                within the domain of the specified direction.

        Returns:
            Bspline | npt.NDArray[np.float32 | np.float64]:
            A B-spline with ``dim - 1`` dimensions when ``dim >= 2``,
            or a NumPy array of shape ``(rank,)`` when ``dim == 1``.
            Rational B-splines preserve the NURBS structure when ``dim >= 2``;
            for ``dim == 1`` the result is projected to physical coordinates.

        Raises:
            ValueError: If ``axis`` is out of range ``[0, dim)``.
            ValueError: If ``value`` is outside the domain of the specified
                direction.

        Example:
            >>> # Slice a surface at v=0.5 to get a curve
            >>> curve = surface.slice(1, 0.5)
            >>> # Slice a volume at w=0.3 to get a surface
            >>> srf = volume.slice(2, 0.3)
            >>> # Composable: volume -> surface -> curve -> point
            >>> pt = volume.slice(2, 0.3).slice(1, 0.5).slice(0, 0.2)
        """
        if axis < 0 or axis >= self.dim:
            raise ValueError(f"axis must be in [0, {self.dim}), got {axis}.")

        space_d = self.space.spaces[axis]
        domain = space_d.domain
        tol = float(space_d.tolerance)
        if value < float(domain[0]) - tol or value > float(domain[1]) + tol:
            raise ValueError(
                f"value {value} is outside the domain [{domain[0]}, {domain[1]}] "
                f"of direction {axis}."
            )

        return _slice_bspline(self, axis, value)

    def boundary(self, axis: int, side: int) -> Bspline | npt.NDArray[np.float32 | np.float64]:
        """Extract the boundary of the B-spline along one parametric direction.

        Returns the restriction of the B-spline to one end of the domain
        in the given direction.

        Args:
            axis (int): Parametric direction (0-indexed).
                Must be in ``[0, dim)``.
            side (int): Which end of the domain: ``0`` for the start,
                ``1`` for the end.

        Returns:
            Bspline | npt.NDArray[np.float32 | np.float64]:
            A B-spline with ``dim - 1`` dimensions when ``dim >= 2``,
            or a NumPy array of shape ``(rank,)`` when ``dim == 1``.

        Raises:
            ValueError: If ``axis`` is out of range ``[0, dim)``.
            ValueError: If ``side`` is not 0 or 1.

        Example:
            >>> # Extract left boundary of a surface along direction 0
            >>> left_curve = surface.boundary(0, 0)
            >>> # Extract right boundary along direction 1
            >>> right_curve = surface.boundary(1, 1)
        """
        if side not in (0, 1):
            raise ValueError(f"side must be 0 or 1, got {side}.")
        if axis < 0 or axis >= self.dim:
            raise ValueError(f"axis must be in [0, {self.dim}), got {axis}.")

        space_d = self.space.spaces[axis]
        domain = space_d.domain
        value = float(domain[0]) if side == 0 else float(domain[1])
        return self.slice(axis, value)
