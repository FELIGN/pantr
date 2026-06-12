"""THB spline function: a :class:`THBSplineSpace` paired with control points.

This module provides :class:`THBSpline`, the hierarchical analogue of
:class:`~pantr.bspline.Bspline`.  It represents a function
``f(u) = Σ_i c_i φ_i(u)`` where ``φ_i`` are the active (truncated) hierarchical
basis functions of a :class:`~pantr.bspline.THBSplineSpace` and ``c_i`` the control
points (one per active dof).  Evaluation locates each point's active leaf cell and
combines the cell's active-basis values with the matching control points.

Main exports:

- :class:`THBSpline`: an evaluable hierarchical B-spline function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import typing as npt

from ._thb_spline_space import THBSplineSpace

if TYPE_CHECKING:
    from collections.abc import Sequence


class THBSpline:
    """An evaluable THB spline function ``f = Σ_i c_i φ_i``.

    The hierarchical analogue of :class:`~pantr.bspline.Bspline`: it pairs a
    :class:`~pantr.bspline.THBSplineSpace` with one control point per active
    hierarchical dof.  Because the hierarchical basis is not of tensor-product
    structure, the control points stay a flat ``(num_total_basis,)`` (scalar) or
    ``(num_total_basis, rank)`` (vector-valued) array, rather than the
    ``(*num_basis, rank)`` grid of :class:`~pantr.bspline.Bspline`.

    Attributes:
        _space (THBSplineSpace): The hierarchical space.
        _control_points (npt.NDArray[np.float64]): Control points reshaped to
            ``(num_total_basis, rank)`` (read-only).
        _scalar (bool): Whether the field is scalar (``control_points`` was 1-D).
    """

    __slots__ = ("_control_points", "_scalar", "_space")

    def __init__(self, space: THBSplineSpace, control_points: npt.ArrayLike) -> None:
        """Create a THB spline function.

        Args:
            space (THBSplineSpace): The hierarchical space.
            control_points (npt.ArrayLike): Control points of shape
                ``(num_total_basis,)`` (scalar) or ``(num_total_basis, rank)``
                (vector-valued), ordered by global dof.

        Raises:
            TypeError: If ``space`` is not a :class:`THBSplineSpace`.
            ValueError: If ``control_points`` is not 1-D/2-D or its leading dimension
                does not equal ``space.num_total_basis``.
        """
        if not isinstance(space, THBSplineSpace):
            raise TypeError(f"space must be a THBSplineSpace; got {type(space).__name__!r}.")
        arr = np.asarray(control_points, dtype=np.float64)
        n = space.num_total_basis
        if arr.ndim == 1:
            scalar = True
            if arr.shape[0] != n:
                raise ValueError(f"control_points must have length {n}; got {arr.shape[0]}.")
            arr = arr.reshape(n, 1)
        elif arr.ndim == 2:  # noqa: PLR2004
            scalar = False
            if arr.shape[0] != n:
                raise ValueError(
                    f"control_points leading dimension must be {n}; got shape {arr.shape!r}."
                )
        else:
            raise ValueError(f"control_points must be 1-D or 2-D; got {arr.ndim}-D.")
        self._space = space
        self._control_points = np.ascontiguousarray(arr, dtype=np.float64)
        # Ensure we own the data before marking read-only; ascontiguousarray may
        # return a view of the input when it is already C-contiguous float64.
        if not self._control_points.flags.owndata:
            self._control_points = self._control_points.copy()
        self._control_points.flags.writeable = False
        self._scalar = scalar

    @property
    def space(self) -> THBSplineSpace:
        """Get the underlying hierarchical space.

        Returns:
            THBSplineSpace: The space supplied at construction time.
        """
        return self._space

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        """Get the control points (read-only view).

        Returns:
            npt.NDArray[np.float64]: Shape ``(num_total_basis,)`` for a scalar field,
            ``(num_total_basis, rank)`` otherwise.  The array is read-only; copy it
            before modifying.
        """
        if self._scalar:
            return self._control_points[:, 0]
        return self._control_points

    @property
    def dim(self) -> int:
        """Get the parametric dimension.

        Returns:
            int: Number of parametric directions.
        """
        return self._space.dim

    @property
    def degree(self) -> tuple[int, ...]:
        """Get the per-direction polynomial degrees.

        Returns:
            tuple[int, ...]: Degree per parametric direction (mirrors
            :attr:`~pantr.bspline.Bspline.degree`).
        """
        return self._space.degrees

    @property
    def rank(self) -> int:
        """Get the output rank (number of value components).

        Returns:
            int: ``1`` for a scalar field; the number of components otherwise.
        """
        return int(self._control_points.shape[1])

    @property
    def dtype(self) -> np.dtype[np.float64]:
        """Get the floating-point dtype of the control points.

        Returns:
            np.dtype[np.float64]: Dtype of the stored control points, always
            ``numpy.float64``.
        """
        return self._control_points.dtype

    def evaluate(
        self,
        pts: npt.ArrayLike,
        out: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the THB spline at ``pts``.

        Each point is located in its active leaf cell; the cell's active-basis values
        (:meth:`THBSplineSpace.tabulate_basis`) are combined with the matching control
        points.  Mirrors :meth:`~pantr.bspline.Bspline.evaluate`.

        Args:
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)``.
            out (npt.NDArray[np.float64] | None): Optional output array of the result
                shape (``(...)`` for a scalar field, ``(..., rank)`` otherwise).
                Allocated when ``None``.

        Returns:
            npt.NDArray[np.float64]: Values of shape ``(...)`` for a scalar field
            or ``(..., rank)`` for a vector-valued field.

        Raises:
            ValueError: If ``pts`` does not have trailing dimension ``dim``, any point
                lies outside the grid domain, or ``out`` has the wrong shape/dtype or
                is not writeable.
            RuntimeError: If the grid has been modified since construction, or a cell
                has no active basis functions (inconsistent space).
        """
        return self._evaluate_orders(pts, (0,) * self.dim, out)

    def evaluate_derivatives(
        self,
        pts: npt.ArrayLike,
        orders: int | Sequence[int],
        out: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        r"""Evaluate a mixed partial derivative of the THB spline at ``pts``.

        Computes the single mixed partial :math:`\partial^{orders} f`, where
        ``orders[k]`` is the derivative order in parametric direction ``k`` (with
        respect to the parametric coordinates).  Mirrors
        :meth:`~pantr.bspline.Bspline.evaluate_derivatives`.

        Args:
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)``.
            orders (int | Sequence[int]): Per-direction derivative orders.  A scalar is
                broadcast to every direction.  Each entry must be ``>= 0``.
            out (npt.NDArray[np.float64] | None): Optional output array of the result
                shape.  Allocated when ``None``.

        Returns:
            npt.NDArray[np.float64]: Derivative values of shape ``(...)`` for a scalar
            field or ``(..., rank)`` for a vector-valued field.

        Raises:
            ValueError: If ``orders`` has the wrong length or a negative entry, if
                ``pts`` does not have trailing dimension ``dim``, any point lies
                outside the grid domain, or ``out`` has the wrong shape/dtype or is
                not writeable.
            RuntimeError: If the grid has been modified since construction, or a cell
                has no active basis functions (inconsistent space).
        """
        return self._evaluate_orders(pts, orders, out)

    def _evaluate_orders(
        self,
        pts: npt.ArrayLike,
        orders: int | Sequence[int],
        out: npt.NDArray[np.float64] | None,
    ) -> npt.NDArray[np.float64]:
        """Shared evaluation of the ``orders`` mixed partial (values when all zero).

        Args:
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)``.
            orders (int | Sequence[int]): Per-direction derivative orders (validated by
                :meth:`THBSplineSpace.tabulate_basis_derivatives`).
            out (npt.NDArray[np.float64] | None): Optional output array of the result
                shape.

        Returns:
            npt.NDArray[np.float64]: Values of shape ``(...)`` / ``(..., rank)``.

        Raises:
            ValueError: If ``pts``/``orders``/``out`` are invalid (see the public
                methods).
            RuntimeError: If the grid is stale or a cell has no active functions.
        """
        self._space._check_not_stale()
        arr = np.asarray(pts, dtype=np.float64)
        if arr.ndim == 0 or arr.shape[-1] != self.dim:
            raise ValueError(
                f"pts must have trailing dimension {self.dim}; got shape {arr.shape!r}."
            )
        batch_shape = arr.shape[:-1]
        flat = arr.reshape(-1, self.dim)
        n_pts = flat.shape[0]
        rank = self.rank
        grid = self._space.grid

        cids = grid.locate_many(flat)
        if n_pts > 0 and int(cids.min()) < 0:
            i = int(np.argmax(cids < 0))
            raise ValueError(f"point {flat[i].tolist()!r} lies outside the grid domain.")

        # Group points by cell via a stable argsort: order[starts[g]:ends[g]]
        # holds the point indices of the g-th occupied cell. Avoids the
        # O(num_cells * num_pts) per-cell boolean masks.
        order = np.argsort(cids, kind="stable")
        sorted_cids = cids[order]
        boundaries = np.flatnonzero(np.diff(sorted_cids)) + 1
        if n_pts > 0:
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [n_pts]))
        else:
            starts = np.empty(0, dtype=np.int64)
            ends = np.empty(0, dtype=np.int64)

        raw = np.empty((n_pts, rank), dtype=np.float64)
        for s, e in zip(starts, ends, strict=True):
            cid = int(sorted_cids[s])
            idx = order[s:e]
            values, dofs = self._space.tabulate_basis_derivatives(cid, flat[idx], orders)
            if dofs.size == 0:
                raise RuntimeError(
                    f"cell {cid} has no active basis functions; "
                    "the THBSplineSpace may be inconsistent."
                )
            raw[idx] = np.asarray(values, dtype=np.float64) @ self._control_points[dofs]

        result = raw[:, 0].reshape(batch_shape) if self._scalar else raw.reshape(*batch_shape, rank)
        if out is None:
            return result
        if out.shape != result.shape:
            raise ValueError(f"out must have shape {result.shape}; got {out.shape}.")
        if out.dtype != np.float64:
            raise ValueError(f"out must have dtype float64; got {out.dtype}.")
        if not out.flags.writeable:
            raise ValueError("out must be writeable.")
        out[...] = result
        return out

    def __repr__(self) -> str:
        """Return a concise representation.

        Returns:
            str: Summary with dimension, rank, and active-function count.
        """
        return (
            f"THBSpline(dim={self.dim}, rank={self.rank}, num_total_basis="
            f"{self._space.num_total_basis})"
        )
