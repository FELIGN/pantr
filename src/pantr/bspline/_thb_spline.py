"""THB spline function: a :class:`THBSplineSpace` paired with coefficients.

This module provides :class:`THBSpline`, the hierarchical analogue of
:class:`~pantr.bspline.Bspline`.  It represents a function
``f(u) = Σ_i c_i φ_i(u)`` where ``φ_i`` are the active (truncated) hierarchical
basis functions of a :class:`~pantr.bspline.THBSplineSpace` and ``c_i`` the
coefficients (one per active dof).  Evaluation locates each point's active leaf
cell and combines the cell's active-basis values with the matching coefficients.

Main exports:

- :class:`THBSpline`: an evaluable hierarchical B-spline function.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from ._thb_spline_space import THBSplineSpace


class THBSpline:
    """An evaluable THB spline function ``f = Σ_i c_i φ_i``.

    Pairs a :class:`~pantr.bspline.THBSplineSpace` with a coefficient per active
    hierarchical dof.  Scalar (``coeffs`` shape ``(num_active,)``) and
    vector-valued (``(num_active, rank)``) fields are both supported.

    Attributes:
        _space (THBSplineSpace): The hierarchical space.
        _coeffs (npt.NDArray[np.float64]): Coefficients reshaped to
            ``(num_active, rank)``.
        _scalar (bool): Whether the field is scalar (``coeffs`` was 1-D).
    """

    __slots__ = ("_coeffs", "_scalar", "_space")

    def __init__(self, space: THBSplineSpace, coeffs: npt.ArrayLike) -> None:
        """Create a THB spline function.

        Args:
            space (THBSplineSpace): The hierarchical space.
            coeffs (npt.ArrayLike): Coefficients of shape ``(num_active,)`` (scalar)
                or ``(num_active, rank)`` (vector-valued), ordered by global dof.

        Raises:
            TypeError: If ``space`` is not a :class:`THBSplineSpace`.
            ValueError: If ``coeffs`` is not 1-D/2-D or its leading dimension does
                not equal ``space.num_active_functions``.
        """
        if not isinstance(space, THBSplineSpace):
            raise TypeError(f"space must be a THBSplineSpace; got {type(space).__name__!r}.")
        arr = np.asarray(coeffs, dtype=np.float64)
        n = space.num_active_functions
        if arr.ndim == 1:
            scalar = True
            if arr.shape[0] != n:
                raise ValueError(f"coeffs must have length {n}; got {arr.shape[0]}.")
            arr = arr.reshape(n, 1)
        elif arr.ndim == 2:  # noqa: PLR2004
            scalar = False
            if arr.shape[0] != n:
                raise ValueError(f"coeffs leading dimension must be {n}; got shape {arr.shape!r}.")
        else:
            raise ValueError(f"coeffs must be 1-D or 2-D; got {arr.ndim}-D.")
        self._space = space
        self._coeffs = np.ascontiguousarray(arr, dtype=np.float64)
        self._scalar = scalar

    @property
    def space(self) -> THBSplineSpace:
        """Get the underlying hierarchical space.

        Returns:
            THBSplineSpace: The space supplied at construction time.
        """
        return self._space

    @property
    def coeffs(self) -> npt.NDArray[np.float64]:
        """Get the coefficients.

        Returns:
            npt.NDArray[np.float64]: Shape ``(num_active,)`` for a scalar field,
            ``(num_active, rank)`` otherwise.
        """
        if self._scalar:
            return self._coeffs[:, 0]
        return self._coeffs

    @property
    def dim(self) -> int:
        """Get the parametric dimension.

        Returns:
            int: Number of parametric directions.
        """
        return self._space.dim

    @property
    def rank(self) -> int:
        """Get the output rank (number of value components).

        Returns:
            int: ``1`` for a scalar field; the number of components otherwise.
        """
        return int(self._coeffs.shape[1])

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the floating-point dtype of the coefficients.

        Returns:
            npt.DTypeLike: ``numpy.float64``.
        """
        return np.float64

    def evaluate(self, pts: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Evaluate the THB spline at ``pts``.

        Each point is located in its active leaf cell; the cell's active-basis
        values (:meth:`THBSplineSpace.tabulate_basis`) are combined with the
        matching coefficients.

        Args:
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)``.

        Returns:
            npt.NDArray[np.float64]: Values of shape ``(...)`` for a scalar field
            or ``(..., rank)`` for a vector-valued field.

        Raises:
            ValueError: If ``pts`` does not have trailing dimension ``dim`` or any
                point lies outside the grid domain.
            RuntimeError: If the grid has been modified since construction.
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

        cids = np.empty(n_pts, dtype=np.int64)
        for i in range(n_pts):
            cid = grid.locate(flat[i])
            if cid is None:
                raise ValueError(f"point {flat[i].tolist()!r} lies outside the grid domain.")
            cids[i] = cid

        out = np.empty((n_pts, rank), dtype=np.float64)
        for cid in np.unique(cids):
            mask = cids == cid
            dofs = self._space.active_basis(int(cid))
            values = self._space.tabulate_basis(int(cid), flat[mask])
            out[mask] = np.asarray(values, dtype=np.float64) @ self._coeffs[dofs]

        if self._scalar:
            return out[:, 0].reshape(batch_shape)
        return out.reshape(*batch_shape, rank)

    def __repr__(self) -> str:
        """Return a concise representation.

        Returns:
            str: Summary with dimension, rank, and active-function count.
        """
        return (
            f"THBSpline(dim={self.dim}, rank={self.rank}, "
            f"num_active={self._space.num_active_functions})"
        )
