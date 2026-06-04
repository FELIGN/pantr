"""Truncated hierarchical B-spline spaces (THB-splines).

This module defines :class:`THBSplineSpace`, a hierarchical spline space built on
a :class:`pantr.grid.HierarchicalGrid`.  It follows the G+Smo *self-evaluating*
model: per level it stores the Kraft selection of active tensor-product B-splines,
and (from PR2 onward) a sparse coefficient vector only for truncated functions.
Untruncated functions are plain tensor-product B-splines.

This first increment implements the **non-truncated hierarchical (HB)** path
(``truncate=False``): nested per-level spaces, the active-function selection, and
value evaluation.  Truncation (``truncate=True``) and derivatives land in later
increments (see GitHub issue #164).

Main exports:

- :class:`THBSplineSpace`: hierarchical B-spline space on a
  :class:`~pantr.grid.HierarchicalGrid`.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from ..grid import HierarchicalGrid
from ._bspline_space_nd import BsplineSpace

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from ._bspline_space_1d import BsplineSpace1D

_Support1D = tuple[
    "npt.NDArray[np.int64]",
    "npt.NDArray[np.int64]",
    "npt.NDArray[np.int64]",
]
"""Per-direction function support at one level.

``(first_basis_per_interval, first_cell_per_function, last_cell_per_function)``,
all ``int64`` arrays.
"""


def _func_support_1d(space: BsplineSpace1D) -> _Support1D:
    """Compute the cell support of every B-spline function of a 1D space.

    The first non-zero function index per interval is obtained from
    :meth:`~pantr.bspline.BsplineSpace1D.tabulate_basis` evaluated at interval
    midpoints (robust to knot multiplicities), then inverted to give, for each
    function ``i``, the inclusive interval (cell) range ``[first_cell, last_cell]``
    it is supported on.

    Args:
        space (BsplineSpace1D): The 1D B-spline space.

    Returns:
        _Support1D: ``(first_basis, first_cell, last_cell)`` where ``first_basis``
        has length ``num_intervals`` and ``first_cell`` / ``last_cell`` have length
        ``num_basis``.
    """
    unique_knots, _ = space.get_unique_knots_and_multiplicity(in_domain=True)
    midpoints = 0.5 * (unique_knots[:-1] + unique_knots[1:])
    _, first_basis_per_pt = space.tabulate_basis(midpoints)
    first_basis = np.asarray(first_basis_per_pt, dtype=np.int64)

    degree = space.degree
    n_basis = space.num_basis
    first_cell = np.full(n_basis, -1, dtype=np.int64)
    last_cell = np.full(n_basis, -1, dtype=np.int64)
    for interval in range(first_basis.shape[0]):
        lo_i = int(first_basis[interval])
        for i in range(lo_i, lo_i + degree + 1):
            if first_cell[i] < 0:
                first_cell[i] = interval
            last_cell[i] = interval
    return first_basis, first_cell, last_cell


class THBSplineSpace:
    r"""Hierarchical B-spline space on a :class:`~pantr.grid.HierarchicalGrid`.

    Built from a root :class:`~pantr.bspline.BsplineSpace` (level 0) and a
    :class:`~pantr.grid.HierarchicalGrid` carrying the active-cell hierarchy.  The
    per-level tensor-product spaces are obtained by uniformly subdividing the root
    space according to the grid's per-direction ``factor``.  The active hierarchical
    basis is the Kraft selection: a level-``l`` tensor-product B-spline is active iff
    its support lies in the level-``l`` subdomain :math:`\Omega_l` but not entirely
    in the finer subdomain :math:`\Omega_{l+1}`.

    This increment implements only the **non-truncated (HB)** basis
    (``truncate=False``); ``truncate=True`` raises :class:`NotImplementedError`
    (truncation lands in a later increment, GitHub issue #164).

    Attributes:
        _root_space (BsplineSpace): The level-0 tensor-product space.
        _grid (HierarchicalGrid): The active-cell hierarchy.
        _truncate (bool): Whether the basis is truncated (always ``False`` here).
        _regularity (tuple[int | None, ...]): Per-direction continuity used when
            subdividing to build finer levels.
        _level_spaces (tuple[BsplineSpace, ...]): Per-level tensor-product spaces;
            index ``l`` is the root subdivided to level ``l``.
        _support (tuple): Per-level, per-direction function-to-cell support arrays
            (one ``_Support1D`` triple per direction per level).
        _active_funcs (tuple[npt.NDArray[np.int64], ...]): Per-level sorted flat
            (C-order) indices of the active tensor-product functions.
        _func_offset (npt.NDArray[np.int64]): Per-level global-dof base; length
            ``num_levels + 1`` (cumulative active-function counts).
        _num_active (int): Total number of active hierarchical functions.
    """

    __slots__ = (
        "_active_funcs",
        "_func_offset",
        "_grid",
        "_level_spaces",
        "_num_active",
        "_regularity",
        "_root_space",
        "_support",
        "_truncate",
    )

    def __init__(
        self,
        root_space: BsplineSpace,
        grid: HierarchicalGrid,
        *,
        truncate: bool = True,
        regularity: int | Sequence[int | None] | None = None,
    ) -> None:
        """Create a hierarchical B-spline space.

        Args:
            root_space (BsplineSpace): The level-0 tensor-product B-spline space.
            grid (HierarchicalGrid): Hierarchical grid whose root knot-span grid
                matches ``root_space``.
            truncate (bool): If ``True`` (default), build the truncated basis.
                **Not yet implemented** — raises :class:`NotImplementedError`.
                Use ``truncate=False`` for the non-truncated hierarchical basis.
            regularity (int | Sequence[int | None] | None): Per-direction continuity
                at the knots inserted when subdividing to finer levels.  A scalar is
                broadcast to every axis; ``None`` (default) uses maximal smoothness.

        Raises:
            TypeError: If ``grid`` is not a :class:`~pantr.grid.HierarchicalGrid`.
            ValueError: If ``grid`` and ``root_space`` disagree on dimension or on
                the root knot-span grid, or if ``regularity`` has the wrong length.
            NotImplementedError: If ``truncate`` is ``True``.
        """
        if not isinstance(grid, HierarchicalGrid):
            raise TypeError(f"grid must be a HierarchicalGrid; got {type(grid).__name__!r}.")
        dim = root_space.dim
        if grid.ndim != dim:
            raise ValueError(f"grid.ndim ({grid.ndim}) must equal root_space.dim ({dim}).")
        if tuple(grid.root.cells_per_axis) != tuple(root_space.num_intervals):
            raise ValueError(
                f"grid root cells_per_axis {tuple(grid.root.cells_per_axis)!r} must match "
                f"root_space.num_intervals {tuple(root_space.num_intervals)!r}."
            )
        if not np.allclose(
            np.asarray(grid.root.bounds, dtype=np.float64),
            np.asarray(root_space.domain, dtype=np.float64),
        ):
            raise ValueError("grid root bounds must match root_space domain.")

        if regularity is None or isinstance(regularity, int):
            reg: tuple[int | None, ...] = (regularity,) * dim
        else:
            reg = tuple(regularity)
            if len(reg) != dim:
                raise ValueError(
                    f"regularity must be a scalar or length-{dim} sequence; got length {len(reg)}."
                )

        if truncate:
            raise NotImplementedError(
                "truncation (truncate=True) is not implemented yet; it lands in a later "
                "increment (GitHub issue #164). Use truncate=False for the non-truncated "
                "hierarchical (HB) basis."
            )

        self._root_space = root_space
        self._grid = grid
        self._truncate = truncate
        self._regularity = reg

        self._level_spaces = self._build_level_spaces()
        self._support = tuple(
            tuple(_func_support_1d(sp1d) for sp1d in level_space.spaces)
            for level_space in self._level_spaces
        )
        self._active_funcs = self._select_active_functions()
        counts = [int(a.shape[0]) for a in self._active_funcs]
        self._func_offset = np.concatenate(([0], np.cumsum(counts, dtype=np.int64))).astype(
            np.int64
        )
        self._num_active = int(self._func_offset[-1])

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_level_spaces(self) -> tuple[BsplineSpace, ...]:
        """Build the nested per-level tensor-product spaces.

        Level ``l + 1`` is obtained from level ``l`` by subdividing every 1D space
        by the grid ``factor`` (skipping axes whose factor is ``1``), which keeps
        the level spaces nested.

        Returns:
            tuple[BsplineSpace, ...]: Spaces of length ``num_levels``.
        """
        factor = self._grid.factor
        reg = self._regularity
        current = list(self._root_space.spaces)
        level_spaces: list[BsplineSpace] = [self._root_space]
        for _ in range(1, self._grid.max_level + 1):
            current = [
                sp if factor[k] == 1 else sp.subdivide(factor[k], reg[k])
                for k, sp in enumerate(current)
            ]
            level_spaces.append(BsplineSpace(current))
        return tuple(level_spaces)

    def _select_active_functions(self) -> tuple[npt.NDArray[np.int64], ...]:
        r"""Compute the Kraft selection of active functions per level.

        A level-``l`` tensor-product function is selected iff its support lies in
        :math:`\Omega_l` (``subdomain_mask``) but not entirely in
        :math:`\Omega_{l+1}` (``subdomain_mask & ~active_leaf_mask``).

        Returns:
            tuple[npt.NDArray[np.int64], ...]: Per-level sorted flat (C-order)
            indices of the active functions.
        """
        dim = self.dim
        active: list[npt.NDArray[np.int64]] = []
        for level in range(self.num_levels):
            num_basis = self._level_spaces[level].num_basis
            subdomain = self._grid.subdomain_mask(level)
            refined = subdomain & ~self._grid.active_leaf_mask(level)
            support = self._support[level]

            true_coords = np.argwhere(subdomain)
            if true_coords.shape[0] == 0:
                active.append(np.empty(0, dtype=np.int64))
                continue
            bbox_lo = true_coords.min(axis=0)
            bbox_hi = true_coords.max(axis=0) + 1

            candidates_per_dir: list[list[int]] = []
            for k in range(dim):
                _, first_cell, last_cell = support[k]
                overlaps = (last_cell >= bbox_lo[k]) & (first_cell < bbox_hi[k])
                candidates_per_dir.append(np.nonzero(overlaps)[0].tolist())

            selected: list[int] = []
            for multi in itertools.product(*candidates_per_dir):
                box = tuple(
                    slice(int(support[k][1][multi[k]]), int(support[k][2][multi[k]]) + 1)
                    for k in range(dim)
                )
                if not bool(subdomain[box].all()):
                    continue
                if bool(refined[box].all()):
                    continue
                selected.append(int(np.ravel_multi_index(multi, num_basis)))
            active.append(np.array(sorted(selected), dtype=np.int64))
        return tuple(active)

    def _cell_contributions(self, cid: int) -> list[tuple[int, int, tuple[int, ...]]]:
        """Return the active functions non-zero on cell ``cid``.

        Args:
            cid (int): Active cell flat id.

        Returns:
            list[tuple[int, int, tuple[int, ...]]]: ``(global_dof, level, multi)``
            triples sorted by ``global_dof``, where ``multi`` is the contributing
            function's flat-free multi-index in its level space.
        """
        cell_level = self._grid.cell_level(cid)
        cell_midx = self._grid.cell_multi_index(cid)
        factor = self._grid.factor
        dim = self.dim
        contribs: list[tuple[int, int, tuple[int, ...]]] = []
        for level in range(cell_level + 1):
            divisor = tuple(factor[k] ** (cell_level - level) for k in range(dim))
            cell_at_level = tuple(cell_midx[k] // divisor[k] for k in range(dim))
            num_basis = self._level_spaces[level].num_basis
            support = self._support[level]
            ranges = []
            for k in range(dim):
                first_basis = support[k][0]
                f0 = int(first_basis[cell_at_level[k]])
                ranges.append(range(f0, f0 + self.degrees[k] + 1))
            active_at_level = self._active_funcs[level]
            offset = int(self._func_offset[level])
            for multi in itertools.product(*ranges):
                flat = int(np.ravel_multi_index(multi, num_basis))
                pos = int(np.searchsorted(active_at_level, flat))
                if pos < active_at_level.shape[0] and int(active_at_level[pos]) == flat:
                    contribs.append((offset + pos, level, multi))
        contribs.sort(key=lambda triple: triple[0])
        return contribs

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def grid(self) -> HierarchicalGrid:
        """Get the underlying hierarchical grid.

        Returns:
            HierarchicalGrid: The active-cell hierarchy this space is built on.
        """
        return self._grid

    @property
    def root_space(self) -> BsplineSpace:
        """Get the level-0 tensor-product space.

        Returns:
            BsplineSpace: The root B-spline space.
        """
        return self._root_space

    @property
    def dim(self) -> int:
        """Get the parametric dimension.

        Returns:
            int: Number of parametric directions.
        """
        return self._root_space.dim

    @property
    def degrees(self) -> tuple[int, ...]:
        """Get the per-direction polynomial degrees.

        Returns:
            tuple[int, ...]: Degree per direction (the same at every level).
        """
        return self._root_space.degrees

    @property
    def num_levels(self) -> int:
        """Get the number of hierarchy levels.

        Returns:
            int: ``grid.max_level + 1``.
        """
        return self._grid.max_level + 1

    @property
    def truncate(self) -> bool:
        """Get whether the hierarchical basis is truncated.

        Returns:
            bool: ``True`` for THB, ``False`` for plain HB.
        """
        return self._truncate

    @property
    def num_active_functions(self) -> int:
        """Get the total number of active hierarchical functions.

        Returns:
            int: Total active-function count across all levels.
        """
        return self._num_active

    @property
    def num_active_functions_per_level(self) -> tuple[int, ...]:
        """Get the number of active functions at each level.

        Returns:
            tuple[int, ...]: Active-function count per level.
        """
        return tuple(int(a.shape[0]) for a in self._active_funcs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def level_space(self, level: int) -> BsplineSpace:
        """Return the tensor-product space at ``level``.

        Args:
            level (int): Hierarchy level in ``[0, num_levels)``.

        Returns:
            BsplineSpace: The root space subdivided to ``level``.

        Raises:
            IndexError: If ``level`` is out of range.
        """
        if not (0 <= level < self.num_levels):
            raise IndexError(f"level must be in [0, {self.num_levels - 1}]; got {level!r}.")
        return self._level_spaces[level]

    def active_function_indices(self, level: int) -> npt.NDArray[np.int64]:
        """Return the flat indices of the active functions at ``level``.

        Args:
            level (int): Hierarchy level in ``[0, num_levels)``.

        Returns:
            npt.NDArray[np.int64]: Sorted flat (C-order) level-``level`` function
            indices selected by the Kraft rule.  A fresh copy is returned.

        Raises:
            IndexError: If ``level`` is out of range.
        """
        if not (0 <= level < self.num_levels):
            raise IndexError(f"level must be in [0, {self.num_levels - 1}]; got {level!r}.")
        indices: npt.NDArray[np.int64] = self._active_funcs[level].copy()
        return indices

    def active_basis(self, cid: int) -> npt.NDArray[np.int64]:
        """Return the global dofs of the active functions non-zero on cell ``cid``.

        Args:
            cid (int): Active cell flat id.

        Returns:
            npt.NDArray[np.int64]: Sorted global hierarchical-dof indices of the
            functions whose support intersects cell ``cid``.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        return np.array([dof for dof, _, _ in self._cell_contributions(cid)], dtype=np.int64)

    def tabulate_basis(
        self,
        cid: int,
        pts: npt.ArrayLike,
        out: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the active hierarchical functions on cell ``cid`` at ``pts``.

        For the non-truncated basis, each active function is a single
        tensor-product B-spline; the value is the product of its 1D B-spline values
        at the corresponding level.  The returned columns are ordered as
        :meth:`active_basis` (sorted global dof).

        Args:
            cid (int): Active cell flat id.
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)`` lying in
                cell ``cid``.
            out (npt.NDArray[np.float64] | None): Optional output array of shape
                ``(..., K)`` with ``K = active_basis(cid).size``.  Allocated when
                ``None``.

        Returns:
            npt.NDArray[np.float64]: Function values of shape ``(..., K)``.

        Raises:
            IndexError: If ``cid`` is out of range.
            ValueError: If ``pts`` does not have trailing dimension ``dim`` or
                ``out`` has the wrong shape, dtype, or is not writeable.
        """
        contribs = self._cell_contributions(cid)
        n_active = len(contribs)

        pts_arr = np.asarray(pts, dtype=np.float64)
        if pts_arr.ndim == 0 or pts_arr.shape[-1] != self.dim:
            raise ValueError(
                f"pts must have trailing dimension {self.dim}; got shape {pts_arr.shape}."
            )
        lead = pts_arr.shape[:-1]
        num_pts = int(np.prod(lead)) if lead else 1
        flat_pts = pts_arr.reshape(num_pts, self.dim)
        out_shape = (*lead, n_active)

        if out is None:
            result = np.empty(out_shape, dtype=np.float64)
        else:
            if out.shape != out_shape:
                raise ValueError(f"out must have shape {out_shape}; got {out.shape}.")
            if out.dtype != np.float64:
                raise ValueError(f"out must have dtype float64; got {out.dtype}.")
            if not out.flags.writeable:
                raise ValueError("out must be writeable.")
            result = out

        buffer = np.empty((num_pts, n_active), dtype=np.float64)
        eval_cache: dict[
            tuple[int, int], tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]
        ] = {}
        point_index = np.arange(num_pts)
        for col, (_, level, multi) in enumerate(contribs):
            column = np.ones(num_pts, dtype=np.float64)
            for k in range(self.dim):
                key = (level, k)
                if key not in eval_cache:
                    sp1d = self._level_spaces[level].spaces[k]
                    values, first_basis = sp1d.tabulate_basis(np.ascontiguousarray(flat_pts[:, k]))
                    eval_cache[key] = (
                        np.asarray(values, dtype=np.float64),
                        np.asarray(first_basis, dtype=np.int64),
                    )
                values, first_basis = eval_cache[key]
                local = multi[k] - first_basis
                degree_k = self.degrees[k]
                valid = (local >= 0) & (local <= degree_k)
                gathered = values[point_index, np.clip(local, 0, degree_k)]
                column *= np.where(valid, gathered, 0.0)
            buffer[:, col] = column

        result[...] = buffer.reshape(out_shape)
        return result

    def __repr__(self) -> str:
        """Return a compact developer-friendly representation.

        Returns:
            str: Shows dimension, degrees, level count, and active-function count.
        """
        return (
            f"THBSplineSpace(dim={self.dim}, degrees={self.degrees}, "
            f"num_levels={self.num_levels}, num_active_functions={self._num_active}, "
            f"truncate={self._truncate})"
        )
