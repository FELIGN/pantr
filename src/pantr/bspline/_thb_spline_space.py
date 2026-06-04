"""Truncated hierarchical B-spline spaces (THB-splines).

This module defines :class:`THBSplineSpace`, a hierarchical spline space built on
a :class:`pantr.grid.HierarchicalGrid`.  It follows the G+Smo *self-evaluating*
model: per level it stores the Kraft selection of active tensor-product B-splines,
and a coefficient vector only for truncated functions.  Untruncated functions are
plain tensor-product B-splines.  Both the truncated (THB, default) and non-truncated
(HB) bases are supported via the ``truncate`` flag.

Main exports:

- :class:`THBSplineSpace`: hierarchical B-spline space on a
  :class:`~pantr.grid.HierarchicalGrid`.
"""

from __future__ import annotations

import itertools
import string
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ..grid import HierarchicalGrid
from ._bspline_knot_insertion_core import _compute_oslo_matrix_1d_core
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


class _TruncCoeffs(NamedTuple):
    """Stored representation of a truncated function.

    ``rep_level`` is the finest level at which the function is expressed.
    ``box_lo[k]`` is the per-direction lower function index of the coefficient
    box; ``coeffs.shape[k] == box_hi[k] - box_lo[k]`` (``box_hi`` is implicit
    in the array shape).  ``coeffs`` holds the function's coefficients in the
    level-``rep_level`` tensor-product basis.
    """

    rep_level: int
    box_lo: tuple[int, ...]
    coeffs: npt.NDArray[np.float64]


class _BasisEval1D(NamedTuple):
    """Cached result of a single 1D basis evaluation.

    ``values`` has shape ``(num_pts, degree + 1)``; ``first_basis`` has shape
    ``(num_pts,)``.  Both come from a single call to
    :meth:`~pantr.bspline.BsplineSpace1D.tabulate_basis`.
    """

    values: npt.NDArray[np.float64]
    first_basis: npt.NDArray[np.int64]


_EvalCache = dict[tuple[int, int, int], _BasisEval1D]
"""Per-call cache of 1D basis evaluations keyed by ``(level, direction, order)``.

``order`` is the derivative order evaluated in that direction (``0`` for values).
"""

_EINSUM_MAX_DIM = 24
"""Maximum parametric dimension supported by the single-letter einsum subscript scheme.

``string.ascii_lowercase`` provides 26 letters; the einsum needs ``dim`` letters for the
coefficient axes plus one for the point axis, leaving a safe ceiling of 24 dimensions.
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
    assert np.all(first_cell >= 0), (
        f"B-spline function(s) with empty support detected at indices "
        f"{np.where(first_cell < 0)[0].tolist()}. This indicates an invalid B-spline space."
    )
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

    With ``truncate=True`` (the default) the *truncated* hierarchical basis (THB) is
    built: each active function that straddles a finer-level refinement boundary has
    its components on active finer functions removed (Giannelli-Jüttler-Speleers
    truncation), restoring the partition of unity.  Only truncated functions store a
    coefficient vector (in the finest tensor-product basis their support reaches);
    untruncated functions remain plain tensor-product B-splines.  With
    ``truncate=False`` the non-truncated hierarchical basis (HB) is built.

    This space is a snapshot of the grid at construction time.  Calling
    :meth:`~pantr.grid.HierarchicalGrid.refine` on the underlying grid after
    construction invalidates this space; subsequent calls to :meth:`active_basis` or
    :meth:`tabulate_basis` will raise :class:`RuntimeError`.  Create a new
    :class:`THBSplineSpace` from the updated grid instead.

    Note:
        :meth:`active_basis` lists functions whose *untruncated* support covers a
        cell; under truncation a few of those may evaluate to exactly zero on the
        cell.  :meth:`tabulate_basis` always returns the correct (possibly zero)
        values.

    Attributes:
        _root_space (BsplineSpace): The level-0 tensor-product space.
        _grid (HierarchicalGrid): The active-cell hierarchy (snapshot reference).
        _truncate (bool): Whether the truncated (THB) basis is used; ``False`` for
            the plain hierarchical (HB) basis.
        _regularity (tuple[int | None, ...]): Per-direction continuity used when
            subdividing to build finer levels.
        _level_spaces (tuple[BsplineSpace, ...]): Per-level tensor-product spaces;
            index ``l`` is the root subdivided to level ``l``.
        _support (tuple): Per-level, per-direction function-to-cell support arrays;
            each entry is a tuple of ``(first_basis, first_cell, last_cell)`` int64
            arrays for one direction at one level.
        _active_funcs (tuple[npt.NDArray[np.int64], ...]): Per-level sorted flat
            (C-order) indices of the active tensor-product functions.
        _func_offset (npt.NDArray[np.int64]): Per-level global-dof base; length
            ``num_levels + 1`` (cumulative active-function counts).
        _num_active (int): Total number of active hierarchical functions.
        _grid_snapshot (tuple[int, int]): ``(max_level, num_cells)`` captured at
            construction; used to detect post-construction grid mutations.
        _trunc (dict): Map from global dof (int) to ``_TruncCoeffs``; only
            truncated functions appear (empty when ``truncate=False``).
    """

    __slots__ = (
        "_active_funcs",
        "_func_offset",
        "_grid",
        "_grid_snapshot",
        "_level_spaces",
        "_num_active",
        "_regularity",
        "_root_space",
        "_support",
        "_trunc",
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
            truncate (bool): If ``True`` (default), build the truncated (THB) basis;
                if ``False``, build the non-truncated hierarchical (HB) basis.
            regularity (int | Sequence[int | None] | None): Per-direction continuity
                at the knots inserted when subdividing to finer levels.  A scalar is
                broadcast to every axis; ``None`` (default) uses maximal smoothness.
                Each non-``None`` entry must satisfy ``-1 <= regularity[k] < degree[k]``.

        Raises:
            TypeError: If ``root_space`` is not a :class:`~pantr.bspline.BsplineSpace`
                or ``grid`` is not a :class:`~pantr.grid.HierarchicalGrid`.
            ValueError: If ``grid`` and ``root_space`` disagree on dimension or on
                the root knot-span grid, if ``regularity`` has the wrong length, or
                if any per-direction regularity value is out of range.
        """
        if not isinstance(root_space, BsplineSpace):
            raise TypeError(
                f"root_space must be a BsplineSpace; got {type(root_space).__name__!r}."
            )
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
        for k, (r, d) in enumerate(zip(reg, root_space.degrees, strict=False)):
            if r is not None and not (-1 <= r < d):
                raise ValueError(
                    f"regularity[{k}]={r!r} must be in [-1, degree[{k}]-1={d - 1}]; got {r!r}."
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
        self._grid_snapshot = (grid.max_level, grid.num_cells)
        self._trunc = self._compute_truncated_coeffs() if truncate else {}

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

        A level-``l`` tensor-product function is selected iff its support lies
        entirely in :math:`\Omega_l` (``subdomain_mask``) but not entirely in the
        further-refined region (``subdomain_mask & ~active_leaf_mask``).

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

    def _build_oslo_matrices(self) -> tuple[tuple[npt.NDArray[np.float64], ...], ...]:
        """Build the per-direction two-scale (Oslo) matrices between levels.

        Entry ``[m][k]`` is the refinement matrix ``alpha`` of shape
        ``(num_basis_{m+1,k}, num_basis_{m,k})`` such that a level-``m`` B-spline
        ``B_i`` equals ``sum_j alpha[j, i] B_j`` in the level-``(m+1)`` basis (the
        identity when ``factor[k] == 1``).

        Returns:
            tuple[tuple[npt.NDArray[np.float64], ...], ...]: Matrices indexed by
            ``[m][k]`` for ``m`` in ``[0, num_levels - 2]``.
        """
        mats: list[tuple[npt.NDArray[np.float64], ...]] = []
        for m in range(self.num_levels - 1):
            per_dir: list[npt.NDArray[np.float64]] = []
            for k in range(self.dim):
                old = self._level_spaces[m].spaces[k]
                new = self._level_spaces[m + 1].spaces[k]
                alpha = _compute_oslo_matrix_1d_core(old.degree, old.knots, new.knots)
                per_dir.append(np.asarray(alpha, dtype=np.float64))
            mats.append(tuple(per_dir))
        return tuple(mats)

    @staticmethod
    def _refine_box(
        coeffs: npt.NDArray[np.float64],
        box_lo: list[int],
        box_hi: list[int],
        oslo_m: tuple[npt.NDArray[np.float64], ...],
    ) -> tuple[npt.NDArray[np.float64], list[int], list[int]]:
        """Refine a dense coefficient box from one level to the next.

        Applies, per direction, the two-scale matrix restricted to the current
        function box, growing the box to the band of non-zero finer functions.

        Args:
            coeffs (npt.NDArray[np.float64]): Coefficients over the current box.
            box_lo (list[int]): Per-direction lower function index of the box.
            box_hi (list[int]): Per-direction upper (exclusive) function index.
            oslo_m (tuple[npt.NDArray[np.float64], ...]): Per-direction two-scale
                matrices for this level transition.

        Returns:
            tuple[npt.NDArray[np.float64], list[int], list[int]]: Refined
            coefficients and fresh lists ``(box_lo, box_hi)`` for the next
            level; the input lists are not modified.

        Raises:
            ValueError: If the Oslo matrix slice for any direction is entirely
                zero, indicating a degenerate box or invalid knot refinement.
        """
        new_lo = list(box_lo)
        new_hi = list(box_hi)
        out = coeffs
        for k in range(out.ndim):
            alpha = oslo_m[k]
            cols = alpha[:, box_lo[k] : box_hi[k]]
            rows = np.nonzero(np.any(cols != 0.0, axis=1))[0]
            if rows.size == 0:
                raise ValueError(
                    f"_refine_box: Oslo matrix slice for direction {k} "
                    f"(columns [{box_lo[k]}:{box_hi[k]}]) is entirely zero — "
                    "degenerate or invalid knot refinement."
                )
            nlo, nhi = int(rows[0]), int(rows[-1]) + 1
            sub = alpha[nlo:nhi, box_lo[k] : box_hi[k]]
            contracted = np.asarray(np.tensordot(sub, out, axes=([1], [k])), dtype=np.float64)
            out = np.moveaxis(contracted, 0, k)
            new_lo[k], new_hi[k] = nlo, nhi
        return out, new_lo, new_hi

    @staticmethod
    def _truncate_box(
        coeffs: npt.NDArray[np.float64],
        box_lo: list[int],
        box_hi: list[int],
        active_at_level: npt.NDArray[np.int64],
        num_basis: tuple[int, ...],
    ) -> bool:
        """Zero basis coefficients at active-function positions (in place); report if any zeroed.

        Args:
            coeffs (npt.NDArray[np.float64]): Coefficients over the box (modified in
                place).
            box_lo (list[int]): Per-direction lower function index of the box.
            box_hi (list[int]): Per-direction upper (exclusive) function index.
            active_at_level (npt.NDArray[np.int64]): Sorted flat indices of the active
                functions at the refined level (the level whose basis ``coeffs`` is
                expressed in).
            num_basis (tuple[int, ...]): Per-direction function counts at the refined
                level.

        Returns:
            bool: ``True`` iff at least one coefficient was zeroed.
        """
        ranges = [np.arange(box_lo[k], box_hi[k]) for k in range(coeffs.ndim)]
        mesh = np.meshgrid(*ranges, indexing="ij")
        flat = np.ravel_multi_index([m.ravel() for m in mesh], num_basis)
        is_active = np.isin(flat, active_at_level).reshape(coeffs.shape)
        if not bool(is_active.any()):
            return False
        coeffs[is_active] = 0.0
        return True

    def _compute_truncated_coeffs(self) -> dict[int, _TruncCoeffs]:
        """Build the truncated-coefficient map for the THB basis.

        For each active function that straddles a finer refinement boundary, the
        function is represented in successively finer bases (two-scale refinement),
        zeroing the components on active finer functions at each level (truncation),
        until its support no longer reaches deeper refinement.  Truncation is applied
        at each level in the support chain, not only the first; a function may be
        truncated against active sets at multiple levels before its support clears all
        refinement.  Untruncated functions are omitted.

        Returns:
            dict[int, _TruncCoeffs]: Map from global dof to ``(rep_level, box_lo,
            coeffs)`` for every truncated function.
        """
        trunc: dict[int, _TruncCoeffs] = {}
        if self.num_levels == 1:
            return trunc
        oslo = self._build_oslo_matrices()
        refined = [
            self._grid.subdomain_mask(m) & ~self._grid.active_leaf_mask(m)
            for m in range(self.num_levels)
        ]
        dim = self.dim
        for level in range(self.num_levels - 1):
            num_basis = self._level_spaces[level].num_basis
            offset = int(self._func_offset[level])
            for pos, flat in enumerate(self._active_funcs[level].tolist()):
                multi = np.unravel_index(int(flat), num_basis)
                box_lo = [int(multi[k]) for k in range(dim)]
                box_hi = [int(multi[k]) + 1 for k in range(dim)]
                coeffs = np.ones((1,) * dim, dtype=np.float64)
                rep = level
                any_zeroed = False
                m = level
                while m + 1 < self.num_levels:
                    support_m = self._support[m]
                    cell_box = tuple(
                        slice(
                            int(support_m[k][1][box_lo[k]]),
                            int(support_m[k][2][box_hi[k] - 1]) + 1,
                        )
                        for k in range(dim)
                    )
                    if not bool(refined[m][cell_box].any()):
                        break
                    coeffs, box_lo, box_hi = self._refine_box(coeffs, box_lo, box_hi, oslo[m])
                    m += 1
                    rep = m
                    zeroed = self._truncate_box(
                        coeffs,
                        box_lo,
                        box_hi,
                        self._active_funcs[m],
                        self._level_spaces[m].num_basis,
                    )
                    any_zeroed = any_zeroed or zeroed
                # A function whose support enters a refined region but whose
                # coefficient box at every finer level has no overlap with active
                # finer functions requires no truncation (remains a plain B-spline).
                if any_zeroed:
                    trunc[offset + pos] = _TruncCoeffs(rep, tuple(box_lo), coeffs)
        return trunc

    def _check_not_stale(self) -> None:
        """Raise if the grid has been modified since this space was constructed.

        Raises:
            RuntimeError: If the grid's ``max_level`` or ``num_cells`` differs from
                the snapshot taken at construction.
        """
        if (self._grid.max_level, self._grid.num_cells) != self._grid_snapshot:
            raise RuntimeError(
                "THBSplineSpace is stale: the underlying HierarchicalGrid has been modified "
                "after construction. Create a new THBSplineSpace from the updated grid."
            )

    def _cell_contributions(self, cid: int) -> list[tuple[int, int, tuple[int, ...]]]:
        """Return the active functions non-zero on cell ``cid``.

        Args:
            cid (int): Active cell flat id in ``[0, grid.num_cells)``.

        Returns:
            list[tuple[int, int, tuple[int, ...]]]: ``(global_dof, level, multi)``
            triples sorted by ``global_dof``, where ``multi`` is the per-axis function
            index tuple (multi-index) in its level space.
        """
        self._check_not_stale()
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
        """Get the number of hierarchy levels at construction time.

        Returns:
            int: Number of levels; stable even if the grid is later refined.
        """
        return len(self._level_spaces)

    @property
    def truncate(self) -> bool:
        """Get whether the hierarchical basis is truncated.

        Returns:
            bool: ``True`` for the truncated (THB) basis, ``False`` for the plain
            hierarchical (HB) basis.
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
            cid (int): Active cell flat id in ``[0, grid.num_cells)``.

        Returns:
            npt.NDArray[np.int64]: Sorted global hierarchical-dof indices of the
            functions whose support intersects cell ``cid``.

        Raises:
            IndexError: If ``cid`` is out of range ``[0, grid.num_cells)``.
            RuntimeError: If the grid has been modified since construction.
        """
        return np.array([dof for dof, _, _ in self._cell_contributions(cid)], dtype=np.int64)

    def _basis_1d_cached(
        self,
        level: int,
        k: int,
        order: int,
        flat_pts: npt.NDArray[np.float64],
        eval_cache: _EvalCache,
    ) -> _BasisEval1D:
        """Evaluate (and cache) the level-``level`` 1D basis (or a derivative).

        Args:
            level (int): Hierarchy level whose 1D space is evaluated.
            k (int): Parametric direction.
            order (int): Derivative order in direction ``k`` (``0`` for values).
            flat_pts (npt.NDArray[np.float64]): All parametric points of shape
                ``(num_pts, dim)``; column ``k`` is used.
            eval_cache (_EvalCache): Per-call cache keyed by ``(level, k, order)``.

        Returns:
            _BasisEval1D: ``(values, first_basis)`` where ``values`` holds the
            ``order``-th derivative of each local basis function (the function
            values when ``order == 0``).
        """
        key = (level, k, order)
        cached = eval_cache.get(key)
        if cached is None:
            sp1d = self._level_spaces[level].spaces[k]
            pts_k = np.ascontiguousarray(flat_pts[:, k])
            if order == 0:
                values, first_basis = sp1d.tabulate_basis(pts_k)
                deriv = np.asarray(values, dtype=np.float64)
            else:
                all_deriv, first_basis = sp1d.tabulate_basis_derivatives(pts_k, order)
                deriv = np.asarray(all_deriv, dtype=np.float64)[:, order, :]
            cached = _BasisEval1D(deriv, np.asarray(first_basis, dtype=np.int64))
            eval_cache[key] = cached
        return cached

    def _truncated_column(
        self,
        entry: _TruncCoeffs,
        orders: tuple[int, ...],
        flat_pts: npt.NDArray[np.float64],
        eval_cache: _EvalCache,
    ) -> npt.NDArray[np.float64]:
        """Evaluate one truncated function (or a derivative) from its coefficients.

        Computes ``sum_multi coeffs[multi] * prod_k D^orders[k] B^rep_{box_lo[k] +
        multi_k}(pt)`` over the stored coefficient box via a tensor contraction,
        where ``D^orders[k]`` is the ``orders[k]``-th derivative in direction ``k``.

        Args:
            entry (_TruncCoeffs): ``(rep_level, box_lo, coeffs)`` for the function.
            orders (tuple[int, ...]): Per-direction derivative orders (all ``0`` for
                function values).
            flat_pts (npt.NDArray[np.float64]): Points of shape ``(num_pts, dim)``.
            eval_cache (_EvalCache): Per-call 1D-basis evaluation cache.

        Returns:
            npt.NDArray[np.float64]: Values of shape ``(num_pts,)``.
        """
        rep_level, box_lo, coeffs = entry
        dim = self.dim
        num_pts = flat_pts.shape[0]
        point_index = np.arange(num_pts)
        if dim > _EINSUM_MAX_DIM:
            raise NotImplementedError(
                f"_truncated_column uses single-letter einsum subscripts; "
                f"only dim <= {_EINSUM_MAX_DIM} is supported, got dim={dim}."
            )
        value_mats: list[npt.NDArray[np.float64]] = []
        for k in range(dim):
            values, first_basis = self._basis_1d_cached(
                rep_level, k, orders[k], flat_pts, eval_cache
            )
            degree_k = self.degrees[k]
            width = coeffs.shape[k]
            vmat = np.zeros((num_pts, width), dtype=np.float64)
            for j in range(width):
                local = (box_lo[k] + j) - first_basis
                valid = (local >= 0) & (local <= degree_k)
                vmat[:, j] = np.where(valid, values[point_index, np.clip(local, 0, degree_k)], 0.0)
            value_mats.append(vmat)

        letters = string.ascii_lowercase
        func_subs = letters[:dim]
        pt_sub = letters[dim]
        subscripts = f"{func_subs},{','.join(pt_sub + func_subs[k] for k in range(dim))}->{pt_sub}"
        column = np.asarray(np.einsum(subscripts, coeffs, *value_mats), dtype=np.float64)
        return column

    def _tabulate_orders(
        self,
        cid: int,
        pts: npt.ArrayLike,
        orders: tuple[int, ...],
        out: npt.NDArray[np.float64] | None,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the active functions' ``orders`` mixed partial on cell ``cid``.

        Shared implementation for :meth:`tabulate_basis` (``orders`` all zero) and
        :meth:`tabulate_basis_derivatives`.

        Args:
            cid (int): Active cell flat id in ``[0, grid.num_cells)``.
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)`` in the cell.
            orders (tuple[int, ...]): Per-direction derivative orders.
            out (npt.NDArray[np.float64] | None): Optional output of shape ``(..., K)``.

        Returns:
            npt.NDArray[np.float64]: Values of shape ``(..., K)``.

        Raises:
            IndexError: If ``cid`` is out of range ``[0, grid.num_cells)``.
            ValueError: If ``pts`` does not have trailing dimension ``dim``, if any
                point lies outside cell ``cid``, or if ``out`` has the wrong shape,
                dtype, or is not writeable.
            RuntimeError: If the grid has been modified since construction.
        """
        contribs = self._cell_contributions(cid)  # validates cid; raises if stale
        n_active = len(contribs)

        pts_arr = np.asarray(pts, dtype=np.float64)
        if pts_arr.ndim == 0 or pts_arr.shape[-1] != self.dim:
            raise ValueError(
                f"pts must have trailing dimension {self.dim}; got shape {pts_arr.shape}."
            )
        lead = pts_arr.shape[:-1]
        num_pts = int(np.prod(lead)) if lead else 1
        flat_pts = pts_arr.reshape(num_pts, self.dim)

        cell_lo, cell_hi = self._grid.cell_bounds(cid)
        _tol = 1e-12
        if not (np.all(flat_pts >= cell_lo - _tol) and np.all(flat_pts <= cell_hi + _tol)):
            raise ValueError(
                f"pts must lie inside cell {cid!r} with bounds lo={cell_lo}, hi={cell_hi}."
            )

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
        eval_cache: _EvalCache = {}
        point_index = np.arange(num_pts)
        for col, (gdof, level, multi) in enumerate(contribs):
            entry = self._trunc.get(gdof)
            if entry is None:
                column = np.ones(num_pts, dtype=np.float64)
                for k in range(self.dim):
                    values, first_basis = self._basis_1d_cached(
                        level, k, orders[k], flat_pts, eval_cache
                    )
                    local = multi[k] - first_basis
                    degree_k = self.degrees[k]
                    valid = (local >= 0) & (local <= degree_k)
                    gathered = values[point_index, np.clip(local, 0, degree_k)]
                    column *= np.where(valid, gathered, 0.0)
                buffer[:, col] = column
            else:
                buffer[:, col] = self._truncated_column(entry, orders, flat_pts, eval_cache)

        result[...] = buffer.reshape(out_shape)
        return result

    def tabulate_basis(
        self,
        cid: int,
        pts: npt.ArrayLike,
        out: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the active hierarchical functions on cell ``cid`` at ``pts``.

        Untruncated functions are a single tensor-product B-spline (the product of
        their 1D B-spline values).  Truncated functions are evaluated from their
        stored coefficients in the finest tensor-product basis their support reaches.
        The returned columns are ordered as :meth:`active_basis` (sorted global dof);
        a listed truncated function may evaluate to exactly zero on the cell.

        Args:
            cid (int): Active cell flat id in ``[0, grid.num_cells)``.
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)`` lying in
                cell ``cid``.  Points outside the cell's bounds raise
                :class:`ValueError`.
            out (npt.NDArray[np.float64] | None): Optional output array of shape
                ``(..., K)`` with ``K = active_basis(cid).size``.  Allocated when
                ``None``.

        Returns:
            npt.NDArray[np.float64]: Function values of shape ``(..., K)``.

        Raises:
            IndexError: If ``cid`` is out of range ``[0, grid.num_cells)``.
            ValueError: If ``pts`` does not have trailing dimension ``dim``, if any
                point lies outside the bounds of cell ``cid``, or if ``out`` has the
                wrong shape, dtype, or is not writeable.
            RuntimeError: If the grid has been modified since construction.
        """
        return self._tabulate_orders(cid, pts, (0,) * self.dim, out)

    def tabulate_basis_derivatives(
        self,
        cid: int,
        pts: npt.ArrayLike,
        orders: int | Sequence[int],
        out: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        r"""Evaluate a mixed partial derivative of the active functions on cell ``cid``.

        Computes the single mixed partial :math:`\partial^{orders}` of each active
        hierarchical function, where ``orders[k]`` is the derivative order in
        parametric direction ``k`` (derivatives are with respect to the parametric
        coordinates).  Untruncated functions differentiate as a tensor product of 1D
        B-spline derivatives; truncated functions apply their stored coefficients to
        the B-spline derivatives at their representation level.  The returned columns
        are ordered as :meth:`active_basis` (sorted global dof).

        Args:
            cid (int): Active cell flat id in ``[0, grid.num_cells)``.
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)`` lying in
                cell ``cid``.  Points outside the cell's bounds raise
                :class:`ValueError`.
            orders (int | Sequence[int]): Per-direction derivative orders.  A scalar
                is broadcast to every direction.  Each entry must be ``>= 0``; orders
                exceeding the degree yield zero.
            out (npt.NDArray[np.float64] | None): Optional output array of shape
                ``(..., K)`` with ``K = active_basis(cid).size``.  Allocated when
                ``None``.

        Returns:
            npt.NDArray[np.float64]: Derivative values of shape ``(..., K)``.

        Raises:
            IndexError: If ``cid`` is out of range ``[0, grid.num_cells)``.
            ValueError: If ``orders`` has the wrong length or a negative entry, if
                ``pts`` does not have trailing dimension ``dim``, if any point lies
                outside cell ``cid``, or if ``out`` has the wrong shape, dtype, or is
                not writeable.
            RuntimeError: If the grid has been modified since construction.
        """
        if isinstance(orders, int):
            orders_t = (orders,) * self.dim
        else:
            orders_t = tuple(int(o) for o in orders)
            if len(orders_t) != self.dim:
                raise ValueError(
                    f"orders must be a scalar or length-{self.dim} sequence; "
                    f"got length {len(orders_t)}."
                )
        if any(o < 0 for o in orders_t):
            raise ValueError(f"orders must be non-negative; got {orders_t!r}.")
        return self._tabulate_orders(cid, pts, orders_t, out)

    def __repr__(self) -> str:
        """Return a compact string representation.

        Returns:
            str: Shows dimension, degrees, level count, active-function count, and
            truncation flag.
        """
        return (
            f"THBSplineSpace(dim={self.dim}, degrees={self.degrees}, "
            f"num_levels={self.num_levels}, num_active_functions={self._num_active}, "
            f"truncate={self._truncate})"
        )
