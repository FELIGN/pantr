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

import copy
import itertools
import string
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ..grid import HierarchicalGrid
from ._bspline_knot_insertion_core import _compute_oslo_matrix_1d_core
from ._bspline_space_nd import BsplineSpace
from ._thb_eval_core import _combine_tp_values

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


def _check_out_array(
    out: npt.NDArray[np.float64] | npt.NDArray[np.int64],
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    name: str,
) -> None:
    """Validate an output array's shape, dtype, and writeability.

    Args:
        out (npt.NDArray[np.float64] | npt.NDArray[np.int64]): The output array.
        shape (tuple[int, ...]): The required shape.
        dtype (npt.DTypeLike): The required dtype.
        name (str): The parameter name, used in error messages.

    Raises:
        ValueError: If ``out`` has the wrong shape or dtype, or is not writeable.
    """
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {out.shape}.")
    if out.dtype != dtype:
        raise ValueError(f"{name} must have dtype {np.dtype(dtype).name}; got {out.dtype}.")
    if not out.flags.writeable:
        raise ValueError(f"{name} must be writeable.")


def _box_all_true(
    mask: npt.NDArray[np.bool_],
    lo: npt.NDArray[np.int64],
    hi: npt.NDArray[np.int64],
) -> npt.NDArray[np.bool_]:
    """Test, for a batch of axis-aligned boxes, whether ``mask`` is all-``True`` inside.

    Each box ``b`` spans the half-open range ``[lo[b, d], hi[b, d])`` per axis ``d``.
    A summed-area table over ``~mask`` makes each box's all-``True`` test
    (``mask[box].all()`` ⟺ no ``False`` cell in the box) an O(``2**ndim``) lookup per box.
    Table construction is O(``N * ndim``) where ``N`` is the total cell count of ``mask``.

    Args:
        mask (npt.NDArray[np.bool_]): The ``ndim``-dimensional boolean mask.
        lo (npt.NDArray[np.int64]): Box lower corners, shape ``(n_boxes, ndim)``.
        hi (npt.NDArray[np.int64]): Box upper corners (exclusive), shape ``(n_boxes, ndim)``.

    Returns:
        npt.NDArray[np.bool_]: Shape ``(n_boxes,)``; ``True`` where the box is all-``True``.
    """
    dim = mask.ndim
    prefix = np.pad((~mask).astype(np.int64), [(1, 0)] * dim)
    for ax in range(dim):
        prefix = np.cumsum(prefix, axis=ax)
    total = np.zeros(lo.shape[0], dtype=np.int64)
    for corner in itertools.product((0, 1), repeat=dim):
        idx = tuple(hi[:, d] if corner[d] else lo[:, d] for d in range(dim))
        sign = (-1) ** (dim - sum(corner))
        total = total + sign * prefix[idx]
    return np.asarray(total == 0, dtype=np.bool_)


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
    if not np.all(first_cell >= 0):
        raise RuntimeError(
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
    basis is the Kraft selection :cite:p:`kraft1997hierarchical,vuong2011hierarchical`:
    a level-``l`` tensor-product B-spline is active iff its support lies in the
    level-``l`` subdomain :math:`\Omega_l` but not entirely in the finer subdomain
    :math:`\Omega_{l+1}`.

    With ``truncate=True`` (the default) the *truncated* hierarchical basis (THB) is
    built: each active function that straddles a finer-level refinement boundary has its
    components on active finer functions removed (Giannelli-Jüttler-Speleers truncation
    :cite:p:`giannelli2012thb`), restoring the partition of unity.  Only truncated
    functions store a coefficient vector (in the finest tensor-product basis their support
    reaches); untruncated functions remain plain tensor-product B-splines.  With
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

    References:
        Adaptive isogeometric algorithms for hierarchical splines
        :cite:p:`garau2018algorithms`.  Per-element multi-level Bézier extraction
        (used for element assembly and visualization) is provided by
        :class:`~pantr.bspline.MultiLevelExtraction`, following
        :cite:t:`dangella2018multilevel`.

    Attributes:
        _root_space (BsplineSpace): The level-0 tensor-product space.
        _grid (HierarchicalGrid): The active-cell hierarchy (snapshot reference).
        _truncate (bool): Whether the truncated (THB) basis is used; ``False`` for
            the plain hierarchical (HB) basis.
        _regularity (tuple[int | None, ...]): Per-direction continuity used when
            subdividing to build finer levels.
        _level_spaces (tuple[BsplineSpace, ...]): Per-level tensor-product spaces;
            index ``l`` is the root subdivided to level ``l``.
        _support (tuple): Per-level, per-direction function-to-cell support
            arrays; ``_support[level][k]`` is the
            ``(first_basis, first_cell, last_cell)`` int64 triple
            (a ``_Support1D``) for direction ``k`` at ``level``.
        _active_funcs (tuple[npt.NDArray[np.int64], ...]): Per-level sorted flat
            (C-order) indices of the active tensor-product functions.
        _func_offset (npt.NDArray[np.int64]): Per-level global-dof base; length
            ``num_levels + 1`` (cumulative active-function counts).
        _num_active (int): Total number of active hierarchical functions.
        _grid_snapshot (tuple[int, int, int]): ``(max_level, num_cells, version)``
            captured at construction; used to detect post-construction grid
            mutations (the grid's :attr:`~pantr.grid.HierarchicalGrid.version`
            counter catches mutations the other two cannot distinguish).
        _trunc (dict): Map from global dof (``int``) to ``_TruncCoeffs``;
            only truncated functions appear (empty when ``truncate=False``).
    """

    __slots__ = (
        "_active_funcs",
        "_contrib_cache",
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
        self._grid_snapshot = (grid.max_level, grid.num_cells, grid.version)
        self._trunc = self._compute_truncated_coeffs() if truncate else {}
        # Lazy per-cell cache of _cell_contributions (populated on first access). The
        # space is an immutable construction-time snapshot, so cached results stay valid.
        self._contrib_cache: dict[int, list[tuple[int, int, tuple[int, ...]]]] = {}

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

            candidates_per_dir: list[npt.NDArray[np.int64]] = []
            for k in range(dim):
                _, first_cell, last_cell = support[k]
                overlaps = (last_cell >= bbox_lo[k]) & (first_cell < bbox_hi[k])
                candidates_per_dir.append(np.nonzero(overlaps)[0].astype(np.int64))

            # Enumerate candidate multi-indices and batch the support-box all-checks via
            # a summed-area table: selected iff the support box lies entirely in the
            # subdomain (Ω_l) but not entirely in the further-refined region.
            mesh = np.meshgrid(*candidates_per_dir, indexing="ij")
            multis = np.stack([m.ravel() for m in mesh], axis=-1)  # (n_cand, dim)
            box_lo = np.empty_like(multis)
            box_hi = np.empty_like(multis)
            for k in range(dim):
                _, first_cell, last_cell = support[k]
                box_lo[:, k] = first_cell[multis[:, k]]
                box_hi[:, k] = last_cell[multis[:, k]] + 1
            in_subdomain = _box_all_true(subdomain, box_lo, box_hi)
            in_refined = _box_all_true(refined, box_lo, box_hi)
            selected = multis[in_subdomain & ~in_refined]
            flats = np.ravel_multi_index([selected[:, k] for k in range(dim)], num_basis)
            active.append(np.sort(flats).astype(np.int64))
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
            contracted = np.tensordot(sub, out, axes=([1], [k]))
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
                    if len(box_lo) != coeffs.ndim:
                        raise RuntimeError(
                            f"_compute_truncated_coeffs: box_lo length {len(box_lo)} "
                            f"!= coeffs.ndim {coeffs.ndim}."
                        )
                    coeffs.flags.writeable = False
                    trunc[offset + pos] = _TruncCoeffs(rep, tuple(box_lo), coeffs)
        return trunc

    def _check_not_stale(self) -> None:
        """Raise if the grid has been modified since this space was constructed.

        Raises:
            RuntimeError: If the grid's ``max_level``, ``num_cells``, or mutation
                ``version`` differs from the snapshot taken at construction.
        """
        current = (self._grid.max_level, self._grid.num_cells, self._grid.version)
        if current != self._grid_snapshot:
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

        Note:
            Results are memoized per ``cid`` in ``self._contrib_cache`` (the space is an
            immutable snapshot).  The returned list is the cached object; callers must
            not mutate it.
        """
        self._check_not_stale()
        cached = self._contrib_cache.get(cid)
        if cached is not None:
            return cached
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
        self._contrib_cache[cid] = contribs
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
    def num_total_basis(self) -> int:
        """Get the total number of active hierarchical basis functions.

        Mirrors :attr:`~pantr.bspline.BsplineSpace.num_total_basis` (the hierarchical
        basis is not tensor-product, so there is no per-direction ``num_basis``).

        Returns:
            int: Total active-function count across all levels.
        """
        return self._num_active

    @property
    def num_basis_per_level(self) -> tuple[int, ...]:
        """Get the number of active basis functions at each level.

        Returns:
            tuple[int, ...]: Active-function count per level.
        """
        return tuple(int(a.shape[0]) for a in self._active_funcs)

    @property
    def domain(self) -> npt.NDArray[np.float32 | np.float64]:
        """Get the parametric domain bounds.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Shape ``(dim, 2)`` ``[lo, hi]`` per
            direction (from the root space).
        """
        return self._root_space.domain

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the floating-point dtype of the space.

        Returns:
            npt.DTypeLike: Always ``numpy.float64`` (THB evaluation is float64).
        """
        return np.float64

    @property
    def tolerance(self) -> float:
        """Get the numerical tolerance.

        Returns:
            float: The root space's tolerance.
        """
        return self._root_space.tolerance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def level_space(self, level: int) -> BsplineSpace:
        """Return the tensor-product space at ``level``.

        Returns a construction-time snapshot; not affected by subsequent grid
        mutations (no stale check is performed).

        Args:
            level (int): Hierarchy level in ``[0, num_levels)``.

        Returns:
            BsplineSpace: The root space subdivided to ``level``.

        Raises:
            ValueError: If ``level`` is out of range.
        """
        if not (0 <= level < self.num_levels):
            raise ValueError(f"level must be in [0, {self.num_levels - 1}]; got {level!r}.")
        return self._level_spaces[level]

    def active_function_indices(self, level: int) -> npt.NDArray[np.int64]:
        """Return the flat indices of the active functions at ``level``.

        Returns a construction-time snapshot; not affected by subsequent grid
        mutations (no stale check is performed).

        Args:
            level (int): Hierarchy level in ``[0, num_levels)``.

        Returns:
            npt.NDArray[np.int64]: Sorted flat (C-order) level-``level`` function
            indices selected by the Kraft rule.  A fresh copy is returned.

        Raises:
            ValueError: If ``level`` is out of range.
        """
        if not (0 <= level < self.num_levels):
            raise ValueError(f"level must be in [0, {self.num_levels - 1}]; got {level!r}.")
        indices: npt.NDArray[np.int64] = self._active_funcs[level].copy()
        return indices

    def active_basis(self, cid: int) -> npt.NDArray[np.int64]:
        """Return the global dofs of the active functions whose support intersects cell ``cid``.

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

    def restrict(self, cell_ids: npt.ArrayLike) -> THBSplineSpaceRestriction:
        """Return the windowed sub-space over a subset of active cells.

        Windows this space to the root-cell-aligned bounding box of ``cell_ids``: the
        hierarchical grid is restricted (:meth:`pantr.grid.HierarchicalGrid.restrict`),
        the root space is windowed (:meth:`pantr.bspline.BsplineSpace.restrict`), and a
        new :class:`THBSplineSpace` is rebuilt on the sub-grid (re-running the Kraft
        active-function selection and truncation).

        Unlike the tensor-product :meth:`pantr.bspline.BsplineSpace.restrict`, the
        windowed THB basis equals the global one only over the **interior** cells --
        those whose entire (cross-level) function-support-closure lies inside the
        window -- because Kraft selection and truncation depend on the subdomain near
        the window boundary. Callers make the cells they care about interior by padding
        ``cell_ids`` with a support-closure halo.

        Args:
            cell_ids (npt.ArrayLike): Active cell flat ids to span; duplicates ignored.

        Returns:
            THBSplineSpaceRestriction: The windowed :class:`THBSplineSpace` and a
            read-only ``local_to_global_dof`` map; entry ``d`` is the global
            hierarchical dof of local dof ``d`` when the local function matches a
            globally-active function of the same level and multi-index, else ``-1``.
            Values are exact over interior cells; functions near the window boundary
            may map to ``-1``.

        Raises:
            ValueError: If ``cell_ids`` is empty.
            TypeError: If ``cell_ids`` is not integer-valued.
            IndexError: If any cell id is out of range ``[0, grid.num_cells)``.
        """
        self._check_not_stale()
        grid_restr = self._grid.restrict(cell_ids)
        sub_grid = grid_restr.grid
        if not isinstance(sub_grid, HierarchicalGrid):
            raise RuntimeError(
                f"restrict: expected HierarchicalGrid from grid.restrict; "
                f"got {type(sub_grid).__name__!r}. This is a bug in HierarchicalGrid.restrict."
            )
        dim = self.dim
        factor = self._grid.factor

        # Root-cell bounding box of the window (the sub-grid's root spans it exactly).
        r_lo = [
            int(np.searchsorted(self._grid.root.breakpoints[k], sub_grid.root.breakpoints[k][0]))
            for k in range(dim)
        ]
        r_hi = [r_lo[k] + sub_grid.root.cells_per_axis[k] for k in range(dim)]

        # Window the root space to that box and rebuild the THB space on the sub-grid.
        root_ni = self._root_space.num_intervals
        box = [np.arange(r_lo[k], r_hi[k]) for k in range(dim)]
        root_cells = np.ravel_multi_index(
            tuple(m.ravel() for m in np.meshgrid(*box, indexing="ij")), root_ni
        )
        windowed_root = self._root_space.restrict(root_cells).space
        sub_space = THBSplineSpace(
            windowed_root, sub_grid, truncate=self._truncate, regularity=self._regularity
        )

        # Map each sub active function (level, sub_multi) to the global dof of the same
        # (level, sub_multi + per-level window origin), or -1 if not globally active.
        local_to_global_dof = np.full(sub_space.num_total_basis, -1, dtype=np.int64)
        sub_offset = 0
        for level in range(sub_space.num_levels):
            origin = [
                int(self._support[level][k][0][r_lo[k] * factor[k] ** level]) for k in range(dim)
            ]
            glob_num_basis = self._level_spaces[level].num_basis
            glob_active = self._active_funcs[level]
            glob_offset = int(self._func_offset[level])
            sub_active = sub_space.active_function_indices(level)
            sub_num_basis = sub_space.level_space(level).num_basis
            for sub_pos, sub_flat in enumerate(sub_active.tolist()):
                sub_multi = np.unravel_index(sub_flat, sub_num_basis)
                glob_flat = int(
                    np.ravel_multi_index(
                        tuple(int(sub_multi[k]) + origin[k] for k in range(dim)), glob_num_basis
                    )
                )
                gpos = int(np.searchsorted(glob_active, glob_flat))
                if gpos < glob_active.shape[0] and int(glob_active[gpos]) == glob_flat:
                    local_to_global_dof[sub_offset + sub_pos] = glob_offset + gpos
            sub_offset += int(sub_active.shape[0])
        assert sub_offset == sub_space.num_total_basis
        local_to_global_dof.flags.writeable = False
        return THBSplineSpaceRestriction(
            sub_space, local_to_global_dof, grid_restr.local_to_global_cell
        )

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
            # Safety: _tabulate_orders validated flat_pts against cell_lo/hi (one
            # check per dimension via broadcasting) before calling here.  Cell bounds
            # are a strict subset of each level-space's parametric domain, so pts_k
            # is guaranteed in-domain.  Do NOT use validate=False from any other
            # call site without re-verifying this invariant.
            if order == 0:
                values, first_basis = sp1d.tabulate_basis(pts_k, validate=False)
                deriv = np.asarray(values, dtype=np.float64)
            else:
                all_deriv, first_basis = sp1d.tabulate_basis_derivatives(
                    pts_k, order, validate=False
                )
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
            npt.NDArray[np.float64]: Function values of shape ``(num_pts,)``.

        Raises:
            NotImplementedError: If ``self.dim > _EINSUM_MAX_DIM`` (24); the
                einsum subscript scheme requires one letter per axis.
        """
        rep_level, box_lo, coeffs = entry
        dim = self.dim
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
            # local[p, j] = (box_lo[k] + j) - first_basis[p]; gather + mask in one shot.
            local = (box_lo[k] + np.arange(width))[None, :] - first_basis[:, None]
            valid = (local >= 0) & (local <= degree_k)
            gathered = np.take_along_axis(values, np.clip(local, 0, degree_k), axis=1)
            value_mats.append(np.where(valid, gathered, 0.0))

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
        out_basis: npt.NDArray[np.float64] | None,
        out_dofs: npt.NDArray[np.int64] | None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Evaluate the active functions' ``orders`` mixed partial on cell ``cid``.

        Shared implementation for :meth:`tabulate_basis` (``orders`` all zero) and
        :meth:`tabulate_basis_derivatives`.

        Args:
            cid (int): Active cell flat id in ``[0, grid.num_cells)``.
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)`` lying in
                cell ``cid``.  A tolerance of ``1e-12`` is applied at the cell
                boundary; points further outside raise :class:`ValueError`.
            orders (tuple[int, ...]): Per-direction derivative orders.
            out_basis (npt.NDArray[np.float64] | None): Optional output array of shape
                ``(..., K)`` with ``K = active_basis(cid).size``.  Allocated when
                ``None``.
            out_dofs (npt.NDArray[np.int64] | None): Optional output array of shape
                ``(K,)`` for the global dofs.  Allocated when ``None``.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: ``(values, dofs)``
            of shapes ``(..., K)`` and ``(K,)``.

        Raises:
            IndexError: If ``cid`` is out of range ``[0, grid.num_cells)``.
            ValueError: If ``pts`` does not have trailing dimension ``dim``, if any
                point lies outside cell ``cid``, or if ``out_basis``/``out_dofs`` has
                the wrong shape, dtype, or is not writeable.
            RuntimeError: If the grid has been modified since construction.
        """
        contribs = self._cell_contributions(cid)  # validates cid; raises if stale
        n_active = len(contribs)
        dofs = np.array([gdof for gdof, _, _ in contribs], dtype=np.int64)

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

        if out_basis is None:
            result = np.empty(out_shape, dtype=np.float64)
        else:
            _check_out_array(out_basis, out_shape, np.float64, "out_basis")
            result = out_basis

        if out_dofs is None:
            dofs_result = dofs
        else:
            _check_out_array(out_dofs, (n_active,), np.int64, "out_dofs")
            out_dofs[...] = dofs
            dofs_result = out_dofs

        buffer = np.empty((num_pts, n_active), dtype=np.float64)
        eval_cache: _EvalCache = {}
        dim = self.dim
        degrees_arr = np.asarray(self.degrees, dtype=np.int64)
        max_order = int(degrees_arr.max()) + 1

        # Truncated functions are evaluated individually (coefficient contraction);
        # untruncated ones are grouped by level and combined in a single batched kernel
        # call (the common, hot case).
        untrunc_by_level: dict[int, tuple[list[int], list[tuple[int, ...]]]] = {}
        for col, (gdof, level, multi) in enumerate(contribs):
            entry = self._trunc.get(gdof)
            if entry is None:
                cols, multis = untrunc_by_level.setdefault(level, ([], []))
                cols.append(col)
                multis.append(multi)
            else:
                buffer[:, col] = self._truncated_column(entry, orders, flat_pts, eval_cache)

        for level, (cols, multis) in untrunc_by_level.items():
            vals = np.zeros((dim, num_pts, max_order), dtype=np.float64)
            first_basis = np.empty((dim, num_pts), dtype=np.int64)
            for k in range(dim):
                values_k, fb_k = self._basis_1d_cached(level, k, orders[k], flat_pts, eval_cache)
                vals[k, :, : values_k.shape[1]] = values_k
                first_basis[k] = fb_k
            block = _combine_tp_values(
                vals, first_basis, np.asarray(multis, dtype=np.int64), degrees_arr
            )
            buffer[:, cols] = block

        result[...] = buffer.reshape(out_shape)
        return result, dofs_result

    def tabulate_basis(
        self,
        cid: int,
        pts: npt.ArrayLike,
        out_basis: npt.NDArray[np.float64] | None = None,
        out_dofs: npt.NDArray[np.int64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Evaluate the active hierarchical functions on cell ``cid`` at ``pts``.

        Untruncated functions are a single tensor-product B-spline (the product of
        their 1D B-spline values).  Truncated functions are evaluated from their
        stored coefficients in the finest tensor-product basis their support reaches.
        The returned columns are ordered as ``dofs`` (the sorted global dofs, equal to
        :meth:`active_basis`); a listed truncated function may evaluate to exactly zero
        on the cell.  Mirrors the ``(basis, first_basis)`` two-return of
        :meth:`~pantr.bspline.BsplineSpace.tabulate_basis`.

        Args:
            cid (int): Active cell flat id in ``[0, grid.num_cells)``.
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)`` lying in
                cell ``cid``.  Points outside the cell's bounds raise
                :class:`ValueError`.
            out_basis (npt.NDArray[np.float64] | None): Optional output array of shape
                ``(..., K)`` with ``K = active_basis(cid).size``.  Allocated when
                ``None``.
            out_dofs (npt.NDArray[np.int64] | None): Optional output array of shape
                ``(K,)`` for the dofs.  Allocated when ``None``.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: ``(values, dofs)`` of
            shapes ``(..., K)`` and ``(K,)``.

        Raises:
            IndexError: If ``cid`` is out of range ``[0, grid.num_cells)``.
            ValueError: If ``pts`` does not have trailing dimension ``dim``, if any
                point lies outside the bounds of cell ``cid``, or if ``out_basis`` /
                ``out_dofs`` has the wrong shape, dtype, or is not writeable.
            RuntimeError: If the grid has been modified since construction.
        """
        return self._tabulate_orders(cid, pts, (0,) * self.dim, out_basis, out_dofs)

    def tabulate_basis_derivatives(
        self,
        cid: int,
        pts: npt.ArrayLike,
        orders: int | Sequence[int],
        out_basis: npt.NDArray[np.float64] | None = None,
        out_dofs: npt.NDArray[np.int64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        r"""Evaluate a mixed partial derivative of the active functions on cell ``cid``.

        Computes the single mixed partial :math:`\partial^{orders}` of each active
        hierarchical function, where ``orders[k]`` is the derivative order in
        parametric direction ``k`` (derivatives are with respect to the parametric
        coordinates).  Untruncated functions differentiate as a tensor product of 1D
        B-spline derivatives; truncated functions apply their stored coefficients to
        the B-spline derivatives at their representation level.  The returned columns
        are ordered as ``dofs`` (the sorted global dofs, equal to :meth:`active_basis`).

        Args:
            cid (int): Active cell flat id in ``[0, grid.num_cells)``.
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)`` lying in
                cell ``cid``.  Points outside the cell's bounds raise
                :class:`ValueError`.
            orders (int | Sequence[int]): Per-direction derivative orders.  A scalar
                is broadcast to every direction.  Each entry must be ``>= 0``; orders
                exceeding the degree yield zero.
            out_basis (npt.NDArray[np.float64] | None): Optional output array of shape
                ``(..., K)`` with ``K = active_basis(cid).size``.  Allocated when
                ``None``.
            out_dofs (npt.NDArray[np.int64] | None): Optional output array of shape
                ``(K,)`` for the dofs.  Allocated when ``None``.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: ``(values, dofs)`` of
            shapes ``(..., K)`` and ``(K,)``.

        Raises:
            IndexError: If ``cid`` is out of range ``[0, grid.num_cells)``.
            ValueError: If ``orders`` has the wrong length or a negative entry, if
                ``pts`` does not have trailing dimension ``dim``, if any point lies
                outside cell ``cid``, or if ``out_basis`` / ``out_dofs`` has the wrong
                shape, dtype, or is not writeable.
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
        return self._tabulate_orders(cid, pts, orders_t, out_basis, out_dofs)

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def refine(
        self,
        cell_ids: npt.ArrayLike,
        *,
        admissible_class: int | None = 2,
    ) -> THBSplineSpace:
        """Return a new space with the marked cells refined.

        This method does not mutate ``self`` or its grid: a fresh grid is refined
        and a new :class:`THBSplineSpace` is built; ``self`` and its grid are unchanged.

        With ``admissible_class=m`` (the default ``m=2``) the refinement is graded so
        the resulting mesh is admissible of class ``m`` (the truncated functions
        acting on any cell span at most ``m`` successive levels), following the
        recursive refinement-neighborhood algorithm of Carraturo et al. (2019).  This
        assumes the current mesh is already admissible of class ``m`` (true for the
        root and for any mesh built via graded :meth:`refine`).  With
        ``admissible_class=None`` exactly the marked cells are refined (no grading).

        Args:
            cell_ids (npt.ArrayLike): Flat ids of active cells to refine.
            admissible_class (int | None): Admissibility class ``m >= 2`` to maintain,
                or ``None`` for ungraded refinement.  Defaults to ``2``.

        Returns:
            THBSplineSpace: A new space on the refined grid (same ``root_space``,
            ``truncate``, and ``regularity``).

        Raises:
            IndexError: If any id is outside ``[0, grid.num_cells)``.
            ValueError: If ``admissible_class`` is an integer ``< 2``.
            RuntimeError: If the grid has been modified since construction.
        """
        self._check_not_stale()
        if admissible_class is not None and admissible_class < 2:  # noqa: PLR2004
            raise ValueError(
                f"admissible_class must be an integer >= 2 or None; got {admissible_class!r}."
            )
        ids = np.unique(np.asarray(cell_ids, dtype=np.int64).ravel())
        bad = [int(x) for x in ids if int(x) < 0 or int(x) >= self._grid.num_cells]
        if bad:
            raise IndexError(
                f"cell_ids must lie in [0, {self._grid.num_cells}); got out-of-range id(s): {bad}."
            )
        # Convert to (level, midx) on the original grid before any refinement, since
        # flat ids are reassigned by every grid.refine call.
        marked = [(self._grid.cell_level(int(c)), self._grid.cell_multi_index(int(c))) for c in ids]
        grid_copy = copy.deepcopy(self._grid)
        for level, midx in marked:
            if admissible_class is None:
                if grid_copy.is_active_leaf(level, midx):
                    grid_copy.refine(level, list(midx), [i + 1 for i in midx])
            else:
                self._refine_recursive(grid_copy, level, midx, admissible_class)
        return THBSplineSpace(
            self._root_space,
            grid_copy,
            truncate=self._truncate,
            regularity=self._regularity,
        )

    def _refine_recursive(
        self,
        grid: HierarchicalGrid,
        level: int,
        midx: tuple[int, ...],
        m: int,
    ) -> None:
        """Refine cell ``(level, midx)`` on ``grid``, grading for class-``m`` admissibility.

        Refines every cell in the refinement neighborhood (recursively, at the coarser
        level ``level - m + 1``) before subdividing ``(level, midx)``, per Algorithm 4
        of Carraturo et al. (2019).

        Args:
            grid (HierarchicalGrid): The (mutable) grid copy being refined.
            level (int): Level of the cell to refine.
            midx (tuple[int, ...]): Per-axis index of the cell at ``level``.
            m (int): Admissibility class (``>= 2``).

        Raises:
            RecursionError: Unreachable in practice — recursion depth is bounded by
                ``level <= grid.max_level``, which is bounded by available memory long
                before Python's default recursion limit.
        """
        for nlevel, nmidx in self._refinement_neighborhood(level, midx, m, grid):
            self._refine_recursive(grid, nlevel, nmidx, m)
        if grid.is_active_leaf(level, midx):
            grid.refine(level, list(midx), [i + 1 for i in midx])

    def _refinement_neighborhood(
        self,
        level: int,
        midx: tuple[int, ...],
        m: int,
        grid: HierarchicalGrid,
    ) -> list[tuple[int, tuple[int, ...]]]:
        """Return the refinement neighborhood of cell ``(level, midx)`` for class ``m``.

        Implements Definition 3.4 of Carraturo et al. (2019). Finds all cells at level
        ``level - m + 1`` that are parents of a level-``level - m + 2`` cell touched by
        any B-spline whose support covers the containing cell of ``(level, midx)`` at
        level ``level - m + 2``.

        Args:
            level (int): Level of the cell.
            midx (tuple[int, ...]): Per-axis index of the cell at ``level``.
            m (int): Admissibility class (``>= 2``, so ``level - m + 2 <= level``).
            grid (HierarchicalGrid): The grid copy whose active set is queried.

        Returns:
            list[tuple[int, tuple[int, ...]]]: ``(level - m + 1, parent_midx)`` cells in
            the neighborhood that are currently active leaves.
        """
        dim = self.dim
        factor = self._grid.factor
        k_nbr = level - m + 1
        if k_nbr < 0:
            return []
        k_ext = level - m + 2  # = k_nbr + 1; <= level because m >= 2
        # k_ext < len(self._support) because level <= original max_level = num_levels - 1
        assert k_ext < len(self._support), (
            f"k_ext={k_ext} out of range; level={level}, m={m}, num_levels={self.num_levels}"
        )
        support_ext = self._support[k_ext]
        # Containing cell of (level, midx) at level k_ext.
        q = tuple(midx[d] // factor[d] ** (level - k_ext) for d in range(dim))
        parent_ranges = []
        for d in range(dim):
            first_basis, first_cell, last_cell = support_ext[d]
            fb = int(first_basis[q[d]])
            s_lo = int(first_cell[fb])
            s_hi = int(last_cell[fb + self.degrees[d]]) + 1
            parent_ranges.append(range(s_lo // factor[d], (s_hi - 1) // factor[d] + 1))
        return [
            (k_nbr, p) for p in itertools.product(*parent_ranges) if grid.is_active_leaf(k_nbr, p)
        ]

    def coarsen(
        self,
        cell_ids: npt.ArrayLike,
        *,
        admissible_class: int | None = 2,
    ) -> THBSplineSpace:
        """Return a new space with the marked cells coarsened away.

        A parent cell is reactivated (its children removed) only when **all** of its
        children are marked active leaves, mirroring the coarsening algorithm of
        Carraturo et al. (2019, Alg. 5).  With ``admissible_class=None`` this is the
        exact inverse of :meth:`refine`: ``space.refine(cells).coarsen(children_of(cells))``
        recovers ``space``.  With ``admissible_class=m`` the guard may suppress some
        coarsenings, so the recovery holds only when the guard permits them all.

        With ``admissible_class=m`` (the default ``m=2``) a parent is reactivated only
        if its coarsening neighborhood (Def. 3.5) is empty, so the resulting mesh stays
        admissible of class ``m``.  With ``admissible_class=None`` that guard is skipped.

        The space is immutable: a fresh grid is coarsened and a new
        :class:`THBSplineSpace` is built; ``self`` and its grid are unchanged.
        An empty ``cell_ids`` is valid and returns an unchanged copy of the space.

        Args:
            cell_ids (npt.ArrayLike): Flat ids of active leaf cells to coarsen away.
                An empty array is valid and produces an unchanged copy.
            admissible_class (int | None): Admissibility class ``m >= 2`` to maintain,
                or ``None`` to skip the admissibility guard.  Defaults to ``2``.

        Returns:
            THBSplineSpace: A new space on the coarsened grid (same ``root_space``,
            ``truncate``, and ``regularity``).

        Raises:
            IndexError: If any id is outside ``[0, grid.num_cells)``.
            ValueError: If ``admissible_class`` is an integer ``< 2``.
            RuntimeError: If the grid has been modified since construction.
        """
        self._check_not_stale()
        if admissible_class is not None and admissible_class < 2:  # noqa: PLR2004
            raise ValueError(
                f"admissible_class must be an integer >= 2 or None; got {admissible_class!r}."
            )
        ids = np.unique(np.asarray(cell_ids, dtype=np.int64).ravel())
        bad = [int(x) for x in ids if int(x) < 0 or int(x) >= self._grid.num_cells]
        if bad:
            raise IndexError(
                f"cell_ids must lie in [0, {self._grid.num_cells}); got out-of-range id(s): {bad}."
            )
        dim = self.dim
        factor = self._grid.factor
        marked = {(self._grid.cell_level(int(c)), self._grid.cell_multi_index(int(c))) for c in ids}
        parents = {
            (level - 1, tuple(midx[d] // factor[d] for d in range(dim)))
            for level, midx in marked
            if level >= 1
        }
        grid_copy = copy.deepcopy(self._grid)
        for parent_level, pmidx in sorted(parents, key=lambda pc: -pc[0]):
            children = [
                tuple(pmidx[d] * factor[d] + off[d] for d in range(dim))
                for off in itertools.product(*(range(factor[d]) for d in range(dim)))
            ]
            if not all(grid_copy.is_active_leaf(parent_level + 1, c) for c in children):
                continue
            if not all((parent_level + 1, c) in marked for c in children):
                continue
            if admissible_class is not None and not self._coarsening_neighborhood_empty(
                parent_level, pmidx, admissible_class, grid_copy
            ):
                continue
            grid_copy.coarsen(parent_level, list(pmidx), [p + 1 for p in pmidx])
        return THBSplineSpace(
            self._root_space,
            grid_copy,
            truncate=self._truncate,
            regularity=self._regularity,
        )

    def _coarsening_neighborhood_empty(
        self,
        parent_level: int,
        pmidx: tuple[int, ...],
        m: int,
        grid: HierarchicalGrid,
    ) -> bool:
        """Return whether the coarsening neighborhood of a parent is empty (Def. 3.5).

        The neighborhood is the set of active cells at level ``parent_level + m``
        contained in the multilevel support extension (at level ``parent_level + 1``)
        of the parent's children.  When it is empty, reactivating the parent preserves
        class-``m`` admissibility (Carraturo et al. 2019).

        Args:
            parent_level (int): Level of the parent being considered for coarsening.
            pmidx (tuple[int, ...]): Per-axis index of the parent at ``parent_level``.
            m (int): Admissibility class (``>= 2``).
            grid (HierarchicalGrid): The grid copy whose active set is queried.

        Returns:
            bool: ``True`` iff no active cell at level ``parent_level + m`` lies in the
            support extension of the parent's children.

        Note:
            Assumes ``parent_level + 1 < self.num_levels`` and ``m >= 2``; both are
            guaranteed by the calling context in :meth:`coarsen`.  No input validation
            is performed.
        """
        dim = self.dim
        factor = self._grid.factor
        support = self._support[parent_level + 1]
        ext_lo: list[int] = []
        ext_hi: list[int] = []
        for d in range(dim):
            first_basis, first_cell, last_cell = support[d]
            c_lo = pmidx[d] * factor[d]
            c_hi = (pmidx[d] + 1) * factor[d]
            fmin = int(first_basis[c_lo])
            fmax = int(first_basis[c_hi - 1]) + self.degrees[d]
            ext_lo.append(int(first_cell[fmin]))
            ext_hi.append(int(last_cell[fmax]) + 1)
        target = parent_level + m
        if target > grid.max_level:
            return True
        box_lo = [ext_lo[d] * factor[d] ** (m - 1) for d in range(dim)]
        box_hi = [ext_hi[d] * factor[d] ** (m - 1) for d in range(dim)]
        for blk_lo, blk_hi in grid.active_blocks(target):
            if all(max(box_lo[d], blk_lo[d]) < min(box_hi[d], blk_hi[d]) for d in range(dim)):
                return False
        return True

    # ------------------------------------------------------------------
    # Prolongation
    # ------------------------------------------------------------------

    def _dof_level(self, dof: int) -> int:
        """Return the hierarchy level that owns global active-function ``dof``.

        Args:
            dof (int): Global active-function index. Caller must ensure
                ``0 <= dof < num_total_basis``; out-of-range values produce a
                nonsensical level without raising.

        Returns:
            int: The level whose dof range (per ``_func_offset``) contains ``dof``.
        """
        return int(np.searchsorted(self._func_offset, dof, side="right")) - 1

    def _finest_tp_coeffs(
        self,
        dof: int,
        oslo: tuple[tuple[npt.NDArray[np.float64], ...], ...],
        target_level: int,
    ) -> tuple[list[int], npt.NDArray[np.float64]]:
        """Express active function ``dof`` in the level-``target_level`` TP basis.

        Takes the function's native representation (a single B-spline for untruncated
        functions, the stored coefficients for truncated ones) and refines it purely
        (two-scale, no truncation) up to ``target_level``.

        Args:
            dof (int): Global active-function index.
            oslo (tuple[tuple[npt.NDArray[np.float64], ...], ...]): Per-level,
                per-direction two-scale matrices; must be indexed from at least
                ``0`` through ``target_level - 1``.  Only ``oslo[start..target_level-1]``
                is accessed, where ``start`` is the dof's native or representation level.
            target_level (int): Level whose TP basis the result is expressed in.

        Returns:
            tuple[list[int], npt.NDArray[np.float64]]: ``(box_lo, coeffs)`` over the
            level-``target_level`` function box.
        """
        dim = self.dim
        level = self._dof_level(dof)
        pos = dof - int(self._func_offset[level])
        flat = int(self._active_funcs[level][pos])
        entry = self._trunc.get(dof)
        if entry is None:
            multi = np.unravel_index(flat, self._level_spaces[level].num_basis)
            box_lo = [int(multi[d]) for d in range(dim)]
            box_hi = [int(multi[d]) + 1 for d in range(dim)]
            coeffs = np.ones((1,) * dim, dtype=np.float64)
            start = level
        else:
            start = entry.rep_level
            box_lo = list(entry.box_lo)
            box_hi = [entry.box_lo[d] + entry.coeffs.shape[d] for d in range(dim)]
            coeffs = entry.coeffs
        for lvl in range(start, target_level):
            coeffs, box_lo, box_hi = self._refine_box(coeffs, box_lo, box_hi, oslo[lvl])
        return box_lo, coeffs

    def prolongation_to(self, fine: THBSplineSpace) -> npt.NDArray[np.float64]:
        """Return the prolongation matrix from this space to a refinement ``fine``.

        The hierarchical spaces are nested (``V_h ⊆ V_h'``), so every function of this
        (coarse) space lies in ``fine``.  The returned matrix ``P`` maps a
        coefficient vector in this space's basis to the coefficients of the **same
        function** in ``fine``'s basis: if ``u`` are coarse coefficients, ``P @ u`` are
        the fine coefficients.

        It is built by expressing every coarse and every fine basis function in the
        common finest tensor-product basis of ``fine`` and solving the (consistent)
        linear systems in the least-squares sense.

        Args:
            fine (THBSplineSpace): A refinement of this space (same ``root_space``,
                ``factor``, ``regularity``, and ``truncate``; more levels / refined
                cells).

        Returns:
            npt.NDArray[np.float64]: Matrix ``P`` of shape
            ``(fine.num_total_basis, self.num_total_basis)``.

        Raises:
            TypeError: If ``fine`` is not a :class:`THBSplineSpace`.
            ValueError: If ``fine`` is not a refinement of this space (mismatched
                root/factor/regularity/truncation, fewer levels, or the prolongation
                residual is non-negligible).
        """
        if not isinstance(fine, THBSplineSpace):
            raise TypeError(f"fine must be a THBSplineSpace; got {type(fine).__name__!r}.")
        mismatches: list[str] = []
        if fine.dim != self.dim:
            mismatches.append(f"dim: self={self.dim} vs fine={fine.dim}")
        if fine._truncate != self._truncate:
            mismatches.append(f"truncate: self={self._truncate} vs fine={fine._truncate}")
        if tuple(fine._grid.factor) != tuple(self._grid.factor):
            mismatches.append(f"factor: self={self._grid.factor} vs fine={fine._grid.factor}")
        if fine._regularity != self._regularity:
            mismatches.append(f"regularity: self={self._regularity} vs fine={fine._regularity}")
        if fine.num_levels < self.num_levels:
            mismatches.append(
                f"fine.num_levels={fine.num_levels} < self.num_levels={self.num_levels}"
            )
        if fine.dim == self.dim and not all(
            np.array_equal(fine._root_space.spaces[k].knots, self._root_space.spaces[k].knots)
            for k in range(self.dim)
        ):
            mismatches.append("root knot vectors differ")
        if mismatches:
            raise ValueError(
                "fine must be a refinement of this space; mismatches: "
                + "; ".join(mismatches)
                + "."
            )

        target = fine.num_levels - 1
        oslo = fine._build_oslo_matrices()
        num_basis_finest = fine.level_space(target).num_basis

        def column_flats(
            space: THBSplineSpace, dof: int
        ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
            box_lo, coeffs = space._finest_tp_coeffs(dof, oslo, target)
            ranges = [np.arange(box_lo[d], box_lo[d] + coeffs.shape[d]) for d in range(self.dim)]
            mesh = np.meshgrid(*ranges, indexing="ij")
            flats = np.ravel_multi_index([g.ravel() for g in mesh], num_basis_finest)
            return flats, coeffs.ravel()

        coarse_cols = [column_flats(self, i) for i in range(self.num_total_basis)]
        fine_cols = [column_flats(fine, j) for j in range(fine.num_total_basis)]

        touched = sorted(
            {int(f) for flats, _ in (*coarse_cols, *fine_cols) for f in flats.tolist()}
        )
        touched_arr = np.array(touched, dtype=np.int64)
        n_rows = touched_arr.shape[0]

        coarse_mat = np.zeros((n_rows, self.num_total_basis), dtype=np.float64)
        for i, (flats, vals) in enumerate(coarse_cols):
            coarse_mat[np.searchsorted(touched_arr, flats), i] = vals
        fine_mat = np.zeros((n_rows, fine.num_total_basis), dtype=np.float64)
        for j, (flats, vals) in enumerate(fine_cols):
            fine_mat[np.searchsorted(touched_arr, flats), j] = vals

        solution, *_ = np.linalg.lstsq(fine_mat, coarse_mat, rcond=None)
        prolongation: npt.NDArray[np.float64] = np.asarray(solution, dtype=np.float64)
        diff = np.abs(fine_mat @ prolongation - coarse_mat)
        residual = float(diff.max()) if diff.size > 0 else 0.0
        if residual > 1e-8 * (1.0 + float(np.abs(coarse_mat).max())):
            raise ValueError(
                f"fine is not a refinement of this space (prolongation residual {residual:.2e})."
            )
        return prolongation

    def restriction_to(self, coarse: THBSplineSpace) -> npt.NDArray[np.float64]:
        """Return the restriction matrix from this space to a coarsening ``coarse``.

        ``self`` must be a refinement of ``coarse``.  The restriction is the algebraic
        pseudo-inverse of the prolongation, ``R = pinv(P)`` with
        ``P = coarse.prolongation_to(self)``.  It is assembly-free (no mass matrix) and
        satisfies ``R @ P == I``, so restricting a prolonged coarse field recovers it
        exactly; for a general fine field ``R @ u_fine`` is the least-squares
        (coefficient-space) projection onto the coarse space.

        Args:
            coarse (THBSplineSpace): A coarsening of this space (``self`` is a
                refinement of ``coarse``).

        Returns:
            npt.NDArray[np.float64]: Matrix ``R`` of shape
            ``(coarse.num_total_basis, self.num_total_basis)``.

        Raises:
            TypeError: If ``coarse`` is not a :class:`THBSplineSpace`.
            ValueError: If ``self`` is not a refinement of ``coarse``.
        """
        if not isinstance(coarse, THBSplineSpace):
            raise TypeError(f"coarse must be a THBSplineSpace; got {type(coarse)!r}.")
        prolongation = coarse.prolongation_to(self)
        restriction: npt.NDArray[np.float64] = np.asarray(
            np.linalg.pinv(prolongation), dtype=np.float64
        )
        return restriction

    def __repr__(self) -> str:
        """Return a compact string representation.

        Returns:
            str: Shows dimension, degrees, level count, active-function count, and
            truncation flag.
        """
        return (
            f"THBSplineSpace(dim={self.dim}, degrees={self.degrees}, "
            f"num_levels={self.num_levels}, num_total_basis={self._num_active}, "
            f"truncate={self._truncate})"
        )


class THBSplineSpaceRestriction(NamedTuple):
    """Result of :meth:`THBSplineSpace.restrict`: a windowed THB space with its maps.

    - ``space`` -- the windowed :class:`THBSplineSpace` rebuilt on the restricted grid;
      its basis equals the global basis pointwise over interior cells (those whose
      function-support-closure lies inside the window).
    - ``local_to_global_dof`` -- read-only ``(space.num_total_basis,)`` map; entry ``d``
      is the global hierarchical dof of local dof ``d`` when the local function matches a
      globally-active function of the same level and multi-index, else ``-1`` (local
      function active in the sub-space but absent from the global active set -- arises
      near the window boundary where Kraft selection may differ). Values are exact over
      interior cells.
    - ``local_to_global_cell`` -- read-only ``(space.grid.num_cells,)`` map; entry ``c``
      is the global flat cell id of local cell ``c``. Same ordering as
      :attr:`space.grid <THBSplineSpace.grid>`.
    """

    space: THBSplineSpace
    local_to_global_dof: npt.NDArray[np.int64]
    local_to_global_cell: npt.NDArray[np.int64]
