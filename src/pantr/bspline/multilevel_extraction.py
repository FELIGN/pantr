r"""Multi-level Bézier extraction for truncated hierarchical B-spline spaces.

This module provides :class:`MultiLevelExtraction`, the hierarchical counterpart of
:class:`~pantr.bspline.SpanwiseElementExtraction`.  It exposes, per active cell, the
*multi-level extraction operator* and the *multi-level Bézier extraction operator* of
D'Angella et al. (2017) / D'Angella (2021, ch. 4), which flatten the (truncated)
hierarchical basis on an element into a fixed single-level reference basis.

On an active cell :math:`\epsilon` of level ``L`` the active hierarchical functions
:math:`H^\epsilon` are a linear combination of the level-``L`` tensor-product B-splines
:math:`N^{\epsilon,L}` with support on :math:`\epsilon` (the *multi-level extraction
operator* :math:`M^\epsilon`), and composing with the standard per-element Bézier
extraction :math:`E^\epsilon` gives the *multi-level Bézier extraction* :math:`C^\epsilon`:

.. math::
    H^\epsilon = M^\epsilon N^{\epsilon,L} = M^\epsilon E^\epsilon B = C^\epsilon B,

mapping a fixed Bernstein reference basis ``B`` (on :math:`[0, 1]^d`) to the active
hierarchical functions on the cell.

Note:
    :math:`M^\epsilon` is built from the space's already-truncated coefficients (the
    Giannelli-Jüttler-Speleers truncation, which keeps and refines forward the passive
    functions that straddle a refinement boundary), so it is correct on narrow refinement
    bands.  It does **not** use the activeness-restricted local truncation of
    D'Angella et al. (2017, §3.6.1), which drops such functions; see Eq. 4.7 of the 2021
    thesis for the corrected predicate.

Main exports:

- :class:`MultiLevelExtraction`: per-element multi-level (Bézier) extraction operators.
"""

from __future__ import annotations

import math
from typing import NamedTuple, cast

import numpy as np
import numpy.typing as npt

from ..basis._basis_utils import _allocate_or_validate_out
from ._bspline_knot_insertion_core import _compute_oslo_rows_1d_core
from ._multilevel_extraction_core import _windowed_multilevel_rows
from ._thb_spline_space import THBSplineSpace
from .spanwise_element_extraction import (
    ExtractionTarget,
    SpanwiseElementExtraction,
    TargetLike,
    _coerce_target,
)


class _WindowTables(NamedTuple):
    """Flat per-level tables the windowed extraction kernel reads.

    Every array is read-only.  Sizes are linear in the number of 1D cells per level, not
    in the number of functions of the tensor-product levels.
    """

    factor: npt.NDArray[np.int64]
    """Per-direction subdivision factor, shape ``(dim,)``."""
    degrees: npt.NDArray[np.int64]
    """Per-direction degree, shape ``(dim,)``."""
    num_basis: npt.NDArray[np.int64]
    """Per-level, per-direction function count, shape ``(num_levels, dim)``."""
    first_basis: npt.NDArray[np.int64]
    """Concatenated first non-zero function index per cell, every level and direction."""
    first_basis_offset: npt.NDArray[np.int64]
    """Start of each ``(level, direction)`` in ``first_basis``."""
    two_scale: npt.NDArray[np.float64]
    """Concatenated two-scale blocks restricted to the element windows."""
    two_scale_offset: npt.NDArray[np.int64]
    """Start of each ``(transition, direction)`` in ``two_scale``."""
    active: npt.NDArray[np.int64]
    """Concatenated sorted active flat indices; position is the global dof."""
    func_offset: npt.NDArray[np.int64]
    """Per-level global dof base, shape ``(num_levels + 1,)``."""


def _window_two_scale_blocks(
    space: THBSplineSpace, level: int, direction: int, width: int
) -> npt.NDArray[np.float64]:
    """Return the two-scale blocks of one direction restricted to the element windows.

    For each cell ``j`` of level ``level + 1`` (in direction ``direction``) with parent
    ``j // factor``, the block maps the ``p + 1`` functions of level ``level`` supported
    on the parent to the ``p + 1`` functions of level ``level + 1`` supported on ``j``.
    Built from the banded Oslo rows, so the dense refinement matrix is never formed.

    Args:
        space (THBSplineSpace): The hierarchical space.
        level (int): Coarse level of the transition, in ``[0, num_levels - 1)``.
        direction (int): Parametric direction.
        width (int): Padded block width, ``max(degrees) + 1``.

    Returns:
        npt.NDArray[np.float64]: Blocks of shape ``(num_cells_fine, width, width)``, zero
        outside the leading ``(p + 1) x (p + 1)``.
    """
    coarse = space.level_space(level).spaces[direction]
    fine = space.level_space(level + 1).spaces[direction]
    degree = coarse.degree
    size = degree + 1
    alphas, first_col = _compute_oslo_rows_1d_core(
        degree,
        np.asarray(coarse.knots, dtype=np.float64),
        np.asarray(fine.knots, dtype=np.float64),
    )
    fb_coarse = space._level_support(level)[direction][0]
    fb_fine = space._level_support(level + 1)[direction][0]
    factor = space.grid.factor[direction]
    cells = np.arange(fb_fine.shape[0], dtype=np.int64)
    local = np.arange(size, dtype=np.int64)
    fine_rows = fb_fine[:, None] + local[None, :]  # (cells, size)
    coarse_cols = fb_coarse[cells // factor][:, None] + local[None, :]  # (cells, size)
    band_index = coarse_cols[:, None, :] - first_col[fine_rows][:, :, None]  # (cells, a, b)
    in_band = (band_index >= 0) & (band_index <= degree)
    # `_compute_oslo_rows_1d_core` is typed `Any`; pin the dtype so the annotation holds.
    gathered = np.asarray(
        np.take_along_axis(alphas[fine_rows], np.clip(band_index, 0, degree), axis=2),
        dtype=np.float64,
    )  # (cells, a, b)
    blocks = np.zeros((cells.shape[0], width, width), dtype=np.float64)
    blocks[:, :size, :size] = np.where(in_band, gathered, 0.0)
    return blocks


def _build_window_tables(space: THBSplineSpace) -> _WindowTables:
    """Flatten the per-level data the windowed extraction kernel needs.

    Args:
        space (THBSplineSpace): The hierarchical space.

    Returns:
        _WindowTables: The frozen tables.
    """
    dim = space.dim
    num_levels = space.num_levels
    width = max(space.degrees) + 1

    fb_parts: list[npt.NDArray[np.int64]] = []
    fb_offset = np.empty((num_levels, dim), dtype=np.int64)
    start = 0
    for m in range(num_levels):
        for k in range(dim):
            part = np.asarray(space._level_support(m)[k][0], dtype=np.int64)
            fb_offset[m, k] = start
            start += part.shape[0]
            fb_parts.append(part)

    ts_parts: list[npt.NDArray[np.float64]] = []
    # At least one row so an unrefined space still gets a well-formed 2D array; the
    # kernel never reads it then, since a level-0 cell takes no two-scale step.
    ts_offset = np.zeros((max(num_levels - 1, 1), dim), dtype=np.int64)
    start = 0
    for m in range(num_levels - 1):
        for k in range(dim):
            blocks = _window_two_scale_blocks(space, m, k, width)
            ts_offset[m, k] = start
            start += blocks.shape[0]
            ts_parts.append(blocks)
    two_scale = (
        np.concatenate(ts_parts, axis=0)
        if ts_parts
        else np.zeros((0, width, width), dtype=np.float64)
    )

    tables = _WindowTables(
        factor=np.asarray(space.grid.factor, dtype=np.int64),
        degrees=np.asarray(space.degrees, dtype=np.int64),
        num_basis=np.asarray(
            [space.level_space(m).num_basis for m in range(num_levels)], dtype=np.int64
        ).reshape(num_levels, dim),
        first_basis=np.concatenate(fb_parts),
        first_basis_offset=fb_offset,
        two_scale=two_scale,
        two_scale_offset=ts_offset,
        active=np.concatenate([space.active_function_indices(m) for m in range(num_levels)]).astype(
            np.int64
        ),
        func_offset=np.asarray(space.level_offsets, dtype=np.int64),
    )
    for array in tables:
        array.flags.writeable = False
    return tables


class MultiLevelExtraction:
    r"""Per-element multi-level (Bézier) extraction for a :class:`THBSplineSpace`.

    Mirrors :class:`~pantr.bspline.SpanwiseElementExtraction`: it is constructed from a
    space and a ``target`` reference basis, caches the single-level per-level extractions,
    and exposes per-element operators via :meth:`operator`.  Because hierarchical
    refinement introduces a non-constant number of active functions per cell (and the
    hierarchical basis is not of tensor-product structure), the operators are ragged
    across cells; there is consequently no constant-shape ``tabulate`` / ``ops_1d``.

    For a cell with ``K = active_basis(cid).size`` active functions, degree ``p``, and
    dimension ``d`` (so ``n = (p + 1) ** d`` single-level functions on the cell):

    - :meth:`multilevel_operator` returns :math:`M^\epsilon` of shape ``(K, n)`` mapping
      the level-``L`` tensor-product B-splines on the cell to the active hierarchical
      functions (independent of ``target``).
    - :meth:`operator` returns :math:`C^\epsilon = M^\epsilon E^\epsilon` of shape
      ``(K, n)`` mapping the ``target`` reference basis (Bernstein on :math:`[0, 1]^d`
      for ``"bezier"``) to the active hierarchical functions.

    The operators' rows are ordered as :meth:`active_basis` (sorted global dof), which
    omits the functions that vanish on the cell, so no row is identically zero.

    References:
        Multi-level Bézier extraction for hierarchical local refinement
        :cite:p:`dangella2018multilevel`.

    Attributes:
        _space (THBSplineSpace): The hierarchical space being extracted.
        _target (ExtractionTarget): The single-level reference basis.
        _tables (_WindowTables): Frozen flat tables read by the windowed extraction
            kernel: per-level first-basis indices, the two-scale blocks restricted to
            the element windows, and the active-function index sets.  Their size is
            linear in the 1D cell counts, and no per-cell or per-function state is kept.
        _ext (dict[int, SpanwiseElementExtraction]): Cache of per-level single-level
            extractions, built lazily.
    """

    __slots__ = ("_ext", "_space", "_tables", "_target")

    def __init__(self, space: THBSplineSpace, target: TargetLike = ExtractionTarget.BEZIER) -> None:
        """Create a multi-level extraction for a hierarchical space.

        Args:
            space (THBSplineSpace): The truncated (or non-truncated) hierarchical space.
            target (TargetLike): Single-level reference basis: an
                :class:`ExtractionTarget` or its legacy string spelling.  Defaults to
                :attr:`ExtractionTarget.BEZIER`.

        Raises:
            TypeError: If ``space`` is not a :class:`THBSplineSpace`.
            ValueError: If ``target`` is not a recognized tag.
        """
        if not isinstance(space, THBSplineSpace):
            raise TypeError(f"space must be a THBSplineSpace; got {type(space).__name__!r}.")
        resolved_target = _coerce_target(target)
        self._space = space
        self._target = resolved_target
        self._tables = _build_window_tables(space)
        self._ext: dict[int, SpanwiseElementExtraction] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def space(self) -> THBSplineSpace:
        """Get the underlying hierarchical space.

        Returns:
            THBSplineSpace: The space supplied at construction time.
        """
        return self._space

    @property
    def target(self) -> ExtractionTarget:
        """Get the single-level reference basis.

        Returns:
            ExtractionTarget: The element-local basis each level's extraction maps
                onto.  A string passed at construction is resolved to its enum
                member, so this never returns a string.
        """
        return self._target

    @property
    def dim(self) -> int:
        """Get the parametric dimension.

        Returns:
            int: Number of parametric directions.
        """
        return self._space.dim

    @property
    def dtype(self) -> type[np.float64]:
        """Get the floating-point dtype of the operators.

        Returns:
            type[np.float64]: Always ``numpy.float64``; Oslo matrices and all operators
            are computed in double precision.
        """
        return np.float64

    @property
    def num_elements(self) -> int:
        """Get the number of active cells (elements).

        Returns:
            int: ``space.grid.num_cells``.
        """
        return self._space.grid.num_cells

    def __len__(self) -> int:
        """Return the number of active cells.

        Returns:
            int: ``num_elements``.
        """
        return self.num_elements

    # ------------------------------------------------------------------
    # Per-element operators
    # ------------------------------------------------------------------

    def active_basis(self, cid: int) -> npt.NDArray[np.int64]:
        """Return the global dofs labelling the rows of the operators on cell ``cid``.

        Args:
            cid (int): Active cell flat id in ``[0, num_elements)``.

        Returns:
            npt.NDArray[np.int64]: Sorted global hierarchical-dof indices (the operator
            rows), as returned by :meth:`THBSplineSpace.active_basis`.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        return self._space.active_basis(cid)

    def multilevel_operator(
        self,
        cid: int,
        *,
        out: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        r"""Return the multi-level extraction operator :math:`M^\epsilon` on cell ``cid``.

        :math:`M^\epsilon` (shape ``(K, n)``) maps the level-``L`` tensor-product
        B-splines with support on the cell to the active hierarchical functions
        (``H^\epsilon = M^\epsilon N^{\epsilon,L}``).  Rows follow :meth:`active_basis`;
        columns are the ``(p + 1) ** d`` single-level functions on the cell in C-order.

        Args:
            cid (int): Active cell flat id in ``[0, num_elements)``.
            out (npt.NDArray[np.float64] | None): Optional output array of shape
                ``(K, n)`` where ``K = active_basis(cid).size`` and
                ``n = (p + 1) ** d``.  Allocated when ``None``.

        Built by the windowed per-element recursion: working memory is
        ``(L + 1) * n`` rows of ``n`` doubles on a level-``L`` cell, independent of how
        far the functions' supports reach beyond the cell.

        Returns:
            npt.NDArray[np.float64]: The operator :math:`M^\epsilon`.

        Raises:
            IndexError: If ``cid`` is out of range.
            ValueError: If ``out`` has the wrong shape, dtype, or is not writeable.
            RuntimeError: If the extraction kernel and the space disagree on which
                functions are non-zero on ``cid`` (see ``_active_rows``).
        """
        rows = self._active_rows(cid)
        result = cast(
            npt.NDArray[np.float64],
            _allocate_or_validate_out(out, rows.shape, np.float64),
        )
        result[...] = rows
        return result

    def operator(
        self,
        cid: int,
        *,
        out: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        r"""Return the multi-level Bézier extraction :math:`C^\epsilon` on cell ``cid``.

        :math:`C^\epsilon = M^\epsilon E^\epsilon` (shape ``(K, n)``) maps the ``target``
        reference basis on the cell to the active hierarchical functions
        (``H^\epsilon = C^\epsilon B``).  For the Bézier target, ``B`` is the Bernstein
        basis on :math:`[0, 1]^d`.  Rows follow :meth:`active_basis`.

        Args:
            cid (int): Active cell flat id in ``[0, num_elements)``.
            out (npt.NDArray[np.float64] | None): Optional output array of shape
                ``(K, n)`` where ``K = active_basis(cid).size`` and
                ``n = (p + 1) ** d``.  Allocated when ``None``.

        Returns:
            npt.NDArray[np.float64]: The operator :math:`C^\epsilon`.

        Raises:
            IndexError: If ``cid`` is out of range.
            ValueError: If ``out`` has the wrong shape, dtype, or is not writeable.
            RuntimeError: If the extraction kernel and the space disagree on which
                functions are non-zero on ``cid`` (see ``_active_rows``).
        """
        space = self._space
        level = space.grid.cell_level(cid)
        cell_midx = space.grid.cell_multi_index(cid)
        level_ext = self._level_extraction(level)
        n_in = int(np.prod(level_ext.input_shape_per_dir))
        multilevel = self._active_rows(cid)
        result = cast(
            npt.NDArray[np.float64],
            _allocate_or_validate_out(out, (multilevel.shape[0], n_in), np.float64),
        )
        single_level_f64 = np.asarray(level_ext.operator(cell_midx), dtype=np.float64)
        result[...] = multilevel @ single_level_f64
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _level_extraction(self, level: int) -> SpanwiseElementExtraction:
        """Return (building and caching on first call) the single-level extraction for ``level``.

        Args:
            level (int): Hierarchy level.

        Returns:
            SpanwiseElementExtraction: Extraction of ``space.level_space(level)`` with
            this object's ``target``.
        """
        ext = self._ext.get(level)
        if ext is None:
            ext = SpanwiseElementExtraction(self._space.level_space(level), self._target)
            self._ext[level] = ext
        return ext

    def _active_rows(self, cid: int) -> npt.NDArray[np.float64]:
        r"""Return the rows of :math:`M^\epsilon` for the functions active on cell ``cid``.

        The kernel emits a row for every function whose tensor-product support covers the
        cell, in increasing dof order, with a structural ``nonzero`` flag computed on the
        way; the rows kept are the flagged ones.  :meth:`THBSplineSpace.active_basis`
        decides the same set by a different route (the stored truncated coefficients), and
        ``tests/test_multilevel_extraction.py`` pins that the two agree.  The count is
        compared here as well, against the space's memoized contribution list, and equal
        counts mean equal sets: the kernel's flag is exact reachability over the zero
        pattern of the two-scale table, and the space's coefficients are cancellation-free
        sums over the same table, so a function the kernel flags zero is exactly zero in
        the space too, and the space's set can only be smaller (a positive coefficient
        underflowing to ``0.0``).  A disagreement therefore shows as a count mismatch and
        raises instead of misaligning rows and dofs.

        Args:
            cid (int): Active cell flat id in ``[0, num_elements)``.

        Returns:
            npt.NDArray[np.float64]: Shape ``(K, n)`` with ``K = active_basis(cid).size``.

        Raises:
            IndexError: If ``cid`` is out of range.
            RuntimeError: If the kernel's flagged rows and the space's active functions on
                ``cid`` differ in number.
        """
        rows, _, nonzero = self._windowed_rows(cid)
        kept = rows[nonzero]
        expected = int(self._space.contributions(cid)[0].shape[0])
        if kept.shape[0] != expected:
            raise RuntimeError(
                f"cell {cid}: the extraction kernel flags {kept.shape[0]} non-zero rows but "
                f"the space lists {expected} active functions"
            )
        return kept

    def _windowed_rows(
        self, cid: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
        r"""Run the windowed extraction kernel on one cell.

        Args:
            cid (int): Active cell flat id in ``[0, num_elements)``.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
            ``(rows, dofs, nonzero)`` of shapes ``(R, n)``, ``(R,)`` and ``(R,)``: the
            rows of :math:`M^\epsilon` for every active function whose tensor-product
            support covers the cell, their global dofs in increasing order, and whether
            each row has a non-zero entry (``False`` for a truncated function that
            vanishes on the cell).  The flagged rows are those
            :meth:`THBSplineSpace.active_basis` lists.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        grid = self._space.grid
        level = grid.cell_level(cid)  # validates cid
        cell_multi = np.asarray(grid.cell_multi_index(cid), dtype=np.int64)
        tables = self._tables
        n_single = math.prod(p + 1 for p in self._space.degrees)
        # Every row belongs to a function in one level's window, and there are
        # ``level + 1`` windows of ``n_single`` functions.
        capacity = (level + 1) * n_single
        rows = np.empty((capacity, n_single), dtype=np.float64)
        dofs = np.empty(capacity, dtype=np.int64)
        nonzero = np.empty(capacity, dtype=np.bool_)
        count = _windowed_multilevel_rows(
            level,
            cell_multi,
            tables.factor,
            tables.degrees,
            tables.num_basis,
            tables.first_basis,
            tables.first_basis_offset,
            tables.two_scale,
            tables.two_scale_offset,
            tables.active,
            tables.func_offset,
            self._space.truncate,
            rows,
            dofs,
            nonzero,
        )
        return rows[:count], dofs[:count], nonzero[:count]

    def __repr__(self) -> str:
        """Return a compact string representation.

        Returns:
            str: Shows dimension, target, and element count.
        """
        return (
            f"MultiLevelExtraction(dim={self.dim}, "
            f"target=ExtractionTarget.{self._target.name}, "
            f"num_elements={self.num_elements})"
        )
