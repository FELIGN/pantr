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

from typing import TYPE_CHECKING, get_args

import numpy as np

from ._thb_spline_space import THBSplineSpace
from .spanwise_element_extraction import SpanwiseElementExtraction, Target

if TYPE_CHECKING:
    import numpy.typing as npt


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

    The operators' rows are ordered as :meth:`active_basis` (sorted global dof).

    Attributes:
        _space (THBSplineSpace): The hierarchical space being extracted.
        _target (Target): The single-level reference basis tag.
        _oslo (tuple): Cached per-level, per-direction two-scale matrices.
        _ext (dict[int, SpanwiseElementExtraction]): Cache of per-level single-level
            extractions, built lazily.
    """

    __slots__ = ("_ext", "_oslo", "_space", "_target")

    def __init__(self, space: THBSplineSpace, target: Target = "bezier") -> None:
        """Create a multi-level extraction for a hierarchical space.

        Args:
            space (THBSplineSpace): The truncated (or non-truncated) hierarchical space.
            target (Target): Single-level reference basis, one of ``"bezier"``,
                ``"lagrange"``, ``"cardinal"``.  Defaults to ``"bezier"``.

        Raises:
            TypeError: If ``space`` is not a :class:`THBSplineSpace`.
            ValueError: If ``target`` is not a recognized tag.
        """
        if not isinstance(space, THBSplineSpace):
            raise TypeError(f"space must be a THBSplineSpace; got {type(space).__name__!r}.")
        if target not in get_args(Target):
            valid = ", ".join(repr(v) for v in get_args(Target))
            raise ValueError(f"Unknown target {target!r}; expected one of {valid}.")
        self._space = space
        self._target = target
        self._oslo = space._build_oslo_matrices()
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
    def target(self) -> Target:
        """Get the single-level reference basis tag.

        Returns:
            Target: One of ``"bezier"``, ``"lagrange"``, ``"cardinal"``.
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
    def dtype(self) -> npt.DTypeLike:
        """Get the floating-point dtype of the operators.

        Returns:
            npt.DTypeLike: ``numpy.float64``.
        """
        return np.float64

    @property
    def num_elements(self) -> int:
        """Get the number of active cells (elements).

        Returns:
            int: ``space.grid.num_cells``.
        """
        return self._space.grid.num_cells

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
            RuntimeError: If the grid has been modified since construction.
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
                ``(K, n)``.  Allocated when ``None``.

        Returns:
            npt.NDArray[np.float64]: The operator :math:`M^\epsilon`.

        Raises:
            IndexError: If ``cid`` is out of range.
            ValueError: If ``out`` has the wrong shape, dtype, or is not writeable.
            RuntimeError: If the grid has been modified since construction.
        """
        space = self._space
        grid = space.grid
        level = grid.cell_level(cid)
        cell_midx = grid.cell_multi_index(cid)
        dim = space.dim
        degrees = space.degrees
        support = space._support[level]

        contribs = space._cell_contributions(cid)
        n_active = len(contribs)
        first_basis = [int(support[d][0][cell_midx[d]]) for d in range(dim)]
        n_per = tuple(degrees[d] + 1 for d in range(dim))
        n_single = int(np.prod(n_per))

        result = self._allocate_or_validate(out, (n_active, n_single))
        result[...] = 0.0
        for row, (_, origin_level, multi) in enumerate(contribs):
            box_lo, coeffs = self._element_coeffs(origin_level, multi, level)
            block = np.zeros(n_per, dtype=np.float64)
            src_slices: list[slice] = []
            dst_slices: list[slice] = []
            covered = True
            for d in range(dim):
                offset = first_basis[d] - box_lo[d]
                j0 = max(0, -offset)
                j1 = min(n_per[d], coeffs.shape[d] - offset)
                if j1 <= j0:
                    covered = False
                    break
                dst_slices.append(slice(j0, j1))
                src_slices.append(slice(offset + j0, offset + j1))
            if covered:
                block[tuple(dst_slices)] = coeffs[tuple(src_slices)]
            result[row] = block.ravel()
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
        (``H^\epsilon = C^\epsilon B``).  For ``target="bezier"``, ``B`` is the Bernstein
        basis on :math:`[0, 1]^d`.  Rows follow :meth:`active_basis`.

        Args:
            cid (int): Active cell flat id in ``[0, num_elements)``.
            out (npt.NDArray[np.float64] | None): Optional output array of shape
                ``(K, n)``.  Allocated when ``None``.

        Returns:
            npt.NDArray[np.float64]: The operator :math:`C^\epsilon`.

        Raises:
            IndexError: If ``cid`` is out of range.
            ValueError: If ``out`` has the wrong shape, dtype, or is not writeable.
            RuntimeError: If the grid has been modified since construction.
        """
        space = self._space
        level = space.grid.cell_level(cid)
        cell_midx = space.grid.cell_multi_index(cid)
        multilevel = self.multilevel_operator(cid)
        single_level = self._level_extraction(level).operator(cell_midx)
        product: npt.NDArray[np.float64] = np.asarray(
            multilevel @ np.asarray(single_level, dtype=np.float64), dtype=np.float64
        )
        result = self._allocate_or_validate(out, product.shape)
        result[...] = product
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _level_extraction(self, level: int) -> SpanwiseElementExtraction:
        """Return the cached single-level extraction for ``level``.

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

    def _element_coeffs(
        self,
        origin_level: int,
        multi: tuple[int, ...],
        target_level: int,
    ) -> tuple[list[int], npt.NDArray[np.float64]]:
        """Express a hierarchical function on a cell in the level-``target_level`` basis.

        Refines the originating B-spline ``B^{origin_level}_multi`` from its origin level
        up to ``target_level``, applying the (active-function) truncation at each finer
        level when the space is truncated.  Truncations beyond ``target_level`` do not
        affect values on a level-``target_level`` leaf cell, so this gives the function's
        exact coefficients in the level-``target_level`` tensor-product basis over the
        cell — even when the function's globally-stored representation lives at a finer
        level than ``target_level``.

        Args:
            origin_level (int): Level the function originates at.
            multi (tuple[int, ...]): Per-axis function index at ``origin_level``.
            target_level (int): The cell's level; the basis the result is expressed in.

        Returns:
            tuple[list[int], npt.NDArray[np.float64]]: ``(box_lo, coeffs)`` over the
            level-``target_level`` function box.
        """
        space = self._space
        dim = space.dim
        box_lo = [int(multi[d]) for d in range(dim)]
        box_hi = [int(multi[d]) + 1 for d in range(dim)]
        coeffs = np.ones((1,) * dim, dtype=np.float64)
        for lvl in range(origin_level, target_level):
            coeffs, box_lo, box_hi = THBSplineSpace._refine_box(
                coeffs, box_lo, box_hi, self._oslo[lvl]
            )
            if space._truncate:
                THBSplineSpace._truncate_box(
                    coeffs,
                    box_lo,
                    box_hi,
                    space._active_funcs[lvl + 1],
                    space._level_spaces[lvl + 1].num_basis,
                )
        return box_lo, coeffs

    @staticmethod
    def _allocate_or_validate(
        out: npt.NDArray[np.float64] | None,
        shape: tuple[int, ...],
    ) -> npt.NDArray[np.float64]:
        """Allocate a fresh ``float64`` array or validate a provided ``out``.

        Args:
            out (npt.NDArray[np.float64] | None): Candidate output array, or ``None``.
            shape (tuple[int, ...]): Required shape.

        Returns:
            npt.NDArray[np.float64]: ``out`` (validated) or a fresh array.

        Raises:
            ValueError: If ``out`` has the wrong shape, dtype, or is not writeable.
        """
        if out is None:
            return np.empty(shape, dtype=np.float64)
        if out.shape != shape:
            raise ValueError(f"out must have shape {shape}; got {out.shape}.")
        if out.dtype != np.float64:
            raise ValueError(f"out must have dtype float64; got {out.dtype}.")
        if not out.flags.writeable:
            raise ValueError("out must be writeable.")
        return out

    def __repr__(self) -> str:
        """Return a compact string representation.

        Returns:
            str: Shows dimension, target, and element count.
        """
        return (
            f"MultiLevelExtraction(dim={self.dim}, target={self._target!r}, "
            f"num_elements={self.num_elements})"
        )
