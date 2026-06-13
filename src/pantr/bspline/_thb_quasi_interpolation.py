r"""Quasi-interpolation onto THB spline spaces (Speleers-Manni, effortless).

This module provides :func:`quasi_interpolate_thb_spline`, the hierarchical
counterpart of :func:`~pantr.bspline.quasi_interpolate_bspline`.  Following
Speleers-Manni (2016), the hierarchical quasi-interpolant is assembled with no
inter-level coupling: each active hierarchical dof receives the coefficient of a
per-level tensor-product quasi-interpolant,

.. math::
    Q_H f = \sum_{\ell} \sum_{\beta \in A_\ell} \lambda^\ell_\beta(f)\,
            \operatorname{trunc}(B^\ell_\beta),

so ``c[d] = λ^{l}_β(f)`` for the active function ``(l, β)`` of global dof ``d``
(truncation already lives in the basis).  The per-level functional is the
Lee-Lyche-Mørken local projector, evaluated on a **level-l active leaf cell** inside
``supp(B^l_β)``.  On such a cell, truncation zeros every coarser active function's
component on the active ``B^l_β``, so ``λ^l_β(s) = c_β`` for any THB spline ``s`` and
``Q_H`` reproduces the THB space exactly.

Main exports:

- :func:`quasi_interpolate_thb_spline`: quasi-interpolate a callable onto a
  :class:`~pantr.bspline.THBSplineSpace`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import numpy as np
from numpy import typing as npt

from ._bspline_quasi_interpolation import (
    QIKind,
    _contract_weights,
    _evaluate_func,
    _interval_interior_points,
    _local_weight_row,
    _tensor_point_grid,
)
from ._bspline_space_factory import get_greville_abscissae
from ._thb_spline import THBSpline
from ._thb_spline_space import THBSplineSpace

if TYPE_CHECKING:
    from collections.abc import Callable


def quasi_interpolate_thb_spline(
    func: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    space: THBSplineSpace,
    *,
    kind: QIKind = "llm",
) -> THBSpline:
    """Quasi-interpolate a callable onto a THB spline space.

    Assembles the Speleers-Manni hierarchical quasi-interpolant: every active dof gets
    the coefficient of the per-level Lee-Lyche-Mørken local projector, sampled on a
    level-``l`` active leaf cell inside the function's support.  For the truncated
    (THB) basis this reproduces any THB spline exactly; for the non-truncated (HB)
    basis it remains a valid local approximant but is not an exact projector.

    Args:
        func (Callable): Function to quasi-interpolate.  Called once on an
            ``(M, dim)`` point array and must return ``(M,)`` (scalar) or
            ``(M, rank)`` (vector-valued).
        space (THBSplineSpace): The target hierarchical space.
        kind (QIKind): Quasi-interpolant kind.  Only ``"llm"`` (Lee-Lyche-Mørken) is
            currently supported.  Defaults to ``"llm"``.

    Returns:
        THBSpline: A THB spline whose evaluation quasi-interpolates ``func``.

    Raises:
        TypeError: If ``space`` is not a :class:`~pantr.bspline.THBSplineSpace`.
        ValueError: If ``kind`` is not recognized, or if ``func`` returns an output
            with an invalid shape (0-D, more than 2-D, or wrong leading dimension).
        RuntimeError: If the grid has been modified since ``space`` was constructed,
            or an active dof has no leaf cell at its level (inconsistent space).
    """
    if not isinstance(space, THBSplineSpace):
        raise TypeError(f"space must be a THBSplineSpace; got {type(space).__name__!r}.")
    if kind not in get_args(QIKind):
        valid = ", ".join(repr(v) for v in get_args(QIKind))
        raise ValueError(f"Unknown kind {kind!r}; expected one of {valid}.")
    space._check_not_stale()

    grid = space.grid
    dim = space.dim
    num_active = space.num_total_basis

    # Candidate leaf cells per dof: cells whose level equals the contributing
    # function's level (i.e. level-l active leaf cells inside supp(B^l_β)).
    candidates: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
    for cid in range(grid.num_cells):
        cell_lvl = grid.cell_level(cid)
        for dof, level, multi in space._cell_contributions(cid):
            if level == cell_lvl:
                candidates.setdefault(dof, []).append((cid, multi))

    greville_cache: dict[tuple[int, int], npt.NDArray[np.float64]] = {}

    def _greville(level: int, k: int) -> npt.NDArray[np.float64]:
        key = (level, k)
        cached = greville_cache.get(key)
        if cached is None:
            cached = np.asarray(
                get_greville_abscissae(space.level_space(level).spaces[k]), dtype=np.float64
            )
            greville_cache[key] = cached
        return cached

    orders = tuple(d + 1 for d in space.degrees)
    block = int(np.prod(orders))
    all_points = np.empty((num_active * block, dim), dtype=np.float64)
    dof_weights: list[list[npt.NDArray[np.float64]]] = [[] for _ in range(num_active)]

    for dof in range(num_active):
        cand = candidates.get(dof)
        if not cand:
            raise RuntimeError(
                f"active dof {dof} has no leaf cell at its level; the THB space is inconsistent."
            )
        level = space._dof_level(dof)
        multi = cand[0][1]
        target = np.array([_greville(level, k)[multi[k]] for k in range(dim)], dtype=np.float64)

        # Pick the candidate leaf cell whose centre is nearest the Greville point
        # (any leaf cell is exact for splines; this limits the local extrapolation).
        best_cid, best_multi = cand[0]
        best_dist = np.inf
        for cell_id, cell_multi in cand:
            lo, hi = grid.cell_bounds(cell_id)
            dist = float(np.linalg.norm(0.5 * (lo + hi) - target))
            if dist < best_dist:
                best_dist, best_cid, best_multi = dist, cell_id, cell_multi
        lo, hi = grid.cell_bounds(best_cid)
        level_space = space.level_space(level)
        per_dir_points: list[npt.NDArray[np.float64]] = []
        for k in range(dim):
            pts = _interval_interior_points(float(lo[k]), float(hi[k]), orders[k])
            per_dir_points.append(pts)
            # best_multi[k] is the per-direction index within the level-l
            # tensor-product basis (not the global hierarchical dof index),
            # which is what _local_weight_row expects.
            dof_weights[dof].append(_local_weight_row(level_space.spaces[k], pts, best_multi[k]))
        all_points[dof * block : (dof + 1) * block] = _tensor_point_grid(per_dir_points)

    values, rank, scalar = _evaluate_func(func, all_points)
    value_tensor = values.reshape(num_active, *orders, rank)

    coeffs = np.empty((num_active, rank), dtype=np.float64)
    for dof in range(num_active):
        coeffs[dof] = _contract_weights(dof_weights[dof], value_tensor[dof])

    if scalar:
        return THBSpline(space, coeffs[:, 0])
    return THBSpline(space, coeffs)
