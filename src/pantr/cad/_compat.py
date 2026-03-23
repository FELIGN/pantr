"""B-spline compatibility: match degrees and knot vectors across objects.

Provides :func:`compat`, which takes N B-splines and returns new objects
that share the same degree and knot vector along each specified axis.
This is the prerequisite for operations like ``ruled``, ``coons``, and
``join`` that combine control points from different B-splines.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy import typing as npt

from ..bspline import Bspline, BsplineSpace, BsplineSpace1D
from ..bspline._bspline_product import (
    _get_interior_breakpoints_and_mults,
    _lookup_mults_in_space,
)


def _remap_domain_1d(
    space_1d: BsplineSpace1D,
    new_domain: tuple[float, float],
) -> BsplineSpace1D:
    """Affinely remap a 1D B-spline space to a new parameter domain.

    Args:
        space_1d: Original 1D space.
        new_domain: Target domain ``(a_new, b_new)``.

    Returns:
        BsplineSpace1D: New space with remapped knots, same degree.
    """
    a_old, b_old = space_1d.domain
    a_new, b_new = new_domain
    scale = (b_new - a_new) / (b_old - a_old)
    new_knots = a_new + (space_1d.knots - a_old) * scale
    return BsplineSpace1D(new_knots, space_1d.degree, periodic=space_1d.periodic)


def _remap_bspline(bspline: Bspline, axis: int, new_domain: tuple[float, float]) -> Bspline:
    """Remap one axis of a B-spline to a new domain (pure reparametrization).

    Control points are unchanged because this is a parameter-space
    transformation only.

    Args:
        bspline: Input B-spline.
        axis: Parametric axis to remap.
        new_domain: Target domain ``(a_new, b_new)``.

    Returns:
        Bspline: New B-spline with remapped knot vector on *axis*.
    """
    spaces = list(bspline.space.spaces)
    spaces[axis] = _remap_domain_1d(spaces[axis], new_domain)
    new_space = BsplineSpace(spaces)
    return Bspline(new_space, bspline.control_points.copy(), is_rational=bspline.is_rational)


def _merge_breakpoints_n_way(
    bp_list: list[npt.NDArray[np.float32 | np.float64]],
    mult_list: list[npt.NDArray[np.int_]],
    tol: float,
) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.int_]]]:
    """Merge interior breakpoints from N spaces, returning per-space deficits.

    Args:
        bp_list: Interior breakpoints per B-spline (sorted, ascending).
        mult_list: Multiplicities per B-spline, matching *bp_list*.
        tol: Tolerance for coincidence.

    Returns:
        tuple: ``(union_bp, deficits)`` where *union_bp* is the sorted union
        of all breakpoints and *deficits[i]* is an int array of knots-to-insert
        counts for B-spline *i* at each union breakpoint.
    """
    # Collect all breakpoints into a sorted union
    non_empty = [bp for bp in bp_list if bp.size > 0]
    if not non_empty:
        return np.empty(0, dtype=np.float64), [np.empty(0, dtype=np.int_) for _ in bp_list]

    union_bp: npt.NDArray[np.float64] = np.unique(np.concatenate(non_empty)).astype(np.float64)

    # Target multiplicity = element-wise max across all spaces
    target_mults = np.zeros(len(union_bp), dtype=np.int_)
    per_space_mults: list[npt.NDArray[np.int_]] = []
    for bp, mult in zip(bp_list, mult_list, strict=True):
        m = _lookup_mults_in_space(union_bp, bp, mult, tol)
        target_mults = np.maximum(target_mults, m)
        per_space_mults.append(m)

    # Deficit per space = target - current
    deficits = [target_mults - m for m in per_space_mults]
    return union_bp, deficits


def compat(
    *bsplines: Bspline,
    axes: int | Sequence[int] | None = None,
) -> list[Bspline]:
    """Make B-splines compatible along specified parametric axes.

    Returns new B-splines that share the same polynomial degree and
    knot vector along the given axes.  The geometric mapping of each
    B-spline is preserved exactly.

    The algorithm proceeds in three stages:

    1. **Same domain** -- remap knot vectors to the common envelope
       ``[min(starts), max(ends)]`` per axis.
    2. **Same degree** -- elevate each B-spline to the maximum degree
       per axis.
    3. **Merge knots** -- insert knots so all B-splines share the
       same interior breakpoints with the maximum multiplicity.

    Periodic B-splines are converted to open form before processing.

    Args:
        *bsplines: Two or more B-splines to make compatible.
            All must have the same parametric dimension.
        axes: Parametric axes along which to operate.  ``None``
            (default) means all axes.  A single ``int`` or a sequence
            of ``int`` selects specific axes.

    Returns:
        list[Bspline]: New B-splines with identical knot structure
        along the specified axes.

    Raises:
        ValueError: If fewer than 2 B-splines are provided.
        ValueError: If B-splines have different parametric dimensions.
        ValueError: If any axis index is out of range.
    """
    if len(bsplines) < 2:  # noqa: PLR2004
        return list(bsplines)

    dims = {b.dim for b in bsplines}
    if len(dims) != 1:
        raise ValueError(f"All B-splines must have the same parametric dimension, got {dims}.")
    dim = dims.pop()

    # Normalize axes
    if axes is None:
        axes_list = list(range(dim))
    elif isinstance(axes, int):
        axes_list = [axes]
    else:
        axes_list = list(axes)
    for a in axes_list:
        if a < 0 or a >= dim:
            raise ValueError(f"Axis {a} out of range for dim={dim}.")
    if not axes_list:
        return list(bsplines)

    # Convert periodic to open
    results: list[Bspline] = []
    for b in bsplines:
        needs_open = any(b.space.spaces[a].periodic for a in axes_list)
        results.append(b.to_open_bspline() if needs_open else b)

    # Stage 1: Same domain
    results = _same_domain(results, axes_list)

    # Stage 2: Same degree
    results = _same_degree(results, axes_list, dim)

    # Stage 3: Merge knots
    results = _merge_knots(results, axes_list, dim)

    return results


def _same_domain(results: list[Bspline], axes_list: list[int]) -> list[Bspline]:
    """Remap all B-splines to a common domain per axis.

    Args:
        results: List of B-splines.
        axes_list: Axes to process.

    Returns:
        list[Bspline]: B-splines with remapped domains.
    """
    for a in axes_list:
        domains = [
            (float(b.space.spaces[a].domain[0]), float(b.space.spaces[a].domain[1]))
            for b in results
        ]
        common_start = min(d[0] for d in domains)
        common_end = max(d[1] for d in domains)
        common = (common_start, common_end)
        for i, b in enumerate(results):
            if domains[i] != common:
                results[i] = _remap_bspline(b, a, common)
    return results


def _same_degree(results: list[Bspline], axes_list: list[int], dim: int) -> list[Bspline]:
    """Elevate degrees to the maximum per axis.

    Args:
        results: List of B-splines.
        axes_list: Axes to process.
        dim: Parametric dimension.

    Returns:
        list[Bspline]: B-splines with elevated degrees.
    """
    max_degrees = [0] * dim
    for a in axes_list:
        max_degrees[a] = max(b.space.spaces[a].degree for b in results)

    for i, b in enumerate(results):
        increments = tuple(
            max_degrees[a] - b.space.spaces[a].degree if a in axes_list else 0 for a in range(dim)
        )
        if any(inc > 0 for inc in increments):
            results[i] = b.elevate_degree(increments)
    return results


def _merge_knots(results: list[Bspline], axes_list: list[int], dim: int) -> list[Bspline]:
    """Insert knots so all B-splines share the same knot vectors per axis.

    Args:
        results: List of B-splines (already same domain and degree).
        axes_list: Axes to process.
        dim: Parametric dimension.

    Returns:
        list[Bspline]: B-splines with merged knot vectors.
    """
    for a in axes_list:
        tol = results[0].space.spaces[a].tolerance

        # Collect interior breakpoints and multiplicities
        bp_list = []
        mult_list = []
        for b in results:
            bp, mult = _get_interior_breakpoints_and_mults(b.space.spaces[a], tol)
            bp_list.append(bp)
            mult_list.append(mult)

        union_bp, deficits = _merge_breakpoints_n_way(bp_list, mult_list, tol)
        if union_bp.size == 0:
            continue

        for i, deficit in enumerate(deficits):
            knots_to_insert = np.repeat(union_bp, deficit).astype(results[i].dtype)
            if knots_to_insert.size == 0:
                continue
            if dim == 1:
                results[i] = results[i].insert_knots(knots_to_insert)
            else:
                per_dim: list[npt.NDArray[np.float64] | None] = [None] * dim
                per_dim[a] = knots_to_insert
                results[i] = results[i].insert_knots(per_dim)
    return results
