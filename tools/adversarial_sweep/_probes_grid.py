"""Probes for :mod:`pantr.grid`.

``pantr.grid`` is Tier-1 surface for the C++ port: :class:`~pantr.grid.TensorProductGrid`
and :class:`~pantr.grid.HierarchicalGrid` back ``locate_many`` with Numba kernels (per-axis
search and top-down hierarchy descent), and :class:`~pantr.grid.BVH` backs ``query_aabb``
with an iterative stack-based traversal -- both run under ``nopython=True`` with no bounds
checking, so an off-by-one span or a stack overrun corrupts memory silently rather than
raising. The pure-Python arithmetic (cell indexing, facet/neighbour resolution, block
bookkeeping) is lower Numba risk but still worth hostile inputs: it feeds the kernels their
descriptor arrays, and a bug there produces the same silently-wrong cell ids.

Every invariant here is checked against an *independent* oracle -- brute-force overlap
tests via :class:`pantr.geometry.AABB`, geometric adjacency from ``cell_bounds``, or a
closed-form volume/tiling formula -- never a second call into the same algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pantr.bspline import BsplineSpace, BsplineSpace1D
from pantr.geometry import AABB
from pantr.grid import (
    BVH,
    CellTags,
    FacetTags,
    HierarchicalGrid,
    Partition,
    TensorProductGrid,
    cell_quadrature,
    hierarchical_grid,
    overlay,
    partition_grid,
    tensor_product_grid,
    uniform_grid,
)
from pantr.quad import gauss_legendre_quadrature
from pantr.tolerance import get_default

from ._axes import Profile, dims, domains
from ._core import Case, custom, expected_shape

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpy.typing as npt

GROUP = "grid"
"""Registry name of this probe group."""

_EPS64 = float(np.finfo(np.float64).eps)
"""``float64`` machine epsilon; every derived tolerance below is a multiple of this."""

_UNIFORM_SPACING_EPS_FACTOR = 16
"""Mirrors ``pantr.grid._tensor_product_grid._UNIFORM_SPACING_EPS_FACTOR``, the multiple of
``eps`` ``is_uniform`` compares spacing differences against, RELATIVE to the coordinate
scale ``|first| + |last|``. Needed here only to derive a perturbation guaranteed to sit
well above it; not a probe tolerance itself. It replaces a mirror of the absolute ``1e-10``
this code used to carry, and the difference is the point of this group: an absolute
tolerance made the verdict depend on where the domain sat, which is exactly what these
cases sweep."""

# Deepest level probed for HierarchicalGrid refinement. `factor ** level` overflowing
# int64 in `_hier_collect_cell_bounds_core` (`_hier_core.py:328-334`) is already logged
# for level ~63+; capping here at 20 keeps this sweep from re-reporting that known issue.
_MAX_PROBED_LEVEL = 20


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------


def _grid_tag(ndim: int, domain: tuple[float, float]) -> str:
    """Build a short case-label tag from a dimension and a domain.

    Args:
        ndim (int): Spatial dimension.
        domain (tuple[float, float]): ``(lo, hi)`` domain.

    Returns:
        str: ``"d{ndim}_[{lo},{hi}]"``.
    """
    lo, hi = domain
    return f"d{ndim}_[{lo:g},{hi:g}]"


def _uniform_breakpoints(
    ndim: int, domain: tuple[float, float], n_per_axis: int
) -> list[npt.NDArray[np.float64]]:
    """Build identical uniform per-axis breakpoint arrays.

    Args:
        ndim (int): Spatial dimension.
        domain (tuple[float, float]): ``(lo, hi)`` domain, shared by every axis.
        n_per_axis (int): Number of cells per axis.

    Returns:
        list[npt.NDArray[np.float64]]: One ``n_per_axis + 1``-length array per axis.
    """
    lo, hi = domain
    return [np.linspace(lo, hi, n_per_axis + 1, dtype=np.float64) for _ in range(ndim)]


def _tp_grid(ndim: int, domain: tuple[float, float], n_per_axis: int = 3) -> TensorProductGrid:
    """Build a uniform :class:`TensorProductGrid`.

    Args:
        ndim (int): Spatial dimension.
        domain (tuple[float, float]): ``(lo, hi)`` domain, shared by every axis.
        n_per_axis (int): Number of cells per axis. Defaults to 3.

    Returns:
        TensorProductGrid: The constructed grid.
    """
    return TensorProductGrid(_uniform_breakpoints(ndim, domain, n_per_axis))


def _volume(lo: npt.NDArray[np.float64], hi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute per-row axis-aligned box volumes.

    Args:
        lo (npt.NDArray[np.float64]): Lower corners, shape ``(n, ndim)``.
        hi (npt.NDArray[np.float64]): Upper corners, shape ``(n, ndim)``.

    Returns:
        npt.NDArray[np.float64]: Shape ``(n,)`` volumes.
    """
    return np.prod(hi - lo, axis=1)


# ---------------------------------------------------------------------------
# Invariant factories shared across TensorProductGrid and HierarchicalGrid
# ---------------------------------------------------------------------------


def _collect_vs_percell_predicate(
    grid: TensorProductGrid | HierarchicalGrid,
    *,
    exact: bool,
    tol: float = 0.0,
) -> Callable[[object], str | None]:
    """Build a predicate comparing ``collect_cell_bounds()`` to per-cell ``cell_bounds``.

    ``HierarchicalGrid.collect_cell_bounds`` documents *exact* agreement with
    per-cell ``cell_bounds`` (`_hierarchical_grid.py:1420-1421`), so callers pass
    ``exact=True`` and get a bitwise (``np.array_equal``) check.
    ``TensorProductGrid.collect_cell_bounds`` makes no such claim -- its result is a
    vectorized re-derivation from the same breakpoint arrays -- so callers pass a
    derived ``tol`` instead.

    Args:
        grid (TensorProductGrid | HierarchicalGrid): The grid whose per-cell
            ``cell_bounds`` is the oracle.
        exact (bool): Require bitwise equality rather than a tolerance.
        tol (float): Absolute tolerance when ``exact`` is ``False``.

    Returns:
        Callable[[object], str | None]: Predicate over the ``(cell_lo, cell_hi)`` result
        of ``collect_cell_bounds()``, reporting the worst mismatching cell.
    """

    def predicate(result: object) -> str | None:
        cell_lo, cell_hi = result  # type: ignore[misc]
        worst = 0.0
        worst_cid = -1
        for cid in range(grid.num_cells):
            lo, hi = grid.cell_bounds(cid)
            diff_lo = float(np.max(np.abs(lo - cell_lo[cid])))
            diff_hi = float(np.max(np.abs(hi - cell_hi[cid])))
            diff = max(diff_lo, diff_hi)
            if diff > worst:
                worst, worst_cid = diff, cid
        limit = 0.0 if exact else tol
        if worst > limit:
            kind = "bitwise" if exact else f"tol {tol:.3e}"
            return (
                f"collect_cell_bounds vs cell_bounds mismatch at cid={worst_cid}: "
                f"{worst:.3e} ({kind})"
            )
        return None

    return predicate


def _tiling_predicate(
    dom_lo: npt.NDArray[np.float64],
    dom_hi: npt.NDArray[np.float64],
    num_cells: int,
) -> Callable[[object], str | None]:
    """Build a predicate asserting cell bounds tile the domain exactly.

    Volume: summing ``num_cells`` per-cell volumes (each an ``ndim``-fold product)
    carries at most ``O(ndim * eps)`` relative rounding per product plus
    ``O(num_cells * eps)`` from the summation; an explicit factor of 8 covers both
    constants. Coverage: the cell union's own extreme corners must not escape the
    domain by more than the same bound.

    Args:
        dom_lo (npt.NDArray[np.float64]): Domain lower corner, shape ``(ndim,)``.
        dom_hi (npt.NDArray[np.float64]): Domain upper corner, shape ``(ndim,)``.
        num_cells (int): Total cell count (sets the summation error).

    Returns:
        Callable[[object], str | None]: Predicate over the ``(cell_lo, cell_hi)``
        result of ``collect_cell_bounds()``.
    """
    ndim = dom_lo.shape[0]
    domain_vol = float(np.prod(dom_hi - dom_lo))
    scale = max(abs(domain_vol), 1.0)
    tol = 8.0 * num_cells * ndim * _EPS64 * scale

    def predicate(result: object) -> str | None:
        cell_lo, cell_hi = result  # type: ignore[misc]
        volumes = _volume(cell_lo, cell_hi)
        total = float(np.sum(volumes))
        if abs(total - domain_vol) > tol:
            return (
                f"cell volumes sum to {total:.6e}, domain volume is {domain_vol:.6e} "
                f"(tol {tol:.3e})"
            )
        escape_lo = float(np.max(dom_lo - cell_lo.min(axis=0)))
        escape_hi = float(np.max(cell_hi.max(axis=0) - dom_hi))
        worst = max(escape_lo, escape_hi)
        if worst > tol:
            return f"cell bounds escape the domain by {worst:.3e} > {tol:.3e}"
        return None

    return predicate


def _locate_roundtrip_case(
    group_tag: str,
    grid: TensorProductGrid | HierarchicalGrid,
    *,
    params: dict[str, Any],
    profile: Profile,
) -> Iterator[Case]:
    """Yield the ``locate`` / ``locate_many`` round-trip cases for one grid.

    Every cell's centroid (from ``collect_cell_bounds``, not from ``locate`` itself)
    must locate back to its own cell id, singly and in a batch; a point far outside
    the domain and non-finite coordinates must locate to nothing. The individual
    ``nan``-only and ``inf``-only single-point cases are ``FULL``-only; the smoke
    profile keeps only the roundtrip, the far-outside point, and the combined
    nan/inf batch (which already exercises both non-finite kinds together).

    Args:
        group_tag (str): Label prefix identifying the grid under test.
        grid (TensorProductGrid | HierarchicalGrid): The grid to probe.
        params (dict[str, Any]): Axis values recorded in the case.
        profile (Profile): Sweep width.

    Yields:
        Case: The round-trip and outside/non-finite cases.
    """

    def run_roundtrip() -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        cell_lo, cell_hi = grid.collect_cell_bounds()
        centroids = 0.5 * (cell_lo + cell_hi)
        single = np.array(
            [-1 if (c := grid.locate(centroids[i])) is None else c for i in range(grid.num_cells)],
            dtype=np.int64,
        )
        batch = grid.locate_many(centroids)
        return single, batch

    def check_roundtrip(result: object) -> str | None:
        single, batch = result  # type: ignore[misc]
        expected = np.arange(grid.num_cells, dtype=np.int64)
        if not np.array_equal(single, expected):
            bad = np.flatnonzero(single != expected)
            return f"locate(centroid) mismatch at cids {bad[:5].tolist()}"
        if not np.array_equal(batch, expected):
            bad = np.flatnonzero(batch != expected)
            return f"locate_many mismatch at cids {bad[:5].tolist()}"
        return None

    # Being outside the grid is a documented *return value*, not a refusal:
    # `locate` documents "Non-finite coordinates (NaN or infinity) are outside
    # every cell" and returns None, and `locate_many` documents "-1 for points
    # outside the grid (including points with NaN or infinite coordinates)". So
    # every case in this helper must return; the returned value is what the
    # invariants grade.
    yield Case(
        GROUP,
        f"{group_tag}_locate_roundtrip",
        grid.locate,
        run_roundtrip,
        params,
        invariants=(custom("locate-roundtrip", check_roundtrip),),
        must_succeed=True,
    )

    # The domain's own lower corner, not an arbitrary cell's bounds: after refinement
    # flat cell id 0 need not sit at the domain corner (ids are reassigned level by
    # level, and the coarsest surviving block can be anywhere), so `cell_bounds(0)`
    # is not a reliable "near the corner" reference once the grid has been refined.
    if grid.num_cells:
        cell_lo_all, _ = grid.collect_cell_bounds()
        dom_lo = cell_lo_all.min(axis=0)
    else:
        dom_lo = np.zeros(grid.ndim)
    ndim = grid.ndim
    outside = dom_lo - 10.0 - np.arange(ndim, dtype=np.float64)

    def check_none(result: object) -> str | None:
        return None if result is None else f"expected None outside the domain; got {result!r}"

    yield Case(
        GROUP,
        f"{group_tag}_locate_outside",
        grid.locate,
        lambda outside=outside: grid.locate(outside),
        {**params, "kind": "far-outside"},
        invariants=(custom("outside-is-none", check_none),),
        must_succeed=True,
    )

    if profile is Profile.FULL:
        nan_pt = np.full(ndim, np.nan)
        inf_pt = np.full(ndim, np.inf)
        for name, pt in (("nan", nan_pt), ("inf", inf_pt)):
            yield Case(
                GROUP,
                f"{group_tag}_locate_{name}",
                grid.locate,
                lambda pt=pt: grid.locate(pt),
                {**params, "kind": name},
                invariants=(custom(f"{name}-is-none", check_none),),
                finite_inputs=False,
                must_succeed=True,
            )

    def check_all_outside(result: object) -> str | None:
        arr = np.asarray(result)
        if np.any(arr != -1):
            return f"locate_many should report -1 for every non-finite row; got {arr.tolist()}"
        return None

    yield Case(
        GROUP,
        f"{group_tag}_locate_many_nonfinite",
        grid.locate_many,
        lambda ndim=ndim: grid.locate_many(
            np.array([[np.nan] * ndim, [np.inf] * ndim, [-np.inf] * ndim])
        ),
        {**params, "kind": "nan-inf-batch"},
        invariants=(custom("nonfinite-is-minus-one", check_all_outside),),
        finite_inputs=False,
        must_succeed=True,
    )


# ---------------------------------------------------------------------------
# TensorProductGrid: construction, uniformity, tiling, locate, restrict
# ---------------------------------------------------------------------------


def _tp_construction_cases(profile: Profile) -> Iterator[Case]:
    """Yield :class:`TensorProductGrid` construction cases across dimension and domain.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One construction, one tiling check, and one collect-vs-percell check
        per ``(ndim, domain)``.
    """
    ndims = dims(profile, max_dim=4) if profile is Profile.FULL else (2,)
    for ndim in ndims:
        for domain in domains(profile):
            tag = _grid_tag(ndim, domain)
            n_per_axis = 2 if ndim >= 4 else 3  # noqa: PLR2004 -- keep n**4 cheap
            grid = _tp_grid(ndim, domain, n_per_axis)
            params = {"ndim": ndim, "domain": domain, "n_per_axis": n_per_axis}

            # Uniform breakpoints from np.linspace are strictly increasing, finite,
            # and have >= 2 entries per axis -- every precondition __init__ states --
            # and collect_cell_bounds documents no Raises: at all, so all three below
            # are legal by construction.
            yield Case(
                GROUP,
                f"tp_construct_{tag}",
                TensorProductGrid,
                lambda ndim=ndim, domain=domain, n=n_per_axis: _tp_grid(ndim, domain, n),
                params,
                invariants=(
                    custom(
                        "cell-count",
                        lambda r, n=n_per_axis, ndim=ndim: None
                        if r.num_cells == n**ndim
                        else f"num_cells={r.num_cells}, expected {n**ndim}",
                    ),
                ),
                must_succeed=True,
            )

            dom_lo, dom_hi = grid.bounds[:, 0].copy(), grid.bounds[:, 1].copy()
            yield Case(
                GROUP,
                f"tp_tiling_{tag}",
                TensorProductGrid.collect_cell_bounds,
                grid.collect_cell_bounds,
                params,
                invariants=(
                    custom("cell-tiling", _tiling_predicate(dom_lo, dom_hi, grid.num_cells)),
                ),
                must_succeed=True,
            )
            yield Case(
                GROUP,
                f"tp_collect_vs_percell_{tag}",
                TensorProductGrid.collect_cell_bounds,
                grid.collect_cell_bounds,
                params,
                # Not an exactness claim in the docstring (see the predicate's own
                # docstring): a broadcast re-derivation from the same breakpoints
                # carries at most one rounding per axis beyond the direct read.
                invariants=(
                    custom(
                        "collect-vs-percell",
                        _collect_vs_percell_predicate(
                            grid, exact=False, tol=4.0 * _EPS64 * max(abs(dom_hi).max(), 1.0)
                        ),
                    ),
                ),
                must_succeed=True,
            )

    if profile is not Profile.FULL:
        return

    # Malformed constructions: each is a documented rejection ("ValueError: If
    # breakpoints is empty, any axis has fewer than two entries, or any axis is
    # non-finite or not strictly increasing.", `_tensor_product_grid.py:86-89`).
    # tp_flat_axis ([0, 0, 1]) is caught by the same strictly-increasing clause,
    # not a separate one.
    yield Case(
        GROUP,
        "tp_empty_breakpoints",
        TensorProductGrid,
        lambda: TensorProductGrid([]),
        {},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "tp_too_short_axis",
        TensorProductGrid,
        lambda: TensorProductGrid([np.array([0.0])]),
        {},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "tp_nonfinite_axis",
        TensorProductGrid,
        lambda: TensorProductGrid([np.array([0.0, np.nan, 1.0])]),
        {},
        finite_inputs=False,
        must_reject=True,
    )
    yield Case(
        GROUP,
        "tp_non_increasing_axis",
        TensorProductGrid,
        lambda: TensorProductGrid([np.array([0.0, 1.0, 0.5])]),
        {},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "tp_flat_axis",
        TensorProductGrid,
        lambda: TensorProductGrid([np.array([0.0, 0.0, 1.0])]),
        {},
        must_reject=True,
    )


def _tp_uniformity_cases(profile: Profile) -> Iterator[Case]:
    """Yield ``is_uniform`` corner cases across every domain magnitude.

    An exactly-uniform grid must report ``True`` on every domain, including the
    translated ``[1e6, 1e6 + 1]`` one, and a perturbed grid must report ``False`` on
    every domain too. Both halves are the point: under the absolute tolerance this
    code used to compare against, the translated domain reported the EXACT grid as
    non-uniform, and the tiny domain reported the perturbed one as uniform. The
    verdict must not depend on where the domain sits.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One ``True``-expected and one ``False``-expected case per domain.
    """
    for domain in domains(profile):
        tag = _grid_tag(2, domain)
        breakpoints = _uniform_breakpoints(2, domain, 4)
        exact_grid = TensorProductGrid(breakpoints)
        # Both grids below are validly constructed and `is_uniform` is a property
        # with no `Raises:`; the True/False verdict is graded by the invariant,
        # not by the flag.
        yield Case(
            GROUP,
            f"tp_is_uniform_true_{tag}",
            TensorProductGrid.is_uniform.fget,  # type: ignore[attr-defined]
            lambda g=exact_grid: g.is_uniform,
            {"domain": domain, "kind": "exact"},
            invariants=(
                custom(
                    "expected-uniform",
                    lambda r: None if r else "exactly-spaced grid reported non-uniform",
                ),
            ),
            must_succeed=True,
        )

        span = domain[1] - domain[0]
        perturbed = [bp.copy() for bp in breakpoints]
        spacing = span / 4.0
        # Two derived candidates, whichever is larger.
        #
        # `spacing * 1e-7` is the relative perturbation this axis is meant to probe: a
        # tenth of a part per million of a cell, far larger than round-off and far
        # smaller than the cell, so the breakpoints stay strictly increasing.
        #
        # It is not always above the code's own threshold, which is
        # `16 eps (|lo| + |hi|)`. On the translated `[1e6, 1e6 + 1]` domain that
        # threshold is ~7e-9 while `spacing * 1e-7` is ~2.5e-8 -- above it, but by a
        # factor of only 3.5, which is not a margin. So the floor is a thousand times
        # the threshold itself rather than a constant: it scales with the domain the
        # way the threshold does, which the old absolute floor could not.
        tolerance = _UNIFORM_SPACING_EPS_FACTOR * _EPS64 * (abs(domain[0]) + abs(domain[1]))
        perturbation = max(spacing * 1e-7, 1e3 * tolerance)
        perturbed[0][2] += perturbation
        perturbed_grid = TensorProductGrid(perturbed)
        yield Case(
            GROUP,
            f"tp_is_uniform_false_{tag}",
            TensorProductGrid.is_uniform.fget,  # type: ignore[attr-defined]
            lambda g=perturbed_grid: g.is_uniform,
            {"domain": domain, "kind": "perturbed"},
            invariants=(
                custom(
                    "expected-nonuniform",
                    lambda r: None if not r else "spacing-perturbed grid reported uniform",
                ),
            ),
            arrays={"breakpoints_axis0": perturbed[0]},
            must_succeed=True,
        )


def _tp_locate_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`TensorProductGrid.locate` round-trip cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: See :func:`_locate_roundtrip_case`.
    """
    ndims = dims(profile, max_dim=4) if profile is Profile.FULL else (2,)
    for ndim in ndims:
        for domain in domains(profile):
            n_per_axis = 2 if ndim >= 4 else 3  # noqa: PLR2004 -- keep n**4 cheap
            grid = _tp_grid(ndim, domain, n_per_axis)
            yield from _locate_roundtrip_case(
                f"tp_{_grid_tag(ndim, domain)}",
                grid,
                params={"ndim": ndim, "domain": domain, "n_per_axis": n_per_axis},
                profile=profile,
            )


def _tp_neighbor_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`TensorProductGrid.neighbor_across_facet` symmetry cases.

    For every interior facet, crossing to the neighbour and back across the
    opposite local facet must return the original cell -- a self-consistency
    property that a bug in the per-axis arithmetic (an off-by-one at a domain
    edge, most likely) would break asymmetrically.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One symmetry check per ``(ndim, domain)``, plus a boundary check.
    """
    ndims = dims(profile, max_dim=3) if profile is Profile.FULL else (2,)
    for ndim in ndims:
        for domain in domains(profile):
            grid = _tp_grid(ndim, domain, 3)
            tag = _grid_tag(ndim, domain)

            def check_symmetry(_: object, grid: TensorProductGrid = grid) -> str | None:
                for cid in range(grid.num_cells):
                    for lfid in range(grid.num_local_facets(cid)):
                        nbr = grid.neighbor_across_facet(cid, lfid)
                        if nbr is None:
                            continue
                        axis, side = grid.local_facet_axis_side(cid, lfid)
                        opposite = 2 * axis + (1 - side)
                        back = grid.neighbor_across_facet(nbr, opposite)
                        if back != cid:
                            return (
                                f"cell {cid} facet {lfid} -> {nbr}, but {nbr}'s opposite facet "
                                f"{opposite} -> {back} (expected {cid})"
                            )
                return None

            # The thunk is a placeholder (`lambda: None`); the real calls happen
            # inside the invariant, so `must_succeed` guards this case against a
            # future refactor rather than grading anything today.
            yield Case(
                GROUP,
                f"tp_neighbor_symmetry_{tag}",
                TensorProductGrid.neighbor_across_facet,
                lambda: None,
                {"ndim": ndim, "domain": domain},
                invariants=(custom("neighbor-symmetry", check_symmetry),),
                must_succeed=True,
            )

    if profile is not Profile.FULL:
        return
    grid = _tp_grid(2, (0.0, 1.0), 2)
    # "IndexError: If cid or lfid is out of range."
    yield Case(
        GROUP,
        "tp_neighbor_lfid_out_of_range",
        TensorProductGrid.neighbor_across_facet,
        lambda: grid.neighbor_across_facet(0, 4),
        {"kind": "lfid-oob"},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "tp_neighbor_cid_out_of_range",
        TensorProductGrid.neighbor_across_facet,
        lambda: grid.neighbor_across_facet(grid.num_cells, 0),
        {"kind": "cid-oob"},
        must_reject=True,
    )


def _tp_restrict_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`TensorProductGrid.restrict` cases.

    The docstring's exactness claim ("never re-based or re-clamped") licenses a
    bitwise comparison between the sub-grid's cell bounds and the original grid's,
    reached through ``local_to_global_cell``.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: Corner selections plus the malformed-input rejections.
    """
    grid = _tp_grid(3, (0.0, 5.0), 4)

    def check_exact_slice(result: object) -> str | None:
        restriction = result
        for local_cid in range(restriction.grid.num_cells):  # type: ignore[attr-defined]
            global_cid = int(restriction.local_to_global_cell[local_cid])  # type: ignore[attr-defined]
            lo_sub, hi_sub = restriction.grid.cell_bounds(local_cid)  # type: ignore[attr-defined]
            lo_glob, hi_glob = grid.cell_bounds(global_cid)
            if not (np.array_equal(lo_sub, lo_glob) and np.array_equal(hi_sub, hi_glob)):
                return (
                    f"restrict cell {local_cid} (-> global {global_cid}) bounds "
                    "are not a pure slice"
                )
        return None

    selectors = {
        "single_corner": np.array([0]),
        "all_cells": np.arange(grid.num_cells),
    }
    if profile is Profile.FULL:
        selectors["single_center"] = np.array([grid.num_cells // 2])
        selectors["two_opposite_corners"] = np.array([0, grid.num_cells - 1])
    # All ids below are in [0, num_cells).
    for name, ids in selectors.items():
        yield Case(
            GROUP,
            f"tp_restrict_{name}",
            TensorProductGrid.restrict,
            lambda ids=ids: grid.restrict(ids),
            {"kind": name, "n_ids": int(ids.size)},
            invariants=(custom("restrict-is-pure-slice", check_exact_slice),),
            must_succeed=True,
        )

    if profile is not Profile.FULL:
        return
    yield Case(
        GROUP,
        "tp_restrict_empty",
        TensorProductGrid.restrict,
        lambda: grid.restrict([]),
        {},
        must_reject=True,  # "ValueError: If cell_ids is empty."
    )
    yield Case(
        GROUP,
        "tp_restrict_out_of_range",
        TensorProductGrid.restrict,
        lambda: grid.restrict([grid.num_cells]),
        {},
        # "IndexError: If any cell id is out of range [0, num_cells)."
        must_reject=True,
    )
    yield Case(
        GROUP,
        "tp_restrict_non_integer",
        TensorProductGrid.restrict,
        lambda: grid.restrict(np.array([0.5])),
        {},
        must_reject=True,  # "TypeError: If cell_ids is not integer-valued."
    )


def _uniform_grid_factory_cases(profile: Profile) -> Iterator[Case]:
    """Yield :func:`pantr.grid.uniform_grid` factory corner cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One-cell-per-axis, anisotropic counts, and malformed-input rejections.
    """
    for ndim in dims(profile, max_dim=4):
        bounds = np.array([[0.0, 1.0 + d] for d in range(ndim)])
        yield Case(
            GROUP,
            f"uniform_grid_single_cell_d{ndim}",
            uniform_grid,
            lambda bounds=bounds, ndim=ndim: uniform_grid(bounds, 1),
            {"ndim": ndim, "cells": 1},
            invariants=(
                custom(
                    "single-cell",
                    lambda r: None if r.num_cells == 1 else f"num_cells={r.num_cells}, expected 1",
                ),
            ),
            must_succeed=True,
        )
        if profile is Profile.FULL and ndim > 1:
            anisotropic = tuple(range(2, 2 + ndim))
            yield Case(
                GROUP,
                f"uniform_grid_anisotropic_d{ndim}",
                uniform_grid,
                lambda bounds=bounds, anisotropic=anisotropic: uniform_grid(bounds, anisotropic),
                {"ndim": ndim, "cells": anisotropic},
                invariants=(
                    custom(
                        "cells-per-axis-matches",
                        lambda r, anisotropic=anisotropic: None
                        if r.cells_per_axis == anisotropic
                        else f"cells_per_axis={r.cells_per_axis}, expected {anisotropic}",
                    ),
                ),
                must_succeed=True,
            )

    if profile is not Profile.FULL:
        return
    # "ValueError: If bounds does not have shape (ndim, 2), any axis has lo >= hi,
    # cells has the wrong length, or any count is < 1." The degenerate and
    # inverted cases below are both caught by the single `lo >= hi` clause.
    yield Case(
        GROUP,
        "uniform_grid_zero_cells",
        uniform_grid,
        lambda: uniform_grid([[0.0, 1.0]], 0),
        {},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "uniform_grid_degenerate_bounds",
        uniform_grid,
        lambda: uniform_grid([[1.0, 1.0]], 2),
        {},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "uniform_grid_inverted_bounds",
        uniform_grid,
        lambda: uniform_grid([[1.0, 0.0]], 2),
        {},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "uniform_grid_cells_wrong_length",
        uniform_grid,
        lambda: uniform_grid([[0.0, 1.0], [0.0, 1.0]], [2, 2, 2]),
        {},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "uniform_grid_bad_shape",
        uniform_grid,
        lambda: uniform_grid(np.zeros((2, 3)), 2),
        {},
        must_reject=True,
    )


def _tensor_product_from_space_cases(profile: Profile) -> Iterator[Case]:
    """Yield :func:`pantr.grid.tensor_product_grid` cases built from a B-spline space.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: A clamped-space success and the documented periodic rejection.
    """
    if profile is not Profile.FULL:
        return
    clamped = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]), 2)
    space = BsplineSpace([clamped, clamped])
    yield Case(
        GROUP,
        "tensor_product_grid_from_clamped_space",
        tensor_product_grid,
        lambda space=space: tensor_product_grid(space),
        {"kind": "clamped"},
        must_succeed=True,
    )
    periodic_knots = np.arange(-2, 6, dtype=np.float64)
    periodic = BsplineSpace1D(periodic_knots, 2, periodic=True)
    periodic_space = BsplineSpace([clamped, periodic])
    yield Case(
        GROUP,
        "tensor_product_grid_from_periodic_space",
        tensor_product_grid,
        lambda periodic_space=periodic_space: tensor_product_grid(periodic_space),
        {"kind": "periodic-axis"},
        must_reject=True,  # "ValueError: If any direction of space is periodic."
    )


# ---------------------------------------------------------------------------
# HierarchicalGrid: construction, refinement, bounds, locate, masks
# ---------------------------------------------------------------------------


def _deep_refine_chain(grid: HierarchicalGrid, target_level: int) -> HierarchicalGrid:
    """Refine a hierarchical grid's origin cell down to ``target_level``.

    Repeatedly refines only the single active cell at multi-index ``(0, ..., 0)``,
    so the total active-cell count grows linearly in ``target_level`` rather than
    exponentially.

    Args:
        grid (HierarchicalGrid): Grid whose origin cell is refined; left unchanged.
        target_level (int): Deepest level to reach.

    Returns:
        HierarchicalGrid: A new grid refined down to ``target_level``.
    """
    ndim = grid.ndim
    origin_lo = tuple(0 for _ in range(ndim))
    origin_hi = tuple(1 for _ in range(ndim))
    for level in range(target_level):
        grid = grid.refine(level, origin_lo, origin_hi)
    return grid


def _hier_construction_cases(profile: Profile) -> Iterator[Case]:
    """Yield :class:`HierarchicalGrid` construction cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One construction per ``(ndim, factor)`` plus malformed-input rejections.
    """
    factors: tuple[int | tuple[int, ...], ...] = (2, 3) if profile is Profile.FULL else (2,)
    for ndim in dims(profile, max_dim=4):
        root = _tp_grid(ndim, (0.0, 1.0), 2)
        for factor in factors:
            yield Case(
                GROUP,
                f"hier_construct_d{ndim}_f{factor}",
                HierarchicalGrid,
                lambda root=root, factor=factor: HierarchicalGrid(root, factor),
                {"ndim": ndim, "factor": factor},
                invariants=(
                    custom(
                        "starts-at-root",
                        lambda r, root=root: None
                        if r.num_cells == root.num_cells
                        else f"num_cells={r.num_cells}, expected {root.num_cells}",
                    ),
                ),
                must_succeed=True,
            )
        if profile is Profile.FULL and ndim >= 2:  # noqa: PLR2004 -- anisotropic factor needs 2 axes
            # Alternate 2/1 per axis so every ndim gets a genuinely mixed factor
            # (a factor of 1 on an axis prevents subdivision in that direction,
            # which is explicitly legal).
            anisotropic = tuple(2 if k % 2 == 0 else 1 for k in range(ndim))
            yield Case(
                GROUP,
                f"hier_construct_anisotropic_d{ndim}",
                HierarchicalGrid,
                lambda root=root, anisotropic=anisotropic: HierarchicalGrid(root, anisotropic),
                {"ndim": ndim, "factor": anisotropic},
                must_succeed=True,
            )

    if profile is not Profile.FULL:
        return
    root2 = _tp_grid(2, (0.0, 1.0), 2)
    yield Case(
        GROUP,
        "hier_bad_root_type",
        HierarchicalGrid,
        lambda: HierarchicalGrid(object(), 2),
        {},
        must_reject=True,  # "TypeError: If root is not a TensorProductGrid."
    )
    # "ValueError: If factor has the wrong length or any entry is < 1."
    yield Case(
        GROUP,
        "hier_factor_wrong_length",
        HierarchicalGrid,
        lambda: HierarchicalGrid(root2, (2, 2, 2)),
        {},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "hier_factor_below_one",
        HierarchicalGrid,
        lambda: HierarchicalGrid(root2, 0),
        {},
        must_reject=True,
    )


def _hier_refine_coarsen_cases(profile: Profile) -> Iterator[Case]:
    """Yield refine/refine_cells/coarsen depth and no-op/inverse cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: Depth ladder (1, 2, 3, capped-deep), a no-op refine, an exact
        coarsen-undoes-refine round trip, and malformed-input rejections.
    """
    levels = (1, 2, 3, _MAX_PROBED_LEVEL) if profile is Profile.FULL else (1, 2)
    for target_level in levels:
        root = _tp_grid(2, (0.0, 1.0), 2)
        grid = HierarchicalGrid(root, 2)

        def build(
            grid: HierarchicalGrid = grid, target_level: int = target_level
        ) -> HierarchicalGrid:
            return _deep_refine_chain(grid, target_level)

        yield Case(
            GROUP,
            f"hier_refine_depth_{target_level}",
            HierarchicalGrid.refine,
            build,
            {"target_level": target_level},
            invariants=(
                custom(
                    "reaches-target-level",
                    lambda r, target_level=target_level: None
                    if r.max_level == target_level
                    else f"max_level={r.max_level}, expected {target_level}",
                ),
            ),
            # Each step refines an existing level with lo < hi inside the level's
            # domain.
            must_succeed=True,
        )

    if profile is not Profile.FULL:
        return

    # Union semantics: refining a region with no active cells there is a no-op.
    root = _tp_grid(2, (0.0, 1.0), 4)
    grid = HierarchicalGrid(root, 2)
    grid = grid.refine(0, (0, 0), (1, 1))

    def refine_noop() -> tuple[int, int]:
        before = grid.num_cells
        # Already fully refined away at level 0: the returned grid is a distinct
        # object but its cell count must match, which is what this case checks.
        after = grid.refine(0, (0, 0), (1, 1))
        return before, after.num_cells

    yield Case(
        GROUP,
        "hier_refine_noop_on_refined_region",
        HierarchicalGrid.refine,
        refine_noop,
        {"kind": "noop"},
        invariants=(
            custom(
                "unchanged",
                lambda r: None
                if r[0] == r[1]
                else f"num_cells changed {r[0]} -> {r[1]} on a no-op refine",
            ),
        ),
        # "If the intersection with active blocks at level is empty, the call is
        # a silent no-op."
        must_succeed=True,
    )

    # Coarsen exactly undoes a matching refine.
    root2 = _tp_grid(2, (0.0, 1.0), 3)
    grid2 = HierarchicalGrid(root2, 3)

    def refine_then_coarsen() -> tuple[int, int]:
        before = grid2.num_cells
        after = grid2.refine(0, (1, 1), (2, 2)).coarsen(0, (1, 1), (2, 2))
        return before, after.num_cells

    yield Case(
        GROUP,
        "hier_coarsen_undoes_refine",
        HierarchicalGrid.coarsen,
        refine_then_coarsen,
        {"kind": "round-trip"},
        invariants=(
            custom(
                "restored",
                lambda r: None
                if r[0] == r[1]
                else f"num_cells {r[0]} -> refine -> coarsen -> {r[1]}",
            ),
        ),
        must_succeed=True,
    )

    root3 = _tp_grid(2, (0.0, 1.0), 2)
    grid3 = HierarchicalGrid(root3, 2)
    # "ValueError: If level is out of range, lo/hi have the wrong length, any
    # lo[k] >= hi[k], or [lo, hi) falls entirely outside the level-level domain."
    yield Case(
        GROUP,
        "hier_refine_level_out_of_range",
        HierarchicalGrid.refine,
        lambda: grid3.refine(5, (0, 0), (1, 1)),
        {"kind": "level-oob"},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "hier_refine_lo_ge_hi",
        HierarchicalGrid.refine,
        lambda: grid3.refine(0, (1, 1), (1, 1)),
        {"kind": "lo-ge-hi"},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "hier_coarsen_partial_region",
        HierarchicalGrid.coarsen,
        lambda: grid3.refine(0, (0, 0), (1, 1)).coarsen(0, (0, 0), (2, 2)),
        {"kind": "partially-refined-region"},
        # "...or the region is not fully refined to exactly level level+1."
        must_reject=True,
    )
    yield Case(
        GROUP,
        "hier_refine_cells_out_of_range",
        HierarchicalGrid.refine_cells,
        lambda: grid3.refine_cells([grid3.num_cells]),
        {"kind": "cid-oob"},
        must_reject=True,  # "IndexError: If any id in cell_ids is out of range."
    )


def _hier_bounds_cases(profile: Profile) -> Iterator[Case]:
    """Yield ``collect_cell_bounds`` / ``export_cells`` / tiling cases for a refined hierarchy.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: The exact ``collect_cell_bounds`` check, the ``export_cells`` bound
        check, and the tiling check, for each dimension and refinement depth.
    """
    for ndim in dims(profile, max_dim=4):
        for target_level in (2,) if profile is not Profile.FULL else (1, 2, 3):
            root = _tp_grid(ndim, (0.0, 1.0), 2)
            grid = HierarchicalGrid(root, 2)
            grid = _deep_refine_chain(grid, target_level)
            tag = f"d{ndim}_l{target_level}"
            params = {"ndim": ndim, "target_level": target_level}

            # `collect_cell_bounds` docstring claims exact agreement with `cell_bounds`
            # (`_hierarchical_grid.py:1420-1421`): bitwise check. All three cases below
            # run on a validly refined grid and none of the three methods documents a
            # `Raises:` section.
            yield Case(
                GROUP,
                f"hier_collect_vs_percell_{tag}",
                HierarchicalGrid.collect_cell_bounds,
                grid.collect_cell_bounds,
                params,
                invariants=(
                    custom(
                        "collect-vs-percell-exact",
                        _collect_vs_percell_predicate(grid, exact=True),
                    ),
                ),
                must_succeed=True,
            )

            dom_lo, dom_hi = root.bounds[:, 0].copy(), root.bounds[:, 1].copy()
            yield Case(
                GROUP,
                f"hier_tiling_{tag}",
                HierarchicalGrid.collect_cell_bounds,
                grid.collect_cell_bounds,
                params,
                invariants=(
                    custom("cell-tiling", _tiling_predicate(dom_lo, dom_hi, grid.num_cells)),
                ),
                must_succeed=True,
            )

            # `export_cells` docstring (`_hierarchical_grid.py:1502-1516`) claims only
            # `8 * eps * |coordinate|` agreement, and says bitwise agreement is
            # unattainable in general -- do not tighten this to exact equality.
            yield Case(
                GROUP,
                f"hier_export_cells_bound_{tag}",
                HierarchicalGrid.export_cells,
                grid.export_cells,
                params,
                invariants=(custom("export-cells-bound", _export_cells_predicate(grid)),),
                must_succeed=True,
            )


def _export_cells_predicate(grid: HierarchicalGrid) -> Callable[[object], str | None]:
    """Build a predicate checking ``export_cells`` against the docstring's own bound.

    Recomputes each cell's expected corners directly from ``cell_bounds`` (never
    from ``export_cells``'s own lattice-deduplication code path), so this is an
    independent oracle, not a mirror.

    Args:
        grid (HierarchicalGrid): The grid whose ``cell_bounds`` is the oracle.

    Returns:
        Callable[[object], str | None]: Predicate over the ``(points, conn)`` result.
    """
    ndim = grid.ndim

    def predicate(result: object) -> str | None:
        points, conn = result  # type: ignore[misc]
        worst_violation = 0.0
        worst_cid = -1
        for cid in range(grid.num_cells):
            lo, hi = grid.cell_bounds(cid)
            corners = points[conn[cid]]
            for corner_id in range(corners.shape[0]):
                expected = np.where(
                    [(corner_id >> k) & 1 for k in range(ndim)],
                    hi,
                    lo,
                )
                tol = 8.0 * _EPS64 * np.maximum(np.abs(expected), 1.0)
                violation = float(np.max(np.abs(corners[corner_id] - expected) - tol))
                if violation > worst_violation:
                    worst_violation, worst_cid = violation, cid
        if worst_violation > 0.0:
            return (
                f"export_cells corner exceeds the docstring's 8*eps*|coordinate| bound "
                f"by {worst_violation:.3e} at cid={worst_cid}"
            )
        return None

    return predicate


def _hier_locate_cases(profile: Profile) -> Iterator[Case]:
    """Yield locate round-trip cases for a refined hierarchical grid.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: See :func:`_locate_roundtrip_case`.
    """
    for ndim in dims(profile, max_dim=3):
        for domain in domains(profile):
            root = _tp_grid(ndim, domain, 2)
            grid = HierarchicalGrid(root, 2)
            grid = _deep_refine_chain(grid, 2)
            yield from _locate_roundtrip_case(
                f"hier_{_grid_tag(ndim, domain)}",
                grid,
                profile=profile,
                params={"ndim": ndim, "domain": domain, "target_level": 2},
            )


def _hier_mask_cases(profile: Profile) -> Iterator[Case]:
    """Yield ``active_leaf_mask`` / ``subdomain_mask`` / ``level_cells_per_axis`` cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: Mask-shape and content checks, plus malformed-level rejections.
    """
    root = _tp_grid(2, (0.0, 1.0), 2)
    grid = HierarchicalGrid(root, 2)
    grid = _deep_refine_chain(grid, 3)

    levels = range(grid.max_level + 1) if profile is Profile.FULL else (0, grid.max_level)
    for level in levels:
        yield Case(
            GROUP,
            f"hier_active_leaf_mask_l{level}",
            HierarchicalGrid.active_leaf_mask,
            lambda level=level: grid.active_leaf_mask(level),
            {"level": level},
            invariants=(expected_shape(grid.level_cells_per_axis(level)),),
            must_succeed=True,
        )
        yield Case(
            GROUP,
            f"hier_subdomain_mask_l{level}",
            HierarchicalGrid.subdomain_mask,
            lambda level=level: grid.subdomain_mask(level),
            {"level": level},
            invariants=(expected_shape(grid.level_cells_per_axis(level)),),
            must_succeed=True,
        )

    if profile is not Profile.FULL:
        return
    # This looks like an out-of-range case and is not: `level_cells_per_axis`
    # documents "Must be >= 0; values above max_level are accepted and return
    # the geometrically valid count", and its only `Raises:` is "ValueError: If
    # level < 0". So `must_succeed`, in deliberate contrast with
    # `active_leaf_mask`/`subdomain_mask` right below, which DO require
    # level <= max_level.
    yield Case(
        GROUP,
        "hier_level_cells_per_axis_above_max",
        HierarchicalGrid.level_cells_per_axis,
        lambda: grid.level_cells_per_axis(grid.max_level + 5),
        {"kind": "above-max-level"},
        must_succeed=True,
    )
    yield Case(
        GROUP,
        "hier_level_cells_per_axis_negative",
        HierarchicalGrid.level_cells_per_axis,
        lambda: grid.level_cells_per_axis(-1),
        {"kind": "negative-level"},
        must_reject=True,  # "ValueError: If level < 0."
    )
    yield Case(
        GROUP,
        "hier_active_leaf_mask_above_max",
        HierarchicalGrid.active_leaf_mask,
        lambda: grid.active_leaf_mask(grid.max_level + 1),
        {"kind": "above-max-level"},
        must_reject=True,  # "ValueError: If level is outside [0, max_level]."
    )


def _hier_hanging_neighbor_cases(profile: Profile) -> Iterator[Case]:
    """Yield ``hanging_neighbors`` geometric-adjacency cases at a level interface.

    The oracle recomputes touching from ``cell_bounds`` (independent of
    ``hanging_neighbors``'s own block-descent code path): a returned neighbour must
    share the queried facet's plane and overlap its extent on every other axis.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One adjacency check over every cell/facet of a once-refined grid.
    """
    root = _tp_grid(2, (0.0, 1.0), 2)
    grid = HierarchicalGrid(root, 2)
    grid = grid.refine(0, (0, 0), (1, 1))  # refine one root cell; leaves a hanging interface

    def check_touching(_: object) -> str | None:
        for cid in range(grid.num_cells):
            lo, hi = grid.cell_bounds(cid)
            for lfid in range(grid.num_local_facets(cid)):
                axis, side = grid.local_facet_axis_side(cid, lfid)
                plane = hi[axis] if side == 1 else lo[axis]
                for nbr in grid.hanging_neighbors(cid, lfid):
                    n_lo, n_hi = grid.cell_bounds(nbr)
                    n_plane = n_lo[axis] if side == 1 else n_hi[axis]
                    if abs(n_plane - plane) > 8.0 * _EPS64 * max(abs(plane), 1.0):
                        return (
                            f"cell {cid} facet {lfid}: neighbor {nbr} does not touch "
                            "the facet plane"
                        )
                    other_axes = [k for k in range(grid.ndim) if k != axis]
                    overlap_lo = np.maximum(lo[other_axes], n_lo[other_axes])
                    overlap_hi = np.minimum(hi[other_axes], n_hi[other_axes])
                    if np.any(overlap_lo > overlap_hi):
                        return (
                            f"cell {cid} facet {lfid}: neighbor {nbr} does not overlap "
                            "the facet extent"
                        )
        return None

    # The thunk is a placeholder (`lambda: None`); the real calls happen inside
    # the invariant, so `must_succeed` guards this case against a future
    # refactor rather than grading anything today.
    yield Case(
        GROUP,
        "hier_hanging_neighbors_touch",
        HierarchicalGrid.hanging_neighbors,
        lambda: None,
        {"kind": "geometric-adjacency"},
        invariants=(custom("hanging-neighbors-touch", check_touching),),
        must_succeed=True,
    )


def _hier_restrict_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`HierarchicalGrid.restrict` cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: Corner selections plus the malformed-input rejections.
    """
    root = _tp_grid(2, (0.0, 1.0), 3)
    grid = HierarchicalGrid(root, 2)
    grid = grid.refine(0, (1, 1), (2, 2))

    selectors = {
        "single_leaf": np.array([0]),
        "all_leaves": np.arange(grid.num_cells),
    }
    if profile is Profile.FULL:
        selectors["deep_leaf"] = np.array([grid.num_cells - 1])
    for name, ids in selectors.items():
        yield Case(
            GROUP,
            f"hier_restrict_{name}",
            HierarchicalGrid.restrict,
            lambda ids=ids: grid.restrict(ids),
            {"kind": name, "n_ids": int(ids.size)},
            invariants=(
                custom(
                    "restrict-returns-hierarchical-grid",
                    lambda r: None
                    if isinstance(r.grid, HierarchicalGrid)  # type: ignore[attr-defined]
                    else f"expected HierarchicalGrid, got {type(r.grid).__name__}",  # type: ignore[attr-defined]
                ),
            ),
            must_succeed=True,
        )

    if profile is not Profile.FULL:
        return
    # Same documented clauses as TensorProductGrid.restrict.
    yield Case(
        GROUP,
        "hier_restrict_empty",
        HierarchicalGrid.restrict,
        lambda: grid.restrict([]),
        {},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "hier_restrict_out_of_range",
        HierarchicalGrid.restrict,
        lambda: grid.restrict([grid.num_cells]),
        {},
        must_reject=True,
    )


def _hierarchical_grid_factory_case(profile: Profile) -> Iterator[Case]:
    """Yield a case exercising the standalone :func:`hierarchical_grid` factory.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: A single factory-equivalence check.
    """
    if profile is not Profile.FULL:
        return
    root = _tp_grid(2, (0.0, 1.0), 2)
    yield Case(
        GROUP,
        "hierarchical_grid_factory",
        hierarchical_grid,
        lambda root=root: hierarchical_grid(root, 2),
        {"kind": "factory"},
        invariants=(
            custom(
                "matches-constructor",
                lambda r, root=root: None
                if r.num_cells == root.num_cells
                else f"num_cells={r.num_cells}, expected {root.num_cells}",
            ),
        ),
        must_succeed=True,
    )


# ---------------------------------------------------------------------------
# BVH: construction, degenerate configurations, query completeness
# ---------------------------------------------------------------------------


def _bvh_query_predicate(
    cell_lo: npt.NDArray[np.float64],
    cell_hi: npt.NDArray[np.float64],
    query_box: AABB,
) -> Callable[[object], str | None]:
    """Build a predicate comparing ``query_aabb`` to a brute-force overlap scan.

    The brute-force oracle calls :meth:`pantr.geometry.AABB.overlaps` (an
    elementwise NumPy comparison), an entirely different code path from the BVH's
    iterative stack traversal.

    Args:
        cell_lo (npt.NDArray[np.float64]): Per-cell lower corners, shape ``(n, ndim)``.
        cell_hi (npt.NDArray[np.float64]): Per-cell upper corners, shape ``(n, ndim)``.
        query_box (AABB): The query box.

    Returns:
        Callable[[object], str | None]: Predicate over the ``query_aabb`` result.
    """

    def predicate(result: object) -> str | None:
        brute = {
            cid
            for cid in range(cell_lo.shape[0])
            if AABB(cell_lo[cid], cell_hi[cid]).overlaps(query_box)
        }
        got = {int(c) for c in result}  # type: ignore[attr-defined]
        if got != brute:
            missing = sorted(brute - got)[:5]
            extra = sorted(got - brute)[:5]
            return f"query_aabb mismatch vs brute force: missing={missing} extra={extra}"
        return None

    return predicate


def _bvh_from_grid_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`Grid.query_aabb` completeness cases over a small tensor-product grid.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: Whole-domain, single-cell, shared-face, degenerate, outside, and
        unbounded query boxes, each checked against the brute-force oracle.
    """
    grid = _tp_grid(2, (0.0, 1.0), 4)
    cell_lo, cell_hi = grid.collect_cell_bounds()
    domain_lo, domain_hi = grid.bounds[:, 0], grid.bounds[:, 1]

    # A box spanning a single interior shared face (zero thickness on one axis):
    # touching counts as overlapping, so both cells across the face must be hit.
    face_box = AABB(np.array([0.25, 0.0]), np.array([0.25, 1.0]))
    boxes = {
        "whole_domain": AABB(domain_lo, domain_hi),
        "single_cell": AABB(cell_lo[0], cell_hi[0]),
        # AABB rejects NaN bounds at construction (`geometry.py:147-151`), so a
        # literal NaN query box cannot be built; an unbounded (inf) box is the
        # closest hostile substitute and exercises the same inf-arithmetic path
        # in the BVH's overlap test.
        "unbounded": AABB.unbounded(2),
    }
    if profile is Profile.FULL:
        boxes["shared_face"] = face_box
        boxes["zero_volume_point"] = AABB(cell_lo[0], cell_lo[0])
        boxes["entirely_outside"] = AABB(domain_lo - 10.0, domain_lo - 5.0)
    # `query_aabb`'s only documented `Raises:` is "ValueError: If aabb.ndim !=
    # self.ndim", and every box here is built at the grid's own ndim. An
    # unbounded (inf) box is a legal AABB.
    for name, box in boxes.items():
        yield Case(
            GROUP,
            f"bvh_query_{name}",
            grid.query_aabb,
            lambda box=box: grid.query_aabb(box),
            {"kind": name},
            invariants=(custom("query-completeness", _bvh_query_predicate(cell_lo, cell_hi, box)),),
            must_succeed=True,
        )


def _bvh_direct_construction_cases(profile: Profile) -> Iterator[Case]:
    """Yield direct :class:`BVH` constructor cases (dtype/shape strictness).

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: Wrong dtype, mismatched shapes, and ``n_nodes`` mismatch rejections.
    """
    if profile is not Profile.FULL:
        return
    lo = np.zeros((1, 2), dtype=np.float64)
    hi = np.ones((1, 2), dtype=np.float64)
    idx = np.array([-1], dtype=np.int64)
    cell = np.array([0], dtype=np.int64)
    yield Case(
        GROUP,
        "bvh_direct_wrong_dtype",
        BVH,
        lambda: BVH(lo.astype(np.float32), hi, idx, idx, cell, n_cells=1),
        {"kind": "wrong-dtype"},
        must_reject=True,  # "TypeError: If any array has the wrong dtype."
    )
    # "ValueError: If shapes are inconsistent, ... n_nodes != 2 * n_cells - 1 ..."
    yield Case(
        GROUP,
        "bvh_direct_shape_mismatch",
        BVH,
        lambda: BVH(lo, np.ones((2, 2)), idx, idx, cell, n_cells=1),
        {"kind": "shape-mismatch"},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "bvh_direct_n_nodes_mismatch",
        BVH,
        lambda: BVH(lo, hi, idx, idx, cell, n_cells=2),
        {"kind": "n-nodes-mismatch"},
        must_reject=True,
    )


def _bvh_n_nodes_predicate(n_cells: int) -> Callable[[object], str | None]:
    """Build a predicate asserting ``BVH.n_nodes == 2 * n_cells - 1`` (0 when empty).

    Args:
        n_cells (int): Number of cells the tree was built over.

    Returns:
        Callable[[object], str | None]: Predicate over the resulting :class:`BVH`.
    """
    expected = 2 * n_cells - 1 if n_cells > 0 else 0

    def predicate(result: object) -> str | None:
        got = result.n_nodes  # type: ignore[attr-defined]
        if got != expected:
            return f"n_nodes={got}, expected {expected}"
        return None

    return predicate


def _bvh_degenerate_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`BVH.from_cell_bounds` cases over degenerate cell configurations.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: Zero cells, one cell, all-identical boxes, all-collapsed points, a
        line arrangement, and touching-face boxes, each checked against the
        brute-force overlap oracle.
    """
    configs: dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {
        "zero_cells": (np.zeros((0, 2)), np.zeros((0, 2))),
        "one_cell": (np.zeros((1, 2)), np.ones((1, 2))),
        # Touching-face pair: two boxes sharing exactly the x=1 face.
        "touching_faces": (
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.array([[1.0, 1.0], [2.0, 1.0]]),
        ),
    }
    if profile is Profile.FULL:
        configs["all_identical"] = (np.zeros((5, 2)), np.ones((5, 2)))
        configs["all_collapsed_to_point"] = (np.full((5, 2), 0.5), np.full((5, 2), 0.5))
        n_line = 6
        line_lo = np.stack([np.arange(n_line, dtype=np.float64), np.zeros(n_line)], axis=1)
        line_hi = line_lo + np.array([1.0, 1.0])
        configs["line_arrangement"] = (line_lo, line_hi)

    query = AABB(np.array([0.4, 0.4]), np.array([0.6, 0.6]))
    # Every config here satisfies hi >= lo, is finite, and has ndim >= 1;
    # `n_cells == 0` is explicitly legal and returns an empty tree.
    for name, (lo, hi) in configs.items():
        yield Case(
            GROUP,
            f"bvh_degenerate_{name}",
            BVH.from_cell_bounds,
            lambda lo=lo, hi=hi: BVH.from_cell_bounds(lo, hi),
            {"kind": name, "n_cells": int(lo.shape[0])},
            invariants=(
                custom(
                    "n-nodes-formula",
                    _bvh_n_nodes_predicate(int(lo.shape[0])),
                ),
            ),
            must_succeed=True,
        )
        if lo.shape[0] > 0:
            yield Case(
                GROUP,
                f"bvh_degenerate_query_{name}",
                BVH.query_aabb,
                lambda lo=lo, hi=hi: BVH.from_cell_bounds(lo, hi).query_aabb(query),
                {"kind": name, "n_cells": int(lo.shape[0])},
                invariants=(custom("query-completeness", _bvh_query_predicate(lo, hi, query)),),
                must_succeed=True,
            )

    if profile is not Profile.FULL:
        return
    bad_hi = np.array([[0.5, 0.5]])
    bad_lo = np.array([[1.0, 1.0]])
    yield Case(
        GROUP,
        "bvh_from_cell_bounds_hi_below_lo",
        BVH.from_cell_bounds,
        lambda: BVH.from_cell_bounds(bad_lo, bad_hi),
        {"kind": "hi-below-lo"},
        # "ValueError: If shapes are inconsistent, ndim is < 1, any cell has
        # hi < lo, ..."
        must_reject=True,
    )
    # Still `must_reject`, though the code rejects non-finite corners at
    # `_bvh.py:322-326` while `from_cell_bounds`'s `Raises:` section does not
    # list non-finiteness. A NaN corner is not a box, so refusing it is right
    # and returning would be the finding -- what is missing is the `Raises:`
    # entry, a documentation gap rather than a reason to withhold the flag.
    yield Case(
        GROUP,
        "bvh_from_cell_bounds_nonfinite",
        BVH.from_cell_bounds,
        lambda: BVH.from_cell_bounds(np.array([[np.nan, 0.0]]), np.array([[1.0, 1.0]])),
        {"kind": "nan"},
        finite_inputs=False,
        must_reject=True,
    )


# ---------------------------------------------------------------------------
# Partition / partition_grid
# ---------------------------------------------------------------------------


def _partition_direct_cases(profile: Profile) -> Iterator[Case]:
    """Yield direct :class:`Partition` constructor and accessor cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: ``n_parts == 1``, ``n_parts == n_cells``, and malformed-input
        rejections.
    """
    yield Case(
        GROUP,
        "partition_direct_single_rank",
        Partition,
        lambda: Partition(np.zeros(5, dtype=np.int32), 1),
        {"kind": "n_parts=1"},
        invariants=(
            custom(
                "owns-everything",
                lambda r: None
                if np.array_equal(r.owned_cells(0), np.arange(5))
                else "rank 0 does not own every cell",
            ),
        ),
        must_succeed=True,
    )
    owner_one_each = np.arange(5, dtype=np.int32)
    yield Case(
        GROUP,
        "partition_direct_one_cell_per_rank",
        Partition,
        lambda: Partition(owner_one_each, 5),
        {"kind": "n_parts=n_cells"},
        invariants=(
            custom(
                "every-rank-owns-one",
                lambda r: None
                if all(r.owned_cells(rank).size == 1 for rank in range(5))
                else "some rank does not own exactly one cell",
            ),
        ),
        must_succeed=True,
    )

    if profile is not Profile.FULL:
        return
    # "ValueError: If n_parts < 1, cell_owner is not 1D integer, or any owner is
    # outside [-1, n_parts)."
    yield Case(
        GROUP,
        "partition_direct_zero_parts",
        Partition,
        lambda: Partition(np.zeros(3, dtype=np.int32), 0),
        {},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "partition_direct_owner_out_of_range",
        Partition,
        lambda: Partition(np.array([0, 1, 2], dtype=np.int32), 2),
        {},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "partition_direct_non_integer",
        Partition,
        lambda: Partition(np.array([0.0, 1.0]), 2),
        {},
        must_reject=True,
    )
    part = Partition(np.array([0, 1], dtype=np.int32), 2)
    yield Case(
        GROUP,
        "partition_direct_owned_cells_bad_rank",
        Partition.owned_cells,
        lambda: part.owned_cells(5),
        {"kind": "rank-oob"},
        must_reject=True,  # "ValueError: If rank is outside [0, n_parts)."
    )


def _partition_active_predicate(
    n_parts: int, active_mask: npt.NDArray[np.bool_] | None
) -> Callable[[object], str | None]:
    """Build a predicate asserting owners exactly partition the active set.

    Args:
        n_parts (int): Requested part count.
        active_mask (npt.NDArray[np.bool_] | None): Activity mask, or ``None`` when
            every cell is active.

    Returns:
        Callable[[object], str | None]: Predicate over the resulting
        :class:`Partition`.
    """

    def predicate(result: object) -> str | None:
        owner = result.cell_owner  # type: ignore[attr-defined]
        if active_mask is not None and np.any(owner[~active_mask] != -1):
            return "an inactive cell was assigned an owner"
        active_owner = owner if active_mask is None else owner[active_mask]
        out_of_range = active_owner.size and (
            int(active_owner.min()) < 0 or int(active_owner.max()) >= n_parts
        )
        if out_of_range:
            return f"an active cell owner falls outside [0, {n_parts})"
        return None

    return predicate


def _every_rank_nonempty_predicate(n_parts: int) -> Callable[[object], str | None]:
    """Build a predicate asserting every rank owns at least one cell.

    Args:
        n_parts (int): Requested part count.

    Returns:
        Callable[[object], str | None]: Predicate over the resulting
        :class:`Partition`.
    """

    def predicate(result: object) -> str | None:
        for rank in range(n_parts):
            if result.owned_cells(rank).size == 0:  # type: ignore[attr-defined]
                return f"rank {rank} owns zero cells"
        return None

    return predicate


def _partition_grid_cases(profile: Profile) -> Iterator[Case]:
    """Yield :func:`pantr.grid.partition_grid` corner cases across backends.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: ``n_parts == n_cells``, ``n_parts == 1``, an all-but-one-inactive
        selection, and malformed-input rejections.
    """
    grid = uniform_grid([[0.0, 1.0], [0.0, 1.0]], [3, 3])
    backends = ("block", "rcb", "auto") if profile is Profile.FULL else ("auto",)
    for backend in backends:
        # The grid is 3x3 = 9 cells and 9 = 3 * 3 factors onto the two axes, so
        # even the "block" backend is satisfiable here.
        yield Case(
            GROUP,
            f"partition_grid_n_parts_eq_n_cells_{backend}",
            partition_grid,
            lambda backend=backend: partition_grid(grid, grid.num_cells, backend=backend),
            {"backend": backend, "n_parts": grid.num_cells},
            invariants=(
                custom("every-rank-nonempty", _every_rank_nonempty_predicate(grid.num_cells)),
                custom("partitions-active-set", _partition_active_predicate(grid.num_cells, None)),
            ),
            must_succeed=True,
        )
        yield Case(
            GROUP,
            f"partition_grid_single_rank_{backend}",
            partition_grid,
            lambda backend=backend: partition_grid(grid, 1, backend=backend),
            {"backend": backend, "n_parts": 1},
            invariants=(
                custom("every-rank-nonempty", _every_rank_nonempty_predicate(1)),
                custom("partitions-active-set", _partition_active_predicate(1, None)),
            ),
            must_succeed=True,
        )

    if profile is not Profile.FULL:
        return

    active = np.zeros(grid.num_cells, dtype=bool)
    active[0] = True
    yield Case(
        GROUP,
        "partition_grid_all_but_one_inactive",
        partition_grid,
        lambda: partition_grid(grid, 1, backend="rcb", cell_active=active),
        {"backend": "rcb", "n_parts": 1, "kind": "all-but-one-inactive"},
        invariants=(custom("partitions-active-set", _partition_active_predicate(1, active)),),
        must_succeed=True,
    )
    # "ValueError: If n_parts < 1; if backend is unknown; if 'block' is used on
    # a non-TensorProductGrid or with weights/activity; if n_parts cannot be
    # factored onto the axes ('block'); ... or if n_parts exceeds the number of
    # active cells ('rcb')."
    yield Case(
        GROUP,
        "partition_grid_zero_parts",
        partition_grid,
        lambda: partition_grid(grid, 0),
        {},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "partition_grid_unknown_backend",
        partition_grid,
        lambda: partition_grid(grid, 2, backend="nope"),
        {},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "partition_grid_block_rejects_weights",
        partition_grid,
        lambda: partition_grid(grid, 3, backend="block", cell_weights=np.ones(grid.num_cells)),
        {},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "partition_grid_block_unfactorable",
        partition_grid,
        lambda: partition_grid(grid, 5, backend="block"),
        {"kind": "prime-part-count"},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "partition_grid_n_parts_exceeds_active",
        partition_grid,
        lambda: partition_grid(grid, grid.num_cells + 1, backend="rcb"),
        {"kind": "n_parts-too-large"},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "partition_grid_rcb_on_hierarchical",
        partition_grid,
        lambda: partition_grid(HierarchicalGrid(grid, 2), 3, backend="rcb"),
        {"kind": "non-tensor-product"},
        invariants=(
            custom(
                "every-rank-nonempty",
                _every_rank_nonempty_predicate(3),
            ),
        ),
        must_succeed=True,  # "'rcb' accepts any grid."
    )


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------


def _overlay_symmetry_predicate() -> Callable[[object], str | None]:
    """Build a predicate asserting ``overlay(a, b)`` and ``overlay(b, a)`` agree.

    Args:
        None

    Returns:
        Callable[[object], str | None]: Predicate over a ``(overlay_ab, overlay_ba)``
        tuple.
    """
    atol = get_default(np.float64)
    # The merge itself tolerates `atol`-close breakpoints; comparing two
    # independent merges of the same data allows the same slack again, times 2
    # for the two roundings (one per merge direction).
    tol = 2.0 * atol

    def predicate(result: object) -> str | None:
        overlay_ab, overlay_ba = result  # type: ignore[misc]
        for d in range(overlay_ab.ndim):
            bp_ab, bp_ba = overlay_ab.breakpoints[d], overlay_ba.breakpoints[d]
            if bp_ab.shape != bp_ba.shape:
                return (
                    f"axis {d}: overlay(a,b) has {bp_ab.shape[0]} breakpoints, "
                    f"overlay(b,a) has {bp_ba.shape[0]}"
                )
            diff = float(np.max(np.abs(bp_ab - bp_ba)))
            if diff > tol:
                return f"axis {d}: overlay(a,b) vs overlay(b,a) differ by {diff:.3e} > {tol:.3e}"
        return None

    return predicate


def _overlay_containment_predicate(
    grid_a: TensorProductGrid, grid_b: TensorProductGrid
) -> Callable[[object], str | None]:
    """Build a predicate asserting every overlay cell lies inside one cell of each input.

    Args:
        grid_a (TensorProductGrid): First input grid.
        grid_b (TensorProductGrid): Second input grid.

    Returns:
        Callable[[object], str | None]: Predicate over a ``(overlay_ab, overlay_ba)``
        tuple (only ``overlay_ab`` is checked; symmetric by construction).
    """

    def predicate(result: object) -> str | None:
        overlay_ab, _ = result  # type: ignore[misc]
        cell_lo, cell_hi = overlay_ab.collect_cell_bounds()
        centroids = 0.5 * (cell_lo + cell_hi)
        if np.any(grid_a.locate_many(centroids) < 0):
            return "an overlay cell centroid falls outside grid_a"
        if np.any(grid_b.locate_many(centroids) < 0):
            return "an overlay cell centroid falls outside grid_b"
        return None

    return predicate


def _overlay_cases(profile: Profile) -> Iterator[Case]:
    """Yield :func:`pantr.grid.overlay` cases across dimension and domain.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: Symmetry and containment checks per ``(ndim, domain)``, plus the
        malformed-input rejections.
    """
    ndims = dims(profile, max_dim=3) if profile is Profile.FULL else (2,)
    for ndim in ndims:
        for domain in domains(profile):
            grid_a = _tp_grid(ndim, domain, 3)
            grid_b = _tp_grid(ndim, domain, 4)
            tag = _grid_tag(ndim, domain)

            def run(
                grid_a: TensorProductGrid = grid_a, grid_b: TensorProductGrid = grid_b
            ) -> tuple[TensorProductGrid, TensorProductGrid]:
                return overlay(grid_a, grid_b), overlay(grid_b, grid_a)

            # Both grids share ndim and the identical domain, so the per-axis
            # intersection is the whole domain.
            yield Case(
                GROUP,
                f"overlay_symmetry_{tag}",
                overlay,
                run,
                {"ndim": ndim, "domain": domain},
                invariants=(custom("overlay-symmetric", _overlay_symmetry_predicate()),),
                must_succeed=True,
            )
            yield Case(
                GROUP,
                f"overlay_containment_{tag}",
                overlay,
                run,
                {"ndim": ndim, "domain": domain},
                invariants=(
                    custom(
                        "overlay-contained",
                        _overlay_containment_predicate(grid_a, grid_b),
                    ),
                ),
                must_succeed=True,
            )

    if profile is not Profile.FULL:
        return
    a = _tp_grid(2, (0.0, 1.0), 2)
    b = _tp_grid(2, (2.0, 3.0), 2)
    # "ValueError: If the grids have different ndim, or if their domains do not
    # overlap on some axis"
    yield Case(
        GROUP,
        "overlay_disjoint_domains",
        overlay,
        lambda: overlay(a, b),
        {"kind": "disjoint"},
        must_reject=True,
    )
    c = _tp_grid(3, (0.0, 1.0), 2)
    yield Case(
        GROUP,
        "overlay_ndim_mismatch",
        overlay,
        lambda: overlay(a, c),
        {"kind": "ndim-mismatch"},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "overlay_wrong_type",
        overlay,
        lambda: overlay(a, object()),
        {"kind": "wrong-type"},
        must_reject=True,  # "TypeError: If either argument is not a TensorProductGrid."
    )


# ---------------------------------------------------------------------------
# CellTags / FacetTags
# ---------------------------------------------------------------------------


def _tags_cases(profile: Profile) -> Iterator[Case]:
    """Yield :class:`CellTags` / :class:`FacetTags` round-trip and overflow cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: A set/getitem/to_dense round trip, the documented ``to_dense``
        overflow, and malformed-input rejections.
    """
    num_cells = 6

    def cell_round_trip() -> npt.NDArray[np.int64]:
        tags = CellTags(num_cells)
        tags.set("labels", np.array([1, 3, 5]), np.array([10, 30, 50]))
        return tags.to_dense("labels", fill=-1)

    def check_cell_round_trip(result: object) -> str | None:
        expected = np.array([-1, 10, -1, 30, -1, 50])
        if np.array_equal(result, expected):
            return None
        return f"to_dense() = {result}, expected {expected}"

    yield Case(
        GROUP,
        "cell_tags_round_trip",
        CellTags.to_dense,
        cell_round_trip,
        {"kind": "round-trip"},
        invariants=(custom("round-trip-values", check_cell_round_trip),),
        must_succeed=True,
    )

    def cell_overflow() -> npt.NDArray[Any]:
        tags = CellTags(num_cells)
        tags.set("big", np.array([0]), np.array([1000]))
        return tags.to_dense("big", dtype=np.int8)

    # "OverflowError: If any stored value cannot be represented exactly in
    # dtype". `must_reject` is still the right flag for a non-ValueError
    # rejection: it grades *returning*, and silently truncating 1000 to an
    # int8 is exactly the finding.
    yield Case(
        GROUP,
        "cell_tags_to_dense_overflow",
        CellTags.to_dense,
        cell_overflow,
        {"kind": "int8-overflow"},
        must_reject=True,
    )

    facets_per_cell = 4
    num_facet_cells = 3
    tagged_value_a = 7
    tagged_value_b = 9

    def facet_round_trip() -> npt.NDArray[np.int64]:
        tags = FacetTags(num_facet_cells, facets_per_cell)
        tags.set("bc", np.array([[0, 1], [2, 3]]), np.array([tagged_value_a, tagged_value_b]))
        return tags.to_dense("bc", fill=-1)

    def check_facet_round_trip(result: object) -> str | None:
        arr = np.asarray(result)
        if arr[0, 1] != tagged_value_a or arr[2, 3] != tagged_value_b:
            return f"facet round trip landed wrong: {arr}"
        return None

    yield Case(
        GROUP,
        "facet_tags_round_trip",
        FacetTags.to_dense,
        facet_round_trip,
        {"kind": "round-trip"},
        invariants=(custom("round-trip-values", check_facet_round_trip),),
        must_succeed=True,
    )

    def facet_overflow() -> npt.NDArray[Any]:
        tags = FacetTags(num_facet_cells, facets_per_cell)
        tags.set("big", np.array([[0, 0]]), np.array([1000]))
        return tags.to_dense("big", dtype=np.int8)

    yield Case(
        GROUP,
        "facet_tags_to_dense_overflow",
        FacetTags.to_dense,
        facet_overflow,
        {"kind": "int8-overflow"},
        must_reject=True,
    )

    if profile is not Profile.FULL:
        return
    # "ValueError: If ids is not 1-D, contains duplicates, or has an id out of
    # range"
    yield Case(
        GROUP,
        "cell_tags_duplicate_ids",
        CellTags.set,
        lambda: CellTags(num_cells).set("dup", np.array([1, 1]), np.array([1, 2])),
        {"kind": "duplicate-ids"},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "cell_tags_id_out_of_range",
        CellTags.set,
        lambda: CellTags(num_cells).set("oob", np.array([num_cells]), np.array([1])),
        {"kind": "id-oob"},
        must_reject=True,
    )
    # "ValueError: If keys does not have shape (M, 2), contains a duplicate or
    # out-of-range key, ..."
    yield Case(
        GROUP,
        "facet_tags_lfid_out_of_range",
        FacetTags.set,
        lambda: FacetTags(num_facet_cells, facets_per_cell).set(
            "oob", np.array([[0, facets_per_cell]]), np.array([1])
        ),
        {"kind": "lfid-oob"},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "facet_tags_duplicate_keys",
        FacetTags.set,
        lambda: FacetTags(num_facet_cells, facets_per_cell).set(
            "dup", np.array([[0, 0], [0, 0]]), np.array([1, 2])
        ),
        {"kind": "duplicate-keys"},
        must_reject=True,
    )


# ---------------------------------------------------------------------------
# cell_quadrature
# ---------------------------------------------------------------------------


def _quadrature_weight_sum_predicate(
    cell_lo: npt.NDArray[np.float64], cell_hi: npt.NDArray[np.float64], num_points: int
) -> Callable[[object], str | None]:
    """Build a predicate asserting per-cell weights sum to the cell volume.

    Only valid for a rule whose reference weights sum to 1
    (``gauss_legendre_quadrature`` does); the docstring's claim
    (`_cell_quadrature.py:8-10`) is conditioned on exactly that.

    Args:
        cell_lo (npt.NDArray[np.float64]): Selected cells' lower corners.
        cell_hi (npt.NDArray[np.float64]): Selected cells' upper corners.
        num_points (int): Quadrature point count per cell.

    Returns:
        Callable[[object], str | None]: Predicate over the ``(points, weights)``
        result.
    """
    volumes = _volume(cell_lo, cell_hi)
    ndim = cell_lo.shape[1]
    # Each cell's weight sum is `num_points` reference weights (summing to 1 up to
    # their own O(num_points * eps) rounding) scaled by a volume computed as an
    # `ndim`-fold product (one rounding per factor). Factor 8 covers both constants.
    scale = np.maximum(np.abs(volumes), 1.0)
    tol = 8.0 * num_points * ndim * _EPS64 * scale

    def predicate(result: object) -> str | None:
        _, weights = result  # type: ignore[misc]
        sums = np.sum(weights, axis=1)
        bad = np.abs(sums - volumes) > tol
        if np.any(bad):
            cid = int(np.argmax(np.abs(sums - volumes) - tol))
            return (
                f"cell {cid}: weight sum {sums[cid]:.6e} vs volume {volumes[cid]:.6e} "
                f"(tol {tol[cid]:.3e})"
            )
        return None

    return predicate


def _quadrature_points_shape_predicate(shape: tuple[int, ...]) -> Callable[[object], str | None]:
    """Build a predicate checking the ``points`` half of a ``cell_quadrature`` result.

    ``expected_shape`` (`_core.py`) shapes its check for a single array; ``cell_quadrature``
    returns a ``(points, weights)`` pair, so this checks ``result[0]`` directly instead of
    feeding the whole tuple to ``np.shape``.

    Args:
        shape (tuple[int, ...]): Expected ``points`` shape.

    Returns:
        Callable[[object], str | None]: Predicate over the ``(points, weights)`` result.
    """

    def predicate(result: object) -> str | None:
        points, _ = result  # type: ignore[misc]
        got = tuple(points.shape)
        if got != shape:
            return f"points.shape={got}, expected {shape}"
        return None

    return predicate


def _cell_quadrature_cases(profile: Profile) -> Iterator[Case]:
    """Yield :func:`pantr.grid.cell_quadrature` cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: Weight-sum-to-volume checks over several cell selections, plus the
        malformed-input rejections.
    """
    ndims = dims(profile, max_dim=3) if profile is Profile.FULL else (2,)
    for ndim in ndims:
        grid = _tp_grid(ndim, (0.0, 5.0), 3)
        rule = gauss_legendre_quadrature(ndim, 3)
        cell_lo_all, cell_hi_all = grid.collect_cell_bounds()

        selectors: dict[str, npt.NDArray[np.int64] | None] = {
            "all_cells": None,
            "single_cell": np.array([0]),
            "empty_selection": np.zeros(0, dtype=np.int64),
        }
        # The empty selection is an explicitly-typed empty int64 array, which
        # vacuously satisfies "each in [0, num_cells)"; all three below are
        # legal.
        for name, ids in selectors.items():
            expected_lo = cell_lo_all if ids is None else cell_lo_all[ids]
            expected_hi = cell_hi_all if ids is None else cell_hi_all[ids]
            yield Case(
                GROUP,
                f"cell_quadrature_{name}_d{ndim}",
                cell_quadrature,
                lambda grid=grid, rule=rule, ids=ids: cell_quadrature(grid, rule, ids),
                {"ndim": ndim, "kind": name},
                invariants=(
                    custom(
                        "weight-sums-to-volume",
                        _quadrature_weight_sum_predicate(expected_lo, expected_hi, rule.num_points),
                    ),
                    custom(
                        "points-shape",
                        _quadrature_points_shape_predicate(
                            (expected_lo.shape[0], rule.num_points, ndim)
                        ),
                    ),
                )
                if name != "empty_selection"
                else (),
                must_succeed=True,
            )

    if profile is not Profile.FULL:
        return
    grid2 = _tp_grid(2, (0.0, 1.0), 2)
    rule3 = gauss_legendre_quadrature(3, 2)
    # "ValueError: If rule.ndim != grid.ndim, cells is not a 1D integer
    # array-like, or any id is outside [0, num_cells)."
    yield Case(
        GROUP,
        "cell_quadrature_ndim_mismatch",
        cell_quadrature,
        lambda: cell_quadrature(grid2, rule3),
        {"kind": "ndim-mismatch"},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "cell_quadrature_cell_out_of_range",
        cell_quadrature,
        lambda: cell_quadrature(
            grid2, gauss_legendre_quadrature(2, 2), np.array([grid2.num_cells])
        ),
        {"kind": "cell-oob"},
        must_reject=True,
    )
    yield Case(
        GROUP,
        "cell_quadrature_non_integer_cells",
        cell_quadrature,
        lambda: cell_quadrature(grid2, gauss_legendre_quadrature(2, 2), np.array([0.5])),
        {"kind": "non-integer-cells"},
        must_reject=True,
    )


# ---------------------------------------------------------------------------
# Registry entry point
# ---------------------------------------------------------------------------


def cases(profile: Profile) -> Iterator[Case]:
    """Yield every case in this group.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: The group's cases.
    """
    yield from _tp_construction_cases(profile)
    yield from _tp_uniformity_cases(profile)
    yield from _tp_locate_cases(profile)
    yield from _tp_neighbor_cases(profile)
    yield from _tp_restrict_cases(profile)
    yield from _uniform_grid_factory_cases(profile)
    yield from _tensor_product_from_space_cases(profile)

    yield from _hier_construction_cases(profile)
    yield from _hier_refine_coarsen_cases(profile)
    yield from _hier_bounds_cases(profile)
    yield from _hier_locate_cases(profile)
    yield from _hier_mask_cases(profile)
    yield from _hier_hanging_neighbor_cases(profile)
    yield from _hier_restrict_cases(profile)
    yield from _hierarchical_grid_factory_case(profile)

    yield from _bvh_from_grid_cases(profile)
    yield from _bvh_direct_construction_cases(profile)
    yield from _bvh_degenerate_cases(profile)

    yield from _partition_direct_cases(profile)
    yield from _partition_grid_cases(profile)

    yield from _overlay_cases(profile)

    yield from _tags_cases(profile)

    yield from _cell_quadrature_cases(profile)
