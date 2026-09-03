"""Parity of the C++ `HierarchicalGrid` against the Python oracle it was ported from.

`tests/parity/test_grid.py` covers the three hierarchical *kernels* -- the free functions
both backends call. This module covers the *type*: `pantr.grid.HierarchicalGrid` is now a
wrapper over a handle, and what has to agree is a whole object's behaviour, including the
four operations that build one from another.

Why this file is a sweep and not a table of fixtures
----------------------------------------------------

The refinement half of this port carried a defect that no hand-written fixture found and
that a small sweep did not either: `coarsen_cells` demoted parents in *descending* level
order where the oracle demotes them ascending, which changes the flat cell numbering of
the result and nothing else. It first appeared at **case 3553 of a 4000-case random
sweep**, and it was invisible at 400. Three properties made it that hard to see: it needs
two families at different levels named in one call, it needs the shallower demotion to
change the deeper one's block partition, and it shows up only in ids -- every count and
every coordinate still agrees.

So the shape of this module is fixed by that defect rather than chosen: a **generative
sweep** over random hierarchies and random operation sequences, comparing everything both
backends can answer, cell by cell. The fixtures below the sweep exist for the properties a
random draw would only reach by accident, not as the main check.

`_sweep` is one function called twice: at :data:`_SHIPPED_CASES` by the test that runs in
CI, and at ten times that by a ``slow``-marked test that does not. The ticket asks for the
bound to be verified over a sweep at least ten times the one that ships; keeping the wide
run **in the repository** rather than in a shell history is what makes "verified" a thing
the next reader can re-run.

What agrees, quantity by quantity
----------------------------------

The split is per quantity with an argument for each, and it is not one decision applied in
bulk.

**Exactly** -- every count, level, multi-index, block corner, mask entry, cell id,
connectivity row, boundary facet, restriction map and ``repr``. Each of these is a
*verdict* rather than a displaced value: `design/backend_parity.md` Rule 11 states that no
tolerance bounds one, and a bounded comparison could not even see two answers of different
length. This is also the only bar the determinism rule admits here, because finite
precision does not bite: they are integers, indices, counts and flags.

**Bit for bit** -- the cell corners, the collected bounds and the exported vertices.
Three floating-point quantities, and the claim is the strong one because the two
implementations evaluate the **same expression in the same order**, statement for
statement, with every product *named* rather than inlined:

    size = (root_hi - root_lo) / factor ** level
    offset = sub_index * size
    lo = root_lo + offset
    hi = lo + size

`-ffp-contract=on` confines fusion to a single expression, so a named product cannot be
contracted on any target, and numpy never fuses. That is the same mechanism
`tests/parity/test_grid.py` records for the descent, and it is load-bearing rather than
decorative there: the descent's truncation turns this value into a **cell id**, so one
differing bit is a differing verdict. Claiming a bound on the corners while the descent
needs the bits would assert less for no gain, and a break would be actionable -- it would
mean the two sides had stopped matching expression for expression, which is a finding and
not a rebuild artefact.

**Against an exact rational oracle** -- the corners again, this time on their own rather
than against each other. See `test_cell_bounds_match_the_exact_rational_corner`. That is
the independent accuracy check, it is asserted **per backend**, and its bound is derived
below rather than fitted.

Nothing here is licensed by Rule 8. That rule is an **excluding** gate -- it says where a
parity claim may not be made at all -- and it does not make one true.

The extension's absence is a skip, not a collection error
----------------------------------------------------------

There is no module-level ``from pantr import _pantr_cpp`` and no module-level
``pytestmark``. The extension is reached through :func:`_bindings`, and the gate is on the
**C++ parameter** of each parametrized test rather than on the test, so that in an
installation without the extension the Python half of every per-backend property still
runs. `test_bspline_bezier_extraction.py` has the same shape for the same reason.
"""

from __future__ import annotations

import itertools
import pickle
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Final, NamedTuple, TypeVar

import numpy as np
import pytest

from pantr._backend import Backend, use_backend
from pantr.grid import HierarchicalGrid, TensorProductGrid, hierarchical_grid
from tests._parity_harness import (
    Field,
    assert_accuracy,
    assert_object_parity,
    bitwise_parity,
    demand_cpp_backend,
    derived_accuracy,
    exact_parity,
    unit_roundoff,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

_T = TypeVar("_T")

_U: Final = unit_roundoff(np.float64)
"""Half an ulp: the relative error of one correctly rounded ``float64`` operation.

Taken from :func:`tests._parity_harness.unit_roundoff` rather than spelled out again.
The quantity is the project's, not this file's, and two spellings of one bound drift
apart the moment either is corrected.
"""

_SHIPPED_CASES: Final = 500
"""Random cases the CI sweep draws.

Chosen for runtime, not for confidence: the sweep compares roughly thirty quantities per
case in interpreted loops, and this is what keeps it a few seconds. The confidence comes
from :func:`test_the_wide_sweep_agrees`, which runs ten times this and is what the ticket
asks be verified; the shipped run is the regression guard that has to stay affordable.
"""

_WIDE_FACTOR: Final = 10
"""How much wider the ``slow`` sweep is than the shipped one."""

_BACKENDS: Final = (
    pytest.param(Backend.PYTHON, id="python"),
    pytest.param(Backend.CPP, id="cpp"),
)
"""The two backends, for the tests that state a property of each one separately."""

_VERDICT_WHY: Final = (
    "cell ids, levels, multi-indices, block corners, mask entries, counts and "
    "connectivity rows are verdicts rather than displaced values; "
    "design/backend_parity.md Rule 11 is explicit that no tolerance bounds one, and a "
    "bounded comparison could not see two answers of different length at all"
)

_CORNER_WHY: Final = (
    "both backends build a corner as the same four named operations in the same order "
    "-- size = (root_hi - root_lo) / factor ** level, offset = sub_index * size, "
    "lo = root_lo + offset, hi = lo + size -- and naming the products is what stops "
    "-ffp-contract=on fusing either of them, since contraction is confined to a single "
    "expression. The descent turns this same value into a cell id, so one differing bit "
    "is a differing verdict rather than a displacement"
)


def _bindings() -> Any:
    """Import the extension, deferred and in one place.

    Module level would break this file's whole point. The tests that state a property of
    *each* backend are meant to run their Python half in an installation with no
    extension, and a top-level ``from pantr import _pantr_cpp`` turns that into a
    collection error for the file -- every test, both halves. So the import is here, and
    every caller is gated on the extension being present.

    Returns:
        Any: The :mod:`pantr._pantr_cpp` module.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    return _pantr_cpp


def _demand_the_extension_if_needed(backend: Backend) -> None:
    """Require the compiled extension, and only for the half that uses it.

    A test parametrized over both backends and *also* taking the ``cpp_backend`` fixture
    skips **both** halves when the extension is absent, which silently drops the Python
    half -- and the Python half of the exactness check is the only thing here that would
    catch the oracle regressing. Marking the parameter would be neater and pytest forbids
    it: ``pytest.param`` refuses ``pytest.mark.usefixtures``.

    Args:
        backend (Backend): The backend this case runs under.
    """
    if backend is Backend.CPP:
        demand_cpp_backend()


def _both(call: Callable[[], _T]) -> tuple[_T, _T]:
    """Run one callable under each backend and return ``(python, cpp)``.

    Args:
        call (Callable[[], _T]): What to run.

    Returns:
        tuple[_T, _T]: The Python backend's result, then the C++ backend's.
    """
    with use_backend(Backend.PYTHON):
        reference = call()
    with use_backend(Backend.CPP):
        actual = call()
    return reference, actual


def _gamma(m: int) -> float:
    """Return ``gamma_m = m u / (1 - m u)``, the standard product-of-roundings factor.

    Args:
        m (int): The number of roundings.

    Returns:
        float: The factor.
    """
    return m * _U / (1.0 - m * _U)


# ---------------------------------------------------------------------------
# The cases: a hierarchy, then a literal sequence of operations
# ---------------------------------------------------------------------------


class _Case(NamedTuple):
    """One hierarchy and the operations to replay on it, as literal arguments.

    The operations are recorded as *concrete* calls rather than as a rule for drawing
    them, and that is what makes the case replayable: the arguments a random draw picks
    depend on the grid it drew them from, so drawing them again under the other backend
    would be a different experiment. Recording them once, under the oracle, and replaying
    the same literal calls means a backend that disagrees about ``num_cells`` or about a
    level's extent shows up as a failure rather than as a differently shaped draw.

    Attributes:
        breakpoints (tuple[tuple[float, ...], ...]): The root's per-axis breakpoints.
        factor (tuple[int, ...]): The per-direction subdivision factor.
        operations (tuple[tuple[str, tuple[Any, ...]], ...]): Method name and literal
            arguments, in application order. Only the calls that actually applied.
        restrict_ids (tuple[int, ...]): The cell ids to restrict the final grid over.
    """

    breakpoints: tuple[tuple[float, ...], ...]
    factor: tuple[int, ...]
    operations: tuple[tuple[str, tuple[Any, ...]], ...]
    restrict_ids: tuple[int, ...]


def _draw_case(rng: np.random.Generator) -> _Case:
    """Draw one random hierarchy and a random sequence of operations on it.

    Small on purpose. One to three axes, one to three root cells per axis, a subdivision
    factor per axis drawn from ``{1, 2, 3}`` with at least one axis subdividing, and one
    to five operations. A large grid does not find more here: the defects this sweep
    exists for live in how blocks are *split, peeled and merged*, and an irregular
    hierarchy of thirty cells exercises that harder per second than a regular one of
    three thousand.

    A factor of ``1`` on an axis is drawn deliberately: it is the anisotropic case where
    a level's extent does not grow on that axis, and it is where an index computed as
    ``midx // factor ** level`` and one computed as a running product part company.

    The operations are drawn against the *oracle*, and only those that applied are kept:
    a random box is often invalid -- refusing it is covered by the unit tests -- and a
    case whose sequence is half refusals compares an almost-unrefined grid.

    Args:
        rng (np.random.Generator): The source of randomness.

    Returns:
        _Case: The case, with literal arguments.
    """
    ndim = int(rng.integers(1, 4))
    factor = [int(rng.choice([1, 2, 3])) for _ in range(ndim)]
    if all(f == 1 for f in factor):
        factor[int(rng.integers(0, ndim))] = 2
    breakpoints = []
    for _ in range(ndim):
        interior = rng.uniform(-1.0, 1.0, int(rng.integers(1, 4)) - 1)
        breakpoints.append(tuple(np.sort(np.concatenate([[-1.0, 1.0], interior])).tolist()))

    with use_backend(Backend.PYTHON):
        grid = hierarchical_grid(TensorProductGrid([list(bp) for bp in breakpoints]), factor)
        applied: list[tuple[str, tuple[Any, ...]]] = []
        for _ in range(int(rng.integers(1, 6))):
            name = str(rng.choice(["refine", "refine", "refine_cells", "coarsen", "coarsen_cells"]))
            if name in ("refine", "coarsen"):
                level = int(rng.integers(0, grid.max_level + 1))
                extent = grid.level_cells_per_axis(level)
                lo = tuple(int(rng.integers(0, extent[k])) for k in range(ndim))
                hi = tuple(int(rng.integers(lo[k] + 1, extent[k] + 1)) for k in range(ndim))
                args: tuple[Any, ...] = (level, lo, hi)
            else:
                count = int(rng.integers(1, min(grid.num_cells, 8) + 1))
                args = (tuple(sorted({int(i) for i in rng.integers(0, grid.num_cells, count)})),)
            try:
                grid = getattr(grid, name)(*args)
            except ValueError:
                continue  # an invalid box; the refusal paths are the unit tests' job
            applied.append((name, args))
        count = int(rng.integers(1, min(grid.num_cells, 5) + 1))
        requested = tuple(sorted({int(i) for i in rng.integers(0, grid.num_cells, count)}))

    return _Case(tuple(breakpoints), tuple(factor), tuple(applied), requested)


def _build(case: _Case) -> HierarchicalGrid:
    """Replay a case in whichever backend is active.

    Args:
        case (_Case): The case.

    Returns:
        HierarchicalGrid: The final grid.
    """
    grid = hierarchical_grid(TensorProductGrid([list(bp) for bp in case.breakpoints]), case.factor)
    for name, args in case.operations:
        grid = getattr(grid, name)(*args)
    return grid


# ---------------------------------------------------------------------------
# Readers: one quantity each, flattened so a Field compares one array
# ---------------------------------------------------------------------------


def _cell_levels(grid: HierarchicalGrid) -> npt.NDArray[np.int64]:
    """Every cell's refinement level, in cell-id order.

    Args:
        grid (HierarchicalGrid): The grid.

    Returns:
        npt.NDArray[np.int64]: Shape ``(num_cells,)``.
    """
    return np.array([grid.cell_level(c) for c in range(grid.num_cells)], dtype=np.int64)


def _cell_multi_indices(grid: HierarchicalGrid) -> npt.NDArray[np.int64]:
    """Every cell's per-axis index at its own level, in cell-id order.

    Args:
        grid (HierarchicalGrid): The grid.

    Returns:
        npt.NDArray[np.int64]: Shape ``(num_cells, ndim)``.
    """
    rows = [grid.cell_multi_index(c) for c in range(grid.num_cells)]
    return np.array(rows, dtype=np.int64).reshape(grid.num_cells, grid.ndim)


def _cell_id_round_trip(grid: HierarchicalGrid) -> npt.NDArray[np.int64]:
    """``cell_id(cell_level(c), cell_multi_index(c))`` for every cell.

    The inverse pair, compared as its own quantity rather than assumed from the two
    halves: the two backends could agree on every level and every multi-index and still
    disagree about which ``(level, midx)`` resolves back to which id, because the
    resolution walks the packed block descriptor rather than reading a table.

    Args:
        grid (HierarchicalGrid): The grid.

    Returns:
        npt.NDArray[np.int64]: Shape ``(num_cells,)``; ``-1`` where the round trip
        failed, which is itself a difference worth seeing.
    """
    out = []
    for c in range(grid.num_cells):
        back = grid.cell_id(grid.cell_level(c), grid.cell_multi_index(c))
        out.append(-1 if back is None else back)
    return np.array(out, dtype=np.int64)


def _active_blocks(grid: HierarchicalGrid) -> npt.NDArray[np.int64]:
    """Every level's active-leaf rectangles, flattened in level and stored order.

    The stored order is part of the answer and not an implementation detail: flat cell ids
    are handed out block by block, so two grids whose blocks are the same *set* in a
    different order number their cells differently. That is exactly the shape of the
    `coarsen_cells` defect this module's docstring describes.

    Args:
        grid (HierarchicalGrid): The grid.

    Returns:
        npt.NDArray[np.int64]: Flat corners, with a ``-1`` separator per level so that two
        different level splits of the same rectangles cannot compare equal.
    """
    out: list[int] = []
    for level in range(grid.max_level + 1):
        for lo, hi in grid.active_blocks(level):
            out.extend(lo)
            out.extend(hi)
        out.append(-1)
    return np.array(out, dtype=np.int64)


def _masks(grid: HierarchicalGrid) -> npt.NDArray[np.int64]:
    """Both level masks at every level, flattened, with their shapes interleaved.

    The shape travels with the values because the masks are compared flat: two masks of
    the same total size and different shape are different answers, and a flat comparison
    alone would not see it.

    Args:
        grid (HierarchicalGrid): The grid.

    Returns:
        npt.NDArray[np.int64]: Flat shapes and mask entries.
    """
    out: list[int] = []
    for level in range(grid.max_level + 1):
        out.extend(grid.level_cells_per_axis(level))
        out.append(-1)
        out.extend(int(v) for v in grid.active_leaf_mask(level).ravel())
        out.append(-1)
        out.extend(int(v) for v in grid.subdomain_mask(level).ravel())
        out.append(-1)
    return np.array(out, dtype=np.int64)


def _adjacency(grid: HierarchicalGrid) -> npt.NDArray[np.int64]:
    """Every cell's neighbours and every facet's hanging neighbours, flattened.

    Ragged by nature -- a facet on a refined interface has as many hanging neighbours as
    the interface has finer cells -- so each list is terminated rather than padded, and a
    list of a different length therefore lands on the separator and fails.

    Args:
        grid (HierarchicalGrid): The grid.

    Returns:
        npt.NDArray[np.int64]: Flat, ``-1``-terminated lists.
    """
    out: list[int] = []
    for cid in range(grid.num_cells):
        out.extend(grid.neighbors(cid))
        out.append(-1)
        for lfid in range(grid.num_local_facets(cid)):
            out.extend(grid.hanging_neighbors(cid, lfid))
            out.append(-1)
            axis, side = grid.local_facet_axis_side(cid, lfid)
            across = grid.neighbor_across_facet(cid, lfid)
            out.extend((axis, side, -1 if across is None else across))
            out.append(int(grid.is_mesh_boundary_facet(cid, lfid)))
        out.extend(grid.child_cells(cid))
        out.append(-1)
    return np.array(out, dtype=np.int64)


def _corners(grid: HierarchicalGrid) -> npt.NDArray[np.float64]:
    """Every cell's ``(lo, hi)`` corners from :meth:`cell_bounds`, and its AABB's.

    The AABB is included because it is a *different* path to the same numbers -- a
    generic default that unpacks ``cell_bounds`` into a domain type -- so comparing it
    catches a wrapper that reassembled the box on the wrong side.

    Args:
        grid (HierarchicalGrid): The grid.

    Returns:
        npt.NDArray[np.float64]: Shape ``(num_cells, 4, ndim)``.
    """
    rows = []
    for cid in range(grid.num_cells):
        lo, hi = grid.cell_bounds(cid)
        box = grid.cell_aabb(cid)
        rows.append(np.stack([lo, hi, np.asarray(box.lo), np.asarray(box.hi)]))
    return np.asarray(rows, dtype=np.float64).reshape(grid.num_cells, 4, grid.ndim)


def _probe_points(grid: HierarchicalGrid) -> npt.NDArray[np.float64]:
    """Query points that attack the descent's truncation rather than its interior.

    Every cell's lower corner, its midpoint, and one ulp inside and outside the lower
    corner on axis 0, plus one point far outside. An ordinary interior point agrees under
    almost any defect; a point exactly on a face is where the two descents could part
    company, and the ulp neighbours are where a fused multiply-add in the descent would
    show first.

    Args:
        grid (HierarchicalGrid): The grid.

    Returns:
        npt.NDArray[np.float64]: Shape ``(npts, ndim)``.
    """
    rows: list[npt.NDArray[np.float64]] = []
    for cid in range(grid.num_cells):
        lo, hi = grid.cell_bounds(cid)
        rows.append(lo)
        rows.append(0.5 * (lo + hi))
        for direction in (1.0, -1.0):
            nudged = lo.copy()
            nudged[0] = np.nextafter(nudged[0], direction * np.inf)
            rows.append(nudged)
    rows.append(np.full(grid.ndim, 1e3))
    return np.asarray(rows, dtype=np.float64)


def _located(grid: HierarchicalGrid, points: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    """Batch and single point location, side by side.

    Both, because they are different code paths -- ``locate_many`` runs the compiled
    kernel and ``locate`` walks the descent one point at a time -- and the contract is
    that they agree.

    Args:
        grid (HierarchicalGrid): The grid.
        points (npt.NDArray[np.float64]): The query points.

    Returns:
        npt.NDArray[np.int64]: Shape ``(2, npts)``.
    """
    batch = grid.locate_many(points)
    single = np.array([-1 if (c := grid.locate(p)) is None else c for p in points], dtype=np.int64)
    return np.stack([batch, single])


def _restriction(grid: HierarchicalGrid, ids: tuple[int, ...]) -> npt.NDArray[np.int64]:
    """A restriction's index maps and the sub-grid's own active set, flattened.

    Args:
        grid (HierarchicalGrid): The grid.
        ids (tuple[int, ...]): The cell ids to span.

    Returns:
        npt.NDArray[np.int64]: Flat maps, then the sub-grid's blocks.
    """
    sub, local_to_global, in_subset = grid.restrict(list(ids))
    # An assertion as well as a narrowing: `restrict` promises the same concrete type it
    # was called on, and `GridRestriction.grid` is annotated with the protocol.
    assert isinstance(sub, HierarchicalGrid)
    return np.concatenate(
        [
            np.asarray(local_to_global, dtype=np.int64),
            np.array([-1], dtype=np.int64),
            np.asarray(in_subset, dtype=np.int64),
            np.array([-1], dtype=np.int64),
            _active_blocks(sub),
            np.array([sub.num_cells, sub.max_level], dtype=np.int64),
        ]
    )


def _fields(case: _Case, points: npt.NDArray[np.float64]) -> tuple[Field, ...]:
    """The state two hierarchies have to agree on, and the claim governing each piece.

    Args:
        case (_Case): The case, for the restriction request.
        points (npt.NDArray[np.float64]): The query points, drawn once so that both
            backends are asked the same question.

    Returns:
        tuple[Field, ...]: The fields, in the order a failure should be read.
    """
    return (
        Field("ndim", exact_parity(why=_VERDICT_WHY)),
        Field("num_cells", exact_parity(why=_VERDICT_WHY)),
        Field("max_level", exact_parity(why=_VERDICT_WHY)),
        Field("factor", exact_parity(why=_VERDICT_WHY)),
        Field("root.cells_per_axis", exact_parity(why=_VERDICT_WHY)),
        Field("repr", exact_parity(why=_VERDICT_WHY), read=repr),
        Field("cell_levels", exact_parity(why=_VERDICT_WHY), read=_cell_levels),
        Field("cell_multi_indices", exact_parity(why=_VERDICT_WHY), read=_cell_multi_indices),
        Field("cell_id_round_trip", exact_parity(why=_VERDICT_WHY), read=_cell_id_round_trip),
        Field("active_blocks", exact_parity(why=_VERDICT_WHY), read=_active_blocks),
        Field("level_masks", exact_parity(why=_VERDICT_WHY), read=_masks),
        Field("adjacency", exact_parity(why=_VERDICT_WHY), read=_adjacency),
        Field(
            "boundary_facets",
            exact_parity(why=_VERDICT_WHY),
            read=lambda g: g.boundary_facets(),
        ),
        Field(
            "located",
            exact_parity(why=_VERDICT_WHY),
            read=lambda g: _located(g, points),
        ),
        Field(
            "export.conn",
            exact_parity(why=_VERDICT_WHY),
            read=lambda g: g.export_cells()[1],
        ),
        Field(
            "restriction",
            exact_parity(why=_VERDICT_WHY),
            read=lambda g: _restriction(g, case.restrict_ids),
        ),
        Field("cell_corners", bitwise_parity(why=_CORNER_WHY), read=_corners),
        Field(
            "collected_bounds",
            bitwise_parity(why=_CORNER_WHY),
            read=lambda g: np.stack(g.collect_cell_bounds()),
        ),
        Field(
            "export.points",
            bitwise_parity(why=_CORNER_WHY),
            read=lambda g: g.export_cells()[0],
        ),
        Field(
            "root.breakpoints",
            bitwise_parity(
                why="every breakpoint a grid hands out is a copy of one it was "
                "given, so there is no rounding to bound"
            ),
            read=lambda g: np.concatenate([np.asarray(bp) for bp in g.root.breakpoints]),
        ),
    )


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


class _SweepReport(NamedTuple):
    """What a sweep observed, so a caller can refuse a vacuous one.

    Attributes:
        cases (int): How many cases ran.
        cells (int): Total cells compared.
        vertices (int): Total exported vertices compared.
        coordinates (int): Total floating-point coordinates compared.
        deepest (int): The deepest ``max_level`` any case reached.
        operations (dict[str, int]): How many of each operation actually applied.
    """

    cases: int
    cells: int
    vertices: int
    coordinates: int
    deepest: int
    operations: dict[str, int]


def _sweep(cases: int, seed: int) -> _SweepReport:
    """Draw ``cases`` random hierarchies and assert both backends agree on all of them.

    Args:
        cases (int): How many cases to draw.
        seed (int): The generator seed.

    Returns:
        _SweepReport: What was compared, for the caller's own vacuity assertions.

    Raises:
        AssertionError: On the first case where any field disagrees.
    """
    rng = np.random.default_rng(seed)
    cells = 0
    vertices = 0
    coordinates = 0
    deepest = 0
    operations: dict[str, int] = {}
    for index in range(cases):
        case = _draw_case(rng)
        for name, _ in case.operations:
            operations[name] = operations.get(name, 0) + 1
        with use_backend(Backend.PYTHON):
            reference = _build(case)
            points = _probe_points(reference)
        with use_backend(Backend.CPP):
            actual = _build(case)
        assert_object_parity(
            py=reference,
            cpp=actual,
            fields=_fields(case, points),
            context=f"HierarchicalGrid, sweep case {index} of seed {seed}",
        )
        cells += reference.num_cells
        exported = reference.export_cells()[0]
        vertices += int(exported.shape[0])
        coordinates += int(exported.size) + reference.num_cells * 2 * reference.ndim
        deepest = max(deepest, reference.max_level)
    return _SweepReport(cases, cells, vertices, coordinates, deepest, operations)


def _assert_the_sweep_was_not_vacuous(report: _SweepReport) -> None:
    """Fail if a sweep that passed had nothing in it to disagree about.

    A sweep whose draws all degenerated -- every operation refused, every hierarchy left
    at one level -- passes for a grid that was never refined, which is the one way a
    generative test can be green and worthless. The thresholds are proportions of the
    draw rather than absolutes, so they stay meaningful at either sweep size.

    Args:
        report (_SweepReport): What the sweep observed.

    Raises:
        AssertionError: If the draw degenerated.
    """
    assert report.deepest >= 3, (
        f"the deepest hierarchy drawn reached level {report.deepest}; below three levels "
        "no case can have a coarser active leaf hiding a finer one, which is where the "
        "descent, the facet walk and coarsen's refusals all live"
    )
    assert report.cells > 20 * report.cases, (
        f"{report.cells} cells over {report.cases} cases is under twenty per case, so "
        "most draws stayed near the unrefined root"
    )
    # `coarsen` is the rarest of the four by a wide margin and legitimately so: it
    # demands a box that is fully refined to exactly one level deeper, which a random box
    # usually is not, and the refusals are the unit tests' business rather than this
    # sweep's. So the floor is set low enough that the *acceptance rate* is not what this
    # guard is measuring -- it is here to catch a draw that degenerated to nothing.
    floor = max(5, report.cases // 50)
    for name in ("refine", "refine_cells", "coarsen", "coarsen_cells"):
        applied = report.operations.get(name, 0)
        assert applied > floor, (
            f"only {applied} of the {report.cases} cases applied a `{name}`; that "
            "operation is effectively untested by this sweep"
        )


def test_the_sweep_agrees(cpp_backend: None) -> None:
    """Both backends agree, quantity by quantity, over the shipped random sweep.

    The regression guard. :func:`test_the_wide_sweep_agrees` is the same comparison ten
    times wider and is not run in CI; this one is what a future edit trips over.
    """
    del cpp_backend
    report = _sweep(_SHIPPED_CASES, seed=20260903)
    _assert_the_sweep_was_not_vacuous(report)


@pytest.mark.slow
def test_the_wide_sweep_agrees(cpp_backend: None) -> None:
    """The same comparison over ten times as many cases, on a different seed.

    Kept in the repository rather than in a shell history: the ticket asks that the bound
    be verified over a sweep at least ten times the one that ships, and a verification
    that cannot be re-run is a claim rather than a check. A different seed, so that this
    is a wider *draw* and not merely a longer prefix of the shipped one.
    """
    del cpp_backend
    report = _sweep(_SHIPPED_CASES * _WIDE_FACTOR, seed=20260904)
    _assert_the_sweep_was_not_vacuous(report)


# ---------------------------------------------------------------------------
# The independent accuracy check: an exact rational corner
# ---------------------------------------------------------------------------


def _exact_corners(grid: HierarchicalGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every cell's corners computed in exact rational arithmetic.

    An oracle that is independent of *both* implementations rather than a re-run of
    either. A ``float64`` breakpoint is a rational number exactly, and a cell's corner is
    a rational function of two of them:

        lo[k] = b + s * (b' - b) / factor[k] ** level
        hi[k] = b + (s + 1) * (b' - b) / factor[k] ** level

    where ``b`` and ``b'`` bound the containing root cell and ``s`` is the cell's index
    within it. :class:`fractions.Fraction` evaluates that with no rounding at all, and the
    conversion back to ``float64`` at the end is one correctly rounded operation whose
    error the bound accounts for.

    Args:
        grid (HierarchicalGrid): The grid.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``(lo, hi, base)`` of shape
        ``(num_cells, ndim)``. ``base`` is the containing root cell's lower breakpoint,
        which the bound needs and which is exact.
    """
    ndim = grid.ndim
    breakpoints = [[Fraction(float(v)) for v in np.asarray(bp)] for bp in grid.root.breakpoints]
    lo = np.empty((grid.num_cells, ndim), dtype=np.float64)
    hi = np.empty((grid.num_cells, ndim), dtype=np.float64)
    base = np.empty((grid.num_cells, ndim), dtype=np.float64)
    for cid in range(grid.num_cells):
        level = grid.cell_level(cid)
        midx = grid.cell_multi_index(cid)
        for k in range(ndim):
            span = grid.factor[k] ** level
            root_cell, sub = divmod(midx[k], span)
            low = breakpoints[k][root_cell]
            width = breakpoints[k][root_cell + 1] - low
            lo[cid, k] = float(low + sub * width / span)
            hi[cid, k] = float(low + (sub + 1) * width / span)
            base[cid, k] = float(low)
    return lo, hi, base


def _corner_accuracy_bound(
    lo: np.ndarray, hi: np.ndarray, base: np.ndarray
) -> npt.NDArray[np.float64]:
    r"""Bound each computed corner's distance from its exact rational value.

    **Derivation.** Write ``b``, ``b'`` for the containing root cell's breakpoints,
    ``F = factor[k] ** level`` and ``s`` for the index within the root cell; ``s`` and
    ``F`` are integers below ``2**53`` on any grid this suite builds, so both are exact
    in ``float64`` and neither contributes a rounding. Both backends evaluate

        w = fl(b' - b)          one rounding: w = w_e (1 + d1)
        q = fl(w / F)           one rounding: q = (w_e / F)(1 + d1)(1 + d2)
        p = fl(s * q)           one rounding: p = o (1 + d1)(1 + d2)(1 + d3), o = s w_e / F
        lo = fl(b + p)          one rounding
        hi = fl(lo + q)         one rounding

    with every ``|di| <= u``. From ``|p - o| <= gamma_3 |o|`` and one more rounding on the
    sum, ``|lo - lo_e| <= u |lo| + gamma_4 |lo - b|``. For ``hi`` the errors of ``lo`` and
    of ``q`` both enter and a fifth rounding is added, giving
    ``|hi - hi_e| <= 2 u M + gamma_4 |lo - b| + gamma_2 |hi - b|`` with
    ``M = max(|lo|, |hi|)``. Since ``0 <= lo - b <= hi - b`` and
    ``gamma_4 + gamma_2 <= gamma_6``, one expression covers both corners:

        bound = 2 u M + gamma_6 max(|lo - b|, |hi - b|)

    which is the ``hi`` bound and dominates the ``lo`` one. It is deliberately **not** a
    relative bound: ``|x - b|`` is bounded by the root cell's width and not by ``|x|``, so
    a corner small against the width of the root cell holding it -- which the drawn
    breakpoints straddle zero to produce -- would break a relative form.
    ``design/backend_parity.md`` Rule 2 is the general statement, including its converse:
    a flat absolute bound would be vacuous on a corner near the origin.

    **Stated hypothesis: coordinates are normal, not subnormal.** The expression is a
    product of relative factors with magnitudes, so below roughly ``2e-308`` both terms
    underflow to exactly zero while the difference they bound is still a nonzero
    subnormal, and the expression stops implementing its own inequality even though the
    inequality still holds in the reals. Nothing here reaches that domain -- every
    breakpoint drawn is in ``[-1, 1]`` and every width is bounded below by the draw -- so
    the hypothesis holds by construction rather than by luck. It is carried rather than
    dropped because it is a property of the expression, not of this sweep.

    **Exercised at ``float64`` only.** The C++ hierarchy is bound at ``double`` alone, so
    ``float32`` storage has no numeric test of this quantity in either backend and the
    rounding counts above are checked at one format and carried to the other by the
    derivation alone.

    Args:
        lo (np.ndarray): Exact lower corners, ``(num_cells, ndim)``.
        hi (np.ndarray): Exact upper corners, same shape.
        base (np.ndarray): Containing root cells' lower breakpoints, same shape.

    Returns:
        npt.NDArray[np.float64]: The elementwise bound, same shape.
    """
    magnitude = np.maximum(np.abs(lo), np.abs(hi))
    offset = np.maximum(np.abs(lo - base), np.abs(hi - base))
    return np.asarray(2.0 * _U * magnitude + _gamma(6) * offset, dtype=np.float64)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cell_bounds_match_the_exact_rational_corner(backend: Backend) -> None:
    """Each backend's cell corners match an exact rational oracle within a derived bound.

    The **independent** check `design/backend_parity.md` asks for beside a parity claim,
    and it is independent in the way that matters: it never runs either implementation's
    arithmetic. A parity comparison of two backends that make the same mistake passes;
    this one does not.

    Asserted per backend, so the Python half runs in an installation with no extension.
    """
    _demand_the_extension_if_needed(backend)
    rng = np.random.default_rng(4711)
    worst = 0.0
    compared = 0
    for _ in range(40):
        case = _draw_case(rng)
        with use_backend(backend):
            grid = _build(case)
            computed_lo = np.empty((grid.num_cells, grid.ndim), dtype=np.float64)
            computed_hi = np.empty_like(computed_lo)
            for cid in range(grid.num_cells):
                computed_lo[cid], computed_hi[cid] = grid.cell_bounds(cid)
            exact_lo, exact_hi, base = _exact_corners(grid)
        bound = _corner_accuracy_bound(exact_lo, exact_hi, base)
        claim = derived_accuracy(
            bound=bound, why=_corner_accuracy_bound.__doc__ or "see _corner_accuracy_bound"
        )
        for name, computed, exact in (
            ("lo", computed_lo, exact_lo),
            ("hi", computed_hi, exact_hi),
        ):
            deviation = assert_accuracy(
                computed, exact, claim, context=f"HierarchicalGrid.cell_bounds {name}"
            )
            worst = max(worst, deviation.max_ratio_to_bound)
        compared += computed_lo.size * 2
    assert compared > 2000, (
        f"only {compared} corners were compared, so the draw degenerated and this bound "
        "was barely exercised"
    )
    # The bound has to be reached, or it asserts nothing: a bound never approached is
    # indistinguishable from one that is orders of magnitude too loose.
    assert worst > 0.0, (
        "every corner matched the exact rational value bit for bit, so this bound was "
        "never approached at all. That is possible on a dyadic draw and is a finding "
        "about the draw rather than good news: widen it until some corner rounds."
    )


# ---------------------------------------------------------------------------
# The pickle
# ---------------------------------------------------------------------------


def _tagged_grid() -> HierarchicalGrid:
    """A refined, tagged grid whose root is tagged too, for the pickle tests.

    Returns:
        HierarchicalGrid: The grid.
    """
    root = TensorProductGrid([[0.0, 0.5, 1.0], [0.0, 0.25, 1.0]])
    root.cell_tags.set("root_mark", np.array([0, 3], dtype=np.int64), np.array([9, 9]))
    grid = hierarchical_grid(root, (2, 3)).refine(0, (0, 0), (1, 1))
    grid = grid.refine_cells([0, 1])
    grid.cell_tags.set("mark", np.arange(3, dtype=np.int64), np.array([7, 7, 7]))
    grid.facet_tags.set(
        "wall", np.array([[0, 0], [1, 1]], dtype=np.int64), np.array([5, 6], dtype=np.int64)
    )
    return grid


def _pickle_fields() -> tuple[Field, ...]:
    """What a round-tripped grid has to reproduce.

    Returns:
        tuple[Field, ...]: The fields.
    """
    return (
        Field("num_cells", exact_parity(why=_VERDICT_WHY)),
        Field("max_level", exact_parity(why=_VERDICT_WHY)),
        Field("factor", exact_parity(why=_VERDICT_WHY)),
        Field("repr", exact_parity(why=_VERDICT_WHY), read=repr),
        Field("active_blocks", exact_parity(why=_VERDICT_WHY), read=_active_blocks),
        Field("cell_levels", exact_parity(why=_VERDICT_WHY), read=_cell_levels),
        Field(
            "cell_tags.names",
            exact_parity(why=_VERDICT_WHY),
            read=lambda g: tuple(g.cell_tags.names),
        ),
        Field(
            "cell_tags.mark",
            exact_parity(why=_VERDICT_WHY),
            read=lambda g: np.stack(g.cell_tags["mark"]),
        ),
        Field(
            "facet_tags.wall",
            exact_parity(why=_VERDICT_WHY),
            read=lambda g: np.asarray(g.facet_tags["wall"][1]),
        ),
        Field(
            "root.cell_tags.root_mark",
            exact_parity(why=_VERDICT_WHY),
            read=lambda g: np.stack(g.root.cell_tags["root_mark"]),
        ),
        Field("cell_corners", bitwise_parity(why=_CORNER_WHY), read=_corners),
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_reduce_round_trips_within_one_backend(backend: Backend) -> None:
    """A grid pickled and unpickled in one backend comes back with the same state.

    The active set travels as the per-level rectangles and is rebuilt through the
    normalizing path, so this also pins that renormalizing an already-normal level is the
    identity: were it not, every flat cell id would move and ``cell_levels`` would
    disagree.
    """
    _demand_the_extension_if_needed(backend)
    with use_backend(backend):
        grid = _tagged_grid()
        clone = pickle.loads(pickle.dumps(grid))
        assert clone is not grid
        assert clone.root is not grid.root
        assert_object_parity(
            py=grid, cpp=clone, fields=_pickle_fields(), context=f"pickle within {backend.name}"
        )


def test_reduce_round_trips_across_the_two_backends(cpp_backend: None) -> None:
    """All four writer/reader pairs agree, so the backend flag is not a data format.

    A pickle written under one backend has to load under the other, tags and all. Were it
    not so, choosing a backend would silently change what is on disk -- which is the
    failure a value comparison inside one backend cannot see at all.
    """
    del cpp_backend
    payloads = {}
    for writer in (Backend.PYTHON, Backend.CPP):
        with use_backend(writer):
            payloads[writer] = pickle.dumps(_tagged_grid())
    for writer, reader in itertools.product((Backend.PYTHON, Backend.CPP), repeat=2):
        with use_backend(Backend.PYTHON):
            reference = _tagged_grid()
        with use_backend(reader):
            restored = pickle.loads(payloads[writer])
        assert isinstance(restored, HierarchicalGrid)
        assert_object_parity(
            py=reference,
            cpp=restored,
            fields=_pickle_fields(),
            context=f"pickle written under {writer.name}, read under {reader.name}",
        )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_pickling_an_untagged_grid_keeps_the_memo_slots_lazy(backend: Backend) -> None:
    """An untagged grid round-trips without materializing a registry on either side.

    The laziness is part of the contract the oracle documents and the unit suite asserts,
    and it is the wrapper's rather than the implementation's: the C++ grid's registries
    are eager. So it can only survive a round trip if ``__reduce__`` reads the memo slots
    instead of the properties.
    """
    _demand_the_extension_if_needed(backend)
    with use_backend(backend):
        grid = hierarchical_grid(TensorProductGrid([[0.0, 1.0, 2.0]]), 2).refine(0, (0,), (1,))
        assert grid._cell_tags is None
        clone = pickle.loads(pickle.dumps(grid))
        assert clone._cell_tags is None
        assert clone._facet_tags is None
        assert clone._bvh is None
        assert clone.num_cells == grid.num_cells


# ---------------------------------------------------------------------------
# The two contracts the wrapper owes rather than forwards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cell_id_refuses_every_position_that_is_not_an_active_leaf(backend: Backend) -> None:
    """``cell_id`` answers ``None`` on all four ways of not naming an active leaf.

    A wrong length, a negative entry, an entry past the level's extent, and a level
    outside the hierarchy. The sweep only ever asks it about positions that *are* active
    -- it uses the round trip -- so each refusal has its own case here, and a mutation
    pass found the negative one missing before this test named it.

    The wrong-length case is the one the wrapper owes rather than forwards. The oracle
    folds a wrong length into its "not an active leaf" answer;
    `cpp/include/pantr/core/error.hpp` puts value and range checks in C++ and leaves shape
    coercion to the wrapper, so the C++ grid *raises* on it instead. That is a difference
    in the **type** of the answer rather than in a message, which is why the wrapper
    catches the case before forwarding.
    """
    _demand_the_extension_if_needed(backend)
    with use_backend(backend):
        grid = hierarchical_grid(TensorProductGrid([[0.0, 1.0], [0.0, 1.0]]), 2)
        refined = grid.refine(0, (0, 0), (1, 1))
        for level, midx in (
            (0, (0,)),  # too short
            (0, (0, 0, 0)),  # too long
            (0, ()),  # empty
            (0, (-1, 0)),  # negative on axis 0
            (0, (0, -1)),  # negative on axis 1
            (-1, (0, 0)),  # level below the hierarchy
            (5, (0, 0)),  # level above it
            (0, (7, 0)),  # past the level's extent
        ):
            assert grid.cell_id(level, midx) is None, f"({level}, {midx}) resolved"
            assert grid.is_active_leaf(level, midx) is False, f"({level}, {midx}) was a leaf"
        # A position that exists but has been refined away is not a leaf either, while
        # its children are -- without this pair the refusals above would pass on a
        # `cell_id` that refused everything.
        assert refined.cell_id(0, (0, 0)) is None
        assert refined.cell_id(1, (0, 0)) is not None
        assert grid.cell_id(0, (0, 0)) == 0


def test_the_handle_itself_raises_on_a_wrongly_sized_multi_index(cpp_backend: None) -> None:
    """The C++ handle raises where the wrapper answers ``None``.

    Pinned rather than assumed, because it is the whole reason the wrapper carries that
    branch: if the handle ever started returning ``None`` itself, the branch would become
    dead code that nothing would flag, and if the wrapper's branch were deleted the two
    backends would differ in the type of the answer with no test naming why.
    """
    del cpp_backend
    handle = _bindings().HierarchicalGrid(
        _bindings().TensorProductGrid([np.array([0.0, 1.0]), np.array([0.0, 1.0])]),
        np.array([2, 2], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="midx must have 2 entries"):
        handle.cell_id(0, np.array([0], dtype=np.int64))
    assert handle.cell_id(0, np.array([0, 0], dtype=np.int64)) == 0


@pytest.mark.parametrize("backend", _BACKENDS)
def test_the_operations_hand_their_result_the_receivers_root(backend: Backend) -> None:
    """``refine``, ``refine_cells``, ``coarsen`` and ``coarsen_cells`` share the root.

    All four keep the root they were given, and the wrapper carries the receiver's root
    *object* into the result. Without that the C++ path would hand back a fresh handle
    around an equal-but-distinct copy, on which a caller's root tags would be missing --
    a silent loss that no value comparison of the hierarchy would see.

    ``restrict`` is the deliberate exception and is checked here too: its sub-grid has a
    *windowed* root, so sharing would be wrong rather than merely unnecessary.
    """
    _demand_the_extension_if_needed(backend)
    with use_backend(backend):
        root = TensorProductGrid([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
        grid = hierarchical_grid(root, 2)
        assert grid.root is root
        grid = grid.refine(0, (0, 0), (2, 2))
        assert grid.root is root
        root.cell_tags.set("marked", np.array([0], dtype=np.int64), np.array([3]))

        for result in (
            grid.refine(1, (0, 0), (1, 1)),
            grid.refine_cells([0]),
            grid.coarsen(0, (0, 0), (2, 2)),
            grid.coarsen_cells(list(range(grid.num_cells))),
            grid._copy(),
        ):
            assert result.root is root
            assert result.root.cell_tags["marked"][1].tolist() == [3]

        sub = grid.restrict([0]).grid
        assert isinstance(sub, HierarchicalGrid)
        assert sub.root is not root
        assert sub.root.cells_per_axis != root.cells_per_axis
