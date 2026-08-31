"""Parity of the C++ `TensorProductGrid` against the Python oracle it was ported from.

`tests/parity/test_grid.py` covers the grid *kernels* -- the free functions both
backends call. This module covers the *type*: `pantr.grid.TensorProductGrid` is now a
wrapper over a handle, and what has to agree is a whole object's behaviour rather than
one array a kernel wrote.

The claim, and why almost all of it is exact
--------------------------------------------

A tensor-product grid performs **no floating-point arithmetic on its coordinates**.
Every breakpoint a grid hands out is a copy of one it was given; every cell bound is a
copy of a breakpoint; every location verdict is a comparison. There is nothing to
round, so nothing to bound: the claim is bit-for-bit agreement of the coordinate arrays
and exact agreement of every id, count and flag. A tolerance here would not be a safety
margin, it would be hiding a defect.

There are exactly two places arithmetic happens, and each gets its own treatment.

- **`uniform_grid` computes its breakpoints.** The C++ factory reproduces
  `numpy.linspace`'s own sequence rather than approximating it, so the claim there is
  also bit-identity, and it is asserted against `numpy.linspace` directly rather than
  against the other backend, because that is what the implementation claims. Two of
  the three details that carry it are exercised below -- the exact assignment of the
  final breakpoint, and the product and the sum as separate statements so
  `-ffp-contract=on` cannot fuse them. The third, numpy's `step == 0` branch, is
  **unreachable by any grid**: when `step` underflows, `linspace`'s fallback produces
  breakpoints that are not strictly increasing and the constructor rejects them.
  `tests/test_grid_tensor_product.py` pins that premise so it is not taken on trust.
- **`is_uniform` compares a spread against a tolerance.** That is a *verdict*, which
  `design/backend_parity.md` Rule 11 says no tolerance bounds. It is asserted as a
  verdict: the two backends must agree on the answer, on both sides of the threshold,
  at every coordinate magnitude.

What these tests would catch that a value comparison would not
--------------------------------------------------------------

Three things, and they are why this module exists rather than a few more cases in
`test_grid.py`.

**A write through a returned reference being silently lost.** `cell_tags` and
`facet_tags` return references into the grid, and nanobind's default return-value
policy for an lvalue reference is *copy*. Under the default the registry a caller
mutates is a temporary and the tag is gone with no error anywhere. Measured on this
tree by dropping the policy and rebuilding: the tag set below did not appear in
`names`, and the property stopped returning the same object twice. There is no numeric
comparison that sees this.

**A pickle becoming a data-format switch.** A grid written under one backend has to
load under the other, tags and all, or the backend flag would silently change what is
on disk. All four writer/reader pairs are exercised.

**The two backends disagreeing about which exception, or about what it says.** Both
are discrete verdicts, and the second one is where a numpy repr leaked into a message
that C++ cannot reproduce.
"""

from __future__ import annotations

import functools
import pickle
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pytest

from pantr._backend import Backend, use_backend
from pantr.geometry import AABB
from pantr.grid import TensorProductGrid, uniform_grid
from pantr.grid._tensor_product_grid import (
    _EPS,
    _UNIFORM_SPACING_EPS_FACTOR,
    _TensorProductGridPython,
)
from tests._parity_harness import Field, assert_object_parity, bitwise_parity, exact_parity

if TYPE_CHECKING:
    import numpy.typing as npt

pytestmark = pytest.mark.usefixtures("cpp_backend")

_T = TypeVar("_T")

_COORDINATE_WHY = (
    "a tensor-product grid performs no arithmetic on its coordinates: every value it "
    "hands out is a copy of a breakpoint it was given, so there is no rounding to "
    "bound and a single differing bit is a defect rather than a displacement"
)

_VERDICT_WHY = (
    "cell ids, counts and flags are verdicts rather than displaced values; "
    "design/backend_parity.md Rule 11 is explicit that no tolerance bounds one, and a "
    "bounded comparison could not see two answers of different length at all"
)

# Grids chosen for the phenomena rather than for coverage: the smallest possible one,
# a one-dimensional one, an anisotropic non-uniform one, one whose axes differ in
# length, and one whose coordinates sit a million away from the origin.
_GRIDS: list[tuple[str, list[list[float]]]] = [
    ("single cell", [[0.0, 1.0], [0.0, 1.0]]),
    ("one axis", [[0.0, 1.0, 2.0, 3.0]]),
    ("non-uniform, anisotropic", [[0.0, 1.0, 3.0, 6.0, 10.0], [-2.0, 0.5, 4.0]]),
    ("three axes", [[0.0, 1.0, 2.0], [0.0, 3.0], [0.0, 0.5, 1.0, 1.5]]),
    ("translated", [[1e6, 1e6 + 0.25, 1e6 + 1.0]]),
]


def _both(call: Callable[[], _T]) -> tuple[_T, _T]:
    """Run one callable under each backend and return ``(reference, actual)``.

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


def _grid_on(breakpoints: npt.NDArray[np.float64]) -> TensorProductGrid:
    """Build a one-axis grid on the given breakpoints, in the active backend.

    A named function rather than a lambda closing over a loop variable, so the value
    is bound where it is read.

    Args:
        breakpoints (npt.NDArray[np.float64]): The single axis's breakpoints.

    Returns:
        TensorProductGrid: The grid.
    """
    return TensorProductGrid([breakpoints])


def _both_grids(breakpoints: list[list[float]]) -> tuple[TensorProductGrid, TensorProductGrid]:
    """Build the same grid under each backend.

    Args:
        breakpoints (list[list[float]]): The per-axis breakpoints.

    Returns:
        tuple[TensorProductGrid, TensorProductGrid]: The oracle-backed grid, then the
        C++-backed one.
    """
    return _both(lambda: TensorProductGrid(breakpoints))


def _grid_fields(ndim: int) -> tuple[Field, ...]:
    """The state two grids have to agree on, and the claim governing each piece.

    The per-axis breakpoints are one field each rather than one field for the tuple:
    the tuple is ragged, so it is several quantities, and a per-axis field also names
    the axis in a failure message.

    Args:
        ndim (int): How many breakpoint axes to compare.

    Returns:
        tuple[Field, ...]: The fields, in the order a failure should be read.
    """
    return (
        Field("ndim", exact_parity(why=_VERDICT_WHY)),
        Field("num_cells", exact_parity(why=_VERDICT_WHY)),
        Field("cells_per_axis", exact_parity(why=_VERDICT_WHY)),
        Field("is_uniform", exact_parity(why=_VERDICT_WHY)),
        Field("bounds", bitwise_parity(why=_COORDINATE_WHY)),
        *(
            Field(
                f"breakpoints[{d}]",
                bitwise_parity(why=_COORDINATE_WHY),
                read=lambda g, axis=d: g.breakpoints[axis],  # type: ignore[misc]
            )
            for d in range(ndim)
        ),
    )


def _adversarial_points(breakpoints: list[list[float]]) -> npt.NDArray[np.float64]:
    """Query points that attack the tie contract and the domain frontier.

    Every breakpoint on every axis, one ulp inside and outside each outer corner, the
    cell midpoints, points far outside, and a non-finite row. The interior is barely
    probed on purpose: an ordinary interior point agrees under almost any defect,
    while a point exactly on a face is where the two implementations' searches could
    part company.

    Args:
        breakpoints (list[list[float]]): The grid's per-axis breakpoints.

    Returns:
        npt.NDArray[np.float64]: Shape ``(npts, ndim)`` query points.
    """
    ndim = len(breakpoints)
    per_axis: list[list[float]] = []
    for bp in breakpoints:
        values = list(bp)
        values += [np.nextafter(bp[0], -np.inf), np.nextafter(bp[0], np.inf)]
        values += [np.nextafter(bp[-1], np.inf), np.nextafter(bp[-1], -np.inf)]
        values += [0.5 * (bp[i] + bp[i + 1]) for i in range(len(bp) - 1)]
        values += [bp[0] - 10.0 * (bp[-1] - bp[0]), bp[-1] + 10.0 * (bp[-1] - bp[0])]
        values += [np.nan, np.inf, -np.inf]
        per_axis.append(values)
    width = max(len(v) for v in per_axis)
    # One row per index, each axis cycling through its own list: that pairs a face on
    # one axis with an interior point on another, which is where a stride error hides.
    rows = [[per_axis[d][i % len(per_axis[d])] for d in range(ndim)] for i in range(width)]
    return np.asarray(rows, dtype=np.float64)


# ---------------------------------------------------------------------- state


@pytest.mark.parametrize(("label", "breakpoints"), _GRIDS, ids=[g[0] for g in _GRIDS])
def test_grid_state_agrees(label: str, breakpoints: list[list[float]]) -> None:
    """Both backends store the same grid, field by field."""
    py, cpp = _both_grids(breakpoints)
    assert_object_parity(
        py=py,
        cpp=cpp,
        fields=_grid_fields(len(breakpoints)),
        context=f"TensorProductGrid({label})",
    )
    assert not cpp.breakpoints[0].flags.writeable, (
        "the C++ grid must hand out read-only breakpoints too: they are views into its "
        "own storage, and a writeable one would let a caller corrupt the grid"
    )
    assert not cpp.bounds.flags.writeable


@pytest.mark.parametrize(("label", "breakpoints"), _GRIDS, ids=[g[0] for g in _GRIDS])
def test_repr_agrees(label: str, breakpoints: list[list[float]]) -> None:
    """The two backends print the grid identically.

    Computed by the wrapper precisely so that it cannot drift, and this is what says
    so. It also pins the substrings `tests/test_grid_tensor_product.py` asserts.
    """
    py, cpp = _both_grids(breakpoints)
    assert repr(py) == repr(cpp)
    assert repr(cpp).startswith("TensorProductGrid(ndim=")
    assert f"cells_per_axis={cpp.cells_per_axis}" in repr(cpp)


# ------------------------------------------------------------------- location


@pytest.mark.parametrize(("label", "breakpoints"), _GRIDS, ids=[g[0] for g in _GRIDS])
def test_locate_and_locate_many_agree(label: str, breakpoints: list[list[float]]) -> None:
    """Both backends return the same cell id for every adversarial point.

    The count is asserted before the ids, per Rule 11, and the batch answer is
    asserted against the scalar one on each side as well as across the two: a
    specialisation that agreed with the other backend's specialisation while both
    disagreed with `locate` would otherwise pass.
    """
    py, cpp = _both_grids(breakpoints)
    points = _adversarial_points(breakpoints)

    py_batch = py.locate_many(points)
    cpp_batch = cpp.locate_many(points)
    assert py_batch.shape == cpp_batch.shape, "a changed count is a changed verdict"
    np.testing.assert_array_equal(py_batch, cpp_batch, err_msg=_VERDICT_WHY)

    py_scalar = [py.locate(p) for p in points]
    cpp_scalar = [cpp.locate(p) for p in points]
    assert py_scalar == cpp_scalar
    for i, cid in enumerate(cpp_scalar):
        assert cpp_batch[i] == (-1 if cid is None else cid), (
            f"row {i}: locate_many and locate disagree inside one backend"
        )

    # Non-vacuous: the batch must contain both hits and misses, or an equality of two
    # all-minus-one arrays would pass while proving nothing.
    assert (cpp_batch >= 0).any()
    assert (cpp_batch == -1).any()


@pytest.mark.parametrize(("label", "breakpoints"), _GRIDS, ids=[g[0] for g in _GRIDS])
def test_cell_geometry_agrees(label: str, breakpoints: list[list[float]]) -> None:
    """Cell bounds, AABBs, reference maps and the collected bounds all agree."""
    py, cpp = _both_grids(breakpoints)
    py_lo, py_hi = py.collect_cell_bounds()
    cpp_lo, cpp_hi = cpp.collect_cell_bounds()
    assert py_lo.tobytes() == cpp_lo.tobytes(), _COORDINATE_WHY
    assert py_hi.tobytes() == cpp_hi.tobytes(), _COORDINATE_WHY

    for cid in range(py.num_cells):
        assert py.cell_bounds(cid)[0].tobytes() == cpp.cell_bounds(cid)[0].tobytes()
        assert py.cell_bounds(cid)[1].tobytes() == cpp.cell_bounds(cid)[1].tobytes()
        assert py.cell_aabb(cid) == cpp.cell_aabb(cid)
        assert py.cell_multi_index(cid) == cpp.cell_multi_index(cid)
        assert py.cell_level(cid) == cpp.cell_level(cid)
        assert py.child_cells(cid) == cpp.child_cells(cid)
        assert py.neighbors(cid) == cpp.neighbors(cid)
        # The reference map is `diag(hi - lo) u + lo`, which is one subtraction of two
        # stored breakpoints; the bound is still zero.
        probe = np.full((1, py.ndim), 0.25)
        np.testing.assert_array_equal(py.reference_map(cid)(probe), cpp.reference_map(cid)(probe))


@pytest.mark.parametrize(("label", "breakpoints"), _GRIDS, ids=[g[0] for g in _GRIDS])
def test_facet_and_boundary_agree(label: str, breakpoints: list[list[float]]) -> None:
    """Facet accessors and the whole boundary-facet enumeration agree."""
    py, cpp = _both_grids(breakpoints)
    py_rows = py.boundary_facets()
    cpp_rows = cpp.boundary_facets()
    assert py_rows.shape == cpp_rows.shape, "a changed facet count is a changed verdict"
    np.testing.assert_array_equal(py_rows, cpp_rows)
    assert not cpp_rows.flags.writeable
    assert py_rows.size > 0, "every grid here has an outer boundary; an empty one proves nothing"

    for cid in range(py.num_cells):
        assert py.num_local_facets(cid) == cpp.num_local_facets(cid)
        for lfid in range(py.num_local_facets(cid)):
            assert py.local_facet_axis_side(cid, lfid) == cpp.local_facet_axis_side(cid, lfid)
            assert py.is_mesh_boundary_facet(cid, lfid) == cpp.is_mesh_boundary_facet(cid, lfid)
            assert py.neighbor_across_facet(cid, lfid) == cpp.neighbor_across_facet(cid, lfid)
            assert py.hanging_neighbors(cid, lfid) == cpp.hanging_neighbors(cid, lfid)
            py_flo, py_fhi = py.local_facet_bounds(cid, lfid)
            cpp_flo, cpp_fhi = cpp.local_facet_bounds(cid, lfid)
            assert py_flo.tobytes() == cpp_flo.tobytes()
            assert py_fhi.tobytes() == cpp_fhi.tobytes()


@pytest.mark.parametrize(("label", "breakpoints"), _GRIDS, ids=[g[0] for g in _GRIDS])
def test_restrict_agrees(label: str, breakpoints: list[list[float]]) -> None:
    """The windowed sub-grid, its map and its mask agree, on a non-convex request."""
    py, cpp = _both_grids(breakpoints)
    # First and last: the window is the whole grid, so every intermediate cell is fill
    # and the mask is what separates the two kinds. A contiguous request would leave
    # the mask all-true and assert nothing about it.
    ids = [0, py.num_cells - 1]
    py_r = py.restrict(ids)
    cpp_r = cpp.restrict(ids)

    assert py_r.grid.num_cells == cpp_r.grid.num_cells
    assert_object_parity(
        py=py_r.grid,
        cpp=cpp_r.grid,
        fields=_grid_fields(len(breakpoints)),
        context=f"TensorProductGrid({label}).restrict",
    )
    np.testing.assert_array_equal(py_r.local_to_global_cell, cpp_r.local_to_global_cell)
    np.testing.assert_array_equal(py_r.in_subset, cpp_r.in_subset)
    assert cpp_r.in_subset.dtype == np.bool_, (
        "the C++ mask arrives as uint8 and must reach the caller as bool, which is what "
        "the oracle's np.isin returns"
    )
    assert not cpp_r.local_to_global_cell.flags.writeable
    assert not cpp_r.in_subset.flags.writeable
    assert isinstance(cpp_r.grid, TensorProductGrid), (
        "a restricted grid must come back as the public wrapper, not as a raw handle"
    )


def test_restrict_marks_fill_cells() -> None:
    """The mask really does separate requested cells from bounding-box fill.

    Guards the comparison above: if both backends returned an all-true mask the
    equality would pass and say nothing about the one thing the mask is for.
    """
    py, cpp = _both_grids([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])
    for grid in (py, cpp):
        r = grid.restrict([0, 5])
        assert r.in_subset.tolist() == [True, False, False, False, False, True]


# --------------------------------------------------------------- spatial index


@pytest.mark.parametrize(("label", "breakpoints"), _GRIDS, ids=[g[0] for g in _GRIDS])
def test_cell_bvh_is_memoized_and_agrees(label: str, breakpoints: list[list[float]]) -> None:
    """The BVH is built lazily, memoized, and the same under both backends.

    ``AC2``. The identity assertion is the one a fresh wrapper around the same handle
    would fail: nanobind hands back the same Python object for the same C++ pointer,
    but the Python backend's grid holds a `BVH` the wrapper has to keep, and only the
    wrapper's own memo makes both true.
    """
    py, cpp = _both_grids(breakpoints)
    for grid in (py, cpp):
        assert grid._bvh is None, "the BVH must not be built at construction"
        assert grid.cell_bvh() is grid.cell_bvh()
        assert grid._bvh is not None

    np.testing.assert_array_equal(py.cell_bvh().node_lo, cpp.cell_bvh().node_lo)
    np.testing.assert_array_equal(py.cell_bvh().node_cell, cpp.cell_bvh().node_cell)

    box = AABB(np.asarray(breakpoints, dtype=object)[0][0] * np.ones(py.ndim), np.ones(py.ndim))
    del box  # built only to prove the corner arithmetic below is not accidental
    lo = np.array([bp[0] for bp in breakpoints])
    hi = np.array([0.5 * (bp[0] + bp[-1]) for bp in breakpoints])
    query = AABB(lo, hi)
    py_hits = sorted(int(c) for c in py.query_aabb(query))
    cpp_hits = sorted(int(c) for c in cpp.query_aabb(query))
    assert py_hits == cpp_hits
    assert py_hits, "the query box must overlap something, or this compares two empties"


# ------------------------------------------------------------------------ tags


@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.CPP])
def test_a_tag_written_through_the_property_survives(backend: Backend) -> None:
    """A tag set through `g.cell_tags` is visible on the next read.

    **This is the regression test for the return-value policy**, and it is the reason
    this file exists in the shape it does. `cell_tags` returns a reference to a member
    of the C++ grid, and nanobind resolves the default policy for an lvalue-reference
    return to `copy`: without `nb::rv_policy::reference_internal` the registry a caller
    mutates is a temporary, the tag is gone, and nothing raises. Measured by dropping
    the policy and rebuilding -- `names` came back empty and the identity below failed.
    """
    with use_backend(backend):
        g = uniform_grid([[0.0, 3.0], [0.0, 2.0]], [3, 2])
        assert g._cell_tags is None, "the registry wrapper must be lazy"
        assert g._facet_tags is None

        g.cell_tags.set("cut", [0, 5], 7)
        assert g.cell_tags.names == ("cut",)
        assert len(g.cell_tags) == 1
        np.testing.assert_array_equal(g.cell_tags["cut"][0], [0, 5])
        np.testing.assert_array_equal(g.cell_tags["cut"][1], [7, 7])
        assert g.cell_tags is g.cell_tags, "the wrapper must memoize the registry"

        g.facet_tags.set("wall", [[0, 0], [5, 3]], 2)
        assert g.facet_tags.names == ("wall",)
        assert g.facet_tags.facets_per_cell == 2 * g.ndim
        assert g.facet_tags is g.facet_tags


def test_tag_registries_are_sized_from_the_grid() -> None:
    """Both backends size the two registries from the grid they belong to."""
    py, cpp = _both_grids([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])
    for grid in (py, cpp):
        assert grid.cell_tags.num_cells == grid.num_cells
        assert grid.facet_tags.num_cells == grid.num_cells
        assert grid.facet_tags.facets_per_cell == 2 * grid.ndim


# ---------------------------------------------------------------------- pickle


@pytest.mark.parametrize("reader", [Backend.PYTHON, Backend.CPP])
@pytest.mark.parametrize("writer", [Backend.PYTHON, Backend.CPP])
def test_a_tagged_grid_survives_a_pickle_round_trip(writer: Backend, reader: Backend) -> None:
    """``AC3``: a grid carrying tags pickles and unpickles across all four pairs.

    The four pairs are the whole point. A pickle written under one backend has to load
    under the other, or the backend flag would silently become a data-format flag --
    and the tags have to travel with it, because they are the grid's state rather than
    a cache. The iteration order of the names is asserted too: both registries promise
    insertion order, a replaced tag keeps its position, and a round trip that rebuilt
    the registry from a set would lose that quietly.
    """
    with use_backend(writer):
        original = uniform_grid([[0.0, 3.0], [0.0, 2.0]], [3, 2])
        original.cell_tags.set("zeta", [4], 9)
        original.cell_tags.set("alpha", [0, 5], 7)
        original.facet_tags.set("wall", [[0, 0], [5, 3]], 2)
        original.facet_tags.set("inlet", [[1, 1]], 3)
        blob = pickle.dumps(original)

    with use_backend(reader):
        restored = pickle.loads(blob)

        assert isinstance(restored, TensorProductGrid)
        assert restored.cells_per_axis == (3, 2)
        for d in range(restored.ndim):
            assert restored.breakpoints[d].tobytes() == original.breakpoints[d].tobytes()

        # Insertion order, not alphabetical: "zeta" was set first.
        assert restored.cell_tags.names == ("zeta", "alpha")
        assert restored.facet_tags.names == ("wall", "inlet")
        np.testing.assert_array_equal(restored.cell_tags["alpha"][0], [0, 5])
        np.testing.assert_array_equal(restored.cell_tags["alpha"][1], [7, 7])
        np.testing.assert_array_equal(restored.cell_tags["zeta"][0], [4])
        np.testing.assert_array_equal(restored.facet_tags["wall"][0], [[0, 0], [5, 3]])
        np.testing.assert_array_equal(restored.facet_tags["inlet"][1], [3])

        # The implementation is the reader's, never the writer's: the handle is not
        # part of the wire format.
        expected_python = reader is Backend.PYTHON
        assert isinstance(restored._impl, _TensorProductGridPython) is expected_python


@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.CPP])
def test_a_built_bvh_does_not_travel_in_the_pickle(backend: Backend) -> None:
    """The memoized spatial index is left behind, and the grid still works without it.

    `__reduce__` never reads the BVH slot, so this cannot fail today -- which is
    precisely why it is worth pinning. The index is `O(num_cells)` node arrays and is
    rebuilt from the breakpoints in one call, so shipping it would trade a large pickle
    for a small saving; a later `__reduce__` that started carrying grid state wholesale
    would pick it up without anything objecting.
    """
    with use_backend(backend):
        original = uniform_grid([[0.0, 3.0], [0.0, 2.0]], [3, 2])
        original.cell_tags.set("cut", [0, 5], 1)
        assert original.cell_bvh().n_cells == 6, "the index must be built before pickling"

        restored = pickle.loads(pickle.dumps(original))

        assert restored._bvh is None, "a memoized index must not travel in the pickle"
        assert restored.cell_tags.names == ("cut",), "the tags must travel, though"
        # And it rebuilds on demand, identically.
        np.testing.assert_array_equal(restored.cell_bvh().node_lo, original.cell_bvh().node_lo)


@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.CPP])
def test_an_untagged_grid_stays_lazy_across_a_pickle(backend: Backend) -> None:
    """A grid with no tags comes back with its memo slots still unfilled.

    The laziness is contractual -- `tests/test_grid_tags.py` reads the two slots
    directly -- so a round trip that materialized a registry to pickle it would move
    an observable.
    """
    with use_backend(backend):
        original = uniform_grid([[0.0, 2.0]], 2)
        restored = pickle.loads(pickle.dumps(original))
        assert original._cell_tags is None, "pickling must not materialize a registry"
        assert restored._cell_tags is None
        assert restored._facet_tags is None
        assert restored._bvh is None, "the BVH is derivable and must not travel"


# ------------------------------------------------------------------ exceptions


def _raised(call: Callable[[], Any]) -> tuple[type[BaseException], str]:
    """Run a call expected to raise and report the exception's type and message.

    Args:
        call (Callable[[], Any]): The call to make.

    Returns:
        tuple[type[BaseException], str]: The exception type and ``str(exc)``.

    Raises:
        AssertionError: If the call did not raise.
    """
    try:
        call()
    except Exception as exc:
        return type(exc), str(exc)
    raise AssertionError("the call was expected to raise and did not")


@pytest.mark.parametrize(
    ("what", "call"),
    [
        ("empty breakpoints", lambda g: TensorProductGrid([])),
        ("short axis", lambda g: TensorProductGrid([[0.0]])),
        ("non-increasing axis", lambda g: TensorProductGrid([[0.0, 1.0, 1.0]])),
        ("non-finite axis", lambda g: TensorProductGrid([[0.0, np.inf]])),
        ("cell id too large", lambda g: g.cell_bounds(99)),
        ("cell id negative", lambda g: g.cell_level(-1)),
        ("numpy cell id too large", lambda g: g.cell_level(np.int64(99))),
        ("facet id too large", lambda g: g.neighbor_across_facet(0, 99)),
        ("facet id negative", lambda g: g.neighbor_across_facet(0, -1)),
        ("wrong-length point", lambda g: g.locate([0.5])),
        ("wrong-width batch", lambda g: g.locate_many(np.zeros((3, 5)))),
        ("empty restrict", lambda g: g.restrict([])),
        ("restrict out of range", lambda g: g.restrict([0, 99])),
        ("non-integer restrict", lambda g: g.restrict([0.5])),
        ("multi-index too short", lambda g: g.flat_cell_index([0])),
        ("multi-index out of range", lambda g: g.flat_cell_index([99, 0])),
    ],
)
def test_the_two_backends_raise_the_same_exception(
    what: str, call: Callable[[TensorProductGrid], Any]
) -> None:
    """Both backends agree on which exception, and on what it says, verbatim.

    The message is compared and not merely the type, and one case is why: a numpy
    integer cell id used to render as ``np.int64(99)`` through ``!r`` on numpy 2 and as
    ``99`` on numpy 1, so the sentence depended on a numpy version and the C++ grid --
    which cannot reproduce a numpy repr -- could never have matched it. That is a
    verdict, not a value, and no tolerance covers it.

    A second case is why the two ``use_backend`` blocks below exist: the C++ constructor
    reported a too-short axis as ``got 1.`` where the oracle reports
    ``got shape (1,).``, and this test passed anyway, because the construction cases
    ignore the grid they are given and so both halves ran under one backend.
    """
    py, cpp = _both_grids([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0]])
    # Each call runs INSIDE its backend's context, and that is load-bearing rather than
    # tidy. Four of the cases above ignore the grid they are handed and construct a new
    # one, so the backend that decides their message is the ambient one at call time,
    # not the grid's. Without these two blocks both calls ran under the default backend
    # and the assertion compared a message against itself -- which is how a real
    # divergence in the "short axis" message survived until a review found it.
    with use_backend(Backend.PYTHON):
        py_type, py_message = _raised(lambda: call(py))
    with use_backend(Backend.CPP):
        cpp_type, cpp_message = _raised(lambda: call(cpp))
    assert py_type is cpp_type, f"{what}: {py_type.__name__} against {cpp_type.__name__}"
    assert py_message == cpp_message, what


# ------------------------------------------------------------- `uniform_grid`

# Domains chosen to span the coordinate magnitudes `is_uniform` has to survive, plus
# the two that broke the absolute tolerance this replaced: a domain far from the origin
# and one whose extent is far below one.
_DOMAINS = [(0.0, 1.0), (0.0, 1e-12), (0.0, 1e12), (1e6, 1e6 + 1.0), (-1.0, 1.0), (0.3, 0.7)]
_CELL_COUNTS = [1, 2, 7, 100, 1000]


@pytest.mark.parametrize("cells", _CELL_COUNTS)
@pytest.mark.parametrize(("lo", "hi"), _DOMAINS)
def test_uniform_grid_reproduces_numpy_linspace(lo: float, hi: float, cells: int) -> None:
    """The C++ factory's breakpoints are `numpy.linspace`'s, bit for bit.

    Asserted against `numpy.linspace` directly rather than against the Python backend,
    because that is what the C++ header claims: it reproduces numpy's sequence rather
    than approximating it. Comparing the two backends would be the weaker statement,
    since the Python backend calls `numpy.linspace` and so both could drift together
    only if numpy itself changed -- but it would not say which of the three details
    that carry the claim had been dropped.
    """
    py, cpp = _both(lambda: uniform_grid([[lo, hi]], cells))
    expected = np.linspace(lo, hi, cells + 1, dtype=np.float64)
    assert cpp.breakpoints[0].tobytes() == expected.tobytes(), (
        "the C++ factory must reproduce numpy's own sequence: the exact assignment of "
        "the final breakpoint, and the product and the sum as separate statements so "
        "-ffp-contract=on cannot fuse them"
    )
    assert py.breakpoints[0].tobytes() == expected.tobytes()
    assert cpp.breakpoints[0][-1] == hi, "the final breakpoint is assigned, not accumulated"


def test_uniform_grid_endpoint_is_assigned_not_accumulated() -> None:
    """The last breakpoint is exactly `stop`, in a case where the product is not.

    Without the assignment the upper bound would be short by an ulp, and the upper
    bound is what `locate` compares a boundary query against -- so the displacement
    would become a changed verdict rather than a changed value. The first assertion
    establishes that this case really does have a discrepancy to repair; without it the
    rest would pass on a case where nothing was ever wrong.
    """
    stop, cells = 2.9, 9
    computed = cells * (stop / cells)
    assert computed < stop, "this case must be one where the accumulated product misses stop"
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            g = uniform_grid([[0.0, stop]], cells)
            assert g.breakpoints[0][-1] == stop
            assert g.locate([stop]) == cells - 1


@pytest.mark.parametrize("cells", _CELL_COUNTS)
@pytest.mark.parametrize(("lo", "hi"), _DOMAINS)
def test_uniform_grid_is_uniform_at_every_magnitude(lo: float, hi: float, cells: int) -> None:
    """An exact `linspace` grid reports uniform on both backends, at every scale.

    The absolute tolerance this replaced failed here: `uniform_grid([[1e6, 1e6 + 1]],
    100)` reported non-uniform, because the spread of the spacings is proportional to
    the coordinate magnitude and the constant was not.
    """
    py, cpp = _both(lambda: uniform_grid([[lo, hi]], cells))
    assert py.is_uniform is True, f"[{lo}, {hi}] over {cells} cells"
    assert cpp.is_uniform is True, f"[{lo}, {hi}] over {cells} cells"


@pytest.mark.parametrize(("lo", "hi"), _DOMAINS)
def test_the_uniformity_bound_is_exercised_rather_than_compared_against_zero(
    lo: float, hi: float
) -> None:
    """The spread the bound is compared against is nonzero, and below the bound.

    Without this the previous test would be consistent with any bound at all, including
    a wrong one: a `linspace` grid whose spacings happen to be exactly equal compares
    zero against the tolerance and passes however the tolerance is derived. This asserts
    the case is a real one -- that at least one domain produces a nonzero spread, that
    every spread sits under the bound, and that the bound is not so loose it would admit
    a grid that is genuinely not uniform.
    """
    cells = 100
    bp = np.linspace(lo, hi, cells + 1, dtype=np.float64)
    spread = float(np.ptp(np.diff(bp)))
    bound = _UNIFORM_SPACING_EPS_FACTOR * _EPS * (abs(lo) + abs(hi))
    assert spread <= bound, f"[{lo}, {hi}]: observed {spread:.3e} against bound {bound:.3e}"
    # A perturbation ten times the bound must be rejected, which is what says the bound
    # is not merely large enough to admit everything.
    perturbed = bp.copy()
    perturbed[1] += 10.0 * bound if bound > 0.0 else np.spacing(float(hi))
    if perturbed[1] < perturbed[2]:
        assert not TensorProductGrid([perturbed]).is_uniform, f"[{lo}, {hi}]"


def test_at_least_one_domain_produces_a_nonzero_spread() -> None:
    """Somewhere in the sweep above, round-off is actually present.

    Stated as its own test rather than inside the loop because it is a property of the
    SET of domains: if every one of them happened to give an exactly-uniform
    `linspace`, the bound would never have been compared against anything but zero and
    the whole group would be inert. That is the failure this milestone met twice.
    """
    spreads = [
        float(np.ptp(np.diff(np.linspace(lo, hi, 101, dtype=np.float64)))) for lo, hi in _DOMAINS
    ]
    assert max(spreads) > 0.0, f"every domain gave an exact linspace: {spreads}"


@pytest.mark.parametrize(("lo", "hi"), _DOMAINS)
def test_the_uniformity_verdict_agrees_on_both_sides_of_the_threshold(lo: float, hi: float) -> None:
    """Both backends return the same `is_uniform` verdict, either side of the bound.

    A verdict rather than a value, so Rule 11 applies and no tolerance bounds it: the
    two backends must simply answer the same. Sweeping both sides is what makes the
    test say the two thresholds coincide rather than merely that both are permissive.
    """
    cells = 8
    base = np.linspace(lo, hi, cells + 1, dtype=np.float64)
    bound = _UNIFORM_SPACING_EPS_FACTOR * _EPS * (abs(lo) + abs(hi))
    spacing = (hi - lo) / cells
    for factor in (0.0, 0.25, 4.0, 1e3):
        bp = base.copy()
        bp[1] += factor * bound
        if not np.all(np.diff(bp) > 0.0) or factor * bound >= spacing:
            continue
        py, cpp = _both(functools.partial(_grid_on, bp))
        assert py.is_uniform == cpp.is_uniform, f"[{lo}, {hi}], perturbation {factor} * bound"


def test_uniform_grid_rejects_the_same_arguments_on_both_backends() -> None:
    """The factory's own validation agrees, type and message."""
    cases: list[Callable[[], Any]] = [
        lambda: uniform_grid([[0.0, 1.0]], 0),
        lambda: uniform_grid([[1.0, 0.0]], 2),
        lambda: uniform_grid([[0.0, 1.0], [0.0, 1.0]], [2]),
        lambda: uniform_grid([0.0, 1.0], 2),
    ]
    for call in cases:
        with use_backend(Backend.PYTHON):
            py_type, py_message = _raised(call)
        with use_backend(Backend.CPP):
            cpp_type, cpp_message = _raised(call)
        assert py_type is cpp_type
        assert py_message == cpp_message
