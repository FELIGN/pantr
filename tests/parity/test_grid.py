"""Parity between the Numba and C++ implementations of the `pantr.grid` kernels.

The claim is an **equality**, for all nine kernels, and it is the strongest one in
the tree so far. Two of the three paths can argue it structurally rather than
measure it:

- **Tensor-product point location performs no floating-point arithmetic at all.**
  Every binary operation in `_locate_core.py` is on an integer index and every use
  of a coordinate is a comparison, established by walking the oracle's syntax tree
  rather than by reading it. There is no rounding to bound, no accumulation order
  to preserve and no site a compiler could contract.
- **The BVH's every discrete verdict rests on a bit-identical quantity.** The node
  AABB is a running min/max over corners that are copied rather than computed; the
  longest-axis tie compares one subtraction with a strict ``>``; and the centroid is
  ``0.5 * (lo + hi)``, where the addition rounds once and the multiplication by
  ``0.5`` is exact *and follows the addition*, so there is no product for a compiler
  to fuse into it. The median split is a stable sort, and stability determines the
  output permutation uniquely, so `std::stable_sort` reproduces
  ``np.argsort(kind="mergesort")`` by argument rather than by coincidence.

The third path cannot, and the tests below say so where it applies. The
hierarchical descent rewrites ``lo[k]`` each level from ``lo[k] + j * size_k``, a
multiply feeding an add, and that value decides the next level's truncation -- a
**discrete verdict**, which ``design/backend_parity.md`` Rule 11 is explicit that no
tolerance bounds. So the port keeps the multiply and the add as separate statements,
which is what makes the equality hold: `-ffp-contract=on` confines fusion to within
a single expression, so a named product cannot be contracted on any target. An
earlier version of this file argued instead from a sweep of 300000 descents, which
was worthless -- it ran on a build with no `-march`, where the baseline ISA has no
fused multiply-add at all, so it compared two unfused implementations.

So the operational lesson of Rule 11 applies here in full: **every cell id is
asserted on its own, before and separately from any comparison of coordinates.** A
changed id is a changed verdict, not a displaced value, and a bounded comparison
cannot see two answers of different length at all.

The parity harness is used for the coordinate arrays and **not** for the cell ids,
and the split is not arbitrary. ``tests/_parity_harness`` compares floats bit for bit
through a bit view and has no integer path -- ``FloatArray`` is its whole domain --
because until now no ported kernel returned a *verdict* as its primary output.
:mod:`pantr.grid` is the first that does, so the ids are asserted with a plain array
equality carrying the same reasoning in its message. For an integer, "bitwise" and
"equal" are the same assertion and the machinery buys nothing.

There is no dtype axis. The oracle is ``float64``-only -- none of the three kernel
modules mentions ``float32`` -- so the C++ registers no ``float`` overload, which
Rule 8 requires: a parity claim is only defined where the comparison can say
something.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.geometry import AABB
from pantr.grid import BVH, TensorProductGrid, uniform_grid
from pantr.grid._hierarchical_grid import _HierarchicalGridPython
from tests._parity_harness import (
    Field,
    assert_object_parity,
    assert_parity,
    bitwise_parity,
    exact_parity,
)

_LOCATE_WHY = (
    "the kernel performs no floating-point arithmetic: every binary operation is on "
    "an integer index and every use of a coordinate is a comparison, so there is "
    "nothing for either backend to round differently"
)

_BVH_WHY = (
    "every discrete verdict rests on a bit-identical quantity -- the node box is a "
    "running min/max over copied corners, the longest-axis tie is one subtraction "
    "compared with a strict >, and the centroid's multiply by 0.5 follows its add so "
    "no contraction site exists -- and the median split's stable sort has a uniquely "
    "determined output permutation"
)

_HIER_WHY = (
    "the descent multiplies then adds, and the two statements are kept SEPARATE so "
    "that no target can contract them into a fused multiply-add: -ffp-contract=on "
    "confines fusion to within one expression, and the oracle never fuses at all "
    "(numba defaults to fastmath=False). That makes the equality a property of how "
    "the statement is written rather than of the build. Inline the product again and "
    "this fails at root cell [1.0, 2.0], factor 10, x = 1.71 -- but only on a build "
    "that has a fused multiply-add, so the disassembly is the other half of the "
    "evidence. See cpp/include/pantr/grid/hierarchical.hpp"
)


_T = TypeVar("_T")


def _both(call: Callable[[], _T]) -> tuple[_T, _T]:
    """Run one callable under each backend and return ``(reference, actual)``."""
    with use_backend(Backend.PYTHON):
        reference = call()
    with use_backend(Backend.CPP):
        actual = call()
    return reference, actual


def _assert_the_two_grids_differ() -> None:
    """Fail if a tensor-product grid holds the same implementation under both backends.

    The trip-wire for the vacuity described in `test_tensor_product_locate_is_bitwise`.
    `TensorProductGrid` chooses its implementation when it is CONSTRUCTED, so a grid
    built outside `_both` is compared against itself and the comparison says nothing.
    Asserting the two classes differ is cheap and catches the mistake at the site that
    made it, rather than leaving a green test that checks nothing.

    Raises:
        AssertionError: If the two backends hand back the same implementation class.
    """
    with use_backend(Backend.PYTHON):
        py_impl = type(TensorProductGrid([np.array([0.0, 1.0])])._impl)
    with use_backend(Backend.CPP):
        cpp_impl = type(TensorProductGrid([np.array([0.0, 1.0])])._impl)
    assert py_impl is not cpp_impl, (
        f"both backends built a {py_impl.__name__}: the grid above was compared "
        f"against itself, so this test asserts nothing about the port"
    )


def _adversarial_points_1d() -> npt.NDArray[np.float64]:
    """Points that attack the tie contract and the domain frontier, not the interior.

    Every interior breakpoint, both outer corners, one ulp inside and outside each
    corner, and a handful of ordinary interior points so the sweep is not all
    frontier.
    """
    breaks = [0.0, 1.0, 2.0, 3.0]
    pts = list(breaks)
    pts += [np.nextafter(0.0, -1.0), np.nextafter(3.0, 4.0)]
    pts += [np.nextafter(0.0, 1.0), np.nextafter(3.0, 2.0)]
    pts += [np.nextafter(1.0, 0.0), np.nextafter(1.0, 2.0)]
    pts += [0.5, 1.5, 2.5, -10.0, 10.0]
    return np.asarray(pts, dtype=np.float64).reshape(-1, 1)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_tensor_product_locate_is_bitwise(cpp_backend: None, ndim: int) -> None:
    """Both backends assign every point to the same cell of a tensor-product grid.

    The ids are integers, so "bitwise" and "equal" coincide and the assertion is
    exact by nature rather than by tolerance.

    **The grid is built inside the closure, and that is load-bearing.** Since the
    tensor-product grid became a wrapper over a handle, the backend is chosen when the
    grid is CONSTRUCTED rather than when a kernel is fetched. A grid built outside
    ``_both`` therefore holds one implementation and ``_both`` runs it twice, so the
    comparison passes for any port at all. That is what this test used to do; the
    assertion below is what stops it happening again.
    """
    del cpp_backend
    rng = np.random.default_rng(20260825)
    interior = rng.uniform(-0.5, 3.5, size=(200, ndim))
    corners = np.array(np.meshgrid(*([[0.0, 1.0, 2.0, 3.0]] * ndim))).reshape(ndim, -1).T
    points = np.vstack([interior, corners])

    def locate() -> npt.NDArray[np.int64]:
        return uniform_grid(np.tile([0.0, 3.0], (ndim, 1)), 3).locate_many(points)

    reference, actual = _both(locate)
    _assert_the_two_grids_differ()

    np.testing.assert_array_equal(
        actual,
        reference,
        err_msg=f"TensorProductGrid.locate_many disagreed at ndim {ndim}. {_LOCATE_WHY}",
    )


def test_tensor_product_locate_holds_at_the_tie_contract(cpp_backend: None) -> None:
    """The tie contract survives the port, on the frontier rather than the interior.

    Called out separately from the sweep above because the tie -- an interior
    breakpoint belongs to the LOWER cell -- is a discrete verdict, and a sweep of
    random interior points would never reach one.
    """
    del cpp_backend
    points = _adversarial_points_1d()

    def locate() -> npt.NDArray[np.int64]:
        # Constructed inside the closure; see `test_tensor_product_locate_is_bitwise`.
        return uniform_grid(np.tile([0.0, 3.0], (1, 1)), 3).locate_many(points)

    reference, actual = _both(locate)

    np.testing.assert_array_equal(
        actual,
        reference,
        err_msg="a cell id changed on the tie contract, which is a changed verdict "
        "and not a displaced value",
    )
    assert reference[0] == 0, "the lower corner belongs to the first cell"
    assert reference[3] == 2, "the upper corner belongs to the last cell"
    assert reference[1] == 0, "an interior breakpoint belongs to the lower cell"


def _random_boxes(n: int, ndim: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A deterministic spread of overlapping axis-aligned boxes."""
    rng = np.random.default_rng(4242 + n + ndim)
    lo = rng.uniform(-2.0, 2.0, size=(n, ndim))
    hi = lo + rng.uniform(0.05, 0.9, size=(n, ndim))
    return lo, hi


_BVH_EXACT_WHY = (
    "the node indices, the leaf cell ids and the three counts are the tree's SHAPE. "
    "design/backend_parity.md Rule 11: a differing verdict is not a displaced value, "
    "so there is no bound to derive and nothing but a reason to state -- and a "
    "comparison of elements cannot see two answers of different length at all"
)


def _bvh_fields() -> tuple[Field, ...]:
    """The state two BVHs have to agree on, and the claim governing each piece.

    Eight fields rather than five, because a hierarchy is its arrays *and* its
    counts, and a port that got `ndim` wrong on an empty tree would leave every
    array empty and equal.

    **The two float arrays carry ``bitwise_parity`` and the six integer pieces carry
    ``exact_parity``, and that is the same claim rather than a weaker one.** The
    harness is float-only under a ``ParityClaim`` by construction -- it compares
    through a bit view and computes a difference in ``float64`` -- and its BITWISE
    failure message tells the reader to check ``__fp_contract__``, which decides
    nothing about a node index. ``exact_parity`` is elementwise ``!=`` with a reason
    attached: no tolerance is introduced anywhere here, and none may be.

    Returns:
        tuple[Field, ...]: The fields, in the order a failure should be read.
    """
    coordinates = ("node_lo", "node_hi")
    shape = ("node_left", "node_right", "node_cell", "ndim", "n_cells", "n_nodes")
    return (
        *(Field(name, bitwise_parity(why=_BVH_WHY)) for name in coordinates),
        *(Field(name, exact_parity(why=_BVH_EXACT_WHY)) for name in shape),
    )


@pytest.mark.parametrize("n_cells", [1, 2, 3, 9, 64])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_bvh_node_arrays_are_bitwise(cpp_backend: None, n_cells: int, ndim: int) -> None:
    """Both backends build the identical tree, node array by node array.

    The strongest check in this file, and deliberately not a query comparison: it
    compares the whole structure, so the stable sort's permutation and every
    tie-break agree rather than merely producing the same answer to one question.

    That matters beyond parity. ``design/bvh.md`` records that the five node arrays
    are **public API** -- a downstream consumer walks the tree itself with its own
    predicate -- so a port that got the layout subtly right for queries and wrong for
    traversal would break that consumer without failing a query test.
    """
    del cpp_backend
    lo, hi = _random_boxes(n_cells, ndim)

    reference, actual = _both(lambda: BVH.from_cell_bounds(lo, hi))

    # `_both` returns (reference, actual); `assert_object_parity` takes (py, cpp).
    # The two orders differ across this suite's modules, so the keywords are used
    # rather than the position -- the harness's own docstring says why.
    assert_object_parity(
        py=reference,
        cpp=actual,
        fields=_bvh_fields(),
        context=f"BVH.from_cell_bounds, {n_cells} cells in {ndim}-D",
    )


@pytest.mark.parametrize("n_cells", [1, 5, 40])
def test_bvh_query_returns_the_same_cells(cpp_backend: None, n_cells: int) -> None:
    """Both backends return the same overlapping cells, count asserted first.

    The count is asserted on its own before the ids, per Rule 11: two results of
    different length are a changed verdict, and comparing them elementwise cannot
    see the difference at all.
    """
    del cpp_backend
    lo, hi = _random_boxes(n_cells, 2)
    boxes = [
        AABB(lo.min(axis=0) - 0.1, lo.min(axis=0) + 0.5 * (hi.max(axis=0) - lo.min(axis=0))),
        AABB(lo.min(axis=0) - 5.0, hi.max(axis=0) + 5.0),  # everything
        AABB(hi.max(axis=0) + 1.0, hi.max(axis=0) + 2.0),  # nothing
        AABB(lo[0], lo[0]),  # a degenerate box on a corner
    ]

    def query(b: AABB) -> npt.NDArray[np.int64]:
        return BVH.from_cell_bounds(lo, hi).query_aabb(b)

    matched = 0
    for i, box in enumerate(boxes):
        reference, actual = _both(partial(query, box))
        assert len(actual) == len(reference), (
            f"box {i} matched {len(actual)} cells against the oracle's {len(reference)}; "
            "a changed count is a changed verdict, not a displaced value"
        )
        np.testing.assert_array_equal(actual, reference, err_msg=f"box {i} cell ids")
        matched += len(reference)
    assert matched > 0, "no box matched anything, so this test asserted nothing"


def test_bvh_longest_axis_tie_keeps_the_lower_axis(cpp_backend: None) -> None:
    """Pin the split-axis tie-break, which no other test in the repo can see.

    ``_random_boxes`` draws continuous random floats, so two axis extents are never
    exactly equal and the tie-break is never reached. Confirmed by fault injection:
    replacing the longest-axis comparison in ``bvh.hpp`` with ``>=`` -- which flips the
    tie from the lower axis to the higher -- leaves every other test in this file and
    all three C++ suites passing, so the direction is otherwise unpinned in both
    backends at once.

    Four unit cells at the corners of a square give the root an exact ``3.0 == 3.0``
    extent tie, and the two candidate axes order the centroids differently, so the
    choice is observable in ``node_cell``: axis 0 yields the leaf order ``[0, 2, 1,
    3]`` and axis 1 would yield ``[0, 1, 2, 3]``. Asserted as a direction and not only
    as agreement, because two backends that both flipped would still agree.
    """
    del cpp_backend
    cell_lo = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    cell_hi = cell_lo + 1.0

    reference, actual = _both(lambda: BVH.from_cell_bounds(cell_lo, cell_hi))

    extent = reference.node_hi[0] - reference.node_lo[0]
    assert extent[0] == extent[1], (
        f"the fixture stopped being a tie: root extents are {extent.tolist()}, so this "
        "test no longer reaches the tie-break it exists to pin"
    )
    assert [int(c) for c in reference.node_cell if c >= 0] == [0, 2, 1, 3], (
        "the longest-axis tie no longer keeps the LOWER axis; splitting on axis 1 "
        "instead of axis 0 gives the leaf order [0, 1, 2, 3]"
    )
    np.testing.assert_array_equal(
        actual.node_cell,
        reference.node_cell,
        err_msg=f"the backends split the tied axis differently. {_BVH_WHY}",
    )
    for name in ("node_left", "node_right"):
        np.testing.assert_array_equal(
            getattr(actual, name), getattr(reference, name), err_msg=f"{name} on a tied axis"
        )
    for name in ("node_lo", "node_hi"):
        assert_parity(
            getattr(actual, name),
            getattr(reference, name),
            bitwise_parity(why=_BVH_WHY),
            context=f"BVH.{name} on a tied split axis",
        )


def _two_level_grid() -> _HierarchicalGridPython:
    """A hierarchical grid with a genuinely refined region.

    A single ``_HierarchicalGridPython`` alone returns a **single-level** grid, and on
    one of those the descent below never runs: the root lookup finds an active cell at
    level 0 and returns, so the contraction site these tests exist to exercise is never
    reached. So this refines a corner, and the callers assert that it worked rather than
    trusting that it did.

    Built as the **oracle** rather than through :func:`pantr.grid.hierarchical_grid`, and
    that is load-bearing here in a way it is not in an ordinary test. This file compares
    *kernels*: the oracle reaches for one per call, through
    :func:`~pantr.grid._grid_backend.hier_locate_points_kernel` and its siblings, so
    wrapping the same grid in ``use_backend`` twice runs the two kernels. The public
    class picks its implementation once, when it is constructed, so a public grid built
    here would answer from whichever backend was active at construction and ``_both``
    would compare a backend against itself -- green, and testing nothing.
    """
    root = uniform_grid(np.tile([0.0, 4.0], (2, 1)), 4)
    grid = _HierarchicalGridPython(root, 2)
    grid = grid.refine_cells([0, 1, 4])
    return grid


def _assert_more_than_one_level(grid: _HierarchicalGridPython) -> None:
    """Fail if the fixture degenerated to a single level, which would test nothing."""
    levels = {grid._decode_flat_id(cid)[0] for cid in range(grid.num_cells)}
    assert len(levels) > 1, (
        f"the fixture has only level(s) {sorted(levels)}; on a single-level grid the "
        "descent never runs and these tests assert nothing about it"
    )


def test_hierarchical_locate_ids_are_identical(cpp_backend: None) -> None:
    """Both backends assign every point to the same hierarchical cell.

    This is the path whose equality is measured rather than derived, so the ids are
    asserted as a verdict on their own -- there is no numeric comparison here at all
    to fold them into.
    """
    del cpp_backend
    grid = _two_level_grid()
    _assert_more_than_one_level(grid)
    rng = np.random.default_rng(99)
    pts = np.vstack(
        [
            rng.uniform(-0.5, 4.5, size=(400, 2)),
            np.array([[0.0, 0.0], [4.0, 4.0], [2.0, 2.0], [1.0, 3.0]]),  # corners and breakpoints
        ]
    )

    reference, actual = _both(lambda: grid.locate_many(pts))

    np.testing.assert_array_equal(
        actual,
        reference,
        err_msg="a hierarchical cell id changed between backends. That is a changed "
        "verdict, not a displaced value, and it is the outcome the descent's "
        "contraction site was measured NOT to produce -- so re-measure rather "
        "than loosen anything.",
    )
    assert (np.asarray(reference) >= 0).any(), "every point landed outside; nothing was asserted"


def test_hierarchical_cell_bounds_are_bitwise(cpp_backend: None) -> None:
    """Both backends materialize identical per-cell bounds.

    Unlike the ids above these are coordinates, so this is where the descent's
    ``root_lo + sub * size`` contraction site would show as a displacement if it
    showed at all.
    """
    del cpp_backend
    grid = _two_level_grid()
    _assert_more_than_one_level(grid)

    reference, actual = _both(lambda: grid.collect_cell_bounds())

    ref_lo, ref_hi = reference
    act_lo, act_hi = actual
    assert_parity(
        act_lo,
        ref_lo,
        bitwise_parity(why=_HIER_WHY),
        context="HierarchicalGrid.collect_cell_bounds lower corners",
    )
    assert_parity(
        act_hi,
        ref_hi,
        bitwise_parity(why=_HIER_WHY),
        context="HierarchicalGrid.collect_cell_bounds upper corners",
    )


def test_hierarchical_encode_and_decode_agree(cpp_backend: None) -> None:
    """Both backends round-trip every flat id through decode and back through encode.

    Two of the eight dispatched kernels are reached only here, and both are pure
    integer arithmetic, so the equality is structural and the round trip is the
    property that makes the test non-vacuous: it checks the two kernels against each
    other rather than each against a table.
    """
    del cpp_backend
    grid = _two_level_grid()
    _assert_more_than_one_level(grid)
    n = grid.num_cells
    assert n > 0

    def round_trip() -> list[tuple[int, tuple[int, ...], int | None]]:
        out: list[tuple[int, tuple[int, ...], int | None]] = []
        for cid in range(n):
            level, midx = grid._decode_flat_id(cid)
            back = grid._encode_midx(level, midx)
            out.append((level, tuple(midx), back))
        return out

    reference, actual = _both(round_trip)

    assert actual == reference, "a decode/encode round trip differed between backends"
    for cid, (_, _, back) in enumerate(reference):
        assert back == cid, f"the round trip of id {cid} returned {back}"


def _non_dyadic_grid() -> _HierarchicalGridPython:
    """A hierarchy whose descent arithmetic is inexact, unlike ``_two_level_grid``.

    ``_two_level_grid`` subdivides by 2 over ``[0, 4]``, so the per-axis child index
    ``j`` is 0 or 1 and ``j * size_k`` is *exact* -- there is no rounding in that
    product for a fused multiply-add to remove. That fixture therefore cannot
    exercise the one contraction site ``hierarchical.hpp`` names as this kernel's
    parity risk, which is the site both hierarchical parity claims rest on.

    This one subdivides by 3 over non-dyadic spans that straddle zero, so ``size_k``
    is inexact, ``j`` reaches 2, and ``lo[k] + j * size_k`` is a **cancellation** --
    the regime where a fused and an unfused sum differ by the most. The straddle is
    deliberate: ``hierarchical.hpp`` argues that fusing cannot bite because ``|lo|``
    does not shrink while ``|j * size|`` shrinks geometrically, and that argument does
    not hold once the two terms have opposite signs.
    """
    root = uniform_grid(np.array([[-1.0, 2.0 / 3.0], [-0.1, 0.3]]), 3)
    grid = _HierarchicalGridPython(root, 3)  # the oracle: see `_two_level_grid`
    grid = grid.refine_cells([0, 4, 8])
    grid = grid.refine_cells([9, 15])
    return grid


def _non_dyadic_points(grid: _HierarchicalGridPython) -> npt.NDArray[np.float64]:
    """Interior points plus the cell corners and their ulp neighbours.

    The frontier half is the point of it. A truncation `int((x - lo) / size_k)` only
    changes when a one-ulp move in `lo` carries `(x - lo) / size_k` across an integer,
    so a sweep of random interior points is the one thing that cannot detect the
    contraction hazard this fixture exists for -- the same reason
    `_adversarial_points_1d` exists for the tensor-product tie contract.
    """
    lo, hi = grid.collect_cell_bounds()
    corners = np.vstack([lo, hi])
    nudged = [corners]
    for direction in (-np.inf, np.inf):
        nudged.append(np.nextafter(corners, direction))
        nudged.append(np.nextafter(np.nextafter(corners, direction), direction))
    rng = np.random.default_rng(20260825)
    interior = np.column_stack(
        [rng.uniform(-1.0, 2.0 / 3.0, size=4000), rng.uniform(-0.1, 0.3, size=4000)]
    )
    return np.vstack([*nudged, interior])


def test_hierarchical_locate_ids_survive_a_non_dyadic_descent(cpp_backend: None) -> None:
    """Both backends return the same ids where the descent's product is inexact.

    The same verdict assertion as the two-level test, in the regime that test cannot
    reach. A changed id here is a changed verdict, not a displaced value.
    """
    del cpp_backend
    grid = _non_dyadic_grid()
    _assert_more_than_one_level(grid)
    points = _non_dyadic_points(grid)

    reference, actual = _both(lambda: grid.locate_many(points))

    located = int((reference >= 0).sum())
    assert located > len(points) // 2, (
        f"only {located} of {len(points)} points landed in a cell; the fixture has "
        "drifted off its own domain and is no longer exercising the descent"
    )
    np.testing.assert_array_equal(
        actual, reference, err_msg=f"a hierarchical cell id changed. {_HIER_WHY}"
    )


def test_hierarchical_cell_bounds_are_bitwise_on_a_non_dyadic_grid(cpp_backend: None) -> None:
    """Both backends materialize identical bounds where ``sub_ik * size_k`` is inexact.

    The companion to the ids above, on the coordinates rather than the verdict. With
    subdivision factor 3 the offset ``sub_ik`` reaches 2 and the product genuinely
    rounds, which is what the shipped factor-2 fixture cannot produce.
    """
    del cpp_backend
    grid = _non_dyadic_grid()
    _assert_more_than_one_level(grid)

    reference, actual = _both(lambda: grid.collect_cell_bounds())

    ref_lo, ref_hi = reference
    act_lo, act_hi = actual
    assert_parity(
        act_lo,
        ref_lo,
        bitwise_parity(why=_HIER_WHY),
        context="collect_cell_bounds lower corners on a non-dyadic grid",
    )
    assert_parity(
        act_hi,
        ref_hi,
        bitwise_parity(why=_HIER_WHY),
        context="collect_cell_bounds upper corners on a non-dyadic grid",
    )


def test_the_descent_counterexample_runs_through_the_real_kernels(cpp_backend: None) -> None:
    """Both backends agree on the input that separates a fused descent from an unfused one.

    The companion to :func:`test_the_descent_counterexample_is_a_genuine_discriminator`,
    and the half that exercises the compiled code. That one asserts the input is still
    a discriminator in exact arithmetic; this one asserts the two shipped kernels
    resolve it the same way.

    On a build with no ``-march`` this passes whether or not the port keeps its two
    statements separate, because the baseline ISA has no fused multiply-add to
    contract into -- so on its own it proves nothing here. Its value is that it
    **fails** the moment ``design/simd.md``'s ISA ladder lands and the product is
    inlined again: measured, an inlined build at ``-march=native`` returns cell 71
    where the oracle returns 70.

    The grid is the counterexample's own geometry: root cell ``[1.0, 2.0]``,
    subdivision factor 10, refined twice so the descent chains, and the query point
    ``x = 1.71``.
    """
    del cpp_backend
    root = TensorProductGrid([np.array([1.0, 2.0])])
    grid = _HierarchicalGridPython(root, 10)  # the oracle: see `_two_level_grid`
    grid = grid.refine_cells([0])
    point = np.array([[1.71]])
    first = grid.locate_many(point)
    grid = grid.refine_cells([int(first[0])])

    reference, actual = _both(lambda: grid.locate_many(point))

    np.testing.assert_array_equal(
        actual,
        reference,
        err_msg="the two backends resolved the descent counterexample to different "
        "cells. If this build has a fused multiply-add, the product at "
        "hierarchical.hpp's descent site has been inlined again; see that file.",
    )
    level, _ = grid._decode_flat_id(int(reference[0]))
    assert level == 2, (
        f"the point landed at level {level}, so the descent did not chain twice and "
        "this case no longer reaches the contraction site it exists to guard"
    )


def test_the_descent_counterexample_is_a_genuine_discriminator() -> None:
    """The input that separates a fused descent from an unfused one still separates them.

    This is the case that refuted an earlier version of this file's equality claim:
    root cell ``[1.0, 2.0]``, factor 10, ``x = 1.71``, a decimal literal rather than
    a bit-twiddled value. At level 0 the unfused sum is ``1.7000000000000002`` and
    the fused one is ``1.7``, one ulp apart; at level 1 the quotients are
    ``0.9999999999999778`` and exactly ``1.0``, so the truncation gives child 0
    against child 1 and the two arithmetics land in different cells.

    **This test does not need the extension and does not compare backends**, and that
    is deliberate. On a build with no ``-march`` there is no fused multiply-add
    instruction at all, so a backend comparison here would pass whether or not the
    port keeps its two statements separate -- it would be exactly the kind of
    comfortable case that let the original claim stand. What this asserts instead is
    that the input remains a **discriminator**: that a fused evaluation really does
    diverge from an unfused one on it. If a future numpy, libm or Python changed
    something that made this input benign, the guard in
    ``cpp/include/pantr/grid/hierarchical.hpp`` would be resting on nothing and this
    test says so before the parity suite silently stops testing anything.

    The paired assertion -- that the *shipped* kernels agree -- is what
    :func:`test_hierarchical_locate_ids_are_identical` and the non-dyadic descent
    test do. Their value rises the moment ``design/simd.md``'s ISA ladder lands.
    """
    lo0, hi0, x, factor = 1.0, 2.0, 1.71, 10

    size = (hi0 - lo0) / factor
    j = int((x - lo0) / size)
    assert j == 7, f"the fixture drifted: expected child 7, got {j}"

    # The unfused half needs no `math.fma`, so it is pinned on every supported
    # Python. These are the exact bits the shipped kernels and the oracle produce.
    unfused = lo0 + j * size
    assert unfused.hex() == "0x1.b333333333334p+0", (
        f"the unfused level-0 sum drifted to {unfused.hex()}; this case is built on its exact value"
    )
    child_unfused = int((x - unfused) / ((unfused + size - unfused) / factor))
    assert child_unfused == 0, f"the unfused descent no longer takes child 0, got {child_unfused}"

    # The fused half needs a correctly rounded fused multiply-add. `math.fma` is
    # Python 3.13+, and this project supports 3.11, so it is gated rather than
    # assumed -- the alternative would be an AttributeError on two of the four
    # supported versions.
    if not hasattr(math, "fma"):  # pragma: no cover - version-dependent
        pytest.skip("math.fma is Python 3.13+; the unfused half above still ran")

    fused = math.fma(float(j), size, lo0)
    assert fused.hex() == "0x1.b333333333333p+0", f"the fused level-0 sum drifted to {fused.hex()}"
    assert unfused != fused, (
        "level 0 no longer distinguishes a fused sum from an unfused one, so this "
        "input has stopped being a counterexample and the guard in the header rests "
        "on nothing"
    )
    assert abs(unfused - fused) == abs(np.nextafter(fused, math.inf) - fused), (
        "the two sums are no longer exactly one ulp apart"
    )

    # The divergence has to survive into the next level's truncation to change a
    # cell id; one ulp in a coordinate is not by itself a changed verdict.
    child_fused = int((x - fused) / ((fused + size - fused) / factor))
    assert child_unfused != child_fused, (
        f"level 1 now agrees ({child_unfused}), so the one-ulp difference no longer "
        "reaches the truncation and this case pins nothing about the descent"
    )


def test_bvh_node_boxes_agree_bit_for_bit_on_a_signed_zero_tie(cpp_backend: None) -> None:
    """Both backends keep the same zero when a node box's min or max ties.

    The node box is a running min/max over corners, and on a tie both sides keep the
    **incumbent** -- Python's two-argument ``min`` returns its first argument, and
    the C++ only assigns on a strict ``<``. That agreement is one of the three
    tie-breaks ``bvh.hpp`` argues its equality from, and until this test nothing
    exercised it: mutating those comparisons to ``<=``/``>=`` left the whole parity
    suite green, because a fixture of continuous random floats never produces a tie
    at all, and a tie between two *identical* values is unobservable.

    ``-0.0`` and ``+0.0`` are what make it observable: equal in value, different in
    bits, so which one survives is visible.

    **This test asserts that the two backends agree, and deliberately not which zero
    they keep.** ``CLAUDE.md`` records that ``np.minimum(-0.0, 0.0)`` returns
    different signs on different NumPy versions in the CI matrix, so pinning the sign
    would pin something no platform promises. The oracle does not use that ufunc --
    it uses the builtin ``min`` inside a kernel -- but the agreement is the property
    the port needs either way, and it is the one that survives a NumPy upgrade.

    It compares bit patterns directly rather than through
    :func:`~tests._parity_harness.assert_parity`, and that is required rather than a
    preference: the harness's comparison exempts signed-zero disagreements on
    purpose, so the one hazard this test exists for is the one it cannot see.
    """
    del cpp_backend
    # Two cells sharing a corner at zero, written with opposite signs. Whichever the
    # min keeps, both backends must keep the same one.
    lo = np.array([[-0.0, 1.0], [0.0, -1.0]])
    hi = np.array([[2.0, 3.0], [1.0, -0.0]])

    reference, actual = _both(lambda: BVH.from_cell_bounds(lo, hi))

    for name in ("node_lo", "node_hi"):
        ref_bits = np.asarray(getattr(reference, name), dtype=np.float64).view(np.uint64)
        act_bits = np.asarray(getattr(actual, name), dtype=np.float64).view(np.uint64)
        np.testing.assert_array_equal(
            act_bits,
            ref_bits,
            err_msg=f"the two backends kept different zeros in {name}. One of them "
            "stopped keeping the incumbent on a min/max tie, which is one of the "
            "three tie-breaks bvh.hpp's equality argument rests on.",
        )


def test_bvh_split_permutation_agrees_on_a_large_tied_set(cpp_backend: None) -> None:
    """Both backends split an all-tied centroid set the same way.

    The median split sorts by centroid, and ``bvh.hpp`` argues the two backends agree
    because a **stable** sort's output permutation is uniquely determined. That
    argument is sound and, until this test, nothing checked it: replacing
    ``std::stable_sort`` with ``std::sort`` left the whole parity suite green.

    Two things have to be true for the substitution to be visible, and the existing
    fixtures satisfy neither. The centroids must actually tie, which continuous
    random floats never do. And there must be **enough** of them: libstdc++'s
    introsort falls back to insertion sort below roughly sixteen elements, and
    insertion sort happens to be stable, so a small tied set cannot tell the two
    apart. Measured, ten tied pairs separate them and four cells do not.

    Every cell here is centred on the origin, so every centroid is exactly ``0.0``
    and the sort has nothing to order -- the resulting tree is decided entirely by
    which permutation the sort leaves behind, and that shows up in ``node_cell``.
    """
    del cpp_backend
    n_cells = 24
    half = np.arange(1, n_cells + 1, dtype=np.float64).reshape(-1, 1)
    lo, hi = -half, half

    reference, actual = _both(lambda: BVH.from_cell_bounds(lo, hi))

    np.testing.assert_array_equal(
        actual.node_cell,
        reference.node_cell,
        err_msg="the two backends split an all-tied centroid set differently, so one "
        "of them is not using a stable sort. bvh.hpp's equality argument is that a "
        "stable sort's permutation is uniquely determined; this is what checks it.",
    )
    assert len(set(reference.node_cell.tolist())) > n_cells // 2, (
        "the fixture stopped producing distinct leaves, so it no longer observes a permutation"
    )
