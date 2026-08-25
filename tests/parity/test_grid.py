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
tolerance bounds. Measured over 300000 descents it never diverges, and the reason is
that the perturbation is never introduced rather than that it fails to grow, but
that is evidence and not proof.

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

from collections.abc import Callable
from functools import partial
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.geometry import AABB
from pantr.grid import BVH, HierarchicalGrid, hierarchical_grid, uniform_grid
from tests._parity_harness import assert_parity, bitwise_parity

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
    "the descent's `lo + j * size` is a contraction site and this equality is "
    "MEASURED rather than derived: 300000 descents of twelve levels, 23.8% of the "
    "child decisions at a non-power-of-two j, half the points one ulp off a child "
    "boundary, and no divergence. Fusing moves a sum only when its two terms are "
    "comparable in magnitude, and in a descent the second shrinks geometrically. "
    "A build on a target that contracts differently would be the thing to re-measure"
)


_T = TypeVar("_T")


def _both(call: Callable[[], _T]) -> tuple[_T, _T]:
    """Run one callable under each backend and return ``(reference, actual)``."""
    with use_backend(Backend.PYTHON):
        reference = call()
    with use_backend(Backend.CPP):
        actual = call()
    return reference, actual


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
    """
    del cpp_backend
    grid = uniform_grid(np.tile([0.0, 3.0], (ndim, 1)), 3)
    rng = np.random.default_rng(20260825)
    interior = rng.uniform(-0.5, 3.5, size=(200, ndim))
    corners = np.array(np.meshgrid(*([[0.0, 1.0, 2.0, 3.0]] * ndim))).reshape(ndim, -1).T
    points = np.vstack([interior, corners])

    reference, actual = _both(lambda: grid.locate_many(points))

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
    grid = uniform_grid(np.tile([0.0, 3.0], (1, 1)), 3)
    points = _adversarial_points_1d()

    reference, actual = _both(lambda: grid.locate_many(points))

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

    for name in ("node_lo", "node_hi"):
        assert_parity(
            getattr(actual, name),
            getattr(reference, name),
            bitwise_parity(why=_BVH_WHY),
            context=f"BVH.from_cell_bounds {name}, {n_cells} cells in {ndim}-D",
        )
    for name in ("node_left", "node_right", "node_cell"):
        np.testing.assert_array_equal(
            getattr(actual, name),
            getattr(reference, name),
            err_msg=f"BVH.from_cell_bounds {name} differs at {n_cells} cells in {ndim}-D. "
            "The two backends built trees of different SHAPE, which no tolerance could "
            f"bound. {_BVH_WHY}",
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


def _two_level_grid() -> HierarchicalGrid:
    """A hierarchical grid with a genuinely refined region.

    ``hierarchical_grid`` alone returns a **single-level** grid, and on one of those
    the descent below never runs: the root lookup finds an active cell at level 0 and
    returns, so the contraction site these tests exist to exercise is never reached.
    So this refines a corner, and the callers assert that it worked rather than
    trusting that it did.
    """
    root = uniform_grid(np.tile([0.0, 4.0], (2, 1)), 4)
    grid = hierarchical_grid(root, 2)
    grid.refine_cells([0, 1, 4])
    return grid


def _assert_more_than_one_level(grid: HierarchicalGrid) -> None:
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
