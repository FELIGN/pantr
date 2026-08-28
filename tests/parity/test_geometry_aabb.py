"""Parity of the C++ `AABB` against the pure-Python oracle it was ported from.

`cpp/include/pantr/geometry/aabb.hpp` is the first type in the C++ core rather than
the first kernel, and that changes what a parity claim here means.

The claim, and why it is an equality
------------------------------------

Every operation on a box is min, max, a comparison, or a copy. There is no
accumulation, no cancellation and no transcendental call, so there is **no forward
error to bound** and the claim is exact equality of the corner arrays, not
agreement within a tolerance. A tolerance here would not be a safety margin; it
would be hiding a defect, because nothing in these operations can move a bit.

`transform` is the one exception and it is bounded rather than exact in principle:
it multiplies and sums. In practice the sum is over `ndim` terms and numpy's
`np.sum` is pairwise only above a block of 8, so for every box pantr builds the two
summation orders coincide and the results are bit-identical. The tests below stay
inside that regime deliberately and say so; a box beyond 8 axes would need the
claim restated as a bound, and `design/backend_parity.md` Rule 9 is the shape that
restatement would take.

What these tests would catch
----------------------------

The three that matter are the ones no numeric comparison reaches: that the two
implementations agree on **which exception** they raise and, verbatim, on **what
it says**,
that they agree on the **empty** cases where the answer is a discrete verdict
rather than a number, and that `PANTR_BACKEND` does not change the wrapper's
identity semantics -- `__eq__`, `__hash__` and the pickle round trip are computed
by the wrapper precisely so that they cannot drift, and this is what says so.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pytest

from pantr._backend import Backend, use_backend
from pantr.geometry import AABB, _AABBPython

pytestmark = pytest.mark.usefixtures("cpp_backend")


def _cpp(lo: Any, hi: Any) -> Any:
    """Build a box in the C++ implementation, bypassing the active backend.

    Args:
        lo (Any): Lower corner.
        hi (Any): Upper corner.

    Returns:
        Any: The C++ box.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    return _pantr_cpp.AABB(
        np.ascontiguousarray(lo, dtype=np.float64), np.ascontiguousarray(hi, dtype=np.float64)
    )


def _both(lo: Any, hi: Any) -> tuple[_AABBPython, Any]:
    """Build the same box in both implementations.

    Args:
        lo (Any): Lower corner.
        hi (Any): Upper corner.

    Returns:
        tuple[_AABBPython, Any]: The oracle box and the C++ box.
    """
    return _AABBPython(lo, hi), _cpp(lo, hi)


# Corner cases chosen for the phenomena, not for coverage: an ordinary box, a
# degenerate one, an empty one, one unbounded on a single axis, and one whose
# corner is a negative zero.
_BOXES = [
    ([0.0, 0.0], [1.0, 2.0]),
    ([1.0, 1.0], [1.0, 1.0]),
    ([5.0, 0.0], [3.0, 1.0]),
    ([0.0, -np.inf], [1.0, np.inf]),
    ([-0.0], [1.0]),
    ([-np.inf], [np.inf]),
]


@pytest.mark.parametrize(("lo", "hi"), _BOXES)
def test_corners_are_bit_identical(lo: Any, hi: Any) -> None:
    """The stored corners agree bit for bit, signed zero included."""
    py, cpp = _both(lo, hi)
    assert py.lo.tobytes() == cpp.lo.tobytes()
    assert py.hi.tobytes() == cpp.hi.tobytes()
    assert not cpp.lo.flags.writeable, "the C++ box must hand out read-only corners too"


@pytest.mark.parametrize(("lo", "hi"), _BOXES)
def test_predicates_agree(lo: Any, hi: Any) -> None:
    """`is_empty` and `contains_point` return the same verdict on both sides."""
    py, cpp = _both(lo, hi)
    assert py.is_empty() == cpp.is_empty()
    probe = np.zeros(len(lo), dtype=np.float64)
    assert py.contains_point(probe) == cpp.contains_point(probe)


@pytest.mark.parametrize(("lo_b", "hi_b"), _BOXES)
@pytest.mark.parametrize(("lo_a", "hi_a"), _BOXES)
def test_binary_operations_agree(lo_a: Any, hi_a: Any, lo_b: Any, hi_b: Any) -> None:
    """`overlaps`, `union` and `intersect` agree over every pair of the corner cases.

    Exhaustive over the pair rather than sampled: the interesting combinations are
    empty-against-containing and unbounded-against-degenerate, and a sample can
    miss exactly those.
    """
    if len(lo_a) != len(lo_b):
        pytest.skip("dimensions must match; the mismatch case is tested separately")
    py_a, cpp_a = _both(lo_a, hi_a)
    py_b, cpp_b = _both(lo_b, hi_b)

    assert py_a.overlaps(py_b) == cpp_a.overlaps(cpp_b)

    py_u, cpp_u = py_a.union(py_b), cpp_a.union(cpp_b)
    assert py_u.lo.tobytes() == cpp_u.lo.tobytes()
    assert py_u.hi.tobytes() == cpp_u.hi.tobytes()

    py_i, cpp_i = py_a.intersect(py_b), cpp_a.intersect(cpp_b)
    assert (py_i is None) == (cpp_i is None)
    if py_i is not None and cpp_i is not None:
        assert py_i.lo.tobytes() == cpp_i.lo.tobytes()
        assert py_i.hi.tobytes() == cpp_i.hi.tobytes()


@pytest.mark.parametrize("radius", [0.0, 0.5, -0.25, -10.0])
@pytest.mark.parametrize(("lo", "hi"), _BOXES)
def test_pad_agrees(lo: Any, hi: Any, radius: float) -> None:
    """`pad` agrees exactly, including the negative radius that empties a box."""
    py, cpp = _both(lo, hi)
    py_p = py.pad(radius)
    cpp_p = cpp.pad(np.full(len(lo), radius, dtype=np.float64))
    assert py_p.lo.tobytes() == cpp_p.lo.tobytes()
    assert py_p.hi.tobytes() == cpp_p.hi.tobytes()


class _Affine:
    """A minimal `_AffineMap` for the transform tests.

    Attributes:
        dim (int): Spatial dimension.
        matrix (npt.NDArray[np.float64]): The linear part.
        offset (npt.NDArray[np.float64]): The translation.
    """

    def __init__(self, matrix: Any, offset: Any) -> None:
        """Store the two parts as float64 arrays.

        Args:
            matrix (Any): The linear part.
            offset (Any): The translation.
        """
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        self.offset = np.ascontiguousarray(offset, dtype=np.float64)
        self.dim = int(self.matrix.shape[0])


@pytest.mark.parametrize(
    "matrix",
    [
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, -1.0], [1.0, 0.0]],
        # A zero row: it must project an unbounded axis out rather than make NaN.
        [[1.0, 0.0], [0.0, 0.0]],
        [[2.0, 3.0], [-1.0, 0.5]],
    ],
)
def test_transform_agrees(matrix: Any) -> None:
    """`transform` agrees bit for bit at ndim = 2, inside numpy's naive-sum block.

    At two axes `np.sum` performs the same two additions in the same order as the
    C++ loop, so the equality claim holds rather than a bound. See this module's
    docstring for where it would stop.
    """
    affine = _Affine(matrix, [0.25, -0.5])
    for lo, hi in _BOXES:
        if len(lo) != 2:
            continue
        py, cpp = _both(lo, hi)
        py_t = py.transform(affine)
        cpp_t = cpp.transform(affine.matrix, affine.offset)
        assert py_t.lo.tobytes() == cpp_t.lo.tobytes(), f"lo differs for {lo}, {hi}"
        assert py_t.hi.tobytes() == cpp_t.hi.tobytes(), f"hi differs for {lo}, {hi}"


def test_transform_agrees_on_inexact_matrices() -> None:
    """`transform` agrees bit for bit on matrices that are NOT exactly representable.

    This is the test that was missing, and its absence hid a real defect. Every
    matrix in `test_transform_agrees` is built from 0, +/-1, 2, 3 and 0.5, all
    exact in binary, so every product and partial sum is exact and *any*
    accumulation order reproduces the oracle. The suite passed while the C++ loop
    seeded its accumulator with the offset instead of adding it last, which
    diverges on 94% of random inputs at ndim = 3.

    Random inputs from a fixed seed, so a failure is reproducible, and wide
    enough that a single-rounding difference cannot hide in it.
    """
    rng = np.random.default_rng(20260827)
    for _ in range(200):
        ndim = int(rng.integers(2, 8))
        affine = _Affine(rng.normal(size=(ndim, ndim)), rng.normal(size=ndim))
        lo = rng.normal(size=ndim)
        hi = lo + np.abs(rng.normal(size=ndim))
        py, cpp = _both(lo, hi)
        py_t = py.transform(affine)
        cpp_t = cpp.transform(affine.matrix, affine.offset)
        assert py_t.lo.tobytes() == cpp_t.lo.tobytes()
        assert py_t.hi.tobytes() == cpp_t.hi.tobytes()


def test_past_seven_axes_the_claim_is_a_bound_not_an_equality() -> None:
    """Above ndim = 7 the two summation orders differ, and only a bound survives.

    numpy's `np.sum` blocks pairwise, so a sequential C++ loop reproduces it
    exactly only while ndim <= 7; from ndim = 8 the orders differ on about half
    of random inputs. Measured by `scripts/measure_aabb_transform_summation.py`.

    This does not assert that they differ -- that would pin numpy's blocking,
    which is not ours -- but it does assert the disagreement stays within one
    reordering's worth of rounding. Exceeding that would mean a defect rather
    than a summation order.
    """
    rng = np.random.default_rng(11)
    ndim = 9
    for _ in range(50):
        affine = _Affine(rng.normal(size=(ndim, ndim)), rng.normal(size=ndim))
        lo = rng.normal(size=ndim)
        hi = lo + np.abs(rng.normal(size=ndim))
        py, cpp = _both(lo, hi)
        py_t = py.transform(affine)
        cpp_t = cpp.transform(affine.matrix, affine.offset)
        for a, b in ((py_t.lo, cpp_t.lo), (py_t.hi, cpp_t.hi)):
            # Reordering a sum of n terms perturbs it by at most (n - 1)
            # roundings of the largest partial sum, itself bounded by the sum of
            # the magnitudes; the max entry stands in for that here.
            budget = (ndim - 1) * float(np.finfo(np.float64).eps) * float(np.abs(a).max())
            assert np.all(np.abs(a - b) <= budget)


def _message_of(fn: Any) -> str:
    """Run `fn` and return the text of the ValueError it raises.

    Args:
        fn (Any): A zero-argument call expected to raise.

    Returns:
        str: The exception's message.

    Raises:
        AssertionError: If `fn` did not raise `ValueError`.
    """
    try:
        fn()
    except ValueError as exc:
        return str(exc)
    raise AssertionError("expected a ValueError and got none")


@pytest.mark.parametrize(
    ("build", "what"),
    [
        (lambda cls: cls(np.array([0.0, 1.0]), np.array([1.0])), "corner shapes differ"),
        (lambda cls: cls(np.array([np.nan]), np.array([1.0])), "NaN in a corner"),
        (lambda cls: cls(np.zeros(0), np.zeros(0)), "zero dimensions"),
        (lambda cls: cls.unbounded(0), "unbounded with ndim 0"),
        (lambda cls: cls.empty(0), "empty with ndim 0"),
        (
            lambda cls: cls(np.zeros(2), np.ones(2)).contains_point(np.zeros(3)),
            "wrong point length",
        ),
        (lambda cls: cls(np.zeros(1), np.ones(1)).pad(np.array([np.inf])), "non-finite pad"),
        (
            lambda cls: cls(np.zeros(1), np.ones(1)).union(cls(np.zeros(2), np.ones(2))),
            "dimension mismatch",
        ),
    ],
)
def test_error_messages_agree_verbatim(build: Any, what: str) -> None:
    """Both implementations raise `ValueError` and say **exactly** the same thing.

    Verbatim, not a substring. An earlier version of this test used
    `pytest.raises(match=...)`, which is a substring search, and it passed while
    five of these six messages differed between the backends -- the C++ side was
    dropping the offending values, spelling the class separator `::` instead of
    `.`, and reporting a bare length where the oracle reports a shape tuple.

    The message is part of the claim rather than decoration: a caller that
    catches on it would otherwise see `PANTR_BACKEND` change what the library
    says rather than only how fast it is.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    oracle = _message_of(lambda: build(_AABBPython))
    ported = _message_of(lambda: build(_pantr_cpp.AABB))
    assert oracle == ported, f"{what}: oracle said {oracle!r}, C++ said {ported!r}"


def test_transform_on_an_empty_box_agrees_about_a_malformed_map() -> None:
    """An empty box short-circuits before the map is validated, on both backends.

    The oracle returns an empty box without looking at `affine` at all, so a
    malformed map against an empty box is silently accepted. The wrapper's C++
    branch validated first and raised, which made `PANTR_BACKEND` decide whether
    the call raised or returned -- the one thing the backend switch must never
    do.

    This pins the behaviour rather than endorsing it. Whether the oracle's check
    order is right is a separate question about the oracle.
    """
    wrong = _Affine(np.eye(3), np.zeros(3))
    with use_backend(Backend.PYTHON):
        from_python = AABB.empty(2).transform(wrong)
    with use_backend(Backend.CPP):
        from_cpp = AABB.empty(2).transform(wrong)
    assert from_python == from_cpp
    assert from_cpp.is_empty()


def test_dimension_mismatch_messages_agree() -> None:
    """The binary operations report a dimension mismatch identically."""
    from pantr import _pantr_cpp  # noqa: PLC0415

    match = r"AABB\.union: dimension mismatch"
    py_a = _AABBPython(np.zeros(2), np.ones(2))
    py_b = _AABBPython(np.zeros(3), np.ones(3))
    with pytest.raises(ValueError, match=match):
        py_a.union(py_b)

    cpp_a = _pantr_cpp.AABB(np.zeros(2), np.ones(2))
    cpp_b = _pantr_cpp.AABB(np.zeros(3), np.ones(3))
    with pytest.raises(ValueError, match=match):
        cpp_a.union(cpp_b)


@pytest.mark.parametrize(("lo", "hi"), _BOXES)
def test_the_wrapper_is_backend_invariant(lo: Any, hi: Any) -> None:
    """`__eq__`, `__hash__` and the pickle round trip do not move with the backend.

    These three are computed by the wrapper rather than delegated, so that a dict
    built under one backend stays readable under the other and a pickle written
    under one loads under the other. Nothing else in the suite would notice if
    that stopped being true.
    """
    with use_backend(Backend.PYTHON):
        py_box = AABB(lo, hi)
    with use_backend(Backend.CPP):
        cpp_box = AABB(lo, hi)

    assert isinstance(cpp_box._impl, type(_cpp(lo, hi)))
    assert isinstance(py_box._impl, _AABBPython)
    assert py_box == cpp_box
    assert hash(py_box) == hash(cpp_box)
    assert repr(py_box) == repr(cpp_box)
    assert pickle.loads(pickle.dumps(cpp_box)) == py_box
