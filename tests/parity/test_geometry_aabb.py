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
implementations agree on **which exception** they raise and on **what it says**,
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


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda cls: cls(np.array([0.0, 1.0]), np.array([1.0])), "must share shape"),
        (lambda cls: cls(np.array([np.nan]), np.array([1.0])), "must not contain NaN"),
    ],
)
def test_construction_errors_agree(build: Any, match: str) -> None:
    """Both implementations raise `ValueError`, and say the same thing.

    The message is part of the claim, not decoration: a caller that catches on it
    would otherwise see `PANTR_BACKEND` change the library's behaviour rather than
    only its speed.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    with pytest.raises(ValueError, match=match):
        build(_AABBPython)
    with pytest.raises(ValueError, match=match):
        build(_pantr_cpp.AABB)


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
