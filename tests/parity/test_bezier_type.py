"""Parity of the `Bezier` value type: the state, the rejections, and the wire format.

What this file compares is a *value*, not a computation. `pantr.bezier.Bezier`
stores control points and a flag and answers four questions about them; between
the Python oracle and the C++ port there is no arithmetic at all, only a copy. So
every claim here is bitwise or exact, and a tolerance anywhere in this file would
be hiding a transcription error rather than allowing for rounding. The kernels
that *do* arithmetic are the subject of `test_bezier_arithmetic.py`, which builds
its Béziers through this same type.

Three things are checked that a field-by-field state comparison does not reach:

- **The rejections, verbatim.** Both implementations refuse the same malformed
  control nets, and a caller writing ``pytest.raises(ValueError, match=...)``
  must not have to know which backend built the object. The messages are
  therefore compared character for character rather than by exception type.
- **The copy at construction, and the read-only view on the way out.** The C++
  value does not alias the caller's array at either end, which the oracle does.
  This is the one place the two backends differ on purpose: it is
  FELIGN/pantr#338's defect, fixed on the side the port owns, and
  FELIGN/pantr#375 is the ticket that fixes the other. Both halves are asserted
  here, because a criterion covering only construction leaves the other half of a
  bidirectional defect unpinned.
- **The wire format.** ``__reduce__`` pickles by the constructor's arguments, so a
  pickle written under one backend loads under the other. Without that the backend
  switch would silently become a data-format switch, and a C++ handle is not
  picklable in any case.

The dtype is part of the state and is compared as such. It is the axis on which
this type differs from the two already-ported ones: `AABB` and `AffineTransform`
coerce everything to ``float64`` and are bound at ``double`` only, while a Bézier
stores ``float32`` too and the C++ side therefore registers two classes.
"""

from __future__ import annotations

import copy
import pickle
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.bezier import Bezier
from pantr.bezier._bezier import _BezierPython
from tests._parity_harness import (
    Field,
    assert_object_parity,
    bitwise_parity,
    exact_parity,
)

pytestmark = pytest.mark.usefixtures("cpp_backend")

DTYPES: Final = (np.float64, np.float32)
"""Both storage formats. Unlike `AABB`, a Bézier has a genuine `float32` oracle."""

_STORED_WHY: Final = (
    "the value type stores the coefficients it is handed and performs no arithmetic on "
    "them, so the two backends can only differ by a transcription error -- a wrong stride, "
    "a narrowing cast, a byte lost to the shape. Every one of those is visible bitwise and "
    "invisible under any tolerance wide enough to be called a rounding budget."
)

_DERIVED_WHY: Final = (
    "dim, degree and rank are subtractions on the shape, computed in exact integer "
    "arithmetic on both sides. A difference is an off-by-one in the shape handling, which "
    "is exact or wrong."
)

_FLAG_WHY: Final = "the rationality flag is stored and returned; nothing transforms it."

_DTYPE_WHY: Final = (
    "the storage format is the state a caller reads through `Bezier.dtype`, and on the C++ "
    "side it is carried by the class of the handle. A disagreement means `_impl_class` "
    "picked the wrong class, which would silently narrow or widen the geometry."
)

FIELDS: Final = (
    Field("control_points", bitwise_parity(why=_STORED_WHY)),
    Field("is_rational", exact_parity(why=_FLAG_WHY)),
    Field("dim", exact_parity(why=_DERIVED_WHY)),
    Field("degree", exact_parity(why=_DERIVED_WHY)),
    Field("rank", exact_parity(why=_DERIVED_WHY)),
    Field("dtype", exact_parity(why=_DTYPE_WHY)),
)
"""Every piece of a Bézier's state, one field each.

`degree` is a homogeneous tuple of ints, which is one quantity and stays one
field; the harness refuses a value mixing element kinds, and this is not one.
There is deliberately no derived-convenience field beyond these: `dtype` is a
function of `control_points` and is named anyway, because it is the axis the two
C++ classes differ on and a wrong one is otherwise only visible as a rounding.
"""


def _control_points(
    shape: tuple[int, ...], dtype: npt.DTypeLike, *, seed: int
) -> npt.NDArray[np.float32 | np.float64]:
    """Build a control net whose values span many decades.

    Magnitudes spread over the format's exponent range, so a narrowing cast in the
    port shows up as a changed bit pattern rather than as a value that happens to
    survive both formats.

    Args:
        shape (tuple[int, ...]): The control-net shape, `(*degrees_plus_1, rank)`.
        dtype (npt.DTypeLike): Storage format.
        seed (int): Seed for the draw.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The control points, C-contiguous.
    """
    rng = np.random.default_rng(seed)
    size = int(np.prod(shape))
    mantissa = rng.uniform(-1.0, 1.0, size=size)
    exponent = rng.integers(-20, 20, size=size)
    return np.ascontiguousarray((mantissa * np.float64(2.0) ** exponent).reshape(shape), dtype)


def _both(
    control_points: npt.NDArray[np.float32 | np.float64], *, is_rational: bool
) -> tuple[Bezier, Bezier]:
    """Build the same Bézier under each backend.

    Args:
        control_points (npt.NDArray[np.float32 | np.float64]): The control net.
        is_rational (bool): Whether the last coordinate is a weight.

    Returns:
        tuple[Bezier, Bezier]: `(py, cpp)`, in the order `assert_object_parity`
        names its arguments.
    """
    with use_backend(Backend.PYTHON):
        py = Bezier(control_points, is_rational)
    with use_backend(Backend.CPP):
        cpp = Bezier(control_points, is_rational)
    return py, cpp


def _cpp_class(dtype: npt.DTypeLike) -> Any:
    """The bound C++ class for one storage format.

    Args:
        dtype (npt.DTypeLike): `float32` or `float64`.

    Returns:
        Any: `pantr._pantr_cpp.Bezier32` or `Bezier64`.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    return _pantr_cpp.Bezier32 if np.dtype(dtype) == np.float32 else _pantr_cpp.Bezier64


# The shapes are `(*degrees_plus_1, num_components)`. Between them they cover
# every axis the type's arithmetic touches: one to three parametric directions,
# degree 0 (one coefficient, a constant, which is legal and is the case a
# `shape - 1` underflow would break), an asymmetric degree so a transposed shape
# cannot pass, and a component count of 1 so the rational rank check sits one
# step from its rejection.
SHAPES: Final = (
    (4, 1),
    (4, 3),
    (1, 2),
    (3, 5, 2),
    (2, 3, 4, 3),
)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("is_rational", [False, True])
def test_the_two_backends_hold_the_same_value(
    shape: tuple[int, ...], dtype: npt.DTypeLike, is_rational: bool
) -> None:
    """Every piece of state agrees, at both storage formats.

    What this catches: a stride or shape mistake in the flat-buffer round trip, a
    narrowing cast in the wrong-dtype class, an off-by-one in `degree` or `rank`,
    and the weight column being counted into the rank or out of the storage.
    """
    if is_rational and shape[-1] < 2:
        pytest.skip("a rational Bezier needs a weight and at least one coordinate")
    control_points = _control_points(shape, dtype, seed=hash(shape) % 2**32)
    py, cpp = _both(control_points, is_rational=is_rational)
    assert_object_parity(
        py=py, cpp=cpp, fields=FIELDS, context=f"Bezier({shape}, {dtype}, rational={is_rational})"
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_one_dimensional_control_net_is_reshaped_the_same_way(dtype: npt.DTypeLike) -> None:
    """`(n,)` becomes the scalar field `(n, 1)` on both sides.

    What this catches: the convenience living on only one side, which would make a
    flat coefficient vector a degree-0 Bézier of rank `n` under one backend and a
    degree-`n-1` Bézier of rank 1 under the other -- both well formed, so nothing
    downstream would raise.
    """
    control_points = np.asarray([0.0, 1.0, 4.0, 9.0], dtype=dtype)
    py, cpp = _both(control_points, is_rational=False)
    assert_object_parity(py=py, cpp=cpp, fields=FIELDS, context="Bezier from a 1-D array")
    assert cpp.control_points.shape == (4, 1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_rational", [False, True])
def test_the_ieee_menagerie_transits_unchanged(dtype: npt.DTypeLike, is_rational: bool) -> None:
    """NaN, both zeros, a subnormal and both infinities survive the copy.

    What this catches: the whole premise of this file is that the two backends
    only copy, and a copy is exactly where these values are cheap to test and
    expensive to lose. A caster round-tripping through a wider intermediate, a
    narrowing cast, or a memcpy replaced by element-wise assignment through a
    different type all show first on a NaN payload, on a ``-0.0`` whose sign an
    addition drops, or on a ``float32`` subnormal flushed to zero -- and every one
    of those is invisible to a value comparison and visible bitwise, which is why
    the claim on ``control_points`` is ``bitwise_parity`` and not a tolerance.

    The finite specials and the NaN go through the same ``FIELDS``, so ``dtype``,
    ``degree`` and ``rank`` are checked over them too: a Bézier of NaNs is not a
    special case in the type's eyes and must not become one.

    **The two infinities are asserted separately, and not by choice.**
    ``assert_parity`` computes a diagnostic ``|actual - reference|`` before it
    reports a bitwise result (``tests/_parity_harness.py``), and ``inf - inf``
    raises the IEEE invalid flag, which numpy reports as ``RuntimeWarning:
    invalid value encountered in subtract`` and this suite turns into an error.
    So a bitwise claim cannot presently be made about an array holding an
    infinity, and the harness is where that has to be fixed rather than here.
    NaN is unaffected: quiet-NaN propagation raises no flag, so the harness's
    claim to compare NaN by bit pattern holds as written.
    """
    finfo = np.finfo(np.dtype(dtype))
    finite_specials = np.asarray(
        [
            [0.0, np.nan],
            [-0.0, float(finfo.smallest_subnormal)],
            [float(finfo.max), float(finfo.tiny)],
        ],
        dtype=dtype,
    )
    py, cpp = _both(finite_specials, is_rational=is_rational)
    assert_object_parity(
        py=py, cpp=cpp, fields=FIELDS, context=f"IEEE menagerie ({dtype}, {is_rational})"
    )
    # Asserted on the bytes rather than through the harness, because that is the
    # specific loss this case exists for: `assert_object_parity` would pass if
    # BOTH backends had flushed the subnormal or normalized the signed zero the
    # same way, and it is the transit that is under test, not the agreement.
    assert cpp.control_points.tobytes() == finite_specials.tobytes()

    with_infinities = np.asarray([[np.inf, -np.inf], [1.0, -0.0]], dtype=dtype)
    _, cpp_inf = _both(with_infinities, is_rational=is_rational)
    assert cpp_inf.control_points.tobytes() == with_infinities.tobytes()


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_non_contiguous_control_net_reaches_both_backends_alike(dtype: npt.DTypeLike) -> None:
    """A Fortran-ordered or strided array builds the same Bézier under either backend.

    What this catches: the ``np.ascontiguousarray`` step in ``_new_impl``. The C++
    binding refuses a non-contiguous array outright rather than copying it, so
    without that step the public constructor would raise a ``TypeError`` about C++
    argument types under one backend and succeed under the other. Nothing else
    exercises the branch: ``test_bezier_binding_contract.py`` tests the raw
    binding's refusal, which is the opposite path.
    """
    fortran = np.asfortranarray(np.arange(24.0, dtype=dtype).reshape(4, 3, 2))
    strided = np.arange(48.0, dtype=dtype).reshape(8, 3, 2)[::2]
    for control_points, how in ((fortran, "Fortran-ordered"), (strided, "strided")):
        assert not control_points.flags.c_contiguous, how
        py, cpp = _both(control_points, is_rational=False)
        assert_object_parity(py=py, cpp=cpp, fields=FIELDS, context=f"{how} control net ({dtype})")


def _message_of(build: Any) -> str:
    """The text of the `ValueError` that `build` raises.

    Args:
        build (Any): A no-argument call expected to raise.

    Returns:
        str: The message, or a marker saying what happened instead, so that the
        caller's assertion is the one that reports.
    """
    try:
        build()
    except ValueError as error:
        return str(error)
    return "<did not raise ValueError>"


MALFORMED: Final = (
    (lambda cls, dtype: cls(np.asarray(1.0, dtype=dtype)), "a 0-d array"),
    (lambda cls, dtype: cls(np.zeros((0, 2), dtype=dtype)), "an empty parametric direction"),
    (lambda cls, dtype: cls(np.zeros((2, 0, 3), dtype=dtype)), "an empty second direction"),
    (lambda cls, dtype: cls(np.zeros((3, 0), dtype=dtype)), "no components at all"),
    (lambda cls, dtype: cls(np.zeros((3, 1), dtype=dtype), True), "a weight and nothing else"),
    (lambda cls, dtype: cls(np.zeros((3, 0), dtype=dtype), True), "a rational net of rank -1"),
    (lambda cls, dtype: cls(np.zeros((0, 0), dtype=dtype), True), "bad in both ways at once"),
)
"""Every construction both implementations must refuse, and what makes it bad.

Three of these are the rank check's sides of zero: `rank -1` is the one that made
the C++ subtraction signed, and an unsigned one would reject the input while
reporting a number near 2^64.

**The last case is the only one bad in more than one way, and it is the one that
pins the ORDER of the checks.** Every other entry violates exactly one rule, so
each has one possible message and a reordering is invisible: measured, moving the
oracle's rank check in front of its shape loop left this file entirely green.
`(0, 0)` rational has an empty parametric direction *and* rank -1, so which
message a caller reads is decided by the order alone, and
`cpp/include/pantr/bezier/bezier.hpp` states that order as a cross-language
contract. `cpp/tests/test_bezier_type.cpp` asserts the C++ half against a
hardcoded literal; this is the half that compares it against the live oracle.
"""


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("build", "what"), MALFORMED)
def test_error_messages_agree_verbatim(build: Any, what: str, dtype: npt.DTypeLike) -> None:
    """Both implementations raise `ValueError` and say exactly the same thing.

    What this catches: a reworded message. A caller catching
    `pytest.raises(ValueError, match=...)` must not have to know which backend
    built the object, and the type of the exception alone does not carry that --
    both sides raise `ValueError` for every case here, so only the text can tell
    a reordered check from a faithful one.
    """
    oracle = _message_of(lambda: build(_BezierPython, dtype))
    ported = _message_of(lambda: build(_cpp_class(dtype), dtype))
    assert oracle == ported, f"{what}: oracle said {oracle!r}, C++ said {ported!r}"
    assert not oracle.startswith("<"), f"{what}: neither implementation refused it"


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_cpp_value_does_not_alias_the_array_it_was_built_from(dtype: npt.DTypeLike) -> None:
    """Mutating the constructor's argument afterwards does not move the Bézier.

    What this catches: the C++ constructor taking a view of the caller's buffer
    instead of copying, which would let a validated geometry change under its
    owner's feet -- the way in of FELIGN/pantr#338's defect.
    """
    control_points = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=dtype)
    with use_backend(Backend.CPP):
        bezier = Bezier(control_points)

    before = bezier.control_points.copy()
    control_points[0, 0] = 99.0
    assert np.array_equal(bezier.control_points, before)
    assert not np.shares_memory(bezier.control_points, control_points)


@pytest.mark.parametrize("dtype", DTYPES)
def test_writing_through_control_points_is_refused(dtype: npt.DTypeLike) -> None:
    """The array handed out is read-only, and the Bézier is unchanged either way.

    What this catches: the way *out* of the same defect. A writable view would let
    a caller edit a constructed geometry through the property, which is the half a
    criterion about construction alone leaves unpinned.
    """
    with use_backend(Backend.CPP):
        bezier = Bezier(np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=dtype))

    handed_out = bezier.control_points
    assert not handed_out.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        handed_out[0, 0] = 99.0
    assert bezier.control_points[0, 0] == 0.0


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_view_outlives_the_bezier_it_came_from(dtype: npt.DTypeLike) -> None:
    """The array keeps the C++ storage alive after the handle is dropped.

    What this catches: a view returned without an owner. The values would be read
    from freed memory, which is a use-after-free that usually reads back correct
    and occasionally does not -- so this asserts the values rather than merely that
    nothing crashed.
    """
    import gc  # noqa: PLC0415 -- only this test needs a deterministic collection

    with use_backend(Backend.CPP):
        handed_out = Bezier(np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=dtype)).control_points
    gc.collect()
    assert np.array_equal(handed_out, np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_rational", [False, True])
def test_a_bezier_survives_pickling_across_every_backend_pair(
    dtype: npt.DTypeLike, is_rational: bool
) -> None:
    """A pickle written under one backend loads under the other, at both dtypes.

    What this catches: a `__reduce__` that reaches the implementation rather than
    the constructor's arguments. The C++ handle is not picklable at all, and a
    payload that carried one would make `PANTR_BACKEND` a data-format switch --
    a pickle written on one machine unreadable on another. `copy.deepcopy` goes
    through `__reduce_ex__` by the same route, so it is swept over the same pairs.
    """
    control_points = _control_points((4, 3), dtype, seed=20260829)
    for writer in (Backend.PYTHON, Backend.CPP):
        with use_backend(writer):
            original = Bezier(control_points, is_rational)
            payload = pickle.dumps(original)
        for reader in (Backend.PYTHON, Backend.CPP):
            where = f"{writer.name} -> {reader.name}"
            with use_backend(reader):
                loaded = pickle.loads(payload)
                cloned = copy.deepcopy(original)
            for rebuilt, how in ((loaded, "pickle"), (cloned, "deepcopy")):
                assert rebuilt.control_points.tobytes() == control_points.tobytes(), (
                    f"{where} {how}"
                )
                assert rebuilt.dtype == np.dtype(dtype), f"{where} {how}"
                assert rebuilt.is_rational is is_rational, f"{where} {how}"
                assert rebuilt.degree == original.degree, f"{where} {how}"
                # `rank` folds the flag against the component axis, so it is the
                # one field that would catch a round trip which kept the byte
                # count and the per-direction degrees while moving which axis
                # carries the components.
                assert rebuilt.rank == original.rank, f"{where} {how}"


@pytest.mark.parametrize("dtype", DTYPES)
def test_an_unpickled_bezier_can_still_be_mutated_in_place(dtype: npt.DTypeLike) -> None:
    """The wire format does not carry one backend's read-only flag to the other.

    What this catches: `__reduce__` handing out the C++ backend's read-only view.
    numpy preserves that flag through a pickle, so a payload written under the C++
    backend would rebuild, under the Python backend, a Bézier whose stored array
    cannot be written -- and `reverse(in_place=True)` would raise on an object the
    caller built by ordinary means.
    """
    with use_backend(Backend.CPP):
        payload = pickle.dumps(Bezier(np.asarray([[0.0], [1.0], [2.0]], dtype=dtype)))
    with use_backend(Backend.PYTHON):
        rebuilt = pickle.loads(payload)
        rebuilt.reverse(in_place=True)
    assert np.array_equal(rebuilt.control_points, np.asarray([[2.0], [1.0], [0.0]], dtype=dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_rational", [False, True])
def test_the_in_place_mutators_agree_on_the_value_they_leave_behind(
    dtype: npt.DTypeLike, is_rational: bool
) -> None:
    """`reverse`, `permute_directions` and `transform` agree under both backends.

    What this catches: the C++ path's rebuild-the-implementation strategy losing
    the rationality flag, the dtype or a permuted stride. Only the array's
    *identity* is allowed to differ -- the C++ value owns its storage, so an
    in-place mutation replaces it -- and `tests/test_transform.py` pins that
    identity for the Python backend alone.
    """
    from pantr.transform import AffineTransform  # noqa: PLC0415

    control_points = _control_points((3, 4, 3), dtype, seed=20260830)
    shift = AffineTransform.translation([1.0, 2.0] if is_rational else [1.0, 2.0, 3.0])

    def mutated(backend: Backend) -> Bezier:
        # A fresh copy per backend, and not a convenience: the Python
        # implementation aliases what it is given, so an in-place mutation under
        # that backend edits `control_points` itself and the C++ run would then
        # start from an already-reversed net. Sharing one array here made this
        # test fail on every element while both backends were in fact correct.
        with use_backend(backend):
            bezier = Bezier(control_points.copy(), is_rational)
            bezier.reverse(1, in_place=True)
            bezier.permute_directions([1, 0], in_place=True)
            bezier.transform(shift, in_place=True)
            return bezier

    assert_object_parity(
        py=mutated(Backend.PYTHON),
        cpp=mutated(Backend.CPP),
        fields=FIELDS,
        context=f"in-place reverse, permute and transform ({dtype}, rational={is_rational})",
    )


@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.CPP])
def test_an_unsupported_dtype_is_refused_the_same_way_under_both_backends(
    backend: Backend,
) -> None:
    """A dtype no kernel can evaluate is a `TypeError`, from the wrapper.

    What this catches: the restriction living on the C++ side only, which would
    make `PANTR_BACKEND` decide whether a `float16` control net is accepted. The
    check is a type-kind check, so it belongs in the wrapper: nanobind's default
    translator has no path that produces a `TypeError`.
    """
    with use_backend(backend), pytest.raises(TypeError, match="float32, float64"):
        Bezier(np.zeros((3, 2), dtype=np.float16))


@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.CPP])
def test_integer_control_points_are_cast_to_float64_under_both_backends(
    backend: Backend,
) -> None:
    """The one documented dtype conversion happens on both sides.

    What this catches: the cast being skipped under the C++ backend, where the
    `.noconvert()` on the binding would then refuse an integer array outright and
    turn a documented convenience into a `TypeError`.
    """
    with use_backend(backend):
        bezier = Bezier(np.asarray([[0, 0], [1, 1], [2, 0]], dtype=np.int64))
    assert bezier.dtype == np.float64
    assert bezier.degree == (2,)


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_wrapper_holds_the_implementation_its_backend_selects(dtype: npt.DTypeLike) -> None:
    """The dtype picks the class, and the backend picks the family.

    What this catches: `_impl_class` ignoring its dtype argument. Every other
    assertion in this file would still pass if it always returned `Bezier64`,
    because nothing else here can see which class was chosen -- the `.noconvert()`
    on the binding turns the mistake into a refusal for one direction only.
    """
    control_points = np.zeros((3, 2), dtype=dtype)
    with use_backend(Backend.PYTHON):
        assert isinstance(Bezier(control_points)._impl, _BezierPython)
    with use_backend(Backend.CPP):
        assert isinstance(Bezier(control_points)._impl, _cpp_class(dtype))


def test_mutating_under_a_switched_backend_is_refused_rather_than_converted() -> None:
    """An in-place mutator refuses a backend that is not the one this Bézier was built under.

    Rebuilding the implementation reads the *active* backend, so without this
    check a Bézier built under C++ and reversed inside a
    ``use_backend(Backend.PYTHON)`` block came back as ``_BezierPython`` -- and
    the caller's ``control_points`` went from read-only to writeable underneath
    them, on an array they still held. Converting between two implementations of
    one type is the shape ``design/cross_backend_types.md`` forbids;
    :meth:`pantr.geometry.AABB._peer` already refuses it for a binary operation,
    and this is the same rule for mutation.

    ``Bezier`` is the first ported type with observable mutation, so it is the
    first that could reach this at all: an immutable type only ever produces a
    *derived* object under the other backend, never reconciles a live one.

    Pins that the refusal leaves the object untouched, and that a mutation under
    the matching backend still works -- a check that merely raises everywhere
    would pass the first half.
    """
    control_points = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)

    with use_backend(Backend.CPP):
        bezier = Bezier(control_points, is_rational=False)
    assert not bezier.control_points.flags.writeable

    with use_backend(Backend.PYTHON), pytest.raises(TypeError, match="different backend"):
        bezier.reverse(in_place=True)

    assert isinstance(bezier._impl, _cpp_class(np.dtype(np.float64)))
    assert not bezier.control_points.flags.writeable
    assert bezier.control_points.ravel().tolist() == [0.0, 1.0, 2.0]

    with use_backend(Backend.CPP):
        bezier.reverse(in_place=True)
    assert bezier.control_points.ravel().tolist() == [2.0, 1.0, 0.0]
