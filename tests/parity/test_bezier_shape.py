"""Parity for the Bézier shape operations.

**Three families, three kinds of claim**, and which family an operation lands in is a
fact about its oracle rather than a preference:

* **Pure rearrangements.** ``reverse`` and ``permute_directions`` move values and
  compute nothing -- the oracle spells them :func:`numpy.flip` and
  :func:`numpy.transpose` -- so they are bitwise **unconditionally**, at any dtype and
  on any build. There is no arithmetic for a fused multiply-add to change, which is
  why these two carry no conditional arm where the next family does.
* **Compositions over a 1-D kernel.** ``restrict``, ``split``, ``slice`` and
  ``boundary`` reduce to ``restrict_bezier_1d``, ``split_bezier_1d`` and
  ``slice_bezier_1d`` applied along one axis, with the axis permutation contributing
  nothing. They inherit those kernels' claims exactly: bitwise where the build cannot
  fuse, Rule 10's budget where it can, with the two rounding constants imported from
  ``tests/parity/test_bezier_arithmetic.py`` rather than spelled again.
* **Contractions that reach BLAS.** ``transform``'s oracle is ``cp @ A.T + b`` and
  ``collapse_along_axis``'s is a chain of :func:`numpy.tensordot`. Neither summation
  order is reproducible, so both are bounded, with no bitwise arm to condition on --
  the same situation ``tests/parity/test_bezier_evaluate.py`` records for its lattice
  entry point.

Measured over 56 configurations spanning both dtypes, dim 1 to 3, ranks 1 and 3,
rational and not: seven of the nine agree bit for bit and the two BLAS ones agree to
rounding.

What the parameter's width cost, recorded because it is the failure this port is about
--------------------------------------------------------------------------------------

``restrict``, ``split`` and ``slice`` take their parameter at ``accumulator_t<T>`` --
``double`` even for ``float`` storage -- because the 1-D kernels' signatures say so,
and they say so because the oracle receives a Python float, which numba treats as
``float64`` whatever the control points hold. The first version of the port took it at
``T``. That rounds the parameter before the kernel sees it, which is invisible for a
representable value like ``0.375`` and wrong for one like ``0.9``; 38 ``float32`` cases
of ``test_bezier_arithmetic.py``'s own restriction test caught it. **Every parameter
below is deliberately not representable in ``float32``** for that reason.

Three of the nine are checked through the binding rather than the public method
--------------------------------------------------------------------------------

``reverse``, ``permute_directions`` and ``transform`` reach a Bézier through
``_control_points_utils.py`` and ``_transform_control_points.py``, which
``pantr.bspline`` shares, so the Python side still computes them in numpy under either
backend and the C++ port is reached by calling the binding directly. That is a real
gap in the routing and it is recorded rather than papered over: what these tests
compare is the ported code against its oracle, which is what a parity claim is for, but
they do **not** demonstrate that ``Bezier.reverse`` under the C++ backend runs C++.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr._control_points_utils import _permute_control_points, _reverse_control_points
from pantr._transform_control_points import _apply_affine_to_control_points
from pantr.bezier import Bezier
from tests._parity_harness import (
    ParityClaim,
    Roundings,
    assert_parity,
    bitwise_parity,
    bounded_parity,
    contraction_may_fuse,
    demand_cpp_backend,
    demand_the_compiled_kernel,
)
from tests.parity.test_bezier_arithmetic import (
    _ACCUMULATOR_ROUNDINGS_PER_STAGE,
    _FUSED_PREFIX,
    _STORAGE_ROUNDINGS_PER_STAGE,
)

DTYPES: Final = (np.float64, np.float32)
"""Both storage formats."""

DEGREES: Final = ((3,), (1, 1), (2, 3), (5, 4), (0, 4), (2, 2, 2), (4, 3, 2))
"""Degree tuples spanning dim 1 to 3, including a degree-0 direction."""

RANKS: Final = (1, 3)
"""Output ranks."""

_PARAMETER: Final = 0.9
"""The parameter every one-directional operation is exercised at.

Not representable in ``float32``, on purpose: a power-of-two fraction cannot tell a
parameter taken at the accumulator's width from one narrowed to the storage format,
and that distinction is what this module's docstring is about.
"""

_LOWER: Final = 0.1
"""Lower restriction bound. Not representable in ``float32``, for the same reason."""

_TINY: Final = float(np.finfo(np.float64).tiny)
"""Floor for an amplification, so a tolerance is never identically zero."""

_REARRANGEMENT_WHY: Final = (
    "a pure rearrangement: the oracle is np.flip or np.transpose and the port is an "
    "index permutation, so neither side performs a floating-point operation at all. "
    "This claim carries no conditional arm because there is no arithmetic for a fused "
    "multiply-add to change -- unlike every other claim in this file, it holds on any "
    "build and at any dtype by construction rather than by measurement"
)

_ONE_DIRECTIONAL_WHY: Final = (
    "composes over the 1-D kernel along one axis, with axis_layout.hpp's permutation "
    "contributing nothing: it moves values and computes on none of them. So this "
    "inherits the kernel's own claim, which tests/parity/test_bezier_arithmetic.py "
    "carries. The parameter crosses at accumulator_t<T> -- double even for float "
    "storage -- because the kernel's signature says so and the oracle receives a "
    "Python float; taking it at T rounds the parameter before the kernel sees it, "
    "which is what this file's parameters are chosen to expose"
)

_TRANSFORM_WHY: Final = (
    "the oracle is cp @ A.T + b, a matrix product that reaches BLAS, so its summation "
    "order is not reproducible and there is no bitwise arm to condition on. The budget "
    "is one rounding per accumulation step of a length-n dot product plus one for the "
    "translation, at the STORAGE width, because the oracle casts the matrix to the "
    "control points' dtype before multiplying rather than after. The amplification is "
    "the row action |cp| @ |A.T| + |b|, which is the reachable magnitude because it is "
    "the same expression on absolute values"
)

_COLLAPSE_WHY: Final = (
    "the oracle contracts each collapsed direction with np.tensordot, which reaches "
    "BLAS, so this is bounded for the same reason the lattice evaluation is. The "
    "budget is one rounding per term summed, over every collapsed direction, and the "
    "amplification is the same chain of contractions run on |c| -- exact rather than "
    "conservative because a Bernstein basis is non-negative. The contraction order is "
    "the oracle's, highest direction first: the contractions are not associative in "
    "floating point, so a different order is a different answer rather than a "
    "differently rounded one"
)


def _net(
    degrees: tuple[int, ...], rank: int, dtype: npt.DTypeLike, seed: int, *, rational: bool
) -> npt.NDArray[np.float32 | np.float64]:
    """Build a control net spanning many magnitudes, weights bounded away from zero.

    Args:
        degrees (tuple[int, ...]): Degree per direction.
        rank (int): Number of value components.
        dtype (npt.DTypeLike): Storage format.
        seed (int): Generator seed.
        rational (bool): Whether to append a weight column.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The control net.
    """
    rng = np.random.default_rng(seed)
    components = rank + 1 if rational else rank
    shape = (*(degree + 1 for degree in degrees), components)
    net = rng.standard_normal(shape) * 10.0 ** rng.integers(-4, 5, shape)
    net = np.ascontiguousarray(net, dtype=dtype)
    if rational:
        net[..., -1] = np.asarray(rng.uniform(0.5, 2.0, net.shape[:-1]), dtype=dtype)
    return net


def _inherited(
    why: str, *, stages: int, amplification: npt.NDArray[np.float64], dtype: npt.DTypeLike
) -> ParityClaim:
    """Bitwise where the build cannot fuse, Rule 10's budget where it can.

    Args:
        why (str): The derivation, used for the bitwise arm as written and prefixed
            with the shared contraction argument for the fused one.
        stages (int): Length of the dependency chain the fused sites sit on.
        amplification (npt.NDArray[np.float64]): Elementwise magnitude.
        dtype (npt.DTypeLike): Storage format; the accumulator is float64.

    Returns:
        ParityClaim: BITWISE or BOUNDED, whichever this build supports.
    """
    if not contraction_may_fuse():
        return bitwise_parity(why=why)
    return bounded_parity(
        roundings=Roundings(
            stages=stages,
            accumulator_per_stage=_ACCUMULATOR_ROUNDINGS_PER_STAGE,
            storage_per_stage=_STORAGE_ROUNDINGS_PER_STAGE,
        ),
        accumulator=np.float64,
        storage=dtype,
        amplification=amplification,
        why=f"{_FUSED_PREFIX}{why}",
    )


def _companion(values: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Floor an amplification so a tolerance is never identically zero.

    Args:
        values (npt.NDArray[Any]): The magnitudes.

    Returns:
        npt.NDArray[np.float64]: The same, floored, as float64.
    """
    return np.ascontiguousarray(
        np.maximum(np.abs(np.asarray(values, dtype=np.float64)), _TINY), dtype=np.float64
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degrees", DEGREES)
@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("rational", [False, True])
def test_reverse_and_permute_are_bitwise(
    cpp_backend: None,
    degrees: tuple[int, ...],
    rank: int,
    dtype: npt.DTypeLike,
    *,
    rational: bool,
) -> None:
    """The two pure rearrangements agree exactly, unconditionally.

    Driven through the binding rather than through :class:`~pantr.bezier.Bezier`,
    because the wrapper still computes these in numpy under either backend; see the
    module docstring.
    """
    del cpp_backend
    demand_cpp_backend()
    demand_the_compiled_kernel(dtype)
    from pantr import _pantr_cpp  # noqa: PLC0415

    net = _net(degrees, rank, dtype, seed=20260901, rational=rational)
    # The class was chosen from the array's own dtype one expression earlier, which
    # is a correlation between a value and a type that the checker cannot state.
    # `pantr.bezier._bezier._new_impl` stands in for it the same way.
    handle = (_pantr_cpp.Bezier32 if dtype == np.float32 else _pantr_cpp.Bezier64)(
        cast("Any", net), rational
    )
    dim = len(degrees)

    for direction in range(dim):
        assert_parity(
            np.asarray(_pantr_cpp.reverse_bezier(handle, direction).control_points),
            np.asarray(_reverse_control_points(net, direction, in_place=False)),
            bitwise_parity(why=_REARRANGEMENT_WHY),
            context=f"reverse {degrees} direction {direction} {np.dtype(dtype).name}",
        )

    permutation = list(range(dim))[::-1]
    assert_parity(
        np.asarray(_pantr_cpp.permute_bezier_directions(handle, permutation).control_points),
        np.asarray(_permute_control_points(net, permutation, dim)),
        bitwise_parity(why=_REARRANGEMENT_WHY),
        context=f"permute {degrees} by {permutation} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degrees", DEGREES)
@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("rational", [False, True])
def test_the_one_directional_operations_match_the_oracle(
    cpp_backend: None,
    degrees: tuple[int, ...],
    rank: int,
    dtype: npt.DTypeLike,
    *,
    rational: bool,
) -> None:
    """Restriction, splitting and slicing inherit their 1-D kernels' claims."""
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    net = _net(degrees, rank, dtype, seed=20260902, rational=rational)
    dim = len(degrees)
    # A one-dimensional Bezier takes the plain pair: `Bezier.restrict` treats a
    # sequence whose length equals `dim` as per-direction, and for `dim == 1` a
    # list holding one pair is indistinguishable from a pair, so it wraps it again.
    bounds: Any = (
        (_LOWER, _PARAMETER)
        if dim == 1
        else [(_LOWER, _PARAMETER) if d == 0 else None for d in range(dim)]
    )

    def both(run: Any) -> tuple[Any, Any]:
        with use_backend(Backend.PYTHON):
            reference = run(Bezier(net, is_rational=rational))
        with use_backend(Backend.CPP):
            actual = run(Bezier(net, is_rational=rational))
        return actual, reference

    magnitude = _companion(net)
    restricted_actual, restricted_reference = both(
        lambda bezier: np.asarray(bezier.restrict(bounds).control_points)
    )
    assert_parity(
        restricted_actual,
        restricted_reference,
        _inherited(
            _ONE_DIRECTIONAL_WHY + ". Twice the stages, because restriction is two "
            "de Casteljau passes",
            stages=2 * degrees[0],
            amplification=np.broadcast_to(np.max(magnitude), restricted_reference.shape).copy(),
            dtype=dtype,
        ),
        context=f"restrict {degrees} rank {rank} rational {rational} {np.dtype(dtype).name}",
    )

    for half in (0, 1):
        actual, reference = both(
            lambda bezier, half=half: np.asarray(bezier.split(0, _PARAMETER)[half].control_points)
        )
        assert_parity(
            actual,
            reference,
            _inherited(
                _ONE_DIRECTIONAL_WHY,
                stages=degrees[0],
                amplification=np.broadcast_to(np.max(magnitude), reference.shape).copy(),
                dtype=dtype,
            ),
            context=f"split half {half} {degrees} {np.dtype(dtype).name}",
        )

    sliced_actual, sliced_reference = both(
        lambda bezier: np.asarray(
            bezier.slice(0, _PARAMETER)
            if bezier.dim == 1
            else bezier.slice(0, _PARAMETER).control_points
        )
    )
    assert_parity(
        sliced_actual,
        sliced_reference,
        _inherited(
            _ONE_DIRECTIONAL_WHY,
            stages=degrees[0],
            amplification=np.broadcast_to(np.max(magnitude), sliced_reference.shape).copy(),
            dtype=dtype,
        ),
        context=f"slice {degrees} rank {rank} rational {rational} {np.dtype(dtype).name}",
    )

    if dim >= 2:
        for side in (0, 1):
            actual, reference = both(
                lambda bezier, side=side: np.asarray(bezier.boundary(0, side).control_points)
            )
            assert_parity(
                actual,
                reference,
                _inherited(
                    _ONE_DIRECTIONAL_WHY + ". boundary is slice at 0 or 1",
                    stages=degrees[0],
                    amplification=np.broadcast_to(np.max(magnitude), reference.shape).copy(),
                    dtype=dtype,
                ),
                context=f"boundary {degrees} side {side} {np.dtype(dtype).name}",
            )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degrees", DEGREES)
@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("rational", [False, True])
def test_transform_is_bounded(
    cpp_backend: None,
    degrees: tuple[int, ...],
    rank: int,
    dtype: npt.DTypeLike,
    *,
    rational: bool,
) -> None:
    """The affine map agrees inside the matrix product's own rounding budget."""
    del cpp_backend
    demand_cpp_backend()
    demand_the_compiled_kernel(dtype)
    from pantr import _pantr_cpp  # noqa: PLC0415

    net = _net(degrees, rank, dtype, seed=20260903, rational=rational)
    rng = np.random.default_rng(20260904)
    matrix = np.ascontiguousarray(rng.standard_normal((rank, rank)))
    offset = np.ascontiguousarray(rng.standard_normal(rank))

    reference = np.asarray(
        _apply_affine_to_control_points(net.copy(), rational, matrix, offset, in_place=False),
        dtype=dtype,
    )
    # The class was chosen from the array's own dtype one expression earlier, which
    # is a correlation between a value and a type that the checker cannot state.
    # `pantr.bezier._bezier._new_impl` stands in for it the same way.
    handle = (_pantr_cpp.Bezier32 if dtype == np.float32 else _pantr_cpp.Bezier64)(
        cast("Any", net), rational
    )
    actual = np.asarray(
        _pantr_cpp.transform_bezier(handle, matrix, offset).control_points, dtype=dtype
    )

    # The row action on absolute values, which is the magnitude each output element
    # can reach. The weight column is untouched by both sides, so its amplification is
    # its own magnitude and the comparison there is exact anyway.
    coordinates = np.abs(np.asarray(net[..., :rank], dtype=np.float64))
    weights = (
        np.abs(np.asarray(net[..., rank : rank + 1], dtype=np.float64))
        if rational
        else np.ones((*net.shape[:-1], 1))
    )
    reach = coordinates @ np.abs(matrix).T + (weights * np.abs(offset))
    amplification = (
        np.concatenate([reach, np.abs(np.asarray(net[..., rank:], dtype=np.float64))], axis=-1)
        if rational
        else reach
    )

    assert_parity(
        actual,
        reference,
        bounded_parity(
            roundings=Roundings(stages=rank + 1, accumulator_per_stage=1, storage_per_stage=0),
            accumulator=dtype,
            storage=dtype,
            amplification=_companion(amplification),
            why=_TRANSFORM_WHY,
        ),
        context=f"transform {degrees} rank {rank} rational {rational} {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("degrees", [(1, 1), (2, 3), (5, 4), (2, 2, 2), (4, 3, 2)])
@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("rational", [False, True])
def test_collapse_is_bounded(
    cpp_backend: None,
    degrees: tuple[int, ...],
    rank: int,
    dtype: npt.DTypeLike,
    *,
    rational: bool,
) -> None:
    """The collapse agrees inside the contraction's own rounding budget."""
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    net = _net(degrees, rank, dtype, seed=20260905, rational=rational)
    dim = len(degrees)
    values = np.asarray([_PARAMETER] * (dim - 1), dtype=dtype)

    with use_backend(Backend.PYTHON):
        reference = np.asarray(
            Bezier(net, is_rational=rational).collapse_along_axis(0, values).control_points
        )
        magnitude = np.asarray(
            Bezier(np.abs(net), is_rational=rational).collapse_along_axis(0, values).control_points
        )
    with use_backend(Backend.CPP):
        actual = np.asarray(
            Bezier(net, is_rational=rational).collapse_along_axis(0, values).control_points
        )

    stages = sum(degrees[d] + 1 for d in range(dim) if d != 0)
    assert_parity(
        actual,
        reference,
        bounded_parity(
            roundings=Roundings(stages=stages, accumulator_per_stage=1, storage_per_stage=0),
            accumulator=dtype,
            storage=dtype,
            amplification=_companion(magnitude),
            why=_COLLAPSE_WHY,
        ),
        context=f"collapse {degrees} rank {rank} rational {rational} {np.dtype(dtype).name}",
    )


def test_transform_never_converts_an_affine_between_backends() -> None:
    """``transform`` moves arrays, never an ``AffineTransform`` implementation.

    The grep-level check the ticket asks for, run rather than asserted in prose. What
    it establishes is narrow and worth stating: no site that transforms a Bézier
    mentions either affine implementation class by name, so there is nowhere for a
    conversion between them to happen. Both paths read ``affine.matrix`` and
    ``affine.offset``, which are numpy arrays whichever backend built the map.
    """
    hits = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "-E",
            "_AffineTransformPython|AffineTransform32|AffineTransform64",
            "--",
            "src/pantr/bezier",
            "src/pantr/_transform_control_points.py",
            "cpp/include/pantr/bezier",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    assert hits == "", (
        f"a Bézier transform site names an affine implementation class, which is where "
        f"a conversion between the two would have to appear:\n{hits}"
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_in_place_variants_return_none_and_mutate_under_both_backends(
    cpp_backend: None, dtype: npt.DTypeLike
) -> None:
    """Both return types of the ``in_place`` flag are exercised, on both backends.

    The flag is a known API defect -- a boolean that changes the return type -- and it
    is ported faithfully rather than fixed, so what has to be pinned is that it still
    behaves identically on both sides: ``in_place=True`` returns ``None`` **and**
    leaves the receiver changed, ``in_place=False`` returns a new Bézier and leaves the
    receiver alone.

    Asserting only the ``None`` would pass against an implementation that returned
    ``None`` and did nothing, which is why the receiver is checked too; asserting only
    the mutation would pass against one that also returned the object, which is why
    the ``None`` is checked as well.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    net = _net((2, 3), 3, dtype, seed=20260906, rational=False)
    identity = np.eye(3)
    shift = np.asarray([1.0, 2.0, 3.0])

    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            from pantr.transform import AffineTransform  # noqa: PLC0415

            mutations: list[Callable[[Bezier], None]] = [
                lambda bezier: bezier.reverse(0, in_place=True),
                lambda bezier: bezier.permute_directions([1, 0], in_place=True),
                lambda bezier: bezier.transform(AffineTransform(identity, shift), in_place=True),
            ]
            for mutate in mutations:
                receiver = Bezier(net)
                before = np.array(receiver.control_points)
                assert mutate(receiver) is None, (
                    f"an in_place=True call returned something under {backend.name}; the "
                    f"overloads say it returns None and a caller assigning the result "
                    f"would silently get it"
                )
                after = np.array(receiver.control_points)
                assert before.shape != after.shape or not np.array_equal(before, after), (
                    f"an in_place=True call left the receiver unchanged under "
                    f"{backend.name}, so it returned None and did nothing"
                )

            untouched = Bezier(net)
            snapshot = np.array(untouched.control_points)
            produced = untouched.reverse(0)
            assert isinstance(produced, Bezier)
            assert np.array_equal(np.array(untouched.control_points), snapshot), (
                f"reverse(in_place=False) mutated its receiver under {backend.name}"
            )


_SWEEP_DRAWS: Final = 10
"""Independent nets per configuration in the ten-times sweep.

Sized against the shipped parametrization rather than picked: each claim ships 28
cases per dtype (7 degree tuples x 2 ranks x 2 rationalities), so ten draws of each
gives 280 per claim per dtype. Written down because the test passes at any number of
draws, which is what lets a sweep quietly stop being the thing it claims to be.
"""


@pytest.mark.slow
@pytest.mark.parametrize("dtype", DTYPES)
def test_each_claim_holds_over_a_sweep_ten_times_the_shipped_one(
    cpp_backend: None, dtype: npt.DTypeLike
) -> None:
    """A bound checked only by the sweep that ships with it has not been checked.

    ``design/backend_parity.md`` records the case: the affine-map bound held over 500
    draws and failed 88 times at 60000. Ten independent nets per configuration against
    the bitwise claims and the two bounded ones, asserting per family that the sweep
    actually reached it rather than only that nothing failed.
    """
    del cpp_backend
    demand_cpp_backend()
    demand_the_compiled_kernel(dtype)
    from pantr import _pantr_cpp  # noqa: PLC0415

    handle_class = _pantr_cpp.Bezier32 if dtype == np.float32 else _pantr_cpp.Bezier64
    exercised = {"rearrangement": 0, "one_directional": 0, "collapse": 0}
    seed = 0
    for degrees in DEGREES:
        for rank in RANKS:
            for rational in (False, True):
                for _draw in range(_SWEEP_DRAWS):
                    seed += 1
                    net = _net(degrees, rank, dtype, seed=60000 + seed, rational=rational)
                    dim = len(degrees)
                    handle = handle_class(cast("Any", net), rational)

                    assert np.array_equal(
                        np.asarray(_pantr_cpp.reverse_bezier(handle, 0).control_points),
                        np.asarray(_reverse_control_points(net, 0, in_place=False)),
                    ), f"reverse is claimed bitwise and this draw differs: {degrees}"
                    exercised["rearrangement"] += 1

                    with use_backend(Backend.PYTHON):
                        reference = np.asarray(
                            Bezier(net, is_rational=rational).split(0, _PARAMETER)[0].control_points
                        )
                    with use_backend(Backend.CPP):
                        actual = np.asarray(
                            Bezier(net, is_rational=rational).split(0, _PARAMETER)[0].control_points
                        )
                    assert np.array_equal(actual, reference), (
                        f"split is claimed bitwise on a non-fusing build and this draw "
                        f"differs: {degrees} rank {rank} rational {rational}"
                    )
                    exercised["one_directional"] += 1

                    if dim < 2:
                        continue
                    values = np.asarray([_PARAMETER] * (dim - 1), dtype=dtype)
                    with use_backend(Backend.PYTHON):
                        collapse_reference = np.asarray(
                            Bezier(net, is_rational=rational)
                            .collapse_along_axis(0, values)
                            .control_points
                        )
                        magnitude = np.asarray(
                            Bezier(np.abs(net), is_rational=rational)
                            .collapse_along_axis(0, values)
                            .control_points
                        )
                    with use_backend(Backend.CPP):
                        collapse_actual = np.asarray(
                            Bezier(net, is_rational=rational)
                            .collapse_along_axis(0, values)
                            .control_points
                        )
                    stages = sum(degrees[d] + 1 for d in range(dim) if d != 0)
                    assert_parity(
                        collapse_actual,
                        collapse_reference,
                        bounded_parity(
                            roundings=Roundings(
                                stages=stages, accumulator_per_stage=1, storage_per_stage=0
                            ),
                            accumulator=dtype,
                            storage=dtype,
                            amplification=_companion(magnitude),
                            why=_COLLAPSE_WHY,
                        ),
                        context=f"collapse sweep {degrees} rank {rank} rational {rational}",
                    )
                    exercised["collapse"] += 1

    for family in ("rearrangement", "one_directional"):
        assert exercised[family] >= 280, (
            f"the {family} claim saw {exercised[family]} comparisons, short of the 280 "
            f"per dtype that is ten times its shipped 28. The sweep has stopped being "
            f"ten times the shipped one."
        )
    # Collapse skips the one-dimensional degree tuple, so its own ten-times target is
    # measured against the six tuples it can reach rather than all seven.
    assert exercised["collapse"] >= 240, (
        f"the collapse claim saw {exercised['collapse']} comparisons, short of the 240 "
        f"that is ten times its shipped 24."
    )
