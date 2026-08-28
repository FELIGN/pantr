"""Self-tests of the parity harness in `tests/_parity_harness.py`.

Kernel-agnostic checks that the harness's own machinery does what it claims:
a claim cannot be built without a derivation, a bound that exhausts the format
or dwarfs the values it compares is refused rather than passing vacuously, and
the vacuity guard leaves a legitimate absolute floor alone.

The sensitivity probes for the claim kinds themselves --
`test_bitwise_claim_detects_a_one_ulp_difference` and the two
`test_bounded_branch_*` tests -- stay in `tests/parity/test_basis_cardinal_bspline.py`
instead: they drive the probe through `_tabulate` and the cardinal B-spline's own
point sets, which makes them per-kernel rather than harness-only.

The tests below `# --- assert_object_parity ---` cover the object-level entry
point instead. Unlike the array probes, that entry point has no per-kernel home
of its own -- it is exercised by eight not-yet-written consumers -- so its own
sensitivity probes, its two vacuity guards, and the ``read``-callable and
argument-order behaviour it is built on live here. Every test double is a
throwaway `types.SimpleNamespace` or a two-line local class, never a real
`Partition`/`BVH`/etc. and never the compiled extension: what is under test is
the harness, not any kernel it will later be pointed at.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests._parity_harness import (
    Field,
    Roundings,
    absolute_tolerance,
    assert_object_parity,
    assert_parity,
    bitwise_parity,
    bounded_parity,
    demand_a_compiled_seed,
    demand_the_compiled_kernel,
    exact_parity,
    the_jit_is_disabled,
)


def test_a_tolerance_cannot_be_stated_without_a_derivation() -> None:
    """Every way of building a claim without a derivation is refused.

    The harness's whole purpose is that the next five ported modules cannot express
    an underived tolerance. That property is a property of the API, so it is tested
    like any other.
    """
    with pytest.raises(ValueError, match="why"):
        bitwise_parity(why="   ")
    with pytest.raises(ValueError, match="derivation"):
        bounded_parity(
            roundings=Roundings(stages=3, accumulator_per_stage=2, storage_per_stage=0),
            accumulator=np.float64,
            storage=np.float64,
            amplification=np.ones((2, 2)),
            why="",
        )
    with pytest.raises(ValueError, match="bitwise_parity"):
        bounded_parity(
            roundings=Roundings(stages=0, accumulator_per_stage=2, storage_per_stage=0),
            accumulator=np.float64,
            storage=np.float64,
            amplification=np.ones((2, 2)),
            why="zero stages is not a bound",
        )
    with pytest.raises(ValueError, match="finite"):
        bounded_parity(
            roundings=Roundings(stages=3, accumulator_per_stage=2, storage_per_stage=0),
            accumulator=np.float64,
            storage=np.float64,
            amplification=np.array([[np.inf]]),
            why="an overflowed amplification makes the comparison vacuous",
        )


def test_a_budget_that_exhausts_the_format_is_refused() -> None:
    """A bound that accepts every finite result is reported, not returned.

    The failure mode a "derived" tolerance still permits: derive it for a degree so
    large, or a format so narrow, that the bound exceeds 1 in relative terms and the
    comparison stops meaning anything. That is worth an error rather than a pass.
    """
    claim = bounded_parity(
        roundings=Roundings(stages=10**8, accumulator_per_stage=2, storage_per_stage=1),
        accumulator=np.float64,
        storage=np.float32,
        amplification=np.ones((2, 2)),
        why="probe: a budget large enough to exhaust float32",
    )
    with pytest.raises(ValueError, match="vacuous"):
        absolute_tolerance(claim)


def test_a_bound_larger_than_the_values_it_compares_is_refused() -> None:
    """A finite but enormous amplification cannot buy a passing comparison.

    What it catches: an amplification that is finite and non-negative, so
    ``bounded_parity`` accepts it, and large enough that no result could violate
    the bound it produces. The assertion then passes for ever and reports
    agreement that was never measured, which is worse than a failure because
    nothing points at it.

    The amplification used here is not invented. It is what this harness's own
    docstring prescribed until it was corrected: the absolute-value companion of
    the kernel's recurrence, applied to the Legendre three-term recurrence, whose
    two homogeneous solutions are bounded on ``[-1, 1]`` while their absolute-value
    companion grows like ``(1 + sqrt(2))**k``. Measured at degree 700 it reaches
    ``1.7e266``, and the tolerance that follows is ``5.3e253``.
    """
    claim = bounded_parity(
        roundings=Roundings(stages=700, accumulator_per_stage=2, storage_per_stage=0),
        accumulator=np.float64,
        storage=np.float64,
        amplification=np.array([1.7e266, 1.0]),
        why="the absolute-value companion of an oscillatory recurrence, which is not a bound",
    )
    with pytest.raises(AssertionError, match="vacuous"):
        assert_parity(
            np.array([1.0, 0.0]),
            np.array([-1e250, 0.0]),
            claim,
            context="a bound larger than the values it compares",
        )


def test_the_vacuity_guard_leaves_an_absolute_floor_alone() -> None:
    """A legitimate bound on a value that is genuinely near zero still passes.

    The guard compares against the array's largest magnitude rather than each
    element's own, and this is the case that forces that choice: an element whose
    true value is zero is compared under the underflow floor, so its own tolerance
    exceeds its own magnitude by any factor you like. Checking per element would
    reject exactly the case :func:`underflow_floor` exists to serve, which is most
    of a B-spline row.
    """
    claim = bounded_parity(
        roundings=Roundings(stages=4, accumulator_per_stage=2, storage_per_stage=1),
        accumulator=np.float64,
        storage=np.float64,
        amplification=np.array([1.0, 0.0]),
        why="an ordinary claim over a row whose second entry is exactly zero",
    )
    deviation = assert_parity(
        np.array([0.5, 0.0]),
        np.array([0.5, 0.0]),
        claim,
        context="an absolute floor on an exactly zero entry",
    )
    assert deviation.num_differing == 0


def test_a_single_rounding_per_stage_gives_a_bound_that_is_not_zero() -> None:
    """A budget of one float64 rounding per stage must not collapse onto the floor.

    The regression test for a defect the shipped harness carried: the relative
    growth was computed as ``(1 + per_stage)**stages - 1``, and in float64 that is
    **exactly zero** whenever ``per_stage`` is one unit of roundoff. ``1 + eps/2``
    lands on the midpoint between ``1`` and ``1 + eps``, and round-half-to-even
    carries it back to ``1`` because ``1``'s significand is even, so the power is
    ``1.0`` at every stage count and the subtraction gives ``0.0``.

    The consequence was a claim saying BOUNDED while asserting bit-for-bit
    agreement, since all that survived was the underflow floor at about
    ``1e-323``. The existing vacuity guard did not catch it because it tests
    ``per_stage == 0``, which is ``1.11e-16`` here and passes: a budget can be
    non-zero and still produce a zero bound.

    The exact triggering data is ``accumulator_per_stage=1`` with the storage
    format equal to the accumulator, so the narrowing term is zero and the whole
    per-stage budget is a single ``u``. Both stage counts below returned ``0.0``
    before the fix.
    """
    floor_scale = 1e-300
    previous = 0.0

    for stages in (1, 4, 20):
        claim = bounded_parity(
            roundings=Roundings(stages=stages, accumulator_per_stage=1, storage_per_stage=0),
            accumulator=np.float64,
            storage=np.float64,
            amplification=np.ones(1),
            why="one float64 rounding per stage and no narrowing store, the case that "
            "made the power form evaluate to exactly zero",
        )
        tolerance = float(absolute_tolerance(claim)[0])

        assert tolerance > floor_scale, (
            f"at {stages} stages the tolerance is {tolerance:.3g}, down at the underflow "
            f"floor rather than at the rounding scale; the relative term evaluated to "
            f"zero and this BOUNDED claim is asserting bit-for-bit agreement"
        )
        assert tolerance > previous, (
            f"the tolerance did not grow from {previous:.3g} to {tolerance:.3g} when the "
            f"stage count reached {stages}; a bound that ignores its own stage count is "
            f"not accumulating anything"
        )
        previous = tolerance


def test_a_budget_that_reaches_the_runaway_half_is_refused() -> None:
    """Gamma stops bounding anything once the accumulated budget reaches one half.

    ``gamma_m = m u / (1 - m u)`` is a bound only while ``m u`` is small; at one
    half it equals 1 and past it the denominator collapses and then changes sign,
    so the expression stops being an error bound and starts being nonsense. The
    refusal has to happen on the budget rather than on the quotient, because a
    negative quotient looks like a small bound.

    **The budget below is chosen to sit between the old refusal threshold and the
    new one, and that choice is the whole test.** A first version used a stage
    count so enormous that the previous implementation's own guard rejected it
    too, so it passed against the broken code and against the fixed code alike --
    it pinned the new behaviour without discriminating. Here the accumulated
    budget is about 0.6: the current form gives ``gamma = 1.5`` and refuses, while
    the superseded power form gave ``(1 + 8u)^s - 1 = 0.82`` and waved it through
    as a usable bound. So this fails against the old implementation, which is what
    makes it a regression test rather than a description.
    """
    # 8 roundings per stage over this many stages accumulates to 0.6 units of
    # roundoff, derived rather than searched: stages = 0.6 / (8 u).
    stages_reaching_six_tenths = 675539944105574
    claim = bounded_parity(
        roundings=Roundings(
            stages=stages_reaching_six_tenths, accumulator_per_stage=8, storage_per_stage=0
        ),
        accumulator=np.float64,
        storage=np.float64,
        amplification=np.ones(1),
        why="a budget accumulating past the half where gamma stops bounding anything",
    )
    with pytest.raises(ValueError, match="vacuous"):
        absolute_tolerance(claim)


def _refuse_to_skip(gate: Callable[..., None], *args: Any) -> None:
    """Call a gate and fail loudly if it skips, rather than skipping with it.

    A bare call cannot express "this must not skip". :func:`pytest.skip` raises, so
    a gate that has been widened too far aborts the calling test as SKIPPED, which
    exits zero and reads as green. That is not a hypothetical failure mode: skipping
    too much is the exact defect this pair of gates was rewritten to correct, and a
    test that goes quiet on it guards only the direction that never went wrong.

    Args:
        gate (Callable[..., None]): The guard under test.
        *args (Any): Whatever it takes, if anything.

    Raises:
        Failed: Via :func:`pytest.fail`, if the gate skips.
    """
    try:
        gate(*args)
    except pytest.skip.Exception as skipped:
        pytest.fail(f"{gate.__name__} skipped where it must not: {skipped}")


def test_the_seed_gate_fires_at_float64_where_the_width_gate_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two gates own different divergences, and only one of them is about width.

    :func:`demand_the_compiled_kernel` owns storage format: interpreted ``float32``
    intermediates promote in a way that has never been pinned. It is right to let
    ``float64`` past, and an earlier version of this suite concluded from that that
    ``float64`` was safe outright. It is not, for two reasons that are nothing to do
    with width: a seed of ``np.power``, which IEEE 754 does not pin and whose numba
    and numpy implementations disagree by an ulp, and a falling-factorial accumulator
    that wraps at int64 when compiled and grows without bound when interpreted. Past
    that overflow the two paths differ by a factor of about seven, which is why this
    cannot be answered with a tolerance.

    :func:`demand_a_compiled_seed` owns those two, at every width, and is asked for
    per test rather than applied to every bitwise claim: for a kernel of ``+``, ``-``,
    ``*``, ``/`` and ``sqrt`` alone IEEE 754 pins each result, so interpretation
    reproduces the compiled bits by guarantee rather than by luck. Gating every
    bitwise claim instead was tried, and skips most of the parity suite for no gain;
    the figures are in the pull request that made this change, where they were taken.
    """
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    assert the_jit_is_disabled()

    _refuse_to_skip(demand_the_compiled_kernel, np.float64)
    with pytest.raises(pytest.skip.Exception):
        demand_the_compiled_kernel(np.float32)
    with pytest.raises(pytest.skip.Exception):
        demand_a_compiled_seed()

    monkeypatch.setenv("NUMBA_DISABLE_JIT", "0")
    assert not the_jit_is_disabled()
    _refuse_to_skip(demand_the_compiled_kernel, np.float32)
    _refuse_to_skip(demand_a_compiled_seed)


# --- assert_object_parity ---


def test_assert_object_parity_refuses_zero_fields() -> None:
    """An empty field list must not be a silent pass.

    What it catches: a caller (or a future refactor of ``assert_object_parity``
    itself) that drops its field list, e.g. through a filter that empties out, and
    ends up asserting agreement between two objects while comparing nothing of
    them at all.
    """
    with pytest.raises(AssertionError, match="no fields"):
        assert_object_parity(
            SimpleNamespace(),
            SimpleNamespace(),
            fields=(),
            context="a call with no fields",
        )


def test_assert_object_parity_refuses_a_repeated_field_name() -> None:
    """Two fields sharing a name must not silently collapse into one.

    What it catches: the return dict is keyed by name, so a repeated name means
    one field's comparison result is reported under the other's key -- and, if
    the first of the two happens to pass, its false "pass" could paper over the
    second's real failure depending on dict overwrite order.
    """
    claim = exact_parity(why="a probe claim; the values are never read")
    fields = (Field("n", claim), Field("n", claim))
    with pytest.raises(AssertionError, match="share a name"):
        assert_object_parity(
            SimpleNamespace(n=1),
            SimpleNamespace(n=1),
            fields=fields,
            context="a call with a repeated field name",
        )


def test_exact_parity_refuses_an_empty_or_blank_reason() -> None:
    """A claim that a tolerance cannot cover still needs a stated reason.

    What it catches: an ``ExactClaim`` built with no ``why``, which would leave a
    future failure message with nothing to explain what a difference means --
    exactly the gap :func:`bitwise_parity` and :func:`bounded_parity` are already
    guarded against.
    """
    with pytest.raises(ValueError, match="reason"):
        exact_parity(why="")
    with pytest.raises(ValueError, match="reason"):
        exact_parity(why="   ")


def test_an_exact_field_catches_a_differing_element_and_names_it() -> None:
    """An integer field reports which element differed and why exactness applies.

    What it catches: a comparison that reports only "arrays differ" without
    saying which element or what a difference would mean -- or one that silently
    compares nothing (e.g. an accidentally empty offenders mask).
    """
    claim = exact_parity(why="cell ids are indices, and a differing index is a wrong cell")
    fields = [Field("cell_ids", claim)]
    py_obj = SimpleNamespace(cell_ids=np.array([10, 11, 12]))
    cpp_obj = SimpleNamespace(cell_ids=np.array([10, 99, 12]))

    with pytest.raises(AssertionError, match=r"first at \(1,\)") as excinfo:
        assert_object_parity(py_obj, cpp_obj, fields=fields, context="cell id mismatch")

    message = str(excinfo.value)
    assert "cell ids are indices, and a differing index is a wrong cell" in message, (
        "the claim's own why must be quoted verbatim in the failure message"
    )
    assert "99" in message and "11" in message


def test_an_exact_field_catches_a_differing_shape() -> None:
    """A shape mismatch is reported as its own condition, not as an element diff.

    What it catches: an elementwise comparison cannot see a shape change at all
    (it would either raise on broadcasting or, worse, broadcast and compare
    nonsense). Two arrays of different length is itself a changed verdict --
    e.g. a differing number of boundary facets -- and needs a message that says
    so rather than crashing on the elementwise comparison it never reaches.
    """
    claim = exact_parity(why="a differing facet count is a changed verdict")
    fields = [Field("facet_ids", claim)]
    py_obj = SimpleNamespace(facet_ids=np.array([1, 2, 3]))
    cpp_obj = SimpleNamespace(facet_ids=np.array([1, 2]))

    with pytest.raises(AssertionError, match=r"shape \(2,\) against \(3,\)") as excinfo:
        assert_object_parity(py_obj, cpp_obj, fields=fields, context="facet count mismatch")

    assert "changed verdict" in str(excinfo.value)


@pytest.mark.parametrize(
    ("attr", "value", "other_value"),
    [
        pytest.param("count", 7, 8, id="int"),
        pytest.param("flag", True, False, id="bool"),
        pytest.param("name", "vertex", "cell", id="str"),
        pytest.param("shape", (1, 2, 3), (1, 2, 4), id="tuple-of-int"),
        pytest.param("dtype", np.dtype(np.int32), np.dtype(np.float64), id="numpy-dtype"),
    ],
)
def test_an_exact_field_agrees_and_disagrees_on_plain_python_values(
    attr: str, value: object, other_value: object
) -> None:
    """Exactness applies to non-array values too, not only to NumPy arrays.

    What it catches: a comparison that coerces every value through
    ``np.asarray`` and then compares object arrays (which fails or behaves
    oddly for a bare ``bool``/``str``/``dtype``), or one that only handles
    arrays and silently passes (or crashes) on anything else -- the shape most
    of a tag registry's or a build-provenance record's state actually has.
    """
    claim = exact_parity(why=f"a differing {attr} is a changed verdict, not a displaced value")
    fields = [Field(attr, claim)]

    agreeing_py = SimpleNamespace(**{attr: value})
    agreeing_cpp = SimpleNamespace(**{attr: value})
    assert_object_parity(agreeing_py, agreeing_cpp, fields=fields, context=f"agreeing {attr}")

    disagreeing_cpp = SimpleNamespace(**{attr: other_value})
    with pytest.raises(AssertionError, match="exact agreement claimed and violated"):
        assert_object_parity(
            agreeing_py, disagreeing_cpp, fields=fields, context=f"disagreeing {attr}"
        )


def test_an_exact_field_does_not_distinguish_a_tuple_from_an_equal_valued_array() -> None:
    """The container is not the answer, and an exact field compares the answer.

    `_assert_exact` normalises both sides through ``np.asarray`` before comparing,
    so a Python ``tuple`` against an equal-valued ``ndarray`` passes and a tuple
    against a ``list`` does too. Its docstring states that as a decision rather
    than an accident: a backend is free to hand back a different container, and
    ``CellTags``/``FacetTags`` will exercise exactly that -- their Python side
    holds arrays taken out of a `dict` while the C++ side returns whatever the
    binding produces.

    So this pins both directions. A future change making the comparison
    type-strict (``type(actual) is type(reference)``, or the earlier form that
    normalised only when one side was already an array) would fail the first half
    and break those consumers; a change that stopped comparing values at all would
    fail the second.
    """
    claim = exact_parity(why="tag ids must agree exactly, whatever container holds them")
    fields = [Field("ids", claim)]
    py_obj = SimpleNamespace(ids=(1, 2, 3))
    cpp_obj_equal = SimpleNamespace(ids=np.array([1, 2, 3]))

    assert_object_parity(
        py_obj, cpp_obj_equal, fields=fields, context="tuple against equal-valued array"
    )

    cpp_obj_differing = SimpleNamespace(ids=np.array([1, 9, 3]))
    with pytest.raises(AssertionError, match="exact agreement claimed and violated"):
        assert_object_parity(
            py_obj, cpp_obj_differing, fields=fields, context="tuple against differing array"
        )


def test_a_bitwise_field_in_an_object_call_detects_a_one_ulp_difference() -> None:
    """A BITWISE field really goes through `assert_parity`, not a stand-in equality.

    What it catches: `assert_object_parity` routing a `ParityClaim` field through
    something looser than the harness's own float comparison -- e.g. a bare
    ``==`` that would pass on identical arrays too, so this test alone would not
    distinguish it, which is why the second half plants a one-ulp difference.
    """
    claim = bitwise_parity(why="both backends compute this field with identical float ops")
    fields = [Field("value", claim)]
    identical = np.array([1.0, 2.5, -3.25])
    py_obj = SimpleNamespace(value=identical)
    cpp_obj_same = SimpleNamespace(value=identical.copy())
    assert_object_parity(py_obj, cpp_obj_same, fields=fields, context="bitwise field, identical")

    perturbed = identical.copy()
    perturbed[1] = np.nextafter(perturbed[1], np.inf)
    cpp_obj_perturbed = SimpleNamespace(value=perturbed)
    with pytest.raises(AssertionError, match="bitwise parity claimed and violated"):
        assert_object_parity(
            py_obj, cpp_obj_perturbed, fields=fields, context="bitwise field, one ulp off"
        )


def test_a_bounded_field_in_an_object_call_accepts_a_difference_inside_its_bound() -> None:
    """A BOUNDED field's claim really governs, rather than demanding exactness.

    What it catches: `assert_object_parity` routing a `ParityClaim` field through
    a bitwise or exact comparison instead of `assert_parity`, which would reject
    this same, deliberately non-zero, in-bound difference.

    The perturbation is derived from the claim's own tolerance (half of it)
    rather than picked to "look small", so the test does not depend on the
    rounding-budget arithmetic staying at today's constants.
    """
    claim = bounded_parity(
        roundings=Roundings(stages=4, accumulator_per_stage=2, storage_per_stage=1),
        accumulator=np.float64,
        storage=np.float64,
        amplification=np.array([1.0, 1.0]),
        why="probe claim: an arbitrary, non-degenerate rounding budget",
    )
    fields = [Field("value", claim)]
    tolerance = absolute_tolerance(claim)
    delta = float(tolerance[0]) / 2.0

    py_obj = SimpleNamespace(value=np.array([1.0, 2.0]))
    cpp_obj = SimpleNamespace(value=np.array([1.0 + delta, 2.0]))
    assert_object_parity(py_obj, cpp_obj, fields=fields, context="bounded field, half the bound")


def test_assert_object_parity_returns_one_deviation_per_field_and_reports_them() -> None:
    """The BVH-shaped call: a float field, an integer array field, a scalar count.

    What it catches: a return value that silently drops a field's `Deviation`
    (a caller that wants to assert on the margin, not just on the pass, would
    get a `KeyError` or a stale value instead), or a `Deviation` for the float
    field that is always all-zero regardless of what was actually compared --
    which would mean its ``read`` was ignored or its result discarded.
    """
    # 2**-40 is exactly representable next to 1.0 (its exponent is 40 below 1.0's,
    # well inside float64's 52-bit mantissa), so the subtraction that recovers it
    # inside assert_parity is exact too, and the observed deviation can be checked
    # against it with `==` rather than an approximate tolerance.
    delta = 2.0**-40
    float_claim = bounded_parity(
        # 20000 stages of one float64 rounding each accumulate a relative budget of
        # about 2 * 20000 * u ~= 4.4e-12, close to 5x delta: comfortably inside
        # without approaching the "budget exhausts the format" refusal.
        roundings=Roundings(stages=20000, accumulator_per_stage=1, storage_per_stage=0),
        accumulator=np.float64,
        storage=np.float64,
        amplification=np.array([1.0, 1.0]),
        why="probe claim: a budget chosen so the exact delta below sits well inside it",
    )
    fields = [
        Field("positions", float_claim),
        Field("node_ids", exact_parity(why="node ids are indices")),
        Field("n_nodes", exact_parity(why="a node count is a count")),
    ]
    py_obj = SimpleNamespace(
        positions=np.array([1.0, 2.0]), node_ids=np.array([0, 1, 2]), n_nodes=3
    )
    cpp_obj = SimpleNamespace(
        positions=np.array([1.0 + delta, 2.0]), node_ids=np.array([0, 1, 2]), n_nodes=3
    )

    deviations = assert_object_parity(
        py_obj, cpp_obj, fields=fields, context="BVH-shaped mixed fields"
    )

    assert set(deviations) == {"positions", "node_ids", "n_nodes"}
    assert deviations["node_ids"].num_differing == 0
    assert deviations["n_nodes"].num_differing == 0
    assert deviations["positions"].max_absolute == delta, (
        "the float field's Deviation must report the actual measured difference, not a placeholder"
    )
    assert deviations["positions"].num_differing == 1


def test_a_read_callable_reaches_a_mapping_style_field_built_in_a_loop() -> None:
    """The `CellTags`/`FacetTags` shape: state behind `__getitem__`, fields built in a loop.

    What it catches: a `read` that is ignored by `assert_object_parity` (the
    default `getattr` would raise `AttributeError` here, since the mapping's
    state is not a plain attribute), and -- the sharper bug -- a late-binding
    closure in the loop that builds the field list, which would make every
    field read the *last* key. The planted difference is at the *first* key, so
    a late-binding bug would leave it unread and this test would fail to raise.
    """

    class _TagRegistry:
        """A minimal stand-in for `CellTags`/`FacetTags`: state behind `__getitem__`."""

        def __init__(self, data: dict[str, np.ndarray]) -> None:
            self._data = data

        def __getitem__(self, key: str) -> np.ndarray:
            return self._data[key]

    names = ["vertex", "edge", "cell"]
    fields: list[Field] = []
    for name in names:
        # A `def`, not a `lambda`, and the loop variable defaulted into a second
        # parameter: this is the late-binding trap itself, made impossible to write
        # by accident. Writing `lambda obj: obj[name]` inline here would capture
        # `name` by reference, so every field would read the *last* iteration's key.
        def _read(obj: Any, key: str = name) -> Any:
            return obj[key]

        fields.append(Field(name, exact_parity(why="tag ids are indices"), read=_read))

    py_tags = _TagRegistry(
        {"vertex": np.array([1, 2, 3]), "edge": np.array([4, 5]), "cell": np.array([6])}
    )
    cpp_tags_ok = _TagRegistry(
        {"vertex": np.array([1, 2, 3]), "edge": np.array([4, 5]), "cell": np.array([6])}
    )
    assert_object_parity(py_tags, cpp_tags_ok, fields=fields, context="tag registry, agreeing")

    cpp_tags_bad_first_key = _TagRegistry(
        {"vertex": np.array([9, 2, 3]), "edge": np.array([4, 5]), "cell": np.array([6])}
    )
    with pytest.raises(AssertionError, match="vertex"):
        assert_object_parity(
            py_tags,
            cpp_tags_bad_first_key,
            fields=fields,
            context="tag registry, first key differs",
        )


def test_a_read_callable_reaches_a_ragged_indexed_field_built_in_a_loop() -> None:
    """The `TensorProductGrid.breakpoints` shape: a ragged per-axis tuple, reached by index.

    What it catches: the same late-binding hazard as the mapping-style test, in
    the shape a per-axis field list actually has -- a tuple of arrays with
    *different lengths*, so stacking them into one array (which would silently
    fix a late-binding bug by making every axis compare the same data) is not
    even possible here.
    """

    class _RaggedAxes:
        """A minimal stand-in for a per-axis breakpoint tuple of unequal lengths."""

        def __init__(self, breakpoints: tuple[np.ndarray, ...]) -> None:
            self.breakpoints = breakpoints

    py_axes = _RaggedAxes((np.array([0, 1, 2]), np.array([0, 1, 2, 3, 4]), np.array([0, 5])))
    cpp_axes_ok = _RaggedAxes((np.array([0, 1, 2]), np.array([0, 1, 2, 3, 4]), np.array([0, 5])))

    n_axes = len(py_axes.breakpoints)
    fields: list[Field] = []
    for axis in range(n_axes):
        # Same late-binding trap as the mapping-style test above, in the shape a
        # per-axis field list actually has.
        def _read(obj: Any, index: int = axis) -> Any:
            return obj.breakpoints[index]

        fields.append(
            Field(
                f"breakpoints[{axis}]", exact_parity(why="a breakpoint index is exact"), read=_read
            )
        )
    assert_object_parity(py_axes, cpp_axes_ok, fields=fields, context="ragged axes, agreeing")

    cpp_axes_bad_middle_axis = _RaggedAxes(
        (np.array([0, 1, 2]), np.array([0, 1, 9, 3, 4]), np.array([0, 5]))
    )
    with pytest.raises(AssertionError, match=r"breakpoints\[1\]"):
        assert_object_parity(
            py_axes,
            cpp_axes_bad_middle_axis,
            fields=fields,
            context="ragged axes, middle axis differs",
        )


def test_assert_object_parity_argument_order_is_visible_in_the_failure_message() -> None:
    """Swapping `py` and `cpp` swaps which value the failure message calls the oracle.

    `assert_object_parity(py, cpp, ...)` reads `cpp` as ``actual`` (the backend
    under test) and `py` as ``reference`` (the oracle), the opposite convention
    from :func:`~tests._parity_harness.assert_parity`. What this catches: a
    future edit to `assert_object_parity` that passed the two positionally in
    the wrong order internally -- silent for a call whose objects happen to
    agree, and visible only in which side a failure blames.
    """
    claim = exact_parity(why="a scalar count must agree exactly")
    fields = [Field("n", claim)]
    low = SimpleNamespace(n=1)
    high = SimpleNamespace(n=2)

    with pytest.raises(AssertionError, match=r"2 against 1"):
        assert_object_parity(low, high, fields=fields, context="py=low, cpp=high")

    with pytest.raises(AssertionError, match=r"1 against 2"):
        assert_object_parity(high, low, fields=fields, context="py=high, cpp=low")
