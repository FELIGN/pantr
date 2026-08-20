"""Kernel-agnostic harness for checking a C++ port against its Numba oracle.

This module is **infrastructure, not tests**. It is deliberately not named
``test_*.py`` so pytest does not collect it, and it is imported by the per-kernel
parity modules under ``tests/parity/`` (``test_basis_cardinal_bspline.py`` today,
one more per ported module as the port proceeds).

Why it exists
-------------

The port keeps the Numba implementation as the parity oracle, so every ported
kernel needs the same question answered: *how far apart may the two backends be,
and why exactly that far?* Answered once per kernel by hand, that question
produces a bare ``1e-12`` in five different files. The purpose of this module is
to make a bare number **impossible to express**.

There is no ``atol``, no ``rtol`` and no scalar tolerance parameter anywhere in
this API. A parity assertion takes a :class:`ParityClaim`, and a claim can only
be built by one of two factories:

``bitwise_parity(why=...)``
    The two backends perform the identical sequence of IEEE-754 operations in the
    identical order, so the results agree bit for bit. Nothing but a reason is
    required, because nothing but a reason is available.

``bounded_parity(roundings=..., accumulator=..., storage=..., amplification=..., why=...)``
    The two backends differ somewhere, and the difference is bounded by a
    *derivation*: how many sequential stages the recurrence has, how many
    roundings each stage commits in the accumulator and in the storage format,
    which formats those are, and how much the recurrence amplifies a perturbation
    at each output element. The tolerance is computed from those; it cannot be
    typed in.

The keyword arguments are all required and all keyword-only. Supplying them is
the derivation.

The amplification array
-----------------------

``amplification`` is the piece that carries the mathematics of the particular
kernel, and it is an **array**, one entry per output element, not a scalar. It is
the elementwise factor by which the recurrence magnifies a relative perturbation
of its intermediates: for a recurrence that forms convex combinations of
non-negative values it equals the value itself (a relative bound), and for one
that cancels it is strictly larger (the running sum of the magnitudes of the
contributions). Making it an array rather than a number is what lets one bound
cover both the well-conditioned interior of a domain and its badly conditioned
outside, without a second tolerance and without excluding the hard cases.

One way to produce it is a *companion recurrence*: run the kernel's own recurrence
with every coefficient replaced by its absolute value. See
``tests/parity/test_basis_cardinal_bspline.py`` for a worked one.

**That recipe is not general, and where it fails it fails silently.** It bounds a
recurrence whose coefficients form a convex combination, which is the cardinal
B-spline case it was written for: the absolute values change nothing, because the
coefficients are already non-negative and sum to one. For an *oscillatory*
three-term recurrence it is not a bound but a different, exponentially growing
sequence. Legendre is the case at hand: ``P_k`` and its second solution ``Q_k``
are both bounded on ``[-1, 1]``, and taking absolute values in
``((2k-1) x P_{k-1} - (k-1) P_{k-2}) / k`` replaces them by growth like
``(1 + sqrt(2))**k``. Measured at degree 700 the companion reaches ``1.7e266``.

Nothing in the type system stops that: the amplification is finite and
non-negative, which is all :func:`bounded_parity` can check. What stops it is
:func:`assert_parity`, which refuses a bound at least as large as the values being
compared, because such a bound is satisfied by any result at all.

For a quantity built from a *ratio* of recurrence values, bound the ratio. A
Gauss-Legendre node's displacement is limited by ``residual / |P'|``, and its
weight's by the same displacement pushed through ``dw/dx``; both are O(1) objects
where the companion is not.

The underflow floor
-------------------

A bound built only from ``amplification`` is proportional to the magnitude of the
result, and so goes to zero with it. Floating-point error does not: below the
smallest normal the spacing stops shrinking, and a rounding costs an absolute
amount no relative bound can express. A purely relative bound therefore asserts
*bit-for-bit agreement* on every result small enough, which it has no grounds
for and which correct code violates. It did: on the cardinal B-spline's own
oracle point set, float32 at degree 16, the shipped bound was exceeded by a
factor of 1.0e6 by two backends that agree perfectly.

So every tolerance here carries both halves of Higham's model -- a relative term
and an absolute floor. :func:`underflow_floor` supplies the magnitude, and the
counts already in :class:`Roundings` supply how many of them there are. No new
keyword argument was needed for it, which is the point: the budget was always
stated as a count, and only its conversion to a magnitude was incomplete.

What the harness deliberately does not do
-----------------------------------------

It does not know whether the two backends are **right**, only whether they
**agree**. Parity is a consistency check, and a consistency check whose reference
is the code itself certifies nothing. Every per-kernel module using this harness
owes an independent oracle as well: closed-form values, exact rational
arithmetic, an analytic invariant. :func:`assert_accuracy` is the slot for that,
and it takes an elementwise derived bound for the same reason.
"""

from __future__ import annotations

import os
from enum import IntEnum
from typing import Any, Final, NamedTuple, cast

import numpy as np
import numpy.typing as npt
import pytest

__all__ = [
    "AccuracyClaim",
    "BuildProvenance",
    "Deviation",
    "ParityClaim",
    "ParityKind",
    "Roundings",
    "absolute_tolerance",
    "assert_accuracy",
    "assert_parity",
    "bitwise_parity",
    "bounded_parity",
    "build_provenance",
    "contraction_may_fuse",
    "cpp_backend_available",
    "demand_cpp_backend",
    "derived_accuracy",
    "underflow_floor",
    "unit_roundoff",
]

FloatArray = npt.NDArray[np.floating[Any]]
"""Any real-valued NumPy array the backends can produce or consume."""

_REQUIRE_ENV_VAR: Final[str] = "PANTR_REQUIRE_CPP"
"""Environment variable that turns "the extension is absent" from a skip into a failure."""

_BUILD_HINT: Final[str] = (
    "build it with `pip install -e .` (scikit-build-core drives CMake), and set "
    f"{_REQUIRE_ENV_VAR}=1 to make its absence a failure rather than a skip"
)
"""What to do about a missing extension, quoted in every skip reason."""


# ---------------------------------------------------------------------------
# Availability, and making its absence audible
# ---------------------------------------------------------------------------


def cpp_backend_available() -> bool:
    """Report whether the compiled extension is importable in this installation.

    Returns:
        bool: True when ``pantr._pantr_cpp`` was built into this installation.
    """
    from pantr._backend import Backend, available_backends  # noqa: PLC0415

    return Backend.CPP in available_backends()


def require_cpp_backend_is_set() -> bool:
    """Report whether the environment demands the extension be present.

    CLAUDE.md records that a missing optional dependency skips silently here, and
    that a local green built on such a skip has let real failures through. CI sets
    this variable in the job that builds the extension, so that job cannot pass by
    skipping the very tests it exists to run.

    Returns:
        bool: True when ``PANTR_REQUIRE_CPP`` is set to a non-empty value.
    """
    return bool(os.environ.get(_REQUIRE_ENV_VAR, ""))


SKIP_REASON: Final[str] = (
    f"the pantr._pantr_cpp extension is not built in this installation; {_BUILD_HINT}"
)
"""Reason attached to every skipped parity test, printed by ``pytest -rs``."""

_MISSING_BUT_REQUIRED: Final[str] = (
    f"{_REQUIRE_ENV_VAR} is set but pantr._pantr_cpp is not importable, so every C++ "
    f"parity test that asked for it would have been skipped. Build the extension with "
    f"`pip install -e .`, or unset {_REQUIRE_ENV_VAR} if this run is meant to be "
    f"Numba-only."
)
"""Message for the one configuration in which a missing extension must fail."""


def demand_cpp_backend() -> None:
    """Skip, or fail, when the compiled extension is absent.

    **The only supported way for a parity test to require the extension**, and the
    reason a plain ``pytest.mark.skipif`` is not offered here. A bare skip is the
    trap CLAUDE.md names by name: a missing optional dependency skips without
    complaint, and a local green built on such a skip has let real failures through
    in this repository before. Under ``PANTR_REQUIRE_CPP`` the absence is a failure
    instead, which is what stops the CI job that builds the extension from passing
    by skipping the very tests it exists to run.

    Call it from a fixture rather than applying it module-wide: a module-level mark
    silences the test that reports the extension's *presence* along with everything
    else, and that report is the whole point.

    Raises:
        Failed: Via :func:`pytest.fail`, when ``PANTR_REQUIRE_CPP`` is set and the
            extension is missing.
        Skipped: Via :func:`pytest.skip`, when it is missing and not required.
    """
    if cpp_backend_available():
        return
    if require_cpp_backend_is_set():
        pytest.fail(_MISSING_BUT_REQUIRED)
    pytest.skip(SKIP_REASON)


class BuildProvenance(NamedTuple):
    """Which binary produced a measurement, and how it treats contraction.

    The extension reports these itself (``cpp/bindings/pantr_cpp.cpp``), so a
    parity claim can be conditioned on the build rather than on an assumption
    about the build.

    Attributes:
        compiler (str): Compiler name and version, e.g. ``"gcc 14.4.0"``.
        fp_contract (str): ``"available"`` when the target ISA has an FMA the
            compiler may contract into, ``"unavailable-on-target-isa"`` otherwise.
        has_std_mdspan (bool): Whether the standard library supplied
            ``std::mdspan`` rather than the fetched reference implementation.
    """

    compiler: str
    fp_contract: str
    has_std_mdspan: bool


FP_CONTRACT_AVAILABLE: Final[str] = "available"
"""Value of ``__fp_contract__`` meaning the target ISA has an FMA to contract into."""

FP_CONTRACT_UNAVAILABLE: Final[str] = "unavailable-on-target-isa"
"""Value of ``__fp_contract__`` meaning no FMA instruction exists on the target."""


def build_provenance() -> BuildProvenance:
    """Read the compiled extension's self-reported build provenance.

    Returns:
        BuildProvenance: The compiler, the contraction availability and whether
            ``std::mdspan`` came from the standard library.

    Raises:
        RuntimeError: If the extension is not available.
    """
    if not cpp_backend_available():
        raise RuntimeError(SKIP_REASON)

    # Imported here rather than read off a module handle held in pantr._backend:
    # the provenance attributes are declared in src/pantr/_pantr_cpp.pyi, so this
    # spelling is the one mypy can actually check.
    from pantr import _pantr_cpp  # noqa: PLC0415

    return BuildProvenance(
        compiler=str(_pantr_cpp.__compiler__),
        fp_contract=str(_pantr_cpp.__fp_contract__),
        has_std_mdspan=bool(_pantr_cpp.__has_std_mdspan__),
    )


def contraction_may_fuse() -> bool:
    """Report whether this build can fuse a multiply-add at all.

    ``cmake/PantrCompileOptions.cmake`` sets ``-ffp-contract=on`` but no
    ``-march``, so the target is baseline x86-64, which has no FMA instruction and
    therefore nothing to contract into. ``design/simd.md`` schedules
    ``-march=x86-64-v3`` as a later stage, at which point this flips and the
    bounded branch of a parity claim becomes the live one.

    Returns:
        bool: True when the target ISA offers a fused multiply-add.
    """
    return build_provenance().fp_contract == FP_CONTRACT_AVAILABLE


# ---------------------------------------------------------------------------
# The claim types
# ---------------------------------------------------------------------------


class ParityKind(IntEnum):
    """How far apart two backends are allowed to be.

    An :class:`~enum.IntEnum` rather than a string, per the project's rule that a
    closed set of choices is never stringly typed.

    Attributes:
        BITWISE: Identical operation sequence in identical order, so identical
            bits. The strongest claim available, and the one to make whenever it
            holds: a tolerance that is never approached asserts nothing.
        BOUNDED: The backends differ, within a bound derived from a rounding
            budget and an elementwise amplification.
    """

    BITWISE = 0
    BOUNDED = 1


class Roundings(NamedTuple):
    """The rounding budget of one kernel, as a count rather than a magnitude.

    Counting roundings is the part a reader can check against the source; turning
    a count into a magnitude is arithmetic the harness does. Splitting the count
    by format is what makes one budget serve float32 and float64 storage over a
    float64 accumulator, which is the shape every pantr kernel has.

    Attributes:
        stages (int): Sequential stages in the recurrence, i.e. the length of the
            dependency chain from an input to an output element. Not the total
            operation count: independent operations do not compound.
        accumulator_per_stage (int): Roundings each stage commits in the
            accumulator format, counted along the dominant path. Independent
            summands of a non-cancelling sum count once between them, not once
            each.
        storage_per_stage (int): Roundings each stage commits when narrowing to
            the storage format, i.e. stores back into the output array.
    """

    stages: int
    accumulator_per_stage: int
    storage_per_stage: int


class ParityClaim(NamedTuple):
    """A statement about how far two backends may differ, with its derivation.

    Built by :func:`bitwise_parity` or :func:`bounded_parity`; do not construct it
    positionally.

    Attributes:
        kind (ParityKind): Which of the two statements is being made.
        roundings (Roundings | None): The rounding budget. None for BITWISE.
        accumulator (np.dtype[np.floating[Any]] | None): Format intermediates are
            accumulated in. None for BITWISE.
        storage (np.dtype[np.floating[Any]] | None): Format the output array
            holds. None for BITWISE.
        amplification (FloatArray | None): Elementwise factor by which the
            recurrence magnifies a relative perturbation. None for BITWISE.
        why (str): The derivation, or the reason exactness holds. Quoted verbatim
            in any failure message.
    """

    kind: ParityKind
    roundings: Roundings | None
    accumulator: np.dtype[np.floating[Any]] | None
    storage: np.dtype[np.floating[Any]] | None
    amplification: FloatArray | None
    why: str


class AccuracyClaim(NamedTuple):
    """An elementwise absolute error bound against an independent oracle.

    Built by :func:`derived_accuracy`.

    Attributes:
        bound (FloatArray): Elementwise absolute bound, same shape as the result.
        why (str): The derivation. Quoted verbatim in any failure message.
    """

    bound: FloatArray
    why: str


class Deviation(NamedTuple):
    """What a comparison actually observed, for a caller that wants to assert on it.

    Attributes:
        max_absolute (float): Largest absolute difference over all elements.
        max_ratio_to_bound (float): Largest ratio of difference to its own
            elementwise bound. At most 1 for a comparison that passed; 0 when the
            arrays are identical.
        num_differing (int): How many elements differed at all.
    """

    max_absolute: float
    max_ratio_to_bound: float
    num_differing: int


def _float_dtype(dtype: npt.DTypeLike) -> np.dtype[np.floating[Any]]:
    """Normalise a dtype-like value to a floating dtype.

    Args:
        dtype (npt.DTypeLike): A floating-point dtype.

    Returns:
        np.dtype[np.floating[Any]]: The normalised dtype.

    Raises:
        TypeError: If it is not a floating-point dtype.
    """
    resolved = np.dtype(dtype)
    if not np.issubdtype(resolved, np.floating):
        raise TypeError(f"{resolved} is not a floating-point dtype")
    return cast("np.dtype[np.floating[Any]]", resolved)


def unit_roundoff(dtype: npt.DTypeLike) -> float:
    """Return the unit roundoff of a floating format.

    Half an epsilon, not an epsilon: ``eps`` is the gap between 1 and its
    successor, and round-to-nearest commits at most half a gap.

    Args:
        dtype (npt.DTypeLike): A floating-point dtype.

    Returns:
        float: The unit roundoff ``u`` of that format.
    """
    return float(np.finfo(np.dtype(dtype)).eps) / 2.0


def underflow_floor(dtype: npt.DTypeLike) -> float:
    r"""Return the absolute error floor of one rounding in a floating format.

    The other half of the rounding model, and the half a purely relative bound
    silently omits. Higham's model of a floating-point operation is

    .. math::

        \mathrm{fl}(x \mathbin{\mathrm{op}} y)
            = (x \mathbin{\mathrm{op}} y)(1 + \delta) + \eta,
        \qquad |\delta| \le u, \quad |\eta| \le \eta_{\text{fmt}},
        \quad \delta\eta = 0,

    with :math:`\eta` non-zero only when the exact result falls in the gradual-
    underflow range. There the spacing is no longer proportional to the value: it
    is the constant :math:`\lambda`, the smallest positive subnormal, and a
    rounding commits at most :math:`\lambda / 2`. A bound written as
    :math:`u |x|` alone therefore goes to zero with :math:`x` while the true
    error does not, and asserts something false about every result small enough.

    :math:`\lambda` is returned rather than the sharp :math:`\lambda / 2`, and the
    reason is representability rather than a safety margin. Halving the smallest
    subnormal always ties to even and gives zero *in that format* -- true of every
    binary format, not a float64 peculiarity. What differs is the format the
    bounds here are accumulated in, which is float64: so :math:`\lambda / 2`
    survives for float32 (``7.0e-46``) and collapses for float64, where it is
    exactly ``0.0``. Returning the sharp constant would therefore leave float64
    with no floor at all, which is the bug this function exists to fix. The
    resulting factor-of-two over-estimate costs nothing, because the floor only
    ever binds where the relative term has already collapsed.

    Args:
        dtype (npt.DTypeLike): A floating-point dtype.

    Returns:
        float: The absolute floor of one rounding in that format.
    """
    return float(np.finfo(np.dtype(dtype)).smallest_subnormal)


def bitwise_parity(*, why: str) -> ParityClaim:
    """Claim that the two backends produce identical bits.

    Args:
        why (str): Why the two operation sequences are identical. A build-specific
            reason (no FMA on the target ISA, say) must name what would change it.

    Returns:
        ParityClaim: A claim asserting bit-for-bit agreement.

    Raises:
        ValueError: If ``why`` is empty.
    """
    if not why.strip():
        raise ValueError("a bitwise parity claim must say why the operation sequences agree")
    return ParityClaim(
        kind=ParityKind.BITWISE,
        roundings=None,
        accumulator=None,
        storage=None,
        amplification=None,
        why=why,
    )


def bounded_parity(
    *,
    roundings: Roundings,
    accumulator: npt.DTypeLike,
    storage: npt.DTypeLike,
    amplification: FloatArray,
    why: str,
) -> ParityClaim:
    """Claim that the two backends differ within a derived elementwise bound.

    Every argument is required and keyword-only. Together they *are* the
    derivation: there is no parameter into which a fitted number could be typed.

    Args:
        roundings (Roundings): The kernel's rounding budget.
        accumulator (npt.DTypeLike): Format intermediates are accumulated in.
        storage (npt.DTypeLike): Format the output array holds.
        amplification (FloatArray): Elementwise amplification factor, same shape
            as the result. Must be finite and non-negative.
        why (str): The derivation, in prose, including what makes the
            amplification array correct.

    Returns:
        ParityClaim: A claim asserting agreement within the derived bound.

    Raises:
        ValueError: If ``why`` is empty, if the budget is not a positive number of
            stages, or if the amplification array is not finite and non-negative.
    """
    if not why.strip():
        raise ValueError("a bounded parity claim must carry its derivation")
    if roundings.stages < 1:
        raise ValueError("a bounded claim over zero stages is a bitwise claim; use bitwise_parity")
    if roundings.accumulator_per_stage < 0 or roundings.storage_per_stage < 0:
        raise ValueError("rounding counts are counts, so they cannot be negative")
    amp = np.asarray(amplification, dtype=np.float64)
    if not np.all(np.isfinite(amp)) or np.any(amp < 0.0):
        raise ValueError(
            "the amplification array must be finite and non-negative; an infinite "
            "entry means the kernel overflowed and the comparison is vacuous there"
        )
    return ParityClaim(
        kind=ParityKind.BOUNDED,
        roundings=roundings,
        accumulator=_float_dtype(accumulator),
        storage=_float_dtype(storage),
        amplification=amp,
        why=why,
    )


def derived_accuracy(*, bound: FloatArray, why: str) -> AccuracyClaim:
    """Claim an elementwise absolute error bound against an independent oracle.

    Args:
        bound (FloatArray): Elementwise absolute bound, same shape as the result.
        why (str): How the bound was derived.

    Returns:
        AccuracyClaim: The claim.

    Raises:
        ValueError: If ``why`` is empty or the bound is not finite and positive.
    """
    if not why.strip():
        raise ValueError("an accuracy claim must carry its derivation")
    arr = np.asarray(bound, dtype=np.float64)
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("the accuracy bound must be finite and non-negative")
    return AccuracyClaim(bound=arr, why=why)


ONE_SIDED_TO_TWO_SIDED: Final[int] = 2
"""Factor turning a one-sided forward-error bound into a two-sided parity bound.

Neither backend is the exact answer. Each sits within its own forward-error bound
of the exact recurrence, so their difference is bounded by the sum of the two.
The factor is 2 because the two bounds are equal, and it is a derivation rather
than a safety margin: dropping it would be claiming one backend rounds perfectly.
"""


def _relative_growth(claim: ParityClaim) -> float:
    """Accumulate a claim's per-stage rounding budget over its stages.

    Args:
        claim (ParityClaim): A BOUNDED claim.

    Returns:
        float: The relative factor ``(1 + delta)**stages - 1``, the standard
            ``gamma`` of a forward-error analysis.

    Raises:
        ValueError: If the budget is so large that the bound carries no
            information (a relative bound at or above 1 accepts everything).
    """
    assert claim.roundings is not None  # BOUNDED by construction
    assert claim.accumulator is not None
    assert claim.storage is not None

    u_acc = unit_roundoff(claim.accumulator)
    # A store into the accumulator's own format rounds nothing, so it costs
    # nothing. Only a narrowing store does.
    u_store = 0.0 if claim.storage == claim.accumulator else unit_roundoff(claim.storage)

    per_stage = (
        claim.roundings.accumulator_per_stage * u_acc + claim.roundings.storage_per_stage * u_store
    )
    if per_stage == 0.0:
        raise ValueError(
            f"the rounding budget {claim.roundings} commits no rounding at all in "
            f"accumulator {claim.accumulator} with storage {claim.storage}, so the "
            f"relative bound is exactly zero and this BOUNDED claim asserts bit-for-bit "
            f"agreement while saying it does not. Use bitwise_parity, which says so and "
            f"reports a violation as one."
        )
    growth = float((1.0 + per_stage) ** claim.roundings.stages - 1.0)
    if growth >= 1.0:
        raise ValueError(
            f"the rounding budget accumulates to a relative bound of {growth:.3g}, "
            f"which accepts every finite result. {claim.roundings.stages} stages of "
            f"{per_stage:.3g} exhaust the format; the comparison would be vacuous."
        )
    return growth


def _underflow_budget(claim: ParityClaim) -> float:
    """Accumulate a claim's per-stage rounding budget as an ABSOLUTE floor.

    The magnitude half of a budget :class:`Roundings` already carries as a count.
    Each rounding contributes at most :func:`underflow_floor` of its own format
    absolutely, on top of its relative contribution, and the floors add over the
    stages:

    * where the amplification is at most 1 -- inside the span, where every stage
      map is a convex combination of non-negative values -- an absolute
      perturbation is passed on unamplified, so the total is exactly the sum;
    * where the amplification exceeds 1, a floor introduced at one stage can be
      magnified by the later ones, by at most that same amplification. That case
      needs no separate term, because it is already inside the relative one: the
      magnified floor is at most ``amplification * stages * floor``, which is
      below ``gamma * amplification`` whenever ``stages * floor <= gamma``. Both
      sides are linear in the stage count, so that comparison does not depend on
      the degree at all -- it is a fact about the formats. Measured, for the two
      this harness is used with: float64 storage leaves ``gamma`` larger by 307
      orders of magnitude, float32 storage by 38.

    Neither margin is close, so the relative term absorbs the amplified floor
    whole and the simple sum is a bound in both regimes.

    So the floor binds only where the relative term has collapsed -- which is
    exactly where it must, since that is where a purely relative bound asserts
    that two backends may not differ at all.

    Args:
        claim (ParityClaim): A BOUNDED claim.

    Returns:
        float: The absolute floor accumulated over the claim's stages.
    """
    assert claim.roundings is not None  # BOUNDED by construction
    assert claim.accumulator is not None
    assert claim.storage is not None

    eta_acc = underflow_floor(claim.accumulator)
    # A store into the accumulator's own format rounds nothing, so it has no
    # floor either -- the same reasoning as for its relative cost.
    eta_store = 0.0 if claim.storage == claim.accumulator else underflow_floor(claim.storage)

    per_stage = (
        claim.roundings.accumulator_per_stage * eta_acc
        + claim.roundings.storage_per_stage * eta_store
    )
    return claim.roundings.stages * per_stage


def absolute_tolerance(claim: ParityClaim) -> FloatArray:
    """Compute the elementwise absolute tolerance a claim permits.

    Args:
        claim (ParityClaim): The claim.

    Returns:
        FloatArray: Zero everywhere for BITWISE; otherwise
            ``2 * (gamma * amplification + underflow floor)``, the two halves of
            the rounding model that :func:`underflow_floor` describes. Both are
            doubled for the same reason: neither backend is the exact answer, so
            the parity bound is twice a one-sided forward-error bound.
    """
    if claim.kind is ParityKind.BITWISE:
        shape = () if claim.amplification is None else claim.amplification.shape
        return np.zeros(shape, dtype=np.float64)
    assert claim.amplification is not None  # BOUNDED by construction
    relative = _relative_growth(claim) * claim.amplification
    return ONE_SIDED_TO_TWO_SIDED * (relative + _underflow_budget(claim))


def _bit_pattern(array: FloatArray) -> npt.NDArray[np.unsignedinteger[Any]]:
    """View a float array as the unsigned integer of the same width.

    Args:
        array (FloatArray): A contiguous float32 or float64 array.

    Returns:
        npt.NDArray[np.unsignedinteger[Any]]: The raw bit patterns.

    Raises:
        TypeError: If the dtype is neither float32 nor float64.
    """
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype == np.float64:
        return contiguous.view(np.uint64)
    if contiguous.dtype == np.float32:
        return contiguous.view(np.uint32)
    raise TypeError(f"no bit view defined for dtype {contiguous.dtype}")


def _bits_differ(actual: FloatArray, reference: FloatArray) -> npt.NDArray[np.bool_]:
    """Locate elements whose bit patterns differ, treating the two zeros as equal.

    ``-0.0`` and ``+0.0`` have different bit patterns and the same value. CLAUDE.md
    records that the sign of a min/max tie already differs across the NumPy
    versions in the CI matrix, so a suite that pins the sign of a zero pins
    something the platform does not promise. A signed-zero disagreement is
    therefore not counted here; a disagreement in any other bit is.

    Args:
        actual (FloatArray): One result.
        reference (FloatArray): The other.

    Returns:
        npt.NDArray[np.bool_]: True where the two differ in value.
    """
    differ = _bit_pattern(actual) != _bit_pattern(reference)
    both_zero = (actual == 0.0) & (reference == 0.0)
    return np.asarray(differ & ~both_zero)


def _worst_element_report(
    actual: FloatArray,
    reference: FloatArray,
    tolerance: FloatArray,
    offenders: npt.NDArray[np.bool_],
) -> str:
    """Describe the single worst offending element in a failed comparison.

    Args:
        actual (FloatArray): One result.
        reference (FloatArray): The other.
        tolerance (FloatArray): Elementwise tolerance, broadcast to the results.
        offenders (npt.NDArray[np.bool_]): Mask of elements that failed.

    Returns:
        str: A multi-line description naming the index, both values and the ratio.
    """
    difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    penalized = np.where(offenders, difference, -np.inf)
    flat = int(np.argmax(penalized))
    index = np.unravel_index(flat, difference.shape)
    allowed = float(np.broadcast_to(tolerance, difference.shape)[index])
    observed = float(difference[index])
    ratio = observed / allowed if allowed > 0.0 else np.inf
    return (
        f"  worst element {index}:\n"
        f"    actual    = {float(actual[index])!r}\n"
        f"    reference = {float(reference[index])!r}\n"
        f"    |diff|    = {observed:.6e}\n"
        f"    allowed   = {allowed:.6e}\n"
        f"    ratio     = {ratio:.4g}\n"
        f"  {int(np.count_nonzero(offenders))} of {difference.size} elements out of bound"
    )


def _refuse_a_vacuous_bound(
    tolerance: FloatArray, reference: FloatArray, claim: ParityClaim, context: str
) -> None:
    """Refuse a bound so large that no finite result could violate it.

    A tolerance larger than the largest magnitude in the reference array admits
    *any* value at that element, zero included, so the comparison decides nothing.
    That is a different failure from a bound being too loose, and it is invisible:
    the assertion passes, for ever, and reports agreement.

    The check is against the array's **largest** magnitude rather than each
    element's own. Per element it would reject a legitimate absolute floor on a
    value that is genuinely near zero, which is the case
    :func:`underflow_floor` exists to serve. Against the largest, a bound that is
    meaningful anywhere in the array survives, and only a bound that is
    meaningless everywhere is refused.

    The bound this catches is not hypothetical, and it comes from following this
    module's own advice. :func:`bounded_parity` documents the usual way to obtain
    an amplification as "run the kernel's own recurrence with every coefficient
    replaced by its absolute value". That is right for a recurrence whose
    coefficients form a convex combination, which is the cardinal B-spline case it
    was written for. It is catastrophically wrong for an oscillatory three-term
    recurrence such as the Legendre one, where taking absolute values replaces two
    bounded homogeneous solutions by growth like ``(1 + sqrt(2))**k``: measured at
    degree 700, the companion reaches ``1.7e266``, the tolerance it produces is
    ``5.3e253``, and ``assert_parity`` then accepts ``1.0`` against ``-1e250``.

    Args:
        tolerance (FloatArray): The elementwise bound, already broadcast.
        reference (FloatArray): The oracle values being compared against.
        claim (ParityClaim): The claim under test, quoted in the message.
        context (str): What was being computed.

    Raises:
        AssertionError: If the bound exceeds the largest reference magnitude.
    """
    largest = float(np.abs(reference.astype(np.float64)).max(initial=0.0))
    worst = float(np.asarray(tolerance).max(initial=0.0))
    if largest > 0.0 and worst >= largest:
        raise AssertionError(
            f"{context}: the derived bound is vacuous and nothing was compared.\n"
            f"  derivation: {claim.why}\n"
            f"  largest bound {worst:.3e} against largest reference magnitude "
            f"{largest:.3e}\n"
            f"  A bound at least as large as the values being compared is satisfied "
            f"by any result, zero included. Check the amplification: an absolute-value "
            f"companion recurrence is only a bound for a convex-combination recurrence, "
            f"and diverges for an oscillatory one. For those, bound the ratio the "
            f"quantity is actually built from instead."
        )


def assert_parity(
    actual: FloatArray,
    reference: FloatArray,
    claim: ParityClaim,
    *,
    context: str,
) -> Deviation:
    """Assert that two backends agree as far as a claim says they must.

    Args:
        actual (FloatArray): The backend under test, conventionally the C++ one.
        reference (FloatArray): The oracle, conventionally the Numba one.
        claim (ParityClaim): What agreement is being claimed, and why.
        context (str): What was being computed, quoted in a failure message.

    Returns:
        Deviation: What was observed, so a caller can assert on the margin as well
            as on the pass.

    Raises:
        AssertionError: If shapes or dtypes disagree, if any element is not finite
            in one result and finite in the other, or if the claim is violated.
    """
    assert actual.shape == reference.shape, (
        f"{context}: shape {actual.shape} against {reference.shape}"
    )
    assert actual.dtype == reference.dtype, (
        f"{context}: dtype {actual.dtype} against {reference.dtype}"
    )

    finite_actual = np.isfinite(actual)
    assert np.array_equal(finite_actual, np.isfinite(reference)), (
        f"{context}: the two backends disagree about which entries are finite, "
        f"which no tolerance can absorb"
    )

    if claim.kind is ParityKind.BITWISE:
        offenders = _bits_differ(actual, reference)
        difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
        deviation = Deviation(
            max_absolute=float(difference.max(initial=0.0)),
            max_ratio_to_bound=0.0 if not offenders.any() else float(np.inf),
            num_differing=int(np.count_nonzero(offenders)),
        )
        if offenders.any():
            zeros = np.zeros_like(difference)
            raise AssertionError(
                f"{context}: bitwise parity claimed and violated.\n"
                f"  claim: {claim.why}\n"
                f"{_worst_element_report(actual, reference, zeros, offenders)}\n"
                f"  A bitwise claim failing means the two backends no longer perform "
                f"the same operations in the same order. Either the port changed, or "
                f"the build did (check pantr._pantr_cpp.__fp_contract__ and "
                f"__compiler__); do not relax this to a tolerance without deriving one."
            )
        return deviation

    tolerance = np.broadcast_to(absolute_tolerance(claim), actual.shape)
    _refuse_a_vacuous_bound(tolerance, reference, claim, context)
    difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    offenders = np.asarray(difference > tolerance)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(
            tolerance > 0.0, difference / tolerance, np.where(difference > 0.0, np.inf, 0.0)
        )
    deviation = Deviation(
        max_absolute=float(difference.max(initial=0.0)),
        max_ratio_to_bound=float(np.max(ratios, initial=0.0)),
        num_differing=int(np.count_nonzero(difference > 0.0)),
    )
    if offenders.any():
        raise AssertionError(
            f"{context}: the backends differ by more than the derived bound.\n"
            f"  derivation: {claim.why}\n"
            f"  budget: {claim.roundings}, accumulator {claim.accumulator}, "
            f"storage {claim.storage}\n"
            f"{_worst_element_report(actual, reference, tolerance, offenders)}"
        )
    return deviation


def assert_accuracy(
    computed: FloatArray,
    exact: FloatArray,
    claim: AccuracyClaim,
    *,
    context: str,
) -> Deviation:
    """Assert that a result matches an independent oracle within a derived bound.

    Args:
        computed (FloatArray): What a backend produced.
        exact (FloatArray): The oracle's values.
        claim (AccuracyClaim): The elementwise bound and its derivation.
        context (str): What was being computed, quoted in a failure message.

    Returns:
        Deviation: What was observed.

    Raises:
        AssertionError: If the bound is violated.
    """
    assert computed.shape == exact.shape, (
        f"{context}: shape {computed.shape} against oracle shape {exact.shape}"
    )
    difference = np.abs(computed.astype(np.float64) - exact.astype(np.float64))
    bound = np.broadcast_to(claim.bound, difference.shape)
    offenders = np.asarray(difference > bound)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(bound > 0.0, difference / bound, np.where(difference > 0.0, np.inf, 0.0))
    deviation = Deviation(
        max_absolute=float(difference.max(initial=0.0)),
        max_ratio_to_bound=float(np.max(ratios, initial=0.0)),
        num_differing=int(np.count_nonzero(difference > 0.0)),
    )
    if offenders.any():
        raise AssertionError(
            f"{context}: the result is further from the independent oracle than the "
            f"derived bound allows.\n"
            f"  derivation: {claim.why}\n"
            f"{_worst_element_report(computed, exact, bound, offenders)}"
        )
    return deviation
