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

Comparing two objects
---------------------

Until the port reached the domain types, everything here compared two arrays. A
type has several, of mixed kinds, and the arrays alone are not the object: a BVH
is five arrays plus three counts, a tensor-product grid is a ragged tuple of
per-axis breakpoints plus a shape, a tag registry is a mapping.

``tests/parity/test_grid.py`` shows what that costs when it is done by hand --
pull the five BVH arrays out, compare two of them with :func:`assert_parity` and
three with ``np.testing.assert_array_equal``, and write the paragraph explaining
why the integer ones get no tolerance. Eight ported types would be eight copies
of that paragraph, and the copies would drift.

**So the decision, taken once for the whole port: there is one object-level entry
point** -- :func:`assert_object_parity` -- **and no per-type list of stand-in
arrays.** The caller names the state that has to agree, as a sequence of
:class:`Field`, and each field carries its own claim. What that buys is not
brevity: it is that the *reason* travels with the comparison. An integer field
carries an :class:`ExactClaim` saying what a difference would mean, in the same
place a float field carries its rounding budget, and neither can be written
without one.

Three consequences worth knowing before writing the ninth consumer.

**Exactness is its own claim, not a fourth** :class:`ParityKind`. ``BITWISE`` is
a statement about floating point -- the same IEEE-754 operations in the same
order -- and its failure message tells the reader to go and look at
``__fp_contract__``. For a cell id that message is nonsense: there is no rounding
to fuse and no format to blame. :func:`exact_parity` is the claim for anything a
tolerance cannot be applied to, and :func:`assert_parity` stays float-only. It
compares *answers*, not containers: both sides are normalised through
:func:`numpy.asarray`, so a tuple and an equal-valued array agree. Which container
a binding hands back is a contract question and belongs where the contract is
checked, not folded into a value comparison.

**A field is an attribute by default, and a callable when it has to be.** Six of
the eight types this was designed for expose their whole state as attributes.
The other two do not: a tag registry's state is behind ``__getitem__``, keyed by
a name only known at run time, and a ragged per-axis tuple is behind an index. A
``Field`` may therefore carry a ``read`` callable, which is what lets those two
build their field list in a loop instead of falling back to hand-rolled
comparisons -- which is the cost this entry point exists to avoid. It must return
one quantity: handing back ``(keys, values)`` or a whole per-axis tuple is refused,
because those are two and *n* quantities and each wants a field and a claim of its
own. So is a value mixing element kinds, and that one is not cosmetic: an ``int64``
id beside a ``float`` weight promotes to ``float64``, where ``2**53`` and
``2**53 + 1`` are the same number, and the comparison **passes**.

**It refuses to be vacuous, in the two ways it could be.** A call with no fields
would pass for any two objects at all, and a call whose fields repeat a name
would report one comparison under another's key; both are assertion failures
rather than quiet successes. This is the same argument as
``_refuse_a_vacuous_bound``: the harness's whole design is that an assertion that
cannot fail must be impossible to write by accident. One case escapes it, and is
named rather than pretended away -- a ``read`` that ignores the object it is handed
and closes over a constant compares that constant with itself, and nothing here can
see that.

What it does *not* do is decide which fields matter. Derived conveniences
(``BVH.n_nodes`` is the length of an array already compared) may be named or
left out, and an expensive or lazily built one (``Grid.cell_bvh()``) is the
caller's to reach for. The harness has no view on what constitutes the object;
the parity module for each type does, and states it in its field list.

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

import operator
import os
from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import Any, Final, NamedTuple, cast

import numpy as np
import numpy.typing as npt
import pytest

__all__ = [
    "AccuracyClaim",
    "BuildProvenance",
    "Deviation",
    "ExactClaim",
    "Field",
    "ParityClaim",
    "ParityKind",
    "Roundings",
    "absolute_tolerance",
    "assert_accuracy",
    "assert_object_parity",
    "assert_parity",
    "bitwise_parity",
    "bounded_parity",
    "build_provenance",
    "contraction_may_fuse",
    "converged_parity",
    "cpp_backend_available",
    "demand_cpp_backend",
    "derived_accuracy",
    "exact_parity",
    "underflow_floor",
    "unit_roundoff",
]

FloatArray = npt.NDArray[np.floating[Any]]
"""Any real-valued NumPy array the backends can produce or consume."""

_REQUIRE_ENV_VAR: Final[str] = "PANTR_REQUIRE_CPP"

_REFERENCE_HOST_ENV_VAR: Final[str] = "PANTR_REFERENCE_HOST"
"""Marks the machine a liveness guard's numbers were measured on."""
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


def the_jit_is_disabled() -> bool:
    """Report whether Numba compilation is switched off for this process.

    Returns:
        bool: True when ``NUMBA_DISABLE_JIT`` is set to ``"1"``, which is the form
            Numba itself reads and the form ``make coverage`` passes.
    """
    return os.environ.get("NUMBA_DISABLE_JIT", "0") == "1"


def on_the_reference_host() -> bool:
    """Report whether this run is on the machine a liveness guard was calibrated on.

    Returns:
        bool: True when ``PANTR_REFERENCE_HOST`` is set to a non-empty value.
    """
    return bool(os.environ.get(_REFERENCE_HOST_ENV_VAR, ""))


def demand_the_compiled_kernel(dtype: npt.DTypeLike) -> None:
    """Skip a ``float32`` claim about the Python backend when its kernel is not compiled.

    Under ``NUMBA_DISABLE_JIT=1`` the Numba kernels run as interpreted Python over
    numpy, and **that is a different object from the one every bound here
    describes**. The bounds and the bitwise-parity claims were derived for the
    compiled kernel's arithmetic, so asserting them against the interpreted path
    measures something this project does not ship.

    The switch is deterministic, which is what separates this from
    :func:`demand_the_reference_host`: 29 failures, three runs, identical. It is a
    configuration difference, not a host one.

    Gated on ``float32``, which is the divergence this gate owns: the interpreted
    path is not simply computing in ``float64`` and rounding, since its ``float32``
    result matches neither the compiled ``float32`` one nor the ``float64`` one cast
    down. Which intermediates promote has not been pinned, and no claim is made about
    it here.

    **It does not follow that ``float64`` is safe, and an earlier version of this
    docstring said so.** The evidence offered was one case, a degree-5 tabulation at
    11 points whose ``float64`` output is bitwise identical with the JIT on and off.
    That measurement is true and the generalisation drawn from it was not: a bitwise
    claim breaks at ``float64`` too, by a seed that is numpy's ``np.power`` rather
    than numba's, and by an integer accumulator that grows where the compiled one
    wraps. That case belongs to :func:`demand_a_compiled_seed`,
    which gates on the claim rather than on the width, and this gate is left owning
    exactly the storage-format question it was measured for.

    Only ``make coverage`` sets the variable, so a plain ``pytest`` run is untouched.

    Args:
        dtype (npt.DTypeLike): The storage format the calling test is parametrized on.

    Raises:
        Skipped: Via :func:`pytest.skip`, for ``float32`` with the JIT disabled.
    """
    if not the_jit_is_disabled() or np.dtype(dtype) != np.float32:
        return
    pytest.skip(
        "with NUMBA_DISABLE_JIT the Python backend is interpreted, and its float32 "
        "arithmetic is not the compiled kernel's, which is what this claim's bound "
        "was derived for. float64 is bitwise unaffected and still runs."
    )


def demand_a_compiled_seed() -> None:
    """Skip a claim whose kernel diverges under interpretation beyond float rounding.

    A bitwise claim asserts that the two backends **perform the same operations in
    the same order**. Under ``NUMBA_DISABLE_JIT=1`` the Numba side performs none of
    them: it runs as interpreted Python over numpy, where at least two documented
    differences bite, both of them independent of storage width.

    The first is the seed. ``_bernstein_point``, ``_bernstein_point_no_mirror`` and
    ``_evaluate_bezier_1d_core`` all seed a ratio recurrence with ``np.power``, and
    what the affected claims rest on is that *numba's* ``np.power`` agrees with the
    platform libm. Interpreted, numpy's is called instead, and it disagrees by an
    ulp often enough to matter.

    The second is integer width, and it is the one that shows a bitwise claim can
    fail by far more than an ulp. ``_evaluate_bezier_deriv_1d_core`` and
    ``_bernstein_derivs_point`` accumulate a falling factorial in an integer that
    **wraps at int64 when compiled and grows without bound when interpreted**. Past
    the overflow the two paths differ by whole factors, not by rounding.

    **It is called per test, and deliberately not from** :func:`assert_parity`.
    Gating every bitwise claim there was tried and measured: it skips most of the
    parity suite for no gain, and it is not justified. For a kernel built from ``+``,
    ``-``, ``*``, ``/`` and ``sqrt``, IEEE 754 pins every result, so the interpreted
    path reproduces the compiled one's bits by guarantee rather than by luck, and the
    bitwise claim is as true there as anywhere. ``sqrt`` belongs in that list and is
    not an afterthought: the Legendre tabulation rests on it and is deliberately left
    ungated. Only the two constructs above escape the guarantee, and which kernels
    carry them is knowledge a claim does not hold.

    The cost of that choice, stated because it is real, and it is not only about
    kernels not yet written. ``_basis_derivs_point`` in
    ``pantr.bspline._bspline_basis_core`` **already** accumulates a falling factorial
    the same way, and is absent from the set below only because no parity test reaches
    it: it has no C++ counterpart yet. Whoever gives it one inherits this gate as a
    precondition. Grep ``np.power`` and ``fac *=`` under ``src/pantr`` for the current
    set, and note that ``_bincoeff`` is a deliberate non-member, since it casts every
    step to ``np.int64`` explicitly and so wraps identically either way.

    Raises:
        Skipped: Via :func:`pytest.skip`, whenever the JIT is disabled.
    """
    if not the_jit_is_disabled():
        return
    pytest.skip(
        "a bitwise claim names the compiled kernel's operation sequence, and with "
        "NUMBA_DISABLE_JIT the oracle is interpreted python instead: its np.power "
        "is numpy's rather than numba's, and its factorial accumulator grows where "
        "the compiled one wraps at int64. Bounded and converged claims still run."
    )


def demand_the_reference_host(guard: str, measured: str) -> None:
    """Skip a liveness guard on a machine that cannot enforce its numbers.

    **A bound is a property of the code; whether that bound is still approached is
    a property of the host.** The two are different claims and only the first
    transfers. This is what separates them.

    The guards in question assert that a bound is still doing work: that the two
    backends still disagree somewhere, that the worst observed ratio is still close
    to the bound, that the Halley iterate still reaches a two-value limit cycle.
    Each is worth having, because a bound nothing exercises can rot without any
    test noticing. But each was measured on one machine, and the quantities behind
    them are chosen at run time by the CPU: glibc dispatches ``exp`` through IFUNC
    on the processor's features, and numpy dispatches its own loops the same way.
    A different host gives a different ``exp``, and then two implementations that
    disagreed by one ulp may agree exactly.

    That is not a regression. It is a better outcome on that host, and failing the
    build for it teaches everyone to discount a red, which costs more than the
    guard ever earned. Measured: commit 767f502 was run twice on this project's CI
    with no change between the runs, and gave ``6 failed, 309 passed, 3 xfailed``
    and then ``313 passed, 5 xfailed``. Two strict xfails XPASSed in the first and
    not the second.

    So the guard keeps its teeth where its numbers came from, which
    ``scripts/ci_local.sh`` marks, and reports rather than fails anywhere else. The
    correctness assertions in the same tests are **not** gated and run everywhere.

    Args:
        guard (str): What the guard checks, for the skip reason.
        measured (str): Where and to what value it was measured, so a reader can
            re-establish it on a new reference host.

    Raises:
        Skipped: Via :func:`pytest.skip`, when not on the reference host.
    """
    if on_the_reference_host():
        return
    pytest.skip(
        f"{guard} is a property of this host rather than of the code, so it is "
        f"enforced only where it was calibrated ({measured}). Set "
        f"{_REFERENCE_HOST_ENV_VAR} to enforce it here; scripts/ci_local.sh does."
    )


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
    # the provenance attributes are declared in src/pantr/_pantr_cpp/__init__.pyi, so this
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
        CONVERGED: The backends differ, within a bound that comes from the
            **algorithm** rather than from arithmetic: both ran the same
            convergent iteration to the same termination criterion, so each
            answer lies within that criterion of the quantity it converged to.
            Such a bound does not scale with epsilon, does not grow with degree,
            and would hold between two runs of the *same* backend that happened
            to take different iteration paths. Added by the root-finding port,
            which is the first kernel family whose difference is not a rounding
            difference. See ``design/backend_parity.md`` Rule 11.
    """

    BITWISE = 0
    BOUNDED = 1
    CONVERGED = 2


class Roundings(NamedTuple):
    """The rounding budget of one kernel, as a count rather than a magnitude.

    Counting roundings is the part a reader can check against the source; turning
    a count into a magnitude is arithmetic the harness does. Splitting the count
    by format is what makes one budget serve float32 and float64 storage over a
    float64 accumulator, which is the shape every pantr kernel has.

    **This models a dependency chain, and not every bound is one.** A claim whose
    bound is a derived ratio rather than a count -- the displacement of a Newton
    root, or a library function's error in units in the last place amplified by
    something -- has no honest stage count to give. Those are written as
    ``Roundings(1, 1, 0)``, which reduces this record to a way of spelling the
    unit ``u`` and puts the whole derivation in ``amplification`` and ``why``.
    ``design/backend_parity.md`` records that as a known cost of the design: a
    reader taking the field at its word would conclude such a kernel commits one
    rounding, when it may commit thousands that cancel. ``why`` is mandatory and
    is quoted verbatim in every failure message for exactly this reason.

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

    Built by :func:`bitwise_parity`, :func:`bounded_parity` or
    :func:`converged_parity`; do not construct it positionally.

    Attributes:
        kind (ParityKind): Which of the three statements is being made.
        roundings (Roundings | None): The rounding budget. None for BITWISE.
        accumulator (np.dtype[np.floating[Any]] | None): Format intermediates are
            accumulated in. None for BITWISE.
        storage (np.dtype[np.floating[Any]] | None): Format the output array
            holds. None for BITWISE.
        amplification (FloatArray | None): Elementwise factor the relative
            budget is multiplied by to reach an absolute tolerance. It carries
            **two things at once**: the dimensionless amplification the
            computation applies to a relative perturbation, and the magnitude
            that converts relative to absolute. They coincide numerically
            wherever the result is of order one, which is why one array serves
            both; where the result spans decades, or vanishes while its error
            does not, the magnitude has to be multiplied in deliberately. See
            ``design/backend_parity.md`` Rules 2 and 4. None for BITWISE.
        scale (float | None): The magnitude the quantity is meaningful against,
            used **only** by CONVERGED. For most kernels a bound is vacuous when it
            reaches the values it compares, which is what the vacuity guard checks.
            A quantity confined to a **bounded domain** breaks that premise: a curve
            parameter near zero is not an ill-determined quantity, it is a parameter
            near an endpoint, and its natural scale is the domain rather than its own
            magnitude. Where a claim carries a scale, the guard takes the larger of
            the two. None otherwise.
        bound (FloatArray | None): Elementwise absolute bound, used **only**
            by CONVERGED, where the bound is the algorithm's own termination
            criterion and no rounding model produces it. None otherwise. Kept
            separate from ``amplification`` rather than overloading it, because
            the two are different quantities: one is dimensionless and gets
            multiplied by a rounding budget, the other is already a tolerance.
        why (str): The derivation, or the reason exactness holds. Quoted verbatim
            in any failure message.
    """

    kind: ParityKind
    roundings: Roundings | None
    accumulator: np.dtype[np.floating[Any]] | None
    storage: np.dtype[np.floating[Any]] | None
    amplification: FloatArray | None
    why: str
    bound: FloatArray | None = None
    scale: float | None = None


class AccuracyClaim(NamedTuple):
    """An elementwise absolute error bound against an independent oracle.

    Built by :func:`derived_accuracy`.

    Attributes:
        bound (FloatArray): Elementwise absolute bound, same shape as the result.
        why (str): The derivation. Quoted verbatim in any failure message.
    """

    bound: FloatArray
    why: str


class ExactClaim(NamedTuple):
    """A claim that two backends must agree exactly, with the reason.

    The counterpart of :class:`ParityClaim` for everything a tolerance cannot be
    applied to: a cell id, a node index, a count, a flag, a tag name, a dtype.
    ``design/backend_parity.md`` Rule 11 is the distinction -- a differing verdict
    is not a displaced value, and no bound could absorb it, so there is nothing to
    derive and nothing but a reason to state.

    Kept separate from :class:`ParityClaim` rather than added to
    :class:`ParityKind`, because :func:`assert_parity` is float-only by
    construction: it compares bit patterns through a float view and computes a
    difference in ``float64``. Its BITWISE message is also wrong here -- it tells
    the reader to check ``__fp_contract__``, which decides nothing about an
    integer.

    Built by :func:`exact_parity`.

    Attributes:
        why (str): Why exactness is the right claim, and what a difference would
            mean. Quoted verbatim in any failure message.
    """

    why: str


class Field(NamedTuple):
    """One comparable piece of an object, and the claim that governs it.

    Built directly; it carries no derivation of its own, only the claim it points
    at. See :func:`assert_object_parity`.

    Attributes:
        name (str): The attribute read from both objects, and the label a failure
            message uses. Where ``read`` is given the name is only the label, and
            should still say what was compared (``"boundary.ids"``,
            ``"breakpoints[1]"``).
        claim (ParityClaim | ExactClaim): What agreement is claimed for this piece.
        read (Callable[[Any], Any] | None): How to take the value out of an
            object, for the pieces that are not a plain attribute: a tag registry
            is a mapping reached through ``__getitem__``, and a per-axis array
            lives behind an index into a ragged tuple. Defaults to
            ``operator.attrgetter(name)``, which is ``getattr(obj, name)`` for a plain
            name and also follows a dotted one (``"cell_tags.num_cells"``). It must
            return **one** quantity: a sequence of arrays or of sequences is refused,
            and so is a value mixing element kinds, since those are several quantities
            and want several fields. A flat tuple of one kind stays one quantity
            (``names``, ``cells_per_axis``). A ``read`` that ignores its argument and
            closes over a constant is the one vacuous field nothing here can detect.
    """

    name: str
    claim: ParityClaim | ExactClaim
    read: Callable[[Any], Any] | None = None


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
        amplification (FloatArray): Elementwise factor the relative budget is
            multiplied by to reach an absolute tolerance, same shape as the
            result. Carries the dimensionless amplification **and** the magnitude
            that converts relative to absolute; the two coincide only where the
            result is of order one. Must be finite and non-negative.
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


def converged_parity(*, bound: FloatArray, scale: float, why: str) -> ParityClaim:
    """Claim that two backends agree within their shared termination criterion.

    For an iterative kernel whose difference is **not** a rounding difference. Both
    backends run the same convergent iteration and stop on the same criterion, so
    each answer lies within that criterion of the quantity it converged to and the
    two lie within it of each other. Nothing about floating point enters, which is
    why this cannot be spelled with :func:`bounded_parity`: that builds its bound
    from a rounding budget times an amplification, and reaching a given number that
    way would mean choosing an amplification to produce it, which is the fitted
    constant the harness exists to make unsayable.

    **What this claim does not cover.** It assumes the two backends converged to the
    *same* quantity. Where an iteration's control flow can branch differently, that
    assumption is separate and has to be asserted separately, because no tolerance
    bounds a changed verdict.

    Args:
        bound (FloatArray): Elementwise absolute bound, same shape as the result.
            Normally the iteration's termination tolerance.
        scale (float): The magnitude the quantity is meaningful against, which for
            an iterate on a bounded domain is the domain's width rather than the
            iterate's own value. Required rather than defaulted: getting it wrong is
            how a vacuous bound passes, and there is no scale that is right for every
            iteration.
        why (str): The derivation: which criterion, and why each answer lies within
            it of what it converged to.

    Returns:
        ParityClaim: A claim asserting agreement within the algorithmic bound.

    Raises:
        ValueError: If ``why`` is empty, if the bound is not finite and non-negative,
            or if the scale is not finite and positive.
    """
    if not why.strip():
        raise ValueError("a converged parity claim must carry its derivation")
    arr = np.asarray(bound, dtype=np.float64)
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("the convergence bound must be finite and non-negative")
    if not (np.isfinite(scale) and scale > 0.0):
        raise ValueError("the scale must be finite and positive")
    return ParityClaim(
        kind=ParityKind.CONVERGED,
        roundings=None,
        accumulator=None,
        storage=None,
        amplification=None,
        why=why,
        bound=arr,
        scale=float(scale),
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


def exact_parity(*, why: str) -> ExactClaim:
    """Claim that two backends agree exactly, on a quantity no tolerance applies to.

    For integer and boolean arrays, counts, flags, names and dtypes. Not for
    floats: :func:`bitwise_parity` is the exactness claim there, and it says
    something stronger and more fragile -- that the same IEEE-754 operations ran
    in the same order.

    Args:
        why (str): Why exactness holds and what a difference would mean.

    Returns:
        ExactClaim: The claim.

    Raises:
        ValueError: If ``why`` is empty.
    """
    if not why.strip():
        raise ValueError("an exactness claim must carry its reason")
    return ExactClaim(why=why)


_LU_ALLOWANCE: Final = 8
"""Allowance for ``R = || |L||U| ||_inf / ||A||_inf`` in Higham Thm 9.4.

**Measured, not argued.** Exact rational arithmetic over every matrix these
builders factor, across every solvability domain, gives ``R <= 3.73``, with the
classical growth factor ``rho_n`` exactly ``1.000``; an independent sweep of 1508
matrices to degree 200 through Eigen's own ``matrixLU()`` agrees and finds ``R``
crossing 8 only near degree 160. So the margin is 2.1x in domain, and the allowance
stops covering anything past degree 160 -- which no builder here reaches, and which
is the caveat to carry if one ever does.

Deliberately not tightened to 4: measured, that buys one degree on three of ten
(builder, dtype) pairs, because ``kappa_inf`` grows geometrically and is what
actually limits the parity domain.


**Its domain is part of it.** The figures above are measured over the change-of-basis
builders' own matrices. A second consumer must measure ``R`` for ITS matrices before
reusing this, because the constant denotes a growth factor and not a number:
``tests/parity/test_transform_affine.py`` did so and found ``R <= 4.391`` over random
normal matrices to ``n = 6``, which this covers with 1.8x to spare. Reusing it without
that check would be borrowing a value while dropping the quantity it measures.
"""


ONE_SIDED_TO_TWO_SIDED: Final[int] = 2
"""Factor turning a one-sided forward-error bound into a two-sided parity bound.

Neither backend is the exact answer. Each sits within its own forward-error bound
of the exact recurrence, so their difference is bounded by the sum of the two.
The factor is 2 because the two bounds are equal, and it is a derivation rather
than a safety margin: dropping it would be claiming one backend rounds perfectly.
"""


def _relative_growth(claim: ParityClaim) -> float:
    """Accumulate a claim's per-stage rounding budget over its stages.

    Uses Higham's closed form ``gamma_m = m u / (1 - m u)`` on the total budget
    ``m u = stages * per_stage``, rather than the algebraically equivalent
    ``(1 + per_stage)**stages - 1``.

    **The two are not equivalent in floating point, and the second one is
    unusable.** ``1 + eps/2`` sits exactly at the midpoint between ``1`` and
    ``1 + eps``, and round-half-to-even carries it back down to ``1`` because
    ``1``'s significand is even. So for any float64 budget of one rounding per
    stage -- ``per_stage = u`` -- the power form evaluates to ``(1.0)**stages -
    1.0``, which is **exactly zero at every stage count**. The bound then
    collapses onto the underflow floor alone, about ``1e-323``, and a claim that
    says BOUNDED asserts bit-for-bit agreement instead. Measured on the shipped
    implementation before this was changed: one stage and four stages both
    returned ``0.0``.

    The guard below did not catch it, and the reason is worth keeping: it tests
    ``per_stage == 0``, which is ``1.11e-16`` here and passes. A budget can be
    non-zero and still produce a zero bound. Higham's form cannot: its
    denominator stays near 1 and the quotient is representable.

    That the docstring already said "the standard ``gamma`` of a forward-error
    analysis" while the code computed something else is the whole of the defect.
    The standard gamma *is* ``m u / (1 - m u)``, and it is an upper bound on the
    power form rather than an approximation to it, so nothing that passed before
    can fail now.

    Args:
        claim (ParityClaim): A BOUNDED claim.

    Returns:
        float: ``gamma_m = m u / (1 - m u)``, the standard ``gamma`` of a
            forward-error analysis, with ``m u`` the budget accumulated over the
            claim's stages.

    Raises:
        ValueError: If the budget commits no rounding at all, or is so large that
            the bound carries no information (a relative bound at or above 1
            accepts everything).
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
    total = claim.roundings.stages * per_stage
    if total >= 0.5:
        raise ValueError(
            f"the rounding budget accumulates to {total:.3g}, at or past the half "
            f"where gamma stops being a useful bound and runs away to 1. "
            f"{claim.roundings.stages} stages of {per_stage:.3g} exhaust the format; "
            f"the comparison would be vacuous."
        )
    growth = total / (1.0 - total)
    if growth <= 0.0:
        raise ValueError(
            f"the rounding budget {claim.roundings} accumulated to a relative bound of "
            f"{growth!r}, which asserts bit-for-bit agreement while claiming BOUNDED. "
            f"A positive budget reaching zero here means the arithmetic underflowed; "
            f"use bitwise_parity if exactness is what is meant."
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
        FloatArray: Zero everywhere for BITWISE; the stated bound for CONVERGED;
            otherwise
            ``2 * (gamma * amplification + underflow floor)``, the two halves of
            the rounding model that :func:`underflow_floor` describes. Both are
            doubled for the same reason: neither backend is the exact answer, so
            the parity bound is twice a one-sided forward-error bound.
    """
    if claim.kind is ParityKind.BITWISE:
        shape = () if claim.amplification is None else claim.amplification.shape
        return np.zeros(shape, dtype=np.float64)
    if claim.kind is ParityKind.CONVERGED:
        assert claim.bound is not None  # CONVERGED by construction
        # Already a tolerance rather than an amplification, and **not** doubled:
        # the factor of two turns a one-sided forward-error bound into a two-sided
        # one, and this bound is two-sided to begin with. Each backend's answer is
        # within the criterion of the quantity it converged to, and the criterion
        # is the same one, so the difference is bounded by it directly.
        return claim.bound
    assert claim.amplification is not None  # BOUNDED by construction
    relative = _relative_growth(claim) * claim.amplification
    return ONE_SIDED_TO_TWO_SIDED * (relative + _underflow_budget(claim))


def _the_oracle_may_not_be_compiled() -> str:
    """Name the third possibility, when a bitwise failure could be a missing gate.

    The message this appends to offers two causes, the port and the build, and both
    are wrong when the real one is that the oracle was interpreted. That misdirection
    is not hypothetical: it is what the float64 bitwise failures under
    ``make coverage`` reported before :func:`demand_a_compiled_seed` existed, and a
    kernel added later that seeds with ``pow`` and forgets the gate gets it again.

    Returns:
        str: A sentence to append while the JIT is disabled, empty otherwise.
    """
    if not the_jit_is_disabled():
        return ""
    return (
        "\n  But NUMBA_DISABLE_JIT is set, so the oracle is interpreted python and "
        "not the kernel this claim names. If this kernel seeds with pow or "
        "accumulates an integer, it is missing a demand_a_compiled_seed() call and "
        "neither the port nor the build has changed."
    )


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
        AssertionError: If the bound reaches the scale the quantity lives on.
    """
    # A claim that names its own scale is compared against the larger of the two.
    # The default premise, that a bound is vacuous once it reaches the values it
    # compares, fails for a quantity confined to a bounded domain: a curve parameter
    # near zero is not ill-determined, it is near an endpoint. See ParityClaim.scale.
    largest = float(np.abs(reference.astype(np.float64)).max(initial=0.0))
    if claim.scale is not None:
        largest = max(largest, claim.scale)
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
                f"{_the_oracle_may_not_be_compiled()}"
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
            + (
                ""
                if claim.roundings is None
                else f"  budget: {claim.roundings}, accumulator {claim.accumulator}, "
                f"storage {claim.storage}\n"
            )
            + f"{_worst_element_report(actual, reference, tolerance, offenders)}"
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


def _as_comparable(value: Any, *, context: str) -> npt.NDArray[Any]:
    """Turn one field's value into an array, refusing the shapes that would mislead.

    A ``read`` that hands back several pieces at once is the mistake this catches, and
    it is an easy one to make: ``FacetTags[name]`` is ``(keys, values)`` of shapes
    ``(M, 2)`` and ``(M,)``, and ``TensorProductGrid.breakpoints`` is a per-axis tuple
    whose entries have different lengths. Both are two quantities, not one.

    Left to :func:`numpy.asarray` they fail three ways, and the quiet ones are worse.
    Ragged pieces raise a bare ``ValueError`` from deep inside NumPy, naming no field
    and no backend. Equal-length pieces -- ``CellTags[name]`` is ``(ids, values)`` of
    the same length -- stack silently into one ``(2, N)`` block, so the comparison
    passes or fails as a unit and the failure message points at a row index of a thing
    the caller never built. And **pieces of different kinds promote**: a tuple mixing an
    ``int64`` id with a ``float`` weight becomes ``float64``, where ``2**53`` and
    ``2**53 + 1`` are the same number. Measured: that pair passes
    :func:`assert_object_parity` under a ``bitwise`` claim, silently, which is a false
    pass rather than a bad message. So all three are refused, with the fix named.

    **A flat tuple of one kind is left alone, and that is a deliberate limit.**
    ``CellTags.names`` is ``tuple[str, ...]``, ``TensorProductGrid.cells_per_axis`` and
    ``Bezier.degree`` are ``tuple[int, ...]`` -- each one quantity with several
    components. Nothing distinguishes those from two scalars a ``read`` glued together,
    so refusing them would break four attribute fields across three of the eight
    consumers to catch a case that compares correctly anyway: a homogeneous tuple is
    compared elementwise with no promotion, so the only cost is a failure message that
    names a component index rather than a field.

    Args:
        value (Any): What a :class:`Field` read off one of the two objects.
        context (str): What was being compared, quoted in a failure message.

    Returns:
        npt.NDArray[Any]: The value as an array. A scalar becomes 0-d.

    Raises:
        AssertionError: If the value is a sequence of arrays or of sequences, if it
            mixes element kinds, or if it cannot become one array at all.
    """
    if isinstance(value, tuple | list):
        kinds = set()
        for item in value:
            if isinstance(item, np.ndarray | tuple | list):
                raise AssertionError(
                    f"{context}: the field read back a {type(value).__name__} of "
                    f"{len(value)} arrays or sequences, which is that many quantities "
                    f"rather than one. Give each its own Field, with its own claim and "
                    f"its own `read`."
                )
            kinds.add(np.asarray(item).dtype.kind)
        if len(kinds) > 1:
            raise AssertionError(
                f"{context}: the field read back a {type(value).__name__} mixing "
                f"{sorted(kinds)} in one value. Those are different quantities, and "
                f"stacking them promotes to a common dtype that can lose the "
                f"difference between two of them -- int64 above 2**53 against a float "
                f"is the measured case, where two distinct ids compare EQUAL and the "
                f"comparison passes. Give each its own Field."
            )
    try:
        return np.asarray(value)
    except ValueError as exc:
        raise AssertionError(
            f"{context}: the field's value cannot be compared as a single array "
            f"({exc}). A ragged or compound value needs one Field per piece."
        ) from exc


def _assert_exact(
    actual: Any,
    reference: Any,
    claim: ExactClaim,
    *,
    context: str,
) -> Deviation:
    """Assert two values agree exactly, for the quantities a tolerance cannot cover.

    One call handles an array of any dtype, a plain ``int``, ``bool`` or ``str``, a
    tuple of them, and a dtype object, because both sides go through
    :func:`numpy.asarray` first and are then compared elementwise. A scalar becomes
    a 0-d array, and a differing *shape* is reported before any element is looked
    at: two results of different length are a changed verdict, which an elementwise
    comparison cannot see at all.

    **Normalising both sides is deliberate, and it is a decision about what parity
    means.** A backend is free to return a tuple where the oracle returns a list, or
    a NumPy array where it returns a tuple; that is a difference in the *container*
    and not in the answer, and this harness compares answers. What the two backends
    return as types is a binding-contract question, checked where that contract is
    (``tests/parity/test_bezier_binding_contract.py`` is the pattern), not smuggled
    into a value comparison where it would fire on a container nobody chose.

    Args:
        actual (Any): The backend under test, conventionally the C++ one.
        reference (Any): The oracle, conventionally the Python one.
        claim (ExactClaim): Why exactness holds. Quoted in a failure message.
        context (str): What was being compared.

    Returns:
        Deviation: All zeros, since the only outcome that returns is agreement.
            Present so a caller can treat every field uniformly.

    Raises:
        AssertionError: If the two differ in shape or in any element.
    """
    actual_array = _as_comparable(actual, context=context)
    reference_array = _as_comparable(reference, context=context)
    for array in (actual_array, reference_array):
        # `size` guards the empty edge: np.asarray(()) is float64, and a tag registry
        # with no tags has an empty `names`. There is no NaN in an empty array to be
        # wrong about.
        if array.dtype.kind == "f" and array.size > 0:
            raise AssertionError(
                f"{context}: an exact claim was made about a {array.dtype} value. "
                f"Exactness on floating point is bitwise_parity, which says the "
                f"stronger and more fragile thing -- the same IEEE-754 operations in "
                f"the same order -- and which handles NaN through the bit pattern "
                f"rather than through `!=`, where every NaN differs from itself."
            )
    assert actual_array.shape == reference_array.shape, (
        f"{context}: shape {actual_array.shape} against {reference_array.shape}. "
        f"Two results of different shape are a changed verdict, which no comparison "
        f"of elements can see at all.\n  claim: {claim.why}"
    )

    offenders = np.asarray(actual_array != reference_array)
    num_differing = int(np.count_nonzero(offenders))
    if not num_differing:
        return Deviation(max_absolute=0.0, max_ratio_to_bound=0.0, num_differing=0)

    if actual_array.ndim == 0:
        where = ""
        detail = f"{actual!r} against {reference!r}"
    else:
        first = tuple(int(i) for i in np.argwhere(offenders)[0])
        where = f" in {num_differing} of {actual_array.size} entries"
        detail = f"first at {first}: {actual_array[first]!r} against {reference_array[first]!r}"
    raise AssertionError(
        f"{context}: exact agreement claimed and violated{where}.\n"
        f"  claim: {claim.why}\n"
        f"  {detail}"
        f"{_the_oracle_may_not_be_compiled()}"
    )


def assert_object_parity(
    py: Any,
    cpp: Any,
    *,
    fields: Sequence[Field],
    context: str,
) -> dict[str, Deviation]:
    """Assert that two backends' versions of one object agree, field by field.

    The object-level entry point. See "Comparing two objects" in the module
    docstring for why it exists and what it deliberately does not do.

    **The argument order is ``(py, cpp)``**, the opposite of
    :func:`assert_parity`'s ``(actual, reference)``, and the parity modules do not
    agree on which order their two-backend helper returns: ``tests/parity/test_grid.py``
    has ``_both()`` returning ``(reference, actual)``, while ``test_change_basis.py``
    and ``test_basis_cardinal_bspline.py`` have ``_both_backends()`` returning
    ``(actual, reference)``. So
    ``assert_object_parity(*_both(...), ...)`` is right for the first and wrong for
    the other two, and **the order has to be checked at the call site** rather than
    assumed. Passing ``py=`` and ``cpp=`` by keyword removes the question.

    A swap does not flip the verdict -- every comparison here is symmetric in its two
    arguments, and no bound is derived from either -- but it does swap which value a
    failure message calls "actual" and which it calls "reference", which misleads
    whoever reads it about the backend that produced the number.

    Args:
        py (Any): The object the Python backend built. The oracle.
        cpp (Any): The object the C++ backend built. The one under test.
        fields (Sequence[Field]): Which pieces of state have to agree, and under
            which claim. Must be non-empty and must not repeat a name.
        context (str): What was being built, quoted in every failure message.
            Each field appends its own name to it.

    Returns:
        dict[str, Deviation]: What each field's comparison observed, keyed by
            field name, so a caller can assert on a margin as well as on the pass.
            Exact fields report zeros.

    Raises:
        AssertionError: If ``fields`` is empty or repeats a name, or if any field
            violates its claim.
    """
    assert fields, (
        f"{context}: assert_object_parity was given no fields, so it compares "
        f"nothing and would pass for any two objects at all. Name the state that "
        f"has to agree, or delete the call."
    )
    names = [field.name for field in fields]
    assert len(set(names)) == len(names), (
        f"{context}: two fields share a name in {names}. The result is keyed by "
        f"name, so one of the two comparisons would be reported as the other."
    )

    observed: dict[str, Deviation] = {}
    for field in fields:
        read = operator.attrgetter(field.name) if field.read is None else field.read
        where = f"{context}: {field.name}"
        if isinstance(field.claim, ExactClaim):
            observed[field.name] = _assert_exact(read(cpp), read(py), field.claim, context=where)
        else:
            observed[field.name] = assert_parity(
                np.atleast_1d(_as_comparable(read(cpp), context=where)),
                np.atleast_1d(_as_comparable(read(py), context=where)),
                field.claim,
                context=where,
            )
    return observed
