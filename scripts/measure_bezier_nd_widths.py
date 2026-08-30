#!/usr/bin/env python
"""Measure the accumulation width of the n-dimensional Bezier evaluation oracle.

``design/backend_parity.md`` Rule 9 says an oracle's accumulation width is a
**per-kernel fact, not a module convention**, and that a width read off the source
rather than measured is how a port ships exact at float64 and quietly wrong at
float32. The n-d Bezier layer is where that rule bites hardest, because its oracle
is **not** a Numba kernel: it is numpy, and two different numpy functions at that.

    PYTHONPATH="$(pwd)/src" python scripts/measure_bezier_nd_widths.py

Two facts this script exists to establish, both of which decide what the C++ port
in ``cpp/include/pantr/bezier/evaluate.hpp`` may claim.

**One: the house accumulator policy does not transfer.** ``accumulator_t<float>`` is
``double`` (``cpp/include/pantr/core/scalar.hpp``) because *Numba* promotes a float64
scalar against a float32 array. Nothing in the n-d path is Numba. The contraction is
``np.einsum`` or ``np.tensordot`` over arrays that ``_bezier_eval.py`` *requires* to
share the Bezier's dtype, so at float32 the oracle contracts in float32 and a C++
kernel inheriting the house policy would be computing in the wrong arithmetic.

**Two: the two n-d entry points are two different oracles.**
``_evaluate_bezier_nd_pts_array`` contracts with ``np.einsum``;
``_evaluate_bezier_nd_lattice`` contracts with ``np.tensordot``, which reshapes to a
matrix product and reaches BLAS. They are the same mathematics and different
arithmetic, so one C++ kernel cannot be exact against both and each entry point
carries its own parity claim.

What the measurement found
--------------------------

**The einsum path accumulates narrow, in the naive index order -- except where its
trailing block holds exactly one element.** For every output shape whose
non-contracted tail has two or more entries, a left-to-right accumulation in the
storage dtype reproduces ``np.einsum`` bit for bit, at both dtypes and every degree
swept. Where the tail is a single element the contraction collapses to a contiguous
dot product and numpy takes a vectorised reduction with a different summation tree.
The narrow ascending model then reproduces it only while the contraction is short
enough that the vectorised path degenerates -- it starts failing at length 4 in
float32 and at length 3 in float64, and a float64 accumulator reproduces none of it.
That case is not exotic: it is the last contraction of every scalar-valued
non-rational Bezier.

**The tensordot path never accumulates in the naive order**, at either dtype and at
every shape swept, because it is BLAS.

So neither entry point admits a bitwise claim, and both admit the same *kind* of
bound for different reasons. The summation order of a length-``n`` contraction is
not fixed by either oracle, but its forward error is: any order commits at most
``n`` roundings per term, so Higham's ``gamma_n`` against the absolute-value
companion bounds it, and the companion is exact rather than conservative here
because a Bernstein basis is non-negative. ``tests/parity/test_bezier_evaluate.py``
carries the two derivations.

**The spread between the two entry points is reported rather than quoted here.** A
measured number in a permanent artifact that nothing re-measures rots while reading
as current; the figure belongs in this script's output, with the numpy version and
the commit it was run against, which :func:`report_provenance` prints.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt

from pantr.bezier._bezier_utils import _tabulate_bernstein_1d_fast

ONE_SIDED_TO_TWO_SIDED: Final = 2
"""Neither the port nor the oracle is exact, so their difference is bounded by the
sum of the two one-sided forward-error bounds. Spelled here rather than imported
from ``tests/_parity_harness.py`` so this script does not depend on the test tree;
the two must agree, and ``tests/parity/test_bezier_evaluate.py`` is where the claim
that consumes it lives."""

_SEED: Final = 20260830
"""Fixed so a rerun reproduces the table rather than a similar one.

Every measurement gets its **own** generator derived from this, rather than all of
them drawing from one, so that adding a site does not silently move every figure
below it."""

_DTYPES: Final = (np.float32, np.float64)
"""Both storage formats a Bezier may hold. The widths coincide at float64 for the
sites that are naive, and separate at float32 for the ones that are not, which is
why every measurement runs at both."""

_DEGREES: Final = ((1, 1), (2, 3), (3, 2), (5, 4), (8, 2), (2, 2, 2), (4, 3, 2))
"""Degree tuples spanning dim 2 and dim 3, small and large.

The contraction length is ``degree + 1`` per direction, and the summation order is
what is being measured, so the spread of lengths matters more than the spread of
values."""

_RANKS: Final = (1, 2, 3)
"""Output ranks. Rank 1 is the one that matters: a non-rational scalar field
contracts to a trailing block of one element, which is where numpy leaves the naive
order."""

_TRIALS: Final = 24
"""Draws per (degree, rank, dtype) combination. The widths do not need many samples
to separate -- what needs samples is the *rival* count, which is what makes a match
mean something."""

_FloatArray = npt.NDArray[np.float32 | np.float64]
"""A control-point or basis array at either storage format."""

_Schedule = Callable[[Sequence[_FloatArray], _FloatArray], _FloatArray]
"""A contraction schedule: per-direction bases and a control net to raw values."""

_PTS_ARRAY_SITE: Final = "_bezier_eval.py:191-193"
"""The einsum contraction, over an explicit array of points."""

_SCALAR_TAIL_SITE: Final = "_bezier_eval.py:193 rank 1"
"""The same contraction where the trailing block holds one element."""

_LATTICE_SITE: Final = "_bezier_eval.py:251-253"
"""The tensordot contraction, over a lattice."""


class Verdict(NamedTuple):
    """One width hypothesis, measured against the oracle.

    Attributes:
        site (str): Where the operation lives, as ``file:line`` or an entry-point
            name.
        hypothesis (str): The accumulation width and order this model assumes.
        matched (int): Cases in which the model reproduced the oracle bit for bit.
        total (int): Cases run.
        rivals_differ (int): Cases in which this model and its rival disagree. A
            hypothesis whose rival never disagrees has not been discriminated, and
            the match count means nothing.
    """

    site: str
    hypothesis: str
    matched: int
    total: int
    rivals_differ: int


def bits(array: _FloatArray) -> npt.NDArray[np.signedinteger]:
    """Reinterpret a float array as its integer bit patterns, so equality is bitwise.

    Args:
        array (npt.NDArray[np.float32 | np.float64]): The array to reinterpret.

    Returns:
        npt.NDArray[np.signedinteger]: The IEEE 754 bit patterns, same shape.
    """
    view = np.int32 if array.dtype == np.float32 else np.int64
    return np.ascontiguousarray(array).view(view)


def same_bits(left: _FloatArray, right: _FloatArray) -> bool:
    """Compare two arrays bit for bit.

    Args:
        left (npt.NDArray[np.float32 | np.float64]): One array.
        right (npt.NDArray[np.float32 | np.float64]): The other, same shape and
            dtype.

    Returns:
        bool: True when every bit pattern agrees.
    """
    return bool(np.array_equal(bits(left), bits(right)))


# ---------------------------------------------------------------------------
# The two contraction schedules, as the oracle runs them
# ---------------------------------------------------------------------------


def oracle_pts_array(bases: Sequence[_FloatArray], ctrl: _FloatArray) -> _FloatArray:
    """Run the pts-array schedule exactly as ``_bezier_eval.py:191-193`` does.

    Args:
        bases (Sequence[npt.NDArray[np.float32 | np.float64]]): One Bernstein
            tabulation per direction, each of shape ``(n_pts, degree + 1)``.
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points, shape
            ``(*degrees_plus_1, cp_size)``.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Raw values of shape
        ``(n_pts, cp_size)``.
    """
    result: _FloatArray = np.einsum("pi,i...->p...", bases[0], ctrl)
    for basis in bases[1:]:
        result = np.einsum("pj,pj...->p...", basis, result)
    return result


def oracle_lattice(bases: Sequence[_FloatArray], ctrl: _FloatArray) -> _FloatArray:
    """Run the lattice schedule exactly as ``_bezier_eval.py:251-253`` does.

    Args:
        bases (Sequence[npt.NDArray[np.float32 | np.float64]]): One Bernstein
            tabulation per direction, each of shape ``(m_d, degree_d + 1)``.
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points, shape
            ``(*degrees_plus_1, cp_size)``.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Raw values of shape
        ``(*grid_shape, cp_size)``.
    """
    result = ctrl
    for direction, basis in enumerate(bases):
        result = np.tensordot(basis, result, axes=([1], [direction]))
        result = np.moveaxis(result, 0, direction)
    return result


# ---------------------------------------------------------------------------
# The rival models
# ---------------------------------------------------------------------------


def narrow_pts_array(bases: Sequence[_FloatArray], ctrl: _FloatArray) -> _FloatArray:
    """Contract left to right, every operation in the storage format.

    This is the transliteration a C++ kernel writes: one accumulator per output
    element in the Bezier's own dtype, terms added in ascending index order.

    Args:
        bases (Sequence[npt.NDArray[np.float32 | np.float64]]): Per-direction bases.
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Raw values, same shape as
        :func:`oracle_pts_array` returns.
    """
    dtype = ctrl.dtype
    result = _contract_first_narrow(bases[0], ctrl)
    for basis in bases[1:]:
        n_terms = basis.shape[1]
        accumulator = np.zeros((basis.shape[0], *result.shape[2:]), dtype=dtype)
        for term in range(n_terms):
            weight = basis[:, term].reshape((-1, *([1] * (result.ndim - 2))))
            accumulator = (accumulator + weight * result[:, term]).astype(dtype)
        result = accumulator
    return result


def _contract_first_narrow(basis: _FloatArray, ctrl: _FloatArray) -> _FloatArray:
    """Contract direction 0 of the control net against a per-point basis, narrow.

    Args:
        basis (npt.NDArray[np.float32 | np.float64]): Shape ``(n_pts, n_0)``.
        ctrl (npt.NDArray[np.float32 | np.float64]): Shape ``(n_0, ...)``.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Shape ``(n_pts, ...)``.
    """
    dtype = ctrl.dtype
    accumulator = np.zeros((basis.shape[0], *ctrl.shape[1:]), dtype=dtype)
    for term in range(basis.shape[1]):
        weight = basis[:, term].reshape((-1, *([1] * (ctrl.ndim - 1))))
        accumulator = (accumulator + weight * ctrl[term]).astype(dtype)
    return accumulator


def wide_pts_array(bases: Sequence[_FloatArray], ctrl: _FloatArray) -> _FloatArray:
    """Contract the same schedule in float64 and round once, on the final store.

    The rival hypothesis, and the one the house accumulator policy would have
    produced had it been inherited. It runs the **same** left-to-right schedule as
    :func:`narrow_pts_array`, so the only thing separating the two is the width --
    which is why the pair says nothing at float64, where they are one function.

    Args:
        bases (Sequence[npt.NDArray[np.float32 | np.float64]]): Per-direction bases.
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Raw values in the storage format.
    """
    wide = [basis.astype(np.float64) for basis in bases]
    return narrow_pts_array(wide, ctrl.astype(np.float64)).astype(ctrl.dtype)


def narrow_lattice(bases: Sequence[_FloatArray], ctrl: _FloatArray) -> _FloatArray:
    """Contract the lattice schedule left to right in the storage format.

    Args:
        bases (Sequence[npt.NDArray[np.float32 | np.float64]]): Per-direction bases.
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Raw values, same shape as
        :func:`oracle_lattice` returns.
    """
    dtype = ctrl.dtype
    result = ctrl
    for direction, basis in enumerate(bases):
        moved = np.moveaxis(result, direction, 0)
        accumulator = np.zeros((basis.shape[0], *moved.shape[1:]), dtype=dtype)
        for term in range(basis.shape[1]):
            weight = basis[:, term].reshape((-1, *([1] * (moved.ndim - 1))))
            accumulator = (accumulator + weight * moved[term]).astype(dtype)
        result = np.moveaxis(accumulator, 0, direction)
    return np.ascontiguousarray(result)


def wide_lattice(bases: Sequence[_FloatArray], ctrl: _FloatArray) -> _FloatArray:
    """Contract the lattice schedule in float64 and round once, on the final store.

    Like :func:`wide_pts_array`, it runs :func:`narrow_lattice`'s schedule rather
    than the oracle's, so the pair isolates the width and nothing else.

    Args:
        bases (Sequence[npt.NDArray[np.float32 | np.float64]]): Per-direction bases.
        ctrl (npt.NDArray[np.float32 | np.float64]): Control points.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Raw values in the storage format.
    """
    wide = [basis.astype(np.float64) for basis in bases]
    return narrow_lattice(wide, ctrl.astype(np.float64)).astype(ctrl.dtype)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _draw_case(
    rng: np.random.Generator,
    degrees: tuple[int, ...],
    rank: int,
    dtype: npt.DTypeLike,
    n_pts: int,
) -> tuple[list[_FloatArray], _FloatArray]:
    """Draw one control net and the per-direction bases of a point set.

    Args:
        rng (np.random.Generator): Source of the sample.
        degrees (tuple[int, ...]): Degree per parametric direction.
        rank (int): Number of control-point components.
        dtype (npt.DTypeLike): Storage format.
        n_pts (int): Points per direction.

    Returns:
        tuple[list[npt.NDArray[...]], npt.NDArray[...]]: The bases and the net.
    """
    dim = len(degrees)
    shape = (*(degree + 1 for degree in degrees), rank)
    ctrl = rng.standard_normal(shape).astype(dtype)
    bases = [
        _tabulate_bernstein_1d_fast(
            degrees[direction],
            rng.uniform(0.0, 1.0, n_pts).astype(dtype),
            dtype,
        )
        for direction in range(dim)
    ]
    return bases, ctrl


# ---------------------------------------------------------------------------
# The measurements
# ---------------------------------------------------------------------------
#
# Two questions, and they are separate. **Width**: does the oracle accumulate in
# the storage format or in float64? That question only exists at float32, where
# the two answers differ; at float64 the narrow and the wide model are one
# function and the pair says nothing, which the tables below state rather than
# treat as a failed discrimination. **Schedule**: does a left-to-right
# accumulation in ascending index order reproduce the oracle's summation tree?
# That question exists at both dtypes, and it is the one that decides whether a
# bitwise claim is available.


class Case(NamedTuple):
    """One drawn evaluation problem.

    Attributes:
        degrees (tuple[int, ...]): Degree per parametric direction.
        rank (int): Number of control-point components.
        bases (Sequence[npt.NDArray[np.float32 | np.float64]]): Per-direction
            Bernstein tabulations.
        ctrl (npt.NDArray[np.float32 | np.float64]): The control net.
    """

    degrees: tuple[int, ...]
    rank: int
    bases: Sequence[_FloatArray]
    ctrl: _FloatArray


def draw_cases(
    rng: np.random.Generator,
    dtype: npt.DTypeLike,
    *,
    n_pts: int,
    ranks: tuple[int, ...] = _RANKS,
) -> list[Case]:
    """Draw the full sweep of evaluation problems for one dtype.

    Args:
        rng (np.random.Generator): Source of the samples.
        dtype (npt.DTypeLike): Storage format.
        n_pts (int): Points per direction.
        ranks (tuple[int, ...]): Output ranks to sweep. Defaults to
            :data:`_RANKS`.

    Returns:
        list[Case]: One case per (degrees, rank, trial).
    """
    cases: list[Case] = []
    for degrees in _DEGREES:
        for rank in ranks:
            for _trial in range(_TRIALS):
                bases, ctrl = _draw_case(rng, degrees, rank, dtype, n_pts)
                cases.append(Case(degrees, rank, bases, ctrl))
    return cases


def measure_width(
    site: str,
    cases: list[Case],
    oracle: _Schedule,
    narrow: _Schedule,
    wide: _Schedule,
) -> list[Verdict]:
    """Measure which accumulation width reproduces the oracle.

    Args:
        site (str): Where the contraction lives.
        cases (list[Case]): The drawn problems, all at float32.
        oracle (_Schedule): The numpy expression under test.
        narrow (_Schedule): The storage-format model.
        wide (_Schedule): The float64-accumulator model, same schedule.

    Returns:
        list[Verdict]: The narrow model then the wide model.
    """
    hits = [0, 0]
    differ = 0
    for case in cases:
        reference = oracle(case.bases, case.ctrl)
        models = (narrow(case.bases, case.ctrl), wide(case.bases, case.ctrl))
        hits = [hit + same_bits(model, reference) for hit, model in zip(hits, models, strict=True)]
        differ += not same_bits(models[0], models[1])
    return [
        Verdict(site, "accumulate in the storage format", hits[0], len(cases), differ),
        Verdict(site, "accumulate in float64, round on store", hits[1], len(cases), differ),
    ]


def measure_schedule(
    site: str,
    cases: list[Case],
    oracle: _Schedule,
    narrow: _Schedule,
) -> tuple[Verdict, int | None]:
    """Measure whether ascending index order reproduces the oracle's summation tree.

    Args:
        site (str): Where the contraction lives.
        cases (list[Case]): The drawn problems.
        oracle (_Schedule): The numpy expression under test.
        narrow (_Schedule): The ascending-order model in the storage format.

    Returns:
        tuple[Verdict, int | None]: The verdict, and the shortest contraction
        length at which the model failed, or None if it never did. The
        ``rivals_differ`` field carries the number of cases that failed, since the
        rival here is the oracle itself rather than a second model.
    """
    matched = 0
    shortest_failure: int | None = None
    for case in cases:
        if same_bits(narrow(case.bases, case.ctrl), oracle(case.bases, case.ctrl)):
            matched += 1
            continue
        length = max(degree + 1 for degree in case.degrees)
        if shortest_failure is None or length < shortest_failure:
            shortest_failure = length
    total = len(cases)
    return (
        Verdict(site, "ascending index order", matched, total, total - matched),
        shortest_failure,
    )


def report_width(verdicts: list[Verdict], *, expect_a_winner: bool) -> bool:
    """Print a width table and say whether it said what it was expected to say.

    Args:
        verdicts (list[Verdict]): Rival pairs, expected winner first.
        expect_a_winner (bool): True where one of the two widths should reproduce
            every case. False for a contraction that reaches BLAS, where the
            finding is that **neither** does and a matching model would mean the
            expression had stopped being a matrix product.

    Returns:
        bool: True when each pair discriminates and the expectation held.
    """
    sound = True
    for index, verdict in enumerate(verdicts):
        share = f"{verdict.matched}/{verdict.total}"
        print(
            f"  {verdict.site:<28} {verdict.hypothesis:<38} {share:>10} {verdict.rivals_differ:>8}"
        )
        if verdict.rivals_differ == 0:
            sound = False
        won = verdict.matched == verdict.total
        if won != (expect_a_winner and index % 2 == 0):
            sound = False
    return sound


def report_schedule(verdict: Verdict, shortest_failure: int | None) -> None:
    """Print one schedule row.

    Args:
        verdict (Verdict): The measurement.
        shortest_failure (int | None): Shortest contraction length that failed.
    """
    share = f"{verdict.matched}/{verdict.total}"
    where = "never fails" if shortest_failure is None else f"fails from length {shortest_failure}"
    print(f"  {verdict.site:<28} {share:>12} reproduced   {where}")


# ---------------------------------------------------------------------------
# The spread between the two entry points
# ---------------------------------------------------------------------------


def _stack_lattice_points(bases: Sequence[_FloatArray]) -> list[_FloatArray]:
    """Expand per-direction lattice bases into per-point bases, in reshape order.

    ``np.reshape`` on a lattice result lays direction 0 out slowest, so the point
    list that matches it is the cartesian product in the same order.

    Args:
        bases (Sequence[npt.NDArray[np.float32 | np.float64]]): One basis per
            direction, of shape ``(m_d, n_d)``.

    Returns:
        list[npt.NDArray[np.float32 | np.float64]]: One basis per direction, each
        of shape ``(prod(m_d), n_d)``, holding the same values the lattice does.
    """
    counts = [basis.shape[0] for basis in bases]
    indices = np.indices(counts).reshape(len(bases), -1)
    return [basis[indices[direction]] for direction, basis in enumerate(bases)]


def measure_the_spread(cases: list[Case], dtype: npt.DTypeLike) -> None:
    """Report how far apart the two entry points are on the same mathematics.

    Evaluates one Bezier over one point set twice: once through the lattice
    schedule, and once through the pts-array schedule with the lattice written out
    as an explicit list of points. The two compute the same real number and
    disagree in floating point, which is why they cannot share a parity claim.

    The spread is reported as a fraction of the **one-sided** ``gamma_n``
    budget -- ``stages * u`` times the absolute-value companion -- because that is
    the quantity a bound is written in. A bare relative figure would say nothing
    about whether a bound covers it, and a bound is what this measurement exists
    to size.

    Args:
        cases (list[Case]): The drawn problems.
        dtype (npt.DTypeLike): Storage format, for the unit of roundoff.
    """
    unit = float(np.finfo(dtype).eps) / 2.0
    worst_ratio = 0.0
    worst_case = ""
    differing = total = 0
    for case in cases:
        lattice = np.reshape(oracle_lattice(case.bases, case.ctrl), (-1, case.rank))
        stacked = _stack_lattice_points(case.bases)
        points = oracle_pts_array(stacked, case.ctrl)
        companion = oracle_pts_array(stacked, np.abs(case.ctrl))
        stages = sum(basis.shape[1] for basis in case.bases)
        budget = stages * unit * companion.astype(np.float64)
        difference = np.abs(points.astype(np.float64) - lattice.astype(np.float64))
        total += difference.size
        differing += int(np.count_nonzero(difference))
        ratio = np.where(budget > 0.0, difference / np.where(budget > 0.0, budget, 1.0), 0.0)
        if float(np.max(ratio)) > worst_ratio:
            worst_ratio = float(np.max(ratio))
            worst_case = f"degrees {case.degrees} rank {case.rank}"

    name = np.dtype(dtype).name
    print(
        f"  {name}: {differing}/{total} values differ between the entry points; worst is "
        f"{worst_ratio:.3g} of the one-sided budget, at {worst_case}"
    )


def measure_the_margin(cases: list[Case], dtype: npt.DTypeLike) -> float:
    """Report how far the transliteration sits from each oracle, against its bound.

    This is the figure the parity claim is sized by. The C++ port computes the
    ascending-order narrow contraction, so what a bound has to cover is the gap
    between *that* and each oracle -- not the gap between the two oracles, which
    is reported separately because the ticket asks for it but which no single
    claim compares.

    The bound is the two-sided one the harness builds: ``2 * gamma_N * A`` with
    ``N`` the summed contraction lengths and ``A`` the absolute-value companion.
    A ratio at or above 1 means the bound is violated and the derivation is
    wrong; a ratio far below it means the bound says less than it could.

    Args:
        cases (list[Case]): The drawn problems.
        dtype (npt.DTypeLike): Storage format, for the unit of roundoff.

    Returns:
        float: The largest ratio seen, over both entry points.
    """
    unit = float(np.finfo(dtype).eps) / 2.0
    worst = 0.0
    for site, oracle, narrow, expand in (
        (_PTS_ARRAY_SITE, oracle_pts_array, narrow_pts_array, _stack_lattice_points),
        (_LATTICE_SITE, oracle_lattice, narrow_lattice, None),
    ):
        ratio_max = 0.0
        where = ""
        for case in cases:
            bases = case.bases if expand is None else expand(case.bases)
            reference = oracle(bases, case.ctrl).astype(np.float64)
            actual = narrow(bases, case.ctrl).astype(np.float64)
            stages = sum(basis.shape[1] for basis in bases)
            growth = stages * unit / (1.0 - stages * unit)
            companion = oracle(bases, np.abs(case.ctrl)).astype(np.float64)
            bound = ONE_SIDED_TO_TWO_SIDED * growth * np.abs(companion)
            ratio = np.abs(actual - reference) / np.where(bound > 0.0, bound, 1.0)
            if float(np.max(ratio)) > ratio_max:
                ratio_max = float(np.max(ratio))
                where = f"degrees {case.degrees} rank {case.rank}"
        print(f"  {np.dtype(dtype).name} {site:<28} worst {ratio_max:.3g} of the bound, at {where}")
        worst = max(worst, ratio_max)
    return worst


def report_provenance() -> None:
    """Print what the figures above are pinned to, so a quotation carries it.

    A measurement without the commit and the library version it was taken against
    is not reproducible, and a number nobody can re-derive is a number nobody can
    refute.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - not a git tree
        commit = "unknown"
    print(f"\ncommit {commit}, numpy {np.__version__}, python {sys.version.split()[0]}")


def main() -> int:
    """Run every measurement and report.

    Returns:
        int: 0 when every table said what it was expected to say, 1 otherwise.
    """
    seeds = np.random.SeedSequence(_SEED).spawn(6)
    sound = True

    print("Width, at float32 only -- at float64 the two models are one function.")
    print(f"  {'site':<28} {'hypothesis':<38} {'matched':>10} {'discr.':>8}")
    narrow_tail = draw_cases(np.random.default_rng(seeds[0]), np.float32, n_pts=7, ranks=(2, 3))
    sound &= report_width(
        measure_width(
            _PTS_ARRAY_SITE, narrow_tail, oracle_pts_array, narrow_pts_array, wide_pts_array
        ),
        expect_a_winner=True,
    )
    lattice_cases32 = draw_cases(np.random.default_rng(seeds[1]), np.float32, n_pts=5)
    # BLAS: the finding is that neither width reproduces it, so a winner here would
    # mean tensordot had stopped reaching a matrix product and the claim needs
    # rederiving rather than that the model was right.
    sound &= report_width(
        measure_width(_LATTICE_SITE, lattice_cases32, oracle_lattice, narrow_lattice, wide_lattice),
        expect_a_winner=False,
    )

    print("\nSchedule: does ascending index order reproduce the oracle's summation tree?")
    for index, dtype in enumerate(_DTYPES):
        name = np.dtype(dtype).name
        print(f"  -- {name} --")
        wide_tail = draw_cases(
            np.random.default_rng(seeds[2 + index]), dtype, n_pts=7, ranks=(2, 3)
        )
        report_schedule(
            *measure_schedule(_PTS_ARRAY_SITE, wide_tail, oracle_pts_array, narrow_pts_array)
        )
        scalar_tail = draw_cases(
            np.random.default_rng(seeds[2 + index]), dtype, n_pts=7, ranks=(1,)
        )
        report_schedule(
            *measure_schedule(_SCALAR_TAIL_SITE, scalar_tail, oracle_pts_array, narrow_pts_array)
        )
        lattice_cases = draw_cases(np.random.default_rng(seeds[2 + index]), dtype, n_pts=5)
        report_schedule(
            *measure_schedule(_LATTICE_SITE, lattice_cases, oracle_lattice, narrow_lattice)
        )

    print("\nSpread between the two entry points on the same mathematics:")
    for index, dtype in enumerate(_DTYPES):
        measure_the_spread(
            draw_cases(np.random.default_rng(seeds[4 + index]), dtype, n_pts=5), dtype
        )

    print("\nWhere the ascending-order transliteration sits inside its own bound:")
    for index, dtype in enumerate(_DTYPES):
        margin = measure_the_margin(
            draw_cases(np.random.default_rng(seeds[4 + index]), dtype, n_pts=5), dtype
        )
        if margin >= 1.0:
            print(f"  {np.dtype(dtype).name}: the bound is VIOLATED; rederive it.")
            sound = False

    report_provenance()

    if not sound:
        print("\nA width was not confirmed, or its check could not have failed.")
        return 1
    print(
        "\nThe oracle accumulates in the storage format, and ascending index order "
        "reproduces it\nonly where the contraction's trailing block holds two or more "
        "elements. The scalar\ntail and the lattice do not admit a bitwise model, so "
        "both entry points carry a\nbounded claim; see tests/parity/test_bezier_evaluate.py "
        "for the two derivations."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
