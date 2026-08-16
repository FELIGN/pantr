"""Known-bug regressions from the adversarial parameter sweep (August 2026).

Each test here reproduces a bug the sweep in ``tools/adversarial_sweep/`` found on inputs
the rest of the suite does not contain. A bug that is still open carries
``xfail(strict=True)``; when it is fixed the test starts passing, pytest reports a strict
XPASS failure, and the marker comes off, promoting the test to a permanent guard on the
same data. That is the convention ``tests/test_review_regressions.py`` follows for the
June 2026 review, whose markers have all since been removed.

Eleven markers have already come off here: the domain-membership test, closed by the
``np.isclose`` tolerance-leak fix in #289; the tanh-sinh endpoint test, closed by
truncating the rule where the endpoint gap stops being resolvable; the Lagrange
reproducibility test, closed by seeding the barycentric node permutation; the float32
degree-elevation test, closed by allocating the kernels' knot output in the input's dtype;
the periodic degree-reduction hang, closed by enforcing the periodic conversion's own
boundary-multiplicity precondition; the degree-elevation counter mismatch, closed by
emitting the unshared Bézier coefficient at a C^-1 knot and by letting the segment sweep
reach the final span of a degree-0 knot vector; and the root finder that certified a value
that is not a root, closed by evaluating the residual on every exit of the tracking and by
holding the repeated-iterate stop to Corollary 14's actual hypothesis; and four closed
together by the tolerance-semantics pass -- the two knot-snapping tests and the unique-knot
accessor directly, and the restriction that returned a shorter domain than asked for as a
consequence, since its ``b_new + tol`` step ceased to be a no-op once ``tol`` carried the
knot vector's magnitude.

Three remain open, all from the August 2026 triage of the full profile: knot-vector
factories that disagree with their own documentation at zero intervals, a Lagrange
extraction that cannot be built on a degree-0 space its two sibling extractions handle, and
a change of basis that reports numpy's `LinAlgError` for a legal degree.

One test per **root cause**, not per symptom: several of these root causes have many
triggering combinations, and each test names them in a comment rather than repeating
itself. Regenerate the findings with::

    conda run -n pantr python tools/sweep.py --profile full

The triggering data is hardcoded rather than swept, so a test failure here is a statement
about this exact input and nothing else.
"""

from __future__ import annotations

import math
import signal
from types import FrameType

import numpy as np
import pytest

from pantr.basis import LagrangeVariant, tabulate_lagrange_1d
from pantr.bspline import (
    Bspline,
    BsplineSpace,
    BsplineSpace1D,
    create_uniform_open_knots,
    create_uniform_periodic_knots,
    find_roots,
)
from pantr.bspline._bspline_degree_core import _degree_elevate_1d_core
from pantr.change_basis import compute_cardinal_to_bernstein_1d
from pantr.quad import get_tanh_sinh_1d
from pantr.tolerance import get_machine_epsilon


def _scalar_spline(knots: np.ndarray, degree: int, values: list[float]) -> Bspline:
    """Build a scalar 1D B-spline on the given knot vector."""
    space = BsplineSpace([BsplineSpace1D(knots, degree)])
    return Bspline(space, np.asarray(values, dtype=knots.dtype).reshape(-1, 1))


class _CallTimeout(RuntimeError):
    """Raised by :func:`_deadline` when the guarded call does not return in time."""


_TANH_SINH_TAIL: float = 8.0 * math.sqrt(get_machine_epsilon(np.float64))
"""Accuracy floor of a tanh-sinh rule on ``x**-0.5``, from its truncation gap.

The rule is truncated where the distance from a node to the endpoint stops being
representable, so it covers ``[delta, 1 - delta]`` and misses the tail
``int_0^delta x**-0.5 dx = 2 * sqrt(delta)``. The truncation guarantees only
``delta >= eps / 2``; the last node kept sits wherever the step ``h`` puts it, so
``delta`` is a few times that and the floor is a few times ``sqrt(2 * eps) = 2.1e-8``.
Measured over ``n_pts`` from 45 to 400, the error runs from 4.1e-9 to 4.8e-8, the
largest corresponding to ``delta = 5.3 * eps / 2``.

``8 * sqrt(eps) = 1.2e-7`` therefore leaves a factor 2.5 over the worst case measured,
while still failing by six orders of magnitude on the bug this pins, which returned
``inf``.
"""

_HANG_BUDGET_SECONDS = 2.0
"""Budget for a call that must terminate promptly.

Derived from the working case rather than picked: the same reduction on the *clamped*
equivalent of the knot vector below returns in 0.003 s, so this is 600 times the observed
cost and cannot fire on a slow machine. It is the price the suite pays while the bug is
open; once fixed the call returns in milliseconds and the budget is never reached.
"""


# ---------------------------------------------------------------------------
# Degree elevation: the kernel's two returns disagree
# ---------------------------------------------------------------------------


def test_degree_elevation_outputs_are_mutually_consistent() -> None:
    # FIXED, and kept as a regression guard with its original triggering data per this
    # repository's convention that the fix PR un-xfails the tests it closes. It took two
    # changes, not one: what looked like a single defect with two faces is two defects
    # that happened to share a symptom.
    #
    # What it was: `_degree_elevate_1d_core` returns (control_points, knots), usable
    # together only when `control_points.shape[0] == knots.size - new_degree - 1`. Those
    # come from counters `cind` and `kind` maintained independently through the
    # Piegl-Tiller A5.9 walk, and they diverged by the number of interior knots at
    # multiplicity degree + 1 (1, 2 and 3 such knots gave deficit 1, 2, 3, and raising the
    # increment did not change it).
    #
    # The mechanism the original note recorded as *inferred* is now traced and confirmed,
    # for degree >= 1: `lbz` is the index of the first elevated Bezier coefficient a
    # segment contributes, and it started at 1 because A5.9 assumes the previous segment
    # already wrote the shared junction coefficient. At multiplicity degree + 1 the two
    # segments share nothing, so coefficient 0 was dropped -- one control point per jump,
    # while the knot writer emitted the correct `mul + t` knots.
    #
    # Degree 0 had a *second*, independent cause that the deficit arithmetic hid: the
    # sweep ran `while b < m`, and for degree >= 1 the closing block of `degree + 1` equal
    # knots lets the inner run scan reach `m` by itself. At degree 0 that block is a
    # single knot, the scan cannot advance, and the last segment was never processed nor
    # the closing knots written -- so the knot vector kept its trailing zeros and the
    # domain collapsed to a point. Both halves below pin both fixes.
    for degree, mult in ((1, 2), (2, 3), (3, 4), (0, 1)):
        knots = np.concatenate(
            [np.full(degree + 1, 0.0), np.full(mult, 0.5), [0.75], np.full(degree + 1, 1.0)]
        )
        n_basis = knots.size - degree - 1
        control = np.arange(2 * n_basis, dtype=np.float64).reshape(n_basis, 2)
        new_control, new_knots = _degree_elevate_1d_core(degree, control, knots, 1)
        implied = new_knots.size - (degree + 1) - 1
        assert new_control.shape[0] == implied, (
            f"degree {degree}, interior multiplicity {mult}: kernel wrote "
            f"{new_control.shape[0]} control points but its own knot vector "
            f"({new_knots.size} knots at degree {degree + 1}) implies {implied}"
        )
        assert np.all(np.diff(new_knots) >= 0.0), (
            f"degree {degree}, interior multiplicity {mult}: returned knot vector is not "
            f"non-decreasing: {new_knots.tolist()}"
        )

    # The silent face, entirely through public API: a degree-0 spline on [0, 1] with two
    # control points came back with a collapsed domain and a duplicated control point.
    spline = _scalar_spline(np.array([0.0, 0.5, 1.0]), 0, [1.0, 2.0])
    elevated = spline.elevate_degree(1)
    elevated_knots = np.asarray(elevated.space.spaces[0].knots)
    assert elevated_knots[-1] > elevated_knots[0], (
        f"degree elevation collapsed the domain to a point: {elevated_knots.tolist()}"
    )
    # ... and it is still the same step function, off the jump.
    pts = np.array([0.1, 0.4999, 0.5001, 0.9])
    np.testing.assert_array_equal(
        np.asarray(elevated.evaluate(pts)).ravel(), np.array([1.0, 1.0, 2.0, 2.0])
    )


def test_degree_elevation_preserves_float32() -> None:
    # FIXED by allocating the kernel's knot output in the knot vector's own dtype, in both
    # `_degree_elevate_1d_core` and `_degree_reduce_1d_core`. Kept as a regression guard
    # with its original triggering data, per this repository's convention that the fix PR
    # un-xfails the tests it closes.
    #
    # What it was: the elevation kernel allocated its two outputs with different dtypes,
    # `ik = np.zeros(max_new_knots, dtype=np.float64)` hardcoded while
    # `ic = np.zeros(..., dtype=ctrl.dtype)` followed the input. The knot vector therefore
    # came back float64 while the control points stayed float32, and the two faces of that
    # were:
    #   * clamped -- `Bspline.__init__` rejected the pair, so `elevate_degree` raised for
    #     **every** float32 spline, at every degree and every knot count. Measured on
    #     degrees 1-3 with 3, 4 and 6 breakpoints: nine for nine. The message blamed the
    #     caller's control points for the kernel's own hardcoded dtype.
    #   * periodic -- the round trip through open form converted the control points too, so
    #     both came back float64 and the call *succeeded*, silently discarding the
    #     caller's choice of precision.
    #
    # The second face is what made this worth a test rather than a bug report: a silent
    # dtype promotion is invisible until something downstream compares dtypes, and it
    # doubles memory and halves throughput without a word.
    #
    # `_degree_reduce_1d_core` carried the identical hardcoded allocation and raised the
    # identical error, so the third assertion below covers the round trip.
    knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float32)
    clamped = _scalar_spline(knots, 2, [1.0, 2.0, 3.0, 4.0])
    assert np.dtype(clamped.dtype) == np.float32, "precondition: the spline is float32"

    # Face one: it does not work at all.
    try:
        elevated = clamped.elevate_degree(1)
    except ValueError as exc:
        pytest.fail(f"degree elevation of a clamped float32 spline raised: {exc}")
    assert np.dtype(elevated.dtype) == np.float32, (
        f"degree elevation changed the dtype from float32 to {np.dtype(elevated.dtype).name}"
    )

    # Face two: on a periodic space it succeeds and silently promotes. Kept in the same
    # test because it is the same allocation, and separating them would imply two fixes.
    #
    # The control-point count is corrected here: as written when the marker went on, this
    # half built five control points on a space that has three basis functions, so it
    # raised from the `Bspline` constructor before reaching any dtype at all. A strict
    # xfail is satisfied by *any* failure, so the marker hid the fact that this half was
    # never exercising the bug. The knot vector is the original one; only the count
    # changes, and the precondition below now pins it.
    periodic_degree = 2
    periodic_knots = (np.arange(-2, 6, dtype=np.float64) / 3.0).astype(np.float32)
    periodic_space = BsplineSpace1D(periodic_knots, periodic_degree, periodic=True)
    assert periodic_space.num_basis == periodic_knots.size - 2 * periodic_degree - 1, (
        f"precondition: {periodic_knots.size} knots at degree {periodic_degree} give "
        f"{periodic_knots.size - 2 * periodic_degree - 1} periodic basis functions, got "
        f"{periodic_space.num_basis}"
    )
    periodic = Bspline(
        BsplineSpace([periodic_space]),
        np.arange(periodic_space.num_basis, dtype=np.float32).reshape(-1, 1),
    )
    promoted = periodic.elevate_degree(1)
    assert np.dtype(promoted.dtype) == np.float32, (
        f"degree elevation silently promoted a periodic float32 spline to "
        f"{np.dtype(promoted.dtype).name}"
    )

    # The sibling kernel had the same hardcoded allocation, so the round trip back down
    # is what shows both are fixed. Elevation followed by reduction returns the original
    # spline, so the values are asserted too and not only the dtype.
    reduced = elevated.reduce_degree(1)
    assert np.dtype(reduced.dtype) == np.float32, (
        f"degree reduction changed the dtype from float32 to {np.dtype(reduced.dtype).name}"
    )
    sample = np.linspace(0.0, 1.0, 17, dtype=np.float32).reshape(-1, 1)
    before = np.asarray(clamped.evaluate(sample), dtype=np.float64)
    after = np.asarray(reduced.evaluate(sample), dtype=np.float64)
    # Elevating then reducing is exact in exact arithmetic -- the reduction operator
    # interpolates the endpoints and is an exact left-inverse of elevation on the
    # elevated subspace -- so the only error is float32 rounding through the two kernels:
    # `degree + 1` convex combinations each way, on values up to 4. Checked rather than
    # assumed: the same round trip in float64 leaves a residual of 0 to 3.6 eps relative
    # over degrees 1 to 4 and 3, 4 and 6 breakpoints, which is rounding and not method
    # error.
    degree = int(clamped.degree[0])
    tolerance = 2.0 * (degree + 1) * get_machine_epsilon(np.float32) * float(np.abs(before).max())
    assert float(np.max(np.abs(before - after))) <= tolerance


# ---------------------------------------------------------------------------
# Tolerance policy: np.isclose keeps its default rtol
# ---------------------------------------------------------------------------


def test_out_of_domain_points_are_rejected_at_large_knot_magnitude() -> None:
    # FIXED in `fix(bspline,grid): close eight Layer-2 validation gaps and the np.isclose
    # tolerance leak (#289)`, which converted all 26 sites to absolute comparisons. Kept as
    # a regression guard with its original triggering data, per this repository's
    # convention that the fix PR un-xfails the tests it closes.
    #
    # What it was: `_is_in_domain` (`_bspline_knots.py:172-173`) accepted a point when
    # `np.isclose(knot_end, pt, atol=tol)`. Setting `atol` does not clear `rtol`, which
    # stayed at numpy's default 1e-5, so the test was really
    # `|pt - knot_end| <= tol + 1e-5 * |knot_end|`. On a domain of length 1 placed at 1e6
    # that admitted points up to 10.00001 outside -- ten domain lengths -- while the
    # space's stated tolerance is 1e-15.
    #
    # It was one root cause with 26 sites: every `np.isclose(..., atol=tol)` in
    # `pantr.bspline` leaves `rtol` at its default, across `_bspline_restrict.py` (10),
    # `_bspline_knots.py` (5), `_bspline_knot_insertion.py` (4),
    # `_bspline_knot_removal.py` (2), `_bspline_product.py` (2),
    # `_bspline_space_1d.py` (2) and `_bspline_split.py` (1) -- not one of them passed
    # `rtol`. (A 27th call, in `_bspline_quasi_interpolation.py`, is inside a doctest and
    # so not a library site.) Consequences measured at the time, all at |knot| ~ 1e6:
    # `tabulate_basis` accepted a point 10 units outside a unit-length domain and returned
    # a polynomial extrapolation (max|B| = 640 at degree 2, 4.3e43 at degree 62) instead of
    # raising; `remove_knots` refused to remove the interior knot 1000000.3125, calling it
    # the domain start; `insert_knots` reported a false multiplicity clash between knots
    # 0.0625 apart.
    #
    # It was also **memory-unsafe**, which is what lifted it above a wrong-answer bug. On a
    # *periodic* space over the same translated domain, `elevate_degree` and
    # `reduce_degree` made a genuine out-of-bounds access -- `IndexError: index is out of
    # bounds` under NUMBA_BOUNDSCHECK=1, at degrees 1, 2 and 3 in both dtypes. The
    # boundary multiplicity was counted with the same leaky comparison
    # (`_bspline_knot_insertion.py:242-243`), the false count was then used as an index,
    # and with the bounds check off that read was silent. Varying only the domain isolated
    # it:
    #
    #   [0, 1]  [0, 5]  [0, 100]  [0, 1e6]  [1e3, 1e3+1]  [1e4, 1e4+1]  ->  fine
    #   [1e6, 1e6+1]                                                    ->  IndexError
    #
    # So it was not magnitude but *translation*: the offset set the effective tolerance
    # (1e-5 * 1e6 = 10) while the span it had to resolve stayed 1.
    #
    # The author knew the trap -- `_snap_knots` (`_bspline_space_1d.py:211-215`) carries a
    # comment warning about exactly it -- and had avoided it in that one place.
    lo, hi = 1e6, 1e6 + 1.0
    knots = np.concatenate([np.full(3, lo), [lo + 0.5], np.full(3, hi)])
    space = BsplineSpace1D(knots, 2)
    # Two-sided, and stating it that way is what makes it survive a change in how the
    # tolerance is derived: it must be at least one ulp of the endpoint (below that the
    # endpoint is not resolvable and legitimate in-domain points get rejected) and far
    # below the domain length (above that the rejection this test asks for stops
    # happening). It used to read `< 1e-9`, which was the old per-dtype constant seen
    # through this fixture rather than a property of it.
    endpoint_ulp = float(np.spacing(hi))
    assert endpoint_ulp <= float(space.tolerance) < 1e-3 * (hi - lo), (
        f"precondition: the tolerance ({float(space.tolerance):.3e}) must resolve the "
        f"endpoint (ulp {endpoint_ulp:.3e}) without approaching the domain length "
        f"({hi - lo:g})"
    )

    with pytest.raises(ValueError, match="outside the knot vector domain"):
        # 10 domain lengths past the right end.
        space.tabulate_basis(np.array([hi + 10.0]))

    # Guard the other side too, which the original xfail did not: the fix must not
    # over-correct into rejecting points that are legitimately inside. A tolerance that
    # went fully absolute at 1e-15 would fail here, because the endpoint itself is only
    # representable to about 1.2e-10 at this magnitude (one ulp of 1e6).
    inside = np.array([lo, lo + 0.5, hi], dtype=np.float64)
    basis, _ = space.tabulate_basis(inside)
    assert np.all(np.isfinite(basis)), "in-domain points must still be accepted"


# ---------------------------------------------------------------------------
# Knot snapping: two independent defects in one method
# ---------------------------------------------------------------------------


def test_snapping_preserves_a_run_of_identical_knots() -> None:
    # FIXED by electing each group's *first* knot as its representative instead of the
    # group's mean (`BsplineSpace1D._snap_knots`). Kept as a regression guard with its
    # original triggering data, per this repository's convention that the fix PR un-xfails
    # the tests it closes. Choosing rather than averaging is also what makes snapping
    # idempotent, which is the property the exact `denom == 0.0` Cox-de Boor guard wants.
    #
    # What it was: `_snap_knots` replaced every group of knots that round to the same grid
    # point by `np.mean(group, dtype=self.dtype)`. For a clamped end the group is
    # `degree + 1` copies of one value, so the mean must be that value exactly.
    # It was not: the summation loses the low bits once `(degree + 1) * |knot|` passes the
    # format's exact-integer range (2^24 for float32, 2^53 for float64), and the knot
    # moved by 0.125. The consequence was that the space's *reported domain* differed from
    # the one the caller asked for, silently.
    #
    # Which run lengths survive is **not** a clean rule, and it is worth not pretending
    # otherwise: it falls out of NumPy's pairwise-summation blocking interacting with the
    # value's own bit pattern. Measured for float64 at 1e15 + 1, runs of 11, 13, 14, 15,
    # 18-23 and 26-31 copies are already inexact while 12, 16, 17, 24, 25 and 32 are
    # exact. So this bites from **degree 10** in float64 at that magnitude, and the two
    # degrees asserted below are simply the ones the sweep happened to visit -- do not
    # read them as a threshold.
    #
    # One thing moved when the fix landed, and only one: the domain's *lower* end. The
    # original fixture ran from `base` to `base + 1`, which the merge tolerance now
    # (correctly) swallows whole -- at float32 a span of 1 sitting at 1e6 is 16 ulp wide,
    # so every knot in it is within `8 * eps * 1e6 = 0.95` of every other and the whole
    # vector is one knot. That is a different phenomenon and it is asserted separately in
    # `test_a_domain_below_its_own_resolution_collapses`. The run values that trigger
    # *this* bug -- 1000001.0 and 1000000000000001.0, whose means over 63 copies are
    # 1000000.875 and 1000000000000000.9 -- are unchanged; the lower end simply moves to
    # zero so the span is resolvable and the knots stay distinct.
    cases: tuple[tuple[type[np.floating], float], ...] = ((np.float32, 1e6), (np.float64, 1e15))
    for dtype, base in cases:
        degree = 62
        hi = np.asarray(base + 1.0, dtype=dtype)
        mid = np.asarray(0.5 * float(hi), dtype=dtype)
        raw = np.concatenate(
            [
                np.full(degree + 1, 0.0, dtype=dtype),
                [mid],
                np.full(degree + 1, hi, dtype=dtype),
            ]
        )
        run = np.full(degree + 1, hi, dtype=dtype)
        assert float(np.mean(run, dtype=dtype)) != float(hi), (
            f"{np.dtype(dtype).name}: precondition -- averaging the run must still be "
            f"inexact, or this fixture no longer reaches the bug"
        )
        stored = np.asarray(BsplineSpace1D(raw, degree).knots)
        assert stored[-1] == hi, (
            f"{np.dtype(dtype).name}: snapping moved a run of {degree + 1} identical "
            f"knots from {float(hi)!r} to {float(stored[-1])!r}"
        )
        assert stored[degree + 1] == mid, (
            f"{np.dtype(dtype).name}: the interior knot must stay where it was, at "
            f"{float(mid)!r}, not {float(stored[degree + 1])!r}"
        )


def test_a_domain_below_its_own_resolution_is_refused() -> None:
    """A knot vector whose spacing is inside its own coordinate noise is rejected.

    Not a defect but the contract, and the one thing the merge rule takes away. The
    tolerance is ``8 * eps * max(span, |knots[0]|, |knots[-1]|)``, so a mesh of ``n``
    intervals over a span ``s`` at offset ``x`` survives only while
    ``s / n > 8 * eps * max(s, |x|)``; once ``|x|`` dominates that fails at
    ``|x| / s = 1 / (8 * eps * n)``, about ``5.2e5 / n`` in float32.

    ``[1e6, 1e6 + 1]`` in float32 is past it: the ulp there is 0.0625, so the whole
    window holds 16 representable coordinates and an interior knot computed by any route
    is uncertain by ``eps * 1e6 = 0.06``, six percent of the span. No threshold keeps
    such a mesh *and* merges two routes to one knot, so the space cannot be built, and
    saying so beats collapsing silently -- which is what an earlier version of this
    change did, and what turned 525 sweep cases into opaque failures deep in the
    kernels. The same vector in float64 keeps every knot, which is the point: what
    decides is the format's resolution at that magnitude, not the magnitude.
    """
    degree = 2
    lo, hi = 1e6, 1e6 + 1.0
    raw = [lo] * (degree + 1) + [lo + 0.5] + [hi] * (degree + 1)

    with pytest.raises(ValueError, match="collapsed every knot") as excinfo:
        BsplineSpace1D(np.asarray(raw, dtype=np.float32), degree)
    message = str(excinfo.value)
    # The message has to let the reader act without opening our source.
    assert "float32" in message, message
    assert "0.5 apart" in message, message
    assert "Use float64" in message, message

    resolved = BsplineSpace1D(np.asarray(raw, dtype=np.float64), degree)
    assert np.array_equal(np.asarray(resolved.knots), np.asarray(raw, dtype=np.float64)), (
        "the same vector in float64 resolves easily and must be left alone"
    )
    assert resolved.num_intervals == 2


def test_an_already_degenerate_knot_vector_is_still_accepted() -> None:
    """The refusal fires only when *snapping* destroyed the mesh, not when it arrived flat.

    A caller who passes a knot vector that is already a single repeated value asked for
    exactly that and gets it, at any magnitude and in either dtype. Distinguishing the
    two is what keeps the new rejection from being a general ban on degenerate spaces:
    the case worth refusing is the one the caller could not see coming.
    """
    for dtype in (np.float32, np.float64):
        for value in (0.0, 1.0, 1e6, -3.5):
            space = BsplineSpace1D(np.full(8, value, dtype=dtype), 3)
            assert space.num_intervals == 0
            assert np.all(np.asarray(space.knots) == dtype(value))

    # And `snap_knots=False` bypasses merging and the check together, as documented.
    lo, hi = 1e6, 1e6 + 1.0
    raw = np.asarray([lo, lo, lo, lo + 0.5, hi, hi, hi], dtype=np.float32)
    kept = BsplineSpace1D(raw, 2, snap_knots=False)
    assert np.array_equal(np.asarray(kept.knots), raw)


def test_snapping_keeps_knots_the_format_can_resolve() -> None:
    # FIXED by deriving the snapping tolerance from the knot vector's own magnitude
    # instead of from the dtype alone (`_bspline_knots._knot_tolerance`), and by grouping
    # knots by relative gap rather than by rounding them onto a grid. Kept as a regression
    # guard with its original triggering data, per this repository's convention that the
    # fix PR un-xfails the tests it closes.
    #
    # What it was: `_snap_knots` rounded onto a grid of width `tolerance`, which was
    # absolute (`get_strict(float32) = 1e-7`, `get_strict(float64) = 1e-15`) while the
    # quantity it grades -- a gap between knots -- carries the domain's scale. On
    # `[0, 1e-6]` with 20 equal intervals the spacing is 5e-8, below the float32 grid, so
    # half the knots were merged and the space reported 10 intervals instead of 20. The
    # float32 ulp at 1e-6 is 6e-14, so the format resolves those knots with six orders to
    # spare: nothing about the input is unrepresentable.
    #
    # It was the small-domain face of one defect whose large-domain face was that from
    # |knot| ~ 5 upward the same grid merged nothing at all, not even a 1-ulp discrepancy.
    # The tolerance is now `8 * eps * max(span, |knots[0]|, |knots[-1]|)`, which here is
    # 7.6e-12 against a spacing of 5e-8, so the intervals survive with four orders to
    # spare and would survive equally at any other domain scale.
    #
    # Further consequences that were measured at degree 62 on `[0, 1e-6]` in float32,
    # where a periodic knot vector of 63 intervals collapsed to 10: zero-length spans made
    # `tabulate_Bezier_extraction_operators` raise a bare `ZeroDivisionError`, and the
    # basis summed to 0 instead of 1 at the right endpoint.
    n_intervals = 20
    hi = 1e-6
    breaks = np.linspace(0.0, hi, n_intervals + 1, dtype=np.float32)
    raw = np.concatenate([np.array([0.0], dtype=np.float32), breaks, np.array([hi], np.float32)])
    space = BsplineSpace1D(raw, 1)
    assert space.num_intervals == n_intervals, (
        f"asked for {n_intervals} intervals of width {hi / n_intervals:.2e} "
        f"(float32 ulp there is {float(np.spacing(np.float32(hi))):.2e}), got "
        f"{space.num_intervals}"
    )


# ---------------------------------------------------------------------------
# Lagrange tabulation is not reproducible
# ---------------------------------------------------------------------------


def test_lagrange_tabulation_is_reproducible() -> None:
    # FIXED by passing a fixed seed to the scipy interpolator
    # (`_basis_lagrange._BARYCENTRIC_SEED`). Kept as a regression guard with its original
    # triggering data, per this repository's convention that the fix PR un-xfails the
    # tests it closes.
    #
    # What it was: `_tabulate_lagrange_basis_1D_core` constructed
    # `BarycentricInterpolator(nodes_sorted, y_sorted)` with no `rng`. scipy then drew
    # from the unseeded global `numpy.random` state and applied `rng.permutation(n)` to
    # the nodes before computing the barycentric weights; its own documentation says
    # "Specify `rng` for repeatable interpolation".
    #
    # So the same call returned different values in different processes. This was not
    # floating-point nondeterminism with a bound one could derive -- it was an unseeded
    # RNG, and the spread grew with degree: about 1 ulp at degree 3-5, 4.18 absolute on a
    # value scale of 3.75e7 at degree 62 (relative 1.1e-7), and `inf` versus 1e16 across
    # separate processes at degree 62 evaluated outside [0, 1]. Everything downstream
    # inherited it: `tabulate_lagrange`, `compute_lagrange_to_bernstein_1d`,
    # `tabulate_Lagrange_extraction_operators`, and `SpanwiseElementExtraction` with the
    # Lagrange target.
    #
    # Reseeding the global state is what made the failure deterministic here; in
    # production the seeds differed because nobody set one. The assertions are kept as
    # they were, including the derived bound at degree 62, but the result is now bitwise
    # identical and the bound has ceased to be the binding constraint.
    pts = np.array([0.1, 0.5, 0.9])

    # The legacy global state is the point: it is what scipy draws from.
    np.random.seed(0)  # noqa: NPY002 -- scipy reads the legacy global state, not a Generator
    first = np.asarray(tabulate_lagrange_1d(5, LagrangeVariant.EQUISPACES, pts), dtype=np.float64)
    np.random.seed(12345)  # noqa: NPY002 -- see above
    second = np.asarray(tabulate_lagrange_1d(5, LagrangeVariant.EQUISPACES, pts), dtype=np.float64)
    assert np.array_equal(first, second), (
        f"degree 5 differs between calls by up to {np.max(np.abs(first - second)):.3e}"
    )

    np.random.seed(0)  # noqa: NPY002 -- see above
    high_first = np.asarray(
        tabulate_lagrange_1d(62, LagrangeVariant.EQUISPACES, pts), dtype=np.float64
    )
    np.random.seed(12345)  # noqa: NPY002 -- see above
    high_second = np.asarray(
        tabulate_lagrange_1d(62, LagrangeVariant.EQUISPACES, pts), dtype=np.float64
    )
    # Even granting rounding, two evaluations of one basis at one point may differ by no
    # more than the barycentric formula's own error: degree + 1 terms, each carrying at
    # most eps relative, on values of this magnitude.
    scale = float(np.max(np.abs(high_first)))
    bound = 63.0 * get_machine_epsilon(np.float64) * scale
    spread = float(np.max(np.abs(high_first - high_second)))
    assert spread <= bound, (
        f"degree 62 differs between calls by {spread:.3e}, bound {bound:.3e} "
        f"(value scale {scale:.3e})"
    )


# ---------------------------------------------------------------------------
# tanh-sinh places nodes on the endpoints it exists to avoid
# ---------------------------------------------------------------------------


def test_tanh_sinh_nodes_are_interior_and_distinct() -> None:
    # FIXED by truncating the rule where the endpoint gap stops being resolvable, instead
    # of snapping the node onto the boundary and keeping it. Kept as a regression guard
    # with its original triggering data, per this repository's convention that the fix PR
    # un-xfails the tests it closes.
    #
    # What it was: a double-exponential rule exists to avoid evaluating at the endpoints,
    # and `get_tanh_sinh_1d`'s own docstring advertises it as "well suited for integrands
    # with endpoint singularities". Nodes that underflowed to the boundary were snapped
    # there, which the docstring recorded only as making the returned count smaller -- but
    # the snapped node was *kept*, with a nonzero weight (6.5e-17 at n_pts = 110), and from
    # n_pts = 53 a second node snapped onto the same boundary and was returned twice.
    #
    # Consequence: for f(x) = 1/sqrt(x), the advertised use case, the rule returned `inf`
    # rather than 2.0 at every n_pts >= 45. Weights still summed to 1 throughout, so the
    # module's own doctest could not see it. Raising the point count turned a correct
    # answer into `inf`: n_pts = 33 gave 2.0, n_pts = 49 and 129 gave `inf`.
    for n_pts in (45, 53, 110, 129):
        nodes, weights = get_tanh_sinh_1d(n_pts)
        interior = np.count_nonzero((nodes <= 0.0) | (nodes >= 1.0))
        assert interior == 0, (
            f"n_pts {n_pts}: {interior} of {nodes.size} nodes sit on the boundary "
            f"(weights {weights[(nodes <= 0.0) | (nodes >= 1.0)].tolist()})"
        )
        assert np.unique(nodes).size == nodes.size, (
            f"n_pts {n_pts}: {nodes.size - np.unique(nodes).size} duplicated nodes"
        )
        # The consequence, asserted directly: the advertised integrand is finite and
        # correct. The floor is the neglected tail `2 * sqrt(delta)` with `delta ~ eps / 2`
        # the truncation gap, which is 2.1e-8 and is what the rule can reach in float64.
        singular = float(np.sum(weights / np.sqrt(nodes)))
        assert abs(singular - 2.0) <= _TANH_SINH_TAIL, (
            f"n_pts {n_pts}: integral of x**-0.5 is {singular}, not 2.0"
        )


# ---------------------------------------------------------------------------
# Zero intervals is documented as legal and produces a malformed knot vector
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="num_intervals=0 is documented as legal but the knot-vector factories either "
    "return NaN and inf or silently produce one interval instead of none",
)
def test_knot_factories_agree_on_zero_intervals() -> None:
    # `create_uniform_open_knots` and `create_uniform_periodic_knots` both document
    # `num_intervals` as "must be non-negative" and both validate only `>= 0`, so zero is
    # inside their stated contract. `create_cardinal_knots` rejects it with
    # "num_intervals must be at least 1". Three factories in one module, two answers to
    # whether the input is legal, and neither of the accepting two returns a usable vector:
    #
    #   create_uniform_periodic_knots(0, 1) -> [nan, 0.0, inf]
    #   create_uniform_periodic_knots(0, 3) -> [nan, nan, nan, 0.0, inf, inf, inf]
    #   create_uniform_open_knots(0, 3)     -> [0,0,0,0, 1,1,1,1]   (one interval, not zero)
    #
    # The periodic case comes from `np.linspace` over a zero-length span with a division by
    # the interval count; it emits only `RuntimeWarning: invalid value encountered in add`,
    # which nothing raises on, and the caller receives a knot vector full of NaN and inf.
    # `BsplineSpace1D` then rejects it for the wrong reason ("at least 2*degree+2
    # elements"), so the origin of the NaN is never reported.
    #
    # Either answer would be defensible -- reject zero as the cardinal factory does, or
    # return an empty-domain vector -- but the three must agree, and none may return NaN.
    for degree in (1, 2, 3):
        periodic = np.asarray(create_uniform_periodic_knots(0, degree))
        assert np.all(np.isfinite(periodic)), (
            f"create_uniform_periodic_knots(0, {degree}) returned non-finite knots: "
            f"{periodic.tolist()}"
        )
        open_knots = np.asarray(create_uniform_open_knots(0, degree))
        spans = int(np.unique(open_knots).size) - 1
        assert spans == 0, (
            f"create_uniform_open_knots(0, {degree}) returned {spans} interval(s): "
            f"{open_knots.tolist()}"
        )


# ---------------------------------------------------------------------------
# Degree reduction never returns on a periodic linear spline
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="needs SIGALRM to bound the call")
def test_reduce_degree_terminates_on_a_periodic_linear_spline() -> None:
    # FIXED by enforcing `_to_periodic_bspline_1d_impl`'s own documented precondition,
    # `1 <= m_bdy <= degree`. Kept as a regression guard with its original triggering
    # data, per this repository's convention that the fix PR un-xfails the tests it
    # closes.
    #
    # Found by the sweep the hard way: it stalled a 25952-case run indefinitely, which is
    # why the harness now bounds every case (`_core.CASE_TIMEOUT_SECONDS`).
    #
    # The trigger was exact and narrow -- periodic **and** degree 1 **and**
    # `reduce_degree`:
    #
    #   degree 1, periodic,   2 / 3 / 4 / 8 intervals    -> never returned
    #   degree 1, same knots, periodic=False             -> returned in 0.003 s
    #   degree 1, periodic,   domain [0, 1e-6] or [0, 1] -> never returned (scale was not
    #                                                       it)
    #   degree 0, periodic                               -> documented ValueError (the
    #                                                       decrement exceeds the degree)
    #   degree 2 and 3, periodic                         -> documented ValueError (residual
    #                                                       exceeds tolerance)
    #
    # The mechanism: `_degree_reduce_bspline` passes `m_bdy_new = m_bdy - decrement` to
    # `_to_periodic_bspline_1d_impl`, and a maximally smooth periodic space has
    # `m_bdy = 1`, so the argument is 0 at **every** degree. That violates the documented
    # `1 <= m_bdy <= degree`, which nothing checked. `_build_periodic_knot_vector` then
    # builds its per-period tile with multiplicity 0 at the seam, and its right-ghost
    # `while` loop has nothing to append: it increments `shift` forever.
    #
    # Degrees 2 and 3 escaped only by accident -- the C^0 seam check a few lines earlier
    # rejected them first -- so the narrowness of the trigger was a coincidence of check
    # ordering, not a property of degree 1. Degree 1 is the one case where nothing rejects
    # it first, because reducing it leaves a single control point and the seam check
    # compares that point with itself.
    #
    # A clean refusal is the right outcome here rather than a working reduction: the
    # result would be degree 0, and the periodic representation this library uses needs
    # `1 <= m_bdy <= degree`, a range that is empty at degree 0. With no ghost knots there
    # is nothing to wrap, and the periodic form of a piecewise constant is its open form.
    # `spline.to_open_bspline().reduce_degree(1)` does the reduction and returns in
    # milliseconds.
    knots = np.asarray(create_uniform_periodic_knots(4, 1), dtype=np.float64)
    space = BsplineSpace1D(knots, 1, periodic=True)
    spline = Bspline(
        BsplineSpace([space]),
        np.linspace(-1.0, 1.0, 2 * space.num_basis).reshape(space.num_basis, 2),
    )

    def _expire(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        raise _CallTimeout

    previous = signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, _HANG_BUDGET_SECONDS)
    try:
        # Terminating is the point, so the timeout is what this test really guards; the
        # message is asserted as well so that a *different* refusal cannot pass for the
        # fix, which is what the bare `except ValueError` of the xfail version allowed.
        with pytest.raises(ValueError, match=r"boundary multiplicity in \[1, degree\]"):
            spline.reduce_degree(1)
    except _CallTimeout:
        pytest.fail(
            f"reduce_degree(1) on a degree-1 periodic spline did not return within "
            f"{_HANG_BUDGET_SECONDS} s; the clamped equivalent takes 0.003 s"
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)

    # The escape route the message names, on the same data.
    reduced = spline.to_open_bspline().reduce_degree(1)
    assert reduced.degree[0] == 0
    assert not reduced.space.spaces[0].periodic


# ---------------------------------------------------------------------------
# The root finder certifies a value that is not a root
# ---------------------------------------------------------------------------


def test_find_roots_returns_only_genuine_roots() -> None:
    # FIXED, and kept as a regression guard with its original triggering data per this
    # repository's convention that the fix PR un-xfails the tests it closes. It took two
    # changes, and the second is what keeps the first from being a cure worse than the
    # disease.
    #
    # What it was, the most serious finding of the August 2026 triage: `find_roots`
    # returned a parameter at which the spline is not zero, silently, through the public
    # API, on a perfectly ordinary clamped cubic on the unit domain. Nothing warned.
    #
    # The data below is a degree-3 clamped uniform spline on [0, 1] with four intervals
    # and control points alternating +1/-1. It has exactly six sign changes. `find_roots`
    # returned six values; four of them were roots to 1e-16, and two -- 0.375 and 0.625 --
    # were not roots at all. The true zeros nearest them are at 0.369 and 0.630995, so the
    # returned values were off by about 0.006 in the parameter and left a residual of
    # 0.0208 on a curve whose values are bounded by 1.
    #
    # The mechanism, traced through a line-for-line pure-Python mirror of
    # `_bspline_roots_core.py` that reproduced the same wrong roots bit for bit:
    # `_track_zero` treated a repeated iterate (`x == previous_x`) as a certificate of
    # convergence, per Morken-Reimers Lemma 13/Corollary 14, and `_morken_reimers_roots`
    # then gated acceptance on
    #
    #     accepted = residual <= zero_tol if status == _STATUS_CANDIDATE else True
    #
    # so for `_STATUS_CONVERGED` the residual was hard-coded to 0.0 and never compared
    # against the actual function value. Here the first secant estimate lands on 0.375, a
    # single Boehm insertion there produces a control coefficient that is exactly 0.0 for
    # an algebraic reason (the symmetric alternating coefficients on exact rational
    # Greville abscissae), the next iterate is therefore 0.375 again, and the fixed-point
    # test fired after one step -- far from convergence. That is a theorem valid in exact
    # arithmetic applied to a floating-point iteration where a coefficient can reach zero
    # for reasons unrelated to being near a root.
    #
    # The fix makes every exit of `_track_zero` hand back `|f(x)|` and gates acceptance on
    # that residual uniformly, so a status records how the iteration stopped and never
    # whether the value may be reported. On its own that only makes the answer sound: it
    # drops 0.375 and 0.625 and reports four roots where there are six. The second change
    # restores the two: the repeated-iterate stop now tests Corollary 14's actual
    # hypothesis, `degree - 1` active knots collapsed onto the iterate, rather than the
    # bare repetition that a nearly horizontal control-polygon secant also produces.
    #
    # Distinct from the fabricated root closed in #291, which guarded the
    # `x >= knots[index + degree]` branch at a C^-1 knot of multiplicity degree + 1. This
    # knot vector has only simple interior knots and never reaches that branch.
    #
    # It was not a tolerance artifact and did not scale away: the same construction on
    # [0, 5] returned 3.125 with the same residual 0.0208 against a true zero at
    # 3.154997195131751.
    degree = 3
    knots = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0])
    spline = _scalar_spline(knots, degree, [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

    roots = np.asarray(find_roots(spline), dtype=np.float64).ravel()
    assert roots.size, "precondition: the sign-alternating spline has roots to find"

    # The bound is the two terms a returned root may legitimately carry, and nothing
    # else. Evaluation: de Boor over `degree + 1` stages on coefficients of magnitude 1,
    # so `4 * eps`. Location: `find_roots` documents `tol` as a *parametric* tolerance
    # defaulting to `get_strict(float64) = 1e-15`, and a root located to `dt` leaves
    # `|f'| * dt`; the hodograph bounds `|f'|` by `degree * max|delta c| / h_min`
    # = 3 * 2 / 0.25 = 24, and the documented stopping scale here is the domain length 1.
    # A safety factor of 8 covers the unmodelled constants of both, as elsewhere in this
    # suite. The result is 1.99e-13, against an observed residual of 2.1e-2 -- a factor of
    # 1.05e11, so no reasonable sharpening of this bound changes the verdict.
    eps = get_machine_epsilon(np.float64)
    slope_bound = degree * 2.0 / 0.25
    bound = 8.0 * ((degree + 1) * eps + slope_bound * 1e-15)
    residuals = np.abs(np.asarray(spline.evaluate(roots), dtype=np.float64).ravel())
    worst = float(np.max(residuals))
    assert worst <= bound, (
        f"find_roots returned {roots.tolist()}; |f| there is {residuals.tolist()}, "
        f"and the largest exceeds the derived bound {bound:.3e}"
    )

    # Guard the other side too, which the original xfail did not: an xfail is satisfied by
    # *any* failure, so returning four roots instead of six would have satisfied it while
    # losing two genuine zeros. The six below were obtained independently of this method,
    # by a two-million-point sign scan of the spline followed by bisection to convergence;
    # `1e-15` is the parametric tolerance `find_roots` documents, and the domain length is
    # 1, so a root is placed to that much and the comparison needs no safety factor beyond
    # the eight this suite uses elsewhere.
    expected = [
        0.06279438346444505,
        0.200641357614461,
        0.36900056097364975,
        0.6309994390263503,
        0.799358642385539,
        0.9372056165355549,
    ]
    np.testing.assert_allclose(roots, expected, rtol=0.0, atol=8.0 * 1e-15)


# ---------------------------------------------------------------------------
# Restriction silently returns a shorter domain than it was asked for
# ---------------------------------------------------------------------------


def test_restrict_spans_the_requested_window() -> None:
    # FIXED, though not at this site: `tol` here is `space.tolerance`, which now carries
    # the knot vector's own magnitude (`8 * eps * max(span, |knots[0]|, |knots[-1]|)`)
    # instead of a per-dtype constant. Since one ulp of any coordinate in the vector is at
    # most `eps * scale`, the addition below now always moves the bound by at least eight
    # ulp and can no longer be a no-op at any magnitude -- which is the whole mechanism
    # this test pinned. It also steps no further than intended, because distinct knots are
    # now separated by more than that same tolerance by construction. Kept as a regression
    # guard with its original triggering data, per this repository's convention that the
    # fix PR un-xfails the tests it closes.
    #
    # The site's own tolerance policy was not otherwise revisited, so if `restrict` grows
    # a tolerance of its own this guard is what will catch a return of the mechanism.
    #
    # A silent wrong answer through the public API, and the reason it stayed hidden is
    # worth as much as the bug: `Bspline.restrict` carried no invariant in the sweep, so
    # the case was graded only on whether it raised.
    #
    # `_restrict_bspline_impl` (`_bspline_restrict.py`, the extraction step) locates the
    # end of the restricted knot vector with
    #
    #     i_end = int(np.searchsorted(refined_knots, b_new + tol)) - 1
    #
    # The `+ tol` exists to step past the last of the `degree + 1` copies of `b_new` that
    # the preceding insertion produced. But `tol` is the space's *absolute* tolerance,
    # `get_strict(float64) = 1e-15`, and `b_new + 1e-15 == b_new` exactly once half an ulp
    # of `b_new` exceeds 1e-15 -- which is at `|b_new| = 16`, since ulp(x) is 1.78e-15 on
    # [8, 16) and 3.55e-15 on [16, 32). Past that the search finds the *first* copy
    # instead of one past the last, and `degree + 1` knots are dropped from the top.
    #
    # The threshold is exact and was bisected: on [0, span] with four intervals and the
    # window at 25%/75%, span 20 (upper bound 15) is correct and span 24 (upper bound 18)
    # is not.
    #
    # Two faces, and only the loud one was visible before:
    #   * few intervals  -- the truncated vector falls below `2 * degree + 2` and
    #     `BsplineSpace1D` raises "knots must have at least 2*degree+2 elements", blaming
    #     the knot count for an index computed one place too low. 30 sweep findings.
    #   * many intervals -- enough knots survive, and the call *returns a spline over the
    #     wrong domain*. This is the half the test pins first.
    #
    # Attribution: this is the absolute-tolerance-versus-coordinate-magnitude family that
    # already owns two open tests in this file, so the fix belongs with that workstream.
    # It is recorded separately because the site and the failure mode are different --
    # index arithmetic on a knot search, not a snapping merge -- and because a correction
    # to `_snap_knots` would not touch it.
    degree = 2
    span = 100.0
    n_intervals = 20
    breaks = np.linspace(0.0, span, n_intervals + 1)
    knots = np.concatenate([np.full(degree, 0.0), breaks, np.full(degree, span)])
    space = BsplineSpace1D(knots, degree)
    n_basis = space.num_basis
    curve = Bspline(
        BsplineSpace([space]), np.linspace(0.0, 1.0, n_basis, dtype=np.float64).reshape(-1, 1)
    )

    lower, upper = 0.25 * span, 0.75 * span
    assert upper > 16.0, "precondition: the upper bound is past the 16.0 threshold"

    restricted = curve.restrict((lower, upper))
    restricted_knots = np.asarray(restricted.space.spaces[0].knots, dtype=np.float64)
    got_upper = float(restricted_knots[restricted_knots.size - degree - 1])
    # The endpoints are exactly representable here (25.0 and 75.0 are dyadic), so this is
    # an equality, not a tolerance.
    assert got_upper == upper, (
        f"restrict((({lower}, {upper}))) returned a domain ending at {got_upper}, "
        f"{upper - got_upper} short of the window that was asked for"
    )

    # The loud face, on the same mechanism with fewer intervals: the truncated vector is
    # too short and the constructor blames the caller's knot count.
    coarse_breaks = np.linspace(0.0, span, 5)
    coarse_knots = np.concatenate([np.full(degree, 0.0), coarse_breaks, np.full(degree, span)])
    coarse_space = BsplineSpace1D(coarse_knots, degree)
    coarse_n = coarse_space.num_basis
    coarse = Bspline(
        BsplineSpace([coarse_space]),
        np.linspace(0.0, 1.0, coarse_n, dtype=np.float64).reshape(-1, 1),
    )
    coarse_restricted = coarse.restrict((lower, upper))
    assert coarse_restricted.space.spaces[0].num_basis > 0


# ---------------------------------------------------------------------------
# Lagrange extraction is the only one of the three that a degree-0 space defeats
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="tabulate_Lagrange_extraction_operators reaches a change-of-basis builder "
    "that requires degree >= 1, while the Bezier and cardinal extractions of the same "
    "degree-0 space both succeed",
)
def test_lagrange_extraction_handles_a_degree_zero_space() -> None:
    # A degree-0 `BsplineSpace1D` is legal -- the constructor accepts it, and it is what
    # `subdivide(n, regularity=-1)` produces -- and two of its three extraction operators
    # are perfectly happy with it. The third is not: it reaches
    # `compute_lagrange_to_bernstein_1d`, whose documented precondition is "Must be at
    # least 1", and the `ValueError` that surfaces ("Degree must at least 1", the
    # message's own grammar) names a degree the caller never passed.
    #
    # This is a contract inconsistency rather than a numerical defect: the extraction of
    # a piecewise-constant space is well defined -- the Lagrange basis of degree 0 is the
    # single constant function 1, so every element operator is the 1x1 identity -- and
    # nothing about the mathematics forces a refusal. Either the operator should be that
    # identity, or `tabulate_Lagrange_extraction_operators` should document a degree
    # floor its two siblings do not have. It does neither.
    #
    # 22 of the sweep's findings are this, spread across both dtypes and every domain, and
    # they are all one cause: `degree == 0`. (A 23rd finding on the same entry point at
    # degree 62 in float32 is *not* this mechanism -- it is the float32 knot-collapse
    # already pinned by `test_snapping_keeps_knots_the_format_can_resolve` -- which is why
    # the assertion below is at degree 0 only.)
    space = BsplineSpace1D(np.array([0.0, 0.25, 0.5, 0.75, 1.0]), 0)
    assert space.num_intervals == 4, "precondition: four piecewise-constant elements"

    # The two that work, asserted first so a regression in them cannot be mistaken for
    # this bug.
    bezier = np.asarray(space.tabulate_Bezier_extraction_operators())
    assert bezier.shape == (4, 1, 1)
    cardinal = np.asarray(space.tabulate_cardinal_extraction_operators())
    assert cardinal.shape[0] == 4

    lagrange = np.asarray(space.tabulate_Lagrange_extraction_operators())
    assert lagrange.shape == (4, 1, 1), (
        f"degree-0 Lagrange extraction returned shape {lagrange.shape}, expected one "
        f"1x1 identity per element"
    )


# ---------------------------------------------------------------------------
# The cardinal change of basis raises a bare LinAlgError on a legal degree
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="the cardinal change-of-basis builders raise numpy's LinAlgError on a legal "
    "(degree, dtype) pair, an exception type none of them documents and which the caller "
    "cannot tell apart from an illegal argument",
)
def test_cardinal_change_of_basis_reports_its_own_conditioning_limit() -> None:
    # Found only because the August 2026 triage completed the sweep's verdict flags: with
    # no `must_succeed`, this read as an UNDOCUMENTED_REJECTION -- a suspicion nobody had
    # looked at -- because `numpy.linalg.LinAlgError` subclasses `ValueError` and the
    # runner's `Raises:`-driven rule cannot see that the *reason* is undocumented. Five
    # `basis` cases and fifteen `bspline` extraction cases are this one cause.
    #
    # The numerics are not in dispute. The cardinal-to-Bernstein matrix's condition number
    # grows like 4 ** degree; measured with `numpy.linalg.cond` on the returned matrix it
    # is 15 at degree 3, 3.5e10 at degree 10, 1.2e21 at degree 20 and 5.2e32 at degree 30,
    # while float32 can resolve at most 1 / eps = 8.4e6. So at high degree in float32 the
    # inverse genuinely cannot be formed, and refusing is right.
    #
    # What is wrong is the contract. Every one of these builders documents exactly one
    # exception -- "ValueError: If degree is negative, dtype is not float32 or float64, or
    # if `out` is provided and has incorrect shape or dtype" -- and then, for a degree that
    # is not negative and a dtype that is float32, raises `LinAlgError: Singular matrix`
    # from three frames down. The caller is told nothing about a degree limit, cannot
    # discover one from the signature, and gets a message that describes an internal matrix
    # rather than the argument that caused it. Either the limit belongs in the docstring
    # with a `ValueError` that names it, or `LinAlgError` belongs in `Raises:`.
    #
    # The threshold measured on this machine, for `compute_cardinal_to_bernstein_1d`:
    #
    #   float32   degree 3, 10, 15, 20, 25, 30 -> returns    degree 40, 62 -> LinAlgError
    #   float64   degree 3 ... 62              -> returns
    #
    # so the assertion below uses float32 at degree 62, well past the cliff, and pins
    # float64 alongside it to keep the failure attributable to precision rather than to
    # degree alone.
    degree = 62

    # float64 handles the same degree, so this is a precision limit and not a degree one.
    reference = np.asarray(compute_cardinal_to_bernstein_1d(degree, np.float64))
    assert reference.shape == (degree + 1, degree + 1)

    try:
        matrix = compute_cardinal_to_bernstein_1d(degree, np.float32)
    except ValueError as exc:
        # `LinAlgError` is a `ValueError` subclass, so this catches both. The distinction
        # the test insists on is the one the caller has to make: a message naming the
        # argument at fault, not numpy's internal one.
        assert "Singular matrix" not in str(exc), (
            f"compute_cardinal_to_bernstein_1d({degree}, float32) raised numpy's "
            f"{type(exc).__name__}: {exc} -- the documented ValueError should name the "
            f"degree or the precision, and `Raises:` should list whatever is thrown"
        )
        raise
    assert np.asarray(matrix).shape == (degree + 1, degree + 1)


# ---------------------------------------------------------------------------
# The unique-knot accessor returns knots the space does not contain
# ---------------------------------------------------------------------------


def test_unique_knots_are_knots_of_the_space() -> None:
    # FIXED by grouping knots by relative gap and returning the class's first *stored*
    # knot, and by rebuilding `_snap_knots` on top of the same helper so the two can no
    # longer disagree. Kept as a regression guard with its original triggering data, per
    # this repository's convention that the fix PR un-xfails the tests it closes.
    #
    # What it was: `get_unique_knots_and_multiplicity` is meant to *report* the distinct
    # knots of a space, and it reported different numbers. On a float32 space over
    # [0, 1e-6] whose stored interior knots are 2.5e-7, 5.0e-7 and 7.5e-7, the accessor
    # answered 2.0e-7, 5.0e-7 and 8.0e-7 -- each moved by 20% of its own value, to a knot
    # the space does not have.
    #
    # `_get_unique_knots_and_multiplicity_impl` rounded
    # onto a grid of width `tol` to decide which knots are the same,
    #
    #     scale = dtype.type(1.0 / tol)
    #     rounded_knots = np.round(knots * scale) / scale
    #
    # and then returned `rounded_knots[unique_rounded_knots_ids]` -- the rounded values --
    # although `unique_rounded_knots_ids` already indexed the *original* array and
    # `knots[...]` would have returned the knots themselves. Its sibling `_snap_knots`
    # did the same grouping and deliberately did not make this mistake: it wrote back
    # `np.mean(self._knots[mask])`, with a comment about the care being taken. Two
    # implementations of one idea, disagreeing. There is now one: `_snap_knots` calls this
    # helper and repeats its classes, so the space and its accessor cannot diverge.
    #
    # `tol` was the space's absolute tolerance (`get_strict(float32) = 1e-7`), so the
    # grid was coarse compared with a 2.5e-7 spacing and 2.5 rounded to 2 while 7.5
    # rounded to 8. Nothing about the input is unrepresentable: the float32 ulp at 2.5e-7
    # is 1.7e-14, seven orders of magnitude finer than the movement.
    #
    # The float64 face was milder and still wrong: 2.5e-07 came back as
    # 2.5000000000000004e-07, because `round(x * scale) / scale` is not the identity even
    # when the grouping changes nothing. The accessor never returned the array it was
    # given; it now returns entries of it.
    #
    # Consequences, and why this outranks a reporting nuisance. The helper feeds
    # `to_beziers`, knot insertion and removal, quasi-interpolation, and the product's
    # breakpoint mesh. The last is what the sweep found: `Bspline.multiply` builds the
    # product knot vector from these relocated breakpoints, so squaring a degree-2 float32
    # spline on [0, 1e-6] returns a curve whose interior knots sit at 0.2/0.5/0.8 of the
    # domain instead of 0.25/0.5/0.75, and which is wrong by 8.0e-2 *relative* at points
    # nowhere near a knot. Silent, through the public API. The same operands in float64,
    # and the same float32 operands on the unit domain, were exact to rounding -- which is
    # what identified it as the absolute-tolerance-versus-coordinate-magnitude family it
    # was fixed with.
    #
    # Found only after the probe's product invariant stopped sampling at the C^-1
    # breakpoint: the 8% error had been sitting underneath a 15-40% artifact.
    hi = 1e-6
    knots = np.array([0.0, 0.0, 0.0, 0.25 * hi, 0.5 * hi, 0.75 * hi, hi, hi, hi], dtype=np.float32)
    space = BsplineSpace1D(knots, 2)
    stored = np.asarray(space.knots, dtype=np.float64)
    assert np.array_equal(np.asarray(space.knots), knots), (
        "precondition: construction leaves these knots alone -- they do not share a "
        "snapping cell, so this is not the already-pinned `_snap_knots` merge"
    )

    unique, _ = space.get_unique_knots_and_multiplicity(in_domain=True)
    reported = np.asarray(unique, dtype=np.float64)
    missing = [float(x) for x in reported if not np.any(stored == np.float32(x))]
    assert not missing, (
        f"get_unique_knots_and_multiplicity reported {reported.tolist()}, which contains "
        f"{missing} -- values that are not knots of the space {np.unique(stored).tolist()}"
    )
