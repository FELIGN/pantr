"""Known-bug regressions from the adversarial parameter sweep (August 2026).

Each test here reproduces a bug the sweep in ``tools/adversarial_sweep/`` found on inputs
the rest of the suite does not contain. A bug that is still open carries
``xfail(strict=True)``; when it is fixed the test starts passing, pytest reports a strict
XPASS failure, and the marker comes off, promoting the test to a permanent guard on the
same data. That is the convention ``tests/test_review_regressions.py`` follows for the
June 2026 review, whose markers have all since been removed.

Six markers have already come off here: the domain-membership test, closed by the
``np.isclose`` tolerance-leak fix in #289; the tanh-sinh endpoint test, closed by
truncating the rule where the endpoint gap stops being resolvable; the Lagrange
reproducibility test, closed by seeding the barycentric node permutation; the float32
degree-elevation test, closed by allocating the kernels' knot output in the input's dtype;
the periodic degree-reduction hang, closed by enforcing the periodic conversion's own
boundary-multiplicity precondition; and the degree-elevation counter mismatch, closed by
emitting the unshared Bézier coefficient at a C^-1 knot and by letting the segment sweep
reach the final span of a degree-0 knot vector. Three remain open, all in the tolerance
workstream.

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
)
from pantr.bspline._bspline_degree_core import _degree_elevate_1d_core
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
    assert float(space.tolerance) < 1e-9, "precondition: the space carries a strict tolerance"

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


@pytest.mark.xfail(
    strict=True,
    reason="_snap_knots averages each group of equal knots, and np.mean over a run of "
    "degree + 1 identical knots is not the identity at large coordinate magnitude",
)
def test_snapping_preserves_a_run_of_identical_knots() -> None:
    # `_snap_knots` (`_bspline_space_1d.py:215`) replaces every group of knots that round
    # to the same grid point by `np.mean(group, dtype=self.dtype)`. For a clamped end the
    # group is `degree + 1` copies of one value, so the mean must be that value exactly.
    # It is not: the summation loses the low bits once `(degree + 1) * |knot|` passes the
    # format's exact-integer range (2^24 for float32, 2^53 for float64), and the knot
    # moves by 0.125. The consequence is that the space's *reported domain* differs from
    # the one the caller asked for, silently.
    #
    # Which run lengths survive is **not** a clean rule, and it is worth not pretending
    # otherwise: it falls out of NumPy's pairwise-summation blocking interacting with the
    # value's own bit pattern. Measured for float64 at 1e15 + 1, runs of 11, 13, 14, 15,
    # 18-23 and 26-31 copies are already inexact while 12, 16, 17, 24, 25 and 32 are
    # exact. So this bites from **degree 10** in float64 at that magnitude, and the two
    # degrees asserted below are simply the ones the sweep happened to visit -- do not
    # read them as a threshold.
    for dtype, base in ((np.float32, 1e6), (np.float64, 1e15)):
        degree = 62
        hi = np.asarray(base + 1.0, dtype=dtype)
        raw = np.concatenate(
            [
                np.full(degree + 1, base, dtype=dtype),
                [np.asarray(base + 0.5, dtype=dtype)],
                np.full(degree + 1, hi, dtype=dtype),
            ]
        )
        stored = np.asarray(BsplineSpace1D(raw, degree).knots)
        assert stored[-1] == hi, (
            f"{np.dtype(dtype).name}: snapping moved a run of {degree + 1} identical "
            f"knots from {float(hi)!r} to {float(stored[-1])!r}"
        )


@pytest.mark.xfail(
    strict=True,
    reason="the knot-snapping grid is an absolute tolerance, so on a small domain it "
    "merges knots the format resolves easily and the space silently loses intervals",
)
def test_snapping_keeps_knots_the_format_can_resolve() -> None:
    # `_snap_knots` rounds onto a grid of width `tolerance`, which is absolute
    # (`get_strict(float32) = 1e-7`, `get_strict(float64) = 1e-15`) while the quantity it
    # grades -- a gap between knots -- carries the domain's scale. On `[0, 1e-6]` with 20
    # equal intervals the spacing is 5e-8, below the float32 grid, so half the knots are
    # merged and the space reports 10 intervals instead of 20. The float32 ulp at 1e-6 is
    # 6e-14, so the format resolves those knots with six orders to spare: nothing about
    # the input is unrepresentable.
    #
    # This is the small-domain face of an already-recorded tolerance-policy defect whose
    # large-domain face is that from |knot| ~ 5 upward the same grid merges nothing at all.
    # Further consequences measured at degree 62 on `[0, 1e-6]` in float32, where a
    # periodic knot vector of 63 intervals collapses to 10: zero-length spans make
    # `tabulate_Bezier_extraction_operators` raise a bare `ZeroDivisionError`, and the
    # basis sums to 0 instead of 1 at the right endpoint.
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
