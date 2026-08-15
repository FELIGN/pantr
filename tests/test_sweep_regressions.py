"""Known-bug regressions from the adversarial parameter sweep (August 2026).

Every test here is an ``xfail(strict=True)`` reproduction of a bug the sweep in
``tools/adversarial_sweep/`` found on inputs the rest of the suite does not contain. When
a bug is fixed the corresponding test starts passing, pytest reports a strict XPASS
failure, and the marker should be removed, promoting the test to a permanent guard. That
is the same convention ``tests/test_review_regressions.py`` follows for the June 2026
review, whose markers have all since been removed.

One test per **root cause**, not per symptom: several of these root causes have many
triggering combinations, and each test names them in a comment rather than repeating
itself. Regenerate the findings with::

    conda run -n pantr python tools/sweep.py --profile full

The triggering data is hardcoded rather than swept, so a test failure here is a statement
about this exact input and nothing else.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Degree elevation: the kernel's two returns disagree
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="_degree_elevate_1d_core returns a knot vector and control points that "
    "disagree once any interior knot has multiplicity degree + 1",
)
def test_degree_elevation_outputs_are_mutually_consistent() -> None:
    # `_degree_elevate_1d_core` returns (control_points, knots), which are only usable
    # together when `control_points.shape[0] == knots.size - new_degree - 1`. Those two
    # come from counters `cind` and `kind` maintained independently through the
    # Piegl-Tiller A5.9 walk and combined in a single return
    # (`_bspline_degree_core.py:253`), and they diverge.
    #
    # **Measured**: the deficit equals the number of interior knots at multiplicity
    # degree + 1 (1, 2 and 3 such knots give deficit 1, 2, 3) and does not grow with the
    # increment (an increment of 2 still gives deficit 1).
    # **Inferred, not verified against the published algorithm**: A5.9 walks Bezier
    # segments joined at interior knots of multiplicity at most `degree`, so adjacent
    # segments share an endpoint; at multiplicity degree + 1 they share nothing and the
    # shared point is subtracted anyway. Anyone fixing this should trace the `lbz`/`rbz`
    # bookkeeping rather than take that sentence on trust.
    #
    # Two faces, one cause:
    #   * degree >= 1 -- the knot vector is right and the points are short, so
    #     `Bspline.__init__` rejects the result with a message about the *caller's*
    #     control-point count.
    #   * degree 0 -- every interior knot has multiplicity 1 = degree + 1, so the whole
    #     vector is pathological, both outputs come out short, the sizes agree by
    #     accident, and the wrong answer is returned silently (see the second half).
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
    # control points comes back with a collapsed domain and a duplicated control point.
    spline = _scalar_spline(np.array([0.0, 0.5, 1.0]), 0, [1.0, 2.0])
    elevated = spline.elevate_degree(1)
    elevated_knots = np.asarray(elevated.space.spaces[0].knots)
    assert elevated_knots[-1] > elevated_knots[0], (
        f"degree elevation collapsed the domain to a point: {elevated_knots.tolist()}"
    )


@pytest.mark.xfail(
    strict=True,
    reason="_degree_elevate_1d_core allocates its knot output as float64 regardless of "
    "the input dtype, so degree elevation is unusable in float32",
)
def test_degree_elevation_preserves_float32() -> None:
    # `_bspline_degree_core.py:136-137` allocates the two outputs with different dtypes:
    # `ik = np.zeros(max_new_knots, dtype=np.float64)` is hardcoded, while
    # `ic = np.zeros(..., dtype=ctrl.dtype)` follows the input. The knot vector therefore
    # comes back float64 while the control points stay float32, and the two faces of that
    # are:
    #   * clamped -- `Bspline.__init__` rejects the pair, so `elevate_degree` raises for
    #     **every** float32 spline, at every degree and every knot count. Measured on
    #     degrees 1-3 with 3, 4 and 6 breakpoints: nine for nine.
    #   * periodic -- the round trip through open form converts the control points too, so
    #     both come back float64 and the call *succeeds*, silently discarding the
    #     caller's choice of precision.
    #
    # The second face is what makes this worth a test rather than a bug report: a silent
    # dtype promotion is invisible until something downstream compares dtypes, and it
    # doubles memory and halves throughput without a word.
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
    periodic_knots = (np.arange(-2, 6, dtype=np.float64) / 3.0).astype(np.float32)
    periodic = Bspline(
        BsplineSpace([BsplineSpace1D(periodic_knots, 2, periodic=True)]),
        np.arange(5, dtype=np.float32).reshape(5, 1),
    )
    promoted = periodic.elevate_degree(1)
    assert np.dtype(promoted.dtype) == np.float32, (
        f"degree elevation silently promoted a periodic float32 spline to "
        f"{np.dtype(promoted.dtype).name}"
    )


# ---------------------------------------------------------------------------
# Tolerance policy: np.isclose keeps its default rtol
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="domain membership is tested with np.isclose(..., atol=tol), which keeps the "
    "default rtol=1e-5, so the effective tolerance is 1e-5 * |knot| instead of tol",
)
def test_out_of_domain_points_are_rejected_at_large_knot_magnitude() -> None:
    # `_is_in_domain` (`_bspline_knots.py:172-173`) accepts a point when
    # `np.isclose(knot_end, pt, atol=tol)`. Setting `atol` does not clear `rtol`, which
    # stays at numpy's default 1e-5, so the test is really
    # `|pt - knot_end| <= tol + 1e-5 * |knot_end|`. On a domain of length 1 placed at 1e6
    # that admits points up to 10.00001 outside -- ten domain lengths -- while the space's
    # stated tolerance is 1e-15.
    #
    # This is one root cause with 26 sites: every `np.isclose(..., atol=tol)` in
    # `pantr.bspline` leaves `rtol` at its default, across `_bspline_restrict.py` (10),
    # `_bspline_knots.py` (5), `_bspline_knot_insertion.py` (4),
    # `_bspline_knot_removal.py` (2), `_bspline_product.py` (2),
    # `_bspline_space_1d.py` (2) and `_bspline_split.py` (1) -- not one of them passes
    # `rtol`. (A 27th call, in `_bspline_quasi_interpolation.py`, is inside a doctest and
    # so not a library site.) Consequences already
    # measured, all at |knot| ~ 1e6: `tabulate_basis` accepts a point 10 units outside a
    # unit-length domain and returns a polynomial extrapolation (max|B| = 640 at degree 2,
    # 4.3e43 at degree 62) instead of raising; `remove_knots` refuses to remove the
    # interior knot 1000000.3125, calling it the domain start; `insert_knots` reports a
    # false multiplicity clash between knots 0.0625 apart. The author knew the trap --
    # `_snap_knots` (`_bspline_space_1d.py:211-215`) carries a comment warning about
    # exactly it -- and avoided it in that one place.
    lo, hi = 1e6, 1e6 + 1.0
    knots = np.concatenate([np.full(3, lo), [lo + 0.5], np.full(3, hi)])
    space = BsplineSpace1D(knots, 2)
    assert float(space.tolerance) < 1e-9, "precondition: the space carries a strict tolerance"

    with pytest.raises(ValueError, match="outside the knot vector domain"):
        # 10 domain lengths past the right end.
        space.tabulate_basis(np.array([hi + 10.0]))


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


@pytest.mark.xfail(
    strict=True,
    reason="tabulate_lagrange_1d builds a scipy BarycentricInterpolator without the rng "
    "argument, so an unseeded global RNG permutes the nodes and the result changes "
    "between calls",
)
def test_lagrange_tabulation_is_reproducible() -> None:
    # `_basis_lagrange.py:104` constructs `BarycentricInterpolator(nodes_sorted, y_sorted)`
    # with no `rng`. scipy then draws from the unseeded global `numpy.random` state and
    # applies `rng.permutation(n)` to the nodes before computing the barycentric weights
    # (`scipy/interpolate/_polyint.py:708-734`); its own documentation says "Specify `rng`
    # for repeatable interpolation".
    #
    # So the same call returns different values in different processes. This is not
    # floating-point nondeterminism with a bound one could derive -- it is an unseeded
    # RNG, and the spread grows with degree: about 1 ulp at degree 3-5, 4.18 absolute on a
    # value scale of 3.75e7 at degree 62 (relative 1.1e-7), and `inf` versus 1e16 across
    # separate processes at degree 62 evaluated outside [0, 1]. Everything downstream
    # inherits it: `tabulate_lagrange`, `compute_lagrange_to_bernstein_1d`,
    # `tabulate_Lagrange_extraction_operators`, and `SpanwiseElementExtraction` with the
    # Lagrange target.
    #
    # Reseeding the global state is what makes the failure deterministic here; in
    # production the seeds differ because nobody set one.
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


@pytest.mark.xfail(
    strict=True,
    reason="get_tanh_sinh_1d snaps near-endpoint nodes onto 0 and 1 but keeps them, with "
    "a nonzero weight and sometimes duplicated, so the rule evaluates at the endpoints",
)
def test_tanh_sinh_nodes_are_interior_and_distinct() -> None:
    # A double-exponential rule exists to avoid evaluating at the endpoints:
    # `get_tanh_sinh_1d`'s own docstring (`quad.py:370-372`) advertises it as "well suited
    # for integrands with endpoint singularities". Nodes that underflow to the boundary
    # are snapped there, which the docstring records only as making the returned count
    # smaller (`:373-375`) -- but the snapped node is *kept*, with a nonzero weight
    # (6.5e-17 at n_pts = 110), and from n_pts = 53 a second node snaps onto the same
    # boundary and is returned twice.
    #
    # Consequence: for f(x) = 1/sqrt(x), the advertised use case, the rule returns `inf`
    # rather than 2.0 at every n_pts >= 45. Weights still sum to 1 throughout, so the
    # module's own doctest cannot see it.
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
