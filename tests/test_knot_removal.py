"""Tests for B-spline knot removal."""

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_curve_p2() -> Bspline:
    """Create a degree-2 B-spline curve with 4 control points in 3D.

    Knots: [0,0,0,0.5,1,1,1] -> 4 basis functions, degree 2.
    """
    knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    space = BsplineSpace([BsplineSpace1D(knots, 2)])
    cp = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [3.0, 0.0, 0.0]])
    return Bspline(space, cp)


def _make_curve_p3() -> Bspline:
    """Create a degree-3 B-spline curve with 5 control points in 2D.

    Knots: [0,0,0,0,0.5,1,1,1,1] -> 5 basis, degree 3.
    """
    knots = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    space = BsplineSpace([BsplineSpace1D(knots, 3)])
    cp = np.array([[0.0, 0.0], [1.0, 3.0], [2.0, 2.0], [3.0, 3.0], [4.0, 0.0]])
    return Bspline(space, cp)


def _make_surface() -> Bspline:
    """Create a 2D B-spline surface with an interior knot in the u-direction.

    Degree (2, 1), rank 2.
    """
    knots_u = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    knots_v = np.array([0.0, 0.0, 1.0, 1.0])
    space_u = BsplineSpace1D(knots_u, 2)
    space_v = BsplineSpace1D(knots_v, 1)
    space = BsplineSpace([space_u, space_v])
    cp = np.arange(16, dtype=np.float64).reshape(4, 2, 2)
    return Bspline(space, cp)


def _assert_same_geometry(a: Bspline, b: Bspline, atol: float = 1e-12) -> None:
    """Assert two B-splines represent the same geometry via knot insertion.

    Inserts all knots from ``b`` into ``a`` and vice versa to get a common
    representation, then compares control points.
    """
    for i in range(a.dim):
        knots_a = a.space.spaces[i].knots
        knots_b = b.space.spaces[i].knots

        # Knots to insert into a to match b.
        extra_b = _knot_difference(knots_b, knots_a, atol)
        if extra_b.size > 0:
            insert_a: list[npt.NDArray[np.float32 | np.float64] | None] = [None] * a.dim
            insert_a[i] = extra_b
            a = a.insert_knots(insert_a)

        # Knots to insert into b to match a.
        extra_a = _knot_difference(knots_a, knots_b, atol)
        if extra_a.size > 0:
            insert_b: list[npt.NDArray[np.float32 | np.float64] | None] = [None] * b.dim
            insert_b[i] = extra_a
            b = b.insert_knots(insert_b)

    np.testing.assert_allclose(a.control_points, b.control_points, atol=atol)


def _knot_difference(
    knots_a: npt.NDArray[np.float32 | np.float64],
    knots_b: npt.NDArray[np.float32 | np.float64],
    atol: float,
) -> npt.NDArray[np.float32 | np.float64]:
    """Return knots in ``a`` that are not in ``b`` (with multiplicity)."""
    remaining = list(knots_b)
    diff = []
    for val in knots_a:
        found = False
        for j, rem in enumerate(remaining):
            if abs(val - rem) <= atol:
                remaining.pop(j)
                found = True
                break
        if not found:
            diff.append(val)
    return np.array(diff, dtype=knots_a.dtype)


# ===========================================================================
# Round-trip: insert then remove
# ===========================================================================


class TestRoundTrip:
    """Insert knots, then remove them, and verify geometry is preserved."""

    def test_insert_remove_single_knot_p2(self) -> None:
        """Insert then remove a single knot on a degree-2 curve."""
        orig = _make_curve_p2()
        refined = orig.insert_knots(np.array([0.25]))
        reduced = refined.remove_knots(0.25)

        # Knot vectors should match original.
        np.testing.assert_allclose(
            reduced.space.spaces[0].knots, orig.space.spaces[0].knots, atol=1e-14
        )
        # Control points should match original.
        np.testing.assert_allclose(reduced.control_points, orig.control_points, atol=1e-12)

    def test_insert_remove_single_knot_p3(self) -> None:
        """Insert then remove a single knot on a degree-3 curve."""
        orig = _make_curve_p3()
        refined = orig.insert_knots(np.array([0.25]))
        reduced = refined.remove_knots(0.25)

        np.testing.assert_allclose(
            reduced.space.spaces[0].knots, orig.space.spaces[0].knots, atol=1e-14
        )
        np.testing.assert_allclose(reduced.control_points, orig.control_points, atol=1e-12)

    def test_insert_remove_multiple_same_knot(self) -> None:
        """Insert a knot twice, then remove it twice."""
        orig = _make_curve_p3()
        refined = orig.insert_knots(np.array([0.3, 0.3]))
        reduced = refined.remove_knots(0.3, num=2)

        np.testing.assert_allclose(
            reduced.space.spaces[0].knots, orig.space.spaces[0].knots, atol=1e-14
        )
        np.testing.assert_allclose(reduced.control_points, orig.control_points, atol=1e-12)

    def test_insert_remove_multiple_distinct_knots(self) -> None:
        """Insert two different knots, then remove both."""
        orig = _make_curve_p3()
        refined = orig.insert_knots(np.array([0.25, 0.75]))
        reduced = refined.remove_knots(np.array([0.25, 0.75]))

        np.testing.assert_allclose(
            reduced.space.spaces[0].knots, orig.space.spaces[0].knots, atol=1e-14
        )
        np.testing.assert_allclose(reduced.control_points, orig.control_points, atol=1e-12)

    def test_remove_existing_knot_preserves_geometry(self) -> None:
        """Removing an original interior knot still preserves geometry (within tol)."""
        orig = _make_curve_p2()
        reduced = orig.remove_knots(0.5)
        _assert_same_geometry(orig, reduced)


# ===========================================================================
# Partial removal
# ===========================================================================


class TestPartialRemoval:
    """Test removing fewer knots than the full multiplicity."""

    def test_remove_one_of_two(self) -> None:
        """Insert a knot twice, remove only once — multiplicity drops by 1."""
        orig = _make_curve_p3()
        refined = orig.insert_knots(np.array([0.3, 0.3]))

        reduced = refined.remove_knots(0.3, num=1)

        # One copy should remain.
        knots = reduced.space.spaces[0].knots
        count = np.sum(np.isclose(knots, 0.3, atol=1e-12))
        assert count == 1

    def test_num_none_removes_all(self) -> None:
        """num=None should remove as many times as possible."""
        orig = _make_curve_p3()
        refined = orig.insert_knots(np.array([0.3, 0.3]))

        reduced = refined.remove_knots(0.3, num=None)

        knots = reduced.space.spaces[0].knots
        count = np.sum(np.isclose(knots, 0.3, atol=1e-12))
        assert count == 0


# ===========================================================================
# Multi-dimensional
# ===========================================================================


class TestMultiDim:
    """Test knot removal on multi-dimensional B-splines."""

    def test_surface_remove_u_direction(self) -> None:
        """Remove an inserted interior knot from the u-direction of a surface."""
        orig = _make_surface()
        refined = orig.insert_knots([np.array([0.25]), None])
        reduced = refined.remove_knots([np.array([0.25]), None])

        np.testing.assert_allclose(
            reduced.space.spaces[0].knots, orig.space.spaces[0].knots, atol=1e-14
        )
        np.testing.assert_allclose(reduced.control_points, orig.control_points, atol=1e-12)


# ===========================================================================
# Rational (NURBS) support
# ===========================================================================


class TestRational:
    """Test knot removal on rational B-splines (NURBS)."""

    def test_nurbs_round_trip(self) -> None:
        """Insert then remove a knot on a NURBS curve."""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        space = BsplineSpace([BsplineSpace1D(knots, 2)])
        # Homogeneous coords: (w*x, w*y, w) — a rational quadratic.
        cp = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 0.5], [2.0, 0.0, 1.0]])
        nurbs = Bspline(space, cp, is_rational=True)

        refined = nurbs.insert_knots(np.array([0.5]))
        reduced = refined.remove_knots(0.5)

        np.testing.assert_allclose(
            reduced.space.spaces[0].knots, nurbs.space.spaces[0].knots, atol=1e-14
        )
        np.testing.assert_allclose(reduced.control_points, nurbs.control_points, atol=1e-12)


# ===========================================================================
# Error handling
# ===========================================================================


class TestErrors:
    """Test error conditions for remove_knots."""

    def test_knot_not_found(self) -> None:
        """Requesting removal of a non-existent knot raises ValueError."""
        b = _make_curve_p2()
        with pytest.raises(ValueError, match="not found"):
            b.remove_knots(0.123)

    def test_boundary_knot(self) -> None:
        """Attempting to remove a boundary knot raises ValueError."""
        b = _make_curve_p2()
        with pytest.raises(ValueError, match="boundary knot"):
            b.remove_knots(0.0)

    def test_boundary_knot_end(self) -> None:
        """Attempting to remove the end boundary knot raises ValueError."""
        b = _make_curve_p2()
        with pytest.raises(ValueError, match="boundary knot"):
            b.remove_knots(1.0)

    def test_empty_knot_values(self) -> None:
        """Empty knot_values array raises ValueError."""
        b = _make_curve_p2()
        with pytest.raises(ValueError, match="non-empty"):
            b.remove_knots(np.array([]))

    def test_dim_mismatch(self) -> None:
        """Wrong number of direction arrays raises ValueError."""
        srf = _make_surface()
        with pytest.raises(ValueError, match="must match dim"):
            srf.remove_knots([np.array([0.5])])  # only 1 direction, need 2

    def test_periodic_not_supported(self) -> None:
        """Periodic B-splines raise ValueError."""
        knots = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        space = BsplineSpace([BsplineSpace1D(knots, 2, periodic=True)])
        cp = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        b = Bspline(space, cp)
        with pytest.raises(ValueError, match="periodic"):
            b.remove_knots(0.5)

    def test_num_zero_raises(self) -> None:
        """num=0 raises ValueError."""
        b = _make_curve_p2()
        with pytest.raises(ValueError, match="positive"):
            b.remove_knots(0.5, num=0)


# ===========================================================================
# Tolerance rejection
# ===========================================================================


class TestToleranceRejection:
    """Test that tight tolerances prevent removals that would distort geometry."""

    def test_tight_tolerance_prevents_removal(self) -> None:
        """A knot that exists in the original cannot be removed with tol=0."""
        b = _make_curve_p2()
        # The knot 0.5 is part of the original spline definition.
        # Removing it changes the geometry, so with tol=0 it should fail to remove.
        result = b.remove_knots(0.5, tol=0.0)
        # With zero tolerance, geometry must be preserved exactly.
        # The knot should still be present (removal was rejected).
        knots = result.space.spaces[0].knots
        count = np.sum(np.isclose(knots, 0.5, atol=1e-14))
        assert count == 1


class TestRemoveKnotKernelPrecondition:
    """The Layer-2 cap on ``num`` is what keeps the kernel's scratch buffer in bounds.

    ``_remove_knot_1d_core`` sizes ``temp`` as ``(2 * degree + 1, rank)`` once, and
    each removal pass writes two rows further out, so it stays in bounds only while
    ``2 * num <= degree + multiplicity``. Nothing inside the kernel enforces that --
    Layer 3 validates nothing by contract -- and Numba compiles without bounds
    checking, so exceeding it corrupts memory silently rather than raising.

    ``_remove_knot_bspline_1d_impl`` establishes the precondition by clamping ``num``
    to ``min(multiplicity, degree)``, which implies it. These tests pin that clamp:
    if it were dropped, a caller-supplied ``num`` would reach the kernel unbounded.
    """

    @staticmethod
    def _curve_with_multiplicity(degree: int, s: int, n_int: int = 6) -> Bspline:
        """Open degree-``p`` curve on ``[0, n_int]`` carrying knot 3.0 at multiplicity ``s``."""
        interior: list[float] = []
        for x in range(1, n_int):
            interior += [float(x)] * (s if x == 3 else 1)
        knots = np.concatenate(
            [
                np.zeros(degree + 1),
                np.array(interior, dtype=float),
                np.full(degree + 1, float(n_int)),
            ]
        )
        space = BsplineSpace([BsplineSpace1D(knots, degree)])
        n_basis = space.spaces[0].num_basis
        ctrl = np.stack(
            [np.linspace(0.0, 1.0, n_basis), np.linspace(0.0, 1.0, n_basis) ** 2], axis=1
        )
        return Bspline(space, ctrl)

    @pytest.mark.parametrize(
        ("degree", "s"),
        [(2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (4, 3)],
    )
    @pytest.mark.parametrize("requested_num", [1, 5, 100])
    def test_public_removal_never_exceeds_the_kernel_bound(
        self, degree: int, s: int, requested_num: int
    ) -> None:
        """However large a ``num`` the caller asks for, the kernel stays in bounds.

        The removals actually performed must satisfy ``2 * removals <= degree + s``.
        Verified under ``NUMBA_BOUNDSCHECK=1`` that this is exactly where the kernel
        starts writing out of bounds when called directly: e.g. degree 2 with
        ``s == 1`` raises at ``num == 2``, and degree 4 with ``s == 2`` at ``num == 4``.
        """
        curve = self._curve_with_multiplicity(degree, s)
        before = int(np.sum(np.abs(np.asarray(curve.space.spaces[0].knots) - 3.0) <= 1e-14))

        result = curve.remove_knots(3.0, num=requested_num, tol=1e30)

        after = int(np.sum(np.abs(np.asarray(result.space.spaces[0].knots) - 3.0) <= 1e-14))
        removals = before - after
        assert 0 <= removals <= min(s, degree)
        assert 2 * removals <= degree + s

    def test_removal_still_reaches_the_full_multiplicity(self) -> None:
        """The cap must not be so tight that a legitimate full removal is blocked.

        Guards against "fixing" the precondition by clamping ``num`` to something
        smaller: with ``tol`` wide open, all ``s`` copies must still come out.
        """
        curve = self._curve_with_multiplicity(degree=3, s=3)

        result = curve.remove_knots(3.0, tol=1e30)

        knots = np.asarray(result.space.spaces[0].knots)
        assert int(np.sum(np.abs(knots - 3.0) <= 1e-14)) == 0


# ===========================================================================
# The default tolerance
# ===========================================================================


def _random_curve(degree: int, n_el: int, rank: int, scale: float, seed: int) -> Bspline:
    """Build an open curve of the given degree whose control net has size ``scale``."""
    knots = np.concatenate(
        [np.full(degree, 0.0), np.linspace(0.0, 1.0, n_el + 1), np.full(degree, 1.0)]
    )
    space = BsplineSpace([BsplineSpace1D(knots, degree)])
    rng = np.random.default_rng(seed)
    cp = rng.uniform(-1.0, 1.0, size=(space.num_total_basis, rank)) * scale
    return Bspline(space, np.ascontiguousarray(cp))


def _nurbs_curve(  # noqa: PLR0913
    degree: int, n_el: int, rank: int, scale: float, weight_spread: float, seed: int
) -> Bspline:
    """Build a rational curve whose weights and coordinates vary **independently**.

    That independence is the point.  A circle cannot expose a mistake in the
    homogeneous-to-projected conversion, because its weights are structurally tied to
    its coordinates and a rescaling of one is a rescaling of the other.
    """
    knots = np.concatenate(
        [np.full(degree, 0.0), np.linspace(0.0, 1.0, n_el + 1), np.full(degree, 1.0)]
    )
    space = BsplineSpace([BsplineSpace1D(knots, degree)])
    rng = np.random.default_rng(seed)
    n = space.num_total_basis
    pts = rng.uniform(-1.0, 1.0, size=(n, rank)) * scale
    weights = np.exp(rng.uniform(-weight_spread, weight_spread, size=n))
    cp = np.concatenate([pts * weights[:, None], weights[:, None]], axis=1)
    return Bspline(space, np.ascontiguousarray(cp), is_rational=True)


class TestDefaultToleranceIsARoundOffFloor:
    """With no ``tol``, removal is lossless and its verdict is scale covariant.

    The old default was an absolute ``1e-10``, which is a geometric budget wearing no
    units: the same exactly-removable knot came out at geometry scale 1e-6, 1 and 1e3
    and was silently refused at 1e6 and 1e9.  The default is now
    ``8 * (degree + 1) * eps * scale`` with ``scale = max(bbox diagonal, max ||P_i||)``,
    which admits exactly what the reconstruction arithmetic cannot distinguish from an
    exact removal.
    """

    @pytest.mark.parametrize("scale", [1.0e-6, 1.0, 1.0e3, 1.0e6, 1.0e9])
    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5, 6, 7])
    def test_an_exactly_removable_knot_comes_out_at_every_geometry_scale(
        self, degree: int, scale: float
    ) -> None:
        """A just-inserted knot is redundant by construction, so it must always go.

        This is the case the absolute default lost above scale 1e6, where one ulp of a
        coordinate already exceeds ``1e-10``.
        """
        from pantr.bspline._bspline_knot_removal import (  # noqa: PLC0415
            _roundoff_deviation_floor,
        )

        curve = _random_curve(degree, 5, 3, scale, seed=degree)
        inserted = curve.insert_knots(np.array([0.37]))
        before = inserted.space.spaces[0].knots.size

        result = inserted.remove_knots(0.37)

        assert result.space.spaces[0].knots.size == before - 1
        # And the removal delivered what the budget promised.  The control-point
        # distance A5.8 measures is an upper bound on the curve deviation, so the two
        # curves must agree to within the very floor that admitted the removal --
        # a bound derived from the geometry rather than a round number.
        floor = _roundoff_deviation_floor(np.asarray(inserted.control_points), degree)
        t = np.linspace(0.0, 1.0, 401)
        deviation = float(
            np.linalg.norm(
                np.asarray(result.evaluate(t)) - np.asarray(inserted.evaluate(t)), axis=1
            ).max()
        )
        assert deviation <= floor

    @pytest.mark.parametrize("scale", [1.0e-6, 1.0, 1.0e3, 1.0e6, 1.0e9])
    def test_a_knot_the_curve_genuinely_needs_stays_at_every_geometry_scale(
        self, scale: float
    ) -> None:
        """The other half of the verdict: the default must not become a licence.

        Scale covariance is only worth having if both answers travel, so this pins the
        rejection at the same five scales as the acceptance above.
        """
        curve = _random_curve(3, 5, 3, scale, seed=99)
        before = curve.space.spaces[0].knots.size

        result = curve.remove_knots(0.4)

        assert result.space.spaces[0].knots.size == before

    def test_the_floor_is_proportional_to_the_control_net(self) -> None:
        """Rescaling the geometry rescales the floor by the same factor."""
        from pantr.bspline._bspline_knot_removal import (  # noqa: PLC0415
            _roundoff_deviation_floor,
        )

        cp = _random_curve(3, 5, 3, 1.0, seed=4).control_points
        unit = _roundoff_deviation_floor(np.asarray(cp), 3)
        scaled = _roundoff_deviation_floor(np.asarray(cp) * 1.0e6, 3)
        assert scaled == pytest.approx(1.0e6 * unit, rel=1.0e-12)

    def test_the_floor_grows_with_degree_and_is_a_multiple_of_eps(self) -> None:
        """Pin the derivation, not the digits: ``8 * (degree + 1) * eps * scale``."""
        from pantr.bspline._bspline_knot_removal import (  # noqa: PLC0415
            _control_point_scale,
            _roundoff_deviation_floor,
        )

        cp = np.asarray(_random_curve(3, 5, 3, 1.0, seed=4).control_points)
        eps = float(np.finfo(np.float64).eps)
        for degree in (1, 3, 7):
            expected = 8.0 * (degree + 1) * eps * _control_point_scale(cp)
            assert _roundoff_deviation_floor(cp, degree) == pytest.approx(expected, rel=1.0e-14)


class TestRationalDeviationIsMeasuredInProjectedSpace:
    """A caller's budget is a distance between points, not between homogeneous columns.

    The kernel measures a Euclidean distance over every column of the array it is
    given, which for a rational spline is ``[w x, w y, w z, w]``.  Eq. (5.30) of Piegl
    & Tiller (2nd ed., 1997, p. 185) is the conversion, and without it a deviation
    carried by the weight column is graded as though it were a coordinate.
    """

    @staticmethod
    def _max_projected_deviation(a: Bspline, b: Bspline) -> float:
        """Largest distance between the two curves' projected points over the domain."""
        t = np.linspace(0.0, 1.0, 601)
        pa = np.asarray(a.evaluate(t), dtype=np.float64)
        pb = np.asarray(b.evaluate(t), dtype=np.float64)
        return float(np.linalg.norm(pa - pb, axis=1).max())

    @pytest.mark.parametrize("degree", [2, 3, 4])
    @pytest.mark.parametrize("scale", [1.0, 1.0e3, 1.0e6])
    def test_an_accepted_removal_honours_the_requested_budget(
        self, degree: int, scale: float
    ) -> None:
        """Sweeping the perturbation, every accepted removal must be within budget.

        The perturbation is applied to the **weight** alone, which is the case Eq.
        (5.30) exists for; the sweep over sixteen decades is what finds the largest one
        the kernel is willing to accept.  Without the conversion the worst ratio
        measured over this family is ``7.4e5``; with it, ``0.32``.
        """
        base = _nurbs_curve(degree, 6, 3, scale, 1.5, seed=degree * 7)
        inserted = base.insert_knots(np.array([0.41]))
        before = inserted.space.spaces[0].knots.size
        requested = 1.0e-6 * scale
        accepted_any = False

        for eps_w in np.logspace(-16, 0, 33):
            cp = np.array(inserted.control_points)
            cp[cp.shape[0] // 2, -1] *= 1.0 + eps_w
            perturbed = Bspline(inserted.space, cp, is_rational=True)

            result = perturbed.remove_knots(0.41, tol=requested)
            if result.space.spaces[0].knots.size == before:
                continue
            accepted_any = True
            deviation = self._max_projected_deviation(result, perturbed)
            assert deviation <= requested, (
                f"budget broken by {deviation / requested:.4g}x at eps_w={eps_w:.3g}"
            )

        assert accepted_any, "the sweep never produced an accepted removal, so it proves nothing"

    @pytest.mark.parametrize("scale", [1.0e-6, 1.0, 1.0e6])
    def test_a_redundant_knot_still_comes_out_of_a_rational_curve(self, scale: float) -> None:
        """The conversion must not make the default unreachable for a NURBS.

        The round-off floor is deliberately *not* pulled back: it is a statement about
        the arithmetic, and the arithmetic runs in homogeneous coordinates.  Pulling it
        back would divide it by ``1 + |P|_max`` and refuse exact removals on any model
        bigger than a unit.
        """
        base = _nurbs_curve(3, 5, 3, scale, 1.5, seed=21)
        inserted = base.insert_knots(np.array([0.37]))
        before = inserted.space.spaces[0].knots.size

        result = inserted.remove_knots(0.37)

        assert result.space.spaces[0].knots.size == before - 1

    def test_the_conversion_is_the_published_formula(self) -> None:
        """``TOL = d * w_min / (1 + |P|_max)``, checked against a direct computation."""
        from pantr.bspline._bspline_knot_removal import (  # noqa: PLC0415
            _homogeneous_deviation_tolerance,
        )

        curve = _nurbs_curve(3, 5, 3, 100.0, 1.5, seed=13)
        cp = np.asarray(curve.control_points, dtype=np.float64)
        w_min = float(cp[:, -1].min())
        p_max = float(np.linalg.norm(cp[:, :-1] / cp[:, -1:], axis=1).max())

        assert _homogeneous_deviation_tolerance(cp, 1.0e-3) == pytest.approx(
            1.0e-3 * w_min / (1.0 + p_max), rel=1.0e-14
        )
