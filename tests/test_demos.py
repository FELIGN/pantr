"""Execute the documentation demos and validate their results.

Each ``demos/NN_*.py`` script is a standalone Sphinx-Gallery example. Sphinx-Gallery
only runs them during the docs build, so a library change that breaks a demo would
otherwise slip past the test suite. This module runs every numbered demo end-to-end and
then checks its results, so a demo that raises -- or merely emits a warning, since the
suite treats warnings as errors -- or that silently computes the wrong answer fails CI.

Each demo's computed objects are harvested from the namespace ``runpy.run_path`` returns
and handed to a per-demo validator. The validators assert two kinds of property:

* **Invariants** that must hold regardless of the exact numbers -- partition of unity,
  an interpolant passing through its data, the L2 projection being the L2-optimal fit,
  quadrature matching a known analytic value, found roots evaluating to zero,
  geometry-preserving knot/degree operations, a derivative matching finite differences,
  exact transform composition. These survive cosmetic demo edits and pin the
  mathematics.
* **Golden snapshots** of a few headline values (the change-of-basis matrix, the located
  roots, the L2-convergence errors, the THB active-function count). These catch silent
  numerical drift; update the constant deliberately when a result is meant to change.

Interactive rendering is reduced to a no-op at the two entry points the demos use to
draw: :meth:`pyvista.Plotter.show` (reached by ``pantr.viz.plot``, ``Scene.show``, and
PyVista's ``DataSet.plot`` -- all of which ultimately call it) and
:func:`matplotlib.pyplot.show`. The scripts still build every PyVista and Matplotlib
object -- exercising the full library code path -- but never open a window or touch the
GPU, so no display or GL context is required.
"""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

pv = pytest.importorskip("pyvista")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from pantr.bspline import get_greville_abscissae  # noqa: E402

_DEMOS_DIR: Path = Path(__file__).resolve().parent.parent / "demos"
_DEMO_SCRIPTS: list[Path] = sorted(_DEMOS_DIR.glob("[0-9]*_*.py"))

Namespace = dict[str, Any]
"""The module namespace a demo leaves behind, as returned by ``runpy.run_path``."""


# ---------------------------------------------------------------------------
# Sampling helpers


def _axis(geom: Any, d: int, n: int) -> npt.NDArray[np.float64]:
    """Return ``n`` equispaced parameter samples spanning direction ``d``'s domain.

    Args:
        geom: A geometry exposing ``space.spaces[d].domain``.
        d: Parametric direction index.
        n: Number of samples.

    Returns:
        npt.NDArray[np.float64]: The sample parameters, shape ``(n,)``.
    """
    lo, hi = geom.space.spaces[d].domain
    return np.asarray(np.linspace(float(lo), float(hi), n), dtype=np.float64)


def _sample_geom(geom: Any, n: int = 12) -> npt.NDArray[np.float64]:
    """Evaluate a geometry on an ``n`` per-direction grid of in-domain parameters.

    Args:
        geom: A :class:`~pantr.bspline.Bspline`-like object exposing ``space``,
            ``dim``, and ``evaluate``.
        n: Number of samples per parametric direction.

    Returns:
        npt.NDArray[np.float64]: Physical points, shape ``(n**dim, rank)``.
    """
    axes = [_axis(geom, d, n) for d in range(geom.dim)]
    if geom.dim == 1:
        params = axes[0]
    else:
        params = np.stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")], axis=1)
    return np.asarray(geom.evaluate(params), dtype=np.float64)


def _radii_xy(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the radial distance of each point from the z-axis (xy-plane norm).

    Args:
        points: Physical points, shape ``(..., rank)`` with ``rank >= 2``.

    Returns:
        npt.NDArray[np.float64]: Per-point radial distance.
    """
    return np.asarray(np.linalg.norm(points[:, :2], axis=1), dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-demo validators


def _validate_02(ns: Namespace) -> None:
    """Visualization basics: NURBS arc on the unit circle, scalar field, VTK export."""
    arc = ns["arc"]
    assert arc.is_rational, "create_circle should build a rational (NURBS) curve"
    np.testing.assert_allclose(
        _radii_xy(_sample_geom(arc, 40)), 1.0, atol=1e-9, err_msg="arc must lie on the unit circle"
    )
    field = ns["field"]
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(
        np.asarray(field.evaluate(corners)).reshape(-1),
        0.0,
        atol=1e-9,
        err_msg="scalar field must vanish on the boundary corners",
    )
    center = float(np.asarray(field.evaluate(np.array([[0.5, 0.5]]))).reshape(-1)[0])
    np.testing.assert_allclose(
        center, 0.5, rtol=1e-6, err_msg="scalar field centre value regressed"
    )
    mesh = pv.read(ns["out_file"])
    assert mesh.n_cells > 0, "exported .vtu should contain cells"


def _validate_03(ns: Namespace) -> None:
    """Basis gallery: Bernstein/Lagrange/Legendre basis checks + change-of-basis snapshot."""
    bases = ns["bases"]
    bern = np.asarray(bases["Bernstein"])
    np.testing.assert_allclose(
        bern.sum(axis=1), 1.0, atol=1e-12, err_msg="Bernstein basis must be a partition of unity"
    )
    assert (bern >= -1e-14).all(), "Bernstein basis must be non-negative"
    for name in ("Lagrange (equispaced)", "Lagrange (Gauss-Lobatto)"):
        np.testing.assert_allclose(
            np.asarray(bases[name]).sum(axis=1), 1.0, atol=1e-10, err_msg=f"{name} must sum to one"
        )
    legendre = np.asarray(bases["Legendre"])
    np.testing.assert_allclose(
        legendre[:, 0], 1.0, atol=1e-12, err_msg="Legendre P0 must be the constant 1"
    )
    # The Legendre polynomials are L2-orthonormal on [0, 1]; a wrong recurrence or sign
    # would break the off-diagonals. atol covers the 200-point trapezoidal estimate.
    x = np.asarray(ns["x"])
    n_leg = legendre.shape[1]
    gram = np.array(
        [
            [np.trapezoid(legendre[:, i] * legendre[:, j], x) for j in range(n_leg)]
            for i in range(n_leg)
        ]
    )
    np.testing.assert_allclose(
        gram, np.eye(n_leg), atol=5e-3, err_msg="Legendre basis must be L2-orthonormal on [0, 1]"
    )
    expected = np.array(
        [
            [1.0, -1.0833333333333333, 0.7222222222222222, -0.25, 0.0],
            [0.0, 4.0, -3.5555555555555554, 1.3333333333333333, 0.0],
            [0.0, -3.0, 6.666666666666667, -3.0, 0.0],
            [0.0, 1.3333333333333333, -3.5555555555555554, 4.0, 0.0],
            [0.0, -0.25, 0.7222222222222222, -1.0833333333333333, 1.0],
        ]
    )
    np.testing.assert_allclose(
        np.asarray(ns["matrix"]), expected, rtol=1e-6, atol=1e-9, err_msg="change-of-basis drifted"
    )


def _validate_01(ns: Namespace) -> None:
    """Geometry tour: clamped endpoints, Greville snapshot, derivative vs FD, exact circle."""
    curve = ns["curve"]
    cp = np.asarray(ns["control_points"])
    lo, hi = curve.space.spaces[0].domain
    np.testing.assert_allclose(
        np.asarray(curve.evaluate(np.array([lo]))).reshape(-1),
        cp[0],
        atol=1e-12,
        err_msg="clamped curve must start at the first control point",
    )
    np.testing.assert_allclose(
        np.asarray(curve.evaluate(np.array([hi]))).reshape(-1),
        cp[-1],
        atol=1e-12,
        err_msg="clamped curve must end at the last control point",
    )
    np.testing.assert_allclose(
        np.asarray(ns["greville"], dtype=float), [0.0, 0.5, 1.5, 2.5, 3.0], atol=1e-12
    )
    # The first derivative is itself a B-spline; check it against a central difference.
    # Sample interval midpoints, which avoid the interior knots (where f'' jumps and a
    # central difference straddling the kink would be only first-order accurate).
    edges = _axis(curve, 0, 41)
    u = (edges[:-1] + edges[1:]) / 2.0
    h = 1e-6
    fd = (np.asarray(curve.evaluate(u + h)) - np.asarray(curve.evaluate(u - h))) / (2.0 * h)
    np.testing.assert_allclose(
        np.asarray(curve.derivative(0).evaluate(u)),
        fd,
        atol=1e-5,
        err_msg="curve.derivative must match a finite difference of evaluate",
    )
    circle = ns["circle"]
    assert circle.is_rational, "create_circle should build a rational (NURBS) curve"
    np.testing.assert_allclose(_radii_xy(_sample_geom(circle, 40)), 1.0, atol=1e-9)


def _validate_04(ns: Namespace) -> None:
    """Knot ops: insertion/elevation preserve the curve; +3 control points; 3 Bézier cells."""
    curve, refined, elevated = ns["curve"], ns["refined"], ns["elevated"]
    u = _axis(curve, 0, 200)
    base = np.asarray(curve.evaluate(u))
    np.testing.assert_allclose(
        np.asarray(refined.evaluate(u)),
        base,
        atol=1e-10,
        err_msg="knot insertion changed the curve",
    )
    np.testing.assert_allclose(
        np.asarray(elevated.evaluate(u)),
        base,
        atol=1e-10,
        err_msg="degree elevation changed the curve",
    )
    assert elevated.degree[0] == curve.degree[0] + 1, "elevate_degree(1) must raise the degree by 1"
    assert (
        np.asarray(refined.control_points).shape[0] == np.asarray(curve.control_points).shape[0] + 3
    ), "inserting 3 knots must add 3 control points"
    assert ns["beziers"].size == 3, "a quadratic over 3 spans yields 3 Bézier elements"


def _validate_05(ns: Namespace) -> None:
    """Approximation: interpolant hits data, L2 projection is L2-optimal, errors decrease."""
    g, approx = ns["g"], ns["approx"]
    greville = np.asarray(get_greville_abscissae(ns["space"].spaces[0]), dtype=float)
    interp = np.asarray(approx["interpolation"].evaluate(greville)).reshape(-1)
    np.testing.assert_allclose(
        interp,
        g(greville),
        atol=1e-9,
        err_msg="interpolant must match the data at the Greville points",
    )
    l2_error = ns["l2_error"]
    e_interp = l2_error(approx["interpolation"])
    e_l2 = l2_error(approx["L2 projection"])
    e_quasi = l2_error(approx["quasi-interpolation"])
    assert e_l2 <= e_interp + 1e-12, (
        "L2 projection must be at least as L2-accurate as interpolation"
    )
    assert e_l2 <= e_quasi + 1e-12, "L2 projection must be at least as L2-accurate as quasi-interp"
    assert max(e_interp, e_l2, e_quasi) < 0.1, "cubic approximations should be reasonably accurate"
    errors = np.asarray(ns["errors"])  # degree-4 L2 errors (p=4 is last in the loop over {2, 3, 4})
    assert len(errors) == 5, "expected five refinement steps (one per entry of n_elements)"
    assert np.all(np.diff(errors) < 0), "L2 error must decrease under refinement"
    np.testing.assert_allclose(
        errors,
        [
            1.334459087399e-01,
            1.519317692139e-02,
            5.084044045290e-04,
            6.050129650871e-06,
            1.372424227733e-07,
        ],
        rtol=1e-4,
        atol=1e-12,
        err_msg="L2-convergence errors drifted",
    )


def _validate_06(ns: Namespace) -> None:
    """CAD modeling: cylinder/tube radii exact; revolved and ruled surfaces within bounds."""
    cyl = ns["cylinder"]
    assert cyl.is_rational, "cylinder is a rational surface"
    pts = _sample_geom(cyl, 16)
    np.testing.assert_allclose(
        _radii_xy(pts), 0.5, atol=1e-9, err_msg="cylinder radius must be 0.5"
    )
    assert pts[:, 2].min() >= -1e-9 and pts[:, 2].max() <= 1.5 + 1e-9, "cylinder height in [0, 1.5]"
    np.testing.assert_allclose(
        _radii_xy(_sample_geom(ns["tube"], 16)), 0.5, atol=1e-9, err_msg="extruded tube radius 0.5"
    )
    sor = _sample_geom(ns["surface_of_revolution"], 16)
    sor_r = _radii_xy(sor)
    assert sor_r.min() >= 0.2 - 1e-6, "revolved profile radius starts at 0.2"
    assert sor_r.max() <= 0.6 + 1e-6, "revolved profile radius ends at 0.6"
    assert sor[:, 2].min() >= -1e-9 and sor[:, 2].max() <= 1.0 + 1e-9, "revolution height in [0, 1]"
    frustum_r = _radii_xy(_sample_geom(ns["frustum"], 16))
    assert frustum_r.min() >= 0.3 - 1e-6, "frustum radius must not drop below the top radius 0.3"
    assert frustum_r.max() <= 0.7 + 1e-6, "frustum radius must not exceed the bottom radius 0.7"


def _validate_07(ns: Namespace) -> None:
    """Bézier roots: roots are real zeros (snapshot); 3 curve-line intersections on y=y0."""
    poly, roots = ns["poly"], np.asarray(ns["roots"])
    assert roots.size == 2, "find_roots should locate 2 roots in [0, 1]"
    np.testing.assert_allclose(
        np.asarray(poly.evaluate(roots)).reshape(-1),
        0.0,
        atol=1e-9,
        err_msg="located roots must evaluate to zero",
    )
    assert np.all(np.diff(roots) > 0), "roots must be returned sorted"
    assert np.all((roots >= 0.0) & (roots <= 1.0)), "roots must lie in [0, 1]"
    np.testing.assert_allclose(roots, [0.085746323411, 0.728480100989], rtol=1e-6)
    hits = np.asarray(ns["hits"])
    assert hits.shape[0] == 3, "the line y=y0 should cross the curve three times"
    np.testing.assert_allclose(
        hits[:, 1], ns["y0"], atol=1e-9, err_msg="curve-line intersections must lie on y = y0"
    )


def _validate_08(ns: Namespace) -> None:
    """THB: level/active-function counts (snapshot + structural); peak sits in the corner."""
    space = ns["space"]
    assert space.num_levels == 3, "two refinements should yield 3 levels"
    assert space.num_total_basis == 196, "active THB function count regressed"
    assert space.num_total_basis > ns["root"].num_total_basis, (
        "refinement must add active functions"
    )
    samples = np.array([[0.25, 0.25], [0.75, 0.75], [0.9, 0.1], [0.5, 0.5]])
    vals = np.asarray(ns["field"].evaluate(samples)).reshape(-1)
    assert vals.argmax() == 0, "the peak must sit in the refined corner (0.25, 0.25)"
    np.testing.assert_allclose(
        vals[0], 0.8498850528981, rtol=1e-3, err_msg="quasi-interpolated peak value regressed"
    )
    assert np.all(np.abs(vals[1:]) < 1e-6), "field must be ~0 away from the peak"


def _validate_09(ns: Namespace) -> None:
    """Quadrature: integral matches the analytic value; weights sum to area; counts/mesh."""
    integral, exact = ns["integral"], ns["exact"]
    assert abs(integral - exact) < 1e-9, f"quadrature integral {integral} != exact {exact}"
    np.testing.assert_allclose(
        float(np.sum(ns["weights"])),
        1.0,
        atol=1e-12,
        err_msg="weights must sum to the unit-square area",
    )
    assert ns["grid"].num_cells == 256, "a 16x16 grid has 256 cells"
    assert len(ns["hit_cells"]) == 9, "the [0.2, 0.35]^2 window overlaps 9 cells"
    assert ns["ug"].n_cells == ns["grid"].num_cells, "grid mesh must have one cell per grid cell"


def _validate_10(ns: Namespace) -> None:
    """Transforms: composition equals sequential; rotation/shear/scaling act as specified."""
    assert ns["err"] < 1e-12, f"(t2 @ t1) must equal t2(t1(.)) on control points, err={ns['err']}"
    base_cp = np.asarray(ns["base"].control_points)
    rot_cp = np.asarray(ns["rotated"].control_points)
    np.testing.assert_allclose(
        rot_cp[..., 0],
        base_cp[..., 0],
        atol=1e-12,
        err_msg="rotation about x must fix the x-component",
    )
    np.testing.assert_allclose(
        np.hypot(rot_cp[..., 1], rot_cp[..., 2]),
        np.hypot(base_cp[..., 1], base_cp[..., 2]),
        atol=1e-12,
        err_msg="rotation about x must preserve the y-z radius",
    )
    shear_cp = np.asarray(ns["sheared"].control_points)
    np.testing.assert_allclose(
        shear_cp[..., 0],
        base_cp[..., 0] + 0.5 * base_cp[..., 2],
        atol=1e-12,
        err_msg="shear must map x -> x + 0.5 z",
    )
    scaled_z = np.asarray(ns["scaled"].control_points)[..., 2]
    base_z = base_cp[..., 2]
    np.testing.assert_allclose(
        scaled_z.max() - scaled_z.min(),
        1.6 * (base_z.max() - base_z.min()),
        rtol=1e-9,
        err_msg="scaling z by 1.6 must scale the z-extent by 1.6",
    )


_VALIDATORS: dict[str, Callable[[Namespace], None]] = {
    "01_bspline_geometry_tour": _validate_01,
    "02_visualization_basics": _validate_02,
    "03_basis_gallery": _validate_03,
    "04_knot_operations": _validate_04,
    "05_approximation": _validate_05,
    "06_cad_modeling": _validate_06,
    "07_bezier_and_roots": _validate_07,
    "08_thb_adaptive_refinement": _validate_08,
    "09_grids_and_quadrature": _validate_09,
    "10_transforms": _validate_10,
}


# ---------------------------------------------------------------------------
# Tests


def test_demo_scripts_discovered() -> None:
    """Guard against a glob/path mistake silently producing zero parametrized tests."""
    assert _DEMO_SCRIPTS, f"no demo scripts found under {_DEMOS_DIR}"


def test_demos_and_validators_match() -> None:
    """Force every demo to ship result checks, and flag validators with no demo."""
    stems = {p.stem for p in _DEMO_SCRIPTS}
    missing = sorted(stems - _VALIDATORS.keys())
    orphaned = sorted(_VALIDATORS.keys() - stems)
    assert not missing, f"demos without a result validator: {missing}"
    assert not orphaned, f"validators with no matching demo: {orphaned}"


@pytest.fixture
def _stub_rendering(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stub interactive rendering so the demos run headless.

    Sets three things via ``monkeypatch`` (so each is restored on teardown): forces
    ``pv.OFF_SCREEN`` so any ``Plotter`` built before ``show`` skips the display
    connection; stubs ``pv.Plotter.show`` (the VTK render/window step every viz call
    funnels through) to a no-op; and stubs ``matplotlib.pyplot.show`` to a no-op.
    Teardown also closes open Matplotlib figures so the cumulative-figure
    ``RuntimeWarning`` never trips the warnings-as-errors filter.

    Args:
        monkeypatch: Pytest fixture used to patch the rendering entry points.

    Yields:
        None: Control returns to the test with rendering stubbed.
    """
    monkeypatch.setattr(pv, "OFF_SCREEN", True)
    monkeypatch.setattr(pv.Plotter, "show", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    yield
    plt.close("all")


@pytest.mark.usefixtures("_stub_rendering")
@pytest.mark.parametrize("script", _DEMO_SCRIPTS, ids=lambda p: p.stem)
def test_demo_runs(script: Path) -> None:
    """Run a demo end-to-end, then validate its computed results.

    The demo must complete without raising or emitting a warning (the suite is
    warnings-as-errors), and the values it produced must satisfy the demo's registered
    invariant and snapshot checks.

    Args:
        script: Path to the ``demos/NN_*.py`` script to execute.
    """
    namespace = runpy.run_path(str(script), run_name="__main__")
    _VALIDATORS[script.stem](namespace)
