"""Tests for B-spline and THB quasi-interpolation (Lee-Lyche-Mørken / Speleers-Manni)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import (
    Bspline,
    BsplineSpace,
    BsplineSpace1D,
    THBSpline,
    THBSplineSpace,
    create_uniform_space,
    quasi_interpolate_bspline,
    quasi_interpolate_thb_spline,
)
from pantr.grid import HierarchicalGrid, hierarchical_grid, uniform_grid

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_KNOTS_DEG2_4 = np.array([0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0])


def _root_1d() -> BsplineSpace:
    return BsplineSpace([BsplineSpace1D(_KNOTS_DEG2_4, 2)])


def _root_2d() -> BsplineSpace:
    sp = BsplineSpace1D(_KNOTS_DEG2_4, 2)
    return BsplineSpace([sp, sp])


def _grid_1d() -> HierarchicalGrid:
    return hierarchical_grid(uniform_grid([[0.0, 1.0]], 4), 2)


def _grid_2d() -> HierarchicalGrid:
    return hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 4), 2)


def _sample(dim: int, n: int = 9) -> npt.NDArray[np.float64]:
    """Interior sample points in (0,1)^dim."""
    u = np.linspace(0.05, 0.95, n)
    mesh = np.meshgrid(*[u] * dim, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1)


def _thb_reproduction_error(thb: THBSplineSpace, seed: int = 0) -> float:
    """QI of a random THB spline recovers its coefficients."""
    rng = np.random.default_rng(seed)
    coeffs = rng.standard_normal(thb.num_total_basis)
    f = THBSpline(thb, coeffs)
    recovered = quasi_interpolate_thb_spline(f.evaluate, thb)
    return float(np.abs(recovered.coeffs - coeffs).max())


# ──────────────────────────────────────────────────────────────────────────────
# Tensor-product quasi-interpolation
# ──────────────────────────────────────────────────────────────────────────────


class TestTensorProductQI:
    """quasi_interpolate_bspline: polynomial reproduction and projector property."""

    @pytest.mark.parametrize("power", [0, 1, 2])
    def test_reproduces_monomials_1d(self, power: int) -> None:
        sp = _root_1d()
        qi = quasi_interpolate_bspline(lambda p: p[:, 0] ** power, sp)
        xs = np.linspace(0.0, 1.0, 23)
        got = np.asarray(qi.evaluate(xs)).ravel()
        np.testing.assert_allclose(got, xs**power, atol=1e-12)

    def test_reproduces_polynomial_2d(self) -> None:
        sp = _root_2d()
        qi = quasi_interpolate_bspline(lambda p: p[:, 0] ** 2 + p[:, 0] * p[:, 1], sp)
        pts = _sample(2)
        got = np.asarray(qi.evaluate(pts)).ravel()
        np.testing.assert_allclose(got, pts[:, 0] ** 2 + pts[:, 0] * pts[:, 1], atol=1e-12)

    def test_projector_reproduces_arbitrary_spline(self) -> None:
        # Q B_j = B_j: a spline drawn from the space is reproduced exactly.
        sp = _root_2d()
        rng = np.random.default_rng(1)
        cp = rng.standard_normal(sp.num_total_basis)
        spline = Bspline(sp, cp.reshape(*sp.num_basis, 1).astype(sp.dtype))
        qi = quasi_interpolate_bspline(lambda p: np.asarray(spline.evaluate(p)).ravel(), sp)
        np.testing.assert_allclose(qi.control_points.ravel(), cp, atol=1e-11)

    def test_vector_valued(self) -> None:
        sp = _root_1d()
        qi = quasi_interpolate_bspline(lambda p: np.stack([p[:, 0], p[:, 0] ** 2], axis=-1), sp)
        assert qi.rank == 2
        xs = np.linspace(0.0, 1.0, 11)
        got = np.asarray(qi.evaluate(xs))
        np.testing.assert_allclose(got[:, 0], xs, atol=1e-12)
        np.testing.assert_allclose(got[:, 1], xs**2, atol=1e-12)

    def test_bad_space_raises(self) -> None:
        with pytest.raises(TypeError, match="BsplineSpace"):
            quasi_interpolate_bspline(lambda p: p[:, 0], object())  # type: ignore[arg-type]

    def test_bad_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            quasi_interpolate_bspline(lambda p: p[:, 0], _root_1d(), kind="nope")  # type: ignore[arg-type]

    def test_func_zero_d_raises(self) -> None:
        with pytest.raises(ValueError, match="0-D"):
            quasi_interpolate_bspline(lambda p: np.float64(1.0), _root_1d())

    def test_func_3d_raises(self) -> None:
        with pytest.raises(ValueError, match="3-D"):
            quasi_interpolate_bspline(lambda p: np.zeros((p.shape[0], 2, 2)), _root_1d())

    def test_func_wrong_count_raises(self) -> None:
        with pytest.raises(ValueError, match="values"):
            quasi_interpolate_bspline(lambda p: np.zeros(p.shape[0] + 1), _root_1d())


# ──────────────────────────────────────────────────────────────────────────────
# Hierarchical quasi-interpolation — exact reproduction (THB)
# ──────────────────────────────────────────────────────────────────────────────


class TestThbReproduction:
    """Speleers-Manni: the THB QI reproduces any THB spline to machine precision."""

    def test_unrefined(self) -> None:
        assert _thb_reproduction_error(THBSplineSpace(_root_1d(), _grid_1d())) < 1e-10

    def test_1d_two_levels(self) -> None:
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        assert _thb_reproduction_error(THBSplineSpace(_root_1d(), grid)) < 1e-10

    def test_1d_three_levels(self) -> None:
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        grid.refine(1, [0], [2])
        assert _thb_reproduction_error(THBSplineSpace(_root_1d(), grid)) < 1e-10

    def test_2d_corner(self) -> None:
        grid = _grid_2d()
        grid.refine(0, [0, 0], [2, 2])
        assert _thb_reproduction_error(THBSplineSpace(_root_2d(), grid)) < 1e-10

    def test_2d_three_levels(self) -> None:
        grid = _grid_2d()
        grid.refine(0, [1, 1], [3, 3])
        grid.refine(1, [2, 2], [6, 6])
        assert _thb_reproduction_error(THBSplineSpace(_root_2d(), grid)) < 1e-10

    def test_narrow_band_1d(self) -> None:
        # Single-cell-wide multi-level band: exercises the leaf-cell / truncation path.
        grid = _grid_1d()
        grid.refine(0, [1], [2])
        grid.refine(1, [2], [3])
        thb = THBSplineSpace(_root_1d(), grid)
        assert thb.num_levels == 3
        assert _thb_reproduction_error(thb) < 1e-10

    def test_narrow_band_2d(self) -> None:
        grid = _grid_2d()
        grid.refine(0, [1, 1], [2, 2])
        grid.refine(1, [2, 2], [3, 3])
        thb = THBSplineSpace(_root_2d(), grid)
        assert thb.num_levels == 3
        assert _thb_reproduction_error(thb) < 1e-10

    def test_reproduces_polynomials_1d(self) -> None:
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        thb = THBSplineSpace(_root_1d(), grid)
        qi = quasi_interpolate_thb_spline(lambda p: 1.0 - 2.0 * p[:, 0] + 3.0 * p[:, 0] ** 2, thb)
        xs = _sample(1)
        exact = 1.0 - 2.0 * xs[:, 0] + 3.0 * xs[:, 0] ** 2
        np.testing.assert_allclose(np.asarray(qi.evaluate(xs)).ravel(), exact, atol=1e-10)

    def test_reproduces_polynomials_2d(self) -> None:
        grid = _grid_2d()
        grid.refine(0, [0, 0], [2, 2])
        thb = THBSplineSpace(_root_2d(), grid)
        qi = quasi_interpolate_thb_spline(lambda p: p[:, 0] ** 2 + p[:, 0] * p[:, 1] + 1.0, thb)
        pts = _sample(2)
        exact = pts[:, 0] ** 2 + pts[:, 0] * pts[:, 1] + 1.0
        np.testing.assert_allclose(np.asarray(qi.evaluate(pts)).ravel(), exact, atol=1e-10)

    def test_vector_valued(self) -> None:
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        thb = THBSplineSpace(_root_1d(), grid)
        rng = np.random.default_rng(3)
        coeffs = rng.standard_normal((thb.num_total_basis, 2))
        f = THBSpline(thb, coeffs)
        recovered = quasi_interpolate_thb_spline(f.evaluate, thb)
        assert recovered.rank == 2
        np.testing.assert_allclose(recovered.coeffs, coeffs, atol=1e-10)

    def test_multiple_candidate_cells_selects_nearest(self) -> None:
        # Refine two non-adjacent cells so a coarse dof whose support spans both
        # refined regions has more than one leaf-cell candidate; exercises the
        # nearest-Greville selection branch.
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        grid.refine(0, [3], [4])
        thb = THBSplineSpace(_root_1d(), grid)
        assert _thb_reproduction_error(thb) < 1e-10


# ──────────────────────────────────────────────────────────────────────────────
# Convergence smoke + TP consistency
# ──────────────────────────────────────────────────────────────────────────────


class TestConvergence:
    """Optimal approximation order on a smooth function; TP/THB consistency."""

    def test_order_p_plus_one_1d(self) -> None:
        def f(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.sin(2.0 * np.pi * p[:, 0])

        xs = np.linspace(0.0, 1.0, 401)
        errs = []
        for n in (8, 16, 32):
            qi = quasi_interpolate_bspline(f, create_uniform_space([2], [n]))
            got = np.asarray(qi.evaluate(xs)).ravel()
            errs.append(np.abs(got - np.sin(2.0 * np.pi * xs)).max())
        orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
        # degree-2 LLM projector: expect order ≈ 3.
        assert min(orders) > 2.9

    def test_unrefined_thb_matches_tp(self) -> None:
        def f(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.cos(3.0 * p[:, 0])

        root = _root_1d()
        tp = quasi_interpolate_bspline(f, root)
        thb = quasi_interpolate_thb_spline(f, THBSplineSpace(root, _grid_1d()))
        np.testing.assert_allclose(thb.coeffs, tp.control_points.ravel(), atol=1e-12)


# ──────────────────────────────────────────────────────────────────────────────
# Non-truncated (HB) basis
# ──────────────────────────────────────────────────────────────────────────────


class TestHb:
    """truncate=False: exact only on an unrefined grid; otherwise a valid local op."""

    def test_unrefined_hb_equals_thb(self) -> None:
        # With no finer levels HB == THB, so reproduction still holds.
        thb = THBSplineSpace(_root_1d(), _grid_1d(), truncate=False)
        assert _thb_reproduction_error(thb) < 1e-10

    def test_refined_hb_runs(self) -> None:
        grid = _grid_1d()
        grid.refine(0, [0], [2])
        thb = THBSplineSpace(_root_1d(), grid, truncate=False)
        qi = quasi_interpolate_thb_spline(lambda p: np.sin(p[:, 0]), thb)
        assert qi.coeffs.shape == (thb.num_total_basis,)
        assert np.all(np.isfinite(qi.coeffs))

    def test_bad_space_raises(self) -> None:
        with pytest.raises(TypeError, match="THBSplineSpace"):
            quasi_interpolate_thb_spline(lambda p: p[:, 0], _root_1d())  # type: ignore[arg-type]

    def test_bad_kind_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        with pytest.raises(ValueError, match="kind"):
            quasi_interpolate_thb_spline(lambda p: p[:, 0], thb, kind="nope")  # type: ignore[arg-type]

    def test_func_wrong_count_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        with pytest.raises(ValueError, match="values"):
            quasi_interpolate_thb_spline(lambda p: np.zeros(p.shape[0] + 1), thb)

    def test_stale_grid_raises_on_qi(self) -> None:
        grid = _grid_1d()
        thb = THBSplineSpace(_root_1d(), grid)
        grid.refine(0, [0], [2])
        with pytest.raises(RuntimeError, match="stale"):
            quasi_interpolate_thb_spline(lambda p: p[:, 0], thb)


# ──────────────────────────────────────────────────────────────────────────────
# THBSpline wrapper
# ──────────────────────────────────────────────────────────────────────────────


class TestThbSpline:
    """The evaluable THB spline function."""

    def test_evaluate_matches_manual_assembly(self) -> None:
        thb = THBSplineSpace(_root_2d(), _grid_2d())
        rng = np.random.default_rng(5)
        coeffs = rng.standard_normal(thb.num_total_basis)
        spline = THBSpline(thb, coeffs)
        pts = _sample(2, 5)
        got = np.asarray(spline.evaluate(pts)).ravel()
        manual = np.empty(pts.shape[0])
        for i, p in enumerate(pts):
            cid = thb.grid.locate(p)
            assert cid is not None
            vals, dofs = thb.tabulate_basis(cid, p.reshape(1, -1))
            manual[i] = vals[0] @ coeffs[dofs]
        np.testing.assert_allclose(got, manual, atol=1e-13)

    def test_properties_and_repr(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        spline = THBSpline(thb, np.zeros(thb.num_total_basis))
        assert spline.space is thb
        assert spline.dim == 1
        assert spline.rank == 1
        assert spline.dtype == np.float64
        assert spline.coeffs.shape == (thb.num_total_basis,)
        assert "THBSpline" in repr(spline)

    def test_vector_coeffs_shape(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        spline = THBSpline(thb, np.zeros((thb.num_total_basis, 3)))
        assert spline.rank == 3
        assert spline.coeffs.shape == (thb.num_total_basis, 3)
        xs = np.array([[0.1], [0.9]])
        assert np.asarray(spline.evaluate(xs)).shape == (2, 3)

    def test_non_thb_space_raises(self) -> None:
        with pytest.raises(TypeError, match="THBSplineSpace"):
            THBSpline(_root_1d(), np.zeros(9))  # type: ignore[arg-type]

    def test_bad_coeffs_length_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        with pytest.raises(ValueError, match="length"):
            THBSpline(thb, np.zeros(thb.num_total_basis + 1))

    def test_out_of_domain_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        spline = THBSpline(thb, np.zeros(thb.num_total_basis))
        with pytest.raises(ValueError, match="outside"):
            spline.evaluate(np.array([[2.0]]))

    def test_wrong_trailing_dim_raises(self) -> None:
        thb = THBSplineSpace(_root_2d(), _grid_2d())
        spline = THBSpline(thb, np.zeros(thb.num_total_basis))
        with pytest.raises(ValueError, match="trailing dimension"):
            spline.evaluate(np.array([[0.1]]))

    def test_stale_grid_raises_on_evaluate(self) -> None:
        grid = _grid_1d()
        thb = THBSplineSpace(_root_1d(), grid)
        spline = THBSpline(thb, np.zeros(thb.num_total_basis))
        grid.refine(0, [0], [2])
        with pytest.raises(RuntimeError, match="stale"):
            spline.evaluate(np.array([[0.5]]))

    def test_bad_coeffs_ndim_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        with pytest.raises(ValueError, match="1-D or 2-D"):
            THBSpline(thb, np.zeros((thb.num_total_basis, 2, 2)))

    def test_bad_coeffs_2d_leading_dim_raises(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        with pytest.raises(ValueError, match="leading dimension"):
            THBSpline(thb, np.zeros((thb.num_total_basis + 1, 2)))

    def test_coeffs_readonly(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        spline = THBSpline(thb, np.ones(thb.num_total_basis))
        with pytest.raises(ValueError, match="read-only"):
            spline.coeffs[:] = 0.0

    def test_vector_coeffs_readonly(self) -> None:
        thb = THBSplineSpace(_root_1d(), _grid_1d())
        spline = THBSpline(thb, np.ones((thb.num_total_basis, 2)))
        with pytest.raises(ValueError, match="read-only"):
            spline.coeffs[:] = 0.0

    def test_evaluate_vector_refined_2d(self) -> None:
        # Vector-valued evaluate on a refined 2D grid exercises the full
        # multi-level dofs → active_basis → tabulate_basis → matmul path.
        grid = _grid_2d()
        grid.refine(0, [1, 1], [3, 3])
        thb = THBSplineSpace(_root_2d(), grid)
        rng = np.random.default_rng(7)
        coeffs = rng.standard_normal((thb.num_total_basis, 2))
        spline = THBSpline(thb, coeffs)
        pts = _sample(2, 5)
        got = np.asarray(spline.evaluate(pts))
        assert got.shape == (pts.shape[0], 2)
        # Verify against manual assembly.
        for i, p in enumerate(pts):
            cid = thb.grid.locate(p)
            assert cid is not None
            vals, dofs = thb.tabulate_basis(cid, p.reshape(1, -1))
            expected = vals[0] @ coeffs[dofs]
            np.testing.assert_allclose(got[i], expected, atol=1e-13)
