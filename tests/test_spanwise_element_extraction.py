"""Tests for :class:`SpanwiseElementExtraction`.

Covers:

- Target dispatch (``bezier``, ``lagrange``, ``cardinal``) and construction
  errors (unknown target, periodic directions).
- Shape/dtype and identity-mask properties.
- Per-cell :meth:`apply` / :meth:`apply_transpose` / :meth:`apply_MT_K_M` /
  :meth:`apply_M_K_MT` against a dense reference ``kron(M_0, M_1, …)``.
- :meth:`operator` / :meth:`tabulate` round-trip consistency.
- Identity-query correctness: cardinal spaces use the structural mask from
  :meth:`BsplineSpace1D.get_cardinal_intervals`, C⁰ Bézier spaces detect
  numerical identity on every element.
- Cell-index normalization (flat int, tuple, list; IndexError / ValueError
  on invalid input).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.basis import LagrangeVariant
from pantr.bspline import BsplineSpace, BsplineSpace1D, SpanwiseElementExtraction
from pantr.bspline.spanwise_element_extraction import (
    _numerical_identity_mask,
    operand_shape,
)

RNG = np.random.default_rng(20260421)


def _space_2d() -> BsplineSpace:
    """Build a small 2D open-knot space with a mix of cardinal / non-cardinal intervals."""
    sp1 = BsplineSpace1D([0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6], 2)
    return BsplineSpace([sp1, sp1])


def _space_3d() -> BsplineSpace:
    """Build a small 3D space suitable for all four apply variants."""
    sp1 = BsplineSpace1D([0, 0, 0, 1, 2, 3, 3, 3], 2)
    return BsplineSpace([sp1, sp1, sp1])


def _dense_operator(
    ext: SpanwiseElementExtraction, cell_idx: int
) -> npt.NDArray[np.float32 | np.float64]:
    """Build the per-cell Kronecker operator via explicit ``np.kron``."""
    ops, _flags = ext[cell_idx]
    result = ops[0]
    for M in ops[1:]:
        result = np.kron(result, M)
    return result


# ---------------------------------------------------------------- construction


def test_unknown_target_rejected() -> None:
    """Unknown target tags raise ValueError."""
    sp = _space_2d()
    with pytest.raises(ValueError, match="Unknown target"):
        SpanwiseElementExtraction(sp, "spline")  # type: ignore[arg-type]


def test_periodic_rejected() -> None:
    """Periodic directions raise NotImplementedError in v0."""
    sp1 = BsplineSpace1D([0, 1, 2, 3, 4, 5], 1, periodic=True)
    sp = BsplineSpace([sp1])
    with pytest.raises(NotImplementedError, match="periodic"):
        SpanwiseElementExtraction(sp, "bezier")


@pytest.mark.parametrize("target", ["bezier", "lagrange", "cardinal"])
def test_per_direction_shapes(target: str) -> None:
    """``ops_1d`` and identity masks match the space's per-direction sizes."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, target)  # type: ignore[arg-type]
    assert ext.dim == 2  # noqa: PLR2004
    assert ext.num_intervals == (6, 6)
    assert ext.num_total_intervals == 36  # noqa: PLR2004
    assert len(ext.ops_1d) == 2  # noqa: PLR2004
    for ops in ext.ops_1d:
        assert ops.shape == (6, 3, 3)
        assert ops.dtype == ext.dtype
    for mask in ext.is_identity_mask_1d:
        assert mask.shape == (6,)
        assert mask.dtype == np.bool_


# ---------------------------------------------------------------- identity detection


def test_cardinal_identity_structural_matches_space() -> None:
    """Cardinal identity mask equals ``BsplineSpace1D.get_cardinal_intervals``."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, "cardinal")
    for space_1d, mask in zip(sp.spaces, ext.is_identity_mask_1d, strict=True):
        np.testing.assert_array_equal(mask, space_1d.get_cardinal_intervals())


def test_bezier_c0_space_is_numerically_identity_everywhere() -> None:
    """A C⁰ Bézier space has identity Bézier extraction on every element."""
    sp1 = BsplineSpace1D([0, 0, 0, 1, 1, 1, 2, 2, 2], 2)
    sp = BsplineSpace([sp1, sp1])
    ext = SpanwiseElementExtraction(sp, "bezier")
    for mask in ext.is_identity_mask_1d:
        assert mask.all()
    assert ext.is_identity_at(0)
    assert ext.num_identity_elements == ext.num_total_intervals


def test_numerical_identity_mask_ignores_nonsquare() -> None:
    """Non-square operators are never detected as identity."""
    ops = np.empty((2, 3, 4), dtype=np.float64)
    ops[0] = 0.0
    ops[1] = 1.0
    mask = _numerical_identity_mask(ops, 1e-12)
    assert not mask.any()


# ---------------------------------------------------------------- apply correctness


@pytest.mark.parametrize("target", ["bezier", "lagrange", "cardinal"])
@pytest.mark.parametrize("cell_idx", [0, 5, 20, 35])
def test_apply_matches_dense_kron(target: str, cell_idx: int) -> None:
    """``apply`` matches ``kron(M_0, M_1) @ v`` for assorted cells."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, target)  # type: ignore[arg-type]
    n_in = int(np.prod(ext.input_shape_per_dir))
    v = RNG.standard_normal(n_in).astype(ext.dtype)
    result = ext.apply(v, cell_idx)
    expected = _dense_operator(ext, cell_idx) @ v
    np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("target", ["bezier", "lagrange", "cardinal"])
def test_apply_transpose_matches_dense(target: str) -> None:
    """``apply_transpose`` matches ``M.T @ v``."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, target)  # type: ignore[arg-type]
    n_out = int(np.prod(ext.output_shape_per_dir))
    v = RNG.standard_normal(n_out).astype(ext.dtype)
    result = ext.apply_transpose(v, 7)
    expected = _dense_operator(ext, 7).T @ v
    np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("target", ["bezier", "lagrange", "cardinal"])
def test_apply_MT_K_M_matches_dense(target: str) -> None:
    """``apply_MT_K_M`` matches ``M.T @ K @ M``."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, target)  # type: ignore[arg-type]
    n_out = int(np.prod(ext.output_shape_per_dir))
    K = RNG.standard_normal((n_out, n_out)).astype(ext.dtype)
    result = ext.apply_MT_K_M(K, 12)
    M = _dense_operator(ext, 12)
    expected = M.T @ K @ M
    np.testing.assert_allclose(result, expected, atol=1e-11)


@pytest.mark.parametrize("target", ["bezier", "lagrange", "cardinal"])
def test_apply_M_K_MT_matches_dense(target: str) -> None:
    """``apply_M_K_MT`` matches ``M @ K @ M.T``."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, target)  # type: ignore[arg-type]
    n_in = int(np.prod(ext.input_shape_per_dir))
    K = RNG.standard_normal((n_in, n_in)).astype(ext.dtype)
    result = ext.apply_M_K_MT(K, (2, 3))
    M = _dense_operator(ext, int(np.ravel_multi_index((2, 3), ext.num_intervals)))
    expected = M @ K @ M.T
    np.testing.assert_allclose(result, expected, atol=1e-11)


def test_apply_3d_matches_dense() -> None:
    """Sanity check for d=3: apply and bilateral variants match dense references."""
    sp = _space_3d()
    ext = SpanwiseElementExtraction(sp, "bezier")
    cell = (1, 2, 0)
    flat = int(np.ravel_multi_index(cell, ext.num_intervals))

    n_in = int(np.prod(ext.input_shape_per_dir))
    n_out = int(np.prod(ext.output_shape_per_dir))
    v = RNG.standard_normal(n_in).astype(ext.dtype)
    M = _dense_operator(ext, flat)
    np.testing.assert_allclose(ext.apply(v, cell), M @ v, atol=1e-11)

    w = RNG.standard_normal(n_out).astype(ext.dtype)
    np.testing.assert_allclose(ext.apply_transpose(w, flat), M.T @ w, atol=1e-11)

    K = RNG.standard_normal((n_out, n_out)).astype(ext.dtype)
    np.testing.assert_allclose(ext.apply_MT_K_M(K, flat), M.T @ K @ M, atol=1e-10)
    K2 = RNG.standard_normal((n_in, n_in)).astype(ext.dtype)
    np.testing.assert_allclose(ext.apply_M_K_MT(K2, flat), M @ K2 @ M.T, atol=1e-10)


def test_apply_accepts_user_out_and_scratch() -> None:
    """User-provided ``out`` and ``scratch`` are written into and returned."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, "bezier")
    n_in = int(np.prod(ext.input_shape_per_dir))
    n_out = int(np.prod(ext.output_shape_per_dir))
    v = RNG.standard_normal(n_in).astype(ext.dtype)
    out = np.zeros(n_out, dtype=ext.dtype)
    scratch = np.zeros(4 * n_out * n_in, dtype=ext.dtype)
    result = ext.apply(v, 0, out=out, scratch=scratch)
    assert result is out
    np.testing.assert_allclose(result, _dense_operator(ext, 0) @ v, atol=1e-12)


# ---------------------------------------------------------------- operator / tabulate


@pytest.mark.parametrize("target", ["bezier", "lagrange", "cardinal"])
def test_operator_matches_kron(target: str) -> None:
    """``operator(cell)`` equals the explicit Kronecker product at that cell."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, target)  # type: ignore[arg-type]
    for cell in (0, 3, 17, 35):
        np.testing.assert_allclose(ext.operator(cell), _dense_operator(ext, cell), atol=1e-12)


def test_tabulate_has_correct_shape_and_matches_per_cell_operator() -> None:
    """``tabulate()`` stacks per-cell operators in row-major order."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, "lagrange")
    stacked = ext.tabulate()
    n_out = int(np.prod(ext.output_shape_per_dir))
    n_in = int(np.prod(ext.input_shape_per_dir))
    assert stacked.shape == (ext.num_total_intervals, n_out, n_in)
    for flat in range(ext.num_total_intervals):
        np.testing.assert_allclose(stacked[flat], ext.operator(flat), atol=1e-12)


def test_lagrange_variant_is_respected() -> None:
    """Different Lagrange variants produce different operators.

    Uses degree ≥ 3 because for degree ≤ 2 the equispaced and GLL point sets
    coincide, so the resulting extraction operators are identical.
    """
    sp1 = BsplineSpace1D([0, 0, 0, 0, 1, 2, 3, 3, 3, 3], 3)
    sp = BsplineSpace([sp1, sp1])
    eq = SpanwiseElementExtraction(sp, "lagrange", lagrange_variant=LagrangeVariant.EQUISPACES)
    gll = SpanwiseElementExtraction(
        sp, "lagrange", lagrange_variant=LagrangeVariant.GAUSS_LOBATTO_LEGENDRE
    )
    assert not np.allclose(eq.ops_1d[0], gll.ops_1d[0])


# ---------------------------------------------------------------- iteration / indexing


def test_len_and_iter_yield_per_cell_entries() -> None:
    """``len`` and ``__iter__`` match ``num_total_intervals`` and ``__getitem__``."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, "bezier")
    assert len(ext) == ext.num_total_intervals
    items = list(ext)
    assert len(items) == ext.num_total_intervals
    for flat, (ops, flags) in enumerate(items):
        ref_ops, ref_flags = ext[flat]
        assert flags == ref_flags
        for M, M_ref in zip(ops, ref_ops, strict=True):
            np.testing.assert_array_equal(M, M_ref)


def test_cell_index_accepts_flat_and_multi() -> None:
    """Flat ``int``, tuple, list, and ndarray indices are all accepted."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, "bezier")
    multi = (4, 1)
    flat = int(np.ravel_multi_index(multi, ext.num_intervals))
    ops_flat, _ = ext[flat]
    ops_tuple, _ = ext[multi]
    ops_list, _ = ext[[4, 1]]
    ops_arr, _ = ext[np.array(multi)]
    for a, b, c, d in zip(ops_flat, ops_tuple, ops_list, ops_arr, strict=True):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(a, d)


def test_cell_index_out_of_range() -> None:
    """Out-of-range flat and per-direction indices raise IndexError."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, "bezier")
    with pytest.raises(IndexError):
        ext.is_identity_at(ext.num_total_intervals)
    with pytest.raises(IndexError):
        ext.is_identity_at((6, 0))


def test_cell_index_wrong_length() -> None:
    """Per-direction index with wrong length raises ValueError."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, "bezier")
    with pytest.raises(ValueError, match="length"):
        ext.is_identity_at((0, 1, 2))


def test_cell_index_wrong_type() -> None:
    """Non-int, non-sequence cell indices raise TypeError."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, "bezier")
    with pytest.raises(TypeError):
        ext.is_identity_at("0")  # type: ignore[arg-type]


# ---------------------------------------------------------------- operand_shape helper


def test_operand_shape_returns_vector_and_matrix_shapes() -> None:
    """``operand_shape`` returns the expected vector/matrix shapes per ``op_kind``."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, "bezier")
    n_in = int(np.prod(ext.input_shape_per_dir))
    n_out = int(np.prod(ext.output_shape_per_dir))
    assert operand_shape(ext, "apply") == ((n_in,), (n_out,))
    assert operand_shape(ext, "apply_T") == ((n_out,), (n_in,))
    assert operand_shape(ext, "MT_K_M") == ((n_out, n_out), (n_in, n_in))
    assert operand_shape(ext, "M_K_MT") == ((n_in, n_in), (n_out, n_out))
