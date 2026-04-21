"""Tests for :class:`SpanwiseElementExtraction`.

Covers:

- Target dispatch (``bezier``, ``lagrange``, ``cardinal``) and construction
  errors (unknown target, periodic directions, invalid ``identity_tol``).
- Shape/dtype and identity-mask properties.
- Per-cell :meth:`apply` / :meth:`apply_transpose` / :meth:`apply_MT_K_M` /
  :meth:`apply_M_K_MT` against a dense reference ``kron(M_0, M_1, …)``.
- :meth:`operator` / :meth:`tabulate` round-trip consistency, including
  user-provided ``out`` arrays.
- Identity-query correctness: :attr:`is_identity`, :attr:`num_identity_elements`,
  :meth:`per_direction_identity_flags`; cardinal spaces use the structural mask
  from :meth:`BsplineSpace1D.get_cardinal_intervals`; C⁰ Bézier spaces detect
  numerical identity on every element; ``identity_tol`` controls detection.
- Cell-index normalization (flat int, tuple, list, ndarray; negative indices
  rejected; IndexError / ValueError / TypeError on invalid input).
- 1D spaces and rejection of dim > 3 via :exc:`NotImplementedError`.
- Batch :meth:`apply_many` / :meth:`apply_transpose_many` /
  :meth:`apply_MT_K_M_many` / :meth:`apply_M_K_MT_many`: round-trip match
  against per-cell :meth:`apply` for all (d, op_kind, target, dtype) combos,
  flat and per-direction index forms, user-provided ``out``/``scratch``,
  empty batch, and validation errors.
- :func:`normalize_cell_indices`: flat-to-2D conversion, per-direction
  pass-through, and error cases.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr.basis import LagrangeVariant
from pantr.bspline import BsplineSpace, BsplineSpace1D, SpanwiseElementExtraction
from pantr.bspline.spanwise_element_extraction import (
    _numerical_identity_mask,
    normalize_cell_indices,
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
    """Build the per-cell Kronecker operator from ``ops_1d`` without using ``__getitem__``."""
    multi = np.unravel_index(cell_idx, ext.num_intervals)
    result: npt.NDArray[np.float32 | np.float64] = ext.ops_1d[0][int(multi[0])]
    for k in range(1, ext.dim):
        result = np.kron(result, ext.ops_1d[k][int(multi[k])])
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


# ---------------------------------------------------------------- construction validation


def test_identity_tol_negative_rejected() -> None:
    """Negative and NaN ``identity_tol`` raise ValueError."""
    sp = _space_2d()
    with pytest.raises(ValueError, match="non-negative"):
        SpanwiseElementExtraction(sp, "bezier", identity_tol=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        SpanwiseElementExtraction(sp, "bezier", identity_tol=float("nan"))


# ---------------------------------------------------------------- identity queries (extended)


def test_is_identity_property_true_for_c0_bezier() -> None:
    """``is_identity`` is ``True`` when every element has an identity operator."""
    sp1 = BsplineSpace1D([0, 0, 0, 1, 1, 1, 2, 2, 2], 2)
    sp = BsplineSpace([sp1, sp1])
    ext = SpanwiseElementExtraction(sp, "bezier")
    assert ext.is_identity


def test_is_identity_property_false_for_smooth_space() -> None:
    """``is_identity`` is ``False`` when some elements have non-identity operators."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, "bezier")
    assert not ext.is_identity


def test_num_identity_elements_formula_is_product_not_sum() -> None:
    """``num_identity_elements`` uses the tensor-product formula, not a sum."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, "cardinal")
    counts = [int(np.count_nonzero(m)) for m in ext.is_identity_mask_1d]
    assert ext.num_identity_elements == counts[0] * counts[1]
    # Meaningful only when there is more than one identity interval per direction
    assert all(c > 1 for c in counts), "space must have >1 identity per direction"


def test_per_direction_identity_flags_matches_mask_lookup() -> None:
    """``per_direction_identity_flags`` matches per-direction mask lookup for every cell."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, "cardinal")
    for flat in range(ext.num_total_intervals):
        multi = np.unravel_index(flat, ext.num_intervals)
        expected = tuple(
            bool(m[int(i)]) for m, i in zip(ext.is_identity_mask_1d, multi, strict=True)
        )
        assert ext.per_direction_identity_flags(flat) == expected
        assert ext.per_direction_identity_flags(tuple(int(i) for i in multi)) == expected


def test_identity_tol_affects_detection() -> None:
    """A custom ``identity_tol`` changes which elements are flagged as identity."""
    sp1 = BsplineSpace1D([0, 0, 0, 1, 2, 3, 3, 3], 2)
    sp = BsplineSpace([sp1])
    ext_default = SpanwiseElementExtraction(sp, "bezier")
    # With an enormous tolerance every element looks like identity
    ext_huge = SpanwiseElementExtraction(sp, "bezier", identity_tol=1e100)
    assert ext_huge.is_identity_mask_1d[0].all()
    assert ext_huge.identity_tol == 1e100  # noqa: PLR2004
    # The default space is not all-identity (smooth space has non-trivial Bezier ops)
    if not ext_default.is_identity_mask_1d[0].all():
        assert not ext_default.is_identity


# ---------------------------------------------------------------- operator / tabulate (extended)


def test_operator_accepts_user_out() -> None:
    """User-provided ``out`` to ``operator`` is written into and returned."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, "bezier")
    n_out = int(np.prod(ext.output_shape_per_dir))
    n_in = int(np.prod(ext.input_shape_per_dir))
    out = np.zeros((n_out, n_in), dtype=ext.dtype)
    result = ext.operator(7, out=out)
    assert result is out
    np.testing.assert_allclose(result, _dense_operator(ext, 7), atol=1e-12)


def test_tabulate_accepts_user_out() -> None:
    """User-provided ``out`` to ``tabulate`` is written into and returned."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, "bezier")
    n_out = int(np.prod(ext.output_shape_per_dir))
    n_in = int(np.prod(ext.input_shape_per_dir))
    out = np.zeros((ext.num_total_intervals, n_out, n_in), dtype=ext.dtype)
    result = ext.tabulate(out=out)
    assert result is out
    for flat in range(ext.num_total_intervals):
        np.testing.assert_allclose(result[flat], _dense_operator(ext, flat), atol=1e-12)


# ---------------------------------------------------------------- 1D spaces


def test_1d_space_apply_matches_dense() -> None:
    """A 1D space (dim=1) works correctly for all four apply variants."""
    sp1 = BsplineSpace1D([0, 0, 0, 1, 2, 3, 3, 3], 2)
    sp = BsplineSpace([sp1])
    ext = SpanwiseElementExtraction(sp, "bezier")
    assert ext.dim == 1
    n_in = ext.input_shape_per_dir[0]
    n_out = ext.output_shape_per_dir[0]
    for cell in range(ext.num_total_intervals):
        M = ext.ops_1d[0][cell]
        v = RNG.standard_normal(n_in).astype(ext.dtype)
        np.testing.assert_allclose(ext.apply(v, cell), M @ v, atol=1e-12)
        w = RNG.standard_normal(n_out).astype(ext.dtype)
        np.testing.assert_allclose(ext.apply_transpose(w, cell), M.T @ w, atol=1e-12)
        K = RNG.standard_normal((n_out, n_out)).astype(ext.dtype)
        np.testing.assert_allclose(ext.apply_MT_K_M(K, cell), M.T @ K @ M, atol=1e-11)
        K2 = RNG.standard_normal((n_in, n_in)).astype(ext.dtype)
        np.testing.assert_allclose(ext.apply_M_K_MT(K2, cell), M @ K2 @ M.T, atol=1e-11)


# ---------------------------------------------------------------- dim > 3


def test_dim_4_apply_raises_not_implemented() -> None:
    """Calling any apply variant on a 4D space raises NotImplementedError."""
    sp1 = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
    sp = BsplineSpace([sp1, sp1, sp1, sp1])
    ext = SpanwiseElementExtraction(sp, "bezier")
    n_in = int(np.prod(ext.input_shape_per_dir))
    v = np.zeros(n_in, dtype=ext.dtype)
    with pytest.raises(NotImplementedError):
        ext.apply(v, 0)


# ---------------------------------------------------------------- negative indices


def test_negative_flat_index_rejected() -> None:
    """Negative flat indices raise IndexError (Python-style wrap-around is not supported)."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, "bezier")
    with pytest.raises(IndexError):
        ext.is_identity_at(-1)


def test_negative_per_direction_index_rejected() -> None:
    """Negative per-direction indices raise IndexError."""
    sp = _space_2d()
    ext = SpanwiseElementExtraction(sp, "bezier")
    with pytest.raises(IndexError):
        ext.is_identity_at((-1, 0))


# ---------------------------------------------------------------- normalize_cell_indices


def test_normalize_flat_indices_converts_to_2d() -> None:
    """Flat 1-D indices are converted to per-direction 2-D array correctly."""
    num_intervals = (4, 5)
    flat = np.array([0, 1, 9, 19])
    result = normalize_cell_indices(flat, num_intervals)
    assert result.shape == (4, 2)
    for i, f in enumerate(flat):
        expected = np.unravel_index(f, num_intervals)
        np.testing.assert_array_equal(result[i], expected)


def test_normalize_2d_indices_passthrough() -> None:
    """2-D per-direction indices are returned as a numpy array unchanged."""
    num_intervals = (3, 4, 2)
    idx = np.array([[0, 1, 0], [2, 3, 1]])
    result = normalize_cell_indices(idx, num_intervals)
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result, idx)


def test_normalize_list_of_flat_indices() -> None:
    """A plain Python list of flat integers is accepted."""
    num_intervals = (6, 6)
    result = normalize_cell_indices([0, 7, 35], num_intervals)
    assert result.shape == (3, 2)


def test_normalize_empty_batch() -> None:
    """An empty batch produces a (0, d) output array."""
    result = normalize_cell_indices(np.array([], dtype=np.intp), (3, 4))
    assert result.shape == (0, 2)


def test_normalize_flat_out_of_range() -> None:
    """Flat indices outside [0, total) raise ValueError."""
    with pytest.raises(ValueError, match="Flat cell indices"):
        normalize_cell_indices(np.array([12]), (3, 4))  # 3*4=12, so 12 is OOB


def test_normalize_per_direction_out_of_range() -> None:
    """Per-direction indices outside per-direction bounds raise ValueError."""
    with pytest.raises(ValueError, match="cell_indices"):
        normalize_cell_indices(np.array([[3, 0]]), (3, 4))  # dir-0 max is 2


def test_normalize_wrong_ndim_raises() -> None:
    """3-D or 0-D input raises ValueError."""
    with pytest.raises(ValueError, match="ndim"):
        normalize_cell_indices(np.zeros((2, 3, 4), dtype=np.intp), (3, 4))


def test_normalize_wrong_d_raises() -> None:
    """2-D input with wrong number of columns raises ValueError."""
    with pytest.raises(ValueError, match="shape"):
        normalize_cell_indices(np.array([[0, 1, 0]]), (3, 4))  # 3 cols but d=2


# ---------------------------------------------------------------- batch apply helpers


def _space_1d() -> BsplineSpace:
    """Small 1D space for batch d=1 tests."""
    sp1 = BsplineSpace1D([0, 0, 0, 1, 2, 3, 3, 3], 2)
    return BsplineSpace([sp1])


_BATCH_SPACES: list[tuple[int, BsplineSpace]] = [
    (1, _space_1d()),
    (2, _space_2d()),
    (3, _space_3d()),
]

_OP_KINDS = ["apply", "apply_T", "MT_K_M", "M_K_MT"]
_TARGETS = ["bezier", "lagrange", "cardinal"]


def _batch_method(ext: SpanwiseElementExtraction, op_kind: str) -> Any:
    """Return the batch method corresponding to op_kind."""
    if op_kind == "apply":
        return ext.apply_many
    if op_kind == "apply_T":
        return ext.apply_transpose_many
    if op_kind == "MT_K_M":
        return ext.apply_MT_K_M_many
    return ext.apply_M_K_MT_many


def _per_cell_method(ext: SpanwiseElementExtraction, op_kind: str) -> Any:
    """Return the per-cell method corresponding to op_kind."""
    if op_kind == "apply":
        return ext.apply
    if op_kind == "apply_T":
        return ext.apply_transpose
    if op_kind == "MT_K_M":
        return ext.apply_MT_K_M
    return ext.apply_M_K_MT


# ---------------------------------------------------------------- batch correctness


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("op_kind", _OP_KINDS)
@pytest.mark.parametrize(("d", "sp"), _BATCH_SPACES)
def test_batch_apply_matches_per_cell(
    d: int,
    sp: BsplineSpace,
    op_kind: str,
    target: str,
) -> None:
    """Batch apply matches per-cell apply for every (d, op_kind, target) combo."""
    ext = SpanwiseElementExtraction(sp, target)  # type: ignore[arg-type]
    n_in = int(np.prod(ext.input_shape_per_dir))
    n_out = int(np.prod(ext.output_shape_per_dir))
    rng = np.random.default_rng(0)
    n_cells = min(ext.num_total_intervals, 8)
    flat_idx = rng.integers(0, ext.num_total_intervals, size=n_cells)
    dt = ext.dtype

    if op_kind == "apply":
        operand = rng.standard_normal((n_cells, n_in)).astype(dt)
    elif op_kind == "apply_T":
        operand = rng.standard_normal((n_cells, n_out)).astype(dt)
    elif op_kind == "MT_K_M":
        operand = rng.standard_normal((n_cells, n_out, n_out)).astype(dt)
    else:
        operand = rng.standard_normal((n_cells, n_in, n_in)).astype(dt)

    batch_fn = _batch_method(ext, op_kind)
    per_cell_fn = _per_cell_method(ext, op_kind)
    out_batch = batch_fn(operand, flat_idx)

    for c in range(n_cells):
        ref = per_cell_fn(operand[c], int(flat_idx[c]))
        np.testing.assert_allclose(out_batch[c], ref, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("op_kind", _OP_KINDS)
def test_batch_apply_float32_space(op_kind: str) -> None:
    """Batch apply works correctly on a float32 B-spline space."""
    knots = np.array([0, 0, 0, 1, 2, 3, 3, 3], dtype=np.float32)
    sp1 = BsplineSpace1D(knots, 2)
    sp = BsplineSpace([sp1, sp1])
    ext = SpanwiseElementExtraction(sp, "bezier")
    assert ext.dtype == np.float32
    n_in = int(np.prod(ext.input_shape_per_dir))
    n_out = int(np.prod(ext.output_shape_per_dir))
    rng = np.random.default_rng(99)
    n_cells = 4
    flat_idx = rng.integers(0, ext.num_total_intervals, size=n_cells)

    if op_kind == "apply":
        operand = rng.standard_normal((n_cells, n_in)).astype(np.float32)
    elif op_kind == "apply_T":
        operand = rng.standard_normal((n_cells, n_out)).astype(np.float32)
    elif op_kind == "MT_K_M":
        operand = rng.standard_normal((n_cells, n_out, n_out)).astype(np.float32)
    else:
        operand = rng.standard_normal((n_cells, n_in, n_in)).astype(np.float32)

    batch_fn = _batch_method(ext, op_kind)
    per_cell_fn = _per_cell_method(ext, op_kind)
    out_batch = batch_fn(operand, flat_idx)

    for c in range(n_cells):
        ref = per_cell_fn(operand[c], int(flat_idx[c]))
        np.testing.assert_allclose(out_batch[c], ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("op_kind", _OP_KINDS)
def test_batch_apply_per_direction_index_form(op_kind: str) -> None:
    """2-D per-direction cell_indices produce the same result as flat indices."""
    ext = SpanwiseElementExtraction(_space_2d(), "bezier")
    n_in = int(np.prod(ext.input_shape_per_dir))
    n_out = int(np.prod(ext.output_shape_per_dir))
    rng = np.random.default_rng(1)
    n_cells = 5
    flat = rng.integers(0, ext.num_total_intervals, size=n_cells)
    multi = np.stack(np.unravel_index(flat, ext.num_intervals), axis=1)

    if op_kind == "apply":
        operand = rng.standard_normal((n_cells, n_in))
    elif op_kind == "apply_T":
        operand = rng.standard_normal((n_cells, n_out))
    elif op_kind == "MT_K_M":
        operand = rng.standard_normal((n_cells, n_out, n_out))
    else:
        operand = rng.standard_normal((n_cells, n_in, n_in))

    batch_fn = _batch_method(ext, op_kind)
    out_flat = batch_fn(operand, flat)
    out_multi = batch_fn(operand, multi)
    np.testing.assert_allclose(out_flat, out_multi, atol=1e-14)


@pytest.mark.parametrize("op_kind", _OP_KINDS)
def test_batch_apply_identity_pattern_shortcircuits(op_kind: str) -> None:
    """C⁰ Bézier space (all identity) gives the same result as no operator."""
    sp1 = BsplineSpace1D([0, 0, 0, 1, 1, 1, 2, 2, 2], 2)
    sp = BsplineSpace([sp1, sp1])
    ext = SpanwiseElementExtraction(sp, "bezier")
    assert ext.is_identity, "Space should have all-identity extraction"
    n_in = int(np.prod(ext.input_shape_per_dir))
    n_out = int(np.prod(ext.output_shape_per_dir))
    rng = np.random.default_rng(2)
    n_cells = 4
    flat = rng.integers(0, ext.num_total_intervals, size=n_cells)

    if op_kind in ("apply", "apply_T"):
        operand = rng.standard_normal((n_cells, n_in if op_kind == "apply" else n_out))
        out = _batch_method(ext, op_kind)(operand, flat)
        np.testing.assert_allclose(out, operand, atol=1e-12)
    elif op_kind == "MT_K_M":
        K = rng.standard_normal((n_cells, n_out, n_out))
        out = ext.apply_MT_K_M_many(K, flat)
        np.testing.assert_allclose(out, K, atol=1e-12)
    else:
        K = rng.standard_normal((n_cells, n_in, n_in))
        out = ext.apply_M_K_MT_many(K, flat)
        np.testing.assert_allclose(out, K, atol=1e-12)


# ---------------------------------------------------------------- batch edge cases / validation


def test_batch_apply_empty_batch_returns_correct_shape() -> None:
    """An empty batch (n_cells=0) returns an array with the right shape and no errors."""
    ext = SpanwiseElementExtraction(_space_2d(), "bezier")
    n_in = int(np.prod(ext.input_shape_per_dir))
    n_out = int(np.prod(ext.output_shape_per_dir))
    empty_idx = np.array([], dtype=np.intp)
    out = ext.apply_many(np.empty((0, n_in)), empty_idx)
    assert out.shape == (0, n_out)


def test_batch_apply_accepts_user_out_and_scratch() -> None:
    """User-provided ``out`` and ``scratch`` are written into and returned."""
    ext = SpanwiseElementExtraction(_space_2d(), "bezier")
    n_in = int(np.prod(ext.input_shape_per_dir))
    n_out = int(np.prod(ext.output_shape_per_dir))
    rng = np.random.default_rng(3)
    n_cells = 3
    flat = rng.integers(0, ext.num_total_intervals, size=n_cells)
    v = rng.standard_normal((n_cells, n_in))
    out = np.zeros((n_cells, n_out))
    scratch = np.zeros((n_cells, 4 * n_in * n_out))
    result = ext.apply_many(v, flat, out=out, scratch=scratch)
    assert result is out

    for c in range(n_cells):
        ref = ext.apply(v[c], int(flat[c]))
        np.testing.assert_allclose(out[c], ref, atol=1e-12)


def test_batch_apply_wrong_operand_shape_raises() -> None:
    """Wrong operand shape raises ValueError."""
    ext = SpanwiseElementExtraction(_space_2d(), "bezier")
    n_in = int(np.prod(ext.input_shape_per_dir))
    with pytest.raises(ValueError, match="shape"):
        ext.apply_many(np.zeros((3, n_in + 1)), np.array([0, 1, 2]))


def test_batch_apply_wrong_cell_indices_ndim_raises() -> None:
    """3-D cell_indices raises ValueError."""
    ext = SpanwiseElementExtraction(_space_2d(), "bezier")
    n_in = int(np.prod(ext.input_shape_per_dir))
    with pytest.raises(ValueError):
        ext.apply_many(np.zeros((1, n_in)), np.zeros((1, 1, 2), dtype=np.intp))


def test_batch_apply_out_of_range_cell_index_raises() -> None:
    """Out-of-range cell index raises IndexError."""
    ext = SpanwiseElementExtraction(_space_2d(), "bezier")
    n_in = int(np.prod(ext.input_shape_per_dir))
    with pytest.raises((IndexError, ValueError)):
        ext.apply_many(np.zeros((1, n_in)), np.array([ext.num_total_intervals]))


def test_batch_apply_dim4_raises_not_implemented() -> None:
    """A 4D space raises NotImplementedError for all batch apply variants."""
    sp1 = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
    sp = BsplineSpace([sp1, sp1, sp1, sp1])
    ext = SpanwiseElementExtraction(sp, "bezier")
    n_in = int(np.prod(ext.input_shape_per_dir))
    with pytest.raises(NotImplementedError):
        ext.apply_many(np.zeros((1, n_in)), np.array([0]))
