"""Tests for :class:`SpanwiseElementExtraction`.

Covers:

- Target dispatch (``bezier``, ``lagrange``, ``cardinal``) and construction
  errors (unknown target, periodic directions).
- Shape/dtype and identity-mask properties.
- Per-cell :meth:`apply` / :meth:`apply_transpose` / :meth:`apply_MT_K_M` /
  :meth:`apply_M_K_MT` against a dense reference ``kron(M_0, M_1, …)``.
- :meth:`operator` / :meth:`tabulate` round-trip consistency, including
  user-provided ``out`` arrays.
- Identity-query correctness: :attr:`is_identity`, :attr:`num_identity_elements`,
  :meth:`per_direction_identity_flags`; all three targets use structural (exact)
  identity predicates — no numerical tolerance.
- Cell-index normalization (flat int, tuple, list, ndarray; negative indices
  rejected; IndexError / ValueError / TypeError on invalid input).
- 1D spaces and rejection of dim > 3 via :exc:`NotImplementedError`.
- Batch :meth:`apply_many` / :meth:`apply_transpose_many` /
  :meth:`apply_MT_K_M_many` / :meth:`apply_M_K_MT_many`: round-trip match
  against per-cell :meth:`apply` for all (d, op_kind, target, dtype) combos,
  flat and per-direction index forms, user-provided ``out``/``scratch`` for all
  four variants (including bilateral), empty batch for all four variants,
  batch-size-1, aliasing rejection, wrong ``out`` shape, and validation errors
  for :func:`_allocate_or_validate_scratch_many`.
- :func:`normalize_cell_indices`: flat-to-2D conversion, per-direction
  pass-through, negative index rejection, float input rejection, and error cases.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr.basis import LagrangeVariant
from pantr.bspline import BsplineSpace, BsplineSpace1D, SpanwiseElementExtraction
from pantr.bspline._extraction_helpers import (
    _allocate_or_validate_scratch_many,
    _required_scratch_size,
)
from pantr.bspline.spanwise_element_extraction import (
    _bezier_structural_identity_mask,
    _lagrange_structural_identity_mask,
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


def test_bezier_c0_space_is_structurally_identity_everywhere() -> None:
    """A C⁻¹ (Bézier) space has identity Bézier extraction on every element."""
    sp1 = BsplineSpace1D([0, 0, 0, 1, 1, 1, 2, 2, 2], 2)
    sp = BsplineSpace([sp1, sp1])
    ext = SpanwiseElementExtraction(sp, "bezier")
    for mask in ext.is_identity_mask_1d:
        assert mask.all()
    assert ext.is_identity_at(0)
    assert ext.num_identity_elements == ext.num_total_intervals


def test_bezier_structural_identity_mask_only_at_full_multiplicity_knots() -> None:
    """Bezier identity holds only at elements whose boundary knots have mult >= degree+1."""
    # degree 2, interior knots at 1 (mult 2), 2 (mult 3), 3 (mult 2), 4 (mult 3)
    # Elements: [0,1] (mults 3,2), [1,2] (mults 2,3), [2,3] (mults 3,2), [3,4] (mults 2,3)
    sp1 = BsplineSpace1D([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4], 2)
    mask = _bezier_structural_identity_mask(sp1)
    # Elements with both boundary mults >= 3: element 1 ([1,2], mults 2,3 → left=2 < 3, not id)
    # element 0: left=3 ✓, right=2 ✗ → False
    # element 1: left=2 ✗ → False
    # element 2: left=3 ✓, right=2 ✗ → False
    # element 3: left=2 ✗ → False
    assert not mask.any()

    # Now a fully C⁻¹ space: all interior knots have mult = degree+1 = 3
    sp2 = BsplineSpace1D([0, 0, 0, 1, 1, 1, 2, 2, 2], 2)
    mask2 = _bezier_structural_identity_mask(sp2)
    assert mask2.all()

    # Smooth space (mult 1 everywhere): no identity
    sp3 = BsplineSpace1D([0, 0, 0, 1, 2, 3, 3, 3], 2)
    mask3 = _bezier_structural_identity_mask(sp3)
    assert not mask3.any()


def test_lagrange_structural_identity_degree0_all_identity() -> None:
    """Lagrange degree-0 space: every element is identity."""
    sp1 = BsplineSpace1D([0, 1, 2, 3], 0)
    mask = _lagrange_structural_identity_mask(sp1)
    assert mask.all()
    assert len(mask) == sp1.num_intervals


def test_lagrange_structural_identity_degree_positive_never_identity() -> None:
    """Lagrange degree > 0: no element is identity regardless of the knot vector."""
    for degree in (1, 2, 3):
        knots = [0.0] * (degree + 1) + [1.0, 2.0, 3.0] + [3.0] * (degree + 1)
        sp1 = BsplineSpace1D(knots, degree)
        mask = _lagrange_structural_identity_mask(sp1)
        assert not mask.any(), f"degree {degree} should yield no identity elements"


def test_identity_tol_kwarg_removed() -> None:
    """Passing ``identity_tol`` raises ``TypeError`` (parameter no longer exists)."""
    sp = _space_2d()
    with pytest.raises(TypeError):
        SpanwiseElementExtraction(sp, "bezier", identity_tol=1e-10)  # type: ignore[call-arg]


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
    """Flat indices outside [0, total) raise IndexError."""
    with pytest.raises(IndexError, match="Flat cell indices"):
        normalize_cell_indices(np.array([12]), (3, 4))  # 3*4=12, so 12 is OOB


def test_normalize_negative_flat_index_raises() -> None:
    """Negative flat indices raise IndexError."""
    with pytest.raises(IndexError, match="Flat cell indices"):
        normalize_cell_indices(np.array([-1]), (3, 4))


def test_normalize_per_direction_out_of_range() -> None:
    """Per-direction indices outside per-direction bounds raise IndexError."""
    with pytest.raises(IndexError, match="cell_indices"):
        normalize_cell_indices(np.array([[3, 0]]), (3, 4))  # dir-0 max is 2


def test_normalize_float_input_raises() -> None:
    """Float cell_indices raise TypeError."""
    with pytest.raises(TypeError, match="integers"):
        normalize_cell_indices(np.array([0.0, 1.5, 2.9]), (3, 4))


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
    with pytest.raises(IndexError):
        ext.apply_many(np.zeros((1, n_in)), np.array([ext.num_total_intervals]))


def test_batch_apply_dim4_raises_not_implemented() -> None:
    """A 4D space raises NotImplementedError for all batch apply variants."""
    sp1 = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
    sp = BsplineSpace([sp1, sp1, sp1, sp1])
    ext = SpanwiseElementExtraction(sp, "bezier")
    n_in = int(np.prod(ext.input_shape_per_dir))
    with pytest.raises(NotImplementedError):
        ext.apply_many(np.zeros((1, n_in)), np.array([0]))


def test_batch_apply_size_one_matches_per_cell() -> None:
    """A batch of exactly one cell produces the same result as the per-cell method."""
    ext = SpanwiseElementExtraction(_space_2d(), "bezier")
    n_in = int(np.prod(ext.input_shape_per_dir))
    rng = np.random.default_rng(42)
    v = rng.standard_normal((1, n_in))
    cell = 7
    out_batch = ext.apply_many(v, np.array([cell]))
    out_single = ext.apply(v[0], cell)
    np.testing.assert_allclose(out_batch[0], out_single, atol=1e-12)


def test_batch_apply_bilateral_accepts_user_out_and_scratch() -> None:
    """User-provided ``out`` and ``scratch`` work correctly for bilateral variants."""
    ext = SpanwiseElementExtraction(_space_2d(), "bezier")
    n_in = int(np.prod(ext.input_shape_per_dir))
    n_out = int(np.prod(ext.output_shape_per_dir))
    rng = np.random.default_rng(5)
    n_cells = 3
    flat = rng.integers(0, ext.num_total_intervals, size=n_cells)

    # MT_K_M: operand (n_cells, N_out, N_out) → out (n_cells, N_in, N_in)
    K_mtk = rng.standard_normal((n_cells, n_out, n_out))
    scratch_sz = _required_scratch_size(ext.input_shape_per_dir, ext.output_shape_per_dir, "MT_K_M")
    out_mtk = np.zeros((n_cells, n_in, n_in))
    scratch_mtk = np.zeros((n_cells, max(scratch_sz, 1)))
    result_mtk = ext.apply_MT_K_M_many(K_mtk, flat, out=out_mtk, scratch=scratch_mtk)
    assert result_mtk is out_mtk
    for c in range(n_cells):
        ref = ext.apply_MT_K_M(K_mtk[c], int(flat[c]))
        np.testing.assert_allclose(out_mtk[c], ref, atol=1e-11)

    # M_K_MT: operand (n_cells, N_in, N_in) → out (n_cells, N_out, N_out)
    K_mkt = rng.standard_normal((n_cells, n_in, n_in))
    scratch_sz2 = _required_scratch_size(
        ext.input_shape_per_dir, ext.output_shape_per_dir, "M_K_MT"
    )
    out_mkt = np.zeros((n_cells, n_out, n_out))
    scratch_mkt = np.zeros((n_cells, max(scratch_sz2, 1)))
    result_mkt = ext.apply_M_K_MT_many(K_mkt, flat, out=out_mkt, scratch=scratch_mkt)
    assert result_mkt is out_mkt
    for c in range(n_cells):
        ref = ext.apply_M_K_MT(K_mkt[c], int(flat[c]))
        np.testing.assert_allclose(out_mkt[c], ref, atol=1e-11)


def test_batch_apply_empty_batch_bilateral_shapes() -> None:
    """Empty batch returns the correct shape for all four variants."""
    ext = SpanwiseElementExtraction(_space_2d(), "bezier")
    n_in = int(np.prod(ext.input_shape_per_dir))
    n_out = int(np.prod(ext.output_shape_per_dir))
    empty = np.array([], dtype=np.intp)

    assert ext.apply_many(np.empty((0, n_in)), empty).shape == (0, n_out)
    assert ext.apply_transpose_many(np.empty((0, n_out)), empty).shape == (0, n_in)
    assert ext.apply_MT_K_M_many(np.empty((0, n_out, n_out)), empty).shape == (0, n_in, n_in)
    assert ext.apply_M_K_MT_many(np.empty((0, n_in, n_in)), empty).shape == (0, n_out, n_out)


def test_batch_apply_bilateral_aliasing_raises() -> None:
    """Passing the same array as both K and out raises ValueError."""
    ext = SpanwiseElementExtraction(_space_2d(), "bezier")
    n_out = int(np.prod(ext.output_shape_per_dir))
    n_in = int(np.prod(ext.input_shape_per_dir))
    rng = np.random.default_rng(7)
    flat = rng.integers(0, ext.num_total_intervals, size=2)

    K = rng.standard_normal((2, n_out, n_out))
    with pytest.raises(ValueError, match="alias"):
        ext.apply_MT_K_M_many(K, flat, out=K)

    K2 = rng.standard_normal((2, n_in, n_in))
    with pytest.raises(ValueError, match="alias"):
        ext.apply_M_K_MT_many(K2, flat, out=K2)


def test_batch_apply_wrong_out_shape_raises() -> None:
    """Wrong ``out`` shape raises ValueError."""
    ext = SpanwiseElementExtraction(_space_2d(), "bezier")
    n_in = int(np.prod(ext.input_shape_per_dir))
    n_out = int(np.prod(ext.output_shape_per_dir))
    n_cells = 3
    v = np.zeros((n_cells, n_in))
    flat = np.arange(n_cells)
    with pytest.raises(ValueError, match="shape"):
        ext.apply_many(v, flat, out=np.zeros((n_cells, n_out + 1)))


def test_allocate_or_validate_scratch_many_rejects_invalid() -> None:
    """All five rejection branches in ``_allocate_or_validate_scratch_many`` raise ValueError."""
    n_cells, width, dtype = 4, 8, np.float64

    # Wrong ndim
    with pytest.raises(ValueError, match="2D"):
        _allocate_or_validate_scratch_many(np.zeros(n_cells * width), n_cells, width, dtype)

    # Wrong dtype
    with pytest.raises(ValueError, match="dtype"):
        _allocate_or_validate_scratch_many(
            np.zeros((n_cells, width), dtype=np.float32), n_cells, width, dtype
        )

    # Wrong shape[0]
    with pytest.raises(ValueError, match="n_cells"):
        _allocate_or_validate_scratch_many(
            np.zeros((n_cells + 1, width), dtype=dtype), n_cells, width, dtype
        )

    # Too narrow (scratch.shape[1] < alloc_width = max(width, 1))
    with pytest.raises(ValueError, match="width"):
        _allocate_or_validate_scratch_many(
            np.zeros((n_cells, width - 1), dtype=dtype), n_cells, width, dtype
        )

    # d=1 case: scratch_size_per_cell=0 so alloc_width=1; shape[1]=0 must be rejected
    with pytest.raises(ValueError, match="width"):
        _allocate_or_validate_scratch_many(np.zeros((n_cells, 0), dtype=dtype), n_cells, 0, dtype)

    # Not writeable
    ro = np.zeros((n_cells, width), dtype=dtype)
    ro.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        _allocate_or_validate_scratch_many(ro, n_cells, width, dtype)
