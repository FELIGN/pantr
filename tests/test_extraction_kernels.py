"""Correctness and validation tests for the extraction kernels and helpers."""

from __future__ import annotations

import itertools
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline._extraction_helpers import (
    OpKind,
    _allocate_or_validate_scratch,
    _apply_scratch_size,
    _bilateral_scratch_size,
    _dispatch_apply,
    _operation_shapes,
    _prepare_apply_call,
    _required_scratch_size,
    _validate_op_kind,
)
from pantr.bspline._extraction_kernels import (
    apply_kron_1d,
    apply_kron_2d,
    apply_kron_3d,
    apply_kron_M_K_MT_1d,
    apply_kron_M_K_MT_2d,
    apply_kron_M_K_MT_3d,
    apply_kron_MT_K_M_1d,
    apply_kron_MT_K_M_2d,
    apply_kron_MT_K_M_3d,
    apply_kron_T_1d,
    apply_kron_T_2d,
    apply_kron_T_3d,
)

RNG = np.random.default_rng(12345)


def _make_ops(
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    identity: tuple[bool, ...],
    dtype: np.dtype[Any],
) -> list[npt.NDArray[Any]]:
    """Build per-direction 2D operators, using identity for flagged directions."""
    ops: list[npt.NDArray[Any]] = []
    for n_in, n_out, is_id in zip(in_shape, out_shape, identity, strict=True):
        if is_id:
            assert n_in == n_out, "identity direction requires n_in == n_out"
            ops.append(np.eye(n_in, dtype=dtype))
        else:
            ops.append(RNG.standard_normal((n_out, n_in)).astype(dtype))
    return ops


def _full_kron(ops: list[npt.NDArray[Any]]) -> npt.NDArray[Any]:
    """Assemble the full Kronecker product from per-direction operators."""
    result = ops[0]
    for M in ops[1:]:
        result = np.kron(result, M)
    return result


def _reference(
    ops: list[npt.NDArray[Any]],
    operand: npt.NDArray[Any],
    op_kind: str,
) -> npt.NDArray[Any]:
    """Naive reference using the materialized Kronecker product."""
    M = _full_kron(ops)
    if op_kind == "apply":
        return cast(npt.NDArray[Any], M @ operand)
    if op_kind == "apply_T":
        return cast(npt.NDArray[Any], M.T @ operand)
    if op_kind == "MT_K_M":
        return cast(npt.NDArray[Any], M.T @ operand @ M)
    if op_kind == "M_K_MT":
        return cast(npt.NDArray[Any], M @ operand @ M.T)
    raise ValueError(f"unknown op_kind {op_kind}")


def _identity_patterns(d: int) -> list[tuple[bool, ...]]:
    """Enumerate all 2^d identity-flag combinations for dimension d."""
    return list(itertools.product([False, True], repeat=d))


# -- Shape configurations ------------------------------------------------------

# Triples (d, in_shape, out_shape). For identity directions n_in == n_out is
# required, so we use configurations where at least the "square" axes have
# matching sizes; the test harness picks identity flags accordingly.
SHAPE_CONFIGS_SQUARE: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = [
    (1, (3,), (3,)),
    (1, (5,), (5,)),
    (2, (3, 4), (3, 4)),
    (2, (4, 2), (4, 2)),
    (3, (3, 4, 2), (3, 4, 2)),
    (3, (2, 3, 4), (2, 3, 4)),
]

SHAPE_CONFIGS_NONSQUARE: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = [
    (1, (3,), (5,)),
    (1, (5,), (2,)),
    (2, (2, 3), (4, 2)),
    (2, (3, 4), (2, 5)),
    (3, (2, 3, 4), (3, 2, 5)),
    (3, (4, 2, 3), (2, 4, 2)),
]


# -- Correctness tests for square operators (identity patterns) ---------------


@pytest.mark.parametrize(("d", "in_shape", "out_shape"), SHAPE_CONFIGS_SQUARE)
@pytest.mark.parametrize("op_kind", ["apply", "apply_T", "MT_K_M", "M_K_MT"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_kernels_square_all_identity_patterns(
    d: int,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    op_kind: OpKind,
    dtype: np.dtype[Any],
) -> None:
    """Each kernel matches the naive Kronecker reference for every identity pattern."""
    tol = 5e-5 if dtype == np.float32 else 1e-10
    for identity in _identity_patterns(d):
        ops = _make_ops(in_shape, out_shape, identity, np.dtype(dtype))
        in_op_shape, _out_op_shape = _operation_shapes(in_shape, out_shape, op_kind)
        operand = RNG.standard_normal(in_op_shape).astype(dtype)
        expected = _reference(ops, operand, op_kind)

        kernel, args, out = _prepare_apply_call(tuple(ops), identity, operand, None, None, op_kind)
        kernel(*args)
        np.testing.assert_allclose(out, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize(("d", "in_shape", "out_shape"), SHAPE_CONFIGS_NONSQUARE)
@pytest.mark.parametrize("op_kind", ["apply", "apply_T", "MT_K_M", "M_K_MT"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_kernels_nonsquare(
    d: int,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    op_kind: OpKind,
    dtype: np.dtype[Any],
) -> None:
    """Non-square per-direction operators produce correct results (no identity)."""
    tol = 5e-5 if dtype == np.float32 else 1e-10
    identity = tuple([False] * d)
    ops = _make_ops(in_shape, out_shape, identity, np.dtype(dtype))
    in_op_shape, _ = _operation_shapes(in_shape, out_shape, op_kind)
    operand = RNG.standard_normal(in_op_shape).astype(dtype)
    expected = _reference(ops, operand, op_kind)

    kernel, args, out = _prepare_apply_call(tuple(ops), identity, operand, None, None, op_kind)
    kernel(*args)
    np.testing.assert_allclose(out, expected, atol=tol, rtol=tol)


# -- All-identity aliasing tests ----------------------------------------------


@pytest.mark.parametrize(("d", "in_shape", "out_shape"), SHAPE_CONFIGS_SQUARE[:3])
def test_all_identity_apply_aliasing(
    d: int,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
) -> None:
    """``apply`` / ``apply_T`` are copy-through in the all-identity case; out=v is legal."""
    identity = tuple([True] * d)
    ops = _make_ops(in_shape, out_shape, identity, np.dtype(np.float64))
    for op_kind in ("apply", "apply_T"):
        in_shape_op, _ = _operation_shapes(in_shape, out_shape, op_kind)
        v = RNG.standard_normal(in_shape_op).astype(np.float64)
        v_copy = v.copy()
        # Pass v itself as `out`: in the all-identity case this should be a
        # self-copy (no-op for the values), and v remains unchanged.
        kernel, args, out = _prepare_apply_call(tuple(ops), identity, v, v, None, op_kind)
        assert out is v
        kernel(*args)
        np.testing.assert_array_equal(v, v_copy)


@pytest.mark.parametrize(("d", "in_shape", "out_shape"), SHAPE_CONFIGS_SQUARE[:3])
def test_all_identity_bilateral_aliasing(
    d: int,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
) -> None:
    """``MT_K_M`` / ``M_K_MT`` are copy-through in the all-identity case; out=K is legal."""
    identity = tuple([True] * d)
    ops = _make_ops(in_shape, out_shape, identity, np.dtype(np.float64))
    for op_kind in ("MT_K_M", "M_K_MT"):
        in_shape_op, _ = _operation_shapes(in_shape, out_shape, op_kind)
        K = RNG.standard_normal(in_shape_op).astype(np.float64)
        K_copy = K.copy()
        kernel, args, out = _prepare_apply_call(tuple(ops), identity, K, K, None, op_kind)
        assert out is K
        kernel(*args)
        np.testing.assert_array_equal(K, K_copy)


# -- Dispatcher errors --------------------------------------------------------


def test_dispatch_d_too_large_raises() -> None:
    with pytest.raises(NotImplementedError, match=r"d in \{1, 2, 3\}"):
        _dispatch_apply(4, "apply")


def test_dispatch_d_too_small_raises() -> None:
    with pytest.raises(NotImplementedError, match=r"d in \{1, 2, 3\}"):
        _dispatch_apply(0, "apply")


def test_dispatch_unknown_op_kind_raises() -> None:
    with pytest.raises(ValueError, match="Unknown op_kind"):
        _dispatch_apply(2, "banana")  # type: ignore[arg-type]


def test_dispatch_returns_expected_kernels() -> None:
    """Dispatcher maps (op_kind, d) to the right module-level kernel."""
    expected: dict[tuple[str, int], object] = {
        ("apply", 1): apply_kron_1d,
        ("apply", 2): apply_kron_2d,
        ("apply", 3): apply_kron_3d,
        ("apply_T", 1): apply_kron_T_1d,
        ("apply_T", 2): apply_kron_T_2d,
        ("apply_T", 3): apply_kron_T_3d,
        ("MT_K_M", 1): apply_kron_MT_K_M_1d,
        ("MT_K_M", 2): apply_kron_MT_K_M_2d,
        ("MT_K_M", 3): apply_kron_MT_K_M_3d,
        ("M_K_MT", 1): apply_kron_M_K_MT_1d,
        ("M_K_MT", 2): apply_kron_M_K_MT_2d,
        ("M_K_MT", 3): apply_kron_M_K_MT_3d,
    }
    for (op_kind, d), kernel in expected.items():
        assert _dispatch_apply(d, op_kind) is kernel  # type: ignore[arg-type]


# -- Validation errors --------------------------------------------------------


def test_validate_op_kind_accepts_known() -> None:
    for k in ("apply", "apply_T", "MT_K_M", "M_K_MT"):
        assert _validate_op_kind(k) == k


def test_validate_op_kind_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown op_kind"):
        _validate_op_kind("nope")


def test_required_scratch_size_mismatched_shapes_raises() -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        _required_scratch_size((2, 3), (2,), "apply")


def test_operand_wrong_shape_raises() -> None:
    ops = _make_ops((3, 4), (3, 4), (False, False), np.dtype(np.float64))
    bad = RNG.standard_normal(20).astype(np.float64)
    with pytest.raises(ValueError, match="expected shape"):
        _prepare_apply_call(tuple(ops), (False, False), bad, None, None, "apply")


def test_operand_wrong_dtype_raises() -> None:
    ops = _make_ops((3, 4), (3, 4), (False, False), np.dtype(np.float64))
    bad = RNG.standard_normal(12).astype(np.float32)
    with pytest.raises(ValueError, match="expected dtype"):
        _prepare_apply_call(tuple(ops), (False, False), bad, None, None, "apply")


def test_out_wrong_shape_raises() -> None:
    ops = _make_ops((3, 4), (3, 4), (False, False), np.dtype(np.float64))
    v = RNG.standard_normal(12).astype(np.float64)
    bad_out = np.empty(11, dtype=np.float64)
    with pytest.raises(ValueError, match="expected shape"):
        _prepare_apply_call(tuple(ops), (False, False), v, bad_out, None, "apply")


def test_out_not_writable_raises() -> None:
    ops = _make_ops((3, 4), (3, 4), (False, False), np.dtype(np.float64))
    v = RNG.standard_normal(12).astype(np.float64)
    bad_out = np.empty(12, dtype=np.float64)
    bad_out.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        _prepare_apply_call(tuple(ops), (False, False), v, bad_out, None, "apply")


def test_scratch_too_small_raises() -> None:
    ops = _make_ops((3, 4), (3, 4), (False, False), np.dtype(np.float64))
    v = RNG.standard_normal(12).astype(np.float64)
    bad_scratch = np.empty(1, dtype=np.float64)
    with pytest.raises(ValueError, match="smaller than required"):
        _prepare_apply_call(tuple(ops), (False, False), v, None, bad_scratch, "apply")


def test_scratch_wrong_ndim_raises() -> None:
    with pytest.raises(ValueError, match="1D"):
        _allocate_or_validate_scratch(np.empty((3, 3), dtype=np.float64), 4, np.float64)


def test_scratch_wrong_dtype_raises() -> None:
    with pytest.raises(ValueError, match="expected dtype"):
        _allocate_or_validate_scratch(np.empty(16, dtype=np.float32), 4, np.float64)


def test_ops_ndim_raises() -> None:
    bad_op = np.eye(3)[np.newaxis]  # 3D instead of 2D
    v = RNG.standard_normal(3).astype(np.float64)
    with pytest.raises(ValueError, match="must be 2D"):
        _prepare_apply_call((bad_op,), (False,), v, None, None, "apply")


def test_ops_dtype_mismatch_raises() -> None:
    ops: tuple[npt.NDArray[Any], ...] = (
        np.eye(3, dtype=np.float64),
        np.eye(4, dtype=np.float32),
    )
    v = RNG.standard_normal(12).astype(np.float64)
    with pytest.raises(ValueError, match="dtype"):
        _prepare_apply_call(ops, (False, False), v, None, None, "apply")


def test_identity_length_mismatch_raises() -> None:
    ops = _make_ops((3, 4), (3, 4), (False, False), np.dtype(np.float64))
    v = RNG.standard_normal(12).astype(np.float64)
    with pytest.raises(ValueError, match="length"):
        _prepare_apply_call(tuple(ops), (False,), v, None, None, "apply")


# -- Scratch size sanity -------------------------------------------------------


def test_required_scratch_size_apply_d1_is_zero() -> None:
    assert _required_scratch_size((5,), (3,), "apply") == 0
    assert _required_scratch_size((5,), (3,), "apply_T") == 0


def test_apply_scratch_size_d2_matches_kernel_usage() -> None:
    n_in, n_out = (3, 4), (5, 6)
    assert _apply_scratch_size(n_in, n_out) == 2 * (n_out[0] * n_in[1])
    # apply_T: starts at n_out, ends at n_in -- intermediate size n_in[0] * n_out[1].
    assert _apply_scratch_size(n_out, n_in) == 2 * (n_in[0] * n_out[1])


def test_bilateral_scratch_size_d1_matches_kernel_usage() -> None:
    # Stage 0 intermediate (only intermediate) is n_in * n_out.
    assert _bilateral_scratch_size((5,), (3,)) == 2 * (5 * 3)
    assert _bilateral_scratch_size((3,), (5,)) == 2 * (3 * 5)
