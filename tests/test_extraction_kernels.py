"""Correctness and validation tests for the extraction kernels and helpers."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, available_backends, use_backend
from pantr.bspline._extraction_helpers import (
    OpKind,
    _allocate_or_validate_scratch,
    _apply_scratch_size,
    _bilateral_scratch_size,
    _dispatch_apply,
    _operation_shapes,
    _prepare_apply_call,
    _prepare_apply_many_call,
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
    """Naive reference using the materialized Kronecker product.

    Computes in float64 to avoid float32 overflow/subnormal warnings; the
    result is cast back to the original dtype for comparison.
    """
    M = _full_kron([op.astype(np.float64) for op in ops])
    op64 = operand.astype(np.float64)
    if op_kind == "apply":
        result = M @ op64
    elif op_kind == "apply_T":
        result = M.T @ op64
    elif op_kind == "MT_K_M":
        result = M.T @ op64 @ M
    elif op_kind == "M_K_MT":
        result = M @ op64 @ M.T
    else:
        raise ValueError(f"unknown op_kind {op_kind}")
    return result.astype(operand.dtype)


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


def _all_identity_batch(
    d: int, side: int, n_cells: int, dtype: np.dtype[Any]
) -> tuple[
    tuple[npt.NDArray[Any], ...],
    tuple[npt.NDArray[Any], ...],
    tuple[npt.NDArray[Any], ...],
    npt.NDArray[np.intp],
]:
    """Build compact storage in which every referenced element is the identity.

    The compact layout is what the batch kernels actually take: a sentinel row per
    direction, an index map that is all zeros, and an all-true mask. That is exactly
    what :class:`~pantr.bspline.SpanwiseElementExtraction` stores for a direction
    with no non-identity element.

    Args:
        d (int): Number of tensor-product directions.
        side (int): Per-direction extent, equal in and out since identity is square.
        n_cells (int): How many cells the batch visits.
        dtype (np.dtype[Any]): Storage format.

    Returns:
        tuple: ``(ops_1d, idx_maps_1d, is_identity_masks, cell_indices)``.
    """
    n_elements = 3
    ops = tuple(np.zeros((1, side, side), dtype=dtype) for _ in range(d))
    maps = tuple(np.zeros(n_elements, dtype=np.intp) for _ in range(d))
    masks = tuple(np.ones(n_elements, dtype=np.bool_) for _ in range(d))
    cells = np.ascontiguousarray(
        np.stack([np.arange(n_cells) % n_elements for _ in range(d)], axis=1), dtype=np.intp
    )
    return ops, maps, masks, cells


@pytest.mark.parametrize("backend", available_backends())
@pytest.mark.parametrize("d", [1, 2, 3])
def test_all_identity_batch_apply_aliasing(backend: Backend, d: int) -> None:
    """An all-identity batch is copy-through, so ``out=v`` is legal in every backend.

    The batch counterpart of :func:`test_all_identity_apply_aliasing`, and it did not
    exist. ``_prepare_apply_many_call`` permits the aliasing explicitly, by checking
    ``is_identity_masks[k][cell_indices[:, k]].all()`` for every direction; a first
    version of the C++ binding applied that exemption to the single-cell entry points
    and not to the batch ones, so this call raised under the C++ backend and
    succeeded under the Python one. Both existing batch aliasing tests use a *mixed*
    batch, correctly expect rejection, and so never reach the exemption at all.

    Parametrized over the backends this installation actually has, so the divergence
    is caught in one run rather than only by the separate ``PANTR_BACKEND=cpp`` step.
    """
    side, n_cells = 3, 4
    ops, maps, masks, cells = _all_identity_batch(d, side, n_cells, np.dtype(np.float64))
    for op_kind in ("apply", "apply_T"):
        operand = RNG.standard_normal((n_cells, side**d))
        expected = operand.copy()
        with use_backend(backend):
            kernel, args, out = _prepare_apply_many_call(
                ops, maps, masks, cells, operand, operand, None, op_kind
            )
            assert out is operand
            kernel(*args)
        np.testing.assert_array_equal(operand, expected)


@pytest.mark.parametrize("backend", available_backends())
@pytest.mark.parametrize("d", [1, 2, 3])
def test_all_identity_batch_bilateral_aliasing(backend: Backend, d: int) -> None:
    """An all-identity batch is copy-through for the bilateral kinds too; ``out=K`` is legal.

    See :func:`test_all_identity_batch_apply_aliasing` for what this pins.
    """
    side, n_cells = 3, 4
    ops, maps, masks, cells = _all_identity_batch(d, side, n_cells, np.dtype(np.float64))
    for op_kind in ("MT_K_M", "M_K_MT"):
        operand = RNG.standard_normal((n_cells, side**d, side**d))
        expected = operand.copy()
        with use_backend(backend):
            kernel, args, out = _prepare_apply_many_call(
                ops, maps, masks, cells, operand, operand, None, op_kind
            )
            assert out is operand
            kernel(*args)
        np.testing.assert_array_equal(operand, expected)


@pytest.mark.parametrize("backend", available_backends())
def test_mixed_batch_aliasing_is_still_refused(backend: Backend) -> None:
    """One contracting cell in the batch withdraws the exemption, in both backends.

    The other half of the contract, and the half a too-permissive fix would break:
    the exemption is a property of the whole call, because one contracting cell makes
    the shared buffers overlap in a way no later cell's copy undoes.
    """
    side, n_cells = 3, 4
    ops, maps, masks, cells = _all_identity_batch(
        d=2, side=side, n_cells=n_cells, dtype=np.dtype(np.float64)
    )
    # Give direction 0 a real operator at element 1, and point one cell at it.
    contracting = np.zeros((2, side, side))
    contracting[1] = RNG.standard_normal((side, side))
    ops = (contracting, ops[1])
    maps = (np.array([0, 1, 0], dtype=np.intp), maps[1])
    masks = (np.array([True, False, True], dtype=np.bool_), masks[1])

    operand = RNG.standard_normal((n_cells, side**2))
    with use_backend(backend), pytest.raises(ValueError, match="alias"):
        _prepare_apply_many_call(ops, maps, masks, cells, operand, operand, None, "apply")


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
    """Dispatcher maps (op_kind, d) to the right module-level kernel.

    Pinned to :attr:`~pantr._backend.Backend.PYTHON`, because the dispatcher now
    resolves a backend: under the C++ one it hands back an adapter over the binding
    of the same name, and this test is about the table rather than about which
    implementation is in effect.
    """
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
    with use_backend(Backend.PYTHON):
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
