"""Tests for :func:`pantr.mpi.quasi_interpolate_bspline_distributed`.

MPI is not available in the test environment, so the communicator is duck-typed by
``_FakeComm``, which adds ``allgather`` support (needed by this function) on top of
the basic rank/size interface used elsewhere.

The multi-rank equivalence test (distributed result == serial result) lives in
``tests/mpi/test_distributed_mpi.py`` and runs under ``mpiexec``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import pantr.mpi as _pantr_mpi
from pantr.bspline import (
    Bspline,
    BsplineSpace,
    create_thb_space,
    create_uniform_space,
    quasi_interpolate_bspline,
)
from pantr.bspline._bspline_quasi_interpolation import QIKind
from pantr.grid import Partition, partition_grid, tensor_product_grid
from pantr.mpi import (
    DistributedFunction,
    DistributedSpace,
    quasi_interpolate_bspline_distributed,
)

# ---------------------------------------------------------------------------
# Fake communicator with allgather
# ---------------------------------------------------------------------------


class _FakeComm:
    """Minimal stand-in for an mpi4py communicator.

    Supports ``rank``, ``size``, and a single-rank ``allgather`` (returns
    ``[data]``).  For multi-rank simulations pass ``allgather_results`` to
    set the value returned regardless of local data.
    """

    def __init__(
        self,
        rank: int,
        size: int,
        allgather_results: list[Any] | None = None,
    ) -> None:
        self._rank = rank
        self._size = size
        self._allgather_results = allgather_results

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def size(self) -> int:
        return self._size

    def allgather(self, data: Any) -> list[Any]:
        if self._allgather_results is not None:
            return self._allgather_results
        assert self._size == 1, (
            "Multi-rank allgather requires pre-set allgather_results; "
            "use _simulate_distributed_qi() for multi-rank tests."
        )
        return [data]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tp_setup(n_parts: int, n_cells: int = 4) -> tuple[BsplineSpace, Any]:
    """Return a uniform 2D space and a n_parts-partition."""
    space = create_uniform_space([2, 2], [n_cells, n_cells])
    part = partition_grid(tensor_product_grid(space), n_parts)
    return space, part


def _simulate_distributed_qi(
    func: Any,
    space: BsplineSpace,
    n_parts: int,
    *,
    kind: QIKind = "llm",
) -> list[DistributedFunction]:
    """Simulate quasi_interpolate_bspline_distributed across n_parts ranks in one process.

    Runs the QI for each rank sequentially, collecting per-rank contributions
    first, then replays the allgather so each rank assembles the correct global field.
    """
    part = partition_grid(tensor_product_grid(space), n_parts)

    # Pass 1: collect what each rank would contribute to the allgather.
    contributions: list[tuple[Any, Any]] = []
    for rank in range(n_parts):
        ds = DistributedSpace(space, part, _FakeComm(rank, n_parts))
        local = ds.local
        if local is not None:
            assert isinstance(local.space, BsplineSpace)
            local_bspline = quasi_interpolate_bspline(func, local.space, kind=kind)
            cp = np.asarray(local_bspline.control_points, dtype=np.float64)
            cp_flat = cp.reshape(local.space.num_total_basis, -1)
            owned_mask = local.owned_dof_mask
            owned_global_dofs = local.local_to_global_dof[owned_mask]
            owned_coeffs = cp_flat[owned_mask]
        else:
            owned_global_dofs = np.empty(0, dtype=np.int64)
            owned_coeffs = np.empty((0, 0), dtype=np.float64)
        contributions.append((owned_global_dofs, owned_coeffs))

    # Pass 2: run the distributed QI for each rank with the pre-known allgather.
    results: list[DistributedFunction] = []
    for rank in range(n_parts):
        comm = _FakeComm(rank, n_parts, allgather_results=contributions)
        ds = DistributedSpace(space, part, comm)
        dfn = quasi_interpolate_bspline_distributed(func, ds, kind=kind)
        results.append(dfn)
    return results


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_surface() -> None:
    """quasi_interpolate_bspline_distributed is exported from pantr.mpi."""
    assert callable(_pantr_mpi.quasi_interpolate_bspline_distributed)
    assert "quasi_interpolate_bspline_distributed" in _pantr_mpi.__all__


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_returns_distributed_function() -> None:
    """Returns a DistributedFunction on the given distributed space."""
    space = create_uniform_space([2, 2], [4, 4])
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    dfn = quasi_interpolate_bspline_distributed(lambda p: p[:, 0] ** 2, ds)
    assert isinstance(dfn, DistributedFunction)
    assert dfn.distributed_space is ds
    assert dfn.global_function.space is space


def test_local_is_bspline_on_local_space() -> None:
    """Local is a Bspline on the rank's windowed space when it owns cells."""
    space = create_uniform_space([2, 2], [4, 4])
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    dfn = quasi_interpolate_bspline_distributed(lambda p: p[:, 0] + p[:, 1], ds)
    assert dfn.local is not None
    assert isinstance(dfn.local, Bspline)
    assert dfn.local.space is ds.local.space  # type: ignore[union-attr]


def test_empty_rank_has_none_local() -> None:
    """A rank owning no cells has local=None; still participates and returns a valid result."""
    space = create_uniform_space([2, 2], [4, 4])
    n_cells = space.num_total_intervals
    # Rank 0 owns everything; rank 1 is empty.
    part = Partition(np.zeros(n_cells, dtype=np.int64), 2)

    # Rank 1 (empty) contributes nothing; allgather from rank 0 provides the full field.
    ds0 = DistributedSpace(space, part, _FakeComm(0, 2))
    local0 = ds0.local
    assert local0 is not None
    assert isinstance(local0.space, BsplineSpace)
    local_bspline = quasi_interpolate_bspline(lambda p: p[:, 0], local0.space)
    cp = np.asarray(local_bspline.control_points, dtype=np.float64)
    cp_flat = cp.reshape(local0.space.num_total_basis, -1)
    owned_mask = local0.owned_dof_mask
    contrib0 = (local0.local_to_global_dof[owned_mask], cp_flat[owned_mask])
    contrib1 = (np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float64))
    allgather_data = [contrib0, contrib1]

    for rank in range(2):
        comm = _FakeComm(rank, 2, allgather_results=allgather_data)
        ds = DistributedSpace(space, part, comm)
        dfn = quasi_interpolate_bspline_distributed(lambda p: p[:, 0], ds)
        if rank == 0:
            assert dfn.local is not None
        else:
            assert dfn.local is None


# ---------------------------------------------------------------------------
# Scalar and vector functions
# ---------------------------------------------------------------------------


def test_scalar_function() -> None:
    """Scalar func → control points have shape (*num_basis, 1)."""
    space = create_uniform_space([2], [4])
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    dfn = quasi_interpolate_bspline_distributed(lambda p: p[:, 0] ** 2, ds)
    assert dfn.global_function.control_points.shape[-1] == 1


def test_vector_function() -> None:
    """Vector func (rank=2) → control points have shape (*num_basis, 2)."""
    space = create_uniform_space([2], [4])
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    dfn = quasi_interpolate_bspline_distributed(
        lambda p: np.stack([p[:, 0], p[:, 0] ** 2], axis=-1), ds
    )
    assert dfn.global_function.control_points.shape[-1] == 2


# ---------------------------------------------------------------------------
# Single-rank equivalence: distributed == serial
# ---------------------------------------------------------------------------


class TestSingleRankEquivalence:
    """With n_parts=1 the distributed result must equal the serial result exactly."""

    def _check(self, func: Any, space: BsplineSpace) -> None:
        part = partition_grid(tensor_product_grid(space), 1)
        ds = DistributedSpace(space, part, _FakeComm(0, 1))
        dfn = quasi_interpolate_bspline_distributed(func, ds)
        serial = quasi_interpolate_bspline(func, space)
        np.testing.assert_array_equal(
            dfn.global_function.control_points,
            serial.control_points,
        )

    def test_polynomial_1d(self) -> None:
        self._check(lambda p: p[:, 0] ** 2, create_uniform_space([2], [5]))

    def test_polynomial_2d(self) -> None:
        self._check(
            lambda p: p[:, 0] ** 2 + p[:, 0] * p[:, 1],
            create_uniform_space([2, 2], [4, 4]),
        )

    def test_vector_2d(self) -> None:
        self._check(
            lambda p: np.stack([p[:, 0], p[:, 1]], axis=-1),
            create_uniform_space([2, 2], [4, 4]),
        )


# ---------------------------------------------------------------------------
# Multi-rank equivalence: distributed == serial (simulated in one process)
# ---------------------------------------------------------------------------


class TestMultiRankEquivalence:
    """Distributed QI across N simulated ranks must reproduce the serial result."""

    @pytest.mark.parametrize("n_parts", [2, 4])
    def test_scalar_polynomial(self, n_parts: int) -> None:
        func = lambda p: p[:, 0] ** 2 + p[:, 1] ** 2  # noqa: E731
        space = create_uniform_space([2, 2], [4, 4])
        results = _simulate_distributed_qi(func, space, n_parts)
        serial = quasi_interpolate_bspline(func, space)
        for dfn in results:
            np.testing.assert_allclose(
                dfn.global_function.control_points,
                serial.control_points,
                atol=1e-12,
            )

    @pytest.mark.parametrize("n_parts", [2, 4])
    def test_vector_function(self, n_parts: int) -> None:
        func = lambda p: np.stack([p[:, 0], 1.0 - p[:, 1]], axis=-1)  # noqa: E731
        space = create_uniform_space([2, 2], [4, 4])
        results = _simulate_distributed_qi(func, space, n_parts)
        serial = quasi_interpolate_bspline(func, space)
        for dfn in results:
            np.testing.assert_allclose(
                dfn.global_function.control_points,
                serial.control_points,
                atol=1e-12,
            )

    @pytest.mark.parametrize("n_parts", [2, 4])
    def test_local_evaluates_correctly_on_owned_cells(self, n_parts: int) -> None:
        """Each rank's local function reproduces the serial QI at owned cell midpoints."""
        func = lambda p: np.sin(np.pi * p[:, 0]) * np.cos(np.pi * p[:, 1])  # noqa: E731
        space = create_uniform_space([3, 3], [8, 8])
        part = partition_grid(tensor_product_grid(space), n_parts)
        serial = quasi_interpolate_bspline(func, space)

        contributions: list[tuple[Any, Any]] = []
        for rank in range(n_parts):
            ds = DistributedSpace(space, part, _FakeComm(rank, n_parts))
            local = ds.local
            if local is not None:
                assert isinstance(local.space, BsplineSpace)
                local_bspline = quasi_interpolate_bspline(func, local.space)
                cp = np.asarray(local_bspline.control_points, dtype=np.float64)
                cp_flat = cp.reshape(local.space.num_total_basis, -1)
                owned_mask = local.owned_dof_mask
                contributions.append((local.local_to_global_dof[owned_mask], cp_flat[owned_mask]))
            else:
                contributions.append(
                    (np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float64))
                )

        for rank in range(n_parts):
            comm = _FakeComm(rank, n_parts, allgather_results=contributions)
            ds = DistributedSpace(space, part, comm)
            dfn = quasi_interpolate_bspline_distributed(func, ds)
            local = ds.local
            if local is None:
                continue
            assert dfn.local is not None
            assert isinstance(dfn.local.space, BsplineSpace)
            grid = tensor_product_grid(dfn.local.space)
            for lc in np.flatnonzero(local.owned_cell_mask):
                lo, hi = grid.cell_bounds(int(lc))
                mid = (0.5 * (lo + hi))[None]
                np.testing.assert_allclose(
                    dfn.local.evaluate(mid),
                    serial.evaluate(mid),
                    atol=1e-10,
                )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_thb_space_raises_type_error() -> None:
    """TypeError when global_space is THBSplineSpace (not BsplineSpace)."""
    thb = create_thb_space(create_uniform_space([2, 2], [4, 4]))
    part = partition_grid(thb.grid, 1)
    ds = DistributedSpace(thb, part, _FakeComm(0, 1))
    with pytest.raises(TypeError, match="BsplineSpace"):
        quasi_interpolate_bspline_distributed(lambda p: p[:, 0], ds)


def test_unknown_kind_raises() -> None:
    """ValueError for unsupported kind."""
    space = create_uniform_space([2], [4])
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    with pytest.raises(ValueError, match="Unknown kind"):
        quasi_interpolate_bspline_distributed(
            lambda p: p[:, 0],
            ds,
            kind="nope",  # type: ignore[arg-type]
        )


def test_func_shape_error_propagates() -> None:
    """Shape errors from func propagate from the underlying serial QI."""
    space = create_uniform_space([2], [4])
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    with pytest.raises(ValueError):
        quasi_interpolate_bspline_distributed(lambda p: np.zeros(p.shape[0] + 1), ds)


# ---------------------------------------------------------------------------
# dtype propagation
# ---------------------------------------------------------------------------


def test_float32_space_preserves_dtype() -> None:
    """Global control points retain float32 dtype when the space uses float32."""
    space = create_uniform_space([2], [4], dtype=np.float32)
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    dfn = quasi_interpolate_bspline_distributed(lambda p: p[:, 0] ** 2, ds)
    assert dfn.global_function.control_points.dtype == np.float32


@pytest.mark.parametrize("n_parts", [2, 4])
def test_float32_multi_rank_matches_serial(n_parts: int) -> None:
    """float32 distributed QI matches the serial result within float32 tolerance."""
    space = create_uniform_space([2, 2], [4, 4], dtype=np.float32)
    func = lambda p: (p[:, 0] ** 2 + p[:, 1]).astype(np.float32)  # noqa: E731
    results = _simulate_distributed_qi(func, space, n_parts)
    serial = quasi_interpolate_bspline(func, space)
    for dfn in results:
        assert dfn.global_function.control_points.dtype == np.float32
        np.testing.assert_allclose(
            dfn.global_function.control_points.astype(np.float64),
            serial.control_points.astype(np.float64),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# 1D multi-rank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_parts", [2, 4])
def test_1d_multi_rank_matches_serial(n_parts: int) -> None:
    """1D distributed QI matches the serial result across multiple ranks."""
    func = lambda p: np.sin(np.pi * p[:, 0])  # noqa: E731
    space = create_uniform_space([3], [8])
    results = _simulate_distributed_qi(func, space, n_parts)
    serial = quasi_interpolate_bspline(func, space)
    for dfn in results:
        np.testing.assert_allclose(
            dfn.global_function.control_points,
            serial.control_points,
            atol=1e-10,
        )
