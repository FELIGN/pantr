"""Tests for :func:`pantr.mpi.quasi_interpolate_thb_spline_distributed`.

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
    THBSpline,
    THBSplineSpace,
    create_thb_space,
    create_uniform_space,
    quasi_interpolate_thb_spline,
)
from pantr.bspline._bspline_quasi_interpolation import QIKind
from pantr.grid import Partition, partition_grid, tensor_product_grid
from pantr.mpi import (
    DistributedFunction,
    DistributedSpace,
    quasi_interpolate_thb_spline_distributed,
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


def _refined_thb_space() -> THBSplineSpace:
    """Return a two-level THB space (lower-left quadrant refined)."""
    thb = create_thb_space(create_uniform_space([2, 2], [8, 8]))
    return thb.refine_region(0, [0, 0], [4, 4])


def _simulate_distributed_qi(
    func: Any,
    space: THBSplineSpace,
    n_parts: int,
    *,
    kind: QIKind = "llm",
) -> list[DistributedFunction]:
    """Simulate quasi_interpolate_thb_spline_distributed across n_parts ranks in one process.

    Runs the QI for each rank sequentially, collecting per-rank contributions
    first, then replays the allgather so each rank assembles the correct global field.
    """
    part = partition_grid(space.grid, n_parts)

    # Pass 1: collect what each rank would contribute to the allgather.
    contributions: list[tuple[Any, Any]] = []
    for rank in range(n_parts):
        ds = DistributedSpace(space, part, _FakeComm(rank, n_parts))
        local = ds.local
        if local is not None:
            assert isinstance(local.space, THBSplineSpace)
            local_thb = quasi_interpolate_thb_spline(func, local.space, kind=kind)
            coeffs = np.asarray(local_thb.control_points, dtype=np.float64)
            owned_mask = local.owned_dof_mask
            owned_global_dofs = local.local_to_global_dof[owned_mask]
            owned_coeffs = coeffs[owned_mask]
        else:
            owned_global_dofs = np.empty(0, dtype=np.int64)
            owned_coeffs = np.empty(0, dtype=np.float64)
        contributions.append((owned_global_dofs, owned_coeffs))

    # Pass 2: run the distributed QI for each rank with the pre-known allgather.
    results: list[DistributedFunction] = []
    for rank in range(n_parts):
        comm = _FakeComm(rank, n_parts, allgather_results=contributions)
        ds = DistributedSpace(space, part, comm)
        dfn = quasi_interpolate_thb_spline_distributed(func, ds, kind=kind)
        results.append(dfn)
    return results


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_surface() -> None:
    """quasi_interpolate_thb_spline_distributed is exported from pantr.mpi."""
    assert callable(_pantr_mpi.quasi_interpolate_thb_spline_distributed)
    assert "quasi_interpolate_thb_spline_distributed" in _pantr_mpi.__all__


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_returns_distributed_function() -> None:
    """Returns a DistributedFunction on the given distributed space."""
    space = _refined_thb_space()
    part = partition_grid(space.grid, 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    dfn = quasi_interpolate_thb_spline_distributed(lambda p: p[:, 0] ** 2, ds)
    assert isinstance(dfn, DistributedFunction)
    assert dfn.distributed_space is ds
    assert dfn.global_function.space is space


def test_local_is_thb_spline_on_local_space() -> None:
    """Local is a THBSpline on the rank's windowed space when it owns cells."""
    space = _refined_thb_space()
    part = partition_grid(space.grid, 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    dfn = quasi_interpolate_thb_spline_distributed(lambda p: p[:, 0] + p[:, 1], ds)
    assert dfn.local is not None
    assert isinstance(dfn.local, THBSpline)
    assert dfn.local.space is ds.local.space  # type: ignore[union-attr]


def test_empty_rank_has_none_local() -> None:
    """A rank owning no cells has local=None; still participates and returns a valid result."""
    space = _refined_thb_space()
    n_cells = space.grid.num_cells
    # Rank 0 owns everything; rank 1 is empty.
    part = Partition(np.zeros(n_cells, dtype=np.int64), 2)

    # Rank 1 (empty) contributes nothing; allgather from rank 0 provides the full field.
    ds0 = DistributedSpace(space, part, _FakeComm(0, 2))
    local0 = ds0.local
    assert local0 is not None
    assert isinstance(local0.space, THBSplineSpace)
    local_thb = quasi_interpolate_thb_spline(lambda p: p[:, 0], local0.space)
    coeffs = np.asarray(local_thb.control_points, dtype=np.float64)
    owned_mask = local0.owned_dof_mask
    contrib0 = (local0.local_to_global_dof[owned_mask], coeffs[owned_mask])
    contrib1 = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))
    allgather_data = [contrib0, contrib1]

    serial = quasi_interpolate_thb_spline(lambda p: p[:, 0], space)
    for rank in range(2):
        comm = _FakeComm(rank, 2, allgather_results=allgather_data)
        ds = DistributedSpace(space, part, comm)
        dfn = quasi_interpolate_thb_spline_distributed(lambda p: p[:, 0], ds)
        # The empty rank contributes nothing, yet both ranks assemble the full field.
        np.testing.assert_allclose(
            dfn.global_function.control_points, serial.control_points, atol=1e-12
        )
        if rank == 0:
            assert dfn.local is not None
        else:
            assert dfn.local is None


# ---------------------------------------------------------------------------
# Scalar and vector functions
# ---------------------------------------------------------------------------


def test_scalar_function() -> None:
    """Scalar func → scalar THBSpline (1-D control points), matching the serial kind."""
    space = _refined_thb_space()
    part = partition_grid(space.grid, 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    dfn = quasi_interpolate_thb_spline_distributed(lambda p: p[:, 0] ** 2, ds)
    assert dfn.global_function.control_points.ndim == 1
    assert dfn.global_function.control_points.shape == (space.num_total_basis,)


def test_vector_function() -> None:
    """Vector func (rank=2) → control points have shape (num_total_basis, 2)."""
    space = _refined_thb_space()
    part = partition_grid(space.grid, 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    dfn = quasi_interpolate_thb_spline_distributed(
        lambda p: np.stack([p[:, 0], p[:, 0] ** 2], axis=-1), ds
    )
    assert dfn.global_function.control_points.shape[-1] == 2


# ---------------------------------------------------------------------------
# Single-rank equivalence: distributed == serial
# ---------------------------------------------------------------------------


class TestSingleRankEquivalence:
    """With n_parts=1 the distributed result must equal the serial result exactly."""

    def _check(self, func: Any, space: THBSplineSpace) -> None:
        part = partition_grid(space.grid, 1)
        ds = DistributedSpace(space, part, _FakeComm(0, 1))
        dfn = quasi_interpolate_thb_spline_distributed(func, ds)
        serial = quasi_interpolate_thb_spline(func, space)
        np.testing.assert_array_equal(
            dfn.global_function.control_points,
            serial.control_points,
        )

    def test_polynomial(self) -> None:
        self._check(lambda p: p[:, 0] ** 2 + p[:, 0] * p[:, 1], _refined_thb_space())

    def test_vector(self) -> None:
        self._check(
            lambda p: np.stack([p[:, 0], p[:, 1]], axis=-1),
            _refined_thb_space(),
        )


# ---------------------------------------------------------------------------
# Multi-rank equivalence: distributed == serial (simulated in one process)
# ---------------------------------------------------------------------------


class TestMultiRankEquivalence:
    """Distributed QI across N simulated ranks must reproduce the serial result."""

    @pytest.mark.parametrize("n_parts", [2, 4])
    def test_scalar_polynomial(self, n_parts: int) -> None:
        func = lambda p: p[:, 0] ** 2 + p[:, 1] ** 2  # noqa: E731
        space = _refined_thb_space()
        results = _simulate_distributed_qi(func, space, n_parts)
        serial = quasi_interpolate_thb_spline(func, space)
        for dfn in results:
            np.testing.assert_allclose(
                dfn.global_function.control_points,
                serial.control_points,
                atol=1e-12,
            )

    @pytest.mark.parametrize("n_parts", [2, 4])
    def test_vector_function(self, n_parts: int) -> None:
        func = lambda p: np.stack([p[:, 0], 1.0 - p[:, 1]], axis=-1)  # noqa: E731
        space = _refined_thb_space()
        results = _simulate_distributed_qi(func, space, n_parts)
        serial = quasi_interpolate_thb_spline(func, space)
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
        space = _refined_thb_space()
        serial = quasi_interpolate_thb_spline(func, space)
        results = _simulate_distributed_qi(func, space, n_parts)

        for dfn in results:
            local = dfn.distributed_space.local
            if local is None:
                continue
            assert dfn.local is not None
            assert isinstance(dfn.local, THBSpline)
            assert isinstance(local.space, THBSplineSpace)
            grid = local.space.grid
            for lc in np.flatnonzero(local.owned_cell_mask):
                lo, hi = grid.cell_bounds(int(lc))
                mid = (0.5 * (lo + hi))[None]
                np.testing.assert_allclose(
                    dfn.local.evaluate(mid),
                    serial.evaluate(mid),
                    atol=1e-10,
                )

    @pytest.mark.parametrize("n_parts", [2, 4])
    def test_1d_multi_rank_matches_serial(self, n_parts: int) -> None:
        """1D two-level THB QI matches the serial result across multiple ranks."""
        func = lambda p: np.sin(np.pi * p[:, 0])  # noqa: E731
        space = create_thb_space(create_uniform_space([3], [8])).refine_region(0, [0], [4])
        results = _simulate_distributed_qi(func, space, n_parts)
        serial = quasi_interpolate_thb_spline(func, space)
        for dfn in results:
            np.testing.assert_allclose(
                dfn.global_function.control_points,
                serial.control_points,
                atol=1e-10,
            )

    @pytest.mark.parametrize("n_parts", [2, 4])
    def test_three_level_hierarchy_matches_serial(self, n_parts: int) -> None:
        """A 3-level THB space distributes correctly.

        Exercises the per-level leaf-cell selection (issue #240, I2): ranks straddling
        a level-2 refinement must window in active DOFs from all three levels, and the
        Speleers-Manni functional's level-``l`` leaf cell must still resolve inside each
        rank's windowed parametric sub-domain.
        """
        func = lambda p: np.sin(np.pi * p[:, 0]) * np.cos(np.pi * p[:, 1])  # noqa: E731
        space = (
            create_thb_space(create_uniform_space([2, 2], [8, 8]))
            .refine_region(0, [0, 0], [4, 4])
            .refine_region(1, [0, 0], [4, 4])
        )
        assert space.num_levels == 3
        results = _simulate_distributed_qi(func, space, n_parts)
        serial = quasi_interpolate_thb_spline(func, space)
        for dfn in results:
            np.testing.assert_allclose(
                dfn.global_function.control_points,
                serial.control_points,
                atol=1e-10,
            )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_bspline_space_raises_type_error() -> None:
    """TypeError when global_space is BsplineSpace (not THBSplineSpace)."""
    space = create_uniform_space([2, 2], [4, 4])
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    with pytest.raises(TypeError, match="THBSplineSpace"):
        quasi_interpolate_thb_spline_distributed(lambda p: p[:, 0], ds)


def test_unknown_kind_raises() -> None:
    """ValueError for unsupported kind."""
    space = _refined_thb_space()
    part = partition_grid(space.grid, 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    with pytest.raises(ValueError, match="Unknown kind"):
        quasi_interpolate_thb_spline_distributed(
            lambda p: p[:, 0],
            ds,
            kind="nope",  # type: ignore[arg-type]
        )


def test_func_shape_error_propagates() -> None:
    """Shape errors from func propagate from the underlying serial QI."""
    space = _refined_thb_space()
    part = partition_grid(space.grid, 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    with pytest.raises(ValueError):
        quasi_interpolate_thb_spline_distributed(lambda p: np.zeros(p.shape[0] + 1), ds)


# ---------------------------------------------------------------------------
# dtype: THBSpline coefficients are always float64 (matches serial)
# ---------------------------------------------------------------------------


def test_result_is_float64_for_float32_space() -> None:
    """A float32 THB space still yields float64 coefficients, exactly as the serial QI."""
    thb = create_thb_space(create_uniform_space([2, 2], [8, 8], dtype=np.float32))
    space = thb.refine_region(0, [0, 0], [4, 4])
    part = partition_grid(space.grid, 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    dfn = quasi_interpolate_thb_spline_distributed(lambda p: p[:, 0] ** 2, ds)
    serial = quasi_interpolate_thb_spline(lambda p: p[:, 0] ** 2, space)
    assert dfn.global_function.control_points.dtype == np.float64
    assert serial.control_points.dtype == np.float64
    np.testing.assert_array_equal(dfn.global_function.control_points, serial.control_points)
