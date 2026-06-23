"""Tests for :func:`pantr.mpi.l2_project_bspline_distributed`.

MPI is not available in the test environment, so the communicator is duck-typed by
``_FakeComm``, which adds ``allreduce`` support (needed by this function) on top of the
basic rank/size interface used elsewhere.  Multi-rank runs are simulated in one process:
each rank's owned-cell load is assembled, then the ``allreduce`` is replayed so every
rank sees the summed global load.

The real-MPI equivalence test (collective over ``MPI.COMM_WORLD``) lives in
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
    l2_project_bspline,
)
from pantr.grid import Partition, partition_grid, tensor_product_grid
from pantr.mpi import (
    DistributedFunction,
    DistributedSpace,
    l2_project_bspline_distributed,
)

# ---------------------------------------------------------------------------
# Fake communicator with allreduce
# ---------------------------------------------------------------------------


class _FakeComm:
    """Minimal stand-in for an mpi4py communicator.

    Supports ``rank``, ``size``, and a single-rank ``allreduce`` (returns its argument).
    For multi-rank simulations pass ``allreduce_result`` to set the summed value returned
    regardless of local data.
    """

    def __init__(
        self,
        rank: int,
        size: int,
        allreduce_result: Any = None,
    ) -> None:
        self._rank = rank
        self._size = size
        self._allreduce_result = allreduce_result

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def size(self) -> int:
        return self._size

    def allreduce(self, data: Any) -> Any:
        if self._allreduce_result is not None:
            return self._allreduce_result
        assert self._size == 1, (
            "Multi-rank allreduce requires pre-set allreduce_result; "
            "use _simulate_distributed_l2() for multi-rank tests."
        )
        return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grid(lat: Any) -> tuple[np.ndarray, ...]:
    """Return the C-order, flattened meshgrid of a PointsLattice's per-direction nodes.

    Mirrors the serial ``func(lattice)`` convention: each returned array is the flat
    ``(n_total,)`` coordinate vector for one parametric direction, so an expression like
    ``_grid(lat)[0] ** 2`` yields the expected ``(n_total,)`` output.
    """
    return tuple(g.reshape(-1) for g in np.meshgrid(*list(lat.pts_per_dir), indexing="ij"))


def _simulate_distributed_l2(
    func: Any,
    space: BsplineSpace,
    n_parts: int,
    **kwargs: Any,
) -> list[DistributedFunction]:
    """Simulate l2_project_bspline_distributed across n_parts ranks in one process.

    Pass 1 runs each rank up to its local load contribution and sums them (the
    ``allreduce``).  Pass 2 runs the distributed projection for every rank with the
    pre-summed global load, so each rank assembles the identical global field.
    """
    from pantr.bspline._bspline_interpolate import (  # noqa: PLC0415
        _build_l2_mass_and_quad,
        _evaluate_func_on_lattice,
    )
    from pantr.mpi._l2 import _assemble_owned_load, _owned_quad_cell_mask  # noqa: PLC0415
    from pantr.quad import PointsLattice  # noqa: PLC0415

    part = partition_grid(tensor_product_grid(space), n_parts)
    n_quad = kwargs.get("n_quad")
    quadrature = kwargs.get("quadrature", "gauss-legendre")
    boundary_interpolation = kwargs.get("boundary_interpolation", False)

    _, q_nodes, q_weights, _, n_quads = _build_l2_mass_and_quad(
        space, n_quad, quadrature, boundary_interpolation
    )
    lattice = PointsLattice(q_nodes)
    quad_grid_shape = tuple(a.shape[0] for a in q_nodes)
    components, out_dtype = _evaluate_func_on_lattice(func, lattice, quad_grid_shape)

    total: np.ndarray | None = None
    for rank in range(n_parts):
        mask = _owned_quad_cell_mask(space, part.cell_owner, rank, n_quads, quad_grid_shape)
        loads = _assemble_owned_load(space, components, mask, out_dtype, q_nodes, q_weights)
        local_load = np.stack(loads, axis=-1).astype(out_dtype, copy=False)
        total = local_load if total is None else total + local_load

    results: list[DistributedFunction] = []
    for rank in range(n_parts):
        comm = _FakeComm(rank, n_parts, allreduce_result=total)
        ds = DistributedSpace(space, part, comm)
        dfn = l2_project_bspline_distributed(func, ds, **kwargs)
        results.append(dfn)
    return results


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_surface() -> None:
    """l2_project_bspline_distributed is exported from pantr.mpi."""
    assert callable(_pantr_mpi.l2_project_bspline_distributed)
    assert "l2_project_bspline_distributed" in _pantr_mpi.__all__


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_returns_distributed_function() -> None:
    """Returns a DistributedFunction on the given distributed space."""
    space = create_uniform_space([2, 2], [4, 4])
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    dfn = l2_project_bspline_distributed(lambda lat: _grid(lat)[0] ** 2, ds)
    assert isinstance(dfn, DistributedFunction)
    assert dfn.distributed_space is ds
    assert dfn.global_function.space is space


def test_local_is_bspline_on_local_space() -> None:
    """Local is a Bspline on the rank's windowed space when it owns cells."""
    space = create_uniform_space([2, 2], [4, 4])
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    dfn = l2_project_bspline_distributed(lambda lat: _grid(lat)[0] + _grid(lat)[1], ds)
    assert dfn.local is not None
    assert isinstance(dfn.local, Bspline)
    assert dfn.local.space is ds.local.space  # type: ignore[union-attr]


def test_empty_rank_has_none_local() -> None:
    """A rank owning no cells has local=None; still participates and returns a result."""
    space = create_uniform_space([2, 2], [4, 4])
    n_cells = space.num_total_intervals
    # Rank 0 owns everything; rank 1 is empty.
    part = Partition(np.zeros(n_cells, dtype=np.int64), 2)
    func = lambda lat: _grid(lat)[0]  # noqa: E731

    # Build the global load (only rank 0 contributes) and replay the allreduce.
    from pantr.bspline._bspline_interpolate import (  # noqa: PLC0415
        _build_l2_mass_and_quad,
        _evaluate_func_on_lattice,
    )
    from pantr.mpi._l2 import _assemble_owned_load, _owned_quad_cell_mask  # noqa: PLC0415
    from pantr.quad import PointsLattice  # noqa: PLC0415

    _, q_nodes, q_weights, _, n_quads = _build_l2_mass_and_quad(
        space, None, "gauss-legendre", False
    )
    lattice = PointsLattice(q_nodes)
    quad_grid_shape = tuple(a.shape[0] for a in q_nodes)
    components, out_dtype = _evaluate_func_on_lattice(func, lattice, quad_grid_shape)
    mask0 = _owned_quad_cell_mask(space, part.cell_owner, 0, n_quads, quad_grid_shape)
    loads0 = _assemble_owned_load(space, components, mask0, out_dtype, q_nodes, q_weights)
    total = np.stack(loads0, axis=-1).astype(out_dtype, copy=False)

    for rank in range(2):
        comm = _FakeComm(rank, 2, allreduce_result=total)
        ds = DistributedSpace(space, part, comm)
        dfn = l2_project_bspline_distributed(func, ds)
        if rank == 0:
            assert dfn.local is not None
        else:
            assert dfn.local is None


# ---------------------------------------------------------------------------
# Scalar and vector functions
# ---------------------------------------------------------------------------


def test_scalar_function() -> None:
    """Scalar func -> control points have shape (*num_basis, 1)."""
    space = create_uniform_space([2], [4])
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    dfn = l2_project_bspline_distributed(lambda lat: _grid(lat)[0] ** 2, ds)
    assert dfn.global_function.control_points.shape[-1] == 1


def test_vector_function() -> None:
    """Vector func (rank=2) -> control points have shape (*num_basis, 2)."""
    space = create_uniform_space([2], [4])
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    func = lambda lat: np.stack(  # noqa: E731
        [_grid(lat)[0].reshape(-1), (_grid(lat)[0] ** 2).reshape(-1)], axis=-1
    )
    dfn = l2_project_bspline_distributed(func, ds)
    assert dfn.global_function.control_points.shape[-1] == 2


# ---------------------------------------------------------------------------
# Single-rank equivalence: distributed == serial
# ---------------------------------------------------------------------------


class TestSingleRankEquivalence:
    """With n_parts=1 the distributed result must equal the serial result exactly."""

    def _check(self, func: Any, space: BsplineSpace, **kwargs: Any) -> None:
        part = partition_grid(tensor_product_grid(space), 1)
        ds = DistributedSpace(space, part, _FakeComm(0, 1))
        dfn = l2_project_bspline_distributed(func, ds, **kwargs)
        serial = l2_project_bspline(func, space, **kwargs)
        np.testing.assert_allclose(
            dfn.global_function.control_points,
            serial.control_points,
            atol=1e-13,
        )

    def test_polynomial_1d(self) -> None:
        self._check(lambda lat: _grid(lat)[0] ** 2, create_uniform_space([2], [5]))

    def test_polynomial_2d(self) -> None:
        self._check(
            lambda lat: _grid(lat)[0] ** 2 + _grid(lat)[0] * _grid(lat)[1],
            create_uniform_space([2, 2], [4, 4]),
        )

    def test_boundary_interpolation(self) -> None:
        self._check(
            lambda lat: np.sin(_grid(lat)[0]) + _grid(lat)[1] ** 2,
            create_uniform_space([2, 2], [4, 4]),
            boundary_interpolation=True,
        )


# ---------------------------------------------------------------------------
# Multi-rank equivalence: distributed == serial (simulated in one process)
# ---------------------------------------------------------------------------


class TestMultiRankEquivalence:
    """Distributed L2 across N simulated ranks must reproduce the serial result."""

    @pytest.mark.parametrize("n_parts", [2, 4])
    def test_scalar_polynomial(self, n_parts: int) -> None:
        func = lambda lat: np.sin(_grid(lat)[0]) + 0.5 * _grid(lat)[1] ** 2  # noqa: E731
        space = create_uniform_space([2, 2], [4, 4])
        results = _simulate_distributed_l2(func, space, n_parts)
        serial = l2_project_bspline(func, space)
        for dfn in results:
            np.testing.assert_allclose(
                dfn.global_function.control_points,
                serial.control_points,
                atol=1e-12,
            )

    @pytest.mark.parametrize("n_parts", [2, 4])
    def test_vector_function(self, n_parts: int) -> None:
        func = lambda lat: np.stack(  # noqa: E731
            [_grid(lat)[0].reshape(-1), (1.0 - _grid(lat)[1]).reshape(-1)], axis=-1
        )
        space = create_uniform_space([2, 2], [4, 4])
        results = _simulate_distributed_l2(func, space, n_parts)
        serial = l2_project_bspline(func, space)
        for dfn in results:
            np.testing.assert_allclose(
                dfn.global_function.control_points,
                serial.control_points,
                atol=1e-12,
            )

    @pytest.mark.parametrize("n_parts", [2, 4])
    def test_boundary_interpolation(self, n_parts: int) -> None:
        func = lambda lat: np.cos(_grid(lat)[0]) * _grid(lat)[1]  # noqa: E731
        space = create_uniform_space([3, 3], [6, 6])
        for bi in (True, [(True, False), (False, True)]):
            results = _simulate_distributed_l2(func, space, n_parts, boundary_interpolation=bi)
            serial = l2_project_bspline(func, space, boundary_interpolation=bi)
            for dfn in results:
                np.testing.assert_allclose(
                    dfn.global_function.control_points,
                    serial.control_points,
                    atol=1e-12,
                )

    @pytest.mark.parametrize("n_parts", [2, 3, 4])
    def test_partition_independent(self, n_parts: int) -> None:
        """Different partitions all give the serial output (so identical to each other)."""
        func = lambda lat: np.exp(-_grid(lat)[0]) + _grid(lat)[1]  # noqa: E731
        space = create_uniform_space([2, 2], [6, 6])
        serial = l2_project_bspline(func, space)
        results = _simulate_distributed_l2(func, space, n_parts)
        for dfn in results:
            np.testing.assert_allclose(
                dfn.global_function.control_points, serial.control_points, atol=1e-12
            )

    @pytest.mark.parametrize("n_parts", [2, 4])
    def test_local_evaluates_correctly_on_owned_cells(self, n_parts: int) -> None:
        """Each rank's local function reproduces the serial L2 at owned cell midpoints."""
        func = lambda lat: np.sin(np.pi * _grid(lat)[0]) * np.cos(np.pi * _grid(lat)[1])  # noqa: E731
        space = create_uniform_space([3, 3], [8, 8])
        serial = l2_project_bspline(func, space)
        results = _simulate_distributed_l2(func, space, n_parts)
        part = partition_grid(tensor_product_grid(space), n_parts)
        for rank, dfn in enumerate(results):
            ds = DistributedSpace(space, part, _FakeComm(rank, n_parts))
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
        l2_project_bspline_distributed(lambda lat: _grid(lat)[0], ds)


def test_bad_n_quad_raises() -> None:
    """ValueError when n_quad length mismatches the space dimension."""
    space = create_uniform_space([2, 2], [4, 4])
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    with pytest.raises(ValueError, match="n_quad"):
        l2_project_bspline_distributed(lambda lat: _grid(lat)[0], ds, n_quad=[2, 2, 2])


# ---------------------------------------------------------------------------
# dtype propagation
# ---------------------------------------------------------------------------


def test_float32_space_preserves_dtype() -> None:
    """Global control points retain float32 dtype when the space uses float32."""
    space = create_uniform_space([2], [4], dtype=np.float32)
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    dfn = l2_project_bspline_distributed(lambda lat: (_grid(lat)[0] ** 2).astype(np.float32), ds)
    assert dfn.global_function.control_points.dtype == np.float32


@pytest.mark.parametrize("n_parts", [2, 4])
def test_float32_multi_rank_matches_serial(n_parts: int) -> None:
    """float32 distributed L2 matches the serial result within float32 tolerance."""
    space = create_uniform_space([2, 2], [4, 4], dtype=np.float32)
    func = lambda lat: (_grid(lat)[0] ** 2 + _grid(lat)[1]).astype(np.float32)  # noqa: E731
    results = _simulate_distributed_l2(func, space, n_parts)
    serial = l2_project_bspline(func, space)
    for dfn in results:
        assert dfn.global_function.control_points.dtype == np.float32
        np.testing.assert_allclose(
            dfn.global_function.control_points.astype(np.float64),
            serial.control_points.astype(np.float64),
            atol=1e-4,
        )


# ---------------------------------------------------------------------------
# 1D multi-rank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_parts", [2, 4])
def test_1d_multi_rank_matches_serial(n_parts: int) -> None:
    """1D distributed L2 matches the serial result across multiple ranks."""
    func = lambda lat: np.sin(np.pi * _grid(lat)[0])  # noqa: E731
    space = create_uniform_space([3], [8])
    results = _simulate_distributed_l2(func, space, n_parts)
    serial = l2_project_bspline(func, space)
    for dfn in results:
        np.testing.assert_allclose(
            dfn.global_function.control_points,
            serial.control_points,
            atol=1e-12,
        )
