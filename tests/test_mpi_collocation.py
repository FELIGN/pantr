"""Tests for distributed B-spline collocation interpolation / fitting.

Covers :func:`pantr.mpi.interpolate_bspline_distributed` and
:func:`pantr.mpi.fit_bspline_distributed`.  MPI is not available in the default test
environment, so the communicator is duck-typed by ``_FakeComm``, which adds the
``allgather`` collective these functions need.

Multi-rank runs are simulated in a single process: each rank's contribution to the
``allgather`` is collected first, then the distributed routine is replayed per rank with
the full gather pre-set.  The matching real-MPI smoke tests live in
``tests/mpi/test_distributed_mpi.py`` and run under ``mpiexec``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import pantr.mpi as _pantr_mpi
from pantr.bspline import (
    Bspline,
    BsplineSpace,
    create_greville_lattice,
    create_thb_space,
    create_uniform_space,
    fit_bspline,
    interpolate_bspline,
)
from pantr.grid import partition_grid, tensor_product_grid
from pantr.mpi import (
    DistributedFunction,
    DistributedSpace,
    fit_bspline_distributed,
    interpolate_bspline_distributed,
)
from pantr.mpi._collocation import _block_bounds

# ---------------------------------------------------------------------------
# Fake communicator with allgather
# ---------------------------------------------------------------------------


class _FakeComm:
    """Minimal stand-in for an mpi4py communicator.

    Supports ``rank``, ``size``, and ``allgather``.  For a single rank, ``allgather``
    returns ``[data]``.  For multi-rank simulations pass ``allgather_results`` to set the
    value returned regardless of local data.
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
        assert self._size == 1, "multi-rank allgather requires pre-set allgather_results"
        return [data]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_lattice_func(flat_func: Any) -> Any:
    """Adapt a flat-``(M, dim)`` func to the serial PointsLattice calling convention.

    Serial :func:`interpolate_bspline` / :func:`fit_bspline` call ``func(lattice)``; the
    distributed routines call ``func(points)`` on a flat array.  This wrapper lets one
    flat func drive both so equivalence is checked on identical values.
    """
    return lambda lattice: flat_func(lattice.get_all_points(order="C"))


def _simulate_interpolate(
    func: Any,
    space: BsplineSpace,
    n_parts: int,
    **kwargs: Any,
) -> list[DistributedFunction]:
    """Simulate interpolate_bspline_distributed across n_parts ranks in one process.

    Each rank evaluates only its block of the flattened grid; the pre-collected blocks
    play the role of the allgather, so every rank assembles the identical full field.
    """
    part = partition_grid(tensor_product_grid(space), n_parts)
    pts = create_greville_lattice(space).get_all_points(order="C")
    n_total = pts.shape[0]

    blocks: list[Any] = []
    for rank in range(n_parts):
        start, stop = _block_bounds(n_total, n_parts, rank)
        blocks.append(np.asarray(func(pts[start:stop]), dtype=np.float64))

    results: list[DistributedFunction] = []
    for rank in range(n_parts):
        ds = DistributedSpace(space, part, _FakeComm(rank, n_parts, allgather_results=blocks))
        results.append(interpolate_bspline_distributed(func, ds, **kwargs))
    return results


def _simulate_fit_distributed_values(
    func: Any,
    space: BsplineSpace,
    n_parts: int,
    **kwargs: Any,
) -> list[DistributedFunction]:
    """Simulate fit_bspline_distributed with values_distributed=True across n_parts ranks."""
    part = partition_grid(tensor_product_grid(space), n_parts)
    lattice = create_greville_lattice(space)
    pts = lattice.get_all_points(order="C")
    n_total = pts.shape[0]
    full_vals = np.asarray(func(pts), dtype=np.float64)

    blocks: list[Any] = []
    for rank in range(n_parts):
        start, stop = _block_bounds(n_total, n_parts, rank)
        blocks.append(np.ascontiguousarray(full_vals[start:stop]))

    results: list[DistributedFunction] = []
    for rank in range(n_parts):
        ds = DistributedSpace(space, part, _FakeComm(rank, n_parts, allgather_results=blocks))
        results.append(
            fit_bspline_distributed(blocks[rank], lattice, ds, values_distributed=True, **kwargs)
        )
    return results


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_surface() -> None:
    """Both functions are exported from pantr.mpi."""
    assert callable(_pantr_mpi.interpolate_bspline_distributed)
    assert callable(_pantr_mpi.fit_bspline_distributed)
    assert "interpolate_bspline_distributed" in _pantr_mpi.__all__
    assert "fit_bspline_distributed" in _pantr_mpi.__all__


def test_block_bounds_partition() -> None:
    """_block_bounds tiles [0, n_total) into contiguous, non-overlapping blocks."""
    for n_total in (0, 1, 7, 16, 17):
        for n_parts in (1, 2, 3, 4):
            bounds = [_block_bounds(n_total, n_parts, r) for r in range(n_parts)]
            # Contiguous and covering.
            assert bounds[0][0] == 0
            assert bounds[-1][1] == n_total
            for r in range(1, n_parts):
                assert bounds[r][0] == bounds[r - 1][1]
            # Block sizes differ by at most 1.
            sizes = [hi - lo for lo, hi in bounds]
            assert max(sizes) - min(sizes) <= 1


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


def test_interpolate_returns_distributed_function() -> None:
    """Interpolation returns a DistributedFunction on the given distributed space."""
    space = create_uniform_space([2, 2], [4, 4])
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    dfn = interpolate_bspline_distributed(lambda p: p[:, 0] ** 2, ds)
    assert isinstance(dfn, DistributedFunction)
    assert dfn.distributed_space is ds
    assert dfn.global_function.space is space
    assert isinstance(dfn.local, Bspline)


def test_fit_returns_distributed_function() -> None:
    """Fitting returns a DistributedFunction on the given distributed space."""
    space = create_uniform_space([2, 2], [4, 4])
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    lattice = create_greville_lattice(space)
    vals = lattice.get_all_points(order="C")[:, 0] ** 2
    grid_shape = tuple(a.shape[0] for a in lattice.pts_per_dir)
    dfn = fit_bspline_distributed(vals.reshape(grid_shape), lattice, ds)
    assert isinstance(dfn, DistributedFunction)
    assert dfn.global_function.space is space


# ---------------------------------------------------------------------------
# Single-rank equivalence: distributed == serial
# ---------------------------------------------------------------------------


class TestInterpolateSingleRankEquivalence:
    """With n_parts=1 the distributed interpolant must equal the serial one exactly."""

    def _check(self, func: Any, space: BsplineSpace, **kwargs: Any) -> None:
        part = partition_grid(tensor_product_grid(space), 1)
        ds = DistributedSpace(space, part, _FakeComm(0, 1))
        dfn = interpolate_bspline_distributed(func, ds, **kwargs)
        serial = interpolate_bspline(_as_lattice_func(func), space, **kwargs)
        np.testing.assert_array_equal(dfn.global_function.control_points, serial.control_points)

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

    def test_boundary_derivatives(self) -> None:
        self._check(
            lambda p: np.sin(np.pi * p[:, 0]),
            create_uniform_space([3], [6]),
            boundary_derivatives=[(1, 1)],
        )


# ---------------------------------------------------------------------------
# Multi-rank equivalence: distributed == serial (simulated in one process)
# ---------------------------------------------------------------------------


class TestInterpolateMultiRankEquivalence:
    """Distributed interpolation across N simulated ranks must reproduce serial."""

    @pytest.mark.parametrize("n_parts", [1, 2, 3, 4])
    def test_scalar_polynomial(self, n_parts: int) -> None:
        func = lambda p: p[:, 0] ** 2 + p[:, 1] ** 2  # noqa: E731
        space = create_uniform_space([2, 2], [4, 4])
        serial = interpolate_bspline(_as_lattice_func(func), space)
        for dfn in _simulate_interpolate(func, space, n_parts):
            np.testing.assert_allclose(
                dfn.global_function.control_points, serial.control_points, atol=1e-12
            )

    @pytest.mark.parametrize("n_parts", [2, 4])
    def test_vector_function(self, n_parts: int) -> None:
        func = lambda p: np.stack([p[:, 0], 1.0 - p[:, 1]], axis=-1)  # noqa: E731
        space = create_uniform_space([2, 2], [4, 4])
        serial = interpolate_bspline(_as_lattice_func(func), space)
        for dfn in _simulate_interpolate(func, space, n_parts):
            np.testing.assert_allclose(
                dfn.global_function.control_points, serial.control_points, atol=1e-12
            )

    @pytest.mark.parametrize("n_parts", [2, 3])
    def test_1d_transcendental(self, n_parts: int) -> None:
        func = lambda p: np.sin(np.pi * p[:, 0])  # noqa: E731
        space = create_uniform_space([3], [8])
        serial = interpolate_bspline(_as_lattice_func(func), space)
        for dfn in _simulate_interpolate(func, space, n_parts):
            np.testing.assert_allclose(
                dfn.global_function.control_points, serial.control_points, atol=1e-12
            )

    @pytest.mark.parametrize("n_parts", [2, 4])
    def test_different_partitions_identical(self, n_parts: int) -> None:
        """Every rank assembles the identical global field, independent of partition."""
        func = lambda p: np.cos(np.pi * p[:, 0]) * p[:, 1]  # noqa: E731
        space = create_uniform_space([2, 2], [6, 6])
        results = _simulate_interpolate(func, space, n_parts)
        ref = results[0].global_function.control_points
        for dfn in results[1:]:
            np.testing.assert_array_equal(dfn.global_function.control_points, ref)


# ---------------------------------------------------------------------------
# fit: distributed == serial, both replicated and pre-distributed value paths
# ---------------------------------------------------------------------------


class TestFitEquivalence:
    """fit_bspline_distributed matches serial fit_bspline in both value layouts."""

    def _serial(self, func: Any, space: BsplineSpace) -> Bspline:
        lattice = create_greville_lattice(space)
        grid_shape = tuple(a.shape[0] for a in lattice.pts_per_dir)
        vals = np.asarray(func(lattice.get_all_points(order="C")), dtype=np.float64)
        if vals.ndim == 1:
            vals = vals.reshape(grid_shape)
        else:
            vals = vals.reshape(*grid_shape, vals.shape[1])
        return fit_bspline(vals, lattice, space)

    @pytest.mark.parametrize("n_parts", [1, 2, 4])
    def test_replicated_values(self, n_parts: int) -> None:
        """values_distributed=False: every rank holds the full field."""
        func: Any = lambda p: p[:, 0] ** 2 + p[:, 1]  # noqa: E731
        space = create_uniform_space([2, 2], [4, 4])
        serial = self._serial(func, space)

        lattice = create_greville_lattice(space)
        grid_shape = tuple(a.shape[0] for a in lattice.pts_per_dir)
        vals = np.asarray(func(lattice.get_all_points(order="C")), dtype=np.float64)
        vals = vals.reshape(grid_shape)

        part = partition_grid(tensor_product_grid(space), n_parts)
        for rank in range(n_parts):
            ds = DistributedSpace(space, part, _FakeComm(rank, n_parts))
            dfn = fit_bspline_distributed(vals, lattice, ds)
            np.testing.assert_array_equal(dfn.global_function.control_points, serial.control_points)

    @pytest.mark.parametrize("n_parts", [1, 2, 3, 4])
    def test_pre_distributed_values(self, n_parts: int) -> None:
        """values_distributed=True: each rank holds only its flattened block."""
        func = lambda p: p[:, 0] ** 2 + p[:, 1]  # noqa: E731
        space = create_uniform_space([2, 2], [4, 4])
        serial = self._serial(func, space)
        for dfn in _simulate_fit_distributed_values(func, space, n_parts):
            np.testing.assert_allclose(
                dfn.global_function.control_points, serial.control_points, atol=1e-12
            )

    @pytest.mark.parametrize("n_parts", [2, 4])
    def test_pre_distributed_vector_values(self, n_parts: int) -> None:
        """Vector-valued pre-distributed blocks reproduce the serial fit."""
        func = lambda p: np.stack([p[:, 0], 1.0 - p[:, 1]], axis=-1)  # noqa: E731
        space = create_uniform_space([2, 2], [4, 4])
        serial = self._serial(func, space)
        for dfn in _simulate_fit_distributed_values(func, space, n_parts):
            np.testing.assert_allclose(
                dfn.global_function.control_points, serial.control_points, atol=1e-12
            )

    def test_interpolate_matches_fit_distributed(self) -> None:
        """Distributed interpolate == distributed fit on the same Greville values."""
        func = lambda p: np.sin(np.pi * p[:, 0]) * p[:, 1]  # noqa: E731
        space = create_uniform_space([3, 2], [6, 5])
        interp = _simulate_interpolate(func, space, 3)[0]
        fitted = _simulate_fit_distributed_values(func, space, 3)[0]
        np.testing.assert_allclose(
            interp.global_function.control_points,
            fitted.global_function.control_points,
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# dtype propagation
# ---------------------------------------------------------------------------


def test_interpolate_float32_preserves_dtype() -> None:
    """Global control points retain float32 dtype when the space uses float32."""
    space = create_uniform_space([2], [4], dtype=np.float32)
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    dfn = interpolate_bspline_distributed(lambda p: (p[:, 0] ** 2).astype(np.float32), ds)
    assert dfn.global_function.control_points.dtype == np.float32


@pytest.mark.parametrize("n_parts", [2, 4])
def test_interpolate_float32_matches_serial(n_parts: int) -> None:
    """float32 distributed interpolation matches serial within float32 tolerance."""
    space = create_uniform_space([2, 2], [4, 4], dtype=np.float32)
    func = lambda p: (p[:, 0] ** 2 + p[:, 1]).astype(np.float32)  # noqa: E731
    serial = interpolate_bspline(_as_lattice_func(func), space)
    for dfn in _simulate_interpolate(func, space, n_parts):
        assert dfn.global_function.control_points.dtype == np.float32
        np.testing.assert_allclose(
            dfn.global_function.control_points.astype(np.float64),
            serial.control_points.astype(np.float64),
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_interpolate_thb_space_raises_type_error() -> None:
    """TypeError when global_space is a THBSplineSpace."""
    thb = create_thb_space(create_uniform_space([2, 2], [4, 4]))
    part = partition_grid(thb.grid, 1)
    ds = DistributedSpace(thb, part, _FakeComm(0, 1))
    with pytest.raises(TypeError, match="BsplineSpace"):
        interpolate_bspline_distributed(lambda p: p[:, 0], ds)


def test_fit_thb_space_raises_type_error() -> None:
    """TypeError when global_space is a THBSplineSpace (checked before nodes are used)."""
    thb = create_thb_space(create_uniform_space([2, 2], [4, 4]))
    part = partition_grid(thb.grid, 1)
    ds = DistributedSpace(thb, part, _FakeComm(0, 1))
    with pytest.raises(TypeError, match="BsplineSpace"):
        fit_bspline_distributed(np.zeros((5, 5)), [np.linspace(0, 1, 5)], ds)


def test_fit_distributed_scattered_raises() -> None:
    """values_distributed=True with scattered nodes raises ValueError."""
    space = create_uniform_space([2], [4])
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    scattered = np.linspace(0, 1, 10).reshape(-1, 1)
    with pytest.raises(ValueError, match="tensor-product"):
        fit_bspline_distributed(np.zeros(10), scattered, ds, values_distributed=True)


def test_fit_scattered_replicated_falls_back() -> None:
    """Scattered nodes work in the replicated path (serial fallback)."""
    space = create_uniform_space([2], [4])
    part = partition_grid(tensor_product_grid(space), 1)
    ds = DistributedSpace(space, part, _FakeComm(0, 1))
    nodes = np.linspace(0, 1, 20).reshape(-1, 1)
    vals = np.sin(np.pi * nodes[:, 0])
    dfn = fit_bspline_distributed(vals, nodes, ds)
    serial = fit_bspline(vals, nodes, space)
    np.testing.assert_array_equal(dfn.global_function.control_points, serial.control_points)
