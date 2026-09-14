"""Tests for :func:`pantr.mpi.l2_project_bspline_distributed`.

MPI is not available in the test environment, so the communicator is duck-typed by
``_FakeComm``, which adds ``allreduce`` support (needed by this function) on top of the
basic rank/size interface used elsewhere.  Multi-rank runs are simulated in one process
by sharing a single ``_AllreduceHub`` across the per-rank ``_FakeComm`` instances: every
rank's ``comm.allreduce(local_load)`` deposits the *actual* tensor production code passes,
and the hub returns the genuine SUM across all ranks.  The hub never fabricates the
result, so a regression in *what* ``_l2.py`` feeds ``allreduce`` (wrong stacking, wrong
masking) is caught by the equivalence-vs-serial assertions.

The real-MPI equivalence test (collective over ``MPI.COMM_WORLD``) lives in
``tests/mpi/test_distributed_mpi.py`` and runs under ``mpiexec``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

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


class _AllreduceHub:
    """Shared SUM accumulator for a simulated multi-rank ``allreduce``.

    All per-rank :class:`_FakeComm` instances of one simulation share a single hub.  Each
    rank's ``comm.allreduce(local_load)`` deposits the *actual* array production code
    passes; the hub sums every deposited contribution and returns that genuine SUM.  It
    never fabricates the reduced value, so it faithfully exercises what ``_l2.py`` feeds
    ``allreduce``.

    The simulation runs in two passes.  During the *collecting* pass every rank's
    ``allreduce`` deposits its contribution and gets back a (still partial) sum so
    production can finish; once :meth:`freeze` is called the hub stops collecting, and the
    *result* pass returns the complete SUM to every rank without re-depositing.

    Attributes:
        contributions (list[np.ndarray]): The per-rank tensors deposited so far.
        collecting (bool): Whether :meth:`allreduce` should still record deposits.
    """

    def __init__(self) -> None:
        """Initialize an empty, collecting hub."""
        self.contributions: list[np.ndarray] = []
        self.collecting: bool = True

    def allreduce(self, data: np.ndarray) -> np.ndarray:
        """Deposit (while collecting) and return the SUM over deposited contributions.

        Args:
            data (np.ndarray): The tensor a rank passed to ``comm.allreduce``.

        Returns:
            np.ndarray: The element-wise SUM over every contribution deposited so far
            (the complete cross-rank sum once :meth:`freeze` has been called).
        """
        if self.collecting:
            self.contributions.append(np.asarray(data).copy())
        total = self.contributions[0].copy()
        for contrib in self.contributions[1:]:
            total = total + contrib
        return total

    def freeze(self) -> None:
        """Stop collecting so subsequent ``allreduce`` calls return the complete SUM."""
        self.collecting = False


class _FakeComm:
    """Minimal stand-in for an mpi4py communicator.

    Supports ``rank``, ``size``, and a SUM ``allreduce``.  For single-rank use the
    ``allreduce`` returns its (sole) argument.  For multi-rank simulations pass a shared
    :class:`_AllreduceHub`; ``allreduce`` then deposits its argument into the hub and
    returns the genuine SUM over every rank's contribution.

    Attributes:
        hub (_AllreduceHub | None): Shared SUM accumulator for multi-rank simulations.
    """

    def __init__(
        self,
        rank: int,
        size: int,
        hub: _AllreduceHub | None = None,
    ) -> None:
        """Initialize a fake communicator.

        Args:
            rank (int): This rank's id.
            size (int): The communicator size.
            hub (_AllreduceHub | None): Shared SUM accumulator for multi-rank
                simulations.  ``None`` for single-rank use. Defaults to ``None``.
        """
        self._rank = rank
        self._size = size
        self._hub = hub

    @property
    def rank(self) -> int:
        """Get this rank's id.

        Returns:
            int: The rank id.
        """
        return self._rank

    @property
    def size(self) -> int:
        """Get the communicator size.

        Returns:
            int: The number of ranks.
        """
        return self._size

    def allreduce(self, data: Any) -> Any:
        """Sum-reduce ``data`` across the (simulated) communicator.

        Args:
            data (Any): This rank's contribution.

        Returns:
            Any: The genuine SUM across ranks (the hub result for multi-rank, or ``data``
            itself for single-rank).
        """
        if self._hub is not None:
            return self._hub.allreduce(np.asarray(data))
        assert self._size == 1, (
            "Multi-rank allreduce requires a shared hub; "
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
    *,
    part: Partition | None = None,
    **kwargs: Any,
) -> list[DistributedFunction]:
    """Simulate l2_project_bspline_distributed across n_parts ranks in one process.

    A single shared :class:`_AllreduceHub` turns the simulated ``allreduce`` into a genuine
    cross-rank SUM, so the simulation runs the *real* production code on every rank in two
    passes:

    * **Collecting pass** -- run each rank's projection once so it deposits its *actual*
      local load into the hub.  These intermediate results are discarded (an early rank
      sees only a partial sum, but its load tensor still has the correct shape, so
      production completes without error).
    * **Result pass** (after :meth:`_AllreduceHub.freeze`) -- run each rank again; the hub
      now returns the complete global sum, so every rank assembles the identical global
      field.  These results are returned.

    Because the hub reduces exactly what ``_l2.py`` feeds ``allreduce``, a regression in the
    local assembly (wrong stacking, wrong owned-cell masking) surfaces in the
    equivalence-vs-serial assertions instead of being masked by a pre-computed sum.

    Args:
        func (Any): Function to project, in the ``func(lattice)`` serial convention.
        space (BsplineSpace): The global space to project onto.
        n_parts (int): Number of simulated ranks.
        part (Partition | None): Explicit cell partition.  Defaults to a fresh
            ``partition_grid(tensor_product_grid(space), n_parts)``. Defaults to ``None``.
        **kwargs (Any): Forwarded verbatim to ``l2_project_bspline_distributed`` (e.g.
            ``boundary_interpolation``, ``n_quad``).

    Returns:
        list[DistributedFunction]: One result per rank, indexed by rank.
    """
    if part is None:
        part = partition_grid(tensor_product_grid(space), n_parts)
    hub = _AllreduceHub()

    # Collecting pass: every rank deposits its real local load into the hub.
    for rank in range(n_parts):
        ds = DistributedSpace(space, part, _FakeComm(rank, n_parts, hub=hub))
        l2_project_bspline_distributed(func, ds, **kwargs)
    hub.freeze()

    # Result pass: the hub now returns the complete global sum to every rank.
    results: list[DistributedFunction] = []
    for rank in range(n_parts):
        ds = DistributedSpace(space, part, _FakeComm(rank, n_parts, hub=hub))
        results.append(l2_project_bspline_distributed(func, ds, **kwargs))
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

    results = _simulate_distributed_l2(func, space, 2, part=part)
    assert results[0].local is not None
    assert results[1].local is None


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
# Where func is evaluated: the owned box, or the whole lattice
# ---------------------------------------------------------------------------


def _counting(seen: list[int]) -> Callable[[Any], np.ndarray]:
    """Return a scalar ``func(lattice)`` that records each call's point count in ``seen``."""

    def func(lat: Any) -> np.ndarray:
        mesh = _grid(lat)
        seen.append(int(mesh[0].size))
        return np.sin(np.pi * mesh[0]) * np.cos(np.pi * mesh[1]) + 0.25 * mesh[0]

    return func


def _non_box_partition(space: BsplineSpace, n_parts: int) -> Partition:
    """The block partition with rank 1's highest cell handed to rank 0.

    On the grids used here every block is at least two cells wide in each direction, so
    ranks 0 and 1 own non-box sets by construction and every other rank keeps its box.
    """
    owner = np.array(partition_grid(tensor_product_grid(space), n_parts).cell_owner)
    owner[int(np.flatnonzero(owner == 1).max())] = 0
    return Partition(owner, n_parts)


@pytest.mark.parametrize("n_parts", [2, 3, 4])
@pytest.mark.parametrize("n_intervals", [[8, 8], [7, 11]])
def test_block_partition_evaluates_each_owned_box_only(
    n_parts: int, n_intervals: list[int]
) -> None:
    """Under the block partition the collecting pass's per-rank counts sum to serial."""
    space = create_uniform_space([2, 2], n_intervals)
    seen: list[int] = []
    func = _counting(seen)
    serial = l2_project_bspline(func, space)
    serial_points = seen.pop()

    results = _simulate_distributed_l2(func, space, n_parts)

    collecting = seen[:n_parts]
    assert len(seen) == 2 * n_parts  # one call per rank, per simulation pass
    assert sum(collecting) == serial_points, collecting
    for dfn in results:
        np.testing.assert_allclose(
            dfn.global_function.control_points, serial.control_points, atol=1e-12
        )


@pytest.mark.parametrize("quadrature", ["gauss-legendre", "gauss-lobatto"])
@pytest.mark.parametrize("boundary_interpolation", [False, True])
@pytest.mark.parametrize("kind", ["block", "non-box", "empty-rank"])
def test_owned_box_and_whole_lattice_paths_match_serial(
    quadrature: Literal["gauss-legendre", "gauss-lobatto"],
    boundary_interpolation: bool,
    kind: str,
) -> None:
    """Both evaluation paths, alone and mixed in one reduction, reproduce the serial result.

    ``block`` takes the owned-box path on every rank; ``non-box`` takes the whole-lattice
    path on ranks 0 and 1 and the box path on ranks 2 and 3; ``empty-rank`` gives rank 1
    no cells. Gauss-Lobatto puts nodes on the element boundaries, so the box's node
    sub-range starts and ends on a knot shared with the neighbouring rank.
    """
    n_parts = 4
    space = create_uniform_space([3, 2], [8, 6])
    if kind == "block":
        part = partition_grid(tensor_product_grid(space), n_parts)
    elif kind == "non-box":
        part = _non_box_partition(space, n_parts)
    else:
        owner = np.array(partition_grid(tensor_product_grid(space), n_parts).cell_owner)
        owner[owner == 1] = 0
        part = Partition(owner, n_parts)
    kwargs: dict[str, Any] = {
        "quadrature": quadrature,
        "boundary_interpolation": boundary_interpolation,
        "n_quad": [4, 3],
    }
    seen: list[int] = []
    func = _counting(seen)
    serial = l2_project_bspline(func, space, **kwargs)

    results = _simulate_distributed_l2(func, space, n_parts, part=part, **kwargs)

    for dfn in results:
        np.testing.assert_allclose(
            dfn.global_function.control_points, serial.control_points, atol=1e-12
        )


def test_non_box_ranks_evaluate_the_whole_lattice() -> None:
    """Ranks whose owned set is not a box are handed the whole lattice; box ranks are not."""
    n_parts = 4
    space = create_uniform_space([2, 2], [8, 8])
    seen: list[int] = []
    func = _counting(seen)
    l2_project_bspline(func, space)
    serial_points = seen.pop()

    _simulate_distributed_l2(func, space, n_parts, part=_non_box_partition(space, n_parts))

    collecting = seen[:n_parts]
    assert collecting[0] == serial_points
    assert collecting[1] == serial_points
    assert all(count < serial_points for count in collecting[2:]), collecting


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
