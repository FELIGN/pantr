"""Tests for the per-rank MPI thread policy (:mod:`pantr.mpi._thread_policy`).

MPI is not available in the test environment: communicators are duck-typed fakes (the
policy code never imports ``mpi4py``). The autouse conftest fixture restores the Numba
thread count and resets the sticky policy flags after every test.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any

import numba as nb
import numpy as np
import pytest

import pantr
from pantr._parallel import _threads_explicitly_configured
from pantr.bspline import create_uniform_space
from pantr.grid import partition_grid, tensor_product_grid
from pantr.mpi import DistributedSpace, _thread_policy, configure_threads, from_dolfinx

_MAX = int(nb.config.NUMBA_NUM_THREADS)

pytestmark = pytest.mark.skipif(
    _MAX < 2, reason="needs >= 2 available threads to observe throttling"
)


class _FakeComm:
    """Minimal stand-in for an mpi4py communicator (single rank)."""

    rank = 0
    size = 1

    def allgather(self, sendobj: Any) -> list[Any]:
        return [sendobj]


def _build_distributed_space() -> DistributedSpace:
    """Construct a tiny single-rank DistributedSpace through the public API."""
    space = create_uniform_space([2], [4])
    part = partition_grid(tensor_product_grid(space), 1)
    return DistributedSpace(space, part, _FakeComm())


def _fake_mesh(n_cells: int) -> SimpleNamespace:
    """A dolfinx-mesh stand-in: one rank owning all `n_cells` cells."""
    return SimpleNamespace(
        comm=_FakeComm(),
        topology=SimpleNamespace(
            dim=1,
            original_cell_index=np.arange(n_cells, dtype=np.int64),
            index_map=lambda _dim: SimpleNamespace(size_local=n_cells),
        ),
    )


def _unthrottled_and_unmarked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Establish the no-explicit-configuration precondition at full width."""
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    nb.set_num_threads(_MAX)  # raw numba: does not mark explicit configuration
    assert not _threads_explicitly_configured()


# --------------------------------------------------------------------------- #
# Default policy via the entry points
# --------------------------------------------------------------------------- #


def test_distributed_space_applies_default_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing a DistributedSpace limits the process to 1 Numba thread."""
    _unthrottled_and_unmarked(monkeypatch)
    _build_distributed_space()
    assert pantr.get_num_threads() == 1


def test_from_dolfinx_applies_default_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling from_dolfinx limits the process to 1 Numba thread."""
    _unthrottled_and_unmarked(monkeypatch)
    from_dolfinx(_fake_mesh(4), 4)
    assert pantr.get_num_threads() == 1


def test_policy_is_sticky(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once applied, later constructions never re-throttle a raised count."""
    _unthrottled_and_unmarked(monkeypatch)
    _build_distributed_space()
    assert pantr.get_num_threads() == 1
    nb.set_num_threads(_MAX)  # raise again without marking explicit configuration
    _build_distributed_space()
    assert pantr.get_num_threads() == _MAX


# --------------------------------------------------------------------------- #
# Precedence: explicit configuration always wins
# --------------------------------------------------------------------------- #


def test_env_var_disables_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit NUMBA_NUM_THREADS environment variable disables the default."""
    _unthrottled_and_unmarked(monkeypatch)
    monkeypatch.setenv("NUMBA_NUM_THREADS", str(_MAX))
    _build_distributed_space()
    assert pantr.get_num_threads() == _MAX


def test_set_num_threads_disables_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prior pantr.set_num_threads call disables the default."""
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    pantr.set_num_threads(_MAX)
    _build_distributed_space()
    assert pantr.get_num_threads() == _MAX


def test_num_threads_context_disables_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prior pantr.num_threads context disables the default."""
    _unthrottled_and_unmarked(monkeypatch)
    with pantr.num_threads(1):
        pass
    _build_distributed_space()
    assert pantr.get_num_threads() == _MAX


def test_num_threads_active_context_prevents_throttling(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pantr.num_threads block active during construction prevents throttling."""
    _unthrottled_and_unmarked(monkeypatch)
    with pantr.num_threads(_MAX):
        _build_distributed_space()
        assert pantr.get_num_threads() == _MAX


def test_configure_threads_disables_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prior configure_threads call disables the default."""
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    configure_threads(_MAX)
    _build_distributed_space()
    assert pantr.get_num_threads() == _MAX


# --------------------------------------------------------------------------- #
# configure_threads
# --------------------------------------------------------------------------- #


def test_configure_threads_sets_count() -> None:
    """configure_threads applies the requested thread count."""
    configure_threads(1)
    assert pantr.get_num_threads() == 1
    configure_threads(_MAX)
    assert pantr.get_num_threads() == _MAX


def test_configure_threads_validates() -> None:
    """Out-of-range counts raise ValueError (delegated to set_num_threads)."""
    with pytest.raises(ValueError, match="n must be >= 1"):
        configure_threads(0)
    with pytest.raises(ValueError, match="NUMBA_NUM_THREADS"):
        configure_threads(_MAX + 1)


def test_configure_threads_limit_blas_warns_without_threadpoolctl() -> None:
    """limit_blas=True warns when threadpoolctl is absent (and still sets Numba)."""
    if importlib.util.find_spec("threadpoolctl") is not None:
        pytest.skip("threadpoolctl is installed; test requires its absence")
    with pytest.warns(UserWarning, match="threadpoolctl"):
        configure_threads(1, limit_blas=True)
    assert pantr.get_num_threads() == 1


def test_configure_threads_after_default_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """configure_threads overrides a fired default; later entry points are then no-ops."""
    _unthrottled_and_unmarked(monkeypatch)
    _build_distributed_space()
    assert pantr.get_num_threads() == 1
    configure_threads(_MAX)
    assert pantr.get_num_threads() == _MAX
    _build_distributed_space()  # sticky guard prevents re-throttling
    assert pantr.get_num_threads() == _MAX


# --------------------------------------------------------------------------- #
# Test-only reset helper
# --------------------------------------------------------------------------- #


def test_reset_clears_stickiness_and_explicit_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """_reset_policy_for_testing restores the pristine (default-applies) state."""
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    pantr.set_num_threads(_MAX)
    _build_distributed_space()
    assert pantr.get_num_threads() == _MAX  # explicit configuration won
    _thread_policy._reset_policy_for_testing()
    assert not _threads_explicitly_configured()
    _build_distributed_space()
    assert pantr.get_num_threads() == 1  # default applies again
