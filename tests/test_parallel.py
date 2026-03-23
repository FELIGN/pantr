"""Tests for the parallel thread-count control module."""

from __future__ import annotations

import pytest

import pantr
from pantr._parallel import get_num_threads, num_threads, set_num_threads


class TestGetSetNumThreads:
    """Tests for get_num_threads / set_num_threads."""

    def test_get_returns_positive(self) -> None:
        """get_num_threads returns a positive integer."""
        n = get_num_threads()
        assert isinstance(n, int)
        assert n >= 1

    def test_set_and_get_roundtrip(self) -> None:
        """set_num_threads followed by get_num_threads returns the set value."""
        prev = get_num_threads()
        try:
            set_num_threads(1)
            assert get_num_threads() == 1
        finally:
            set_num_threads(prev)

    def test_set_zero_raises(self) -> None:
        """set_num_threads(0) raises ValueError."""
        with pytest.raises(ValueError, match="n must be >= 1"):
            set_num_threads(0)

    def test_set_negative_raises(self) -> None:
        """set_num_threads(-1) raises ValueError."""
        with pytest.raises(ValueError, match="n must be >= 1"):
            set_num_threads(-1)

    def test_set_exceeds_max_raises(self) -> None:
        """set_num_threads beyond NUMBA_NUM_THREADS raises ValueError."""
        import numba as nb  # noqa: PLC0415

        max_threads: int = nb.config.NUMBA_NUM_THREADS
        with pytest.raises(ValueError, match="NUMBA_NUM_THREADS"):
            set_num_threads(max_threads + 1)


class TestNumThreadsContextManager:
    """Tests for the num_threads context manager."""

    def test_restores_previous_value(self) -> None:
        """Thread count is restored after the context manager exits."""
        prev = get_num_threads()
        with num_threads(1):
            assert get_num_threads() == 1
        assert get_num_threads() == prev

    def test_restores_on_exception(self) -> None:
        """Thread count is restored even when an exception is raised."""
        prev = get_num_threads()
        with pytest.raises(RuntimeError), num_threads(1):
            raise RuntimeError("test error")
        assert get_num_threads() == prev

    def test_nested_context_managers(self) -> None:
        """Nested context managers restore correctly."""
        prev = get_num_threads()
        with num_threads(1):
            assert get_num_threads() == 1
            with num_threads(1):
                assert get_num_threads() == 1
            assert get_num_threads() == 1
        assert get_num_threads() == prev

    def test_limit_blas_without_threadpoolctl(self) -> None:
        """limit_blas=True does not error when threadpoolctl is absent."""
        prev = get_num_threads()
        # This should not raise even if threadpoolctl is not installed.
        with num_threads(1, limit_blas=True):
            assert get_num_threads() == 1
        assert get_num_threads() == prev


class TestPublicReExports:
    """Tests for the public API re-exports."""

    def test_set_num_threads_reexported(self) -> None:
        """pantr.set_num_threads is the same function as _parallel.set_num_threads."""
        assert pantr.set_num_threads is set_num_threads

    def test_get_num_threads_reexported(self) -> None:
        """pantr.get_num_threads is the same function as _parallel.get_num_threads."""
        assert pantr.get_num_threads is get_num_threads

    def test_num_threads_reexported(self) -> None:
        """pantr.num_threads is the same function as _parallel.num_threads."""
        assert pantr.num_threads is num_threads
