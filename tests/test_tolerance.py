"""Tests for tolerance utilities."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any, cast

import numpy as np
import pytest
from numpy import typing as npt

from pantr.tolerance import (
    get_conservative,
    get_default,
    get_info,
    get_machine_epsilon,
    get_strict,
)

tol_mod = sys.modules["pantr.tolerance"]

NO_LONGDOUBLE = np.dtype(np.longdouble) == np.dtype(np.float64)

DEFAULT_TOL_F32: float = 1e-6
DEFAULT_TOL_F64: float = 1e-12
DEFAULT_TOL_LD: float = DEFAULT_TOL_F64 if NO_LONGDOUBLE else 1e-15

STRICT_TOL_F64: float = 1e-15
STRICT_TOL_LD: float = STRICT_TOL_F64 if NO_LONGDOUBLE else 1e-18

CONSERVATIVE_TOL_F64: float = 1e-10
CONSERVATIVE_TOL_LD: float = CONSERVATIVE_TOL_F64 if NO_LONGDOUBLE else 1e-12


class TestTolerance:
    """Test suite for tolerance utilities."""

    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            (np.float16, 1e-3),
            (np.float32, 1e-6),
            ("float64", 1e-12),
            (np.longdouble, 1e-12 if NO_LONGDOUBLE else 1e-15),
        ],
    )
    def test_get_default_tolerance(
        self, dtype: np.dtype[np.floating[Any]] | type[np.floating[Any]], expected: float
    ) -> None:
        """Test get_default with various dtypes."""
        assert get_default(dtype) == expected

    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            (np.float16, 1e-4),
            (np.float32, 1e-7),
            ("float64", 1e-15),
            (np.longdouble, 1e-15 if NO_LONGDOUBLE else 1e-18),
        ],
    )
    def test_get_strict_tolerance(
        self, dtype: np.dtype[np.floating[Any]] | type[np.floating[Any]], expected: float
    ) -> None:
        """Test get_strict with various dtypes."""
        assert get_strict(dtype) == expected

    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            (np.float16, 1e-2),
            (np.float32, 1e-5),
            ("float64", 1e-10),
            (np.longdouble, 1e-10 if NO_LONGDOUBLE else 1e-12),
        ],
    )
    def test_get_conservative_tolerance(
        self, dtype: np.dtype[np.floating[Any]] | type[np.floating[Any]], expected: float
    ) -> None:
        """Test get_conservative with various dtypes."""
        assert get_conservative(dtype) == expected

    @pytest.mark.parametrize(
        "dtype",
        [np.float16, np.float32, "float64", np.longdouble],
    )
    def test_get_machine_epsilon(
        self, dtype: np.dtype[np.floating[Any]] | type[np.floating[Any]]
    ) -> None:
        """Test get_machine_epsilon against np.finfo."""
        assert get_machine_epsilon(dtype) == np.finfo(dtype).eps

    def test_invalid_dtype_raises_error(self) -> None:
        """Test that an unsupported dtype raises a ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_default(np.int32)
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_strict("int64")
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_conservative(np.complex64)
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_machine_epsilon(np.uint8)

    def test_get_tolerance_info(self) -> None:
        """Test the get_info dictionary."""
        dtype = np.float64
        info = get_info(dtype)

        assert info["dtype"] == dtype
        assert info["machine_epsilon"] == np.finfo(dtype).eps
        assert info["default_tolerance"] == DEFAULT_TOL_F64
        assert info["strict_tolerance"] == STRICT_TOL_F64
        assert info["conservative_tolerance"] == CONSERVATIVE_TOL_F64
        assert info["precision_bits"] == np.finfo(dtype).precision
        assert info["precision_decimals"] == np.finfo(dtype).precision
        assert info["resolution"] == np.finfo(dtype).resolution
        assert info["max_value"] == np.finfo(dtype).max
        assert info["min_value"] == np.finfo(dtype).tiny

    def test_get_tolerance_info_string_dtype(self) -> None:
        """Test get_info with a string dtype."""
        dtype_str = "float32"
        info = get_info(dtype_str)
        finfo = np.finfo(dtype_str)

        assert info["dtype"] == dtype_str
        assert info["machine_epsilon"] == finfo.eps
        assert info["default_tolerance"] == DEFAULT_TOL_F32

    def test_tolerance_info_keys(self) -> None:
        """Test that get_info returns all expected keys."""
        info = get_info(np.float32)
        expected_keys = {
            "dtype",
            "machine_epsilon",
            "default_tolerance",
            "strict_tolerance",
            "conservative_tolerance",
            "precision_bits",
            "precision_decimals",
            "resolution",
            "max_value",
            "min_value",
        }
        assert set(info.keys()) == expected_keys

    def test_longdouble_else_branch_with_dtype_object(self) -> None:
        """Ensure the np.dtype(np.longdouble) path hits the else-branch."""
        dt = np.dtype(np.longdouble)
        assert get_default(dt) == DEFAULT_TOL_LD
        assert get_strict(dt) == STRICT_TOL_LD
        assert get_conservative(dt) == CONSERVATIVE_TOL_LD

    def test_import_non_alias_longdouble_branch_executes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Force the import-time non-alias branch to execute to improve coverage.

        This replaces the 'numpy' module temporarily with a minimal stub so that
        dtype(np.longdouble) != dtype(np.float64) holds during module reload.
        """
        # Save original numpy
        real_numpy = sys.modules.get("numpy")

        # Build a minimal fake numpy module sufficient for the import-time check
        fake_numpy = types.ModuleType("numpy")

        class _FakeDType:
            def __init__(self, token: object) -> None:
                self._token = token

            def __eq__(self, other: object) -> bool:
                return isinstance(other, _FakeDType) and self._token is other._token

            def __hash__(self) -> int:
                return hash(self._token)

        class _FakeFloating:
            @classmethod
            def __class_getitem__(cls: type[_FakeFloating], _item: object) -> type[_FakeFloating]:
                return cls

        class _FakeDTypeClass:
            def __call__(self, x: object) -> _FakeDType:
                return _FakeDType(x)

            @classmethod
            def __class_getitem__(
                cls: type[_FakeDTypeClass], _item: object
            ) -> type[_FakeDTypeClass]:
                return cls

        # distinct tokens so dtype(longdouble) != dtype(float64)
        fake_numpy.float16 = object()  # type: ignore[attr-defined]
        fake_numpy.float32 = object()  # type: ignore[attr-defined]
        fake_numpy.float64 = object()  # type: ignore[attr-defined]
        fake_numpy.longdouble = object()  # type: ignore[attr-defined]
        fake_numpy.floating = _FakeFloating  # type: ignore[attr-defined]

        def _fake_dtype(x: object) -> _FakeDType:
            return _FakeDType(x)

        fake_numpy.dtype = _FakeDTypeClass()  # type: ignore[attr-defined]
        # Provide attribute used by "from numpy import typing as npt"
        fake_numpy.typing = types.SimpleNamespace()  # type: ignore[attr-defined]

        # Install fake numpy and reload tolerance to execute the else-branch
        sys.modules["numpy"] = fake_numpy

        try:
            importlib.reload(tol_mod)
            # Sanity check that module loaded and presets exist
            assert hasattr(tol_mod, "_TOLERANCE_PRESETS")
        finally:
            # Restore real numpy and reload the module back to normal
            if real_numpy is not None:
                sys.modules["numpy"] = real_numpy
            importlib.reload(tol_mod)

    def test_get_tolerance_else_branch_via_monkeypatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hit the fallback longdouble branch in _get_tolerance to cover line 92."""

        class _FakeEnsuredDType:
            # A fake ensured dtype whose .type won't match float16/32/64
            def __init__(self) -> None:
                self.type = object()

        monkeypatch.setattr(tol_mod, "_ensure_float_dtype", lambda _d: _FakeEnsuredDType())

        # Pass a non-string, non-longdouble sentinel so special-casing is skipped
        sentinel = cast(npt.DTypeLike, object())
        result = get_default(sentinel)
        # Should equal the longdouble tolerance for the default preset
        expected = DEFAULT_TOL_LD
        assert result == expected
