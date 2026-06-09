"""Tests for the hatchling metadata hook in hatch_build.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hatch_build import _MPI_DEPENDENCY, _OPT_OUT_ENV, CustomMetadataHook


def _make_hook(tmp_path: Path, base_deps: list[str] | None = None) -> CustomMetadataHook:
    """Create a hook instance with a minimal pyproject.toml in tmp_path."""
    deps = base_deps if base_deps is not None else ["numpy>=1.26,<3"]
    (tmp_path / "pyproject.toml").write_text(f"[tool.pantr]\nbase-dependencies = {deps!r}\n")
    return CustomMetadataHook(str(tmp_path), {})


def test_mpi_injected_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``mpi4py`` is added to dependencies when ``PANTR_NO_MPI`` is not set."""
    monkeypatch.delenv(_OPT_OUT_ENV, raising=False)
    hook = _make_hook(tmp_path, ["numpy>=1.26,<3"])
    meta: dict[str, Any] = {}
    hook.update(meta)
    assert _MPI_DEPENDENCY in meta["dependencies"]
    assert "numpy>=1.26,<3" in meta["dependencies"]


def test_mpi_not_injected_when_flag_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``mpi4py`` is omitted from dependencies when ``PANTR_NO_MPI`` is set."""
    monkeypatch.setenv(_OPT_OUT_ENV, "1")
    hook = _make_hook(tmp_path, ["numpy>=1.26,<3"])
    meta: dict[str, Any] = {}
    hook.update(meta)
    assert _MPI_DEPENDENCY not in meta["dependencies"]
    assert "numpy>=1.26,<3" in meta["dependencies"]


def test_base_deps_always_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """All base dependencies appear in the result regardless of the MPI flag."""
    base = ["numpy>=1.26,<3", "scipy>=1.11,<2", "numba>=0.63"]
    monkeypatch.setenv(_OPT_OUT_ENV, "1")
    hook = _make_hook(tmp_path, base)
    meta: dict[str, Any] = {}
    hook.update(meta)
    for dep in base:
        assert dep in meta["dependencies"]


def test_missing_base_dependencies_raises(tmp_path: Path) -> None:
    """``RuntimeError`` is raised when ``[tool.pantr] base-dependencies`` is absent."""
    (tmp_path / "pyproject.toml").write_text("[tool.other]\nfoo = 1\n")
    hook = CustomMetadataHook(str(tmp_path), {})
    with pytest.raises(RuntimeError, match="base-dependencies"):
        hook.update({})


def test_missing_pyproject_raises(tmp_path: Path) -> None:
    """``FileNotFoundError`` is raised when ``pyproject.toml`` does not exist."""
    hook = CustomMetadataHook(str(tmp_path), {})
    with pytest.raises(FileNotFoundError):
        hook.update({})
