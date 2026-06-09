"""Tests for the optional :mod:`pantr.mpi` package skeleton and import boundary."""

from __future__ import annotations

import importlib.util

import pytest

import pantr.mpi


def test_import_and_public_surface() -> None:
    """`import pantr.mpi` works regardless of mpi4py, exposing the skeleton API."""
    assert callable(pantr.mpi.mpi_available)
    assert callable(pantr.mpi.require_mpi)
    assert isinstance(pantr.mpi.HAS_MPI, bool)
    assert set(pantr.mpi.__all__) == {"HAS_MPI", "mpi_available", "require_mpi"}


def test_mpi_available_matches_find_spec() -> None:
    """`mpi_available()` agrees with a direct find_spec probe and with `HAS_MPI`."""
    expected = importlib.util.find_spec("mpi4py") is not None
    assert pantr.mpi.mpi_available() is expected
    assert pantr.mpi.HAS_MPI is expected


def test_require_mpi_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """`require_mpi()` raises an informative ImportError when mpi4py is absent."""
    monkeypatch.setattr(pantr.mpi, "mpi_available", lambda: False)
    with pytest.raises(ImportError, match="mpi4py"):
        pantr.mpi.require_mpi()


def test_require_mpi_imports_mpi4py_mpi(monkeypatch: pytest.MonkeyPatch) -> None:
    """When available, `require_mpi()` imports and returns exactly `mpi4py.MPI`."""
    sentinel = object()
    imported: dict[str, str] = {}

    def fake_import(name: str) -> object:
        imported["name"] = name
        return sentinel

    monkeypatch.setattr(pantr.mpi, "mpi_available", lambda: True)
    monkeypatch.setattr(importlib, "import_module", fake_import)
    assert pantr.mpi.require_mpi() is sentinel
    assert imported["name"] == "mpi4py.MPI"


def test_require_mpi_returns_module_when_available() -> None:
    """Integration: with a real mpi4py installed, the returned module is usable."""
    if not pantr.mpi.mpi_available():
        pytest.skip("mpi4py not installed")
    mpi = pantr.mpi.require_mpi()
    assert hasattr(mpi, "COMM_WORLD")


def test_core_does_not_import_pantr_mpi() -> None:
    """No serial-core module imports pantr.mpi (mirrors the import-linter contract)."""
    grimp = pytest.importorskip("grimp")
    graph = grimp.build_graph("pantr")
    mpi_modules = {m for m in graph.modules if m == "pantr.mpi" or m.startswith("pantr.mpi.")}
    assert mpi_modules, "pantr.mpi not found in the import graph"

    offenders: dict[str, list[str]] = {}
    for module in graph.modules:
        if module in mpi_modules:
            continue
        bad = sorted(graph.find_modules_directly_imported_by(module) & mpi_modules)
        if bad:
            offenders[module] = bad
    assert not offenders, f"serial-core modules importing pantr.mpi: {offenders}"
