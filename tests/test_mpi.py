"""Tests for the optional :mod:`pantr.mpi` package skeleton and import boundary."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys

import pytest

import pantr.mpi


def test_import_and_public_surface() -> None:
    """`import pantr.mpi` works regardless of mpi4py, exposing the skeleton API."""
    assert callable(pantr.mpi.mpi_available)
    assert callable(pantr.mpi.require_mpi)
    assert callable(pantr.mpi.from_dolfinx)
    assert callable(pantr.mpi.configure_threads)
    assert isinstance(pantr.mpi.DistributedSpace, type)
    assert isinstance(pantr.mpi.HAS_MPI, bool)
    assert set(pantr.mpi.__all__) == {
        "DistributedSpace",
        "HAS_MPI",
        "configure_threads",
        "from_dolfinx",
        "mpi_available",
        "require_mpi",
    }


def test_mpi_available_matches_find_spec() -> None:
    """`mpi_available()` agrees with a direct find_spec probe and with `HAS_MPI`."""
    expected = importlib.util.find_spec("mpi4py") is not None
    assert pantr.mpi.mpi_available() is expected
    assert pantr.mpi.HAS_MPI is expected


def test_require_mpi_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """`require_mpi()` raises an informative ImportError when mpi4py is absent."""
    monkeypatch.setattr(pantr.mpi, "mpi_available", lambda: False)
    with pytest.raises(ImportError, match="PANTR_NO_MPI"):
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
    assert "name" in imported, "fake_import was never called — monkeypatch target may be wrong"
    assert imported["name"] == "mpi4py.MPI"


def test_require_mpi_raises_on_broken_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """`require_mpi()` wraps non-ImportError load failures as ImportError."""

    def broken_import(name: str) -> object:
        raise OSError("libmpi.so: cannot open shared object file")

    monkeypatch.setattr(pantr.mpi, "mpi_available", lambda: True)
    monkeypatch.setattr(importlib, "import_module", broken_import)
    with pytest.raises(ImportError, match="mpi4py.MPI"):
        pantr.mpi.require_mpi()


def test_require_mpi_returns_module_when_available() -> None:
    """Integration: with a real mpi4py installed, the returned module is usable."""
    if not pantr.mpi.mpi_available():
        pytest.skip("mpi4py not installed")
    mpi = pantr.mpi.require_mpi()
    assert hasattr(mpi, "COMM_WORLD")


def test_pantr_toplevel_does_not_import_mpi() -> None:
    """``import pantr`` must not cause ``pantr.mpi`` to appear in ``sys.modules``."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pantr, sys; assert 'pantr.mpi' not in sys.modules, list(sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_core_does_not_import_pantr_mpi() -> None:
    """No serial-core module directly imports pantr.mpi (mirrors the import-linter contract).

    Note: only direct (one-hop) imports are checked here. Transitive coverage is
    provided by the import-linter contract in pyproject.toml (``make import-lint``).
    """
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
