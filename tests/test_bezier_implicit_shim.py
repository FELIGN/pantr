"""Tests for the ``pantr.bezier.implicit`` deprecation shim.

The implicit quadrature engine moved to :mod:`ocelat.algoim` in ocelat's
Stage 6 refactor; this module verifies that the compatibility shim left in
place still imports, exports the documented public names, and emits a
:class:`DeprecationWarning` on import.

Regression test for the one-release deprecation window: once the shim is
removed, these tests are removed together with it.
"""

from __future__ import annotations

import importlib
import sys
import types
import warnings
from collections.abc import Callable, Iterator

import ocelat.algoim
import pytest

_SHIM_NAME: str = "pantr.bezier.implicit"

_PUBLIC_NAMES: frozenset[str] = frozenset(
    {
        "ImplicitQuadrature",
        "QuadStrategy",
        "ReparamResult",
        "SurfQuadResult",
        "VolQuadResult",
        "monomial_to_bernstein_2d",
        "monomial_to_bernstein_3d",
    }
)


@pytest.fixture
def fresh_shim_import() -> Iterator[Callable[[], types.ModuleType]]:
    """Yield a factory that re-imports the shim with a clean module cache.

    The shim emits its :class:`DeprecationWarning` at import time; once Python
    caches the module in ``sys.modules`` subsequent ``import`` statements are
    silent. Each test that exercises import-time behaviour gets a fresh import
    by evicting the cache before and after the test runs.
    """
    sys.modules.pop(_SHIM_NAME, None)

    def _reimport() -> types.ModuleType:
        return importlib.import_module(_SHIM_NAME)

    try:
        yield _reimport
    finally:
        sys.modules.pop(_SHIM_NAME, None)


def test_shim_emits_deprecation_warning(
    fresh_shim_import: Callable[[], types.ModuleType],
) -> None:
    """Importing the shim raises exactly one ``DeprecationWarning``.

    The message must name both the old and new locations so users can
    migrate without digging through the source.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fresh_shim_import()

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    message = str(deprecations[0].message)
    assert "pantr.bezier.implicit" in message
    assert "ocelat.algoim" in message


def test_shim_reexports_public_api(
    fresh_shim_import: Callable[[], types.ModuleType],
) -> None:
    """All public names re-exported by the shim are the same objects as in ``ocelat.algoim``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        shim = fresh_shim_import()

    assert set(shim.__all__) == set(_PUBLIC_NAMES)
    for name in _PUBLIC_NAMES:
        assert getattr(shim, name) is getattr(ocelat.algoim, name), name


def test_from_import_still_works(
    fresh_shim_import: Callable[[], types.ModuleType],
) -> None:
    """``from pantr.bezier.implicit import ImplicitQuadrature`` yields the canonical class."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fresh_shim_import()

    from pantr.bezier.implicit import (  # noqa: PLC0415
        ImplicitQuadrature as ShimImplicitQuadrature,
    )

    assert ShimImplicitQuadrature is ocelat.algoim.ImplicitQuadrature
