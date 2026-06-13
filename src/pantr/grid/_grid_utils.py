"""Shared Layer-2 helpers for the grid package.

Re-exports the ``float64`` coercion helper from :mod:`pantr.geometry` so that
existing ``from ._grid_utils import _as_float64`` imports keep working while the
implementation lives in a single place (it was previously duplicated here).
"""

from __future__ import annotations

from ..geometry import _as_float64

__all__ = ["_as_float64"]
