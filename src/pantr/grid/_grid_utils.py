"""Re-export shim: exposes ``_as_float64`` from :mod:`pantr.geometry`.

Keeps existing ``from ._grid_utils import _as_float64`` call-sites working
while the implementation lives in a single place.
"""

from __future__ import annotations

from ..geometry import _as_float64

__all__ = ["_as_float64"]
