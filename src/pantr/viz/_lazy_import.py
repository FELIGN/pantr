"""Lazy import helper for optional pyvista dependency."""

from __future__ import annotations

from types import ModuleType


def _import_pyvista() -> ModuleType:
    """Import and return the pyvista module, raising a clear error if absent.

    Returns:
        ModuleType: The ``pyvista`` module.

    Raises:
        ImportError: If pyvista is not installed.
    """
    try:
        import pyvista as pv  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "pyvista is required for visualization. Install it with: pip install pantr[viz]"
        ) from None
    return pv  # type: ignore[return-value,no-any-return]
