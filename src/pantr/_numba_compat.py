"""Numba compatibility shim for type-checker-friendly JIT decoration."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

if TYPE_CHECKING:
    # During type-checking, make the decorator a no-op that preserves types.
    def nb_jit(*args: object, **kwargs: object) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            return func

        return decorator

else:
    # At runtime, use the real Numba decorator.
    import numba as nb

    nb_jit = nb.jit  # type: ignore[attr-defined]
