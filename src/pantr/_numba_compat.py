"""Numba compatibility shim for type-checker-friendly JIT decoration.

Provides decorators and utilities to handle Numba's runtime behavior
while remaining friendly to static type analysts (mypy). It also manages
the synchronization for asynchronous JIT warmup to prevent concurrent-compilation
crashes in parallel kernels.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Event set by the background warmup thread (in __init__.py) once all Numba
# functions have been compiled.  Callers that invoke parallel Numba kernels
# should call ``wait_for_jit_warmup()`` before the first Numba call to
# avoid a concurrent-compilation crash (a known Numba limitation with
# ``parallel=True``).
_warmup_complete = threading.Event()

# Fast flag so the common (post-warmup) path is a single boolean check.
_warmup_done = False


def wait_for_jit_warmup() -> None:
    """Block until the background JIT warmup has finished.

    After the event has been set once, subsequent calls return immediately
    (no lock, no syscall).
    """
    global _warmup_done  # noqa: PLW0603
    if _warmup_done:
        return
    _warmup_complete.wait()
    _warmup_done = True


if TYPE_CHECKING:
    # During type-checking, make the decorator a no-op that preserves types.
    def nb_jit(*args: object, **kwargs: object) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            return func

        return decorator

    # During type-checking, prange is just range.
    nb_prange = range

else:
    # At runtime, use the real Numba decorator and parallel range.
    import numba as nb

    nb_jit = nb.jit  # type: ignore[attr-defined]
    nb_prange = nb.prange  # type: ignore[attr-defined]
