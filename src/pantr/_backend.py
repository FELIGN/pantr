"""Backend selection between the Numba kernels and the C++ prototype.

PaNTr is being ported from Python-with-Numba to a C++ core (see ``design/*.md``).
During the port the **Numba implementation stays as the parity oracle**: the C++
backend is validated against it module by module, with the choice made here, at
the Python level, rather than at build time.

Selection
---------

The environment variable ``PANTR_BACKEND`` chooses the backend for the process:

.. code-block:: console

    PANTR_BACKEND=numba pytest tests/     # the oracle, and the default
    PANTR_BACKEND=cpp   pytest tests/     # the port

Unset, the backend is :attr:`Backend.NUMBA`, which is always available.

**An explicit request never falls back.** If ``PANTR_BACKEND`` names a backend
that is not available, importing :mod:`pantr` fails, loudly, naming what is
missing and how to build it. That is deliberate and it is the whole point of the
rule: a silent downgrade to the other backend would make every A/B measurement
untrustworthy, and the measurement is what the override exists to enable. The
same reasoning appears in ``design/simd.md`` for the ISA-variant override, where
a variant that is requested and missing must fail rather than quietly load
another.

Scope
-----

One kernel is ported so far, the cardinal B-spline tabulation of
:func:`pantr.basis.tabulate_cardinal_bspline_1d`. Selecting the C++ backend
changes that kernel and nothing else; every other function keeps running its
Numba implementation regardless. So a suite run under ``PANTR_BACKEND=cpp`` is
not a C++ run, it is a run in which one kernel is C++.

What is here, and what is not
-----------------------------

**This module is policy only.** It names the backends, resolves the selection
and enforces the never-fall-back rule; it holds no kernel and knows of none.
The map from a selection to the callables lives beside the kernels themselves --
:mod:`pantr.basis._basis_backend` for the basis kernels, one such module per
ported package.

That is what keeps the dependency one-directional: a catalogue imports its
kernels, and its consumers import the policy, so a catalogue placed here would
close a cycle that only lazy imports could hold open -- one more per kernel
ported. An import-linter contract in ``pyproject.toml`` forbids this module from
importing any subpackage of :mod:`pantr`, so the arrangement cannot quietly
revert.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import IntEnum
from typing import Final

__all__ = [
    "Backend",
    "active_backend",
    "available_backends",
    "use_backend",
]

_ENV_VAR: Final[str] = "PANTR_BACKEND"
"""Name of the environment variable that selects the backend."""


class Backend(IntEnum):
    """The available implementations of a ported kernel.

    An :class:`~enum.IntEnum` rather than a string, per the project's convention
    that a closed set of choices is never stringly typed. The environment
    variable is a string because the operating system offers nothing else; it is
    a boundary and is converted on the way in by :func:`_parse_backend_name`.

    Attributes:
        NUMBA: The Numba kernels. Always available, and the parity oracle every
            C++ result is checked against.
        CPP: The C++ prototype, available only when the ``pantr._pantr_cpp``
            extension was built.
    """

    NUMBA = 0
    CPP = 1


def _cpp_extension_is_present() -> bool:
    """Report whether the compiled extension was built into this installation.

    Only presence is recorded, not the module object. Each kernel adapter --
    :func:`pantr.basis._basis_backend._cpp_cardinal_bspline_core` is the one that
    exists so far -- imports ``pantr._pantr_cpp`` by name where it calls it, so
    mypy resolves the call against ``src/pantr/_pantr_cpp.pyi`` and would reject a
    call that no longer matches the binding. A stored module object would be typed
    :class:`~types.ModuleType`, which accepts any attribute and would keep
    typechecking after the signature changed underneath it.

    Returns:
        bool: True when ``pantr._pantr_cpp`` imports.
    """
    try:
        import pantr._pantr_cpp  # noqa: F401, PLC0415  (probe only, deliberately lazy)
    except ImportError:
        return False
    return True


_CPP_AVAILABLE: Final = _cpp_extension_is_present()
"""Whether the C++ backend can be selected in this process.

Resolved once at import: whether the extension exists cannot change during a
run, and re-probing per call would put a `try/except ImportError` on a hot path.
"""


def available_backends() -> tuple[Backend, ...]:
    """List the backends this installation can actually run.

    Returns:
        tuple[Backend, ...]: The available backends, in ascending enum order.
            :attr:`Backend.NUMBA` is always present.
    """
    if not _CPP_AVAILABLE:
        return (Backend.NUMBA,)
    return (Backend.NUMBA, Backend.CPP)


def _parse_backend_name(name: str) -> Backend:
    """Convert the environment variable's string to a :class:`Backend`.

    The single place a backend name crosses from text into the type system.

    Args:
        name (str): Value of ``PANTR_BACKEND``, matched case-insensitively and
            with surrounding whitespace ignored.

    Returns:
        Backend: The named backend.

    Raises:
        ValueError: If ``name`` does not name a backend.
    """
    normalized = name.strip().lower()
    for backend in Backend:
        if backend.name.lower() == normalized:
            return backend
    known = ", ".join(sorted(b.name.lower() for b in Backend))
    raise ValueError(f"{_ENV_VAR}={name!r} does not name a backend. Known backends: {known}.")


def _resolve_backend_from_environment() -> Backend:
    """Read and validate the backend requested by the environment.

    Returns:
        Backend: The requested backend, or :attr:`Backend.NUMBA` when
            ``PANTR_BACKEND`` is unset.

    Raises:
        ValueError: If ``PANTR_BACKEND`` does not name a backend.
        RuntimeError: If it names a backend this installation cannot run. This is
            raised rather than silently falling back, so that a measurement taken
            under an explicit request cannot be a measurement of the other
            backend.
    """
    requested = os.environ.get(_ENV_VAR)
    if requested is None:
        return Backend.NUMBA

    backend = _parse_backend_name(requested)
    if backend in available_backends():
        return backend

    raise RuntimeError(
        f"{_ENV_VAR}={requested!r} was requested but the {backend.name} backend is "
        f"not available in this installation.\n"
        f"  The C++ extension pantr._pantr_cpp was not built.\n"
        f"  Fix: pip install -e . (which builds it through scikit-build-core)\n"
        f"pantr does not fall back to another backend here: an explicit request "
        f"that silently ran something else would make any A/B measurement taken "
        f"under it meaningless."
    )


_PROCESS_DEFAULT: Final[Backend] = _resolve_backend_from_environment()
"""The backend every thread and task starts from, resolved once from the environment."""

_ACTIVE: Final[ContextVar[Backend]] = ContextVar("pantr_active_backend", default=_PROCESS_DEFAULT)
"""The backend in effect, per thread and per task.

A :class:`~contextvars.ContextVar` rather than a module global, with the
environment-resolved value as its default so that a thread or task which entered
no :func:`use_backend` block reads exactly what ``PANTR_BACKEND`` selected.

A global is wrong here for two measured reasons, neither of them exotic. Two
overlapping :func:`use_backend` blocks in different threads lose an update: the
second saves the first's override as its "previous" and restores the process to
*that* on the way out, so the selection stays changed after every block has
exited. And an override in one thread reaches into every other one, so a worker
that never asked for a backend changes kernel mid-flight. Both are the *only*
outcome of their interleaving rather than a rare race, and the binding releases
the GIL specifically so that callers may thread at the Python level, which makes
this the shape the module invites rather than a pathological one.
``tests/test_cpp_parity.py`` pins both.
"""


def active_backend() -> Backend:
    """Report the backend currently in effect.

    Returns:
        Backend: The backend selected by ``PANTR_BACKEND``, or by a
            :func:`use_backend` block enclosing this call **in this thread or
            task**. A thread that entered no such block reads the environment's
            selection, whatever another thread is doing.
    """
    return _ACTIVE.get()


@contextmanager
def use_backend(backend: Backend) -> Iterator[None]:
    """Run a block with a different backend in effect.

    A scoped context manager rather than a setter, so the selection cannot be
    left changed by a function that returns early or raises. Existing tests use
    it to run the same assertions against both backends within one process.

    **The scope is the calling thread or task, not the process.** A thread
    started inside the block runs on the process default (what ``PANTR_BACKEND``
    selected), not on ``backend``; to give it the override, pass the backend into
    the worker and let it open its own block. That is the deliberate direction:
    the alternative reaches into threads that never asked, which is the same
    silent substitution this module refuses to do for a missing backend -- and
    the one place a wrong answer would look like a measurement.

    Args:
        backend (Backend): The backend to select for the duration of the block.

    Yields:
        None: The block runs with ``backend`` active in this thread or task.

    Raises:
        RuntimeError: If ``backend`` is not available. As with the environment
            variable, an explicit request never falls back.

    Example:
        >>> from pantr._backend import Backend, active_backend, use_backend
        >>> with use_backend(Backend.NUMBA):
        ...     active_backend()
        <Backend.NUMBA: 0>
    """
    if backend not in available_backends():
        raise RuntimeError(
            f"the {backend.name} backend is not available in this installation; "
            f"available: {', '.join(b.name for b in available_backends())}"
        )
    token = _ACTIVE.set(backend)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
