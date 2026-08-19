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
untrustworthy, and the measurement is what the override exists to enable.

Two axes, not one
-----------------

``design/simd.md`` schedules a second choice: ship the extension compiled
several times (baseline, ``x86-64-v3``, ``x86-64-v4``) and pick one at import.
That is a **different question** from :class:`Backend`. Which family runs the
kernel, and which build of that family, are independent -- so they are two enums
and two variables, :class:`IsaVariant` and ``PANTR_ISA_VARIANT``, under the same
never-fall-back rule.

Folding the second into the first, as ``CPP_V3`` and ``CPP_V4`` members of
:class:`Backend`, would multiply :func:`available_backends`, the parse, and
every ``is Backend.NUMBA`` branch by the product of the two. It is written down
now rather than when the ladder is built, because the accepted values of an
environment variable are user-facing the moment anyone puts one in a script:
adding an axis later is additive, but re-spelling ``PANTR_BACKEND=cpp`` as
``PANTR_BACKEND=cpp_v3`` is a break. **Today's accepted values are unchanged.**

**No ISA variant is built.** ``design/simd.md`` gates the ladder on a
measurement, ``cmake/PantrCompileOptions.cmake`` sets no ``-march``, and
``scripts/ci_local.sh discipline`` asserts that none appears in the build files.
So :func:`available_isa_variants` reports the baseline alone, and this axis is a
shape rather than a capability.

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
from typing import Final, TypeVar

__all__ = [
    "Backend",
    "IsaVariant",
    "active_backend",
    "active_isa_variant",
    "available_backends",
    "available_isa_variants",
    "use_backend",
]

_ENV_VAR: Final[str] = "PANTR_BACKEND"
"""Name of the environment variable that selects the backend."""

_ISA_ENV_VAR: Final[str] = "PANTR_ISA_VARIANT"
"""Name of the environment variable that selects the ISA variant."""

_Choice = TypeVar("_Choice", bound=IntEnum)
"""One of the closed sets of choices this module resolves from the environment."""


class Backend(IntEnum):
    """The available implementations of a ported kernel.

    An :class:`~enum.IntEnum` rather than a string, per the project's convention
    that a closed set of choices is never stringly typed. The environment
    variable is a string because the operating system offers nothing else; it is
    a boundary and is converted on the way in by :func:`_parse_choice`.

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


class IsaVariant(IntEnum):
    """Which build of the C++ extension runs: the instruction-set ladder.

    A second axis, orthogonal to :class:`Backend`. ``design/simd.md`` settles the
    shape of the ladder -- compile the extension once per instruction-set level,
    pick one at import through a probe module compiled at the toolchain baseline
    -- and gates **building** any of it on a measurement that has been taken but
    not acted on. So this enum has one member today.

    The two questions are kept apart because they compose rather than combine:
    folding the ladder into :class:`Backend` as ``CPP_V3`` and ``CPP_V4`` would
    multiply the parse, the availability list and every branch by the product of
    the two axes, and it would make ``PANTR_BACKEND``'s accepted values change
    when the ladder lands. Adding a member here is additive instead.

    Attributes:
        BASELINE: The extension as it is built today -- no ``-march``, so
            whatever the compiler's default target is. Always the answer until
            the ladder is built.
    """

    BASELINE = 0


def available_isa_variants() -> tuple[IsaVariant, ...]:
    """List the ISA variants this installation can actually load.

    The ladder is not built: ``cmake/PantrCompileOptions.cmake`` sets no
    ``-march`` and ``scripts/ci_local.sh discipline`` asserts that none appears
    in the build files, so the one extension that exists is the baseline. When
    the ladder lands this becomes a probe -- which variant modules were shipped,
    intersected with what the CPU and the operating system actually support, per
    ``design/simd.md``.

    Deliberately independent of :func:`available_backends`. A variant describes
    which build of the C++ family would be used, which is a well-formed answer
    even where the C++ backend is not installed; coupling them would make
    ``PANTR_ISA_VARIANT=baseline`` an error on a Numba-only installation, where
    it is merely inert.

    Returns:
        tuple[IsaVariant, ...]: The available variants, in ascending enum order.
            :attr:`IsaVariant.BASELINE` is always present.
    """
    return (IsaVariant.BASELINE,)


def _parse_choice(value: str, choices: type[_Choice], env_var: str) -> _Choice:
    """Convert an environment variable's string to one of a closed set of choices.

    The single place a name crosses from text into the type system, for both
    axes. The variables are strings because the operating system offers nothing
    else; they are a boundary and are converted here on the way in.

    Args:
        value (str): Value of the variable, matched case-insensitively and with
            surrounding whitespace ignored.
        choices (type[_Choice]): The enum to match against.
        env_var (str): Name of the variable, for the error message.

    Returns:
        _Choice: The named member.

    Raises:
        ValueError: If ``value`` names no member of ``choices``.
    """
    normalized = value.strip().lower()
    for choice in choices:
        if choice.name.lower() == normalized:
            return choice
    known = ", ".join(sorted(c.name.lower() for c in choices))
    raise ValueError(f"{env_var}={value!r} is not one of: {known}.")


def _resolve_from_environment(
    env_var: str,
    choices: type[_Choice],
    available: tuple[_Choice, ...],
    default: _Choice,
    fix_hint: str,
) -> _Choice:
    """Read and validate one axis of the selection, refusing to fall back.

    Shared by both axes, because the rule is one rule: an explicit request that
    cannot be honoured raises rather than quietly running something else. Written
    once so the two cannot drift apart -- the failure that drift would produce is
    an A/B measurement of the wrong thing, which looks like a result.

    Args:
        env_var (str): Name of the environment variable to read.
        choices (type[_Choice]): The enum it selects from.
        available (tuple[_Choice, ...]): The members this installation can run.
        default (_Choice): What an unset variable means.
        fix_hint (str): Indented lines telling the user what is missing and how
            to get it, quoted in the failure.

    Returns:
        _Choice: The requested member, or ``default`` when the variable is unset.

    Raises:
        ValueError: If the variable names no member of ``choices``.
        RuntimeError: If it names one this installation cannot run.
    """
    requested = os.environ.get(env_var)
    if requested is None:
        return default

    chosen = _parse_choice(requested, choices, env_var)
    if chosen in available:
        return chosen

    raise RuntimeError(
        f"{env_var}={requested!r} was requested but {chosen.name} is not available "
        f"in this installation.\n"
        f"{fix_hint}\n"
        f"pantr does not fall back to another choice here: an explicit request "
        f"that silently ran something else would make any A/B measurement taken "
        f"under it meaningless."
    )


_PROCESS_DEFAULT: Final[Backend] = _resolve_from_environment(
    _ENV_VAR,
    Backend,
    available_backends(),
    Backend.NUMBA,
    "  The C++ extension pantr._pantr_cpp was not built.\n"
    "  Fix: pip install -e . (which builds it through scikit-build-core)",
)
"""The backend every thread and task starts from, resolved once from the environment."""

_ACTIVE_ISA_VARIANT: Final[IsaVariant] = _resolve_from_environment(
    _ISA_ENV_VAR,
    IsaVariant,
    available_isa_variants(),
    IsaVariant.BASELINE,
    "  Only the baseline is built: design/simd.md gates the ISA ladder on a\n"
    "  measurement, and no -march reaches the build.",
)
"""The ISA variant in effect, resolved once at import.

A plain constant rather than a :class:`~contextvars.ContextVar`, because there is
nothing to scope: the variant decides which extension module is *loaded*, so
changing it inside a running process would mean unloading one binary and
importing another. :func:`use_backend` has a scoped form because both backends
are loaded at once and choosing between them is a per-call decision; this is not.
"""

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


def active_isa_variant() -> IsaVariant:
    """Report the ISA variant in effect.

    Resolved once at import from ``PANTR_ISA_VARIANT``, and process-wide: unlike
    the backend, the variant selects which binary is loaded, so it has no scoped
    override.

    Returns:
        IsaVariant: The variant selected by ``PANTR_ISA_VARIANT``, or
            :attr:`IsaVariant.BASELINE` when it is unset. The baseline is the
            only variant built today.
    """
    return _ACTIVE_ISA_VARIANT


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
