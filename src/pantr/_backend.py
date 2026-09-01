"""Backend selection between the Numba kernels and the C++ prototype.

PaNTr is being ported from Python-with-Numba to a C++ core (see ``design/*.md``).
During the port the **Numba implementation stays as the parity oracle**: the C++
backend is validated against it module by module, with the choice made here, at
the Python level, rather than at build time.

Unstable
--------

**This module is scaffolding and carries no compatibility promise.** It exists to
run two implementations side by side while the port proceeds, and when the port
ends the reason for it ends too -- so it may be renamed, reshaped or deleted
without a deprecation cycle, and nothing outside :mod:`pantr` should import it.
It is deliberately not re-exported from :mod:`pantr` and not listed in the docs
reference, unlike :mod:`pantr._parallel`, which is maintained to the same
standard but is meant to outlive the port. Making this one public would commit
the project to maintaining it past the point where it has a job.

Said plainly because it is known not to be enough: ``CLAUDE.md`` records that a
downstream consumer already imports pantr's private symbols, where pantr's own CI
cannot see the breakage. This paragraph is the notice; it is not a promise, and a
consumer that pins to this module is choosing to track it.

Selection
---------

The environment variable ``PANTR_BACKEND`` chooses the backend for the process:

.. code-block:: console

    PANTR_BACKEND=python pytest tests/     # the oracle, and the default
    PANTR_BACKEND=cpp    pytest tests/     # the port

Unset, the backend is :attr:`Backend.PYTHON`, which is always available.

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
every ``is Backend.PYTHON`` branch by the product of the two. It is written down
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

Five modules are ported so far, and **selecting the C++ backend changes only what
they cover**. A suite run under ``PANTR_BACKEND=cpp`` is not a C++ run; it is a run
in which those kernels are C++ and everything else is unchanged.

* :mod:`pantr.basis` -- three of the 1D tabulations: cardinal B-spline, Bernstein
  and Legendre. The Lagrange tabulation and everything multidimensional stay on
  Numba.
* :mod:`pantr.quad` -- four of the seven rule generators: Gauss-Legendre, the
  trapezoidal rule, the modified Chebyshev nodes and tanh-sinh, together with the
  Lambert W solve the last of those needs.
* :mod:`pantr.change_basis` -- all eight builders.
* :mod:`pantr.bezier` -- the arithmetic and the root finding, through two
  catalogues rather than one because they are two ports with two parity claims;
  and the :class:`~pantr.bezier.Bezier` **value type**, which is a different kind
  of entry and is listed with the other ported types below.
  **Interpolation is deliberately not ported**, and
  ``design/bezier_interpolation_port.md`` is where that ruling and its measurements
  live; it is worth reading before proposing any further port, because it is the
  template for declining one.
* :mod:`pantr.grid` -- all nine kernels: tensor-product point location, the BVH's
  build and its two query passes, and the five hierarchical addressing kernels.

**Ten types have moved as well, and that is a different kind of entry.** A
catalogue decides which code computes; a ported type decides which object holds the
state, so under ``PANTR_BACKEND=cpp`` the object's data lives in C++ and the Python
class is a wrapper around it. The ten, by the module that exports them:

* :mod:`pantr.geometry` -- :class:`~pantr.geometry.AABB`.
* :mod:`pantr.transform` -- :class:`~pantr.transform.AffineTransform`.
* :mod:`pantr.quad` -- :class:`~pantr.quad.QuadratureRule`.
* :mod:`pantr.bezier` -- :class:`~pantr.bezier.Bezier`.
* :mod:`pantr.grid` -- :class:`~pantr.grid.TensorProductGrid`,
  :class:`~pantr.grid.BVH`, :class:`~pantr.grid.CellTags`,
  :class:`~pantr.grid.FacetTags` and :class:`~pantr.grid.Partition`.
* :mod:`pantr.bspline` -- :class:`~pantr.bspline.BsplineSpace1D`.

**The last of those is why the two lists above are no longer parallel, and the
mismatch is real rather than an omission.** Every other module here earned its place
by porting *kernels*; :mod:`pantr.bspline` has none ported, so it is absent from the
module list, and yet the backend does change which object a caller of
:class:`~pantr.bspline.BsplineSpace1D` holds. Read the module list as "whose kernels
moved" and this one as "whose objects moved", and do not infer either from the other.

Two domain types in the geometry, transform, quad and grid modules have **not**
moved: :class:`pantr.grid.HierarchicalGrid`, which is the Python implementation under
either backend until its own ticket, and :class:`pantr.quad.PointsLattice`, which
``design/cross_backend_types.md`` rules out of the port with four recorded reasons --
the first pending, the second permanent. The rest of what those modules export is not
a candidate either way -- ``Grid`` is a :class:`typing.Protocol` and
``GridRestriction`` is a result record.

**:mod:`pantr.bspline` is counted differently, because it is at the start of its own
front rather than the end of one.** Listing its unported types here would be listing
a work queue: ``BsplineSpace``, ``Bspline``, ``THBSplineSpace``, ``THBSpline`` and the
extraction types are all still Python under either backend, and each has its own
ticket. What is worth stating is the boundary --
:class:`~pantr.bspline.BsplineSpace1D` alone dispatches, and every operation on it
(basis tabulation, the extraction operators, knot insertion, subdivision,
restriction) is still Numba under both backends.

**This list was wrong for two releases and that is worth a sentence.** It said three
modules and named neither half of ``bezier`` while both were merged and dispatching.
It was then wrong again through the type epic, naming one of the nine types above and
saying so in the entry for it rather than fixing itself. Nothing checks a prose list,
so it drifts silently; if you port a module or a type, the change is not finished
until this paragraph names it.

**Reading a change_basis result needs one more fact than the others.** Its
builders take their nodes from :mod:`pantr.quad`, so a call under
``PANTR_BACKEND=cpp`` gets C++ nodes wherever that rule dispatches and numpy nodes
where it does not. For :func:`pantr.change_basis.compute_lagrange_to_bernstein_1d`
that is visible in the answer: with ``CHEBYSHEV_2ND`` nodes in float32 the two
backends start from arrays that differ by one unit of roundoff, so the matrix
cannot be bit-identical however faithful the builder is.


**Three public quadrature rules are deliberately never dispatched** and will
report identical results under either backend:
:func:`pantr.quad.get_gauss_lobatto_legendre_1d`,
:func:`pantr.quad.get_chebyshev_gauss_1st_kind_1d` and
:func:`pantr.quad.get_chebyshev_gauss_2nd_kind_1d` stay on
:mod:`numpy.polynomial`, for reasons ``design/quadrature_algorithms.md`` records.
That matters to anyone reading an A/B measurement: bit-identical output from one
of those three is the switch being a no-op for it, not a parity success.

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

import functools
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import IntEnum
from typing import Any, Final, ParamSpec, Protocol, TypeVar, cast

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
        PYTHON: The Python implementation, always available and the parity
            oracle every C++ result is checked against. Named for the language
            rather than the JIT: this axis names which implementation *family*
            runs a kernel, and whether a given one happens to be Numba-compiled
            (as the basis kernels are) or plain NumPy (as ``pantr.quad`` is) is
            an implementation detail of that family, not a second axis.
        CPP: The C++ prototype, available only when the ``pantr._pantr_cpp``
            extension was built.
    """

    PYTHON = 0
    CPP = 1


def _cpp_extension_is_present() -> bool:
    """Report whether the compiled extension was built into this installation.

    Only presence is recorded, not the module object. Each kernel adapter --
    :func:`pantr.basis._basis_backend._cpp_cardinal_bspline_core` is the one that
    exists so far -- imports ``pantr._pantr_cpp`` by name where it calls it, so
    mypy resolves the call against ``src/pantr/_pantr_cpp/`` and would reject a
    call that no longer matches the binding. A stored module object would be typed
    :class:`~types.ModuleType`, which accepts any attribute and would keep
    typechecking after the signature changed underneath it.

    **The import succeeding is not the answer.** The type stubs live in a
    ``_pantr_cpp/`` directory beside the extension, and a directory of ``.pyi``
    files with no ``__init__.py`` is an importable *namespace package*. Python's
    finder prefers the extension whenever one exists, so a real build is
    unaffected; but on an installation with no extension the import now returns an
    empty module instead of raising, and every C++ parity test would run against
    it rather than skip. ``__file__`` is what separates the two: a namespace
    package has none.

    Returns:
        bool: True when ``pantr._pantr_cpp`` imports and is a real module rather
            than the stub directory seen as a namespace package.
    """
    try:
        import pantr._pantr_cpp  # noqa: PLC0415  (probe only, deliberately lazy)
    except ImportError:
        return False
    return getattr(pantr._pantr_cpp, "__file__", None) is not None


_CPP_AVAILABLE: Final = _cpp_extension_is_present()
"""Whether the C++ backend can be selected in this process.

Resolved once at import: whether the extension exists cannot change during a
run, and re-probing per call would put a `try/except ImportError` on a hot path.
"""


def available_backends() -> tuple[Backend, ...]:
    """List the backends this installation can actually run.

    Returns:
        tuple[Backend, ...]: The available backends, in ascending enum order.
            :attr:`Backend.PYTHON` is always present.
    """
    if not _CPP_AVAILABLE:
        return (Backend.PYTHON,)
    return (Backend.PYTHON, Backend.CPP)


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
    Backend.PYTHON,
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
``tests/parity/test_dispatch.py`` pins both.
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
        >>> with use_backend(Backend.PYTHON):
        ...     active_backend()
        <Backend.PYTHON: 0>
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


_P = ParamSpec("_P")
_R_co = TypeVar("_R_co", covariant=True)


class _BackendKeyedCache(Protocol[_P, _R_co]):
    """The callable :func:`backend_keyed_cache` returns.

    Same call signature as the function it wraps, plus the two attributes
    :func:`functools.lru_cache` is relied on for elsewhere in the library.

    Attributes:
        cache_clear (Callable[[], None]): Drop every entry, for every backend.
        cache_info (Callable[[], Any]): The wrapped cache's hit and miss counts.
    """

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R_co:
        """Call the wrapped function, keyed on the active backend.

        Args:
            *args (_P.args): Positional arguments of the wrapped function.
            **kwargs (_P.kwargs): Keyword arguments of the wrapped function.

        Returns:
            _R_co: The wrapped function's result for this backend.
        """
        ...

    def cache_clear(self) -> None:
        """Drop every cached entry, for every backend.

        Returns:
            None
        """
        ...

    def cache_info(self) -> Any:  # noqa: ANN401 -- functools._CacheInfo is private
        """Report the wrapped cache's statistics.

        Returns:
            Any: The :class:`functools._CacheInfo` named tuple.
        """
        ...


def backend_keyed_cache(
    maxsize: int = 128,
) -> Callable[[Callable[_P, _R_co]], _BackendKeyedCache[_P, _R_co]]:
    """Memoize a function whose result depends on the active backend.

    A drop-in replacement for :func:`functools.lru_cache` on any function that
    reaches a backend-dispatched kernel, however indirectly. **The wrapped
    function's signature does not change**; the backend enters the key inside the
    decorator, read from the calling thread's context at every lookup.

    Why this is not a tuning choice. A plain :func:`functools.lru_cache` on such a
    function is keyed on the arguments alone, so the first backend to populate an
    entry serves every later caller. Two failures follow, and the second is a
    wrong answer in the library rather than in a test:

    * A parity test that computes a value under one backend and then under the
      other is handed **the same object twice** and reports agreement it never
      measured. Measured on ``_cached_lagrange_to_bernstein_matrix``: with one
      Gauss-Legendre node moved by a single ulp between the two calls, ``A is B``
      is ``True``, and the matrices differ only once the cache is dropped.
    * :func:`use_backend` is scoped per thread precisely so callers may thread,
      and the extension releases the GIL to invite it. A process-wide cache is not
      scoped at all, so a thread inside a ``use_backend`` block populates entries
      that a thread outside it then reads.

    Clearing the cache when a :func:`use_backend` block opens and closes fixes the
    first failure and **not** the second: the entry is fresh, and still computed
    under whichever backend asked first. Measured, with one thread inside a block
    and one outside: the thread that entered no block receives the other's value.
    Keying is what closes both, and it costs one tuple element per lookup.

    Args:
        maxsize (int): Entries retained, across all backends together. Defaults to
            128.

    Returns:
        Callable[[Callable[_P, _R_co]], _BackendKeyedCache[_P, _R_co]]: A decorator
            producing a cached callable with the same signature as its argument,
            plus ``cache_clear`` and ``cache_info``.

    Example:
        >>> from pantr._backend import active_backend, backend_keyed_cache
        >>> @backend_keyed_cache(maxsize=4)
        ... def which(degree: int) -> tuple[int, int]:
        ...     return (int(active_backend()), degree)
        >>> which(3) == (int(active_backend()), 3)
        True
        >>> which.cache_info().currsize
        1
    """

    def decorate(function: Callable[_P, _R_co]) -> _BackendKeyedCache[_P, _R_co]:
        # `Any` rather than the ParamSpec: `lru_cache`'s stub types its arguments
        # as `Hashable`, which a ParamSpec cannot satisfy. The public wrapper below
        # carries the real signature, so nothing is lost at a call site.
        @functools.lru_cache(maxsize=maxsize)
        def keyed(_backend: Backend, /, *args: Any, **kwargs: Any) -> _R_co:  # noqa: ANN401
            return function(*args, **kwargs)

        # Same reason as above, at the call site this time: the ParamSpec cannot be
        # passed through `lru_cache`'s `Hashable`-typed parameters, so the call goes
        # through a `Callable[..., _R_co]` view of the same object.
        untyped: Callable[..., _R_co] = keyed

        @functools.wraps(function)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R_co:
            return untyped(active_backend(), *args, **kwargs)

        wrapper.cache_clear = keyed.cache_clear  # type: ignore[attr-defined]
        wrapper.cache_info = keyed.cache_info  # type: ignore[attr-defined]
        return cast("_BackendKeyedCache[_P, _R_co]", wrapper)

    return decorate
