"""Backend-dispatch machinery: the axes, the never-fall-back rule, and use_backend.

Kernel-agnostic tests of `pantr._backend` and the seam it dispatches through --
which backend and which ISA variant are selected, that a missing choice on either
axis raises rather than falling back, and that `use_backend`'s ContextVar-backed
override is scoped correctly across threads. Nothing here asserts on a kernel
result; the parity/accuracy tests of each ported kernel live in their own
sibling module under `tests/parity`.
"""

from __future__ import annotations

import sys
import threading
import types
from collections.abc import Callable

import pytest

from pantr import _numba_compat
from pantr._backend import (
    Backend,
    IsaVariant,
    active_backend,
    active_isa_variant,
    available_isa_variants,
    use_backend,
)
from pantr.basis._basis_backend import cardinal_bspline_core
from pantr.change_basis import _change_basis_backend as _cb_backend
from pantr.quad._quad_backend import (
    chebyshev_nodes_kernel,
    gauss_legendre_kernel,
    lambert_w_kernel,
    tanh_sinh_kernel,
    trapezoidal_kernel,
)
from tests._parity_harness import build_provenance, cpp_backend_available, demand_cpp_backend


def test_cpp_extension_presence_is_declared() -> None:
    """Report the extension's presence as a test outcome rather than as silence.

    CLAUDE.md: "A missing optional dependency skips tests silently", and a local
    green built on such a skip has let real failures through here before. This test
    exists so that state is always in the report -- a pass when the extension is
    there, one visible skip with a build hint when it is not, and a failure when
    ``PANTR_REQUIRE_CPP`` says it had to be.
    """
    demand_cpp_backend()
    assert cpp_backend_available()


def test_a_stub_only_namespace_package_is_not_the_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare import cannot answer whether the extension was built, and does not.

    The type stubs live in ``src/pantr/_pantr_cpp/``, a directory of ``.pyi`` files
    with no ``__init__.py``. Python's finder prefers a real extension whenever one
    exists, but with none built that directory imports as an empty *namespace
    package* rather than raising ``ImportError`` -- so the probe that used to end at
    ``import pantr._pantr_cpp`` would report the C++ backend present on a
    Numba-only installation, and every parity test would fail instead of skipping.
    Measured before the fix: 81 failures in ``tests/parity/test_geometry_aabb.py``
    where 97 skips belong.

    A namespace package has no ``__file__``; a real module does. That is the
    distinction, and it is asserted in both directions so a probe that had degraded
    to a constant would fail here.
    """
    import pantr  # noqa: PLC0415
    from pantr import _backend  # noqa: PLC0415

    def install(module: types.ModuleType) -> None:
        """Put a module where a fresh ``import pantr._pantr_cpp`` would leave it.

        Both places, because the import statement writes to both and reads back
        from the second: it binds ``sys.modules[name]``, and it sets ``name``'s
        last component as an attribute of the parent package. On a module already
        in ``sys.modules`` the statement is a no-op, so patching only the mapping
        would leave the probe reading whatever this process imported for real --
        which, on a machine where the extension is built, is the extension.
        """
        monkeypatch.setitem(sys.modules, "pantr._pantr_cpp", module)
        monkeypatch.setattr(pantr, "_pantr_cpp", module, raising=False)

    namespace_package = types.ModuleType("pantr._pantr_cpp")
    namespace_package.__path__ = []  # type: ignore[attr-defined]
    assert not hasattr(namespace_package, "__file__"), (
        "the fixture must model a namespace package, whose defining trait here is "
        "the absence of __file__"
    )
    install(namespace_package)
    assert not _backend._cpp_extension_is_present()

    compiled = types.ModuleType("pantr._pantr_cpp")
    compiled.__file__ = "/nowhere/_pantr_cpp.so"
    install(compiled)
    assert _backend._cpp_extension_is_present()


def test_build_provenance_is_reported(cpp_backend: None) -> None:
    """Pin the three attributes the parity claim is selected from.

    The claim in :func:`_parity_claim` is conditional on ``__fp_contract__``. A
    binding that stopped reporting it, or reported an unrecognised value, would send
    every parity test down whichever branch a default happened to pick, silently.
    """
    provenance = build_provenance()
    assert provenance.compiler, "the extension must name the compiler that built it"
    assert provenance.fp_contract in {"available", "unavailable-on-target-isa"}, (
        f"unrecognised __fp_contract__ {provenance.fp_contract!r}: the parity claim is "
        f"selected from this value and cannot be selected from a value it does not know"
    )
    assert isinstance(provenance.has_std_mdspan, bool)


def test_jit_warmup_barrier_was_taken() -> None:
    """Assert the session barrier ran before any kernel call.

    No in-process test can observe the race it prevents (the barrier is a
    once-per-process event and the failure is an abort, not an exception), so what
    is testable is that the barrier was taken. CLAUDE.md says exactly this.
    """
    assert _numba_compat._warmup_done, (
        "the session-scoped warmup barrier did not run before the kernel tests; a "
        "parallel=True kernel called during background compilation aborts the process"
    )


def test_the_isa_variant_is_a_separate_axis_from_the_backend() -> None:
    """Which family runs, and which build of it, are two questions with two enums.

    The assertion on :class:`Backend`'s members is the load-bearing one. The
    accepted values of ``PANTR_BACKEND`` are exactly those names, lowercased, and
    they are user-facing the moment anyone writes one into a script -- so folding
    the ISA ladder in later as ``cpp_v3`` would break every such script. Splitting
    the axes now costs nothing and makes that impossible; this pins it.
    """
    assert [b.name for b in Backend] == ["PYTHON", "CPP"], (
        "PANTR_BACKEND's accepted values are the lowercased member names, and "
        "changing them breaks any script that sets the variable"
    )
    assert [v.name for v in IsaVariant] == ["BASELINE"], (
        "design/simd.md gates building an ISA variant on a measurement; a member "
        "here without a module to load would make available_isa_variants() lie"
    )
    assert available_isa_variants() == (IsaVariant.BASELINE,)
    assert active_isa_variant() is IsaVariant.BASELINE


def test_a_requested_choice_that_is_missing_raises_on_either_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The never-fall-back rule is one rule, and it already holds on the ISA axis.

    The backend half is covered below, through the entry points a caller reaches.
    The ISA half has no such entry point yet -- one variant exists and it is
    always available -- so the shared resolver is called directly with the empty
    availability list that an unbuilt variant will produce. That is the case
    ``design/simd.md`` says must fail rather than quietly load another module,
    and testing it now is what keeps the rule from being written twice and
    drifting when the ladder lands.
    """
    from pantr import _backend  # noqa: PLC0415

    monkeypatch.setenv("PANTR_ISA_VARIANT", "baseline")
    with pytest.raises(RuntimeError, match="does not fall back"):
        _backend._resolve_from_environment(
            "PANTR_ISA_VARIANT", IsaVariant, (), IsaVariant.BASELINE, "  nothing is built"
        )

    monkeypatch.setenv("PANTR_ISA_VARIANT", "x86_64_v3")
    with pytest.raises(ValueError, match="PANTR_ISA_VARIANT='x86_64_v3' is not one of"):
        _backend._resolve_from_environment(
            "PANTR_ISA_VARIANT",
            IsaVariant,
            available_isa_variants(),
            IsaVariant.BASELINE,
            "",
        )

    monkeypatch.delenv("PANTR_ISA_VARIANT")
    unset = _backend._resolve_from_environment(
        "PANTR_ISA_VARIANT", IsaVariant, available_isa_variants(), IsaVariant.BASELINE, ""
    )
    assert unset is IsaVariant.BASELINE, "an unset variable must mean the default"


def test_an_unavailable_backend_request_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit request for a missing backend raises rather than running the other.

    ``pantr._backend`` states this as the reason the selector exists: a silent
    downgrade makes every A/B measurement a measurement of the wrong thing, and the
    measurement is what the override is for. It is untestable as written on a
    machine where both backends exist, so the absence is simulated: the extension
    handle is removed and the two entry points are asked for it anyway.
    """
    from pantr import _backend  # noqa: PLC0415

    monkeypatch.setattr(_backend, "_CPP_AVAILABLE", False)

    assert _backend.available_backends() == (Backend.PYTHON,)
    with pytest.raises(RuntimeError, match="not available"):
        cardinal_bspline_core(Backend.CPP)
    with pytest.raises(RuntimeError, match="not available"), use_backend(Backend.CPP):
        pytest.fail("use_backend must refuse an unavailable backend before yielding")

    # And the Numba backend is still reachable, i.e. the guard rejects rather than
    # disabling the selector. Compared by identity against the kernel itself: that
    # the returned object is not None says nothing, since a function reference
    # never is.
    from pantr.basis._basis_core import (  # noqa: PLC0415
        _tabulate_cardinal_Bspline_basis_1D_core,
    )

    assert cardinal_bspline_core(Backend.PYTHON).parallel is (
        _tabulate_cardinal_Bspline_basis_1D_core
    )


def test_overlapping_use_backend_blocks_in_two_threads_do_not_leak(cpp_backend: None) -> None:
    """Two threads whose ``use_backend`` blocks overlap both restore what they found.

    The lost update, forced deterministically with events rather than sleeps: A
    enters, B enters while A is inside, A exits, B exits. Against a plain module
    global, B's saved "previous" is A's override rather than the process default,
    so B restores the process to A's value on the way out and it stays there --
    for the rest of the process, with nothing left in scope to put it back.

    That is not a race in the sense of a rare interleaving. It is the *only*
    outcome of this ordering, which is why the test is deterministic.

    The backend the two threads select is whichever one the process is *not*
    already on, and the assertion is against the ambient value rather than a named
    one. Both matter: this file is run under ``PANTR_BACKEND=python`` and under
    ``PANTR_BACKEND=cpp``, and a fixed backend makes the test assert nothing under
    one of the two.
    """
    ambient = active_backend()
    other = Backend.PYTHON if ambient is Backend.CPP else Backend.CPP
    entered_a = threading.Event()
    entered_b = threading.Event()
    exited_a = threading.Event()
    timeout = 10.0

    def first() -> None:
        with use_backend(other):
            entered_a.set()
            assert entered_b.wait(timeout), "the second thread never entered its block"
        exited_a.set()

    def second() -> None:
        assert entered_a.wait(timeout), "the first thread never entered its block"
        with use_backend(other):
            entered_b.set()
            assert exited_a.wait(timeout), "the first thread never left its block"

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=timeout)
        assert not thread.is_alive(), "a thread did not finish; the events deadlocked"

    assert active_backend() is ambient, (
        f"two use_backend blocks that overlapped have left the process on "
        f"{active_backend().name} rather than the {ambient.name} it started on, so "
        f"every later call in this process runs a kernel nobody asked for"
    )


def test_use_backend_does_not_reach_into_another_thread(cpp_backend: None) -> None:
    """A thread started inside a ``use_backend`` block runs on the process default.

    The other half, and the one that decides an A/B measurement. The binding
    releases the GIL precisely so a caller can thread at the Python level, so
    "one thread is inside an override while another is working" is the shape this
    module invites rather than a pathological one. Against a module global the
    worker silently switches backend mid-flight, which is exactly the silent
    downgrade ``pantr._backend`` exists to prevent -- only in the other
    direction, and without an exception to notice it by.

    The direction asserted here is the deliberate one: an override is scoped to
    the thread (and the task) that took it, so a thread inherits the process
    default rather than its spawner's override. :func:`use_backend` says so.

    The override is whichever backend the process is *not* already on, so that
    the two values being compared always differ. Naming a fixed backend would
    make this assert nothing whenever ``PANTR_BACKEND`` happened to select it.
    """
    ambient = active_backend()
    other = Backend.PYTHON if ambient is Backend.CPP else Backend.CPP
    observed: list[Backend] = []

    with use_backend(other):
        assert active_backend() is other, "the override did not take in its own thread"
        worker = threading.Thread(target=lambda: observed.append(active_backend()))
        worker.start()
        worker.join(timeout=10.0)
        assert not worker.is_alive(), "the observing thread did not finish"

    assert observed == [ambient], (
        f"a thread started inside a use_backend({other.name}) block observed "
        f"{[b.name for b in observed]} rather than the process default "
        f"{ambient.name}, so the selection is process-wide rather than scoped to "
        f"the block that took it"
    )


# The quad catalogue is five accessors rather than one record, and the tests below
# are what keeps it that way. Identity is the assertion throughout, for the reason
# the test above gives: that a returned object is not None says nothing about a
# function reference, and identity also fails if the five are ever re-bundled into
# a record, since a record is not the function it carries.
_QUAD_ACCESSORS = (
    ("gauss_legendre", gauss_legendre_kernel),
    ("lambert_w", lambert_w_kernel),
    ("tanh_sinh", tanh_sinh_kernel),
    ("trapezoidal", trapezoidal_kernel),
    ("chebyshev_nodes", chebyshev_nodes_kernel),
)
"""The quad catalogue's five entry points, named for the failure message."""


@pytest.mark.parametrize(("name", "accessor"), _QUAD_ACCESSORS, ids=[n for n, _ in _QUAD_ACCESSORS])
def test_a_quad_accessor_hands_back_the_python_kernel_itself(
    name: str, accessor: Callable[[Backend | None], object]
) -> None:
    """Each quad accessor returns the Python kernel, by identity, not a wrapper.

    Args:
        name (str): The kernel's name, for the failure message.
        accessor (Callable[[Backend | None], object]): The catalogue entry point.
    """
    from pantr.quad import _rules_core  # noqa: PLC0415

    expected = {
        "gauss_legendre": _rules_core._gauss_legendre_symmetric_core,
        "lambert_w": _rules_core._lambert_w_principal_core,
        "tanh_sinh": _rules_core._generate_tanh_sinh_core,
        "trapezoidal": _rules_core._trapezoidal_core,
        "chebyshev_nodes": _rules_core._modified_chebyshev_nodes_core,
    }[name]
    assert accessor(Backend.PYTHON) is expected, (
        f"{name}: the catalogue returned something other than the kernel itself"
    )


@pytest.mark.parametrize(("name", "accessor"), _QUAD_ACCESSORS, ids=[n for n, _ in _QUAD_ACCESSORS])
def test_a_quad_accessor_hands_back_the_cpp_adapter_itself(
    name: str, accessor: Callable[[Backend | None], object], cpp_backend: None
) -> None:
    """Each quad accessor returns the C++ adapter, by identity, not a wrapper.

    Args:
        name (str): The kernel's name, for the failure message.
        accessor (Callable[[Backend | None], object]): The catalogue entry point.
        cpp_backend (None): Requires the compiled extension.
    """
    from pantr.quad import _quad_backend  # noqa: PLC0415

    expected = {
        "gauss_legendre": _quad_backend._cpp_gauss_legendre,
        "lambert_w": _quad_backend._cpp_lambert_w,
        "tanh_sinh": _quad_backend._cpp_tanh_sinh,
        "trapezoidal": _quad_backend._cpp_trapezoidal,
        "chebyshev_nodes": _quad_backend._cpp_chebyshev_nodes,
    }[name]
    assert accessor(Backend.CPP) is expected, (
        f"{name}: the catalogue returned something other than the adapter itself"
    )


def test_no_quad_accessor_falls_back_when_the_extension_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All five quad accessors refuse an explicit C++ request, not just the first.

    The never-fall-back rule is applied once, in the catalogue's ``_select``, so
    the five cannot drift apart on it. This asserts the property they are meant to
    share rather than the helper's existence, which would pass just as happily if
    someone reinstated five separate copies of the check.
    """
    from pantr import _backend  # noqa: PLC0415

    monkeypatch.setattr(_backend, "_CPP_AVAILABLE", False)

    for name, accessor in _QUAD_ACCESSORS:
        try:
            accessor(Backend.CPP)
        except RuntimeError as exc:
            assert "not available" in str(exc), f"{name}: raised, but not for that reason: {exc}"
        else:
            pytest.fail(f"{name}: an explicit CPP request was served without the extension")


_CHANGE_BASIS_ACCESSORS = (
    ("lagrange_to_bernstein", _cb_backend.lagrange_to_bernstein_kernel),
    ("bernstein_to_lagrange", _cb_backend.bernstein_to_lagrange_kernel),
    ("bernstein_to_cardinal", _cb_backend.bernstein_to_cardinal_kernel),
    ("cardinal_to_bernstein", _cb_backend.cardinal_to_bernstein_kernel),
    ("legendre_to_cardinal", _cb_backend.legendre_to_cardinal_kernel),
    ("cardinal_to_legendre", _cb_backend.cardinal_to_legendre_kernel),
    ("cardinal_dual_legendre_coeffs", _cb_backend.cardinal_dual_legendre_coeffs_kernel),
    ("monomial_to_bernstein", _cb_backend.monomial_to_bernstein_kernel),
)
"""The change_basis catalogue's eight entry points, named for the failure message."""


@pytest.mark.parametrize(
    ("name", "accessor"),
    _CHANGE_BASIS_ACCESSORS,
    ids=[n for n, _ in _CHANGE_BASIS_ACCESSORS],
)
def test_a_change_basis_accessor_hands_back_the_python_kernel_itself(
    name: str, accessor: Callable[[Backend | None], object]
) -> None:
    """Each change_basis accessor returns the Python kernel, by identity.

    Args:
        name (str): The kernel's name, for the failure message.
        accessor (Callable[[Backend | None], object]): The catalogue entry point.
    """
    from pantr.change_basis import _change_basis_core  # noqa: PLC0415

    expected = getattr(_change_basis_core, f"_{name}_core")
    assert accessor(Backend.PYTHON) is expected, (
        f"{name}: the catalogue returned something other than the kernel itself"
    )


@pytest.mark.parametrize(
    ("name", "accessor"),
    _CHANGE_BASIS_ACCESSORS,
    ids=[n for n, _ in _CHANGE_BASIS_ACCESSORS],
)
def test_a_change_basis_accessor_hands_back_the_same_cpp_adapter_every_call(
    name: str, accessor: Callable[[Backend | None], object], cpp_backend: None
) -> None:
    """Each accessor returns the one C++ adapter, by identity, on every call.

    Identity rather than equality, and it is not pedantry. Seven of these adapters
    are produced by a factory that closes over a binding name, so building them
    inside the accessor would hand back a **fresh closure per call** -- correct
    output, a new object every time, and a small allocation on a path that
    ``pantr.bspline``'s extraction calls in a loop. They are module-level constants
    for that reason, and this is the test that keeps them so.

    Args:
        name (str): The kernel's name, for the failure message.
        accessor (Callable[[Backend | None], object]): The catalogue entry point.
        cpp_backend (None): Requires the compiled extension.
    """
    del cpp_backend
    from pantr.change_basis import _change_basis_backend  # noqa: PLC0415

    expected = getattr(_change_basis_backend, f"_cpp_{name}")
    assert accessor(Backend.CPP) is expected, (
        f"{name}: the catalogue returned something other than the adapter itself"
    )
    assert accessor(Backend.CPP) is accessor(Backend.CPP), (
        f"{name}: two calls returned two different objects, so the adapter is being "
        f"rebuilt per call"
    )


def test_no_change_basis_accessor_falls_back_when_the_extension_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All eight accessors refuse an explicit C++ request, not just the first.

    The never-fall-back rule is applied once, in the catalogue's ``_select``, and
    this asserts the property the eight are meant to share rather than the
    helper's existence.
    """
    from pantr import _backend  # noqa: PLC0415

    monkeypatch.setattr(_backend, "_CPP_AVAILABLE", False)
    for name, accessor in _CHANGE_BASIS_ACCESSORS:
        with pytest.raises(RuntimeError, match="not available"):
            accessor(Backend.CPP)
        assert accessor(Backend.PYTHON) is not None, (
            f"{name}: the Python backend must stay available when the extension is not"
        )
