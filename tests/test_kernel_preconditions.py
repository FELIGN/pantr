"""Layer 3 kernels state the preconditions their disclaimer disclaims against.

A Layer 3 kernel validates nothing, by design and by this repository's stated layering.
That disclaimer is only meaningful next to the precondition it is disclaiming against:
*"inputs are assumed to be correct"* names no minimum length, so a direct caller has
nothing to check its input against, and the adversarial sweep has nothing to grade
against either.

**What this module asserts is that the precondition is written down, not what happens
when it is broken.** The behaviour on out-of-contract input is deliberately unspecified
-- some of these kernels read past the array and some happen not to -- and a test that
pinned today's exception would freeze something the library does not promise, and would
fail the moment a build turns bounds checking off.

It is also what keeps ``adversarial_sweep``'s ``out_of_contract`` flag honest. That flag
tells the sweep not to grade a case, on the grounds that the entry point declares the
input illegal. Used where no such declaration exists it would be a way to silence a
finding; this module is the check that the declaration is really there.

Two modules are covered so far: ``pantr.bezier._root_finding_core`` (the seven
coefficient-array kernels, enumerated by hand since the file also has three
scalar-only predicates it would be false to give a length precondition) and
``pantr.bspline._bspline_roots_core`` (all twelve of its kernels, enumerated by
introspecting the module for Numba dispatchers it defines itself, so a new kernel
added without a table entry fails loudly rather than being silently skipped).
"""

from __future__ import annotations

import types

import pytest

from pantr.bezier import _root_finding_core as core
from pantr.bspline import _bspline_roots_core as bspline_core

_COEFFICIENT_KERNELS = (
    core._de_casteljau_eval_scalar,
    core._restrict_scalar,
    core._de_casteljau_eval_and_deriv_scalar,
    core._subdivide_scalar,
    core._count_sign_changes,
    core._clip_hull_to_zero,
    core._newton_polish_scalar,
)
"""Every kernel in the module that takes a Bernstein coefficient array.

The three that do not -- ``_have_same_sign``, ``_have_opposite_signs`` and
``_spans_zero`` -- are deliberately absent. They take two floats, every float is in
contract for them (their docstrings already reason about NaN and infinity), so there is
no minimum to state and inventing one would be a false claim.
"""


def _docstring_of(kernel: object) -> str:
    """Return a kernel's docstring, reaching through Numba's dispatcher wrapper.

    Args:
        kernel (object): A ``@nb_jit``-decorated kernel, or the plain function when the
            JIT is disabled.

    Returns:
        str: The docstring, never ``None``: an undocumented kernel is itself a failure
            this module should report rather than skip.
    """
    doc: object = getattr(kernel, "__doc__", None)
    if doc is None:
        py_func: object = getattr(kernel, "py_func", None)
        doc = getattr(py_func, "__doc__", None)
    assert isinstance(doc, str), f"{kernel} carries no docstring at all"
    return doc


@pytest.mark.parametrize(
    "kernel", _COEFFICIENT_KERNELS, ids=lambda k: getattr(k, "__name__", str(k))
)
def test_a_coefficient_kernel_states_its_minimum_length(kernel: object) -> None:
    """Each coefficient-array kernel names ``len(coeff) >= 1`` in its docstring."""
    doc = _docstring_of(kernel)
    assert "len(coeff) >= 1" in doc, (
        f"{getattr(kernel, '__name__', kernel)} disclaims validation without stating the "
        "minimum coefficient-array length it assumes"
    )


@pytest.mark.parametrize(
    "kernel", _COEFFICIENT_KERNELS, ids=lambda k: getattr(k, "__name__", str(k))
)
def test_a_coefficient_kernel_still_disclaims_validation(kernel: object) -> None:
    """Stating the precondition does not replace the layering disclaimer."""
    doc = _docstring_of(kernel)
    assert "no validation performed" in doc, (
        f"{getattr(kernel, '__name__', kernel)} lost the Layer 3 disclaimer"
    )


def _kernel_name(kernel: object) -> str:
    """Return a dispatcher's own function name, reaching through ``py_func``.

    Args:
        kernel (object): A ``@nb_jit``-decorated kernel.

    Returns:
        str: The wrapped function's ``__name__``.
    """
    py_func: object = getattr(kernel, "py_func", None)
    name: object = getattr(py_func, "__name__", None)
    assert isinstance(name, str), f"{kernel} carries no py_func.__name__"
    return name


def _module_kernels(module: types.ModuleType) -> tuple[object, ...]:
    """Enumerate every Numba dispatcher a module defines at module level.

    Reading the module itself, rather than a hand-written list, is what makes a
    kernel added later without a stated precondition fail this suite instead of
    silently going unchecked: the source of truth is what the module actually
    exports as a JIT-compiled function, not a list a new kernel can fall behind.

    Args:
        module (types.ModuleType): Module to introspect.

    Returns:
        tuple[object, ...]: Every ``@nb_jit``-decorated function the module defines
        itself, sorted by name for a deterministic parametrization order. A plain
        helper such as ``_warmup_numba_functions``, and anything imported from
        elsewhere, are excluded because neither carries Numba's ``py_func``
        attribute pointing back into this module.
    """
    kernels = [
        value
        for value in vars(module).values()
        if getattr(getattr(value, "py_func", None), "__module__", None) == module.__name__
    ]
    assert kernels, (
        f"no Numba dispatcher was found in {module.__name__}; this enumeration relies "
        "on the 'py_func' attribute Numba attaches, which is absent when NUMBA_DISABLE_JIT "
        "is set -- run this file with the JIT enabled"
    )
    return tuple(sorted(kernels, key=_kernel_name))


_BSPLINE_ROOT_KERNELS = _module_kernels(bspline_core)

_BSPLINE_ROOT_PRECONDITIONS: dict[str, tuple[str, ...]] = {
    "_knot_average": ("len(knots) >= index + degree + 1",),
    "_zero_index": ("len(coeffs) >= num_coeffs",),
    "_is_zero_index": ("len(coeffs) >= num_coeffs",),
    "_deboor_point": (
        "len(coeffs) >= span + 1",
        "len(knots) >= span + degree + 1",
    ),
    "_span_at": ("len(knots) >= num_coeffs",),
    "_residual_at": (
        "len(coeffs) >= num_coeffs",
        "len(knots) >= num_coeffs + degree",
    ),
    "_insert_knot": (
        "len(coeffs) >= num_coeffs + 1",
        "len(knots) >= num_coeffs + degree + 2",
    ),
    "_split_at_root": (
        "knots[len(knots) - 1]",
        "len(knots) >= len(coeffs) + degree + 1",
    ),
    "_drop_window_head": (
        "len(coeffs) >= num_coeffs",
        "len(knots) >= num_coeffs + degree + 1",
    ),
    "_track_zero": ("len(knots) >= len(coeffs)",),
    "_morken_reimers_roots": (
        "len(knots) >= coeffs.shape[0] + degree + 1",
        "open (clamped)",
    ),
    "_merge_roots": ("len(radii) >= len(roots)",),
}
"""Every kernel :func:`_module_kernels` finds in ``_bspline_roots_core``, mapped to the
substring(s) its ``Note:`` block must contain.

All twelve of the module's kernels take at least one array, so none is left out the way
the three scalar predicates are for the Bézier module: there is no kernel here for which
a length precondition would be a false claim.
"""


def test_a_bspline_root_kernel_is_enumerated_with_a_precondition() -> None:
    """Every kernel the module defines has a corresponding precondition-table entry.

    A kernel added to the module and left out of the table fails here, and a stale
    table entry for a kernel that no longer exists fails here too -- the two sets
    are checked against each other rather than one being assumed to cover the other.
    """
    found = {_kernel_name(k) for k in _BSPLINE_ROOT_KERNELS}
    tabulated = set(_BSPLINE_ROOT_PRECONDITIONS)
    assert found - tabulated == set(), (
        f"kernel(s) added to the module without a stated precondition: {sorted(found - tabulated)}"
    )
    assert tabulated - found == set(), (
        f"precondition table names kernel(s) no longer in the module: {sorted(tabulated - found)}"
    )


@pytest.mark.parametrize("kernel", _BSPLINE_ROOT_KERNELS, ids=_kernel_name)
def test_a_bspline_root_kernel_states_its_precondition(kernel: object) -> None:
    """Each kernel's docstring names the array-length relation(s) it assumes."""
    name = _kernel_name(kernel)
    doc = _docstring_of(kernel)
    for substring in _BSPLINE_ROOT_PRECONDITIONS[name]:
        assert substring in doc, f"{name} disclaims validation without stating '{substring}'"


@pytest.mark.parametrize("kernel", _BSPLINE_ROOT_KERNELS, ids=_kernel_name)
def test_a_bspline_root_kernel_says_out_of_contract_behavior_is_unspecified(
    kernel: object,
) -> None:
    """Each kernel's ``Note:`` says behaviour on out-of-contract input is unspecified."""
    doc = _docstring_of(kernel)
    name = _kernel_name(kernel)
    assert "unspecified" in doc, (
        f"{name} does not say that its behaviour outside its precondition is unspecified"
    )


@pytest.mark.parametrize("kernel", _BSPLINE_ROOT_KERNELS, ids=_kernel_name)
def test_a_bspline_root_kernel_still_disclaims_validation(kernel: object) -> None:
    """Stating the precondition does not replace the layering disclaimer."""
    doc = _docstring_of(kernel)
    name = _kernel_name(kernel)
    assert "no validation performed" in doc, f"{name} lost the Layer 3 disclaimer"
