"""The Layer 3 kernels of ``pantr.bspline._extraction_kernels`` state their preconditions.

Sibling of :mod:`tests.test_kernel_preconditions` (issue 460 / PR 274d6dc), applying the
same check to the largest remaining module rather than editing that file, since a second
in-flight branch is doing the same for a different module and the two would collide on
one file. See that module's docstring for why the check is *that a precondition is
written down*, not *what happens when it is broken* -- the behaviour on out-of-contract
input stays deliberately unspecified here for the same reason.

This module's kernels are read against shape relations nothing in their signature
enforces (``apply_kron_2d`` reshapes ``v``/``out``/``scratch`` against sizes computed from
``M_0``/``M_1``, not from ``v``/``out`` themselves), so the precondition is a set of
per-array length relations rather than a single scalar length -- what is asserted here is
that each kernel's own docstring states its relation, keeps the Layer 3 disclaimer, and
says a violation is unspecified.

The kernel list is enumerated from the module's own source with :mod:`ast` -- a
``@nb_jit`` dispatcher collapses to a plain function under ``NUMBA_DISABLE_JIT=1``, so
nothing distinguishes a kernel from a helper by introspecting the live module -- and
cross-checked against an explicit expected set below, so a new kernel added to the module
without a corresponding entry here fails this suite rather than passing silently.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from pantr.bspline import _extraction_kernels as extraction_core
from tests.test_kernel_preconditions import _docstring_of

# One marker substring per kernel, taken verbatim from the ``Precondition:`` sentence
# this ticket added to its docstring. A kernel sharing a shape with its siblings (the
# ``_1d``/``_2d``/``_3d`` and ``_many_`` families) may state the reasoning once in the
# module docstring and point to it, but each kernel's own marker below is still unique to
# that kernel's own array names and dimension, so a copy-paste onto the wrong kernel
# fails this test rather than passing by matching a generic phrase.
_EXPECTED_MARKERS: dict[str, str] = {
    "apply_kron_1d": "M_0.shape[0] >= out.shape[0]",
    "apply_kron_2d": "``M_0.shape[1] * M_1.shape[1]``",
    "apply_kron_3d": "M_0.shape[1] * M_1.shape[1] * M_2.shape[1]",
    "apply_kron_T_1d": "v.shape[0] >= M_0.shape[0]",
    "apply_kron_T_2d": "``M_0.shape[0] * M_1.shape[0]``",
    "apply_kron_T_3d": "M_0.shape[0] * M_1.shape[0] * M_2.shape[0]",
    "apply_kron_MT_K_M_1d": "K.shape >= (M_0.shape[0], M_0.shape[0])",
    "apply_kron_MT_K_M_2d": "(M_0.shape[0] * M_1.shape[0]) ** 2",
    "apply_kron_MT_K_M_3d": "(M_0.shape[0] * M_1.shape[0] * M_2.shape[0]) ** 2",
    "apply_kron_M_K_MT_1d": "K.shape >= (M_0.shape[1], M_0.shape[1])",
    "apply_kron_M_K_MT_2d": "(M_0.shape[1] * M_1.shape[1]) ** 2",
    "apply_kron_M_K_MT_3d": "(M_0.shape[1] * M_1.shape[1] * M_2.shape[1]) ** 2",
    "apply_kron_apply_many_1d": ":func:`apply_kron_1d`'s own precondition",
    "apply_kron_apply_many_2d": ":func:`apply_kron_2d`'s own precondition",
    "apply_kron_apply_many_3d": ":func:`apply_kron_3d`'s own precondition",
    "apply_kron_apply_T_many_1d": ":func:`apply_kron_T_1d`'s own precondition",
    "apply_kron_apply_T_many_2d": ":func:`apply_kron_T_2d`'s own precondition",
    "apply_kron_apply_T_many_3d": ":func:`apply_kron_T_3d`'s own precondition",
    "apply_kron_MT_K_M_many_1d": ":func:`apply_kron_MT_K_M_1d`'s own precondition",
    "apply_kron_MT_K_M_many_2d": ":func:`apply_kron_MT_K_M_2d`'s own precondition",
    "apply_kron_MT_K_M_many_3d": ":func:`apply_kron_MT_K_M_3d`'s own precondition",
    "apply_kron_M_K_MT_many_1d": ":func:`apply_kron_M_K_MT_1d`'s own precondition",
    "apply_kron_M_K_MT_many_2d": ":func:`apply_kron_M_K_MT_2d`'s own precondition",
    "apply_kron_M_K_MT_many_3d": ":func:`apply_kron_M_K_MT_3d`'s own precondition",
}
"""Expected precondition marker per kernel. Also the module's cross-check set (AC1):
a kernel added to ``_extraction_kernels`` without an entry here fails
:func:`test_every_kernel_has_an_expected_marker` before it can fail anything else.
The ``_2d`` markers carry their closing backticks so that none is a substring of its
``_3d`` sibling's, which would let a ``_3d`` note pasted onto a ``_2d`` kernel pass.
"""

_IDENTITY_BRANCH_MARKERS: dict[str, str] = {
    "apply_kron_MT_K_M_1d": "out.shape >= (M_0.shape[0], M_0.shape[0])",
    "apply_kron_M_K_MT_1d": "out.shape >= (M_0.shape[1], M_0.shape[1])",
}
"""The identity branch of the two ``_1d`` bilateral kernels copies ``K`` into ``out``
over ``K``'s extent, not over ``out``'s non-identity extent, so ``out`` needs the other
axis of ``M_0`` there. The two coincide only for a square identity operator.
"""


@pytest.mark.parametrize(
    "name", sorted(_IDENTITY_BRANCH_MARKERS), ids=sorted(_IDENTITY_BRANCH_MARKERS)
)
def test_bilateral_1d_kernel_states_its_identity_branch_extent(name: str) -> None:
    """A ``_1d`` bilateral kernel states the ``out`` extent its identity branch writes."""
    doc = _docstring_of(getattr(extraction_core, name))
    marker = _IDENTITY_BRANCH_MARKERS[name]
    assert marker in doc, f"{name} does not state its identity-branch extent ({marker!r})"


def _kernel_names_from_source() -> frozenset[str]:
    """Enumerate every ``@nb_jit``-decorated module-level function by parsing the source.

    Introspecting the live module cannot distinguish a kernel from a helper once
    ``NUMBA_DISABLE_JIT=1`` collapses every dispatcher to a plain function, so this reads
    the module's own source with :mod:`ast` instead of importing and filtering it.

    Returns:
        frozenset[str]: Names of every module-level ``def`` decorated with ``nb_jit``.
    """
    source = inspect.getsource(extraction_core)
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(target, ast.Name) and target.id == "nb_jit":
                names.add(node.name)
                break
    return frozenset(names)


def test_every_kernel_has_an_expected_marker() -> None:
    """The module's kernels, enumerated from source, match the expected set exactly.

    A kernel present in one set and not the other means either a new kernel was added
    without a precondition entry here, or an entry here refers to a kernel that no
    longer exists -- both are reported by name rather than passing silently.
    """
    found = _kernel_names_from_source()
    expected = frozenset(_EXPECTED_MARKERS)
    missing_entries = found - expected
    stale_entries = expected - found
    assert not missing_entries, (
        f"kernel(s) added to _extraction_kernels.py with no precondition entry in "
        f"this test's _EXPECTED_MARKERS: {sorted(missing_entries)}"
    )
    assert not stale_entries, (
        f"_EXPECTED_MARKERS names kernel(s) no longer in _extraction_kernels.py: "
        f"{sorted(stale_entries)}"
    )


@pytest.mark.parametrize("name", sorted(_EXPECTED_MARKERS), ids=sorted(_EXPECTED_MARKERS))
def test_kernel_states_its_precondition(name: str) -> None:
    """Each kernel's docstring contains its expected precondition marker."""
    kernel = getattr(extraction_core, name)
    doc = _docstring_of(kernel)
    marker = _EXPECTED_MARKERS[name]
    assert marker in doc, f"{name} does not state its expected precondition ({marker!r})"


@pytest.mark.parametrize("name", sorted(_EXPECTED_MARKERS), ids=sorted(_EXPECTED_MARKERS))
def test_kernel_still_disclaims_validation(name: str) -> None:
    """Stating the precondition does not replace the Layer 3 disclaimer."""
    kernel = getattr(extraction_core, name)
    doc = _docstring_of(kernel)
    assert "no validation performed" in doc, f"{name} lost the Layer 3 disclaimer"


@pytest.mark.parametrize("name", sorted(_EXPECTED_MARKERS), ids=sorted(_EXPECTED_MARKERS))
def test_kernel_states_violation_is_unspecified(name: str) -> None:
    """Each kernel's docstring says behaviour on a violated precondition is unspecified."""
    kernel = getattr(extraction_core, name)
    doc = _docstring_of(kernel)
    assert "unspecified" in doc, f"{name} does not say violating its precondition is unspecified"
