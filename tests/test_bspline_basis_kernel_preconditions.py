"""The Layer 3 kernels of ``pantr.bspline._bspline_basis_kernels`` state their preconditions.

Sibling of ``tests/test_kernel_preconditions.py``, which applied the same check to
``pantr.bezier._root_finding_core`` (#412 / #460); kept in its own file rather than
extended into that one so a concurrent change to either module's kernel list cannot
collide on the same lines. Reuses that module's ``_docstring_of`` helper -- reaching
through Numba's dispatcher wrapper is the same problem in both files, and
``tests/parity/test_change_basis.py`` importing from
``tests.test_change_basis_domain`` is precedent for importing a sibling test module
directly rather than duplicating it.

**What this module asserts is that the precondition is written down, not what
happens when it is broken.** The 8 kernels are enumerated from the module's own
source (every module-level ``def`` decorated with ``@nb_jit``) rather than a
hand-written list, so a 9th kernel added later and left without a precondition
statement fails here instead of silently not being checked -- and rather than by
runtime type, because a Numba dispatcher is a ``CPUDispatcher`` under the ordinary
JIT and a plain function under ``NUMBA_DISABLE_JIT=1``
(``tests/parity/test_extraction_kernels.py`` notes the same collapse), so no
runtime attribute survives both configurations this suite is run under.
"""

from __future__ import annotations

import ast
import inspect
from types import ModuleType

import pytest

from pantr.bspline import _bspline_basis_kernels as _kernels_module
from tests.test_kernel_preconditions import _docstring_of


def _module_level_nb_jit_kernel_names(module: ModuleType) -> tuple[str, ...]:
    """Return the names of every module-level ``@nb_jit``-decorated function.

    Args:
        module (ModuleType): The module to scan; must expose retrievable source
            (i.e. back a real ``.py`` file, via :func:`inspect.getsource`).

    Returns:
        tuple[str, ...]: Names of the module-level functions decorated with
        ``@nb_jit``, in source order. A function decorated some other way, or a
        module-level function with no decorator at all (``_warmup_numba_functions``
        in this module), is not included.
    """
    tree = ast.parse(inspect.getsource(module))
    names = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for deco in node.decorator_list:
            target = deco.func if isinstance(deco, ast.Call) else deco
            if isinstance(target, ast.Name) and target.id == "nb_jit":
                names.append(node.name)
                break
    return tuple(names)


_KERNEL_NAMES = _module_level_nb_jit_kernel_names(_kernels_module)
_KERNELS = tuple(getattr(_kernels_module, name) for name in _KERNEL_NAMES)

_EXPECTED_KERNEL_NAMES = frozenset(
    {
        "_find_spans_and_first_basis",
        "_find_span_and_first_basis_point",
        "_basis_funcs_point",
        "_compute_basis_nurbs_book_impl",
        "_compute_basis_nurbs_book_serial_impl",
        "_basis_derivs_point",
        "_compute_basis_deriv_nurbs_book_impl",
        "_compute_basis_deriv_nurbs_book_serial_impl",
    }
)
"""The 8 kernels #474 covers. Not consulted to select which kernels are checked --
that is :func:`_module_level_nb_jit_kernel_names`'s job -- only to give a clear
failure (a set difference) if the module's kernel roster has moved since, rather
than a parametrize list silently growing or shrinking underneath the two tests
below.
"""


def test_the_enumerated_kernels_are_the_eight_this_module_covers() -> None:
    """The source-level enumeration finds exactly the 8 kernels #474 names.

    A kernel added to (or removed from) the module changes this set, which is
    exactly the signal a docs-only follow-up ticket like this one needs: the
    parametrized tests below only ever see what this enumeration finds, so a
    silently-added 9th kernel would otherwise just not be checked.
    """
    assert set(_KERNEL_NAMES) == _EXPECTED_KERNEL_NAMES


_PRECONDITION_MARKER = {
    "_find_spans_and_first_basis": "knots.size >= 2 * degree + 2",
    "_find_span_and_first_basis_point": "knots.size >= 2 * degree + 2",
    "_basis_funcs_point": "degree - 1 <= knot_id <= knots.size - 1 -",
    "_compute_basis_nurbs_book_impl": "knots.size >= 2 * degree + 2",
    "_compute_basis_nurbs_book_serial_impl": "knots.size >= 2 * degree + 2",
    "_basis_derivs_point": "degree - 1 <= knot_id <= knots.size - 1 -",
    "_compute_basis_deriv_nurbs_book_impl": "knots.size >= 2 * degree + 2",
    "_compute_basis_deriv_nurbs_book_serial_impl": "knots.size >= 2 * degree + 2",
}
"""Each kernel's precondition, restated as the literal substring its docstring must
carry, derived from that kernel's own index expressions (see the PR body's AC2
table for how each was checked against the compiled kernel). One entry per name in
:data:`_EXPECTED_KERNEL_NAMES`.
"""


def test_every_covered_kernel_has_a_registered_precondition_marker() -> None:
    """The marker table has one entry per kernel this module covers, no more, no less."""
    assert set(_PRECONDITION_MARKER) == _EXPECTED_KERNEL_NAMES


@pytest.mark.parametrize("kernel", _KERNELS, ids=_KERNEL_NAMES)
def test_a_kernel_states_its_precondition(kernel: object) -> None:
    """Each kernel's docstring states the precondition derived from its own reads."""
    name = getattr(kernel, "__name__", str(kernel))
    doc = _docstring_of(kernel)
    marker = _PRECONDITION_MARKER[name]
    assert marker in doc, f"{name} disclaims validation without stating: {marker!r}"


@pytest.mark.parametrize("kernel", _KERNELS, ids=_KERNEL_NAMES)
def test_a_kernel_still_disclaims_validation(kernel: object) -> None:
    """Stating the precondition does not replace the Layer 3 disclaimer."""
    doc = _docstring_of(kernel)
    assert "no validation performed" in doc, (
        f"{getattr(kernel, '__name__', kernel)} lost the Layer 3 disclaimer"
    )


@pytest.mark.parametrize("kernel", _KERNELS, ids=_KERNEL_NAMES)
def test_a_kernel_names_the_knot_vector_property_it_assumes(kernel: object) -> None:
    """Each kernel states that ``knots`` must be non-decreasing.

    Every kernel here either binary-searches the knot vector or relies on knot
    differences being non-negative, so the ordering is part of every contract. The
    assertion matches the stated precondition, whitespace-normalised, rather than the
    bare word, which two of these docstrings already used in an unrelated sentence.
    """
    doc = " ".join(_docstring_of(kernel).split())
    assert "including that ``knots`` is non-decreasing" in doc, (
        f"{getattr(kernel, '__name__', kernel)} does not state that the knot vector must be "
        "non-decreasing"
    )


@pytest.mark.parametrize("kernel", _KERNELS, ids=_KERNEL_NAMES)
def test_a_kernel_leaves_out_of_contract_behavior_unspecified(kernel: object) -> None:
    """Each kernel says behavior outside its precondition is unspecified, not today's.

    Pinning today's behaviour (some of these kernels raise just outside their
    precondition and others return silently) would freeze something the library does
    not promise, and would fail the moment a build turns bounds checking off.
    """
    doc = _docstring_of(kernel)
    assert "unspecified" in doc, (
        f"{getattr(kernel, '__name__', kernel)} states a precondition but pins, "
        "rather than disclaims, its out-of-contract behavior"
    )
