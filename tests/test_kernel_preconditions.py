"""The Layer 3 kernels of ``pantr.bezier._root_finding_core`` state their preconditions.

A Layer 3 kernel validates nothing, by design and by this repository's stated layering.
That disclaimer is only meaningful next to the precondition it is disclaiming against:
*"inputs are assumed to be correct"* names no minimum length, so a direct caller has
nothing to check its input against, and the adversarial sweep has nothing to grade
against either.

**What this module asserts is that the precondition is written down, not what happens
when it is broken.** The behaviour on out-of-contract input is deliberately unspecified
-- three of these kernels read past the array and four happen not to -- and a test that
pinned today's ``IndexError`` would freeze something the library does not promise, and
would fail the moment a build turns bounds checking off.

It is also what keeps ``adversarial_sweep``'s ``out_of_contract`` flag honest. That flag
tells the sweep not to grade a case, on the grounds that the entry point declares the
input illegal. Used where no such declaration exists it would be a way to silence a
finding; this module is the check that the declaration is really there.
"""

from __future__ import annotations

import pytest

from pantr.bezier import _root_finding_core as core

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
