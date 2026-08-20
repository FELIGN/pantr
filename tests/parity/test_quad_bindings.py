"""What the quad bindings refuse, and what they must never accept silently.

`cpp/bindings/quad.cpp` is **Layer 2's C++ half**: the kernels behind it validate
nothing, so every precondition they rely on is established here or not at all.
`tests/parity/test_basis_cardinal_bspline.py` carries the equivalent battery for
the one binding that predates these, and the reason it exists is recorded in
`cpp/bindings/basis.cpp`: a missing ``.noconvert()`` once shipped, and it produced
no exception, no warning, and a plausible-looking array of the right shape and
dtype that the kernel had never written to.

Every test here is written against that failure shape rather than against a
message. What matters is not that a particular `ValueError` is raised but that the
call **cannot succeed while doing the wrong thing**, so each one either asserts an
exception or asserts that the caller's own array came back correct.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pantr.quad._rules import _tanh_sinh_min_gap

_MIN_GAP: float = float(_tanh_sinh_min_gap(np.float64))
"""A threshold the kernel will accept, so a test can vary one argument at a time."""


def _extension() -> Any:
    """Import the compiled extension.

    Returns:
        Any: The ``pantr._pantr_cpp`` module.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (the extension is optional)

    return _pantr_cpp


def _two_output_kernels() -> list[tuple[str, Any]]:
    """List the bindings that write two arrays, which are the ones that can alias.

    Returns:
        list[tuple[str, Any]]: Name and a callable taking ``(n, nodes, weights)``.
    """
    extension = _extension()
    return [
        ("gauss_legendre_symmetric", extension.gauss_legendre_symmetric),
        ("trapezoidal", extension.trapezoidal),
        (
            "generate_tanh_sinh",
            lambda n, nodes, weights: extension.generate_tanh_sinh(n, _MIN_GAP, nodes, weights),
        ),
    ]


@pytest.mark.parametrize("count", [0, -1, -1000])
def test_a_count_below_one_is_refused(count: int, cpp_backend: None) -> None:
    """No kernel accepts a count it cannot express, and neither branch reaches the kernel.

    Zero and negative fail differently and both matter: the parameter is
    ``unsigned``, so nanobind's own caster rejects a negative before any of pantr's
    code runs, while zero passes the caster and is refused in the body. The kernel
    behind it would index from a loop bound of ``SIZE_MAX`` for the first and write
    nothing for the second.

    Args:
        count (int): The rejected count.
        cpp_backend (None): Requires the compiled extension.
    """
    nodes = np.empty(4, dtype=np.float64)
    weights = np.empty(4, dtype=np.float64)
    for name, call in _two_output_kernels():
        with pytest.raises((ValueError, TypeError)):
            call(count, nodes, weights)
        assert name  # every kernel in the family, not just the first


@pytest.mark.parametrize("shortfall", [1, 3])
def test_an_output_too_small_for_the_count_is_refused(shortfall: int, cpp_backend: None) -> None:
    """An undersized output is refused rather than overrun.

    This is a memory-safety precondition, not an accuracy one: nothing in the
    kernels bounds-checks a write, so an output shorter than the count reaches
    undefined behaviour. `cpp/bindings/basis.cpp` records that the equivalent
    mistake there corrupted the heap.

    Args:
        shortfall (int): How many elements short the output is.
        cpp_backend (None): Requires the compiled extension.
    """
    count = 8
    for name, call in _two_output_kernels():
        short = np.empty(count - shortfall, dtype=np.float64)
        full = np.empty(count, dtype=np.float64)
        with pytest.raises(ValueError, match="size"):
            call(count, short, full)
        with pytest.raises(ValueError, match="size"):
            call(count, full, short)
        assert name  # every kernel in the family, not just the first


def test_two_outputs_that_overlap_are_refused(cpp_backend: None) -> None:
    """An aliased output pair is refused, and that is a correctness guard.

    Measured before the guard existed: passing one array as both outputs of
    ``gauss_legendre_symmetric`` returned the **weights** in it, silently. Each
    output is written independently, so the second write overwrites the first and
    the result is a plausible array holding the wrong quantity.

    Both shapes are checked. Passing the same object twice is the obvious mistake;
    two overlapping *views* onto one buffer alias just as destructively and would
    slip past an identity test.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    for name, call in _two_output_kernels():
        same = np.empty(6, dtype=np.float64)
        with pytest.raises(ValueError, match="overlap"):
            call(6, same, same)

        buffer = np.empty(12, dtype=np.float64)
        with pytest.raises(ValueError, match="overlap"):
            call(6, buffer[:6], buffer[3:9])
        assert name  # every kernel in the family, not just the first


def test_disjoint_outputs_are_accepted_and_actually_written(cpp_backend: None) -> None:
    """The guard refuses only what it should, and the accepted call fills the caller's arrays.

    The other half of the aliasing test. A guard that rejected every call would
    pass the test above, and an ``out`` parameter that quietly fills a discarded
    temporary would pass both. The outputs are poisoned with NaN first, so an
    element the kernel failed to write is a NaN rather than a plausible zero.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    for name, call in _two_output_kernels():
        nodes = np.full(6, np.nan, dtype=np.float64)
        weights = np.full(6, np.nan, dtype=np.float64)
        call(6, nodes, weights)
        assert not np.isnan(nodes).any(), f"{name} left a node unwritten in the caller's array"
        assert not np.isnan(weights).any(), f"{name} left a weight unwritten"


def test_a_non_contiguous_output_is_refused_rather_than_silently_discarded(
    cpp_backend: None,
) -> None:
    """A strided output raises instead of being filled into a temporary and dropped.

    The failure `cpp/bindings/basis.cpp` records: without ``.noconvert()`` nanobind
    satisfies a ``c_contig`` output parameter by **converting** the argument into a
    contiguous temporary, filling that, and discarding it. No exception, and the
    caller's array comes back untouched.

    The test therefore asserts both halves: that it raises, and that the caller's
    array is still exactly what it was.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    for name, call in _two_output_kernels():
        strided = np.full(12, np.nan, dtype=np.float64)[::2]
        weights = np.empty(6, dtype=np.float64)
        before = strided.copy()
        with pytest.raises(TypeError):
            call(6, strided, weights)
        assert np.array_equal(strided, before, equal_nan=True), (
            f"{name} wrote into a non-contiguous output it was supposed to refuse"
        )


def test_a_float32_output_is_refused_by_the_double_only_kernels(cpp_backend: None) -> None:
    """The double-only kernels refuse a narrower output rather than converting into it.

    Four of the five quad kernels compute in ``double`` by design, and the Python
    layer narrows afterwards. A float32 output must therefore be a type error, not
    a silent conversion: converting would hand back a temporary the kernel filled
    and the caller never sees.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    for name, call in _two_output_kernels():
        nodes = np.empty(6, dtype=np.float32)
        weights = np.empty(6, dtype=np.float32)
        with pytest.raises(TypeError):
            call(6, nodes, weights)
        assert name  # every kernel in the family, not just the first


@pytest.mark.parametrize("bad_gap", [0.0, -1e-16, float("nan"), float("inf")])
def test_a_threshold_that_cannot_terminate_the_rule_is_refused(
    bad_gap: float, cpp_backend: None
) -> None:
    """tanh-sinh refuses a truncation threshold its own loop could never reach.

    ``gap < min_gap`` is what ends generation. A non-positive or non-finite
    threshold is never satisfied, so the rule would run to ``n`` and emit nodes
    that collapse onto an endpoint once mapped, which is precisely what a
    double-exponential rule exists not to do.

    Args:
        bad_gap (float): The rejected threshold.
        cpp_backend (None): Requires the compiled extension.
    """
    extension = _extension()
    nodes = np.empty(8, dtype=np.float64)
    weights = np.empty(8, dtype=np.float64)
    with pytest.raises(ValueError):
        extension.generate_tanh_sinh(8, bad_gap, nodes, weights)


@pytest.mark.parametrize("argument", [0.0, 1.0, 1.6, -5.0, float("nan")])
def test_lambert_w_refuses_an_argument_off_its_branch(argument: float, cpp_backend: None) -> None:
    """The binding validates a precondition its own kernel deliberately does not.

    This is the one deliberate asymmetry in the family, and it is the right way
    round. The Layer 3 kernels on both sides skip the check because their only
    caller cannot violate it; the binding is a public attribute of an importable
    module with no such guarantee. Below about 1.61 the branch-free start lands on
    the wrong branch and the result is a **wrong number rather than an absent
    one** -- the failure mode that most deserves a guard.

    Args:
        argument (float): An argument off the principal branch's usable range.
        cpp_backend (None): Requires the compiled extension.
    """
    with pytest.raises(ValueError):
        _extension().lambert_w_principal(argument)


@pytest.mark.parametrize("count", [0, 1])
def test_the_chebyshev_nodes_refuse_a_count_that_would_divide_by_zero(
    count: int, cpp_backend: None
) -> None:
    """One node is not a Chebyshev-Lobatto set, and ``n - 1`` is a divisor.

    Args:
        count (int): A count below the minimum of two.
        cpp_backend (None): Requires the compiled extension.
    """
    out = np.empty(max(count, 1), dtype=np.float64)
    with pytest.raises(ValueError):
        _extension().modified_chebyshev_nodes(count, out)
