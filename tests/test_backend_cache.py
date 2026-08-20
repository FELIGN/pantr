"""Memoized results must not be shared between backends.

The bug this file pins. Six memoized helpers in ``change_basis`` and
``bezier/_bezier_degree`` reach a backend-dispatched kernel, and every one of
them was keyed on its arguments alone. So the first backend to populate an entry
served every later caller, whichever backend that caller had selected.

Two consequences, and the first is the reason these tests count cache entries
rather than compare values.

**A parity test through such a helper compares an object with itself.** On
``proto/cpp`` today the two backends agree bit for bit on the cardinal B-spline
kernel, so the shared entry holds the *right* numbers and no assertion fails:
the comparison is vacuous rather than wrong, which is the harder failure to
notice. A test written against values would pass on the broken code. Counting
entries discriminates whatever the values are, which is what makes these
regressions rather than illustrations.

**The threading case is a wrong answer, not a vacuous test.** :func:`use_backend`
is scoped per thread on purpose and the extension releases the GIL to invite
threading, so a thread inside a block populates entries that a thread outside it
then reads. Note that clearing every cache when a block opens and closes does
*not* fix this: the entry is fresh and still computed under whichever backend
asked first. Keying is what fixes it, and
:func:`test_a_cache_does_not_leak_across_threads` is the test that tells the two
remedies apart.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import pantr.bezier._bezier_degree as bezier_degree
from pantr import change_basis
from pantr._backend import (
    Backend,
    active_backend,
    available_backends,
    backend_keyed_cache,
    use_backend,
)
from pantr.basis import LagrangeVariant

_BOTH_BACKENDS = pytest.mark.skipif(
    len(available_backends()) < 2,
    reason="needs both backends; the C++ extension is not built into this install",
)


def _degree() -> int:
    """Get a degree large enough that the matrices are not trivially equal.

    Returns:
        int: A degree the memoized builders all accept.
    """
    return 9


# The six memoized helpers that reach a dispatched kernel, each with a call that
# exercises it. Established by instrumenting both dispatch points and recording
# which helper reached which: the four in `change_basis` reach the cardinal
# B-spline kernel, Gauss-Legendre, or both; the two in `_bezier_degree` reach
# Gauss-Legendre. `_l2_reduction_operator` and `_interpolating_reduction_operator`
# reach neither and are deliberately absent -- keying them would cost a key slot
# for a result that cannot depend on the backend.
_MEMOIZED = [
    pytest.param(
        change_basis._cached_lagrange_to_bernstein_matrix,
        lambda f: f(_degree(), LagrangeVariant.GAUSS_LEGENDRE, np.float64),
        id="lagrange_to_bernstein",
    ),
    pytest.param(
        change_basis._cached_cardinal_to_bernstein_matrix,
        lambda f: f(_degree(), np.float64),
        id="cardinal_to_bernstein",
    ),
    pytest.param(
        change_basis._cached_legendre_to_cardinal_matrix,
        lambda f: f(_degree(), np.float64),
        id="legendre_to_cardinal",
    ),
    pytest.param(
        change_basis._cached_cardinal_to_legendre_matrix,
        lambda f: f(_degree(), np.float64),
        id="cardinal_to_legendre",
    ),
    pytest.param(
        bezier_degree._bernstein_collocation_1d,
        lambda f: f(_degree(), 8),
        id="bernstein_collocation",
    ),
    pytest.param(
        bezier_degree._tensor_gauss_weights,
        lambda f: f((8,)),
        id="tensor_gauss_weights",
    ),
]


@_BOTH_BACKENDS
@pytest.mark.parametrize(("memoized", "call"), _MEMOIZED)
def test_a_memoized_helper_holds_one_entry_per_backend(
    memoized: Any, call: Callable[[Any], Any]
) -> None:
    """Each dispatch-reaching helper stores the two backends separately.

    What it catches: the decorator reverted to a plain ``functools.lru_cache`` on
    any of the six, which is a one-word edit and produces no failing assertion
    anywhere else in the suite while the two backends agree bit for bit.

    Args:
        memoized (Any): The memoized helper under test.
        call (Callable[[Any], Any]): Invokes it with arguments it accepts.
    """
    memoized.cache_clear()
    with use_backend(Backend.PYTHON):
        call(memoized)
    assert memoized.cache_info().currsize == 1, "the first call did not populate the cache"

    with use_backend(Backend.CPP):
        call(memoized)
    assert memoized.cache_info().currsize == 2, (
        "the second backend reused the first backend's entry; this helper is not keyed "
        "on the backend, so a parity comparison through it measures nothing"
    )


@_BOTH_BACKENDS
@pytest.mark.parametrize(("memoized", "call"), _MEMOIZED)
def test_a_memoized_helper_returns_a_distinct_object_per_backend(
    memoized: Any, call: Callable[[Any], Any]
) -> None:
    """Two backends never receive the identical array object.

    The value-level face of the previous test, and the one that states the
    property a parity test actually depends on: ``A is B`` means nothing was
    compared. It is kept separate because it would still pass if the two entries
    happened to be equal, which today they are.

    Args:
        memoized (Any): The memoized helper under test.
        call (Callable[[Any], Any]): Invokes it with arguments it accepts.
    """
    memoized.cache_clear()
    with use_backend(Backend.PYTHON):
        first = call(memoized)
    with use_backend(Backend.CPP):
        second = call(memoized)
    assert first is not second, (
        "both backends were handed the same object, so any comparison between them "
        "is an identity check wearing a parity test's name"
    )


def test_a_cache_does_not_leak_across_threads() -> None:
    """A thread outside a ``use_backend`` block never reads a value from inside one.

    This is the test that distinguishes keying from clearing. A remedy that drops
    every cache when a block opens and closes passes the two tests above and fails
    this one, because the entry the outside thread reads is fresh and still
    computed under the inside thread's backend.

    It uses its own decorated function rather than one of the six, so it reports
    on the mechanism rather than on any caller.

    The two backends are named relative to the **ambient** one rather than
    absolutely, and that is load-bearing: this file is run under
    ``PANTR_BACKEND=python`` and under ``PANTR_BACKEND=cpp``, and a fixed pair
    makes the test fail outright under the second -- a thread that entered no
    block correctly reads whatever the process default is. Naming the ambient
    backend as the override would also make the comparison vacuous, since the
    inside and outside threads would then agree for a reason that has nothing to
    do with keying.
    """

    @backend_keyed_cache(maxsize=8)
    def which(degree: int) -> tuple[int, int]:
        """Record the backend that computed this entry.

        Args:
            degree (int): Stands in for a real argument.

        Returns:
            tuple[int, int]: The active backend and the argument.
        """
        return (int(active_backend()), degree)

    inside_started = threading.Event()
    inside_populated = threading.Event()
    seen: dict[str, tuple[int, int]] = {}
    ambient = active_backend()
    other = next((backend for backend in available_backends() if backend is not ambient), None)
    if other is None:
        pytest.skip(
            "distinguishing keying from clearing needs two backends to differ; this "
            "installation has only "
            f"{ambient.name}, so the outside thread would read the inside thread's "
            "entry legitimately"
        )

    def inside() -> None:
        with use_backend(other):
            inside_started.set()
            which(7)
            inside_populated.set()

    def outside() -> None:
        inside_started.wait(timeout=5.0)
        inside_populated.wait(timeout=5.0)
        seen["value"] = which(7)

    populate = threading.Thread(target=inside)
    read = threading.Thread(target=outside)
    populate.start()
    read.start()
    populate.join(timeout=5.0)
    read.join(timeout=5.0)

    assert seen["value"] == (int(ambient), 7), (
        "a thread that entered no use_backend block was served an entry computed "
        "inside one; clearing the caches on entry and exit does not fix this"
    )


def test_the_two_helpers_that_reach_no_kernel_are_left_alone() -> None:
    """The two reduction operators are deliberately not backend-keyed.

    Recorded as a test because the natural reflex on reading this file is to
    decorate every memoized helper in both modules for symmetry. These two reach
    neither dispatch point, so a backend key would hold two identical entries and
    halve the useful size of the cache for nothing.
    """
    for memoized in (
        bezier_degree._l2_reduction_operator,
        bezier_degree._interpolating_reduction_operator,
    ):
        assert not hasattr(memoized, "__wrapped__") or memoized.cache_info().maxsize == 64, (
            "this helper reaches no dispatched kernel; if that changed, key it and "
            "move it into the parametrization above"
        )
