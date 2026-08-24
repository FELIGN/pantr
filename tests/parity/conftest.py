"""Session fixtures shared by every module under `tests/parity`.

`cpp_backend` requires the compiled extension for the test that asks for it, and
`_jit_warmup_barrier` blocks until the background Numba warmup has finished. Both
were kernel-agnostic to begin with; this is where every ported kernel's parity
module now finds them, instead of each carrying its own copy.
"""

from __future__ import annotations

import pytest

from pantr import _numba_compat
from tests._parity_harness import demand_cpp_backend


@pytest.fixture(scope="session")
def cpp_backend() -> None:
    """Require the compiled extension for the test that asks for it.

    Requested explicitly rather than applied module-wide, so the harness self-tests
    and the Numba-side invariants below still run in an installation without the
    extension -- which is the common local configuration. The skip-or-fail decision
    itself lives in the harness, so the next ported kernel inherits it instead of
    reaching for a plain ``skipif`` that would skip silently.
    """
    demand_cpp_backend()


@pytest.fixture(scope="session", autouse=True)
def _jit_warmup_barrier() -> None:
    """Block until the background Numba warmup finished, before any kernel call.

    Session-scoped because the barrier is a once-per-process event. CLAUDE.md
    records the failure it prevents: a Layer 3 ``parallel=True`` kernel called from
    the main thread while ``pantr.__init__`` is still compiling on a background
    thread **aborts** the interpreter rather than raising, and a kernel-heavy file
    reaching kernels early is exactly the pattern that triggers it. This file calls
    the Layer 3 kernel directly, bypassing the Layer 2 entry points that carry the
    barrier themselves, so it must take the barrier itself.

    Defined here, autouse, so it covers every module under `tests/parity` rather
    than one kernel's file alone.
    """
    _numba_compat.wait_for_jit_warmup()


@pytest.fixture(scope="session", autouse=True)
def the_oracle_is_self_consistent() -> None:
    """Refuse to run if Numba would use SVML, which makes the oracle disagree with itself.

    `_dedup_roots_core` calls `pow`, and it is reached both serially and from
    `_find_roots_batch_core`, which carries ``parallel=True``. SVML's `pow` is not
    bit-identical to glibc's, so with SVML present the batch path and the serial path
    could return different roots **within the Numba backend**, and every parity figure
    measured against it would be measuring two different oracles.

    Not a hypothetical guarded on principle: it is one `pip install icc_rt` away, and
    nothing else in the suite would notice. Numba enables SVML automatically when the
    runtime is importable.
    """
    from numba import config  # noqa: PLC0415

    assert not config.USING_SVML, (
        "numba is using SVML, so its `pow` is not the platform one and the serial and "
        "parallel root-finding paths may disagree inside the oracle itself. Remove "
        "icc_rt / intel-cmplr-lib-rt, or set NUMBA_DISABLE_INTEL_SVML=1, before "
        "trusting any parity figure from this suite."
    )
