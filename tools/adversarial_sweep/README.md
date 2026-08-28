# Adversarial sweep

Generates hostile inputs for pantr's public entry points and runs them with Numba's
bounds check on, hunting the class of bug the test suite structurally cannot reach.

```console
conda run -n pantr python tools/sweep.py --profile smoke
conda run -n pantr python tools/sweep.py --profile full --journal sweep.jsonl
```

`__init__.py` carries the coverage table, the axes crossed, and what is deliberately
left out. This file answers a different question: **which implementation does the
sweep actually grade?**

## It grades the Python backend, and only the Python backend

Every case runs against `Backend.PYTHON`. Nothing here sets `PANTR_BACKEND`, and
unset is the Python backend by definition (`src/pantr/_backend.py`). A run of this
tool says nothing about the C++ backend, in either direction.

That is not an omission waiting to be filled in. It follows from what the sweep
looks for.

## The mechanism, and why it does not carry over

The sweep exists because Layer-3 kernels run under Numba `nopython=True`, where
there is no bounds checking: an out-of-range write corrupts memory silently, a
negative index wraps to the end of the array, and int64 arithmetic overflows
untrapped. `NUMBA_BOUNDSCHECK=1` turns the first two into exceptions, and the sweep
feeds the kernels inputs the test suite does not contain. `assert_boundscheck_active`
in `_core.py` is the canary that proves a given run could have caught an overrun at
all; without it a clean sweep would be indistinguishable from a sweep with the check
switched off.

The C++ side has no equivalent knob. Its counterpart of a kernel precondition is
`PANTR_PRECONDITION`, and `cpp/include/pantr/core/precondition.hpp` states plainly
what it is:

> **It compiles to nothing in a release build**, by construction rather than by
> policy: it is `assert`, so `NDEBUG` removes it.

So in every non-Debug build -- which is every build a Python caller ever loads,
including the wheel and the extension `pip install -e .` produces -- a C++
precondition violation is not reported. It is undefined behaviour, exactly as it
would be in an unchecked Numba kernel, and nothing this tool can set at run time
changes that. Pointing the sweep at the C++ backend would produce a green run that
means nothing.

What does grade the C++ preconditions is `cpp/tests/precondition_*.cpp`: separate
executables, registered by `cpp/tests/CMakeLists.txt` **only** under
`CMAKE_BUILD_TYPE=Debug`, each expected to abort with a specific assertion message.
They are a different instrument for a different build, and they are where a new C++
precondition earns its check.

## The reach shrinks as the port proceeds, deliberately

Each type that moves to C++ leaves this sweep's scope, because its checks stop being
Numba kernel checks. `geometry` is already in that position -- `AABB` and
`AffineTransform` are C++-owned types with Python wrappers -- and the grid, quadrature
and Bezier groups follow as their milestones land. The coverage table in `__init__.py`
therefore describes what is swept **on the Python backend**, and that set gets smaller
over time.

This is recorded rather than fixed. Adding a backend axis to `_core.py`, `_axes.py`
and `__main__.py` was considered and rejected: it would widen the tool to run cases
against an implementation whose checks are compiled out, which buys a larger case
count and no more evidence.
