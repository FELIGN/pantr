---
name: nanobind-status
description: nanobind split mode replaces the abi3 wheel strategy, but was unreleased as of 2026-08-19; re-check before committing
metadata:
  type: reference
---

**Dated 2026-08-19 and time-sensitive. Re-check the release state before choosing the wheel
strategy.** At that date: nanobind stable was **2.15.0**; **3.0.0.dev1** and
**nanobind-backend 1.0.0.dev1** were on PyPI, and the changelog said stabilising split mode
would still take time.

**Split mode** replaces the stable-ABI (abi3) approach and is better on both axes that
mattered. The extension splits into a frontend targeting the **Python 3.10 stable ABI** (one
wheel per platform covering every version) and a tiny version-specific backend shipped from
PyPI. So the Python floor is no longer forced up to 3.12 to get one wheel, and the function
dispatcher stops paying the stable-ABI penalty. Enabled with one CMake argument
(`BACKEND_MODULE`), so it is opt-in and the decision can wait.

Costs: `nanobind-backend` becomes a **runtime** dependency of the wheel, and its composition
with a multi-ISA build (several frontends, one backend) is **unverified** and must be tested
by the skeleton.

**API changes that touch decisions already taken:** `nb::gil_scoped_acquire` can now fail and
grows `is_valid()`, which the callback and plugin seams must consult; `NB_TRAMPOLINE` no
longer takes a `Size`, and instance-level monkey-patching is ignored; return-value policies
and argument annotations became compile-time tags, so "computed" ones are illegal.

Python floor for pantr is **3.12**, decided separately: numpy 2.5 and scipy 1.18 already
require it, so supporting 3.11 would mean supporting frozen-old dependencies.
