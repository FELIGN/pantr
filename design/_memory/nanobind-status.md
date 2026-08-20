---
name: nanobind-status
description: nanobind split mode is verified working and composing with multi-ISA; the mandatory build flags that come with it
metadata:
  type: reference
---

**Verified 2026-08-19/20 on the build server**, superseding the earlier "unreleased, unverified"
status. Full evidence in `design/build_findings.md`.

**Split mode works and composes.** nanobind `3.0.0.dev2` with `nanobind-backend 1.0.0.dev2`
builds, imports and runs; the frontend links as `.abi3.so`, so it does target the stable ABI
as advertised. The open question about multi-ISA is **answered**: two frontends compiled from
one source at different `-march` (baseline and `x86-64-v3`) against a single backend both
import in the same process and both work.

**Split mode and `NB_STATIC` are behaviourally indistinguishable** — identical acceptance
matrix over 17 cases, performance within run-to-run spread. The choice is about packaging
only, so it can be decided on distribution grounds alone.

**Three build settings are mandatory, not optional:**

- **`NOMINSIZE`** plus **`install.strip = false`** (or `NOSTRIP`). `nanobind_add_module()`
  appends `-Os` *after* the build type's `-O3` and the last flag wins, so kernels compile for
  size. Measured **31.4 ms against 10.3 ms** on the same source: a silent 3x aimed at the
  number the port is judged by.
- **`NB_SUPPRESS_WARNINGS`**. In split mode nanobind adds its includes to the extension target
  and marks them `SYSTEM` only under that option; without it a `-Wshadow` in `nb_attr.h` meets
  `-Werror` and the build dies inside a third-party header.

**`nanobind-backend` is a runtime dependency of the wheel.** Without it the import fails,
though with a clear message.

Python floor for pantr is **3.12**, decided separately: numpy 2.5 and scipy 1.18 already
require it.
