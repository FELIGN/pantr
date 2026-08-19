---
name: nanobind-status
description: Split mode is verified to work and to compose with a second compiled variant; the traps are nanobind's build defaults, not its API
metadata:
  type: reference
---

**Measured on the build server, 2026-08-19**, by building and running rather than reading.
Supersedes the earlier version of this note, which recorded split mode as unreleased and its
composition with a multi-ISA build as unverified.

**Versions at that date:** stable **2.15.0**; **3.0.0.dev2** and **nanobind-backend
1.0.0.dev2** on PyPI. The conda env carries **2.14.0**.

## Split mode works, and it composes

`nanobind_add_module(... BACKEND_MODULE nanobind_backend)` builds, imports and runs. The
frontend links as `.abi3.so`, confirming it targets the stable ABI as advertised. **Two
frontends built from one source at different `-march`, sharing a single backend, both import
in the same process and both run** -- which is the multi-ISA composition the earlier note
recorded as unverified and load-bearing.

Two conditions, both found by hitting them:

- **`NB_SUPPRESS_WARNINGS` is required.** In split mode nanobind adds its own include
  directory straight onto the extension target and marks it SYSTEM *only* with that option.
  Without it, `nb_attr.h`'s `using arg = arg_t<>` trips `-Wshadow` and `-Werror` ends the
  build. The `NB_STATIC` path needs a different workaround entirely: retag
  `nanobind-static`'s includes on the *producing* target, because it is built rather than
  imported, so its includes arrive as `-I` and GCC drops a later `-isystem` for the same path.
- **`nanobind-backend` becomes a hard runtime dependency** of the wheel. It fails with a
  clear, actionable message, which is the right failure, but it is a packaging commitment.

**No behavioural difference between split mode and `NB_STATIC`** was found: identical
acceptance across a 17-case matrix, and throughput within run-to-run spread. The differences
are packaging only.

## The real traps are the build defaults, not the API

`nanobind_add_module()` silently applies five things. The first cost this project its
headline measurement:

- **`-Os` appended after the build type's `-O3`**, and the last optimisation flag wins. A
  numerical kernel compiled into the module inherits it: measured **31.4 ms against 10.3 ms**
  on identical source. `NOMINSIZE` turns it off. The single most expensive default.
- **`-Wl,-s`**, so the module is stripped: `nm` reports "no symbols", `perf` cannot name the
  kernel and a segfault gives no backtrace. `NOSTRIP` turns it off.
- `-fno-strict-aliasing` unconditionally, with no option to disable it. Measured as free for
  a simple recurrence kernel; worth watching for a pointer-heavy one, since it means the
  module and a plain executable compile the same header under different alias analysis.
- `-fno-stack-protector` unless `PROTECT_STACK`, and `-ffunction-sections -fdata-sections`
  guarded by a *different* variable than `NOMINSIZE`.

## `nb::ndarray` constrains less than it appears to

Its constraints are not all guarantees. **nanobind satisfies `c_contig` and a dtype mismatch
by CONVERTING to a temporary rather than rejecting.** For an input that is merely wasteful;
for an `out` parameter it is silently wrong -- the kernel fills the temporary and the
caller's array comes back untouched, with no exception. `.noconvert()` closes exactly those
two axes. Rank, writability and device were never convertible and always hard-reject.

`nb::ndarray<const T, ...>` behaves differently from the mutable form, and correctly: a
non-writable array is accepted as an input and refused as an output.

**A masked array is accepted through the buffer protocol with the mask silently ignored**,
which is how a C++ backend can return numbers where the numba oracle raises.

**An unvalidated integer parameter is a segfault, not an error.** `int degree` reaching a
`static_cast<std::size_t>` turns `-1` into `SIZE_MAX`; declaring the binding parameter
`unsigned` makes it a clean `TypeError` and leaves the `.pyi` rendering `degree: int`.

## API changes across 2.x and 3.x

For a module with no bound types, no trampoline and no `gil_scoped_acquire`, **there is no
behavioural difference**: `ndarray_import` is byte-identical 2.14 to 2.15 and equivalent in
3.x. One correction to what was recorded before: `nb::arg` is a **runtime struct in 2.x**
whose `noconvert()` mutates a bool, and only a compile-time tag in 3.x. Source written
against the 3.x spelling still compiles on 2.x, but not for the reason previously given.

Python floor for pantr is **3.12**, decided separately: numpy 2.5 and scipy 1.18 already
require it.
