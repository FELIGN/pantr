---
name: performance-followup
description: Performance is a wanted follow-up on the C++ port, deliberately deferred; here is what is already measured so it is not re-measured
metadata:
  type: project
---

**Pablo wants to go deeper on performance later.** Recorded 2026-08-19, when the C++
infrastructure prototype (PR #335 against `proto/cpp`) deliberately stopped short of it.

## Already measured, do not redo

All on the shared build server, GCC 14.4.0, the cardinal B-spline Cox-de Boor tabulation,
minimum over repeats. Reproduce with `scripts/bench_parity.py` and
`cpp/benchmark/bench_cardinal_bspline.cpp`.

- **Kernel against kernel, single-threaded, 10^6 points:** 1.42x at degree 1, 1.30x at
  degree 3, 0.98x at degree 8. Modest. Both compilers generate reasonable code for the same
  recurrence and there is no free factor waiting in a straight transliteration.
- **At the public entry point** the C++ backend is 2.4x to 4x faster at 100 points and 0.17x
  to 0.29x at 10^6 against numba's 20 threads. The small-input win is not the port: the numba
  cardinal B-spline kernel is `parallel=True` **unconditionally** and, unlike the Bernstein
  kernels, has **no serial twin**, so it pays a fork/join for three points.
- **The ISA ladder gap**, which `design/simd.md` gates stage 2 on: baseline to
  `-march=x86-64-v3` is 1.20x to 1.27x. **`-march=native` (AVX-512 here) is slower than
  `x86-64-v3`**, measuring that note's stated hazard on a real pantr kernel.
- **The trap that dominated everything else:** `nanobind_add_module()` appends `-Os` after
  the build type's `-O3` and the last flag wins, so the kernel was compiled for size inside
  the extension: 31.4 ms against 10.3 ms on identical source. `NOMINSIZE`, plus `NOSTRIP` and
  `install.strip = false` for symbols. Any future measurement must check the effective flags
  first.

## Not done, and what to attack

- **Profiling.** Now possible: the module keeps its symbols, so `perf` names
  `tabulate<double>`. Nobody has profiled anything yet; every number above is wall clock.
- **Threading on the C++ side.** Absent on purpose. It is constrained by
  `design/user_functions_across_the_boundary.md`'s rule that a callback is invoked from one
  thread only, so it is a design decision rather than a tuning knob.
- **Blocking, then SIMD, in that order.** `design/simd.md` is emphatic: the banded
  contraction is bandwidth-bound at ~0.2 flops/byte unblocked, so vectorising it first would
  measure as a disappointment and be mistaken for evidence that SIMD does not help.
- **`simd.md` open question 1 is still open:** does the blocked contraction auto-vectorise?
  If yes, no batch abstraction is needed and xsimd never enters the dependency list. That is
  the first measurement to take, via `-fopt-info-vec-missed` / `-Rpass-missed=loop-vectorize`.
- **The threshold trap.** `_PARALLEL_MIN_NUM_PTS = 4096` and anything like it must be
  **derived**, never measured on this machine: see [[build-machine]].

The tool for this is the `optimizer` agent or the `perf-tune` skill, which is
measurement-first and parity-preserving. Parity here means the harness in
`tests/_parity_harness.py`, which already exists and already knows how to state a bound.
