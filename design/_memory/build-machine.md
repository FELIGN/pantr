---
name: build-machine
description: The C++ port is developed on a shared private Linux server capped at 20 CPUs; the calibration trap that comes with it
metadata:
  type: project
---

The C++ port is developed on a **private single-node Linux server**: 160 logical CPUs over
80 physical, 1 TB RAM, shared with other users. A laptop (Apple Silicon, macOS) is the
second platform.

**Work is capped at 20 CPUs**, and it is enforced in two layers, both already in place:
`taskset -cp 0-19 $$` in `~/.bashrc` (inherited by every child, so any number of agents share
the same 20), and thread-count variables in the conda env's `activate.d`
(`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `NUMBA_NUM_THREADS`, `CMAKE_BUILD_PARALLEL_LEVEL`,
`CTEST_PARALLEL_LEVEL`). Both are needed: affinity alone would leave 128 OpenBLAS threads
fighting over 20 CPUs. `NUMBA_NUM_THREADS` is read at import and cannot be raised later.

**Verified on that machine (2026-08-19):** conda-forge GCC 14 and Clang 18.1.8 both pass a
C++20 concepts probe (the system GCC 10 / Clang 10 are shadowed by the env); `<mdspan>` is
**absent**, so the Kokkos fallback is required; CMake **4.4.2**, which rejects dependencies
declaring `cmake_minimum_required(VERSION < 3.5)`; OpenBLAS 0.3.34, SkylakeX kernel, so the
CPU has **AVX-512** and the whole ISA ladder is measurable locally.

**The trap that comes with the machine:** thresholds calibrated on many cores lie on an
8-core laptop, which is what most users have. pantr already carries one (`_PARALLEL_MIN_NUM_PTS`).
Any new threshold must be **derived**, not measured on the development machine.

**And the BLAS differs between the two machines** (OpenBLAS on the server, Accelerate on the
laptop), so `svd`, `solve` and `lstsq` differ in the last bits. Parity is reproducibility
within a derived bound, never bit-exactness, and reference data must be computed live against
the sibling implementation in the same tree rather than captured from a file.

## The conda env's ruff is out of the pinned range

`pyproject.toml` pins `ruff>=0.12.0,<0.13`; the `pantr` conda env carries **0.16.3**. Newer
rules make it report findings that do not exist for CI, which resolves the pin: measured
2026-08-21 at 12 findings in three untouched files. **A lint result from `conda run -n pantr
ruff` is not the lint CI runs.** The checkout's own `.venv/bin/ruff` has 0.12.12 installed
for exactly this.



## Threaded OpenBLAS destroys small-matrix timings here, 2026-08-24

With the env's `OPENBLAS_NUM_THREADS=20`, a **36-by-36 LAPACK SVD** cost between one and two
orders of magnitude more than the same call pinned to one thread, and its timing varied by
nearly an order of magnitude between batches. Twenty threads synchronizing over a matrix that
fits in L1 is pure overhead, and the box is shared, so the contention is real rather than
theoretical.

This is not the "OpenBLAS spawns 128 threads" trap: the variables are set correctly to 20. It
is that 20 threads on a tiny dense factorization is simply the wrong configuration, and the cap
does not save you from it.

**How to apply: pin the threads before timing anything that calls into BLAS or LAPACK on a small
matrix** (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`). Unpinned numbers here
are noise, and noise that happens to look like a speedup: an unpinned measurement of a cache was
quoted as a 2.9x win in this repo before a pinned re-run showed the cache was not even on the
call path. `scripts/measure_bezier_interpolation_port.py` refuses to run unpinned for this
reason, which is the shape any future timing script should copy.

A separate consequence for the port: a **single-threaded C++ implementation compared against
threaded LAPACK on this box would look far better than it is**. Never take that comparison.
