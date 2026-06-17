# Parallelism

PaNTr uses [Numba](https://numba.pydata.org/) to JIT-compile performance-critical
kernels and distribute work across CPU cores via `prange` (parallel range).
This page explains which operations run in parallel, how to control the
thread count, and how to avoid over-subscription when composing PaNTr with
other parallel libraries.

## What runs in parallel

The following operations are parallelized over **evaluation points**:

| Operation | Kernel | Parallelism dimension |
|---|---|---|
| B-spline basis evaluation (Cox--de Boor) | `_compute_basis_nurbs_book_impl` | points |
| B-spline basis derivatives (DerBasisFuncs) | `_compute_basis_deriv_nurbs_book_impl` | points |
| B-spline evaluation (basis + combine) | `_evaluate_Bspline_basis_combine_1D` | points |
| B-spline derivative evaluation | `_evaluate_Bspline_basis_combine_*_deriv_1D` | points |
| Bernstein basis evaluation | `_tabulate_Bernstein_basis_1D_core` | points |
| Legendre basis evaluation | `_tabulate_Legendre_basis_1D_core` | points |
| Cardinal B-spline basis evaluation | `_tabulate_cardinal_Bspline_basis_1D_core` | points |
| Bezier evaluation and derivatives | `_evaluate_bezier_1d_core` | points |
| Bezier split / slice / restrict | `_split_bezier_1d_core` | control point columns |
| Bezier extraction (`to_beziers`) | `_apply_bezier_extraction_1d_core` | elements |

By default, all available CPU cores are used.

## Controlling the thread count

PaNTr exposes three functions for thread-pool control:

```python
import pantr

# Query the current thread count
n = pantr.get_num_threads()

# Set it globally
pantr.set_num_threads(4)

# Temporarily override with a context manager
with pantr.num_threads(2):
    result = space.tabulate_basis(pts)  # uses 2 threads
# reverts to previous value here
```

Setting the thread count to **1** effectively serializes all parallel kernels.

### BLAS / LAPACK coordination

NumPy and SciPy operations (matrix multiplies, solves, etc.) use their own
thread pools (OpenBLAS, MKL, or Apple Accelerate).  When PaNTr's Numba threads
*and* BLAS threads are both active, the total number of OS threads can exceed
the number of physical cores, causing **over-subscription** and degraded
performance.

The `num_threads` context manager accepts a `limit_blas` flag that throttles
BLAS thread pools for the duration of the block via the
[threadpoolctl](https://github.com/joblib/threadpoolctl) package (a core
dependency of PaNTr):

```python
with pantr.num_threads(4, limit_blas=True):
    # Numba uses 4 threads; BLAS also limited to 4
    result = bspline.evaluate(pts)
```

## Usage patterns

### Pattern 1: PaNTr owns all parallelism (default)

The simplest approach.  Every PaNTr call distributes work across all cores
internally.  User code stays serial:

```python
# PaNTr parallelizes over 100k points automatically
result = bspline.evaluate(pts)
```

Best when each individual call has a large workload (many evaluation points or
many elements).

### Pattern 2: User owns outer parallelism, PaNTr runs serially

When you have many independent objects (curves, surfaces, patches) to process,
it is often better to parallelize *across* objects yourself and let each PaNTr
call run serially.  This avoids nested thread-pool contention:

```python
import concurrent.futures
import pantr

pantr.set_num_threads(1)  # each PaNTr call is serial

with concurrent.futures.ThreadPoolExecutor(8) as pool:
    futures = [pool.submit(curve.evaluate, pts) for curve in curves]
    results = [f.result() for f in futures]
```

This works because **Numba releases the GIL** during execution, so multiple
Python threads genuinely run Numba-compiled code concurrently.

### Pattern 3: Balanced -- user and PaNTr share cores

Split the available cores between PaNTr's inner parallelism and your own outer
parallelism:

```python
pantr.set_num_threads(2)  # PaNTr uses 2 threads per call

with concurrent.futures.ThreadPoolExecutor(4) as pool:
    futures = [pool.submit(curve.evaluate, pts) for curve in curves]
    results = [f.result() for f in futures]
# 4 user threads x 2 Numba threads = 8 total on an 8-core machine
```

## Threading layer

Numba supports several threading backends: **TBB**, **OpenMP**, and
**workqueue**. With the default `NUMBA_THREADING_LAYER=default`, Numba selects
the first that is installed, preferring **TBB → OpenMP → workqueue** — so a
bare environment with neither TBB nor OpenMP available falls back to workqueue.

**TBB is the recommended backend for PaNTr** because it handles nested and
concurrent parallel regions safely — `workqueue` is *not* safe when several
Python threads call PaNTr kernels concurrently (the *Usage patterns* above). The
reliable way to get it is to **install the TBB backend**:

```bash
pip install tbb
```

Numba then selects it automatically (no env var needed). To *force* it — and get
a clear error instead of a silent workqueue fallback if TBB is missing — set the
environment variable before importing PaNTr (or any Numba code):

```bash
export NUMBA_THREADING_LAYER=tbb
```

The maximum number of threads available to Numba is determined by
`NUMBA_NUM_THREADS` (defaults to the number of logical cores).
`pantr.set_num_threads(n)` can lower the count at runtime but cannot exceed
this maximum.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `NUMBA_NUM_THREADS` | Number of logical cores | Maximum thread-pool size |
| `NUMBA_THREADING_LAYER` | `default` (picks `tbb`→`omp`→`workqueue` by availability) | Threading backend; install `tbb` to get TBB, or set this to force a specific layer |
| `NUMBA_DISABLE_JIT` | `0` | Set to `1` to disable JIT entirely (useful for coverage) |
