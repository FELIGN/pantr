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

## Hybrid MPI + threads

Under MPI the two parallel layers can collide: every rank evaluates the same Numba
kernels, and each rank's thread pool defaults to *all* logical cores. Running `R`
ranks on an `n`-core node would launch `R x n` compute threads (plus a BLAS pool per
rank) -- heavy over-subscription, and worse still when the launcher pins each rank to
a single core.

To prevent this, the **first use of any `pantr.mpi` entry point**
(`DistributedSpace`, `from_dolfinx`) applies a process-level default: **one Numba
thread per rank** (flat MPI -- the convention PETSc and dolfinx also follow). It is
applied at most once per process and never overrides a thread count you set yourself.

### Precedence

Explicit configuration always wins over the default. If any of the following is in
effect when an entry point first runs, the policy is a no-op:

| Set by | When | Effect |
|---|---|---|
| `NUMBA_NUM_THREADS` env var | before importing PaNTr | caps *and* fixes the count; the default never fires |
| `pantr.set_num_threads(n)` / `pantr.num_threads(...)` | any time | marks the count user-owned for the rest of the process |
| `pantr.mpi.configure_threads(n)` | any time | sets the per-rank count explicitly |

Once the default has fired, raising the count afterwards (e.g. for serial rank-0
post-processing) sticks -- the policy will not re-throttle on a later entry-point call.

### Choosing threads per rank

For a hybrid run with `k` threads per rank, call `configure_threads` on **every**
rank -- before or after building the distributed space:

```python
import pantr.mpi

pantr.mpi.configure_threads(4)                    # 4 Numba threads on this rank
pantr.mpi.configure_threads(4, limit_blas=True)   # also cap BLAS/LAPACK to 4
```

`configure_threads` can only *lower* the count below `NUMBA_NUM_THREADS` (the
import-time maximum, defaulting to all logical cores). To raise that ceiling, set
`NUMBA_NUM_THREADS` in the environment before launch.

### Pinning: give each rank its cores

Launchers often bind each rank to a single core by default, so a rank's extra threads
would just timeshare that one core. Tell the launcher to hand each rank `k` cores:

```bash
# OpenMPI: k cores per rank
mpiexec -n R --map-by socket:PE=k --bind-to core python run.py
# ...or drop binding entirely and let the OS schedule
mpiexec -n R --bind-to none python run.py
# Slurm
srun --ntasks=R --cpus-per-task=k python run.py
```

### Serial PaNTr under your own `mpiexec`

If you run PaNTr serially across ranks **without** constructing any `pantr.mpi`
object, no entry point fires and the default policy never engages. Throttle each rank
yourself:

```python
pantr.mpi.configure_threads(1)   # or pantr.set_num_threads(1)
```

```{note}
Numba's thread count is **thread-local**: `configure_threads` and the default policy
govern kernels launched from the calling thread -- in SPMD that is the rank's main
thread, where assembly runs. If a rank launches PaNTr kernels from its *own* worker
threads, call `pantr.set_num_threads` inside each one.
```

See [Distributed spaces](distributed-spaces.md) for the MPI distribution layer itself.

## Threading layer

Numba supports several threading backends: **TBB**, **OpenMP**, and
**workqueue** (the default).  TBB is the recommended choice for PaNTr because
it handles nested and concurrent parallel regions safely.  To select it, set
the environment variable before importing PaNTr:

```bash
export NUMBA_THREADING_LAYER=tbb
```

or, in Python before any Numba import:

```python
import os
os.environ["NUMBA_THREADING_LAYER"] = "tbb"
```

The maximum number of threads available to Numba is determined by
`NUMBA_NUM_THREADS` (defaults to the number of logical cores).
`pantr.set_num_threads(n)` can lower the count at runtime but cannot exceed
this maximum.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `NUMBA_NUM_THREADS` | Number of logical cores | Maximum thread-pool size |
| `NUMBA_THREADING_LAYER` | `workqueue` | Threading backend (`tbb`, `omp`, `workqueue`) |
| `NUMBA_DISABLE_JIT` | `0` | Set to `1` to disable JIT entirely (useful for coverage) |
