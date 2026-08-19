# User-supplied functions across the Python / C++ boundary

**Status:** design note for the C++ port. Nothing here is implemented.
**Date:** 2026-08-17.
**Scope:** how a function defined by the user in Python reaches the C++ core, and when it
does not need to. Motivating case: fitting a spline to a user-defined function.
**Companion:** `design/large_data_fitting.md`, which covers the fitting itself.

## The headline

**For fitting, the Python function never crosses into C++.** The evaluation points are
known in advance, so the Python layer evaluates the function once, vectorized, and hands
an array down. `fit_bspline` (pre-evaluated values) is already exactly that entry point,
and the callable-based `interpolate_bspline` can live entirely in the thin Python layer.

That keeps the C++ API callback-free, which two decisions already taken require:

- Tier A must stay device-compatible, so no callbacks inside kernels.
- Callbacks do not cross a flat C ABI.

A callback seam is still needed, but for different callers and different algorithms. Both
are covered below.

## What already works

`src/pantr/bspline/_bspline_interpolate.py:191-240`, `_evaluate_func_on_lattice`, calls
`func(lattice)` **once** for the whole lattice (`:214`). One Python call per fit, not one
per sample point.

That matters more than anything else in this note. A per-point callback at roughly a
microsecond per call would cost `1.3 × 10^8 µs ≈ 130 s` for a `512³` volume, in call
overhead alone, before any arithmetic. Batching is what makes the whole idea viable, and
it is already the contract.

## The problem with the current contract

The callable receives a `PointsLattice` (`src/pantr/quad.py:405`) and must return a flat
array of shape `(n_total,)` or `(n_total, rank)`, which `:224` then reshapes to
`grid_shape`. `PointsLattice` offers two ways to get at the coordinates, and **the safe
one does not scale while the scalable one is not safe.**

| the user calls | scales? | ordering |
|---|---|---|
| `lattice.get_all_points()` | **no.** At `512³` it materializes `1.3 × 10^8 × 3` coordinates, about 3.2 GB, purely to pass them in | safe: pantr's own order, so the flat return matches by construction |
| `lattice.pts_per_dir` | yes: three 1D arrays | **the user's responsibility**, and easy to get wrong |

The second row is the hazard. `np.meshgrid` defaults to `indexing="xy"`, which **swaps the
first two axes**. A user who builds their own grid that way, evaluates, and ravels,
returns an array that `:224` reshapes into `grid_shape` without complaint. The result is
silently transposed in the first two directions. Nothing in the pipeline can detect it,
because a flat array of the right length carries no shape information to check.

## The contract the port should use

Pass the 1D coordinate arrays **shaped for broadcasting**, and require a return of the
**explicit** grid shape:

```python
def f(x, y, z):                       # x: (N1, 1, 1)   y: (1, N2, 1)   z: (1, 1, N3)
    return np.sin(x) * np.cos(y) * z  # -> (N1, N2, N3)
```

This removes both problems at once:

- **No coordinate materialization.** The three 1D arrays total `N1 + N2 + N3` values
  instead of `N1·N2·N3·d`. At `512³` that is about 12 kB instead of 3.2 GB.
- **No implicit ordering.** The returned array's shape states which axis is which, so a
  transposed result is a shape mismatch and is caught, not absorbed.

It is also less work for the user, not more: NumPy broadcasting is the idiom they would
reach for anyway, and no `meshgrid` call is needed.

Vector-valued functions return `(N1, ..., Nd, rank)`. The rank axis last matches the
existing convention.

### It composes with blocking

`design/large_data_fitting.md` leaves open whether a streaming path is needed for volumes
that exceed memory. The broadcasting contract handles it unchanged: a block is just a
sub-lattice, so the callback receives shorter 1D arrays and returns a smaller block. The
flat-`(n_total,)` contract would also work, but it would re-expose the ordering hazard at
every block boundary rather than once.

### What it cannot express

A function that is not elementwise in the coordinates. Anything needing a neighbourhood,
a global reduction, or the full point tuple in a non-broadcastable way. Those still need
a point-list contract, so the point-list path should be retained as the general fallback,
with the lattice path as the documented fast path. This mirrors the tensor-product versus
scattered distinction that already exists for nodes.

## When the callback does have to cross

Three cases. The second justifies the seam on its own, independently of Python.

1. **Adaptive fitting.** Fit, estimate the error, refine where it is large, re-evaluate at
   points **the C++ side chooses**. Python cannot evaluate in advance because the points
   do not exist yet.
2. **A C++ consumer fitting to its own C++ function.** A C++ consumer passes a C++
   lambda: no GIL, no Python, no measurable overhead. The seam is worth having even if the
   Python layer never uses it.
3. Point inversion or root finding on a user-supplied function.

### The seam

Same shape as the two seams already decided (`root_solver_fn`, and `partition_fn` in the
dolfinx sense): a callable in the C++ API, batched, with pantr shipping nothing that
mentions Python.

```cpp
// Evaluate on a sub-lattice. coords[d] holds the 1D coordinates along direction d.
// `out` is contiguous, C-order, shape (coords[0].size(), ..., coords[d-1].size(), rank).
using lattice_fn = std::function<void(std::span<const std::span<const double>> coords,
                                      std::span<double> out)>;
```

### The rule that makes the Python case work

**The callback is invoked from one thread only, at a batch boundary, never from inside a
parallel region.**

This is not a style preference. If a worker in the `std::jthread` pool called into Python,
every worker would need the GIL and the pool would serialize completely, so the
parallelism would cost more than it buys. Batching at the top level keeps the GIL
acquisition to once per batch, where it is amortized to nothing.

Stated as a contract it is enforceable and testable. Left implicit it will be violated by
whoever next parallelizes an outer loop, and the symptom (a parallel build slower than a
serial one) is hard to attribute.

This rule is also why the control flow does **not** need inverting. A "C++ returns a
request for points, the caller supplies them" loop would avoid the GIL question entirely
and would match the stateless-pattern shape used elsewhere in this design, but it costs
the user the ergonomics of passing a lambda, and batching already solves the problem it
would solve. Worth revisiting only if a case appears where the callback must genuinely be
driven from inside a parallel region.

## The escape hatch for many small calls

If an adaptive algorithm ends up needing many small evaluations rather than a few large
ones, there is a path with no GIL and no Python at all: **`numba.cfunc` compiles a Python
function to a real C-callable and exposes its address** (`.address`), which can be passed
as an integer and reinterpreted as a function pointer on the C++ side. This is the
mechanism behind `scipy.LowLevelCallable`.

Cost per call is native. The price is real and should be documented rather than
discovered:

- The signature must match **exactly**; a mismatch is undefined behaviour, not an error.
- There is no error propagation across the boundary. An exception in the user's function
  has nowhere to go.
- The user must have numba installed, which the port otherwise removes as a requirement.

So: an escape hatch, documented as such, not the main path.

## What pantr cannot fix

If the user's Python function is slow, the fit is dominated by the user's function and
pantr can do nothing about it. What the batched contract guarantees is that **pantr adds
one call's overhead, not `N`**. That is the whole of pantr's responsibility here, and it
is worth saying in the documentation so that a slow fit is attributed correctly.

## Epistemic status

- **Verified by reading the code:** that `_evaluate_func_on_lattice` makes a single
  batched call (`:214`); that the return contract is flat `(n_total,)` or
  `(n_total, rank)` reshaped to `grid_shape` (`:213-233`); that `PointsLattice` exposes
  `pts_per_dir` and `get_all_points` and nothing else public
  (`src/pantr/quad.py:405-478`).
- **Derived:** the 130 s per-point-callback figure (from roughly 1 µs per call, which is
  the order of magnitude the project's own conventions cite), the 3.2 GB coordinate
  materialization, and the 12 kB figure for the broadcasting contract.
- **A hazard identified by reading, not observed:** the `np.meshgrid(indexing="xy")`
  transposition. It follows from the documented default and from the unchecked `reshape`,
  but no user has been observed hitting it. It is cheap to make impossible, which is the
  argument for doing so.
- **Not investigated:** whether `numba.cfunc`'s address can be handed to nanobind as
  cleanly as described. The `scipy.LowLevelCallable` precedent is strong but the
  nanobind-side mechanics were not checked.

## Open questions

1. Does the lattice contract pass `d` separate arrays as positional arguments
   (`f(x, y, z)`) or one sequence (`f(coords)`)? Positional reads better in the common 2D
   and 3D cases and is awkward to write generically for arbitrary `d`. A `*coords`
   signature covers both but makes the arity implicit.
2. Should the point-list fallback and the lattice fast path be one function that inspects
   its argument, or two functions? `design/large_data_fitting.md` open question 4 asks the
   same thing about nodes, and the answer should be the same in both places.
3. ~~Is the adaptive-fitting case actually wanted?~~ **Answered: yes.** See
   `design/adaptive_thb_approximation.md`. So the seam does need its GIL wrapper, and the
   single-thread rule above becomes load-bearing rather than precautionary. Note that the
   adaptive loop as designed there re-evaluates nothing: it refines against a residual
   computed from samples already taken, so a Python callback is needed only if the sample
   set itself grows between iterations. Whether it does is the remaining question.
