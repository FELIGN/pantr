# Adaptive approximation with THB splines

**Status:** design note for the C++ port. Nothing here is implemented.
**Date:** 2026-08-17.
**Scope:** the SOLVE / ESTIMATE / MARK / REFINE loop for approximating sampled data by a
THB spline. Wanted as a feature, so this note is about how to build it rather than whether
to.
**Companions:** `design/large_data_fitting.md` for the cost of one fit and for the memory
model, and `design/user_functions_across_the_boundary.md` for where the samples come from.

## The loop

1. Fit an approximation on the current THB space.
2. Measure the error per cell.
3. Mark cells for refinement.
4. Refine the hierarchical grid, producing a new THB space.
5. Repeat until a stopping criterion is met.

Three of those five steps have a non-obvious answer, and one of them is currently blocked
by an API that mutates. Taking them in order of how much they change the design.

## Step 2 is not an estimator. It is the error itself

Adaptive finite elements need an *estimator* because the true solution is unknown. Here it
is known: `f` is given at every sample point. So step 2 is not estimation, it is
**aggregation of the true residual**:

```
e_cell = || f_i - s(x_i) ||   over the samples i falling in that cell
```

This is a scatter-add over samples, `O(N)`, with the sample-to-cell map available in closed
form: for a lattice sample, the containing cell at a given level is a per-axis integer
division, the same arithmetic as `CartesianPartition` in the MPI design. No search, no
tree, no allocation.

Two consequences worth stating plainly:

- The loop is **far more reliable than its finite-element analogue**, because the marking
  decision is based on a measured quantity and not on a bound with unknown constants.
- Every convergence claim about the loop is therefore checkable directly, which removes the
  usual need to argue about estimator efficiency.

Which norm to aggregate in is a real choice: max, mean, or sum of squares per cell. Sum of
squares matches an L2 fit, max is what a geometric tolerance wants, and they mark different
cells. It should be a parameter, and an `enum class`, not a string.

## Step 5 is the hard one, because the data is noisy

**The residual includes the noise.** So marking cells by residual magnitude and refining
means refining wherever the noise is largest, and a THB space with enough levels can fit
noise arbitrarily well. The loop does not converge, it **refines forever** and produces a
progressively worse approximation of the underlying signal while the reported residual
keeps improving.

This is the classic overfitting failure of adaptive approximation on measured data, and it
is not hypothetical for an MRI volume.

The stopping criterion therefore cannot be a user-supplied tolerance on the residual. It
has to be tied to the **noise floor**: stop refining a cell when its residual is consistent
with noise rather than with unresolved signal. The standard mechanisms are a chi-squared
test against an estimated noise variance, or generalized cross-validation, which estimates
the right amount of smoothing from the data without needing the noise level supplied.

**Decided (2026-08-19), and it resolves an apparent conflict.** pantr will **not** carry a
smoothing penalty: regularization is the user's job. Both exact and measured data are in
scope. Those two look incompatible, because unpenalized refinement on noisy data refines
forever.

They reconcile by moving the noise handling **out of the fit and into the stopping
criterion**. "Stop refining this cell once its residual is consistent with noise" is not a
smoothing penalty: it does not change the fit, only when refinement halts. So it does not
contradict the decision that pantr does not regularize.

What it needs is one input: **the noise level, supplied by the user**, which is consistent
with the statistics being the user's responsibility. Estimating it instead (generalized
cross-validation) would drag the statistics back inside and is deliberately not the default.

So the loop carries **two stopping criteria, distinguishable in the API**:

- **exact data** (a function evaluated to machine precision): a plain residual tolerance is
  correct and nothing above applies.
- **measured data**: a noise-aware criterion taking a user-supplied noise level.

Conflating them produces a loop that misbehaves on one of the two, so the distinction has to
be visible at the call, not inferred.

## Step 1 does not separate, and that decides the algorithm

A THB space is **not** a tensor product. It is a hierarchical selection of basis functions
across levels, so the collocation matrix is a general sparse matrix and `_solve_kronecker`
(`src/pantr/bspline/_bspline_interpolate.py:154`) does not apply. Fitting directly on a THB
space means sparse least squares, and the `O(d · p · N)` cost that makes a single-level fit
a tens-of-milliseconds operation is lost.

Doing that once per iteration of an adaptive loop compounds the loss.

### The way out: fit the residual level by level

Each level of a THB space **is** tensor-product on its own grid. So instead of one global
THB fit, fit an increment per level:

```
s_0   = fit(f, level-0 space)                     # separable, cheap
r_0   = f - s_0                                   # residual at the samples
mark cells where r_0 is large
s_1   = fit(r_0, level-1 space restricted to the refined region)
r_1   = r_0 - s_1
...
s     = s_0 + s_1 + s_2 + ...                     # the THB function
```

Every fit is separable on its own level's grid, so every fit is cheap. And the total work
is `O(total)` summed over levels rather than `O(L × total)`, because level `ℓ` only ever
touches the samples inside its refined region, which shrinks geometrically when refinement
is local.

This is the multilevel B-spline approximation idea. **Citations from memory and needing
verification before they go in any published text:** Forsey and Bartels, *Hierarchical
B-spline refinement*, SIGGRAPH 1988, for the hierarchical representation; Lee, Wolberg and
Shin, *Scattered data interpolation with multilevel B-splines*, IEEE TVCG 1997, for the
coarse-to-fine residual algorithm. Both are well known in this area but the exact titles,
venues and years above have not been checked against the sources.

Restricting a level's fit to the refined region does complicate the separability slightly:
the active region at level `ℓ` is generally not a full tensor-product box. Two options,
and this is an open question rather than a settled point:

- Fit on the **bounding box** of the refined region, keeping full separability and paying
  for coefficients outside the active set, then discard them. Simple, and cheap when the
  refined region is compact.
- Fit on a **union of tensor-product boxes** covering the refined region, one separable
  solve each. Better when refinement produces several disjoint features, which for a level
  set it does.

### The alternative, for the record

If the level-wise scheme turns out not to be accurate enough, the fallback is sparse CG on
the full THB system, preconditioned by the level-wise solve. That keeps memory linear and
reuses the machinery, but it gives up the direct-solve cost and needs an iteration count
that has not been estimated.

**Decided (2026-08-19): Eigen supplies the solver in both cases**, so deferring this choice
is safe. `Eigen::SparseMatrix` with `SimplicialLDLT` covers the level-wise banded systems and
a global THB system alike, and Eigen is already a required dependency, so adding the sparse
path later costs nothing in dependency terms. The real cost of a global fit is assembling the
THB system, which is new code either way. The decision stays contained behind the API as long
as the return type does not leak which method produced it.

Quasi-interpolation is a third option, and pantr already has it for THB
(`_thb_quasi_interpolation.py:49`, `quasi_interpolate_thb_spline`). It is local, so it needs
no solve at all and is `O(n)`. For an adaptive loop that re-approximates every iteration
that is very attractive, at the cost of accuracy relative to a true L2 fit. It is the
natural thing to try **first**, precisely because it is the cheapest, and to measure the
accuracy gap against a fit before deciding the loop needs one.

## What blocks this today: refinement mutates

`HierarchicalGrid.refine_cells(self, cell_ids: Sequence[int]) -> None`
(`src/pantr/grid/_hierarchical_grid.py:1298`) returns `None`, so it refines **in place**.
`:453` confirms it: refinement recomputes `_level_base` and `_num_cells` and resets the BVH
and tags on the existing object.

The level-wise scheme above needs **all levels alive simultaneously**, because the final
function is the sum of the per-level increments. With in-place refinement, level `ℓ`'s space
is destroyed when level `ℓ+1` is created, so the increments have nowhere to live and the
accumulated function cannot be assembled.

So refinement must **return a new grid**. That is the project's own "construct then freeze"
rule, and here it is not a style question: the algorithm does not work without it.

Two secondary benefits, once refinement is value-returning:

- The loop becomes restartable and inspectable. Keeping the sequence of spaces means a
  refinement step can be reverted when the stopping criterion decides it went too far,
  which a noise-driven criterion will sometimes want to do.
- Structural sharing makes it cheap. A refined grid differs from its parent by one level's
  active set, so the new object can share the parent's immutable data rather than copying
  it, and the cost of returning a value instead of mutating is close to zero.

`refine` (`:1216`) should be checked for the same problem; only `refine_cells` was read.

## Memory across iterations

`design/large_data_fitting.md` gives the sizes. What is specific to the loop:

- The **residual** array is `N`-sized at level 0, and that is unavoidable.
- At later levels the residual is only needed **inside the refined region**, so it shrinks
  geometrically when refinement is local. Keeping a full `N`-sized residual at every level
  would be the obvious implementation and the wrong one.
- The **coefficients** are `O(number of active functions)` by construction, which is the
  point of THB and needs no special handling.
- Whether the level-0 residual overwrites the input values in place is a real decision:
  in place halves peak memory and destroys the data, out of place doubles it. For a
  memory-mapped MRI volume the input is on disk anyway, which makes in-place attractive.
  Recorded as an open question in the fitting note, since the memory model belongs there.

## Epistemic status

- **Verified by reading the code:** that `refine_cells` returns `None` and therefore mutates
  (`_hierarchical_grid.py:1298`), and that refinement resets cached state in place (`:453`);
  that `_solve_kronecker` is the tensor-product solve path
  (`_bspline_interpolate.py:154`); that THB quasi-interpolation already exists
  (`_thb_quasi_interpolation.py:49`).
- **Derived:** that a THB space is not tensor-product and therefore does not separate; that
  level-wise fitting costs `O(total)` rather than `O(L × total)`; that the sample-to-cell map
  is a per-axis integer division for a lattice.
- **Standard results, cited from memory and needing verification:** the two multilevel
  B-spline references above; the chi-squared and generalized-cross-validation stopping
  criteria, which are standard in smoothing-spline practice but whose precise form for a THB
  space has not been worked out here.
- **Asserted from reasoning, not measured:** that unpenalized adaptive refinement on noisy
  data refines without converging. The argument is that a THB space with enough levels has
  enough degrees of freedom to interpolate noise, which is sound, but the practical onset
  depends on the noise level and the marking threshold and has not been demonstrated on real
  data.
- **Not investigated:** the accuracy gap between THB quasi-interpolation and a true L2 THB
  fit, which decides whether the loop needs a solve at all. This is the cheapest useful
  experiment available and should be the first one run.

## Open questions

1. Exact data or measured data, or both? They need different stopping criteria and only the
   second needs regularization. If both, the distinction should be visible in the API rather
   than inferred.
2. Quasi-interpolation or L2 fit per level? Measure the accuracy gap first; if
   quasi-interpolation is close enough, the whole solve question disappears.
3. Refined region as a bounding box or as a union of boxes? Depends on whether refinement
   produces one compact feature or several, which for a level set is several.
4. Which error norm per cell (max, mean, sum of squares)? They mark different cells.
   Parameter, as an `enum class`.
5. Marking strategy: fixed fraction (refine the worst `θ` of cells, Dörfler-style), fixed
   threshold, or equidistribution? Not discussed here at all, and it changes the number of
   iterations to reach a given accuracy.
6. Does `refine` (`:1216`) have the same mutation problem as `refine_cells`? Only the latter
   was read.
