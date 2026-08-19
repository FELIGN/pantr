# Bézier extraction: subset extraction and cheap cell bounds

**Status:** design note for the C++ port. Nothing here is implemented.
**Date:** 2026-08-17.
**Scope:** the shape of the Bézier-extraction API. Not the extraction mathematics,
which is unchanged and already correct.
**Companions:** `design/large_data_fitting.md`, which reaches the same "apply a banded 1D
operator along axis `d`" primitive from the fitting side (the blocking strategy proposed
below is the same one that note needs), and `design/simd.md`, which explains why that
blocking is what makes vectorization of the shared kernel worth anything at all.

**Validated against:** pantr **0.7.0** (`main`, tag `v0.7.0`), 2026-08-19. Line numbers
below refer to that tree.

## The finding

`Bspline.to_beziers()` is all-or-nothing. It has no way to say *"just these cells"*.

Its main consumer asks about one cell at a time, so it is forced to materialize the
entire decomposition and then index into it. The extraction operators that would
answer the single-cell question directly are already computed and already cached, but
they are not reachable through the public API.

This is an API-shape problem, not a performance-tuning problem. It cannot be fixed by
parallelizing, vectorizing, or moving work to a GPU: those all speed up the step that
is not the bottleneck.

## Evidence

In this repository, `src/pantr/bspline/_bspline_to_beziers.py`:

- Lines 143-155 apply the extraction operators direction by direction in a Numba
  `prange` kernel. This part is efficient and is a small fraction of the total.
- Line 169 transposes the result, which makes it non-contiguous.
- Lines 172-180 then build **one Python `Bezier` object per element**, in a Python
  loop over `np.ndindex`, each preceded by an `np.ascontiguousarray` copy that the
  transpose made necessary:

  ```python
  result = np.empty(num_intervals, dtype=object)
  for idx in np.ndindex(*num_intervals):
      bez_ctrl = np.ascontiguousarray(ctrl_transposed[idx])
      bez_ctrl.flags.writeable = False
      result[idx] = Bezier(bez_ctrl, is_rational=is_rational)
  ```

In the downstream consumer, three places:

- Its single-cell helper, `to_bernstein_on_cell`, calls `bspline.to_beziers()` and then
  immediately indexes one element out of the result. One cell is wanted; every cell is built.
- A second site does the same to build a Bézier hull, stacking every patch's control points.
- All of its implicit-geometry primitives share one `to_beziers` cache, so the cost is paid
  once per spline rather than per call. The cache then holds the full decomposition for the
  object's lifetime.

`to_bernstein_on_cell` is, in effect, the single-cell entry point this API should have
offered in the first place.

## Cost model

All figures below are **derived** from the code and from array shapes, not measured.
A profile may reorder them; the structural factors do not depend on the constants.

Take a scalar level set (rank 1) on a tensor-product grid, degree `p = 3` in each of
`d = 3` directions, `n = 100` cells per axis, so `o = p + 1 = 4`.

| quantity | expression | value |
|---|---|---|
| input, B-spline control points | `(n + p)^d · 8 B` | ≈ 8.7 MB |
| output, Bézier coefficients | `n^d · o^d · 8 B` | ≈ 512 MB |
| blow-up factor | `≈ o^d` for `n ≫ p` | ≈ 59× |
| Python objects built | `n^d` | 10^6 |
| contiguity copies | `n^d`, of `o^d` doubles each | 10^6 |

The object construction and the copies are pure overhead: a consumer needs the
coefficients, never a per-cell Python object. In C++ the same result is one flat array
plus an `mdspan` of shape `(n_el, o^d)`, and the overhead disappears entirely rather
than being reduced.

### Cut cells are a small fraction of all cells

For a level set the interface has codimension 1, so the number of cells it crosses is
`O(n^(d-1))` out of `n^d` total. At `n = 100, d = 3` that is order 10^4 of 10^6, about
1%.

Algoim only needs the Bézier form of the cells the interface actually crosses.
Materializing 10^6 to use 10^4 is a factor-100 waste, and it is the dominant one. It
is also the only one that survives any amount of optimization of the extraction itself,
because it is work that should not happen at all.

## Where cell filtering belongs: not here

A downstream consumer already classifies cells, and its pipeline is more capable than a
sign test in pantr would be on four counts that matter here: it is **conservative by
construction** (a cell it cannot prove inside or outside is labelled cut), it works over a
**boolean combination** of level sets rather than a single one, it prunes **hierarchically**
rather than cell by cell, and the conservative pass is followed by a **certification stage**.

pantr must **not** add a competing classifier. A `cell_may_be_cut` in pantr would be
single-level-set, non-hierarchical, and non-conservative-by-construction, so it would be
a strictly worse duplicate of the above.

What pantr owes the classifier is the primitive the Bernstein probe needs, delivered
without building anything the caller did not ask for.

### The boundary has three columns, not two

"The consumer already has it" is true of the classifier and false of the extraction. Keeping
those separate is the whole point of this note, because collapsing them into "nothing to
do in pantr" leaves the factor-100 waste in place.

| Stays with the consumer | Should move to pantr | pantr already does it, unexposed |
|---|---|---|
| the classification pipeline and its conservativeness | the single-cell Bernstein extraction helper, raising and non-raising | per-cell convex-hull **value** bound |
| the boolean layer over several level sets | | Bézier restriction to a sub-box |
| the certification stage | | knot-span location for a box |
| the predicate probed during hierarchical traversal | | BVH `traverse(visitor)` |
| the cut-cell quadrature algorithm | | |

### Why `to_bernstein_on_cell` belongs in pantr

The consumer's single-cell helper reads, in order:

1. locate the knot span containing the cell box (`_locate_single_knot_span`);
2. `self._bspline.to_beziers()` and index the element;
3. map the cell box to local span coordinates `t0, t1`, clipped to `[0, 1]` to guard
   against floating-point overshoot at the span ends;
4. `bezier.restrict(bounds_per_dim)`, which is **pantr's own method**, with an `is_full`
   fast path that skips it when the box is the whole span;
5. squeeze the trailing `(1,)` axis that rank-1 splines carry;
6. validate the resulting shape and return a contiguous array.

Not one of those six steps is about implicit geometry, CSG, or classification. Every one
is a spline operation, and steps 1, 2 and 4 are operations pantr already has. This
function exists downstream because pantr offers no single-cell entry point, not because it
is the consumer's algorithm.

Two consequences for the port:

- With `extract_cell` plus a restriction path, this becomes a thin wrapper, and the
  `hasattr` probe at `:897` has nothing left to probe.
- Step 5 is a workaround for a pantr shape convention: whether a rank-1 spline's
  coefficients carry a trailing dimension of `1`. The C++ port decides that convention
  fresh, so the workaround should not need to exist. Recorded here so the decision is
  made deliberately rather than inherited.

The `to_` / `try_` pair (raising and non-raising variants of the same operation) is the
same smell that `D23` resolved for root finding: the outcome belongs in the return value,
not in the choice of which of two functions to call.

### On the convex-hull value bound

This is listed as "pantr already does it" rather than as new work, because pantr already
computes the *geometric* version of exactly this: `_bspline_locate.py:13` builds a BVH
over "per-cell physical control-point boxes", which is the convex-hull bound of the
geometric map on each cell. The scalar-value bound is the same operation at rank 1. What
is missing is the entry point, not the capability.

## Proposed API

Names are provisional. The shapes are the point.

```cpp
// Bernstein coefficients of ONE cell, written into a caller-owned buffer.
// No allocation, no object construction. Tier A, device-compatible.
void extract_cell(std::size_t cell_id, std::span<double> out) const;

// An arbitrary SUBSET of cells, in one call, so the sweep can share partial
// contractions across the batch. This is the entry point that is missing today.
void extract_cells(std::span<const std::size_t> cell_ids,
                   std::span<double> out) const;

// A bound on the spline over one cell, WITHOUT extracting it. See below for
// what makes this valid and how tight it is.
std::pair<double, double> cell_value_bounds(std::size_t cell_id) const;

// The extraction operators themselves, already computed and cached today.
// Small: n_el x (p+1)^2 per direction, so kilobytes.
std::span<const double> operators(int direction) const;
```

`to_beziers()` stays, for the caller who genuinely wants all of them. In C++ it returns
a flat array plus an `mdspan` of shape `(n_el, (p+1)^d, rank)`, never a container of
per-cell objects, and its docstring states the memory cost with the `o^d` factor
explicit.

The resulting consumer loop becomes: classify (reading the 8.7 MB control-point array),
then `extract_cells` on the surviving 1%, then Algoim. Instead of: materialize 512 MB,
then classify, then use 1% of it.

## The convex-hull bound

`cell_value_bounds` rests on a claim, so here is what makes it true.

The B-spline basis functions are non-negative and sum to one at every point of the
domain (partition of unity). A spline restricted to one cell is therefore a **convex
combination** of the control points whose supporting basis functions are non-zero on
that cell. Hence

```
min(P_i : i in supp(cell))  <=  s(u)  <=  max(P_i : i in supp(cell))    for all u in cell
```

This is the standard convex-hull property of B-splines; see de Boor, *A Practical Guide
to Splines*. It is not new and is not being claimed as such. The point is that it
requires **no extraction at all**: it reads `(p+1)^d` values out of the control-point
array and compares them.

Two things to be honest about:

1. **It is a bound, not a classification.** If the bound straddles zero the cell may or
   may not be cut. Only the one-sided case is conclusive.
2. **It is looser than the Bernstein bound.** Bézier extraction tightens the convex hull,
   so a cell that the Bernstein coefficients would prove one-signed may be inconclusive
   under the B-spline coefficients. That is acceptable for a pre-filter: if it discards
   most cells for free, the tighter test only runs on the survivors.

The chain is therefore: cheap B-spline bound, then extraction, then the existing
Bernstein probe. Each stage is conservative, so no cut cell is lost at any step.

## A tradeoff to record, because the obvious answer is wrong

Streaming cell by cell is **not** strictly better than the current global sweep. The
global sweep is cheaper in arithmetic.

Applying the directions one after another over the whole array expands progressively,
so with `n` cells per axis and order `o`, in 3D:

| stage | array shape after | work |
|---|---|---|
| direction 0 | `(n·o, n, n)` | `n³o²` |
| direction 1 | `(n·o, n·o, n)` | `n³o³` |
| direction 2 | `(n·o, n·o, n·o)` | `n³o⁴` |

Total `≈ n³o⁴ (1 + 1/o + 1/o²) ≈ 1.3 n³o⁴`.

Per cell, by sum factorization: gather `o^d`, then one 1D operator per direction, so
`d·o^(d+1) = 3o⁴` per cell, times `n³` cells, giving `3 n³o⁴`.

So per-cell extraction does roughly **2.3× more floating-point work** than the global
sweep, because the global sweep shares partial contractions between neighboring cells.
What it buys instead is memory and cache locality: 8.7 MB of input plus one cell's `o^d`
doubles, against 512 MB read cold.

In this regime, arithmetic intensity is between roughly 0.1 and 1.5 flops per byte, so
the memory side is expected to dominate. That expectation is **not measured** and should
be before anything is tuned around it.

The resolution is not to pick one. Extract in **blocks** of cells (a tile of, say, 8³):
partial contractions are shared inside the block, and peak memory stays bounded. This
composes with the consumer loop, which processes a tile and discards it. It also
subsumes the subset case, since a filtered cell list can be grouped into tiles.

## Epistemic status

- **Verified by reading the code**, in the files and at the lines cited: the object
  construction and copies at `_bspline_to_beziers.py:172-180`; that the consumer's
  `to_bernstein_on_cell` calls the full `to_beziers()` and then indexes a single cell; that
  its classifier already exists and already probes cells with a Bernstein-based test; and
  that all six steps of that helper are spline operations with no geometry-classification
  content, read in full.
- **Derived**, not measured: every byte count, flop count, and the 2.3× ratio.
- **Inferred from geometry**, not measured on real inputs: that cut cells are `O(n^(d-1))`
  and therefore around 1% at `n = 100, d = 3`. The exponent is structural; the constant
  is not, and a badly aligned or highly curved interface will do worse.
- **Not investigated:** whether the consumer's cache makes the one-time
  512 MB acceptable in practice for the problem sizes actually being run.

## Open questions

1. Does the block/tile size want to be a parameter, or derived from a cache-size
   estimate? A derived default with an override is the usual answer, but the derivation
   needs writing down rather than guessing a number.
2. `extract_cells` takes a cell-id list. For a tile-shaped subset an id list is wasteful.
   Worth deciding whether a second entry point taking a box of cell indices is warranted,
   or whether the id list is good enough given that the consumer's filter produces an
   arbitrary set anyway.
3. `to_bernstein_on_cell` should move to pantr (see the boundary section for why). What
   is genuinely open is the shape it takes there: a single call taking a cell box, or the
   composition `locate span` + `extract_cell` + `restrict` left to the caller. The
   composition is more honest about what happens; the single call is what the consumer
   actually wants. Probably both, with the single call as the documented path.
4. Whether a rank-1 spline's coefficients carry a trailing dimension of `1`. Inherited
   from the Python conventions and currently worked around downstream. The port should
   decide it rather than reproduce it.
4. Noted in passing, out of scope here: the consumer's single-cell helper does
   `if not hasattr(sample_patch, "control_points")`, which is capability probing on a
   project type. Once the return type is a flat array plus a shape, that check has
   nothing to probe and should go.
