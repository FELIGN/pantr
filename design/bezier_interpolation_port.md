# Bézier interpolation: why the block stays in Python

**Status:** decided 2026-08-24. The memoization described in "What ships instead" is
implemented; nothing else here is.
**Scope:** `src/pantr/bezier/_bezier_interpolate.py` (756 lines) and the part of
`src/pantr/_interpolation_utils.py` it uses. This is the third and last block of the
`bezier` port, after the arithmetic (#348, #349) and the root finding (#353).
**Decision:** the block is **not ported**. `bezier` closes as *ported except interpolation*.

The other two blocks were ported because measurement said they would pay. This one was
measured the same way and said the opposite, so the note records the numbers rather than the
intention, and states what would have to change for the answer to change.

## What the block does

Both public entry points, `interpolate_bezier` (samples a callable) and `fit_bezier`
(pre-evaluated values), recover Bernstein coefficients from a Bernstein Vandermonde system.
The matrix is severely ill-conditioned at high degree, so every path inverts it through a
**truncated SVD pseudo-inverse**: singular values below `tol * sigma[0]` are zeroed, with
`tol = SVD_TOL_FACTOR * eps` and `SVD_TOL_FACTOR = 100.0`
(`_interpolation_utils.resolve_svd_tolerance`).

Three call sites take their own `np.linalg.svd`, and which of them a call reaches is not
obvious from the module's shape:

| site | matrix | determined by | on a public path? |
|---|---|---|---|
| `_build_bernstein_pinv` | 1D nodes, square or tall | the nodes, `tol`, `degree` | **yes**, every tensor-product call, once per direction |
| `_fit_from_scattered` | tensor-product basis at scattered points | the caller's points | **yes**, the scattered branch of `fit_bezier` |
| `_bernstein_vandermonde_svd` | square, at modified Chebyshev nodes | `(n, dtype)` alone | **no** |

The last row is worth pausing on, because the module docstring presents it as a supporting
utility and it reads like the main one. Nothing in this repository calls it outside the tests:
its only caller is `_bernstein_interpolate`, which nothing here calls either.

## The five measurements that decided it

`scripts/measure_bezier_interpolation_port.py` reproduces all of them, and writes out the
matrices and the Eigen comparison program that measurement 2 needs a compiler for. Run it as its
docstring says: it refuses to time anything unless the thread counts are pinned.

**1. The SVD dominates, and it was not memoized.** It accounted for between a seventh and a
half of a tensor-product call, and about three fifths of a scattered one, and every call
recomputed it, once per parametric direction.

**2. Eigen's `BDCSVD` and LAPACK's `gesdd` agree on the rank.** This was the expected blocker.
The truncation is a *discrete verdict*, so two SVD implementations disagreeing about whether a
singular value clears the threshold would move the pseudo-inverse by a rank-one term of order
`1 / (tol * sigma[0])` rather than by a few ulps. That is `backend_parity.md`'s Rule 11 family,
and no tolerance bounds it.

It does not happen. Over the real Bernstein Vandermonde matrices at orders 10, 20, 25, 30, 36
and 39, in both dtypes, the two implementations chose the **same rank in every case**. The
regime where it could bite at all is float32 from order 20 up, where truncation actually
fires; float64 does not truncate at any order measured, up to 39, and its closest approach
to the threshold is more than two decades away. The tightest float32 case, order 36, sits about
two parts in a
thousand above the cut, and the two implementations placed it within about one part in ten
thousand of each other: roughly a factor of twenty of headroom.

That is **measured on the matrices this code builds, not proved**. Six orders is thin evidence
for a discrete verdict, which is precisely why it is not by itself a reason to port.

**3. Eigen's SVD is not the faster one.** `BDCSVD` came in between about four tenths and about
one and a half times LAPACK's time on the same matrices, faster at float32 and slower at
float64. There is no speed argument for moving the dominant term.

**4. The portable numerical work is about a fifteenth of a call.** This is the sharpest of the
four. Once the factorization is memoized, a warm one-dimensional call is dominated by
Python-level work, and the part a C++ kernel would actually replace, one `tensordot` per
direction, is around seven percent of it. Twice that again goes to the caller's own Python
function, which `interpolate_bezier` is *handed* and no port can remove. The rest is
validation, lattice construction and component splitting: portable in principle, but only by
moving the entry point itself into C++, which a Python callable in the signature forbids.

So the ceiling is not a small constant factor. It is a few percent.

## The scope facts that settle it

- **The downstream consumer imports neither entry point.** It takes `Bezier`,
  `_de_casteljau_eval_scalar` and one BVH constant. Interpolation is not on its surface.
- **It never uses float32 near pantr.** The one regime carrying the rank-discontinuity risk is
  a regime nothing in Stage 1 scope enters.
- **`_fit_from_scattered` is shared with `bspline`** (`_bspline_interpolate.py`), which is
  explicitly out of Stage 1 scope. Porting it would drag a boundary this stage drew.

So the risk lives where nobody goes, and the gain lands on functions nobody in scope calls.

## What ships instead

One memoization, of a pure function of hashable inputs, bitwise identical to what it replaces.

**`_build_bernstein_pinv`.** It is the single site both `interpolate_bezier` and `fit_bezier`
reach for a tensor-product fit, once per parametric direction, and its truncated SVD is the
largest cost in the call. It is memoized on
`(nodes, tol, degree)`, keyed on the nodes' **bytes and dtype** rather than the array object,
so a regenerated node set hits the same entry. That is what makes it work at all: the default
path builds its Chebyshev nodes fresh on every call, so a key based on identity would never
hit. The dtype belongs in the key because two arrays of different dtype and length can share a
buffer length, and the tolerance and degree belong in it because the truncation depends on
both.

Measured with threads pinned, a tensor-product call runs roughly three to eight times faster,
the gain growing with the order. A square two-dimensional grid gains twice over, since both
directions share one node set and therefore one entry. The scattered path does not move: it
goes through `_fit_from_scattered`, whose Vandermonde is built from the caller's own points, and
that site is
left alone.

**`_bernstein_vandermonde_svd` was memoized too, and then it was not.** Its matrix is
determined outright by `(n, dtype)`, since it generates its own nodes, so memoizing it is just
as sound and just as free. It was dropped anyway, because it has **no production caller in this
repository**: only tests reach it, and its own caller `_bernstein_interpolate` has none either.
A cache on a function nothing calls is speculative, and it would have needed its own size
policy to sit beside the one below. The function now carries a note saying so, and this section
is the record of the option, in case the "resultant pipeline" its module docstring names turns
out to be real. It did gain a first direct test on the way past.

The entry point hands out a **copy**, so callers keep the independent, writable array they have
always had. Copying is quadratic in the order where the factorization it avoids is cubic, and
building the byte key is cheaper still, so neither erodes the saving.

Building the pseudo-inverse from the factors, and applying it, together cost a few percent of
the factorization, so there was nothing to gain by memoizing at a finer grain.

**A size guard keeps the cache from becoming a leak.** Entry count alone does not bound memory:
an entry is a whole `(degree + 1)`-by-`n_pts` matrix, and `n_pts` is the caller's to choose. The
shape that pays worst is a wide, low-degree fit against very many nodes, where the saving scales
with the degree while the retained memory scales with the node count, and which is usually
performed once anyway. Above `_PINV_CACHE_MAX_ELEMENTS` the factorization is computed and
discarded. That cap is a **policy choice and says so in its docstring**, not a derivation; it is
set so a full cache stays under twenty megabytes while still admitting every square case this
module is accurate at.

## A machine artifact worth recording

On the development machine, with the thread-count variables set to 20, a 36-by-36 LAPACK SVD
cost between one and two orders of magnitude more than the same call pinned to one thread, and
its timing varied by nearly an order of magnitude between batches. Twenty threads synchronizing
over a matrix that fits in L1 is pure overhead, and the box is shared.

The consequence is procedural: **any timing of this block on a many-core machine is worthless
unless threads are pinned**, and a comparison against a single-threaded C++ implementation on
those numbers would flatter the port by a factor that has nothing to do with the code. The
figures above were taken with threads pinned for this reason.

## What would reopen this

- The downstream consumer starts importing `interpolate_bezier` or `fit_bezier`, or starts
  passing float32 into them. Both are checkable by grep over its tree.
- `bspline` enters the port's scope, which brings `_fit_from_scattered` with it and changes the
  shared-site argument.
- A profile of real downstream work shows interpolation on a hot path. Nothing measured so far
  suggests it.
- `_bernstein_interpolate` acquires a caller, in which case memoizing its factorization is the
  cheap thing to do before anything else is considered.

If it does reopen, the parity claim cannot be an equality. It would need a derived bound plus a
**named exclusion** under Rule 8 for float32 at order 20 and above, and the rank agreement would
need far more than six orders behind it.

## Epistemic status

- **Verified by running it:** all five measurements above, and the machine artifact. The rank
  agreement was checked by feeding the same matrices to both implementations and comparing
  singular values and the resulting ranks.
- **Verified by reading the code:** the three SVD sites and what determines each matrix; that
  `_build_bernstein_pinv` is the only one on the tensor-product path, that
  `_fit_from_scattered` is the only one on the scattered path, and that
  `_bernstein_vandermonde_svd` is on neither; that the former is a pure function of its three
  arguments and the latter of `(n, dtype)`; that `_fit_from_scattered` is imported by `bspline`;
  that no in-tree caller mutates what either returns.
- **Taken from an earlier measurement, not re-checked here:** what the downstream consumer
  imports and that it never passes float32 (recorded 2026-08-21 against that repository, which
  moves).
- **Asserted, not proved:** that Eigen and LAPACK agree on the rank in general. Six orders in
  two dtypes is evidence, not a theorem, and the note leans on the scope facts rather than on
  this.
- **Not investigated:** why LAPACK's threaded path is so much worse here than its serial one.
  It is recorded as an artifact affecting measurement, not diagnosed.
- **Got wrong once, recorded so it is not repeated.** The first version of this work quoted a
  speedup measured *unpinned*, and it was thread noise rather than a saving; and it memoized
  `_bernstein_vandermonde_svd` believing it was the site on the public path, which it is not.
  Both were caught the same way, by instrumenting the cache and finding it took no hits at all.
  A speedup that survives neither a thread pin nor a hit counter is not a speedup.

## Open questions

- The module docstring of `_bezier_interpolate.py` says its supporting utilities are "used
  internally and by the resultant pipeline". No resultant pipeline exists in this repository,
  and `_bernstein_interpolate` has no caller here outside its tests. Either the phrase is stale
  or it names an out-of-tree consumer; it was left untouched because neither could be
  established from here, and it is the reason the second memoization was kept rather than
  dropped as dead.
