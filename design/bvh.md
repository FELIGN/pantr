# The BVH: keep pantr's own, and what would change that

**Status:** decided. Validated against pantr **0.7.0**.
**Date:** 2026-08-14, revalidated 2026-08-19.
**Scope:** whether the C++ port should adopt a header-only third-party BVH, and what to
improve instead.

**Validated against:** pantr **0.7.0** (`main`, tag `v0.7.0`), 2026-08-19. Line numbers
below refer to that tree.

## The decision

The port **keeps and ports pantr's own BVH** (`src/pantr/grid/_bvh.py` plus
`_bvh_core.py`, 664 lines) rather than adopting a header-only third-party library.

## Why, and the stated reason has changed

**The reason first given for this decision is no longer true. It is corrected here rather
than deleted, because everything below was written under it.**

As written on 2026-08-14 and revalidated on 2026-08-19, the argument was: the BVH is not a
black box, it is a public data structure that a downstream consumer traverses with its own
predicate; that consumer does not call `query_aabb` at all, it walks the tree
stack-iteratively, probing at each *internal* node, pruning subtrees and delegating at
leaves. The five-array node layout (`node_lo` / `node_hi` / `node_left` / `node_right` /
`node_cell`) was therefore public API rather than implementation detail, and swapping
implementations meant rewriting that consumer's classifier.

**A census of that consumer's checkout on 2026-08-29, against its own head of that day,
found zero references to `BVH`, `node_left` or `query_aabb`, and zero uses of
`pantr.grid._bvh_core._BVH_STACK_DEPTH`.** The two `node_lo` hits are its own local
variables in an unrelated file. From this subsystem it imports `Partition` and
`tensor_product_grid` and nothing else, and it writes no attribute of any pantr object. So
the caller the paragraph above describes does not exist any more, and **that sentence must
not be cited again as written.**

Re-run the census before relying on this: the repository moves, and the previous census was
a week stale when it was contradicted. The command is a grep for `import pantr` over the
sibling checkouts, then three separate counts -- private symbols, public symbols, and
**attribute writes**. The last is the one the earlier census omitted, which is how three
writable `BVH` attributes went unexamined until the port privatised them.

**The decision itself stands, on the remaining grounds.** The layout was kept unchanged
through the port with this known: a repository that moved once can move back, the node
arrays are public surface of this library whether or not anyone reads them today, and the
three secondary reasons below never depended on that consumer at all. `traverse(visitor)`
is still ordered first among the improvements, for reasons of its own.

Three secondary reasons:

- **The only query is `query_aabb`** (`_bvh.py:356`), axis-aligned box overlap. Every
  candidate library is tuned for *ray* queries, so their main advantages (SAH quality, ray
  packet traversal, SIMD ray-box tests) buy nothing here.
- **Preorder determinism is a declared invariant** in the module docstring, and several
  fast ray-tracing BVHs reorder primitives or build non-deterministically in parallel.
- **The two-pass count-then-emit query** is exactly what the no-dynamic-allocation kernel
  discipline requires; third-party libraries use callbacks or push into a `std::vector`.

## What would reverse it

Reopen the question only if **both**: a `traverse(visitor)` API lands first, hiding the
layout behind a callable predicate, **and** the query pattern gains genuine ray queries.
Neither alone is enough.

## What to improve instead, in this order

1. **A `traverse(visitor)` API**, so the consumer stops reading the five node arrays. This
   comes first because it is what makes the other three non-breaking.
2. **Optional binned SAH**, selected by an `enum class`. It would help the real weak case:
   `_bspline_locate.py` indexes per-cell *physical* control-point boxes, which overlap
   heavily on curved geometry and vary by `2^level` on THB spaces, and median-of-longest-axis
   splits produce fat overlapping nodes exactly there. **It requires re-deriving the fixed
   stack depth**, since median splits' `ceil(log2 N) + 1` height bound no longer holds.
3. **`int32` node indices** instead of `int64`. With `N` cells there are at most `2N - 1`
   nodes.
4. **A decision on clustering 4 to 8 cells per leaf.** Today it is one cell per leaf and
   `2N - 1` nodes. Clustering cuts the node count several-fold and improves cache behaviour
   at the cost of more leaf tests. **It breaks the layout, so it is now or never.**

## Not a weakness

Point location on a tensor-product grid already avoids the tree entirely
(`src/pantr/grid/_locate_core.py:4`, an independent binary search per axis). The BVH appears
only where there is no structure to exploit, which is its correct place.

## Candidates surveyed, and why each was set aside

Kept for the day the question returns, with the caveat that **all of these
characterizations predate the May 2026 knowledge cutoff and are unverified against current
releases**. Re-check feature sets, C++ standard requirements, licences and determinism
guarantees before adopting any of them.

- **`madmann91/bvh`** (v2): header-only C++20; SAH, binned SAH and sweep SAH builders,
  spatial splits, closest-point queries. The most general-purpose of the set and the first
  to look at again.
- **`tinybvh`** (Jacco Bikker): single header, very fast, ray-tracing focused, SIMD/AVX
  paths. Reorders primitives, which is a determinism concern.
- **`nanort`**: single header, ray tracing.
- **Embree** (Intel): *not* header-only, a heavy binary dependency with ISPC kernels.
  Excellent for rays.

## Epistemic status

- **Verified by reading the code**, revalidated against 0.7.0: the five node-array
  properties and `query_aabb` as the only query (`_bvh.py`); that tensor-product point
  location does not use the tree (`_locate_core.py:4`).
- **Measured, and now false:** that a downstream consumer traverses the node arrays directly
  with its own predicate at internal nodes. True when measured on 2026-08-14; **refuted by
  the 2026-08-29 census**, which found no reference to the BVH at all. See the correction
  above.
- **Derived:** that ray-tuned advantages do not transfer to box-overlap queries.
- **Not measured:** how much a binned SAH build would actually help on the physical
  control-point boxes. That is the experiment that would justify item 2.
