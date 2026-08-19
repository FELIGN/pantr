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

## Why, and the reason is not the obvious one

**The BVH is not a black box here. It is a public data structure that a consumer traverses
with its own predicate.** A downstream consumer does not call `query_aabb`; it walks the
tree itself, stack-iteratively, probing at each *internal* node with its own predicate,
pruning subtrees and delegating at leaves. So the five-array node layout
(`node_lo` / `node_hi` / `node_left` / `node_right` / `node_cell`, exposed as properties)
is **public API**, not an implementation detail, and swapping implementations means
rewriting that consumer's classifier.

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
  properties and `query_aabb` as the only query (`_bvh.py`); that a downstream consumer
  traverses the node arrays directly with its own predicate at internal nodes; that
  tensor-product point location does not use the tree (`_locate_core.py:4`).
- **Derived:** that ray-tuned advantages do not transfer to box-overlap queries.
- **Not measured:** how much a binned SAH build would actually help on the physical
  control-point boxes. That is the experiment that would justify item 2.
