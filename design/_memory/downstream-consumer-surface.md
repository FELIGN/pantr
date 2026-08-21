---
name: downstream-consumer-surface
description: What the downstream consumer actually imports from pantr, measured 2026-08-21; the PointsLattice debt is discharged and bezier is the exposed module
metadata:
  type: project
---

**Measured on 2026-08-21**, by grep over the consumer's own checkout, on Pablo's other machine.
This replaces two PRs' worth of guessing. It is a snapshot: that repository moves, so re-run
before relying on it for anything expensive.

**The `PointsLattice` debt is discharged.** The consumer never mentions `PointsLattice` or
`pts_per_dir`. Freezing them (#338, #344) broke nothing.

**The package split broke nothing either, and none of the nine restored names was needed.** The
consumer imports exactly one name from `pantr.change_basis`: `compute_monomial_to_bernstein_1d`,
which is public, in `__all__`, and never moved. Restoring `LagrangeVariant` and the other eight
was cheap insurance and correct as policy, but not one of them was in use.

**What it does import privately, and this is the part that matters next:**

| symbol | module | on `main`? |
|---|---|---|
| `_de_casteljau_eval_scalar` | `pantr.bezier._root_finding_core` | yes, a `@nb_jit` kernel |
| `Bezier` | `pantr.bezier._bezier` | yes, but the class is *public* as `pantr.bezier.Bezier`; only the path is private |
| `_BVH_STACK_DEPTH` | `pantr.grid._bvh_core` | yes, `Final[int] = 128` |

**So `bezier` is the first port that will touch symbols the consumer imports**, unlike `quad` and
`change_basis`. `grid` too. A Layer 3 Numba kernel is exactly what a port reorganises, so
`_de_casteljau_eval_scalar` needs deciding deliberately rather than discovering. See
[[active-task]].

**On float32 and the width question: the consumer settles it.** The one function it uses is the
one builder that runs no solve, so widening the Gram products and the LU to double cannot affect
it. It never passes `dtype=np.float32` anywhere near pantr; float32 appears in its tree only in
an STL writer and three unrelated tests.

**But it depends on exact float64 values**, deliberately and with a test that says so. It has a
test literally labelled `DEPENDENCY TRIPWIRE` asserting `compute_monomial_to_bernstein_1d(d)[:, 0]`
is all exact ones; it compares whole matrices with `assert_array_equal`; and it converts entries
to `Fraction(float(v))` for an exact-rational soundness proof. Verified 2026-08-21 that all of
that survives the C++ port, both backends, degrees 0 to 56.

See [[naming-and-licensing]] for why the consumer is never named in anything written down.
