# Root-finding dimension: what pantr can provide, and what is gated

**Status:** design note for the C++ port. Nothing here is implemented.
**Date:** 2026-08-18.
**Scope:** which operations need root finding beyond one dimension, what pantr can ship on
its own, and how the gap to a certifying nD solver is expressed rather than hidden.
**Depends on two earlier decisions**, restated here so this note stands alone:

- Root finding is reached through a **batched seam**, a callable in the C++ API
  (`root_solver_fn`) that pantr ships its own implementations of and that a higher-capability
  provider can fill from outside. pantr's own code never names or includes that provider.
- The return type is **`RootResult`**: a list of `Root { x, tier, provenance, solver_id }`
  plus a `SolveStatus` bitmask. Solvers that certify nothing report `tier = NONE`. The tier
  enumeration runs from `NONE` through bracket-based uniqueness up to interval-arithmetic
  certification, so a weaker solver is expressible in the same type rather than needing a
  different one.

This note completes them by settling *which operations* need which tier.

**Decided (2026-08-19): the tier ladder is pantr's own, defined by what certificates mean,
and any external solver is translated in the adapter.** An earlier version of this plan had
pantr's enums share the *underlying integer values* of a particular external solver so the
adapter could be a `static_cast`. That is the wrong trade. It saves a `switch` over a handful
of values once per root, which is nothing, and in exchange it is **more** fragile: if the
external enum is renumbered or gains a value in the middle, pantr's mirrored enum is silently
wrong, whereas an exhaustive `switch` in a translation table fails to compile. Defining the
ladder by meaning (no certificate, bracket uniqueness, ball uniqueness, box uniqueness,
certified singular) also means a *second* certifying solver maps onto the same ladder without
renegotiating anything, and pantr publishes no one else's taxonomy.

**Naming:** the higher-capability nD solver is referred to throughout as *the plugin*. It is
a separate, privately licensed library and is deliberately not named here.

**Validated against:** pantr **0.7.0** (`main`, tag `v0.7.0`), 2026-08-19. Line numbers
below refer to that tree.

## The question

The plugin solves nD polynomial systems robustly and with certificates. pantr can ship a
good 1D solver without difficulty. Two dimensions is already hard, and certified two
dimensions is genuinely hard. So: which features does that gate, and is the answer "you have
point inversion on surfaces only if you install the plugin"?

## Two facts that reframe it

**Point inversion in nD already works, with no root solver at all.**
`src/pantr/bspline/_bspline_locate.py` does it in two stages: candidate cells from a BVH over
per-cell control-point boxes (the convex-hull property, valid for NURBS when every weight is
positive), then **Newton per candidate cell**, batched over all query points. Its docstring
already states precisely what it does and does not promise:

> *What is guaranteed is `F(ref_coords[i]) == points[i]` within the tolerance, and not that
> `ref_coords[i]` is any particular preimage. A mapping whose Jacobian determinant changes
> sign folds, and a folded mapping sends several parametric points to the same physical
> point; every one of them is a correct answer.*

A residual guarantee, and no uniqueness or exhaustiveness. That is `tier = NONE` written in
prose. So the answer to the framing question is **no**: surface and volume point inversion
already exist. What the plugin adds is a certificate, not a capability.

**pantr has no intersection operations.** `src/pantr/cad/` is entirely constructive:
extrusion, ruled, revolution, sweep, Coons, join, rectangle, disk, cylinder, compatibility
and validation. Every `intersect` match elsewhere in the package is axis-aligned box overlap
or grid-cell overlap, not curve or surface intersection.

So this is not a question about degrading an existing feature. It is a question about **which
future features are gated**, decidable per feature before any of them is built.

## The dimension table

| operation | system | unknowns |
|---|---|---|
| zeros of a scalar spline | `s(t) = 0` | **1** |
| point inversion on a curve | `C'(t) · (C(t) − P) = 0` | **1** |
| curvature extrema, inflections | one scalar equation | **1** |
| trimming curve in the parameter domain | one scalar equation | **1** |
| quadrature and extraction by dimension recursion | 1D per slice | **1** |
| point inversion on a surface | two orthogonality equations | 2 |
| curve-curve intersection in 2D | `C1(s) − C2(t) = 0` | 2 |
| curve self-intersection in 2D, closest point between curves | | 2 |
| curve-surface, ray-surface intersection | `C(t) = S(u, v)` | 3 |
| point inversion on a volume map | | 3 |
| surface-surface intersection | 3 equations, **4 unknowns** | a *curve* |
| silhouette, `n(u,v) · v = 0` | 1 equation, **2 unknowns** | a *curve* |

A nuance worth recording: curve-curve intersection **in 3D** is 3 equations in 2 unknowns, so
generically there is no solution. The 3D question is the closest-approach problem, which is
2 unknowns and belongs one row up.

## Two structural facts the table exposes

**The two most intimidating operations are not root-finding problems.** Surface-surface
intersection and silhouette computation are underdetermined: their solution sets are curves.
They are solved by tracing (marching plus local solves), not by finding isolated roots. An nD
root solver does not solve them; it helps find starting points. So they are much less gated
than they look.

**Dimension recursion converts a whole family of nD problems into 1D.** Slicing along one
direction and root-finding the resulting univariate polynomials is how Algoim-style quadrature
works, and it is why the downstream quadrature consumer's own root finder is one-dimensional.
Implicit-geometry work, cut-cell quadrature and marching-type extraction are all reachable
with a 1D solver.

Between those two facts, the genuinely gated set is smaller than the table's middle rows
suggest.

## Three categories, not two

**1. Reducible to 1D. Ship complete.** No plugin, no asymmetry. This is a larger set than it
first appears, per the dimension-recursion point above.

**2. nD where a good answer is a useful answer. Ship uncertified, return `tier = NONE`.**
Surface and volume point inversion (already shipped this way), closest point, projection. The
caller gets a correct residual; what they do not get is "this is the only preimage" or "all
solutions were found". With the plugin selected, the same call returns a higher tier and the
caller's code does not change.

**3. nD where an uncertified answer is dangerous. Do not ship uncertified.**

This is where the instinct to degrade gracefully is wrong. Curve-curve intersection feeding a
boolean operation: a missed intersection produces a **wrong solid**, and nothing downstream
can detect it. The same holds for anything whose correctness depends on the *count* of
solutions rather than on their positions: the topology of a trimmed region, winding numbers,
whether two profiles cross.

A status bit reading "the search was not exhaustive" is not sufficient protection, because
callers ignore status bits. The correct behaviour is for the operation to declare that it
requires a certifying solver and to fail loudly when one is not available. A missing feature
is better than a silently wrong solid.

## The missing piece: a required tier per operation

The return type already carries `tier` and `status`. What is missing is one declaration:
**each operation states the minimum tier it requires**, and solver selection fails loudly
when the chosen solver cannot reach it.

That turns the asymmetry into something explicit, machine-checkable and stated in one place,
instead of distributed across twenty docstrings as prose. It also means there is no need for
feature flags, no need for two parallel APIs, and no silent behaviour change when a plugin
appears or disappears: one API, the certificate as part of the contract, and a stated minimum
per operation.

Concretely it wants two things beyond what is already decided: a per-solver declaration of
the maximum tier it can produce, and a per-operation minimum. Both are small, both are
`enum class` valued, and both are checkable at solver-selection time rather than at the point
of failure.

## The middle rung: subdivision with convex-hull rejection

"Two dimensions is not feasible" is true of **certified-unique** two dimensions. It is not
true of exhaustive-to-tolerance two dimensions, and the distinction matters because it lets
pantr's own solvers occupy a middle rung rather than sitting at `NONE`.

The method: subdivide both Bézier pieces recursively, discard any pair whose control-point
convex hulls do not overlap, and stop when the parameter boxes are below tolerance. Roughly
200 lines, and it is what most working CAD libraries actually do.

**Why the discard is safe.** A Bézier piece lies inside the convex hull of its control points,
because the Bernstein basis is non-negative and sums to one. So if two hulls are disjoint, the
two pieces cannot meet, and discarding that pair cannot discard a solution. Every intersection
therefore lies inside one of the reported boxes. That is a real guarantee, of a weaker kind
than uniqueness: *no root outside the reported boxes*.

**Where it must round outward.** The hull-overlap test decides in floating point whether two
boxes are disjoint. A test that answers "disjoint" when they in fact touch discards a
solution and destroys the guarantee. So the separation test must be conservative: round the
hulls outward, and treat "too close to call" as overlapping. This is the exact-arithmetic to
floating-point crossing, and the guarantee above holds only with that rounding in place.

**Its named weak spot.** Subdivision separates well for transversal intersections and badly
for tangential ones: near tangency the hulls keep overlapping, the recursion runs to maximum
depth, and the result is a fat box that may contain one root, two, or none. The count is
exactly what a boolean operation needs, and near-tangency is exactly where booleans go wrong.
So the middle rung shrinks category 3 but does not empty it: **exhaustive-to-tolerance is
enough for a boolean when intersections are transversal and the tolerance is derived, and it
is not enough near tangency.** That boundary should be a documented, reported condition rather
than a silent one, which argues for a distinct status bit for "maximum subdivision depth
reached" rather than folding it into a generic non-exhaustive flag.

## What this means for the asymmetry

The asymmetry stops being binary. Instead of "with the plugin you can invert points on
surfaces, without it you cannot", the picture is a ladder:

| | without the plugin | with it |
|---|---|---|
| 1D problems | full, and certified where a bracket proves uniqueness | same |
| point inversion, closest point (nD) | works, `tier = NONE` | higher tier, same call |
| 2D intersection, transversal | exhaustive to a derived tolerance | certified unique |
| 2D intersection, near-tangential | reported as depth-limited | certified, or an honest refusal |
| counting-dependent topology | requires the declared minimum tier | available |

Nothing changes shape when the plugin arrives or leaves. The same functions return the same
type; the tier improves and the operations with a high declared minimum become available.

## Epistemic status

- **Verified by reading the code:** that `locate` performs nD point inversion via BVH
  candidates plus batched Newton, and disclaims uniqueness and preimage identity in the terms
  quoted (`src/pantr/bspline/_bspline_locate.py`, module docstring); that `src/pantr/cad/`
  contains only constructive operations and no intersection; that the package's other
  `intersect` occurrences are box and grid-cell overlap.
- **Derived from standard geometry:** every row of the dimension table, including that
  surface-surface intersection and silhouettes are underdetermined and that 3D curve-curve
  intersection is generically empty.
- **Argued, with the argument given:** that subdivision with hull rejection is exhaustive to
  box tolerance. The convex-hull step is standard; the outward-rounding requirement is stated
  because the guarantee fails without it. Neither has been implemented or tested here.
- **Asserted from experience of the failure mode, not measured:** that subdivision degrades
  badly near tangency. The mechanism (hulls fail to separate, recursion hits maximum depth) is
  sound, but no measurement of how close to tangency it starts to matter was made, and that
  threshold is what would decide whether the middle rung is adequate for booleans in practice.
- **Not investigated:** whether the 200-line estimate for subdivision plus rejection is
  realistic once outward rounding, the O(n·m) segment pairing and the depth-limit reporting are
  included. It is an estimate of the core recursion only.

## Open questions

1. Is curve-curve intersection actually wanted in pantr, or does it belong to a consumer? It
   is the operation that forces this whole discussion, and pantr does not have it today.
2. Where does the required-tier minimum live: on the operation's options struct, or as a
   compile-time property of the operation? Compile-time catches it earlier but makes the set
   of operations less uniform.
3. Should a solver advertise a maximum tier, or should the operation query "can you certify
   uniqueness in a box?" as a capability question? The tier is simpler; the capability query is
   more honest when a solver is strong in one dimension and weak in another, which is exactly
   pantr's own situation.
4. How close to tangency does subdivision stop being adequate? This is measurable with a
   parametrized family of near-tangential curve pairs and would settle whether category 3
   contains booleans or not.
