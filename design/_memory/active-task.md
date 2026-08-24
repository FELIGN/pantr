---
name: active-task
description: bezier's root-finding block merged; block C (interpolation) is next, and what the review of block B established
metadata:
  type: project
---

**Updated 2026-08-24.** `bezier`'s root-finding block is **merged** (#353, six PRs now on
`proto/cpp`). Remaining in `bezier`: **block C, `_bezier_interpolate`**, 756 lines, no kernels,
an SVD pseudo-inverse of a Bernstein Vandermonde with **rank truncation** whose threshold is
discontinuous. Checkable, because the matrices are deterministic per degree: verify no singular
value sits within a few eps of the threshold. First use of Eigen outside `change_basis`.

The question block B opened is **closed and it dissolved**: `_de_casteljau_eval_scalar` never had
to move. All kernel-to-kernel calls in that block are inside `nopython`, so no catalogue can be
inserted between two of them and the dispatch boundary is forced up to `_find_roots.py`. See
[[downstream-consumer-surface]].

**What block B's review established, and it is the reusable part:**

- **A conditional parity claim's other branch ships unevaluated, and that is where the bug is.**
  This port's fused branch called a harness function with an argument it does not take, and 133
  tests passed over it because none reached it. Found by building at `-march=native` and running
  the suite there. Do that as part of finishing any port that carries a conditional claim.
- **Do not predict an acceptance width, certify it.** Two predicted bounds were refuted. The one
  that ships restricts the Bernstein net to each flank in exact rational arithmetic and uses the
  convex-hull property to prove no point there is acceptable. Scale-invariant, dimensionally
  correct, multiplicity-agnostic, and it needs no derivative. See Rule 11.
- **Derive against the threshold the code states, not one you invent for it.** The root mistake
  under three of the four refutations was deriving a forward error when `_clip_roots_core`
  computes its own `zero_tol`. For the simplest polynomial in the suite the real band was exactly
  twice the claim.
- **Numba's `float()` does not widen a `float32`.** Type unification across assignments does.
  Three of six measured widths follow from that alone.

**Two facts to carry into block C and into grid.** `scripts/measure_root_finding_widths.py` is
the shape a port's specification should take: rival models per site, measured against the kernel,
with a discrimination count so a match cannot come from a check that could not fail. And
`design/backend_parity.md` now has **eleven** rules; Rule 11 is the first about a discrete verdict
rather than a displacement.

**Block C reconnaissance, already measured, do not repeat it.** `_bezier_interpolate.py` truncates
at `sigma_i < 100 * eps * sigma_0`. At **float64 that branch is dead** up to n=39, closest approach
350x. At **float32 it fires from n=19**, and at n=36 a singular value sits **0.22% above the
threshold**, which two SVD implementations will not agree on. When the rank differs the
pseudo-inverse moves by a rank-1 term of order `1/(tol*sigma_0)`: a verdict, not a displacement.
The open question that may change everything is whether the pseudo-inverse is **cached**; if the
SVD runs once per degree per process, not porting this block is a serious option.

**Three tickets filed 2026-08-24**, all pre-existing and none fixed: #351, #352, #354. All three
are tolerance-derivation defects in the same kernels and one coherent rework may close all three.
The C++ reproduces #351 and #352 deliberately, asserted by name.

**One review finding downgraded rather than fixed:** `ParityClaim` carries optional fields
meaningful for one variant. Pre-existing for four; this port added two more.
