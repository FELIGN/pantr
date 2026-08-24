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

**Open, not blocking.** A third pre-existing defect is reproduced and localised but **not filed**:
Bézier clipping loses a root whose computed residual is exactly zero, at a tolerance below the
float32 noise floor. Yuksel finds it at every tolerance. Awaiting Pablo's go-ahead. And one review
finding was downgraded rather than fixed: `ParityClaim` carries optional fields meaningful for one
variant, which is pre-existing and which a tagged union would fix across every parity module.
