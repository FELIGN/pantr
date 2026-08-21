---
name: active-task
description: The pantr C++ port: change_basis merged, three of six Stage 1 modules done, and what the deep review taught
metadata:
  node_type: memory
  type: project
  modified: 2026-08-21T00:00:00.000Z
---

**`change_basis` is MERGED** into `proto/cpp` (PR #347, 18 commits, rebase-merged so the SHAs
differ from the branch's). Nothing is in flight, no worktree, remote has only `main` and
`proto/cpp`. Three of Stage 1's six modules are ported: `basis` (cardinal B-spline, Bernstein,
Legendre), `quad`, `change_basis`.

**`bezier` is the next module, not `geometry` or `transform`.** An earlier brief recommended
those two for being smallest, from line counts. Each is a **single class** (`AABB`,
`AffineTransform`) whose arithmetic is over 2- or 3-element arrays, and D2 keeps the class in
Python, so porting them ports almost nothing. `bezier` is where Eigen's SVD gets used, which is
half the justification for having taken the dependency.

**Eigen is now a dependency of the shipped extension**, with its licence files vendored in
`licenses/`. See [[decisions-pointer]].

**Rule 8 exists and the next three ports inherit it**, in the corrected form: a parity claim
needs a bound that can still say something, so the parity domain is where `constant * n * kappa
* eps < 1`. Its first version claimed the excluded degrees had no correct digits, which is false
(3.2 digits at the first excluded degree) and was caught by the review, not by me.

**The deep review is the thing to read before the next port**: `reviews/347-2026-08-21.md` in the
repo. Nineteen findings, and the shape matters more than the list: **the C++ was right and the
prose about it was wrong, repeatedly**. A fabricated citation to a test that never existed, an
exponential bound contradicting a PROVED result in the file I was editing, the wrong Higham
theorem, three false completeness claims, two numbers measured from one run each. See
[[reviewing-my-own-measurements]], which said exactly this about me before the session started.

**Two facts from it worth carrying into `bezier`:**
- The LU growth factor for these matrices is measured, not assumed: `R <= 3.73` in exact
  rational arithmetic, `rho` exactly 1. Eigen's `PartialPivLU` has **no tolerance at all** on its
  path (only `is_exactly_zero`), so the `computeFromTridiagonal` defect class cannot occur there.
- `np.linalg.solve(A, eye(n))` is Du Croz and Higham's Method A for **inversion**, not a solve,
  and numpy makes it bitwise identical to `inv`. It bounds `A X - I` and says nothing about
  `X A - I`: measured 2.7e-13 against 1.2e-4 at the last accepted degree.

**The downstream grep is DONE** (2026-08-21) and it changes the plan for `bezier`. Nothing was
broken by `PointsLattice` or by the package split. But the consumer imports
`pantr.bezier._root_finding_core._de_casteljau_eval_scalar`, `pantr.bezier._bezier.Bezier` and
`pantr.grid._bvh_core._BVH_STACK_DEPTH`, all present on `main`. **`bezier` is therefore the first
port that touches symbols it uses**, and a Numba kernel is exactly what a port reorganises, so
decide `_de_casteljau_eval_scalar`'s fate deliberately. Full detail in
[[downstream-consumer-surface]].

**The float32 width question is answered, and the answer is that widening is free.** The only
change_basis function the consumer uses is the one builder that runs no solve, and it never asks
for float32. So the Gram products and the LU can move to double whenever we want, which also
raises the float32 degree ceilings (8 to 14, 12 to 26). It is now a decision about pantr alone.

**Open and not ours**: #336 (THB extraction performance, outside Stage 1). **Open and ours**:
#341 (tanh-sinh endpoint distance, an enhancement).

**One question left for Pablo, deliberately unanswered in the PR**: `pyproject.toml` declares
`license = "MIT"` while the wheel now contains statically linked MPL-2.0 code. The licence text
ships, which is required either way; whether the declaration should change is not a port's call.

See [[decisions-pointer]], [[dispatching-agents]], [[performance-followup]] and
[[expired-guard-rationales]].
