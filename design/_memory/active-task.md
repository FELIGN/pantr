---
name: active-task
description: bezier is closed; block C was ruled NOT ported and the reason is measured, PR #355 open. The merge authority's stated terminus no longer exists.
metadata:
  type: project
---

**Updated 2026-08-24.** `bezier` is **closed**, but not the way it was planned. Blocks A (#348),
the FMA bound (#349) and B (#353) are merged ports. **Block C, interpolation, was ruled NOT
ported**, and PR **#355** carries that ruling plus the thing that turned out to be worth doing.

**The reasoning is in `design/bezier_interpolation_port.md`** and reproducible from
`scripts/measure_bezier_interpolation_port.py`. Short form: the blocker everyone expected
(Eigen and LAPACK disagreeing about the truncation rank, a discrete verdict no tolerance bounds)
**does not fire** on the real matrices, twelve cases including the two tightest. But nothing
positive replaced it. Eigen's SVD is not faster; the downstream consumer imports neither entry
point and never passes float32, which is the only regime the truncation fires in; the scattered
site is shared with out-of-scope `bspline`; and once the factorization is memoized the numerical
work C++ would replace is **about 7% of a warm call**, the rest being Python including a callable
the API is handed.

**What shipped instead:** `_build_bernstein_pinv`, the one site both public entry points reach,
is memoized on the nodes' bytes. Bitwise identical, three to eight times faster with threads
pinned.

**A ruling is owed.** The merge authority granted 2026-08-24 says it "expires when block C
lands". Block C is not landing as a port, so its terminus no longer exists. #355 is also not a
port PR and carries no parity claim, so its conditions do not really apply either. It was left
unmerged deliberately. See [[merge-authority]].

**Next after this is `grid`**, which needs a fresh ruling anyway: the BVH's inclusive-face tie
contract produces a discrete verdict no tolerance bounds, the same family as Rule 11.

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

**Two facts to carry into grid.** `scripts/measure_root_finding_widths.py` is the shape a port's
specification should take: rival models per site, measured against the kernel, with a
discrimination count so a match cannot come from a check that could not fail. And
`design/backend_parity.md` has **eleven** rules; Rule 11 is the first about a discrete verdict
rather than a displacement.

**Three tickets filed 2026-08-24**, all pre-existing and none fixed: #351, #352, #354. All three
are tolerance-derivation defects in the same kernels and one coherent rework may close all three.
The C++ reproduces #351 and #352 deliberately, asserted by name.

**One review finding downgraded rather than fixed:** `ParityClaim` carries optional fields
meaningful for one variant. Pre-existing for four; block B added two more.

**One piece of doc rot left alone deliberately:** `_bezier_interpolate.py`'s module docstring
says its helpers serve "the resultant pipeline". No such pipeline exists in the repo, and
`_bernstein_interpolate` has no caller there outside tests. Either stale or naming an out-of-tree
consumer; not establishable from this machine.
