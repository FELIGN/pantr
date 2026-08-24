---
name: decisions-pointer
description: The C++ port's design decisions live in design/*.md; six PRs merged, nothing in flight, and backend_parity.md now has eleven rules
metadata:
  type: project
---

pantr is being ported from Python (with Numba kernels) to a C++ core with nanobind
bindings. **The design decisions live in `design/*.md` in this repository.** Read them
before proposing anything about the port; they are long, cross-referenced, and each one
carries an epistemic-status section separating what was verified from what was derived.

**They are sealed against 0.7.0 but were written without compiling a line.** Four were
amended on 2026-08-19 when the compiler contradicted them — epistemic status and open
questions only, never a decision. If the code refutes a note again, say so and let Pablo
rule; do not silently work around it and do not silently obey it.

Two decisions that govern everything else:

- **Dual backend.** The existing Python implementation stays as the **parity oracle**. The
  C++ backend is validated against it, module by module, with dispatch at the Python level
  via `PANTR_BACKEND`, and an explicit request never falls back. Note that the oracle is
  **not always Numba**: `geometry`, `transform`, `quad`, `change_basis` and `tolerance` are
  pure NumPy, while `basis` (18 kernels), `grid` (15) and `bezier` (41) are Numba. That
  distinction changes both the parity evidence required and what a speed figure means; see
  `design/build_findings.md`, the note to read after this one.
- **Stage 1 scope** is what a downstream consumer actually imports: `geometry`, `grid`,
  `quad`, `transform`, `change_basis`, `bezier`. Roughly 8-10k lines, not 47k. `bspline`,
  THB, extraction, `cad`, `multipatch` and `mpi` stay in Python for now.

**Status, 2026-08-24.** Six PRs are **merged** into `proto/cpp`: the build skeleton (#335),
`quad` (#337), `change_basis` (#347), `bezier`'s arithmetic block (#348), its FMA parity bound
(#349) and its root-finding block (#353). Nothing is in flight. One `bezier` PR remains,
interpolation. See [[active-task]].

**Eigen is a dependency of the shipped extension** as of #347, and that is the architectural
decision this cycle added. It was test-only before, fenced off from `pantr::core` so that taking
it would have to be deliberate; `change_basis` needs a dense solve and `bezier` needs a truncated
SVD, both in Stage 1. Measured: a cold editable install goes 7.56 s to 12.19 s and the build tree
6.4 MB to 50 MB, with `GIT_SHALLOW TRUE` doing most of the saving; the incremental rebuild is
unchanged. Eigen's licence files are vendored in `licenses/` because the wheel now contains
compiled MPL-2.0 code -- **but `pyproject.toml` still declares MIT, and whether that should
change is an open question for Pablo, not a port's call.**
`design/large_data_fitting.md` had claimed Eigen was already required and that was false;
corrected in place, along with the same stale premise in
`design/adaptive_thb_approximation.md`.

**`design/backend_parity.md` now has ELEVEN rules.** Rule 11, added 2026-08-24 by the
root-finding port, is the first about a **discrete verdict** rather than a displacement: an exact
tie does not survive contraction, and where a tie-break decides a count no tolerance covers it. It
also carries the two predicted bounds that were refuted before the certified one replaced them,
and the operational lesson that a conditional claim's unevaluated branch is where the bug is.
Rule 9 gained the mechanism under its widths: numba's `float()` does not widen a `float32`.

**Before it,** ten rules. Rule 10, added 2026-08-24, states the
contraction bound: one rounding removed per fused site, one budget for all eight Bezier kernels,
and only the amplification differing. It closed the gap Rule 9's section had declared, and that
section is corrected in place rather than deleted, because the reason it gave for deferring was
refuted rather than superseded. Rule 10 also restates the harness's factor of two as an
acknowledged safety margin: its stated justification is about two one-sided forward errors, and
a contraction budget is already a backend-to-backend difference.

**Before it,** Rule 9, added by the bezier arithmetic port,
is the first that is about reproducing an oracle *exactly* rather than about bounding a
difference, because that port is the first whose every kernel can be bit-exact: an oracle's
accumulation width is a per-kernel fact, not a module convention, and getting it wrong is
invisible at float64. Before it, and #347's own deep review corrected the
newest one. Rule 8: a parity claim is only defined where the bound can still say something, so
the parity domain is the accuracy domain and the excluded degrees must be named. Its first
version also claimed those degrees have no correct digits, which is false, and one of its two
margin figures was invented.

**Three notes were added by PR 2 and they govern everything ported after it.**
`design/backend_parity.md` is the one to read first: it states what a bound may claim and in
which frame, and each of its **eight** rules carries the failure that produced it rather than
the principle it illustrates. Rule 8, added 2026-08-21 by the change_basis port, is the one to read before writing any
parity claim over a solve: a parity claim is only defined where the quantity has digits, so the
parity domain is the *accuracy* domain and not the *solvability* domain, and the excluded
degrees have to be named rather than quietly skipped. Rule 7 is the one that changes how CI is read: a bound is a property of
the code, but whether it is *approached* is a property of the host, so a liveness guard is
enforced only where its numbers were measured. Rule 6 is the least obvious: a sensitivity
derived at a distinguished point set is **not** that sensitivity at the image of that set
under a coordinate map, and comparing the two arrays by their maxima cannot see the
difference. Its last section now answers whether the harness generalized (yes in vocabulary,
no in meaning) instead of deferring it. `design/build_findings.md`, written by Pablo, is
where both PRs' build findings live consolidated. `design/cross_backend_types.md` settles that **no type crosses the
boundary** (types are Python-owned, only arrays and scalars cross), that `dtype` is an output
format rather than a computation precision, and, since #343, **the two catalogue rules**: a
record only when the consumer needs more than one kernel at once, and a catalogue entry that
mirrors its module's public surface. `design/quadrature_algorithms.md` records
why Gauss-Legendre is Newton rather than Golub-Welsch, and why Eigen was declined there.

**PR 2 refuted its own brief's premise, and that is the pattern to expect.** The brief assumed
numpy computes Gauss nodes by a nonsymmetric companion matrix; reading the source showed it is
already symmetric Golub-Welsch plus a Newton polish. Measurement then chose a different
algorithm than the one specified. The rule above held: the note was not silently worked
around, Pablo ruled, and the decision is recorded.

The port is still a **prototype** on a branch, not a landing on `main`. Two modules are
ported: `pantr.basis`'s cardinal B-spline (#335) and `pantr.quad`'s rule generation (#337).
Within `quad` only four of the seven public rules dispatch, and `pantr._backend`'s Scope
section names the three that never do, because identical output from one of those is the
switch being a no-op rather than a parity success.

One working note is deliberately **not** in this repository (it documents a sibling
project's mechanism in detail and quotes its code); it stays on the laptop. See
[[naming-and-licensing]].
