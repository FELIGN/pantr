---
name: decisions-pointer
description: The C++ port's design decisions live in design/*.md in the repository, not in memory; two modules are ported and nothing is in flight
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

**Status, 2026-08-20.** Both PRs are **merged** into `proto/cpp`: the build skeleton (#335)
and the `quad` module (#337). Nothing is in flight. See [[active-task]].

**Three notes were added by PR 2 and they govern everything ported after it.**
`design/backend_parity.md` is the one to read first: it states what a bound may claim and in
which frame, and each of its **six** rules carries the failure that produced it rather than
the principle it illustrates. Rule 6 is the newest and the least obvious: a sensitivity
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
