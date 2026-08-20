---
name: active-task
description: The pantr C++ port after both PRs merged, the four follow-ups now filed, and the one piece of work still waiting
metadata:
  node_type: memory
  type: project
  modified: 2026-08-20T23:55:00.000Z
---

**PR 2 of the C++ port is MERGED**: `pantr.quad`'s rule generation, PR #337 into `proto/cpp`,
36 commits, CI green. PR 1 was #335. **No work is in flight.** `proto/cpp` is at `8e8f37c`,
the merge commit; a local checkout can sit well behind it, so fetch before reading its tip.

Four measured parity results, since they set expectations for the ports that follow:
Gauss-Legendre and trapezoidal are **bit-identical** in both storage formats, Lambert W is
within one ulp, modified Chebyshev is exact in float64 and one unit of roundoff in float32,
and tanh-sinh is bounded with its node **count** identical. Exactness is a property of the
BUILD, not of the kernels: `-march=native` moves 1994 of 2143 values and `-ffp-contract=off`
restores them.

**The four follow-ups are FILED**, each root-caused with a verified reproduction, none a
regression from #337: **#338** `PointsLattice` neither copies nor freezes its arrays while
`QuadratureRule` in the same package does both; **#339** the basis parity tests assume a
compiled kernel and fail under `NUMBA_DISABLE_JIT=1`, which is what breaks `make coverage`
with the extension present; **#340** `ci_local.sh`'s split-mode probe shares the persistent
build dir, one cause behind both the wrong-nanobind resolution and the non-idempotency;
**#341** an enhancement, not a bug, to keep tanh-sinh's endpoint distance.

Two corrections to what was believed before they were filed, both found by re-checking rather
than by trusting the note. The tanh-sinh plateau is **not a defect**: the docstring has
documented and quantified it since #291, and its model predicts the measured error to three
digits, so it was filed as an enhancement. And the proof that #339 predates the port does
**not** work by running the old test file unchanged, which fails on API drift instead; it
needs the mechanical `Backend.NUMBA` to `Backend.PYTHON` rename applied first.

**One piece of work waiting**: the `QuadKernels` redesign, where the record's stated
justification is exercised by no call site and the catalogue's signature convention is
unwritten and inconsistent between the two ports that exist.

**The backend cache-keying fix is NOT waiting, contrary to what was planned.** It was to be
its own PR and it shipped inside #337 instead, as commit `b5c562b`. The merged version is
better than the standalone branch was: the standalone one named the two backends absolutely
in its threading test, which fails once the suite runs under `PANTR_BACKEND=cpp`, and the
merged one names them relative to the ambient backend and skips when only one exists. Do not
go looking for that PR. Verified `14 passed` under each backend.

A session-scoped handoff brief carried every directive verbatim, the twenty-nine decisions in
force, and what the cycle taught. That path does not outlive its session. **The durable
equivalents are in the repository**: `design/backend_parity.md`, `design/build_findings.md`
and `design/quadrature_algorithms.md`.

See [[decisions-pointer]], [[performance-followup]] and [[reviewing-my-own-measurements]].
