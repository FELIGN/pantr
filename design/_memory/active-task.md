---
name: active-task
description: The pantr C++ port: bezier's arithmetic merged, four of six Stage 1 modules done, and the two bezier blocks still to go
metadata:
  node_type: memory
  type: project
  modified: 2026-08-22T00:00:00.000Z
---

**`bezier`'s arithmetic block is MERGED** into `proto/cpp` (PR #348, 16 commits, rebase-merged;
merged tree verified byte-identical to the tested one). Nothing is in flight, no worktree.
**Four of Stage 1's six modules are ported**: `basis`, `quad`, `change_basis`, and now block A of
`bezier`.

**`bezier` is being ported in three PRs, and two remain.** The split was by risk, not by size:

- **A, merged.** The arithmetic: evaluate, derivatives, elevate, slice, split, restrict, the
  Bernstein product, plus the dense reduction-operator apply. Nothing iterative, no solve.
- **B, next.** Root finding: `_root_finding_core` (7 kernels), `_clipping_core` (2),
  `_yuksel_core` (4), `_batch_core` (3), `_find_roots`, `_root_finding`. ~1580 lines, 16 kernels,
  all iterative, so convergence tolerances and **no bit-exactness**. It also holds
  `_de_casteljau_eval_scalar`, which the downstream consumer imports; decide its fate first.
- **C, after that.** `_bezier_interpolate`, 756 lines, no kernels, but an SVD with **rank
  truncation** whose threshold is discontinuous: `sigma >= tol * sigma[0]` decides the rank, so
  LAPACK and Eigen disagreeing by a few eps on a `sigma` near the filo flips it and the result
  differs by O(1), not within a bound. The matrices are deterministic per degree, so it is
  checkable: verify no `sigma` sits within a few eps of the threshold. It also touches
  `src/pantr/_interpolation_utils.py`, shared with `bspline` and `mpi`.

**Block A is the first port whose whole surface is bit-exact.** `quad` could not (different
algorithms), `change_basis` could not (a solve). The cost was reproducing three different
accumulation widths the oracle uses without announcing them, which is now Rule 9 in
`design/backend_parity.md`.

**Read `reviews/348-2026-08-21.md` before the next block.** Nineteen findings, and see
[[reviewing-my-own-measurements]] for the shape: my justifications failed, twice against evidence
already in this repository.

**The one thing deliberately not fixed, and it is a real gap:** no parity bound is derived for a
build that can fuse a multiply-add. `tests/parity/test_bezier_arithmetic.py` skips all its
assertions there, so on the first `-march=x86-64-v3` build the module loses 519 of 519 and the
suite stays green. The reason originally given was refuted, `test_quad_gauss_legendre.py` already
derives and probes such a bound on this non-fusing host. **It is a prerequisite for the ISA
ladder of `design/simd.md`**, and the tolerance audit left a derivation sketch per kernel in the
review artifact.

**Ground the deep review never covered**, because a session limit killed three agents: the four
strings in `tests/parity/test_bezier_arithmetic.py` that justify the bitwise equalities have not
been adversarially audited, and the accumulation widths were never verified against numba's typed
IR. Both are worth doing at the start of block B rather than never.

**Two things found and left alone, both pre-existing on `proto/cpp`:**
- `_bezier_degree.py:35` imports `_tabulate_Bernstein_basis_1D_serial_core` straight from
  `basis._basis_core`, bypassing that catalogue, so under `PANTR_BACKEND=cpp` the rational degree
  search's collocation matrix mixes C++ nodes with a Numba tabulation. Now covered by eight
  rational parity cases; measured, it moves no verdict.
- `_AUTO_REDUCTION_TOL_FACTOR` is dilation-invariant but not translation-invariant: the same
  curve reduces to degree 3 at the origin and degree 2 at offset 1e2.

**Open and not ours**: #336 (THB extraction performance). **Open and ours**: #341 (tanh-sinh
endpoint distance).

See [[decisions-pointer]], [[proto-cpp-pr-mechanics]], [[dispatching-agents]],
[[downstream-consumer-surface]] and [[performance-followup]].
