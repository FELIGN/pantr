---
name: active-task
description: The pantr C++ port: bezier's arithmetic and its FMA parity bound are merged; two bezier blocks remain
metadata:
  node_type: memory
  type: project
  modified: 2026-08-24T00:00:00.000Z
---

**The FMA parity bound is MERGED** (PR #349, five commits, rebase-merged; merged tree
verified byte-identical to the tested one). It closes the one gap `bezier`'s arithmetic port
declared. Nothing is in flight, no worktree, `proto/cpp` clean.
**Four of Stage 1's six modules are ported**: `basis`, `quad`, `change_basis`, and block A of
`bezier`.

**What #349 settled, because the next port inherits it.** Rule 10 of
`design/backend_parity.md`: contraction removes one rounding per fused site, the budget is
`Roundings(stages, 3, 2)` for every kernel, and only the amplification differs. Claims are now
selected at run time rather than skipped, so a build raising the ISA baseline no longer turns
519 assertions into skips. `scripts/measure_bezier_fma_bound.py` reproduces the fourteen fused
sites by disassembly, the slack per kernel, and the A2.3 majorant check.

**The amplification is the part that took three attempts, and the pattern generalises.**
Where a kernel's stages are convex combinations, the tight amplification is the *same operation
run on `|c|`*, elementwise. Where they are not, it is the absolute row action of the operator.
Where the recursion itself has signs, as A2.3 does, neither works and you need the **majorant of
the recursion**: run it with every coefficient replaced by its modulus. `max|c|` is correct for
all of them and useless for most.

**`bezier` has two blocks left.**

- **B, next.** Root finding: `_root_finding_core` (7 kernels), `_clipping_core` (2),
  `_yuksel_core` (4), `_batch_core` (3), `_find_roots`, `_root_finding`. ~1580 lines, 16 kernels,
  all iterative, so convergence tolerances and **no bit-exactness**. It holds
  `_de_casteljau_eval_scalar`, which the downstream consumer imports; decide its fate first.
- **C, after that.** `_bezier_interpolate`, 756 lines, no kernels, but an SVD with **rank
  truncation** whose threshold is discontinuous. The matrices are deterministic per degree, so
  it is checkable: verify no `sigma` sits within a few eps of the threshold. Touches
  `src/pantr/_interpolation_utils.py`, shared with `bspline` and `mpi`.

**Ground the reviews never covered**, and it is now small: whether the harness's multiplicative
`gamma_m` is merely conservative for an additive accumulation, and whether
`sum_j |B^(k)_j(s)|` attains its bound pointwise in `s` or only at the endpoints. Neither is
load-bearing; the second feeds no amplification any more. Both are named in Rule 10 and in #349.
The `mathematician` lens died on a session limit twice this cycle without returning anything.

**Two things found and left alone, both pre-existing on `proto/cpp`:**
- `_bezier_degree.py:35` imports `_tabulate_Bernstein_basis_1D_serial_core` straight from
  `basis._basis_core`, bypassing that catalogue.
- `_AUTO_REDUCTION_TOL_FACTOR` is dilation-invariant but not translation-invariant.

**Open and not ours**: #336. **Open and ours**: #341.

See [[decisions-pointer]], [[proto-cpp-pr-mechanics]], [[dispatching-agents]],
[[downstream-consumer-surface]], [[reviewing-my-own-measurements]] and [[performance-followup]].
