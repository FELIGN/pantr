---
name: decisions-pointer
description: The C++ port's design decisions live in design/*.md in the repository, not in memory
metadata:
  type: project
---

pantr is being ported from Python (with Numba kernels) to a C++ core with nanobind
bindings. **The design decisions live in `design/*.md` in this repository.** Read them
before proposing anything about the port; they are long, cross-referenced, and each one
carries an epistemic-status section separating what was verified from what was derived.

Two that govern everything else:

- **Dual backend.** The existing Python implementation stays as the **parity oracle**. The
  C++ backend is validated against it, module by module, with dispatch at the Python level.
  Note that the oracle is **not always Numba**: `geometry`, `transform`, `quad`,
  `change_basis` and `tolerance` are pure NumPy, while `basis` (18 kernels), `grid` (15) and
  `bezier` (41) are Numba. That distinction changes both the parity evidence required and
  what a speed figure means — see `design/build_findings.md`, which is the note to read after
  this one.
- **Stage 1 scope** is what a downstream consumer actually imports: `geometry`, `grid`,
  `quad`, `transform`, `change_basis`, `bezier`. Roughly 8-10k lines, not 47k. `bspline`,
  THB, extraction, `cad`, `multipatch` and `mpi` stay in Python for now.

The port is a **prototype** on a branch, not a landing on `main`. It is gated on an
in-progress bug study of the Python library: that study blocks porting modules, not the
architecture record or the build skeleton.

One working note is deliberately **not** in this repository (it documents a sibling
project's mechanism in detail and quotes its code); it stays on the laptop. See
[[naming-and-licensing]].
