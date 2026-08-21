---
name: downstream-check-owed
description: One outstanding verification against the unnameable downstream consumer, owed since PointsLattice was frozen
metadata:
  type: project
---

**`PointsLattice` became immutable** (issue #338, PR #344, merged into `proto/cpp` as
`782e46d`). Its coordinate arrays are copied at construction and `pts_per_dir` hands out
read-only views, so `lattice.pts_per_dir[0][:] = ...` now raises `ValueError`.

**The check nobody has run:** whether the downstream consumer writes through that property.
`PointsLattice` ships from `main` and `pts_per_dir` is public, so this is not limited to the
private symbols the usual warning is about, and pantr's own CI cannot see the breakage. It is a
one-minute grep over that repository for `pts_per_dir` and `PointsLattice`. Pablo has to run
it; it cannot be run from here.

Contrast with the catalogue work (#343), which was safe precisely because
`pantr.quad._quad_backend` and `pantr.basis._basis_backend` **do not exist on `main`**: the whole
dual-backend layer is prototype-only, so nothing outside the branch can import it. That test,
"is this symbol on `main`?", is the cheap way to tell a free reshape from a risky one.

See [[active-task]] and [[naming-and-licensing]].
