---
name: merge-authority
description: Standing authority to merge a port PR without asking, with three conditions; its stated terminus dissolved when block C was ruled not-ported and it needs re-granting
metadata:
  type: feedback
---

**Granted 2026-08-24.** Supersedes the older rule that merging always needs a separate
ask. I may merge a port PR into `proto/cpp` **without asking** when all three hold:

1. the review left no `C`ritical and no `I`mportant finding unresolved;
2. CI is green on both `proto/cpp` jobs;
3. the parity claim is either an equality or a derived bound, and it is written down.

**If any one fails, stop and report.** The grant is not a licence to merge and then
explain.

**Why:** Pablo does not want to guide each implementation step. A per-PR merge ask is a
hard stop that adds nothing when the three conditions already encode what he would check.
Revert is cheap on a prototype branch, which is what makes the trade sane.

**How to apply:** the horizon was **bezier only**, blocks B and C, and it does **not** extend
to `grid`, deliberately: the BVH's inclusive-face tie contract produces a discrete verdict that
no tolerance bounds, and that decision goes to Pablo before it is frozen into a second
implementation.

**The horizon dissolved rather than expired, 2026-08-24.** Block C was ruled **not ported**
(PR #355, `design/bezier_interpolation_port.md`), so the event the grant named as its terminus
never happens. #355 itself was left **unmerged** and put to Pablo, on two grounds: it is not a
port PR and carries no parity claim, so condition 3 has nothing to bite on; and this session's
own rules forbid dispatching agents unless asked, so condition 1's review could not be run the
way the grant assumes. **Re-ask before merging anything on `proto/cpp`.** See [[active-task]]
and [[decisions-pointer]].

Two related standing permissions granted at the same time:

- **Agents may be dispatched for review only**, at the depth the toolkit's own rule picks:
  light for a bounded change with no new numerical content, deep for new public surface or
  new mathematics. Never for implementing. This resolves a real contradiction, since the
  standing rule since #347 is to review before merging a port while the session rule
  forbade dispatching agents at all. **Note it does not always survive:** a later session
  carried an explicit "do not call the Agent tool unless the user requested it", which wins,
  and then condition 1 above cannot be satisfied. Say so rather than merging unreviewed.
- **Tickets are still approved before creation**, but **batched at each PR boundary**
  rather than interrupting. Something that *blocks* the port goes up immediately.

**Report at each merge**, not per commit: what was measured and what was found.
