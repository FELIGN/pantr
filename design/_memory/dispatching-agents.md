---
name: dispatching-agents
description: A brief is not an isolation mechanism; agents that mutate need a worktree each
metadata:
  type: feedback
---

**Agents that build or mutate need one worktree each. An instruction not to mutate is not a
mechanism.**

**Why.** A twelve-agent deep review of pantr PR #347 was dispatched against a single worktree,
with several authorised to mutate transiently and revert. They collided: one agent found
another's injected mutation already in the tree before it started reading, a second had
`change_basis.hpp` swapped to `FullPivLU` under it mid-session, and a third reverted a
*committed* fix. Every tracked file was restored and verified against `HEAD`, so nothing reached
the branch, but any measurement taken during those windows was suspect and the decisive numbers
had to be re-run on a clean tree.

Relaunching with "**DO NOT MODIFY ANY TRACKED FILE**" in the brief did not hold either: one
agent modified one anyway. That is the point. A brief is a request; a separate worktree is a
guarantee.

**How to apply.** Give `isolation: "worktree"` to any agent that will build, mutate, or measure.
Read-only lenses can share. And expect a session limit to kill several at once: seven of the
twelve died together, and the three that were not relaunched left their ground unreviewed, which
has to be stated in the report rather than quietly omitted.

See [[active-task]].
