---
name: reviewing-my-own-measurements
description: In numerical work my prose fails far more often than my code, because nothing checks prose
metadata:
  node_type: memory
  type: feedback
  modified: 2026-08-20T23:30:00.000Z
---

A six-lens deep review of the quad port found six critical defects. **Not one was a wrong
number in the shipped code.** Every one was a claim *about* a measurement: units swapped
(`eps` quoted as `u`, halving every stated margin), a test range fitted to one machine's
libm, a figure correct for one sweep cited in a file whose constants describe another, a
sensitivity formula evaluated in a coordinate its derivation does not cover.

**Why:** the suite checks code on every commit and checks prose never. In a numerics PR the
prose *is* the deliverable, because the bounds, their derivations and the contract other
work inherits live in docstrings and design notes.

**How to apply:**
- **Re-measure a number before writing it into a permanent artifact**, and say which sweep
  produced it. "Measured over a twelve-size probe, not this file's own constants" is the
  sentence that would have prevented four of the six.
- **A sampled sweep can miss every failing case.** Sweep exhaustively when the cost allows;
  a recorded slack of 0.43 turned out to be 0.947 exhaustively.
- **Never compare two arrays by their maxima.** A max ratio of 0.9999991 hid an elementwise
  factor of 1.9e12.
- **A transliteration contract is a claim about the binary.** `nm -D` and `objdump`, not the
  source.
- **State absolute vs relative explicitly.** I conflated them five times in one cycle.
- Prefer a guard that makes the mistake unrepresentable over a comment warning against it:
  refusing an array with no negative entry beats a docstring saying "pass the unmapped nodes".

See [[active-task]].

## The same pattern held over the follow-up work, 2026-08-21

Six more inherited claims did not reproduce, two of them written by me the day before. Each was
caught by running rather than by reading, and each check cost seconds.

- The tanh-sinh accuracy floor was **documented behaviour**, not a defect, and its documented
  model predicts the measured error to three digits.
- A ticket I filed named a root cause (storage narrower than the accumulator) that is **not
  established**; what is measured is only that float64 is bitwise unaffected by the JIT switch
  and float32 is not.
- A ticket's own specification was **wrong about its own fix**: copy-and-freeze closes two of
  three holes, because numpy's writeable flag governs the data and not `shape`.
- Two "separate defects" were **one** cause; a fix said to be waiting had **already shipped**;
  a record's stated justification was **false** rather than unexercised.

**And the strongest tool this cycle was repeating a run.** Two failures looked identical, both
saying the result depends on how the code is run. Repeating separated them: one gave a different
answer on an unchanged tree (host variation, non-deterministic) and the other gave the same
answer three times (configuration variation). The remedies are different, and picking the wrong
one would have introduced a host concept where no host was involved.

Corollary worth keeping: **a guard that fires intermittently is worse than no guard**, because
it teaches everyone to discount reds.

