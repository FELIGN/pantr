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
