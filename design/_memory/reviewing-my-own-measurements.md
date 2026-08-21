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


## The change_basis cycle, 2026-08-21: it recurred, and a deep review is what caught it

**Nineteen findings, and almost none was a computation error.** The kernels were bit-exact
against their oracles, the refactor was behaviour-identical builder by builder, the degree-domain
derivation survived every number checked. What failed was the prose, again, and this note existed
before the session started.

- **A fabricated citation.** Three places cited a parity test file that had never existed on any
  branch, one of them inside a string printed in every failure message. Worse than a dead link:
  it stood in for a test that should have been written, and the gap it hid was real.
- **A bound contradicting a PROVED result in the file I was editing.** I derived
  `2^degree * eps` for the Legendre recurrence from scratch while `design/backend_parity.md`
  Rule 4 records `Theta(n^2)` as proved, and I was editing that same file to add Rule 8.
- **The wrong theorem cited**, Higham 9.4 while quoting 9.5's shape minus its `n^2`.
- **A rule I wrote claiming a property of the mathematics** when it was a limitation of my
  constant: "no correct digits above the accuracy domain" is false, 3.2 digits survive.
- **Three false completeness claims** and **two numbers from one run each**, one refuted by the
  next run.

**What generalises.** Everything above passed ruff, mypy, 37 local checks and the full suite under
both backends. **None of that machinery reads prose.** The only things that caught them were
adversarial agents told to attack a specific claim, and in three cases the fix was a measurement
I could have run in a minute at the time of writing.

**So: when writing a bound, a citation, or a completeness claim, run the check that would refute
it before writing it down**, not after. And when amending a document, read what it already proves;
twice this cycle the answer was four directories away or in the same file.

See [[dispatching-agents]] for how to run those adversarial agents without them colliding.
