---
name: reviewing-my-own-measurements
description: In numerical work my prose fails far more often than my code, because nothing checks prose; six cycles, and the newest failure was a measurement rather than a claim
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


## The bezier cycle, 2026-08-22: the prose failed differently, and the tree already knew

Fourth cycle, nineteen findings again, and the kernels right again. **What changed is the
shape of the failure**: not fabricated citations this time but *justifications*, and three of
the four I wrote for my own decisions were refuted. Two of them by evidence already sitting in
this repository.

- **I justified a design decision with a mechanism that does not exist.** The record-shaped
  catalogue entry was defended by "a `ContextVar` scoped to a thread or task lets two
  resolutions in one call diverge". That inverts it: each thread gets its own context and a
  task runs in a copy, so the property I named as the danger is what makes the failure
  impossible. Measured, 0 of 2000 both ways. The code was harmless; the sentence was in the
  design note that governs the next ports.
- **I justified *not* doing work with a problem the tree had already solved.** I skipped
  deriving a parity bound for a fusing build because "a bound for a branch no host can execute
  would ship untested". `tests/parity/test_quad_gauss_legendre.py` derives one **and probes it
  on this same non-fusing host**, three lines of it. I had read that file during the quad port.
- **A tolerance whose mechanism I asserted without checking.** I claimed Sterbenz exactness of
  `c_1 - c_0`; the kernel never subtracts two control points. It was vacuous on its own data
  (error exactly zero) and false off it by 1.7e5.
- **Figures with no artifact.** Every headline count lived in a scratch directory. Real, three
  later reproduced exactly by a reviewer, and none re-derivable by anyone else. `bernstein.hpp`
  had been pointing at a committed test for its own claim since the previous port.

**The new rule, and it is cheap: before writing why a decision is right, grep the tree for a
sibling that faced it.** Two of the four would have died on one `grep`. This is the same lesson
as "read what the document already proves", widened from documents to code.

**Second new rule: a measurement quoted in a permanent artifact needs a committed script.** Not
because the number is likely wrong, but because a number nobody can re-derive is a number
nobody can refute, which is the wrong shape for a design note.

And one that goes the other way, worth recording so the tally stays honest: **a lens overstated
a finding and I caught it by measuring.** An agent reported that `PANTR_BACKEND` changes what
the library accepts. The public API refuses the mismatch identically on both backends; the
asymmetry exists only for a direct Layer 3 call, where nothing validates by design. Do not
relay a finding you have not reproduced, in either direction.


## The FMA-bound cycle, 2026-08-24: the prose and the code desynchronised inside one session

Fifth cycle, and the shape is new. Not a fabricated citation, not a justification refuted by
the tree, but **a justification that was true when written and false four hours later**, because
the code moved under it and nothing checks that a docstring still describes its function.

I wrote five `fused_why` strings saying "max|c| bounds the whole triangle, and it is the right
magnitude". Then a vacuity guard fired, I replaced the amplification with the absolute-value
companion, which is strictly smaller and can be smaller by twelve decades, and I never reopened
the strings. Those strings are **printed verbatim in every failure message**, so the artifact
shipped a derivation for code it no longer described. A tolerance audit found it; nothing in
ruff, mypy, 37 discipline checks or 548 passing tests could.

**The rule this adds: when you change what a constant or an array *is*, grep for the prose that
names it, in the same edit.** Not later, not at review time. The window between the change and
the review is exactly where this defect lives, and it is invisible from inside the change.

**Two more from the same cycle, both mine, both measured rather than argued:**

- **A bound can be correct and vacuous per element while a guard sized on the array's maximum
  sees nothing.** A flat `p!/(p-k)! 2^k max|c|` made 45 of 280 non-zero float32 values carry a
  tolerance at least as large as their own magnitude. The guard compares the largest tolerance
  against the largest value, and a flat amplification is sized for exactly that element. Rule 6's
  own disease, committed inside a rule written after it.
- **A measured figure identical across two dtypes is structural, not rounding.** The corrected
  amplification was exceeded by 4.32x at float64 and 4.318x at float32; that agreement is what
  said "your derivation is missing a term" rather than "your test data is unlucky".

And one about my own measurement, caught before it reached Pablo: my first vacuity count was
165 of 400, inflated by rows where the reference is exactly zero and both backends agree
trivially. The honest count is 45 of 280. **Excluding the degenerate rows is part of the
measurement, not a refinement of it.**


## The interpolation cycle, 2026-08-24: the measurement itself was wrong, not the claim about it

Sixth cycle, and it inverts the pattern. The previous five were prose failing while the numbers
held. This time **the number was wrong at the source**, and the prose faithfully reported it.

I measured a memoization at **1.09x to 2.88x** and reported it to Pablo as an input to a
decision. Both halves of that were wrong:

- It was taken **unpinned** on a 20-thread box, where a small LAPACK SVD's timing varies by
  nearly an order of magnitude between batches. The 2.88x was thread noise. Pinned, the same
  comparison gives 0.99x to 1.10x. See [[build-machine]].
- **The cache was not on the call path at all.** I had memoized `_bernstein_vandermonde_svd`
  believing the public entry points reached it. They reach `_build_bernstein_pinv`. Neither
  `grep` nor the passing tests nor the plausible-looking speedup revealed it.

**What caught it: `cache_info()`.** Zero hits, zero misses, after thousands of calls. One line.

**How to apply, and these are cheap:**

- **Before timing anything that calls BLAS or LAPACK, pin the threads.** A speedup that does not
  survive a thread pin is not a speedup. Bake the refusal into the script rather than remembering.
- **A cache gets a hit counter before it gets a benchmark.** `cache_info()`, or an equivalent
  assertion in a test, proves the thing you are measuring is the thing that runs. The benchmark
  cannot: an improvement and a coincidence look identical.
- **Follow the call graph before optimising, do not infer it from names.** `_bernstein_interpolate`
  and `_bernstein_vandermonde_svd` read like the main path and are reached by nothing.
- **When the fix lands somewhere else, delete the first attempt rather than keeping it "since it
  is free".** I nearly shipped a second cache on an uncalled function out of sunk cost. It would
  have needed its own size policy beside the real one, which is how a module accumulates.

**And one that went right, worth recording for the tally.** I predicted Eigen's `BDCSVD` would
be slower than LAPACK's `gesdd` and was ready to build an argument on it. I benchmarked instead
and it is competitive, sometimes faster. The prediction would have been a confident, plausible,
entirely wrong premise under a design decision. **Benchmark the premise you are about to lean
on, especially when it feels obvious.**
