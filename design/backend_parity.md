# Backend parity: what a bound may claim, and in which frame

**Status:** complete for the five ports that exist. Twelve rules, each with the failure that produced
it, and the remaining module ports inherit them. The verdict on whether the parity harness
generalized is now written, from the `quad` port's tests rather than predicted before them.
**Date:** 2026-08-20, amended the same day after a deep review closed two open proofs,
refuted a claim and tightened a bound. Amended 2026-08-21 by the change-of-basis
port: Rule 7 (a liveness guard belongs to the host it was measured on) and Rule 8 (a
parity claim needs a bound that can say something), and again by that port's own deep
review, which found Rule 8 overstating its boundary and one of its two margin figures
invented. Both are corrected in place rather than edited away. Amended again the same day by
the Bézier arithmetic port with Rule 9 (an oracle's accumulation width is a per-kernel fact),
which is the first rule here that is about reproducing an oracle exactly rather than about
bounding a difference, because that port is the first whose every kernel can be bit-exact.
Amended 2026-08-22 with Rule 10 (contraction removes one rounding per fused site), which closes
the gap Rule 9's section declared: the Bézier claims are now conditional on what the build can
do rather than skipped where it can fuse. Amended 2026-08-24 by the root-finding port with
Rule 11 (an exact tie does not survive contraction, and a tie-break is a verdict rather than
a displacement), and with a section added to Rule 9 recording that numba's `float()` does not
widen a `float32`, which is the mechanism the three earlier width surprises rest on. Amended
2026-08-26 with Rule 12 (the interpreted oracle is a different object), which enumerates the
three ways `NUMBA_DISABLE_JIT=1` changes what the oracle computes, and records that a
width-dependent regression test can be inert in that configuration without saying so.

**Validated against:** `proto/cpp` at `765d9b9`, the `quad` port on `feat/cpp-quad`, and the
root-finding port on `feat/cpp-bezier-roots`, whose fused branch was exercised against an
extension built at `-march=native` rather than reasoned about.

## The one fact everything else follows from

**The Python implementation is the oracle.** Parity says the two backends agree; it does not
say either is right. So a shared error is invisible to every parity test that will ever be
written, and each ported module owes an *independent* check as well: exact rational arithmetic,
a closed form, an analytic invariant, a convergence order. `tests/_parity_harness.py` has a
separate slot for that (`assert_accuracy`) and it is not optional.

The infrastructure PR measured what this is worth: an independent exact-rational oracle caught
a bug injected into **both** backends at once, by a factor of 5e14 over its bound. Nothing else
in the suite could have.

### What the oracle becomes when Numba is retired, decided 2026-08-27

Every rule below compares C++ against the Python implementation with its Numba kernels
compiled. The end state chosen on 2026-08-27 is a C++ core usable with no interpreter, which
retires Numba, and on that day the twelve rules would have nothing left to measure against.
The successor was decided before more porting rather than after:

**Numba goes; the pure Python/NumPy implementation stays, and it is the oracle.** It remains
*re-derivable* -- a kernel ported next year still gets an oracle, which a set of frozen
recorded outputs could not give it -- and Rule 12 has already enumerated how that interpreted
path differs from the compiled one, so the characterisation work is done rather than owed.

Two consequences, and both are obligations rather than remarks:

- **Rule 12 stops being a note about a test configuration and becomes the definition of the
  oracle.** Its three divergences are no longer things that happen under
  `NUMBA_DISABLE_JIT=1`; they are properties of the reference. Any bound stated against the
  compiled oracle has to be re-read against the interpreted one before Numba is removed, and
  Rule 12's own claim that the three are *all* of them becomes load-bearing at that moment.
- **An oracle with no users rots silently.** Once nothing but the tests calls the Python
  path, a defect in it is invisible until a parity test blames C++ for it. The independent
  check in the paragraph above is what covers this, and it stops being "not optional" as a
  matter of rigour and starts being the only thing standing between a rotted oracle and a
  false parity failure.

**Refined the same day, and the wording above was too weak.** "The Python implementation
stays" reads as *indefinitely*, and that is not what was decided: the dual backend exists so
that the Python side can serve as an oracle **while the port runs**, and the end state is C++
alone. So the oracle stays for the duration of the port and goes with it, rather than becoming
a permanent second implementation. Nothing above changes operationally -- a kernel ported at
any point during the port still has an oracle, because the oracle outlives every kernel that
needs one -- but the two readings differ about what the repository looks like afterwards, and
only one of them is right.

**Epistemic status.** The choice is ruled, not derived. That Rule 12's three divergences are
exhaustive is *claimed* in Rule 12 and was not re-checked here; two of the three were found by
a review rather than by the rule's author, which is the reason to re-check it rather than
inherit it.

## Rule 1: state a bound in the frame the comparison happens in

**This is the rule that was violated first and would have been violated silently.**

A bound derived about a quantity in one coordinate frame is not a bound on that quantity after
an affine map, even an exact one. Transport it: multiply by the map's Jacobian, and add the
map's own rounding if the map rounds.

**The worked case, which is the one that defeated the first attempt.** Gauss-Legendre nodes are
derived on `[-1, 1]` and returned on `[0, 1]` through `x_01 = (xi + 1) / 2`. The node bound
carries a quantization term `u * |xi|`, an absolute floor that is largest at `xi = -1`.
Rewriting that term naively in the target frame, as `u * |x_01|`, evaluates it at
`x_01 = 3e-6` instead of at `|xi| = 1`, and **fails by a factor of 37.8 at n = 700**, on correct
code. Transporting it instead, as `(1/2) * u * |xi|`, holds with a worst ratio of 0.987; adding
the map's own rounding, `2 u x_01`, brings the worst ratio to 0.4935. That third term is
**required**, not decorative: `(xi + 1) * 0.5` rounds for 1761 of the nodes swept.

Note what the naive rewrite does. It does not loosen the bound, it **inverts** it: the array
spans six decades of magnitude at n = 700, and re-expressing an absolute floor with the mapped
magnitude makes it eleven orders too small at one end of that array and correct at the other.

The one place the tree already did this right is `tests/test_quad_tanh_sinh.py::_resolvable`,
which compares `min(x, 1 - x) >= eps/2` on `[0, 1]` against a generator threshold of `gap >= eps`
on `[-1, 1]`. Same threshold, transported by the same 1/2. Copy that.

**Consequence for the port, and it is a design constraint rather than a preference:** the C++
side produces `[-1, 1]` data and the map stays shared and Python-side, so it is common mode and
cancels exactly. See `design/cross_backend_types.md`.

## Rule 2: absolute where the quantity vanishes, never relative

A relative per-element bound on a quantity whose smallest entry decays with `n` is unbounded in
`n`. This is a rule about **weights**, not a fact about Gauss-Legendre, and it applies unchanged
to any rule whose weights vanish at the ends, tanh-sinh included.

Measured on Gauss-Legendre against `leggauss`: the relative disagreement grows like `n^2.7` and
reaches 2.5e6 units of roundoff at n = 400, while the absolute disagreement stays between about
25 and 280 units of roundoff to n = 1000. The smallest weight decays like `n^-1.96` and the
largest absolute error sits exactly there. A relative bound that passes at n = 400 would have to
be about 1e-9, which lets a real defect through at n = 8.

**And the converse is a trap too.** A *flat* absolute bound, taken as the array's largest
tolerance, is vacuous on the small entries. On the tanh-sinh golden file, `atol = 5e-15` is
larger than 10 of the 382 node distances and 8 of the 382 weights it exists to guard, so it
permits 100% relative error there. Demonstrated rather than argued: moving the outermost node by
**one ulp**, the smallest change that exists, doubles its distance to the endpoint, and the
absolute deviation that produces is 1.1e-16, invisible to any absolute tolerance that file could
sensibly carry. A double-exponential rule is chosen *because* the endpoint cluster is resolved,
and a flat tolerance guards everything except that.

So: **one comparison cannot guard a rule whose elements span decades.** Use two, and say which
mechanism each one covers.

## Rule 3: a bound at least as large as the values it compares is refused

`tests/_parity_harness.py` now refuses it, and the reason is worth stating because the
amplification that triggered it came from following the harness's own advice.

The harness documented a *companion recurrence* as the way to obtain an amplification: run the
kernel's own recurrence with every coefficient replaced by its absolute value. That is a bound
for a recurrence whose coefficients form a **convex combination**, which is the cardinal
B-spline it was written for, where the absolute values change nothing. It is not a bound for an
oscillatory three-term recurrence: for Legendre, `P_k` and its second solution are both bounded
on `[-1, 1]` while the absolute-value companion grows like `(1 + sqrt(2))^k`, reaching **1.7e266
at degree 700**. The tolerance that follows is 5.3e253, and `assert_parity` accepted **1.0
against -1e250** as agreement.

Nothing in the type system stops that: the amplification is finite and non-negative, which is
all the constructor can check. The guard compares against the array's **largest** magnitude
rather than each element's own, deliberately: per element it would reject a legitimate absolute
floor on an entry that is genuinely zero, which is what the underflow floor exists to serve.

**For a quantity built from a ratio of recurrence values, bound the ratio.** That is Rule 4.

## Rule 4: a node bound inherits a ratio, not a numerator

This is the subtlest of the six and the one most likely to be got wrong by someone doing the
obvious thing. Rule 6 is its companion: this one is about which quantity a bound inherits, that
one about which coordinate it is then evaluated in.

**Statement.** Let `P_n` be evaluated in binary64 by the forward three-term recurrence
`P_k = ((2k-1) x P_{k-1} - (k-1) P_{k-2}) / k`, round to nearest, no FMA, `k < 2^52`, with `P_0`
and `P_1` exact. Write `C(n, x)` for a bound on `|fl(P_n(x)) - P_n(x)| / u`. Then:

- `sup_x C(n, x)` is **`Theta(n^2)`**, attained at the endpoints, with closed form
  `C(n, ±1) = (7/4) n^2 + (9/4) n - 4 H_n`. **PROVED**, both halves.
- The ratio `C(n, x_i) / |P'_n(x_i)|` at the Gauss nodes is **bounded by 7/2 uniformly in `n`**.
  **PROVED, and non-asymptotically**: `C(n, x_i)/|P'_n(x_i)| = (w_i/2) Σ_i ≤ 7/2 - (3/2) w_i` for
  every `n ≥ 2` and every node. Measurement had put it at 3.4999940516 over a sweep to n = 2048;
  the `(3/2) w_i` deficit is exactly the `O(1/n^2)` approach that sweep saw.

**Both were SUPPORTED here until a later pass closed them, and the route is worth recording
because the obvious one does not work.** The proof goes through a closed form for the recurrence's
discrete Green's kernel, `g(n,k) = Σ_{m=k}^{n} P_{m-k} P_{n-m} / m`, an identity in `Q[x]` proved
by Cauchy product against `(1 - 2xz + z²) G² = 1`. From it `|g| ≤ H_n - H_{k-1}` on `[-1, 1]`
gives the supremum, and Christoffel-Darboux — `Σ_j (2j+1) P_j(x_i)² = 2/w_i`, exactly — gives the
ratio after AM-GM, with `7/2` forced as `sup_j (coeff of P_j²)/(2j+1)`. **`7/2` is not a fitted
constant; it is that supremum.**

The earlier attempt asked for an asymptotic argument through the Bessel envelope of `P_k`, and
that route gives a constant near **6**, not `7/2`, because the constant depends on the
*oscillation* of `P_k` rather than on its envelope. Christoffel-Darboux escapes it by summing the
squares in closed form. `|x| ≤ 1` is required: the kernel bound already fails at `x = 1 + 1e-6`.

Both blow up at the same rate in the same place, each carrying `sqrt(n) (1 - x^2)^{-3/4}`, and
that cancellation is why the node displacement is flat in `n` rather than growing.

**The trap, stated because it inverts the claim rather than loosening it.** Feed the
`x`-**uniform** constant `(7/4) n^2` into the same division and you get `2.2 n^{3/2} u` at an
interior node, which is 7e4 u at n = 1000 against a measured 0.5 u. Any node argument must carry
the `(1 - x^2)^{-3/4}` through the division; dropping it does not give a conservative answer.

**Why a measurement said `O(n)` and was wrong.** Maximizing the recurrence error over the
**Gauss nodes** fits `n^1.2` here and `n^1.6` elsewhere; maximizing over all `x` fits `n^1.9`.
The worst `x` sits at `n * theta` of about 0.02 to 0.95 where `x = cos(theta)`, which is
**inside** the outermost node at `n * theta ≈ 2.40`, so the node set never samples it. At
n = 256 the true maximum is 35 times what the nodes see. **This is the general warning:** a
constant measured on the point set the algorithm happens to use is a statistic of that set, not
a bound on the function.

**Practical consequence.** The node displacement is `3.5 u`, flat in `n`, for the `P_n` term
(`4.5 u` including the update's own rounding). **Units matter here and were once stated wrong in
this file**: `u = eps/2` throughout, the measured worst displacement against `leggauss` is
`1.0 u = 0.5 eps`, and an earlier version quoted the `eps` figure while calling it `u`. Every
margin built on it was therefore overstated by two: the provable constant is **3.5x** the
observation, not 7x. That is still the honest gap between a bound and a measurement, and it
should not be closed by quoting the measurement.

## Rule 5: name the arithmetic, and do not assume no underflow

**A hypothesis nobody had stated turned out to be false, and it is worth carrying as the
example.** Higham's model `fl(x op y) = (x op y)(1 + delta) + eta` with `|delta| <= u` is what
every bound in this tree is built on, and the relative half of it silently assumes the operands
are normal.

**Counterexample. REFUTED.** At `n = 3`, `x = 2^-1074`: `fl(5 x P_2)` is
`fl(-2.5 * 2^-1074)`, which rounds to `-2 * 2^-1074`, a **20% relative error** that no
`|delta| <= u` can express. The recurrence bound is violated there by a factor of 5.75e14.

The repair is an additive `2 alpha n^2 H_n` term with `alpha = 2^-1074`, not a hypothesis
excluding subnormal input. At n = 1e6 that term is 1.4e-310, below the smallest normal, so it
costs nothing anywhere it is not needed. This is the same shape as the underflow floor the
infrastructure PR added to the harness after a float32 bound was exceeded by six orders of
magnitude at degree 12: **a relative model needs an absolute companion, and the place it bites
is never where the derivation was checked.**

Every bound in this tree therefore says which arithmetic it lives in. A statement about the
reals is not a statement about binary64, and the crossing is where these derivations fail.

## Rule 6: a sensitivity belongs to the coordinate it was derived in, and comparing maxima cannot tell you otherwise

Rules 1 and 2 are about choosing the wrong frame when a bound is **derived**. This one is about
deriving in the right frame and then **evaluating** in the wrong one, which is harder to see
because the derivation on the page is correct.

**The instance.** The Gauss-Legendre weight bound uses `A(s) = 2|s| / (1 - s^2)`, the logarithmic
derivative of the weight, which follows from the Legendre differential equation collapsing to
`P'' = 2x P' / (1 - x^2)` **at a root of `P_n`**. It is therefore a function of the Legendre
coordinate. The public entry point returns the rule mapped onto `[0, 1]`, and the claim was built
from those mapped nodes.

**Substituting `t = (1 + s)/2` into `A` is not a change of frame, it is a different function.**
`A` is even in `s`; `B(t) = 2t / (1 - t^2)` is not. The map sends *both* singular ends of
`[-1, 1]` to `t = 0` and `t = 1`, while `B` is singular only at the second, so **half the array
loses its singularity**. The two cross at `s = -1/3` (from `3s^2 + 7s + 2 = 0`): above it the
mapped form is looser, below it **tighter**, and the shortfall grows without bound. Measured
3.9x at n = 2 and 1.9e12 at n = 2000 on the amplification alone.

**How it survived a review, which is the part to carry.** A reviewer compared the two
amplification arrays by their **maxima** and got `0.9999991`, and reasonably concluded the frames
might be interchangeable. That number is real and has an exact reason: `w_map B(t) / (w_ref A(s))
= 1 + (1 - s)/(s(3 + s))`, so the weight halves exactly as the sensitivity doubles at `s -> 1`,
and both arrays peak at the outermost node where `1 - s = O(1/n^2)`. **The maximum sits at the one
point where the frame error cancels exactly.** A scalar summary of two arrays is not a comparison
of two arrays.

**The repair needs no new constant**, which is the tell that the frame was the whole problem: pass
the `[-1, 1]` nodes for the sensitivity and the mapped weights for the magnitude. The *relative*
perturbation is frame-invariant, so `w_map (kappa A(s) + 5)` is exactly right. Written in the
mapped coordinate the correct expression is `|2t - 1| / (2t(1 - t))`: symmetric under
`t -> 1 - t`, singular at both ends, zero at `t = 1/2`. `B` has none of those three properties.

**And the guard is cheaper than the derivation.** A Gauss rule is symmetric about the origin, so
any rule of two points or more has a node below `-1/2`. An amplification array built from nodes
with no negative entry at all is the mapped one, handed over by mistake, and refusing it is one
line. **Where a bound depends on a frame, make the frame a precondition rather than a convention.**

## Rule 7: a bound is a property of the code; whether it is approached is a property of the host

A bound that nothing exercises can rot without any test noticing, so the parity suite grew
guards that assert the opposite of a bound: that the two backends still **disagree** somewhere,
that the worst observed ratio is still close to the bound, that the Halley iterate still reaches
a two-value limit cycle. Each of those was a good instinct and each was written as a hard
assertion.

**They are not assertions about this library.** `glibc` selects its `exp` through IFUNC on the
processor's features, and numpy dispatches its own loops the same way, so which arguments the
two round differently is a fact about the machine. Change the host and two implementations that
differed by one ulp can agree exactly. Nothing about the port has changed; the port has got a
better result on that host, and the suite calls it a failure.

**Measured, and this is what settles it against argument.** Commit `767f502` was run twice on
this project's CI with nothing changed between the runs. The first gave `6 failed, 309 passed,
3 xfailed`; the second `313 passed, 5 xfailed`. Two strict `xfail`s XPASSed in the first and not
the second. Same runner image, same numpy, same numba, same compiler. One of the six failures
compares numpy against itself with no C++ involved at all, which is what rules out the change
under review as the cause.

So: **the bound stays a hard assertion everywhere. The liveness guard is enforced only where its
numbers were measured**, which `scripts/ci_local.sh` marks with `PANTR_REFERENCE_HOST`, and
reports with a written reason anywhere else. `tests/_parity_harness.py`'s
`demand_the_reference_host` is the only supported way to write one, for the same reason
`demand_cpp_backend` is the only supported way to require the extension.

The same applies to an `xfail(strict=True)` whose expected failure is a disagreement between two
libraries: `strict` becomes `on_the_reference_host()`. An `xfail` whose expected failure is
structural stays strict, and the two kinds are easy to tell apart because the reason text says
which it is. Of the five strict `xfail`s in `test_quad_shakedown.py`, two say an XPASS would mean
the platform's libraries stopped straddling a threshold, and those two are gated; the other
three are about what a binding accepts and what an emptied rule returns, and stay strict.

**The cost of getting this wrong is not the red.** It is that an intermittent red teaches
everyone to discount reds, which is worth less than no guard at all.

## Rule 8: a parity claim is only defined where the quantity has digits

Added by the change-of-basis port, which is the first module whose kernels **solve a linear
system**, and the first where Rule 3 turned out to have teeth against a bound that was derived
correctly.

Five of that module's eight builders invert an ill-conditioned matrix. Both backends are
backward stable -- LAPACK `gesv` on one side, `Eigen::PartialPivLU` on the other -- so the
honest bound on their disagreement is the standard one, of order `kappa_inf * eps` times a
small multiple of the size. That derivation is not in question. What it produces is.

`pantr.change_basis` defines each builder's domain as the degrees where
`kappa_inf * eps < 1`, which is where the solve is still defined. At the top of that range the
parity bound is therefore *of order one*, and the values being compared are of order one too, so
Rule 3's guard fires and refuses the claim. It is right to. A bound as large as the values it
compares is satisfied by any result, zero included.

**The resolution is not a tighter bound; it is a smaller domain.** Parity is asserted where
`32 n kappa_inf eps < 1`, which is the **accuracy** domain rather than the **solvability**
domain. The two differ per builder and per dtype: for `bernstein_to_cardinal` in float64 the
module accepts degrees to 26 and parity is claimable to 22; for `cardinal_to_bernstein` it
accepts to 14 and parity reaches 9.

**What that boundary is, stated carefully, because the first version of this rule overstated
it.** It said "above the accuracy domain the answer has no correct digits in either backend",
and that is false. Measured against the exact rational matrix, `bernstein_to_cardinal` in
float64:

| degree | `32 n kappa eps` | true relative error | correct digits | this rule |
|---|---|---|---|---|
| 22 | 0.95 | 2.4e-4 | 3.6 | included |
| 23 | 3.79 | 5.8e-4 | **3.2** | excluded |
| 26 (top of solvability) | 263 | 4.6e-2 | **1.3** | excluded |

Three digits survive at the first excluded degree and one at the last. Worse for the claim, the
backends *agree* there: at degree 23 they differ by 1.0e-3 against `kappa eps = 4.9e-3`, so a
claim carrying `kappa eps` rather than the full constant would be non-vacuous and would pass.

So the honest statement is narrower and is about the constant rather than about the
mathematics. **The cut is where *this* bound, with *its* constant, stops being able to say
anything** -- roughly `log10(constant * n)` digits above zero, about three here. Below that cut
a passing test is evidence; above it the harness refuses the claim, correctly, because a bound
larger than the values it compares admits any answer including zero.

What follows for a reader is the part that matters: **the excluded degrees are not degrees where
the port is unchecked because checking is meaningless. They are degrees where this bound is too
loose to check, and a sharper bound would reach some of them.** That is a limitation to record,
not a property to claim.

Two obligations come with it.

**Name the excluded degrees.** A test that quietly stops where its bound runs out reads as full
coverage of the module. `tests/parity/test_change_basis.py` carries a test whose only job is to
enumerate the degrees each builder accepts but this file cannot compare, and which fails if that
gap is ever empty for every builder at once -- because that would mean either the bound had
collapsed or the tabulated domains had moved, and both are findings rather than good news.

**Measure the margin over the whole domain, not at its top.** At the top degree `kappa` is by
construction as large as it gets, so the bound is at its loosest there and a margin measured
only there is misleading. Measured across every degree tested, per builder in float64, the
bound is between **130 and 1800** times the observed difference at its tightest point. At the
top degrees it is far looser, and by an amount that varies over three orders of magnitude
between builders rather than sitting in a narrow band -- an earlier version of this rule quoted
"between 1e6 and 5e6" for it, which took the two largest and presented them as the range. Only
the tightest-point figures are quoted here and in the test's docstring, because they are the
ones that say whether the bound asserts anything.

### What this rule does not license

A different but valid algorithm passing is not a failure of the bound. Measured: swapping
`PartialPivLU` for `FullPivLU` in the C++ Gram projection moves the answer by `1.9e-11` at
degree 10 and the parity tests accept it, correctly -- any backward-stable LU has to be
admitted, since that is exactly what the bound claims. What the bound must still catch is a
*wrong* answer, and it does: a `1e-6` relative perturbation of the Gram matrix is caught by all
five bounded builders and by the margin test.

## Rule 9: an oracle's accumulation width is a per-kernel fact, not a module convention

Added by the Bézier arithmetic port, which is the first module where **every** kernel can be
bit-exact against its oracle, and therefore the first where reproducing the oracle exactly was
the whole of the work rather than a bonus on top of a bound.

`_bezier_core.py` holds seven kernels and uses **three different accumulation widths** among
them, none of them announced:

- the four de Casteljau kernels compute each step in `float64` and round once on the store,
  because their scalars are Python floats and numba promotes a `float64` scalar against a
  `float32` array;
- `_evaluate_bezier_deriv_1d_core` does the opposite. It opens with `dtype = pts.dtype` and
  allocates the `ndu` table, the `a` ping-pong and the derivative table in it, so at `float32`
  the entire A2.3 recursion is `float32`;
- degree elevation and the Bernstein product mix, their coefficient tables being `float64`
  unconditionally against a destination in the control points' dtype, so each `+=` computes wide
  and rounds narrow.

**The root-finding port found the mechanism underneath this, and it is worse than a convention
nobody wrote down.** `float()` does not widen a `float32` in numba's `nopython` mode.
`float(coeff[0])` reads as a promotion and is a no-op; so does `float(np.min(coeff))`. What
widens is **type unification across assignments**, so a variable seeded with `0.0` or
`float("inf")` and later given a `float32` is `float64` throughout, while one that only ever
holds an array element stays narrow however many `float()` calls surround it.

Six sites in the root-finding block carry a `float()` a reader would take for a cast. None
promotes, and three of the five widths there follow from that alone. Two consequences worth
carrying forward:

- **A width cannot be read off the source, and cannot be inferred per kernel or per module.**
  `_batch_core.py:87` divides in `float64` and `_yuksel_core.py:309` divides the same shape of
  expression in `float32`; what separates them is whether some *other* assignment forced the
  variable wider. The transliteration has to go expression by expression.
- **The destination's dtype says nothing about the width the operation ran at.**
  `derivs[0, i] = coeff[i + 1] - coeff[i]` writes into a `float64` array and subtracts in
  `float32`, widening only on the store. The model that widened first reproduced 88 of 4000
  float32 vectors.

`scripts/measure_root_finding_widths.py` measures every one of these behaviourally, two rival
models per site against the kernel, and reports how often the two disagree so a match cannot come
from a check that could not fail. **That script, not a reading of the source, is what a
transliteration should be written against.** The sharpest case is the degree-1 base case
`c0 / (c0 - c1)`, where the `float64` model reproduced *none* of 20000 pairs: a C++
`double root = c0 / (c0 - c1);` breaks parity on every `float32` input, and the difference reads
as round-off rather than as a defect.

**None of this is visible at `float64`, where all three coincide.** Measured on one kernel:
accumulating narrow where the oracle accumulates wide moves 125 of 630 `float32` values.
Widening where the oracle narrows is the same error the other way and was caught the same way,
in `_evaluate_bezier_deriv_1d_core`, by a mutation that failed nine parity cases.

So the rule is procedural rather than mathematical: **read the width off each kernel, one at a
time, and mutation-test that the parity suite would notice if you got it wrong.** Inferring it
from the module, or from the kernel next door, is what produces a port that is exact at
`float64` and quietly wrong at `float32` -- which is the half of the matrix nobody reads first.

### The sharper case, and why measurement rather than reading found it

Within a single kernel, `_evaluate_bezier_1d_core` seeds its two branches from bases of
different width. Above the mirror threshold it raises `u`, which is the point array's own dtype;
below it it raises `1 - u`, which the literal `1.0` has already promoted to `float64`. Reading
the source does not make that leap out, and the consequence is a single value in a whole-kernel
sweep: at degree 17 and `u = 0.75`, where `0.75^17 = 3^17 / 2^34` needs 27 significand bits, the
wide seed survives and the narrow one rounds.

A sweep that had used round parameters and a modest degree would have missed it entirely. What
found it was a case list built to be adversarial about *representability* rather than about
magnitude.

### Every figure above is reproducible, and that had to be fixed

An earlier version of this rule quoted counts whose only artifact was a scratch directory. They
were real, and three were later reproduced exactly by an independent reviewer, but a number
nobody else can re-derive is a number nobody else can refute, which is the wrong shape for a
design note. `scripts/measure_bezier_parity.py` now reproduces the sweep counts and the `pow`
agreement, and prints the rebuild command for the two figures that need `-march=native`.
`cpp/include/pantr/basis/bernstein.hpp` had been doing this correctly for its own claim all
along, by pointing at a committed test.

### The `float32` half is far less contraction-sensitive, and that is a consequence

Three kernels now measured, with `-march=native` against the same build without it:

| kernel | `float64` values moved | `float32` values moved |
|---|---|---|
| de Casteljau (slice) | 125 / 630 | 2 / 630 |
| reduction-operator apply | 237 / 970 | 0 / 970 |

The asymmetry is the wide accumulator doing its job: a fused multiply-add changes the `double`
intermediate, and the narrowing store to `float32` then discards most of the difference. Worth
knowing before the ISA ladder of `design/simd.md` lands, because it says the `float64` path is
where a contraction bound will actually be needed.

### What was NOT here, and was a real gap -- closed 2026-08-22 by Rule 10

**No parity bound was derived for a fusing build.** Unlike the Bernstein tabulation, every
kernel in this module contains an `a * b + c * d` site, and
`tests/parity/test_bezier_arithmetic.py` skipped rather than weakened when `__fp_contract__`
reported a fused multiply-add. **The reason given for deferring it was refuted**: it read "a
bound written for a branch no host in this project can execute would ship untested", and
`tests/parity/test_quad_gauss_legendre.py` had been deriving such a bound and probing it on this
same non-fusing host since the `quad` port. Three lines of it. Rule 7 licenses conditioning a
claim on the host; it does not license leaving the other branch unwritten.

Rule 10 is that bound. The three tests that still skip do so for reasons the bound genuinely
cannot cover -- a discrete verdict, a wrapped factorial, operands not observable from the public
surface -- and each names its own.

The `float32` asymmetry above survived the derivation and is now explained rather than observed:
seven of the eight kernels contract in a `float64` accumulator, so only a straddled narrowing
store carries a difference to a `float32` output.


## Rule 10: contraction removes one rounding per fused site, and only the amplification differs

Added by the Bézier arithmetic port's follow-up, which derived the bound Rule 9's section
deferred. It is the answer to a question the four earlier ports could postpone: what does a
parity claim say on a build whose target ISA has a fused multiply-add?

**The budget is one line and it is the same for every kernel.** At a fused site the oracle
computes `fl(a + fl(b*c))` and the C++ backend `fl(a + b*c)`. Writing the first as
`(a + b*c(1 + d1))(1 + d2)` and the second as `(a + b*c)(1 + d3)` with every `|d| <= u`, the two
differ by at most

    |b*c| * u * (1 + u)  +  |a + b*c| * 2u,

so **three accumulator roundings per stage**. Where the storage format is narrower than the
accumulator, the two pre-store values can fall either side of a rounding boundary, and a straddle
costs one ulp, so **two storage roundings per stage**. In the harness that is
`Roundings(stages, 3, 2)`, which serves all eight kernels unchanged.

**What differs per kernel is the amplification, and getting it from `max|c|` to something usable
was the whole of the work.** `max|c|` is a correct magnitude for every convex kernel here and it
is useless: on a net spanning twelve decades whose output cancels to order one, it bounds a
`float32` de Casteljau by 25 against values of 0.75, and Rule 3 refuses that. Seven parity cases
failed exactly that way, all `float32`, all at degree 25 or within 1e-8 of an endpoint.

The fix is the one Rule 3's own failure message prescribes, and it is licensed here because these
recurrences are convex: **run the same operation on `|c|` and use the result, elementwise.** Every
weight is non-negative and they sum to one, so the absolute-value companion is exactly the
magnitude reachable at each output element, and it is tight rather than merely valid. Two kernels
are not convex and must not use it:

| kernel | stages | narrowing stores | amplification |
|---|---|---|---|
| `evaluate`, `slice`, `split` | `p` | one per stage | the absolute-value companion |
| `restrict` | `2p`, two passes | one per stage | the absolute-value companion |
| `degree_elevate` | `min(p, t) + 1` | one per stage | the absolute-value companion |
| `scalar_bernstein_product` | `min(p,q) + 1` | one per stage | `sum_i C(p,i)C(q,k-i)|a_i||b_{k-i}| / C(p+q,k)` |
| reduction apply | `p + 1` | **one in total** | `|R| @ |c|`, since `R` has negative entries |
| `evaluate_deriv` | `2p + k + 2` | one per stage | the A2.3 majorant's row action, accumulator = storage |

**Three of those six entries were wrong in the first version and a tolerance audit found them.**
`degree_elevate`'s chain is `min(p, t) + 1` and not `p + 1`, because the accumulation into `out[i]`
runs `j` from `max(0, i - t)` to `min(p, i)`; charging `p + 1` was 13x too loose at `p = 25`. The
reduction apply accumulates into a `float64` local and narrows **once per output element**, outside
the loop over the operator row, so charging a store per stage over-counted it by `p + 1`, measured
up to 52x at `float32`. And a zero-stage claim -- degree 0, where every one of these kernels
short-circuits -- was being clamped to one stage instead of being what it is, which is bitwise;
`bounded_parity`'s own guard says so and the clamp was suppressing it.

### Why the derivative kernel cannot use a companion, and what it uses instead

The convex kernels get the absolute-value companion because their weights are non-negative and sum
to one. **A2.3 differences, so it has no such companion**, and the first attempt at its
amplification failed twice in instructive ways.

`p!/(p-k)! * 2^k * max|c|` is correct, follows from
`B^(k) = p!/(p-k)! sum_j (Delta^k c)_j B_{j,p-k}` with `||Delta^k||_inf <= 2^k`, and is **unusable**:
it bounds the operator norm rather than the row, and applies the *input* maximum where the
comparison happens per output element. On the parity suite's own parametrization that made 45 of
280 non-zero `float32` values carry a tolerance at least as large as their own magnitude, worst
case 1.6e5 times. Rule 3's guard did not fire, because it compares the largest tolerance against
the largest magnitude and a flat amplification is sized for exactly that element. **That is Rule 6's
failure committed inside Rule 10**: a scalar summary hiding an elementwise disagreement.

Replacing it with the row action of the finished operator -- `sum_j |B^(k)_{j,p}(s_i)| |c_{j,r}|`,
obtained by driving the kernel with the identity net -- fixed the vacuity and was then **exceeded by
4.32x** at degree 17, orders 4 and 6, identically at both dtypes. The identical figure is the
signature: a structural shortfall, not rounding. The finished row is not enough precisely because
the recursion is not convex, so its partial sums can exceed what survives to the output, and no
telescoping argument recovers them.

What works is the **majorant of the recursion itself**: run A2.3 with every coefficient replaced by
its modulus. For a linear recursion with signed coefficients that bounds every intermediate, by
induction, and with the signs gone the partial sums are monotone so the final value majorises all of
them. Two lines change from the oracle, both in the `a` table. Checked over 75563 (point, basis)
entries: **0 violations, and a largest ratio of exactly 1.000000**, so it is attained rather than
conservative.

### The harness's factor of two is a margin here, not a derivation

`absolute_tolerance` multiplies by `ONE_SIDED_TO_TWO_SIDED = 2`, whose docstring justifies it by
"each backend sits within its own forward-error bound of the exact recurrence, so their difference
is bounded by the sum of the two". **That justification does not apply to these claims.** The budget
above is already a backend-to-backend *difference*, derived as one, not a one-sided forward error.
The other parity files pass one-sided counts and say so.

The factor stays, and its provenance for Rule 10 is restated rather than inherited: it is an
acknowledged safety factor of two, covering the compiler's freedom in choosing which of two products
to fuse at a site with both, and the re-rounding of operands that have already diverged. That second
one is first order rather than second, because correctly rounded arithmetic on inputs one ulp apart
does not track the gap.

### The premise the bound rests on, and it is not ours

**The oracle never fuses.** That is not obvious: numba compiles for the host CPU, so on any
machine with an FMA it could. It does not, because LLVM does not contract without `fastmath`, and
no pantr kernel sets it. Verified discriminatingly rather than by observing a zero: on this host
`a * b + c` compiles to no FMA under `fastmath=False` and to one under `fastmath=True`.

That is a property of numba's defaults and could change under us, which would make the bound
describe the wrong difference -- it assumes the extra rounding is always on the oracle's side.
`test_the_oracle_does_not_contract_a_multiply_add` pins it.

### What the enumeration cost, and why reading the source was not enough on its own

Fourteen sites fuse. Reading the source predicted fourteen and the disassembly of a
`-march=native` build found the same fourteen, which is the check being reported rather than a
coincidence worth relying on next time. What reading alone would **not** have given is the
*width* of each: seven of the eight kernels contract in a `float64` accumulator whatever the
storage, so at `float32` only a straddled narrowing store carries the difference out, and they
barely move. `evaluate_deriv` emits four storage-width FMAs and is the exception -- Rule 9
reappearing, now on the bounded side.

**Contraction is the only mechanism.** `-march=native` with `-ffp-contract=off` on top restores
bit-identity exactly while the vectorisation stays, which separates the two the same way the
`quad` port separated them.

### Two build facts that will outlive this rule

`-DCMAKE_CXX_FLAGS=-ffp-contract=off` **does nothing**: `PantrCompileOptions.cmake` adds
`-ffp-contract=on` as a target option, which lands after `CMAKE_CXX_FLAGS` and wins. The first
build made to isolate contraction was byte-identical to the fusing one and only
`compile_commands.json` showed it.

And **Eigen does not compile under `-march=native` with `PANTR_WERROR=ON`**: its AVX512
`TrsmKernel.h` trips `-Wmaybe-uninitialized` in its own code. That is a decision waiting for the
ISA ladder of `design/simd.md`, not for this rule.

`scripts/measure_bezier_fma_bound.py` reproduces every figure above.

## The two corrections the infrastructure PR made, kept here because they generalize

**The underflow floor.** `eta` per rounding, the unhalved smallest subnormal. Without it the
shipped bound was exceeded by a factor of 1e6 on the cardinal B-spline oracle set, in float32 at
degree 16. One entry in a degree list away from turning the suite red against correct code.

**Hull propagation of the amplification.** A bound built from the *computed* coefficient
collapses when that coefficient rounds to exactly zero while the true one does not, which
happens for `1 - u` with `u` below about `eps/4`. Widening each computed coefficient by its own
propagated error makes the companion cover both. Without it the bound could be violated by
`1/u`, a factor of `2^53`.

## Did the harness generalize? Yes in vocabulary, no in meaning

This section was left unwritten on purpose until `quad`'s parity tests existed, because a
contract three more ports inherit should not be sealed from one and a half data points. The
tests exist now. The answer has three parts and the middle one is the useful one.

### It generalized in vocabulary: no new kind, no new field

All five `quad` comparisons are expressible with what was already there. `bitwise_parity` covers
Gauss-Legendre on the shipped build, Lambert W, the trapezoidal rule, and the modified Chebyshev
nodes in float64. `bounded_parity` with `Roundings` and an `amplification` array covers the
modified Chebyshev nodes in float32, tanh-sinh, and Gauss-Legendre's unreachable FMA branch. **No
third `ParityKind` was needed and no second magnitude array.**

The proposal for a `bound=` factory taking an elementwise array directly was refused before the
tests were written, on the grounds that it is one step from laundering a fitted tolerance. **The
refusal held up, and the reason turned out to be different from the one given.** It is not that
such a factory is dangerous in itself: what keeps a bound honest is the derivation in `why` plus a
test asserting the bound is *reached*, and both apply to any constructor. It is that
`bounded_parity` makes the unit `u` structural. Every bound below is a multiple of a unit of
roundoff and it is impossible to write one that is not.

### It did not generalize in meaning, and two of the four claims say something they do not mean

`Roundings` is documented as a count along a dependency chain: stages, roundings per stage. That
is what the cardinal B-spline kernel is. **Two of `quad`'s bounded claims are not chains at all**,
and both were written as `Roundings(stages=1, accumulator_per_stage=1, storage_per_stage=0)` --
literally "one rounding" -- with the entire derivation living in `amplification` and `why`:

- **Gauss-Legendre nodes.** The bound is a *ratio*, `C(n, x_i) / |P'_n(x_i)| <= 7/2`, plus one
  rounding. A ratio is not a rounding count, and no stage count expresses it.
- **tanh-sinh.** The bound is a libm budget in units in the last place, amplified by
  `2 omega` through an exponential, plus a summation-order difference between numpy's pairwise
  sum and a plain accumulation. None of the three is a stage.

So in those two, `Roundings` has been reduced from a count to **a way of writing the unit `u`**.
A reader who takes the field at its documented word will conclude the Gauss-Legendre kernel
commits one rounding, which is false: it commits about `5 n` of them and the bound survives
because they cancel. **This is a real cost and it is recorded rather than fixed**, because the
alternative -- a constructor whose argument is the bound itself -- is the one that was refused,
and refusing it is still right. The mitigation is that `why` is mandatory, is quoted verbatim in
every failure message, and in both cases carries the actual derivation.

`amplification` has the matching problem. Its docstring says "elementwise factor by which the
recurrence magnifies a relative perturbation", which is half of what it must carry: the other
half is **the magnitude that turns a relative bound into an absolute one**. For the cardinal
B-spline the two coincide numerically, because every value is in `[0, 1]` and of order one. For
`quad` they separate visibly -- Gauss-Legendre weights span two and a half decades at
n = 700 and their *mapped nodes* span five and a half, so the
magnitude has to be multiplied in by hand, while the nodes are of order one and do not. **A field
doing two jobs that happen to coincide in the only consumer is exactly what one consumer cannot
reveal.**

### Why it had to be touched twice, and why neither was visible at design time

**The vacuous-bound hole.** A bound at least as large as the values it compares was accepted: a
reference of `-1e250` against an actual of `1.0` passed. Fixed before `quad` needed it, in its own
commit.

**The gamma form, which is the instructive one.** `_relative_growth` computed
`(1 + per_stage)**stages - 1`. In binary64 that is **exactly zero** whenever `per_stage` is one
unit of roundoff, at every stage count: `1 + eps/2` lands on the midpoint between `1` and
`1 + eps` and round-half-to-even carries it back, because `1`'s significand is even. The claim
then said BOUNDED while asserting bit-for-bit agreement.

**It was invisible because the only consumer never hit it.** The cardinal B-spline budget is
`accumulator_per_stage = 2`, so `per_stage = eps`, which added to `1` is representable and the
expression works. The defect needs *exactly one* rounding per stage -- which is the ordinary shape
of a kernel that does not narrow, and is the first thing `quad` asked for. And the guard written to
catch precisely this class tested `per_stage == 0`, which is `1.11e-16` here and passes: **a
budget can be non-zero and still produce a zero bound.**

The general lesson, and it is the one to carry into the next port: **a single consumer cannot
distinguish "the vocabulary is right" from "the vocabulary happens to work at this consumer's
parameter values."** Both defects were in the parameter regions the first consumer did not visit,
not in the design. The second consumer is where a contract is first tested, and the third will
find something else.

### What the next port should expect to add

Nothing structural, on this evidence. But **`Roundings` and `amplification` both need their
docstrings corrected** to say what they actually carry, and a fourth consumer whose bound is again
neither a chain nor a magnitude would be the point at which the field's meaning, rather than its
documentation, has to change.

## Rule 11: an exact tie does not survive contraction, and a tie-break is a verdict

Added by the Bézier root-finding port, the first whose kernels are **iterative** and whose
results carry a discrete verdict as well as a value: how many roots there are, which intervals
survive, which hull vertices are kept. Rules 1 to 10 all bound a displacement. None of them says
anything about a count, and a count is what this block can get wrong.

**The mechanism.** The convex-hull clipper's orientation predicate is `a*b - c*d`. On an exactly
collinear control polygon the two products are equal reals and the expression evaluates to
**exactly zero**, so `cross >= 0` holds and Andrew's monotone chain pops the vertex. Under
`-ffp-contract=on` on a target with a fused multiply-add, the compiler may compute
`fma(a, b, -(c*d))`, which returns the exact residual of the rounding in `c*d`: a tiny nonzero
value of arbitrary sign. The tie-break becomes a coin toss.

Measured on 200000 constructed collinear triples: 184224 evaluate to exactly zero unfused, and
fusing flips the branch on **130538**. Measured against the kernel itself, an unfused
transliteration agrees on all 4800 collinear cases and a fused one on 3833.

A collinear control polygon is not exotic. A degree-elevated linear polynomial, an affine segment
and a constant all produce one.

**Why this is a different kind of hazard, and one sentence here was wrong.** Rule 10's
contraction bound absorbs a displacement because the two backends compute the same quantity to
different last bits. Here contraction changes which branch is taken, and a branch is not a
quantity a tolerance can bound.

The first version of this section then said that downstream of a flipped tie-break the two
backends compute different things. **Measured, that is false for the mechanism it names.**
Andrew's monotone chain with `cross >= 0` pops a collinear vertex and with a tiny negative
residue keeps it, but either way the hull is the same **polyline**: keeping a collinear vertex
splits one edge into two collinear pieces and the crossing lies on exactly one of them, at the
same point. Over the collinear family at every degree and both contraction orders, vertex-set
flips do occur, the returned interval changes in fewer cases than the vertex set does, and when
it changes it changes by **one to two ulp**. That is a displacement, which is exactly what a
tolerance covers.

**The conclusion stands, for a mechanism this rule did not name.** A count can change, but
through threshold comparisons on displaced values rather than through exact ties: fusion moves
the subdivided net by ulps, and `_count_sign_changes`, `abs(local[0]) <= local_zero_tol` and
`c_min > local_zero_tol` all read it. That mechanism is generic and has nothing to do with
collinearity, which means a collinear adversary is aimed at the wrong target. How close that
cliff sits: perturbing one coefficient by one ulp across the parity matrix changes the root
count in **199 of 1464 perturbations**.

**What follows for a parity claim.** Assert the count on its own, before and separately from any
numeric comparison. `tests/parity/test_bezier_root_finding.py` does, and its failure message says
that a changed count is a changed verdict rather than a displaced value. Folding the two together
is the error to avoid: a bounded claim compares elementwise and cannot see two results of
different length at all.

**What is measured and what is not.** Over 732 public results on a `-march=native` build, 619
were identical, 113 moved in value and **zero changed the root set**. That is evidence and not a
proof: the tie-break demonstrably can flip, so a set change is possible and simply did not occur.
The count agreement is asserted per case rather than derived.

### The bound where contraction is live: certify it, do not predict it

Two predicted bounds were tried here and both were refuted. They are recorded because the way
each failed is more useful than the formula that replaced them.

**The first was fitted where one term happened to dominate.** Over 732 float64-heavy results the
worst displacement was `4.951e-13` against a `param_tol` of `1e-12`, so the bracketing tolerance
looked like the whole bound and looked *tight*, which is normally the sign a derivation is right.
At `float32` a degree-5 case moved by `2e-8`. The term it omitted is around `1e-15` at float64
and cannot be seen there. **A measurement cannot refute a bound in a regime where the missing
term is invisible.**

**The second predicted the acceptance component's half-width as `eps_f / |f'(r)|`, capped at
`eps_f^(1/3)`, and an adversarial review took it apart on four counts:**

- `eps_f = (degree + 1) u max|c|` is **false at float64**, by an exact-rational counterexample at
  degree 1 exceeding it by 1.126x. The literature constant is Mainar and Peña's `gamma_2n`
  (1999), corrected by Hermes (2019) to `gamma_3n` because the original did not charge the
  rounding of `1 - t`, which this kernel commits.
- **It used the wrong epsilon.** The kernel accepts a candidate on `abs(f_final) <= zero_tol`,
  and computes `zero_tol` itself. For `f(t) = t - 1/2`, the simplest polynomial in the parity
  matrix, the acceptance band is **exactly twice** what the bound claimed.
- The linearisation needs a curvature hypothesis nobody stated, and there is a measured window
  at `|f'| ~ 2 sqrt(eps)` where it fails by 2.75x **and the cap has not yet engaged**. The two
  thresholds are unrelated quantities; the cap only fires three decades further down.
- **The cap is not scale-invariant and is dimensionally wrong.** `eps_f^(1/3)` has units of
  `[f]^(1/3)` and is compared against `[t]`; the correct cluster half-width is
  `(eps / |a_m|)^(1/m)`. Measured, a triple root came back **bit-identical over 2^60 of
  coefficient scaling** while the cap moved by a factor of a million. A bound that moves when
  the bounded quantity does not is not a bound. This one is inherited: `_dedup_roots_core`'s own
  `radius_cap` has the same two defects, and predates the port.

**Certifying answers all four at once, and it is the cheaper thing to write.** Rather than
predicting the component's width from derivatives at the root, prove it: restrict the Bernstein
net to each flanking interval and use the convex-hull property, which says the graph over an
interval lies within the range of that interval's own coefficients. If they are all above `eps`
or all below `-eps`, no point of that flank can be accepted, so the component cannot reach past
the half-width being tested. Binary search the half-width; certification is monotone in it.

The certificate is scale-invariant because scaling the coefficients scales the band with them;
dimensionally correct because a half-width is the thing it searches for; and multiplicity-
agnostic because no derivative appears. The restriction is done in **exact rational arithmetic**,
which makes the hull bound a proof rather than an estimate needing its own error term, and it is
written out rather than borrowed from `_subdivide_scalar`, so a bug in the kernels cannot
certify itself.

Verified on the case the old cap existed for: a triple root certifies at `1.221e-04` for a
coefficient scale of 1, 1e6 and 1e-6 alike. Over the parity matrix it bounds **83 of 86** cases
below the parameter interval; the three it does not are `float32` polynomials whose roots have
no digits left, which Rule 8 excludes by name.

The band it certifies against is the algorithm's own `zero_tol` plus the evaluation's forward
error at the corrected `gamma_3n`, written first-order as `n u_T + 3n u_64`. Note which epsilon
that is: the one the code uses, not one derived for it.

### A claim kind the harness did not have

None of this can be written with `bounded_parity`, which builds its bound from a rounding budget
times an amplification. Reaching a given number that way would mean choosing an amplification to
produce it, which is the fitted constant the harness exists to make unsayable. `ParityKind`
gained a third member, `CONVERGED`, carrying an absolute bound and the scale its quantity lives
on.

### The branch that ships unevaluated is the one that is broken

Rule 9's section deferred the fused claim entirely and Rule 10 derived it, but on the shipped
build neither branch runs: the target carries no `-march`, so nothing fuses and every claim takes
its bitwise arm. **This port's fused branch shipped broken during development** -- it called
`bounded_parity` with an argument that function does not take -- and 133 tests passed over it,
because none of them reached it. It was found by building the extension at `-march=native` and
running the suite there.

An unexercised claim is not a weak claim, it is an unevaluated one. Building the fusing variant
and running against it is cheap, the recipe is in `scripts/measure_bezier_fma_bound.py`, and it
should be part of finishing any port that carries a conditional claim.

## Rule 12: the interpreted oracle is a different object, and which claims survive it is knowable

`make coverage` runs the whole suite with `NUMBA_DISABLE_JIT=1`, because coverage.py cannot
trace machine code numba compiled. That is not a second opinion about the same kernels. It
replaces the oracle with interpreted python over numpy, and **three things then differ, no
more and no fewer**. Naming them is what lets a claim be gated precisely instead of a whole
suite being skipped.

Every figure below is printed by `scripts/measure_interpreted_divergences.py`, so it can be
re-measured rather than re-read. Expect the `pow` count to move with a numpy or numba upgrade;
what must not move is that it is non-zero.

**One: storage width, which is Rule 9 read in the other direction.** Rule 9 records that
`float()` does not widen a `float32` in `nopython` mode. Interpreted, `float()` is CPython's
and *does* widen, so every one of the six sites Rule 9 enumerates promotes where the compiled
kernel does not. `demand_the_compiled_kernel(dtype)` owns this, and only at `float32`.

**Two: `pow`, which IEEE 754 does not pin.** `_bernstein_point`, `_bernstein_point_no_mirror`
and `_evaluate_bezier_1d_core` seed a ratio recurrence with `np.power`, and what their bitwise
claims rest on -- stated in their own `why` text -- is that *numba's* `np.power` agrees with the
platform libm. Interpreted, numpy's is called instead. Measured: they differ by exactly one ulp
in 2887 of 60000 arguments swept over degrees 0 to 29.

**Three: integer width, which is not a rounding question at all.** `_bernstein_derivs_point` and
`_evaluate_bezier_deriv_1d_core` accumulate a falling factorial in an integer that wraps at
int64 when compiled and grows without bound when interpreted. At degree 25 order 16 the true
value is `42744736671436800000` and its two's-complement wrap is `5851248524017696768`, a ratio
of 7.305, which is what the parity failure reports. No tolerance answers this: the claim has to
be skipped, not loosened. `_bincoeff` is the deliberate counter-example, casting each step to
`np.int64` explicitly and so wrapping identically either way.

**And the complementary half, which is what stops the gate being written too wide.** For a
kernel built from `+`, `-`, `*`, `/` and `sqrt`, IEEE 754 pins every result, so the interpreted
path reproduces the compiled path's bits **by guarantee rather than by luck**, and a bitwise
claim over it is as true there as anywhere. This was not obvious and cost two wrong attempts.
Keying the gate on "the JIT is off" skipped 433 test cases to fix 15; keying it on "the claim is
bitwise" skipped 442 of 495 `float64` cases, no better. Keying it on the two constructs above
skips 79 and costs no statements of coverage at all.

So: a **bitwise** claim over a kernel carrying `pow` or an integer accumulator must be gated. A
**bounded** or **converged** claim need not be, because a bound derived from a rounding budget
absorbs a seed that moved by an ulp. The three gates live in `tests/_parity_harness.py` and
`tests/test_jit_disabled_configuration.py` runs the configuration in the fast suite, since
nothing else outside `make coverage` exercises it.

### The consequence nobody had drawn: a regression test can be inert in this configuration

`TestSignTestUnderflow` pins the fix for FELIGN/pantr#351, an underflow in a product-form sign
test. **It cannot exercise that defect under `NUMBA_DISABLE_JIT=1` at all**, and this follows
from the first divergence above rather than from anything about the test. The defect needs a
`float32` product to underflow; interpreted, `float()` widens the operands to `float64` and the
product is about `-1e-46`, nowhere near it. At the frontier coefficient the narrow product is
exactly zero and the widened one is not, which is the whole difference.

So `make coverage` would never have caught a regression of #351, before or after the gates were
written. Nothing said so, and nothing in the output distinguishes a test that ran and passed
from one that ran a different computation and passed.

**The general form, and it is the part worth carrying:** the interpreted path is a different
object, so a test that pins a *width-dependent* defect is inert there, silently. Rule 9 already
says a width cannot be read off the source. This adds that a width-dependent **test** cannot be
assumed to run in every configuration the suite is executed in, and that the suite reports
nothing when it does not.

## Epistemic status

- **PROVED, with the argument recorded outside this repository:** the closed form
  `C(n, ±1) = (7/4) n^2 + (9/4) n - 4 H_n`, by induction; the exact Green's-function
  representation of the recurrence's error, with the Casoratian `P_{k-1} S_k - P_k S_{k-1} = 1/k`
  for the second solution `S` (`S_0 = 0, S_1 = 1`), which is the route that closes because the
  `artanh` in the usual second solution cancels identically and leaves a polynomial kernel. The
  proofs and their runnable checks are in the shared corpus under `proof-techniques/proofs/`,
  dated 2026-08-20; they were checked step by step by a separate agent. **Also PROVED, by a later
  pass:** `sup_x C(n, x) = C(n, ±1)` attained, and the `7/2` ratio bound non-asymptotically, both
  through the Green's-kernel closed form above.
- **SUPPORTED:** the `Theta(n^2)` growth as a *fit*, at `n^1.936` over all `x` (the bound itself is
  now proved); the `2.6` weight-amplification cancellation, monotone and convergent to 2.566323
  over a sweep to n = 12000, and re-measured by a test.
- **What was open and is now closed differently than this note once said.** An earlier version of
  this section said "one interchange step in the supremum argument is named there as unclosed".
  That was wrong twice: the corpus names **two** unbounded steps, and they belong to the `7/2`
  **limit** argument, not to the supremum, whose gap was a missing kernel bound and not an
  interchange at all. Still genuinely open: that `7/2` is the exact *limit* rather than some
  constant below it. The bound does not depend on the answer.
- **REFUTED, with the counterexample:** that the recurrence error is `O(n)` (it is `Theta(n^2)`,
  and the measurement that said otherwise sampled the wrong point set); that Higham's relative
  model needs no absolute companion in the subnormal range; that the absolute-value companion
  recurrence bounds an oscillatory recurrence.
- **Measured, not derived:** every parity figure quoted against `leggauss` and against the
  tanh-sinh golden file. Those compare two implementations rather than one implementation
  against the truth, and they are stated as measurements throughout.
- **Not investigated:** whether the ratio bound of Rule 4 survives beyond n = 2048, and whether
  the vendor-documented transcendental error is the same on the platforms CI runs that this
  machine does not. `CLAUDE.md` already records that numpy behavior differs across the test
  matrix, so this is a live question rather than a formality.

## Open questions

1. Rule 4's `7/2` is a swept supremum, not a theorem. The limit argument for it exists but names
   two interchange steps as unbounded. Does it close, and does the constant move if it does?
2. The third bound category, a vendor-documented transcendental error, has no home in the
   harness's vocabulary yet and no source consulted. glibc publishes a maximum-known-error table;
   nobody here has read it, and until someone does that category is a placeholder.
3. Rule 2 says use two comparisons. Is there a rule for **how many**, or does each kernel argue
   it? Four comparisons per rule was proposed for `quad` and never justified as the right number.
4. Rule 12 enumerates three divergences and claims they are all of them. Two were found by a
   parity failure and one by reading; none was found by looking for the *fourth*. A
   transcendental other than `pow` in a future kernel would be one, and nothing would report it
   until a bitwise claim over that kernel failed under `make coverage`.
