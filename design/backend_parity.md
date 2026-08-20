# Backend parity: what a bound may claim, and in which frame

**Status:** partial, on purpose. The rules below are settled and three further module ports
inherit them. The verdict on whether the parity harness generalized from one kernel to four
is deliberately not written yet; it is a deliverable of the `quad` port and belongs here once
that port's tests exist rather than before.
**Date:** 2026-08-20.
**Scope:** how the two backends are compared, and what a comparison is allowed to assert. Not
which algorithm computes what, which is `design/quadrature_algorithms.md`, and not what may
cross the boundary, which is `design/cross_backend_types.md`.

**Validated against:** `proto/cpp` at `765d9b9`, plus the `quad` port on `feat/cpp-quad`.

## The one fact everything else follows from

**The Python implementation is the oracle.** Parity says the two backends agree; it does not
say either is right. So a shared error is invisible to every parity test that will ever be
written, and each ported module owes an *independent* check as well: exact rational arithmetic,
a closed form, an analytic invariant, a convergence order. `tests/_parity_harness.py` has a
separate slot for that (`assert_accuracy`) and it is not optional.

The infrastructure PR measured what this is worth: an independent exact-rational oracle caught
a bug injected into **both** backends at once, by a factor of 5e14 over its bound. Nothing else
in the suite could have.

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

This is the subtlest of the five and the one most likely to be got wrong by someone doing the
obvious thing.

**Statement.** Let `P_n` be evaluated in binary64 by the forward three-term recurrence
`P_k = ((2k-1) x P_{k-1} - (k-1) P_{k-2}) / k`, round to nearest, no FMA, `k < 2^52`, with `P_0`
and `P_1` exact. Write `C(n, x)` for a bound on `|fl(P_n(x)) - P_n(x)| / u`. Then:

- `sup_x C(n, x)` is **`Theta(n^2)`**. Closed form `C(n, ±1) = (7/4) n^2 + (9/4) n - 4 H_n`.
  **PROVED** as a closed form by induction; **SUPPORTED** as the supremum.
- The ratio `C(n, x_i) / |P'_n(x_i)|` at the Gauss nodes is **bounded by 7/2 uniformly in `n`**.
  **SUPPORTED**: swept exhaustively over every `n` from 2 to 600 and selectively to 2048, worst
  value 3.4999940516, monotone, always attained at the outermost node.

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
(`4.5 u` including the update's own rounding). Measured displacement against `leggauss` is
`0.5 u`, so the provable constant is 7x the observation, which is the honest gap between a bound
and a measurement and should not be closed by quoting the measurement.

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

## The two corrections the infrastructure PR made, kept here because they generalize

**The underflow floor.** `eta` per rounding, the unhalved smallest subnormal. Without it the
shipped bound was exceeded by a factor of 1e6 on the cardinal B-spline oracle set, in float32 at
degree 16. One entry in a degree list away from turning the suite red against correct code.

**Hull propagation of the amplification.** A bound built from the *computed* coefficient
collapses when that coefficient rounds to exactly zero while the true one does not, which
happens for `1 - u` with `u` below about `eps/4`. Widening each computed coefficient by its own
propagated error makes the companion cover both. Without it the bound could be violated by
`1/u`, a factor of `2^53`.

## What is deliberately not settled here

**Whether the harness's vocabulary generalized.** `bounded_parity` models a dependency chain
with a per-stage rounding count. Two of `quad`'s four comparisons are not chains: a Newton fixed
point is limited by the conditioning of a simple root, and a rule generator's transcendental
evaluations are limited by what the vendor documents about its own library rather than by
anything visible in our source. Whether that is expressible as a third budget against a second
magnitude, or wants a distinct kind, is answered by writing the tests, not by predicting them.
This section gets written after they exist. A contract three ports inherit should not be sealed
from one and a half data points.

## Epistemic status

- **PROVED, with the argument recorded outside this repository:** the closed form
  `C(n, ±1) = (7/4) n^2 + (9/4) n - 4 H_n`, by induction; the exact Green's-function
  representation of the recurrence's error, with the Casoratian `P_{k-1} S_k - P_k S_{k-1} = 1/k`
  for the second solution `S` (`S_0 = 0, S_1 = 1`), which is the route that closes because the
  `artanh` in the usual second solution cancels identically and leaves a polynomial kernel. The
  proofs and their runnable checks are in the shared corpus under `proof-techniques/proofs/`,
  dated 2026-08-20; they were checked step by step by a separate agent, and one interchange step
  in the supremum argument is named there as unclosed.
- **SUPPORTED:** the `7/2` ratio bound, swept to n = 2048; the `3.5 u` node displacement; the
  `Theta(n^2)` growth, fitted at `n^1.936` over all `x`.
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
