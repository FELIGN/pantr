# Which algorithm computes each quadrature rule, and why not the obvious one

**Status:** decided for the four rules in the `quad` port's scope. The other three are
recorded as unported, on purpose.
**Date:** 2026-08-20.
**Scope:** the algorithm behind each 1D rule in the C++ port, and the measurements that
chose it. Not the tolerance policy that compares the two backends, which is
`design/backend_parity.md`, and not the seam shape, which is
`design/cross_backend_types.md`.
**Companions:** `design/large_data_fitting.md`, which holds Eigen's appraisal and its
scheduled first use.

**Validated against:** pantr **0.7.0** and `proto/cpp` at `765d9b9`.

## The decision

**Gauss-Legendre is computed by Newton on the Legendre three-term recurrence.** Not by
Golub-Welsch, and not with Eigen.

The port was specified to use Golub-Welsch on the tridiagonal Jacobi matrix, on the
understanding that numpy computes the nodes by eigenvalues of a companion matrix and that
the two would therefore be genuinely different algorithms whose disagreement had to be
bounded from the conditioning of the eigenvalue problem. **Reading numpy's source refutes the
premise.** `numpy.polynomial.legendre.leggauss` builds `legcompanion`, which for Legendre is
**symmetric** (measured: `max |M - M^T| = 0` exactly, and `max |M - J| = 1.1e-16` against the
Jacobi matrix, so it is the Jacobi matrix to one ulp), calls `eigvalsh`, the *symmetric*
solver, then applies **one Newton polish**, then computes the weights from the classical
derivative formula, symmetrizes them and rescales the sum to 2. numpy is already doing
Golub-Welsch with a Newton refinement on top.

That changes what the choice is between, and the measurements then decide it in one
direction.

## The measurements

Against a `__float128` reference (about 18 spare digits), and against `leggauss` itself.

| | node error vs reference | node difference vs `leggauss` | cost at n = 512 |
|---|---|---|---|
| Eigen Golub-Welsch | 5.1e-15 at n = 512 | ~5e-15 | 40 462 us |
| Newton on `P_n` | **5.8e-17**, flat in n | **<= 0.5 eps**, flat in n | **1 919 us** |

Eighty-eight times more accurate on the nodes, twenty-one times faster, in about thirty-five
lines with no dependency and no allocation. The node agreement with `leggauss` was verified
independently over 46 sizes from n = 1 to n = 400, even and odd, and is bit-identical at some
of them.

Two structural reasons behind those numbers, both worth keeping because they generalize.

**Newton converges to the root; an eigensolve inherits its sweep's accumulated backward
error.** So the Newton node error is flat in n at well under one ulp of the node value, while
the QR error grows. numpy gets the same flatness for the same reason, from its single polish
step, which is why the two land so close.

**Eighty-five percent of Eigen's time builds an n by n orthogonal matrix of which one row is
used.** Eigenvalues alone cost 6 409 us at n = 512 against 40 462 us with vectors. Skipping
the tridiagonal reduction saves almost nothing by comparison, 1.37x at n = 256.

## Choosing Golub-Welsch would have cost parity slack for nothing

This is the part that matters for the next port to read. A Newton port lands within about
6e-17 of pantr's current nodes; an Eigen Golub-Welsch port lands about 5e-15 away. Eighty-five
times more slack in the parity bound, bought in exchange for being slower and less accurate.
**The algorithm that is closest to the oracle is also the best one on its own merits here**,
and that coincidence is not general: it happens because the oracle already contains a Newton
step.

## The weights are a separate problem, and it is not the algorithm's

Measured, and it is the finding that shaped the bound. The relative disagreement in the
weights against `leggauss` grows like `n^2.7`, reaching 2.5e6 units of roundoff at n = 400,
while the absolute disagreement stays around 25 to 120 units of roundoff. The smallest
weight, at the endpoint, decays like `n^-1.96`, and the largest absolute error sits exactly
there.

Three candidate causes were tested and two are refuted:

- **Not the eigenvector weights.** Substituting the classical `2/((1-x^2) P'^2)` formula for
  the eigenvector first components changes the error by no meaningful factor.
- **Not the nodes, and not the root-finder.** Recomputing the weights **from numpy's own
  nodes** reproduces the discrepancy exactly, to the last digit: 1.44, 27.75 and 23.07 units
  of roundoff at n = 16, 128 and 512, the same numbers the full Newton port gives.
- **It is the weight formula's own rounding.** numpy evaluates `1/(fm*df)` with a
  max-normalization, then symmetrizes, then rescales the sum to 2. The port evaluates
  `2/((1-x^2) P'^2)`. Two algebraically equal expressions, evaluated differently, differing
  by a few units of roundoff in absolute terms, which at a weight of size `n^-2` is enormous
  in relative terms.

So the disagreement is **irreducible** without transliterating numpy's expression sequence
verbatim, which would trade accuracy for mimicry.

**That "less accurate" was an assertion when it was written, and it is now measured, by three
orders of magnitude.** A later pass compared both against `mpmath` at 50 digits, which is the
one check the parity apparatus structurally cannot make: two implementations of the same
recurrence agree perfectly while both being wrong, and the C++ is a deliberate transliteration.
Maximum absolute weight error, in units of roundoff:

| n | this port | `leggauss` |
|---|---|---|
| 16 | 1.06 | 3.93 |
| 128 | 1.37 | 56.9 |
| 1000 | 0.62 | 557 |
| 2048 | 1.12 | 1005 |

**This port's weight error is flat in `n` and at the arithmetic floor**; numpy's grows. Decomposed
against truth, the two terms are the exact formula evaluated at the shipped float64 node (0.24 to
1.25 u) and the float64 evaluation of the formula (0.40 to 1.14 u), both about one unit of
roundoff. Nothing is being lost anywhere, and no better result is available from float64 nodes.
The nodes themselves come out at 0.30 to 0.90 u against the exact roots with no trend to n = 2048,
so "flat in `n`" survives as a measurement against truth and not only against numpy.

So the honest statement is not that the port's weights are "a small improvement". They are near
the floor while numpy's are up to 900x off it, and the earlier framing of the discrepancy as two
expressions differing understated which side the difference lives on.

**One consequence for the suite, and it is uncomfortable.**
`tests/test_quad.py::test_agrees_with_an_independent_implementation` is the only oracle in the
tree that is not this code, and its bounds are set by *numpy's* error rather than by the port's.
At n = 1000 the measured difference is 279 u against a 1024 u bound, of which the port contributes
0.3 u. **The port's weights could degrade by roughly 500x before that test fires.** It is checking
that the two agree, which is what it says, but a reader could take it for a check that the port is
accurate, and it is nothing like tight enough for that.

**Also derivable, and now derived.** The `n^2.7` relative growth was a fit. At a root the Legendre
equation gives `dw/w = 6x dx/(1 - x^2)`, and with `1 - x_max^2 ~ 5.78/n^2` and `dx ~ u/2` the
predicted relative error is `3 n^2 u / 5.78`, which is 4.8e-10 at n = 2048 against 7.0e-11
measured, an upper bound behaving as one should. That turns `design/backend_parity.md`'s "an
absolute bound is the only defensible shape" from a measurement into a derivation.

## What follows for the bound, stated here and derived elsewhere

A **relative** per-element bound on the weights is refuted by measurement, at n = 400 by a
factor of 6e3. An absolute one is the only defensible shape. The same reasoning applies to
any rule whose weights vanish at the ends, which includes tanh-sinh, so it is stated in
`design/backend_parity.md` as a rule about weights rather than a fact about Gauss-Legendre.

## Lambert W, for the tanh-sinh step size

`scipy.special.lambertw` is replaced by Halley on `w e^w = x`. The argument range is bounded
by construction, `x = 0.6*pi*(n-1)`, so `x` lies in about `[1.9, 1500]` for `n` in
`[2, 800]`, entirely on the principal branch.

**A branch-free asymptotic start and a fixed iteration count, with no convergence test.** The
fixed count is not a convenience: a residual test is a comparison on the scalar, which is
Tier B under `design/automatic_differentiation.md`, and it would put data-dependent control
flow inside a kernel. The count is **four**, and there are two arguments for it. The second is
the operative one.

**The convergence chain. PROVED, with one SUPPORTED constant.** Write `f(w) = w e^w - x`, `w*`
for its root, `e_k = w_k - w*` the **absolute** error, and `u` for the unit roundoff. Halley on
`f` is Newton on `g = f / sqrt(f')`, and `g'' = f * Phi` with
`Phi = e^{-w/2} (1+w)^{-5/2} (w^2 + 4w + 6) / 4`. Because `g''` carries `f` as an explicit
factor, `g''(w*) = 0` holds structurally rather than by cancellation, so applying Lagrange's
remainder twice gives

    |e_1| <= |e_0|^3 * sup_I |g'''| / (2 |g'(w_0)|)   =:  M |e_0|^3

**Hypotheses.** (H1) `f' > 0` on the interval `I` containing `w_0` and `w*`. Required; it is what
makes `g` well defined and the Newton step finite. It holds because `x >= 1.885` forces
`I` inside `[0.37, infinity)`. (H2) `M <= 0.6886`, obtained from a grid over `I` rather than in
closed form: **SUPPORTED, not proved**, and it is the one gap in this argument. (H3) Real
arithmetic. The chain is stated in exact arithmetic and compared against `u` only at the end;
the floating-point evaluation of one Halley step is a separate matter and is not bounded here.

With `e_0 = 0.455`, measured as the largest absolute initial error over the argument range, the
chain reads `0.455, 6.5e-2, 1.1e-4, 4.9e-13, 4.5e-38`. **Three steps miss `u` and four clear
it**, and the count is insensitive to `M` anywhere in `[0.28, 1]`, so the gap in (H2) does not
reach the conclusion.

**The frame, which is where an earlier version of this note was wrong.** `0.455` is an
**absolute** error and `0.551` is a **relative** one, and they are equal only at the smallest
`w*`. A chain written in the relative frame needs the relative constant `K(w*) * w*^2`, which is
not bounded by 1: it is 0.170 at `x = 1.885` and reaches **53.9** at `x = 1e12`. So the earlier
claim that the chain closes "with unit constant" was true in one frame and false in the other,
and it reached the right count only because the worst initial error and the worst constant sit
at opposite ends of the range. A check of the form "is `|e_1| <= K |e_0|^3`?" passes at
`w* = 5, e = 2.76` while one step reduces the error by a factor of only 2.4 and the four-step
chain has stopped closing. **REFUTED**, and replaced by the absolute chain above.

**The validity threshold, which is the argument that actually decides the count. SUPPORTED.**
The branch-free start is `w_0 = L1 - L2 + L2/L1` with `L1 = log(x)`, `L2 = log(L1)`, and it is
only usable while `x` is large enough to keep `w_0` on the principal branch. Bisecting on the
decay factor with everything else held fixed, at `n = 2`, which is the binding case because `x`
is smallest there: **three** steps reach one unit of roundoff down to a decay factor of 0.5932,
and **four** down to 0.5097. The shipped value is 0.6. So three iterations sit **1.1%** away
from failing and four sit **18%** away. That margin, not the chain, is why the count is four. The extra step costs one exponential and one logarithm **per rule**,
not per node.

**A coupling neither constant records, and it is required.** `_TANH_SINH_DECAY_FACTOR` sets the
smallest argument this kernel ever sees, `x_min = 0.6 * pi * (n - 1)` at `n = 2`. Measured,
varying the factor with everything else fixed: at 0.50 the result is 2.8e4 units of roundoff
after four steps, at 0.40 it is 2.4e16, at 0.32 it is not a number. Neither definition mentions
the other today, so a change to the decay factor for better discretization error would produce a
silently wrong step size with no error signal.

**The failure mechanism stated above and in two source files was wrong, and the correct one is
sharper. REFUTED, with the boundary measured.** Both said the start "lands negative, off the
principal branch, and no number of Halley steps recovers". Negative is not the boundary: the
start goes negative at `x = 1.7105` while four steps still converge below that, and Halley
recovers from a negative start down to about `w_0 = -0.38`. **The true no-recovery boundary is
`w_0 = -1`, the branch point of `W`** where `W_0` and `W_{-1}` meet, which the start crosses at
`x = 1.488218`. Measured against it: a twelve-step cliff sits at `x = 1.48823`, agreeing to five
digits. So the requirement is that the branch-free start land *in the basin*, above the branch
point; the fixed count of four then demands headroom above that, which is what the 1.61 guard in
`cpp/bindings/quad.cpp` buys, clearing the four-step cliff at 1.60292 by 0.44%. The
`log(log(x))` threshold is exactly `x > 1`, i.e. a decay factor of `1/pi = 0.3183`, not "about
0.318" by measurement.

## Where `_TANH_SINH_DECAY_FACTOR = 0.6` comes from, recorded now that it is known

This note used to call the constant's provenance unrecorded. It is derivable, and the derivation
names it.

The rule samples `t = (i+1)h` for odd `n` and `(i+1/2)h` for even, so `t_max = h(n-1)/2` exactly.
Equating the classical double-exponential discretization error `exp(-2 pi d / h)` with the
truncation error `exp(-(pi/2) e^{t_max})` gives `t_max e^{t_max} = 2d(n-1)`, that is
`t_max = W(2d(n-1))`. The kernel's own `decay_arg` is `2 c (pi/2) (n-1)`, which is `2d(n-1)`
with

> **`d = c pi / 2`. The decay factor is the assumed half-width of the strip of analyticity of the
> transformed integrand, as a fraction of the map's own limit `pi/2`. `0.6` means `d = 0.3 pi`.**

Confirmed two ways. The balance ratio `(2 pi d/h) / ((pi/2) e^{t_max})` comes out at exactly
`n/(n-1)`: 1.06667 at n = 16, 1.003922 at n = 256. And for `f = 1/(a^2 + x^2)`, whose transformed
singularity sits at `t* = i asin(atan(a)/(pi/2))`, the predicted optimum `c = (2/pi) Im t*`
matches the measured argmin at every pole distance tried (0.640 predicted against 0.600 measured
at `a = 4`, 0.0997 against 0.100 at `a = 0.25`).

**Verdict on the value: near-optimal for what the rule advertises**, with a penalty of at most 3x
against the per-case optimum on endpoint singularities and on integrands analytic in a wide
region. It is **poor for a pole within `|Im x| <~ 1`**, where the penalty reaches 1.6e4, and it is
more conservative than the literature default `d = pi/2` (c = 1) while beating it on every smooth
case tested. The floor `c >~ 0.51` imposed by the four-step Halley solver means the constant
**cannot be lowered** to suit a near-pole integrand without also touching the step count. That
coupling was recorded above in one direction only.

**One inconsistency, measured and deliberately not changed.** The `W` argument uses `(n-1)`,
correct for `t_max = h(n-1)/2`, while the divisor is `n`. The consistent balance would be
`h = 2W(2d(n-1))/(n-1)`. So the realized decay factor is `n`-dependent and only approaches the
nominal 0.6 asymptotically: 0.317 at n = 4, 0.559 at n = 64, 0.587 at n = 256. Head to head the
shipped `n` divisor is *better* on every smooth integrand at every `n` tested and mildly worse on
the singular ones at n <= 48, so it stays. But 0.6 is a nominal value, and nothing said so.

## The three rules that are not ported, and why that is a decision

`get_gauss_lobatto_legendre_1d`, `get_chebyshev_gauss_1st_kind_1d` and
`get_chebyshev_gauss_2nd_kind_1d` stay on `numpy.polynomial`, so
`from numpy.polynomial import chebyshev, legendre` survives in the module after this port.
`scipy` does not: it had exactly one use, the Lambert W above.

The two Chebyshev rules are closed-form trigonometry and would be nearly free to port. Gauss-
Lobatto would not: it takes the roots of `P'_N` with **no** Newton polish, and its error
against a polished version grows like `N`, reaching about 101 units of roundoff at N = 257,
which is the opposite of the Gauss-Legendre nodes' flat behavior and for exactly the reason
given above. Porting it means a second conditioning derivation, not a second transliteration.

## Epistemic status

- **Verified by reading the source:** that `numpy.polynomial.legendre.leggauss` uses
  `legcompanion` plus `eigvalsh` plus one Newton polish plus the classical weight formula,
  symmetrization and a rescale; that `legcompanion` is exactly symmetric.
- **Measured 2026-08-20, independently of the survey that first reported it:** the 0.5 eps
  node agreement over 46 sizes; the `n^2.7` relative and roughly flat absolute weight
  disagreement; that recomputing weights from numpy's own nodes reproduces it exactly; the
  Halley initial error of 0.551 and the 1 ulp result after four iterations over `x` up to
  1e12; that substituting `leggauss` for the Newton implementation leaves the entire test
  suite green, 6138 tests.
- **Measured by the Eigen survey and not reproduced here:** the `__float128` reference
  errors, the timings, and the 85% eigenvector-accumulation share. The conclusions they
  support were each cross-checked against a second, independent measurement.
- **Asserted:** that numpy's weight formula is the less accurate of the two. The survey
  measured it against a high-precision reference; it was not re-derived.
- **Not investigated:** whether the Newton implementation remains well conditioned far beyond
  n = 1000. The sweep stops there, and the initial guess is an asymptotic form whose quality
  improves with n, so the expectation is that it does, which is an expectation and not a
  measurement.

## Open questions

1. Should the two Chebyshev rules be ported after all, given they are closed-form and nearly
   free? The argument against is that `get_modified_chebyshev_nodes_1d`, which **is** ported,
   already answers the only question they would: whether the C++ and numpy trigonometric
   functions agree. They do for `cos`, exactly, on every argument tested.
2. Gauss-Lobatto's missing Newton polish is a small, contained accuracy improvement to the
   **Python** implementation, independent of any port. Worth doing on its own merits, or is a
   101-unit-of-roundoff node error at N = 257 within what its callers need?
3. What is the largest `n` any caller actually asks for? Every scaling statement here is
   swept to n = 1000, and `n_pts` has no upper bound in the API.
