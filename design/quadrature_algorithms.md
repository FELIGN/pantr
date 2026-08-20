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
verbatim, which would trade accuracy for mimicry: the survey found numpy's weight formula to
be the *less* accurate of the two. The port keeps the better formula and states the
consequence, which is that its weights are a small improvement on the Python's rather than a
copy of them.

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
from failing and four sit **18%** away, in a constant whose provenance is unrecorded (see
below). That margin, not the chain, is why the count is four. The extra step costs one exponential and one logarithm **per rule**,
not per node.

**A coupling neither constant records, and it is required.** `_TANH_SINH_DECAY_FACTOR` sets the
smallest argument this kernel ever sees, `x_min = 0.6 * pi * (n - 1)` at `n = 2`. Below a decay
factor of about 0.51 the start lands negative, off the principal branch, and **no number of
Halley steps recovers**; below about 0.318, `L2 = log(log(x))` is not real. Measured, varying
the factor with everything else fixed: at 0.50 the result is 2.8e4 units of roundoff after four
steps, at 0.40 it is 2.4e16, at 0.32 it is not a number. Neither definition mentions the other
today, so a change to the decay factor for better discretization error would produce a silently
wrong step size with no error signal.

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
