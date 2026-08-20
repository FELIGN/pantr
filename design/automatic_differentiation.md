# Automatic differentiation: what actually needs it

**Status:** design note for the C++ port. Nothing here is implemented.
**Date:** 2026-08-18.
**Scope:** whether pantr needs AD, for which optimization workflows, and whether templating
the scalar type is enough to keep the option open.
**Conclusion up front:** **no flavor of the intended work requires AD inside the library.**
Templating on the scalar type is enough to keep the door open, and it is worth doing anyway
for a different reason.

**Depends on:** the AD tiering (Tier A generic in the scalar, Tier B value-only, Tier C
analytic derivative rule instead of differentiating an iteration), and the decision that no
tape-based reverse-mode type ever enters the library.

**Validated against:** pantr **0.7.0** (`main`, tag `v0.7.0`), 2026-08-19. Line numbers
below refer to that tree.

## Four flavors, and they need different things

The motivating application is finite-element assembly, with shape and topology optimization
on top. Those are not one workload.

| flavor | design variables | count | is pantr differentiated? |
|---|---|---|---|
| density-based topology optimization (SIMP) | element densities | 10^4 to 10^7 | **no, not at all** |
| topology optimization by a spline level set | level-set coefficients | 10^3 to 10^6 | yes, but **linearly** |
| shape optimization on control points | boundary control points | 10^2 to 10^4 | yes, but **linearly** |
| shape optimization on CAD parameters | angles, radii, lengths | 10 to 10^2 | yes, and **nonlinearly** |

Only the last row wants AD, and it is the row with the fewest variables.

## Where the geometry enters the stiffness matrix at all

For `K_ij = ∫ B_i^T D B_j det(J) dξ` with `∇_x N = J^{-T} ∇_ξ N` and `J = ∂F/∂ξ`, the design
parameters enter **only through `J` and `det J`**. Since `F` is linear in the control points:

- `∂J/∂P` is the derivative of the basis functions. Analytic, and already computed.
- `∂(det J)/∂J` is the cofactor matrix. Analytic.
- `∂(J^{-T})/∂J` is analytic and tedious.
- `dP/dθ` is the only piece whose difficulty depends on the flavor.

So the question "does pantr need AD" reduces entirely to "how hard is `dP/dθ`".

## Density-based topology optimization: the geometry is fixed

`K(ρ) = Σ_e ρ_e^p K_e`, with the mesh and the geometry map unchanged. `dK/dρ_e = p ρ_e^(p-1) K_e`.

pantr contributes `K_e`, computed once per element, and is never differentiated. Ten million
design variables and zero geometry derivatives. This flavor is out of scope for AD entirely,
which is worth stating because "topology optimization" usually means this one.

## Spline level-set topology optimization: linearity plus the shape derivative

This is the flavor of interest, and it needs no AD.

Let `φ(x) = Σ_j c_j N_j(x)` with the coefficients `c_j` as design variables, and
`Ω = {φ < 0}`. Two facts do all the work:

**φ is linear in `c`.** So `∂φ/∂c_j = N_j(x)`, which pantr already tabulates.

**The shape derivative turns the whole thing into a boundary integral.** For `J(Ω) = ∫_Ω f dx`
the classical result is `dJ = ∫_∂Ω f V_n ds` with `V_n` the outward normal velocity.
Perturbing `φ → φ + ε δφ` moves the zero level set: the displacement `δx` satisfies
`δx · ∇φ = −ε δφ`, and with `n = ∇φ / |∇φ|` the normal displacement is
`δx · n = −ε δφ / |∇φ|`. Taking `δφ = N_j` gives `V_n,j = −N_j / |∇φ|`, hence

```
dJ/dc_j  =  − ∫_∂Ω  f · N_j / |∇φ|  ds
```

and for a compliance-type objective the same expression with `f` replaced by the
adjoint-weighted energy density.

**Why this beats AD, concretely.** The right-hand side is a boundary integral of the basis
functions against a scalar field, which is an **assembly**: one pass over the interface
quadrature, scattering into the `j` index. Its cost is `O(interface)`, and the interface only
touches the cut cells, which are on the order of 1% of the total. Reverse-mode AD over the
whole construction would need a tape across every cell to compute the same numbers. The
analytic route is cheaper, exact, and adds no dependency.

**The hypothesis it needs, stated because it can fail.** `|∇φ| ≠ 0` on the interface. This is
the standard non-degeneracy assumption for a level set, and it fails where the level set has a
critical point on the zero set. Any implementation should detect and report that rather than
divide by a small number.

## Shape optimization on control points: the same argument

`F` is linear in `P`, so `∂F/∂P` is the basis functions and `∂J/∂P` their derivatives. Both
already exist. Forward and reverse are the same object transposed, so the count of design
variables does not force taping. No AD.

## CAD parameters: the only flavor that wants AD, and a hazard in it

Here `θ` are the parameters of a construction, and `dP/dθ` means differentiating pantr's own
`cad/` layer. Some of those operations are linear in their parameters (extrusion displacement,
ruled, Coons, join) and hand-differentiating them is trivial. Others are not.

**The hazard, verified:** `src/pantr/cad/_primitives.py:94`, `create_circle`, builds the exact
conic representation with rational quadratic B-splines, and its docstring states that *the
number of spans depends on the sweep angle*. So the angle decides the **knot vector and the
number of control points**, not merely their values. `dP/dθ` is not even well defined as a
fixed-size object across a span-count change. `create_revolution`, `create_disk` and
`create_cylinder` all reach this through `create_circle`.

The rule that follows: **a parameter that changes discrete structure is Tier B, value-only.**
The angle decides the span count from its value; the coefficients computed from it can be
scalar-generic. So the constructor is **mixed**, and AD through it is valid only within a fixed
span count, away from the transition angles.

This is the same distinction as the knot-span search, appearing somewhere new. It is exactly
the kind of thing that produces silently wrong derivatives if it is not written down.

## AD through cut-cell quadrature is valid, correcting an earlier claim

An earlier version of this reasoning held that differentiating through a cut-cell quadrature
construction was unsound, because the classification is piecewise constant. That was wrong,
and the correction matters.

**Generically the classification is locally constant in `θ`**, so it contributes nothing to the
derivative. What does vary, and varies smoothly, is the position and weight of each quadrature
point inside a cut cell, because the interface moves continuously.

And the result is better than merely valid. If the discrete functional is
`J = Σ w_i(θ) g(x_i(θ))`, then AD returns **the exact derivative of that**, not an approximation
of something else. The shape-derivative formula differentiates the *exact* integral; AD
differentiates the *discrete functional that is actually minimized*. For optimization the
latter is often preferable, because it makes the discrete problem self-consistent: the gradient
is the gradient of what the line search evaluates.

Two conditions attach.

**The 1D root finding inside needs an analytic rule, not AD.** It is iterative, so
differentiating it is the mistake of differentiating the iteration rather than the fixed point.
The implicit function theorem gives it directly: if `φ(x*(θ); θ) = 0` then
`dx*/dθ = −(∂φ/∂θ) / (∂φ/∂x)`, one division. Roots by rule, everything downstream of the roots
by AD.

**The non-generic case is a kink, not an error.** At the `θ` values where a cell changes
classification the discrete functional has a corner. That set has measure zero, but optimization
iterates walk through it, so occasional gradient discontinuities will be observed. This is a
line-search concern, not a correctness one, and it should be documented so it is not
misdiagnosed as a bug in the derivative.

## So: is AD needed, and are templates enough?

**Needed: no.** Three of the four flavors are served by linearity plus an analytic rule, and
the fourth has few enough parameters that hand-derived formulas are viable. AD is a
convenience for one flavor, not a requirement of any.

**Templates enough: yes**, and precisely because AD is not needed now. Keeping the door open
costs four disciplines, all of which are cheap up front and expensive to retrofit:

1. Comparisons and branches on the scalar (span search, convergence tests) go through
   `value_of`, never through the scalar directly.
2. No `std::floor` and no integer casts on the scalar.
3. Unqualified calls with a using-declaration (`using std::sqrt; sqrt(x)`), never
   `std::sqrt(x)`, which hard-blocks any AD type.
4. Parameters that change discrete structure are value-only.

With those four respected, introducing a forward-mode `Dual<T, N>` later is mechanical. `N`
derivative components rather than one, so a pass with `N` CAD parameters amortizes the value
computation instead of costing `N` passes; it stays trivially copyable, tape-free and
memcpy-safe.

## The motivation is precision and SIMD, not AD

Templating is not free: compile times, the tiering discipline, and a `value_of` indirection
throughout. If AD is probably not needed, the honest justification has to come from elsewhere,
and it does.

**`float32` halves memory and doubles the SIMD lane count.** For the large-image fitting case
that is a concrete, scale-relevant benefit, not a hypothetical one. AD compatibility then rides
along for free, because the discipline that admits `Dual` is the same discipline that admits
`float`.

This changes which part is load-bearing, and that matters under pressure. If compile times
become painful, `float32` support is kept and the AD discipline could be relaxed. The reverse
would be the wrong trade. Knowing which of the two justifies the design is what prevents
sacrificing the wrong one.

## Epistemic status

- **Verified by reading the code:** that `create_circle` (`src/pantr/cad/_primitives.py:94`)
  makes the span count depend on the sweep angle, and that `create_revolution`, `create_disk`
  and `create_cylinder` reach it; that `cad/` is otherwise constructive.
- **Derived, with the derivation given:** the shape-derivative expression for a spline level
  set, including the `V_n = −δφ/|∇φ|` step and the `|∇φ| ≠ 0` hypothesis it rests on.
- **Standard results used without re-deriving:** `dJ = ∫_∂Ω f V_n ds` for a domain functional,
  and the implicit-function-theorem derivative of a simple root. Both are textbook; neither was
  checked against a source here.
- **Corrected during this analysis:** the claim that AD through cut-cell quadrature is unsound.
  It is valid generically, and gives the derivative of the discrete functional. The correction
  came from the observation that an infinitesimal parameter change does not reclassify cells.
- **Asserted, not measured:** that the boundary-assembly route is cheaper than reverse-mode AD
  for the level-set flavor. The argument is `O(interface)` against `O(volume)` plus a tape, which
  is sound in scaling, but no implementation of either was timed.
- **Not investigated:** how often optimization iterates actually land near a classification
  transition in practice, which is what decides whether the kink is a nuisance or a real
  obstacle for the line search.

## Open questions

1. Are the CAD-parameter derivatives wanted at all, or is the level-set flavor the whole
   requirement? If the latter, AD has no consumer and the four disciplines are pure option
   value.
2. Should `dP/dθ` for the linear constructors (extrusion, ruled, Coons, join) be exposed
   explicitly as operators, in the same way `∂F/∂P` is? They are trivial, and exposing them
   removes the temptation to reach for AD.
3. What is a reasonable `N` for `Dual<T, N>`? Fixed at compile time is fastest and forces a
   recompile per parameter count; runtime-sized loses the trivially-copyable property that
   makes the whole scheme cheap.
4. How should the classification-transition kink be surfaced? A status flag from the quadrature
   construction saying "a cell changed classification relative to the previous call" would let
   an optimizer detect it, but requires the construction to be stateful, which it otherwise is
   not.
