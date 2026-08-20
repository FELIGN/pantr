# What crosses the backend boundary, and what does not

**Status:** decided. Governs every module ported after the infrastructure PR.
**Date:** 2026-08-20.
**Scope:** the contract between the Python and C++ backends: which values may cross, what
shape a kernel entry point takes, and who owns a type. Not how the two are compared, which
is `design/backend_parity.md`.
**Companions:** `design/user_functions_across_the_boundary.md`, whose broadcasting contract
this note's rule generalizes; `design/automatic_differentiation.md`, for the scalar tiering
a kernel signature has to respect.

**Validated against:** pantr **0.7.0**, and the C++ prototype as merged into `proto/cpp`
at `765d9b9`. Line numbers below refer to that tree.

## The decision

**Types are owned by Python. Only arrays and scalars cross the boundary.**

There is no `pantr::PointsLattice`, no `pantr::QuadratureRule`, and no C++ counterpart of
any other pantr class. A binding takes the arrays a Python object already holds and hands
back arrays; the Python object is constructed, validated and consumed on the Python side,
by one implementation, whichever backend computed the numbers inside it.

## Why the question had to be settled before it was cheap to get wrong

The dual dispatch was designed for **functions**. `PANTR_BACKEND` selects which
implementation of a computation runs, and the two implementations agree on their arguments
because those arguments are arrays. Nothing in that design says what happens when a
*type* is ported, and the failure it invites is specific: two backends holding two classes
that represent the same thing, and a value produced by one being passed to the other.

`PointsLattice` is the cheapest possible place to fix that, which is why it was fixed here
rather than later. It is a container of one 1D coordinate array per direction and nothing
else. The expensive place would have been an axis-aligned box, with equality and hashing
semantics that two implementations would have to agree on exactly.

## What the code says, and it argues for the rule rather than against it

Read across the package, `PointsLattice` turns out to be a poor candidate for a port on its
own merits, before any cross-backend argument is made.

- **It is transient.** It is constructed, consumed and discarded inside one call. No class
  in the package stores one as an attribute.
- **Its only method with work in it is one the design already rejects.**
  `get_all_points` (`src/pantr/quad.py:551`) is a `meshgrid`, and
  `design/user_functions_across_the_boundary.md` establishes that materializing it is what
  the port must avoid: about 3.2 GB at `512^3` against 12 kB for the three 1D arrays, a
  factor of 250 000. Porting the class would mean porting that method, which is the one
  operation nobody should be calling on large input.
- **It carries no identity.** No `__eq__`, no `__hash__`, no `__repr__`, no pickle support
  beyond the default. So there is no semantics for two implementations to disagree about,
  which is exactly what makes it a cheap decision and a poor test of the hard case.
- **It is dispatched on, at about twenty sites**, by `isinstance` across `basis`, `bezier`,
  `bspline` and `mpi`. Two classes would mean every one of those sites silently accepting
  one and rejecting the other.

## The rule in the form a kernel author needs

A kernel entry point takes and returns only these:

| crosses | does not cross |
|---|---|
| contiguous arrays of `float32` or `float64` | any pantr class |
| integer sizes, counts and degrees | Python callables |
| an integer status or count as a return value | `dtype` objects, enums as strings |
| an `IntEnum` value, as an integer | anything with an invariant to maintain |

Validation, allocation, dtype normalization and the construction of any object stay in
Python, above the seam, exactly where `CLAUDE.md` already puts Layer 2.

### `dtype` is an output format, not a computation precision

Measured across all seven of `quad`'s rules: for six of them the `dtype` argument selects
the dtype of the returned arrays and nothing else, because the rule is computed in
`float64` and cast once at the end by `_scale_and_cast_nodes_and_weights`
(`src/pantr/quad.py:47-48`).

This is a contract and not an implementation detail, because a C++ kernel templated on the
scalar type gets it wrong silently. Instantiating a Newton Gauss-Legendre solver at `float`
gives a measured relative weight error of **1.46e-3 at n = 200**, since
`w = 2/((1-x^2) P'^2)` cancels as `x` approaches the endpoints. The nodes survive, because
Newton is self-correcting; the weights do not. Python is correct to 0.5 ulp for the same
input, purely because it never computes in `float32`.

Two exceptions, both measured, both of which a port must preserve rather than tidy:

- `get_modified_chebyshev_nodes_1d` **is** computed wholly in the requested dtype, and
  differs from a cast-from-`float64` result by 6.5e-8, about one `float32` ulp over
  the upper half of `[0, 1]`.
- For `get_tanh_sinh_1d`, `dtype` sets the truncation point and therefore the **length** of
  the result: 190 nodes in `float32` against 252 in `float64` for `n_pts = 400`. The data
  is `float64` either way.

So the port's rule is: **compute in `double`; template only the output store type, or do
not template at all and cast in the adapter.** A rule generator has no differentiable input,
so `Real` genericity buys nothing here that `design/automatic_differentiation.md` wants, and
costs a `value_of` on every comparison plus the trap above.

### A kernel whose output length is not a function of its inputs returns that length

Most kernels write into a caller-sized buffer and return nothing, as the infrastructure PR's
one kernel does. A double-exponential rule cannot: it stops where the endpoint gap stops
being representable, so the count is discovered during the computation
(`src/pantr/quad.py:380,420` allocates the worst case and returns `count`).

The seam for that shape is

```
count = kernel(n, min_gap, out_nodes[n], out_weights[n])
```

with the count as an **integer return value**, and the Python side slicing `out[:count]`.
The alternative, an output parameter carrying the count back, would put a mutable
out-parameter in a signature that has to survive a flat C ABI, and it hides in the argument
list the one number the caller cannot proceed without.

### Produce values in the frame they are computed in, not in the frame they are returned in

`_scale_and_cast_nodes_and_weights` maps `[-1, 1]` onto `[0, 1]`. **The C++ side produces
`[-1, 1]` data and that function stays shared, in Python.**

This is a parity decision wearing a layering costume, and it is worth stating because the
opposite looks tidier. If the C++ computed natively on `[0, 1]`, the two backends would run
different arithmetic for the map as well as for the rule, and the map's own conditioning
would have to enter the bound: near `x = -1`, `fl(x + 1)` for `x = -1 + delta` carries an
absolute error of one unit roundoff against a result of size `delta`, which is about `1e-5`
at n = 400. That is the same endpoint-conditioning problem the weights already have,
imported for free into the nodes. Shared, the map is common mode and cancels exactly.

## The one violation of this contract in the tree

`src/pantr/bspline/_bspline_eval.py:223` and `:565` read `pts._pts_per_dir[0]`, reaching
past the public `pts_per_dir` property that the other nineteen consumers use. It is not a
bug and it changes no behavior. It is recorded because this note declares the internals of a
Python-owned type to be nobody's contract, and a site reaching into them contradicts that in
the one place a reader would look for permission.

## What this note deliberately does not settle

Whether a **certificate-bearing** result should be a type. The rule above is stated for
containers of numbers, and every type in the port's scope so far is one. A result carrying a
tier, a provenance and a status (`design/root_finding_dimension.md`) is a different case: it
has an invariant, so "only arrays cross" would mean unpacking and reassembling it across the
boundary, and the reassembly is where an invariant gets dropped. That case should be decided
when the first such operation is ported, with this note's rule as the default rather than as
the answer.

## Epistemic status

- **Verified by reading the code:** that `PointsLattice` exposes `dim`, `dtype`,
  `pts_per_dir` and `get_all_points` and nothing else public (`src/pantr/quad.py:478-576`);
  that it defines no `__eq__`, `__hash__` or `__repr__`; that no class in the package stores
  one; that it is dispatched on by `isinstance` at about twenty sites; that
  `_bspline_eval.py:223` and `:565` read the private attribute.
- **Measured 2026-08-20:** that six of seven rules ignore `dtype` except as an output
  format, with `get_modified_chebyshev_nodes_1d` the exception and `get_tanh_sinh_1d` a
  third case where `dtype` decides the length; that a `float32` Newton Gauss-Legendre gives
  1.46e-3 relative weight error at n = 200 while the Python path gives 0.5 ulp.
- **Derived:** the endpoint conditioning of `fl(x + 1)` that argues for keeping the
  `[-1, 1]` frame, and the 3.2 GB against 12 kB figures, which are restated from
  `design/user_functions_across_the_boundary.md` rather than re-derived here.
- **Asserted, not measured:** that two backends holding two classes for one concept is the
  failure worth designing against. No such pair was ever built here, so this is reasoning
  about a shape rather than a report of one going wrong.
- **Not investigated:** whether the downstream consumer that imports pantr's private symbols
  depends on any of `quad`'s five private names. `_generate_tanh_sinh` is the plausible one:
  it is the only one returning something the public API cannot give, namely the rule on
  `[-1, 1]` in `float64` together with the effective count.

## Open questions

1. Does the certificate-bearing case above want a different rule, or does unpacking and
   reassembling across the boundary turn out to be acceptable when the reassembly is a
   single function that the invariant is tested on?
2. Should `_bspline_eval.py`'s two private reads be normalized to the public property? Two
   lines, no behavior change, but they sit in files no current port touches, and the rule
   for a bug met in passing is that a fix belongs inside files the work already opened.
3. `PointsLattice` and `QuadratureRule` both pickle today, and `mpi` constructs them for
   collective calls. If either class moves module, `__module__` has to be rebound or an
   older pantr cannot unpickle what a newer one sends. That is a packaging consequence of
   this note's "Python owns the type" rule rather than of anything in it, but it is the kind
   of thing that surfaces as an MPI failure far from its cause.
