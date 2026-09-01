# What a pickled B-spline space loses, and the bound on it

**Status:** accepted, 2026-09-01. Written for ticket #396, slice 3.
**Scope:** why `BsplineSpace1D.__reduce__` reconstructs from the constructor's
arguments, what that costs, and the bound on the one quantity it does not restore
exactly. Not what an accessor hands back, which is
`design/bspline_ownership_lifetime.md`. Not where derived quantities live, which is
`design/bspline_derived_caches.md`.

**Validated against:** `proto/cpp` at `7ba20d3` plus `feat/396-bspline-space-1d-bind`.

## The decision

`__reduce__` names the constructor's arguments -- the knots, the degree, the
periodicity flag, and `snap_knots=False` -- and nothing else. It does **not** carry
the implementation handle, and it does not carry any derived quantity.

The handle is the easy half: a C++ handle is not picklable, and admitting one would
make a pickle written under `PANTR_BACKEND=cpp` unreadable under the default
backend, so the backend switch would silently become a data-format switch. The rule
is the one commit `11f22a7` established for the affine map.

Two smaller choices come with it. The knots are **copied out of the read-only
view**, because pickle preserves the writeable flag and the oracle's constructor
expects to own a writable array. And snapping is **off** on the way back in: the
stored vector is already snapped, so re-snapping is a no-op by idempotence, and
skipping it makes the reconstruction independent of a scan.

## The consequence: the tolerance is recomputed, not restored

The tolerance is a *derived* quantity, so carrying it would put something in the
wire format that is not a constructor argument. It is therefore recomputed from the
stored knots -- and the stored knots are the **snapped** vector, whereas the
original's tolerance was computed from the vector **as supplied**. Where snapping
moved the last knot, the scale moved with it, and the two tolerances differ.

### The bound

> `|tol' - tol| / tol  <=  (m - 1) * 8 * eps`

where `m` is the multiplicity of the **last** knot class.

**Derivation.** Snapping replaces each knot class by its *first* knot. So the first
knot of the vector never moves, and only the last one does. Every step inside a
class is at most one tolerance, so a class of `m` knots spans at most `m - 1` of
them, and the last knot moves by at most `(m - 1) * tol`. The scale is
`max(hi - lo, |lo|, |hi|)`, a maximum over three arguments and therefore
1-Lipschitz in each; only `hi` moved, so the scale moves by no more than the same
amount. The tolerance is `8 * eps` times the scale, a fixed multiple, so the
relative change in the tolerance is the relative change in the scale. Recomputing
the scale and the tolerance introduces one rounding each, adding a further relative
`eps` or so, which the measured margin absorbs.

### The `m - 1` is not decoration

A first version of this argument claimed a flat `8 * eps`, reasoning that the scale
moves by at most one tolerance. **That ignores chaining**: knots grouped by gap
form a class as wide as the chain, not as wide as the threshold. A sweep built to
chain the final class **exceeded the flat bound by a factor of 4.4**.

### What was measured

`tests/parity/test_bspline_space_1d.py::_tolerance_drift_sweep`, called at 20000
draws per seed rather than the tenth of that the suite ships:

| seed | 11 | 101 | 202 | 303 | 404 | 505 | 606 |
|---|---|---|---|---|---|---|---|
| worst / `(m-1) * 8 * eps` | 0.716 | 0.755 | 0.729 | 0.734 | 0.732 | 0.734 | 0.733 |
| worst / flat `8 * eps` | 4.403 | 4.531 | 4.480 | 4.403 | 4.400 | 4.403 | 4.608 |

Every draw drifted, so the bound is compared against something rather than against
zero. The margin is stable at roughly three quarters of the bound across 140000
draws and is not approaching it. Identical to three figures under both backends,
which is itself a parity result: the drift is a property of the arithmetic rather
than of the implementation.

To reproduce:

```
python -c "from pantr._backend import Backend, use_backend; \
           import tests.parity.test_bspline_space_1d as t; \
           print(t._tolerance_drift_sweep(20000, seed=11))"
```

### Why this is acceptable rather than a defect

The tolerance decides whether two knots are the same knot. A relative change of
order `m * eps` in the threshold can only flip that verdict for a pair whose
difference lies within the same relative distance of the threshold -- the hairline
band `pantr/bspline/knots.hpp` already documents for the two knot-differencing
arithmetics. A space whose classification depends on that band is one whose mesh is
at the format's noise floor, and the constructor refuses the cases where that has
consequences.

**What would change this decision:** a caller for whom a round trip must be
bit-identical in every field. The fix would then be to carry the tolerance in the
reduction and validate it on reconstruction, which trades the "constructor
arguments and nothing else" property for exactness. Nobody has asked.

## What the tests pin

`test_the_reduce_tolerance_drift_stays_inside_its_bound` asserts three things, and
the second and third are what make it a check rather than a formality:

1. the bound holds;
2. the drift is **non-zero** on the sweep, because a bound only ever compared
   against zero has not been checked;
3. the **flat** `8 * eps` form is still exceeded, so dropping the `m - 1` again
   cannot go unnoticed.
