# Porting the `Grid` hierarchy: where the vtable goes, and where it does not

**Status:** proposed, 2026-08-30. Written for ticket #386, and binding on #387 and #395 if
adopted. **Sections marked *(provisional)* were not compiled or run at the time of writing.**
**Date:** 2026-08-30.
**Scope:** the C++ shape of the generic `Grid` layer, how inheritance crosses the backend
seam, which of the base's concrete methods move, and what the Python surface becomes.
Not how the two backends are compared, which is `design/backend_parity.md`.
**Companions:** `design/cross_backend_types.md` (the ownership ruling this obeys),
`design/bvh.md` (the cache this layer owns), `design/backend_parity.md` Rule 8 (why the
`float` axis stays out of the bindings).

**Validated against:** `proto/cpp` at `1fe9849`, worktree `feat/386-grid-hierarchy`. Every
line number below was read in that tree. Python behaviour was measured on the `pantr`
environment's CPython **3.14.6**.

## The decision in one paragraph

The generic layer becomes a **CRTP mixin, `pantr::grid::GridBase<Derived, T>`, holding its own
state and dispatching by name hiding**; there is no vtable, and a C++ grid is a compile-time
type. **Inheritance does not cross the seam at all** -- after the port the Python side has no
implementation inheritance between grids, only a `typing.Protocol` named `Grid` for annotation
and a private ABC named `_GridPython` for the temporary oracle. A Python-defined grid remains
possible and remains a *Python* grid; it can never become a C++ one, and no callback bridge
should be built to pretend otherwise. The wrapper classes are siblings that each forward to a
handle, sharing forwarding code through a private, stateless `_GridWrapper` base.

## Findings against the ticket as written

Three of these are measured refutations, not opinions. They are listed first because they
change what #386 must do.

### F1 (critical). Turning `Grid` into a body-less Protocol breaks a must-not-edit test

`tests/test_grid_hierarchical.py:1721` is

```python
np_testing.assert_array_equal(g.boundary_facets(), Grid.boundary_facets(g))
```

inside `TestBoundaryFacets.test_override_agrees_with_abc_default`. It calls `Grid`'s default
**unbound**, passing a `HierarchicalGrid` as `self`, to check the specialisation against the
generic. `Grid` is imported from `pantr.grid` at `tests/test_grid_hierarchical.py:16`, and
that file is on #386's *must pass without being edited* list.

If `Grid` becomes a Protocol whose members are `...`, `Grid.boundary_facets(g)` returns
`None`, and `numpy.testing.assert_array_equal(array, None)` raises `AssertionError`
(measured). The test fails.

**The repair is available and cheap: a `typing.Protocol` may carry default implementations,
and an unbound call through it works.** Measured on 3.14.6: for a Protocol `G` with a real
`default()` body, `G.default(duck_typed_object)` returns the correct value. So `Grid` keeps
the nineteen concrete bodies verbatim; only the six primitives become `...`.

This has a consequence worth stating plainly, because it is the opposite of what the ticket
implies: **the Python `Grid` does not stop carrying the generic algorithm.** It stops being
*inherited from*. Those are different retirements and only the second one happens here.

### F2 (important). Only one of the two `test_grid_abc.py` tests actually breaks

The ticket says to delete both `:57-60` and `:63-76` because "that string is `ABCMeta`'s; a
Protocol raises a different message". Measured on 3.14.6:

| construct | result |
|---|---|
| `G()` where `G` is a Protocol | `TypeError: Protocols cannot be instantiated` -- does **not** match `"abstract"` |
| explicit subclass of a Protocol omitting a **plain** member | instantiates fine, no error |
| explicit subclass of a Protocol omitting an `@abc.abstractmethod` member | `TypeError: Can't instantiate abstract class _Incomplete without an implementation for abstract method 'prim'` -- **matches `"abstract"`** |

`typing.Protocol`'s metaclass `typing._ProtocolMeta` derives from `abc.ABCMeta` (verified),
so abstract enforcement is still live for anything that *explicitly* inherits.

So `test_grid_is_abstract` (`:57-60`) does break and should go. `test_incomplete_subclass_is_abstract`
(`:63-76`) **survives unchanged** provided the six primitives keep their `@abc.abstractmethod`
decorators -- which they must anyway, on `_GridPython`. Re-point it at `_GridPython` and it
keeps testing exactly what it tested before. One deletion, not two.

### F3 (important). A Protocol without `__slots__ = ()` silently gives every explicit subclass a `__dict__`

Measured: `abc.ABC.__slots__` is `()`, `Protocol.__slots__` is `()`, but `Generic` declares
none. A Protocol that omits `__slots__ = ()` therefore leaks a `__dict__` into every explicit
subclass even when that subclass declares `__slots__ = ()`. `Grid` today is slot-clean
(`src/pantr/grid/_grid.py:71`) and both concrete grids depend on that.

**Write `__slots__ = ()` on the `Grid` Protocol.** With it, an explicit subclass declaring
`__slots__ = ()` has no `__dict__` (measured). This is a one-line requirement that nothing
would have caught: no test asserts the absence of `__dict__`, and the failure is a silent
per-instance memory regression plus the quiet return of settable attributes.

### F4 (important). The Protocol cannot be both the consumer's contract and the implementer's

`Grid` today plays two roles that inheritance fuses and structural typing cannot:

- **what a consumer may rely on** -- all twenty-four public members, because `src/pantr/viz/_grid.py:54`
  annotates `grid: Grid` and `src/pantr/grid/_cell_quadrature.py:31` does too;
- **what an implementer must supply** -- six primitives, the rest being inherited.

A Protocol has no "inherited" half, so it can only express the first. Consequently `Grid` must
list all twenty-four public members, and #386's `AC6` -- "a class that omits **one of the six
primitives**" -- does not test what it says: omitting `cell_aabb` fails identically. Restate
`AC6` as *a class that does not satisfy the `Grid` Protocol is rejected where a `Grid` is
expected*, and keep the six-primitive claim where it is actually enforced, which is
`_GridPython`'s `__abstractmethods__` (F2) and the C++ concept.

### F5 (recommended). The `float` instantiation contradicts a written module decision, and must be scoped

`src/pantr/grid/_grid_backend.py` states that `pantr.grid`'s oracle is `float64`-only, that
there is therefore no dtype axis, and that **the C++ registers no `float` overload** -- citing
`design/backend_parity.md` Rule 8, that a parity claim is only defined where the comparison can
say something.

#386's `AC1`/`AC7` require the mixin to be instantiated at `float` as a census device, and a
negative TU for "a hook hard-coding `double` at a `float` grid". Those are good compile-time
checks and they do not conflict with Rule 8 -- but only because the `float` instantiation is
**never bound and never compared**. Say so in the header, or the next ticket binds
`TensorProductGridFloat32` for symmetry with `Bezier32`/`Bezier64` and opens a parity claim
Rule 8 forbids.

## The C++ shape

Everything in this section was **compiled and run** before it was written down. See
"What was compiled" for the artefacts and the commands.

### `GridBase<Derived>`: a CRTP mixin that owns its state, dispatching by name hiding

```cpp
template <class Derived> struct grid_traits;          // primary, deliberately undefined

template <class Derived>
class GridBase {
  public:
    using scalar_type = typename grid_traits<Derived>::scalar_type;

    std::int64_t ndim() const noexcept;               // base state, not a primitive
    std::int64_t num_cells() const noexcept;          // base state, not a primitive

    /* the generic defaults, as ordinary non-virtual members */

    CellTags&       cell_tags() noexcept;             // eager member
    const CellTags& cell_tags() const noexcept;
    const BVH<scalar_type>& cell_bvh() const;         // fills a mutable std::optional

  protected:
    GridBase(std::int64_t ndim, std::int64_t num_cells);
    const Derived& self() const noexcept;
    Derived&       self() noexcept;

  private:
    std::int64_t ndim_, num_cells_;
    CellTags cell_tags_;
    FacetTags facet_tags_;
    mutable std::optional<BVH<scalar_type>> bvh_;
};
```

**"Closed" means: every grid is a `GridBase<itself>`, and the concept says so in one line**
(`std::derived_from<G, GridBase<G>>`). Not a variant, not a sealed virtual base. There is no
runtime grid type and no heterogeneous container of grids -- which is a real cost, stated
below, and one nothing in the tree pays today.

### Four departures from the ticket, each of which removes machinery rather than adding it

**D-a. `ndim` and `num_cells` are base state, so there are four primitives, not six** (and
three once `restrict` becomes a default -- see below).
The ticket's mechanism 3 -- "grow the concept with a non-const state accessor and delete the
cache accessor" -- exists to solve a problem that only arises if the state lives in `Derived`.
It does not have to. A CRTP base cannot initialise members from `self()` in its own constructor
(`Derived` is not yet constructed), which is presumably why the state was pushed outward; the
ordinary repair is to **pass the two sizes up from `Derived`'s constructor**, computed from
`Derived`'s own constructor arguments by a static helper. Then:

- `cell_tags()` has trivial const and non-const overloads and no `const_cast` is near it;
- `facet_tags_` is sized `2 * ndim` once, at construction;
- `num_cells()` is a field read inside the hot loops rather than a forwarded call;
- nothing can hand out the cache slot, because the slot is `private` in the mixin;
- the concept shrinks to the handful of things a grid author actually writes.

The invariant this makes legible, which the Python version leaves implicit: **`facet_tags_`
hard-codes `2 * ndim` facets per cell, so `num_local_facets` is not a specialisable hook.**
Assert that (`num_local_facets` must not appear in any grid's traits bitmask) rather than
leaving it to be discovered.

This departure has a dependency: it assumes a grid's cell count is fixed at construction.
`HierarchicalGrid` mutates today (`_rebuild`, `src/pantr/grid/_hierarchical_grid.py:523-579`),
but **#378 makes refinement return a new grid**, and #378 blocks #393/#394, which block #395.
The DAG already guarantees the ordering. **If #378 is ever dropped, this departure must be
revisited** -- that is the one thing that would reverse it.

**D-b. Name hiding does the dispatch; the traits bitmask is a *declaration*, checked, never a
dispatch input.** A default is an ordinary member of the mixin, so a `Derived` that declares
the same name hides it, and every call -- from the outside and from inside another default via
`self()` -- picks the specialisation. Nothing needs to read the bitmask at runtime or at
compile time to route a call.

This matters because the ticket's alternative (dispatch reads the bitmask) has a failure mode
name hiding does not: **a hook that is written but not declared is silently ignored, and the
default runs.** That is a wrong answer with no diagnostic. Under name hiding the hook is used,
and the census below turns the inconsistency into a compile error instead. Verified: the
"written but undeclared" translation unit is rejected by all four compilers.

**D-c. Detect a hook by its member-pointer *type*, not by a `requires` probe.** The ticket
rejects detection because a `requires` probe answers `true` for a hook with the wrong return
type and for one with a const-qualified output span, "both measured". Both observations are
correct and neither refutes detection -- they refute *that* probe. Compare instead:

```cpp
template <class D> constexpr bool redeclares_locate_many() noexcept {
    return !std::is_same_v<decltype(&D::locate_many), decltype(&GridBase<D>::locate_many)>;
}
```

If `D` does not redeclare the hook, `&D::locate_many` names the mixin's member and has type
`R (GridBase<D>::*)(...)`. If it does, the type is `R (D::*)(...)`. The two are the same type
exactly when the hook is absent -- and this stays correct when the return type is wrong or a
parameter is const-qualified, because those still yield a `D::*` type. So detection is
reliable, and the bitmask stops being the only source of truth and becomes something to
check *against*.

Known limits, worth writing in the header: `&D::name` is ill-formed if `D` overloads the name
or makes it a template. Both are design errors here, and both fail loudly.

**D-d. The census asserts the signature, so the convention is enforced rather than documented.**
Two `static_assert`s per hook, one line each in the grid's own translation unit:

```cpp
static_assert(declares<G>(Hook::locate_many) == detail::redeclares_locate_many<G>(), "...");
static_assert(!declares<G>(Hook::locate_many)
              || std::is_same_v<decltype(&G::locate_many),
                                std::vector<std::int64_t> (G::*)(std::span<const T>) const>, "...");
```

plus `template class GridBase<G>;` to force every default body. The ticket asks for the hook
signature convention to be *stated* in the header before any negative test is written, because
"a negative translation unit cannot be written against a convention that has not been written
down". Correct -- and the stronger form is available: state it **as the assertion**, once, in
a macro the grid's census expands. A convention nobody can paraphrase cannot drift.

### The trap that the concept has to be written against, and it is not the one the ticket names

`std::span<T>` **converts implicitly to** `std::span<const T>` (measured). So a concept written
as `requires(std::span<T> out) { g.cell_bounds(cid, out, out); }` is satisfied by a
`cell_bounds` taking `std::span<const T>` -- which cannot write to its output at all. The
grid compiles, satisfies the concept, and returns whatever was in the caller's buffer.

**Measured: that translation unit was accepted by g++ 14.4, g++ 10.5, clang++ 18.1 and
clang++ 10.0 alike, with `-Wall -Wextra -Werror`.** It is rejected once the concept pins the
primitives by exact member-pointer type rather than by callability:

```cpp
template <class G>
concept GridLike =
    std::derived_from<G, GridBase<G>>
    && std::is_same_v<decltype(&G::cell_bounds),
                      void (G::*)(std::int64_t, std::span<typename G::scalar_type>,
                                  std::span<typename G::scalar_type>) const>
    && /* locate, neighbor_across_facet the same way */;
```

The ticket places this failure mode among the *hooks*. It lands on the **primitives**, where
the output spans actually are, and the hook census would never have looked.

### `GridRestriction` and the self-referential constraint

The ticket's mechanism 2 holds and is adopted unchanged: `GridRestriction<G>` is
**unconstrained**, and `make_restriction` carries the `GridLike` constraint plus the
size-agreement check. Verified: a grid whose own `restrict()` returns
`GridRestriction<Derived>` -- naming the template from inside the class being defined --
compiles on all four compilers. What is lost is what the ticket says is lost: the type is
nameable for a non-grid, and aggregate initialisation bypasses the factory.

### What it costs at the call site

Nothing at all in the ordinary case: a default is a non-virtual member and `self()` is a
`static_cast`, so a whole default inlines into its caller.

In the hot loop the difference against a virtual base is large, and it is an **inlining**
difference rather than an indirect-call difference -- the primitive's body and its
`std::optional` return cannot cross the call.

Measured 2026-08-30 with `g++ 14.4 -O2`, one core via `taskset`, on the generic
`boundary_facets` loop over a 100^3 grid, 6e6 neighbour queries. The harness is described
under "What was compiled" below and is not committed:

| | CRTP | virtual | ratio |
|---|---|---|---|
| neighbour queries only | 5.7 ms | 85.8 ms | **15.1x** |
| the full loop, with the row `push_back` | 10.8 ms | 73.7 ms | **6.8x** |

**A first version of this benchmark reported 1.05x and was wrong.** It held the virtual grid
in a `const VBase&` bound to a local, so GCC devirtualized the call and measured CRTP against
CRTP. The number above hides the dynamic type behind a factory in a separate translation unit.
Anyone re-running this should check the same thing before believing a small ratio.

Why this loop and not another: `boundary_facets` is generic for **both** concrete grids at the
point #386 lands -- `TensorProductGrid` does not specialise it, and it is `O(num_cells * 2 * ndim)`
neighbour queries by construction. It is the largest generic loop in the layer.

Two honest caveats. The ratio is a property of a *cheap* primitive; a heavier one amortises the
call. And `HierarchicalGrid` **does** specialise `boundary_facets`
(`src/pantr/grid/_hierarchical_grid.py:975-1036`), so this measures the path a tensor-product
grid takes, not every grid.

### What CRTP costs, stated rather than waved past

- **There is no runtime grid type.** `std::vector<Grid*>`, "a grid chosen from a config file",
  and a non-template function taking any grid are all unavailable to a C++ consumer without
  writing their own variant or interface. Nothing in pantr does any of these today: the census
  found four `Grid` subclasses, and every one of the twenty-six grid `isinstance` sites in the
  tree tests a **concrete** type, never the base.
- **Every generic algorithm over grids is a template**, so it lives in a header and its errors
  are template errors. `cell_quadrature` (#388) is the first one that will feel this.
- **Compile time and code size** scale with grids x scalar types. Two grids x two scalars is
  four instantiations of nineteen defaults, which is nothing; it is worth a note only because
  the census deliberately forces all of them.


### `restrict` is a throwing default, not a fourth primitive

The ticket's mechanism 6 makes `restrict` required. **It does not have to be, and the compiler
is not what forces it.** Verified on all four compilers: a mixin default

```cpp
[[nodiscard]] GridRestriction<Derived> restrict(std::span<const std::int64_t>) const {
    throw std::logic_error("this grid kind does not support restrict().");
}
```

compiles and throws. No `Derived` is constructed, so nothing needs `Derived` to be
default-constructible, and `GridRestriction<Derived>` is only instantiated at the explicit
instantiation, where `Derived` is complete.

Three reasons to prefer it:

- **It preserves the Python contract verbatim.** `src/pantr/grid/_grid.py:290-300` states
  "Restriction is an *optional* grid capability" and raises `NotImplementedError`. Making it
  required changes the documented contract of a public method for a reason internal to the C++
  layer, which is the wrong direction of causation.
- **The ticket already contradicts itself on this.** It calls the base's `NotImplementedError`
  "reachable only through the test subclass this ticket removes", and then, three paragraphs
  later, preserves that subclass by re-basing `_PlainGrid` onto `_GridPython`. `_PlainGrid`
  supplies no `restrict`, so the path stays reachable.
- **The stated argument does not support the stated conclusion.** "One concept that excludes no
  grid is weight without enforcement" is an argument against adding a second `Restrictable`
  concept. It is not an argument for promoting `restrict` to a primitive; a throwing default is
  the third option and it costs three lines.

The usual objection -- a base advertising an operation not every subtype supports is a Liskov
violation -- does not bind in the C++ layer, because **nobody holds a `GridBase<D>&`
polymorphically**; there is no substitutability to violate. It binds on the *Python* `Grid`
Protocol, which is the real substitutability abstraction here -- and there `restrict` must
appear regardless, because consumers annotate `Grid`, with `NotImplementedError` documented
exactly as it is today.

So: `restrict` is `Hook::restrict`, censused like the rest, with signature
`GridRestriction<G> (G::*)(std::span<const std::int64_t>) const`. The concept has **three**
primitives: `cell_bounds`, `locate`, `neighbor_across_facet`.

## How inheritance crosses the seam: it does not

**After the port there is no implementation inheritance between grids on the Python side, and
no inheritance relationship crosses the boundary in either direction.** The Python surface
becomes three separate things that today are one:

| name | what it is | who uses it |
|---|---|---|
| `pantr.grid.Grid` | a `typing.Protocol`, `__slots__ = ()`, five `@abc.abstractmethod` primitives and **nineteen real default bodies** | annotations (`viz/_grid.py:54`, `_cell_quadrature.py:31`, `mpi/_create.py`), and the unbound differential call at `test_grid_hierarchical.py:1721` |
| `pantr.grid._GridPython` | today's ABC verbatim: `abc.ABC`, `__slots__`, the same five abstract and nineteen concrete methods | the temporary Python oracle -- base of `_TensorProductGridPython`, of `HierarchicalGrid` until #395, and of `_PlainGrid` |
| `pantr.grid._GridWrapper` | a private, **stateless** forwarding base: every method is `return self._impl.foo(...)` plus the wrap/unwrap of domain types | `TensorProductGrid` (#387) and `HierarchicalGrid` (#395), the two wrappers |

`_GridPython` **must not inherit from `Grid`.** Structural typing makes it satisfy the Protocol
without inheriting, and inheriting buys nothing while risking F3's `__dict__` leak. The same
goes for `_GridWrapper`.

### Who holds the vtable

Nobody. There is no vtable anywhere in the design: C++ dispatches by name hiding at compile
time, and the Python wrappers are unrelated classes that happen to satisfy the same Protocol.
What was one dispatch mechanism (Python MRO) becomes two independent ones that never meet.

### Can a Python-defined grid still exist? Yes -- as a *Python* grid, and only that

A user (or a test) subclasses `_GridPython`, supplies the primitives, and gets the nineteen
Python defaults. That keeps working under both backends and costs nothing new, because it is
what `_PlainGrid` already is.

What such a grid can **never** be is a C++ grid. `GridBase<Derived>` is a template; `Derived`
must be a compile-time type; there is no runtime extension point to register into.

### The callback bridge is buildable, and must not be built

For completeness, because "impossible" would be wrong: a single C++ type
`PyGrid : GridBase<PyGrid>` closing over a `nb::object` and forwarding the three primitives
into Python would compile and work. It is rejected on two independent grounds.

- **It is the shape `design/cross_backend_types.md` forbids**, in the form the amendment
  singles out: an object that is half one implementation and half the other, with values
  converted across the boundary per call.
- **It re-imports the cost the port exists to remove, into the loop that motivates it.**
  The generic `boundary_facets` makes `num_cells * 2 * ndim` primitive calls -- 6e6 on a
  100^3 grid. A C++-to-Python call with argument conversion is on the order of a hundred
  nanoseconds *at best*, against the 0.95 ns measured for the inlined native primitive. That
  is the object-mode-in-the-inner-loop failure the house rules name for Numba, arriving at a
  different boundary.

And it is unnecessary: the ruling is that the backend becomes C++ only, so a permanent
two-way bridge would be scaffolding built to be deleted.

### What actually breaks, and it is not "user subclassing"

The ticket measured zero `isinstance(x, Grid)` sites on the base; an independent census over
`src`, `tests`, `tools` and `scripts` reproduced that -- all twenty-six grid `isinstance` sites
test a concrete type. So no dispatch breaks. Three things do:

1. **`Grid()` stops raising a message containing "abstract"** -- `test_grid_abc.py:57-60`. Delete
   it, or re-point it at `_GridPython`.
2. **`isinstance(x, Grid)` becomes a `TypeError`** rather than `False`, because `Grid` should
   **not** be `@runtime_checkable`. Making it runtime-checkable would launder an unearned
   guarantee: a `runtime_checkable` Protocol checks only that the *names* exist, never the
   signatures, so it would answer `True` for an object that cannot actually serve as a grid.
   There are no such sites today; the cost is that a future one gets an error instead of a
   wrong answer, which is the right trade.
3. **A `Grid`-typed value loses `__init__`.** Nothing constructs a `Grid` today, so this is
   only a documentation change.

## The split: which of the nineteen defaults move

Reasons by group, since per-method reasons would be nineteen restatements of five.

**Group A -- moves, because it is a loop over the primitives.** `boundary_facets`,
`is_mesh_boundary_facet`, `neighbors`, `hanging_neighbors`, `locate_many`,
`collect_cell_bounds`. Each is `O(num_cells)` or `O(num_cells * 2 * ndim)` interpreter-level
calls today. **This group is the entire measurable payoff of the ticket**; see the 15x above.

**Group B -- moves, because it constructs a C++-owned domain type.** `cell_aabb` (`AABB`),
`reference_map` (`AffineTransform`), `cell_bvh` and `query_aabb` (`BVH`), `cell_tags` and
`facet_tags` (`CellTags`/`FacetTags`), `restrict` (a grid). All seven of those types are
already C++-owned. A Python method that pulled arrays across the seam and reassembled one
would be reassembly-on-the-far-side -- the move `design/cross_backend_types.md` names as where
an invariant gets dropped. Moving them means the wrapper only ever re-wraps a handle.

**Group C -- moves, because a concrete grid specialises it, or will.** `cell_level`,
`child_cells`, `num_local_facets`, `local_facet_axis_side`, `local_facet_bounds`. Individually
trivial; that is not the criterion. **A hook and the default it replaces must live in the same
language**, or the differential test that compares them -- the mechanism that caught defects in
the already-merged ports, and #387's `AC4` -- has nothing to compare against in one place.
`cell_level` is the concrete case: `HierarchicalGrid` specialises it
(`src/pantr/grid/_hierarchical_grid.py:1131-1145`).

**Group D -- moves, and is *duplicated* in the wrapper, deliberately.** `_check_cid`,
`_check_lfid`. C++ must validate, because the amendment's premise is a C++ program with no
interpreter above it. Python keeps its own only where the *message* is contractual. Measured:
nanobind maps `std::out_of_range` to `IndexError` **preserving `what()`**
(`nanobind/src/nb_internals.cpp:160-161`), and the must-pass assertions are
`pytest.raises(IndexError, match="out of range")`
(`tests/test_grid_hierarchical.py:1162,1164,1171,1610,1612`). So a C++ `std::out_of_range`
carrying today's text satisfies them with **no** wrapper-side check. Duplicate only where a
different exception *type* is owed -- the precedent is the tag registries, which raise
`KeyError` from the wrapper because nanobind has no path to it (`cpp/bindings/grid_tags.cpp:15-17`).

**Group E -- stays in Python, because it is about Python's calling convention, not about grids.**
`_normalize_points` (`npt.ArrayLike` coercion, the `(ndim,)` to `(1, ndim)` promotion, and the
exact `ValueError` text), `iter_cells` (returns a Python iterator over `range`), `__repr__`,
`__reduce__`. C++ takes a typed span; deciding what a `list[list[float]]` means is the
wrapper's job permanently. Precedent: `AABB.__init__` coerces, then hands down ravelled float64
(`src/pantr/geometry.py:566-568`).

**Nothing in Group A, B or C stays behind.** The one method that looks like it should and does
not is `iter_cells`: it is in Group E because its *return type* is a Python iterator, not
because the work is small.

## The wrapper's shape

`_GridWrapper` is a **private, stateless forwarding base**: `__slots__ = ()`, no `__init__`, and
every method a one-liner over `self._impl` plus the wrap/unwrap of a domain type. Each concrete
wrapper declares `__slots__ = ("_impl", "_cell_tags", "_facet_tags", "_bvh")` and its own
`__init__`, `__repr__`, `__reduce__`, and grid-specific surface.

This is inheritance used for reuse, which is normally the shape to refuse. It is right here for
one reason and it should be stated rather than assumed: **the docstring is the user-facing
contract** (`CLAUDE.md`, Layer 1), so twenty-four forwarders duplicated across `TensorProductGrid`
and `HierarchicalGrid` is forty-eight docstrings and **two copies of the contract that will drift**.
The base holds no state and imposes no protocol beyond "you have `_impl`", so nothing about the
usual objection applies. The composition alternative -- a module of free functions
`_fwd_cell_aabb(impl, cid)` -- still needs twenty-four call sites per wrapper and saves nothing.

`_GridWrapper` must not inherit from `Grid` (F3) and must not inherit from `_GridPython` (that
would be the mixed object `cross_backend_types.md` forbids: Python defaults over a C++ handle,
inside a class the C++ backend hands to users).

### Three things the wrapper must get right, all of them silent when got wrong

**W1 (critical). `cell_tags()` must be bound with `nb::rv_policy::reference_internal`.**
Verified at the nanobind source (`nb_cast.h:461-464`): for a method returning `T&`, both
`automatic` and `automatic_reference` resolve to **`rv_policy::copy`**. With the default policy
`g.cell_tags.set(...)` mutates a temporary and the write is silently lost. `reference_internal`
also ties the returned object's lifetime to the grid, which is the second thing needed.
`rv_policy` appears exactly **once** in the whole binding tree today
(`cpp/bindings/geometry.cpp:84`), so this is unexercised ground and no existing binding models it.

**W2 (important). The wrapper must memoise the `CellTags`, `FacetTags` and `BVH` wrappers, and
identity is asserted.** `tests/test_grid_tags.py:152` is `assert ct is g.cell_tags  # cached` and
`tests/test_grid_tensor_product.py:185` is `assert g.cell_bvh() is g.cell_bvh()`. #387 names the
BVH case; it does not name the tags identity assertion. Memoising also removes a worse failure
mode than identity: a property that *constructed* a fresh `CellTags(num_cells)` instead of
wrapping the existing handle would hand back an empty registry, and every assertion about a
tag set earlier would fail far from the cause.

**W3 (important, and it lands on #387 rather than here).
`tests/test_grid_tags.py:149-150` reads `g._cell_tags` and `g._facet_tags` directly**, asserting
they are `None` before first use. That file is on **#387's** must-pass-*without-being-edited*
list. A `__slots__`-based wrapper has no such attribute under those names unless the memo slots
are deliberately named `_cell_tags`/`_facet_tags` **and** initialised to `None`, which is
achievable and is the cheapest fix. #387 spotted the identical problem at
`tests/test_grid_tensor_product.py:180` for `_bvh` and proposed a `bvh_is_built()` predicate;
it did not spot these two. Either name the memo slots to match, or move
`test_grid_tags.py` onto #387's editable list -- but the choice has to be made, because as
written #387's must-pass set is not satisfiable.

Since eager C++ tags mean the registry is never `None` on the C++ side, "has it been touched"
becomes a different question there. Keeping the wrapper's memo slots `None` until first access
preserves both the assertion and the laziness the Python contract advertises, at the cost of one
extra branch per property read.

### What `_GridPython` is for, and when it dies

It is the **oracle**, and nothing else. Three consumers: `_TensorProductGridPython` (#387),
`HierarchicalGrid` until #395, and `_PlainGrid` in `tests/test_grid_abc.py`. It is exactly
today's `Grid` class body, renamed, with the module docstring's "abstract base class" prose
moved to `Grid`.

It needs `docs/conf.py`'s `nitpick_ignore` because `HierarchicalGrid` is public, documented with
`:show-inheritance:` (`docs/api/reference.md:67-69`), `nitpicky = True` is set
(`docs/conf.py:212`), and the docs build under `-W`. Add `("py:class", "_GridPython")` beside
`("py:class", "_AffineMap")` at `docs/conf.py:234`, which is already there for exactly this
reason. **Add `("py:class", "_GridWrapper")` at the same time** -- #387's `TensorProductGrid` will
show it, and discovering that during #387's docs build costs a round trip.

It dies when the last of the three is gone, which is #395 plus whatever retires the Python
backend. **It is scaffolding and should be documented as scaffolding**, so that "keep parity with
`_GridPython`" never becomes a reason not to improve the C++.

## What #387 and #395 inherit from this

**#387, `TensorProductGrid`.** Declares `Hook::locate_many | Hook::collect_cell_bounds |
Hook::restrict` in its `grid_traits`; the census then rejects a fourth hook written by accident
and a declared hook never written. Its `AC4` -- "compares each of the three specialised defaults
against the generic one on the same input" -- is served directly by
`g.pantr::grid::GridBase<G>::collect_cell_bounds()`, the qualified call that reaches the hidden
default (verified compiling and running on all four compilers). **The derived class must not
privately alias its base**, or the differential test cannot name it; that cost me a compile in
the prototype. This is the exact C++ analogue of `Grid.boundary_facets(g)` at
`test_grid_hierarchical.py:1721`, which is a pleasing symmetry and worth preserving on purpose.

**#395, `HierarchicalGrid`.** Three inheritances, one of them a release:

- Its five hooks (`locate_many`, `collect_cell_bounds`, `hanging_neighbors`, `boundary_facets`,
  `cell_level`) go in the bitmask; nothing else changes.
- The `invalidate_caches()` that #395 asks for is **probably unnecessary and should not be
  built here.** #395 wants it because `_rebuild` writes the base's private slots today
  (`src/pantr/grid/_hierarchical_grid.py:541-544`). But #378 makes refinement return a *new*
  grid, and #378 blocks #393/#394, which block #395 -- so by the time #395 runs, there is
  nothing to invalidate: the new grid has fresh empty caches and the old one's are still valid.
  **The consequence is worth naming: with #378, a C++ grid is immutable after construction
  except for the BVH memo, so construct-then-freeze finally holds for grids** -- it does not
  today. Ship #386 without `invalidate_caches()`; let #395 add three lines if it turns out to
  be needed. A protected mutator nothing calls is the kind of seam that becomes load-bearing by
  accident.
- Five private-slot reads in the must-pass `tests/test_grid_hierarchical.py`
  (`:353`, `:359`, `:536`, `:549`, `:1090`) survive #386 unchanged, because `HierarchicalGrid`
  stays on `_GridPython` here. **They break at #395**, and W3 above is the same problem one
  ticket earlier. #395 should inherit W3's resolution rather than rediscover it.

**A latent gap, unrelated to the port but surfaced by the census:** `HierarchicalGrid` does
**not** override `child_cells`, so the one grid kind that has refinement children returns `()`
from the method that reports them. That is a defect in today's Python, not something the port
introduces; it should be a ticket, not a fix folded into #386.

## Migration order

1. `cpp/include/pantr/grid/grid.hpp` -- traits, `Hook`, the mixin, the concept,
   `GridRestriction` + `make_restriction`, and the census macros. Nothing Python.
2. `cpp/tests/test_grid_defaults.cpp` -- the synthetic zero-hook grid and the hand-computed
   expectations, plus the lazy-cache build/reuse test (`AC3`).
3. The negative translation units, **eight of them** (the ticket's six plus written-but-undeclared
   and declared-but-unwritten), each asserted *rejected* by both dev compilers.
4. Python: `Grid` becomes the Protocol with bodies; `_GridPython` is the renamed ABC;
   `pantr/grid/__init__.py:16-20` prose updated; `docs/conf.py` gains two `nitpick_ignore`
   entries.
5. `tests/test_grid_abc.py`: `_PlainGrid` re-based on `_GridPython`; **one** test deleted;
   `test_incomplete_subclass_is_abstract` re-pointed at `_GridPython` and kept.
6. `tests/typing/` -- the mypy negative harness, restated per F4.

Steps 1 to 3 touch no Python and can land before step 4. That matters because step 4 is the only
part that can break a must-pass file.

## Alternatives rejected

**A virtual base with a sealed derivation set.** Simpler in every respect: one binding, a
non-templated `GridRestriction`, a runtime grid type, generic algorithms as ordinary functions,
and a Python subclass reachable through a trampoline. Rejected on the measurement: **15.1x** on
the generic `boundary_facets` neighbour loop, **6.8x** on the whole method, and the loss is
inlining rather than the indirect call, so it does not shrink with better branch prediction.
`boundary_facets` is generic for `TensorProductGrid` and `O(num_cells * 2 * ndim)` by
construction. **What would change this:** if a C++ consumer needs to hold a grid whose kind is
decided at run time, or if `cell_quadrature`-style generic algorithms become numerous enough
that header-only templates hurt more than the loop gains. Neither is true today.

**`std::variant<TensorProductGrid<T>, HierarchicalGrid<T>>` plus free-function defaults.** The
most literal reading of "closed hierarchy", and it keeps a runtime type. Rejected because it
buys the runtime type at the price of editing the variant for every new grid *and* losing the
`Derived`-typed `restrict` return, while the CRTP shape gives the same closure guarantee in one
concept line. It stays available as a thin façade over the CRTP types if a runtime grid is ever
needed; that is a strictly additive change.

**A `PyGrid` trampoline so a Python subclass can be a C++ grid.** Buildable; rejected on the two
grounds in the seam section. **What would change this:** a decision to keep Python as a
permanent extension point, which contradicts the standing ruling that the backend becomes C++
only.

**Dispatching on the traits bitmask (the ticket's mechanism 4).** Rejected in favour of name
hiding: bitmask dispatch silently ignores a hook that was written but not declared, and it forces
distinct names for hook and default. See D-b.

**`Grid` as a five-member Protocol (what an implementer supplies).** Rejected: every site
annotating `Grid` loses the methods it uses -- `src/pantr/viz/_grid.py:54`,
`src/pantr/grid/_cell_quadrature.py:31`, `src/pantr/grid/_partition_grid.py:34`, and
`src/pantr/mpi/_create.py:87`, which is a real local annotation (`grid: Grid`) satisfied by a
`HierarchicalGrid` on one branch and a `TensorProductGrid` on another. The first of those must
not be edited. See F4.

**`Grid` as `@runtime_checkable`.** Rejected: it checks names, not signatures, so it would answer
`True` for objects that cannot serve as grids. Zero sites need it.

## Bad practices flagged, for the user's inspection rather than for fixing here

- **`CLAUDE.md`'s "Validation lives exclusively in Layer 2" no longer describes the code.** Once
  a C++ program links pantr with no interpreter above it, validation *must* be in C++, and the
  Python layer's job becomes coercion plus exception-type translation. Group D above is written
  against the new reality; the prose should be updated once, rather than each port deciding
  locally what the rule now means.
- **`src/pantr/grid/_grid_backend.py`'s "no `float` overload" and #386's `float` census pull in
  opposite directions.** Both are right; the reconciliation (instantiate, never bind) has to be
  written down or the next ticket resolves it the other way. See F5.
- **`HierarchicalGrid._rebuild` writing its base class's private slots**
  (`_hierarchical_grid.py:541-544` against `_grid.py:71`) is already on the record as something
  #395 must fix. Noted here only because #378 may delete the need for it entirely, which is a
  better outcome than the protected accessor #395 currently plans.

## Epistemic status

- **Compiled and run, 2026-08-30, on g++ 14.4.0, clang++ 18.1.8, g++ 10.5.0 and clang++ 10.0.0,
  all with `-std=c++20 -Wall -Wextra -Werror`:** the mixin with base-held state; traits-supplied
  `scalar_type`; name-hiding dispatch, including the qualified call that reaches the hidden
  default; the member-pointer-type detector; the two-assertion census; explicit instantiation at
  `float` and `double`; `GridRestriction<G>` unconstrained with a constrained factory, named as a
  return type inside the class being defined; the lazy `mutable std::optional` cache built from a
  const grid; and a throwing default `restrict`. Eight negative translation units rejected by all
  four.
- **Measured, 2026-08-30, on this machine:** the `span<T>` to `span<const T>` conversion that
  makes a callability-based concept accept an unwritable output span, and its repair; the
  CRTP-versus-virtual ratios, with the first, devirtualized attempt recorded as wrong.
- **Measured on CPython 3.14.6:** every Protocol claim in F1 to F4 -- unbound calls through a
  Protocol default, the two instantiation messages, `_ProtocolMeta` deriving from `ABCMeta`, and
  the `__slots__`/`__dict__` behaviour.
- **Verified by reading the source:** nanobind's `rv_policy` resolution for an lvalue-reference
  return (`nb_cast.h:461-464`) and its `std::out_of_range` to `IndexError` mapping
  (`nb_internals.cpp:160-161`).
- **Verified by reading the tree:** the override census for all four `Grid` subclasses; the
  twenty-six grid `isinstance` sites, none on the base; every private-slot read in the must-pass
  files.
- **Asserted, not measured:** the cost of a `nb::object` callback in the rejected trampoline. The
  argument does not depend on the figure -- any per-call cost above a few nanoseconds settles it
  against 0.95 ns -- but the number itself is an estimate.
- **Not investigated:** whether the not-yet-public downstream consumer imports any of
  `_grid.py`'s private names. **`_grid.py` defines `_check_cid`, `_check_lfid` and
  `_normalize_points`, and this design moves all three. Nothing in pantr's CI can see that
  repository, so the check is owed before #386 lands.** `Grid` itself changing kind is the
  larger exposure: a consumer subclassing `Grid` would have to move to `_GridPython`.

## What was compiled, and how to reproduce it

The prototype is **not committed**: it exists to settle the mechanisms, and its home is
`cpp/include/pantr/grid/grid.hpp` plus `cpp/tests/` once #386 is built. What follows is enough
to rebuild it from this note.

**Eleven translation units**, each self-contained on pantr-free stand-ins for `BVH`, `CellTags`
and `FacetTags`, so that the mechanism is isolated from pantr's own headers:

| TU | asserts |
|---|---|
| `t_positive` | a zero-hook synthetic grid: every default runs and agrees with a hand-computed value; the BVH cache is unbuilt, then built, then the *same object*; a const grid reads tags but the non-const one mutates the same object; `facet_tags` is sized `2 * ndim`; `restrict` round-trips |
| `t_hooked` | a two-hook grid: name hiding routes the call to the hook, and `g.GridBase<G>::boundary_facets()` still reaches the hidden default -- the differential oracle |
| `t_restrict_default` | a grid with no `restrict`: compiles, throws at run time |
| `n1` | a const grid calling `invalidate_caches()` -- **rejected** |
| `n2` | a const grid calling `cell_tags().set(...)` -- **rejected** |
| `n3` | a hook returning `vector<int32_t>` where the default returns `vector<int64_t>` -- **rejected** |
| `n4` | a primitive whose output span is `span<const T>` -- **rejected**, but only after the concept was changed from callability to exact member-pointer type |
| `n5` | a hook hard-coding `double`, at the `float` instantiation -- **rejected** |
| `n6` | a grid derived from `GridBase` with no `grid_traits` specialisation -- **rejected at the grid's own definition** |
| `n7` | a hook written but not declared in the traits -- **rejected** |
| `n8` | a hook declared in the traits but not written -- **rejected** |

Every one was run against all four compilers:

```bash
for cxx in g++ clang++ g++-10 clang++-10; do
  $cxx -std=c++20 -Wall -Wextra -Werror -I. -fsyntax-only "$tu".cpp
done
```

`n7` and `n8` are **not** in the ticket's `AC2` list and should be added: together they are what
makes the traits bitmask trustworthy, since without them a declaration can disagree with the
class in either direction.

The CRTP-versus-virtual benchmark is two implementations of one `boundary_facets` loop over a
`nx * ny * nz` box grid whose `neighbor_across_facet` is index arithmetic returning
`std::optional<std::int64_t>`. The **only** subtlety, and the one that produced a wrong answer
first time: the virtual grid must be reached through a `std::unique_ptr<VBase>` returned by a
factory **compiled in a separate translation unit**, or GCC devirtualizes the call and the
benchmark compares CRTP against CRTP. Both a count-only and a `push_back`ing variant were timed,
with warm-up passes, on one pinned core.
