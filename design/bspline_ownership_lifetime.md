# Holding another domain type across the binding: who owns it, and who keeps it alive

**Status:** proposed, 2026-08-31. Written for ticket #393, and binding on #395 to #401 if
adopted.
**Date:** 2026-08-31.
**Scope:** what an accessor hands back when a C++-owned domain type holds another one, how the
returned value's lifetime is guaranteed, which `rv_policy` each binding owes, and what the
Python wrapper must memoise. Not where derived caches live, which is
`design/bspline_derived_caches.md`. Not how the two backends are compared, which is
`design/backend_parity.md`.
**Companions:** `design/cross_backend_types.md` (the ownership ruling this obeys),
`design/grid_hierarchy_port.md` (the wrapper pattern this extends, and whose `W1` this
corrects), `design/bspline_derived_caches.md` (the memo slots this note's identity contracts
require).

**Validated against:** `proto/cpp` at `a45e935`, worktree `design/393-394-bspline-decisions`,
plus `feat/387-tensor-product-grid` at `d7b8654` read through `git show` for the one shipped
wrapper precedent. Every Python line number below was read in one of those two trees and says
which. **nanobind 2.14.0**, CPython **3.14.6** (GIL enabled: `sys._is_gil_enabled()` is `True`
and `sysconfig.get_config_var("Py_GIL_DISABLED")` is `0`), g++ **14.4.0** at `-O2`.

## The decision in one paragraph

**Classify the accessor, not the type.** Every accessor that returns a domain object falls into
one of three classes, and the class settles the C++ storage, the `rv_policy`, the wrapper's memo
and the test: **V**, the accessor *constructs* its result (about 31 of the roughly 40 accessors
in `pantr.bspline`) -- return by value, no policy, no lifetime relationship; **H**, the accessor
hands out a subobject the owner *holds* (eight sites, plus `HierarchicalGrid.root`) -- the owner
stores **`std::shared_ptr<const T>`** and returns a copy of it, so the *value* is shared and the
owner's death is irrelevant; **A**, the accessor hands out the owner's numeric storage -- a
`const`-scalar `nb::ndarray` view with the owner's Python object as its owner, which is
`bezier_type.cpp`'s shipped precedent. **No accessor in `pantr.bspline` uses
`rv_policy::reference_internal`**, and that is the decision's whole content: `reference_internal`
puts the lifetime guarantee in the *binding*, and the standing ruling is that the binding goes
away. A `shared_ptr<const T>` puts it in the *type*, where the interpreter-free C++ consumer
gets it too. `THBSplineSpace.grid` is the single exception to `const`, for a reason
`grid/tags.hpp` already argued.

## Findings against the ticket as written

Five of these are measured. They are first because three of them change what the dependent
tickets must do, and one of them corrects a claim already in the tree.

### F1 (critical). The `rv_policy` fact this ticket hands me is true of a method and false of a property -- and the tree states it of a property

The brief, `design/grid_hierarchy_port.md` W1, and `cpp/bindings/grid_types.cpp`'s header (on
`feat/387-tensor-product-grid`, lines 16-30) all say: for a function returning an lvalue
reference, `automatic` and `automatic_reference` resolve to `rv_policy::copy`, so under the
default policy a write through such an accessor mutates a temporary and is silently lost.

**The mechanism is real** -- `nanobind/nb_cast.h:455-473`, `infer_policy<T>`, does exactly that
for `std::is_lvalue_reference_v<T>`. **The conclusion does not reach a property.**
`nb::class_::def_prop_ro` delegates to `def_prop_rw` (`nanobind/nb_class.h:732-735`), and
`def_prop_rw` builds the getter with `rv_policy::reference_internal` passed **positionally,
ahead of the caller's `extra...`** (`nb_class.h:693-703`). So a property getter never sees
`automatic`, and its default is already `reference_internal`.

Measured, both directions:

| binding of a method returning `Inner&` | aliases the member | write visible | `Inner` copies per access | owner kept alive |
|---|---|---|---|---|
| `.def("m", &Outer::single)` | **no** | **no** | 1 | no |
| `.def("m", &Outer::single, nb::rv_policy::reference_internal)` | yes | yes | 0 | yes |
| `.def_prop_ro("p", &Outer::single)` -- no policy named | **yes** | **yes** | 0 | **yes** |
| `.def_prop_ro("p", &Outer::single, nb::rv_policy::copy)` | no | no | 1 | no |

The consequence for the tree: of the three names in `grid_types.cpp`'s paragraph, `cell_tags`
and `facet_tags` are `def_prop_ro` (lines 379-382), where the explicit `reference_internal` is
**documentation rather than a repair**, and `cell_bvh` is a plain `.def` (line 383), where it is
the repair. Nothing there is wrong-behaving; the *rule* an engineer would carry forward from it
is. Naming the policy on a property remains the right style -- it is the only spelling that
survives someone changing `def_prop_ro` to `.def` -- but it must be documented as belt-and-braces,
not as the thing that makes the write land.

`def_prop_ro_static` is a third case worth one line: it routes through `def_prop_rw_static`,
which uses `rv_policy::reference` (`nb_class.h:712-729`) -- aliasing with **no** keep-alive,
because a static getter has no self. Nothing in pantr uses one; a future one returning a
reference would dangle silently.

### F2 (critical). `nb::keep_alive<0, 1>` does not repair the copy, and it silences the symptom that would have found it

Measured: `.def("m", &Outer::single, nb::keep_alive<0, 1>())` -- no `rv_policy` -- keeps the
owner alive **and still hands back a copy**. The write through it is still lost, and one
`Inner` copy is still made per access.

This is the worst available combination, and it is the one a reader reaches for. The natural
diagnosis of "my child object dangled" is "add a keep-alive", and doing that makes the
lifetime test pass while leaving the aliasing broken. `keep_alive` and `rv_policy` are
orthogonal: the first is applied post-call in the function dispatcher, the second decides
whether an instance is created and copied into or made to point at existing storage
(`nb_type.cpp:1963-2050`).

So: **a keep-alive annotation is never evidence that an accessor aliases.** Under this note's
decision the point is moot for `pantr.bspline`, because no accessor there uses either
mechanism -- but #395 inherits a grid whose `cell_tags` genuinely must alias, and that is where
the trap is live.

### F3 (important). The policy propagates into a container, and each element gets its own keep-alive

`nanobind/stl/detail/nb_list.h:60-67` forwards the caller's `rv_policy` to every element's
caster. So a method returning `const std::vector<Inner>&`:

| binding | elements alias | copies per access | an escaped *element* blocks the owner's destructor |
|---|---|---|---|
| default policy | no | 3 (one per element) | no |
| `reference_internal` | yes | 0 | **yes** |
| `rv_policy::reference` | yes | 0 | no |

Measured with a destructor counter in the C++ and with `sys.getrefcount` in the Python. The
third column is the part that is not obvious: the keep-alive is installed on each *element*, not
only on the returned list, so `x = space.spaces[0]` alone is enough to pin the owner. The
container therefore is **not** a separate hazard from the scalar case under
`reference_internal`, which was the open question in #393's own wording.

It stays a separate hazard under the default policy, and worse than the scalar one: at 16384
doubles per nested object, one access of a 3-element container cost 109 microseconds against
0.44 microseconds for the aliasing form (table under "What it costs").

### F4 (critical). A value assertion does not detect a broken lifetime. A refcount assertion does, for free

Measured, with `rv_policy::reference` (aliases, no keep-alive), after destroying the owner and
running `gc.collect()`:

| escaped child | reads back |
|---|---|
| a single member (`Inner&` into the owner) | `100` -- **the correct value** |
| an element of the owner's `std::vector<Inner>` | `1096836676` -- garbage |

Both are use-after-free. The first one *passed*. So the obvious test -- escape the child, drop
the owner, assert the value -- is exactly the test that reports success on a broken design, and
it reports it in the case a reviewer is most likely to write by hand (a scalar member, small,
whose bytes are still intact).

Two deterministic detectors, measured:

- **`sys.getrefcount` on the owner's handle, before and after the access.** The delta is `1`
  exactly when a keep-alive was installed and `0` otherwise -- measured across all seven
  binding shapes above, with no C++ change and no annotation. This is the detector to use.
- **`weakref.ref` on the owner's handle**, which needs `nb::is_weak_referenceable()` on the
  bound class; without it, `weakref.ref` on a nanobind instance raises
  `TypeError: cannot create weak reference` (measured), and `gc.get_referents(child)` is `[]`,
  so the keep-alive edge is invisible to the GC module.

`sys.getrefcount`'s *absolute* value is a CPython implementation detail; the delta is not, and
the delta is what to assert. On a free-threaded build the function is documented as unreliable
for immortalised objects; a bound instance is an ordinary heap object, and no CI leg is
free-threaded today (verified: `.github/workflows/ci.yaml:20,80,131` list `3.11`, `3.13`,
`3.14`, none of them `t`).

### F5 (important). The nine accessors the ticket names are not the nine that matter, and the real count is about forty

Verified by reading each site.

**Three of the nine carry no lifetime question at all**, because they hold a freshly constructed
object rather than a subobject of anything: `BsplineSpaceRestriction.space` (the field is filled
at `src/pantr/bspline/_bspline_space_nd.py:390` from a `BsplineSpace(...)` built on the spot),
`THBSplineSpaceRestriction.space` (`_thb_spline_space.py:1064`), and `LocalSpace.space`
(`_local_space.py:353,478`, from `global_space.restrict(window)`).

**A fourth is not the aggregation case at all.** `ExtractionStructView`
(`spanwise_element_extraction.py:1119-1196`) has **no `space` field** -- its fields
(`:1152-1158`) are three per-direction array bundles plus integer shape metadata. So #399's
premise, that its lifetime "is exactly what the aggregation-lifetime note settles", is half
right: it is class **A** below, the array-view case that `cpp/bindings/bezier_type.cpp:43-56`
already settled, and it needs nothing from the aggregation rule.

**What is left is eight held accessors**, and the ticket misses three of them:
`BsplineSpace.spaces` (`_bspline_space_nd.py:68-74`), `Bspline.space` (`_bspline.py:126-136`),
`THBSpline.space` (`_thb_spline.py:87-94`), `MultiLevelExtraction.space`
(`multilevel_extraction.py:121-128`), `SpanwiseElementExtraction.space`
(`spanwise_element_extraction.py:210-217`), `THBSplineSpace.grid`
(`_thb_spline_space.py:809-816`), and -- not in the ticket -- `THBSplineSpace.root_space`
(`:818-825`), `THBSplineSpace.level_space` (`:917-934`), plus `HierarchicalGrid.root`
(`_hierarchical_grid.py:477-484`) on the grid side of #395.

**And roughly thirty-one more accessors return a domain object by construction**, which is the
number that matters for effort: 17 on `Bspline`, 3 on `BsplineSpace1D`, 6 on `THBSplineSpace`,
2 on `THBSpline`, 1 on `BsplineSpace`, plus the module-level factories. They are all class **V**
and none of them needs a decision. **Restate #393's "at least nine accessors qualify" as "eight
hold, about thirty-one construct, and the classification is the deliverable."**

### F6 (critical). Two identity assertions in the suite are stronger than "the same object twice", and one of them fixes the C++ constructor's signature

- `tests/test_bspline_space.py:89` is `assert space.spaces[0] is space_1d`. The object that
  comes back is **the constructor argument's own Python object**. No C++ object can supply
  that; only the wrapper can, by keeping what it was built from.
- `tests/test_transform.py:634` is `assert s2.space is s.space  # same space object`, where
  `s2 = s.transform(...)`. A **derived** object hands back the **source's** space wrapper.
  Wrapping `s2._impl.space` afresh gives a different Python object even when the C++ pointer is
  identical, so this one cannot be satisfied by memoisation alone.

The first has a consequence one level down. If `space.spaces[0] is space_1d` holds at the
wrapper level, then `space.spaces[0]._impl is space_1d._impl` holds too -- so a C++
`BsplineSpace` that *copied* its 1D spaces would present two Python objects agreeing on identity
over two different C++ objects, which is a divergence a parity test would eventually find and
nobody would enjoy diagnosing. **The identity contract, not performance, is what requires the
C++ constructor to share rather than copy.**

## The decision

### The three classes

| | what the accessor does | C++ storage and signature | binding | wrapper | lifetime guarantee |
|---|---|---|---|---|---|
| **V** *value* | constructs its result | returns `T` by value | default policy (resolves to `move`) | `_wrap` the fresh handle | none needed; the owner may die immediately |
| **H** *held* | hands out a subobject the owner keeps | stores `std::shared_ptr<const T>`; returns a copy of it | default policy, plus `#include <nanobind/stl/shared_ptr.h>` | memoise the wrapper in a named `__slots__` slot | the **value** is shared; the owner is irrelevant |
| **A** *array view* | hands out the owner's numeric storage | returns `std::span<const T>` | `nb::ndarray<const T, ...>(ptr, shape, self)` | pass through unchanged | the array holds a reference to the owner's Python object |

**No `rv_policy::reference_internal` anywhere in `pantr.bspline`.** State it as a rule so that a
reviewer can grep for a violation.

### Why H is `shared_ptr<const T>` and not `reference_internal`

Four reasons, in the order they decide it.

1. **`reference_internal` is a guarantee that lives in the binding, and the binding is
   scheduled for deletion.** The standing ruling on this port is the user's, verbatim: *"mi idea
   sobre el backend de pantr es que sea solo C++. La idea de tener python o C++ es solo temporal,
   para que sirva de oraculo durante el desarrollo. En el futuro, solo C++."* A C++ consumer
   writing `const BsplineSpace1D& s = sp.space(0);` and then letting `sp` go out of scope gets no
   protection from a nanobind policy, and nothing in the type warned it. `shared_ptr<const T>` is
   a guarantee the type carries, so both consumers get the same one.
2. **The tree already made this exact call once, for the same reason.**
   `cpp/include/pantr/grid/tags.hpp:20-29` holds each tag behind `std::shared_ptr<const Tag>`
   precisely so a handed-out view outlives a replacement, and says so: *"the port would otherwise
   have introduced a use-after-free the pre-port class could not have."* This note generalises
   that ruling from a tag's arrays to a nested space.
3. **It is the only mechanism that reproduces `Bspline`'s in-place semantics, and the
   alternatives fail in two different silent ways.**
   `Bspline.reverse(direction, in_place=True)` and `permute_directions(..., in_place=True)`
   **replace** `self._space` (`_bspline.py:1035-1039`, `1101-1106`) -- verified by execution:
   `before = b.space; b.reverse(0, in_place=True); before is b.space` is `False`, and `before`
   keeps the old space. That is the contract to preserve. Measured, three storage shapes, an
   escaped nested object across one in-place reseat:

   | owner stores | accessor returns | escapee reads after the reseat | nested destructor ran during the reseat | `escapee is owner.space` |
   |---|---|---|---|---|
   | the space **by value** | `Space&`, `reference_internal` | **`2`** -- the *new* space | yes | **`True`** |
   | `shared_ptr<const Space>` | `const Space&`, `reference_internal` | `1` -- correct | **yes** | `False` |
   | `shared_ptr<const Space>` | `shared_ptr<const Space>` | `1` -- correct | no | `False` |

   The first row is a **silent wrong answer**: the escaped object becomes the new space, with no
   error and with `is` still reporting identity. The second row is the shape someone will propose
   as the best of both -- store the handle, hand out a reference, name the policy -- and it is a
   **use-after-free that returned the correct value**, which is F4's pattern again. Only the third
   row reproduces today's Python.
4. **It composes, and `reference_internal` does not compose cleanly.**
   `Bspline` -> `BsplineSpace` -> `BsplineSpace1D` is two levels. Under `shared_ptr` each level
   hands out one atomic increment and no level knows about any other. Under `reference_internal`
   the keep-alive is installed against `cleanup->self()`, so `b.space.spaces[0]` pins the
   `BsplineSpace` and the `BsplineSpace` pins the `Bspline` -- correct, but only because each
   intermediate Python object exists long enough to be the `self` of the next call, which is a
   property of the expression the caller wrote rather than of the design.

Immutability is what makes sharing safe, and it is already true: verified by reading, every
domain type in `src/pantr/bspline/` is immutable after construction except `Bspline` (whose
`_space` slot is *reseated*, never mutated) and the `HierarchicalGrid` reachable through
`THBSplineSpace.grid`. `shared_ptr<const T>` over an immutable `T` is a value with a cheap copy;
it is construct-then-freeze with sharing added, not mutation added.

### The one exception to `const`

`THBSplineSpace` stores **`std::shared_ptr<HierarchicalGrid>`**, non-const, and hands out a copy
of it. The reason is narrow and is not "grids are special": a grid holds `CellTags` and
`FacetTags`, which `cpp/include/pantr/grid/tags.hpp:14-19` already reasons as the accumulating-
container exception to construct-then-freeze, and `thb.grid.cell_tags.set(...)` works today. A
`shared_ptr<const HierarchicalGrid>` would reach only the `const` overload of `cell_tags()` and
would silently remove the ability to tag cells through a THB space's grid.

This is safe only because **#378 makes refinement return a new grid**. Until #378 lands, sharing
a mutable grid between a `THBSplineSpace` and its caller is what the staleness counter exists to
police (`_hierarchical_grid.py:505-517`, read by `_thb_spline_space.py:745-757`). **If #378 is
dropped, this exception must be revisited** and the grid must go out as a copy instead. That is
the one thing that would reverse it. #393 and #394 are both already blocked on #378 in the DAG,
so the ordering is guaranteed.

### The C++ shape, and the accessor pair

```cpp
class BsplineSpace {
  public:
    /// Share direction `d`'s 1D space.  The returned handle keeps its value alive
    /// independently of this space, so a caller may outlive the owner.
    [[nodiscard]] std::shared_ptr<const BsplineSpace1D> space(std::int64_t d) const;

    /// Borrow direction `d`'s 1D space.  Valid while `*this` is, and NOT bound: an
    /// inner loop must not pay an atomic pair per access.  See the `_ref` rule below.
    [[nodiscard]] const BsplineSpace1D& space_ref(std::int64_t d) const noexcept;

    /// Share every direction, in axis order.
    [[nodiscard]] std::span<const std::shared_ptr<const BsplineSpace1D>> spaces() const noexcept;

  private:
    std::vector<std::shared_ptr<const BsplineSpace1D>> spaces_;
};
```

Two constructors, and the first is the one the binding calls:

```cpp
/// Share the given 1D spaces.  This is what preserves the wrapper's identity contract.
explicit BsplineSpace(std::vector<std::shared_ptr<const BsplineSpace1D>> spaces);

/// Copy the given 1D spaces, for a C++ caller holding values rather than handles.
explicit BsplineSpace(std::span<const BsplineSpace1D> spaces);
```

**The `_ref` rule.** An owning accessor and a borrowing accessor exist side by side; the
borrowing one carries the `_ref` suffix and **is never bound**. Measured, g++ 14.4 `-O2`, one
pinned core, accessor bodies in a separate translation unit so the call is real: `space_ref(d)`
costs **5.83 ns** per access and `space(d)` costs **14.92 ns**, a difference of **9.1 ns** which
is the uncontended atomic increment/decrement pair. Nine nanoseconds is nothing next to a 200 ns
Python call and everything next to a de Boor step, so the pair is load-bearing rather than
decorative. Enforce "never bound" with a test over the bound surface -- assert that no method
name exposed by `pantr._pantr_cpp` ends in `_ref` -- because there is no `static_assert` for
absence and review alone will not hold across nine tickets.

### The binding rules, in the form a binding author needs

```
class V (constructs)     .def("insert_knots", &BsplineSpace1D::insert_knots, nb::arg(...))
                         -> no policy.  infer_policy resolves a by-value return to `move`.
                            Declaring reference_internal here is a SILENT NO-OP for lifetime
                            (measured: the policy is downgraded to `move` and no keep-alive
                            is installed), so do not write it.

class H (holds)          .def_prop_ro("spaces", &BsplineSpace::spaces_shared)
                         -> no policy.  The value is a shared_ptr; ownership travels in the
                            return value.  #include <nanobind/stl/shared_ptr.h>.

class A (array view)      .def_prop_ro("knots", [](nb::handle self) { ... })
                         -> nb::ndarray<const T, nb::numpy, nb::ndim<N>>(ptr, shape, self).
                            `const T` is what sets the read-only flag; `self` is what keeps
                            the storage alive.  Precedent: cpp/bindings/bezier_type.cpp:43-56.
```

For class H the binding hands back `std::shared_ptr<const T>`, not `const T&`, so nanobind's
`shared_ptr` caster runs and no policy question arises. Measured: the returned Python object is
identity-stable while alive (`so.at(0) is so.at(0)` is `True`, because `nb_type_put` looks the
pointer up in `inst_c2p` before creating an instance -- `nb_type.cpp:2077-2116`), the C++
`use_count` goes from 1 to 2 on the way out, and the element reads correctly after the owner is
destroyed with only the owner's own destructor having run.

### The wrapper's shape

`#387`'s pattern verbatim, with one addition. Each wrapper declares
`__slots__ = ("_impl", "_spaces", ...)`, a raising `__setattr__`/`__delattr__`, and fills memo
slots through `object.__setattr__` (`src/pantr/grid/_tensor_product_grid.py:660,704-731` on
`feat/387-tensor-product-grid`).

The addition is **seeding**, and it is what satisfies F6:

```python
def __init__(self, spaces):
    validated = _validated_spaces(spaces)          # a tuple of BsplineSpace1D wrappers
    self._take(_impl_class()([s._impl for s in validated]), spaces=validated)

@classmethod
def _wrap(cls, impl, *, spaces=None):
    """Adopt an already-valid implementation, optionally with its nested wrappers."""

def _take(self, impl, *, spaces=None):
    object.__setattr__(self, "_impl", impl)
    object.__setattr__(self, "_spaces", spaces)    # None => build lazily on first read
```

Two rules follow from it, and both must be written into the wrapper's docstring because neither
is discoverable:

- **Seed from the constructor.** A wrapper built from Python arguments keeps those arguments'
  wrappers, so `space.spaces[0] is space_1d` holds. A wrapper built by `_wrap` from a C++ handle
  seeds `None` and builds the tuple on first read, because there is nothing for it to be
  identical to.
- **Propagate when the nested object is known unchanged.** A method that returns a derived object
  sharing this one's nested object passes its own wrapper down:
  `type(self)._wrap(new_impl, space=self.space)`. In the current suite there is exactly one site
  that needs this, `Bspline.transform` (non-in-place), pinned by
  `tests/test_transform.py:634`. Every other `is` assertion in the suite is constructor
  identity, which seeding covers: `tests/test_bspline_space.py:89`,
  `tests/test_bspline.py:22,241`, `tests/test_multilevel_extraction.py:103`,
  `tests/test_quasi_interpolation.py:314`, `tests/test_grid_hierarchical.py:199`
  (`assert g.root is root`, which is #395's), `tests/test_mpi_collocation.py:188,202`,
  `tests/test_mpi_qi.py:153`, `tests/test_mpi_l2.py:251`, `tests/test_mpi_thb_qi.py:151`.
  The three `dfn.local.space is ds.local.space` assertions
  (`tests/test_mpi_qi.py:164`, `test_mpi_l2.py:262`, `test_mpi_thb_qi.py:162`) compare two Python
  `LocalSpace` records and are untouched by any of this.

There is no reference cycle to worry about under class H: the child holds no Python reference to
the owner, so the owner's wrapper -> child wrapper -> child handle chain is acyclic and plain
reference counting collects it. That is a second, quieter advantage over `reference_internal`,
where the child handle does hold the owner handle.

`__reduce__` reconstructs from the public arrays and the nested wrappers, never from the handle
-- the rule commit `11f22a7` established for the affine map. Sharing survives a single pickle for
free, because `pickle` memoises: dumping `(b, b.space)` restores a pair that shares one space.
Sharing does **not** survive two independent `dumps` calls, which is also true today.

**`__reduce__` is an addition here, not a change, and every one of the nine tickets owes one.**
Verified: `grep -rn "__reduce__\|__getstate__\|__setstate__" src/pantr/bspline/` at `a45e935`
returns nothing -- not one class in the module defines any of them, so every type pickles today
through the default protocol over its `__dict__` or its `__slots__`. The moment a slot holds a
nanobind handle that default fails, and it fails at `dumps` time with a `TypeError` about the
handle rather than anywhere near the design decision that caused it. The milestone's cross-cutting
requirement already asks for a round-trip per type under both backends; what this note adds is
*what the reduction must contain*: the public arrays and the nested **wrappers**, so that the
identity contracts of F6 survive the round trip through the seeding rule rather than by accident.

## The failure modes, and the test that catches each

Every one of these is silent. That is the point of the section: the design is cheap to get right
and its wrong versions do not announce themselves.

| # | mistake | what a user sees | what catches it |
|---|---|---|---|
| M1 | class H bound as `const T&` with no policy | a fresh copy per access; `space.spaces[0] is space_1d` **fails**; a memoised derived cache (see the companion note) is cold on every access, so a loop recomputes it every iteration | the identity assertion already in `tests/test_bspline_space.py:89`, and a parity test that the same access twice is the same object |
| M2 | class H bound as `const T&` with `reference_internal` "because the grid does it" | works today, dangles for the interpreter-free C++ consumer; and for `Bspline`, an escaped `space` silently starts reporting the *new* space after `reverse(in_place=True)` | a C++ unit test that outlives the owner: build the owner in a scope, take the handle out, let the owner die, read. Under `shared_ptr` it passes; under a raw reference ASAN reports the use-after-free. Plus a Python test asserting `sys.getrefcount(owner._impl)` is **unchanged** by the access -- the delta-0 assertion is what pins "no keep-alive was installed", i.e. that nobody quietly reverted to `reference_internal` |
| M3 | `keep_alive<0, 1>` added to fix a dangle | the dangle stops, the copy stays, the write is still lost (F2) | assert the *aliasing* separately from the *lifetime*: `np.shares_memory(space.spaces[0].knots, space.knots_along(0))`. A copy breaks memory sharing even though the values agree |
| M4 | `reference_internal` declared on a class-V accessor | nothing; it is a complete no-op. Measured on the pair `copy_at` / `copy_at_refint`: refcount delta 0, no aliasing, owner destroyed, in both | a lint over the bindings: no `rv_policy` on a method whose C++ return type is not a reference or pointer. Cheap to write, and it is the only detector, because the runtime behaviour is identical |
| M5 | class A bound with a non-`const` scalar | the view comes back **writeable**, so a caller can mutate validated geometry from outside | `assert not arr.flags.writeable`, which `pantr.bezier` already asserts; plus a mutation attempt inside `pytest.raises(ValueError)` |
| M6 | class A bound with no owner argument | nanobind **copies** silently and the array comes back writeable, so both M5's symptom and a lost aliasing contract | `np.shares_memory(...)` against the owner's storage. `grid_types.cpp:116-121` on `feat/387` already records this exact trap; `tests/test_spanwise_element_extraction.py:1681` is the assertion shape |
| M7 | the C++ constructor copies its nested spaces instead of sharing | two Python objects that compare identical over two different C++ objects (F6) | a parity assertion that `space.spaces[0]._impl is space_1d._impl` under the C++ backend |
| M8 | the wrapper builds its `spaces` tuple lazily and forgets to seed from the constructor | `space.spaces[0] is space_1d` fails; values all agree | `tests/test_bspline_space.py:89`, unedited |
| M9 | a `_ref` accessor gets bound | a dangling reference reachable from Python with no policy anywhere to blame | the bound-surface test: no exported method name ends in `_ref` |

**Where these tests live.** `tests/parity/test_bezier_binding_contract.py` is the established
shape -- a parity file whose subject is what the *binding* guarantees rather than what the
mathematics does, gated on a `cpp_backend` fixture, with one test per refusal and the silent
failure it prevents named in the docstring. Each of #395 to #400 owes a
`tests/parity/test_<type>_binding_contract.py` section carrying M1 to M9's assertions for the
accessors it adds. That file is also the natural home for the bound-surface test (M9), since it is
already the file that asserts things about the extension rather than about a result.

**The one test shape to require per held accessor**, because it is deterministic, needs no C++
instrumentation, and fails on the design error rather than on the weather:

```python
def test_space_accessor_shares_rather_than_pins(...):
    owner = BsplineSpace([s1d_a, s1d_b])
    impl = owner._impl
    before = sys.getrefcount(impl)
    child = owner.spaces[0]
    assert sys.getrefcount(impl) == before      # no keep-alive: the value is shared, not pinned
    assert child is s1d_a                       # constructor identity survives
    del owner
    gc.collect()
    assert child.num_basis == expected          # the value outlived its owner
```

The first assertion is the one that would not be written without this note, and it is the one
that catches M2. The third one alone is the test F4 shows can pass on a broken design.

## What it costs at the call site

Measured on this machine, 2026-08-31, nanobind 2.14.0, CPython 3.14.6, g++ 14.4 `-O2`; 20000
accesses per timing, best of 5. The nested object holds `width` doubles.

| accessor shape | width 4 | 64 | 1024 | 16384 |
|---|---|---|---|---|
| `.def` + default policy (copies) | 198 ns | 206 ns | 322 ns | 6990 ns |
| `.def` + `reference_internal` | 213 ns | 205 ns | 216 ns | 212 ns |
| `def_prop_ro` (property) | 225 ns | 212 ns | 227 ns | 217 ns |
| by-value accessor (`copy_at`) | 223 ns | 233 ns | 349 ns | 7547 ns |
| container of 3, default policy | 409 ns | 430 ns | 1082 ns | 109040 ns |
| container of 3, `reference_internal` | 447 ns | 448 ns | 461 ns | 443 ns |
| `shared_ptr<const T>` accessor | 224 ns | -- | 233 ns | 229 ns |

Three things to read out of it, and one to read out of the companion note:

- **At the Python boundary the choice is free below about 1000 doubles** -- everything is the
  ~200 ns call overhead -- and it is a factor of 33 (scalar) to 246 (container of three) above
  it. So the copy option is not merely unfashionable; on a knot vector of any size it is a
  measurable per-access cost that no later tuning removes.
- **`shared_ptr` is indistinguishable from `reference_internal` at the Python boundary.** The
  atomic pair is 9.1 ns against a 200 ns call. Choosing the safer mechanism costs nothing here.
- **Inside C++ the atomic pair is 9.1 ns against a 5.8 ns borrow**, which is why `space_ref`
  exists and is not bound.
- **The copy option's real cost is not in this table.** A by-value accessor hands back an object
  with a *cold* derived cache, so `for d in range(dim): space.spaces[d].num_intervals` recomputes
  the memo on every iteration instead of once per space. That is a shape change, not a constant
  factor, and it is the argument that kills the copy option outright rather than on price. See
  `design/bspline_derived_caches.md`.

## What each ticket in the milestone inherits

- **#395 `HierarchicalGrid`.** One held accessor, `root` -> `TensorProductGrid`
  (`_hierarchical_grid.py:477-484`): store `std::shared_ptr<const TensorProductGrid<double>>`.
  Its own `cell_tags` / `facet_tags` / `cell_bvh` are **not** class H -- they are the grid mixin's
  own members and keep `reference_internal` exactly as #386/#387 have them, because a tag
  registry must alias (it is the accumulating-container exception, `tags.hpp:14-19`). So #395 is
  the one ticket in the milestone that uses both mechanisms, and the F2 trap is live in it.
  Also: F1 means its `cell_bvh` binding is the one where the explicit policy is load-bearing, and
  its two tag properties are the ones where it is documentation.
- **#396 `BsplineSpace1D` and `BsplineSpace`.** The whole of the H rule, plus F6's two
  constructors and the seeding rule. `BsplineSpaceRestriction` stays a Python `NamedTuple` whose
  `space` field holds a freshly built wrapper -- class V, nothing to do. Its `AC` set should
  include M1, M7 and M8 by name; each is one assertion.
- **#397 `THBSplineSpace`.** Three held accessors (`grid`, `root_space`, `level_space`), one of
  which is the `const` exception. `level_space(level)` returns a `shared_ptr` from a
  `std::vector<std::shared_ptr<const BsplineSpace>>`; the wrapper memoises per level in a
  `dict[int, BsplineSpace]` memo slot, which is the one place a dict memo is right here because
  the key is genuinely data. `THBSplineSpaceRestriction.space` is class V.
- **#398 `Bspline`.** One held accessor (`space`), the reseat semantics of reason 3 above, the
  one propagation site (`transform`), and **the largest downstream exposure in the milestone**;
  see the next section. Its `control_points` is class A and its writability is a genuine break --
  flagged below rather than decided here.
- **#399 the extraction machinery.** `SpanwiseElementExtraction.space` is the only class-H
  accessor; `ExtractionStructView` is class A throughout (F5), so what it needs is
  `bezier_type.cpp`'s view idiom and `tests/test_spanwise_element_extraction.py:1681`'s
  `shares_memory` assertion, not this note's H rule. Its `compact_ops_1d` arrays are already
  frozen read-only at `spanwise_element_extraction.py:188-190`, so the `const T` scalar matches
  today's contract with no behaviour change.
- **#400 `THBSpline` and `MultiLevelExtraction`.** Two held accessors, both plain. The deferred
  distribution cluster keeps `LocalSpace` as a Python `NamedTuple` whose `space` field holds a
  wrapper -- class V by F5, so the deferral costs this note nothing.
- **#401 scaffolding removal.** The `_adopt` helper
  (`src/pantr/grid/_grid.py:744-768` on `feat/387`) exists only because the Python oracle exists,
  and the bspline wrappers will grow their own copies of it. Name them in #401's list on the way
  in, not on the way out.

## Alternatives rejected

**`rv_policy::reference_internal` throughout, matching the grid.** The cheapest thing to write,
the most consistent-looking, and it works today. Rejected on reason 1 above: the guarantee lives
in the binding, and on reason 3, which is a wrong answer rather than a dangling pointer.
**What would change it:** a ruling that the interpreter-free C++ consumer is hypothetical after
all. It is not -- it is the premise of the 2026-08-27 amendment.

**Return the nested object by value and accept the copy.** No lifetime question at all, and for a
`BsplineSpace1D` of a dozen knots the copy is genuinely free (measured: 198 ns against 213 ns at
width 4). Rejected on three grounds, in increasing order of weight: it is a factor of 33 at
16384 doubles; it breaks `tests/test_bspline_space.py:89` and `tests/test_transform.py:634`
outright, because a copy cannot be the object the caller passed in; and it hands back an object
with a cold derived cache, turning a memo into a per-access recomputation.
**What would change it:** nothing available. Even dropping both identity assertions leaves the
cold-cache problem.

**`std::unique_ptr<const T>` plus a raw borrowing accessor.** Expresses single ownership honestly
and costs no atomics. Rejected because it cannot satisfy `tests/test_transform.py:634` at all: two
`Bspline`s sharing one space is the contract, and unique ownership forbids it.

**An intern table -- a `WeakKeyDictionary` from handle to wrapper -- so that wrapping the same
handle twice yields the same wrapper.** This would make F6b automatic and remove the propagation
rule. Rejected as too much machinery for one site: it needs `nb::is_weak_referenceable()` on
every bound class, adds a weak-map lookup to every nested access, and introduces a failure mode
(a wrapper resurrected from the table with a different backend's handle) that the explicit form
cannot have. **What would change it:** propagation sites multiplying past a handful. There is one
today.

**Store `shared_ptr<const T>` but hand out `const T&` with `reference_internal`.** The
apparent best of both: ownership in the type, no atomic per access, and a policy that pins the
owner. Rejected on the measurement in reason 3's table: reseating the pointer frees the pointee
while the escaped Python object still aliases it, and the escapee then read the *correct* value,
so nothing announces it. The keep-alive pins the **owner**, which is not what needs to stay
alive.

**`std::enable_shared_from_this` on the nested types, with `reference_internal` on the binding.**
nanobind supports it -- `nb_type_put_common` checks `has_shared_from_this` and takes out a second
`shared_ptr` sharing ownership (`nb_type.cpp:2024-2030`) -- so this would give the shared-lifetime
behaviour *through* a reference-returning accessor. Rejected because it makes the safety depend on
the owner having created the object through a `shared_ptr` in the first place, which nothing in the
type enforces: a stack-allocated `BsplineSpace1D` would `throw std::bad_weak_ptr` at the first
access. Storing the `shared_ptr` is the same guarantee with the failure removed.

## What this design changes for the downstream consumer

`pantr.bspline` is that repository's largest exposure, and pantr's CI cannot see it. Flagged
explicitly rather than assumed free. Nothing here was checked against that repository; it is not
this note's to read.

**No signature and no name changes.** Every accessor keeps its name, its arity and its return
type as Python sees it. `BsplineSpace.spaces` still returns a tuple, `Bspline.space` still returns
a `BsplineSpace`.

**Four behaviour changes, all of them consequences of the port rather than of this note's choice
between mechanisms:**

1. **A nested object handed out is no longer the same *C++* object as an independently constructed
   equal one**, and `__eq__` does not exist on any of these classes to paper over it -- verified:
   none of `BsplineSpace1D`, `BsplineSpace`, `THBSplineSpace`, `Bspline`, `THBSpline`,
   `TensorProductGrid`, `HierarchicalGrid` defines `__eq__` or `__hash__`. A consumer comparing
   spaces with `==` is comparing identity today and will still be comparing identity after. No
   change, but worth stating because it looks like the kind of thing a port would alter.
2. **`Bspline(space, cp)` copies `cp`.** The C++ value type copies at construction, so
   `cp[0] = ...` after construction stops being visible. This is not new policy: the same
   decision is already shipped for `Bezier` and recorded at
   `tests/test_bspline_conversion.py:22-32`, which skips `test_copy_false` under the C++ backend
   with the reason *"the C++ value copies its control points at construction, so `copy=False`
   shares nothing under that backend"*. #398 inherits both the decision and the skip idiom.
3. **`Bspline.control_points` becomes a read-only view.** `Bezier`'s precedent
   (`src/pantr/bezier/_bezier.py:473-484`) is explicit: under the C++ backend it is a read-only
   view of the object's own storage, writing raises, and it stays valid after the object is
   dropped. Today `Bspline.control_points` is the live writable array -- verified by execution:
   `b.control_points is b._control_points` is `True` and `b.control_points[0,0] = 99` changes the
   object. **This is the one change in the milestone that will break working downstream code
   silently-turned-loud** (an `IndexError`-free `ValueError: assignment destination is read-only`
   at the write). It is #398's to schedule and the user's to approve; this note only records that
   the `Bezier` precedent settles the *shape* and not the *timing*.
4. **`THBSplineSpace` may stop raising `RuntimeError: THBSplineSpace is stale`.** Once #378 makes
   refinement return a new grid, the staleness the version counter detects cannot arise from
   pantr's own API. Whether the check and its exception stay as a guard against a hand-mutated
   grid is #378's and #397's call; a consumer catching that `RuntimeError` should be told either
   way.

## Bad practices flagged, for the user's inspection rather than for fixing here

- **`grid_types.cpp`'s and `grid_hierarchy_port.md` W1's `rv_policy` paragraph is right about the
  mechanism and wrong about the site** (F1). Both name `cell_tags` -- a `def_prop_ro`, where
  `reference_internal` is already the default -- as the case where the default policy would lose
  a write. The behaviour of the shipped code is correct; the sentence teaches a rule that is
  false for half the bindings it will be applied to. **Cost of keeping:** an engineer on #395
  reads it, concludes "lvalue reference implies name the policy", and either adds noise to
  properties (harmless) or, believing the converse, omits it from a `.def` (a silent copy). **Cost
  of fixing:** two comment edits and one sentence in the design note, no behaviour change. Fixing
  it is a `docs(cpp)` commit that belongs to whoever next opens either file.
- **`src/pantr/grid/_grid.py`'s `cell_bvh` docstring says concurrent first calls "may each build a
  valid tree and one write wins, costing redundant construction."** True of the Python
  implementation. **False of the C++ one**, where two threads writing one `std::optional` is a
  data race and therefore undefined behaviour, not a lost update. The claim is a Python
  implementation detail written into the contract of a method both backends implement. Argued and
  measured in `design/bspline_derived_caches.md`; recorded here because the sentence sits in a
  file #395 will edit.
- **`Bspline.control_points` hands out the live writable array with no defensive copy** while
  `THBSpline.control_points` freezes its at construction
  (`src/pantr/bspline/_thb_spline.py:84`). Two sibling types, opposite answers, and the mutable
  one is the type whose `_beziers_cache` and `_locate_cache` are invalidated only by the three
  sanctioned `in_place=` methods -- so a write through `control_points` desynchronises both caches
  with nothing raising. That is a defect in today's Python, not something the port introduces, and
  the port is what will remove it (change 3 above). Worth a ticket so the removal is deliberate
  rather than a side effect.

## Epistemic status

- **Measured, 2026-08-31, on this machine, nanobind 2.14.0 / CPython 3.14.6 / g++ 14.4 `-O2`:**
  every row of F1's, F2's, F3's and F4's tables; that `reference_internal` on a by-value return
  is a no-op, on the pair `copy_at` / `copy_at_refint`; that `sys.getrefcount` on the owner moves
  by exactly 1 under a keep-alive and 0 without; that a `weakref` on a nanobind instance raises
  unless the class is annotated; that a bare `rv_policy::reference` escapee read the correct value
  in the scalar case and garbage in the container case; the seven-row cost table; the 5.83 ns
  against 14.92 ns `space_ref` / `space` difference. The harness is described under "What was
  compiled" and is not committed.
- **Verified by reading nanobind's source at the installed version:** `infer_policy`'s
  lvalue-reference resolution (`include/nanobind/nb_cast.h:455-473`); `def_prop_ro`'s delegation
  and `def_prop_rw`'s positional `reference_internal` (`include/nanobind/nb_class.h:693-703`,
  `732-735`); `def_prop_rw_static`'s `rv_policy::reference` (`:712-729`); the element-wise policy
  forward (`include/nanobind/stl/detail/nb_list.h:60-67`); `nb_type_put`'s `inst_c2p` lookup and
  `nb_type_put_common`'s keep-alive, `destruct` flag and `shared_from_this` branch
  (`src/nb_type.cpp:1963-2050`, `2052-2124`).
- **Verified by reading the tree at `a45e935`:** the eight held accessors and the roughly
  thirty-one constructing ones, each at the line cited in F5; that `ExtractionStructView` has no
  `space` field; that no `__reduce__`, `__getstate__` or `__setstate__` exists anywhere in
  `src/pantr/bspline/`; that no domain class in `bspline` or `grid` defines `__eq__` or
  `__hash__`; the shared-`Tag` rationale at `cpp/include/pantr/grid/tags.hpp:14-29`; the array-view
  rationale at `cpp/bindings/bezier_type.cpp:43-56`; the CI Python matrix.
- **Verified by reading `feat/387-tensor-product-grid` at `d7b8654` through `git show`:** the
  wrapper pattern, `_adopt`, the memo slots, the raising `__setattr__`, the `view_of` owner
  argument, and the four `rv_policy` sites. That worktree was not entered.
- **Verified by execution in the `pantr` env:** that `b.space` changes identity across
  `reverse(direction=0, in_place=True)`; that `b.control_points is b._control_points` and the
  array is writable.
- **Measured, 2026-08-31:** the three-row reseat table in reason 3, with a destructor counter
  in the nested type, so the middle row's use-after-free is a count and not an inference.
- **Asserted, not measured:** that a raw-reference version of a class-H accessor is a
  use-after-free for a C++ consumer. The Python-side analogue was measured (F4); the C++ analogue
  follows from the object model and was not put under ASAN, because the design does not contain
  the construct.
- **Not investigated:** whether the not-yet-public downstream consumer writes through
  `Bspline.control_points`, or catches `THBSplineSpace`'s staleness `RuntimeError`. Both are owed
  before #398 and #397 land. Also not investigated: whether `SpanwiseElementExtraction`'s
  `ops_1d` decompression is on any consumer's hot path, which would change #399's laziness answer
  rather than its lifetime answer.

## What was compiled, and how to reproduce it

Not committed: it exists to settle the mechanisms, and its home is the bindings and the parity
tests once the tickets are built.

Three nanobind extensions over a pantr-free stand-in pair -- `Inner` (a nested domain type
holding one `std::vector<double>`) and `Outer` (a holder with one `Inner` member and a
`std::vector<Inner>`) -- plus a `SharedOuter` holding `std::vector<std::shared_ptr<const Inner>>`.
`Inner` counts its own copy constructions and destructions and `Outer` its destructions, which is
what makes the copy and lifetime columns measurements rather than inferences. Eleven accessors
were bound over the same two C++ functions, differing only in `.def` versus `def_prop_ro` and in
the policy or `keep_alive` annotation, so that every row of every table above differs in exactly
one thing.

```bash
NBROOT=$(python -c "import nanobind,os;print(os.path.dirname(nanobind.__file__))")
PYINC=$(python -c "import sysconfig;print(sysconfig.get_paths()['platinclude'])")
g++ -O2 -std=c++20 -fPIC -fvisibility=hidden -shared -DNDEBUG \
    -I$NBROOT/include -I$NBROOT/ext/robin_map/include -I$PYINC \
    life.cpp $NBROOT/src/nb_combined.cpp -o life.so
```

The C++ accessor benchmark is a separate two-translation-unit build so the accessor bodies cannot
inline into the loop -- the same precaution `design/grid_hierarchy_port.md` records getting wrong
first time, for the same reason. `-O2`, no LTO, `taskset` to one core, best of seven over 1e8
iterations, the direction index varying so the branch is not constant-folded.

One thing to check before believing a small ratio here, by analogy with that note's
devirtualisation trap: if `space` and `space_ref` time the same, the `shared_ptr` copy was
elided. Both accessors must be defined in a translation unit the benchmark does not include, and
the object must be `const` so that neither body can be specialised on the loop.
