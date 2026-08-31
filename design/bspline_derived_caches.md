# Where a derived cache lives once the spaces are C++-owned

**Status:** proposed, 2026-08-31. Written for ticket #394, and binding on #395 to #401 if
adopted.
**Date:** 2026-08-31.
**Scope:** which side of the seam a memoised derived quantity lives on, whether the Python
wrapper keeps `__slots__`, how a lazy memo is made thread-safe, and the fate of the
process-global knot cache. Not what an accessor hands back, which is
`design/bspline_ownership_lifetime.md`. Not how the two backends are compared, which is
`design/backend_parity.md`.
**Companions:** `design/cross_backend_types.md` (the ownership ruling this obeys),
`design/grid_hierarchy_port.md` (whose `D-a` this adopts and whose lazy-BVH shape this
declines to copy), `design/bspline_ownership_lifetime.md` (the wrapper memo slots this note
authorises and bounds).

**Validated against:** `proto/cpp` at `a45e935`, worktree `design/393-394-bspline-decisions`,
plus `feat/387-tensor-product-grid` at `d7b8654` read through `git show`. Python timings on the
`pantr` env, CPython **3.14.6**, GIL enabled. C++ timings g++ **14.4.0** `-O2`, one pinned core.
TSan is g++ 14.4's `-fsanitize=thread`.

## The decision in one paragraph

**A derived quantity of a domain type's frozen state lives in the C++ type. The Python wrapper
memoises presentation and nothing else** -- the wrapper object standing in front of a nested
C++-owned type, per `design/bspline_ownership_lifetime.md` -- and it keeps `__slots__` with a
raising `__setattr__`; **`functools.cached_property` does not survive the port and the wrapper
does not relax.** That is not "both", because the two memos hold different kinds of thing: the
C++ memo holds the value, the wrapper's slot holds the Python object presenting it, and there is
exactly one value. **The fourteen `cached_property` sites do not become fourteen
`mutable std::optional`s**: seven of them (all of `BsplineSpace`'s) stop being memos at all, six
become ordinary eager fields, and the one that allocates joins the other allocating derived
arrays in **a single grouped memo behind one flag**. A lazy memo is guarded by
**double-checked locking over an `std::atomic<bool>`** -- not a bare `mutable std::optional`,
which is a data race demonstrated under TSan, and not `std::call_once`, whose first call was
measured at about 1.6 microseconds, ten times the whole construction of a small space. The
`lru_cache(128)` at `src/pantr/bspline/_bspline_space_1d.py:40` **is deleted as part of #396**,
because it carries the same latent backend-collision defect that
`src/pantr/_backend.py:525-555` already documents for a sibling, and its measured saving is a
factor of 1.4 to 2.1 against a Numba dispatch the port removes anyway.

## Findings against the ticket as written

### F1 (important). The count is right and the framing is not: seven of the fourteen are not caches

Verified by reading both classes. `BsplineSpace1D` has 7 `cached_property` and `BsplineSpace`
has 7, so the ticket's fourteen is exact. (`src/pantr/bspline/` has 16 in total; the other two
are on `SpanwiseElementExtraction`, outside the ticket's scope and inside #399's.)

But **all seven of `BsplineSpace`'s are O(dim) reductions over its own children**, with `dim`
at most 3 in every use in the tree: `degrees`, `tolerance`, `num_basis`, `num_total_basis`,
`num_intervals`, `num_total_intervals`, `domain`
(`src/pantr/bspline/_bspline_space_nd.py:76,85,108,117,126,135,144`). Each reads a scalar off
each 1D space and combines them. In C++ they are either fields set in the constructor or
three-iteration loops in the accessor. **They are memos only because a Python attribute read
that walks three objects costs more than a `__dict__` hit** -- measured, 56 to 60 ns for a warm
`cached_property` hit -- which is a fact about CPython, not about the mathematics.

So: **`BsplineSpace` acquires zero memos.** That removes half the problem before any mechanism
is chosen, and it should be said in #396's acceptance criteria, because "port fourteen memos" and
"port one" are different tickets.

### F2 (critical). The GIL is not available as the memo's protection, and the tree already proves it

The obvious reasoning -- nanobind holds the GIL for the duration of a bound call, so a memo
filled inside one is serialised against every other Python thread -- is true of a call that does
not release it. **The extension releases the GIL at 19 sites already**
(`basis.cpp` 1, `change_basis.cpp` 3, `quad.cpp` 5, `bezier.cpp` 10, counted under `cpp/bindings/` at `a45e935`),
and `src/pantr/_backend.py:551-552` states the intent in the same breath as the hazard: *"use_backend
is scoped per thread precisely so callers may thread, and the extension releases the GIL to invite
it."*

Every one of those 19 is a kernel binding rather than a method of a domain type, so **no bound
method releases the GIL today**. That is the honest state. It is also not a place to build on: the
method that fills the biggest of these memos, `get_unique_knots_and_multiplicity`, was measured
at 16.9 microseconds on a 2055-knot vector, and a 17-microsecond bound call holding the GIL is
exactly what gets a `gil_scoped_release` at the first performance pass. A memo whose safety
depends on nobody adding one line to its binding is not safe.

And the ruling settles it regardless: *"En el futuro, solo C++."* There is no GIL in the future
this port is for.

**Numba is not the hazard here, and the ticket's framing should be corrected.** A
`parallel=True` kernel runs with the GIL released, but a `nopython` kernel receives arrays and
never calls into the extension, so those threads cannot reach a C++ memo. The hazards are, in
order of likelihood: a Python-level thread pool over Layer 1 (serialised by the GIL only while
no binding releases it), a `gil_scoped_release` added to a space method, a free-threaded CPython
build (no CI leg today: `.github/workflows/ci.yaml:20,80,131` list `3.11`, `3.13`, `3.14`), and
the interpreter-free C++ consumer, where nothing is serialised by anything.

### F3 (critical). A bare `mutable std::optional` memo is a data race, and no value test will find it

Measured. A struct with `mutable std::optional<std::vector<double>> memo` filled on first const
access, hammered by 8 threads:

- **Under TSan: 4 data races reported, every stack frame in the unsynchronised accessor.** The
  first is a read of the `optional`'s engaged flag racing a write of it; the rest are on the
  contained `vector`'s pointer triple. Zero races reported for the `call_once` and eager variants
  in the same binary.
- **Without TSan: 60 runs of 8 threads each produced the correct answer 60 times.**

So the failure is real -- concurrent conflicting non-atomic accesses are a data race and
therefore undefined behaviour under the C++ standard's `[intro.races]` rule -- and it is
invisible to every test that checks a value. It is not a "lost update, costing redundant
construction": two threads assigning one `std::optional<std::vector<double>>` can leave a torn
pointer triple, double-free the loser's buffer, or leak it.

**Two consequences for the tree.** `cpp/include/pantr/grid/grid.hpp`'s
`mutable std::optional<BVH<scalar_type>> bvh_` is that construct. And
`src/pantr/grid/_grid.py`'s `cell_bvh` docstring says concurrent first calls *"may each build a
valid tree and one write wins, costing redundant construction"* -- which is a correct description
of the **Python** implementation, where the GIL makes the write atomic, written into the contract
of a method both backends implement. Both are recorded under "Bad practices flagged"; neither is
this note's to fix.

### F4 (important). Laziness is nearly free if it is done right, and eager is nearly free only for what does not allocate

Measured, g++ 14.4 `-O2`, one pinned core, constructor bodies in a separate translation unit,
best of 5 over 2e5 (or 2e4 for the largest) iterations. Each column constructs a fresh object and
then uses one thing from it. `p = 3` throughout; `n_knots` is the whole vector.

| n_knots | construct only | + eager scalars | + eager arrays | `call_once`, then one use | DCLP, then one use | DCLP, hot path per access |
|---|---|---|---|---|---|---|
| 11 | 36 ns | 42 ns | 173 ns | 1793 ns | 188 ns | 7.99 ns |
| 39 | 62 ns | 69 ns | 446 ns | 1955 ns | 484 ns | 4.44 ns |
| 263 | 588 ns | 535 ns | 3178 ns | 4620 ns | 3165 ns | 2.94 ns |
| 2055 | 3326 ns | 3465 ns | 18058 ns | 24323 ns | 22025 ns | 2.93 ns |

These are construct-and-destroy microbenchmarks, so they are allocator-sensitive: a second run of
the same binary moved every absolute number by up to a factor of two while leaving every ratio
within a row unchanged. **Read the ratios, not the absolutes.**

- **Eager scalars are free.** Within noise of bare construction at every size, in both directions
  (`535 < 588` and `3465 > 3326` are the same noise). What they cost is two bounded scans of at
  most `degree + 1` knots each, plus arithmetic.
- **Eager arrays cost 4.8x to 7.2x the bare construction**, because they are three more heap
  allocations, and at small `n` the allocations, not the scan, are the cost. My own first estimate
  -- "a small constant factor on an already-O(n) constructor" -- was wrong, and the measurement is
  what refuted it.
- **`std::call_once` costs about 1.6 microseconds on its first call**, and the cost is
  size-independent (1793 - 173 = 1620 ns at n=11; 1955 - 446 = 1509 ns at n=39), which is the
  signature of glibc's `pthread_once` issuing a `FUTEX_WAKE` after running the initialiser even
  with no waiters. At n=11 that is **ten times the entire construction**.
- **Double-checked locking over an `atomic<bool>` costs, on first use, within noise of computing
  eagerly** (188 vs 173, 484 vs 446, 3165 vs 3178, 22025 vs 18058) **and 2.9 to 8.0 ns per access
  thereafter.** It is free when the derived data is never touched and free when it is.

So the rule is not "eager because laziness is dangerous". It is: **eager where there is nothing to
allocate, DCLP-lazy where there is, and never `call_once`.**

### F5 (important). The `lru_cache`'s own key handling is the same order as the computation it caches

`_cached_unique_knots_and_multiplicity` (`_bspline_space_1d.py:40`, `maxsize=128`) is keyed on
`(self._knots.tobytes(), dtype.str, size)`, built fresh at every call site
(`_bspline_space_1d.py:320`). So a cache **hit** pays an O(n) buffer copy, an O(n) hash of a
never-before-seen `bytes`, and an O(n) comparison against the stored key. Measured in the `pantr`
env, warm cache:

| n_knots | `tobytes` + tuple, alone | kernel called directly | the cached method (a hit) | what the cache saves |
|---|---|---|---|---|
| 11 | 671 ns | 4303 ns | 2384 ns | 1.8x |
| 39 | 605 ns | 4743 ns | 2214 ns | 2.1x |
| 263 | 1487 ns | 8075 ns | 4247 ns | 1.9x |
| 2055 | 2738 ns | 23051 ns | 16894 ns | 1.4x |

A factor of 1.4 to 2.1, against a kernel whose 4.3 microseconds for an eleven-element scan is
Numba dispatch overhead rather than work. **The port removes the thing the cache is hiding.** For
comparison, the same three arrays computed in C++ inside the constructor cost 137 ns at n=11 and
about 14.7 microseconds at n=2055 (the eager-arrays column minus the construct-only column above),
and that is with the allocations included.

The saving is real but small, it is bought with a process-global 128-entry cache, and it is
strictly dominated by owning the derived data per object: once `BsplineSpace1D` holds it, the
scan runs **once per space** rather than once per call, with no key to build.

### F6 (critical). The `lru_cache` carries the exact latent defect a sibling cache was already fixed for, and #396 is when it becomes live

`src/pantr/_backend.py:525-555` documents `backend_keyed_cache` and why a plain `lru_cache` was
not enough for `_cached_lagrange_to_bernstein_matrix`: the key omitted the backend, so *"the first
backend to populate an entry serves every later caller"*, with a measured consequence -- a parity
test handed the same object twice, reporting agreement it never measured, with `A is B` true and
the matrices differing only after a cache clear.

`_cached_unique_knots_and_multiplicity` has the same key shape and the same omission. It is
**not** a live bug today: `_get_unique_knots_and_multiplicity_impl`
(`src/pantr/bspline/_bspline_knots.py:286`) is a plain Numba function with no backend dispatch,
verified by reading its definition. It becomes live the moment #396 gives that scan a C++
implementation, which is #396's whole point.

**So the removal is not cleanup after the port; it is a step inside #396, ahead of the C++
dispatch.** Doing it the other way round means a parity test that cannot fail.

## The decision

### Where each kind of thing lives

| what | lives | mechanism | why not the other side |
|---|---|---|---|
| a derived quantity of the type's own frozen state | **C++** | an eager field, or a member of the single grouped lazy block | the interpreter-free consumer needs it; a Python-side memo is a memo the ruling deletes |
| the Python wrapper object for a nested C++-owned domain type | **the wrapper** | a named `__slots__` memo slot, filled through `object.__setattr__` | C++ cannot hold a `PyObject*` without inverting the dependency, and the identity contracts are Python-level facts |
| a numpy view over C++ storage | **neither** | constructed per access from a `std::span`, with the owner as the array's owner | it is not a cache; it is a presentation, and it is cheap (`bezier_type.cpp:43-56`) |
| anything else | **nowhere** | -- | see "What must never appear" |

**Nothing is cached on both sides**, and the reason is worth stating precisely because "both" is
the trap the ticket names: the C++ memo holds a **value**, the wrapper's slot holds the **Python
object presenting that value**. There is one value. A wrapper slot holding a *number* or an
*array copy* would be the second truth, and it is forbidden -- which is exactly what dropping
`cached_property` enforces, since `cached_property` caches values and nothing else.

### The wrapper keeps `__slots__`. `cached_property` does not survive

`__slots__` naming every memo slot, a raising `__setattr__` and `__delattr__`, and
`object.__setattr__` for the memo fills -- `feat/387`'s
`src/pantr/grid/_tensor_product_grid.py:660,704-731` verbatim. Three reasons, and the first is
the one that decides it:

1. **The memo slots have to be named, because must-pass tests read them.**
   `design/grid_hierarchy_port.md` W3 records `tests/test_grid_tags.py:149-150` reading
   `g._cell_tags` and `g._facet_tags` directly and asserting they are `None` before first use, in
   a file on #387's must-pass-unedited list. A `__dict__` gives you `cached_property` and takes
   away the guarantee that the slot exists under a known name before it is filled.
2. **A `__dict__` silently returns settable attributes to a type documented immutable.** It is
   the same failure `design/grid_hierarchy_port.md` F3 measured for a `Protocol` missing
   `__slots__ = ()`: nothing in the suite asserts the absence of `__dict__`, so the regression is
   invisible.
3. **`cached_property` caches values**, which is the second truth. Its per-hit cost, 56 to 60 ns
   measured, is also not an argument for it: a forwarded C++ field read is the same order.

The cost of keeping `__slots__` is that each memo is three lines instead of a decorator, and that
`mypy` needs a `# type: ignore[misc]` on the `object.__setattr__` fill. #387 already pays both.

### What the fourteen become

`BsplineSpace1D`, 7 sites:

| today | becomes | why |
|---|---|---|
| `num_basis` (`:293`) | eager `std::int64_t` field | size arithmetic with a periodic branch; today it is a Numba call |
| `num_intervals` (`:325`) | eager `std::int64_t` field | one non-allocating count of distinct interior knots, **fused into the validation scan the constructor already runs** |
| `domain` (`:404`) | eager `std::array<double, 2>` | two indexed reads |
| `_left_end_open` (`:421`) | eager `bool` | a bounded scan of at most `degree + 1` knots |
| `_right_end_open` (`:439`) | eager `bool` | as above |
| `_bezier_like_knots` (`:457`) | eager `bool` | derived from the three above |
| `_first_basis_per_interval_cached` (`:340`) | **the grouped lazy block** | allocates `O(num_intervals)` |

plus the public `get_unique_knots_and_multiplicity` (`:298`), whose two arrays join the same
block. `in_domain=True` is a subrange of `in_domain=False`, so the block stores the full pair once
and the in-domain form is a subspan -- one memo, two views, and no second flag.

`BsplineSpace`, 7 sites: **all seven become eager fields or three-iteration accessors, and none
becomes a memo** (F1).

So the milestone's memo count for the two space classes is **one**, not fourteen. Say so in
#396's acceptance criteria.

### The grouped lazy block, and its guard

```cpp
class BsplineSpace1D {
  public:
    /// The distinct knots and their multiplicities, over the whole vector.
    [[nodiscard]] std::pair<std::span<const double>, std::span<const std::int64_t>>
    unique_knots() const;

    /// The same, restricted to the domain: a subrange of `unique_knots()`.
    [[nodiscard]] std::pair<std::span<const double>, std::span<const std::int64_t>>
    unique_knots_in_domain() const;

    /// First supported basis index per interval.
    [[nodiscard]] std::span<const std::int64_t> first_basis_per_interval() const;

  private:
    /// Everything derived that allocates.  One struct, one flag, one fill.
    struct Derived {
        std::vector<double> unique;
        std::vector<std::int64_t> mult;
        std::vector<std::int64_t> first_basis;
        std::size_t domain_begin, domain_end;   ///< the in-domain subrange of `unique`
    };

    /// Fill `derived_` at most once, from any number of threads.
    const Derived& derived() const {
        if (!derived_ready_.load(std::memory_order_acquire)) {
            const std::lock_guard<std::mutex> guard(derived_mutex_);
            if (!derived_ready_.load(std::memory_order_relaxed)) {
                derived_ = build_derived(knots_, degree_, tol_);
                derived_ready_.store(true, std::memory_order_release);
            }
        }
        return *derived_;
    }

    std::vector<double> knots_;
    std::int64_t degree_, num_basis_, num_intervals_;
    /* ... the eager scalars ... */
    mutable std::mutex derived_mutex_;
    mutable std::atomic<bool> derived_ready_{false};
    mutable std::optional<Derived> derived_;
};
```

Four things about it that are load-bearing rather than decorative:

- **The `acquire` load and the `release` store are the pair that makes it correct.** The release
  store publishes every write `build_derived` made; the acquire load is what a reader
  synchronises with. Dropping either, or using `relaxed` for both, restores F3's race with the
  ceremony still visible. The second, `relaxed` load is inside the lock and is correct as such.
- **The memo slot is `private` and no accessor hands it out.** That is #386's `D-a` generalised:
  base state and memo slots stay private, and nothing outside the class can reach the slot to
  invalidate it. The Python analogue -- `HierarchicalGrid._rebuild` writing its base's private
  slots (`_hierarchical_grid.py:541-544` against `_grid.py:71`) -- is the shape this removes.
- **One block, one mutex.** Fourteen flags would be fourteen mutexes at 40 bytes each on Linux
  and fourteen chances to get the ordering wrong. Group by "computed by the same scan", which for
  `BsplineSpace1D` is all of it.
- **`build_derived` is a free function taking the frozen state**, so it can be tested without an
  object and cannot accidentally read a half-initialised `this`.

**The public contract, stated so that it is a contract and not a habit:**

> Every non-mutating public accessor on a C++-owned pantr domain type is safe to call
> concurrently from any number of threads on the same object, with no external locking. A lazy
> memo behind such an accessor is filled at most once and published atomically.

The exception, and there is exactly one: a **mutating** accessor is not, and the only mutable
members left in the domain layer are a grid's `CellTags` and `FacetTags`, which
`cpp/include/pantr/grid/tags.hpp:14-19` already reasons as the accumulating-container exception
and which say plainly *"Nothing here is thread-safe; the shared pointer buys lifetime, not
concurrency."* That sentence is correct and should stay.

### Does #386's shape generalise? Half of it

The ticket asks directly. The answer is that #386 made two separable decisions and they
generalise differently.

- **`D-a` -- base state passed up from the derived constructor, memo slot private in the mixin --
  generalises and should be copied.** It is why `HierarchicalGrid` needs no
  `invalidate_caches()`, and the same reasoning applies here: a `BsplineSpace1D`'s derived block
  is a function of its constructor arguments, so it never needs invalidating and nothing outside
  needs to reach it.
- **The unsynchronised `mutable std::optional<BVH>` does not generalise**, and by F3 it is a race
  in `#386`'s own code rather than a shape to imitate. The repair is the three lines above:
  an `atomic<bool>` and a `std::mutex` beside the existing `optional`. Measured cost, first use
  within noise of eager and 2.9 to 8.0 ns per subsequent access -- against a BVH build that is
  `O(num_cells)`. **#395 should make that change** (it is the ticket that edits both the mixin and
  the grid wrapper) rather than each later type deciding locally.

And **construct-then-freeze does hold for the spaces, more cleanly than for the grids.** With
#378, a grid is immutable after construction except for the BVH memo *and* its two tag
registries. A `BsplineSpace1D` or `BsplineSpace` is immutable after construction except for one
memo whose fill is unobservable -- same value, same span, every time. That is the strongest form
of the rule available: **the only mutation is one that no caller can observe**, which is the test
`CLAUDE.md` sets for a reasoned exception, rather than "it is fast".

`Bspline` is the type where it does not hold, because two of its methods take `in_place=True` and
reseat its space. Under `design/bspline_ownership_lifetime.md`'s storage the repair is available
and should be taken: **an in-place method replaces the whole derived block rather than
invalidating pieces of it**, so there is no `_beziers_cache = None` path and no way to reseat one
without the other. Today's two invalidation sites (`_bspline.py:1037-1038,1104-1105`) become one
assignment. #398 owns it, and it is the shape to prefer even if `in_place=` survives.

### The rest of the caches, by ticket

Not part of the fourteen, but governed by the same rule, and each one is a decision a ticket would
otherwise take alone.

| cache | today | becomes | ticket |
|---|---|---|---|
| `_cached_unique_knots_and_multiplicity`, `lru_cache(128)` | `_bspline_space_1d.py:40`, process-global, byte-keyed | **deleted**, ahead of the C++ dispatch (F5, F6) | #396 |
| `THBSplineSpace._contrib_cache` | `dict[int, list[tuple]]`, unbounded, per cell id, `:492` | **one flat CSR table** (offsets + entries) filled for *all* cells behind one DCLP flag | #397 |
| `THBSplineSpace._max_active_per_cell` | `int \| None` slot, `:493`, first call sweeps every cell | a field of that table, computed by the same sweep | #397 |
| `MultiLevelExtraction._ext` | `dict[int, SpanwiseElementExtraction]`, `:114` | `std::vector<std::shared_ptr<const SpanwiseElementExtraction>>` indexed by level, one DCLP fill; levels are few | #400 |
| `MultiLevelExtraction._coeffs_cache` | `dict[(int, tuple[int,...], int), ...]`, unbounded, `:112` | a hash map -- **legitimately**, its keys are data -- but it **needs a stated bound** and has none | #400 |
| `Bspline._beziers_cache`, `_locate_cache` | `None`-guarded slots, three invalidation sites | two members of one derived block, replaced wholesale by an in-place method | #398 |
| `SpanwiseElementExtraction.ops_1d` | `cached_property`, decompresses compact storage, `:273` | DCLP-lazy block; the accessor returns a **read-only view of the memo**, never a copy | #399 |
| `SpanwiseElementExtraction.num_identity_elements` | `cached_property`, `:391` | eager field | #399 |
| the grid mixin's `bvh_` | `mutable std::optional`, unsynchronised | DCLP, three lines (F3) | #395 |

Three of these deserve a sentence beyond the row.

**`_contrib_cache` is the only one where the shape change is large**, and it buys three things at
once: a flat CSR table is a fraction of the footprint of a dict of lists of tuples; filling all
cells at once removes the per-cell locking question entirely; and returning a `span` removes the
unenforced convention its own docstring carries today, *"the returned list is the cached object;
callers must not mutate it"* (`_thb_spline_space.py:770-773`), read directly by two external call
sites. **#397 owes a footprint measurement on a realistic adaptive hierarchy** before it commits,
because a lazy-all-cells fill trades memory for the dict's incrementality; the fallback if it is
too large is per-level rather than per-cell granularity, not a return to per-cell.

**`_coeffs_cache` is the one cache in the milestone that is genuinely a mapping over data**, which
`CLAUDE.md` permits, and it is also the one with no bound at all: its key includes a basis
multi-index, so its size is bounded by the number of distinct queries a caller makes, not by
anything about the object. Under a long-lived process that is leak-shaped. #400 must state the
bound or add one; this note declines to pick a number, because the right bound is a function of
the access pattern in `_element_coeffs` and nobody has measured it.

**`ops_1d` returning a view of its memo rather than a copy is not an optimisation**, it is what
keeps `tests/test_spanwise_element_extraction.py:1681`'s `np.shares_memory` assertion true and
what stops an `O(n_elements * p^2)` array being rebuilt per access.

### What must never appear

Three prohibitions, each with the concrete failure behind it rather than as a matter of taste.

1. **No process-global cache in the C++ layer.** It has no owner, so no bound is tied to
   anything, and it is the shape that produced the measured backend collision
   (`_backend.py:539-547`). If the same knot vector is genuinely built into many spaces and the
   scan is genuinely hot, the answer is to share the *space* -- which
   `design/bspline_ownership_lifetime.md`'s `shared_ptr<const BsplineSpace1D>` makes a one-line
   thing to do -- not to cache the scan behind everybody's back.
2. **No cache keyed on a domain object.** None exists today: verified, none of `BsplineSpace1D`,
   `BsplineSpace`, `THBSplineSpace`, `Bspline`, `THBSpline`, `TensorProductGrid` or
   `HierarchicalGrid` defines `__eq__` or `__hash__`, and every memoisation site in `bspline` and
   `grid` deliberately keys on a primitive or on a byte serialisation instead. The port must not
   create the need for one, because it would require value equality on a type whose equality
   semantics nobody has specified and which two backends would then have to agree on exactly.
3. **No cached value handed out writable.** `BsplineSpace.domain`
   (`_bspline_space_nd.py:144-156`) does exactly that today: it is the only array-shaped memo in
   `src/pantr/bspline/` that is not frozen with `.flags.writeable = False`, and a caller writing
   through it corrupts the cache for the object's whole life. Reproduced by execution:
   `d = s.domain; d[0, 0] = 999.0; s.domain[0, 0]` reads `999.0`, and no test in `tests/`
   inspects `domain.flags`. That is a bug in today's Python, not something the port introduces,
   and the port removes it -- a C++-side `std::array<double, 2>` presented as a `const`-scalar
   view cannot be written. **It is worth its own ticket so the fix is deliberate and gets a
   regression test**, rather than arriving as an unremarked side effect of #396.

## What it costs at the call site

- **In the wrapper, per access:** one `None` check plus a forwarded attribute read, against the
  56 to 60 ns a `cached_property` hit costs today. Same order; no measurable change to Layer 1.
- **In C++, per access to a memoised quantity:** 2.9 to 8.0 ns for the DCLP acquire load and the
  `optional` dereference, against 0 for an eager field. The spread is a cache-locality artefact
  of the microbenchmark, not a size effect.
- **In C++, at construction:** free for the eager scalars; the grouped block is not paid unless
  it is touched, and costs within noise of eager when it is (F4).
- **In a hot loop:** the memo is a design object here in the same sense the loop is. A de Boor
  or Cox-de Boor sweep should hoist `derived()` out of the loop into a local reference **once**,
  and take `first_basis_per_interval()` as a `span` before the loop rather than per element. 3 ns
  per element over a `num_intervals * (p+1)` sweep is a measurable tax for no reason, and the
  fix is a local, not a mechanism. Say it in the header next to the accessor, because the
  accessor looks free and is not.
- **What the port removes from the hot path:** the `tobytes()` key. Today every call of the
  public `get_unique_knots_and_multiplicity` builds an O(n) copy of the knot vector and hashes it
  -- 671 ns at 11 knots, 2738 ns at 2055 -- and there are 94 call sites of that name and its
  underlying impl across `src/` and `tests/`. That cost goes to zero, and it goes to zero because
  the data is owned rather than because anything was tuned.

## What each ticket in the milestone inherits

- **#395 `HierarchicalGrid`.** The three-line DCLP repair to the mixin's `bvh_`, and the
  correction to `_grid.py`'s `cell_bvh` docstring. Also `D-a` confirmed: no
  `invalidate_caches()`, because with #378 there is nothing to invalidate.
- **#396 `BsplineSpace1D` / `BsplineSpace`.** The whole of the above. Its `AC` set should say
  **one** memo, not fourteen; should require the `lru_cache` deleted *before* the C++ scan is
  dispatched (F6); and should include a TSan leg over the C++ unit tests, because that is the
  only gate that can see F3. **That is new infrastructure, and the note should not pretend
  otherwise.** CI's existing sanitizer job is ASan + UBSan through the `gcc-debug` preset
  (`.github/workflows/cpp.yaml:243`, `CMakePresets.json:36-41`), and
  `-fsanitize=address` cannot be combined with `-fsanitize=thread`, so TSan needs its own preset
  and its own job. #396 should weigh a permanent fourth `cpp.yaml` leg against a `gcc-tsan` preset
  run from `scripts/ci_local.sh` and recorded per PR that adds a memo. Either is defensible; what
  is not defensible is claiming the memo is race-free with nothing having looked, since F3 shows
  60 clean runs prove nothing.
- **#397 `THBSplineSpace`.** The CSR conversion, the footprint measurement it is contingent on,
  and the `span`-returning accessor that retires the "callers must not mutate" convention.
- **#398 `Bspline`.** One derived block replaced wholesale by an in-place method; the three
  invalidation sites collapse to one assignment.
- **#399 the extraction machinery.** Two memos, one eager and one DCLP-lazy, plus the
  view-not-copy rule on `ops_1d`.
- **#400 `THBSpline` / `MultiLevelExtraction`.** The level-indexed vector, and a stated bound on
  `_coeffs_cache` -- which is a genuine open question, not a mechanical port.
- **#401 scaffolding removal.** Nothing from this note is scaffolding; the memos are the design.
  What #401 should check is that no `cached_property` came back and no `__dict__` reappeared:
  a one-line test asserting `not hasattr(obj, "__dict__")` on each ported wrapper.

## Alternatives rejected

**Relax the wrapper: give it a `__dict__` and keep the fourteen `cached_property` sites.** By far
the cheapest to write, and it would leave Layer 1 untouched. Rejected because the resulting
memos are on the Python side of a seam the ruling deletes -- an interpreter-free consumer gets
none of them and must recompute -- and because a `__dict__` silently returns settable attributes
to a type documented immutable, which nothing in the suite would catch.
**What would change it:** a decision that the Python layer is permanent. It is not.

**Cache on both sides: C++ for the value, the wrapper for the numpy presentation of it.** Looks
cheapest of all and is what the ticket predicts someone will reach for. Rejected on a concrete
scenario rather than on principle: `BsplineSpace1D.get_unique_knots_and_multiplicity` returns two
arrays. If the wrapper caches the numpy views and the C++ caches the buffers, then the arrays are
correct exactly as long as nobody adds a second path that rebuilds the C++ block -- and the day
someone does, the wrapper serves views into freed storage. The cost of *not* caching the
presentation is one `nb::ndarray` construction per access, about 200 ns, which is the same order
as the Python call that asked for it.

**Eager everything.** No `mutable`, no flag, no mutex, and construct-then-freeze in its purest
form. Rejected on F4's measurement: the allocating derived arrays cost 4.8x to 7.2x the bare
construction, and a `THBSplineSpace` builds one `BsplineSpace` per level whose derived arrays may
never be touched. **What would change it:** a measurement showing the block is touched on
essentially every space that gets built, which would make the flag pure overhead. #396 is where
that could be measured; the DCLP form costs so little when it *is* touched that the answer
probably would not move.

**`std::call_once`.** The idiomatic spelling, and correct. Rejected on the measured 1.6
microsecond first call -- ten times the whole construction of a small space -- which is a
`FUTEX_WAKE` glibc issues even with no waiters. DCLP is the same guarantee with that cost removed.
**What would change it:** a C++ standard library whose `call_once` fast-path-initialises without
the syscall. Worth re-measuring if the toolchain floor moves.

**`std::shared_mutex`, or an atomic pointer published with `compare_exchange`.** Both correct; the
first is heavier than a plain mutex for a fill that happens once, and the second requires the
memo to be heap-allocated so that a pointer can be published, which adds an indirection to every
subsequent read. DCLP over `optional` keeps the value inline.

**Keep the `lru_cache` and key it on the backend, as `backend_keyed_cache` does.** The
conservative option: it closes F6 with a one-line decorator swap and leaves the measured 1.4-2.1x
saving in place. Rejected because it preserves a process-global cache whose hit costs an O(n) key
build (F5) in order to memoise a scan that, once owned, runs once per object. It is the right
answer only if the derived data does **not** move into the type -- that is, only if this note's
main decision is reversed.

## Bad practices flagged, for the user's inspection rather than for fixing here

- **`cpp/include/pantr/grid/grid.hpp`'s `mutable std::optional<BVH<scalar_type>> bvh_` is a data
  race** for any consumer without a GIL (F3). **Cost of keeping:** undefined behaviour reachable
  by the interpreter-free consumer the port exists for, invisible to every value test, and the
  shape gets copied by the next nine types. **Cost of fixing:** three lines and a measured 2.9 to
  8.0 ns per access. #395 is the ticket that already edits that file.
- **`src/pantr/grid/_grid.py`'s `cell_bvh` docstring states a Python implementation detail as the
  method's contract** -- *"may each build a valid tree and one write wins"* -- for a method both
  backends implement, where it is false of one of them. **Cost of keeping:** a reader concludes
  the race is benign and writes the same shape into a space. **Cost of fixing:** one docstring, no
  behaviour change.
- **`BsplineSpace.domain` hands out its cached array unfrozen** while every other array-shaped
  memo in `src/pantr/bspline/` is frozen, and nothing tests it (prohibition 3 above). Reproduced.
  Wants a ticket and a regression test, not a silent fix inside #396.
- **`THBSplineSpace._contrib_cache` and `MultiLevelExtraction._coeffs_cache` are unbounded
  dicts**, and `_contrib_cache`'s docstring carries an unenforced "callers must not mutate the
  returned list" contract that two external call sites rely on. Both are pre-existing; both are
  removed by the port if #397 and #400 take the rows above, which is the argument for taking them.
- **`CLAUDE.md`'s "Performance notes" say "change-of-basis matrices and unique knots are cached to
  avoid recomputation across calls".** After #396 that is no longer how unique knots work, and
  the sentence is the kind that outlives its subject. One line, when #396 lands.

## Epistemic status

- **Measured, 2026-08-31, on this machine:** F4's six-column table (g++ 14.4 `-O2`, one pinned
  core, constructor bodies in a separate translation unit, best of 5); F5's four-column table
  (CPython 3.14.6 in the `pantr` env, best of 5 over 2e4 accesses); the DCLP hot-path 2.9 to
  8.0 ns; the `cached_property` hit at 56 to 60 ns; the `call_once` first-call cost isolated as
  size-independent across two sizes. The construct-and-destroy benchmarks moved by up to 2x
  between runs of the same binary while every within-row ratio held; the tables report one run and
  the text says which readings survive that.
- **Measured under ThreadSanitizer (g++ 14.4 `-fsanitize=thread`), 8 threads:** 4 data races in
  the bare `mutable std::optional` variant, all frames in its accessor; 0 in the `call_once` and
  eager variants in the same binary. **And measured without TSan: 60 of 60 runs produced the
  correct total**, which is the point of the finding.
- **Verified by reading the tree at `a45e935`:** all 16 `cached_property` sites and the 7 + 7
  split; the `lru_cache`'s key construction at `_bspline_space_1d.py:320` and its definition at
  `:40`; that `_get_unique_knots_and_multiplicity_impl` is not backend-dispatched
  (`_bspline_knots.py:286`); the 19 `gil_scoped_release` sites, their four files, and that all are kernel bindings rather than methods of a domain type;
  `backend_keyed_cache`'s recorded measurement (`_backend.py:525-555`); that no domain class in
  `bspline` or `grid` defines `__eq__` or `__hash__`; `_contrib_cache`'s and `_coeffs_cache`'s
  shapes and unboundedness; the CI Python matrix.
- **Verified by execution in the `pantr` env:** that `BsplineSpace.domain` hands out a writable
  array and a write through it persists in the cache; that `functools.cached_property` on a
  `__slots__` class raises `TypeError: No '__dict__' attribute ... to cache`.
- **Derived, not measured:** that fusing `num_intervals`'s distinct-knot count into the
  constructor's validation scan is free. It is one comparison per knot inside a loop that already
  reads every knot, so it cannot be more than a constant factor on a pass that is already there;
  the eager-scalars column supports the conclusion for two bounded scans, not for a full one.
- **Asserted, not measured:** that the CSR form of `_contrib_cache` is smaller than the dict of
  lists of tuples. It is very likely -- a `list[tuple[int,int,tuple[int,...]]]` per cell is
  several PyObjects per entry -- but nobody has weighed either, and **#397 owes the measurement
  before it commits to a lazy-all-cells fill.**
- **Not investigated:** whether the not-yet-public downstream consumer calls
  `BsplineSpace1D.get_unique_knots_and_multiplicity` in a loop, which is the one place the
  `lru_cache`'s removal could be felt as a regression rather than as a cleanup. It is on the
  exposure list as a method of `BsplineSpace1D`, and the check is owed before #396 lands. Also
  not investigated: the footprint or hit rate of `_coeffs_cache` under any real workload, which
  is what #400 needs to state its bound.

## What was compiled, and how to reproduce it

Not committed.

**The C++ construction benchmark** is four structs over one knot vector, differing only in what
the constructor computes: nothing derived; the O(1) scalars; the scalars plus the three arrays;
and the scalars plus a lazily-filled block behind `std::call_once` or behind double-checked
locking. All four constructors and the `build_derived` helper live in a translation unit the
benchmark does not include, so no body can inline into the loop -- the precaution
`design/grid_hierarchy_port.md` records getting wrong the first time. `-O2`, no LTO, `taskset` to
one core.

```bash
g++ -O2 -std=c++20 -pthread -c impl.cpp -o impl.o
g++ -O2 -std=c++20 -pthread bench.cpp impl.o -o bench && taskset -c 2 ./bench
```

**The race demonstration** is the same three memo shapes in one binary, each hammered by N
threads that all first-touch the memo:

```bash
g++ -O1 -g -std=c++20 -fsanitize=thread -pthread memo.cpp -o memo_tsan && ./memo_tsan 8
g++ -O2    -std=c++20                  -pthread memo.cpp -o memo_fast
for i in $(seq 1 60); do ./memo_fast 8; done   # every total correct; this is the point
```

One thing to check before believing a null result from the TSan run, by analogy with the
devirtualisation trap: if it reports nothing at all, confirm the threads really do race by
checking that at least one race is reported for the deliberately-bare variant. A TSan build that
optimised the accessor into the constructor, or a run where one thread finishes before the next
starts, reports clean for the wrong reason.
