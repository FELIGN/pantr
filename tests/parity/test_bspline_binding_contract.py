"""What the `pantr.bspline` bindings guarantee, as opposed to what they compute.

`tests/parity/test_bspline_space_1d.py`, `tests/parity/test_bspline_space_nd.py` and
`tests/parity/test_bspline_type.py` compare the two backends' *answers*. This file is
about the *bindings*: what the arrays they hand out alias, how long they stay valid,
what a constructor refuses to convert, what a nested object's identity and lifetime
are, and one rule that is about the shape of the bound surface rather than about any
one call.

The spaces come first and the field -- ``pantr::bspline::Bspline``, which holds a
space *and* an array of its own -- is the last section.

Every claim here is exact -- an identity, a flag, a refusal, a name -- so nothing in
this file carries a tolerance, and `design/backend_parity.md` Rule 8 does not bite.

## Why these and not others

Each of the five below is a failure that is **silent**, which is the reason a
binding needs tests of its own at all.

**A view that is really a copy.** Every array accessor here returns an
``nb::ndarray`` over storage the space owns, with the space as the array's owner.
Bound without that owner argument, nanobind copies instead and the array comes back
writeable -- and every value assertion in the sibling file still passes. What the
copy costs is not correctness but shape: the derived block exists so that a loop over
intervals reads it instead of recomputing an O(n) scan, and a copying property makes
the natural spelling of such a loop quadratic in nothing.

**A view that is writeable.** ``const T`` as the array's scalar is what nanobind
turns into the read-only flag. Without it a caller can rewrite a validated space's
knots from the outside, leaving the space's own derived block describing a vector it
no longer holds.

**A view that outlives its owner.** The owner argument is also what keeps the storage
alive. Dropping the space while an array of its knots is still held is the ordinary
way this arises, and the failure is a use-after-free that usually reads back the
correct value -- `design/bspline_ownership_lifetime.md` F4 measured exactly that.

**A dtype the constructor converts rather than refuses.** The two classes carry the
storage format in their names and nothing else carries it. Widening a ``float32``
knot vector into the 64-bit class multiplies the space's tolerance by about four
orders, because it is ``8 * eps(T) * scale``; narrowing does the reverse and moves
the knots. ``.noconvert()`` is what makes
:func:`pantr.bspline._bspline_space_1d._impl_class` picking the wrong class loud.

**A borrowing accessor reaching Python.** `design/bspline_ownership_lifetime.md`
requires that no ``_ref`` accessor is ever bound: it is a reference with no owner and
no policy, so a caller can hold it past the owner's death with nothing to blame.
``BsplineSpace1D`` has none to bind. **The tensor-product ``BsplineSpace`` does** --
``space_ref(d)``, which borrows a direction and costs a measured 5.83 ns against
``space(d)``'s 14.92 ns -- so the rule stops being vacuous with this type, and the
assertion over the whole bound surface stops being a formality.

## What the tensor product adds: a nested object rather than an array

``BsplineSpace`` is the first type in this front that holds another domain type, so
four more silent failures become possible and each gets a test below. They are
``design/bspline_ownership_lifetime.md``'s M1, M2, M7 and M8, and the note's own
summary of why value assertions cannot stand in for them is worth restating in one
line each:

**The constructor copies its directions instead of sharing them (M7).** Every value
agrees, and ``space.spaces[0] is space_1d`` still holds at the wrapper level because
the wrapper keeps what it was built from -- so two Python objects would report
identity over two different C++ objects. The assertion is on the *implementations*.

**A keep-alive is installed where the value should have been shared (M2).** That is
what reverting to ``rv_policy::reference_internal`` looks like from Python, and it
works today while dangling for a consumer with no interpreter. The detector is a
``sys.getrefcount`` delta of exactly zero on the owner.

**The wrapper forgets to seed its ``_spaces`` slot from the constructor (M1, M8).**
Values all agree; only the identity moves.

**A direction does not outlive its owner.** The whole reason class H stores a
``shared_ptr<const T>``: the guarantee lives in the type rather than in the binding.
``design/bspline_ownership_lifetime.md`` F4 measured a scalar read after free
returning the *correct* value, so this is asserted on state a destroyed space would
have had to overwrite.
"""

from __future__ import annotations

import gc
import sys
from typing import Any

import numpy as np
import pytest

from pantr._backend import Backend, use_backend
from pantr.bspline import BsplineSpace, BsplineSpace1D

_KNOTS = np.asarray([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0], dtype=np.float64)
"""A clamped quadratic with one repeated interior knot, so every accessor is non-trivial."""

_OTHER_KNOTS = np.asarray([10.0, 10.0, 11.0, 12.0, 12.0], dtype=np.float64)
"""A second direction, differing from ``_KNOTS`` in degree, counts, domain and scale.

Distinct in every reduced quantity, deliberately, for the reason
``tests/parity/test_bspline_space_nd.py`` gives at length: two agreeing directions
make a transposition and a wrong reduction invisible at once.
"""


def _bindings() -> Any:
    """Import the extension.

    Deferred and in one place, matching `test_bezier_binding_contract.py`: the
    module is optional, and every caller below is already gated on the
    ``cpp_backend`` fixture.

    Returns:
        Any: The :mod:`pantr._pantr_cpp` module.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    return _pantr_cpp


def _cpp_space(dtype: Any = np.float64) -> BsplineSpace1D:
    """Build a space under the C++ backend.

    Args:
        dtype (Any): The storage format.

    Returns:
        BsplineSpace1D: The space, holding a C++ handle.
    """
    with use_backend(Backend.CPP):
        return BsplineSpace1D(np.asarray(_KNOTS, dtype=dtype), 2)


def _array_accessors(space: BsplineSpace1D) -> dict[str, np.ndarray[Any, Any]]:
    """Every array a space hands out, by name.

    Args:
        space (BsplineSpace1D): The space to read.

    Returns:
        dict[str, np.ndarray]: One entry per array accessor, so a test can assert a
        property of all of them and name the one that failed.
    """
    unique, mult = space.get_unique_knots_and_multiplicity()
    unique_in, mult_in = space.get_unique_knots_and_multiplicity(in_domain=True)
    return {
        "knots": space.knots,
        "unique_knots": unique,
        "multiplicity": mult,
        "unique_knots_in_domain": unique_in,
        "multiplicity_in_domain": mult_in,
        "first_basis_per_interval": space.first_basis_per_interval(),
    }


def test_every_array_the_space_hands_out_is_read_only(cpp_backend: None) -> None:
    """A caller cannot rewrite a validated space's storage from the outside."""
    space = _cpp_space()
    for name, array in _array_accessors(space).items():
        assert not array.flags.writeable, f"{name} came back writeable"
        with pytest.raises(ValueError):
            array[0] = 0


def test_every_array_is_a_view_rather_than_a_copy(cpp_backend: None) -> None:
    """Two reads of one accessor view one buffer.

    ``shares_memory`` is what distinguishes a view from a copy; the values agree
    either way, which is why no assertion on them can stand in for this.
    """
    space = _cpp_space()
    first = _array_accessors(space)
    second = _array_accessors(space)
    for name in first:
        assert np.shares_memory(first[name], second[name]), f"{name} was copied out"


def test_the_in_domain_classes_are_a_subrange_of_the_whole(cpp_backend: None) -> None:
    """One memo and two views of it, rather than two scans.

    True of the C++ implementation and **not** of the oracle, which runs a second
    scan for the in-domain form -- so this is a binding claim and lives here rather
    than in the shared suite, where it would pass under the default backend for a
    reason that has nothing to do with the binding.
    """
    space = _cpp_space()
    unique, mult = space.get_unique_knots_and_multiplicity()
    unique_in, mult_in = space.get_unique_knots_and_multiplicity(in_domain=True)

    assert np.shares_memory(unique, unique_in)
    assert np.shares_memory(mult, mult_in)


def test_an_array_outlives_the_space_it_came_from(cpp_backend: None) -> None:
    """The array keeps its owner alive, so dropping the space does not free the storage.

    A use-after-free here would usually read back the correct value, so the check
    that actually bites is the refcount one: taking the array must *raise* the
    space's reference count, which is what says a keep-alive was installed at all.
    """
    space = _cpp_space()
    handle = space._impl
    before = sys.getrefcount(handle)

    knots = space.knots
    assert sys.getrefcount(handle) > before, (
        "taking a view did not reference its owner, so nothing keeps the storage "
        "alive once the space is dropped"
    )

    expected = np.asarray(_KNOTS)
    del space, handle
    gc.collect()

    np.testing.assert_array_equal(knots, expected)


@pytest.mark.parametrize(
    ("class_name", "wrong_dtype"),
    [("BsplineSpace1D32", np.float64), ("BsplineSpace1D64", np.float32)],
)
def test_the_space_refuses_a_dtype_it_would_have_to_cast(
    cpp_backend: None, class_name: str, wrong_dtype: Any
) -> None:
    """Each class takes only its own storage format.

    Reached through :mod:`pantr._pantr_cpp` rather than through
    :class:`~pantr.bspline.BsplineSpace1D`, whose wrapper picks the matching class
    and so can never present the mismatch. The extension is importable and both
    names are public attributes of a public module, so this surface is real.
    """
    cls = getattr(_bindings(), class_name)
    with pytest.raises(TypeError):
        cls(np.asarray(_KNOTS, dtype=wrong_dtype), 2)


def test_no_bound_method_is_a_borrowing_accessor(cpp_backend: None) -> None:
    """No name on any bound class of this front ends in ``_ref``.

    The rule from `design/bspline_ownership_lifetime.md`. It was vacuous while only
    ``BsplineSpace1D`` existed, which hands out spans of its own storage rather than
    references to nested objects; ``pantr::bspline::BsplineSpace`` has a real
    ``space_ref``, the two hierarchical classes have three between them and
    ``pantr::bspline::Bspline`` has one, so from here on this is the assertion that
    keeps them unbound. There is no ``static_assert`` for absence, review alone will
    not hold across the rest of this front, and the cost of the rule being broken is
    a dangling reference with no policy anywhere to blame.
    """
    bindings = _bindings()
    offenders = []
    for class_name in _SPACE_CLASSES + _FIELD_CLASSES:
        cls = getattr(bindings, class_name)
        offenders += [f"{class_name}.{name}" for name in dir(cls) if name.endswith("_ref")]

    assert not offenders, f"borrowing accessors reached Python: {offenders}"
    # A vacuity guard on the guard: the C++ type does have a borrowing accessor, so
    # if every name were somehow absent from `dir` this test would pass for the wrong
    # reason. Asserting a name that IS bound is what says `dir` sees the surface.
    assert "space" not in dir(bindings.BsplineSpace64), (
        "the owning single-direction accessor is not bound either, so `dir` may not "
        "be reporting this class's methods at all"
    )
    assert "spaces" in dir(bindings.BsplineSpace64)
    # And the same for the hierarchical class, which carries three of these.
    assert "root_space" in dir(bindings.THBSplineSpace64)
    assert "level_space" in dir(bindings.THBSplineSpace64)
    # And for the field, whose one borrowing accessor is `space_ref`: the owning twin
    # is bound under the bare name, so seeing it is what says `dir` reports this
    # class's surface at all.
    assert "space" in dir(bindings.Bspline64)
    assert "control_points" in dir(bindings.Bspline64)


_SPACE_CLASSES = (
    "BsplineSpace1D32",
    "BsplineSpace1D64",
    "BsplineSpace32",
    "BsplineSpace64",
    "THBSplineSpace32",
    "THBSplineSpace64",
)
"""Every space class the extension registers, for the rules asserted over all of them.

The two hierarchical classes are what make the ``_ref`` rule bite hardest: they have
three borrowing accessors -- ``root_space_ref``, ``grid_ref`` and ``level_space_ref`` --
against the tensor-product type's one, and each hands out a reference to a nested
object rather than a span of the owner's own storage.
"""


def _cpp_tensor_product() -> BsplineSpace:
    """Build a two-direction tensor-product space under the C++ backend.

    Returns:
        BsplineSpace: The space, holding a C++ handle whose directions are the C++
        handles of two univariate spaces.
    """
    with use_backend(Backend.CPP):
        return BsplineSpace([BsplineSpace1D(_KNOTS, 2), BsplineSpace1D(_OTHER_KNOTS, 1)])


def test_the_tensor_product_shares_its_directions_rather_than_copying(
    cpp_backend: None,
) -> None:
    """The C++ space holds the very handles it was built from.

    M7. Compared at the *implementation* level, because the wrapper level cannot see
    it: the wrapper keeps the univariate wrappers it was built from, so
    ``space.spaces[0] is one_d`` holds whether the C++ constructor shared or copied,
    and a copying constructor would leave two Python objects reporting identity over
    two different C++ objects.
    """
    with use_backend(Backend.CPP):
        first = BsplineSpace1D(_KNOTS, 2)
        second = BsplineSpace1D(_OTHER_KNOTS, 1)
        space = BsplineSpace([first, second])

    directions = space._impl.spaces
    assert directions[0] is first._impl, "direction 0 is not the handle the space was given"
    assert directions[1] is second._impl, "direction 1 is not the handle the space was given"

    # And the wrapper's own contract, M1 and M8: seeded from the constructor, so the
    # objects a caller passed in come back.
    assert space.spaces[0] is first
    assert space.spaces[1] is second


def test_taking_a_direction_does_not_pin_its_owner(cpp_backend: None) -> None:
    """Reading ``spaces`` installs no keep-alive on the tensor-product space.

    M2's assertion, and **not** the detector M2 says it is. Its docstring used to
    claim this was "the one detector that distinguishes class H from a reversion to
    ``rv_policy::reference_internal``". Measured on this binding, it is not:
    rebinding a direction accessor to return a reference with
    ``rv_policy::reference_internal`` leaves this delta at zero, the handed-out
    object identical to the one passed in, and its value readable after the owner
    dies.

    The reason is specific and is what to carry forward: a direction always arrives
    *from Python*, so it already has a live instance that ``nb_type_put``'s
    ``inst_c2p`` lookup finds, and no new instance is created for a keep-alive to be
    installed on. ``design/bspline_ownership_lifetime.md`` F3 and F4 measured the
    delta on a stand-in whose nested object was constructed in C++, where it does
    move -- so the note's measurement is right and its application to this accessor
    was wrong. The detector stays live for an accessor whose nested object has no
    instance of its own, which ``THBSplineSpace.level_space`` is and this is not.

    **What decides class H here is the C++ test**:
    ``cpp/tests/test_bspline_space_nd.cpp`` compares addresses and reads a direction
    after its space is destroyed, and that is the gate a reversion fails.

    Kept because a non-zero delta would still be a finding -- it would mean a
    keep-alive appeared where the design says none should -- and read **while the
    returned objects are alive**, since a keep-alive lives on the returned object and
    is released with it. Both holdings are checked, because F4 records two correct
    measurements of such a delta disagreeing over exactly that: the whole container
    against a single element.
    """
    space = _cpp_tensor_product()
    impl = space._impl
    before = sys.getrefcount(impl)

    directions = impl.spaces
    assert len(directions) == 2
    assert sys.getrefcount(impl) == before, (
        "holding the whole tuple of directions changed the owner's reference count, "
        "which means a keep-alive was installed per element: the accessor is aliasing "
        "into the owner rather than sharing the value, and it will dangle for a caller "
        "with no interpreter"
    )

    one = directions[0]
    del directions
    gc.collect()
    assert sys.getrefcount(impl) == before, (
        "holding one escaped direction changed the owner's reference count, so a "
        "keep-alive was installed on it"
    )

    del one
    gc.collect()
    assert sys.getrefcount(impl) == before


def test_a_direction_outlives_the_tensor_product_space(cpp_backend: None) -> None:
    """A direction taken out of a space still knows its own state after the space dies.

    The property that justifies storing ``shared_ptr<const T>`` in the type instead of
    annotating the binding. Asserted on the direction's counts rather than on the
    handle being non-null, because F4 measured a scalar read after free returning the
    correct value -- so several tensor-product spaces are built and dropped in between
    to churn the freed storage.

    Necessary and not sufficient, and the same measurement is why: for a nested
    object that arrived from Python the value survives the owner's death under
    ``reference_internal`` too, because the object that comes back is the caller's
    own instance and that instance owns the C++ value. So this fails on a design
    that hands out nothing at all, and passes on the reversion. The C++ test is
    the gate.
    """
    space = _cpp_tensor_product()
    direction = space._impl.spaces[0]
    expected_basis = direction.num_basis
    expected_knots = np.array(direction.knots)

    del space
    gc.collect()
    with use_backend(Backend.CPP):
        for _ in range(64):
            BsplineSpace([BsplineSpace1D(_OTHER_KNOTS, 1)])
    gc.collect()

    assert direction.num_basis == expected_basis
    np.testing.assert_array_equal(direction.knots, expected_knots)


def test_the_tensor_product_domain_is_a_read_only_view(cpp_backend: None) -> None:
    """The C++ ``domain`` aliases the space's own storage and cannot be written.

    M5 and M6 for this type's one array accessor. ``const T`` as the scalar is what
    nanobind turns into the read-only flag; the owner argument is what makes it a view
    rather than a silent copy, and ``shares_memory`` is what tells the two apart --
    the values agree either way.

    A dimensionless space is checked alongside, because that is the one case where the
    block is empty and the stand-in address the binding substitutes for a possibly
    null ``data()`` is what stops nanobind reading it as "no array".
    """
    space = _cpp_tensor_product()
    domain = space._impl.domain

    assert domain.shape == (2, 2)
    assert not domain.flags.writeable, "the domain view came back writeable"
    with pytest.raises(ValueError):
        domain[0, 0] = 0.0
    assert np.shares_memory(space._impl.domain, space._impl.domain), (
        "two reads of the domain do not view one buffer, so it is being copied out"
    )

    with use_backend(Backend.CPP):
        empty = BsplineSpace([])
    empty_domain = empty._impl.domain
    assert empty_domain.shape == (0, 2)
    assert not empty_domain.flags.writeable


def test_the_wrapper_copies_the_domain_out_of_the_cpp_view(cpp_backend: None) -> None:
    """``BsplineSpace.domain`` does not hand the C++ read-only view straight through.

    The half of the domain contract that is a *binding* claim; the backend-independent
    half -- that a write through it corrupts nothing, which is the defect the port
    retires -- lives in ``tests/test_bspline_space.py`` so that it runs without the
    extension too.

    What this one catches is the copy being dropped as an optimisation: the view is
    read-only, so a caller who has always written to ``space.domain`` would start
    getting a ``ValueError`` under the C++ backend and not under the Python one. The
    consumer census for this port found no such caller, but the divergence would be
    real and silent in the suite, since nothing else compares the two arrays' flags.
    """
    space = _cpp_tensor_product()
    domain = space.domain

    assert domain.flags.writeable, (
        "the wrapper handed the C++ view through, so the domain is read-only under "
        "the C++ backend and writable under the Python one"
    )
    assert not np.shares_memory(domain, space._impl.domain), (
        "the wrapper's domain aliases the space's own storage, so a write would "
        "reach validated state from outside"
    )
    assert space.domain is not domain, "the wrapper is caching the domain again"


@pytest.mark.parametrize(
    ("class_name", "direction_class"),
    [("BsplineSpace32", "BsplineSpace1D64"), ("BsplineSpace64", "BsplineSpace1D32")],
)
def test_the_tensor_product_refuses_a_direction_of_the_wrong_width(
    cpp_backend: None, class_name: str, direction_class: str
) -> None:
    """Each tensor-product class takes only its own storage format's directions.

    The counterpart of the univariate ``.noconvert()`` check, and it needs no
    annotation: ``BsplineSpace<T>`` can hold only ``BsplineSpace1D<T>``, and the two
    univariate classes are unrelated nominal types, so nanobind has no conversion to
    apply. Asserted rather than argued, because "there is no conversion to suppress"
    is exactly the kind of claim that stops being true when a caster is added.
    """
    bindings = _bindings()
    cls = getattr(bindings, class_name)
    wrong = getattr(bindings, direction_class)
    dtype = np.float32 if direction_class.endswith("32") else np.float64
    direction = wrong(np.asarray(_KNOTS, dtype=dtype), 2)
    with pytest.raises(TypeError):
        cls([direction])


def _cpp_thb_space() -> Any:
    """Build a two-level, two-dimensional THB space handle under the C++ backend.

    Cut 1 of FELIGN/pantr#397 ships no Python wrapper, so this builds the handle
    through the bindings the way ``tests/parity/test_thb_spline_space.py`` does. A
    real refinement rather than a flat hierarchy, so the level-dependent accessors
    have more than one level to hand out.

    Returns:
        Any: A ``THBSplineSpace64`` handle over a corner-refined unit square.
    """
    cpp = _bindings()
    knots = np.asarray([0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0], dtype=np.float64)
    directions = [cpp.BsplineSpace1D64(knots, 2, False, True) for _ in range(2)]
    breakpoints = [np.linspace(0.0, 1.0, 5) for _ in range(2)]
    grid = cpp.HierarchicalGrid(
        cpp.TensorProductGrid(breakpoints), np.asarray([2, 2], dtype=np.int64)
    )
    grid = grid.refine(0, np.asarray([0, 0], dtype=np.int64), np.asarray([2, 2], dtype=np.int64))
    return cpp.THBSplineSpace64(cpp.BsplineSpace64(directions), grid, True, [None, None])


def _thb_array_accessors(space: Any) -> dict[str, np.ndarray[Any, Any]]:
    """Every array the hierarchical space hands out, by name.

    The per-level and per-cell accessors are sampled at a level and a cell that are
    both non-trivial: level 1 exists only because the space was refined, and cell 0 of
    a corner refinement carries functions from both levels.

    Args:
        space (Any): The ``THBSplineSpace64`` handle to read.

    Returns:
        dict[str, np.ndarray]: One entry per array accessor, so a test can assert a
        property of all of them and name the one that failed.
    """
    # The vacuity guard, in the helper so all three tests inherit it: `shares_memory` on
    # two empty arrays is False on some builds and True on others, and a read-only
    # assertion over an empty array asserts nothing at all. A case flattened by a future
    # edit would make every test below pass for the wrong reason.
    assert space.num_levels > 1, "the case stopped being hierarchical"
    assert space.num_truncated > 0, "the case stopped truncating, so no accessor is stressed"

    dof, level, multi = space.contributions(0)
    accessors = {
        "domain": space.domain,
        "level_offsets": space.level_offsets,
        "active_function_indices(0)": space.active_function_indices(0),
        "active_function_indices(1)": space.active_function_indices(1),
        "active_basis(0)": space.active_basis(0),
        "contributions(0).dof": dof,
        "contributions(0).level": level,
        "contributions(0).multi": multi,
    }
    empty = [name for name, array in accessors.items() if array.size == 0]
    assert not empty, f"these accessors came back empty, so nothing below tests them: {empty}"
    return accessors


def test_every_array_the_hierarchical_space_hands_out_is_read_only(cpp_backend: None) -> None:
    """The rule the univariate and tensor-product types already carry, for this one too.

    The reviewer of FELIGN/pantr#438 found the hierarchical type getting only the
    borrowed-accessor check while the three properties this file exists for went
    unasserted on it. The implementation was correct; nothing said so, and nothing
    would have said otherwise -- the parity suite compares values, and a copy or a
    writeable view agrees on every value it hands back.
    """
    space = _cpp_thb_space()
    for name, array in _thb_array_accessors(space).items():
        assert not array.flags.writeable, f"{name} came back writeable"
        with pytest.raises(ValueError):
            array.reshape(-1)[0] = 0


def test_every_hierarchical_array_is_a_view_rather_than_a_copy(cpp_backend: None) -> None:
    """Two reads of one accessor view one buffer, for the hierarchical type as well.

    ``shares_memory`` is what distinguishes a view from a copy; the values agree
    either way, which is why no assertion on them can stand in for this.
    """
    space = _cpp_thb_space()
    first = _thb_array_accessors(space)
    second = _thb_array_accessors(space)
    for name in first:
        assert np.shares_memory(first[name], second[name]), f"{name} was copied out"


def test_a_hierarchical_array_outlives_the_space_it_came_from(cpp_backend: None) -> None:
    """Each array keeps its owner alive, so dropping the space does not free the storage.

    The refcount half is the one that bites, for the reason
    :func:`test_an_array_outlives_the_space_it_came_from` gives: a use-after-free here
    would usually read back the correct value. Every accessor is checked rather than
    one, because the keep-alive is installed per binding and an omission on one of
    eight is exactly the shape this would miss.
    """
    space = _cpp_thb_space()
    for name in _thb_array_accessors(space):
        before = sys.getrefcount(space)
        taken = _thb_array_accessors(space)[name]
        assert sys.getrefcount(space) > before, (
            f"{name} did not reference its owner, so nothing keeps the storage alive "
            f"once the space is dropped"
        )
        del taken

    expected = {name: np.array(a) for name, a in _thb_array_accessors(space).items()}
    kept = _thb_array_accessors(space)
    del space
    gc.collect()
    for name, array in kept.items():
        np.testing.assert_array_equal(array, expected[name])


def test_a_space_has_no_instance_dictionary() -> None:
    """The wrapper is immutable, and ``__slots__`` is what makes that true.

    A ``__dict__`` would silently return settable attributes to a type documented
    immutable, and would let a memo be attached to the wrapper -- the second truth
    `design/bspline_derived_caches.md` forbids, since the value is memoized in the
    implementation and there is exactly one of it.

    Not parametrized over the backends and not gated on the extension: ``__slots__``
    belongs to the wrapper class, which is the same class whichever implementation
    it holds, so a second run would duplicate the first. Gating it would have made
    it skip in the configuration where it is the only thing checking this.
    """
    space = BsplineSpace1D(_KNOTS, 2)
    assert not hasattr(space, "__dict__")
    with pytest.raises(AttributeError):
        space.some_new_attribute = 1  # type: ignore[attr-defined]

    tensor_product = BsplineSpace([space, BsplineSpace1D(_OTHER_KNOTS, 1)])
    assert not hasattr(tensor_product, "__dict__")
    with pytest.raises(AttributeError):
        tensor_product.some_new_attribute = 1
    # The tensor product goes further than its univariate direction and refuses to
    # rebind its own slots too, which is what `design/bspline_derived_caches.md` asks
    # for and what `src/pantr/grid/_tensor_product_grid.py` ships: `_spaces` carries
    # an identity contract, so a caller reseating it would leave a space whose
    # directions disagree with its counts.
    with pytest.raises(AttributeError):
        tensor_product._impl = None  # type: ignore[assignment]
    with pytest.raises(AttributeError):
        del tensor_product._spaces


# ===========================================================================
# The field: `pantr::bspline::Bspline`
# ===========================================================================
#
# The field is the second type in this front that holds another domain type, and
# the first that holds one *and* an array of its own. So both halves of
# `design/bspline_ownership_lifetime.md` are live on one class: its `space` is
# class H and its `control_points` is class A, and the four silent failures
# above (M1, M2, M7, M8) plus M5 and M6 all apply to it at once.
#
# What it adds that no space could: the field is the only ported type whose
# Python surface *mutates*, so `Bspline._mutate` is the one place any of these
# invariants can be broken after construction. Its own contract -- that the
# wrapper refuses every attribute write, so nothing but `_mutate` can reseat the
# value or leave the derived block behind -- is asserted at the end of this file.

_FIELD_CLASSES = ("Bspline32", "Bspline64")
"""The two field classes the extension registers.

Swept by the ``_ref`` rule alongside the space classes: ``pantr::bspline::Bspline``
has a ``space_ref()`` of its own, which borrows the space rather than copying the
handle and must not reach Python.
"""


def _cpp_field(dtype: Any = np.float64, *, is_rational: bool = False) -> Any:
    """Build a two-direction field under the C++ backend.

    The basis counts are 6 and 3 -- ``_KNOTS`` is a quadratic over nine knots and
    ``_OTHER_KNOTS`` a linear over five -- so the control net is ``(6, 3, rank)`` and
    no permutation of its shape is another admissible shape for this space.

    Args:
        dtype (Any): The storage format.
        is_rational (bool): Whether the last stored component is a weight.

    Returns:
        Any: A :class:`~pantr.bspline.Bspline` holding a C++ handle.
    """
    from pantr.bspline import Bspline  # noqa: PLC0415

    with use_backend(Backend.CPP):
        space = BsplineSpace(
            [
                BsplineSpace1D(np.asarray(_KNOTS, dtype=dtype), 2),
                BsplineSpace1D(np.asarray(_OTHER_KNOTS, dtype=dtype), 1),
            ]
        )
        control_points = np.arange(6 * 3 * 3, dtype=dtype).reshape(6, 3, 3)
        return Bspline(space, control_points, is_rational=is_rational)


def test_the_field_shares_its_space_rather_than_copying(cpp_backend: None) -> None:
    """The C++ field holds the very space handle it was built from.

    M7 for the field. Compared at the *implementation* level, because the wrapper
    level cannot see it: the wrapper keeps the space wrapper it was built from, so
    ``field.space is space`` holds whether the C++ constructor shared or copied, and
    a copying constructor would leave two Python objects reporting identity over two
    different C++ objects.
    """
    from pantr.bspline import Bspline  # noqa: PLC0415

    with use_backend(Backend.CPP):
        space = BsplineSpace([BsplineSpace1D(_KNOTS, 2), BsplineSpace1D(_OTHER_KNOTS, 1)])
        field = Bspline(space, np.arange(6 * 3 * 3, dtype=np.float64).reshape(6, 3, 3))

    assert field._impl.space is space._impl, (
        "the field's space is not the handle it was given, so the C++ constructor copied"
    )
    # And the wrapper's own contract, M1 and M8: seeded from the constructor, so the
    # object a caller passed in comes back.
    assert field.space is space
    # Identity-stable across two reads, which is what `nb_type_put`'s `inst_c2p`
    # lookup buys and what a memo on the wrapper would otherwise have to fake.
    assert field._impl.space is field._impl.space


def test_taking_the_fields_space_does_not_pin_the_field(cpp_backend: None) -> None:
    """Reading ``space`` installs no keep-alive on the field.

    M2 for the field, and it is worth saying plainly that this **does not** decide
    class H, whatever M2 says: measured, binding this accessor as ``space_ref`` with
    ``rv_policy::reference_internal`` leaves the delta at zero, the handed-out object
    identical to the one passed in, and its value readable after the field dies. A
    field's space always arrives from Python, so it already has an instance that
    ``nb_type_put``'s ``inst_c2p`` lookup finds and no keep-alive is created for the
    delta to see. See the sibling assertion on ``BsplineSpace.spaces`` above for the
    full argument, and ``cpp/tests/test_bspline_type.cpp`` for the gate that does
    decide it.

    Kept because a non-zero delta would still be a finding -- a keep-alive where the
    design says none should be -- and read **while the escaped space is alive**, since
    a keep-alive lives on the returned object and is released with it.
    """
    field = _cpp_field()
    impl = field._impl
    before = sys.getrefcount(impl)

    taken = impl.space
    assert taken.dim == 2
    assert sys.getrefcount(impl) == before, (
        "holding the escaped space changed the field's reference count, which means a "
        "keep-alive was installed: the accessor is aliasing into the owner rather "
        "than sharing the value, and it will dangle for a caller with no interpreter"
    )

    del taken
    gc.collect()
    assert sys.getrefcount(impl) == before


def test_the_space_outlives_the_field(cpp_backend: None) -> None:
    """A space taken out of a field still knows its own state after the field dies.

    The property that justifies storing ``shared_ptr<const BsplineSpace<T>>`` in the
    type instead of annotating the binding. Asserted on the space's counts rather
    than on the handle being non-null, because F4 measured a scalar read after free
    returning the correct value -- so fields are built and dropped in between to
    churn the freed storage.

    Necessary and not sufficient, and the same measurement is why: for a nested
    object that arrived from Python the value survives the owner's death under
    ``reference_internal`` too, because the object that comes back is the caller's
    own instance and that instance owns the C++ value. So this fails on a design
    that hands out nothing at all, and passes on the reversion. The C++ test is
    the gate.
    """
    field = _cpp_field()
    space = field._impl.space
    expected_total = space.num_total_basis
    expected_degrees = space.degrees

    del field
    gc.collect()
    for _ in range(64):
        _cpp_field()
    gc.collect()

    assert space.num_total_basis == expected_total
    assert space.degrees == expected_degrees


def test_the_fields_control_points_are_a_read_only_view(cpp_backend: None) -> None:
    """The C++ ``control_points`` aliases the field's own storage and cannot be written.

    M5 and M6 for the field's one array accessor, and the pair matters more here than
    for a space: a write through a writeable view would leave both of the field's
    derived memos -- the Bézier decomposition and the point-inversion context --
    describing a geometry it no longer holds, with nothing raising. That is the defect
    ``design/bspline_ownership_lifetime.md`` records in today's Python, and the
    read-only view is the half of its removal this front owns.

    ``shares_memory`` is what tells a view from a copy; the values agree either way.

    **The ``self`` owner argument is not observable from Python, and that was measured
    rather than assumed.** Dropping it leaves the array aliasing the same storage,
    still read-only, still valid after the field is dropped, and still taking a
    reference on the field -- so M6's symptom (a silent copy, handed out writeable)
    does not appear. The reason is F1 applied to an array: ``def_prop_ro`` passes
    ``rv_policy::reference_internal`` positionally, so the property already ties its
    return to ``self``. M6 is live for an array bound on a plain ``.def``, where the
    default policy is ``automatic``.

    The reference-count assertion below is therefore a check that *a* keep-alive
    exists, not that the owner argument supplied it: exactly ``1`` while the array is
    alive and ``0`` once it dies. A zero while it is alive would mean the property had
    been turned into a method and lost its default policy, which is the one way this
    accessor can start handing out an unowned view.
    """
    field = _cpp_field()
    impl = field._impl
    before = sys.getrefcount(impl)
    handed_out = impl.control_points

    assert not handed_out.flags.writeable, (
        "the scalar is not `const T`, so a caller can rewrite a validated geometry"
    )
    with pytest.raises(ValueError, match="read-only"):
        handed_out[0, 0, 0] = 99.0
    assert np.shares_memory(handed_out, impl.control_points), (
        "two reads do not share memory, so the accessor copies and the whole net is "
        "duplicated on every access"
    )
    assert sys.getrefcount(impl) == before + 1, (
        "the array does not hold a reference to the field, so the `self` owner "
        "argument is missing: the view aliases storage it does not keep alive, and it "
        "reads freed memory the moment the field is dropped"
    )

    del handed_out
    gc.collect()
    assert sys.getrefcount(impl) == before, (
        "the reference the array held was not released, which is a leak rather than a "
        "dangle -- and it would make the assertion above pass for the wrong reason"
    )


def test_the_control_points_view_outlives_the_field(cpp_backend: None) -> None:
    """The array keeps the C++ storage alive after the field handle is dropped.

    The owner argument is what does that. Without it the values are read from freed
    memory, which is a use-after-free that usually reads back correct and
    occasionally does not -- so this asserts the values, and churns the allocator in
    between rather than trusting one read.
    """
    field = _cpp_field()
    handed_out = field._impl.control_points
    expected = np.array(handed_out)

    del field
    gc.collect()
    for _ in range(64):
        _cpp_field()
    gc.collect()

    np.testing.assert_array_equal(handed_out, expected)


@pytest.mark.parametrize(
    ("class_name", "space_dtype", "net_dtype"),
    [
        ("Bspline64", np.float64, np.float32),
        ("Bspline32", np.float32, np.float64),
    ],
)
def test_the_field_refuses_a_control_net_it_would_have_to_cast(
    class_name: str, space_dtype: Any, net_dtype: Any, cpp_backend: None
) -> None:
    """Handing a field class a net of the other width is a refusal, not a cast.

    The ``.noconvert()`` on the binding. Without it nanobind casts silently, so
    :func:`pantr.bspline._bspline._impl_class` picking the wrong class would narrow or
    widen a caller's geometry with nothing to notice -- the class name is the only
    thing carrying the storage format.

    The space must be of the class's own width, so that the *net* is the only thing
    wrong: a space of the other width is refused by its own caster and would make
    this pass for the wrong reason.
    """
    bindings = _bindings()
    with use_backend(Backend.CPP):
        space = BsplineSpace([BsplineSpace1D(np.asarray(_KNOTS, dtype=space_dtype), 2)])
    net = np.arange(6 * 2, dtype=net_dtype).reshape(6, 2)

    cls = getattr(bindings, class_name)
    # The control-point argument is the only ill-typed one: the space handle matches.
    with pytest.raises(TypeError):
        cls(space._impl, net, False)


def test_a_field_has_no_instance_dictionary() -> None:
    """The field wrapper refuses every attribute write, so ``_mutate`` is the only writer.

    A ``__dict__`` would let a memo be attached to the wrapper -- the second truth
    ``design/bspline_derived_caches.md`` forbids -- and, worse for this type than for
    a space, it would let a caller reseat ``_impl`` or ``_space`` without discarding
    the derived block, which is exactly the invalidation hole that note asks this
    front to close.

    Not parametrized over the backends and not gated on the extension: ``__slots__``
    and ``__setattr__`` belong to the wrapper class, which is the same class whichever
    implementation it holds, so a second run would duplicate the first. Gating it
    would have made it skip in the configuration where it is the only thing checking
    this.
    """
    from pantr.bspline import Bspline  # noqa: PLC0415

    space = BsplineSpace([BsplineSpace1D(_KNOTS, 2)])
    field = Bspline(space, np.arange(6 * 2, dtype=np.float64).reshape(6, 2))

    assert not hasattr(field, "__dict__")
    with pytest.raises(AttributeError):
        field.some_new_attribute = 1
    with pytest.raises(AttributeError):
        field._impl = None  # type: ignore[assignment]
    with pytest.raises(AttributeError):
        field._space = None  # type: ignore[assignment]
    with pytest.raises(AttributeError):
        del field._derived
    # The block itself is reachable and fillable, which is what the two memos need;
    # what is unspellable is replacing one of them without the other, because only
    # `_take` writes the slot the block lives in.
    assert field._derived.beziers is None
    assert field._derived.locate is None
