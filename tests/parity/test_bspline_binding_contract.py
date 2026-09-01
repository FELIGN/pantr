"""What the `BsplineSpace1D` binding guarantees, as opposed to what it computes.

`tests/parity/test_bspline_space_1d.py` compares the two backends' *answers*. This
file is about the *binding*: what the arrays it hands out alias, how long they stay
valid, what the constructor refuses to convert, and one rule that is about the shape
of the bound surface rather than about any one call.

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
``BsplineSpace1D`` has none to bind, so the rule holds vacuously here -- and it is
asserted over the whole bound surface anyway, because the next type in this front
will have one and there is no ``static_assert`` for absence.
"""

from __future__ import annotations

import gc
import sys
from typing import Any

import numpy as np
import pytest

from pantr._backend import Backend, use_backend
from pantr.bspline import BsplineSpace1D

_KNOTS = np.asarray([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0], dtype=np.float64)
"""A clamped quadratic with one repeated interior knot, so every accessor is non-trivial."""


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
    """No name on either class ends in ``_ref``.

    The rule from `design/bspline_ownership_lifetime.md`. Vacuous for this type,
    which hands out spans of its own storage rather than references to nested
    objects, and asserted anyway: there is no ``static_assert`` for absence, review
    alone will not hold across the rest of this front, and the cost of the rule
    being broken is a dangling reference with no policy anywhere to blame.
    """
    bindings = _bindings()
    offenders = []
    for class_name in ("BsplineSpace1D32", "BsplineSpace1D64"):
        cls = getattr(bindings, class_name)
        offenders += [f"{class_name}.{name}" for name in dir(cls) if name.endswith("_ref")]

    assert not offenders, f"borrowing accessors reached Python: {offenders}"


@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.CPP], ids=["python", "cpp"])
def test_a_space_has_no_instance_dictionary(cpp_backend: None, backend: Backend) -> None:
    """The wrapper is immutable, and ``__slots__`` is what makes that true.

    A ``__dict__`` would silently return settable attributes to a type documented
    immutable, and would let a memo be attached to the wrapper -- the second truth
    `design/bspline_derived_caches.md` forbids, since the value is memoized in the
    implementation and there is exactly one of it.
    """
    with use_backend(backend):
        space = BsplineSpace1D(_KNOTS, 2)
    assert not hasattr(space, "__dict__")
    with pytest.raises(AttributeError):
        space.some_new_attribute = 1  # type: ignore[attr-defined]
