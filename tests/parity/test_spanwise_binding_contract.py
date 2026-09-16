"""What the `SpanwiseElementExtraction` bindings guarantee, as opposed to what they compute.

This is the sibling of `test_bspline_binding_contract.py`, for
`pantr::bspline::SpanwiseElementExtraction<T>`
(`cpp/include/pantr/bspline/spanwise_extraction.hpp`, bound by
`cpp/bindings/bspline_spanwise_extraction.cpp`). That file's own comment states every
guarantee asserted below and why each matters; this module pins them rather than
re-deriving them.

Every claim here is exact -- a view, a flag, a refusal, a name, an identity -- so
nothing in this file carries a tolerance.

## Why these and not others

Each of the following is a failure that is **silent**: the parity suite compares
*values*, and every one of these bugs leaves the values unchanged.

**A view that is really a copy.** `compact_ops_1d`, `idx_maps_1d`,
`is_identity_mask_1d` and `ops_1d` are all `nb::ndarray` views over storage the
extraction owns, with the extraction as the array's owner. A copying binding would
satisfy every value assertion elsewhere in the suite and would additionally break
`pantr.bspline.make_struct_view`, which bundles these three arrays by reference for
a downstream `@njit` caller to unbox.

**The `ops_1d` memo rebuilt per access rather than filled once.** `ops_1d` is the one
property behind a `LazySlot`: the C++ side decompresses every direction's dense block
on the first call and memoises it. An implementation that rebuilt the block on every
access would still return the right values and would silently make the property
quadratic in a sweep over elements, which is exactly the cost the memo exists to
avoid.

**A view that is writeable.** `const T` as the array's scalar is what nanobind turns
into the read-only flag. Without it a caller can rewrite the extraction's compacted
operators from the outside, leaving `idx_maps_1d` and `is_identity_mask_1d`
describing a block that no longer matches.

**A view that outlives its owner.** The `self` owner argument is what keeps the
extraction's storage alive once every other reference to it is dropped. Losing it is
a use-after-free that usually reads back the correct value, which is why the
assertion that actually bites is the refcount one, not the value read after `gc`.

**A dtype the constructor converts rather than refuses.** The two classes carry the
storage format in their names and nothing else does. Without `.noconvert()` nanobind
would cast a `float64` operator block into `SpanwiseElementExtraction32` silently,
changing every value the extraction hands out in a way that would be attributed to
the operator builders rather than to the cast.

**A borrowing accessor reaching Python.** `space_ref()` exists in C++ for an inner
loop that must not pay an atomic pair per access, and is deliberately not bound.

**The nested space copied instead of shared, or its identity not preserved.** The
extraction stores `std::shared_ptr<const BsplineSpace<T>>`, so `ext.space is space`
must hold for a space that came from Python, and taking it out must not pin the
extraction alive artificially nor let the space die with it.

## What every value/range check in `spanwise_extraction.hpp` guards against

`compact()` and the constructor validate shapes and ranges nothing in the type
signature can express: the target's range, the bundle count against the space's
dimension, each direction's element count and mask length, and that no operator
block is degenerate. Each is a `std::invalid_argument` mapped by nanobind to
`ValueError`; the rank and contiguity of the incoming arrays are instead expressed in
the binding's own signature (`nb::ndim<3>`, `nb::c_contig`), so violating those is a
`TypeError` raised before the C++ constructor ever runs.
"""

from __future__ import annotations

import gc
import sys
from typing import Any

import numpy as np
import pytest

from pantr._backend import Backend, use_backend
from pantr.bspline import BsplineSpace, BsplineSpace1D, SpanwiseElementExtraction

_DIR0_KNOTS = np.asarray([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0], dtype=np.float64)
"""A clamped quadratic with one repeated interior knot: 3 elements, none Bézier-identity."""

_DIR1_KNOTS = np.asarray([10.0, 10.0, 11.0, 12.0, 12.0], dtype=np.float64)
"""A second direction, differing in degree, counts, domain and scale: 2 elements.

Distinct from `_DIR0_KNOTS` in every reduced quantity, for the reason
`test_bspline_binding_contract.py` gives: two agreeing directions make a
transposition or a wrong reduction invisible.
"""

_EXTRACTION_CLASSES = ("SpanwiseElementExtraction32", "SpanwiseElementExtraction64")
"""The two scalar-width classes the extension registers."""


def _bindings() -> Any:
    """Import the extension.

    Deferred and in one place, matching `test_bspline_binding_contract.py`: the
    module is optional, and every caller below is already gated on the
    ``cpp_backend`` fixture.

    Returns:
        Any: The :mod:`pantr._pantr_cpp` module.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    return _pantr_cpp


def _cpp_space(dtype: Any = np.float64) -> BsplineSpace:
    """Build a two-direction tensor-product space under the C++ backend.

    Neither direction has a Bézier-identity element under either knot vector, so a
    ``"bezier"`` extraction built from this space compacts every element in every
    direction and no property below has to special-case an identity row.

    Args:
        dtype (Any): The storage format.

    Returns:
        BsplineSpace: The space, holding a C++ handle whose two directions have
        3 and 2 elements respectively.
    """
    with use_backend(Backend.CPP):
        return BsplineSpace(
            [
                BsplineSpace1D(np.asarray(_DIR0_KNOTS, dtype=dtype), 2),
                BsplineSpace1D(np.asarray(_DIR1_KNOTS, dtype=dtype), 1),
            ]
        )


def _cpp_extraction(dtype: Any = np.float64) -> tuple[SpanwiseElementExtraction, BsplineSpace]:
    """Build a ``"bezier"`` extraction over `_cpp_space` under the C++ backend.

    Args:
        dtype (Any): The storage format.

    Returns:
        tuple[SpanwiseElementExtraction, BsplineSpace]: The extraction and the space
        object it was built from, so identity assertions have both sides to compare.
    """
    with use_backend(Backend.CPP):
        space = _cpp_space(dtype)
        return SpanwiseElementExtraction(space, "bezier"), space


def _array_accessors(impl: Any) -> dict[str, np.ndarray[Any, Any]]:
    """Every array the C++ extraction hands out, by name, flattened over direction.

    Args:
        impl (Any): The ``SpanwiseElementExtraction{32,64}`` handle to read.

    Returns:
        dict[str, np.ndarray]: One entry per (property, direction) pair, so a test
        can assert a property of all of them and name the one that failed.
    """
    accessors: dict[str, np.ndarray[Any, Any]] = {}
    for name in ("compact_ops_1d", "idx_maps_1d", "is_identity_mask_1d", "ops_1d"):
        for k, array in enumerate(getattr(impl, name)):
            accessors[f"{name}[{k}]"] = array
    return accessors


def _direction_bundle(
    n_elements: int, n_out: int, n_in: int, dtype: Any
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Build one direction's raw constructor arguments: a dense block and a mask.

    Values are arbitrary but distinct per entry, which is enough for every check
    below: none of them reads the operators as anything but storage to compact.

    Args:
        n_elements (int): The direction's element count.
        n_out (int): The operator's row count.
        n_in (int): The operator's column count.
        dtype (Any): The storage format.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(operators, mask)`` of shapes
        ``(n_elements, n_out, n_in)`` and ``(n_elements,)``, the mask all-``False``
        so every element is compacted and none is treated as identity.
    """
    operators = (np.arange(n_elements * n_out * n_in, dtype=dtype) + 1.0).reshape(
        n_elements, n_out, n_in
    )
    mask = np.zeros(n_elements, dtype=np.bool_)
    return operators, mask


def _good_bundles(
    space_impl: Any, dtype: Any, n_out: int = 2, n_in: int = 2
) -> tuple[list[np.ndarray[Any, Any]], list[np.ndarray[Any, Any]]]:
    """Build a valid per-direction bundle for every direction of `space_impl`.

    Args:
        space_impl (Any): The C++ tensor-product space handle.
        dtype (Any): The storage format.
        n_out (int): The operator row count to use for every direction.
        n_in (int): The operator column count to use for every direction.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: ``(operators, masks)``, one entry
        per direction, each shaped to match that direction's element count.
    """
    operators: list[np.ndarray[Any, Any]] = []
    masks: list[np.ndarray[Any, Any]] = []
    for n_elements in space_impl.num_intervals:
        ops, mask = _direction_bundle(int(n_elements), n_out, n_in, dtype)
        operators.append(ops)
        masks.append(mask)
    return operators, masks


# ===========================================================================
# Arrays: read-only, views, the ops_1d memo, and lifetime
# ===========================================================================


def test_every_extraction_array_is_read_only(cpp_backend: None) -> None:
    """A caller cannot rewrite a validated extraction's storage from the outside."""
    ext, _ = _cpp_extraction()
    for name, array in _array_accessors(ext._impl).items():
        assert not array.flags.writeable, f"{name} came back writeable"
        with pytest.raises(ValueError):
            array.reshape(-1)[0] = 0


def test_every_extraction_array_is_a_view_rather_than_a_copy(cpp_backend: None) -> None:
    """Two reads of one accessor view one buffer, for every array the extraction hands out.

    ``shares_memory`` is what distinguishes a view from a copy; the values agree
    either way, which is why no value assertion elsewhere in the suite can stand in
    for this.
    """
    ext, _ = _cpp_extraction()
    first = _array_accessors(ext._impl)
    second = _array_accessors(ext._impl)
    for name in first:
        assert np.shares_memory(first[name], second[name]), f"{name} was copied out"


def test_compact_ops_1d_is_the_same_storage_the_wrapper_hands_out_via_factors(
    cpp_backend: None,
) -> None:
    """``factors()`` reads compact rows by reference, not by copy.

    ``pantr.bspline.make_struct_view`` bundles ``compact_ops_1d`` by reference for a
    downstream ``@njit`` caller, and :meth:`SpanwiseElementExtraction.factors` reads
    the very same storage for its non-identity directions. A compacting constructor
    that copied its input arrays instead of aliasing its own compacted block would
    satisfy every value check here and still break both consumers.
    """
    ext, _ = _cpp_extraction()
    saw_non_identity = False
    for cell in range(ext.num_total_intervals):
        for k, (is_identity, op) in enumerate(ext.factors(cell)):
            if is_identity:
                continue
            saw_non_identity = True
            assert op is not None
            assert np.shares_memory(op, ext._impl.compact_ops_1d[k]), (
                f"factors()'s direction-{k} operator does not share memory with "
                "compact_ops_1d, so one of the two is a copy"
            )
    assert saw_non_identity, "expected at least one non-identity direction; the case was flattened"


def test_ops_1d_views_the_memo_rather_than_rebuilding_it(cpp_backend: None) -> None:
    """Three reads of ``ops_1d`` all view the same decompressed block.

    ``ops_1d`` is the one property behind a `LazySlot`, filled on the first call and
    published atomically. An implementation that decompressed the dense block afresh
    on every access would return identical *values* and would silently turn a sweep
    over ``num_total_intervals`` elements into an ``O(n)`` rebuild per element, which
    is the cost the memo exists to remove -- so this is checked on its own rather
    than folded into the generic view-vs-copy sweep above.
    """
    ext, _ = _cpp_extraction()
    reads = [ext._impl.ops_1d for _ in range(3)]
    for d in range(ext.dim):
        for other in reads[1:]:
            assert np.shares_memory(reads[0][d], other[d]), (
                f"direction {d} of ops_1d was rebuilt between reads, so the memo "
                "is not doing its job"
            )


def test_an_extraction_array_outlives_the_extraction_it_came_from(cpp_backend: None) -> None:
    """Every array keeps the extraction alive, so dropping it does not free the storage.

    A use-after-free here would usually read back the correct value, so the check
    that actually bites is the refcount one: taking an array must *raise* the
    extraction's reference count, which is what says a keep-alive was installed at
    all. Checked for every array accessor, not one, since the keep-alive is installed
    per binding and an omission on one of four is exactly the shape this would miss.
    """
    ext, _ = _cpp_extraction()
    for name in ("compact_ops_1d", "idx_maps_1d", "is_identity_mask_1d", "ops_1d"):
        impl = ext._impl
        before = sys.getrefcount(impl)
        taken = getattr(impl, name)
        assert sys.getrefcount(impl) > before, (
            f"{name} did not reference its owner, so nothing keeps the storage alive "
            "once the extraction is dropped"
        )
        del taken

    expected = {name: np.array(a) for name, a in _array_accessors(ext._impl).items()}
    kept = _array_accessors(ext._impl)
    del ext
    gc.collect()
    # Churn the allocator so a use-after-free would have somewhere else to land.
    for _ in range(64):
        other, other_space = _cpp_extraction()
        del other, other_space
    gc.collect()
    for name, array in kept.items():
        np.testing.assert_array_equal(array, expected[name])


def test_scalar_properties_are_plain_python_values(cpp_backend: None) -> None:
    """``num_identity_elements``, ``is_identity``, the shape tuples and ``num_intervals``.

    Plain Python ``int``/``bool``/``tuple`` values rather than a 0-d array or any
    other object that could alias the extraction's storage. That is what "cannot
    dangle" means here: an ``int`` or a ``bool`` owns no buffer to free, so unlike the
    array accessors above this needs no owner argument and no lifetime test -- only
    the type check, because a future change returning e.g. ``np.int64`` or a 0-d view
    would reopen exactly the question the array tests above answer for the others.
    """
    ext, _ = _cpp_extraction()
    impl = ext._impl
    assert type(impl.num_identity_elements) is int
    assert type(impl.is_identity) is bool
    assert type(impl.num_intervals) is tuple
    assert type(impl.input_shape_per_dir) is tuple
    assert type(impl.output_shape_per_dir) is tuple
    for counts in (impl.num_intervals, impl.input_shape_per_dir, impl.output_shape_per_dir):
        assert all(type(n) is int for n in counts), counts


# ===========================================================================
# The `_ref` rule
# ===========================================================================


def test_no_bound_extraction_method_is_a_borrowing_accessor(cpp_backend: None) -> None:
    """No name on either extraction class ends in ``_ref``.

    ``space_ref()`` exists in C++ and is deliberately not bound: it borrows the space
    rather than sharing the handle, which is what saves the atomic pair inside a
    C++-side loop, and a caller holding it past the extraction's death would have
    nothing to blame.
    """
    bindings = _bindings()
    offenders = []
    for class_name in _EXTRACTION_CLASSES:
        cls = getattr(bindings, class_name)
        offenders += [f"{class_name}.{name}" for name in dir(cls) if name.endswith("_ref")]
    assert not offenders, f"borrowing accessors reached Python: {offenders}"

    # A vacuity guard on the guard: `space_ref` is a real, bound-in-C++ name on the
    # underlying type, so this is a live rule rather than a check over an empty set.
    # Asserting a name that IS bound is what says `dir()` is reporting this class's
    # surface at all -- otherwise the assertion above would pass because `dir()` saw
    # nothing, not because nothing borrows.
    for class_name in _EXTRACTION_CLASSES:
        cls = getattr(bindings, class_name)
        assert "space" in dir(cls), (
            f"the owning accessor is not bound on {class_name} either, so dir() may "
            "not be reporting this class's methods at all"
        )


# ===========================================================================
# The dtype refusal
# ===========================================================================


@pytest.mark.parametrize(
    ("class_name", "right_dtype", "wrong_dtype"),
    [
        ("SpanwiseElementExtraction64", np.float64, np.float32),
        ("SpanwiseElementExtraction32", np.float32, np.float64),
    ],
    ids=["64_given_float32_operators", "32_given_float64_operators"],
)
def test_extraction_refuses_operators_of_the_wrong_width(
    cpp_backend: None, class_name: str, right_dtype: Any, wrong_dtype: Any
) -> None:
    """Each class takes only its own storage format's operator blocks.

    The space is built at the class's *own* width, so the operators are the only
    thing wrong: a space of the other width would be refused by its own caster and
    would make this pass for the wrong reason. ``.noconvert()`` on the ``operators``
    argument is what turns a silent narrowing or widening cast into this refusal --
    and it has to propagate through nanobind's `std::vector` caster to each element's
    own caster, which is the assumption this test pins rather than assumes.
    """
    bindings = _bindings()
    cls = getattr(bindings, class_name)
    space = _cpp_space(right_dtype)
    operators, masks = _good_bundles(space._impl, wrong_dtype)
    with pytest.raises(TypeError):
        cls(space._impl, 0, "irrelevant", operators, masks)


# ===========================================================================
# The nested space: shared, identity-stable, and independently alive
# ===========================================================================


def test_the_extraction_shares_its_space_rather_than_copying(cpp_backend: None) -> None:
    """The C++ extraction holds the very space handle it was built from.

    Compared at the *implementation* level, because the wrapper level cannot see it:
    the wrapper keeps the space object it was built from, so ``ext.space is space``
    holds whether the C++ constructor shared or copied, and a copying constructor
    would leave two Python objects reporting identity over two different C++ objects.
    """
    ext, space = _cpp_extraction()
    assert ext._impl.space is space._impl, (
        "the extraction's space is not the handle it was given, so the C++ constructor copied"
    )
    # The wrapper's own contract: the object a caller passed in comes back.
    assert ext.space is space
    # Identity-stable across two reads, which is what `nb_type_put`'s `inst_c2p`
    # lookup buys and what a memo on the wrapper would otherwise have to fake.
    assert ext._impl.space is ext._impl.space


def test_taking_the_extractions_space_does_not_pin_the_extraction(cpp_backend: None) -> None:
    """Reading ``space`` installs no keep-alive on the extraction.

    **Belt-and-braces, not a live detector, and that is worth stating plainly.**
    ``cpp/bindings/bspline_types.cpp:94-106`` measured the equivalent accessor on
    ``BsplineSpace.spaces`` and found the refcount delta unchanged under *both* the
    shared-handle design and a reversion to ``rv_policy::reference_internal``: a
    space handed to the extraction's constructor always arrives *from Python*, so it
    already has a live instance that ``nb_type_put``'s ``inst_c2p`` lookup finds, and
    no new instance is created for a keep-alive to be installed on. The same argument
    applies here verbatim -- ``space`` always arrives from Python, never built by the
    extraction itself -- so a delta of zero is expected under either design and
    proves nothing about which one is in force.

    Kept anyway because a *non-zero* delta would still be a finding: it would mean a
    keep-alive appeared where the design says none should, aliasing into the
    extraction rather than sharing the value. What actually decides class H here is
    the C++ test, `cpp/tests/test_spanwise_extraction.cpp` (``space().get() ==
    space.get()``).
    """
    ext, _ = _cpp_extraction()
    impl = ext._impl
    before = sys.getrefcount(impl)

    taken = impl.space
    assert taken.dim == ext.dim
    assert sys.getrefcount(impl) == before, (
        "holding the escaped space changed the extraction's reference count, which "
        "means a keep-alive was installed: the accessor is aliasing into the owner "
        "rather than sharing the value, and it will dangle for a caller with no "
        "interpreter"
    )

    del taken
    gc.collect()
    assert sys.getrefcount(impl) == before


def test_the_space_outlives_the_extraction(cpp_backend: None) -> None:
    """A space taken out of an extraction still knows its own state after it dies.

    **Necessary and not sufficient, matching the field's own equivalent test.** For a
    space that arrived from Python, the value survives the owner's death under
    ``reference_internal`` too, because the object handed back is the caller's own
    instance and that instance owns the C++ value regardless of what the extraction
    does -- so this passes on a reversion as readily as on the real design, and what
    actually decides it is the C++ test, `cpp/tests/test_spanwise_extraction.cpp`'s
    ``check_a_space_outlives_the_extraction``. Kept as the Python-level record of the
    guarantee, and churned against freed memory the way `test_bspline_binding_
    contract.py` does, because F4 there measured a scalar read after free returning
    the *correct* value on this exact class of bug.
    """
    ext, _ = _cpp_extraction()
    space = ext._impl.space
    expected_total = space.num_total_intervals
    expected_intervals = space.num_intervals

    del ext
    gc.collect()
    for _ in range(64):
        other, other_space = _cpp_extraction()
        del other, other_space
    gc.collect()

    assert space.num_total_intervals == expected_total
    assert space.num_intervals == expected_intervals


def test_an_extraction_has_no_instance_dictionary() -> None:
    """The wrapper is immutable, and ``__slots__`` is what makes that true.

    A ``__dict__`` would silently return settable attributes to a type documented
    immutable. Not parametrized over the backends and not gated on the extension:
    ``__slots__`` belongs to the wrapper class, which is the same class whichever
    implementation it holds, so this needs no C++ handle to check.
    """
    space = BsplineSpace([BsplineSpace1D(_DIR0_KNOTS, 2), BsplineSpace1D(_DIR1_KNOTS, 1)])
    ext = SpanwiseElementExtraction(space, "bezier")
    assert not hasattr(ext, "__dict__")
    with pytest.raises(AttributeError):
        ext.some_new_attribute = 1
    with pytest.raises(AttributeError):
        ext._impl = None  # type: ignore[assignment]
    with pytest.raises(AttributeError):
        del ext._space


# ===========================================================================
# Every value/range refusal the C++ type owns
# ===========================================================================


@pytest.mark.parametrize("target", [-1, 3], ids=["below_range", "above_range"])
def test_extraction_refuses_a_target_outside_the_enum(cpp_backend: None, target: int) -> None:
    """``target`` must name a member of ``ExtractionTarget``; nothing else is a member.

    `is_extraction_target` in `spanwise_extraction.hpp` bounds it to ``[0, 2]``. A
    value outside that range would cast to an ``ExtractionTarget`` no branch handles,
    so the constructor refuses it instead of admitting an unreachable enum value.
    """
    bindings = _bindings()
    space = _cpp_space()
    operators, masks = _good_bundles(space._impl, np.float64)
    with pytest.raises(ValueError):
        bindings.SpanwiseElementExtraction64(space._impl, target, "irrelevant", operators, masks)


def test_extraction_refuses_a_bundle_count_that_disagrees_with_the_spaces_dimension(
    cpp_backend: None,
) -> None:
    """One bundle is required per direction of the space; the two-direction space gets one.

    Caught by the constructor's own ``directions.size() != num_intervals.size()``
    check, distinct from the per-direction shape checks `compact()` makes.
    """
    bindings = _bindings()
    space = _cpp_space()
    operators, masks = _good_bundles(space._impl, np.float64)
    with pytest.raises(ValueError):
        bindings.SpanwiseElementExtraction64(space._impl, 0, "irrelevant", operators[:1], masks[:1])


def test_extraction_refuses_an_operator_count_disagreeing_with_the_directions_elements(
    cpp_backend: None,
) -> None:
    """A direction's operator block must have one row per element of that direction.

    Direction 0 of `_cpp_space` has 3 elements; handed 4 operators, `compact()`
    refuses before it ever reads the mask.
    """
    bindings = _bindings()
    space = _cpp_space()
    operators, masks = _good_bundles(space._impl, np.float64)
    wrong_ops, wrong_mask = _direction_bundle(4, 2, 2, np.float64)
    operators[0], masks[0] = wrong_ops, wrong_mask
    with pytest.raises(ValueError):
        bindings.SpanwiseElementExtraction64(space._impl, 0, "irrelevant", operators, masks)


def test_extraction_refuses_a_mask_length_disagreeing_with_the_directions_elements(
    cpp_backend: None,
) -> None:
    """A direction's identity mask must have one entry per element of that direction.

    The operator block for direction 0 is left correctly shaped (3 rows), so this
    isolates the mask-length check in `compact()` from the operator-count check
    the previous test exercises.
    """
    bindings = _bindings()
    space = _cpp_space()
    operators, masks = _good_bundles(space._impl, np.float64)
    _, wrong_mask = _direction_bundle(4, 2, 2, np.float64)
    masks[0] = wrong_mask
    with pytest.raises(ValueError):
        bindings.SpanwiseElementExtraction64(space._impl, 0, "irrelevant", operators, masks)


@pytest.mark.parametrize(("n_out", "n_in"), [(0, 2), (2, 0)], ids=["zero_rows", "zero_columns"])
def test_extraction_refuses_a_degenerate_operator_block(
    cpp_backend: None, n_out: int, n_in: int
) -> None:
    """An operator with no rows or no columns is refused, not sentinel-padded.

    `compact()` reads `n_out` and `n_in` off the operator array and would index out
    of range on a degenerate one, so it is refused up front with a diagnosis instead.
    """
    bindings = _bindings()
    space = _cpp_space()
    operators, masks = _good_bundles(space._impl, np.float64)
    degenerate_ops, degenerate_mask = _direction_bundle(3, n_out, n_in, np.float64)
    operators[0], masks[0] = degenerate_ops, degenerate_mask
    with pytest.raises(ValueError):
        bindings.SpanwiseElementExtraction64(space._impl, 0, "irrelevant", operators, masks)


def test_extraction_refuses_a_noncontiguous_operator_array(cpp_backend: None) -> None:
    """A strided operator block is refused rather than silently reinterpreted.

    `nb::c_contig` in the binding's signature is what turns this into a `TypeError`
    at the call, before the C++ constructor -- and its own shape checks -- ever run.
    """
    bindings = _bindings()
    space = _cpp_space()
    operators, masks = _good_bundles(space._impl, np.float64)
    wide, _ = _direction_bundle(3, 2, 4, np.float64)
    strided = wide[:, :, ::2]
    assert strided.shape == (3, 2, 2)
    assert not strided.flags["C_CONTIGUOUS"]
    operators[0] = strided
    with pytest.raises(TypeError):
        bindings.SpanwiseElementExtraction64(space._impl, 0, "irrelevant", operators, masks)


def test_extraction_refuses_a_wrong_rank_operator_array(cpp_backend: None) -> None:
    """A 2D operator array is refused rather than reinterpreted as one row.

    `nb::ndim<3>` in the binding's signature is what turns this into a `TypeError` at
    the call, matching the contiguity check above.
    """
    bindings = _bindings()
    space = _cpp_space()
    operators, masks = _good_bundles(space._impl, np.float64)
    operators[0] = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(TypeError):
        bindings.SpanwiseElementExtraction64(space._impl, 0, "irrelevant", operators, masks)
