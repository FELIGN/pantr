"""Parity of the C++ ``THBSplineSpace`` against the Python oracle it was ported from.

Covers the *construction and query* half of FELIGN/pantr#397: the per-level spaces, the
Kraft selection, the truncation coefficients, the per-cell contribution table, and
refinement and coarsening.

**Both sides are reached through ``pantr.bspline.THBSplineSpace``**, under
:func:`~pantr._backend.use_backend`, since FELIGN/pantr#494 made that class a wrapper
that dispatches to the C++ type. An earlier version of this file reached the C++ side
through :mod:`pantr._pantr_cpp` directly and said so here, because the class was still
pure Python and the wrapper was the next cut; that is the cut, and going through the
public class is what makes the comparison say anything about what a caller gets.
:func:`test_the_wrapper_forwards_every_bound_member` is the per-member statement of it,
and :func:`test_construction_selects_the_backends_own_implementation` pins that the two
builders really do build two different things.

The ``__reduce__`` round trip the old docstring called owed is here too, in
:func:`test_a_thb_space_survives_pickling_across_every_backend_pair`.

What is **not** compared, and why it is not an omission
-------------------------------------------------------

Basis tabulation, the windowed restriction and the three prolongation operators have no
C++ counterpart. Since #494 they are computed on the wrapper against the forwarded
accessors, so one body serves both backends and a *numeric* cross-backend comparison of
them would be comparing one implementation with itself -- except through its two inputs,
which are the level spaces and the truncation coefficients, and both of those are
compared here directly. A tolerance for such a comparison is also not available honestly:
it would have to bound the effect of a last-bit difference in a knot vector on a
B-spline value, which is a conditioning argument nobody in this tree has derived. The
level-space **knot vectors** are therefore compared instead, which is the quantity that
argument would have needed, and they are compared against the space's own knot tolerance
-- a bound the library already derives and already uses to decide whether two knots are
the same one.

What agrees, quantity by quantity
---------------------------------

The split is per quantity with an argument for each, and it is not one decision applied in
bulk.

**Exactly** -- ``num_levels``, ``num_total_basis``, ``num_basis_per_level``, ``regularity``,
``num_truncated``, the per-level
active function index sets, ``level_offsets``, ``dof_level``, the per-cell ``active_basis``
lists, the contribution table's levels and multi-indices, ``max_active_per_cell``, the set
of truncated dofs, and each truncated function's representation level, box origin and box
shape. Every one is an index, a count or a set membership, so no rounding takes place and
bit-identity is the only criterion that says anything. A bounded comparison could not even
see two answers of different length.

The ``active_basis`` lists leave out a truncated function that vanishes on the cell, and
that decision reads zeros off the truncation coefficients, which below agree only within a
bound. It is still exact: a coefficient is a sum of non-negative terms, so it is zero in
both backends or in neither, as the section on the bound says of a zero coefficient.

**Within a derived bound** -- the truncation coefficients, and the level-space knot
vectors. The knots are graded against ``BsplineSpace1D.tolerance``, which is the
library's own absolute parametric tolerance for exactly this question. They are
**observed** to agree bit for bit over the shipped sweep, which is stronger than the
bound requires and is not counted: unlike the coefficients, nothing here reports a
per-run figure for them, and reading one off a past run would be reading a number that
nothing re-measures. The coefficients are
floating point, and bit-identity is not available for them: the oracle's ``_refine_box``
contracts through :func:`numpy.tensordot`, which reshapes and calls BLAS, whose summation
order is the implementation's -- ``CLAUDE.md`` records that a quantity round-off dominates
can differ between Accelerate and OpenBLAS by orders of magnitude. The C++ side sums in
index order. So the two run different summation orders on the same terms, and requiring
the bits would forbid a transformation nobody should be forbidding.

Observed rather than required: most coefficients do come out bit-identical, and
:class:`_SweepReport` counts how many so a reader can see the figure for the run in front
of them rather than a stale one written here. That is welcome and is not the criterion:
requiring it would forbid the summation-order difference the paragraph above allows.

The bound, and why it has no cancellation term
-----------------------------------------------

Every two-scale (Oslo) coefficient is non-negative, truncation only *zeroes* entries, and
the contraction sums non-negative products. So there is no cancellation anywhere in the
truncation, and the standard inner-product result (Higham, *Accuracy and Stability of
Numerical Algorithms*, 2nd ed., Theorem 3.1) applies in its relative form: a length-``n``
inner product of non-negative terms satisfies ``|fl(s) - s| <= gamma_n * s`` rather than
the ``gamma_n * sum|terms|`` a signed sum would owe. That is what lets the amplification
below be the coefficient's own magnitude.

One stage is one direction of one level transition, and its length is the coefficient
box's width in that direction *before* the transition. Composing stages multiplies their
``(1 + gamma)`` factors, and ``(1 + gamma_a)(1 + gamma_b) <= 1 + gamma_{a + b}``, so the
whole chain is ``gamma_N`` with ``N`` the sum of the contraction lengths over every
direction and every transition.

``N`` is bounded above without tracking the walk: the box widths grow monotonically, so
each stage's length is at most the *final* width in that direction, and a function
traverses at most ``num_levels - 1`` transitions. Hence

    N <= (num_levels - 1) * sum_k shape[k]

with ``shape`` the stored coefficient box. That is what :func:`_coefficient_roundings`
computes, per function, from the function's own box rather than from a worst case over the
space -- a global count would be looser by the ratio of the largest box to this one.

The factor of two that turns a one-sided forward-error bound into a two-sided parity bound
is the harness's, applied in :func:`tests._parity_harness.absolute_tolerance`, and is not
repeated here.

A zero coefficient gets a zero tolerance, and that is sound rather than an oversight: a
sum of non-negative terms is zero only when every term is zero, and both backends sum the
same terms, so an entry that is exactly zero in one is exactly zero in the other. The
non-cancellation argument above is what makes that true; it would be wrong for a signed
sum.

The independent accuracy check
-------------------------------

``design/backend_parity.md`` requires an independent check that the answer is *right*, not
only that two implementations agree. The one used here is the defining property of the
truncated basis (Giannelli-Jüttler-Speleers 2012, Thm 6): the truncated functions form a
**partition of unity**. Expressed in the finest level's tensor-product basis that is a
statement about the coefficients alone and needs no evaluation -- the coefficient vectors
of all the active functions, pushed to the finest level by pure two-scale refinement, must
sum to the all-ones vector.

:func:`_partition_of_unity_defect` runs that on the **C++** space's coefficients. It uses
the oracle's two-scale kernel to do the pushing, which is shared machinery rather than the
thing under test: what is being checked is the *selection and the truncation*, and that
kernel is independently pinned in ``cpp/tests/test_bspline_knot_insertion.cpp`` against the
cardinal B-spline's binomial stencil, which is a closed form neither implementation knows
about.

The check discriminates, which is the part a passing identity does not establish on its
own: :func:`test_the_untruncated_basis_fails_the_identity` runs the same computation on
the HB space, where the sum is nowhere near one.
"""

from __future__ import annotations

import contextlib
import copy
import math
import pickle
from typing import TYPE_CHECKING, Any, Final, NamedTuple

import numpy as np
import pytest

from pantr._backend import Backend, use_backend
from pantr.bspline import BsplineSpace, BsplineSpace1D, THBSplineSpace
from pantr.bspline._bspline_space_nd import _BsplineSpaceNDPython
from pantr.grid import hierarchical_grid, uniform_grid
from tests._parity_harness import (
    Field,
    Roundings,
    assert_object_parity,
    bounded_parity,
    exact_parity,
    unit_roundoff,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpy.typing as npt

_SHIPPED_CASES: Final = 200
"""Random cases the CI sweep draws.

Chosen for runtime rather than for confidence: each case builds two whole spaces, compares
every cell's contribution list in an interpreted loop, and then rebuilds both four more
times for the refinement and coarsening comparison. The confidence comes from
:func:`test_the_wide_sweep_agrees`, which runs ten times this and is what the ticket asks
be verified; the shipped run is the regression guard that has to stay affordable.
"""

_WIDE_FACTOR: Final = 10
"""How much wider the ``slow`` sweep is than the shipped one."""

_U: Final = unit_roundoff(np.float64)
"""Half an ulp, taken from the harness rather than spelled out again here."""

_VERDICT_WHY: Final = (
    "counts, level indices, flat function indices, global dofs, multi-indices and box "
    "origins are verdicts rather than displaced values; no rounding takes place on an "
    "integer, so bit-identity is the only criterion available here. A bounded comparison "
    "could not see two answers of different length at all"
)

_COEFFICIENT_WHY: Final = (
    "the truncation coefficients are built by contracting a coefficient box through the "
    "two-scale matrices, direction by direction and level by level. Every Oslo "
    "coefficient is non-negative and truncation only zeroes entries, so no sum here "
    "cancels and Higham Thm 3.1's inner-product bound applies in its relative form: one "
    "stage of length n costs gamma_n of the value itself, not of the sum of magnitudes. "
    "Stages compose as (1 + gamma_a)(1 + gamma_b) <= 1 + gamma_{a+b}, and the box widths "
    "grow monotonically, so N = (num_levels - 1) * sum_k shape[k] bounds the total. "
    "Bit-identity is NOT claimed because the oracle contracts through numpy.tensordot, "
    "hence BLAS, whose summation order is the implementation's while the C++ sums in "
    "index order"
)


class _Case(NamedTuple):
    """One hierarchy both backends are asked to build.

    Attributes:
        degrees (tuple[int, ...]): Per-direction polynomial degree.
        num_elements (tuple[int, ...]): Per-direction root element count.
        factor (tuple[int, ...]): Per-direction refinement factor.
        bounds (tuple[tuple[float, float], ...]): Per-direction parametric domain.
        refinements (tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]): The
            ``(level, lo, hi)`` boxes refined in order, on the grid each previous one
            produced.
        truncate (bool): Whether the truncated basis is built.
        regularity (tuple[int | None, ...]): Per-direction continuity at inserted knots.
        dtype (Any): The root space's storage format.
    """

    degrees: tuple[int, ...]
    num_elements: tuple[int, ...]
    factor: tuple[int, ...]
    bounds: tuple[tuple[float, float], ...]
    refinements: tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]
    truncate: bool
    regularity: tuple[int | None, ...]
    dtype: Any


class _SweepReport(NamedTuple):
    """What a sweep observed, so a test can assert it was not vacuous.

    Attributes:
        cases (int): Cases actually compared.
        coefficients (int): Truncation coefficients compared, over all cases.
        bit_identical (int): How many of those agreed bit for bit. Reported, never
            required; see the module docstring.
        truncated_cases (int): Cases whose truncation produced at least one coefficient.
        worst_ratio (float): Largest ratio of an observed difference to its own bound.
    """

    cases: int
    coefficients: int
    bit_identical: int
    truncated_cases: int
    worst_ratio: float


def _bindings() -> Any:
    """Import the extension, deferred and in one place.

    Module level would turn an installation without the extension into a collection
    error for this whole file, including the tests that state a property of the oracle
    alone.

    Returns:
        Any: The :mod:`pantr._pantr_cpp` module.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    return _pantr_cpp


def _open_knots(degree: int, num_elements: int, lo: float, hi: float) -> npt.NDArray[np.float64]:
    """A clamped uniform knot vector.

    Args:
        degree (int): Polynomial degree.
        num_elements (int): Number of equal spans.
        lo (float): Domain start.
        hi (float): Domain end.

    Returns:
        npt.NDArray[np.float64]: The knot vector, ``num_elements + 1 + 2 * degree`` long.
    """
    inner = np.linspace(lo, hi, num_elements + 1)
    return np.concatenate([np.full(degree, lo), inner, np.full(degree, hi)])


@contextlib.contextmanager
def _the_oracle() -> Iterator[None]:
    """Run a block with the Python backend in effect.

    **Every oracle call that constructs a space needs this, not only the constructor.**
    ``THBSplineSpace`` is still pure Python and builds its per-level spaces by calling
    ``BsplineSpace1D.subdivide``, which dispatches on the *ambient* backend -- so
    ``refine``, ``refine_region`` and ``coarsen``, each of which rebuilds, would under
    ``PANTR_BACKEND=cpp`` produce level spaces holding C++ handles over a root space
    holding Python ones, and ``BsplineSpace`` refuses that mixture by design.

    That is not a defect in the oracle: it is what
    ``design/cross_backend_types.md`` forbids and what
    ``_bspline_space_nd._new_impl`` exists to catch. It is a property of this file's
    oracle *helper*, and it only shows under the backend CI's own parity job runs the
    whole suite with. Both sweeps failed under it before this was added, and passed
    under the default backend, which is exactly the shape ``CLAUDE.md`` warns about --
    a local green that the C++ leg does not share.

    Yields:
        None: With the Python backend selected for the calling thread.
    """
    with use_backend(Backend.PYTHON):
        yield


def _space(case: _Case, backend: Backend) -> THBSplineSpace:
    """Build a case's space through the public class, under one backend.

    One builder for both sides since FELIGN/pantr#494: the class dispatches, so the
    backend in effect while it is constructed is the whole difference between the oracle
    and the port. Everything below is the ordinary public construction path a caller
    would write.

    Args:
        case (_Case): The hierarchy to build.
        backend (Backend): The backend to build it under.

    Returns:
        THBSplineSpace: The space, holding that backend's implementation.
    """
    with use_backend(backend):
        directions = [
            BsplineSpace1D(
                _open_knots(case.degrees[k], case.num_elements[k], *case.bounds[k]).astype(
                    case.dtype
                ),
                case.degrees[k],
            )
            for k in range(len(case.degrees))
        ]
        grid = hierarchical_grid(
            uniform_grid([list(b) for b in case.bounds], list(case.num_elements)),
            list(case.factor),
        )
        for level, lo, hi in case.refinements:
            grid = grid.refine(level, list(lo), list(hi))
        return THBSplineSpace(
            BsplineSpace(directions),
            grid,
            truncate=case.truncate,
            regularity=list(case.regularity),
        )


def _python_space(case: _Case) -> THBSplineSpace:
    """Build a case's space over the Python oracle.

    Args:
        case (_Case): The hierarchy to build.

    Returns:
        THBSplineSpace: The space, holding ``_THBSplineSpacePython``.
    """
    return _space(case, Backend.PYTHON)


def _cpp_space(case: _Case) -> THBSplineSpace:
    """Build a case's space over the C++ type.

    Args:
        case (_Case): The hierarchy to build.

    Returns:
        THBSplineSpace: The space, holding a ``THBSplineSpace32`` or
        ``THBSplineSpace64`` handle.
    """
    return _space(case, Backend.CPP)


def _coefficient_roundings(num_levels: int, shape: tuple[int, ...]) -> Roundings:
    """The rounding budget of one truncated function's coefficient box.

    ``N = (num_levels - 1) * sum_k shape[k]``; the module docstring carries the
    derivation and what makes it an upper bound rather than an estimate.

    Args:
        num_levels (int): The space's level count.
        shape (tuple[int, ...]): The stored coefficient box's per-direction width.

    Returns:
        Roundings: One accumulator rounding per stage, over ``N`` stages.
    """
    stages = max(1, (num_levels - 1) * sum(shape))
    return Roundings(stages=stages, accumulator_per_stage=1, storage_per_stage=0)


def _truncated_dofs(space: THBSplineSpace, num_total_basis: int) -> dict[int, Any]:
    """The truncation entries of a space, keyed by global dof.

    ``truncated`` answers per dof rather than handing out a mapping, so enumerating the
    truncated set means asking about every dof. That is the binding's shape and the
    wrapper's, and it is also what ``num_truncated`` exists to make checkable without
    the walk.

    Args:
        space (THBSplineSpace): The space to read.
        num_total_basis (int): How many dofs to ask about.

    Returns:
        dict[int, Any]: ``dof -> (rep_level, box_lo, coeffs)`` for every truncated
        function.
    """
    entries: dict[int, Any] = {}
    for dof in range(num_total_basis):
        entry = space.truncated(dof)
        if entry is not None:
            entries[dof] = entry
    assert len(entries) == space.num_truncated, (
        f"num_truncated says {space.num_truncated} but {len(entries)} dofs answer "
        f"truncated() with an entry"
    )
    return entries


def _partition_of_unity_defect(space: THBSplineSpace, oracle: THBSplineSpace) -> float:
    """How far the active basis is from summing to one, in the finest level's basis.

    Pushes every active function's coefficient vector to the finest level by pure
    two-scale refinement -- no truncation -- and returns
    ``max |sum_i c_i - 1|`` over the finest tensor-product basis.

    The two-scale kernel is the oracle's, used as shared machinery: what is under test is
    ``space``'s *selection and truncation*, and the kernel is independently pinned in
    ``cpp/tests/test_bspline_knot_insertion.cpp`` against the cardinal B-spline's binomial
    stencil. The module docstring says why that composition is honest.

    Args:
        space (THBSplineSpace): The space whose coefficients are checked.
        oracle (THBSplineSpace): The oracle for the same hierarchy, which supplies the
            level spaces the two-scale matrices are built from. Passing it explicitly
            rather than reading it off ``space`` is what keeps the two-scale machinery the
            oracle's while the coefficients graded are ``space``'s.

    Returns:
        float: The largest absolute deviation from one.
    """
    from pantr.bspline._bspline_knot_insertion_core import (  # noqa: PLC0415
        _compute_oslo_matrix_1d_core,
    )

    dim = oracle.dim
    top = oracle.num_levels - 1
    oslo = [
        [
            np.asarray(
                _compute_oslo_matrix_1d_core(
                    oracle.level_space(m).spaces[k].degree,
                    oracle.level_space(m).spaces[k].knots,
                    oracle.level_space(m + 1).spaces[k].knots,
                ),
                dtype=np.float64,
            )
            for k in range(dim)
        ]
        for m in range(top)
    ]
    total = np.zeros(tuple(oracle.level_space(top).num_basis), dtype=np.float64)
    entries = _truncated_dofs(space, oracle.num_total_basis)

    for dof in range(oracle.num_total_basis):
        entry = entries.get(dof)
        if entry is None:
            level = oracle.dof_level(dof)
            position = dof - int(oracle.level_offsets[level])
            flat = int(oracle.active_function_indices(level)[position])
            multi = np.unravel_index(flat, oracle.level_space(level).num_basis)
            box_lo = [int(m) for m in multi]
            coeffs = np.ones((1,) * dim, dtype=np.float64)
            start = level
        else:
            start = int(entry[0])
            box_lo = [int(x) for x in entry[1]]
            coeffs = np.asarray(entry[2], dtype=np.float64)

        for level in range(start, top):
            new_lo: list[int] = []
            out = coeffs
            for k in range(dim):
                alpha = oslo[level][k]
                cols = alpha[:, box_lo[k] : box_lo[k] + out.shape[k]]
                rows = np.nonzero(np.any(cols != 0.0, axis=1))[0]
                lo_row, hi_row = int(rows[0]), int(rows[-1]) + 1
                sub = alpha[lo_row:hi_row, box_lo[k] : box_lo[k] + out.shape[k]]
                out = np.moveaxis(np.tensordot(sub, out, axes=([1], [k])), 0, k)
                new_lo.append(lo_row)
            coeffs, box_lo = out, new_lo

        window = tuple(slice(box_lo[k], box_lo[k] + coeffs.shape[k]) for k in range(dim))
        total[window] += coeffs

    return float(np.abs(total - 1.0).max())


def _compare(case: _Case, variant: int = 0) -> tuple[int, int, float]:
    """Compare both backends' spaces for one case, quantity by quantity.

    Args:
        case (_Case): The hierarchy to build under both.
        variant (int): Which rebuilding operation to compare on this case; see
            :func:`_compare_the_operations` for why it is one rather than all four.

    Returns:
        tuple[int, int, float]: How many truncation coefficients were compared, how many
        of those were bit-identical, and the worst ratio of a difference to its bound.

    Raises:
        AssertionError: If any quantity violates its claim.
    """
    # A draw can name a configuration both backends refuse -- a `float32` mesh on a
    # domain at 1e6 is finer than the format resolves there, and knot snapping collapses
    # it. Skipping it would lose a comparison; requiring the same refusal from both keeps
    # it as one, and the refusals are the part of a port most easily left un-ported.
    try:
        py = _python_space(case)
    except ValueError as refused:
        with pytest.raises(ValueError) as caught:
            _cpp_space(case)
        assert str(caught.value) == str(refused), (
            f"the two backends refuse {case!r} with different messages:\n"
            f"  python: {refused}\n  cpp:    {caught.value}"
        )
        return 0, 0, 0.0
    cpp = _cpp_space(case)
    context = f"THBSplineSpace{case!r}"

    fields = [
        Field("num_levels", exact_parity(why=_VERDICT_WHY)),
        Field("num_total_basis", exact_parity(why=_VERDICT_WHY)),
        Field("num_basis_per_level", exact_parity(why=_VERDICT_WHY)),
        Field("num_truncated", exact_parity(why=_VERDICT_WHY)),
        Field("dim", exact_parity(why=_VERDICT_WHY)),
        Field("degrees", exact_parity(why=_VERDICT_WHY)),
        Field("truncate", exact_parity(why=_VERDICT_WHY)),
        Field(
            "level_offsets",
            exact_parity(why=_VERDICT_WHY),
            read=lambda space: np.asarray(space.level_offsets, dtype=np.int64),
        ),
    ]
    for level in range(py.num_levels):
        fields.append(
            Field(
                f"active_function_indices[{level}]",
                exact_parity(why=_VERDICT_WHY),
                read=lambda space, level=level: np.asarray(  # type: ignore[misc]
                    space.active_function_indices(level), dtype=np.int64
                ),
            )
        )
    assert_object_parity(py=py, cpp=cpp, fields=fields, context=context)
    # `regularity` is compared here rather than as a Field: it mixes `None` with `int`,
    # and the harness refuses a field whose value mixes element kinds -- rightly, since
    # stacking them would promote to a common dtype. It is a tuple of verdicts either way.
    assert py.regularity == cpp.regularity, (
        f"{context}: regularity disagrees; python {py.regularity!r} vs cpp {cpp.regularity!r}"
    )
    _compare_the_level_knots(py, cpp, context)

    assert py.max_active_per_cell() == cpp.max_active_per_cell(), (
        f"{context}: max_active_per_cell disagrees. {_VERDICT_WHY}"
    )
    for cid in range(py.grid.num_cells):
        expected = np.asarray(py.active_basis(cid), dtype=np.int64)
        actual = np.asarray(cpp.active_basis(cid), dtype=np.int64)
        assert np.array_equal(expected, actual), (
            f"{context}: active_basis({cid}) disagrees. {_VERDICT_WHY}"
        )
        for name, expected_block, actual_block in zip(
            ("dofs", "levels", "multi-indices"),
            py.contributions(cid),
            cpp.contributions(cid),
            strict=True,
        ):
            assert np.array_equal(expected_block, actual_block), (
                f"{context}: the contribution {name} of cell {cid} disagree. {_VERDICT_WHY}"
            )

    for dof in range(py.num_total_basis):
        assert py.dof_level(dof) == cpp.dof_level(dof), f"{context}: dof_level({dof}) disagrees"

    py_entries = _truncated_dofs(py, py.num_total_basis)
    cpp_entries = _truncated_dofs(cpp, py.num_total_basis)
    assert set(py_entries) == set(cpp_entries), (
        f"{context}: the set of truncated dofs disagrees; "
        f"only python {sorted(set(py_entries) - set(cpp_entries))}, "
        f"only cpp {sorted(set(cpp_entries) - set(py_entries))}"
    )

    compared = 0
    identical = 0
    worst = 0.0
    for dof, py_entry in sorted(py_entries.items()):
        cpp_entry = cpp_entries[dof]
        expected = np.ascontiguousarray(py_entry[2], dtype=np.float64)
        actual = np.ascontiguousarray(cpp_entry[2], dtype=np.float64)
        where = f"{context}: dof {dof}"
        assert int(py_entry[0]) == int(cpp_entry[0]), f"{where}: rep_level disagrees"
        assert tuple(py_entry[1]) == tuple(int(x) for x in cpp_entry[1]), (
            f"{where}: box origin disagrees"
        )
        assert expected.shape == actual.shape, f"{where}: box shape disagrees"

        observed = assert_object_parity(
            py=expected,
            cpp=actual,
            fields=[
                Field(
                    "coefficients",
                    bounded_parity(
                        roundings=_coefficient_roundings(py.num_levels, expected.shape),
                        accumulator=np.float64,
                        storage=np.float64,
                        amplification=np.abs(expected).ravel(),
                        why=_COEFFICIENT_WHY,
                    ),
                    read=lambda block: np.asarray(block, dtype=np.float64).ravel(),
                )
            ],
            context=where,
        )
        compared += int(expected.size)
        identical += int(
            (expected.view(np.uint64) == actual.view(np.uint64)).sum() if expected.size else 0
        )
        worst = max(worst, observed["coefficients"].max_ratio_to_bound)

    _compare_the_operations(py, cpp, context, variant)
    return compared, identical, worst


def _compare_the_level_knots(py: THBSplineSpace, cpp: THBSplineSpace, context: str) -> None:
    """Grade the per-level knot vectors, which is what the unported operations rest on.

    Basis tabulation, the windowed restriction and the prolongation live on the wrapper
    and are one body under both backends, so the only way they can disagree is through
    their inputs: the truncation coefficients, compared elsewhere here, and these. The
    two implementations build them by different code -- ``BsplineSpace1D.subdivide`` on
    one side, ``build_level_spaces`` on the other -- so agreement is a claim rather than
    a tautology.

    The bound is **the space's own** ``tolerance``: an absolute parametric tolerance the
    library derives for deciding whether two knots are the same one, which is exactly the
    question here -- two coordinates obtained by different routes, which is the case its
    own derivation is written for. No constant is minted for it.

    The two routes were **engineered** to agree rather than merely observed to:
    ``cpp/include/pantr/bspline/knot_insertion.hpp`` spells the subdivision points as
    ``numpy``'s own expression rather than a tidier equivalent, and says so. So a run
    where this fires at all is a run where that agreement broke, and the tolerance is
    what separates "broke by an ulp" from "broke". Bit-identity is not required and is
    not counted.

    Args:
        py (THBSplineSpace): The oracle's space.
        cpp (THBSplineSpace): The port's space for the same hierarchy.
        context (str): What is being compared, quoted in every failure message.

    Raises:
        AssertionError: If a level's shape or knots disagree beyond the tolerance.
    """
    for level in range(py.num_levels):
        for k in range(py.dim):
            one = py.level_space(level).spaces[k]
            other = cpp.level_space(level).spaces[k]
            expected = np.asarray(one.knots, dtype=np.float64)
            actual = np.asarray(other.knots, dtype=np.float64)
            where = f"{context}: level {level}, direction {k}"
            assert expected.shape == actual.shape, f"{where}: the knot vectors differ in length"
            tolerance = max(float(one.tolerance), float(other.tolerance))
            worst = float(np.abs(expected - actual).max()) if expected.size else 0.0
            assert worst <= tolerance, (
                f"{where}: the knot vectors differ by {worst:.3e}, above the space's own "
                f"knot tolerance {tolerance:.3e}"
            )


def _compare_the_operations(
    py: THBSplineSpace, cpp: THBSplineSpace, context: str, variant: int
) -> None:
    """Compare one rebuilding operation of an already-compared pair.

    Both sides rebuild: the operation returns a new space over a new grid, so what is
    compared is the space that comes back. Folded into the sweep rather than pinned to one
    fixture because the ordering hazards here are the ones a fixture does not reach --
    FELIGN/pantr#395's own port carried a defect in ``coarsen_cells``' demotion order
    that first appeared at case 3553 of a 4000-case sweep and was invisible at 400.

    **One of the four variants per case, rotating, rather than all four.** Constructing a
    space is by far the most expensive thing this sweep does, so comparing all four
    quadruples its cost for coverage a rotation buys at a quarter of the price: every
    variant still gets a quarter of the cases, which at the wide width is hundreds each.
    The cost is what forces the choice: this sweep is the slowest test in ``tests/parity``
    and the C++ CI job runs it twice, once at each width.

    The marked set is derived from the space rather than drawn again, so a failure is
    reproducible from the case alone: the first cell of each level for the refinement, and
    for the coarsening **every cell above level 0**, not only the deepest ones.

    That last word is the load-bearing one. Both implementations demote parents deepest
    first, so that a veto is decided against a mesh whose finer coarsenings have already
    happened -- and a marked set confined to one level puts every parent at one level too,
    where the ordering cannot be observed at all. Measured: reversing the demotion order
    in the C++ survived this sweep while the marked set was the deepest level alone, and
    is caught once it spans several. FELIGN/pantr#395's own port carried a defect of
    exactly this shape, in exactly this operation.

    Args:
        py (THBSplineSpace): The oracle's space.
        cpp (THBSplineSpace): The port's space for the same hierarchy.
        context (str): What is being compared, quoted in every failure message.
        variant (int): Selects the operation, modulo four.

    Raises:
        AssertionError: If any rebuilt quantity disagrees.
    """
    seen: set[int] = set()
    one_per_level: list[int] = []
    for cid in range(py.grid.num_cells):
        level = py.grid.cell_level(cid)
        if level not in seen:
            seen.add(level)
            one_per_level.append(cid)
    # Every cell above level 0 for the coarsening, not only the deepest: see the note
    # in the docstring on what a single-level marked set cannot see.
    above_the_root = [cid for cid in range(py.grid.num_cells) if py.grid.cell_level(cid) >= 1]

    coarsening = variant % 4 >= 2
    admissible: int | None = 2 if variant % 2 == 0 else None
    marked = np.array(above_the_root if coarsening else one_per_level, dtype=np.int64)
    kind = "coarsen" if coarsening else "refine"
    where = f"{context}: {kind}({'graded' if admissible is not None else 'ungraded'})"

    with _the_oracle():
        expected = (
            py.coarsen(marked, admissible_class=admissible)
            if coarsening
            else py.refine(marked, admissible_class=admissible)
        )
    actual = (
        cpp.coarsen(marked, admissible_class=admissible)
        if coarsening
        else cpp.refine(marked, admissible_class=admissible)
    )

    assert expected.num_levels == actual.num_levels, f"{where}: num_levels disagrees"
    assert expected.grid.num_cells == actual.grid.num_cells, (
        f"{where}: the rebuilt grid has a different cell count"
    )
    assert expected.num_total_basis == actual.num_total_basis, f"{where}: num_total_basis disagrees"
    for level in range(expected.num_levels):
        assert np.array_equal(
            expected.active_function_indices(level),
            np.asarray(actual.active_function_indices(level)),
        ), f"{where}: the level-{level} active set disagrees"


def _draw_case(rng: np.random.Generator) -> _Case | None:
    """Draw one random hierarchy.

    Args:
        rng (np.random.Generator): The source of randomness.

    Returns:
        _Case | None: The case, or ``None`` when the draw produced a degenerate
        refinement box that both backends would refuse.
    """
    dim = int(rng.integers(1, 4))
    degrees = tuple(int(rng.integers(1, 4)) for _ in range(dim))
    num_elements = tuple(int(rng.integers(2, 6)) for _ in range(dim))
    factor = [int(rng.integers(1, 4)) for _ in range(dim)]
    if all(f == 1 for f in factor):
        # A hierarchy that refines nothing anywhere has one level and no truncation, so
        # it would compare the constructor and nothing else.
        factor[0] = 2
    # Three domains, one of them far from the origin, so a scale-dependent defect in the
    # subdivision points has somewhere to show.
    domain_lo = float(rng.choice([0.0, -1.0, 1.0e6]))
    width = float(rng.choice([1.0, 1.0e-6, 3.0]))
    bounds = tuple((domain_lo, domain_lo + width) for _ in range(dim))
    regularity = tuple(
        None if rng.integers(0, 2) else int(rng.integers(0, degrees[k])) for k in range(dim)
    )

    # Each box is drawn inside the window the previous refinement created, at that
    # window's own level-`level` indices -- `[lo * factor, hi * factor)` -- so every
    # refinement lands on cells that exist and are active leaves. Drawing in
    # `[0, extent)` instead would name a box outside the level's domain as soon as the
    # first window did not start at the origin, which is what a first version of this
    # generator did.
    refinements: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    window_lo = [0] * dim
    extent = list(num_elements)
    for level in range(int(rng.integers(1, 4))):
        lo = tuple(window_lo[k] + int(rng.integers(0, extent[k])) for k in range(dim))
        hi = tuple(
            min(window_lo[k] + extent[k], lo[k] + int(rng.integers(1, 3))) for k in range(dim)
        )
        if any(hi[k] <= lo[k] for k in range(dim)):
            return None
        refinements.append((level, lo, hi))
        window_lo = [lo[k] * factor[k] for k in range(dim)]
        extent = [(hi[k] - lo[k]) * factor[k] for k in range(dim)]
        if any(n == 0 for n in extent):
            break
    if not refinements:
        return None

    return _Case(
        degrees=degrees,
        num_elements=num_elements,
        factor=tuple(factor),
        bounds=bounds,
        refinements=tuple(refinements),
        truncate=bool(rng.integers(0, 2)),
        regularity=regularity,
        dtype=np.float32 if rng.integers(0, 4) == 0 else np.float64,
    )


def _sweep(cases: int, seed: int) -> _SweepReport:
    """Draw and compare random hierarchies.

    Args:
        cases (int): How many draws to attempt.
        seed (int): The generator's seed, so a failure is reproducible.

    Returns:
        _SweepReport: What the sweep observed.
    """
    rng = np.random.default_rng(seed)
    compared = 0
    identical = 0
    truncated_cases = 0
    worst = 0.0
    ran = 0
    for _ in range(cases):
        case = _draw_case(rng)
        if case is None:
            continue
        case_compared, case_identical, case_worst = _compare(case, variant=ran)
        ran += 1
        compared += case_compared
        identical += case_identical
        truncated_cases += int(case_compared > 0)
        worst = max(worst, case_worst)
    return _SweepReport(
        cases=ran,
        coefficients=compared,
        bit_identical=identical,
        truncated_cases=truncated_cases,
        worst_ratio=worst,
    )


def _assert_not_vacuous(report: _SweepReport) -> None:
    """Refuse a sweep that compared nothing worth comparing.

    A sweep whose draws all degenerated, or none of which truncated anything, would pass
    while exercising neither the truncation nor the bound. Both are what this file is for.

    Args:
        report (_SweepReport): What the sweep observed.

    Raises:
        AssertionError: If the sweep is vacuous on either count.
    """
    assert report.cases > 0, "every draw degenerated; the sweep compared nothing"
    assert report.truncated_cases > 0, (
        "no case in the sweep produced a truncation coefficient, so the bound was never "
        "exercised and this run says nothing about the half of the port that has digits"
    )
    assert report.worst_ratio <= 1.0, "a difference exceeded its bound"


def test_the_sweep_agrees(cpp_backend: None) -> None:
    """The two backends build the same space, over a sweep of random hierarchies.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    _assert_not_vacuous(_sweep(_SHIPPED_CASES, seed=20260904))


@pytest.mark.slow
def test_the_wide_sweep_agrees(cpp_backend: None) -> None:
    """The same sweep, ten times wider.

    FELIGN/pantr#397 asks that the bound be verified over a sweep at least ten times the
    one that ships. Keeping the wide run in the repository rather than in a shell history
    is what makes "verified" something the next reader can re-run.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    _assert_not_vacuous(_sweep(_SHIPPED_CASES * _WIDE_FACTOR, seed=20260905))


def _reference_case(*, truncate: bool = True, dim: int = 2) -> _Case:
    """A small hierarchy with a real truncation in it, for the pinned tests.

    Args:
        truncate (bool): Whether the truncated basis is built.
        dim (int): Parametric dimension.

    Returns:
        _Case: A two-refinement dyadic hierarchy on the unit domain.
    """
    return _Case(
        degrees=(2,) * dim,
        num_elements=(4,) * dim,
        factor=(2,) * dim,
        bounds=((0.0, 1.0),) * dim,
        refinements=((0, (0,) * dim, (2,) * dim), (1, (0,) * dim, (2,) * dim)),
        truncate=truncate,
        regularity=(None,) * dim,
        dtype=np.float64,
    )


def test_the_truncated_basis_is_a_partition_of_unity(cpp_backend: None) -> None:
    """The C++ truncation satisfies the identity that defines the truncated basis.

    Giannelli-Jüttler-Speleers (2012), Thm 6. The module docstring says why this is an
    independent check rather than a rerun of the port, and what shared machinery it uses.

    The bound is the same one the parity claim carries, evaluated at this case's own box
    sizes and summed over the functions that overlap one finest-level function -- at most
    ``prod(degree + 1)`` of them, since that is how many tensor-product functions of any
    one level are non-zero on a cell, and the levels a hierarchical function can come from
    are at most ``num_levels``.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    case = _reference_case()
    oracle = _python_space(case)
    defect = _partition_of_unity_defect(_cpp_space(case), oracle)

    overlapping = math.prod(d + 1 for d in oracle.degrees) * oracle.num_levels
    widest = max(
        (max(entry[2].shape) for entry in _truncated_dofs(oracle, oracle.num_total_basis).values()),
        default=1,
    )
    stages = (oracle.num_levels - 1) * oracle.dim * widest
    gamma = stages * _U / (1.0 - stages * _U)
    bound = overlapping * gamma
    assert defect <= bound, (
        f"the truncated basis is not a partition of unity: worst defect {defect:.3e} "
        f"against {bound:.3e}"
    )
    assert bound < 1.0e-10, (
        "the vacuity guard: this bound is loose enough to accept a basis that is not a "
        "partition of unity at all, so passing it would say nothing"
    )


def test_the_untruncated_basis_fails_the_identity(cpp_backend: None) -> None:
    """The same computation on the HB basis is nowhere near one.

    Without this the identity above would be consistent with a check that cannot fail --
    the truncation is exactly what restores the partition of unity, so a run that reported
    a small defect for the *untruncated* basis would be measuring nothing.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    case = _reference_case(truncate=False)
    oracle = _python_space(case)
    defect = _partition_of_unity_defect(_cpp_space(case), oracle)
    assert defect > 0.1, (
        f"the untruncated basis summed to within {defect:.3e} of one, so the identity "
        f"above cannot distinguish a truncated basis from an untruncated one"
    )


_VANISHING_CASES: Final = (
    _reference_case(dim=1),
    _reference_case(dim=2),
    _reference_case(dim=3)._replace(refinements=((0, (0, 0, 0), (2, 2, 2)),)),
    _reference_case()._replace(degrees=(3, 2), factor=(3, 2), regularity=(1, 0)),
    _reference_case()._replace(dtype=np.float32),
)
"""Hierarchies in which some truncated functions vanish on some cells."""


@pytest.mark.parametrize(
    "case", _VANISHING_CASES, ids=["1d", "2d", "3d", "factor32-c1c0", "float32"]
)
def test_no_backend_lists_a_function_that_vanishes_on_the_cell(
    cpp_backend: None, case: _Case
) -> None:
    """Both backends list, per cell, exactly the functions that do not vanish there.

    The expected list does not come from either space's contribution table. It comes
    from :class:`~pantr.bspline.MultiLevelExtraction`'s windowed kernel, which pushes each
    function supported on the cell through the two-scale blocks restricted to the cell's
    windows and flags the rows that come out structurally zero -- a different route over
    different data from the stored truncation coefficients both spaces read. The kernel's
    own flags are pinned against direct evaluation in
    ``tests/test_multilevel_extraction.py``.

    Args:
        cpp_backend (None): Requires the compiled extension.
        case (_Case): The hierarchy to build under both backends.
    """
    from pantr.bspline import MultiLevelExtraction  # noqa: PLC0415

    py = _python_space(case)
    cpp = _cpp_space(case)
    with _the_oracle():
        ext = MultiLevelExtraction(py)
    dropped = 0
    widest = 0
    wrong: dict[str, list[int]] = {"python": [], "cpp": []}
    for cid in range(py.grid.num_cells):
        _, dofs, nonzero = ext._windowed_rows(cid)
        expected = dofs[nonzero]
        widest = max(widest, int(expected.size))
        dropped += int((~nonzero).sum())
        for name, space in (("python", py), ("cpp", cpp)):
            if not np.array_equal(np.asarray(space.active_basis(cid)), expected):
                wrong[name].append(cid)
    assert dropped > 0, "no function vanishes on any cell of this case, so it tests nothing"
    assert wrong == {"python": [], "cpp": []}, f"cells whose active list is wrong: {wrong}"
    assert py.max_active_per_cell() == widest
    assert cpp.max_active_per_cell() == widest


def test_a_float32_root_space_over_a_float64_grid_agrees(cpp_backend: None) -> None:
    """The shipped pairing of a narrow root space with the always-`float64` grid.

    ``pantr.grid`` is ``float64``-only by its own port's ruling while a root B-spline
    space stores whatever it was handed, and ``tests/test_thb_spline_space.py``'s
    ``test_a_float32_root_space_is_still_graded_in_float64`` ships that combination. The
    C++ type's grid scalar is therefore ``double`` independently of its own, and this is
    what pins that the pairing is representable and agrees.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    case = _reference_case()._replace(dtype=np.float32)
    compared, _, worst = _compare(case)
    assert compared > 0, "this case truncated nothing, so it exercised no float32 knots"
    assert worst <= 1.0


@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.CPP], ids=["python", "cpp"])
def test_level_space_zero_shares_the_root_handle(cpp_backend: None, backend: Backend) -> None:
    """``level_space(0)`` hands back the object the space was built from, not a copy.

    ``design/bspline_ownership_lifetime.md`` F6 makes this an identity contract rather
    than a convenience, and since FELIGN/pantr#494 it has two halves. At the wrapper
    level ``thb.level_space(0) is thb.root_space`` must hold under both backends, which
    only the wrapper can supply: no C++ object can hand back the constructor argument's
    own Python object. One level down the C++ handles must be the same handle too, or
    two Python objects would agree on identity over two different C++ objects -- which is
    what fixes the C++ constructor's signature to take handles rather than values.

    Args:
        cpp_backend (None): Requires the compiled extension.
        backend (Backend): The backend the space is built under.
    """
    space = _space(_reference_case(), backend)
    assert space.level_space(0) is space.root_space
    assert space.level_space(0)._impl is space.root_space._impl
    # And the finer levels are genuinely different objects, or the assertion above would
    # hold for a type that returned one space for every level.
    assert space.level_space(1) is not space.root_space
    assert space.level_space(1)._impl is not space.root_space._impl
    # Memoised, so the same level gives the same wrapper twice rather than a fresh one
    # over the same handle -- the second half of the same contract.
    assert space.level_space(1) is space.level_space(1)


def test_every_rebuilding_oracle_call_stays_wholly_python(cpp_backend: None) -> None:
    """A rebuilt oracle space is Python at every level, and without the guard it is not.

    This is the test :func:`_the_oracle` was missing. ``THBSplineSpace`` rebuilds itself
    inside ``refine``, ``refine_region`` and ``coarsen`` by calling
    ``BsplineSpace1D.subdivide``, which dispatches on the *ambient* backend -- so under
    ``PANTR_BACKEND=cpp`` an unguarded rebuild hands back a space whose level 0 is the
    Python root it was built from and whose finer levels are C++ handles.

    **That mixture does not raise, which is what makes it worth a test.** It is silent,
    and a silently hybrid oracle delegates its finer levels to the very C++ knot
    insertion this cut ports, so a defect there would appear on both sides of every
    comparison built on it and no parity assertion could see it. The two pinned tests
    below cannot catch it: their case has a uniform ``factor``, every axis subdivides
    together, and ``_bspline_space_nd._new_impl``'s refusal never fires.

    The second half is a control rather than a second assertion. Without it, a change
    that made ``subdivide`` backend-stable would leave the guard decorative and this test
    still green; with it, that change turns the control red and says so.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """

    def levels(space: THBSplineSpace) -> list[type]:
        return [type(space.level_space(k)._impl) for k in range(space.num_levels)]

    with use_backend(Backend.CPP):
        oracle = _python_space(_reference_case())
        marked = np.array([0, 1, 2], dtype=np.int64)
        deepest = np.array(
            [c for c in range(oracle.grid.num_cells) if oracle.grid.cell_level(c) == 2],
            dtype=np.int64,
        )
        rebuilds: tuple[tuple[str, Callable[[], THBSplineSpace]], ...] = (
            ("refine", lambda: oracle.refine(marked, admissible_class=None)),
            ("refine_region", lambda: oracle.refine_region(0, [0, 0], [2, 2], admissible_class=2)),
            ("coarsen", lambda: oracle.coarsen(deepest, admissible_class=None)),
        )

        assert levels(oracle) == [_BsplineSpaceNDPython] * oracle.num_levels

        for name, rebuild in rebuilds:
            with _the_oracle():
                guarded = rebuild()
            assert levels(guarded) == [_BsplineSpaceNDPython] * guarded.num_levels, (
                f"{name} under the guard left a level that is not the oracle's own type: "
                f"{[t.__name__ for t in levels(guarded)]}"
            )

        for name, rebuild in rebuilds:
            unguarded = levels(rebuild())
            assert unguarded[1:] and all(
                impl is not _BsplineSpaceNDPython for impl in unguarded[1:]
            ), (
                f"the control failed: {name} outside the guard produced a wholly Python "
                f"space anyway ({[t.__name__ for t in unguarded]}), so the ambient backend "
                f"no longer reaches BsplineSpace1D.subdivide and _the_oracle is dead weight"
            )


def test_refinement_and_coarsening_agree(cpp_backend: None) -> None:
    """The graded refine, the ungraded refine, and coarsen, all agree afterwards.

    Each returns a new space, so what is compared is the space it returns: the level
    count, the active sets and the truncation the rebuilt space carries.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    case = _reference_case()
    oracle = _python_space(case)
    cpp = _cpp_space(case)
    marked = np.array([0, 1, 2], dtype=np.int64)

    for admissible in (2, None):
        with _the_oracle():
            expected = oracle.refine(marked, admissible_class=admissible)
        with use_backend(Backend.CPP):
            actual = cpp.refine(marked, admissible_class=admissible)
        assert expected.num_levels == actual.num_levels
        assert expected.num_total_basis == actual.num_total_basis
        assert tuple(expected.num_basis_per_level) == tuple(actual.num_basis_per_level)
        for level in range(expected.num_levels):
            assert np.array_equal(
                expected.active_function_indices(level),
                np.asarray(actual.active_function_indices(level)),
            )

    with _the_oracle():
        expected = oracle.refine_region(0, [0, 0], [2, 2], admissible_class=2)
    with use_backend(Backend.CPP):
        actual = cpp.refine_region(0, [0, 0], [2, 2], admissible_class=2)
    assert expected.num_total_basis == actual.num_total_basis

    finest = np.array(
        [c for c in range(oracle.grid.num_cells) if oracle.grid.cell_level(c) == 2],
        dtype=np.int64,
    )
    with _the_oracle():
        expected = oracle.coarsen(finest, admissible_class=None)
    with use_backend(Backend.CPP):
        actual = cpp.coarsen(finest, admissible_class=None)
    assert expected.num_levels == actual.num_levels
    assert expected.num_total_basis == actual.num_total_basis
    for level in range(expected.num_levels):
        assert np.array_equal(
            expected.active_function_indices(level),
            np.asarray(actual.active_function_indices(level)),
        )


@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.CPP], ids=["python", "cpp"])
def test_refine_and_coarsen_never_hand_back_the_receivers_grid(
    cpp_backend: None, backend: Backend
) -> None:
    """A no-op refinement still returns a space over a grid of its own.

    Two spaces sharing one grid would make a tag set through one visible through the
    other, which is what the oracle's ``grid._copy()`` prevents and what
    ``HierarchicalGrid::refine_cells`` with no ids does on the C++ side. The root grid
    is shared on purpose, and that is asserted beside it so the two are not confused.

    Args:
        cpp_backend (None): Requires the compiled extension.
        backend (Backend): The backend the space is built under.
    """
    space = _space(_reference_case(), backend)
    nothing = np.array([], dtype=np.int64)
    with use_backend(backend):
        for derived in (space.refine(nothing), space.coarsen(nothing)):
            assert derived.grid is not space.grid
            assert derived.grid._impl is not space.grid._impl
            assert derived.grid.root is space.grid.root
            assert derived.root_space is space.root_space


# ----------------------------------------------------------------------------
# The wrapper itself: construction, the forwards, and the round trip
# ----------------------------------------------------------------------------

_PROBE_REGION: Final = ([2, 2], [4, 4])
"""The level-0 box the ``refine_region`` probes refine.

Outside the reference case's own refinement, so both gradings change the space, and
positioned so the graded one pulls in a neighbourhood the ungraded one does not.
"""


def _cells_at(space: Any, level: int) -> npt.NDArray[np.int64]:
    """The flat ids of every active cell at one level, from a space or a raw handle.

    Both carry a ``grid`` answering ``num_cells`` and ``cell_level``, which is all this
    needs, so one reader serves the wrapper and the handle it holds.

    Args:
        space (Any): The space or handle to read.
        level (int): The level to collect.

    Returns:
        npt.NDArray[np.int64]: The ids, ascending; empty if the level has none.
    """
    grid = space.grid
    return np.ascontiguousarray(
        [cid for cid in range(grid.num_cells) if grid.cell_level(cid) == level], dtype=np.int64
    )


def _refine_pair_wrapper(space: THBSplineSpace) -> tuple[Any, Any]:
    """Fingerprint a graded and an ungraded refinement, through the public class.

    Args:
        space (THBSplineSpace): The space to refine.

    Returns:
        tuple[Any, Any]: The graded and ungraded fingerprints.
    """
    marked = _cells_at(space, 1)[:1]
    return (
        _fingerprint(space.refine(marked, admissible_class=2)),
        _fingerprint(space.refine(marked, admissible_class=None)),
    )


def _refine_pair_impl(impl: Any) -> tuple[Any, Any]:
    """The same two refinements, through the implementation the wrapper holds.

    Args:
        impl (Any): The oracle instance or the C++ handle.

    Returns:
        tuple[Any, Any]: The graded and ungraded fingerprints.
    """
    marked = _cells_at(impl, 1)[:1]
    return (_fingerprint(impl.refine(marked, 2)), _fingerprint(impl.refine(marked, None)))


def _refine_region_pair_wrapper(space: THBSplineSpace) -> tuple[Any, Any]:
    """Fingerprint a graded and an ungraded region refinement, through the public class.

    Args:
        space (THBSplineSpace): The space to refine.

    Returns:
        tuple[Any, Any]: The graded and ungraded fingerprints.
    """
    lo, hi = _PROBE_REGION
    return (
        _fingerprint(space.refine_region(1, lo, hi, admissible_class=2)),
        _fingerprint(space.refine_region(1, lo, hi, admissible_class=None)),
    )


def _refine_region_pair_impl(impl: Any) -> tuple[Any, Any]:
    """The same two region refinements, through the implementation.

    Args:
        impl (Any): The oracle instance or the C++ handle.

    Returns:
        tuple[Any, Any]: The graded and ungraded fingerprints.
    """
    lo = np.ascontiguousarray(_PROBE_REGION[0], dtype=np.int64)
    hi = np.ascontiguousarray(_PROBE_REGION[1], dtype=np.int64)
    return (
        _fingerprint(impl.refine_region(1, lo, hi, 2)),
        _fingerprint(impl.refine_region(1, lo, hi, None)),
    )


def _coarsen_pair_wrapper(space: THBSplineSpace) -> tuple[Any, Any]:
    """Fingerprint a graded and an ungraded coarsening, through the public class.

    Args:
        space (THBSplineSpace): The space to coarsen.

    Returns:
        tuple[Any, Any]: The graded and ungraded fingerprints.
    """
    marked = _cells_at(space, 1)
    return (
        _fingerprint(space.coarsen(marked, admissible_class=2)),
        _fingerprint(space.coarsen(marked, admissible_class=None)),
    )


def _coarsen_pair_impl(impl: Any) -> tuple[Any, Any]:
    """The same two coarsenings, through the implementation.

    Args:
        impl (Any): The oracle instance or the C++ handle.

    Returns:
        tuple[Any, Any]: The graded and ungraded fingerprints.
    """
    marked = _cells_at(impl, 1)
    return (_fingerprint(impl.coarsen(marked, 2)), _fingerprint(impl.coarsen(marked, None)))


def _fingerprint(space: Any) -> tuple[int, int, tuple[int, ...]]:
    """A small, wholly integer summary of a space, for comparing two of them.

    The three rebuilding members return a space rather than a value, and a returned
    handle is a different object on every call, so identity says nothing about them. The
    counts do: they are what the rebuilt hierarchy is, and every one is an integer.

    Args:
        space (Any): A :class:`~pantr.bspline.THBSplineSpace` or a raw handle.

    Returns:
        tuple[int, int, tuple[int, ...]]: ``(num_levels, num_total_basis,
        num_basis_per_level)``.
    """
    return (
        int(space.num_levels),
        int(space.num_total_basis),
        tuple(int(count) for count in space.num_basis_per_level),
    )


_FORWARDS: Final[tuple[tuple[str, Callable[[Any], Any], Callable[[Any], Any]], ...]] = (
    # Held objects: the wrapper presents a wrapper, so what has to agree one level down
    # is the handle. `is` rather than equality, per
    # `design/bspline_ownership_lifetime.md` F6.
    ("root_space", lambda s: s.root_space._impl, lambda i: i.root_space),
    ("grid", lambda s: s.grid._impl, lambda i: i.grid),
    ("level_space", lambda s: s.level_space(1)._impl, lambda i: i.level_space(1)),
    # Scalars and tuples.
    ("dim", lambda s: s.dim, lambda i: i.dim),
    ("degrees", lambda s: s.degrees, lambda i: tuple(i.degrees)),
    ("num_levels", lambda s: s.num_levels, lambda i: i.num_levels),
    ("truncate", lambda s: s.truncate, lambda i: i.truncate),
    ("regularity", lambda s: s.regularity, lambda i: tuple(i.regularity)),
    ("num_total_basis", lambda s: s.num_total_basis, lambda i: i.num_total_basis),
    (
        "num_basis_per_level",
        lambda s: s.num_basis_per_level,
        lambda i: tuple(i.num_basis_per_level),
    ),
    ("tolerance", lambda s: s.tolerance, lambda i: i.tolerance),
    ("num_truncated", lambda s: s.num_truncated, lambda i: i.num_truncated),
    ("max_active_per_cell", lambda s: s.max_active_per_cell(), lambda i: i.max_active_per_cell()),
    ("dof_level", lambda s: s.dof_level(3), lambda i: i.dof_level(3)),
    # Arrays. The wrapper copies where its contract promises a writable array, so the
    # values must agree rather than the buffers.
    ("level_offsets", lambda s: np.asarray(s.level_offsets), lambda i: np.asarray(i.level_offsets)),
    ("domain", lambda s: np.asarray(s.domain), lambda i: np.asarray(i.domain)),
    (
        "active_function_indices",
        lambda s: np.asarray(s.active_function_indices(0)),
        lambda i: np.asarray(i.active_function_indices(0)),
    ),
    (
        "active_basis",
        lambda s: np.asarray(s.active_basis(0)),
        lambda i: np.asarray(i.active_basis(0)),
    ),
    (
        "contributions",
        lambda s: tuple(np.asarray(block) for block in s.contributions(0)),
        lambda i: tuple(np.asarray(block) for block in i.contributions(0)),
    ),
    (
        "truncated",
        lambda s: _read_truncated(s),
        lambda i: _read_truncated(i),
    ),
    # Rebuilding members: a returned space, compared by what it is, under **both**
    # gradings. The pair is what makes the probe see an `admissible_class` a forward
    # dropped, and the arguments are chosen so that neither half is a no-op on the
    # reference case -- `test_the_rebuilding_probes_are_not_no_ops` is what keeps both
    # properties from quietly lapsing.
    ("refine", _refine_pair_wrapper, _refine_pair_impl),
    ("refine_region", _refine_region_pair_wrapper, _refine_region_pair_impl),
    ("coarsen", _coarsen_pair_wrapper, _coarsen_pair_impl),
    ("__repr__", repr, repr),
)
"""Every member the binding registers, with how to read it through each side.

The list is spelled out rather than derived so that ``pytest`` reports one case per
member, which is what FELIGN/pantr#494's AC1 asks for;
:func:`test_the_forward_table_names_every_bound_member` is what keeps it from going
stale as the binding grows.
"""


def _read_truncated(space: Any) -> tuple[Any, ...]:
    """Read every truncation entry of a space into plain, comparable values.

    Args:
        space (Any): A :class:`~pantr.bspline.THBSplineSpace` or a raw handle.

    Returns:
        tuple[Any, ...]: One ``(dof, rep_level, box_lo, coeffs)`` per truncated function,
        in dof order, with the coefficients as ``float64`` arrays.
    """
    out: list[Any] = []
    for dof in range(int(space.num_total_basis)):
        entry = space.truncated(dof)
        if entry is not None:
            out.append(
                (
                    dof,
                    int(entry[0]),
                    tuple(int(lo) for lo in entry[1]),
                    np.ascontiguousarray(entry[2], dtype=np.float64),
                )
            )
    return tuple(out)


def _truncation_shape(entries: tuple[Any, ...]) -> tuple[Any, ...]:
    """The wholly integer part of a truncation: which dofs, at which level, over which box.

    What survives a cross-backend round trip as a verdict rather than as a displaced
    value, so it can be compared exactly whatever recomputed the coefficients.

    Args:
        entries (tuple[Any, ...]): What :func:`_read_truncated` returned.

    Returns:
        tuple[Any, ...]: One ``(dof, rep_level, box_lo, coeffs.shape)`` per entry.
    """
    return tuple((dof, rep, box, coeffs.shape) for dof, rep, box, coeffs in entries)


def _agree(left: Any, right: Any) -> bool:
    """Whether two reads of the same member agree.

    Held objects are compared by identity, arrays element by element, and everything else
    by equality. Not a tolerance: every value here is an index, a count, a flag, a domain
    end or a truncation coefficient that the wrapper only copies, so the wrapper and the
    thing it forwards to must agree exactly or the forward is doing arithmetic it has no
    business doing.

    Args:
        left (Any): The value read through the wrapper.
        right (Any): The value read through the implementation it holds.

    Returns:
        bool: Whether they agree.
    """
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _agree(one, other) for one, other in zip(left, right, strict=True)
        )
    if type(left).__module__ == "pantr._pantr_cpp" or type(right).__module__ == "pantr._pantr_cpp":
        return left is right
    return bool(left == right)


@pytest.mark.parametrize(
    ("member", "via_wrapper", "via_impl"), _FORWARDS, ids=[name for name, _, _ in _FORWARDS]
)
def test_the_wrapper_forwards_every_bound_member(
    cpp_backend: None,
    member: str,
    via_wrapper: Callable[[Any], Any],
    via_impl: Callable[[Any], Any],
) -> None:
    """Under the C++ backend, each bound member reaches the C++ type and agrees with it.

    FELIGN/pantr#494's AC1, one case per member. Reading the same quantity twice -- once
    through ``pantr.bspline.THBSplineSpace`` and once through the handle it holds -- is
    what makes this fail if the forward is removed: the public read raises
    :class:`AttributeError` instead, and a forward reimplemented on the wrapper rather
    than delegated would have to reproduce the handle's answer exactly to stay green.

    The three held members are compared by **handle identity**, not by value, which is
    ``design/bspline_ownership_lifetime.md`` F6 and is the one a value comparison could
    not see.

    Args:
        cpp_backend (None): Requires the compiled extension.
        member (str): The bound member's name, for the failure message.
        via_wrapper (Callable[[Any], Any]): Reads it through the public class.
        via_impl (Callable[[Any], Any]): Reads it through the handle.
    """
    space = _cpp_space(_reference_case())
    with use_backend(Backend.CPP):
        expected = via_impl(space._impl)
        actual = via_wrapper(space)
    assert _agree(actual, expected), (
        f"{member}: the wrapper and the C++ type it holds disagree.\n"
        f"  through the wrapper: {actual!r}\n  through the handle:  {expected!r}"
    )


def test_the_forward_table_names_every_bound_member(cpp_backend: None) -> None:
    """The per-member sweep above covers the binding's whole surface, not a snapshot of it.

    A hand-written list of members is a promise that rots the moment the binding gains
    one, and nothing else in the tree would notice: the new member would simply have no
    forward and no test. This reads the surface off the extension and compares.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    cpp = _bindings()
    covered = {name for name, _, _ in _FORWARDS}
    for handle_type in (cpp.THBSplineSpace64, cpp.THBSplineSpace32):
        bound = {name for name in dir(handle_type) if not name.startswith("_")} | {"__repr__"}
        assert covered == bound, (
            f"the forward table and {handle_type.__name__} disagree; "
            f"bound but untested {sorted(bound - covered)}, "
            f"tested but not bound {sorted(covered - bound)}"
        )
    assert covered <= set(dir(THBSplineSpace)), (
        f"bound members the public class does not expose: "
        f"{sorted(covered - set(dir(THBSplineSpace)))}"
    )


def test_the_rebuilding_probes_are_not_no_ops(cpp_backend: None) -> None:
    """The three rebuilding forwards are probed with operations that change the space.

    Without this the sweep above is vacuous for exactly three of its twenty-four cases,
    and silently: ``refine``, ``refine_region`` and ``coarsen`` return a space, so a
    forward replaced by ``return self`` still hands back something with a fingerprint,
    and if the probe's marked cells happen to refine or coarsen nothing then that
    fingerprint is the receiver's and the case passes. **Measured:** with the probes'
    earlier arguments, stubbing ``coarsen`` and ``refine_region`` to ``return self`` left
    every test in this file green.

    One assertion per member carries both properties, because the stronger one implies the
    weaker and a check that cannot fail on its own is not worth writing. The two gradings
    must differ: that catches a forward which drops ``admissible_class``, and it also
    catches two no-ops, since two calls that both change nothing return the receiver's
    fingerprint twice. What it deliberately does **not** demand is that the *graded* call
    change the space -- a graded call is entitled to be a no-op, and ``coarsen``'s
    admissibility guard vetoes the reference case's whole level, which is the behaviour
    under test rather than a badly chosen probe.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    space = _cpp_space(_reference_case())
    with use_backend(Backend.CPP):
        base = _fingerprint(space)
        for member, probe in (
            ("refine", _refine_pair_wrapper),
            ("refine_region", _refine_region_pair_wrapper),
            ("coarsen", _coarsen_pair_wrapper),
        ):
            graded, ungraded = probe(space)
            assert graded != ungraded, (
                f"{member}: the probe's two gradings agree ({graded} against the "
                f"receiver's {base}), so its forwarding case would pass both on a "
                f"forward that dropped admissible_class and, if they also equal the "
                f"receiver's, on one that did nothing at all"
            )
            assert graded != ungraded, (
                f"{member}: the probe's two gradings agree, so its forwarding case would "
                f"pass on a forward that dropped admissible_class"
            )


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        pytest.param(Backend.PYTHON, "_THBSplineSpacePython", id="python"),
        pytest.param(Backend.CPP, "THBSplineSpace64", id="cpp"),
    ],
)
def test_construction_selects_the_backends_own_implementation(
    cpp_backend: None, backend: Backend, expected: str
) -> None:
    """Constructing under each backend builds that backend's type, asserted on ``_impl``.

    FELIGN/pantr#494's AC1b. Construction is counted apart from the forwards because
    there is no forward to remove: what would fail is the selection itself, and the only
    place it is visible is the type of the implementation the wrapper ends up holding.

    The ``float32`` root is checked alongside, because the class is chosen per dtype as
    well as per backend and a selection that ignored the dtype would still pass the
    ``float64`` half.

    Args:
        cpp_backend (None): Requires the compiled extension.
        backend (Backend): The backend to build under.
        expected (str): The implementation class name it must select.
    """
    space = _space(_reference_case(), backend)
    assert type(space._impl).__name__ == expected

    narrow = _space(_reference_case()._replace(dtype=np.float32), backend)
    narrow_expected = "_THBSplineSpacePython" if backend is Backend.PYTHON else "THBSplineSpace32"
    assert type(narrow._impl).__name__ == narrow_expected


@pytest.mark.parametrize("truncate", [True, False], ids=["thb", "hb"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["f64", "f32"])
def test_a_thb_space_survives_pickling_across_every_backend_pair(
    cpp_backend: None, dtype: Any, truncate: bool
) -> None:
    """A pickle written under one backend loads under the other, at both dtypes.

    FELIGN/pantr#494's AC2, on the pattern ``tests/parity/test_bspline_type.py`` sets for
    :class:`~pantr.bspline.Bspline`. What this catches is a ``__reduce__`` that reaches
    the implementation rather than the constructor's arguments: a C++ handle is not
    picklable at all, and a payload carrying one would make ``PANTR_BACKEND`` a
    data-format switch -- a pickle written on one machine unreadable on another.
    :func:`copy.deepcopy` goes through ``__reduce_ex__`` by the same route, so it is
    swept over the same pairs.

    The reconstruction is compared on everything the space is rather than on its
    implementation: the counts, the active sets, the truncation and the root knots. The
    knots are the field that would catch a round trip which kept the counts while losing
    the geometry.

    Args:
        cpp_backend (None): Requires the compiled extension.
        dtype (Any): The root space's storage format.
        truncate (bool): Whether the truncated basis is built.
    """
    case = _reference_case()._replace(dtype=dtype, truncate=truncate)
    for writer in (Backend.PYTHON, Backend.CPP):
        original = _space(case, writer)
        payload = pickle.dumps(original)
        for reader in (Backend.PYTHON, Backend.CPP):
            where = f"{writer.name} -> {reader.name}"
            with use_backend(reader):
                loaded = pickle.loads(payload)
                cloned = copy.deepcopy(original)
            for rebuilt, how in ((loaded, "pickle"), (cloned, "deepcopy")):
                assert rebuilt.num_levels == original.num_levels, f"{where} {how}"
                assert rebuilt.num_total_basis == original.num_total_basis, f"{where} {how}"
                assert rebuilt.num_basis_per_level == original.num_basis_per_level, f"{where} {how}"
                assert rebuilt.truncate is original.truncate, f"{where} {how}"
                assert rebuilt.regularity == original.regularity, f"{where} {how}"
                assert rebuilt.num_truncated == original.num_truncated, f"{where} {how}"
                assert rebuilt.grid.num_cells == original.grid.num_cells, f"{where} {how}"
                for level in range(original.num_levels):
                    assert np.array_equal(
                        rebuilt.active_function_indices(level),
                        original.active_function_indices(level),
                    ), f"{where} {how}: level {level}"
                for k in range(original.dim):
                    np.testing.assert_array_equal(
                        np.asarray(rebuilt.root_space.spaces[k].knots),
                        np.asarray(original.root_space.spaces[k].knots),
                        err_msg=f"{where} {how}: direction {k}",
                    )
                rebuilt_entries = _read_truncated(rebuilt)
                original_entries = _read_truncated(original)
                assert _truncation_shape(rebuilt_entries) == _truncation_shape(original_entries), (
                    f"{where} {how}: the truncation did not survive the round trip"
                )
                # The coefficients are only compared bit for bit **within** one backend.
                # Across a pair the reconstruction recomputes them on the reader's side,
                # which is the very cross-backend situation `_COEFFICIENT_WHY` says needs
                # a derived bound rather than bit-identity -- and `_compare` is where
                # that comparison already happens, with the bound. Asserting equality
                # here would be a bit-identity criterion on a floating-point quantity
                # obtained by two summation orders, which is exactly what this file
                # refuses everywhere else; it passes today and would fail on a machine
                # whose BLAS sums differently.
                if writer is reader:
                    assert _agree(rebuilt_entries, original_entries), (
                        f"{where} {how}: the truncation coefficients moved within one "
                        f"backend, where nothing recomputes them differently"
                    )


def test_a_nested_object_from_the_other_backend_is_refused(cpp_backend: None) -> None:
    """Building over a root space or a grid from the other backend raises, both ways.

    The refusal exists because the two implementations take their nested objects
    differently: the C++ class takes handles, and handing it a Python oracle raises a
    nanobind ``TypeError`` naming C++ types -- loud, if unreadable. The oracle takes the
    *wrappers* and would **succeed**, yielding a space whose hierarchical logic runs in
    Python over C++ values, which no parity claim covers and nothing announces.
    ``design/cross_backend_types.md`` forbids that second shape, which is why the check
    is in ``_new_impl`` rather than left to whichever constructor happens to complain.

    Both directions are exercised, and the silent one is the reason for the test.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    case = _reference_case()
    built: dict[Backend, tuple[BsplineSpace, Any]] = {}
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            directions = [
                BsplineSpace1D(
                    _open_knots(case.degrees[k], case.num_elements[k], *case.bounds[k]),
                    case.degrees[k],
                )
                for k in range(len(case.degrees))
            ]
            grid = hierarchical_grid(
                uniform_grid([list(b) for b in case.bounds], list(case.num_elements)),
                list(case.factor),
            )
            built[backend] = (BsplineSpace(directions), grid)

    for backend, other in ((Backend.PYTHON, Backend.CPP), (Backend.CPP, Backend.PYTHON)):
        own_root, own_grid = built[backend]
        foreign_root, foreign_grid = built[other]
        with use_backend(backend):
            with pytest.raises(ValueError, match="root_space must come from the active backend"):
                THBSplineSpace(foreign_root, own_grid)
            with pytest.raises(ValueError, match="grid must come from the active backend"):
                THBSplineSpace(own_root, foreign_grid)


@pytest.mark.parametrize("backend", [Backend.PYTHON, Backend.CPP], ids=["python", "cpp"])
def test_the_borrowed_arrays_are_read_only_and_the_copied_ones_are_not(
    cpp_backend: None, backend: Backend
) -> None:
    """Each array member is writable exactly where its own docstring says it is.

    Two different failures, and neither is visible in a value comparison. A member
    documented as a fresh copy that hands out a view lets a caller corrupt storage the
    C++ space owns, for its whole life and for every other reader of it. A member
    documented read-only that hands out a writable array is the same defect one step
    later, since the wrapper would then be the only thing standing between a caller and
    that storage.

    ``design/bspline_ownership_lifetime.md`` records this class of regression as silent,
    which is why it is asserted rather than left to the read of the binding.

    Args:
        cpp_backend (None): Requires the compiled extension.
        backend (Backend): The backend the space is built under.
    """
    space = _space(_reference_case(), backend)
    truncated = [
        entry
        for entry in (space.truncated(dof) for dof in range(space.num_total_basis))
        if entry is not None
    ]
    assert truncated, "this case truncated nothing, so the read-only half is untested"

    for name, block in (
        ("level_offsets", space.level_offsets),
        ("domain", space.domain),
        ("active_function_indices", space.active_function_indices(0)),
        ("active_basis", space.active_basis(0)),
    ):
        assert block.flags.writeable, f"{name} is documented as a fresh copy but is read-only"

    for name, block in (
        ("contributions.dofs", space.contributions(0)[0]),
        ("contributions.levels", space.contributions(0)[1]),
        ("contributions.multi_indices", space.contributions(0)[2]),
        ("truncated.coeffs", truncated[0][2]),
    ):
        assert not block.flags.writeable, (
            f"{name} is documented read-only but a caller could write through it"
        )


@pytest.mark.parametrize(
    ("regularity", "message"),
    [
        pytest.param(
            [None, None, None],
            "regularity must be a scalar or length-2 sequence; got length 3.",
            id="regularity-length",
        ),
        pytest.param(
            [2, None],
            "regularity[0]=2 must be in [-1, degree[0]-1=1]; got 2.",
            id="regularity-range",
        ),
    ],
)
def test_the_construction_refusals_carry_one_message(
    cpp_backend: None, regularity: list[int | None], message: str
) -> None:
    """Both backends refuse the same construction, with the same wording.

    Character for character, as every other refusal in this port is, so that a caller
    matching on the message keeps working when the backend changes underneath it. Since
    FELIGN/pantr#494 the refusal is reached through the public class, which is where a
    caller meets it: the wrapper keeps only the two ``isinstance`` checks and the scalar
    broadcast, and hands everything else to the implementation, so the message a caller
    sees is the implementation's under both backends.

    The literal is pinned as well as compared, so the test fails if the two backends
    agree on a *changed* message rather than only if they disagree.

    Args:
        cpp_backend (None): Requires the compiled extension.
        regularity (list[int | None]): The refused per-direction continuity.
        message (str): The exact text expected.
    """
    case = _reference_case()
    seen: dict[str, str] = {}
    for backend in (Backend.PYTHON, Backend.CPP):
        with use_backend(backend):
            directions = [
                BsplineSpace1D(_open_knots(2, 4, 0.0, 1.0), 2) for _ in range(len(case.degrees))
            ]
            grid = hierarchical_grid(
                uniform_grid([list(b) for b in case.bounds], list(case.num_elements)),
                list(case.factor),
            )
            with pytest.raises(ValueError) as caught:
                THBSplineSpace(BsplineSpace(directions), grid, regularity=regularity)
        seen[backend.name] = str(caught.value)
    assert seen["PYTHON"] == message, f"the oracle's wording moved: {seen['PYTHON']!r}"
    assert seen["CPP"] == message, f"the port's wording moved: {seen['CPP']!r}"


def test_an_out_of_range_cell_id_is_refused_the_oracles_way(cpp_backend: None) -> None:
    """Both backends name **every** offending id, in the same kind and the same words.

    The kind matters as much as the wording: the oracle raises ``IndexError`` and nanobind
    maps ``std::out_of_range`` to that, so a caller catching one keeps working. What had
    to be brought over deliberately is the *list*: a first version of the C++ threw on the
    first bad id it met, which is a real loss for a caller debugging several.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    case = _reference_case()
    py = _python_space(case)
    cpp = _cpp_space(case)
    past_the_end = py.grid.num_cells
    bad = np.array([past_the_end + 5, -1, past_the_end], dtype=np.int64)
    # Sorted and deduplicated, since both implementations report the offending ids after
    # `numpy.unique`'s ordering rather than in the order they were handed.
    expected_message = (
        f"cell_ids must lie in [0, {past_the_end}); got out-of-range id(s): "
        f"[-1, {past_the_end}, {past_the_end + 5}]."
    )

    for name in ("refine", "coarsen"):
        with _the_oracle(), pytest.raises(IndexError) as expected:
            getattr(py, name)(bad, admissible_class=2)
        with use_backend(Backend.CPP), pytest.raises(IndexError) as actual:
            getattr(cpp, name)(bad, admissible_class=2)
        assert str(actual.value) == str(expected.value), (
            f"{name} refuses out-of-range ids with a different message under the two backends"
        )
        assert str(expected.value) == expected_message, (
            f"{name}'s wording moved under both backends at once: {expected.value}"
        )
