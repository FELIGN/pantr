"""Parity of the C++ ``THBSplineSpace`` against the Python oracle it was ported from.

Covers the *construction and query* half of FELIGN/pantr#397: the per-level spaces, the
Kraft selection, the truncation coefficients, the per-cell contribution table, and
refinement and coarsening. Basis tabulation, the windowed restriction and the
prolongation operators are not ported yet and are not compared here.

The C++ side is reached through :mod:`pantr._pantr_cpp` directly rather than through
``pantr.bspline.THBSplineSpace``, because that class is still pure Python: this cut lands
the C++ type and its binding, and the wrapper that dispatches to it is the next one. So
there is no ``__reduce__`` round trip in this file either -- nothing here is picklable
yet, and the round trip is owed by the cut that introduces the wrapper.

What agrees, quantity by quantity
---------------------------------

The split is per quantity with an argument for each, and it is not one decision applied in
bulk.

**Exactly** -- ``num_levels``, ``num_total_basis``, ``num_basis_per_level``, the per-level
active function index sets, ``level_offsets``, ``dof_level``, the per-cell ``active_basis``
lists, the contribution table's levels and multi-indices, ``max_active_per_cell``, the set
of truncated dofs, and each truncated function's representation level, box origin and box
shape. Every one is an index, a count or a set membership, so no rounding takes place and
bit-identity is the only criterion that says anything. A bounded comparison could not even
see two answers of different length.

**Within a derived bound** -- the truncation coefficients, and *only* those. They are
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
import math
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


def _python_space(case: _Case) -> THBSplineSpace:
    """Build the oracle's space for a case.

    Args:
        case (_Case): The hierarchy to build.

    Returns:
        THBSplineSpace: The space, built under the Python backend.
    """
    with _the_oracle():
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


def _cpp_space(case: _Case) -> Any:
    """Build the C++ handle for a case.

    Args:
        case (_Case): The hierarchy to build.

    Returns:
        Any: A ``THBSplineSpace32`` or ``THBSplineSpace64`` handle.
    """
    cpp = _bindings()
    narrow = np.dtype(case.dtype) == np.float32
    one_d = cpp.BsplineSpace1D32 if narrow else cpp.BsplineSpace1D64
    tensor = cpp.BsplineSpace32 if narrow else cpp.BsplineSpace64
    hierarchical = cpp.THBSplineSpace32 if narrow else cpp.THBSplineSpace64

    directions = [
        one_d(
            np.ascontiguousarray(
                _open_knots(case.degrees[k], case.num_elements[k], *case.bounds[k]),
                dtype=case.dtype,
            ),
            case.degrees[k],
            False,
            True,
        )
        for k in range(len(case.degrees))
    ]
    breakpoints = [
        np.ascontiguousarray(np.linspace(*case.bounds[k], case.num_elements[k] + 1))
        for k in range(len(case.degrees))
    ]
    grid = cpp.HierarchicalGrid(
        cpp.TensorProductGrid(breakpoints), np.ascontiguousarray(case.factor, dtype=np.int64)
    )
    for level, lo, hi in case.refinements:
        grid = grid.refine(
            level,
            np.ascontiguousarray(lo, dtype=np.int64),
            np.ascontiguousarray(hi, dtype=np.int64),
        )
    return hierarchical(tensor(directions), grid, case.truncate, list(case.regularity))


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


def _truncated_dofs(space: Any, num_total_basis: int) -> dict[int, Any]:
    """The truncation entries of either backend, keyed by global dof.

    Args:
        space (Any): The oracle's space or a C++ handle.
        num_total_basis (int): How many dofs to ask about.

    Returns:
        dict[int, Any]: ``dof -> (rep_level, box_lo, coeffs)`` for every truncated
        function. The oracle's ``_TruncCoeffs`` and the handle's tuple unpack the same
        way, which is why one reader serves both.
    """
    if isinstance(space, THBSplineSpace):
        return {int(dof): entry for dof, entry in space._trunc.items()}
    entries: dict[int, Any] = {}
    for dof in range(num_total_basis):
        entry = space.truncated(dof)
        if entry is not None:
            entries[dof] = entry
    return entries


def _partition_of_unity_defect(space: Any, oracle: THBSplineSpace) -> float:
    """How far the active basis is from summing to one, in the finest level's basis.

    Pushes every active function's coefficient vector to the finest level by pure
    two-scale refinement -- no truncation -- and returns
    ``max |sum_i c_i - 1|`` over the finest tensor-product basis.

    The two-scale kernel is the oracle's, used as shared machinery: what is under test is
    ``space``'s *selection and truncation*, and the kernel is independently pinned in
    ``cpp/tests/test_bspline_knot_insertion.cpp`` against the cardinal B-spline's binomial
    stencil. The module docstring says why that composition is honest.

    Args:
        space (Any): The space whose coefficients are checked; the oracle or a handle.
        oracle (THBSplineSpace): The oracle for the same hierarchy, which supplies the
            level spaces the two-scale matrices are built from. Passing it explicitly
            rather than reading it off ``space`` is what lets the same function grade a
            C++ handle.

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
            level = oracle._dof_level(dof)
            position = dof - int(oracle._func_offset[level])
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
        Field("dim", exact_parity(why=_VERDICT_WHY)),
        Field("degrees", exact_parity(why=_VERDICT_WHY)),
        Field("truncate", exact_parity(why=_VERDICT_WHY)),
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

    assert py.max_active_per_cell() == cpp.max_active_per_cell(), (
        f"{context}: max_active_per_cell disagrees. {_VERDICT_WHY}"
    )
    for cid in range(py.grid.num_cells):
        expected = np.asarray(py.active_basis(cid), dtype=np.int64)
        actual = np.asarray(cpp.active_basis(cid), dtype=np.int64)
        assert np.array_equal(expected, actual), (
            f"{context}: active_basis({cid}) disagrees. {_VERDICT_WHY}"
        )
        dofs, levels, multis = cpp.contributions(cid)
        contributions = py._cell_contributions(cid)
        assert [int(d) for d in dofs] == [t[0] for t in contributions]
        assert [int(x) for x in levels] == [t[1] for t in contributions]
        assert [tuple(int(v) for v in row) for row in np.asarray(multis)] == [
            tuple(t[2]) for t in contributions
        ], f"{context}: the contribution multi-indices of cell {cid} disagree"

    for dof in range(py.num_total_basis):
        assert py._dof_level(dof) == cpp.dof_level(dof), f"{context}: dof_level({dof}) disagrees"

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
        expected = np.asarray(py_entry.coeffs, dtype=np.float64)
        actual = np.ascontiguousarray(cpp_entry[2], dtype=np.float64)
        where = f"{context}: dof {dof}"
        assert int(py_entry.rep_level) == int(cpp_entry[0]), f"{where}: rep_level disagrees"
        assert tuple(py_entry.box_lo) == tuple(int(x) for x in cpp_entry[1]), (
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


def _compare_the_operations(py: THBSplineSpace, cpp: Any, context: str, variant: int) -> None:
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
        cpp (Any): The C++ handle for the same hierarchy.
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
    actual = cpp.coarsen(marked, admissible) if coarsening else cpp.refine(marked, admissible)

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
        (max(e.coeffs.shape) for e in oracle._trunc.values()),
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


def test_level_space_zero_shares_the_root_handle(cpp_backend: None) -> None:
    """``level_space(0)`` hands back the object the space was built from, not a copy.

    ``design/bspline_ownership_lifetime.md`` F6 makes this an identity contract rather
    than a convenience: it is what the oracle's ``thb.level_space(0) is thb.root_space``
    asserts, and reproducing it is what fixes the C++ constructor's signature to take
    handles rather than values.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    case = _reference_case()
    oracle = _python_space(case)
    assert oracle.level_space(0) is oracle.root_space

    cpp = _cpp_space(case)
    assert cpp.level_space(0) is cpp.root_space
    # And the finer levels are genuinely different objects, or the assertion above would
    # hold for a type that returned one space for every level.
    assert cpp.level_space(1) is not cpp.root_space


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
        actual = cpp.refine(marked, admissible)
        assert expected.num_levels == actual.num_levels
        assert expected.num_total_basis == actual.num_total_basis
        assert tuple(expected.num_basis_per_level) == tuple(actual.num_basis_per_level)
        for level in range(expected.num_levels):
            assert np.array_equal(
                expected.active_function_indices(level),
                np.asarray(actual.active_function_indices(level)),
            )

    lo = np.array([0, 0], dtype=np.int64)
    hi = np.array([2, 2], dtype=np.int64)
    with _the_oracle():
        expected = oracle.refine_region(0, [0, 0], [2, 2], admissible_class=2)
    actual = cpp.refine_region(0, lo, hi, 2)
    assert expected.num_total_basis == actual.num_total_basis

    finest = np.array(
        [c for c in range(oracle.grid.num_cells) if oracle.grid.cell_level(c) == 2],
        dtype=np.int64,
    )
    with _the_oracle():
        expected = oracle.coarsen(finest, admissible_class=None)
    actual = cpp.coarsen(finest, None)
    assert expected.num_levels == actual.num_levels
    assert expected.num_total_basis == actual.num_total_basis
    for level in range(expected.num_levels):
        assert np.array_equal(
            expected.active_function_indices(level),
            np.asarray(actual.active_function_indices(level)),
        )


def test_refine_and_coarsen_never_hand_back_the_receivers_grid(cpp_backend: None) -> None:
    """A no-op refinement still returns a space over a grid of its own.

    Two spaces sharing one grid would make a tag set through one visible through the
    other, which is what the oracle's ``grid._copy()`` prevents and what
    ``HierarchicalGrid::refine_cells`` with no ids does here.

    Args:
        cpp_backend (None): Requires the compiled extension.
    """
    cpp = _cpp_space(_reference_case())
    nothing = np.array([], dtype=np.int64)
    assert cpp.refine(nothing, 2).grid is not cpp.grid
    assert cpp.coarsen(nothing, 2).grid is not cpp.grid


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(
            lambda cpp, root, grid: cpp.THBSplineSpace64(root, grid, True, [None, None, None]),
            "regularity must be a scalar or length-2 sequence; got length 3.",
            id="regularity-length",
        ),
        pytest.param(
            lambda cpp, root, grid: cpp.THBSplineSpace64(root, grid, True, [2, None]),
            "regularity[0]=2 must be in [-1, degree[0]-1=1]; got 2.",
            id="regularity-range",
        ),
    ],
)
def test_the_refusals_carry_the_oracles_message(
    cpp_backend: None, build: Any, message: str
) -> None:
    """The C++ type refuses what the oracle refuses, with the oracle's wording.

    Character for character, as every other refusal in this port is, so that a caller
    matching on the message keeps working when the backend changes underneath it.

    Args:
        cpp_backend (None): Requires the compiled extension.
        build (Any): Builds the refused space from the module, root space and grid.
        message (str): The exact text expected.
    """
    cpp = _bindings()
    case = _reference_case()
    directions = [
        cpp.BsplineSpace1D64(np.ascontiguousarray(_open_knots(2, 4, 0.0, 1.0)), 2, False, True)
        for _ in range(2)
    ]
    root = cpp.BsplineSpace64(directions)
    breakpoints = [np.ascontiguousarray(np.linspace(0.0, 1.0, 5)) for _ in range(2)]
    grid = cpp.HierarchicalGrid(
        cpp.TensorProductGrid(breakpoints), np.ascontiguousarray(case.factor, dtype=np.int64)
    )
    with pytest.raises(ValueError, match=r".*") as caught:
        build(cpp, root, grid)
    assert str(caught.value) == message


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

    for name in ("refine", "coarsen"):
        with _the_oracle(), pytest.raises(IndexError) as expected:
            getattr(py, name)(bad, admissible_class=2)
        with pytest.raises(IndexError) as actual:
            getattr(cpp, name)(bad, 2)
        assert str(actual.value) == str(expected.value), (
            f"{name} refuses out-of-range ids with a different message under the two backends"
        )
