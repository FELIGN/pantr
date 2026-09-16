"""Parity, accuracy and mutation coverage for :class:`SpanwiseElementExtraction`.

``SpanwiseElementExtraction`` is a wrapper over one of two implementations chosen by
``pantr.bspline.spanwise_element_extraction._impl_class``: the oracle
``_SpanwiseElementExtractionPython`` or ``pantr._pantr_cpp.SpanwiseElementExtraction32``
/ ``...64``. Both take the per-direction *dense* operators and identity masks already
built -- by the existing, separately-ported Bézier and Lagrange builders, or by the
unported cardinal one -- and do only the **compaction**: keep the non-identity rows,
build the index map, keep the mask, and lazily decompress ``ops_1d`` again.

Four questions are asked here, in order, each with its own oracle
(``ST-oracle-from-a-different-route``):

1. **Is the compaction itself exact?** (headline claim). Both implementations are
   built from *one* set of operator arrays constructed directly in this file, so the
   comparison isolates what this type added from what the builders do. The oracle
   route is the C++ implementation itself compared against the Python oracle, given
   bit-identical input -- legitimate here because the claim is about a pure
   selection-and-copy operation that performs no arithmetic, so "the same bits in"
   forces "the same bits out" on both sides regardless of implementation language.
2. **Does the public constructor still agree end to end**, now that the operators
   come from the real dispatched builders? For ``BEZIER`` and ``LAGRANGE`` the claim
   is the builder's own, imported unchanged from
   ``tests/parity/test_bspline_bezier_extraction.py`` and
   ``tests/parity/test_bspline_lagrange_extraction.py`` respectively -- inventing a
   fresh one here would duplicate a derivation this file has no business owning.

   For ``CARDINAL`` there is no C++ *extraction* builder, but that is not the same
   as "both backends run the same computation" -- an assumption this file's first
   draft made and measurement disproved. ``A_e = C_e @ card_to_bzr``, exactly the
   Lagrange target's own shape, and ``card_to_bzr`` (the cardinal-to-Bernstein
   change of basis) **is** separately dispatched by :mod:`pantr.change_basis`,
   which does not agree bit for bit between backends (measured: a gap of ``1.5e-14``
   at degree 3, float64; ``2.6e-05`` at float32). So the claim is bounded, derived
   the same way the Lagrange one is: a contraction-summation term plus an absolute
   term for the matrix's own gap. See ``_cardinal_direction_claims``.
3. **Is either backend's answer independently right?** Two oracles that share no
   code with the port:

   - **Partition of unity.** For target basis ``T`` and B-spline basis ``N`` on one
     element, ``N_i = sum_j C_ij T_j``. Both bases sum to one on the element, so
     ``1 = sum_i N_i = sum_j (sum_i C_ij) T_j = sum_j T_j``, and ``T``'s linear
     independence forces ``sum_i C_ij = 1`` for every column ``j`` -- a check on
     *columns*, which is what lets it catch a transposed operator. ``design/
     extraction_port.md``'s "The independent accuracy check" states this for
     Bézier; it generalises to Lagrange and cardinal by naming the hypothesis it
     actually needs, that the *target* basis is a partition of unity, which holds
     for Bernstein, Lagrange and cardinal B-splines alike.
   - **Exact integer arithmetic.** With small-integer operator entries and an
     integer cell, ``kron(M_0, ..., M_{d-1})`` is exactly representable in
     binary64, so ``SpanwiseElementExtraction.operator(cell)`` must equal a
     Python-integer :func:`numpy.kron` computation exactly. Independent of
     anything B-spline, and the check that would catch a transposed direction or a
     swapped mode order -- an error class both backends would share, and which the
     partition-of-unity check above cannot see because a transposition does not
     change a column sum.

4. **Would any of the above actually fail on a wrong answer?** Every accuracy check
   above gets a negative control: the correct data is perturbed and the same
   assertion is shown to raise. The headline compaction claim is mutation-checked
   too, by monkeypatching the *bound* C++ properties in a throwaway scratch script
   (never in this file, since ``ST-tests-only`` forbids touching anything else) to
   drop the sentinel row or transpose a compact block, and confirming the
   comparison in this file would have caught it. That evidence is in the dispatch
   report, not reproducible here without mutating shared global state.

Tolerances, by target
----------------------

- **BEZIER**: :func:`tests.parity.test_bspline_bezier_extraction._column_sum_bound`,
  reused unchanged. It already bounds this exact deviation for these exact
  operators.
- **LAGRANGE**: :func:`tests.parity.test_bspline_lagrange_extraction._column_sum_tolerance`,
  reused unchanged, compared against the change-of-basis matrix's own column sum
  rather than against one (``L`` is not itself exactly column-stochastic in
  floating point).
- **CARDINAL**: there is no derived bound anywhere in the tree, and none is invented
  here. Two terms, kept visibly separate:

  (a) the **summation** error of adding ``n_out`` signed entries,
      ``gamma_{n_out - 1} * sum_i |C[e, i, j]|`` -- Higham, *Accuracy and Stability
      of Numerical Algorithms*, 2nd ed. (SIAM, doi:10.1137/1.9780898718027), eq.
      (4.4), with the absolute-value companion ``design/backend_parity.md`` Rule 10
      prescribes for a non-convex operator (the cardinal-to-Bernstein matrix has
      negative entries; Bézier's and Lagrange's do not).
  (b) an **admitted heuristic** multiple of (a), covering the error already baked
      into the cardinal builder's own entries, which is not derived anywhere and
      which measurement shows dominates (a) by a wide margin.

  ``_CARDINAL_HEURISTIC_MULTIPLE``'s docstring carries how the multiple was chosen:
  it is **observed, not proved**, and it is admitted as a heuristic rather than
  presented as a derivation, per the doctrine's explicit allowance for one. The
  cardinal cases in this file stop at degree 3, where the multiple was calibrated;
  a wider degree range is out of scope and not claimed.

Rule 12 (interpreted-oracle gate)
----------------------------------

Nothing added by this port touches ``np.power`` or an integer accumulator: the
compaction is index arithmetic and copies, and the Kronecker assembly is
:func:`numpy.kron`. So no test here needs ``demand_a_compiled_seed`` or
``demand_the_compiled_kernel`` -- unlike the Bézier and Lagrange *builder* claims
this file imports, which already carry their own gates where needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, NamedTuple, cast

import numpy as np
import pytest

from pantr._backend import Backend, use_backend
from pantr.basis import LagrangeVariant
from pantr.bspline import BsplineSpace, BsplineSpace1D, ExtractionTarget, SpanwiseElementExtraction
from pantr.bspline.spanwise_element_extraction import _SpanwiseElementExtractionPython
from pantr.change_basis import _cached_cardinal_to_bernstein_matrix
from tests._parity_harness import (
    Field,
    Roundings,
    assert_accuracy,
    assert_object_parity,
    bitwise_parity,
    bounded_parity,
    derived_accuracy,
    exact_parity,
    unit_roundoff,
)
from tests.parity.test_bspline_bezier_extraction import _CASES as _BEZIER_CASES
from tests.parity.test_bspline_bezier_extraction import (
    _Case,
    _column_sum_bound,
)
from tests.parity.test_bspline_bezier_extraction import (
    _claim as _bezier_claim,
)
from tests.parity.test_bspline_bezier_extraction import (
    _draw as _draw_bezier_case,
)
from tests.parity.test_bspline_lagrange_extraction import _CASES as _LAGRANGE_CASES
from tests.parity.test_bspline_lagrange_extraction import (
    _bezier as _lagrange_bezier_ops,
)
from tests.parity.test_bspline_lagrange_extraction import (
    _column_sum_tolerance,
    _matrix_under,
)
from tests.parity.test_bspline_lagrange_extraction import (
    _companion as _lagrange_companion,
)
from tests.parity.test_bspline_lagrange_extraction import (
    _product_claim as _lagrange_product_claim,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy import typing as npt

pytestmark = pytest.mark.usefixtures("cpp_backend")
"""Every test here needs the compiled extension; see the module docstring point 4.

A missing extension SKIPS the whole file (or FAILS it under ``PANTR_REQUIRE_CPP``),
via :func:`tests._parity_harness.demand_cpp_backend`. That also means the
Python-only accuracy checks do not run without the extension, which is a coarser
gate than ``tests/parity/test_bspline_bezier_extraction.py`` chose -- see the
Escalations in the dispatch report for why that trade-off was made here.
"""

DTYPES: Final = (np.float64, np.float32)
"""The two storage formats every implementation is instantiated for."""

_TARGETS: Final = (ExtractionTarget.BEZIER, ExtractionTarget.LAGRANGE, ExtractionTarget.CARDINAL)
"""Every element-local basis the type supports."""

_BACKENDS: Final = (Backend.PYTHON, Backend.CPP)
"""The two backends, for the tests that state a property of each one separately."""


def _cpp_module() -> Any:
    """Import the extension, deferred so a missing one only skips this module's tests.

    Module level would break the point of gating on ``cpp_backend`` rather than
    failing at collection: a top-level ``from pantr import _pantr_cpp`` raises before
    any fixture runs. Same shape as the Bézier and Lagrange parity files' own
    ``_bindings()``.

    Returns:
        Any: The :mod:`pantr._pantr_cpp` module.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415  (deferred on purpose, see above)

    return _pantr_cpp


def _cpp_impl_class(dtype: npt.DTypeLike) -> Any:
    """The compiled ``SpanwiseElementExtraction`` class for one storage format.

    Args:
        dtype (npt.DTypeLike): ``np.float32`` or ``np.float64``.

    Returns:
        Any: ``SpanwiseElementExtraction32`` or ``SpanwiseElementExtraction64``.
    """
    module = _cpp_module()
    if np.dtype(dtype) == np.float32:
        return module.SpanwiseElementExtraction32
    return module.SpanwiseElementExtraction64


# ---------------------------------------------------------------------------
# Section 1 -- the type's own contribution: the compaction is exact
# ---------------------------------------------------------------------------


class _SynDirection(NamedTuple):
    """One direction of a synthetic compaction case: a placeholder space plus operators.

    The knot vector and degree exist only to make ``BsplineSpace1D`` produce the
    wanted element count; the compaction under test never reads an operator's
    numeric content or a space's degree, only ``num_intervals``. Keeping them
    distinct across directions (different degree, element count and identity
    pattern) is what makes a per-direction bug -- an off-by-one in which
    direction's mask is read, say -- visible instead of accidentally symmetric.

    Attributes:
        knots (tuple[float, ...]): The knot vector fixing this direction's element
            count.
        degree (int): The polynomial degree of the placeholder space.
        size (int): ``n_out == n_in`` for this direction's synthetic operator.
        identity (tuple[bool, ...]): One flag per element; its length is this
            direction's element count.
    """

    knots: tuple[float, ...]
    degree: int
    size: int
    identity: tuple[bool, ...]


_MIXED_DIR: Final = _SynDirection((0.0, 0.0, 1.0, 2.0, 3.0, 3.0), 1, 2, (False, True, False))
"""3 elements, one identity in the middle -- exercises a non-trivial index map."""

_NON_IDENTITY_DIR: Final = _SynDirection((0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0), 2, 3, (False, False))
"""2 elements, none identity -- the fully-non-identity case the brief asks for."""

_ALL_IDENTITY_DIR: Final = _SynDirection((0.0, 1.0, 2.0, 3.0, 4.0), 0, 1, (True, True, True, True))
"""4 elements, all identity -- exercises the sentinel row (n_compact == 1, n_elements == 4)."""


def _synthetic_operators(
    directions: Sequence[_SynDirection], dtype: npt.DTypeLike
) -> tuple[list[npt.NDArray[Any]], list[npt.NDArray[np.bool_]]]:
    """Build one dense operator block and identity mask per direction.

    Every non-identity entry is a small integer, exactly representable in both
    ``float32`` and ``float64``, so both storage formats see bit-identical input and
    a dtype-dependent divergence cannot masquerade as a compaction bug. Identity
    elements are filled with the exact ``eye`` matrix, though the type discards
    whatever value an identity row carries at construction -- what decompression
    hands back for an identity element comes from the mask, never from this array.
    The returned dense block is therefore already what a correct ``ops_1d`` must
    equal, which :func:`test_the_exact_integer_kron_oracle` uses directly.

    Args:
        directions (Sequence[_SynDirection]): One entry per direction.
        dtype (npt.DTypeLike): Storage format.

    Returns:
        tuple[list[npt.NDArray[Any]], list[npt.NDArray[np.bool_]]]: ``(operators,
        masks)``, one dense ``(n_elements, size, size)`` array and one
        ``(n_elements,)`` mask per direction.
    """
    operators: list[npt.NDArray[Any]] = []
    masks: list[npt.NDArray[np.bool_]] = []
    counter = 1.0
    for direction in directions:
        n = len(direction.identity)
        size = direction.size
        mask = np.asarray(direction.identity, dtype=np.bool_)
        block = np.empty((n, size, size), dtype=dtype)
        eye = np.eye(size, dtype=dtype)
        for e in range(n):
            if mask[e]:
                block[e] = eye
            else:
                entries = counter + np.arange(size * size, dtype=np.float64).reshape(size, size)
                block[e] = entries.astype(dtype)
                counter += size * size
        block.flags.writeable = False
        operators.append(block)
        masks.append(mask)
    return operators, masks


def _build_space_pair(
    directions: Sequence[_SynDirection], dtype: npt.DTypeLike
) -> tuple[BsplineSpace, BsplineSpace]:
    """Build the same tensor-product space, once per backend.

    Args:
        directions (Sequence[_SynDirection]): One entry per direction.
        dtype (npt.DTypeLike): Storage format.

    Returns:
        tuple[BsplineSpace, BsplineSpace]: ``(py_space, cpp_space)``, geometrically
        identical spaces built under the two backends.
    """

    def _one(backend: Backend) -> BsplineSpace:
        with use_backend(backend):
            return BsplineSpace(
                [
                    BsplineSpace1D(np.asarray(d.knots, dtype=dtype), d.degree, snap_knots=False)
                    for d in directions
                ]
            )

    return _one(Backend.PYTHON), _one(Backend.CPP)


class _CompactionCase(NamedTuple):
    """One dimension's worth of synthetic directions.

    Attributes:
        label (str): What the case exercises.
        directions (tuple[_SynDirection, ...]): The per-direction specs, in order.
    """

    label: str
    directions: tuple[_SynDirection, ...]


_COMPACTION_CASES: Final = (
    _CompactionCase("dim 1", (_MIXED_DIR,)),
    _CompactionCase("dim 2", (_MIXED_DIR, _NON_IDENTITY_DIR)),
    _CompactionCase("dim 3", (_MIXED_DIR, _NON_IDENTITY_DIR, _ALL_IDENTITY_DIR)),
)
"""Dimensions 1 through 3, asymmetric throughout, covering the all-identity and the
fully-non-identity direction across the sweep."""


def _indexed_reader(attr: str, d: int) -> Callable[[Any], Any]:
    """Build a ``Field.read`` callable for one per-direction tuple attribute.

    A ``def`` rather than a lambda capturing ``d`` by default argument: mypy
    (correctly) cannot infer a lambda's parameter type from how it is later called
    through :class:`~tests._parity_harness.Field`'s ``Callable[[Any], Any]`` slot,
    and every field here needs the same shape, so one factory serves all of them.

    Args:
        attr (str): The tuple attribute name (``"compact_ops_1d"``, ``"ops_1d"``,
            ``"idx_maps_1d"`` or ``"is_identity_mask_1d"``).
        d (int): The direction index.

    Returns:
        Callable[[Any], Any]: A function reading ``getattr(impl, attr)[d]``.
    """

    def _read(impl: Any) -> Any:
        return getattr(impl, attr)[d]

    return _read


def _state_fields(
    dim: int,
    compact_claims: Sequence[Any],
    dense_claims: Sequence[Any] | None = None,
) -> list[Field]:
    """Every field the two implementations of a ``SpanwiseElementExtraction`` must agree on.

    Works identically on a raw implementation object (oracle instance or C++
    handle) and on the public ``SpanwiseElementExtraction`` wrapper, since the
    wrapper exposes the same property names it reads off its own ``_impl``.

    ``compact_ops_1d`` and ``ops_1d`` need separate claim objects whenever a
    direction has some but not all identity elements: the compact array then has
    fewer rows than the dense one, so an elementwise amplification array sized for
    one does not broadcast against the other. Where every direction is either
    fully identity or fully non-identity the two shapes coincide and passing only
    ``compact_claims`` is enough -- turning a compact row into a dense one is a pure
    copy or an exact ``eye`` fill either way, adding no rounding beyond what built
    the compact block, so the same claim is valid for both when the shapes agree.

    Args:
        dim (int): Number of directions.
        compact_claims (Sequence[Any]): One parity claim per direction, for
            ``compact_ops_1d``.
        dense_claims (Sequence[Any] | None): One parity claim per direction, for
            ``ops_1d``. Defaults to ``compact_claims`` when the two are known to
            share a shape.

    Returns:
        list[Field]: The fields for :func:`tests._parity_harness.assert_object_parity`.
    """
    if dense_claims is None:
        dense_claims = compact_claims
    fields: list[Field] = []
    for d in range(dim):
        fields.append(
            Field(
                f"compact_ops_1d[{d}]",
                compact_claims[d],
                read=_indexed_reader("compact_ops_1d", d),
            )
        )
        fields.append(Field(f"ops_1d[{d}]", dense_claims[d], read=_indexed_reader("ops_1d", d)))
        fields.append(
            Field(
                f"idx_maps_1d[{d}]",
                exact_parity(why="an index into a compact row, counted the same way on both sides"),
                read=_indexed_reader("idx_maps_1d", d),
            )
        )
        fields.append(
            Field(
                f"is_identity_mask_1d[{d}]",
                exact_parity(why="the mask is copied verbatim from the constructor argument"),
                read=_indexed_reader("is_identity_mask_1d", d),
            )
        )
    fields.extend(
        [
            Field(
                "num_intervals",
                exact_parity(why="read straight off the shared space"),
                read=lambda impl: impl.num_intervals,
            ),
            Field(
                "num_total_intervals",
                exact_parity(why="the product of num_intervals"),
                read=lambda impl: impl.num_total_intervals,
            ),
            Field(
                "input_shape_per_dir",
                exact_parity(why="each direction's own operator column count"),
                read=lambda impl: impl.input_shape_per_dir,
            ),
            Field(
                "output_shape_per_dir",
                exact_parity(why="each direction's own operator row count"),
                read=lambda impl: impl.output_shape_per_dir,
            ),
            Field(
                "num_identity_elements",
                exact_parity(why="a product of per-direction identity counts from the same masks"),
                read=lambda impl: impl.num_identity_elements,
            ),
            Field(
                "is_identity",
                exact_parity(why="a conjunction over the same masks"),
                read=lambda impl: impl.is_identity,
            ),
            Field(
                "target",
                exact_parity(why="stored at construction and handed back unread"),
                read=lambda impl: impl.target,
            ),
            Field(
                "lagrange_variant",
                exact_parity(why="stored at construction and handed back unread"),
                read=lambda impl: impl.lagrange_variant,
            ),
            Field(
                "dim",
                exact_parity(why="the shared space's own dimension"),
                read=lambda impl: impl.dim,
            ),
        ]
    )
    return fields


@pytest.mark.parametrize("dtype", DTYPES, ids=["float64", "float32"])
@pytest.mark.parametrize("target", _TARGETS, ids=[t.name for t in _TARGETS])
@pytest.mark.parametrize("case", _COMPACTION_CASES, ids=[c.label for c in _COMPACTION_CASES])
def test_the_compaction_matches_the_oracle_exactly(
    case: _CompactionCase, target: ExtractionTarget, dtype: npt.DTypeLike
) -> None:
    """The C++ compaction agrees with the oracle's, given the identical input arrays.

    Would catch: a wrong sentinel row (a compact block with the wrong number of rows
    for an all-identity direction), an off-by-one in the index map, a mask copied
    from the wrong direction, a decompression that reads the wrong compact row, or
    the target/variant/shape metadata mixed up between directions. ``target`` and
    ``lagrange_variant`` are opaque payload here -- this type never interprets
    either -- so covering all three targets and an arbitrary variant string checks
    only that they are threaded through unchanged, not any target-specific
    arithmetic (that is Section 2 and 3's job).

    Args:
        case (_CompactionCase): The per-direction specs.
        target (ExtractionTarget): Opaque metadata, threaded through unread.
        dtype (npt.DTypeLike): Storage format.
    """
    variant = LagrangeVariant.EQUISPACES
    py_space, cpp_space = _build_space_pair(case.directions, dtype)
    operators, masks = _synthetic_operators(case.directions, dtype)

    oracle = _SpanwiseElementExtractionPython(
        py_space._impl, int(target), str(variant), operators, masks
    )
    cpp_impl = _cpp_impl_class(dtype)(cpp_space._impl, int(target), str(variant), operators, masks)

    ops_why = (
        "compaction only selects and copies rows of the SAME input arrays handed to "
        "both constructors, and decompression either copies a compact row back out or "
        "writes the exact eye(n_out, n_in); no arithmetic runs on either side, so the "
        "same input bits force the same output bits regardless of implementation "
        "language"
    )
    claims = [bitwise_parity(why=ops_why) for _ in case.directions]
    assert_object_parity(
        py=oracle,
        cpp=cpp_impl,
        fields=_state_fields(len(case.directions), claims),
        context=f"{case.label} target={target.name} dtype={np.dtype(dtype).name}",
    )


def test_the_compaction_comparison_catches_a_flipped_mask() -> None:
    """A mask that disagrees between the two constructor calls is caught.

    Would catch: nothing in the port itself -- this is the negative control for
    :func:`test_the_compaction_matches_the_oracle_exactly`, showing that the
    comparison mechanism can fail at all. The two constructors are handed the SAME
    operator arrays but DIFFERENT masks for direction 0; without this, a green
    :func:`test_the_compaction_matches_the_oracle_exactly` would be equally
    consistent with "the comparison never fails".
    """
    dtype = np.float64
    directions = (_MIXED_DIR, _NON_IDENTITY_DIR)
    variant = LagrangeVariant.EQUISPACES
    target = ExtractionTarget.BEZIER
    py_space, cpp_space = _build_space_pair(directions, dtype)
    operators, masks = _synthetic_operators(directions, dtype)

    oracle = _SpanwiseElementExtractionPython(
        py_space._impl, int(target), str(variant), operators, masks
    )
    flipped_masks = [m.copy() for m in masks]
    flipped_masks[0][0] = not flipped_masks[0][0]
    cpp_impl = _cpp_impl_class(dtype)(
        cpp_space._impl, int(target), str(variant), operators, flipped_masks
    )

    claims = [bitwise_parity(why="isolating the mask perturbation") for _ in directions]
    with pytest.raises(AssertionError):
        assert_object_parity(
            py=oracle,
            cpp=cpp_impl,
            fields=_state_fields(len(directions), claims),
            context="flipped mask",
        )


# ---------------------------------------------------------------------------
# Section 2 -- end-to-end parity through the public constructor
# ---------------------------------------------------------------------------


def _build_extraction(
    directions: Sequence[_Case],
    dtype: npt.DTypeLike,
    target: ExtractionTarget,
    variant: LagrangeVariant,
    backend: Backend,
) -> SpanwiseElementExtraction:
    """Build a ``SpanwiseElementExtraction`` end to end, under one backend.

    Args:
        directions (Sequence[_Case]): One knot vector and degree per direction.
        dtype (npt.DTypeLike): Storage format.
        target (ExtractionTarget): The element-local basis.
        variant (LagrangeVariant): The Lagrange point distribution.
        backend (Backend): Which backend to build under.

    Returns:
        SpanwiseElementExtraction: The extraction.
    """
    with use_backend(backend):
        space = BsplineSpace(
            [
                BsplineSpace1D(np.asarray(c.knots, dtype=dtype), c.degree, snap_knots=False)
                for c in directions
            ]
        )
        return SpanwiseElementExtraction(space, target, lagrange_variant=variant)


def _find_case(cases: Sequence[_Case], label: str) -> _Case:
    """Look up one case by its label, for a readable list of directions.

    Args:
        cases (Sequence[_Case]): The table to search.
        label (str): The case's label.

    Returns:
        _Case: The matching case.
    """
    return next(c for c in cases if c.label == label)


_BEZIER_END_TO_END_DIRECTIONS: Final = (
    _find_case(_BEZIER_CASES, "clamped uniform quadratic"),
    _find_case(_BEZIER_CASES, "interior knots at full multiplicity"),
    _find_case(_BEZIER_CASES, "clamped uniform cubic"),
)
"""Three directions: fully non-identity (3 el), fully identity (2 el), fully
non-identity again (4 el) -- asymmetric degree, element count and identity pattern,
built from the SAME table the Bézier builder's own parity file ships."""

_LAGRANGE_END_TO_END_DIRECTIONS: Final = (
    _find_case(_LAGRANGE_CASES, "clamped uniform quadratic"),
    _find_case(_LAGRANGE_CASES, "interior knots at full multiplicity"),
    _find_case(_LAGRANGE_CASES, "clamped uniform cubic"),
)
"""Same three knot vectors, from the Lagrange builder's own table (which excludes
degree 0, so this is that table's own filtering, not a fresh restriction)."""

_CARDINAL_END_TO_END_DIRECTIONS: Final = (
    _Case(
        "uniform degree 2, 6 elements",
        [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0],
        2,
        True,
    ),
    _Case("uniform degree 1, 3 elements", [0.0, 0.0, 1.0, 2.0, 3.0, 3.0], 1, True),
)
"""Two directions with a mix of cardinal (identity) and non-cardinal elements each,
verified by direct construction: masks ``[F,T,T,T,T,F]`` and ``[F,T,F]``."""


@pytest.mark.parametrize("dtype", DTYPES, ids=["float64", "float32"])
def test_the_bezier_end_to_end_extraction_matches_the_oracle(dtype: npt.DTypeLike) -> None:
    """A real ``BEZIER`` extraction, built end to end, matches the oracle.

    Unlike Section 1, the operators here come from the dispatched Bézier builders
    (:mod:`pantr.bspline._extraction_backend`), so the claim is that builder's own,
    imported unchanged rather than re-derived -- inventing a second one here would
    duplicate ``tests/parity/test_bspline_bezier_extraction.py``'s derivation.

    Would catch: this type's compaction disagreeing once fed the builders' real
    (non-integer, non-hand-picked) output, or a wiring bug between
    ``_build_direction_operators`` and the per-direction builder calls.

    Args:
        dtype (npt.DTypeLike): Storage format.
    """
    directions = _BEZIER_END_TO_END_DIRECTIONS
    variant = LagrangeVariant.EQUISPACES
    py_ext = _build_extraction(directions, dtype, ExtractionTarget.BEZIER, variant, Backend.PYTHON)
    cpp_ext = _build_extraction(directions, dtype, ExtractionTarget.BEZIER, variant, Backend.CPP)
    claims = [_bezier_claim(c, dtype) for c in directions]
    assert_object_parity(
        py=py_ext,
        cpp=cpp_ext,
        fields=_state_fields(len(directions), claims),
        context=f"bezier end-to-end in {np.dtype(dtype).name}",
    )


def _lagrange_direction_claim(case: _Case, dtype: npt.DTypeLike, variant: LagrangeVariant) -> Any:
    """The Lagrange builder's own end-to-end claim for one direction.

    Mirrors ``tests/parity/test_bspline_lagrange_extraction.py::test_the_layer_2_path_agrees``
    exactly: measures the gap between the two backends' own change-of-basis
    matrices rather than assuming it is zero, and folds it in as an extra absolute
    term when it is not.

    Args:
        case (_Case): The knot vector.
        dtype (npt.DTypeLike): Storage format.
        variant (LagrangeVariant): The node family.

    Returns:
        Any: The parity claim.
    """
    matrix_py = _matrix_under(Backend.PYTHON, case.degree, variant, dtype)
    matrix_cpp = _matrix_under(Backend.CPP, case.degree, variant, dtype)
    gap = float(
        np.abs(matrix_cpp.astype(np.float64) - matrix_py.astype(np.float64)).max(initial=0.0)
    )
    bezier_ops = _lagrange_bezier_ops(case, dtype)
    extra = None
    if gap > 0.0:
        row_sums = np.abs(bezier_ops.astype(np.float64)).sum(axis=2)
        extra = np.broadcast_to(gap * row_sums[:, :, None], bezier_ops.shape).copy()
    return _lagrange_product_claim(case, dtype, matrix_py, bezier_ops, extra_absolute=extra)


@pytest.mark.parametrize("dtype", DTYPES, ids=["float64", "float32"])
@pytest.mark.parametrize(
    "variant",
    (LagrangeVariant.EQUISPACES, LagrangeVariant.GAUSS_LEGENDRE),
    ids=["equispaces", "gauss_legendre"],
)
def test_the_lagrange_end_to_end_extraction_matches_the_oracle(
    variant: LagrangeVariant, dtype: npt.DTypeLike
) -> None:
    """A real ``LAGRANGE`` extraction, built end to end, matches the oracle.

    Args:
        variant (LagrangeVariant): The node family.
        dtype (npt.DTypeLike): Storage format.
    """
    directions = _LAGRANGE_END_TO_END_DIRECTIONS
    py_ext = _build_extraction(
        directions, dtype, ExtractionTarget.LAGRANGE, variant, Backend.PYTHON
    )
    cpp_ext = _build_extraction(directions, dtype, ExtractionTarget.LAGRANGE, variant, Backend.CPP)
    claims = [_lagrange_direction_claim(c, dtype, variant) for c in directions]
    assert_object_parity(
        py=py_ext,
        cpp=cpp_ext,
        fields=_state_fields(len(directions), claims),
        context=f"lagrange end-to-end {variant.name} in {np.dtype(dtype).name}",
    )


def _cardinal_matrix_under(backend: Backend, degree: int, dtype: npt.DTypeLike) -> npt.NDArray[Any]:
    """The cardinal-to-Bernstein change-of-basis matrix one backend builds.

    Args:
        backend (Backend): Which backend builds it.
        degree (int): Polynomial degree.
        dtype (npt.DTypeLike): Storage format.

    Returns:
        npt.NDArray[Any]: The read-only ``(degree+1, degree+1)`` matrix.
    """
    # The one place `dtype` is narrowed for the cache's signature, matching
    # `tests.parity.test_bspline_lagrange_extraction._matrix_under`'s own cast: every
    # caller here passes a member of DTYPES, so this restates what the parametrization
    # already guarantees and does not widen anything.
    resolved = cast("np.dtype[np.float32 | np.float64]", np.dtype(dtype))
    with use_backend(backend):
        return np.asarray(_cached_cardinal_to_bernstein_matrix(degree, resolved))


def _restrict_to_compact(
    full: npt.NDArray[np.float64], mask: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """Restrict a per-element array to the rows this type's compaction keeps.

    Mirrors ``_SpanwiseElementExtractionPython.__init__``'s own rule exactly: the
    non-identity rows, in ascending order, or one zero row when there are none.

    Args:
        full (npt.NDArray[np.float64]): One row per element.
        mask (npt.NDArray[np.bool_]): The identity mask, same length as ``full``.

    Returns:
        npt.NDArray[np.float64]: The compacted rows.
    """
    non_identity = full[~mask]
    if non_identity.shape[0] == 0:
        return np.zeros((1, *full.shape[1:]), dtype=np.float64)
    return non_identity


def _cardinal_direction_claims(case: _Case, dtype: npt.DTypeLike) -> tuple[Any, Any]:
    """The end-to-end parity claim for one cardinal direction, as ``(compact, dense)``.

    **Measured, not assumed.** There is no C++ builder for the cardinal
    *extraction* operator, but ``A_e = C_e @ card_to_bzr`` (see
    ``pantr.bspline._bspline_extraction._tabulate_Bspline_cardinal_1D_extraction_impl``),
    and ``card_to_bzr`` -- the cardinal-to-Bernstein change of basis -- IS
    separately dispatched by :mod:`pantr.change_basis`. Measured: it does not agree
    bit for bit between backends (a gap of ``1.5e-14`` at degree 3, float64;
    ``2.6e-05`` at float32), so the naive "same Numba function on both sides"
    assumption this file's first draft made is false, and the claim here is bounded
    instead, derived exactly the way the Lagrange target's own is:

    - a **contraction** term: ``degree + 1`` terms summed by a BLAS ``gemm`` on the
      oracle's side and an ascending loop on the C++ side, bounded by
      ``gamma_{degree+1} |C_e|^T |card_to_bzr|`` (Higham, *Accuracy and Stability of
      Numerical Algorithms*, 2nd ed., section 3.1), with the absolute-value
      companion Rule 10 prescribes since ``card_to_bzr`` has negative entries;
    - an **absolute** term for the matrix's own measured gap, propagated as
      ``sum_k |C[i,k]| |d(card_to_bzr)[k,j]| <= gap * row_sum(C)`` and folded into
      the amplification after dividing by this claim's own gamma and the harness's
      factor of two -- the identical transformation
      ``tests/parity/test_bspline_lagrange_extraction.py::test_the_layer_2_path_agrees``
      applies to its own matrix gap.

    Verified over three uniform cases (degree 1, 2 and 3, both storage formats)
    before this was trusted: the worst observed ratio of measured deviation to this
    bound was ``1.0`` (degree 1, tight but not exceeded), never above it.

    Args:
        case (_Case): The knot vector.
        dtype (npt.DTypeLike): Storage format.

    Returns:
        tuple[Any, Any]: ``(compact_claim, dense_claim)``. They differ only in
        which rows of the per-element companion they carry -- the dense claim
        covers every element, the compact one only the non-identity rows this type
        actually stores, in the ascending order the compaction keeps them.
    """
    knots = np.asarray(case.knots, dtype=dtype)
    space = BsplineSpace1D(knots, case.degree, snap_knots=False)
    with use_backend(Backend.PYTHON):
        bezier_ops = np.asarray(space.tabulate_Bezier_extraction_operators())
        mask = np.asarray(space.get_cardinal_intervals())

    matrix_py = _cardinal_matrix_under(Backend.PYTHON, case.degree, dtype)
    matrix_cpp = _cardinal_matrix_under(Backend.CPP, case.degree, dtype)
    gap = float(
        np.abs(matrix_cpp.astype(np.float64) - matrix_py.astype(np.float64)).max(initial=0.0)
    )

    terms = case.degree + 1
    u = unit_roundoff(dtype)
    m = float(terms)
    local_gamma = (m * u) / (1.0 - m * u)
    amplification = _lagrange_companion(bezier_ops, matrix_py)
    why = (
        "A_e = C_e @ card_to_bzr, the same shape as the Lagrange target's own product; "
        "card_to_bzr is measured to differ between backends (a real gap here, unlike the "
        "Lagrange target's zero gap at EQUISPACES), so this is bounded rather than "
        "bitwise. Two terms: gamma_{degree+1} times the absolute-value companion "
        "|C_e| @ |card_to_bzr| for the contraction (Higham section 3.1, blocking only "
        "tightens it), plus gap * row_sum(|C_e|) / (2 * gamma_{degree+1}) for the "
        "matrix's own measured gap, which the harness's factor of two and this claim's "
        "gamma convert back into an amplification. Verified against the measured "
        "deviation at degree 1, 2 and 3 in both storage formats before being trusted; "
        "worst observed ratio to this bound was 1.0"
    )
    if gap > 0.0:
        row_sums = np.abs(bezier_ops.astype(np.float64)).sum(axis=2)
        amplification = amplification + (gap * row_sums[:, :, None]) / (2.0 * local_gamma)

    dense_claim = bounded_parity(
        roundings=Roundings(stages=terms, accumulator_per_stage=1, storage_per_stage=0),
        accumulator=dtype,
        storage=dtype,
        amplification=amplification,
        why=why,
    )
    compact_amplification = _restrict_to_compact(amplification, mask)
    compact_claim = bounded_parity(
        roundings=Roundings(stages=terms, accumulator_per_stage=1, storage_per_stage=0),
        accumulator=dtype,
        storage=dtype,
        amplification=compact_amplification,
        why=why,
    )
    return compact_claim, dense_claim


@pytest.mark.parametrize("dtype", DTYPES, ids=["float64", "float32"])
def test_the_cardinal_end_to_end_extraction_matches_the_oracle(dtype: npt.DTypeLike) -> None:
    """A real ``CARDINAL`` extraction, built end to end, matches the oracle within the bound.

    Not bitwise, though the extraction operator itself has no C++ builder: the
    change-of-basis matrix it contracts against does, and measurement shows it does
    not agree bit for bit between backends. See :func:`_cardinal_direction_claims`.

    Args:
        dtype (npt.DTypeLike): Storage format.
    """
    directions = _CARDINAL_END_TO_END_DIRECTIONS
    variant = LagrangeVariant.EQUISPACES
    py_ext = _build_extraction(
        directions, dtype, ExtractionTarget.CARDINAL, variant, Backend.PYTHON
    )
    cpp_ext = _build_extraction(directions, dtype, ExtractionTarget.CARDINAL, variant, Backend.CPP)
    compact_claims = []
    dense_claims = []
    for c in directions:
        compact_claim, dense_claim = _cardinal_direction_claims(c, dtype)
        compact_claims.append(compact_claim)
        dense_claims.append(dense_claim)
    assert_object_parity(
        py=py_ext,
        cpp=cpp_ext,
        fields=_state_fields(len(directions), compact_claims, dense_claims),
        context=f"cardinal end-to-end in {np.dtype(dtype).name}",
    )


# ---------------------------------------------------------------------------
# Section 3a -- independent accuracy: partition of unity, per target
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES, ids=["float64", "float32"])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["python", "cpp"])
@pytest.mark.parametrize(
    "case",
    [c for c in _BEZIER_CASES if c.accuracy],
    ids=[c.label for c in _BEZIER_CASES if c.accuracy],
)
def test_the_bezier_columns_of_ops_1d_sum_to_one(
    case: _Case, backend: Backend, dtype: npt.DTypeLike
) -> None:
    """Every column of a real, decompressed ``ops_1d`` block sums to one.

    Reads ``ops_1d`` off a real ``SpanwiseElementExtraction``, so this exercises the
    actual decompression path (not the raw builder output the reference file
    checks), against the same bound: decompression is a pure copy or an exact
    ``eye`` fill and adds no rounding beyond what the builder already committed.

    Args:
        case (_Case): The knot vector.
        backend (Backend): Which implementation builds the extraction.
        dtype (npt.DTypeLike): Storage format.
    """
    ext = _build_extraction(
        (case,), dtype, ExtractionTarget.BEZIER, LagrangeVariant.EQUISPACES, backend
    )
    ops = np.asarray(ext.ops_1d[0])
    column_sums = ops.astype(np.float64).sum(axis=1)
    assert_accuracy(
        column_sums,
        np.ones_like(column_sums),
        derived_accuracy(
            bound=np.full(column_sums.shape, _column_sum_bound(case, dtype)),
            why=(
                "reused unchanged from tests.parity.test_bspline_bezier_extraction."
                "_column_sum_bound, which bounds exactly this deviation for exactly these "
                "operators; decompression through ops_1d adds no further rounding"
            ),
        ),
        context=f"{case.label} in {np.dtype(dtype).name} on {backend.name}",
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=["float64", "float32"])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["python", "cpp"])
@pytest.mark.parametrize(
    "case",
    [c for c in _LAGRANGE_CASES if c.accuracy],
    ids=[c.label for c in _LAGRANGE_CASES if c.accuracy],
)
def test_the_lagrange_columns_of_ops_1d_sum_to_the_matrix_columns(
    case: _Case, backend: Backend, dtype: npt.DTypeLike
) -> None:
    """Every column of a real, decompressed Lagrange ``ops_1d`` sums to the matrix's own.

    ``EQUISPACES`` only: its change-of-basis matrix is common mode between the two
    backends (measured zero gap), which is what lets one matrix, built once, stand
    for "the matrix this extraction used" regardless of which backend built the
    extraction. A wider variant sweep of this identity is
    ``tests/parity/test_bspline_lagrange_extraction.py``'s own, not restated here.

    Args:
        case (_Case): The knot vector.
        backend (Backend): Which implementation builds the extraction.
        dtype (npt.DTypeLike): Storage format.
    """
    variant = LagrangeVariant.EQUISPACES
    matrix = _matrix_under(Backend.PYTHON, case.degree, variant, dtype)
    ext = _build_extraction((case,), dtype, ExtractionTarget.LAGRANGE, variant, backend)
    ops = np.asarray(ext.ops_1d[0])
    bezier_ops = _lagrange_bezier_ops(case, dtype)
    companion = _lagrange_companion(bezier_ops, matrix)
    column_sums = ops.astype(np.float64).sum(axis=1)
    target_sums = np.broadcast_to(np.abs(matrix.astype(np.float64)).sum(axis=0), column_sums.shape)
    assert_accuracy(
        column_sums,
        target_sums,
        derived_accuracy(
            bound=np.broadcast_to(
                _column_sum_tolerance(case, dtype, matrix, companion), column_sums.shape
            ).copy(),
            why=(
                "reused unchanged from tests.parity.test_bspline_lagrange_extraction."
                "_column_sum_tolerance; decompression through ops_1d adds no further rounding"
            ),
        ),
        context=f"{case.label} in {np.dtype(dtype).name} on {backend.name}",
    )


_CARDINAL_HEURISTIC_MULTIPLE: Final = 100.0
"""How many extra copies of the summation term the entries' own error may cost.

**Observed, not proved.** The cardinal-to-Bernstein matrix has negative entries (the
smallest measured, at degree 3, is -6), so the Bézier/Lagrange convexity argument
does not apply and there is no derivation anywhere in the tree for the error already
present in the cardinal builder's own entries -- only for the summation that forms a
column sum from them, which is term (a) in :func:`_cardinal_summation_term`.

Calibrated by sampling ``deviation / (a)`` over 200 random clamped, roughly-uniform
knot vectors per degree in ``{1, 2, 3}`` and both storage formats: the worst
observed ratio was 27, at degree 3, with interior knot multiplicities up to 2. This
multiple leaves about 3.7x of headroom over that worst case, while still being tight
enough that :func:`test_the_cardinal_column_sum_check_catches_a_scaled_column` can
tell a materially wrong operator from a correct one.

**Restricted to degree <= 3.** The same measurement at degree 5 reached a ratio in
the tens of thousands (the operator's own entries reach the hundreds, against 2-6 at
degree 2-3), so this multiple is not claimed there, and no cardinal case in this file
goes past degree 3.
"""


def _cardinal_summation_term(
    ops: npt.NDArray[Any], dtype: npt.DTypeLike
) -> npt.NDArray[np.float64]:
    """Higham eq. (4.4)'s summation bound for one column sum, elementwise.

    Args:
        ops (npt.NDArray[Any]): The ``(n_elements, n_out, n_in)`` operator.
        dtype (npt.DTypeLike): The format the summation runs in.

    Returns:
        npt.NDArray[np.float64]: One bound per ``(element, column)``, term (a) of
        the module docstring's cardinal derivation.
    """
    n_out = ops.shape[1]
    u = unit_roundoff(dtype)
    m = n_out - 1
    gamma = (m * u) / (1.0 - m * u) if m > 0 else 0.0
    abscol = np.abs(ops.astype(np.float64)).sum(axis=1)
    return np.asarray(gamma * abscol, dtype=np.float64)


def _cardinal_column_sum_bound(
    ops: npt.NDArray[Any], dtype: npt.DTypeLike
) -> npt.NDArray[np.float64]:
    """The full cardinal column-sum bound: term (a) plus the admitted heuristic multiple.

    Args:
        ops (npt.NDArray[Any]): The ``(n_elements, n_out, n_in)`` operator.
        dtype (npt.DTypeLike): Storage format.

    Returns:
        npt.NDArray[np.float64]: One bound per ``(element, column)``.
    """
    term = _cardinal_summation_term(ops, dtype)
    return np.asarray((1.0 + _CARDINAL_HEURISTIC_MULTIPLE) * term, dtype=np.float64)


_CARDINAL_ACCURACY_CASES: Final = (
    _Case(
        "uniform quadratic, 6 elements",
        [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0],
        2,
        True,
    ),
    _Case(
        "uniform cubic, 8 elements",
        [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 8.0, 8.0],
        3,
        True,
    ),
)
"""Two uniform vectors at the degrees :data:`_CARDINAL_HEURISTIC_MULTIPLE` was
calibrated for."""


@pytest.mark.parametrize("dtype", DTYPES, ids=["float64", "float32"])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["python", "cpp"])
@pytest.mark.parametrize(
    "case", _CARDINAL_ACCURACY_CASES, ids=[c.label for c in _CARDINAL_ACCURACY_CASES]
)
def test_the_cardinal_columns_of_ops_1d_are_within_the_heuristic_bound(
    case: _Case, backend: Backend, dtype: npt.DTypeLike
) -> None:
    """Every column of a real, decompressed cardinal ``ops_1d`` sums to one within the bound.

    Args:
        case (_Case): The knot vector.
        backend (Backend): Which implementation builds the extraction.
        dtype (npt.DTypeLike): Storage format.
    """
    ext = _build_extraction(
        (case,), dtype, ExtractionTarget.CARDINAL, LagrangeVariant.EQUISPACES, backend
    )
    ops = np.asarray(ext.ops_1d[0])
    column_sums = ops.astype(np.float64).sum(axis=1)
    assert_accuracy(
        column_sums,
        np.ones_like(column_sums),
        derived_accuracy(
            bound=_cardinal_column_sum_bound(ops, dtype),
            why=(
                "(a) Higham eq. (4.4)'s summation bound, gamma_{n_out-1} * sum_i|C[e,i,j]|, "
                "computed from this operator's own entries; (b) an ADMITTED HEURISTIC "
                "multiple of (a) -- see _CARDINAL_HEURISTIC_MULTIPLE -- covering the error "
                "already present in the cardinal builder's own entries, which is not "
                "derived anywhere in the tree. Observed, not proved. Restricted to degree "
                "<= 3, which this case does not exceed"
            ),
        ),
        context=f"{case.label} in {np.dtype(dtype).name} on {backend.name}",
    )


def test_the_column_sum_checks_are_not_vacuous() -> None:
    """Each partition-of-unity check is compared against a nonzero deviation somewhere.

    A bound compared only against zero has not been checked. All three targets'
    checks sweep case tables that include non-dyadic knot vectors, so this pins that
    at least one of them actually rounds -- rather than every column sum landing on
    exactly one, which a builder that did nothing at all would also produce.
    """
    dtype = np.float32
    worst_bezier = 0.0
    for case in _BEZIER_CASES:
        if not case.accuracy:
            continue
        ext = _build_extraction(
            (case,), dtype, ExtractionTarget.BEZIER, LagrangeVariant.EQUISPACES, Backend.PYTHON
        )
        ops = np.asarray(ext.ops_1d[0])
        deviation = float(np.abs(ops.astype(np.float64).sum(axis=1) - 1.0).max(initial=0.0))
        worst_bezier = max(worst_bezier, deviation)
    assert worst_bezier > 0.0, "no Bezier case rounded at all; the column-sum bound is vacuous"

    worst_lagrange = 0.0
    variant = LagrangeVariant.EQUISPACES
    for case in _LAGRANGE_CASES:
        if not case.accuracy:
            continue
        matrix = _matrix_under(Backend.PYTHON, case.degree, variant, dtype)
        ext = _build_extraction((case,), dtype, ExtractionTarget.LAGRANGE, variant, Backend.PYTHON)
        ops = np.asarray(ext.ops_1d[0])
        target_sums = np.abs(matrix.astype(np.float64)).sum(axis=0)
        deviation = float(np.abs(ops.astype(np.float64).sum(axis=1) - target_sums).max(initial=0.0))
        worst_lagrange = max(worst_lagrange, deviation)
    assert worst_lagrange > 0.0, "no Lagrange case rounded at all; the column-sum bound is vacuous"

    worst_cardinal = 0.0
    for case in _CARDINAL_ACCURACY_CASES:
        ext = _build_extraction(
            (case,), dtype, ExtractionTarget.CARDINAL, LagrangeVariant.EQUISPACES, Backend.PYTHON
        )
        ops = np.asarray(ext.ops_1d[0])
        deviation = float(np.abs(ops.astype(np.float64).sum(axis=1) - 1.0).max(initial=0.0))
        worst_cardinal = max(worst_cardinal, deviation)
    assert worst_cardinal > 0.0, "no cardinal case rounded at all; the column-sum bound is vacuous"


# ---------------------------------------------------------------------------
# Section 3b -- independent accuracy: the exact integer Kronecker oracle
# ---------------------------------------------------------------------------


def _wrap_impl(space: BsplineSpace, impl: Any) -> SpanwiseElementExtraction:
    """Build a ``SpanwiseElementExtraction`` around an impl this file constructed directly.

    Repeats exactly what ``SpanwiseElementExtraction.__init__`` does with
    ``object.__setattr__`` -- construct-then-freeze, one field at a time -- with the
    caller's own impl in place of one built through the real constructor's
    ``_new_impl``. Needed because the exact-integer oracle below wants small-integer
    operators no builder produces, and the public constructor always rebuilds
    operators through the real B-spline builders.

    Args:
        space (BsplineSpace): The space to report from ``.space``.
        impl (Any): An already-built oracle instance or C++ handle.

    Returns:
        SpanwiseElementExtraction: The wrapper.
    """
    wrapper = object.__new__(SpanwiseElementExtraction)
    object.__setattr__(wrapper, "_impl", impl)
    object.__setattr__(wrapper, "_space", space)
    object.__setattr__(wrapper, "_compact_ops_1d", tuple(impl.compact_ops_1d))
    object.__setattr__(wrapper, "_idx_maps_1d", tuple(impl.idx_maps_1d))
    object.__setattr__(wrapper, "_is_identity_mask_1d", tuple(impl.is_identity_mask_1d))
    object.__setattr__(wrapper, "_dense_ops_1d", None)
    return wrapper


def _direct_impl(
    space: BsplineSpace,
    backend: Backend,
    target: ExtractionTarget,
    variant: LagrangeVariant,
    arrays: tuple[Sequence[npt.NDArray[Any]], Sequence[npt.NDArray[np.bool_]]],
) -> Any:
    """Construct one backend's implementation directly from given operator arrays.

    Args:
        space (BsplineSpace): The space, built under ``backend``.
        backend (Backend): Which implementation to build.
        target (ExtractionTarget): Opaque metadata.
        variant (LagrangeVariant): Opaque metadata.
        arrays (tuple[Sequence[npt.NDArray[Any]], Sequence[npt.NDArray[np.bool_]]]):
            ``(operators, masks)``, one dense operator and one mask per direction.
            ``operators[0].dtype`` fixes the storage format.

    Returns:
        Any: The oracle instance or the C++ handle.
    """
    operators, masks = arrays
    if backend is Backend.PYTHON:
        return _SpanwiseElementExtractionPython(
            space._impl, int(target), str(variant), operators, masks
        )
    return _cpp_impl_class(operators[0].dtype)(
        space._impl, int(target), str(variant), operators, masks
    )


class _KronCase(NamedTuple):
    """One dimension's worth of directions for the exact-integer oracle.

    Attributes:
        label (str): What the case exercises.
        directions (tuple[_SynDirection, ...]): The per-direction specs.
    """

    label: str
    directions: tuple[_SynDirection, ...]


_KRON_CASES: Final = (
    _KronCase("dim 1", (_MIXED_DIR,)),
    _KronCase(
        "dim 3, some fully non-identity cells",
        (_NON_IDENTITY_DIR, _NON_IDENTITY_DIR, _MIXED_DIR),
    ),
)
"""``dim 3`` includes cells where every direction is simultaneously non-identity, so
the Kronecker product actually combines three factors rather than degenerating to
one block or an eye."""


@pytest.mark.parametrize("dtype", DTYPES, ids=["float64", "float32"])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["python", "cpp"])
@pytest.mark.parametrize("case", _KRON_CASES, ids=[c.label for c in _KRON_CASES])
def test_the_exact_integer_kron_oracle(
    case: _KronCase, backend: Backend, dtype: npt.DTypeLike
) -> None:
    """``operator(cell)`` equals a Python-integer ``numpy.kron`` computation exactly.

    Independent of anything B-spline: the operators are hand-picked small integers,
    so ``kron(M_0, ..., M_{d-1})`` is exactly representable in binary64 regardless of
    storage format. Would catch a transposed direction or a swapped mode order in
    either backend's compaction or in the wrapper's own Kronecker assembly -- an
    error class a parity comparison between the two backends cannot see, since both
    would share the same mistake, and the partition-of-unity check cannot see either,
    since a transposition does not change a column sum.

    Args:
        case (_KronCase): The per-direction specs.
        backend (Backend): Which implementation to build.
        dtype (npt.DTypeLike): Storage format.
    """
    dim = len(case.directions)
    py_space, cpp_space = _build_space_pair(case.directions, dtype)
    space = py_space if backend is Backend.PYTHON else cpp_space
    operators, masks = _synthetic_operators(case.directions, dtype)
    variant = LagrangeVariant.EQUISPACES
    target = ExtractionTarget.BEZIER
    impl = _direct_impl(space, backend, target, variant, (operators, masks))
    ext = _wrap_impl(space, impl)

    num_intervals = tuple(len(d.identity) for d in case.directions)
    total = 1
    for n in num_intervals:
        total *= n
    nontrivial_seen = False
    for flat in range(total):
        multi = np.unravel_index(flat, num_intervals) if dim > 1 else (flat,)
        expected = operators[0][multi[0]].astype(np.int64)
        for k in range(1, dim):
            expected = np.kron(expected, operators[k][multi[k]].astype(np.int64))
        computed = np.asarray(ext.operator(flat), dtype=np.float64)
        assert_accuracy(
            computed,
            expected.astype(np.float64),
            derived_accuracy(
                bound=np.zeros_like(expected, dtype=np.float64),
                why=(
                    "every operator entry is a small integer, so kron(M_0, ..., M_{d-1}) "
                    "is exactly representable in binary64 and the wrapper's materialised "
                    "operator must equal it bit for bit"
                ),
            ),
            context=f"{case.label} cell {flat} on {backend.name} in {np.dtype(dtype).name}",
        )
        all_non_identity = all(
            not direction.identity[multi[i]] for i, direction in enumerate(case.directions)
        )
        if dim >= 2 and all_non_identity:
            nontrivial_seen = True

    if dim >= 2:
        assert nontrivial_seen, (
            "no tested cell combined non-identity operators from every direction, so the "
            "sweep never exercised a genuine multi-factor Kronecker product and the float "
            "result was trivially a single stored block"
        )


def test_the_exact_integer_oracle_catches_a_transposed_direction() -> None:
    """The exact-integer oracle catches a deliberately transposed direction.

    Would catch: nothing in the port itself -- this is the negative control. It
    builds the impl from a transposed copy of direction 0's operators, and compares
    the resulting ``operator(cell)`` against the kron of the ORIGINAL (correct)
    operators, exactly as a real index-order bug would be caught if it existed.
    """
    dtype = np.float64
    directions = (_NON_IDENTITY_DIR, _MIXED_DIR)
    py_space, _ = _build_space_pair(directions, dtype)
    operators, masks = _synthetic_operators(directions, dtype)
    variant = LagrangeVariant.EQUISPACES
    target = ExtractionTarget.BEZIER

    wrong_operators = [np.ascontiguousarray(np.swapaxes(operators[0], 1, 2)), operators[1]]
    impl = _SpanwiseElementExtractionPython(
        py_space._impl, int(target), str(variant), wrong_operators, masks
    )
    ext = _wrap_impl(py_space, impl)

    expected = np.kron(operators[0][0].astype(np.int64), operators[1][0].astype(np.int64))
    computed = np.asarray(ext.operator(0), dtype=np.float64)
    with pytest.raises(AssertionError):
        assert_accuracy(
            computed,
            expected.astype(np.float64),
            derived_accuracy(bound=np.zeros_like(expected, dtype=np.float64), why="mutation check"),
            context="transposed direction",
        )


# ---------------------------------------------------------------------------
# Section 4 -- negative controls for the partition-of-unity checks
# ---------------------------------------------------------------------------


def test_the_bezier_column_sum_check_catches_a_scaled_column() -> None:
    """A materially scaled column fails the Bézier column-sum check.

    Would catch: nothing in the port -- the negative control for
    :func:`test_the_bezier_columns_of_ops_1d_sum_to_one`. Scales one column by a
    factor far above the derived bound (which is on the order of a few units of
    roundoff), and confirms :func:`~tests._parity_harness.assert_accuracy` raises.
    """
    dtype = np.float64
    case = _find_case(_BEZIER_CASES, "clamped uniform quadratic")
    ext = _build_extraction(
        (case,), dtype, ExtractionTarget.BEZIER, LagrangeVariant.EQUISPACES, Backend.PYTHON
    )
    ops = np.array(ext.ops_1d[0], copy=True, dtype=np.float64)
    ops[0, :, 0] *= 1.0 + 1.0e-6
    column_sums = ops.sum(axis=1)
    bound = np.full(column_sums.shape, _column_sum_bound(case, dtype))
    with pytest.raises(AssertionError):
        assert_accuracy(
            column_sums,
            np.ones_like(column_sums),
            derived_accuracy(bound=bound, why="mutation check"),
            context="scaled column",
        )


def test_the_lagrange_column_sum_check_catches_a_scaled_column() -> None:
    """A materially scaled column fails the Lagrange column-sum check.

    Would catch: nothing in the port -- the negative control for
    :func:`test_the_lagrange_columns_of_ops_1d_sum_to_the_matrix_columns`.
    """
    dtype = np.float64
    variant = LagrangeVariant.EQUISPACES
    case = _find_case(_LAGRANGE_CASES, "clamped uniform quadratic")
    matrix = _matrix_under(Backend.PYTHON, case.degree, variant, dtype)
    ext = _build_extraction((case,), dtype, ExtractionTarget.LAGRANGE, variant, Backend.PYTHON)
    bezier_ops = _lagrange_bezier_ops(case, dtype)
    companion = _lagrange_companion(bezier_ops, matrix)
    ops = np.array(ext.ops_1d[0], copy=True, dtype=np.float64)
    ops[0, :, 0] *= 1.0 + 1.0e-6
    column_sums = ops.sum(axis=1)
    target_sums = np.broadcast_to(np.abs(matrix.astype(np.float64)).sum(axis=0), column_sums.shape)
    bound = np.broadcast_to(
        _column_sum_tolerance(case, dtype, matrix, companion), column_sums.shape
    )
    with pytest.raises(AssertionError):
        assert_accuracy(
            column_sums,
            target_sums,
            derived_accuracy(bound=bound.copy(), why="mutation check"),
            context="scaled column",
        )


def test_the_cardinal_column_sum_check_catches_a_scaled_column() -> None:
    """A materially scaled column fails the cardinal column-sum check.

    Would catch: nothing in the port -- the negative control for
    :func:`test_the_cardinal_columns_of_ops_1d_are_within_the_heuristic_bound`. The
    cardinal bound is far looser than Bézier's or Lagrange's (it carries the admitted
    heuristic multiple), so the perturbation here is a full 50% scale rather than a
    few units of roundoff, to discriminate against that looser bound.
    """
    dtype = np.float64
    case = _CARDINAL_ACCURACY_CASES[0]
    ext = _build_extraction(
        (case,), dtype, ExtractionTarget.CARDINAL, LagrangeVariant.EQUISPACES, Backend.PYTHON
    )
    ops = np.array(ext.ops_1d[0], copy=True, dtype=np.float64)
    bound = _cardinal_column_sum_bound(ops, dtype)
    ops[0, :, 0] *= 1.5
    column_sums = ops.sum(axis=1)
    with pytest.raises(AssertionError):
        assert_accuracy(
            column_sums,
            np.ones_like(column_sums),
            derived_accuracy(bound=bound, why="mutation check"),
            context="scaled column",
        )


# ---------------------------------------------------------------------------
# Section 5 -- AC8: the claim and the column-sum bound over a sweep 10x the shipped one
# ---------------------------------------------------------------------------


_SHIPPED_END_TO_END_DRAWS: Final = 2
"""One 3-direction Bézier end-to-end combination, times the two dtypes -- what
:func:`test_the_bezier_end_to_end_extraction_matches_the_oracle` ships."""


@pytest.mark.slow
@pytest.mark.parametrize("dtype", DTYPES, ids=["float64", "float32"])
def test_the_bezier_claim_and_column_sum_hold_over_a_sweep_ten_times_the_shipped_one(
    dtype: npt.DTypeLike,
) -> None:
    """The end-to-end parity claim and the column-sum bound, over a larger random sweep.

    Mirrors the mechanism ``tests/parity/test_bspline_bezier_extraction.py`` uses for
    its own 10x sweep (``test_the_claim_holds_over_a_sweep_ten_times_the_shipped_one``):
    draw random per-direction cases from the same family the shared algorithm
    handles (:func:`tests.parity.test_bspline_bezier_extraction._draw`), assemble
    them into spaces of dimension 1 through 3, and assert both the end-to-end parity
    claim and the column-sum bound on each.

    Args:
        dtype (npt.DTypeLike): Storage format.
    """
    draws = 10 * _SHIPPED_END_TO_END_DRAWS
    checked = 0
    worst_column_sum = 0.0
    worst_column_ratio = 0.0
    for draw in range(draws):
        dim = 1 + (draw % 3)
        directions = tuple(
            _draw_bezier_case(np.random.default_rng(800_000 + 31 * draw + 7 * d), dtype)
            for d in range(dim)
        )
        variant = LagrangeVariant.EQUISPACES
        py_ext = _build_extraction(
            directions, dtype, ExtractionTarget.BEZIER, variant, Backend.PYTHON
        )
        cpp_ext = _build_extraction(
            directions, dtype, ExtractionTarget.BEZIER, variant, Backend.CPP
        )
        claims = [_bezier_claim(c, dtype) for c in directions]
        context = f"draw {draw}, dim {dim}, in {np.dtype(dtype).name}"
        assert_object_parity(
            py=py_ext, cpp=cpp_ext, fields=_state_fields(dim, claims), context=context
        )

        for d, case in enumerate(directions):
            ops = np.asarray(cpp_ext.ops_1d[d])
            column_sums = ops.astype(np.float64).sum(axis=1)
            deviation = float(np.abs(column_sums - 1.0).max(initial=0.0))
            bound_value = _column_sum_bound(case, dtype)
            worst_column_sum = max(worst_column_sum, deviation)
            if bound_value > 0.0:
                worst_column_ratio = max(worst_column_ratio, deviation / bound_value)
        checked += 1

    assert checked == draws, f"the sweep ran {checked} cases, expected {draws}"
    assert worst_column_sum > 0.0, (
        "no drawn case rounded at all, so the partition-of-unity bound was compared "
        "against zero throughout the sweep"
    )
    print(
        f"AC8 sweep ({np.dtype(dtype).name}): {checked} draws, worst column-sum deviation "
        f"{worst_column_sum:.3e}, worst ratio to bound {worst_column_ratio:.3e}"
    )
