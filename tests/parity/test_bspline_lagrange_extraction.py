"""Parity for the Lagrange extraction operator builder and its identity mask.

``A_e = C_e @ L``: the Bézier operator of interval ``e``, post-multiplied by the
Lagrange-to-Bernstein matrix ``L[j, k] = B_j(x_k)`` of the same degree. The Bézier
half is ``tests/parity/test_bspline_bezier_extraction.py``'s and is reused from
there rather than restated -- the case table, the chain length and the Bézier column
bound are imported, so the two files cannot drift on them.

**The claim is bounded, not bitwise, and that is the one way this target differs in
kind from its Bézier parent.** The oracle contracts the product with
:func:`numpy.matmul`, which reaches a BLAS ``gemm`` and sums the ``degree + 1`` terms
in an unspecified, possibly blocked order; the C++ runs a plain ascending loop. Both
accumulate in the storage format -- ``sgemm`` at ``float32`` on one side and
``design/backend_parity.md`` Rule 9's "the accumulator is `T`" on the other -- so the
budget is a rounding per term with no narrowing store, and the amplification is the
absolute-value companion Rule 10 prescribes. Measured on this machine, the two
backends do differ, by one unit of roundoff at degree 3 and above and not at all
below; :func:`test_the_two_backends_still_differ_somewhere` is the guard that keeps
the bound from silently becoming a comparison against zero.

**What a bounded claim cannot see here, and where it is checked instead.** Nothing in
this file would fail if the C++ accumulated the contraction in ``double``, which
``design/backend_parity.md`` Rule 9 forbids. That is not a hole in the bound and no
constant would close it: the bound is derived from each backend's forward error against
the exact product, and a wider accumulator has a **smaller** forward error, so it lies
inside the same bound by construction. Rule 8 records the general form -- "a different
but valid algorithm passing is not a failure of the bound". Measured alongside the
argument: mutating the accumulator leaves all cases green, and the gap it opens at
``float32`` is one unit in the last place, the same size as the two backends' own
disagreement. ``cpp/tests/test_bspline_extraction.cpp``'s
``check_the_accumulator_is_the_storage_type`` is where the width is pinned, on the C++
side where there is no BLAS and both operands can be chosen.

The exception is an **identity** change of basis, which happens at degree 1 for the
equispaced, Gauss-Lobatto-Legendre and second-kind Chebyshev families. Every term is
then ``C[i,k] * 0`` or ``C[i,j] * 1``, so nothing rounds in either summation order
and the claim falls back to the Bézier one, bit for bit or Rule 10's fused budget
according to the build.

Why the matrix is an argument, and what that buys the claim
----------------------------------------------------------

Both kernels take ``L`` rather than building it. :mod:`pantr.change_basis` owns the
tabulation and caches it per ``(degree, variant, dtype)``, and
:class:`pantr.basis.LagrangeVariant` is a :class:`~enum.StrEnum`, which numba types
as ``unicode_type`` and then silently mis-compares against a captured member -- so
the variant must not approach a kernel at all.

For the claim the consequence is that ``L`` is **common mode**:
:func:`test_the_operators_agree` hands the same array to both backends, so what it
measures is the extraction. ``L``'s own parity is
``tests/parity/test_change_basis.py::test_lagrange_to_bernstein_is_bitwise``, and it
is bitwise except where the node family itself dispatches.
:func:`test_the_layer_2_path_agrees` is the end-to-end companion that does not hold
it fixed: it measures the gap between the two backends' matrices and carries it
through the product, exactly as that file measures its node gap.

The three independent accuracy oracles
--------------------------------------

Parity says the two backends agree, not that either is right, and a transposed index
would be invisible to it. None of the three below is the Python implementation.

**Exact rational values, hand-derived.** At degree 2 with equispaced nodes,
``L = [[1, 1/4, 0], [0, 1/2, 0], [0, 1/4, 1]]`` follows from ``L[j,k] = B_j(x_k)`` at
``x = 0, 1/2, 1``, and the dyadic uniform Bézier tables are halves, so every entry of
the product and every partial sum of the contraction is a binary rational. The
comparison carries a **zero** bound in both storage formats.
``cpp/tests/test_bspline_extraction.cpp`` carries the same tables.

**The column sums reproduce the matrix's own.** ``sum_i A[i,j] = sum_k (sum_i
C[i,k]) L[k,j] = sum_k L[k,j]``, using the Bézier column sums. Compared against
``sum_k L[k,j]`` **as the matrix actually sums**, not against one, so ``L``'s own
tabulation error is folded in exactly rather than bounded. Columns and not rows: the
quadratic table's rows sum to ``1.25, 1.75, 0.625``, so this is what catches a
transposition.

It also pins that no entry is negative, which is a correction rather than a
restatement: ``design/extraction_port.md`` says the Lagrange-to-Bernstein matrix has
negative entries and that a bound assuming convexity would be false for it. It does
not. Its columns are the Bernstein basis evaluated at a node in ``[0, 1]``, so they
are non-negative and sum to one, and ``A_e`` is a product of two column-stochastic
matrices.

**The columns are the B-spline basis at the nodes.** The defining property: the
Lagrange basis is cardinal at its own nodes, so ``A_e[:, k] = N_e(x_k)`` with ``x_k``
the ``k``-th node mapped into the interval. Checked against
:meth:`~pantr.bspline.BsplineSpace1D.tabulate_basis`, which is Cox-de Boor and shares
no code with the extraction path.

**It is checked on a dyadic family only, and the restriction is deliberate.** At
degree 1 and 2 with equispaced nodes on dyadic knots, every node, every mapped point
and every intermediate of both routes is a binary rational of a few bits, so nothing
rounds and the bound is **zero**. Above degree 2 the equispaced nodes become thirds
and both routes round, and the bound would have to compose the Cox-de Boor
evaluation error, the error of the ``pow``-seeded ratio recurrence inside ``L``, and
the affine map's rounding amplified by ``max|N'|``. Two of those three are foreign
derivations and one of them is the transcendental category
``design/backend_parity.md``'s open question 2 records as having no vocabulary in the
harness and no source consulted. A bound composed of three unsharp terms would be
satisfied by any result, which is the failure Rule 3 refuses; so the check keeps the
degrees where it is exact and says why it stops.

What the table leaves out
-------------------------

**Degree 0.** :func:`pantr.change_basis.compute_lagrange_to_bernstein_1d` refuses a
degree below 1, so the Lagrange target has no degree-0 case at all. That is
pre-existing behaviour rather than anything this port introduced, and
:func:`test_degree_zero_is_refused_by_both_backends` pins it so the omission is a
recorded fact rather than a gap.

**The vector whose first in-domain knot is repeated.** Kept as a parity case and
excluded from the accuracy oracles, for the reason the Bézier file gives: the shared
algorithm's sliding window misaligns there and both backends are wrong together.

Rule 12
-------

The Bézier half is built from ``+``, ``-``, ``*`` and ``/``, all of which IEEE 754
pins, and the product is :func:`numpy.matmul` on the Python side whatever the JIT
setting. So nothing here needs a gate on the interpreted oracle, and the bitwise
branch is as true under ``NUMBA_DISABLE_JIT=1`` as anywhere.

The tests that state a property of *each* backend take the extension requirement on
the C++ **parameter** rather than on the test, because taking it on the test would
skip the Python half too, and the Python half of the accuracy checks is the only
thing here that would catch the **oracle** regressing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, NamedTuple, cast

import numpy as np
import pytest

from pantr._backend import Backend, use_backend
from pantr.basis import LagrangeVariant
from pantr.basis._basis_lagrange import _get_lagrange_points
from pantr.bspline import BsplineSpace1D
from pantr.bspline._bspline_extraction import (
    _tabulate_Bspline_Bezier_1D_extraction_impl,
    _tabulate_Bspline_Lagrange_1D_extraction_impl,
)
from pantr.bspline._extraction_backend import (
    _KERNELS,
    lagrange_extraction_kernel,
    lagrange_identity_mask_kernel,
)
from pantr.bspline.spanwise_element_extraction import _lagrange_structural_identity_mask
from pantr.change_basis import _cached_lagrange_to_bernstein_matrix
from tests._parity_harness import (
    Field,
    Roundings,
    assert_accuracy,
    assert_object_parity,
    assert_parity,
    bounded_parity,
    contraction_may_fuse,
    demand_cpp_backend,
    demand_the_reference_host,
    derived_accuracy,
    exact_parity,
    unit_roundoff,
)
from tests.parity.test_bspline_bezier_extraction import (
    _CASES as _BEZIER_CASES,
)
from tests.parity.test_bspline_bezier_extraction import (
    _Case,
    _claim,
    _column_sum_bound,
    _draw,
    _insertion_stages,
)

if TYPE_CHECKING:
    from numpy import typing as npt

DTYPES: Final = [np.float64, np.float32]
"""The two storage formats the builder is instantiated for."""

_BACKENDS: Final = (
    pytest.param(Backend.PYTHON, id="python"),
    pytest.param(Backend.CPP, id="cpp"),
)
"""The two backends, for the tests that state a property of each one separately."""

_VARIANTS: Final = list(LagrangeVariant)
"""Every node family, because which of them makes ``L`` the identity is the branch."""

_CASES: Final = tuple(entry for entry in _BEZIER_CASES if entry.degree >= 1)
"""The Bézier table minus its degree-0 vector, which this target refuses.

Imported rather than restated: the two builders are the same builder up to one
matrix product, so a vector worth exercising on one is worth exercising on the other,
and two copies of the table would drift.
"""


def _bindings() -> Any:
    """Import the extension, deferred and in one place.

    Module level would break the extension-skip property: the tests parametrized
    over both backends are meant to run their Python half in an installation with no
    extension, and a top-level ``from pantr import _pantr_cpp`` turns that into a
    collection error for the file -- every test, both halves. Same shape and same
    reason as the Bézier file's.

    Returns:
        Any: The :mod:`pantr._pantr_cpp` module.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    return _pantr_cpp


def _demand_the_extension_if_needed(backend: Backend) -> None:
    """Require the compiled extension, and only for the half that uses it.

    Args:
        backend (Backend): The backend this case runs under.
    """
    if backend is Backend.CPP:
        demand_cpp_backend()


def _matrix_under(
    backend: Backend, degree: int, variant: LagrangeVariant, dtype: npt.DTypeLike
) -> npt.NDArray[Any]:
    """The Lagrange-to-Bernstein matrix one backend builds.

    The one place ``dtype`` is narrowed for the cache's signature. Every caller here
    passes a member of :data:`DTYPES`, so the cast restates what the parametrization
    already guarantees and does not widen anything; it exists because
    :data:`numpy.typing.DTypeLike` is what a pytest parameter can carry.

    Args:
        backend (Backend): Which backend builds it.
        degree (int): Polynomial degree, at least 1.
        variant (LagrangeVariant): The node family.
        dtype (npt.DTypeLike): Storage format, ``float32`` or ``float64``.

    Returns:
        npt.NDArray[Any]: The read-only ``(degree+1, degree+1)`` matrix.
    """
    resolved = cast("np.dtype[np.float32 | np.float64]", np.dtype(dtype))
    with use_backend(backend):
        return np.asarray(_cached_lagrange_to_bernstein_matrix(degree, variant, resolved))


def _matrix(degree: int, variant: LagrangeVariant, dtype: npt.DTypeLike) -> npt.NDArray[Any]:
    """The matrix both backends are handed, so the change of basis is common mode.

    :func:`pantr.change_basis._cached_lagrange_to_bernstein_matrix` is keyed on the
    active backend, so calling it inside each half of a parity comparison would hand
    the two sides different matrices and fold the change of basis into a claim about
    the extraction. Taken once, under Python, and passed to both.

    Args:
        degree (int): Polynomial degree, at least 1.
        variant (LagrangeVariant): The node family.
        dtype (npt.DTypeLike): Storage format.

    Returns:
        npt.NDArray[Any]: The read-only ``(degree+1, degree+1)`` matrix.
    """
    return _matrix_under(Backend.PYTHON, degree, variant, dtype)


def _bezier(case: _Case, dtype: npt.DTypeLike) -> npt.NDArray[Any]:
    """The Bézier operators of one case, under the Python backend.

    Used only to build the amplification: the absolute-value companion of the product
    is ``|C| @ |L|``, and ``C`` is what the contraction actually reads.

    Args:
        case (_Case): The knot vector.
        dtype (npt.DTypeLike): Storage format.

    Returns:
        npt.NDArray[Any]: The ``(n_intervals, degree+1, degree+1)`` Bézier operators.
    """
    knots = np.asarray(case.knots, dtype=dtype)
    space = BsplineSpace1D(knots, case.degree, snap_knots=False)
    with use_backend(Backend.PYTHON):
        return np.asarray(
            _tabulate_Bspline_Bezier_1D_extraction_impl(space.knots, case.degree, space.tolerance)
        )


def _build_kernel(
    case: _Case, dtype: npt.DTypeLike, matrix: npt.NDArray[Any], backend: Backend
) -> npt.NDArray[Any]:
    """Run one backend's Lagrange kernel on a matrix both are given.

    The catalogue accessor rather than the Layer 2 entry point, so the matrix stays
    fixed across the comparison.

    Args:
        case (_Case): The knot vector.
        dtype (npt.DTypeLike): Storage format.
        matrix (npt.NDArray[Any]): The change-of-basis matrix.
        backend (Backend): Which implementation to run.

    Returns:
        npt.NDArray[Any]: The ``(n_intervals, degree+1, degree+1)`` operators.
    """
    knots = np.asarray(case.knots, dtype=dtype)
    space = BsplineSpace1D(knots, case.degree, snap_knots=False)
    out = np.empty((space.num_intervals, case.degree + 1, case.degree + 1), dtype=dtype)
    lagrange_extraction_kernel(backend)(space.knots, case.degree, space.tolerance, matrix, out)
    return out


def _build_layer_2(
    case: _Case, dtype: npt.DTypeLike, variant: LagrangeVariant, backend: Backend
) -> npt.NDArray[Any]:
    """Run one backend's whole Layer 2 path, matrix and all.

    Args:
        case (_Case): The knot vector.
        dtype (npt.DTypeLike): Storage format.
        variant (LagrangeVariant): The node family.
        backend (Backend): Which implementation to run.

    Returns:
        npt.NDArray[Any]: The operators.
    """
    knots = np.asarray(case.knots, dtype=dtype)
    space = BsplineSpace1D(knots, case.degree, snap_knots=False)
    with use_backend(backend):
        return np.asarray(
            _tabulate_Bspline_Lagrange_1D_extraction_impl(
                space.knots, case.degree, space.tolerance, lagrange_variant=variant
            )
        )


def _is_the_identity(matrix: npt.NDArray[Any]) -> bool:
    """Whether a change-of-basis matrix is exactly the identity.

    The same predicate both kernels apply, and the branch the claim turns on.

    Args:
        matrix (npt.NDArray[Any]): The matrix.

    Returns:
        bool: True when every entry matches the identity's bit for bit.
    """
    return bool(np.array_equal(matrix, np.eye(matrix.shape[0], dtype=matrix.dtype)))


def _companion(bezier: npt.NDArray[Any], matrix: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """The absolute-value companion of the product, elementwise.

    ``design/backend_parity.md`` Rule 10 prescribes running the kernel on absolute
    values rather than taking ``max|M| max|v|``, and this is that: the magnitude
    reachable at each output element of ``C @ L``. Both factors happen to be
    non-negative here, so it coincides with the answer, but it is formed from
    absolute values so that a future non-negative-losing change does not silently
    invalidate it.

    Args:
        bezier (npt.NDArray[Any]): The Bézier operators.
        matrix (npt.NDArray[Any]): The change-of-basis matrix.

    Returns:
        npt.NDArray[np.float64]: The companion, shaped like the operators.
    """
    return np.asarray(
        np.matmul(
            np.abs(bezier.astype(np.float64)),
            np.abs(matrix.astype(np.float64)),
        ),
        dtype=np.float64,
    )


def _column_sums(operators: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Sum each operator's columns, in ``float64`` whatever the operators are.

    The cast comes **before** the sum, and that is the whole point: ``ndarray.sum``
    accumulates in the array's own dtype, so summing a ``float32`` operator and then
    casting would put ``gamma^{float32}_{degree}`` of this test's own arithmetic into
    a quantity meant to measure the operator's. Casting first is exact and leaves only
    ``gamma^{float64}_{degree}``, which is what :func:`_column_sum_tolerance` charges.

    One spelling for all three call sites, so the bound and the measurement cannot
    disagree about which arithmetic the sum lives in -- ``design/backend_parity.md``
    Rule 5's "name the arithmetic", applied to a test rather than to a kernel.

    Args:
        operators (npt.NDArray[Any]): The ``(n_intervals, degree+1, degree+1)``
            operators.

    Returns:
        npt.NDArray[np.float64]: One column sum per ``(interval, column)``.
    """
    return np.asarray(operators.astype(np.float64).sum(axis=1), dtype=np.float64)


def _gamma(roundings: int, dtype: npt.DTypeLike) -> float:
    """Higham's ``gamma_m = m u / (1 - m u)`` for a given rounding count.

    The closed form rather than the truncation ``m u``, and the same one
    :func:`tests._parity_harness.absolute_tolerance` applies internally. Recomputed
    here only where an **absolute** term has to be converted into the amplification
    the harness multiplies by ``gamma``; ``tests/parity/test_change_basis.py`` does
    the same conversion for the same reason.

    Args:
        roundings (int): The accumulated rounding count ``m``.
        dtype (npt.DTypeLike): The format the roundings happen in.

    Returns:
        float: ``gamma_m``.

    Raises:
        AssertionError: If the budget runs away to one, where gamma bounds nothing.
    """
    u = unit_roundoff(dtype)
    total = roundings * u
    if total >= 1.0:
        raise AssertionError(f"gamma runs away to one at {roundings} roundings and u={u:.3e}")
    return total / (1.0 - total)


def _product_claim(
    case: _Case,
    dtype: npt.DTypeLike,
    matrix: npt.NDArray[Any],
    bezier: npt.NDArray[Any],
    extra_absolute: npt.NDArray[np.float64] | None = None,
) -> Any:
    """The parity claim for the Lagrange operators of one case.

    Three branches, in order of strength.

    **An identity change of basis** contracts ``C[i,k] * 0`` and ``C[i,j] * 1`` only.
    Every product is exact, every partial sum adds an exact zero to an exact value,
    and no summation order can separate the two backends. So the claim is the Bézier
    one, imported unchanged from that file: bitwise, or Rule 10's fused budget on a
    build whose insertions may contract.

    **Otherwise, on a build with no fused multiply-add**, the Bézier halves agree bit
    for bit, so both contractions read the *same* ``C`` and the whole difference is
    the summation order of a length-``degree + 1`` dot product. Higham, *Accuracy and
    Stability of Numerical Algorithms*, 2nd ed., SIAM 2002, section 3.1: for any
    summation order ``|fl(x^T y) - x^T y| <= gamma_n |x|^T |y|`` with ``n`` the
    length, and pp. 62-64 note that blocking only tightens it. So the budget is
    ``Roundings(degree + 1, 1, 0)`` -- one accumulator rounding per term, and no
    storage rounding because the accumulator **is** the storage format on both sides
    (``design/backend_parity.md`` Rule 9, and ``sgemm`` accumulates in ``float32``).
    The harness doubles it, which is the step that turns each side's one-sided
    forward-error bound into a bound on their difference.

    **On a fusing build** the Bézier halves may differ too, by Rule 10's budget of
    three accumulator roundings over the insertion chain, amplified by one since
    every Bézier entry lies in ``[0, 1]``. That difference reaches an output element
    weighted by ``sum_k |L[k,j]|``, and the contraction itself may fuse, so its own
    budget rises from one rounding per term to three. Both terms are folded into a
    single claim by monotonicity of ``gamma``: charging the **longer** of the two
    chains to both and adding the two amplifications is an upper bound on charging
    each its own. The max rather than the sum, because the max is already enough and
    is the tighter of the two.

    **This branch is not reasoned about, it is run.** ``design/backend_parity.md``
    Rule 11 records the sibling Bézier port shipping a fused branch that called
    :func:`bounded_parity` with an argument it does not take, green over 133 tests
    because nothing reached it. So this one was exercised against an extension built
    at ``-march=native``, where ``contraction_may_fuse()`` is true, and the result is
    recorded in the PR that added it rather than assumed here.

    Args:
        case (_Case): The knot vector.
        dtype (npt.DTypeLike): Storage format.
        matrix (npt.NDArray[Any]): The change-of-basis matrix both backends read.
        bezier (npt.NDArray[Any]): The Bézier operators, for the companion.
        extra_absolute (npt.NDArray[np.float64] | None): An absolute term to add,
            for the end-to-end case where the two backends' matrices differ.
            Defaults to None.

    Returns:
        Any: The parity claim.
    """
    if _is_the_identity(matrix) and extra_absolute is None:
        return _claim(case, dtype)

    terms = case.degree + 1
    per_stage = 1
    amplification = _companion(bezier, matrix)
    why_head = (
        "the contraction is a dot product of degree + 1 terms, summed by a BLAS gemm on the "
        "oracle's side and by an ascending loop on the C++ side. Higham, Accuracy and "
        "Stability of Numerical Algorithms, 2nd ed., section 3.1: any summation order is "
        "within gamma_n |x|^T |y| of the exact product, and blocking only tightens it "
        "(pp. 62-64). The accumulator is the storage format on both sides -- Rule 9, and "
        "sgemm accumulates in float32 -- so nothing narrows on the store and "
        "storage_per_stage is zero. The amplification is the absolute-value companion "
        "|C| @ |L|, elementwise, which Rule 10 prescribes over max|M| max|v|. The "
        "hypothesis this rests on: the Bezier halves agree bit for bit, which "
        "tests/parity/test_bspline_bezier_extraction.py claims and which "
        "contraction_may_fuse() is the one condition known to break, so the whole "
        "difference here is the contraction's"
    )
    why_tail = ""

    if contraction_may_fuse():
        # The longer of the two chains, charged to both, rather than their sum: gamma is
        # monotone, so gamma_{3S} and gamma_{3n} are each at most gamma_{3 max(S, n)},
        # and the max is the tighter of the two valid choices.
        terms = max(terms, _insertion_stages(case, dtype))
        per_stage = 3
        column = np.abs(matrix.astype(np.float64)).sum(axis=0)
        amplification = amplification + np.broadcast_to(column, amplification.shape)
        why_tail += (
            ". This build's target ISA has a fused multiply-add, so both stages may "
            "contract and Rule 10's three accumulator roundings per fused site apply to "
            "each. The Bezier halves may then differ by gamma over the insertion chain, "
            "amplified by one since every Bezier entry lies in [0, 1], and that reaches an "
            "output element weighted by sum_k |L[k,j]|, which is the second amplification "
            "term. The two stages are folded into one claim by charging the longer of the "
            "two chains to both and adding their amplifications, which is an upper bound "
            "because gamma is monotone. The companion is formed from the oracle's Bezier "
            "operator rather than the exact one; the difference is second order in u and "
            "is absorbed by the budget"
        )

    if extra_absolute is not None:
        amplification = amplification + extra_absolute / (2.0 * _gamma(terms * per_stage, dtype))
        why_tail += (
            ". The two backends are handed different change-of-basis matrices here, and the "
            "measured gap between them propagates as sum_k |C[i,k]| |dL[k,j]| <= gap times "
            "the row sum of C. That is an absolute term, so it is divided by the claim's own "
            "gamma and by the harness's factor of two before being added to the "
            "amplification, which is what turns it back into the tolerance it started as"
        )

    return bounded_parity(
        roundings=Roundings(stages=terms, accumulator_per_stage=per_stage, storage_per_stage=0),
        accumulator=dtype,
        storage=dtype,
        amplification=amplification,
        why=why_head + why_tail,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("variant", _VARIANTS, ids=[v.name for v in _VARIANTS])
@pytest.mark.parametrize("case", _CASES, ids=[entry.label for entry in _CASES])
def test_the_operators_agree(
    cpp_backend: None, case: _Case, variant: LagrangeVariant, dtype: npt.DTypeLike
) -> None:
    """The two backends build the same operators from the same matrix.

    Args:
        cpp_backend (None): Requires the compiled extension.
        case (_Case): The knot vector.
        variant (LagrangeVariant): The node family.
        dtype (npt.DTypeLike): Storage format.
    """
    matrix = _matrix(case.degree, variant, dtype)
    reference = _build_kernel(case, dtype, matrix, Backend.PYTHON)
    actual = _build_kernel(case, dtype, matrix, Backend.CPP)
    assert_parity(
        actual,
        reference,
        _product_claim(case, dtype, matrix, _bezier(case, dtype)),
        context=f"{case.label} {variant.name} in {np.dtype(dtype).name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("variant", _VARIANTS, ids=[v.name for v in _VARIANTS])
@pytest.mark.parametrize("case", _CASES, ids=[entry.label for entry in _CASES])
def test_the_layer_2_path_agrees(
    cpp_backend: None, case: _Case, variant: LagrangeVariant, dtype: npt.DTypeLike
) -> None:
    """End to end, where each backend also builds its own change-of-basis matrix.

    The companion to :func:`test_the_operators_agree`, which holds the matrix fixed.
    Here the matrix is whatever :mod:`pantr.change_basis` hands each backend, so the
    difference between the two matrices is measured and carried through the product
    rather than assumed away. Measured on this machine it is zero everywhere except
    ``float32`` at the second-kind Chebyshev family, whose nodes dispatch.

    Args:
        cpp_backend (None): Requires the compiled extension.
        case (_Case): The knot vector.
        variant (LagrangeVariant): The node family.
        dtype (npt.DTypeLike): Storage format.
    """
    matrix_py = _matrix_under(Backend.PYTHON, case.degree, variant, dtype)
    matrix_cpp = _matrix_under(Backend.CPP, case.degree, variant, dtype)
    gap = float(
        np.abs(matrix_cpp.astype(np.float64) - matrix_py.astype(np.float64)).max(initial=0.0)
    )

    bezier = _bezier(case, dtype)
    extra = None
    if gap > 0.0:
        # |dA[i,j]| = |sum_k C[i,k] dL[k,j]| <= gap * sum_k |C[i,k]|, broadcast over j.
        row_sums = np.abs(bezier.astype(np.float64)).sum(axis=2)
        extra = np.broadcast_to(gap * row_sums[:, :, None], bezier.shape).copy()

    reference = _build_layer_2(case, dtype, variant, Backend.PYTHON)
    actual = _build_layer_2(case, dtype, variant, Backend.CPP)
    assert_parity(
        actual,
        reference,
        _product_claim(case, dtype, matrix_py, bezier, extra_absolute=extra),
        context=f"{case.label} {variant.name} in {np.dtype(dtype).name} (matrix gap {gap:.3e})",
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("variant", _VARIANTS, ids=[v.name for v in _VARIANTS])
@pytest.mark.parametrize("case", _CASES, ids=[entry.label for entry in _CASES])
def test_the_identity_mask_agrees(
    cpp_backend: None, case: _Case, variant: LagrangeVariant, dtype: npt.DTypeLike
) -> None:
    """The two backends mark the same elements as already-Lagrange.

    A boolean verdict per element, so the claim is exactness rather than a tolerance:
    ``design/backend_parity.md`` Rule 11's distinction. Both sides reach it by the
    same exact comparison of the matrix against the identity followed by the same
    integer comparisons of the multiplicities.

    Args:
        cpp_backend (None): Requires the compiled extension.
        case (_Case): The knot vector.
        variant (LagrangeVariant): The node family.
        dtype (npt.DTypeLike): Storage format.
    """
    space = BsplineSpace1D(np.asarray(case.knots, dtype=dtype), case.degree, snap_knots=False)
    with use_backend(Backend.PYTHON):
        reference = _lagrange_structural_identity_mask(space, variant)
    with use_backend(Backend.CPP):
        actual = _lagrange_structural_identity_mask(space, variant)
    assert_object_parity(
        py=reference,
        cpp=actual,
        fields=[
            Field(
                name="identity mask",
                claim=exact_parity(
                    why=(
                        "the mask is an exact comparison of the change-of-basis matrix "
                        "against the identity, and then two integer comparisons of a knot "
                        "multiplicity against degree + 1. Both sides reach the same boolean "
                        "by the same exact arithmetic, so a difference is a defect rather "
                        "than a rounding"
                    )
                ),
                read=lambda mask: mask,
            )
        ],
        context=f"{case.label} {variant.name} in {np.dtype(dtype).name}",
    )


# ---------------------------------------------------------------------------
# The three independent accuracy oracles
# ---------------------------------------------------------------------------


class _Exact(NamedTuple):
    """One knot vector whose Lagrange operator entries are exact binary rationals.

    Attributes:
        label (str): What the case is.
        knots (list[float]): The knot vector.
        degree (int): The polynomial degree.
        operators (list[list[list[float]]]): The exact operators, one per element.
    """

    label: str
    knots: list[float]
    degree: int
    operators: list[list[list[float]]]


_EXACT_CASES: Final = (
    # `C @ L` with the Bézier tables of `test_bspline_bezier_extraction._EXACT_CASES`
    # and `L = [[1, 1/4, 0], [0, 1/2, 0], [0, 1/4, 1]]`, the equispaced degree-2
    # Lagrange-to-Bernstein matrix. Worked out entry by entry; the C++ file carries
    # the same tables and derives the Bézier half from `N = C @ B`.
    _Exact(
        "quadratic open, three uniform elements",
        [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0],
        2,
        [
            [[1.0, 0.25, 0.0], [0.0, 0.625, 0.5], [0.0, 0.125, 0.5]],
            [[0.5, 0.125, 0.0], [0.5, 0.75, 0.5], [0.0, 0.125, 0.5]],
            [[0.5, 0.125, 0.0], [0.5, 0.625, 0.0], [0.0, 0.25, 1.0]],
        ],
    ),
    # Both operators of the unclamped uniform vector are the clamped case's middle
    # one, because an interior element's operator depends only on the local knot
    # pattern. Post-multiplying by the same matrix preserves that.
    _Exact(
        "quadratic unclamped, two uniform elements",
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        2,
        [
            [[0.5, 0.125, 0.0], [0.5, 0.75, 0.5], [0.0, 0.125, 0.5]],
            [[0.5, 0.125, 0.0], [0.5, 0.75, 0.5], [0.0, 0.125, 0.5]],
        ],
    ),
    # Degree 1: the Bézier operator is the identity at any knots, and the equispaced
    # degree-1 matrix is the identity too, so the product is.
    _Exact(
        "linear, three elements",
        [0.0, 0.0, 1.0, 2.0, 3.0, 3.0],
        1,
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
    ),
)
"""Knot vectors whose equispaced Lagrange operators are exactly representable."""


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("case", _EXACT_CASES, ids=[entry.label for entry in _EXACT_CASES])
def test_matches_the_exact_rational_operators(
    case: _Exact, backend: Backend, dtype: npt.DTypeLike
) -> None:
    """Each backend reproduces the hand-derived exact table, with a zero bound.

    Args:
        case (_Exact): The knot vector and its exact operators.
        backend (Backend): Which implementation to run.
        dtype (npt.DTypeLike): Storage format.
    """
    _demand_the_extension_if_needed(backend)
    computed = _build_layer_2(
        _Case(case.label, case.knots, case.degree, True),
        dtype,
        LagrangeVariant.EQUISPACES,
        backend,
    )
    exact = np.asarray(case.operators, dtype=dtype)
    assert computed.shape == exact.shape, f"{case.label}: shape {computed.shape} vs {exact.shape}"
    assert_accuracy(
        computed,
        exact,
        derived_accuracy(
            bound=np.zeros_like(exact, dtype=np.float64),
            why=(
                "the Bezier entries of this vector are halves at degree 2 and zeros and "
                "ones at degree 1, and the equispaced Lagrange-to-Bernstein matrix is "
                "[[1, 1/4, 0], [0, 1/2, 0], [0, 1/4, 1]] at degree 2 and the identity at "
                "degree 1. So every product and every partial sum of the contraction is a "
                "binary rational of a few bits, exactly representable in both storage "
                "formats. No rounding occurs, so the bound is zero rather than derived "
                "from one"
            ),
        ),
        context=f"{case.label} in {np.dtype(dtype).name} on {backend.name}",
    )


def test_the_exact_tables_are_not_trivial() -> None:
    """A zero bound says something only if the values it guards are not all forced.

    The Bézier builder starts from the identity, so a table that is the identity
    everywhere would pass against a builder that did nothing at all and then
    multiplied by nothing. This pins that the tables carry entries strictly between
    zero and one, that they are not all the identity, that transposing one changes
    it, and -- what separates them from the Bézier file's -- that they are not the
    Bézier tables either, so an implementation that skipped the product would fail.
    """
    interesting = [
        np.asarray(case.operators, dtype=np.float64) for case in _EXACT_CASES if case.degree >= 2
    ]
    assert interesting, "no exact case has an answer that is not forced"
    for table in interesting:
        strictly_inside = (table > 0.0) & (table < 1.0)
        assert strictly_inside.any(), "an exact table is entirely zeros and ones"
        identity = np.broadcast_to(np.eye(table.shape[1]), table.shape)
        assert not np.array_equal(table, identity), "an exact table is the identity everywhere"
        assert not np.array_equal(table, np.swapaxes(table, 1, 2)), (
            "an exact table is symmetric, so comparing against it could not tell a "
            "transposed operator from the right one"
        )

    quadratic = next(case for case in _EXACT_CASES if case.degree == 2 and "open" in case.label)
    bezier = _bezier(_Case(quadratic.label, quadratic.knots, 2, True), np.float64)
    assert not np.allclose(np.asarray(quadratic.operators), bezier), (
        "the exact Lagrange table equals the Bezier one, so a builder that forgot the "
        "change of basis entirely would pass against it"
    )


def _column_sum_tolerance(
    case: _Case, dtype: npt.DTypeLike, matrix: npt.NDArray[Any], companion: npt.NDArray[Any]
) -> npt.NDArray[np.float64]:
    """How far a Lagrange operator's column sum may sit from the matrix's own.

    The invariant is ``sum_i A[i,j] = sum_k (sum_i C[i,k]) L[k,j] = sum_k L[k,j]``.
    Four terms, each named because each is a different mechanism:

    * **The Bézier column defect.** ``|sum_i C[i,k] - 1|`` is what the Bézier file's
      :func:`~tests.parity.test_bspline_bezier_extraction._column_sum_bound` bounds,
      and it is imported rather than re-derived. It reaches column ``j`` weighted by
      ``sum_k |L[k,j]|``. Using it here is a slight over-estimate, since it also
      charges the ``degree`` roundings of a summation this term does not perform.
    * **The contraction.** Each ``A[i,j]`` is within ``gamma_{degree+1}`` of the exact
      product of the stored operands, times the companion; summed over ``i`` that is
      ``gamma_{degree+1}`` times the companion's own column sum.
    * **The two sums this test forms.** Both run in ``float64`` over ``degree + 1``
      values and each costs ``gamma^{float64}_{degree}`` times its own absolute column
      sum. That both are ``float64`` is a property of :func:`_column_sums` and of the
      ``astype`` in front of the matrix's sum, not something to assume:
      :meth:`numpy.ndarray.sum` accumulates in the array's own dtype, so a cast placed
      *after* the sum would leave a ``float32`` accumulation charged at ``float64``
      rates and understate this term by eight orders of magnitude. The contraction
      term above happens to dominate the shortfall for every degree, by monotonicity
      of ``gamma``, which is exactly what would have kept the error invisible.

    No term is a fitted constant and none is a truncation: ``gamma`` is the closed
    form throughout, because the Bézier chain grows with the element count.

    **Stated hypothesis: no underflow in the entries.** The rounding model is purely
    relative, and the Bézier file records that subnormal ``float32`` entries are
    reachable with mixed per-gap knot ratios. The bound was observed to hold there by
    a wide margin; observing is not covering.

    Args:
        case (_Case): The knot vector, which fixes the Bézier chain length.
        dtype (npt.DTypeLike): Storage format.
        matrix (npt.NDArray[Any]): The change-of-basis matrix.
        companion (npt.NDArray[Any]): ``|C| @ |L|``, elementwise.

    Returns:
        npt.NDArray[np.float64]: One bound per ``(interval, column)``.
    """
    column_of_matrix = np.abs(matrix.astype(np.float64)).sum(axis=0)
    column_of_companion = companion.sum(axis=1)

    bezier_defect = _column_sum_bound(case, dtype) * column_of_matrix
    contraction = _gamma(case.degree + 1, dtype) * column_of_companion
    outer = _gamma(case.degree, np.float64) * (column_of_companion + column_of_matrix)
    return np.asarray(bezier_defect + contraction + outer, dtype=np.float64)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("variant", _VARIANTS, ids=[v.name for v in _VARIANTS])
@pytest.mark.parametrize(
    "case",
    [entry for entry in _CASES if entry.accuracy],
    ids=[e.label for e in _CASES if e.accuracy],
)
def test_the_columns_sum_to_the_matrix_columns(
    case: _Case, variant: LagrangeVariant, backend: Backend, dtype: npt.DTypeLike
) -> None:
    """Every operator's columns sum to the change-of-basis matrix's, in either backend.

    The analytic oracle, and the one that catches a transposition: the Bézier column
    sums are one, so ``sum_i A[i,j] = sum_k L[k,j]``, while the row sums are not one
    on either factor.

    It also pins non-negativity, which ``design/extraction_port.md`` denied for this
    target: every Lagrange node lies in ``[0, 1]`` where the Bernstein basis is
    non-negative, so ``L`` is entrywise non-negative and so is the product.

    Args:
        case (_Case): The knot vector.
        variant (LagrangeVariant): The node family.
        backend (Backend): Which implementation to run.
        dtype (npt.DTypeLike): Storage format.
    """
    _demand_the_extension_if_needed(backend)
    matrix = _matrix(case.degree, variant, dtype)
    operators = _build_kernel(case, dtype, matrix, backend)
    companion = _companion(_bezier(case, dtype), matrix)

    assert matrix.min() >= 0.0, (
        f"{variant.name} degree {case.degree}: the change-of-basis matrix has a negative "
        "entry, so the columns of the product are not a convex combination and the "
        "non-negativity claim below does not follow"
    )
    assert operators.min() >= 0.0, (
        "a Lagrange operator entry is negative, though both factors are non-negative"
    )

    column_sums = _column_sums(operators)
    target = np.broadcast_to(np.abs(matrix.astype(np.float64)).sum(axis=0), column_sums.shape)
    assert_accuracy(
        column_sums,
        target,
        derived_accuracy(
            bound=np.broadcast_to(
                _column_sum_tolerance(case, dtype, matrix, companion), column_sums.shape
            ).copy(),
            why=(
                "sum_i A[i,j] = sum_k (sum_i C[i,k]) L[k,j] = sum_k L[k,j], using the Bezier "
                "column sums. Compared against the matrix's own column sum rather than "
                "against one, so its tabulation error is folded in exactly instead of "
                "bounded. Four terms: the Bezier column defect weighted by sum_k |L[k,j]|, "
                "the contraction's gamma_{degree+1} times the companion's column sum, and "
                "the two float64 sums this test forms. Hypothesis: no underflow in the "
                "entries"
            ),
        ),
        context=f"{case.label} {variant.name} in {np.dtype(dtype).name} on {backend.name}",
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_column_sum_check_is_not_vacuous(dtype: npt.DTypeLike) -> None:
    """The column-sum bound is compared against a nonzero error somewhere.

    ``design/backend_parity.md``'s rule the hard way: a bound compared only against
    zero has not been checked. A dyadic vector with an equispaced matrix rounds
    nothing; a Gauss-Legendre matrix at a non-uniform vector does.

    Args:
        dtype (npt.DTypeLike): Storage format.
    """
    worst_ratio = 0.0
    worst = 0.0
    for case in _CASES:
        if not case.accuracy:
            continue
        for variant in _VARIANTS:
            matrix = _matrix(case.degree, variant, dtype)
            operators = _build_kernel(case, dtype, matrix, Backend.PYTHON)
            companion = _companion(_bezier(case, dtype), matrix)
            target = np.abs(matrix.astype(np.float64)).sum(axis=0)
            deviation = np.abs(_column_sums(operators) - target)
            bound = _column_sum_tolerance(case, dtype, matrix, companion)
            assert np.all(deviation <= bound), f"{case.label} {variant.name}: bound exceeded"
            worst = max(worst, float(deviation.max(initial=0.0)))
            worst_ratio = max(worst_ratio, float((deviation / bound).max(initial=0.0)))
    assert worst > 0.0, (
        "every case in the table has exactly-summing columns, so the derived bound is "
        "only ever compared against zero and asserts nothing"
    )
    assert worst_ratio > 1e-3, (
        f"the worst observed deviation reaches only {worst_ratio:.2e} of the bound, which "
        "is far enough below it that the check would pass against a materially wrong answer"
    )


class _Dyadic(NamedTuple):
    """A knot vector on which the node-tabulation check is exact.

    Attributes:
        label (str): What the case is.
        knots (list[float]): The knot vector, dyadic throughout.
        degree (int): The polynomial degree, 1 or 2, so the equispaced nodes are
            dyadic too.
    """

    label: str
    knots: list[float]
    degree: int


_DYADIC_CASES: Final = (
    _Dyadic("quadratic uniform open", [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0], 2),
    _Dyadic("quadratic unclamped uniform", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], 2),
    _Dyadic("interior knot of multiplicity two", [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0], 2),
    _Dyadic("linear, three elements", [0.0, 0.0, 1.0, 2.0, 3.0, 3.0], 1),
)
"""Vectors where every node, mapped point and intermediate is a short binary rational."""


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("case", _DYADIC_CASES, ids=[entry.label for entry in _DYADIC_CASES])
def test_the_columns_are_the_basis_at_the_nodes(
    case: _Dyadic, backend: Backend, dtype: npt.DTypeLike
) -> None:
    """Column ``k`` of an operator is the B-spline basis at the ``k``-th node.

    The defining property of the target, checked against Cox-de Boor, which shares no
    code with the extraction path. The right endpoint of each interval is skipped:
    :meth:`~pantr.bspline.BsplineSpace1D.tabulate_basis` resolves a point on a knot to
    the interval starting there, so at ``xi = 1`` it reports the *next* interval's
    window and the two vectors index different basis functions. The alignment of every
    point that is checked is asserted rather than assumed.

    The bound is zero, and the module docstring says why this family and no larger one.

    Args:
        case (_Dyadic): The knot vector.
        backend (Backend): Which implementation to run.
        dtype (npt.DTypeLike): Storage format.
    """
    _demand_the_extension_if_needed(backend)
    knots = np.asarray(case.knots, dtype=dtype)
    space = BsplineSpace1D(knots, case.degree, snap_knots=False)
    operators = _build_layer_2(
        _Case(case.label, case.knots, case.degree, True),
        dtype,
        LagrangeVariant.EQUISPACES,
        backend,
    )
    breaks = space.get_unique_knots_and_multiplicity(in_domain=True)[0]
    first = space.first_basis_per_interval()
    nodes = np.arange(case.degree + 1, dtype=dtype) / np.dtype(dtype).type(case.degree)

    checked = 0
    for interval in range(space.num_intervals):
        left, right = breaks[interval], breaks[interval + 1]
        for k in range(case.degree + 1):
            if float(nodes[k]) >= 1.0:
                continue
            point = left + nodes[k] * (right - left)
            values, window = space.tabulate_basis(np.asarray([point], dtype=dtype))
            assert int(window[0]) == int(first[interval]), (
                f"{case.label}: the point at node {k} of interval {interval} landed in "
                f"another interval's basis window, so the comparison would be misaligned"
            )
            assert_accuracy(
                np.asarray(values[0]),
                np.asarray(operators[interval, :, k]),
                derived_accuracy(
                    bound=np.zeros(case.degree + 1, dtype=np.float64),
                    why=(
                        "every knot, every equispaced node at degree 1 or 2, the mapped "
                        "point and every intermediate of both routes is a binary rational "
                        "of a few bits, so neither the Cox-de Boor recurrence nor the "
                        "extraction rounds and the two agree exactly. Above degree 2 the "
                        "nodes become thirds and this stops holding, which is why the "
                        "family is what it is"
                    ),
                ),
                context=(
                    f"{case.label} interval {interval} node {k} in "
                    f"{np.dtype(dtype).name} on {backend.name}"
                ),
            )
            checked += 1
    assert checked >= case.degree * space.num_intervals, (
        f"{case.label}: only {checked} node comparisons ran, so the check is thinner than "
        "the table implies"
    )


def test_the_node_tabulation_check_is_not_trivial() -> None:
    """The vectors it compares are not forced to agree by being all zeros and ones.

    A degree-1 operator is the identity, so its columns are unit vectors and the
    comparison would hold against almost anything. This pins that the quadratic cases
    put values strictly inside ``(0, 1)`` in front of it, and that the columns of one
    operator are not all equal -- which is what makes a permuted node order visible.
    """
    interesting = 0
    for case in _DYADIC_CASES:
        if case.degree < 2:
            continue
        operators = _build_layer_2(
            _Case(case.label, case.knots, case.degree, True),
            np.float64,
            LagrangeVariant.EQUISPACES,
            Backend.PYTHON,
        )
        assert ((operators > 0.0) & (operators < 1.0)).any(), (
            f"{case.label}: every entry is a zero or a one"
        )
        distinct = {tuple(operators[0, :, k]) for k in range(case.degree + 1)}
        assert len(distinct) == case.degree + 1, (
            f"{case.label}: two columns of the first operator are equal, so a permuted "
            "node order would be invisible here"
        )
        interesting += 1
    assert interesting > 0, "no dyadic case has an answer that is not forced"


# ---------------------------------------------------------------------------
# What the port does not do, and what the binding refuses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_degree_zero_is_refused_by_both_backends(backend: Backend) -> None:
    """The Lagrange target has no degree-0 case, and that predates this port.

    :func:`pantr.change_basis.compute_lagrange_to_bernstein_1d` refuses a degree
    below 1, and Layer 2 asks it for the matrix before reaching either kernel. Pinned
    so that the absence of a degree-0 row in every table above is a recorded fact
    rather than an untested gap, and so that a change to that refusal shows up here.

    Args:
        backend (Backend): Which implementation to run.
    """
    _demand_the_extension_if_needed(backend)
    knots = np.asarray([0.0, 1.0], dtype=np.float64)
    space = BsplineSpace1D(knots, 0, snap_knots=False)
    with use_backend(backend), pytest.raises(ValueError, match="[Dd]egree must at least 1"):
        space.tabulate_Lagrange_extraction_operators()

    # The mask does have a degree-0 answer, because Layer 2 short-circuits before
    # asking for a matrix. Both backends take that branch, so it never reaches a
    # kernel and there is nothing to dispatch.
    with use_backend(backend):
        mask = _lagrange_structural_identity_mask(space, LagrangeVariant.EQUISPACES)
    assert mask.tolist() == [True]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("variant", _VARIANTS, ids=[v.name for v in _VARIANTS])
def test_the_oracles_in_place_product_matches_a_non_aliased_one(
    variant: LagrangeVariant, dtype: npt.DTypeLike
) -> None:
    """The Python kernel's ``np.matmul(out, L, out=out)`` is not corrupted by aliasing.

    The oracle forms the product in place, with ``out`` as both the first operand and
    the destination. That is safe only because :func:`numpy.matmul` detects the overlap
    and buffers -- a claim about a third party, across a numpy range this project's own
    ``CLAUDE.md`` records as behaving differently between local and CI. So it is checked
    rather than asserted in a comment: the in-place answer must equal the one computed
    into a fresh array, bit for bit. A numpy that stopped buffering would corrupt the
    later rows from the earlier ones and fail here.

    The call is pre-existing -- the port relocated it, it did not introduce it -- so this
    also guards the oracle rather than only the port.

    Args:
        variant (LagrangeVariant): The node family.
        dtype (npt.DTypeLike): Storage format.
    """
    for case in _CASES:
        matrix = _matrix(case.degree, variant, dtype)
        bezier = _bezier(case, dtype)
        aliased = bezier.copy()
        np.matmul(aliased, matrix, out=aliased)
        separate = np.matmul(bezier, matrix)
        assert aliased.tobytes() == separate.tobytes(), (
            f"{case.label} {variant.name} in {np.dtype(dtype).name}: the in-place matmul "
            "differs from the non-aliased one, so numpy is no longer buffering the "
            "overlap the oracle relies on"
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("variant", _VARIANTS, ids=[v.name for v in _VARIANTS])
def test_every_node_family_is_ascending_inside_the_unit_interval(
    variant: LagrangeVariant, dtype: npt.DTypeLike
) -> None:
    """The hypothesis both structural claims rest on, checked rather than reasoned.

    Two claims in this port need it and neither could survive without it. That the
    Lagrange-to-Bernstein matrix is entrywise non-negative and column-stochastic follows
    from the nodes lying in ``[0, 1]``, where the Bernstein basis is; that is what makes
    the Lagrange operator column-stochastic and the amplification tight. And the
    soundness of the mask's all-false branch -- written out beside
    ``lagrange_structural_identity_mask`` in ``cpp/include/pantr/bspline/extraction.hpp``
    -- ends by ruling out the reversal permutation, which needs the nodes to be
    **ascending** rather than merely inside the interval.

    Swept past the degrees any table here uses, since a family that reordered or
    overshot at high degree would break both claims silently.

    Args:
        variant (LagrangeVariant): The node family.
        dtype (npt.DTypeLike): Storage format.
    """
    for n_pts in range(2, 40):
        nodes = _get_lagrange_points(variant, n_pts, np.dtype(dtype))
        assert np.all(np.diff(nodes) > 0), (
            f"{variant.name} at {n_pts} nodes is not strictly ascending, so the mask's "
            "all-false branch is no longer sound"
        )
        assert nodes.min() >= 0.0 and nodes.max() <= 1.0, (
            f"{variant.name} at {n_pts} nodes leaves [0, 1], so the Bernstein basis is "
            "not non-negative there and neither the amplification nor the "
            "column-stochasticity claim follows"
        )


def test_the_claim_is_not_vacuous(cpp_backend: None) -> None:
    """A parity claim asserts nothing unless the two paths could have differed.

    The specific risk: if the catalogue handed back the same callable for both
    backends, every assertion above would pass for the wrong reason. Identity is
    taken against the catalogue rather than against a kernel's ``repr`` or
    ``py_func``, neither of which exists under ``NUMBA_DISABLE_JIT=1``.
    """
    python_builder = lagrange_extraction_kernel(Backend.PYTHON)
    cpp_builder = lagrange_extraction_kernel(Backend.CPP)
    assert python_builder is not cpp_builder
    python_mask = lagrange_identity_mask_kernel(Backend.PYTHON)
    cpp_mask = lagrange_identity_mask_kernel(Backend.CPP)
    assert python_mask is not cpp_mask

    # The apply catalogue is the sibling table; the builders must not have landed in
    # it, since the two are separate C++ registrations with separate claims.
    assert python_builder not in _KERNELS.values()
    assert cpp_builder not in _KERNELS.values()


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_two_backends_still_differ_somewhere(cpp_backend: None, dtype: npt.DTypeLike) -> None:
    """The bounded claim is compared against a nonzero difference somewhere.

    ``design/backend_parity.md`` Rule 7: a bound nothing approaches can rot without a
    test noticing, so the guard asserts the opposite of the bound -- but it is a fact
    about the *host*, not about this library, since which summation order the BLAS
    picks depends on the build and the processor. Enforced only where it was
    measured, and reported with a reason anywhere else.

    Args:
        cpp_backend (None): Requires the compiled extension.
        dtype (npt.DTypeLike): Storage format.
    """
    worst = 0.0
    for case in _CASES:
        for variant in _VARIANTS:
            matrix = _matrix(case.degree, variant, dtype)
            reference = _build_kernel(case, dtype, matrix, Backend.PYTHON)
            actual = _build_kernel(case, dtype, matrix, Backend.CPP)
            worst = max(
                worst,
                float(
                    np.abs(actual.astype(np.float64) - reference.astype(np.float64)).max(
                        initial=0.0
                    )
                ),
            )
    demand_the_reference_host(
        guard="the two backends still disagree on at least one Lagrange operator",
        measured=(
            "one unit of roundoff at degree 3 and above, in both storage formats, and "
            "exactly zero below. Which arguments a BLAS gemm and an ascending loop round "
            "differently is a property of the build and the processor, so this is not "
            "enforced off the machine it was measured on"
        ),
    )
    assert worst > 0.0, (
        "the two backends agree bit for bit on every case in the table, so the bounded "
        "claim is only ever compared against zero. Either the dispatch collapsed onto one "
        "implementation or the bound should be strengthened to bitwise"
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_a_strided_out_is_filled(backend: Backend, dtype: npt.DTypeLike) -> None:
    """A caller's non-contiguous ``out`` comes back filled, under either backend.

    The binding declares ``out`` C-contiguous, so the adapter in
    :mod:`pantr.bspline._extraction_backend` computes into a fresh buffer and copies
    back. Nothing else in this file reaches that path, and a silently unfilled ``out``
    would look like a wrong answer far from its cause.

    Args:
        backend (Backend): Which implementation to run.
        dtype (npt.DTypeLike): Storage format.
    """
    _demand_the_extension_if_needed(backend)
    case = _CASES[0]
    expected = _build_layer_2(case, dtype, LagrangeVariant.EQUISPACES, backend)
    canvas = np.full((2 * expected.shape[0], *expected.shape[1:]), np.nan, dtype=dtype)
    strided = canvas[::2]
    assert not strided.flags["C_CONTIGUOUS"]
    knots = np.asarray(case.knots, dtype=dtype)
    space = BsplineSpace1D(knots, case.degree, snap_knots=False)
    with use_backend(backend):
        returned = _tabulate_Bspline_Lagrange_1D_extraction_impl(
            space.knots, case.degree, space.tolerance, out=strided
        )
    assert returned is strided
    assert np.array_equal(strided, expected), "the strided out was not filled with the answer"


def test_the_builder_binding_refuses_a_wrongly_shaped_matrix(cpp_backend: None) -> None:
    """A change-of-basis matrix that is not ``(degree + 1)`` square is a ``ValueError``.

    No counterpart in the oracle, where Layer 2 takes the matrix from a cache keyed on
    the degree. Reaching the header with a wrong shape would be undefined behaviour,
    so the binding refuses instead of asserting.
    """
    knots = np.asarray(_CASES[0].knots, dtype=np.float64)
    out = np.empty((3, 3, 3), dtype=np.float64)
    for shape in ((2, 3), (3, 2), (4, 4)):
        with pytest.raises(ValueError, match=r"lagrange_to_bernstein must have shape \(3, 3\)"):
            _bindings().lagrange_extraction_1d(knots, 2, 0.0, np.eye(*shape, dtype=np.float64), out)
    with pytest.raises(ValueError, match=r"lagrange_to_bernstein must have shape \(3, 3\)"):
        _bindings().lagrange_structural_identity_mask(
            np.asarray([3, 1, 3], dtype=np.intp),
            2,
            np.eye(2, dtype=np.float64),
            np.empty(2, dtype=np.bool_),
        )


def test_the_builder_binding_refuses_the_knot_vectors_the_oracle_refuses(
    cpp_backend: None,
) -> None:
    """The Lagrange entry point runs the same refusals as its Bézier sibling.

    Shared code on the C++ side, so what this pins is that the Lagrange binding calls
    it, and that it does so **before** the matrix check -- otherwise a negative degree
    would be reported as a shape complaint about a matrix nobody could have sized.
    """
    matrix = np.eye(3, dtype=np.float64)
    out = np.empty((1, 3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="knots must be non-decreasing"):
        _bindings().lagrange_extraction_1d(
            np.asarray([0.0, 0.0, 1.0, 0.5, 1.0, 1.0]), 2, 0.0, matrix, out
        )
    with pytest.raises(ValueError, match=r"knots must have at least 2\*degree\+2 elements"):
        _bindings().lagrange_extraction_1d(np.asarray([0.0, 0.0, 1.0]), 2, 0.0, matrix, out)
    with pytest.raises(ValueError, match="tol must be non-negative"):
        _bindings().lagrange_extraction_1d(
            np.asarray(_CASES[0].knots, dtype=np.float64), 2, -1.0, matrix, out
        )
    # Degree first, matrix second: a negative degree cannot size a matrix, so the
    # message must be the oracle's rather than the shape one.
    with pytest.raises(ValueError, match="degree must be non-negative"):
        _bindings().lagrange_extraction_1d(
            np.asarray([0.0, 0.0, 1.0, 1.0]), -1, 0.0, np.eye(1, dtype=np.float64), out
        )


def test_the_lagrange_bindings_refuse_a_dtype_they_would_have_to_cast(cpp_backend: None) -> None:
    """``.noconvert()`` on every array, and on the matrix for two separate reasons.

    In the builder a widened matrix would change the accumulation width of the
    contraction, which ``design/backend_parity.md`` Rule 9 makes part of the contract,
    so ``knots``, the matrix and ``out`` must all be the same format and a mixed call
    matches no overload.

    **The mask is the deliberate exception, and it is not a gap.** Its matrix is the
    only float it takes, so both overloads are legal and a ``float32`` matrix selects
    the ``float`` one rather than being cast. What ``.noconvert()`` still buys there
    is that an integer matrix is refused instead of being widened into a comparison
    against bits the caller never held.
    """
    knots = np.asarray(_CASES[0].knots, dtype=np.float64)
    out = np.empty((3, 3, 3), dtype=np.float64)
    with pytest.raises(TypeError):
        _bindings().lagrange_extraction_1d(knots, 2, 0.0, np.eye(3, dtype=np.float32), out)
    with pytest.raises(TypeError):
        _bindings().lagrange_extraction_1d(
            knots.astype(np.float32), 2, 0.0, np.eye(3, dtype=np.float64), out
        )
    with pytest.raises(TypeError):
        _bindings().lagrange_extraction_1d(
            np.asarray(_CASES[0].knots, dtype=np.int64), 2, 0.0, np.eye(3), out
        )

    multiplicity = np.asarray([3, 1, 3], dtype=np.intp)
    narrow = np.empty(2, dtype=np.bool_)
    wide = np.empty(2, dtype=np.bool_)
    _bindings().lagrange_structural_identity_mask(
        multiplicity, 2, np.eye(3, dtype=np.float32), narrow
    )
    _bindings().lagrange_structural_identity_mask(
        multiplicity, 2, np.eye(3, dtype=np.float64), wide
    )
    assert narrow.tolist() == wide.tolist(), (
        "the two matrix overloads of the mask disagree on the identity"
    )
    with pytest.raises(TypeError):
        _bindings().lagrange_structural_identity_mask(
            multiplicity, 2, np.eye(3, dtype=np.int64), narrow
        )
    with pytest.raises(TypeError):
        _bindings().lagrange_structural_identity_mask(
            np.asarray([3, 1, 3], dtype=np.int32),
            2,
            np.eye(3, dtype=np.float64),
            narrow,
        )


def test_the_mask_binding_refuses_its_own_bad_calls(cpp_backend: None) -> None:
    """The Lagrange mask entry point's three non-matrix refusals.

    The Numba kernel validates nothing and Layer 2 reaches it only from a space, so
    these guard a direct caller -- including a C++ one, for whom the binding's checks
    are the only ones there are.
    """
    multiplicity = np.asarray([3, 1, 3], dtype=np.intp)
    matrix = np.eye(3, dtype=np.float64)
    with pytest.raises(ValueError, match="degree must be non-negative"):
        _bindings().lagrange_structural_identity_mask(
            multiplicity, -1, matrix, np.empty(2, dtype=np.bool_)
        )
    with pytest.raises(ValueError, match="at least one class"):
        _bindings().lagrange_structural_identity_mask(
            np.empty(0, dtype=np.intp), 2, matrix, np.empty(0, dtype=np.bool_)
        )
    with pytest.raises(ValueError, match="out must have 2 elements"):
        _bindings().lagrange_structural_identity_mask(
            multiplicity, 2, matrix, np.empty(3, dtype=np.bool_)
        )


@pytest.mark.slow
@pytest.mark.parametrize("dtype", DTYPES)
def test_the_claim_holds_over_a_sweep_ten_times_the_shipped_one(
    cpp_backend: None, dtype: npt.DTypeLike
) -> None:
    """A bound checked only by the sweep that ships with it has not been checked.

    The shipped parametrization is ``len(_CASES)`` knot vectors per (dtype, variant).
    This draws ten times that many per dtype, cycling the variant, from the family the
    Bézier algorithm handles, and asserts the parity claim, the identity mask and the
    column-sum invariant on each. Degree-0 draws are discarded, since this target
    refuses them.

    Args:
        cpp_backend (None): Requires the compiled extension.
        dtype (npt.DTypeLike): Storage format.
    """
    wanted = 10 * len(_CASES) * len(_VARIANTS)
    checked = 0
    worst_column = 0.0
    draw = 0
    while checked < wanted:
        rng = np.random.default_rng(910_000 + 13 * draw)
        case = _draw(rng, dtype)
        draw += 1
        if case.degree < 1:
            continue
        variant = _VARIANTS[checked % len(_VARIANTS)]
        matrix = _matrix(case.degree, variant, dtype)
        bezier = _bezier(case, dtype)
        reference = _build_kernel(case, dtype, matrix, Backend.PYTHON)
        actual = _build_kernel(case, dtype, matrix, Backend.CPP)
        context = f"{case.label} {variant.name} in {np.dtype(dtype).name} (draw {draw})"
        assert_parity(
            actual, reference, _product_claim(case, dtype, matrix, bezier), context=context
        )

        space = BsplineSpace1D(np.asarray(case.knots, dtype=dtype), case.degree, snap_knots=False)
        with use_backend(Backend.PYTHON):
            mask_reference = _lagrange_structural_identity_mask(space, variant)
        with use_backend(Backend.CPP):
            mask_actual = _lagrange_structural_identity_mask(space, variant)
        assert np.array_equal(mask_actual, mask_reference), f"{context}: the masks differ"

        companion = _companion(bezier, matrix)
        target = np.abs(matrix.astype(np.float64)).sum(axis=0)
        deviation = np.abs(_column_sums(actual) - target)
        bound = _column_sum_tolerance(case, dtype, matrix, companion)
        assert np.all(deviation <= bound), f"{context}: a column sum missed the bound"
        worst_column = max(worst_column, float(deviation.max(initial=0.0)))
        checked += 1

    assert checked == wanted, f"the sweep ran {checked} cases, expected {wanted}"
    assert worst_column > 0.0, (
        "no drawn case rounded at all, so the column-sum bound was compared against zero "
        "throughout the sweep"
    )
