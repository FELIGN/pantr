r"""Parity of the eight change-of-basis builders against their Python oracle.

`cpp/include/pantr/change_basis/change_basis.hpp` names this file as the place its
parity claims are derived. This is the first ported module whose builders are not
all bit-identical, and the reason is worth stating before the derivation: it is the
first one that **solves a linear system**.

Three groups, and the split is structural
-----------------------------------------

**Two builders are bitwise.** `compute_lagrange_to_bernstein_1d` is a Bernstein
tabulation transposed, and `compute_monomial_to_bernstein_1d` is a table of
binomial quotients. Neither runs a solve or a matrix product, so both inherit the
bit-exactness of the kernels beneath them.

**One is bounded but never approaches its bound.**
`compute_legendre_to_cardinal_1d` is a Gram projection whose Gram matrix is the
identity up to round-off -- the new basis is orthonormal and the quadrature
integrates every product it forms exactly -- so its solve is perfectly
conditioned and the two backends differ only by the summation order of the mixed
matrix.

**Five carry a genuine bound**, because each inverts an ill-conditioned matrix.

The bound
---------

Both backends compute the same matrix by the same three steps: tabulate the two
bases at the quadrature nodes, form ``G = B^T W B`` and ``C = B^T W A``, and solve
``G M^T = C``. The tabulations are bit-identical (measured; see
`tests/parity/test_basis_tabulations.py`), so they
contribute nothing. The other two steps do:

1. **The products.** numpy contracts them with a BLAS ``gemm``, Eigen with its own
   blocked product. Both sum the same ``n_quad`` terms in different orders, so each
   perturbs ``G`` and ``C`` by a relative ``n u`` (Higham, *Accuracy and Stability
   of Numerical Algorithms*, 2nd ed., SIAM 2002, §3.1). The ``W`` factor
   contributes nothing: multiplying by a diagonal adds only exact zeros to a dot
   product.

2. **The solve.** ``numpy.linalg.solve`` is LAPACK ``gesv`` and the port uses
   ``Eigen::PartialPivLU``. Both are LU with partial pivoting and both are backward
   stable, so each computed solution satisfies ``(A + dA) x = b`` with
   ``||dA||_inf <= c rho gamma_n ||A||_inf`` and hence a forward error of order
   ``kappa_inf(A) gamma_{3n}`` (ibid., Thm 9.4).

Adding the two and writing ``eps = 2 u``, one backend's relative error is at most
about ``4 n kappa_inf rho u``, i.e. ``2 n kappa_inf rho eps``. The harness doubles
it, since neither side is the exact answer.

**The constant used below is 32, and ``rho`` is why.** Partial pivoting bounds the
growth factor only by ``2**(n-1)``, which is worthless here, and in practice it is
a small number for these matrices. So 32 folds in an allowance of ``rho <= 8`` on
top of the ``4 n kappa eps``, and that allowance is **argued, not proved**: it is
the one place in this file where a constant is not derived. What is measured is
the margin, and `test_the_bound_is_approached_but_not_by_orders_of_magnitude`
reports it so a future reader can see whether 8 was generous or lucky.

Which ``kappa``, per builder
----------------------------

Each is the condition number of the matrix that builder's solve actually uses, in
**exact rational arithmetic** -- the helpers in `tests/test_change_basis_domain.py`
rebuild each matrix from closed forms rather than from what pantr computes. That
matters: `pantr.change_basis`'s module docstring measures
``numpy.linalg.cond`` on the *formed* Bernstein/cardinal matrix at 8.6 times too
high at degree 13 and 4.9 times too low at degree 15, either of which would move
the bound.

* ``bernstein_to_cardinal``: the Bernstein Gram matrix it solves with.
* ``cardinal_to_bernstein``: the **product** of two condition numbers. It inverts
  ``A``, which is itself the previous builder's Gram solve and so carries a
  relative error of ``kappa_inf(G_bern) eps``; inverting a perturbed matrix
  multiplies that by ``kappa_inf(A)``.
* ``legendre_to_cardinal``: one, its Gram being the identity.
* ``cardinal_to_legendre`` and ``cardinal_dual_legendre_coeffs``: ``kappa_inf(A)``
  with ``A`` the Legendre-to-cardinal matrix, which unlike the Bernstein pair
  **is** accurate, so no product is needed.
* ``bernstein_to_lagrange``: ``kappa_inf(C)`` with ``C`` the Lagrange-to-Bernstein
  matrix, which is a pure tabulation and therefore exact to a rounding.

The parity domain is smaller than the solvability domain, and that is the point
---------------------------------------------------------------------------------

`tests/_parity_harness.py` refuses a bound at least as large as the values it
compares, which is Rule 3 of `design/backend_parity.md`. Here that guard has teeth:
each builder's **solvability** domain runs to where ``kappa_inf eps < 1``, and at
the top of that range the bound above is of order one -- so the comparison would
decide nothing and the harness rightly rejects it.

So parity is asserted over the **accuracy** domain instead: the degrees where
``32 n kappa_inf eps < 1``. The two differ, per builder and per dtype, and
`test_the_excluded_degrees_are_named` prints the gap rather than letting it pass
as coverage. This is not a weakening. Above the accuracy domain the *answer* has no
correct digits in either backend, so there is nothing for a parity claim to be
about.

One independent confirmation fell out of it. For ``cardinal_to_bernstein`` in
float64 the derivation above stops at degree 9, and `pantr.change_basis`'s own
docstring -- written before this port, from a different argument -- says its
accuracy bound "reaches 100% after degree 10". Two derivations, one degree apart.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Final

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.basis import LagrangeVariant
from pantr.basis._basis_lagrange import _get_lagrange_points
from pantr.change_basis import (
    compute_bernstein_to_cardinal_1d,
    compute_bernstein_to_lagrange_1d,
    compute_cardinal_dual_legendre_coeffs_1d,
    compute_cardinal_to_bernstein_1d,
    compute_cardinal_to_legendre_1d,
    compute_lagrange_to_bernstein_1d,
    compute_legendre_to_cardinal_1d,
    compute_monomial_to_bernstein_1d,
)
from tests._parity_harness import (
    Roundings,
    assert_parity,
    bitwise_parity,
    bounded_parity,
    demand_the_compiled_kernel,
)
from tests.test_change_basis_domain import (
    _bernstein_gram_exact,
    _bernstein_to_cardinal_exact,
    _kappa_inf,
    _lagrange_to_bernstein_exact,
    _legendre_to_cardinal_exact,
)

_GROWTH_ALLOWANCE: Final = 8
"""Allowance for the LU growth factor ``rho``; see this module's docstring.

The one constant here that is argued rather than derived. Partial pivoting bounds
``rho`` only by ``2**(n-1)``, which no test could use, and the measured margin is
reported by :func:`test_the_bound_is_approached_but_not_by_orders_of_magnitude`.
"""

_ROUNDINGS_PER_BACKEND: Final = 4
"""``n u`` from the two products plus ``3 n u`` from the solve; see the docstring."""

_CONSTANT: Final = _GROWTH_ALLOWANCE * _ROUNDINGS_PER_BACKEND
"""The 32 of ``32 n kappa_inf eps``."""


def _kappa_for_bernstein_to_cardinal(degree: int) -> Decimal:
    """Condition number of the Bernstein Gram matrix, in exact arithmetic.

    Args:
        degree (int): Polynomial degree.

    Returns:
        Decimal: ``kappa_inf`` of the matrix the projection solves with.
    """
    return _kappa_inf(_bernstein_gram_exact(degree))


def _kappa_for_cardinal_to_bernstein(degree: int) -> Decimal:
    """Product of two exact condition numbers; see this module's docstring.

    Args:
        degree (int): Polynomial degree.

    Returns:
        Decimal: ``kappa_inf(A) * kappa_inf(G_bern)``.
    """
    return _kappa_inf(_bernstein_to_cardinal_exact(degree)) * _kappa_inf(
        _bernstein_gram_exact(degree)
    )


def _kappa_for_legendre_to_cardinal(degree: int) -> Decimal:
    """One: the Legendre Gram matrix under this quadrature is the identity.

    Args:
        degree (int): Polynomial degree, unused.

    Returns:
        Decimal: ``Decimal(1)``.
    """
    del degree
    return Decimal(1)


def _kappa_for_cardinal_to_legendre(degree: int) -> Decimal:
    """Condition number of the Legendre-to-cardinal matrix, in exact arithmetic.

    Args:
        degree (int): Polynomial degree.

    Returns:
        Decimal: ``kappa_inf`` of the matrix this builder inverts.
    """
    return _kappa_inf(_legendre_to_cardinal_exact(degree))


def _kappa_for_bernstein_to_lagrange(degree: int) -> Decimal:
    """Condition number of the Lagrange-to-Bernstein matrix, in exact arithmetic.

    Args:
        degree (int): Polynomial degree.

    Returns:
        Decimal: ``kappa_inf`` of the matrix this builder inverts.
    """
    return _kappa_inf(_lagrange_to_bernstein_exact(degree, LagrangeVariant.EQUISPACES))


_Builder = Callable[..., npt.NDArray[np.float32 | np.float64]]
"""A public change-of-basis builder, called as ``(degree, dtype)``."""

_SOLVING_BUILDERS: Final = (
    ("bernstein_to_cardinal", compute_bernstein_to_cardinal_1d, _kappa_for_bernstein_to_cardinal),
    ("cardinal_to_bernstein", compute_cardinal_to_bernstein_1d, _kappa_for_cardinal_to_bernstein),
    ("legendre_to_cardinal", compute_legendre_to_cardinal_1d, _kappa_for_legendre_to_cardinal),
    ("cardinal_to_legendre", compute_cardinal_to_legendre_1d, _kappa_for_cardinal_to_legendre),
    (
        "cardinal_dual_legendre",
        compute_cardinal_dual_legendre_coeffs_1d,
        _kappa_for_cardinal_to_legendre,
    ),
)
"""The five builders whose bound is a condition number, with the kappa each uses.

``bernstein_to_lagrange`` is not here because it takes a node variant as well as a
degree; it has its own test.
"""


def _relative_bound(degree: int, kappa: Decimal, dtype: npt.DTypeLike) -> float:
    """The relative parity bound ``32 n kappa_inf eps``.

    Args:
        degree (int): Polynomial degree; ``n`` is ``degree + 1``.
        kappa (Decimal): The exact condition number for this builder.
        dtype (npt.DTypeLike): The output dtype.

    Returns:
        float: The relative bound.
    """
    eps = float(np.finfo(np.dtype(dtype)).eps)
    return _CONSTANT * (degree + 1) * float(kappa) * eps


def _node_disagreement(variant: LagrangeVariant, n_pts: int, dtype: npt.DTypeLike) -> float:
    """Largest gap between the Lagrange nodes the two backends produce.

    An *input* difference rather than an output claim: the node families resolve
    through :mod:`pantr.quad`, three of whose rules dispatch, so for some of them
    the two backends do not start from the same array. Measuring it here keeps the
    claim about the change-of-basis builder rather than about the quadrature.

    Args:
        variant (LagrangeVariant): The node family.
        n_pts (int): Number of nodes.
        dtype (npt.DTypeLike): The dtype the nodes are built in.

    Returns:
        float: ``max |x_cpp - x_python|``, zero when the two agree bit for bit.
    """
    with use_backend(Backend.PYTHON):
        reference = _get_lagrange_points(variant, n_pts, dtype)
    with use_backend(Backend.CPP):
        actual = _get_lagrange_points(variant, n_pts, dtype)
    return float(np.abs(actual.astype(np.float64) - reference.astype(np.float64)).max(initial=0.0))


def _amplification_for(relative: float, largest: float, dtype: npt.DTypeLike) -> float:
    """Convert a relative bound into the amplification the harness multiplies by ``u``.

    :func:`tests._parity_harness.absolute_tolerance` returns
    ``2 * (gamma * amplification)``, and with the unit-``u`` ``Roundings`` used here
    ``gamma`` is ``u`` of the *accumulator* format. So an amplification of
    ``relative * largest / eps`` yields an absolute tolerance of exactly
    ``relative * largest``, since ``eps`` is ``2 u``.

    The magnitude is the array's largest entry rather than each entry's own: a
    solve's forward error is bounded in norm, so an individually tiny entry can
    still carry an error of the norm's size, and an elementwise magnitude would
    claim otherwise.

    Args:
        relative (float): The derived relative bound.
        largest (float): The largest magnitude in the reference array.
        dtype (npt.DTypeLike): The format the claim accumulates in.

    Returns:
        float: The amplification to hand to :func:`bounded_parity`.
    """
    return relative * largest / float(np.finfo(np.dtype(dtype)).eps)


def _accuracy_domain(
    kappa_of: Callable[[int], Decimal],
    dtype: npt.DTypeLike,
    solvable_to: int,
    first: int = 0,
) -> int:
    """Largest degree at which the parity bound is still below one.

    Above it the bound would exceed the values being compared and
    :func:`tests._parity_harness.assert_parity` would refuse it, which is Rule 3
    of ``design/backend_parity.md`` doing its job rather than a limitation here.

    Args:
        kappa_of (Callable[[int], Decimal]): The builder's exact condition number.
        dtype (npt.DTypeLike): The output dtype.
        solvable_to (int): The builder's tabulated degree limit for this dtype.
        first (int): Lowest degree to consider. Defaults to 0; the Lagrange pair
            needs 1, since two of its node families are undefined at one point.

    Returns:
        int: The largest degree with ``32 n kappa_inf eps < 1``, or ``first - 1``
            if none.
    """
    largest = first - 1
    for degree in range(first, solvable_to + 1):
        if _relative_bound(degree, kappa_of(degree), dtype) < 1.0:
            largest = degree
        else:
            break
    return largest


def _both_backends(
    builder: _Builder, degree: int, dtype: npt.DTypeLike, **kwargs: object
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Run one builder under each backend.

    Args:
        builder (_Builder): The public builder to call.
        degree (int): Polynomial degree.
        dtype (npt.DTypeLike): The output dtype.
        **kwargs (object): Extra keyword arguments for the builder.

    Returns:
        tuple: ``(cpp_result, python_result)``, in the order
            :func:`tests._parity_harness.assert_parity` expects.
    """
    with use_backend(Backend.PYTHON):
        reference = builder(degree, dtype=dtype, **kwargs)
    with use_backend(Backend.CPP):
        actual = builder(degree, dtype=dtype, **kwargs)
    return actual, reference


# --------------------------------------------------------------------------
# The two builders that solve nothing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("degree", [1, 2, 5, 12, 20])
@pytest.mark.parametrize("variant", list(LagrangeVariant))
def test_lagrange_to_bernstein_is_bitwise(
    cpp_backend: None, degree: int, dtype: npt.DTypeLike, variant: LagrangeVariant
) -> None:
    """The transposed tabulation is bit-identical, at every node family.

    No solve and no product: the matrix *is* the Bernstein basis evaluated at the
    Lagrange nodes, and that tabulation is bit-identical between the backends. The
    node families are parametrized because two of the five are ones
    :mod:`pantr._backend` never dispatches, so the nodes are the same array on both
    sides by construction and the claim tests the tabulation alone.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    actual, reference = _both_backends(
        compute_lagrange_to_bernstein_1d, degree, dtype, lagrange_variant=variant
    )
    context = f"lagrange_to_bernstein degree {degree} {np.dtype(dtype).name} {variant.name}"

    node_gap = _node_disagreement(variant, degree + 1, dtype)
    if node_gap == 0.0:
        assert_parity(
            actual,
            reference,
            bitwise_parity(
                why=(
                    "the matrix is the Bernstein basis tabulated at the Lagrange nodes and "
                    "transposed. No solve, no product, the tabulation itself is bitwise "
                    "(tests/parity/test_basis_tabulations.py), and this node family hands "
                    "both backends the same array"
                )
            ),
            context=context,
        )
        return

    # The one case where the nodes themselves differ; see this module's docstring.
    largest = float(np.abs(reference.astype(np.float64)).max(initial=0.0))
    eps = float(np.finfo(np.dtype(dtype)).eps)
    amplification = np.full(reference.shape, (degree * node_gap) / eps + largest)
    assert_parity(
        actual,
        reference,
        bounded_parity(
            roundings=Roundings(stages=1, accumulator_per_stage=1, storage_per_stage=0),
            accumulator=dtype,
            storage=dtype,
            amplification=amplification,
            why=(
                f"the two backends are handed different nodes: {variant.name} resolves "
                f"through pantr.quad.get_modified_chebyshev_nodes_1d, which dispatches and "
                f"which the quad port measured as exact in float64 and one unit of roundoff "
                f"in float32. Measured here at {node_gap:.3e}. Given identical nodes the "
                f"tabulation is bitwise, so the whole difference is that perturbation "
                f"propagated: |B'_(j,n)| = |n (B_(j-1,n-1) - B_(j,n-1))| <= n since every "
                f"Bernstein value lies in [0, 1], so the matrix moves by at most n times the "
                f"node gap, plus one rounding of the result"
            ),
        ),
        context=context,
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("degree", [0, 1, 3, 10, 30, 56])
def test_monomial_to_bernstein_is_bitwise(
    cpp_backend: None, degree: int, dtype: npt.DTypeLike
) -> None:
    """The binomial quotients are bit-identical through degree 56.

    Both backends form ``C(i, j) / C(degree, j)`` as the correctly rounded quotient
    of two exactly represented integers: Python's ``math.comb`` is an exact integer
    and its ``/`` rounds once, and the C++ side builds a Pascal triangle by
    addition, which keeps every entry exact while it stays at or below ``2**53``.

    Degree 56 is the last degree at which that holds -- ``C(56, 28)`` is the largest
    central binomial below ``2**53`` -- and it is in the list on purpose, as the
    frontier case. The builder carries no degree limit, so above 56 the two may
    round differently and no claim is made.
    """
    del cpp_backend
    actual, reference = _both_backends(compute_monomial_to_bernstein_1d, degree, dtype)
    assert_parity(
        actual,
        reference,
        bitwise_parity(
            why=(
                "both backends divide two exactly represented integers, so both get the "
                "correctly rounded quotient of the same exact rational. Holds while every "
                "binomial is at or below 2**53, i.e. through degree 56"
            )
        ),
        context=f"monomial_to_bernstein degree {degree} {np.dtype(dtype).name}",
    )


# --------------------------------------------------------------------------
# The six that solve
# --------------------------------------------------------------------------

_SOLVABLE_TO: Final = {
    ("bernstein_to_cardinal", "float64"): 26,
    ("bernstein_to_cardinal", "float32"): 12,
    ("cardinal_to_bernstein", "float64"): 14,
    ("cardinal_to_bernstein", "float32"): 8,
    ("legendre_to_cardinal", "float64"): 20,
    ("legendre_to_cardinal", "float32"): 20,
    ("cardinal_to_legendre", "float64"): 12,
    ("cardinal_to_legendre", "float32"): 6,
    ("cardinal_dual_legendre", "float64"): 12,
    ("cardinal_dual_legendre", "float32"): 6,
}
"""Largest degree each builder accepts, per dtype.

Read off ``pantr.change_basis``'s tabulated limits, except for
``legendre_to_cardinal``, which carries none -- it solves with an identity Gram --
and is capped here at 20 only to bound the test's runtime.
"""


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(("label", "builder", "kappa_of"), _SOLVING_BUILDERS)
def test_a_solving_builder_agrees_within_its_condition_number(
    cpp_backend: None,
    label: str,
    builder: _Builder,
    kappa_of: Callable[[int], Decimal],
    dtype: npt.DTypeLike,
) -> None:
    """The two backends agree to ``32 n kappa_inf eps``, over the accuracy domain.

    The degrees tested are every one at which that bound is below one. Above them
    the harness refuses the claim as vacuous, which is Rule 3 and is correct: the
    answer has no digits there in either backend.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    top = _accuracy_domain(kappa_of, dtype, _SOLVABLE_TO[(label, np.dtype(dtype).name)])
    assert top >= 0, f"{label}/{np.dtype(dtype).name}: no degree has a usable bound"

    for degree in range(top + 1):
        actual, reference = _both_backends(builder, degree, dtype)
        largest = float(np.abs(reference.astype(np.float64)).max(initial=0.0))
        relative = _relative_bound(degree, kappa_of(degree), dtype)
        # The magnitude that turns the relative bound into an absolute one is the
        # array's largest entry, not each entry's own. A solve's forward error is
        # bounded in norm, so an individually tiny entry can still carry an error
        # of the norm's size, and an elementwise magnitude would claim otherwise.
        amplification = np.full(reference.shape, _amplification_for(relative, largest, dtype))
        assert_parity(
            actual,
            reference,
            bounded_parity(
                roundings=Roundings(stages=1, accumulator_per_stage=1, storage_per_stage=0),
                accumulator=dtype,
                storage=dtype,
                amplification=amplification,
                why=(
                    f"32 n kappa_inf eps with kappa_inf = {float(kappa_of(degree)):.4e} "
                    f"computed in exact rational arithmetic, n = {degree + 1}. "
                    f"The budget is a derived ratio rather than a dependency chain, so "
                    f"Roundings is the unit-u spelling; see this module's docstring for "
                    f"the derivation and for why the constant is 32"
                ),
            ),
            context=f"{label} degree {degree} {np.dtype(dtype).name}",
        )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("variant", list(LagrangeVariant))
def test_bernstein_to_lagrange_agrees_within_its_condition_number(
    cpp_backend: None, dtype: npt.DTypeLike, variant: LagrangeVariant
) -> None:
    """Same claim as the five above, for the builder that takes a node family.

    Its condition number depends on the family: equispaced nodes are the worst,
    and the exact ``kappa_inf`` is recomputed here per variant rather than assumed
    to be the equispaced one.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)

    def kappa_of(degree: int) -> Decimal:
        return _kappa_inf(_lagrange_to_bernstein_exact(degree, variant))

    solvable = {
        LagrangeVariant.EQUISPACES: {"float64": 37, "float32": 17},
        LagrangeVariant.GAUSS_LEGENDRE: {"float64": 51, "float32": 22},
        LagrangeVariant.GAUSS_LOBATTO_LEGENDRE: {"float64": 52, "float32": 23},
        LagrangeVariant.CHEBYSHEV_1ST: {"float64": 52, "float32": 23},
        LagrangeVariant.CHEBYSHEV_2ND: {"float64": 52, "float32": 23},
    }[variant][np.dtype(dtype).name]

    top = _accuracy_domain(kappa_of, dtype, min(solvable, 20), first=1)
    assert top >= 1, f"{variant.name}/{np.dtype(dtype).name}: no degree has a usable bound"

    for degree in range(1, top + 1):
        actual, reference = _both_backends(
            compute_bernstein_to_lagrange_1d, degree, dtype, lagrange_variant=variant
        )
        largest = float(np.abs(reference.astype(np.float64)).max(initial=0.0))
        relative = _relative_bound(degree, kappa_of(degree), dtype)
        amplification = np.full(reference.shape, _amplification_for(relative, largest, dtype))
        assert_parity(
            actual,
            reference,
            bounded_parity(
                roundings=Roundings(stages=1, accumulator_per_stage=1, storage_per_stage=0),
                accumulator=dtype,
                storage=dtype,
                amplification=amplification,
                why=(
                    f"32 n kappa_inf eps with kappa_inf = {float(kappa_of(degree)):.4e} for "
                    f"{variant.name} nodes, computed in exact rational arithmetic"
                ),
            ),
            context=f"bernstein_to_lagrange degree {degree} {variant.name} {np.dtype(dtype).name}",
        )


# --------------------------------------------------------------------------
# What the claims above do and do not cover
# --------------------------------------------------------------------------


def test_the_excluded_degrees_are_named() -> None:
    """Every degree inside the solvability domain but outside the parity domain.

    A test that quietly stops at the accuracy domain would read as full coverage of
    the module. This one fails if the gap is ever empty for all builders -- which
    would mean the bound had become vacuous or the domains had merged -- and
    otherwise records the gap in its own assertion message.
    """
    gaps: list[str] = []
    for label, _builder, kappa_of in _SOLVING_BUILDERS:
        for dtype in (np.float64, np.float32):
            name = np.dtype(dtype).name
            solvable = _SOLVABLE_TO[(label, name)]
            top = _accuracy_domain(kappa_of, dtype, solvable)
            if top < solvable:
                gaps.append(f"{label}/{name}: parity to {top}, solvable to {solvable}")

    assert gaps, (
        "no builder has a degree it accepts but this file cannot compare, which would "
        "mean either the bound collapsed or the tabulated domains changed. Both are "
        "findings rather than good news"
    )


def test_the_bound_is_approached_but_not_by_orders_of_magnitude(cpp_backend: None) -> None:
    """The bound is loose, and this measures by how much rather than assuming.

    A tolerance never approached asserts nothing, and the constant in this file
    carries an argued allowance of ``rho <= 8`` for the LU growth factor. So the
    margin is measured over **every** degree tested, not only the top one, and the
    distinction matters: at the top of each accuracy domain ``kappa`` is by
    construction as large as it gets and the bound is at its loosest, while lower
    down it tracks the observed difference far more closely.

    Measured on the development server, worst ratio of observed difference to
    derived bound, per builder, over its whole accuracy domain in float64:
    ``bernstein_to_cardinal`` 2.0e-3, ``cardinal_to_bernstein`` 8.7e-4,
    ``legendre_to_cardinal`` 7.8e-3, ``cardinal_to_legendre`` 5.6e-4. So the bound
    is between roughly 130 and 1800 times what is observed at its tightest point,
    which is the usual price of a condition-number argument: it is a worst case
    over all perturbation directions, and two implementations of the same algorithm
    do not perturb in the worst one.

    The assertion is global rather than per builder: somewhere in this file's
    coverage the bound must come within ``1e-4`` of being reached. That threshold
    sits about 80 times below the measured 7.8e-3, so it is not fitted to the
    measurement; it is there to fail if the bounds all go vacuous at once, which is
    what a wrong ``kappa`` or a wrong conversion to absolute would do.
    """
    del cpp_backend
    tightest = 0.0
    where = ""
    for label, builder, kappa_of in _SOLVING_BUILDERS:
        for dtype in (np.float64, np.float32):
            name = np.dtype(dtype).name
            top = _accuracy_domain(kappa_of, dtype, _SOLVABLE_TO[(label, name)])
            for degree in range(top + 1):
                actual, reference = _both_backends(builder, degree, dtype)
                largest = float(np.abs(reference.astype(np.float64)).max(initial=0.0))
                bound = _relative_bound(degree, kappa_of(degree), dtype) * largest
                observed = float(
                    np.abs(actual.astype(np.float64) - reference.astype(np.float64)).max(
                        initial=0.0
                    )
                )
                if bound <= 0.0:
                    continue
                ratio = observed / bound
                if ratio > tightest:
                    tightest, where = ratio, f"{label}/{name} at degree {degree}"
                assert ratio <= 1.0, (
                    f"{label}/{name} degree {degree}: observed {observed:.3e} against a "
                    f"bound of {bound:.3e}. The parity assertions should have caught this "
                    f"first, so reaching it here means the two disagree about the bound"
                )

    assert tightest >= 1e-4, (
        f"the tightest point in this file's whole coverage is {tightest:.3e} of its bound "
        f"({where}). Every bound here is now loose enough to admit a wrong answer, which is "
        f"what a mistaken kappa or a mistaken relative-to-absolute conversion looks like"
    )
