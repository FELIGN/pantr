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
   blocked product, summing the same ``n_quad`` terms in different orders. There
   are **two** of them, ``G`` and ``C``, and **both** are amplified by
   ``kappa_inf`` in the forward error, since a perturbation of the right-hand side
   enters with the same condition number as one of the matrix. Each costs
   ``gamma_{n+1}`` (Higham, *Accuracy and Stability of Numerical Algorithms*,
   2nd ed., SIAM 2002, §3.1; blocking only tightens this, per pp. 62-64).

   The relative size of the perturbation is **not** ``n u`` for every basis, and
   this is the step that a first version of this file got wrong. Write

       c = || |B|^T W |B| ||_inf / || B^T W B ||_inf

   for the cancellation factor of a Gram matrix. For a **non-negative** basis --
   Bernstein and every Lagrange family here -- nothing cancels and ``c = 1``
   exactly. For the **orthonormal Legendre** basis the Gram is the identity by
   exact quadrature, so every off-diagonal zero is a *cancelled* sum and ``c``
   grows like ``0.88 n`` (measured 7.95 at degree 8, 18.34 at 20). Higham states
   the general fact next to the equation cited above, p. 63: high relative accuracy
   is not guaranteed when ``|x^T y|`` is much smaller than ``|x|^T |y|``. Since
   ``kappa_inf`` is 1 for that basis, nothing else absorbs it, so ``c`` is computed
   per builder below rather than assumed.

   The ``W`` factor contributes no *summation* error -- multiplying by a diagonal
   adds only exact zeros to a dot product -- but it does cost one rounding per
   entry, which the count above carries.

2. **The solve.** ``numpy.linalg.solve`` is LAPACK ``gesv`` and the port uses
   ``Eigen::PartialPivLU``. Both are LU with partial pivoting and both are backward
   stable. The applicable result is Higham **Thm 9.4**, which is *componentwise*:
   ``|dA| <= gamma_{3n} |L||U|``, pivoting-agnostic. It is worth naming which
   theorem this is not: the *normwise* partial-pivoting form,
   ``||dA||_inf <= n^2 gamma_{3n} rho_n ||A||_inf``, is **Thm 9.5**, and read
   literally it would put an ``n^2`` in front of everything below. This file used
   to cite 9.4 and quote 9.5's shape without its ``n^2``; the bound it produced is
   still sound, but by the componentwise route, and that route needs its own
   allowance.

   The allowance the componentwise route needs is **not** the classical growth
   factor ``rho_n = max|U|/max|A|`` but

       R = || |L| |U| ||_inf / ||A||_inf

   which is what Thm 9.4 literally multiplies, and which can exceed ``rho_n`` by up
   to ``n``.

Adding the two and writing ``eps = 2 u``, the two-sided relative bound is

    (c_G + c_C + 3 R) * n * kappa_inf * eps

with ``c_C = 1`` always (the mixed matrix pairs the new basis against a
non-negative one) and ``c_G`` as above.

**``R`` is measured, not assumed, and that is new.** Over every matrix these
builders feed a solve -- Bernstein Gram, Legendre Gram, Legendre-to-cardinal,
Lagrange-to-Bernstein in all five node families -- in exact rational arithmetic
across every degree in every solvability domain: ``R <= 3.73`` and the classical
``rho_n`` is exactly ``1.000``. An independent sweep of 1508 matrices out to degree
200 through Eigen's own ``matrixLU()`` agrees, and finds ``R`` crossing 8 only near
degree 160. The allowance used below is **8**, so the margin is 2.1x in domain.

Keeping 8 rather than tightening to 4 is deliberate and was quantified: halving it
buys **one degree** on three of ten (builder, dtype) pairs, because ``kappa_inf``
grows geometrically and is what limits the parity domain. What was worth fixing was
not the size of the allowance but the two errors beside it -- the missing second
product, and ``c_G``.

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
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt
import pytest

from pantr._backend import Backend, use_backend
from pantr.basis import LagrangeVariant, tabulate_bernstein_1d, tabulate_legendre_1d
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
from pantr.quad import get_gauss_legendre_1d
from tests._parity_harness import (
    Roundings,
    assert_parity,
    bitwise_parity,
    bounded_parity,
    demand_a_compiled_seed,
    demand_the_compiled_kernel,
)
from tests.test_change_basis_domain import (
    _bernstein_gram_exact,
    _bernstein_to_cardinal_exact,
    _kappa_inf,
    _lagrange_to_bernstein_exact,
    _legendre_to_cardinal_exact,
)

_LU_ALLOWANCE: Final = 8
"""Allowance for ``R = || |L||U| ||_inf / ||A||_inf`` in Higham Thm 9.4.

**Measured, not argued.** Exact rational arithmetic over every matrix these
builders factor, across every solvability domain, gives ``R <= 3.73``, with the
classical growth factor ``rho_n`` exactly ``1.000``; an independent sweep of 1508
matrices to degree 200 through Eigen's own ``matrixLU()`` agrees and finds ``R``
crossing 8 only near degree 160. So the margin is 2.1x in domain, and the allowance
stops covering anything past degree 160 -- which no builder here reaches, and which
is the caveat to carry if one ever does.

Deliberately not tightened to 4: measured, that buys one degree on three of ten
(builder, dtype) pairs, because ``kappa_inf`` grows geometrically and is what
actually limits the parity domain.
"""

_MIXED_CANCELLATION: Final = 1.0
"""``c_C``, the cancellation factor of the mixed matrix ``C = B_new^T W B_old``.

Exactly one: ``B_old`` is a cardinal B-spline or Bernstein basis, both non-negative
on ``[0, 1]``, so no entry of the contraction cancels against another. Unlike
``c_G`` this does not need computing per builder.
"""


def _gram_cancellation(
    new_basis: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]
) -> float:
    """Compute ``c_G``, the Gram matrix's cancellation factor.

    ``|| |B|^T W |B| ||_inf / || B^T W B ||_inf``. One for a non-negative basis,
    where nothing cancels; of order ``0.88 n`` for the orthonormal Legendre basis,
    whose Gram is the identity precisely because its off-diagonal sums cancel. See
    this module's docstring for why that factor is not absorbed elsewhere.

    Args:
        new_basis (npt.NDArray[np.float64]): The new basis at the quadrature nodes,
            shape ``(n_quad, n_new)``.
        weights (npt.NDArray[np.float64]): The quadrature weights.

    Returns:
        float: The cancellation factor, at least one.
    """
    scaled = new_basis.T * weights
    gram = scaled @ new_basis
    magnitude = np.abs(new_basis).T * np.abs(weights) @ np.abs(new_basis)
    denominator = float(np.abs(gram).sum(axis=1).max())
    if denominator == 0.0:
        return 1.0
    return max(1.0, float(np.abs(magnitude).sum(axis=1).max()) / denominator)


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


class _SolvingBuilder(NamedTuple):
    """One builder whose parity bound is a condition number, and what that bound needs.

    A record rather than a tuple because four parallel fields in a parametrize list
    is exactly the shape the project's style rules send to a named struct, and
    because two of the four are only meaningful together.

    Attributes:
        label (str): The builder's short name, for failure messages.
        builder (_Builder): The public function, called as ``(degree, dtype=...)``.
        kappa_of (Callable[[int], Decimal]): Its exact condition number per degree.
        basis_name (str): The basis its Gram projection uses, for ``c_G``:
            ``"bernstein"``, ``"legendre"``, or ``"none"``.
    """

    label: str
    builder: _Builder
    kappa_of: Callable[[int], Decimal]
    basis_name: str


_SOLVING_BUILDERS: Final = (
    _SolvingBuilder(
        "bernstein_to_cardinal",
        compute_bernstein_to_cardinal_1d,
        _kappa_for_bernstein_to_cardinal,
        "bernstein",
    ),
    _SolvingBuilder(
        "cardinal_to_bernstein",
        compute_cardinal_to_bernstein_1d,
        _kappa_for_cardinal_to_bernstein,
        "bernstein",
    ),
    _SolvingBuilder(
        "legendre_to_cardinal",
        compute_legendre_to_cardinal_1d,
        _kappa_for_legendre_to_cardinal,
        "legendre",
    ),
    _SolvingBuilder(
        "cardinal_to_legendre",
        compute_cardinal_to_legendre_1d,
        _kappa_for_cardinal_to_legendre,
        "legendre",
    ),
    _SolvingBuilder(
        "cardinal_dual_legendre",
        compute_cardinal_dual_legendre_coeffs_1d,
        _kappa_for_cardinal_to_legendre,
        "legendre",
    ),
)
"""The five builders whose bound is a condition number, with the kappa each uses.

``bernstein_to_lagrange`` is not here because it takes a node variant as well as a
degree; it has its own test.
"""


def _relative_bound(
    degree: int, kappa: Decimal, dtype: npt.DTypeLike, gram_cancellation: float = 1.0
) -> float:
    """The relative parity bound ``(c_G + c_C + 3 R) n kappa_inf eps``.

    Args:
        degree (int): Polynomial degree; ``n`` is ``degree + 1``.
        kappa (Decimal): The exact condition number for this builder.
        dtype (npt.DTypeLike): The output dtype.
        gram_cancellation (float): ``c_G`` for this builder's Gram matrix. One for a
            non-negative basis. Defaults to 1.0.

    Returns:
        float: The two-sided relative bound.
    """
    eps = float(np.finfo(np.dtype(dtype)).eps)
    constant = gram_cancellation + _MIXED_CANCELLATION + 3 * _LU_ALLOWANCE
    return constant * (degree + 1) * float(kappa) * eps


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


def _cancellation_for(new_basis_name: str, degree: int, dtype: npt.DTypeLike) -> float:
    """``c_G`` for the basis a given builder projects onto, at this degree.

    Computed rather than tabulated, so a non-negative basis returns exactly one by
    arithmetic instead of by assertion, and the Legendre growth is measured at the
    degree it is used at.

    Args:
        new_basis_name (str): ``"bernstein"``, ``"legendre"``, or ``"none"`` for a
            builder that runs no Gram projection.
        degree (int): Polynomial degree.
        dtype (npt.DTypeLike): The dtype the builder was asked for.

    Returns:
        float: The cancellation factor.
    """
    if new_basis_name == "none":
        return 1.0
    with use_backend(Backend.PYTHON):
        points, weights = get_gauss_legendre_1d(degree + 1, dtype)
        tabulate = tabulate_bernstein_1d if new_basis_name == "bernstein" else tabulate_legendre_1d
        basis = tabulate(degree, points)
    return _gram_cancellation(
        np.asarray(basis, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    )


def _accuracy_domain(
    kappa_of: Callable[[int], Decimal],
    dtype: npt.DTypeLike,
    solvable_to: int,
    first: int = 0,
    basis_name: str = "none",
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
        basis_name (str): Which basis this builder projects onto, for ``c_G``.
            Defaults to "none".

    Returns:
        int: The largest degree with ``32 n kappa_inf eps < 1``, or ``first - 1``
            if none.
    """
    largest = first - 1
    for degree in range(first, solvable_to + 1):
        cancellation = _cancellation_for(basis_name, degree, dtype)
        if _relative_bound(degree, kappa_of(degree), dtype, cancellation) < 1.0:
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
    # Built on the Bernstein tabulation, so it inherits its `np.power` seed.
    demand_a_compiled_seed()
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

_BERNSTEIN_TO_LAGRANGE_SOLVABLE: Final = {
    LagrangeVariant.EQUISPACES: {"float64": 37, "float32": 17},
    LagrangeVariant.GAUSS_LEGENDRE: {"float64": 51, "float32": 22},
    LagrangeVariant.GAUSS_LOBATTO_LEGENDRE: {"float64": 52, "float32": 23},
    LagrangeVariant.CHEBYSHEV_1ST: {"float64": 52, "float32": 23},
    LagrangeVariant.CHEBYSHEV_2ND: {"float64": 52, "float32": 23},
}
"""Largest degree ``compute_bernstein_to_lagrange_1d`` accepts, per node family.

Read off ``_BERNSTEIN_TO_LAGRANGE_MAX_DEGREE`` in
:mod:`pantr.change_basis._builders`. One copy, because two tests need it and the
census below is the one that would silently stop counting if they drifted.
"""


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
@pytest.mark.parametrize("entry", _SOLVING_BUILDERS, ids=[e.label for e in _SOLVING_BUILDERS])
def test_a_solving_builder_agrees_within_its_condition_number(
    cpp_backend: None, entry: _SolvingBuilder, dtype: npt.DTypeLike
) -> None:
    """The two backends agree to ``32 n kappa_inf eps``, over the accuracy domain.

    The degrees tested are every one at which that bound is below one. Above them
    the harness refuses the claim as vacuous, which is Rule 3 and is correct: the
    answer has no digits there in either backend.
    """
    del cpp_backend
    demand_the_compiled_kernel(dtype)
    label, builder, kappa_of, basis_name = entry
    top = _accuracy_domain(
        kappa_of, dtype, _SOLVABLE_TO[(label, np.dtype(dtype).name)], basis_name=basis_name
    )
    assert top >= 0, f"{label}/{np.dtype(dtype).name}: no degree has a usable bound"

    for degree in range(top + 1):
        actual, reference = _both_backends(builder, degree, dtype)
        largest = float(np.abs(reference.astype(np.float64)).max(initial=0.0))
        relative = _relative_bound(
            degree, kappa_of(degree), dtype, _cancellation_for(basis_name, degree, dtype)
        )
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

    solvable = _BERNSTEIN_TO_LAGRANGE_SOLVABLE[variant][np.dtype(dtype).name]

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

    # The Lagrange pair is not in `_SOLVING_BUILDERS` because it takes a node
    # family as well as a degree, and it was therefore missing from this census
    # entirely -- the one builder whose gap nobody was counting.
    for variant in LagrangeVariant:
        for dtype in (np.float64, np.float32):
            name = np.dtype(dtype).name
            solvable = _BERNSTEIN_TO_LAGRANGE_SOLVABLE[variant][name]

            def kappa_of_variant(degree: int, variant: LagrangeVariant = variant) -> Decimal:
                return _kappa_inf(_lagrange_to_bernstein_exact(degree, variant))

            top = _accuracy_domain(kappa_of_variant, dtype, solvable, first=1)
            if top < solvable:
                gaps.append(
                    f"bernstein_to_lagrange[{variant.name}]/{name}: parity to {top}, "
                    f"solvable to {solvable}"
                )

    for label, _builder, kappa_of, basis_name in _SOLVING_BUILDERS:
        for dtype in (np.float64, np.float32):
            name = np.dtype(dtype).name
            solvable = _SOLVABLE_TO[(label, name)]
            top = _accuracy_domain(kappa_of, dtype, solvable, basis_name=basis_name)
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

    The assertion excludes ``legendre_to_cardinal``, and that exclusion is the
    point rather than a convenience. Its ``kappa`` is the hard-wired constant one,
    and it also happens to be where the global tightest ratio lands (at degree 2,
    with ``n = 3``). A gate whose maximum sits there claims to catch a wrong
    ``kappa`` while never evaluating one, which is the failure this test was
    written to prevent and was itself committing.

    So the threshold is applied to the builders whose bound actually carries a
    condition number. It must come within ``1e-4`` of being reached, about 80 times
    below the measured tightest point among those, so it is not fitted to the
    measurement; it is there to fail if those bounds go vacuous at once, which is
    what a wrong ``kappa`` or a wrong conversion to absolute would do.
    """
    del cpp_backend
    tightest = 0.0
    where = ""
    tightest_with_kappa = 0.0
    where_with_kappa = ""
    for label, builder, kappa_of, basis_name in _SOLVING_BUILDERS:
        for dtype in (np.float64, np.float32):
            name = np.dtype(dtype).name
            top = _accuracy_domain(
                kappa_of, dtype, _SOLVABLE_TO[(label, name)], basis_name=basis_name
            )
            for degree in range(top + 1):
                actual, reference = _both_backends(builder, degree, dtype)
                largest = float(np.abs(reference.astype(np.float64)).max(initial=0.0))
                cancellation = _cancellation_for(basis_name, degree, dtype)
                bound = _relative_bound(degree, kappa_of(degree), dtype, cancellation) * largest
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
                if float(kappa_of(degree)) > 1.0 and ratio > tightest_with_kappa:
                    tightest_with_kappa = ratio
                    where_with_kappa = f"{label}/{name} at degree {degree}"
                assert ratio <= 1.0, (
                    f"{label}/{name} degree {degree}: observed {observed:.3e} against a "
                    f"bound of {bound:.3e}. The parity assertions should have caught this "
                    f"first, so reaching it here means the two disagree about the bound"
                )

    assert tightest_with_kappa >= 1e-4, (
        f"among the builders whose bound carries a condition number, the tightest point is "
        f"{tightest_with_kappa:.3e} of its bound ({where_with_kappa}). Every such bound is now "
        f"loose enough to admit a wrong answer, which is what a mistaken kappa or a mistaken "
        f"relative-to-absolute conversion looks like. (Global tightest, including the "
        f"kappa == 1 builder: {tightest:.3e} at {where}.)"
    )
