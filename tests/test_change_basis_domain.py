r"""Tests for the supported ``(degree, dtype)`` domain of the change-of-basis builders.

This file is where the domain's derivation lives. The limits tabulated in
``pantr.change_basis`` are not measured with :func:`numpy.linalg.cond` and are not fitted:
each one is the largest degree at which the matrix its builder solves with satisfies
:math:`\kappa_\infty(M)\,\varepsilon < 1`, and this module recomputes :math:`\kappa_\infty`
in exact rational (or high-precision decimal) arithmetic at that degree and at the one past
it.

Asking :func:`numpy.linalg.cond` for the same number does not settle it, and not because that
estimator is weak: applied to the *exact* matrix it agrees with the exact
:math:`\kappa_\infty` to five digits even at :math:`4 \times 10^{17}`. The trouble is the
matrix. The Bernstein-to-cardinal matrix pantr forms is itself the output of a Gram solve, so
near the boundary it is no longer the matrix the domain is about, and its computed
:math:`\kappa_\infty` comes out 8.6 times too large at degree 13 and 4.9 times too small at
degree 15 -- either of which moves the crossing by a degree.

The exact matrices are built from closed forms rather than from pantr's own tabulations, so
the derivation is independent of the code it grades: the cardinal B-splines from their
explicit truncated-power expansion, the Bernstein Gram matrix from its binomial closed form,
and the shifted Legendre polynomials from their integer monomial coefficients. The Lagrange
node families are the exception and are taken from pantr, since the boundary has to describe
the matrix pantr actually forms.
"""

from collections.abc import Callable
from decimal import Decimal, getcontext
from fractions import Fraction
from math import comb, factorial

import numpy as np
import numpy.typing as npt
import pytest

from pantr.basis import LagrangeVariant
from pantr.basis._basis_lagrange import _get_lagrange_points
from pantr.change_basis import (
    _BERNSTEIN_TO_CARDINAL_MAX_DEGREE,
    _BERNSTEIN_TO_LAGRANGE_MAX_DEGREE,
    _CARDINAL_TO_BERNSTEIN_MAX_DEGREE,
    _CARDINAL_TO_LEGENDRE_MAX_DEGREE,
    _cached_cardinal_to_bernstein_matrix,
    _DegreeLimit,
    compute_bernstein_to_cardinal_1d,
    compute_bernstein_to_lagrange_1d,
    compute_cardinal_dual_legendre_coeffs_1d,
    compute_cardinal_to_bernstein_1d,
    compute_cardinal_to_legendre_1d,
    compute_lagrange_to_bernstein_1d,
    compute_legendre_to_cardinal_1d,
    compute_monomial_to_bernstein_1d,
)

_EXACT_PRECISION = 60
r"""Significant decimal digits carried by the exact-arithmetic derivation.

Gauss-Jordan elimination on a matrix of condition number :math:`\kappa` loses on the order
of :math:`\log_{10}\kappa` digits, and no :math:`\kappa` reached here exceeds
:math:`10^{17}` -- the boundary being where :math:`\kappa\varepsilon \approx 1` and
:math:`1/\varepsilon_{64} = 4.5 \times 10^{15}`. Sixty digits therefore leaves more than
forty correct, against the one digit the comparison with 1 actually needs. Checked
directly: the Chebyshev degree-53 case returns the same :math:`\kappa_\infty` at 40, 60 and
120 digits.
"""

_FloatType = type[np.float32] | type[np.float64]
"""Either of the two floating-point formats every builder accepts."""

_DTYPES: tuple[_FloatType, ...] = (np.float32, np.float64)
"""The two formats, as a parametrization axis."""

_ExactMatrix = list[list[Decimal]]
"""Square matrix carried at ``_EXACT_PRECISION`` digits."""

getcontext().prec = _EXACT_PRECISION


def _max_degree(limit: _DegreeLimit, dtype: _FloatType) -> int:
    """Select the entry of a ``_DegreeLimit`` that applies to ``dtype``.

    Args:
        limit (_DegreeLimit): Tabulated limits for one builder.
        dtype (_FloatType): Output format.

    Returns:
        int: The largest degree that format supports.
    """
    return limit.float32 if dtype is np.float32 else limit.float64


# --------------------------------------------------------------------------------------
# Exact arithmetic: the matrices the builders solve with
# --------------------------------------------------------------------------------------


def _as_decimal(value: Fraction) -> Decimal:
    """Convert an exact rational to a ``Decimal`` at the ambient precision.

    Args:
        value (Fraction): Exact rational number.

    Returns:
        Decimal: The same number, rounded to ``_EXACT_PRECISION`` digits.
    """
    return Decimal(value.numerator) / Decimal(value.denominator)


def _norm_inf(matrix: _ExactMatrix) -> Decimal:
    """Compute the induced infinity norm, the largest absolute row sum.

    Args:
        matrix (_ExactMatrix): Square matrix.

    Returns:
        Decimal: ``max_i sum_j |matrix[i][j]|``.
    """
    return max(sum((abs(entry) for entry in row), Decimal(0)) for row in matrix)


def _invert(matrix: _ExactMatrix) -> _ExactMatrix:
    """Invert a square matrix by Gauss-Jordan elimination with partial pivoting.

    Args:
        matrix (_ExactMatrix): Square, non-singular matrix.

    Returns:
        _ExactMatrix: The inverse, at the ambient precision.
    """
    size = len(matrix)
    tableau = [
        [*matrix[row], *(Decimal(1) if row == col else Decimal(0) for col in range(size))]
        for row in range(size)
    ]
    for col in range(size):
        pivot_row = max(range(col, size), key=lambda row: abs(tableau[row][col]))
        tableau[col], tableau[pivot_row] = tableau[pivot_row], tableau[col]
        pivot = tableau[col][col]
        tableau[col] = [entry / pivot for entry in tableau[col]]
        for row in range(size):
            if row != col and tableau[row][col] != 0:
                factor = tableau[row][col]
                tableau[row] = [
                    entry - factor * pivot_entry
                    for entry, pivot_entry in zip(tableau[row], tableau[col], strict=True)
                ]
    return [row[size:] for row in tableau]


def _kappa_inf(matrix: _ExactMatrix) -> Decimal:
    """Compute ``||M||_inf * ||M^-1||_inf``.

    The infinity norm is the one the perturbation bound for a linear system is usually
    stated in, which is why the domain uses it rather than the 2-norm
    :func:`numpy.linalg.cond` returns by default.

    Args:
        matrix (_ExactMatrix): Square, non-singular matrix.

    Returns:
        Decimal: The infinity-norm condition number.
    """
    return _norm_inf(matrix) * _norm_inf(_invert(matrix))


def _cardinal_monomial(degree: int) -> list[list[Fraction]]:
    r"""Expand the cardinal B-splines of ``degree`` in the monomial basis on ``[0, 1]``.

    Uses the explicit truncated-power form quoted by
    :func:`~pantr.basis.tabulate_cardinal_bspline_1d`,

    .. math::

        B_{p,i}(t) = \frac{1}{p!} \sum_{j=0}^{p-i} \binom{p+1}{j} (-1)^j (t + p - i - j)^p ,

    whose coefficients are rational, so the expansion is exact.

    Args:
        degree (int): Polynomial degree.

    Returns:
        list[list[Fraction]]: Row ``i`` holds the monomial coefficients of the ``i``-th
        cardinal B-spline, lowest power first.
    """
    rows: list[list[Fraction]] = []
    for index in range(degree + 1):
        coeffs = [Fraction(0)] * (degree + 1)
        for term in range(degree - index + 1):
            shift = degree - index - term
            sign = Fraction((-1) ** term * comb(degree + 1, term))
            for power in range(degree + 1):
                coeffs[power] += sign * comb(degree, power) * Fraction(shift) ** (degree - power)
        rows.append([coeff / factorial(degree) for coeff in coeffs])
    return rows


def _bernstein_to_cardinal_exact(degree: int) -> _ExactMatrix:
    """Build ``A`` with ``cardinal = A @ bernstein``, exactly.

    This is what :func:`compute_bernstein_to_cardinal_1d` returns and what
    :func:`compute_cardinal_to_bernstein_1d` inverts. Obtained by converting each cardinal
    B-spline's exact monomial expansion into the Bernstein basis with
    ``t**k = sum_i [C(i,k)/C(p,k)] B_{p,i}(t)``.

    Args:
        degree (int): Polynomial degree.

    Returns:
        _ExactMatrix: The ``(degree+1, degree+1)`` matrix.
    """
    cardinal = _cardinal_monomial(degree)
    mono_to_bern = [
        [
            Fraction(comb(row, col), comb(degree, col)) if col <= row else Fraction(0)
            for col in range(degree + 1)
        ]
        for row in range(degree + 1)
    ]
    return [
        [
            _as_decimal(
                sum(
                    (
                        mono_to_bern[col][power] * cardinal[row][power]
                        for power in range(degree + 1)
                    ),
                    Fraction(0),
                )
            )
            for col in range(degree + 1)
        ]
        for row in range(degree + 1)
    ]


def _legendre_to_cardinal_exact(degree: int) -> _ExactMatrix:
    """Build ``A`` with ``cardinal = A @ legendre``, exactly.

    This is what :func:`~pantr.change_basis.compute_legendre_to_cardinal_1d` returns and
    what :func:`compute_cardinal_to_legendre_1d` inverts. Because the shifted Legendre basis
    is orthonormal, ``A[j, k]`` is the integral of the ``j``-th cardinal B-spline against
    the ``k``-th basis function; that integral is rational and the ``sqrt(2k+1)``
    normalization is carried in decimal.

    Args:
        degree (int): Polynomial degree.

    Returns:
        _ExactMatrix: The ``(degree+1, degree+1)`` matrix.
    """
    cardinal = _cardinal_monomial(degree)
    shifted_legendre = [
        [
            Fraction((-1) ** (order + power) * comb(order, power) * comb(order + power, power))
            if power <= order
            else Fraction(0)
            for power in range(degree + 1)
        ]
        for order in range(degree + 1)
    ]

    def integral(left: list[Fraction], right: list[Fraction]) -> Fraction:
        """Integrate the product of two monomial-coefficient vectors over ``[0, 1]``."""
        return sum(
            (
                lhs * rhs / (i + j + 1)
                for i, lhs in enumerate(left)
                if lhs
                for j, rhs in enumerate(right)
                if rhs
            ),
            Fraction(0),
        )

    return [
        [
            _as_decimal(integral(cardinal[row], shifted_legendre[order]))
            * Decimal(2 * order + 1).sqrt()
            for order in range(degree + 1)
        ]
        for row in range(degree + 1)
    ]


def _bernstein_gram_exact(degree: int) -> _ExactMatrix:
    """Build the Bernstein Gram matrix on ``[0, 1]``, exactly.

    ``G[i, j] = C(p,i) C(p,j) / ((2p+1) C(2p, i+j))``. This is the matrix
    :func:`compute_bernstein_to_cardinal_1d`'s projection solves with.

    Args:
        degree (int): Polynomial degree.

    Returns:
        _ExactMatrix: The ``(degree+1, degree+1)`` matrix.
    """
    return [
        [
            _as_decimal(
                Fraction(
                    comb(degree, row) * comb(degree, col),
                    (2 * degree + 1) * comb(2 * degree, row + col),
                )
            )
            for col in range(degree + 1)
        ]
        for row in range(degree + 1)
    ]


def _lagrange_to_bernstein_exact(degree: int, variant: LagrangeVariant) -> _ExactMatrix:
    """Build ``C[i, m] = B_{p,i}(x_m)`` at pantr's own Lagrange nodes, exactly.

    This is what :func:`~pantr.change_basis.compute_lagrange_to_bernstein_1d` returns and
    what :func:`compute_bernstein_to_lagrange_1d` inverts. The nodes come from pantr, as
    their exact float64 values, because the domain has to describe the matrix pantr forms.

    Args:
        degree (int): Polynomial degree.
        variant (LagrangeVariant): Node family.

    Returns:
        _ExactMatrix: The ``(degree+1, degree+1)`` matrix.
    """
    nodes = [Decimal(float(node)) for node in _get_lagrange_points(variant, degree + 1, np.float64)]

    def power(base: Decimal, exponent: int) -> Decimal:
        """Raise to a non-negative integer power, with ``base ** 0 == 1`` at ``base == 0``."""
        return Decimal(1) if exponent == 0 else base**exponent

    return [
        [
            Decimal(comb(degree, index)) * power(node, index) * power(1 - node, degree - index)
            for node in nodes
        ]
        for index in range(degree + 1)
    ]


# --------------------------------------------------------------------------------------
# The derivation: every tabulated limit is the crossing of kappa_inf * eps = 1
# --------------------------------------------------------------------------------------

_CARDINAL_TO_BERNSTEIN = "cardinal_to_bernstein"
"""Label of the one entry whose float64 limit is capped below the conditioning crossing."""

_SOLVED_MATRICES: tuple[tuple[str, Callable[[int], _ExactMatrix], _DegreeLimit], ...] = (
    (_CARDINAL_TO_BERNSTEIN, _bernstein_to_cardinal_exact, _CARDINAL_TO_BERNSTEIN_MAX_DEGREE),
    ("bernstein_to_cardinal", _bernstein_gram_exact, _BERNSTEIN_TO_CARDINAL_MAX_DEGREE),
    ("cardinal_to_legendre", _legendre_to_cardinal_exact, _CARDINAL_TO_LEGENDRE_MAX_DEGREE),
    *(
        (
            f"bernstein_to_lagrange[{variant.name}]",
            (lambda degree, variant=variant: _lagrange_to_bernstein_exact(degree, variant)),
            _BERNSTEIN_TO_LAGRANGE_MAX_DEGREE[variant],
        )
        for variant in LagrangeVariant
    ),
)
"""Each solving builder, the matrix it solves with, and the degrees it claims to support."""


@pytest.mark.parametrize(
    ("label", "build", "limit"), _SOLVED_MATRICES, ids=[case[0] for case in _SOLVED_MATRICES]
)
@pytest.mark.parametrize("dtype", _DTYPES, ids=("float32", "float64"))
def test_tabulated_limit_is_the_crossing_of_kappa_eps(
    label: str,
    build: Callable[[int], _ExactMatrix],
    limit: _DegreeLimit,
    dtype: _FloatType,
) -> None:
    """Re-derive each tabulated degree from exact arithmetic.

    The claim under test is that the tabulated degree is the *largest* one satisfying
    ``kappa_inf(M) * eps(dtype) < 1``: the supported degree has to be inside and the next
    degree outside. A test asserting only the supported side would pass for any limit small
    enough, which is why both sides are here.
    """
    max_degree = _max_degree(limit, dtype)
    eps = Decimal(float(np.finfo(dtype).eps))
    fidelity_capped = label == _CARDINAL_TO_BERNSTEIN and dtype is np.float64

    inside = _kappa_inf(build(max_degree)) * eps
    assert inside < 1, (
        f"{label}/{np.dtype(dtype).name}: kappa_inf * eps = {float(inside):.3g} at the "
        f"supported degree {max_degree}, so the tabulated limit claims a digit the solve "
        f"does not retain"
    )

    outside = _kappa_inf(build(max_degree + 1)) * eps
    if fidelity_capped:
        assert outside < 1, (
            "this entry is documented as capped below the conditioning crossing by measured "
            "fidelity; if conditioning now binds first, that reasoning is stale"
        )
    else:
        assert outside >= 1, (
            f"{label}/{np.dtype(dtype).name}: kappa_inf * eps = {float(outside):.3g} at "
            f"degree {max_degree + 1}, still below 1, so the domain is a degree tighter than "
            f"the criterion requires and refuses a usable result"
        )


def test_cardinal_to_bernstein_float64_limit_is_set_by_fidelity_not_conditioning() -> None:
    """Pin the one entry whose limit does not come from ``kappa_inf * eps``.

    ``compute_cardinal_to_bernstein_1d`` inverts the output of a Gram solve, so its input
    carries an error of order ``kappa_inf(G_bern) * eps`` and ``kappa_inf(A) * eps`` is not
    a bound on what it returns. Measured against the exact inverse, the float64 result is
    still within a few percent at degree 12 and is off by a factor of ten at degree 13,
    which is why 12 is tabulated where conditioning alone would have allowed 14.

    Degree 13 is refused now, so the second half cannot go through the builder and repeats
    its two-line body instead. That is deliberate rather than a mirror of the algorithm: the
    claim being tested is precisely that *the computation the cap forbids* returns garbage,
    so the forbidden computation is the thing that has to be run.
    """
    eps = Decimal(float(np.finfo(np.float64).eps))
    conditioning_allows = 14
    assert _kappa_inf(_bernstein_to_cardinal_exact(conditioning_allows)) * eps < 1
    assert _kappa_inf(_bernstein_to_cardinal_exact(conditioning_allows + 1)) * eps >= 1

    def relative_error(degree: int, got: npt.NDArray[np.float64]) -> float:
        """Relative infinity-norm error of a computed matrix against the exact inverse."""
        exact = _invert(_bernstein_to_cardinal_exact(degree))
        residual = [
            [Decimal(float(got[i][j])) - exact[i][j] for j in range(degree + 1)]
            for i in range(degree + 1)
        ]
        return float(_norm_inf(residual) / _norm_inf(exact))

    supported = _CARDINAL_TO_BERNSTEIN_MAX_DEGREE.float64
    assert supported < conditioning_allows, "the cap is what this test is about"

    inside = relative_error(
        supported,
        np.asarray(compute_cardinal_to_bernstein_1d(supported, np.float64), dtype=np.float64),
    )
    assert inside < 1.0, (
        f"degree {supported} is inside the domain but its relative error is {inside:.3g}: "
        f"the returned matrix carries no information"
    )

    refused = supported + 1
    forward = np.asarray(compute_bernstein_to_cardinal_1d(refused, np.float64), dtype=np.float64)
    computed = np.asarray(np.linalg.solve(forward, np.eye(refused + 1)), dtype=np.float64)
    outside = relative_error(refused, computed)
    assert outside >= 1.0, (
        f"degree {refused} is refused, yet the computation it forbids has relative error "
        f"{outside:.3g}: the cap refuses a usable result"
    )


def test_unknown_lagrange_variant_raises_value_error_not_key_error() -> None:
    """An unrecognized node family must not leave the builder through ``KeyError``.

    The domain lookup is keyed by ``LagrangeVariant``, so a bare ``dict[...]`` would turn a
    bad variant into a ``KeyError`` -- an undocumented exception type, which is the very
    thing this domain exists to stop escaping. It is coerced through ``LagrangeVariant(...)``
    instead.
    """
    with pytest.raises(ValueError, match="not a valid LagrangeVariant"):
        compute_bernstein_to_lagrange_1d(3, "not_a_node_family", np.float64)  # type: ignore[arg-type]

    # A plain string naming a real family still works: LagrangeVariant is a StrEnum.
    from_string = np.asarray(compute_bernstein_to_lagrange_1d(3, "equispaces", np.float64))  # type: ignore[arg-type]
    from_member = np.asarray(
        compute_bernstein_to_lagrange_1d(3, LagrangeVariant.EQUISPACES, np.float64)
    )
    assert np.array_equal(from_string, from_member)


def test_every_lagrange_variant_declares_a_domain() -> None:
    """Every node family needs a limit, or it silently receives an unbounded domain."""
    assert set(_BERNSTEIN_TO_LAGRANGE_MAX_DEGREE) == set(LagrangeVariant)


# --------------------------------------------------------------------------------------
# The contract: what a caller sees at the boundary
# --------------------------------------------------------------------------------------

_Builder = Callable[[int, _FloatType], npt.NDArray[np.float32 | np.float64]]
"""A change-of-basis builder reduced to its ``(degree, dtype)`` arguments."""

_BOUNDARY_CASES: tuple[tuple[str, _Builder, _DegreeLimit], ...] = (
    (
        "compute_cardinal_to_bernstein_1d",
        compute_cardinal_to_bernstein_1d,
        _CARDINAL_TO_BERNSTEIN_MAX_DEGREE,
    ),
    (
        "compute_bernstein_to_cardinal_1d",
        compute_bernstein_to_cardinal_1d,
        _BERNSTEIN_TO_CARDINAL_MAX_DEGREE,
    ),
    (
        "compute_cardinal_to_legendre_1d",
        compute_cardinal_to_legendre_1d,
        _CARDINAL_TO_LEGENDRE_MAX_DEGREE,
    ),
    (
        "compute_cardinal_dual_legendre_coeffs_1d",
        compute_cardinal_dual_legendre_coeffs_1d,
        _CARDINAL_TO_LEGENDRE_MAX_DEGREE,
    ),
    *(
        (
            "compute_bernstein_to_lagrange_1d",
            (
                lambda degree, dtype, variant=variant: compute_bernstein_to_lagrange_1d(
                    degree, variant, dtype
                )
            ),
            _BERNSTEIN_TO_LAGRANGE_MAX_DEGREE[variant],
        )
        for variant in LagrangeVariant
    ),
)
"""Each builder that refuses, with the limits it enforces."""

_BOUNDARY_IDS = [
    *(case[0] for case in _BOUNDARY_CASES[:4]),
    *(f"compute_bernstein_to_lagrange_1d[{variant.name}]" for variant in LagrangeVariant),
]
"""Readable ids for ``_BOUNDARY_CASES``, distinguishing the five node families."""


@pytest.mark.parametrize(("name", "builder", "limit"), _BOUNDARY_CASES, ids=_BOUNDARY_IDS)
@pytest.mark.parametrize("dtype", _DTYPES, ids=("float32", "float64"))
def test_boundary_succeeds_inside_and_refuses_outside(
    name: str,
    builder: _Builder,
    limit: _DegreeLimit,
    dtype: _FloatType,
) -> None:
    """The largest supported degree returns a matrix; the next one raises ``ValueError``.

    Both halves matter. Without the first, a domain that refused everything would pass;
    without the second, one that refused nothing would.
    """
    max_degree = _max_degree(limit, dtype)

    matrix = np.asarray(builder(max_degree, dtype))
    assert matrix.shape == (max_degree + 1, max_degree + 1)
    assert matrix.dtype == dtype
    assert np.all(np.isfinite(matrix))

    with pytest.raises(ValueError) as excinfo:
        builder(max_degree + 1, dtype)
    assert not isinstance(excinfo.value, np.linalg.LinAlgError), (
        "numpy's LinAlgError subclasses ValueError; the point of the domain is that the "
        "caller gets a message about the argument, not about an internal matrix"
    )
    message = str(excinfo.value)
    assert name in message
    assert str(max_degree + 1) in message
    assert np.dtype(dtype).name in message
    assert f"<= {limit.float32} in float32" in message
    assert f"<= {limit.float64} in float64" in message


def test_ticket_311_overflow_reproduction_now_refuses() -> None:
    """The issue's own reproduction: degree 36 in float32 returned 74 infinities.

    Measured on the parent commit, ``compute_cardinal_to_bernstein_1d(36, np.float32)``
    returned an array with 74 non-finite entries out of 1369 and emitted
    ``RuntimeWarning: overflow encountered in cast``. Under this project's
    ``filterwarnings = error`` that warning would fail this test on its own, so the check
    below pins the absence of the warning as well as the refusal.
    """
    with pytest.raises(ValueError, match="outside the supported domain"):
        compute_cardinal_to_bernstein_1d(36, np.float32)


_UNLIMITED_BUILDERS: tuple[tuple[str, _Builder], ...] = (
    ("compute_legendre_to_cardinal_1d", compute_legendre_to_cardinal_1d),
    ("compute_monomial_to_bernstein_1d", compute_monomial_to_bernstein_1d),
    *(
        (
            f"compute_lagrange_to_bernstein_1d[{variant.name}]",
            (
                lambda degree, dtype, variant=variant: compute_lagrange_to_bernstein_1d(
                    degree, variant, dtype
                )
            ),
        )
        for variant in LagrangeVariant
    ),
)
"""The builders documented as carrying no degree limit, in either dtype."""


@pytest.mark.parametrize(
    ("name", "builder"), _UNLIMITED_BUILDERS, ids=[case[0] for case in _UNLIMITED_BUILDERS]
)
@pytest.mark.parametrize("dtype", _DTYPES, ids=("float32", "float64"))
def test_builders_without_a_domain_accept_a_high_degree(
    name: str, builder: _Builder, dtype: _FloatType
) -> None:
    """A builder documented as unlimited must stay unlimited.

    The module docstring says these run no solve that could go singular, so they carry no
    degree limit. Nothing else pins that: extending the domain pattern to one of them by
    mistake would be an over-restrictive validation on a legitimate call, and every other
    test here would still pass. Degree 60 is far past every tabulated limit in the module.
    """
    degree = 60
    matrix = np.asarray(builder(degree, dtype))
    assert matrix.shape == (degree + 1, degree + 1)
    assert np.all(np.isfinite(matrix)), f"{name} returned non-finite entries at degree {degree}"


def test_far_outside_the_domain_still_refuses_cleanly() -> None:
    """Well past the boundary, not just one degree past, the refusal is still the clean one.

    Degree 40 in float32 is past where the *unguarded* computation starts producing
    infinities (degree 34 for the Bernstein pair, 32 for the Legendre pair) and, for two of
    these, past where it starts raising ``LinAlgError`` (37 and 38). The guard has to fire
    first in every case, or one of those escapes instead.
    """
    for builder in (
        compute_cardinal_to_bernstein_1d,
        compute_bernstein_to_cardinal_1d,
        compute_cardinal_to_legendre_1d,
        compute_cardinal_dual_legendre_coeffs_1d,
    ):
        with pytest.raises(ValueError, match="outside the supported domain") as excinfo:
            builder(40, np.float32)
        assert not isinstance(excinfo.value, np.linalg.LinAlgError)


def test_dtype_is_rejected_before_the_degree_domain() -> None:
    """An unsupported dtype must still report the dtype, not a domain the dtype has no entry in.

    ``_validate_degree_in_domain`` runs after ``_prepare_square_out``, so the dtype error
    comes first. That ordering is what lets the domain lookup assume a validated dtype, and
    nothing else pins it.
    """
    with pytest.raises(ValueError, match="dtype must be float32 or float64"):
        compute_cardinal_to_bernstein_1d(100, np.float16)


def test_refusal_leaves_a_caller_supplied_out_untouched() -> None:
    """Refusing must not scribble in the caller's buffer.

    ``out`` is validated (and so may be a real caller array) before the domain check runs, so
    the check has to refuse without writing. A sentinel makes that observable.
    """
    degree = _CARDINAL_TO_BERNSTEIN_MAX_DEGREE.float32 + 1
    out = np.full((degree + 1, degree + 1), 7.0, dtype=np.float32)
    with pytest.raises(ValueError, match="outside the supported domain"):
        compute_cardinal_to_bernstein_1d(degree, np.float32, out=out)
    assert np.array_equal(out, np.full_like(out, 7.0)), "the refused call wrote into `out`"


def test_cached_wrapper_propagates_the_domain_refusal() -> None:
    """``functools.lru_cache`` must not swallow or reshape the refusal.

    The extraction paths reach these builders through the cached wrappers, so a caller there
    has to see the same message a direct call gives.
    """
    with pytest.raises(ValueError, match="compute_cardinal_to_bernstein_1d"):
        _cached_cardinal_to_bernstein_matrix(
            _CARDINAL_TO_BERNSTEIN_MAX_DEGREE.float32 + 1, np.dtype(np.float32)
        )


def test_extraction_degrees_stay_inside_the_domain() -> None:
    """The Bezier extraction paths are unaffected by the domain (issue #311).

    ``pantr.bspline._bspline_extraction`` and
    ``pantr.bspline.spanwise_element_extraction`` reach two of these builders through the
    cached matrices: the Lagrange-to-Bernstein one, which carries no limit, and the
    cardinal-to-Bernstein one, which does. Their highest tested spline degree is 4, so this
    asserts the margin rather than re-running those suites: were a future change to tighten
    the limit below 4, the extraction tests would start failing for a reason that has
    nothing to do with extraction.
    """
    highest_extraction_degree = 4
    assert _CARDINAL_TO_BERNSTEIN_MAX_DEGREE.float32 >= highest_extraction_degree
    assert _CARDINAL_TO_BERNSTEIN_MAX_DEGREE.float64 >= highest_extraction_degree
