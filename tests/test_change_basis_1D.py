"""Tests for change_basis_1D module."""

import functools
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from pantr.basis import (
    LagrangeVariant,
    tabulate_bernstein_1d,
    tabulate_cardinal_bspline_1d,
    tabulate_lagrange_1d,
    tabulate_legendre_1d,
)
from pantr.change_basis import (
    _cached_cardinal_to_bernstein_matrix,
    _cached_cardinal_to_legendre_matrix,
    _cached_lagrange_to_bernstein_matrix,
    _cached_legendre_to_cardinal_matrix,
    _compute_change_basis_1D,
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
from pantr.tolerance import get_machine_epsilon

_COND_SAFETY_FACTOR = 64
"""
Safety factor multiplying the first-order error bound ``cond(A) * eps``.

**Only one of the two products is bounded, and which one is a property of the
algorithm, not of the matrices.** Every pair here obtains its inverse direction by
solving ``A B = I``, which bounds the *right* residual ``||A B - I||`` and says nothing
about ``||B A - I||``. Du Croz & Higham, *Stability of methods for matrix inversion*,
IMA J. Numer. Anal. **12** (1992) 1-19 (DOI ``10.1093/imanum/12.1.1``), call this
Method A -- literally this algorithm, one solve per unit vector under GEPP -- and bound
it in eq. (3.2) as ``|A X - I| <= c'_n u |L||U||X| + O(u^2)``. Their §4 states the
principle: *"only one of the left and right residuals is guaranteed to be small; which
one depends on whether the method is derived by solving AX = I or XA = I."* Measured
here at degree 28, as ``(forward ratio, reversed ratio)`` in units of ``cond(A) * eps``:
``numpy.linalg.solve(A, I)`` gives ``(0.11, 1067)``, the transposed solve
``solve(A.T, I).T`` gives ``(4073, 0.11)``, and ``scipy.linalg.inv`` (LAPACK ``getri``)
gives ``(1843, 0.13)``. Same matrix throughout: the protection follows the route. Any
reimplementation must solve on the same side or the docstrings here become true of the
other product -- a row/column-major transposition is enough to swap it silently. The
same wrong-parity claim was once in the LINPACK manual; that paper corrects it on p. 16.

The derived half, for the forward product. With ``|dA| <= gamma_{3n} |L||U|`` (ASNA 2nd
ed., Thm 9.4, for one right-hand side; the extension to ``n`` columns of the identity is
eq. (3.2) above), the ``3n`` covering the factorisation and the two triangular solves,
and the check itself forming ``A @ B`` for another ``gamma_n``:

    ||A B - I||_2  <=  (3n * rho_norm + n) * cond(A) * eps

with ``rho_norm = || |L||U| ||_2 / ||A||_2``, which is what a *normwise* bound needs --
not the classical growth factor ``max|u_ij| / max|a_ij|``. The distinction is not
cosmetic here: over the 46 matrices this file inverts, the classical factor is 0.750 to
1.155 while ``rho_norm`` runs 1.000 to **2.109** (Gauss-Legendre, degree 10) and grows
with degree. The highest degree tested is 11, so ``n = 12`` and the first-order
requirement is ``3 * 12 * 2.109 + 12 ~ 88``.

**So 64 is 0.73x the first-order bound, not above it.** The constant is therefore *not*
justified by that bound; it is justified by the margins measured below, which sit two
orders of magnitude under it. Stated plainly because the reverse was claimed before: the
first-order bound is what the constant would have to clear to be derived, and it does
not clear it. Because ``n`` is baked in, the derivation is stated for ``degree <= 11``;
past that it must be recomputed, and ``rho_norm`` grows.

**The reversed product is graded at the same tier, and that is false past degree 20.**
From ``B = A^-1 (I + R)`` one gets ``B A - I = A^-1 R A``, hence a priori only
``||B A - I|| <= cond(A) * ||A B - I||``, i.e. a ``cond(A)**2 * eps`` tier. No stronger
bound exists: the reference above declines to name a size for the unprotected side, and
its Table 3.2 is a published counterexample. Holding the reversed product to
``cond(A) * eps`` is an *empirical* claim, true over the degrees tested and false beyond
them. Measured with the products formed in exact rational arithmetic (forming them in
float64 costs ``~n * eps * cond(A)``, the very tier being graded), ``||B A - I||_2``
first exceeds ``64 * cond(A) * eps`` at degree **20** for equispaced and Chebyshev-1st
nodes (ratios 107 and 117), 21 for Gauss-Lobatto and Chebyshev-2nd, 22 for
Gauss-Legendre, reaching 4.5e3 by degree 30 while the bound is still meaningful -- it
stays below 0.1 through degree 32, so there are thirteen degrees where the assertion is
both meaningful and wrong. The crossing degree moves by about 2 with the LAPACK entry
point (at degree 20 the ratio is 107 through ``numpy.linalg.solve``, 23 through
``scipy.linalg.solve``), so a degree cap records one build rather than a bound.
**Do not extend any parametrization in this file past degree 11 without re-measuring the
reversed product.** The other two pairs never cross only because ``cond(A)`` grows 20-30x
per degree against 2.6x, so ``64 * cond(A) * eps`` passes 0.1 -- and the assertion goes
vacuous -- at degree 13 (Bernstein/cardinal) and 11 (Legendre/cardinal), before the
reversed product has room to fail.

Measured margins (``atol`` over observed), worst over the degrees actually tested:

* 151x for the Bernstein/cardinal forward product, whose ratio to ``cond(A) * eps``
  peaks at 0.43 (degree 11), having been 0.30 at degree 8;
* **31x** for the Lagrange/Bernstein *reversed* product, whose ratio peaks at 2.04
  (Chebyshev 1st, degree 11) -- the binding case. Equispaced at degree 10, 1.58, is not
  the worst: the peak moves to Chebyshev at the top of the tested range, so the binding
  variant is not fixed across degrees and a narrower parametrization would have found a
  smaller ratio;
* 6.2x for biorthogonality at degree 8, which contracts the duals against a
  ``2 * degree + 2`` point rule, accumulating on top of the solve.

The residual is route-dependent at order-of-magnitude scale, so the margin depends on
which routine forms the inverse. Measured on the binding case, ratio of
``||B A - I||`` to ``cond(A) * eps``: ``scipy.linalg.inv`` 0.16,
``scipy.linalg.lu_factor``+``lu_solve`` 0.60, ``scipy.linalg.solve`` 0.60,
``numpy.linalg.solve`` (what this module uses) and ``numpy.linalg.inv`` 2.04 -- these two
bit-identical, so they are one entry point, not two -- and a QR-based inverse 4.17. The
worst LU route leaves 64 a margin of **31x**; including the QR route, which is a
different factorisation and outside the bound derived above, leaves **15x**. Quoting a
best-to-worst *spread* instead would be meaningless: it is driven by whichever route is
unusually good, and reads 2042x on the Bernstein/cardinal pair whose worst route is a
harmless 0.65.

The constant stays local rather than becoming ``pantr.tolerance.get_default(dtype) *
cond``, for two reasons: ``cond`` is an amplification factor, not "the magnitude of the
quantity being compared" that the presets are documented to be multiplied by, and
``pantr/tolerance.py`` explicitly exempts ``pantr.change_basis`` from the presets,
requiring it to report the accuracy it attains per degree. That the two factors both
equal 64 is a coincidence of two different derivations.
"""

_ROUNDOFF_FACTOR = 64
"""
Multiple of machine epsilon for quantities that carry no conditioning penalty.

Applies to the well-conditioned direction (Legendre to cardinal) and to the
partition-of-unity identity, both of which are built on the Legendre Gram matrix,
which is the identity. Measured errors are flat in degree at 2-4 ulp, so a fixed
multiple of epsilon is the right form; 64 leaves ~32x margin.
"""

_COMPOSED_ROUNDOFF_FACTOR = 256
"""
Multiple of machine epsilon for an identity chaining two changes of basis.

Used by the consistency check against the Bernstein pair, whose right-hand side is a
product of two separately-computed matrices. Measured error grows to 2.7e-15 at
degree 8, so the single-map factor of 64 would leave only 5x; 256 restores ~21x.
"""


def _conditioning_atol(
    forward: npt.NDArray[np.float32 | np.float64],
    dtype: npt.DTypeLike = np.float64,
) -> float:
    """Return the attainable accuracy for inverting a change-of-basis map.

    Every pair in this module builds one direction directly and obtains the other by
    an LU solve against the identity. The inverse direction therefore cannot do
    better than roughly ``cond(forward) * eps``, whatever the algorithm, and that
    quantity spans many orders of magnitude across the degrees tested (``1.1e3`` at
    degree 4 for the Legendre pair, ``3.0e8`` at degree 8). Tests of an inverse
    direction, of anything built from one, and of biorthogonality are all bounded by
    this rather than by a flat tolerance.

    Pass the **float64** matrix even when grading a float32 result: the condition
    number is a property of the two bases, not of the dtype the result is stored in,
    and estimating it in float32 would itself be conditioning-limited.

    Args:
        forward (npt.NDArray[np.float32 | np.float64]): The directly-built direction
            of the pair, whose conditioning bounds the inverse direction.
        dtype (npt.DTypeLike): Dtype whose machine epsilon sets the noise floor.
            Defaults to np.float64.

    Returns:
        float: Absolute tolerance for identities involving the inverse map.
    """
    cond = float(np.linalg.cond(np.asarray(forward, dtype=np.float64)))
    return _COND_SAFETY_FACTOR * cond * get_machine_epsilon(dtype)


def _assert_inverse_pair(
    forward: npt.NDArray[np.float32 | np.float64],
    inverse: npt.NDArray[np.float32 | np.float64],
    atol: float,
) -> None:
    """Assert both products of an inverse pair, each against what actually bounds it.

    The two directions are *not* the same claim, and grading them alike is what
    :data:`_COND_SAFETY_FACTOR` documents as the one empirical step in this file:

    * ``forward @ inverse`` is bounded by ``atol`` and that is a derived bound;
    * ``inverse @ forward`` has no bound of that tier. What it does satisfy, for any
      matrix, degree, dtype and LAPACK path, is ``||B A - I|| <= cond(A) ||A B - I||``,
      which follows from the exact identity ``B A - I = A^-1 (A B - I) A`` with no
      hypothesis at all. That inequality is asserted here as a theorem.

      It is a statement about the *exact* residuals, and both sides are measured in
      floating point, so the assertion carries an additive allowance for the
      measurement itself. Forming either product costs at most
      ``order * eps * ||A||_2 ||B||_2`` (the standard inner-product bound, with
      ``gamma_k ~ k * eps``), which can inflate the left-hand residual and deflate the
      right-hand one, giving
      ``left <= cond(A) * (right + e) + e`` with ``e`` that quantity. Without it the
      assertion compares two noise-level numbers at low degree, where both residuals
      are a few ulp and their ratio is meaningless: measured at degree 2 of the
      Bernstein/cardinal pair, ``5.8e-17`` against a naive right-hand side of
      ``2.2e-17``. The allowance is negligible where the claim has teeth -- at degree
      11 of the Lagrange pair it is ``8e-12`` against a bound of ``1.2e-9``.

    Holding the reversed product to ``atol`` as well is kept, because it is the
    sharper statement wherever it is true and it is what caught the defect ticket #300
    fixed. But it is an **acceptance threshold, not a tolerance** -- the same
    distinction ``test_acceptance_thresholds_of_ticket_300`` draws -- so it is asserted
    last and named as such, and it is false past degree 20 (see
    :data:`_COND_SAFETY_FACTOR`).

    What each assertion is worth, measured as how close it comes to firing (1.0 fires):

    * the derived forward bound, 0.0002 to 0.007;
    * the theorem, 0.0000 to 0.0065 -- it has four orders of slack and will not catch a
      numerical regression. That is not its job. It cannot fire spuriously at any
      degree, so it is what separates "the empirical tier stopped holding" from "the
      pair stopped being an inverse pair": past degree 20 the threshold below fails
      while this still passes, and the two verdicts together say which happened;
    * the threshold, 0.0009 to 0.032 -- the discriminating one, 31x from firing at the
      binding case.

    A caveat on what this cannot see. Building the inverse on the *wrong side*
    (``solve(A.T, I).T``) is caught only where conditioning is large enough to expose
    it: the forward residual then lands at 1.5e-4, 0.36 and 41 times ``atol`` at degrees
    2, 8 and 11 of the Bernstein/cardinal pair, so it passes below degree 11 and fails
    above. The primary defence against a parity slip is therefore the documentation on
    each builder, not this check.

    Args:
        forward (npt.NDArray[np.float32 | np.float64]): The directly-built direction.
        inverse (npt.NDArray[np.float32 | np.float64]): The LU-solved direction.
        atol (float): The derived bound from :func:`_conditioning_atol`.
    """
    a64 = np.asarray(forward, dtype=np.float64)
    b64 = np.asarray(inverse, dtype=np.float64)
    right_residual = _identity_residual(forward, inverse)
    left_residual = _identity_residual(inverse, forward)
    cond = float(np.linalg.cond(a64))
    order = forward.shape[0]

    assert right_residual <= atol, f"derived bound on forward @ inverse: {right_residual} > {atol}"

    # A theorem, not a measurement: sound at every degree, dtype and LAPACK path.
    # `measurement` is what forming either product can itself contribute.
    measurement = (
        order
        * get_machine_epsilon(np.float64)
        * float(np.linalg.norm(a64, 2))
        * float(np.linalg.norm(b64, 2))
    )
    derived_reversed = cond * (right_residual + measurement) + measurement
    assert left_residual <= derived_reversed, (
        f"cond(A) ||A B - I|| does not bound ||B A - I||: {left_residual} > {derived_reversed}"
    )

    # Acceptance threshold rather than a tolerance; see this function's docstring.
    assert left_residual <= atol, (
        f"reversed product above the observed tier: {left_residual} > {atol}"
    )


def _identity_residual(
    left: npt.NDArray[np.float32 | np.float64],
    right: npt.NDArray[np.float32 | np.float64],
) -> float:
    """Return the spectral norm of ``left @ right - I``.

    Both operands are promoted to float64 before the product so that the residual
    measures the error carried by the two matrices, not the error of the check
    itself. That matters for the float32 assertions, where a float32 product would
    add a rounding of its own at the very magnitude being graded.

    For a float64 pair there is no higher precision left to promote to, and the
    check is *not* free there either: forming ``left @ right`` costs about
    ``n * eps * cond``, which is the tier being graded. Measured on the
    Bernstein/cardinal forward product at degree 11, the ratio to ``cond(A) * eps``
    is 0.425 as computed here and 0.056 with the product formed in exact rational
    arithmetic, so at least 87% of the reported residual is this function's own
    rounding. The direction is safe -- a check can inflate a residual, never mask
    one -- so every margin quoted in :data:`_COND_SAFETY_FACTOR` is conservative.
    But the peak ratios quoted there are properties of this measurement, not of the
    matrices, and re-deriving the constant from them would build in that slack
    twice.

    Args:
        left (npt.NDArray[np.float32 | np.float64]): Left factor of the product.
        right (npt.NDArray[np.float32 | np.float64]): Right factor of the product.

    Returns:
        float: ``||left @ right - I||_2``, the quantity ticket #300 tabulates.
    """
    product = np.asarray(left, dtype=np.float64) @ np.asarray(right, dtype=np.float64)
    return float(np.linalg.norm(product - np.eye(product.shape[0]), ord=2))


class TestLagrangeToBernsteinBasisOperator:
    """Test the compute_lagrange_to_bernstein_1d function."""

    def test_degree_zero_error(self) -> None:
        """Test that degree lower than 1 raises ValueError."""
        with pytest.raises(ValueError, match="Degree must at least 1"):
            compute_lagrange_to_bernstein_1d(0)

    def test_negative_degree_error(self) -> None:
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="Degree must at least 1"):
            compute_lagrange_to_bernstein_1d(-1)

    def test_invalid_dtype_error(self) -> None:
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_lagrange_to_bernstein_1d(2, dtype=np.int32)
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_lagrange_to_bernstein_1d(2, dtype=np.float16)

    def test_string_dtype_accepted(self) -> None:
        """String dtype aliases are accepted via np.dtype() normalisation."""
        result32 = compute_lagrange_to_bernstein_1d(2, dtype="float32")
        assert result32.dtype == np.float32
        result64 = compute_lagrange_to_bernstein_1d(2, dtype="float64")
        assert result64.dtype == np.float64

    def test_out_parameter(self) -> None:
        """Test that out parameter works correctly."""
        degree = 2
        variant = LagrangeVariant.EQUISPACES

        # Test with None (default)
        result1 = compute_lagrange_to_bernstein_1d(degree, variant)
        assert result1.shape == (degree + 1, degree + 1)
        assert result1.dtype == np.float64

        # Test with provided out array (correct shape and dtype)
        out = np.empty((degree + 1, degree + 1), dtype=np.float64)
        result2 = compute_lagrange_to_bernstein_1d(degree, variant, out=out)
        assert result2 is out
        np.testing.assert_array_almost_equal(result1, result2)

        # Test with float32
        out_f32 = np.empty((degree + 1, degree + 1), dtype=np.float32)
        result3 = compute_lagrange_to_bernstein_1d(degree, variant, dtype=np.float32, out=out_f32)
        assert result3 is out_f32
        assert result3.dtype == np.float32

    def test_out_parameter_validation(self) -> None:
        """Test that out parameter validation works correctly."""
        degree = 2
        variant = LagrangeVariant.EQUISPACES

        # Wrong shape
        out_wrong_shape = np.empty((degree + 2, degree + 1), dtype=np.float64)
        with pytest.raises(ValueError, match="Output array has shape"):
            compute_lagrange_to_bernstein_1d(degree, variant, out=out_wrong_shape)

        # Wrong dtype
        out_wrong_dtype = np.empty((degree + 1, degree + 1), dtype=np.float32)
        with pytest.raises(ValueError, match="Output array has dtype"):
            compute_lagrange_to_bernstein_1d(degree, variant, out=out_wrong_dtype)

        # Not writeable
        out_readonly = np.empty((degree + 1, degree + 1), dtype=np.float64)
        out_readonly.setflags(write=False)
        with pytest.raises(ValueError, match="Output array is not writeable"):
            compute_lagrange_to_bernstein_1d(degree, variant, out=out_readonly)


class TestBernsteinToLagrangeBasisOperator:
    """Test the create_Bernstein_to_Lagrange_basis function."""

    def test_degree_zero_error(self) -> None:
        """Test that degree lower than 1 raises ValueError."""
        with pytest.raises(ValueError, match="Degree must at least 1"):
            compute_bernstein_to_lagrange_1d(0)

    def test_negative_degree_error(self) -> None:
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="Degree must at least 1"):
            compute_bernstein_to_lagrange_1d(-1)

    def test_invalid_dtype_error(self) -> None:
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_bernstein_to_lagrange_1d(2, dtype=np.int32)
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_bernstein_to_lagrange_1d(2, dtype=np.float16)

    def test_string_dtype_accepted(self) -> None:
        """String dtype aliases are accepted via np.dtype() normalisation."""
        result32 = compute_bernstein_to_lagrange_1d(2, dtype="float32")
        assert result32.dtype == np.float32
        result64 = compute_bernstein_to_lagrange_1d(2, dtype="float64")
        assert result64.dtype == np.float64

    def test_out_parameter(self) -> None:
        """Test that out parameter works correctly."""
        degree = 2
        variant = LagrangeVariant.EQUISPACES

        # Test with None (default)
        result1 = compute_bernstein_to_lagrange_1d(degree, variant)
        assert result1.shape == (degree + 1, degree + 1)
        assert result1.dtype == np.float64

        # Test with provided out array (correct shape and dtype)
        out = np.empty((degree + 1, degree + 1), dtype=np.float64)
        result2 = compute_bernstein_to_lagrange_1d(degree, variant, out=out)
        assert result2 is out
        np.testing.assert_array_almost_equal(result1, result2)

        # Test with float32
        out_f32 = np.empty((degree + 1, degree + 1), dtype=np.float32)
        result3 = compute_bernstein_to_lagrange_1d(degree, variant, dtype=np.float32, out=out_f32)
        assert result3 is out_f32
        assert result3.dtype == np.float32

    def test_out_parameter_validation(self) -> None:
        """Test that out parameter validation works correctly."""
        degree = 2
        variant = LagrangeVariant.EQUISPACES

        # Wrong shape
        out_wrong_shape = np.empty((degree + 2, degree + 1), dtype=np.float64)
        with pytest.raises(ValueError, match="Output array has shape"):
            compute_bernstein_to_lagrange_1d(degree, variant, out=out_wrong_shape)

        # Wrong dtype
        out_wrong_dtype = np.empty((degree + 1, degree + 1), dtype=np.float32)
        with pytest.raises(ValueError, match="Output array has dtype"):
            compute_bernstein_to_lagrange_1d(degree, variant, out=out_wrong_dtype)

        # Not writeable
        out_readonly = np.empty((degree + 1, degree + 1), dtype=np.float64)
        out_readonly.setflags(write=False)
        with pytest.raises(ValueError, match="Output array is not writeable"):
            compute_bernstein_to_lagrange_1d(degree, variant, out=out_readonly)

    def test_inverse_relationship(self) -> None:
        """Test that Bernstein to Lagrange is inverse of Lagrange to Bernstein."""
        degree = 2
        variant = LagrangeVariant.EQUISPACES
        lagrange_to_bernstein = compute_lagrange_to_bernstein_1d(degree, variant)
        bernstein_to_lagrange = compute_bernstein_to_lagrange_1d(degree, variant)

        # Should be inverse matrices
        identity = lagrange_to_bernstein @ bernstein_to_lagrange
        np.testing.assert_array_almost_equal(identity, np.eye(degree + 1))

    @pytest.mark.parametrize(
        "variant",
        [
            LagrangeVariant.EQUISPACES,
            LagrangeVariant.GAUSS_LEGENDRE,
            LagrangeVariant.GAUSS_LOBATTO_LEGENDRE,
            LagrangeVariant.CHEBYSHEV_1ST,
            LagrangeVariant.CHEBYSHEV_2ND,
        ],
    )
    @pytest.mark.parametrize("degree", [1, 2, 3, 4])
    def test_values(self, degree: int, variant: LagrangeVariant) -> None:
        """Test Bernstein evaluations transformed with the operator return Lagrange evaluations."""
        n_pts = 10
        tt = np.linspace(0.0, 1.0, n_pts)

        bernsteins = tabulate_bernstein_1d(degree, tt)
        lagranges = tabulate_lagrange_1d(degree, variant, tt)

        C = compute_bernstein_to_lagrange_1d(degree, variant)
        np.testing.assert_array_almost_equal(bernsteins @ C.T, lagranges)

        C_inv = compute_lagrange_to_bernstein_1d(degree, variant)
        np.testing.assert_array_almost_equal(lagranges @ C_inv.T, bernsteins)

    @pytest.mark.parametrize(
        "variant",
        [
            LagrangeVariant.EQUISPACES,
            LagrangeVariant.GAUSS_LEGENDRE,
            LagrangeVariant.GAUSS_LOBATTO_LEGENDRE,
            LagrangeVariant.CHEBYSHEV_1ST,
            LagrangeVariant.CHEBYSHEV_2ND,
        ],
    )
    @pytest.mark.parametrize("degree", [1, 4, 7, 10, 11])
    def test_roundtrip_tracks_conditioning(self, degree: int, variant: LagrangeVariant) -> None:
        """Both products of the pair must be the identity to ``K * cond(C) * eps``.

        The same derived bound the Bernstein<->cardinal and Legendre<->cardinal pairs
        are held to, so all three inverse pairs in this module are graded alike. This
        pair is far better conditioned than the other two -- ``cond(C)`` reaches only
        ``3.0e3`` at degree 10 with the worst (equispaced) nodes -- so the bound stays
        near the noise floor and the test pins that rather than rescuing it.
        """
        forward = compute_lagrange_to_bernstein_1d(degree, variant)
        inverse = compute_bernstein_to_lagrange_1d(degree, variant)
        atol = _conditioning_atol(forward)

        _assert_inverse_pair(forward, inverse, atol)

    @pytest.mark.parametrize(
        "variant",
        [
            LagrangeVariant.EQUISPACES,
            LagrangeVariant.GAUSS_LEGENDRE,
            LagrangeVariant.GAUSS_LOBATTO_LEGENDRE,
            LagrangeVariant.CHEBYSHEV_1ST,
            LagrangeVariant.CHEBYSHEV_2ND,
        ],
    )
    @pytest.mark.parametrize("degree", [1, 4, 7, 10])
    def test_roundtrip_float32(self, degree: int, variant: LagrangeVariant) -> None:
        """The pair round-trips in float32 across every node family.

        This is the test behind the docstring claim that, alone among the three pairs,
        this one stays usable in single precision over the whole range of practical
        degrees. Degree 10 is inside every variant's certified range: the derived bound
        stays below 0.1 through degree 11 for equispaced nodes and through degree 14 for
        the Gauss and Chebyshev families, so no degree tested here is a vacuous claim.
        """
        forward = compute_lagrange_to_bernstein_1d(degree, variant, dtype=np.float32)
        inverse = compute_bernstein_to_lagrange_1d(degree, variant, dtype=np.float32)
        assert forward.dtype == np.float32
        assert inverse.dtype == np.float32

        atol = _conditioning_atol(compute_lagrange_to_bernstein_1d(degree, variant), np.float32)
        _assert_inverse_pair(forward, inverse, atol)


class TestCardinalToBernsteinRoundTrip:
    """Pin the Bernstein<->cardinal pair to the accuracy its conditioning allows.

    Regression tests for ticket #300: the two directions used to be independent Gram
    projections, and the inverse one solved against the *cardinal* Gram matrix, whose
    condition number is ``cond(A)**2`` times the Bernstein Gram matrix's. The round
    trip therefore tracked ``cond(A)**2 * eps`` instead of ``cond(A) * eps`` and had
    no correct digits at all by degree 9.
    """

    @pytest.mark.parametrize("degree", range(12))
    def test_roundtrip_tracks_conditioning_not_its_square(self, degree: int) -> None:
        """Both products of the pair must be the identity to ``K * cond(A) * eps``.

        This is the ticket's discriminating measurement. From degree 5 on, the bound
        sits below ``cond(A)**2 * eps`` by more than the safety factor, so passing it
        rules out the squared law rather than merely tolerating problem conditioning:
        at degree 8 the bound is ``2.8e-8`` while ``cond(A)**2 * eps`` is ``8.4e-4``.
        """
        forward = compute_bernstein_to_cardinal_1d(degree)
        inverse = compute_cardinal_to_bernstein_1d(degree)
        atol = _conditioning_atol(forward)

        _assert_inverse_pair(forward, inverse, atol)

    @pytest.mark.parametrize("degree", range(12))
    def test_inverse_map_reproduces_bernstein_basis(self, degree: int) -> None:
        """``B @ cardinal(x)`` must equal ``bernstein(x)`` at arbitrary points.

        The property the matrix is defined by, and the one a caller moving
        coefficients between the two representations actually depends on. Checked away
        from the quadrature nodes used to build the pair, so a rule that happened to be
        exact only at its own nodes would not pass. The bound is the same
        conditioning-limited one: the map amplifies the ``O(1)`` cardinal values by
        ``||A^-1||`` and cancels back down to ``O(1)``. Strictly it is the *reversed*
        product that governs here, since ``B @ cardinal = B A @ bernstein``; see
        ``_COND_SAFETY_FACTOR`` on why that tier is observed rather than derived.

        ``rtol=0`` is required: the default ``rtol=1e-7`` would dominate the derived
        ``atol`` on every entry of magnitude near 1, leaving the largest Bernstein
        values graded a hundred million times more loosely than intended.
        """
        rng = np.random.default_rng(20260817 + degree)
        pts = rng.random(200)

        result = (
            compute_cardinal_to_bernstein_1d(degree) @ tabulate_cardinal_bspline_1d(degree, pts).T
        )

        np.testing.assert_allclose(
            result,
            tabulate_bernstein_1d(degree, pts).T,
            rtol=0,
            atol=_conditioning_atol(compute_bernstein_to_cardinal_1d(degree)),
        )

    def test_acceptance_thresholds_of_ticket_300(self) -> None:
        """Pin the two round-trip numbers ticket #300 accepts the fix against.

        These two literals are the ticket's acceptance thresholds, not tolerances: they
        record the fitness of one configuration rather than bounding an algorithm, which
        is why they are stated rather than derived. They are *tighter* than
        ``_COND_SAFETY_FACTOR * cond(A) * eps`` (``2.8e-8`` at degree 8, ``4.7e-7`` at
        degree 9), so the derived test above cannot enforce them and this one does.

        Margins are thin by the standards of this file -- 7.7x at degree 8, where the
        observed residual moves by a factor of 3.8 across LAPACK entry points -- so a
        failure here is a signal to re-measure, not necessarily a regression.
        """
        for degree, threshold in ((8, 1e-9), (9, 1e-8)):
            residual = _identity_residual(
                compute_bernstein_to_cardinal_1d(degree),
                compute_cardinal_to_bernstein_1d(degree),
            )
            assert residual <= threshold, f"degree {degree}: {residual:.3e} > {threshold:.0e}"

    @pytest.mark.parametrize("degree", range(7))
    def test_roundtrip_float32(self, degree: int) -> None:
        """The pair round-trips in float32 over the degrees where the bound is meaningful.

        From degree 7 on the derived bound ``_COND_SAFETY_FACTOR * cond(A) * eps32``
        exceeds 0.1 (``0.98`` at degree 7 against ``0.076`` at degree 6) and the
        assertion would be vacuous, so those degrees are deliberately not claimed.
        """
        forward = compute_bernstein_to_cardinal_1d(degree, dtype=np.float32)
        inverse = compute_cardinal_to_bernstein_1d(degree, dtype=np.float32)
        assert forward.dtype == np.float32
        assert inverse.dtype == np.float32

        atol = _conditioning_atol(compute_bernstein_to_cardinal_1d(degree), np.float32)
        assert _identity_residual(forward, inverse) <= atol

    @pytest.mark.parametrize("degree", range(12))
    def test_cached_matrix_round_trips_like_the_builder(self, degree: int) -> None:
        """The cached hot-path matrix must be the same inverse the builder returns.

        The extraction operators consume the cached copy rather than the builder, so
        the accuracy pinned above has to reach them unchanged.
        """
        cached = _cached_cardinal_to_bernstein_matrix(degree, np.dtype(np.float64))
        forward = compute_bernstein_to_cardinal_1d(degree)

        np.testing.assert_array_equal(cached, compute_cardinal_to_bernstein_1d(degree))
        assert _identity_residual(forward, cached) <= _conditioning_atol(forward)


class TestCardinalToBernsteinBasisOperator:
    """Test the create_cardinal_to_Bernstein_basis function."""

    def test_inverse_relationship(self) -> None:
        """Test that cardinal to Bernstein is inverse of Bernstein to cardinal."""
        degree = 2
        bernstein_to_cardinal = compute_bernstein_to_cardinal_1d(degree)
        cardinal_to_bernstein = compute_cardinal_to_bernstein_1d(degree)

        # Should be inverse matrices
        identity = bernstein_to_cardinal @ cardinal_to_bernstein
        np.testing.assert_array_almost_equal(identity, np.eye(degree + 1))

    def test_negative_degree_error(self) -> None:
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="Degree must be non-negative"):
            compute_cardinal_to_bernstein_1d(-1)
        with pytest.raises(ValueError, match="Degree must be non-negative"):
            compute_bernstein_to_cardinal_1d(-1)

    def test_invalid_dtype_error(self) -> None:
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_bernstein_to_cardinal_1d(2, dtype=np.int32)
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_bernstein_to_cardinal_1d(2, dtype=np.float16)
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_cardinal_to_bernstein_1d(2, dtype=np.int32)
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_cardinal_to_bernstein_1d(2, dtype=np.float16)

    def test_string_dtype_accepted(self) -> None:
        """String dtype aliases are accepted via np.dtype() normalisation."""
        for fn in (compute_bernstein_to_cardinal_1d, compute_cardinal_to_bernstein_1d):
            result32 = fn(2, dtype="float32")
            assert result32.dtype == np.float32
            result64 = fn(2, dtype="float64")
            assert result64.dtype == np.float64

    def test_values(self) -> None:
        """Test that cardinal evaluations transformed with operator return Bernstein evaluations."""
        for degree in [1, 2, 3, 4]:
            n_pts = 10
            tt = np.linspace(0.0, 1.0, n_pts)
            bernsteins = tabulate_bernstein_1d(degree, tt)
            cardinals = tabulate_cardinal_bspline_1d(degree, tt)

            C = compute_cardinal_to_bernstein_1d(degree)
            np.testing.assert_array_almost_equal(bernsteins, cardinals @ C.T)

            C_inv = compute_bernstein_to_cardinal_1d(degree)
            np.testing.assert_array_almost_equal(cardinals, bernsteins @ C_inv.T)

    def test_out_parameter(self) -> None:
        """Test that out parameter works correctly for cardinal-Bernstein functions."""
        degree = 2

        # Test compute_bernstein_to_cardinal_1d
        result1 = compute_bernstein_to_cardinal_1d(degree)
        assert result1.shape == (degree + 1, degree + 1)
        assert result1.dtype == np.float64

        out = np.empty((degree + 1, degree + 1), dtype=np.float64)
        result2 = compute_bernstein_to_cardinal_1d(degree, out=out)
        assert result2 is out
        np.testing.assert_array_almost_equal(result1, result2)

        # Test compute_cardinal_to_bernstein_1d
        result3 = compute_cardinal_to_bernstein_1d(degree)
        assert result3.shape == (degree + 1, degree + 1)
        assert result3.dtype == np.float64

        out2 = np.empty((degree + 1, degree + 1), dtype=np.float64)
        result4 = compute_cardinal_to_bernstein_1d(degree, out=out2)
        assert result4 is out2
        np.testing.assert_array_almost_equal(result3, result4)

        # Test with float32
        out_f32 = np.empty((degree + 1, degree + 1), dtype=np.float32)
        result5 = compute_bernstein_to_cardinal_1d(degree, dtype=np.float32, out=out_f32)
        assert result5 is out_f32
        assert result5.dtype == np.float32

    def test_out_parameter_validation(self) -> None:
        """Test that out parameter validation works correctly for cardinal-Bernstein functions."""
        degree = 2

        # Wrong shape
        out_wrong_shape = np.empty((degree + 2, degree + 1), dtype=np.float64)
        with pytest.raises(ValueError, match="Output array has shape"):
            compute_bernstein_to_cardinal_1d(degree, out=out_wrong_shape)
        with pytest.raises(ValueError, match="Output array has shape"):
            compute_cardinal_to_bernstein_1d(degree, out=out_wrong_shape)

        # Wrong dtype
        out_wrong_dtype = np.empty((degree + 1, degree + 1), dtype=np.float32)
        with pytest.raises(ValueError, match="Output array has dtype"):
            compute_bernstein_to_cardinal_1d(degree, out=out_wrong_dtype)
        with pytest.raises(ValueError, match="Output array has dtype"):
            compute_cardinal_to_bernstein_1d(degree, out=out_wrong_dtype)

        # Not writeable
        out_readonly = np.empty((degree + 1, degree + 1), dtype=np.float64)
        out_readonly.setflags(write=False)
        with pytest.raises(ValueError, match="Output array is not writeable"):
            compute_bernstein_to_cardinal_1d(degree, out=out_readonly)
        out_readonly2 = np.empty((degree + 1, degree + 1), dtype=np.float64)
        out_readonly2.setflags(write=False)
        with pytest.raises(ValueError, match="Output array is not writeable"):
            compute_cardinal_to_bernstein_1d(degree, out=out_readonly2)


class TestCreateChangeBasis:
    """Test the _create_change_basis private function."""

    def test_invalid_n_quad_pts_error(self) -> None:
        """Test that non-positive n_quad_pts raises ValueError."""
        degree = 2

        def bernstein(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return tabulate_bernstein_1d(degree, pts)

        def cardinal(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return tabulate_cardinal_bspline_1d(degree, pts)

        with pytest.raises(ValueError, match="Number of quadrature points must be positive"):
            _compute_change_basis_1D(bernstein, cardinal, n_quad_pts=0)

        with pytest.raises(ValueError, match="Number of quadrature points must be positive"):
            _compute_change_basis_1D(bernstein, cardinal, n_quad_pts=-1)

    def test_invalid_dtype_error(self) -> None:
        """Test that invalid dtype raises ValueError."""
        degree = 2

        def bernstein(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return tabulate_bernstein_1d(degree, pts)

        def cardinal(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return tabulate_cardinal_bspline_1d(degree, pts)

        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            _compute_change_basis_1D(bernstein, cardinal, n_quad_pts=3, dtype=np.int32)

        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            _compute_change_basis_1D(bernstein, cardinal, n_quad_pts=3, dtype=np.float16)

    def test_string_dtype_accepted(self) -> None:
        """String dtype aliases are accepted via np.dtype() normalisation."""
        degree = 2
        result32 = _compute_change_basis_1D(
            functools.partial(tabulate_bernstein_1d, degree),
            functools.partial(tabulate_cardinal_bspline_1d, degree),
            n_quad_pts=degree + 1,
            dtype="float32",
        )
        assert result32.dtype == np.float32
        result64 = _compute_change_basis_1D(
            functools.partial(tabulate_bernstein_1d, degree),
            functools.partial(tabulate_cardinal_bspline_1d, degree),
            n_quad_pts=degree + 1,
            dtype="float64",
        )
        assert result64.dtype == np.float64

    def test_out_parameter(self) -> None:
        """Test that out parameter works correctly for _compute_change_basis_1D."""
        degree = 2

        def bernstein(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return tabulate_bernstein_1d(degree, pts)

        def cardinal(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return tabulate_cardinal_bspline_1d(degree, pts)

        # Test with None (default)
        result1 = _compute_change_basis_1D(bernstein, cardinal, n_quad_pts=degree + 1)
        assert result1.shape == (degree + 1, degree + 1)
        assert result1.dtype == np.float64

        # Test with provided out array (correct shape and dtype)
        out = np.empty((degree + 1, degree + 1), dtype=np.float64)
        result2 = _compute_change_basis_1D(bernstein, cardinal, n_quad_pts=degree + 1, out=out)
        assert result2 is out
        np.testing.assert_array_almost_equal(result1, result2)

        # Test with float32
        out_f32 = np.empty((degree + 1, degree + 1), dtype=np.float32)
        result3 = _compute_change_basis_1D(
            bernstein, cardinal, n_quad_pts=degree + 1, dtype=np.float32, out=out_f32
        )
        assert result3 is out_f32
        assert result3.dtype == np.float32

    def test_out_parameter_validation(self) -> None:
        """Test that out parameter validation works correctly for _compute_change_basis_1D."""
        degree = 2

        def bernstein(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return tabulate_bernstein_1d(degree, pts)

        def cardinal(
            pts: npt.NDArray[np.float32 | np.float64],
        ) -> npt.NDArray[np.float32 | np.float64]:
            return tabulate_cardinal_bspline_1d(degree, pts)

        # Wrong shape
        out_wrong_shape = np.empty((degree + 2, degree + 1), dtype=np.float64)
        with pytest.raises(ValueError, match="Output array has shape"):
            _compute_change_basis_1D(
                bernstein, cardinal, n_quad_pts=degree + 1, out=out_wrong_shape
            )

        # Wrong dtype
        out_wrong_dtype = np.empty((degree + 1, degree + 1), dtype=np.float32)
        with pytest.raises(ValueError, match="Output array has dtype"):
            _compute_change_basis_1D(
                bernstein, cardinal, n_quad_pts=degree + 1, out=out_wrong_dtype
            )

        # Not writeable
        out_readonly = np.empty((degree + 1, degree + 1), dtype=np.float64)
        out_readonly.setflags(write=False)
        with pytest.raises(ValueError, match="Output array is not writeable"):
            _compute_change_basis_1D(bernstein, cardinal, n_quad_pts=degree + 1, out=out_readonly)


class TestCachedChangeBasisMatrices:
    """Tests for the LRU-cached change-of-basis helpers."""

    def test_lagrange_cached_values_match_uncached(self) -> None:
        """Cached matrix must be numerically identical to the uncached reference."""
        degree = 3
        variant = LagrangeVariant.GAUSS_LOBATTO_LEGENDRE
        dtype = np.dtype(np.float64)
        expected = compute_lagrange_to_bernstein_1d(degree, variant, dtype)
        cached = _cached_lagrange_to_bernstein_matrix(degree, variant, dtype)
        np.testing.assert_array_equal(cached, expected)

    def test_cardinal_cached_values_match_uncached(self) -> None:
        """Cached matrix must be numerically identical to the uncached reference."""
        degree = 3
        dtype = np.dtype(np.float64)
        expected = compute_cardinal_to_bernstein_1d(degree, dtype)
        cached = _cached_cardinal_to_bernstein_matrix(degree, dtype)
        np.testing.assert_array_equal(cached, expected)

    def test_lagrange_cached_array_is_readonly(self) -> None:
        """The cached array must be read-only to prevent mutation of the shared copy."""
        degree = 2
        dtype = np.dtype(np.float32)
        mat = _cached_lagrange_to_bernstein_matrix(degree, LagrangeVariant.EQUISPACES, dtype)
        assert not mat.flags.writeable
        with pytest.raises((ValueError, TypeError)):
            mat[0, 0] = 0.0

    def test_cardinal_cached_array_is_readonly(self) -> None:
        """The cached array must be read-only to prevent mutation of the shared copy."""
        degree = 2
        dtype = np.dtype(np.float64)
        mat = _cached_cardinal_to_bernstein_matrix(degree, dtype)
        assert not mat.flags.writeable
        with pytest.raises((ValueError, TypeError)):
            mat[0, 0] = 0.0

    def test_lagrange_cache_returns_same_object(self) -> None:
        """Repeated calls with identical arguments must return the exact same object."""
        degree = 2
        variant = LagrangeVariant.EQUISPACES
        dtype = np.dtype(np.float64)
        mat1 = _cached_lagrange_to_bernstein_matrix(degree, variant, dtype)
        mat2 = _cached_lagrange_to_bernstein_matrix(degree, variant, dtype)
        assert mat1 is mat2

    def test_cardinal_cache_returns_same_object(self) -> None:
        """Repeated calls with identical arguments must return the exact same object."""
        degree = 3
        dtype = np.dtype(np.float64)
        mat1 = _cached_cardinal_to_bernstein_matrix(degree, dtype)
        mat2 = _cached_cardinal_to_bernstein_matrix(degree, dtype)
        assert mat1 is mat2

    def test_lagrange_cache_is_bounded(self) -> None:
        """The Lagrange cache must have a finite maxsize."""
        info = _cached_lagrange_to_bernstein_matrix.cache_info()
        assert info.maxsize is not None
        assert info.maxsize > 0

    def test_cardinal_cache_is_bounded(self) -> None:
        """The cardinal cache must have a finite maxsize."""
        info = _cached_cardinal_to_bernstein_matrix.cache_info()
        assert info.maxsize is not None
        assert info.maxsize > 0


class TestMonomialToBernsteinBasisOperator:
    """Test the compute_monomial_to_bernstein_1d function."""

    def test_negative_degree_error(self) -> None:
        """Test that negative degree raises ValueError."""
        with pytest.raises(ValueError, match="Degree must be non-negative"):
            compute_monomial_to_bernstein_1d(-1)

    def test_invalid_dtype_error(self) -> None:
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_monomial_to_bernstein_1d(2, dtype=np.int32)
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            compute_monomial_to_bernstein_1d(2, dtype=np.float16)

    def test_string_dtype_accepted(self) -> None:
        """String dtype aliases are accepted via np.dtype() normalisation."""
        result32 = compute_monomial_to_bernstein_1d(2, dtype="float32")
        assert result32.dtype == np.float32
        result64 = compute_monomial_to_bernstein_1d(2, dtype="float64")
        assert result64.dtype == np.float64

    def test_out_parameter(self) -> None:
        """Test that out parameter works correctly."""
        degree = 3

        result1 = compute_monomial_to_bernstein_1d(degree)
        assert result1.shape == (degree + 1, degree + 1)
        assert result1.dtype == np.float64

        out = np.empty((degree + 1, degree + 1), dtype=np.float64)
        result2 = compute_monomial_to_bernstein_1d(degree, out=out)
        assert result2 is out
        np.testing.assert_array_equal(result1, result2)

        out_f32 = np.empty((degree + 1, degree + 1), dtype=np.float32)
        result3 = compute_monomial_to_bernstein_1d(degree, dtype=np.float32, out=out_f32)
        assert result3 is out_f32
        assert result3.dtype == np.float32

    def test_out_parameter_validation(self) -> None:
        """Test that out parameter validation works correctly."""
        degree = 2

        out_wrong_shape = np.empty((degree + 2, degree + 1), dtype=np.float64)
        with pytest.raises(ValueError, match="Output array has shape"):
            compute_monomial_to_bernstein_1d(degree, out=out_wrong_shape)

        out_wrong_dtype = np.empty((degree + 1, degree + 1), dtype=np.float32)
        with pytest.raises(ValueError, match="Output array has dtype"):
            compute_monomial_to_bernstein_1d(degree, out=out_wrong_dtype)

        out_readonly = np.empty((degree + 1, degree + 1), dtype=np.float64)
        out_readonly.setflags(write=False)
        with pytest.raises(ValueError, match="Output array is not writeable"):
            compute_monomial_to_bernstein_1d(degree, out=out_readonly)

    def test_out_parameter_pre_populated_is_zeroed(self) -> None:
        """A caller-provided out array full of ones is zeroed before fill."""
        degree = 2
        out = np.ones((degree + 1, degree + 1), dtype=np.float64)
        result = compute_monomial_to_bernstein_1d(degree, out=out)
        assert result is out
        # Upper triangle must be zero regardless of initial content.
        assert result[0, 1] == 0.0
        assert result[0, 2] == 0.0
        assert result[1, 2] == 0.0

    def test_degree_zero(self) -> None:
        """Degree 0 returns the 1x1 identity."""
        mat = compute_monomial_to_bernstein_1d(0)
        np.testing.assert_array_equal(mat, np.array([[1.0]]))

    def test_known_values_degree_2(self) -> None:
        """Degree 2: 1 = B0+B1+B2, t = (1/2)B1 + B2, t^2 = B2."""
        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        np.testing.assert_array_almost_equal(compute_monomial_to_bernstein_1d(2), expected)

    def test_known_values_degree_3(self) -> None:
        """Degree 3 reference values: M[i, j] = C(i, j) / C(3, j)."""
        expected = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0 / 3.0, 0.0, 0.0],
                [1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
        np.testing.assert_array_almost_equal(compute_monomial_to_bernstein_1d(3), expected)

    @pytest.mark.parametrize("degree", [1, 2, 3, 4, 5, 6])
    def test_polynomial_reconstruction(self, degree: int) -> None:
        """Bernstein coefficients from the matrix must reproduce the monomial polynomial."""
        rng = np.random.default_rng(degree)
        mono = rng.standard_normal(degree + 1)

        bern_coeffs = compute_monomial_to_bernstein_1d(degree) @ mono

        tt = np.linspace(0.0, 1.0, 25)
        p_mono = sum(mono[k] * tt**k for k in range(degree + 1))
        p_bern = tabulate_bernstein_1d(degree, tt) @ bern_coeffs

        np.testing.assert_array_almost_equal(p_bern, p_mono)


class TestLegendreCardinalBasisOperators:
    """Test the Legendre<->cardinal pair and the cardinal-dual Legendre coefficients."""

    def test_degree_one_closed_form(self) -> None:
        """Degree 1 matches the closed form obtained by integrating by hand.

        With ``B_0 = 1 - x``, ``B_1 = x`` and the orthonormal shifted Legendre pair
        ``p0 = 1``, ``p1 = sqrt(3) (2x - 1)``, the coefficient of ``p1`` in ``B_0``
        is ``int_0^1 (1-x) sqrt(3) (2x-1) dx = -sqrt(3)/6 = -1/(2 sqrt(3))``.
        """
        expected = np.array(
            [
                [0.5, -1.0 / (2.0 * np.sqrt(3.0))],
                [0.5, 1.0 / (2.0 * np.sqrt(3.0))],
            ]
        )
        np.testing.assert_allclose(compute_legendre_to_cardinal_1d(1), expected, rtol=0, atol=1e-15)

    @pytest.mark.parametrize("degree", range(9))
    def test_forward_map_reproduces_cardinal_basis(self, degree: int) -> None:
        """``A @ legendre(x)`` must equal ``cardinal(x)`` at arbitrary points, not just nodes.

        This is the property the matrix is defined by, checked away from the
        quadrature points used to build it, so a rule that happened to be exact only
        at its own nodes would not pass.
        """
        rng = np.random.default_rng(20260813 + degree)
        pts = rng.random(50)

        result = compute_legendre_to_cardinal_1d(degree) @ tabulate_legendre_1d(degree, pts).T

        np.testing.assert_allclose(
            result,
            tabulate_cardinal_bspline_1d(degree, pts).T,
            rtol=0,
            atol=_ROUNDOFF_FACTOR * get_machine_epsilon(np.float64),
        )

    @pytest.mark.parametrize("degree", range(9))
    def test_roundtrip(self, degree: int) -> None:
        """Both products of the pair must be the identity, to the attainable accuracy.

        The tolerance is derived rather than flat: the inverse direction cannot beat
        ``cond(A) * eps``, which spans ten orders of magnitude across these degrees.
        """
        forward = compute_legendre_to_cardinal_1d(degree)
        inverse = compute_cardinal_to_legendre_1d(degree)
        identity = np.eye(degree + 1)
        atol = _conditioning_atol(forward)

        np.testing.assert_allclose(inverse @ forward, identity, rtol=0, atol=atol)
        np.testing.assert_allclose(forward @ inverse, identity, rtol=0, atol=atol)

    @pytest.mark.parametrize("degree", range(5))
    def test_roundtrip_float32(self, degree: int) -> None:
        """The pair round-trips in float32 over the degrees where the bound is meaningful.

        From degree 5 on the derived bound ``_COND_SAFETY_FACTOR * cond(A) * eps32``
        exceeds 0.1 and the assertion would be vacuous, so those degrees are
        deliberately not claimed.
        """
        forward = compute_legendre_to_cardinal_1d(degree, dtype=np.float32)
        inverse = compute_cardinal_to_legendre_1d(degree, dtype=np.float32)
        assert forward.dtype == np.float32
        assert inverse.dtype == np.float32

        np.testing.assert_allclose(
            inverse @ forward,
            np.eye(degree + 1, dtype=np.float32),
            rtol=0,
            atol=_conditioning_atol(compute_legendre_to_cardinal_1d(degree), np.float32),
        )

    @pytest.mark.parametrize("degree", range(9))
    def test_partition_of_unity(self, degree: int) -> None:
        """Column sums of ``A`` must be ``[1, 0, ..., 0]``.

        The cardinal basis sums to 1, which is exactly the first orthonormal
        Legendre polynomial, so every higher coefficient must cancel across the
        rows. This direction carries no conditioning penalty.
        """
        expected = np.zeros(degree + 1)
        expected[0] = 1.0

        np.testing.assert_allclose(
            compute_legendre_to_cardinal_1d(degree).sum(axis=0),
            expected,
            rtol=0,
            atol=_ROUNDOFF_FACTOR * get_machine_epsilon(np.float64),
        )

    @pytest.mark.parametrize("degree", range(9))
    def test_dual_biorthogonality(self, degree: int) -> None:
        """The duals must satisfy ``int D_i B_j = delta_ij``, by independent quadrature.

        Integrated with a ``2 * degree + 2`` point Gauss-Legendre rule, which is
        exact for the degree-``2 * degree`` products and is not the rule used to
        build the matrices.
        """
        duals = compute_cardinal_dual_legendre_coeffs_1d(degree)
        pts, weights = get_gauss_legendre_1d(2 * degree + 2)
        cardinal = tabulate_cardinal_bspline_1d(degree, pts)
        legendre = tabulate_legendre_1d(degree, pts)

        gram = (duals @ (legendre.T * weights)) @ cardinal

        np.testing.assert_allclose(
            gram,
            np.eye(degree + 1),
            rtol=0,
            atol=_conditioning_atol(compute_legendre_to_cardinal_1d(degree)),
        )

    @pytest.mark.parametrize("degree", range(9))
    def test_dual_integrates_to_one(self, degree: int) -> None:
        """Every dual must integrate to 1 over the unit interval.

        Summing biorthogonality over ``j`` gives ``int D_i (sum_j B_j) = 1``, and the
        cardinal basis is a partition of unity, so ``int D_i = 1``. Because the
        Legendre basis is orthonormal that integral is the ``p0`` coefficient, i.e.
        the first column, which must therefore be all ones.
        """
        duals = compute_cardinal_dual_legendre_coeffs_1d(degree)

        np.testing.assert_allclose(
            duals[:, 0],
            np.ones(degree + 1),
            rtol=0,
            atol=_conditioning_atol(compute_legendre_to_cardinal_1d(degree)),
        )

    @pytest.mark.parametrize("degree", range(9))
    def test_dual_equals_inverse_transpose(self, degree: int) -> None:
        """The duals must be exactly the transpose of the cardinal-to-Legendre matrix."""
        np.testing.assert_array_equal(
            compute_cardinal_dual_legendre_coeffs_1d(degree),
            compute_cardinal_to_legendre_1d(degree).T,
        )

    @pytest.mark.parametrize("degree", range(9))
    def test_consistency_with_bernstein_pair(self, degree: int) -> None:
        """``A_lc`` must equal ``A_bc @ M_lb``, routing through the Bernstein basis.

        An independent construction of the same matrix: going Legendre to Bernstein
        to cardinal must land where going Legendre to cardinal directly does.
        """
        legendre_to_bernstein = _compute_change_basis_1D(
            new_basis_eval=functools.partial(tabulate_legendre_1d, degree),
            old_basis_eval=functools.partial(tabulate_bernstein_1d, degree),
            n_quad_pts=degree + 1,
        )

        np.testing.assert_allclose(
            compute_legendre_to_cardinal_1d(degree),
            compute_bernstein_to_cardinal_1d(degree) @ legendre_to_bernstein,
            rtol=0,
            atol=_COMPOSED_ROUNDOFF_FACTOR * get_machine_epsilon(np.float64),
        )

    @pytest.mark.parametrize(
        "builder",
        [
            compute_legendre_to_cardinal_1d,
            compute_cardinal_to_legendre_1d,
            compute_cardinal_dual_legendre_coeffs_1d,
        ],
    )
    def test_negative_degree_error(
        self,
        builder: Callable[..., npt.NDArray[np.float32 | np.float64]],
    ) -> None:
        """Degree 0 is allowed but a negative degree is rejected."""
        assert builder(0).shape == (1, 1)
        with pytest.raises(ValueError, match="Degree must be non-negative"):
            builder(-1)

    @pytest.mark.parametrize(
        "builder",
        [
            compute_legendre_to_cardinal_1d,
            compute_cardinal_to_legendre_1d,
            compute_cardinal_dual_legendre_coeffs_1d,
        ],
    )
    def test_out_and_dtype_validation(
        self,
        builder: Callable[..., npt.NDArray[np.float32 | np.float64]],
    ) -> None:
        """A provided ``out`` is returned and filled; a bad one is rejected.

        The two calls run the same computation on the same inputs, so the ``out=`` path
        must reproduce the allocating path bit for bit; ``assert_allclose`` would have
        accepted a relative difference of ``1e-7`` between them.
        """
        degree = 3
        out = np.empty((degree + 1, degree + 1), dtype=np.float64)
        result = builder(degree, np.float64, out)
        assert result is out
        np.testing.assert_array_equal(result, builder(degree))

        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            builder(degree, np.int32)
        with pytest.raises(ValueError):
            builder(degree, np.float64, np.empty((degree, degree), dtype=np.float64))
        with pytest.raises(ValueError):
            builder(degree, np.float64, np.empty((degree + 1, degree + 1), dtype=np.float32))

        readonly = np.empty((degree + 1, degree + 1), dtype=np.float64)
        readonly.flags.writeable = False
        with pytest.raises(ValueError):
            builder(degree, np.float64, readonly)

    def test_cached_variants_are_frozen_and_shared(self) -> None:
        """The cached matrices are read-only, shared per key, and equal to the builders."""
        for cached, builder in (
            (_cached_legendre_to_cardinal_matrix, compute_legendre_to_cardinal_1d),
            (_cached_cardinal_to_legendre_matrix, compute_cardinal_to_legendre_1d),
        ):
            first = cached(3, np.dtype(np.float64))
            assert first is cached(3, np.dtype(np.float64))
            assert not first.flags.writeable
            np.testing.assert_array_equal(first, builder(3))
            with pytest.raises(ValueError):
                first[0, 0] = 0.0
