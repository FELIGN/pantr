r"""Parity of the C++ affine map against the pure-Python oracle it was ported from.

`cpp/include/pantr/transform/affine.hpp` names this file as the place its parity
claims are derived. Unlike `tests/parity/test_geometry_aabb.py`, where every
operation is exact and the claim is a single equality, this port carries **three
different claims**, and telling them apart is most of the work.

Where equality survives
-----------------------

The factories that do nothing but arrange exactly representable numbers --
`identity`, `translation`, `scaling`, `shear` -- reproduce the oracle bit for
bit, and so do `mirror` and `rotation_3d` **given the same axis**, because the
normalization is fused the way the oracle's `ddot` is. Those are asserted as
equalities. Weakening them to a tolerance would hide a defect rather than allow
for one.

Where only a bound survives, and why
------------------------------------

*The inverse.* `np.linalg.inv` is LAPACK `getrf`/`getri`; the port is
`Eigen::PartialPivLU`. Both are LU with partial pivoting and both are backward
stable, so the computed inverses differ by roughly

.. math::

    \|X_{\text{lapack}} - X_{\text{eigen}}\|_\infty
        \lesssim c\, n\, \kappa_\infty(A)\, \varepsilon\, \|A^{-1}\|_\infty ,

with `c` a small constant absorbing the pivot growth. This is the same
derivation `design/backend_parity.md` records for `change_basis`'s solve, reused
rather than reinvented, and `kappa_inf` is computed per case here rather than
assumed.

*Products.* `compose` and `__call__` are `dgemm` in numpy and a loop here: the
same terms summed in a different order, so `n eps` times the sum of magnitudes.

*The trigonometry, which was the surprise.* `rotation_2d` only calls `cos` and
`sin` and negates, so it looked exact. It is not: **the extension's `cos` and the
interpreter's `math.cos` disagree by one ulp on about 0.1% of arguments**, which
no care on the port's side can remove. Measured, with two candidate explanations
ruled out by measurement rather than argument -- numpy's scalar `cos` agrees with
`math.cos` on 200000 of 200000, and although GCC does fuse the `cos`/`sin` pair
into `sincos`, at run time `sincos` and a separate `cos` agree with each other
and both differ from the interpreter. `scripts/measure_affine_transform_parity.py`
reports all of it.

So every claim below is one of: exact, a derived bound, or **one ulp of the
trigonometry**, and each test says which it is.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pantr._backend import Backend, use_backend
from pantr.transform import AffineTransform, _AffineTransformPython
from tests._parity_harness import _LU_ALLOWANCE

pytestmark = pytest.mark.usefixtures("cpp_backend")


EPS = float(np.finfo(np.float64).eps)
"""Machine epsilon for float64, the unit every bound below is stated in."""

TRIG_ULPS = 2.0
"""How far the two libm implementations may disagree, in ulps of the result.

One ulp is what was measured over 500000 angles; two is that with a factor of two
of slack, **declared as slack rather than derived**, because nothing here bounds a
libm's error.

Note what this instrument does and does not do. Near a zero of `cos` an ulp
measure is not too loose but unboundedly too *tight*: at `pi/2`, one ulp of
`cos` is 1e-32, so a libm erring by one ulp of unity would score 1e16 against
this. It can misfire; it cannot mask. Where an absolute form is needed instead --
`rotation_3d`, whose `1 - cos` cancels -- this appears as `TRIG_ULPS * EPS *
magnitude`.
"""

_LU_ALLOWANCE_MEASURED_HERE = 4.391
"""``R = || |L||U| ||_inf / ||A||_inf`` over THIS module's matrices.

Recorded because reusing the harness's ``_LU_ALLOWANCE`` requires it: that
constant denotes a growth factor measured over the change-of-basis builders'
matrices (``R <= 3.73``), and borrowing the number without measuring the quantity
for random normal matrices would be borrowing a value while dropping its meaning.
Measured over 20000 draws to ``n = 6``; the allowance of 8 covers it with 1.8x.
"""


def _normalization_budget(n: int, terms: np.ndarray) -> np.ndarray:
    r"""The absolute gap the two normalizations allow, per matrix entry.

    Absolute rather than relative, because these matrices have entries that
    vanish and a relative gap there is unbounded by cancellation alone --
    ``design/backend_parity.md`` Rule 2.

    **The derivation, and why the previous one was wrong.** The first version
    counted `n - 1` roundings *relative to a fused sequential sum*, on the premise
    that ``np.linalg.norm`` is that sum. `scripts/measure_affine_transform_parity.py`
    refutes the premise: at ``n = 4`` the unfused column is 0.00% and the fused one
    12.00%, so OpenBLAS's ``ddot`` is neither order uniformly and a differential
    count against one of them is not a bound. It was also violated in fact -- 88
    failures in 60000 draws of this module's own generator, which the shipped test
    missed only because its loop stopped at 500.

    This one assumes nothing about summation order. Both sides compute
    :math:`M_{ij} = \delta_{ij} - 2 u_i u_j` with :math:`u = v / \nu` and
    :math:`\nu = \sqrt{\sum v_k^2}`:

    - the squares are all non-negative, so no cancellation is possible and each
      side satisfies :math:`|fl(s) - s| \le \gamma_n s`; two-sided that is
      :math:`n\varepsilon`, whatever order either side sums in;
    - the square root halves it and adds one straddle;
    - the division inherits it and adds one;
    - the product :math:`u_i u_j` doubles it and adds one;
    - multiplying by two is exact, and the final subtract adds one.

    Summing the straddles gives six, so the entry's absolute error is at most
    :math:`(n + 6)\varepsilon` times the sum of the magnitudes of its terms.

    Measured over 60000 draws spanning 300 decades: worst ratio 0.385, no
    violations, against 1.538 and 88 for the version this replaces.

    Args:
        n (int): The vector length being normalized.
        terms (np.ndarray): Sum of the magnitudes of each entry's terms.

    Returns:
        np.ndarray: The per-entry absolute budget.
    """
    return (n + 6) * EPS * terms


def _cpp(matrix: Any, offset: Any) -> Any:
    """Build a map in the C++ implementation, bypassing the active backend.

    Args:
        matrix (Any): The linear part.
        offset (Any): The translation.

    Returns:
        Any: The C++ map.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    return _pantr_cpp.AffineTransform(
        np.ascontiguousarray(matrix, dtype=np.float64),
        np.ascontiguousarray(offset, dtype=np.float64),
    )


def _cpp_cls() -> Any:
    """The bound C++ map class.

    Returns:
        Any: The class, for the static factories.
    """
    from pantr import _pantr_cpp  # noqa: PLC0415

    return _pantr_cpp.AffineTransform


def _both(matrix: Any, offset: Any) -> tuple[_AffineTransformPython, Any]:
    """Build the same map in both implementations.

    Args:
        matrix (Any): The linear part.
        offset (Any): The translation.

    Returns:
        tuple[_AffineTransformPython, Any]: The oracle map and the C++ map.
    """
    return _AffineTransformPython(matrix, offset), _cpp(matrix, offset)


def _ulps_apart(a: np.ndarray, b: np.ndarray) -> float:
    """The largest gap between two arrays, measured in ulps of the larger entry.

    Args:
        a (np.ndarray): One array.
        b (np.ndarray): The other, same shape.

    Returns:
        float: The gap in ulps, zero when the two are bitwise equal.
    """
    scale = np.maximum(np.abs(a), np.abs(b))
    unit = np.where(scale > 0.0, np.spacing(scale), np.spacing(1.0))
    return float(np.max(np.abs(a - b) / unit))


def test_exactly_representable_factories_are_bit_identical() -> None:
    """`identity`, `translation`, `scaling` and `shear` reproduce the oracle exactly.

    Nothing in them rounds: every entry is a copy, a zero or a one. A tolerance
    here would be hiding something.
    """
    cls = _cpp_cls()
    for n in (1, 2, 3, 5):
        assert (
            cls.identity(n).matrix.tobytes() == _AffineTransformPython.identity(n).matrix.tobytes()
        )

    offset = np.array([1.5, -2.0, 1e300])
    assert (
        cls.translation(offset).offset.tobytes()
        == _AffineTransformPython.translation(offset).offset.tobytes()
    )

    factors = np.array([2.0, -0.5, 1e-300])
    assert (
        cls.scaling(factors).matrix.tobytes()
        == _AffineTransformPython.scaling(factors).matrix.tobytes()
    )

    assert (
        cls.shear(4, 1, 3, 0.1).matrix.tobytes()
        == _AffineTransformPython.shear(4, 1, 3, 0.1).matrix.tobytes()
    )


def test_mirror_agrees_within_a_derived_bound() -> None:
    """`mirror` agrees with the oracle to within one missing fusion.

    Not an equality, and the reason is worth stating because two builds were
    tried before accepting it. The oracle fuses its multiply-add **inside**
    OpenBLAS's `ddot`, when it normalizes, and does **not** fuse the array
    expression `eye - 2 * outer(n, n)` that follows. A single translation unit
    has one contraction setting, so it can match the first or the second and not
    both: without FMA the norm differs in the last bits, and with
    `-march=x86-64-v3` the norm matches but the outer product then contracts
    where numpy did not. Measured both ways.

    The normalization goes through `np.linalg.norm`, which routes to a `ddot`
    that fuses its multiply-add. The port fuses too, under the project's
    `-ffp-contract=on`. A build that did not fuse would fail this test, and that
    is the intended signal rather than a nuisance: the equality is conditional on
    the build and the test is where the condition is checked.
    """
    rng = np.random.default_rng(20260828)
    checked = 0
    for _ in range(500):
        normal = rng.normal(size=int(rng.integers(1, 6))) * 10.0 ** rng.integers(-6, 6)
        if not np.isfinite(np.linalg.norm(normal)) or np.linalg.norm(normal) == 0.0:
            continue
        checked += 1
        got = _cpp_cls().mirror(np.ascontiguousarray(normal))
        want = _AffineTransformPython.mirror(normal)
        unit = normal / np.linalg.norm(normal)
        # Terms of entry (i, j): the identity's 1 or 0, and 2 |u_i u_j|.
        terms = np.eye(len(unit)) + 2.0 * np.abs(np.outer(unit, unit))
        budget = _normalization_budget(len(unit), terms)
        assert np.all(np.abs(got.matrix - want.matrix) <= budget), f"normal={normal!r}"
    assert checked > 400, "the sweep skipped too many cases to mean anything"


def test_rotation_3d_agrees_within_a_derived_bound() -> None:
    """`rotation_3d` agrees to within one missing fusion, association included.

    Two things had to match and neither was free. numpy forms the outer product
    and then scales it, so the port must compute `(1 - c) * (u_i u_j)` rather
    than associating to the left; and numpy cannot fuse the scaling into the
    addition, because a materialised array sits between them, so the port has to
    *decline* the contraction its flags would otherwise apply. Both were found by
    this comparison failing.
    """
    rng = np.random.default_rng(11)
    for _ in range(500):
        angle = float(rng.normal() * 10.0)
        axis = rng.normal(size=3) * 10.0 ** rng.integers(-6, 6)
        if not np.isfinite(np.linalg.norm(axis)) or np.linalg.norm(axis) == 0.0:
            continue
        got = _cpp_cls().rotation_3d(angle, np.ascontiguousarray(axis))
        want = _AffineTransformPython.rotation_3d(angle, axis)
        unit = axis / np.linalg.norm(axis)
        # Terms of entry (i, j): cos on the diagonal, (1 - cos) u_i u_j, and
        # sin times a component of the axis.
        magnitude = np.eye(3) + 2.0 * np.abs(np.outer(unit, unit)) + np.abs(unit).max()
        budget = _normalization_budget(3, magnitude) + TRIG_ULPS * EPS * magnitude
        assert np.all(np.abs(got.matrix - want.matrix) <= budget), f"angle={angle} axis={axis!r}"


def test_rotation_2d_agrees_to_within_the_two_libms() -> None:
    """`rotation_2d` agrees to one ulp, and the residue is not the port's to fix.

    The only arithmetic is a negation, which is exact, so any disagreement is the
    trigonometry. Asserting equality here would be asserting that two libm
    implementations agree, which they do not.
    """
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(2000):
        angle = float(rng.normal() * 10.0)
        got = _cpp_cls().rotation_2d(angle).matrix
        want = _AffineTransformPython.rotation_2d(angle).matrix
        worst = max(worst, _ulps_apart(got, want))
    assert worst <= TRIG_ULPS, f"rotation_2d drifted {worst} ulps, over the {TRIG_ULPS} allowed"


def test_inverse_agrees_within_its_condition_number() -> None:
    r"""The two inverses differ by at most `3 R n kappa_inf eps ||X||_inf`.

    Both are backward stable LU with partial pivoting, so this is derived rather
    than fitted. Higham 2nd ed. §14.3 (14.15)-(14.18) covers inversion by LU --
    Method B there is `xGETRI`, which is what `numpy.linalg.inv` calls -- and gives
    the componentwise result. Taking norms as he does at (14.5)-(14.7):

    .. math::

        \frac{\|X - A^{-1}\|_\infty}{\|A^{-1}\|_\infty}
            \le c'_n\, u\, R\, \kappa_\infty(A), \qquad
        R = \frac{\||L||U|\|_\infty}{\|A\|_\infty}

    with :math:`c'_n \sim 3n` from Theorem 9.4's :math:`\gamma_{3n}`, which (14.15)
    invokes. Two-sided, since neither backend is exact, and :math:`u = \varepsilon/2`.

    Three corrections to the version this replaces, and each was a real defect:

    - the constant was ``8``, which is `R` alone. The derivation gives `3R`, so the
      old bound was three times tighter than licensed. `R` is reused from the
      harness rather than minted afresh, and `_LU_ALLOWANCE_MEASURED_HERE` records
      the measurement that licenses reusing it for these matrices.
    - the scale was ``max(1, |X|.max())``, the max-norm, where the derivation gives
      :math:`\|X\|_\infty` -- a row sum, larger by up to `n` and measured 5.2x here.
    - the ``max(1, ...)`` floor destroyed the scale covariance the bound must have:
      :math:`\kappa_\infty` is scale-invariant and :math:`A^{-1}` scales like
      :math:`1/\lambda`, so clamping made the bound degree zero above unit scale
      and the budget-to-answer ratio climbed six orders from input scale 1 to 1e8.

    The domain guard is the ACCURACY domain, not the solvability one. The old
    ``kappa > 1/eps`` admitted a band where the bound exceeded the answer and the
    comparison decided nothing, which is what ``design/backend_parity.md`` Rule 8
    exists to forbid.
    """
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(500):
        n = int(rng.integers(1, 7))
        matrix = rng.normal(size=(n, n))
        kappa = float(np.linalg.cond(matrix, np.inf))
        if not np.isfinite(kappa):
            continue
        py, cpp = _both(matrix, rng.normal(size=n))
        try:
            want = py.inverse.matrix
        except ValueError:
            continue
        got = cpp.inverse().matrix
        inverse_norm = float(np.abs(want).sum(axis=1).max())
        budget = 3.0 * _LU_ALLOWANCE * n * kappa * EPS * inverse_norm
        # Refuse the comparison where the bound no longer says anything, rather
        # than counting a vacuous pass. This is the accuracy domain.
        if budget >= inverse_norm:
            continue
        checked += 1
        assert np.all(np.abs(got - want) <= budget), (
            f"n={n} kappa={kappa:.3e} exceeded {budget:.3e}"
        )
    assert checked > 300, f"only {checked} draws had a bound that said anything"


def test_compose_and_apply_agree_within_a_reordered_sum() -> None:
    """Products differ only by summation order, and by no more than that allows.

    A dot product of `n` terms reordered moves the result by at most `n eps`
    times the sum of the magnitudes of its terms. That sum is computed here per
    entry rather than replaced by the result's own magnitude, which would
    underbound wherever the terms cancel.
    """
    rng = np.random.default_rng(5)
    for _ in range(300):
        n = int(rng.integers(1, 7))
        a_mat, a_off = rng.normal(size=(n, n)), rng.normal(size=n)
        b_mat, b_off = rng.normal(size=(n, n)), rng.normal(size=n)
        py_a, cpp_a = _both(a_mat, a_off)
        py_b, cpp_b = _both(b_mat, b_off)

        got = cpp_a.compose(cpp_b).matrix
        want = py_a.compose(py_b).matrix
        # Sum of magnitudes of the terms of each entry: |A| @ |B|.
        magnitudes = np.abs(a_mat) @ np.abs(b_mat)
        # `n * EPS` is exactly two-sided gamma_n, with no margin. The factor of 2
        # is DECLARED build slack -- FMA, vector width, libm -- not a derivation;
        # `src/pantr/tolerance.py` records 4x as this project's usual allowance
        # for it and this takes half of that.
        #
        # There is no additive floor. The previous version carried `+ EPS`, which
        # is dimensionally wrong (EPS is dimensionless, the entries are not) and
        # made the assertion vacuous below input magnitude 1e-6: at scale 1e-8 the
        # budget was 8607 times the values being compared.
        assert np.all(np.abs(got - want) <= 2 * n * EPS * magnitudes)

        points = rng.normal(size=(int(rng.integers(1, 30)), n))
        out = np.empty_like(points)
        cpp_a.apply(points, out)
        want_pts = py_a(points)
        term_sums = np.abs(points) @ np.abs(a_mat).T + np.abs(a_off)
        # `n + 1`, not `n`: the dot product has n terms and then one more straddle
        # adding the offset. The shipped `n` reached 0.929 of its own budget at
        # n = 2, which is a bound about to break rather than a bound.
        assert np.all(np.abs(out - want_pts) <= 2 * (n + 1) * EPS * term_sums)


def test_about_center_agrees_with_the_oracles_center_argument() -> None:
    """The re-centred factories agree, which is what makes the two APIs one map.

    The oracle takes `center=` on five factories; the port has one
    `about_center`, and the wrapper reassembles the former from the latter. If
    the conjugation were written the wrong way round this is what would notice,
    and no shape check would.
    """
    rng = np.random.default_rng(13)
    for _ in range(300):
        angle = float(rng.normal())
        center = rng.normal(size=2)
        with use_backend(Backend.PYTHON):
            want = AffineTransform.rotation_2d(angle, center=center)
        with use_backend(Backend.CPP):
            got = AffineTransform.rotation_2d(angle, center=center)
        # Conjugation multiplies, so this inherits the product bound, on top of
        # the one ulp the trigonometry already carries.
        # The matrix bound carries NO centre scale. Conjugating by a translation
        # leaves the linear part alone, and the C++ triple product multiplies only
        # by exact zeros and ones, so the matrix is bitwise the un-centred
        # rotation -- measured, 2000 of 2000 draws. Multiplying its budget by
        # |center| scaled a quantity that does not depend on the centre, and went
        # vacuous around |center| ~ 5e14.
        # Spelled with the module's own ulp helper rather than absolutely: near a
        # zero of `cos` the absolute form is up to 4e12 times looser, and this
        # module already asserts exactly this quantity that way at
        # `test_rotation_2d_agrees_to_within_the_two_libms`.
        assert _ulps_apart(got.matrix, want.matrix) <= TRIG_ULPS
        # The OFFSET is where the centre legitimately enters: both sides compute
        # `fl(fl(sum_j R_ij (-c_j)) + c_i)`. Two-sided, at `|c|_max`:
        #
        #   trigonometry   2 eps sum_j |R_ij||c_j|  <=  2 sqrt2 eps |c|_max
        #   the 2-term dot 2 eps sum_j |R_ij c_j|   <=  2 sqrt2 eps |c|_max
        #   the final add    eps (|c_i| + sum_j |R_ij c_j|)  <=  (1 + sqrt2) eps |c|_max
        #
        # so `(1 + 5 sqrt2) eps |c|_max`, about 8.07. The previous version wrote
        # `12.0 * (1 + sqrt2)`, which counts the pushforward twice: the 12 was
        # itself derived as `(2 + 2 + 1) * 2.42` and that 2.42 IS `1 + sqrt2`. The
        # effective budget was 29 eps against a licensed 8.07, and a multiplier
        # that reads as derived and is not is the exact failure this discipline
        # names. Measured worst over 260000 draws, angles pinned near the maximum
        # of |cos| + |sin| and |c| over 600 decades: 2.789 eps |c|_max.
        offset_budget = (1.0 + 5.0 * np.sqrt(2.0)) * EPS * float(np.abs(center).max())
        assert np.all(np.abs(got.offset - want.offset) <= offset_budget)

        # This bound is derived for a 2-D ROTATION specifically: the sqrt2 is
        # `sum_j |R_ij|` for that matrix. It is wrong for `mirror` (1 + 2 sqrt n),
        # for `scaling` (unbounded) and for `rotation_3d`, which is why only
        # rotation_2d is swept here and why the other four factories the wrapper
        # routes through `about_center` are not covered.


def test_the_two_backends_disagree_about_some_exactly_singular_matrices() -> None:
    """Pin the singularity disagreement rather than pretend it is not there.

    `numpy.linalg.inv` factors through LAPACK and the port through Eigen's
    `PartialPivLU`. Both refuse on the same criterion -- a pivot that is exactly
    zero -- but they are different computations, so their pivots differ in the
    last bits and one reaches exact zero where the other does not. Exact
    singularity is a discrete verdict and no tolerance bounds it, which is the
    shape `design/backend_parity.md` Rule 11 records for the BVH's tie contract.

    This asserts the disagreement, so that a change which accidentally removes it
    -- or moves it -- has to come here and say so. It also asserts what does NOT
    disagree: a matrix that is singular structurally rather than by cancellation,
    and every well-conditioned matrix, are decided identically.
    """

    def refuses(cls: Any, matrix: Any) -> bool:
        """Whether an implementation refuses to invert.

        Args:
            cls (Any): The implementation class.
            matrix (Any): The linear part.

        Returns:
            bool: `True` when it raises `ValueError`.
        """
        built = cls(np.ascontiguousarray(matrix, dtype=np.float64), np.zeros(len(matrix)))
        try:
            # `inverse` is a property on the oracle and a method on the C++ class,
            # branched on explicitly: testing `callable` picks the wrong branch,
            # because the oracle's returned map is itself callable.
            _ = built.inverse if cls is _AffineTransformPython else built.inverse()
        except ValueError:
            return True
        return False

    cpp_cls = _cpp_cls()

    # Where they agree, and must keep agreeing.
    for matrix, expected in (
        ([[1.0, 0.0], [0.0, 0.0]], True),  # a zero row: structurally singular
        ([[4.0, 1.0], [1.0, 3.0]], False),  # well conditioned
        ([[1e-300, 0.0], [0.0, 1e-300]], False),  # tiny, kappa_inf = 1
    ):
        assert refuses(_AffineTransformPython, matrix) is expected
        assert refuses(cpp_cls, matrix) is expected, f"backends split on {matrix}"

    # Where they do not, and the disagreement is the thing being pinned.
    r3_is_r1_plus_r2 = [[3.0, 4.0, -1.0], [2.0, 4.0, 1.0], [5.0, 8.0, 0.0]]
    assert not refuses(_AffineTransformPython, r3_is_r1_plus_r2)
    assert refuses(cpp_cls, r3_is_r1_plus_r2), "the disagreement moved; see the header"

    other_way = [[-2.0, -4.0, 0.0], [3.0, 1.0, -4.0], [1.0, -3.0, -4.0]]
    assert refuses(_AffineTransformPython, other_way)
    assert not refuses(cpp_cls, other_way), "the disagreement moved; see the header"


def test_errors_agree_verbatim() -> None:
    """Both implementations raise `ValueError` and say exactly the same thing.

    Verbatim rather than by substring: a substring match cannot see a `::` where
    a `.` belongs, which is the defect that shape of assertion let through in the
    `AABB` port.
    """
    cases: list[tuple[Any, str]] = [
        (lambda cls: cls(np.zeros((2, 3)), np.zeros(2)), "non-square matrix"),
        (lambda cls: cls(np.zeros((2, 2)), np.zeros(3)), "translation length"),
    ]
    cpp_cls = _cpp_cls()
    for build, what in cases:
        with pytest.raises(ValueError) as oracle:
            build(_AffineTransformPython)
        with pytest.raises(ValueError) as ported:
            build(cpp_cls)
        assert str(oracle.value) == str(ported.value), (
            f"{what}: oracle said {str(oracle.value)!r}, C++ said {str(ported.value)!r}"
        )


def test_the_wrapper_is_backend_invariant() -> None:
    """`repr` and the pickle-free public surface do not move with the backend.

    `AffineTransform` has no `__eq__` and no `__hash__`, and the port does not
    add them: a caller relying on identity comparison must keep getting it. What
    the wrapper does own is `__repr__`, computed here rather than delegated so
    the two backends print identically.
    """
    matrix = [[2.0, 0.0], [0.0, 0.5]]
    with use_backend(Backend.PYTHON):
        py_map = AffineTransform(matrix, [1.0, 2.0])
    with use_backend(Backend.CPP):
        cpp_map = AffineTransform(matrix, [1.0, 2.0])
    assert repr(py_map) == repr(cpp_map)
    assert not hasattr(AffineTransform, "__eq__") or AffineTransform.__eq__ is object.__eq__
    assert py_map != cpp_map, "identity comparison must survive the port"
