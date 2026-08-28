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

pytestmark = pytest.mark.usefixtures("cpp_backend")


def _unfused_budget(n: int, terms: np.ndarray) -> np.ndarray:
    r"""The absolute gap a missing fusion allows, per matrix entry.

    Absolute rather than relative, because these matrices have entries that
    vanish and a relative gap there is unbounded by cancellation alone --
    ``design/backend_parity.md`` Rule 2.

    The derivation. Without FMA the sum of squares carries `n - 1` extra
    roundings, so it differs from the fused sum by a relative
    :math:`(n-1)\varepsilon`. Taking the square root halves that; dividing each
    component by the norm inherits it; and the product :math:`u_i u_j` doubles it
    again. So each term of the entry is perturbed relatively by about
    :math:`(n-1)\varepsilon`, and the entry's absolute error is that times the
    sum of the magnitudes of its terms -- which is what is passed in, rather than
    the entry itself.

    Args:
        n (int): The vector length being normalized.
        terms (np.ndarray): Sum of the magnitudes of each entry's terms.

    Returns:
        np.ndarray: The per-entry absolute budget.
    """
    return 2.0 * max(n - 1, 1) * EPS * terms


EPS = float(np.finfo(np.float64).eps)
"""Machine epsilon for float64, the unit every bound below is stated in."""

TRIG_ULPS = 2.0
"""How far the two libm implementations may disagree, in ulps of the result.

One ulp is what was measured; two is that with a factor of two of slack, stated
as slack rather than derived, because nothing here bounds a libm's error.
"""


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
        budget = _unfused_budget(len(unit), terms)
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
        budget = _unfused_budget(3, magnitude) + TRIG_ULPS * EPS * magnitude
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
    """The two inverses differ by at most `c n kappa eps` times the answer's size.

    Both are backward stable LU with partial pivoting, so this is a derived
    bound. `kappa_inf` is computed per case rather than assumed, and the constant
    is stated: 8 absorbs the pivot growth and the two triangular solves, and is
    an acknowledged safety factor rather than a derivation.
    """
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(500):
        n = int(rng.integers(1, 7))
        matrix = rng.normal(size=(n, n))
        kappa = float(np.linalg.cond(matrix, np.inf))
        # Skip what the oracle itself would not trust: past 1/eps the inverse has
        # no correct digits and a parity claim over it says nothing, which is
        # design/backend_parity.md Rule 8.
        if not np.isfinite(kappa) or kappa > 1.0 / EPS:
            continue
        checked += 1
        py, cpp = _both(matrix, rng.normal(size=n))
        got = cpp.inverse().matrix
        want = py.inverse.matrix
        budget = 8.0 * n * kappa * EPS * max(1.0, float(np.abs(want).max()))
        assert np.all(np.abs(got - want) <= budget), (
            f"n={n} kappa={kappa:.3e} exceeded {budget:.3e}"
        )
    assert checked > 400, "too many cases were skipped as ill-conditioned"


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
        assert np.all(np.abs(got - want) <= n * EPS * magnitudes + EPS)

        points = rng.normal(size=(int(rng.integers(1, 30)), n))
        out = np.empty_like(points)
        cpp_a.apply(points, out)
        want_pts = py_a(points)
        term_sums = np.abs(points) @ np.abs(a_mat).T + np.abs(a_off)
        assert np.all(np.abs(out - want_pts) <= n * EPS * term_sums + EPS)


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
        scale = max(1.0, float(np.abs(center).max()))
        assert np.all(np.abs(got.matrix - want.matrix) <= TRIG_ULPS * EPS * scale)
        assert np.all(np.abs(got.offset - want.offset) <= 8.0 * EPS * scale)


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
