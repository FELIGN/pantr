"""Probes for :mod:`pantr.bezier`: the Bezier surface, root finder and interpolation.

Bezier is fixed on the parametric domain ``[0, 1]^rank``, so unlike the B-spline
probes the *domain* axis from ``_axes`` does not apply here -- there is no knot
vector to translate. The analogous hostile axis is control-point **magnitude**:
:func:`_magnitude_variants` sweeps unit-scale, tiny, huge, and *translated*
control-point clouds (``1e6 + O(1)``), which is the one that catches a tolerance
derived from a coordinate's own magnitude rather than from the control polygon's
diagonal -- such a tolerance is not translation invariant.

Two Numba/NumPy kernels are probed directly rather than only through the public
``Bezier`` surface, because they are imported by a private, non-public consumer
and are the sharpest boundscheck targets in the package:

- ``pantr.bezier._root_finding_core._de_casteljau_eval_scalar`` -- a
  ``nopython=True`` kernel with *no* input validation whatsoever.
- ``pantr.bezier._bezier_interpolate._bernstein_interpolate`` -- a pure-NumPy
  Layer 2 helper (truncated-SVD pseudo-inverse), also unvalidated.

Every invariant here is a genuinely claimed property quoted from the docstrings
(degree elevation is exact, degree reduction reproduces the endpoints bit for
bit, split/compose/multiply/derivative agree with their defining identities,
root finding returns bounded genuine roots), never a mirror of the algorithm
under test. Tolerances are derived in-line from machine epsilon, degree, and
coefficient magnitude -- see each helper's docstring for the derivation.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt

from pantr.bezier import (
    Bezier,
    create_from_bspline,
    find_monotone_root,
    find_roots,
    fit_bezier,
    interpolate_bezier,
)
from pantr.bezier._bezier_interpolate import _bernstein_interpolate
from pantr.bezier._root_finding_core import _de_casteljau_eval_scalar
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.quad import PointsLattice
from pantr.tolerance import get_default, get_machine_epsilon
from pantr.transform import AffineTransform

from ._axes import Profile, coeff_specs, degrees, dtypes, point_specs, rng
from ._core import Case, custom, expected_shape

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ._core import Invariant

GROUP = "bezier"
"""Registry name of this probe group."""

_RANK_DIM_FULL: Final = (
    (1, 1),  # scalar curve
    (2, 1),  # planar curve
    (3, 1),  # space curve
    (4, 1),  # curve into R^4, admitted purely because the constructor allows it
    (1, 2),  # scalar surface
    (2, 2),  # planar surface
    (1, 3),  # scalar volume
    (3, 3),  # solid (typical NURBS volume)
)
"""``(rank, dim)`` combinations swept, deliberately including mismatched-but-legal
pairs (a space curve, a scalar surface) rather than only the square ones."""

_RANK_DIM_SMOKE: Final = ((1, 1), (1, 2))


def _rank_dim_combos(profile: Profile) -> tuple[tuple[int, int], ...]:
    """List the ``(rank, dim)`` combinations to sweep.

    Args:
        profile (Profile): Sweep width.

    Returns:
        tuple[tuple[int, int], ...]: ``(rank, dim)`` pairs.
    """
    return _RANK_DIM_SMOKE if profile is Profile.SMOKE else _RANK_DIM_FULL


_EXTRAPOLATION_POINT_FAMILIES: Final = frozenset(
    {"just_outside_right", "just_outside_left", "far_outside", "just_outside"}
)
"""Point families that deliberately evaluate outside ``[0, 1]``.

Bernstein basis functions extrapolate like ``O(t ** n)``, so at a fixed
off-domain parameter a high enough degree overflows even before any
implementation defect is involved; these families are excluded from the
automatic finiteness check rather than asserted finite at every degree.
"""


_MAGNITUDE_FULL: Final = (
    ("scale1", 1.0, 0.0),
    ("scale1e-6", 1e-6, 0.0),
    ("scale1e6", 1e6, 0.0),
    ("translated1e6", 1.0, 1e6),
)
"""Control-point magnitude/translation families: ``value = scale * u + translate``
for ``u`` uniform on ``[-1, 1]``. The translated family is the one that bites a
tolerance derived from a coordinate's own magnitude instead of the control
polygon's diagonal."""

_MAGNITUDE_SMOKE: Final = (("scale1", 1.0, 0.0),)


def _magnitude_variants(profile: Profile) -> tuple[tuple[str, float, float], ...]:
    """List the control-point magnitude/translation families to sweep.

    Args:
        profile (Profile): Sweep width.

    Returns:
        tuple[tuple[str, float, float], ...]: ``(name, scale, translate)`` triples.
    """
    return _MAGNITUDE_SMOKE if profile is Profile.SMOKE else _MAGNITUDE_FULL


def _random_control_points(  # noqa: PLR0913 -- one axis value per keyword, all needed by callers
    rank: int,
    dim: int,
    degree: int,
    dtype: np.dtype[np.float32 | np.float64],
    offset: int,
    *,
    rational: bool = False,
    scale: float = 1.0,
    translate: float = 0.0,
) -> npt.NDArray[np.float32 | np.float64]:
    """Build a random control-point array for one ``(rank, dim, degree)`` case.

    Args:
        rank (int): Geometric output rank (excluding the rational weight).
        dim (int): Parametric dimension.
        degree (int): Polynomial degree, same in every direction.
        dtype (np.dtype[np.float32 | np.float64]): Control-point precision.
        offset (int): Random-stream offset, for reproducible but distinct draws.
        rational (bool): Whether to append a positive weight column. Defaults
            to ``False``.
        scale (float): Multiplicative magnitude factor. Defaults to 1.0.
        translate (float): Additive translation applied after scaling. Defaults
            to 0.0.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Control points, shape
        ``(*(degree + 1,) * dim, rank + (1 if rational else 0))``.
    """
    generator = rng(offset)
    shape = (*((degree + 1,) * dim), rank + (1 if rational else 0))
    cp = (generator.uniform(-1.0, 1.0, shape) * scale + translate).astype(dtype)
    if rational:
        weights = generator.uniform(0.5, 1.5, shape[:-1]).astype(dtype)
        cp[..., -1] = weights
    return cp


def _eps(dtype: npt.DTypeLike) -> float:
    """Fetch machine epsilon for a dtype, as a bare float.

    Args:
        dtype (npt.DTypeLike): Floating dtype.

    Returns:
        float: Machine epsilon.
    """
    return get_machine_epsilon(dtype)


def _magnitude(cp: npt.NDArray[np.floating[Any]]) -> float:
    """Compute the coefficient-magnitude scale used to size a tolerance.

    Args:
        cp (npt.NDArray[np.floating[Any]]): Control points or coefficients.

    Returns:
        float: ``max(|cp|, 1.0)`` -- floored at 1 so a near-zero cloud does not
        collapse the tolerance to zero.
    """
    return float(max(np.max(np.abs(cp)), 1.0)) if cp.size else 1.0


def _convex_combination_tol(
    degree_total: int, dtype: npt.DTypeLike, cp: npt.NDArray[np.floating[Any]]
) -> float:
    """Bound the error of an operation whose outputs are convex combinations of inputs.

    Degree elevation, splitting, and restriction all produce each output
    coefficient as a weighted sum of at most ``degree_total + 1`` input
    coefficients with non-negative weights summing to one (elevation: binomial
    weights; splitting/restriction: repeated de Casteljau lerps). Summing
    ``degree_total + 1`` such terms carries a forward error of at most
    ``(degree_total + 1) * eps`` relative to the largest coefficient magnitude.
    An explicit factor of 8 absorbs the recurrence depth of the underlying
    algorithm (de Casteljau/elevation run in ``O(degree)`` sequential steps,
    each contributing its own rounding).

    Args:
        degree_total (int): Combined polynomial degree driving the term count.
        dtype (npt.DTypeLike): Working precision.
        cp (npt.NDArray[np.floating[Any]]): Control points, for the magnitude scale.

    Returns:
        float: Absolute tolerance.
    """
    return 8.0 * (degree_total + 1) * _eps(dtype) * _magnitude(cp)


def _fd_step_and_tol(
    dtype: npt.DTypeLike, cp: npt.NDArray[np.floating[Any]], degree: int
) -> tuple[float, float]:
    """Compute the central-difference step and noise floor for a derivative check.

    A central difference has truncation error ``O(h**2)`` and rounding error
    ``O(eps / h)``; the standard balance sets ``h = eps ** (1/3)``, which
    equalises both at ``eps ** (2/3)``. The truncation term is ``(h**2 / 6) *
    f'''(xi)``, and the Markov brothers' inequality bounds a degree-``n``
    polynomial's ``k``-th derivative sup-norm by a factor of order ``n**(2k)``
    relative to the polynomial's own sup-norm -- so the third-derivative term
    can grow with degree far faster than the flat ``O(eps**(2/3))`` estimate
    admits. A random Bernstein polynomial is nowhere near that Chebyshev-sharp
    extremal case, so ``(degree + 1) ** 2`` is used as a deliberately modest
    (not the sharp ``n**6``) headroom factor, on top of an explicit constant
    of 8 for the unmodelled recurrence depth.

    Args:
        dtype (npt.DTypeLike): Working precision.
        cp (npt.NDArray[np.floating[Any]]): Control points, for the magnitude scale.
        degree (int): Polynomial degree, which sets the derivative-growth headroom.

    Returns:
        tuple[float, float]: ``(h, tol)`` -- the finite-difference step and the
        absolute tolerance on the derivative residual.
    """
    eps = _eps(dtype)
    h = eps ** (1.0 / 3.0)
    tol = 8.0 * (degree + 1) ** 2 * eps ** (2.0 / 3.0) * _magnitude(cp)
    return h, tol


def _product_tol(
    p_degree: int,
    q_degree: int,
    dtype: npt.DTypeLike,
    magnitude: float,
    *,
    chain_depth: int = 1,
) -> float:
    """Bound the error of a Bernstein product (``multiply`` or ``compose``).

    Each product coefficient sums ``O(p + q)`` cross terms with binomial
    weights that telescope to one, so the relative error accumulates like the
    combined degree relative to the *result's own* magnitude -- callers must
    pass that magnitude explicitly, since it is not always the operands'
    magnitude (a pointwise product's natural scale is the product of both
    factors' magnitudes, not their max). ``chain_depth`` covers a caller that
    builds its result through several *sequential* Bernstein products (e.g.
    ``compose`` computing successive powers ``g, g**2, ..., g**n`` of a
    reparametrization): each of the ``chain_depth`` steps contributes its own
    single-product error, which then also propagates through the remaining
    steps, so the total is bounded by ``chain_depth`` times a single step's
    bound. An explicit factor of 8 covers the unmodelled recurrence constant.

    Args:
        p_degree (int): Degree of the first factor.
        q_degree (int): Degree of the second factor.
        dtype (npt.DTypeLike): Working precision.
        magnitude (float): Magnitude scale of the *result* being checked.
        chain_depth (int): Number of sequential Bernstein products chained to
            build the result. Defaults to 1 (a single product).

    Returns:
        float: Absolute tolerance.
    """
    return 8.0 * (p_degree + q_degree + 1) * _eps(dtype) * magnitude * max(chain_depth, 1)


def _delta_failure(
    name: str, got: npt.NDArray[Any], expected: npt.NDArray[Any], tol: float
) -> str | None:
    """Report the worst absolute deviation between two arrays, if it exceeds a tolerance.

    Shared by every invariant that compares one Bezier evaluation to another
    (or to an analytic oracle), to keep each call site to one line.

    Args:
        name (str): Short label for the quantities being compared, used only
            in the failure message.
        got (npt.NDArray[Any]): The value under test.
        expected (npt.NDArray[Any]): The reference value.
        tol (float): Absolute tolerance.

    Returns:
        str | None: ``None`` when within tolerance, else a short message.
    """
    worst = float(np.max(np.abs(np.asarray(got) - np.asarray(expected))))
    if worst > tol:
        return f"{name}: max|delta| = {worst:.3e} > {tol:.3e}"
    return None


def _bernstein_eval_noise(
    degree: int, dtype: npt.DTypeLike, cp: npt.NDArray[np.floating[Any]]
) -> float:
    """Bound the rounding noise of evaluating a degree-``n`` Bernstein polynomial.

    The de Casteljau recurrence runs ``degree`` sequential lerp stages, each
    contributing ``O(eps)`` relative to the running coefficient magnitude; an
    explicit factor of 8 covers the unmodelled constant.

    Args:
        degree (int): Polynomial degree.
        dtype (npt.DTypeLike): Working precision.
        cp (npt.NDArray[np.floating[Any]]): Coefficients, for the magnitude scale.

    Returns:
        float: Absolute tolerance on ``|f(t)|`` for an alleged root ``t``.
    """
    return 8.0 * (degree + 1) * _eps(dtype) * _magnitude(cp)


def _double_op_identity_invariant(original_cp: npt.NDArray[np.floating[Any]]) -> Invariant:
    """Build an invariant asserting a bitwise round trip back to the original.

    ``reverse`` and ``permute_directions`` are pure permutations of the control
    point array, so applying the inverse operation must reproduce the original
    exactly -- ``==``, not a tolerance.

    Args:
        original_cp (npt.NDArray[np.floating[Any]]): The control points before
            the round trip.

    Returns:
        Invariant: Check reporting any bitwise mismatch.
    """

    def predicate(result: object) -> str | None:
        if not isinstance(result, Bezier):
            return f"expected Bezier, got {type(result).__name__}"
        if not np.array_equal(result.control_points, original_cp):
            return "round trip is not bit-identical to the original control points"
        return None

    return custom("double-op-identity", predicate)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _construct_cases(profile: Profile) -> Iterator[Case]:
    """Yield :class:`~pantr.bezier.Bezier` construction cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile construction.
    """
    corner_degrees = (0, 1, 3, 15, 62) if profile is Profile.FULL else (0, 1)
    combos = _rank_dim_combos(profile) if profile is Profile.FULL else _rank_dim_combos(profile)[:1]
    for dtype in dtypes(profile):
        for rank, dim in combos:
            for degree in corner_degrees:
                for rational in (False, True) if profile is Profile.FULL else (False,):
                    cp = _random_control_points(
                        rank, dim, degree, dtype, offset=1000 + degree, rational=rational
                    )
                    tag = f"r{rank}d{dim}p{degree}_{dtype}_{'rat' if rational else 'nonrat'}"
                    yield Case(
                        GROUP,
                        f"construct_{tag}",
                        Bezier,
                        lambda cp=cp, rational=rational: Bezier(cp, is_rational=rational),
                        {"rank": rank, "dim": dim, "degree": degree, "dtype": str(dtype)},
                        invariants=(
                            custom(
                                "shape-matches-request",
                                lambda r, rank=rank, dim=dim, degree=degree: None
                                if (r.rank == rank and r.dim == dim and r.degree == (degree,) * dim)
                                else f"got rank={r.rank} dim={r.dim} degree={r.degree}",
                            ),
                        ),
                        arrays={"control_points": cp},
                    )

    # Malformed / edge-case constructions: documented rejections or corners.
    yield Case(
        GROUP,
        "construct_empty_cp",
        Bezier,
        lambda: Bezier(np.zeros((0, 2), dtype=np.float64)),
        {"kind": "empty"},
    )
    yield Case(
        GROUP,
        "construct_scalar_cp",
        Bezier,
        lambda: Bezier(np.float64(1.0)),
        {"kind": "0d-scalar"},
    )
    yield Case(
        GROUP,
        "construct_rank_zero",
        Bezier,
        lambda: Bezier(np.zeros((3, 0), dtype=np.float64)),
        {"kind": "rank-zero"},
    )
    yield Case(
        GROUP,
        "construct_1d_list_reshape",
        Bezier,
        lambda: Bezier([0.0, 1.0, 2.0]),
        {"kind": "1d-list"},
        invariants=(expected_shape(()),),
        finite_inputs=False,  # predicate below checks rank/dim, not the raw result
    )
    yield Case(
        GROUP,
        "construct_integer_cp_cast",
        Bezier,
        lambda: Bezier(np.array([[0, 0], [1, 1], [2, 0]], dtype=np.int64)),
        {"kind": "integer-cast"},
        invariants=(
            custom(
                "cast-to-float64",
                lambda r: None if r.dtype == np.float64 else f"got dtype {r.dtype}",
            ),
        ),
    )
    float32_cp = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    yield Case(
        GROUP,
        "construct_float32_untouched",
        Bezier,
        lambda: Bezier(float32_cp),
        {"kind": "float32-passthrough"},
        invariants=(
            custom(
                "float32-preserved",
                lambda r: None if r.dtype == np.float32 else f"got dtype {r.dtype}",
            ),
        ),
        arrays={"control_points": float32_cp},
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _nd_point_families(
    dim: int, dtype: np.dtype[np.float32 | np.float64], profile: Profile
) -> list[tuple[str, npt.NDArray[np.float32 | np.float64], bool]]:
    """Build hostile evaluation-point families for a ``dim >= 2`` Bezier.

    :func:`~._axes.point_specs` only builds 1D coordinate arrays; this mirrors
    its corner families (interior, both corners, out-of-range, empty) for a
    genuinely multi-dimensional parametric point. Each point repeats the same
    coordinate across every direction rather than crossing per-direction
    values -- a deliberate simplification to keep the case count bounded.

    Args:
        dim (int): Parametric dimension, ``>= 2``.
        dtype (np.dtype[np.float32 | np.float64]): Point precision.
        profile (Profile): Sweep width.

    Returns:
        list[tuple[str, npt.NDArray[np.float32 | np.float64], bool]]: ``(name,
        pts, finite)`` triples, ``pts`` of shape ``(n, dim)``.
    """
    families: list[tuple[str, npt.NDArray[np.float32 | np.float64], bool]] = [
        ("interior", np.tile(np.array([[0.31], [0.77]], dtype=dtype), (1, dim)), True),
        ("corner_zero", np.zeros((1, dim), dtype=dtype), True),
        ("corner_one", np.ones((1, dim), dtype=dtype), True),
        (
            "mixed_corner",
            np.array([[0.0 if d % 2 == 0 else 1.0 for d in range(dim)]], dtype=dtype),
            True,
        ),
        ("empty", np.zeros((0, dim), dtype=dtype), True),
    ]
    if profile is Profile.FULL:
        families += [
            ("just_outside", np.full((1, dim), -1.0e-3, dtype=dtype), True),
            ("far_outside", np.full((1, dim), 5.0, dtype=dtype), True),
            ("nan_inf", np.array([[np.nan, *([0.5] * (dim - 1))]], dtype=dtype), False),
        ]
    return families


def _evaluate_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`~pantr.bezier.Bezier.evaluate` cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile evaluation.
    """
    eval_degrees = degrees(profile) if profile is Profile.FULL else degrees(profile)[:1]
    # Explicit, dimension-diverse subset -- `_RANK_DIM_FULL[:4]` would pick four
    # dim=1 combos back to back and never cross the point-family axis with a
    # surface or volume at all.
    combos = (
        ((1, 1), (3, 1), (1, 2), (3, 3))
        if profile is Profile.FULL
        else _rank_dim_combos(profile)[:1]
    )
    for dtype in dtypes(profile):
        for rank, dim in combos:
            for degree in eval_degrees:
                cp = _random_control_points(rank, dim, degree, dtype, offset=2000 + degree)
                bezier = Bezier(cp)
                tag = f"r{rank}d{dim}p{degree}_{dtype}"

                if dim == 1:
                    pt_families = [
                        (spec.name, spec.pts, spec.finite)
                        for spec in point_specs((0.0, 1.0), dtype, profile)
                    ]
                else:
                    pt_families = _nd_point_families(dim, dtype, profile)

                for name, pts, finite in pt_families:
                    # Bernstein extrapolation is mathematically unbounded: at a
                    # fixed off-domain parameter, B_i^n(t) grows like O(t**n),
                    # so a high enough degree genuinely overflows float32 (and
                    # eventually float64) well before any implementation error
                    # is involved -- nothing in `evaluate`'s contract promises
                    # a finite answer for these families at every degree.
                    finite_expected = finite and name not in _EXTRAPOLATION_POINT_FAMILIES
                    yield Case(
                        GROUP,
                        f"evaluate_{tag}_{name}",
                        Bezier.evaluate,
                        lambda bezier=bezier, pts=pts: bezier.evaluate(pts),
                        {
                            "rank": rank,
                            "dim": dim,
                            "degree": degree,
                            "dtype": str(dtype),
                            "points": name,
                        },
                        finite_inputs=finite_expected,
                        arrays={"control_points": cp, "pts": pts},
                    )

                if profile is Profile.FULL:
                    # Rationality crosses degree/dim/dtype separately, at a
                    # single interior point, rather than multiplying into the
                    # point-family sweep above (which is about domain geometry,
                    # not rationality).
                    cp_rat = _random_control_points(
                        rank, dim, degree, dtype, offset=2000 + degree, rational=True
                    )
                    bezier_rat = Bezier(cp_rat, is_rational=True)
                    interior_pts = (
                        np.array([0.31, 0.77], dtype=dtype)
                        if dim == 1
                        else np.tile(np.array([[0.31], [0.77]], dtype=dtype), (1, dim))
                    )
                    yield Case(
                        GROUP,
                        f"evaluate_{tag}_rational_interior",
                        Bezier.evaluate,
                        lambda bezier_rat=bezier_rat,
                        interior_pts=interior_pts: bezier_rat.evaluate(interior_pts),
                        {
                            "rank": rank,
                            "dim": dim,
                            "degree": degree,
                            "dtype": str(dtype),
                            "rational": True,
                        },
                        arrays={"control_points": cp_rat, "pts": interior_pts},
                    )

    # Dtype-mismatch and out-of-range direction: documented rejections.
    bezier64 = Bezier(_random_control_points(1, 1, 3, np.dtype(np.float64), offset=2900))
    pts32 = np.array([0.5], dtype=np.float32)
    yield Case(
        GROUP,
        "evaluate_dtype_mismatch",
        Bezier.evaluate,
        lambda: bezier64.evaluate(pts32),
        {"kind": "dtype-mismatch"},
    )

    # out= round trip: correct shape/dtype accepted and filled bit-identically.
    bezier_out = Bezier(_random_control_points(2, 1, 3, np.dtype(np.float64), offset=2901))
    pts_out = np.array([0.2, 0.6], dtype=np.float64)
    expected = bezier_out.evaluate(pts_out)
    out_buf = np.empty_like(expected)
    yield Case(
        GROUP,
        "evaluate_out_roundtrip",
        Bezier.evaluate,
        lambda: bezier_out.evaluate(pts_out, out=out_buf),
        {"kind": "out-roundtrip"},
        invariants=(
            custom(
                "out-matches-return",
                lambda r, expected=expected: None
                if np.array_equal(r, expected)
                else "out= result differs from the freshly-allocated result",
            ),
        ),
    )
    bad_out = np.empty((3, 7), dtype=np.float64)
    yield Case(
        GROUP,
        "evaluate_out_wrong_shape",
        Bezier.evaluate,
        lambda: bezier_out.evaluate(pts_out, out=bad_out),
        {"kind": "out-wrong-shape"},
    )


def _evaluate_derivatives_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`~pantr.bezier.Bezier.evaluate_derivatives` cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile derivative evaluation.
    """
    deriv_degrees = degrees(profile) if profile is Profile.FULL else degrees(profile)[:1]
    for dtype in dtypes(profile):
        for degree in deriv_degrees:
            for rational in (False, True) if profile is Profile.FULL else (False,):
                cp = _random_control_points(
                    2, 1, degree, dtype, offset=3000 + degree, rational=rational
                )
                bezier = Bezier(cp, is_rational=rational)
                pts = np.array([0.1, 0.5, 0.9], dtype=dtype)
                # `sorted(set(...))` avoids a duplicate order (e.g. degree == 0
                # collides 0 and degree) producing two cases with the same label.
                orders = sorted({0, 1, degree, degree + 3})
                for order in orders:
                    if order == 0 and degree == 0:
                        continue
                    tag = f"p{degree}_{dtype}_{'rat' if rational else 'nonrat'}_o{order}"
                    yield Case(
                        GROUP,
                        f"evaluate_derivatives_{tag}",
                        Bezier.evaluate_derivatives,
                        lambda bezier=bezier, pts=pts, order=order: bezier.evaluate_derivatives(
                            pts, order
                        ),
                        {
                            "degree": degree,
                            "dtype": str(dtype),
                            "rational": rational,
                            "order": order,
                        },
                        arrays={"control_points": cp},
                    )

    # Mismatched orders length and negative order: documented rejections.
    cp_bad = _random_control_points(1, 2, 2, np.dtype(np.float64), offset=3900)
    bezier_bad = Bezier(cp_bad)
    pts_bad = np.array([[0.2, 0.3]], dtype=np.float64)
    yield Case(
        GROUP,
        "evaluate_derivatives_orders_length_mismatch",
        Bezier.evaluate_derivatives,
        lambda: bezier_bad.evaluate_derivatives(pts_bad, [1, 2, 3]),
        {"kind": "orders-length-mismatch"},
    )
    yield Case(
        GROUP,
        "evaluate_derivatives_negative_order",
        Bezier.evaluate_derivatives,
        lambda: bezier_bad.evaluate_derivatives(pts_bad, [-1, 0]),
        {"kind": "negative-order"},
    )


# ---------------------------------------------------------------------------
# Degree elevation, reduction, and minimization
# ---------------------------------------------------------------------------


def _elevate_degree_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`~pantr.bezier.Bezier.elevate_degree` cases.

    Invariant 1: elevation leaves the curve pointwise unchanged (the elevated
    control points are convex combinations of the originals).

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile elevation.
    """
    elevate_degrees = (0, 1, 3, 15) if profile is Profile.FULL else (0, 1)
    for dtype in dtypes(profile):
        for degree in elevate_degrees:
            for inc in (1, 5) if profile is Profile.FULL else (1,):
                for mag_name, scale, translate in _magnitude_variants(profile):
                    cp = _random_control_points(
                        2,
                        1,
                        degree,
                        dtype,
                        offset=4000 + degree + inc,
                        scale=scale,
                        translate=translate,
                    )
                    bezier = Bezier(cp)
                    test_pts = np.array([0.0, 0.13, 0.5, 0.87, 1.0], dtype=dtype)
                    expected = bezier.evaluate(test_pts)
                    tol = _convex_combination_tol(degree + inc, dtype, cp)
                    yield Case(
                        GROUP,
                        f"elevate_p{degree}_inc{inc}_{dtype}_{mag_name}",
                        Bezier.elevate_degree,
                        lambda bezier=bezier, inc=inc: bezier.elevate_degree(inc),
                        {
                            "degree": degree,
                            "increment": inc,
                            "dtype": str(dtype),
                            "magnitude": mag_name,
                        },
                        invariants=(
                            custom(
                                "elevate-preserves-values",
                                lambda r, test_pts=test_pts, expected=expected, tol=tol: (
                                    _delta_failure("elevate", r.evaluate(test_pts), expected, tol)
                                ),
                            ),
                        ),
                        arrays={"control_points": cp},
                    )

    # Documented rejections: negative increment, all-zero increments, length mismatch.
    cp_nd = _random_control_points(1, 2, 2, np.dtype(np.float64), offset=4900)
    bezier_nd = Bezier(cp_nd)
    yield Case(
        GROUP,
        "elevate_negative_increment",
        Bezier.elevate_degree,
        lambda: bezier_nd.elevate_degree(-1),
        {"kind": "negative-increment"},
    )
    yield Case(
        GROUP,
        "elevate_all_zero_increments",
        Bezier.elevate_degree,
        lambda: bezier_nd.elevate_degree([0, 0]),
        {"kind": "all-zero-increments"},
    )
    yield Case(
        GROUP,
        "elevate_length_mismatch",
        Bezier.elevate_degree,
        lambda: bezier_nd.elevate_degree([1, 1, 1]),
        {"kind": "length-mismatch"},
    )


def _reduce_degree_invariant(
    endpoints: npt.NDArray[np.floating[Any]],
    expected_ends: npt.NDArray[np.floating[Any]],
    *,
    to_degree_zero: bool,
    mean_cp: npt.NDArray[np.floating[Any]] | None = None,
    mean_tol: float = 0.0,
) -> Invariant:
    """Build the reduce_degree correctness invariant for one case.

    The endpoint-bit-exactness claim (``_bezier.py:319-320``) is explicit about
    one carve-out: "A reduction to degree 0 cannot honour two conditions with
    one coefficient and returns the plain L2 projection, the mean of the
    control points." So a reduction whose *target* degree is 0 is checked
    against the mean instead of the (inapplicable) bit-exact endpoints.

    Args:
        endpoints (npt.NDArray[np.floating[Any]]): ``[0.0, 1.0]`` in the
            operand's dtype.
        expected_ends (npt.NDArray[np.floating[Any]]): The original curve
            evaluated at the endpoints.
        to_degree_zero (bool): Whether this reduction's target degree is 0.
        mean_cp (npt.NDArray[np.floating[Any]] | None): The mean of the
            original control points, required when ``to_degree_zero``.
        mean_tol (float): Absolute tolerance for the degree-0 mean-projection
            check.

    Returns:
        Invariant: The check appropriate to this reduction's target degree.
    """
    if to_degree_zero:

        def predicate_mean(result: Bezier) -> str | None:
            got = np.asarray(result.control_points[0], dtype=np.float64)
            return _delta_failure("reduce-to-degree-0-mean", got, mean_cp, mean_tol)

        return custom("reduce-degree-zero-is-mean-projection", predicate_mean)

    def predicate_endpoints(result: Bezier) -> str | None:
        got = result.evaluate(endpoints)
        if np.array_equal(got, expected_ends):
            return None
        return f"endpoints differ: got {got}, expected {expected_ends}"

    return custom("reduce-endpoints-bit-exact", predicate_endpoints)


def _reduce_degree_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`~pantr.bezier.Bezier.reduce_degree` cases.

    Invariant 2: reduction reproduces the endpoint values bit for bit
    (``_bezier.py`` states this exactly) -- except when the *target* degree is
    0, where the same docstring documents a different exact property (the mean
    of the control points).

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile reduction.
    """
    reduce_degrees = (1, 2, 3, 15, 62) if profile is Profile.FULL else (1, 2)
    for dtype in dtypes(profile):
        for degree in reduce_degrees:
            for dec in (1, degree) if profile is Profile.FULL else (1,):
                for mag_name, scale, translate in _magnitude_variants(profile):
                    cp = _random_control_points(
                        2,
                        1,
                        degree,
                        dtype,
                        offset=5000 + degree + dec,
                        scale=scale,
                        translate=translate,
                    )
                    bezier = Bezier(cp)
                    endpoints = np.array([0.0, 1.0], dtype=dtype)
                    expected_ends = bezier.evaluate(endpoints)
                    to_zero = degree - dec == 0
                    mean_cp = np.mean(cp, axis=0, dtype=np.float64) if to_zero else None
                    # Averaging `degree + 1` coefficients carries O(degree) * eps
                    # of rounding relative to their magnitude; factor 8 as elsewhere.
                    mean_tol = _convex_combination_tol(degree, dtype, cp) if to_zero else 0.0
                    invariant = _reduce_degree_invariant(
                        endpoints,
                        expected_ends,
                        to_degree_zero=to_zero,
                        mean_cp=mean_cp,
                        mean_tol=mean_tol,
                    )
                    yield Case(
                        GROUP,
                        f"reduce_p{degree}_dec{dec}_{dtype}_{mag_name}",
                        Bezier.reduce_degree,
                        lambda bezier=bezier, dec=dec: bezier.reduce_degree(dec),
                        {
                            "degree": degree,
                            "decrement": dec,
                            "dtype": str(dtype),
                            "magnitude": mag_name,
                        },
                        invariants=(invariant,),
                        arrays={"control_points": cp},
                    )

    # Documented rejections: decrement exceeds degree, negative, all-zero.
    cp = _random_control_points(1, 1, 3, np.dtype(np.float64), offset=5900)
    bezier = Bezier(cp)
    yield Case(
        GROUP,
        "reduce_decrement_exceeds_degree",
        Bezier.reduce_degree,
        lambda: bezier.reduce_degree(4),
        {"kind": "decrement-exceeds-degree"},
    )
    yield Case(
        GROUP,
        "reduce_negative_decrement",
        Bezier.reduce_degree,
        lambda: bezier.reduce_degree(-1),
        {"kind": "negative-decrement"},
    )
    yield Case(
        GROUP,
        "reduce_all_zero_decrements",
        Bezier.reduce_degree,
        lambda: bezier.reduce_degree(0),
        {"kind": "all-zero-decrement"},
    )


def _degree_reduction_error_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`~pantr.bezier.Bezier.degree_reduction_error` cases.

    Invariant 11: the value is the exact ``L2`` error, so it must be finite,
    non-negative always, and at the rounding floor when the curve genuinely has
    lower degree (built here by elevating a lower-degree curve by one).

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile error computation.
    """
    error_degrees = (1, 2, 3, 15) if profile is Profile.FULL else (1, 2)
    for dtype in dtypes(profile):
        eps = _eps(dtype)
        for degree in error_degrees:
            base_cp = _random_control_points(1, 1, degree, dtype, offset=6000 + degree)
            base = Bezier(base_cp)
            elevated = base.elevate_degree(1)
            tol = 100.0 * eps * _magnitude(base_cp)
            yield Case(
                GROUP,
                f"reduction_error_exact_p{degree}_{dtype}",
                Bezier.degree_reduction_error,
                lambda b=elevated: b.degree_reduction_error(1),
                {"degree": degree, "dtype": str(dtype), "kind": "exactly-reducible"},
                invariants=(
                    custom(
                        "error-at-rounding-floor",
                        lambda r, tol=tol: None
                        if (np.isfinite(r) and 0.0 <= r <= tol)
                        else f"error {r!r} not in [0, {tol:.3e}]",
                    ),
                ),
            )
            generic_decs = range(1, degree + 1) if profile is Profile.FULL else (1,)
            for dec in generic_decs:
                generic_cp = _random_control_points(1, 1, degree, dtype, offset=6100 + degree + dec)
                generic = Bezier(generic_cp)
                yield Case(
                    GROUP,
                    f"reduction_error_generic_p{degree}_dec{dec}_{dtype}",
                    Bezier.degree_reduction_error,
                    lambda b=generic, dec=dec: b.degree_reduction_error(dec),
                    {"degree": degree, "decrement": dec, "dtype": str(dtype)},
                    invariants=(
                        custom(
                            "error-finite-nonneg",
                            lambda r: None if (np.isfinite(r) and r >= 0.0) else f"got {r!r}",
                        ),
                    ),
                )


def _minimize_degree_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`~pantr.bezier.Bezier.minimize_degree` cases.

    The only genuinely claimed property without re-implementing the algorithm
    is that the result's degree never exceeds the original's.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile minimization.
    """
    for dtype in dtypes(profile):
        for degree in degrees(profile):
            if degree == 0:
                continue
            cp = _random_control_points(1, 1, degree, dtype, offset=7000 + degree)
            bezier = Bezier(cp)
            yield Case(
                GROUP,
                f"minimize_random_p{degree}_{dtype}",
                Bezier.minimize_degree,
                lambda bezier=bezier: bezier.minimize_degree(),
                {"degree": degree, "dtype": str(dtype), "kind": "random"},
                invariants=(
                    custom(
                        "degree-not-increased",
                        lambda r, degree=degree: None
                        if r.degree[0] <= degree
                        else f"got degree {r.degree[0]} > original {degree}",
                    ),
                ),
                arrays={"control_points": cp},
            )
            if profile is Profile.FULL:
                # A genuinely reducible curve: a straight line elevated to `degree`.
                line_cp = np.linspace(0.0, 1.0, degree + 1, dtype=dtype).reshape(-1, 1)
                line = Bezier(line_cp)
                yield Case(
                    GROUP,
                    f"minimize_reducible_p{degree}_{dtype}",
                    Bezier.minimize_degree,
                    lambda line=line: line.minimize_degree(),
                    {"degree": degree, "dtype": str(dtype), "kind": "reducible-line"},
                    invariants=(
                        custom(
                            "degree-reduced",
                            lambda r, degree=degree: None
                            if r.degree[0] <= degree
                            else f"got degree {r.degree[0]} > original {degree}",
                        ),
                    ),
                )
    yield Case(
        GROUP,
        "minimize_explicit_tol",
        Bezier.minimize_degree,
        lambda: Bezier(np.linspace(0.0, 1.0, 16, dtype=np.float64).reshape(-1, 1)).minimize_degree(
            tol=1e-3
        ),
        {"kind": "explicit-tol"},
    )


# ---------------------------------------------------------------------------
# Split and restrict
# ---------------------------------------------------------------------------


def _split_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`~pantr.bezier.Bezier.split` cases.

    Invariant 3: both halves, reparametrized, reproduce the original.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile split.
    """
    split_degrees = (1, 2, 3, 15, 62) if profile is Profile.FULL else (1, 3)
    split_values = (0.5, 0.1, 0.9, 1e-6) if profile is Profile.FULL else (0.5,)
    for dtype in dtypes(profile):
        for degree in split_degrees:
            for value in split_values:
                for mag_name, scale, translate in _magnitude_variants(profile):
                    cp = _random_control_points(
                        2, 1, degree, dtype, offset=8000 + degree, scale=scale, translate=translate
                    )
                    bezier = Bezier(cp)
                    tol = _convex_combination_tol(degree, dtype, cp)
                    left_pts = np.array([0.1, 0.5, 0.9], dtype=dtype)
                    right_pts = np.array([0.1, 0.5, 0.9], dtype=dtype)
                    expected_left = bezier.evaluate((left_pts * value).astype(dtype))
                    expected_right = bezier.evaluate(
                        (value + right_pts * (1.0 - value)).astype(dtype)
                    )
                    yield Case(
                        GROUP,
                        f"split_p{degree}_v{value:g}_{dtype}_{mag_name}",
                        Bezier.split,
                        lambda bezier=bezier, value=value: bezier.split(0, value),
                        {
                            "degree": degree,
                            "value": value,
                            "dtype": str(dtype),
                            "magnitude": mag_name,
                        },
                        invariants=(
                            custom(
                                "split-reproduces-original",
                                lambda r,
                                left_pts=left_pts,
                                right_pts=right_pts,
                                expected_left=expected_left,
                                expected_right=expected_right,
                                tol=tol: _split_failure(
                                    r, left_pts, right_pts, expected_left, expected_right, tol
                                ),
                            ),
                        ),
                        arrays={"control_points": cp},
                    )

    cp_1d = _random_control_points(1, 1, 3, np.dtype(np.float64), offset=8900)
    bezier_1d = Bezier(cp_1d)
    for boundary_value, tag in ((0.0, "left"), (1.0, "right")):
        yield Case(
            GROUP,
            f"split_at_boundary_{tag}",
            Bezier.split,
            lambda bezier_1d=bezier_1d, boundary_value=boundary_value: bezier_1d.split(
                0, boundary_value
            ),
            {"kind": f"boundary-{tag}", "value": boundary_value},
        )
    yield Case(
        GROUP,
        "split_direction_out_of_range",
        Bezier.split,
        lambda: bezier_1d.split(1, 0.5),
        {"kind": "direction-out-of-range"},
    )


def _split_failure(  # noqa: PLR0913 -- two halves each need points and expected values, plus tol
    result: tuple[Bezier, Bezier],
    left_pts: npt.NDArray[np.floating[Any]],
    right_pts: npt.NDArray[np.floating[Any]],
    expected_left: npt.NDArray[np.floating[Any]],
    expected_right: npt.NDArray[np.floating[Any]],
    tol: float,
) -> str | None:
    """Check that both halves of a split reproduce the original mapping.

    Args:
        result (tuple[Bezier, Bezier]): The ``(left, right)`` pair returned by ``split``.
        left_pts (npt.NDArray[np.floating[Any]]): Test parameters on the left half.
        right_pts (npt.NDArray[np.floating[Any]]): Test parameters on the right half.
        expected_left (npt.NDArray[np.floating[Any]]): Original evaluated at the
            mapped left parameters.
        expected_right (npt.NDArray[np.floating[Any]]): Original evaluated at the
            mapped right parameters.
        tol (float): Absolute tolerance.

    Returns:
        str | None: ``None`` when both halves match, else a short message.
    """
    left, right = result
    got_left = left.evaluate(left_pts)
    got_right = right.evaluate(right_pts)
    worst_left = float(np.max(np.abs(got_left - expected_left)))
    worst_right = float(np.max(np.abs(got_right - expected_right)))
    worst = max(worst_left, worst_right)
    if worst > tol:
        return (
            f"max|delta| = {worst:.3e} > {tol:.3e} (left={worst_left:.3e}, right={worst_right:.3e})"
        )
    return None


def _restrict_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`~pantr.bezier.Bezier.restrict` cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile restriction.
    """
    for dtype in dtypes(profile):
        for degree in (1, 3, 15) if profile is Profile.FULL else (1,):
            cp = _random_control_points(2, 1, degree, dtype, offset=9000 + degree)
            bezier = Bezier(cp)
            bounds = (0.2, 0.8)
            test_pts = np.array([0.0, 0.5, 1.0], dtype=dtype)
            expected = bezier.evaluate(
                (bounds[0] + test_pts * (bounds[1] - bounds[0])).astype(dtype)
            )
            tol = _convex_combination_tol(degree, dtype, cp)
            yield Case(
                GROUP,
                f"restrict_p{degree}_{dtype}",
                Bezier.restrict,
                lambda bezier=bezier, bounds=bounds: bezier.restrict(bounds),
                {"degree": degree, "dtype": str(dtype), "bounds": bounds},
                invariants=(
                    custom(
                        "restrict-reproduces-subinterval",
                        lambda r, test_pts=test_pts, expected=expected, tol=tol: _delta_failure(
                            "restrict", r.evaluate(test_pts), expected, tol
                        ),
                    ),
                ),
                arrays={"control_points": cp},
            )

    cp_1d = _random_control_points(1, 1, 2, np.dtype(np.float64), offset=9900)
    bezier_1d = Bezier(cp_1d)
    yield Case(
        GROUP,
        "restrict_zero_width",
        Bezier.restrict,
        lambda: bezier_1d.restrict((0.4, 0.4)),
        {"kind": "zero-width"},
    )
    yield Case(
        GROUP,
        "restrict_out_of_domain",
        Bezier.restrict,
        lambda: bezier_1d.restrict((-0.1, 0.5)),
        {"kind": "out-of-domain"},
    )


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


def _compose_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`~pantr.bezier.Bezier.compose` cases.

    Invariant 4: ``outer.compose(inner).evaluate(t) == outer.evaluate(inner.evaluate(t))``.
    Combined degree is kept below 62 in the main crossing -- 62 is the known
    ``_bincoeff`` int64 overflow cliff reached through this path
    (``_bezier_compose.py``) -- except for one labelled case documenting the
    boundary at the combined degree 61, one step below the cliff.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile composition.
    """
    for dtype in dtypes(profile):
        # 1-D outer, 1-D inner: the common case, uses the 1D bincoeff kernel.
        for p, q in ((1, 3), (3, 1), (2, 5)) if profile is Profile.FULL else ((1, 2),):
            outer_cp = _random_control_points(1, 1, p, dtype, offset=10000 + p)
            inner_cp = np.linspace(0.0, 1.0, q + 1, dtype=dtype).reshape(-1, 1)
            outer = Bezier(outer_cp)
            inner = Bezier(inner_cp)
            ts = np.array([0.1, 0.4, 0.6, 0.9], dtype=dtype)
            expected = outer.evaluate(inner.evaluate(ts))
            magnitude = max(_magnitude(outer_cp), _magnitude(inner_cp))
            tol = _product_tol(p, q, dtype, magnitude, chain_depth=p)
            yield Case(
                GROUP,
                f"compose_1d1d_p{p}q{q}_{dtype}",
                Bezier.compose,
                lambda outer=outer, inner=inner: outer.compose(inner),
                {"outer_degree": p, "inner_degree": q, "dtype": str(dtype), "kind": "1d-inner"},
                invariants=(_compose_invariant(ts, expected, tol),),
                arrays={"outer_control_points": outer_cp, "inner_control_points": inner_cp},
            )

        if profile is Profile.FULL:
            # A different route to high combined degree: multi-D inner uses the
            # NumPy nD product path (`_bernstein_product_coefficients_nd`,
            # `math.comb`-based), not the Numba 1D bincoeff kernel -- so this is
            # a genuinely different mechanism, not the known cliff.
            outer_cp_nd = _random_control_points(1, 1, 30, dtype, offset=10100)
            inner_cp_nd = _random_control_points(1, 2, 3, dtype, offset=10101)
            outer_nd = Bezier(outer_cp_nd)
            inner_nd = Bezier(inner_cp_nd)
            ts_nd = np.array([[0.2, 0.3], [0.6, 0.8]], dtype=dtype)
            expected_nd = outer_nd.evaluate(inner_nd.evaluate(ts_nd))
            magnitude_nd = max(_magnitude(outer_cp_nd), _magnitude(inner_cp_nd))
            tol_nd = _product_tol(30, 3, dtype, magnitude_nd, chain_depth=30)
            yield Case(
                GROUP,
                f"compose_1d_nd_inner_{dtype}",
                Bezier.compose,
                lambda outer_nd=outer_nd, inner_nd=inner_nd: outer_nd.compose(inner_nd),
                {"outer_degree": 30, "inner_degree": 3, "dtype": str(dtype), "kind": "nd-inner"},
                invariants=(_compose_invariant(ts_nd, expected_nd, tol_nd),),
                arrays={"outer_control_points": outer_cp_nd, "inner_control_points": inner_cp_nd},
            )

            # Boundary marker: combined degree exactly 61, one below the known
            # `_bincoeff` int64-overflow cliff at 62 -- documents that the cliff
            # is sharp, without re-triggering the already-logged bug.
            outer_boundary_cp = _random_control_points(1, 1, 1, dtype, offset=10200)
            inner_boundary_cp = np.linspace(0.0, 1.0, 62, dtype=dtype).reshape(-1, 1)
            outer_boundary = Bezier(outer_boundary_cp)
            inner_boundary = Bezier(inner_boundary_cp)
            ts_boundary = np.array([0.3, 0.7], dtype=dtype)
            expected_boundary = outer_boundary.evaluate(inner_boundary.evaluate(ts_boundary))
            magnitude_boundary = max(_magnitude(outer_boundary_cp), _magnitude(inner_boundary_cp))
            tol_boundary = _product_tol(1, 61, dtype, magnitude_boundary, chain_depth=1)
            yield Case(
                GROUP,
                f"compose_boundary_combined_degree_61_{dtype}",
                Bezier.compose,
                lambda outer_boundary=outer_boundary, inner_boundary=inner_boundary: (
                    outer_boundary.compose(inner_boundary)
                ),
                {"combined_degree": 61, "dtype": str(dtype), "kind": "cliff-boundary"},
                invariants=(_compose_invariant(ts_boundary, expected_boundary, tol_boundary),),
                arrays={
                    "outer_control_points": outer_boundary_cp,
                    "inner_control_points": inner_boundary_cp,
                },
            )

    # Documented rejections.
    outer_rat_cp = _random_control_points(
        1, 1, 2, np.dtype(np.float64), offset=10900, rational=True
    )
    inner_plain_cp = _random_control_points(1, 1, 2, np.dtype(np.float64), offset=10901)
    outer_rat = Bezier(outer_rat_cp, is_rational=True)
    inner_plain = Bezier(inner_plain_cp)
    yield Case(
        GROUP,
        "compose_rational_outer_rejected",
        Bezier.compose,
        lambda: outer_rat.compose(inner_plain),
        {"kind": "rational-outer"},
    )
    outer_plain = Bezier(_random_control_points(1, 1, 2, np.dtype(np.float64), offset=10902))
    inner_rank_mismatch = Bezier(
        _random_control_points(2, 1, 2, np.dtype(np.float64), offset=10903)
    )
    yield Case(
        GROUP,
        "compose_rank_mismatch",
        Bezier.compose,
        lambda: outer_plain.compose(inner_rank_mismatch),
        {"kind": "rank-mismatch"},
    )
    outer_f32 = Bezier(_random_control_points(1, 1, 2, np.dtype(np.float32), offset=10904))
    inner_f64 = Bezier(_random_control_points(1, 1, 2, np.dtype(np.float64), offset=10905))
    yield Case(
        GROUP,
        "compose_dtype_mismatch",
        Bezier.compose,
        lambda: outer_f32.compose(inner_f64),
        {"kind": "dtype-mismatch"},
    )


def _compose_invariant(
    ts: npt.NDArray[np.floating[Any]], expected: npt.NDArray[np.floating[Any]], tol: float
) -> Invariant:
    """Build the compose-matches-evaluation invariant for one case.

    Args:
        ts (npt.NDArray[np.floating[Any]]): Test parameters, in the inner's
            parametric space.
        expected (npt.NDArray[np.floating[Any]]): ``outer.evaluate(inner.evaluate(ts))``.
        tol (float): Absolute tolerance.

    Returns:
        Invariant: Check comparing the composed Bezier's own evaluation to
        ``expected``.
    """

    def predicate(result: Bezier) -> str | None:
        return _delta_failure("compose-vs-eval-through-inner", result.evaluate(ts), expected, tol)

    return custom("compose-matches-eval", predicate)


# ---------------------------------------------------------------------------
# Multiply
# ---------------------------------------------------------------------------


def _multiply_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`~pantr.bezier.Bezier.multiply` cases.

    Invariant 5: ``(f * g).evaluate(t) == f.evaluate(t) * g.evaluate(t)``.
    Multiply uses ``math.comb`` (arbitrary-precision Python ints), not the
    Numba ``_bincoeff`` kernel, so it is not subject to the known int64 cliff;
    degrees here are kept moderate to avoid unrelated float64 conditioning noise.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile product.
    """
    degree_pairs = ((1, 1), (3, 5), (15, 15)) if profile is Profile.FULL else ((1, 1),)
    for dtype in dtypes(profile):
        for p, q in degree_pairs:
            for mag_name, scale, translate in _magnitude_variants(profile):
                cp_f = _random_control_points(
                    1, 1, p, dtype, offset=11000 + p, scale=scale, translate=translate
                )
                cp_g = _random_control_points(
                    1, 1, q, dtype, offset=11000 + q + 1, scale=scale, translate=translate
                )
                f = Bezier(cp_f)
                g = Bezier(cp_g)
                ts = np.array([0.05, 0.3, 0.6, 0.95], dtype=dtype)
                expected = f.evaluate(ts) * g.evaluate(ts)
                # A pointwise product's own magnitude is the *product* of both
                # factors' magnitudes, not their max -- passing the max badly
                # underestimates the tolerance once either factor is scaled up.
                tol = _product_tol(p, q, dtype, _magnitude(cp_f) * _magnitude(cp_g))
                yield Case(
                    GROUP,
                    f"multiply_p{p}q{q}_{dtype}_{mag_name}",
                    Bezier.multiply,
                    lambda f=f, g=g: f.multiply(g),
                    {"p": p, "q": q, "dtype": str(dtype), "magnitude": mag_name},
                    invariants=(
                        custom(
                            "multiply-matches-pointwise-product",
                            lambda r, ts=ts, expected=expected, tol=tol: _delta_failure(
                                "multiply", r.evaluate(ts), expected, tol
                            ),
                        ),
                    ),
                    arrays={"f_control_points": cp_f, "g_control_points": cp_g},
                )

    f_operator = Bezier.__mul__
    f64 = Bezier(_random_control_points(1, 1, 2, np.dtype(np.float64), offset=11900))
    g64 = Bezier(_random_control_points(1, 1, 2, np.dtype(np.float64), offset=11901))
    ts_op = np.array([0.3, 0.7], dtype=np.float64)
    expected_op = f64.evaluate(ts_op) * g64.evaluate(ts_op)
    tol_op = _product_tol(
        2, 2, np.dtype(np.float64), _magnitude(f64.control_points) * _magnitude(g64.control_points)
    )
    yield Case(
        GROUP,
        "multiply_operator",
        f_operator,
        lambda: f64 * g64,
        {"kind": "dunder-mul"},
        invariants=(
            custom(
                "operator-matches-product",
                lambda r, ts_op=ts_op, expected_op=expected_op, tol_op=tol_op: _delta_failure(
                    "operator", r.evaluate(ts_op), expected_op, tol_op
                ),
            ),
        ),
    )

    rank_mismatch = Bezier(_random_control_points(2, 1, 2, np.dtype(np.float64), offset=11902))
    yield Case(
        GROUP,
        "multiply_rank_mismatch",
        Bezier.multiply,
        lambda: f64.multiply(rank_mismatch),
        {"kind": "rank-mismatch"},
    )
    f32 = Bezier(_random_control_points(1, 1, 2, np.dtype(np.float32), offset=11903))
    yield Case(
        GROUP,
        "multiply_dtype_mismatch",
        Bezier.multiply,
        lambda: f64.multiply(f32),
        {"kind": "dtype-mismatch"},
    )


# ---------------------------------------------------------------------------
# Derivative
# ---------------------------------------------------------------------------


def _derivative_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`~pantr.bezier.Bezier.derivative` cases.

    Invariant 6: the returned hodograph agrees with a central difference on
    ``evaluate``, to the finite-difference noise floor.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile derivative.
    """
    derivative_degrees = (1, 2, 3, 15) if profile is Profile.FULL else (1, 2)
    for dtype in dtypes(profile):
        for degree in derivative_degrees:
            for rational in (False, True) if profile is Profile.FULL else (False,):
                for keep_degree in (False, True) if profile is Profile.FULL else (False,):
                    for mag_name, scale, translate in _magnitude_variants(profile):
                        cp = _random_control_points(
                            1,
                            1,
                            degree,
                            dtype,
                            offset=12000 + degree,
                            rational=rational,
                            scale=scale,
                            translate=translate,
                        )
                        bezier = Bezier(cp, is_rational=rational)
                        h, tol = _fd_step_and_tol(dtype, cp, degree)
                        ts = np.array([0.2, 0.5, 0.8], dtype=dtype)
                        ts_safe = np.clip(ts, h, 1.0 - h).astype(dtype)
                        central = (
                            bezier.evaluate((ts_safe + h).astype(dtype))
                            - bezier.evaluate((ts_safe - h).astype(dtype))
                        ) / (2.0 * h)
                        tag = (
                            f"p{degree}_{dtype}_{'rat' if rational else 'nonrat'}"
                            f"_{'keepdeg' if keep_degree else 'lower'}_{mag_name}"
                        )
                        yield Case(
                            GROUP,
                            f"derivative_{tag}",
                            Bezier.derivative,
                            lambda bezier=bezier, keep_degree=keep_degree: bezier.derivative(
                                keep_degree=keep_degree
                            ),
                            {
                                "degree": degree,
                                "dtype": str(dtype),
                                "rational": rational,
                                "keep_degree": keep_degree,
                                "magnitude": mag_name,
                            },
                            invariants=(
                                custom(
                                    "derivative-matches-central-difference",
                                    lambda r, ts_safe=ts_safe, central=central, tol=tol: (
                                        _delta_failure(
                                            "derivative", r.evaluate(ts_safe), central, tol
                                        )
                                    ),
                                ),
                            ),
                            arrays={"control_points": cp},
                        )

    cp_const = _random_control_points(1, 2, 0, np.dtype(np.float64), offset=12900)
    bezier_const = Bezier(cp_const)
    yield Case(
        GROUP,
        "derivative_degree_zero_rejected",
        Bezier.derivative,
        lambda: bezier_const.derivative(direction=0),
        {"kind": "degree-zero-direction"},
    )
    yield Case(
        GROUP,
        "derivative_direction_out_of_range",
        Bezier.derivative,
        lambda: bezier_const.derivative(direction=5),
        {"kind": "direction-out-of-range"},
    )


# ---------------------------------------------------------------------------
# Root finding
# ---------------------------------------------------------------------------


def _exact_bernstein_from_roots(
    roots: Sequence[Fraction], dtype: np.dtype[np.float32 | np.float64]
) -> npt.NDArray[np.float32 | np.float64]:
    """Build exact Bernstein coefficients for a polynomial with known roots.

    Expands ``prod_i (t - r_i)`` into power-basis coefficients using exact
    rational arithmetic, converts to the Bernstein basis of the same degree via
    the standard power-to-Bernstein change of basis (``b_j = sum_{i<=j}
    C(j,i)/C(n,i) * a_i``, Farouki & Rajan 2012), and rounds to ``dtype`` once.
    The algorithm under test never sees this derivation, so the resulting
    coefficients are an independent root-finding oracle.

    Args:
        roots (Sequence[Fraction]): Distinct exact roots in ``(0, 1)``.
        dtype (np.dtype[np.float32 | np.float64]): Target floating dtype.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Bernstein coefficients, shape
        ``(len(roots) + 1,)``.
    """
    n = len(roots)
    poly: list[Fraction] = [Fraction(1)]
    for r in roots:
        new_poly = [Fraction(0)] * (len(poly) + 1)
        for i, c in enumerate(poly):
            new_poly[i] += c * (-r)
            new_poly[i + 1] += c
        poly = new_poly

    bernstein: list[Fraction] = []
    for j in range(n + 1):
        acc = Fraction(0)
        for i in range(j + 1):
            acc += Fraction(math.comb(j, i), math.comb(n, i)) * poly[i]
        bernstein.append(acc)

    return np.array([float(v) for v in bernstein], dtype=dtype)


def _well_separated_roots(n: int) -> list[Fraction]:
    """Build ``n`` roots evenly spaced in ``(0, 1)``, well separated.

    Args:
        n (int): Number of roots.

    Returns:
        list[Fraction]: Exact roots ``k / (n + 1)`` for ``k = 1, ..., n``.
    """
    return [Fraction(k, n + 1) for k in range(1, n + 1)]


def _clustered_roots(n: int) -> list[Fraction]:
    """Build ``n`` roots with one nearly-duplicate pair (a known dedup edge case).

    Args:
        n (int): Number of roots, ``>= 2``.

    Returns:
        list[Fraction]: Exact roots, mostly well separated but with two
        roots ``1e-6`` apart.
    """
    roots = _well_separated_roots(n)
    roots[-1] = roots[-2] + Fraction(1, 1_000_000)
    return roots


def _monotone_control_points(
    degree: int, dtype: np.dtype[np.float32 | np.float64], *, sign_change: bool
) -> npt.NDArray[np.float32 | np.float64]:
    """Build control points for a Bezier certified monotone by construction.

    Strictly increasing control points guarantee a monotone Bezier: the
    hodograph's control points are positive multiples of the (all-positive)
    forward differences, so the derivative never changes sign. This is an
    independent sufficient condition, not the algorithm's own logic.

    Args:
        degree (int): Polynomial degree.
        dtype (np.dtype[np.float32 | np.float64]): Target floating dtype.
        sign_change (bool): If ``True``, the ramp crosses zero (root exists);
            if ``False``, it stays strictly positive (no root).

    Returns:
        npt.NDArray[np.float32 | np.float64]: Control points, shape ``(degree + 1, 1)``.
    """
    lo, hi = (-1.0, 1.0) if sign_change else (0.2, 1.0)
    return np.linspace(lo, hi, degree + 1, dtype=dtype).reshape(-1, 1)


def _roots_bounded_invariant(bezier: Bezier) -> Invariant:
    """Build the invariant that ``find_roots`` returns bounded, genuine roots.

    Args:
        bezier (Bezier): The curve whose roots are being found.

    Returns:
        Invariant: Check enforcing the count bound, containment, and residual.
    """
    degree = bezier.degree[0]
    cp = bezier.control_points[:, 0]
    bound = _bernstein_eval_noise(degree, bezier.dtype, cp)

    def predicate(result: object) -> str | None:
        roots = np.asarray(result)
        if roots.ndim != 1:
            return f"expected a 1D roots array, got shape {roots.shape}"
        if roots.size > max(degree, 1):
            return f"found {roots.size} roots but degree is {degree}"
        if roots.size and (np.any(roots < 0.0) or np.any(roots > 1.0)):
            return f"root outside [0, 1]: {roots}"
        if roots.size:
            vals = np.atleast_1d(bezier.evaluate(roots.astype(bezier.dtype)))
            worst = float(np.max(np.abs(vals)))
            if worst > bound:
                return f"max|f(root)| = {worst:.3e} > {bound:.3e}"
        return None

    return custom("roots-bounded-and-genuine", predicate)


def _root_finding_cases(profile: Profile) -> Iterator[Case]:
    """Yield :func:`~pantr.bezier.find_roots` and :func:`~pantr.bezier.find_monotone_root` cases.

    Invariant 7: at most ``degree`` roots, each in ``[0, 1]``, each a genuine
    root to the derived noise floor. Invariant 8: on a monotone sign-changing
    curve, a bounded root; on a non-crossing one, the documented ``NaN`` sentinel.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile root-finding call.
    """
    root_degrees = (3, 5, 8, 15, 61) if profile is Profile.FULL else (3,)
    for dtype in dtypes(profile):
        for n in root_degrees:
            coeff = _exact_bernstein_from_roots(_well_separated_roots(n), dtype)
            bezier = Bezier(coeff.reshape(-1, 1))
            yield Case(
                GROUP,
                f"find_roots_well_separated_n{n}_{dtype}",
                find_roots,
                lambda bezier=bezier: find_roots(bezier),
                {"n_roots": n, "dtype": str(dtype), "kind": "well-separated"},
                invariants=(_roots_bounded_invariant(bezier),),
                arrays={"coefficients": coeff},
            )

        if profile is Profile.FULL:
            for n in (5, 8):
                coeff = _exact_bernstein_from_roots(_clustered_roots(n), dtype)
                bezier = Bezier(coeff.reshape(-1, 1))
                yield Case(
                    GROUP,
                    f"find_roots_clustered_n{n}_{dtype}",
                    find_roots,
                    lambda bezier=bezier: find_roots(bezier),
                    {"n_roots": n, "dtype": str(dtype), "kind": "clustered-known-dedup-edge-case"},
                    invariants=(_roots_bounded_invariant(bezier),),
                    arrays={"coefficients": coeff},
                )

        # Degenerate: identically zero (every point is a root).
        zero_cp = np.zeros((5, 1), dtype=dtype)
        zero_bezier = Bezier(zero_cp)
        yield Case(
            GROUP,
            f"find_roots_identically_zero_{dtype}",
            find_roots,
            lambda zero_bezier=zero_bezier: find_roots(zero_bezier),
            {"dtype": str(dtype), "kind": "identically-zero"},
        )

        for degree in (3, 15) if profile is Profile.FULL else (3,):
            mono_cp = _monotone_control_points(degree, dtype, sign_change=True)
            mono_bezier = Bezier(mono_cp)
            bound = _bernstein_eval_noise(degree, dtype, mono_cp[:, 0])
            yield Case(
                GROUP,
                f"find_monotone_root_crossing_p{degree}_{dtype}",
                find_monotone_root,
                lambda mono_bezier=mono_bezier: find_monotone_root(mono_bezier),
                {"degree": degree, "dtype": str(dtype), "kind": "sign-change"},
                invariants=(
                    custom(
                        "monotone-root-bounded",
                        lambda r, mono_bezier=mono_bezier, bound=bound: _monotone_root_failure(
                            r, mono_bezier, bound, expect_root=True
                        ),
                    ),
                ),
                arrays={"control_points": mono_cp},
            )

            no_cross_cp = _monotone_control_points(degree, dtype, sign_change=False)
            no_cross_bezier = Bezier(no_cross_cp)
            yield Case(
                GROUP,
                f"find_monotone_root_no_crossing_p{degree}_{dtype}",
                find_monotone_root,
                lambda no_cross_bezier=no_cross_bezier: find_monotone_root(no_cross_bezier),
                {"degree": degree, "dtype": str(dtype), "kind": "no-sign-change"},
                invariants=(
                    custom(
                        "monotone-root-nan-sentinel",
                        lambda r, no_cross_bezier=no_cross_bezier: _monotone_root_failure(
                            r, no_cross_bezier, 0.0, expect_root=False
                        ),
                    ),
                ),
                finite_inputs=False,
                arrays={"control_points": no_cross_cp},
            )

        if profile is Profile.FULL:
            # Batch mode: same-degree sequence.
            batch = [
                Bezier(_exact_bernstein_from_roots(_well_separated_roots(4), dtype).reshape(-1, 1))
                for _ in range(3)
            ]
            yield Case(
                GROUP,
                f"find_roots_batch_{dtype}",
                find_roots,
                lambda batch=batch: find_roots(batch),
                {"n_polys": len(batch), "dtype": str(dtype), "kind": "batch"},
                invariants=(
                    custom(
                        "batch-roots-bounded",
                        lambda r, batch=batch: _batch_roots_failure(r, batch),
                    ),
                ),
            )
            mono_batch = [
                Bezier(_monotone_control_points(4, dtype, sign_change=True)) for _ in range(3)
            ]
            yield Case(
                GROUP,
                f"find_monotone_root_batch_{dtype}",
                find_monotone_root,
                lambda mono_batch=mono_batch: find_monotone_root(mono_batch),
                {"n_polys": len(mono_batch), "dtype": str(dtype), "kind": "batch"},
                invariants=(expected_shape((len(mono_batch),)),),
            )

    # Documented rejections.
    yield Case(
        GROUP,
        "find_roots_wrong_type",
        find_roots,
        lambda: find_roots(42),
        {"kind": "not-a-bezier"},
    )
    surface = Bezier(_random_control_points(1, 2, 2, np.dtype(np.float64), offset=13900))
    yield Case(
        GROUP,
        "find_roots_wrong_dim",
        find_roots,
        lambda: find_roots(surface),
        {"kind": "dim-not-1"},
    )
    vector_curve = Bezier(_random_control_points(2, 1, 2, np.dtype(np.float64), offset=13901))
    yield Case(
        GROUP,
        "find_roots_wrong_rank",
        find_roots,
        lambda: find_roots(vector_curve),
        {"kind": "rank-not-1"},
    )
    scalar_curve = Bezier(_random_control_points(1, 1, 2, np.dtype(np.float64), offset=13902))
    yield Case(
        GROUP,
        "find_roots_nonpositive_tol",
        find_roots,
        lambda: find_roots(scalar_curve, tol=0.0),
        {"kind": "nonpositive-tol"},
    )
    uneven_batch = [
        Bezier(_random_control_points(1, 1, 2, np.dtype(np.float64), offset=13903)),
        Bezier(_random_control_points(1, 1, 3, np.dtype(np.float64), offset=13904)),
    ]
    yield Case(
        GROUP,
        "find_roots_batch_uneven_degree",
        find_roots,
        lambda: find_roots(uneven_batch),
        {"kind": "uneven-batch-degree"},
    )


def _monotone_root_failure(
    result: object, bezier: Bezier, bound: float, *, expect_root: bool
) -> str | None:
    """Check a monotone-root result against its expected sentinel/bounded-root behavior.

    Args:
        result (object): The value returned by ``find_monotone_root``.
        bezier (Bezier): The curve the root was sought on.
        bound (float): Residual bound (ignored when ``expect_root`` is ``False``).
        expect_root (bool): Whether a genuine root is expected.

    Returns:
        str | None: ``None`` when the result matches expectations, else a message.
    """
    if not expect_root:
        if isinstance(result, float) and math.isnan(result):
            return None
        return f"expected NaN, got {result!r}"
    if not isinstance(result, float) or not (0.0 <= result <= 1.0):
        return f"root {result!r} not a float in [0, 1]"
    val = float(np.atleast_1d(bezier.evaluate(np.array([result], dtype=bezier.dtype)))[0])
    if abs(val) > bound:
        return f"|f(root)| = {abs(val):.3e} > {bound:.3e}"
    return None


def _batch_roots_failure(
    result: tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]], batch: Sequence[Bezier]
) -> str | None:
    """Check the batch ``find_roots`` result against the count/containment/residual invariant.

    Args:
        result (tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]): The
            ``(roots, counts)`` pair returned in batch mode.
        batch (Sequence[Bezier]): The curves that were searched.

    Returns:
        str | None: ``None`` when every curve's roots are bounded and genuine,
        else the first failure message.
    """
    roots, counts = result
    for i, bezier in enumerate(batch):
        degree = bezier.degree[0]
        count = int(counts[i])
        if count > degree:
            return f"poly {i}: found {count} roots but degree is {degree}"
        row = roots[i, :count]
        if count and (np.any(row < 0.0) or np.any(row > 1.0)):
            return f"poly {i}: root outside [0, 1]: {row}"
        if count:
            bound = _bernstein_eval_noise(degree, bezier.dtype, bezier.control_points[:, 0])
            vals = np.atleast_1d(bezier.evaluate(row.astype(bezier.dtype)))
            worst = float(np.max(np.abs(vals)))
            if worst > bound:
                return f"poly {i}: max|f(root)| = {worst:.3e} > {bound:.3e}"
    return None


# ---------------------------------------------------------------------------
# Reverse, permute, transform
# ---------------------------------------------------------------------------


def _reverse_permute_transform_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`~pantr.bezier.Bezier.reverse`, ``permute_directions``, and ``transform`` cases.

    Invariant 9: applying ``reverse``/``permute_directions`` twice (with the
    inverse permutation) reproduces the original control points bit for bit.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile reverse/permute/transform.
    """
    combos = _rank_dim_combos(profile)[:3] if profile is Profile.FULL else ((1, 2),)
    for dtype in dtypes(profile):
        for rank, dim in combos:
            for degree in (1, 3) if profile is Profile.FULL else (1,):
                cp = _random_control_points(rank, dim, degree, dtype, offset=14000 + degree)
                bezier = Bezier(cp)
                directions = range(dim) if profile is Profile.FULL else (0,)
                for direction in directions:
                    yield Case(
                        GROUP,
                        f"reverse_twice_r{rank}d{dim}p{degree}_{dtype}_dir{direction}",
                        Bezier.reverse,
                        lambda bezier=bezier, direction=direction: bezier.reverse(
                            direction
                        ).reverse(direction),
                        {"rank": rank, "dim": dim, "degree": degree, "dtype": str(dtype)},
                        invariants=(_double_op_identity_invariant(cp),),
                        arrays={"control_points": cp},
                    )
                    fresh = Bezier(cp.copy())
                    yield Case(
                        GROUP,
                        f"reverse_in_place_returns_none_r{rank}d{dim}_{dtype}_dir{direction}",
                        Bezier.reverse,
                        lambda fresh=fresh, direction=direction: fresh.reverse(
                            direction, in_place=True
                        ),
                        {"rank": rank, "dim": dim, "direction": direction, "kind": "in-place"},
                        invariants=(
                            custom(
                                "in-place-returns-none",
                                lambda r: None if r is None else f"expected None, got {r!r}",
                            ),
                        ),
                    )

                if dim >= 2:  # noqa: PLR2004
                    perm = list(range(dim))
                    perm[0], perm[-1] = perm[-1], perm[0]
                    yield Case(
                        GROUP,
                        f"permute_twice_r{rank}d{dim}p{degree}_{dtype}",
                        Bezier.permute_directions,
                        lambda bezier=bezier, perm=perm: bezier.permute_directions(
                            perm
                        ).permute_directions(perm),
                        {"rank": rank, "dim": dim, "degree": degree, "dtype": str(dtype)},
                        invariants=(_double_op_identity_invariant(cp),),
                        arrays={"control_points": cp},
                    )
                    fresh_perm = Bezier(cp.copy())
                    yield Case(
                        GROUP,
                        f"permute_in_place_returns_none_r{rank}d{dim}_{dtype}",
                        Bezier.permute_directions,
                        lambda fresh_perm=fresh_perm, perm=perm: fresh_perm.permute_directions(
                            perm, in_place=True
                        ),
                        {"rank": rank, "dim": dim, "kind": "in-place"},
                        invariants=(
                            custom(
                                "in-place-returns-none",
                                lambda r: None if r is None else f"expected None, got {r!r}",
                            ),
                        ),
                    )

                identity = AffineTransform.identity(rank)
                yield Case(
                    GROUP,
                    f"transform_identity_r{rank}d{dim}p{degree}_{dtype}",
                    Bezier.transform,
                    lambda bezier=bezier, identity=identity: bezier.transform(identity),
                    {"rank": rank, "dim": dim, "degree": degree, "dtype": str(dtype)},
                    invariants=(
                        custom(
                            "identity-transform-is-noop",
                            lambda r, cp=cp: None
                            if np.array_equal(r.control_points, cp)
                            else "identity transform changed the control points",
                        ),
                    ),
                    arrays={"control_points": cp},
                )
                fresh_transform = Bezier(cp.copy())
                yield Case(
                    GROUP,
                    f"transform_in_place_returns_none_r{rank}d{dim}_{dtype}",
                    Bezier.transform,
                    lambda fresh_transform=fresh_transform, identity=identity: (
                        fresh_transform.transform(identity, in_place=True)
                    ),
                    {"rank": rank, "dim": dim, "kind": "in-place"},
                    invariants=(
                        custom(
                            "in-place-returns-none",
                            lambda r: None if r is None else f"expected None, got {r!r}",
                        ),
                    ),
                )

                if profile is Profile.FULL:
                    singular = AffineTransform(np.zeros((rank, rank)), np.zeros(rank))
                    yield Case(
                        GROUP,
                        f"transform_singular_matrix_r{rank}d{dim}_{dtype}",
                        Bezier.transform,
                        lambda bezier=bezier, singular=singular: bezier.transform(singular),
                        {"rank": rank, "dim": dim, "kind": "singular-matrix"},
                        arrays={"control_points": cp},
                    )

    cp_1d = _random_control_points(1, 1, 2, np.dtype(np.float64), offset=14900)
    bezier_1d = Bezier(cp_1d)
    yield Case(
        GROUP,
        "reverse_direction_out_of_range",
        Bezier.reverse,
        lambda: bezier_1d.reverse(3),
        {"kind": "direction-out-of-range"},
    )
    cp_2d = _random_control_points(1, 2, 2, np.dtype(np.float64), offset=14901)
    bezier_2d = Bezier(cp_2d)
    yield Case(
        GROUP,
        "permute_invalid_permutation",
        Bezier.permute_directions,
        lambda: bezier_2d.permute_directions([0, 0]),
        {"kind": "invalid-permutation"},
    )
    yield Case(
        GROUP,
        "transform_dimension_mismatch",
        Bezier.transform,
        lambda: bezier_2d.transform(AffineTransform.identity(3)),
        {"kind": "dimension-mismatch"},
    )


# ---------------------------------------------------------------------------
# Slice, boundary, collapse
# ---------------------------------------------------------------------------


def _slice_boundary_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`~pantr.bezier.Bezier.slice` and ``boundary`` cases.

    Invariant 10: ``boundary(axis, side)`` agrees with ``slice(axis, 0.0 or 1.0)``.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile slice/boundary.
    """
    for dtype in dtypes(profile):
        for rank, dim in ((1, 2), (2, 2), (1, 3)) if profile is Profile.FULL else ((1, 2),):
            for degree in (2, 15) if profile is Profile.FULL else (2,):
                cp = _random_control_points(rank, dim, degree, dtype, offset=15000 + degree)
                bezier = Bezier(cp)
                slice_values = (0.0, 0.5, 1.0) if profile is Profile.FULL else (0.5,)
                for value in slice_values:
                    yield Case(
                        GROUP,
                        f"slice_r{rank}d{dim}p{degree}_{dtype}_v{value:g}",
                        Bezier.slice,
                        lambda bezier=bezier, value=value: bezier.slice(0, value),
                        {
                            "rank": rank,
                            "dim": dim,
                            "degree": degree,
                            "dtype": str(dtype),
                            "value": value,
                        },
                        arrays={"control_points": cp},
                    )
                for side in (0, 1):
                    yield Case(
                        GROUP,
                        f"boundary_matches_slice_r{rank}d{dim}p{degree}_{dtype}_s{side}",
                        Bezier.boundary,
                        lambda bezier=bezier, side=side: bezier.boundary(0, side),
                        {
                            "rank": rank,
                            "dim": dim,
                            "degree": degree,
                            "dtype": str(dtype),
                            "side": side,
                        },
                        invariants=(
                            custom(
                                "boundary-matches-slice",
                                lambda r, bezier=bezier, side=side: _boundary_matches_slice(
                                    r, bezier, side
                                ),
                            ),
                        ),
                        arrays={"control_points": cp},
                    )

    cp_1d = _random_control_points(1, 1, 2, np.dtype(np.float64), offset=15900)
    bezier_1d = Bezier(cp_1d)
    yield Case(
        GROUP,
        "slice_value_out_of_range",
        Bezier.slice,
        lambda: bezier_1d.slice(0, 1.5),
        {"kind": "value-out-of-range"},
    )
    yield Case(
        GROUP,
        "slice_axis_out_of_range",
        Bezier.slice,
        lambda: bezier_1d.slice(2, 0.5),
        {"kind": "axis-out-of-range"},
    )
    yield Case(
        GROUP,
        "boundary_invalid_side",
        Bezier.boundary,
        lambda: bezier_1d.boundary(0, 2),
        {"kind": "invalid-side"},
    )


def _boundary_matches_slice(result: object, bezier: Bezier, side: int) -> str | None:
    """Check that ``boundary`` and the corresponding ``slice`` agree.

    Args:
        result (object): The value returned by ``boundary(axis, side)``.
        bezier (Bezier): The Bezier the boundary was extracted from.
        side (int): ``0`` or ``1``, the domain end that was extracted.

    Returns:
        str | None: ``None`` when they agree exactly, else a message.
    """
    reference = bezier.slice(0, float(side))
    if isinstance(result, Bezier) and isinstance(reference, Bezier):
        if np.array_equal(result.control_points, reference.control_points):
            return None
        return "boundary(axis, side) differs from slice(axis, 0.0 or 1.0)"
    if isinstance(result, np.ndarray) and isinstance(reference, np.ndarray):
        if np.array_equal(result, reference):
            return None
        return "boundary(axis, side) differs from slice(axis, 0.0 or 1.0)"
    return f"boundary/slice returned mismatched types: {type(result)} vs {type(reference)}"


def _collapse_along_axis_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`~pantr.bezier.Bezier.collapse_along_axis` cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile collapse.
    """
    for dtype in dtypes(profile):
        for rank, dim in ((1, 2), (2, 3)) if profile is Profile.FULL else ((1, 2),):
            for degree in (2, 15) if profile is Profile.FULL else (2,):
                cp = _random_control_points(rank, dim, degree, dtype, offset=16000 + degree)
                bezier = Bezier(cp)
                value_families = (
                    ([0.0] * (dim - 1), [1.0] * (dim - 1), [0.5] * (dim - 1))
                    if profile is Profile.FULL
                    else ([0.5] * (dim - 1),)
                )
                for values in value_families:
                    tag = "_".join(f"{v:g}" for v in values)
                    yield Case(
                        GROUP,
                        f"collapse_r{rank}d{dim}p{degree}_{dtype}_v{tag}",
                        Bezier.collapse_along_axis,
                        lambda bezier=bezier, values=values: bezier.collapse_along_axis(0, values),
                        {
                            "rank": rank,
                            "dim": dim,
                            "degree": degree,
                            "dtype": str(dtype),
                            "values": values,
                        },
                        invariants=(
                            custom(
                                "collapse-degree-preserved",
                                lambda r, degree=degree: None
                                if r.degree == (degree,)
                                else f"got degree {r.degree}, expected {(degree,)}",
                            ),
                        ),
                        arrays={"control_points": cp},
                    )

    cp_1d = _random_control_points(1, 1, 2, np.dtype(np.float64), offset=16900)
    bezier_1d = Bezier(cp_1d)
    yield Case(
        GROUP,
        "collapse_requires_dim_2",
        Bezier.collapse_along_axis,
        lambda: bezier_1d.collapse_along_axis(0, []),
        {"kind": "dim-less-than-2"},
    )
    cp_2d = _random_control_points(1, 2, 2, np.dtype(np.float64), offset=16901)
    bezier_2d = Bezier(cp_2d)
    yield Case(
        GROUP,
        "collapse_values_length_mismatch",
        Bezier.collapse_along_axis,
        lambda: bezier_2d.collapse_along_axis(0, [0.2, 0.3]),
        {"kind": "values-length-mismatch"},
    )
    yield Case(
        GROUP,
        "collapse_value_out_of_range",
        Bezier.collapse_along_axis,
        lambda: bezier_2d.collapse_along_axis(0, [1.5]),
        {"kind": "value-out-of-range"},
    )


# ---------------------------------------------------------------------------
# B-spline round trip
# ---------------------------------------------------------------------------


def _to_bspline_cases(profile: Profile) -> Iterator[Case]:
    """Yield :meth:`~pantr.bezier.Bezier.to_bspline` / ``create_from_bspline`` round trips.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile round trip.
    """
    combos = (
        _rank_dim_combos(profile)[:3] if profile is Profile.FULL else _rank_dim_combos(profile)[:1]
    )
    for dtype in dtypes(profile):
        for rank, dim in combos:
            for degree in (0, 1, 3) if profile is Profile.FULL else (1,):
                cp = _random_control_points(rank, dim, degree, dtype, offset=17000 + degree)
                bezier = Bezier(cp)
                yield Case(
                    GROUP,
                    f"to_bspline_roundtrip_r{rank}d{dim}p{degree}_{dtype}",
                    Bezier.to_bspline,
                    lambda bezier=bezier: create_from_bspline(bezier.to_bspline()),
                    {"rank": rank, "dim": dim, "degree": degree, "dtype": str(dtype)},
                    invariants=(
                        custom(
                            "roundtrip-bit-exact",
                            lambda r, cp=cp: None
                            if np.array_equal(r.control_points, cp)
                            else "to_bspline/create_from_bspline round trip is not bit-identical",
                        ),
                    ),
                    arrays={"control_points": cp},
                )

    non_bezier_space = BsplineSpace(
        [BsplineSpace1D(np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float64), 2)]
    )
    non_bezier_bspline = Bspline(
        non_bezier_space, np.linspace(0.0, 1.0, 4, dtype=np.float64).reshape(-1, 1)
    )
    yield Case(
        GROUP,
        "create_from_bspline_not_bezier_like",
        create_from_bspline,
        lambda: create_from_bspline(non_bezier_bspline),
        {"kind": "not-bezier-like-knots"},
    )


# ---------------------------------------------------------------------------
# Interpolation and fitting
# ---------------------------------------------------------------------------


def _interpolate_fit_cases(profile: Profile) -> Iterator[Case]:
    """Yield :func:`~pantr.bezier.interpolate_bezier` and :func:`~pantr.bezier.fit_bezier` cases.

    The oracle is a known low-degree polynomial (``t**k``), evaluated directly
    by NumPy at the interpolation nodes -- an independent construction, not a
    mirror of the interpolation algorithm. Exact reproduction is expected
    whenever ``degree >= k``.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile interpolation or fit.
    """
    for dtype_name in ("float64",) if profile is not Profile.FULL else ("float64", "float32"):
        for n_pts in (1, 2, 5, 16) if profile is Profile.FULL else (5,):
            k = min(2, n_pts - 1)

            def func(lattice: PointsLattice, k: int = k) -> npt.NDArray[np.floating[Any]]:
                pts = lattice.get_all_points()[:, 0]
                return np.asarray(pts, dtype=np.float64) ** k

            yield Case(
                GROUP,
                f"interpolate_exact_monomial_n{n_pts}_k{k}_{dtype_name}",
                interpolate_bezier,
                lambda n_pts=n_pts: interpolate_bezier(func, n_pts),
                {"n_pts": n_pts, "k": k, "dtype": dtype_name},
                invariants=(
                    custom(
                        "interpolate-reproduces-monomial",
                        lambda r, k=k: _monomial_reproduction_failure(r, k),
                    ),
                ),
            )

            if profile is Profile.FULL and n_pts >= 3:  # noqa: PLR2004
                low_degree = max(n_pts - 2, 0)
                yield Case(
                    GROUP,
                    f"interpolate_least_squares_n{n_pts}_deg{low_degree}",
                    interpolate_bezier,
                    lambda n_pts=n_pts, low_degree=low_degree: interpolate_bezier(
                        func, n_pts, degree=low_degree
                    ),
                    {"n_pts": n_pts, "degree": low_degree, "kind": "least-squares"},
                )

    monomial_invariant_k2 = custom(
        "fit-reproduces-monomial", lambda r: _monomial_reproduction_failure(r, 2)
    )
    for nodes_kind in ("chebyshev", "uniform") if profile is Profile.FULL else ("chebyshev",):
        n_pts = 6
        node_arr = np.linspace(0.0, 1.0, n_pts, dtype=np.float64)
        values = node_arr**2
        yield Case(
            GROUP,
            f"fit_bezier_tensor_product_{nodes_kind}",
            fit_bezier,
            lambda values=values, node_arr=node_arr: fit_bezier(values, node_arr),
            {"n_pts": n_pts, "nodes": nodes_kind},
            invariants=(monomial_invariant_k2,),
        )

    generator = rng(18000)
    scattered_pts = generator.uniform(0.0, 1.0, (12, 1))
    scattered_values = scattered_pts[:, 0] ** 2
    yield Case(
        GROUP,
        "fit_bezier_scattered",
        fit_bezier,
        lambda: fit_bezier(scattered_values, scattered_pts, degree=3),
        {"n_pts": 12, "degree": 3, "kind": "scattered"},
        invariants=(monomial_invariant_k2,),
        arrays={"pts": scattered_pts, "values": scattered_values},
    )

    yield Case(
        GROUP,
        "fit_bezier_scattered_missing_degree",
        fit_bezier,
        lambda: fit_bezier(scattered_values, scattered_pts),
        {"kind": "scattered-missing-degree"},
    )
    yield Case(
        GROUP,
        "interpolate_degree_exceeds_npts",
        interpolate_bezier,
        lambda: interpolate_bezier(func, 4, degree=5),
        {"kind": "degree-exceeds-n_pts"},
    )
    yield Case(
        GROUP,
        "interpolate_bad_return_shape",
        interpolate_bezier,
        lambda: interpolate_bezier(lambda lattice: np.zeros(3), 5),
        {"kind": "bad-return-shape"},
    )


def _monomial_reproduction_failure(result: object, k: int) -> str | None:
    """Check that an interpolated/fitted Bezier reproduces ``t**k`` exactly.

    Args:
        result (object): The Bezier returned by ``interpolate_bezier``/``fit_bezier``.
        k (int): The monomial power the sample values were drawn from.

    Returns:
        str | None: ``None`` when the reproduction is within the SVD/rounding
        floor, else a message.
    """
    if not isinstance(result, Bezier):
        return f"expected Bezier, got {type(result).__name__}"
    if result.degree[0] < k:
        return None  # least-squares path: exact reproduction is not expected
    ts = np.array([0.1, 0.4, 0.6, 0.9], dtype=result.dtype)
    expected = ts.astype(np.float64) ** k
    got = np.asarray(result.evaluate(ts), dtype=np.float64)
    eps = get_machine_epsilon(result.dtype)
    # Truncated-SVD recovery accumulates O(degree) rounding on top of the
    # conditioning of the Bernstein Vandermonde system; a generous factor of
    # 1e3 covers both without masking a genuine failure to reproduce.
    tol = 1.0e3 * eps * max(float(np.max(np.abs(expected))), 1.0)
    worst = float(np.max(np.abs(got - expected)))
    if worst > tol:
        return f"max|delta| = {worst:.3e} > {tol:.3e}"
    return None


# ---------------------------------------------------------------------------
# Direct kernel/helper probes
# ---------------------------------------------------------------------------


def _de_casteljau_kernel_cases(profile: Profile) -> Iterator[Case]:
    """Yield direct probes of ``_de_casteljau_eval_scalar``.

    This Numba kernel performs no input validation whatsoever, per its own
    docstring. A length-0 ``coeff`` is the sharpest boundscheck target: the
    algorithm reads ``coeff[0]`` unconditionally.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile kernel call.
    """
    for dtype in dtypes(profile):
        for length, name in ((0, "len0"), (1, "len1"), (63, "len63")):
            coeff = np.linspace(-1.0, 1.0, length, dtype=dtype)
            yield Case(
                GROUP,
                f"de_casteljau_{name}_{dtype}",
                _de_casteljau_eval_scalar,
                lambda coeff=coeff: _de_casteljau_eval_scalar(coeff, 0.5),
                {"length": length, "dtype": str(dtype), "t": 0.5},
                arrays={"coeff": coeff},
            )

        coeff5 = np.linspace(-1.0, 1.0, 5, dtype=dtype)
        for t, finite in ((-5.0, True), (5.0, True), (float("nan"), False), (float("inf"), False)):
            yield Case(
                GROUP,
                f"de_casteljau_t{t}_{dtype}",
                _de_casteljau_eval_scalar,
                lambda coeff5=coeff5, t=t: _de_casteljau_eval_scalar(coeff5, t),
                {"t": t, "dtype": str(dtype)},
                finite_inputs=finite,
                arrays={"coeff": coeff5},
            )


def _bernstein_interpolate_kernel_cases(profile: Profile) -> Iterator[Case]:
    """Yield direct probes of ``_bernstein_interpolate``.

    This Layer 2 helper performs no input validation either; it is exercised
    here with the shapes and value families the module docstring flags.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile kernel call.
    """
    for dtype in dtypes(profile):
        shapes = (
            (((1,), "shape1"), ((2,), "shape2"), ((63,), "shape63"))
            if profile is Profile.FULL
            else (((1,), "shape1"), ((2,), "shape2"))
        )
        for shape, name in shapes:
            f = np.linspace(0.1, 1.0, shape[0], dtype=dtype)
            yield Case(
                GROUP,
                f"bernstein_interpolate_{name}_{dtype}",
                _bernstein_interpolate,
                lambda f=f: _bernstein_interpolate(f),
                {"shape": shape, "dtype": str(dtype)},
                arrays={"f": f},
            )

        zeros = np.zeros(8, dtype=dtype)
        yield Case(
            GROUP,
            f"bernstein_interpolate_zeros_{dtype}",
            _bernstein_interpolate,
            lambda zeros=zeros: _bernstein_interpolate(zeros),
            {"kind": "all-zeros", "dtype": str(dtype)},
        )

        if profile is Profile.FULL:
            outlier = np.full(8, 1e-3, dtype=dtype)
            outlier[3] = 1e9
            yield Case(
                GROUP,
                f"bernstein_interpolate_outlier_{dtype}",
                _bernstein_interpolate,
                lambda outlier=outlier: _bernstein_interpolate(outlier),
                {"kind": "huge-outlier-among-tiny", "dtype": str(dtype)},
                arrays={"f": outlier},
            )

        nan_f = np.linspace(0.0, 1.0, 8, dtype=dtype)
        nan_f[4] = np.nan
        yield Case(
            GROUP,
            f"bernstein_interpolate_nan_{dtype}",
            _bernstein_interpolate,
            lambda nan_f=nan_f: _bernstein_interpolate(nan_f),
            {"kind": "nan", "dtype": str(dtype)},
            finite_inputs=False,
            arrays={"f": nan_f},
        )

        f_2d = rng(19000).uniform(0.0, 1.0, (5, 4)).astype(dtype)
        yield Case(
            GROUP,
            f"bernstein_interpolate_2d_{dtype}",
            _bernstein_interpolate,
            lambda f_2d=f_2d: _bernstein_interpolate(f_2d),
            {"shape": (5, 4), "dtype": str(dtype)},
            arrays={"f": f_2d},
        )

        if profile is Profile.FULL:
            f_3d = rng(19001).uniform(0.0, 1.0, (3, 3, 3)).astype(dtype)
            yield Case(
                GROUP,
                f"bernstein_interpolate_3d_{dtype}",
                _bernstein_interpolate,
                lambda f_3d=f_3d: _bernstein_interpolate(f_3d),
                {"shape": (3, 3, 3), "dtype": str(dtype)},
                arrays={"f": f_3d},
            )


# ---------------------------------------------------------------------------
# Coefficient-family sanity crossing (partition-independent construction/eval)
# ---------------------------------------------------------------------------


def _coeff_family_cases(profile: Profile) -> Iterator[Case]:
    """Yield evaluate cases crossed with :func:`~._axes.coeff_specs` control-point families.

    Covers random, all-identical (degenerate/collapsed geometry), collinear,
    all-zero, and at-the-zero-threshold control points -- families the rest of
    this module does not otherwise exercise, since they are built from
    ``coeff_specs`` rather than :func:`_random_control_points`.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile evaluate call.
    """
    for dtype in dtypes(profile):
        for degree in (1, 3, 15) if profile is Profile.FULL else (1,):
            shape = (degree + 1, 2)
            for spec in coeff_specs(shape, dtype, profile, offset=20000 + degree):
                bezier = Bezier(spec.values)
                pts = np.array([0.0, 0.3, 0.7, 1.0], dtype=dtype)
                yield Case(
                    GROUP,
                    f"coeff_family_{spec.name}_p{degree}_{dtype}",
                    Bezier.evaluate,
                    lambda bezier=bezier, pts=pts: bezier.evaluate(pts),
                    {"degree": degree, "dtype": str(dtype), "family": spec.name},
                    arrays={"control_points": spec.values},
                )


# ---------------------------------------------------------------------------
# Rational-specific edge cases
# ---------------------------------------------------------------------------


def _rational_weight_cases(profile: Profile) -> Iterator[Case]:
    """Yield evaluate cases probing degenerate rational weights.

    A zero, negative, or near-threshold weight is neither validated by the
    constructor nor documented as rejected by ``evaluate`` -- so whatever
    happens (a rejection, a finite answer, or an ``inf``/``NaN``) is worth
    recording rather than assuming.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile rational evaluation.
    """
    for dtype in dtypes(profile):
        degree = 3
        base = rng(21000).uniform(-1.0, 1.0, (degree + 1, 2)).astype(dtype)
        pts = np.array([0.2, 0.5, 0.8], dtype=dtype)

        for weight_name, weight_value in (
            ("zero", 0.0),
            ("negative", -1.0),
            ("near_threshold", get_default(dtype)),
        ):
            cp = np.concatenate([base, np.ones((degree + 1, 1), dtype=dtype)], axis=-1)
            cp[1, -1] = dtype.type(weight_value)
            bezier = Bezier(cp, is_rational=True)
            yield Case(
                GROUP,
                f"rational_weight_{weight_name}_{dtype}",
                Bezier.evaluate,
                lambda bezier=bezier, pts=pts: bezier.evaluate(pts),
                {"weight": weight_name, "dtype": str(dtype)},
                finite_inputs=False,
                arrays={"control_points": cp},
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def cases(profile: Profile) -> Iterator[Case]:
    """Yield every case in this group.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: The group's cases.
    """
    yield from _construct_cases(profile)
    yield from _evaluate_cases(profile)
    yield from _evaluate_derivatives_cases(profile)
    yield from _elevate_degree_cases(profile)
    yield from _reduce_degree_cases(profile)
    yield from _degree_reduction_error_cases(profile)
    yield from _minimize_degree_cases(profile)
    yield from _split_cases(profile)
    yield from _restrict_cases(profile)
    yield from _compose_cases(profile)
    yield from _multiply_cases(profile)
    yield from _derivative_cases(profile)
    yield from _root_finding_cases(profile)
    yield from _reverse_permute_transform_cases(profile)
    yield from _slice_boundary_cases(profile)
    yield from _collapse_along_axis_cases(profile)
    yield from _to_bspline_cases(profile)
    yield from _interpolate_fit_cases(profile)
    yield from _de_casteljau_kernel_cases(profile)
    yield from _bernstein_interpolate_kernel_cases(profile)
    yield from _coeff_family_cases(profile)
    yield from _rational_weight_cases(profile)
