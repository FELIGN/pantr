"""The parameter axes the sweep crosses, and the hostile input families per axis.

Everything the probe modules cross lives here, so the swept space is stated in one
place and the coverage claim in the report can be read off it. Every family leans on
the corners rather than the middle: degree 0 and a high degree where binomials bite,
knot multiplicities up to ``degree + 1``, non-clamped and periodic knot vectors,
translated domains, empty and single-element inputs, and ``NaN``/``inf``.

Two profiles are provided. ``FULL`` is the sweep proper; ``SMOKE`` is the bounded
subset the CI test runs, small enough that recompiling every kernel into a fresh
Numba cache stays affordable.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Final, NamedTuple

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


class Profile(enum.IntEnum):
    """How wide to open each axis."""

    SMOKE = 0
    """Bounded subset for CI: the corners only, one domain, one dtype per axis."""

    FULL = 1
    """The whole space."""


SEED: Final = 20260814
"""Fixed seed so every random family is reproducible across runs and machines."""

_DEGREES_SMOKE: Final = (0, 1, 3)
_DEGREES_FULL: Final = (0, 1, 2, 3, 15, 62)
"""Degree axis. 0-3 are the corners; 15 is high enough for binomial and Vandermonde
conditioning to bite; 62 is the exact cliff where ``_bincoeff``'s integer recurrence
wraps int64 (``_bspline_degree_core.py:22-67``), so any operation whose combined degree
reaches it is a candidate for silent corruption."""

_DOMAIN_UNIT: Final = (0.0, 1.0)
_DOMAINS_FULL: Final = (
    _DOMAIN_UNIT,
    (0.0, 1e-6),
    (0.0, 5.0),
    (0.0, 100.0),
    (0.0, 1e6),
    (1e6, 1e6 + 1.0),
)
"""Scale and position axis, doubling as the knot-magnitude axis: an interior knot lands
at ``lo + 0.5 * span``, so these domains put one at 0.5, 5e-7, 2.5, 50 and 5e5, plus the
translated case. Magnitude matters because ``BsplineSpace1D``'s knot-snapping tolerance is
*absolute* (``get_strict(float64) = 1e-15``) while the gap it grades scales with the knot,
so from |knot| ~ 5 upward it merges nothing. The translated domain is the one that bites
elsewhere: a tolerance derived from a bounding-box diagonal is translation invariant, an
arithmetic floor is not."""


class KnotSpec(NamedTuple):
    """One knot-vector family instance.

    Attributes:
        name (str): Family identifier, used in the case label.
        knots (npt.NDArray[np.float32 | np.float64]): The knot vector.
        periodic (bool): Value to pass as ``BsplineSpace1D(periodic=...)``.
    """

    name: str
    knots: npt.NDArray[np.float32 | np.float64]
    periodic: bool


class PointSpec(NamedTuple):
    """One evaluation-point family instance.

    Attributes:
        name (str): Family identifier, used in the case label.
        pts (npt.NDArray[np.float32 | np.float64]): The points, shape ``(n,)``.
        finite (bool): ``False`` when the family deliberately contains ``NaN``/``inf``,
            which switches off the automatic finiteness check on the result.
    """

    name: str
    pts: npt.NDArray[np.float32 | np.float64]
    finite: bool


class CoeffSpec(NamedTuple):
    """One coefficient / control-point family instance.

    Attributes:
        name (str): Family identifier, used in the case label.
        values (npt.NDArray[np.float32 | np.float64]): The coefficient array.
    """

    name: str
    values: npt.NDArray[np.float32 | np.float64]


def degrees(profile: Profile) -> tuple[int, ...]:
    """List the polynomial degrees to sweep.

    Degree 0 is included deliberately: it has been a defect source in this codebase.
    The high degrees are where factorials, binomials and the Bernstein recurrence
    lose precision or overflow.

    Args:
        profile (Profile): Sweep width.

    Returns:
        tuple[int, ...]: Degrees, ascending.
    """
    return _DEGREES_SMOKE if profile is Profile.SMOKE else _DEGREES_FULL


def dtypes(profile: Profile) -> tuple[np.dtype[np.float32 | np.float64], ...]:
    """List the working precisions to sweep.

    Args:
        profile (Profile): Sweep width.

    Returns:
        tuple[np.dtype[np.float32 | np.float64], ...]: ``float64`` alone for the smoke
            profile, both precisions for the full sweep.
    """
    if profile is Profile.SMOKE:
        return (np.dtype(np.float64),)
    return (np.dtype(np.float64), np.dtype(np.float32))


def domains(profile: Profile) -> tuple[tuple[float, float], ...]:
    """List the parametric/physical domains to sweep.

    Args:
        profile (Profile): Sweep width.

    Returns:
        tuple[tuple[float, float], ...]: ``(lo, hi)`` pairs covering unit, tiny, huge
            and translated domains.
    """
    return (_DOMAIN_UNIT,) if profile is Profile.SMOKE else _DOMAINS_FULL


def dims(profile: Profile, *, max_dim: int = 3) -> tuple[int, ...]:
    """List the parametric dimensions to sweep.

    Args:
        profile (Profile): Sweep width.
        max_dim (int): Highest dimension the entry point admits. Pass 4 to probe
            ``n ** dim`` overflow where the API allows it. Defaults to 3.

    Returns:
        tuple[int, ...]: Dimensions, ascending.
    """
    top = min(max_dim, 2 if profile is Profile.SMOKE else max_dim)
    return tuple(range(1, top + 1))


def rng(offset: int = 0) -> np.random.Generator:
    """Build a reproducible random generator.

    Args:
        offset (int): Stream offset so distinct families draw distinct values while
            staying reproducible. Defaults to 0.

    Returns:
        np.random.Generator: Seeded generator.
    """
    return np.random.default_rng(SEED + offset)


# ---------------------------------------------------------------------------
# Knot vector families
# ---------------------------------------------------------------------------


def open_uniform_knots(
    degree: int,
    n_intervals: int,
    domain: tuple[float, float],
    dtype: np.dtype[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Build a clamped uniform knot vector.

    Args:
        degree (int): Polynomial degree.
        n_intervals (int): Number of non-empty knot spans.
        domain (tuple[float, float]): ``(lo, hi)`` parametric interval.
        dtype (np.dtype[np.float32 | np.float64]): Knot precision.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Knot vector of length
            ``n_intervals + 2 * degree + 1``.
    """
    lo, hi = domain
    breaks = np.linspace(lo, hi, n_intervals + 1, dtype=dtype)
    return np.concatenate(
        [np.full(degree, lo, dtype=dtype), breaks, np.full(degree, hi, dtype=dtype)]
    )


_LADDER_MAX_DEGREE: Final = 4
"""Highest degree whose multiplicity ladder is swept exhaustively rather than by corners."""


def interior_multiplicities(degree: int, profile: Profile) -> tuple[int, ...]:
    """List the interior knot multiplicities to sweep for one degree.

    Multiplicity ``degree + 1`` is the prime suspect on every operation that takes a
    spline: it makes the spline discontinuous (C^-1), which
    :meth:`pantr.bspline.BsplineSpace1D.subdivide` produces as a documented public
    feature with ``regularity=-1``, and which several algorithms silently assume away.
    The whole ladder is swept up to degree ``_LADDER_MAX_DEGREE``; above it only the
    corners are, since the interior of the ladder repeats one continuity class per step
    and the case count is multiplied by every other axis.

    Args:
        degree (int): Polynomial degree.
        profile (Profile): Sweep width.

    Returns:
        tuple[int, ...]: Multiplicities, ascending, each in ``[1, degree + 1]``.
    """
    if profile is not Profile.FULL:
        return tuple(sorted({1, min(2, degree + 1)}))
    if degree <= _LADDER_MAX_DEGREE:
        return tuple(range(1, degree + 2))
    return tuple(sorted({1, 2, degree - 1, degree, degree + 1}))


def knot_specs(
    degree: int,
    domain: tuple[float, float],
    dtype: np.dtype[np.float32 | np.float64],
    profile: Profile,
) -> tuple[KnotSpec, ...]:
    """Build the hostile knot-vector families for one ``(degree, domain, dtype)``.

    Covers, in order: clamped uniform; clamped non-uniform (graded spans); the interior
    multiplicities of :func:`interior_multiplicities`; a periodic/unclamped vector; the
    minimal legal clamped vector (one Bezier span); the minimal legal unclamped one; a
    non-clamped uniform vector; each end clamped alone; an over-clamped end; a knot run
    a hair inside the left boundary; and three malformed vectors (unsorted, ``NaN``,
    all-equal) which the sweep expects to be *rejected*, not accepted.

    Args:
        degree (int): Polynomial degree.
        domain (tuple[float, float]): ``(lo, hi)`` parametric interval.
        dtype (np.dtype[np.float32 | np.float64]): Knot precision.
        profile (Profile): Sweep width; ``SMOKE`` drops the multiplicity ladder above
            2 and the malformed vectors.

    Returns:
        tuple[KnotSpec, ...]: The families.
    """
    lo, hi = domain
    span = hi - lo
    full = profile is Profile.FULL
    specs: list[KnotSpec] = [
        KnotSpec("open_uniform", open_uniform_knots(degree, 4, domain, dtype), False),
    ]

    # Clamped, non-uniform: geometrically graded interior spans.
    graded = lo + span * np.array([0.1, 0.3, 0.7], dtype=dtype)
    specs.append(
        KnotSpec(
            "open_graded",
            np.concatenate(
                [
                    np.full(degree + 1, lo, dtype=dtype),
                    graded,
                    np.full(degree + 1, hi, dtype=dtype),
                ]
            ),
            False,
        )
    )

    for mult in interior_multiplicities(degree, profile):
        interior = np.concatenate(
            [
                np.full(mult, lo + 0.5 * span, dtype=dtype),
                np.array([lo + 0.75 * span], dtype=dtype),
            ]
        )
        specs.append(
            KnotSpec(
                f"interior_mult{mult}",
                np.concatenate(
                    [
                        np.full(degree + 1, lo, dtype=dtype),
                        interior,
                        np.full(degree + 1, hi, dtype=dtype),
                    ]
                ),
                False,
            )
        )

    # Periodic / unclamped: uniform knots covering degree extra spans on each side.
    n_per = max(2, degree + 1)
    step = span / n_per
    periodic = np.arange(-degree, n_per + degree + 1, dtype=np.float64) * step + lo
    specs.append(KnotSpec("periodic_uniform", periodic.astype(dtype), True))
    specs.append(KnotSpec("unclamped_uniform", periodic.astype(dtype), False))

    # Minimal legal lengths: one Bezier span clamped, one span unclamped.
    specs.append(
        KnotSpec(
            "minimal_clamped",
            np.concatenate(
                [np.full(degree + 1, lo, dtype=dtype), np.full(degree + 1, hi, dtype=dtype)]
            ),
            False,
        )
    )
    minimal_unclamped = lo + span * np.linspace(-degree, degree + 1, 2 * degree + 2, dtype=dtype)
    specs.append(KnotSpec("minimal_unclamped", minimal_unclamped, False))

    # One end clamped only.
    tail = lo + span * np.arange(1, degree + 2, dtype=dtype) / (degree + 1)
    specs.append(
        KnotSpec(
            "left_clamped_only",
            np.concatenate([np.full(degree + 1, lo, dtype=dtype), tail]),
            False,
        )
    )
    head = lo - span * np.arange(degree + 1, 0, -1, dtype=dtype) / (degree + 1)
    specs.append(
        KnotSpec(
            "right_clamped_only",
            np.concatenate([head, np.full(degree + 1, hi, dtype=dtype)]),
            False,
        )
    )

    if not full:
        return tuple(specs)

    # Over-clamped left end: multiplicity degree + 2 at the domain start.
    specs.append(
        KnotSpec(
            "over_clamped_left",
            np.concatenate(
                [
                    np.full(degree + 2, lo, dtype=dtype),
                    np.array([lo + 0.5 * span], dtype=dtype),
                    np.full(degree + 1, hi, dtype=dtype),
                ]
            ),
            False,
        )
    )

    # A knot run a hair inside the left boundary: spans far below any snap tolerance.
    hair = lo + span * np.full(degree, 1e-13, dtype=dtype)
    specs.append(
        KnotSpec(
            "hair_run_at_left",
            np.concatenate(
                [
                    np.full(degree + 1, lo, dtype=dtype),
                    hair,
                    np.full(degree + 1, hi, dtype=dtype),
                ]
            ),
            False,
        )
    )

    # Malformed vectors: these must be rejected, and a rejection is not a finding.
    # `open_uniform_knots(degree, 4, ...)` always has 2 * degree + 5 entries, so the
    # swap below is unconditional.
    unsorted = open_uniform_knots(degree, 4, domain, dtype).copy()
    mid = unsorted.size // 2
    unsorted[mid], unsorted[mid - 1] = unsorted[mid - 1], unsorted[mid] + span
    specs.append(KnotSpec("unsorted", unsorted, False))

    with_nan = open_uniform_knots(degree, 4, domain, dtype).copy()
    with_nan[with_nan.size // 2] = np.nan
    specs.append(KnotSpec("with_nan", with_nan, False))

    specs.append(KnotSpec("all_equal", np.full(2 * degree + 2, lo, dtype=dtype), False))
    specs.append(KnotSpec("too_short", np.full(degree + 1, lo, dtype=dtype), False))

    return tuple(specs)


# ---------------------------------------------------------------------------
# Evaluation point families
# ---------------------------------------------------------------------------


def point_specs(
    domain: tuple[float, float],
    dtype: np.dtype[np.float32 | np.float64],
    profile: Profile,
    *,
    breakpoints: npt.NDArray[np.float32 | np.float64] | None = None,
) -> tuple[PointSpec, ...]:
    """Build the hostile evaluation-point families for one domain.

    Args:
        domain (tuple[float, float]): ``(lo, hi)`` parametric interval.
        dtype (np.dtype[np.float32 | np.float64]): Point precision.
        profile (Profile): Sweep width.
        breakpoints (npt.NDArray[np.float32 | np.float64] | None): Interior knot
            values to hit exactly. When ``None`` the interior-knot family is skipped.

    Returns:
        tuple[PointSpec, ...]: The families.
    """
    lo, hi = domain
    span = hi - lo
    step = np.spacing(np.float64(max(abs(lo), abs(hi), 1.0))).astype(np.float64)

    specs: list[PointSpec] = [
        PointSpec("interior", np.array([lo + 0.31 * span, lo + 0.77 * span], dtype=dtype), True),
        PointSpec("right_endpoint", np.array([hi], dtype=dtype), True),
        PointSpec("left_endpoint", np.array([lo], dtype=dtype), True),
    ]
    if profile is Profile.FULL:
        specs += [
            PointSpec("just_outside_right", np.array([hi + 8.0 * step], dtype=dtype), True),
            PointSpec("just_outside_left", np.array([lo - 8.0 * step], dtype=dtype), True),
            PointSpec("far_outside", np.array([lo - span, hi + span], dtype=dtype), True),
            PointSpec("empty", np.zeros(0, dtype=dtype), True),
            PointSpec("single", np.array([lo + 0.5 * span], dtype=dtype), True),
            PointSpec("nan_inf", np.array([np.nan, np.inf, -np.inf], dtype=dtype), False),
        ]
    else:
        specs.append(PointSpec("empty", np.zeros(0, dtype=dtype), True))
    if breakpoints is not None and breakpoints.size:
        specs.append(PointSpec("at_knots", np.unique(breakpoints).astype(dtype), True))
    return tuple(specs)


# ---------------------------------------------------------------------------
# Coefficient / control point families
# ---------------------------------------------------------------------------


def coeff_specs(
    shape: tuple[int, ...],
    dtype: np.dtype[np.float32 | np.float64],
    profile: Profile,
    *,
    offset: int = 0,
) -> tuple[CoeffSpec, ...]:
    """Build the hostile coefficient families for a given array shape.

    Covers random, all-identical (collapsed geometry), collinear, identically zero,
    and a value sitting exactly on the library's own default zero threshold.

    Args:
        shape (tuple[int, ...]): Shape of the coefficient array.
        dtype (np.dtype[np.float32 | np.float64]): Coefficient precision.
        profile (Profile): Sweep width.
        offset (int): Random-stream offset. Defaults to 0.

    Returns:
        tuple[CoeffSpec, ...]: The families.
    """
    from pantr.tolerance import get_default  # noqa: PLC0415 -- keeps import cost off module load

    generator = rng(offset)
    specs: list[CoeffSpec] = [
        CoeffSpec("random", generator.uniform(-1.0, 1.0, shape).astype(dtype)),
        CoeffSpec("identical", np.full(shape, 0.5, dtype=dtype)),
    ]
    if profile is Profile.FULL:
        n = int(np.prod(shape)) if shape else 1
        ramp = np.linspace(0.0, 1.0, max(n, 1), dtype=dtype).reshape(shape) if shape else None
        if ramp is not None:
            specs.append(CoeffSpec("collinear", ramp))
        specs += [
            CoeffSpec("zeros", np.zeros(shape, dtype=dtype)),
            CoeffSpec("at_zero_threshold", np.full(shape, get_default(dtype), dtype=dtype)),
        ]
    return tuple(specs)
