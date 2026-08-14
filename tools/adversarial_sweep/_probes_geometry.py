"""Probes for :mod:`pantr.geometry` and :mod:`pantr.transform`.

Both are pure NumPy, so the yield here is contract violations and translation
sensitivity rather than Numba overruns: the bounding-box invariants that matter are
that a transformed box still encloses the images of every corner, and that
``union``/``intersect``/``pad`` keep their set relations at a domain translated to
``[1e6, 1e6 + 1]`` as well as on the unit box.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from pantr.geometry import AABB
from pantr.transform import AffineTransform

from ._axes import Profile, dims, domains, rng
from ._core import Case, custom

if TYPE_CHECKING:
    from collections.abc import Iterator

GROUP = "geometry"
"""Registry name of this probe group."""


def _corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Enumerate the corners of an axis-aligned box.

    Args:
        lo (np.ndarray): Lower corner, shape ``(ndim,)``.
        hi (np.ndarray): Upper corner, shape ``(ndim,)``.

    Returns:
        np.ndarray: Corners, shape ``(2 ** ndim, ndim)``.
    """
    return np.array(list(itertools.product(*zip(lo, hi, strict=True))), dtype=np.float64)


def _encloses_transformed_corners(
    lo: np.ndarray, hi: np.ndarray, affine: AffineTransform
) -> object:
    """Build a predicate asserting the transformed AABB encloses every mapped corner.

    Args:
        lo (np.ndarray): Lower corner of the source box.
        hi (np.ndarray): Upper corner of the source box.
        affine (AffineTransform): The map applied to the box.

    Returns:
        object: Predicate suitable for :func:`pantr_sweep._core.custom`.
    """
    mapped = affine(_corners(lo, hi))
    scale = float(max(np.max(np.abs(mapped)), 1.0))
    # The bound must hold up to the rounding of the matrix-vector product itself:
    # ndim fused multiply-adds on values of magnitude `scale` carry at most
    # ndim * eps * scale of error, and the enclosure is computed by the same product.
    slack = mapped.shape[1] * float(np.finfo(np.float64).eps) * scale * 4.0

    def predicate(result: object) -> str | None:
        box = result
        if not isinstance(box, AABB):
            return f"expected AABB, got {type(box).__name__}"
        below = float(np.max(box.lo - mapped.min(axis=0)))
        above = float(np.max(mapped.max(axis=0) - box.hi))
        worst = max(below, above)
        if worst > slack:
            return f"transformed box misses a corner by {worst:.3e} > {slack:.3e}"
        return None

    return predicate


def _aabb_cases(profile: Profile) -> Iterator[Case]:
    """Yield the :class:`pantr.geometry.AABB` cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile construction or operation.
    """
    generator = rng(11)
    for ndim in dims(profile, max_dim=4):
        for lo_val, hi_val in domains(profile):
            lo = np.full(ndim, lo_val)
            hi = np.full(ndim, hi_val)
            tag = f"d{ndim}_[{lo_val:g},{hi_val:g}]"

            yield Case(
                GROUP,
                f"aabb_construct_{tag}",
                AABB,
                lambda lo=lo, hi=hi: AABB(lo, hi),
                {"ndim": ndim, "lo": lo_val, "hi": hi_val},
                arrays={"lo": lo, "hi": hi},
            )
            yield Case(
                GROUP,
                f"aabb_degenerate_{tag}",
                AABB,
                lambda lo=lo: AABB(lo, lo),
                {"ndim": ndim, "lo": lo_val, "kind": "collapsed"},
            )
            yield Case(
                GROUP,
                f"aabb_inverted_{tag}",
                AABB,
                lambda lo=lo, hi=hi: AABB(hi, lo),
                {"ndim": ndim, "kind": "inverted"},
            )

            box = AABB(lo, hi)
            other = AABB(lo + 0.5 * (hi - lo), hi + 0.5 * (hi - lo))

            yield Case(
                GROUP,
                f"aabb_contains_center_{tag}",
                AABB.contains_point,
                lambda box=box, lo=lo, hi=hi: box.contains_point(0.5 * (lo + hi)),
                {"ndim": ndim, "lo": lo_val, "hi": hi_val},
                invariants=(
                    custom("center-inside", lambda r: None if r else "center reported out"),
                ),
            )
            yield Case(
                GROUP,
                f"aabb_union_encloses_{tag}",
                AABB.union,
                lambda box=box, other=other: box.union(other),
                {"ndim": ndim, "lo": lo_val, "hi": hi_val},
                invariants=(
                    custom(
                        "union-encloses",
                        lambda r, box=box, other=other: None
                        if (
                            np.all(r.lo <= box.lo)
                            and np.all(r.lo <= other.lo)
                            and np.all(r.hi >= box.hi)
                            and np.all(r.hi >= other.hi)
                        )
                        else f"union {r} does not enclose both inputs",
                    ),
                ),
            )
            yield Case(
                GROUP,
                f"aabb_intersect_contained_{tag}",
                AABB.intersect,
                lambda box=box, other=other: box.intersect(other),
                {"ndim": ndim, "lo": lo_val, "hi": hi_val},
                invariants=(
                    custom(
                        "intersect-contained",
                        lambda r, box=box, other=other: None
                        if r is None
                        or (
                            np.all(r.lo >= np.minimum(box.lo, other.lo))
                            and np.all(r.hi <= np.maximum(box.hi, other.hi))
                        )
                        else f"intersection {r} escapes both inputs",
                    ),
                ),
            )
            yield Case(
                GROUP,
                f"aabb_pad_negative_{tag}",
                AABB.pad,
                lambda box=box, lo=lo, hi=hi: box.pad(-0.75 * float(np.max(hi - lo))),
                {"ndim": ndim, "kind": "pad-collapses-box"},
            )

            if profile is Profile.FULL:
                nan_lo = lo.copy()
                nan_lo[0] = np.nan
                yield Case(
                    GROUP,
                    f"aabb_nan_lo_{tag}",
                    AABB,
                    lambda nan_lo=nan_lo, hi=hi: AABB(nan_lo, hi),
                    {"ndim": ndim, "kind": "nan"},
                    finite_inputs=False,
                )
                inf_hi = hi.copy()
                inf_hi[0] = np.inf
                yield Case(
                    GROUP,
                    f"aabb_inf_hi_{tag}",
                    AABB,
                    lambda lo=lo, inf_hi=inf_hi: AABB(lo, inf_hi),
                    {"ndim": ndim, "kind": "inf"},
                    finite_inputs=False,
                )
                yield Case(
                    GROUP,
                    f"aabb_shape_mismatch_{tag}",
                    AABB,
                    lambda lo=lo, ndim=ndim: AABB(lo, np.zeros(ndim + 1)),
                    {"ndim": ndim, "kind": "shape-mismatch"},
                )
                yield Case(
                    GROUP,
                    f"aabb_from_bounds_{tag}",
                    AABB.from_bounds,
                    lambda box=box: AABB.from_bounds(box.as_bounds()),
                    {"ndim": ndim, "kind": "round-trip"},
                    invariants=(
                        custom(
                            "as_bounds-round-trip",
                            lambda r, box=box: None
                            if r == box
                            else f"from_bounds(as_bounds()) != original: {r} vs {box}",
                        ),
                    ),
                )

                # Transformed enclosure: the invariant a translated domain breaks first.
                matrix = np.asarray(generator.uniform(-1.5, 1.5, (ndim, ndim)))
                matrix += ndim * np.eye(ndim)  # keep it comfortably non-singular
                affine = AffineTransform(matrix, np.full(ndim, 0.25 * (hi_val - lo_val)))
                yield Case(
                    GROUP,
                    f"aabb_transform_{tag}",
                    AABB.transform,
                    lambda box=box, affine=affine: box.transform(affine),
                    {"ndim": ndim, "lo": lo_val, "hi": hi_val, "kind": "affine"},
                    invariants=(
                        custom("encloses-corners", _encloses_transformed_corners(lo, hi, affine)),
                    ),
                    arrays={"lo": lo, "hi": hi, "matrix": matrix, "offset": affine.offset},
                )
                singular = np.zeros((ndim, ndim))
                yield Case(
                    GROUP,
                    f"aabb_transform_singular_{tag}",
                    AABB.transform,
                    lambda box=box, singular=singular, ndim=ndim: box.transform(
                        AffineTransform(singular, np.zeros(ndim))
                    ),
                    {"ndim": ndim, "kind": "singular-matrix"},
                )

    for ndim in (0, -1):
        yield Case(
            GROUP,
            f"aabb_unbounded_ndim{ndim}",
            AABB.unbounded,
            lambda ndim=ndim: AABB.unbounded(ndim),
            {"ndim": ndim},
            finite_inputs=False,
        )
        yield Case(
            GROUP,
            f"aabb_empty_ndim{ndim}",
            AABB.empty,
            lambda ndim=ndim: AABB.empty(ndim),
            {"ndim": ndim},
            finite_inputs=False,
        )


def _transform_cases(profile: Profile) -> Iterator[Case]:
    """Yield the :class:`pantr.transform.AffineTransform` cases.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: One hostile construction or operation.
    """
    generator = rng(12)
    for ndim in dims(profile, max_dim=4):
        eye = np.eye(ndim)
        yield Case(
            GROUP,
            f"affine_identity_d{ndim}",
            AffineTransform.identity,
            lambda ndim=ndim: AffineTransform.identity(ndim),
            {"ndim": ndim},
        )
        yield Case(
            GROUP,
            f"affine_singular_inverse_d{ndim}",
            AffineTransform.inverse,
            lambda ndim=ndim: AffineTransform(np.zeros((ndim, ndim))).inverse,
            {"ndim": ndim, "kind": "singular"},
        )
        yield Case(
            GROUP,
            f"affine_nonsquare_d{ndim}",
            AffineTransform,
            lambda ndim=ndim: AffineTransform(np.zeros((ndim, ndim + 1))),
            {"ndim": ndim, "kind": "non-square"},
        )
        yield Case(
            GROUP,
            f"affine_zero_scaling_d{ndim}",
            AffineTransform.scaling,
            lambda ndim=ndim: AffineTransform.scaling(np.zeros(ndim)),
            {"ndim": ndim, "kind": "zero-factor"},
        )
        yield Case(
            GROUP,
            f"affine_nan_translation_d{ndim}",
            AffineTransform,
            lambda eye=eye, ndim=ndim: AffineTransform(eye, np.full(ndim, np.nan)),
            {"ndim": ndim, "kind": "nan-translation"},
            finite_inputs=False,
        )

        if profile is not Profile.FULL:
            continue

        matrix = np.asarray(generator.uniform(-1.0, 1.0, (ndim, ndim))) + ndim * eye
        for offset_scale in (1.0, 1e6):
            offset = offset_scale * np.asarray(generator.uniform(-1.0, 1.0, ndim))
            affine = AffineTransform(matrix, offset)
            yield Case(
                GROUP,
                f"affine_inverse_round_trip_d{ndim}_off{offset_scale:g}",
                AffineTransform.inverse,
                lambda affine=affine: affine.inverse.compose(affine),
                {"ndim": ndim, "kind": "round-trip", "offset_scale": offset_scale},
                invariants=(
                    custom(
                        "inverse-composes-to-identity",
                        lambda r, ndim=ndim, matrix=matrix, offset=offset: _identity_failure(
                            r, ndim, matrix, offset
                        ),
                    ),
                ),
                arrays={"matrix": matrix, "offset": offset},
            )
        yield Case(
            GROUP,
            f"affine_rotation_nan_d{ndim}",
            AffineTransform.rotation_2d,
            lambda: AffineTransform.rotation_2d(np.nan),
            {"kind": "nan-angle"},
            finite_inputs=False,
        )
        yield Case(
            GROUP,
            f"affine_mirror_zero_normal_d{ndim}",
            AffineTransform.mirror,
            lambda ndim=ndim: AffineTransform.mirror(np.zeros(ndim)),
            {"ndim": ndim, "kind": "zero-normal"},
        )


def _identity_failure(
    result: object, ndim: int, matrix: np.ndarray, offset: np.ndarray
) -> str | None:
    """Check that a composed inverse is the identity to a conditioning-scaled bound.

    Both parts are checked, because they fail differently: the linear part is bounded
    by ``cond(A) * eps``, while the translation part carries the *offset's own*
    magnitude through ``-A^-1 b``, so a domain translated to ``1e6`` moves its
    attainable accuracy up by six orders. Checking only the matrix would call a
    translated round trip clean no matter how badly the offset cancelled.

    Args:
        result (object): The composed transform.
        ndim (int): Dimension.
        matrix (np.ndarray): The matrix that was inverted, whose condition number sets
            the attainable accuracy.
        offset (np.ndarray): The translation, whose magnitude scales the residual.

    Returns:
        str | None: ``None`` when the identity holds, otherwise the deviation.
    """
    if not isinstance(result, AffineTransform):
        return f"expected AffineTransform, got {type(result).__name__}"
    eps = float(np.finfo(np.float64).eps)
    cond = float(np.linalg.cond(matrix))
    tol = 16.0 * ndim * cond * eps
    worst = float(np.max(np.abs(result.matrix - np.eye(ndim))))
    if worst > tol:
        return f"max|A^-1 A - I| = {worst:.3e} > {tol:.3e} (cond = {cond:.3e})"
    scale = max(float(np.max(np.abs(offset))), 1.0)
    tol_offset = tol * scale
    worst_offset = float(np.max(np.abs(result.offset)))
    if worst_offset > tol_offset:
        return f"max|A^-1 b + b'| = {worst_offset:.3e} > {tol_offset:.3e} (scale {scale:.1e})"
    return None


def cases(profile: Profile) -> Iterator[Case]:
    """Yield every case in this group.

    Args:
        profile (Profile): Sweep width.

    Yields:
        Case: The group's cases.
    """
    yield from _aabb_cases(profile)
    yield from _transform_cases(profile)
