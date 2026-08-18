"""Join two B-spline patches along a parametric axis.

Provides :func:`join` for C0-concatenation of B-splines.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from ..bspline import Bspline, BsplineSpace, BsplineSpace1D
from ..tolerance import get_conservative
from ._compat import _remap_bspline, make_compat
from ._validation import _coarsest_dtype, _column_groups, _promote_to_rational


def _prepare_for_join(
    bspline1: Bspline,
    bspline2: Bspline,
    axis: int,
) -> tuple[Bspline, Bspline, bool]:
    """Prepare two B-splines for joining along an axis.

    Makes them compatible on non-join axes, promotes rational if mixed,
    clamps if needed, elevates to max degree on the join axis, and
    remaps b2 so it starts where b1 ends.

    Args:
        bspline1: First B-spline.
        bspline2: Second B-spline.
        axis: Join axis.

    Returns:
        tuple: ``(b1, b2, is_rational)`` ready for concatenation.
    """
    dim = bspline1.dim

    # Make compatible on non-join axes
    non_join = [i for i in range(dim) if i != axis]
    if non_join:
        b1, b2 = make_compat(bspline1, bspline2, axes=non_join)
    else:
        b1, b2 = bspline1, bspline2

    # Promote to rational if mixed
    is_rational = b1.is_rational or b2.is_rational
    if is_rational:
        b1 = _promote_to_rational(b1)
        b2 = _promote_to_rational(b2)

    # Clamp if needed
    if any(s.periodic or not s.has_open_knots() for s in b1.space.spaces):
        b1 = b1.to_open_bspline()
    if any(s.periodic or not s.has_open_knots() for s in b2.space.spaces):
        b2 = b2.to_open_bspline()

    # Elevate to max degree on the join axis
    p1 = b1.degree[axis]
    p2 = b2.degree[axis]
    p = max(p1, p2)
    if p > p1:
        inc = [0] * dim
        inc[axis] = p - p1
        b1 = b1.elevate_degree(inc)
    if p > p2:
        inc = [0] * dim
        inc[axis] = p - p2
        b2 = b2.elevate_degree(inc)

    # Remap b2 so it starts where b1 ends
    u1_end = float(b1.space.spaces[axis].domain[1])
    u2_start = float(b2.space.spaces[axis].domain[0])
    u2_end = float(b2.space.spaces[axis].domain[1])
    new_end = u1_end + (u2_end - u2_start)
    b2 = _remap_bspline(b2, axis, (u1_end, new_end))

    return b1, b2, is_rational


def _verify_shared_boundary(
    row1: npt.NDArray[np.float32 | np.float64],
    row2: npt.NDArray[np.float32 | np.float64],
    axis: int,
    is_rational: bool,
) -> None:
    """Verify that the two control rows meeting at the junction agree, before averaging them.

    The two rows are not independent data: they are two readings of the same shared face,
    and ``_concatenate_along_axis`` fuses them by averaging.  When they agree the
    average is that same row and the join is exact; when they do not, it is a new row
    belonging to neither input, and the result misses both by *half the gap* -- an error
    that grows with the mistake rather than staying near roundoff.  Refusing is the same
    verdict :func:`~pantr.cad.create_coons_surface` and
    :func:`~pantr.cad.create_coons_volume` reach on the same class of input.

    Both rows are compared in full, not only at the corners of the shared face.  Two
    patches meeting exactly at both ends of the seam and disagreeing in between produce a
    result that misses them along the whole interior of it, so the corners are necessary
    but not sufficient.  By this point ``_prepare_for_join`` has made the non-join axes
    compatible, so the two rows span the same space, where a spline is determined by its
    control points; comparing those is therefore an exact comparison of the two boundary
    patches and needs no sampling.

    The tolerance is ``_verify_corners_2d``'s tier -- its justification lives there and is
    not restated -- times the largest absolute value over the two rows compared, applied
    to what is graded here, which is a control-point coefficient rather than a point on a
    curve.

    **Why that magnitude.**  Not because a genuinely shared seam arrives bitwise equal;
    that is true only when one side is refined and the other left alone.  When each input
    carries its own interior knots on a non-join axis -- two surfaces meeting at a seam,
    each with its own transverse knot vector, which is the ordinary case -- ``make_compat``
    refines *both*, by different operator sequences, and the readings differ.  Every step
    between the caller's data and this comparison (rational promotion, join-axis
    elevation, which leaves the clamped row fixed, and non-join-axis elevation and
    insertion) is a convex combination, so its absolute rounding error is bounded by
    ``c * eps * max|coefficient|`` with ``max|coefficient|`` non-increasing under convex
    combination.  The floor is therefore ``O(eps) * scale``, and measured it is **0.5 to
    3.3 ulps** of ``scale`` -- flat in the number of insertions, since knot insertion is
    convex and the two sides' errors are correlated rather than independent -- across
    degrees 2 to 5 and magnitudes ``1e-9`` to ``1e15``.  Allowing a few ulps of build slack
    for FMA, vector width and libm, ``16 * eps * scale`` covers it and the conservative
    tier clears that by a further factor of 250.  That margin is the tier's own robustness
    reading, not an accident: the slack is for seam data the caller *computed* rather than
    copied -- a transform, an intersection, a reparametrization -- and is sized to catch
    pieces given in the wrong place or the wrong orientation, which miss by a fraction of
    the model rather than by epsilons.

    This is also what justifies the ``* scale`` multiplier twice over: ``max|row|`` is both
    the magnitude that transports ``eps`` into the compared space (verified proportional
    over 24 decades) and, by the convex hull below, the geometric magnitude it maps onto.

    **The scale is the seam's, not the model's**, since the two rows are the whole of what
    is compared.  That is never looser than the Coons rule, whose population spans the
    whole skeleton, because the seam is part of the model.  It is stricter, and the
    contract it puts on the caller is worth stating: *the seam must be supplied to within
    the tier times its own magnitude, not the model's*.  A seam near the origin on a large
    model, whose coordinates came out of a model-scale chain and so carry model-scale
    absolute error, is refused.  Erring that way is deliberate; it is the direction that
    reports rather than the direction that accepts.

    **Each column group is graded against its own magnitude.**  A rational row carries
    homogeneous coordinates, in units of length times weight, beside a dimensionless
    weight.  One max over the row would grade the weight against a threshold proportional
    to the model's *length*, which is dimensionally wrong and breaks scale invariance:
    sending ``x -> lambda*x`` scales the coordinates and leaves the weights alone.
    Measured on a seam containing a control point at the origin, the accepted geometric
    error then grows linearly with coordinate magnitude, from 0.12 times the tier at
    magnitude 1 to **1.2e7 times the tier at 1e8**, while the non-rational control stays
    flat at 0.50 at every magnitude.  Grading the coordinate columns against
    ``max|X|`` and the weight column against ``max|w|`` restores it: under
    ``x -> lambda*x`` both group scales and both group gaps carry ``lambda``, and under a
    global weight rescale ``w -> mu*w`` the homogeneous coordinates ``X = w*x`` carry
    ``mu`` alongside the weights, so both groups carry it too.  It does not disarm the
    check: a *one-sided* weight rescale still leaves ``gap_w = (mu - 1)*w`` against
    ``tier * max(w, mu*w)`` and is still refused.  The weight column's own roundoff floor
    against ``max|w|`` is measured at ``<= 3.2 * eps``, independent of magnitude, degree
    and refinement, so the tier clears it by about 1290.

    The tier is evaluated at the **coarser of the two inputs' precisions**, not at
    float64, because pantr's B-spline layer accepts float32 as well.  If the true shared
    face is ``F``, the two stored readings satisfy ``|row_d - F| <= 0.5 * eps_d * |F|``, so
    for a genuinely shared face ``|row1 - row2| <= 0.5 * (eps_1 + eps_2) * scale``, which
    is about ``0.5 * eps_coarse * scale``.  That is a hard floor on any achievable
    agreement: any tolerance below it refuses a perfect seam.  The tier clears it by 8192.
    Taking the *promoted* type instead would be exactly backwards, missing the floor by a
    factor of ``5.4e8`` for a mixed pair.  The consequence is worth stating plainly, since
    "the same tier as the Coons path" reads as "the same strictness" and it is not: at
    float32 the tier is ``4.88e-4`` relative, so a float32 join accepts a seam gap of
    ``1e-5`` of the seam scale and refuses ``1e-3``.  It still catches what the check exists
    for; it does not catch a small displacement error that float64 would refuse.  The rule
    also grades by *container* rather than by provenance, so float32-resolution data
    already copied into a float64 array is graded at float64.

    For non-rational inputs the basis is non-negative and sums to one -- and a tensor
    product of non-negative partitions of unity is one -- so the largest control-point gap
    bounds the largest gap between the two boundary patches themselves: the number
    reported is then a distance in space.  Measured over 400 random pairs the ratio
    saturates at exactly 1.  For rational inputs it is not: the comparison is of
    **homogeneous** control points, which is what the average combines, and turning that
    into a distance would need a lower bound on the weights that this function does not
    compute; measured, the ratio reaches ``9e4`` at weights of 0.01.  Comparing them
    homogeneously is still the right test -- two rows that project to the same points
    under different weights average to a row whose weights are neither input's, so the
    curve on both sides of the seam is not the curve either input described -- but the
    reported gap is then a coefficient gap.

    One consequence of comparing coefficients: what is required is that the two boundary
    patches agree **as parametrized**, which is strictly stronger than agreeing as point
    sets.  That is the right condition for a join, whose result has to be one spline, but
    it means two seams tracing the same curve at different speeds are refused.

    Args:
        row1 (npt.NDArray[np.float32 | np.float64]): The first B-spline's control row at
            the end of the join axis, of shape ``(*shared_face_shape, rank)``.
        row2 (npt.NDArray[np.float32 | np.float64]): The second B-spline's control row at
            the start of the join axis, of the same shape.
        axis (int): Join axis, named in the message so a caller can tell which seam failed.
        is_rational (bool): Whether the rows are homogeneous, i.e. whether the trailing
            column is a weight and has to be graded against its own magnitude.

    Raises:
        ValueError: If the two rows disagree anywhere on the shared face, in either column
            group, by more than that group's tolerance.
    """
    tier = get_conservative(_coarsest_dtype(row1, row2))
    groups = _column_groups(is_rational=is_rational)

    # Coordinates first: a seam given in the wrong place fails there, and it is the more
    # informative half to report when both groups are out.
    for group in groups:
        left, right = row1[..., group.columns], row2[..., group.columns]
        # No floor: a group whose two halves are all zero has no scale, and is also
        # bitwise equal, so a zero tolerance is the right answer there rather than a
        # hazard.
        scale = max(float(np.abs(left).max()), float(np.abs(right).max()))
        tol = tier * scale
        gaps = np.abs(left - right)
        gap = float(gaps.max())
        if gap <= tol:
            continue

        # Report the control point that disagrees most, so the gap named is the gap
        # compared.  The whole point is printed even though one group decided the verdict.
        per_point = gaps.max(axis=-1)
        index = np.unravel_index(int(np.argmax(per_point)), per_point.shape)
        # A curve's shared face is a single control point, whose index is the empty tuple:
        # printing it would read like a defect rather than like a location.
        where = f" at index {tuple(int(i) for i in index)}" if index else ""
        raise ValueError(
            f"Shared boundary mismatch on axis {axis}{where}: "
            f"the first B-spline gives {row1[index]}, the second gives {row2[index]} "
            f"(gap {gap:.3e}, above {tol:.3e} at boundary {group.name} scale {scale:.3e})."
        )


def _concatenate_along_axis(
    b1: Bspline,
    b2: Bspline,
    axis: int,
    is_rational: bool,
) -> Bspline:
    """Concatenate two prepared B-splines along an axis.

    Verifies that the two shared boundary control rows agree, averages them, merges knot
    vectors, and builds the result.

    Args:
        b1: First B-spline (left side).
        b2: Second B-spline (right side, already remapped).
        axis: Join axis.
        is_rational: Whether the result is rational.

    Returns:
        Bspline: The concatenated B-spline.

    Raises:
        ValueError: If the two shared boundary control rows disagree, as
            ``_verify_shared_boundary`` grades it.
    """
    dim = b1.dim
    p = b1.degree[axis]
    cp1 = b1.control_points
    cp2 = b2.control_points

    # Slice indices for the join axis (+ rank dimension)
    ndim = dim + 1
    idx_left: list[int | slice | None] = [slice(None)] * ndim
    idx_right: list[int | slice | None] = [slice(None)] * ndim
    idx_b1_last: list[int | slice | None] = [slice(None)] * ndim
    idx_b2_first: list[int | slice | None] = [slice(None)] * ndim

    idx_left[axis] = slice(0, -1)
    idx_right[axis] = slice(1, None)
    idx_b1_last[axis] = -1
    idx_b2_first[axis] = 0

    cp_left = cp1[tuple(idx_left)]
    cp_right = cp2[tuple(idx_right)]

    row1 = cp1[tuple(idx_b1_last)]
    row2 = cp2[tuple(idx_b2_first)]
    _verify_shared_boundary(row1, row2, axis, is_rational)
    # Exact when the two rows agree, and they are known to by now: (x + x) / 2 returns x
    # for every finite x whose double does not overflow.  Averaging rather than picking a
    # side keeps the result symmetric in its two inputs.
    cp_mid = (row1 + row2) / 2.0

    idx_expand: list[int | slice | None] = [slice(None)] * ndim
    idx_expand[axis] = np.newaxis
    cp_mid_expanded = cp_mid[tuple(idx_expand)]

    new_cp = np.concatenate([cp_left, cp_mid_expanded, cp_right], axis=axis)

    # Merge knot vectors.  The junction knots take the incoming knots' dtype: built from a
    # Python float they would be float64, and concatenating would promote a float32 knot
    # vector while the control points stayed float32, which BsplineSpace1D refuses.
    u_junction = float(b1.space.spaces[axis].domain[1])
    kl = b1.space.spaces[axis].knots[: -p - 1]
    kr = b2.space.spaces[axis].knots[p + 1 :]
    kc = np.full(p, u_junction, dtype=kl.dtype)
    new_knots = np.concatenate([kl, kc, kr])

    spaces = list(b1.space.spaces)
    spaces[axis] = BsplineSpace1D(new_knots, degree=p)
    new_space = BsplineSpace(spaces)

    result = Bspline(new_space, new_cp, is_rational=is_rational)

    # Try to remove junction knots for smoother join
    if p > 1:
        result = _try_remove_junction_knots(result, axis, u_junction, p)

    return result


def join(bspline1: Bspline, bspline2: Bspline, axis: int) -> Bspline:
    """Join two B-splines along a parametric axis with C0 continuity.

    The two inputs are made compatible on all non-join axes (degree and
    knots), elevated to the same degree on the join axis, and then
    concatenated.  The shared boundary control points are averaged.

    Optionally, knots are removed at the junction to recover higher
    smoothness when the geometry permits it.

    **The two inputs must already meet.**  The face where they join is not independent
    data: ``bspline1``'s boundary there and ``bspline2``'s are two readings of the same
    thing, and averaging them is exact only because they agree.  Pieces that do not meet
    are refused rather than reconciled, since the average of two disagreeing rows lies on
    neither input and the result would then pass through neither, missing each by half
    the gap.  For rational inputs -- including a rational joined to a polynomial, which is
    promoted first -- agreement is required of the **homogeneous** control points, weights
    included, so two boundaries that describe the same points of space under different
    weights are refused too.

    Args:
        bspline1: First B-spline (left side of the join).
        bspline2: Second B-spline (right side of the join).
        axis: Parametric axis along which to join (0-indexed).

    Returns:
        Bspline: A single B-spline spanning both inputs.

    Raises:
        ValueError: If the inputs have different parametric dimensions.
        ValueError: If *axis* is out of range.
        ValueError: If the two inputs disagree anywhere on the face they are joined
            along, by more than ``4096 * eps`` times the largest absolute control-point
            value over the two boundary rows compared. This is
            :func:`create_coons_surface`'s tier, justified in ``_verify_corners_2d``, with
            two differences that follow from what is graded here and are derived in
            ``_verify_shared_boundary``: ``eps`` is that of the coarser of the two inputs'
            **own precisions** rather than always float64, so a float32 seam is graded at
            float32; and the scale is taken over the **seam** rather than over the whole
            boundary skeleton, so a seam must agree to within the tier times its own
            magnitude, not the model's. For a rational join the homogeneous coordinates
            and the weights are graded against their own magnitudes separately, since they
            do not share a physical dimension.
    """
    dim = bspline1.dim
    if dim != bspline2.dim:
        raise ValueError(
            f"Both B-splines must have the same dim, got {bspline1.dim} and {bspline2.dim}."
        )
    if axis < 0 or axis >= dim:
        raise ValueError(f"axis must be in [0, {dim}), got {axis}.")

    b1, b2, is_rational = _prepare_for_join(bspline1, bspline2, axis)
    return _concatenate_along_axis(b1, b2, axis, is_rational)


def _try_remove_junction_knots(
    bspline: Bspline,
    axis: int,
    u_junction: float,
    p: int,
) -> Bspline:
    """Try removing junction knots to increase smoothness.

    Attempts to remove up to ``p - 1`` copies of the junction knot.

    Args:
        bspline: The joined B-spline.
        axis: Join axis.
        u_junction: Knot value at the junction.
        p: Degree on the join axis.

    Returns:
        Bspline: B-spline with some junction knots removed, or the
        original if removal was not possible.
    """
    dim = bspline.dim
    if dim == 1:
        return bspline.remove_knots(u_junction, num=p - 1)

    per_dim: list[npt.ArrayLike | None] = [None] * dim
    per_dim[axis] = np.array([u_junction])
    return bspline.remove_knots(per_dim, num=p - 1)
