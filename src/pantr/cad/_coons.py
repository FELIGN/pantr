"""Coons patch and volume constructions.

Provides :func:`create_coons_surface` (bilinear blending from 4 boundary curves)
and :func:`create_coons_volume` (trilinear blending from 6 boundary faces).
"""

from __future__ import annotations

from itertools import combinations
from typing import NamedTuple

import numpy as np
from numpy import typing as npt

from ..bspline import Bspline, BsplineSpace, BsplineSpace1D
from ..tolerance import get_conservative
from ._compat import make_compat
from ._operations import create_ruled
from ._primitives import _linear_space_1d, create_bilinear
from ._validation import _promote_to_rational

_DIRECTION_NAMES = ("u", "v", "w")
"""tuple[str, str, str]: Names of a Coons volume's parametric directions, in axis order."""


class _EdgeReadings(NamedTuple):
    """One edge of a Coons volume, as read off each of the two faces that carry it.

    Two faces share a whole edge, so each of the twelve edges can be read twice, and the
    two readings must agree for the volume to interpolate both faces.

    Attributes:
        label (str): Edge name, ``{free}_{fixed}{side}_{fixed}{side}`` (e.g. ``w_u0_v1``).
        face_a (str): Name of the first face carrying the edge (e.g. ``face_u0``).
        curve_a (Bspline): The edge as ``face_a`` sees it.
        face_b (str): Name of the second face carrying the edge.
        curve_b (Bspline): The edge as ``face_b`` sees it.
        corners (tuple[str, str]): Labels of the volume corners at the low and high ends
            of the edge, in the ``(i,j,k)`` form of :func:`_corner_label`.
    """

    label: str
    face_a: str
    curve_a: Bspline
    face_b: str
    curve_b: Bspline
    corners: tuple[str, str]


def _corner_label(sides: tuple[int, int, int]) -> str:
    """Name a corner of a Coons volume by the side it occupies in each direction.

    Args:
        sides (tuple[int, int, int]): Side (0 or 1) along u, v and w.

    Returns:
        str: Corner label ``"(i,j,k)"``, matching the ``"(i,j)"`` form of the 2D messages.
    """
    return f"({sides[0]},{sides[1]},{sides[2]})"


def _combine_control_points(
    bsplines: list[Bspline],
    signs: list[int],
) -> tuple[npt.NDArray[np.float64], bool]:
    """Linearly combine control points of compatible B-splines.

    Args:
        bsplines: B-splines that share the same space (after compat).
        signs: Coefficients (+1 or -1) for each B-spline.

    Returns:
        tuple: ``(control_points, is_rational)`` for the combined result.
    """
    is_rational = any(b.is_rational for b in bsplines)
    if is_rational:
        bsplines = [_promote_to_rational(b) for b in bsplines]

    cp = np.zeros_like(bsplines[0].control_points, dtype=np.float64)
    for b, s in zip(bsplines, signs, strict=True):
        cp += s * b.control_points.astype(np.float64)
    return cp, is_rational


def create_coons_surface(
    curves: tuple[tuple[Bspline, Bspline], tuple[Bspline, Bspline]],
) -> Bspline:
    """Construct a Coons patch from four boundary curves.

    Builds a bilinearly blended surface from four boundary curves using
    the formula ``S = R0 + R1 - B``, where *R0* and *R1* are ruled
    surfaces and *B* is the bilinear interpolant of the four corners.

    The curve layout is::

               C_u1
        o--------------o
        |  v           |
        |  ^           |
        |  |     C_v1  |
        |  +---> u     |
        o--------------o
        C_v0   C_u0

    Args:
        curves: ``((C_v0, C_v1), (C_u0, C_u1))`` -- two pairs of
            opposite boundary curves.  ``C_v0`` and ``C_v1`` run in the
            v-direction (left/right).  ``C_u0`` and ``C_u1`` run in the
            u-direction (bottom/top).  All must be 1D B-splines.

    Returns:
        Bspline: A 2D B-spline surface.

    Raises:
        ValueError: If any curve is not 1D.
        ValueError: If corner points are not geometrically consistent, i.e. two curves
            disagree at a shared corner by more than ``4096 * eps`` times the largest
            absolute coordinate over all eight corner values.
    """
    (c_v0, c_v1), (c_u0, c_u1) = curves

    for c in (c_v0, c_v1, c_u0, c_u1):
        if c.dim != 1:
            raise ValueError(f"All curves must be 1D, got dim={c.dim}.")

    # Make opposite pairs compatible
    c_u0, c_u1 = make_compat(c_u0, c_u1)
    c_v0, c_v1 = make_compat(c_v0, c_v1)

    # Extract and verify corner points
    p00 = np.asarray(c_u0.boundary(0, 0), dtype=np.float64)
    p10 = np.asarray(c_u0.boundary(0, 1), dtype=np.float64)
    p01 = np.asarray(c_u1.boundary(0, 0), dtype=np.float64)
    p11 = np.asarray(c_u1.boundary(0, 1), dtype=np.float64)

    _verify_corners_2d((p00, p10, p01, p11), c_v0, c_v1)

    # R0: ruled in v from C_v0 to C_v1, then transpose to (u, v)
    r0 = create_ruled(c_v0, c_v1).permute_directions([1, 0])
    # R1: ruled in u from C_u0 to C_u1 (already in (u, v) order)
    r1 = create_ruled(c_u0, c_u1)
    # B: bilinear from corners
    corners = np.zeros((2, 2, p00.size), dtype=np.float64)
    corners[0, 0] = p00
    corners[1, 0] = p10
    corners[0, 1] = p01
    corners[1, 1] = p11
    b = create_bilinear(corners)

    r0, r1, b = make_compat(r0, r1, b)

    cp, is_rational = _combine_control_points([r0, r1, b], [+1, +1, -1])
    return Bspline(r0.space, cp, is_rational=is_rational)


def _verify_corners_2d(
    u_corners: tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ],
    c_v0: Bspline,
    c_v1: Bspline,
) -> None:
    """Verify that corner points from u-curves match v-curves.

    The tolerance is ``get_conservative(float64) * scale`` with ``scale`` the largest
    absolute coordinate over **all eight** corner values, not just the pair under test.

    **Why the conservative tier.** Both curves meeting at a corner read it straight off a
    clamped control point, and neither knot insertion nor degree elevation moves a clamped
    endpoint, so :func:`~pantr.cad.make_compat` leaves it alone: two curves that genuinely
    share a corner produce **bitwise equal** values and need no tolerance at all. The slack
    exists only for a corner the caller *computed* rather than copied -- an intersection, a
    reparametrization, a transform -- and the length of that chain is not something this
    function can see. So the threshold is not sized against a known accumulation; it is
    sized to catch the mistake the check exists for, which is curves given in the wrong
    order or the wrong orientation, missing by a fraction of the patch rather than by
    epsilons. Failing the other way is the expensive one: a ``ValueError`` on a
    geometrically perfect patch. The conservative tier is the widest the table sanctions
    without a derivation of its own, and at unit scale it lands on ``9.1e-13``, within a
    factor of 1.1 of the ``1e-12`` this check used before it acquired a scale.
    Taking it over all eight is what makes the verdict independent of which corner
    happens to sit at the origin: a corner at ``(0, 0, 0)`` supplies no scale of its
    own, and grading it against zero would demand bitwise agreement of a coordinate the
    caller may well have computed.

    A fixed absolute tolerance cannot do this. Measured across twelve decades of
    ``1e-12 / L``, a one-ulp corner mismatch was rejected at ``L >= 1e6`` -- a
    micron-unit model of a metre-scale part failing on a geometrically perfect patch --
    while a ``1e-9`` *relative* gap was accepted at ``L = 1e-6``.

    ``np.allclose(..., rtol=0)`` is deliberately not used: with ``rtol`` zeroed it is
    exactly ``max|pu - qv| <= atol``, so it buys nothing over writing that, and it hides
    which of the two comparisons is doing the work.

    Args:
        u_corners (tuple[npt.NDArray[np.float64], ...]): Corners
            ``(p00, p10, p01, p11)`` extracted from u-curves.
        c_v0 (Bspline): Left boundary curve (v-direction at u=0).
        c_v1 (Bspline): Right boundary curve (v-direction at u=1).

    Raises:
        ValueError: If any pair of corners does not match.
    """
    p00, p10, p01, p11 = u_corners
    q00 = np.asarray(c_v0.boundary(0, 0), dtype=np.float64)
    q01 = np.asarray(c_v0.boundary(0, 1), dtype=np.float64)
    q10 = np.asarray(c_v1.boundary(0, 0), dtype=np.float64)
    q11 = np.asarray(c_v1.boundary(0, 1), dtype=np.float64)

    pairs = [
        ("(0,0)", p00, q00),
        ("(1,0)", p10, q10),
        ("(0,1)", p01, q01),
        ("(1,1)", p11, q11),
    ]
    # No floor: eight corners all at the origin have no scale, and are also bitwise
    # equal, so a zero tolerance is the right answer there rather than a hazard.
    scale = max(float(np.abs(corner).max()) for _, pu, qv in pairs for corner in (pu, qv))
    tol = get_conservative(np.float64) * scale

    for label, pu, qv in pairs:
        gap = float(np.abs(pu - qv).max())
        if gap > tol:
            raise ValueError(
                f"Corner {label} mismatch: u-curve gives {pu}, v-curve gives {qv} "
                f"(gap {gap:.3e}, above {tol:.3e} at corner scale {scale:.3e})."
            )


def create_coons_volume(
    faces: tuple[
        tuple[Bspline, Bspline],
        tuple[Bspline, Bspline],
        tuple[Bspline, Bspline],
    ],
) -> Bspline:
    """Construct a trivariate Coons blending volume from 6 boundary faces.

    Uses the trilinear Coons formula:

    ``V = (R_u + R_v + R_w) - (B_uv + B_uw + B_vw) + T``

    where *R* are ruled volumes from opposite face pairs, *B* are
    bilinear blend volumes from edge quadruples, and *T* is the
    trilinear corner interpolant.

    Edges and corners are derived automatically from face boundaries.

    The six faces are not independent data: each of the twelve edges is shared by two of
    them and each of the eight corners by three, so they have to agree where they meet.
    Faces that do not are refused rather than blended, because the formula spreads the
    inconsistency over the volume and the result then misses faces the caller supplied
    correctly.

    **Rational faces are supported, and the blend is applied in homogeneous space.**  All
    seven terms are formed from homogeneous coefficients ``(w x, w y, w z, w)`` and the
    result is projected only when it is evaluated, so the formula stays the linear one
    above and nothing has to be said about how weights combine under division.  That is
    what makes the boundary property survive: projection is pointwise, so it commutes with
    restriction to a face, and a blend whose *homogeneous* restriction to ``u = 0`` is
    ``face_u0``'s homogeneous data projects to ``face_u0`` itself.  A polynomial face
    among rational ones is promoted with unit weights, so mixed input takes the same path
    rather than a third one.

    That argument has a hypothesis, and it is the one the checks below enforce: the faces
    must agree in **homogeneous** data, weights included, not merely projectively.  A
    homogeneous representation is unique only up to one constant scaling all of a patch's
    weights, so two faces can trace the same edge in space while their coefficients differ
    by a factor, and then the inclusion-exclusion does not cancel.  This is measured rather
    than feared: scaling one ``w``-face's homogeneous control points by 2 leaves that
    face's geometry bitwise unchanged yet, with the checks disabled, makes the volume miss
    each of the four faces adjacent to it by 17% of the model.

    Face labelling convention:

    - ``faces[0] = (face_u0, face_u1)``: faces at u=0 and u=1,
      each a surface parameterized by (v, w).
    - ``faces[1] = (face_v0, face_v1)``: faces at v=0 and v=1,
      each a surface parameterized by (u, w).
    - ``faces[2] = (face_w0, face_w1)``: faces at w=0 and w=1,
      each a surface parameterized by (u, v).

    Args:
        faces: Three pairs of opposite boundary faces.

    Returns:
        Bspline: A 3D (trivariate) B-spline volume.  Because it returns rather than raising,
        the six faces agree where they meet, and the volume therefore restricts to each of
        them on the corresponding boundary.  That interpolation is exact in exact arithmetic;
        the floating-point residual is **observed** to stay near ``1e-15`` relative, three
        orders below the tolerance the checks below use, and is not a proved bound.

    Raises:
        ValueError: If any face is not a 2D B-spline.
        ValueError: If two faces meeting at a corner of the volume disagree there by more
            than ``4096 * eps`` times the largest absolute coordinate over all
            twenty-four corner readings -- the same tier and the same scale rule as
            :func:`create_coons_surface`, derived in ``_verify_corners_3d``.
        ValueError: If two faces sharing an edge disagree along it by more than
            ``4096 * eps`` times the largest absolute control-point coordinate over the
            twelve compared edges, derived in ``_verify_edges_3d``.
        ValueError: If the blend of rational faces carries a control weight that is not
            above zero, so that the weight field of the result cannot be certified
            positive.  The bound is ``w(u) >= min_i w_i`` and is sufficient rather than
            necessary, so this refuses some volumes whose weight field never vanishes;
            ``_verify_positive_weights`` derives it.
    """
    (face_u0, face_u1), (face_v0, face_v1), (face_w0, face_w1) = faces

    for f in (face_u0, face_u1, face_v0, face_v1, face_w0, face_w1):
        if f.dim != 2:  # noqa: PLR2004
            raise ValueError(f"All faces must be 2D B-splines, got dim={f.dim}.")

    # Make opposite face pairs compatible
    face_u0, face_u1 = make_compat(face_u0, face_u1)
    face_v0, face_v1 = make_compat(face_v0, face_v1)
    face_w0, face_w1 = make_compat(face_w0, face_w1)

    # Read every edge from both faces that carry it, and refuse faces that disagree
    edge_readings = _extract_edge_pairs((face_u0, face_u1), (face_v0, face_v1), (face_w0, face_w1))
    _verify_corners_3d(edge_readings)
    _verify_edges_3d(edge_readings)

    # The construction reads each edge from the first of its two carriers; the other
    # reading is only there to be checked against, and by now is known to agree.
    edges = {r.label: r.curve_a for r in edge_readings}

    # Extract corners from edges, in the space the other six terms are combined in
    rational_faces = any(
        f.is_rational for f in (face_u0, face_u1, face_v0, face_v1, face_w0, face_w1)
    )
    corners = _extract_corners(edges, is_rational=rational_faces)

    # Build 3 ruled volumes
    r_u = create_ruled(face_u0, face_u1).permute_directions([2, 0, 1])
    r_v = create_ruled(face_v0, face_v1).permute_directions([0, 2, 1])
    r_w = create_ruled(face_w0, face_w1)  # already (u, v, w)

    # Build 3 bilinear blend volumes
    # B_uv: raw axes (u, v, w) → permute to (u, v, w) = identity
    b_uv = _build_bilinear_volume(
        edges["w_u0_v0"],
        edges["w_u1_v0"],
        edges["w_u0_v1"],
        edges["w_u1_v1"],
        permutation=(0, 1, 2),
    )
    # B_uw: raw axes (u, w, v) → permute to (u, v, w)
    b_uw = _build_bilinear_volume(
        edges["v_u0_w0"],
        edges["v_u1_w0"],
        edges["v_u0_w1"],
        edges["v_u1_w1"],
        permutation=(0, 2, 1),
    )
    # B_vw: raw axes (v, w, u) → permute to (u, v, w)
    b_vw = _build_bilinear_volume(
        edges["u_v0_w0"],
        edges["u_v1_w0"],
        edges["u_v0_w1"],
        edges["u_v1_w1"],
        permutation=(2, 0, 1),
    )

    # Build trilinear volume
    t = _build_trilinear_volume(corners, is_rational=rational_faces)

    # Make all 7 terms compatible and combine
    r_u, r_v, r_w, b_uv, b_uw, b_vw, t = make_compat(r_u, r_v, r_w, b_uv, b_uw, b_vw, t)

    cp, is_rational = _combine_control_points(
        [r_u, r_v, r_w, b_uv, b_uw, b_vw, t],
        [+1, +1, +1, -1, -1, -1, +1],
    )
    volume = Bspline(r_u.space, cp, is_rational=is_rational)
    if is_rational:
        _verify_positive_weights(volume)
    return volume


def _extract_edge_pairs(
    u_faces: tuple[Bspline, Bspline],
    v_faces: tuple[Bspline, Bspline],
    w_faces: tuple[Bspline, Bspline],
) -> tuple[_EdgeReadings, ...]:
    """Extract the 12 edges of the volume, each from both faces that carry it.

    An edge free in direction *f* is fixed in the other two, and the face pair of either
    fixed direction carries it: the edge free in *w* at ``u=i, v=j`` is a boundary of
    ``face_u{i}`` and a boundary of ``face_v{j}``.  ``curve_a`` is the reading from the
    face of the lower fixed direction, which is the one the construction goes on to use,
    and reproduces what this function returned before it read the w-faces at all.

    Within a face the two surviving directions keep their volume order, so a fixed
    direction sits at face axis 0 when it precedes *f* and at face axis 1 when it follows.

    Args:
        u_faces: ``(face_u0, face_u1)`` at u=0 and u=1, parameterized by (v, w).
        v_faces: ``(face_v0, face_v1)`` at v=0 and v=1, parameterized by (u, w).
        w_faces: ``(face_w0, face_w1)`` at w=0 and w=1, parameterized by (u, v).

    Returns:
        tuple[_EdgeReadings, ...]: The 12 edges, ordered by free direction w, then v,
        then u, and named as :func:`_extract_corners` and the blend volumes expect.
    """
    face_pairs = (u_faces, v_faces, w_faces)

    # All faces are 2D, so boundary() returns Bspline (not ndarray).
    # We assert to satisfy mypy.
    def _bdy(face: Bspline, ax: int, side: int) -> Bspline:
        result = face.boundary(ax, side)
        assert isinstance(result, Bspline)
        return result

    readings: list[_EdgeReadings] = []
    for free in (2, 1, 0):
        axis_a, axis_b = (axis for axis in range(3) if axis != free)
        name_free, name_a, name_b = (_DIRECTION_NAMES[d] for d in (free, axis_a, axis_b))
        face_axis_a = 0 if axis_b < free else 1
        face_axis_b = 0 if axis_a < free else 1
        for side_a in (0, 1):
            for side_b in (0, 1):
                sides = [0, 0, 0]
                sides[axis_a], sides[axis_b] = side_a, side_b
                ends = []
                for end in (0, 1):
                    sides[free] = end
                    ends.append(_corner_label((sides[0], sides[1], sides[2])))
                readings.append(
                    _EdgeReadings(
                        label=f"{name_free}_{name_a}{side_a}_{name_b}{side_b}",
                        face_a=f"face_{name_a}{side_a}",
                        curve_a=_bdy(face_pairs[axis_a][side_a], face_axis_a, side_b),
                        face_b=f"face_{name_b}{side_b}",
                        curve_b=_bdy(face_pairs[axis_b][side_b], face_axis_b, side_a),
                        corners=(ends[0], ends[1]),
                    )
                )
    return tuple(readings)


def _verify_corners_3d(edges: tuple[_EdgeReadings, ...]) -> None:
    """Verify that the three faces meeting at each corner of the volume agree there.

    The tolerance is :func:`_verify_corners_2d`'s, which is where its derivation lives:
    ``get_conservative(float64)`` times the largest absolute coordinate over **all**
    corner readings, so the verdict does not depend on which corner sits at the origin.
    Three faces meet at a corner of a volume rather than two curves at a corner of a
    patch, so the scale is taken over twenty-four readings instead of eight and each
    corner is graded by three comparisons instead of one; nothing else changes.

    The w-faces are the reason this exists.  Every edge, and so every corner, used to be
    read off the u- and v-faces alone, which agree with themselves by construction, so a
    w-face contradicting them was never compared with anything.

    Args:
        edges (tuple[_EdgeReadings, ...]): The 12 edges of the volume, read from both
            carriers, as :func:`_extract_edge_pairs` returns them.

    Raises:
        ValueError: If two faces meeting at a corner disagree there by more than the
            tolerance.
    """
    # Each (corner, face) pair is an endpoint of two of the twelve edges; read it once.
    readings: dict[tuple[str, str], npt.NDArray[np.float64]] = {}
    for edge in edges:
        for end, corner in enumerate(edge.corners):
            for face, curve in ((edge.face_a, edge.curve_a), (edge.face_b, edge.curve_b)):
                if (corner, face) not in readings:
                    readings[corner, face] = np.asarray(curve.boundary(0, end), dtype=np.float64)

    # No floor: corners all at the origin have no scale, and are also bitwise equal,
    # so a zero tolerance is the right answer there rather than a hazard.
    scale = max(float(np.abs(point).max()) for point in readings.values())
    tol = get_conservative(np.float64) * scale

    for sides in np.ndindex(2, 2, 2):
        corner = _corner_label((int(sides[0]), int(sides[1]), int(sides[2])))
        meeting = tuple(f"face_{name}{s}" for name, s in zip(_DIRECTION_NAMES, sides, strict=True))
        for face_a, face_b in combinations(meeting, 2):
            point_a, point_b = readings[corner, face_a], readings[corner, face_b]
            gap = float(np.abs(point_a - point_b).max())
            if gap > tol:
                raise ValueError(
                    f"Corner {corner} mismatch: {face_a} gives {point_a}, "
                    f"{face_b} gives {point_b} "
                    f"(gap {gap:.3e}, above {tol:.3e} at corner scale {scale:.3e})."
                )


def _verify_edges_3d(edges: tuple[_EdgeReadings, ...]) -> None:
    """Verify that the two faces sharing each edge of the volume agree along all of it.

    Agreeing at the eight corners is necessary but not sufficient: two faces share a whole
    edge, and a volume built from faces that meet only at the endpoints of a shared edge
    misses one of them by the size of the disagreement in between.  Agreeing on all twelve
    edges *is* sufficient, and a face's interior is free: displacing an interior control
    point of one face by half the model size leaves all six faces interpolated to ``2e-15``.
    That is the classical compatibility condition for transfinite interpolation, observed
    here rather than proved.

    Made compatible, the two readings span the same space, where a spline is determined by
    its control points; comparing those is therefore an exact comparison of the two curves
    and needs no sampling.  The tolerance is :func:`_verify_corners_2d`'s tier and rule
    applied to what is graded here, which is a control-point coefficient rather than a
    point on the curve: ``get_conservative(float64)`` times the largest absolute
    control-point coordinate over all twelve compared edges.

    **Why any slack at all.** Two readings of a genuinely shared edge are usually **bitwise
    equal**, and not only when the two faces store it identically: reaching the common space
    runs the same degree elevation and knot insertion on the same coefficients, so the
    roundoff is the same on both sides and cancels.  Measured on a shared edge arriving at
    degree 1 from one face and degree 2 from the other, and on one arriving with different
    knot vectors, the gap was exactly zero in both cases.  So the threshold is not sized
    against that roundoff.  As in two dimensions it exists for face data the caller
    *computed* rather than copied -- a transform, an intersection, a reparametrization --
    and is sized to catch faces given in the wrong order or the wrong orientation, which
    miss by a fraction of the model rather than by epsilons.

    For non-rational faces the convex-hull property makes the control-point gap an upper
    bound on the geometric gap, so the number reported is a distance in space.  For rational
    faces it is not: the comparison is of **homogeneous** control points, which is what the
    seven-term formula combines, and turning that into a distance would need a lower bound
    on the weights that this function does not compute.  Comparing them is still the right
    test -- scaling one face's homogeneous control points by a constant leaves its geometry
    untouched yet made the volume miss four of the six faces by 17% of the model before this
    check existed -- but the reported gap is then a coefficient gap, not a distance.

    Args:
        edges (tuple[_EdgeReadings, ...]): The 12 edges of the volume, read from both
            carriers, as :func:`_extract_edge_pairs` returns them.

    Raises:
        ValueError: If two faces sharing an edge disagree along it by more than the
            tolerance.
    """
    compared: list[tuple[_EdgeReadings, npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
    for edge in edges:
        curve_a, curve_b = edge.curve_a, edge.curve_b
        if curve_a.is_rational or curve_b.is_rational:
            curve_a, curve_b = _promote_to_rational(curve_a), _promote_to_rational(curve_b)
        curve_a, curve_b = make_compat(curve_a, curve_b)
        compared.append(
            (
                edge,
                np.asarray(curve_a.control_points, dtype=np.float64),
                np.asarray(curve_b.control_points, dtype=np.float64),
            )
        )

    scale = max(float(np.abs(cp).max()) for _, cp_a, cp_b in compared for cp in (cp_a, cp_b))
    tol = get_conservative(np.float64) * scale

    for edge, cp_a, cp_b in compared:
        gap = float(np.abs(cp_a - cp_b).max())
        if gap > tol:
            raise ValueError(
                f"Edge {edge.label} mismatch: {edge.face_a} and {edge.face_b} disagree "
                f"along the edge they share "
                f"(gap {gap:.3e}, above {tol:.3e} at edge scale {scale:.3e})."
            )


def _verify_positive_weights(volume: Bspline) -> None:
    """Refuse a rational Coons volume whose weight field is not certified positive.

    A NURBS volume is ``X(u) / w(u)`` and is only defined where its weight field stays
    away from zero.  Positive weights on the six faces do **not** carry over to the
    blend: the seven-term formula is an inclusion-exclusion and subtracts three of its
    terms, so a weight field assembled from strictly positive faces can dip to zero or
    below inside, and the volume then has a pole the caller never put there.

    **What a control weight settles.**  The result's weight field is
    ``w(u) = sum_i N_i(u) w_i`` over its own tensor-product B-spline basis, which is
    non-negative and sums to one on the domain -- a tensor product of non-negative
    partitions of unity is one -- so

        ``w(u) - min_j w_j = sum_i N_i(u) (w_i - min_j w_j) >= 0``

    for every ``u``.  The weight field therefore never goes below the smallest control
    weight, and reading the weight column of the **result's** control points bounds it
    from below without evaluating anything.

    **Sufficient, not necessary, and deliberately so.**  The converse of that inequality
    is false: ``w(u)`` is a convex combination of the ``w_i`` only at the ends, and in
    between the basis is strictly positive on more than one coefficient, so a volume
    whose weight field never comes near zero can still carry a negative control weight
    and be refused here.  This check is therefore conservative: everything it accepts has
    a positive weight field, but it does not accept everything that has one.  It reports
    the bound it can prove rather than pretending to detect the pole itself, which would
    need a global minimum of ``w`` over the domain.

    Args:
        volume (Bspline): The rational volume just assembled, whose control points carry
            a trailing weight column.

    Raises:
        ValueError: If any control weight of *volume* is zero or negative, so that the
            lower bound no longer keeps the weight field away from zero.
    """
    weights = volume.control_points[..., -1]
    smallest = float(weights.min())
    if smallest > 0.0:
        return

    index = tuple(int(i) for i in np.unravel_index(int(np.argmin(weights)), weights.shape))
    raise ValueError(
        f"Coons volume weight is not certified positive: control weight {smallest:.3e} "
        f"at index {index} is not above zero, so the bound w(u) >= min_i w_i no longer "
        "keeps the blended weight field away from zero and the volume may have a pole. "
        "The seven-term blend subtracts three of its terms, so positive weights on the "
        "six faces do not imply positive weights on the volume; supply faces whose "
        "weights vary less across the model. The test is sufficient, not necessary: a "
        "volume whose weight field never vanishes can still fail it."
    )


def _homogeneous_endpoint(curve: Bspline, side: int) -> npt.NDArray[np.float32 | np.float64]:
    """Read one end of a curve as a coefficient, without projecting a rational one.

    :meth:`~pantr.bspline.Bspline.boundary` **projects**: sliced down to a point, a
    rational curve is divided by its weight and returns ``rank`` components where its
    coefficients have ``rank + 1``.  Every other term of the Coons blend is combined
    homogeneously, so a corner read that way is both one component short and in a
    different space from the terms it is subtracted from.

    Viewing the same coefficients as an ordinary curve of one rank higher is what stops
    before the division: the de Boor recurrence underneath is linear and does not consult
    ``is_rational`` (:func:`~pantr.bspline._bspline_slice._slice_bspline` branches on it
    only for the final ``X / w``), so the arithmetic is identical and the result is the
    homogeneous corner the blend needs.

    Args:
        curve (Bspline): A 1-D B-spline, rational or not.
        side (int): Which end of the domain: ``0`` for the start, ``1`` for the end.

    Returns:
        npt.NDArray[np.float32 | np.float64]: The curve's coefficient at that end, of
        shape ``(rank,)`` when *curve* is non-rational and ``(rank + 1,)`` when it is
        rational, in *curve*'s own dtype.
    """
    homogeneous = Bspline(curve.space, curve.control_points, is_rational=False)
    return np.asarray(homogeneous.boundary(0, side))


def _extract_corners(
    edges: dict[str, Bspline], *, is_rational: bool
) -> npt.NDArray[np.float32 | np.float64]:
    """Extract the volume's 8 corner coefficients from the four edges free in u.

    The array is sized and filled in **one** space, which is what the corner term of the
    blend needs and what this used to get wrong: it took its width from the edges'
    coefficients, which are homogeneous for a rational face, and filled it from
    :meth:`~pantr.bspline.Bspline.boundary`, which projects.  A rational volume then
    failed with a NumPy broadcast error naming shapes ``(3,)`` and ``(4,)`` rather than
    anything the caller had supplied (pantr issue 309).  :func:`_homogeneous_endpoint` is
    the read that stays in the space the width was taken from.

    *is_rational* is the **volume's**, not these four edges': the corner term is one of
    seven that are added together, so it has to live in the same space as the other six
    even when the faces that happen to carry the u-edges are polynomial.  Promoting them
    gives those corners a unit weight, which is the right value exactly when the faces
    that *are* rational carry a unit weight there too -- and :func:`_verify_edges_3d` has
    already required that, since it compares each edge's two readings homogeneously and a
    polynomial reading promotes to weight one.

    Args:
        edges (dict[str, Bspline]): The 12 named edge curves.
        is_rational (bool): Whether the volume being built is rational, i.e. whether the
            corners are wanted as homogeneous coefficients.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Corner coefficients of shape
        ``(2, 2, 2, rank)``, indexed ``[i, j, k]`` by side along u, v and w, where
        ``rank`` counts the weight when *is_rational*.
    """
    u_edges = {(j, k): edges[f"u_v{j}_w{k}"] for j in (0, 1) for k in (0, 1)}
    if is_rational:
        u_edges = {sides: _promote_to_rational(e) for sides, e in u_edges.items()}

    sample = u_edges[0, 0].control_points
    corners = np.empty((2, 2, 2, sample.shape[-1]), dtype=sample.dtype)
    for (j, k), edge in u_edges.items():
        for i in (0, 1):
            corners[i, j, k] = _homogeneous_endpoint(edge, i)
    return corners


def _build_trilinear_volume(
    corners: npt.NDArray[np.float32 | np.float64], *, is_rational: bool
) -> Bspline:
    """Build the trilinear corner term of the Coons blend from the 8 corner coefficients.

    :func:`~pantr.cad.create_trilinear` cannot serve here, for two reasons that both come
    from its being a public *geometric* primitive: it caps the rank at three, so it
    rejects the homogeneous ``(w x, w y, w z, w)`` corner of a rational face, and it
    builds its space at float64 whatever it was handed, which a float32 model cannot then
    be combined with.  This is the same construction as :func:`_build_bilinear_volume`'s,
    one direction further.

    Args:
        corners (npt.NDArray[np.float32 | np.float64]): Corner coefficients of shape
            ``(2, 2, 2, rank)``, as :func:`_extract_corners` returns them.
        is_rational (bool): Whether *corners* carries a trailing weight column.

    Returns:
        Bspline: A 3D, degree-(1, 1, 1) B-spline volume in (u, v, w) order.
    """
    knots = _linear_space_1d(dtype=corners.dtype).knots
    space = BsplineSpace([BsplineSpace1D(knots.copy(), degree=1) for _ in range(3)])
    return Bspline(space, corners, is_rational=is_rational)


def _build_bilinear_volume(
    e_00: Bspline,
    e_10: Bspline,
    e_01: Bspline,
    e_11: Bspline,
    permutation: tuple[int, int, int],
) -> Bspline:
    """Build a trivariate B-spline bilinear in two directions, free in one.

    The four input edges are 1D curves that share the same free parameter.
    They are placed at the four combinations of the two bilinear parameters.
    The raw axes are ``(bilinear_0, bilinear_1, free)``, then permuted to
    ``(u, v, w)`` via *permutation*.

    Args:
        e_00: Edge at (bilinear_0=0, bilinear_1=0).
        e_10: Edge at (bilinear_0=1, bilinear_1=0).
        e_01: Edge at (bilinear_0=0, bilinear_1=1).
        e_11: Edge at (bilinear_0=1, bilinear_1=1).
        permutation: Permutation such that ``new[i] = old[permutation[i]]``,
            mapping from raw (bilinear_0, bilinear_1, free) to (u, v, w).

    Returns:
        Bspline: A 3D B-spline volume in (u, v, w) order.
    """
    e_00, e_10, e_01, e_11 = make_compat(e_00, e_10, e_01, e_11)

    is_rational = any(e.is_rational for e in (e_00, e_10, e_01, e_11))
    if is_rational:
        e_00 = _promote_to_rational(e_00)
        e_10 = _promote_to_rational(e_10)
        e_01 = _promote_to_rational(e_01)
        e_11 = _promote_to_rational(e_11)

    cp = e_00.control_points
    n_free = cp.shape[0]
    rank_full = cp.shape[-1]

    # Shape: (2, 2, n_free, rank_full) = (bilinear_0, bilinear_1, free, rank)
    new_cp = np.empty((2, 2, n_free, rank_full), dtype=np.float64)
    new_cp[0, 0] = e_00.control_points
    new_cp[1, 0] = e_10.control_points
    new_cp[0, 1] = e_01.control_points
    new_cp[1, 1] = e_11.control_points

    lin = _linear_space_1d()
    spaces_raw = [
        BsplineSpace1D(lin.knots.copy(), degree=1),
        BsplineSpace1D(lin.knots.copy(), degree=1),
        e_00.space.spaces[0],
    ]
    space = BsplineSpace(spaces_raw)
    vol = Bspline(space, new_cp, is_rational=is_rational)

    return vol.permute_directions(list(permutation))
