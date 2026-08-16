"""Coons patch and volume constructions.

Provides :func:`create_coons_surface` (bilinear blending from 4 boundary curves)
and :func:`create_coons_volume` (trilinear blending from 6 boundary faces).
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from ..bspline import Bspline, BsplineSpace, BsplineSpace1D
from ..tolerance import get_conservative
from ._compat import make_compat
from ._operations import create_ruled
from ._primitives import _linear_space_1d, create_bilinear, create_trilinear
from ._validation import _promote_to_rational

_CORNER_MATCH_TIER = get_conservative
"""Which :mod:`pantr.tolerance` tier grades a corner mismatch.

The conservative tier, ``4096 * eps``, and the choice is about what this check is
*for*. Both curves meeting at a corner read it straight off a clamped control point,
and knot insertion and degree elevation both leave a clamped endpoint untouched, so
:func:`~pantr.cad.make_compat` does not move it: two curves that genuinely share a
corner produce **bitwise equal** values, and no tolerance at all is needed for them.
The slack exists only for a corner the caller *computed* rather than copied, by an
intersection, a reparametrization or a transform, and the length of that chain is not
something this function can see.

So the threshold is not sized against a known accumulation; it is sized to catch the
mistake the check exists for. That mistake is curves given in the wrong order or the
wrong orientation, which misses by a fraction of the patch, not by epsilons. Failing
in the other direction is the expensive one: a ``ValueError`` on a geometrically
perfect patch. The conservative tier is the widest the table sanctions without a
derivation of its own, and at unit scale it lands on ``9.1e-13``, within a factor of
1.1 of the ``1e-12`` this check used before it acquired a scale.
"""


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
            absolute coordinate over all eight corner values (see
            :func:`_verify_corners_2d`).
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

    The tolerance is ``_CORNER_MATCH_TIER(float64) * scale`` with ``scale`` the largest
    absolute coordinate over **all eight** corner values, not just the pair under test.
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
    tol = _CORNER_MATCH_TIER(np.float64) * scale

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
        Bspline: A 3D (trivariate) B-spline volume.

    Raises:
        ValueError: If any face is not a 2D B-spline.
    """
    (face_u0, face_u1), (face_v0, face_v1), (face_w0, face_w1) = faces

    for f in (face_u0, face_u1, face_v0, face_v1, face_w0, face_w1):
        if f.dim != 2:  # noqa: PLR2004
            raise ValueError(f"All faces must be 2D B-splines, got dim={f.dim}.")

    # Make opposite face pairs compatible
    face_u0, face_u1 = make_compat(face_u0, face_u1)
    face_v0, face_v1 = make_compat(face_v0, face_v1)
    face_w0, face_w1 = make_compat(face_w0, face_w1)

    # Extract edges from faces
    edges = _extract_edges((face_u0, face_u1), (face_v0, face_v1), (face_w0, face_w1))

    # Extract corners from edges
    corners = _extract_corners(edges)

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
    t = create_trilinear(corners)

    # Make all 7 terms compatible and combine
    r_u, r_v, r_w, b_uv, b_uw, b_vw, t = make_compat(r_u, r_v, r_w, b_uv, b_uw, b_vw, t)

    cp, is_rational = _combine_control_points(
        [r_u, r_v, r_w, b_uv, b_uw, b_vw, t],
        [+1, +1, +1, -1, -1, -1, +1],
    )
    return Bspline(r_u.space, cp, is_rational=is_rational)


def _extract_edges(
    u_faces: tuple[Bspline, Bspline],
    v_faces: tuple[Bspline, Bspline],
    w_faces: tuple[Bspline, Bspline],
) -> dict[str, Bspline]:
    """Extract 12 edges from 6 faces.

    Each edge is extracted from the first face that contains it.
    Edge naming: ``{free_param}_{fixed1}_{fixed2}``.

    Args:
        u_faces: ``(face_u0, face_u1)`` at u=0 and u=1, parameterized by (v, w).
        v_faces: ``(face_v0, face_v1)`` at v=0 and v=1, parameterized by (u, w).
        w_faces: Not used for edge extraction but reserved for future validation.

    Returns:
        dict[str, Bspline]: Dictionary of 12 named edge curves.
    """
    face_u0, face_u1 = u_faces
    face_v0, face_v1 = v_faces

    # All faces are 2D, so boundary() returns Bspline (not ndarray).
    # We cast to satisfy mypy.
    def _bdy(face: Bspline, ax: int, side: int) -> Bspline:
        result = face.boundary(ax, side)
        assert isinstance(result, Bspline)
        return result

    return {
        # w-edges (free=w): from face_u (v,w) boundaries at v=0,1
        "w_u0_v0": _bdy(face_u0, 0, 0),
        "w_u0_v1": _bdy(face_u0, 0, 1),
        "w_u1_v0": _bdy(face_u1, 0, 0),
        "w_u1_v1": _bdy(face_u1, 0, 1),
        # v-edges (free=v): from face_u (v,w) boundaries at w=0,1
        "v_u0_w0": _bdy(face_u0, 1, 0),
        "v_u0_w1": _bdy(face_u0, 1, 1),
        "v_u1_w0": _bdy(face_u1, 1, 0),
        "v_u1_w1": _bdy(face_u1, 1, 1),
        # u-edges (free=u): from face_v (u,w) boundaries at w=0,1
        "u_v0_w0": _bdy(face_v0, 1, 0),
        "u_v0_w1": _bdy(face_v0, 1, 1),
        "u_v1_w0": _bdy(face_v1, 1, 0),
        "u_v1_w1": _bdy(face_v1, 1, 1),
    }


def _extract_corners(edges: dict[str, Bspline]) -> npt.NDArray[np.float64]:
    """Extract 8 corner points from edges.

    Args:
        edges: Dictionary of 12 named edge curves.

    Returns:
        npt.NDArray[np.float64]: Array of shape ``(2, 2, 2, rank)``.
    """
    e = edges["u_v0_w0"]
    rank = e.control_points.shape[-1]
    corners = np.zeros((2, 2, 2, rank), dtype=np.float64)

    # Corners from u-edges (boundary of a 1D curve returns ndarray)
    corners[0, 0, 0] = np.asarray(edges["u_v0_w0"].boundary(0, 0))
    corners[1, 0, 0] = np.asarray(edges["u_v0_w0"].boundary(0, 1))
    corners[0, 1, 0] = np.asarray(edges["u_v1_w0"].boundary(0, 0))
    corners[1, 1, 0] = np.asarray(edges["u_v1_w0"].boundary(0, 1))
    corners[0, 0, 1] = np.asarray(edges["u_v0_w1"].boundary(0, 0))
    corners[1, 0, 1] = np.asarray(edges["u_v0_w1"].boundary(0, 1))
    corners[0, 1, 1] = np.asarray(edges["u_v1_w1"].boundary(0, 0))
    corners[1, 1, 1] = np.asarray(edges["u_v1_w1"].boundary(0, 1))
    return corners


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
