"""Tests for Coons surface and volume constructions."""

from __future__ import annotations

import re

import numpy as np
import pytest
from numpy import typing as npt
from numpy.testing import assert_allclose

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.cad import (
    create_bilinear,
    create_circle,
    create_coons_surface,
    create_coons_volume,
    create_extrusion,
    create_line,
    create_ruled,
)
from pantr.cad._coons import _extract_edge_pairs
from pantr.tolerance import get_conservative, get_machine_epsilon

_RANK_3D = 3

_FacePairs = tuple[
    tuple[Bspline, Bspline],
    tuple[Bspline, Bspline],
    tuple[Bspline, Bspline],
]
"""The argument of :func:`~pantr.cad.create_coons_volume`: three pairs of opposite faces."""

_VOLUME_FACE_LABELS = ("u0", "u1", "v0", "v1", "w0", "w1")
"""Flat order of the six boundary faces, matching the pairs of :data:`_FacePairs`."""

_FACE_SAMPLES = np.linspace(0.0, 1.0, 11)
"""Normalized per-direction samples used to compare a volume's boundary against a face."""


def _flatten_faces(faces: _FacePairs) -> list[Bspline]:
    """Flatten three pairs of opposite faces into ``(u0, u1, v0, v1, w0, w1)`` order.

    Args:
        faces (_FacePairs): Three pairs of opposite boundary faces.

    Returns:
        list[Bspline]: The six faces in :data:`_VOLUME_FACE_LABELS` order.
    """
    return [face for pair in faces for face in pair]


def _to_domain(spline: Bspline, axis: int, s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Map normalized parameters in ``[0, 1]`` onto ``spline``'s domain along ``axis``.

    ``make_compat`` remaps domains affinely, so normalized parameters are what makes a
    volume parameter and a face parameter name the same geometric point.

    Args:
        spline (Bspline): The B-spline whose domain is the target.
        axis (int): Parametric direction of *spline*.
        s (npt.NDArray[np.float64]): Normalized parameters in ``[0, 1]``.

    Returns:
        npt.NDArray[np.float64]: Parameters in ``spline``'s own domain along *axis*.
    """
    a, b = (float(x) for x in spline.space.spaces[axis].domain)
    return a + s * (b - a)


def _face_interpolation_gaps(vol: Bspline, faces: _FacePairs) -> dict[str, float]:
    """Measure how far a Coons volume's boundary is from each face it was built from.

    A Coons volume's defining property is that it interpolates all six boundary faces,
    so every entry must vanish to roundoff.  Nothing else in this file checks more than
    one face.

    Args:
        vol (Bspline): The trivariate volume under test.
        faces (_FacePairs): The six faces it was built from.

    Returns:
        dict[str, float]: Face label from :data:`_VOLUME_FACE_LABELS` mapped to
        ``max |V restricted to that face - the face|`` over :data:`_FACE_SAMPLES`.
    """
    pairs = np.array([[a, b] for a in _FACE_SAMPLES for b in _FACE_SAMPLES])
    gaps: dict[str, float] = {}
    for k, (label, face) in enumerate(zip(_VOLUME_FACE_LABELS, _flatten_faces(faces), strict=True)):
        axis, side = k // 2, np.array([float(k % 2)])
        free = [c for c in range(3) if c != axis]
        vol_params = np.empty((pairs.shape[0], 3))
        vol_params[:, axis] = _to_domain(vol, axis, side)[0]
        vol_params[:, free[0]] = _to_domain(vol, free[0], pairs[:, 0])
        vol_params[:, free[1]] = _to_domain(vol, free[1], pairs[:, 1])
        face_params = np.column_stack(
            [_to_domain(face, 0, pairs[:, 0]), _to_domain(face, 1, pairs[:, 1])]
        )
        got = np.asarray(vol.evaluate(vol_params), dtype=np.float64)
        want = np.asarray(face.evaluate(face_params), dtype=np.float64)
        gaps[label] = float(np.linalg.norm(got - want, axis=1).max())
    return gaps


def _reported_numbers(message: str) -> tuple[float, float, float]:
    """Parse the gap, tolerance and scale a mismatch message reports.

    Whatever the message says about the tolerance is a claim about the comparison that was
    actually made, so it is worth checking rather than trusting.  The name between ``at``
    and ``scale`` is the population the tolerance was taken over -- ``corner``, ``edge
    coordinate``, ``edge weight`` -- and is checked separately by the tests that care which
    column group decided the verdict.

    Args:
        message (str): The ``ValueError`` message.

    Returns:
        tuple[float, float, float]: ``(gap, tol, scale)`` as printed.
    """
    num = r"([-+0-9.eE]+)"
    found = re.search(rf"gap {num}, above {num} at [a-z -]+ scale {num}", message)
    assert found is not None, f"unparseable mismatch message: {message!r}"
    gap, tol, scale = (float(g) for g in found.groups())
    return gap, tol, scale


class TestCoonsSurface:
    """Test the coons_surface function."""

    def test_four_straight_lines_gives_bilinear(self) -> None:
        """Test Coons from 4 straight lines matches bilinear."""
        c_u0 = create_line([0, 0, 0], [1, 0, 0])
        c_u1 = create_line([0, 1, 0], [1, 1, 0])
        c_v0 = create_line([0, 0, 0], [0, 1, 0])
        c_v1 = create_line([1, 0, 0], [1, 1, 0])

        srf = create_coons_surface(((c_v0, c_v1), (c_u0, c_u1)))
        assert srf.dim == 2
        assert srf.rank == _RANK_3D

        # Should match bilinear
        corners = np.array(
            [[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]]],
            dtype=np.float64,
        )
        ref = create_bilinear(corners)
        t = np.linspace(0, 1, 10)
        params = np.array([[u, v] for u in t for v in t])
        assert_allclose(srf.evaluate(params), ref.evaluate(params), atol=1e-13)

    def test_evaluate_corners(self) -> None:
        """Test that Coons surface evaluates correctly at corners."""
        c_u0 = create_line([0, 0, 0], [3, 0, 0])
        c_u1 = create_line([0, 2, 0], [3, 2, 0])
        c_v0 = create_line([0, 0, 0], [0, 2, 0])
        c_v1 = create_line([3, 0, 0], [3, 2, 0])

        srf = create_coons_surface(((c_v0, c_v1), (c_u0, c_u1)))
        pts = srf.evaluate(
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ]
            )
        )
        assert_allclose(pts[0], [0, 0, 0], atol=1e-14)
        assert_allclose(pts[1], [3, 0, 0], atol=1e-14)
        assert_allclose(pts[2], [0, 2, 0], atol=1e-14)
        assert_allclose(pts[3], [3, 2, 0], atol=1e-14)

    def test_non_planar_coons(self) -> None:
        """Test Coons with non-planar boundaries."""
        c_u0 = create_line([0, 0, 0], [1, 0, 0])
        c_u1 = create_line([0, 0, 1], [1, 0, 1])
        c_v0 = create_line([0, 0, 0], [0, 0, 1])
        c_v1 = create_line([1, 0, 0], [1, 0, 1])

        srf = create_coons_surface(((c_v0, c_v1), (c_u0, c_u1)))
        # Should be a flat rectangle in the xz plane
        pt = srf.evaluate(np.array([[0.5, 0.5]]))
        assert_allclose(pt, [0.5, 0, 0.5], atol=1e-13)

    def test_non_1d_curve_raises(self) -> None:
        """Test that a surface as input raises ValueError."""
        srf = create_bilinear()
        crv = create_line([0, 0, 0], [1, 0, 0])
        with pytest.raises(ValueError, match="1D"):
            create_coons_surface(((crv, crv), (srf, crv)))

    def test_corner_mismatch_raises(self) -> None:
        """Test that inconsistent corners raise ValueError."""
        c_u0 = create_line([0, 0, 0], [1, 0, 0])
        c_u1 = create_line([0, 1, 0], [1, 1, 0])
        c_v0 = create_line([0, 0, 0], [0, 1, 0])
        c_v1 = create_line([2, 0, 0], [2, 1, 0])  # wrong: should start at (1,0,0)

        with pytest.raises(ValueError, match="mismatch"):
            create_coons_surface(((c_v0, c_v1), (c_u0, c_u1)))


class TestCoonsCornerToleranceScaleCovariance:
    """The corner check must reach the same verdict at every model scale.

    Its tolerance is ``4096 * eps`` times the largest absolute coordinate over all
    eight corner values.  Before, a fixed ``atol=1e-12`` rejected a one-ulp mismatch
    on a metre-scale part measured in microns, and accepted a ``1e-9`` relative gap
    on a micron-scale one.
    """

    @staticmethod
    def _square(scale: float, corner_shift: float = 0.0) -> tuple[Bspline, ...]:
        """Build the four boundary curves of a square of side ``scale``.

        ``corner_shift`` displaces the start of the v1 curve, so it is the single
        corner inconsistency under test.
        """
        s = scale
        c_u0 = create_line([0, 0, 0], [s, 0, 0])
        c_u1 = create_line([0, s, 0], [s, s, 0])
        c_v0 = create_line([0, 0, 0], [0, s, 0])
        c_v1 = create_line([s + corner_shift, 0, 0], [s, s, 0])
        return c_v0, c_v1, c_u0, c_u1

    @pytest.mark.parametrize("scale", [1.0e-6, 1.0e-3, 1.0, 1.0e3, 1.0e6, 1.0e9])
    def test_a_one_ulp_corner_mismatch_is_accepted_at_every_scale(self, scale: float) -> None:
        """A geometrically perfect patch must never be refused for a last-bit disagreement.

        This is the direction the absolute constant failed in: at ``scale >= 1e6`` one
        ulp of the coordinate already exceeds ``1e-12``.
        """
        shift = float(np.nextafter(scale, np.inf) - scale)
        c_v0, c_v1, c_u0, c_u1 = self._square(scale, corner_shift=shift)
        srf = create_coons_surface(((c_v0, c_v1), (c_u0, c_u1)))
        assert srf.dim == 2

    @pytest.mark.parametrize(
        "scale", [1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-3, 1.0, 1.0e3, 1.0e6, 1.0e9]
    )
    def test_a_relative_corner_gap_is_rejected_at_every_scale(self, scale: float) -> None:
        """And this is the direction it failed in on a small model.

        A gap of one part per million of the model size is a real modelling mistake at
        any scale; the absolute constant accepted it once the *absolute* gap fell below
        ``1e-12``, i.e. from ``scale = 1e-6`` down.  The list has to reach ``1e-9`` to
        pin that: at exactly ``1e-6`` the product ``1e-6 * 1e-6`` rounds a hair above
        ``1e-12`` and the old code still rejected, so a range stopping there passes on
        the unfixed code and proves nothing.
        """
        c_v0, c_v1, c_u0, c_u1 = self._square(scale, corner_shift=1.0e-6 * scale)
        with pytest.raises(ValueError, match="mismatch"):
            create_coons_surface(((c_v0, c_v1), (c_u0, c_u1)))

    def test_the_scale_comes_from_all_eight_corners_not_the_pair_under_test(self) -> None:
        """A patch touching the origin is graded like the rest of the patch.

        The ``(0, 0)`` corner supplies no magnitude of its own.  Reading the scale off
        only the pair being compared would grade it against zero and demand bitwise
        agreement there while allowing ``4096 * eps * s`` at the other three.
        """
        s = 1.0e6
        c_u0 = create_line([float(np.nextafter(0.0, 1.0)), 0, 0], [s, 0, 0])
        c_u1 = create_line([0, s, 0], [s, s, 0])
        c_v0 = create_line([0, 0, 0], [0, s, 0])
        c_v1 = create_line([s, 0, 0], [s, s, 0])
        srf = create_coons_surface(((c_v0, c_v1), (c_u0, c_u1)))
        assert srf.dim == 2

    @pytest.mark.parametrize("scale", [1.0e-6, 1.0, 1.0e9])
    def test_a_gross_mismatch_is_still_caught_at_every_scale(self, scale: float) -> None:
        """The mistake the check exists for: a curve given in the wrong place."""
        c_v0, c_v1, c_u0, c_u1 = self._square(scale, corner_shift=scale)
        with pytest.raises(ValueError, match="mismatch"):
            create_coons_surface(((c_v0, c_v1), (c_u0, c_u1)))


class TestCoonsVolume:
    """Test the coons_volume function."""

    def _make_cube_faces(
        self,
    ) -> tuple[
        tuple[Bspline, Bspline],
        tuple[Bspline, Bspline],
        tuple[Bspline, Bspline],
    ]:
        """Build 6 planar faces of the unit cube [0,1]^3."""
        # face_u0: u=0 plane, parameterized by (v, w)
        face_u0 = create_bilinear(
            np.array([[[0, 0, 0], [0, 0, 1]], [[0, 1, 0], [0, 1, 1]]], dtype=np.float64)
        )
        # face_u1: u=1 plane
        face_u1 = create_bilinear(
            np.array([[[1, 0, 0], [1, 0, 1]], [[1, 1, 0], [1, 1, 1]]], dtype=np.float64)
        )
        # face_v0: v=0 plane, parameterized by (u, w)
        face_v0 = create_bilinear(
            np.array([[[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [1, 0, 1]]], dtype=np.float64)
        )
        # face_v1: v=1 plane
        face_v1 = create_bilinear(
            np.array([[[0, 1, 0], [0, 1, 1]], [[1, 1, 0], [1, 1, 1]]], dtype=np.float64)
        )
        # face_w0: w=0 plane, parameterized by (u, v)
        face_w0 = create_bilinear(
            np.array([[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]]], dtype=np.float64)
        )
        # face_w1: w=1 plane
        face_w1 = create_bilinear(
            np.array([[[0, 0, 1], [0, 1, 1]], [[1, 0, 1], [1, 1, 1]]], dtype=np.float64)
        )
        return (face_u0, face_u1), (face_v0, face_v1), (face_w0, face_w1)

    def test_cube_properties(self) -> None:
        """Test that Coons volume from cube faces has correct properties."""
        faces = self._make_cube_faces()
        vol = create_coons_volume(faces)
        assert vol.dim == _RANK_3D
        assert vol.rank == _RANK_3D

    def test_cube_evaluate_center(self) -> None:
        """Test evaluation at the center of the unit cube."""
        faces = self._make_cube_faces()
        vol = create_coons_volume(faces)
        pt = vol.evaluate(np.array([[0.5, 0.5, 0.5]]))
        assert_allclose(pt, [0.5, 0.5, 0.5], atol=1e-13)

    def test_cube_evaluate_corners(self) -> None:
        """Test evaluation at all 8 corners of the unit cube."""
        faces = self._make_cube_faces()
        vol = create_coons_volume(faces)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    pt = vol.evaluate(np.array([[float(i), float(j), float(k)]]))
                    assert_allclose(pt, [float(i), float(j), float(k)], atol=1e-13)

    def test_cube_boundaries_match_faces(self) -> None:
        """Test that volume boundaries match the input faces."""
        faces = self._make_cube_faces()
        vol = create_coons_volume(faces)
        # Evaluate on the u=0 face
        t = np.linspace(0, 1, 5)
        params_face_u0 = np.array([[0.0, v, w] for v in t for w in t])
        pts_vol = vol.evaluate(params_face_u0)
        # Compare with face_u0 evaluation
        params_face = np.array([[v, w] for v in t for w in t])
        (face_u0, _), _, _ = faces
        pts_face = face_u0.evaluate(params_face)
        assert_allclose(pts_vol, pts_face, atol=1e-12)

    def test_non_2d_face_raises(self) -> None:
        """Test that a 1D input raises ValueError."""
        crv = create_line([0, 0, 0], [1, 0, 0])
        srf = create_bilinear()
        with pytest.raises(ValueError, match="2D"):
            create_coons_volume(((crv, crv), (srf, srf), (srf, srf)))


class TestCoonsVolumeFaceConsistency:
    """Six faces of a volume are not independent data, and must be checked as such.

    Every edge of a Coons volume is shared by two of the six faces and every corner by
    three, so faces given in the wrong order or the wrong orientation contradict each
    other where they meet.  ``create_coons_volume`` used to read all twelve edges off
    the u- and v-faces alone, so a disagreement involving a w-face was resolved silently
    in favour of whichever face the edge happened to come from: the volume then missed,
    by the full size of the inconsistency, two faces the caller had supplied correctly
    (pantr issue 301).  ``create_coons_surface`` has always refused such input.

    The tolerance is the surface path's: ``get_conservative(float64)`` times the largest
    absolute coordinate of what is being compared.  Its derivation is in
    ``pantr.cad._coons._verify_corners_2d``.
    """

    @staticmethod
    def _cube_face_corners(scale: float = 1.0) -> list[npt.NDArray[np.float64]]:
        """Return corner arrays for the six planar faces of the cube ``[0, scale]^3``.

        Each face is indexed by its own two parameters, in increasing volume-axis order:
        ``u0``/``u1`` by ``(v, w)``, ``v0``/``v1`` by ``(u, w)``, ``w0``/``w1`` by
        ``(u, v)``.  At ``scale = 1`` these are, entry for entry, the arrays of the
        reproduction filed with pantr issue 301.

        Args:
            scale (float): Side of the cube. Defaults to 1.0.

        Returns:
            list[npt.NDArray[np.float64]]: Six ``(2, 2, 3)`` corner arrays in
            :data:`_VOLUME_FACE_LABELS` order.
        """
        s = scale
        return [
            np.array([[[0, 0, 0], [0, 0, s]], [[0, s, 0], [0, s, s]]], dtype=float),
            np.array([[[s, 0, 0], [s, 0, s]], [[s, s, 0], [s, s, s]]], dtype=float),
            np.array([[[0, 0, 0], [0, 0, s]], [[s, 0, 0], [s, 0, s]]], dtype=float),
            np.array([[[0, s, 0], [0, s, s]], [[s, s, 0], [s, s, s]]], dtype=float),
            np.array([[[0, 0, 0], [0, s, 0]], [[s, 0, 0], [s, s, 0]]], dtype=float),
            np.array([[[0, 0, s], [0, s, s]], [[s, 0, s], [s, s, s]]], dtype=float),
        ]

    @staticmethod
    def _pairs(faces: list[Bspline]) -> _FacePairs:
        """Group six faces in :data:`_VOLUME_FACE_LABELS` order into opposite pairs.

        Args:
            faces (list[Bspline]): The six boundary faces.

        Returns:
            _FacePairs: The argument ``create_coons_volume`` expects.
        """
        u0, u1, v0, v1, w0, w1 = faces
        return ((u0, u1), (v0, v1), (w0, w1))

    def _cube(self, scale: float = 1.0) -> _FacePairs:
        """Build the six consistent planar faces of the cube ``[0, scale]^3``.

        Args:
            scale (float): Side of the cube. Defaults to 1.0.

        Returns:
            _FacePairs: Six mutually consistent bilinear faces.
        """
        return self._pairs([create_bilinear(a) for a in self._cube_face_corners(scale)])

    def _cube_with_lifted_corner(
        self, face: int, corner: tuple[int, int], lift: float, scale: float = 1.0
    ) -> _FacePairs:
        """Build cube faces with one corner of one face displaced along *z*.

        The three faces meeting at that corner no longer agree, so no volume can
        interpolate all six.  Which of the three carries the displacement is the point:
        the edges were derived from the u- and v-faces only, so a displacement on a
        w-face went unnoticed while the same error on a u-face did not.

        Args:
            face (int): Index into :data:`_VOLUME_FACE_LABELS` of the face to displace.
            corner (tuple[int, int]): Corner of that face, in its own ``(p, q)`` indices.
            lift (float): Displacement along *z*.
            scale (float): Side of the cube. Defaults to 1.0.

        Returns:
            _FacePairs: Six faces that disagree at exactly one corner.
        """
        arrays = self._cube_face_corners(scale)
        arrays[face][corner][2] += lift
        return self._pairs([create_bilinear(a) for a in arrays])

    def _hexahedron(self, corners: npt.NDArray[np.float64]) -> _FacePairs:
        """Build the six bilinear faces of the hexahedron with the given eight corners.

        Faces sharing an edge take identical control points there by construction, so any
        eight corners give a consistent set, however skewed.

        Args:
            corners (npt.NDArray[np.float64]): Shape ``(2, 2, 2, 3)``, indexed
                ``[i][j][k]`` by the side along u, v and w.

        Returns:
            _FacePairs: Six mutually consistent bilinear faces.
        """
        p = corners
        return self._pairs(
            [
                create_bilinear(p[0]),
                create_bilinear(p[1]),
                create_bilinear(p[:, 0]),
                create_bilinear(p[:, 1]),
                create_bilinear(p[:, :, 0]),
                create_bilinear(p[:, :, 1]),
            ]
        )

    @staticmethod
    def _bezier_patch(
        control_points: npt.NDArray[np.float64], degree_u: int, degree_v: int
    ) -> Bspline:
        """Build a single-Bezier-span B-spline surface of the given bidegree.

        ``create_bilinear`` only reaches bidegree ``(1, 1)``, which cannot bulge in the
        interior of an edge.

        Args:
            control_points (npt.NDArray[np.float64]): Shape
                ``(degree_u + 1, degree_v + 1, 3)``.
            degree_u (int): Degree in the first parametric direction.
            degree_v (int): Degree in the second.

        Returns:
            Bspline: A clamped, non-rational surface on ``[0, 1]^2``.
        """
        spaces = [
            BsplineSpace1D(np.array([0.0] * (d + 1) + [1.0] * (d + 1)), d)
            for d in (degree_u, degree_v)
        ]
        return Bspline(BsplineSpace(spaces), np.asarray(control_points, dtype=np.float64))

    def _cube_with_quadratic_w0(self, bulge: float) -> _FacePairs:
        """Build unit-cube faces with ``face_w0`` a bidegree-``(2, 1)`` patch.

        Its middle row of control points is the degree elevation of the straight ``v = 0``
        and ``v = 1`` edges displaced by *bulge* at ``v = 1``.  At ``bulge = 0`` the patch
        is the same square ``create_bilinear`` gives, so the six faces stay consistent
        while the shared u-edges now meet at different degrees.  A non-zero *bulge* moves
        only the **interior** of the edge shared with ``face_v1``: all eight corners still
        agree, so nothing short of an edge comparison can see it.

        Args:
            bulge (float): Out-of-plane displacement of the middle control point at ``v = 1``.

        Returns:
            _FacePairs: Six faces, consistent iff ``bulge == 0``.
        """
        faces = [create_bilinear(a) for a in self._cube_face_corners()]
        w0 = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.5, 0.0, 0.0], [0.5, 1.0, bulge]],
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            ]
        )
        faces[4] = self._bezier_patch(w0, 2, 1)
        return self._pairs(faces)

    def _cube_with_a_far_interior_point(self, reach: float, corner_lift: float) -> _FacePairs:
        """Build unit-cube faces whose shared ``v = 1, w = 0`` edge bulges *consistently*.

        ``face_v1`` and ``face_w0`` both become bidegree ``(2, 1)`` carrying the **same**
        interior control point ``(0.5, 1, reach)`` on the edge they share, so the faces stay
        mutually consistent however large *reach* is.  What it changes is the *scale* the edge
        comparison derives: that scale is the largest coordinate over all twelve compared
        edges, so a distant-but-legitimate control point widens the edge tolerance for every
        other edge too.  The eight corners are untouched by it and stay ``O(1)``.

        Args:
            reach (float): Out-of-plane position of the shared interior control point.
            corner_lift (float): Displacement along *z* of ``face_w0``'s ``(u=1, v=0)``
                corner, which the other two faces meeting there do not receive.

        Returns:
            _FacePairs: Six faces, consistent iff ``corner_lift == 0``.
        """
        faces = [create_bilinear(a) for a in self._cube_face_corners()]
        w0 = np.array(  # indexed (u, v)
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.5, 0.0, 0.0], [0.5, 1.0, reach]],
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            ]
        )
        v1 = np.array(  # indexed (u, w); its w = 0 column repeats w0's interior point
            [
                [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
                [[0.5, 1.0, reach], [0.5, 1.0, 1.0]],
                [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            ]
        )
        w0[2][0][2] += corner_lift
        faces[3] = self._bezier_patch(v1, 2, 1)
        faces[4] = self._bezier_patch(w0, 2, 1)
        return self._pairs(faces)

    # -- the reproduction -------------------------------------------------------------

    def test_the_consistent_cube_is_accepted_and_interpolates_all_six_faces(self) -> None:
        """The ``dw = 0`` control of the reproduction must be unaffected.

        It also pins what nothing else in this file checked: **all six** faces are
        interpolated, not just ``face_u0``.  The construction is exact here, so the
        residual is graded against the conservative tier at unit scale.
        """
        faces = self._cube()
        gaps = _face_interpolation_gaps(create_coons_volume(faces), faces)
        assert set(gaps) == set(_VOLUME_FACE_LABELS)
        assert max(gaps.values()) <= get_conservative(np.float64), gaps

    def test_a_w_face_corner_off_the_plane_is_refused(self) -> None:
        """The reproduction of pantr issue 301: ``face_w0``'s ``(u=1, v=1)`` corner lifted.

        Accepted before the fix, and the volume then missed ``face_u1`` and ``face_v1``
        by the whole ``0.5``.
        """
        faces = self._cube_with_lifted_corner(face=4, corner=(1, 1), lift=0.5)
        with pytest.raises(ValueError, match=r"Corner \(1,1,0\) mismatch") as excinfo:
            create_coons_volume(faces)
        message = str(excinfo.value)
        assert "face_w0" in message, message
        gap, tol, scale = _reported_numbers(message)
        assert gap == pytest.approx(0.5)
        assert scale == pytest.approx(1.0)
        assert tol == pytest.approx(get_conservative(np.float64), rel=1e-3)

    def test_a_u_face_corner_disagreement_is_refused(self) -> None:
        """The same corner, displaced on ``face_u1`` instead.

        The edges came from the u-faces, so this direction of the error was the one the
        construction could in principle have noticed; it did not, because it never
        compared anything.
        """
        faces = self._cube_with_lifted_corner(face=1, corner=(1, 0), lift=0.5)
        with pytest.raises(ValueError, match=r"Corner \(1,1,0\) mismatch") as excinfo:
            create_coons_volume(faces)
        assert "face_u1" in str(excinfo.value)

    def test_a_v_face_corner_disagreement_is_refused(self) -> None:
        """The same corner again, displaced on ``face_v1``."""
        faces = self._cube_with_lifted_corner(face=3, corner=(1, 0), lift=0.5)
        with pytest.raises(ValueError, match=r"Corner \(1,1,0\) mismatch") as excinfo:
            create_coons_volume(faces)
        assert "face_v1" in str(excinfo.value)

    # -- disagreement away from the corners -------------------------------------------

    def test_an_edge_interior_disagreement_is_refused(self) -> None:
        """Two faces sharing an edge must agree along the whole edge, not just its ends.

        All eight corners agree here and the volume still misses ``face_v1`` by ``0.25``,
        so the corner comparison alone is not enough.
        """
        with pytest.raises(ValueError, match=r"Edge u_v1_w0 mismatch") as excinfo:
            create_coons_volume(self._cube_with_quadratic_w0(0.5))
        message = str(excinfo.value)
        assert "face_v1" in message, message
        assert "face_w0" in message, message

    def test_an_edge_interior_disagreement_is_graded_at_the_edge_tolerance(self) -> None:
        """The edge check's own tolerance must be the boundary, not merely large enough.

        The corner check has a scale sweep of its own; this is the edge check's counterpart,
        on a defect that moves only an edge's interior so no corner comparison can reach it.
        A quarter of the tolerance is accepted and four times it is refused, which pins the
        magnitude rather than only the gross-mismatch behaviour the sibling tests exercise.
        """
        tol = float(get_conservative(np.float64))

        faces = self._cube_with_quadratic_w0(0.25 * tol)
        gaps = _face_interpolation_gaps(create_coons_volume(faces), faces)
        assert max(gaps.values()) <= tol, gaps

        with pytest.raises(ValueError, match=r"Edge u_v1_w0 mismatch"):
            create_coons_volume(self._cube_with_quadratic_w0(4.0 * tol))

    def test_a_corner_defect_survives_an_edge_scale_widened_by_a_distant_control_point(
        self,
    ) -> None:
        """The corner check is not redundant: it grades at a scale the edge check cannot.

        Both checks derive their tolerance from the largest coordinate over everything they
        grade, but they grade different populations.  A legitimate control point far from the
        model's corners widens the *edge* tolerance without touching the *corner* one, so a
        small genuine corner defect can sit below the edge tolerance and above the corner one.

        Here a consistent interior point at ``z = 1e6`` on the edge ``face_v1`` and
        ``face_w0`` share takes the edge tolerance to about ``9e-7`` while the corners stay at
        ``O(1)``, so the corner tolerance stays at about ``9e-13``.  A ``1e-8`` corner
        disagreement then falls between the two.  Verified by construction: with the corner
        comparison removed, this input is accepted and the resulting volume misses a face.

        The same defect at ``reach = 1`` is caught by *either* check, which is why the rest of
        the suite passes with the corner comparison deleted and why this case is needed.
        """
        assert create_coons_volume(self._cube_with_a_far_interior_point(1e6, 0.0)) is not None

        # At unit scale either check reaches the defect, so only that it is refused matters.
        with pytest.raises(ValueError):
            create_coons_volume(self._cube_with_a_far_interior_point(1.0, 1e-8))

        # The discriminating case: the edge tolerance is now ~1e-6, so only the corner
        # comparison can still see a 1e-8 disagreement. Remove it and this input is accepted.
        with pytest.raises(ValueError, match=r"^Corner ") as excinfo:
            create_coons_volume(self._cube_with_a_far_interior_point(1e6, 1e-8))
        assert "face_w0" in str(excinfo.value), str(excinfo.value)

    def test_a_degree_mismatch_along_a_shared_edge_is_accepted(self) -> None:
        """A consistent volume must not be refused for how its faces are represented.

        ``face_w0`` is bidegree ``(2, 1)`` while its neighbours are ``(1, 1)``, so
        ``make_compat`` has to elevate one reading of each shared u-edge and not the other
        before they can be compared.  The measured control-point gap is nevertheless exactly
        zero: the elevation runs on coefficients that already agree, so whatever it rounds it
        rounds identically on both sides.  What this pins is therefore representation
        independence, not the tolerance.
        """
        faces = self._cube_with_quadratic_w0(0.0)
        gaps = _face_interpolation_gaps(create_coons_volume(faces), faces)
        assert max(gaps.values()) <= get_conservative(np.float64), gaps

    def test_a_knot_mismatch_along_a_shared_edge_is_accepted(self) -> None:
        """Refining one face changes its control points but not the geometry.

        The shared edges then carry different knot vectors, which ``make_compat`` merges.
        As with the degree mismatch above, the resulting gap is exactly zero rather than a
        rounding, so this pins representation independence.
        """
        u0, u1, v0, v1, w0, w1 = _flatten_faces(self._cube())
        faces = self._pairs([u0, u1, v0, v1, w0.insert_knots([[0.25, 0.5], [0.5]]), w1])
        gaps = _face_interpolation_gaps(create_coons_volume(faces), faces)
        assert max(gaps.values()) <= get_conservative(np.float64), gaps

    def test_a_generic_hexahedron_interpolates_all_six_faces(self) -> None:
        """A non-planar consistent volume, built from eight arbitrary corners.

        Faces sharing an edge get identical control points there by construction, so the
        set is consistent while no face is planar and no two are parallel.
        """
        p = np.array(
            [
                [[[0.0, 0.0, 0.0], [0.1, -0.2, 3.0]], [[-0.3, 2.0, 0.2], [0.4, 2.3, 2.7]]],
                [[[4.0, 0.3, -0.1], [3.6, 0.2, 3.1]], [[4.2, 1.8, 0.4], [3.9, 2.1, 2.8]]],
            ]
        )
        faces = self._hexahedron(p)
        gaps = _face_interpolation_gaps(create_coons_volume(faces), faces)
        assert max(gaps.values()) <= get_conservative(np.float64) * float(np.abs(p).max()), gaps

    # -- the tolerance is relative, and is the number the message reports --------------

    @pytest.mark.parametrize("scale", [1.0e-6, 1.0e-3, 1.0, 1.0e3, 1.0e6, 1.0e9])
    def test_a_gross_corner_disagreement_is_refused_at_every_model_scale(
        self, scale: float
    ) -> None:
        """A cube whose corner is out by a tenth of its side is wrong at any size."""
        faces = self._cube_with_lifted_corner(face=4, corner=(1, 1), lift=0.1 * scale, scale=scale)
        with pytest.raises(ValueError, match=r"Corner \(1,1,0\) mismatch") as excinfo:
            create_coons_volume(faces)
        gap, tol, reported_scale = _reported_numbers(str(excinfo.value))
        assert gap == pytest.approx(0.1 * scale)
        assert reported_scale == pytest.approx(scale)
        assert tol == pytest.approx(get_conservative(np.float64) * scale, rel=1e-3)

    @pytest.mark.parametrize("scale", [1.0e-6, 1.0, 1.0e9])
    def test_a_corner_disagreement_below_the_tolerance_is_accepted_at_every_scale(
        self, scale: float
    ) -> None:
        """A geometrically sound volume must never be refused for a rounding.

        A quarter of the reported tolerance passes and four times it does not, which ties
        the number in the message to the comparison actually performed.
        """
        tol = get_conservative(np.float64) * scale
        accepted = self._cube_with_lifted_corner(
            face=4, corner=(1, 1), lift=0.25 * tol, scale=scale
        )
        vol = create_coons_volume(accepted)
        assert vol.dim == _RANK_3D
        refused = self._cube_with_lifted_corner(face=4, corner=(1, 1), lift=4.0 * tol, scale=scale)
        with pytest.raises(ValueError, match="mismatch"):
            create_coons_volume(refused)

    def test_both_scales_come_from_all_the_readings_not_the_one_under_test(self) -> None:
        """A short edge next to the origin is graded like the rest of the model.

        This is the 3D counterpart of
        :meth:`TestCoonsCornerToleranceScaleCovariance.test_the_scale_comes_from_all_eight_corners_not_the_pair_under_test`,
        and it pins both new scales at once.  The hexahedron spans ``1e6`` but one of its
        twelve edges runs from the origin to ``(0, 0, 1)``, and one face's reading of that
        edge's far end is displaced by ``1e-9``.  Against the model that is ``1e-15``
        relative, far below the tolerance, and must be accepted; against the edge's own
        magnitude, or against the three readings at that corner, it is ``1e-9`` relative and
        would be refused.  Without this, deriving either scale locally passes every other
        test in the file.
        """
        big, displacement = 1.0e6, 1.0e-9
        p = np.array(
            [
                [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, big, 0.0], [0.0, big, big]]],
                [[[big, 0.0, 0.0], [big, 0.0, big]], [[big, big, 0.0], [big, big, big]]],
            ]
        )
        faces = _flatten_faces(self._hexahedron(p))
        # face_u0 is indexed (v, w), so [0, 1] is its reading of the corner (0, 0, 1)
        corners = np.array(faces[0].control_points)
        corners[0, 1, 0] += displacement
        faces[0] = Bspline(faces[0].space, corners)

        pairs = self._pairs(faces)
        gaps = _face_interpolation_gaps(create_coons_volume(pairs), pairs)
        assert max(gaps.values()) <= get_conservative(np.float64) * big, gaps

    def test_a_face_interior_is_free_provided_the_shared_edges_agree(self) -> None:
        """Only the boundary of a face is shared, so only the boundary has to agree.

        ``face_w0`` is the exact degree elevation of the bilinear face to bidegree
        ``(2, 2)`` with its single interior control point displaced by half the model size.
        Every shared edge is untouched, and all six faces are still interpolated -- which is
        why comparing the twelve edges is the right condition and comparing whole faces
        would be wrong.
        """
        p = np.array(
            [
                [[[0.0, 0.0, 0.0], [0.1, -0.2, 3.0]], [[-0.3, 2.0, 0.2], [0.4, 2.3, 2.7]]],
                [[[4.0, 0.3, -0.1], [3.6, 0.2, 3.1]], [[4.2, 1.8, 0.4], [3.9, 2.1, 2.8]]],
            ]
        )
        faces = _flatten_faces(self._hexahedron(p))
        elevated = faces[4].elevate_degree([1, 1])
        corners = np.array(elevated.control_points)
        corners[1, 1, 2] += 2.0
        faces[4] = Bspline(elevated.space, corners)

        pairs = self._pairs(faces)
        gaps = _face_interpolation_gaps(create_coons_volume(pairs), pairs)
        assert max(gaps.values()) <= get_conservative(np.float64) * float(np.abs(p).max()), gaps

    # -- rational faces ---------------------------------------------------------------

    @staticmethod
    def _weighted(face: Bspline, factor: float) -> Bspline:
        """Return the same rational map with homogeneous control points scaled by *factor*.

        Multiplying homogeneous control points by a constant cancels in the projection, so
        the surface is unchanged whatever *factor* is.  What does not cancel is the seven-term
        Coons formula, which sums homogeneous control points across the terms.

        Args:
            face (Bspline): A non-rational surface.
            factor (float): Scale applied to the homogeneous control points.

        Returns:
            Bspline: A rational surface describing the same map as *face*.
        """
        cp = np.asarray(face.control_points, dtype=np.float64)
        homogeneous = np.concatenate([cp, np.ones((*cp.shape[:-1], 1))], axis=-1) * factor
        return Bspline(face.space, homogeneous, is_rational=True)

    def test_a_rational_face_with_unit_weights_is_accepted(self) -> None:
        """Promoting one face to a NURBS with unit weights changes nothing."""
        faces = _flatten_faces(self._cube())
        original = faces[4]
        faces[4] = self._weighted(original, 1.0)
        grid = np.array([[a, b] for a in _FACE_SAMPLES for b in _FACE_SAMPLES])
        assert_allclose(faces[4].evaluate(grid), original.evaluate(grid), atol=1e-15)

        pairs = self._pairs(faces)
        gaps = _face_interpolation_gaps(create_coons_volume(pairs), pairs)
        assert max(gaps.values()) <= get_conservative(np.float64), gaps

    def test_a_rational_face_whose_weights_disagree_with_its_neighbours_is_refused(self) -> None:
        """The same surface, weighted differently, is still an inconsistency here.

        Scaling ``face_w0``'s homogeneous control points leaves its geometry identical -- the
        assertion below says so -- but the formula combines homogeneous control points, and
        before this check that input produced a volume missing four of the six faces by 17%
        of the model.  It is refused as an edge disagreement, which is what it is.
        """
        faces = _flatten_faces(self._cube())
        original = faces[4]
        faces[4] = self._weighted(original, 2.0)
        grid = np.array([[a, b] for a in _FACE_SAMPLES for b in _FACE_SAMPLES])
        assert_allclose(faces[4].evaluate(grid), original.evaluate(grid), atol=1e-15)

        with pytest.raises(ValueError, match=r"Edge \w+ mismatch") as excinfo:
            create_coons_volume(self._pairs(faces))
        assert "face_w0" in str(excinfo.value)

    # -- the twelve edges are read from the right faces --------------------------------

    def test_the_twelve_edges_are_read_from_the_two_faces_that_carry_them(self) -> None:
        """Pin the edge table itself, since the axis arithmetic that builds it is terse.

        An edge read from the wrong face axis or the wrong side would compare two *different*
        edges, which both accepts real disagreements and rejects sound input.  The expected
        table is derived here from the labelling convention alone: an edge free in one
        direction is fixed in the other two, and is carried by the face pair of each.  The
        assertion on the ``(corner, face)`` set is the one that rules out the original bug's
        blind spot, where corners were only ever read from the u- and v-faces.
        """
        p = np.array(
            [
                [[[0.0, 0.0, 0.0], [0.1, -0.2, 3.0]], [[-0.3, 2.0, 0.2], [0.4, 2.3, 2.7]]],
                [[[4.0, 0.3, -0.1], [3.6, 0.2, 3.1]], [[4.2, 1.8, 0.4], [3.9, 2.1, 2.8]]],
            ]
        )
        expected: dict[str, tuple[str, str, str, str]] = {}
        for a in (0, 1):
            for b in (0, 1):
                expected[f"w_u{a}_v{b}"] = (
                    f"face_u{a}",
                    f"face_v{b}",
                    f"({a},{b},0)",
                    f"({a},{b},1)",
                )
                expected[f"v_u{a}_w{b}"] = (
                    f"face_u{a}",
                    f"face_w{b}",
                    f"({a},0,{b})",
                    f"({a},1,{b})",
                )
                expected[f"u_v{a}_w{b}"] = (
                    f"face_v{a}",
                    f"face_w{b}",
                    f"(0,{a},{b})",
                    f"(1,{a},{b})",
                )

        readings = _extract_edge_pairs(*self._hexahedron(p))
        assert {r.label for r in readings} == set(expected)
        for r in readings:
            assert (r.face_a, r.face_b, *r.corners) == expected[r.label], r.label
            assert r.face_a != r.face_b, r.label
            # both readings really are that edge: their ends are the corners the label names
            for curve in (r.curve_a, r.curve_b):
                for end, corner in enumerate(r.corners):
                    i, j, k = (int(c) for c in corner.strip("()").split(","))
                    assert_allclose(np.asarray(curve.boundary(0, end)), p[i][j][k], atol=0.0)

        pairs_seen = {(c, f) for r in readings for c in r.corners for f in (r.face_a, r.face_b)}
        assert pairs_seen == {
            (f"({i},{j},{k})", f"face_{name}{side}")
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
            for name, side in (("u", i), ("v", j), ("w", k))
        }


class TestCoonsVolumeRationalFaces:
    """Six rational faces blend like six polynomial ones, because the blend is homogeneous.

    ``create_coons_volume`` used to fail on all-rational faces with a NumPy broadcast
    error naming shapes ``(3,)`` and ``(4,)`` (pantr issue 309): the corner array was
    sized from the edges' *homogeneous* coefficients and filled from
    :meth:`~pantr.bspline.Bspline.boundary`, which **projects**.  Every other term of the
    seven-term formula was already built homogeneously, so the fix was to read the corner
    the same way rather than to reopen what the formula means for a NURBS volume.

    What that buys is the boundary property, unchanged: projection is pointwise, so it
    commutes with restriction to a face, and a blend whose homogeneous restriction to
    ``u = 0`` is ``face_u0``'s homogeneous data projects to ``face_u0`` itself.  Its
    hypothesis is that the faces agree **homogeneously**, weights included, which
    ``_verify_edges_3d`` enforces and
    ``TestCoonsVolumeFaceConsistency.test_a_rational_face_whose_weights_disagree_with_its_neighbours_is_refused``
    pins.
    """

    @staticmethod
    def _unit_cube_faces() -> _FacePairs:
        """Build the six polynomial faces of the unit cube, exactly as pantr issue 309 does.

        Returns:
            _FacePairs: The reproduction's six bilinear faces.
        """
        arrays = [
            np.array([[[0, 0, 0], [0, 0, 1]], [[0, 1, 0], [0, 1, 1]]], dtype=float),
            np.array([[[1, 0, 0], [1, 0, 1]], [[1, 1, 0], [1, 1, 1]]], dtype=float),
            np.array([[[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [1, 0, 1]]], dtype=float),
            np.array([[[0, 1, 0], [0, 1, 1]], [[1, 1, 0], [1, 1, 1]]], dtype=float),
            np.array([[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]]], dtype=float),
            np.array([[[0, 0, 1], [0, 1, 1]], [[1, 0, 1], [1, 1, 1]]], dtype=float),
        ]
        u0, u1, v0, v1, w0, w1 = (create_bilinear(a) for a in arrays)
        return ((u0, u1), (v0, v1), (w0, w1))

    @staticmethod
    def _as_rational(face: Bspline, factor: float = 1.0) -> Bspline:
        """Rewrite a polynomial face as a rational one with every weight *factor*.

        Multiplying **all** of a patch's homogeneous coefficients by one constant leaves
        the projected geometry bitwise unchanged, since ``(k w x) / (k w) = x`` exactly for
        the powers of two used here.  At ``factor = 1`` this is the reproduction's wrapper.

        Args:
            face (Bspline): A non-rational face.
            factor (float): Constant scaling of the homogeneous coefficients.
                Defaults to 1.0.

        Returns:
            Bspline: The same geometry, carried rationally.
        """
        cp = np.asarray(face.control_points, dtype=np.float64)
        weights = np.full((*cp.shape[:-1], 1), factor)
        return Bspline(face.space, np.concatenate([cp * factor, weights], axis=-1), True)

    def _rational_cube_faces(self) -> _FacePairs:
        """Wrap the reproduction's six faces as rational with all weights 1.

        Returns:
            _FacePairs: The face set that used to raise a broadcast error.
        """
        return tuple(  # type: ignore[return-value]
            tuple(self._as_rational(f) for f in pair) for pair in self._unit_cube_faces()
        )

    # -- the reproduction -------------------------------------------------------------

    def test_the_polynomial_control_still_interpolates_all_six_faces(self) -> None:
        """The reproduction's control must be unaffected: it built before and must still.

        Nothing about the rational path may disturb it, so this checks the property the
        volume exists for rather than merely that the call returns.
        """
        faces = self._unit_cube_faces()
        gaps = _face_interpolation_gaps(create_coons_volume(faces), faces)
        assert max(gaps.values()) <= get_conservative(np.float64), gaps

    def test_all_rational_faces_reproduce_the_polynomial_control(self) -> None:
        """The reproduction's rational face set builds, and to the same geometry.

        Weights are all 1, so the volume is the polynomial one written rationally and the
        two must agree to roundoff -- an oracle that does not go through the same code.
        """
        polynomial = create_coons_volume(self._unit_cube_faces())
        rational = create_coons_volume(self._rational_cube_faces())

        assert rational.is_rational
        assert_allclose(np.asarray(rational.control_points)[..., -1], 1.0, atol=0.0)

        pts = np.asarray(
            [[u, v, w] for u in (0.0, 0.25, 0.5, 1.0) for v in (0.0, 0.5, 0.9) for w in (0.1, 1.0)]
        )
        assert_allclose(
            np.asarray(rational.evaluate(pts)),
            np.asarray(polynomial.evaluate(pts)),
            atol=get_conservative(np.float64),
            rtol=0.0,
        )

    def test_all_rational_faces_are_interpolated(self) -> None:
        """The rational volume restricts to each of the six rational faces it was built from."""
        faces = self._rational_cube_faces()
        gaps = _face_interpolation_gaps(create_coons_volume(faces), faces)
        assert max(gaps.values()) <= get_conservative(np.float64), gaps

    def test_mixed_faces_take_the_rational_path_rather_than_a_third_one(self) -> None:
        """Some faces rational and some not must behave like all-rational, not like a third case.

        Which faces are rational also decides *where* the old crash was reachable: the
        corners were read off the v-faces alone, so a rational u-face never triggered it
        while a rational v-face did.  Both are checked here, in both directions.
        """
        polynomial = self._unit_cube_faces()
        rational = self._rational_cube_faces()
        pts = np.asarray([[0.2, 0.4, 0.6], [0.0, 1.0, 0.5], [1.0, 0.0, 0.0]])
        want = np.asarray(create_coons_volume(polynomial).evaluate(pts))

        for rational_pair in range(3):
            faces: _FacePairs = tuple(  # type: ignore[assignment]
                rational[i] if i == rational_pair else polynomial[i] for i in range(3)
            )
            volume = create_coons_volume(faces)
            assert volume.is_rational, rational_pair
            assert_allclose(
                np.asarray(volume.evaluate(pts)),
                want,
                atol=get_conservative(np.float64),
                rtol=0.0,
                err_msg=f"rational pair {rational_pair}",
            )
            gaps = _face_interpolation_gaps(volume, faces)
            assert max(gaps.values()) <= get_conservative(np.float64), (rational_pair, gaps)

    # -- non-unit weights, against an oracle built by a different code path -------------

    @staticmethod
    def _quarter_annulus_prism() -> Bspline:
        """Build a quarter-annulus prism as a NURBS volume, without any Coons machinery.

        The body is a ruled surface between two concentric quarter arcs, extruded along
        *z*: degree ``(2, 1, 1)``, genuinely rational, with the arcs' middle weights at
        ``1/sqrt(2)``.  Being degree 1 in *v* and *w* is what makes it an oracle for the
        blend rather than merely an input to it: the Coons operator is
        ``P = P_u + P_v + P_w - P_u P_v - P_u P_w - P_v P_w + P_u P_v P_w``, and a body
        with ``P_v V = P_w V = V`` collapses that sum to ``V`` exactly, so the volume
        rebuilt from its own six faces must be the body itself.

        Returns:
            Bspline: The exact NURBS quarter-annulus prism of inner radius 1, outer
            radius 2 and height 3.
        """
        arcs = (
            create_circle(radius=1.0, angle=np.pi / 2),
            create_circle(radius=2.0, angle=np.pi / 2),
        )
        return create_extrusion(create_ruled(*arcs), [0.0, 0.0, 3.0])

    def test_non_unit_weights_rebuild_an_independently_constructed_volume(self) -> None:
        """A body with real weights, taken apart and blended back, must come back unchanged.

        The oracle is built by ``create_circle``/``create_ruled``/``create_extrusion``,
        which share no code with the Coons blend, and its weights are ``1/sqrt(2)`` rather
        than 1, so unit weights cannot hide a mistake in how the corner term is carried.
        """
        oracle = self._quarter_annulus_prism()
        assert oracle.is_rational
        weights = np.unique(np.asarray(oracle.control_points)[..., -1])
        assert weights.min() < 1.0, weights

        boundaries = [oracle.boundary(axis, side) for axis in range(3) for side in (0, 1)]
        faces: _FacePairs = tuple(  # type: ignore[assignment]
            (boundaries[2 * axis], boundaries[2 * axis + 1]) for axis in range(3)
        )
        volume = create_coons_volume(faces)

        pts = np.asarray(
            [[u, v, w] for u in (0.0, 0.3, 1.0) for v in (0.0, 0.5, 1.0) for w in (0.0, 0.7, 1.0)]
        )
        got = np.asarray(volume.evaluate(pts))
        assert_allclose(got, np.asarray(oracle.evaluate(pts)), atol=1e-14, rtol=0.0)

    def test_non_unit_weights_land_on_the_analytic_cylinder(self) -> None:
        """The rebuilt prism's curved faces must lie on the circles they represent exactly.

        This is the check that does not consult pantr at all: a NURBS quarter arc of
        radius *r* is exact, so every point of the rebuilt inner and outer face must
        satisfy ``x^2 + y^2 = r^2``.  A corner term carried in the wrong space perturbs
        the weight field and shows up here as a radius that drifts off *r*.
        """
        oracle = self._quarter_annulus_prism()
        boundaries = [oracle.boundary(axis, side) for axis in range(3) for side in (0, 1)]
        faces: _FacePairs = tuple(  # type: ignore[assignment]
            (boundaries[2 * axis], boundaries[2 * axis + 1]) for axis in range(3)
        )
        volume = create_coons_volume(faces)

        s = _FACE_SAMPLES
        for radial, radius in ((0.0, 1.0), (1.0, 2.0)):
            pts = np.asarray([[u, radial, w] for u in s for w in s])
            xyz = np.asarray(volume.evaluate(pts))
            got = np.hypot(xyz[:, 0], xyz[:, 1])
            assert_allclose(got, radius, atol=get_conservative(np.float64) * radius, rtol=0.0)

    # -- the weight field the blend produces -------------------------------------------

    @staticmethod
    def _cube_faces_with_interior_weight(interior: float) -> _FacePairs:
        """Build unit-cube faces carrying one interior control point of weight *interior*.

        Each face is the same square, refined to a 3x3 control net by an interior knot at
        0.5 in both directions, with every boundary weight left at 1 so all twelve edges
        still agree.  Only the middle weight is free, which is the point: a face's
        interior is the part the consistency checks do not constrain.

        With every face's interior weight at *t*, the blend's interior control weight is
        ``3t - 3 + 1 = 3t - 2`` -- three ruled terms contributing *t*, three bilinear terms
        and the corner term contributing 1 -- so it crosses zero at ``t = 2/3`` while every
        input weight stays strictly positive.

        Args:
            interior (float): Weight of each face's middle control point.

        Returns:
            _FacePairs: Six mutually consistent faces with strictly positive weights.
        """
        knots = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        space = BsplineSpace([BsplineSpace1D(knots.copy(), 1), BsplineSpace1D(knots.copy(), 1)])
        samples = np.array([0.0, 0.5, 1.0])
        faces = []
        for pair in TestCoonsVolumeRationalFaces._unit_cube_faces():
            for face in pair:
                corners = np.asarray(face.control_points, dtype=np.float64)
                net = np.empty((3, 3, 3))
                for a, u in enumerate(samples):
                    for b, v in enumerate(samples):
                        net[a, b] = (
                            (1 - u) * (1 - v) * corners[0, 0]
                            + u * (1 - v) * corners[1, 0]
                            + (1 - u) * v * corners[0, 1]
                            + u * v * corners[1, 1]
                        )
                weights = np.ones((3, 3, 1))
                weights[1, 1, 0] = interior
                faces.append(Bspline(space, np.concatenate([net * weights, weights], -1), True))
        return ((faces[0], faces[1]), (faces[2], faces[3]), (faces[4], faces[5]))

    def test_positive_face_weights_do_not_imply_a_positive_blend(self) -> None:
        """Strictly positive weights on all six faces can still blend to a negative weight.

        The formula is an inclusion-exclusion and subtracts three of its seven terms, so
        the weight field is not a convex combination of the faces' weights.  At interior
        weight 0.5 every input weight is in ``[0.5, 1]`` and the blend's middle control
        weight is ``3 * 0.5 - 2 = -0.5``; at degree 1 with a knot at 0.5 the middle basis
        function is the only non-zero one at the centre, so the weight field *itself* is
        ``-0.5`` there and the volume genuinely has a pole.
        """
        with pytest.raises(ValueError, match="weight is not certified positive"):
            create_coons_volume(self._cube_faces_with_interior_weight(0.5))

    def test_a_positive_blend_of_varying_weights_is_accepted(self) -> None:
        """The guard must not refuse a weight field that stays positive.

        At interior weight 0.7 the same construction lands on ``3 * 0.7 - 2 = 0.1``: still
        far from the faces' own weights, still accepted, and the volume still interpolates
        all six faces.  Without this, refusing everything rational would pass the test above.
        """
        faces = self._cube_faces_with_interior_weight(0.7)
        volume = create_coons_volume(faces)
        weights = np.asarray(volume.control_points)[..., -1]
        assert_allclose(weights.min(), 0.1, atol=get_conservative(np.float64))
        gaps = _face_interpolation_gaps(volume, faces)
        assert max(gaps.values()) <= get_conservative(np.float64), gaps


class TestCoonsPrecisionAndColumnScale:
    """A comparison is graded at the precision it is made in, against the scale it grades.

    Two independent defects with opposite failure modes lived in the two boundary checks
    (pantr issue 319):

    1. the tier was read at float64 whatever the inputs carried, so a **float32 model was
       refused for a disagreement it cannot express** -- one float32 ulp at magnitude 1 is
       ``1.19e-07`` against a bar of ``9.09e-13``, tighter by ``5.4e5``;
    2. the rational edge comparison took one magnitude over a homogeneous control row,
       mixing coordinates (length times weight) with weights (dimensionless), so a
       **weight error was graded against the model's length** and was accepted once the
       model was large.

    The corrected rules are ``pantr.cad._join._verify_shared_boundary``'s, which carries
    their derivations and measurements.
    """

    # -- mechanism 1: the tier is read at the inputs' own precision ---------------------

    @staticmethod
    def _line(a: list[float], b: list[float], dtype: npt.DTypeLike) -> Bspline:
        """Build the ticket's degree-1 two-point curve in a given precision.

        Args:
            a (list[float]): Start control point.
            b (list[float]): End control point.
            dtype (npt.DTypeLike): Precision of both the knots and the control points.

        Returns:
            Bspline: A clamped linear curve on ``[0, 1]``.
        """
        space = BsplineSpace([BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0], dtype=dtype), 1)])
        return Bspline(space, np.asarray([a, b], dtype=dtype))

    def _unit_square_curves(
        self, dtype: npt.DTypeLike, offset: float
    ) -> tuple[tuple[Bspline, Bspline], tuple[Bspline, Bspline]]:
        """Build the ticket's four curves, with the ``(1,0)`` corner displaced by *offset*.

        Args:
            dtype (npt.DTypeLike): Precision of all four curves.
            offset (float): Displacement along *x* of ``C_v1``'s start, which ``C_u0``'s
                end reads as ``1``.

        Returns:
            tuple: ``((C_v0, C_v1), (C_u0, C_u1))``, as ``create_coons_surface`` takes them.
        """
        return (
            (
                self._line([0, 0, 0], [0, 1, 0], dtype),
                self._line([1 + offset, 0, 0], [1, 1, 0], dtype),
            ),
            (
                self._line([0, 0, 0], [1, 0, 0], dtype),
                self._line([0, 1, 0], [1, 1, 0], dtype),
            ),
        )

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_a_corner_one_ulp_out_is_accepted_in_either_precision(
        self, dtype: npt.DTypeLike
    ) -> None:
        """The ticket's reproduction: as close as the format allows must never be refused.

        This is the whole of mechanism 1.  Both precisions must reach the same verdict,
        because in each the corner is one ulp of *that* format out -- one ulp at magnitude
        1 being the format's machine epsilon by definition -- while the tier is 4096 of
        them.  float64 accepted it and float32 did not, reporting a corner mismatch that
        is not one and was ``5.4e5`` below anything float32 can express.
        """
        curves = self._unit_square_curves(dtype, get_machine_epsilon(dtype))
        surface = create_coons_surface(curves)
        assert np.dtype(surface.control_points.dtype) == np.dtype(dtype)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_a_corner_a_thousand_ulps_out_is_refused_in_either_precision(
        self, dtype: npt.DTypeLike
    ) -> None:
        """Reading the tier at float32 must not amount to accepting everything.

        The tier is ``4096 * eps`` in every format, so the same *relative* mistake is
        refused in both -- what changes is only what "relative" resolves to.  Without
        this, a check that accepted its inputs unconditionally would pass the test above.
        """
        offset = 1.0e5 * get_machine_epsilon(dtype)
        curves = self._unit_square_curves(dtype, offset)
        with pytest.raises(ValueError, match=r"Corner \(1,0\) mismatch") as excinfo:
            create_coons_surface(curves)
        gap, tol, scale = _reported_numbers(str(excinfo.value))
        assert gap == pytest.approx(offset, rel=1e-3)
        assert scale == pytest.approx(1.0 + offset, rel=1e-3)
        assert tol == pytest.approx(get_conservative(dtype) * scale, rel=1e-3)

    @staticmethod
    def _cube_faces(dtype: npt.DTypeLike, scale: float = 1.0) -> _FacePairs:
        """Build the six consistent faces of the cube ``[0, scale]^3`` in a given precision.

        ``create_bilinear`` builds at float64 unconditionally, so a float32 model has to
        be assembled directly.

        Args:
            dtype (npt.DTypeLike): Precision of the knots and control points.
            scale (float): Side of the cube. Defaults to 1.0.

        Returns:
            _FacePairs: Six mutually consistent bilinear faces.
        """
        knots = np.array([0.0, 0.0, 1.0, 1.0], dtype=dtype)
        space = BsplineSpace([BsplineSpace1D(knots.copy(), 1), BsplineSpace1D(knots.copy(), 1)])
        s = scale
        arrays = [
            [[[0, 0, 0], [0, 0, s]], [[0, s, 0], [0, s, s]]],
            [[[s, 0, 0], [s, 0, s]], [[s, s, 0], [s, s, s]]],
            [[[0, 0, 0], [0, 0, s]], [[s, 0, 0], [s, 0, s]]],
            [[[0, s, 0], [0, s, s]], [[s, s, 0], [s, s, s]]],
            [[[0, 0, 0], [0, s, 0]], [[s, 0, 0], [s, s, 0]]],
            [[[0, 0, s], [0, s, s]], [[s, 0, s], [s, s, s]]],
        ]
        f = [Bspline(space, np.asarray(a, dtype=dtype)) for a in arrays]
        return ((f[0], f[1]), (f[2], f[3]), (f[4], f[5]))

    def test_a_float32_model_survives_the_blend_in_float32(self) -> None:
        """Reading the tier at float32 is not enough on its own: the blend must build too.

        Grading a float32 patch correctly only moves the failure downstream if the
        construction still assembles float64 coefficients over a float32 space, which
        ``Bspline`` refuses outright, or mixes precisions inside one ``BsplineSpace``,
        which is refused as well.  Both a patch and a volume are checked, since the two
        hit different builders.
        """
        curves = self._unit_square_curves(np.float32, 0.0)
        surface = create_coons_surface(curves)
        assert np.dtype(surface.control_points.dtype) == np.dtype(np.float32)

        faces = self._cube_faces(np.float32)
        volume = create_coons_volume(faces)
        assert np.dtype(volume.control_points.dtype) == np.dtype(np.float32)

        # ``evaluate`` requires points in the B-spline's own dtype, which is what
        # ``_face_interpolation_gaps`` cannot supply; check ``face_u0`` directly instead.
        s = np.asarray(_FACE_SAMPLES, dtype=np.float32)
        face_params = np.asarray([[v, w] for v in s for w in s], dtype=np.float32)
        volume_params = np.asarray([[0.0, v, w] for v in s for w in s], dtype=np.float32)
        assert_allclose(
            np.asarray(volume.evaluate(volume_params), dtype=np.float64),
            np.asarray(faces[0][0].evaluate(face_params), dtype=np.float64),
            atol=get_conservative(np.float32),
            rtol=0.0,
        )

    # -- mechanism 2: a weight is graded against the weights ----------------------------

    @staticmethod
    def _as_rational(face: Bspline) -> Bspline:
        """Rewrite a polynomial face rationally, with every weight 1.

        Args:
            face (Bspline): A non-rational face.

        Returns:
            Bspline: The same geometry, carried rationally.
        """
        cp = np.asarray(face.control_points, dtype=np.float64)
        weights = np.ones((*cp.shape[:-1], 1))
        return Bspline(face.space, np.concatenate([cp, weights], axis=-1), True)

    def _cube_with_displaced_weight(self, scale: float, gap: float) -> _FacePairs:
        """Build the ticket's cube with the two w-faces rational and one weight out by *gap*.

        Only the w-faces are rational, which is what keeps the reproduction on the
        pre-existing path rather than on the one pantr issue 309 unblocked.  The displaced
        control point is the corner at the origin, so its homogeneous *coordinates* are
        zero however the weight moves: the coordinate columns agree exactly and only the
        weight column can decide the verdict.

        Args:
            scale (float): Side of the cube.
            gap (float): Amount added to one weight, against weights of 1.

        Returns:
            _FacePairs: Six faces whose weights disagree at one corner.
        """
        pairs = self._cube_faces(np.float64, scale)
        faces = _flatten_faces(pairs)
        faces[4], faces[5] = self._as_rational(faces[4]), self._as_rational(faces[5])
        cp = np.asarray(faces[4].control_points).copy()
        cp[0, 0, -1] += gap
        faces[4] = Bspline(faces[4].space, cp, is_rational=True)
        u0, u1, v0, v1, w0, w1 = faces
        return ((u0, u1), (v0, v1), (w0, w1))

    @pytest.mark.parametrize("scale", [1.0, 1.0e2, 1.0e4, 1.0e6, 1.0e8])
    def test_a_weight_error_is_refused_at_every_model_scale(self, scale: float) -> None:
        """The ticket's reproduction: four times the tier against weights of 1, at five sizes.

        A weight is dimensionless, so this is a gross error at any ``scale``.  Graded
        against one magnitude over the homogeneous row it was refused at ``scale = 1`` and
        accepted at every larger size -- the failure the edge check exists to stop,
        reappearing on the rational path once the model is big.
        """
        gap = 4.0 * get_conservative(np.float64)
        with pytest.raises(ValueError, match=r"Edge \w+ mismatch") as excinfo:
            create_coons_volume(self._cube_with_displaced_weight(scale, gap))

        message = str(excinfo.value)
        assert "weight scale" in message, message
        reported_gap, tol, reported_scale = _reported_numbers(message)
        assert reported_gap == pytest.approx(gap)
        assert reported_scale == pytest.approx(1.0)
        assert tol == pytest.approx(get_conservative(np.float64), rel=1e-3)

    @pytest.mark.parametrize("scale", [1.0, 1.0e4, 1.0e8])
    def test_a_weight_within_the_tier_is_accepted_at_every_model_scale(self, scale: float) -> None:
        """The other side of the same rule: a quarter of the tier must pass everywhere.

        Splitting the columns tightens the weight comparison by the model's size, so it
        has to be checked that it did not tighten past what a weight can be stored to.
        """
        gap = 0.25 * get_conservative(np.float64)
        volume = create_coons_volume(self._cube_with_displaced_weight(scale, gap))
        assert volume.is_rational

    @pytest.mark.parametrize("scale", [1.0, 1.0e4, 1.0e8])
    def test_a_one_sided_weight_rescale_is_still_refused_at_every_scale(self, scale: float) -> None:
        """Splitting the groups must not disarm the check the volume gained in pantr issue 307.

        Scaling one face's homogeneous control points by ``mu`` leaves its geometry
        untouched, so only a homogeneous comparison can see it.  Both groups carry ``mu``,
        which is exactly why the split preserves the verdict: the weight gap
        ``(mu - 1) w`` is graded against ``tier * max(w, mu w)`` and is refused.
        """
        pairs = self._cube_faces(np.float64, scale)
        faces = _flatten_faces(pairs)
        rational = self._as_rational(faces[4])
        cp = np.asarray(rational.control_points) * 2.0
        faces[4] = Bspline(rational.space, cp, is_rational=True)

        grid = np.array([[a, b] for a in _FACE_SAMPLES for b in _FACE_SAMPLES])
        assert_allclose(
            np.asarray(faces[4].evaluate(grid)),
            np.asarray(pairs[2][0].evaluate(grid)),
            atol=0.0,
        )

        u0, u1, v0, v1, w0, w1 = faces
        with pytest.raises(ValueError, match=r"Edge \w+ mismatch"):
            create_coons_volume(((u0, u1), (v0, v1), (w0, w1)))

    @pytest.mark.parametrize("scale", [1.0e-6, 1.0, 1.0e8])
    def test_a_consistent_rational_cube_is_accepted_at_every_scale(self, scale: float) -> None:
        """Neither rule may refuse correct input, at any model size.

        The control for both mechanisms: all six faces rational, all weights 1, nothing
        displaced.  It also pins that the coordinate group's scale still spans the model,
        since a cube of side ``1e8`` is graded against ``1e8`` and not against 1.
        """
        pairs = self._cube_faces(np.float64, scale)
        faces = _flatten_faces(pairs)
        rational = [self._as_rational(f) for f in faces]
        volume = create_coons_volume(
            (
                (rational[0], rational[1]),
                (rational[2], rational[3]),
                (rational[4], rational[5]),
            )
        )
        gaps = _face_interpolation_gaps(
            volume,
            (
                (rational[0], rational[1]),
                (rational[2], rational[3]),
                (rational[4], rational[5]),
            ),
        )
        assert max(gaps.values()) <= get_conservative(np.float64) * scale, gaps

    def test_a_coordinate_error_on_a_rational_edge_names_the_coordinate_group(self) -> None:
        """The split must leave the coordinate half doing its job, and say which half decided.

        One control point of ``face_w0`` has its **whole homogeneous row** doubled, so the
        point it projects to is unchanged and the corner comparison, which projects, sees
        nothing.  What is left is an edge disagreement of ``s`` in the coordinate columns
        beside one of 1 in the weight column, and it must be reported against the model's
        length rather than against the weights: the two are not comparable, and reporting
        the wrong one would tell the caller to look at a quantity that is fine.
        """
        scale = 100.0
        pairs = self._cube_faces(np.float64, scale)
        faces = _flatten_faces(pairs)
        faces[4], faces[5] = self._as_rational(faces[4]), self._as_rational(faces[5])
        cp = np.asarray(faces[4].control_points).copy()
        cp[1, 1] *= 2.0
        faces[4] = Bspline(faces[4].space, cp, is_rational=True)

        u0, u1, v0, v1, w0, w1 = faces
        with pytest.raises(ValueError, match=r"Edge \w+ mismatch") as excinfo:
            create_coons_volume(((u0, u1), (v0, v1), (w0, w1)))

        message = str(excinfo.value)
        assert "coordinate scale" in message, message
        gap, tol, reported_scale = _reported_numbers(message)
        assert gap == pytest.approx(scale)
        assert reported_scale == pytest.approx(2.0 * scale)
        assert tol == pytest.approx(get_conservative(np.float64) * 2.0 * scale, rel=1e-3)
