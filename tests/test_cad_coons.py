"""Tests for Coons surface and volume constructions."""

from __future__ import annotations

import re

import numpy as np
import pytest
from numpy import typing as npt
from numpy.testing import assert_allclose

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.cad import create_bilinear, create_coons_surface, create_coons_volume, create_line
from pantr.tolerance import get_conservative

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

    @staticmethod
    def _reported_numbers(message: str) -> tuple[float, float, float]:
        """Parse the gap, tolerance and scale a mismatch message reports.

        Whatever the message says about the tolerance is a claim about the comparison
        that was actually made, so it is worth checking rather than trusting.

        Args:
            message (str): The ``ValueError`` message.

        Returns:
            tuple[float, float, float]: ``(gap, tol, scale)`` as printed.
        """
        num = r"([-+0-9.eE]+)"
        found = re.search(rf"gap {num}, above {num} at [a-z-]+ scale {num}", message)
        assert found is not None, f"unparseable mismatch message: {message!r}"
        gap, tol, scale = (float(g) for g in found.groups())
        return gap, tol, scale

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
        gap, tol, scale = self._reported_numbers(message)
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

    def test_a_degree_mismatch_along_a_shared_edge_is_accepted(self) -> None:
        """A consistent volume must not be refused for how its faces are represented.

        ``face_w0`` is bidegree ``(2, 1)`` while its neighbours are ``(1, 1)``, so
        ``make_compat`` has to elevate before the edges can be compared, and the check
        has to absorb the roundoff of doing so.
        """
        faces = self._cube_with_quadratic_w0(0.0)
        gaps = _face_interpolation_gaps(create_coons_volume(faces), faces)
        assert max(gaps.values()) <= get_conservative(np.float64), gaps

    def test_a_knot_mismatch_along_a_shared_edge_is_accepted(self) -> None:
        """Refining one face changes its control points but not the geometry.

        The shared edges then carry different knot vectors, which ``make_compat`` merges;
        the check must still find them equal.
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
        faces = self._pairs(
            [
                create_bilinear(p[0]),
                create_bilinear(p[1]),
                create_bilinear(p[:, 0]),
                create_bilinear(p[:, 1]),
                create_bilinear(p[:, :, 0]),
                create_bilinear(p[:, :, 1]),
            ]
        )
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
        gap, tol, reported_scale = self._reported_numbers(str(excinfo.value))
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
