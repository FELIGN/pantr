"""Detection of conforming interfaces among a set of B-spline patches.

Defines :func:`detect_interfaces`, which finds every pair of patch faces that
coincide geometrically *and* carry compatible tangential knot vectors, together
with the axis correspondence relating them.

Only conforming interfaces are recognized. Two faces that occupy the same region
of space but disagree in degree or interior knots are rejected rather than
approximated: accepting a nested refinement on one side needs a projection the
caller has to opt into, and silently pairing them would produce control-point
matches that are simply wrong.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt

from ..tolerance import get_strict
from ._interface import Interface, face_axis_side, tangential_axes
from ._match import match_face_cps

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..bspline import Bspline


class _Face(NamedTuple):
    """One side of a candidate interface: a patch and one of its faces.

    Attributes:
        patch (Bspline): The patch.
        face (int): Face id on that patch.
    """

    patch: Bspline
    face: int


class _Orientation(NamedTuple):
    """A candidate correspondence between the tangential axes of two faces.

    Attributes:
        axis_map (tuple[int, ...]): Patch-``b`` axis per tangential axis of face a.
        flips (tuple[bool, ...]): Whether each of those axis pairs runs opposed.
    """

    axis_map: tuple[int, ...]
    flips: tuple[bool, ...]


class _Tolerances(NamedTuple):
    """The three tolerances an interface check needs, kept dimensionally separate.

    Comparing all three quantities against one number is the classic silent error
    here: a patch measured in millimetres would accept weight differences a metre
    apart, and a knot vector normalized to ``[0, 1]`` has no length scale at all.

    Attributes:
        geometric (float): For control-point coordinates. The relative tolerance
            scaled by the joint bounding-box diagonal, so it tracks problem size.
        parametric (float): For knot vectors already normalized to ``[0, 1]``.
            Dimensionless, so it is the bare relative tolerance.
        weight (float): For rational weights. Scaled by the largest weight in play
            rather than by any length.
    """

    geometric: float
    parametric: float
    weight: float


def _joint_tolerances(patch_a: Bspline, patch_b: Bspline, tol: float | None) -> _Tolerances:
    """Derive the geometric, parametric and weight tolerances for a patch pair.

    Args:
        patch_a (Bspline): First patch.
        patch_b (Bspline): Second patch.
        tol (float | None): Dimensionless relative tolerance override, or ``None``
            for the strict preset of the patches' own dtypes.

    Returns:
        _Tolerances: The three derived absolute tolerances.
    """
    # Not ``space.tolerance``: that is an *absolute* parametric length carrying the
    # patch's own knot magnitude, and none of the three quantities compared here is
    # a parametric coordinate of a single patch. What is wanted is the bare
    # dimensionless factor, which each leg below scales by the magnitude that
    # applies to it. The strict tier is what ``space.tolerance`` was built from
    # before, so this keeps the tier the interface check has always used and only
    # stops it travelling through a quantity that has since acquired units.
    relative = (
        max(get_strict(patch_a.space.dtype), get_strict(patch_b.space.dtype))
        if tol is None
        else float(tol)
    )
    rank = patch_a.rank
    coords = np.concatenate(
        [
            patch_a.control_points.reshape(-1, patch_a.control_points.shape[-1])[:, :rank],
            patch_b.control_points.reshape(-1, patch_b.control_points.shape[-1])[:, :rank],
        ]
    )
    extent = np.asarray(coords.max(axis=0) - coords.min(axis=0), dtype=np.float64)
    diagonal = float(np.linalg.norm(extent))
    # A degenerate joint bounding box (both patches a single point) leaves no scale
    # to borrow, so fall back to the bare relative tolerance rather than to zero.
    geometric = relative * diagonal if diagonal > 0.0 else relative

    weight_scale = 1.0
    if patch_a.is_rational or patch_b.is_rational:
        weights = [
            np.abs(patch.control_points[..., -1]).max()
            for patch in (patch_a, patch_b)
            if patch.is_rational
        ]
        weight_scale = max(1.0, float(max(weights)))
    return _Tolerances(geometric=geometric, parametric=relative, weight=relative * weight_scale)


def _normalized_knots(
    space_1d_knots: npt.NDArray[np.floating], flip: bool
) -> npt.NDArray[np.float64]:
    """Affinely map a knot vector onto ``[0, 1]``, optionally reversing it.

    Args:
        space_1d_knots (npt.NDArray[np.floating]): Knot vector.
        flip (bool): Whether to reverse the parametrization.

    Returns:
        npt.NDArray[np.float64]: The normalized (and possibly reversed) knots,
        ascending in both cases.
    """
    knots = np.asarray(space_1d_knots, dtype=np.float64)
    span = knots[-1] - knots[0]
    normalized = (knots - knots[0]) / span if span > 0.0 else np.zeros_like(knots)
    if flip:
        normalized = 1.0 - normalized[::-1]
    return normalized


def _face_corner_coords(patch: Bspline, face: int) -> npt.NDArray[np.float64]:
    """Return the geometric coordinates of a face's corner control points.

    Args:
        patch (Bspline): The patch.
        face (int): Face id.

    Returns:
        npt.NDArray[np.float64]: Array of shape ``(2 ** (dim - 1), rank)``, in an
        unspecified order -- callers compare corner *sets*.
    """
    axis, side = face_axis_side(face)
    num_basis = patch.space.num_basis
    slab = 0 if side == 0 else num_basis[axis] - 1
    coords = patch.control_points[..., : patch.rank]
    corners = []
    tangential = tangential_axes(face, patch.dim)
    for combo in itertools.product(*[(0, num_basis[k] - 1) for k in tangential]):
        index: list[int] = [0] * patch.dim
        index[axis] = slab
        for k, axis_t in enumerate(tangential):
            index[axis_t] = combo[k]
        corners.append(coords[tuple(index)])
    return np.asarray(corners, dtype=np.float64)


def _corner_sets_match(side_a: _Face, side_b: _Face, tol: float) -> bool:
    """Check whether two faces have the same corner control points, as a set.

    A cheap pre-filter before the full verification: it rejects the vast majority
    of face pairs without touching the interior control points, and it deliberately
    ignores ordering, since the ordering is what the orientation search determines.

    Args:
        side_a (_Face): First side.
        side_b (_Face): Second side.
        tol (float): Absolute geometric tolerance.

    Returns:
        bool: ``True`` if the two corner sets coincide within ``tol``.
    """
    corners_a = _face_corner_coords(side_a.patch, side_a.face)
    corners_b = _face_corner_coords(side_b.patch, side_b.face)
    if corners_a.shape != corners_b.shape:
        return False
    distances = np.linalg.norm(corners_a[:, None, :] - corners_b[None, :, :], axis=-1)
    # Every corner of each face must have a partner on the other; a bijection is not
    # required here because coincident corners are legitimate on degenerate patches.
    return bool((distances.min(axis=1) <= tol).all() and (distances.min(axis=0) <= tol).all())


def _knots_agree(side_a: _Face, side_b: _Face, orientation: _Orientation, tol: float) -> bool:
    """Check that the tangential knot vectors agree under the given correspondence.

    Args:
        side_a (_Face): First side.
        side_b (_Face): Second side.
        orientation (_Orientation): Candidate axis correspondence.
        tol (float): Dimensionless tolerance for knots normalized to ``[0, 1]``.

    Returns:
        bool: ``True`` if every matched direction agrees in degree and in
        normalized knot vector.
    """
    space_a, space_b = side_a.patch.space, side_b.patch.space
    for k, axis_a in enumerate(tangential_axes(side_a.face, space_a.dim)):
        one_d_a = space_a.spaces[axis_a]
        one_d_b = space_b.spaces[orientation.axis_map[k]]
        if one_d_a.degree != one_d_b.degree:
            return False
        knots_a = _normalized_knots(one_d_a.knots, flip=False)
        knots_b = _normalized_knots(one_d_b.knots, flip=orientation.flips[k])
        if knots_a.shape != knots_b.shape:
            return False
        if not bool(np.all(np.abs(knots_a - knots_b) <= tol)):
            return False
    return True


def _cps_agree(
    side_a: _Face, side_b: _Face, orientation: _Orientation, tolerances: _Tolerances
) -> bool:
    """Check that every matched control point pair coincides, weights included.

    The pairing comes from :func:`~pantr.multipatch.match_face_cps`, so this cannot
    drift from the index arithmetic a caller will later use to unify dofs.

    Args:
        side_a (_Face): First side.
        side_b (_Face): Second side.
        orientation (_Orientation): Candidate axis correspondence.
        tolerances (_Tolerances): Geometric and weight tolerances.

    Returns:
        bool: ``True`` if all matched coordinates agree within the geometric
        tolerance and all matched weights within the weight tolerance.
    """
    patch_a, patch_b = side_a.patch, side_b.patch
    try:
        cps_a, cps_b = match_face_cps(
            patch_a.space,
            side_a.face,
            patch_b.space,
            side_b.face,
            orientation.axis_map,
            orientation.flips,
        )
    except ValueError:
        return False

    rank = patch_a.rank
    flat_a = patch_a.control_points.reshape(-1, patch_a.control_points.shape[-1])
    flat_b = patch_b.control_points.reshape(-1, patch_b.control_points.shape[-1])
    if not bool(
        np.all(np.abs(flat_a[cps_a, :rank] - flat_b[cps_b, :rank]) <= tolerances.geometric)
    ):
        return False
    if patch_a.is_rational:
        return bool(np.all(np.abs(flat_a[cps_a, -1] - flat_b[cps_b, -1]) <= tolerances.weight))
    return True


def _orientation_candidates(face_b: int, dim: int) -> list[_Orientation]:
    """Enumerate every axis correspondence a face pair could have.

    There are ``(dim - 1)! * 2 ** (dim - 1)`` of them: 1 in 1D, 2 in 2D, 8 in 3D.
    Enumerating and verifying is what makes detection robust on symmetric faces,
    where the corner correspondence alone does not determine the orientation.

    Args:
        face_b (int): Face id on the second patch.
        dim (int): Parametric dimension.

    Returns:
        list[_Orientation]: Every candidate correspondence, cheapest-first in the
        sense that the identity permutation without flips comes first.
    """
    tangential_b = tangential_axes(face_b, dim)
    return [
        _Orientation(axis_map=permutation, flips=flips)
        for permutation in itertools.permutations(tangential_b)
        for flips in itertools.product([False, True], repeat=len(tangential_b))
    ]


def verify_interface(
    patches: Sequence[Bspline], interface: Interface, *, tol: float | None = None
) -> bool:
    """Check that an interface actually holds for a set of patches.

    Applies the same two conditions detection requires: every matched control point
    pair coincides, and the tangential knot vectors agree after normalization.

    Args:
        patches (Sequence[Bspline]): The patches the interface refers to.
        interface (Interface): The interface to verify.
        tol (float | None): Dimensionless relative tolerance override. Defaults to
            the strict preset of the two patches' dtypes.

    Returns:
        bool: ``True`` if the interface holds.

    Raises:
        IndexError: If the interface refers to a patch index that does not exist.
    """
    side_a = _Face(patches[interface.patch_a], interface.face_a)
    side_b = _Face(patches[interface.patch_b], interface.face_b)
    orientation = _Orientation(interface.axis_map, interface.flips)
    tolerances = _joint_tolerances(side_a.patch, side_b.patch, tol)
    if not _knots_agree(side_a, side_b, orientation, tolerances.parametric):
        return False
    return _cps_agree(side_a, side_b, orientation, tolerances)


def detect_interfaces(
    patches: Sequence[Bspline], *, tol: float | None = None
) -> tuple[Interface, ...]:
    """Find the conforming interfaces among a set of patches.

    Every pair of faces in canonical order is considered, including two faces of the
    same patch, which is how a closed ring is found. A pair becomes an interface
    only when both conditions hold: all matched control points coincide, and the
    matched tangential knot vectors agree after affine normalization to ``[0, 1]``.
    A face pair that occupies the same space but disagrees in degree or interior
    knots is *not* an interface here.

    Args:
        patches (Sequence[Bspline]): The patches to inspect. All must share the same
            parametric dimension.
        tol (float | None): Dimensionless relative tolerance override. Defaults, per
            pair, to the strict preset of the two dtypes; it is scaled by the joint
            bounding-box diagonal for coordinates and used bare for knots.

    Returns:
        tuple[Interface, ...]: The interfaces found, sorted by
        ``(patch_a, face_a, patch_b, face_b)``.

    Raises:
        ValueError: If the patches do not all share the same parametric dimension.

    Example:
        >>> import numpy as np
        >>> from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
        >>> space = BsplineSpace([BsplineSpace1D([0, 0, 1, 1], 1)] * 2)
        >>> left = Bspline(space, np.array([[0.0, 0], [0, 1], [1, 0], [1, 1]]))
        >>> right = Bspline(space, np.array([[1.0, 0], [1, 1], [2, 0], [2, 1]]))
        >>> detect_interfaces([left, right])
        (Interface(patch_a=0, face_a=1, patch_b=1, face_b=0, axis_map=(1,), flips=(False,)),)
    """
    if not patches:
        return ()
    dims = {patch.dim for patch in patches}
    if len(dims) != 1:
        raise ValueError(f"All patches must share one parametric dimension; got {sorted(dims)}")
    dim = dims.pop()

    sides = [(p, f) for p in range(len(patches)) for f in range(2 * dim)]
    found: list[Interface] = []
    for (patch_a, face_a), (patch_b, face_b) in itertools.combinations(sides, 2):
        side_a = _Face(patches[patch_a], face_a)
        side_b = _Face(patches[patch_b], face_b)
        tolerances = _joint_tolerances(side_a.patch, side_b.patch, tol)
        if not _corner_sets_match(side_a, side_b, tolerances.geometric):
            continue
        for orientation in _orientation_candidates(face_b, dim):
            if not _knots_agree(side_a, side_b, orientation, tolerances.parametric):
                continue
            if not _cps_agree(side_a, side_b, orientation, tolerances):
                continue
            found.append(
                Interface(
                    patch_a=patch_a,
                    face_a=face_a,
                    patch_b=patch_b,
                    face_b=face_b,
                    axis_map=tuple(orientation.axis_map),
                    flips=tuple(orientation.flips),
                )
            )
            break
    return tuple(sorted(found, key=lambda i: (i.patch_a, i.face_a, i.patch_b, i.face_b)))
