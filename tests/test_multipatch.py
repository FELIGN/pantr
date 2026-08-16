"""Tests for the :mod:`pantr.multipatch` topology and detection code."""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.multipatch import Interface, MultiPatch, detect_interfaces, match_face_cps
from pantr.multipatch._interface import tangential_axes


def _open_space_1d(degree: int, n_interior: int) -> BsplineSpace1D:
    """Build an open 1D space on ``[0, 1]`` with ``n_interior`` single interior knots."""
    interior = [(i + 1) / (n_interior + 1) for i in range(n_interior)]
    return BsplineSpace1D([0.0] * (degree + 1) + interior + [1.0] * (degree + 1), degree)


def _lattice_patch(space: BsplineSpace, offset: npt.ArrayLike | None = None) -> Bspline:
    """Build a patch whose control points form a regular lattice on the unit box.

    The lattice is deliberately anisotropic in index count, so a wrong axis
    correspondence produces a shape mismatch or a coordinate mismatch rather than
    silently working.
    """
    num_basis = space.num_basis
    grids = np.meshgrid(*[np.linspace(0.0, 1.0, n) for n in num_basis], indexing="ij")
    cps = np.stack(grids, axis=-1)
    if offset is not None:
        cps = cps + np.asarray(offset, dtype=cps.dtype)
    return Bspline(space, cps.copy())


def _anisotropic_space_2d() -> BsplineSpace:
    """A 2D space with different basis counts per direction."""
    return BsplineSpace([_open_space_1d(2, 1), _open_space_1d(2, 0)])


def _anisotropic_space_3d() -> BsplineSpace:
    """A 3D space with three different basis counts."""
    return BsplineSpace([_open_space_1d(1, 0), _open_space_1d(2, 0), _open_space_1d(2, 1)])


# ---------------------------------------------------------------- Interface record


def test_interface_is_frozen_and_hashable() -> None:
    """Interfaces are immutable, comparable and usable as dict keys."""
    iface = Interface(patch_a=0, face_a=1, patch_b=1, face_b=0, axis_map=(1,), flips=(False,))

    with pytest.raises(dataclasses.FrozenInstanceError):
        iface.patch_a = 5  # type: ignore[misc]
    assert iface == Interface(0, 1, 1, 0, (1,), (False,))
    assert {iface: "value"}[Interface(0, 1, 1, 0, (1,), (False,))] == "value"
    assert iface.dim == 2


def test_interface_rejects_self_face() -> None:
    """A face cannot be joined to itself."""
    with pytest.raises(ValueError, match="cannot join a face to itself"):
        Interface(patch_a=0, face_a=1, patch_b=0, face_b=1, axis_map=(1,), flips=(False,))


def test_interface_rejects_out_of_range_face() -> None:
    """Face ids must be in range for the dimension implied by ``axis_map``."""
    with pytest.raises(ValueError, match="out of range for dim=2"):
        Interface(patch_a=0, face_a=4, patch_b=1, face_b=0, axis_map=(1,), flips=(False,))


def test_interface_rejects_bad_axis_map() -> None:
    """``axis_map`` must be a permutation of ``face_b``'s tangential axes."""
    with pytest.raises(ValueError, match="must be a permutation"):
        Interface(patch_a=0, face_a=1, patch_b=1, face_b=0, axis_map=(0,), flips=(False,))
    with pytest.raises(ValueError, match="must be a permutation"):
        Interface(patch_a=0, face_a=1, patch_b=1, face_b=0, axis_map=(1, 1), flips=(False, False))


def test_interface_rejects_flips_length_mismatch() -> None:
    """``flips`` must have the same length as ``axis_map``."""
    with pytest.raises(ValueError, match="equal length"):
        Interface(patch_a=0, face_a=1, patch_b=1, face_b=0, axis_map=(1,), flips=())


def test_interface_requires_canonical_order() -> None:
    """The two sides must be given in canonical order."""
    with pytest.raises(ValueError, match="canonical order"):
        Interface(patch_a=1, face_a=0, patch_b=0, face_b=1, axis_map=(1,), flips=(False,))


# ---------------------------------------------------------------- match_face_cps


def _brute_force_match(  # noqa: PLR0913
    space_a: BsplineSpace,
    face_a: int,
    space_b: BsplineSpace,
    face_b: int,
    axis_map: tuple[int, ...],
    flips: tuple[bool, ...],
) -> tuple[list[int], list[int]]:
    """Pair face control points with explicit nested loops, as an independent oracle.

    Deliberately written without any of the vectorized machinery under test: it walks
    the face's tangential index space in C-order and computes both flat ids by hand.
    """
    dim = space_a.dim
    n_a, n_b = space_a.num_basis, space_b.num_basis
    axis_a, side_a = divmod(face_a, 2)
    axis_b, side_b = divmod(face_b, 2)
    tang_a = tangential_axes(face_a, dim)
    slab_a = 0 if side_a == 0 else n_a[axis_a] - 1
    slab_b = 0 if side_b == 0 else n_b[axis_b] - 1

    ids_a: list[int] = []
    ids_b: list[int] = []
    for combo in itertools.product(*[range(n_a[axis]) for axis in tang_a]):
        index_a = [0] * dim
        index_b = [0] * dim
        index_a[axis_a] = slab_a
        index_b[axis_b] = slab_b
        for k, axis in enumerate(tang_a):
            index_a[axis] = combo[k]
            target = axis_map[k]
            index_b[target] = n_b[target] - 1 - combo[k] if flips[k] else combo[k]
        flat_a = 0
        for axis in range(dim):
            flat_a = flat_a * n_a[axis] + index_a[axis]
        flat_b = 0
        for axis in range(dim):
            flat_b = flat_b * n_b[axis] + index_b[axis]
        ids_a.append(flat_a)
        ids_b.append(flat_b)
    return ids_a, ids_b


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_match_face_cps_matches_brute_force(dim: int) -> None:
    """Every face pair and orientation agrees with an independent nested-loop oracle."""
    degrees = [1, 2, 2][:dim]
    interior = [0, 1, 2][:dim]
    space = BsplineSpace([_open_space_1d(d, n) for d, n in zip(degrees, interior, strict=True)])

    for face_a, face_b in itertools.product(range(2 * dim), repeat=2):
        tang_b = tangential_axes(face_b, dim)
        for axis_map in itertools.permutations(tang_b):
            for flips in itertools.product([False, True], repeat=len(tang_b)):
                sizes_a = [space.num_basis[a] for a in tangential_axes(face_a, dim)]
                sizes_b = [space.num_basis[a] for a in axis_map]
                if sizes_a != sizes_b:
                    with pytest.raises(ValueError, match="different numbers of basis functions"):
                        match_face_cps(space, face_a, space, face_b, axis_map, flips)
                    continue
                cps_a, cps_b = match_face_cps(space, face_a, space, face_b, axis_map, flips)
                expected_a, expected_b = _brute_force_match(
                    space, face_a, space, face_b, axis_map, flips
                )
                assert cps_a.tolist() == expected_a
                assert cps_b.tolist() == expected_b


def test_match_face_cps_is_sorted_and_readonly() -> None:
    """``cps_a`` comes back sorted ascending, and both outputs are frozen."""
    space = _anisotropic_space_3d()
    cps_a, cps_b = match_face_cps(space, 1, space, 0, (1, 2), (False, False))

    assert np.all(np.diff(cps_a) > 0)
    assert not cps_a.flags.writeable
    assert not cps_b.flags.writeable
    assert cps_a.shape == cps_b.shape


def test_match_face_cps_validation() -> None:
    """Out-of-range faces, bad permutations and length mismatches are rejected."""
    space = _anisotropic_space_2d()

    with pytest.raises(ValueError, match="face_a=4 out of range"):
        match_face_cps(space, 4, space, 0, (1,), (False,))
    with pytest.raises(ValueError, match="face_b=-1 out of range"):
        match_face_cps(space, 0, space, -1, (1,), (False,))
    with pytest.raises(ValueError, match="must be a permutation"):
        match_face_cps(space, 1, space, 0, (0,), (False,))
    with pytest.raises(ValueError, match="equal length"):
        match_face_cps(space, 1, space, 0, (1,), (False, False))
    with pytest.raises(ValueError, match="same parametric dimension"):
        match_face_cps(space, 1, _anisotropic_space_3d(), 0, (1,), (False,))


# ---------------------------------------------------------------- detection


def test_detect_translated_1d() -> None:
    """Two 1D patches meeting at a point give one interface with empty axis data."""
    space = BsplineSpace([_open_space_1d(2, 1)])
    left = _lattice_patch(space)
    right = _lattice_patch(space, offset=[1.0])

    assert detect_interfaces([left, right]) == (
        Interface(patch_a=0, face_a=1, patch_b=1, face_b=0, axis_map=(), flips=()),
    )


def test_detect_translated_2d() -> None:
    """Side-by-side 2D patches give the interface stated in the specification."""
    space = _anisotropic_space_2d()
    left = _lattice_patch(space)
    right = _lattice_patch(space, offset=[1.0, 0.0])

    assert detect_interfaces([left, right]) == (
        Interface(patch_a=0, face_a=1, patch_b=1, face_b=0, axis_map=(1,), flips=(False,)),
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_detect_translated_3d_all_axes(axis: int) -> None:
    """Translating along each axis exercises all six 3D faces."""
    space = _anisotropic_space_3d()
    first = _lattice_patch(space)
    offset = np.zeros(3)
    offset[axis] = 1.0
    second = _lattice_patch(space, offset=offset)

    expected_tangential = tuple(k for k in range(3) if k != axis)
    assert detect_interfaces([first, second]) == (
        Interface(
            patch_a=0,
            face_a=2 * axis + 1,
            patch_b=1,
            face_b=2 * axis,
            axis_map=expected_tangential,
            flips=(False, False),
        ),
    )


def test_detect_recovers_permuted_axes_2d() -> None:
    """A patch whose axes were swapped is still matched, with the axis map recording it."""
    space = _anisotropic_space_2d()
    first = _lattice_patch(space)
    second = _lattice_patch(space, offset=[1.0, 0.0]).permute_directions([1, 0])
    assert second is not None

    interfaces = detect_interfaces([first, second])

    assert interfaces == (
        Interface(patch_a=0, face_a=1, patch_b=1, face_b=2, axis_map=(0,), flips=(False,)),
    )


def test_detect_recovers_flipped_axis_2d() -> None:
    """Reversing the tangential direction of the neighbour is recorded as a flip."""
    space = _anisotropic_space_2d()
    first = _lattice_patch(space)
    swapped = _lattice_patch(space, offset=[1.0, 0.0]).permute_directions([1, 0])
    assert swapped is not None
    second = swapped.reverse(0)
    assert second is not None

    interfaces = detect_interfaces([first, second])

    assert interfaces == (
        Interface(patch_a=0, face_a=1, patch_b=1, face_b=2, axis_map=(0,), flips=(True,)),
    )


def test_detected_matches_coincide_geometrically() -> None:
    """The control points a detected interface pairs really do coincide in space.

    Guards the whole chain rather than the record alone: a wrong ``axis_map`` or
    ``flips`` would still produce a plausible-looking Interface, but the paired
    coordinates would not agree.
    """
    space = _anisotropic_space_2d()
    first = _lattice_patch(space)
    swapped = _lattice_patch(space, offset=[1.0, 0.0]).permute_directions([1, 0])
    assert swapped is not None
    second = swapped.reverse(0)
    assert second is not None

    (interface,) = detect_interfaces([first, second])
    cps_a, cps_b = match_face_cps(
        first.space,
        interface.face_a,
        second.space,
        interface.face_b,
        interface.axis_map,
        interface.flips,
    )
    coords_a = first.control_points.reshape(-1, first.rank)[cps_a]
    coords_b = second.control_points.reshape(-1, second.rank)[cps_b]

    assert cps_a.size > 1, "a degenerate one-point match would make this vacuous"
    np.testing.assert_allclose(coords_a, coords_b, atol=first.space.tolerance)


def test_detect_corner_only_no_interface() -> None:
    """Patches touching at a single corner share no face."""
    space = _anisotropic_space_2d()
    first = _lattice_patch(space)
    diagonal = _lattice_patch(space, offset=[1.0, 1.0])

    assert detect_interfaces([first, diagonal]) == ()


def test_detect_disjoint_no_interface() -> None:
    """Patches that do not touch share no face."""
    space = _anisotropic_space_2d()
    first = _lattice_patch(space)
    far = _lattice_patch(space, offset=[5.0, 0.0])

    assert detect_interfaces([first, far]) == ()


def test_detect_rejects_refined_face_knots() -> None:
    """Equal geometry but a refined knot vector on the shared face is not conforming."""
    coarse = _anisotropic_space_2d()
    refined = BsplineSpace([_open_space_1d(2, 1), _open_space_1d(2, 1)])
    first = _lattice_patch(coarse)
    second = _lattice_patch(refined, offset=[1.0, 0.0])

    assert detect_interfaces([first, second]) == ()


def test_detect_rejects_degree_mismatch() -> None:
    """A different degree along the shared face is not conforming either."""
    first = _lattice_patch(_anisotropic_space_2d())
    other = BsplineSpace([_open_space_1d(2, 1), _open_space_1d(1, 0)])
    second = _lattice_patch(other, offset=[1.0, 0.0])

    assert detect_interfaces([first, second]) == ()


def _rational_pair(weight_perturbation: float = 0.0) -> tuple[Bspline, Bspline]:
    """Build two rational patches sharing a face, optionally breaking one weight.

    The neighbour's shared column is copied from the first patch, weights included,
    so the pair matches exactly regardless of how homogeneous coordinates are stored.
    """
    space = BsplineSpace([_open_space_1d(2, 0), _open_space_1d(2, 0)])
    num_basis = space.num_basis
    grids = np.meshgrid(*[np.linspace(0.0, 1.0, n) for n in num_basis], indexing="ij")
    weights = 1.0 + 0.5 * grids[1]
    first_cps = np.stack([*grids, weights], axis=-1)

    second_cps = first_cps.copy()
    second_cps[..., 0] += 1.0
    # The shared column: patch b's low face must equal patch a's high face exactly.
    second_cps[0, ...] = first_cps[-1, ...]
    if weight_perturbation:
        second_cps[0, 0, -1] += weight_perturbation

    return (
        Bspline(space, first_cps, is_rational=True),
        Bspline(space, second_cps, is_rational=True),
    )


def test_detect_rational_matching_weights() -> None:
    """Two rational patches whose shared face agrees in weights are matched."""
    first, second = _rational_pair()

    assert detect_interfaces([first, second]) == (
        Interface(patch_a=0, face_a=1, patch_b=1, face_b=0, axis_map=(1,), flips=(False,)),
    )


def test_detect_rational_weight_mismatch_rejected() -> None:
    """Identical coordinates but a differing weight breaks the match."""
    first, second = _rational_pair(weight_perturbation=0.25)

    assert detect_interfaces([first, second]) == ()


def test_detect_self_interface() -> None:
    """A patch whose two opposite faces coincide yields a self-interface."""
    space = BsplineSpace([_open_space_1d(1, 0), _open_space_1d(1, 0)])
    # Degenerate in the first direction: both faces of axis 0 sit on the same edge.
    cps = np.array([[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 1.0]]])
    patch = Bspline(space, cps)

    interfaces = detect_interfaces([patch])

    assert interfaces == (
        Interface(patch_a=0, face_a=0, patch_b=0, face_b=1, axis_map=(1,), flips=(False,)),
    )


def test_detect_l_shape_has_two_interfaces() -> None:
    """Three patches in an L share exactly two faces."""
    space = BsplineSpace([_open_space_1d(2, 0), _open_space_1d(2, 0)])
    corner = _lattice_patch(space)
    right = _lattice_patch(space, offset=[1.0, 0.0])
    above = _lattice_patch(space, offset=[0.0, 1.0])

    interfaces = detect_interfaces([corner, right, above])

    assert len(interfaces) == 2
    assert {(i.patch_a, i.patch_b) for i in interfaces} == {(0, 1), (0, 2)}


def test_detect_scale_invariance() -> None:
    """Detection survives a large uniform rescaling of the geometry.

    The geometric tolerance is scaled by the size of the geometry, so a problem
    measured in different units must give the same topology. An absolute epsilon
    would quietly stop matching here.
    """
    space = _anisotropic_space_2d()
    for scale in (1e-4, 1.0, 1e5):
        first = Bspline(space, _lattice_patch(space).control_points * scale)
        second = Bspline(space, _lattice_patch(space, offset=[1.0, 0.0]).control_points * scale)
        assert detect_interfaces([first, second]) == (
            Interface(patch_a=0, face_a=1, patch_b=1, face_b=0, axis_map=(1,), flips=(False,)),
        ), f"failed at scale {scale}"


def test_detect_translation_invariance() -> None:
    """Detection survives moving the geometry far from the origin.

    Rescaling and translating are different tests, and the joint bounding-box diagonal
    alone tracks only the first. Two unit patches at ``x = 1e6`` keep a diagonal of
    order one, while every coordinate difference is now formed from numbers of order
    ``1e6`` and so carries an absolute error of ``eps * 1e6``; a diagonal-only
    tolerance of ``2.5e-18`` sits eight orders below that noise floor. Taking the
    largest coordinate magnitude alongside the diagonal is what fixes it.

    The second patch is displaced by **one ulp of the coordinate magnitude**, which is
    what makes this test discriminating rather than decorative. Two patches built by
    the same expression have bitwise equal face coordinates and match under any
    tolerance at all, including a hopeless one; two patches that reached the same face
    by different arithmetic differ by a rounding at their own magnitude, and that is
    the gap the diagonal-only rule refuses to forgive.
    """
    space = _anisotropic_space_2d()
    expected = (Interface(patch_a=0, face_a=1, patch_b=1, face_b=0, axis_map=(1,), flips=(False,)),)
    for shift in (0.0, 1.0e3, 1.0e6, 1.0e9):
        one_ulp = float(np.spacing(shift)) if shift > 0.0 else 0.0
        first = Bspline(space, _lattice_patch(space).control_points + shift)
        second = Bspline(
            space, _lattice_patch(space, offset=[1.0, 0.0]).control_points + shift + one_ulp
        )
        assert detect_interfaces([first, second]) == expected, f"failed at shift {shift}"


def test_detect_keeps_separate_patches_separate_far_from_the_origin() -> None:
    """The looser tolerance far from the origin must not start inventing interfaces.

    The magnitude term multiplies the geometric tolerance by ``1e6`` at this offset,
    so the guard worth having is that a genuine gap is still a gap: these two patches
    are a whole patch-width apart and must stay unmatched.
    """
    space = _anisotropic_space_2d()
    shift = 1.0e6
    first = Bspline(space, _lattice_patch(space).control_points + shift)
    apart = Bspline(space, _lattice_patch(space, offset=[2.0, 0.0]).control_points + shift)
    assert detect_interfaces([first, apart]) == ()


def test_detect_requires_common_dim() -> None:
    """Mixing parametric dimensions is an error, not an empty result."""
    flat = _lattice_patch(_anisotropic_space_2d())
    solid = _lattice_patch(_anisotropic_space_3d())

    with pytest.raises(ValueError, match="must share one parametric dimension"):
        detect_interfaces([flat, solid])


# ---------------------------------------------------------------- MultiPatch


def test_multipatch_detects_by_default() -> None:
    """Interfaces are detected when none are supplied."""
    space = _anisotropic_space_2d()
    mp = MultiPatch([_lattice_patch(space), _lattice_patch(space, offset=[1.0, 0.0])])

    assert len(mp) == 2
    assert mp.dim == 2
    assert mp.interfaces == (
        Interface(patch_a=0, face_a=1, patch_b=1, face_b=0, axis_map=(1,), flips=(False,)),
    )
    assert mp.patches[0].control_points.shape == (*space.num_basis, 2)


def test_multipatch_accepts_valid_supplied_interfaces() -> None:
    """A correct supplied interface is kept as given."""
    space = _anisotropic_space_2d()
    supplied = (Interface(patch_a=0, face_a=1, patch_b=1, face_b=0, axis_map=(1,), flips=(False,)),)
    mp = MultiPatch([_lattice_patch(space), _lattice_patch(space, offset=[1.0, 0.0])], supplied)

    assert mp.interfaces == supplied


def test_multipatch_rejects_false_interface() -> None:
    """A supplied interface that does not hold geometrically is rejected."""
    space = _anisotropic_space_2d()
    wrong = (Interface(patch_a=0, face_a=1, patch_b=1, face_b=0, axis_map=(1,), flips=(True,)),)

    with pytest.raises(ValueError, match="does not hold for the given geometry"):
        MultiPatch([_lattice_patch(space), _lattice_patch(space, offset=[1.0, 0.0])], wrong)


def test_multipatch_rejects_out_of_range_patch_index() -> None:
    """An interface naming a patch that does not exist is rejected."""
    space = _anisotropic_space_2d()
    bad = (Interface(patch_a=0, face_a=1, patch_b=7, face_b=0, axis_map=(1,), flips=(False,)),)

    with pytest.raises(ValueError, match="out of range"):
        MultiPatch([_lattice_patch(space), _lattice_patch(space, offset=[1.0, 0.0])], bad)


def test_multipatch_requires_a_patch() -> None:
    """An empty patch list is an error."""
    with pytest.raises(ValueError, match="at least one patch"):
        MultiPatch([])


def test_multipatch_rejects_mixed_dim() -> None:
    """Patches of different parametric dimension cannot be combined."""
    with pytest.raises(ValueError, match="must share dim"):
        MultiPatch(
            [_lattice_patch(_anisotropic_space_2d()), _lattice_patch(_anisotropic_space_3d())]
        )


def test_multipatch_rejects_mixed_rationality() -> None:
    """A rational and a polynomial patch cannot be combined."""
    rational, _ = _rational_pair()
    polynomial = _lattice_patch(BsplineSpace([_open_space_1d(2, 0), _open_space_1d(2, 0)]))

    with pytest.raises(ValueError, match="must share is_rational"):
        MultiPatch([rational, polynomial])


def test_multipatch_rejects_mixed_rank() -> None:
    """Patches embedded in different physical dimensions cannot be combined."""
    space = BsplineSpace([_open_space_1d(2, 0), _open_space_1d(2, 0)])
    planar = _lattice_patch(space)
    num_basis = space.num_basis
    spatial = Bspline(space, np.zeros((*num_basis, 3)))

    with pytest.raises(ValueError, match="must share rank"):
        MultiPatch([planar, spatial])
