"""Control-point index arithmetic across a pair of patch faces.

Defines :func:`match_face_cps`, which pairs the flat control-point ids on two
faces under a given axis correspondence. Pure index arithmetic: no geometry is
read, so it says nothing about whether the two faces actually coincide in space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ._interface import face_axis_side, tangential_axes

if TYPE_CHECKING:
    from ..bspline import BsplineSpace


def _face_slab_index(num_basis: tuple[int, ...], face: int) -> int:
    """Return the index along a face's own axis that the face occupies.

    Args:
        num_basis (tuple[int, ...]): Per-direction basis counts of the patch.
        face (int): Face id.

    Returns:
        int: ``0`` for a low face, ``num_basis[axis] - 1`` for a high one.
    """
    axis, side = face_axis_side(face)
    return 0 if side == 0 else num_basis[axis] - 1


def _validate_face(space: BsplineSpace, face: int, name: str) -> None:
    """Check that a face id is in range for a space.

    Args:
        space (BsplineSpace): The patch space.
        face (int): Face id.
        name (str): Argument name, for the error message.

    Raises:
        ValueError: If the face id is out of range.
    """
    if not 0 <= face < 2 * space.dim:
        raise ValueError(
            f"{name}={face} out of range for a {space.dim}D space; "
            f"expected 0 <= {name} < {2 * space.dim}"
        )


def face_index_grids(space: BsplineSpace, face: int) -> tuple[npt.NDArray[np.int64], ...]:
    """Build the per-tangential-axis index grids of a face, in C-order.

    Args:
        space (BsplineSpace): The patch space.
        face (int): Face id.

    Returns:
        tuple[npt.NDArray[np.int64], ...]: One array per tangential axis, each of
        the face's tangential shape, holding that axis's index. Empty for a 1D
        patch, whose faces are single points.
    """
    num_basis = space.num_basis
    tangential = tangential_axes(face, space.dim)
    ranges = [np.arange(num_basis[axis], dtype=np.int64) for axis in tangential]
    if not ranges:
        return ()
    return tuple(np.meshgrid(*ranges, indexing="ij"))


def match_face_cps(  # noqa: PLR0913
    space_a: BsplineSpace,
    face_a: int,
    space_b: BsplineSpace,
    face_b: int,
    axis_map: tuple[int, ...],
    flips: tuple[bool, ...],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Pair the flat control-point ids of two faces under an axis correspondence.

    Flat ids are C-order over :attr:`~pantr.bspline.BsplineSpace.num_basis`. The
    ``k``-th tangential axis of ``face_a``, in increasing-axis order, is taken to
    correspond to patch axis ``axis_map[k]`` of ``space_b``, running backwards
    when ``flips[k]``.

    This is index arithmetic only. It does not check that the two faces coincide
    geometrically; :func:`~pantr.multipatch.detect_interfaces` is what establishes
    that.

    Args:
        space_a (BsplineSpace): First patch space.
        face_a (int): Face id on ``space_a``.
        space_b (BsplineSpace): Second patch space.
        face_b (int): Face id on ``space_b``.
        axis_map (tuple[int, ...]): Patch-``b`` axis per tangential axis of
            ``face_a``; a permutation of ``face_b``'s tangential axes.
        flips (tuple[bool, ...]): Whether each of those axis pairs runs opposed.

    Returns:
        tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ``(cps_a, cps_b)``,
        equal-length read-only arrays of flat ids. ``cps_a`` is sorted ascending
        and ``cps_b[i]`` is the id matched to ``cps_a[i]``.

    Raises:
        ValueError: If a face id is out of range, the two spaces differ in
            dimension, ``axis_map`` is not a permutation of ``face_b``'s
            tangential axes, ``flips`` has a different length, or the matched
            tangential directions hold different numbers of basis functions.

    Example:
        >>> import numpy as np
        >>> from pantr.bspline import BsplineSpace, BsplineSpace1D
        >>> space = BsplineSpace([BsplineSpace1D([0, 0, 1, 1], 1)] * 2)
        >>> a, b = match_face_cps(space, 1, space, 0, (1,), (False,))
        >>> a.tolist(), b.tolist()
        ([2, 3], [0, 1])
    """
    if space_a.dim != space_b.dim:
        raise ValueError(
            f"Both spaces must have the same parametric dimension; "
            f"got {space_a.dim} and {space_b.dim}"
        )
    _validate_face(space_a, face_a, "face_a")
    _validate_face(space_b, face_b, "face_b")

    dim = space_a.dim
    tang_a = tangential_axes(face_a, dim)
    expected = set(tangential_axes(face_b, dim))
    if len(flips) != len(axis_map):
        raise ValueError(
            f"flips and axis_map must have equal length; got {len(flips)} and {len(axis_map)}"
        )
    if len(axis_map) != dim - 1:
        raise ValueError(f"axis_map must have length {dim - 1} for a {dim}D space")
    if set(axis_map) != expected or len(set(axis_map)) != len(axis_map):
        raise ValueError(
            f"axis_map={axis_map} must be a permutation of the tangential axes "
            f"{tuple(sorted(expected))} of face_b={face_b}"
        )

    n_a, n_b = space_a.num_basis, space_b.num_basis
    for k, axis_a in enumerate(tang_a):
        if n_a[axis_a] != n_b[axis_map[k]]:
            raise ValueError(
                f"Matched tangential directions hold different numbers of basis functions: "
                f"axis {axis_a} of space_a has {n_a[axis_a]}, "
                f"axis {axis_map[k]} of space_b has {n_b[axis_map[k]]}"
            )

    axis_a_own, _ = face_axis_side(face_a)
    axis_b_own, _ = face_axis_side(face_b)
    grids = face_index_grids(space_a, face_a)

    idx_a: list[npt.NDArray[np.int64] | int] = [0] * dim
    idx_b: list[npt.NDArray[np.int64] | int] = [0] * dim
    idx_a[axis_a_own] = _face_slab_index(n_a, face_a)
    idx_b[axis_b_own] = _face_slab_index(n_b, face_b)
    for k, axis_a in enumerate(tang_a):
        axis_b = axis_map[k]
        idx_a[axis_a] = grids[k]
        idx_b[axis_b] = (n_b[axis_b] - 1 - grids[k]) if flips[k] else grids[k]

    cps_a = np.ravel_multi_index(tuple(idx_a), n_a).ravel().astype(np.int64)
    cps_b = np.ravel_multi_index(tuple(idx_b), n_b).ravel().astype(np.int64)
    cps_a.flags.writeable = False
    cps_b.flags.writeable = False
    return cps_a, cps_b
