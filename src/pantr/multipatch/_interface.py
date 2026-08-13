"""Interface record between two B-spline patch faces.

Defines :class:`Interface`, the immutable description of one conforming contact
between two patch faces: which faces meet, how their tangential axes correspond,
and which of those axes run in opposite directions.
"""

from __future__ import annotations

import dataclasses


def face_axis_side(face: int) -> tuple[int, int]:
    """Split a face id into its parametric axis and side.

    Faces are numbered ``2 * axis + side`` with ``side == 0`` the low-parameter
    face and ``side == 1`` the high one, matching
    :meth:`pantr.grid.Grid.local_facet_axis_side`.

    Args:
        face (int): Face id.

    Returns:
        tuple[int, int]: ``(axis, side)``.
    """
    return divmod(face, 2)


def tangential_axes(face: int, dim: int) -> tuple[int, ...]:
    """Return the patch axes tangential to a face, in increasing order.

    Args:
        face (int): Face id.
        dim (int): Parametric dimension of the patch.

    Returns:
        tuple[int, ...]: The ``dim - 1`` axes other than the face's own axis.
    """
    axis, _ = face_axis_side(face)
    return tuple(k for k in range(dim) if k != axis)


@dataclasses.dataclass(frozen=True)
class Interface:
    """A conforming contact between two B-spline patch faces.

    Frozen, so instances are hashable and usable as dictionary keys, and ordered
    canonically: ``(patch_a, face_a)`` is lexicographically smaller than
    ``(patch_b, face_b)``. A patch may meet itself on two different faces, which
    is how a closed ring is expressed.

    ``axis_map`` and ``flips`` describe the reparametrization between the two
    faces. Entry ``k`` refers to the ``k``-th tangential axis of ``face_a`` taken
    in increasing-axis order; ``axis_map[k]`` is the *patch axis of* ``patch_b``
    it corresponds to, and ``flips[k]`` says whether the two run in opposite
    directions.

    Attributes:
        patch_a (int): Index of the first patch.
        face_a (int): Face id on ``patch_a``, as ``2 * axis + side``.
        patch_b (int): Index of the second patch.
        face_b (int): Face id on ``patch_b``.
        axis_map (tuple[int, ...]): Length ``dim - 1``; the patch-``b`` axis
            corresponding to each tangential axis of ``face_a``. A permutation of
            the tangential axes of ``face_b``.
        flips (tuple[bool, ...]): Length ``dim - 1``; ``True`` where that
            tangential axis of ``face_a`` runs against its partner's direction.
    """

    patch_a: int
    face_a: int
    patch_b: int
    face_b: int
    axis_map: tuple[int, ...]
    flips: tuple[bool, ...]

    def __post_init__(self) -> None:
        """Validate the face ids, the axis permutation, and the canonical order.

        Raises:
            ValueError: If the interface joins a face to itself, a face id is out
                of range for the implied dimension, ``axis_map`` is not a
                permutation of ``face_b``'s tangential axes, ``flips`` has a
                different length, or the two sides are not in canonical order.
        """
        dim = len(self.axis_map) + 1
        if len(self.flips) != len(self.axis_map):
            raise ValueError(
                f"flips and axis_map must have equal length; "
                f"got {len(self.flips)} and {len(self.axis_map)}"
            )
        for name, face in (("face_a", self.face_a), ("face_b", self.face_b)):
            if not 0 <= face < 2 * dim:
                raise ValueError(
                    f"{name}={face} out of range for dim={dim}; expected 0 <= face < {2 * dim}"
                )
        if self.patch_a == self.patch_b and self.face_a == self.face_b:
            raise ValueError(
                f"An interface cannot join a face to itself "
                f"(patch {self.patch_a}, face {self.face_a})"
            )
        expected = set(tangential_axes(self.face_b, dim))
        if set(self.axis_map) != expected or len(set(self.axis_map)) != len(self.axis_map):
            raise ValueError(
                f"axis_map={self.axis_map} must be a permutation of the tangential axes "
                f"{tuple(sorted(expected))} of face_b={self.face_b}"
            )
        if (self.patch_a, self.face_a) >= (self.patch_b, self.face_b):
            raise ValueError(
                f"Interface sides must be in canonical order: "
                f"({self.patch_a}, {self.face_a}) must be lexicographically smaller than "
                f"({self.patch_b}, {self.face_b})"
            )

    @property
    def dim(self) -> int:
        """Get the parametric dimension of the patches this interface joins.

        Returns:
            int: ``len(axis_map) + 1``.
        """
        return len(self.axis_map) + 1
