"""Affine transformations for geometric objects.

Provides the :class:`AffineTransform` class, which represents an affine map
``T(x) = A @ x + b`` in *n*-dimensional space, together with factory methods
for common transformations (translation, rotation, scaling, mirroring, shear)
and composition operators.

Main exports:

- :class:`AffineTransform` — immutable affine-transformation object.
"""

from __future__ import annotations

import functools

import numpy as np
from numpy import typing as npt


class AffineTransform:
    """An affine transformation T(x) = A x + b in n-dimensional space.

    The transformation is defined by a square matrix ``A`` (the linear part)
    and a translation vector ``b``.  Instances are immutable: every mutation
    returns a new :class:`AffineTransform`.

    Attributes:
        _matrix (npt.NDArray[np.float64]): The ``(n, n)`` linear part.
        _translation (npt.NDArray[np.float64]): The ``(n,)`` translation.
    """

    _matrix: npt.NDArray[np.float64]
    _translation: npt.NDArray[np.float64]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        matrix: npt.ArrayLike,
        translation: npt.ArrayLike | None = None,
    ) -> None:
        """Create an affine transformation from a matrix and translation.

        Args:
            matrix (npt.ArrayLike): The ``(n, n)`` linear part of the
                transformation.  Must be a square 2-D array.
            translation (npt.ArrayLike | None): The ``(n,)`` translation
                vector.  If ``None``, defaults to the zero vector.

        Raises:
            ValueError: If *matrix* is not 2-D or not square.
            ValueError: If *translation* length does not match the matrix
                dimension.
        """
        mat = np.ascontiguousarray(np.asarray(matrix, dtype=np.float64))
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:  # noqa: PLR2004
            raise ValueError(f"matrix must be a square 2-D array, got shape {mat.shape}.")

        n = mat.shape[0]

        tvec: npt.NDArray[np.float64]
        if translation is None:
            tvec = np.zeros(n, dtype=np.float64)
        else:
            tvec = np.ascontiguousarray(np.asarray(translation, dtype=np.float64))
            if tvec.shape != (n,):
                raise ValueError(f"translation must have shape ({n},), got {tvec.shape}.")

        mat.flags.writeable = False
        tvec.flags.writeable = False
        self._matrix = mat
        self._translation = tvec

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Get the spatial dimension of the transformation.

        Returns:
            int: Dimension *n* of the transformation.
        """
        return int(self._matrix.shape[0])

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        """Get the linear part of the transformation.

        Returns:
            npt.NDArray[np.float64]: Read-only ``(n, n)`` matrix.
        """
        return self._matrix

    @property
    def offset(self) -> npt.NDArray[np.float64]:
        """Get the translation (offset) part of the transformation.

        Returns:
            npt.NDArray[np.float64]: Read-only ``(n,)`` vector.
        """
        return self._translation

    @functools.cached_property
    def inverse(self) -> AffineTransform:
        """Get the inverse transformation.

        Computed once and cached; subsequent accesses are free. Safe because
        an :class:`AffineTransform` is immutable.

        Returns:
            AffineTransform: The inverse such that ``T @ T.inverse`` is the
            identity.

        Raises:
            ValueError: If the matrix is singular.
        """
        try:
            inv_mat = np.linalg.inv(self._matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Cannot invert a singular affine transformation.") from exc
        inv_trans = -inv_mat @ self._translation
        return AffineTransform(inv_mat, inv_trans)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def identity(n: int) -> AffineTransform:
        """Create the identity transformation in *n* dimensions.

        Args:
            n (int): Spatial dimension.

        Returns:
            AffineTransform: The identity map.
        """
        return AffineTransform(np.eye(n))

    @staticmethod
    def translation(offset: npt.ArrayLike) -> AffineTransform:
        """Create a pure translation.

        Args:
            offset (npt.ArrayLike): Translation vector of length *n*.

        Returns:
            AffineTransform: A translation by *offset*.
        """
        b = np.asarray(offset, dtype=np.float64).ravel()
        return AffineTransform(np.eye(len(b)), b)

    @staticmethod
    def scaling(
        factors: float | npt.ArrayLike,
        *,
        center: npt.ArrayLike | None = None,
    ) -> AffineTransform:
        """Create a scaling transformation.

        Args:
            factors (float | npt.ArrayLike): If a scalar, isotropic scaling
                is applied and *center* (or a separate call) determines the
                dimension.  If an array, anisotropic scaling along each axis.
            center (npt.ArrayLike | None): Optional center point. If given,
                the scaling is performed about this point rather than the
                origin.

        Returns:
            AffineTransform: The scaling transformation.

        Raises:
            ValueError: If *factors* is a scalar and *center* is ``None``
                (dimension cannot be inferred), if any factor is non-finite or
                zero (singular transform), or if *center* has the wrong shape.
        """
        f = np.asarray(factors, dtype=np.float64)
        if f.ndim == 0:
            # Scalar — need center to know dimension.
            if center is None:
                raise ValueError(
                    "An isotropic scaling factor requires a center or an "
                    "array of per-axis factors so the dimension can be "
                    "inferred."
                )
            fval = float(f)
            if not np.isfinite(fval):
                raise ValueError(f"scaling factors must be finite, got {fval!r}.")
            if fval == 0.0:
                raise ValueError(
                    f"scaling factors must be non-zero (singular transform), got {fval!r}."
                )
            c = np.asarray(center, dtype=np.float64).ravel()
            f = np.full(len(c), fval)
        else:
            f = f.ravel()
            if not np.all(np.isfinite(f)):
                raise ValueError(f"scaling factors must be finite, got {f!r}.")
            if np.any(f == 0.0):
                raise ValueError(
                    f"scaling factors must be non-zero (singular transform), got {f!r}."
                )
            if center is not None:
                c = np.asarray(center, dtype=np.float64).ravel()
                if c.shape != (len(f),):
                    raise ValueError(f"center must have shape ({len(f)},), got {c.shape}.")

        mat = np.diag(f)
        t = AffineTransform(mat)
        if center is not None:
            t = _apply_center(t, center)
        return t

    @staticmethod
    def rotation_2d(
        angle: float,
        *,
        center: npt.ArrayLike | None = None,
    ) -> AffineTransform:
        """Create a 2-D counter-clockwise rotation.

        Args:
            angle (float): Rotation angle in radians.
            center (npt.ArrayLike | None): Optional center of rotation.

        Returns:
            AffineTransform: The 2-D rotation.
        """
        c, s = np.cos(angle), np.sin(angle)
        mat = np.array([[c, -s], [s, c]], dtype=np.float64)
        t = AffineTransform(mat)
        if center is not None:
            t = _apply_center(t, center)
        return t

    @staticmethod
    def rotation_3d(
        angle: float,
        axis: int | npt.ArrayLike = 2,
        *,
        center: npt.ArrayLike | None = None,
    ) -> AffineTransform:
        """Create a 3-D rotation via the Rodrigues formula.

        Args:
            angle (float): Rotation angle in radians.
            axis (int | npt.ArrayLike): Rotation axis. An ``int`` in
                ``{0, 1, 2}`` selects the corresponding coordinate axis
                (x, y, z).  An array-like of length 3 specifies an arbitrary
                axis (will be normalised internally).
            center (npt.ArrayLike | None): Optional center of rotation.

        Returns:
            AffineTransform: The 3-D rotation.

        Raises:
            ValueError: If an integer axis is not in ``{0, 1, 2}``.
            ValueError: If a vector axis has zero norm.
        """
        if isinstance(axis, int | np.integer):
            axis_int = int(axis)
            if axis_int not in (0, 1, 2):
                raise ValueError(f"Integer axis must be 0, 1, or 2, got {axis_int}.")
            u = np.zeros(3, dtype=np.float64)
            u[axis_int] = 1.0
        else:
            u = np.asarray(axis, dtype=np.float64).ravel()
            if u.shape != (3,):
                raise ValueError(f"Rotation axis must have shape (3,), got {u.shape}.")
            norm = float(np.linalg.norm(u))
            if norm == 0.0 or not np.isfinite(norm):
                raise ValueError(f"Rotation axis must be a finite non-zero vector, got {u!r}.")
            u = u / norm

        # Rodrigues rotation matrix: R = I cos(t) + (1-cos(t)) u u^T + sin(t) [u]x
        c, s = np.cos(angle), np.sin(angle)
        ux, uy, uz = u
        K = np.array(
            [[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]],
            dtype=np.float64,
        )
        mat = c * np.eye(3) + (1.0 - c) * np.outer(u, u) + s * K

        t = AffineTransform(mat)
        if center is not None:
            t = _apply_center(t, center)
        return t

    @staticmethod
    def mirror(
        normal: npt.ArrayLike,
        *,
        center: npt.ArrayLike | None = None,
    ) -> AffineTransform:
        """Create a reflection (mirror) across a hyperplane.

        The hyperplane passes through the origin (or *center*) and has the
        given *normal* vector.  The Householder formula is used:
        ``A = I - 2 n nᵀ``.

        Args:
            normal (npt.ArrayLike): Normal vector of the mirror plane.  Will
                be normalised internally.
            center (npt.ArrayLike | None): Optional point on the mirror
                plane.

        Returns:
            AffineTransform: The reflection.

        Raises:
            ValueError: If *normal* has zero norm.
        """
        n = np.asarray(normal, dtype=np.float64).ravel()
        norm = float(np.linalg.norm(n))
        if norm == 0.0 or not np.isfinite(norm):
            raise ValueError(f"Mirror normal must be a finite non-zero vector, got {n!r}.")
        n = n / norm
        mat = np.eye(len(n)) - 2.0 * np.outer(n, n)
        t = AffineTransform(mat)
        if center is not None:
            t = _apply_center(t, center)
        return t

    @staticmethod
    def shear(
        dim: int,
        component: int,
        direction: int,
        factor: float,
    ) -> AffineTransform:
        """Create a shear transformation.

        The resulting map adds ``factor * x[direction]`` to
        ``x[component]``, leaving all other components unchanged.

        Args:
            dim (int): Spatial dimension.
            component (int): The axis that is modified.
            direction (int): The axis whose value drives the shear.
            factor (float): Shear magnitude.

        Returns:
            AffineTransform: The shear transformation.

        Raises:
            ValueError: If *component* equals *direction*.
            ValueError: If *component* or *direction* is out of range.
        """
        if component == direction:
            raise ValueError("component and direction must differ.")
        if not (0 <= component < dim):
            raise ValueError(f"component must be in [0, {dim}), got {component}.")
        if not (0 <= direction < dim):
            raise ValueError(f"direction must be in [0, {dim}), got {direction}.")
        mat = np.eye(dim, dtype=np.float64)
        mat[component, direction] = float(factor)
        return AffineTransform(mat)

    # ------------------------------------------------------------------
    # Composition and application
    # ------------------------------------------------------------------

    def compose(self, other: AffineTransform) -> AffineTransform:
        """Compose this transformation with *other*.

        Returns the transformation ``self(other(x))``.

        Args:
            other (AffineTransform): The inner transformation.

        Returns:
            AffineTransform: The composed transformation.

        Raises:
            ValueError: If the dimensions do not match.
        """
        if self.dim != other.dim:
            raise ValueError(
                f"Cannot compose transforms of different dimensions ({self.dim} and {other.dim})."
            )
        new_mat = self._matrix @ other._matrix
        new_trans = self._matrix @ other._translation + self._translation
        return AffineTransform(new_mat, new_trans)

    def __matmul__(self, other: object) -> AffineTransform:
        """Compose via the ``@`` operator.

        Args:
            other (object): Must be an :class:`AffineTransform`.

        Returns:
            AffineTransform: The composed transformation (``self`` after
            ``other``).
        """
        if not isinstance(other, AffineTransform):
            return NotImplemented
        return self.compose(other)

    def __call__(
        self,
        points: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Apply the transformation to a set of points.

        Args:
            points (npt.ArrayLike): Points with shape ``(..., n)``.

        Returns:
            npt.NDArray[np.float64]: Transformed points with the same shape.

        Raises:
            ValueError: If the last dimension of *points* does not match
                ``self.dim``.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[-1] != self.dim:
            raise ValueError(
                f"Points last dimension ({pts.shape[-1]}) must match "
                f"transform dimension ({self.dim})."
            )
        return np.asarray(pts @ self._matrix.T + self._translation, dtype=np.float64)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            str: Representation showing dimension and matrix/translation.
        """
        return (
            f"AffineTransform(dim={self.dim}, "
            f"matrix={self._matrix.tolist()}, "
            f"translation={self._translation.tolist()})"
        )


# ------------------------------------------------------------------
# Module-private helpers
# ------------------------------------------------------------------


def _apply_center(
    transform: AffineTransform,
    center: npt.ArrayLike,
) -> AffineTransform:
    """Conjugate *transform* by a translation to/from *center*.

    Computes ``translate(center) @ transform @ translate(-center)`` so that
    the linear part of *transform* is applied about *center* rather than the
    origin.

    Args:
        transform (AffineTransform): A linear (or affine) transformation.
        center (npt.ArrayLike): The center point.

    Returns:
        AffineTransform: The re-centred transformation.

    Raises:
        ValueError: If ``center`` does not have shape ``(transform.dim,)``.
    """
    c = np.asarray(center, dtype=np.float64).ravel()
    if c.shape != (transform.dim,):
        raise ValueError(
            f"center must have shape ({transform.dim},), got {np.asarray(center).shape}."
        )
    t_neg = AffineTransform.translation(-c)
    t_pos = AffineTransform.translation(c)
    return t_pos @ transform @ t_neg
