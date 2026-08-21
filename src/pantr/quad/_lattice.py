"""Tensor-product evaluation grids of points (:class:`PointsLattice`)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ..basis import LagrangeVariant


def _frozen_copy(
    pts: npt.NDArray[np.float32 | np.float64],
) -> npt.NDArray[np.float32 | np.float64]:
    """Take a read-only, contiguous copy of one direction's coordinates.

    Args:
        pts (npt.NDArray[np.float32 | np.float64]): The caller's coordinate array.

    Returns:
        npt.NDArray[np.float32 | np.float64]: A copy the caller cannot reach and
            nobody can write to.
    """
    frozen = np.ascontiguousarray(pts).copy()
    frozen.flags.writeable = False
    return frozen


class PointsLattice:
    """A tensor-product grid of evaluation points in multiple dimensions.

    Stores one 1D array of coordinates per spatial direction and provides
    helpers for constructing the full set of grid points or querying grid
    metadata.

    Immutable once constructed: each coordinate array is copied on the way in
    and the stored copies are read-only, so what
    :meth:`_validate_pts_per_dir` checked at construction stays true for the
    object's whole life. :class:`pantr.quad.QuadratureRule` does the same, and
    for the same reason.

    Attributes:
        _pts_per_dir (tuple[npt.NDArray[np.float32 | np.float64], ...]): One
            1D coordinate array per spatial dimension. All arrays share the same
            dtype.
    """

    def __init__(self, pts_per_dir: Iterable[npt.NDArray[np.float32 | np.float64]]) -> None:
        """Initialize the points lattice.

        Args:
            pts_per_dir (Iterable[npt.NDArray[np.float32 | np.float64]]): The points per dimension.
                All points must have the same dtype.

        Raises:
            ValueError: If the dimension is less than 1 or the points have different dtypes.
        """
        # Snapshot, then freeze, in that order. ``tuple()`` copies the container
        # and not the arrays, so without the copy the caller keeps a live handle
        # on the lattice's coordinates, and without the freeze so does anyone
        # who reads :attr:`pts_per_dir`. Validation below would then describe a
        # state the object need not still be in: ``arr.shape = (n, 1)`` reshapes
        # in place and breaks the 1-D invariant with no error raised.
        self._pts_per_dir: tuple[npt.NDArray[np.float32 | np.float64], ...] = tuple(
            _frozen_copy(pts) for pts in pts_per_dir
        )
        self._validate_pts_per_dir()

    def _validate_pts_per_dir(self) -> None:
        """Validate the per-direction coordinate arrays.

        Raises:
            ValueError: If the number of dimensions is less than 1, if arrays
                have differing dtypes, if any array is not 1D, or if any array
                is empty.
        """
        if self.dim < 1:
            raise ValueError("Points lattice must have at least 1 dimension")
        if not all(pts.dtype == self.dtype for pts in self._pts_per_dir):
            raise ValueError("All points must have the same dtype")
        for pts in self._pts_per_dir:
            if pts.ndim != 1:
                raise ValueError("All points must be 1D")
            if pts.shape[0] == 0:
                raise ValueError("All points must have at least 1 point")

    @property
    def dim(self) -> int:
        """Get the dimension of the points lattice.

        Returns:
            int: Number of spatial dimensions.
        """
        return len(self._pts_per_dir)

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the dtype of the points lattice.

        Returns:
            npt.DTypeLike: The numpy floating-point dtype shared by all
            coordinate arrays.
        """
        return self._pts_per_dir[0].dtype

    @property
    def pts_per_dir(self) -> tuple[npt.NDArray[np.float32 | np.float64], ...]:
        """Get the points per dimension.

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], ...]: One 1D coordinate
            array for each spatial dimension, read-only.
        """
        # Fresh read-only views rather than the stored arrays themselves, because
        # ``writeable = False`` stops writes to the data and not changes to the
        # metadata: ``arr.shape = (n, 1)`` reshapes a read-only array in place and
        # would leave the lattice holding a 2-D array while :attr:`dim` still
        # reported 1. A view carries its own shape, so that lands on the caller's
        # copy and not on ours. Measured at about 640 ns for three directions,
        # against evaluations that read this once per direction per call and cost
        # milliseconds.
        return tuple(pts.view() for pts in self._pts_per_dir)

    def get_all_points(
        self, order: Literal["C", "F"] = "C"
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Get all points in the points lattice.

        Args:
            order (Literal["C", "F"]): The order of the points. Defaults to "C".
                "C" means the last index varies fastest, "F" means the first index varies fastest.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The dim-dimensional points
                in the lattice. It has shape: (n_pts, dim).
        """
        tp_coords = np.meshgrid(*self._pts_per_dir, indexing="ij")
        if order == "C":  # Last index varies fastest
            return cast(
                npt.NDArray[np.float32 | np.float64],
                np.array(tp_coords).reshape(self.dim, -1).T,  # (n_pts, dim)
            )
        # order == "F": first index varies fastest.  meshgrid(indexing="xy") only
        # swaps the first two axes, so a Fortran-order ravel of each "ij"
        # coordinate grid is required for dim >= 3.
        return cast(
            npt.NDArray[np.float32 | np.float64],
            np.stack([c.ravel(order="F") for c in tp_coords], axis=-1),  # (n_pts, dim)
        )


def create_lagrange_points_lattice(
    lagrange_variant: LagrangeVariant,
    n_pts_per_dir: Iterable[int],
    dtype: npt.DTypeLike = np.float64,
) -> PointsLattice:
    """Create a Lagrange points lattice for tensor-product evaluation.

    Builds a :class:`PointsLattice` whose per-direction coordinate arrays are
    the Lagrange nodes of the specified variant on ``[0, 1]``.

    Args:
        lagrange_variant (LagrangeVariant): The variant of the Lagrange basis
            (e.g., equispaced, Gauss-Legendre, Gauss-Lobatto-Legendre, etc.).
        n_pts_per_dir (Iterable[int]): Number of points per spatial dimension.
            Each value must be at least 1.
        dtype (npt.DTypeLike): Floating-point dtype for the coordinates.
            Must be float32 or float64. Defaults to np.float64.

    Returns:
        PointsLattice: A lattice whose per-direction coordinate arrays are the
        Lagrange nodes for the given variant and point counts.

    Raises:
        ValueError: If any value in ``n_pts_per_dir`` is less than 1.
    """
    # Lazy import to avoid circular dependency
    from ..basis._basis_lagrange import _get_lagrange_points  # noqa: PLC0415

    if any(n_pts < 1 for n_pts in n_pts_per_dir):
        raise ValueError("All number of points must be at least 1")

    pts_per_dir = tuple(
        _get_lagrange_points(lagrange_variant, n_pts, dtype) for n_pts in n_pts_per_dir
    )
    return PointsLattice(pts_per_dir)
