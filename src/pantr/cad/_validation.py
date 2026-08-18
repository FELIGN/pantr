"""Shared validation and conversion helpers for CAD functions.

Provides utility functions used across the ``cad`` subpackage for
normalizing point arrays, promoting B-splines between rational and
non-rational representations, and grading a comparison of control points
against a tolerance that matches what it compares.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from .._control_points_utils import _append_unit_weight_column

if TYPE_CHECKING:
    from numpy import typing as npt

    from ..bspline import Bspline

_PHYSICAL_DIM = 3
"""int: Default physical dimension for all CAD primitives."""


class _ColumnGroup(NamedTuple):
    """One group of control-point columns that share a physical dimension.

    A rational control point carries homogeneous coordinates, whose units are length times
    weight, beside a dimensionless weight.  Grading both against one magnitude makes the
    verdict depend on where the model sits, so each group is graded against its own.

    Attributes:
        name (str): What the group holds, as the error message names it.
        columns (slice): The columns of a control-point array the group covers.
    """

    name: str
    columns: slice


_EUCLIDEAN_COLUMNS = (_ColumnGroup("coordinate", slice(None)),)
"""tuple[_ColumnGroup, ...]: Column groups of non-rational control points -- coordinates only."""

_HOMOGENEOUS_COLUMNS = (
    _ColumnGroup("coordinate", slice(0, -1)),
    _ColumnGroup("weight", slice(-1, None)),
)
"""tuple[_ColumnGroup, ...]: Column groups of rational control points -- coordinates, weight."""


def _column_groups(*, is_rational: bool) -> tuple[_ColumnGroup, ...]:
    """Split a control-point array's columns into groups that share one physical dimension.

    A relative tolerance's scale has to span exactly one dimension, so the caller grades
    each group against its own magnitude: sending ``x -> lambda x`` scales the coordinates
    and leaves the weights alone, and one magnitude over both would make the verdict on a
    weight drift with the size of the model.

    Args:
        is_rational (bool): Whether the array's trailing column is a weight.

    Returns:
        tuple[_ColumnGroup, ...]: The groups, coordinates first -- data given in the wrong
        place fails there, and it is the more informative half to report when both are out.
    """
    return _HOMOGENEOUS_COLUMNS if is_rational else _EUCLIDEAN_COLUMNS


def _coarsest_dtype(
    *arrays: npt.NDArray[np.float32 | np.float64],
) -> np.dtype[np.floating[Any]]:
    """Return the coarsest floating precision among the arrays about to be compared.

    A tolerance tier read at float64 while the data is float32 refuses agreement the
    inputs cannot express: if the true shared value is ``F``, two stored readings satisfy
    ``|r1 - r2| <= 0.5 (eps1 + eps2) |F|``, a hard floor on any achievable agreement.
    Reading the tier at the coarsest precision present clears that floor; taking the
    *promoted* type instead is backwards by ``eps32 / eps64``, about ``5.4e8``.

    The rule grades by **container** rather than by provenance, so float32-resolution data
    already copied into a float64 array is graded at float64.

    Args:
        *arrays (npt.NDArray[np.float32 | np.float64]): The arrays being compared.  At
            least one is required.

    Returns:
        np.dtype[np.floating[Any]]: The dtype with the largest machine epsilon.
    """
    return max((a.dtype for a in arrays), key=lambda d: float(np.finfo(d).eps))


def _pad_to_3d(point: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert a point array to a 3D float64 vector, zero-padding if shorter.

    Args:
        point: Array-like with 1 to 3 elements.

    Returns:
        npt.NDArray[np.float64]: A 1D array of shape ``(3,)``.

    Raises:
        ValueError: If the input has more than 3 elements.
    """
    p = np.asarray(point, dtype=np.float64).ravel()
    if p.size > _PHYSICAL_DIM:
        raise ValueError(f"Point must have at most {_PHYSICAL_DIM} coordinates, got {p.size}.")
    out = np.zeros(_PHYSICAL_DIM, dtype=np.float64)
    out[: p.size] = p
    return out


def _promote_to_rational(bspline: Bspline) -> Bspline:
    """Add unit weights to a non-rational B-spline, returning a rational one.

    If the B-spline is already rational, it is returned unchanged.

    Args:
        bspline: A B-spline curve, surface, or volume.

    Returns:
        Bspline: A rational B-spline with unit weights appended.
    """
    if bspline.is_rational:
        return bspline

    from ..bspline import Bspline as BsplineCls  # noqa: PLC0415

    new_cp = _append_unit_weight_column(bspline.control_points)
    return BsplineCls(bspline.space, new_cp, is_rational=True)
