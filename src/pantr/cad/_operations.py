"""Constructive operations: extrusion, revolution, ruled, and sweep.

Provides functions that create higher-dimensional B-spline objects
by combining existing ones: extrusion along a vector, revolution
around an axis, ruled interpolation, and translational sweep.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from ..bspline import Bspline, BsplineSpace
from ..transform import AffineTransform
from ._compat import make_compat
from ._primitives import _linear_space_1d, create_circle
from ._validation import _PHYSICAL_DIM, _pad_to_3d, _promote_to_rational

_MAX_DIM_FOR_OPERATIONS = 2
_NORM_TOL = 1e-14


def create_extrusion(bspline: Bspline, displacement: npt.ArrayLike) -> Bspline:
    """Extrude a B-spline curve or surface along a displacement vector.

    Creates a new B-spline with one additional parametric dimension by
    translating the input along the given vector.  The new direction is
    appended as the last parametric axis with degree 1 and knots
    ``[0, 0, 1, 1]``.

    Args:
        bspline: Input curve (dim=1) or surface (dim=2).
        displacement: Translation vector (up to 3D, zero-padded).

    Returns:
        Bspline: A B-spline with ``dim + 1`` parametric dimensions.

    Raises:
        ValueError: If ``bspline.dim > 2``.

    Example:
        >>> from pantr.cad import create_circle, create_extrusion
        >>> cyl = create_extrusion(create_circle(), [0, 0, 1])
        >>> cyl.dim
        2
    """
    if bspline.dim > _MAX_DIM_FOR_OPERATIONS:
        raise ValueError(
            f"create_extrusion requires dim <= {_MAX_DIM_FOR_OPERATIONS}, got {bspline.dim}."
        )

    disp = _pad_to_3d(displacement)
    cp = bspline.control_points
    orig_shape = cp.shape[:-1]  # (*num_basis,)
    rank_full = cp.shape[-1]

    new_cp = np.empty((*orig_shape, 2, rank_full), dtype=cp.dtype)
    new_cp[..., 0, :] = cp

    if bspline.is_rational:
        # Weighted homogeneous: (w*x, w*y, w*z, w)
        # Translated: (w*x + w*dx, w*y + w*dy, w*z + w*dz, w)
        new_cp[..., 1, :] = cp
        weights = cp[..., _PHYSICAL_DIM : _PHYSICAL_DIM + 1]
        new_cp[..., 1, :_PHYSICAL_DIM] = cp[..., :_PHYSICAL_DIM] + weights * disp
    else:
        new_cp[..., 1, :] = cp + disp[:rank_full].astype(cp.dtype)

    spaces = [*bspline.space.spaces, _linear_space_1d(dtype=bspline.dtype)]
    new_space = BsplineSpace(spaces)
    return Bspline(new_space, new_cp, is_rational=bspline.is_rational)


def create_ruled(bspline1: Bspline, bspline2: Bspline) -> Bspline:
    """Construct a ruled surface or volume between two B-splines.

    Creates a new B-spline by linearly interpolating control points
    between *bspline1* (at parameter 0) and *bspline2* (at parameter 1)
    along a new last parametric axis with degree 1.

    The two inputs are first made compatible via :func:`make_compat` so they
    share the same degree and knot vectors.  If one is rational and the
    other is not, the non-rational one is promoted.

    Args:
        bspline1: First boundary (curve or surface, dim <= 2).
        bspline2: Second boundary (same dim as *bspline1*).

    Returns:
        Bspline: A B-spline with ``dim + 1`` parametric dimensions.

    Raises:
        ValueError: If the inputs have different parametric dimensions.
        ValueError: If either input has ``dim > 2``.

    Example:
        >>> from pantr.cad import create_circle, create_ruled
        >>> annulus = create_ruled(create_circle(radius=0.5), create_circle(radius=1.0))
        >>> annulus.dim
        2
    """
    if bspline1.dim != bspline2.dim:
        raise ValueError(
            f"Both B-splines must have the same dim, got {bspline1.dim} and {bspline2.dim}."
        )
    if bspline1.dim > _MAX_DIM_FOR_OPERATIONS:
        raise ValueError(f"ruled requires dim <= {_MAX_DIM_FOR_OPERATIONS}, got {bspline1.dim}.")

    b1, b2 = make_compat(bspline1, bspline2)

    # Promote to rational if needed
    is_rational = b1.is_rational or b2.is_rational
    if is_rational:
        b1 = _promote_to_rational(b1)
        b2 = _promote_to_rational(b2)

    cp1 = b1.control_points
    cp2 = b2.control_points
    rank_full = cp1.shape[-1]

    new_cp = np.empty((*cp1.shape[:-1], 2, rank_full), dtype=cp1.dtype)
    new_cp[..., 0, :] = cp1
    new_cp[..., 1, :] = cp2

    spaces = [*b1.space.spaces, _linear_space_1d(dtype=b1.dtype)]
    new_space = BsplineSpace(spaces)
    return Bspline(new_space, new_cp, is_rational=is_rational)


def _normalize_axis_vector(axis: int | npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert an axis specification to a unit 3D vector.

    Args:
        axis: An ``int`` (coordinate axis) or array-like direction.

    Returns:
        npt.NDArray[np.float64]: Unit vector of shape ``(3,)``.

    Raises:
        ValueError: If the vector is zero.
    """
    if isinstance(axis, int | np.integer):
        v = np.zeros(_PHYSICAL_DIM, dtype=np.float64)
        v[int(axis)] = 1.0
        return v

    v = np.zeros(_PHYSICAL_DIM, dtype=np.float64)
    arr = np.asarray(axis, dtype=np.float64).ravel()
    v[: arr.size] = arr
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Rotation axis must be non-zero.")
    v /= norm
    return v


def _build_axis_alignment_transform(
    pt: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
) -> AffineTransform:
    """Build a transform that translates *pt* to origin and aligns *v* with Z.

    Args:
        pt: Point on the rotation axis (shape ``(3,)``).
        v: Unit rotation axis vector (shape ``(3,)``).

    Returns:
        AffineTransform: The combined translate-then-rotate transform.
    """
    z_hat = np.array([0.0, 0.0, 1.0])
    n = np.cross(v, z_hat)
    gamma = float(np.arccos(np.clip(v[2], -1.0, 1.0)))

    t_translate = AffineTransform.translation(-pt)
    if np.linalg.norm(n) > _NORM_TOL:
        t_rotate = AffineTransform.rotation_3d(gamma, axis=n)
        return t_rotate @ t_translate
    elif v[2] < 0:
        # v = -z, flip via rotation by pi around x
        t_rotate = AffineTransform.rotation_3d(np.pi, axis=0)
        return t_rotate @ t_translate
    else:
        return t_translate


def _revolve_control_points(
    cw: npt.NDArray[np.float64],
    arc: Bspline,
) -> npt.NDArray[np.float64]:
    """Revolve weighted homogeneous control points around the Z axis.

    For each control point in *cw*, builds a per-point transform
    ``M = Rz(theta) * Tz(z) * Sxy(rho)`` and applies it to the arc
    control points, multiplying by the original weight.

    Args:
        cw: Control points in Z-aligned frame, shape ``(*num_basis, 4)``.
        arc: Circular arc B-spline.

    Returns:
        npt.NDArray[np.float64]: Revolved control points of shape
        ``(*num_basis, n_arc, 4)``.
    """
    aw = arc.control_points
    nrb_shape = cw.shape[:-1]
    n_arc = aw.shape[0]
    qw = np.empty((*nrb_shape, n_arc, _PHYSICAL_DIM + 1), dtype=np.float64)

    wx = cw[..., 0]
    wy = cw[..., 1]
    wz = cw[..., 2]
    w = cw[..., _PHYSICAL_DIM]
    rho = np.hypot(wx, wy)
    theta = np.arctan2(wy, wx)
    theta[theta < 0] += 2 * np.pi
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    for idx in np.ndindex(nrb_shape):
        r = float(rho[idx])
        r_cos = r * float(cos_t[idx])
        r_sin = r * float(sin_t[idx])

        m = np.zeros((4, 4), dtype=np.float64)
        m[0, 0] = r_cos
        m[0, 1] = -r_sin
        m[1, 0] = r_sin
        m[1, 1] = r_cos
        m[2, 3] = float(wz[idx])
        m[3, 3] = 1.0

        qi = aw @ m.T
        qi[..., _PHYSICAL_DIM] *= float(w[idx])
        qw[idx] = qi

    return qw


def create_revolution(
    bspline: Bspline,
    point: npt.ArrayLike,
    axis: int | npt.ArrayLike = 2,
    angle: float | tuple[float, float] | None = None,
) -> Bspline:
    """Revolve a B-spline curve or surface around an axis.

    Creates a new B-spline with one additional parametric dimension
    (the angular direction, appended last).  The input is first
    promoted to rational if needed, then transformed to a coordinate
    system aligned with the rotation axis, revolved via the circular
    arc construction, and transformed back.

    The angular direction inherits the same span/continuity structure
    as :func:`create_circle`: one span per 90 degrees, C0 at arc junctions.

    Args:
        bspline: Input curve (dim=1) or surface (dim=2) with rank 3.
        point: A point on the rotation axis (up to 3D, zero-padded).
        axis: Rotation axis.  An ``int`` in ``{0, 1, 2}`` selects
            a coordinate axis.  An array-like of length 3 specifies
            an arbitrary axis direction (normalised internally).
        angle: Sweep specification (same as :func:`create_circle`).

    Returns:
        Bspline: A rational B-spline with ``dim + 1`` dimensions.

    Raises:
        ValueError: If ``bspline.dim > 2``.
        ValueError: If ``bspline.rank != 3``.
    """
    if bspline.dim > _MAX_DIM_FOR_OPERATIONS:
        raise ValueError(
            f"create_revolution requires dim <= {_MAX_DIM_FOR_OPERATIONS}, got {bspline.dim}."
        )
    if bspline.rank != _PHYSICAL_DIM:
        raise ValueError(f"create_revolution requires rank == {_PHYSICAL_DIM}, got {bspline.rank}.")

    pt = _pad_to_3d(point)
    v = _normalize_axis_vector(axis)
    t = _build_axis_alignment_transform(pt, v)

    nrb = _promote_to_rational(bspline)
    nrb_transformed = nrb.transform(t)
    assert nrb_transformed is not None

    arc = create_circle(angle=angle)
    qw = _revolve_control_points(np.asarray(nrb_transformed.control_points, dtype=np.float64), arc)

    spaces = [*nrb_transformed.space.spaces, *arc.space.spaces]
    new_space = BsplineSpace(spaces)
    result = Bspline(new_space, qw, is_rational=True)
    result_back = result.transform(t.inverse)
    assert result_back is not None
    return result_back


def create_sweep(section: Bspline, trajectory: Bspline) -> Bspline:
    """Construct the translational sweep of a section along a trajectory.

    Creates a new B-spline by summing the section and trajectory
    geometries: ``S(u, v) = section(u) + trajectory(v)``.  The
    trajectory direction is appended as the last parametric axis.

    For rational B-splines the product formula applies: the result
    weight is the product of section and trajectory weights, and the
    weighted coordinates combine as
    ``w_s * C_t + w_t * C_s``.

    Args:
        section: Section curve (dim=1) or surface (dim=2).
        trajectory: Trajectory curve (dim=1).

    Returns:
        Bspline: A B-spline with ``section.dim + 1`` dimensions.

    Raises:
        ValueError: If ``section.dim > 2``.
        ValueError: If ``trajectory.dim != 1``.
    """
    if section.dim > _MAX_DIM_FOR_OPERATIONS:
        raise ValueError(
            f"create_sweep requires section.dim <= {_MAX_DIM_FOR_OPERATIONS}, got {section.dim}."
        )
    if trajectory.dim != 1:
        raise ValueError(f"create_sweep requires trajectory.dim == 1, got {trajectory.dim}.")

    is_rational = section.is_rational or trajectory.is_rational

    if is_rational:
        sec = _promote_to_rational(section)
        traj = _promote_to_rational(trajectory)
        cp_s = sec.control_points
        cp_t = traj.control_points

        ws = cp_s[..., _PHYSICAL_DIM]
        wt = cp_t[..., _PHYSICAL_DIM]
        cs = cp_s[..., :_PHYSICAL_DIM]
        ct = cp_t[..., :_PHYSICAL_DIM]

        # Weighted coords: w_t * C_s + w_s * C_t
        term_s = cs[..., np.newaxis, :] * wt[..., np.newaxis]
        term_t = ct * ws[..., np.newaxis, np.newaxis]
        new_coords = term_s + term_t
        new_weights = ws[..., np.newaxis] * wt

        new_cp = np.concatenate(
            [new_coords, new_weights[..., np.newaxis]],
            axis=-1,
        )
        spaces = [*sec.space.spaces, *traj.space.spaces]
    else:
        cp_s = section.control_points
        cp_t = trajectory.control_points
        new_cp = cp_s[..., np.newaxis, :] + cp_t
        spaces = [*section.space.spaces, *trajectory.space.spaces]

    new_space = BsplineSpace(spaces)
    return Bspline(new_space, new_cp, is_rational=is_rational)
