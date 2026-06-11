"""Tensor-product to VTK Bézier point reordering.

Maps the natural tensor-product (row-major) control point layout used by pantr
to the point ordering expected by VTK's higher-order Bézier cell types
(``VTK_BEZIER_CURVE``, ``VTK_BEZIER_QUADRILATERAL``, ``VTK_BEZIER_HEXAHEDRON``).

VTK ordering convention: **corners → edges → faces → interior**, where each
sub-entity is itself ordered recursively.

All functions in this module are **pure NumPy** — no pyvista dependency — so
they can be tested without installing pyvista.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy import typing as npt


def _tp_flat_index(indices: tuple[int, ...], shape: tuple[int, ...]) -> int:
    """Convert a multi-index into a flat (row-major) index.

    Args:
        indices: Multi-index tuple, one per dimension.
        shape: Shape of the tensor-product array.

    Returns:
        int: Flat row-major index.

    Note:
        No input validation is performed.
    """
    flat = 0
    stride = 1
    for i in range(len(shape) - 1, -1, -1):
        flat += indices[i] * stride
        stride *= shape[i]
    return flat


@lru_cache(maxsize=64)
def vtk_ordering_curve(degree: int) -> npt.NDArray[np.intp]:
    """Return permutation from tensor-product to VTK ordering for a Bézier curve.

    VTK curve ordering: ``[start, end, interior_0, interior_1, ...]``
    i.e. the two endpoints first, then interior points in order.

    Args:
        degree: Polynomial degree (≥ 1).

    Returns:
        NDArray[intp]: Permutation array of length ``degree + 1``.
    """
    n = degree + 1
    if degree <= 1:
        return np.arange(n, dtype=np.intp)
    # Corners: 0, degree
    order = [0, degree]
    # Interior: 1, 2, ..., degree-1
    order.extend(range(1, degree))
    return np.array(order, dtype=np.intp)


@lru_cache(maxsize=64)
def vtk_ordering_quad(degree_u: int, degree_v: int) -> npt.NDArray[np.intp]:
    """Return permutation from tensor-product to VTK ordering for a Bézier quad.

    Tensor-product control points have shape ``(degree_u + 1, degree_v + 1)``
    and are stored row-major (u varies slowest).

    VTK quad ordering:
    1. 4 corners (VTK vertex order)
    2. 4 edges (interior points only, VTK edge order)
    3. Face interior (row-major within the interior)

    VTK quad corner mapping (tensor-product ``(i, j)`` indices)::

        VTK vertex 0 → (0, 0)
        VTK vertex 1 → (degree_u, 0)
        VTK vertex 2 → (degree_u, degree_v)
        VTK vertex 3 → (0, degree_v)

    VTK quad edge ordering (interior nodes; per
    ``vtkHigherOrderQuadrilateral::PointIndexFromIJK`` every run is in
    **increasing** index order, including the top edge -- the high-order node
    layout does not follow the linear cell's winding direction)::

        Edge 0: j=0        — increasing i
        Edge 1: i=degree_u — increasing j
        Edge 2: j=degree_v — increasing i
        Edge 3: i=0        — increasing j

    The face interior runs ``i`` fastest (``(i-1) + (degree_u-1)*(j-1)``).

    Args:
        degree_u: Polynomial degree in u direction (≥ 1).
        degree_v: Polynomial degree in v direction (≥ 1).

    Returns:
        NDArray[intp]: Permutation array of length ``(degree_u+1) * (degree_v+1)``.
    """
    nu = degree_u + 1
    nv = degree_v + 1
    shape = (nu, nv)
    order: list[int] = []

    # --- Corners (4) ---
    corners = [(0, 0), (degree_u, 0), (degree_u, degree_v), (0, degree_v)]
    for c in corners:
        order.append(_tp_flat_index(c, shape))

    # --- Edges (interior points only) ---
    # Edge 0: bottom (j=0), i = 1..degree_u-1
    for i in range(1, degree_u):
        order.append(_tp_flat_index((i, 0), shape))
    # Edge 1: right (i=degree_u), j = 1..degree_v-1
    for j in range(1, degree_v):
        order.append(_tp_flat_index((degree_u, j), shape))
    # Edge 2: top (j=degree_v), i = 1..degree_u-1 (increasing, per VTK)
    for i in range(1, degree_u):
        order.append(_tp_flat_index((i, degree_v), shape))
    # Edge 3: left (i=0), j = 1..degree_v-1
    for j in range(1, degree_v):
        order.append(_tp_flat_index((0, j), shape))

    # --- Face interior (i fastest, per VTK) ---
    for j in range(1, degree_v):
        for i in range(1, degree_u):
            order.append(_tp_flat_index((i, j), shape))

    return np.array(order, dtype=np.intp)


@lru_cache(maxsize=64)
def vtk_ordering_hex(  # noqa: PLR0912
    degree_u: int, degree_v: int, degree_w: int
) -> npt.NDArray[np.intp]:
    """Return permutation from tensor-product to VTK ordering for a Bézier hex.

    Tensor-product control points have shape
    ``(degree_u + 1, degree_v + 1, degree_w + 1)`` stored row-major
    (u varies slowest).

    VTK hexahedron ordering:
    1. 8 corners
    2. 12 edges (interior points only)
    3. 6 faces (interior points only)
    4. Volume interior

    VTK hex corner mapping (tensor-product ``(i, j, k)`` indices)::

        VTK vertex 0 → (0, 0, 0)
        VTK vertex 1 → (degree_u, 0, 0)
        VTK vertex 2 → (degree_u, degree_v, 0)
        VTK vertex 3 → (0, degree_v, 0)
        VTK vertex 4 → (0, 0, degree_w)
        VTK vertex 5 → (degree_u, 0, degree_w)
        VTK vertex 6 → (degree_u, degree_v, degree_w)
        VTK vertex 7 → (0, degree_v, degree_w)

    VTK hex edge ordering (12 edges)::

        Edge  0: vtx 0→1 — bottom face, j=0, k=0, increasing i
        Edge  1: vtx 1→2 — bottom face, i=pu, k=0, increasing j
        Edge  2: vtx 3→2 — bottom face, j=pv, k=0, increasing i
        Edge  3: vtx 0→3 — bottom face, i=0, k=0, increasing j
        Edge  4: vtx 4→5 — top face, j=0, k=pw, increasing i
        Edge  5: vtx 5→6 — top face, i=pu, k=pw, increasing j
        Edge  6: vtx 7→6 — top face, j=pv, k=pw, increasing i
        Edge  7: vtx 4→7 — top face, i=0, k=pw, increasing j
        Edge  8: vtx 0→4 — vertical, i=0, j=0, increasing k
        Edge  9: vtx 1→5 — vertical, i=pu, j=0, increasing k
        Edge 10: vtx 2→6 — vertical, i=pu, j=pv, increasing k
        Edge 11: vtx 3→7 — vertical, i=0, j=pv, increasing k

    VTK hex face ordering (6 faces, interior points only; per
    ``vtkHigherOrderHexahedron::PointIndexFromIJK`` the i-normal face pair
    comes first, then j-normal, then k-normal, with the lower face of each
    pair first; the first in-face index varies fastest)::

        Face 0: i=0    (left)    — (j, k) interior, j fastest
        Face 1: i=pu   (right)   — (j, k) interior, j fastest
        Face 2: j=0    (front)   — (i, k) interior, i fastest
        Face 3: j=pv   (back)    — (i, k) interior, i fastest
        Face 4: k=0    (bottom)  — (i, j) interior, i fastest
        Face 5: k=pw   (top)     — (i, j) interior, i fastest

    The volume interior runs ``i`` fastest, then ``j``, then ``k``.

    Args:
        degree_u: Polynomial degree in u direction (≥ 1).
        degree_v: Polynomial degree in v direction (≥ 1).
        degree_w: Polynomial degree in w direction (≥ 1).

    Returns:
        NDArray[intp]: Permutation array of length
            ``(degree_u+1) * (degree_v+1) * (degree_w+1)``.
    """
    pu, pv, pw = degree_u, degree_v, degree_w
    shape = (pu + 1, pv + 1, pw + 1)
    order: list[int] = []

    def flat(i: int, j: int, k: int) -> int:
        return _tp_flat_index((i, j, k), shape)

    # --- 8 Corners ---
    corners = [
        (0, 0, 0),
        (pu, 0, 0),
        (pu, pv, 0),
        (0, pv, 0),
        (0, 0, pw),
        (pu, 0, pw),
        (pu, pv, pw),
        (0, pv, pw),
    ]
    for c in corners:
        order.append(flat(*c))

    # --- 12 Edges (interior points only) ---
    # Edge 0: vtx0→vtx1 (j=0, k=0, increasing i)
    for i in range(1, pu):
        order.append(flat(i, 0, 0))
    # Edge 1: vtx1→vtx2 (i=pu, k=0, increasing j)
    for j in range(1, pv):
        order.append(flat(pu, j, 0))
    # Edge 2: vtx3→vtx2 (j=pv, k=0, increasing i)
    for i in range(1, pu):
        order.append(flat(i, pv, 0))
    # Edge 3: vtx0→vtx3 (i=0, k=0, increasing j)
    for j in range(1, pv):
        order.append(flat(0, j, 0))
    # Edge 4: vtx4→vtx5 (j=0, k=pw, increasing i)
    for i in range(1, pu):
        order.append(flat(i, 0, pw))
    # Edge 5: vtx5→vtx6 (i=pu, k=pw, increasing j)
    for j in range(1, pv):
        order.append(flat(pu, j, pw))
    # Edge 6: vtx7→vtx6 (j=pv, k=pw, increasing i)
    for i in range(1, pu):
        order.append(flat(i, pv, pw))
    # Edge 7: vtx4→vtx7 (i=0, k=pw, increasing j)
    for j in range(1, pv):
        order.append(flat(0, j, pw))
    # Edge 8: vtx0→vtx4 (i=0, j=0, increasing k)
    for k in range(1, pw):
        order.append(flat(0, 0, k))
    # Edge 9: vtx1→vtx5 (i=pu, j=0, increasing k)
    for k in range(1, pw):
        order.append(flat(pu, 0, k))
    # Edge 10: vtx2→vtx6 (i=pu, j=pv, increasing k)
    for k in range(1, pw):
        order.append(flat(pu, pv, k))
    # Edge 11: vtx3→vtx7 (i=0, j=pv, increasing k)
    for k in range(1, pw):
        order.append(flat(0, pv, k))

    # --- 6 Faces (interior points only; i-pair, j-pair, k-pair, per VTK) ---
    # Face 0: i=0 (left), interior (j, k), j fastest
    for k in range(1, pw):
        for j in range(1, pv):
            order.append(flat(0, j, k))
    # Face 1: i=pu (right), interior (j, k), j fastest
    for k in range(1, pw):
        for j in range(1, pv):
            order.append(flat(pu, j, k))
    # Face 2: j=0 (front), interior (i, k), i fastest
    for k in range(1, pw):
        for i in range(1, pu):
            order.append(flat(i, 0, k))
    # Face 3: j=pv (back), interior (i, k), i fastest
    for k in range(1, pw):
        for i in range(1, pu):
            order.append(flat(i, pv, k))
    # Face 4: k=0 (bottom), interior (i, j), i fastest
    for j in range(1, pv):
        for i in range(1, pu):
            order.append(flat(i, j, 0))
    # Face 5: k=pw (top), interior (i, j), i fastest
    for j in range(1, pv):
        for i in range(1, pu):
            order.append(flat(i, j, pw))

    # --- Volume interior (i fastest, then j, then k, per VTK) ---
    for k in range(1, pw):
        for j in range(1, pv):
            for i in range(1, pu):
                order.append(flat(i, j, k))

    return np.array(order, dtype=np.intp)


def vtk_ordering(degree: tuple[int, ...]) -> npt.NDArray[np.intp]:
    """Return the tensor-product to VTK Bézier point ordering permutation.

    Dispatches to the appropriate ordering function based on the parametric
    dimension (1D, 2D, or 3D).

    Args:
        degree: Polynomial degrees per parametric direction. Length determines
            the parametric dimension (1, 2, or 3).

    Returns:
        NDArray[intp]: Permutation array mapping flat tensor-product indices
        to VTK ordering.

    Raises:
        ValueError: If the parametric dimension is not 1, 2, or 3.
    """
    _dispatch = {
        1: lambda d: vtk_ordering_curve(d[0]),
        2: lambda d: vtk_ordering_quad(d[0], d[1]),
        3: lambda d: vtk_ordering_hex(d[0], d[1], d[2]),
    }
    dim = len(degree)
    if dim not in _dispatch:
        raise ValueError(f"Unsupported parametric dimension {dim}. Expected 1, 2, or 3.")
    return _dispatch[dim](degree)
