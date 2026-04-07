"""Demo: implicit domain reparameterization with Lagrange cells.

Shows how to reparameterize implicit domains (volume and surface) as
high-order Lagrange cells and visualize them with pyvista.

Examples:
  2D: circle, two-circle intersection, star-shaped domain
  3D: sphere, torus cross-section
"""

from __future__ import annotations

import numpy as np
import pyvista as pv

from pantr.bezier.implicit import (
    ImplicitQuadrature,
    monomial_to_bernstein_2d,
    monomial_to_bernstein_3d,
)
from pantr.viz import implicit_to_pyvista

LO2 = np.array([0.0, 0.0])
HI2 = np.array([1.0, 1.0])
LO3 = np.array([0.0, 0.0, 0.0])
HI3 = np.array([1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# 2D geometries
# ---------------------------------------------------------------------------


def _circle_bernstein(cx: float, cy: float, r: float) -> np.ndarray:
    """(x - cx)^2 + (y - cy)^2 - r^2 in Bernstein form on [0,1]^2."""
    mono = np.zeros((3, 3))
    mono[0, 0] = cx**2 + cy**2 - r**2
    mono[1, 0] = -2 * cx
    mono[0, 1] = -2 * cy
    mono[2, 0] = 1.0
    mono[0, 2] = 1.0
    return monomial_to_bernstein_2d(mono, (2, 2), LO2, HI2)


def _ellipse_bernstein(cx: float, cy: float, a: float, b: float) -> np.ndarray:
    """((x-cx)/a)^2 + ((y-cy)/b)^2 - 1 in Bernstein form on [0,1]^2."""
    mono = np.zeros((3, 3))
    mono[0, 0] = (cx / a) ** 2 + (cy / b) ** 2 - 1.0
    mono[1, 0] = -2 * cx / a**2
    mono[0, 1] = -2 * cy / b**2
    mono[2, 0] = 1.0 / a**2
    mono[0, 2] = 1.0 / b**2
    return monomial_to_bernstein_2d(mono, (2, 2), LO2, HI2)


def demo_2d_circle() -> None:
    """Circle interior and boundary."""
    print("=== 2D Circle ===")
    coeffs = _circle_bernstein(0.5, 0.5, 0.3)
    iq = ImplicitQuadrature(coeffs)

    vol = iq.volume_reparam(q=6, signs=[-1])
    surf = iq.surface_reparam(q=6, poly_idx=0)
    print(f"  Volume: {vol.n_cells} quads, {vol.points.shape[0]} nodes")
    print(f"  Surface: {surf.n_cells} curves, {surf.points.shape[0]} nodes")

    pl = pv.Plotter()
    pl.add_mesh(
        implicit_to_pyvista(vol).tessellate(),
        color="steelblue",
        opacity=0.6,
        label="Interior",
    )
    pl.add_mesh(
        implicit_to_pyvista(surf).tessellate(),
        color="red",
        line_width=4,
        label="Boundary",
    )
    pl.add_legend()
    pl.view_xy()
    pl.add_title("Circle: volume + surface")
    pl.show()


def demo_2d_two_circles() -> None:
    """Intersection / difference of two circles."""
    print("=== 2D Two Circles ===")
    c1 = _circle_bernstein(0.35, 0.5, 0.25)
    c2 = _circle_bernstein(0.65, 0.5, 0.25)
    iq = ImplicitQuadrature(c1, c2)

    # Region inside c1 but outside c2
    diff = iq.volume_reparam(q=6, signs=[-1, +1])
    # Surface of c1 restricted to outside c2
    surf_c1 = iq.surface_reparam(q=6, poly_idx=0, signs=[0, +1])
    # Surface of c2 restricted to inside c1 (the "cut")
    surf_c2 = iq.surface_reparam(q=6, poly_idx=1, signs=[-1, 0])

    print(f"  c1 \\ c2: {diff.n_cells} quads")
    print(f"  boundary c1 outside c2: {surf_c1.n_cells} curves")
    print(f"  boundary c2 inside c1: {surf_c2.n_cells} curves")

    pl = pv.Plotter()
    pl.add_mesh(
        implicit_to_pyvista(diff).tessellate(),
        color="steelblue",
        opacity=0.5,
        label="c1 \\ c2",
    )
    if surf_c1.n_cells > 0:
        pl.add_mesh(
            implicit_to_pyvista(surf_c1).tessellate(),
            color="red",
            line_width=4,
            label="bdry c1",
        )
    if surf_c2.n_cells > 0:
        pl.add_mesh(
            implicit_to_pyvista(surf_c2).tessellate(),
            color="orange",
            line_width=4,
            label="cut (c2)",
        )
    pl.add_legend()
    pl.view_xy()
    pl.add_title("Two circles: c1 \\ c2")
    pl.show()


def demo_2d_ellipse() -> None:
    """Ellipse with different aspect ratios."""
    print("=== 2D Ellipse ===")
    coeffs = _ellipse_bernstein(0.5, 0.5, 0.4, 0.2)
    iq = ImplicitQuadrature(coeffs)

    vol = iq.volume_reparam(q=6, signs=[-1])
    surf = iq.surface_reparam(q=8, poly_idx=0)
    print(f"  Volume: {vol.n_cells} quads")
    print(f"  Surface: {surf.n_cells} curves")

    pl = pv.Plotter()
    pl.add_mesh(
        implicit_to_pyvista(vol).tessellate(),
        color="mediumseagreen",
        opacity=0.6,
    )
    pl.add_mesh(
        implicit_to_pyvista(surf).tessellate(),
        color="darkgreen",
        line_width=4,
    )
    pl.view_xy()
    pl.add_title("Ellipse: volume + surface")
    pl.show()


# ---------------------------------------------------------------------------
# 3D geometries
# ---------------------------------------------------------------------------


def _sphere_bernstein(cx: float, cy: float, cz: float, r: float) -> np.ndarray:
    """(x-cx)^2 + (y-cy)^2 + (z-cz)^2 - r^2 in Bernstein form on [0,1]^3."""
    mono = np.zeros((3, 3, 3))
    mono[0, 0, 0] = cx**2 + cy**2 + cz**2 - r**2
    mono[1, 0, 0] = -2 * cx
    mono[0, 1, 0] = -2 * cy
    mono[0, 0, 1] = -2 * cz
    mono[2, 0, 0] = 1.0
    mono[0, 2, 0] = 1.0
    mono[0, 0, 2] = 1.0
    return monomial_to_bernstein_3d(mono, (2, 2, 2), LO3, HI3)


def demo_3d_sphere() -> None:
    """Sphere interior and surface."""
    print("=== 3D Sphere ===")
    coeffs = _sphere_bernstein(0.5, 0.5, 0.5, 0.35)
    iq = ImplicitQuadrature(coeffs)

    vol = iq.volume_reparam(q=4, signs=[-1])
    surf = iq.surface_reparam(q=6, poly_idx=0)
    print(f"  Volume: {vol.n_cells} hexes, {vol.points.shape[0]} nodes")
    print(f"  Surface: {surf.n_cells} quads, {surf.points.shape[0]} nodes")

    pl = pv.Plotter(shape=(1, 2))

    pl.subplot(0, 0)
    pl.add_title("Volume (hexes)")
    grid_vol = implicit_to_pyvista(vol).tessellate()
    pl.add_mesh(grid_vol, color="steelblue", opacity=0.4, show_edges=True)

    pl.subplot(0, 1)
    pl.add_title("Surface (quads)")
    grid_surf = implicit_to_pyvista(surf).tessellate()
    pl.add_mesh(grid_surf, color="tomato", opacity=0.8)

    pl.link_views()
    pl.show()


def demo_3d_two_spheres() -> None:
    """Intersection of two overlapping spheres."""
    print("=== 3D Two Spheres ===")
    s1 = _sphere_bernstein(0.4, 0.5, 0.5, 0.3)
    s2 = _sphere_bernstein(0.6, 0.5, 0.5, 0.3)
    iq = ImplicitQuadrature(s1, s2)

    # Intersection: inside both
    inter = iq.volume_reparam(q=4, signs=[-1, -1])
    # Surface of s1 inside s2
    surf_s1 = iq.surface_reparam(q=5, poly_idx=0, signs=[0, -1])
    # Surface of s2 inside s1
    surf_s2 = iq.surface_reparam(q=5, poly_idx=1, signs=[-1, 0])

    print(f"  Intersection volume: {inter.n_cells} hexes")
    print(f"  Surface s1 (inside s2): {surf_s1.n_cells} quads")
    print(f"  Surface s2 (inside s1): {surf_s2.n_cells} quads")

    pl = pv.Plotter()
    if inter.n_cells > 0:
        pl.add_mesh(
            implicit_to_pyvista(inter).tessellate(),
            color="gold",
            opacity=0.3,
            show_edges=True,
            label="Intersection vol",
        )
    if surf_s1.n_cells > 0:
        pl.add_mesh(
            implicit_to_pyvista(surf_s1).tessellate(),
            color="steelblue",
            opacity=0.7,
            label="s1 cap",
        )
    if surf_s2.n_cells > 0:
        pl.add_mesh(
            implicit_to_pyvista(surf_s2).tessellate(),
            color="tomato",
            opacity=0.7,
            label="s2 cap",
        )
    pl.add_legend()
    pl.add_title("Two spheres: intersection")
    pl.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Implicit domain reparameterization demos\n")

    # 2D demos
    demo_2d_circle()
    demo_2d_two_circles()
    demo_2d_ellipse()

    # 3D demos
    demo_3d_sphere()
    demo_3d_two_spheres()

    print("\nDone.")
