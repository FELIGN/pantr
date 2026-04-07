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
from pantr.viz import implicit_to_pyvista, quadrature_to_pyvista

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

    vol = iq.volume_reparam(q=10, signs=[-1])
    surf = iq.surface_reparam(q=20, poly_idx=0)
    print(f"  Volume: {vol.n_cells} quads, {vol.points.shape[0]} nodes")
    print(f"  Surface: {surf.n_cells} curves, {surf.points.shape[0]} nodes")

    pl = pv.Plotter()
    pl.add_mesh(
        implicit_to_pyvista(vol),
        color="steelblue",
        opacity=0.6,
        label="Interior",
    )
    pl.add_mesh(
        implicit_to_pyvista(surf),
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

    # Region inside c1 but outside c2 (crescent)
    diff = iq.volume_reparam(q=10, signs=[-1, +1])
    # Full boundary of the crescent domain
    boundary = iq.surface_reparam(q=15, poly_idx=0, signs=[-1, +1])

    print(f"  c1 \\ c2: {diff.n_cells} quads")
    print(f"  boundary: {boundary.n_cells} curves")

    pl = pv.Plotter()
    pl.add_mesh(implicit_to_pyvista(diff), color="steelblue", opacity=0.5, label="c1 \\ c2")
    if boundary.n_cells > 0:
        pl.add_mesh(
            implicit_to_pyvista(boundary),
            color="red",
            line_width=4,
            label="boundary",
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

    vol = iq.volume_reparam(q=10, signs=[-1])
    surf = iq.surface_reparam(q=20, poly_idx=0)
    print(f"  Volume: {vol.n_cells} quads")
    print(f"  Surface: {surf.n_cells} curves")

    pl = pv.Plotter()
    pl.add_mesh(implicit_to_pyvista(vol), color="mediumseagreen", opacity=0.6)
    pl.add_mesh(implicit_to_pyvista(surf), color="darkgreen", line_width=4)
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

    vol = iq.volume_reparam(q=5, signs=[-1])
    surf = iq.surface_reparam(q=10, poly_idx=0)
    print(f"  Volume: {vol.n_cells} hexes, {vol.points.shape[0]} nodes")
    print(f"  Surface: {surf.n_cells} quads, {surf.points.shape[0]} nodes")

    pl = pv.Plotter(shape=(1, 2))

    pl.subplot(0, 0)
    pl.add_title("Volume (hexes)")
    pl.add_mesh(implicit_to_pyvista(vol), color="steelblue", opacity=0.4, show_edges=True)

    pl.subplot(0, 1)
    pl.add_title("Surface (quads)")
    pl.add_mesh(implicit_to_pyvista(surf), color="tomato", opacity=0.8)

    pl.link_views()
    pl.show()


def demo_3d_two_spheres() -> None:
    """Intersection of two overlapping spheres."""
    print("=== 3D Two Spheres ===")
    s1 = _sphere_bernstein(0.4, 0.5, 0.5, 0.3)
    s2 = _sphere_bernstein(0.6, 0.5, 0.5, 0.3)
    iq = ImplicitQuadrature(s1, s2)

    # Intersection: inside both
    inter = iq.volume_reparam(q=5, signs=[-1, -1])
    # Full boundary of the intersection domain (both caps)
    boundary = iq.surface_reparam(q=10, poly_idx=0, signs=[-1, -1])

    print(f"  Intersection volume: {inter.n_cells} hexes")
    print(f"  Boundary: {boundary.n_cells} quads")

    pl = pv.Plotter()
    if inter.n_cells > 0:
        pl.add_mesh(
            implicit_to_pyvista(inter),
            color="gold",
            opacity=0.3,
            show_edges=True,
            label="Intersection vol",
        )
    if boundary.n_cells > 0:
        pl.add_mesh(
            implicit_to_pyvista(boundary),
            color="tomato",
            opacity=0.7,
            label="Boundary",
        )
    pl.add_legend()
    pl.add_title("Two spheres: intersection")
    pl.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def demo_2d_circle_with_quadrature() -> None:
    """Circle with quadrature points overlaid on the reparameterization."""
    print("=== 2D Circle + Quadrature ===")
    coeffs = _circle_bernstein(0.5, 0.5, 0.3)
    iq = ImplicitQuadrature(coeffs)

    # Reparameterization
    vol = iq.volume_reparam(q=10, signs=[-1])
    surf = iq.surface_reparam(q=20, poly_idx=0)

    # Quadrature
    vol_quad = iq.volume_quad(q=4)
    surf_quad = iq.surface_quad(q=6)
    vol_pts, vol_wts = vol_quad
    surf_pts, surf_sw, surf_nw = surf_quad
    print(f"  Volume quad: {len(vol_wts)} points")
    print(f"  Surface quad: {len(surf_sw)} points")

    # Filter volume quad to inside domain (phi < 0).
    phi_vals = iq.eval_poly(0, vol_pts)
    inside = phi_vals < 0
    vol_quad_inside = (vol_pts[inside], vol_wts[inside])

    pl = pv.Plotter()

    # Reparameterization as background
    pl.add_mesh(implicit_to_pyvista(vol), color="steelblue", opacity=0.3)
    pl.add_mesh(implicit_to_pyvista(surf), color="grey", line_width=2)

    # Volume quadrature points (spheres, coloured by weight)
    vol_cloud = quadrature_to_pyvista(vol_quad_inside)
    pl.add_mesh(
        vol_cloud,
        scalars="weight",
        render_points_as_spheres=True,
        point_size=12,
        cmap="viridis",
        scalar_bar_args={"title": "Volume weight"},
    )

    # Surface quadrature points (spheres, coloured by scalar weight)
    surf_cloud = quadrature_to_pyvista(surf_quad)
    pl.add_mesh(
        surf_cloud,
        scalars="weight",
        render_points_as_spheres=True,
        point_size=10,
        cmap="plasma",
        scalar_bar_args={"title": "Surface weight"},
    )

    pl.view_xy()
    pl.add_title("Circle: reparam + quadrature points")
    pl.show()


def demo_3d_sphere_with_quadrature() -> None:
    """Sphere with surface quadrature and normal arrows."""
    print("=== 3D Sphere + Surface Quadrature ===")
    coeffs = _sphere_bernstein(0.5, 0.5, 0.5, 0.35)
    iq = ImplicitQuadrature(coeffs)

    # Reparameterization
    surf_reparam = iq.surface_reparam(q=10, poly_idx=0)
    print(f"  Surface reparam: {surf_reparam.n_cells} quads")

    # Surface quadrature with normals
    surf_quad = iq.surface_quad(q=6)
    _, surf_sw, _ = surf_quad
    print(f"  Surface quad: {len(surf_sw)} points")

    pl = pv.Plotter()

    # Reparameterized surface (translucent)
    pl.add_mesh(implicit_to_pyvista(surf_reparam), color="tomato", opacity=0.4)

    # Surface quadrature: points + normal arrows
    surf_cloud, arrows = quadrature_to_pyvista(surf_quad, show_normals=True, normal_scale=0.05)
    pl.add_mesh(
        surf_cloud,
        scalars="weight",
        render_points_as_spheres=True,
        point_size=10,
        cmap="plasma",
        scalar_bar_args={"title": "Surface weight"},
    )
    pl.add_mesh(arrows, color="navy", opacity=0.7)

    pl.add_title("Sphere: surface reparam + quadrature + normals")
    pl.show()


if __name__ == "__main__":
    print("Implicit domain reparameterization demos\n")

    # 2D demos
    demo_2d_circle()
    demo_2d_two_circles()
    demo_2d_ellipse()

    # 3D demos
    demo_3d_sphere()
    demo_3d_two_spheres()

    # Quadrature overlay demos
    demo_2d_circle_with_quadrature()
    demo_3d_sphere_with_quadrature()

    print("\nDone.")
