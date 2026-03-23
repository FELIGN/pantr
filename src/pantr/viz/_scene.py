"""Composable visualization scene for multiple geometries.

Provides:

- :class:`Scene`: a builder for adding multiple geometries with per-geometry
  rendering options (color, opacity, control points, knot lines).
- :func:`plot`: a convenience function for quick visualization of one or
  more geometries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._control_points import control_points_mesh, control_polygon_mesh
from ._knot_lines import knot_lines_meshes
from ._lazy_import import _import_pyvista
from ._vtk_cells import to_pyvista

if TYPE_CHECKING:
    import pyvista as pv

    from ..bezier import Bezier
    from ..bspline import Bspline


_DEFAULT_CP_COLOR = "red"
_DEFAULT_CP_SIZE = 8.0
_DEFAULT_POLYGON_COLOR = "gray"
_DEFAULT_KNOT_COLOR = "black"
_DEFAULT_KNOT_WIDTH = 2.0


class _GeometryEntry:
    """Internal record of a geometry added to a Scene."""

    def __init__(  # noqa: PLR0913
        self,
        geom: Bspline | Bezier,
        *,
        color: str | None = None,
        opacity: float = 1.0,
        show_control_points: bool = False,
        show_control_polygon: bool = False,
        show_knot_lines: bool = False,
        control_point_color: str = _DEFAULT_CP_COLOR,
        control_point_size: float = _DEFAULT_CP_SIZE,
        control_polygon_color: str = _DEFAULT_POLYGON_COLOR,
        knot_line_color: str = _DEFAULT_KNOT_COLOR,
        knot_line_width: float = _DEFAULT_KNOT_WIDTH,
        scalar_name: str = "scalar",
        scalar_bar: bool = True,
        elevation: bool = False,
    ) -> None:
        """Initialize a geometry entry.

        Args:
            geom: The geometry to visualize.
            color: Surface color. ``None`` uses the default colormap for
                scalar fields or pyvista's default for geometric objects.
            opacity: Surface opacity (0.0 to 1.0).
            show_control_points: Render control points as spheres.
            show_control_polygon: Render wireframe connecting control points.
            show_knot_lines: Render knot lines (B-splines only).
            control_point_color: Color for control point spheres.
            control_point_size: Point size for control points.
            control_polygon_color: Color for control polygon wireframe.
            knot_line_color: Color for knot lines.
            knot_line_width: Line width for knot lines.
            scalar_name: Name for scalar point data.
            scalar_bar: Show scalar bar for scalar fields.
            elevation: Use scalar as elevation coordinate.
        """
        self.geom = geom
        self.color = color
        self.opacity = opacity
        self.show_control_points = show_control_points
        self.show_control_polygon = show_control_polygon
        self.show_knot_lines = show_knot_lines
        self.control_point_color = control_point_color
        self.control_point_size = control_point_size
        self.control_polygon_color = control_polygon_color
        self.knot_line_color = knot_line_color
        self.knot_line_width = knot_line_width
        self.scalar_name = scalar_name
        self.scalar_bar = scalar_bar
        self.elevation = elevation


class Scene:
    """Composable multi-geometry visualization scene.

    Add geometries with :meth:`add`, then render with :meth:`show` or
    create a plotter with :meth:`to_plotter`.

    Example:
        >>> scene = Scene()
        >>> scene.add(surface, color="blue", show_knot_lines=True)
        >>> scene.add(curve, color="red", show_control_points=True)
        >>> scene.show()
    """

    def __init__(self) -> None:
        """Initialize an empty scene."""
        self._entries: list[_GeometryEntry] = []

    def add(  # noqa: PLR0913
        self,
        geom: Bspline | Bezier,
        *,
        color: str | None = None,
        opacity: float = 1.0,
        show_control_points: bool = False,
        show_control_polygon: bool = False,
        show_knot_lines: bool = False,
        control_point_color: str = _DEFAULT_CP_COLOR,
        control_point_size: float = _DEFAULT_CP_SIZE,
        control_polygon_color: str = _DEFAULT_POLYGON_COLOR,
        knot_line_color: str = _DEFAULT_KNOT_COLOR,
        knot_line_width: float = _DEFAULT_KNOT_WIDTH,
        scalar_name: str = "scalar",
        scalar_bar: bool = True,
        elevation: bool = False,
    ) -> Scene:
        """Add a geometry to the scene.

        Args:
            geom: B-spline or Bézier geometry to add.
            color: Surface color. ``None`` uses the default colormap for
                scalar fields or pyvista's default for geometric objects.
            opacity: Surface opacity (0.0 to 1.0).
            show_control_points: Render control points as spheres.
            show_control_polygon: Render wireframe connecting control points.
            show_knot_lines: Render knot lines (B-splines only).
            control_point_color: Color for control point spheres.
            control_point_size: Point size for control points.
            control_polygon_color: Color for control polygon wireframe.
            knot_line_color: Color for knot lines.
            knot_line_width: Line width for knot lines.
            scalar_name: Name for scalar point data.
            scalar_bar: Show scalar bar for scalar fields.
            elevation: Use scalar as elevation coordinate.

        Returns:
            Scene: Self, for method chaining.
        """
        self._entries.append(
            _GeometryEntry(
                geom,
                color=color,
                opacity=opacity,
                show_control_points=show_control_points,
                show_control_polygon=show_control_polygon,
                show_knot_lines=show_knot_lines,
                control_point_color=control_point_color,
                control_point_size=control_point_size,
                control_polygon_color=control_polygon_color,
                knot_line_color=knot_line_color,
                knot_line_width=knot_line_width,
                scalar_name=scalar_name,
                scalar_bar=scalar_bar,
                elevation=elevation,
            )
        )
        return self

    def to_plotter(self, **plotter_kwargs: object) -> pv.Plotter:
        """Create a pyvista Plotter with all geometries added.

        Args:
            **plotter_kwargs: Keyword arguments passed to
                ``pv.Plotter()``.

        Returns:
            pv.Plotter: A configured plotter (not yet shown).

        Raises:
            ImportError: If pyvista is not installed.
        """
        pv = _import_pyvista()
        plotter = pv.Plotter(**plotter_kwargs)

        for entry in self._entries:
            _add_entry_to_plotter(plotter, entry)

        return plotter  # type: ignore[no-any-return]

    def show(self, **plotter_kwargs: object) -> pv.Plotter:
        """Render the scene interactively.

        Args:
            **plotter_kwargs: Keyword arguments passed to
                ``pv.Plotter()``.

        Returns:
            pv.Plotter: The plotter after showing.

        Raises:
            ImportError: If pyvista is not installed.
        """
        plotter = self.to_plotter(**plotter_kwargs)
        plotter.show()
        return plotter


def _add_entry_to_plotter(plotter: pv.Plotter, entry: _GeometryEntry) -> None:
    """Add a single geometry entry to a pyvista plotter.

    Args:
        plotter: Target pyvista plotter.
        entry: Geometry entry with rendering options.
    """
    from ..bspline import Bspline as BsplineCls  # noqa: PLC0415

    grid = to_pyvista(
        entry.geom,
        scalar_name=entry.scalar_name,
        elevation=entry.elevation,
    )

    # Determine mesh_kwargs for the main geometry
    mesh_kwargs: dict[str, Any] = {
        "opacity": entry.opacity,
    }
    if entry.color is not None:
        mesh_kwargs["color"] = entry.color
    elif entry.scalar_name in grid.point_data:
        mesh_kwargs["scalars"] = entry.scalar_name
        mesh_kwargs["show_scalar_bar"] = entry.scalar_bar

    plotter.add_mesh(grid, **mesh_kwargs)

    # Control points
    if entry.show_control_points:
        cp_mesh = control_points_mesh(entry.geom)
        plotter.add_mesh(
            cp_mesh,
            color=entry.control_point_color,
            point_size=entry.control_point_size,
            render_points_as_spheres=True,
        )

    # Control polygon
    if entry.show_control_polygon:
        poly_mesh = control_polygon_mesh(entry.geom)
        plotter.add_mesh(
            poly_mesh,
            color=entry.control_polygon_color,
            style="wireframe",
            line_width=entry.knot_line_width,
        )

    # Knot lines
    if entry.show_knot_lines and isinstance(entry.geom, BsplineCls):
        for kl_mesh in knot_lines_meshes(entry.geom):
            plotter.add_mesh(
                kl_mesh,
                color=entry.knot_line_color,
                line_width=entry.knot_line_width,
                render_points_as_spheres=True,
                point_size=entry.control_point_size * 0.5,
            )


def plot(  # noqa: PLR0913
    *geoms: Bspline | Bezier,
    color: str | None = None,
    opacity: float = 1.0,
    show_control_points: bool = False,
    show_control_polygon: bool = False,
    show_knot_lines: bool = False,
    scalar_name: str = "scalar",
    scalar_bar: bool = True,
    elevation: bool = False,
    **plotter_kwargs: object,
) -> pv.Plotter:
    """Quick visualization of one or more geometries.

    Creates a :class:`Scene`, adds all geometries with the same rendering
    options, and shows the result interactively.

    For finer control (per-geometry colors, mixing different options), use
    :class:`Scene` directly.

    Args:
        *geoms: One or more B-spline or Bézier geometries.
        color: Surface color for all geometries.
        opacity: Surface opacity for all geometries.
        show_control_points: Render control points as spheres.
        show_control_polygon: Render control polygon wireframe.
        show_knot_lines: Render knot lines (B-splines only).
        scalar_name: Name for scalar point data.
        scalar_bar: Show scalar bar for scalar fields.
        elevation: Use scalar as elevation coordinate.
        **plotter_kwargs: Additional keyword arguments for ``pv.Plotter()``.

    Returns:
        pv.Plotter: The plotter after showing.

    Raises:
        ImportError: If pyvista is not installed.
    """
    scene = Scene()
    for geom in geoms:
        scene.add(
            geom,
            color=color,
            opacity=opacity,
            show_control_points=show_control_points,
            show_control_polygon=show_control_polygon,
            show_knot_lines=show_knot_lines,
            scalar_name=scalar_name,
            scalar_bar=scalar_bar,
            elevation=elevation,
        )
    return scene.show(**plotter_kwargs)
