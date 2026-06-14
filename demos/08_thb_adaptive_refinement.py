"""
THB-splines: adaptive local refinement
=======================================

Truncated hierarchical B-splines (THB-splines) add resolution only where it is
needed, keeping coarse basis functions elsewhere. Refinement leaves the
coefficients of functions that stay active at coarser levels unchanged
(truncation preserves coefficients). This demo refines toward a localized
feature, quasi-interpolates a peaked function onto the hierarchical space, and
renders the field with its hierarchical mesh and per-level control net (see
:mod:`pantr.viz`).
"""

import numpy as np

from pantr import viz
from pantr.bspline import THBSplineSpace, create_uniform_space, quasi_interpolate_thb_spline
from pantr.grid import hierarchical_grid, uniform_grid

# %%
# Build a graded hierarchical space
# ---------------------------------
# Start from a biquadratic 8x8 grid and refine the lower-left block twice, so
# fine cells cluster where a peak will sit.
root = create_uniform_space([2, 2], [8, 8])
grid = hierarchical_grid(uniform_grid([[0.0, 1.0], [0.0, 1.0]], 8), 2)
grid.refine(0, [0, 0], [4, 4])  # level 0 -> 1 on the lower-left quarter
grid.refine(1, [0, 0], [4, 4])  # level 1 -> 2 on the lower-left of that
space = THBSplineSpace(root, grid)
print(f"{space.num_levels} levels, {space.num_total_basis} active functions")


# %%
# Quasi-interpolate a peaked field
# --------------------------------
def peak(pts):
    """A sharp Gaussian bump centred in the refined corner."""
    x, y = pts[:, 0], pts[:, 1]
    return np.exp(-(((x - 0.25) ** 2 + (y - 0.25) ** 2) / 0.004))


field = quasi_interpolate_thb_spline(peak, space)

# %%
# Render the field, hierarchical mesh, and per-level control net
# --------------------------------------------------------------
# ``elevation=True`` lifts the scalar field into a surface; the knot lines are the
# active-cell boundaries (denser in the refined corner); the control net is drawn
# per level, coloured by level.
viz.plot(field, elevation=True, show_knot_lines=True, show_control_polygon=True)

# %%
# Just the hierarchical mesh
# --------------------------
# The active cells alone make the octree-like refinement structure obvious.
viz.plot(field, elevation=True, show_knot_lines=True, color="lightsteelblue")
