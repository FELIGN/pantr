"""
Affine transformations
=======================

:class:`~pantr.transform.AffineTransform` represents an affine map ``T(x) = A x + b``
with factory methods for translation, scaling, rotation, mirroring, and shear.
Transforms compose with ``@`` and apply to any geometry via ``geom.transform(T)``.
Because a transform acts on the *control points* -- not on sampled points -- the result
is another exact :class:`~pantr.bspline.Bspline`, not a discretization. This tutorial
builds a few transforms and applies them to a single base shape (the layout trick used
back in :doc:`/tutorials/04_cad_modeling`).
"""

import numpy as np

from pantr import viz
from pantr.cad import create_cylinder
from pantr.transform import AffineTransform

base = create_cylinder(radius=0.4, height=1.0)

# %%
# Individual transforms
# ---------------------
# Each transform produces a new, independent geometry; the original is untouched.
rotated = base.transform(AffineTransform.rotation_3d(np.pi / 4, axis=0))  # tilt about x
scaled = base.transform(AffineTransform.scaling([1.0, 1.0, 1.6]))  # stretch in z
sheared = base.transform(AffineTransform.shear(dim=3, component=0, direction=2, factor=0.5))

scene = viz.Scene()
for i, geom in enumerate([base, rotated, scaled, sheared]):
    placed = geom.transform(AffineTransform.translation([1.5 * i, 0.0, 0.0]))
    scene.add(placed, color="lightsteelblue", show_knot_lines=True)
scene.show()

# %%
# Composing transforms
# --------------------
# ``@`` composes transforms right-to-left, exactly like matrix multiplication:
# ``(T2 @ T1)(x) == T2(T1(x))``. Here we rotate, then scale, then translate.
combined = (
    AffineTransform.translation([0.0, 0.0, 0.5])
    @ AffineTransform.scaling([1.3, 1.3, 1.3])
    @ AffineTransform.rotation_3d(np.pi / 3, axis=1)
)
viz.plot(base.transform(combined), color="thistle", show_knot_lines=True)

# %%
# Composition equals sequential application
# -----------------------------------------
t1 = AffineTransform.rotation_3d(0.7, axis=2)
t2 = AffineTransform.translation([1.0, -0.5, 0.2])
once = base.transform(t2 @ t1)
twice = base.transform(t1).transform(t2)
err = float(np.max(np.abs(once.control_points - twice.control_points)))
print(f"max |(t2@t1) - t2∘t1| on control points = {err:.2e}")
