# PaNTr demos

Standalone, runnable scripts that illustrate the main features of PaNTr. Each is
also a [Sphinx-Gallery](https://sphinx-gallery.github.io) source (reST header +
``# %%`` cells), so the documentation build renders them as an interactive
gallery — but they run on their own too:

```bash
pip install "pantr[viz]"            # the 3-D demos need the viz extra
python demos/01_visualization_basics.py
```

The 3-D demos open an interactive PyVista window. The plotting demos use Numba
kernels; if you hit a Numba background-warmup threading error on a fresh process,
run with JIT disabled (the documentation build does this automatically):

```bash
NUMBA_DISABLE_JIT=1 python demos/05_approximation.py
```

| Demo | Shows | Modules |
|---|---|---|
| `01_visualization_basics` | `plot`/`Scene`, control polygon, knot lines, scalar fields, VTK export | `viz` |
| `02_basis_gallery` | Bernstein / Lagrange / Legendre bases; change-of-basis matrix | `basis`, `change_basis` |
| `03_bspline_geometry_tour` | curves/surfaces, derivatives, Greville points, NURBS circle | `bspline`, `cad` |
| `04_knot_operations` | knot insertion, degree elevation, Bézier extraction | `bspline` |
| `05_approximation` | interpolation / L2 projection / quasi-interpolation; convergence | `bspline` |
| `06_cad_modeling` | primitives + extrude / revolve / ruled; assembly | `cad`, `transform` |
| `07_bezier_and_roots` | Bézier surface; Bernstein root finding; curve–line intersection | `bezier` |
| `08_thb_adaptive_refinement` | THB local refinement; hierarchical mesh + per-level control net | `bspline`, `grid`, `viz` |
| `09_grids_and_quadrature` | grids, `cell_quadrature` integration, BVH query, grid rendering | `grid`, `quad`, `geometry` |
| `10_transforms` | affine translation / rotation / scaling / shear; composition | `transform` |
