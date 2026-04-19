"""Deprecation shim -- re-exports ``ocelat.algoim`` under the old name.

The Saye 2022 Bernstein/Bezier implicit-quadrature engine used to live in
``pantr.bezier.implicit``. In ocelat's Stage 6 refactor it moved verbatim to
:mod:`ocelat.algoim`, where it serves as the Layer-3 backend for
immersed-FEM CSG quadrature.

This shim re-exports the same public names so existing call sites keep
working. Importing from ``pantr.bezier.implicit`` now emits a
:class:`DeprecationWarning` directing callers to the new location. The shim
will be removed one release cycle after the move.

Switch imports to ``from ocelat.algoim import ...``.

Re-exported names:

- :class:`~ocelat.algoim.ImplicitQuadrature`
- :class:`~ocelat.algoim.QuadStrategy`
- :class:`~ocelat.algoim.ReparamResult`
- :class:`~ocelat.algoim.SurfQuadResult`
- :class:`~ocelat.algoim.VolQuadResult`
- :func:`~ocelat.algoim.monomial_to_bernstein_2d`
- :func:`~ocelat.algoim.monomial_to_bernstein_3d`
"""

from __future__ import annotations

import warnings

from ocelat.algoim import (
    ImplicitQuadrature,
    QuadStrategy,
    ReparamResult,
    SurfQuadResult,
    VolQuadResult,
    monomial_to_bernstein_2d,
    monomial_to_bernstein_3d,
)

warnings.warn(
    "pantr.bezier.implicit has moved to ocelat.algoim. This shim will be "
    "removed in the next release; update imports to "
    "'from ocelat.algoim import ...'.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ImplicitQuadrature",
    "QuadStrategy",
    "ReparamResult",
    "SurfQuadResult",
    "VolQuadResult",
    "monomial_to_bernstein_2d",
    "monomial_to_bernstein_3d",
]
