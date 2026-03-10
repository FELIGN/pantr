"""Public API surface for PaNTr.

Defines package metadata and exported interfaces.
"""

from typing import Final

# Private API imports (accessible but not in __all__)
# Users can access private functions via: pantr._basis_impl._function_name, etc.
from . import (
    _basis_utils,  # noqa: F401
)
from ._bspline_space_factory import (
    create_cardinal_Bspline_knot_vector,
    create_uniform_open_knot_vector,
    create_uniform_periodic_knot_vector,
)

# Public API imports
from .basis import (
    LagrangeVariant,
    tabulate_Bernstein_basis,
    tabulate_Bernstein_basis_1D,
    tabulate_cardinal_Bspline_basis,
    tabulate_cardinal_Bspline_basis_1D,
    tabulate_Lagrange_basis,
    tabulate_Lagrange_basis_1D,
    tabulate_Legendre_basis_1D,
)
from .bspline_space_1D import BsplineSpace1D
from .bspline_space_nd import BsplineSpace
from .change_basis import (
    compute_Bernstein_to_cardinal_change_basis_1D,
    compute_Bernstein_to_Lagrange_change_basis_1D,
    compute_cardinal_to_Bernstein_change_basis_1D,
    compute_Lagrange_to_Bernstein_change_basis_1D,
)
from .quad import (
    PointsLattice,
    create_Lagrange_points_lattice,
    get_chebyshev_gauss_1st_kind_quadrature_1D,
    get_chebyshev_gauss_2nd_kind_quadrature_1D,
    get_gauss_legendre_quadrature_1D,
    get_gauss_lobatto_legendre_quadrature_1D,
    get_trapezoidal_quadrature_1D,
)
from .tolerance import (
    ToleranceInfo,
    get_conservative_tolerance,
    get_default_tolerance,
    get_machine_epsilon,
    get_strict_tolerance,
    get_tolerance_info,
)

# Package metadata
__version__: Final[str] = "0.1.0"
__license__: Final[str] = "MIT"
__author__: Final[str] = "Pablo Antolin <pablo.antolin@epfl.ch>"

# Public interface: only functions/classes that don't start with _
__all__ = [
    "BsplineSpace",
    "BsplineSpace1D",
    "LagrangeVariant",
    "PointsLattice",
    "ToleranceInfo",
    "__author__",
    "__license__",
    "__version__",
    "compute_Bernstein_to_Lagrange_change_basis_1D",
    "compute_Bernstein_to_cardinal_change_basis_1D",
    "compute_Lagrange_to_Bernstein_change_basis_1D",
    "compute_cardinal_to_Bernstein_change_basis_1D",
    "create_Lagrange_points_lattice",
    "create_cardinal_Bspline_knot_vector",
    "create_uniform_open_knot_vector",
    "create_uniform_periodic_knot_vector",
    "get_chebyshev_gauss_1st_kind_quadrature_1D",
    "get_chebyshev_gauss_2nd_kind_quadrature_1D",
    "get_conservative_tolerance",
    "get_default_tolerance",
    "get_gauss_legendre_quadrature_1D",
    "get_gauss_lobatto_legendre_quadrature_1D",
    "get_machine_epsilon",
    "get_strict_tolerance",
    "get_tolerance_info",
    "get_trapezoidal_quadrature_1D",
    "tabulate_Bernstein_basis",
    "tabulate_Bernstein_basis_1D",
    "tabulate_Lagrange_basis",
    "tabulate_Lagrange_basis_1D",
    "tabulate_Legendre_basis_1D",
    "tabulate_cardinal_Bspline_basis",
    "tabulate_cardinal_Bspline_basis_1D",
]

# Defer numba JIT compilation warmups to a background thread to prevent
# blocking module import, allowing immediate interaction unless Numba
# functions are called right away.
import logging
import threading
from typing import TYPE_CHECKING

if not TYPE_CHECKING:

    def _async_warmup() -> None:
        try:
            logger = logging.getLogger(__name__)
            logger.debug("Starting Numba JIT warmup...")
            from . import (  # noqa: PLC0415
                _basis_core,
                _bspline_basis_core,
                _bspline_eval,
                _bspline_extraction,
                _bspline_knots,
            )

            _basis_core._warmup_numba_functions()
            _bspline_basis_core._warmup_numba_functions()
            _bspline_eval._warmup_numba_functions()
            _bspline_extraction._warmup_numba_functions()
            _bspline_knots._warmup_numba_functions()
            logger.debug("Finished Numba JIT warmup.")
        except Exception:
            # During process teardown (e.g. short scripts), background Numba caching
            # might fail due to unavailable module locators. We silently ignore this.
            pass

    threading.Thread(target=_async_warmup, daemon=True).start()
