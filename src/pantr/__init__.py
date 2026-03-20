"""Public API surface for PaNTr.

Defines package metadata and exported interfaces for polynomial and NURBS
geometric modeling.

The main modules are:
- :mod:`pantr.basis`: 1D polynomial basis evaluation (Bernstein, Lagrange, etc.).
- :mod:`pantr.bspline`: B-spline spaces, geometric objects, and knot vector factories.
- :mod:`pantr.change_basis`: Transformation matrices between different bases.
- :mod:`pantr.quad`: Quadrature rules and evaluation grid helpers.
- :mod:`pantr.tolerance`: Uniform floating-point tolerance utilities.
"""

from typing import Final

# Private API imports (accessible but not in __all__)
# Users can access private functions via: pantr._basis_impl._function_name, etc.
from . import (
    _basis_utils,  # noqa: F401
)
from ._bspline_space_factory import (
    create_cardinal,
    create_uniform_open,
    create_uniform_periodic,
)
from .basis import (
    LagrangeVariant,
    tabulate_bernstein,
    tabulate_bernstein_1d,
    tabulate_cardinal_bspline,
    tabulate_cardinal_bspline_1d,
    tabulate_lagrange,
    tabulate_lagrange_1d,
    tabulate_legendre_1d,
)
from .bezier import Bezier
from .bspline import BsplineSpace, BsplineSpace1D
from .change_basis import (
    compute_bernstein_to_cardinal_1d,
    compute_bernstein_to_lagrange_1d,
    compute_cardinal_to_bernstein_1d,
    compute_lagrange_to_bernstein_1d,
)
from .quad import (
    PointsLattice,
    create_lagrange_points_lattice,
    get_chebyshev_gauss_1st_kind_1d,
    get_chebyshev_gauss_2nd_kind_1d,
    get_gauss_legendre_1d,
    get_gauss_lobatto_legendre_1d,
    get_trapezoidal_1d,
)
from .tolerance import (
    ToleranceInfo,
    get_conservative,
    get_default,
    get_info,
    get_machine_epsilon,
    get_strict,
)

# Package metadata
__version__: Final[str] = "0.1.0"
__license__: Final[str] = "MIT"
__author__: Final[str] = "Pablo Antolin <pablo.antolin@epfl.ch>"

# Public interface: only functions/classes that don't start with _
__all__ = [
    "Bezier",
    "BsplineSpace",
    "BsplineSpace1D",
    "LagrangeVariant",
    "PointsLattice",
    "ToleranceInfo",
    "__author__",
    "__license__",
    "__version__",
    "compute_bernstein_to_cardinal_1d",
    "compute_bernstein_to_lagrange_1d",
    "compute_cardinal_to_bernstein_1d",
    "compute_lagrange_to_bernstein_1d",
    "create_cardinal",
    "create_lagrange_points_lattice",
    "create_uniform_open",
    "create_uniform_periodic",
    "get_chebyshev_gauss_1st_kind_1d",
    "get_chebyshev_gauss_2nd_kind_1d",
    "get_conservative",
    "get_default",
    "get_gauss_legendre_1d",
    "get_gauss_lobatto_legendre_1d",
    "get_info",
    "get_machine_epsilon",
    "get_strict",
    "get_trapezoidal_1d",
    "tabulate_bernstein",
    "tabulate_bernstein_1d",
    "tabulate_cardinal_bspline",
    "tabulate_cardinal_bspline_1d",
    "tabulate_lagrange",
    "tabulate_lagrange_1d",
    "tabulate_legendre_1d",
]

# Defer numba JIT compilation warmups to a background thread to prevent
# blocking module import, allowing immediate interaction unless Numba
# functions are called right away.
import logging
import threading
from typing import TYPE_CHECKING

if not TYPE_CHECKING:

    def _async_warmup() -> None:
        from ._numba_compat import _warmup_complete  # noqa: PLC0415

        try:
            logger = logging.getLogger(__name__)
            logger.debug("Starting Numba JIT warmup...")
            from . import (  # noqa: PLC0415
                _bspline_basis_core,
                _bspline_eval,
                _bspline_extraction,
                _bspline_knot_insertion_core,
                _bspline_knots,
            )

            # _basis_core kernels use parallel=True. Numba's default threading
            # layer (workqueue) is not safe for concurrent parallel calls from
            # multiple Python threads.  Compiling them here (from a background
            # thread) while the main thread may also call them leads to a crash.
            # Instead they compile lazily on first user call (always from the
            # main / caller thread) and are cached to disk by Numba's cache=True.
            _bspline_basis_core._warmup_numba_functions()
            _bspline_eval._warmup_numba_functions()
            _bspline_extraction._warmup_numba_functions()
            _bspline_knot_insertion_core._warmup_numba_functions()
            _bspline_knots._warmup_numba_functions()
            from . import _bspline_blossom_core  # noqa: PLC0415

            _bspline_blossom_core._warmup_numba_functions()
            logger.debug("Finished Numba JIT warmup.")
        except Exception:
            # During process teardown (e.g. short scripts), background Numba caching
            # might fail due to unavailable module locators. We silently ignore this.
            pass
        finally:
            # Always signal completion so callers are never blocked indefinitely.
            _warmup_complete.set()

    threading.Thread(target=_async_warmup, daemon=True).start()
