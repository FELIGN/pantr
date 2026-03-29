"""Public API surface for PaNTr.

Defines package metadata and exported interfaces for polynomial and NURBS
geometric modeling.

The main modules are:
- :mod:`pantr.basis`: 1D polynomial basis evaluation (Bernstein, Lagrange, etc.).
- :mod:`pantr.bspline`: B-spline spaces, geometric objects, and knot vector factories.
- :mod:`pantr.change_basis`: Transformation matrices between different bases.
- :mod:`pantr.quad`: Quadrature rules and evaluation grid helpers.
- :mod:`pantr.tolerance`: Uniform floating-point tolerance utilities.
- :mod:`pantr.transform`: Affine transformations for geometric objects.
- :mod:`pantr.cad`: Constructive geometry for B-spline curves, surfaces, and volumes.
- :mod:`pantr.root_finding`: Root finding for polynomials in the Bernstein basis.
"""

from typing import Final

from . import (
    basis,
    bezier,
    bspline,
    cad,
    change_basis,
    quad,
    root_finding,
    tolerance,
    transform,
)
from ._parallel import get_num_threads, num_threads, set_num_threads
from .basis import _basis_utils  # noqa: F401

# Package metadata
__version__: Final[str] = "0.1.0"
__license__: Final[str] = "MIT"
__author__: Final[str] = "Pablo Antolin <pablo.antolin@epfl.ch>"

__all__ = [
    "__author__",
    "__license__",
    "__version__",
    "basis",
    "bezier",
    "bspline",
    "cad",
    "change_basis",
    "get_num_threads",
    "num_threads",
    "quad",
    "root_finding",
    "set_num_threads",
    "tolerance",
    "transform",
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
            from .bspline import (  # noqa: PLC0415
                _bspline_basis_core,
                _bspline_eval,
                _bspline_extraction,
                _bspline_knot_insertion_core,
                _bspline_knot_removal_core,
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
            _bspline_knot_removal_core._warmup_numba_functions()
            _bspline_knots._warmup_numba_functions()
            from .bspline import _bspline_blossom_core  # noqa: PLC0415

            _bspline_blossom_core._warmup_numba_functions()
            from .bezier import _bezier_core  # noqa: PLC0415

            _bezier_core._warmup_numba_functions()
            from .bezier import (  # noqa: PLC0415
                _batch_core,
                _clipping_core,
                _root_finding_core,
                _yuksel_core,
            )

            _root_finding_core._warmup_numba_functions()
            _yuksel_core._warmup_numba_functions()
            _clipping_core._warmup_numba_functions()
            _batch_core._warmup_numba_functions()
            logger.debug("Finished Numba JIT warmup.")
        except Exception:
            # During process teardown (e.g. short scripts), background Numba caching
            # might fail due to unavailable module locators. We silently ignore this.
            pass
        finally:
            # Always signal completion so callers are never blocked indefinitely.
            _warmup_complete.set()

    threading.Thread(target=_async_warmup, daemon=True).start()
