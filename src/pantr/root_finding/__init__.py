"""Root finding for polynomials in the Bernstein basis.

.. deprecated::
    This module has been moved to :mod:`pantr.bezier`. Import from
    :mod:`pantr.bezier` instead.
"""

from pantr.bezier._root_finding import (
    find_roots,
    find_roots_batch,
    solve_monotone_root,
    solve_monotone_root_batch,
)

__all__ = [
    "find_roots",
    "find_roots_batch",
    "solve_monotone_root",
    "solve_monotone_root_batch",
]
