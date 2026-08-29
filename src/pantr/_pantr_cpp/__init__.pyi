"""Type stub for the compiled C++ extension.

The extension is a nanobind module, so mypy cannot see into it: without this
package every use of :mod:`pantr._pantr_cpp` is an ``attr-defined`` error under
the project's strict configuration, and the kernel adapters under ``pantr.basis``
and ``pantr.quad`` are what use it. The stub is written by hand rather than
generated, because it is short enough to read in one sitting and a generated one
would need regenerating on every signature change anyway.

**It is a promise that has to be kept by hand.** Nothing checks these files
against ``cpp/bindings/``; if a binding's signature changes and the stub does
not, mypy will happily typecheck a call that fails at run time. The parity tests
exercise the real calls, which is what actually catches it.

Why it is a package
-------------------

It was one 1464-line file until the port reached the point where a dozen tickets
each had to add a type to it, all of them editing the same file at the same time.
The split is by binding area -- one stub module per group of
``cpp/bindings/*.cpp`` -- so a ticket that ports a type edits its own file plus
one import line here.

``mypy.ini`` sets ``strict = True``, which turns off implicit re-export: a name
merely imported below would not be visible as ``pantr._pantr_cpp.<name>``. Every
import is therefore written in the redundant-looking ``X as X`` form, which is
what marks it re-exported. Dropping the ``as`` on one line is enough to break
every caller of that symbol, so a new symbol is added in both places or in
neither.

Note:
    Adding this directory next to the built extension makes ``pantr._pantr_cpp``
    importable as an empty *namespace package* on an installation where the
    extension was never built, where it used to raise ``ImportError``. The
    extension wins whenever it exists, so nothing changes for a real build; but
    :func:`pantr._backend._cpp_extension_is_present` can no longer answer the
    question with a bare import, and does not.
"""

from typing import Final

from ._basis import bernstein_to_cardinal_1d as bernstein_to_cardinal_1d
from ._basis import bernstein_to_lagrange_1d as bernstein_to_lagrange_1d
from ._basis import cardinal_dual_legendre_coeffs_1d as cardinal_dual_legendre_coeffs_1d
from ._basis import cardinal_to_bernstein_1d as cardinal_to_bernstein_1d
from ._basis import cardinal_to_legendre_1d as cardinal_to_legendre_1d
from ._basis import lagrange_to_bernstein_1d as lagrange_to_bernstein_1d
from ._basis import legendre_to_cardinal_1d as legendre_to_cardinal_1d
from ._basis import monomial_to_bernstein_1d as monomial_to_bernstein_1d
from ._basis import tabulate_bernstein_1d as tabulate_bernstein_1d
from ._basis import tabulate_cardinal_bspline_1d as tabulate_cardinal_bspline_1d
from ._basis import tabulate_legendre_1d as tabulate_legendre_1d
from ._bezier import Bezier32 as Bezier32
from ._bezier import Bezier64 as Bezier64
from ._bezier import apply_reduction_operator as apply_reduction_operator
from ._bezier import clip_roots as clip_roots
from ._bezier import dedup_roots as dedup_roots
from ._bezier import degree_elevate_bezier_1d as degree_elevate_bezier_1d
from ._bezier import evaluate_bezier_1d as evaluate_bezier_1d
from ._bezier import evaluate_bezier_deriv_1d as evaluate_bezier_deriv_1d
from ._bezier import find_roots_batch as find_roots_batch
from ._bezier import restrict_bezier_1d as restrict_bezier_1d
from ._bezier import scalar_bernstein_product_1d as scalar_bernstein_product_1d
from ._bezier import slice_bezier_1d as slice_bezier_1d
from ._bezier import solve_monotone_root as solve_monotone_root
from ._bezier import solve_monotone_root_batch as solve_monotone_root_batch
from ._bezier import split_bezier_1d as split_bezier_1d
from ._bezier import yuksel_roots as yuksel_roots
from ._geometry import AABB as AABB
from ._grid import bvh_build as bvh_build
from ._grid import bvh_query_count as bvh_query_count
from ._grid import bvh_query_emit as bvh_query_emit
from ._grid import decode_flat_id as decode_flat_id
from ._grid import encode_midx as encode_midx
from ._grid import hier_collect_cell_bounds as hier_collect_cell_bounds
from ._grid import hier_locate_points as hier_locate_points
from ._grid import locate_points as locate_points
from ._quad import gauss_legendre_symmetric as gauss_legendre_symmetric
from ._quad import generate_tanh_sinh as generate_tanh_sinh
from ._quad import lambert_w_principal as lambert_w_principal
from ._quad import modified_chebyshev_nodes as modified_chebyshev_nodes
from ._quad import trapezoidal as trapezoidal
from ._transform import AffineTransform as AffineTransform

__compiler__: Final[str]
"""Compiler and version that built the extension, e.g. ``"gcc 14.4.0"``."""

__has_std_mdspan__: Final[bool]
"""Whether the build used ``std::mdspan`` rather than the Kokkos fallback."""

__fp_contract__: Final[str]
"""Whether the target ISA can fuse a multiply-add.

``"available"`` or ``"unavailable-on-target-isa"``. Read by
``tests/parity/test_basis_cardinal_bspline.py`` to choose between asserting bit-exact parity with
the numba oracle and asserting the derived FMA bound: with no fused instruction
on the target there is no rounding difference for a tolerance to absorb.
"""

class CapacityError(RuntimeError):
    """A fixed internal limit of the C++ implementation was exceeded.

    Raised where the failure is a property of the implementation rather than a
    defect in the argument -- the BVH's fixed traversal stack is the one case in
    this port. Defined in ``cpp/include/pantr/core/error.hpp`` and translated by
    the single ``nb::exception`` in ``cpp/bindings/pantr_cpp.cpp``, which also
    records why the tag registries deliberately raise ``KeyError`` from the
    Python wrapper instead.
    """
