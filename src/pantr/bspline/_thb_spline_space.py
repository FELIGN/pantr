"""Truncated hierarchical B-spline spaces (THB-splines).

This module defines :class:`THBSplineSpace`, a hierarchical spline space built on
a :class:`pantr.grid.HierarchicalGrid`.  It follows the G+Smo *self-evaluating*
model: per level it stores the Kraft selection of active tensor-product B-splines,
and a coefficient vector only for truncated functions.  Untruncated functions are
plain tensor-product B-splines.  Both the truncated (THB, default) and non-truncated
(HB) bases are supported via the ``truncate`` flag.

Since ``design/cross_backend_types.md`` the *value* is owned by C++
(``cpp/include/pantr/bspline/thb_space.hpp``) and :class:`THBSplineSpace` is a wrapper
holding one implementation of it, exactly as :mod:`pantr.bspline._bspline_space_nd`
does for the tensor-product type. There are two implementations and they are not two
spaces: :class:`_THBSplineSpacePython` is the oracle the port is checked against, and
the C++ handle is the thing being checked. :func:`_impl_class` picks between them, per
process and per dtype.

**The oracle's public surface is exactly the C++ type's.** That is what lets the
wrapper forward without asking which backend it holds, and it is why the oracle gained
:attr:`~_THBSplineSpacePython.regularity`, :attr:`~_THBSplineSpacePython.level_offsets`,
:attr:`~_THBSplineSpacePython.num_truncated`,
:meth:`~_THBSplineSpacePython.contributions`, :meth:`~_THBSplineSpacePython.dof_level`
and :meth:`~_THBSplineSpacePython.truncated` when the wrapper landed: each was already
there as private state and had to become a member for the forward to have something to
call.

**What the wrapper computes rather than forwards, and why it is not left on the
oracle.** Basis tabulation, the windowed :meth:`~THBSplineSpace.restrict` and the three
prolongation operators have no C++ counterpart yet. They are *computations over* a
space rather than properties *of* one, so they live on the wrapper and are written
against the forwarded accessors alone -- :meth:`~THBSplineSpace.contributions`,
:meth:`~THBSplineSpace.truncated`, :meth:`~THBSplineSpace.level_space`,
:meth:`~THBSplineSpace.active_function_indices`, :attr:`~THBSplineSpace.level_offsets`
and :meth:`~THBSplineSpace.dof_level` -- so that one body serves both backends. The
alternative, a second always-Python space kept beside ``_impl`` to serve them, would
duplicate the state and let the two drift after a :meth:`~THBSplineSpace.refine`, which
is a wrong answer rather than a crash. This is the same line
:mod:`pantr.bspline._bspline_space_nd` draws, and the mixed dispatch it produces is the
temporary seam the type front introduces; a cleanup ticket removes it once the whole
front lands.

The wrapper keeps the root space and the grid it was built from, in ``_root_space`` and
``_grid``, so that ``thb.grid is grid`` holds. ``design/bspline_ownership_lifetime.md``
F6 records why that is an identity contract rather than a convenience, and it is what
requires the C++ constructor to *share* its nested objects rather than copy them.

Main exports:

- :class:`THBSplineSpace`: hierarchical B-spline space on a
  :class:`~pantr.grid.HierarchicalGrid`.
"""

from __future__ import annotations

import itertools
import math
import string
from typing import TYPE_CHECKING, Any, Final, NamedTuple, NoReturn, TypeAlias, cast

import numpy as np
from scipy import sparse

from .._backend import Backend, active_backend, available_backends
from ..grid import HierarchicalGrid, hierarchical_grid, tensor_product_grid
from ..tolerance import get_conservative, get_strict
from ._bspline_knot_insertion_core import _compute_oslo_matrix_1d_core
from ._bspline_space_nd import BsplineSpace, _stored_dtype
from ._bspline_space_nd import _impl_class as _space_impl_class
from ._thb_eval_core import _combine_tp_values

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    import numpy.typing as npt

    from .._pantr_cpp import THBSplineSpace32 as _CppTHB32
    from .._pantr_cpp import THBSplineSpace64 as _CppTHB64
    from ._bspline_space_1d import BsplineSpace1D

    _Impl: TypeAlias = "_THBSplineSpacePython | _CppTHB32 | _CppTHB64"
    """The implementation a :class:`THBSplineSpace` holds: the oracle, or a handle.

    Type-checking only, and the same alias :mod:`pantr.bspline._bspline_space_nd`
    declares for the tensor-product case: the three are unrelated nominal types that
    happen to offer the same surface, which is the port's whole claim.
    """

_Support1D = tuple[
    "npt.NDArray[np.int64]",
    "npt.NDArray[np.int64]",
    "npt.NDArray[np.int64]",
]
"""Per-direction function support at one level.

``(first_basis_per_interval, first_cell_per_function, last_cell_per_function)``,
all ``int64`` arrays.
"""


class _TruncCoeffs(NamedTuple):
    """Stored representation of a truncated function.

    ``rep_level`` is the finest level at which the function is expressed.
    ``box_lo[k]`` is the per-direction lower function index of the coefficient
    box; ``coeffs.shape[k] == box_hi[k] - box_lo[k]`` (``box_hi`` is implicit
    in the array shape).  ``coeffs`` holds the function's coefficients in the
    level-``rep_level`` tensor-product basis.
    """

    rep_level: int
    box_lo: tuple[int, ...]
    coeffs: npt.NDArray[np.float64]


class _BasisEval1D(NamedTuple):
    """Cached result of a single 1D basis evaluation.

    ``values`` has shape ``(num_pts, degree + 1)``; ``first_basis`` has shape
    ``(num_pts,)``.  Both come from a single call to
    :meth:`~pantr.bspline.BsplineSpace1D.tabulate_basis`.
    """

    values: npt.NDArray[np.float64]
    first_basis: npt.NDArray[np.int64]


_EvalCache = dict[tuple[int, int, int], _BasisEval1D]
"""Per-call cache of 1D basis evaluations keyed by ``(level, direction, order)``.

``order`` is the derivative order evaluated in that direction (``0`` for values).
"""

_EINSUM_MAX_DIM = 24
"""Maximum parametric dimension supported by the single-letter einsum subscript scheme.

``string.ascii_lowercase`` provides 26 letters; the einsum needs ``dim`` letters for the
coefficient axes plus one for the point axis, leaving a safe ceiling of 24 dimensions.
"""

_CELL_MEMBERSHIP_SAFETY: Final[float] = 2.0
"""Extra factor on the strict tier for deciding that a point lies in a cell.

Together with :func:`~pantr.tolerance.get_strict` this is ``8 * eps``, the same rule and
the same number as :func:`~pantr.bspline._bspline_knots._knot_tolerance` -- deliberately,
because it is the same question. A cell boundary is a parametric coordinate, and asking
whether a point has fallen off it is asking whether two parametric coordinates are the
same one.

What has to be covered is the caller's route to the point. A point *in* cell
``[lo, hi]`` is almost always formed as ``lo + xi * (hi - lo)`` for some ``xi`` in
``[0, 1]``: a subtraction, a multiplication and an addition, three roundings, each on a
quantity no larger than ``max(|lo|, |hi|)``, so the computed point sits within
``3 * eps * max(|lo|, |hi|)`` of the exact one. The bounds themselves arrive from
:meth:`~pantr.grid.HierarchicalGrid.cell_bounds`, which subdivides the axis by the same
kind of arithmetic and at the same magnitude. Four epsilons cover either route and eight
covers both plus an FMA contraction, which is the doubling
:data:`~pantr.bspline._bspline_knots._KNOT_MERGE_SAFETY` applies for the same reason.

The magnitude it multiplies is the **cell's own**, not the axis's: see
:func:`_cell_membership_tolerance`.
"""


def _cell_membership_tolerance(
    cell_lo: npt.NDArray[np.float64], cell_hi: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Get the per-axis absolute slack allowed at a cell boundary.

    ``_CELL_MEMBERSHIP_SAFETY * get_strict(float64) * max(|lo|, |hi|, hi - lo)`` per
    axis, i.e. ``8 * eps`` times the cell's own magnitude. The magnitude is exactly
    what :func:`~pantr.bspline._bspline_knots._knot_scale` computes, applied to the
    two-element vector ``[lo, hi]`` rather than to a whole knot vector.

    The scale is the **cell's**, not the axis's, and the difference is not cosmetic.
    A point in the cell is formed from ``lo`` and ``hi``, so its round-off is relative
    to those two coordinates; a cell of width ``1e-3`` on a domain reaching ``1e6``
    would, on the axis scale, accept a point half a billion cell-widths outside. The
    coordinate terms ``|lo|`` and ``|hi|`` are what carry the case of a *narrow* cell
    far from the origin, where the width alone would demand agreement the arithmetic
    that produced the bounds cannot deliver.

    No floor is applied, for the reason
    :func:`~pantr.bspline._bspline_knots._knot_scale` gives: a floor of one would be a
    physical choice this layer is not entitled to make, and it destroys covariance on
    a domain smaller than one unit.

    The dtype is ``float64`` and not the root space's, which is where this departs from
    :func:`~pantr.bspline._bspline_knots._knot_tolerance`'s ``get_strict(knots.dtype)``.
    The reason is that it grades different objects: ``cell_bounds`` returns ``float64``
    whatever the root space is made of, and :meth:`THBSplineSpace._tabulate_orders` casts
    the query points to ``float64`` before the comparison, so ``float64`` *is* the
    precision of both sides here. A caller whose points carry only ``float32``
    information should widen them before asking, rather than have the containment check
    widened by eight million for everyone.

    Args:
        cell_lo (npt.NDArray[np.float64]): Per-axis lower cell bound, shape ``(dim,)``.
        cell_hi (npt.NDArray[np.float64]): Per-axis upper cell bound, same shape.

    Returns:
        npt.NDArray[np.float64]: Per-axis absolute tolerance, shape ``(dim,)``, in the
        units of the parametric coordinates.
    """
    scale = np.maximum(np.maximum(np.abs(cell_lo), np.abs(cell_hi)), cell_hi - cell_lo)
    return np.asarray(_CELL_MEMBERSHIP_SAFETY * get_strict(np.float64) * scale, dtype=np.float64)


def _prolongation_residual_tolerance(max_coarse_val: float) -> float:
    """Get the residual above which a coarse column is not reproduced by the fine basis.

    What is graded is ``max_i |A x - b|`` over every column of the prolongation, where
    ``b`` is a coarse function's coefficient vector in a common tensor-product level and
    ``A``'s columns are the candidate fine functions' vectors in the same basis. All the
    entries are refinement (Oslo) coefficients, so the quantity is dimensionless and its
    natural size is ``max_coarse_val = ||b||_inf`` -- measured to be exactly ``1.0`` for
    every genuine refinement, the partition-of-unity value. The ``1 +`` is the floor that
    keeps the threshold positive for a coarse function whose column is empty.

    ``np.linalg.lstsq(rcond=None)`` dispatches to LAPACK's ``gelsd``, whose normwise
    backward stability is the standard result for an SVD-based least-squares solve
    (Golub & Van Loan, *Matrix Computations*, 4th ed., section 5.5; Higham, *Accuracy and
    Stability of Numerical Algorithms*, 2nd ed., chapter 20): the computed ``x_hat`` is
    the exact least-squares solution of ``(A + dA) y ~= b + db`` with
    ``||dA|| <= c eps ||A||`` and ``||db|| <= c eps ||b||``.

    The step from that to a residual bound is worth spelling out, because ``x_hat`` does
    *not* satisfy the perturbed system exactly -- a least-squares solution leaves its own
    residual ``r_pert``. Writing ``x`` for the exact solution of the unperturbed system,
    which for a genuine refinement is consistent (``A x = b``), and using ``x`` as a trial
    vector for the perturbed minimisation:

        ||r_pert|| <= ||(A + dA) x - (b + db)|| = ||dA x - db|| <= ||dA|| ||x|| + ||db||,
        ||A x_hat - b|| <= ||r_pert|| + ||dA|| ||x_hat|| + ||db||
                        <= 2 (||dA|| max(||x||, ||x_hat||) + ||db||),

    which is ``O(c eps (||A|| ||x|| + ||b||))``. The factor of two is that argument's, not
    a safety pad. Every factor on the right is of order one here: the entries of ``A``,
    ``x`` and ``b`` are all refinement coefficients in ``[0, 1]``. The residual is
    therefore a small multiple of ``eps``, and ``c`` is the part no closed form is
    available for. The code grades ``max_i |A x - b|``, an infinity norm, which is at most
    the two-norm the bound is stated in, so the bound covers it.

    **Measured**, over 132 configurations (1D, 2D and 3D; degrees 2 to 5; 4 to 16
    elements per axis; truncated and non-truncated; one and two refinement levels;
    domains ``[0, 1]``, ``[0, 1e-6]`` and ``[1e6, 1e6 + 1]``): the worst residual is
    ``18.5 * eps``. It came out bit-identical between ``[0, 1]`` and ``[1e6, 1e6 + 1]``
    in every one of those configurations, which is a stronger statement than the
    quantity's dimensionlessness requires -- that only forces the residual to be
    *comparable* across scales, not bitwise equal -- so it is reported as observed and
    nothing is built on it.

    The tier is :func:`~pantr.tolerance.get_conservative`, ``4096 * eps``, whose stated
    meaning is a long accumulation -- which an SVD-based least-squares solve is. Against
    the worst measured that is a safety factor of ``4096 * 2 / 18.5 = 443``, and it is
    what stands in for the unknown ``c``.

    A ``sqrt(M)`` term with ``M = A.shape[0]`` was considered, since ``||b||_2`` carries
    one, and **rejected on measurement**: over the same runs ``M`` ranges from 13 to
    19683 and the residual does not track it (``res / eps`` is 2.0 at ``M = 13`` and 16.5
    at ``M = 19683``; normalizing by ``sqrt(M)`` makes the spread *worse*, from 3.10 down
    to 0.10). Carrying it would loosen the gate 140-fold on the largest systems for a
    growth that is not there. If one ever appears, that term is where it belongs.

    Args:
        max_coarse_val (float): Largest absolute coefficient over every coarse column,
            ``||b||_inf``.

    Returns:
        float: The residual threshold, in the coefficients' own dimensionless units.
    """
    return get_conservative(np.float64) * (1.0 + max_coarse_val)


def _check_out_array(
    out: npt.NDArray[np.float64] | npt.NDArray[np.int64],
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    name: str,
) -> None:
    """Validate an output array's shape, dtype, and writeability.

    Args:
        out (npt.NDArray[np.float64] | npt.NDArray[np.int64]): The output array.
        shape (tuple[int, ...]): The required shape.
        dtype (npt.DTypeLike): The required dtype.
        name (str): The parameter name, used in error messages.

    Raises:
        ValueError: If ``out`` has the wrong shape or dtype, or is not writeable.
    """
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {out.shape}.")
    if out.dtype != dtype:
        raise ValueError(f"{name} must have dtype {np.dtype(dtype).name}; got {out.dtype}.")
    if not out.flags.writeable:
        raise ValueError(f"{name} must be writeable.")


def _box_all_true(
    mask: npt.NDArray[np.bool_],
    lo: npt.NDArray[np.int64],
    hi: npt.NDArray[np.int64],
) -> npt.NDArray[np.bool_]:
    """Test, for a batch of axis-aligned boxes, whether ``mask`` is all-``True`` inside.

    Each box ``b`` spans the half-open range ``[lo[b, d], hi[b, d])`` per axis ``d``.
    A summed-area table over ``~mask`` makes each box's all-``True`` test
    (``mask[box].all()`` ⟺ no ``False`` cell in the box) an O(``2**ndim``) lookup per box.
    Table construction is O(``N * ndim``) where ``N`` is the total cell count of ``mask``.

    Args:
        mask (npt.NDArray[np.bool_]): The ``ndim``-dimensional boolean mask.
        lo (npt.NDArray[np.int64]): Box lower corners, shape ``(n_boxes, ndim)``.
        hi (npt.NDArray[np.int64]): Box upper corners (exclusive), shape ``(n_boxes, ndim)``.

    Returns:
        npt.NDArray[np.bool_]: Shape ``(n_boxes,)``; ``True`` where the box is all-``True``.
    """
    dim = mask.ndim
    prefix = np.pad((~mask).astype(np.int64), [(1, 0)] * dim)
    for ax in range(dim):
        prefix = np.cumsum(prefix, axis=ax)
    total = np.zeros(lo.shape[0], dtype=np.int64)
    for corner in itertools.product((0, 1), repeat=dim):
        idx = tuple(hi[:, d] if corner[d] else lo[:, d] for d in range(dim))
        sign = (-1) ** (dim - sum(corner))
        total = total + sign * prefix[idx]
    return np.asarray(total == 0, dtype=np.bool_)


def _func_support_1d(space: BsplineSpace1D) -> _Support1D:
    """Compute the cell support of every B-spline function of a 1D space.

    The first non-zero function index per interval comes from
    :meth:`~pantr.bspline.BsplineSpace1D.first_basis_per_interval`, which is then
    inverted to give, for each function ``i``, the inclusive interval (cell) range
    ``[first_cell, last_cell]`` it is supported on.

    Args:
        space (BsplineSpace1D): The 1D B-spline space.

    Returns:
        _Support1D: ``(first_basis, first_cell, last_cell)`` where ``first_basis``
        has length ``num_intervals`` and ``first_cell`` / ``last_cell`` have length
        ``num_basis``.
    """
    first_basis = space.first_basis_per_interval()

    degree = space.degree
    n_basis = space.num_basis
    first_cell = np.full(n_basis, -1, dtype=np.int64)
    last_cell = np.full(n_basis, -1, dtype=np.int64)
    for interval in range(first_basis.shape[0]):
        lo_i = int(first_basis[interval])
        for i in range(lo_i, lo_i + degree + 1):
            if first_cell[i] < 0:
                first_cell[i] = interval
            last_cell[i] = interval
    if not np.all(first_cell >= 0):
        raise RuntimeError(
            f"B-spline function(s) with empty support detected at indices "
            f"{np.where(first_cell < 0)[0].tolist()}. This indicates an invalid B-spline space."
        )
    return first_basis, first_cell, last_cell


def _refine_box(
    coeffs: npt.NDArray[np.float64],
    box_lo: list[int],
    box_hi: list[int],
    oslo_m: tuple[npt.NDArray[np.float64], ...],
) -> tuple[npt.NDArray[np.float64], list[int], list[int]]:
    """Refine a dense coefficient box from one level to the next.

    Applies, per direction, the two-scale matrix restricted to the current
    function box, growing the box to the band of non-zero finer functions.

    Module-level rather than a method because both sides of the wrapper need it:
    :meth:`_THBSplineSpacePython._compute_truncated_coeffs` builds the stored boxes
    with it, and :meth:`THBSplineSpace._finest_tp_coeffs` replays the same refinement
    for the prolongation, which has no C++ counterpart and therefore cannot reach the
    oracle's own copy.

    Args:
        coeffs (npt.NDArray[np.float64]): Coefficients over the current box.
        box_lo (list[int]): Per-direction lower function index of the box.
        box_hi (list[int]): Per-direction upper (exclusive) function index.
        oslo_m (tuple[npt.NDArray[np.float64], ...]): Per-direction two-scale
            matrices for this level transition.

    Returns:
        tuple[npt.NDArray[np.float64], list[int], list[int]]: Refined
        coefficients and fresh lists ``(box_lo, box_hi)`` for the next
        level; the input lists are not modified.

    Raises:
        ValueError: If the Oslo matrix slice for any direction is entirely
            zero, indicating a degenerate box or invalid knot refinement.
    """
    new_lo = list(box_lo)
    new_hi = list(box_hi)
    out = coeffs
    for k in range(out.ndim):
        alpha = oslo_m[k]
        cols = alpha[:, box_lo[k] : box_hi[k]]
        rows = np.nonzero(np.any(cols != 0.0, axis=1))[0]
        if rows.size == 0:
            raise ValueError(
                f"_refine_box: Oslo matrix slice for direction {k} "
                f"(columns [{box_lo[k]}:{box_hi[k]}]) is entirely zero — "
                "degenerate or invalid knot refinement."
            )
        nlo, nhi = int(rows[0]), int(rows[-1]) + 1
        sub = alpha[nlo:nhi, box_lo[k] : box_hi[k]]
        contracted = np.tensordot(sub, out, axes=([1], [k]))
        out = np.moveaxis(contracted, 0, k)
        new_lo[k], new_hi[k] = nlo, nhi
    return out, new_lo, new_hi


def _build_oslo_matrices(
    level_spaces: Sequence[BsplineSpace],
) -> tuple[tuple[npt.NDArray[np.float64], ...], ...]:
    """Build the per-direction two-scale (Oslo) matrices between consecutive levels.

    Entry ``[m][k]`` is the refinement matrix ``alpha`` of shape
    ``(num_basis_{m+1,k}, num_basis_{m,k})`` such that a level-``m`` B-spline
    ``B_i`` equals ``sum_j alpha[j, i] B_j`` in the level-``(m+1)`` basis (the
    identity when ``factor[k] == 1``).

    Takes the level spaces rather than a THB space, for the reason :func:`_refine_box`
    gives: the oracle holds them as a tuple and the wrapper reaches them one
    :meth:`~THBSplineSpace.level_space` call at a time, and both need the same
    matrices.

    Args:
        level_spaces (Sequence[BsplineSpace]): The per-level tensor-product spaces,
            in level order.

    Returns:
        tuple[tuple[npt.NDArray[np.float64], ...], ...]: Matrices indexed by
        ``[m][k]`` for ``m`` in ``[0, len(level_spaces) - 2]``.
    """
    mats: list[tuple[npt.NDArray[np.float64], ...]] = []
    for m in range(len(level_spaces) - 1):
        per_dir: list[npt.NDArray[np.float64]] = []
        for old, new in zip(level_spaces[m].spaces, level_spaces[m + 1].spaces, strict=True):
            alpha = _compute_oslo_matrix_1d_core(old.degree, old.knots, new.knots)
            per_dir.append(np.asarray(alpha, dtype=np.float64))
        mats.append(tuple(per_dir))
    return tuple(mats)


def _impl_class(dtype: np.dtype[Any]) -> type[_THBSplineSpacePython] | type[Any]:
    """The implementation class the active backend and the dtype select.

    The backend is per process rather than per instance, for the reason
    :func:`pantr.bspline._bspline_space_nd._impl_class` gives. It bites here through
    :meth:`THBSplineSpace.refine`, :meth:`THBSplineSpace.refine_region` and
    :meth:`THBSplineSpace.coarsen`, each of which returns a space: a per-instance
    choice would let a derived space cross implementations, which is the
    reconciliation ``design/cross_backend_types.md`` forbids.

    Args:
        dtype (np.dtype[Any]): The root space's storage format.

    Returns:
        type: The oracle under the Python backend, and the C++ class for that
        storage format otherwise.

    Raises:
        RuntimeError: If the C++ backend is requested and is not available.
    """
    if active_backend() is Backend.PYTHON:
        return _THBSplineSpacePython
    if Backend.CPP not in available_backends():
        raise RuntimeError("the CPP backend is not available in this installation")
    from pantr import _pantr_cpp  # noqa: PLC0415  (optional, imported only when selected)

    if dtype == np.float32:
        return _pantr_cpp.THBSplineSpace32
    return _pantr_cpp.THBSplineSpace64


def _new_impl(
    root_space: BsplineSpace,
    grid: HierarchicalGrid,
    truncate: bool,
    regularity: Sequence[int | None],
    dtype: np.dtype[Any],
) -> _Impl:
    """Build a hierarchical space in whichever implementation the backend selects.

    The two implementations take their nested objects differently, and the difference
    is the whole reason this function exists rather than one expression. The C++ class
    takes the root space's and the grid's **implementations**, which is what makes it
    share them; the oracle takes the **wrappers**, because its own body is written
    against their public surfaces -- ``grid.subdomain_mask``, ``space.subdivide`` and a
    dozen more -- and rewriting two thousand lines of the parity oracle to work one
    layer down would risk the very thing the oracle exists to be a fixed point for.

    That asymmetry is what makes the cross-backend check below necessary, and it is
    not symmetric either. Handing a Python oracle to the C++ class raises a nanobind
    ``TypeError`` naming C++ types, which is loud but unreadable; handing a C++ handle
    to the oracle **succeeds**, and yields a hybrid whose hierarchical logic runs in
    Python over C++ values, which no parity claim covers and nothing announces.
    ``design/cross_backend_types.md`` forbids exactly that second shape, so both
    directions are refused here with one message.

    Args:
        root_space (BsplineSpace): The level-0 tensor-product space.
        grid (HierarchicalGrid): The active-cell hierarchy.
        truncate (bool): Whether to build the truncated (THB) basis.
        regularity (Sequence[int | None]): Per-direction continuity, already
            broadcast to one entry per direction by :class:`THBSplineSpace`.
        dtype (np.dtype[Any]): The root space's storage format.

    Returns:
        _Impl: The implementation object; an oracle instance or a C++ handle.

    Raises:
        ValueError: If ``root_space`` or ``grid`` was built under a different backend.
        RuntimeError: If the C++ backend is requested and is not available.
    """
    cls = _impl_class(dtype)
    if not isinstance(root_space._impl, _space_impl_class(dtype)):
        raise ValueError(
            "root_space must come from the active backend; it was built under a different one."
        )
    if not isinstance(grid._impl, _grid_impl_class()):
        raise ValueError(
            "grid must come from the active backend; it was built under a different one."
        )
    if cls is _THBSplineSpacePython:
        return _THBSplineSpacePython(root_space, grid, truncate, regularity)
    # `cls` is one of the C++ classes here. The checker cannot narrow an identity test
    # on a `type[...]` union, so it still admits the oracle and then reports the
    # wrapper types as wrong for the handle parameters.
    cpp_cls: Any = cls
    return cast("_Impl", cpp_cls(root_space._impl, grid._impl, truncate, list(regularity)))


def _grid_impl_class() -> type[Any]:
    """The hierarchical-grid implementation class the active backend builds.

    :mod:`pantr.grid` offers no per-dtype ``_impl_class`` the way the B-spline modules
    do -- its hierarchy is ``float64``-only -- so the pair is reached here instead. The
    imports are lazy for the reason every ``_pantr_cpp`` import in the tree is: the
    extension is optional, and the oracle class is only needed on the branch that
    names it.

    Returns:
        type[Any]: ``pantr.grid._hierarchical_grid._HierarchicalGridPython`` under the
        Python backend, and the bound ``pantr._pantr_cpp.HierarchicalGrid`` otherwise.

    Raises:
        RuntimeError: If the C++ backend is requested and is not available.
    """
    if active_backend() is Backend.PYTHON:
        from ..grid._hierarchical_grid import (  # noqa: PLC0415  (lazy, one branch only)
            _HierarchicalGridPython,
        )

        return _HierarchicalGridPython
    if Backend.CPP not in available_backends():
        raise RuntimeError("the CPP backend is not available in this installation")
    from pantr import _pantr_cpp  # noqa: PLC0415  (optional, imported only when selected)

    return _pantr_cpp.HierarchicalGrid


def _supported_functions(
    space: _THBSplineSpacePython | THBSplineSpace, cid: int
) -> list[tuple[int, int, tuple[int, ...]]]:
    """Return the active functions whose tensor-product support covers cell ``cid``.

    A superset of :meth:`THBSplineSpace.contributions`: it also lists truncated
    functions that vanish on the cell.  Its use is a support closure, where such a
    function still matters because its Kraft status shapes the truncation of the
    others.  Not cached.

    Module-level, and reading only the three private accessors both classes offer under
    the same names, because the oracle needs it to build its own contribution table
    while the wrapper needs it for the halo closure -- and the C++ type has no
    counterpart to forward to, so a second copy on the wrapper would be the duplication
    the port exists to remove.

    Args:
        space (_THBSplineSpacePython | THBSplineSpace): The space to read.
        cid (int): Active cell flat id in ``[0, grid.num_cells)``.

    Returns:
        list[tuple[int, int, tuple[int, ...]]]: ``(global_dof, level, multi)`` triples
        sorted by ``global_dof``.

    Raises:
        IndexError: If ``cid`` is out of range ``[0, grid.num_cells)``.
    """
    grid = space.grid
    cell_level = grid.cell_level(cid)
    cell_midx = grid.cell_multi_index(cid)
    factor = grid.factor
    dim = space.dim
    degrees = space.degrees
    offsets = space._offsets()
    contribs: list[tuple[int, int, tuple[int, ...]]] = []
    for level in range(cell_level + 1):
        divisor = tuple(factor[k] ** (cell_level - level) for k in range(dim))
        cell_at_level = tuple(cell_midx[k] // divisor[k] for k in range(dim))
        num_basis = space.level_space(level).num_basis
        support = space._level_support(level)
        ranges = []
        for k in range(dim):
            first_basis = support[k][0]
            f0 = int(first_basis[cell_at_level[k]])
            ranges.append(range(f0, f0 + degrees[k] + 1))
        active_at_level = space._active_at(level)
        offset = int(offsets[level])
        for multi in itertools.product(*ranges):
            flat = int(np.ravel_multi_index(multi, num_basis))
            pos = int(np.searchsorted(active_at_level, flat))
            if pos < active_at_level.shape[0] and int(active_at_level[pos]) == flat:
                contribs.append((offset + pos, level, multi))
    contribs.sort(key=lambda triple: triple[0])
    return contribs


def _cell_ids(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Coerce an ``ArrayLike`` of cell or corner indices into what the bindings take.

    Python's calling convention rather than validation. The C++ entry points declare
    their index arguments ``noconvert()``, so they take a contiguous ``int64`` array
    and nothing else; the oracle would accept anything :func:`numpy.asarray` swallows.
    Converting here is what keeps the two answering the same call, and it is also where
    a nested sequence is flattened -- both implementations read a flat run of indices.

    The conversion is deliberately :func:`numpy.asarray`'s and not a stricter one: that
    is what the Python implementation has always done for these arguments, so tightening
    it here would change behaviour under both backends in a ticket that is wiring, not
    re-deciding.

    Args:
        values (npt.ArrayLike): Flat cell ids, or a per-direction corner.

    Returns:
        npt.NDArray[np.int64]: A contiguous 1D ``int64`` array.
    """
    return np.ascontiguousarray(np.asarray(values, dtype=np.int64).ravel())


def _as_grid(value: object, root: object) -> HierarchicalGrid:
    """Present an implementation's grid as the public wrapper.

    The two backends hand back different things from ``impl.grid``. The oracle is
    built on the public types, so its grid is already a
    :class:`~pantr.grid.HierarchicalGrid`; the C++ space owns a
    ``pantr::grid::HierarchicalGrid`` and its binding hands back the raw handle. Both
    have to reach the caller as the one public class, and this is the single place
    that reconciles them -- the same shape, and for the same reason, as
    ``pantr.grid._grid._grid_value``.

    Written as "already the right type, or adopt it" rather than as a test on which
    backend is active, because that is the actual question and it stays true if a
    third implementation ever appears. It is scaffolding all the same: it exists
    because the oracle exists, and it goes when that does.

    Args:
        value (object): What the implementation returned for its grid.
        root (object): The level-0 :class:`~pantr.grid.TensorProductGrid` to seed the
            new wrapper's root memo with, so that a derived grid keeps the root
            object -- and its tags -- the space was built over.

    Returns:
        HierarchicalGrid: ``value`` if it already is one, otherwise a wrapper
        adopting it.
    """
    if isinstance(value, HierarchicalGrid):
        return value
    return HierarchicalGrid._wrap_over(cast("Any", value), cast("Any", root))


class _THBSplineSpacePython:
    r"""The pure-Python hierarchical B-spline space: the port's parity oracle.

    Reached only through :class:`THBSplineSpace`, which is the class a caller holds.
    Its public surface is **exactly** ``pantr::bspline::THBSplineSpace``'s bound one,
    member for member and message for message, because that is what lets the wrapper
    forward without asking which implementation it holds. Anything that is *not* on
    the C++ type is private here, and anything the C++ type has but this one lacks
    would be a forward with nothing to call.

    Unlike :class:`pantr.bspline._bspline_space_nd._BsplineSpaceNDPython`, this oracle
    holds the root space and the grid as **wrappers** rather than as their
    implementations. The module docstring and :func:`_new_impl` carry the reason and
    the check that makes it safe.

    Built from a root :class:`~pantr.bspline.BsplineSpace` (level 0) and a
    :class:`~pantr.grid.HierarchicalGrid` carrying the active-cell hierarchy.  The
    per-level tensor-product spaces are obtained by uniformly subdividing the root
    space according to the grid's per-direction ``factor``.  The active hierarchical
    basis is the Kraft selection :cite:p:`kraft1997hierarchical,vuong2011hierarchical`:
    a level-``l`` tensor-product B-spline is active iff its support lies in the
    level-``l`` subdomain :math:`\Omega_l` but not entirely in the finer subdomain
    :math:`\Omega_{l+1}`.

    With ``truncate=True`` (the default) the *truncated* hierarchical basis (THB) is
    built: each active function that straddles a finer-level refinement boundary has its
    components on active finer functions removed (Giannelli-Jüttler-Speleers truncation
    :cite:p:`giannelli2012thb`), restoring the partition of unity.  Only truncated
    functions store a coefficient vector (in the finest tensor-product basis their support
    reaches); untruncated functions remain plain tensor-product B-splines.  With
    ``truncate=False`` the non-truncated hierarchical basis (HB) is built.

    A :class:`~pantr.grid.HierarchicalGrid` is immutable, so this space cannot go
    stale: :meth:`~pantr.grid.HierarchicalGrid.refine` returns a *new* grid and leaves
    the one held here untouched.  :meth:`refine`, :meth:`refine_region` and
    :meth:`coarsen` are the same shape one level up -- each returns a new space over a
    grid of its own, including when it refines or coarsens nothing.

    Note:
        A function is active on a cell when its tensor-product support covers the cell
        and it does not vanish identically there.  Under truncation a coarse function can
        vanish on cells inside the refined region; :meth:`active_basis`,
        :meth:`tabulate_basis` and :meth:`max_active_per_cell` omit it on those cells.

    References:
        Adaptive isogeometric algorithms for hierarchical splines
        :cite:p:`garau2018algorithms`.  Per-element multi-level Bézier extraction
        (used for element assembly and visualization) is provided by
        :class:`~pantr.bspline.MultiLevelExtraction`, following
        :cite:t:`dangella2018multilevel`.

    Attributes:
        _root_space (BsplineSpace): The level-0 tensor-product space.
        _grid (HierarchicalGrid): The active-cell hierarchy.  Immutable, and never
            shared with another space's grid: the grid's tag registries and BVH memo
            are mutable even though its cell decomposition is not.
        _truncate (bool): Whether the truncated (THB) basis is used; ``False`` for
            the plain hierarchical (HB) basis.
        _regularity (tuple[int | None, ...]): Per-direction continuity used when
            subdividing to build finer levels.
        _level_spaces (tuple[BsplineSpace, ...]): Per-level tensor-product spaces;
            index ``l`` is the root subdivided to level ``l``.
        _support (tuple): Per-level, per-direction function-to-cell support
            arrays; ``_support[level][k]`` is the
            ``(first_basis, first_cell, last_cell)`` int64 triple
            (a ``_Support1D``) for direction ``k`` at ``level``.
        _active_funcs (tuple[npt.NDArray[np.int64], ...]): Per-level sorted flat
            (C-order) indices of the active tensor-product functions.
        _func_offset (npt.NDArray[np.int64]): Per-level global-dof base; length
            ``num_levels + 1`` (cumulative active-function counts).
        _num_active (int): Total number of active hierarchical functions.
        _trunc (dict): Map from global dof (``int``) to ``_TruncCoeffs``;
            only truncated functions appear (empty when ``truncate=False``).
        _max_active_per_cell (int | None): Memoized :meth:`max_active_per_cell`
            result; ``None`` until first requested.
    """

    __slots__ = (
        "_active_funcs",
        "_contrib_cache",
        "_func_offset",
        "_grid",
        "_level_spaces",
        "_max_active_per_cell",
        "_num_active",
        "_regularity",
        "_root_space",
        "_support",
        "_trunc",
        "_truncate",
    )

    def __init__(
        self,
        root_space: BsplineSpace,
        grid: HierarchicalGrid,
        truncate: bool,
        regularity: Sequence[int | None],
    ) -> None:
        """Create a hierarchical B-spline space.

        Positional throughout and with ``regularity`` already broadcast, because the
        C++ constructor is, and :func:`_new_impl` calls the two with one call shape.
        :class:`THBSplineSpace` is what offers the keyword form and the scalar.

        Args:
            root_space (BsplineSpace): The level-0 tensor-product B-spline space.
            grid (HierarchicalGrid): Hierarchical grid whose root knot-span grid
                matches ``root_space``.
            truncate (bool): If ``True``, build the truncated (THB) basis; if
                ``False``, build the non-truncated hierarchical (HB) basis.
            regularity (Sequence[int | None]): Per-direction continuity at the knots
                inserted when subdividing to finer levels.  ``None`` in an entry uses
                maximal smoothness.  Each non-``None`` entry must satisfy
                ``-1 <= regularity[k] < degree[k]``.

        Raises:
            ValueError: If ``grid`` and ``root_space`` disagree on dimension or on
                the root knot-span grid, if ``regularity`` has the wrong length, or
                if any per-direction regularity value is out of range.
        """
        dim = root_space.dim
        if grid.ndim != dim:
            raise ValueError(f"grid.ndim ({grid.ndim}) must equal root_space.dim ({dim}).")
        if tuple(grid.root.cells_per_axis) != tuple(root_space.num_intervals):
            raise ValueError(
                f"grid root cells_per_axis {tuple(grid.root.cells_per_axis)!r} must match "
                f"root_space.num_intervals {tuple(root_space.num_intervals)!r}."
            )
        # Absolute comparison against the space's own knot tolerance.  Bare
        # ``np.allclose`` would apply numpy's defaults (``rtol=1e-5``, ``atol=1e-8``),
        # which accept a 3e-6 relative domain mismatch at every scale and, through the
        # atol leg, pass ``[0, 1e-9]`` against ``[0, 2e-9]`` -- a factor-of-two wrong
        # domain.
        if not np.all(
            np.abs(
                np.asarray(grid.root.bounds, dtype=np.float64)
                - np.asarray(root_space.domain, dtype=np.float64)
            )
            <= root_space.tolerance
        ):
            raise ValueError("grid root bounds must match root_space domain.")

        reg = tuple(regularity)
        if len(reg) != dim:
            raise ValueError(
                f"regularity must be a scalar or length-{dim} sequence; got length {len(reg)}."
            )
        for k, (r, d) in enumerate(zip(reg, root_space.degrees, strict=False)):
            if r is not None and not (-1 <= r < d):
                raise ValueError(
                    f"regularity[{k}]={r!r} must be in [-1, degree[{k}]-1={d - 1}]; got {r!r}."
                )

        self._root_space = root_space
        self._grid = grid
        self._truncate = truncate
        self._regularity = reg

        self._level_spaces = self._build_level_spaces()
        self._support = tuple(
            tuple(_func_support_1d(sp1d) for sp1d in level_space.spaces)
            for level_space in self._level_spaces
        )
        self._active_funcs = self._select_active_functions()
        for block in self._active_funcs:
            block.flags.writeable = False
        counts = [int(a.shape[0]) for a in self._active_funcs]
        self._func_offset = np.concatenate(([0], np.cumsum(counts, dtype=np.int64))).astype(
            np.int64
        )
        self._func_offset.flags.writeable = False
        self._num_active = int(self._func_offset[-1])
        self._trunc = self._compute_truncated_coeffs() if truncate else {}
        # Lazy per-cell cache of `contributions` (populated on first access). The space
        # is an immutable construction-time snapshot, so cached results stay valid.
        self._contrib_cache: dict[
            int, tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]
        ] = {}
        self._max_active_per_cell: int | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_level_spaces(self) -> tuple[BsplineSpace, ...]:
        """Build the nested per-level tensor-product spaces.

        Level ``l + 1`` is obtained from level ``l`` by subdividing every 1D space
        by the grid ``factor`` (skipping axes whose factor is ``1``), which keeps
        the level spaces nested.

        Returns:
            tuple[BsplineSpace, ...]: Spaces of length ``num_levels``.
        """
        factor = self._grid.factor
        reg = self._regularity
        current = list(self._root_space.spaces)
        level_spaces: list[BsplineSpace] = [self._root_space]
        for _ in range(1, self._grid.max_level + 1):
            current = [
                sp if factor[k] == 1 else sp.subdivide(factor[k], reg[k])
                for k, sp in enumerate(current)
            ]
            level_spaces.append(BsplineSpace(current))
        return tuple(level_spaces)

    def _select_active_functions(self) -> tuple[npt.NDArray[np.int64], ...]:
        r"""Compute the Kraft selection of active functions per level.

        A level-``l`` tensor-product function is selected iff its support lies
        entirely in :math:`\Omega_l` (``subdomain_mask``) but not entirely in the
        further-refined region (``subdomain_mask & ~active_leaf_mask``).

        Returns:
            tuple[npt.NDArray[np.int64], ...]: Per-level sorted flat (C-order)
            indices of the active functions.
        """
        dim = self.dim
        active: list[npt.NDArray[np.int64]] = []
        for level in range(self.num_levels):
            num_basis = self._level_spaces[level].num_basis
            subdomain = self._grid.subdomain_mask(level)
            refined = subdomain & ~self._grid.active_leaf_mask(level)
            support = self._support[level]

            true_coords = np.argwhere(subdomain)
            if true_coords.shape[0] == 0:
                active.append(np.empty(0, dtype=np.int64))
                continue
            bbox_lo = true_coords.min(axis=0)
            bbox_hi = true_coords.max(axis=0) + 1

            candidates_per_dir: list[npt.NDArray[np.int64]] = []
            for k in range(dim):
                _, first_cell, last_cell = support[k]
                overlaps = (last_cell >= bbox_lo[k]) & (first_cell < bbox_hi[k])
                candidates_per_dir.append(np.nonzero(overlaps)[0].astype(np.int64))

            # Enumerate candidate multi-indices and batch the support-box all-checks via
            # a summed-area table: selected iff the support box lies entirely in the
            # subdomain (Ω_l) but not entirely in the further-refined region.
            mesh = np.meshgrid(*candidates_per_dir, indexing="ij")
            multis = np.stack([m.ravel() for m in mesh], axis=-1)  # (n_cand, dim)
            box_lo = np.empty_like(multis)
            box_hi = np.empty_like(multis)
            for k in range(dim):
                _, first_cell, last_cell = support[k]
                box_lo[:, k] = first_cell[multis[:, k]]
                box_hi[:, k] = last_cell[multis[:, k]] + 1
            in_subdomain = _box_all_true(subdomain, box_lo, box_hi)
            in_refined = _box_all_true(refined, box_lo, box_hi)
            selected = multis[in_subdomain & ~in_refined]
            flats = np.ravel_multi_index([selected[:, k] for k in range(dim)], num_basis)
            active.append(np.sort(flats).astype(np.int64))
        return tuple(active)

    @staticmethod
    def _truncate_box(
        coeffs: npt.NDArray[np.float64],
        box_lo: list[int],
        box_hi: list[int],
        active_at_level: npt.NDArray[np.int64],
        num_basis: tuple[int, ...],
    ) -> bool:
        """Zero basis coefficients at active-function positions (in place); report if any zeroed.

        Args:
            coeffs (npt.NDArray[np.float64]): Coefficients over the box (modified in
                place).
            box_lo (list[int]): Per-direction lower function index of the box.
            box_hi (list[int]): Per-direction upper (exclusive) function index.
            active_at_level (npt.NDArray[np.int64]): Sorted flat indices of the active
                functions at the refined level (the level whose basis ``coeffs`` is
                expressed in).
            num_basis (tuple[int, ...]): Per-direction function counts at the refined
                level.

        Returns:
            bool: ``True`` iff at least one coefficient was zeroed.
        """
        ranges = [np.arange(box_lo[k], box_hi[k]) for k in range(coeffs.ndim)]
        mesh = np.meshgrid(*ranges, indexing="ij")
        flat = np.ravel_multi_index([m.ravel() for m in mesh], num_basis)
        is_active = np.isin(flat, active_at_level).reshape(coeffs.shape)
        if not bool(is_active.any()):
            return False
        coeffs[is_active] = 0.0
        return True

    def _compute_truncated_coeffs(self) -> dict[int, _TruncCoeffs]:
        """Build the truncated-coefficient map for the THB basis.

        For each active function that straddles a finer refinement boundary, the
        function is represented in successively finer bases (two-scale refinement),
        zeroing the components on active finer functions at each level (truncation),
        until its support no longer reaches deeper refinement.  Truncation is applied
        at each level in the support chain, not only the first; a function may be
        truncated against active sets at multiple levels before its support clears all
        refinement.  Untruncated functions are omitted.

        Returns:
            dict[int, _TruncCoeffs]: Map from global dof to ``(rep_level, box_lo,
            coeffs)`` for every truncated function.
        """
        trunc: dict[int, _TruncCoeffs] = {}
        if self.num_levels == 1:
            return trunc
        oslo = _build_oslo_matrices(self._level_spaces)
        refined = [
            self._grid.subdomain_mask(m) & ~self._grid.active_leaf_mask(m)
            for m in range(self.num_levels)
        ]
        dim = self.dim
        for level in range(self.num_levels - 1):
            num_basis = self._level_spaces[level].num_basis
            offset = int(self._func_offset[level])
            for pos, flat in enumerate(self._active_funcs[level].tolist()):
                multi = np.unravel_index(int(flat), num_basis)
                box_lo = [int(multi[k]) for k in range(dim)]
                box_hi = [int(multi[k]) + 1 for k in range(dim)]
                coeffs = np.ones((1,) * dim, dtype=np.float64)
                rep = level
                any_zeroed = False
                m = level
                while m + 1 < self.num_levels:
                    support_m = self._support[m]
                    cell_box = tuple(
                        slice(
                            int(support_m[k][1][box_lo[k]]),
                            int(support_m[k][2][box_hi[k] - 1]) + 1,
                        )
                        for k in range(dim)
                    )
                    if not bool(refined[m][cell_box].any()):
                        break
                    coeffs, box_lo, box_hi = _refine_box(coeffs, box_lo, box_hi, oslo[m])
                    m += 1
                    rep = m
                    zeroed = self._truncate_box(
                        coeffs,
                        box_lo,
                        box_hi,
                        self._active_funcs[m],
                        self._level_spaces[m].num_basis,
                    )
                    any_zeroed = any_zeroed or zeroed
                # A function whose support enters a refined region but whose
                # coefficient box at every finer level has no overlap with active
                # finer functions requires no truncation (remains a plain B-spline).
                if any_zeroed:
                    if len(box_lo) != coeffs.ndim:
                        raise RuntimeError(
                            f"_compute_truncated_coeffs: box_lo length {len(box_lo)} "
                            f"!= coeffs.ndim {coeffs.ndim}."
                        )
                    coeffs.flags.writeable = False
                    trunc[offset + pos] = _TruncCoeffs(rep, tuple(box_lo), coeffs)
        return trunc

    def _vanishes_on_cell(
        self, entry: _TruncCoeffs, cell_level: int, cell_midx: tuple[int, ...]
    ) -> bool:
        """Decide exactly whether a truncated function is identically zero on a cell.

        On the cell, the function is the sum of its stored coefficients against the
        level-``rep_level`` B-splines supported on the cell's level-``rep_level``
        descendants.  (``rep_level >= L`` for every function supported on a level-``L``
        cell: the cell's ancestors at the coarser levels are refined, so the truncation
        chain in :meth:`_compute_truncated_coeffs` does not stop before level ``L``.)
        Those B-splines are linearly independent on each descendant -- the ones supported
        on a non-empty knot span span the polynomials of degree ``p`` there -- so the
        stored function vanishes on the cell iff every stored coefficient in that window
        is zero.  That is what :meth:`tabulate_basis` evaluates, so the two agree with no
        further hypothesis.

        That a stored ``0.0`` is a true zero, and that both backends store the same zeros,
        needs two hypotheses:

        - every two-scale (Oslo) coefficient is nonnegative, which holds for any
          nondecreasing knot vector, any subdivision factor and any regularity, and
          truncation only sets entries to ``0.0``; so a stored coefficient is a
          cancellation-free sum of products and is ``0.0`` exactly when every product has
          a zero factor, whatever the summation order;
        - no product of positive two-scale coefficients along a chain underflows to zero;
          :func:`~pantr.bspline._multilevel_extraction_core._windowed_multilevel_rows`
          bounds the depth at which that can happen under uniform dyadic subdivision and
          states that no bound is derived for non-uniform knots or other
          factors.  If it did happen, the function would be dropped from the cell here,
          which is a stronger consequence than a zero row, and
          :class:`~pantr.bspline.MultiLevelExtraction` raises on the resulting mismatch.

        Args:
            entry (_TruncCoeffs): The function's stored representation.
            cell_level (int): Level ``L`` of the cell.
            cell_midx (tuple[int, ...]): Per-axis index of the cell at level ``L``.

        Returns:
            bool: ``True`` iff the function is identically zero on the cell.
        """
        rep = entry.rep_level
        window: list[slice] = []
        for k in range(self.dim):
            fine = self._grid.factor[k] ** rep
            coarse = self._grid.factor[k] ** cell_level
            # The level-`rep` descendants of the cell, as an inclusive range of indices.
            first_cell = cell_midx[k] * fine // coarse
            last_cell = ((cell_midx[k] + 1) * fine - 1) // coarse
            first_basis = self._support[rep][k][0]
            lo = int(first_basis[first_cell]) - entry.box_lo[k]
            hi = int(first_basis[last_cell]) + self.degrees[k] + 1 - entry.box_lo[k]
            window.append(slice(max(lo, 0), max(hi, 0)))
        return not bool(np.any(entry.coeffs[tuple(window)]))

    def contributions(
        self, cid: int
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Return the active functions that do not vanish on cell ``cid``.

        A function qualifies iff its tensor-product support covers the cell and, when it is
        truncated, it is not identically zero there (:meth:`_vanishes_on_cell`).  The
        functions whose support merely covers the cell are :func:`_supported_functions`.

        Three parallel arrays rather than three calls, because a caller wanting the
        multi-indices wants the dofs beside them; the C++ counterpart returns the same
        triple for the same reason.

        Args:
            cid (int): Active cell flat id in ``[0, grid.num_cells)``.

        Returns:
            tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
            ``(dofs, levels, multi_indices)`` sorted by global dof, of shapes ``(K,)``,
            ``(K,)`` and ``(K, dim)``.  All three are read-only.

        Raises:
            IndexError: If ``cid`` is out of range ``[0, grid.num_cells)``.

        Note:
            Results are memoized per ``cid`` in ``self._contrib_cache`` (the space is an
            immutable snapshot), and the arrays are the cached objects, which is why
            they are read-only.  :class:`THBSplineSpace` is what copies them on the way
            out where its own contract promises a writable array.
        """
        cached = self._contrib_cache.get(cid)
        if cached is not None:
            return cached
        supported = _supported_functions(self, cid)  # validates cid
        cell_level = self._grid.cell_level(cid)
        cell_midx = self._grid.cell_multi_index(cid)
        contribs = [
            triple
            for triple in supported
            if (entry := self._trunc.get(triple[0])) is None
            or not self._vanishes_on_cell(entry, cell_level, cell_midx)
        ]
        dofs = np.array([triple[0] for triple in contribs], dtype=np.int64)
        levels = np.array([triple[1] for triple in contribs], dtype=np.int64)
        multis = np.array([triple[2] for triple in contribs], dtype=np.int64).reshape(
            len(contribs), self.dim
        )
        for block in (dofs, levels, multis):
            block.flags.writeable = False
        table = (dofs, levels, multis)
        self._contrib_cache[cid] = table
        return table

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def grid(self) -> HierarchicalGrid:
        """Get the underlying hierarchical grid.

        Returns:
            HierarchicalGrid: The active-cell hierarchy this space is built on.
        """
        return self._grid

    @property
    def root_space(self) -> BsplineSpace:
        """Get the level-0 tensor-product space.

        Returns:
            BsplineSpace: The root B-spline space.
        """
        return self._root_space

    @property
    def dim(self) -> int:
        """Get the parametric dimension.

        Returns:
            int: Number of parametric directions.
        """
        return self._root_space.dim

    @property
    def degrees(self) -> tuple[int, ...]:
        """Get the per-direction polynomial degrees.

        Returns:
            tuple[int, ...]: Degree per direction (the same at every level).
        """
        return self._root_space.degrees

    @property
    def num_levels(self) -> int:
        """Get the number of hierarchy levels at construction time.

        Returns:
            int: Number of levels; stable even if the grid is later refined.
        """
        return len(self._level_spaces)

    @property
    def truncate(self) -> bool:
        """Get whether the hierarchical basis is truncated.

        Returns:
            bool: ``True`` for the truncated (THB) basis, ``False`` for the plain
            hierarchical (HB) basis.
        """
        return self._truncate

    @property
    def regularity(self) -> tuple[int | None, ...]:
        """Get the per-direction continuity used to build the finer levels.

        Returns:
            tuple[int | None, ...]: One entry per direction, ``None`` where maximal
            smoothness was asked for.  Already broadcast, so its length is ``dim``.
        """
        return self._regularity

    @property
    def num_total_basis(self) -> int:
        """Get the total number of active hierarchical basis functions.

        Mirrors :attr:`~pantr.bspline.BsplineSpace.num_total_basis` (the hierarchical
        basis is not tensor-product, so there is no per-direction ``num_basis``).

        Returns:
            int: Total active-function count across all levels.
        """
        return self._num_active

    @property
    def num_basis_per_level(self) -> tuple[int, ...]:
        """Get the number of active basis functions at each level.

        Returns:
            tuple[int, ...]: Active-function count per level.
        """
        return tuple(int(a.shape[0]) for a in self._active_funcs)

    @property
    def level_offsets(self) -> npt.NDArray[np.int64]:
        """Get the per-level base of the global dof numbering.

        Returns:
            npt.NDArray[np.int64]: Length ``num_levels + 1``; entry ``l`` is the first
            global dof of level ``l`` and the last entry is ``num_total_basis``.
            Read-only, and the space's own storage.
        """
        return self._func_offset

    @property
    def num_truncated(self) -> int:
        """Get how many active functions the truncation actually touched.

        Returns:
            int: The number of dofs for which :meth:`truncated` is not ``None``;
            always ``0`` when ``truncate`` is ``False``.
        """
        return len(self._trunc)

    @property
    def domain(self) -> npt.NDArray[np.float32 | np.float64]:
        """Get the parametric domain bounds.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Shape ``(dim, 2)`` ``[lo, hi]`` per
            direction (from the root space).
        """
        return self._root_space.domain

    @property
    def tolerance(self) -> float:
        """Get the numerical tolerance.

        Returns:
            float: The root space's tolerance.
        """
        return self._root_space.tolerance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def level_space(self, level: int) -> BsplineSpace:
        """Return the tensor-product space at ``level``.

        Fixed at construction, and the grid it was built from is immutable, so
        nothing can change it afterwards.

        Args:
            level (int): Hierarchy level in ``[0, num_levels)``.

        Returns:
            BsplineSpace: The root space subdivided to ``level``.

        Raises:
            ValueError: If ``level`` is out of range.
        """
        if not (0 <= level < self.num_levels):
            raise ValueError(f"level must be in [0, {self.num_levels - 1}]; got {level!r}.")
        return self._level_spaces[level]

    def active_function_indices(self, level: int) -> npt.NDArray[np.int64]:
        """Return the flat indices of the active functions at ``level``.

        Fixed at construction, and the grid it was built from is immutable, so
        nothing can change it afterwards.

        Args:
            level (int): Hierarchy level in ``[0, num_levels)``.

        Returns:
            npt.NDArray[np.int64]: Sorted flat (C-order) level-``level`` function
            indices selected by the Kraft rule.  Read-only, and the space's own
            storage; :class:`THBSplineSpace` is what copies it on the way out.

        Raises:
            ValueError: If ``level`` is out of range.
        """
        if not (0 <= level < self.num_levels):
            raise ValueError(f"level must be in [0, {self.num_levels - 1}]; got {level!r}.")
        return self._active_funcs[level]

    def active_basis(self, cid: int) -> npt.NDArray[np.int64]:
        """Return the global dofs of the active functions that do not vanish on cell ``cid``.

        A function is listed iff its tensor-product support covers the cell and it is not
        identically zero there.  Only a truncated function can be supported on a cell and
        vanish on it, which happens inside a refined region where truncation has removed
        every component the function had on the cell.

        Args:
            cid (int): Active cell flat id in ``[0, grid.num_cells)``.

        Returns:
            npt.NDArray[np.int64]: Sorted global hierarchical-dof indices of the
            functions non-zero on cell ``cid``.  Read-only, and the space's own
            storage; :class:`THBSplineSpace` is what copies it on the way out.

        Raises:
            IndexError: If ``cid`` is out of range ``[0, grid.num_cells)``.
        """
        return self.contributions(cid)[0]

    def max_active_per_cell(self) -> int:
        """Return the largest number of active functions on any single cell.

        The width a fixed-size dofmap needs: ``max(active_basis(cid).size)`` over every
        active cell. On an unrefined space this is ``prod(degree + 1)``; near a level
        interface a cell also sees the coarser functions overlapping it, so the count
        grows there.

        Computed once and cached, since the space is a construction-time snapshot.

        Returns:
            int: Maximum active-function count over all cells (``>= 1``).

        Note:
            Counts exactly what :meth:`active_basis` returns, so functions that vanish on
            a cell are not counted there. Truncation only annihilates functions, so with
            ``truncate=True`` the value is at most the HB basis's on the same grid, and
            strictly less wherever a coarse function vanishes on the widest cells.

            A function supported on a level-``L`` cell lies, at its own level ``m``,
            among the ``prod(degree + 1)`` level-``m`` functions supported on the cell's
            ancestor, so a cell lists at most ``(L + 1) * prod(degree + 1)`` functions.
            The HB basis reaches that growth on a hierarchy refined repeatedly around one
            region. Under truncation a coarse function refined through several levels of
            active finer functions vanishes on the deepest cells and is not listed, so the
            THB count there stays far below the bound. It is not bounded by
            ``prod(degree + 1)`` either: more functions than that can be non-zero on one
            cell, and are then linearly dependent there.

            Visits every cell, so the first call populates the per-cell contribution
            cache for the whole grid -- the same cache :meth:`active_basis` fills lazily,
            but warmed in full.
        """
        if self._max_active_per_cell is None:
            self._max_active_per_cell = max(
                int(self.contributions(cid)[0].shape[0]) for cid in range(self._grid.num_cells)
            )
        return self._max_active_per_cell

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def refine(
        self,
        cell_ids: npt.NDArray[np.int64],
        admissible_class: int | None,
    ) -> _THBSplineSpacePython:
        """Return a new space with the marked cells refined.

        This method does not mutate ``self`` or its grid: the grid is refined by
        rebinding, and a new :class:`_THBSplineSpacePython` is built on the result; ``self``
        and its grid are unchanged.

        With ``admissible_class=m`` (the default ``m=2``) the refinement is graded so
        the resulting mesh is admissible of class ``m`` (the truncated functions
        acting on any cell span at most ``m`` successive levels), following the
        recursive refinement-neighborhood algorithm of Carraturo et al. (2019).  This
        assumes the current mesh is already admissible of class ``m`` (true for the
        root and for any mesh built via graded :meth:`refine`).  With
        ``admissible_class=None`` exactly the marked cells are refined (no grading).

        Args:
            cell_ids (npt.ArrayLike): Flat ids of active cells to refine.
            admissible_class (int | None): Admissibility class ``m >= 2`` to maintain,
                or ``None`` for ungraded refinement.  Defaults to ``2``.

        Returns:
            _THBSplineSpacePython: A new space on the refined grid (same ``root_space``,
            ``truncate``, and ``regularity``).

        Raises:
            IndexError: If any id is outside ``[0, grid.num_cells)``.
            ValueError: If ``admissible_class`` is an integer ``< 2``.
        """
        self._check_admissible_class(admissible_class)
        ids = np.unique(np.asarray(cell_ids, dtype=np.int64).ravel())
        bad = [int(x) for x in ids if int(x) < 0 or int(x) >= self._grid.num_cells]
        if bad:
            raise IndexError(
                f"cell_ids must lie in [0, {self._grid.num_cells}); got out-of-range id(s): {bad}."
            )
        # Convert to (level, midx) on the original grid before any refinement, since
        # flat ids are reassigned by every grid.refine call.
        marked = [(self._grid.cell_level(int(c)), self._grid.cell_multi_index(int(c))) for c in ids]
        return self._refine_marked(marked, admissible_class)

    def refine_region(
        self,
        level: int,
        lo: npt.NDArray[np.int64],
        hi: npt.NDArray[np.int64],
        admissible_class: int | None,
    ) -> _THBSplineSpacePython:
        """Return a new space with the active cells in a rectangular region refined.

        The region is the integer cell-index box ``[lo, hi)`` at ``level`` (in
        level-``level`` coordinates), matching the convention of
        :meth:`pantr.grid.HierarchicalGrid.refine`.  Only the currently-active leaf
        cells inside the box are refined; the rest of the box (already refined, or
        not present at ``level``) is ignored.  If the box contains no active leaf
        cells, the call is a no-op and returns a space equivalent to ``self``.  This
        is the region-based counterpart of :meth:`refine`, which marks individual
        cells by flat id.

        Like :meth:`refine`, this does not mutate ``self`` or its grid: the grid is
        refined by rebinding, and a new :class:`_THBSplineSpacePython` is returned.  Calls
        chain, so successive regions refine progressively (graded by default).

        Args:
            level (int): Level at which the box lives.  Must satisfy
                ``0 <= level <= grid.max_level``.
            lo (npt.NDArray[np.int64]): Per-direction start index (inclusive), in
                level-``level`` coordinates.
            hi (npt.NDArray[np.int64]): Per-direction end index (exclusive), in
                level-``level`` coordinates.
            admissible_class (int | None): Admissibility class ``m >= 2`` to maintain
                (graded refinement), or ``None`` for ungraded refinement.  Defaults
                to ``2``.  See :meth:`refine`.

        Returns:
            _THBSplineSpacePython: A new space on the refined grid (same ``root_space``,
            ``truncate``, and ``regularity``).

        Raises:
            ValueError: If ``admissible_class`` is an integer ``< 2``, ``level`` is
                out of range, ``lo``/``hi`` have the wrong length, any
                ``lo[k] >= hi[k]``, or any part of ``[lo, hi)`` lies outside the
                level domain.
        """
        self._check_admissible_class(admissible_class)
        lo_t, hi_t = self._validate_region(level, lo, hi)
        # Enumerate the active leaves in the box on the original grid (flat ids are
        # reassigned by every grid.refine call, so capture cells up front).
        marked = [
            (level, midx)
            for midx in itertools.product(*(range(lo_t[k], hi_t[k]) for k in range(self.dim)))
            if self._grid.is_active_leaf(level, midx)
        ]
        return self._refine_marked(marked, admissible_class)

    @staticmethod
    def _check_admissible_class(admissible_class: int | None) -> None:
        """Validate the ``admissible_class`` argument shared by the refine methods.

        Args:
            admissible_class (int | None): The class value to check.

        Raises:
            ValueError: If ``admissible_class`` is an integer ``< 2``.
        """
        if admissible_class is not None and admissible_class < 2:  # noqa: PLR2004
            raise ValueError(
                f"admissible_class must be an integer >= 2 or None; got {admissible_class!r}."
            )

    def _validate_region(
        self,
        level: int,
        lo: npt.NDArray[np.int64],
        hi: npt.NDArray[np.int64],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Validate a ``[lo, hi)`` cell-index box at ``level`` and normalize to tuples.

        Applies the same four checks as :meth:`pantr.grid.HierarchicalGrid.refine`:
        ``level`` range, ``lo``/``hi`` lengths, ``lo < hi`` per axis, and ``[lo, hi)``
        within ``[0, level_cells_per_axis(level))`` on every axis.

        Args:
            level (int): Level the box lives at.
            lo (npt.NDArray[np.int64]): Per-direction start index (inclusive).
            hi (npt.NDArray[np.int64]): Per-direction end index (exclusive).

        Returns:
            tuple[tuple[int, ...], tuple[int, ...]]: The validated ``(lo, hi)`` tuples.

        Raises:
            ValueError: If ``level`` is out of range, ``lo``/``hi`` have the wrong
                length, any ``lo[k] >= hi[k]``, or ``[lo, hi)`` is out of bounds.
        """
        ndim = self.dim
        max_level = self._grid.max_level
        if not 0 <= int(level) <= max_level:
            raise ValueError(f"level must be in [0, {max_level}]; got {level!r}.")
        lo_t = tuple(int(x) for x in lo)
        hi_t = tuple(int(x) for x in hi)
        if len(lo_t) != ndim or len(hi_t) != ndim:
            raise ValueError(f"lo and hi must have length {ndim}; got {len(lo_t)} and {len(hi_t)}.")
        if any(lo_k >= hi_k for lo_k, hi_k in zip(lo_t, hi_t, strict=False)):
            raise ValueError(
                f"lo must be strictly less than hi in every dimension; "
                f"got lo={lo_t!r}, hi={hi_t!r}."
            )
        n_per_axis = self._grid.level_cells_per_axis(level)
        for k in range(ndim):
            if lo_t[k] < 0 or hi_t[k] > n_per_axis[k]:
                raise ValueError(
                    f"[lo, hi) out of bounds at level {level}: "
                    f"axis {k} needs [0, {n_per_axis[k]}), got [{lo_t[k]}, {hi_t[k]})."
                )
        return lo_t, hi_t

    def _refine_marked(
        self,
        marked: list[tuple[int, tuple[int, ...]]],
        admissible_class: int | None,
    ) -> _THBSplineSpacePython:
        """Return a new space over this space's grid with the marked cells refined.

        Shared by :meth:`refine` and :meth:`refine_region`.  Mutates nothing: the
        grid is refined by rebinding, so ``self`` and its grid are untouched.
        Callers are responsible for capturing ``marked`` against the original grid
        before any refinement (flat ids are reassigned by every refine).

        When ``marked`` refines nothing the grid is copied instead, so the returned
        space never holds this space's grid object.  The cell decomposition is
        immutable, but a grid also carries two tag registries and a BVH memo, which
        belong to whoever holds the grid, and sharing them across two spaces would
        make a tag set through one visible through the other.

        Args:
            marked (list[tuple[int, tuple[int, ...]]]): ``(level, midx)`` pairs of
                cells to refine, captured on the original grid.
            admissible_class (int | None): Admissibility class to maintain, or
                ``None`` for ungraded refinement.

        Returns:
            _THBSplineSpacePython: A new space on the refined grid (same ``root_space``,
            ``truncate``, and ``regularity``).
        """
        grid = self._grid
        for level, midx in marked:
            if admissible_class is None:
                if grid.is_active_leaf(level, midx):
                    grid = grid.refine(level, list(midx), [i + 1 for i in midx])
            else:
                grid = self._refine_recursive(grid, level, midx, admissible_class)
        return _THBSplineSpacePython(
            self._root_space,
            grid if grid is not self._grid else grid._copy(),
            self._truncate,
            self._regularity,
        )

    def _refine_recursive(
        self,
        grid: HierarchicalGrid,
        level: int,
        midx: tuple[int, ...],
        m: int,
    ) -> HierarchicalGrid:
        """Return ``grid`` with cell ``(level, midx)`` refined, graded for class ``m``.

        Refines every cell in the refinement neighborhood (recursively, at the coarser
        level ``level - m + 1``) before subdividing ``(level, midx)``, per Algorithm 4
        of Carraturo et al. (2019).  Each step queries the grid produced by the
        previous one, as the in-place version queried the grid it had just mutated.

        Args:
            grid (HierarchicalGrid): The grid to refine.  Not modified.
            level (int): Level of the cell to refine.
            midx (tuple[int, ...]): Per-axis index of the cell at ``level``.
            m (int): Admissibility class (``>= 2``).

        Returns:
            HierarchicalGrid: The refined grid; ``grid`` itself when the cell is not
            an active leaf and its neighborhood is empty.

        Raises:
            RecursionError: Unreachable in practice — recursion depth is bounded by
                ``level <= grid.max_level``, which is bounded by available memory long
                before Python's default recursion limit.
        """
        for nlevel, nmidx in self._refinement_neighborhood(level, midx, m, grid):
            grid = self._refine_recursive(grid, nlevel, nmidx, m)
        if grid.is_active_leaf(level, midx):
            grid = grid.refine(level, list(midx), [i + 1 for i in midx])
        return grid

    def _refinement_neighborhood(
        self,
        level: int,
        midx: tuple[int, ...],
        m: int,
        grid: HierarchicalGrid,
    ) -> list[tuple[int, tuple[int, ...]]]:
        """Return the refinement neighborhood of cell ``(level, midx)`` for class ``m``.

        Implements Definition 3.4 of Carraturo et al. (2019). Finds all cells at level
        ``level - m + 1`` that are parents of a level-``level - m + 2`` cell touched by
        any B-spline whose support covers the containing cell of ``(level, midx)`` at
        level ``level - m + 2``.

        Args:
            level (int): Level of the cell.
            midx (tuple[int, ...]): Per-axis index of the cell at ``level``.
            m (int): Admissibility class (``>= 2``, so ``level - m + 2 <= level``).
            grid (HierarchicalGrid): The grid whose active set is queried.

        Returns:
            list[tuple[int, tuple[int, ...]]]: ``(level - m + 1, parent_midx)`` cells in
            the neighborhood that are currently active leaves.
        """
        dim = self.dim
        factor = self._grid.factor
        k_nbr = level - m + 1
        if k_nbr < 0:
            return []
        k_ext = level - m + 2  # = k_nbr + 1; <= level because m >= 2
        # k_ext < len(self._support) because level <= original max_level = num_levels - 1
        assert k_ext < len(self._support), (
            f"k_ext={k_ext} out of range; level={level}, m={m}, num_levels={self.num_levels}"
        )
        support_ext = self._support[k_ext]
        # Containing cell of (level, midx) at level k_ext.
        q = tuple(midx[d] // factor[d] ** (level - k_ext) for d in range(dim))
        parent_ranges = []
        for d in range(dim):
            first_basis, first_cell, last_cell = support_ext[d]
            fb = int(first_basis[q[d]])
            s_lo = int(first_cell[fb])
            s_hi = int(last_cell[fb + self.degrees[d]]) + 1
            parent_ranges.append(range(s_lo // factor[d], (s_hi - 1) // factor[d] + 1))
        return [
            (k_nbr, p) for p in itertools.product(*parent_ranges) if grid.is_active_leaf(k_nbr, p)
        ]

    def coarsen(
        self,
        cell_ids: npt.NDArray[np.int64],
        admissible_class: int | None,
    ) -> _THBSplineSpacePython:
        """Return a new space with the marked cells coarsened away.

        A parent cell is reactivated (its children removed) only when **all** of its
        children are marked active leaves, mirroring the coarsening algorithm of
        Carraturo et al. (2019, Alg. 5).  That rule is
        :meth:`~pantr.grid.HierarchicalGrid.coarsen_cells`, which this method drives one
        parent at a time so the admissibility guard below can veto a parent without
        affecting the rest.  With ``admissible_class=None`` this is the exact inverse of
        :meth:`refine`: ``space.refine(cells).coarsen(children_of(cells))`` recovers
        ``space``.  With ``admissible_class=m`` the guard may suppress some coarsenings,
        so the recovery holds only when the guard permits them all.

        With ``admissible_class=m`` (the default ``m=2``) a parent is reactivated only
        if its coarsening neighborhood (Def. 3.5) is empty, so the resulting mesh stays
        admissible of class ``m``.  With ``admissible_class=None`` that guard is skipped.

        The space is immutable: the grid is coarsened by rebinding, and a new
        :class:`_THBSplineSpacePython` is built on the result; ``self`` and its grid are
        unchanged.  An empty ``cell_ids``, and one that coarsens nothing, both return
        an equivalent new space over a copy of this space's grid -- a copy rather than
        the same object, so the two spaces never share the grid's tag registries.

        Args:
            cell_ids (npt.ArrayLike): Flat ids of active leaf cells to coarsen away.
                An empty array is valid and coarsens nothing.
            admissible_class (int | None): Admissibility class ``m >= 2`` to maintain,
                or ``None`` to skip the admissibility guard.  Defaults to ``2``.

        Returns:
            _THBSplineSpacePython: A new space on the coarsened grid (same ``root_space``,
            ``truncate``, and ``regularity``).

        Raises:
            IndexError: If any id is outside ``[0, grid.num_cells)``.
            ValueError: If ``admissible_class`` is an integer ``< 2``.
        """
        if admissible_class is not None and admissible_class < 2:  # noqa: PLR2004
            raise ValueError(
                f"admissible_class must be an integer >= 2 or None; got {admissible_class!r}."
            )
        ids = np.unique(np.asarray(cell_ids, dtype=np.int64).ravel())
        bad = [int(x) for x in ids if int(x) < 0 or int(x) >= self._grid.num_cells]
        if bad:
            raise IndexError(
                f"cell_ids must lie in [0, {self._grid.num_cells}); got out-of-range id(s): {bad}."
            )
        dim = self.dim
        factor = self._grid.factor
        marked = {(self._grid.cell_level(int(c)), self._grid.cell_multi_index(int(c))) for c in ids}
        parents = {
            (level - 1, tuple(midx[d] // factor[d] for d in range(dim)))
            for level, midx in marked
            if level >= 1
        }
        num_children = math.prod(factor)
        grid = self._grid
        # Deepest parent first, so a veto is decided against a mesh whose finer
        # coarsenings have already happened.
        for parent_level, pmidx in sorted(parents, key=lambda pc: -pc[0]):
            # Name this parent's marked children on the current grid -- every coarsening
            # reassigns flat ids, so they are resolved afresh here rather than kept.
            child_ids: list[int] = []
            for child in itertools.product(
                *(range(pmidx[d] * factor[d], (pmidx[d] + 1) * factor[d]) for d in range(dim))
            ):
                cid = grid.cell_id(parent_level + 1, child)
                if cid is not None and (parent_level + 1, child) in marked:
                    child_ids.append(cid)
            # An incomplete family is one `coarsen_cells` would skip anyway, so leaving
            # here costs nothing and skips the only expensive test in the loop.  Measured
            # on a 2050-cell mesh with 1537 cells marked, so most families are incomplete:
            # 9.4 ms with this line, 19.7 ms without it, 9.6 ms for the pre-refactor loop
            # this replaces -- which had the same order and which it therefore matches.
            if len(child_ids) < num_children:
                continue
            if admissible_class is not None and not self._coarsening_neighborhood_empty(
                parent_level, pmidx, admissible_class, grid
            ):
                continue
            # coarsen_cells applies the rule itself -- it demotes the parent only if all
            # of its children are named active leaves, which is Alg. 5's condition.
            grid = grid.coarsen_cells(child_ids)
        return _THBSplineSpacePython(
            self._root_space,
            grid if grid is not self._grid else grid._copy(),
            self._truncate,
            self._regularity,
        )

    def _coarsening_neighborhood_empty(
        self,
        parent_level: int,
        pmidx: tuple[int, ...],
        m: int,
        grid: HierarchicalGrid,
    ) -> bool:
        """Return whether the coarsening neighborhood of a parent is empty (Def. 3.5).

        The neighborhood is the set of active cells at level ``parent_level + m``
        contained in the multilevel support extension (at level ``parent_level + 1``)
        of the parent's children.  When it is empty, reactivating the parent preserves
        class-``m`` admissibility (Carraturo et al. 2019).

        Args:
            parent_level (int): Level of the parent being considered for coarsening.
            pmidx (tuple[int, ...]): Per-axis index of the parent at ``parent_level``.
            m (int): Admissibility class (``>= 2``).
            grid (HierarchicalGrid): The grid whose active set is queried.

        Returns:
            bool: ``True`` iff no active cell at level ``parent_level + m`` lies in the
            support extension of the parent's children.

        Note:
            Assumes ``parent_level + 1 < self.num_levels`` and ``m >= 2``; both are
            guaranteed by the calling context in :meth:`coarsen`.  No input validation
            is performed.
        """
        dim = self.dim
        factor = self._grid.factor
        support = self._support[parent_level + 1]
        ext_lo: list[int] = []
        ext_hi: list[int] = []
        for d in range(dim):
            first_basis, first_cell, last_cell = support[d]
            c_lo = pmidx[d] * factor[d]
            c_hi = (pmidx[d] + 1) * factor[d]
            fmin = int(first_basis[c_lo])
            fmax = int(first_basis[c_hi - 1]) + self.degrees[d]
            ext_lo.append(int(first_cell[fmin]))
            ext_hi.append(int(last_cell[fmax]) + 1)
        target = parent_level + m
        if target > grid.max_level:
            return True
        box_lo = [ext_lo[d] * factor[d] ** (m - 1) for d in range(dim)]
        box_hi = [ext_hi[d] * factor[d] ** (m - 1) for d in range(dim)]
        for blk_lo, blk_hi in grid.active_blocks(target):
            if all(max(box_lo[d], blk_lo[d]) < min(box_hi[d], blk_hi[d]) for d in range(dim)):
                return False
        return True

    def _offsets(self) -> npt.NDArray[np.int64]:
        """Return this space's level-offset array, read-only.

        Named as :meth:`THBSplineSpace._offsets` is, so that
        :func:`_supported_functions` can serve the oracle and the wrapper from one
        body.

        Returns:
            npt.NDArray[np.int64]: Length ``num_levels + 1``, read-only.
        """
        return self._func_offset

    def _active_at(self, level: int) -> npt.NDArray[np.int64]:
        """Return the level's active-function index array, read-only.

        Args:
            level (int): Hierarchy level in ``[0, num_levels)``.

        Returns:
            npt.NDArray[np.int64]: Sorted flat (C-order) indices, read-only.
        """
        return self._active_funcs[level]

    def _level_support(self, level: int) -> tuple[_Support1D, ...]:
        """Return the level's per-direction function-to-cell support.

        Args:
            level (int): Hierarchy level in ``[0, num_levels)``.

        Returns:
            tuple[_Support1D, ...]: One ``(first_basis, first_cell, last_cell)`` triple
            per direction.
        """
        return self._support[level]

    # ------------------------------------------------------------------
    # The dof's own level and truncation
    # ------------------------------------------------------------------

    def dof_level(self, dof: int) -> int:
        """Return the hierarchy level that owns global active-function ``dof``.

        Args:
            dof (int): Global active-function index in ``[0, num_total_basis)``.

        Returns:
            int: The level whose dof range (per :attr:`level_offsets`) contains
            ``dof``.

        Raises:
            IndexError: If ``dof`` is out of range ``[0, num_total_basis)``.
        """
        self._check_dof(dof)
        return int(np.searchsorted(self._func_offset, dof, side="right")) - 1

    def truncated(self, dof: int) -> tuple[int, tuple[int, ...], npt.NDArray[np.float64]] | None:
        """Return the stored representation of a truncated function, or ``None``.

        ``None`` is what distinguishes a plain tensor-product B-spline from a truncated
        one that happens to carry an all-ones box, and it is what every untruncated
        function gives -- including every function of a space built with
        ``truncate=False``.

        Args:
            dof (int): Global active-function index in ``[0, num_total_basis)``.

        Returns:
            tuple[int, tuple[int, ...], npt.NDArray[np.float64]] | None:
            ``(rep_level, box_lo, coeffs)`` where ``coeffs`` holds the function in the
            level-``rep_level`` tensor-product basis over the box starting at
            ``box_lo``, or ``None`` if the truncation left the function alone.  The
            coefficients are read-only.

        Raises:
            IndexError: If ``dof`` is out of range ``[0, num_total_basis)``.
        """
        self._check_dof(dof)
        return self._trunc.get(dof)

    def _check_dof(self, dof: int) -> None:
        """Refuse a global dof outside the active basis.

        Its wording is the C++ type's, character for character, because
        :class:`THBSplineSpace` forwards to whichever of the two it holds and a caller
        matching on the message must not be able to tell them apart.

        Args:
            dof (int): The dof to check.

        Raises:
            IndexError: If ``dof`` is outside ``[0, num_total_basis)``.
        """
        if not (0 <= dof < self._num_active):
            raise IndexError(f"dof {dof} is out of range [0, {self._num_active}).")

    def __repr__(self) -> str:
        """Return a compact string representation.

        Returns:
            str: Shows dimension, degrees, level count, active-function count, and
            truncation flag.
        """
        return (
            f"THBSplineSpace(dim={self.dim}, degrees={self.degrees}, "
            f"num_levels={self.num_levels}, num_total_basis={self._num_active}, "
            f"truncate={self._truncate})"
        )


class THBSplineSpace:
    r"""Hierarchical B-spline space on a :class:`~pantr.grid.HierarchicalGrid`.

    Built from a root :class:`~pantr.bspline.BsplineSpace` (level 0) and a
    :class:`~pantr.grid.HierarchicalGrid` carrying the active-cell hierarchy.  The
    per-level tensor-product spaces are obtained by uniformly subdividing the root
    space according to the grid's per-direction ``factor``.  The active hierarchical
    basis is the Kraft selection :cite:p:`kraft1997hierarchical,vuong2011hierarchical`:
    a level-``l`` tensor-product B-spline is active iff its support lies in the
    level-``l`` subdomain :math:`\Omega_l` but not entirely in the finer subdomain
    :math:`\Omega_{l+1}`.

    With ``truncate=True`` (the default) the *truncated* hierarchical basis (THB) is
    built: each active function that straddles a finer-level refinement boundary has its
    components on active finer functions removed (Giannelli-Jüttler-Speleers truncation
    :cite:p:`giannelli2012thb`), restoring the partition of unity.  Only truncated
    functions store a coefficient vector (in the finest tensor-product basis their support
    reaches); untruncated functions remain plain tensor-product B-splines.  With
    ``truncate=False`` the non-truncated hierarchical basis (HB) is built.

    A :class:`~pantr.grid.HierarchicalGrid` is immutable, so this space cannot go
    stale: :meth:`~pantr.grid.HierarchicalGrid.refine` returns a *new* grid and leaves
    the one held here untouched.  :meth:`refine`, :meth:`refine_region` and
    :meth:`coarsen` are the same shape one level up -- each returns a new space over a
    grid of its own, including when it refines or coarsens nothing.

    **This class is a wrapper.** The value -- the level spaces, the Kraft selection, the
    truncation coefficients, the per-cell contribution table -- is owned by an
    implementation chosen by :func:`_impl_class`, which is the C++ type
    (``cpp/include/pantr/bspline/thb_space.hpp``) or the oracle
    :class:`_THBSplineSpacePython`.  Basis tabulation, :meth:`restrict` and the three
    prolongation operators have no C++ counterpart yet and are computed here, against
    the forwarded accessors alone so that one body serves both backends; the module
    docstring carries why that rather than a second always-Python space.

    Instances are immutable, and that is enforced rather than documented: ``__slots__``
    means there is no ``__dict__`` to attach anything to, and ``__setattr__`` refuses
    even a rebinding of the slots.  The wrapper fills them through
    ``object.__setattr__``, which is the pattern ``design/bspline_derived_caches.md``
    asks for and :mod:`pantr.bspline._bspline_space_nd` already ships.

    Note:
        A function is active on a cell when its tensor-product support covers the cell
        and it does not vanish identically there.  Under truncation a coarse function can
        vanish on cells inside the refined region; :meth:`active_basis`,
        :meth:`tabulate_basis` and :meth:`max_active_per_cell` omit it on those cells.

    References:
        Adaptive isogeometric algorithms for hierarchical splines
        :cite:p:`garau2018algorithms`.  Per-element multi-level Bézier extraction
        (used for element assembly and visualization) is provided by
        :class:`~pantr.bspline.MultiLevelExtraction`, following
        :cite:t:`dangella2018multilevel`.

    Attributes:
        _impl: The implementation this wrapper holds; see :func:`_impl_class`.  Its
            type is the private ``_Impl`` alias, a union of three unrelated nominal
            types with no documented form to name here.
        _root_space (BsplineSpace): The root wrapper this space was built from, so that
            ``thb.root_space is root_space`` holds.  A *presentation* memo, never a
            second truth: every count, bound and tolerance comes from ``_impl``.
        _grid (HierarchicalGrid): The grid wrapper, on the same terms -- and carrying
            the grid's tag registries, which is why re-wrapping the handle per access
            would be wrong rather than merely wasteful.
        _level_space_memo (dict[int, BsplineSpace]): One wrapper per level, built on
            first request.  ``design/bspline_ownership_lifetime.md`` names this the one
            place a dict memo is right here, because the key is genuinely data.
        _active_memo (dict[int, npt.NDArray[np.int64]]): The per-level active-function
            index arrays as the implementation owns them, read-only.
        _support_memo (dict[int, tuple[_Support1D, ...]]): Per-level, per-direction
            function-to-cell support, derived from the level spaces for the operations
            that have no C++ counterpart.
        _level_offsets_memo (npt.NDArray[np.int64] | None): The implementation's own
            level-offset array, read-only; ``None`` until first requested.
    """

    __slots__ = (
        "_active_memo",
        "_grid",
        "_impl",
        "_level_offsets_memo",
        "_level_space_memo",
        "_root_space",
        "_support_memo",
    )

    _impl: _Impl
    """The implementation this wrapper holds; see :func:`_impl_class`."""

    _root_space: BsplineSpace
    """The root wrapper this space was built from; see the class docstring."""

    _grid: HierarchicalGrid
    """The grid wrapper this space was built over; see the class docstring."""

    _level_space_memo: dict[int, BsplineSpace]
    """One level-space wrapper per level, built on first request."""

    _active_memo: dict[int, npt.NDArray[np.int64]]
    """The implementation's own per-level active-function arrays, read-only."""

    _support_memo: dict[int, tuple[_Support1D, ...]]
    """Per-level, per-direction function support, derived on first request."""

    _level_offsets_memo: npt.NDArray[np.int64] | None
    """The implementation's own level-offset array, read-only; ``None`` until read."""

    def __init__(
        self,
        root_space: BsplineSpace,
        grid: HierarchicalGrid,
        *,
        truncate: bool = True,
        regularity: int | Sequence[int | None] | None = None,
    ) -> None:
        """Create a hierarchical B-spline space.

        The two ``isinstance`` checks and the scalar broadcast are Python's calling
        convention rather than validation: neither implementation accepts a scalar
        ``regularity``, and neither can raise a readable refusal for an argument that
        is not a wrapper at all, since reaching ``._impl`` is what the wrapper does
        first.  Everything else -- the dimension match, the root bounds, the
        regularity's length and range -- is the implementation's, which is what keeps
        the messages identical under both backends.

        Args:
            root_space (BsplineSpace): The level-0 tensor-product B-spline space.
            grid (HierarchicalGrid): Hierarchical grid whose root knot-span grid
                matches ``root_space``.
            truncate (bool): If ``True`` (default), build the truncated (THB) basis;
                if ``False``, build the non-truncated hierarchical (HB) basis.
            regularity (int | Sequence[int | None] | None): Per-direction continuity
                at the knots inserted when subdividing to finer levels.  A scalar is
                broadcast to every axis; ``None`` (default) uses maximal smoothness.
                Each non-``None`` entry must satisfy ``-1 <= regularity[k] < degree[k]``.

        Raises:
            TypeError: If ``root_space`` is not a :class:`~pantr.bspline.BsplineSpace`
                or ``grid`` is not a :class:`~pantr.grid.HierarchicalGrid`.
            ValueError: If ``grid`` and ``root_space`` disagree on dimension or on
                the root knot-span grid, if ``regularity`` has the wrong length, if any
                per-direction regularity value is out of range, or if either nested
                object was built under a different backend.
            RuntimeError: If the C++ backend is requested and is not available.
        """
        if not isinstance(root_space, BsplineSpace):
            raise TypeError(
                f"root_space must be a BsplineSpace; got {type(root_space).__name__!r}."
            )
        if not isinstance(grid, HierarchicalGrid):
            raise TypeError(f"grid must be a HierarchicalGrid; got {type(grid).__name__!r}.")
        if regularity is None or isinstance(regularity, int):
            reg: Sequence[int | None] = (regularity,) * root_space.dim
        else:
            reg = tuple(regularity)
        impl = _new_impl(root_space, grid, bool(truncate), reg, _stored_dtype(root_space.spaces))
        self._take(impl, root_space, grid)

    @classmethod
    def _wrap_over(
        cls, impl: _Impl, root_space: BsplineSpace, grid: HierarchicalGrid
    ) -> THBSplineSpace:
        """Wrap an implementation, adopting the nested wrappers it shares.

        The path :meth:`refine`, :meth:`refine_region` and :meth:`coarsen` take.  All
        three keep the root space they were given and produce a new grid, so the new
        wrapper inherits the receiver's root **object** and is handed the grid wrapper
        that was built for the new handle.  Without it the C++ path would hand back a
        fresh wrapper around an equal-but-distinct root, and a caller that had tagged
        that root -- or that holds it by identity -- would find a different one.
        ``design/bspline_ownership_lifetime.md`` F6 records why that is a contract.

        Args:
            impl (_Impl): The implementation object to adopt, with no re-validation.
            root_space (BsplineSpace): The root wrapper ``impl`` shares.
            grid (HierarchicalGrid): The grid wrapper for ``impl``'s own grid.

        Returns:
            THBSplineSpace: A wrapper around ``impl``.
        """
        self = object.__new__(cls)
        self._take(impl, root_space, grid)
        return self

    def _take(self, impl: _Impl, root_space: BsplineSpace, grid: HierarchicalGrid) -> None:
        """Hold an implementation, with every memo slot cleared.

        One place rather than two, so that no construction path can leave a slot
        uninitialised -- which surfaces as a bare ``AttributeError`` from inside a
        property, a long way from the constructor that skipped it.

        Args:
            impl (_Impl): The implementation to hold.
            root_space (BsplineSpace): The root wrapper to present.
            grid (HierarchicalGrid): The grid wrapper to present.
        """
        object.__setattr__(self, "_impl", impl)
        object.__setattr__(self, "_root_space", root_space)
        object.__setattr__(self, "_grid", grid)
        object.__setattr__(self, "_level_space_memo", {})
        object.__setattr__(self, "_active_memo", {})
        object.__setattr__(self, "_support_memo", {})
        object.__setattr__(self, "_level_offsets_memo", None)

    def _derived(self, impl: _Impl) -> THBSplineSpace:
        """Wrap a space this one produced: the same root space, a grid of its own.

        Args:
            impl (_Impl): The implementation :meth:`refine`, :meth:`refine_region` or
                :meth:`coarsen` returned.

        Returns:
            THBSplineSpace: A wrapper around ``impl``.
        """
        return type(self)._wrap_over(impl, self._root_space, _as_grid(impl.grid, self._grid.root))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        """Refuse to set an attribute, because a space is immutable.

        Args:
            name (str): The attribute a caller tried to set.
            value (object): The value it tried to set.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError(f"{type(self).__name__} is immutable; cannot set {name!r}")

    def __delattr__(self, name: str) -> NoReturn:
        """Refuse to delete an attribute, because a space is immutable.

        Args:
            name (str): The attribute a caller tried to delete.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError(f"{type(self).__name__} is immutable; cannot delete {name!r}")

    def __reduce__(
        self,
    ) -> tuple[
        Callable[..., THBSplineSpace],
        tuple[BsplineSpace, HierarchicalGrid, bool, tuple[int | None, ...]],
    ]:
        """Pickle by the constructor's arguments rather than by implementation.

        The C++ handle is not picklable and must not become part of the wire format:
        a pickle written under the C++ backend has to load under the Python one and
        the other way round, or the backend switch would silently become a
        data-format switch.  ``design/bspline_pickle_tolerance.md`` is where that rule
        and its reasons live.

        The root space and the grid go out as **wrappers**, not as their
        implementations, which is what carries their own ``__reduce__`` into this one's
        round trip -- the univariate tolerance-drift bound for the one, the root grid's
        tags and the per-level active blocks for the other.  Everything derived from
        them, which is everything else this space holds, is rebuilt by the constructor
        rather than shipped.

        A module-level rebuilder rather than ``(type(self), args)`` because ``truncate``
        and ``regularity`` are keyword-only on the constructor and a two-tuple reduction
        has nowhere to put a keyword.

        Returns:
            tuple: :func:`_rebuild_thb_space` and the arguments to replay.
        """
        return (_rebuild_thb_space, (self.root_space, self.grid, self.truncate, self.regularity))

    # ------------------------------------------------------------------
    # Forwarded properties
    # ------------------------------------------------------------------

    @property
    def grid(self) -> HierarchicalGrid:
        """Get the underlying hierarchical grid.

        The object this space was built over, or the one built for the grid a
        refinement produced -- not a re-wrapping of the handle per access, so
        ``thb.grid is grid`` holds and a tag set through it is seen again.

        Returns:
            HierarchicalGrid: The active-cell hierarchy this space is built on.
        """
        return self._grid

    @property
    def root_space(self) -> BsplineSpace:
        """Get the level-0 tensor-product space.

        The constructor argument's own object, on the same terms as :attr:`grid`, and
        the same object a refinement of this space hands back.

        Returns:
            BsplineSpace: The root B-spline space.
        """
        return self._root_space

    @property
    def dim(self) -> int:
        """Get the parametric dimension.

        Returns:
            int: Number of parametric directions.
        """
        return int(self._impl.dim)

    @property
    def degrees(self) -> tuple[int, ...]:
        """Get the per-direction polynomial degrees.

        Returns:
            tuple[int, ...]: Degree per direction (the same at every level).
        """
        return tuple(int(degree) for degree in self._impl.degrees)

    @property
    def num_levels(self) -> int:
        """Get the number of hierarchy levels at construction time.

        Returns:
            int: Number of levels; stable even if the grid is later refined.
        """
        return int(self._impl.num_levels)

    @property
    def truncate(self) -> bool:
        """Get whether the hierarchical basis is truncated.

        Returns:
            bool: ``True`` for the truncated (THB) basis, ``False`` for the plain
            hierarchical (HB) basis.
        """
        return bool(self._impl.truncate)

    @property
    def regularity(self) -> tuple[int | None, ...]:
        """Get the per-direction continuity used to build the finer levels.

        Returns:
            tuple[int | None, ...]: One entry per direction, ``None`` where maximal
            smoothness was asked for.  Already broadcast, so its length is :attr:`dim`
            whatever the constructor was handed.
        """
        return tuple(self._impl.regularity)

    @property
    def num_total_basis(self) -> int:
        """Get the total number of active hierarchical basis functions.

        Mirrors :attr:`~pantr.bspline.BsplineSpace.num_total_basis` (the hierarchical
        basis is not tensor-product, so there is no per-direction ``num_basis``).

        Returns:
            int: Total active-function count across all levels.
        """
        return int(self._impl.num_total_basis)

    @property
    def num_basis_per_level(self) -> tuple[int, ...]:
        """Get the number of active basis functions at each level.

        Returns:
            tuple[int, ...]: Active-function count per level.
        """
        return tuple(int(count) for count in self._impl.num_basis_per_level)

    @property
    def level_offsets(self) -> npt.NDArray[np.int64]:
        """Get the per-level base of the global dof numbering.

        Returns:
            npt.NDArray[np.int64]: Length ``num_levels + 1``; entry ``l`` is the first
            global dof of level ``l`` and the last entry is :attr:`num_total_basis`.
            A fresh, writable array per call under both backends.
        """
        return np.array(self._offsets())

    @property
    def num_truncated(self) -> int:
        """Get how many active functions the truncation actually touched.

        Returns:
            int: The number of dofs for which :meth:`truncated` is not ``None``;
            always ``0`` when :attr:`truncate` is ``False``.
        """
        return int(self._impl.num_truncated)

    @property
    def domain(self) -> npt.NDArray[np.float32 | np.float64]:
        """Get the parametric domain bounds.

        A fresh, writable array per call under both backends, and a copy of what the
        implementation owns rather than a view of it -- the C++ side hands out a
        read-only view of the root space's storage and the oracle builds an array, and
        a caller must not be able to tell which built the space.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Shape ``(dim, 2)`` ``[lo, hi]`` per
            direction, in the root space's own dtype.
        """
        return np.array(self._impl.domain)

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the floating-point dtype of the space.

        Not forwarded, because neither implementation carries it: the root space's
        storage format decides the C++ class but says nothing about the arithmetic
        here, which is ``float64`` throughout.

        Returns:
            npt.DTypeLike: Always ``numpy.float64`` (THB evaluation is float64).
        """
        return np.float64

    @property
    def tolerance(self) -> float:
        """Get the numerical tolerance.

        Returns:
            float: The root space's tolerance.
        """
        return float(self._impl.tolerance)

    def __repr__(self) -> str:
        """Return a compact string representation.

        Forwarded rather than formatted here, unlike
        :meth:`pantr.grid.HierarchicalGrid.__repr__`: the C++ ``to_string`` was written
        to reproduce the oracle's wording character for character, so forwarding is what
        puts that claim under test on every call rather than hiding it.

        Returns:
            str: Shows dimension, degrees, level count, active-function count, and
            truncation flag.
        """
        return repr(self._impl)

    # ------------------------------------------------------------------
    # Forwarded queries
    # ------------------------------------------------------------------

    def level_space(self, level: int) -> BsplineSpace:
        """Return the tensor-product space at ``level``.

        Fixed at construction, and the grid it was built from is immutable, so
        nothing can change it afterwards.  The wrapper is memoised per level, so the
        same level gives the same object twice; level ``0`` gives
        :attr:`root_space` itself, because both implementations share rather than copy
        the root they were built from.

        Args:
            level (int): Hierarchy level in ``[0, num_levels)``.

        Returns:
            BsplineSpace: The root space subdivided to ``level``.

        Raises:
            ValueError: If ``level`` is out of range.
        """
        value = self._impl.level_space(level)  # validates level
        key = int(level)
        cached = self._level_space_memo.get(key)
        if cached is not None:
            return cached
        space = self._as_space(value)
        self._level_space_memo[key] = space
        return space

    def _as_space(self, value: object) -> BsplineSpace:
        """Present an implementation's level space as the public wrapper.

        The oracle is built on the public types, so its level space already is one; the
        C++ space owns ``pantr::bspline::BsplineSpace<T>`` handles and hands back the
        raw one.  Level ``0``'s handle is the root space's own, which is what
        ``design/bspline_ownership_lifetime.md`` F6 asks to be reused rather than
        re-wrapped -- a fresh wrapper over the same handle would differ from
        :attr:`root_space` on identity while agreeing on every value.

        Args:
            value (object): What the implementation returned for the level.

        Returns:
            BsplineSpace: ``value`` if it already is one, this space's root if it is
            the root's own handle, and a wrapper adopting it otherwise.
        """
        if isinstance(value, BsplineSpace):
            return value
        if value is self._root_space._impl:
            return self._root_space
        return BsplineSpace._wrap_over(cast("Any", value), self._root_space.spaces)

    def _offsets(self) -> npt.NDArray[np.int64]:
        """Return the implementation's own level-offset array, read-only.

        What :attr:`level_offsets` copies from.  Kept separate because the operations
        that have no C++ counterpart read it once per dof, and a copy per read would
        make the prolongation quadratic in the level count for nothing.

        Returns:
            npt.NDArray[np.int64]: Length ``num_levels + 1``, read-only.
        """
        cached = self._level_offsets_memo
        if cached is None:
            cached = np.asarray(self._impl.level_offsets, dtype=np.int64)
            object.__setattr__(self, "_level_offsets_memo", cached)
        return cached

    def _active_at(self, level: int) -> npt.NDArray[np.int64]:
        """Return the level's active-function index array, read-only.

        What :meth:`active_function_indices` copies from, for the reason
        :meth:`_offsets` gives.

        Args:
            level (int): Hierarchy level in ``[0, num_levels)``.

        Returns:
            npt.NDArray[np.int64]: Sorted flat (C-order) indices, read-only.

        Raises:
            ValueError: If ``level`` is out of range.
        """
        value = self._impl.active_function_indices(level)  # validates level
        key = int(level)
        cached = self._active_memo.get(key)
        if cached is None:
            cached = np.asarray(value, dtype=np.int64)
            self._active_memo[key] = cached
        return cached

    def _level_support(self, level: int) -> tuple[_Support1D, ...]:
        """Return the level's per-direction function-to-cell support.

        Derived from :meth:`level_space` rather than forwarded: neither implementation
        exposes it, and the operations that have no C++ counterpart --
        :meth:`restrict` and the prolongation -- need it.  Memoised per level because
        :func:`_func_support_1d` walks every function of the level.

        Args:
            level (int): Hierarchy level in ``[0, num_levels)``.

        Returns:
            tuple[_Support1D, ...]: One ``(first_basis, first_cell, last_cell)`` triple
            per direction.

        Raises:
            ValueError: If ``level`` is out of range.
        """
        key = int(level)
        cached = self._support_memo.get(key)
        if cached is None:
            cached = tuple(_func_support_1d(one_d) for one_d in self.level_space(level).spaces)
            self._support_memo[key] = cached
        return cached

    def active_function_indices(self, level: int) -> npt.NDArray[np.int64]:
        """Return the flat indices of the active functions at ``level``.

        Fixed at construction, and the grid it was built from is immutable, so
        nothing can change it afterwards.

        Args:
            level (int): Hierarchy level in ``[0, num_levels)``.

        Returns:
            npt.NDArray[np.int64]: Sorted flat (C-order) level-``level`` function
            indices selected by the Kraft rule.  A fresh writable copy per call under
            both backends, so a caller may mutate it and cannot corrupt a C++-owned
            array.

        Raises:
            ValueError: If ``level`` is out of range.
        """
        return np.array(self._active_at(level))

    def active_basis(self, cid: int) -> npt.NDArray[np.int64]:
        """Return the global dofs of the active functions that do not vanish on cell ``cid``.

        A function is listed iff its tensor-product support covers the cell and it is not
        identically zero there.  Only a truncated function can be supported on a cell and
        vanish on it, which happens inside a refined region where truncation has removed
        every component the function had on the cell.

        Args:
            cid (int): Active cell flat id in ``[0, grid.num_cells)``.

        Returns:
            npt.NDArray[np.int64]: Sorted global hierarchical-dof indices of the
            functions non-zero on cell ``cid``.  A fresh writable copy per call under
            both backends.

        Raises:
            IndexError: If ``cid`` is out of range ``[0, grid.num_cells)``.
        """
        return np.array(self._impl.active_basis(cid))

    def contributions(
        self, cid: int
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Return the active functions on cell ``cid`` with their levels and multi-indices.

        The same set :meth:`active_basis` lists, as three parallel arrays rather than
        three calls: a caller wanting the multi-indices wants the dofs beside them, and
        a second lookup would re-check ``cid``.

        Args:
            cid (int): Active cell flat id in ``[0, grid.num_cells)``.

        Returns:
            tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
            ``(dofs, levels, multi_indices)`` sorted by global dof, of shapes ``(K,)``,
            ``(K,)`` and ``(K, dim)``.  **Read-only** under both backends and not
            copied, because this is the hot path behind :meth:`tabulate_basis`;
            :meth:`active_basis` is the copying form of its first column.

        Raises:
            IndexError: If ``cid`` is out of range ``[0, grid.num_cells)``.
        """
        dofs, levels, multis = self._impl.contributions(cid)
        return (
            np.asarray(dofs, dtype=np.int64),
            np.asarray(levels, dtype=np.int64),
            np.asarray(multis, dtype=np.int64),
        )

    def max_active_per_cell(self) -> int:
        """Return the largest number of active functions on any single cell.

        The width a fixed-size dofmap needs: ``max(active_basis(cid).size)`` over every
        active cell. On an unrefined space this is ``prod(degree + 1)``; near a level
        interface a cell also sees the coarser functions overlapping it, so the count
        grows there.

        Computed once and cached by the implementation, since the space is a
        construction-time snapshot.

        Returns:
            int: Maximum active-function count over all cells (``>= 1``).

        Note:
            Counts exactly what :meth:`active_basis` returns, so functions that vanish on
            a cell are not counted there. Truncation only annihilates functions, so with
            ``truncate=True`` the value is at most the HB basis's on the same grid, and
            strictly less wherever a coarse function vanishes on the widest cells.

            A function supported on a level-``L`` cell lies, at its own level ``m``,
            among the ``prod(degree + 1)`` level-``m`` functions supported on the cell's
            ancestor, so a cell lists at most ``(L + 1) * prod(degree + 1)`` functions.
            The HB basis reaches that growth on a hierarchy refined repeatedly around one
            region. Under truncation a coarse function refined through several levels of
            active finer functions vanishes on the deepest cells and is not listed, so the
            THB count there stays far below the bound. It is not bounded by
            ``prod(degree + 1)`` either: more functions than that can be non-zero on one
            cell, and are then linearly dependent there.

            Visits every cell, so the first call populates the per-cell contribution
            cache for the whole grid -- the same cache :meth:`active_basis` fills lazily,
            but warmed in full.
        """
        return int(self._impl.max_active_per_cell())

    def dof_level(self, dof: int) -> int:
        """Return the hierarchy level that owns global active-function ``dof``.

        Args:
            dof (int): Global active-function index in ``[0, num_total_basis)``.

        Returns:
            int: The level whose dof range (per :attr:`level_offsets`) contains ``dof``.

        Raises:
            IndexError: If ``dof`` is out of range ``[0, num_total_basis)``.
        """
        return int(self._impl.dof_level(dof))

    def truncated(self, dof: int) -> tuple[int, tuple[int, ...], npt.NDArray[np.float64]] | None:
        """Return the stored representation of a truncated function, or ``None``.

        ``None`` is what distinguishes a plain tensor-product B-spline from a truncated
        one that happens to carry an all-ones box, and it is what every untruncated
        function gives -- including every function of a space built with
        ``truncate=False``.

        Args:
            dof (int): Global active-function index in ``[0, num_total_basis)``.

        Returns:
            tuple[int, tuple[int, ...], npt.NDArray[np.float64]] | None: A named
            ``(rep_level, box_lo, coeffs)`` triple, where ``coeffs`` holds the function
            in the level-``rep_level`` tensor-product basis over the box whose lower
            corner is ``box_lo`` and whose extent is ``coeffs.shape``; ``None`` if the
            truncation left the function alone.  The coefficients are **read-only**
            under both backends -- a view of C++-owned storage on one side and the
            oracle's own frozen array on the other.

        Raises:
            IndexError: If ``dof`` is out of range ``[0, num_total_basis)``.
        """
        entry = self._impl.truncated(dof)
        if entry is None:
            return None
        rep_level, box_lo, coeffs = entry
        return _TruncCoeffs(
            int(rep_level),
            tuple(int(lo) for lo in box_lo),
            np.asarray(coeffs, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # Forwarded refinement
    # ------------------------------------------------------------------

    def refine(
        self,
        cell_ids: npt.ArrayLike,
        *,
        admissible_class: int | None = 2,
    ) -> THBSplineSpace:
        """Return a new space with the marked cells refined.

        This method does not mutate ``self`` or its grid: the grid is refined by
        rebinding, and a new :class:`THBSplineSpace` is built on the result; ``self``
        and its grid are unchanged.

        With ``admissible_class=m`` (the default ``m=2``) the refinement is graded so
        the resulting mesh is admissible of class ``m`` (the truncated functions
        acting on any cell span at most ``m`` successive levels), following the
        recursive refinement-neighborhood algorithm of Carraturo et al. (2019).  This
        assumes the current mesh is already admissible of class ``m`` (true for the
        root and for any mesh built via graded :meth:`refine`).  With
        ``admissible_class=None`` exactly the marked cells are refined (no grading).

        Args:
            cell_ids (npt.ArrayLike): Flat ids of active cells to refine.
            admissible_class (int | None): Admissibility class ``m >= 2`` to maintain,
                or ``None`` for ungraded refinement.  Defaults to ``2``.

        Returns:
            THBSplineSpace: A new space on the refined grid (same ``root_space``,
            ``truncate``, and ``regularity``).

        Raises:
            IndexError: If any id is outside ``[0, grid.num_cells)``.
            ValueError: If ``admissible_class`` is an integer ``< 2``.
        """
        return self._derived(self._impl.refine(_cell_ids(cell_ids), admissible_class))

    def refine_region(
        self,
        level: int,
        lo: Sequence[int],
        hi: Sequence[int],
        *,
        admissible_class: int | None = 2,
    ) -> THBSplineSpace:
        """Return a new space with the active cells in a rectangular region refined.

        The region is the integer cell-index box ``[lo, hi)`` at ``level`` (in
        level-``level`` coordinates), matching the convention of
        :meth:`pantr.grid.HierarchicalGrid.refine`.  Only the currently-active leaf
        cells inside the box are refined; the rest of the box (already refined, or
        not present at ``level``) is ignored.  If the box contains no active leaf
        cells, the call is a no-op and returns a space equivalent to ``self``.  This
        is the region-based counterpart of :meth:`refine`, which marks individual
        cells by flat id.

        Like :meth:`refine`, this does not mutate ``self`` or its grid: the grid is
        refined by rebinding, and a new :class:`THBSplineSpace` is returned.  Calls
        chain, so successive regions refine progressively (graded by default).

        Args:
            level (int): Level at which the box lives.  Must satisfy
                ``0 <= level <= grid.max_level``.
            lo (Sequence[int]): Per-direction start index (inclusive), in
                level-``level`` coordinates.
            hi (Sequence[int]): Per-direction end index (exclusive), in
                level-``level`` coordinates.
            admissible_class (int | None): Admissibility class ``m >= 2`` to maintain
                (graded refinement), or ``None`` for ungraded refinement.  Defaults
                to ``2``.  See :meth:`refine`.

        Returns:
            THBSplineSpace: A new space on the refined grid (same ``root_space``,
            ``truncate``, and ``regularity``).

        Raises:
            ValueError: If ``admissible_class`` is an integer ``< 2``, ``level`` is
                out of range, ``lo``/``hi`` have the wrong length, any
                ``lo[k] >= hi[k]``, or any part of ``[lo, hi)`` lies outside the
                level domain.
        """
        return self._derived(
            self._impl.refine_region(level, _cell_ids(lo), _cell_ids(hi), admissible_class)
        )

    def coarsen(
        self,
        cell_ids: npt.ArrayLike,
        *,
        admissible_class: int | None = 2,
    ) -> THBSplineSpace:
        """Return a new space with the marked cells coarsened away.

        A parent cell is reactivated (its children removed) only when **all** of its
        children are marked active leaves, mirroring the coarsening algorithm of
        Carraturo et al. (2019, Alg. 5).  That rule is
        :meth:`~pantr.grid.HierarchicalGrid.coarsen_cells`, which this method drives one
        parent at a time so the admissibility guard below can veto a parent without
        affecting the rest.  With ``admissible_class=None`` this is the exact inverse of
        :meth:`refine`: ``space.refine(cells).coarsen(children_of(cells))`` recovers
        ``space``.  With ``admissible_class=m`` the guard may suppress some coarsenings,
        so the recovery holds only when the guard permits them all.

        With ``admissible_class=m`` (the default ``m=2``) a parent is reactivated only
        if its coarsening neighborhood (Def. 3.5) is empty, so the resulting mesh stays
        admissible of class ``m``.  With ``admissible_class=None`` that guard is skipped.

        The space is immutable: the grid is coarsened by rebinding, and a new
        :class:`THBSplineSpace` is built on the result; ``self`` and its grid are
        unchanged.  An empty ``cell_ids``, and one that coarsens nothing, both return
        an equivalent new space over a copy of this space's grid -- a copy rather than
        the same object, so the two spaces never share the grid's tag registries.

        Args:
            cell_ids (npt.ArrayLike): Flat ids of active leaf cells to coarsen away.
                An empty array is valid and coarsens nothing.
            admissible_class (int | None): Admissibility class ``m >= 2`` to maintain,
                or ``None`` to skip the admissibility guard.  Defaults to ``2``.

        Returns:
            THBSplineSpace: A new space on the coarsened grid (same ``root_space``,
            ``truncate``, and ``regularity``).

        Raises:
            IndexError: If any id is outside ``[0, grid.num_cells)``.
            ValueError: If ``admissible_class`` is an integer ``< 2``.
        """
        return self._derived(self._impl.coarsen(_cell_ids(cell_ids), admissible_class))

    # ------------------------------------------------------------------
    # Computed here: basis tabulation
    # ------------------------------------------------------------------

    def _basis_1d_cached(
        self,
        level: int,
        k: int,
        order: int,
        flat_pts: npt.NDArray[np.float64],
        eval_cache: _EvalCache,
    ) -> _BasisEval1D:
        """Evaluate (and cache) the level-``level`` 1D basis (or a derivative).

        Args:
            level (int): Hierarchy level whose 1D space is evaluated.
            k (int): Parametric direction.
            order (int): Derivative order in direction ``k`` (``0`` for values).
            flat_pts (npt.NDArray[np.float64]): All parametric points of shape
                ``(num_pts, dim)``; column ``k`` is used.
            eval_cache (_EvalCache): Per-call cache keyed by ``(level, k, order)``.

        Returns:
            _BasisEval1D: ``(values, first_basis)`` where ``values`` holds the
            ``order``-th derivative of each local basis function (the function
            values when ``order == 0``).
        """
        key = (level, k, order)
        cached = eval_cache.get(key)
        if cached is None:
            sp1d = self.level_space(level).spaces[k]
            pts_k = np.ascontiguousarray(flat_pts[:, k])
            # Safety: _tabulate_orders validated flat_pts against cell_lo/hi (one
            # check per dimension via broadcasting) before calling here.  Cell bounds
            # are a strict subset of each level-space's parametric domain, so pts_k
            # is guaranteed in-domain.  Do NOT use validate=False from any other
            # call site without re-verifying this invariant.
            if order == 0:
                values, first_basis = sp1d.tabulate_basis(pts_k, validate=False)
                deriv = np.asarray(values, dtype=np.float64)
            else:
                all_deriv, first_basis = sp1d.tabulate_basis_derivatives(
                    pts_k, order, validate=False
                )
                deriv = np.asarray(all_deriv, dtype=np.float64)[:, order, :]
            cached = _BasisEval1D(deriv, np.asarray(first_basis, dtype=np.int64))
            eval_cache[key] = cached
        return cached

    def _truncated_column(
        self,
        entry: tuple[int, tuple[int, ...], npt.NDArray[np.float64]],
        orders: tuple[int, ...],
        flat_pts: npt.NDArray[np.float64],
        eval_cache: _EvalCache,
    ) -> npt.NDArray[np.float64]:
        """Evaluate one truncated function (or a derivative) from its coefficients.

        Computes ``sum_multi coeffs[multi] * prod_k D^orders[k] B^rep_{box_lo[k] +
        multi_k}(pt)`` over the stored coefficient box via a tensor contraction,
        where ``D^orders[k]`` is the ``orders[k]``-th derivative in direction ``k``.

        Args:
            entry (tuple[int, tuple[int, ...], npt.NDArray[np.float64]]):
                ``(rep_level, box_lo, coeffs)`` for the function, as :meth:`truncated`
                returns it.
            orders (tuple[int, ...]): Per-direction derivative orders (all ``0`` for
                function values).
            flat_pts (npt.NDArray[np.float64]): Points of shape ``(num_pts, dim)``.
            eval_cache (_EvalCache): Per-call 1D-basis evaluation cache.

        Returns:
            npt.NDArray[np.float64]: Function values of shape ``(num_pts,)``.

        Raises:
            NotImplementedError: If ``self.dim > _EINSUM_MAX_DIM`` (24); the
                einsum subscript scheme requires one letter per axis.
        """
        rep_level, box_lo, coeffs = entry
        dim = self.dim
        if dim > _EINSUM_MAX_DIM:
            raise NotImplementedError(
                f"_truncated_column uses single-letter einsum subscripts; "
                f"only dim <= {_EINSUM_MAX_DIM} is supported, got dim={dim}."
            )
        value_mats: list[npt.NDArray[np.float64]] = []
        for k in range(dim):
            values, first_basis = self._basis_1d_cached(
                rep_level, k, orders[k], flat_pts, eval_cache
            )
            degree_k = self.degrees[k]
            width = coeffs.shape[k]
            # local[p, j] = (box_lo[k] + j) - first_basis[p]; gather + mask in one shot.
            local = (box_lo[k] + np.arange(width))[None, :] - first_basis[:, None]
            valid = (local >= 0) & (local <= degree_k)
            gathered = np.take_along_axis(values, np.clip(local, 0, degree_k), axis=1)
            value_mats.append(np.where(valid, gathered, 0.0))

        letters = string.ascii_lowercase
        func_subs = letters[:dim]
        pt_sub = letters[dim]
        subscripts = f"{func_subs},{','.join(pt_sub + func_subs[k] for k in range(dim))}->{pt_sub}"
        column = np.asarray(np.einsum(subscripts, coeffs, *value_mats), dtype=np.float64)
        return column

    def _tabulate_orders(
        self,
        cid: int,
        pts: npt.ArrayLike,
        orders: tuple[int, ...],
        out_basis: npt.NDArray[np.float64] | None,
        out_dofs: npt.NDArray[np.int64] | None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Evaluate the active functions' ``orders`` mixed partial on cell ``cid``.

        Shared implementation for :meth:`tabulate_basis` (``orders`` all zero) and
        :meth:`tabulate_basis_derivatives`.

        Args:
            cid (int): Active cell flat id in ``[0, grid.num_cells)``.
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)`` lying in
                cell ``cid``.  Per axis, ``8 * eps * max(|lo|, |hi|, hi - lo)`` of slack
                is allowed at the cell boundary (see
                :func:`_cell_membership_tolerance`); points further outside raise
                :class:`ValueError`.
            orders (tuple[int, ...]): Per-direction derivative orders.
            out_basis (npt.NDArray[np.float64] | None): Optional output array of shape
                ``(..., K)`` with ``K = active_basis(cid).size``.  Allocated when
                ``None``.
            out_dofs (npt.NDArray[np.int64] | None): Optional output array of shape
                ``(K,)`` for the global dofs.  Allocated when ``None``.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: ``(values, dofs)``
            of shapes ``(..., K)`` and ``(K,)``.

        Raises:
            IndexError: If ``cid`` is out of range ``[0, grid.num_cells)``.
            ValueError: If ``pts`` does not have trailing dimension ``dim``, if any
                point lies outside cell ``cid``, or if ``out_basis``/``out_dofs`` has
                the wrong shape, dtype, or is not writeable.
        """
        contrib_dofs, contrib_levels, contrib_multis = self.contributions(cid)  # validates cid
        n_active = int(contrib_dofs.shape[0])
        dofs = np.array(contrib_dofs)

        pts_arr = np.asarray(pts, dtype=np.float64)
        if pts_arr.ndim == 0 or pts_arr.shape[-1] != self.dim:
            raise ValueError(
                f"pts must have trailing dimension {self.dim}; got shape {pts_arr.shape}."
            )
        lead = pts_arr.shape[:-1]
        num_pts = int(np.prod(lead)) if lead else 1
        flat_pts = pts_arr.reshape(num_pts, self.dim)

        cell_lo, cell_hi = self.grid.cell_bounds(cid)
        tol = _cell_membership_tolerance(cell_lo, cell_hi)
        if not (np.all(flat_pts >= cell_lo - tol) and np.all(flat_pts <= cell_hi + tol)):
            raise ValueError(
                f"pts must lie inside cell {cid!r} with bounds lo={cell_lo}, hi={cell_hi} "
                f"(slack {tol})."
            )

        out_shape = (*lead, n_active)

        if out_basis is None:
            result = np.empty(out_shape, dtype=np.float64)
        else:
            _check_out_array(out_basis, out_shape, np.float64, "out_basis")
            result = out_basis

        if out_dofs is None:
            dofs_result = dofs
        else:
            _check_out_array(out_dofs, (n_active,), np.int64, "out_dofs")
            out_dofs[...] = dofs
            dofs_result = out_dofs

        buffer = np.empty((num_pts, n_active), dtype=np.float64)
        eval_cache: _EvalCache = {}
        dim = self.dim
        degrees_arr = np.asarray(self.degrees, dtype=np.int64)
        max_order = int(degrees_arr.max()) + 1

        # Truncated functions are evaluated individually (coefficient contraction);
        # untruncated ones are grouped by level and combined in a single batched kernel
        # call (the common, hot case).
        untrunc_by_level: dict[int, list[int]] = {}
        for col in range(n_active):
            entry = self.truncated(int(contrib_dofs[col]))
            if entry is None:
                untrunc_by_level.setdefault(int(contrib_levels[col]), []).append(col)
            else:
                buffer[:, col] = self._truncated_column(entry, orders, flat_pts, eval_cache)

        for level, cols in untrunc_by_level.items():
            vals = np.zeros((dim, num_pts, max_order), dtype=np.float64)
            first_basis = np.empty((dim, num_pts), dtype=np.int64)
            for k in range(dim):
                values_k, fb_k = self._basis_1d_cached(level, k, orders[k], flat_pts, eval_cache)
                vals[k, :, : values_k.shape[1]] = values_k
                first_basis[k] = fb_k
            block = _combine_tp_values(
                vals,
                first_basis,
                np.ascontiguousarray(contrib_multis[cols], dtype=np.int64),
                degrees_arr,
            )
            buffer[:, cols] = block

        result[...] = buffer.reshape(out_shape)
        return result, dofs_result

    def tabulate_basis(
        self,
        cid: int,
        pts: npt.ArrayLike,
        out_basis: npt.NDArray[np.float64] | None = None,
        out_dofs: npt.NDArray[np.int64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        """Evaluate the active hierarchical functions on cell ``cid`` at ``pts``.

        Untruncated functions are a single tensor-product B-spline (the product of
        their 1D B-spline values).  Truncated functions are evaluated from their
        stored coefficients in the finest tensor-product basis their support reaches.
        The returned columns are ordered as ``dofs`` (the sorted global dofs, equal to
        :meth:`active_basis`), so no column belongs to a function that vanishes on the
        cell.  Mirrors the ``(basis, first_basis)`` two-return of
        :meth:`~pantr.bspline.BsplineSpace.tabulate_basis`.

        Args:
            cid (int): Active cell flat id in ``[0, grid.num_cells)``.
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)`` lying in
                cell ``cid``.  Per axis, ``8 * eps * max(|lo|, |hi|, hi - lo)`` of
                slack is allowed at the boundary; points further outside raise
                :class:`ValueError`.
            out_basis (npt.NDArray[np.float64] | None): Optional output array of shape
                ``(..., K)`` with ``K = active_basis(cid).size``.  Allocated when
                ``None``.
            out_dofs (npt.NDArray[np.int64] | None): Optional output array of shape
                ``(K,)`` for the dofs.  Allocated when ``None``.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: ``(values, dofs)`` of
            shapes ``(..., K)`` and ``(K,)``.

        Raises:
            IndexError: If ``cid`` is out of range ``[0, grid.num_cells)``.
            ValueError: If ``pts`` does not have trailing dimension ``dim``, if any
                point lies outside the bounds of cell ``cid``, or if ``out_basis`` /
                ``out_dofs`` has the wrong shape, dtype, or is not writeable.
        """
        return self._tabulate_orders(cid, pts, (0,) * self.dim, out_basis, out_dofs)

    def tabulate_basis_derivatives(
        self,
        cid: int,
        pts: npt.ArrayLike,
        orders: int | Sequence[int],
        out_basis: npt.NDArray[np.float64] | None = None,
        out_dofs: npt.NDArray[np.int64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        r"""Evaluate a mixed partial derivative of the active functions on cell ``cid``.

        Computes the single mixed partial :math:`\partial^{orders}` of each active
        hierarchical function, where ``orders[k]`` is the derivative order in
        parametric direction ``k`` (derivatives are with respect to the parametric
        coordinates).  Untruncated functions differentiate as a tensor product of 1D
        B-spline derivatives; truncated functions apply their stored coefficients to
        the B-spline derivatives at their representation level.  The returned columns
        are ordered as ``dofs`` (the sorted global dofs, equal to :meth:`active_basis`).

        Args:
            cid (int): Active cell flat id in ``[0, grid.num_cells)``.
            pts (npt.ArrayLike): Parametric points of shape ``(..., dim)`` lying in
                cell ``cid``.  Per axis, ``8 * eps * max(|lo|, |hi|, hi - lo)`` of
                slack is allowed at the boundary; points further outside raise
                :class:`ValueError`.
            orders (int | Sequence[int]): Per-direction derivative orders.  A scalar
                is broadcast to every direction.  Each entry must be ``>= 0``; orders
                exceeding the degree yield zero.
            out_basis (npt.NDArray[np.float64] | None): Optional output array of shape
                ``(..., K)`` with ``K = active_basis(cid).size``.  Allocated when
                ``None``.
            out_dofs (npt.NDArray[np.int64] | None): Optional output array of shape
                ``(K,)`` for the dofs.  Allocated when ``None``.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: ``(values, dofs)`` of
            shapes ``(..., K)`` and ``(K,)``.

        Raises:
            IndexError: If ``cid`` is out of range ``[0, grid.num_cells)``.
            ValueError: If ``orders`` has the wrong length or a negative entry, if
                ``pts`` does not have trailing dimension ``dim``, if any point lies
                outside cell ``cid``, or if ``out_basis`` / ``out_dofs`` has the wrong
                shape, dtype, or is not writeable.
        """
        if isinstance(orders, int):
            orders_t = (orders,) * self.dim
        else:
            orders_t = tuple(int(o) for o in orders)
            if len(orders_t) != self.dim:
                raise ValueError(
                    f"orders must be a scalar or length-{self.dim} sequence; "
                    f"got length {len(orders_t)}."
                )
        if any(o < 0 for o in orders_t):
            raise ValueError(f"orders must be non-negative; got {orders_t!r}.")
        return self._tabulate_orders(cid, pts, orders_t, out_basis, out_dofs)

    # ------------------------------------------------------------------
    # Computed here: the windowed restriction
    # ------------------------------------------------------------------

    def restrict(self, cell_ids: npt.ArrayLike) -> THBSplineSpaceRestriction:
        """Return the windowed sub-space over a subset of active cells.

        Windows this space to the root-cell-aligned bounding box of ``cell_ids``: the
        hierarchical grid is restricted (:meth:`pantr.grid.Grid.restrict`),
        the root space is windowed (:meth:`pantr.bspline.BsplineSpace.restrict`), and a
        new :class:`THBSplineSpace` is rebuilt on the sub-grid (re-running the Kraft
        active-function selection and truncation).

        Unlike the tensor-product :meth:`pantr.bspline.BsplineSpace.restrict`, the
        windowed THB basis equals the global one only over the **interior** cells --
        those whose entire (cross-level) function-support-closure lies inside the
        window -- because Kraft selection and truncation depend on the subdomain near
        the window boundary. Callers make the cells they care about interior by padding
        ``cell_ids`` with a support-closure halo.

        Args:
            cell_ids (npt.ArrayLike): Active cell flat ids to span; duplicates ignored.

        Returns:
            THBSplineSpaceRestriction: The windowed :class:`THBSplineSpace` and a
            read-only ``local_to_global_dof`` map; entry ``d`` is the global
            hierarchical dof of local dof ``d`` when the local function matches a
            globally-active function of the same level and multi-index, else ``-1``.
            Values are exact over interior cells; functions near the window boundary
            may map to ``-1``.

        Raises:
            ValueError: If ``cell_ids`` is empty.
            TypeError: If ``cell_ids`` is not integer-valued.
            IndexError: If any cell id is out of range ``[0, grid.num_cells)``.
        """
        grid_restr = self.grid.restrict(cell_ids)
        sub_grid = grid_restr.grid
        if not isinstance(sub_grid, HierarchicalGrid):
            raise RuntimeError(
                f"restrict: expected HierarchicalGrid from grid.restrict; "
                f"got {type(sub_grid).__name__!r}. This is a bug in HierarchicalGrid.restrict."
            )
        dim = self.dim
        factor = self.grid.factor

        # Root-cell bounding box of the window (the sub-grid's root spans it exactly).
        r_lo = [
            int(np.searchsorted(self.grid.root.breakpoints[k], sub_grid.root.breakpoints[k][0]))
            for k in range(dim)
        ]
        r_hi = [r_lo[k] + sub_grid.root.cells_per_axis[k] for k in range(dim)]

        # Window the root space to that box and rebuild the THB space on the sub-grid.
        root_ni = self.root_space.num_intervals
        box = [np.arange(r_lo[k], r_hi[k]) for k in range(dim)]
        root_cells = np.ravel_multi_index(
            tuple(m.ravel() for m in np.meshgrid(*box, indexing="ij")), root_ni
        )
        windowed_root = self.root_space.restrict(root_cells).space
        sub_space = THBSplineSpace(
            windowed_root, sub_grid, truncate=self.truncate, regularity=self.regularity
        )

        # Map each sub active function (level, sub_multi) to the global dof of the same
        # (level, sub_multi + per-level window origin), or -1 if not globally active.
        local_to_global_dof = np.full(sub_space.num_total_basis, -1, dtype=np.int64)
        offsets = self._offsets()
        sub_offset = 0
        for level in range(sub_space.num_levels):
            origin = [
                int(self._level_support(level)[k][0][r_lo[k] * factor[k] ** level])
                for k in range(dim)
            ]
            glob_num_basis = self.level_space(level).num_basis
            glob_active = self._active_at(level)
            glob_offset = int(offsets[level])
            sub_active = sub_space._active_at(level)
            sub_num_basis = sub_space.level_space(level).num_basis
            for sub_pos, sub_flat in enumerate(sub_active.tolist()):
                sub_multi = np.unravel_index(sub_flat, sub_num_basis)
                glob_flat = int(
                    np.ravel_multi_index(
                        tuple(int(sub_multi[k]) + origin[k] for k in range(dim)), glob_num_basis
                    )
                )
                gpos = int(np.searchsorted(glob_active, glob_flat))
                if gpos < glob_active.shape[0] and int(glob_active[gpos]) == glob_flat:
                    local_to_global_dof[sub_offset + sub_pos] = glob_offset + gpos
            sub_offset += int(sub_active.shape[0])
        assert sub_offset == sub_space.num_total_basis
        local_to_global_dof.flags.writeable = False
        return THBSplineSpaceRestriction(
            sub_space, local_to_global_dof, grid_restr.local_to_global_cell
        )

    # ------------------------------------------------------------------
    # Computed here: prolongation and restriction operators
    # ------------------------------------------------------------------

    def _finest_tp_coeffs(
        self,
        dof: int,
        oslo: tuple[tuple[npt.NDArray[np.float64], ...], ...],
        target_level: int,
    ) -> tuple[list[int], npt.NDArray[np.float64]]:
        """Express active function ``dof`` in the level-``target_level`` TP basis.

        Takes the function's native representation (a single B-spline for untruncated
        functions, the stored coefficients for truncated ones) and refines it purely
        (two-scale, no truncation) up to ``target_level``.

        Args:
            dof (int): Global active-function index.
            oslo (tuple[tuple[npt.NDArray[np.float64], ...], ...]): Per-level,
                per-direction two-scale matrices; must be indexed from at least
                ``0`` through ``target_level - 1``.  Only ``oslo[start..target_level-1]``
                is accessed, where ``start`` is the dof's native or representation level.
            target_level (int): Level whose TP basis the result is expressed in.

        Returns:
            tuple[list[int], npt.NDArray[np.float64]]: ``(box_lo, coeffs)`` over the
            level-``target_level`` function box.
        """
        dim = self.dim
        level = self.dof_level(dof)
        pos = dof - int(self._offsets()[level])
        flat = int(self._active_at(level)[pos])
        entry = self.truncated(dof)
        if entry is None:
            multi = np.unravel_index(flat, self.level_space(level).num_basis)
            box_lo = [int(multi[d]) for d in range(dim)]
            box_hi = [int(multi[d]) + 1 for d in range(dim)]
            coeffs = np.ones((1,) * dim, dtype=np.float64)
            start = level
        else:
            start = entry[0]
            box_lo = list(entry[1])
            box_hi = [entry[1][d] + entry[2].shape[d] for d in range(dim)]
            coeffs = entry[2]
        for lvl in range(start, target_level):
            coeffs, box_lo, box_hi = _refine_box(coeffs, box_lo, box_hi, oslo[lvl])
        return box_lo, coeffs

    def prolongation_to(self, fine: THBSplineSpace) -> npt.NDArray[np.float64]:
        """Return the prolongation matrix from this space to a refinement ``fine``.

        The hierarchical spaces are nested (``V_h ⊆ V_h'``), so every function of this
        (coarse) space lies in ``fine``.  The returned matrix ``P`` maps a
        coefficient vector in this space's basis to the coefficients of the **same
        function** in ``fine``'s basis: if ``u`` are coarse coefficients, ``P @ u`` are
        the fine coefficients.

        It is built column-by-column following the local two-scale construction used
        in practice (Garau & Vazquez 2018; D'Angella et al. 2018): each coarse
        function is matched against only the fine functions over its support,
        expressed in the deepest level present there (not the global finest level),
        and reproduced by a small local least-squares solve.  Functions far from the
        refinement yield trivial (identity) columns, so cost and sparsity follow the
        refined region.

        For a large refinement prefer :meth:`prolongation_to_sparse`, which computes the
        same entries with the same arithmetic but stores only the non-zeros: this dense
        variant allocates ``n_fine * n_coarse`` float64 regardless of how few columns the
        refinement actually touches.

        Args:
            fine (THBSplineSpace): A refinement of this space (same ``root_space``,
                ``factor``, ``regularity``, and ``truncate``; more levels / refined
                cells).

        Returns:
            npt.NDArray[np.float64]: Matrix ``P`` of shape
            ``(fine.num_total_basis, self.num_total_basis)``.

        Raises:
            TypeError: If ``fine`` is not a :class:`THBSplineSpace`.
            ValueError: If ``fine`` is not a refinement of this space (mismatched
                root/factor/regularity/truncation, fewer levels, or a prolongation
                residual above ``4096 * eps * (1 + max_coarse_value)``, where
                ``max_coarse_value`` is the largest coefficient of any coarse column).
        """
        self._check_is_refinement(fine)
        return self._assemble_prolongation(fine)

    def prolongation_to_sparse(self, fine: THBSplineSpace) -> sparse.csr_array:
        """Return the prolongation to a refinement ``fine`` in sparse CSR form.

        The same matrix :meth:`prolongation_to` returns, entry for entry: the columns come
        from the same local two-scale solves in the same order, so no entry is rounded
        differently and ``toarray()`` **compares equal** to the dense result. Only the
        storage differs.

        **Equal as values, not bit for bit**, and the gap is exactly the sign of a zero.
        The *stored* entries are bitwise what the solves produced, this array's own
        ``data`` included; but :meth:`scipy.sparse.csr_array.toarray` densifies by adding
        into a ``+0.0``-filled buffer, and ``+0.0 + -0.0`` is ``+0.0``, so a column that
        solved to a negative zero comes back positive. ``-0.0 == 0.0``, so nothing that
        compares values can see it, and nothing downstream of a matrix-vector product can
        either. ``tests/test_thb_spline_space.py`` pins both halves.

        That storage is the whole point at scale. The dense variant is
        ``O(n_fine * n_coarse)`` -- for ``1e5`` fine by ``9e4`` coarse dofs, some 72 GB --
        while the matrix has only a handful of entries per column: a coarse function is
        matched against the fine functions overlapping its support, and away from the
        refined region that is one entry of value ``1.0``. Sparse storage is ``O(nnz)``.

        Entries are stored exactly as solved, including any that happen to be zero, since
        they are the two-scale coefficients rather than a thresholded approximation. Call
        ``eliminate_zeros()`` on the result if a caller wants them dropped.

        Args:
            fine (THBSplineSpace): A refinement of this space, on the same terms as
                :meth:`prolongation_to`.

        Returns:
            scipy.sparse.csr_array: Matrix ``P`` of shape
            ``(fine.num_total_basis, self.num_total_basis)``, ``float64``.

        Raises:
            TypeError: If ``fine`` is not a :class:`THBSplineSpace`.
            ValueError: If ``fine`` is not a refinement of this space -- the same
                mismatches and residual check as :meth:`prolongation_to`.
        """
        self._check_is_refinement(fine)
        return self._assemble_prolongation_sparse(fine)

    def _check_is_refinement(self, fine: THBSplineSpace) -> None:
        """Raise unless ``fine`` is a structurally compatible refinement of ``self``.

        Checks everything decidable without assembling: type, dimension, truncation mode,
        subdivision factor, regularity, level count, and root knot vectors. The remaining
        condition -- that the coarse basis is actually reproducible in the fine one -- is
        a residual check during assembly.

        Args:
            fine (THBSplineSpace): Candidate refinement.

        Raises:
            TypeError: If ``fine`` is not a :class:`THBSplineSpace`.
            ValueError: If any structural property disagrees; the message lists every
                mismatch found, not just the first.
        """
        if not isinstance(fine, THBSplineSpace):
            raise TypeError(f"fine must be a THBSplineSpace; got {type(fine).__name__!r}.")
        mismatches: list[str] = []
        if fine.dim != self.dim:
            mismatches.append(f"dim: self={self.dim} vs fine={fine.dim}")
        if fine.truncate != self.truncate:
            mismatches.append(f"truncate: self={self.truncate} vs fine={fine.truncate}")
        if tuple(fine.grid.factor) != tuple(self.grid.factor):
            mismatches.append(f"factor: self={self.grid.factor} vs fine={fine.grid.factor}")
        if fine.regularity != self.regularity:
            mismatches.append(f"regularity: self={self.regularity} vs fine={fine.regularity}")
        if fine.num_levels < self.num_levels:
            mismatches.append(
                f"fine.num_levels={fine.num_levels} < self.num_levels={self.num_levels}"
            )
        if fine.dim == self.dim and not all(
            np.array_equal(fine.root_space.spaces[k].knots, self.root_space.spaces[k].knots)
            for k in range(self.dim)
        ):
            mismatches.append("root knot vectors differ")
        if mismatches:
            raise ValueError(
                "fine must be a refinement of this space; mismatches: "
                + "; ".join(mismatches)
                + "."
            )

    def _prolongation_columns(
        self, fine: THBSplineSpace
    ) -> Iterator[tuple[int, npt.NDArray[np.int64], npt.NDArray[np.float64]]]:
        """Yield the prolongation one column at a time, as ``(column, rows, values)``.

        The shared driver behind both :meth:`prolongation_to` and
        :meth:`prolongation_to_sparse`, so the two cannot drift: each coarse function is
        represented by only the fine functions over its support, expressed in a common
        *local* tensor-product level (the deepest present there) rather than the global
        finest level.  Functions far from the refinement stay at their own level (a trivial
        solve, often the identity), so the cost follows the refined region rather than the
        whole finest grid.  This is the local two-scale view of the change of basis (Garau
        & Vazquez 2018; D'Angella et al. 2018): coarse functions expanded by the refinement
        mask and matched against the active fine (truncated) basis.

        Columns with no candidate fine functions are skipped rather than yielded empty.

        Args:
            fine (THBSplineSpace): A validated refinement of ``self``.

        Yields:
            tuple[int, npt.NDArray[np.int64], npt.NDArray[np.float64]]: The coarse dof
            index, the fine dof indices its column occupies, and the values there.

        Raises:
            ValueError: If some column cannot reproduce its coarse function -- a
                residual above :func:`_prolongation_residual_tolerance` -- i.e. ``fine``
                is not a refinement of ``self``.

        Note:
            The residual is only known once every column has been solved, so the check
            runs after the last one. A consumer that abandons the iterator early therefore
            skips it; both callers here exhaust it.
        """
        oslo = _build_oslo_matrices([fine.level_space(lvl) for lvl in range(fine.num_levels)])
        root_cells = self.grid.level_cells_per_axis(0)
        n_coarse, n_fine = self.num_total_basis, fine.num_total_basis

        # Map each root cell to the fine functions whose support covers it.
        fine_box = [self._func_root_box(fine, j) for j in range(n_fine)]
        cell_to_fine: dict[int, list[int]] = {}
        for j, (_, lo, hi) in enumerate(fine_box):
            for cell in self._cells_in_box(lo, hi, root_cells):
                cell_to_fine.setdefault(cell, []).append(j)

        max_residual = 0.0
        max_coarse_val = 0.0
        for i in range(n_coarse):
            _, c_lo, c_hi = self._func_root_box(self, i)
            candidates = sorted(
                {
                    j
                    for c in self._cells_in_box(c_lo, c_hi, root_cells)
                    for j in cell_to_fine.get(c, [])
                }
            )
            sol, residual, coarse_val = self._prolong_column(fine, oslo, i, candidates, fine_box)
            if sol is not None:
                yield i, np.array(candidates, dtype=np.int64), sol
            max_residual = max(max_residual, residual)
            max_coarse_val = max(max_coarse_val, coarse_val)

        residual_tol = _prolongation_residual_tolerance(max_coarse_val)
        if max_residual > residual_tol:
            raise ValueError(
                f"fine is not a refinement of this space (prolongation residual "
                f"{max_residual:.2e}, above {residual_tol:.2e})."
            )

    def _assemble_prolongation(self, fine: THBSplineSpace) -> npt.NDArray[np.float64]:
        """Scatter the prolongation columns into a dense matrix.

        Args:
            fine (THBSplineSpace): A validated refinement of ``self``.

        Returns:
            npt.NDArray[np.float64]: Prolongation ``P`` of shape
            ``(fine.num_total_basis, self.num_total_basis)``.

        Raises:
            ValueError: If a column cannot reproduce its coarse function -- a residual
                above :func:`_prolongation_residual_tolerance` -- i.e. ``fine`` is not a
                refinement of ``self``.
        """
        shape = (fine.num_total_basis, self.num_total_basis)
        prolongation: npt.NDArray[np.float64] = np.zeros(shape, dtype=np.float64)
        for column, rows, values in self._prolongation_columns(fine):
            prolongation[rows, column] = values
        return prolongation

    def _assemble_prolongation_sparse(self, fine: THBSplineSpace) -> sparse.csr_array:
        """Gather the prolongation columns into a CSR array.

        Args:
            fine (THBSplineSpace): A validated refinement of ``self``.

        Returns:
            scipy.sparse.csr_array: Prolongation ``P`` of shape
            ``(fine.num_total_basis, self.num_total_basis)``, ``float64``.

        Raises:
            ValueError: If a column cannot reproduce its coarse function -- a residual
                above :func:`_prolongation_residual_tolerance` -- i.e. ``fine`` is not a
                refinement of ``self``.

        Note:
            Assembled through COO, whose duplicate summation never fires: each column is
            emitted exactly once and its rows are distinct, so the stored values are the
            solved ones unaltered.
        """
        shape = (fine.num_total_basis, self.num_total_basis)
        all_rows: list[npt.NDArray[np.int64]] = []
        all_cols: list[npt.NDArray[np.int64]] = []
        all_values: list[npt.NDArray[np.float64]] = []
        for column, rows, values in self._prolongation_columns(fine):
            all_rows.append(rows)
            all_cols.append(np.full(rows.shape[0], column, dtype=np.int64))
            all_values.append(values)

        if not all_values:
            return sparse.csr_array(shape, dtype=np.float64)
        coo = sparse.coo_array(
            (
                np.concatenate(all_values),
                (np.concatenate(all_rows), np.concatenate(all_cols)),
            ),
            shape=shape,
        )
        return coo.tocsr()

    def _prolong_column(
        self,
        fine: THBSplineSpace,
        oslo: tuple[tuple[npt.NDArray[np.float64], ...], ...],
        i: int,
        candidates: list[int],
        fine_box: list[tuple[int, list[int], list[int]]],
    ) -> tuple[npt.NDArray[np.float64] | None, float, float]:
        """Solve the local system reproducing coarse function ``i`` in the fine basis.

        Args:
            fine (THBSplineSpace): The refinement.
            oslo (tuple[tuple[npt.NDArray[np.float64], ...], ...]): Per-level,
                per-direction two-scale matrices of ``fine``.
            i (int): Coarse global dof whose column is computed.
            candidates (list[int]): Fine dofs whose support covers ``i``'s support.
            fine_box (list[tuple[int, list[int], list[int]]]): ``(rep_level, lo, hi)``
                per fine dof (from :meth:`_func_root_box`).

        Returns:
            tuple[npt.NDArray[np.float64] | None, float, float]: ``(solution,
            residual, max_coarse_value)``.  ``solution`` is the column restricted to
            ``candidates`` (``None`` if there are no candidates).
        """
        c_level = self._func_root_box(self, i)[0]
        level = max([c_level, *(fine_box[j][0] for j in candidates)])
        num_basis = fine.level_space(level).num_basis
        c_flats, c_vals = self._tp_column(self, i, oslo, level, num_basis)
        coarse_val = float(np.abs(c_vals).max()) if c_vals.size else 0.0
        cols = [self._tp_column(fine, j, oslo, level, num_basis) for j in candidates]
        rows = sorted(
            {int(f) for f in c_flats.tolist()}
            | {int(f) for flats, _ in cols for f in flats.tolist()}
        )
        row_of = {d: r for r, d in enumerate(rows)}
        rhs = np.zeros(len(rows), dtype=np.float64)
        rhs[[row_of[int(f)] for f in c_flats.tolist()]] = c_vals
        if not candidates:
            return None, (float(np.abs(rhs).max()) if rhs.size else 0.0), coarse_val
        amat = np.zeros((len(rows), len(candidates)), dtype=np.float64)
        for cj, (flats, vals) in enumerate(cols):
            amat[[row_of[int(f)] for f in flats.tolist()], cj] = vals
        sol, *_ = np.linalg.lstsq(amat, rhs, rcond=None)
        residual = float(np.abs(amat @ sol - rhs).max()) if rhs.size else 0.0
        return np.asarray(sol, dtype=np.float64), residual, coarse_val

    @staticmethod
    def _func_root_box(space: THBSplineSpace, dof: int) -> tuple[int, list[int], list[int]]:
        """Return ``(rep_level, lo, hi)``: the inclusive root-cell support box of ``dof``.

        Args:
            space (THBSplineSpace): The space owning ``dof``.
            dof (int): Global active-function index.

        Returns:
            tuple[int, list[int], list[int]]: The function's representation level
            (``rep_level`` if truncated, else its native level) and the inclusive
            per-direction root-cell support box ``(lo, hi)``.
        """
        dim = space.dim
        factor = space.grid.factor
        level = space.dof_level(dof)
        pos = dof - int(space._offsets()[level])
        flat = int(space._active_at(level)[pos])
        multi = np.unravel_index(flat, space.level_space(level).num_basis)
        sup = space._level_support(level)
        lo = [0] * dim
        hi = [0] * dim
        for k in range(dim):
            p = factor[k] ** level
            lo[k] = int(sup[k][1][int(multi[k])]) // p
            hi[k] = int(sup[k][2][int(multi[k])]) // p
        entry = space.truncated(dof)
        return (entry[0] if entry is not None else level), lo, hi

    @staticmethod
    def _cells_in_box(lo: list[int], hi: list[int], root_cells: tuple[int, ...]) -> list[int]:
        """Return the flat root-cell ids inside the inclusive box ``[lo, hi]``.

        Args:
            lo (list[int]): Per-direction lower root-cell index (inclusive).
            hi (list[int]): Per-direction upper root-cell index (inclusive).
            root_cells (tuple[int, ...]): Per-direction root-cell counts.

        Returns:
            list[int]: Flat root-cell ids in the box.
        """
        prod = itertools.product(*(range(lo[k], hi[k] + 1) for k in range(len(lo))))
        return [int(np.ravel_multi_index(c, root_cells)) for c in prod]

    @staticmethod
    def _tp_column(
        space: THBSplineSpace,
        dof: int,
        oslo: tuple[tuple[npt.NDArray[np.float64], ...], ...],
        lvl: int,
        num_basis: tuple[int, ...],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        """Express ``dof`` in the level-``lvl`` tensor-product basis as flat indices and coeffs.

        Args:
            space (THBSplineSpace): The space owning ``dof``.
            dof (int): Global active-function index.
            oslo (tuple[tuple[npt.NDArray[np.float64], ...], ...]): Two-scale matrices
                covering levels up to ``lvl``.
            lvl (int): Target level (``>=`` the function's representation level).
            num_basis (tuple[int, ...]): Per-direction function counts at ``lvl``.

        Returns:
            tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]: Flat level-``lvl``
            TP indices and the corresponding coefficients.
        """
        box_lo, coeffs = space._finest_tp_coeffs(dof, oslo, lvl)
        ranges = [np.arange(box_lo[d], box_lo[d] + coeffs.shape[d]) for d in range(space.dim)]
        mesh = np.meshgrid(*ranges, indexing="ij")
        flats = np.ravel_multi_index([g.ravel() for g in mesh], num_basis)
        return flats, coeffs.ravel()

    def restriction_to(self, coarse: THBSplineSpace) -> npt.NDArray[np.float64]:
        """Return the restriction matrix from this space to a coarsening ``coarse``.

        ``self`` must be a refinement of ``coarse``.  The restriction is the algebraic
        pseudo-inverse of the prolongation, ``R = pinv(P)`` with
        ``P = coarse.prolongation_to(self)``.  It is assembly-free (no mass matrix) and
        satisfies ``R @ P == I``, so restricting a prolonged coarse field recovers it
        exactly; for a general fine field ``R @ u_fine`` is the least-squares
        (coefficient-space) projection onto the coarse space.

        Args:
            coarse (THBSplineSpace): A coarsening of this space (``self`` is a
                refinement of ``coarse``).

        Returns:
            npt.NDArray[np.float64]: Matrix ``R`` of shape
            ``(coarse.num_total_basis, self.num_total_basis)``.

        Raises:
            TypeError: If ``coarse`` is not a :class:`THBSplineSpace`.
            ValueError: If ``self`` is not a refinement of ``coarse``.
        """
        if not isinstance(coarse, THBSplineSpace):
            raise TypeError(f"coarse must be a THBSplineSpace; got {type(coarse)!r}.")
        prolongation = coarse.prolongation_to(self)
        restriction: npt.NDArray[np.float64] = np.asarray(
            np.linalg.pinv(prolongation), dtype=np.float64
        )
        return restriction


def _rebuild_thb_space(
    root_space: BsplineSpace,
    grid: HierarchicalGrid,
    truncate: bool,
    regularity: Sequence[int | None],
) -> THBSplineSpace:
    """Rebuild a pickled hierarchical space from the constructor's arguments.

    A module-level function because ``truncate`` and ``regularity`` are keyword-only
    on :class:`THBSplineSpace` and a ``(callable, args)`` reduction has nowhere to put
    a keyword; naming the class directly would need them positional, which the public
    signature deliberately is not.

    Args:
        root_space (BsplineSpace): The level-0 tensor-product space.
        grid (HierarchicalGrid): The active-cell hierarchy.
        truncate (bool): Whether the truncated (THB) basis was built.
        regularity (Sequence[int | None]): Per-direction continuity, already broadcast.

    Returns:
        THBSplineSpace: The reconstructed space, built under the **reader's** backend
        rather than the writer's, which is what makes the wire format backend-free.
    """
    return THBSplineSpace(root_space, grid, truncate=truncate, regularity=regularity)


class THBSplineSpaceRestriction(NamedTuple):
    """Result of :meth:`THBSplineSpace.restrict`: a windowed THB space with its maps.

    - ``space`` -- the windowed :class:`THBSplineSpace` rebuilt on the restricted grid;
      its basis equals the global basis pointwise over interior cells (those whose
      function-support-closure lies inside the window).
    - ``local_to_global_dof`` -- read-only ``(space.num_total_basis,)`` map; entry ``d``
      is the global hierarchical dof of local dof ``d`` when the local function matches a
      globally-active function of the same level and multi-index, else ``-1`` (local
      function active in the sub-space but absent from the global active set -- arises
      near the window boundary where Kraft selection may differ). Values are exact over
      interior cells.
    - ``local_to_global_cell`` -- read-only ``(space.grid.num_cells,)`` map; entry ``c``
      is the global flat cell id of local cell ``c``. Same ordering as
      :attr:`space.grid <THBSplineSpace.grid>`.
    """

    space: THBSplineSpace
    local_to_global_dof: npt.NDArray[np.int64]
    local_to_global_cell: npt.NDArray[np.int64]


def create_thb_space(
    root: BsplineSpace,
    factor: int | Sequence[int] = 2,
    *,
    truncate: bool = True,
    regularity: int | Sequence[int | None] | None = None,
) -> THBSplineSpace:
    """Create a trivial (unrefined) THB-spline space from a B-spline space.

    Convenience factory that wraps ``root`` in a single-level
    :class:`~pantr.grid.HierarchicalGrid` (its knot-span grid, ready to subdivide by
    ``factor``) and builds the corresponding :class:`THBSplineSpace`.  The result has
    one level and its active basis coincides with that of ``root``; refine it with
    :meth:`THBSplineSpace.refine` or :meth:`THBSplineSpace.refine_region`.

    It is the ergonomic entry point for lifting an existing :class:`BsplineSpace` into
    a single-level hierarchy, leaving the two-argument :class:`THBSplineSpace`
    constructor for callers that build the hierarchical grid explicitly.

    Args:
        root (BsplineSpace): The level-0 tensor-product B-spline space.
        factor (int | Sequence[int]): Per-direction subdivision factor used when the
            space is later refined.  A scalar is broadcast to every axis.  Each entry
            must be ``>= 1``.  Defaults to ``2`` (dyadic refinement).
        truncate (bool): If ``True`` (default), build the truncated (THB) basis; if
            ``False``, build the non-truncated hierarchical (HB) basis.
        regularity (int | Sequence[int | None] | None): Per-direction continuity at
            the knots inserted when subdividing to finer levels.  See
            :class:`THBSplineSpace`.  Defaults to ``None`` (maximal smoothness).

    Returns:
        THBSplineSpace: A single-level THB space over ``root``.

    Raises:
        ValueError: If any ``factor`` entry is ``< 1``, ``factor`` has the wrong
            length, or ``regularity`` is out of range for ``root``'s degrees.

    Example:
        >>> from pantr.bspline import create_uniform_space, create_thb_space
        >>> thb = create_thb_space(create_uniform_space([2, 2], [8, 8]))
        >>> thb.num_levels
        1
        >>> thb = thb.refine_region(0, [0, 0], [4, 4])  # refine the lower-left quarter
        >>> thb.num_levels
        2
    """
    grid = hierarchical_grid(tensor_product_grid(root), factor)
    return THBSplineSpace(root, grid, truncate=truncate, regularity=regularity)
