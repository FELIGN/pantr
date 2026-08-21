"""The Python implementations of the change-of-basis builders.

Layer 3 in the layering of ``CLAUDE.md``: pure computation, no validation, and
the parity oracle the C++ port in ``cpp/include/pantr/change_basis`` is measured
against.

Every kernel here takes its nodes, and where it needs one its quadrature weights,
as arrays rather than computing them. Two reasons, and the second is the binding
one. It keeps these signatures identical to the C++ ones, so the catalogue in
:mod:`pantr.change_basis._change_basis_backend` selects between two functions
rather than between two conventions. And two of the five Lagrange node families
are ones :mod:`pantr._backend` says are deliberately never dispatched, so a kernel
that fetched its own nodes would have to reach through the dispatch to get them.

Why these reach for Layer 3 rather than the public tabulators
-------------------------------------------------------------

:func:`pantr.basis.tabulate_bernstein_1d` and its siblings are the **public Layer
1 surface**, and they are dispatched. Calling them from here would be wrong twice
over. It would follow whatever backend is ambient, so asking the catalogue for
*this* kernel while the ambient backend is C++ would run a Python solve over C++
tabulations -- a third implementation that is neither backend and that no parity
claim covers. And it would make these functions Layer 1 consumers while their C++
counterparts call Layer 3 kernels, so the two halves of one catalogue entry would
not be peers: this side would validate its inputs and raise, the other would not.

So these ask :mod:`pantr.basis._basis_backend` for the **Python** kernels by name
and drive them through the same Layer 2 helper the public functions use. That
makes "the Python kernel" mean one thing regardless of ambient state, which is the
property a parity oracle has to have, and it makes the docstring note below true
rather than aspirational. Measured, it also removes about 6.7 us per tabulation at
degree 6 -- roughly six times the raw kernel's own cost -- paid twice inside every
inverse builder.

The helper is Layer 2 rather than Layer 3, and that is deliberate: it owns the
dtype normalization and the ``out`` allocation these kernels would otherwise
duplicate, and it is what selects the serial Bernstein twin below
``_PARALLEL_MIN_NUM_PTS``. What it does not do is dispatch, which is the whole
point.
"""

from __future__ import annotations

from collections.abc import Callable
from math import comb

import numpy as np
import numpy.typing as npt

from .._backend import Backend
from ..basis._basis_1D import _tabulate_basis_1D_impl_helper
from ..basis._basis_backend import (
    CoreKernels,
    bernstein_core,
    cardinal_bspline_core,
    legendre_core,
)

_Array = npt.NDArray[np.float32 | np.float64]
"""A float32 or float64 array, the only two dtypes these kernels handle."""


def _tabulate_in_python(
    which: Callable[[Backend], CoreKernels],
    degree: int,
    nodes: _Array,
    out: _Array | None = None,
) -> _Array:
    """Tabulate one basis with the Python kernels, whatever backend is ambient.

    Args:
        which (Callable[[Backend], CoreKernels]): The basis's catalogue accessor.
        degree (int): Degree of the basis.
        nodes (_Array): Evaluation points.
        out (_Array | None): Output array, or ``None`` to allocate. Defaults to None.

    Returns:
        _Array: The tabulated values, shape ``(nodes.size, degree + 1)``.
    """
    kernels = which(Backend.PYTHON)
    return _tabulate_basis_1D_impl_helper(
        degree, nodes, kernels.parallel, out, core_func_serial=kernels.serial
    )


def _gram_projection(new_basis: _Array, old_basis: _Array, weights: _Array, out: _Array) -> None:
    """Solve ``G M^T = C`` for the change-of-basis matrix ``M``.

    ``G`` is the Gram matrix of the new basis under the given quadrature and ``C``
    the mixed matrix between the two bases, so row ``i`` of the result holds the
    ``i``-th old basis function expanded in the new basis.

    Args:
        new_basis (_Array): New-basis values at the quadrature nodes,
            ``(n_quad, n_new)``.
        old_basis (_Array): Old-basis values at the same nodes, ``(n_quad, n_old)``.
        weights (_Array): Quadrature weights, length ``n_quad``.
        out (_Array): Output of shape ``(n_old, n_new)``, filled in place.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call one of the ``compute_*`` builders instead.
    """
    weights_diag = np.diag(weights)
    gram = new_basis.T @ weights_diag @ new_basis
    mixed = new_basis.T @ weights_diag @ old_basis
    out[:] = np.linalg.solve(gram, mixed).T


def _invert(forward: _Array, out: _Array) -> None:
    """Invert ``forward`` by one LU solve against the identity.

    Args:
        forward (_Array): The square matrix to invert.
        out (_Array): Output of the same shape, filled in place.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call one of the ``compute_*`` builders instead.
    """
    out[:] = np.linalg.solve(forward, np.eye(forward.shape[0], dtype=out.dtype))


def _lagrange_to_bernstein_core(degree: int, nodes: _Array, out: _Array) -> None:
    """Tabulate the Bernstein basis at the Lagrange nodes, transposed.

    The Lagrange basis is cardinal at its own nodes, so ``C[j, k]`` is simply
    ``B_j(x_k)`` and no solve is involved at any degree.

    Args:
        degree (int): Polynomial degree. Assumed at least 1.
        nodes (_Array): The ``degree + 1`` Lagrange nodes.
        out (_Array): Output of shape ``(degree + 1, degree + 1)``, filled in place.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`~pantr.change_basis.compute_lagrange_to_bernstein_1d`.
    """
    _tabulate_in_python(bernstein_core, degree, nodes, out.T)


def _bernstein_to_lagrange_core(degree: int, nodes: _Array, out: _Array) -> None:
    """Invert the Lagrange-to-Bernstein matrix.

    Args:
        degree (int): Polynomial degree. Assumed at least 1.
        nodes (_Array): The ``degree + 1`` Lagrange nodes.
        out (_Array): Output of shape ``(degree + 1, degree + 1)``, filled in place.

    Note:
        Inputs are assumed to be correct (no validation performed), including the
        degree domain that keeps the solve non-singular.
        For general use, call :func:`~pantr.change_basis.compute_bernstein_to_lagrange_1d`.
    """
    forward = np.empty_like(out)
    _lagrange_to_bernstein_core(degree, nodes, forward)
    _invert(forward, out)


def _bernstein_to_cardinal_core(degree: int, nodes: _Array, weights: _Array, out: _Array) -> None:
    """Project the cardinal B-spline basis onto the Bernstein basis.

    Args:
        degree (int): Polynomial degree. Assumed non-negative.
        nodes (_Array): Quadrature nodes on ``[0, 1]``, ``degree + 1`` of them.
        weights (_Array): The matching quadrature weights.
        out (_Array): Output of shape ``(degree + 1, degree + 1)``, filled in place.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`~pantr.change_basis.compute_bernstein_to_cardinal_1d`.
    """
    new_basis = _tabulate_in_python(bernstein_core, degree, nodes)
    old_basis = _tabulate_in_python(cardinal_bspline_core, degree, nodes)
    _gram_projection(new_basis, old_basis, weights, out)


def _cardinal_to_bernstein_core(degree: int, nodes: _Array, weights: _Array, out: _Array) -> None:
    """Invert the Bernstein-to-cardinal matrix.

    Args:
        degree (int): Polynomial degree. Assumed non-negative.
        nodes (_Array): Quadrature nodes on ``[0, 1]``, ``degree + 1`` of them.
        weights (_Array): The matching quadrature weights.
        out (_Array): Output of shape ``(degree + 1, degree + 1)``, filled in place.

    Note:
        Inputs are assumed to be correct (no validation performed), including the
        degree domain.
        For general use, call :func:`~pantr.change_basis.compute_cardinal_to_bernstein_1d`.
    """
    forward = np.empty_like(out)
    _bernstein_to_cardinal_core(degree, nodes, weights, forward)
    _invert(forward, out)


def _legendre_to_cardinal_core(degree: int, nodes: _Array, weights: _Array, out: _Array) -> None:
    """Project the cardinal B-spline basis onto the orthonormal Legendre basis.

    Args:
        degree (int): Polynomial degree. Assumed non-negative.
        nodes (_Array): Quadrature nodes on ``[0, 1]``, ``degree + 1`` of them.
        weights (_Array): The matching quadrature weights.
        out (_Array): Output of shape ``(degree + 1, degree + 1)``, filled in place.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`~pantr.change_basis.compute_legendre_to_cardinal_1d`.
    """
    new_basis = _tabulate_in_python(legendre_core, degree, nodes)
    old_basis = _tabulate_in_python(cardinal_bspline_core, degree, nodes)
    _gram_projection(new_basis, old_basis, weights, out)


def _cardinal_to_legendre_core(degree: int, nodes: _Array, weights: _Array, out: _Array) -> None:
    """Invert the Legendre-to-cardinal matrix.

    Args:
        degree (int): Polynomial degree. Assumed non-negative.
        nodes (_Array): Quadrature nodes on ``[0, 1]``, ``degree + 1`` of them.
        weights (_Array): The matching quadrature weights.
        out (_Array): Output of shape ``(degree + 1, degree + 1)``, filled in place.

    Note:
        Inputs are assumed to be correct (no validation performed), including the
        degree domain.
        For general use, call :func:`~pantr.change_basis.compute_cardinal_to_legendre_1d`.
    """
    forward = np.empty_like(out)
    _legendre_to_cardinal_core(degree, nodes, weights, forward)
    _invert(forward, out)


def _cardinal_dual_legendre_coeffs_core(
    degree: int, nodes: _Array, weights: _Array, out: _Array
) -> None:
    """Transpose the cardinal-to-Legendre matrix.

    Args:
        degree (int): Polynomial degree. Assumed non-negative.
        nodes (_Array): Quadrature nodes on ``[0, 1]``, ``degree + 1`` of them.
        weights (_Array): The matching quadrature weights.
        out (_Array): Output of shape ``(degree + 1, degree + 1)``, filled in place.

    Note:
        Inputs are assumed to be correct (no validation performed), including the
        degree domain.
        For general use, call
        :func:`~pantr.change_basis.compute_cardinal_dual_legendre_coeffs_1d`.
    """
    straight = np.empty_like(out)
    _cardinal_to_legendre_core(degree, nodes, weights, straight)
    out[:] = straight.T


def _monomial_to_bernstein_core(degree: int, out: _Array) -> None:
    """Fill the lower-triangular matrix ``C(i, j) / C(degree, j)``.

    Args:
        degree (int): Polynomial degree. Assumed non-negative.
        out (_Array): Output of shape ``(degree + 1, degree + 1)``, filled in place.

    Note:
        Inputs are assumed to be correct (no validation performed).
        For general use, call :func:`~pantr.change_basis.compute_monomial_to_bernstein_1d`.
    """
    out[:] = 0
    for i in range(degree + 1):
        for j in range(i + 1):
            out[i, j] = comb(i, j) / comb(degree, j)
