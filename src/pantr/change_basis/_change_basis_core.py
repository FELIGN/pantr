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

Why these call the Python tabulators explicitly
-----------------------------------------------

:func:`pantr.basis.tabulate_bernstein_1d` and its siblings are themselves
dispatched now. A kernel here that called them plainly would follow whatever
backend is in effect, so asking the catalogue for *this* kernel while the ambient
backend is C++ would run a Python solve over C++ tabulations -- a third
implementation that is neither backend and that no parity claim covers.

So the tabulations below are taken inside ``use_backend(Backend.PYTHON)``. It
makes "the Python kernel" mean one thing regardless of ambient state, which is
the property a parity oracle has to have. The cost is one
:class:`~contextvars.ContextVar` set and reset per call, in front of an
``O(n^3)`` solve.
"""

from __future__ import annotations

from math import comb

import numpy as np
import numpy.typing as npt

from .._backend import Backend, use_backend
from ..basis import (
    tabulate_bernstein_1d,
    tabulate_cardinal_bspline_1d,
    tabulate_legendre_1d,
)

_Array = npt.NDArray[np.float32 | np.float64]
"""A float32 or float64 array, the only two dtypes these kernels handle."""


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
    with use_backend(Backend.PYTHON):
        tabulate_bernstein_1d(degree, nodes, out=out.T)


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
    with use_backend(Backend.PYTHON):
        new_basis = tabulate_bernstein_1d(degree, nodes)
        old_basis = tabulate_cardinal_bspline_1d(degree, nodes)
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
    with use_backend(Backend.PYTHON):
        new_basis = tabulate_legendre_1d(degree, nodes)
        old_basis = tabulate_cardinal_bspline_1d(degree, nodes)
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
