"""Regression tests for the Cox-de Boor denominator guard (issue #257).

The ``BasisFuncs``/``DerBasisFuncs`` kernels in ``_bspline_basis_core.py`` guard the
Cox-de Boor recurrence denominator with ``denom < tol`` where ``tol`` is an absolute,
per-dtype preset (see ``pantr.tolerance``) and ``denom`` is a knot difference. Because
``tol`` does not scale with the knot vector's parametric span, the guard is not
invariant under affine reparametrization of the knots: a genuinely nonzero knot span
on a tiny domain can be smaller than ``tol`` and get incorrectly zeroed, breaking the
partition of unity.

The xfail test below isolates this behavior at the kernel level (bypassing
:class:`~pantr.bspline.BsplineSpace1D`'s construction-time knot snapping, which is a
separate concern) with a knot vector whose domain is small enough that a genuine,
nonzero local knot difference falls below the absolute tolerance.
"""

import numpy as np
import pytest

from pantr.bspline._bspline_basis_core import _compute_basis_nurbs_book_serial_impl


class TestScaleDependentDenominatorGuard:
    """Known-bug regression: absolute tol guard is not affine-invariant."""

    @pytest.mark.xfail(
        strict=True,
        reason="issue #257: absolute tol guard zeroes genuine tiny-domain knot spans",
    )
    def test_tiny_domain_partition_of_unity(self) -> None:
        """A tiny-but-nonzero domain must still satisfy the partition of unity.

        Knot spans are scaled down to ``1e-16`` while ``tol`` stays at the float64
        strict preset (``1e-15``, see :func:`pantr.tolerance.get_strict`). Every local
        knot span is then well below ``tol`` even though it is not zero, so the
        current absolute-tolerance guard incorrectly collapses every Cox-de Boor
        contribution and the basis functions no longer sum to one.
        """
        scale = 1e-16
        knots = np.array([0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0], dtype=np.float64) * scale
        degree = 2
        tol = 1e-15  # spline.tolerance for float64 (strict preset)
        pts = np.array([0.4 * scale], dtype=np.float64)

        basis = np.empty((1, degree + 1), dtype=np.float64)
        first_basis = np.empty(1, dtype=np.int_)
        _compute_basis_nurbs_book_serial_impl(knots, degree, False, tol, pts, basis, first_basis)

        np.testing.assert_allclose(basis.sum(axis=-1), 1.0, rtol=1e-13)
