"""Regression tests for Bernstein ratio-recurrence underflow near u=1 (issue #258).

The O(p) ratio recurrence in ``_bernstein_point`` seeds the forward pass from
``B_0 = (1 - u)^p``, which underflows to exact zero for ``u`` close enough to
1 at high degree. Once the seed flushes, every subsequent term (a positive
multiple of the previous one) stays zero, so ``sum_i B_i(u)`` collapses to 0
instead of 1: partition of unity is silently broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from pantr.basis._basis_core import _bernstein_point


@pytest.mark.xfail(
    reason="issue #258: forward recurrence seed (1-u)**p underflows to exact 0.0 "
    "for u this close to 1 at degree 30, so sum(B_i) collapses to 0 instead of 1",
    strict=True,
)
def test_partition_of_unity_degree30_near_u1_regression() -> None:
    """Partition of unity fails for degree 30 at u = 1 - 1e-16 (pre-fix)."""
    degree = 30
    u = 1.0 - 1e-16
    out_row = np.empty(degree + 1, dtype=np.float64)
    _bernstein_point(np.int32(degree), np.float64(u), out_row)

    eps = np.finfo(np.float64).eps
    assert abs(out_row.sum() - 1.0) <= 8 * degree * eps
