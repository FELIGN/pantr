"""Tests for the exact binomial-coefficient kernel (``_bincoeff``)."""

from __future__ import annotations

import math

import pytest

from pantr.bspline._bspline_degree_core import _bincoeff

_EXACT_UP_TO: int = 61
"""
Largest ``n`` for which ``_bincoeff(n, k)`` is exact for every ``k``.

The multiplicative recurrence's largest intermediate is ``C(n, k) * min(k, n - k)``, and
``max_k C(61, k) * min(k, 61 - k) = 6.98e18`` still fits in ``int64`` while the ``n = 62``
value ``1.44e19`` does not. :func:`test_the_envelope_boundary_is_where_int64_ends` pins
both halves of that statement, so this constant cannot drift away from its derivation.
"""

_FLOAT_EXACT_UP_TO: int = 56
"""
Largest ``n`` for which every ``C(n, k)`` is representable in ``float64``.

``C(56, 28) = 7.65e15`` is below ``2**53 = 9.01e15`` and ``C(57, 28) = 1.50e16`` is above
it. Past this point the kernel still computes the exact integer, but the ``float`` return
carries only its correctly-rounded value.
"""


class TestBincoeffExactness:
    """The kernel agrees with :func:`math.comb` everywhere inside its envelope."""

    def test_exhaustive_up_to_the_envelope(self) -> None:
        """Every ``(n, k)`` with ``n <= 61`` matches ``math.comb`` exactly."""
        mismatches = [
            (n, k)
            for n in range(_EXACT_UP_TO + 1)
            for k in range(n + 1)
            if _bincoeff(n, k) != float(math.comb(n, k))
        ]
        assert mismatches == []

    @pytest.mark.parametrize(
        ("n", "k", "expected"),
        [
            (48, 25, 30957699535776),
            (57, 28, 15033633249770520),
            (60, 30, 118264581564861424),
            (61, 30, 232714176627630544),
        ],
    )
    def test_values_the_lgamma_route_got_wrong(self, n: int, k: int, expected: int) -> None:
        """Hardcoded witnesses of the bug this kernel replaced.

        The previous ``floor(0.5 + exp(lgamma ...))`` formula returned values too large by
        380 at ``(57, 28)``, 5600 at ``(60, 30)`` and 1136 at ``(61, 30)``; compiled on
        this platform it already differed at ``(48, 25)``, where CPython's own ``lgamma``
        still agreed. Reverting to any logarithm-based route fails this test.
        """
        assert _bincoeff(n, k) == float(expected)

    def test_the_envelope_boundary_is_where_int64_ends(self) -> None:
        """``_EXACT_UP_TO`` is exactly where the recurrence's intermediates stop fitting."""
        int64_max = 2**63 - 1

        def worst_intermediate(n: int) -> int:
            return max(math.comb(n, k) * min(k, n - k) for k in range(n + 1))

        assert worst_intermediate(_EXACT_UP_TO) <= int64_max
        assert worst_intermediate(_EXACT_UP_TO + 1) > int64_max

    def test_the_float_return_boundary(self) -> None:
        """``_FLOAT_EXACT_UP_TO`` is exactly where ``float64`` stops holding the integer."""
        assert max(math.comb(_FLOAT_EXACT_UP_TO, k) for k in range(_FLOAT_EXACT_UP_TO + 1)) <= 2**53
        assert math.comb(_FLOAT_EXACT_UP_TO + 1, 28) > 2**53

    @pytest.mark.parametrize("n", [0, 1, 5, 12, 30, 61])
    def test_symmetry(self, n: int) -> None:
        """``C(n, k) == C(n, n - k)``, which is also the branch the recurrence picks."""
        for k in range(n + 1):
            assert _bincoeff(n, k) == _bincoeff(n, n - k)

    @pytest.mark.parametrize(("n", "k"), [(5, -1), (5, 6), (0, 1), (10, 11), (3, -7)])
    def test_out_of_range_k_is_zero(self, n: int, k: int) -> None:
        """``k`` outside ``[0, n]`` gives ``0.0``, as the elevation tables rely on."""
        assert _bincoeff(n, k) == 0.0

    @pytest.mark.parametrize("n", [0, 1, 7, 40])
    def test_edges_are_one(self, n: int) -> None:
        """``C(n, 0) == C(n, n) == 1``."""
        assert _bincoeff(n, 0) == 1.0
        assert _bincoeff(n, n) == 1.0

    def test_returns_a_float(self) -> None:
        """The return type stays ``float`` so the call sites are untouched."""
        value = _bincoeff(10, 4)
        assert isinstance(value, float)
        assert value == 210.0
