"""Tests for tolerance utilities.

The presets are **dimensionless relative** tolerances, ``K * eps(dtype)`` with a
safety factor ``K`` that is the same in every precision. The tests below pin that
shape rather than a table of twelve numbers: the constancy of ``K`` across dtypes,
the ordering of the three tiers, ``K`` being an exact power of two, and every
preset being at least one epsilon (the defect the previous table had -- its strict
float16 and float32 entries were *below* one ulp at 1.0, so they asked for bitwise
equality). The two dtypes the library actually uses are additionally pinned to
their literal values, so a silent drift in ``K`` is caught.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pantr.tolerance import (
    get_conservative,
    get_default,
    get_info,
    get_machine_epsilon,
    get_strict,
)

DTYPES: tuple[Any, ...] = (np.float16, np.float32, np.float64, np.longdouble)
"""Every floating dtype the tolerance module accepts."""

PRESETS: tuple[tuple[str, Any, float], ...] = (
    ("strict", get_strict, 4.0),
    ("default", get_default, 64.0),
    ("conservative", get_conservative, 4096.0),
)
"""Each preset with the number of machine epsilons its docstring claims."""

STRICT_TOL_F32: float = 4.76837158203125e-07
"""``4 * eps(float32)``, written out so a change to the factor is visible here."""

STRICT_TOL_F64: float = 8.881784197001252e-16
"""``4 * eps(float64)``."""

DEFAULT_TOL_F32: float = 7.62939453125e-06
"""``64 * eps(float32)``."""

DEFAULT_TOL_F64: float = 1.4210854715202004e-14
"""``64 * eps(float64)``."""

CONSERVATIVE_TOL_F32: float = 0.00048828125
"""``4096 * eps(float32)``."""

CONSERVATIVE_TOL_F64: float = 9.094947017729282e-13
"""``4096 * eps(float64)``."""


class TestTolerance:
    """Test suite for tolerance utilities."""

    @pytest.mark.parametrize(
        ("getter", "dtype", "expected"),
        [
            (get_strict, np.float32, STRICT_TOL_F32),
            (get_strict, "float64", STRICT_TOL_F64),
            (get_default, np.float32, DEFAULT_TOL_F32),
            (get_default, "float64", DEFAULT_TOL_F64),
            (get_conservative, np.float32, CONSERVATIVE_TOL_F32),
            (get_conservative, "float64", CONSERVATIVE_TOL_F64),
        ],
    )
    def test_preset_values_for_the_supported_dtypes(
        self, getter: Any, dtype: Any, expected: float
    ) -> None:
        """The two dtypes the library uses resolve to their pinned literal values."""
        assert getter(dtype) == expected

    @pytest.mark.parametrize(("name", "getter", "eps_factor"), PRESETS)
    def test_preset_is_a_constant_number_of_epsilons(
        self, name: str, getter: Any, eps_factor: float
    ) -> None:
        """Every dtype gets the *same* safety factor -- the property the old table lacked.

        The shipped table was 0.10/0.84/4.50 epsilons of strict for float16/float32/
        float64, so "strict" meant something different in each precision and two of the
        three were below one rounding.
        """
        for dtype in DTYPES:
            eps = float(np.finfo(dtype).eps)
            assert getter(dtype) == eps_factor * eps, (
                f"{name}({np.dtype(dtype).name}) is "
                f"{getter(dtype) / eps:g} epsilons, not {eps_factor:g}"
            )

    @pytest.mark.parametrize(("name", "getter", "eps_factor"), PRESETS)
    def test_safety_factor_is_an_exact_power_of_two(
        self, name: str, getter: Any, eps_factor: float
    ) -> None:
        """``K`` is a power of two, so it is exact in every format and in C++."""
        assert eps_factor == 2.0 ** round(np.log2(eps_factor)), name
        for dtype in DTYPES:
            ratio = getter(dtype) / float(np.finfo(dtype).eps)
            assert ratio == eps_factor

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_every_preset_admits_at_least_one_rounding(self, dtype: Any) -> None:
        """No preset may fall below one ulp at 1.0, which would demand bitwise equality."""
        eps = float(np.finfo(dtype).eps)
        for name, getter, _ in PRESETS:
            assert getter(dtype) >= eps, (
                f"{name}({np.dtype(dtype).name}) = {getter(dtype):g} is below one "
                f"epsilon ({eps:g}): it asks for bitwise equality, not a tolerance"
            )

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_the_three_tiers_are_ordered(self, dtype: Any) -> None:
        """Strict is tighter than default, which is tighter than conservative."""
        assert get_strict(dtype) < get_default(dtype) < get_conservative(dtype)

    def test_longdouble_needs_no_platform_branch(self) -> None:
        """``longdouble`` follows the same formula whether or not it aliases float64.

        The previous table carried a separate longdouble column and an import-time
        branch on ``np.dtype(np.longdouble) == np.dtype(np.float64)``. Reading the
        epsilon from the platform removes both: where longdouble *is* float64 the two
        answers coincide because the two epsilons do.
        """
        eps = float(np.finfo(np.longdouble).eps)
        assert get_strict(np.longdouble) == 4.0 * eps
        assert get_strict(np.dtype(np.longdouble)) == 4.0 * eps
        assert get_strict("longdouble") == 4.0 * eps
        if np.dtype(np.longdouble) == np.dtype(np.float64):
            assert get_strict(np.longdouble) == get_strict(np.float64)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_get_machine_epsilon(self, dtype: Any) -> None:
        """Test get_machine_epsilon against np.finfo."""
        assert get_machine_epsilon(dtype) == np.finfo(dtype).eps

    def test_invalid_dtype_raises_error(self) -> None:
        """Test that an unsupported dtype raises a ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_default(np.int32)
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_strict("int64")
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_conservative(np.complex64)
        with pytest.raises(ValueError, match="Unsupported dtype"):
            get_machine_epsilon(np.uint8)

    def test_get_tolerance_info(self) -> None:
        """Test get_info returns the expected structure and values."""
        dtype = np.float64
        info = get_info(dtype)

        assert info["dtype"] == dtype
        assert info["machine_epsilon"] == np.finfo(dtype).eps
        assert info["default_tolerance"] == DEFAULT_TOL_F64
        assert info["strict_tolerance"] == STRICT_TOL_F64
        assert info["conservative_tolerance"] == CONSERVATIVE_TOL_F64
        assert info["precision_bits"] == np.finfo(dtype).precision
        assert info["precision_decimals"] == np.finfo(dtype).precision
        assert info["resolution"] == np.finfo(dtype).resolution
        assert info["max_value"] == np.finfo(dtype).max
        assert info["min_value"] == np.finfo(dtype).tiny

    def test_get_tolerance_info_string_dtype(self) -> None:
        """Test get_info with a string dtype preserves the original representation."""
        dtype_str = "float32"
        finfo = np.finfo(dtype_str)
        info = get_info(dtype_str)

        assert info["dtype"] == dtype_str
        assert info["machine_epsilon"] == finfo.eps
        assert info["default_tolerance"] == DEFAULT_TOL_F32

    def test_tolerance_info_keys(self) -> None:
        """Test that get_info returns exactly the documented keys."""
        info = get_info(np.float32)
        expected_keys = {
            "dtype",
            "machine_epsilon",
            "default_tolerance",
            "strict_tolerance",
            "conservative_tolerance",
            "precision_bits",
            "precision_decimals",
            "resolution",
            "max_value",
            "min_value",
        }
        assert set(info.keys()) == expected_keys

    def test_half_precision_conservative_tier_saturates(self) -> None:
        """float16 cannot absorb 4096 roundings, and the module says so rather than clipping.

        11 significant bits give ``eps = 9.77e-4``, so the conservative tier is 4.0 -- a
        relative tolerance above one, which accepts everything. Asserted here so the
        documented edge cannot drift into a silent clamp.
        """
        assert get_conservative(np.float16) == 4.0
        assert get_default(np.float16) < 1.0
