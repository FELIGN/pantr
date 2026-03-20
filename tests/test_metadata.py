"""Smoke tests for package metadata.

Validates public attributes exposed via the package API.
"""

from __future__ import annotations

import importlib
from typing import Final

import pantr


def test_package_all_exports() -> None:
    """Ensure all expected symbols are exported."""
    # Check that metadata is in __all__
    expected_metadata: Final[set[str]] = {"__version__", "__license__", "__author__"}
    assert expected_metadata.issubset(set(pantr.__all__))

    # Expected public API symbols (functions/classes that don't start with _)
    expected_public_api: Final[set[str]] = {
        # Bezier
        "Bezier",
        # Basis functions
        "LagrangeVariant",
        "tabulate_bernstein",
        "tabulate_bernstein_1d",
        "tabulate_cardinal_bspline",
        "tabulate_cardinal_bspline_1d",
        "tabulate_lagrange",
        "tabulate_lagrange_1d",
        "tabulate_legendre_1d",
        # B-spline space
        "BsplineSpace",
        "BsplineSpace1D",
        "create_cardinal",
        "create_uniform_open",
        "create_uniform_periodic",
        # Change of basis
        "compute_bernstein_to_lagrange_1d",
        "compute_bernstein_to_cardinal_1d",
        "compute_cardinal_to_bernstein_1d",
        "compute_lagrange_to_bernstein_1d",
        # Quadrature
        "PointsLattice",
        "create_lagrange_points_lattice",
        "get_chebyshev_gauss_1st_kind_1d",
        "get_chebyshev_gauss_2nd_kind_1d",
        "get_gauss_legendre_1d",
        "get_gauss_lobatto_legendre_1d",
        "get_trapezoidal_1d",
        # Tolerance
        "ToleranceInfo",
        "get_conservative",
        "get_default",
        "get_machine_epsilon",
        "get_strict",
        "get_info",
    }

    # All expected public API should be in __all__
    assert expected_public_api.issubset(set(pantr.__all__))

    # Check that no private symbols (starting with _) are in __all__
    private_in_all = {name for name in pantr.__all__ if name.startswith("_")}
    # Only allow metadata (__version__, __author__, __license__) to start with __
    assert private_in_all.issubset(expected_metadata)

    # Verify __all__ contains exactly the expected items
    expected_all = expected_metadata | expected_public_api
    assert set(pantr.__all__) == expected_all


def test_package_metadata_values() -> None:
    """Validate the package metadata constants."""
    assert pantr.__version__ == "0.1.0"
    assert pantr.__license__ == "MIT"
    assert pantr.__author__ == "Pablo Antolin <pablo.antolin@epfl.ch>"


def test_metadata_import_stability() -> None:
    """Verify metadata survives module reloads."""
    module = importlib.reload(pantr)
    assert module.__version__ == "0.1.0"
