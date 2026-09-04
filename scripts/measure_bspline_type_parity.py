#!/usr/bin/env python
"""Sweep the `Bspline` value type's parity claim far past the size that ships.

``tests/parity/test_bspline_type.py`` asserts that the two backends hold the same
state, over a hand-written table of eight shapes and a generated sweep of 24 draws.
That is a suite, and a suite has to stay fast. This script runs the *same* generator
over a draw large enough for the claim to mean something: a sweep checked only by its
own shipped size has not been checked.

It is **not** a test. The parity suite asserts; this reports, because what a reader
deciding whether to believe the claim wants is the size of the sweep and the worst
disagreement found, not a green tick.

What it sweeps, and why each axis is there:

- **the storage format**, because ``float32`` is the half of the matrix nobody reads
  first and the only one where a narrowing cast is visible;
- **the parametric dimension**, 1 to 3, because a tensor-product layout error needs
  more than one axis to show;
- **the degree and the interior knot multiplicities**, because a repeated interior
  knot changes the basis count and therefore the net's shape;
- **the periodic flag**, because a periodic direction's basis count is not
  ``len(knots) - degree - 1`` and the net is laid out on the smaller number;
- **the component count and the rationality flag**, because ``rank`` folds one
  against the other;
- **coefficients spanning the format's exponent range**, because a value that
  survives both formats hides a cast that a wide one would expose.

Run it as::

    PYTHONPATH="$(pwd)/src" .venv/bin/python scripts/measure_bspline_type_parity.py [draws]

``draws`` defaults to 2400, which is 100 times the shipped sweep. The exit status is
non-zero if any draw disagrees, so it is usable as a one-off gate as well as a
report.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Final, NamedTuple

import numpy as np
import numpy.typing as npt

from pantr._backend import Backend, available_backends, use_backend
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D

DEFAULT_DRAWS: Final = 2400
"""One hundred times the sweep `tests/parity/test_bspline_type.py` ships."""

SEED: Final = 20260904
"""Fixed, so the report is reproducible; the draw count is what varies."""

_HALF: Final = 0.5
"""An even split between two choices of a draw."""

_PERIODIC_SHARE: Final = 0.3
"""Share of directions drawn periodic.

Below a half on purpose: a periodic direction restricts the knot vector -- strictly
increasing, and `3 * degree + 2` knots at least -- so drawing them evenly would spend
most of the sweep on the narrower family.
"""

_MAX_REPORTED: Final = 20
"""How many disagreeing draws to print before summarizing the rest."""


class Draw(NamedTuple):
    """One randomly generated B-spline field, described independently of any backend.

    Attributes:
        knots (tuple[npt.NDArray[Any], ...]): One knot vector per direction.
        degrees (tuple[int, ...]): One degree per direction.
        periodic (tuple[bool, ...]): Whether each direction wraps.
        components (int): The length of the stored component axis.
        is_rational (bool): Whether the last stored component is a weight.
        dtype (np.dtype[Any]): The storage format.
    """

    knots: tuple[npt.NDArray[Any], ...]
    degrees: tuple[int, ...]
    periodic: tuple[bool, ...]
    components: int
    is_rational: bool
    dtype: np.dtype[Any]


def _knot_vector(
    rng: np.random.Generator, degree: int, *, periodic: bool, dtype: np.dtype[Any]
) -> npt.NDArray[Any]:
    """Draw one knot vector.

    A periodic direction gets a strictly increasing vector, which is what the wrap
    needs; a clamped one gets ``degree + 1`` repeats at each end and interior knots
    of multiplicity 1 to ``degree``, so the basis count varies with the
    multiplicities rather than only with the interval count.

    Args:
        rng (np.random.Generator): The generator.
        degree (int): The polynomial degree.
        periodic (bool): Whether the direction wraps.
        dtype (np.dtype[Any]): The storage format.

    Returns:
        npt.NDArray[Any]: A non-decreasing knot vector of at least ``2 * degree + 2``
        entries.
    """
    if periodic:
        # A periodic direction needs `3 * degree + 2` knots, not `2 * degree + 2`:
        # its basis count is `len - 2 * degree - 1` once the wrap and the regularity
        # at the domain's start are taken off a strictly increasing vector, and
        # `BsplineSpace1D` refuses a count below `degree + 1`. Measured, not guessed:
        # `2 * degree + 2` gives a count of 1 at every degree and was refused.
        length = 3 * degree + 2 + int(rng.integers(0, 6))
        steps = rng.uniform(0.5, 2.0, size=length - 1)
        return np.asarray(np.concatenate([[0.0], np.cumsum(steps)]), dtype=dtype)

    intervals = 1 + int(rng.integers(0, 4))
    breaks = np.cumsum(rng.uniform(0.5, 2.0, size=intervals + 1))
    interior: list[float] = []
    for value in breaks[1:-1]:
        interior += [float(value)] * (1 + int(rng.integers(0, max(degree, 1))))
    vector = np.concatenate(
        [
            np.full(degree + 1, breaks[0]),
            np.asarray(interior, dtype=np.float64),
            np.full(degree + 1, breaks[-1]),
        ]
    )
    return np.asarray(vector, dtype=dtype)


def _draw(rng: np.random.Generator) -> Draw:
    """Draw one field description.

    Args:
        rng (np.random.Generator): The generator.

    Returns:
        Draw: The description, which :func:`_build` turns into a field under
        whichever backend is active.
    """
    dtype = np.dtype(np.float32) if rng.random() < _HALF else np.dtype(np.float64)
    dim = 1 + int(rng.integers(0, 3))
    degrees = tuple(int(rng.integers(0, 4)) for _ in range(dim))
    periodic = tuple(bool(rng.random() < _PERIODIC_SHARE) for _ in range(dim))
    knots = tuple(
        _knot_vector(rng, degree, periodic=wraps, dtype=dtype)
        for degree, wraps in zip(degrees, periodic, strict=True)
    )
    components = 1 + int(rng.integers(0, 4))
    is_rational = bool(rng.random() < _HALF) and components > 1
    return Draw(knots, degrees, periodic, components, is_rational, dtype)


def _coefficients(rng: np.random.Generator, total: int, dtype: np.dtype[Any]) -> npt.NDArray[Any]:
    """Draw coefficients spanning the storage format's exponent range.

    Magnitudes spread over many decades, so a narrowing cast in the port shows up as
    a changed bit pattern rather than as a value that happens to survive both
    formats.

    Args:
        rng (np.random.Generator): The generator.
        total (int): How many scalars.
        dtype (np.dtype[Any]): The storage format.

    Returns:
        npt.NDArray[Any]: The coefficients.
    """
    limit = 30 if dtype == np.float32 else 100
    mantissa = rng.uniform(-1.0, 1.0, size=total)
    exponent = rng.integers(-limit, limit, size=total)
    return np.asarray(mantissa * np.float64(2.0) ** exponent, dtype=dtype)


def _build(draw: Draw, coefficients: npt.NDArray[Any]) -> Bspline:
    """Build ``draw``'s field under whichever backend is active.

    Args:
        draw (Draw): The description.
        coefficients (npt.NDArray[Any]): The flat coefficient buffer.

    Returns:
        Bspline: The field.
    """
    space = BsplineSpace(
        [
            BsplineSpace1D(vector, degree, periodic=wraps)
            for vector, degree, wraps in zip(draw.knots, draw.degrees, draw.periodic, strict=True)
        ]
    )
    return Bspline(space, coefficients, draw.is_rational)


def _disagreements(py: Bspline, cpp: Bspline) -> list[str]:
    """Every field on which the two backends disagree.

    The same seven quantities ``tests/parity/test_bspline_type.py``'s ``FIELDS``
    compares, under the same claims: bitwise for the coefficients, exact for
    everything else.

    Args:
        py (Bspline): What the Python backend built.
        cpp (Bspline): What the C++ backend built.

    Returns:
        list[str]: One entry per disagreeing field, empty when they agree.
    """
    found = []
    if py.control_points.tobytes() != cpp.control_points.tobytes():
        found.append("control_points")
    if py.control_points.shape != cpp.control_points.shape:
        found.append("control_points.shape")
    for name in ("is_rational", "dim", "degree", "rank"):
        if getattr(py, name) != getattr(cpp, name):
            found.append(name)
    if np.dtype(py.dtype) != np.dtype(cpp.dtype):
        found.append("dtype")
    if py.space.num_basis != cpp.space.num_basis:
        found.append("space.num_basis")
    return found


def _closed_form_rank(draw: Draw, num_basis: tuple[int, ...], total: int) -> int:
    """``draw``'s rank from the buffer size and the basis counts, in exact integers.

    The independent half: nothing here reads a constructed field.

    Args:
        draw (Draw): The description.
        num_basis (tuple[int, ...]): The per-direction basis counts.
        total (int): The number of scalars in the buffer.

    Returns:
        int: The rank a well-formed field would report.
    """
    return total // math.prod(num_basis) - (1 if draw.is_rational else 0)


def main(argv: list[str]) -> int:
    """Run the sweep and report.

    Args:
        argv (list[str]): Command-line arguments; ``argv[1]`` is the draw count.

    Returns:
        int: 0 when every draw agreed, 1 otherwise.
    """
    if Backend.CPP not in available_backends():
        print("the pantr._pantr_cpp extension is not built; nothing to compare against")
        return 1

    draws = int(argv[1]) if len(argv) > 1 else DEFAULT_DRAWS
    rng = np.random.default_rng(SEED)
    failures: list[str] = []
    dtype_counts: dict[str, int] = {"float32": 0, "float64": 0}
    dim_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
    periodic_draws = 0
    rational_draws = 0
    largest_net = 0

    for index in range(draws):
        draw = _draw(rng)
        # The basis counts come from the space, but only to size the buffer; the rank
        # is then checked against the closed form below, which does not.
        with use_backend(Backend.PYTHON):
            probe_space = BsplineSpace(
                [
                    BsplineSpace1D(vector, degree, periodic=wraps)
                    for vector, degree, wraps in zip(
                        draw.knots, draw.degrees, draw.periodic, strict=True
                    )
                ]
            )
            num_basis = probe_space.num_basis
        total = probe_space.num_total_basis * draw.components
        coefficients = _coefficients(rng, total, draw.dtype)

        with use_backend(Backend.PYTHON):
            py = _build(draw, coefficients)
        with use_backend(Backend.CPP):
            cpp = _build(draw, coefficients)

        found = _disagreements(py, cpp)
        expected_rank = _closed_form_rank(draw, num_basis, total)
        if cpp.rank != expected_rank:
            found.append(f"rank against its closed form ({cpp.rank} != {expected_rank})")
        if tuple(cpp.control_points.shape) != (*num_basis, draw.components):
            found.append("the stored shape against the basis counts")
        if found:
            failures.append(f"draw {index} ({draw.dtype.name}, dim {len(draw.degrees)}): {found}")

        dtype_counts[draw.dtype.name] += 1
        dim_counts[len(draw.degrees)] += 1
        periodic_draws += int(any(draw.periodic))
        rational_draws += int(draw.is_rational)
        largest_net = max(largest_net, total)

    print(f"draws                : {draws}")
    print(f"  by dtype           : {dtype_counts}")
    print(f"  by parametric dim  : {dim_counts}")
    print(f"  with a periodic dir: {periodic_draws}")
    print(f"  rational           : {rational_draws}")
    print(f"  largest net        : {largest_net} scalars")
    print(f"disagreements        : {len(failures)}")
    for line in failures[:_MAX_REPORTED]:
        print(f"  {line}")
    if len(failures) > _MAX_REPORTED:
        print(f"  ... and {len(failures) - _MAX_REPORTED} more")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
