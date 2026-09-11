#!/usr/bin/env python
"""Measure the arithmetic width of every site in the B-spline degree kernels.

``design/backend_parity.md`` Rule 9 says an oracle's accumulation width is a
**per-kernel fact, not a module convention**. ``_degree_elevate_1d_core`` is sharper
than that: the width varies *within* the kernel, because Numba types each SSA version
of a variable separately and the four assignments to ``tmp1`` do not agree. This script
is the measurement ``cpp/include/pantr/bspline/degree.hpp`` is written against.

    PYTHONPATH="$(pwd)/src" python scripts/measure_bspline_degree_widths.py

Three things are printed, and the middle one is the one that would catch a wrong
transliteration.

**One, Numba's own inferred type for every arithmetic site**, read out of
``inspect_types`` rather than out of the source. That says what the compiler decided;
it does not say whether the decision is observable.

**Two, a behavioural check.** A rival model of A5.9 is run against the kernel, bit for
bit, once per width policy. The policies differ from the true one at a single site
each, so a policy that fails names the site that is load-bearing -- and the true policy
matching is only evidence because the others do not. Everything coincides at
``float64``, which is exactly Rule 9's warning, so the table is reported per dtype.

**Three, the reason ``elevate_degree`` refuses an unclamped direction in C++.** A5.9
walks segments until a run of equal knots reaches the last index of the vector, and an
unclamped vector has no such run, so the walk reads past the control array. Numba does
not bounds check in ``nopython`` mode: compiled, the call returns uninitialised memory;
interpreted, the same call raises ``IndexError``. Both halves are printed.
"""

from __future__ import annotations

import io
import os
import re
import sys
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from pantr.bspline._bspline_degree_core import _bincoeff, _degree_elevate_1d_core
from pantr.bspline._bspline_derivative import _derivative_ctrl_1d

FloatArray = npt.NDArray[np.floating[Any]]

_SITES: Final[tuple[str, ...]] = (
    "inv",
    "numer",
    "ua",
    "ub",
    "den",
    "bet",
    "gam",
    "alf",
    "tmp1",
    "tmp2",
)
"""Every variable in ``_degree_elevate_1d_core`` that holds a rounded value."""

_POLICIES: Final[tuple[str, ...]] = (
    "oracle",
    "wide-blend-product",
    "wide-ebpts-accumulator",
    "wide-alfs-division",
    "wide-knot-weights",
)
"""The true width policy, and one rival differing from it at a single site."""


def _inferred_types(kernel: Any, dtype: npt.DTypeLike) -> dict[str, str]:  # noqa: ANN401 -- see Args
    """Read Numba's inferred type for every site of the elevation kernel.

    Args:
        kernel (Any): The elevation kernel as a numba dispatcher.
            ``numba.core.registry.CPUDispatcher`` is not statically typed and is
            private, so there is no narrower annotation -- the same reason
            ``scripts/measure_bspline_tabulation_widths.py`` gives.
        dtype (npt.DTypeLike): Storage format to compile the kernel for.

    Returns:
        dict[str, str]: SSA variable name to its Numba type, for the sites of
        :data:`_SITES`.

    Raises:
        RuntimeError: If no ``inspect_types`` block matches the requested format,
            which means numba's dump format moved.
    """
    knots = np.array([0, 0, 0, 0, 0.25, 0.5, 0.5, 1, 1, 1, 1], dtype=dtype)
    ctrl = np.arange(7.0, dtype=dtype).reshape(7, 1)
    kernel(3, ctrl, knots, 2)

    buffer = io.StringIO()
    kernel.inspect_types(file=buffer)

    # One block per compiled signature, and the kernel is compiled for both formats by
    # the time this runs, so the block has to be selected rather than the first match
    # taken -- which is how an earlier version of this script reported float32 widths
    # under the float64 heading.
    wanted = f"Array({np.dtype(dtype).name}, 2,"
    found: dict[str, str] = {}
    in_block = False
    for line in buffer.getvalue().splitlines():
        if line.startswith("_degree_elevate_1d_core ("):
            in_block = wanted in line
            continue
        if not in_block:
            continue
        match = re.match(r"\s*#\s+([A-Za-z_][A-Za-z_0-9.]*)\s*=\s*.*::\s*(\S+)\s*$", line)
        if match and match.group(1).split(".")[0] in _SITES:
            found.setdefault(match.group(1), match.group(2))
    if not found:
        raise RuntimeError(f"no inspect_types block for {wanted}; the dump format moved")
    return found


def _elevate_model(  # noqa: PLR0912, PLR0915
    degree: int,
    ctrl: FloatArray,
    knots: FloatArray,
    increment: int,
    policy: str,
) -> tuple[FloatArray, FloatArray]:
    """Run A5.9 with the widths a given policy prescribes.

    A transliteration of ``_degree_elevate_1d_core`` in plain Python, with every
    rounding written out, so that one site at a time can be given the wrong width.

    Args:
        degree (int): Original degree.
        ctrl (FloatArray): Control points of shape ``(n_pts, rank)``.
        knots (FloatArray): Knot vector.
        increment (int): Degrees to add.
        policy (str): One of :data:`_POLICIES`.

    Returns:
        tuple[FloatArray, FloatArray]: Elevated control points and knot vector.
    """
    fmt = ctrl.dtype.type
    n_pts, rank = ctrl.shape
    d, t = degree, increment
    n = n_pts - 1
    ph, ph2 = d + t, (d + t) // 2
    m = n + d + 1

    bezalfs = np.zeros((d + 1, ph + 1), dtype=np.float64)
    alfs = np.zeros(d, dtype=np.float64)
    bpts = np.zeros((d + 1, rank), dtype=ctrl.dtype)
    ebpts = np.zeros((ph + 1, rank), dtype=ctrl.dtype)
    nextbpts = np.zeros((d + 1, rank), dtype=ctrl.dtype)

    bezalfs[0, 0] = 1.0
    bezalfs[d, ph] = 1.0
    for i in range(1, ph2 + 1):
        inv = 1.0 / _bincoeff(ph, i)
        for j in range(max(0, i - t), min(d, i) + 1):
            bezalfs[j, i] = inv * _bincoeff(d, j) * _bincoeff(t, i - j)
    for i in range(ph2 + 1, ph):
        for j in range(max(0, i - t), min(d, i) + 1):
            bezalfs[j, i] = bezalfs[d - j, ph - i]

    kind, r, a, b, cind = ph + 1, -1, d, d + 1, 1
    ua = knots[0]
    ik = np.zeros(len(knots) + t * len(knots), dtype=knots.dtype)
    ic = np.zeros((n_pts + t * len(knots), rank), dtype=ctrl.dtype)

    ic[0] = ctrl[0]
    ik[: ph + 1] = ua
    bpts[: d + 1] = ctrl[: d + 1]

    while b <= m:
        run_start = b
        while b < m and knots[b] == knots[b + 1]:
            b += 1
        mul = b - run_start + 1
        ub = knots[b]
        oldr, r = r, d - mul

        if oldr > 0:
            lbz = (oldr + 2) // 2
        elif oldr < 0 and a != d:
            lbz = 0
        else:
            lbz = 1
        rbz = ph - (r + 1) // 2 if r > 0 else ph

        if r > 0:
            numer = ub - ua
            for q in range(d, mul, -1):
                if policy == "wide-alfs-division":
                    alfs[q - mul - 1] = float(numer) / (float(knots[a + q]) - float(ua))
                else:
                    alfs[q - mul - 1] = numer / (knots[a + q] - ua)
            for j in range(1, r + 1):
                save, s = r - j, mul + j
                for q in range(d, s - 1, -1):
                    for ii in range(rank):
                        kept = alfs[q - s] * float(bpts[q, ii])
                        carried = (1.0 - alfs[q - s]) * float(bpts[q - 1, ii])
                        bpts[q, ii] = fmt(kept + carried)
                nextbpts[save] = bpts[d]

        for i in range(lbz, ph + 1):
            ebpts[i] = 0.0
            wide = np.zeros(rank, dtype=np.float64)
            for j in range(max(0, i - t), min(d, i) + 1):
                for ii in range(rank):
                    term = bezalfs[j, i] * float(bpts[j, ii])
                    if policy == "wide-ebpts-accumulator":
                        wide[ii] += term
                        ebpts[i, ii] = fmt(wide[ii])
                    else:
                        ebpts[i, ii] = fmt(float(ebpts[i, ii]) + term)

        if oldr > 1:
            first, last = kind - 2, kind
            if policy == "wide-knot-weights":
                den = float(ub) - float(ua)
                bet = (float(ub) - float(ik[kind - 1])) / den
            else:
                den = ub - ua
                bet = (ub - ik[kind - 1]) / den
            for tr in range(1, oldr):
                i, j = first, last
                kj = j - kind + 1
                while j - i > tr:
                    if i < cind:
                        if policy == "wide-knot-weights":
                            alf = (float(ub) - float(ik[i])) / (float(ua) - float(ik[i]))
                        else:
                            alf = (ub - ik[i]) / (ua - ik[i])
                        for ii in range(rank):
                            if policy == "wide-blend-product":
                                kept = float(alf) * float(ic[i, ii])
                            else:
                                kept = alf * ic[i, ii]
                            carried = (1.0 - float(alf)) * float(ic[i - 1, ii])
                            ic[i, ii] = fmt(float(kept) + carried)
                    if j >= lbz:
                        if j - tr <= kind - ph + oldr:
                            if policy == "wide-knot-weights":
                                weight = (float(ub) - float(ik[j - tr])) / den
                            else:
                                weight = (ub - ik[j - tr]) / den
                        else:
                            weight = bet
                        for ii in range(rank):
                            if policy == "wide-blend-product":
                                kept = float(weight) * float(ebpts[kj, ii])
                            else:
                                kept = weight * ebpts[kj, ii]
                            carried = (1.0 - float(weight)) * float(ebpts[kj + 1, ii])
                            ebpts[kj, ii] = fmt(float(kept) + carried)
                    i, j, kj = i + 1, j - 1, kj - 1
                first, last = first - 1, last + 1

        if a != d:
            for _ in range(ph - oldr):
                ik[kind] = ua
                kind += 1

        for j in range(lbz, rbz + 1):
            ic[cind] = ebpts[j]
            cind += 1

        if b < m:
            bpts[:r] = nextbpts[:r]
            for j in range(max(0, r), d + 1):
                bpts[j] = ctrl[b - d + j]
            a, b, ua = b, b + 1, ub
        else:
            ik[kind : kind + ph + 1] = ub
            break

    return ic[:cind].copy(), ik[: kind + ph + 1].copy()


def _sweep() -> list[tuple[int, FloatArray, int]]:
    """Build the case list the policies are compared over.

    Returns:
        list[tuple[int, FloatArray, int]]: ``(degree, knots, increment)`` per case,
        clamped at both ends and covering interior multiplicities 1 to ``degree + 1``.
    """
    rng = np.random.default_rng(20260911)
    cases: list[tuple[int, FloatArray, int]] = []
    for _ in range(120):
        degree = int(rng.integers(1, 7))
        interior: list[float] = []
        for value in np.sort(rng.uniform(0.05, 0.95, int(rng.integers(0, 5)))):
            interior += [round(float(value), 3)] * int(rng.integers(1, degree + 2))
        knots = np.array([0.0] * (degree + 1) + interior + [1.0] * (degree + 1))
        if len(knots) - degree - 1 < degree + 1:
            continue
        cases.append((degree, knots, int(rng.integers(1, 4))))
    return cases


def _compare_policies(dtype: npt.DTypeLike) -> dict[str, tuple[int, int]]:
    """Run every policy against the kernel and count exact reproductions.

    Args:
        dtype (npt.DTypeLike): Storage format for the control points and knots.

    Returns:
        dict[str, tuple[int, int]]: Policy to ``(matched, compared)`` array counts.
    """
    rng = np.random.default_rng(7)
    tally = {policy: [0, 0] for policy in _POLICIES}
    for degree, knots_64, increment in _sweep():
        knots = knots_64.astype(dtype)
        rows = len(knots) - degree - 1
        ctrl = rng.uniform(-1e3, 1e3, (rows, 2)).astype(dtype)
        truth, truth_knots = _degree_elevate_1d_core(degree, ctrl, knots, increment)
        for policy in _POLICIES:
            model, model_knots = _elevate_model(degree, ctrl, knots, increment, policy)
            tally[policy][1] += 1
            same = model.shape == truth.shape and np.array_equal(
                model.view(np.uint8), truth.view(np.uint8)
            )
            same = same and np.array_equal(model_knots.view(np.uint8), truth_knots.view(np.uint8))
            tally[policy][0] += int(same)
    return {policy: (count[0], count[1]) for policy, count in tally.items()}


def _derivative_widths(dtype: npt.DTypeLike) -> tuple[int, int]:
    """Count how often a `float64` derivative model differs from the oracle.

    ``_derivative_ctrl_1d`` is plain numpy, so NEP 50 decides its widths: the Python
    ``int`` degree is weak and is converted to the array's format rather than promoting
    it, and the output array is allocated at the control points' dtype.

    Args:
        dtype (npt.DTypeLike): Storage format for the control points and knots.

    Returns:
        tuple[int, int]: ``(arrays where a float64 model differs, arrays compared)``.
    """
    rng = np.random.default_rng(11)
    differed = 0
    compared = 0
    for degree, knots_64, _ in _sweep():
        knots = knots_64.astype(dtype)
        rows = len(knots) - degree - 1
        ctrl = rng.uniform(-1e3, 1e3, (rows, 2)).astype(dtype)
        truth = _derivative_ctrl_1d(knots, degree, ctrl)

        wide_knots = knots.astype(np.float64)
        wide = _derivative_ctrl_1d(wide_knots, degree, ctrl.astype(np.float64)).astype(dtype)
        compared += 1
        differed += int(not np.array_equal(wide.view(np.uint8), truth.view(np.uint8)))
    return differed, compared


def _unclamped_walk() -> None:
    """Demonstrate that A5.9 steps past its control array on an unclamped vector.

    The compiled half returns whatever the read found, so the values it prints are not
    a measurement of anything and must not be quoted: two calls on the same input are
    compared here so that the report says *that* rather than a number. The interpreted
    half is the reproducible one.
    """
    knots = np.array([-0.3, -0.2, -0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.1, 1.2, 1.3])
    ctrl = np.arange(7.0).reshape(7, 1)
    print("  knots  :", knots.tolist())
    print("  ctrl   : 7 rows, degree 3, increment 1")
    print("  the walk needs ctrl[7], and ctrl has 7 rows")
    state = "interpreted" if os.environ.get("NUMBA_DISABLE_JIT") == "1" else "compiled"
    try:
        runs = [_degree_elevate_1d_core(3, ctrl, knots, 1)[0] for _ in range(2)]
    except IndexError as error:
        print(f"  {state}: IndexError: {error}")
    else:
        agree = np.array_equal(runs[0], runs[1])
        print(f"  {state}: returned {runs[0].shape[0]} coefficients, no exception raised")
        print(f"  two calls on the same input agree bit for bit: {agree}")
        print("  the tail holds whatever the out-of-bounds read found, so it is not a figure")
    other = "without NUMBA_DISABLE_JIT" if state == "interpreted" else "with NUMBA_DISABLE_JIT=1"
    print(f"  Run again {other} to see the other half.")


def main() -> int:
    """Print the three measurements.

    Returns:
        int: 0.
    """
    compiled = hasattr(_degree_elevate_1d_core, "inspect_types")
    if not compiled:
        # `design/backend_parity.md` Rule 12: under `NUMBA_DISABLE_JIT=1` the oracle is
        # a different object. There is no inferred type to read and no width to measure
        # -- CPython's `float()` widens where numba's does not -- so the two sections
        # that are about widths are not merely skipped, they would be about something
        # else. The out-of-bounds walk is the half this configuration answers, and it
        # answers it better than the compiled one does.
        print("NUMBA_DISABLE_JIT is set, so the kernel is interpreted python.")
        print("The width sections need the compiled kernel and are not run.\n")
        print("== A5.9 on an unclamped knot vector ==")
        _unclamped_walk()
        return 0

    for dtype in (np.float32, np.float64):
        name = np.dtype(dtype).name
        print(f"== Numba's inferred type per site, at {name} ==")
        for site, inferred in sorted(_inferred_types(_degree_elevate_1d_core, dtype).items()):
            print(f"  {site:<10} {inferred}")
        print()

    print("== A5.9 width policies against the kernel, exact array agreement ==")
    for dtype in (np.float32, np.float64):
        name = np.dtype(dtype).name
        print(f"  {name}:")
        for policy, (matched, compared) in _compare_policies(dtype).items():
            print(f"    {policy:<24} {matched:>4} / {compared}")
    print()

    print("== The derivative's width: a float64 model against the oracle ==")
    for dtype in (np.float32, np.float64):
        differed, compared = _derivative_widths(dtype)
        print(f"  {np.dtype(dtype).name}: differs on {differed} of {compared} arrays")
    print()

    print("== A5.9 on an unclamped knot vector ==")
    _unclamped_walk()
    return 0


if __name__ == "__main__":
    sys.exit(main())
