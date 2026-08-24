#!/usr/bin/env python
"""Reproduce the four measurements behind the decision not to port Bézier interpolation.

``design/bezier_interpolation_port.md`` rules that ``_bezier_interpolate.py`` stays in
Python, and rests that ruling on numbers. This is where they come from.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        PYTHONPATH="$(pwd)/src" .venv/bin/python \
        scripts/measure_bezier_interpolation_port.py

**Pin the threads.** On a many-core box a threaded LAPACK SVD of a matrix that fits in
L1 costs orders of magnitude more than the same call on one thread, and varies wildly
between batches. Section 5 measures that directly; every other section is meaningless
without the pin, which is why the script refuses to run unpinned unless told to.

The four measurements
---------------------

1. **What the SVD costs**, as a share of a call, and how many times a call takes one.
2. **What memoizing it buys**, and that the results are unchanged bit for bit.
3. **How much of a warm call a port could reach at all**, once the factorization is no
   longer being rebuilt.
4. **Where the truncation threshold actually bites**: per order and dtype, the rank
   LAPACK chooses and how close the nearest singular value comes to the cut.
5. **Whether Eigen would choose the same rank.** This half needs a C++ compiler, so the
   script writes the matrices and a self-contained comparison program, then prints the
   two commands that build and run it. Pass ``--emit-cpp DIR`` to produce them.

6 is the threading artifact, reported for the record rather than as an input.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import pathlib
import sys
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt

from pantr._interpolation_utils import resolve_svd_tolerance
from pantr.bezier import fit_bezier, interpolate_bezier
from pantr.bezier._bezier_interpolate import (
    _build_bernstein_pinv,
    _compute_bernstein_pinv,
)
from pantr.bezier._bezier_utils import _tabulate_bernstein_1d_fast
from pantr.quad import PointsLattice, get_modified_chebyshev_nodes_1d

ORDERS: tuple[int, ...] = (10, 20, 25, 30, 36, 39)
"""Orders sampled throughout. Spans below, across and above where float32 starts truncating."""

_MEMO_ATTR = "_bernstein_pinv_cached"
"""Name of the cache that :func:`_memoization_bypassed` swaps out, as an attribute."""

_THREAD_VARS: tuple[str, ...] = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
"""Variables that must read ``1`` for a timing on this script to mean anything."""


def _best_of(fn: Callable[[], Any], reps: int, batches: int = 5) -> float:
    """Time a callable, returning the fastest batch mean in milliseconds.

    The minimum over batches is used rather than the mean because the contaminant on a
    shared machine is always additive.

    Args:
        fn (Callable[[], Any]): The callable to time. Called for effect.
        reps (int): Repetitions per batch.
        batches (int): Number of batches. Defaults to 5.

    Returns:
        float: Milliseconds per call, fastest batch.
    """
    for _ in range(min(reps, 20)):
        fn()
    best = float("inf")
    for _ in range(batches):
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, (time.perf_counter() - start) / reps * 1e3)
    return best


def _count_svd_calls(fn: Callable[[], Any]) -> tuple[int, float, float]:
    """Run a callable with ``np.linalg.svd`` instrumented.

    Args:
        fn (Callable[[], Any]): The callable to run once.

    Returns:
        tuple[int, float, float]: Number of SVD calls, milliseconds spent inside them,
        and total milliseconds for the callable.
    """
    calls = 0
    inside = 0.0
    real = np.linalg.svd

    def counting(
        matrix: npt.NDArray[np.floating[Any]], *, full_matrices: bool = True
    ) -> tuple[
        npt.NDArray[np.floating[Any]],
        npt.NDArray[np.floating[Any]],
        npt.NDArray[np.floating[Any]],
    ]:
        nonlocal calls, inside
        start = time.perf_counter()
        result = real(matrix, full_matrices=full_matrices)
        inside += time.perf_counter() - start
        calls += 1
        return result

    np.linalg.svd = counting  # type: ignore[assignment]
    try:
        start = time.perf_counter()
        fn()
        total = time.perf_counter() - start
    finally:
        np.linalg.svd = real
    return calls, inside * 1e3, total * 1e3


def _cubic(lattice: PointsLattice) -> npt.NDArray[np.floating[Any]]:
    """Sample a univariate cubic on a lattice.

    Args:
        lattice (PointsLattice): The sampling lattice passed by ``interpolate_bezier``.

    Returns:
        npt.NDArray[np.floating[Any]]: Values at the lattice points.
    """
    return np.asarray(lattice.get_all_points()[:, 0] ** 3)


def _biquadratic(lattice: PointsLattice) -> npt.NDArray[np.floating[Any]]:
    """Sample a bivariate product on a lattice.

    Args:
        lattice (PointsLattice): The sampling lattice passed by ``interpolate_bezier``.

    Returns:
        npt.NDArray[np.floating[Any]]: Values at the lattice points.
    """
    points = lattice.get_all_points()
    return np.asarray((points[:, 0] * points[:, 1]) ** 2)


def _cases() -> list[tuple[str, Callable[[], Any]]]:
    """Build the call shapes both measurements use.

    Returns:
        list[tuple[str, Callable[[], Any]]]: Labelled zero-argument calls.
    """
    rng = np.random.default_rng(0)
    scattered_pts = rng.random((200, 2))
    scattered_vals = scattered_pts[:, 0] ** 2 + scattered_pts[:, 1]
    return [
        ("1D order 10", lambda: interpolate_bezier(_cubic, 10)),
        ("1D order 30", lambda: interpolate_bezier(_cubic, 30)),
        ("1D order 39", lambda: interpolate_bezier(_cubic, 39)),
        ("2D 30x30", lambda: interpolate_bezier(_biquadratic, [30, 30])),
        ("1D lsq 40/11", lambda: interpolate_bezier(_cubic, 40, degree=10)),
        ("scattered 200", lambda: fit_bezier(scattered_vals, nodes=scattered_pts, degree=[5, 5])),
    ]


def measure_svd_share() -> None:
    """Report how much of a call the SVD is, and how many the call takes."""
    print("\n1. What the SVD costs, per call, with the memoization bypassed")
    print("   " + "-" * 68)
    print(f"   {'case':14s} {'svd calls':>9s} {'svd ms':>8s} {'total ms':>9s} {'share':>7s}")
    for label, fn in _cases():
        fn()  # let Numba compile before anything is timed
        with _memoization_bypassed():
            shares = [_count_svd_calls(fn) for _ in range(9)]
        shares.sort(key=lambda row: row[2])
        calls, inside, total = shares[len(shares) // 2]
        print(f"   {label:14s} {calls:9d} {inside:8.3f} {total:9.3f} {100 * inside / total:6.1f}%")
    print("   Every tensor-product call reaches _build_bernstein_pinv, one factorization per")
    print("   direction, and this is what it costs when it rebuilds them every time.")


def measure_memoization() -> None:
    """Report the speedup from memoizing, and that it changes no result."""
    print("\n2. What memoizing the pseudo-inverse buys")
    print("   " + "-" * 68)
    print(f"   {'case':14s} {'no memo ms':>11s} {'memoized ms':>12s} {'speedup':>8s}")

    for label, fn in _cases():
        fn()
        memoized = _best_of(fn, 200)
        with _memoization_bypassed():
            plain = _best_of(fn, 200)
        print(f"   {label:14s} {plain:11.3f} {memoized:12.3f} {plain / memoized:7.2f}x")
    print("   The scattered case does not move: its Vandermonde is built from the caller's")
    print("   own points by _fit_from_scattered, which is a different site and not memoized.")

    with _memoization_bypassed():
        plain_result = interpolate_bezier(_cubic, 30).control_points.copy()
    memo_result = interpolate_bezier(_cubic, 30).control_points
    print(
        f"   Memoized and unmemoized results identical bit for bit: "
        f"{np.array_equal(plain_result, memo_result)}"
    )

    nodes = get_modified_chebyshev_nodes_1d(39, np.float64)
    matrix = _build_bernstein_pinv(nodes)
    copy_cost = _best_of(matrix.copy, 2000)
    key_cost = _best_of(nodes.tobytes, 2000)
    factorize = _best_of(lambda: _compute_bernstein_pinv(nodes, None, None), 200)
    print(
        f"   Order 39: the copy and the key cost {copy_cost * 1e3:.2f} us and "
        f"{key_cost * 1e3:.2f} us against {factorize * 1e3:.2f} us to rebuild"
    )


@contextlib.contextmanager
def _memoization_bypassed() -> Iterator[None]:
    """Route the pseudo-inverse around its cache for the duration of the block.

    Yields:
        None: The block runs with every call recomputing the factorization.
    """
    module = sys.modules[_build_bernstein_pinv.__module__]
    original = getattr(module, _MEMO_ATTR)
    setattr(module, _MEMO_ATTR, _keyless_passthrough)
    try:
        yield
    finally:
        setattr(module, _MEMO_ATTR, original)


def _keyless_passthrough(
    node_bytes: bytes,
    dtype: np.dtype[np.floating[Any]],
    tol: float | None,
    degree: int | None,
) -> npt.NDArray[np.floating[Any]]:
    """Stand in for the cache, recomputing on every call.

    Args:
        node_bytes (bytes): The node array's buffer, as the cache receives it.
        dtype (np.dtype[np.floating[Any]]): The node array's dtype.
        tol (float | None): SVD truncation tolerance.
        degree (int | None): Target Bernstein degree.

    Returns:
        npt.NDArray[np.floating[Any]]: The freshly computed pseudo-inverse.
    """
    return _compute_bernstein_pinv(np.frombuffer(node_bytes, dtype=dtype).copy(), tol, degree)


def measure_portable_fraction() -> None:
    """Report how much of a warm call a C++ port could even reach."""
    print("\n3. How much of a warm call is portable at all")
    print("   " + "-" * 68)
    print(
        f"   {'case':14s} {'warm ms':>8s} {'apply ms':>9s} "
        f"{'callable ms':>12s} {'apply share':>12s}"
    )
    for order in (10, 30, 39):
        nodes = get_modified_chebyshev_nodes_1d(order, np.float64)
        lattice = PointsLattice([nodes])
        values = np.asarray(_cubic(lattice))
        pinv = _build_bernstein_pinv(nodes)

        def whole_call(count: int = order) -> object:
            return interpolate_bezier(_cubic, count)

        def apply_once(
            matrix: npt.NDArray[np.floating[Any]] = pinv,
            data: npt.NDArray[np.floating[Any]] = values,
        ) -> object:
            return np.tensordot(matrix, data, axes=([1], [0]))

        def sample_once(grid: PointsLattice = lattice) -> object:
            return _cubic(grid)

        warm = _best_of(whole_call, 300)
        apply_cost = _best_of(apply_once, 2000)
        callable_cost = _best_of(sample_once, 2000)
        print(
            f"   1D order {order:<5d} {warm:8.3f} {apply_cost:9.4f} {callable_cost:12.4f} "
            f"{100 * apply_cost / warm:11.1f}%"
        )
    print("   Everything but the apply is Python-level: validation, lattice construction,")
    print("   component splitting, and the caller's own function. interpolate_bezier is")
    print("   handed a Python callable, so no port removes that column.")


def vandermonde(n: int, dtype: npt.DTypeLike) -> npt.NDArray[np.floating[Any]]:
    """Build the Bernstein Vandermonde this module factorizes.

    Args:
        n (int): Number of nodes and coefficients.
        dtype (npt.DTypeLike): Floating dtype.

    Returns:
        npt.NDArray[np.floating[Any]]: The ``n``-by-``n`` matrix.
    """
    nodes = get_modified_chebyshev_nodes_1d(max(n, 2), dtype)[:n]
    return _tabulate_bernstein_1d_fast(n - 1, nodes, dtype)


def measure_truncation() -> dict[str, dict[str, Any]]:
    """Report where the rank truncation actually fires, and by how little.

    Returns:
        dict[str, dict[str, Any]]: Per case, the LAPACK singular values, the resolved
        tolerance and the timing, keyed ``"<n>_<dtype>"``.
    """
    print("\n4. Where the truncation threshold bites, per order and dtype")
    print("   " + "-" * 68)
    print(f"   {'case':16s} {'kept':>7s} {'nearest ratio to the cut':>26s} {'ms':>7s}")
    out: dict[str, dict[str, Any]] = {}
    for dtype in (np.float64, np.float32):
        for n in ORDERS:
            matrix = vandermonde(n, dtype)
            sigma = np.linalg.svd(matrix, full_matrices=False)[1]
            tol = resolve_svd_tolerance(np.dtype(dtype), None)
            ratios = sigma / sigma[0] / tol
            kept = int((ratios >= 1.0).sum())
            nearest = float(ratios[int(np.argmin(np.abs(np.log10(ratios))))])

            def factorize(target: npt.NDArray[np.floating[Any]] = matrix) -> object:
                return np.linalg.svd(target, full_matrices=False)

            ms = _best_of(factorize, 200)
            key = f"{n}_{np.dtype(dtype).name}"
            out[key] = {"sigma": [float(x) for x in sigma], "tol": tol, "ms": ms}
            print(f"   {key:16s} {kept:3d}/{n:<3d} {nearest:26.6f} {ms:7.4f}")
    print("   A ratio near 1 is a singular value sitting on the cut: that is where two")
    print("   implementations could disagree about the rank, and the verdict would jump.")
    return out


def emit_cpp(target: pathlib.Path, reference: dict[str, dict[str, Any]]) -> None:
    """Write the matrices and an Eigen comparison program, and print how to run it.

    Args:
        target (pathlib.Path): Directory to write into. Created if absent.
        reference (dict[str, dict[str, Any]]): The LAPACK result from
            :func:`measure_truncation`, written alongside for comparison.
    """
    target.mkdir(parents=True, exist_ok=True)
    for key in reference:
        n_text, dtype_name = key.split("_")
        matrix = vandermonde(int(n_text), np.dtype(dtype_name))
        np.savetxt(target / f"V_{key}.txt", np.asarray(matrix, dtype=np.float64), fmt="%.17g")
    with (target / "lapack_sigma.txt").open("w") as handle:
        for key, record in reference.items():
            handle.write(f"{key} {record['tol']:.17g} ")
            handle.write(" ".join(f"{x:.17g}" for x in record["sigma"]) + "\n")
    (target / "compare_eigen.cpp").write_text(_CPP_SOURCE)
    print(f"\n   Wrote the matrices and comparison program to {target}")
    print("   Build and run it with:")
    print(
        f"       g++ -O3 -DNDEBUG -ffp-contract=on -w "
        f"-I build/gcc/_deps/eigen-src -o {target}/compare_eigen {target}/compare_eigen.cpp"
    )
    print(f"       {target}/compare_eigen {target}")
    print("   It prints one line per case and flags any where the two ranks differ.")


_CPP_SOURCE = r"""// Compare Eigen's BDCSVD against the LAPACK singular values dumped beside it.
// Written by scripts/measure_bezier_interpolation_port.py; see
// design/bezier_interpolation_port.md for what the comparison is for.
#include <Eigen/Dense>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::vector<double>> load_matrix(const std::string& path) {
    std::ifstream file(path);
    std::string line;
    std::vector<std::vector<double>> rows;
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        std::vector<double> row;
        double value = 0.0;
        while (stream >> value) row.push_back(value);
        if (!row.empty()) rows.push_back(row);
    }
    return rows;
}

template <class Scalar>
std::vector<double> eigen_singular_values(const std::vector<std::vector<double>>& rows) {
    using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    Mat matrix(rows.size(), rows[0].size());
    for (std::size_t i = 0; i < rows.size(); ++i)
        for (std::size_t j = 0; j < rows[0].size(); ++j)
            matrix(i, j) = static_cast<Scalar>(rows[i][j]);
    Eigen::BDCSVD<Mat> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    std::vector<double> out;
    for (int i = 0; i < svd.singularValues().size(); ++i)
        out.push_back(static_cast<double>(svd.singularValues()(i)));
    return out;
}

int rank_above(const std::vector<double>& sigma, double tol) {
    int kept = 0;
    for (double value : sigma)
        if (value >= tol * sigma[0]) ++kept;
    return kept;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    std::ifstream reference(dir + "/lapack_sigma.txt");
    std::string line;
    int disagreements = 0;
    while (std::getline(reference, line)) {
        std::istringstream stream(line);
        std::string key;
        double tol = 0.0;
        stream >> key >> tol;
        std::vector<double> lapack;
        double value = 0.0;
        while (stream >> value) lapack.push_back(value);

        const auto rows = load_matrix(dir + "/V_" + key + ".txt");
        const bool single = key.find("float32") != std::string::npos;
        const auto eigen = single ? eigen_singular_values<float>(rows)
                                  : eigen_singular_values<double>(rows);

        const int rank_lapack = rank_above(lapack, tol);
        const int rank_eigen = rank_above(eigen, tol);
        double worst = 0.0;
        for (std::size_t i = 0; i < lapack.size(); ++i)
            worst = std::max(worst, std::abs(eigen[i] - lapack[i]) / lapack[0]);
        const bool differs = rank_lapack != rank_eigen;
        disagreements += differs;
        std::printf("%-16s rank lapack=%3d eigen=%3d  max |dsigma|/sigma0=%.3e%s\n",
                    key.c_str(), rank_lapack, rank_eigen, worst,
                    differs ? "   <== RANKS DIFFER" : "");
    }
    std::printf("\n%d case(s) where the two implementations chose a different rank.\n",
                disagreements);
    return 0;
}
"""


def measure_threading_artifact() -> None:
    """Report how much a threaded LAPACK SVD of a tiny matrix costs, versus one thread."""
    print("\n6. The threading artifact, for the record")
    print("   " + "-" * 68)
    pinned = all(os.environ.get(name) == "1" for name in _THREAD_VARS)
    matrix = vandermonde(36, np.float64)
    spread = [
        _best_of(lambda: np.linalg.svd(matrix, full_matrices=False), 100, batches=1)
        for _ in range(5)
    ]
    state = "pinned to one thread" if pinned else f"unpinned ({_thread_setting()})"
    print(f"   Order 36, float64, {state}: {min(spread):.4f} to {max(spread):.4f} ms")
    if pinned:
        print("   Re-run without the pin to see the same call cost orders of magnitude more,")
        print("   and vary between batches. Twenty threads over a matrix that fits in L1 is")
        print("   pure synchronization, and no timing taken that way means anything.")


def _thread_setting() -> str:
    """Describe the thread-count environment in one line.

    Returns:
        str: The relevant variables and their values, or a note that none are set.
    """
    parts = [f"{name}={os.environ[name]}" for name in _THREAD_VARS if name in os.environ]
    return ", ".join(parts) if parts else "no thread variables set"


def main() -> int:
    """Run the measurements.

    Returns:
        int: Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-cpp",
        type=pathlib.Path,
        metavar="DIR",
        help="write the matrices and the Eigen comparison program into DIR",
    )
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="run the timings even though the thread counts are not pinned to 1",
    )
    args = parser.parse_args()

    pinned = all(os.environ.get(name) == "1" for name in _THREAD_VARS)
    if not pinned and not args.allow_unpinned:
        print(f"Threads are not pinned ({_thread_setting()}).", file=sys.stderr)
        print("Timings taken this way are noise; see section 5. Re-run with", file=sys.stderr)
        print(f"  {' '.join(f'{name}=1' for name in _THREAD_VARS)} ...", file=sys.stderr)
        print("or pass --allow-unpinned to measure the artifact deliberately.", file=sys.stderr)
        return 1

    print("Bézier interpolation port: the measurements behind design/bezier_interpolation_port.md")
    print(f"numpy {np.__version__}, {_thread_setting()}")
    measure_svd_share()
    measure_memoization()
    measure_portable_fraction()
    reference = measure_truncation()
    print("\n5. Whether Eigen would choose the same rank")
    print("   " + "-" * 68)
    if args.emit_cpp is None:
        print("   Needs a compiler. Re-run with --emit-cpp DIR to write the matrices and the")
        print("   comparison program, along with the two commands that build and run it.")
    else:
        emit_cpp(args.emit_cpp, reference)
    measure_threading_artifact()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
