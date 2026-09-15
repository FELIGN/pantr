#!/usr/bin/env python
"""Time ``Bezier.evaluate`` end to end with the contraction specialised on its block size.

This is the measurement issue #481 asks for first (its ``AC1``), and it gates the
rest of that ticket. ``scripts/measure_bezier_degree_templating.py`` timed
``contract_leading_axis`` in isolation and found that fixing its *trailing block
size* at compile time pays at small blocks. ``design/simd.md`` open question 2
records that and names what it does not settle: whether the win survives at the
level of ``evaluate``, where the Python entry point, the allocations, the Bernstein
tabulation and the earlier directions' contractions all share the call. This script
answers that and implements nothing::

    PYTHONPATH="$(pwd)/src" python scripts/measure_bezier_cp_size_dispatch.py --cpu <n>

The script sets ``PANTR_BACKEND=cpp`` for the timed processes itself. Where an
editable install's finder shadows ``src/``, put a shim that removes it on
``PYTHONPATH`` too: the script refuses to time a ``pantr`` from anywhere but this
checkout.

What is built, and where
------------------------

Two copies of the extension, each from its own ``git archive`` export of **one
commit SHA** (resolved once, default ``HEAD``), so the only difference between them
is the patch:

``shipped``
    The tree as committed.
``variant``
    The same tree with ``contract_leading_axis`` in its copy of
    ``cpp/include/pantr/bezier/evaluate.hpp`` rewritten to switch on ``stride``:
    each of :data:`_SPECIALISED_STRIDES` goes to a template whose block size is a
    compile-time constant, every other width falls through to the shipped body.
    The specialised body is **derived from the shipped one** by fixing ``stride``,
    not written out here, so it cannot go stale against the header; the patched
    function is printed in full. Nothing is written to this checkout's
    ``cpp/include``.

The dispatch is a ``switch`` inside the kernel, taken once per call, which is the
shape the ticket's specification would ship. Both builds use the project's own
CMake, so its numerical flags (``-O3``, ``-ffp-contract=on``, no ``-ffast-math``,
no ``-march``) hold for both by construction.

With ``--against <commit>`` nothing is patched: ``shipped`` is ``--commit`` and
``variant`` is that second commit as committed. That is how the dispatch which
landed is timed against the tree before it, with the same arms, controls and
identity check::

    python scripts/measure_bezier_cp_size_dispatch.py --commit <before> --against <after> --cpu <n>

The patched mode reads the kernel's pre-dispatch shape and refuses a header where
``contract_leading_axis`` is no longer a single-parameter template, so run it with
``--commit`` at a tree from before the dispatch landed.

How the timing is kept honest on a shared host
----------------------------------------------

- **Three arms, not two.** ``shipped`` runs in two independent processes, ``A``
  and ``A2``, beside ``variant`` in ``B``. ``A2/A`` is the setup's own spread:
  two processes running the identical extension. A ``B/A`` difference is called
  resolved only where its confidence interval excludes one **and** is disjoint
  from ``A2/A``'s. A difference smaller than that is not evidence.
- **Interleaved at the finest grain.** For every repetition and every cell the three
  arms are timed back to back, in an order rotated across cells and repetitions, so
  drifting load hits all three alike and no arm always goes first.
- **Paired ratios.** Each repetition yields ``B/A`` and ``A2/A`` from samples taken
  seconds apart; the medians of those ratios, not ratios of medians, are reported.
- **Fresh processes per block.** The repetitions are split into blocks, each with
  new processes, so a lucky or unlucky memory layout in one process becomes part
  of the spread rather than a bias of one arm.
- **One core, one thread.** Every timed process is pinned to ``--cpu`` and runs
  with ``OMP_NUM_THREADS``, ``NUMBA_NUM_THREADS`` and ``OPENBLAS_NUM_THREADS`` at
  one; the Numba warmup thread is awaited before anything is timed.
- **Two built-in controls.** A one-dimensional Bézier never reaches
  ``contract_leading_axis`` (``evaluate`` routes it to the fused 1-D kernel), so its
  cells can only resolve through a build or process effect; how many of them do is
  the false-positive rate of the verdict, and their ``B/A`` is what relinking a
  changed object does to code the patch never touched. And each timed process reports which
  extension it loaded, which backend is active and which class holds the Bézier,
  and the script stops if any of it is not what was built.

What it cannot show
-------------------

A loaded host inflates the spread, so an effect near the noise floor is reported as
unresolved rather than as absent: a failed check is not a disproof. It measures one
compiler at the toolchain baseline and ``float64`` only. The output-identity check
at the end compares the two builds' results over the sweep, byte for byte. It is
**not** the ticket's ``AC2``, which asks for both compilers, every ISA level and a
control that can fail; but a difference means the variant timed is not the shipped
arithmetic, so the script exits non-zero on one rather than report its speed.

No figure from any run appears in this file. Numbers belong in the output, next to
the commit, the compiler and the load average, which the script prints.
"""

from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import itertools
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable, Sequence
from io import BytesIO
from pathlib import Path
from typing import IO, Final, NamedTuple, NoReturn

_SPECIALISED_STRIDES: Final = (1, 2, 3, 4)
"""Block sizes the variant fixes at compile time.

Issue #481's closed set: the last direction of both schedules contracts against
`cp_size` values, one for a scalar field and at most four for a rational curve or
surface in 3-D."""

_DIMS: Final = (1, 2, 3)
"""Parametric dimensions. One is the control described in the module docstring."""

_DEGREES: Final = (1, 2, 3, 4, 5, 6)
"""Degrees per direction, the same in every direction."""

_CP_SIZES: Final = (1, 2, 3, 4)
"""Stored components per control point, weight column included."""

_RATIONAL_CP_SIZE: Final = 4
"""The component count built as a rational Bézier in 3-D rather than a rank-4 field.

That is where pantr reaches four components, so the projection it pays is part of
the call being timed."""

_GRID_POINTS: Final = (2, 8, 32)
"""Evaluation points per direction. Two is a per-element quadrature rule, the coarse
grid the ticket names; 32 is a tabulation grid."""

_FORMS: Final = ("points", "lattice")
"""How the grid is passed: a `(n_pts, dim)` array or a `PointsLattice`.

They reach the two different schedules of `evaluate.hpp`, which put their small
blocks in different places, so both are timed. A one-dimensional Bézier takes the
same path for both and is timed once."""

_ARMS: Final = ("A", "A2", "B")
"""The timed processes: shipped, shipped again, variant."""

_CONFIDENCE: Final = 0.95
"""Coverage of the order-statistic interval around each median ratio."""

_KERNEL_SIGNATURE: Final = "void contract_leading_axis("
"""How the kernel is found in `evaluate.hpp`, rather than by a line number."""

_HEADER: Final = Path("cpp/include/pantr/bezier/evaluate.hpp")
"""The header the variant patches, relative to an exported tree."""

_EXTENSION_GLOB: Final = "cpp/bindings/_pantr_cpp*.so"
"""Where the project's CMake leaves the extension, relative to a build tree."""


class Cell(NamedTuple):
    """One configuration of the sweep.

    Attributes:
        dim (int): Parametric dimension.
        degree (int): Degree in every direction.
        cp_size (int): Stored components per control point, weight included.
        form (str): One of :data:`_FORMS`.
        grid (int): Evaluation points per direction.
    """

    dim: int
    degree: int
    cp_size: int
    form: str
    grid: int


class Estimate(NamedTuple):
    """A median with its distribution-free confidence interval.

    Attributes:
        median (float): The sample median.
        low (float): Lower end of the interval.
        high (float): Upper end of the interval.
    """

    median: float
    low: float
    high: float


class Row(NamedTuple):
    """Everything reported for one cell.

    Attributes:
        cell (Cell): The configuration.
        calls (int): `evaluate` calls per timed sample.
        shipped (float): Median seconds per call of arm A.
        shipped_iqr (float): Interquartile range of arm A over its median.
        variant (float): Median seconds per call of arm B.
        variant_iqr (float): Interquartile range of arm B over its median.
        ratio (Estimate): `B/A`, paired per repetition.
        null (Estimate): `A2/A`, paired per repetition.
        verdict (str): ``faster``, ``slower`` or ``unresolved``.
    """

    cell: Cell
    calls: int
    shipped: float
    shipped_iqr: float
    variant: float
    variant_iqr: float
    ratio: Estimate
    null: Estimate
    verdict: str


def repo_root() -> Path:
    """Locate the checkout this script lives in.

    Returns:
        Path: The repository root, from this file rather than the working directory.
    """
    return Path(__file__).resolve().parent.parent


def sweep(dims: Sequence[int], degrees: Sequence[int], grids: Sequence[int]) -> list[Cell]:
    """Enumerate the cells, in the order they are timed.

    Args:
        dims (Sequence[int]): Parametric dimensions.
        degrees (Sequence[int]): Degrees.
        grids (Sequence[int]): Points per direction.

    Returns:
        list[Cell]: Every combination, with the lattice form dropped at dimension one.
    """
    cells: list[Cell] = []
    for dim, degree, cp_size, form, grid in itertools.product(
        dims, degrees, _CP_SIZES, _FORMS, grids
    ):
        if dim == 1 and form == "lattice":
            continue
        cells.append(Cell(dim, degree, cp_size, form, grid))
    return cells


# ---- building the two extensions ------------------------------------------------------------


def export_tree(commit: str, dest: Path) -> None:
    """Write a commit's tracked files to a directory.

    Args:
        commit (str): Anything `git archive` accepts.
        dest (Path): An empty directory.
    """
    archive = subprocess.run(
        ["git", "archive", "--format=tar", commit],
        capture_output=True,
        check=True,
        cwd=repo_root(),
    ).stdout
    with tarfile.open(fileobj=BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")


def specialise_kernel(header: str) -> tuple[str, str]:
    """Rewrite `contract_leading_axis` to dispatch small block sizes to fixed variants.

    Args:
        header (str): The text of `evaluate.hpp`.

    Returns:
        tuple[str, str]: The patched header, and the patched region alone for the
        report.

    Raises:
        LookupError: If the kernel is not found exactly once in the shape this
            function reads, or if fixing `stride` in its body changes nothing. Both
            mean the header moved and the variant would not be the one described.
    """
    lines = header.split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith(_KERNEL_SIGNATURE)]
    if len(starts) != 1 or lines[starts[0] - 1] != "template <Real T>":
        raise LookupError(f"expected one templated {_KERNEL_SIGNATURE!r}, found {len(starts)}")
    first = starts[0] - 1
    opening = next(i for i in range(starts[0], len(lines)) if lines[i].endswith(") {"))
    closing = next(i for i in range(opening, len(lines)) if lines[i] == "}")
    signature = lines[first : opening + 1]
    body = "\n".join(lines[opening + 1 : closing])

    fixed_body, count = re.subn(r"\bstride\b", "Stride", body)
    if count == 0:
        raise LookupError("the kernel body no longer reads `stride`; nothing to specialise")
    fixed = "contract_leading_axis_fixed_stride"
    cases = "\n".join(
        f"        case {s}: {fixed}<T, {s}>(weights, block, out); return;"
        for s in _SPECIALISED_STRIDES
    )
    region = "\n".join(
        [
            "// Scratch variant written by scripts/measure_bezier_cp_size_dispatch.py.",
            "template <Real T, std::size_t Stride>",
            f"void {fixed}(std::span<const T> weights,",
            "    std::span<const T> block, std::span<T> out) {",
            fixed_body,
            "}",
            "",
            *signature,
            "    switch (stride) {",
            cases,
            "        default: break;",
            "    }",
            body,
            "}",
        ]
    )
    patched = "\n".join([*lines[:first], region, *lines[closing + 1 :]])
    return patched, region


def build_extension(source: Path, build: Path, cmake_args: Sequence[str]) -> Path:
    """Configure and build the extension module of one exported tree.

    Args:
        source (Path): The exported tree.
        build (Path): The build directory.
        cmake_args (Sequence[str]): Extra configure arguments. Both arms get the
            user's; the variant also gets the shipped build's fetched sources.

    Returns:
        Path: The built extension.

    Raises:
        RuntimeError: If the build leaves anything but exactly one extension.
    """
    configure = [
        "cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DPANTR_BUILD_PYTHON=ON",
        "-DPANTR_BUILD_TESTS=OFF", "-DPANTR_BUILD_BENCHMARK=OFF",
        f"-DPython_EXECUTABLE={sys.executable}", *cmake_args,
    ]  # fmt: skip
    subprocess.run(configure, check=True, capture_output=True, text=True)
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "_pantr_cpp"],
        check=True,
        capture_output=True,
        text=True,
    )
    built = sorted(build.glob(_EXTENSION_GLOB))
    if len(built) != 1:
        raise RuntimeError(f"expected one extension under {build}, found {built}")
    return built[0]


def fetched_sources(build: Path) -> list[str]:
    """Point a second configure at the dependency sources a first one fetched.

    Args:
        build (Path): The first build tree.

    Returns:
        list[str]: `FETCHCONTENT_SOURCE_DIR_*` arguments for each source present.
    """
    arguments: list[str] = []
    for name in ("mdspan", "eigen"):
        src = build / "_deps" / f"{name}-src"
        if src.is_dir():
            arguments.append(f"-DFETCHCONTENT_SOURCE_DIR_{name.upper()}={src}")
    return arguments


def sha256(path: Path) -> str:
    """Hash a file.

    Args:
        path (Path): The file.

    Returns:
        str: The hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_value(build: Path, key: str) -> str:
    """Read one entry of a build tree's `CMakeCache.txt`.

    Args:
        build (Path): The build tree.
        key (str): The cache variable.

    Returns:
        str: Its value, or an empty string when absent.
    """
    match = re.search(rf"^{key}:\w+=(.*)$", (build / "CMakeCache.txt").read_text(), re.MULTILINE)
    return match.group(1) if match else ""


# ---- the timed process ----------------------------------------------------------------------


class _ExtensionFinder(importlib.abc.MetaPathFinder):
    """Serve `pantr._pantr_cpp` from one file, so each arm imports its own build.

    A finder rather than a pre-seeded `sys.modules` entry: the import machinery then
    loads the module itself and binds it on the parent package, which
    `pantr._backend` reads.

    Attributes:
        path (str): The extension to serve.
    """

    def __init__(self, path: str) -> None:
        """Remember the extension to serve.

        Args:
            path (str): The extension file.
        """
        self.path = path

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: object = None
    ) -> importlib.machinery.ModuleSpec | None:
        """Answer for the extension's name and for nothing else.

        Args:
            fullname (str): The module being imported.
            path (Sequence[str] | None): The parent package's search path.
            target (object): Unused.

        Returns:
            importlib.machinery.ModuleSpec | None: A spec for the chosen file, or
            None to defer to the other finders.
        """
        if fullname != "pantr._pantr_cpp":
            return None
        return importlib.util.spec_from_file_location(fullname, self.path)


def _evaluation_call(cell: Cell, index: int) -> Callable[[], object]:
    """Build one cell's Bézier and points, and return the call that is timed.

    The data is seeded by the cell's index, so every arm evaluates the same numbers.

    Args:
        cell (Cell): The configuration.
        index (int): Its position in the sweep.

    Returns:
        Callable[[], object]: `bezier.evaluate(points)`, already run once.

    Raises:
        RuntimeError: If the Bézier does not hold the C++ implementation.
    """
    import numpy as np  # noqa: PLC0415  (only the timed process needs these)
    import numpy.typing as npt  # noqa: PLC0415

    from pantr import _pantr_cpp  # noqa: PLC0415
    from pantr.bezier import Bezier  # noqa: PLC0415
    from pantr.quad import PointsLattice  # noqa: PLC0415

    rng = np.random.default_rng([0x481, index])
    rational = cell.cp_size == _RATIONAL_CP_SIZE
    shape = (*([cell.degree + 1] * cell.dim), cell.cp_size)
    control = rng.uniform(-1.0, 1.0, size=shape)
    if rational:
        control[..., -1] = rng.uniform(0.5, 1.5, size=shape[:-1])
    bezier = Bezier(control, is_rational=rational)
    if not isinstance(bezier._impl, _pantr_cpp.Bezier64):
        raise RuntimeError(f"Bezier holds {type(bezier._impl).__name__}, not the C++ type")
    # Cell midpoints, so no point sits on an endpoint; the same points for both forms.
    axis = (np.arange(cell.grid, dtype=np.float64) + 0.5) / cell.grid
    pts: npt.NDArray[np.float64] | PointsLattice
    if cell.form == "lattice":
        pts = PointsLattice([axis] * cell.dim)
    elif cell.dim == 1:
        pts = axis
    else:
        mesh = np.meshgrid(*([axis] * cell.dim), indexing="ij")
        pts = np.stack([m.ravel() for m in mesh], axis=1)
    call = functools.partial(bezier.evaluate, pts)
    call()
    return call


def _calls_per_sample(call: Callable[[], object], seconds: float) -> int:
    """Size a sample so it lasts about `seconds`.

    The doubling loop terminates because `main` refuses a non-positive `seconds` and
    every `evaluate` call takes positive time, so `elapsed` grows with `n`.

    Args:
        call (Callable[[], object]): The timed call.
        seconds (float): The target duration.

    Returns:
        int: Calls per sample, at least one.
    """
    n, elapsed = 1, 0.0
    while elapsed < seconds / 4:
        start = time.perf_counter()
        for _ in range(n):
            call()
        elapsed = time.perf_counter() - start
        n *= 2
    return max(1, math.ceil(seconds / (elapsed / (n // 2))))


def _seconds_per_call(call: Callable[[], object], n: int) -> float:
    """Time `n` calls with the collector off.

    Args:
        call (Callable[[], object]): The timed call.
        n (int): How many.

    Returns:
        float: Mean seconds per call over the sample.
    """
    gc.disable()
    try:
        start = time.perf_counter_ns()
        for _ in range(n):
            call()
        elapsed = time.perf_counter_ns() - start
    finally:
        gc.enable()
    return elapsed * 1e-9 / n


def child(extension: str) -> NoReturn:
    """Serve timing requests from the parent over stdin and stdout.

    Args:
        extension (str): The extension this process must import. The process exits
            when the parent closes the request stream.

    Raises:
        RuntimeError: If the process imported a `pantr`, an extension or a backend
            other than the one it was started for.
        ValueError: On a request this protocol does not have.
    """
    sys.meta_path.insert(0, _ExtensionFinder(extension))
    import numpy as np  # noqa: PLC0415  (only the timed process needs these)

    import pantr  # noqa: PLC0415
    from pantr import _pantr_cpp  # noqa: PLC0415
    from pantr._backend import Backend, active_backend  # noqa: PLC0415
    from pantr._numba_compat import wait_for_jit_warmup  # noqa: PLC0415

    wait_for_jit_warmup()
    if not Path(pantr.__file__ or "").is_relative_to(repo_root() / "src"):
        raise RuntimeError(f"pantr imported from {pantr.__file__}, not from this checkout")
    if getattr(_pantr_cpp, "__file__", None) != extension:
        raise RuntimeError(f"extension {getattr(_pantr_cpp, '__file__', None)} != {extension}")
    if active_backend() is not Backend.CPP:
        raise RuntimeError(f"active backend is {active_backend()!r}, not CPP")

    calls: list[Callable[[], object]] = []
    for line in sys.stdin:
        request = json.loads(line)
        answer: dict[str, object]
        if request["op"] == "setup":
            calls = [_evaluation_call(Cell(*raw), i) for i, raw in enumerate(request["cells"])]
            answer = {"extension": _pantr_cpp.__file__, "affinity": sorted(os.sched_getaffinity(0))}
        elif request["op"] == "calibrate":
            answer = {"calls": _calls_per_sample(calls[request["cell"]], request["seconds"])}
        elif request["op"] == "time":
            answer = {"seconds": _seconds_per_call(calls[request["cell"]], request["calls"])}
        elif request["op"] == "digest":
            out = np.asarray(calls[request["cell"]]())
            answer = {"sha": hashlib.sha256(out.tobytes()).hexdigest()}
        else:
            raise ValueError(f"unknown request {request!r}")
        sys.stdout.write(json.dumps(answer) + "\n")
        sys.stdout.flush()
    # Skip interpreter finalization. A process whose affinity mask holds a single
    # CPU hangs at exit after `import pantr`, whatever the backend and whatever
    # `NUMBA_NUM_THREADS` says, while a two-CPU mask exits normally. That is a
    # defect outside this measurement and is reported rather than fixed here;
    # every answer has been flushed and this process owns no files, so nothing is
    # lost by not finalizing.
    os._exit(0)


class Arm:
    """One timed process, driven over a pipe.

    Attributes:
        name (str): One of :data:`_ARMS`.
        process (subprocess.Popen[str]): The child.
    """

    def __init__(self, name: str, extension: Path, cpu: int | None) -> None:
        """Start the child pinned to one core, with one thread everywhere.

        Args:
            name (str): One of :data:`_ARMS`.
            extension (Path): The extension it must import.
            cpu (int | None): The core to pin to, or None to inherit the mask.
        """
        self.name = name
        env = dict(os.environ)
        env.update(
            PANTR_BACKEND="cpp",
            OMP_NUM_THREADS="1",
            NUMBA_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTHONPATH=os.pathsep.join(
                [str(repo_root() / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]
            ),
        )
        pin = ["taskset", "-c", str(cpu)] if cpu is not None else []
        command = [*pin, sys.executable, __file__, "--child", str(extension)]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env
        )

    def ask(self, **request: object) -> dict[str, object]:
        """Send one request and wait for its answer.

        Args:
            **request (object): The request fields.

        Returns:
            dict[str, object]: The decoded answer.

        Raises:
            RuntimeError: If the child died instead of answering.
        """
        stdin: IO[str] | None = self.process.stdin
        stdout: IO[str] | None = self.process.stdout
        if stdin is None or stdout is None:  # Popen was given PIPE for both
            raise RuntimeError(f"arm {self.name} has no pipes")
        stdin.write(json.dumps(request) + "\n")
        stdin.flush()
        line = stdout.readline()
        if not line:
            raise RuntimeError(f"arm {self.name} exited with {self.process.wait()}")
        answer: dict[str, object] = json.loads(line)
        return answer

    def close(self) -> None:
        """Close the request stream, which ends the child's loop, and reap it.

        Raises:
            RuntimeError: If the child exited with anything but 0.
        """
        if self.process.stdin is not None:
            self.process.stdin.close()
        code = self.process.wait()
        if code != 0:
            raise RuntimeError(f"arm {self.name} exited with {code}")


# ---- statistics -----------------------------------------------------------------------------


def median_interval(values: Sequence[float], confidence: float = _CONFIDENCE) -> Estimate:
    """Estimate a median with a distribution-free interval from order statistics.

    The interval is `[x_(k), x_(n-k+1)]` with the largest `k` such that a
    Binomial(n, 1/2) variable falls below `k` with probability at most
    `(1 - confidence) / 2`. It assumes independent samples and nothing about their
    distribution, which is what a load-contaminated timing allows.

    Args:
        values (Sequence[float]): The samples.
        confidence (float): Coverage. Defaults to :data:`_CONFIDENCE`.

    Returns:
        Estimate: The median and the interval; the interval is the full range when
        there are too few samples to reach the coverage.
    """
    ordered = sorted(values)
    n = len(ordered)
    tail = (1.0 - confidence) / 2.0
    k, below = 0, 0.0
    while True:
        below_next = below + math.comb(n, k) / 2.0**n
        if below_next > tail:
            break
        below, k = below_next, k + 1
    low = ordered[k - 1] if k >= 1 else ordered[0]
    high = ordered[n - k] if k >= 1 else ordered[-1]
    return Estimate(statistics.median(ordered), low, high)


def relative_iqr(values: Sequence[float]) -> float:
    """Measure spread as the interquartile range over the median.

    Args:
        values (Sequence[float]): The samples.

    Returns:
        float: `(Q3 - Q1) / median`.
    """
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return (q3 - q1) / statistics.median(values)


def verdict(ratio: Estimate, null: Estimate) -> str:
    """Decide whether the variant's difference exceeds the setup's own.

    Args:
        ratio (Estimate): `B/A`.
        null (Estimate): `A2/A`.

    Returns:
        str: ``faster`` or ``slower`` where the `B/A` interval excludes one and does
        not overlap the `A2/A` interval, else ``unresolved``.
    """
    if ratio.high < 1.0 and ratio.high < null.low:
        return "faster"
    if ratio.low > 1.0 and ratio.low > null.high:
        return "slower"
    return "unresolved"


# ---- the driver -----------------------------------------------------------------------------


def uptime() -> str:
    """Read the host's uptime line, which carries its load averages.

    Returns:
        str: The output of `uptime`, or the load averages when it is unavailable.
    """
    if shutil.which("uptime"):
        return subprocess.run(["uptime"], capture_output=True, text=True, check=True).stdout.strip()
    return "load average " + ", ".join(f"{x:.2f}" for x in os.getloadavg())


def cpu_model() -> str:
    """Name the processor.

    Returns:
        str: The first `model name` of `/proc/cpuinfo`, else what `platform` reports.
    """
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        match = re.search(r"^model name\s*:\s*(.*)$", cpuinfo.read_text(), re.MULTILINE)
        if match:
            return match.group(1)
    return platform.processor() or platform.machine()


def set_up(arm: Arm, cells: list[Cell], extension: Path, cpu: int | None) -> None:
    """Have a timed process build the sweep, and check it is the process that was meant.

    Args:
        arm (Arm): The process.
        cells (list[Cell]): The sweep.
        extension (Path): The extension it was started with.
        cpu (int | None): The core it was pinned to, if any.

    Raises:
        RuntimeError: If it loaded another extension or runs outside its core.
    """
    who = arm.ask(op="setup", cells=[list(cell) for cell in cells])
    if who["extension"] != str(extension):
        raise RuntimeError(f"arm {arm.name} loaded {who['extension']}, not {extension}")
    if cpu is not None and who["affinity"] != [cpu]:
        raise RuntimeError(f"arm {arm.name} runs on CPUs {who['affinity']}, not [{cpu}]")


def run_blocks(
    cells: list[Cell],
    extensions: dict[str, Path],
    args: argparse.Namespace,
) -> tuple[dict[tuple[int, str], list[float]], list[int], dict[str, list[str]]]:
    """Time every cell in every arm, interleaved, over fresh processes per block.

    Args:
        cells (list[Cell]): The sweep.
        extensions (dict[str, Path]): The extension each arm imports.
        args (argparse.Namespace): The parsed command line.

    Returns:
        tuple[dict[tuple[int, str], list[float]], list[int], dict[str, list[str]]]:
        Seconds per call keyed by cell index and arm, the calls per sample per
        cell, and the output digests per arm from the last block.
    """
    samples: dict[tuple[int, str], list[float]] = {}
    counts: list[int] = []
    digests: dict[str, list[str]] = {}
    for block in range(args.blocks):
        arms = [Arm(name, extensions[name], args.cpu) for name in _ARMS]
        try:
            for arm in arms:
                set_up(arm, cells, extensions[arm.name], args.cpu)
            if block == 0:
                for index in range(len(cells)):
                    answer = arms[0].ask(op="calibrate", cell=index, seconds=args.seconds)
                    counts.append(int(str(answer["calls"])))
            for rep in range(args.reps):
                for index in range(len(cells)):
                    shift = (block * args.reps + rep + index) % len(arms)
                    for arm in arms[shift:] + arms[:shift]:
                        answer = arm.ask(op="time", cell=index, calls=counts[index])
                        seconds = float(str(answer["seconds"]))
                        samples.setdefault((index, arm.name), []).append(seconds)
                print(
                    f"  block {block + 1}/{args.blocks}, repetition {rep + 1}/{args.reps}",
                    file=sys.stderr,
                )
            if block == args.blocks - 1:
                for arm in arms:
                    digests[arm.name] = [
                        str(arm.ask(op="digest", cell=i)["sha"]) for i in range(len(cells))
                    ]
        finally:
            for arm in arms:
                arm.close()
    return samples, counts, digests


def summarise(
    cells: list[Cell], samples: dict[tuple[int, str], list[float]], counts: list[int]
) -> list[Row]:
    """Reduce the samples to one row per cell.

    Args:
        cells (list[Cell]): The sweep.
        samples (dict[tuple[int, str], list[float]]): From :func:`run_blocks`.
        counts (list[int]): Calls per sample per cell.

    Returns:
        list[Row]: One per cell, in sweep order.
    """
    rows: list[Row] = []
    for index, cell in enumerate(cells):
        a, a2, b = (samples[(index, name)] for name in _ARMS)
        ratio = median_interval([y / x for x, y in zip(a, b, strict=True)])
        null = median_interval([y / x for x, y in zip(a, a2, strict=True)])
        rows.append(
            Row(
                cell,
                counts[index],
                statistics.median(a),
                relative_iqr(a),
                statistics.median(b),
                relative_iqr(b),
                ratio,
                null,
                verdict(ratio, null),
            )
        )
    return rows


def print_table(rows: list[Row]) -> None:
    """Print one line per cell.

    Args:
        rows (list[Row]): From :func:`summarise`.
    """
    print(
        f"{'dim':>3} {'p':>2} {'cp':>2} {'form':>7} {'grid':>4} {'calls':>6} "
        f"{'A us':>10} {'iqr%':>5} {'B us':>10} {'iqr%':>5} "
        f"{'B/A':>6} {'95% interval':>15} {'A2/A':>6} {'95% interval':>15}  verdict"
    )
    for row in rows:
        c = row.cell
        print(
            f"{c.dim:>3} {c.degree:>2} {c.cp_size:>2} {c.form:>7} {c.grid:>4} {row.calls:>6} "
            f"{row.shipped * 1e6:>10.4g} {row.shipped_iqr * 100:>5.1f} "
            f"{row.variant * 1e6:>10.4g} {row.variant_iqr * 100:>5.1f} "
            f"{row.ratio.median:>6.3f} [{row.ratio.low:.3f}, {row.ratio.high:.3f}] "
            f"{row.null.median:>6.3f} [{row.null.low:.3f}, {row.null.high:.3f}]  {row.verdict}"
        )


def print_summary(rows: list[Row]) -> None:
    """Print the verdict counts per dimension, form and grid, and the control's rate.

    Args:
        rows (list[Row]): From :func:`summarise`.
    """
    print("\nPer (dim, form, grid), over degrees and component counts:")
    print(
        f"{'dim':>3} {'form':>7} {'grid':>4} {'cells':>5} {'faster':>6} {'slower':>6} "
        f"{'median B/A':>10} {'min B/A':>8} {'median null width':>17}"
    )
    keyed = sorted({(r.cell.dim, r.cell.form, r.cell.grid) for r in rows})
    for dim, form, grid in keyed:
        group = [r for r in rows if (r.cell.dim, r.cell.form, r.cell.grid) == (dim, form, grid)]
        faster = sum(r.verdict == "faster" for r in group)
        slower = sum(r.verdict == "slower" for r in group)
        width = statistics.median(r.null.high - r.null.low for r in group)
        print(
            f"{dim:>3} {form:>7} {grid:>4} {len(group):>5} {faster:>6} {slower:>6} "
            f"{statistics.median(r.ratio.median for r in group):>10.3f} "
            f"{min(r.ratio.median for r in group):>8.3f} {width:>17.3f}"
        )
    control = [r for r in rows if r.cell.dim == 1]
    nd = [r for r in rows if r.cell.dim > 1]
    if control:
        resolved_control = sum(r.verdict != "unresolved" for r in control)
        offsets = sorted(r.ratio.median for r in control)
        print(
            f"\nControl (dim 1 never reaches the kernel): {resolved_control} of {len(control)} "
            "cells resolved, the verdict's false-positive rate on this run. Its B/A medians"
            f"\nrun from {offsets[0]:.3f} to {offsets[-1]:.3f} (median "
            f"{statistics.median(offsets):.3f}): what the rebuild alone does to code the "
            "patch never touched."
        )
    for label in ("faster", "slower"):
        print(f"n-d cells {label}: {sum(r.verdict == label for r in nd)} of {len(nd)}")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv (Sequence[str] | None): Arguments, or None for `sys.argv`.

    Returns:
        argparse.Namespace: The parsed options.
    """
    parser = argparse.ArgumentParser(description="End-to-end cp_size dispatch timing, #481 AC1.")
    parser.add_argument("--child", metavar="SO", help=argparse.SUPPRESS)
    parser.add_argument(
        "--commit", default="HEAD", help="commit for shipped, and variant unless --against"
    )
    parser.add_argument("--against", help="build the variant from this commit, unpatched")
    parser.add_argument("--cpu", type=int, help="pin every timed process to this CPU")
    parser.add_argument("--blocks", type=int, default=3, help="fresh sets of processes")
    parser.add_argument("--reps", type=int, default=10, help="repetitions per block")
    parser.add_argument("--seconds", type=float, default=0.02, help="target seconds per sample")
    parser.add_argument("--dims", type=int, nargs="+", default=list(_DIMS))
    parser.add_argument("--degrees", type=int, nargs="+", default=list(_DEGREES))
    parser.add_argument("--grids", type=int, nargs="+", default=list(_GRID_POINTS))
    parser.add_argument("--cmake-arg", action="append", default=[], help="passed to both builds")
    parser.add_argument("--work-dir", type=Path, help="keep the trees and builds here")
    parser.add_argument("--raw", type=Path, help="write every sample to this JSON file")
    return parser.parse_args(argv)


def main() -> int:
    """Dispatch to the timed process or to the measurement.

    Returns:
        int: What :func:`measure` returned, or 1 when there are too few samples for
        the interval to exclude anything. A timed process never returns here.
    """
    args = parse_arguments()
    if args.child is not None:
        child(args.child)
    if args.seconds <= 0:
        print("--seconds must be positive.", file=sys.stderr)
        return 1
    # The interval can exclude anything only once x_(1) is itself inside it, which
    # needs P(all n samples on one side of the median) = 2^-n below the tail mass.
    if 0.5 ** (args.blocks * args.reps) > (1.0 - _CONFIDENCE) / 2.0:
        print("Too few repetitions for the interval to exclude anything.", file=sys.stderr)
        return 1
    if args.work_dir is not None:
        return measure(args.work_dir, args)
    with tempfile.TemporaryDirectory(prefix="pantr-481-") as scratch:
        return measure(Path(scratch), args)


def resolve(revision: str) -> str:
    """Resolve a revision to the commit SHA both exports are taken from.

    Args:
        revision (str): Anything `git rev-parse` accepts.

    Returns:
        str: The full SHA.
    """
    return subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        capture_output=True, text=True, check=True, cwd=repo_root(),
    ).stdout.strip()  # fmt: skip


def export_arms(work: Path, commit: str, against: str | None) -> tuple[dict[str, Path], str, str]:
    """Export the two source trees, patching the variant's unless it has its own commit.

    Args:
        work (Path): The directory the trees go under.
        commit (str): The shipped arm's commit, and the variant's too when `against`
            is None.
        against (str | None): The variant's commit, exported unpatched, or None to
            patch `commit`'s tree with :func:`specialise_kernel`.

    Returns:
        tuple[dict[str, Path], str, str]: The trees by arm name, the patched region
        (empty when nothing was patched), and a line saying what each arm is.
    """
    trees = {name: work / f"{name}-src" for name in ("shipped", "variant")}
    for name, tree in trees.items():
        tree.mkdir(parents=True, exist_ok=False)
        export_tree(against if name == "variant" and against is not None else commit, tree)
    if against is not None:
        return trees, "", f"shipped = commit {commit}; variant = commit {against}, unpatched"
    header = trees["variant"] / _HEADER
    patched, region = specialise_kernel(header.read_text())
    header.write_text(patched)
    return trees, region, f"commit {commit} (both arms); variant = that commit + the patch below"


def measure(work: Path, args: argparse.Namespace) -> int:
    """Build both extensions, time them, and report.

    Args:
        work (Path): An empty or absent directory for the trees and builds.
        args (argparse.Namespace): The parsed command line.

    Returns:
        int: 0 when the run completed with both arms verified, 1 when the two builds
        turned out identical, used different compilers, changed on disk, or computed
        different outputs.
    """
    commit = resolve(args.commit)
    against = resolve(args.against) if args.against is not None else None
    load_before = uptime()
    cells = sweep(args.dims, args.degrees, args.grids)

    trees, region, provenance = export_arms(work, commit, against)
    print("Building the shipped extension ...", file=sys.stderr)
    shipped = build_extension(trees["shipped"], work / "shipped-build", args.cmake_arg)
    print("Building the variant extension ...", file=sys.stderr)
    variant = build_extension(
        trees["variant"],
        work / "variant-build",
        [*args.cmake_arg, *fetched_sources(work / "shipped-build")],
    )
    if sha256(shipped) == sha256(variant):
        print("The two builds are identical: nothing differs to time.", file=sys.stderr)
        return 1
    compiler = cache_value(work / "shipped-build", "CMAKE_CXX_COMPILER")
    if cache_value(work / "variant-build", "CMAKE_CXX_COMPILER") != compiler:
        print("The two builds used different compilers.", file=sys.stderr)
        return 1
    compiler_version = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()[0]
    hashes = {"shipped": sha256(shipped), "variant": sha256(variant)}

    print(
        f"Timing {len(cells)} cells x {len(_ARMS)} arms x {args.blocks * args.reps} samples",
        file=sys.stderr,
    )
    extensions = {"A": shipped, "A2": shipped, "B": variant}
    samples, counts, digests = run_blocks(cells, extensions, args)
    load_after = uptime()
    if {"shipped": sha256(shipped), "variant": sha256(variant)} != hashes:
        print("An extension changed on disk while it was being timed.", file=sys.stderr)
        return 1

    print("\n#481 AC1: Bezier.evaluate end to end, shipped (A, A2) against variant (B)")
    print("=" * 100)
    print(provenance)
    print(f"python {sys.version.split()[0]}, {platform.platform()}")
    print(f"cpu {cpu_model()}; timed processes pinned to CPU {args.cpu}, one thread each")
    print(f"compiler {compiler_version} ({compiler})")
    for name, path in (("shipped", shipped), ("variant", variant)):
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))
        print(f"{name}: {path} (mtime {mtime}, sha256 {hashes[name][:16]})")
    print(f"load before: {load_before}")
    print(f"load after:  {load_after}")
    print(
        f"SHARED HOST: {os.cpu_count()} logical CPUs, used by others while this ran (load "
        "averages above); every spread below includes their load."
    )
    print(f"invocation: {' '.join([Path(sys.executable).name, *sys.argv])}")
    if region:
        print("\nThe variant's kernel, as built:\n")
        print(region)

    rows = summarise(cells, samples, counts)
    print(
        "\nSeconds are per evaluate call (us). iqr% is the arm's interquartile range over its"
        "\nmedian. B/A and A2/A are medians of per-repetition paired ratios with a 95%"
        "\norder-statistic interval; below one means B is faster. 'faster'/'slower' only where"
        "\nthe B/A interval excludes one and misses the A2/A interval.\n"
    )
    print_table(rows)
    print_summary(rows)

    differing = sum(a != b for a, b in zip(digests["A"], digests["B"], strict=True))
    print(
        f"\nOutput identity, shipped against variant: {len(cells) - differing} of {len(cells)} "
        "cells byte-identical (a sanity check on what was timed, not #481 AC2)."
    )
    if args.raw is not None:
        args.raw.write_text(
            json.dumps(
                {
                    "commit": commit,
                    "against": against,
                    "cells": [list(c) for c in cells],
                    "calls": counts,
                    "samples": {f"{i}:{arm}": v for (i, arm), v in samples.items()},
                }
            )
        )
    return 1 if differing else 0


if __name__ == "__main__":
    raise SystemExit(main())
