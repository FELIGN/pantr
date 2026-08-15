"""Command-line entry point for the adversarial parameter sweep.

The sweep must run with Numba's bounds check enabled *and* a cache directory that
does not already hold an unchecked compilation of the kernels, because
``NUMBA_BOUNDSCHECK=1`` is silently ignored when a stale cache entry exists (the
cache key does not include the flag) and every pantr kernel is ``cache=True``. Rather
than rely on the caller getting that right, :func:`main` re-executes itself once with
a fresh temporary cache directory when the environment is not already configured.

Run it through the launcher, which puts ``src`` and ``tools`` on the path::

    conda run -n pantr python tools/sweep.py --profile smoke
    conda run -n pantr python tools/sweep.py --profile full --journal sweep.jsonl

Exit codes: ``0`` no bugs found, ``1`` bugs found, ``3`` the harness itself is
unusable (the bounds-check canary did not fire), ``4`` the run did not complete
because a probe generator raised while building a case. ``4`` is deliberately
distinct from ``1``: an incomplete run's counts mean nothing, and reading a
truncation as "some findings, run finished" is the one mistake that would make a
clean report untrustworthy.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import tempfile
from typing import TYPE_CHECKING, Final, TextIO

if TYPE_CHECKING:
    from ._core import Summary

_REEXEC_FLAG: Final = "PANTR_SWEEP_CONFIGURED"
"""Environment marker set on re-exec so the fix-up can never loop."""

EXIT_CLEAN: Final = 0
EXIT_FINDINGS: Final = 1
EXIT_HARNESS_UNUSABLE: Final = 3
EXIT_INCOMPLETE: Final = 4


def _launcher_path() -> pathlib.Path:
    """Locate ``tools/sweep.py``, the script a re-exec must invoke.

    Returns:
        pathlib.Path: Absolute path to the launcher.
    """
    return pathlib.Path(__file__).resolve().parents[1] / "sweep.py"


def _reexec_with_boundscheck() -> None:
    """Re-execute this process with the bounds check on and a fresh Numba cache.

    Does nothing when the environment is already configured, or when the marker
    shows this process is itself the result of a re-exec.
    """
    if os.environ.get(_REEXEC_FLAG) == "1":
        return
    env = dict(os.environ)
    env[_REEXEC_FLAG] = "1"
    env["NUMBA_BOUNDSCHECK"] = "1"
    if not env.get("NUMBA_CACHE_DIR"):
        env["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="pantr-sweep-nbcache-")
    print(
        f"[sweep] re-exec with NUMBA_BOUNDSCHECK=1 NUMBA_CACHE_DIR={env['NUMBA_CACHE_DIR']}",
        file=sys.stderr,
        flush=True,
    )
    launcher = str(_launcher_path())
    os.execve(sys.executable, [sys.executable, launcher, *sys.argv[1:]], env)


def _build_parser(group_names: tuple[str, ...]) -> argparse.ArgumentParser:
    """Build the command-line parser.

    Args:
        group_names (tuple[str, ...]): Registered sweep group names.

    Returns:
        argparse.ArgumentParser: The parser.
    """
    parser = argparse.ArgumentParser(
        prog="sweep.py",
        description="Adversarial parameter sweep over pantr's public entry points.",
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "full"),
        default="full",
        help="Sweep width: 'smoke' is the bounded CI subset, 'full' the whole space.",
    )
    parser.add_argument(
        "--group",
        action="append",
        choices=group_names,
        metavar="NAME",
        help="Restrict to one group; repeatable. Defaults to every group.",
    )
    parser.add_argument("--list-groups", action="store_true", help="Print the groups and exit.")
    parser.add_argument(
        "--journal",
        type=pathlib.Path,
        default=None,
        help="Write the per-case JSONL records here instead of stdout.",
    )
    parser.add_argument(
        "--dump-npz",
        type=pathlib.Path,
        default=None,
        help="Persist each case's declared input arrays as .npz parity fixtures.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip cases before this index, to resume past a hang or a hard crash.",
    )
    parser.add_argument(
        "--max-cases", type=int, default=None, help="Stop after this many executed cases."
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Enumerate the selected cases and report how many there are, running none.",
    )
    return parser


def _report(summary: Summary, stream: TextIO) -> None:
    """Print the human-readable summary and every finding.

    Args:
        summary (Summary): Aggregate sweep result.
        stream (TextIO): Text stream to print to.
    """
    print(f"\n{'=' * 78}", file=stream)
    if summary.aborted is not None:
        print(
            f"RUN INCOMPLETE: {summary.aborted}\n"
            "The counts below cover only the cases built before the failure and mean "
            "nothing as a coverage claim.",
            file=stream,
        )
    print(f"cases run: {summary.total}", file=stream)
    for name, count in summary.counts.items():
        print(f"  {name:24s} {count}", file=stream)

    if summary.warned:
        print(
            f"\n--- WARNED WHILE RETURNING ({len(summary.warned)}): NumPy reports int64 "
            "overflow and invalid operations this way ---",
            file=stream,
        )
        for outcome in summary.warned:
            print(
                f"  [{outcome.index}] {outcome.case.group}/{outcome.case.label}"
                f"  {','.join(outcome.warnings)}",
                file=stream,
            )

    if summary.suspected:
        print(
            f"\n--- SUSPECTED ({len(summary.suspected)}): "
            "ValueError/TypeError not listed in the entry point's Raises: section ---",
            file=stream,
        )
        for outcome in summary.suspected:
            print(
                f"  [{outcome.index}] {outcome.case.group}/{outcome.case.label}\n"
                f"      {outcome.detail}",
                file=stream,
            )

    if summary.findings:
        print(f"\n--- BUGS ({len(summary.findings)}) ---", file=stream)
        for outcome in summary.findings:
            print(
                f"  [{outcome.index}] {outcome.case.group}/{outcome.case.label}\n"
                f"      kind: {outcome.kind}\n"
                f"      {outcome.detail}",
                file=stream,
            )
    else:
        print("\nno bugs found", file=stream)


def main(argv: list[str] | None = None) -> int:
    """Run the sweep.

    Args:
        argv (list[str] | None): Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        int: Process exit code.
    """
    _reexec_with_boundscheck()

    from ._axes import Profile  # noqa: PLC0415 -- deferred until the env is fixed up
    from ._core import CanaryError, Verdict, assert_boundscheck_active, run_sweep  # noqa: PLC0415
    from ._registry import GROUPS, iter_cases  # noqa: PLC0415

    parser = _build_parser(tuple(GROUPS))
    args = parser.parse_args(argv)

    if args.list_groups:
        for name in GROUPS:
            print(name)
        return EXIT_CLEAN

    try:
        print(f"[sweep] {assert_boundscheck_active()}", file=sys.stderr, flush=True)
    except CanaryError as exc:
        print(f"[sweep] CANARY FAILED: {exc}", file=sys.stderr, flush=True)
        return EXIT_HARNESS_UNUSABLE

    profile = Profile.SMOKE if args.profile == "smoke" else Profile.FULL
    groups = tuple(args.group) if args.group else tuple(GROUPS)
    print(f"[sweep] profile={profile.name} groups={','.join(groups)}", file=sys.stderr, flush=True)

    if args.count_only:
        total = sum(1 for _ in iter_cases(profile, groups))
        print(f"cases: {total}")
        return EXIT_CLEAN

    journal = args.journal.open("w", encoding="utf-8") if args.journal else sys.stdout
    try:
        summary = run_sweep(
            iter_cases(profile, groups),
            journal=journal,
            progress=sys.stderr,
            start_index=args.start_index,
            max_cases=args.max_cases,
            dump_dir=args.dump_npz,
        )
    finally:
        if journal is not sys.stdout:
            journal.close()

    _report(summary, sys.stderr)
    if summary.aborted is not None:
        return EXIT_INCOMPLETE
    return EXIT_FINDINGS if summary.counts[Verdict.BUG.name] else EXIT_CLEAN


if __name__ == "__main__":
    raise SystemExit(main())
