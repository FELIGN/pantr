"""Registry mapping sweep group names to their case generators.

Adding a probe module means importing it here and adding one entry to
:data:`GROUPS`; the CLI's ``--group`` choices and ``--list-groups`` output are
derived from it.

Enumeration is guarded per group. A probe generator builds its cases lazily, so a
pantr call in a generator *body* -- constructing a fixture, evaluating a reference
for an invariant -- raises during iteration rather than inside a case, where the
runner would classify it. Unguarded, that ends the whole sweep: the remaining groups
never run and the exit status looks like an ordinary findings report. So
:func:`iter_cases` converts such a failure into one synthetic case that re-raises when
run, which the runner then reports as a finding with its traceback, and carries on with
the next group. The cases after the failure *within* that group are still lost --
Python closes a generator that raises -- but five groups out of six are not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from . import (
    _probes_basis,
    _probes_bezier,
    _probes_bspline,
    _probes_geometry,
    _probes_grid,
    _probes_quad,
)
from ._core import Case

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from ._axes import Profile

GROUPS: Final[dict[str, Callable[[Profile], Iterator[Case]]]] = {
    _probes_geometry.GROUP: _probes_geometry.cases,
    _probes_quad.GROUP: _probes_quad.cases,
    _probes_basis.GROUP: _probes_basis.cases,
    _probes_bspline.GROUP: _probes_bspline.cases,
    _probes_bezier.GROUP: _probes_bezier.cases,
    _probes_grid.GROUP: _probes_grid.cases,
}
"""Group name to case generator. Iteration order fixes the sweep's case indices."""


def _enumeration_failure(group: str, exc: Exception) -> Case:
    """Wrap an enumeration failure as a case that reports it.

    Args:
        group (str): The group whose generator raised.
        exc (Exception): What it raised.

    Returns:
        Case: A case whose thunk re-raises, so the runner classifies and journals it.
    """

    def reraise() -> None:
        raise exc

    return Case(
        group,
        "__enumeration_failed__",
        reraise,
        reraise,
        {"error": type(exc).__name__, "message": str(exc)},
        must_succeed=True,
    )


def _guarded(
    group: str, generator: Callable[[Profile], Iterator[Case]], profile: Profile
) -> Iterator[Case]:
    """Yield a group's cases, turning an enumeration failure into a reported case.

    Args:
        group (str): Group name, for the synthetic case's label.
        generator (Callable[[Profile], Iterator[Case]]): The group's case generator.
        profile (Profile): Sweep width.

    Yields:
        Case: The group's cases, followed by one synthetic case if enumeration failed.
    """
    try:
        yield from generator(profile)
    except Exception as exc:  # one bad group must not end the sweep
        yield _enumeration_failure(group, exc)


def iter_cases(profile: Profile, groups: Iterable[str]) -> Iterator[Case]:
    """Yield the cases of the selected groups, in registry order.

    Args:
        profile (Profile): Sweep width.
        groups (Iterable[str]): Group names to include.

    Yields:
        Case: The selected cases.

    Raises:
        KeyError: If a name is not a registered group.
    """
    wanted = set(groups)
    unknown = wanted - set(GROUPS)
    if unknown:
        raise KeyError(f"unknown sweep groups: {sorted(unknown)}")
    for name, generator in GROUPS.items():
        if name in wanted:
            yield from _guarded(name, generator, profile)
