"""Registry mapping sweep group names to their case generators.

Adding a probe module means importing it here and adding one entry to
:data:`GROUPS`; the CLI's ``--group`` choices and ``--list-groups`` output are
derived from it.
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

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from ._axes import Profile
    from ._core import Case

GROUPS: Final[dict[str, Callable[[Profile], Iterator[Case]]]] = {
    _probes_geometry.GROUP: _probes_geometry.cases,
    _probes_quad.GROUP: _probes_quad.cases,
    _probes_basis.GROUP: _probes_basis.cases,
    _probes_bspline.GROUP: _probes_bspline.cases,
    _probes_bezier.GROUP: _probes_bezier.cases,
    _probes_grid.GROUP: _probes_grid.cases,
}
"""Group name to case generator. Iteration order fixes the sweep's case indices."""


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
            yield from generator(profile)
