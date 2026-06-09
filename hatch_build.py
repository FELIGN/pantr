"""Hatchling metadata hook for the env-var-conditional MPI dependency.

PaNTr depends on ``mpi4py`` by default. Setting the ``PANTR_NO_MPI`` environment
variable at build/install time drops that dependency, yielding a serial-only,
MPI-free install. The always-on runtime dependencies live in
``[tool.pantr] base-dependencies`` in ``pyproject.toml``; this hook reads them and
appends ``mpi4py`` unless ``PANTR_NO_MPI`` is set in the build environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from hatchling.metadata.plugin.interface import MetadataHookInterface

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_MPI_DEPENDENCY = "mpi4py>=3.1,<5"
"""Optional MPI dependency, added unless the opt-out flag is set."""

_OPT_OUT_ENV = "PANTR_NO_MPI"
"""Environment variable that, when set to a non-empty value, drops ``mpi4py``."""


class CustomMetadataHook(MetadataHookInterface):
    """Inject ``mpi4py`` as a default dependency unless ``PANTR_NO_MPI`` is set."""

    def update(self, metadata: dict[str, Any]) -> None:
        """Set ``metadata['dependencies']`` to the base deps plus optional ``mpi4py``.

        Args:
            metadata: The project metadata mapping to mutate in place. Its
                ``dependencies`` key is populated with the resolved dependency list.
        """
        pyproject = Path(self.root) / "pyproject.toml"
        with pyproject.open("rb") as fh:
            data = tomllib.load(fh)

        deps: list[str] = list(data["tool"]["pantr"]["base-dependencies"])
        if not os.environ.get(_OPT_OUT_ENV):
            deps.append(_MPI_DEPENDENCY)
        metadata["dependencies"] = deps
