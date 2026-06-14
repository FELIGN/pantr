"""Sphinx configuration for PaNTr documentation.

Initializes metadata, extensions, and build parameters.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from datetime import date
from pathlib import Path
from typing import Final

import pyvista
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper

# Disable Numba JIT during documentation build. This avoids issues with
# JIT caching and potential concurrent-compilation crashes while Sphinx
# imports the package.
os.environ["NUMBA_DISABLE_JIT"] = "1"

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
SRC_PATH: Final[Path] = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

pantr_spec = importlib.util.spec_from_file_location("pantr", SRC_PATH / "pantr" / "__init__.py")
if pantr_spec is None or pantr_spec.loader is None:
    msg = f"Unable to locate pantr package at {SRC_PATH / 'pantr' / '__init__.py'}"
    raise ImportError(msg)
pantr = importlib.util.module_from_spec(pantr_spec)
pantr_spec.loader.exec_module(pantr)
CURRENT_YEAR: Final[int] = date.today().year

project = "PaNTr"
author = "Pablo Antolin"
copyright = f"{CURRENT_YEAR}, Pablo Antolin"  # pylint: disable=redefined-builtin

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "pyvista.ext.viewer_directive",
    "sphinx_rtd_dark_mode",
]

OPTIONAL_EXTENSIONS: Final[list[str]] = [
    "sphinx_rtd_dark_mode",
]

for ext in OPTIONAL_EXTENSIONS:
    try:
        if importlib.util.find_spec(ext) is None:
            raise ImportError
    except (ImportError, ModuleNotFoundError):
        if ext in extensions:
            warnings.warn(
                f"Skipping optional Sphinx extension {ext!r}: module not found.",
                stacklevel=1,
            )
            extensions = [e for e in extensions if e != ext]

_INTERSPHINX_DIR: Final[Path] = PROJECT_ROOT / "docs" / "_intersphinx"


def _local_inv(name: str) -> str | None:
    """Return path to a locally cached intersphinx inventory, or None to download."""
    path = _INTERSPHINX_DIR / f"{name}.inv"
    return str(path) if path.exists() else None


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", _local_inv("python")),
    "numpy": ("https://numpy.org/doc/stable", _local_inv("numpy")),
    "scipy": ("https://docs.scipy.org/doc/scipy/", _local_inv("scipy")),
    "matplotlib": ("https://matplotlib.org/stable", _local_inv("matplotlib")),
}

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_attr_annotations = True

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Optional heavy dependencies are mocked so the docs build without installing
# them. pantr imports them lazily at runtime, so autodoc can still import the
# modules. `mpi4py` / `pymetis` back `pantr.mpi`. `pyvista` is NOT mocked: the
# Sphinx-Gallery build executes the `demos/` scripts and needs it for real.
autodoc_mock_imports = ["mpi4py", "pymetis"]

# --- Sphinx-Gallery + PyVista (interactive demo gallery) --------------------
# Execute the standalone scripts in ``demos/`` and embed their output. PyVista
# renders off-screen; ``vtk-osmesa`` provides the GL stack on the headless CI /
# Read-the-Docs builders (see ``.readthedocs.yaml`` and the CI docs job).
# ``BUILDING_GALLERY`` makes each ``plotter.show()`` export both a screenshot
# (thumbnail) and a self-contained ``.vtksz`` scene; ``DynamicScraper`` embeds
# the latter as an interactive vtk.js widget (no server needed at view time).
pyvista.OFF_SCREEN = True
pyvista.BUILDING_GALLERY = True

sphinx_gallery_conf = {
    # Absolute path so the demo directory resolves the same regardless of the
    # build's working directory (relative paths can collide with sibling git
    # worktrees during sphinx-gallery's duplicate-filename check).
    "examples_dirs": str(PROJECT_ROOT / "demos"),
    "gallery_dirs": "auto_examples",
    # Only files whose name starts with a number (``01_…``) are executed.
    "filename_pattern": r"[/\\]\d+_",
    "image_scrapers": ("matplotlib", DynamicScraper()),
    "within_subsection_order": "FileNameSortKey",
    "remove_config_comments": True,
    "matplotlib_animations": False,
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "linkify",
    "smartquotes",
]

html_theme = "sphinx_rtd_theme"
try:
    if importlib.util.find_spec("sphinx_rtd_theme") is None:
        raise ImportError
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        "sphinx_rtd_theme not found. Falling back to 'alabaster'.",
        stacklevel=1,
    )
    html_theme = "alabaster"
html_static_path = ["_static"]
html_show_sourcelink = True

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "sticky_navigation": True,
}

html_context = {
    "display_github": True,
    "github_user": "pantolin",
    "github_repo": "pantr",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_logo = None
html_favicon = None

pygments_style = "default"
pygments_dark_style = "native"

version = pantr.__version__
release = pantr.__version__

nitpicky = True

# Private NumPy typing internals are not exposed in the public inventory.
# CellIndex, Target, and QIKind are Literal/Union type aliases; Sphinx resolves
# them as py:class but they have no class inventory entry.
nitpick_ignore = [
    ("py:class", "numpy._typing._array_like._Buffer"),
    ("py:class", "numpy._typing._array_like._SupportsArray"),
    ("py:class", "numpy._typing._dtype_like._DTypeDict"),
    ("py:class", "numpy._typing._dtype_like._SupportsDType"),
    ("py:class", "numpy._typing._nested_sequence._NestedSequence"),
    # Napoleon strips the npt. prefix from npt.ArrayLike / npt.DTypeLike
    ("py:class", "ArrayLike"),
    ("py:class", "DTypeLike"),
    # types.ModuleType return annotation: stripped to a bare name by autodoc and
    # not carried in the Python inventory under that short form.
    ("py:class", "ModuleType"),
    # pathlib.Path appears as a bare `Path` in pantr.viz.save's type annotation.
    ("py:class", "Path"),
    # Private structural protocol; not in __all__ and not cross-referenceable
    ("py:class", "_AffineMap"),
    ("py:class", "CellIndex"),
    ("py:class", "CellIndicesBatch"),
    ("py:class", "Target"),
    ("py:class", "QIKind"),
]

# Short-form NumPy aliases used in type annotations (np.*, npt.*, numpy.*)
# are not resolvable via intersphinx: np/npt are local import aliases, and
# even numpy.float32 etc. may be listed as py:data rather than py:class in
# numpy's inventory. Suppress the cross-reference lookup failures for all of
# them; canonical types like numpy.typing.NDArray are not used in the source.
nitpick_ignore_regex = [
    ("py:class", r"np\.\w+"),
    ("py:class", r"npt\.\w+"),
    ("py:class", r"numpy\.\w+"),
    # pantr.viz annotations reference pyvista via the local alias `pv`; pyvista
    # exposes no cross-referenceable inventory in this build.
    ("py:class", r"pv\.\w+"),
    ("py:class", r"pyvista\..*"),
]

suppress_warnings: list[str] = []

intersphinx_timeout = 15
