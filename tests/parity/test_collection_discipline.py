"""Guard that a missing extension skips these tests rather than failing collection.

The parity suite is meant to run in an installation with no compiled extension -- that is
the common local configuration, and the Python half of every per-backend property still
has something to say there. The mechanism is
:func:`tests._parity_harness.demand_cpp_backend`, reached through the ``cpp_backend``
fixture, which skips or (under ``PANTR_REQUIRE_CPP``) fails per test.

A module-level ``from pantr import _pantr_cpp`` defeats it before any of that runs: the
import raises while pytest is *collecting*, which is reported as an error against the whole
session, so a run that should have skipped a handful of tests instead reports nothing about
any of the rest. ``test_extraction_kernels.py`` carried one, and the same defect had already
been fixed once in another parity module -- which is the reason this is a guard over the
directory and not a note in one file.

Parsed rather than grepped: the point of the fix is that such an import belongs *inside* a
function body, and only the syntax tree tells those two apart.

What counts as "not inside a function" is the whole subtlety, and a first version of this
guard got it wrong: a ``class`` body and an ``if`` or ``try`` block all execute while the
module is being exec'd -- which is collection time, the very moment in question -- and only
a function body is deferred. Three decoys built to check it (``if True:``, ``try/except
ImportError``, and a class-body import) all escaped a check that looked at top-level
statements alone. So the rule enforced here is the positive one the parity modules already
follow: the import lives **inside a function**, reached through a helper behind the
``cpp_backend`` fixture. A module-level ``try``/``except ImportError`` is flagged too. It
would not break collection, but it is not the convention either, and what it leads to is a
``_pantr_cpp = None`` sentinel tested ad hoc in place of the one fixture that decides
skip-or-fail.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_PARITY_DIR = pathlib.Path(__file__).resolve().parent

_EXTENSION = "_pantr_cpp"


def _imports_extension(node: ast.AST) -> bool:
    """Report whether a statement imports the compiled extension.

    Args:
        node (ast.AST): The statement to inspect.

    Returns:
        bool: True for an ``import``/``from`` statement naming the extension.
    """
    if not isinstance(node, ast.Import | ast.ImportFrom):
        return False
    module = getattr(node, "module", None) or ""
    return _EXTENSION in module or any(_EXTENSION in alias.name for alias in node.names)


def _extension_imports_outside_a_function(source: str) -> list[int]:
    """Find imports of the compiled extension that run while the module is exec'd.

    Recurses into every block that executes at import time -- ``if``, ``try``, ``with``,
    ``for``, ``while``, and a ``class`` body -- and stops at a function, whose body runs
    only when it is called.

    Args:
        source (str): The module's source text.

    Returns:
        list[int]: The line number of each offending import, in file order.
    """
    offending: list[int] = []

    def walk(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if _imports_extension(node):
                offending.append(node.lineno)
                continue
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    walk([child])
                elif isinstance(child, ast.ExceptHandler):
                    walk(child.body)

    walk(ast.parse(source).body)
    return sorted(offending)


def _parity_modules() -> list[pathlib.Path]:
    """Collect the parity test modules this rule applies to.

    Returns:
        list[pathlib.Path]: Every ``.py`` file in the parity directory, sorted.
    """
    return sorted(_PARITY_DIR.glob("*.py"))


def test_the_guard_has_modules_to_guard() -> None:
    # A directory glob that stops matching would leave this file green while checking
    # nothing at all, which is the failure mode of every discovered parametrization.
    modules = _parity_modules()
    assert len(modules) > 1, f"only {len(modules)} module(s) found under {_PARITY_DIR.name}/"


@pytest.mark.parametrize("module", _parity_modules(), ids=lambda path: path.stem)
def test_the_extension_is_not_imported_outside_a_function(module: pathlib.Path) -> None:
    offending = _extension_imports_outside_a_function(module.read_text(encoding="utf-8"))
    assert not offending, (
        f"{module.name} imports the extension where it runs at import time (line(s) "
        f"{', '.join(str(line) for line in offending)}); an installation without it then "
        f"fails collection instead of skipping. Import it inside a function, behind the "
        f"cpp_backend fixture."
    )
