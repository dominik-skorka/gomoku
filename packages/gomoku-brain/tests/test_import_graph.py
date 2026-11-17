"""Module-level imports inside ``gomoku_brain`` must form a DAG.

Import cycles used to be defused by load order (a back-edge placed after
the symbols it reaches for).  That is fragile, so this test pins the
invariant: no cycle among *module-scope* imports.  Imports inside
function bodies (lazy JIT / optional-dependency loads) are ignored.
"""

from __future__ import annotations

import ast
from pathlib import Path

import gomoku_brain

_PKG = Path(gomoku_brain.__file__).parent
_PREFIX = "gomoku_brain"


def _module_scope_imports(tree: ast.Module) -> set[str]:
    found: set[str] = set()

    def visit(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                if node.module == _PREFIX:
                    found.update(f"{_PREFIX}.{a.name}" for a in node.names)
                elif node.module.startswith(f"{_PREFIX}."):
                    found.add(node.module)
            elif isinstance(node, ast.Import):
                found.update(a.name for a in node.names if a.name.startswith(f"{_PREFIX}."))
            elif isinstance(node, ast.If | ast.Try | ast.With):
                # TYPE_CHECKING-only imports never execute at runtime.
                if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
                    continue
                for field in ("body", "orelse", "finalbody"):
                    visit(getattr(node, field, []))
                for handler in getattr(node, "handlers", []):
                    visit(handler.body)

    visit(tree.body)
    return found


def _graph() -> dict[str, set[str]]:
    modules = {f"{_PREFIX}.{p.stem}": p for p in _PKG.glob("*.py") if p.stem != "__init__"}
    graph: dict[str, set[str]] = {}
    for name, path in modules.items():
        deps = _module_scope_imports(ast.parse(path.read_text()))
        graph[name] = {d for d in deps if d in modules and d != name}
    return graph


def _find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    state: dict[str, int] = {}  # 1 = in progress, 2 = done
    stack: list[str] = []

    def dfs(node: str) -> list[str] | None:
        state[node] = 1
        stack.append(node)
        for dep in sorted(graph[node]):
            if state.get(dep) == 1:
                return [*stack[stack.index(dep) :], dep]
            if dep not in state and (cycle := dfs(dep)) is not None:
                return cycle
        stack.pop()
        state[node] = 2
        return None

    for node in sorted(graph):
        if node not in state and (cycle := dfs(node)) is not None:
            return cycle
    return None


def test_module_scope_imports_are_acyclic():
    cycle = _find_cycle(_graph())
    assert cycle is None, "import cycle: " + " -> ".join(cycle)
