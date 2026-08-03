#!/usr/bin/env python3
"""Validate Python syntax and resolvable in-repo imports (pre-commit / pre-push)."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "site-packages",
        "dist",
        "build",
    }
)
SKIP_FILE_PREFIXES = ("._",)


def repo_root() -> Path:
    path = Path.cwd().resolve()
    for parent in [path, *path.parents]:
        if (parent / ".git").is_dir():
            return parent
    return path


def should_skip(path: Path) -> bool:
    if path.name.startswith(SKIP_FILE_PREFIXES):
        return True
    if any(part in SKIP_DIR_NAMES for part in path.parts):
        return True
    if path.name.startswith("test_") and "tests" in path.parts:
        return True
    return False


def iter_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if should_skip(path):
            continue
        if path.stat().st_size < 10:
            continue
        files.append(path)
    return sorted(files)


def module_paths(root: Path) -> dict[str, Path]:
    """Map dotted module names to files/packages under repo root."""
    mapping: dict[str, Path] = {}
    for path in iter_python_files(root):
        rel = path.relative_to(root)
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]
        if not parts:
            continue
        mapping[".".join(parts)] = path
    return mapping


def resolve_module(module: str | None, level: int, package: str | None, root: Path) -> str | None:
    if level:
        if not package:
            return None
        pkg_parts = package.split(".")
        if level > len(pkg_parts):
            return None
        base = pkg_parts[: len(pkg_parts) - level + 1]
        if module:
            return ".".join(base + module.split("."))
        return ".".join(base)
    return module


def package_for_file(path: Path, root: Path) -> str | None:
    rel = path.relative_to(root)
    parts = list(rel.parts)
    if parts[-1] != "__init__.py":
        parts[-1] = parts[-1][:-3]
    else:
        parts = parts[:-1]
    if not parts:
        return None
    return ".".join(parts)


def check_syntax(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"{path}: read failed: {exc}"]
    try:
        compile(source, str(path), "exec", dont_inherit=True, optimize=0)
        ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        loc = f"line {exc.lineno}" if exc.lineno else "unknown line"
        errors.append(f"{path}: syntax error ({loc}): {exc.msg}")
    return errors


def check_imports(path: Path, root: Path, modules: dict[str, Path]) -> list[str]:
    errors: list[str] = []
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return errors

    local_tops = {name.split(".")[0] for name in modules}
    package = package_for_file(path, root)

    def module_exists(name: str) -> bool:
        if name in modules:
            return True
        return any(mod == name or mod.startswith(name + ".") for mod in modules)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            candidates = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            resolved = resolve_module(node.module, node.level or 0, package, root)
            if not resolved:
                if node.level:
                    errors.append(
                        f"{path}:{node.lineno}: unresolved relative import "
                        f"from {'.' * (node.level or 0)}{node.module or ''}"
                    )
                continue
            candidates = [resolved]
        else:
            continue

        for name in candidates:
            top = name.split(".")[0]
            if top not in local_tops:
                continue
            if not module_exists(name):
                errors.append(f"{path}:{node.lineno}: unresolved local import {name!r}")
    return errors


def validate(paths: list[Path], root: Path) -> list[str]:
    modules = module_paths(root)
    errors: list[str] = []
    for path in paths:
        if should_skip(path):
            continue
        if not path.is_file():
            continue
        try:
            path.relative_to(root)
        except ValueError:
            continue
        errors.extend(check_syntax(path))
        errors.extend(check_imports(path, root, modules))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        help="Python files to validate (default: all under repo when --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate every Python file in the git repo (pre-push)",
    )
    args = parser.parse_args()

    root = repo_root()
    if args.all:
        paths = iter_python_files(root)
    elif args.files:
        paths = [Path(f).resolve() for f in args.files]
    else:
        paths = iter_python_files(root)

    errors = validate(paths, root)
    if errors:
        print("Python validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
