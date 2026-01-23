#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/introspect/_examples.py

"""Usage example finding utilities."""

from __future__ import annotations

import importlib
from pathlib import Path

from ._resolve import resolve_object


def find_examples(
    dotted_path: str,
    search_paths: list[str] | None = None,
    max_results: int = 10,
) -> dict:
    """
    Find usage examples of a function/class in tests and examples.

    Parameters
    ----------
    dotted_path : str
        Dotted path to search for (e.g., 'scitex.plt.plot')
    search_paths : list[str] | None
        Paths to search. If None, auto-detect from module location
    max_results : int
        Maximum number of examples to return

    Returns
    -------
    dict
        examples: list[dict] - Each with file, line, context
        count: int
    """
    obj, error = resolve_object(dotted_path)
    if error:
        return {"success": False, "error": error}

    name = getattr(obj, "__name__", dotted_path.split(".")[-1])

    if search_paths is None:
        search_paths = []

        module = dotted_path.split(".")[0]
        try:
            mod = importlib.import_module(module)
            if hasattr(mod, "__file__") and mod.__file__:
                mod_dir = Path(mod.__file__).parent
                project_root = mod_dir.parent
                for subdir in ["tests", "test", "examples", "example"]:
                    test_dir = project_root / subdir
                    if test_dir.exists():
                        search_paths.append(str(test_dir))
        except ImportError:
            pass

    if not search_paths:
        return {
            "success": True,
            "examples": [],
            "count": 0,
            "message": "No test/example directories found",
        }

    examples = []

    for search_path in search_paths:
        path = Path(search_path)
        if not path.exists():
            continue

        for py_file in path.rglob("*.py"):
            try:
                content = py_file.read_text()
            except Exception:
                continue

            if name not in content:
                continue

            lines = content.split("\n")
            for i, line in enumerate(lines):
                if name in line:
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    context = "\n".join(lines[start:end])

                    examples.append(
                        {
                            "file": str(py_file),
                            "line": i + 1,
                            "context": context,
                        }
                    )

                    if len(examples) >= max_results:
                        break

            if len(examples) >= max_results:
                break

        if len(examples) >= max_results:
            break

    return {
        "success": True,
        "examples": examples,
        "count": len(examples),
        "search_paths": search_paths,
    }
