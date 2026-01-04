#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-10 23:48:44 (ywatanabe)"

"""Analyze scitex module dependencies."""

import ast
from pathlib import Path
from collections import defaultdict


def find_scitex_imports(file_path):
    """Extract scitex.* imports from a Python file."""
    try:
        with open(file_path) as f:
            tree = ast.parse(f.read())
    except:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("scitex."):
                # Extract the submodule (e.g., 'logging' from 'scitex.logging')
                parts = node.module.split(".")
                if len(parts) >= 2:
                    imports.append(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("scitex."):
                    parts = alias.name.split(".")
                    if len(parts) >= 2:
                        imports.append(parts[1])

    return imports


def analyze_module_dependencies(scitex_root):
    """Analyze dependencies for each scitex submodule."""
    scitex_path = Path(scitex_root)
    dependencies = defaultdict(set)

    # Iterate through each submodule directory
    for module_dir in scitex_path.iterdir():
        if not module_dir.is_dir() or module_dir.name.startswith("_"):
            continue

        module_name = module_dir.name

        # Find all Python files in this module
        for py_file in module_dir.rglob("*.py"):
            imports = find_scitex_imports(py_file)
            for imp in imports:
                if imp != module_name:  # Don't count self-imports
                    dependencies[module_name].add(imp)

    return dependencies


if __name__ == "__main__":
    scitex_root = "/home/ywatanabe/proj/scitex-code/src/scitex"
    deps = analyze_module_dependencies(scitex_root)

    print("Module Dependencies:")
    print("=" * 50)
    for module, imports in sorted(deps.items()):
        print(f"\n{module}:")
        for imp in sorted(imports):
            print(f"  - scitex-{imp}")

# EOF
