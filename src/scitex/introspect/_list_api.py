#!/usr/bin/env python3
# Timestamp: 2025-01-27
# File: /home/ywatanabe/proj/scitex-python/src/scitex/introspect/_list_api.py

"""Module API listing utilities."""

import importlib
import inspect
import sys
import warnings
from typing import Any, List, Optional, Set, Union

import pandas as pd


def list_api(
    module: Union[str, Any],
    columns: List[str] = ["Type", "Name", "Docstring", "Depth"],
    prefix: str = "",
    max_depth: int = 5,
    visited: Optional[Set[str]] = None,
    docstring: bool = False,
    tree: bool = True,
    current_depth: int = 0,
    print_output: bool = False,
    skip_depwarnings: bool = True,
    drop_duplicates: bool = True,
    root_only: bool = False,
) -> pd.DataFrame:
    """
    List the API of a module recursively and return as a DataFrame.

    Like a recursive `dir()` that shows the entire module tree.

    Example
    -------
    >>> df = list_api(scitex)
    >>> print(df)
       Type           Name                    Docstring  Depth
    0    M            scitex  Module description              0
    1    F  scitex.some_function  Function description        1
    2    C  scitex.SomeClass  Class description               1
    ...

    Parameters
    ----------
    module : Union[str, Any]
        Module to inspect (string name or actual module)
    columns : List[str]
        Columns to include in output DataFrame
    prefix : str
        Prefix for nested modules
    max_depth : int
        Maximum recursion depth
    visited : Optional[Set[str]]
        Set of visited modules to prevent cycles
    docstring : bool
        Whether to include docstrings
    tree : bool
        Whether to display tree structure
    current_depth : int
        Current recursion depth
    print_output : bool
        Whether to print results
    skip_depwarnings : bool
        Whether to skip DeprecationWarnings
    drop_duplicates : bool
        Whether to remove duplicate module entries
    root_only : bool
        Whether to show only root-level modules

    Returns
    -------
    pd.DataFrame
        Module structure with specified columns
    """
    return _list_api_impl(
        module=module,
        prefix=prefix,
        max_depth=max_depth,
        visited=visited,
        docstring=docstring,
        tree=tree,
        current_depth=current_depth,
        print_output=print_output,
        skip_depwarnings=skip_depwarnings,
        drop_duplicates=drop_duplicates,
        root_only=root_only,
    )[columns]


def _list_api_impl(
    module: Union[str, Any],
    columns: List[str] = ["Type", "Name", "Docstring", "Depth"],
    prefix: str = "",
    max_depth: int = 5,
    visited: Optional[Set[str]] = None,
    docstring: bool = False,
    tree: bool = True,
    current_depth: int = 0,
    print_output: bool = False,
    skip_depwarnings: bool = True,
    drop_duplicates: bool = True,
    root_only: bool = False,
) -> pd.DataFrame:
    """Internal implementation of list_api."""
    if skip_depwarnings:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    if isinstance(module, str):
        # Normalize hyphens to underscores for Python module names
        module_name = module.replace("-", "_")
        try:
            module = importlib.import_module(module_name)
        except ImportError as err:
            print(f"Error importing module {module_name}: {err}")
            return pd.DataFrame(columns=columns)

    if visited is None:
        visited = set()

    content_list = []

    try:
        module_name = getattr(module, "__name__", "")
        if max_depth < 0 or module_name in visited:
            return pd.DataFrame(content_list, columns=columns)

        visited.add(module_name)
        base_name = module_name.split(".")[-1]
        full_path = f"{prefix}.{base_name}" if prefix else base_name

        try:
            module_version = (
                f" (v{module.__version__})" if hasattr(module, "__version__") else ""
            )
            content_list.append(("M", full_path, module_version, current_depth))
        except Exception:
            pass

        for name, obj in inspect.getmembers(module):
            if name.startswith("_"):
                continue

            obj_name = f"{full_path}.{name}"

            if inspect.ismodule(obj):
                # Only recurse into direct submodules (not sibling packages)
                obj_mod_name = getattr(obj, "__name__", "")
                # Check if this is a direct submodule of current module
                if (
                    obj_mod_name.startswith(module_name + ".")
                    and obj_mod_name not in visited
                ):
                    content_list.append(
                        (
                            "M",
                            obj_name,
                            obj.__doc__ if docstring and obj.__doc__ else "",
                            current_depth + 1,  # Children are one level deeper
                        )
                    )
                    try:
                        sub_df = _list_api_impl(
                            obj,
                            columns=columns,
                            prefix=full_path,
                            max_depth=max_depth - 1,
                            visited=visited,
                            docstring=docstring,
                            tree=tree,
                            current_depth=current_depth + 1,
                            print_output=print_output,
                            skip_depwarnings=skip_depwarnings,
                            drop_duplicates=drop_duplicates,
                            root_only=root_only,
                        )
                        if sub_df is not None and not sub_df.empty:
                            content_list.extend(sub_df.values.tolist())
                    except Exception as err:
                        print(f"Error processing module {obj_name}: {err}")
            elif inspect.isfunction(obj):
                # Only include functions defined in this module (not re-exported from siblings)
                obj_module = getattr(obj, "__module__", "")
                # Check if function is defined in current module or its submodules
                if obj_module == module_name or obj_module.startswith(
                    module_name + "."
                ):
                    content_list.append(
                        (
                            "F",
                            obj_name,
                            obj.__doc__ if docstring and obj.__doc__ else "",
                            current_depth + 1,  # Children are one level deeper
                        )
                    )
            elif inspect.isclass(obj):
                # Only include classes defined in this module (not re-exported from siblings)
                obj_module = getattr(obj, "__module__", "")
                # Check if class is defined in current module or its submodules
                if obj_module == module_name or obj_module.startswith(
                    module_name + "."
                ):
                    content_list.append(
                        (
                            "C",
                            obj_name,
                            obj.__doc__ if docstring and obj.__doc__ else "",
                            current_depth + 1,  # Children are one level deeper
                        )
                    )

    except Exception as err:
        print(f"Error processing module structure: {err}")
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(content_list, columns=columns)

    if drop_duplicates:
        df = df.drop_duplicates(subset="Name", keep="first")

    if root_only:
        mask = df["Name"].str.count(r"\.") <= 1
        df = df[mask]

    if tree and current_depth == 0 and print_output:
        _print_module_contents(df)

    return df[columns]


def _print_module_contents(df: pd.DataFrame) -> None:
    """Prints module contents in tree structure.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing module structure
    """
    df_sorted = df.sort_values(["Depth", "Name"])
    depth_last = {}

    for index, row in df_sorted.iterrows():
        depth = row["Depth"]
        is_last = (
            index == len(df_sorted) - 1 or df_sorted.iloc[index + 1]["Depth"] <= depth
        )

        prefix = ""
        for d in range(depth):
            if d == depth - 1:
                prefix += "└── " if is_last else "├── "
            else:
                prefix += "    " if depth_last.get(d, False) else "│   "

        print(f"{prefix}({row['Type']}) {row['Name']}{row['Docstring']}")
        depth_last[depth] = is_last


if __name__ == "__main__":
    import scitex

    sys.setrecursionlimit(10_000)
    df = list_api(scitex, docstring=True, print_output=False, columns=["Name"])
    print(scitex.pd.round(df))
