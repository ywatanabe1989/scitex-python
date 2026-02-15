#!/usr/bin/env python3
"""CSV/DataFrame column resolution for seaborn-style data= parameter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

import scitex as stx


def resolve_columns(
    data: Union[pd.DataFrame, str, Path],
    **col_map: Any,
) -> Dict[str, np.ndarray]:
    """Resolve string column names from a DataFrame or CSV to arrays.

    Parameters
    ----------
    data : DataFrame, str, or Path
        DataFrame or path to CSV file.
    **col_map
        Keyword arguments mapping param names to values.
        String values are treated as column names and resolved.
        Non-string values are passed through unchanged.

    Returns
    -------
    dict
        Resolved arrays keyed by parameter name.

    Raises
    ------
    ValueError
        If a string column name is not found in the DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    >>> resolved = resolve_columns(df, x="a", y="b")
    >>> resolved["x"]
    array([1, 2, 3])
    """
    if isinstance(data, (str, Path)):
        data = stx.io.load(str(data))

    result = {}
    for key, val in col_map.items():
        if isinstance(val, str):
            if val not in data.columns:
                raise ValueError(
                    f"Column '{val}' not found in data. "
                    f"Available columns: {list(data.columns)}"
                )
            result[key] = data[val].dropna().values
        else:
            result[key] = val
    return result


def resolve_groups(
    data: Union[pd.DataFrame, str, Path],
    value_col: str,
    group_col: str,
) -> Tuple[List[np.ndarray], List[str]]:
    """Split a DataFrame into groups by a column.

    Parameters
    ----------
    data : DataFrame, str, or Path
        DataFrame or path to CSV file.
    value_col : str
        Column containing measurement values.
    group_col : str
        Column containing group labels.

    Returns
    -------
    groups : list of np.ndarray
        One array per group, NaN values dropped.
    names : list of str
        Group names in sorted order.

    Raises
    ------
    ValueError
        If value_col or group_col is not found in the DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({"score": [1, 2, 3, 4], "group": ["A", "A", "B", "B"]})
    >>> groups, names = resolve_groups(df, "score", "group")
    >>> names
    ['A', 'B']
    """
    if isinstance(data, (str, Path)):
        data = stx.io.load(str(data))

    for col_name, col_label in [
        (value_col, "value_col"),
        (group_col, "group_col"),
    ]:
        if col_name not in data.columns:
            raise ValueError(
                f"{col_label} '{col_name}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )

    groups: List[np.ndarray] = []
    names: List[str] = []
    for name, group in data.groupby(group_col):
        groups.append(group[value_col].dropna().values)
        names.append(str(name))
    return groups, names
