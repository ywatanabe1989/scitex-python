#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_metadata/_verification.py

"""
CSV/JSON consistency verification for figure metadata.

Provides functions to verify that exported CSV data matches
the JSON metadata declarations.
"""



def assert_csv_json_consistency(csv_path: str, json_path: str = None) -> None:
    """
    Assert that CSV data file and its JSON metadata are consistent.

    Raises AssertionError if the column names don't match.

    Parameters
    ----------
    csv_path : str
        Path to the CSV data file
    json_path : str, optional
        Path to the JSON metadata file. If not provided, assumes
        the JSON is at the same location with .json extension.

    Raises
    ------
    AssertionError
        If CSV and JSON column names don't match
    FileNotFoundError
        If CSV or JSON files don't exist

    Examples
    --------
    >>> assert_csv_json_consistency('/tmp/plot.csv')  # Passes silently if valid
    >>> # Or use in tests:
    >>> try:
    ...     assert_csv_json_consistency('/tmp/plot.csv')
    ... except AssertionError as e:
    ...     print(f"Validation failed: {e}")
    """
    result = verify_csv_json_consistency(csv_path, json_path)

    if result["errors"]:
        raise FileNotFoundError("\n".join(result["errors"]))

    if not result["valid"]:
        msg_parts = ["CSV/JSON consistency check failed:"]
        if result["missing_in_csv"]:
            msg_parts.append(
                f"  columns_actual missing in CSV: {result['missing_in_csv']}"
            )
        if result["extra_in_csv"]:
            msg_parts.append(f"  Extra columns in CSV: {result['extra_in_csv']}")
        if result.get("data_ref_missing"):
            msg_parts.append(
                f"  data_ref columns missing in CSV: {result['data_ref_missing']}"
            )
        raise AssertionError("\n".join(msg_parts))


def verify_csv_json_consistency(csv_path: str, json_path: str = None) -> dict:
    """
    Verify consistency between CSV data file and its JSON metadata.

    This function checks that:
    1. Column names in the CSV file match those declared in JSON's columns_actual
    2. Artist data_ref values in JSON match actual CSV column names

    Parameters
    ----------
    csv_path : str
        Path to the CSV data file
    json_path : str, optional
        Path to the JSON metadata file. If not provided, assumes
        the JSON is at the same location with .json extension.

    Returns
    -------
    dict
        Verification result with keys:
        - 'valid': bool - True if CSV and JSON are consistent
        - 'csv_columns': list - Column names found in CSV
        - 'json_columns': list - Column names declared in JSON
        - 'data_ref_columns': list - Column names from artist data_ref
        - 'missing_in_csv': list - Columns in JSON but not in CSV
        - 'extra_in_csv': list - Columns in CSV but not in JSON
        - 'data_ref_missing': list - data_ref columns not found in CSV
        - 'errors': list - Any error messages

    Examples
    --------
    >>> result = verify_csv_json_consistency('/tmp/plot.csv')
    >>> print(result['valid'])
    True
    >>> print(result['missing_in_csv'])
    []
    """
    import json
    import os

    import pandas as pd

    result = {
        "valid": False,
        "csv_columns": [],
        "json_columns": [],
        "data_ref_columns": [],
        "missing_in_csv": [],
        "extra_in_csv": [],
        "data_ref_missing": [],
        "errors": [],
    }

    # Determine JSON path
    if json_path is None:
        base, _ = os.path.splitext(csv_path)
        json_path = base + ".json"

    # Check files exist
    if not os.path.exists(csv_path):
        result["errors"].append(f"CSV file not found: {csv_path}")
        return result
    if not os.path.exists(json_path):
        result["errors"].append(f"JSON file not found: {json_path}")
        return result

    try:
        # Read CSV columns
        df = pd.read_csv(csv_path, nrows=0)  # Just read header
        csv_columns = list(df.columns)
        result["csv_columns"] = csv_columns
    except Exception as e:
        result["errors"].append(f"Error reading CSV: {e}")
        return result

    try:
        # Read JSON metadata
        with open(json_path) as f:
            metadata = json.load(f)

        # Get columns_actual from data section
        json_columns = []
        if "data" in metadata and "columns_actual" in metadata["data"]:
            json_columns = metadata["data"]["columns_actual"]
        result["json_columns"] = json_columns

        # Extract data_ref columns from artists
        data_ref_columns = _extract_data_ref_columns(metadata)
        result["data_ref_columns"] = data_ref_columns

    except Exception as e:
        result["errors"].append(f"Error reading JSON: {e}")
        return result

    # Compare columns_actual with CSV
    csv_set = set(csv_columns)
    json_set = set(json_columns)

    result["missing_in_csv"] = list(json_set - csv_set)
    result["extra_in_csv"] = list(csv_set - json_set)

    # Check data_ref columns exist in CSV (if there are any)
    if data_ref_columns:
        data_ref_set = set(data_ref_columns)
        result["data_ref_missing"] = list(data_ref_set - csv_set)

    # Valid only if columns_actual matches AND data_ref columns are found
    result["valid"] = (
        len(result["missing_in_csv"]) == 0
        and len(result["extra_in_csv"]) == 0
        and len(result["data_ref_missing"]) == 0
    )

    return result


def _extract_data_ref_columns(metadata: dict) -> list:
    """
    Extract data_ref column names from metadata.

    Skip 'derived_from' key as it contains descriptive text, not CSV column names.
    Also skip 'row_index' as it's a numeric index, not a column name.
    """
    data_ref_columns = []
    skip_keys = {"derived_from", "row_index"}

    if "axes" in metadata:
        for ax_key, ax_data in metadata["axes"].items():
            if "artists" in ax_data:
                for artist in ax_data["artists"]:
                    if "data_ref" in artist:
                        for key, val in artist["data_ref"].items():
                            if key not in skip_keys and isinstance(val, str):
                                data_ref_columns.append(val)

    return data_ref_columns


# EOF
