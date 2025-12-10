#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scitex/plt/utils/metadata/_csv_verification.py

"""
CSV-JSON consistency verification utilities.

This module provides functions to verify that CSV columns match JSON metadata exactly.
"""

import json
import pandas as pd
from pathlib import Path


def assert_csv_json_consistency(csv_path: str, json_path: str = None) -> None:
    """
    Assert that CSV columns match JSON metadata exactly.

    Raises AssertionError if inconsistencies are found.

    Parameters
    ----------
    csv_path : str
        Path to CSV file
    json_path : str, optional
        Path to JSON file (defaults to csv_path with .json extension)
    """
    result = verify_csv_json_consistency(csv_path, json_path)
    if result["status"] != "consistent":
        raise AssertionError(f"CSV-JSON inconsistency: {result['message']}")


def verify_csv_json_consistency(csv_path: str, json_path: str = None) -> dict:
    """
    Verify CSV columns match JSON metadata.

    Parameters
    ----------
    csv_path : str
        Path to CSV file
    json_path : str, optional
        Path to JSON file (defaults to csv_path with .json extension)

    Returns
    -------
    dict
        {"status": "consistent"|"inconsistent"|"error", "message": str, "details": dict}
    """
    # Determine JSON path
    if json_path is None:
        json_path = str(Path(csv_path).with_suffix('.json'))

    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        csv_columns = set(df.columns)

        # Load JSON
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        # Extract column names from JSON metadata
        json_columns = set()
        if "data" in metadata and "columns_actual" in metadata["data"]:
            json_columns = set(metadata["data"]["columns_actual"])

        # Compare
        if csv_columns == json_columns:
            return {
                "status": "consistent",
                "message": "CSV columns match JSON metadata",
                "details": {"column_count": len(csv_columns)}
            }
        else:
            missing_in_json = csv_columns - json_columns
            missing_in_csv = json_columns - csv_columns
            return {
                "status": "inconsistent",
                "message": f"CSV-JSON mismatch: {len(missing_in_json)} columns in CSV not in JSON, "
                          f"{len(missing_in_csv)} columns in JSON not in CSV",
                "details": {
                    "csv_columns": sorted(csv_columns),
                    "json_columns": sorted(json_columns),
                    "missing_in_json": sorted(missing_in_json),
                    "missing_in_csv": sorted(missing_in_csv),
                }
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error verifying consistency: {e}",
            "details": {}
        }
