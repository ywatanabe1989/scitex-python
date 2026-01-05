#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/io/_bundle.py

"""
SciTeX .statsz Bundle I/O - Statistics-specific bundle operations.

Handles:
    - Statistical results specification validation
    - Comparison metadata management
    - P-value and effect size validation
"""

import json
from pathlib import Path
from typing import Any, Dict, List

__all__ = [
    "validate_statsz_spec",
    "load_statsz_bundle",
    "save_statsz_bundle",
    "STATSZ_SCHEMA_SPEC",
]

# Schema specification for .statsz bundles
STATSZ_SCHEMA_SPEC = {
    "name": "scitex.stats.stats",
    "version": "1.0.0",
    "required_fields": ["schema"],
    "optional_fields": ["comparisons", "metadata", "descriptive", "test_results"],
}


def validate_statsz_spec(spec: Dict[str, Any]) -> List[str]:
    """Validate .statsz-specific fields.

    Args:
        spec: The specification dictionary to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []

    if "comparisons" in spec:
        comparisons = spec["comparisons"]
        if not isinstance(comparisons, list):
            errors.append("'comparisons' must be a list")
        else:
            for i, comp in enumerate(comparisons):
                if not isinstance(comp, dict):
                    errors.append(f"comparisons[{i}] must be a dictionary")
                    continue

                # Validate p_value if present
                if "p_value" in comp:
                    p = comp["p_value"]
                    if not isinstance(p, (int, float)):
                        errors.append(f"comparisons[{i}].p_value must be numeric")
                    elif not (0 <= p <= 1):
                        errors.append(
                            f"comparisons[{i}].p_value must be between 0 and 1"
                        )

                # Validate effect_size if present
                if "effect_size" in comp:
                    es = comp["effect_size"]
                    if isinstance(es, dict):
                        if "value" in es and not isinstance(es["value"], (int, float)):
                            errors.append(
                                f"comparisons[{i}].effect_size.value must be numeric"
                            )
                    elif not isinstance(es, (int, float)):
                        errors.append(
                            f"comparisons[{i}].effect_size must be numeric or dict"
                        )

    # Validate test_results if present
    if "test_results" in spec:
        test_results = spec["test_results"]
        if not isinstance(test_results, (dict, list)):
            errors.append("'test_results' must be a dictionary or list")

    # Validate descriptive if present
    if "descriptive" in spec:
        descriptive = spec["descriptive"]
        if not isinstance(descriptive, dict):
            errors.append("'descriptive' must be a dictionary")

    return errors


def load_statsz_bundle(bundle_dir: Path) -> Dict[str, Any]:
    """Load .statsz bundle contents from directory.

    Args:
        bundle_dir: Path to the bundle directory.

    Returns:
        Dictionary with loaded bundle contents.
    """
    result = {}

    # Load specification
    spec_file = bundle_dir / "stats.json"
    if spec_file.exists():
        with open(spec_file, "r") as f:
            result["spec"] = json.load(f)
    else:
        result["spec"] = None

    # Load supplementary data files if present
    data_file = bundle_dir / "data.csv"
    if data_file.exists():
        try:
            import pandas as pd
            result["data"] = pd.read_csv(data_file)
        except ImportError:
            with open(data_file, "r") as f:
                result["data"] = f.read()

    return result


def save_statsz_bundle(data: Dict[str, Any], dir_path: Path) -> None:
    """Save .statsz bundle contents to directory.

    Args:
        data: Bundle data dictionary.
        dir_path: Path to the bundle directory.
    """
    # Save specification
    spec = data.get("spec", {})
    spec_file = dir_path / "stats.json"
    with open(spec_file, "w") as f:
        json.dump(spec, f, indent=2)

    # Save supplementary data if present
    if "data" in data:
        data_file = dir_path / "data.csv"
        df = data["data"]
        if hasattr(df, "to_csv"):
            df.to_csv(data_file, index=False)
        else:
            with open(data_file, "w") as f:
                f.write(str(df))

    # Save summary report if present
    if "report" in data:
        report_file = dir_path / "report.md"
        with open(report_file, "w") as f:
            f.write(data["report"])


# EOF
