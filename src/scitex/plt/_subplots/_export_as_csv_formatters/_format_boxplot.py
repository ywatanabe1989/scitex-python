#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_boxplot.py
# ----------------------------------------
import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np
import pandas as pd
import scitex


def _format_boxplot(id, tracked_dict, kwargs):
    """Format data from a boxplot call."""
    # Check if tracked_dict is empty or not a dictionary
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Get the args and kwargs from tracked_dict
    args = tracked_dict.get("args", [])
    call_kwargs = tracked_dict.get("kwargs", {})

    # Get labels if provided (for consistent naming with stats)
    labels = call_kwargs.get("labels", None)

    # Extract data if available
    if len(args) >= 1:
        x = args[0]

        # One box plot
        from scitex.types import is_listed_X as scitex_types_is_listed_X

        if isinstance(x, np.ndarray) or scitex_types_is_listed_X(x, [float, int]):
            df = pd.DataFrame(x)
            # Use label if single box and labels provided
            if labels and len(labels) == 1:
                df.columns = [f"{id}_{labels[0]}"]
            else:
                df.columns = [f"{id}_boxplot_0"]
        else:
            # Multiple boxes
            import scitex.pd

            df = scitex.pd.force_df({i_x: _x for i_x, _x in enumerate(x)})

            # Use labels if provided, otherwise use numeric indices
            if labels and len(labels) == len(df.columns):
                df.columns = [f"{id}_{label}" for label in labels]
            else:
                df.columns = [f"{id}_boxplot_{col}" for col in df.columns]

        df = df.apply(lambda col: col.dropna().reset_index(drop=True))
        return df

    # No valid data available
    return pd.DataFrame()
