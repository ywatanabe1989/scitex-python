#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 12:20:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_hexbin.py

import numpy as np
import pandas as pd


def _format_hexbin(id, tracked_dict, kwargs):
    """Format data from a hexbin call.

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to hexbin

    Returns:
        pd.DataFrame: Formatted data from hexbin (input x, y data)
    """
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    if 'args' in tracked_dict:
        args = tracked_dict['args']
        if isinstance(args, tuple) and len(args) >= 2:
            x = np.asarray(args[0]).flatten()
            y = np.asarray(args[1]).flatten()

            # Ensure same length
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]

            df = pd.DataFrame({f"{id}_hexbin_x": x, f"{id}_hexbin_y": y})
            return df

    return pd.DataFrame()


# EOF
