#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 12:20:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_step.py

import numpy as np
import pandas as pd


def _format_step(id, tracked_dict, kwargs):
    """Format data from a step plot call.

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to step

    Returns:
        pd.DataFrame: Formatted data from step plot
    """
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    if 'args' in tracked_dict:
        args = tracked_dict['args']
        if isinstance(args, tuple) and len(args) > 0:
            if len(args) == 1:
                y = np.asarray(args[0])
                x = np.arange(len(y))
            elif len(args) >= 2:
                x = np.asarray(args[0])
                y = np.asarray(args[1])
            else:
                return pd.DataFrame()

            df = pd.DataFrame({f"{id}_step_x": x, f"{id}_step_y": y})
            return df

    return pd.DataFrame()


# EOF
