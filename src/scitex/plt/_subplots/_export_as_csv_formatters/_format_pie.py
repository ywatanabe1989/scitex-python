#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 12:20:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_pie.py

import numpy as np
import pandas as pd


def _format_pie(id, tracked_dict, kwargs):
    """Format data from a pie chart call.

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to pie

    Returns:
        pd.DataFrame: Formatted data from pie chart
    """
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    if 'args' in tracked_dict:
        args = tracked_dict['args']
        if isinstance(args, tuple) and len(args) > 0:
            x = np.asarray(args[0])

            data = {f"{id}_pie_values": x}

            # Add labels if provided
            labels = kwargs.get('labels', None)
            if labels is not None:
                data[f"{id}_pie_labels"] = labels

            df = pd.DataFrame(data)
            return df

    return pd.DataFrame()


# EOF
