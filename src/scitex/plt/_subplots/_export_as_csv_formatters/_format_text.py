#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-10 12:00:00 (ywatanabe)"
# File: ./src/scitex/plt/_subplots/_export_as_csv_formatters/_format_text.py

"""CSV formatter for text() calls - uses standard column naming."""

from __future__ import annotations

import pandas as pd

from scitex.plt.utils._csv_column_naming import get_csv_column_name

from ._format_plot import _parse_tracking_id


def _format_text(id, tracked_dict, kwargs):
    """Format data from a text call.

    Uses standard column naming convention:
    (ax-row-{r}-col-{c}_trace-id-{id}_variable-{var}).

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to text

    Returns:
        pd.DataFrame: Formatted text position data
    """
    # Check if tracked_dict is empty or not a dictionary
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Parse tracking ID to get axes position
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    # Get standard column names
    x_col = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
    y_col = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
    content_col = get_csv_column_name("content", ax_row, ax_col, trace_id=trace_id)

    # Get the args from tracked_dict
    args = tracked_dict.get("args", [])

    # Extract x, y, and text content if available
    if len(args) >= 2:
        x, y = args[0], args[1]
        text_content = args[2] if len(args) >= 3 else None

        data = {x_col: [x], y_col: [y]}

        if text_content is not None:
            data[content_col] = [text_content]

        return pd.DataFrame(data)

    return pd.DataFrame()


# EOF
