#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 12:40:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_imshow.py

import numpy as np
import pandas as pd


def _format_imshow(id, tracked_dict, kwargs):
    """Format data from an imshow call.

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to imshow

    Returns:
        pd.DataFrame: Formatted data from imshow (flattened image with row, col indices)
    """
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Check for pre-formatted image_df (from plot_imshow wrapper)
    if tracked_dict.get("image_df") is not None:
        return tracked_dict.get("image_df")

    # Handle raw args from __getattr__ proxied calls
    if 'args' in tracked_dict:
        args = tracked_dict['args']
        if isinstance(args, tuple) and len(args) > 0:
            img = np.asarray(args[0])

            # Handle 2D grayscale image
            if img.ndim == 2:
                rows, cols = img.shape
                row_indices, col_indices = np.meshgrid(
                    range(rows), range(cols), indexing="ij"
                )

                df = pd.DataFrame(
                    {
                        f"{id}_imshow_row": row_indices.flatten(),
                        f"{id}_imshow_col": col_indices.flatten(),
                        f"{id}_imshow_value": img.flatten(),
                    }
                )
                return df

            # Handle RGB/RGBA images (3D array)
            elif img.ndim == 3:
                rows, cols, channels = img.shape
                row_indices, col_indices = np.meshgrid(
                    range(rows), range(cols), indexing="ij"
                )

                data = {
                    f"{id}_imshow_row": row_indices.flatten(),
                    f"{id}_imshow_col": col_indices.flatten(),
                }

                # Add channel data (R, G, B, A)
                channel_names = ['R', 'G', 'B', 'A'][:channels]
                for c, name in enumerate(channel_names):
                    data[f"{id}_imshow_{name}"] = img[:, :, c].flatten()

                return pd.DataFrame(data)

    return pd.DataFrame()