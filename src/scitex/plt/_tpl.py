#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-03-31 11:58:28 (ywatanabe)"

import numpy as np
import termplotlib as tpl


def termplot(*args):
    """
    Plots given y values against x using termplotlib, or plots a single y array against its indices if x is not provided.

    Parameters:
    - *args: Accepts either one argument (y values) or two+ arguments (x and y values, extras ignored).

    Returns:
    None. Displays the plot in the terminal.
    """
    if len(args) == 1:
        y = args[0]
        x = np.arange(len(y))
    elif len(args) >= 2:
        x, y = args[0], args[1]
    else:
        raise ValueError("termplot requires at least one argument (y values)")

    fig = tpl.figure()
    fig.plot(x, y)
    fig.show()
