#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-21 13:14:04 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/mpl/get_dir_ax.py


"""Top-level docstring here"""

# Imports
import scitex as stx

# # Parameters
# CONFIG = stx.io.load_configs() # For imported files using `./config/*.yaml`


# Functions and Classes
@stx.session
def main(
    # arg1,
    # kwarg1="value1",
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    rng_manager=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Help message for `$ python __file__ --help`"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    dir_ax_list = dir(ax)
    dir_ax_str = "\n".join(dir_ax_list)

    stx.io.save(
        dir_ax_str,
        "./dir_ax.txt",
        symlink_to="./data/dev/plt/mpl",
    )

    return 0


if __name__ == "__main__":
    main()

# EOF
