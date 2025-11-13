#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-14 09:11:20 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_session_plt_io.py


"""Minimal Demonstration for scitex.{session,io,plt}"""

import numpy as np
import scitex as stx


def demo_without_qr(filename, verbose=False):
    """Show metadata without QR code (just embedded)."""

    # matplotlib.pyplot wrapper.
    fig, ax = stx.plt.subplots()

    t = np.linspace(0, 2, 1000)
    signal = np.sin(2 * np.pi * 5 * t) * np.exp(-t / 2)

    ax.plot_line(t, signal)  # Original plot for automatic CSV export
    ax.set_xyt(
        "Time (s)",
        "Amplitude",
        "Clean Figure (metadata embedded, no QR overlay)",
    )

    # Saving: stx.io.save(obj, rel_path, **kwargs)
    stx.io.save(
        fig,
        filename,
        metadata={"exp": "s01", "subj": "S001"},  # with meatadata embedding
        symlink_to="./data",  # Symlink for centralized outputs
        verbose=verbose,  # Automatic terminal logging (no manual print())
    )
    fig.close()

    # Loading: stx.io.load(path)
    ldir = __file__.replace(".py", "_out")
    img, meta = stx.io.load(
        f"{ldir}/{filename}",
        verbose=verbose,
    )


@stx.session.session
def main(filename="demo_fig_with_metadata.jpg", verbose=True):
    """Run all demos."""

    demo_without_qr(filename, verbose=verbose)

    return 0


if __name__ == "__main__":
    main()

# (.env-3.11) (wsl) scitex-code $ ./examples/demo_session_plt_io.py -h
# usage: demo_session_plt_io.py [-h] [--filename FILENAME] [--verbose VERBOSE]

# Run all demos.

# options:
#   -h, --help           show this help message and exit
#   --filename FILENAME  (default: demo_fig_with_metadata.jpg)
#   --verbose VERBOSE    (default: True)
# (.env-3.11) (wsl) scitex-code $ ./examples/demo_session_plt_io.py

# ========================================
# SciTeX v2.1.3
# 2025Y-11M-14D-08h56m55s_JDUS (PID: 2374042)

# /home/ywatanabe/proj/scitex-code/./examples/demo_session_plt_io.py

# Arguments:
#     filename: demo_fig_with_metadata.jpg
#     verbose: True
# ========================================

# INFO: Running main with args: {'filename': 'demo_fig_with_metadata.jpg', 'verbose': True}
# (.env-3.11) (wsl) scitex-code $ ./examples/demo_session_plt_io.py -h
# usage: demo_session_plt_io.py [-h] [--filename FILENAME] [--verbose VERBOSE]

# Run all demos.

# options:
#   -h, --help           show this help message and exit
#   --filename FILENAME  (default: demo_fig_with_metadata.jpg)
#   --verbose VERBOSE    (default: True)
# (.env-3.11) (wsl) scitex-code $ ./examples/demo_session_plt_io.py

# ========================================
# SciTeX v2.1.3
# 2025Y-11M-14D-09h08m33s_2DKi (PID: 2396675)

# /home/ywatanabe/proj/scitex-code/./examples/demo_session_plt_io.py

# Arguments:
#     filename: demo_fig_with_metadata.jpg
#     verbose: True
# ========================================

# INFO: Running main with args: {'filename': 'demo_fig_with_metadata.jpg', 'verbose': True}
# INFO: 📝 Saving figure with metadata to: /home/ywatanabe/proj/scitex-code/examples/demo_session_plt_io_out/demo_fig_with_metadata.jpg
# INFO:   • Auto-added URL: https://scitex.ai
# INFO:   • Embedded metadata: {'exp': 's01', 'subj': 'S001', 'url': 'https://scitex.ai'}
# SUCC: Saved to: ./examples/demo_session_plt_io_out/demo_fig_with_metadata.csv (38.0 KiB)
# SUCC: Symlinked: /home/ywatanabe/proj/scitex-code/examples/demo_session_plt_io_out/demo_fig_with_metadata.csv -> /home/ywatanabe/proj/scitex-code/examples/demo_session_plt_io_out/demo_fig_with_metadata.csv/demo_fig_with_metadata.csv
# SUCC: Saved to: ./examples/demo_session_plt_io_out/demo_fig_with_metadata.jpg (241.6 KiB)
# SUCC: Symlinked: /home/ywatanabe/proj/scitex-code/examples/demo_session_plt_io_out/demo_fig_with_metadata.jpg -> /home/ywatanabe/proj/scitex-code/data/demo_fig_with_metadata.jpg
# INFO: ✅ Loading image with metadata from: /home/ywatanabe/proj/scitex-code/./examples/demo_session_plt_io_out/demo_fig_with_metadata.jpg
# INFO:   • Embedded metadata found:
# INFO:     - exp: s01
# INFO:     - subj: S001
# INFO:     - url: https://scitex.ai

# SUCC: Congratulations! The script completed: /home/ywatanabe/proj/scitex-code/examples/demo_session_plt_io_out/FINISHED_SUCCESS/2025Y-11M-14D-09h08m33s_2DKi-main/
# (.env-3.11) (wsl) scitex-code $ ls -al ./data/demo_fig_with_metadata.jpg
# lrwxrwxrwx 1 ywatanabe ywatanabe 62 Nov 14 09:08 ./data/demo_fig_with_metadata.jpg -> ../examples/demo_session_plt_io_out/demo_fig_with_metadata.jpg
# (.env-3.11) (wsl) scitex-code $ tree ./examples/demo_session_plt_io*
# ./examples/demo_session_plt_io_out
# ├── demo_fig_with_metadata.csv
# ├── demo_fig_with_metadata.jpg
# ├── FINISHED_SUCCESS
# │   ├── 2025Y-11M-14D-09h07m28s_j5gY-main
# │   │   ├── CONFIGS
# │   │   │   ├── CONFIG.pkl
# │   │   │   └── CONFIG.yaml
# │   │   └── logs
# │   │       ├── stderr.log
# │   │       └── stdout.log
# │   └── 2025Y-11M-14D-09h08m33s_2DKi-main
# │       ├── CONFIGS
# │       │   ├── CONFIG.pkl
# │       │   └── CONFIG.yaml
# │       └── logs
# │           ├── stderr.log
# │           └── stdout.log
# └── RUNNING
#     └── 2025Y-11M-14D-08h56m37s_jPQu-main
#         └── logs
#             ├── stderr.log
#             └── stdout.log
# ./examples/demo_session_plt_io.py  [error opening dir]

# 11 directories, 13 files
# (.env-3.11) (wsl) scitex-code $ head ./examples/demo_session_plt_io_out/demo_fig_with_metadata.csv
# ax_00_plot_line_0_line_x,ax_00_plot_line_0_line_y
# 0.0,0.0
# 0.06279040531731951,0.002002002002002002
# 0.12520711420365782,0.004004004004004004
# 0.18700423504710992,0.006006006006006006
# 0.24793881631482095,0.008008008008008008
# 0.30777180069339666,0.01001001001001001
# 0.3662689619330633,0.012012012012012012
# 0.4232018207282413,0.014014014014014014
# 0.4783485360573639,0.016016016016016016

# EOF
