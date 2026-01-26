#!/usr/bin/env python3
"""Session script template - full @stx.session with all injections."""

TEMPLATE = {
    "name": "Session Script",
    "description": "@stx.session decorated script with auto CLI, config injection, and output tracking",
    "filename": "script.py",
    "usage": """
Usage:
  python script.py                    # Run with defaults
  python script.py --kwarg1 value     # Override parameter
  python script.py --help             # Show auto-generated help

Output Structure:
  script_out/
  ├── output.csv                      # stx.io.save() files go here (ROOT)
  ├── figure.png
  └── FINISHED_SUCCESS/<session_id>/  # Session metadata only
""",
    "content": '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "{timestamp}"
# File: {filepath}

"""
{docstring}

Usage
-----
$ python {filename} --help           # Show auto-generated CLI help
$ python {filename}                  # Run with default parameters
$ python {filename} --kwarg1 value   # Override parameters

Injected Global Variables
-------------------------
CONFIG : DotDict (access: CONFIG.key or CONFIG['key'])
    Session configuration with paths and metadata.

    Key Paths:
    - CONFIG.ID        : Session ID (e.g., '2025Y-01M-20D-10h30m00s_XyZ1')
    - CONFIG.FILE      : This script's absolute path
    - CONFIG.SDIR_OUT  : Output ROOT directory ({filename_stem}_out/)
                         USE THIS for stx.io.save() - files go here!
    - CONFIG.SDIR_RUN  : Session run directory (inside RUNNING/ or FINISHED_*)
                         Contains logs and CONFIG snapshots only
    - CONFIG.PID       : Process ID
    - CONFIG.ARGS      : Parsed CLI arguments as dict

    YAML Config (from ./config/*.yaml):
    - CONFIG.<FILENAME>.<key> : Values from config files

plt : module
    matplotlib.pyplot configured for session

COLORS : DotDict
    Color palette: COLORS.blue, COLORS.red, COLORS.green, etc.

rng : RandomStateManager
    Reproducible random number generator (seeded)

logger : SciTeXLogger
    Configured logger: logger.info(), logger.warning(), logger.error()

Output Directory Structure
--------------------------
{filename_stem}_out/                    <- CONFIG.SDIR_OUT (save files here!)
├── output.csv                          <- stx.io.save(df, "output.csv")
├── figure.png                          <- stx.io.save(fig, "figure.png")
├── figure.csv                          <- Auto-exported plot data
└── FINISHED_SUCCESS/
    └── <session_id>/                   <- CONFIG.SDIR_RUN (logs/configs)
        ├── CONFIGS/
        │   ├── CONFIG.pkl
        │   └── CONFIG.yaml
        └── logs/
            ├── stdout.log
            └── stderr.log
"""

import numpy as np
import scitex as stx


@stx.session
def main(
    # Uncomment and modify as needed:
    # input_file,                  # Required argument -> CLI positional
    # n_samples=100,               # Optional argument -> CLI --n-samples
    # threshold=0.05,              # Optional argument -> CLI --threshold
    # verbose=True,                # Optional argument -> CLI --verbose
    CONFIG=stx.INJECTED,           # Auto-injected from ./config/*.yaml
    plt=stx.INJECTED,              # Pre-configured matplotlib
    COLORS=stx.INJECTED,           # Color palette
    rng=stx.INJECTED,              # Random number generator
    logger=stx.INJECTED,           # Session logger
):
    """
    Script description shown in --help output.

    This docstring becomes the CLI help message.
    """
    # Log session info
    logger.info(f"Session ID: {{CONFIG.ID}}")
    logger.info(f"Output dir: {{CONFIG.SDIR_OUT}}")  # Where files are saved

    # Generate example data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create figure
    fig, ax = plt.subplots()
    ax.plot_line(x, y, color=COLORS.blue, label="sin(x)")
    ax.set_xyt("X", "Y", "Example Figure")
    ax.legend_()

    # Save outputs to CONFIG.SDIR_OUT (automatic)
    stx.io.save(fig, "figure.png")       # -> {filename_stem}_out/figure.png
    # stx.io.save(df, "results.csv")     # -> {filename_stem}_out/results.csv

    return 0  # Return 0 for success


if __name__ == "__main__":
    main()

# EOF
''',
}

# EOF
