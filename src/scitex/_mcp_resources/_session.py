#!/usr/bin/env python3
# Timestamp: 2026-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/_mcp_resources/_session.py
"""Session output tree documentation resource."""

from __future__ import annotations

__all__ = ["register_session_resources"]

SESSION_TREE = """\
# @stx.session Output Directory Structure
==========================================

When you use @stx.session, outputs are automatically organized:

```
script.py                          # Your script
script_out/                        # Output directory (auto-created)
├── output.npy                     # Your saved files (ROOT level)
├── results.json                   # All outputs here, NOT in session dir
├── figure.png                     # Figures
├── figure.csv                     # Auto-exported plot data
├── RUNNING/                       # Currently running sessions
│   └── 2025Y-01M-20D-09h30m00s_AbC1-main/
│       ├── CONFIGS/
│       │   ├── CONFIG.pkl         # Python config object (pickle)
│       │   └── CONFIG.yaml        # Human-readable config snapshot
│       └── logs/
│           ├── stdout.log         # All print() output captured
│           └── stderr.log         # All errors captured
├── FINISHED_SUCCESS/              # Completed sessions (moved from RUNNING)
│   └── <session_id>-main/
│       ├── CONFIGS/...
│       └── logs/...
└── FINISHED_FAILED/               # Failed sessions (errors)
    └── <session_id>-main/
        └── ...
data/                              # Central navigation via symlinks
└── output.npy -> ../script_out/output.npy
```

## Key Points

1. **Session ID Format**: `YYYY'Y'-MM'M'-DD'D'-HH'h'MM'm'SS's'_XXXX-funcname`
   - Example: `2026Y-01M-20D-09h37m01s_boSr-main`

2. **Output File Placement**:
   - Files saved with `stx.io.save(obj, "filename")` go to `script_out/` ROOT
   - NOT inside the session subdirectory (CONFIGS/logs only there)

3. **Symlinks for Central Navigation**:
   ```python
   stx.io.save(arr, "output.npy", symlink_to="./data")
   # Creates: ./data/output.npy -> ../script_out/output.npy
   ```
   - Use `./data` directory to accumulate outputs from multiple scripts
   - Easy navigation without digging into individual script_out directories

4. **CONFIG Object** (available as `CONFIG=stx.INJECTED`):
   ```python
   CONFIG.ID          # "2026Y-01M-20D-09h37m01s_boSr"
   CONFIG.FILE        # "/path/to/script.py"
   CONFIG.SDIR_OUT    # "/path/to/script_out"
   CONFIG.SDIR_RUN    # "/path/to/script_out/RUNNING/<session_id>"
   CONFIG.PID         # 12345
   CONFIG.ARGS        # {"n_points": 100, ...}
   ```

5. **YAML Config Loading** (from `./config/*.yaml`):
   ```yaml
   # ./config/experiment.yaml
   model:
     hidden_size: 256
     num_layers: 3
   training:
     batch_size: 32
   ```
   Access: `CONFIG.experiment.model.hidden_size  # 256`

6. **Automatic Cleanup**:
   - On success: RUNNING -> FINISHED_SUCCESS
   - On error: RUNNING -> FINISHED_FAILED
   - All print()/stderr captured in logs/

## Example Script

```python
#!/usr/bin/env python3
import scitex as stx
import numpy as np

@stx.session
def main(n_points=100, CONFIG=stx.INJECTED, plt=stx.INJECTED):
    \"\"\"Generate sample data and plot.\"\"\"

    x = np.linspace(0, 10, n_points)
    y = np.sin(x) * np.exp(-x/5)

    fig, ax = stx.plt.subplots()
    ax.stx_line(x, y)
    ax.set_xyt("X", "Y", "Damped Sine")

    # symlink_to for central navigation
    stx.io.save(fig, "sine.png", symlink_to="./data")
    fig.close()

    return 0

if __name__ == "__main__":
    main()
```

Output:
```
SUCC: Saved to: ./script_out/sine.png (241.6 KiB)
SUCC: Symlinked: /path/script_out/sine.png ->
SUCC:            /path/data/sine.png
```

Tree after running:
```
script.py
script_out/
├── sine.png                       # Figure at ROOT level
├── sine.csv                       # Auto-exported data
└── FINISHED_SUCCESS/
    └── 2026Y-01M-20D-09h37m01s_boSr-main/
        ├── CONFIGS/
        │   ├── CONFIG.pkl
        │   └── CONFIG.yaml
        └── logs/
            ├── stdout.log
            └── stderr.log
data/
└── sine.png -> ../script_out/sine.png
```
"""


def register_session_resources(mcp) -> None:
    """Register session tree resource."""

    @mcp.resource("scitex://session-tree")
    def session_tree() -> str:
        """Explain the @stx.session output directory structure."""
        return SESSION_TREE


# EOF
