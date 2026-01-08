<!-- ---
!-- Timestamp: 2025-11-18 10:14:48
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/session/README.md
!-- --- -->

# scitex.session

Experiment session management for reproducible scientific computing.

## Overview

scitex.session provides lifecycle management for scientific experiments with automatic logging, reproducibility settings, and session tracking.

## Quick Start

```python
import scitex

@scitex.session
def main(
    CONFIG=scitex.INJECTED,
    plt=scitex.INJECTED,
    COLORS=scitex.INJECTED,
    rng=scitex.INJECTED,
):
    """Args injected by @scitex.session decorator"""
    print(f"Session ID: {CONFIG['ID']}")
    # Your experiment code here

if __name__ == "__main__":
    main()
```

## Core Functions

### @session.session Decorator

The recommended way to use scitex.session is through the `@scitex.session` decorator, which automatically handles session initialization and cleanup.

```python
import scitex

@scitex.session
def main(
    CONFIG=scitex.INJECTED,
    plt=scitex.INJECTED,
    COLORS=scitex.INJECTED,
    rng=scitex.INJECTED,
):
    """Args injected by @scitex.session decorator"""
    print(f"Session ID: {CONFIG['ID']}")

if __name__ == "__main__":
    main()
```

The decorator automatically injects the following parameters:
- `CONFIG`: Configuration dictionary with session metadata
- `plt`: Configured matplotlib.pyplot module
- `COLORS`: Color cycle dictionary
- `rng`: RandomStateManager instance

Using `scitex.INJECTED` as default values makes it explicit that these parameters are injected by the decorator.

Decorator Parameters:
- file: Script file path (auto-detected if None)
- sdir: Save directory (auto-generated if None)
- seed: Random seed for reproducibility (default: 42)
- agg: Use matplotlib Agg backend (default: False)
- verbose: Print detailed information (default: True)

### session.start() (Advanced)

Initialize experiment session with reproducibility settings. Note: The decorator approach is recommended over direct start()/close() calls.

Parameters:
- sys: Python sys module for I/O redirection
- plt: Matplotlib pyplot module
- file: Script file path (auto-detected if None)
- sdir: Save directory (auto-generated if None)
- seed: Random seed for reproducibility (default: 42)
- agg: Use matplotlib Agg backend (default: False)
- verbose: Print detailed information (default: True)

Returns:
- CONFIGS: Configuration dictionary with session metadata
- stdout, stderr: Redirected output streams
- plt: Configured matplotlib.pyplot module
- COLORS: Color cycle dictionary
- rng: RandomStateManager instance

### session.close() (Advanced)

Close experiment session and finalize logging. Note: The decorator handles this automatically.

Parameters:
- CONFIG: Configuration dictionary from start()
- message: Completion message (default: ':)')
- notify: Send notification (default: False)
- verbose: Print verbose output (default: True)
- exit_status: 0=success, 1=error, None=finished

## Features

### Automatic Logging

- Redirects stdout/stderr to log files
- Saves logs to SDIR/logs/
- Removes ANSI escape codes
- Captures all print statements

### Reproducibility

- Fixed random seeds via RandomStateManager
- Supports os, random, numpy, torch
- Records all configuration parameters
- Timestamps and session IDs

### Session Tracking

- Unique session IDs (4 characters)
- Process ID (PID) tracking
- Start/end timestamps
- Runtime calculation

### Directory Management

- Auto-generates save directories
- RUNNING/ for active sessions
- FINISHED/ for completed sessions
- FINISHED_SUCOLORSESS/ and FINISHED_ERROR/ based on exit status

### Configuration Management

- Saves CONFIG as .pkl and .yaml
- Includes all session metadata
- Command-line arguments captured
- Path objects and strings supported

## Advanced Usage

### Custom Save Directory

```python
import scitex

@scitex.session(sdir="/custom/path/")
def main():
    # Your experiment code here
    pass

if __name__ == "__main__":
    main()
```

### Debug Mode

Set IS_DEBUG in ./config/IS_DEBUG.yaml:

```yaml
IS_DEBUG: true
```

Session IDs will be prefixed with "DEBUG_"

### Session Manager

```python
from scitex.session import SessionManager

manager = SessionManager()
active = manager.get_active_sessions()
info = manager.get_session(session_id)
```

## Directory Structure

```
/path/to/script.py
/path/to/script_out/
├── RUNNING/
│   └── XXXX/              # Session ID
│       ├── logs/
│       │   ├── stdout.log
│       │   └── stderr.log
│       └── CONFIGS/
│           ├── CONFIG.pkl
│           └── CONFIG.yaml
├── FINISHED/
├── FINISHED_SUCOLORSESS/
└── FINISHED_ERROR/
```

## Configuration Object

CONFIG contains:
- ID: Session identifier
- PID: Process ID
- START_DATETIME: Session start timestamp (datetime object)
- END_DATETIME: Session end timestamp (datetime object)
- RUN_DURATION: Formatted runtime string (HH:MM:SS)
- FILE: Script file path (Path object)
- SDIR_OUT: Base output directory (Path object)
- SDIR_RUN: Current session directory (Path object)
- ARGS: Command-line arguments (dict)
- EXIT_STATUS: Exit code (0=success, 1=error)

## Matplotlib Integration

```python
import scitex

@scitex.session(
    fig_size_mm=(160, 100),
    dpi_save=300,
    hide_top_right_spines=True,
    alpha=0.9
)
def main():
    # plt and COLORS are automatically available
    plt.plot([1, 2, 3], color=COLORS[0])
    plt.show()

if __name__ == "__main__":
    main()
```

- plt is replaced with scitex.plt wrapper
- COLORS provides color cycle dictionary
- Automatic style configuration

## Random State Management

```python
import scitex

@scitex.session(seed=42)
def main():
    # rng is automatically available
    random_array = rng.random((10, 10))
    print(random_array)

if __name__ == "__main__":
    main()
```

- rng is global RandomStateManager
- Automatically fixes seeds for all libraries
- Reproducible across runs

<!-- EOF -->
