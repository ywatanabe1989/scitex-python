<!-- ---
!-- Timestamp: 2025-10-16 02:47:57
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex_repo/src/scitex/session/README.md
!-- --- -->

# scitex.session

Experiment session management for reproducible scientific computing.

## Overview

scitex.session provides lifecycle management for scientific experiments with automatic logging, reproducibility settings, and session tracking.

## Quick Start

```python
import sys
import matplotlib.pyplot as plt
from scitex import session

CONFIG, sys.stdout, sys.stderr, plt, CC, rng_manager = session.start(sys, plt)

# Your experiment code here

session.close(CONFIG)
```

## Core Functions

### session.start()

Initialize experiment session with reproducibility settings.

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
- CC: Color cycle dictionary
- rng: RandomStateManager instance

### session.close()

Close experiment session and finalize logging.

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
- FINISHED_SUCCESS/ and FINISHED_ERROR/ based on exit status

### Configuration Management

- Saves CONFIG as .pkl and .yaml
- Includes all session metadata
- Command-line arguments captured
- Path objects and strings supported

## Advanced Usage

### Custom Save Directory

```python
CONFIG, *_ = session.start(sys, plt, sdir="/custom/path/")
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
├── FINISHED_SUCCESS/
└── FINISHED_ERROR/
```

## Configuration Object

CONFIG contains:
- ID: Session identifier
- PID: Process ID
- START_TIME: Session start timestamp
- END_TIME: Session end timestamp
- RUN_TIME: Formatted runtime (HH:MM:SS)
- FILE: Script file path
- SDIR: Save directory path
- SDIR_PATH: Path object version
- ARGS: Command-line arguments
- EXIT_STATUS: Exit code

## Matplotlib Integration

```python
CONFIG, sys.stdout, sys.stderr, plt, CC, rng_manager = session.start(
    sys, plt,
    fig_size_mm=(160, 100),
    dpi_save=300,
    hide_top_right_spines=True,
    alpha=0.9
)
```

- plt is replaced with scitex.plt wrapper
- CC provides color cycle dictionary
- Automatic style configuration

## Random State Management

```python
CONFIG, *_, rng_manager = session.start(sys, plt, seed=42)

random_array = rng.random((10, 10))
```

- rng is global RandomStateManager
- Automatically fixes seeds for all libraries
- Reproducible across runs

<!-- EOF -->