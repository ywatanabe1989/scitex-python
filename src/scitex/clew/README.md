<!-- ---
!-- Timestamp: 2026-02-01 08:47:14
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-python/src/scitex/verify/README.md
!-- --- -->

# scitex.clew Module

Hash-based verification system for reproducible scientific computations.

## Overview

The verify module provides cryptographic tracking of scientific pipelines, enabling researchers to:
- **Detect changes** in input/output files
- **Trace dependencies** through processing chains
- **Verify reproducibility** by re-executing scripts
- **Visualize workflows** as directed acyclic graphs (DAGs)

![Verification DAG](dag.png)

*Example DAG showing verification states: ✓ verified (green), ✗ failed (red)*

## Architecture

```
scitex/clew/
├── __init__.py          # Public API and convenience functions
├── _hash.py             # SHA256 hashing utilities
├── _db.py               # SQLite database for storing hashes
├── _tracker.py          # Session tracking integration
├── _chain.py            # Chain verification logic
├── _rerun.py            # Re-execution verification
├── _integration.py      # Hooks for stx.io and @stx.session
├── _visualize.py        # Re-exports from _viz/
└── _viz/
    ├── _mermaid.py      # Mermaid DAG generation
    ├── _json.py         # JSON DAG export
    ├── _format.py       # Terminal output formatting
    ├── _colors.py       # Color constants
    ├── _templates.py    # HTML templates
    └── _utils.py        # Shared utilities
```

## Core Components

### Hash Utilities (`_hash.py`)

```python
from scitex.clew import hash_file, hash_files, hash_directory

# Single file
h = hash_file("data.csv")  # SHA256 hex string

# Multiple files
hashes = hash_files(["a.csv", "b.csv"])  # {path: hash}

# Directory (recursive)
h = hash_directory("output/")  # Combined hash of all files
```

### Database (`_db.py`)

SQLite-based storage for verification records:

```python
from scitex.clew import get_db

db = get_db()  # ~/.scitex/verify.db by default

# Record a run
db.record_run(
    session_id="abc123",
    script_path="/path/to/script.py",
    script_hash="sha256...",
    config_hash="sha256...",
    status="success"
)

# Record file hashes
db.record_file_hash(
    session_id="abc123",
    file_path="/path/to/output.csv",
    file_hash="sha256...",
    role="output"  # or "input"
)

# Query runs
runs = db.list_runs(limit=10, status="success")
chain = db.get_chain("abc123")  # Parent session IDs
```

### Session Tracker (`_tracker.py`)

Integrates with `@stx.session`:

```python
from scitex.clew import SessionTracker, start_tracking, stop_tracking

# Manual tracking (usually automatic via @stx.session)
tracker = start_tracking(session_id="abc123")
tracker.add_input("/path/to/input.csv")
tracker.add_output("/path/to/output.csv")
stop_tracking()
```

### Chain Verification (`_chain.py`)

```python
from scitex.clew import verify_file, verify_run, verify_chain

# Verify single file
file_result = verify_file("/path/to/output.csv")
print(file_result.is_verified)  # True if hash matches

# Verify session run
run_result = verify_run("abc123")
print(run_result.is_verified)  # True if all files match
print(run_result.mismatched_files)  # List of changed files

# Verify entire chain
chain_result = verify_chain("/path/to/final_output.csv")
print(chain_result.is_verified)  # True if all runs verified
print(chain_result.chain_length)  # Number of runs in chain
for run in chain_result.runs:
    print(f"{run.session_id}: {run.status}")
```

### Verification Levels

```python
from scitex.clew import VerificationLevel, VerificationStatus

# Levels
VerificationLevel.CACHE     # Fast hash comparison (✓)
VerificationLevel.RERUN     # Re-execution verification (✓✓)

# Statuses
VerificationStatus.VERIFIED
VerificationStatus.UNVERIFIED
VerificationStatus.FAILED
VerificationStatus.UNKNOWN
```

### Re-execution Verification (`_rerun.py`)

```python
from scitex.clew import verify_by_rerun

# Re-run script and verify outputs match
result = verify_by_rerun("/path/to/output.csv")
print(result.is_verified)  # True if re-execution produces same hashes
print(result.level)  # VerificationLevel.RERUN
```

## Visualization (`_visualize.py`)

### Terminal Output

```python
from scitex.clew import format_status, format_chain_verification

# Git status-like output
print(format_status())

# Chain visualization
chain = verify_chain("output.csv")
print(format_chain_verification(chain))
```

### Mermaid DAG

```python
from scitex.clew import generate_mermaid_dag

mermaid = generate_mermaid_dag(target_file="output.csv")
# Returns:
# graph TD
#     script_0["✓ 🐍 analyze.py"]:::verified
#     file_0[("✓ 📊 output.csv")]:::file_ok
#     script_0 --> file_0
#     classDef verified fill:#90EE90...
```

### HTML/PNG/SVG Export

```python
from scitex.clew import render_dag

# Interactive HTML
render_dag("dag.html", target_file="output.csv", show_hashes=True)

# Static images (requires mmdc)
render_dag("dag.png", target_file="output.csv")
render_dag("dag.svg", target_file="output.csv")

# Raw formats
render_dag("dag.mmd", target_file="output.csv")  # Mermaid code
render_dag("dag.json", target_file="output.csv")  # Graph structure
```

## Integration Hooks (`_integration.py`)

Automatically called by `@stx.session` and `stx.io`:

```python
from scitex.clew import on_session_start, on_session_close, on_io_load, on_io_save

# Session lifecycle
on_session_start(session_id, script_path, config_hash)
on_session_close(session_id, status="success")

# I/O tracking
on_io_load(file_path)  # Records as input
on_io_save(file_path)  # Records as output
```

## CLI Commands

```bash
# List runs
scitex clew list [--limit N] [--status success|failed]

# Check status
scitex clew status

# Verify specific run
scitex clew run SESSION_ID [--from-scratch]

# Trace dependencies
scitex clew chain FILE_PATH

# Database stats
scitex clew stats
```

## MCP Tools

Available via MCP protocol:

| Tool | Description |
|------|-------------|
| `verify_list` | List tracked runs |
| `verify_run` | Verify specific run |
| `verify_chain` | Trace dependencies |
| `verify_status` | Show changed items |
| `verify_stats` | Database statistics |
| `verify_mermaid` | Generate Mermaid DAG |

## Database Schema

```sql
-- Runs table
CREATE TABLE runs (
    session_id TEXT PRIMARY KEY,
    script_path TEXT,
    script_hash TEXT,
    config_hash TEXT,
    status TEXT,
    started_at TIMESTAMP,
    ended_at TIMESTAMP
);

-- File hashes table
CREATE TABLE file_hashes (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    file_path TEXT,
    file_hash TEXT,
    role TEXT,  -- 'input' or 'output'
    recorded_at TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES runs(session_id)
);

-- Verification records table
CREATE TABLE verifications (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    level TEXT,  -- 'cache' or 'rerun'
    status TEXT,  -- 'verified', 'failed'
    verified_at TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES runs(session_id)
);
```

## Examples

See `examples/scitex/clew/` for complete working examples:

- `00_run_all.sh` - Run complete pipeline
- `01-08` - Multi-branch processing pipeline
- `09_demo_verification.py` - Verification states demo
- `10_programmatic_verification.py` - API usage examples

## See Also

- `examples/scitex/clew/README.md` - Usage examples with DAG visualization
- `@stx.session` decorator - Automatic session tracking
- `stx.io` module - File I/O with hash tracking

<!-- EOF -->
