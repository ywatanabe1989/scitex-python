<!-- ---
!-- Timestamp: 2025-12-09
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/config/README.md
!-- --- -->

# SciTeX Configuration Module

Centralized configuration management for the SciTeX ecosystem.

## Overview

This module provides:
- **ScitexConfig**: YAML-based configuration with env var substitution (Scholar pattern)
- **ScitexPaths**: Centralized path manager for all SciTeX directories
- **PriorityConfig**: Dict-based configuration resolver (for programmatic use)
- **CLI**: `scitex config list` command to view configuration

## Priority Resolution Patterns

### ScitexConfig (YAML-based, Recommended)

Follows the Scholar module's `CascadeConfig` pattern:

```
1. Direct value (highest) - Thread-safe, explicit parameter
2. Config (YAML)         - Values from default.yaml
3. Environment variable  - SCITEX_*, etc.
4. Default value         - Fallback (lowest)
```

**Key feature**: YAML can reference env vars using `${VAR:-default}` syntax.

### PriorityConfig (Dict-based)

For programmatic use with a dictionary:

```
1. Direct value (highest) - Explicit parameter
2. Config dict           - From passed dictionary
3. Environment variable  - SCITEX_*, etc.
4. Default value         - Fallback (lowest)
```

## Quick Start

### Using ScitexConfig (YAML-based)

```python
from scitex.config import get_config

# Load default configuration
config = get_config()

# Resolve values with precedence: direct → config → env → default
log_level = config.resolve("logging.level", default="INFO")
debug = config.resolve("debug.enabled", default=False, type=bool)

# Access raw config values
print(config.get("scitex_dir"))  # Value from YAML

# Load custom config
custom_config = get_config("/path/to/custom.yaml")
```

### Using ScitexPaths (Recommended for paths)

```python
import scitex

# Access via global PATHS constant
print(scitex.PATHS.logs)           # ~/.scitex/logs
print(scitex.PATHS.capture)        # ~/.scitex/capture
print(scitex.PATHS.scholar_library) # ~/.scitex/scholar/library

# Or import directly
from scitex.config import get_paths

paths = get_paths()  # Cached singleton

# Use resolve() for configurable paths in modules
cache_dir = paths.resolve("cache", user_provided_value)  # direct → default
```

### Using get_scitex_dir()

```python
from scitex.config import get_scitex_dir

# Simple usage - respects SCITEX_DIR env var
base_dir = get_scitex_dir()  # Returns Path

# With direct override (thread-safe)
base_dir = get_scitex_dir("/custom/path")
```

### Using PriorityConfig

```python
from scitex.config import PriorityConfig

config = PriorityConfig(
    config_dict={"port": 3000, "debug": True},
    env_prefix="SCITEX_"
)

# Resolves: direct → env (SCITEX_PORT) → config_dict → default
port = config.resolve("port", direct_val=None, default=8000, type=int)
```

## Pattern for Module Implementation

When implementing a module that needs configurable paths, use the `resolve()` method:

```python
from typing import Optional
from scitex.config import get_paths

class MyModule:
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Args:
            cache_dir: Custom cache directory. If None, uses default
                      from SCITEX_DIR environment variable.
        """
        # resolve() handles: direct value → default (from SCITEX_DIR env → ~/.scitex)
        self.cache_dir = get_paths().resolve("cache", cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
```

The `resolve(path_name, direct_val)` method:
- Returns `direct_val` if provided (highest priority)
- Otherwise returns the default path from `SCITEX_DIR` environment variable
- Falls back to `~/.scitex/<path>` if `SCITEX_DIR` is not set

This pattern ensures:
- **Thread-safety**: Direct values allow multi-user/multi-project scenarios
- **Configurability**: Environment variables for deployment flexibility
- **Sensible defaults**: Works out-of-the-box with `~/.scitex`
- **Consistency**: Same pattern as `PriorityConfig.resolve()`

## Directory Structure

```
$SCITEX_DIR/                    # Default: ~/.scitex
├── browser/                    # Browser module data
│   ├── screenshots/            # Browser debugging screenshots
│   ├── sessions/               # Shared browser sessions
│   └── persistent/             # Persistent browser profiles
├── cache/                      # General cache
│   └── functions/              # Function cache (joblib)
├── capture/                    # Screen captures
├── impact_factor_cache/        # Impact factor data
├── logs/                       # Log files
├── openathens_cache/           # OpenAthens auth cache
├── rng/                        # Random number generator state
├── scholar/                    # Scholar module
│   ├── cache/                  # Scholar-specific cache
│   └── library/                # PDF library
├── screenshots/                # General screenshots
├── test_monitor/               # Test monitoring screenshots
└── writer/                     # Writer module data
```

## CLI Commands

```bash
# List all configured paths
scitex config list

# Include environment variable info
scitex config list --env

# Show only existing directories
scitex config list --exists

# Output as JSON
scitex config list --json

# Initialize all directories
scitex config init
scitex config init --dry-run

# Show a specific path
scitex config show logs
scitex config show scholar_library
```

## YAML Configuration

### default.yaml Format

The `default.yaml` file supports environment variable substitution using `${VAR:-default}` syntax:

```yaml
# Base directory
scitex_dir: ${SCITEX_DIR:-"~/.scitex"}

# Nested configuration
logging:
  level: ${SCITEX_LOG_LEVEL:-"INFO"}
  file_logging: ${SCITEX_FILE_LOGGING:-true}

debug:
  enabled: ${SCITEX_DEBUG:-false}
```

### Environment Variable Substitution

| Syntax | Description | Example |
|--------|-------------|---------|
| `${VAR}` | Use env var (null if not set) | `${SCITEX_DIR}` |
| `${VAR:-default}` | Use env var or default | `${SCITEX_DIR:-"~/.scitex"}` |

### How It Works

1. **YAML loads with substitution**: When YAML is loaded, `${VAR:-default}` expressions are replaced with environment variable values or defaults
2. **Resolution respects priority**: When you call `config.resolve()`, it checks: direct → config (from YAML) → env → default

Example:
```yaml
# In default.yaml
log_level: ${SCITEX_LOG_LEVEL:-"INFO"}
```

```python
# If SCITEX_LOG_LEVEL is not set:
config.get("logging.level")  # Returns "INFO" (from YAML default)

# If SCITEX_LOG_LEVEL=DEBUG:
config.get("logging.level")  # Returns "DEBUG" (substituted at load time)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SCITEX_DIR` | Base directory for all SciTeX data | `~/.scitex` |
| `SCITEX_LOG_LEVEL` | Logging level | `INFO` |
| `SCITEX_DEBUG` | Enable debug mode | `false` |

## .env File Support

The module automatically loads `.env` files from:
1. Current working directory (`./.env`)
2. Home directory (`~/.env`)

Example `.env`:
```bash
SCITEX_DIR=/data/scitex
SCITEX_LOG_LEVEL=DEBUG
```

Note: Environment variables set in shell take precedence over `.env` values.

## Available Paths

Access via `scitex.PATHS.<name>` or `get_paths().<name>`:

| Property | Path |
|----------|------|
| `base` | `$SCITEX_DIR` |
| `logs` | `$SCITEX_DIR/logs` |
| `cache` | `$SCITEX_DIR/cache` |
| `function_cache` | `$SCITEX_DIR/cache/functions` |
| `capture` | `$SCITEX_DIR/capture` |
| `screenshots` | `$SCITEX_DIR/screenshots` |
| `rng` | `$SCITEX_DIR/rng` |
| `browser` | `$SCITEX_DIR/browser` |
| `browser_screenshots` | `$SCITEX_DIR/browser/screenshots` |
| `browser_sessions` | `$SCITEX_DIR/browser/sessions` |
| `browser_persistent` | `$SCITEX_DIR/browser/persistent` |
| `test_monitor` | `$SCITEX_DIR/test_monitor` |
| `impact_factor_cache` | `$SCITEX_DIR/impact_factor_cache` |
| `openathens_cache` | `$SCITEX_DIR/openathens_cache` |
| `scholar` | `$SCITEX_DIR/scholar` |
| `scholar_cache` | `$SCITEX_DIR/scholar/cache` |
| `scholar_library` | `$SCITEX_DIR/scholar/library` |
| `writer` | `$SCITEX_DIR/writer` |

## Thread-Safe Usage

For multi-user or multi-project scenarios, pass explicit paths:

```python
from scitex.config import ScitexPaths

# Each user/project gets isolated paths
user1_paths = ScitexPaths(base_dir="/data/user1/.scitex")
user2_paths = ScitexPaths(base_dir="/data/user2/.scitex")

# Use in module initialization
processor = DataProcessor(cache_dir=user1_paths.cache)
```

## Migration from Hardcoded Paths

If you have code with hardcoded paths like:

```python
# Old (hardcoded)
cache_dir = Path.home() / ".scitex" / "cache"
```

Replace with:

```python
# New (configurable)
from scitex.config import get_paths
cache_dir = get_paths().cache
```

<!-- EOF -->
