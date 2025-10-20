# scitex.capture

Screen capture and monitoring utilities optimized for WSL and Windows.

## Overview

`scitex.capture` provides comprehensive screen capture capabilities with both:
1. **Direct API**: Python functions for direct screen capture
2. **MCP Server**: Model Context Protocol server for AI agent integration

## Quick Start

```python
from scitex import capture

# Take a screenshot
capture.snap("debug message")

# Multi-monitor
capture.snap(capture_all=True)

# Continuous monitoring
capture.start()
# ... do work ...
capture.stop()

# Create GIF from session
capture.gif()
```

## Core API

### capture.snap() / capture.take()
Take a single screenshot.

```python
from scitex import capture

# Basic screenshot
capture.snap("checkpoint_1")

# High quality
capture.snap("important_state", quality=95)

# Specific monitor
capture.snap("monitor_2", monitor=1)

# All monitors
capture.snap("all_screens", capture_all=True)
```

### capture.start() / capture.stop()
Continuous monitoring.

```python
from scitex import capture

# Start monitoring (every 1 second by default)
capture.start()

# Do your work
process_data()

# Stop monitoring
capture.stop()
```

### capture.gif()
Create animated GIF from latest monitoring session.

```python
from scitex import capture

# After monitoring session
capture.gif()  # Creates GIF from latest session

# Custom options
from scitex.capture import create_gif_from_latest_session
create_gif_from_latest_session(duration=0.5, optimize=True)
```

### capture.get_info()
Get comprehensive system display information.

```python
info = capture.get_info()
print(info['Monitors'])  # Monitor details
print(info['Windows'])   # Window information
print(info['VirtualDesktops'])  # Virtual desktop info
```

### capture.capture_window()
Capture specific window by handle.

```python
# Get window information
info = capture.get_info()
windows = info['Windows']['Details']

# Capture first window
if windows:
    handle = windows[0]['Handle']
    path = capture.capture_window(handle)
```

## MCP Server Integration

The module includes an MCP server (`mcp_server.py`) that exposes scitex.capture functionality to AI agents via the Model Context Protocol.

### MCP Functions Available:
- `mcp__cammy__capture_screenshot` - Single screenshots
- `mcp__cammy__start_monitoring` - Start continuous capture
- `mcp__cammy__stop_monitoring` - Stop continuous capture
- `mcp__cammy__get_monitoring_status` - Get status
- `mcp__cammy__create_gif` - Create GIF from sessions
- `mcp__cammy__list_sessions` - List available sessions
- `mcp__cammy__get_info` - Get system information
- `mcp__cammy__list_windows` - List all windows
- `mcp__cammy__capture_window` - Capture specific window
- And more...

## Features

- **WSL Support**: Seamlessly captures Windows host screens from WSL
- **Multi-Monitor**: Support for multiple monitors and all-monitor capture
- **JPEG Compression**: Configurable quality for smaller file sizes
- **Continuous Monitoring**: Background thread for automatic screenshots
- **GIF Creation**: Generate timeline GIFs from monitoring sessions
- **Window Capture**: Capture specific windows by handle
- **Thread-Safe**: Safe for concurrent operations
- **Human-Readable**: Timestamps and organized output

## Advanced Usage

### Custom Output Directory
```python
from scitex.capture import utils

utils.capture(
    message="custom_location",
    output_dir="/my/custom/path",
    quality=90
)
```

### Monitoring with Callbacks
```python
from scitex.capture.capture import CaptureManager

manager = CaptureManager()

def on_capture(filepath):
    print(f"Captured: {filepath}")

def on_error(error):
    print(f"Error: {error}")

manager.start_capture(
    interval=2.0,
    quality=60,
    on_capture=on_capture,
    on_error=on_error
)
```

### GIF from Custom Files
```python
from scitex.capture.gif import create_gif_from_files

gif_path = create_gif_from_files(
    image_paths=["img1.png", "img2.png", "img3.png"],
    output_path="/tmp/demo.gif",
    duration=0.5
)
```

### GIF from Pattern
```python
from scitex.capture.gif import create_gif_from_pattern

gif_path = create_gif_from_pattern(
    pattern="/tmp/screenshots/*.jpg",
    duration=0.3
)
```

## Configuration

Default cache location: `~/.cache/scitex_capture`

Quality settings:
- Monitoring: 60 (balance of quality and size)
- Single screenshots: 85 (higher quality)
- High-quality debug: 95

## CLI Usage

The module can also be used from command line:

```bash
# Take screenshot with message
python -m scitex.capture "my_message"

# Take screenshot (no message)
python -m scitex.capture

# Capture all monitors
python -m scitex.capture --all "all_screens"

# Capture specific app
python -m scitex.capture --app chrome "browser_state"

# Start monitoring
python -m scitex.capture --start

# Stop monitoring
python -m scitex.capture --stop

# Create GIF from latest session
python -m scitex.capture --gif

# Start MCP server
python -m scitex.capture --mcp

# List windows
python -m scitex.capture --list

# Show display info
python -m scitex.capture --info
```

## API Reference

### Main Functions

- `capture.snap(message, **kwargs)` - Take screenshot (primary API)
- `capture.take(message, **kwargs)` - Take screenshot (alternative)
- `capture.start()` - Start monitoring
- `capture.stop()` - Stop monitoring
- `capture.gif()` - Create GIF from latest session
- `capture.get_info()` - Get display/window information
- `capture.list_windows()` - List all windows (alias for get_info)
- `capture.capture_window(handle)` - Capture specific window

### GIF Functions

- `create_gif_from_session(session_id, **kwargs)` - GIF from session ID
- `create_gif_from_latest_session(**kwargs)` - GIF from latest session
- `create_gif_from_files(image_paths, **kwargs)` - GIF from file list
- `create_gif_from_pattern(pattern, **kwargs)` - GIF from glob pattern

### Utils

- `utils.capture(...)` - Low-level capture function
- `utils.start_monitor(...)` - Low-level monitoring start
- `utils.stop_monitor()` - Low-level monitoring stop

## Examples

See `examples/capture_examples/` for comprehensive usage examples:
- `basic_usage.py` - Core functionality demos
- `debugging_workflow.py` - Practical debugging scenarios

## Technical Details

### PowerShell Scripts
Located in `powershell/` directory:
- `capture_single_monitor.ps1` - Single monitor capture with DPI awareness
- `capture_all_monitors.ps1` - All monitors combined
- `capture_window_by_handle.ps1` - Window-specific capture
- `detect_monitors_and_desktops.ps1` - System information enumeration

### Fallback Mechanisms
1. PowerShell scripts (preferred for WSL)
2. Inline PowerShell commands
3. Native tools (mss, scrot)

## See Also

- Configuration: `config/capture.yaml`
- Examples: `examples/capture_examples/`
- MCP Server: `src/scitex/capture/mcp_server.py`

EOF
