# SciTeX Writer

Python wrapper around scitex-writer shell scripts for LaTeX compilation.

## Features

- ✅ **Compile Documents**: Manuscript, supplementary, revision
- ✅ **Watch Mode**: Auto-recompile on file changes
- ✅ **Exit Code Handling**: Proper error detection and reporting
- ✅ **Output Parsing**: Extract errors and warnings from LaTeX logs
- ✅ **Integration**: Works with scitex.project for project management

## Installation

```bash
# Install scitex package
pip install scitex

# Clone scitex-writer (required for shell scripts)
git clone git@github.com:ywatanabe1989/scitex-writer /tmp/scitex-writer
```

## Quick Start

### Compile Manuscript

```python
from scitex.writer import compile_manuscript
from pathlib import Path

# Compile manuscript
result = compile_manuscript(project_dir=Path("/path/to/writer-project"))

if result.success:
    print(f"✓ PDF created: {result.output_pdf}")
    print(f"  Duration: {result.duration:.2f}s")
else:
    print(f"✗ Compilation failed (exit code {result.exit_code})")
    for error in result.errors:
        print(f"  Error: {error}")
```

### Watch Mode (Auto-Recompile)

```python
from scitex.writer import watch_manuscript
from pathlib import Path

# Watch and auto-recompile on file changes
def on_compile():
    print("Recompiled!")

watch_manuscript(
    project_dir=Path("/path/to/writer-project"),
    on_compile=on_compile
)
```

### Integration with SciTeXProject

```python
from scitex.project import SciTeXProject
from scitex.writer import compile_manuscript
from pathlib import Path

# Load project
project = SciTeXProject.load_from_directory(Path("/path/to/project"))

# Get writer directory (creates if doesn't exist)
writer_dir = project.get_scitex_directory('writer')

# Copy writer template to project
from scitex.writer import copy_template
copy_template(
    source=Path("/tmp/scitex-writer"),
    dest=writer_dir
)

# Compile
result = compile_manuscript(project_dir=writer_dir)
```

## API Reference

### `compile_manuscript(project_dir, additional_args=None, timeout=300)`

Compile manuscript document.

**Args:**
- `project_dir` (Path): Path to writer project directory (containing `01_manuscript/`)
- `additional_args` (List[str], optional): Additional arguments for compilation
- `timeout` (int): Timeout in seconds (default: 300)

**Returns:**
- `CompilationResult`: Compilation result with status, outputs, and errors

**Example:**
```python
result = compile_manuscript(Path("/path/to/project"))
print(f"Success: {result.success}")
print(f"Exit code: {result.exit_code}")
print(f"PDF: {result.output_pdf}")
print(f"Errors: {len(result.errors)}")
```

### `compile_supplementary(project_dir, additional_args=None, timeout=300)`

Compile supplementary materials.

**Args:**
- `project_dir` (Path): Path containing `02_supplementary/`
- `additional_args` (List[str], optional): Additional arguments
- `timeout` (int): Timeout in seconds

**Returns:**
- `CompilationResult`

### `compile_revision(project_dir, track_changes=False, additional_args=None, timeout=300)`

Compile revision responses.

**Args:**
- `project_dir` (Path): Path containing `03_revision/`
- `track_changes` (bool): Enable change tracking (default: False)
- `additional_args` (List[str], optional): Additional arguments
- `timeout` (int): Timeout in seconds

**Returns:**
- `CompilationResult`

### `CompilationResult`

Dataclass containing compilation results.

**Attributes:**
- `success` (bool): Whether compilation succeeded (exit code 0)
- `exit_code` (int): Process exit code
- `stdout` (str): Standard output from compilation
- `stderr` (str): Standard error from compilation
- `output_pdf` (Path, optional): Path to generated PDF
- `diff_pdf` (Path, optional): Path to diff PDF with tracked changes
- `log_file` (Path, optional): Path to compilation log file
- `duration` (float): Compilation duration in seconds
- `errors` (List[str]): Parsed LaTeX errors
- `warnings` (List[str]): Parsed LaTeX warnings

**Example:**
```python
result = compile_manuscript(project_dir)

if result.success:
    print(f"✓ Success in {result.duration:.2f}s")
    print(f"  Output: {result.output_pdf}")
    if result.warnings:
        print(f"  Warnings: {len(result.warnings)}")
else:
    print(f"✗ Failed with exit code {result.exit_code}")
    for error in result.errors:
        print(f"  {error}")
```

### `watch_manuscript(project_dir, interval=2, on_compile=None, timeout=None)`

Watch and auto-recompile manuscript on file changes.

**Args:**
- `project_dir` (Path): Writer project directory
- `interval` (int): Check interval in seconds (default: 2)
- `on_compile` (Callable, optional): Callback after each compilation
- `timeout` (int, optional): Timeout in seconds (None = infinite)

**Example:**
```python
import time

def callback():
    print(f"[{time.strftime('%H:%M:%S')}] Recompiled")

watch_manuscript(
    project_dir=Path("/path/to/project"),
    on_compile=callback
)
```

### `create_writer_project(dest, name, template_source=None)`

Create new writer project from template.

**Args:**
- `dest` (Path): Destination directory
- `name` (str): Project name
- `template_source` (Path, optional): Custom template (default: auto-detect)

**Returns:**
- `Path`: Path to created project

**Example:**
```python
project_dir = create_writer_project(
    dest=Path("/path/to/my-paper"),
    name="My Research Paper"
)
```

### `WriterConfig.from_directory(project_dir)`

Create configuration from project directory.

**Args:**
- `project_dir` (Path): Writer project root

**Returns:**
- `WriterConfig`: Configuration object

**Example:**
```python
config = WriterConfig.from_directory(Path("/path/to/project"))
print(config.manuscript_dir)
print(config.shared_dir)
config.validate()  # Check structure
```

## Directory Structure

Writer projects follow this structure:

```
writer-project/
├── compile                   # Main compilation script
├── 01_manuscript/            # Manuscript
│   ├── contents/
│   │   ├── abstract.tex
│   │   ├── introduction.tex
│   │   ├── methods.tex
│   │   ├── results.tex
│   │   ├── discussion.tex
│   │   ├── figures/
│   │   └── tables/
│   ├── manuscript.pdf        # Output
│   ├── manuscript_diff.pdf   # Diff (if generated)
│   └── logs/                 # Compilation logs
├── 02_supplementary/         # Supplementary materials
├── 03_revision/              # Revision responses
├── shared/                   # Shared resources
│   ├── bib_files/
│   │   └── bibliography.bib
│   ├── title.tex
│   ├── authors.tex
│   └── latex_styles/
└── scripts/                  # Shell scripts
    └── shell/
        ├── compile_manuscript.sh
        ├── compile_supplementary.sh
        ├── compile_revision.sh
        └── watch_compile.sh
```

## Exit Codes

- `0`: Success
- `1`: General compilation error
- `124`: Timeout
- `127`: Compile script not found

## Error Handling

```python
result = compile_manuscript(project_dir)

if not result.success:
    print(f"Compilation failed with exit code {result.exit_code}")

    # Parse errors
    for error in result.errors:
        print(f"LaTeX Error: {error}")

    # Check stderr
    if result.stderr:
        print(f"stderr: {result.stderr}")

    # Check log file
    if result.log_file:
        print(f"Check log: {result.log_file}")
```

## Requirements

- Python 3.8+
- scitex-writer shell scripts (cloned to `/tmp/scitex-writer` or `~/proj/scitex-writer`)
- LaTeX distribution (handled by scitex-writer containers)

## Troubleshooting

### "scitex-writer compile script not found"

Clone scitex-writer:
```bash
git clone git@github.com:ywatanabe1989/scitex-writer /tmp/scitex-writer
```

### Compilation timeout

Increase timeout:
```python
result = compile_manuscript(project_dir, timeout=600)  # 10 minutes
```

### Permission errors

Make compile script executable:
```bash
chmod +x /tmp/scitex-writer/compile
```

## See Also

- [scitex.project](../project/README.md) - Project management
- [scitex-writer GitHub](https://github.com/ywatanabe1989/scitex-writer) - Shell scripts repository
