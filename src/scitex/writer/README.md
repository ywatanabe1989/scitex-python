# SciTeX Writer

Thin wrapper delegating to [scitex-writer](https://github.com/ywatanabe1989/scitex-writer) package.

## Installation

```bash
pip install scitex-writer
```

## Usage

```python
from scitex.writer import Writer
from pathlib import Path

# Create or attach to a writer project
writer = Writer(Path("my_paper"))

# Compile manuscript
result = writer.compile_manuscript()
if result.success:
    print(f"PDF created: {result.output_pdf}")

# Compile supplementary
result = writer.compile_supplementary()

# Compile revision with change tracking
result = writer.compile_revision(track_changes=True)
```

## Source of Truth

The implementation lives in the `scitex-writer` package. This module (`scitex.writer`) simply re-exports from `scitex_writer`:

- `Writer` - Main class for manuscript compilation
- `CompilationResult` - Compilation result dataclass
- `ManuscriptTree`, `SupplementaryTree`, `RevisionTree` - Document tree structures
- `bib`, `compile`, `figures`, `guidelines`, `project`, `prompts`, `tables` - Submodules

## Direct Import

You can also import directly from scitex-writer:

```python
from scitex_writer import Writer, CompilationResult
```

## Documentation

See [scitex-writer documentation](https://github.com/ywatanabe1989/scitex-writer) for full API reference.
