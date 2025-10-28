<!-- ---
!-- Timestamp: 2025-10-28 16:26:56
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/README.md
!-- --- -->

# SciTeX Writer

Python interface for LaTeX manuscript compilation.

## Usage

```python
from scitex.writer import Writer
from pathlib import Path

# # Standalone manuscript (isolated git repository - default)
# writer = Writer(Path("my_paper"))

# Project-integrated manuscript (use parent's git repository)
writer = Writer(Path("my_project/scitex/writer"), git_strategy='parent')

# No git (temporary work)
writer = Writer(Path("temp_work"), git_strategy=None)

# Document operations (git-based version control)
intro = writer.manuscript.introduction
lines = intro.read()            # Read file
intro.write(lines + ["# New"])  # Write file

intro.commit("Update intro")     # Commit to git
intro.save("Update intro")      # Alias for commit()

intro.history()                 # Show git log
intro.diff()                    # Show git diff vs HEAD

intro.checkout("HEAD~1")        # Restore from previous version
intro.checkout("HEAD")          # Restore from HEAD

# Compilation
result = writer.compile_manuscript()
if result.success:
    print(f"PDF: {result.output_pdf}")

# Utilities
pdf = writer.get_pdf()
writer.watch()
writer.delete()
```

**Git Strategies:**
- **`'child'` (default)**: Isolated git repository in project directory
  - Self-contained version history
  - Can use git directly in project directory
- **`'parent'`**: Use existing parent git repository
  - Manuscript tracked in project's git repo
  - Better for code + paper reproducibility
- **`None`**: Disable git (for temporary work)

## Writer Class

```
Writer(project_dir, name=None, git_strategy='child')
├── manuscript: ManuscriptDocument
│   ├── .abstract → DocumentSection
│   ├── .introduction → DocumentSection
│   ├── .methods → DocumentSection
│   ├── .results → DocumentSection
│   └── .discussion → DocumentSection
├── supplementary: SupplementaryDocument
│   └── .<any_name> → DocumentSection
├── revision: RevisionDocument
│   └── .<any_name> → DocumentSection
│
├── compile_manuscript(timeout=300) → CompilationResult
├── compile_supplementary(timeout=300) → CompilationResult
├── compile_revision(track_changes=False, timeout=300) → CompilationResult
├── watch(on_compile=None)
├── get_pdf(doc_type='manuscript') → Path
└── delete() → bool

DocumentSection
├── read() → content (uses scitex.io.load)
├── write(content) → bool
├── commit(message) → bool (git)
├── save(message) → bool (alias for commit)
├── history() → List[str] (git log)
├── diff() → str (git diff)
├── checkout(ref='HEAD') → bool (restore from git)
└── .path → Path
```

## CompilationResult

```python
result.success       # bool
result.exit_code     # int
result.output_pdf    # Path
result.duration      # float (seconds)
result.errors        # List[str]
result.warnings      # List[str]
result.stdout        # str
result.stderr        # str
result.log_file      # Path
```

## Requirements

- Python 3.8+
- LaTeX distribution
- Git (for version control)

<!-- EOF -->