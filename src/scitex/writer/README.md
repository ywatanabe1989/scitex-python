<!-- ---
!-- Timestamp: 2025-10-29 05:24:57
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/README.md
!-- --- -->

# SciTeX Writer

Python interface for LaTeX manuscript compilation.

## Usage

```python
from scitex.writer import Writer
from pathlib import Path

 # Standalone manuscript (isolated git repository - default)
writer = Writer(Path("my_paper"))

# # Project-integrated manuscript (use parent's git repository)
# writer = Writer(Path("my_project/scitex/writer"), git_strategy='parent')
#  
# # No git (temporary work)
# writer = Writer(Path("temp_work"), git_strategy=None)

# Document operations (git-based version control)
intro = writer.manuscript.contents.introduction
lines = intro.read()            # Read file (uses scitex.io or fallback to plain text)
intro.write(lines + ["# New"])  # Write file

intro.commit("Update intro")     # Commit to git with message
# intro.save() is not an alias - use commit() instead

intro.history()                 # Show git log (returns list of commit messages)
intro.diff()                    # Show uncommitted changes vs HEAD (returns diff string)
intro.diff(ref="HEAD~1")        # Show uncommitted changes vs previous commit
intro.diff(ref="main")          # Show uncommitted changes vs main branch

intro.checkout("HEAD~1")        # Restore from previous version
intro.checkout("HEAD")          # Restore from HEAD (returns bool: success)

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

## API Reference

### Writer Class

```python
Writer(project_dir, name=None, git_strategy='child')

# Attributes:
writer.project_dir       # Path to project
writer.project_name      # Project name
writer.git_root          # Git repository root (if using git)

# Document trees:
writer.manuscript        # ManuscriptTree with contents and sections
writer.supplementary     # SupplementaryTree
writer.revision          # RevisionTree
writer.scripts           # ScriptsTree with compilation and utility scripts

# Methods:
writer.compile_manuscript(timeout=300)        # → CompilationResult
writer.compile_supplementary(timeout=300)     # → CompilationResult
writer.compile_revision(track_changes=False)  # → CompilationResult
writer.watch(on_compile=None)                 # Auto-recompile on changes
writer.get_pdf(doc_type='manuscript')         # → Path or None
writer.delete()                               # → bool
```

### DocumentSection Class

All manuscript/supplementary/revision sections are DocumentSection instances:

```python
section = writer.manuscript.contents.introduction  # Example

# Methods:
section.read()                     # → content (str or list)
section.write(content)             # → bool
section.commit(message)            # → bool (git add + commit)
section.history()                  # → List[str] (git log messages)
section.diff(ref="HEAD")          # → str (uncommitted changes vs ref, "" if none)
section.diff_between(ref1, ref2)  # → str (compare any two git states)
section.checkout(ref="HEAD")      # → bool (restore from git reference)

# Attributes:
section.path                       # → Path to file
section.git_root                   # → Path to git root (if available)
```

### Available Sections

**Manuscript Contents** (`writer.manuscript.contents.*`):
- Core: abstract, introduction, methods, results, discussion
- Metadata: title, authors, keywords, journal_name
- Optional: graphical_abstract, highlights, data_availability, additional_info, wordcount
- References: bibliography
- Directories: figures/, tables/, latex_styles/

**Supplementary & Revision**: Similar structure, customize as needed

## Usage Examples

### Basic Read/Write/Commit Workflow

```python
intro = writer.manuscript.contents.introduction

# 1. Read current content
lines = intro.read()

# 2. Modify and write
lines.append("New paragraph...")
intro.write(lines)

# 3. Check what changed (returns "" if no changes, diff string if changed)
changes = intro.diff()
if changes:
    print("Changes detected:")
    print(changes)

    # 4. Commit when satisfied
    intro.commit("Added new paragraph")
else:
    print("No changes to commit")
```

### Working with Git History

```python
# View full version history
history = intro.history()
for commit in history:
    print(commit)  # Output: "abc1234 Commit message"

# Compare uncommitted changes against different versions
diff_prev = intro.diff(ref="HEAD~1")  # Uncommitted changes vs last commit
diff_main = intro.diff(ref="main")    # Uncommitted changes vs main branch
diff_tag = intro.diff(ref="v1.0")     # Uncommitted changes vs tag

# Compare two arbitrary versions (no uncommitted changes needed)
diff = intro.diff_between("v1.0", "v2.0")        # Between tags
diff = intro.diff_between("HEAD~2", "HEAD")      # Between commits
diff = intro.diff_between("main", "develop")     # Between branches

# Time-based comparisons with human-readable timestamps
diff = intro.diff_between("1 week ago", "now")   # Last week's changes
diff = intro.diff_between("2 days ago", "HEAD")  # Last 2 days
diff = intro.diff_between("2025-10-20", "2025-10-28")  # Between dates

# Restore previous version
intro.checkout("HEAD~1")  # Restore to last commit
intro.checkout("main")    # Restore to main branch

# After restore, commit if needed
intro.commit("Reverted to previous version")
```

### Working with Scripts

```python
# Access compilation scripts
scripts = writer.scripts

# View script paths
compile_script = scripts.compile_manuscript.path
watch_script = scripts.watch_compile.path

# Read script content
content = scripts.compile_manuscript.read()

# Modify scripts (with git tracking)
new_content = scripts.compile_manuscript.read()
# ... modify content ...
scripts.compile_manuscript.write(new_content)
scripts.compile_manuscript.commit("Update compilation script")

# View script history
history = scripts.compile_manuscript.history()
diff = scripts.compile_manuscript.diff()
```

### Typical Edit Workflow

```python
# 1. Read current content
content = intro.read()

# 2. Make edits
updated = content + "\n\nNew section..."
intro.write(updated)

# 3. Review changes
print(intro.diff())

# 4. Commit when satisfied
if intro.diff():
    intro.commit("Add new section")
else:
    print("No changes to commit")

# 5. View history
print(intro.history())
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

## Project Structure

Writer creates and manages the following directory structure:

```
project_dir/
├── 01_manuscript/          # Main manuscript
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── methods.tex
│   ├── results.tex
│   ├── discussion.tex
│   └── main.tex
├── 02_supplementary/       # Supplementary materials
│   ├── figures/
│   └── tables/
├── 03_revision/            # Revision/response documents
└── .git/                   # Git repository (if using 'child' or 'parent' strategy)
```

## Error Handling

Writer handles errors gracefully with clear logging:

```python
try:
    writer = Writer("/path/to/project")
except RuntimeError as e:
    # Missing required directories (01_manuscript, 02_supplementary, 03_revision)
    print(f"Invalid project structure: {e}")

try:
    result = writer.compile_manuscript()
    if not result.success:
        print(f"Compilation failed: {result.errors}")
except Exception as e:
    print(f"Compilation error: {e}")
```

**Common Issues:**
- Invalid project structure: Ensure all 3 required directories exist
- Git initialization failure: Check git installation and permissions
- LaTeX compilation error: Check .tex files syntax and LaTeX installation

## Testing

Run the comprehensive test suite:

```bash
# Run all Writer tests
python -m pytest src/scitex/writer/tests/test_writer_integration.py -v

# Run specific test class
python -m pytest src/scitex/writer/tests/test_writer_integration.py::TestProjectAttachment -v

# Run with coverage
python -m pytest src/scitex/writer/tests/test_writer_integration.py --cov=scitex.writer
```

**Test Coverage:**
- Project attachment and creation
- Structure validation
- Git strategy handling (child, parent, None)
- Project name handling
- Child git cleanup for parent strategy

## Recent Improvements (2025-10-28 - Continued)

### Bug Fixes
- Fixed incomplete initialization that left `git_root` uninitialized
- Removed blocking debug code (`ipdb.set_trace()`)
- Fixed return type annotation in `_attach_or_create_project()`
- Improved error handling with proper logging throughout

### New Features
- **Structure Validation**: Automatically verifies project has required directories (01_manuscript, 02_supplementary, 03_revision) when attaching
- **Child Git Cleanup**: Automatically removes project's `.git/` when using `'parent'` strategy and parent repo is found, preventing nested git issues

### Enhanced Logging
- Detailed initialization logs with project name, directory, and git strategy
- Clear strategy selection and progression messages
- Explicit error messages with full context
- Success confirmations for key operations

### Enhanced diff() Capability
- **New `diff_between()` method** for comparing any two arbitrary git states
  - Compare commits: `diff_between("HEAD~2", "HEAD")`
  - Compare releases: `diff_between("v1.0", "v2.0")`
  - Compare branches: `diff_between("main", "develop")`
  - **Time-aware ref resolution**: `diff_between("1 week ago", "now")`
  - **Timestamp-based queries**: `diff_between("2025-10-20", "2025-10-28")`
  - Supports human-readable specifications without breaking git functionality
- **Reference resolution** (`_resolve_ref()`) handles:
  - Standard git refs (HEAD, branches, tags, commit hashes)
  - Relative time: "2 days ago", "1 week ago", "24 hours ago"
  - Absolute dates: "2025-10-28", "2025-10-28 14:30"
  - Timestamp-based commit finding with `git log --before`
- 15 new tests for diff_between functionality (all passing)

### Comprehensive Testing
- 99 tests covering all Writer and DocumentSection functionality
- DocumentSection operations fully tested (read/write/commit/history/diff/diff_between/checkout)
- End-to-end workflow testing (project creation → document editing → git operations)
- Temporal queries and reference resolution tested
- Error handling and edge cases covered
- All tests passing ✅

**Test Coverage:**
- Writer initialization and configuration
- Project creation and attachment
- Git strategy handling (child, parent, None)
- Document section operations (read, write, commit)
- Git operations (history, diff, checkout)
- Tree structure verification
- Error handling

**Run tests:**
```bash
pytest src/scitex/writer/tests/ -v
```

## Requirements

- Python 3.8+
- LaTeX distribution
- Git (for version control)

<!-- EOF -->