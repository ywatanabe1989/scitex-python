<!-- ---
!-- Timestamp: 2025-10-29 09:34:33
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/EVAL.md
!-- --- -->

## ⚠️ Areas for Improvement

### 1. Inconsistent Delegation in scitex.git

**Problem:** The git module has unclear boundaries between modules.

```python
# git/_branch.py - Does TOO MUCH
def git_branch_rename(repo_path: Path, new_name: str, verbose: bool = True) -> bool:
    # ❌ Validation here
    valid, error = validate_path(repo_path, must_exist=True)

    # ❌ Git check here
    if not (repo_path / ".git").exists():

    # ❌ Name validation here
    valid, error = validate_branch_name(new_name)

    # ❌ Directory context here
    with _in_directory(repo_path):

    # ✅ Finally, the actual operation
    result = sh(["git", "branch", "-M", new_name], ...)
```

**Better approach (like scitex.sh):**

```python
# git/_operations.py (new file)
def _execute_git_command(repo_path: Path, args: List[str], verbose: bool) -> dict:
    """Execute git command with standard error handling."""
    with _in_directory(repo_path):
        return sh(args, verbose=verbose, return_as="dict")

# git/_branch.py (simplified)
def git_branch_rename(repo_path: Path, new_name: str, verbose: bool = True) -> bool:
    validate_git_repo(repo_path)  # Delegate to _validation.py
    validate_branch_name(new_name)  # Already exists

    result = _execute_git_command(
        repo_path,
        ["git", "branch", "-M", new_name],
        verbose
    )

    if not result["success"]:
        logger.error(f"Failed to rename branch: {result['stderr']}")
        return False

    logger.info(f"Branch renamed to {new_name}")
    return True
```

### 2. Repetitive Validation Code

**Current state - Same validation in every function:**

```python
# Repeated in _branch.py, _commit.py, _remote.py, _workflow.py:
valid, error = validate_path(repo_path, must_exist=True)
if not valid:
    logger.error(error)
    return False

if not (repo_path / ".git").exists():
    logger.error(f"Not a git repository: {repo_path}")
    return False
```

**Better approach:**

```python
# git/_validation.py (enhanced)
def validate_git_repo(repo_path: Path) -> None:
    """Validate path is a git repository. Raises ValueError if not."""
    valid, error = validate_path(repo_path, must_exist=True)
    if not valid:
        raise ValueError(error)

    if not (repo_path / ".git").exists():
        raise ValueError(f"Not a git repository: {repo_path}")

# Then in operations:
def git_branch_rename(repo_path: Path, new_name: str, verbose: bool = True) -> bool:
    try:
        validate_git_repo(repo_path)  # Single line!
        validate_branch_name(new_name)
    except ValueError as e:
        logger.error(str(e))
        return False

    # ... rest of function
```

### 3. **`scitex.writer` Has Over-Delegation**

The writer module has **too many** small files with unclear responsibilities:
```
writer/
├── _compile/
│   ├── _runner.py       # Runs scripts
│   ├── _parser.py       # Parses output
│   └── _validator.py    # Validates structure
├── _git/
│   ├── _operations.py   # Git operations
│   ├── _retry.py        # Just re-exports git_retry!
│   └── _strategy.py     # Git strategy
├── _project/
│   ├── _create.py       # Creates projects
│   ├── _trees.py        # Creates tree objects
│   └── _validate.py     # Validates structure
**Problem:** Too many layers of indirection. For example:

```python
# writer/_git/_retry.py - Entire file:
from scitex.git import git_retry
__all__ = ["git_retry"]
# That's it! Why does this file exist?
```

**Better approach:** Consolidate related functionality:

```python
# writer/_compile.py (single file instead of _compile/ directory)
def compile_manuscript(project_dir: Path, timeout: int = 300) -> CompilationResult:
    """Compile manuscript with validation, execution, and parsing."""
    _validate_project_structure(project_dir)  # Private function
    result = _run_compile_script(project_dir, "manuscript", timeout)  # Private
    return _parse_compilation_result(result)  # Private

# Only expose public functions at module level
```

### 4. Inconsistent Error Handling

**Current mix:**

```python
# Some functions return bool
def git_init(repo_path: Path) -> bool:
    if something_wrong:
        return False

# Others return Optional
def get_remote_url(repo_path: Path) -> Optional[str]:
    if something_wrong:
        return None

# Writer raises exceptions
def validate_structure(project_dir: Path) -> None:
    if invalid:
        raise RuntimeError("Invalid structure")
```

**Better approach - Pick ONE pattern per module:**

```python
# Option A: Result objects (best for complex operations)
@dataclass
class GitResult:
    success: bool
    error: Optional[str] = None
    value: Optional[Any] = None

def git_init(repo_path: Path) -> GitResult:
    if problem:
        return GitResult(success=False, error="Problem description")
    return GitResult(success=True, value=repo_path)

# Option B: Exceptions (best for validation/setup)
def validate_structure(project_dir: Path) -> None:
    """Raises ValueError if invalid."""
    if not valid:
        raise ValueError("Specific error message")
```

### 5. Missing Facade Pattern for Complex Operations
The Writer class directly accesses many low-level operations. Consider a facade:
python# writer/_facade.py
class WriterOperations:
    """High-level operations facade."""
    
    def __init__(self, project_dir: Path, git_root: Optional[Path]):
        self.project_dir = project_dir
        self.git_root = git_root
    
    def compile_document(self, doc_type: str, **options) -> CompilationResult:
        """One-stop compilation with all necessary steps."""
        self._validate_structure()
        script = self._get_compile_script(doc_type)
        result = self._execute_script(script, **options)
        return self._parse_result(result)
    
    def commit_section(self, section_path: Path, message: str) -> bool:
        """One-stop commit with retry logic."""
        return git_retry(lambda: self._do_commit(section_path, message))

# Then Writer becomes simpler:
class Writer:
    def __init__(self, project_dir: Path, ...):
        self._ops = WriterOperations(project_dir, git_root)
        # ...
    
    def compile_manuscript(self, timeout: int = 300) -> CompilationResult:
        return self._ops.compile_document("manuscript", timeout=timeout)

📋 Specific Recommendations
For scitex.git:

Create git/_operations.py - Common execution logic
Enhance git/_validation.py - Centralized validation with exceptions
Simplify public functions - They should just orchestrate, not implement

python# Pattern for all git operations:
def git_operation(repo_path: Path, args...) -> bool:
    try:
        validate_git_repo(repo_path)     # Delegate validation
        validate_specific_args(args)      # Delegate validation
    except ValueError as e:
        logger.error(str(e))
        return False
    
    result = execute_git_command(        # Delegate execution
        repo_path, 
        ["git", "operation", *args]
    )
    
    return handle_git_result(result)     # Delegate result handling
For scitex.writer:

Consolidate over-split modules:

_compile/ → single _compile.py file
_git/ → remove (use scitex.git directly)
_project/ → merge _create.py and _validate.py


Create facade for common workflows
Standardize on exceptions for validation (you're already moving this direction)

For scitex.template:

Already well-structured! The _clone_project.py → delegates to:

_copy.py - File operations
_rename.py - Package renaming
_customize.py - Reference updates
_git_strategy.py - Git handling

This is excellent delegation. Use this as the model for other modules.


🎯 Summary
What's working well:

scitex.sh - Exemplary module structure ⭐
scitex.template - Good delegation pattern ⭐
Type safety throughout
Security-first design

What needs improvement:

scitex.git - Too much logic in public functions, needs better delegation
scitex.writer - Over-split into too many small files
Inconsistent error handling patterns
Repetitive validation code

Core principle to follow:

Each public function should be 90% delegation, 10% orchestration

pythondef public_operation(args):
    """This should be SHORT and READ like plain English."""
    validate_inputs(args)              # Delegate
    prepared = prepare_data(args)       # Delegate
    result = execute_operation(prepared)  # Delegate
    return handle_result(result)        # Delegate
Would you like me to provide refactored versions of specific modules following these principles?

<!-- EOF -->