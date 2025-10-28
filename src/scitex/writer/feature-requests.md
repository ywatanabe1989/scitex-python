<!-- ---
!-- Timestamp: 2025-10-28 15:54:13
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/feature-requests.md
!-- --- -->

## Issues Found in scitex-cloud integration

### 1. Compilation Script Discovery (CRITICAL)
**Problem**: `_find_compile_script()` in `compile.py` looks in:
- `/tmp/scitex-writer/compile`
- `~/proj/scitex-writer/compile`
- `__file__.parent / "scripts" / "compile"`

But the actual scitex-writer template is at:
- `~/proj/scitex-code/my_paper/compile` ← This location is never checked!

**Result**: Compilation always fails with "scitex-writer compile script not found"

**Solution**: Add `~/proj/scitex-code/my_paper` to the search locations in `_find_compile_script()`:
```python
locations = [
    Path("/tmp/scitex-writer/compile"),
    Path.home() / "proj" / "scitex-writer" / "compile",
    Path.home() / "proj" / "scitex-code" / "my_paper" / "compile",  # ← ADD THIS
    Path(__file__).parent / "scripts" / "compile",
]
```

Or use environment variable fallback:
```python
locations = [
    Path(os.getenv("SCITEX_WRITER_TEMPLATE_PATH", "")) / "compile" if os.getenv("SCITEX_WRITER_TEMPLATE_PATH") else None,
    Path("/tmp/scitex-writer/compile"),
    Path.home() / "proj" / "scitex-code" / "my_paper" / "compile",
    Path.home() / "proj" / "scitex-writer" / "compile",
    Path(__file__).parent / "scripts" / "compile",
]
locations = [l for l in locations if l]  # Filter None
```

### 2. Git Strategy Graceful Degradation
- [ ] git_strategy="parent" should be degraded to "child" (the writer directory itself) when no parent .git found
- This prevents errors in django when initializing writer directory for projects without parent git repo

<!-- EOF -->