<!-- Test Organization Plan for Writer Module -->

# Test Organization Plan

## Current State

**Total source modules:** 28
**Current test files:** 5 (flat structure)
**Test coverage:** 2/28 modules have dedicated tests

### Existing Tests
- `test_writer.py` - Tests Writer.py (38 tests)
- `test_writer_integration.py` - Integration tests for Writer (18 tests, 2 skipped)
- `test_document_section.py` - Tests DocumentSection (13 tests)
- `test_diff_between.py` - Tests DocumentSection.diff_between() (15 tests)
- `test_document_workflow.py` - End-to-end workflow tests (15 tests)

**Total:** 97 passing tests, 2 skipped

## Proposed Directory Structure

```
tests/scitex/writer/
├── __init__.py
├── run_test.sh
├── TODO.md
├── TEST_PLAN.md
│
├── test_writer.py                    # Keep (Writer class core tests)
├── test__integration.py              # Rename from test_writer_integration.py
│
├── dataclasses/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── test__DocumentSection.py  # Merge test_document_section.py + test_diff_between.py
│   │   └── test__Document.py         # NEW
│   │
│   ├── tree/
│   │   ├── __init__.py
│   │   ├── test__ManuscriptTree.py   # NEW
│   │   ├── test__SupplementaryTree.py # NEW
│   │   ├── test__RevisionTree.py     # NEW
│   │   ├── test__ScriptsTree.py      # NEW
│   │   ├── test__SharedTree.py       # NEW
│   │   └── test__ConfigTree.py       # NEW
│   │
│   ├── contents/
│   │   ├── __init__.py
│   │   ├── test__ManuscriptContents.py   # NEW
│   │   ├── test__SupplementaryContents.py # NEW
│   │   └── test__RevisionContents.py      # NEW
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── test__WriterConfig.py     # NEW
│   │   └── test__CONSTANTS.py        # NEW
│   │
│   └── results/
│       ├── __init__.py
│       ├── test__CompilationResult.py # NEW
│       └── test__LaTeXIssue.py       # NEW
│
├── compile/                          # Note: remove leading _ for test dirs
│   ├── __init__.py
│   ├── test__parser.py               # NEW
│   ├── test__runner.py               # NEW
│   └── test__validator.py            # NEW
│
├── project/
│   ├── __init__.py
│   ├── test__create.py               # NEW
│   ├── test__validate.py             # NEW (extract from test_document_workflow.py)
│   └── test__trees.py                # NEW
│
└── utils/
    ├── __init__.py
    ├── test__parse_latex_logs.py     # NEW
    ├── test__parse_script_args.py    # NEW
    └── test__watch.py                # NEW
```

## Migration Strategy

### Phase 1: Reorganize Existing Tests (Priority)
1. Create subdirectory structure
2. Move/merge existing test files:
   - Keep `test_writer.py` at root (tests Writer.py)
   - Rename `test_writer_integration.py` → `test__integration.py`
   - Merge `test_document_section.py` + `test_diff_between.py` → `dataclasses/core/test__DocumentSection.py`
   - Extract validation tests from `test_document_workflow.py` → `project/test__validate.py`
   - Keep workflow integration tests at root or in `test__integration.py`

### Phase 2: Create High-Priority New Tests
Focus on frequently-used modules:

**Priority 1 - Core Dataclasses:**
- `dataclasses/tree/test__ManuscriptTree.py`
- `dataclasses/tree/test__SupplementaryTree.py`
- `dataclasses/tree/test__RevisionTree.py`
- `dataclasses/config/test__WriterConfig.py`

**Priority 2 - Compilation:**
- `compile/test__parser.py`
- `compile/test__runner.py`
- `compile/test__validator.py`
- `dataclasses/results/test__CompilationResult.py`

**Priority 3 - Project Management:**
- `project/test__create.py`
- `project/test__trees.py`

### Phase 3: Create Lower-Priority Tests
- Contents dataclasses
- Utils modules
- Remaining tree classes

## Test Naming Convention

Following scitex conventions:
- Single underscore in source: `_DocumentSection.py` → `test__DocumentSection.py`
- Double underscore in source: `__init__.py` → `test____init__.py`
- Integration tests: `test__integration.py`

## Success Criteria

1. All existing 97 tests still pass after reorganization
2. Directory structure mirrors src/scitex/writer/
3. At least 15 core modules have dedicated tests (up from 2)
4. Test discovery works correctly with pytest
5. run_test.sh works with new structure

## Notes

- Keep integration tests separate from unit tests
- Use fixtures for common setup (temp directories, git repos)
- Mock external dependencies (git commands) in unit tests
- Use real git operations in integration tests
- Follow existing test patterns for consistency

<!-- EOF -->
