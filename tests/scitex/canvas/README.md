<!-- ---
!-- Timestamp: 2025-12-19
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/tests/scitex/fig/README.md
!-- --- -->

# Tests for scitex.fig

**Test coverage for the Unified Element API and coordinate system utilities**

---

## Test Files

### `test__element_api.py`

Tests for the `Figz` class and unified element API:

| Test Class | Coverage |
|------------|----------|
| `TestFigzCreate` | Bundle creation, size, extensions, constraints |
| `TestUnifiedElementAPI` | `add_element()`, `get_element()`, `list_element_ids()`, `remove_element()` |
| `TestElementCoordinates` | Position normalization, default values |
| `TestLegacyCompatibility` | Deprecated panel API, migration |
| `TestBundleRepr` | String representation |

**Key tests:**

```python
def test_add_text_element(figz)     # Basic text element
def test_add_shape_element(figz)    # Arrow/shape elements
def test_list_elements_by_type()    # Filter by element type
def test_update_element_position()  # Modify positions
def test_replace_element_same_id()  # ID collision handling
def test_add_panel_deprecated()     # Legacy API warnings
```

### `test__layout.py`

Tests for coordinate system and layout utilities:

| Test Class | Coverage |
|------------|----------|
| `TestNormalizePosition` | Position format conversion |
| `TestNormalizeSize` | Size format conversion |
| `TestToAbsolute` | Local → absolute transforms |
| `TestToRelative` | Absolute → local transforms |
| `TestElementBounds` | Bounding box extraction |
| `TestValidateWithinBounds` | Bounds checking |
| `TestAutoLayoutGrid` | Automatic grid layout |
| `TestCoordinateSystemIntegration` | End-to-end coordinate transforms |

**Key tests:**

```python
def test_with_parent_adds_offset()      # Basic coordinate transform
def test_nested_transform()             # Multi-level nesting
def test_inverse_of_to_absolute()       # Round-trip transform
def test_move_parent_moves_children()   # Parent movement propagation
def test_four_elements_grid()           # 2x2 grid layout
```

---

## Running Tests

```bash
# All fig tests
pytest tests/scitex/fig/ -v

# Specific test file
pytest tests/scitex/fig/test__element_api.py -v

# Specific test class
pytest tests/scitex/fig/test__layout.py::TestToAbsolute -v

# With coverage
pytest tests/scitex/fig/ --cov=src/scitex/fig --cov-report=term-missing
```

---

## Test Coverage Summary

| Module | Tests | Coverage |
|--------|-------|----------|
| `_bundle.py` | 20 | Unified element API, legacy compatibility |
| `layout.py` | 24 | Coordinate transforms, validation, auto-layout |

**Total: 44 tests**

---

## Test Fixtures

Common fixtures used across tests:

```python
@pytest.fixture
def figz(tmp_path):
    """Create a test Figz bundle."""
    bundle_path = tmp_path / "test.stx"
    return Figz.create(bundle_path, "Test Figure")
```

---

## Related Documentation

- [scitex.fig README](../../../src/scitex/fig/README.md) - API documentation
- [Examples](../../../examples/fig/) - Usage examples

<!-- EOF -->
