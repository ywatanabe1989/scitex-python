# Migration Guide: Zotero Module Relocation

## What Changed

The Zotero integration has been moved to the unified `integration` directory for better organization and consistency with other reference manager integrations.

### Old Structure
```
scitex/scholar/
├── zotero/
│   ├── __init__.py
│   ├── mapper.py
│   ├── importer.py
│   ├── exporter.py
│   └── linker.py
```

### New Structure
```
scitex/scholar/
├── integration/
│   ├── zotero/
│   │   ├── __init__.py
│   │   ├── mapper.py
│   │   ├── importer.py
│   │   ├── exporter.py
│   │   └── linker.py
│   ├── mendeley/
│   ├── refworks/
│   └── ...
```

## How to Migrate

### Update Import Statements

**Old (Deprecated):**
```python
from scitex.scholar.zotero import ZoteroImporter, ZoteroExporter, ZoteroLinker
```

**New (Recommended):**
```python
# Option 1: Import from unified integration
from scitex.scholar.integration import (
    ZoteroImporter,
    ZoteroExporter,
    ZoteroLinker,
)

# Option 2: Import from specific zotero submodule
from scitex.scholar.integration.zotero import (
    ZoteroImporter,
    ZoteroExporter,
    ZoteroLinker,
)
```

### Backward Compatibility

The old import path still works and will continue to work for backward compatibility:

```python
# This still works, but shows deprecation warning
from scitex.scholar.zotero import ZoteroImporter
```

**Deprecation Warning:**
```
DeprecationWarning: Importing from scitex.scholar.zotero is deprecated.
Please use: from scitex.scholar.integration.zotero import ...
```

## Code Examples

### Before (Old)
```python
from scitex.scholar.zotero import ZoteroImporter

importer = ZoteroImporter(
    library_id="123456",
    api_key="your_key"
)
papers = importer.import_collection("ML Papers")
```

### After (New)
```python
from scitex.scholar.integration.zotero import ZoteroImporter

importer = ZoteroImporter(
    library_id="123456",
    api_key="your_key"
)
papers = importer.import_collection("ML Papers")
```

**Note**: Only the import statement changes. All class names, methods, and functionality remain identical.

## CLI Migration

### Old
```bash
python -m scitex.scholar.zotero import --collection "ML"
```

### New
```bash
# Both still work, new path recommended
python -m scitex.scholar.integration.zotero import --collection "ML"
```

## Benefits of New Structure

1. **Unified Organization**: All reference managers in one place
2. **Consistent API**: Same pattern for Zotero, Mendeley, etc.
3. **Easier Discovery**: Users can find all integrations easily
4. **Shared Infrastructure**: Base classes in `integration/base.py`
5. **Better Documentation**: Single comprehensive README

## Migration Timeline

- **Now**: Both paths work (old path shows deprecation warning)
- **Next Release**: Old path continues to work
- **Future**: Old path may be removed (with advance notice)

**Recommendation**: Update your imports at your convenience. There's no rush, but updating now prevents future warnings.

## Need Help?

If you have issues migrating:
1. Check that `integration/zotero/` exists
2. Verify imports resolve correctly
3. Run tests to ensure functionality unchanged

## Author

Yusuke Watanabe (ywatanabe@scitex.ai)
