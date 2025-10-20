# Reference Manager Integrations for SciTeX Scholar

Unified integration framework for multiple reference management systems with consistent API across all platforms.

## Supported Reference Managers

| Manager | Status | Import | Export | Sync | Notes |
|---------|--------|--------|--------|------|-------|
| **Zotero** | ✅ Complete | ✅ | ✅ | ✅ | Full implementation with pyzotero |
| **Mendeley** | ✅ Complete | ✅ | ✅ | ✅ | Uses mendeley-python-sdk |
| **RefWorks** | 🔄 Template | ⚠️ | ⚠️ | ⚠️ | API access required |
| **Paperpile** | 🔄 Template | ⚠️ | ⚠️ | ⚠️ | BibTeX/web API |
| **EndNote** | 🔄 Template | ⚠️ | ⚠️ | ⚠️ | XML format support |
| **CiteDrive** | 🔄 Template | ⚠️ | ⚠️ | ⚠️ | BibTeX-based |
| **Papers** | 🔄 Template | ⚠️ | ⚠️ | ⚠️ | macOS integration |

## Architecture

All reference manager integrations follow the same pattern using base classes:

```python
from scitex.scholar.integration.base import (
    BaseImporter,     # Import from external system
    BaseExporter,     # Export to external system
    BaseLinker,       # Live synchronization
    BaseMapper,       # Data format conversion
)
```

### Common Features

Every integration provides:

1. **Import**
   - Import collections/folders
   - Filter by tags
   - Include PDFs and annotations
   - Batch processing with progress

2. **Export**
   - Export to collections/folders
   - BibTeX format
   - RIS format
   - Update existing items

3. **Link**
   - Live synchronization
   - Citation insertion (BibTeX/RIS/formatted text)
   - APA/MLA/Chicago styles
   - Callback system for events

## Installation

### Zotero
```bash
pip install pyzotero
```

### Mendeley
```bash
pip install mendeley
```

### Others
Most use standard file formats (BibTeX, RIS) or REST APIs.

## Usage

### Unified Import (Recommended)

```python
# Import all reference managers from unified location
from scitex.scholar.integration import (
    ZoteroImporter, ZoteroExporter, ZoteroLinker,
    MendeleyImporter, MendeleyExporter, MendeleyLinker,
)
```

Or import individually:

```python
from scitex.scholar.integration.zotero import ZoteroImporter
from scitex.scholar.integration.mendeley import MendeleyImporter
```

### Zotero

```python
from scitex.scholar.integration.zotero import (
    ZoteroImporter,
    ZoteroExporter,
    ZoteroLinker
)

# Import
importer = ZoteroImporter(
    library_id="123456",
    api_key="your_key",
    project="research"
)
papers = importer.import_collection("Machine Learning")

# Export
exporter = ZoteroExporter(library_id="123456", api_key="your_key")
exporter.export_papers(papers, collection_name="From SciTeX")

# Sync
linker = ZoteroLinker(library_id="123456", api_key="your_key")
linker.start_sync(auto_import=True)
```

**Backward Compatibility Note**: The old import path `from scitex.scholar.zotero import ...` still works but will show a deprecation warning. Please update to the new path.

### Mendeley

```python
from scitex.scholar.integration.mendeley import (
    MendeleyImporter,
    MendeleyExporter,
    MendeleyLinker
)

# Import
importer = MendeleyImporter(
    app_id="your_app_id",
    app_secret="your_secret",
    access_token="your_token",
    project="research"
)
papers = importer.import_collection(collection_name="Deep Learning")

# Export
exporter = MendeleyExporter(
    app_id="your_app_id",
    app_secret="your_secret",
    access_token="your_token"
)
exporter.export_papers(papers, collection_name="From SciTeX")

# Sync
linker = MendeleyLinker(
    app_id="your_app_id",
    app_secret="your_secret",
    access_token="your_token",
    sync_interval=60
)
linker.start_sync(auto_import=True)
```

### RefWorks

```python
from scitex.scholar.integration.refworks import (
    RefWorksImporter,
    RefWorksExporter
)

# Import (using RIS/BibTeX files)
importer = RefWorksImporter(project="research")
papers = importer.import_from_file("refworks_export.ris")

# Export
exporter = RefWorksExporter()
exporter.export_as_ris(papers, "for_refworks.ris")
```

### Paperpile

```python
from scitex.scholar.integration.paperpile import (
    PaperpileImporter,
    PaperpileExporter
)

# Import from BibTeX
importer = PaperpileImporter(project="research")
papers = importer.import_from_bibtex("paperpile_library.bib")

# Export
exporter = PaperpileExporter()
exporter.export_as_bibtex(papers, "for_paperpile.bib")
```

### EndNote

```python
from scitex.scholar.integration.endnote import (
    EndNoteImporter,
    EndNoteExporter
)

# Import from XML
importer = EndNoteImporter(project="research")
papers = importer.import_from_xml("endnote_library.xml")

# Export
exporter = EndNoteExporter()
exporter.export_as_xml(papers, "for_endnote.xml")
exporter.export_as_ris(papers, "for_endnote.ris")
```

### CiteDrive

```python
from scitex.scholar.integration.citedrive import (
    CiteDriveImporter,
    CiteDriveExporter
)

# Import
importer = CiteDriveImporter(
    api_key="your_key",
    project="research"
)
papers = importer.import_collection("My Papers")

# Export
exporter = CiteDriveExporter(api_key="your_key")
exporter.export_papers(papers)
```

### Papers (macOS)

```python
from scitex.scholar.integration.papers import (
    PapersImporter,
    PapersExporter
)

# Import from Papers library
importer = PapersImporter(
    library_path="~/Library/Application Support/Papers3",
    project="research"
)
papers = importer.import_collection("Research")

# Export
exporter = PapersExporter()
exporter.export_as_bibtex(papers, "for_papers.bib")
```

## Unified CLI

All reference managers can be accessed through a unified CLI:

```bash
# Import
python -m scitex.scholar.refman import zotero \
    --library-id 123456 --api-key YOUR_KEY \
    --collection "ML Papers" --project research

# Export
python -m scitex.scholar.refman export mendeley \
    --app-id YOUR_ID --app-secret YOUR_SECRET \
    --project research --collection "From SciTeX"

# Sync
python -m scitex.scholar.refman sync zotero \
    --library-id 123456 --api-key YOUR_KEY \
    --interval 60 --auto-import
```

## Configuration

Store credentials in Scholar config:

```json
{
    "integration": {
        "zotero": {
            "library_id": "123456",
            "api_key": "your_key"
        },
        "mendeley": {
            "app_id": "your_id",
            "app_secret": "your_secret"
        },
        "paperpile": {
            "api_key": "your_key"
        }
    }
}
```

## Implementation Guide

To add a new reference manager:

1. **Create mapper** (convert formats):
```python
from scitex.scholar.integration.base import BaseMapper

class MyRefManMapper(BaseMapper):
    def external_to_paper(self, item):
        # Convert external format to Paper
        pass

    def paper_to_external(self, paper):
        # Convert Paper to external format
        pass
```

2. **Create importer**:
```python
from scitex.scholar.integration.base import BaseImporter

class MyRefManImporter(BaseImporter):
    def _create_mapper(self):
        return MyRefManMapper()

    def import_collection(self, ...):
        # Fetch items from external system
        # Convert using mapper
        # Return Papers
        pass
```

3. **Create exporter**:
```python
from scitex.scholar.integration.base import BaseExporter

class MyRefManExporter(BaseExporter):
    def _create_mapper(self):
        return MyRefManMapper()

    def export_papers(self, papers, ...):
        # Convert using mapper
        # Push to external system
        pass
```

4. **Create linker** (optional for live sync):
```python
from scitex.scholar.integration.base import BaseLinker

class MyRefManLinker(BaseLinker):
    def _create_importer(self):
        return MyRefManImporter(...)

    def _create_exporter(self):
        return MyRefManExporter(...)

    def _create_mapper(self):
        return MyRefManMapper()
```

## API Reference

### BaseImporter

```python
class BaseImporter:
    def __init__(credentials, project, config)
    def import_collection(collection_id, collection_name, limit, **kwargs) -> Papers
    def import_by_tags(tags, match_all, limit, **kwargs) -> Papers
    def import_all(limit, **kwargs) -> Papers
    def import_to_library(papers, update_existing) -> Dict[str, str]
```

### BaseExporter

```python
class BaseExporter:
    def __init__(credentials, project, config)
    def export_papers(papers, collection_name, **kwargs) -> Dict[str, str]
    def export_as_bibtex(papers, output_path) -> Path
    def export_as_ris(papers, output_path) -> Path
```

### BaseLinker

```python
class BaseLinker:
    def __init__(credentials, project, config, sync_interval)
    def register_callback(callback: Callable[[str, Paper], None])
    def insert_citation(paper, format, style) -> str
```

### BaseMapper

```python
class BaseMapper:
    def __init__(config)
    def external_to_paper(item) -> Paper
    def paper_to_external(paper) -> Dict
```

## Data Mapping

All mappers handle:
- **Authors**: Various name formats → "Last, First"
- **Dates**: Various date formats → year (int)
- **Identifiers**: DOI, PMID, arXiv, etc.
- **Tags/Keywords**: Platform-specific → list
- **Collections**: Platform-specific → Scholar projects
- **Annotations**: Platform-specific → special fields

Source tracking is preserved for all fields using `_engines` lists.

## File Formats

### BibTeX Import/Export
All managers support BibTeX as a universal interchange format:
```python
# Import from any BibTeX file
papers = importer.import_from_bibtex("library.bib")

# Export for any system
exporter.export_as_bibtex(papers, "export.bib")
```

### RIS Import/Export
Standard RIS format for systems without APIs:
```python
papers = importer.import_from_ris("library.ris")
exporter.export_as_ris(papers, "export.ris")
```

## Troubleshooting

### Authentication Issues
- Check credentials are correct
- Verify API keys have appropriate permissions
- For OAuth systems, ensure tokens are not expired

### Missing Dependencies
```bash
# Zotero
pip install pyzotero

# Mendeley
pip install mendeley
```

### Rate Limiting
Most APIs have rate limits:
- Add delays between requests
- Use batch operations when available
- Cache responses when possible

## Examples

See `/home/ywatanabe/proj/scitex_repo/src/scitex/scholar/examples/`:
- `zotero_integration.py` - Complete Zotero examples
- `mendeley_integration.py` - Mendeley examples
- `refman_comparison.py` - Compare multiple managers

## Contributing

To add support for a new reference manager:

1. Create directory: `integration/yourmanager/`
2. Implement: `mapper.py`, `importer.py`, `exporter.py`, `linker.py`
3. Follow existing patterns (see Zotero or Mendeley)
4. Add tests and documentation
5. Update this README

## License

Part of SciTeX Scholar module - see main project LICENSE.

## Author

Yusuke Watanabe (ywatanabe@scitex.ai)
