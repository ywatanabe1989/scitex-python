# Zotero Integration for SciTeX Scholar

Bidirectional integration with Zotero reference manager for seamless bibliography management.

## Features

### Import
- Bibliography with collections and tags
- PDF annotations and notes
- Complete paper metadata
- Batch import with progress tracking

### Export
- Export papers to Zotero collections
- Manuscripts as preprint entries
- Project metadata
- Citation files (.bib, .ris)

### Link
- Live synchronization
- Auto-update on library changes
- Citation insertion (BibTeX, RIS, formatted text)
- Tag-based organization

## Installation

```bash
pip install pyzotero
```

## Setup

1. Get your Zotero credentials:
   - **Library ID**: Found in Zotero settings or group URL
   - **API Key**: Generate at https://www.zotero.org/settings/keys

2. Set environment variables (optional):
   ```bash
   export ZOTERO_LIBRARY_ID="your_library_id"
   export ZOTERO_API_KEY="your_api_key"
   ```

## Usage

### Import from Zotero

```python
from scitex.scholar.zotero import ZoteroImporter

# Initialize importer
importer = ZoteroImporter(
    library_id="123456",
    library_type="user",  # or "group"
    api_key="your_api_key",
    project="my_research"
)

# Import a collection
papers = importer.import_collection(
    collection_name="Machine Learning",
    include_pdfs=True,
    include_annotations=True
)

# Import by tags
papers = importer.import_by_tags(
    tags=["deep learning", "NLP"],
    match_all=False  # OR logic
)

# Import all items
papers = importer.import_all(limit=100)

# Save to Scholar library
importer.import_to_library(papers)
```

### Export to Zotero

```python
from scitex.scholar.zotero import ZoteroExporter
from scitex.scholar.core import Papers

# Initialize exporter
exporter = ZoteroExporter(
    library_id="123456",
    library_type="user",
    api_key="your_api_key",
    project="my_research"
)

# Load papers from Scholar library
papers = Papers.from_project("my_research")

# Export to Zotero collection
results = exporter.export_papers(
    papers,
    collection_name="From SciTeX",
    create_collection=True,
    update_existing=True
)

# Export as BibTeX
exporter.export_as_bibtex(
    papers,
    output_path="./output/papers.bib"
)

# Export as RIS
exporter.export_as_ris(
    papers,
    output_path="./output/papers.ris"
)
```

### Live Synchronization

```python
from scitex.scholar.zotero import ZoteroLinker

# Initialize linker
linker = ZoteroLinker(
    library_id="123456",
    library_type="user",
    api_key="your_api_key",
    project="my_research",
    sync_interval=60  # seconds
)

# Register callback for sync events
def on_sync(event_type, paper):
    print(f"{event_type}: {paper.metadata.basic.title}")

linker.register_callback(on_sync)

# Start continuous sync
linker.start_sync(
    bidirectional=True,
    auto_import=True,
    auto_export=False
)

# Or perform single sync
stats = linker.sync_once()
print(f"Imported: {stats['imported']}, Exported: {stats['exported']}")
```

### Citation Insertion

```python
from scitex.scholar.zotero import ZoteroLinker
from scitex.scholar.core import Paper

linker = ZoteroLinker(...)
paper = Paper.from_file("path/to/metadata.json")

# BibTeX format
bibtex = linker.insert_citation(paper, format="bibtex")

# RIS format
ris = linker.insert_citation(paper, format="ris")

# Formatted text (APA, MLA, Chicago)
citation_apa = linker.insert_citation(paper, format="text", style="apa")
citation_mla = linker.insert_citation(paper, format="text", style="mla")
citation_chicago = linker.insert_citation(paper, format="text", style="chicago")
```

## Architecture

### Data Mapping

The `ZoteroMapper` class handles bidirectional conversion between:
- **Zotero format**: JSON items from Zotero API
- **Scholar format**: Pydantic Paper objects with nested metadata

Key mappings:
- Creators → Authors (with name parsing)
- Tags → Keywords
- Collections → Projects
- Attachments → PDF URLs/paths
- Annotations → Special fields

### Import/Export Flow

**Import Flow:**
```
Zotero API → ZoteroMapper.zotero_to_paper() → Paper → LibraryManager.save_resolved_paper()
```

**Export Flow:**
```
Paper → ZoteroMapper.paper_to_zotero() → Zotero API
```

### Synchronization

The `ZoteroLinker` monitors library versions and syncs changes:
- Tracks last sync version
- Polls Zotero for updates
- Converts and saves to Scholar library
- Notifies registered callbacks

## CLI Usage

```bash
# Import from Zotero
python -m scitex.scholar.zotero import \
    --library-id 123456 \
    --api-key YOUR_KEY \
    --collection "Machine Learning" \
    --project ml_research

# Export to Zotero
python -m scitex.scholar.zotero export \
    --library-id 123456 \
    --api-key YOUR_KEY \
    --project ml_research \
    --collection "From SciTeX"

# Start live sync
python -m scitex.scholar.zotero sync \
    --library-id 123456 \
    --api-key YOUR_KEY \
    --project ml_research \
    --interval 60
```

## Configuration

Store credentials in Scholar config:

```python
# ~/.scitex/scholar/config.json
{
    "zotero": {
        "library_id": "123456",
        "library_type": "user",
        "api_key": "your_api_key_here"
    }
}
```

Then use without explicit credentials:

```python
from scitex.scholar.zotero import ZoteroImporter
from scitex.scholar.config import ScholarConfig

config = ScholarConfig()
importer = ZoteroImporter(
    library_id=config.zotero.library_id,
    api_key=config.zotero.api_key,
    project="my_research"
)
```

## API Reference

### ZoteroImporter

```python
class ZoteroImporter:
    def __init__(library_id, library_type, api_key, project, config)
    def import_collection(collection_id, collection_name, limit, include_pdfs, include_annotations) -> Papers
    def import_by_tags(tags, match_all, limit, include_pdfs, include_annotations) -> Papers
    def import_all(limit, include_pdfs, include_annotations) -> Papers
    def import_to_library(papers, update_existing) -> Dict[str, str]
```

### ZoteroExporter

```python
class ZoteroExporter:
    def __init__(library_id, library_type, api_key, project, config)
    def export_papers(papers, collection_name, create_collection, update_existing) -> Dict[str, str]
    def export_as_bibtex(papers, output_path) -> Path
    def export_as_ris(papers, output_path) -> Path
```

### ZoteroLinker

```python
class ZoteroLinker:
    def __init__(library_id, library_type, api_key, project, config, sync_interval)
    def register_callback(callback: Callable[[str, Paper], None])
    def start_sync(bidirectional, auto_import, auto_export)
    def stop_sync()
    def sync_once(bidirectional, auto_import, auto_export) -> Dict[str, int]
    def insert_citation(paper, format, style) -> str
    def get_sync_status() -> Dict[str, Any]
```

### ZoteroMapper

```python
class ZoteroMapper:
    def __init__(config)
    def zotero_to_paper(zotero_item) -> Paper
    def paper_to_zotero(paper) -> Dict[str, Any]
```

## Troubleshooting

### Authentication Issues

If you get authentication errors:
1. Check your API key is correct
2. Verify library ID matches your Zotero account
3. Ensure API key has appropriate permissions

### Sync Not Working

If synchronization doesn't detect changes:
1. Check `sync_interval` is appropriate
2. Verify Zotero library version is updating
3. Check network connectivity

### Missing Dependencies

If pyzotero is not found:
```bash
pip install pyzotero
```

## Examples

See `/home/ywatanabe/proj/scitex_repo/src/scitex/scholar/examples/zotero_integration.py` for complete examples.

## License

Part of SciTeX Scholar module - see main project LICENSE.

## Author

Yusuke Watanabe (ywatanabe@scitex.ai)
