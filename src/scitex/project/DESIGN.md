<!-- ---
!-- Timestamp: 2025-10-24 01:24:37
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/project/DESIGN.md
!-- --- -->

# SciTeX Project Management Architecture

## Design Principles

### 1. Standalone First
The `scitex` package must work independently without Django or Gitea:
- Pure Python dataclasses and functions
- Local filesystem-based metadata storage
- No external service dependencies for core functionality
- Optional integrations via adapters

### 2. Self-Contained Projects
Each project is like a GitHub repository:
- Complete metadata in `scitex/` directory
- Git repository (optional but recommended)
- No external database required
- Portable across systems

### 3. One-to-One Relationships
When integrated with cloud services:
- **Local Project** ↔ **Django Project** ↔ **Gitea Repository**
- Each layer maintains strict 1:1 mapping
- Synchronization via unique identifiers

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Layer 3: Cloud Services (Optional)        │
│  - Django ORM models (web interface, auth, collaboration)   │
│  - Gitea API (remote git hosting, code review)              │
│  - Uses Layer 1 via adapters                                 │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ Adapters
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Layer 2: CLI & SDK (scitex package)             │
│  - Command-line interface                                    │
│  - API for programmatic access                               │
│  - Uses Layer 1                                              │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│            Layer 1: Core (Pure Python, No Dependencies)      │
│  - SciTeXProject dataclass                                   │
│  - Metadata persistence (scitex/)                           │
│  - Validation & naming logic                                 │
│  - Storage calculations                                      │
│  - Directory structure management                            │
└─────────────────────────────────────────────────────────────┘
```

## Layer 1: Core (This Implementation)

### Directory Structure

```
neural-decoding/                  # Project root (name = directory name)
├── .git/                         # Git repository (optional)
├── scitex/                       # SciTeX directory (visible, user-facing)
│   ├── .metadata/                # Hidden metadata subdirectory
│   │   ├── config.json           # Project configuration (project_id, created_at)
│   │   ├── metadata.json         # Project metadata (name, description, tags)
│   │   ├── integrations.json    # Cloud integration info (Django, Gitea, GitHub)
│   │   └── history.jsonl         # Activity log (JSONL format)
│   ├── scholar/                  # Literature management (created on-demand)
│   │   ├── bibliography.bib      # Enriched citations
│   │   └── papers/               # Downloaded PDFs
│   ├── writer/                   # LaTeX compilation (created on-demand)
│   │   ├── build/                # Compilation artifacts
│   │   └── templates/            # Document templates
│   ├── code/                     # Analysis tracking (created on-demand)
│   │   ├── experiments/          # Experiment metadata
│   │   └── results/              # Analysis outputs
│   └── viz/                      # Visualizations (created on-demand)
│       ├── figures/              # Generated figures
│       └── plots/                # Interactive plots
├── data/                         # User's data (from template)
├── scripts/                      # User's scripts
├── docs/                         # User's documentation
└── README.md                     # User's README
```

**Why `scitex/` is visible (not hidden):**
- Users need to see and interact with `scitex/scholar/`, `scitex/writer/`, etc.
- Makes SciTeX feature directories discoverable
- Consistent with user's mental model of their project structure
- Only `scitex/.metadata/` is hidden (like `.git/`)

### Metadata Files

#### `scitex/.metadata/config.json`
```json
{
  "project_id": "proj_abc123xyz",
  "version": "1.0.0",
  "created_at": "2025-10-24T01:00:00Z",
  "scitex_version": "0.1.0"
}
```

#### `scitex/.metadata/metadata.json`
```json
{
  "name": "neural-decoding",
  "slug": "neural-decoding",
  "description": "Deep learning for neural signal decoding",
  "owner": "ywatanabe",
  "visibility": "private",
  "template": "research",
  "tags": ["neuroscience", "deep-learning", "python"],
  "updated_at": "2025-10-24T01:00:00Z",
  "last_activity": "2025-10-24T01:00:00Z",
  "storage_used": 1048576
}
```

#### `scitex/.metadata/integrations.json` (Optional)
```json
{
  "cloud": {
    "enabled": true,
    "django_project_id": 42,
    "api_url": "https://scitex.cloud/api/v1/projects/42/",
    "synced_at": "2025-10-24T01:00:00Z"
  },
  "gitea": {
    "enabled": true,
    "repo_id": 123,
    "repo_name": "neural-decoding-research",
    "owner": "ywatanabe",
    "clone_url": "https://gitea.scitex.cloud/ywatanabe/neural-decoding-research.git",
    "synced_at": "2025-10-24T01:00:00Z"
  },
  "github": {
    "enabled": false
  }
}
```

#### `scitex/.metadata/history.jsonl` (Activity Log)
```jsonl
{"timestamp": "2025-10-24T01:00:00Z", "action": "created", "user": "ywatanabe"}
{"timestamp": "2025-10-24T01:05:00Z", "action": "storage_calculated", "size": 1048576}
{"timestamp": "2025-10-24T01:10:00Z", "action": "cloud_sync", "status": "success"}
```

## Core Classes

### SciTeXProject (Dataclass)

**Responsibilities:**
- Project metadata management
- Validation and naming rules
- Local persistence (read/write `scitex/` files)
- Storage calculation
- Activity tracking

**Not Responsible For:**
- Django ORM operations
- Gitea API calls
- Network operations
- User authentication

### ProjectValidator

**Responsibilities:**
- Name validation (GitHub/Gitea compatible)
- Slug generation
- Directory name sanitization
- URL parsing

### ProjectMetadataStore

**Responsibilities:**
- Read/write `scitex/` files
- JSON serialization
- Atomic file operations
- Backup/recovery

### ProjectStorageCalculator

**Responsibilities:**
- Directory size calculation
- Efficient tree traversal
- Ignore patterns (.git, scitex, etc.)

## Django Integration (Layer 3)

### How Django Uses SciTeXProject

```python
# In scitex-cloud/apps/project_app/models.py

from scitex.project import SciTeXProject, ProjectValidator

class Project(models.Model):
    # Django-specific fields
    id = models.AutoField(primary_key=True)
    owner = models.ForeignKey(User, ...)

    # Core fields (synced with SciTeXProject)
    name = models.CharField(max_length=200)
    slug = models.SlugField(...)
    description = models.TextField()

    # Integration fields
    scitex_project_id = models.CharField(max_length=100, unique=True)
    local_path = models.CharField(max_length=500)

    def to_scitex_project(self) -> SciTeXProject:
        """Convert Django model to SciTeXProject dataclass"""
        return SciTeXProject.load_from_directory(self.local_path)

    def sync_from_scitex(self):
        """Update Django model from local SciTeXProject"""
        scitex_proj = self.to_scitex_project()
        self.name = scitex_proj.name
        self.slug = scitex_proj.slug
        self.storage_used = scitex_proj.storage_used
        # ... sync other fields

    def sync_to_scitex(self):
        """Update local SciTeXProject from Django model"""
        scitex_proj = self.to_scitex_project()
        scitex_proj.name = self.name
        scitex_proj.slug = self.slug
        scitex_proj.save()  # Writes to scitex/

    def validate_name(self):
        """Use core validation logic"""
        is_valid, error = ProjectValidator.validate_repository_name(self.name)
        if not is_valid:
            raise ValidationError(error)
```

## Usage Examples

### Standalone (No Cloud)

```python
from scitex.project import SciTeXProject
from pathlib import Path

# Create new project
project = SciTeXProject.create(
    name="neural-decoding",
    path=Path("/home/user/projects/neural-decoding"),
    owner="ywatanabe",
    description="Deep learning for neural signal decoding",
    template="research",
    tags=["neuroscience", "deep-learning"]
)

# Load existing project
project = SciTeXProject.load_from_directory(
    Path("/home/user/projects/neural-decoding")
)

# Update metadata
project.description = "Updated: Multi-channel neural decoding"
project.tags.append("pytorch")
project.save()

# Calculate storage
project.update_storage_usage()
print(f"Storage used: {project.storage_used / 1024**2:.2f} MB")

# Validate name (using validator directly)
from scitex.project import validate_name
is_valid, error = validate_name("my-project-name")
if is_valid:
    print("Valid project name!")
```

### With Cloud Integration

```python
from scitex.project import SciTeXProject
from pathlib import Path
import os

# Create project locally
project = SciTeXProject.create(
    name="neural-decoding",
    path=Path("/home/user/projects/neural-decoding"),
    owner="ywatanabe",
    description="Deep learning for neural signal decoding"
)

# Cloud integration (future feature - via scitex.project.integrations)
# This would enable syncing with Django/Gitea
# cloud = CloudIntegration(
#     api_url="https://scitex.cloud",
#     api_key=os.environ["SCITEX_CLOUD_API_KEY"]
# )
# cloud.link_project(project, django_project_id=42)
# cloud.sync_to_cloud(project)
```

## Naming Rules (GitHub/Gitea Compatible)

### Valid Characters
- Alphanumeric: `a-z`, `A-Z`, `0-9`
- Special: `-`, `_`, `.`

### Constraints
- Length: 1-100 characters
- Cannot start/end with: `-`, `_`, `.`
- Cannot contain spaces
- Must not be empty

### Examples
```python
✓ "my-research-project"
✓ "neural_decoding_2025"
✓ "project.v2.0"
✓ "ML-Research"

✗ "my project"  # Space
✗ "-research"   # Starts with hyphen
✗ "project_"    # Ends with underscore
✗ "proj@2025"   # Invalid character
```

## Storage Calculation

### Included
- All user files
- `.git/` directory (repository size)
- `scitex/` directory (feature data)

### Excluded from UI Display (but counted)
- `scitex/` directory (metadata)
- Temporary files (`.tmp`, `__pycache__`)

### Implementation
```python
def calculate_storage(self) -> int:
    """Calculate total storage in bytes"""
    total = 0
    for path in self.path.rglob('*'):
        if path.is_file() and not self._should_ignore(path):
            total += path.stat().st_size
    return total

def _should_ignore(self, path: Path) -> bool:
    """Check if path should be ignored in calculations"""
    ignore_patterns = ['scitex', '__pycache__', '*.pyc', '.tmp']
    return any(path.match(pattern) for pattern in ignore_patterns)
```

## Migration Strategy

### Phase 1: Implement Core (This Phase)
1. ✓ Design architecture
2. Create `SciTeXProject` dataclass
3. Implement metadata persistence
4. Extract validation logic
5. Add storage calculation

### Phase 2: Django Integration
1. Add `scitex_project_id` field to Django Project model
2. Add `to_scitex_project()` / `sync_from_scitex()` methods
3. Update views to use core validation
4. Migrate existing projects to have `scitex/` directories

### Phase 3: CLI Enhancement
1. Add `scitex project init` command
2. Add `scitex project info` command
3. Add `scitex project sync` command (cloud)
4. Add `scitex project validate` command

## Testing Strategy

### Unit Tests (Pure Python)
- ProjectValidator: All naming rules
- SciTeXProject: CRUD operations
- ProjectMetadataStore: File I/O
- ProjectStorageCalculator: Size calculations

### Integration Tests
- Create → Load → Update → Delete workflow
- Template integration
- Cloud sync (mocked)

### End-to-End Tests (with Django)
- Django ↔ Local sync
- One-to-one relationship enforcement
- Migration of existing projects

## Benefits

### For Users
1. **Portability**: Projects work anywhere (local, cloud, containers)
2. **Offline-First**: No internet required for core functionality
3. **Transparency**: Metadata in readable JSON files
4. **Git-Friendly**: All metadata is version-controlled

### For Developers
1. **Testability**: Pure Python, easy to test
2. **Maintainability**: Clear separation of concerns
3. **Extensibility**: Easy to add new features
4. **Reusability**: Core logic shared across CLI, SDK, and Django

## Future Enhancements

1. **Project Templates**: Rich templates with custom structure
2. **Project Sync**: Multi-device synchronization
3. **Project Export**: Archive projects for sharing
4. **Project Analytics**: Usage statistics and insights
5. **Project Hooks**: Custom scripts on project events

<!-- EOF -->