# SciTeX Project Management

Standalone project management for SciTeX research projects. Works independently without Django or Gitea.

## Features

- ✅ **Self-Contained**: Each project is a directory with `scitex/.metadata/` for metadata
- ✅ **Portable**: Projects work anywhere (local, cloud, containers)
- ✅ **Offline-First**: No internet required for core functionality
- ✅ **Git-Friendly**: Metadata in version-controlled JSON files
- ✅ **Name Validation**: GitHub/Gitea compatible naming rules
- ✅ **Storage Tracking**: Calculate project storage usage
- ✅ **Activity History**: Track project events in `scitex/.metadata/history.jsonl`

## Installation

```bash
pip install scitex
```

## Quick Start

### Create a New Project

```python
from scitex.project import SciTeXProject
from pathlib import Path

# Create project
project = SciTeXProject.create(
    name="neural-decoding",
    path=Path("/home/user/projects/neural-decoding"),
    owner="ywatanabe",
    description="Deep learning for neural signal decoding",
    visibility="private",
    tags=["neuroscience", "deep-learning"]
)

print(f"Created project: {project.project_id}")
# Output: Created project: proj_abc123xyz
print(f"Project slug: {project.slug}")
# Output: Project slug: neural-decoding
```

### Load Existing Project

```python
from scitex.project import SciTeXProject
from pathlib import Path

# Load from directory
project = SciTeXProject.load_from_directory(
    Path("/home/user/projects/neural-decoding")
)

print(f"{project.name} by {project.owner}")
# Output: neural-decoding by ywatanabe
print(f"Description: {project.description}")
# Output: Description: Deep learning for neural signal decoding
```

### Update Project Metadata

```python
# Load project
project = SciTeXProject.load_from_directory(
    Path("/home/user/projects/neural-decoding")
)

# Update description
project.description = "Updated: Deep learning for multi-channel neural decoding"

# Add tags
project.tags.append("pytorch")

# Save changes
project.save()

# Log activity
project.log_activity("description_updated", user="ywatanabe")
```

### Calculate Storage

```python
# Update storage usage
size = project.update_storage_usage()
print(f"Storage: {size / 1024**2:.2f} MB")
# Output: Storage: 42.50 MB

# Get detailed breakdown
breakdown = project.get_storage_breakdown()
print(f"User files: {breakdown['user_files'] / 1024**2:.2f} MB")
print(f"Git repo: {breakdown['git'] / 1024**2:.2f} MB")
```

### Validate Project Names

```python
from scitex.project import validate_name, generate_slug

# Validate name
is_valid, error = validate_name("my-research-project")
if is_valid:
    print("Valid name!")
else:
    print(f"Invalid: {error}")

# Generate slug from name
slug = generate_slug("My Research Project 2025")
print(slug)
# Output: my-research-project-2025

# Invalid examples
validate_name("my project")  # Error: Cannot contain spaces
validate_name("-invalid")     # Error: Cannot start with hyphen
validate_name("a" * 101)      # Error: Max 100 characters
```

### Work with SciTeX Feature Directories

```python
# Get scholar directory (creates if doesn't exist)
scholar_dir = project.get_scitex_directory('scholar')
bib_file = scholar_dir / 'bibliography.bib'

# Get writer directory
writer_dir = project.get_scitex_directory('writer')
build_dir = writer_dir / 'build'

# Available features: 'scholar', 'writer', 'code', 'viz', 'metadata'
```

### View Project History

```python
# Get last 10 activities
history = project.get_history(limit=10)

for entry in history:
    print(f"{entry['timestamp']}: {entry['action']}")
# Output:
# 2025-10-24T01:00:00Z: created
# 2025-10-24T01:05:00Z: storage_calculated
# 2025-10-24T01:10:00Z: description_updated
```

## Project Structure

When you create a SciTeX project, the directory looks like:

```
neural-decoding/                  # Project name = directory name
├── .git/                         # Git repository (optional)
├── scitex/                       # SciTeX directory (visible, user-facing)
│   ├── .metadata/                # Hidden metadata subdirectory
│   │   ├── config.json           # Project configuration
│   │   ├── metadata.json         # Project metadata
│   │   ├── integrations.json    # Cloud integration info
│   │   └── history.jsonl         # Activity log
│   ├── scholar/                  # Literature management (created on-demand)
│   ├── writer/                   # LaTeX compilation (created on-demand)
│   ├── code/                     # Analysis tracking (created on-demand)
│   └── viz/                      # Visualizations (created on-demand)
├── data/                         # Your data (from template)
├── scripts/                      # Your scripts
├── docs/                         # Your documentation
└── README.md                     # Your README
```

**Note:**
- The project name typically matches the directory name (e.g., `neural-decoding`)
- The `scitex/` directory is visible so users can interact with features like `scholar/`, `writer/`, etc.
- Only the `.metadata/` subdirectory is hidden (like `.git/`)

## API Reference

### SciTeXProject

Main project class.

#### `SciTeXProject.create(...)`

Create a new project.

```python
project = SciTeXProject.create(
    name="my-project",             # Required: project name (becomes slug)
    path=Path("/path/to/my-project"),  # Required: project directory
    owner="username",              # Required: owner username
    description="Description",     # Optional: project description
    visibility="private",          # Optional: 'public' or 'private'
    template="research",           # Optional: template used
    tags=["tag1", "tag2"],        # Optional: list of tags
    init_git=True                  # Optional: initialize git repo
)
```

#### `SciTeXProject.load_from_directory(path)`

Load existing project.

```python
project = SciTeXProject.load_from_directory(Path("/path/to/project"))
```

#### `project.save()`

Save metadata to `.scitex/`.

```python
project.description = "Updated"
project.save()
```

#### `project.update_storage_usage(include_git=True)`

Calculate and update storage usage.

```python
size_bytes = project.update_storage_usage()
```

#### `project.get_storage_breakdown()`

Get detailed storage breakdown by category.

```python
breakdown = project.get_storage_breakdown()
# Returns: {'total': int, 'git': int, 'scitex': int, 'user_files': int, 'breakdown': {...}}
```

#### `project.validate()`

Validate project metadata and structure.

```python
try:
    project.validate()
    print("Project is valid")
except ValueError as e:
    print(f"Validation error: {e}")
```

#### `project.init_git()`

Initialize git repository.

```python
success = project.init_git()
```

#### `project.is_git_repository()`

Check if project is a git repo.

```python
if project.is_git_repository():
    print("Git repo exists")
```

#### `project.get_scitex_directory(feature)`

Get path to SciTeX feature directory.

```python
scholar_dir = project.get_scitex_directory('scholar')
# Features: 'scholar', 'writer', 'code', 'viz', 'metadata'
```

#### `project.get_history(limit=None)`

Get project activity history.

```python
history = project.get_history(limit=10)
```

#### `project.log_activity(action, **kwargs)`

Log project activity.

```python
project.log_activity('experiment_completed', experiment_id='exp_001')
```

### Validators

#### `validate_name(name)`

Validate project name.

```python
from scitex.project import validate_name

is_valid, error = validate_name("my-project")
if not is_valid:
    print(error)
```

#### `generate_slug(name)`

Generate URL-safe slug.

```python
from scitex.project import generate_slug

slug = generate_slug("My Research Project")
# Returns: 'my-research-project'
```

#### `extract_repo_name(url)`

Extract repo name from Git URL.

```python
from scitex.project import extract_repo_name

name = extract_repo_name("https://github.com/user/my-repo.git")
# Returns: 'my-repo'
```

### Storage Utilities

#### `calculate_storage(path, include_git=True, include_metadata=False)`

Calculate storage for a directory.

```python
from scitex.project import calculate_storage
from pathlib import Path

size = calculate_storage(Path("/path/to/project"))
```

#### `format_size(bytes)`

Format bytes to human-readable string.

```python
from scitex.project import format_size

print(format_size(1048576))
# Output: '1.00 MB'
```

## Naming: Name vs Slug

### Quick Summary

- **Standalone mode**: `name` = `slug` (automatically set)
- **Django/Cloud mode**: `name` is your choice, `slug` is globally unique URL identifier

### For Standalone Users (Local Projects)

When using `scitex.project` without Django, keep it simple:

```python
project = SciTeXProject.create(
    name="neural-decoding",  # Your project name
    path=Path("/home/user/projects/neural-decoding"),
    owner="ywatanabe"
)

# slug is automatically set to "neural-decoding" (same as name)
print(project.slug)  # Output: neural-decoding
```

**Best practice**: Make `name` match the directory name.

### For Django/Cloud Users

When integrated with scitex-cloud, Django manages slug uniqueness:

```
URL: https://scitex.cloud/{username}/{slug}/
                          └────┬────┘ └──┬──┘
                            separate  separate
                           path parts

Example: https://scitex.cloud/ywatanabe/neural-decoding/
```

If multiple users create projects with the same name, Django ensures unique slugs:
- User `ywatanabe` → name=`"neural-decoding"` → slug=`"neural-decoding"`
- User `tanaka` → name=`"neural-decoding"` → slug=`"neural-decoding-1"`

This is like GitHub: two users can have repos with the same name because URLs include username.

See [NAMING.md](./NAMING.md) for detailed explanation.

## Naming Rules

Project names must follow GitHub/Gitea repository naming conventions:

### Valid Characters (URL-Safe)
- Letters: `a-z`, `A-Z`
- Numbers: `0-9`
- Special: `-`, `_`, `.`

**Note**: "URL-safe" means these characters work in URLs without encoding. The actual URL structure is `/{username}/{slug}/` (separate path segments), not `/{username-slug}/` (concatenated).

### Constraints
- Length: 1-100 characters
- Cannot start or end with: `-`, `_`, `.`
- Cannot contain spaces

### Examples

✅ Valid:
```
neural-decoding        # Hyphen-separated
my_project             # Underscore
project.v2.0           # With periods
ML-Research            # Mixed case
```

❌ Invalid:
```
my project             # Space (not URL-safe)
-research              # Starts with hyphen
project_               # Ends with underscore
proj@2025              # Invalid character (@)
user/project           # Forward slash (use separate path segments in URLs)
```

## Django Integration

See [DJANGO_INTEGRATION.md](./DJANGO_INTEGRATION.md) for how to use `scitex.project` with Django models.

## Architecture

See [DESIGN.md](./DESIGN.md) for detailed architecture documentation.

## Contributing

This module is part of the SciTeX ecosystem. See the main repository for contribution guidelines.

## License

MIT License - see LICENSE file for details.
