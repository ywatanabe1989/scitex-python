<!-- ---
!-- Timestamp: 2025-10-29 05:56:40
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/template/README.md
!-- --- -->

# SciTeX Template Module

Project template creation utilities for SciTeX ecosystem.

## Available Templates

### 1. Research Project (`clone_research`)
Full scientific workflow structure for research projects.

**Use case:** Scientific research with data analysis, experiments, and paper writing

**Template:** [scitex_template_research](https://github.com/ywatanabe1989/scitex_template_research.git)

**Features:**
- `scripts/` - Analysis and preprocessing scripts
- `data/` - Raw and processed data management
- `docs/` - Manuscripts, notes, and references
- `results/` - Analysis outputs and reports
- `config/` - Project configuration files

### 2. Python Package (`clone_pip_project`)
Pip-installable Python package template.

**Use case:** Creating distributable Python packages for PyPI

**Template:** [pip-project-template](https://github.com/ywatanabe1989/pip-project-template.git)

**Features:**
- `src/` - Package source code
- `tests/` - Unit and integration tests
- `docs/` - Sphinx documentation
- `setup.py` - Package configuration
- CI/CD - GitHub Actions workflows

### 3. Singularity Container (`clone_singularity`)
Container-based project with Singularity.

**Use case:** Reproducible computational environments with containers

**Template:** [singularity-template](https://github.com/ywatanabe1989/singularity-template.git)

**Features:**
- Singularity definition files
- Container build scripts
- Environment specifications
- Deployment configuration

### 4. Paper Directory (`clone_writer_directory`)
Academic paper writing template with scitex-writer.

**Use case:** Writing academic papers with LaTeX and BibTeX management

**Template:** [scitex-writer](git@github.com:ywatanabe1989/scitex-writer.git)

**Features:**
- LaTeX document structure
- BibTeX bibliography management
- Figure and table organization
- Manuscript tracking

## Git Strategy Options

All template functions support a `git_strategy` parameter to control git initialization:

- `"child"` (default): Creates new isolated git repository with initial commit and main/develop branches
- `"parent"`: Uses existing parent git repository (degrades to "child" if not found)
- `"origin"`: Preserves template's original git history for upstream tracking
- `None`: Disables git initialization entirely

## Usage

### Python API

```python
from scitex.template import (
    clone_research,
    clone_pip_project,
    clone_singularity,
    clone_writer_directory,
)

# Create a research project
clone_research("my_research_project")

# Create with specific git strategy
clone_pip_project("my_package", "~/projects", git_strategy="parent")

# Preserve template git history
clone_singularity("my_container", git_strategy="origin")

# No git initialization
clone_writer_directory("my_paper", "~/papers", git_strategy=None)
```

### Command Line

```bash
# Research project
python -m scitex clone_research my_research_project
python -m scitex clone_research my_project ~/projects

# Python package
python -m scitex clone_pip_project my_package
python -m scitex clone_pip_project my_package ~/packages

# Singularity container
python -m scitex clone_singularity my_container
python -m scitex clone_singularity my_container ~/containers

# Paper directory
python -m scitex clone_writer_directory my_paper
python -m scitex clone_writer_directory my_paper ~/papers
```

### Get Template Information

```python
from scitex.template import get_available_templates_info

templates = get_available_templates_info()
for template in templates:
    print(f"{template['name']}: {template['description']}")
    print(f"  Use case: {template['use_case']}")
    print(f"  GitHub: {template['github_url']}")
```

## What Happens When You Create a Project

1. **Clone Template**: Downloads the template repository to a temporary directory
2. **Handle Git History**: Based on git_strategy:
   - `"child"`: Removes template's `.git` directory
   - `"origin"`: Preserves template's `.git` directory
   - `"parent"`: Removes template's `.git` directory
3. **Copy to Target**: Copies template files to your target directory
4. **Customize**: Renames template package/directories to match your project name
5. **Update References**: Updates all references to the template name in files
6. **Initialize Git**: Based on git_strategy:
   - `"child"`: Creates new git repository with initial commit and `main`/`develop` branches
   - `"parent"`: Uses existing parent repository
   - `"origin"`: Keeps template's git history
   - `None`: Skips git initialization

## Requirements

- Git must be installed and accessible in PATH
- SSH keys configured for GitHub (for templates using SSH URLs)

<!-- EOF -->