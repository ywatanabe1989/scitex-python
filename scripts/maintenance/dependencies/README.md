<!-- ---
!-- Timestamp: 2026-01-08 01:55:00
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/scripts/maintenance/dependencies/README.md
!-- --- -->

# Dependency Analysis Scripts

Scripts for analyzing dependencies within the scitex package.

## Scripts

### list_deps.py

Analyzes cross-module dependencies between scitex submodules.

```bash
python scripts/maintenance/dependencies/list_deps.py
```

Output shows which scitex submodules import other scitex submodules:

```
Module Dependencies:
==================================================

ai:
  - scitex.config
  - scitex.logging

plt:
  - scitex.io
  - scitex.stats
```

### list_internal_imports.py

Lists internal imports within each scitex submodule.

```bash
python scripts/maintenance/dependencies/list_internal_imports.py
```

Output shows the internal import paths used within each module:

```
Internal Imports by Module:
==================================================

plt:
  - scitex.plt._subplots
  - scitex.plt.ax._plot
  - scitex.plt.utils
```

## Use Cases

- Identify circular dependencies between modules
- Understand module coupling before refactoring
- Verify module isolation and layering
- Plan dependency cleanup

<!-- EOF -->
