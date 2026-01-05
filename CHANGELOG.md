# Changelog

All notable changes to SciTeX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.10.3] - 2026-01-06

### Fixed
- **Dependency**: Add direct `llvmlite>=0.39.0` constraint to force Python 3.11+ compatible version
  - `umap-learn>=0.5.4` alone was insufficient; `pynndescent` still pulled old `llvmlite==0.36.0`

## [2.10.2] - 2026-01-06

### Fixed
- **Dependency**: Pin `umap-learn>=0.5.4` for Python 3.11+ compatibility (llvmlite issue)

## [2.10.1] - 2026-01-06

### Fixed
- **Documentation**: Updated README to recommend `scitex[all]` as primary installation

## [2.10.0] - 2026-01-05

### Added
- **CI/CD Infrastructure**: Separate workflow files for all 41 modules
- **Datetime Module**: New `datetime` module with `dt` alias for time operations
- **Comprehensive Test Suites**: Major test improvements across modules
  - AI module: 577 tests passing
  - NN module: 498 tests passing
  - IO module: 506 tests passing
  - Writer module: 414 tests (expanded from 276)
  - CLI module: 201 tests
  - SH module: 149 tests
  - Scholar module: Core and storage tests

### Changed
- **License**: Updated to AGPL-3.0
- Updated README with improved installation section
- Reorganized project structure (removed unused externals directory)

### Fixed
- HTML tag order and markdown link syntax in README
- DSP module test failures (reduced from 154 to 78)
- Resource module test failures and source bugs
- Repro module JAX circular import bug
- Web module tests and source bugs
- Gen module tests improvements

### Security
- Added .env.zenrows to .gitignore to prevent credential commits

## [2.9.0] - 2025-12-28

### Added
- **FTS Bundle System**: New node kinds for annotations and images
  - `text` kind for text annotations with fontsize, fontweight, alignment
  - `shape` kind for shapes (rectangle, ellipse, arrow, line) with styling
  - `image` kind for embedded images
- **Node Categories**: Clear categorization of node kinds
  - DATA_LEAF_KINDS: plot, table, stats (require payload data files)
  - ANNOTATION_LEAF_KINDS: text, shape (no payload required)
  - IMAGE_LEAF_KINDS: image (require payload image file)
  - COMPOSITE_KINDS: figure (contain children)
- **Theme Enhancements**: FigureTitle and Caption dataclasses for attribute access
- **Matplotlib Analysis Tools**: Signature analysis tools in `dev/plt/mpl/`
- **CI/CD Improvements**: Added module-specific coverage targets

### Changed
- Reorganized demo plotters to `dev/plt/demo_plotters/` subdirectory
- Updated FTS examples to use new `kind` attribute instead of `type`
- Theme figure_title and caption now use dataclass attribute access

### Fixed
- Various import and attribute access fixes across modules
- Stats module now has optional torch dependency with numpy fallback

### Removed
- Deprecated VIS planning documentation (now in .gitignore)

## [2.8.1] - 2025-12-27

### Changed
- Made torch optional in stats module with numpy fallback
- Added cross-process FIFO queue for MCP audio server

## [2.8.0] - 2025-12-20

### Added
- FTS bundle examples
- Updated existing examples for FTS compatibility
- Tests for FTS refactoring

### Changed
- Core modules updated for FTS integration
- Consolidated error and warning classes in logging module
