# ============================================
# SciTeX Makefile
# https://scitex.ai
# ============================================

# Use bash for proper echo -e support
SHELL := /bin/bash

.PHONY: help install install-dev install-all \
	clean test test-cov lint format check \
	build release upload upload-test \
	build-all release-all upload-all upload-test-all \
	sync-tests sync-examples sync-redirect \
	show-version tag

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
CYAN := \033[0;36m
GRAY := \033[0;90m
NC := \033[0m

# ============================================
# Default target
# ============================================

.DEFAULT_GOAL := help

help:
	@echo -e ""
	@echo -e "$(GREEN)╔═══════════════════════════════════════════════════════╗$(NC)"
	@echo -e "$(GREEN)║             SciTeX Development Makefile               ║$(NC)"
	@echo -e "$(GREEN)╚═══════════════════════════════════════════════════════╝$(NC)"
	@echo -e ""
	@echo -e "$(CYAN)📦 Installation:$(NC)"
	@echo -e "  make install           Install package (development mode)"
	@echo -e "  make install-dev       Install with dev dependencies"
	@echo -e "  make install-all       Install with all optional dependencies"
	@echo -e ""
	@echo -e "$(CYAN)🔧 Development:$(NC)"
	@echo -e "  make test              Run all tests"
	@echo -e "  make test MODULE=xxx   Run tests for specific module (e.g., config, stats)"
	@echo -e "  make test-cov          Run tests with coverage"
	@echo -e "  make test-cov MODULE=xxx  Run module tests with coverage"
	@echo -e "  make lint              Check code style (ruff)"
	@echo -e "  make lint-fix          Auto-fix lint issues"
	@echo -e "  make format            Format code (ruff)"
	@echo -e "  make format-check      Check formatting without changes"
	@echo -e "  make check             Run all checks (format-check + lint + test)"
	@echo -e ""
	@echo -e "$(CYAN)🧹 Maintenance:$(NC)"
	@echo -e "  make clean             Remove build/test/cache artifacts"
	@echo -e "  make sync-tests        Sync test files with source structure"
	@echo -e "  make sync-examples     Sync example files with source structure"
	@echo -e ""
	@echo -e "$(CYAN)📤 Build & Release (scitex only):$(NC)"
	@echo -e "  make build             Build package"
	@echo -e "  make upload-test       Upload to TestPyPI"
	@echo -e "  make upload            Upload to PyPI"
	@echo -e "  make release           Build, tag, and upload to PyPI"
	@echo -e ""
	@echo -e "$(CYAN)📤 Build & Release (scitex + scitex-python):$(NC)"
	@echo -e "  make sync-redirect     Sync redirect package version"
	@echo -e "  make build-all         Build both packages"
	@echo -e "  make upload-test-all   Upload both to TestPyPI"
	@echo -e "  make upload-all        Upload both to PyPI"
	@echo -e "  make release-all       Build, tag, and upload both to PyPI"
	@echo -e ""
	@echo -e "$(CYAN)📋 Other:$(NC)"
	@echo -e "  make show-version      Show current version"
	@echo -e "  make tag               Create git tag from version"
	@echo -e ""

# ============================================
# Installation
# ============================================

install:
	@echo -e "$(CYAN)📦 Installing scitex in development mode...$(NC)"
	@pip install -e .
	@echo -e "$(GREEN)✅ Installation complete$(NC)"

install-dev:
	@echo -e "$(CYAN)📦 Installing scitex with dev dependencies...$(NC)"
	@pip install -e ".[dev]"
	@echo -e "$(GREEN)✅ Installation complete$(NC)"

install-all:
	@echo -e "$(CYAN)📦 Installing scitex with all dependencies...$(NC)"
	@pip install -e ".[all,dev]"
	@echo -e "$(GREEN)✅ Installation complete$(NC)"

# ============================================
# Development
# ============================================

clean:
	@./scripts/maintenance/clean.sh

test:
ifdef MODULE
	@./scripts/maintenance/test.sh $(MODULE)
else
	@./scripts/maintenance/test.sh
endif

test-cov:
ifdef MODULE
	@./scripts/maintenance/test.sh $(MODULE) --cov
else
	@./scripts/maintenance/test.sh --cov
endif

lint:
	@./scripts/maintenance/lint.sh

lint-fix:
	@./scripts/maintenance/lint.sh --fix

format:
	@./scripts/maintenance/format.sh

format-check:
	@./scripts/maintenance/format.sh --check

check: format-check lint test
	@echo -e ""
	@echo -e "$(GREEN)✅ All checks passed!$(NC)"

# ============================================
# Synchronization
# ============================================

sync-tests:
	@echo -e "$(CYAN)🔄 Syncing test files with source...$(NC)"
	@./tests/sync_tests_with_source.sh

sync-examples:
	@echo -e "$(CYAN)🔄 Syncing example files with source...$(NC)"
	@./examples/sync_examples_with_source.sh

# ============================================
# Build & Release (main package)
# ============================================

build: clean
	@echo -e "$(CYAN)🏗️  Building source and wheel distributions...$(NC)"
	@python -m build
	@echo -e "$(GREEN)✅ Build complete$(NC)"

upload-test: build
	@echo -e "$(CYAN)📤 Uploading to TestPyPI...$(NC)"
	@python -m twine upload --repository testpypi dist/*
	@echo -e "$(GREEN)✅ Upload to TestPyPI complete$(NC)"

upload: build
	@echo -e "$(CYAN)📤 Uploading to PyPI...$(NC)"
	@python -m twine upload dist/*
	@echo -e "$(GREEN)✅ Upload to PyPI complete$(NC)"

release: clean build tag upload
	@echo -e ""
	@echo -e "$(GREEN)✅ Release complete!$(NC)"
	@VERSION=$$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	echo -e "$(CYAN)Version $$VERSION released to PyPI$(NC)"

# ============================================
# Build & Release (both packages)
# ============================================

sync-redirect:
	@echo -e "$(CYAN)🔄 Syncing scitex-python version...$(NC)"
	@./scripts/release.sh sync

build-all: clean
	@echo -e "$(CYAN)🏗️  Building both packages...$(NC)"
	@./scripts/release.sh build
	@echo -e "$(GREEN)✅ Build complete$(NC)"

upload-test-all: build-all
	@echo -e "$(CYAN)📤 Uploading both packages to TestPyPI...$(NC)"
	@./scripts/release.sh upload-test
	@echo -e "$(GREEN)✅ Upload to TestPyPI complete$(NC)"

upload-all: build-all
	@echo -e "$(CYAN)📤 Uploading both packages to PyPI...$(NC)"
	@./scripts/release.sh upload
	@echo -e "$(GREEN)✅ Upload to PyPI complete$(NC)"

release-all: clean build-all tag
	@echo -e "$(CYAN)🚀 Releasing both packages to PyPI...$(NC)"
	@./scripts/release.sh upload
	@echo -e ""
	@echo -e "$(GREEN)✅ Release complete!$(NC)"
	@VERSION=$$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	echo -e "$(CYAN)Version $$VERSION released: scitex and scitex-python$(NC)"

# ============================================
# Version & Tagging
# ============================================

show-version:
	@VERSION=$$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	echo -e "$(CYAN)Current version: $(GREEN)$$VERSION$(NC)"

tag:
	@echo -e "$(CYAN)🏷️  Creating git tag...$(NC)"
	@VERSION=$$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	echo -e "$(GRAY)Version: $$VERSION$(NC)"; \
	git tag -a v$$VERSION -m "Release v$$VERSION"; \
	git push origin v$$VERSION; \
	echo -e "$(GREEN)✅ Tag v$$VERSION created and pushed$(NC)"

# EOF
