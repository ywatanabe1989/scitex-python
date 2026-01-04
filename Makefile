# ============================================
# SciTeX Makefile
# https://scitex.ai
# ============================================

# Use bash for proper echo -e support
SHELL := /bin/bash

.PHONY: help install install-dev install-all \
	clean test test-fast test-full test-lf test-ff test-nf test-inc test-unit test-changed lint format check \
	test-stats-cov test-config-cov test-logging-cov \
	build release upload upload-test \
	build-all release-all upload-all upload-test-all \
	sync-extras sync-tests sync-examples sync-redirect \
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
	@echo -e "$(CYAN)🧪 Testing (fastest first):$(NC)"
	@echo -e "  make test-lf           Re-run last-failed tests only"
	@echo -e "  make test-inc          Incremental (only affected by changes)"
	@echo -e "  make test-changed      Tests for git-changed files"
	@echo -e "  make test-unit         Only @unit marked tests"
	@echo -e "  make test MODULE=plt   Single module tests"
	@echo -e "  make test-fast         Skip @slow tests"
	@echo -e "  make test-ff           Failed first, then rest"
	@echo -e "  make test-nf           New tests first"
	@echo -e "  make test              Full suite with coverage"
	@echo -e "  make test-full         Including slow/integration"
	@echo -e ""
	@echo -e "$(CYAN)🔍 Linting:$(NC)"
	@echo -e "  make lint              Check code style (ruff)"
	@echo -e "  make lint-fix          Auto-fix lint issues"
	@echo -e ""
	@echo -e "$(CYAN)✨ Formatting:$(NC)"
	@echo -e "  make format            Format code (ruff)"
	@echo -e "  make format-check      Check formatting without changes"
	@echo -e ""
	@echo -e "$(CYAN)✅ Combined:$(NC)"
	@echo -e "  make check             format-check + lint + test-fast"
	@echo -e ""
	@echo -e "$(CYAN)🧹 Maintenance:$(NC)"
	@echo -e "  make clean             Remove build/test/cache artifacts"
	@echo -e "  make sync-extras       Sync pyproject.toml extras from imports"
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
	@./scripts/maintenance/test.sh $(MODULE) --cov
else
	@./scripts/maintenance/test.sh --cov
endif

test-fast:
ifdef MODULE
	@./scripts/maintenance/test.sh $(MODULE) --fast --cov
else
	@./scripts/maintenance/test.sh --fast --cov
endif

test-full:
ifdef MODULE
	@./scripts/maintenance/test.sh $(MODULE) --cov
else
	@./scripts/maintenance/test.sh --cov
endif

# Cache-based test targets for faster iteration
test-lf:
	@echo -e "$(CYAN)🔄 Re-running last-failed tests...$(NC)"
	@./scripts/maintenance/test.sh --lf

test-ff:
	@echo -e "$(CYAN)🔄 Running failed tests first...$(NC)"
	@./scripts/maintenance/test.sh --ff

test-nf:
	@echo -e "$(CYAN)🔄 Running new tests first...$(NC)"
	@./scripts/maintenance/test.sh --nf

test-inc:
	@echo -e "$(CYAN)🔬 Running incremental tests (affected by changes)...$(NC)"
	@./scripts/maintenance/test.sh --testmon

test-unit:
	@echo -e "$(CYAN)⚡ Running unit tests only...$(NC)"
	@./scripts/maintenance/test.sh -m unit

test-changed:
	@echo -e "$(CYAN)📝 Running tests for git-changed files...$(NC)"
	@./scripts/maintenance/test.sh --changed

lint:
	@./scripts/maintenance/lint.sh

lint-fix:
	@./scripts/maintenance/lint.sh --fix

format:
	@./scripts/maintenance/format.sh

format-check:
	@./scripts/maintenance/format.sh --check

check: format-check lint test-fast
	@echo -e ""
	@echo -e "$(GREEN)✅ All checks passed!$(NC)"

# Module-specific coverage targets for CI
test-stats-cov:
	@./scripts/maintenance/test.sh stats --cov

test-config-cov:
	@./scripts/maintenance/test.sh config --cov

test-logging-cov:
	@./scripts/maintenance/test.sh logging --cov

# ============================================
# Synchronization & Dependencies
# ============================================

sync-extras:
	@echo -e "$(CYAN)📋 Syncing pyproject.toml extras from imports...$(NC)"
	@python scripts/maintenance/generate_module_deps.py --update-pyproject --include-empty 2>/dev/null || echo -e "$(YELLOW)Script not available$(NC)"
	@echo -e "$(GREEN)✅ pyproject.toml extras updated$(NC)"

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
