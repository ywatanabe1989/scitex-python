.PHONY: help \
	clean \
	clean-build \
	clean-pyc \
	clean-test \
	build \
	install \
	install-dev \
	test \
	test-all \
	test-stats-cov \
	test-logging-cov \
	lint \
	format \
	check \
	release \
	upload \
	upload-test \
	tag

help:
	@echo "SciTeX Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make install          - Install package in development mode"
	@echo "  make install-dev      - Install package with dev dependencies"
	@echo "  make clean            - Remove all build, test, and Python artifacts"
	@echo "  make build            - Build source and wheel distributions"
	@echo "  make test             - Run tests with pytest"
	@echo "  make test-stats-cov   - Run stats module tests with coverage"
	@echo "  make test-logging-cov - Run logging module tests with coverage"
	@echo "  make lint             - Check code style with flake8"
	@echo "  make format           - Format code with black and isort"
	@echo "  make check            - Run all checks (lint, format, test)"
	@echo "  make release          - Build and upload to PyPI (requires version bump)"
	@echo "  make upload-test      - Upload to TestPyPI"
	@echo "  make tag              - Create git tag from current version"

clean: clean-build clean-pyc clean-test clean-old clean-claude

clean-claude:
	@echo "Cleaning Claude files..."
	rm -rf docs/to_claude
	rm -rf docs/from_agents
	rm -rf docs/from_user
	rm -rf CLAUDE.md
	rm -rf .claude
	rm -rf mgmt
	rm -rf management
	rm -rf .mcp.json

clean-old:
	@echo "Cleaning obsolete files..."
	find . -type d -name '.old' -exec rm -rf {} +
	find . -type d -name 'legacy' -exec rm -rf {} +

clean-build:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	@echo "Cleaning Python artifacts..."
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	# find . -name '*~' -exec rm -f {} +


clean-test:
	@echo "Cleaning test artifacts..."
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/

build: clean
	@echo "Building source and wheel distributions..."
	python -m build

install:
	@echo "Installing package in development mode..."
	pip install -e .

install-dev:
	@echo "Installing package with dev dependencies..."
	pip install -e ".[dev]"

test:
	@echo "Running tests..."
	pytest tests/ -v

test-all:
	@echo "Running all tests with coverage..."
	pytest tests/ -v --cov=src/scitex --cov-report=html --cov-report=term

test-stats-cov:
	@echo "Running stats module tests with coverage..."
	@mkdir -p htmlcov
	pytest tests/scitex/stats/ \
		--cov=src/scitex/stats \
		--cov-report=term-missing \
		--cov-report=html:htmlcov/stats \
		--cov-report=json:coverage-stats.json \
		--cov-report=xml:coverage-stats.xml \
		-v
	@echo ""
	@echo "Coverage reports generated:"
	@echo "  - HTML: htmlcov/stats/index.html"
	@echo "  - JSON: coverage-stats.json"
	@echo "  - XML: coverage-stats.xml"

test-logging-cov:
	@echo "Running logging module tests with coverage..."
	@mkdir -p htmlcov
	pytest tests/scitex/logging/ \
		--cov=src/scitex/logging \
		--cov-report=term-missing \
		--cov-report=html:htmlcov/logging \
		--cov-report=json:coverage-logging.json \
		--cov-report=xml:coverage-logging.xml \
		-v
	@echo ""
	@echo "Coverage reports generated:"
	@echo "  - HTML: htmlcov/logging/index.html"
	@echo "  - JSON: coverage-logging.json"
	@echo "  - XML: coverage-logging.xml"

lint:
	@echo "Checking code style..."
	flake8 src/scitex tests/

format:
	@echo "Formatting code..."
	black src/scitex tests/
	isort src/scitex tests/

check: lint test
	@echo "All checks passed!"

tag:
	@echo "Creating git tag..."
	@VERSION=$$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	echo "Current version: $$VERSION"; \
	git tag -a v$$VERSION -m "Release v$$VERSION"; \
	git push origin v$$VERSION; \
	echo "Tag v$$VERSION created and pushed"

upload-test: build
	@echo "Uploading to TestPyPI..."
	python -m twine upload --repository testpypi dist/*

upload: build
	@echo "Uploading to PyPI..."
	python -m twine upload dist/*

release: clean build tag upload
	@echo "Release complete!"
	@VERSION=$$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	echo "Version $$VERSION released to PyPI"

# Development helpers
dev-setup: install-dev
	@echo "Setting up development environment..."
	pre-commit install

show-version:
	@grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'

bump-patch:
	@echo "Bumping patch version..."
	@CURRENT=$$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	echo "Current version: $$CURRENT"; \
	read -p "Enter new version: " NEW_VERSION; \
	sed -i "s/__version__ = \".*\"/__version__ = \"$$NEW_VERSION\"/" src/scitex/__version__.py; \
	sed -i "s/version = \".*\"/version = \"$$NEW_VERSION\"/" pyproject.toml; \
	echo "Version bumped to $$NEW_VERSION"
