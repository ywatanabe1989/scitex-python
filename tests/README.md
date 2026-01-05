# SciTeX Testing Guide

## Quick Start

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests (recommended for development)
make test-lf      # Re-run last failed (fastest)
make test-inc     # Only tests affected by changes
make test-changed # Tests for git-changed files
```

## Test Commands (sorted by speed)

### Fastest (use during development)

| Command | Description | When to Use |
|---------|-------------|-------------|
| `make test-lf` | Re-run last failed tests | Fixing a specific failure |
| `make test-inc` | Incremental (testmon) | After code changes |
| `make test-changed` | Git-changed files only | Before staging |
| `make test-unit` | Only `@unit` marked tests | Quick sanity check |

### Medium Speed

| Command | Description | When to Use |
|---------|-------------|-------------|
| `make test MODULE=plt` | Single module | Working on one module |
| `make test-fast` | Skip `@slow` tests | Before commit |
| `make test-ff` | Failed first, then rest | After fixing bugs |
| `make test-nf` | New tests first | After adding tests |

### Full Suite

| Command | Description | When to Use |
|---------|-------------|-------------|
| `make test` | All tests with coverage | Before push |
| `make test-full` | Including slow/integration | Before release |

## Test Markers

Use markers to categorize tests:

```python
import pytest

@pytest.mark.unit
def test_fast_isolated():
    """Fast, no external dependencies."""
    assert 1 + 1 == 2

@pytest.mark.slow
def test_heavy_computation():
    """Takes >5 seconds."""
    result = expensive_operation()
    assert result is not None

@pytest.mark.integration
def test_with_database():
    """Requires external resources."""
    db = connect_to_database()
    assert db.is_connected()

@pytest.mark.gpu
def test_cuda_operation():
    """Requires GPU."""
    import torch
    assert torch.cuda.is_available()

@pytest.mark.network
def test_api_call():
    """Requires network access."""
    response = requests.get("https://api.example.com")
    assert response.ok
```

### Running by Marker

```bash
# Run only unit tests
./scripts/maintenance/test.sh -m unit

# Skip slow tests
./scripts/maintenance/test.sh -m "not slow"

# Run integration tests only
./scripts/maintenance/test.sh -m integration
```

## Directory Structure

```
tests/
├── conftest.py          # Shared fixtures and hooks
├── README.md            # This file
├── results/             # Test results and coverage
│   ├── test-*.json      # Test results
│   ├── coverage-*.json  # Coverage data
│   └── *.log            # Test logs
├── htmlcov/             # HTML coverage reports
│   └── {module}/        # Per-module coverage
└── scitex/              # Test modules (mirrors src/scitex/)
    ├── ai/
    ├── plt/
    ├── io/
    └── ...
```

## Recommended Workflow

### During Development

```bash
# 1. Make code changes
vim src/scitex/plt/something.py

# 2. Run affected tests only
make test-inc

# 3. If tests fail, fix and re-run failures
make test-lf

# 4. Before commit
make test-changed
```

### Before Push

```bash
# Run fast tests (skip slow)
make test-fast

# Or full suite with coverage
make test
```

## Advanced Options

### test.sh Options

```bash
./scripts/maintenance/test.sh [OPTIONS] [MODULE]

Options:
  -c, --cov              Enable coverage reporting
  -v, --verbose          Verbose output (default)
  -q, --quiet            Quiet output
  -f, --fast             Skip @slow tests
  -x, --exitfirst        Stop on first failure
  -s, --sequential       Disable parallel (xdist)
  -n, --parallel N       Number of workers (default: auto)
  -k PATTERN             Run tests matching pattern
  -m, --marker MARKER    Run tests with marker

Cache options:
  --lf, --last-failed    Re-run only last failed
  --ff, --failed-first   Failed tests first
  --nf, --new-first      New tests first
  --sw, --stepwise       Stop/resume on failure
  --cache-clear          Clear cache

Incremental:
  --testmon              Only affected tests
  --changed              Git-changed files

Protection:
  --strict-root          Fail if tests pollute root dir
```

### Examples

```bash
# Run plt module tests with coverage
./scripts/maintenance/test.sh plt --cov

# Run tests matching "export" in io module
./scripts/maintenance/test.sh io -k export

# Run only unit tests, stop on first failure
./scripts/maintenance/test.sh -m unit -x

# Incremental tests for stats module
./scripts/maintenance/test.sh stats --testmon
```

## Writing Good Tests

### Use Fixtures

```python
import pytest

@pytest.fixture
def sample_data():
    """Shared test data."""
    return {"key": "value"}

def test_with_fixture(sample_data):
    assert sample_data["key"] == "value"
```

### Use tmp_path (NOT project root)

```python
def test_file_output(tmp_path):
    """Use tmp_path to avoid polluting project root."""
    output_file = tmp_path / "output.txt"
    output_file.write_text("test")
    assert output_file.exists()
```

### Skip When Dependencies Missing

```python
pytest.importorskip("torch")

def test_torch_operation():
    import torch
    # ...
```

## Coverage Reports

After running tests with `--cov`:

```bash
# View HTML report
open tests/htmlcov/{module}/index.html

# View JSON report
cat tests/results/coverage-{module}.json
```
