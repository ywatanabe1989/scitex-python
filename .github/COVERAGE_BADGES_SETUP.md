# Coverage Badges Setup Guide

This guide explains how to set up the dynamic coverage badges for the stats and logging modules.

## Prerequisites

1. A GitHub account with repository access
2. A Gist to store badge data

## Setup Steps

### 1. Create a Personal Access Token (PAT)

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a descriptive name like "SciTeX Coverage Badges"
4. Select the **gist** scope only
5. Generate and copy the token

### 2. Create a Public Gist

1. Go to https://gist.github.com/
2. Create a new gist with any filename (e.g., `coverage-badges.json`)
3. Add any initial content (it will be overwritten)
4. Make sure it's **public**
5. Copy the Gist ID from the URL (e.g., `https://gist.github.com/username/GIST_ID`)

### 3. Add GitHub Secrets

Add these secrets to your repository (Settings → Secrets and variables → Actions → New repository secret):

- **GIST_TOKEN**: Your personal access token from step 1
- **GIST_ID**: Your gist ID from step 2

### 4. Update README Badge URLs

Replace `GIST_ID` in the README.md badges with your actual Gist ID:

```markdown
[![Stats Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/USERNAME/GIST_ID/raw/scitex-stats-coverage.json)](...)
[![Logging Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/USERNAME/GIST_ID/raw/scitex-logging-coverage.json)](...)
```

Replace:
- `USERNAME` with your GitHub username
- `GIST_ID` with your actual Gist ID

## How It Works

1. When changes are pushed to `src/scitex/stats/` or `tests/scitex/stats/`, the stats coverage workflow runs
2. The workflow generates coverage reports and extracts the percentage
3. The dynamic badge action updates the gist with the new coverage data
4. The README badge displays the updated coverage percentage with color coding:
   - 🟢 Green: > 80%
   - 🟡 Yellow: 60-80%
   - 🔴 Red: < 60%

## Local Testing

To run coverage tests locally:

```bash
# Stats module coverage
make test-stats-cov

# Logging module coverage
make test-logging-cov
```

Reports are generated in:
- `htmlcov/stats/index.html` - Stats HTML report
- `htmlcov/logging/index.html` - Logging HTML report
- `coverage-stats.json` - Stats JSON report
- `coverage-logging.json` - Logging JSON report

## Troubleshooting

### Badges Not Updating

1. Check that the workflows are enabled in GitHub Actions
2. Verify secrets are correctly set
3. Make sure the Gist is public
4. Check workflow runs for errors

### Badge Shows "invalid"

This usually means:
- The Gist URL is incorrect
- The Gist doesn't contain the expected JSON file yet (run the workflow once)
- The Gist is private instead of public

### Workflow Fails

Common issues:
- Missing or incorrect secrets (GIST_TOKEN, GIST_ID)
- Python dependencies not installed
- Test failures preventing coverage generation

## Manual Badge Update

If you need to manually update badge data, the gist should contain JSON files like:

**scitex-stats-coverage.json**:
```json
{
  "schemaVersion": 1,
  "label": "stats coverage",
  "message": "85.5%",
  "color": "brightgreen"
}
```

**scitex-logging-coverage.json**:
```json
{
  "schemaVersion": 1,
  "label": "logging coverage",
  "message": "92.3%",
  "color": "brightgreen"
}
```
