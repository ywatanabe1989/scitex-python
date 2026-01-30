# DOI Validation - scitex.scholar

Validates DOI accessibility by checking resolution at https://doi.org/<DOI>

## Components

### 1. DOIValidator (Core Class)

**Location**: `scitex/scholar/utils/validation/DOIValidator.py`

**Features**:
- Validates DOI format (must start with "10.", contain "/", etc.)
- Checks DOI accessibility via https://doi.org/<DOI>
- Supports retries on timeout
- Returns validation status + resolved URL
- Batch validation with rate limiting

**Usage**:
```python
from scitex.scholar.utils import DOIValidator

validator = DOIValidator()

# Validate single DOI
is_valid, message, status_code, resolved_url = validator.validate_doi("10.1186/1751-0473-8-7")

# Batch validation
results = validator.validate_batch(
    dois=["10.1186/1751-0473-8-7", "10.1038/invalid"],
    delay=0.5,  # Be polite to DOI service
    progress_callback=lambda i, total, doi, valid: print(f"[{i}/{total}] {doi}: {valid}")
)
```

### 2. Pipeline Integration

**Location**: `scitex/scholar/pipelines/ScholarPipelineMetadataSingle.py`

DOI validation is **optional** during metadata enrichment (disabled by default to avoid extra requests):

```python
from scitex.scholar.pipelines import ScholarPipelineMetadataSingle

# With validation enabled
pipeline = ScholarPipelineMetadataSingle(validate_dois=True)
enriched_paper = await pipeline.enrich_paper_async(paper)

# Without validation (default - faster)
pipeline = ScholarPipelineMetadataSingle(validate_dois=False)
```

When enabled, invalid DOIs are **rejected** and not saved to paper metadata.

### 3. Library Validation Script

**Location**: `scitex/scholar/utils/validation/validate_library_dois.py`

Validates all DOIs in an existing library:

```bash
# Validate all DOIs in library
python -m scitex.scholar.utils.validation.validate_library_dois

# Validate with custom delay
python -m scitex.scholar.utils.validation.validate_library_dois --delay 1.0

# Remove invalid DOIs from metadata files
python -m scitex.scholar.utils.validation.validate_library_dois --fix

# Save report to JSON
python -m scitex.scholar.utils.validation.validate_library_dois --output report.json
```

## Example Output

```
================================================================================
DOI VALIDATION REPORT
================================================================================

[1/102] Git can facilitate greater reproducibility...
  Paper ID: 714B8892
  DOI: 10.1186/1751-0473-8-7
  ✓ Valid (resolved)
  Resolved to: https://scfbm.biomedcentral.com/articles/10.1186/1751-0473-8-7

[2/102] A Novel Method for Neural Data Analysis...
  Paper ID: 7C6C8325
  DOI: 10.1038/s41598-023-12345-6
  ✗ DOI Not Found (404)

================================================================================
VALIDATION SUMMARY
================================================================================
Total papers: 102
Papers with DOI: 53
  ✓ Valid DOIs: 51 (96.2%)
  ✗ Invalid DOIs: 2 (3.8%)
Papers without DOI: 49
```

## Validation Logic

1. **Format Check**: Validates DOI format (10.XXXX/suffix)
2. **HTTP Request**: HEAD request to https://doi.org/<DOI>
3. **Response Analysis**:
   - 200-399: Valid DOI (follows redirects to publisher)
   - 404: Invalid DOI (not found in DOI system)
   - 405: Retry with GET (some publishers don't support HEAD)
   - Timeout: Retry if enabled
4. **Result**: Returns (is_valid, message, status_code, resolved_url)

## Design Decisions

### Why Validation is Optional in Pipeline?

DOI validation adds HTTP requests, which:
- Increases enrichment time (0.5s delay per paper × 100 papers = 50 seconds)
- May hit rate limits if validating large batches
- Most DOIs from API sources (CrossRef, Semantic Scholar) are already valid

**Recommendation**:
- **Disable** during initial enrichment (faster)
- **Enable** for final validation or critical use cases
- **Run** standalone validation script periodically on existing library

### Rate Limiting

Default delay: **0.5 seconds** between requests
- Respects DOI service (https://doi.org)
- Avoids triggering rate limits
- Configurable via `delay` parameter

## Testing

```bash
# Test the validator directly
python -m scitex.scholar.utils.validation.DOIValidator

# Test on your library (dry run - no modifications)
python -m scitex.scholar.utils.validation.validate_library_dois

# Test with fix mode (removes invalid DOIs)
python -m scitex.scholar.utils.validation.validate_library_dois --fix
```

## Integration Points

1. **During Enrichment** (optional): ScholarPipelineMetadataSingle
2. **After Import**: Standalone validation script
3. **Periodic Maintenance**: Re-validate library to catch DOIs that become invalid

## Error Handling

| Error | Behavior |
|-------|----------|
| Empty DOI | Returns (False, "Empty DOI", 0, None) |
| Invalid format | Returns (False, "Invalid DOI format", 0, None) |
| 404 Not Found | Returns (False, "DOI Not Found (404)", 404, None) |
| Timeout | Retries (configurable), then returns (False, "Timeout", 0, None) |
| Connection error | Returns (False, "Connection Error", 0, None) |
| Valid DOI | Returns (True, "Valid (resolved)", 200, resolved_url) |

## Future Enhancements

- [ ] Cache validation results (avoid re-validating same DOIs)
- [ ] Bulk validation API (if DOI.org provides one)
- [ ] DOI normalization (handle variations in format)
- [ ] Integration with DOI metadata enrichment (get title, authors from DOI.org)
