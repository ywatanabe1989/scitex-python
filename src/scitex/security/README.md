# SciTeX Security Module

Reusable security utilities for the SciTeX ecosystem. Works everywhere: local, cloud, CLI.

## Features

- Check GitHub security alerts (secrets, Dependabot, code scanning)
- Save alerts to timestamped files
- Python API and CLI interface
- Reusable across all SciTeX projects

## Installation

The module is part of the `scitex` package (editable install):

```bash
cd ~/proj/scitex-code
pip install -e .
```

## Prerequisites

Install and authenticate GitHub CLI:

```bash
# Install
sudo apt install gh  # or: brew install gh

# Authenticate
gh auth login
```

## Usage

### Python API

```python
from scitex.security import check_github_alerts, save_alerts_to_file

# Check alerts for current repo
alerts = check_github_alerts()

# Check specific repo
alerts = check_github_alerts(repo="owner/repo")

# Save to file
file_path = save_alerts_to_file(alerts)
print(f"Saved to: {file_path}")

# Check if there are open alerts
total = sum(
    len([a for a in alerts[key] if a.get("state") == "open"])
    for key in alerts
)
if total > 0:
    print(f"Found {total} security issues!")
```

### Command Line

```bash
# Check current repo
python -m scitex.security.cli check

# Check specific repo
python -m scitex.security.cli check --repo SciTeX-AI/scitex-cloud

# Save to file
python -m scitex.security.cli check --save

# Save to custom directory
python -m scitex.security.cli check --save --output-dir /path/to/logs

# Show latest report
python -m scitex.security.cli latest
```

### From Django (scitex-cloud)

```python
# In any Django app
from scitex.security import check_github_alerts, save_alerts_to_file

def check_security_view(request):
    try:
        alerts = check_github_alerts()
        file_path = save_alerts_to_file(
            alerts,
            output_dir=settings.BASE_DIR / "logs" / "security"
        )
        # Process alerts...
    except GitHubSecurityError as e:
        # Handle error...
```

### From Anywhere

```python
# Any Python script, anywhere
from scitex.security import check_github_alerts

alerts = check_github_alerts(repo="SciTeX-AI/scitex-cloud")
print(f"Secrets: {len(alerts['secrets'])}")
print(f"Dependabot: {len(alerts['dependabot'])}")
print(f"Code Scanning: {len(alerts['code_scanning'])}")
```

## Output Format

Alerts are saved to `logs/security/security-<timestamp>.txt`:

```
==================================================
GitHub Security Alerts Report
Generated: 2025-11-03 12:00:00
==================================================

### SECRET SCANNING ALERTS ###

- [open] Checkout.com Production Secret Key
  Created: 2025-11-03T10:42:00Z
  URL: https://github.com/SciTeX-AI/scitex-cloud/security/secret-scanning/1

==================================================

### DEPENDABOT VULNERABILITY ALERTS ###

- [open] HIGH: Django SQL Injection vulnerability
  Package: django
  CVE: CVE-2024-12345
  URL: https://github.com/SciTeX-AI/scitex-cloud/security/dependabot/1

==================================================

### CODE SCANNING ALERTS ###

No open code scanning alerts

==================================================

### SUMMARY ###

Total open alerts: 2
  - Secrets: 1
  - Dependabot: 1
  - Code Scanning: 0

⚠️  ACTION REQUIRED: Security issues found!
```

## Integration Examples

### Cron Job

```bash
# Check daily at 9 AM
0 9 * * * cd ~/proj/scitex-cloud && python -m scitex.security.cli check --save
```

### GitHub Actions

```yaml
- name: Check Security
  run: |
    pip install -e ~/proj/scitex-code
    python -m scitex.security.cli check --save
```

### Django Management Command

Create `apps/security_app/management/commands/check_security.py`:

```python
from django.core.management.base import BaseCommand
from scitex.security import check_github_alerts, save_alerts_to_file
from django.conf import settings

class Command(BaseCommand):
    help = 'Check GitHub security alerts'

    def handle(self, *args, **options):
        alerts = check_github_alerts()
        file_path = save_alerts_to_file(
            alerts,
            output_dir=settings.BASE_DIR / "logs" / "security"
        )
        self.stdout.write(f"Report saved to: {file_path}")
```

Then run:
```bash
python manage.py check_security
```

## Error Handling

```python
from scitex.security import check_github_alerts, GitHubSecurityError

try:
    alerts = check_github_alerts()
except GitHubSecurityError as e:
    if "not found" in str(e):
        print("Install GitHub CLI: https://cli.github.com/")
    elif "Not authenticated" in str(e):
        print("Run: gh auth login")
    else:
        print(f"Error: {e}")
```

## API Reference

### `check_github_alerts(repo=None)`
Fetch all security alerts.

**Args:**
- `repo` (str, optional): Repository in format 'owner/repo'

**Returns:**
- `dict`: Dictionary with keys 'secrets', 'dependabot', 'code_scanning'

**Raises:**
- `GitHubSecurityError`: If GitHub CLI not installed or not authenticated

### `save_alerts_to_file(alerts, output_dir=None, create_symlink=True)`
Save alerts to timestamped file.

**Args:**
- `alerts` (dict): Dictionary from `check_github_alerts()`
- `output_dir` (Path, optional): Output directory
- `create_symlink` (bool): Create 'security-latest.txt' symlink

**Returns:**
- `Path`: Path to saved file

### `get_latest_alerts_file(security_dir=None)`
Get path to latest alerts file.

**Args:**
- `security_dir` (Path, optional): Security directory

**Returns:**
- `Path` or `None`: Path to latest file

## Testing

```bash
cd ~/proj/scitex-code
pytest tests/test_security.py
```

## See Also

- Main security documentation: `~/proj/scitex-cloud/security.md`
- GitHub Security: https://docs.github.com/en/code-security
