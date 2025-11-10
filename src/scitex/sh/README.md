<!-- ---
!-- Timestamp: 2025-10-29 07:23:56
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/sh/README.md
!-- --- -->

# scitex.sh - Shell Command Execution Module

Safe shell command execution with injection protection.

## Structure

- `__init__.py` - Main interface (sh, sh_run functions)
- `_execute.py` - Core execution logic
- `_security.py` - Security validation and quoting
- `_types.py` - Type definitions
- `test_sh_simple.py` - Test file

## Usage

```python
import scitex

# List format (ONLY format allowed - safe by design)
result = scitex.sh(["ls", "-la", "/home"])

# With user input (safe - arguments are literal)
user_input = "../malicious; rm -rf /"
result = scitex.sh(["cat", user_input])

# For filtering (instead of pipes)
result = scitex.sh(["ls", "-la"])
py_files = [l for l in result['stdout'].split('\n') if '.py' in l]

# sh_run always returns dict
result = scitex.sh_run(["echo", "test"])
if result['success']:
    print(result['stdout'])
```

## Security Features

1. List format only - String commands are rejected with TypeError
2. shell=False always - No shell interpretation of special characters
3. Null byte validation - Blocks common injection vector
4. Arguments as literals - ;, |, & are treated as literal text

## Backward Compatibility

Old imports still work:

```python
from scitex.sh import sh, sh_run, quote
```

# EOF

<!-- EOF -->