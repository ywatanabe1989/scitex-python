#!/usr/bin/env python3
"""YAML configuration file template."""

TEMPLATE = {
    "name": "Configuration YAML",
    "description": "YAML configuration file for @stx.session scripts",
    "filename": "config/config.yaml",
    "usage": """
Usage:
  Place in ./config/ directory
  Access via CONFIG parameter: CONFIG.config.data.input_dir
""",
    "content": """# Configuration file for SciTeX session
# Place in ./config/ directory (auto-loaded by @stx.session)
# Access via CONFIG parameter: CONFIG.<filename>.<key>
# Example: CONFIG.config.data.input_dir (for this file named config.yaml)

# Data paths
data:
  input_dir: ./data/raw
  output_dir: ./data/processed

# Analysis parameters
analysis:
  n_samples: 100
  threshold: 0.05
  seed: 42

# Figure settings
figure:
  dpi: 300
  format: png
  width_mm: 180
  height_mm: 120

# Logging
logging:
  level: INFO
  save_log: true
""",
}

# EOF
