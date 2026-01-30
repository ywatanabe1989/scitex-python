<!-- ---
!-- Timestamp: 2026-01-30 11:36:29
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-python/README.md
!-- --- -->

<p align="center">
  <a href="https://scitex.ai">
    <img src="docs/assets/images/scitex-logo-blue-cropped.png" alt="SciTeX" width="400">
  </a>
</p>

<p align="center">
  <a href="https://badge.fury.io/py/scitex"><img src="https://badge.fury.io/py/scitex.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/scitex/"><img src="https://img.shields.io/pypi/pyversions/scitex.svg" alt="Python Versions"></a>
  <a href="https://github.com/ywatanabe1989/scitex-python/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ywatanabe1989/scitex-python" alt="License"></a>
  <img src="https://img.shields.io/badge/uv-recommended-blue" alt="uv recommended">
</p>

<p align="center">
  <a href="https://scitex.ai">scitex.ai</a> · <code>pip install scitex</code>
</p>

---

**Modular Python toolkit for scientific research automation — with verification features to ease peer review.**

Empowers both human researchers and AI agents. Each module works independently and adapts to various academic workflows, handling primary and secondary research with real or simulated data.

> **SciTeX is fully modular; every module is opt-in and exposed only when operational.**

## 🎬 Demo

**40 min, zero human intervention** — AI agent conducts full research pipeline:

> Literature search → Data analysis → Statistics → Figures → 21-page manuscript → Peer review simulation

<p align="center">
  <a href="https://scitex.ai/demos/watch/scitex-automated-research/" title="▶ Watch full demo at scitex.ai/demos/">
    <img src="docs/assets/images/scitex-demo.gif" alt="SciTeX Demo" width="800">
  </a>
</p>

## 📦 Installation


``` bash
uv pip install scitex          # Core (minimal)
uv pip install scitex[plt,stats,scholar]  # Typical research setup
uv pip install scitex[all]     # Recommended: Full installation
```

## ⚙️ Configuration

Modular environment configuration via `.env.d/`:

<details>

```bash
# 1. Copy examples
cp -r .env.d.examples .env.d

# 2. Edit with your credentials
$EDITOR .env.d/

# 3. Source in shell (~/.bashrc or ~/.zshrc)
source /path/to/.env.d/entry.src
```

**Structure:**
```
.env.d/
├── entry.src              # Single entry point
├── 00_scitex.env          # Base settings (SCITEX_DIR)
├── 00_crossref-local.env  # CrossRef database
├── 00_figrecipe.env       # Plotting config
├── 01_scholar.env         # OpenAthens, API keys
├── 01_audio.env           # TTS backends
└── ...                    # Per-module configs
```

→ **[Full configuration reference](./.env.d.examples/README.md)**

</details>

## Three Interfaces

<details>
<summary><strong>🐍 Python API for Humans and AI Agents</strong></summary>

<br>

**`@stx.session`** — Reproducible Experiment Tracking

```python
import scitex as stx

@stx.session
def main(filename="demo.jpg"):
    fig, ax = stx.plt.subplots()
    ax.plot_line(t, signal)
    ax.set_xyt("Time (s)", "Amplitude", "Title")
    stx.io.save(fig, filename)
    return 0
```

**Output**:
```
script_out/FINISHED_SUCCESS/2025-01-08_12-30-00_AbC1/
├── demo.jpg                    # Figure with embedded metadata
├── demo.csv                    # Auto-exported plot data
├── CONFIGS/CONFIG.yaml         # Reproducible parameters
└── logs/{stdout,stderr}.log    # Execution logs
```

**`stx.io`** — Universal File I/O (30+ formats)

```python
stx.io.save(df, "output.csv")
stx.io.save(fig, "output.jpg")
df = stx.io.load("output.csv")
```

**`stx.stats`** — Publication-Ready Statistics (23 tests)

```python
result = stx.stats.test_ttest_ind(group1, group2, return_as="dataframe")
# Includes: p-value, effect size, CI, normality check, power
```

→ **[Full module status](./docs/MODULE_STATUS.md)**

</details>

<details>
<summary><strong>🖥️ CLI Commands for Humans and AI Agents</strong></summary>

<br>

```bash
scitex --help-recursive              # Show all commands
scitex scholar fetch "10.1038/..."   # Download paper by DOI
scitex scholar bibtex refs.bib       # Enrich BibTeX
scitex stats recommend               # Suggest statistical tests
scitex audio speak "Done"            # Text-to-speech
scitex capture snap                  # Screenshot

# List available APIs and tools
scitex list-python-apis              # List all Python APIs (210 items)
scitex mcp list-tools                # List all MCP tools (148 tools)
scitex introspect api scitex.stats   # List APIs for specific module
```

→ **[Full CLI reference](./docs/CLI_COMMANDS.md)**

</details>

<details>
<summary><strong>🔧 MCP Tools — 148 tools for AI Agents</strong></summary>

<br>

Turn AI agents into autonomous scientific researchers.

**Typical workflow**: Scholar (find papers) → Stats (analyze) → Plt (visualize) → Writer (manuscript) → Capture (verify)

| Category | Tools | Description |
|----------|-------|-------------|
| scholar | 23 | PDF download, metadata enrichment |
| stats | 10 | Automated statistical testing |
| plt | 9 | Matplotlib figure creation |
| capture | 12 | Screen monitoring and capture |
| audio | 10 | Text-to-speech, audio playback |
| introspect | 11 | Python introspection |
| diagram | 7 | Mermaid and flowchart creation |
| canvas | 7 | Scientific figure canvas |
| social | 9 | Social media posting |
| template | 4 | Project scaffolding |
| ui | 5 | Notifications |
| writer | 1 | LaTeX manuscript compilation |

**Claude Desktop** (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "scitex": {
      "command": "/home/ywatanabe/.venv/bin/scitex",
      "args": ["mcp", "start"],
      "env": {
        "SCITEX_ENV_SRC": "${SCITEX_ENV_SRC}"
      }
    }
  }
}
```

→ **[Full MCP tool reference](./docs/MCP_TOOLS.md)**

</details>

## 🧩 Standalone Packages

SciTeX integrates several standalone packages that can be used independently:

<details>

| Package | scitex Module | Description |
|---------|--------------|-------------|
| [figrecipe](https://github.com/ywatanabe1989/figrecipe) | `scitex.plt` | Publication-ready matplotlib figures |
| [crossref-local](https://github.com/ywatanabe1989/crossref-local) | `scitex.scholar.crossref_scitex` | Local CrossRef database (167M+ papers) |
| [openalex-local](https://github.com/ywatanabe1989/openalex-local) | `scitex.scholar.openalex_scitex` | Local OpenAlex database (250M+ papers) |
| [socialia](https://github.com/ywatanabe1989/socialia) | `scitex.social` | Social media posting (Twitter, LinkedIn) |
| [scitex-writer](https://github.com/ywatanabe1989/scitex-writer) | `scitex.writer` | LaTeX manuscript compilation |
| [scitex-dataset](https://github.com/ywatanabe1989/scitex-dataset) | `scitex.dataset` | Scientific dataset access |

Each package works standalone or as part of scitex:

```bash
pip install figrecipe        # Use independently
pip install scitex[plt]      # Or via scitex
```

</details>

## 📖 Documentation

- **[Read the Docs](https://scitex-python.readthedocs.io/)**: Complete API reference
- **[Example Notebooks](./examples/notebooks/)**: 25+ Jupyter notebooks

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

---

<p align="center">
  <a href="https://scitex.ai" target="_blank"><img src="docs/assets/images/scitex-icon-navy-inverted.png" alt="SciTeX" width="40"/></a>
  <br>
  AGPL-3.0 · ywatanabe@scitex.ai
</p>

<!-- EOF -->