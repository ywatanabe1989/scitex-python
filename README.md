<!-- ---
!-- Timestamp: 2026-01-20 09:23:14
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-python/README.md
!-- --- -->

<p align="center">
  <a href="https://scitex.ai">
    <img src="docs/assets/images/scitex-logo-navy-bg.png" alt="SciTeX" width="400">
  </a>
</p>

<p align="center">
  <a href="https://scitex.ai">scitex.ai</a> · <code>pip install scitex</code>
</p>

---

**Modular Python framework for AI-driven scientific research automation.**

Extended capabilities for AI agents in academic work — each component works independently and adapts to various workflows. Handles both primary and secondary research with real or simulated data.

- **108 MCP tools** for Claude/AI agent integration
- **Standardized outputs** with reproducible session tracking
- **Fast boilerplates** — 70% less code for research pipelines
- **30+ I/O formats** with unified API

## 🎬 Demo

**40 min, zero human intervention** — AI agent conducts full research pipeline:

> Literature search → Data analysis → Statistics → Figures → 21-page manuscript → Peer review simulation

<p align="center">
  <a href="https://scitex.ai/demos/watch/scitex-automated-research/" title="▶ Watch full demo at scitex.ai/demos/">
    <img src="docs/assets/images/scitex-demo.gif" alt="SciTeX Demo" width="800">
  </a>
</p>

<p align="center">
  <a href="https://badge.fury.io/py/scitex"><img src="https://badge.fury.io/py/scitex.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/scitex/"><img src="https://img.shields.io/pypi/pyversions/scitex.svg" alt="Python Versions"></a>
  <a href="https://github.com/ywatanabe1989/scitex-python/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ywatanabe1989/scitex-python" alt="License"></a>
  <img src="https://img.shields.io/badge/uv-recommended-blue" alt="uv recommended">
</p>

<details>
<summary><strong>🔧 MCP Tools - 108 tools for Claude Integration</strong></summary>

<br>

Turn AI agents into autonomous scientific researchers. Tools allow AI to "reach out" of the chat window and perform real-world actions — compiling LaTeX, searching databases, running statistical tests.

**Typical workflow**: Scholar (find papers) → Stats (analyze) → Plt (visualize) → Writer (manuscript) → Capture (verify)

| Category | Tools | Description |
|----------|-------|-------------|
| audio | 10 | Text-to-speech, audio playback |
| canvas | 7 | Scientific figure canvas |
| capture | 12 | Screen monitoring and capture |
| diagram | 7 | Mermaid and flowchart creation |
| introspect | 11 | Python introspection (signatures, source, docs) |
| plt | 9 | Matplotlib figure creation |
| scholar | 23 | PDF download, metadata enrichment |
| social | 9 | Social media posting |
| stats | 10 | Automated statistical testing |
| template | 4 | Project scaffolding |
| ui | 5 | Notifications |
| writer | 1 | LaTeX manuscript compilation |

```bash
scitex mcp list-tools             # List all tools with full descriptions
scitex serve                      # Start MCP server (stdio)
```

→ **[Full tool reference](./docs/MCP_TOOLS.md)**

</details>

<details>
<summary><strong>Module Status - Tests & Installation</strong></summary>

<br>

**Core**: io, path, str, dict, types, config, utils, decorators, logging, gen
**Data Science**: stats, pd, linalg, plt, dsp
**AI/ML**: ai, nn, torch
**System**: cli, sh, git, session, resource, db
**Research**: writer, tex, msword, scholar, diagram
**Web**: web, browser
**Other**: audio, capture, repro, benchmark, security, dt, dev, schema, bridge

→ **[Full module status](./docs/MODULE_STATUS.md)**

</details>

## 📦 Installation

``` bash
uv pip install scitex[all]     # Recommended: Full installation with all modules
uv pip install scitex          # Core only (numpy, pandas, PyYAML, tqdm)
uv pip install scitex[heavy]   # Include heavy deps (torch, mne, optuna, etc.)
```

> **Note**: Heavy dependencies (torch, mne, optuna, catboost, jax, tensorflow, umap-learn)
> are optional and NOT included by default. Modules gracefully handle missing dependencies.

<details>
<summary><strong>Arial Font Setup for Figures</strong></summary>

``` bash
# Ubuntu/WSL
sudo apt install -y ttf-mscorefonts-installer
mkdir -p ~/.local/share/fonts/windows
cp /mnt/c/Windows/Fonts/arial*.ttf ~/.local/share/fonts/windows/
fc-cache -fv
rm ~/.cache/matplotlib -rf
```

</details>

## Why SciTeX?

SciTeX automates research analysis code.

<details>
<summary><strong><code>@scitex.session</code></strong> — Reproducible Experiment Tracking + Auto-CLI</summary>

```python
import scitex as stx

@stx.session
def main(
    CONFIG=stx.INJECTED,   # Auto-injected from ./config/*.yaml
    plt=stx.INJECTED,      # Pre-configured matplotlib
    logger=stx.INJECTED,   # Session logger
):
    """Docstring becomes --help description."""
    stx.io.save(results, "results.csv")
    return 0
```

**Output Structure**:
```
script_out/FINISHED_SUCCESS/2025-01-08_12-30-00_AbC1/
├── results.csv
├── CONFIGS/{CONFIG.pkl, CONFIG.yaml}
└── logs/{stderr.log, stdout.log}
```

</details>

<details>
<summary><strong><code>scitex.io</code></strong> — Universal File I/O (30+ formats)</summary>

```python
import scitex as stx

stx.io.save(df, "output.csv")
stx.io.save(arr, "output.npy")
stx.io.save(fig, "output.jpg")

df = stx.io.load("output.csv")
```

</details>

<details>
<summary><strong><code>scitex.plt</code></strong> — Auto-Export Figures with Data</summary>

```python
fig, ax = stx.plt.subplots()
ax.plot_line(x, y)  # Data tracked
stx.io.save(fig, "plot.png")  # Creates: plot.png + plot.csv
```

</details>

<details>
<summary><strong><code>scitex.stats</code></strong> — Publication-Ready Statistics (23 tests)</summary>

```python
result = stx.stats.test_ttest_ind(
    group1, group2,
    return_as="dataframe"  # or "latex", "markdown"
)
# Includes: p-value, effect size, CI, normality check, power
```

</details>

<details>
<summary><strong><code>scitex.scholar</code></strong> — Literature Management</summary>

```bash
scitex scholar bibtex papers.bib --project myresearch
```

Enriches BibTeX with abstracts, DOIs, impact factors.

</details>

<details>
<summary><strong><code>scitex.ai</code></strong> — Unified AI Interface (7 providers)</summary>

```python
from scitex.ai.genai import GenAI

ai = GenAI(provider="openai")
response = ai("Explain this data pattern")

ai = GenAI(provider="anthropic", model="claude-3-opus-20240229")
```

</details>

## 🖥️ CLI Commands

```bash
scitex --help-recursive    # Show all available commands
```

See **[docs/CLI_COMMANDS.md](./docs/CLI_COMMANDS.md)** for full reference.

## 🔌 MCP Server Configuration

<details>
<summary><strong>Claude Desktop Setup</strong></summary>

**All-in-One** (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "scitex": { "command": "scitex-mcp-server" }
  }
}
```

**Individual servers**: `scitex-audio`, `scitex-capture`, `scitex-scholar`, etc.

</details>

## 🚀 Quick Start

### Research Analysis with **70% Less Code**

**With SciTeX** ([57 lines](./examples/demo_session_plt_io.py)):

```python
import scitex as stx

@stx.session
def main(filename="demo.jpg"):
    fig, ax = stx.plt.subplots()
    ax.plot_line(t, signal)
    ax.set_xyt("Time (s)", "Amplitude", "Title")
    stx.io.save(fig, filename, metadata={"exp": "s01"}, symlink_to="./data")
    return 0
```

<details>
<summary><strong>Equivalent without SciTeX: 188 lines</strong></summary>

See [examples/demo_session_plt_io_pure_python.py](./examples/demo_session_plt_io_pure_python.py) — requires 3.3× more code for identical output.

</details>

## 📖 Documentation

- **[Read the Docs](https://scitex.readthedocs.io/)**: Complete API reference
- **[Example Notebooks](./examples/notebooks/)**: 25+ Jupyter notebooks
- **[MCP Tools](./docs/MCP_TOOLS.md)**: 108 AI agent tools
- **[Module Status](./docs/MODULE_STATUS.md)**: Test badges and install info

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

---

<p align="center">
  <a href="https://scitex.ai" target="_blank"><img src="docs/assets/images/scitex-icon-navy-inverted.png" alt="SciTeX" width="40"/></a>
  <br>
  AGPL-3.0 · ywatanabe@scitex.ai
</p>

<!-- EOF -->
