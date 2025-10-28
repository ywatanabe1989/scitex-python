# SciTeX Containers

Modular Singularity containers for SciTeX dependencies.

## Overview

Instead of installing ~10GB of Python packages and system dependencies, SciTeX uses lightweight Singularity containers that are downloaded on-demand.

### Available Containers

| Container | Size | Includes | Required For |
|-----------|------|----------|--------------|
| `scitex-core.sif` | ~500MB | Python 3.10, numpy, scipy, pandas, matplotlib | Core functionality |
| `scitex-writer.sif` | ~2GB | Full TeXLive, ImageMagick, Mermaid CLI | LaTeX compilation |
| `scitex-scholar.sif` | ~1GB | Chrome (headless), Selenium, scholarly, pymed | Literature management, web scraping |
| `scitex-ml.sif` | ~3GB | PyTorch, Transformers, scikit-learn, optuna | Machine learning (optional) |

## Building Containers

### Prerequisites

- Linux system (or WSL2 on Windows)
- Singularity/Apptainer installed
- `sudo` access
- ~10GB free disk space
- Good internet connection

### Install Singularity

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y singularity-container
```

**Or install from source:**
```bash
# See: https://sylabs.io/guides/latest/user-guide/quick_start.html
```

### Build All Containers

```bash
cd /home/ywatanabe/proj/scitex-code/containers
sudo ./build_containers.sh
```

This will create all `.sif` files in `../build/containers/`:
- `scitex-core.sif`
- `scitex-writer.sif`
- `scitex-scholar.sif`
- `scitex-ml.sif`

**Estimated time**: 30-60 minutes (depending on internet speed)

### Build Specific Container

```bash
sudo ./build_containers.sh core      # Just core (~5 min)
sudo ./build_containers.sh writer    # Just writer (~15 min)
sudo ./build_containers.sh scholar   # Just scholar (~10 min)
sudo ./build_containers.sh ml        # Just ml (~20 min)
```

## Using Containers

### Basic Usage

```bash
# Run Python script in core container
singularity exec scitex-core.sif python your_script.py

# Compile LaTeX in writer container
singularity exec scitex-writer.sif pdflatex manuscript.tex

# Run web scraping in scholar container (with Chrome)
singularity exec scitex-scholar.sif python scrape_papers.py

# Train ML model in ml container (with GPU)
singularity exec --nv scitex-ml.sif python train.py
```

### App-Specific Commands

```bash
# Writer container apps
singularity run --app pdflatex scitex-writer.sif manuscript.tex
singularity run --app bibtex scitex-writer.sif manuscript
singularity run --app convert scitex-writer.sif image.png image.pdf

# Scholar container apps
singularity run --app chromium scitex-scholar.sif --dump-dom https://example.com
```

### Binding Directories

By default, Singularity binds your home directory. To bind additional directories:

```bash
singularity exec --bind /data:/data scitex-core.sif python script.py
```

## Chrome/Browser in scitex-scholar

### Headless Browser Architecture

The `scitex-scholar` container includes **Chromium** running in **headless mode**, which means:

✅ **No GUI required** - Works on HPC clusters without display
✅ **No X11 forwarding needed** - Runs completely headless
✅ **Selenium support** - Full browser automation
✅ **Container-safe** - Runs with `--no-sandbox` flag

### Example: Web Scraping with Chrome

```python
# scrape_scholar.py
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# Configure for container environment
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--disable-gpu')
chrome_options.binary_location = '/usr/bin/chromium-browser'

service = Service('/usr/bin/chromedriver')
driver = webdriver.Chrome(service=service, options=chrome_options)

# Scrape page
driver.get('https://scholar.google.com/scholar?q=deep+learning')
print(driver.page_source)
driver.quit()
```

**Run in container:**
```bash
singularity exec scitex-scholar.sif python scrape_scholar.py
```

### Why This Works

1. **Chromium is installed** inside the container (not on host)
2. **ChromeDriver is included** for Selenium automation
3. **Headless mode** means no display needed
4. **Container isolation** prevents permission issues

### Alternative: Using scholarly (No Browser)

For Google Scholar, you can use the `scholarly` Python package which doesn't need a browser:

```python
from scholarly import scholarly

# Search for papers
search_query = scholarly.search_pubs('machine learning')
paper = next(search_query)
print(paper['bib']['title'])
```

This is faster and doesn't require Chrome!

## Testing Containers

### Test Core Container

```bash
singularity exec scitex-core.sif python -c "import numpy; print(numpy.__version__)"
singularity exec scitex-core.sif python -c "import pandas; print(pandas.__version__)"
```

### Test Writer Container

```bash
singularity exec scitex-writer.sif pdflatex --version
singularity exec scitex-writer.sif bibtex --version
singularity exec scitex-writer.sif convert --version
```

### Test Scholar Container (Chrome)

```bash
# Check Chrome version
singularity exec scitex-scholar.sif chromium-browser --version

# Check ChromeDriver
singularity exec scitex-scholar.sif chromedriver --version

# Test headless Chrome
singularity exec scitex-scholar.sif chromium-browser \
    --headless --no-sandbox --dump-dom https://example.com

# Test Selenium
singularity exec scitex-scholar.sif python -c "import selenium; print(selenium.__version__)"
```

### Test ML Container

```bash
singularity exec scitex-ml.sif python -c "import torch; print(torch.__version__)"
singularity exec scitex-ml.sif python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
singularity exec --nv scitex-ml.sif python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Distributing Containers

### Option 1: GitHub Releases

1. Build containers locally
2. Create GitHub release: `v1.0.0`
3. Upload `.sif` files as release assets
4. Users download via `wget` or Python code

```bash
# Example download
wget https://github.com/ywatanabe1989/scitex-code/releases/download/v1.0.0/scitex-core.sif
```

### Option 2: Singularity Hub / Container Library

```bash
# Push to Singularity Library
singularity push scitex-core.sif library://ywatanabe/scitex/core:1.0.0

# Users pull with
singularity pull library://ywatanabe/scitex/core:1.0.0
```

## Container Sizes

Expected sizes after building:

```
scitex-core.sif        ~500 MB
scitex-writer.sif      ~2.0 GB
scitex-scholar.sif     ~1.0 GB
scitex-ml.sif          ~3.0 GB
───────────────────────────────
Total:                 ~6.5 GB
```

Users only download what they need!

## Troubleshooting

### "Cannot build, permission denied"

Run with `sudo`:
```bash
sudo ./build_containers.sh
```

### "Singularity not found"

Install Singularity:
```bash
sudo apt-get install singularity-container
```

### Chrome fails in scholar container

Make sure to use headless flags:
```python
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
```

### GPU not detected in ml container

Use `--nv` flag:
```bash
singularity exec --nv scitex-ml.sif python train.py
```

## Next Steps

1. **Build containers**: `sudo ./build_containers.sh`
2. **Test containers**: Run test commands above
3. **Upload to GitHub**: Create release and upload `.sif` files
4. **Update Python code**: Implement auto-download in `scitex/containers.py`
5. **Update documentation**: Add container usage to main README

## See Also

- [Singularity Documentation](https://sylabs.io/docs/)
- [Container Definition Files](https://sylabs.io/guides/latest/user-guide/definition_files.html)
- [../docs/CONTAINERIZATION_STRATEGY.md](../docs/CONTAINERIZATION_STRATEGY.md) - Overall strategy
