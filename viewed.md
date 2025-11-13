94 lines & 182 words

# Repository View
#### Repository: `/home/ywatanabe/proj/scitex-code/examples/metadata_demo`
#### Output: `/home/ywatanabe/proj/scitex-code/viewed.md`

## Configurations
##### Tree:
- Maximum depth: 3
- .gitignore respected
- Blacklist expresssions:
```plaintext
node_modules,.*,*.py[cod],__pycache__,*.elc,env,env-[0-9]*.[0-9]*,[1-2][0-9][0-9
][0-9]Y-*,htmlcov,*.sif,*.img,*.image,*.sandbox,*.log,logs,build,dist,*_back,*_b
ackup,*old*,.old,RUNNING,FINISHED
```

#### File content:
- Number of head: 50
- Whitelist extensions:
```plaintext
.txt,.md,.org,.el,.sh,.py,.yaml,.yml,.json,.def
```
- Blacklist expressions:
```plaintext
*.mat,*.npy,*.npz,*.csv,*.pkl,*.jpg,*.jpeg,*.mp4,*.pth,*.db*,*.out,*.err,*.cbm,*
.pt,*.egg-info,*.aux,*.pdf,*.png,*.tiff,*.wav
```


## Tree contents
/home/ywatanabe/proj/scitex-code/examples/metadata_demo/
├── demo_jpg_out
│   └── demo_fig_with_metadata.jpg
└── demo_jpg.py


## File contents

### `/home/ywatanabe/proj/scitex-code/examples/metadata_demo/demo_jpg.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-14 08:25:22 (ywatanabe)"


"""
Demo: Minimal metadata embedding in JPG file
"""

from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import scitex as stx


def demo_without_qr(filename):
    """Show metadata without QR code (just embedded)."""

    fig, ax = stx.plt.subplots()

    t = np.linspace(0, 2, 1000)
    signal = np.sin(2 * np.pi * 5 * t) * np.exp(-t / 2)

    ax.plot(t, signal, "b-", linewidth=2)
    ax.set_xyt(
        "Time (s)",
        "Amplitude",
        "Clean Figure (metadata embedded, no QR overlay)",
    )

    # Saving
    stx.io.save(
        fig,
        filename,
        metadata={"exp": "s01", "subj": "S001"},
        symlink_from_cwd=True,
    )
    plt.close()

    # Loading
    ldir = __file__.replace(".py", "_out")
    img, meta = stx.io.load(f"{ldir}/{filename}")
    pprint(meta)


@stx.session.session
def main(filename="demo_fig_with_metadata.jpg"):
    """Run all demos."""

    demo_without_qr(filename)

...
```

