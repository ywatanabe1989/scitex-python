<!-- ---
!-- Timestamp: 2025-12-21 11:02:24
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/TODO.md
!-- --- -->

## Simplify Things
## Explict Names
## Kinds
### figure
### plot

#### Current
```
├── artifacts
│   ├── cache
│   │   ├── geometry_px.json
│   │   ├── hitmap.png
│   │   ├── hitmap.svg
│   │   └── render_manifest.json
│   └── exports
│       ├── figure.pdf
│       ├── figure.png
│       └── figure.svg
├── canonical
│   ├── data_info.json
│   ├── encoding.json
│   ├── spec.json
│   └── theme.json
├── children
└── payload
    └── data.csv

```

#### Revised
```
├── artifacts
│   ├── cache
│   │   ├── coordinates/
    
│   │   ├── hitmap.png
│   │   ├── hitmap.svg
│   │   └── render_manifest.json
│   └── exports
│       ├── figure.pdf
│       ├── figure.png
│       └── figure.svg
├── canonical
│   ├── data_info.json
│   ├── encoding.json
│   ├── spec.json
│   └── theme.json
├── children
└── payload
    └── data.csv

```

### image
### shape
### stats
### table
### text

<!-- EOF -->