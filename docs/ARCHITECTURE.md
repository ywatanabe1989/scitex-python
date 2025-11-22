<!-- ---
!-- Timestamp: 2025-11-22 01:29:04
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex-cloud/docs/ARCHITECTURE.md
!-- --- -->

## SciTeX Cloud
- [ ] Live at http:127.0.0.1:8000 (https://scitex.ai)

## SciTeX Files
- [ ] Live at http:127.0.0.1:8000/<username>/<project-name>

## SciTeX Writer
- [ ] https://github.com/ywatanabe1989/scitex-writer
- [ ] pip install scitex
  - [ ] import scitex.writer
  - [ ] $ scitex writer --help
  - [ ] http:127.0.0.1:8000/writer/
- [ ] Section-separated Writing
- [ ] Collaborative Writing
- [ ] Always linked to Files (AI-Native)
- [ ] Statistics incorporated (optional with scitex.stats)
- [ ] Context incorporated (optional with SciTeX Scholar, Vis, Code, Files)

## SciTeX Scholar
- [ ] https://github.com/ywatanabe1989/scitex-code/src/scitex/scholar
- [ ] pip install scitex
  - [ ] import scitex.scholar
  - [ ] $ scitex scholar --help
  - [ ] http:127.0.0.1:8000/scholar/bibtex/
- [ ] Abstract appended (AI-Native)


## SciTeX Vis
- [ ] https://github.com/ywatanabe1989/scitex-code/src/scitex/{plt,vis}
- [ ] pip install scitex
  - [ ] import scitex.plt
  - [ ] import scitex.vis
  - [ ] http:127.0.0.1:8000/vis/sigma/
- [ ] Reproducibility
  - [ ] Metadata Embedded
  - [ ] CSV Created
  - [ ] JSON Created (style)
  - [ ] Plot <-> text data
- [ ] Flexibility
  - [ ] Style Change
  - [ ] mm-level adjustment
- [ ] UI
  - [ ] GUI for Layout
  - [ ] GUI for style change
- [ ] (Optional) Statistics with scitex.stats
- [ ] (Optional) Metadta with scitex.plt

## SciTeX Code
- [ ] https://github.com/ywatanabe1989/scitex-code/src/scitex/{io,logging,plt,vis,...}
- [ ] pip install scitex
  - [ ] import scitex
  - [ ] $ scitex ...
  - [ ] http:127.0.0.1:8000/code/
- [ ] Work in local, scitex.ai, and self-hostable
- [ ] Reusable, reproducible modules availabel

## Loosely Coupling Diagram

## Loosely Coupling Diagram

Each module can work independently. However, loosely coupling enables synergy.

``` mermaid
graph TB
    subgraph "User Access"
        PY[Python API]
        CLI[Command Line]
        WEB[Web Browser]
    end
    
    subgraph "SciTeX Modules with Unique Strengths"
        FILES["📁 Files Hub<br/>└─ Central Integration<br/>└─ Version Control<br/>└─ User/Project Structure"]
        
        WRITER["✍️ Writer<br/><b>Strengths:</b><br/>• Section-Separated Writing<br/>• Collaborative Editing<br/>• AI-Native (Auto Files)<br/>• Stats Integration<br/>• Context-Aware"]
        
        SCHOLAR["📚 Scholar<br/><b>Strength:</b><br/>• Abstract Appended<br/>  (AI-Native)<br/>• Citation Enrichment<br/>• Auto-metadata"]
        
        VIS["📊 Vis<br/><b>Strengths:</b><br/>• Reproducibility<br/>  - Metadata Embedded<br/>  - CSV + JSON Export<br/>  - Plot ↔ Text<br/>• Flexibility<br/>  - Style Change<br/>  - mm-level Precision<br/>• GUI Layout/Style"]
        
        CODE["💻 Code<br/><b>Strengths:</b><br/>• Work Anywhere<br/>  (Local/Cloud/Self-host)<br/>• Reusable Modules<br/>• Reproducible<br/>  by Default"]
    end
    
    subgraph "Infrastructure"
        CLOUD[SciTeX Cloud<br/>scitex.ai]
        DB[(Database)]
    end
    
    PY --> FILES & WRITER & SCHOLAR & CODE & VIS
    CLI --> FILES & WRITER & SCHOLAR & CODE & VIS
    WEB --> CLOUD
    
    FILES <-->|LaTeX, PDF<br/>Sections| WRITER
    FILES <-->|BibTeX, PDFs<br/>Abstracts| SCHOLAR
    FILES <-->|Scripts, Data<br/>Outputs| CODE
    FILES <-->|Data Files<br/>Figures| VIS
    
    WRITER <-.->|Auto-Citations| SCHOLAR
    WRITER <-.->|Embed Stats| CODE
    WRITER <-.->|Include Figs| VIS
    CODE <-.->|Generate Plots| VIS
    
    WRITER & SCHOLAR & CODE & VIS & FILES --> CLOUD
    CLOUD --> DB
    
    style FILES fill:#ffd700,stroke:#ff8c00,stroke-width:4px,color:#000
    style WRITER fill:#90EE90,stroke:#228B22,stroke-width:3px,color:#000
    style SCHOLAR fill:#87CEEB,stroke:#4682B4,stroke-width:3px,color:#000
    style CODE fill:#DDA0DD,stroke:#9370DB,stroke-width:3px,color:#000
    style VIS fill:#FFB6C1,stroke:#FF69B4,stroke-width:3px,color:#000
```

<!-- EOF -->