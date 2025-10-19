<!-- ---
!-- Timestamp: 2025-10-17 23:10:00
!-- Author: ywatanabe
!-- File: /home/ywatanabe/proj/scitex_repo/src/scitex/scholar/TODO.md
!-- --- -->

# Scholar Module TODO

## For Development

### Impact Factor
- [ ] Combine core related to local CrossRef database

### Failure Analysis
- [ ] Identify unsupported journals and publishers
- [ ] Determine which translators require improvement
- [ ] Collect and organize screenshots/logs for debugging

## Zotero compatibility

- [ ] Migrate zotero-translators-python to this module
- [x] Import from zotero
- [x] Export to zotero

### Reference Management

All reference managers are now organized under `integration/` directory.

**Priority Guide:**
- 🔥 Critical (80% user coverage)
- ⭐ Important (growing platforms)
- 📦 Nice-to-have (niche/specialized)

---

#### 🔥 Tier 1: Critical - Already Complete

##### Zotero - ✅ Complete (Most Popular - Open Source)
- [x] Import: Bibliography with collections/tags, PDF annotations, paper metadata
- [x] Export: Manuscripts as preprint entries, project metadata, citation files (.bib, .ris)
- [x] Link: Live citation insertion, auto-update on library changes, tagged items
- [x] Moved to unified `integration/zotero/` location
- [x] Backward compatibility wrapper maintained
- **Market**: Largest user base, completely free, academia standard
- **Note**: Enrichment (citations, impact factors) handled by existing Scholar pipeline

##### Mendeley - ✅ Complete (Commercial Leader - Elsevier)
- [x] Import: Reference library, groups, annotations
- [x] Export: Bibliography exports, project citations
- [x] Link: Real-time citation sync, collaborative groups
- [x] Complete implementation in `integration/mendeley/`
- **Market**: Large user base in sciences, social features
- **Note**: Enrichment handled by existing Scholar pipeline

---

#### ⭐ Tier 2: Important - Next Priority

##### Paperpile - 🔄 Template Ready (Modern/Growing)
- [ ] Import: Reference library via BibTeX
- [ ] Export: Bibliography exports
- [ ] Link: Web API integration
- [x] Template structure created in `integration/paperpile/`
- **Market**: Fast-growing, Google Docs integration (unique), $36/year
- **Why Next**: Represents modern researchers, web-first approach
- **Implementation Priority**: MEDIUM-HIGH
  - [ ] Web API client
  - [ ] Google Docs sync
  - [ ] PDF annotation sync

##### EndNote - 🔄 Template Ready (Traditional Standard)
- [ ] Import: Reference library via XML/RIS
- [ ] Export: Bibliography exports, XML format
- [ ] Link: Desktop integration
- [x] Template structure created in `integration/endnote/`
- **Market**: Established in institutions, medical/clinical fields, $250+
- **Why Important**: Still used in many universities, institutional requirement
- **Implementation Priority**: MEDIUM
  - [ ] XML parser for EndNote library
  - [ ] RIS import/export
  - [ ] Desktop file sync

---

#### 📦 Tier 3: Nice-to-have - Lower Priority

##### RefWorks - 🔄 Template Ready (Institutional/Legacy)
- [ ] Import: Reference library via RIS/BibTeX
- [ ] Export: Bibliography exports
- [ ] Link: API integration (requires API access)
- [x] Template structure created in `integration/refworks/`
- **Market**: Institutional only, declining relevance
- **Why Low Priority**: Provided by libraries, limited individual adoption
- **Implementation Priority**: LOW

##### Papers (macOS) - 🔄 Template Ready (Platform-Specific)
- [ ] Import: Local library database
- [ ] Export: Bibliography exports
- [ ] Link: macOS integration
- [x] Template structure created in `integration/papers/`
- **Market**: macOS/iOS only, beautiful UI, $79+
- **Why Low Priority**: Limited to Apple ecosystem
- **Implementation Priority**: LOW

##### CiteDrive - 🔄 Template Ready (LaTeX-Focused)
- [ ] Import: BibTeX-based import
- [ ] Export: Bibliography exports
- [ ] Link: Real-time collaboration
- [x] Template structure created in `integration/citedrive/`
- **Market**: Small but growing, LaTeX/Overleaf users
- **Why Low Priority**: Very specific use case (LaTeX workflow)
- **Implementation Priority**: LOW

---

### Infrastructure Completed
- [x] Base classes for all reference managers (`integration/base.py`)
- [x] Unified API pattern (BaseImporter, BaseExporter, BaseLinker, BaseMapper)
- [x] Comprehensive documentation (`integration/README.md`)
- [x] Migration guide for Zotero relocation (`integration/MIGRATION.md`)
- [x] Citation insertion (BibTeX, RIS, APA, MLA, Chicago)
- [x] Source tracking for all metadata fields

---

### Recommended Development Order

**Phase 1: Core Complete** ✅
- Zotero and Mendeley implementations done
- Enrichment handled by existing Scholar pipeline (skip duplication)

**Phase 2: Add Paperpile (if needed)** ⭐
- Paperpile API client implementation
- Google Docs integration
- Modern web-based workflow

**Phase 3: Add EndNote (if needed)** ⭐
- EndNote XML parser
- Institutional workflows
- Medical/clinical field support

**Phase 4: Nice-to-haves (user-driven)** 📦
- RefWorks (if institutional demand)
- Papers (if macOS-focused users)
- CiteDrive (if LaTeX community demand)

**Phase 5: Advanced Features (if needed)**
- Unified CLI for all reference managers
- Comprehensive integration tests
- Performance optimization for large libraries

---

### Market Coverage Summary

**Current Coverage (Zotero + Mendeley)**: ~80% of researchers
- ✅ Open source leader
- ✅ Commercial leader
- ✅ Most disciplines covered
- ✅ Both individual and institutional users

**With Paperpile**: ~85-90% coverage
- ✅ Modern/web-first users
- ✅ Google Docs integration
- ✅ Growing academic segment

**With EndNote**: ~90-95% coverage
- ✅ Traditional institutions
- ✅ Medical/clinical fields
- ✅ University requirements

**Diminishing Returns**: RefWorks, Papers, CiteDrive add <5% each

<!-- EOF -->
