# SciTeX MCP Tools (145 total)

Model Context Protocol tools for AI agent integration.

```bash
scitex mcp list-tools             # List all tools with full descriptions
scitex serve                      # Start MCP server (stdio)
scitex serve -t http --port 8085  # HTTP transport
```

## Tools by Category

| Category | Tool | Description |
|----------|------|-------------|
| **audio** (12) | `audio_announce_context` | Announce cwd and git branch |
|  | `audio_check_audio_status` | Check audio connectivity |
|  | `audio_clear_audio_cache` | Clear audio cache |
|  | `audio_generate_audio` | Generate speech audio file |
|  | `audio_list_audio_files` | List generated audio |
|  | `audio_list_backends` | List TTS backends |
|  | `audio_list_voices` | List voices for backend |
|  | `audio_play_audio` | Play audio file |
|  | `audio_speak` | Text-to-speech with fallback engines |
|  | `audio_speak_local` | TTS on server (local playback) |
|  | `audio_speak_relay` | TTS via relay (remote playback) |
|  | `audio_speech_queue_status` | Get speech queue status |
| **canvas** (7) | `canvas_add_panel` | Add panel to canvas |
|  | `canvas_canvas_exists` | Check canvas exists |
|  | `canvas_create_canvas` | Create figure canvas |
|  | `canvas_export_canvas` | Export to PNG/PDF/SVG |
|  | `canvas_list_canvases` | List all canvases |
|  | `canvas_list_panels` | List panels in canvas |
|  | `canvas_remove_panel` | Remove panel |
| **capture** (12) | `capture_analyze_screenshot` | Analyze for errors |
|  | `capture_capture_screenshot` | Capture screenshot |
|  | `capture_capture_window` | Capture specific window |
|  | `capture_clear_cache` | Clear screenshot cache |
|  | `capture_create_gif` | Create animated GIF |
|  | `capture_get_info` | Get monitor/window info |
|  | `capture_get_monitoring_status` | Get monitoring status |
|  | `capture_list_recent_screenshots` | List recent screenshots |
|  | `capture_list_sessions` | List monitoring sessions |
|  | `capture_list_windows` | List visible windows |
|  | `capture_start_monitoring` | Start continuous capture |
|  | `capture_stop_monitoring` | Stop monitoring |
| **diagram** (7) | `diagram_compile_graphviz` | Compile to Graphviz |
|  | `diagram_compile_mermaid` | Compile to Mermaid |
|  | `diagram_create_diagram` | Create from YAML spec |
|  | `diagram_get_paper_modes` | Get paper layout modes |
|  | `diagram_get_preset` | Get preset config |
|  | `diagram_list_presets` | List diagram presets |
|  | `diagram_split_diagram` | Split large diagrams |
| **introspect** (12) | `introspect_q` | Get function signature (like func?) |
|  | `introspect_qq` | Get source code (like func??) |
|  | `introspect_dir` | List module members (like dir()) |
|  | `introspect_api` | List full API tree |
|  | `introspect_docstring` | Get docstring |
|  | `introspect_exports` | Get __all__ exports |
|  | `introspect_examples` | Find usage examples |
|  | `introspect_class_hierarchy` | Get class MRO |
|  | `introspect_type_hints` | Get type hints |
|  | `introspect_imports` | Get module imports |
|  | `introspect_dependencies` | Get dependencies |
|  | `introspect_call_graph` | Get call graph |
| **plt** (9) | `plt_compose` | Compose multi-panel figure |
|  | `plt_crop` | Crop whitespace |
|  | `plt_extract_data` | Extract plotted data |
|  | `plt_get_plot_types` | Get supported plot types |
|  | `plt_info` | Get recipe info |
|  | `plt_list_styles` | List style presets |
|  | `plt_plot` | Create figure from spec |
|  | `plt_reproduce` | Reproduce from recipe |
|  | `plt_validate` | Validate recipe |
| **scholar** (23) | `scholar_add_papers_to_project` | Add papers to project |
|  | `scholar_authenticate` | SSO authentication |
|  | `scholar_cancel_job` | Cancel job |
|  | `scholar_check_auth_status` | Check auth status |
|  | `scholar_create_project` | Create project |
|  | `scholar_download_pdf` | Download single PDF |
|  | `scholar_download_pdfs_batch` | Batch PDF download |
|  | `scholar_enrich_bibtex` | Enrich BibTeX metadata |
|  | `scholar_export_papers` | Export papers |
|  | `scholar_fetch_papers` | Fetch papers (async) |
|  | `scholar_get_job_result` | Get job result |
|  | `scholar_get_job_status` | Get job status |
|  | `scholar_get_library_status` | Get library status |
|  | `scholar_list_jobs` | List background jobs |
|  | `scholar_list_projects` | List projects |
|  | `scholar_logout` | Logout from auth |
|  | `scholar_parse_bibtex` | Parse BibTeX file |
|  | `scholar_parse_pdf_content` | Parse PDF content |
|  | `scholar_resolve_dois` | Resolve DOIs from titles |
|  | `scholar_resolve_openurls` | Resolve OpenURLs |
|  | `scholar_search_papers` | Search papers |
|  | `scholar_start_job` | Start pending job |
|  | `scholar_validate_pdfs` | Validate PDF files |
| **social** (9) | `social_analytics` | Get analytics |
|  | `social_check` | Check platform connection |
|  | `social_check_availability` | Check socialia installed |
|  | `social_delete` | Delete post |
|  | `social_feed` | Get platform feed |
|  | `social_me` | Get user profile |
|  | `social_post` | Post to platform |
|  | `social_status` | Check config status |
|  | `social_thread` | Post thread |
| **stats** (10) | `stats_correct_pvalues` | Multiple comparison correction |
|  | `stats_describe` | Descriptive statistics |
|  | `stats_effect_size` | Calculate effect size |
|  | `stats_format_results` | Format in journal style |
|  | `stats_normality_test` | Test normality |
|  | `stats_p_to_stars` | P-value to stars |
|  | `stats_posthoc_test` | Post-hoc comparisons |
|  | `stats_power_analysis` | Power/sample size calc |
|  | `stats_recommend_tests` | Recommend statistical tests |
|  | `stats_run_test` | Execute statistical test |
| **template** (6) | `template_clone_template` | Clone template |
|  | `template_get_code_template` | Get code template (session, io, plt, stats, etc.) |
|  | `template_get_template_info` | Get template info |
|  | `template_list_code_templates` | List code templates |
|  | `template_list_git_strategies` | List git strategies |
|  | `template_list_templates` | List project templates |
| **ui** (5) | `ui_available_notification_backends` | Available backends |
|  | `ui_get_notification_config` | Get notification config |
|  | `ui_list_notification_backends` | List backends |
|  | `ui_notify` | Send notification |
|  | `ui_notify_by_level` | Notify by level |
| **writer** (1) | `writer_usage` | LaTeX compilation guide |

## Configuration

### All-in-One Server (Recommended)

**Claude Desktop** (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "scitex": {
      "command": "scitex-mcp-server"
    }
  }
}
```

### Individual Servers

```json
{
  "mcpServers": {
    "scitex-audio": { "command": "scitex-audio" },
    "scitex-capture": { "command": "scitex-capture" },
    "scitex-scholar": { "command": "scitex-scholar" }
  }
}
```
