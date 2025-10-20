# scitex.capture TODO

## Completed
- [x] Integrate cammy into scitex-code as scitex.capture
- [x] MCP server integration (mcp_server.py)
- [x] Documentation updated
- [x] Module accessible via `from scitex import capture`
- [x] Python API: `python -m scitex.capture <command>`

## In Progress
- [ ] Ensure `python -m scitex.capture --mcp` works for MCP server mode

## Recent Changes (2025-10-19)
- [x] Updated cache directory from `~/.cache/cammy` to `~/.scitex/capture`
- [x] Added automatic migration from legacy location
- [x] Consistent with other scitex modules (scholar, writer, etc.)

## Future Enhancements
- [ ] Add tests for MCP integration
- [ ] Add examples using MCP client
- [ ] Integrate with scitex.logging for consistent logging
- [ ] Add configuration file support integration with config/capture.yaml
- [ ] Add option to clean up legacy ~/.cache/cammy after successful migration

## Notes
The module now supports both:
1. Direct Python API: `capture.snap()`, `capture.start()`, etc.
2. MCP Server: Exposes functionality to AI agents via Model Context Protocol

Module structure:
- `__init__.py` - Public API exports
- `capture.py` - Core capture functionality
- `mcp_server.py` - MCP server implementation
- `utils.py` - Utility functions
- `gif.py` - GIF creation
- `cli.py` - CLI interface
- `__main__.py` - Module entry point
- `powershell/` - Windows PowerShell scripts

EOF
