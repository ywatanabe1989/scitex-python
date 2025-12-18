# Refactoring Plan: File Size Compliance

## COMPLETED - 2025-12-19

All target files have been refactored to comply with the 512-line limit.

## Summary

| File | Original | Final | Max Allowed | Status |
|------|----------|-------|-------------|--------|
| src/scitex/io/_save.py | 2110 | 445 | 512 | ✅ Complete |
| src/scitex/fig/_bundle.py | 1675 | 508 | 512 | ✅ Complete |
| src/scitex/cli/convert.py | 404 | 404 | 512 | ✅ Already Compliant |
| src/scitex/plt/_bundle.py | 345 | 345 | 512 | ✅ Already Compliant |
| src/scitex/fig/editor/edit/bundle_resolver.py | 323 | 323 | 512 | ✅ Already Compliant |

## Extracted Modules

### src/scitex/io/_save_modules/
- `_stx_bundle.py` - STX bundle saving (103 lines)
- `_pltz_stx.py` - PLTZ to STX conversion (194 lines)
- `_pltz_bundle.py` - PLTZ bundle saving (352 lines)
- `_legends.py` - Separate legend saving (87 lines)
- `_image_csv.py` - Image with CSV export (492 lines)
- `_figure_utils.py` - Figure data extraction (91 lines)
- `_symlink.py` - Symlink utilities (58 lines)

### src/scitex/fig/_figz_modules/
- `_render.py` - Preview rendering (178 lines)
- `_save.py` - Bundle saving (144 lines)
- `_validate.py` - Bundle validation (95 lines)
- `_geometry.py` - Geometry extraction (73 lines)
- `_element_ops.py` - Element operations (130 lines)
- `_legacy.py` - Deprecated methods (62 lines)
