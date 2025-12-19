#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/_bundle.py

"""Figz - Unified Element API for .stx bundles."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from scitex.io.bundle import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    TYPE_DEFAULTS,
    ZipBundle,
    generate_bundle_id,
    normalize_spec,
)

from ._figz_modules import (
    FigzCaptionMixin,
    FigzLegacyMixin,
    extract_geometry,
    figure_to_stx_bytes,
    get_content_extension,
    process_content,
    render_preview_internal,
    save_to_directory,
    save_to_zip,
    validate_can_add_child,
)
from .layout import normalize_position, normalize_size


def _is_directory_bundle(path: Path) -> bool:
    return path.suffix == ".d" and path.is_dir()


def _is_stx_path(path: Path) -> bool:
    return path.suffix == ".stx" or (path.suffix == ".d" and path.stem.endswith(".stx"))


__all__ = ["Figz"]


class Figz(FigzCaptionMixin, FigzLegacyMixin):
    """Unified Element API for .stx bundles."""

    SCHEMA = {"name": SCHEMA_NAME, "version": SCHEMA_VERSION}
    DEFAULT_SIZE_MM = {"width": 170, "height": 120}
    CONTAINER_TYPES = {"figure", "plot"}
    LEAF_TYPES = {"text", "shape", "image", "stats", "symbol", "equation", "comment"}

    def __init__(
        self,
        path: Union[str, Path],
        name: Optional[str] = None,
        size_mm: Optional[Union[Dict[str, float], tuple]] = None,
        bundle_type: str = "figure",
    ):
        self.path = Path(path)
        self._spec: Optional[Dict[str, Any]] = None
        self._style: Optional[Dict[str, Any]] = None
        self._modified = False
        if self.path.exists():
            self._is_dir = _is_directory_bundle(self.path)
            self._is_stx = _is_stx_path(self.path)
            self._load()
            if name is not None or size_mm is not None:
                self._check_conflicts(name, size_mm, bundle_type)
        else:
            if name is None:
                raise FileNotFoundError(
                    f"Bundle not found: {self.path}. Provide 'name' to create."
                )
            self._create_new(name, size_mm, bundle_type)

    def _load(self) -> None:
        if self._is_dir:
            spec_path, style_path = self.path / "spec.json", self.path / "style.json"
            if spec_path.exists():
                with open(spec_path, encoding="utf-8") as f:
                    self._spec = normalize_spec(json.load(f), "figure")
                self._migrate_panels_to_elements()
            else:
                self._spec = self._create_default_spec("Untitled")
            self._style = (
                json.load(open(style_path, encoding="utf-8"))
                if style_path.exists()
                else {}
            )
        else:
            with ZipBundle(self.path, mode="r") as zb:
                try:
                    self._spec = normalize_spec(zb.read_json("spec.json"), "figure")
                    self._migrate_panels_to_elements()
                except FileNotFoundError:
                    self._spec = self._create_default_spec("Untitled")
                try:
                    self._style = zb.read_json("style.json")
                except FileNotFoundError:
                    self._style = {}

    def _migrate_panels_to_elements(self) -> None:
        if not self._spec:
            return
        panels = self._spec.get("panels", [])
        if not panels:
            return
        elements = self._spec.get("elements", [])
        element_ids = {e["id"] for e in elements}
        for p in panels:
            if p["id"] not in element_ids:
                elements.append(
                    {
                        "id": p["id"],
                        "type": "plot",
                        "mode": "embed",
                        "ref": p.get("plot", f"{p['id']}.pltz"),
                        "position": normalize_position(p.get("position")),
                        "size": normalize_size(p.get("size")),
                        **({} if "label" not in p else {"label": p["label"]}),
                    }
                )
        self._spec["elements"] = elements

    def _check_conflicts(self, name, size_mm, bundle_type) -> None:
        conflicts = []
        if isinstance(size_mm, (list, tuple)):
            size_mm = {"width": size_mm[0], "height": size_mm[1]}
        if name and self._spec.get("title") and self._spec.get("title") != name:
            conflicts.append(f"title: {self._spec.get('title')} vs {name}")
        if size_mm:
            es = self._spec.get("size_mm", {})
            if es and (es.get("width"), es.get("height")) != (
                size_mm.get("width"),
                size_mm.get("height"),
            ):
                conflicts.append("size_mm mismatch")
        if self._spec.get("type") and self._spec.get("type") != bundle_type:
            conflicts.append(f"type: {self._spec.get('type')} vs {bundle_type}")
        if conflicts:
            raise ValueError(f"Conflict loading {self.path}: " + ", ".join(conflicts))

    def _create_default_spec(self, name, size_mm=None, bundle_type="figure"):
        return {
            "schema": self.SCHEMA,
            "type": bundle_type,
            "bundle_id": generate_bundle_id(),
            "constraints": TYPE_DEFAULTS.get(
                bundle_type, {"allow_children": True, "max_depth": 3}
            ),
            "title": name,
            "size_mm": size_mm or self.DEFAULT_SIZE_MM,
            "elements": [],
        }

    def _create_new(self, name, size_mm=None, bundle_type="figure") -> None:
        if isinstance(size_mm, (list, tuple)):
            size_mm = {"width": size_mm[0], "height": size_mm[1]}
        is_dir = self.path.suffix == ".d" and self.path.stem.endswith((".stx", ".figz"))
        if self.path.suffix == ".figz" or (
            self.path.suffix == ".d" and self.path.stem.endswith(".figz")
        ):
            warnings.warn(
                ".figz deprecated. Use .stx", DeprecationWarning, stacklevel=3
            )
        if self.path.suffix not in (".stx", ".figz") and not str(self.path).endswith(
            (".stx.d", ".figz.d")
        ):
            self.path = self.path.with_suffix(".stx")
        self._is_dir, self._is_stx = is_dir, _is_stx_path(self.path)
        self._spec, self._style, self._modified = (
            self._create_default_spec(name, size_mm, bundle_type),
            {},
            True,
        )
        if is_dir:
            self.path.mkdir(parents=True, exist_ok=True)
            json.dump(
                self._spec,
                open(self.path / "spec.json", "w", encoding="utf-8"),
                indent=2,
            )
            json.dump(
                self._style,
                open(self.path / "style.json", "w", encoding="utf-8"),
                indent=2,
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with ZipBundle(self.path, mode="w") as zb:
                zb.write_json("spec.json", self._spec)
                zb.write_json("style.json", self._style)

    @classmethod
    def load_or_create(
        cls, path, name=None, size_mm=None, bundle_type="figure"
    ) -> Figz:
        return cls(path, name=name, size_mm=size_mm, bundle_type=bundle_type)

    @classmethod
    def create(cls, path, name, size_mm=None, bundle_type="figure") -> Figz:
        p = Path(path)
        if p.suffix == ".figz" or (p.suffix == ".d" and p.stem.endswith(".figz")):
            warnings.warn(
                ".figz deprecated. Use .stx", DeprecationWarning, stacklevel=2
            )
        if p.suffix not in (".stx", ".figz") and not str(p).endswith(
            (".stx.d", ".figz.d")
        ):
            p = p.with_suffix(".stx")
        is_dir = p.suffix == ".d" and p.stem.endswith((".stx", ".figz"))
        spec = {
            "schema": cls.SCHEMA,
            "type": bundle_type,
            "bundle_id": generate_bundle_id(),
            "constraints": TYPE_DEFAULTS.get(
                bundle_type, {"allow_children": True, "max_depth": 3}
            ),
            "title": name,
            "size_mm": size_mm or cls.DEFAULT_SIZE_MM,
            "elements": [],
        }
        if is_dir:
            p.mkdir(parents=True, exist_ok=True)
            json.dump(spec, open(p / "spec.json", "w", encoding="utf-8"), indent=2)
            json.dump({}, open(p / "style.json", "w", encoding="utf-8"), indent=2)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            with ZipBundle(p, mode="w") as zb:
                zb.write_json("spec.json", spec)
                zb.write_json("style.json", {})
        return cls(p)

    # Properties
    @property
    def spec(self) -> Dict[str, Any]:
        return self._spec or {}

    @spec.setter
    def spec(self, value):
        self._spec, self._modified = value, True

    @property
    def style(self) -> Dict[str, Any]:
        return self._style or {}

    @style.setter
    def style(self, value):
        self._style, self._modified = value, True

    @property
    def bundle_id(self):
        return self.spec.get("bundle_id")

    @property
    def bundle_type(self):
        return self.spec.get("type", "figure")

    @property
    def elements(self) -> List[Dict[str, Any]]:
        return self.spec.get("elements", [])

    @property
    def size_mm(self):
        return self.spec.get("size_mm", self.DEFAULT_SIZE_MM)

    @property
    def constraints(self):
        return self.spec.get("constraints", TYPE_DEFAULTS.get(self.bundle_type, {}))

    @property
    def is_directory(self) -> bool:
        return self._is_dir

    # Element API
    def add_element(
        self, element_id, element_type, content=None, position=None, size=None, **kwargs
    ):
        if self._spec is None:
            self._spec = self._create_default_spec("Untitled")
        pos, sz = normalize_position(position), normalize_size(size) if size else None
        element = {"id": element_id, "type": element_type, "position": pos}
        if sz:
            element["size"] = sz
        element.update(kwargs)
        ref_path, content_bytes = None, None
        if element_type in ("plot", "figure", "stats", "image"):
            content_bytes = self._process_content(
                element_type, content, element_id, element
            )
            if content_bytes:
                ext = self._get_content_extension(element_type, element)
                ref_path = f"children/{element_id}{ext}"
                element["mode"], element["ref"] = "embed", ref_path
                if element_type in ("plot", "figure"):
                    validate_can_add_child(
                        content_bytes,
                        element_id,
                        self.bundle_id,
                        self.constraints,
                        self.bundle_type,
                        self._spec,
                    )
        else:
            # Handle inline elements (text, symbol, equation, comment, shape)
            from ._figz_modules import process_inline_element

            process_inline_element(element_type, content, element)
        self._spec["elements"] = [
            e for e in self._spec.get("elements", []) if e["id"] != element_id
        ]
        self._spec["elements"].append(element)
        self._write_element(ref_path, content_bytes)
        self._modified = False

    def _process_content(self, element_type, content, element_id, element):
        return process_content(
            element_type, content, element_id, element, figure_to_stx_bytes
        )

    def _get_content_extension(self, element_type, element):
        return get_content_extension(element_type, element)

    def _write_element(self, ref_path, content_bytes):
        if self._is_dir:
            json.dump(
                self._spec,
                open(self.path / "spec.json", "w", encoding="utf-8"),
                indent=2,
            )
            if ref_path and content_bytes:
                (self.path / "children").mkdir(exist_ok=True)
                open(self.path / ref_path, "wb").write(content_bytes)
        else:
            with ZipBundle(self.path, mode="a") as zb:
                zb.write_json("spec.json", self._spec)
                if ref_path and content_bytes:
                    zb.write_bytes(ref_path, content_bytes)

    def get_element(self, element_id):
        return next((e for e in self.elements if e["id"] == element_id), None)

    def get_element_content(self, element_id):
        elem = self.get_element(element_id)
        if not elem or "ref" not in elem:
            return None
        try:
            if self._is_dir:
                return open(self.path / elem["ref"], "rb").read()
            with ZipBundle(self.path, mode="r") as zb:
                return zb.read_bytes(elem["ref"])
        except FileNotFoundError:
            return None

    def remove_element(self, element_id):
        if self._spec:
            self._spec["elements"] = [
                e for e in self._spec.get("elements", []) if e["id"] != element_id
            ]
            self._modified = True

    def update_element_position(self, element_id, x_mm, y_mm):
        elem = self.get_element(element_id)
        if elem:
            elem["position"] = {"x_mm": x_mm, "y_mm": y_mm}
            self._modified = True

    def update_element_size(self, element_id, width_mm, height_mm):
        elem = self.get_element(element_id)
        if elem:
            elem["size"] = {"width_mm": width_mm, "height_mm": height_mm}
            self._modified = True

    def list_element_ids(self, element_type=None):
        if element_type:
            return [e["id"] for e in self.elements if e.get("type") == element_type]
        return [e["id"] for e in self.elements]

    # Save/Load
    def save(self, path=None, verbose=True) -> Path:
        from scitex.logging import getLogger
        from scitex.path._getsize import getsize
        from scitex.str._readable_bytes import readable_bytes

        logger = getLogger(__name__)
        orig_path, orig_is_dir = self.path, self._is_dir
        if path:
            self.path = Path(path)
            self._is_dir = self.path.suffix == ".d"
        if self._is_dir:
            save_to_directory(
                self.path,
                self._spec,
                self._style,
                self.elements,
                self.render_preview_format,
                self._extract_geometry,
                orig_path,
                orig_is_dir,
            )
        else:
            save_to_zip(
                self.path,
                self._spec,
                self._style,
                self.elements,
                self.render_preview_format,
                orig_path,
                orig_is_dir,
            )
        self._modified = False
        if verbose:
            try:
                logger.success(
                    f"Saved to: {self.path} ({readable_bytes(getsize(self.path))})"
                )
            except:
                logger.success(f"Saved to: {self.path}")
        return self.path

    def pack(self, output_path=None) -> Figz:
        if not self._is_dir:
            raise ValueError("Bundle is already ZIP")
        output_path = (
            Path(output_path) if output_path else self.path.parent / self.path.stem
        )
        self.save()
        from scitex.io.bundle import pack as bundle_pack

        bundle_pack(self.path, output_path)
        return Figz(output_path)

    def unpack(self, output_path=None) -> Figz:
        if self._is_dir:
            raise ValueError("Bundle is already directory")
        output_path = (
            Path(output_path)
            if output_path
            else self.path.parent / f"{self.path.name}.d"
        )
        self.save()
        import zipfile

        output_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.path, "r") as zf:
            zf.extractall(output_path)
        return Figz(output_path)

    def auto_crop(self, margin_mm=5.0):
        from .layout import auto_crop_layout, content_bounds

        if not self.elements:
            return {
                "original_size": self.size_mm.copy(),
                "new_size": self.size_mm.copy(),
                "offset": {"x_mm": 0, "y_mm": 0},
                "bounds": None,
            }
        orig = self.size_mm.copy()
        bounds = content_bounds(self.elements)
        shifted, new_sz = auto_crop_layout(self.elements, margin_mm)
        offset = (
            {"x_mm": bounds["x_mm"] - margin_mm, "y_mm": bounds["y_mm"] - margin_mm}
            if bounds
            else {"x_mm": 0, "y_mm": 0}
        )
        self._spec["elements"] = shifted
        self._spec["size_mm"] = {
            "width": new_sz["width_mm"],
            "height": new_sz["height_mm"],
        }
        self._modified = True
        return {
            "original_size": orig,
            "new_size": {"width": new_sz["width_mm"], "height": new_sz["height_mm"]},
            "offset": offset,
            "bounds": bounds,
        }

    def get_content_bounds(self):
        from .layout import content_bounds

        return content_bounds(self.elements)

    def _extract_geometry(self, actual_size_px=None):
        return extract_geometry(
            self.elements, self.size_mm, actual_size_px=actual_size_px
        )

    def render_preview(self, dpi=150):
        return self.render_preview_format("png", dpi)

    def render_preview_format(self, fmt="png", dpi=150):
        return render_preview_internal(
            self.elements,
            self.size_mm,
            self.get_element_content,
            fmt,
            dpi,
            bundle_path=self.path if self._is_dir else None,
        )

    def __repr__(self):
        return f"Figz({self.path.name!r}, type={self.bundle_type!r}, elements={self.list_element_ids()})"


# EOF
