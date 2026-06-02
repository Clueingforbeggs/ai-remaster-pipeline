from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .config import IMAGE_EXTS, ROOT, VIDEO_EXTS
from .manifests import read_manifest
from .paths import resolve, resolve_video_source, safe_stem
from .runtime_settings import load_settings

PROJECT_SCHEMA_VERSION = 2
PROJECT_JSON_NAME = "project.json"


def bind_context(context: dict) -> None:
    globals().update(context)


def source_signature(source_text: str) -> tuple[str, int, int] | None:
    if not source_text:
        return None
    source = resolve_video_source(source_text)
    if not source.exists() or source.suffix.lower() not in VIDEO_EXTS:
        return None
    stat = source.stat()
    return str(source), stat.st_size, stat.st_mtime_ns

def source_analysis_key(signature: tuple[str, int, int]) -> str:
    return "\0".join(str(part) for part in signature)

def project_payload(settings: dict[str, dict[str, str]]) -> dict:
    stored_settings = {
        stage: {key: value for key, value in values.items() if key != "openai_api_key"}
        for stage, values in settings.items()
    }
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "app": "AI Remaster Pipeline",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "settings": stored_settings,
    }

def write_project_file(path: Path, settings: dict[str, dict[str, str]]) -> None:
    payload = project_payload(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(PROJECT_JSON_NAME, json.dumps(payload, indent=2) + "\n")
        for asset in project_asset_paths(settings):
            archive.write(asset, asset.relative_to(ROOT).as_posix())

def read_project_file(path: Path) -> dict[str, dict[str, str]]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            try:
                data = json.loads(archive.read(PROJECT_JSON_NAME).decode("utf-8-sig"))
            except KeyError as exc:
                raise RuntimeError("Project bundle does not contain project.json.") from exc
            extract_project_assets(archive)
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Project file is not valid JSON: {exc}") from exc
    return load_project_payload(data)

def load_project_payload(data: dict) -> dict[str, dict[str, str]]:
    try:
        version = int(data.get("schema_version", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Project file has an invalid schema_version.") from exc
    if version < 1:
        raise RuntimeError("Project file does not include a supported schema_version.")
    if version > PROJECT_SCHEMA_VERSION:
        raise RuntimeError(f"Project schema {version} is newer than this ARP build supports.")
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise RuntimeError("Project file does not contain settings.")
    loaded = load_settings()
    for stage, values in settings.items():
        if stage in loaded and isinstance(values, dict):
            loaded[stage].update({str(key): str(value) for key, value in values.items()})
    if not loaded.get("global", {}).get("source"):
        loaded.setdefault("global", {})["expand_outpaint"] = "true"
    return loaded

def project_asset_paths(settings: dict[str, dict[str, str]]) -> list[Path]:
    candidates: list[str] = []
    for key in ("shots", "references", "colour", "recomp"):
        manifest = settings.get(key, {}).get("manifest", "")
        if manifest:
            candidates.append(manifest)
    assets: list[Path] = []
    seen: set[Path] = set()
    for text in candidates:
        manifest = resolve(text)
        if not project_asset_is_bundleable(manifest) or manifest in seen:
            continue
        seen.add(manifest)
        assets.append(manifest)
        for row in read_manifest(manifest):
            for field in ("source_reference", "color_reference"):
                image = resolve(row.get(field, ""))
                if project_asset_is_bundleable(image) and image not in seen:
                    seen.add(image)
                    assets.append(image)
    return assets

def project_asset_is_bundleable(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    if path.suffix.lower() not in IMAGE_EXTS | {".csv", ".txt"}:
        return False
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return False
    return True

def extract_project_assets(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        if name == PROJECT_JSON_NAME or name.startswith("/") or ".." in Path(name).parts:
            continue
        if Path(name).suffix.lower() not in IMAGE_EXTS | {".csv", ".txt"}:
            continue
        target = ROOT / name
        try:
            target.resolve().relative_to(ROOT.resolve())
        except ValueError:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target.open("wb") as dest:
            dest.write(source.read())

def project_default_path(settings: dict[str, dict[str, str]]) -> Path:
    source = resolve_video_source(settings.get("global", {}).get("source", ""))
    stem = safe_stem(source.name if source.name else "arp_project")
    return ROOT / "projects" / f"{stem}.arpp"

def project_save_suggestion(settings: dict[str, dict[str, str]], project_path: Path | None = None) -> Path:
    if project_path:
        return project_path
    default_path = project_default_path(settings)
    last_dir = last_browse_dir(settings)
    return (last_dir / default_path.name) if last_dir else default_path

def last_browse_dir(settings: dict[str, dict[str, str]] | None = None) -> Path | None:
    values = settings or (APP.settings if "APP" in globals() else {})
    text = values.get("global", {}).get("last_browse_dir", "") if isinstance(values, dict) else ""
    if not text:
        return None
    path = resolve(str(text))
    return path if path.exists() and path.is_dir() else None
