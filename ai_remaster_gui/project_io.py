from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, VIDEO_EXTS
from .paths import resolve, resolve_video_source, safe_stem
from .runtime_settings import load_settings

PROJECT_SCHEMA_VERSION = 1


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
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "app": "AI Remaster Pipeline",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
    }

def read_project_file(path: Path) -> dict[str, dict[str, str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Project file is not valid JSON: {exc}") from exc
    version = int(data.get("schema_version", 0) or 0)
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
