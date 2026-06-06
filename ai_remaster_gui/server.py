from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

from .config import (
    ASPECT_PREVIEW_DIR,
    CONFIG_FILE,
    FILE_PREVIEW_DIR,
    IMAGE_EXTS,
    MEDIA_CLIP_DIR,
    OUTPAINT_PROMPT,
    PREVIEW_DIR,
    REFERENCE_PROMPT,
    REFERENCE_PROMPT_SUFFIX,
    ROOT,
    SCRIPTS,
    SETTINGS_FILE,
    STATIC_DIR,
    TEXT_EXTS,
    VIDEO_EXTS,
    current_config,
)
from .manifests import (
    manifest_source_video,
    read_manifest,
    read_manifest_details,
    read_outpaint_chunk_rows,
    update_manifest_row,
    write_manifest_details,
    write_outpaint_chunk_rows,
)
from .models import COLORIZE_STAGE_KEYS, STAGES, Stage, output_stage
from .paths import aspect_slug, even_int, newest, parse_aspect, rel, resolve, resolve_video_source, safe_stem
from .project_io import (
    last_browse_dir,
    project_default_path,
    project_payload,
    project_save_suggestion,
    read_project_file,
    source_analysis_key,
    source_signature,
    write_project_file,
    bind_context as bind_project_context,
)
from .process_utils import (
    count_lines_matching,
    first_int_after,
    format_duration,
    outpaint_chunk_progress,
    outpaint_eta_label,
    terminate_process_tree,
)
from .references import (
    color_reference_for_source,
    color_reference_outputs,
    colorized_output_for_manifest,
    colorized_outputs_for_manifest,
    delete_color_reference,
    extract_reference_frame,
    file_mtime,
    format_timecode,
    install_custom_color_reference,
    manifest_fps,
    merge_manifest_shots,
    parse_time_seconds,
    preview_reference_frame,
    recomposition_output_for,
    recent_color_references,
    reference_name_for_time,
    openai_reference_regeneration_command,
    reference_regeneration_command,
    accept_reference_edit,
    regenerate_reference_image,
    reference_edit_preview_command,
    revert_reference_edit,
    sam_reference_mask,
    selected_seconds_from_reference,
    split_manifest_shot,
    shot_rows,
    shot_views,
    update_shot_boundary,
    update_shot_fade,
    bind_context as bind_references_context,
)
from .file_dialogs import (
    applescript_quote,
    browse_initial_path,
    browse_path,
    browse_path_kdialog,
    browse_path_linux,
    browse_path_macos,
    browse_path_windows,
    browse_path_zenity,
    parse_duration,
    remember_browse_dir,
    bind_context as bind_file_dialogs_context,
)
from .lifecycle import (
    create_server,
    ensure_comfy_available_for_stage,
    install_shutdown_handlers,
    start_comfy_if_needed,
    stop_started_comfy,
    bind_context as bind_lifecycle_context,
)
from .outpaint_guides import (
    _build_guide_frames_view,
    _get_guide_manifest,
    _composite_guide_in_place,
    _guide_source_seconds,
    _parse_guide_frames,
    _save_guide_frames,
    accept_guide_edit,
    add_guide_frame,
    chunk_frame_preview,
    clear_guide_frame_image,
    guide_edit_preview_command,
    guide_frame_generation_command,
    remove_guide_frame,
    outpaint_end_guide_generation_command,
    outpaint_guide_generation_command,
    revert_guide_edit,
    sam_guide_mask,
    save_guide_frame,
    save_qwen_input_copy,
    upload_guide_frame_image,
    bind_context as bind_outpaint_guides_context,
)
from .cache import cache_state, delete_cache_category, delete_cache_file, human_size, bind_context as bind_cache_context
from .runtime_settings import APP_VERSION, default_qwen_workflow, load_settings, qwen_masked_workflow_for, qwen_workflow_for
from .system_status import system_status
from .media import (
    aspect_preview,
    aspect_preview_at,
    aspect_preview_at_for_settings,
    aspect_preview_cached,
    aspect_preview_for_settings,
    auto_crop_for_settings,
    current_crop_values,
    detect_letterbox_crop,
    draw_source_frame_border,
    extract_video_frame,
    ensure_source_section_clip,
    extract_video_frame_at,
    ffmpeg_aspect_preview,
    ffprobe_basic_info,
    ffprobe_info,
    ffprobe_info_from_data,
    file_preview,
    export_media_file,
    file_preview_cached,
    generate_video_previews,
    human_bitrate,
    local_tool,
    parse_rate,
    patterned_canvas,
    pipeline_source_text,
    preview_pipeline_source_text,
    safe_preview_name,
    section_float,
    section_relative_seconds,
    source_info,
    source_section_is_active,
    source_section_output_for,
    source_section_state,
    source_info_cached,
    source_monochrome,
    source_monochrome_cached,
    source_previews,
    source_previews_cached,
    source_previews_for_analysis,
    video_dimensions,
    video_metrics,
    media_clip_path,
    bind_context as bind_media_context,
)

MODEL_SIZE_MULTIPLE = 32


def combine_outpaint_prompt(prompt: str, suffix: str) -> str:
    base = (prompt or "").strip()
    extra = (suffix or "").strip()
    if not base:
        return extra
    if not extra:
        return base
    separator = " " if base.endswith((".", "!", "?", ":")) else ". "
    return f"{base}{separator}{extra}"




DEFAULT_ANCHOR_PROMPT = (
    "Fill the black outpaint margins with a natural continuation of this black-and-white film frame. "
    "Preserve the centre/original frame area, composition, lighting, paper, clothing, and background. "
    "If hands or fingers extend into the new margins, make them anatomically natural with five fingers and normal joints. "
    "Do not colorize. Do not add text, captions, logos, or unrelated new objects."
)








class PipelineApp:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.project_path: Path | None = None
        self.log: list[str] = []
        self.process: subprocess.Popen[str] | None = None
        self.running_stage = ""
        self.running_stage_key = ""
        self.running_reference_manifest = ""
        self.running_reference_index: int | None = None
        self.run_started_at = 0.0
        self.lock = threading.Lock()
        self.source_analysis_lock = threading.Lock()
        self.source_analysis_status: dict[str, dict[str, str | int | bool]] = {}
        self.source_analysis_results: dict[str, dict] = {}
        self.source_analysis_threads: set[str] = set()

    def normalize_loaded_source_state(self) -> None:
        source_text = self.settings.get("global", {}).get("source", "")
        if source_text:
            source = resolve_video_source(source_text)
            if source.exists() and str(source) != source_text:
                self.settings.setdefault("global", {})["source"] = str(source)
                self.log.append(f"Resolved source material path to: {source}")
            self.clear_derived_stage_inputs()
            self.hydrate_stage_inputs("global")

    def colorize_enabled(self) -> bool:
        return self.settings.get("global", {}).get("colorize", "true") == "true"

    def outpaint_enabled(self) -> bool:
        return self.settings.get("global", {}).get("expand_outpaint", "true") == "true"

    def upscale_enabled(self) -> bool:
        return self.settings.get("global", {}).get("upscale", "false") == "true"

    def active_stages(self) -> tuple[Stage, ...]:
        by_key = {stage.key: stage for stage in STAGES}
        stages: list[Stage] = []
        if self.outpaint_enabled():
            stages.append(by_key["outpaint"])
        if self.colorize_enabled():
            stages.extend(by_key[key] for key in ("shots", "references", "colour"))
        if self.outpaint_enabled() or self.colorize_enabled():
            stages.append(by_key["recomp"])
        if self.upscale_enabled():
            stages.append(by_key["upscale"])
        return tuple(stages)

    def save(self) -> None:
        SETTINGS_FILE.write_text(json.dumps(self.settings, indent=2) + "\n", encoding="utf-8")

    def files_for(self, stage: Stage) -> list[dict[str, str | int]]:
        exts = VIDEO_EXTS | IMAGE_EXTS | TEXT_EXTS
        scoped_prefixes = self.stage_file_prefixes(stage.key)
        out = []
        for folder_text in stage.folders:
            folder = ROOT / folder_text
            if not folder.exists():
                continue
            for path in folder.rglob("*"):
                if path.is_file() and path.suffix.lower() in exts and self.stage_file_matches(stage.key, path, scoped_prefixes):
                    try:
                        stat = path.stat()
                        preview = file_preview(path)
                    except FileNotFoundError:
                        continue
                    except OSError as exc:
                        self.log.append(f"Skipped unreadable file while refreshing {stage.title}: {rel(path)} ({exc})")
                        continue
                    out.append({"path": rel(path), "size": stat.st_size, "mtime": int(stat.st_mtime), "preview": preview})
        return sorted(out, key=lambda item: str(item["path"]).lower())

    def stage_file_prefixes(self, stage_key: str) -> tuple[str, ...]:
        source = self.settings.get("global", {}).get("source", "")
        if stage_key == "outpaint" and source:
            stem = safe_stem(resolve(source).name)
            values = self.settings.get("outpaint", {})
            return (stem,)
        return ()

    def stage_file_matches(self, stage_key: str, path: Path, prefixes: tuple[str, ...]) -> bool:
        if stage_key != "outpaint" or not prefixes:
            return True
        name = path.stem
        return any(name == prefix or name.startswith(prefix + "_") for prefix in prefixes)

    def progress(self) -> list[dict[str, str]]:
        rows = []
        for stage in self.active_stages():
            expected = [resolve(path) for path in self.expected_outputs(stage.key) if path]
            existing = [path for path in expected if path.exists()]
            ready = bool(expected) and len(existing) == len(expected)
            latest = max(existing, key=lambda path: path.stat().st_mtime_ns) if existing else None
            rows.append({"stage": stage.title, "status": "Ready" if ready else "Waiting", "latest": rel(latest) if latest else ""})
        return rows

    def phase_progress(self) -> dict:
        current = self.estimate_running_progress()
        stages = []
        completed = 0.0
        active_label = ""
        active = self.active_stages()
        for stage in active:
            title = stage.title
            latest = next((item["latest"] for item in self.progress() if item["stage"] == title), "")
            if self.running_stage_key == stage.key and current:
                percent = current["percent"]
                label = current["label"]
                active_label = label
            elif latest:
                percent = 100
                label = "Ready"
            else:
                percent = 0
                label = "Waiting"
            completed += percent / 100
            stages.append({"key": stage.key, "stage": title, "percent": percent, "label": label})
        global_percent = int(round((completed / max(1, len(active))) * 100))
        global_label = f"{global_percent}% complete"
        if active_label:
            global_label = f"{global_label} - {active_label}"
        return {"global": {"percent": global_percent, "label": global_label}, "stages": stages}

    def estimate_running_progress(self) -> dict:
        if not self.running_stage_key:
            return {}
        elapsed = max(0.0, time.time() - self.run_started_at)
        log_text = "\n".join(self.log[-300:])
        lower = log_text.lower()
        percent = min(90, 5 + int(elapsed / 60 * 20))
        label = "Running"
        if self.running_stage_key == "outpaint":
            chunk = outpaint_chunk_progress(log_text)
            milestones = [
                ("checking model", 8, "Checking models"),
                ("downloading model", 10, "Downloading models"),
                ("downloaded:", 11, "Model download complete"),
                ("preparing expanded outpaint canvas", 12, "Preparing expanded canvas"),
                ("reuse prepared outpaint input", 20, "Prepared input reused"),
                ("wrote prepared outpaint input", 20, "Prepared input written"),
                ("prepared expanded canvas for comfyui", 25, "Prepared for ComfyUI"),
                ("splitting prepared canvas", 28, "Splitting into chunks"),
                ("waiting for comfyui", 30, "Waiting for ComfyUI"),
                ("queued comfyui prompt", 40, "Queued in ComfyUI"),
                ("outpaint chunk", 42, "Outpainting chunks"),
                ("wrote raw comfy render", 82, "Raw outpaint render written"),
                ("reuse raw comfy render", 82, "Raw outpaint render reused"),
                ("wrote outpainted video", 100, "Outpainted video written"),
            ]
            for token, value, text in milestones:
                if token in lower and value >= percent:
                    percent, label = value, text
            if chunk["total"] and percent < 100:
                rendering = chunk["current"] > chunk["done"] and ("queued comfyui prompt" in lower or "sending prompt nodes" in lower)
                active_fraction = 0.5 if rendering else 0.2 if chunk["current"] > chunk["done"] else 0.0
                chunk_fraction = min(1.0, (chunk["done"] + active_fraction) / chunk["total"])
                percent = max(percent, min(95, 35 + int(chunk_fraction * 55)))
                eta = outpaint_eta_label(elapsed, chunk["done"], chunk["current"], chunk["total"])
                if chunk["done"] >= chunk["total"]:
                    label = f"Chunks complete, finalizing{eta}"
                elif rendering:
                    label = f"Chunk {chunk['current']}/{chunk['total']} rendering in ComfyUI ({chunk['done']} done){eta}"
                else:
                    label = f"Chunk {chunk['current']}/{chunk['total']} ({chunk['done']} done){eta}"
        elif self.running_stage_key == "shots":
            if "detected " in lower:
                percent, label = max(percent, 75), "Shots detected"
            if "wrote manifest" in lower:
                percent, label = 100, "Manifest written"
        elif self.running_stage_key == "references":
            if self.running_reference_index is not None:
                label = f"Regenerating shot {self.running_reference_index + 1}"
                if "queued comfyui prompt" in lower or "waiting for comfyui" in lower:
                    percent = max(percent, 35)
                    label = f"Shot {self.running_reference_index + 1}: waiting for ComfyUI"
                if "copied comfyui output" in lower or "wrote " in lower:
                    percent = max(percent, 85)
                    label = f"Shot {self.running_reference_index + 1}: saving reference"
                if "regenerated colour reference" in lower or "finished with exit code 0" in lower:
                    percent = 100
                    label = f"Shot {self.running_reference_index + 1}: complete"
            else:
                reference_log = log_text
                for marker in ("scripts\\qwen_colorize_references.py", "scripts/qwen_colorize_references.py"):
                    index = reference_log.rfind(marker)
                    if index >= 0:
                        reference_log = reference_log[index:]
                        break
                rows = first_int_after(reference_log, "Rows:")
                done = min(rows, count_lines_matching(reference_log, ("Reuse ", "Wrote "))) if rows else 0
                if rows:
                    percent = min(99, int((done / rows) * 100))
                    label = f"{done}/{rows} references"
        elif self.running_stage_key == "colour":
            colour_log = log_text
            for marker in ("scripts\\colorize_video.py", "scripts/colorize_video.py"):
                index = colour_log.rfind(marker)
                if index >= 0:
                    colour_log = colour_log[index:]
                    break
            colour_lower = colour_log.lower()
            segment = 0
            total = 0
            for line in colour_log.splitlines():
                marker = "Colorize segment "
                if marker not in line:
                    continue
                tail = line.split(marker, 1)[1].split(" ", 1)[0]
                if "/" not in tail:
                    continue
                try:
                    left, right = tail.split("/", 1)
                    segment = int(left.strip())
                    total = max(total, int(right.strip()))
                except ValueError:
                    pass
            if segment and total:
                percent = max(percent, min(99, int(((segment - 1) / total) * 100)))
                label = f"Colorizing segment {segment}/{total}"
            if "reuse colorized video" in colour_lower:
                percent, label = max(percent, 75), "Existing colorized video reused"
            if "finished with exit code 0" in colour_lower:
                percent, label = 100, "Colorization complete"
        elif self.running_stage_key == "recomp":
            if "wrote composite" in lower:
                percent, label = 100, "Composite written"
            else:
                label = "Compositing"
        elif self.running_stage_key == "upscale":
            label = "Upscaling"
            milestones = [
                ("queueing flashvsr", 20, "Queueing FlashVSR in ComfyUI"),
                ("queued comfyui prompt", 40, "Queued in ComfyUI"),
                ("sending prompt nodes", 42, "Sending FlashVSR prompt"),
                ("wrote upscaled video", 100, "Upscaled video written"),
                ("reuse upscaled video", 100, "Upscaled video ready"),
            ]
            for token, value, text in milestones:
                if token in lower and value >= percent:
                    percent, label = value, text
        return {"key": self.running_stage_key, "stage": self.running_stage, "percent": percent, "label": label}

    def state(self, view: str = "") -> dict:
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            source_text = self.settings.get("global", {}).get("source", "")
            source_media = self.source_media_state(source_text)
            section = source_section_state(self.settings)
            aspect_preview = aspect_preview_for_settings(self.settings) if view == "outpaint" else source_media["aspect_preview"]
            outpaint_chunks = outpaint_chunks_state(self.settings) if view == "outpaint" else {"manifest": "", "rows": []}
            shots = shot_views(self.settings) if view in {"shots", "references", "colour"} else {"manifest": "", "rows": []}
            cache = cache_state() if view == "cache" else {}
            return {
                "root": str(ROOT),
                "version": APP_VERSION,
                "stages": [stage.__dict__ | {"files": self.files_for(stage)} for stage in (*self.active_stages(), output_stage())],
                "settings": self.settings,
                "progress": self.progress(),
                "phase_progress": self.phase_progress(),
                "expected_outputs": {stage.key: self.expected_outputs(stage.key) for stage in (*self.active_stages(), output_stage())},
                "existing_outputs": {stage.key: self.existing_outputs(stage.key) for stage in (*self.active_stages(), output_stage())},
                "upscale_preview": self.upscale_preview_state(),
                "output_selection": self.output_selection_state(),
                "source_previews": source_media["previews"],
                "source_info": source_media["info"],
                "source_section": section,
                "project_path": str(self.project_path) if self.project_path else "",
                "source_monochrome": source_media["monochrome"],
                "source_analysis": source_media["analysis"],
                "aspect_preview": aspect_preview,
                "outpaint_chunks": outpaint_chunks,
                "shot_views": shots,
                "cache": cache,
                "system_status": system_status(),
                "running": running,
                "running_stage": self.running_stage,
                "running_reference": {
                    "manifest": self.running_reference_manifest,
                    "index": self.running_reference_index,
                } if self.running_reference_index is not None else None,
                "log": "\n".join(self.log[-800:]),
                "log_count": len(self.log),
            }

    def update_settings(self, stage: str, values: dict[str, str]) -> None:
        previous_source = self.settings.get("global", {}).get("source", "") if stage == "global" else ""
        if stage == "global" and "source" in values:
            source = resolve_video_source(str(values.get("source", "")))
            if source.exists() and str(source) != str(values.get("source", "")):
                values = dict(values)
                values["source"] = str(source)
                self.log.append(f"Resolved source material path to: {source}")
            if str(values.get("source", "")) != previous_source:
                values = dict(values)
                values["section_start"] = "0"
                values["section_end"] = source_duration_text(source) if source.exists() else ""
        self.settings.setdefault(stage, {}).update({key: str(value) for key, value in values.items()})
        if stage == "global" and {"source", "section_start", "section_end"} & set(values):
            self.log.append(f"Loading source material: {values.get('source')}")
            self.clear_derived_stage_inputs()
            self.hydrate_stage_inputs("global")
        elif stage == "global" and ({"colorize", "expand_outpaint", "upscale"} & set(values)):
            self.hydrate_stage_inputs("global")
        elif stage == "colour" and "method" in values:
            if values.get("method") in {"deepexemplar", "colormnet"}:
                self.settings.setdefault("recomp", {})["colorization_method"] = str(values["method"])
            self.hydrate_stage_inputs("colour")
        elif stage == "recomp" and "colorization_method" in values:
            preferred = colorized_output_for_manifest(self.settings.get("colour", {}).get("manifest", ""), str(values.get("colorization_method", "deepexemplar")))
            if preferred:
                self.settings.setdefault("recomp", {})["colorized_video"] = preferred
        if stage == "shots" and "outpainted_video" in values:
            manifest = manifest_for_outpainted(values.get("outpainted_video", ""))
            self.settings.setdefault("references", {}).setdefault("manifest", manifest)
            self.settings.setdefault("colour", {}).setdefault("manifest", manifest)
        self.save()

    def source_media_state(self, source_text: str) -> dict:
        signature = source_signature(source_text)
        if signature is None:
            if source_text:
                source = resolve(source_text)
                self.log.append(f"Source analysis skipped; file was not found or is not a supported video: {source}")
            return {"previews": [], "info": {}, "monochrome": True, "aspect_preview": "", "analysis": {}}

        key = source_analysis_key(signature)
        basic_info = {"file": rel(Path(signature[0])), "size": human_size(signature[1])}
        with self.source_analysis_lock:
            result = self.source_analysis_results.get(key)
            status = self.source_analysis_status.get(key)
            if result:
                analysis = dict(status or {})
                analysis.update({"ready": True, "message": "Source analysis complete"})
                return {
                    "previews": result.get("previews", []),
                    "info": result.get("info", basic_info),
                    "monochrome": result.get("monochrome", True),
                    "aspect_preview": result.get("aspect_preview", ""),
                    "analysis": analysis,
                }
            if key not in self.source_analysis_threads:
                self.source_analysis_threads.add(key)
                self.source_analysis_status[key] = {
                    "ready": False,
                    "percent": 1,
                    "message": "Queued source analysis",
                    "source": signature[0],
                }
                thread = threading.Thread(target=self.analyze_source_media, args=(signature, key), daemon=True)
                thread.start()
                status = self.source_analysis_status[key]

        return {
            "previews": [],
            "info": basic_info,
            "monochrome": True,
            "aspect_preview": "",
            "analysis": dict(status or {"ready": False, "percent": 1, "message": "Queued source analysis"}),
        }

    def set_source_analysis_status(self, key: str, percent: int, message: str, ready: bool = False) -> None:
        with self.source_analysis_lock:
            current = dict(self.source_analysis_status.get(key, {}))
            current.update({"ready": ready, "percent": percent, "message": message})
            self.source_analysis_status[key] = current

    def analyze_source_media(self, signature: tuple[str, int, int], key: str) -> None:
        source = Path(signature[0])
        info: dict[str, str] = {"file": rel(source), "size": human_size(signature[1])}
        previews: list[str] = []
        monochrome = True
        aspect = ""
        try:
            self.set_source_analysis_status(key, 8, "Reading basic source metadata")
            info.update(ffprobe_basic_info(source))

            self.set_source_analysis_status(key, 35, "Generating a few source preview frames")
            previews = list(source_previews_for_analysis(signature, info, lambda percent, message: self.set_source_analysis_status(key, percent, message)))

            self.set_source_analysis_status(key, 76, "Checking whether the source is black and white")
            monochrome = source_monochrome_cached(*signature)
            self.apply_detected_source_tone(signature[0], monochrome)

            with self.source_analysis_lock:
                self.source_analysis_results[key] = {
                    "previews": previews,
                    "info": info,
                    "monochrome": monochrome,
                    "aspect_preview": aspect,
                }
                self.source_analysis_status[key] = {
                    "ready": True,
                    "percent": 100,
                    "message": "Source analysis complete",
                    "source": signature[0],
                }
                self.source_analysis_threads.discard(key)
            self.log.append(f"Source analysis complete: {source}")
        except Exception as exc:
            with self.source_analysis_lock:
                self.source_analysis_status[key] = {
                    "ready": False,
                    "percent": 100,
                    "message": f"Source analysis failed: {exc}",
                    "source": signature[0],
                }
                self.source_analysis_threads.discard(key)
            self.log.append(f"Source analysis failed for {source}: {exc}")

    def apply_detected_source_tone(self, source_path: str, monochrome: bool) -> None:
        with self.lock:
            selected = self.settings.get("global", {}).get("source", "")
            current = resolve_video_source(selected) if selected else None
            if not current or str(current) != source_path:
                return
            self.settings.setdefault("global", {})["colorize"] = "true" if monochrome else "false"
            self.hydrate_stage_inputs("global")
            self.save()

    def clear_overview(self) -> None:
        self.settings.setdefault("global", {}).update({"source": "", "expand_outpaint": "true", "colorize": "true", "upscale": "false", "section_start": "0", "section_end": ""})
        self.clear_derived_stage_inputs()
        self.log.append("Cleared source material from the Overview.")
        self.save()

    def save_project(self, save_as: bool = False) -> dict[str, str]:
        if save_as or not self.project_path:
            suggested = project_save_suggestion(self.settings, self.project_path)
            selected = browse_path("project_save", str(suggested))
            if not selected:
                return {"path": ""}
            path = resolve(selected)
            if path.suffix.lower() != ".arpp":
                path = path.with_suffix(".arpp")
            self.project_path = path
        else:
            path = self.project_path
        write_project_file(path, self.settings)
        self.log.append(f"Saved ARP project: {path}")
        return {"path": str(path)}

    def load_project(self) -> dict[str, str]:
        selected = browse_path("project_open", "")
        if not selected:
            return {"path": ""}
        path = resolve(selected)
        loaded = read_project_file(path)
        self.settings = loaded
        self.project_path = path
        self.hydrate_stage_inputs("")
        self.save()
        self.log.append(f"Loaded ARP project: {path}")
        return {"path": str(path)}

    def clear_derived_stage_inputs(self) -> None:
        for stage_key, keys in {
            "outpaint": ("source", "output", "outpainted_video", "manifest", "colorized_video"),
            "shots": ("outpainted_video", "manifest", "colorized_video"),
            "references": ("manifest", "outpainted_video", "colorized_video"),
            "colour": ("manifest", "outpainted_video", "colorized_video"),
            "recomp": ("outpainted_video", "source", "colorized_video", "output", "manifest"),
            "upscale": ("input_video", "output"),
            "output": ("output", "outpainted_video", "manifest", "colorized_video"),
        }.items():
            stage_settings = self.settings.setdefault(stage_key, {})
            for key in keys:
                stage_settings[key] = ""

    def hydrate_stage_inputs(self, completed_stage: str = "") -> None:
        if not self.outpaint_enabled():
            outpainted_text = pipeline_source_text(self.settings)
            if outpainted_text:
                self.settings.setdefault("shots", {})["outpainted_video"] = outpainted_text
                self.settings.setdefault("recomp", {})["outpainted_video"] = outpainted_text
                manifest = manifest_for_outpainted(outpainted_text)
                self.settings.setdefault("references", {})["manifest"] = manifest
                self.settings.setdefault("colour", {})["manifest"] = manifest
                self.settings.setdefault("recomp", {})["manifest"] = manifest
                self.log.append(f"Updated Shot Detection input: {outpainted_text}")
            outpainted = None
        else:
            expected_outpainted = resolve(self.expected_outputs("outpaint")[0]) if self.expected_outputs("outpaint") else None
            if completed_stage == "global" and not (expected_outpainted and expected_outpainted.exists()):
                outpainted = None
            else:
                outpainted = expected_outpainted if expected_outpainted and expected_outpainted.exists() else None
            if outpainted:
                outpainted_text = rel(outpainted)
                self.settings.setdefault("shots", {})["outpainted_video"] = outpainted_text
                self.settings.setdefault("recomp", {})["outpainted_video"] = outpainted_text
                manifest = manifest_for_outpainted(outpainted_text)
                self.settings.setdefault("references", {})["manifest"] = manifest
                self.settings.setdefault("colour", {})["manifest"] = manifest
                self.settings.setdefault("recomp", {})["manifest"] = manifest
                self.log.append(f"Updated Shot Detection input: {outpainted_text}")
        if self.outpaint_enabled() and not outpainted:
            for stage_key in ("shots", "recomp"):
                self.settings.setdefault(stage_key, {})["outpainted_video"] = ""
            for stage_key in ("references", "colour", "recomp"):
                self.settings.setdefault(stage_key, {})["manifest"] = ""
        expected_manifest = resolve(self.expected_outputs("shots")[0]) if self.expected_outputs("shots") else None
        manifest = expected_manifest if expected_manifest and expected_manifest.exists() else None
        if manifest:
            manifest_text = rel(manifest)
            self.settings.setdefault("references", {})["manifest"] = manifest_text
            self.settings.setdefault("colour", {})["manifest"] = manifest_text
            self.settings.setdefault("recomp", {})["manifest"] = manifest_text
            self.log.append(f"Updated manifest inputs: {manifest_text}")
        expected_colorized_text = self.expected_outputs("colour")[0] if self.expected_outputs("colour") else ""
        preferred_colorized_text = colorized_output_for_manifest(
            self.settings.get("colour", {}).get("manifest", ""),
            self.settings.get("recomp", {}).get("colorization_method", self.settings.get("colour", {}).get("method", "deepexemplar")),
        )
        if preferred_colorized_text and resolve(preferred_colorized_text).exists():
            expected_colorized_text = preferred_colorized_text
        expected_colorized = resolve(expected_colorized_text) if expected_colorized_text else None
        colorized = expected_colorized if expected_colorized and expected_colorized.exists() else None
        if self.colorize_enabled() and colorized:
            self.settings.setdefault("recomp", {})["colorized_video"] = rel(colorized)
        elif not self.colorize_enabled():
            self.settings.setdefault("recomp", {})["colorized_video"] = ""
        source = self.settings.get("global", {}).get("source")
        if source:
            self.settings.setdefault("recomp", {})["source"] = pipeline_source_text(self.settings)
        output = recomposition_output_for(self.settings.get("recomp", {}).get("outpainted_video", ""))
        if output:
            self.settings.setdefault("recomp", {})["output"] = output
        upscale_input = self.upscale_input_for()
        if upscale_input:
            self.settings.setdefault("upscale", {})["input_video"] = upscale_input
            upscale_output = upscale_output_for(upscale_input, self.settings.get("upscale", {}))
            if upscale_output:
                self.settings.setdefault("upscale", {})["output"] = upscale_output
        selected = self.output_selection_state().get("path", "")
        if selected:
            self.settings.setdefault("output", {})["output"] = selected
        self.save()

    def expected_outputs(self, stage_key: str) -> list[str]:
        values = self.settings.get(stage_key, {})
        if stage_key == "outpaint":
            if not self.outpaint_enabled():
                return []
            source = pipeline_source_text(self.settings)
            return [outpaint_output_for(source, values.get("target_aspect", "16:9"), values.get("target_height", "720"))] if source else []
        if stage_key == "shots":
            manifest = manifest_for_outpainted(values.get("outpainted_video", ""))
            return [manifest] if manifest else []
        if stage_key == "references":
            return color_reference_outputs(values.get("manifest", ""))
        if stage_key == "colour":
            return colorized_outputs_for_manifest(values.get("manifest", ""), values.get("method", "deepexemplar"))
        if stage_key == "recomp":
            output = values.get("output") or recomposition_output_for(values.get("outpainted_video", ""))
            return [output] if output else []
        if stage_key == "upscale":
            source = self.upscale_input_for() or values.get("input_video")
            output = upscale_output_for(source, values) or values.get("output")
            return [output] if output else []
        if stage_key == "output":
            output = self.output_selection_state().get("path", "")
            return [output] if output else []
        return []

    def existing_outputs(self, stage_key: str) -> list[str]:
        return [path for path in self.expected_outputs(stage_key) if path and resolve(path).exists()]

    def command_for(self, stage_key: str) -> list[str]:
        values = self.settings[stage_key]
        config = current_config()
        py = sys.executable
        cmd = [py, "-u"]
        add = cmd.extend
        if stage_key == "outpaint":
            cmd.append(str(SCRIPTS / "outpaint_video.py"))
            add(["--source", pipeline_source_text(self.settings)])
            add(["--target-aspect", values.get("target_aspect", "16:9")])
            add(["--target-height", str(resolved_outpaint_height(pipeline_source_text(self.settings), values.get("target_height", "720")))])
            add(["--chunk-seconds", values.get("chunk_seconds", "20")])
            add(["--overlap-frames", values.get("overlap_frames", "8")])
            add(["--prompt", values.get("prompt") or OUTPAINT_PROMPT])
            if values.get("negative_prompt"):
                add(["--negative-prompt", values.get("negative_prompt", "")])
            if values.get("guide_strength"):
                add(["--guide-strength", values.get("guide_strength", "0.7")])
            if values.get("guide_end_strength"):
                add(["--guide-end-strength", values.get("guide_end_strength", "1.0")])
            if values.get("outpaint_all_black_regions", "false") == "true":
                add(["--outpaint-all-black-regions"])
            manifest = outpaint_chunk_manifest_for(pipeline_source_text(self.settings), values)
            if manifest:
                add(["--chunk-manifest", manifest])
            for key in ("crop_left", "crop_right", "crop_top", "crop_bottom"):
                add([f"--{key.replace('_', '-')}", values.get(key, "0")])
            add(["--comfy-dir", config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))])
            add(["--comfy-url", config.get("comfy_url", "http://127.0.0.1:8188")])
        elif stage_key == "shots":
            cmd.append(str(SCRIPTS / "generate_references.py"))
            add(["--source-video", values.get("outpainted_video", "")])
            manifest = manifest_for_outpainted(values.get("outpainted_video", ""))
            if manifest:
                add(["--output-manifest", manifest])
            for key in ("sample_seconds", "shot_threshold", "min_shot_seconds"):
                add([f"--{key.replace('_', '-')}", values.get(key, "")])
            if values.get("limit"):
                add(["--limit", values["limit"]])
            # Extract reference frames at model-safe dimensions (matching the prepared canvas)
            # so thumbnails are consistent with what LTX works with.
            outpaint_values = self.settings.get("outpaint", {})
            source_text = pipeline_source_text(self.settings)
            aspect = outpaint_values.get("target_aspect", "16:9")
            height_text = outpaint_values.get("target_height", "720")
            ref_w, ref_h = outpaint_work_size_for_source(source_text, aspect, height_text)
            add(["--frame-width", str(ref_w), "--frame-height", str(ref_h)])
        elif stage_key == "references":
            if values.get("method", "qwen") == "openai":
                cmd.append(str(SCRIPTS / "openai_generate_reference.py"))
                add(["--manifest", values.get("manifest", ""), "--api-key", values.get("openai_api_key", "")])
                add(["--model", values.get("openai_image_model", "gpt-image-2") or "gpt-image-2"])
                add(["--prompt", values.get("prompt", ""), "--prompt-suffix", values.get("prompt_suffix", "")])
                add(["--size", values.get("openai_image_size", "auto"), "--quality", values.get("openai_image_quality", "auto")])
                if values.get("openai_send_references", "false") == "true":
                    add(["--reference-count", "3"])
            else:
                cmd.append(str(SCRIPTS / "qwen_colorize_references.py"))
                workflow = qwen_workflow_for(values, config)
                comfy_url = values.get("comfy_url") or config.get("comfy_url", "http://127.0.0.1:8188")
                comfy_dir = config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))
                comfy_output = values.get("comfy_output_root") or str(Path(comfy_dir) / "output")
                add(["--manifest", values.get("manifest", ""), "--workflow", workflow, "--comfy-url", comfy_url])
                add(["--comfy-dir", comfy_dir, "--comfy-output-root", comfy_output])
                add(["--model-backend", values.get("model_backend", "gguf"), "--gguf-model", values.get("gguf_model", "qwen-image-edit-2511-Q4_K_M.gguf")])
                add(["--prompt", values.get("prompt", ""), "--prompt-suffix", values.get("prompt_suffix", ""), "--load-image-node-id", values.get("load_image_node_id", "auto"), "--save-node-id", values.get("save_node_id", "auto")])
                if values.get("prompt_node_id"):
                    add(["--prompt-node-id", values["prompt_node_id"]])
            if values.get("limit"):
                add(["--limit", values["limit"]])
        elif stage_key == "colour":
            cmd.append(str(SCRIPTS / "colorize_video.py"))
            add(["--manifest", values.get("manifest", "")])
            method = values.get("method", "deepexemplar")
            add(["--method", method])
            output = colorized_output_for_manifest(values.get("manifest", ""), method)
            if output:
                add(["--output", output])
            add(["--comfy-dir", config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))])
            add(["--comfy-url", config.get("comfy_url", "http://127.0.0.1:8188")])
            add(["--comfy-output-root", str(Path(config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))) / "output")])
            add(["--crf", values.get("crf", "18")])
            add(["--colormnet-memory-mode", values.get("colormnet_memory_mode", "balanced")])
            add(["--colormnet-feature-encoder", values.get("colormnet_feature_encoder", "resnet50")])
            if values.get("colormnet_text_guidance"):
                add(["--colormnet-text-guidance", values["colormnet_text_guidance"]])
            for key in ("frame_propagate", "use_half_resolution", "use_torch_compile", "use_sage_attention"):
                flag = "--" + key.replace("_", "-")
                add([flag if values.get(key, "false") == "true" else "--no-" + flag[2:]])
        elif stage_key == "recomp":
            cmd.append(str(SCRIPTS / "final_composite.py"))
            output = values.get("output") or recomposition_output_for(values.get("outpainted_video", ""))
            add(["--outpainted", values.get("outpainted_video", ""), "--source", values.get("source", ""), "--output", output])
            if self.colorize_enabled() and values.get("colorized_video"):
                add(["--colorized", values["colorized_video"]])
            add(["--feather-pixels", values.get("feather_pixels", "80"), "--saturation", values.get("saturation", "0.82"), "--temperature", values.get("temperature", "-0.015"), "--color-opacity", values.get("color_opacity", "1.0"), "--encoder", values.get("encoder", "h264")])
            outpaint_values = self.settings.get("outpaint", {})
            for key in ("crop_left", "crop_right", "crop_top", "crop_bottom"):
                add([f"--{key.replace('_', '-')}", outpaint_values.get(key, "0")])
            # Pass delivery dimensions so final_composite upscales from the model-safe LTX output
            # (e.g. 704p) back to the user's intended resolution (e.g. 720p).
            source_text = pipeline_source_text(self.settings)
            aspect = outpaint_values.get("target_aspect", "16:9")
            height_text = outpaint_values.get("target_height", "720")
            delivery_w, delivery_h = outpaint_size_for_source(source_text, aspect, height_text)
            add(["--output-width", str(delivery_w), "--output-height", str(delivery_h)])
        elif stage_key == "upscale":
            source = self.upscale_input_for() or values.get("input_video")
            output = upscale_output_for(source, values) or values.get("output")
            if not source or not output:
                return []
            cmd = self.upscale_command(values, source, output)
        if values.get("force") == "true":
            cmd.append("--force")
        if values.get("dry_run") == "true":
            cmd.append("--dry-run")
        return [part for part in cmd if part != ""]

    def run_stage(self, stage_key: str) -> tuple[bool, str]:
        if stage_key == "outpaint" and not self.outpaint_enabled():
            return False, "Expand using Outpainting is disabled on the Overview tab."
        if stage_key in COLORIZE_STAGE_KEYS and not self.colorize_enabled():
            return False, "Colorize is disabled on the Overview tab."
        if stage_key == "recomp" and not (self.outpaint_enabled() or self.colorize_enabled()):
            return False, "Recomposition is only needed when Outpainting or Colorize is enabled."
        if stage_key == "upscale" and not self.upscale_enabled():
            return False, "Upscale is disabled on the Overview tab."
        stage = next(item for item in STAGES if item.key == stage_key)
        if stage_key == "upscale":
            self.hydrate_stage_inputs("upscale")
            if not self.upscale_input_for():
                return False, "Upscaling input is not available yet. Choose source material, or run Recomposition first when earlier phases are enabled."
        values = self.settings[stage_key]
        missing = [key for key in stage.required if not values.get(key)]
        if stage_key == "outpaint" and not self.settings.get("global", {}).get("source"):
            missing = ["source material on the Global tab"]
        if missing:
            return False, "Missing settings: " + ", ".join(missing)
        if stage_key == "references" and values.get("method", "qwen") == "openai" and not values.get("openai_api_key", "").strip():
            return False, "Add your OpenAI API key in Settings before running OpenAI Reference Generation."
        try:
            self.ensure_pipeline_source()
        except Exception as exc:
            return False, f"Could not prepare selected source section: {exc}"
        if stage_key == "upscale":
            source_text = self.upscale_input_for()
            if not source_text or not resolve(source_text).exists():
                return False, "Upscaling input is not available yet. Run Recomposition first when earlier phases are enabled."
        needs_comfy = (
            stage_key in {"outpaint", "colour"}
            or (stage_key == "references" and values.get("method", "qwen") != "openai")
            or stage_key == "upscale"
        )
        if needs_comfy:
            ok, message = ensure_comfy_available_for_stage(stage.title)
            if not ok:
                return False, message
        with self.lock:
            if self.process and self.process.poll() is None:
                return False, "A command is already running."
            self.running_stage = stage.title
            self.running_stage_key = stage.key
            self.run_started_at = time.time()
            cmd = self.command_for(stage_key)
            self.log.append("> " + redact_command_for_log(cmd))
            kwargs: dict = {"cwd": ROOT, "text": True, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            self.process = subprocess.Popen(cmd, **kwargs)
            threading.Thread(target=self._collect_output, args=(stage_key,), daemon=True).start()
        return True, "Started " + stage.title

    def run_outpaint_chunk(self, index: int) -> tuple[bool, str]:
        if not self.settings.get("global", {}).get("source"):
            return False, "Choose source material on the Overview tab first."
        try:
            self.ensure_pipeline_source()
        except Exception as exc:
            return False, f"Could not prepare selected source section: {exc}"
        ok, message = ensure_comfy_available_for_stage("Outpainting")
        if not ok:
            return False, message
        with self.lock:
            if self.process and self.process.poll() is None:
                return False, "A command is already running."
            self.running_stage = f"Outpainting chunk {index + 1}"
            self.running_stage_key = "outpaint"
            self.run_started_at = time.time()
            cmd = self.command_for("outpaint")
            cmd.extend(["--only-chunk", str(index), "--force"])
            self.log.append("> " + " ".join(cmd))
            kwargs: dict = {"cwd": ROOT, "text": True, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            self.process = subprocess.Popen(cmd, **kwargs)
            threading.Thread(target=self._collect_output, args=("outpaint",), daemon=True).start()
        return True, f"Started outpaint chunk {index + 1}"

    def run_reference_regeneration(self, manifest_text: str, index: int, provider: str = "qwen") -> tuple[bool, str]:
        provider = "openai" if (provider == "openai" or self.settings.get("references", {}).get("method") == "openai") else "qwen"
        if provider == "qwen":
            ok, message = ensure_comfy_available_for_stage("Reference Generation")
            if not ok:
                return False, message
        try:
            if provider == "openai":
                cmd, output = openai_reference_regeneration_command(manifest_text, index)
            else:
                cmd, output = reference_regeneration_command(manifest_text, index)
        except Exception as exc:
            return False, str(exc)
        with self.lock:
            if self.process and self.process.poll() is None:
                return False, "A command is already running."
            self.running_stage = "Reference Generation"
            self.running_stage_key = "references"
            self.running_reference_manifest = manifest_text
            self.running_reference_index = index
            self.run_started_at = time.time()
            label = "OpenAI" if provider == "openai" else "Qwen"
            self.log.append(f"Regenerating colour reference with {label} for shot {index + 1}: {output}")
            self.log.append("> " + redact_command_for_log(cmd))
            kwargs: dict = {"cwd": ROOT, "text": True, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            try:
                self.process = subprocess.Popen(cmd, **kwargs)
            except Exception as exc:
                self.running_stage = ""
                self.running_stage_key = ""
                self.running_reference_manifest = ""
                self.running_reference_index = None
                self.run_started_at = 0.0
                self.log.append(f"Could not start reference regeneration: {exc}")
                return False, f"Could not start reference regeneration: {exc}"
            threading.Thread(target=self._collect_output, args=("references",), daemon=True).start()
        return True, f"Started {provider} reference regeneration for shot {index + 1}."

    def run_reference_edit_preview(self, manifest_text: str, index: int, instruction: str, mask_data: str = "", sampled_color: str = "") -> tuple[bool, str, str]:
        ok, message = ensure_comfy_available_for_stage("Reference Editing")
        if not ok:
            return False, message, ""
        try:
            cmd, output = reference_edit_preview_command(manifest_text, index, instruction, mask_data, sampled_color)
        except Exception as exc:
            return False, str(exc), ""
        with self.lock:
            if self.process and self.process.poll() is None:
                return False, "A command is already running.", output
            self.running_stage = "Reference Editing"
            self.running_stage_key = "references"
            self.running_reference_manifest = manifest_text
            self.running_reference_index = index
            self.run_started_at = time.time()
            mode = "masked" if mask_data else "unmasked"
            self.log.append(f"Generating {mode} reference edit preview for shot {index + 1}: {output}")
            self.log.append("> " + redact_command_for_log(cmd))
            kwargs: dict = {"cwd": ROOT, "text": True, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            try:
                self.process = subprocess.Popen(cmd, **kwargs)
            except Exception as exc:
                self.running_stage = ""
                self.running_stage_key = ""
                self.running_reference_manifest = ""
                self.running_reference_index = None
                self.run_started_at = 0.0
                self.log.append(f"Could not start reference edit preview: {exc}")
                return False, f"Could not start reference edit preview: {exc}", output
            threading.Thread(target=self._collect_output, args=("references",), daemon=True).start()
        return True, f"Started reference edit preview for shot {index + 1}.", output

    def run_outpaint_end_guide_generation(self, index: int, prompt: str) -> tuple[bool, str]:
        ok, message = ensure_comfy_available_for_stage("End Guide Frame Generation")
        if not ok:
            return False, message
        try:
            cmd, output_rel, prepared_canvas, source_seconds = outpaint_end_guide_generation_command(index, prompt)
        except Exception as exc:
            return False, str(exc)
        output = resolve(output_rel)
        with self.lock:
            if self.process and self.process.poll() is None:
                return False, "A command is already running."
            self.running_stage = f"Generating end guide frame for chunk {index + 1}"
            self.running_stage_key = "outpaint"
            self.run_started_at = time.time()
            self.log.append(f"Generating Qwen end guide frame for chunk {index + 1}: {output_rel}")
            self.log.append("> " + " ".join(cmd))
            kwargs: dict = {"cwd": ROOT, "text": True, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            try:
                self.process = subprocess.Popen(cmd, **kwargs)
            except Exception as exc:
                self.running_stage = ""
                self.running_stage_key = ""
                self.run_started_at = 0.0
                self.log.append(f"Could not start end guide frame generation: {exc}")
                return False, f"Could not start end guide frame generation: {exc}"
            threading.Thread(
                target=self._collect_output_guide,
                args=(output, prepared_canvas),
                kwargs={"source_seconds": source_seconds},
                daemon=True,
            ).start()
        return True, f"Started Qwen end guide frame generation for chunk {index + 1}."

    def run_outpaint_guide_generation(self, index: int, prompt: str) -> tuple[bool, str]:
        ok, message = ensure_comfy_available_for_stage("Guide Frame Generation")
        if not ok:
            return False, message
        try:
            cmd, output_rel, prepared_canvas, source_seconds = outpaint_guide_generation_command(index, prompt)
        except Exception as exc:
            return False, str(exc)
        output = resolve(output_rel)
        with self.lock:
            if self.process and self.process.poll() is None:
                return False, "A command is already running."
            self.running_stage = f"Generating guide frame for chunk {index + 1}"
            self.running_stage_key = "outpaint"
            self.run_started_at = time.time()
            self.log.append(f"Generating Qwen guide frame for chunk {index + 1}: {output_rel}")
            self.log.append("> " + " ".join(cmd))
            kwargs: dict = {"cwd": ROOT, "text": True, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            try:
                self.process = subprocess.Popen(cmd, **kwargs)
            except Exception as exc:
                self.running_stage = ""
                self.running_stage_key = ""
                self.run_started_at = 0.0
                self.log.append(f"Could not start guide frame generation: {exc}")
                return False, f"Could not start guide frame generation: {exc}"
            threading.Thread(
                target=self._collect_output_guide,
                args=(output, prepared_canvas),
                kwargs={"source_seconds": source_seconds},
                daemon=True,
            ).start()
        return True, f"Started Qwen guide frame generation for chunk {index + 1}."

    def run_guide_frame_generation(self, chunk_index: int, guide_index: int, frame_idx: int, prompt: str) -> tuple[bool, str]:
        ok, message = ensure_comfy_available_for_stage("Guide Frame Generation")
        if not ok:
            return False, message
        try:
            cmd, output_rel, prepared_canvas, source_seconds = guide_frame_generation_command(chunk_index, guide_index, frame_idx, prompt)
        except Exception as exc:
            return False, str(exc)
        output = resolve(output_rel)
        with self.lock:
            if self.process and self.process.poll() is None:
                return False, "A command is already running."
            self.running_stage = f"Generating guide frame {guide_index} for chunk {chunk_index + 1}"
            self.running_stage_key = "outpaint"
            self.run_started_at = time.time()
            self.log.append(f"Generating Qwen guide (chunk {chunk_index + 1}, guide {guide_index}, frame_idx={frame_idx}): {output_rel}")
            self.log.append("> " + " ".join(cmd))
            kwargs: dict = {"cwd": ROOT, "text": True, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            try:
                self.process = subprocess.Popen(cmd, **kwargs)
            except Exception as exc:
                self.running_stage = ""
                self.running_stage_key = ""
                self.run_started_at = 0.0
                self.log.append(f"Could not start guide frame generation: {exc}")
                return False, f"Could not start guide frame generation: {exc}"
            threading.Thread(
                target=self._collect_output_guide,
                args=(output, prepared_canvas),
                kwargs={"source_seconds": source_seconds},
                daemon=True,
            ).start()
        return True, f"Started Qwen guide frame generation for chunk {chunk_index + 1}, guide {guide_index}."

    def upscale_input_for(self) -> str:
        if self.outpaint_enabled() or self.colorize_enabled():
            recomposed = self.settings.get("recomp", {}).get("output") or recomposition_output_for(self.settings.get("recomp", {}).get("outpainted_video", ""))
            return recomposed
        return pipeline_source_text(self.settings)

    def upscale_command(self, values: dict[str, str], source: str, output: str) -> list[str]:
        config = current_config()
        cmd = [sys.executable, "-u", str(SCRIPTS / "upscale_video.py")]
        add = cmd.extend
        add(["--input", source])
        add(["--target-width", str(values.get("target_width", "3840")), "--target-height", str(values.get("target_height", "2160"))])
        add(["--output", output])
        add(["--comfy-dir", config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))])
        add(["--comfy-url", config.get("comfy_url", "http://127.0.0.1:8188")])
        add(["--comfy-output-root", str(Path(config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))) / "output")])
        add(["--flashvsr-model", values.get("flashvsr_model", "FlashVSR-v1.1")])
        add(["--flashvsr-mode", values.get("flashvsr_mode", "tiny")])
        add(["--flashvsr-scale", values.get("flashvsr_scale", "2")])
        add(["--flashvsr-seed", values.get("flashvsr_seed", "0")])
        add(["--flashvsr-tiled-vae" if values.get("flashvsr_tiled_vae", "true") == "true" else "--no-flashvsr-tiled-vae"])
        add(["--flashvsr-tiled-dit" if values.get("flashvsr_tiled_dit", "true") == "true" else "--no-flashvsr-tiled-dit"])
        if values.get("flashvsr_unload_dit", "false") == "true":
            add(["--flashvsr-unload-dit"])
        return [part for part in cmd if part != ""]

    def upscale_preview_state(self) -> dict[str, str]:
        values = self.settings.get("upscale", {})
        preview_source = values.get("preview_source", "")
        preview_output = values.get("preview_output", "")
        if preview_source and preview_output and resolve(preview_output).exists():
            return {"source": preview_source, "output": preview_output, "exists": "true"}
        source = self.upscale_input_for() or values.get("input_video")
        output = upscale_preview_output_for(source, values)
        exists = bool(output and resolve(output).exists())
        return {"source": source, "output": output, "exists": "true" if exists else "false"}

    def output_selection_state(self) -> dict[str, str]:
        upscale = self.settings.get("upscale", {})
        upscale_output = upscale_output_for(self.upscale_input_for() or upscale.get("input_video"), upscale) or upscale.get("output")
        recomposed = self.settings.get("recomp", {}).get("output") or recomposition_output_for(self.settings.get("recomp", {}).get("outpainted_video", ""))
        if upscale_output and resolve(upscale_output).exists():
            return {"path": upscale_output, "kind": "upscaled", "label": "Upscaled output"}
        if recomposed and resolve(recomposed).exists():
            return {"path": recomposed, "kind": "recomposed", "label": "Recomposed output"}
        if self.upscale_enabled() and upscale_output:
            return {"path": upscale_output, "kind": "upscaled_pending", "label": "Pending upscaled output"}
        if recomposed:
            return {"path": recomposed, "kind": "recomposed_pending", "label": "Pending recomposed output"}
        return {"path": "", "kind": "", "label": ""}

    def run_upscale_preview(self) -> tuple[bool, str]:
        values = self.settings.get("upscale", {})
        source_text = self.upscale_input_for() or values.get("input_video")
        if not source_text:
            return False, "Choose a source and enable Upscale before generating a preview."
        seconds = max(0.1, float(values.get("preview_seconds", "6") or 6))
        try:
            source, start, end, key = self.upscale_preview_clip_source(seconds)
            if not source.exists():
                return False, f"Upscale preview input does not exist yet: {rel(source)}"
            clip = media_clip_path(source, start, end, key)
            output = upscale_preview_output_for(rel(clip), values)
            cmd = self.upscale_command(values, rel(clip), output)
        except Exception as exc:
            return False, f"Could not prepare upscale preview: {exc}"
        with self.lock:
            if self.process and self.process.poll() is None:
                return False, "A command is already running."
            self.running_stage = "Upscale Preview"
            self.running_stage_key = "upscale"
            self.run_started_at = time.time()
            values["preview_source"] = rel(clip)
            values["preview_output"] = output
            self.save()
            self.log.append(f"Generating upscale preview: {output}")
            self.log.append("> " + redact_command_for_log(cmd))
            kwargs: dict = {"cwd": ROOT, "text": True, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            try:
                self.process = subprocess.Popen(cmd, **kwargs)
            except Exception as exc:
                self.running_stage = ""
                self.running_stage_key = ""
                self.run_started_at = 0.0
                self.log.append(f"Could not start upscale preview: {exc}")
                return False, f"Could not start upscale preview: {exc}"
            threading.Thread(target=self._collect_output, args=("upscale_preview",), daemon=True).start()
        return True, "Started upscale preview."

    def upscale_preview_clip_source(self, preview_seconds: float) -> tuple[Path, float, float, str]:
        if not (self.outpaint_enabled() or self.colorize_enabled()):
            global_settings = self.settings.get("global", {})
            source_text = global_settings.get("source", "")
            source = resolve_video_source(source_text)
            if source_section_is_active(self.settings):
                start = section_float(global_settings.get("section_start", "0"), 0.0)
                section_end = section_float(global_settings.get("section_end", ""), start + preview_seconds)
                end = min(section_end, start + preview_seconds) if section_end > start else start + preview_seconds
                return source, start, end, f"upscale_preview_src_{start:.3f}_{end:.3f}"
            return source, 0.0, preview_seconds, f"upscale_preview_src_{preview_seconds:.3f}"

        source_text = self.upscale_input_for() or self.settings.get("upscale", {}).get("input_video", "")
        return resolve(source_text), 0.0, preview_seconds, f"upscale_preview_{preview_seconds:.3f}"

    def run_guide_edit_preview(self, chunk_index: int, guide_index: int, instruction: str, mask_data: str = "", sampled_color: str = "") -> tuple[bool, str, str]:
        ok, message = ensure_comfy_available_for_stage("Guide Frame Editing")
        if not ok:
            return False, message, ""
        try:
            cmd, output = guide_edit_preview_command(chunk_index, guide_index, instruction, mask_data, sampled_color)
        except Exception as exc:
            return False, str(exc), ""
        with self.lock:
            if self.process and self.process.poll() is None:
                return False, "A command is already running.", output
            self.running_stage = f"Editing guide frame {guide_index + 1} for chunk {chunk_index + 1}"
            self.running_stage_key = "outpaint"
            self.run_started_at = time.time()
            mode = "masked" if mask_data else "unmasked"
            self.log.append(f"Generating {mode} guide edit preview (chunk {chunk_index + 1}, guide {guide_index + 1}): {output}")
            self.log.append("> " + redact_command_for_log(cmd))
            kwargs: dict = {"cwd": ROOT, "text": True, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            try:
                self.process = subprocess.Popen(cmd, **kwargs)
            except Exception as exc:
                self.running_stage = ""
                self.running_stage_key = ""
                self.run_started_at = 0.0
                self.log.append(f"Could not start guide edit preview: {exc}")
                return False, f"Could not start guide edit preview: {exc}", output
            threading.Thread(target=self._collect_output, args=("outpaint",), daemon=True).start()
        return True, f"Started guide edit preview for chunk {chunk_index + 1}, guide {guide_index + 1}.", output

    def _collect_output_guide(self, output: Path, prepared_canvas: Path, source_seconds: float | None = None) -> None:
        """Like _collect_output but composites the guide in-place after a successful Qwen run."""
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            with self.lock:
                self.log.append(line.rstrip())
        code = self.process.wait()
        with self.lock:
            self.log.append(f"Process finished with exit code {code}.")
            self.running_stage = ""
            self.running_stage_key = ""
            self.running_reference_manifest = ""
            self.running_reference_index = None
            self.run_started_at = 0.0
            if code == 0 and output.exists() and prepared_canvas.exists():
                try:
                    _composite_guide_in_place(output, prepared_canvas, source_seconds=source_seconds)
                    self.log.append("Guide frame composited with source fill and corner inpaint.")
                except Exception as exc:
                    self.log.append(f"Warning: guide compositing failed (guide used as-is): {exc}")
            if code == 0:
                self.hydrate_stage_inputs("outpaint")

    def run_all(self) -> tuple[bool, str]:
        threading.Thread(target=self._run_all_worker, daemon=True).start()
        return True, "Started whole remaster queue."

    def _run_all_worker(self) -> None:
        for stage in self.active_stages():
            ok, message = self.run_stage(stage.key)
            if not ok:
                with self.lock:
                    self.log.append(f"Skipping {stage.title}: {message}")
                continue
            while self.process and self.process.poll() is None:
                time.sleep(0.5)
            if self.process and self.process.returncode:
                break

    def ensure_pipeline_source(self) -> None:
        ensure_source_section_clip(self.settings)

    def _collect_output(self, stage_key: str) -> None:
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            with self.lock:
                self.log.append(line.rstrip())
        code = self.process.wait()
        with self.lock:
            self.log.append(f"Process finished with exit code {code}.")
            self.running_stage = ""
            self.running_stage_key = ""
            self.running_reference_manifest = ""
            self.running_reference_index = None
            self.run_started_at = 0.0
            if code == 0 and stage_key != "upscale_preview":
                self.hydrate_stage_inputs(stage_key)
            elif code == 0 and stage_key == "upscale_preview":
                self.log.append("Upscale preview ready.")

    def stop(self) -> None:
        with self.lock:
            if self.process and self.process.poll() is None:
                terminate_process_tree(self.process)
                self.log.append("Stop requested.")


APP = PipelineApp()












































def manifest_for_outpainted(outpainted_text: str) -> str:
    if not outpainted_text:
        return ""
    source = resolve(outpainted_text)
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in source.name.replace(" ", "_"))
    return rel(ROOT / "manifests" / "references" / f"colorize_manifest_{Path(stem).stem}_shots_auto.csv")


def outpaint_output_for(source_text: str, aspect: str, target_height_text: str = "720") -> str:
    if not source_text:
        return ""
    source = resolve_video_source(source_text)
    if str(target_height_text or "").strip().lower() in {"source", "source height", "original"}:
        width, height = outpaint_size_for_source(source_text, aspect, target_height_text)
    else:
        width, height = outpaint_work_size_for_source(source_text, aspect, target_height_text)
    values = APP.settings.get("outpaint", {}) if "APP" in globals() else {}
    crops = [int(float(values.get(key, "0") or 0)) for key in ("crop_left", "crop_right", "crop_top", "crop_bottom")]
    crop = "" if not any(crops) else f"_crop{crops[0]}-{crops[1]}-{crops[2]}-{crops[3]}"
    black_region = "_allblack" if values.get("outpaint_all_black_regions", "false") == "true" else ""
    return rel(ROOT / "intermediate" / "outpainted" / f"{safe_stem(source.name)}_{aspect_slug(aspect)}_{width}x{height}{crop}{black_region}_outpainted.mp4")


def upscale_target_size(values: dict[str, str]) -> tuple[int, int]:
    try:
        width = even_int(int(float(values.get("target_width", "3840") or 3840)))
    except ValueError:
        width = 3840
    try:
        height = even_int(int(float(values.get("target_height", "2160") or 2160)))
    except ValueError:
        height = 2160
    return max(2, width), max(2, height)


def upscale_output_for(source_text: str, values: dict[str, str]) -> str:
    if not source_text:
        return ""
    source = resolve(source_text)
    width, height = upscale_target_size(values)
    method = "flashvsr"
    return rel(ROOT / "output" / "upscaled" / f"{safe_stem(source.name)}_{method}_{width}x{height}.mp4")


def upscale_preview_output_for(source_text: str, values: dict[str, str]) -> str:
    if not source_text:
        return ""
    source = resolve(source_text)
    width, height = upscale_target_size(values)
    method = "flashvsr"
    seconds = safe_stem(str(values.get("preview_seconds", "6") or "6"))
    return rel(ROOT / "output" / "upscaled" / "previews" / f"{safe_stem(source.name)}_{method}_{width}x{height}_preview_{seconds}s.mp4")


def source_duration_text(source: Path) -> str:
    try:
        duration = float(video_metrics(source).get("duration") or 0)
    except Exception:
        return ""
    return f"{duration:.3f}" if duration > 0 else ""


def outpaint_size_for(aspect: str, target_height_text: str = "720") -> tuple[int, int]:
    try:
        height = int(float(target_height_text or "720"))
    except ValueError:
        height = 720
    return even_int(height * parse_aspect(aspect)), even_int(height)


def source_video_height(source_text: str) -> int:
    try:
        source = resolve_video_source(source_text)
        metrics = video_metrics(source)
        return even_int(int(metrics.get("height") or 720))
    except Exception:
        return 720


def resolved_outpaint_height(source_text: str, target_height_text: str = "720") -> int:
    if str(target_height_text or "").strip().lower() in {"source", "source height", "original"}:
        return source_video_height(source_text)
    try:
        return even_int(int(float(target_height_text or "720")))
    except ValueError:
        return 720


def outpaint_size_for_source(source_text: str, aspect: str, target_height_text: str = "720") -> tuple[int, int]:
    height = resolved_outpaint_height(source_text, target_height_text)
    return even_int(height * parse_aspect(aspect)), height


def model_safe_int(value: int, multiple: int = MODEL_SIZE_MULTIPLE) -> int:
    value = max(multiple, int(value))
    lower = max(multiple, (value // multiple) * multiple)
    upper = lower if lower == value else lower + multiple
    return lower if value - lower <= upper - value else upper


def outpaint_work_size_for_source(source_text: str, aspect: str, target_height_text: str = "720") -> tuple[int, int]:
    width, height = outpaint_size_for_source(source_text, aspect, target_height_text)
    return model_safe_int(width), model_safe_int(height)


def outpaint_crop_slug(values: dict[str, str]) -> str:
    crops = [int(float(values.get(key, "0") or 0)) for key in ("crop_left", "crop_right", "crop_top", "crop_bottom")]
    return "" if not any(crops) else f"_crop{crops[0]}-{crops[1]}-{crops[2]}-{crops[3]}"


def outpaint_black_region_slug(values: dict[str, str]) -> str:
    return "_allblack" if values.get("outpaint_all_black_regions", "false") == "true" else ""


def outpaint_chunk_dir_for(source_text: str, values: dict[str, str]) -> Path:
    source = resolve_video_source(source_text)
    aspect = values.get("target_aspect", "16:9")
    width, height = outpaint_work_size_for_source(source_text, aspect, values.get("target_height", "720"))
    return ROOT / ".cache" / "outpaint_chunks" / f"{safe_stem(source.name)}_{aspect_slug(aspect)}_{width}x{height}{outpaint_crop_slug(values)}{outpaint_black_region_slug(values)}"


def outpaint_chunk_manifest_for(source_text: str, values: dict[str, str]) -> str:
    if not source_text:
        return ""
    source = resolve_video_source(source_text)
    aspect = values.get("target_aspect", "16:9")
    width, height = outpaint_work_size_for_source(source_text, aspect, values.get("target_height", "720"))
    return rel(ROOT / "manifests" / "outpaint_chunks" / f"{safe_stem(source.name)}_{aspect_slug(aspect)}_{width}x{height}{outpaint_crop_slug(values)}{outpaint_black_region_slug(values)}_chunks.csv")


def outpaint_chunk_offset_slug(row: dict[str, str]) -> str:
    try:
        offset_x = int(float(row.get("offset_x", "0") or 0))
        offset_y = int(float(row.get("offset_y", "0") or 0))
    except ValueError:
        offset_x = offset_y = 0
    return "" if not (offset_x or offset_y) else f"_ox{offset_x:+d}_oy{offset_y:+d}"


def outpaint_prepared_for(source_text: str, values: dict[str, str]) -> Path:
    source = resolve_video_source(source_text)
    aspect = values.get("target_aspect", "16:9")
    height_text = values.get("target_height", "720")
    work_w, work_h = outpaint_work_size_for_source(source_text, aspect, height_text)
    del_w, del_h = outpaint_size_for_source(source_text, aspect, height_text)
    delivery_tag = f"_from{del_w}x{del_h}" if (del_w != work_w or del_h != work_h) else ""
    mode_tag = "_allblack" if values.get("outpaint_all_black_regions", "false") == "true" else "_lifted"
    return ROOT / "intermediate" / "outpaint_prepared" / f"{source.stem}_{work_w}x{work_h}{delivery_tag}{outpaint_crop_slug(values)}{mode_tag}.mp4"


def ensure_outpaint_prepared_canvas(source_text: str, values: dict[str, str]) -> Path:
    source = resolve_video_source(source_text)
    prepared = outpaint_prepared_for(source_text, values)
    if prepared.exists():
        return prepared

    cmd = [
        sys.executable,
        str(SCRIPTS / "prepare_outpaint_input.py"),
        "--source",
        str(source),
        "--target-aspect",
        values.get("target_aspect", "16:9"),
        "--black-lift",
        str(values.get("black_lift", "0.018") or "0.018"),
        "--gamma",
        str(values.get("gamma", "1.06") or "1.06"),
        "--output",
        str(prepared),
        "--crop-left",
        str(values.get("crop_left", "0") or "0"),
        "--crop-right",
        str(values.get("crop_right", "0") or "0"),
        "--crop-top",
        str(values.get("crop_top", "0") or "0"),
        "--crop-bottom",
        str(values.get("crop_bottom", "0") or "0"),
        "--target-width",
        str(outpaint_work_size_for_source(source_text, values.get("target_aspect", "16:9"), values.get("target_height", "720"))[0]),
        "--target-height",
        str(outpaint_work_size_for_source(source_text, values.get("target_aspect", "16:9"), values.get("target_height", "720"))[1]),
        "--delivery-width",
        str(outpaint_size_for_source(source_text, values.get("target_aspect", "16:9"), values.get("target_height", "720"))[0]),
        "--delivery-height",
        str(outpaint_size_for_source(source_text, values.get("target_aspect", "16:9"), values.get("target_height", "720"))[1]),
    ]
    if values.get("outpaint_all_black_regions", "false") == "true":
        cmd.append("--outpaint-all-black-regions")
    APP.log.append(f"Preparing expanded canvas for guide frame: {rel(prepared)}")
    APP.log.append("> " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, check=False, capture_output=True, text=True)
    for line in (result.stdout or "").splitlines():
        APP.log.append(line)
    for line in (result.stderr or "").splitlines():
        APP.log.append(line)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "Could not prepare expanded outpaint canvas.")
    if not prepared.exists():
        raise RuntimeError(f"Prepared expanded canvas was not created: {prepared}")
    return prepared


def outpaint_chunks_state(settings: dict) -> dict:
    try:
        ensure_source_section_clip(settings)
    except Exception as exc:
        return {"manifest": "", "rows": [], "error": f"Could not prepare selected source section: {exc}"}

    source_text = pipeline_source_text(settings)
    if not source_text:
        return {"manifest": "", "rows": []}
    source = resolve_video_source(source_text)
    if not source.exists():
        return {"manifest": "", "rows": [], "error": f"Source material is not a readable file: {source}"}
    values = settings.get("outpaint", {})
    metrics = video_metrics(source)
    fps = metrics.get("fps") or 24.0
    total_frames = int(metrics.get("frames") or 0)
    if total_frames <= 0:
        message = f"Outpaint chunk preview skipped; could not count frames in: {source}"
        APP.log.append(message)
        return {"manifest": "", "rows": [], "error": message}
    try:
        chunk_seconds = float(values.get("chunk_seconds", "20") or 20)
    except ValueError:
        chunk_seconds = 20.0
    try:
        overlap_frames = int(float(values.get("overlap_frames", "8") or 8))
    except ValueError:
        overlap_frames = 8
    chunk_dir = outpaint_chunk_dir_for(source_text, values)
    manifest = resolve(outpaint_chunk_manifest_for(source_text, values))
    existing = read_outpaint_chunk_rows(manifest)
    ranges = outpaint_chunk_ranges(total_frames, fps, chunk_seconds, overlap_frames, existing)
    global_prompt = values.get("prompt") or OUTPAINT_PROMPT
    global_negative = values.get("negative_prompt", "")
    rows = []
    for index, start_frame, end_frame in ranges:
        row = dict(existing.get(index, {}))
        row.setdefault("offset_x", "0")
        row.setdefault("offset_y", "0")
        offset_slug = outpaint_chunk_offset_slug(row)
        prepared = chunk_dir / f"prepared_{index:04d}_{start_frame:06d}_{end_frame:06d}{offset_slug}.mp4"
        raw = chunk_dir / f"raw_{index:04d}_{start_frame:06d}_{end_frame:06d}{offset_slug}.mp4"
        row.update({
            "chunk_index": str(index),
            "start_frame": str(start_frame),
            "end_frame": str(end_frame),
            "start_seconds": f"{start_frame / fps:.6f}",
            "end_seconds": f"{end_frame / fps:.6f}",
            "prepared_path": rel(prepared),
            "raw_path": rel(raw),
        })
        row.setdefault("custom_seconds", "")
        if not row.get("seed"):
            row["seed"] = str(42 + index)
        row.setdefault("prompt_suffix", "")
        row.setdefault("negative_suffix", "")
        row.setdefault("guide_image", "")
        row.setdefault("guide_strength", "0.7")
        row.setdefault("guide_end_image", "")
        row.setdefault("guide_end_strength", "1.0")
        row.setdefault("guide_frames", "")
        rows.append(row)
    write_outpaint_chunk_rows(manifest, rows)
    view_rows = []
    for row in rows:
        raw = resolve(row["raw_path"])
        prepared = resolve(row["prepared_path"])
        start_seconds = float(row["start_seconds"])
        end_seconds = float(row["end_seconds"])
        middle_seconds = (start_seconds + end_seconds) / 2
        length_frames = int(row["end_frame"]) - int(row["start_frame"])
        aspect = values.get("target_aspect", "16:9")
        guides = _build_guide_frames_view(row, source_text, aspect, start_seconds, end_seconds, fps, length_frames)
        view_rows.append(row | {
            "index": int(row["chunk_index"]),
            "start": float(row["start_seconds"]),
            "end": float(row["end_seconds"]),
            "fps": fps,
            "total_frames": total_frames,
            "length_frames": length_frames,
            "max_length_frames": max(1, total_frames - int(row["start_frame"])),
            "start_label": format_timecode(float(row["start_seconds"])),
            "end_label": format_timecode(float(row["end_seconds"])),
            "raw_exists": raw.exists(),
            "raw_mtime": int(raw.stat().st_mtime_ns) if raw.exists() else 0,
            "prepared_exists": prepared.exists(),
            "guides": guides,
            "source_start_preview": "",
            "source_middle_preview": "",
            "source_end_preview": "",
            "raw_start_preview": "",
            "raw_middle_preview": "",
            "raw_end_preview": "",
            "effective_prompt": combine_outpaint_prompt(global_prompt, row.get("prompt_suffix", "")),
            "effective_negative_prompt": combine_outpaint_prompt(global_negative, row.get("negative_suffix", "")),
        })
    return {"manifest": rel(manifest), "rows": view_rows}


def outpaint_chunk_preview(settings: dict, chunk_index: int, kind: str, position: str) -> str:
    chunks = outpaint_chunks_state(settings)
    row = next((r for r in chunks.get("rows", []) if int(r.get("index", -1)) == chunk_index), None)
    if row is None:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")

    position = position if position in {"start", "middle", "end"} else "middle"
    fps = max(1.0, float(row.get("fps", 24) or 24))
    start_seconds = float(row.get("start", 0.0) or 0.0)
    end_seconds = float(row.get("end", start_seconds) or start_seconds)
    duration = max(0.0, end_seconds - start_seconds)

    if position == "start":
        offset = 0.0
    elif position == "end":
        offset = max(0.0, duration - (1.0 / fps))
    else:
        offset = duration / 2

    if kind == "raw":
        raw = resolve(str(row.get("raw_path", "")))
        if not raw.exists():
            return ""
        return chunk_frame_preview(raw, offset, f"raw_{chunk_index}_{position}")

    source_text = pipeline_source_text(settings)
    if not source_text:
        return ""
    aspect = settings.get("outpaint", {}).get("target_aspect", "16:9")
    try:
        offset_x = int(float(row.get("offset_x", "0") or 0))
        offset_y = int(float(row.get("offset_y", "0") or 0))
    except ValueError:
        offset_x = offset_y = 0
    return aspect_preview_at(source_text, aspect, start_seconds + offset, offset_x, offset_y)












def outpaint_chunk_ranges(total_frames: int, fps: float, default_seconds: float, overlap_frames: int, existing: dict[int, dict[str, str]]) -> list[tuple[int, int, int]]:
    ranges = []
    start = 0
    index = 0
    while start < total_frames:
        seconds = default_seconds
        custom = existing.get(index, {}).get("custom_seconds", "")
        if custom:
            try:
                seconds = float(custom)
            except ValueError:
                seconds = default_seconds
        chunk_frames = total_frames if seconds <= 0 else max(1, int(round(seconds * fps)))
        end = min(total_frames, start + chunk_frames)
        ranges.append((index, start, end))
        if end >= total_frames:
            break
        overlap = max(0, min(overlap_frames, chunk_frames - 1))
        start += max(1, chunk_frames - overlap)
        index += 1
    return ranges


def _truthy_payload_value(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def redact_command_for_log(cmd: list[str]) -> str:
    redacted: list[str] = []
    hide_next = False
    for part in cmd:
        if hide_next:
            redacted.append("[redacted]")
            hide_next = False
            continue
        redacted.append(part)
        if part in {"--api-key", "--openai-api-key"}:
            hide_next = True
    return " ".join(redacted)


def update_outpaint_chunk(index: int, seed: str, prompt_suffix: str, custom_seconds: str = "", negative_suffix: str = "", guide_strength: str = "", guide_end_strength: str = "", custom_length=None, offset_x: str = "0", offset_y: str = "0") -> None:
    state = outpaint_chunks_state(APP.settings)
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    rows = read_outpaint_chunk_rows(resolve(str(manifest_text)))
    if index not in rows:
        raise IndexError(f"Outpaint chunk not found: {index + 1}")
    row = rows[index]
    row["seed"] = str(int(float(seed or row.get("seed") or 42 + index)))
    row["prompt_suffix"] = prompt_suffix
    row["negative_suffix"] = negative_suffix
    row["offset_x"] = str(int(float(offset_x or 0)))
    row["offset_y"] = str(int(float(offset_y or 0)))
    use_custom_length = _truthy_payload_value(custom_length) if custom_length is not None else bool(custom_seconds)
    if use_custom_length and custom_seconds:
        row["custom_seconds"] = f"{max(0.1, float(custom_seconds)):.3f}"
    else:
        row["custom_seconds"] = ""
    if guide_strength:
        try:
            row["guide_strength"] = f"{max(0.0, min(1.0, float(guide_strength))):.3f}"
        except ValueError:
            pass
    if guide_end_strength:
        try:
            row["guide_end_strength"] = f"{max(0.0, min(1.0, float(guide_end_strength))):.3f}"
        except ValueError:
            pass
    ordered = [rows[key] for key in sorted(rows)]
    write_outpaint_chunk_rows(resolve(str(manifest_text)), ordered)
    APP.log.append(f"Saved outpaint chunk {index + 1}: seed {row['seed']}")


def remove_cached_file(path: Path) -> bool:
    removed = False
    for candidate in (path, path.with_suffix(path.suffix + ".sig.json"), path.with_suffix(path.suffix + ".partial")):
        try:
            if candidate.exists() and candidate.is_file():
                candidate.unlink()
                removed = True
        except PermissionError:
            APP.log.append(f"Could not delete cached file because it is open in another process: {rel(candidate)}")
        except OSError as exc:
            APP.log.append(f"Could not delete cached file {rel(candidate)}: {exc}")
    return removed


def clear_cached_guide_frames(manifest: Path, index: int) -> int:
    guide_dir = ROOT / "intermediate" / "outpaint_guides" / manifest.stem
    if not guide_dir.exists():
        # Also check legacy path name used before the anchorâ†’guide rename.
        guide_dir = ROOT / "intermediate" / "outpaint_anchors" / manifest.stem
        if not guide_dir.exists():
            return 0
    removed = 0
    for path in guide_dir.glob(f"chunk_{index:04d}_*"):
        if path.is_file() and remove_cached_file(path):
            removed += 1
    return removed


def install_outpaint_guide(index: int) -> dict[str, str]:
    state = outpaint_chunks_state(APP.settings)
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    manifest = resolve(str(manifest_text))
    rows = read_outpaint_chunk_rows(manifest)
    if index not in rows:
        raise IndexError(f"Outpaint chunk not found: {index + 1}")

    current = rows[index].get("guide_image", "")
    selected = browse_path("image", current)
    if not selected:
        return {"selected": "", "guide_image": current}

    source = resolve(selected)
    if source.suffix.lower() not in IMAGE_EXTS:
        raise RuntimeError("Choose a PNG or JPEG image for the outpaint guide frame.")
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(source)

    target_dir = ROOT / "intermediate" / "outpaint_guides" / manifest.stem
    target = target_dir / f"chunk_{index:04d}_guide{source.suffix.lower()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

    rows[index]["guide_image"] = rel(target)
    write_outpaint_chunk_rows(manifest, [rows[key] for key in sorted(rows)])
    APP.log.append(f"Installed outpaint guide frame for chunk {index + 1}: {rel(target)}")
    return {"selected": selected, "guide_image": rel(target)}


def clear_outpaint_guide(index: int) -> dict[str, str]:
    state = outpaint_chunks_state(APP.settings)
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    manifest = resolve(str(manifest_text))
    rows = read_outpaint_chunk_rows(manifest)
    if index not in rows:
        raise IndexError(f"Outpaint chunk not found: {index + 1}")
    removed = clear_cached_guide_frames(manifest, index)
    rows[index]["guide_image"] = ""
    if "anchor_image" in rows[index]:
        rows[index]["anchor_image"] = ""
    write_outpaint_chunk_rows(manifest, [rows[key] for key in sorted(rows)])
    suffix = f" and deleted {removed} cached file(s)" if removed else ""
    APP.log.append(f"Cleared outpaint guide frame for chunk {index + 1}{suffix}")
    return {"guide_image": ""}


def clear_outpaint_anchor(index: int) -> dict[str, str]:
    return clear_outpaint_guide(index)


def install_outpaint_end_guide(index: int) -> dict[str, str]:
    state = outpaint_chunks_state(APP.settings)
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    manifest = resolve(str(manifest_text))
    rows = read_outpaint_chunk_rows(manifest)
    if index not in rows:
        raise IndexError(f"Outpaint chunk not found: {index + 1}")

    current = rows[index].get("guide_end_image", "")
    selected = browse_path("image", current)
    if not selected:
        return {"selected": "", "guide_end_image": current}

    source = resolve(selected)
    if source.suffix.lower() not in IMAGE_EXTS:
        raise RuntimeError("Choose a PNG or JPEG image for the outpaint end guide frame.")
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(source)

    target_dir = ROOT / "intermediate" / "outpaint_guides" / manifest.stem
    target = target_dir / f"chunk_{index:04d}_guide_end{source.suffix.lower()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

    rows[index]["guide_end_image"] = rel(target)
    write_outpaint_chunk_rows(manifest, [rows[key] for key in sorted(rows)])
    APP.log.append(f"Installed outpaint end guide frame for chunk {index + 1}: {rel(target)}")
    return {"selected": selected, "guide_end_image": rel(target)}


def clear_outpaint_end_guide(index: int) -> dict[str, str]:
    state = outpaint_chunks_state(APP.settings)
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    manifest = resolve(str(manifest_text))
    rows = read_outpaint_chunk_rows(manifest)
    if index not in rows:
        raise IndexError(f"Outpaint chunk not found: {index + 1}")
    # Remove the end guide file if it's in our managed directory.
    current = rows[index].get("guide_end_image", "")
    if current:
        path = resolve(current)
        remove_cached_file(path)
    rows[index]["guide_end_image"] = ""
    write_outpaint_chunk_rows(manifest, [rows[key] for key in sorted(rows)])
    APP.log.append(f"Cleared outpaint end guide frame for chunk {index + 1}")
    return {"guide_end_image": ""}














































































































































































from .http_handler import Handler, bind_context as bind_http_handler_context

bind_cache_context(globals())
bind_project_context(globals())
bind_media_context(globals())
bind_references_context(globals())
bind_file_dialogs_context(globals())
bind_lifecycle_context(globals())
bind_outpaint_guides_context(globals())
bind_http_handler_context(globals())





















APP.normalize_loaded_source_state()


def main() -> int:
    os.chdir(ROOT)
    install_shutdown_handlers()
    if os.environ.get("AI_REMASTER_NO_COMFY_AUTOSTART") != "1":
            start_comfy_if_needed()
    host = "127.0.0.1"
    requested_port = int(os.environ.get("AI_REMASTER_GUI_PORT", "8765"))
    server = create_server(host, requested_port)
    url = f"http://{host}:{server.server_port}/"
    print(f"AI Remaster GUI {APP_VERSION} running at {url}")
    if os.environ.get("AI_REMASTER_NO_BROWSER") != "1":
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        stop_started_comfy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
