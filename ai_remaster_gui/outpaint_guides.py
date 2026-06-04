from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from .config import IMAGE_EXTS, ROOT, SCRIPTS
from .manifests import read_outpaint_chunk_rows, write_outpaint_chunk_rows
from .media import extract_video_frame_at
from .paths import rel, resolve, resolve_video_source


def bind_context(context: dict) -> None:
    globals().update(context)


def chunk_frame_preview(source: Path, seconds: float, suffix: str) -> str:
    if not source.exists():
        return ""
    return extract_video_frame_at(source, FILE_PREVIEW_DIR / "chunks", f"{suffix}_{int(seconds * 1000):010d}", seconds)

def _parse_guide_frames(row: dict[str, str]) -> list[dict]:
    """Return the guide_frames list for a manifest row, migrating from old fields if needed."""
    raw = row.get("guide_frames", "").strip()
    if raw:
        try:
            frames = json.loads(raw)
            if isinstance(frames, list):
                return frames
        except json.JSONDecodeError:
            pass
    # Migrate from legacy guide_image / guide_end_image fields.
    frames: list[dict] = []
    if row.get("guide_image"):
        try:
            strength = float(row.get("guide_strength", "0.7") or "0.7")
        except ValueError:
            strength = 0.7
        frames.append({"frame_idx": 0, "strength": round(strength, 3), "image": row["guide_image"]})
    if row.get("guide_end_image"):
        try:
            strength = float(row.get("guide_end_strength", "1.0") or "1.0")
        except ValueError:
            strength = 1.0
        frames.append({"frame_idx": -1, "strength": round(strength, 3), "image": row["guide_end_image"]})
    return frames

def _save_guide_frames(manifest: Path, chunk_index: int, frames: list[dict]) -> None:
    """Persist guide_frames JSON back to the manifest row."""
    rows = read_outpaint_chunk_rows(manifest)
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    rows[chunk_index]["guide_frames"] = json.dumps(frames)
    write_outpaint_chunk_rows(manifest, [rows[k] for k in sorted(rows)])

def _guide_source_seconds(row: dict, frame_idx: int, fps: float) -> float:
    """Convert a frame_idx (possibly negative) to absolute seconds in the prepared canvas."""
    start = float(row.get("start", 0.0))
    end = float(row.get("end", 0.0))
    length_frames = int(row.get("end_frame", 0)) - int(row.get("start_frame", 0))
    if frame_idx < 0:
        actual = max(0, length_frames + frame_idx)
    else:
        actual = frame_idx
    return max(start, min(end - (1.0 / max(1.0, fps)), start + actual / max(1.0, fps)))

def _build_guide_frames_view(
    row: dict,
    source_text: str,
    aspect: str,
    start_seconds: float,
    end_seconds: float,
    fps: float,
    length_frames: int,
) -> list[dict]:
    """Build the view list for guide frames, including thumbnail previews."""
    frames = _parse_guide_frames(row)
    view = []
    for i, gf in enumerate(frames):
        frame_idx = int(gf.get("frame_idx", 0))
        strength = float(gf.get("strength", 0.7))
        image_rel = gf.get("image", "")
        image_path = resolve(image_rel) if image_rel else None
        image_exists = bool(image_path and image_path.exists())
        source_secs = _guide_source_seconds(
            {"start": start_seconds, "end": end_seconds,
             "start_frame": str(int(start_seconds * fps)),
             "end_frame": str(int(end_seconds * fps))},
            frame_idx, fps,
        )
        view.append({
            "guide_index": i,
            "frame_idx": frame_idx,
            "strength": strength,
            "image": image_rel,
            "image_exists": image_exists,
            "image_mtime": int(image_path.stat().st_mtime_ns) if image_exists and image_path else 0,
            "source_preview": "",
        })
    return view

def guide_frame_generation_command(chunk_index: int, guide_index: int, frame_idx: int, prompt: str) -> tuple[list[str], str, Path, float]:
    """Build the Qwen generation command for any guide frame position."""
    state = outpaint_chunks_state(APP.settings)
    rows = state.get("rows", [])
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    if chunk_index < 0 or chunk_index >= len(rows):
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")

    row = rows[chunk_index]
    fps = float(row.get("fps", 24) or 24)
    source_seconds = _guide_source_seconds(row, frame_idx, fps)

    source_text = pipeline_source_text(APP.settings)
    if not source_text:
        raise RuntimeError("No source material is selected.")
    range_source = ensure_outpaint_prepared_canvas(source_text, APP.settings.get("outpaint", {}))
    cache_key = f"gf_qwen_{int(source_seconds * 1000):010d}"
    preview_rel = chunk_frame_preview(range_source, source_seconds, cache_key)
    source_img = resolve(preview_rel) if preview_rel else Path("")
    if not source_img.is_file():
        raise FileNotFoundError(f"Could not extract source frame for Qwen guide at {source_seconds:.3f}s from {range_source}.")

    manifest = resolve(str(manifest_text))
    output_dir = ROOT / "intermediate" / "outpaint_guides" / manifest.stem
    output = output_dir / f"chunk_{chunk_index:04d}_guide_{guide_index:02d}_qwen.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_cached_file(output)
    source_img = save_qwen_input_copy(source_img, output.with_name(f"chunk_{chunk_index:04d}_guide_{guide_index:02d}_qwen_input{source_img.suffix.lower() or '.jpg'}"))

    # Pre-write the output path into guide_frames so the thumbnail updates immediately.
    stored = read_outpaint_chunk_rows(manifest)
    if chunk_index in stored:
        frames = _parse_guide_frames(stored[chunk_index])
        if 0 <= guide_index < len(frames):
            frames[guide_index]["image"] = rel(output)
        else:
            frames.append({"frame_idx": frame_idx, "strength": 0.7, "image": rel(output)})
        stored[chunk_index]["guide_frames"] = json.dumps(frames)
        write_outpaint_chunk_rows(manifest, [stored[k] for k in sorted(stored)])

    values = APP.settings.get("references", {})
    config = current_config()
    workflow = qwen_workflow_for(values, config)
    if not workflow:
        raise RuntimeError("No Qwen Image Edit workflow found. Install/configure ComfyUI first.")
    guide_prompt = prompt.strip() or DEFAULT_ANCHOR_PROMPT
    cmd = [
        sys.executable, "-u",
        str(SCRIPTS / "generate_single_reference.py"),
        "--source-image", str(source_img),
        "--output", str(output),
        "--workflow", workflow,
        "--comfy-url", values.get("comfy_url") or config.get("comfy_url", "http://127.0.0.1:8188"),
        "--comfy-dir", config.get("comfy_dir", str(ROOT / "tools" / "comfyui")),
        "--comfy-output-root", values.get("comfy_output_root") or str(Path(config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))) / "output"),
        "--model-backend", values.get("model_backend", "gguf"),
        "--gguf-model", values.get("gguf_model", "qwen-image-edit-2511-Q4_K_M.gguf"),
        "--prompt", guide_prompt,
        "--prompt-suffix", "",
        "--load-image-node-id", values.get("load_image_node_id", "auto"),
        "--save-node-id", values.get("save_node_id", "auto"),
        "--no-normalize-to-source-size",
        "--force",
    ]
    if values.get("prompt_node_id"):
        cmd.extend(["--prompt-node-id", values["prompt_node_id"]])
    return cmd, rel(output), resolve(range_source), source_seconds

def _composite_guide_in_place(output: Path, prepared_canvas: Path, source_seconds: float | None = None) -> None:
    """Composite a Qwen guide PNG with actual source content, then inpaint black corners, in-place.

    Steps:
      1. Scale the Qwen guide to exactly match the LTX canvas size (stretch, not crop).
      2. Overlay actual source pixels from the prepared canvas wherever they are non-black
         (i.e. the source content area â€” e.g. 960Ã—704 centred in 1280Ã—704).  This ensures
         pixel-accurate alignment between the guide and the prepared canvas regardless of any
         sub-pixel shifts introduced by Qwen's internal patch processing.
      3. Inpaint any remaining near-black pixels (corners where both guide and source are black).
    Saves the result back over *output*.

    source_seconds: timestamp in the prepared canvas to use as the source frame.
    Defaults to t=0 (actual first frame).
    """
    import cv2
    import numpy as np
    from PIL import Image as PILImage

    # Read prepared canvas dimensions and extract the source frame.
    cap = cv2.VideoCapture(str(prepared_canvas))
    canvas_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    canvas_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    seek_ms = (source_seconds * 1000.0) if source_seconds is not None else 0.0
    cap.set(cv2.CAP_PROP_POS_MSEC, seek_ms)
    ok, src_frame = cap.read()
    if not ok or src_frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, src_frame = cap.read()
    cap.release()
    if not ok or src_frame is None:
        raise RuntimeError(f"Could not read frame from prepared canvas: {prepared_canvas}")
    if src_frame.shape[1] != canvas_w or src_frame.shape[0] != canvas_h:
        src_frame = cv2.resize(src_frame, (canvas_w, canvas_h), interpolation=cv2.INTER_LANCZOS4)

    with PILImage.open(output) as img:
        img_w, img_h = img.size
        guide_rgb = img.convert("RGB")

    # Preserve the raw Qwen output alongside the composited result for inspection.
    raw_copy = output.with_name(output.stem + "_raw" + output.suffix)
    if not raw_copy.exists():
        import shutil as _shutil
        _shutil.copy2(output, raw_copy)

    resampling = getattr(PILImage, "Resampling", PILImage).LANCZOS

    # Step 1: scale Qwen output to fit within the canvas, preserving AR.
    # Use the tighter dimension (min) so the image never exceeds either canvas axis.
    # For landscape (e.g. 1280Ã—704): Qwen is typically slightly wider in AR, so width is the
    # tighter constraint and the result is e.g. 1280Ã—691 with spare pixels top/bottom.
    # For portrait (e.g. 704Ã—1280): height is typically the tighter constraint.
    # A calibrated 1px-left / 1px-down nudge is applied for pixel-perfect alignment.
    scale = min(canvas_w / img_w, canvas_h / img_h)
    new_w = max(1, int(round(img_w * scale)))
    new_h = max(1, int(round(img_h * scale)))
    resized = guide_rgb.resize((new_w, new_h), resampling)

    nominal_x = (canvas_w - new_w) // 2
    nominal_y = (canvas_h - new_h) // 2
    paste_x = nominal_x - 1   # 1 px left
    paste_y = nominal_y + 1   # 1 px down

    canvas_pil = PILImage.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    # Clip to canvas bounds, cropping the source image if the offset is negative.
    src_x0 = max(0, -paste_x)
    src_y0 = max(0, -paste_y)
    dst_x0 = max(0, paste_x)
    dst_y0 = max(0, paste_y)
    blit_w = min(new_w - src_x0, canvas_w - dst_x0)
    blit_h = min(new_h - src_y0, canvas_h - dst_y0)
    if blit_w > 0 and blit_h > 0:
        region = resized.crop((src_x0, src_y0, src_x0 + blit_w, src_y0 + blit_h))
        canvas_pil.paste(region, (dst_x0, dst_y0))

    canvas_bgr = cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)

    # Step 2: blend the source frame's content pixels over the centre with a soft edge.
    # black_lift raises all source pixels above 0; the padding margins are exact black (0,0,0).
    # A ~10px Gaussian feather at the content boundary avoids a hard seam where Qwen's
    # outpainting meets the composited source pixels.
    src_is_content = np.any(src_frame > 4, axis=2)
    if src_is_content.any():
        feather_px = 10
        alpha = cv2.GaussianBlur(
            src_is_content.astype(np.float32),
            (feather_px * 2 + 1, feather_px * 2 + 1),
            feather_px / 2,
        )
        # Mask the alpha back to zero outside the content area so the blur never
        # bleeds black pillar pixels into Qwen's outpainting â€” feather is inward only.
        alpha = (alpha * src_is_content.astype(np.float32))[:, :, np.newaxis]
        canvas_bgr = (
            src_frame.astype(np.float32) * alpha
            + canvas_bgr.astype(np.float32) * (1.0 - alpha)
        ).clip(0, 255).astype(np.uint8)

    # Step 3: inpaint the small corner triangles that remain black
    # (top/bottom strips outside both the Qwen letterbox and the source content area).
    still_black = np.all(canvas_bgr <= 4, axis=2).astype(np.uint8) * 255
    if still_black.any():
        canvas_bgr = cv2.inpaint(canvas_bgr, still_black, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    PILImage.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)).save(output, format="PNG")

def save_qwen_input_copy(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    shutil.copy2(source, target)
    return target

def outpaint_guide_generation_command(index: int, prompt: str) -> tuple[list[str], str, Path]:
    state = outpaint_chunks_state(APP.settings)
    rows = state.get("rows", [])
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    if index < 0 or index >= len(rows):
        raise IndexError(f"Outpaint chunk not found: {index + 1}")

    row = rows[index]
    fps = float(row.get("fps", 24) or 24)
    start_seconds = float(row.get("start", 0.0))
    guide_source_seconds = start_seconds
    source_text = pipeline_source_text(APP.settings)
    if not source_text:
        raise RuntimeError("No source material is selected.")
    range_source = ensure_outpaint_prepared_canvas(source_text, APP.settings.get("outpaint", {}))
    preview_rel = chunk_frame_preview(range_source, guide_source_seconds, "source_guide_qwen")
    source = resolve(preview_rel) if preview_rel else Path("")
    if not source.is_file():
        raise FileNotFoundError(f"Could not extract source frame for Qwen guide at {guide_source_seconds:.3f}s from {range_source}.")

    manifest = resolve(str(manifest_text))
    output_dir = ROOT / "intermediate" / "outpaint_guides" / manifest.stem
    output = output_dir / f"chunk_{index:04d}_guide_qwen.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_cached_file(output)
    source = save_qwen_input_copy(source, output.with_name(f"chunk_{index:04d}_guide_qwen_input{source.suffix.lower() or '.jpg'}"))

    stored = read_outpaint_chunk_rows(manifest)
    if index not in stored:
        raise IndexError(f"Outpaint chunk not found in manifest: {index + 1}")
    stored[index]["guide_image"] = rel(output)
    write_outpaint_chunk_rows(manifest, [stored[key] for key in sorted(stored)])

    values = APP.settings.get("references", {})
    config = current_config()
    workflow = qwen_workflow_for(values, config)
    if not workflow:
        raise RuntimeError("No Qwen Image Edit workflow found. Install/configure ComfyUI first.")
    guide_prompt = prompt.strip() or DEFAULT_ANCHOR_PROMPT
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPTS / "generate_single_reference.py"),
        "--source-image",
        str(source),
        "--output",
        str(output),
        "--workflow",
        workflow,
        "--comfy-url",
        values.get("comfy_url") or config.get("comfy_url", "http://127.0.0.1:8188"),
        "--comfy-dir",
        config.get("comfy_dir", str(ROOT / "tools" / "comfyui")),
        "--comfy-output-root",
        values.get("comfy_output_root") or str(Path(config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))) / "output"),
        "--model-backend",
        values.get("model_backend", "gguf"),
        "--gguf-model",
        values.get("gguf_model", "qwen-image-edit-2511-Q4_K_M.gguf"),
        "--prompt",
        guide_prompt,
        "--prompt-suffix",
        "",
        "--load-image-node-id",
        values.get("load_image_node_id", "auto"),
        "--save-node-id",
        values.get("save_node_id", "auto"),
        "--no-normalize-to-source-size",
        "--force",
    ]
    if values.get("prompt_node_id"):
        cmd.extend(["--prompt-node-id", values["prompt_node_id"]])
    return cmd, rel(output), resolve(range_source), guide_source_seconds

def outpaint_end_guide_generation_command(index: int, prompt: str) -> tuple[list[str], str, Path, float]:
    state = outpaint_chunks_state(APP.settings)
    rows = state.get("rows", [])
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    if index < 0 or index >= len(rows):
        raise IndexError(f"Outpaint chunk not found: {index + 1}")

    row = rows[index]
    fps = float(row.get("fps", 24) or 24)
    end_seconds = float(row.get("end", 0.0))
    # Use the last meaningful frame (end - 1/fps) as the Qwen source for the end guide.
    guide_source_seconds = max(float(row.get("start", 0.0)), end_seconds - (1.0 / max(1.0, fps)))
    source_text = pipeline_source_text(APP.settings)
    if not source_text:
        raise RuntimeError("No source material is selected.")
    range_source = ensure_outpaint_prepared_canvas(source_text, APP.settings.get("outpaint", {}))
    preview_rel = chunk_frame_preview(range_source, guide_source_seconds, "source_guide_end_qwen")
    source = resolve(preview_rel) if preview_rel else Path("")
    if not source.is_file():
        raise FileNotFoundError(f"Could not extract source frame for Qwen end guide at {guide_source_seconds:.3f}s from {range_source}.")

    manifest = resolve(str(manifest_text))
    output_dir = ROOT / "intermediate" / "outpaint_guides" / manifest.stem
    output = output_dir / f"chunk_{index:04d}_guide_end_qwen.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_cached_file(output)
    source = save_qwen_input_copy(source, output.with_name(f"chunk_{index:04d}_guide_end_qwen_input{source.suffix.lower() or '.jpg'}"))

    stored = read_outpaint_chunk_rows(manifest)
    if index not in stored:
        raise IndexError(f"Outpaint chunk not found in manifest: {index + 1}")
    stored[index]["guide_end_image"] = rel(output)
    write_outpaint_chunk_rows(manifest, [stored[key] for key in sorted(stored)])

    values = APP.settings.get("references", {})
    config = current_config()
    workflow = qwen_workflow_for(values, config)
    if not workflow:
        raise RuntimeError("No Qwen Image Edit workflow found. Install/configure ComfyUI first.")
    guide_prompt = prompt.strip() or DEFAULT_ANCHOR_PROMPT
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPTS / "generate_single_reference.py"),
        "--source-image",
        str(source),
        "--output",
        str(output),
        "--workflow",
        workflow,
        "--comfy-url",
        values.get("comfy_url") or config.get("comfy_url", "http://127.0.0.1:8188"),
        "--comfy-dir",
        config.get("comfy_dir", str(ROOT / "tools" / "comfyui")),
        "--comfy-output-root",
        values.get("comfy_output_root") or str(Path(config.get("comfy_dir", str(ROOT / "tools" / "comfyui"))) / "output"),
        "--model-backend",
        values.get("model_backend", "gguf"),
        "--gguf-model",
        values.get("gguf_model", "qwen-image-edit-2511-Q4_K_M.gguf"),
        "--prompt",
        guide_prompt,
        "--prompt-suffix",
        "",
        "--load-image-node-id",
        values.get("load_image_node_id", "auto"),
        "--save-node-id",
        values.get("save_node_id", "auto"),
        "--no-normalize-to-source-size",
        "--force",
    ]
    if values.get("prompt_node_id"):
        cmd.extend(["--prompt-node-id", values["prompt_node_id"]])
    return cmd, rel(output), resolve(range_source), guide_source_seconds


def _get_guide_manifest() -> tuple[Path, dict[int, dict[str, str]], str]:
    state = outpaint_chunks_state(APP.settings)
    manifest_text = state.get("manifest", "")
    if not manifest_text:
        raise RuntimeError("No outpaint chunk manifest is available yet.")
    manifest = resolve(str(manifest_text))
    rows = read_outpaint_chunk_rows(manifest)
    return manifest, rows, manifest_text

def add_guide_frame(chunk_index: int) -> dict:
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    frames.append({"frame_idx": 0, "strength": 0.7, "image": ""})
    _save_guide_frames(manifest, chunk_index, frames)
    APP.log.append(f"Added guide frame to chunk {chunk_index + 1} (total: {len(frames)})")
    return {"guide_index": len(frames) - 1}

def remove_guide_frame(chunk_index: int, guide_index: int) -> dict:
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    removed = frames.pop(guide_index)
    if removed.get("image"):
        remove_cached_file(resolve(removed["image"]))
    _save_guide_frames(manifest, chunk_index, frames)
    APP.log.append(f"Removed guide frame {guide_index} from chunk {chunk_index + 1}")
    return {"removed": guide_index}

def save_guide_frame(chunk_index: int, guide_index: int, frame_idx: int, strength: float) -> dict:
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    frames[guide_index]["frame_idx"] = int(frame_idx)
    frames[guide_index]["strength"] = round(max(0.0, min(1.0, float(strength))), 3)
    _save_guide_frames(manifest, chunk_index, frames)
    APP.log.append(f"Saved guide frame {guide_index} for chunk {chunk_index + 1}: frame_idx={frame_idx}, strength={strength:.2f}")
    return {"frame_idx": frame_idx, "strength": frames[guide_index]["strength"]}

def upload_guide_frame_image(chunk_index: int, guide_index: int) -> dict:
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    current = frames[guide_index].get("image", "")
    selected = browse_path("image", current)
    if not selected:
        return {"selected": "", "image": current}
    source = resolve(selected)
    if source.suffix.lower() not in IMAGE_EXTS:
        raise RuntimeError("Choose a PNG or JPEG image for the guide frame.")
    target_dir = ROOT / "intermediate" / "outpaint_guides" / manifest.stem
    target = target_dir / f"chunk_{chunk_index:04d}_guide_{guide_index:02d}{source.suffix.lower()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    frames[guide_index]["image"] = rel(target)
    _save_guide_frames(manifest, chunk_index, frames)
    APP.log.append(f"Uploaded guide frame {guide_index} for chunk {chunk_index + 1}: {rel(target)}")
    return {"selected": selected, "image": rel(target)}

def clear_guide_frame_image(chunk_index: int, guide_index: int) -> dict:
    manifest, rows, _ = _get_guide_manifest()
    if chunk_index not in rows:
        raise IndexError(f"Outpaint chunk not found: {chunk_index + 1}")
    frames = _parse_guide_frames(rows[chunk_index])
    if guide_index < 0 or guide_index >= len(frames):
        raise IndexError(f"Guide frame {guide_index} not found in chunk {chunk_index + 1}")
    current = frames[guide_index].get("image", "")
    if current:
        remove_cached_file(resolve(current))
    frames[guide_index]["image"] = ""
    _save_guide_frames(manifest, chunk_index, frames)
    APP.log.append(f"Cleared guide frame {guide_index} image for chunk {chunk_index + 1}")
    return {"image": ""}
