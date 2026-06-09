"""ComfyUI graph builders for the Create Audio Track phase.

These mirror the in-code-graph approach used by ``upscale_video.py`` (``flashvsr_prompt``):
rather than shipping fragile workflow JSON, we build the prompt graph in Python and use
ComfyUI's ``/object_info`` to discover each node's input names and defaults. That keeps the
graphs resilient to differing model filenames and node versions across installs.

Three generators are provided:

* :func:`run_music_cue`  - Stable Audio Open (ComfyUI core audio nodes) text->music.
* :func:`run_sfx_chunk`  - MMAudio (kijai/ComfyUI-MMAudio) video->synchronized audio.
* :func:`run_caption`    - best-effort local Qwen-VL captioning of a single frame.

The model parts depend on the user's ComfyUI install; :func:`comfy_api.ensure_node_types`
raises a clear "install this custom node" error when a required node type is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_api import (
    ensure_node_types,
    extract_output_files,
    object_info,
    queue_prompt,
    wait_for_prompt,
)
from common import copy_to_comfy_input, newest_output

AUDIO_SUFFIXES = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}

# Node class names, overridable so a user can point at an equivalent node without code edits.
MUSIC_CHECKPOINT_NODE = "CheckpointLoaderSimple"
MUSIC_TEXT_ENCODE_NODE = "CLIPTextEncode"
MUSIC_EMPTY_LATENT_NODE = "EmptyLatentAudio"
MUSIC_SAMPLER_NODE = "KSampler"
MUSIC_VAE_DECODE_NODE = "VAEDecodeAudio"
SAVE_AUDIO_NODE = "SaveAudio"
LOAD_VIDEO_NODE = "VHS_LoadVideo"
LOAD_IMAGE_NODE = "LoadImage"

MMAUDIO_MODEL_LOADER = "MMAudioModelLoader"
MMAUDIO_FEATURE_LOADER = "MMAudioFeatureUtilsLoader"
MMAUDIO_SAMPLER = "MMAudioSampler"


def default_from_spec(spec: Any) -> Any:
    """Return a node input's default value from an /object_info spec (combo or {default:})."""
    if isinstance(spec, (list, tuple)):
        if len(spec) > 1 and isinstance(spec[1], dict) and "default" in spec[1]:
            return spec[1]["default"]
        if spec and isinstance(spec[0], list) and spec[0]:
            return spec[0][0]
    return None


def combo_values(spec: Any) -> list[str]:
    if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], list):
        return [str(value) for value in spec[0]]
    return []


def choose_combo_value(spec: Any, *needles: str) -> str | None:
    values = combo_values(spec)
    lowered = [(value, value.lower()) for value in values]
    for needle in needles:
        needle = needle.lower()
        for value, lower in lowered:
            if needle in lower:
                return value
    return None


def spec_type(spec: Any) -> Any:
    if isinstance(spec, (list, tuple)) and spec:
        return spec[0]
    return None


def node_input_groups(info: dict[str, Any], class_type: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    spec = info.get(class_type, {}).get("input", {})
    for group in ("required", "optional"):
        merged.update(spec.get(group) or {})
    return merged


def node_defaults(info: dict[str, Any], class_type: str, skip: set[str] = frozenset()) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, spec in node_input_groups(info, class_type).items():
        if name in skip:
            continue
        value = default_from_spec(spec)
        if value is not None:
            out[name] = value
    return out


def mmaudio_model_inputs(info: dict[str, Any]) -> dict[str, Any]:
    inputs = node_defaults(info, MMAUDIO_MODEL_LOADER)
    group = node_input_groups(info, MMAUDIO_MODEL_LOADER)
    if "mmaudio_model" in group:
        model = choose_combo_value(group["mmaudio_model"], "mmaudio_large", "mmaudio_medium", "mmaudio_small", "mmaudio")
        if model:
            inputs["mmaudio_model"] = model
    return inputs


def mmaudio_feature_inputs(info: dict[str, Any]) -> dict[str, Any]:
    inputs = node_defaults(info, MMAUDIO_FEATURE_LOADER)
    group = node_input_groups(info, MMAUDIO_FEATURE_LOADER)
    model_hints = {
        "vae_model": ("mmaudio_vae", "_vae", "vae"),
        "synchformer_model": ("synchformer",),
        "clip_model": ("clip", "dfn"),
    }
    for name, needles in model_hints.items():
        if name in group:
            value = choose_combo_value(group[name], *needles)
            if value:
                inputs[name] = value
    return inputs


def extract_audio_files(history_entry: dict[str, Any], output_root: Path) -> list[Path]:
    files: list[Path] = []
    for output in (history_entry.get("outputs") or {}).values():
        if not isinstance(output, dict):
            continue
        for key in ("audio", "audios"):
            for item in output.get(key, []) or []:
                filename = item.get("filename") if isinstance(item, dict) else None
                if not filename:
                    continue
                subfolder = item.get("subfolder") or ""
                files.append(output_root / subfolder / filename)
    existing = [path for path in files if path.exists()]
    # Fall back to the generic extractor (some SaveAudio variants report under "gifs"/"videos").
    if not existing:
        existing = [p for p in extract_output_files(history_entry, output_root) if p.suffix.lower() in AUDIO_SUFFIXES]
    return existing


def extract_text_outputs(history_entry: dict[str, Any]) -> str:
    """Collect any string values surfaced in node UI outputs (e.g. caption / ShowText nodes)."""
    found: list[str] = []
    for output in (history_entry.get("outputs") or {}).values():
        if not isinstance(output, dict):
            continue
        for key in ("text", "string", "caption", "STRING"):
            value = output.get(key)
            if isinstance(value, str):
                found.append(value)
            elif isinstance(value, list):
                found.extend(str(item) for item in value if isinstance(item, (str, int, float)))
    found = [text.strip() for text in found if str(text).strip()]
    return max(found, key=len) if found else ""


# ── Music: Stable Audio Open ──────────────────────────────────────────────────


def music_prompt_graph(
    info: dict[str, Any],
    *,
    checkpoint: str,
    prompt: str,
    negative: str,
    seconds: float,
    steps: int,
    cfg: float,
    seed: int,
    prefix: str,
) -> dict[str, Any]:
    sampler_defaults = node_defaults(
        info,
        MUSIC_SAMPLER_NODE,
        skip={"model", "positive", "negative", "latent_image"},
    )
    sampler_defaults.update({"seed": seed, "steps": steps, "cfg": cfg, "denoise": 1.0})
    latent_defaults = node_defaults(info, MUSIC_EMPTY_LATENT_NODE)
    latent_defaults["seconds"] = round(float(seconds), 3)
    latent_defaults.setdefault("batch_size", 1)
    return {
        "1": {"class_type": MUSIC_CHECKPOINT_NODE, "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": MUSIC_TEXT_ENCODE_NODE, "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": MUSIC_TEXT_ENCODE_NODE, "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": MUSIC_EMPTY_LATENT_NODE, "inputs": latent_defaults},
        "5": {
            "class_type": MUSIC_SAMPLER_NODE,
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                **sampler_defaults,
            },
        },
        "6": {"class_type": MUSIC_VAE_DECODE_NODE, "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": SAVE_AUDIO_NODE, "inputs": {"audio": ["6", 0], "filename_prefix": prefix}},
    }


def ensure_checkpoint_choice(info: dict[str, Any], checkpoint: str) -> None:
    group = node_input_groups(info, MUSIC_CHECKPOINT_NODE)
    choices = combo_values(group.get("ckpt_name"))
    if not choices or checkpoint in choices:
        return
    available = ", ".join(choices) if choices else "none"
    raise RuntimeError(
        f"Stable Audio checkpoint '{checkpoint}' is not available in ComfyUI's checkpoint list. "
        f"ComfyUI currently reports: {available}. "
        f"Place the Stable Audio Open file at ComfyUI/models/checkpoints/{checkpoint}, then fully restart ComfyUI. "
        f"If it is missing, accept the gated Hugging Face licence for stabilityai/stable-audio-open-1.0 "
        f"and authenticate with 'hf auth login' or HF_TOKEN before retrying."
    )


def run_music_cue(
    comfy_url: str,
    comfy_output_root: Path,
    *,
    checkpoint: str,
    prompt: str,
    negative: str,
    seconds: float,
    steps: int,
    cfg: float,
    seed: int,
    prefix: str,
    poll_seconds: float,
) -> Path:
    ensure_node_types(
        comfy_url,
        {
            MUSIC_CHECKPOINT_NODE: "ComfyUI (core)",
            MUSIC_EMPTY_LATENT_NODE: "ComfyUI (core, Stable Audio)",
            MUSIC_VAE_DECODE_NODE: "ComfyUI (core, Stable Audio)",
            SAVE_AUDIO_NODE: "ComfyUI (core)",
        },
        "music score generation",
    )
    info = object_info(comfy_url)
    ensure_checkpoint_choice(info, checkpoint)
    graph = music_prompt_graph(
        info,
        checkpoint=checkpoint,
        prompt=prompt,
        negative=negative,
        seconds=seconds,
        steps=steps,
        cfg=cfg,
        seed=seed,
        prefix=prefix,
    )
    prompt_id = queue_prompt(comfy_url, graph)
    history = wait_for_prompt(comfy_url, prompt_id, poll_seconds)
    return newest_output(extract_audio_files(history, comfy_output_root), AUDIO_SUFFIXES, "music cue audio")


# ── Sound effects: MMAudio (video -> synchronized audio) ──────────────────────


def sfx_prompt_graph(
    info: dict[str, Any],
    *,
    video_name: str,
    prompt: str,
    negative: str,
    seconds: float,
    steps: int,
    cfg: float,
    seed: int,
    prefix: str,
) -> dict[str, Any]:
    model_inputs = mmaudio_model_inputs(info)
    feature_inputs = mmaudio_feature_inputs(info)
    sampler_inputs = node_defaults(
        info,
        MMAUDIO_SAMPLER,
        skip={"mmaudio_model", "feature_utils", "features", "images"},
    )
    # Wire the inputs MMAudioSampler actually exposes (names vary slightly by version).
    sampler_group = node_input_groups(info, MMAUDIO_SAMPLER)
    sampler_inputs["images"] = ["3", 0]
    for model_key in ("mmaudio_model", "model"):
        if model_key in sampler_group:
            sampler_inputs[model_key] = ["1", 0]
            break
    for feature_key in ("feature_utils", "features", "mmaudio_featureutils"):
        if feature_key in sampler_group:
            sampler_inputs[feature_key] = ["2", 0]
            break
    if "duration" in sampler_group:
        sampler_inputs["duration"] = round(float(seconds), 3)
    if "steps" in sampler_group:
        sampler_inputs["steps"] = steps
    if "cfg" in sampler_group:
        sampler_inputs["cfg"] = cfg
    if "seed" in sampler_group:
        sampler_inputs["seed"] = seed
    if "prompt" in sampler_group:
        sampler_inputs["prompt"] = prompt
    if "negative_prompt" in sampler_group:
        sampler_inputs["negative_prompt"] = negative
    return {
        "1": {"class_type": MMAUDIO_MODEL_LOADER, "inputs": model_inputs},
        "2": {"class_type": MMAUDIO_FEATURE_LOADER, "inputs": feature_inputs},
        "3": {
            "class_type": LOAD_VIDEO_NODE,
            "inputs": {
                "video": video_name,
                "force_rate": 0.0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
                "format": "None",
            },
        },
        "4": {"class_type": MMAUDIO_SAMPLER, "inputs": sampler_inputs},
        "5": {"class_type": SAVE_AUDIO_NODE, "inputs": {"audio": ["4", 0], "filename_prefix": prefix}},
    }


def run_sfx_chunk(
    comfy_url: str,
    comfy_dir: Path,
    comfy_output_root: Path,
    *,
    proxy_video: Path,
    prompt: str,
    negative: str,
    seconds: float,
    steps: int,
    cfg: float,
    seed: int,
    prefix: str,
    poll_seconds: float,
) -> Path:
    ensure_node_types(
        comfy_url,
        {
            MMAUDIO_MODEL_LOADER: "ComfyUI-MMAudio",
            MMAUDIO_FEATURE_LOADER: "ComfyUI-MMAudio",
            MMAUDIO_SAMPLER: "ComfyUI-MMAudio",
            LOAD_VIDEO_NODE: "ComfyUI-VideoHelperSuite",
            SAVE_AUDIO_NODE: "ComfyUI (core)",
        },
        "sound-effects generation (MMAudio)",
    )
    info = object_info(comfy_url)
    video_name = copy_to_comfy_input(proxy_video, comfy_dir, "arp_audio_sfx")
    graph = sfx_prompt_graph(
        info,
        video_name=video_name,
        prompt=prompt,
        negative=negative,
        seconds=seconds,
        steps=steps,
        cfg=cfg,
        seed=seed,
        prefix=prefix,
    )
    prompt_id = queue_prompt(comfy_url, graph)
    history = wait_for_prompt(comfy_url, prompt_id, poll_seconds)
    return newest_output(extract_audio_files(history, comfy_output_root), AUDIO_SUFFIXES, "MMAudio chunk audio")


# ── Captioning: best-effort local Qwen-VL ─────────────────────────────────────


def caption_graph(info: dict[str, Any], *, image_name: str, node_class: str, question: str) -> dict[str, Any] | None:
    if node_class not in info:
        return None
    group = node_input_groups(info, node_class)
    image_input = next((name for name, spec in group.items() if spec_type(spec) == "IMAGE"), None)
    if not image_input:
        return None
    inputs = node_defaults(info, node_class, skip={image_input})
    inputs[image_input] = ["1", 0]
    # Set a question/instruction on the first STRING-typed widget, if the node has one.
    text_input = next(
        (name for name, spec in group.items() if spec_type(spec) == "STRING" and name != image_input),
        None,
    )
    if text_input:
        inputs[text_input] = question
    graph: dict[str, Any] = {
        "1": {"class_type": LOAD_IMAGE_NODE, "inputs": {"image": image_name}},
        "2": {"class_type": node_class, "inputs": inputs},
    }
    # If a ShowText sink exists, route the caption text through it so it lands in history outputs.
    if "ShowText|pysssss" in info:
        graph["3"] = {"class_type": "ShowText|pysssss", "inputs": {"text": ["2", 0]}}
    return graph


def run_caption(
    comfy_url: str,
    comfy_dir: Path,
    *,
    image_path: Path,
    node_class: str,
    question: str,
    poll_seconds: float,
) -> str:
    """Caption one frame with a local Qwen-VL node. Returns "" if unavailable/failed."""
    if not node_class:
        return ""
    info = object_info(comfy_url)
    image_name = copy_to_comfy_input(image_path, comfy_dir, "arp_audio_caption")
    graph = caption_graph(info, image_name=image_name, node_class=node_class, question=question)
    if graph is None:
        return ""
    prompt_id = queue_prompt(comfy_url, graph)
    history = wait_for_prompt(comfy_url, prompt_id, poll_seconds)
    return extract_text_outputs(history)
