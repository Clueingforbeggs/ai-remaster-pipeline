from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any

from comfy_api import ensure_node_types, extract_output_files, object_info, queue_prompt, wait_for_comfy, wait_for_prompt
from common import ROOT, copy_to_comfy_input, file_fingerprint, find_ffmpeg, load_local_config, newest_output as newest_comfy_output, replace_with_retry, resolve_path, root_relative, safe_stem, resumable_output, video_info, write_signature


config = load_local_config()


def default_output(source: Path, width: int, height: int) -> Path:
    suffix = f"flashvsr_{width}x{height}" if width and height else "flashvsr"
    return ROOT / "output" / "upscaled" / f"{safe_stem(source.name)}_{suffix}.mp4"


def signature(args: argparse.Namespace, source: Path, output_width: int, output_height: int) -> dict[str, Any]:
    return {
        "version": 4,
        "tool": "upscale_video.py",
        "method": "flashvsr_ultra_fast",
        "source": root_relative(source),
        "source_fingerprint": file_fingerprint(source),
        "target_width": output_width,
        "target_height": output_height,
        "comfy_dir": root_relative(resolve_path(args.comfy_dir)),
        "comfy_url": args.comfy_url,
        "flashvsr_model": args.flashvsr_model,
        "flashvsr_mode": args.flashvsr_mode,
        "flashvsr_scale": args.flashvsr_scale,
        "flashvsr_tiled_vae": args.flashvsr_tiled_vae,
        "flashvsr_tiled_dit": args.flashvsr_tiled_dit,
        "flashvsr_unload_dit": args.flashvsr_unload_dit,
        "flashvsr_seed": args.flashvsr_seed,
        "fps": args.fps,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upscale an ARP video.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--target-width", type=int, default=3840)
    parser.add_argument("--target-height", type=int, default=2160)
    parser.add_argument("--comfy-dir", default=config.get("comfy_dir", str(ROOT / "tools" / "comfyui")))
    parser.add_argument("--comfy-url", default=config.get("comfy_url", "http://127.0.0.1:8188"))
    parser.add_argument("--comfy-output-root", default="")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--flashvsr-model", choices=["FlashVSR", "FlashVSR-v1.1"], default="FlashVSR-v1.1")
    parser.add_argument("--flashvsr-mode", choices=["tiny", "tiny-long", "full"], default="tiny")
    parser.add_argument("--flashvsr-scale", type=int, default=2)
    parser.add_argument("--flashvsr-tiled-vae", dest="flashvsr_tiled_vae", action="store_true", default=True)
    parser.add_argument("--no-flashvsr-tiled-vae", dest="flashvsr_tiled_vae", action="store_false")
    parser.add_argument("--flashvsr-tiled-dit", dest="flashvsr_tiled_dit", action="store_true", default=True)
    parser.add_argument("--no-flashvsr-tiled-dit", dest="flashvsr_tiled_dit", action="store_false")
    parser.add_argument("--flashvsr-unload-dit", dest="flashvsr_unload_dit", action="store_true", default=False)
    parser.add_argument("--no-flashvsr-unload-dit", dest="flashvsr_unload_dit", action="store_false")
    parser.add_argument("--flashvsr-seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--ffmpeg", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def default_from_spec(spec: Any) -> Any:
    if isinstance(spec, (list, tuple)):
        if len(spec) > 1 and isinstance(spec[1], dict) and "default" in spec[1]:
            return spec[1]["default"]
        if spec and isinstance(spec[0], list) and spec[0]:
            return spec[0][0]
    return None


def flashvsr_node_inputs(args: argparse.Namespace, class_type: str, source_node: str, info: dict[str, Any]) -> dict[str, Any]:
    inputs: dict[str, Any] = {"frames": [source_node, 0]}
    input_info = info.get(class_type, {}).get("input", {})
    accepted: set[str] = set()
    for group in ("required", "optional"):
        for name, spec in (input_info.get(group) or {}).items():
            accepted.add(name)
            if name == "frames":
                continue
            value = default_from_spec(spec)
            if value is not None:
                inputs[name] = value

    # Use the Ultra Fast node's real input names so ARP logs and saved settings map
    # directly to the Comfy node.
    for name, value in {
        "model": args.flashvsr_model,
        "mode": args.flashvsr_mode,
        "scale": args.flashvsr_scale,
        "tiled_vae": bool(args.flashvsr_tiled_vae),
        "tiled_dit": bool(args.flashvsr_tiled_dit),
        "unload_dit": bool(args.flashvsr_unload_dit),
        "seed": args.flashvsr_seed,
    }.items():
        if not accepted or name in accepted:
            inputs[name] = value
    return inputs


def flashvsr_prompt(video_name: str, fps: float, args: argparse.Namespace, prefix: str, info: dict[str, Any]) -> dict[str, Any]:
    class_type = "FlashVSRNode"
    return {
        "1": {
            "class_type": "VHS_LoadVideo",
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
        "2": {"class_type": class_type, "inputs": flashvsr_node_inputs(args, class_type, "1", info)},
        "3": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["2", 0],
                "audio": ["1", 2],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": prefix,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 16,
                "save_metadata": True,
                "pingpong": False,
                "save_output": True,
            },
        },
    }


def flashvsr_run(args: argparse.Namespace, source: Path, partial: Path, output_width: int, output_height: int) -> Path:
    comfy_dir = resolve_path(args.comfy_dir)
    comfy_output_root = resolve_path(args.comfy_output_root) if args.comfy_output_root else comfy_dir / "output"
    if not (comfy_dir / "main.py").exists():
        raise FileNotFoundError(f"ComfyUI main.py not found: {comfy_dir / 'main.py'}")
    wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)
    required_nodes = {
        "VHS_LoadVideo": "ComfyUI-VideoHelperSuite",
        "VHS_VideoCombine": "ComfyUI-VideoHelperSuite",
        "FlashVSRNode": "ComfyUI-FlashVSR_Ultra_Fast",
    }
    ensure_node_types(args.comfy_url, required_nodes, "FlashVSR upscaling")
    info = object_info(args.comfy_url)
    video_name = copy_to_comfy_input(source, comfy_dir, "arp_upscale")
    fps = args.fps or float(video_info(source)["fps"])
    prefix = f"arp_upscale/{safe_stem(source.name)}_flashvsr_{output_width}x{output_height}"
    prompt = flashvsr_prompt(video_name, fps, args, prefix, info)
    prompt_id = queue_prompt(args.comfy_url, prompt)
    print(f"Queued ComfyUI prompt: {prompt_id}", flush=True)
    history = wait_for_prompt(args.comfy_url, prompt_id, args.poll_seconds)
    produced = newest_comfy_output(extract_output_files(history, comfy_output_root), {".mp4", ".mov", ".mkv", ".webm"}, "FlashVSR video")
    partial.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, partial)
    return partial


def fit_dimensions(source_width: int, source_height: int, target_width: int, target_height: int) -> tuple[int, int]:
    if target_width <= 0 and target_height <= 0:
        return source_width * 4, source_height * 4
    if target_width <= 0:
        target_width = round(target_height * source_width / source_height)
    if target_height <= 0:
        target_height = round(target_width * source_height / source_width)
    return max(2, target_width // 2 * 2), max(2, target_height // 2 * 2)


def scale_video(ffmpeg: str, source: Path, output: Path, width: int, height: int) -> None:
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vf",
        f"scale={width}:{height}:flags=lanczos,setsar=1",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)


def run(args: argparse.Namespace) -> int:
    source = resolve_path(args.input)
    if not source.exists():
        raise FileNotFoundError(f"Input video not found for upscaling: {source}")

    info = video_info(source)
    output_width, output_height = fit_dimensions(int(info["width"]), int(info["height"]), args.target_width, args.target_height)
    output = resolve_path(args.output) if args.output else default_output(source, output_width, output_height)
    sig = signature(args, source, output_width, output_height)

    if not args.force and resumable_output(output, sig, video_like=source, width=output_width, height=output_height):
        print(f"Reuse upscaled video: {output}", flush=True)
        return 0
    if args.dry_run:
        print(f"Would upscale {source} -> {output} using FlashVSR in ComfyUI at {args.comfy_url} ({args.comfy_dir})", flush=True)
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    raw_partial = output.with_suffix(output.suffix + ".flashvsr.partial" + output.suffix)
    final_partial = output.with_suffix(output.suffix + ".partial" + output.suffix)
    for path in (raw_partial, final_partial):
        if path.exists():
            path.unlink()

    print(f"Queueing FlashVSR in ComfyUI: {source}", flush=True)
    flashvsr_run(args, source, raw_partial, output_width, output_height)

    if not raw_partial.exists():
        raise RuntimeError(f"FlashVSR finished but did not create expected output: {raw_partial}")
    raw_info = video_info(raw_partial)
    if raw_info["width"] == output_width and raw_info["height"] == output_height:
        replace_with_retry(raw_partial, final_partial, "Upscaled preview")
    else:
        scale_video(find_ffmpeg(args.ffmpeg), raw_partial, final_partial, output_width, output_height)
        raw_partial.unlink(missing_ok=True)
    if output.exists():
        output.unlink()
    replace_with_retry(final_partial, output, "Upscaled output")
    write_signature(output, sig)
    print(f"Wrote upscaled video: {output}", flush=True)
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
