from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import qwen_colorize_references as qwen
from comfy_api import extract_output_files, node_by_id, queue_prompt, set_widget, wait_for_comfy, wait_for_prompt, workflow_to_prompt
from common import ROOT, copy_to_comfy_input, file_fingerprint, newest_output, resolve_path, root_relative, resumable_output, write_signature
from dependency_manager import ensure_qwen_image_edit_models


def iter_nodes(workflow: dict[str, Any]):
    yield from qwen.iter_workflow_nodes(workflow)


def node_text(node: dict[str, Any]) -> str:
    parts = [str(node.get("id", "")), str(node.get("type", "")), str(node.get("class_type", "")), str(node.get("title", ""))]
    parts.extend(str(value) for value in qwen.node_widget_values(node))
    return " ".join(parts).lower()


def resolve_mask_node_id(workflow: dict[str, Any], explicit: str | None) -> str:
    if explicit and explicit.lower() != "auto":
        return explicit
    candidates = []
    for node in iter_nodes(workflow):
        class_type = node.get("class_type") or node.get("type")
        if class_type not in {"LoadImage", "LoadImageMask"}:
            continue
        text = node_text(node)
        if "mask" in text:
            candidates.append(node)
    if candidates:
        return str(candidates[0].get("id"))
    raise RuntimeError(
        "The masked Qwen workflow does not expose a mask LoadImage/LoadImageMask node. "
        "Set Reference settings -> Masked edit workflow to a Qwen inpaint workflow with a visible mask image loader."
    )


def resolve_source_node_id(workflow: dict[str, Any], explicit: str | None) -> str:
    if explicit and explicit.lower() != "auto":
        return explicit
    candidates = []
    for node in iter_nodes(workflow):
        class_type = node.get("class_type") or node.get("type")
        if class_type != "LoadImage":
            continue
        text = node_text(node)
        if "mask" not in text:
            candidates.append(node)
    if candidates:
        return str(candidates[0].get("id"))
    return qwen.resolve_node_id(workflow, explicit, {"LoadImage"})


def patch_masked_workflow(args: argparse.Namespace, workflow: dict[str, Any], source_path: Path, mask_path: Path, output_path: Path, prompt: str) -> dict[str, Any]:
    if qwen.has_frontend_subgraphs(workflow):
        raise RuntimeError("Masked edits currently require a workflow with visible top-level image and mask loader nodes.")
    comfy_dir = resolve_path(args.comfy_dir)
    comfy_image = copy_to_comfy_input(source_path, comfy_dir, "arp_qwen_ref_edits")
    comfy_mask = copy_to_comfy_input(mask_path, comfy_dir, "arp_qwen_ref_masks")
    load_id = resolve_source_node_id(workflow, args.load_image_node_id)
    mask_id = resolve_mask_node_id(workflow, args.mask_image_node_id)
    save_id = qwen.resolve_node_id(workflow, args.save_node_id, {"SaveImage"})
    prompt_id = qwen.resolve_node_id(workflow, args.prompt_node_id, {"TextEncodeQwenImageEditPlus", "CLIPTextEncode"}, prefer_title="positive")
    set_widget(node_by_id(workflow, load_id), args.load_image_widget, comfy_image)
    set_widget(node_by_id(workflow, mask_id), args.mask_image_widget, comfy_mask)
    set_widget(node_by_id(workflow, prompt_id), args.prompt_widget, prompt)
    save_node = node_by_id(workflow, save_id)
    prefix = str(Path("ai_remaster_qwen_edits") / output_path.stem).replace("\\", "/")
    set_widget(save_node, args.save_prefix_widget, prefix)
    print(f"Patched masked workflow nodes: source={load_id}, mask={mask_id}, prompt={prompt_id}, save={save_id}", flush=True)
    return workflow_to_prompt(workflow, save_id)


def normalize_to_source_size(path: Path, source_path: Path, final_path: Path | None = None) -> None:
    qwen.normalize_to_source_size(path, source_path, final_path)


def signature(args: argparse.Namespace, workflow_path: Path, source_path: Path, mask_path: Path, prompt: str) -> dict[str, Any]:
    return {
        "version": 1,
        "tool": "edit_reference_image.py",
        "source": root_relative(source_path),
        "source_fingerprint": file_fingerprint(source_path),
        "mask": root_relative(mask_path),
        "mask_fingerprint": file_fingerprint(mask_path),
        "workflow": root_relative(workflow_path),
        "workflow_fingerprint": file_fingerprint(workflow_path),
        "prompt": prompt,
        "model_backend": args.model_backend,
        "gguf_model": args.gguf_model if args.model_backend == "gguf" else None,
        "normalize_to_source_size": not args.no_normalize_to_source_size,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Edit one colour reference image with a masked Qwen Image Edit workflow.")
    parser.add_argument("--source-image", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workflow", required=True, type=Path)
    parser.add_argument("--instruction", required=True)
    config = qwen.load_local_config()
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-dir", type=Path, default=Path(config.get("comfy_dir", ROOT / "tools" / "comfyui")))
    parser.add_argument("--comfy-output-root", type=Path, default=ROOT / "tools" / "comfyui" / "output")
    parser.add_argument("--model-backend", choices=["gguf", "safetensors"], default="gguf")
    parser.add_argument("--gguf-model", default="qwen-image-edit-2511-Q4_K_M.gguf")
    parser.add_argument("--load-image-node-id", default="auto")
    parser.add_argument("--load-image-widget", default="0")
    parser.add_argument("--prompt-node-id")
    parser.add_argument("--prompt-widget", default="0")
    parser.add_argument("--save-node-id", default="auto")
    parser.add_argument("--save-prefix-widget", default="0")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--no-normalize-to-source-size", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--print-api-prompt", action="store_true")
    parser.add_argument("--mask-image-node-id", default="auto")
    parser.add_argument("--mask-image-widget", default="0")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = resolve_path(args.source_image)
    mask = resolve_path(args.mask)
    output = resolve_path(args.output)
    workflow_path = resolve_path(args.workflow)
    if not source.is_file():
        raise FileNotFoundError(f"Reference source image not found: {source}")
    if not mask.is_file():
        raise FileNotFoundError(f"Reference mask image not found: {mask}")
    sig = signature(args, workflow_path, source, mask, args.instruction)
    if not args.force and resumable_output(output, sig, image_like=source):
        print(f"Reuse masked Qwen reference edit: {output}", flush=True)
        return 0
    if not args.dry_run:
        comfy_dir = resolve_path(args.comfy_dir)
        if args.model_backend == "gguf" and not (comfy_dir / "custom_nodes" / "ComfyUI-GGUF").exists():
            raise FileNotFoundError(f"ComfyUI-GGUF is required for Qwen GGUF: {comfy_dir / 'custom_nodes' / 'ComfyUI-GGUF'}")
        ensure_qwen_image_edit_models(comfy_dir)
        print(f"Waiting for ComfyUI at {args.comfy_url}...", flush=True)
        wait_for_comfy(args.comfy_url, timeout_seconds=180, poll_seconds=args.poll_seconds)
    print(f"Masked Qwen reference edit: {source} + {mask} -> {output}", flush=True)
    print(f"Qwen edit prompt: {args.instruction}", flush=True)
    workflow = qwen.load_workflow(workflow_path)
    qwen.patch_qwen_model_backend(args, workflow)
    prompt_payload = patch_masked_workflow(args, workflow, source, mask, output, args.instruction)
    if args.print_api_prompt:
        import json

        print(json.dumps(prompt_payload, indent=2), flush=True)
    if args.dry_run:
        return 0
    prompt_id = queue_prompt(args.comfy_url, prompt_payload)
    print(f"Queued ComfyUI prompt: {prompt_id}", flush=True)
    history = wait_for_prompt(args.comfy_url, prompt_id, args.poll_seconds)
    produced = newest_output(extract_output_files(history, resolve_path(args.comfy_output_root)))
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".partial")
    shutil.copy2(produced, tmp)
    if not args.no_normalize_to_source_size:
        normalize_to_source_size(tmp, source, output)
    tmp.replace(output)
    write_signature(output, sig)
    print(f"Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
