from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

from common import ROOT, resolve_path

DEFAULT_MODEL = "gpt-image-2"
BOUNDARY = "----arp-openai-image-edit"


def multipart_field(name: str, value: str) -> bytes:
    return (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
        f"{value}\r\n"
    ).encode("utf-8")


def multipart_file(name: str, path: Path) -> bytes:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    header = (
        f"--{BOUNDARY}\r\n"
        f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8")
    return header + path.read_bytes() + b"\r\n"


def build_body(args: argparse.Namespace, source: Path) -> bytes:
    parts = [
        multipart_field("model", args.model),
        multipart_field("prompt", args.prompt),
        multipart_field("size", args.size),
        multipart_field("quality", args.quality),
        multipart_file("image", source),
        f"--{BOUNDARY}--\r\n".encode("utf-8"),
    ]
    return b"".join(parts)


def normalize_to_source_size(path: Path, source_path: Path) -> None:
    try:
        from PIL import Image
    except Exception:
        return
    with Image.open(source_path) as source_image:
        target_size = source_image.size
    with Image.open(path) as image:
        if image.size == target_size:
            return
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image.convert("RGB").resize(target_size, resampling).save(path, format="PNG")
    print(f"Normalized OpenAI output to source size: {path} ({target_size[0]}x{target_size[1]})", flush=True)


def generate(args: argparse.Namespace) -> Path:
    source = resolve_path(args.source_image)
    output = resolve_path(args.output)
    if not source.is_file():
        raise FileNotFoundError(f"Reference source image not found: {source}")
    if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise RuntimeError("OpenAI image edits require a PNG, JPEG, or WebP source image.")
    token = args.api_key.strip()
    if not token:
        raise RuntimeError("Missing OpenAI API key.")

    prompt = " ".join(part.strip() for part in (args.prompt, args.prompt_suffix, args.add_prompt) if part and part.strip())
    args.prompt = prompt
    body = build_body(args, source)
    request = urllib.request.Request(
        "https://api.openai.com/v1/images/edits",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={BOUNDARY}",
        },
        method="POST",
    )
    print(f"OpenAI image edit: {source} -> {output}", flush=True)
    print(f"OpenAI model: {args.model}", flush=True)
    if args.dry_run:
        return output
    try:
        with urllib.request.urlopen(request, timeout=args.timeout, context=ssl.create_default_context()) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {text}") from exc

    images = payload.get("data") or []
    if not images or not images[0].get("b64_json"):
        raise RuntimeError("OpenAI API response did not contain an image.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".partial")
    temp.write_bytes(base64.b64decode(images[0]["b64_json"]))
    if not args.no_normalize_to_source_size:
        normalize_to_source_size(temp, source)
    temp.replace(output)
    print(f"Wrote {output}", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colorize one reference image with the OpenAI Images API.")
    parser.add_argument("--source-image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--add-prompt", default="")
    parser.add_argument("--size", default="auto")
    parser.add_argument("--quality", default="auto")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--no-normalize-to-source-size", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    generate(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise
