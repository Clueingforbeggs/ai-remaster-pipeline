from __future__ import annotations

from pathlib import Path
from typing import Any


def guide_output_size_for_prepared(prepared_canvas: Path, canvas_w: int, canvas_h: int) -> tuple[int, int]:
    """Return the guide image size to store/use after Qwen.

    Guides should stay on the same model-safe canvas as the prepared video. LTX expects
    factor-of-32-friendly dimensions, so a 480p source-height workflow uses 864x480 and the
    720p delivery workflow keeps its established 1280x704 work canvas.
    """
    return canvas_w, canvas_h


def resize_frame_for_qwen(prepared_frame: Any, prepared_canvas: Path) -> Any:
    """Resize a prepared OpenCV frame to the guide/Qwen size for this canvas."""
    height, width = prepared_frame.shape[:2]
    guide_w, guide_h = guide_output_size_for_prepared(prepared_canvas, width, height)
    if (guide_w, guide_h) == (width, height):
        return prepared_frame
    import cv2

    return cv2.resize(prepared_frame, (guide_w, guide_h), interpolation=cv2.INTER_LANCZOS4)


def _edge_bbox_from_rgb_pixels(width: int, height: int, get_pixel, threshold: int = 4) -> tuple[int, int, int, int] | None:
    xs: list[int] = []
    ys: list[int] = []
    for y in range(height):
        for x in range(width):
            r, g, b = get_pixel(x, y)
            if r > threshold or g > threshold or b > threshold:
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        return None
    return min(xs), max(xs), min(ys), max(ys)


DEFAULT_EDGE_MASK_OVERLAP_PX = 6


def _save_edge_mask(width: int, height: int, bbox: tuple[int, int, int, int] | None, target: Path, overlap_px: int = DEFAULT_EDGE_MASK_OVERLAP_PX) -> Path:
    from PIL import Image as PILImage

    mask = PILImage.new("L", (width, height), 0)
    mask_pixels = mask.load()
    if bbox is None:
        for y in range(height):
            for x in range(width):
                mask_pixels[x, y] = 255
    else:
        left, right, top, bottom = bbox
        mask_left = left > 0
        mask_right = right < width - 1
        mask_top = top > 0
        mask_bottom = bottom < height - 1
        if not (mask_left or mask_right or mask_top or mask_bottom):
            for y in range(height):
                for x in range(width):
                    mask_pixels[x, y] = 255
        else:
            overlap = max(0, int(overlap_px))
            for y in range(height):
                for x in range(width):
                    edge = (
                        (mask_left and x < min(width, left + overlap))
                        or (mask_right and x > max(-1, right - overlap))
                        or (mask_top and y < min(height, top + overlap))
                        or (mask_bottom and y > max(-1, bottom - overlap))
                    )
                    if edge:
                        mask_pixels[x, y] = 255
    target.parent.mkdir(parents=True, exist_ok=True)
    mask.save(target, format="PNG")
    return target


def save_edge_mask_for_image(source: Path, target: Path, overlap_px: int = DEFAULT_EDGE_MASK_OVERLAP_PX) -> Path:
    """Create a mask for exact-black prepared-canvas edges with a small inward overlap."""
    from PIL import Image as PILImage

    with PILImage.open(source) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        pixels = rgb.load()
        bbox = _edge_bbox_from_rgb_pixels(width, height, lambda x, y: pixels[x, y])
    return _save_edge_mask(width, height, bbox, target, overlap_px)


def save_edge_mask_for_frame(frame: Any, target: Path, overlap_px: int = DEFAULT_EDGE_MASK_OVERLAP_PX) -> Path:
    """Create an edge mask for an OpenCV BGR frame."""
    height, width = frame.shape[:2]
    bbox = _edge_bbox_from_rgb_pixels(width, height, lambda x, y: frame[y, x])
    return _save_edge_mask(width, height, bbox, target, overlap_px)
