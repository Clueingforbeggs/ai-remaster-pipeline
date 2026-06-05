from __future__ import annotations

import base64
import io
import threading
from pathlib import Path

import numpy as np
from PIL import Image

_LOCK = threading.Lock()
_PREDICTOR = None
_MODEL_ID = "facebook/sam2.1-hiera-large"


def sam2_mask_for_image(image_path: Path, points: list[dict], width: int, height: int) -> dict[str, str]:
    width = max(1, int(width))
    height = max(1, int(height))
    predictor = _sam2_predictor()
    with Image.open(image_path) as img:
        image = np.array(img.convert("RGB"))

    positive_points = []
    negative_points = []
    image_h, image_w = image.shape[:2]
    for point in points:
        x = float(point.get("x", 0)) * image_w / max(1, width)
        y = float(point.get("y", 0)) * image_h / max(1, height)
        if str(point.get("label", "add")) == "subtract":
            negative_points.append([x, y])
        else:
            positive_points.append([x, y])

    if not positive_points:
        raise RuntimeError("SAM2 selection needs at least one positive point.")

    point_coords = np.array([*positive_points, *negative_points], dtype=np.float32)
    point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points), dtype=np.int32)

    predictor.set_image(image)
    masks, scores, _logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    best = int(np.argmax(scores)) if len(scores) else 0
    mask = (masks[best].astype(np.uint8) * 255)
    if image_w != width or image_h != height:
        resampling = getattr(Image, "Resampling", Image).NEAREST
        mask = np.array(Image.fromarray(mask).resize((width, height), resampling))

    buffer = io.BytesIO()
    Image.fromarray(mask).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "mask": f"data:image/png;base64,{encoded}",
        "provider": f"SAM 2.1 Hiera Large ({_MODEL_ID})",
    }


def _sam2_predictor():
    global _PREDICTOR
    with _LOCK:
        if _PREDICTOR is not None:
            return _PREDICTOR
        try:
            import torch
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError(
                "SAM2 is not installed. Re-run install_windows.bat or install the sam2 package in ARP's Python environment."
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # from_pretrained uses Hugging Face cache. First use may download the
        # SAM 2.1 Hiera Large weights; later masks reuse the cached checkpoint.
        predictor = SAM2ImagePredictor.from_pretrained(_MODEL_ID, device=device)
        _PREDICTOR = predictor
        return predictor
