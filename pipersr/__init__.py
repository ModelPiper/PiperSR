"""
PiperSR — Real-time 2x super-resolution on Apple Neural Engine.

First ANE-native SR model. Built by Ben Racicot.
https://modelpiper.com | https://github.com/ModelPiper/PiperSR

Usage:
    from pipersr import upscale
    result = upscale("photo.png")
    result.save("photo_2x.png")

License: AGPL-3.0 (code), CC BY 4.0 (weights)
Attribution required: "Powered by PiperSR from ModelPiper — https://modelpiper.com"
"""

__version__ = "1.0.1"

import time
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image

MODEL_NAME = "PiperSR_2x.mlpackage"

# Search order: package-bundled → repo-local → user cache
_SEARCH_PATHS = [
    Path(__file__).parent / MODEL_NAME,
    Path(__file__).parent.parent / MODEL_NAME,
    Path.home() / ".cache" / "pipersr" / MODEL_NAME,
]


def _find_model():
    for p in _SEARCH_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"PiperSR model not found.\n"
        f"Searched: {', '.join(str(p) for p in _SEARCH_PATHS)}\n"
        f"Download from: https://modelpiper.com/models/pipersr"
    )


def load_model():
    """Load the PiperSR CoreML model for ANE inference."""
    model_path = _find_model()
    return ct.models.MLModel(
        str(model_path),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )


def upscale(input_path, output_path=None):
    """
    Upscale an image 2x using PiperSR on Apple Neural Engine.

    The model runs on fixed-size tiles; images of any size are split into tiles,
    upscaled on the ANE, and stitched back together.

    Args:
        input_path: Path to input image (str or Path).
        output_path: Path to save result. If None, returns PIL Image without saving.

    Returns:
        PIL Image of the upscaled result.
    """
    model = load_model()
    spec = model.get_spec()
    in_name = spec.description.input[0].name
    out_name = spec.description.output[0].name
    tile = spec.description.input[0].type.imageType.width
    scale = spec.description.output[0].type.imageType.width // tile

    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    pad_w = (tile - w % tile) % tile
    pad_h = (tile - h % tile) % tile
    canvas = Image.new("RGB", (w + pad_w, h + pad_h))
    canvas.paste(img, (0, 0))
    out = Image.new("RGB", (canvas.width * scale, canvas.height * scale))

    t0 = time.perf_counter()
    for y in range(0, canvas.height, tile):
        for x in range(0, canvas.width, tile):
            t = canvas.crop((x, y, x + tile, y + tile))
            r = model.predict({in_name: t})[out_name]
            if not isinstance(r, Image.Image):
                r = Image.fromarray(np.asarray(r))
            out.paste(r, (x * scale, y * scale))
    elapsed = time.perf_counter() - t0

    out_img = out.crop((0, 0, w * scale, h * scale))
    print(f"PiperSR: {w}x{h} → {out_img.width}x{out_img.height} in {elapsed*1000:.1f}ms (ANE)")

    if output_path:
        out_img.save(output_path)
        print(f"Saved to {output_path}")

    return out_img
