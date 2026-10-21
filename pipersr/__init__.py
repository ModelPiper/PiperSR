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

__version__ = "1.0.2"

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
        f"Download from: https://huggingface.co/ModelPiper/PiperSR-2x"
    )


def load_model():
    """Load the PiperSR CoreML model for ANE inference."""
    model_path = _find_model()
    return ct.models.MLModel(
        str(model_path),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )


def _upscale_tiles(model, img, overlap=8):
    """Upscale an RGB PIL image with a loaded model, tile by tile.

    Tiles overlap by `overlap` pixels on each side and only their centers are kept,
    so pixels near tile seams still see their neighbours.
    """
    spec = model.get_spec()
    in_name = spec.description.input[0].name
    out_name = spec.description.output[0].name
    tile = spec.description.input[0].type.imageType.width
    scale = spec.description.output[0].type.imageType.width // tile
    step = tile - 2 * overlap

    w, h = img.size
    nx = -(-w // step)
    ny = -(-h // step)
    # Mirror-pad the borders; black padding bleeds dark edges into the result
    canvas = Image.fromarray(np.pad(
        np.asarray(img),
        ((overlap, ny * step - h + overlap), (overlap, nx * step - w + overlap), (0, 0)),
        mode="reflect",
    ))
    out = Image.new("RGB", (nx * step * scale, ny * step * scale))
    m = overlap * scale

    for j in range(ny):
        for i in range(nx):
            x, y = i * step, j * step
            t = canvas.crop((x, y, x + tile, y + tile))
            r = model.predict({in_name: t})[out_name]
            if not isinstance(r, Image.Image):
                r = Image.fromarray(np.asarray(r))
            out.paste(r.crop((m, m, m + step * scale, m + step * scale)), (x * scale, y * scale))

    return out.crop((0, 0, w * scale, h * scale))


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
    img = Image.open(input_path).convert("RGB")
    w, h = img.size

    t0 = time.perf_counter()
    out_img = _upscale_tiles(model, img)
    elapsed = time.perf_counter() - t0

    print(f"PiperSR: {w}x{h} → {out_img.width}x{out_img.height} in {elapsed*1000:.1f}ms (ANE)")

    if output_path:
        out_img.save(output_path)
        print(f"Saved to {output_path}")

    return out_img
