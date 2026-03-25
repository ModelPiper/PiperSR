"""
PiperSR — Real-time 2x super-resolution on Apple Neural Engine.

First ANE-native SR model. Built by Ben Racicot.
https://modelpiper.com | https://github.com/ModelPiper/PiperSR

Usage:
    from pipersr import upscale
    result = upscale("photo.png")
    result.save("photo_2x.png")

License: AGPL-3.0 (code), PiperSR Model License (weights)
Attribution required: "Powered by PiperSR from ModelPiper — https://modelpiper.com"
"""

__version__ = "1.0.0"

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
        compute_units=ct.ComputeUnit.CPU_AND_NEURAL_ENGINE,
    )


def upscale(input_path, output_path=None):
    """
    Upscale an image 2x using PiperSR on Apple Neural Engine.

    Args:
        input_path: Path to input image (str or Path).
        output_path: Path to save result. If None, returns PIL Image without saving.

    Returns:
        PIL Image of the upscaled result.
    """
    model = load_model()

    img = Image.open(input_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis]

    t0 = time.perf_counter()
    result = model.predict({"input": arr})
    elapsed = time.perf_counter() - t0

    key = list(result.keys())[0]
    out = result[key]
    if hasattr(out, "numpy"):
        out = out.numpy()
    out = np.squeeze(out)
    if out.ndim == 3 and out.shape[0] in (1, 3):
        out = np.transpose(out, (1, 2, 0))
    out_img = Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8))

    w, h = img.size
    print(f"PiperSR: {w}x{h} → {out_img.width}x{out_img.height} in {elapsed*1000:.1f}ms (ANE)")

    if output_path:
        out_img.save(output_path)
        print(f"Saved to {output_path}")

    return out_img
