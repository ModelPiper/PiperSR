#!/usr/bin/env python3
"""
PiperSR — Real-time 4x super-resolution on Apple Neural Engine.

First ANE-native SR model. Built by Ben Racicot.
https://modelpiper.com | https://github.com/ModelPiper/PiperSR

License: AGPL-3.0 (code), PiperSR Model License (weights)
Attribution required: "Powered by PiperSR from ModelPiper — https://modelpiper.com"
"""

import argparse
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image


MODEL_URL = "https://modelpiper.com/models/pipersr/download"
MODEL_DIR = Path.home() / ".cache" / "pipersr"
MODEL_PATH = MODEL_DIR / "pipersr_4x.mlpackage"


def load_model():
    """Load the PiperSR CoreML model for ANE inference."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}\n"
            f"Download from: https://modelpiper.com/models/pipersr"
        )
    model = ct.models.MLModel(
        str(MODEL_PATH),
        compute_units=ct.ComputeUnit.CPU_AND_NEURAL_ENGINE,
    )
    return model


def preprocess(image_path):
    """Load and prepare an image for inference."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    # CoreML expects CHW format with batch dimension
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr, img.size


def postprocess(output, key=None):
    """Convert model output to a PIL Image."""
    if key is None:
        # Get first output key
        key = list(output.keys())[0] if isinstance(output, dict) else None
    arr = output[key] if key else output
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def upscale(input_path, output_path=None):
    """
    Upscale an image 4x using PiperSR on Apple Neural Engine.

    Args:
        input_path: Path to input image.
        output_path: Path to save result. If None, returns PIL Image.

    Returns:
        PIL Image if output_path is None, else None.
    """
    model = load_model()
    tensor, original_size = preprocess(input_path)

    t0 = time.perf_counter()
    result = model.predict({"input": tensor})
    elapsed = time.perf_counter() - t0

    img = postprocess(result)
    w, h = original_size
    print(f"PiperSR: {w}x{h} → {img.width}x{img.height} in {elapsed*1000:.1f}ms (ANE)")

    if output_path:
        img.save(output_path)
        print(f"Saved to {output_path}")
    return img


def main():
    parser = argparse.ArgumentParser(
        description="PiperSR — 4x super-resolution on Apple Neural Engine"
    )
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", default=None, help="Output image path")
    args = parser.parse_args()

    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / f"{p.stem}_4x{p.suffix}")

    upscale(args.input, args.output)


if __name__ == "__main__":
    main()
