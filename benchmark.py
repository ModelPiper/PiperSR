#!/usr/bin/env python3
"""
PiperSR Benchmark — Reproduce PSNR and FPS metrics.

First ANE-native SR model. Built by Ben Racicot.
https://modelpiper.com | https://github.com/BenRacicot/PiperSR

License: AGPL-3.0 (code), PiperSR Model License (weights)
Attribution required: "Powered by PiperSR from ModelPiper — https://modelpiper.com"
"""

import argparse
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image


MODEL_DIR = Path.home() / ".cache" / "pipersr"
MODEL_PATH = MODEL_DIR / "pipersr_4x.mlpackage"

# Set5 images — standard SR benchmark
SET5_NAMES = ["baby", "bird", "butterfly", "head", "woman"]


def load_model():
    model = ct.models.MLModel(
        str(MODEL_PATH),
        compute_units=ct.ComputeUnit.CPU_AND_NEURAL_ENGINE,
    )
    return model


def compute_psnr(sr, hr):
    """Compute PSNR between super-resolved and high-resolution images."""
    sr_arr = np.array(sr, dtype=np.float64)
    hr_arr = np.array(hr, dtype=np.float64)
    # Crop to match dimensions (standard SR eval practice)
    h = min(sr_arr.shape[0], hr_arr.shape[0])
    w = min(sr_arr.shape[1], hr_arr.shape[1])
    sr_arr = sr_arr[:h, :w]
    hr_arr = hr_arr[:h, :w]
    mse = np.mean((sr_arr - hr_arr) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


def benchmark_psnr(model, dataset_dir):
    """Evaluate PSNR on a dataset of LR/HR image pairs."""
    dataset_dir = Path(dataset_dir)
    lr_dir = dataset_dir / "LR"
    hr_dir = dataset_dir / "HR"

    if not lr_dir.exists() or not hr_dir.exists():
        print(f"Expected {lr_dir} and {hr_dir} directories.")
        print("Download Set5 from: https://github.com/jbhuang0604/SelfExSR")
        return

    psnrs = []
    for name in sorted(lr_dir.glob("*.png")):
        hr_path = hr_dir / name.name
        if not hr_path.exists():
            continue

        lr_img = Image.open(name).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        arr = np.array(lr_img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis]

        result = model.predict({"input": arr})
        key = list(result.keys())[0]
        sr_arr = result[key]
        if hasattr(sr_arr, "numpy"):
            sr_arr = sr_arr.numpy()
        sr_arr = np.squeeze(sr_arr)
        if sr_arr.shape[0] in (1, 3):
            sr_arr = np.transpose(sr_arr, (1, 2, 0))
        sr_img = Image.fromarray(np.clip(sr_arr * 255, 0, 255).astype(np.uint8))

        psnr = compute_psnr(sr_img, hr_img)
        psnrs.append(psnr)
        print(f"  {name.stem}: {psnr:.2f} dB")

    if psnrs:
        avg = np.mean(psnrs)
        print(f"\n  Average PSNR: {avg:.2f} dB")
    return psnrs


def benchmark_fps(model, resolution=(320, 240), warmup=10, iterations=100):
    """Measure inference FPS at a given resolution."""
    h, w = resolution
    dummy = np.random.rand(1, 3, h, w).astype(np.float32)

    print(f"\n  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        model.predict({"input": dummy})

    print(f"  Benchmarking ({iterations} iterations at {w}x{h})...")
    t0 = time.perf_counter()
    for _ in range(iterations):
        model.predict({"input": dummy})
    elapsed = time.perf_counter() - t0

    fps = iterations / elapsed
    latency = elapsed / iterations * 1000
    print(f"\n  FPS: {fps:.1f}")
    print(f"  Latency: {latency:.1f} ms")
    print(f"  Resolution: {w}x{h} → {w*4}x{h*4}")
    return fps, latency


def main():
    parser = argparse.ArgumentParser(
        description="PiperSR Benchmark — Reproduce PSNR and FPS"
    )
    parser.add_argument(
        "--dataset", "-d", default=None,
        help="Path to dataset directory with LR/ and HR/ subdirectories"
    )
    parser.add_argument(
        "--resolution", "-r", default="320x240",
        help="Resolution for FPS benchmark (WxH, default: 320x240)"
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=100,
        help="Number of iterations for FPS benchmark"
    )
    args = parser.parse_args()

    print("PiperSR Benchmark")
    print("=" * 50)
    print("https://modelpiper.com")
    print()

    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        print(f"Download from: https://modelpiper.com/models/pipersr")
        return

    model = load_model()
    print(f"Model loaded: {MODEL_PATH}")

    if args.dataset:
        print(f"\nPSNR Benchmark ({args.dataset}):")
        print("-" * 40)
        benchmark_psnr(model, args.dataset)

    w, h = (int(x) for x in args.resolution.split("x"))
    print(f"\nFPS Benchmark:")
    print("-" * 40)
    benchmark_fps(model, resolution=(h, w), iterations=args.iterations)

    print()
    print("Powered by PiperSR from ModelPiper — https://modelpiper.com")


if __name__ == "__main__":
    main()
