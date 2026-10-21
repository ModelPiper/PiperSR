#!/usr/bin/env python3
"""
PiperSR Benchmark — Reproduce PSNR and FPS metrics.

First ANE-native SR model. Built by Ben Racicot.
https://modelpiper.com | https://github.com/ModelPiper/PiperSR

License: AGPL-3.0 (code), CC BY 4.0 (weights)
Attribution required: "Powered by PiperSR from ModelPiper — https://modelpiper.com"
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from pipersr import _find_model, _upscale_tiles, load_model

SCALE = 2


def _luma(arr):
    """ITU-R BT.601 luma (Y channel), the standard for SR PSNR reporting."""
    arr = arr.astype(np.float64) / 255.0
    return 16.0 + 65.481 * arr[..., 0] + 128.553 * arr[..., 1] + 24.966 * arr[..., 2]


def compute_psnr(sr, hr, shave=SCALE):
    """PSNR on the Y channel with a `shave`-pixel border cropped (standard SR eval)."""
    sr_y = _luma(np.asarray(sr))
    hr_y = _luma(np.asarray(hr))
    h = min(sr_y.shape[0], hr_y.shape[0])
    w = min(sr_y.shape[1], hr_y.shape[1])
    sr_y = sr_y[shave:h - shave, shave:w - shave]
    hr_y = hr_y[shave:h - shave, shave:w - shave]
    mse = np.mean((sr_y - hr_y) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


def benchmark_psnr(model, dataset_dir):
    """Evaluate PSNR on a dataset of LR/HR image pairs (bicubic x2 LR)."""
    dataset_dir = Path(dataset_dir)
    lr_dir = dataset_dir / "LR"
    hr_dir = dataset_dir / "HR"

    if not lr_dir.exists() or not hr_dir.exists():
        print(f"Expected {lr_dir} and {hr_dir} directories.")
        print("Set5 (HR + bicubic x2 LR): https://huggingface.co/datasets/eugenesiow/Set5")
        return

    psnrs = []
    for lr_path in sorted(lr_dir.glob("*.png")):
        hr_path = hr_dir / lr_path.name
        if not hr_path.exists():
            continue

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        sr_img = _upscale_tiles(model, lr_img)

        psnr = compute_psnr(sr_img, hr_img)
        psnrs.append(psnr)
        print(f"  {lr_path.stem}: {psnr:.2f} dB")

    if psnrs:
        print(f"\n  Average PSNR (Y): {np.mean(psnrs):.2f} dB")
    return psnrs


def benchmark_fps(model, resolution=(360, 640), warmup=10, iterations=100):
    """Measure tiled frame throughput at a given resolution (H, W)."""
    h, w = resolution
    rng = np.random.default_rng(0)
    frame = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))

    print(f"\n  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        _upscale_tiles(model, frame)

    print(f"  Benchmarking ({iterations} iterations at {w}x{h})...")
    t0 = time.perf_counter()
    for _ in range(iterations):
        _upscale_tiles(model, frame)
    elapsed = time.perf_counter() - t0

    fps = iterations / elapsed
    latency = elapsed / iterations * 1000
    print(f"\n  FPS: {fps:.1f}")
    print(f"  Latency: {latency:.1f} ms/frame")
    print(f"  Resolution: {w}x{h} → {w*SCALE}x{h*SCALE}")
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
        "--resolution", "-r", default="640x360",
        help="Resolution for FPS benchmark (WxH, default: 640x360)"
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

    try:
        model = load_model()
    except FileNotFoundError as e:
        print(e)
        return
    print(f"Model loaded: {_find_model()}")

    if args.dataset:
        print(f"\nPSNR Benchmark ({args.dataset}):")
        print("-" * 40)
        benchmark_psnr(model, args.dataset)

    w, h = (int(x) for x in args.resolution.split("x"))
    print(f"\nFPS Benchmark (128px tiles, Python):")
    print("-" * 40)
    benchmark_fps(model, resolution=(h, w), iterations=args.iterations)

    print()
    print("Powered by PiperSR from ModelPiper — https://modelpiper.com")


if __name__ == "__main__":
    main()
