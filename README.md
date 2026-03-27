# PiperSR

**First super-resolution model designed and optimized for Apple Neural Engine.**

2x upscale. 453K parameters. 928 KB. Real-time video at 48 FPS. Runs entirely on the ANE in every Mac, iPhone, and iPad — zero GPU, zero cloud.

---

## At a Glance

| | PiperSR |
|---|---|
| **Parameters** | 453,000 |
| **Model size** | 928 KB (CoreML FP16) |
| **Scale factor** | 2x |
| **PSNR (Set5)** | 37.54 dB |
| **PSNR (Set14)** | 33.21 dB |
| **PSNR (BSD100)** | 31.98 dB |
| **PSNR (Urban100)** | 31.38 dB |
| **FPS (360p → 720p)** | 48 FPS (M2 Max) |
| **ANE latency** | 20.8 ms/frame |
| **Compute** | Apple Neural Engine only |
| **CPU/GPU fallback ops** | Zero |
| **Precision** | FP16 |
| **Training cost** | ~$6 (RunPod A6000) |

---

## How It Compares

### Quality vs. Parameter Count (2x upscale)

Models sorted by parameter count. PSNR measured on standard academic benchmarks (bicubic degradation).

| Model | Params | Size | Set5 | Set14 | BSD100 | Urban100 | Year |
|---|---:|---:|---:|---:|---:|---:|---|
| FSRCNN | 12K | ~50 KB | 36.94 | 32.54 | 31.73 | — | 2016 |
| ESPCN | ~20K | ~100 KB | ~36.7 | ~34.5 | ~34.3 | — | 2016 |
| **SPAN-S** | **426K** | **~1.7 MB** | **38.06** | **33.73** | **32.21** | **32.20** | **2024** |
| **PiperSR** | **453K** | **928 KB** | **37.54** | **33.21** | **31.98** | **31.38** | **2026** |
| SPAN | 498K | ~2 MB | 38.08 | 33.71 | 32.22 | 32.24 | 2024 |
| OmniSR | 792K | 3.3 MB | 38.12 | 33.70 | 32.22 | 32.38 | 2023 |
| SwinIR-light | 878K | ~3.5 MB | 38.14 | 33.86 | 32.31 | 32.76 | 2021 |
| EDSR-baseline | 1.5M | ~6 MB | 38.02 | 33.57 | 32.12 | — | 2017 |
| SwinIR | 11.9M | ~48 MB | 38.42 | 34.46 | 32.53 | 33.81 | 2021 |
| Real-ESRGAN | 16.7M | 64 MB | — | — | — | — | 2021 |
| HAT | 20.8M | ~80 MB | 38.63 | 34.86 | 32.62 | 34.45 | 2023 |

> Real-ESRGAN is GAN-trained for perceptual quality on real-world degradations, not bicubic benchmarks — PSNR comparison is not applicable.

### Size Comparison

```
PiperSR     928 KB  ██
SPAN-S     ~1.7 MB  ███
SPAN       ~2.0 MB  ████
OmniSR      3.3 MB  ██████
SwinIR-lt  ~3.5 MB  ███████
EDSR-base  ~6.0 MB  ████████████
SwinIR     ~48  MB  ████████████████████████████████████████████████████████████████████████████████████████████████
Real-ESRGAN  64 MB  █████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
```

PiperSR is **69x smaller** than Real-ESRGAN and **4x smaller** than SwinIR-light.

### Speed: Apple Silicon (2x upscale, 360p → 720p)

No other model in this parameter class was designed for ANE. These are real measurements, not theoretical throughput.

| Model | Hardware | FPS | Compute | Real-time? |
|---|---|---:|---|---|
| **PiperSR** | **M2 Max** | **48** | **ANE** | **Yes (1.6x real-time)** |
| SPAN | M2 Mac (ailia) | ~7 | GPU | No |
| Real-ESRGAN | M2 Mac (ailia) | ~0.3 | GPU | No |

> SPAN and Real-ESRGAN measurements from ailia SDK M2 Mac benchmarks. PiperSR is **~7x faster** than SPAN and **~160x faster** than Real-ESRGAN on the same hardware family.

### The Tradeoff

PiperSR is **0.5 dB below SPAN** on Set5 (37.54 vs 38.06). That's the cost of building for ANE instead of GPU. Here's what you get in return:

| | PiperSR | SPAN-S |
|---|---|---|
| Set5 PSNR | 37.54 dB | 38.06 dB |
| Model size | 928 KB | ~1.7 MB |
| Apple Silicon FPS | 48 (ANE) | ~7 (GPU) |
| Real-time video | Yes | No |
| GPU usage | 0% | 100% |
| Runs on iPhone/iPad | Yes (ANE) | Requires GPU |

0.5 dB is below the perceptual threshold for most content. 48 FPS vs ~7 FPS is the difference between real-time video and a slideshow.

---

## Quick Start

[![PyPI](https://img.shields.io/pypi/v/pipersr)](https://pypi.org/project/pipersr/)

```bash
pip install pipersr
```

**CLI:**

```bash
pipersr -i photo.png -o photo_2x.png
```

**Python:**

```python
from pipersr import upscale

result = upscale("photo.png")
result.save("photo_2x.png")
```

Model weights (928 KB) are bundled in the package. No downloads, no API keys, no accounts.

## Download

The CoreML model (`PiperSR_2x.mlpackage`, 928 KB) is included in this repo. Clone and run — no separate download needed.

Also available on [ModelPiper.com](https://modelpiper.com/models/pipersr).

---

## Architecture

```
Input (H×W×3, FP16)
  │
  ├─ Head: Conv 3×3 (3 → 64 channels)
  │
  ├─ Body: 6 Residual Blocks
  │    ┌──────────────────────────────┐
  │    │  Conv 3×3 (64ch)             │
  │    │  BatchNorm                    │
  │    │  SiLU                         │
  │    │  Conv 3×3 (64ch)             │
  │    │  BatchNorm                    │
  │    │  + Residual                   │
  │    └──────────────────────────────┘
  │
  ├─ Tail: Conv 3×3 (64 → 12ch) → PixelShuffle(2)
  │
  Output (2H×2W×3, FP16)
```

**453K parameters. 5 unique MIL ops.** Every operation — `conv`, `batch_norm`, `silu`, `add`, `pixel_shuffle` — is native to the Neural Engine instruction set. Zero ops fall back to CPU or GPU.

### Why These Choices

| Decision | Why |
|---|---|
| **BatchNorm** (not RMSNorm) | RMSNorm decomposes into 4 ops that ALL fall back to CPU. Switching to BatchNorm gave a **2.5x throughput increase** — the single biggest optimization in the project |
| **SiLU** (not SwiGLU) | Single ANE-native op. SwiGLU adds complexity for marginal gain at this model scale |
| **No attention** | At 128×128 tiles, attention = 16K tokens. Adds compute without clear quality benefit for a conv SR model at 453K params |
| **PixelShuffle** (not transposed conv) | Deterministic reshape, no checkerboard artifacts, maps to simple memory ops on ANE |
| **64 channels** | ANE tile-aligned. 48ch (used by SPAN) wastes ~25% of each ANE tile on padding |
| **`.cpuAndNeuralEngine`** (not `.all`) | `.all` is 23.6% slower — CoreML silently misroutes pure-ANE ops onto the GPU |

### Video Pipeline (Double-Buffered)

For real-time video, PiperSR uses a double-buffered pipeline where each hardware unit works on a different frame simultaneously:

```
Frame N:    [CPU convertIn] → [ANE predict: 20.8ms] → [Metal GPU convertOut]
Frame N+1:                    [CPU convertIn] ─────── → [ANE predict: 20.8ms] → ...
```

| Resolution | Input → Output | ANE Predict | Streaming FPS | Real-time? |
|---|---|---:|---:|---|
| 360p | 640×360 → 1280×720 | 20.8 ms | 48 FPS | 1.6x real-time |
| 480p | 854×480 → 1708×960 | 32.7 ms | 30 FPS | 1.0x real-time |
| 720p | 1280×720 → 2560×1440 | 71.6 ms | 14 FPS | No |

Available in [ToolPiper](https://toolpiper.com) for real-time video upscaling on macOS.

---

## Benchmarks

Reproducible. Run `python benchmark.py` on your own hardware.

```bash
# PSNR evaluation (requires Set5 dataset with LR/ and HR/ directories)
python benchmark.py --dataset /path/to/Set5

# FPS benchmark at 360p
python benchmark.py --resolution 640x360 --iterations 200
```

### Full PSNR Results

| Dataset | PiperSR | Bicubic | Delta |
|---|---:|---:|---:|
| Set5 | 37.54 dB | 33.66 dB | +3.88 dB |
| Set14 | 33.21 dB | 30.24 dB | +2.97 dB |
| BSD100 | 31.98 dB | 29.56 dB | +2.42 dB |
| Urban100 | 31.38 dB | 26.88 dB | +4.50 dB |

### Inference Speed by Hardware

| Hardware | Mode | FPS | Latency |
|---|---|---:|---:|
| M2 Max | Full-frame 360p | 48.0 | 20.8 ms |
| M2 Max | Full-frame 480p | 30.0 | 32.7 ms |
| M2 Max | Full-frame 720p | 14.0 | 71.6 ms |
| M2 | Tiled 128×128 (static weights) | 125.6 | 7.96 ms |
| M2 | Tiled 128×128 (dynamic weights) | 44.5 | 22.5 ms |

> Static (baked) weights are 2.82x faster than dynamic — the main bottleneck is weight loading, not ANE compute.

---

## Why ANE?

Every Apple Silicon chip has a Neural Engine — a dedicated inference accelerator that most ML models completely ignore. The entire ML ecosystem targets CUDA. Models get "converted" to CoreML as an afterthought, and the conversion is usually bad: misaligned tensor shapes cause pipeline stalls, unsupported ops silently fall back to CPU, and full-frame predictions can't exploit ANE parallelism.

PiperSR was designed from scratch for ANE, not converted from a GPU model. Every dimension, every operation, every data type was chosen based on measured ANE hardware characteristics. The result: real-time super-resolution that uses **zero GPU**. Your GPU stays free for rendering, compositing, or whatever else you're doing.

## Supported Hardware

Any Apple Silicon device:

| Platform | Chips |
|---|---|
| **Mac** | M1, M1 Pro/Max/Ultra through M5 |
| **iPhone** | A15+ |
| **iPad** | M1+ |

Performance scales with ANE generation. M2 Max benchmarks shown above.

## For Video & Real-Time

PiperSR is an image model. For real-time video super-resolution with double-buffered frame scheduling, Metal GPU output conversion, and streaming — see [ToolPiper](https://toolpiper.com), which integrates PiperSR into a production video pipeline.

---

## Model Details

- **Task:** Single-image 2x super-resolution
- **Architecture:** 6-block residual CNN, 64 channels, BatchNorm + SiLU, PixelShuffle upsampling
- **Input:** RGB image, any resolution (CoreML flexible input)
- **Output:** 2x upscaled RGB image, FP16
- **Format:** CoreML .mlpackage
- **Compute:** Apple Neural Engine via `.cpuAndNeuralEngine`
- **Training data:** Standard SR datasets (DIV2K, Flickr2K)
- **Training cost:** ~$6 total on RunPod A6000 instances

For the full model card, see [MODEL_CARD.md](MODEL_CARD.md).

## License

**Free for everyone** — personal, academic, and commercial. Just include a link back:

```
Powered by PiperSR from ModelPiper — https://modelpiper.com
```

Use it in your app, your product, your research. We ask for attribution so others can find the project.

- Code: [AGPL-3.0](LICENSE)
- Model weights: [PiperSR Model License](MODEL_LICENSE)

## Links

- [ModelPiper.com](https://modelpiper.com) — Download PiperSR, browse benchmarks, explore on-device models
- [ToolPiper](https://toolpiper.com) — Local macOS AI toolkit with real-time video upscaling
- [Ben Racicot](https://github.com/BenRacicot) — Author

## Citation

```bibtex
@software{pipersr2026,
  author = {Racicot, Ben},
  title = {PiperSR: Real-Time Super-Resolution for Apple Neural Engine},
  year = {2026},
  url = {https://github.com/ModelPiper/PiperSR},
  note = {453K parameters, 928 KB CoreML. Available at https://modelpiper.com}
}
```
