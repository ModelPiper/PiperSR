# PiperSR

**First super-resolution model designed and optimized for Apple Neural Engine.**

2x upscale. 453K parameters. 928 KB. Real-time 360p → 720p video at 44.4 FPS. Runs entirely on the ANE in every Mac, iPhone, and iPad — zero GPU, zero cloud.

## Get PiperSR

- **ToolPiper Studio (recommended):** the main way to use PiperSR. A local macOS AI toolkit with real-time 360p → 720p video upscaling, at [ModelPiper.com](https://modelpiper.com).
- **[Hugging Face](https://huggingface.co/ModelPiper/PiperSR-2x):** CoreML packages, plus 256px and 720p video variants.
- **[PyPI](https://pypi.org/project/pipersr/):** `pip install pipersr`, then `pipersr -i photo.png`. See [Quick Start](#quick-start).

---

## At a Glance

| | PiperSR |
|---|---|
| **Parameters** | 453,000 |
| **Model size** | 928 KB (CoreML FP16) |
| **Scale factor** | 2x |
| **PSNR (Set5)** | 37.54 dB |
| **PSNR (Set14)** | 33.16 dB |
| **PSNR (BSD100)** | 31.99 dB |
| **FPS (360p → 720p)** | 44.4 FPS (M2 Max) |
| **ANE latency** | 22.5 ms/frame |
| **Compute** | Apple Neural Engine only |
| **CPU/GPU fallback ops** | Zero |
| **Precision** | FP16 |
| **Training cost** | ~$6 (RunPod A6000) |

---

## How It Compares

### Quality vs. Parameter Count (2x upscale)

Models sorted by parameter count (2x models). Y-channel PSNR in dB on standard benchmarks with bicubic degradation, taken from each model's paper.

| Model | Params | Set5 | Set14 | BSD100 | Urban100 | Year | Paper |
|---|---:|---:|---:|---:|---:|---|---|
| FSRCNN | 12K | 37.00 | 32.63 | 31.53 | 29.88 | 2016 | [ECCV 2016](https://arxiv.org/abs/1608.00367) |
| ESPCN | 21K | 36.83 | 32.40 | 31.29 | 29.48 | 2016 | [CVPR 2016](https://arxiv.org/abs/1609.05158) |
| SAFMN | 228K | 38.00 | 33.54 | 32.16 | 31.84 | 2023 | [ICCV 2023](https://arxiv.org/abs/2302.13800) |
| BSRN | 332K | 38.10 | 33.74 | 32.24 | 32.34 | 2022 | [CVPRW 2022](https://arxiv.org/abs/2205.05996) |
| ShuffleMixer | 394K | 38.01 | 33.63 | 32.17 | 31.89 | 2022 | [NeurIPS 2022](https://arxiv.org/abs/2205.15175) |
| SPAN-S | 411K | 38.06 | 33.73 | 32.21 | 32.20 | 2024 | [CVPRW 2024](https://arxiv.org/abs/2311.12770) |
| **PiperSR** | **453K** | **37.54** | **33.16** | **31.99** | **—** | **2026** | this repo |
| SPAN | 481K | 38.08 | 33.71 | 32.22 | 32.24 | 2024 | [CVPRW 2024](https://arxiv.org/abs/2311.12770) |
| Omni-SR | 772K | 38.22 | 33.98 | 32.36 | 33.05 | 2023 | [CVPR 2023](https://arxiv.org/abs/2304.10244) |
| SwinIR-light | 878K | 38.14 | 33.86 | 32.31 | 32.76 | 2021 | [ICCVW 2021](https://arxiv.org/abs/2108.10257) |
| EDSR-baseline | 1.37M | 37.99 | 33.57 | 32.16 | 31.98 | 2017 | [CVPRW 2017](https://arxiv.org/abs/1707.02921) |
| SwinIR | 11.8M | 38.42 | 34.46 | 32.53 | 33.81 | 2021 | [ICCVW 2021](https://arxiv.org/abs/2108.10257) |
| Real-ESRGAN | 16.7M | — | — | — | — | 2021 | [ICCVW 2021](https://arxiv.org/abs/2107.10833) |
| HAT | 20.6M | 38.63 | 34.86 | 32.62 | 34.45 | 2023 | [CVPR 2023](https://arxiv.org/abs/2205.04437) |

> Exceptions to "from each paper": FSRCNN's BSD100 and Urban100 scores are from [CARN](https://arxiv.org/abs/1803.08664) (Table 1). ESPCN's paper has no 2x results, so its row is the retrain cited by ShuffleMixer and SAFMN. EDSR-baseline's row is from [IMDN](https://arxiv.org/abs/1909.11856) (Table 5). Models were trained on DIV2K or DF2K; see each paper.
>
> Real-ESRGAN is GAN-trained for perceptual quality on real-world degradations, not bicubic benchmarks. Its paper reports no PSNR, so a PSNR comparison is not applicable.
>
> PiperSR has no Urban100 score: its 100 test images were accidentally included in PiperSR's training data, so a score on them would not be a fair test.

### Size

PiperSR has 453K parameters: about half as many as SwinIR-light and 37x fewer than Real-ESRGAN. The whole CoreML package is 928 KB at FP16.

### Speed: Apple Silicon

No other model in this parameter class was designed for ANE. These are real measurements, not theoretical throughput.

| Model | Task | Hardware | FPS | Compute | Real-time? |
|---|---|---|---:|---|---|
| **PiperSR** | **2x, 640×360 → 1280×720** | **M2 Max** | **44.4** | **ANE** | **Yes (1.5x real-time)** |
| SPAN-S | 4x, 510×339 input | M2 Mac (ailia) | ~7 | GPU | No |
| Real-ESRGAN | 3.5x, 510×339 input | M2 Mac (ailia) | ~0.3 | GPU | No |

> SPAN-S and Real-ESRGAN timings are from ailia's [SPAN write-up](https://tech.ailia.ai/en/span-efficient-super-resolution-model-using-parameter-free-attention-d938764aa208/) (139 ms and 2,989 ms per image on an M2 Mac GPU via MPS). Those runs use a different scale and input size, so treat this as a rough comparison rather than a like-for-like benchmark.

### The Tradeoff

PiperSR is **0.5 dB below SPAN-S** on Set5 (37.54 vs 38.06). That's the cost of building for ANE instead of GPU. Here's what you get in return:

| | PiperSR | SPAN-S |
|---|---|---|
| Set5 PSNR | 37.54 dB | 38.06 dB |
| Parameters | 453K | 411K |
| Apple Silicon FPS | 44.4 (ANE) | ~7 (GPU, see above) |
| Real-time video | Yes | No |
| GPU usage | 0% | 100% |
| Runs on iPhone/iPad | Yes (ANE) | Requires GPU |

0.5 dB is below the perceptual threshold for most content. 44.4 FPS vs ~7 FPS is the difference between real-time video and a slideshow.

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

The CoreML model (`PiperSR_2x.mlpackage`, 928 KB) is included in this repo. Clone and run. No separate download needed.

Also available on [Hugging Face](https://huggingface.co/ModelPiper/PiperSR-2x), along with 256px and 720p video variants.

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

**453K parameters. 5 network ops.** The compiled model is just `conv`, `silu`, `add`, `pixel_shuffle` and `clip` (BatchNorm is fused into the convolutions at export). CoreML's compute plan places every one of them on the Neural Engine. Only the input image scaling runs on CPU.

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
Frame N:    [CPU convertIn] → [ANE predict: 22.5ms] → [Metal GPU convertOut]
Frame N+1:                    [CPU convertIn] ─────── → [ANE predict: 22.5ms] → ...
```

| Resolution | Input → Output | ANE Predict | Streaming FPS | Real-time? |
|---|---|---:|---:|---|
| 360p | 640×360 → 1280×720 | 22.5 ms | 44.4 FPS | 1.5x real-time |
| 480p | 854×480 → 1708×960 | 32.7 ms | 30 FPS | 1.0x real-time |
| 720p | 1280×720 → 2560×1440 | 71.6 ms | 14 FPS | No |

Available in [ToolPiper Studio](https://modelpiper.com) for real-time video upscaling on macOS.

---

## Benchmarks

Reproducible. Run `python benchmark.py` on your own hardware.

```bash
# PSNR on Set5 (HR images in HR/, bicubic x2 inputs in LR/)
python benchmark.py --dataset /path/to/Set5

# Tiled FPS at 360p
python benchmark.py --resolution 640x360 --iterations 200
```

PSNR is measured on the Y channel with a 2px border crop, the standard for SR papers. `benchmark.py` reproduces every PiperSR score in this README from the shipped CoreML model, using [Set5 from Hugging Face](https://huggingface.co/datasets/eugenesiow/Set5) and the [EDSR benchmark sets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar). Its FPS mode times the bundled 128px tile model from Python. The full-frame numbers below come from the double-buffered video pipeline described above.

### Full PSNR Results

| Dataset | PiperSR | Bicubic | Delta |
|---|---:|---:|---:|
| Set5 | 37.54 dB | 33.66 dB | +3.88 dB |
| Set14 | 33.16 dB | 30.24 dB | +2.92 dB |
| BSD100 | 31.99 dB | 29.56 dB | +2.43 dB |

> No Urban100 score: its 100 test images were accidentally included in PiperSR's training data, so a score on them would not be a fair test.

### Inference Speed by Hardware

| Hardware | Mode | FPS | Latency |
|---|---|---:|---:|
| M2 Max | Full-frame 360p | 44.4 | 22.5 ms |
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

PiperSR is an image model. For real-time video super-resolution with double-buffered frame scheduling, Metal GPU output conversion, and streaming — see [ToolPiper Studio](https://modelpiper.com), which integrates PiperSR into a production video pipeline.

---

## Model Details

- **Task:** Single-image 2x super-resolution
- **Architecture:** 6-block residual CNN, 64 channels, BatchNorm + SiLU, PixelShuffle upsampling
- **Input:** 128×128 RGB tile (`input_image`). The `pipersr` package tiles images of any size
- **Output:** 256×256 RGB tile (`output_image`)
- **Format:** CoreML .mlpackage
- **Compute:** Apple Neural Engine via `.cpuAndNeuralEngine`
- **Training data:** LSDIR + DIV2K (85,791 images). Urban100's 100 test images were also included by mistake, so no Urban100 score is reported
- **Training cost:** ~$6 total on RunPod A6000 instances

For the full model card, see [MODEL_CARD.md](MODEL_CARD.md).

## License

**The model weights are [CC BY 4.0](MODEL_LICENSE) — fully permissive.** Use them for anything: personal, academic, commercial. Ship them in your app, build a product, **make money with it.** Seriously, that's allowed and encouraged.

The only thing we ask: **link back to [ModelPiper.com](https://modelpiper.com)** as attribution, so others can find the project.

```
Powered by PiperSR from ModelPiper — https://modelpiper.com
```

- Model weights: [CC BY 4.0](MODEL_LICENSE) — permissive, commercial use encouraged, attribution = a backlink to modelpiper.com
- Source code: [AGPL-3.0](LICENSE)

## Links

- [ModelPiper.com](https://modelpiper.com): Download PiperSR, browse benchmarks, explore on-device models
- [Hugging Face](https://huggingface.co/ModelPiper/PiperSR-2x): Model card and CoreML packages
- [ToolPiper Studio](https://modelpiper.com): Local macOS AI toolkit with real-time video upscaling
- [Ben Racicot](https://github.com/BenRacicot): Author

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
