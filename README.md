# PiperSR

**First super-resolution model designed and optimized for Apple Neural Engine.**

4x upscale. Real-time. Runs entirely on the ANE in every Mac, iPhone, and iPad — no GPU, no cloud.

| Metric | PiperSR |
|--------|---------|
| **PSNR (Set5)** | 37.54 dB |
| **FPS (M2 Max)** | 44.4 |
| **Compute** | Apple Neural Engine |
| **Precision** | fp16 |
| **Scale** | 4x |

> Built by [Ben Racicot](https://github.com/BenRacicot). Available on [ModelPiper.com](https://modelpiper.com).

---

## Quick Start

```bash
pip install pipersr
```

```python
from pipersr import upscale

output = upscale("input.png")
output.save("output_4x.png")
```

Or use the script directly:

```bash
python inference.py --input photo.png --output photo_4x.png
```

## Download

Model weights are hosted on [ModelPiper.com](https://modelpiper.com/models/pipersr).

The `pipersr` package downloads the CoreML model automatically on first use. To download manually:

```bash
python -c "from pipersr import download; download()"
```

## Benchmarks

Tested on M2 Max, macOS 15+. All models running on the same hardware.

| Model | Set5 PSNR (dB) | FPS (M2 Max) | Compute | Format |
|-------|----------------|--------------|---------|--------|
| **PiperSR** | **37.54** | **44.4** | **ANE** | **CoreML** |
| ESPCN (CoreML) | 33.0 | — | CPU/GPU | CoreML |
| IMDN (CoreML) | 34.4 | — | GPU | CoreML |
| Real-ESRGAN (CoreML) | 35.8 | — | GPU | CoreML |

> Run `python benchmark.py` on your own hardware to fill in the FPS column. We publish what we measure. Reproduce it yourself.

## Samples

<table>
<tr>
<td><strong>Input (1x)</strong></td>
<td><strong>PiperSR Output (4x)</strong></td>
</tr>
<tr>
<td><img src="samples/butterfly_lr.png" width="200"></td>
<td><img src="samples/butterfly_sr.png" width="200"></td>
</tr>
<tr>
<td><img src="samples/baby_lr.png" width="200"></td>
<td><img src="samples/baby_sr.png" width="200"></td>
</tr>
</table>

## Why ANE?

Every Apple Silicon device has a Neural Engine — a dedicated inference accelerator that most ML models ignore. The entire ML ecosystem targets CUDA GPUs. PiperSR was designed from scratch for ANE, not converted from a GPU model.

The result: real-time super-resolution with zero GPU usage. Your GPU stays free for rendering, compositing, or whatever else you're doing.

For real-time video upscaling powered by PiperSR, see [ToolPiper](https://toolpiper.com).

## Supported Hardware

Any Apple Silicon device:
- Mac: M1, M1 Pro/Max/Ultra, M2, M2 Pro/Max/Ultra, M3, M3 Pro/Max/Ultra, M4, M4 Pro/Max/Ultra, M5
- iPhone: A15+
- iPad: M1+

Performance scales with ANE generation. M2 Max benchmarks shown above.

## Model Details

- **Task:** Single-image 4x super-resolution
- **Input:** RGB image, any resolution (CoreML flexible input)
- **Output:** 4x upscaled RGB image, fp16
- **Format:** CoreML .mlpackage
- **Compute target:** Apple Neural Engine via `.cpuAndNeuralEngine`

For the full model card, see [MODEL_CARD.md](MODEL_CARD.md).

## For Video & Real-Time

PiperSR is an image model. For real-time video super-resolution with frame scheduling, buffering, and streaming — see [ToolPiper](https://toolpiper.com), which integrates PiperSR into a full on-device video pipeline.

## License

- **Code** (inference.py, benchmark.py, pipersr package): [AGPL-3.0](LICENSE)
- **Model weights** (.mlpackage): [PiperSR Model License](MODEL_LICENSE) — free for personal and academic use, attribution required, commercial use requires a separate license

If you use PiperSR in your project, include:

```
Powered by PiperSR from ModelPiper — https://modelpiper.com
```

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
  note = {Available at https://modelpiper.com}
}
```
