# PiperSR Model Card

## Model Overview

| Field | Value |
|-------|-------|
| **Name** | PiperSR |
| **Task** | Single-image super-resolution (4x) |
| **Author** | [Ben Racicot](https://github.com/BenRacicot) |
| **Organization** | [ModelPiper](https://modelpiper.com) |
| **Format** | CoreML .mlpackage (fp16) |
| **Target Hardware** | Apple Neural Engine (Apple Silicon) |
| **Compute Units** | `.cpuAndNeuralEngine` |
| **License** | [PiperSR Model License](MODEL_LICENSE) |

## Intended Use

PiperSR is designed for real-time image and video super-resolution on Apple Silicon devices. It targets the Apple Neural Engine specifically, leaving the GPU free for other workloads.

**Primary use cases:**
- Photo enhancement and upscaling
- Real-time video upscaling (via [ToolPiper](https://toolpiper.com))
- On-device preprocessing for downstream vision tasks
- Research and benchmarking of ANE inference

**Out of scope:**
- Medical imaging (not validated for diagnostic use)
- Satellite/aerial imagery (not trained on this domain)
- Arbitrary scale factors (4x only)

## Performance

### Image Quality (PSNR in dB, higher is better)

| Dataset | PiperSR |
|---------|---------|
| Set5 | 37.54 |
| Set14 | — |
| BSD100 | — |
| Urban100 | — |

### Inference Speed

| Hardware | FPS | Latency (ms) |
|----------|-----|---------------|
| M2 Max | 44.4 | 22.5 |
| M1 | — | — |
| M3 Pro | — | — |

> Benchmarks measured with `benchmark.py`. We encourage independent reproduction.

## Input / Output

- **Input:** RGB image, any resolution (CoreML flexible input shape)
- **Output:** RGB image at 4x input resolution, fp16 precision
- **Color space:** sRGB

## Limitations

- Optimized for natural photographic content. Performance on text, line art, or synthetic images may vary.
- 4x scale factor only. No 2x or 8x variants at this time.
- Requires Apple Silicon. Will not run on Intel Macs or non-Apple hardware.
- CoreML format only. No PyTorch, ONNX, or TensorFlow export provided.

## Ethical Considerations

Super-resolution can enhance images in ways that add plausible but fabricated detail. Upscaled images should not be represented as original-resolution captures, particularly in forensic, journalistic, or legal contexts.

## How to Cite

```bibtex
@software{pipersr2026,
  author = {Racicot, Ben},
  title = {PiperSR: Real-Time Super-Resolution for Apple Neural Engine},
  year = {2026},
  url = {https://github.com/BenRacicot/PiperSR},
  note = {Available at https://modelpiper.com}
}
```

## Contact

- Model & weights: [ModelPiper.com](https://modelpiper.com)
- Real-time video: [ToolPiper](https://toolpiper.com)
- Author: [Ben Racicot](https://github.com/BenRacicot)
- Commercial licensing: license@modelpiper.com
