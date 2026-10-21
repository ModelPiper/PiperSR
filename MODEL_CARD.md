# PiperSR Model Card

## Model Overview

| Field | Value |
|-------|-------|
| **Name** | PiperSR |
| **Task** | Single-image super-resolution (2x) |
| **Author** | [Ben Racicot](https://github.com/BenRacicot) |
| **Organization** | [ModelPiper](https://modelpiper.com) |
| **Format** | CoreML .mlpackage (fp16) |
| **Target Hardware** | Apple Neural Engine (Apple Silicon) |
| **Compute Units** | `.cpuAndNeuralEngine` |
| **License** | [CC BY 4.0](MODEL_LICENSE) |
| **Training data** | LSDIR + DIV2K (85,791 images), L1 + wavelet loss, 64 → 128 px progressive patches |

## Intended Use

PiperSR is designed for real-time image and video super-resolution on Apple Silicon devices. It targets the Apple Neural Engine specifically, leaving the GPU free for other workloads.

**Primary use cases:**
- Photo enhancement and upscaling
- Real-time 360p → 720p video upscaling (via [ToolPiper Studio](https://modelpiper.com))
- On-device preprocessing for downstream vision tasks
- Research and benchmarking of ANE inference

**Out of scope:**
- Medical imaging (not validated for diagnostic use)
- Satellite/aerial imagery (not trained on this domain)
- Arbitrary scale factors (2x only)

## Performance

### Image Quality (PSNR in dB, higher is better)

| Dataset | PiperSR |
|---------|---------|
| Set5 | 37.54 |
| Set14 | 33.16 |
| BSD100 | 31.99 |

Scores are for the shipped CoreML model (FP16, 128px tiles). The PyTorch training checkpoint scored 37.54 / 33.21 / 31.98.

No Urban100 score is reported: its 100 test images were accidentally included in the training data, so a score on them would not be a fair test.

### Inference Speed

| Hardware | FPS | Latency (ms) |
|----------|-----|---------------|
| M2 Max | 44.4 | 22.5 |

> PSNR is Y channel with a 2px border crop. `python benchmark.py --dataset /path/to/Set5` reproduces these scores. We encourage independent reproduction.

## Input / Output

- **Input:** 128×128 RGB image tile (`input_image`). The `pipersr` package splits images of any size into tiles and stitches the result
- **Output:** 256×256 RGB image tile (`output_image`)
- **Color space:** sRGB

## Limitations

- Optimized for natural photographic content. Performance on text, line art, or synthetic images may vary.
- 2x scale factor only. No 4x or 8x variants at this time.
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
  url = {https://github.com/ModelPiper/PiperSR},
  note = {Available at https://modelpiper.com}
}
```

## Contact

- Model & weights: [ModelPiper.com](https://modelpiper.com)
- Real-time video: [ToolPiper Studio](https://modelpiper.com)
- Author: [Ben Racicot](https://github.com/BenRacicot)
- Questions / enterprise support: license@modelpiper.com
