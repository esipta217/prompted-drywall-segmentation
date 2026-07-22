# prompted-drywall-segmentation

## Overview

This project implements a text-conditioned segmentation model for drywall quality inspection. Given an input image and a natural language prompt, the model outputs a binary segmentation mask identifying one of two defect types:

- Taping area (drywall joints)
- Cracks

The system is designed around a single question: can a single fine-tuned vision-language model handle multiple defect categories through prompting alone, rather than training separate models per defect type.

## Goal

Train a model that understands prompts such as:

- `segment crack`
- `segment wall crack`
- `segment taping area`
- `segment joint tape`

and produces consistent, accurate segmentation masks regardless of which phrasing is used.

## Datasets

**Drywall Taping Area**
https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect

**Crack Dataset**
https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36

Both datasets were converted from COCO polygon annotations to binary masks. Final split sizes:

| Dataset | Train | Valid | Test |
|---|---|---|---|
| Taping | 715 | 153 | 154 |
| Cracks | 3758 | 805 | 806 |

## Model

- **Base model:** CLIPSeg (`CIDAS/clipseg-rd64-refined`), 150.7M parameters
- **Type:** Prompt-conditioned segmentation via joint image-text embedding
- **Why CLIPSeg:** unlike single-purpose segmentation architectures (UNet, DeepLab), CLIPSeg conditions on natural language, so one fine-tuned model handles both defect categories without separate heads or separate training runs

## Ablation: zero-shot vs fine-tuned

Before fine-tuning, the pre-trained CLIPSeg checkpoint was evaluated directly on a sample of the test set to establish a baseline. This confirms fine-tuning is necessary rather than assumed:

| | Zero-shot (pre-trained) | Fine-tuned (this model) |
|---|---|---|
| Global mIoU | measured in Cell 8b, see notebook | 0.573 |
| Global mDice | measured in Cell 8b, see notebook | 0.709 |

The delta between these two numbers is the actual contribution of the fine-tuning step, not just the raw CLIPSeg capability.

## Training details

| Parameter | Value |
|---|---|
| Epochs | 20 |
| Batch size | 4 |
| Image size | 352 x 352 |
| Optimizer | AdamW, weight decay 1e-4 |
| LR schedule | 3e-4, cosine annealing |
| Loss | BCE (pos_weight=8) + Dice, weighted 0.5/0.5 |
| Augmentation | horizontal flip only (train split) |
| Gradient clipping | max norm 1.0 |
| Device | T4 GPU |
| Seed | 42 (Python, NumPy, PyTorch CPU/CUDA, deterministic cudnn) |

The `pos_weight=8` in the BCE term compensates for the heavy foreground/background imbalance typical of thin-crack and joint-line masks, where positive pixels are a small minority of the image.

## Results

**Cracks**

| Prompt | mIoU | Dice |
|---|---|---|
| segment crack | 0.5559 | 0.6955 |
| segment wall crack | 0.5557 | 0.6954 |

**Taping area**

| Prompt | mIoU | Dice |
|---|---|---|
| segment taping area | 0.6631 | 0.7825 |
| segment joint tape | 0.6636 | 0.7828 |

**Overall (sample-weighted across all test images, not an average of the four rows above)**

- Global mIoU: 0.5730
- Global Dice: 0.7094

Prompt-phrasing variance is under 0.1 points of mIoU/Dice within each category, indicating the fine-tuned text encoder has learned a stable, prompt-invariant representation for each defect class rather than overfitting to one exact phrasing.

## Why cracks score lower than taping

This gap (roughly 10 points of mIoU) is a real, documented limitation, not noise:

- Hairline cracks (1-2 px wide) become sub-pixel after resizing to 352x352, so fine detail is lost before the model ever sees it
- Crack boundaries are inherently harder to hand-annotate consistently than taping-joint edges, which adds label noise to the ceiling of achievable score
- Taping joints have more uniform width and contrast; cracks vary widely in scale and visibility

## Documented failure modes

- **Hairline cracks under 2px:** lost to downsampling resolution; would need multi-scale or tiled inference at native resolution to recover
- **False positives on grout lines/tile seams:** visually similar linear features to cracks; not currently included as hard negatives in training
- **Fully painted-over joints:** no RGB signal exists in these cases; this is a sensing limitation, not a model limitation
- **Extreme lighting (over/under-exposed):** measurable accuracy drop; CLAHE preprocessing or lighting-augmented training data would help
- **Ragged ground-truth boundaries:** some ambiguity in original polygon annotations inflates apparent false-negative rate slightly

## Visual results

Example predictions (original / ground truth / model prediction overlays) are generated in the notebook's evaluation section and saved to `outputs/visuals/`.

## Performance and footprint

| Metric | Value |
|---|---|
| Training time | approx. 93 minutes (T4 GPU, 20 epochs) |
| Best epoch | 18/20 (Val Dice 0.7188) |
| Model size | approx. 603 MB |
| Inference (with test-time augmentation) | approx. 94 ms/image |
| Inference (without TTA) | approx. 50 ms/image, roughly 20 FPS |

Test-time augmentation averages predictions from the original image and its horizontal flip, which improves boundary stability at roughly 2x inference cost.

## Output format

- PNG masks, single-channel, values `{0, 255}`
- Naming convention: `{image_id}__{prompt_slug}.png`, e.g. `123__segment_crack.png`

## How to run

All training, evaluation, and inference code lives in a single Jupyter notebook designed to run end-to-end on Google Colab (T4 GPU).

```bash
git clone https://github.com/esipta217/prompted-drywall-segmentation
cd prompted-drywall-segmentation
```

Open `prompted_drywall_segmentation.ipynb` in Colab and run cells sequentially:

1. Setup and data download (Cells 1-6)
2. Zero-shot baseline ablation (Cell 8b)
3. Training (Cell 9)
4. Evaluation, visualization, and inference (Cells 10-14)

Checkpoints save to `checkpoints/best_model.pth`; masks and visual comparisons save to `outputs/`.

## Deployment notes

The current model runs at roughly 20 FPS without TTA on a T4, suitable for batch inspection workflows. For real-time or edge deployment, the following are documented options, not yet implemented:

- INT8 post-training quantization: expected reduction to roughly 150 MB, roughly 25 ms/image, minor Dice degradation
- ONNX export to remove Python inference overhead
- TensorRT on Jetson-class hardware for on-device inference without cloud connectivity
- SAM 2 refinement using CLIPSeg output as a prompt for tighter boundary tracing

## Key highlights

- Prompt-based segmentation across two distinct defect types with one fine-tuned model
- Quantified zero-shot vs fine-tuned improvement, not just post-training numbers
- Documented, honest failure analysis rather than only reporting favorable metrics
- Test-time augmentation for boundary stability
- Deterministic, seeded training for reproducibility

## Author

Esipta

## License

MIT — see `LICENSE`.
