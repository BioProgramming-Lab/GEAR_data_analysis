# Final GUV auto-annotation pipeline

This is the inference pipeline used for the final reviewed-label package.

## Contents

- `cnn_pipeline.py` — end-to-end detector + score gate + attachment classifier
- `cnn_center_detector.py` — center/radius detector model and inference
- `cnn_stage2.py` — image cropping and MobileNet transforms
- `cnn_binary_gate.py` — binary attached/no-filament head
- `common.py` — image loading and data paths
- `field_paths.py` — maps field tags to the TIFs in `Final/OriData`
- `models/` — final detector and `attachment_reviewed_v3` ensemble

Only inference is included. Training, review tooling, and manual-review code are
not part of this package.

## Run

From this directory:

```bash
python cnn_pipeline.py --all-fields
python cnn_pipeline.py --field='-200osmo_-200-1'
```

Outputs go to `outputs/` by default. For each field, the pipeline writes:

- `<field>.csv` — proposals with center `x`, `y`, radius `r`, detector score,
  score-gate decision, attachment probability, and final `attached`/`none`
  type
- `<field>.jpg` — overlay preview
- `predictions.csv` and `summary.csv` — combined results

The default model is `models/attachment_reviewed_v3`; `--type-threshold 0.5`
is the default classification threshold. `--all-fields` covers the 54 active
TIFs in `Final/OriData`.

## Environment

```bash
python -m pip install -r requirements.txt
```

The pipeline uses PyTorch; MPS, CUDA, or CPU is selected automatically.
