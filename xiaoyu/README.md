# Final GUV attachment analysis package

This package contains the final data, per-image annotations, auto-annotation
pipeline, and statistical figures for the GUV attachment analysis.

## Directory layout

```text
Final/
├── OriData/
│   └── GUV_filament_quantification/   # 54 active source TIFs
├── annotations/                       # one CSV per TIF
├── statistics/
│   ├── figures/                       # one composition figure per experiment
│   └── scripts/analyze_final.py
└── workflow/
    └── auto_annotation/               # final inference scripts and models
```

## Data

- **54 TIFs** are included, preserving the source `osmolarity/`,
  `ficoll_vs_protrusion/`, `quantification-DNA_dosage/`, and
  `quantification_phase/` structure.
- The DNA channel used is the green `*d1.TIF` image.

## Annotations

Each TIF has a same-stem CSV under `annotations/` with the same relative path.
The columns are:

```text
object_id,cx_px,cy_px,r_px,diameter_um,class
```

`class` is `attached`, `none`, or `unsure`. `not_guv` objects are excluded.
There are **18,570 final objects across 54 fields**: 4,494 attached, 13,062
none, and 1,014 unsure. The annotations include manually added GUVs and
corrected centers/radii; coordinates are in full-image pixels..

## Auto-annotation pipeline

See `workflow/auto_annotation/README.md`. The packaged model is the newest
reviewed-label attachment model (`attachment_reviewed_v3`) with the final
center detector. Only inference is included; manual-review code is not part of
this package.

## Statistics

See `statistics/README.md`. Composition figures use the size gates **<15, 15–20, 20–25, and ≥25 µm**
and are provided for:

- osmolarity
- Ficoll vs protrusion
- DNA dosage
- quantification_phase

The analysis code reads the per-TIF annotations directly.
