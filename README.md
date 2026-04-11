# GEAR Data Analysis

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Conda](https://img.shields.io/badge/conda-env-green.svg)](environment.yml)

> Quantitative analysis toolkit for spatial fluorescence microscopy imaging

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Environment Setup](#environment-setup)
- [Data Requirements](#data-requirements)
- [Project Structure](#project-structure)
- [Typical Workflows](#typical-workflows)
  - [1. Straightening](#1-straightening-if-needed)
  - [2. Figure Analysis](#2-figure-analysis)
- [Outputs](#outputs)
- [License](#license)

---

## Overview

This repository contains Jupyter notebooks for **quantitative analysis of spatial fluorescence microscopy images**. The workflows cover straightened fluorescence profiles, kymographs, ROI time courses, peak-position measurements, and skeleton-length quantification for figure-specific analysis tasks.

### Analysis Pipeline

1. **Manual Annotation**: Filament boundaries are manually annotated using [LabelMe](https://github.com/wkentaro/labelme) to generate polygon masks
2. **Skeleton Extraction**: Masks are processed with the [Skan](https://jni.github.io/skan/) library to extract filament skeletons, with the longest path defined as the principal axis
3. **Image Straightening**: Images are straightened along the principal axis using bilinear interpolation
4. **Binning & Quantification**: The straightened axis is divided into N equal-length bins, and integrated fluorescence is calculated for each bin
5. **Normalization**: Intensity profiles and kymographs are normalized by the global maximum intensity across the dataset (scaling to 0–1)
6. **Smoothing**: Profiles are optionally smoothed with cubic splines

> **Note**: All normalization and quantification steps are performed independently per fluorescence channel.

---

## Features

- Automated fluorescence profile extraction along filament axes
- Multi-channel fluorescence analysis support
- Kymograph generation for time-series data
- Peak-position and skeleton-length quantification helpers
- Publication-ready SVG outputs
- One formal CSV export per figure for downstream analysis
- Reproducible workflows with pinned dependencies

---

## Environment Setup

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
- Python 3.10

### Key Dependencies

All dependencies are pinned in `environment.yml`:

- **Image Processing**: OpenCV, scikit-image, Pillow, tifffile
- **Scientific Computing**: NumPy, SciPy, Pandas
- **Skeleton Analysis**: Skan
- **Visualization**: Matplotlib, ipywidgets
- **Jupyter**: IPython, ipykernel

### Installation

```bash
# Clone the repository
git clone git@github.com:BioProgramming-Lab/GEAR_data_analysis.git
cd GEAR_data_analysis

# Create and activate conda environment
conda env create -f environment.yml
conda activate GEAR_data_analysis

# Install the local analysis package in editable mode
# (already included in environment.yml; rerun manually after package-layout changes)
python -m pip install -e .

# (Optional) Register kernel for Jupyter
python -m ipykernel install --user --name GEAR_data_analysis
```

The shared analysis code lives under `src/gear_data_analysis/` and is intended to be used as an **editable local package** so notebooks, scripts, and helpers all import the same codebase.

### Running Notebooks

Launch Jupyter Notebook or JupyterLab and select the `GEAR_data_analysis` kernel:

```bash
# Option 1: Jupyter Notebook
jupyter notebook

# Option 2: JupyterLab
jupyter lab
```

---

## Data Requirements

### Input Data Format

| Component | Description |
|-----------|-------------|
| **Channel Folders** | Typically named `c1`, `c2`, etc. Files are matched by normalized basename (for example `*_c1` with `*_c3`, or `ACA-5-m` with `ACA-5-g`) |
| **ROI/Mask Files** | LabelMe-style JSON polygons, usually one JSON per image with the same basename as the image |
| **Image Files** | TIFF format (single-frame or multi-frame for time-series) |

### Important Notes

- **Straightening**: Target channel must have corresponding JSON annotations; mapping channels must match dimensions of the target
- **Time-series**: Provide multi-frame TIFFs per channel for kymograph generation
- **File Organization**: Maintain consistent naming conventions across channels for automatic pairing

---

## Project Structure

Each figure is organized in its own folder with a standardized numbering system for data and notebooks. The final notebook and final output directory should always follow the **current figure ID**.

### Folder Naming Convention

```
FigureFolder/
├── 00_*_raw/                    # Raw input data
│   ├── c1_channel1/
│   ├── c2_channel2/
│   └── *.json                   # LabelMe annotations
├── 01_*_plot.ipynb              # Analysis notebook when no figure-specific preprocessing is needed
└── 02_*_plot_output/            # Final outputs (SVG, CSV)

FigureFolderWithPreprocessing/
├── 00_*_raw/
├── 01_*_cropper.ipynb           # Optional figure-specific preprocessing notebook
├── 02_*_cropper_output/
├── 03_*_plot.ipynb
└── 04_*_plot_output/

Repository root/
├── src/gear_data_analysis/      # Shared reusable analysis package
│   ├── _bootstrap.py            # Notebook/package bootstrap helpers
│   ├── csv_output_common.py
│   ├── straightener_common.py
│   ├── profile_common.py
│   ├── split_profile_common.py
│   ├── kymograph_common.py
│   ├── roi_timecourse_common.py
│   ├── skeleton_length_common.py
│   └── peak_position_common.py
├── scripts/                    # Local-only helper scripts (ignored)
└── Fig*/
```

In a few inherited workflows, the `00_*` raw folder or the preprocessing notebook/output may still carry an earlier panel letter because the source asset was shared with the previous panel in the manuscript. That is expected. The important rule is that the **final plot notebook**, **final output directory**, and **formal exported filenames** follow the current figure ID.

### Numbering System

| Prefix | Content Type | Description |
|--------|--------------|-------------|
| `00_*` | Raw Data | Original input images and annotations; may occasionally retain an upstream/shared panel ID |
| `01_*` | Analysis / Preprocessing | First figure-level step; usually the plot notebook, or a preprocessing notebook if one exists |
| `02_*` | Output / Intermediate | Final plot outputs for simple figures, or intermediate outputs when an extra preprocessing step exists |
| `03_*` | Analysis | Plot notebook for figures that already have a `01_*` preprocessing step |
| `04_*` | Final Output | Final plot outputs for figures that already have a `01_*` preprocessing step |

### Example Structure

Simple figure:
```text
Fig2f/
├── 00_Fig2f_raw/
├── 01_Fig2f_plot.ipynb
└── 02_Fig2f_plot_output/
```

Figure with extra preprocessing:
```text
FigS7g/
├── 00_FigS7f_raw/
├── 01_FigS7f_cropper.ipynb
├── 02_FigS7f_cropper_output/
├── 03_FigS7g_plot.ipynb
└── 04_FigS7g_plot_output/
```

### Execution Order

1. **Preprocessing** (if applicable): Run a figure-specific `01_*` notebook such as a cropper
2. **Analysis**: Run the plot notebook for that figure. For simple figures this is usually `01_*_plot.ipynb`; for figures with extra preprocessing it is typically `03_*_plot.ipynb`
3. **Export**: Figures and CSV data are saved to the next numbered `*_plot_output/` directory for that figure

Final CSV rule:

- Each figure should keep exactly one formal CSV in its output directory, named only by the current figure ID such as `FigS8d.csv`.
- If a figure internally generates multiple tables during plotting, they should be consolidated into that one figure-level CSV with metadata columns that identify the subpanel / channel / condition.

Figures that quantify polygon or mask skeleton lengths can follow the same notebook layout while using the shared `gear_data_analysis.skeleton_length_common` helper instead of `gear_data_analysis.straightener_common`.
Figures that quantify peak positions from straightened profiles can use the shared `gear_data_analysis.peak_position_common` helper while still pulling their straightened stacks from `gear_data_analysis.straightener_common`.

### Path Configuration

All notebooks use **relative paths** with `pathlib.Path` for portability. They first try the installed editable package and only fall back to the local `src/` path when needed:

```python
import sys
from pathlib import Path

try:
    from gear_data_analysis._bootstrap import bootstrap_notebook
except ModuleNotFoundError:
    REPO_ROOT = ...
    SRC_ROOT = REPO_ROOT / "src"
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from gear_data_analysis._bootstrap import bootstrap_notebook

REPO_ROOT, SRC_ROOT = bootstrap_notebook(Path.cwd().resolve())

from gear_data_analysis.straightener_common import STRAIGHTENER_CONFIGS

# Input paths (relative to notebook location)
membrane_dir = Path('00_Fig2f_raw/c2_membrane')
ftsz_dir = Path('00_Fig2f_raw/c1_FtsZ')

# Output paths
output_svg_file = Path('02_Fig2f_plot_output/Fig2f.svg')
output_csv_file = Path('02_Fig2f_plot_output/Fig2f.csv')
```

> **Note**: Plot notebooks are still intended to run from the figure folder. The shared package resolves raw-data paths from the repository root, so no intermediate straightener outputs need to be written to disk. The editable install keeps package imports and repository-relative data paths aligned to the same checkout.

---

## Typical Workflows

### 1. Straightening (if needed)

Image straightening aligns filaments along a principal axis for consistent analysis. In this repository, straightening is provided by the shared package module `gear_data_analysis.straightener_common` and is called directly inside the relevant plot notebooks.

#### Steps

1. **Open the analysis notebook** for the figure that needs straightening (for example `Fig1h/01_Fig1h_plot.ipynb`)

2. **Configure parameters**:
   ```python
   from gear_data_analysis.straightener_common import (
       STRAIGHTENER_CONFIGS,
       build_named_straightened_images,
   )

   straightened_images = build_named_straightened_images(STRAIGHTENER_CONFIGS['Fig1h'])
   ```

3. **Review the shared configuration** in `src/gear_data_analysis/straightener_common.py`:
   - target channel with LabelMe annotations
   - mapping channels to straighten with the same mask/skeleton
   - skeleton start preference (`'top'`, `'bottom'`, `'left'`, `'right'`)
   - optional padding behavior for canvas-standardized outputs

4. **Run the notebook**: The straightening algorithm runs in memory during notebook execution

5. **Inspect results**: The returned arrays are consumed directly by the plotting code, so no intermediate TIFF stacks need to be exported

#### Configuration Options

| Parameter | Description | Options |
|-----------|-------------|---------|
| `start_preference` | Preferred skeleton start point | `'top'`, `'bottom'`, `'left'`, `'right'` |
| `pad_to_canvas` | Pad images to uniform size | `True`, `False` |
| `output_width` / `output_height` | Target padded canvas size (pixels) | Integer width / height |

---

### 2. Figure Analysis

Generate quantitative plots and export data for publication.

#### Steps

1. **Open the analysis notebook** (e.g., `Fig2f/01_Fig2f_plot.ipynb`, or `FigS7g/03_FigS7g_plot.ipynb` if the figure has an extra preprocessing step)

2. **Configure input paths**:
   ```python
   import sys
   from pathlib import Path

   try:
       from gear_data_analysis._bootstrap import bootstrap_notebook
   except ModuleNotFoundError:
       REPO_ROOT = ...
       SRC_ROOT = REPO_ROOT / "src"
       if str(SRC_ROOT) not in sys.path:
           sys.path.insert(0, str(SRC_ROOT))
       from gear_data_analysis._bootstrap import bootstrap_notebook

   REPO_ROOT, SRC_ROOT = bootstrap_notebook(Path.cwd().resolve())

   from gear_data_analysis.straightener_common import (
       STRAIGHTENER_CONFIGS,
       build_named_straightened_images,
       build_straightened_stack_map,
   )

   # For ROI analysis
   membrane_dir = Path('00_Fig2f_raw/c2_membrane')
   protein_dir = Path('00_Fig2f_raw/c1_FtsZ')

   # For figures that need straightening before plotting
   straightened_images = build_named_straightened_images(STRAIGHTENER_CONFIGS['Fig1h'])
   straightened_stacks = build_straightened_stack_map(STRAIGHTENER_CONFIGS['Fig4c'])

   # Example accessors
   single_image = straightened_images['example_channel_straight.tif']
   time_series_stack = straightened_stacks['c1_membrane']
   ```

3. **Adjust analysis parameters** (if needed):
   - Number of bins for axial quantification
   - Smoothing parameters for spline interpolation
   - Normalization method (global vs. per-image)
   - Colormap settings for kymographs

4. **Run analysis**: Execute all cells to generate plots and quantitative data

5. **Review outputs**:
   - **SVG files**: Vector graphics for publication
   - **CSV files**: The single formal figure-level table, including measurements and metadata needed for downstream analysis

#### Output Files

Standardized outputs now use the **figure ID as the basename whenever possible**.

```text
*_plot_output/
├── FigX.svg                  # Single primary plot
├── FigX.csv                  # The one formal CSV export for that figure
├── FigX_c1_membrane.svg      # Channel-specific companion plot when needed
└── FigX001.svg               # Optional per-example SVG outputs when needed
```

Avoid descriptive suffixes such as `*_intensity_data.csv`, `*_raw_data.csv`, or `*_normalized_data.csv`. If a workflow internally generates multiple tables, consolidate them into one formal figure-level CSV with metadata columns that identify the subpanel / channel / condition.

---

## Outputs

All analysis notebooks generate standardized output files for reproducibility and publication.

### File Types

| Extension | Content | Description |
|-----------|---------|-------------|
| `.csv` | Quantitative Data | Raw and normalized intensity values, bin positions, statistical metrics |
| `.svg` | Vector Graphics | Publication-ready plots (editable in Illustrator, Inkscape, etc.) |
| `.tif` / `.tiff` | Optional Intermediate Images | Figure-specific preprocessing outputs such as cropped stacks |

### Typical Output Structure

```text
*_plot_output/
├── FigX.svg                      # Single exported plot
├── FigX.csv                      # Single exported data table
├── FigX_channel-name.svg         # Optional channel-specific companion plot
└── FigX###.svg                   # Optional per-sample SVG outputs for multi-example figures
```

### Data Format

CSV files typically contain:
- **Position or measurement data**: For example bin number, distance along axis, peak position, or skeleton length
- **Quantitative values**: Per-channel fluorescence values or other derived measurements, typically with raw and normalized columns stored in the same CSV when relevant
- **Metadata**: Image identifiers, channel names, conditions, analysis parameters, or table-type labels used to merge multiple internal tables into one export

---


## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 BioProgramming-Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---


## Acknowledgments

- [LabelMe](https://github.com/wkentaro/labelme) for annotation tools
- [Skan](https://jni.github.io/skan/) for skeleton analysis
- The scientific Python community for foundational libraries


---

**Made with Python and Jupyter**
