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

This repository contains Jupyter notebooks for **quantitative analysis of spatial fluorescence microscopy images**. The workflow enables precise quantification of fluorescence distribution along cellular filaments and structures.

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
- Publication-ready SVG outputs
- CSV data export for downstream analysis
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

# (Optional) Register kernel for Jupyter
python -m ipykernel install --user --name GEAR_data_analysis
```

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
| **Channel Folders** | Typically named `c1`, `c2`, etc. Files are matched by sorted filename order unless otherwise specified in a notebook |
| **ROI/Mask Files** | LabelMe-style JSON polygons, usually one JSON per image with the same basename as the image |
| **Image Files** | TIFF format (single-frame or multi-frame for time-series) |

### Important Notes

- **Straightening**: Target channel must have corresponding JSON annotations; mapping channels must match dimensions of the target
- **Time-series**: Provide multi-frame TIFFs per channel for kymograph generation
- **File Organization**: Maintain consistent naming conventions across channels for automatic pairing

---

## Project Structure

Each figure is organized in its own folder with a standardized numbering system for data and notebooks.

### Folder Naming Convention

```
FigureFolder/
├── 00_*_raw/                    # Raw input data
│   ├── c1_channel1/
│   ├── c2_channel2/
│   └── *.json                   # LabelMe annotations
├── 01_*_straightener.ipynb      # Preprocessing notebook
├── 02_*_straightener_output/    # Straightened images
├── 03_*_plot.ipynb             # Analysis and plotting notebook
└── 04_*_plot_output/           # Final outputs (SVG, CSV)
```

### Numbering System

| Prefix | Content Type | Description |
|--------|--------------|-------------|
| `00_*` | Raw Data | Original input images and annotations |
| `01_*` | Preprocessing | Straightener/cropper notebooks |
| `02_*` | Intermediate | Straightener outputs or plot outputs (ROI-only figures) |
| `03_*` | Analysis | Main plotting and analysis notebooks |
| `04_*` | Final Output | Publication-ready plots and data tables |

### Example Structure

```
FigS7b/
├── 00_FigS7b_raw/
│   ├── c1_membrane/
│   │   ├── image001.tif
│   │   ├── image001.json
│   │   └── ...
│   └── c2_FtsZ/
│       ├── image001.tif
│       └── ...
├── 01_FigS7b_straightener.ipynb
├── 02_FigS7b_straightener_output/
│   ├── c1_membrane_straightened/
│   └── c2_FtsZ_straightened/
├── 03_FigS7b_plot.ipynb
└── 04_FigS7b_plot_output/
    ├── FigS7b.svg
    └── FigS7b_data.csv
```

### Execution Order

1. **Preprocessing** (if applicable): Run `01_*_straightener.ipynb` or `01_*_cropper.ipynb` to generate `02_*` outputs
2. **Analysis**: Run `03_*_plot.ipynb` using the generated outputs (or raw data if no preprocessing is needed)
3. **Export**: Figures and CSV data are saved to `04_*_plot_output/`

### Path Configuration

All notebooks use **relative paths** with `pathlib.Path` for portability:

```python
from pathlib import Path

# Input paths (relative to notebook location)
membrane_dir = Path('00_Fig2F_raw/c2_membrane')
ftsz_dir = Path('00_Fig2F_raw/c1_FtsZ')

# Output paths
output_svg_file = Path('04_Fig2F_plot_output/Fig2F.svg')
output_csv_file = Path('04_Fig2F_plot_output/Fig2F_data.csv')
```

> **Note**: Paths are resolved from the notebook's working directory (typically the figure folder). If running from a different directory, adjust paths accordingly.

---

## Typical Workflows

### 1. Straightening (if needed)

Image straightening aligns filaments along a principal axis for consistent analysis.

#### Steps

1. **Open the straightener notebook** (e.g., `Fig5g/01_Fig5g_Straightener.ipynb`)

2. **Configure parameters**:
   ```python
   TARGET_DIR = Path('00_raw/c1_target')      # Channel with JSON annotations
   MAPPING_DIRS = [                            # Additional channels to process
       Path('00_raw/c2_channel2'),
       Path('00_raw/c3_channel3')
   ]
   OUTPUT_DIR = Path('02_straightener_output') # Destination for output

   # Optional parameters
   SKELETON_START_PREFERENCE = 'top'           # or 'bottom', 'left', 'right'
   PAD_TO_CANVAS = True                        # Pad to uniform canvas size
   ```

3. **Run validation**: Check file counts and image dimensions match across channels

4. **Process images**: Execute the straightening algorithm

5. **Inspect results**: Use the navigation widget to review skeleton extraction and orientation

6. **Export**: Save individual TIFFs or multi-frame stacks

#### Configuration Options

| Parameter | Description | Options |
|-----------|-------------|---------|
| `SKELETON_START_PREFERENCE` | Preferred skeleton start point | `'top'`, `'bottom'`, `'left'`, `'right'` |
| `PAD_TO_CANVAS` | Pad images to uniform size | `True`, `False` |
| `OUTPUT_CANVAS_SIZE` | Target output dimensions (pixels) | `(width, height)` tuple |

---

### 2. Figure Analysis

Generate quantitative plots and export data for publication.

#### Steps

1. **Open the analysis notebook** (e.g., `Fig5g/03_Fig5g_plot.ipynb`)

2. **Configure input paths**:
   ```python
   from pathlib import Path

   # For ROI analysis
   membrane_dir = Path('00_raw/c1_membrane')
   protein_dir = Path('00_raw/c2_protein')

   # For kymographs (time-series)
   stack_file = Path('02_straightener_output/timeseries_stack.tif')

   # For axial projections
   channel1_file = Path('02_straightener_output/c1_straightened.tif')
   channel2_file = Path('02_straightener_output/c2_straightened.tif')
   ```

3. **Adjust analysis parameters** (if needed):
   - Number of bins for axial quantification
   - Smoothing parameters for spline interpolation
   - Normalization method (global vs. per-image)
   - Colormap settings for kymographs

4. **Run analysis**: Execute all cells to generate plots and quantitative data

5. **Review outputs**:
   - **SVG files**: Vector graphics for publication
   - **CSV files**: Raw and normalized intensity data

#### Output Files

```
04_plot_output/
├── FigX_profile.svg          # Fluorescence profile plot
├── FigX_kymograph.svg        # Time-series kymograph
├── FigX_data_raw.csv         # Raw intensity values
└── FigX_data_normalized.csv  # Normalized (0-1) values
```

---

## Outputs

All analysis notebooks generate standardized output files for reproducibility and publication.

### File Types

| Extension | Content | Description |
|-----------|---------|-------------|
| `.csv` | Quantitative Data | Raw and normalized intensity values, bin positions, statistical metrics |
| `.svg` | Vector Graphics | Publication-ready plots (editable in Illustrator, Inkscape, etc.) |
| `.tif` / `.tiff` | Processed Images | Straightened images, single-frame or multi-frame stacks |

### Typical Output Structure

```
04_plot_output/
├── FigX_profile.svg              # Axial fluorescence profile
├── FigX_kymograph.svg            # Time-series heatmap
├── FigX_overlay.svg              # Multi-channel overlay
├── FigX_raw_data.csv             # Raw intensity measurements
├── FigX_normalized_data.csv      # Scaled 0-1 intensities
└── FigX_statistics.csv           # Summary statistics
```

### Data Format

CSV files typically contain:
- **Position data**: Bin number, distance along axis (µm or normalized 0-1)
- **Intensity data**: Per-channel fluorescence values
- **Metadata**: Image identifiers, channel names, analysis parameters

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

## Contact

For questions or collaborations:
- **GitHub Issues**: [Report bugs or request features](https://github.com/BioProgramming-Lab/GEAR_data_analysis/issues)
- **Repository**: [BioProgramming-Lab/GEAR_data_analysis](https://github.com/BioProgramming-Lab/GEAR_data_analysis)

---

**Made with Python and Jupyter**
