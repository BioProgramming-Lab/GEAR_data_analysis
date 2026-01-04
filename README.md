# GEAR_data_analysis

## Overview
This repository contains notebooks for quantitative analysis of spatial fluorescence. Filament boundaries are
manually annotated in each image using LabelMe (https://github.com/wkentaro/labelme) to generate polygon masks.
Masks are processed with the Skan library to extract filament skeletons, with the longest path defined as the
principal axis. Images are straightened along this axis using bilinear interpolation to compute fluorescence
profiles. The straightened axis is divided into N equal-length bins (see analysis code), and integrated
fluorescence is calculated for each bin. Intensity profiles and kymographs are normalized by the global maximum
intensity across the dataset (scaling values to 0–1), and profiles are optionally smoothed with cubic splines.
All normalization and quantification steps are performed independently per fluorescence channel.

## Environment Setup
Dependencies are pinned in `environment.yml` (Python 3.10, OpenCV, NumPy/SciPy/Pandas, scikit-image, skan,
matplotlib, Pillow, ipywidgets, tifffile, etc.).

```bash
# clone and enter
git clone git@github.com:BioProgramming-Lab/GEAR_data_analysis.git
cd GEAR_data_analysis

# create and activate
conda env create -f environment.yml
conda activate GEAR_data_analysis

# (optional) register kernel for Jupyter
python -m ipykernel install --user --name GEAR_data_analysis
```

Run notebooks in Jupyter Notebook/Lab using the GEAR_data_analysis kernel.

## Data Requirements

- Channel folders typically named c1, c2, ...; files are matched by sorted filename order unless otherwise
  specified in a notebook.
- ROI/mask: LabelMe-style JSON polygons, usually one JSON per image with the same basename as the image.
- Straightening: Target channel must have JSON; mapping channels must match dimensions of the target.
- For time-series stacks (kymographs), provide multi-frame TIFFs per channel.

## Typical Workflows

### 1) Straightening (if needed)

- Open the relevant *_Straightener.ipynb (e.g., Fig5g/DIVISION-FIG5/Fig5g_Straightener.ipynb,
  Fig5f/raw_data/-MinCDE/Fig5f01_Straightener.ipynb).
- Configure:
  - TARGET_DIR: channel with JSON annotations.
  - MAPPING_DIRS: other channels to straighten together.
  - OUTPUT_DIR: destination for straightened TIFFs.
  - Optional: SKELETON_START_PREFERENCE, PAD_TO_CANVAS, output canvas size.
- Run validation (checks file counts and dimensions), then processing.
- Use the navigation widget to inspect skeleton/START orientation; use save controls to export individual TIFFs
  or stacks.

### 2) Figure Analysis

- Open the figure notebook (Fig5g/Fig5g.ipynb, Fig5c/Fig5c.ipynb, Fig3d/Fig3d.ipynb, etc.).
- Edit the path variables at the top to point to your data:
  - ROI analyses: two channel folders + output CSV/SVG paths.
  - Kymographs: multi-channel TIFF stacks; adjust axis/colormaps/normalization if needed.
  - Axial projections (Fig5g): two straightened TIFF paths; adjust smoothing if desired.
- Run all cells to generate plots (SVG) and data tables (CSV) in the figure folder or configured output paths.

## Outputs

- *.csv: projection or intensity data (often raw and normalized).
- *.svg: publication-ready plots.
- Straightening: single-frame or stack TIFFs per channel in OUTPUT_DIR.
