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

## Project Structure, Numbering, and Run Order

- Each figure is stored in its own folder (e.g., Fig2F, Fig4g, FigS7b).
- Folder numbering (typical pattern):
  - `00_*_raw` (or similar): raw input data for that figure.
  - `02_*_straightener_output`: outputs from straightening/cropping steps.
  - `04_*_plot_output` (or `02_*_plot_output` in some ROI-only figures): final plots and CSVs.
- Notebook numbering (typical pattern):
  - `01_*_straightener.ipynb` / `01_*_cropper.ipynb`: preprocessing (straighten/crop/prepare inputs).
  - `03_*_plot.ipynb`: analysis and plotting.

Storage layout example (typical):
- | `FigS7b/`
- | `00_FigS7b_raw/`
- | `00_FigS7b_raw/c1_membrane/`
- | `00_FigS7b_raw/c2_FtsZ/`
- | `01_FigS7b_straightener.ipynb`
- | `02_FigS7b_straightener_output/`
- | `03_FigS7b_plot.ipynb`
- | `04_FigS7b_plot_output/`

Recommended run order:
1) If a figure folder contains a `01_*` notebook, run it first to generate the `02_*` outputs.
2) Run the `03_*_plot` notebook using the generated outputs (or raw data if no straightener is needed).
3) Export figures and CSVs into the `04_*_plot_output` (or configured output) folder.

## Relative Paths

Path variables at the top of each notebook should use `pathlib.Path` with relative paths. Example:

```python
from pathlib import Path

membrane_dir = Path('00_Fig2F_raw/c2_membrane')
output_svg_file = Path('02_Fig2F_plot_output/Fig2F.svg')
```

Relative paths are resolved from the notebook's working directory (typically the figure folder), which keeps the
project portable across machines without editing absolute paths. If you run a notebook from a different working
directory, adjust the paths accordingly.

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
