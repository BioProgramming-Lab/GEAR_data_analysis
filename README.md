# GEAR_data_analysis
GEAR_data_analysis

# EAR_data_analysis –
Quantitative analysis of spatial fluorescence was performed using a custom Python-based pipeline. Filament boundaries were manually annotated in each image using LabelMe (https://github.com/wkentaro/labelme) to generate polygon masks. These masks were then processed with the Skan library to extract filament skeletons, with the longest path defined as the principal axis. The filament image was then computationally straightened along this axis using bilinear interpolation to facilitate the calculation of fluorescence profiles along the principal axis. The straightened axis was divided into N equal-length bins (see analysis code), and integrated fluorescence was calculated for each bin. Fluorescence intensity profiles and kymographs were normalized by dividing all raw intensity values by the global maximum intensity recorded across the entire dataset. This scaled all data between 0 and 1, preserving the relative brightness between individual samples. For visualization, the resulting intensity profiles were smoothed using cubic spline interpolation. All normalization and quantification steps were performed independently for each fluorescence channel.

  ## Environment Setup
  Dependencies are pinned in `environment.yml` (Python 3.10, OpenCV, NumPy/SciPy/Pandas, scikit-image, skan,
  matplotlib, Pillow, ipywidgets, tifffile, etc.).

  ```bash
  # create and activate
  conda env create -f environment.yml
  conda activate GEAR_data_analysis

  # (optional) register kernel for Jupyter
  python -m ipykernel install --user --name GEAR_data_analysis

  Run notebooks in Jupyter Notebook/Lab using the GEAR_data_analysis kernel.

  ## Data Requirements

  - Channel folders typically named c1, c2, ...; files are matched by sorted filename order unless otherwise
    specified in a notebook.
  - ROI/mask: LabelMe-style JSON polygons, usually one JSON per image with the same basename as the image.
  - Straightening: Target channel must have JSON; mapping channels must match dimensions of the target.
  - For time-series stacks (kymographs), provide multi-frame TIFFs per channel.

  ## Typical Workflows

  ### 1) Straightening (if needed)

  - Open the relevant *_Straightener.ipynb (e.g., Fig5g/DIVISION-FIG5/Fig5g_Straightener.ipynb, Fig5f/
    raw_data/-MinCDE/Fig5f01_Straightener.ipynb).
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
