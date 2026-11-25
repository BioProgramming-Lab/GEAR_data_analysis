"""Shared paths and image helpers for the GUV attachment re-analysis."""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

PROJ = Path(__file__).resolve().parents[2]  # Final/
DATA = PROJ / "OriData" / "GUV_filament_quantification"
DNA = DATA / "quantification-DNA_dosage"
RESULTS = PROJ / "results"
DNA_RESULTS = RESULTS / "dna"
MODELS = PROJ / "models"


def dna_tifs():
    return sorted(DNA.glob("*/*d1.TIF"))


TIF_BY_TAG = {f"{p.parent.name}_{p.stem}": p for p in dna_tifs()}


def tag_of(path):
    return f"{path.parent.name}_{path.stem}"


def field_tif(tag):
    return TIF_BY_TAG[tag]


def load_green(path):
    with Image.open(path) as im:
        arr = np.asarray(im)
    return (arr[..., 1] if arr.ndim == 3 else arr).copy()


def burned_in_mask(green):
    h, _ = green.shape
    strip = np.zeros_like(green, dtype=bool)
    strip[int(h * 0.90):, :] = True
    bright = green >= 100
    hline = ndi.binary_opening(bright, structure=np.ones((1, 150), dtype=bool))
    hline &= strip
    return ndi.binary_dilation(hline, structure=np.ones((9, 9), dtype=bool))


def drop_burned_in_labels(masks, green, frac=0.3, max_height=25):
    burned = burned_in_mask(green)
    out = masks.copy()
    removed = []
    for lab in range(1, int(masks.max()) + 1):
        ys, xs = np.where(masks == lab)
        if len(ys) == 0:
            continue
        height = ys.max() - ys.min() + 1
        if height <= max_height and burned[ys, xs].mean() > frac:
            out[ys, xs] = 0
            removed.append(lab)
    remap = np.zeros(int(masks.max()) + 1, dtype=masks.dtype)
    for new, old in enumerate(l for l in range(1, int(masks.max()) + 1)
                              if l not in removed):
        remap[old] = new + 1
    return remap[out], removed
