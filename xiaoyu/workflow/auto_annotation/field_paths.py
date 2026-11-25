"""Map final TIF fields to images under Final/OriData.

The annotation directory defines the active field set, so TIFs without a final
annotation are not processed by the packaged pipeline.
"""
from pathlib import Path

from common import DATA

ANN = DATA.parents[1] / "annotations"


def _tag(rel):
    folder = rel.parts[0]
    stem = rel.stem
    if folder == "ficoll_vs_protrusion":
        condition = stem.split("-", 1)[0]
    else:
        condition = rel.parts[1]
    return f"{condition}_{stem}"


TIF_BY_TAG = {}
for ann in sorted(ANN.rglob("*.csv")):
    rel = ann.relative_to(ANN)
    tif = DATA / rel.with_suffix(".TIF")
    if tif.exists():
        TIF_BY_TAG[_tag(rel)] = tif


def all_fields():
    return sorted(TIF_BY_TAG)


def field_image(tag):
    return TIF_BY_TAG[tag]
