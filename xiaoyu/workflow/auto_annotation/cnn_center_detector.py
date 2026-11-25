"""Prototype a single center + radius detector for GUV proposals.

Trains a frozen MobileNetV3-Small FPN on manual circles and evaluates
leave-one-image-out on the three-image set.

Usage:
  cnn_center_detector.py [--test-image 13] [--test-image all]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from skimage.exposure import equalize_adapthist
from skimage.feature import peak_local_max
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_green

PROJ = Path(__file__).resolve().parent
REG = PROJ / "labels" / "registry"
IMAGES = REG / "images"
MANUAL = REG / "source_b_fullimage_annotations.csv"
DATALABEL = PROJ.parent / "DataLabel"
OUT = PROJ / "results" / "center_detector_v2"
STRIDE = 4
SIGMA = 2.0


class MobileNetFPN(nn.Module):
    def __init__(self, unfreeze="none"):
        super().__init__()
        m = mobilenet_v3_small(weights=None)
        self.b1 = m.features[0:2]
        self.b2 = m.features[2:4]
        self.b3 = m.features[4:9]
        self.b4 = m.features[9:13]
        for p in self.parameters():
            p.requires_grad = False
        if unfreeze in ("b4", "b3b4"):
            for p in self.b4.parameters():
                p.requires_grad = True
        if unfreeze == "b3b4":
            for p in self.b3.parameters():
                p.requires_grad = True
        self.l4 = nn.Conv2d(576, 64, 1)
        self.l3 = nn.Conv2d(48, 64, 1)
        self.l2 = nn.Conv2d(24, 64, 1)
        self.l1 = nn.Conv2d(16, 64, 1)
        self.s3 = nn.Conv2d(64, 64, 3, padding=1)
        self.s2 = nn.Conv2d(64, 64, 3, padding=1)
        self.s1 = nn.Conv2d(64, 64, 3, padding=1)
        self.head = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                  nn.ReLU(inplace=True))
        self.heat = nn.Conv2d(64, 1, 1)
        self.rad = nn.Conv2d(64, 1, 1)
        self.off = nn.Conv2d(64, 2, 1)
        nn.init.constant_(self.heat.bias, -4.0)
        nn.init.constant_(self.rad.bias, 3.0)
        nn.init.constant_(self.off.bias, 0.0)

    def encoder_parameters(self):
        return [p for module in (self.b1, self.b2, self.b3, self.b4)
                for p in module.parameters() if p.requires_grad]

    def train(self, mode=True):
        super().train(mode)
        for module in (self.b1, self.b2, self.b3, self.b4):
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x):
        c1 = self.b1(x)
        c2 = self.b2(c1)
        c3 = self.b3(c2)
        c4 = self.b4(c3)
        p4 = self.l4(c4)
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.s3(p3)
        p2 = self.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p2 = self.s2(p2)
        p1 = self.l1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="nearest")
        p1 = self.s1(p1)
        f = self.head(p1)
        return self.heat(f), self.rad(f), self.off(f)


def load_manual():
    rows = []
    for r in csv.DictReader(open(MANUAL)):
        if r["label_detail"] == "edge":
            edge = 1
        else:
            edge = int(r["edge"])
        if edge or r["label_detail"] == "uncertain":
            continue
        rows.append({
            "image": r["group"],
            "id": r["object_id"],
            "x": float(r["cx"]),
            "y": float(r["cy"]),
            "r": float(r["r"]),
            "detail": r["label_detail"],
            "edge": 0,
        })
    return rows


def normalize(green):
    green = green.astype(np.float32)
    lo, hi = np.percentile(green, (1.0, 99.0))
    return np.clip((green - lo) / max(hi - lo, 1e-6), 0, 1)


def load_images(paths):
    return {key: normalize(load_green(path)) for key, path in paths.items()}


def load_all_data(include_a):
    manual = load_manual()
    b_images = sorted({r["image"] for r in manual})
    paths = {im: IMAGES / f"{im}.png" for im in b_images}
    centers = {im: [[r["x"], r["y"]] for r in manual if r["image"] == im]
               for im in b_images}
    radii = {im: [r["r"] for r in manual if r["image"] == im] for im in b_images}
    negatives = {im: [] for im in b_images}
    if include_a:
        labels = DATALABEL / "attachment" / "labels.csv"
        for r in csv.DictReader(open(labels)):
            key = r["field"]
            paths[key] = Path(r["source_tif"])
            centers.setdefault(key, []).append([float(r["cx_px"]), float(r["cy_px"])])
            radii.setdefault(key, []).append(float(r["r_px"]))
            negatives.setdefault(key, [])
        notg = DATALABEL / "attachment" / "not_guv.csv"
        for r in csv.DictReader(open(notg)):
            key = r["field"]
            negatives.setdefault(key, []).append([float(r["cx_px"]), float(r["cy_px"])])
    images = load_images(paths)
    centers = {k: np.array(v, np.float32) for k, v in centers.items()}
    radii = {k: np.array(v, np.float32) for k, v in radii.items()}
    negatives = {k: np.array(v, np.float32) for k, v in negatives.items()}
    return manual, images, centers, radii, negatives


def make_targets(centers, radii, size):
    ho, wo = size // STRIDE, size // STRIDE
    heat = np.zeros((ho, wo), np.float32)
    rad = np.zeros((ho, wo), np.float32)
    off = np.zeros((2, ho, wo), np.float32)
    yy, xx = np.mgrid[0:ho, 0:wo]
    for (cx, cy), r in zip(centers, radii):
        ox, oy = cx / STRIDE, cy / STRIDE
        x0 = int(max(0, ox - 3 * SIGMA))
        x1 = int(min(wo, ox + 3 * SIGMA + 1))
        y0 = int(max(0, oy - 3 * SIGMA))
        y1 = int(min(ho, oy + 3 * SIGMA + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        ix, iy = int(round(ox)), int(round(oy))
        if not (0 <= ix < wo and 0 <= iy < ho):
            continue
        g = np.exp(-((xx[y0:y1, x0:x1] - ix) ** 2 +
                     (yy[y0:y1, x0:x1] - iy) ** 2) / (2 * SIGMA ** 2))
        np.maximum(heat[y0:y1, x0:x1], g, out=heat[y0:y1, x0:x1])
        heat[iy, ix] = 1.0
        rad[iy, ix] = np.log(max(r, 1.0))
        off[0, iy, ix] = ox - ix
        off[1, iy, ix] = oy - iy
    return heat, rad, off


def sample_crop(img, centers, radii, size, rng, augment=True, negatives=None,
                rotation=True):
    h, w = img.shape
    use_negative = negatives is not None and len(negatives) > 0 and rng.random() < 0.2
    crop = None
    sub_centers = None
    sub_radii = None
    for _ in range(30):
        if use_negative:
            cx, cy = negatives[int(rng.integers(0, len(negatives)))]
        else:
            if len(centers) == 0:
                break
            cx, cy = centers[int(rng.integers(0, len(centers)))]
        x0 = int(round(cx - size / 2 + rng.uniform(-size / 4, size / 4)))
        y0 = int(round(cy - size / 2 + rng.uniform(-size / 4, size / 4)))
        x0 = int(np.clip(x0, 0, w - size))
        y0 = int(np.clip(y0, 0, h - size))
        crop = img[y0:y0 + size, x0:x0 + size].copy()
        local = centers - np.array([x0, y0])
        keep = ((local[:, 0] >= 2) & (local[:, 0] < size - 2) &
                (local[:, 1] >= 2) & (local[:, 1] < size - 2))
        if use_negative or keep.sum() >= 1:
            sub_centers = local[keep]
            sub_radii = radii[keep]
            break
    if crop is None or sub_centers is None:
        x0 = int(rng.integers(0, max(1, w - size)))
        y0 = int(rng.integers(0, max(1, h - size)))
        crop = img[y0:y0 + size, x0:x0 + size].copy()
        local = centers - np.array([x0, y0])
        keep = ((local[:, 0] >= 2) & (local[:, 0] < size - 2) &
                (local[:, 1] >= 2) & (local[:, 1] < size - 2))
        sub_centers = local[keep]
        sub_radii = radii[keep]
    if augment:
        k = int(rng.integers(0, 4)) if rotation else 0
        if k:
            crop = np.rot90(crop, k).copy()
            for _ in range(k):
                x = sub_centers[:, 0].copy()
                sub_centers[:, 0] = sub_centers[:, 1]
                sub_centers[:, 1] = size - 1 - x
        if rng.random() < 0.5:
            crop = crop[::-1]
            sub_centers[:, 1] = size - 1 - sub_centers[:, 1]
        if rng.random() < 0.5:
            crop = crop[:, ::-1]
            sub_centers[:, 0] = size - 1 - sub_centers[:, 0]
        crop = np.clip(crop * rng.uniform(0.8, 1.2) + rng.uniform(-0.05, 0.05), 0, 1)
        if rng.random() < 0.3:
            crop = np.clip(crop ** rng.uniform(0.8, 1.2), 0, 1)
        if rng.random() < 0.5:
            crop = equalize_adapthist(
                crop, clip_limit=float(rng.uniform(0.005, 0.02)),
                kernel_size=int(rng.choice([64, 96, 128])),
            ).astype(np.float32)
    heat, rad, off = make_targets(sub_centers, sub_radii, size)
    return crop.astype(np.float32), heat, rad, off


def focal_loss(logits, target):
    pred = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1 - pos
    neg_w = (1 - target).pow(4)
    pos_loss = -torch.log(pred) * pos * (1 - pred).pow(2)
    neg_loss = -torch.log(1 - pred) * neg_w * neg * pred.pow(2)
    npos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / npos


def radius_loss(pred, target, heat):
    mask = heat > 0.99
    if mask.sum() == 0:
        return pred.sum() * 0.0
    return F.l1_loss(pred[mask], target[mask])


def offset_loss(pred, target, heat):
    mask = (heat > 0.99)[:, 0]
    if mask.sum() == 0:
        return pred.sum() * 0.0
    pred = pred.permute(0, 2, 3, 1)[mask]
    target = target.permute(0, 2, 3, 1)[mask]
    return F.l1_loss(pred, target)


def evaluate_loss(model, batches, device, offset_weight=1.0):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for x, th, tr, to in batches:
            x = x.to(device)
            th = th.to(device)
            tr = tr.to(device)
            to = to.to(device)
            hlog, rlog, olog = model(x)
            loss = (focal_loss(hlog, th) + 0.2 * radius_loss(rlog, tr, th)
                    + offset_weight * offset_loss(olog, to, th))
            total += float(loss.item()) * len(x)
            n += len(x)
    return total / max(n, 1)


def make_batch(samples, device):
    x = torch.from_numpy(np.stack([s[0] for s in samples])).float().unsqueeze(1)
    x = x.repeat(1, 3, 1, 1).to(device)
    th = torch.from_numpy(np.stack([s[1] for s in samples])).float().unsqueeze(1).to(device)
    tr = torch.from_numpy(np.stack([s[2] for s in samples])).float().unsqueeze(1).to(device)
    to = torch.from_numpy(np.stack([s[3] for s in samples])).float().to(device)
    return x, th, tr, to


def fixed_batches(images, centers, radii, size, count, seed, device, batch=4):
    rng = np.random.default_rng(seed)
    names = list(images)
    samples = []
    for i in range(count):
        name = names[i % len(names)]
        c = centers[name]
        r = radii[name]
        samples.append(sample_crop(images[name], c, r, size, rng, augment=False))
    return [make_batch(samples[i:i + batch], device)
            for i in range(0, len(samples), batch)]


def infer_full(model, image, device, threshold=0.1, min_distance=4,
               use_offset=True):
    x = torch.from_numpy(image).float()[None, None].repeat(1, 3, 1, 1).to(device)
    model.eval()
    with torch.no_grad():
        hlog, rlog, olog = model(x)
    heat = torch.sigmoid(hlog)[0, 0].cpu().numpy()
    rad = rlog[0, 0].cpu().numpy()
    off = olog[0].cpu().numpy()
    coords = peak_local_max(
        heat, min_distance=min_distance, threshold_abs=threshold,
        exclude_border=False,
    )
    out = []
    for y, x in coords:
        r = float(np.exp(rad[y, x]))
        if 5 <= r <= 150:
            dx = float(off[0, y, x]) if use_offset else 0.0
            dy = float(off[1, y, x]) if use_offset else 0.0
            out.append(((x + dx) * STRIDE, (y + dy) * STRIDE,
                        r, float(heat[y, x])))
    return np.array(out) if out else np.zeros((0, 4))
