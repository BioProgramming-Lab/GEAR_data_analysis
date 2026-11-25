"""Train and evaluate the 3-class attachment gate on GUV crops.

Classes: 0 = no_filament, 1 = attached, 2 = not_guv.

Training data:
- DataLabel/attachment/labels.csv (671 crops)
- DataLabel/attachment/not_guv.csv (79 crops generated with the same 4r crop)

Evaluation:
- field-wise OOF on the 7 reviewed fields
- external 3-image set using center_detector_v2 proposals
- centered manual crops for classifier-only metrics

Usage:
  cnn_stage2.py [--epochs 40] [--models small,mobilenet]
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_green

PROJ = Path(__file__).resolve().parent.parent
GEAR = PROJ.parent
DATALABEL = GEAR / "DataLabel"
REG = PROJ / "labels" / "registry"
IMAGES = REG / "images"
CENT = PROJ / "results" / "center_detector_v2"
OUT = PROJ / "results" / "cnn_v2"
CLASSES = ("none", "attached", "not_guv")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def safe(text):
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", str(text)).strip("_")[:120]


def crop_green(green, cx, cy, r):
    side = int(round(max(64.0, min(384.0, 4.0 * float(r)))))
    x0 = int(round(float(cx) - side / 2))
    y0 = int(round(float(cy) - side / 2))
    x1, y1 = x0 + side, y0 + side
    h, w = green.shape
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - w), max(0, y1 - h)
    tile = green[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if pl or pt or pr or pb:
        tile = np.pad(tile, ((pt, pb), (pl, pr)), mode="edge")
    tile = np.asarray(Image.fromarray(tile).resize((128, 128), Image.BILINEAR))
    return np.stack([tile] * 3, -1)


def load_explicit():
    rows = list(csv.DictReader(open(DATALABEL / "attachment" / "labels" / "body_decisions.csv")))
    return {(r["field"], str(r["body_id"])): bool(r.get("user_decision"))
            for r in rows}


def load_samples():
    explicit_map = load_explicit()
    samples = []
    for r in csv.DictReader(open(DATALABEL / "attachment" / "labels.csv")):
        samples.append({
            "path": str(DATALABEL / r["image"]),
            "label": 1 if r["label"] == "attached" else 0,
            "field": r["field"],
            "body_id": str(r["body_id"]),
            "explicit": int(explicit_map.get((r["field"], str(r["body_id"])), False)),
            "source": "attachment",
        })
    crop_dir = OUT / "crops" / "not_guv"
    crop_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    for r in csv.DictReader(open(DATALABEL / "attachment" / "not_guv.csv")):
        field = r["field"]
        if field not in cache:
            cache[field] = load_green(r["source_tif"])
        arr = crop_green(cache[field], r["cx_px"], r["cy_px"], r["r_px"])
        path = crop_dir / f"{safe(field)}_{safe(r['body_id'])}.png"
        Image.fromarray(arr).save(path)
        samples.append({
            "path": str(path),
            "label": 2,
            "field": field,
            "body_id": str(r["body_id"]),
            "explicit": 1,
            "source": "not_guv",
        })
    return samples


def transforms_for(model_name, train):
    if train:
        ops = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation((90, 90), fill=0)], p=0.25),
            transforms.RandomApply([transforms.RandomRotation((180, 180), fill=0)], p=0.25),
            transforms.RandomApply([transforms.RandomRotation((270, 270), fill=0)], p=0.25),
            transforms.RandomAffine(degrees=0, translate=(0.10, 0.10),
                                    scale=(0.9, 1.1), fill=0),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ]
    else:
        ops = [transforms.ToTensor()]
    if model_name == "mobilenet":
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    else:
        ops.append(transforms.Normalize([0.5], [0.25]))
    return transforms.Compose(ops)


class CropDataset(Dataset):
    def __init__(self, samples, model_name, train):
        self.samples = samples
        self.model_name = model_name
        self.tf = transforms_for(model_name, train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        im = Image.open(s["path"])
        im = im.convert("L" if self.model_name == "small" else "RGB")
        if im.size != (128, 128):
            im = im.resize((128, 128), Image.BILINEAR)
        x = self.tf(im)
        return x, s["label"], idx


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 3))

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def make_model(name, device):
    if name == "small":
        return SmallCNN().to(device)
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(576, 3))
    return model.to(device)


def model_parameters(model, name):
    if name == "small":
        return [p for p in model.parameters() if p.requires_grad]
    return [p for p in model.parameters() if p.requires_grad]


def split_val(samples, frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in (0, 1, 2):
        group = [s for s in samples if s["label"] == label]
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * frac))) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val


def train_fold(train_samples, test_samples, model_name, args, device):
    train_samples, val_samples = split_val(train_samples, 0.15, args.seed)
    counts = np.bincount([s["label"] for s in train_samples], minlength=3).astype(float)
    weights = counts.sum() / np.clip(counts, 1, None)
    weights = weights / weights.mean()
    lossfn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=device))
    model = make_model(model_name, device)
    opt = torch.optim.AdamW(model_parameters(model, model_name), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    train_loader = DataLoader(CropDataset(train_samples, model_name, True),
                              batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(CropDataset(val_samples, model_name, False),
                            batch_size=args.batch, shuffle=False, num_workers=0)
    best = (1e9, None)
    for epoch in range(args.epochs):
        model.train()
        if model_name == "mobilenet":
            model.features.eval()
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossfn(model(x), y)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                loss = lossfn(model(x), y)
                total += float(loss.item()) * len(y)
                n += len(y)
        val_loss = total / max(n, 1)
        if val_loss < best[0]:
            best = (val_loss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    model.eval()
    probs, labels = predict_loader(model, test_samples, model_name, args.batch, device)
    return model, probs, labels, best[0]


def predict_loader(model, samples, model_name, batch, device):
    loader = DataLoader(CropDataset(samples, model_name, False),
                        batch_size=batch, shuffle=False, num_workers=0)
    probs, labels = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            probs.append(torch.softmax(model(x), dim=1).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def predict_arrays(models, arrays, model_name, batch, device):
    tf = transforms_for(model_name, False)
    outs = []
    for i in range(0, len(arrays), batch):
        chunk = arrays[i:i + batch]
        xs = []
        for arr in chunk:
            im = Image.fromarray(arr).convert("L" if model_name == "small" else "RGB")
            if im.size != (128, 128):
                im = im.resize((128, 128), Image.BILINEAR)
            xs.append(tf(im))
        x = torch.stack(xs).to(device)
        with torch.no_grad():
            probs = [torch.softmax(model(x), dim=1).cpu().numpy() for model in models]
        outs.append(np.mean(probs, axis=0))
    return np.concatenate(outs) if outs else np.zeros((0, 3))


def auc_score(y, score):
    y = np.asarray(y)
    score = np.asarray(score)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, score):
    y = np.asarray(y)
    score = np.asarray(score)
    if y.sum() == 0:
        return None
    order = np.argsort(-score)
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def multiclass_metrics(y, prob):
    pred = prob.argmax(1)
    cm = np.zeros((3, 3), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    per = {}
    f1s = []
    for c in range(3):
        tp = cm[c, c]
        prec = tp / max(cm[:, c].sum(), 1)
        rec = tp / max(cm[c, :].sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        per[CLASSES[c]] = {"precision": float(prec), "recall": float(rec), "f1": float(f1)}
        f1s.append(f1)
    guv = np.isin(y, [0, 1])
    type_auc = None
    if guv.any():
        type_auc = auc_score((y[guv] == 1).astype(int),
                             prob[guv, 1] / np.clip(prob[guv, 0] + prob[guv, 1], 1e-9, None))
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per,
        "confusion": cm.tolist(),
        "gate_auc": auc_score((y == 2).astype(int), prob[:, 2]),
        "gate_ap": average_precision((y == 2).astype(int), prob[:, 2]),
        "type_auc": type_auc,
        "type_ap": average_precision((y[guv] == 1).astype(int),
                                     prob[guv, 1] / np.clip(prob[guv, 0] + prob[guv, 1], 1e-9, None)) if guv.any() else None,
        "predicted": {CLASSES[c]: int((pred == c).sum()) for c in range(3)},
    }


def load_manual():
    rows = []
    for r in csv.DictReader(open(REG / "source_b_fullimage_annotations.csv")):
        edge = 1 if r["label_detail"] == "edge" else int(r["edge"])
        if edge or r["label_detail"] == "uncertain":
            continue
        rows.append({
            "image": r["group"], "id": r["object_id"],
            "x": float(r["cx"]), "y": float(r["cy"]), "r": float(r["r"]),
            "detail": r["label_detail"],
            "label": 1 if r["label_detail"] == "attached_clear" else 0
            if r["label_detail"] == "none" else None,
        })
    return rows


def eval_external(models, model_name, manual, device, args, images=("10", "12", "13")):
    greens = {im: np.asarray(Image.open(IMAGES / f"{im}.png")) for im in images}
    rows = []
    centered = []
    for im, green in greens.items():
        props = np.loadtxt(CENT / f"test_{im}" / "proposals.csv", delimiter=",", skiprows=1)
        if props.ndim == 1:
            props = props[None]
        arrays = [crop_green(green, p[0], p[1], p[2]) for p in props]
        probs = predict_arrays(models, arrays, model_name, args.batch, device) if arrays else np.zeros((0, 3))
        manual_im = [m for m in manual if m["image"] == im]
        mtree = cKDTree(np.array([[m["x"], m["y"]] for m in manual_im]))
        for i, p in enumerate(props):
            d, j = mtree.query([p[0], p[1]])
            m = manual_im[j]
            matched = d <= m["r"]
            rows.append({
                "image": im, "x": p[0], "y": p[1], "r": p[2],
                "matched": int(matched),
                "manual_id": m["id"] if matched else "",
                "manual_label": "" if not matched or m["label"] is None else m["label"],
                "offset_r": float(d / m["r"]),
                "p_none": probs[i, 0], "p_attached": probs[i, 1], "p_not_guv": probs[i, 2],
                "pred": int(probs[i].argmax()),
            })
        carr = [crop_green(green, m["x"], m["y"], m["r"]) for m in manual_im]
        cprob = predict_arrays(models, carr, model_name, args.batch, device) if carr else np.zeros((0, 3))
        for m, p in zip(manual_im, cprob):
            centered.append({
                "image": im, "id": m["id"], "manual_label": m["label"],
                "p_none": p[0], "p_attached": p[1], "p_not_guv": p[2],
                "pred": int(p.argmax()),
            })
    out_dir = OUT / f"external_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "proposals.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "centered.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(centered[0].keys()))
        w.writeheader()
        w.writerows(centered)

    matched = [r for r in rows if r["matched"] and r["manual_label"] in (0, 1)]
    unmatched = [r for r in rows if not r["matched"]]
    typed = [r for r in matched if r["pred"] != 2]
    manual_guv = [m for m in manual if m["label"] in (0, 1) and m["image"] in images]
    e2e_any = 0
    e2e_nearest = 0
    for m in manual_guv:
        cand = [r for r in rows if r["image"] == m["image"] and r["matched"]
                and r["manual_id"] == m["id"]]
        if not cand:
            continue
        if any(r["pred"] == m["label"] for r in cand):
            e2e_any += 1
        best = min(cand, key=lambda r: r["offset_r"])
        if best["pred"] == m["label"]:
            e2e_nearest += 1
    centered_typed = [r for r in centered if r["manual_label"] in (0, 1) and r["pred"] != 2]
    return {
        "matched_proposals": len(matched),
        "unmatched_proposals": len(unmatched),
        "matched_gate_reject_rate": float(np.mean([r["pred"] == 2 for r in matched])) if matched else None,
        "unmatched_gate_reject_rate": float(np.mean([r["pred"] == 2 for r in unmatched])) if unmatched else None,
        "matched_type_accuracy": float(np.mean([r["pred"] == r["manual_label"] for r in typed])) if typed else None,
        "e2e_any_recall": float(e2e_any / max(len(manual_guv), 1)),
        "e2e_nearest_recall": float(e2e_nearest / max(len(manual_guv), 1)),
        "centered_gate_reject_rate": float(np.mean([r["pred"] == 2 for r in centered])) if centered else None,
        "centered_type_accuracy": float(np.mean([r["pred"] == r["manual_label"] for r in centered_typed])) if centered_typed else None,
        "n_manual_guv": len(manual_guv),
    }


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="small,mobilenet")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}")
    samples = load_samples()
    fields = sorted({s["field"] for s in samples})
    print(f"{len(samples)} samples, {len(fields)} fields, "
          f"labels={np.bincount([s['label'] for s in samples], minlength=3).tolist()}")
    manual = load_manual()
    summary = {"samples": len(samples), "fields": fields, "models": {}}

    for model_name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {model_name} ===")
        oof_prob = np.zeros((len(samples), 3))
        oof_label = np.array([s["label"] for s in samples])
        fold_models = []
        fold_info = {}
        for field in fields:
            train = [s for s in samples if s["field"] != field]
            test = [s for s in samples if s["field"] == field]
            idx = [i for i, s in enumerate(samples) if s["field"] == field]
            t0 = time.time()
            model, prob, labels, val = train_fold(train, test, model_name, args, device)
            oof_prob[idx] = prob
            fold_models.append(model)
            m = multiclass_metrics(labels, prob)
            fold_info[field] = {"n": len(test), "metrics": m,
                                "val_loss": float(val),
                                "seconds": round(time.time() - t0, 1)}
            print(f"  {field}: n={len(test)} acc={m['accuracy']:.3f} "
                  f"f1={m['macro_f1']:.3f} gate_auc={m['gate_auc']} "
                  f"({time.time()-t0:.0f}s)")
        metrics = multiclass_metrics(oof_label, oof_prob)
        explicit = np.array([bool(s["explicit"]) for s in samples])
        explicit_metrics = multiclass_metrics(oof_label[explicit], oof_prob[explicit])
        pred = oof_prob.argmax(1)
        with open(OUT / f"oof_{model_name}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["field", "body_id", "label", "explicit", "p_none",
                        "p_attached", "p_not_guv", "pred"])
            for s, p, y in zip(samples, oof_prob, pred):
                w.writerow([s["field"], s["body_id"], s["label"], s["explicit"],
                            round(float(p[0]), 6), round(float(p[1]), 6),
                            round(float(p[2]), 6), int(y)])
        print(f"  OOF: acc={metrics['accuracy']:.3f} f1={metrics['macro_f1']:.3f} "
              f"gate_auc={metrics['gate_auc']:.3f} type_auc={metrics['type_auc']}")
        print(f"  explicit: n={explicit_metrics['n']} acc={explicit_metrics['accuracy']:.3f}")
        ext = eval_external(fold_models, model_name, manual, device, args)
        print(f"  external: e2e_any={ext['e2e_any_recall']:.3f} "
              f"e2e_nearest={ext['e2e_nearest_recall']:.3f} "
              f"type_acc={ext['matched_type_accuracy']} "
              f"gate_reject={ext['matched_gate_reject_rate']}")
        summary["models"][model_name] = {
            "oof": metrics,
            "explicit": explicit_metrics,
            "folds": fold_info,
            "external": ext,
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
