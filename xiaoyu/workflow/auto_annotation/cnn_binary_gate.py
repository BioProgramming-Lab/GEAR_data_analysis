"""Binary GUV/not-GUV gate and binary attached/none type model.

Gate fusion: reject if CNN p_not_guv >= 0.5 OR center-detector score < 0.12.
The score threshold is a fixed design point from the score diagnostic.

Usage: cnn_binary_gate.py [--epochs 30] [--seeds 0,1,2]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

import cnn_stage2

PROJ = Path(__file__).resolve().parent.parent
REVIEW = PROJ / "results" / "cnn_v2" / "hardneg_review"
CENT = PROJ / "results" / "center_detector_v2"
IMAGES = PROJ / "labels" / "registry" / "images"
OUT = PROJ / "results" / "cnn_v2_binary"
SCORE_THRESHOLD = 0.12


class BinaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        m = mobilenet_v3_small(weights=None)
        self.features = m.features
        for p in self.features.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(576, 1))

    def forward(self, x):
        f = self.features(x)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.head(f).squeeze(1)


def split_val(samples, frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in (0, 1):
        group = [s for s in samples if s["label"] == label]
        rng.shuffle(group)
        n = max(1, int(round(len(group) * frac))) if len(group) > 1 else 0
        val.extend(group[:n])
        train.extend(group[n:])
    return train, val


def fit_temperature(logits, labels):
    logits = torch.tensor(logits, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    def nll(logt):
        t = float(np.exp(logt))
        p = torch.sigmoid(logits / t).clamp(1e-6, 1 - 1e-6)
        return float(F.binary_cross_entropy(p, labels))

    res = minimize_scalar(nll, bounds=(-2.0, 2.0), method="bounded")
    return float(np.exp(res.x))


def train_binary(train_samples, val_samples, seeds, args, device):
    models, temperatures = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        npos = sum(1 for s in train_samples if s["label"] == 1)
        nneg = len(train_samples) - npos
        pos_weight = torch.tensor([max(nneg / max(npos, 1), 1.0)], device=device)
        lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        model = BinaryNet().to(device)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=1e-4,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        train_loader = DataLoader(
            cnn_stage2.CropDataset(train_samples, "mobilenet", True),
            batch_size=args.batch, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            cnn_stage2.CropDataset(val_samples, "mobilenet", False),
            batch_size=args.batch, shuffle=False, num_workers=0,
        )
        best = (1e9, None)
        for epoch in range(args.epochs):
            model.train()
            model.features.eval()
            for x, y, _ in train_loader:
                x, y = x.to(device), y.float().to(device)
                opt.zero_grad()
                loss = lossfn(model(x), y)
                loss.backward()
                opt.step()
            sched.step()
            model.eval()
            total, n = 0.0, 0
            with torch.no_grad():
                for x, y, _ in val_loader:
                    x, y = x.to(device), y.float().to(device)
                    loss = lossfn(model(x), y)
                    total += float(loss.item()) * len(y)
                    n += len(y)
            val = total / max(n, 1)
            if val < best[0]:
                best = (val, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        model.load_state_dict(best[1])
        model.eval()
        logits, labels = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                logits.extend(model(x.to(device)).cpu().numpy().tolist())
                labels.extend(y.numpy().tolist())
        temperatures.append(fit_temperature(logits, labels))
        models.append(model)
    return models, temperatures


def predict_logits(models, paths, batch, device):
    arrays = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    tf = cnn_stage2.transforms_for("mobilenet", False)
    out = []
    for i in range(0, len(arrays), batch):
        xs = []
        for arr in arrays[i:i + batch]:
            im = Image.fromarray(arr).convert("RGB")
            if im.size != (128, 128):
                im = im.resize((128, 128), Image.BILINEAR)
            xs.append(tf(im))
        x = torch.stack(xs).to(device)
        with torch.no_grad():
            out.append(np.stack([m(x).cpu().numpy() for m in models]))
    return np.concatenate(out, axis=1) if out else np.zeros((len(models), 0))


def predict_mean(models, temps, paths, batch, device):
    logits = predict_logits(models, paths, batch, device)
    probs = 1.0 / (1.0 + np.exp(-logits / np.array(temps)[:, None]))
    return probs.mean(0)


def load_review():
    decisions = json.load(open(REVIEW / "hardneg_labels.json"))["decisions"]
    cands = {r["id"]: r for r in csv.DictReader(open(REVIEW / "candidates.csv"))}
    score_cache = {}
    for image in ("10", "12", "13"):
        props = np.loadtxt(CENT / f"test_{image}" / "proposals.csv", delimiter=",", skiprows=1)
        score_cache[image] = (props[:, :2], props[:, 3])
    out = []
    for cid, label in decisions.items():
        if label not in ("guv", "not_guv") or cid not in cands:
            continue
        c = cands[cid]
        pts, scores = score_cache[c["image"]]
        j = int(np.argmin(np.hypot(pts[:, 0] - float(c["x"]), pts[:, 1] - float(c["y"]))))
        out.append({
            "id": cid, "image": c["image"],
            "path": str(REVIEW / "crops128" / f"{cid}.png"),
            "expected": 1 if label == "not_guv" else 0,
            "score": float(scores[j]),
        })
    return out


def eval_image(gate, gt, typ, tt, image, manual, review, args, device):
    green = np.asarray(Image.open(IMAGES / f"{image}.png"))
    props = np.loadtxt(CENT / f"test_{image}" / "proposals.csv", delimiter=",", skiprows=1)
    paths = []
    crop_dir = OUT / "proposal_crops" / image
    crop_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(props):
        arr = cnn_stage2.crop_green(green, p[0], p[1], p[2])
        path = crop_dir / f"{i:04d}.png"
        Image.fromarray(arr).save(path)
        paths.append(str(path))
    p_gate = predict_mean(gate, gt, paths, args.batch, device)
    p_type = predict_mean(typ, tt, paths, args.batch, device)
    manual_im = [m for m in manual if m["image"] == image]
    mtree = cKDTree(np.array([[m["x"], m["y"]] for m in manual_im]))
    rows = []
    for i, p in enumerate(props):
        d, j = mtree.query([p[0], p[1]])
        m = manual_im[j]
        rows.append({
            "image": image, "x": p[0], "y": p[1], "r": p[2], "score": p[3],
            "p_gate": p_gate[i], "p_type": p_type[i],
            "matched": int(d <= m["r"]), "manual_id": m["id"] if d <= m["r"] else "",
            "manual_label": m["label"] if d <= m["r"] and m["label"] is not None else "",
            "offset_r": float(d / m["r"]),
        })
    rev = [r for r in review if r["image"] == image]
    p_rev = predict_mean(gate, gt, [r["path"] for r in rev], args.batch, device)
    for r, p in zip(rev, p_rev):
        r["p_gate"] = p
    centered = [m for m in manual_im if m["label"] in (0, 1)]
    carr = [cnn_stage2.crop_green(green, m["x"], m["y"], m["r"]) for m in centered]
    cdir = OUT / "centered_crops" / image
    cdir.mkdir(parents=True, exist_ok=True)
    cpaths = []
    for i, (m, arr) in enumerate(zip(centered, carr)):
        path = cdir / f"{i:04d}.png"
        Image.fromarray(arr).save(path)
        cpaths.append(str(path))
    p_cent = predict_mean(typ, tt, cpaths, args.batch, device)

    modes = {
        "cnn": lambda row: row["p_gate"] >= 0.5,
        "score": lambda row: row["score"] < SCORE_THRESHOLD,
        "combined": lambda row: row["p_gate"] >= 0.5 or row["score"] < SCORE_THRESHOLD,
    }
    eval_out = {}
    for name, reject in modes.items():
        matched = [r for r in rows if r["matched"] and r["manual_label"] in (0, 1)]
        unmatched = [r for r in rows if not r["matched"]]
        typed = [r for r in matched if not reject(r)]
        e2e = 0
        for m in manual_im:
            if m["label"] not in (0, 1):
                continue
            cand = [r for r in rows if r["matched"] and r["manual_id"] == m["id"]]
            if not cand:
                continue
            best = min(cand, key=lambda r: r["offset_r"])
            if not reject(best) and (best["p_type"] >= 0.5) == bool(m["label"]):
                e2e += 1
        not_rej = [reject(r) for r in rev if r["expected"] == 1]
        guv_rej = [reject(r) for r in rev if r["expected"] == 0]
        eval_out[name] = {
            "matched_gate_reject_rate": float(np.mean([reject(r) for r in matched])) if matched else None,
            "unmatched_gate_reject_rate": float(np.mean([reject(r) for r in unmatched])) if unmatched else None,
            "matched_type_accuracy": float(np.mean([
                (r["p_type"] >= 0.5) == bool(r["manual_label"]) for r in typed
            ])) if typed else None,
            "e2e_nearest_type_recall": float(e2e / max(len([m for m in manual_im if m["label"] in (0, 1)]), 1)),
            "reviewed_not_guv_reject_rate": float(np.mean(not_rej)) if not_rej else None,
            "reviewed_guv_false_reject_rate": float(np.mean(guv_rej)) if guv_rej else None,
        }
    centered_acc = float(np.mean([
        (p >= 0.5) == bool(m["label"]) for m, p in zip(centered, p_cent)
    ])) if centered else None
    type_auc_centered = cnn_stage2.auc_score(
        np.array([m["label"] for m in centered]), p_cent,
    ) if centered else None
    return {
        "image": image,
        "n_proposals": int(len(props)),
        "n_reviewed": len(rev),
        "centered_type_accuracy": centered_acc,
        "centered_type_auc": type_auc_centered,
        "modes": eval_out,
    }


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}")

    cnn_stage2.OUT = PROJ / "results" / "cnn_v2"
    a = cnn_stage2.load_samples()
    gate_samples = [
        dict(s, label=1 if s["label"] == 2 else 0) for s in a
    ]
    type_samples = [dict(s, label=1 if s["label"] == 1 else 0) for s in a if s["label"] in (0, 1)]
    gtrain, gval = split_val(gate_samples, 0.15, seed=0)
    ttrain, tval = split_val(type_samples, 0.15, seed=0)
    print(f"gate train={len(gtrain)} val={len(gval)} type train={len(ttrain)} val={len(tval)}")

    gate, gt = train_binary(gtrain, gval, seeds, args, device)
    typ, tt = train_binary(ttrain, tval, seeds, args, device)
    print("gate temperatures", gt)
    print("type temperatures", tt)

    manual = cnn_stage2.load_manual()
    review = load_review()
    results = {"score_threshold": SCORE_THRESHOLD, "seeds": seeds,
               "gate_temperatures": gt, "type_temperatures": tt, "images": {}}
    for image in ("10", "12", "13"):
        r = eval_image(gate, gt, typ, tt, image, manual, review, args, device)
        results["images"][image] = r
        print(f"\n{image}: centered type acc={r['centered_type_accuracy']:.3f}")
        for name, m in r["modes"].items():
            print(f"  {name}: matched_reject={m['matched_gate_reject_rate']} "
                  f"unmatched_reject={m['unmatched_gate_reject_rate']} "
                  f"type={m['matched_type_accuracy']} "
                  f"e2e={m['e2e_nearest_type_recall']:.3f} "
                  f"rev_not={m['reviewed_not_guv_reject_rate']} "
                  f"rev_guv_false={m['reviewed_guv_false_reject_rate']}")

    def pool(name, key):
        vals = []
        weights = []
        for r in results["images"].values():
            v = r["modes"][name][key]
            if v is None:
                continue
            vals.append(v)
            weights.append(r["n_proposals"] if "gate_reject" in key or "type_accuracy" in key else r["n_reviewed"])
        return float(np.average(vals, weights=weights)) if vals else None

    results["pooled"] = {
        name: {key: pool(name, key) for key in (
            "matched_gate_reject_rate", "unmatched_gate_reject_rate",
            "matched_type_accuracy", "e2e_nearest_type_recall",
            "reviewed_not_guv_reject_rate", "reviewed_guv_false_reject_rate",
        )} for name in ("cnn", "score", "combined")
    }
    results["centered_type_accuracy"] = float(np.mean([
        r["centered_type_accuracy"] for r in results["images"].values()
        if r["centered_type_accuracy"] is not None
    ]))
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    print("\npooled:", json.dumps(results["pooled"], indent=2))
    print(f"\nwrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
