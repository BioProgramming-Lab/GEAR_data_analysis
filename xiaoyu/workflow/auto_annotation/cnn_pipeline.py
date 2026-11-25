"""Final GUV pipeline: center detector v2 -> score gate -> binary type.

Usage:
  cnn_pipeline.py --all-fields
  cnn_pipeline.py --field "-200osmo_-200-1"
  cnn_pipeline.py --image /path/to/image.TIF --field custom
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))
import cnn_center_detector as cd
import cnn_stage2
import cnn_binary_gate as cg
from field_paths import field_image, all_fields
from common import load_green

OUT = PROJ / "outputs"
SCORE_GATE = 0.12


def load_models(detector_path, attachment_prefix, device):
    det = cd.MobileNetFPN(unfreeze="b4").to(device)
    det.load_state_dict(torch.load(detector_path, map_location=device), strict=False)
    det.eval()
    config = json.loads(Path(attachment_prefix + ".json").read_text())
    models = []
    for i, _ in enumerate(config["seeds"]):
        m = cg.BinaryNet().to(device)
        m.load_state_dict(torch.load(f"{attachment_prefix}_seed{i}.pt", map_location=device))
        m.eval()
        models.append(m)
    return det, models, config["temperatures"]


def predict_type(arrays, models, temps, device, batch=32):
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
            logits = np.stack([m(x).cpu().numpy() for m in models])
        probs = 1.0 / (1.0 + np.exp(-logits / np.array(temps)[:, None]))
        out.append(probs.mean(0))
    return np.concatenate(out) if out else np.zeros(0)


def nms(proposals, dist_factor=0.5):
    if len(proposals) == 0:
        return proposals
    order = np.argsort(-proposals[:, 3])
    keep = []
    for i in order:
        x, y, r, score = proposals[i]
        ok = True
        for px, py, pr, _ in keep:
            if np.hypot(x - px, y - py) < dist_factor * min(r, pr):
                ok = False
                break
        if ok:
            keep.append((x, y, r, score))
    return np.array(keep)


def enhance(green):
    lo, hi = np.percentile(green, (1.0, 99.5))
    x = np.clip((green - lo) / max(hi - lo, 1e-9), 0, 1) ** 0.7
    return np.asarray((x * 255).astype(np.uint8))


def run_field(field, image_path, det, models, temps, device, out_dir, score_gate,
              nms_dist=0.5, type_threshold=0.5):
    green = load_green(image_path)
    props = cd.infer_full(
        det, cd.normalize(green), device, threshold=0.03,
        min_distance=4, use_offset=False,
    )
    if nms_dist > 0:
        props = nms(props, nms_dist)
    rows = []
    accepted = []
    for i, p in enumerate(props):
        x, y, r, score = [float(v) for v in p]
        gate = score >= score_gate
        rows.append({
            "field": field, "proposal_id": i + 1,
            "x": round(x, 2), "y": round(y, 2), "r": round(r, 2),
            "score": round(score, 5), "score_gate": int(gate),
            "p_attached": "", "type": "rejected" if not gate else "",
        })
        if gate:
            accepted.append((i, green, x, y, r))
    arrays = [cnn_stage2.crop_green(g, x, y, r) for _, g, x, y, r in accepted]
    probs = predict_type(arrays, models, temps, device)
    for (i, _, _, _, _), p in zip(accepted, probs):
        rows[i]["p_attached"] = round(float(p), 5)
        rows[i]["type"] = "attached" if p >= type_threshold else "none"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "".join(c if c.isalnum() or c in "._+-" else "_" for c in field)[:100]
    with open(out_dir / f"{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [
            "field", "proposal_id", "x", "y", "r", "score", "score_gate",
            "p_attached", "type",
        ])
        w.writeheader()
        w.writerows(rows)
    base = np.stack([enhance(green)] * 3, -1)
    im = Image.fromarray(base)
    d = ImageDraw.Draw(im)
    for r in rows:
        if r["type"] == "attached":
            color = (255, 70, 70)
        elif r["type"] == "none":
            color = (80, 140, 255)
        else:
            color = (130, 130, 130)
        x, y, rad = r["x"], r["y"], r["r"]
        d.ellipse((x - rad, y - rad, x + rad, y + rad), outline=color, width=2)
    im.save(out_dir / f"{tag}.jpg", quality=90)
    return rows


def field_list():
    return all_fields()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="")
    ap.add_argument("--image", default="")
    ap.add_argument("--fields", default="")
    ap.add_argument("--all-fields", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--detector", default=str(PROJ / "models" / "center_detector_v2_final.pt"))
    ap.add_argument("--attachment-prefix", default=str(PROJ / "models" / "attachment_reviewed_v3"))
    ap.add_argument("--score-gate", type=float, default=SCORE_GATE)
    ap.add_argument("--nms-dist", type=float, default=0.5)
    ap.add_argument("--type-threshold", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = "mps" if (args.device == "auto" and torch.backends.mps.is_available()) else \
             "cuda" if (args.device == "auto" and torch.cuda.is_available()) else \
             "cpu" if args.device == "auto" else args.device
    det, models, temps = load_models(args.detector, args.attachment_prefix, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.image:
        fields = [args.field or Path(args.image).stem]
        paths = {fields[0]: args.image}
    elif args.all_fields:
        fields = field_list()
        paths = {f: field_image(f) for f in fields}
    else:
        fields = [s.strip() for s in (args.fields or args.field).split(",") if s.strip()]
        paths = {f: field_image(f) for f in fields}
    all_rows = []
    summary = []
    for field in fields:
        rows = run_field(field, paths[field], det, models, temps, device, out,
                         args.score_gate, args.nms_dist, args.type_threshold)
        all_rows.extend(rows)
        acc = [r for r in rows if r["score_gate"]]
        summary.append({
            "field": field, "proposals": len(rows), "accepted": len(acc),
            "attached": sum(1 for r in acc if r["type"] == "attached"),
            "none": sum(1 for r in acc if r["type"] == "none"),
            "rejected": len(rows) - len(acc),
        })
        print(field, summary[-1])
    with open(out / "predictions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    (out / "summary.json").write_text(json.dumps({
        "score_gate": args.score_gate, "fields": summary,
        "totals": {
            "proposals": sum(s["proposals"] for s in summary),
            "accepted": sum(s["accepted"] for s in summary),
            "attached": sum(s["attached"] for s in summary),
            "none": sum(s["none"] for s in summary),
        },
    }, indent=2))
    print(f"wrote {out / 'predictions.csv'}")


if __name__ == "__main__":
    main()
