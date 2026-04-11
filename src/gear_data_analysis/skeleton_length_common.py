from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw

from .straightener_common import extract_skeleton_and_longest_path


def iter_json_files(inputs: Iterable[str | Path]) -> list[Path]:
    json_files: set[Path] = set()
    for item in inputs:
        path = Path(item).expanduser().resolve()
        if path.is_file() and path.suffix.lower() == ".json":
            json_files.add(path)
            continue
        if path.is_dir():
            json_files.update(p.resolve() for p in path.rglob("*.json"))
    return sorted(json_files)


def iter_czi_files(inputs: Iterable[str | Path]) -> list[Path]:
    czi_files: set[Path] = set()
    for item in inputs:
        path = Path(item).expanduser().resolve()
        if path.is_file() and path.suffix.lower() == ".czi":
            czi_files.add(path)
            continue
        if path.is_dir():
            czi_files.update(p.resolve() for p in path.rglob("*.czi"))
            czi_files.update(p.resolve() for p in path.rglob("*.CZI"))
    return sorted(czi_files)


def make_relative_display_path(path: Path, base: Path) -> str:
    path = Path(path).resolve()
    base = Path(base).resolve()
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def resolve_image_path(json_path: Path, labelme_data: dict) -> Path | None:
    candidate_paths: list[Path] = []
    image_path = labelme_data.get("imagePath")

    if image_path:
        candidate_paths.append(json_path.parent / image_path)
        candidate_paths.append(json_path.parent / Path(image_path).name)

    for suffix in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"):
        candidate_paths.append(json_path.with_suffix(suffix))

    seen: set[Path] = set()
    for candidate in candidate_paths:
        resolved_candidate = candidate.expanduser().resolve()
        if resolved_candidate in seen:
            continue
        seen.add(resolved_candidate)
        if resolved_candidate.exists():
            return resolved_candidate

    return None


def load_image_array(image_path: Path | None) -> np.ndarray | None:
    if image_path is None:
        return None
    with Image.open(image_path) as image:
        return np.asarray(image)


def extract_polygon_records(labelme_data: dict) -> list[dict[str, object]]:
    polygon_records: list[dict[str, object]] = []
    for shape_index, shape in enumerate(labelme_data.get("shapes", []), start=1):
        if shape.get("shape_type") != "polygon":
            continue
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        polygon_records.append(
            {
                "polygon_index": shape_index,
                "label": shape.get("label", ""),
                "points_xy": np.asarray(points, dtype=np.float64),
            }
        )
    return polygon_records


def extract_polygon_point_sets(labelme_data: dict) -> list[np.ndarray]:
    return [
        np.asarray(shape.get("points", []), dtype=np.float64)
        for shape in labelme_data.get("shapes", [])
        if shape.get("shape_type") == "polygon" and len(shape.get("points", [])) >= 3
    ]


def rasterize_polygon_local(points_xy: np.ndarray, padding: int = 2) -> dict[str, object]:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    min_x = int(math.floor(points_xy[:, 0].min())) - padding
    min_y = int(math.floor(points_xy[:, 1].min())) - padding
    max_x = int(math.ceil(points_xy[:, 0].max())) + padding
    max_y = int(math.ceil(points_xy[:, 1].max())) + padding

    width = max_x - min_x + 1
    height = max_y - min_y + 1
    if width <= 0 or height <= 0:
        raise ValueError("Invalid local polygon bounds.")

    shifted_points_xy = points_xy - np.array([min_x, min_y], dtype=np.float64)
    canvas = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(canvas)
    drawer.polygon([tuple(point) for point in shifted_points_xy], outline=1, fill=1)
    binary_mask = np.array(canvas, dtype=np.uint8)

    if np.count_nonzero(binary_mask) == 0:
        raise ValueError("Polygon rasterization produced an empty mask.")

    return {
        "binary_mask": binary_mask,
        "shifted_points_xy": shifted_points_xy,
        "offset_xy": np.array([min_x, min_y], dtype=np.float64),
        "bbox_xyxy": (min_x, min_y, max_x, max_y),
    }


def rasterize_polygons(labelme_data: dict) -> tuple[np.ndarray, int]:
    width = int(labelme_data["imageWidth"])
    height = int(labelme_data["imageHeight"])
    canvas = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(canvas)

    polygon_count = 0
    for shape in labelme_data.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        drawer.polygon([tuple(point) for point in points], outline=1, fill=1)
        polygon_count += 1

    if polygon_count == 0:
        raise ValueError("No valid polygon shapes found in JSON.")

    return np.array(canvas, dtype=np.uint8), polygon_count


def extract_pixel_size_from_czi(czi_path: Path) -> dict[str, object]:
    czi_path = Path(czi_path).expanduser().resolve()
    text = czi_path.read_text(errors="ignore")

    def extract_distance(axis: str) -> float:
        pattern = (
            rf'<Distance Id="{axis}">\s*'
            rf"<Value>([^<]+)</Value>\s*"
            rf"(?:<DefaultUnitFormat>([^<]+)</DefaultUnitFormat>)?"
        )
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"Could not find scaling metadata for axis {axis} in {czi_path.name}.")

        raw_value = float(match.group(1))
        return raw_value * 1e6 if raw_value < 1e-3 else raw_value

    size_x_match = re.search(r"<SizeX>(\d+)</SizeX>", text)
    size_y_match = re.search(r"<SizeY>(\d+)</SizeY>", text)

    return {
        "reference_czi_path": str(czi_path),
        "pixel_size_x_um": extract_distance("X"),
        "pixel_size_y_um": extract_distance("Y"),
        "size_x": int(size_x_match.group(1)) if size_x_match else None,
        "size_y": int(size_y_match.group(1)) if size_y_match else None,
    }


def infer_expected_export_size(json_files: Iterable[Path]) -> tuple[int, int] | None:
    sizes = []
    for json_path in json_files:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        sizes.append((int(data["imageWidth"]), int(data["imageHeight"])))
    if not sizes:
        return None
    width = statistics.mode([item[0] for item in sizes])
    height = statistics.mode([item[1] for item in sizes])
    return width, height


def select_dataset_calibration(
    czi_root: Path | None, expected_export_size: tuple[int, int] | None = None
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    czi_files = iter_czi_files([czi_root]) if czi_root else []
    if not czi_files:
        return None, []

    metadata_rows = [extract_pixel_size_from_czi(path) for path in czi_files]
    candidate_rows = metadata_rows

    if expected_export_size is not None:
        expected_width, expected_height = expected_export_size
        size_filtered = [
            row
            for row in metadata_rows
            if row["size_x"] is not None
            and row["size_y"] is not None
            and row["size_x"] >= expected_width
            and row["size_y"] >= expected_height
        ]
        if size_filtered:
            candidate_rows = size_filtered

    grouped: dict[tuple[float, float], list[dict[str, object]]] = {}
    for row in candidate_rows:
        key = (round(float(row["pixel_size_x_um"]), 12), round(float(row["pixel_size_y_um"]), 12))
        grouped.setdefault(key, []).append(row)

    if len(grouped) == 1:
        selected_group = next(iter(grouped.values()))
        selected = dict(selected_group[0])
        selected["selection_note"] = "Selected unique calibration group after size filtering."
        selected["candidate_count"] = len(selected_group)
        return selected, metadata_rows

    ordered_groups = sorted(grouped.values(), key=len, reverse=True)
    if ordered_groups and len(ordered_groups[0]) > len(candidate_rows) / 2:
        selected = dict(ordered_groups[0][0])
        selected["selection_note"] = "Selected majority calibration group after size filtering."
        selected["candidate_count"] = len(ordered_groups[0])
        return selected, metadata_rows

    raise ValueError("Multiple conflicting calibration groups found in the provided CZI files.")


def measure_path_length(path_coordinates_rc: np.ndarray) -> float:
    if len(path_coordinates_rc) < 2:
        return 0.0
    deltas = np.diff(path_coordinates_rc.astype(np.float64), axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def measure_path_length_um(
    path_coordinates_rc: np.ndarray, pixel_size_x_um: float, pixel_size_y_um: float
) -> float:
    if len(path_coordinates_rc) < 2:
        return 0.0
    deltas = np.diff(path_coordinates_rc.astype(np.float64), axis=0)
    scaled_deltas = np.column_stack(
        [deltas[:, 1] * pixel_size_x_um, deltas[:, 0] * pixel_size_y_um]
    )
    return float(np.linalg.norm(scaled_deltas, axis=1).sum())


def analyze_polygon_lengths_from_raw(
    input_root: Path,
    czi_root: Path | None,
    base_dir: Path,
    padding: int = 2,
    display_root_name: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    input_root_path = Path(input_root)
    display_root_name = input_root_path.name if display_root_name is None else display_root_name
    input_root = input_root_path.resolve()
    base_dir = Path(base_dir).resolve()
    json_files = iter_json_files([input_root])
    expected_export_size = infer_expected_export_size(json_files)
    calibration, _ = select_dataset_calibration(czi_root, expected_export_size)

    rows: list[dict[str, object]] = []
    for json_path in json_files:
        labelme_data = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = resolve_image_path(json_path, labelme_data)
        image_id = json_path.parent.name
        image_name = image_path.name if image_path is not None else (labelme_data.get("imagePath") or "")

        for polygon in extract_polygon_records(labelme_data):
            polygon_mask = rasterize_polygon_local(polygon["points_xy"], padding=padding)[
                "binary_mask"
            ]
            longest_path_coords_rc = extract_skeleton_and_longest_path(
                polygon_mask, start_preference="top"
            )
            try:
                json_display_path = str(Path(display_root_name) / json_path.relative_to(input_root))
            except ValueError:
                json_display_path = make_relative_display_path(json_path, base_dir)
            row: dict[str, object] = {
                "image_id": image_id,
                "image_name": image_name,
                "json_path": json_display_path,
                "polygon_index": polygon["polygon_index"],
                "longest_skeleton_length_px": measure_path_length(longest_path_coords_rc),
                "longest_skeleton_length_um": "",
            }
            if calibration is not None:
                row["longest_skeleton_length_um"] = measure_path_length_um(
                    longest_path_coords_rc,
                    float(calibration["pixel_size_x_um"]),
                    float(calibration["pixel_size_y_um"]),
                )
            rows.append(row)

    return rows, calibration


def analyze_grouped_lengths_from_raw(
    group_dirs: Iterable[Path],
    base_dir: Path,
    reference_czi: Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    base_dir = Path(base_dir).resolve()
    resolved_group_dirs = [Path(group_dir).resolve() for group_dir in group_dirs]
    calibration = extract_pixel_size_from_czi(reference_czi.resolve()) if reference_czi else None

    rows: list[dict[str, object]] = []
    for json_path in iter_json_files(resolved_group_dirs):
        labelme_data = json.loads(json_path.read_text(encoding="utf-8"))
        binary_mask, _ = rasterize_polygons(labelme_data)
        longest_path_coords_rc = extract_skeleton_and_longest_path(
            binary_mask, start_preference="top"
        )
        image_path = resolve_image_path(json_path, labelme_data)
        row: dict[str, object] = {
            "group": make_relative_display_path(json_path.parent.resolve(), base_dir),
            "file_name": json_path.name,
            "image_name": image_path.name if image_path is not None else (labelme_data.get("imagePath") or ""),
            "json_path": make_relative_display_path(json_path, base_dir),
            "longest_skeleton_length_px": measure_path_length(longest_path_coords_rc),
            "longest_skeleton_length_um": "",
        }
        if calibration is not None:
            row["longest_skeleton_length_um"] = measure_path_length_um(
                longest_path_coords_rc,
                float(calibration["pixel_size_x_um"]),
                float(calibration["pixel_size_y_um"]),
            )
        rows.append(row)

    return rows, calibration


def write_csv_rows(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows available to write to {output_path}.")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_figs3e_values(csv_path: Path) -> tuple[np.ndarray, bool]:
    df = pd.read_csv(csv_path)
    use_um = (
        "longest_skeleton_length_um" in df.columns
        and df["longest_skeleton_length_um"].notna().all()
    )
    value_key = "longest_skeleton_length_um" if use_um else "longest_skeleton_length_px"
    values = df[value_key].astype(float).to_numpy()
    return values, use_um


def choose_histogram_bins(values: np.ndarray, use_um: bool) -> np.ndarray:
    if use_um:
        return np.arange(0, 26, 1)

    if len(values) > 1:
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            bin_width = 2 * iqr / (len(values) ** (1 / 3))
            bin_count = max(8, int(math.ceil((values.max() - values.min()) / bin_width)))
        else:
            bin_count = max(8, int(math.sqrt(len(values))))
    else:
        bin_count = 8

    bins = np.linspace(values.min(), values.max(), bin_count + 1)
    if np.allclose(bins[0], bins[-1]):
        bins = np.linspace(values.min() - 0.5, values.max() + 0.5, bin_count + 1)
    return bins


def render_figs3e_hist_from_csv(
    csv_path: Path, svg_path: Path, png_path: Path | None = None
) -> None:
    values, use_um = load_figs3e_values(csv_path)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["svg.fonttype"] = "none"

    bins = choose_histogram_bins(values, use_um=use_um)
    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    ax.hist(
        values,
        bins=bins,
        color="#4C72B0",
        edgecolor="#355C8C",
        linewidth=0.9,
        alpha=0.95,
        rwidth=0.84,
    )
    ax.set_xlim(0, 25)
    ax.set_xticks(np.arange(0, 26, 5))
    ax.set_xlabel("Length(μm)" if use_um else "Length(px)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=14, fontweight="bold")
    ax.set_title(f"Length distribution of GEARs (N={len(values)})", fontsize=16, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.tick_params(axis="both", labelsize=12, width=1.6, length=6)

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    svg_path = Path(svg_path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    if png_path is not None:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def build_figs4b_plot_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    group_order = [
        ("more1-c2", ">=2"),
        ("1-c2", "1"),
        ("0-c2", "0"),
    ]

    rows = []
    for group_name, puncta_label in group_order:
        group_lengths = df.loc[df["group"] == group_name, "longest_skeleton_length_um"].astype(float)
        for length_um in group_lengths:
            rows.append({"length_um": length_um, "puncta_label": puncta_label})

    plot_df = pd.DataFrame(rows)
    plot_df["puncta_label"] = pd.Categorical(
        plot_df["puncta_label"],
        categories=[label for _, label in group_order],
        ordered=True,
    )
    return plot_df


def render_figs4b_boxstrip_from_csv(
    csv_path: Path, svg_path: Path, png_path: Path | None = None
) -> None:
    plot_df = build_figs4b_plot_dataframe(csv_path)

    plt.rcParams["svg.fonttype"] = "none"
    sns.set_style("white")

    fig, ax = plt.subplots(figsize=(4.8, 4.1), constrained_layout=True)

    sns.boxplot(
        data=plot_df,
        x="length_um",
        y="puncta_label",
        order=list(plot_df["puncta_label"].cat.categories),
        orient="h",
        width=0.48,
        whis=(0, 100),
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 2.0},
        whiskerprops={"color": "black", "linewidth": 2.0},
        capprops={"color": "black", "linewidth": 2.0},
        medianprops={"color": "black", "linewidth": 2.2},
        saturation=1,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="length_um",
        y="puncta_label",
        order=list(plot_df["puncta_label"].cat.categories),
        orient="h",
        jitter=0.16,
        size=8.5,
        color="black",
        linewidth=0,
        ax=ax,
    )

    ax.set_title(f"N={len(plot_df)}", fontsize=18, fontweight="normal", pad=8)
    ax.set_xlabel("Filament length (μm)", fontsize=13, fontweight="normal")
    ax.set_ylabel("Number of DNA puncta", fontsize=13, fontweight="normal")
    ax.set_xlim(0, 10)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.tick_params(axis="both", labelsize=11, width=1.8, length=8)

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("normal")

    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    svg_path = Path(svg_path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    if png_path is not None:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def run_figs3e_figure(fig_dir: Path) -> dict[str, object]:
    fig_dir = Path(fig_dir).resolve()
    raw_root = fig_dir / "00_FigS3e_raw"
    czi_root = raw_root / "Multi-filaments-czi"
    output_dir = fig_dir / "02_FigS3e_plot_output"
    output_csv = output_dir / "FigS3e.csv"
    output_svg = output_dir / "FigS3e.svg"

    rows, calibration = analyze_polygon_lengths_from_raw(
        input_root=raw_root,
        czi_root=czi_root,
        base_dir=fig_dir / "00_FigS3e_raw",
        padding=2,
        display_root_name="Multi-filaments",
    )
    write_csv_rows(rows, output_csv)
    render_figs3e_hist_from_csv(output_csv, output_svg)
    return {
        "rows": rows,
        "calibration": calibration,
        "output_csv": output_csv,
        "output_svg": output_svg,
    }


def run_figs4b_figure(fig_dir: Path) -> dict[str, object]:
    fig_dir = Path(fig_dir).resolve()
    raw_root = fig_dir / "00_FigS4b_raw"
    group_dirs = [raw_root / "0-c2", raw_root / "1-c2", raw_root / "more1-c2"]
    reference_czi = raw_root / "Snap-8223-Create Image Subset and Split-01.czi"
    output_dir = fig_dir / "02_FigS4b_plot_output"
    output_csv = output_dir / "FigS4b.csv"
    output_svg = output_dir / "FigS4b.svg"

    rows, calibration = analyze_grouped_lengths_from_raw(
        group_dirs=group_dirs,
        base_dir=raw_root,
        reference_czi=reference_czi,
    )
    write_csv_rows(rows, output_csv)
    render_figs4b_boxstrip_from_csv(output_csv, output_svg)
    return {
        "rows": rows,
        "calibration": calibration,
        "output_csv": output_csv,
        "output_svg": output_svg,
    }
