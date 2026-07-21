from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class RoiTimecourseConfig:
    membrane_dir: Path
    secondary_dir: Path
    output_svg_file: Path
    output_csv_file: Path
    time_per_frame: float
    secondary_column_key: str
    secondary_plot_label: str
    time_from_sorted_order: bool = False
    x_limit: tuple[float, float] = (0, 60)
    membrane_plot_label: str = "Membrane"
    membrane_color: str = "#B200F4"
    secondary_color: str = "#00C000"


def extract_timepoint(filename: str) -> int:
    match = re.search(r"t(\d+)", filename)
    return int(match.group(1)) if match else 0


def read_tif_as_grayscale(tif_path: str | os.PathLike[str] | Path) -> np.ndarray:
    with Image.open(tif_path) as img:
        return np.array(img.convert("L"))


def find_json_file(
    timepoint: int,
    membrane_dir: Path,
    secondary_dir: Path,
) -> Path | None:
    patterns = [
        f"*t{timepoint:02d}*.json",
        f"*_t{timepoint:02d}_*.json",
        f"*-t{timepoint:02d}-*.json",
    ]

    for directory in (membrane_dir, secondary_dir):
        if not directory.exists():
            continue
        for pattern in patterns:
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[0]
    return None


def calculate_roi_intensity(image_array: np.ndarray, json_path: Path) -> tuple[float | None, np.ndarray | None]:
    if not json_path.exists():
        return None, None

    with json_path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)

    shapes = annotation.get("shapes") or []
    if not shapes:
        return None, None

    points = shapes[0]["points"]
    mask = Image.new("L", (image_array.shape[1], image_array.shape[0]), 0)
    ImageDraw.Draw(mask).polygon([tuple(point) for point in points], outline=1, fill=1)
    mask_array = np.array(mask)

    roi_pixels = image_array[mask_array == 1]
    if len(roi_pixels) == 0:
        return None, mask_array
    return float(np.mean(roi_pixels)), mask_array


def create_file_mapping(
    membrane_dir: Path,
    secondary_dir: Path,
    secondary_column_key: str,
) -> list[dict[str, object]]:
    membrane_tif_files = sorted(
        (path.name for path in membrane_dir.iterdir() if path.suffix.lower() == ".tif"),
        key=extract_timepoint,
    )
    secondary_tif_files = sorted(
        (path.name for path in secondary_dir.iterdir() if path.suffix.lower() == ".tif"),
        key=extract_timepoint,
    )

    file_mapping: list[dict[str, object]] = []

    print("=== File Mapping ===")
    for membrane_file in membrane_tif_files:
        membrane_timepoint = extract_timepoint(membrane_file)

        secondary_file = None
        for candidate in secondary_tif_files:
            if extract_timepoint(candidate) == membrane_timepoint:
                secondary_file = candidate
                break

        json_file_path = find_json_file(membrane_timepoint, membrane_dir, secondary_dir)

        if secondary_file and json_file_path:
            mapping = {
                "timepoint": membrane_timepoint,
                "membrane_tif": membrane_file,
                "secondary_tif": secondary_file,
                "json_path": json_file_path,
            }
            file_mapping.append(mapping)
            json_dir = (
                membrane_dir.name
                if json_file_path.parent == membrane_dir
                else secondary_dir.name
            )
            print(
                f"t{membrane_timepoint:02d}: {membrane_file} + {secondary_file} "
                f"-> {json_file_path.name} (from {json_dir})"
            )
        else:
            missing = []
            if not secondary_file:
                missing.append(f"{secondary_column_key} file for {membrane_file}")
            if not json_file_path:
                missing.append(f"JSON annotation for t{membrane_timepoint:02d}")
            print(f"Warning: Missing {', '.join(missing)}")

    print(f"\nTotal mappings created: {len(file_mapping)}\n")
    return file_mapping


def process_mapped_files(
    file_mapping: list[dict[str, object]],
    membrane_dir: Path,
    secondary_dir: Path,
    time_per_frame: float,
    secondary_column_key: str,
    time_from_sorted_order: bool = False,
) -> pd.DataFrame:
    data: list[dict[str, float | int]] = []

    print("=== Processing Files and Calculating ROI Intensities ===")

    membrane_raw_key = "Membrane_Raw_Intensity"
    secondary_raw_key = f"{secondary_column_key}_Raw_Intensity"

    for frame_order, mapping in enumerate(file_mapping):
        timepoint = int(mapping["timepoint"])
        membrane_tif_path = membrane_dir / str(mapping["membrane_tif"])
        secondary_tif_path = secondary_dir / str(mapping["secondary_tif"])
        json_path = Path(mapping["json_path"])

        membrane_image = read_tif_as_grayscale(membrane_tif_path)
        secondary_image = read_tif_as_grayscale(secondary_tif_path)

        membrane_intensity, roi_mask = calculate_roi_intensity(membrane_image, json_path)

        if membrane_intensity is not None and roi_mask is not None:
            secondary_roi_pixels = secondary_image[roi_mask == 1]
            secondary_intensity = (
                float(np.mean(secondary_roi_pixels)) if len(secondary_roi_pixels) > 0 else 0.0
            )

            if time_from_sorted_order:
                time_adjusted = frame_order * time_per_frame
            else:
                time_adjusted = (timepoint - 1) * time_per_frame
            data.append(
                {
                    "Frame": timepoint,
                    "Time/min": time_adjusted,
                    membrane_raw_key: membrane_intensity,
                    secondary_raw_key: secondary_intensity,
                }
            )

            print(
                f"t{timepoint:02d}: Membrane_ROI={membrane_intensity:.2f}, "
                f"{secondary_column_key}_ROI={secondary_intensity:.2f}"
            )
        else:
            print(f"Warning: t{timepoint:02d} unable to obtain ROI information")

    df = pd.DataFrame(data)
    if df.empty:
        return df

    frames_sorted = sorted(df["Frame"].unique())
    membrane_norm_key = "Membrane_Normalized_Intensity"
    secondary_norm_key = f"{secondary_column_key}_Normalized_Intensity"

    if len(frames_sorted) >= 2:
        first_frame, second_frame = frames_sorted[0], frames_sorted[1]

        i1_mem = df.loc[df["Frame"] == first_frame, membrane_raw_key].iloc[0]
        i2_mem = df.loc[df["Frame"] == second_frame, membrane_raw_key].iloc[0]
        denom_mem = i1_mem - i2_mem
        if denom_mem == 0:
            denom_mem = 1.0
        df[membrane_norm_key] = (df[membrane_raw_key] - i2_mem) / denom_mem

        i1_secondary = df.loc[df["Frame"] == first_frame, secondary_raw_key].iloc[0]
        i2_secondary = df.loc[df["Frame"] == second_frame, secondary_raw_key].iloc[0]
        denom_secondary = i1_secondary - i2_secondary
        if denom_secondary == 0:
            denom_secondary = 1.0
        df[secondary_norm_key] = (df[secondary_raw_key] - i2_secondary) / denom_secondary
    else:
        df[membrane_norm_key] = 0.0
        df[secondary_norm_key] = 0.0

    return df


def plot_roi_timecourse(
    df: pd.DataFrame,
    output_svg_file: Path,
    secondary_column_key: str,
    secondary_plot_label: str,
    membrane_plot_label: str,
    x_limit: tuple[float, float],
    membrane_color: str,
    secondary_color: str,
) -> None:
    plt.figure(figsize=(6, 4))

    ax = plt.gca()
    membrane_norm_key = "Membrane_Normalized_Intensity"
    secondary_norm_key = f"{secondary_column_key}_Normalized_Intensity"

    ax.plot(
        df["Time/min"],
        df[membrane_norm_key],
        marker="o",
        markersize=6,
        color=membrane_color,
        label=membrane_plot_label,
        zorder=10,
        clip_on=False,
    )
    ax.plot(
        df["Time/min"],
        df[secondary_norm_key],
        marker="o",
        markersize=6,
        color=secondary_color,
        label=secondary_plot_label,
        zorder=5,
        clip_on=False,
    )

    ax.set_xlabel("Time (min)", fontsize=16)
    ax.set_ylabel("Normalized intensity", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis="x", which="major", length=7, width=2, labelsize=14)
    ax.tick_params(axis="x", which="minor", length=4, width=1)
    ax.tick_params(axis="y", which="major", length=7, width=2, labelsize=14)

    normalized_cols = [col for col in df.columns if col.endswith("Normalized_Intensity")]
    plot_min = float(df[normalized_cols].min().min())
    plot_max = float(df[normalized_cols].max().max())
    y_min = min(0.0, plot_min)
    y_max = max(1.0, plot_max)
    if y_min < 0:
        y_min -= 0.02
    if y_max > 1:
        y_max += 0.05
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(
        ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
    )
    ax.set_xlim(*x_limit)

    ax.legend(loc="upper right", fontsize=14, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    output_svg_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_svg_file, format="svg", bbox_inches="tight", dpi=300)


def print_intensity_statistics(df: pd.DataFrame, secondary_column_key: str) -> None:
    membrane_raw_key = "Membrane_Raw_Intensity"
    membrane_norm_key = "Membrane_Normalized_Intensity"
    secondary_raw_key = f"{secondary_column_key}_Raw_Intensity"
    secondary_norm_key = f"{secondary_column_key}_Normalized_Intensity"

    print("\n=== Intensity Statistics ===")
    print("Membrane intensity statistics:")
    print(
        f"  Raw intensity range: {df[membrane_raw_key].min():.2f} - "
        f"{df[membrane_raw_key].max():.2f}"
    )
    print(
        f"  Normalized range: {df[membrane_norm_key].min():.2f} - "
        f"{df[membrane_norm_key].max():.2f}"
    )

    print(f"\n{secondary_column_key} intensity statistics:")
    print(
        f"  Raw intensity range: {df[secondary_raw_key].min():.2f} - "
        f"{df[secondary_raw_key].max():.2f}"
    )
    print(
        f"  Normalized range: {df[secondary_norm_key].min():.2f} - "
        f"{df[secondary_norm_key].max():.2f}"
    )


def run_roi_timecourse(config: RoiTimecourseConfig) -> pd.DataFrame:
    file_mapping = create_file_mapping(
        membrane_dir=config.membrane_dir,
        secondary_dir=config.secondary_dir,
        secondary_column_key=config.secondary_column_key,
    )
    df = process_mapped_files(
        file_mapping=file_mapping,
        membrane_dir=config.membrane_dir,
        secondary_dir=config.secondary_dir,
        time_per_frame=config.time_per_frame,
        secondary_column_key=config.secondary_column_key,
        time_from_sorted_order=config.time_from_sorted_order,
    )

    print("\n=== Processing Results ===")
    print(f"Successfully processed {len(df)} timepoints")
    print("Data preview:")
    print(df.head())

    config.output_csv_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.output_csv_file, index=False)
    print(f"\nData successfully saved to CSV file: {config.output_csv_file}")

    plot_roi_timecourse(
        df=df,
        output_svg_file=config.output_svg_file,
        secondary_column_key=config.secondary_column_key,
        secondary_plot_label=config.secondary_plot_label,
        membrane_plot_label=config.membrane_plot_label,
        x_limit=config.x_limit,
        membrane_color=config.membrane_color,
        secondary_color=config.secondary_color,
    )
    print(f"Image successfully saved to: {config.output_svg_file}")
    print_intensity_statistics(df, config.secondary_column_key)
    return df
