from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .csv_output_common import combine_export_tables, remove_files
from .straightener_common import (
    STRAIGHTENER_CONFIGS,
    build_straightened_stack_map,
    to_grayscale_frames,
)

DEFAULT_CENTER_WINDOW_MAX = 0.25
DEFAULT_NUM_BINS_PER_HALF = 20


@dataclass(frozen=True)
class PeakAnalysisResult:
    source_key: str
    plot_title: str
    frame_count: int
    center_window_max: float
    center_peak_count: int
    center_peak_fraction: float
    num_tied_peak_frames: int
    output_profile_svg: Path
    output_profile_csv: Path
    output_peak_csv: Path


def _format_peak_positions(values: list[float]) -> str:
    return ";".join(f"{value:.3f}" for value in values)


def analyze_unfolded_peak_stack(
    input_source,
    output_profile_svg: Path,
    output_profile_csv: Path,
    output_peak_csv: Path,
    *,
    num_bins_full: int,
    center_window_max: float,
    normalize_intensity: bool,
    plot_title: str,
    source_key: str,
) -> PeakAnalysisResult:
    output_profile_svg = Path(output_profile_svg)
    output_profile_csv = Path(output_profile_csv)
    output_peak_csv = Path(output_peak_csv)

    for path in (output_profile_svg, output_profile_csv, output_peak_csv):
        path.parent.mkdir(parents=True, exist_ok=True)

    frames = to_grayscale_frames(input_source)
    num_frames = len(frames)

    print("--- Configuration ---")
    print(f"Dataset: {source_key}")
    print(f"Output profile SVG: {output_profile_svg}")
    print(f"Output profile CSV: {output_profile_csv}")
    print(f"Output peak CSV: {output_peak_csv}")
    print(f"Number of Full-Width Bins: {num_bins_full}")
    print(f"Center window: |distance from center| <= {center_window_max}")
    print(f"Normalize Intensity: {normalize_intensity}")
    print("---------------------\n")
    print(f"Loaded in-memory stack successfully (contains {num_frames} frames).")

    bins = np.linspace(-1.0, 1.0, num_bins_full + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    full_index = pd.Index(bin_centers, name="position")

    binned_profiles: list[pd.Series] = []
    frame_names: list[str] = []
    peak_rows: list[dict[str, object]] = []

    for frame_index, image in enumerate(frames):
        frame_name = f"frame_{frame_index}"
        if image is None or image.ndim != 2 or image.shape[0] == 0 or image.shape[1] == 0:
            print(f"Warning: Skipping {frame_name} due to invalid image data.")
            continue

        profile = np.mean(image, axis=0)
        x_centered = np.linspace(-1.0, 1.0, image.shape[1])
        df_full = pd.DataFrame({"position": x_centered, "intensity": profile})
        df_full["bin"] = pd.cut(
            df_full["position"],
            bins=bins,
            labels=bin_centers,
            include_lowest=True,
        )

        profile_full = df_full.groupby("bin", observed=True).agg(
            intensity=("intensity", "mean"),
            pixel_count=("bin", "count"),
        )
        expected_bin_pixels = image.shape[1] / num_bins_full
        profile_full["intensity"] *= profile_full["pixel_count"] / expected_bin_pixels
        profile_full = profile_full.reindex(full_index, fill_value=0)

        if normalize_intensity and profile_full["intensity"].max() > 0:
            profile_full["intensity"] /= profile_full["intensity"].max()

        intensity_series = profile_full["intensity"]
        binned_profiles.append(intensity_series)
        frame_names.append(frame_name)

        peak_value = float(intensity_series.max())
        peak_positions = [
            float(position)
            for position, value in intensity_series.items()
            if np.isclose(value, peak_value)
        ]
        primary_peak_position = min(peak_positions, key=lambda value: (abs(value), value))
        has_center_peak = any(abs(value) <= center_window_max + 1e-12 for value in peak_positions)

        peak_rows.append(
            {
                "frame_name": frame_name,
                "peak_value": peak_value,
                "primary_peak_position": primary_peak_position,
                "primary_peak_distance": abs(primary_peak_position),
                "num_peak_bins": len(peak_positions),
                "peak_positions": peak_positions,
                "has_center_peak": has_center_peak,
            }
        )

    if not binned_profiles:
        raise RuntimeError("No frames were processed successfully.")

    df_profiles = pd.concat(
        [profile.reindex(full_index, fill_value=0) for profile in binned_profiles],
        axis=1,
    )
    df_profiles.columns = frame_names

    overall_mean = df_profiles.mean(axis=1)
    profile_output_df = df_profiles.copy()
    profile_output_df["overall_mean"] = overall_mean
    profile_output_df.to_csv(output_profile_csv)

    peak_df = pd.DataFrame(peak_rows)
    peak_export_df = peak_df.copy()
    peak_export_df["peak_positions"] = peak_export_df["peak_positions"].apply(_format_peak_positions)
    peak_export_df.to_csv(output_peak_csv, index=False)

    center_peak_count = int(peak_df["has_center_peak"].sum())
    center_peak_fraction = center_peak_count / len(peak_df)
    tied_peak_count = int((peak_df["num_peak_bins"] > 1).sum())

    fig, axes = plt.subplots(2, 1, figsize=(9, 10), gridspec_kw={"height_ratios": [3, 2]})

    axes[0].axvspan(-center_window_max, center_window_max, color="#f4cccc", alpha=0.45, label="Center")
    for column in df_profiles.columns:
        axes[0].plot(df_profiles.index, df_profiles[column], color="gray", lw=1.2, alpha=0.25)
    axes[0].plot(overall_mean.index, overall_mean.values, color="black", lw=2.2, label="Overall mean")
    axes[0].set_xlim(-1.0, 1.0)
    axes[0].set_ylim(bottom=0)
    axes[0].set_xlabel("Normalized position centered at equator", fontsize=14)
    axes[0].set_ylabel("Peak-normalized intensity", fontsize=14)
    axes[0].set_title(f"{plot_title} unfolded profiles (n={len(peak_df)})", fontsize=18, pad=14)
    axes[0].axvline(0.0, color="#444444", lw=1.0, ls="--", alpha=0.8)
    axes[0].legend(loc="upper right")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].tick_params(axis="both", which="major", width=1.2, length=5, labelsize=11)

    peak_count_series = peak_df["primary_peak_position"].value_counts().sort_index()
    peak_count_series = peak_count_series.reindex(full_index, fill_value=0)
    axes[1].axvspan(-center_window_max, center_window_max, color="#f4cccc", alpha=0.45)
    axes[1].bar(
        peak_count_series.index.astype(float),
        peak_count_series.values,
        width=(2.0 / num_bins_full) * 0.9,
        color="#4c78a8",
        edgecolor="white",
    )
    axes[1].axvline(0.0, color="#444444", lw=1.0, ls="--", alpha=0.8)
    axes[1].set_xlim(-1.0, 1.0)
    axes[1].set_xlabel("Primary peak bin position", fontsize=14)
    axes[1].set_ylabel("Frame count", fontsize=14)
    axes[1].set_title(
        f"Center peaks: {center_peak_count}/{len(peak_df)} ({center_peak_fraction:.1%})",
        fontsize=14,
        pad=10,
    )
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].tick_params(axis="both", which="major", width=1.2, length=5, labelsize=11)

    plt.tight_layout()
    plt.savefig(output_profile_svg, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    print("\nAnalysis complete!")
    print(f"Center peaks in window: {center_peak_count}/{len(peak_df)} ({center_peak_fraction:.1%})")
    print(f"Profile SVG saved to: {output_profile_svg}")
    print(f"Profile CSV saved to: {output_profile_csv}")
    print(f"Peak CSV saved to: {output_peak_csv}")
    print("-" * 50)

    return PeakAnalysisResult(
        source_key=source_key,
        plot_title=plot_title,
        frame_count=len(peak_df),
        center_window_max=center_window_max,
        center_peak_count=center_peak_count,
        center_peak_fraction=center_peak_fraction,
        num_tied_peak_frames=tied_peak_count,
        output_profile_svg=output_profile_svg,
        output_profile_csv=output_profile_csv,
        output_peak_csv=output_peak_csv,
    )


def run_figs8d_figure(fig_dir: Path) -> tuple[PeakAnalysisResult, ...]:
    output_dir = fig_dir / "02_FigS8d_plot_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = (
        {
            "source_key": "FigS8d_minus",
            "plot_title": "- MinCDE",
            "stack_key": "c2_FtsZ",
            "output_profile_svg": output_dir / "FigS8d_minus.svg",
            "output_profile_csv": output_dir / "FigS8d_minus.csv",
            "output_peak_csv": output_dir / "FigS8d_minus_peak.csv",
        },
        {
            "source_key": "FigS8d_plus",
            "plot_title": "+ MinCDE",
            "stack_key": "c2_FtsZ",
            "output_profile_svg": output_dir / "FigS8d_plus.svg",
            "output_profile_csv": output_dir / "FigS8d_plus.csv",
            "output_peak_csv": output_dir / "FigS8d_plus_peak.csv",
        },
    )

    results: list[PeakAnalysisResult] = []
    for dataset in datasets:
        stack_map = build_straightened_stack_map(STRAIGHTENER_CONFIGS[dataset["source_key"]])
        results.append(
            analyze_unfolded_peak_stack(
                input_source=stack_map[dataset["stack_key"]],
                output_profile_svg=dataset["output_profile_svg"],
                output_profile_csv=dataset["output_profile_csv"],
                output_peak_csv=dataset["output_peak_csv"],
                num_bins_full=DEFAULT_NUM_BINS_PER_HALF * 2,
                center_window_max=DEFAULT_CENTER_WINDOW_MAX,
                normalize_intensity=True,
                plot_title=dataset["plot_title"],
                source_key=dataset["source_key"],
            )
        )

    combined_csv_path = output_dir / "FigS8d.csv"
    combine_export_tables(
        combined_csv_path,
        [
            {
                "path": result.output_profile_csv,
                "metadata": {
                    "table_type": "profile",
                    "condition_key": result.source_key,
                    "condition_label": result.plot_title,
                },
            }
            for result in results
        ]
        + [
            {
                "path": result.output_peak_csv,
                "metadata": {
                    "table_type": "peak",
                    "condition_key": result.source_key,
                    "condition_label": result.plot_title,
                },
            }
            for result in results
        ],
    )
    remove_files(
        [result.output_profile_csv for result in results]
        + [result.output_peak_csv for result in results]
    )

    return tuple(results)
