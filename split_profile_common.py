from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from straightener_common import to_grayscale_frames


def analyze_split_profile_stack(
    input_source,
    output_svg_file,
    output_csv_file,
    num_bins=20,
    normalize_intensity=True,
    plot_title=None,
):
    for path in [output_csv_file, output_svg_file]:
        output_dir = os.path.dirname(path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    print("--- Configuration ---")
    print(f"Output SVG: {output_svg_file}")
    print(f"Output CSV: {output_csv_file}")
    print(f"Number of Bins: {num_bins}")
    print(f"Normalize Intensity: {normalize_intensity}")
    print("---------------------\n")

    try:
        frames = to_grayscale_frames(input_source)
        num_frames = len(frames)
        print(f"Loaded in-memory stack successfully (contains {num_frames} frames).")
        print("Initiating frame-by-frame split-profile analysis...")
    except Exception as exc:
        raise RuntimeError(f"Could not prepare input frames: {exc}") from exc

    all_half_profiles = []
    processed_half_names = []

    for frame_index, image in enumerate(frames):
        frame_name = f"frame_{frame_index}"
        try:
            if image is None or image.ndim != 2 or image.shape[0] == 0 or image.shape[1] == 0:
                print(f"Warning: Skipping {frame_name} due to invalid image data.")
                continue

            profile = np.mean(image, axis=0)
            x_centered = np.linspace(-1.0, 1.0, image.shape[1])
            x_distance = np.abs(x_centered)
            df_full = pd.DataFrame(
                {
                    "x_distance": x_distance,
                    "intensity": profile,
                    "original_x": x_centered,
                }
            )

            bins = np.linspace(0, 1.0, num_bins + 1)
            bin_labels = (bins[:-1] + bins[1:]) / 2
            expected_bin_pixels = image.shape[1] / 2 / num_bins

            df_right = df_full[df_full["original_x"] >= 0].copy()
            if not df_right.empty:
                df_right["bin"] = pd.cut(
                    df_right["x_distance"],
                    bins=bins,
                    labels=bin_labels,
                    include_lowest=True,
                )
                profile_right = df_right.groupby("bin", observed=True).agg(
                    intensity=("intensity", "mean"),
                    pixel_count=("bin", "count"),
                )
                profile_right["intensity"] *= (
                    profile_right["pixel_count"] / expected_bin_pixels
                )
                if normalize_intensity and profile_right["intensity"].max() > 0:
                    profile_right["intensity"] /= profile_right["intensity"].max()
                all_half_profiles.append(profile_right["intensity"])
                processed_half_names.append(f"{frame_name}_right")

            df_left = df_full[df_full["original_x"] < 0].copy()
            if not df_left.empty:
                df_left["bin"] = pd.cut(
                    df_left["x_distance"],
                    bins=bins,
                    labels=bin_labels,
                    include_lowest=True,
                )
                profile_left = df_left.groupby("bin", observed=True).agg(
                    intensity=("intensity", "mean"),
                    pixel_count=("bin", "count"),
                )
                profile_left["intensity"] *= (
                    profile_left["pixel_count"] / expected_bin_pixels
                )
                if normalize_intensity and profile_left["intensity"].max() > 0:
                    profile_left["intensity"] /= profile_left["intensity"].max()
                all_half_profiles.append(profile_left["intensity"])
                processed_half_names.append(f"{frame_name}_left")
        except Exception as exc:
            print(f"An error occurred while processing {frame_name}: {exc}")

    if not all_half_profiles:
        raise RuntimeError("No frames were processed successfully.")

    print("\n" + "-" * 50)
    print("All profiles processed. Aggregating data...")

    full_index = pd.Index(bin_labels, name="distance_from_center")
    df_all_halves = pd.concat(
        [profile.reindex(full_index, fill_value=0) for profile in all_half_profiles],
        axis=1,
    )
    df_all_halves.columns = processed_half_names

    overall_mean = df_all_halves.mean(axis=1)
    results_df = df_all_halves.copy()
    results_df["overall_mean"] = overall_mean
    results_df.to_csv(output_csv_file)

    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(8, 6))

    for column in df_all_halves.columns:
        ax.plot(df_all_halves.index, df_all_halves[column], color="gray", lw=1.5, alpha=0.3)

    ax.plot(overall_mean.index, overall_mean.values, color="black", lw=2, ls="-", label="Overall Mean")

    if plot_title:
        ax.set_title(f"{plot_title} (n={num_frames})", fontsize=20, pad=20)
    ax.set_xlabel("Normalized distance from equator", fontsize=16)
    ylabel_text = "Peak-Normalized intensity" if normalize_intensity else "Compensated Intensity"
    ax.set_ylabel(ylabel_text, fontsize=16)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(bottom=0)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.5)
    ax.tick_params(axis="both", which="major", width=1.5, length=6, labelsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_svg_file, dpi=300, bbox_inches="tight", facecolor="white")

    print("\nAnalysis complete!")
    print(f"SVG plot saved to: {output_svg_file}")
    print(f"CSV data file saved to: {output_csv_file}")
    print("-" * 50)
    plt.show()
