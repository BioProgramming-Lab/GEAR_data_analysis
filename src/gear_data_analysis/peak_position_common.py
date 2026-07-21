from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .csv_output_common import combine_export_tables, remove_files
from .straightener_common import (
    STRAIGHTENER_CONFIGS,
    build_straightened_stack_map,
    to_grayscale_frames,
)

DEFAULT_CENTER_WINDOW_MAX = 0.25
DEFAULT_NUM_BINS_PER_HALF = 20
FIGS8D_MINUS_KEY = "FigS8d_minus"
FIGS8D_PLUS_KEY = "FigS8d_plus"
FIGS11D_KEY = "FigS11d"
FIGS11E_KEY = "FigS11e"
FIGS12C_KEY = "FigS12c"
FIGS12D_KEY = "FigS12d"


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
    plot_mode: str = "peak_analysis",
) -> PeakAnalysisResult:
    output_profile_svg = Path(output_profile_svg)
    output_profile_csv = Path(output_profile_csv)
    output_peak_csv = Path(output_peak_csv)

    if plot_mode not in {"peak_analysis", "individual_only"}:
        raise ValueError(f"Unsupported unfolded plot mode: {plot_mode}")

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
    if plot_mode == "peak_analysis":
        print(f"Center window: |distance from center| <= {center_window_max}")
    print(f"Normalize Intensity: {normalize_intensity}")
    print(f"Plot mode: {plot_mode}")
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

    if plot_mode == "individual_only":
        fig, ax = plt.subplots(figsize=(6.2, 4.3))
        for column in df_profiles.columns:
            ax.plot(
                df_profiles.index,
                df_profiles[column],
                color="#a8a8a8",
                lw=1.15,
                alpha=0.42,
            )
        ax.axvline(0.0, color="#555555", lw=1.1, ls="--", alpha=0.9)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        x_ticks = np.linspace(-1.0, 1.0, 9)
        ax.set_xticks(x_ticks, [f"{value:.2f}" for value in x_ticks])
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.set_xlabel("Normalized position along the filament", fontsize=14)
        ax.set_ylabel("Normalized intensity", fontsize=14)
        ax.set_title(f"{plot_title} (n={len(peak_df)})", fontsize=16, pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.4)
        ax.spines["bottom"].set_linewidth(1.4)
        ax.tick_params(axis="both", which="major", width=1.4, length=5, labelsize=11)
        fig.tight_layout(pad=1.1)
    else:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(9, 10),
            gridspec_kw={"height_ratios": [3, 2]},
        )

        axes[0].axvspan(
            -center_window_max,
            center_window_max,
            color="#f4cccc",
            alpha=0.45,
            label="Center",
        )
        for column in df_profiles.columns:
            axes[0].plot(
                df_profiles.index,
                df_profiles[column],
                color="gray",
                lw=1.2,
                alpha=0.25,
            )
        axes[0].plot(
            overall_mean.index,
            overall_mean.values,
            color="black",
            lw=2.2,
            label="Overall mean",
        )
        axes[0].set_xlim(-1.0, 1.0)
        axes[0].set_ylim(bottom=0)
        axes[0].set_xlabel("Normalized position centered at equator", fontsize=14)
        axes[0].set_ylabel("Peak-normalized intensity", fontsize=14)
        axes[0].set_title(
            f"{plot_title} unfolded profiles (n={len(peak_df)})",
            fontsize=18,
            pad=14,
        )
        axes[0].axvline(0.0, color="#444444", lw=1.0, ls="--", alpha=0.8)
        axes[0].legend(loc="upper right")
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)
        axes[0].tick_params(axis="both", which="major", width=1.2, length=5, labelsize=11)

        peak_count_series = peak_df["primary_peak_position"].value_counts().sort_index()
        peak_count_series = peak_count_series.reindex(full_index, fill_value=0)
        axes[1].axvspan(
            -center_window_max,
            center_window_max,
            color="#f4cccc",
            alpha=0.45,
        )
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

        fig.tight_layout()

    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(
            output_profile_svg,
            format="svg",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    _show_figure_if_interactive(fig)

    print("\nAnalysis complete!")
    if plot_mode == "peak_analysis":
        print(
            f"Center peaks in window: {center_peak_count}/{len(peak_df)} "
            f"({center_peak_fraction:.1%})"
        )
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


def run_unfolded_peak_figure(
    datasets: list[dict[str, object]],
    combined_csv_path: Path,
    *,
    num_bins_full: int = DEFAULT_NUM_BINS_PER_HALF * 2,
    center_window_max: float = DEFAULT_CENTER_WINDOW_MAX,
    normalize_intensity: bool = True,
    plot_mode: str = "peak_analysis",
) -> tuple[PeakAnalysisResult, ...]:
    """Run the shared FigS8d-style unfolded peak analysis over one or more datasets.

    Each entry in ``datasets`` must provide an already-built in-memory frame
    stack (``input_source``), a ``plot_title``, a ``source_key``, and the three
    output paths (``output_profile_svg``, ``output_profile_csv``,
    ``output_peak_csv``). ``plot_mode="peak_analysis"`` produces the legacy
    two-panel profile + peak-position figure, while ``"individual_only"``
    produces a single panel containing only individual unfolded profiles. The
    per-dataset CSVs are merged into ``combined_csv_path`` (profiles first, then
    peaks, in dataset order) and the intermediate CSVs are removed.
    """
    results: list[PeakAnalysisResult] = []
    for dataset in datasets:
        results.append(
            analyze_unfolded_peak_stack(
                input_source=dataset["input_source"],
                output_profile_svg=dataset["output_profile_svg"],
                output_profile_csv=dataset["output_profile_csv"],
                output_peak_csv=dataset["output_peak_csv"],
                num_bins_full=num_bins_full,
                center_window_max=center_window_max,
                normalize_intensity=normalize_intensity,
                plot_title=dataset["plot_title"],
                source_key=dataset["source_key"],
                plot_mode=plot_mode,
            )
        )

    combine_export_tables(
        Path(combined_csv_path),
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


def _load_combined_profile_summary(combined_csv_path: Path) -> dict[str, dict[str, object]]:
    df = pd.read_csv(combined_csv_path)
    profile_df = df.loc[df["table_type"] == "profile"].copy()
    if profile_df.empty:
        raise ValueError(f"No profile rows found in {combined_csv_path}.")

    frame_columns = [
        column
        for column in profile_df.columns
        if str(column).startswith("frame_") and str(column).removeprefix("frame_").isdigit()
    ]
    if not frame_columns:
        raise ValueError(f"No frame columns found in {combined_csv_path}.")

    summaries: dict[str, dict[str, object]] = {}
    for condition_key, condition_df in profile_df.groupby("condition_key", sort=False):
        condition_df = condition_df.sort_values("position")
        frame_values = condition_df[frame_columns].apply(pd.to_numeric, errors="coerce")
        mean_values = pd.to_numeric(condition_df["overall_mean"], errors="coerce")
        if mean_values.isna().all():
            mean_values = frame_values.mean(axis=1)

        summaries[str(condition_key)] = {
            "label": str(condition_df["condition_label"].iloc[0]),
            "position": pd.to_numeric(condition_df["position"], errors="coerce").to_numpy(dtype=float),
            "mean": mean_values.to_numpy(dtype=float),
            "sem": frame_values.sem(axis=1).fillna(0.0).to_numpy(dtype=float),
            "frames": frame_values.to_numpy(dtype=float),
            "frame_count": int(frame_values.notna().sum(axis=1).max()),
        }

    return summaries


def _style_compact_profile_axis(ax) -> None:
    ax.axvline(0.0, color="#7899ad", linestyle="--", linewidth=1.6, alpha=0.9)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xticks(np.linspace(-1.0, 1.0, 9))
    ax.tick_params(axis="both", which="major", width=1.8, length=6, labelsize=12)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _show_figure_if_interactive(fig) -> None:
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def render_figs8d_compact_profile_variants(combined_csv_path: Path, output_dir: Path) -> list[Path]:
    summaries = _load_combined_profile_summary(combined_csv_path)
    missing_keys = [key for key in (FIGS8D_MINUS_KEY, FIGS8D_PLUS_KEY) if key not in summaries]
    if missing_keys:
        raise ValueError(f"Missing FigS8d profile condition(s): {', '.join(missing_keys)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        FIGS8D_MINUS_KEY: "#1f77b4",
        FIGS8D_PLUS_KEY: "#ff7f0e",
    }
    output_paths = [
        output_dir / "FigS8d_merged_profiles.svg",
        output_dir / "FigS8d_mirrored_profiles.svg",
    ]

    with plt.rc_context(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 14,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    ):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for condition_key in (FIGS8D_MINUS_KEY, FIGS8D_PLUS_KEY):
            summary = summaries[condition_key]
            x = summary["position"]
            mean = summary["mean"]
            sem = summary["sem"]
            color = colors[condition_key]
            ax.fill_between(
                x,
                np.clip(mean - sem, 0.0, None),
                mean + sem,
                color=color,
                alpha=0.16,
                linewidth=0,
            )
            ax.plot(
                x,
                mean,
                color=color,
                linewidth=2.8,
                label=f"{summary['label']} mean",
            )

        _style_compact_profile_axis(ax)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Normalized position along the filament", fontsize=14)
        ax.set_ylabel("Normalized intensity", fontsize=14)
        ax.set_title("Merged intensity profiles", fontsize=16, pad=10)
        ax.legend(loc="upper right", fontsize=12)
        fig.tight_layout(pad=1.2)
        fig.savefig(output_paths[0], format="svg", dpi=300, bbox_inches="tight", facecolor="white")
        _show_figure_if_interactive(fig)

        fig, ax = plt.subplots(figsize=(6.5, 8.4))
        mirror_order = (
            (FIGS8D_MINUS_KEY, 1.0, ""),
            (FIGS8D_PLUS_KEY, -1.0, " mirrored"),
        )
        for condition_key, direction, suffix in mirror_order:
            summary = summaries[condition_key]
            x = summary["position"]
            mean = summary["mean"]
            sem = summary["sem"]
            color = colors[condition_key]
            y_mean = direction * mean
            if direction > 0:
                y_low = np.clip(mean - sem, 0.0, None)
                y_high = mean + sem
            else:
                y_low = -(mean + sem)
                y_high = -np.clip(mean - sem, 0.0, None)

            ax.fill_between(x, y_low, y_high, color=color, alpha=0.16, linewidth=0)
            ax.plot(
                x,
                y_mean,
                color=color,
                linewidth=2.8,
                label=f"{summary['label']} mean{suffix}",
            )

        ax.axhline(0.0, color="#333333", linewidth=1.6)
        _style_compact_profile_axis(ax)
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_yticklabels(["1.0", "0.5", "0", "0.5", "1.0"])
        ax.set_xlabel("Normalized position along the filament", fontsize=14)
        ax.set_ylabel("Normalized intensity", fontsize=14)
        ax.set_title("Mirrored intensity profiles", fontsize=16, pad=10)
        ax.legend(loc="upper right", fontsize=12)
        fig.tight_layout(pad=1.2)
        fig.savefig(output_paths[1], format="svg", dpi=300, bbox_inches="tight", facecolor="white")
        _show_figure_if_interactive(fig)

    print(f"Compact merged profile SVG saved to: {output_paths[0]}")
    print(f"Compact mirrored profile SVG saved to: {output_paths[1]}")
    return output_paths


def render_figs8d_mirrored_individual_profiles(combined_csv_path: Path, output_dir: Path) -> Path:
    summaries = _load_combined_profile_summary(combined_csv_path)
    missing_keys = [key for key in (FIGS8D_MINUS_KEY, FIGS8D_PLUS_KEY) if key not in summaries]
    if missing_keys:
        raise ValueError(f"Missing FigS8d profile condition(s): {', '.join(missing_keys)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "FigS8d_mirrored_individual_profiles.svg"

    colors = {
        FIGS8D_MINUS_KEY: "#1f77b4",
        FIGS8D_PLUS_KEY: "#ff7f0e",
    }

    with plt.rc_context(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 14,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    ):
        fig, ax = plt.subplots(figsize=(6.5, 8.4))
        mirror_order = (
            (FIGS8D_MINUS_KEY, 1.0, ""),
            (FIGS8D_PLUS_KEY, -1.0, " mirrored"),
        )
        for condition_key, direction, suffix in mirror_order:
            summary = summaries[condition_key]
            x = summary["position"]
            frames = summary["frames"]
            mean = summary["mean"]
            color = colors[condition_key]

            for frame_index, frame_values in enumerate(frames.T):
                mask = ~np.isnan(frame_values)
                ax.plot(
                    x[mask],
                    direction * frame_values[mask],
                    color="gray",
                    linewidth=1.2,
                    alpha=0.25,
                    linestyle="-",
                    zorder=1,
                )

            ax.plot(
                x,
                direction * mean,
                color=color,
                linewidth=3.0,
                label=f"{summary['label']} mean{suffix}",
                zorder=3,
            )

        ax.axhline(0.0, color="#333333", linewidth=1.6)
        _style_compact_profile_axis(ax)
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_yticklabels(["1.0", "0.5", "0", "0.5", "1.0"])
        ax.set_xlabel("Normalized position along the filament", fontsize=14)
        ax.set_ylabel("Normalized intensity", fontsize=14)
        ax.set_title("Mirrored individual intensity profiles", fontsize=16, pad=10)

        handles, labels = ax.get_legend_handles_labels()
        handles.insert(
            0,
            Line2D(
                [0],
                [0],
                color="gray",
                linewidth=1.2,
                alpha=0.35,
                linestyle="-",
            ),
        )
        labels.insert(0, "individual profiles")
        ax.legend(handles, labels, loc="upper right", fontsize=11)

        fig.tight_layout(pad=1.2)
        fig.savefig(output_path, format="svg", dpi=300, bbox_inches="tight", facecolor="white")
        _show_figure_if_interactive(fig)

    print(f"Mirrored individual profile SVG saved to: {output_path}")
    return output_path


def _build_unfolded_dataset(
    output_dir: Path,
    *,
    source_key: str,
    stack_key: str,
    plot_title: str,
    figure_stem: str,
) -> dict[str, object]:
    stack_map = build_straightened_stack_map(STRAIGHTENER_CONFIGS[source_key])
    return {
        "input_source": stack_map[stack_key],
        "plot_title": plot_title,
        "source_key": source_key,
        "output_profile_svg": output_dir / f"{figure_stem}.svg",
        "output_profile_csv": output_dir / f"{figure_stem}_profile.csv",
        "output_peak_csv": output_dir / f"{figure_stem}_peak.csv",
    }


def run_figs8d_figure(fig_dir: Path) -> tuple[PeakAnalysisResult, ...]:
    output_dir = fig_dir / "02_FigS8d_plot_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Intermediate profile/peak CSVs are merged into FigS8d.csv and then deleted,
    # so only the two SVGs and the combined CSV persist.
    datasets = [
        _build_unfolded_dataset(
            output_dir,
            source_key="FigS8d_minus",
            stack_key="c2_FtsZ",
            plot_title="- MinCDE",
            figure_stem="FigS8d_minus",
        ),
        _build_unfolded_dataset(
            output_dir,
            source_key="FigS8d_plus",
            stack_key="c2_FtsZ",
            plot_title="+ MinCDE",
            figure_stem="FigS8d_plus",
        ),
    ]

    combined_csv_path = output_dir / "FigS8d.csv"
    results = run_unfolded_peak_figure(datasets, combined_csv_path)
    render_figs8d_compact_profile_variants(combined_csv_path, output_dir)
    render_figs8d_mirrored_individual_profiles(combined_csv_path, output_dir)
    return results


def run_figs11de_figure(fig_dir: Path) -> tuple[PeakAnalysisResult, ...]:
    """Render individual-only unfolded MinCDE profiles as FigS11d/e."""
    output_dir = Path(fig_dir) / "02_FigS11de_plot_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [
        _build_unfolded_dataset(
            output_dir,
            source_key=FIGS11D_KEY,
            stack_key="c2_FtsZ",
            plot_title="-MinCDE",
            figure_stem="FigS11d",
        ),
        _build_unfolded_dataset(
            output_dir,
            source_key=FIGS11E_KEY,
            stack_key="c2_FtsZ",
            plot_title="+MinCDE",
            figure_stem="FigS11e",
        ),
    ]
    return run_unfolded_peak_figure(
        datasets,
        output_dir / "FigS11de.csv",
        plot_mode="individual_only",
    )


def run_figs12cd_figure(fig_dir: Path) -> tuple[PeakAnalysisResult, ...]:
    """Render individual-only unfolded -MinC and -MinDE profiles as FigS12c/d."""
    output_dir = Path(fig_dir) / "02_FigS12cde_plot_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [
        _build_unfolded_dataset(
            output_dir,
            source_key=FIGS12C_KEY,
            stack_key="c2_FtsZ",
            plot_title="-MinC",
            figure_stem="FigS12c",
        ),
        _build_unfolded_dataset(
            output_dir,
            source_key=FIGS12D_KEY,
            stack_key="c3_FtsZ",
            plot_title="-MinDE",
            figure_stem="FigS12d",
        ),
    ]
    return run_unfolded_peak_figure(
        datasets,
        output_dir / "FigS12cd.csv",
        plot_mode="individual_only",
    )


def run_figs9a_figure(fig_dir: Path) -> tuple[PeakAnalysisResult, ...]:
    """FigS9a (- MinC, FtsZ channel) using the FigS8d unfolded peak method."""
    output_dir = fig_dir / "02_FigS9a_plot_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _build_unfolded_dataset(
        output_dir,
        source_key="FigS9a",
        stack_key="c2_FtsZ",
        plot_title="- MinC",
        figure_stem="FigS9a",
    )
    return run_unfolded_peak_figure([dataset], output_dir / "FigS9a.csv")


def run_figs9b_figure(fig_dir: Path) -> tuple[PeakAnalysisResult, ...]:
    """FigS9b (- MinDE, FtsZ channel) using the FigS8d unfolded peak method."""
    output_dir = fig_dir / "02_FigS9b_plot_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _build_unfolded_dataset(
        output_dir,
        source_key="FigS9b",
        stack_key="c3_FtsZ",
        plot_title="- MinDE",
        figure_stem="FigS9b",
    )
    return run_unfolded_peak_figure([dataset], output_dir / "FigS9b.csv")
