from __future__ import annotations

import os
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


DEFAULT_NUM_BINS = 20
FIG5G_MINUS_KEY = "Fig5g_minus"
FIG5G_PLUS_KEY = "Fig5g_plus"
FIGS8D_MINUS_KEY = "FigS8d_minus"
FIGS8D_PLUS_KEY = "FigS8d_plus"
FIGS9A_KEY = "FigS9a"
FIGS9B_KEY = "FigS9b"
FIGS12C_KEY = "FigS12c"
FIGS12D_KEY = "FigS12d"


@dataclass(frozen=True)
class SplitProfileResult:
    source_key: str
    condition_label: str
    output_svg: Path
    output_csv: Path
    frame_count: int
    half_profile_count: int


def calculate_folded_half_profile_table(
    input_source,
    *,
    num_bins: int = DEFAULT_NUM_BINS,
    normalize_intensity: bool = True,
) -> tuple[pd.DataFrame, int]:
    frames = to_grayscale_frames(input_source)
    records: list[dict[str, object]] = []
    bins = np.linspace(0, 1.0, num_bins + 1)
    bin_labels = (bins[:-1] + bins[1:]) / 2
    full_index = pd.Index(bin_labels, name="distance_from_center")

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
            expected_bin_pixels = image.shape[1] / 2 / num_bins

            for half_name, selector in (
                ("right", df_full["original_x"] >= 0),
                ("left", df_full["original_x"] < 0),
            ):
                df_half = df_full[selector].copy()
                if df_half.empty:
                    continue

                df_half["bin"] = pd.cut(
                    df_half["x_distance"],
                    bins=bins,
                    labels=bin_labels,
                    include_lowest=True,
                )
                half_profile = df_half.groupby("bin", observed=True).agg(
                    intensity=("intensity", "mean"),
                    pixel_count=("bin", "count"),
                )
                half_profile["intensity"] *= (
                    half_profile["pixel_count"] / expected_bin_pixels
                )

                raw_intensity = half_profile["intensity"].reindex(full_index, fill_value=0.0)
                pixel_count = half_profile["pixel_count"].reindex(full_index, fill_value=0)
                normalization_factor = float(raw_intensity.max())
                if normalization_factor > 0:
                    normalized_intensity = raw_intensity / normalization_factor
                else:
                    normalized_intensity = raw_intensity.copy()
                plot_intensity = (
                    normalized_intensity if normalize_intensity else raw_intensity
                )

                profile_name = f"{frame_name}_{half_name}"
                for distance_from_center in full_index:
                    records.append(
                        {
                            "frame_index": frame_index,
                            "half": half_name,
                            "profile_name": profile_name,
                            "distance_from_center": float(distance_from_center),
                            "raw_intensity": float(raw_intensity.loc[distance_from_center]),
                            "normalized_intensity": float(
                                normalized_intensity.loc[distance_from_center]
                            ),
                            "plot_intensity": float(plot_intensity.loc[distance_from_center]),
                            "normalization_factor": normalization_factor,
                            "pixel_count": int(pixel_count.loc[distance_from_center]),
                            "expected_bin_pixels": float(expected_bin_pixels),
                            "frame_height": int(image.shape[0]),
                            "frame_width": int(image.shape[1]),
                        }
                    )
        except Exception as exc:
            print(f"An error occurred while processing {frame_name}: {exc}")

    if not records:
        raise RuntimeError("No frames were processed successfully.")

    return pd.DataFrame.from_records(records), len(frames)


def _standard_error(values: pd.DataFrame | pd.Series, *, axis: int = 0):
    if isinstance(values, pd.Series):
        count = values.count()
        if count == 0:
            return 0.0
        se_value = values.std(ddof=1) / np.sqrt(count)
        if pd.isna(se_value):
            return 0.0
        return se_value

    counts = values.count(axis=axis)
    se_values = values.std(axis=axis, ddof=1) / np.sqrt(counts)
    if isinstance(se_values, pd.Series):
        return se_values.fillna(0.0)
    if pd.isna(se_values):
        return 0.0
    return se_values


def _unique_in_order(values: pd.Series) -> list[str]:
    return list(dict.fromkeys(values.astype(str).tolist()))


def analyze_split_profile_stack(
    input_source,
    output_svg_file,
    output_csv_file,
    num_bins=DEFAULT_NUM_BINS,
    normalize_intensity=True,
    plot_title=None,
    source_key=None,
    condition_label=None,
    render_plot=True,
):
    output_paths = [output_csv_file]
    if render_plot:
        output_paths.append(output_svg_file)
    for path in output_paths:
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
        profile_table, num_frames = calculate_folded_half_profile_table(
            input_source,
            num_bins=num_bins,
            normalize_intensity=normalize_intensity,
        )
        print(f"Loaded in-memory stack successfully (contains {num_frames} frames).")
        print("Initiating frame-by-frame split-profile analysis...")
    except Exception as exc:
        raise RuntimeError(f"Could not prepare input frames: {exc}") from exc

    print("\n" + "-" * 50)
    print("All profiles processed. Aggregating data...")

    bins = np.linspace(0, 1.0, num_bins + 1)
    full_index = pd.Index((bins[:-1] + bins[1:]) / 2, name="distance_from_center")
    processed_half_names = _unique_in_order(profile_table["profile_name"])
    df_all_halves = (
        profile_table.pivot(
            index="distance_from_center",
            columns="profile_name",
            values="plot_intensity",
        )
        .reindex(index=full_index, columns=processed_half_names)
        .fillna(0.0)
    )

    overall_mean = df_all_halves.mean(axis=1)
    results_df = df_all_halves.copy()
    results_df["overall_mean"] = overall_mean
    results_df.to_csv(output_csv_file)

    if render_plot:
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
        plt.close(fig)
    else:
        print("Skipping individual profile plot; merged output only.")

    print("\nAnalysis complete!")
    if render_plot:
        print(f"SVG plot saved to: {output_svg_file}")
    print(f"CSV data file saved to: {output_csv_file}")
    print("-" * 50)

    return SplitProfileResult(
        source_key=str(source_key) if source_key is not None else "",
        condition_label=str(
            condition_label if condition_label is not None else plot_title or ""
        ),
        output_svg=Path(output_svg_file),
        output_csv=Path(output_csv_file),
        frame_count=num_frames,
        half_profile_count=len(processed_half_names),
    )


def run_folded_split_profile_figure(
    datasets: list[dict[str, object]],
    combined_csv_path: Path,
    *,
    num_bins: int = DEFAULT_NUM_BINS,
    normalize_intensity: bool = True,
) -> tuple[SplitProfileResult, ...]:
    """Run the folded split-profile workflow and merge intermediate CSVs.

    This intentionally uses the legacy folded definition from
    ``analyze_split_profile_stack``: left and right halves are converted to
    distance-from-equator profiles and treated as separate half-profiles.
    """
    results: list[SplitProfileResult] = []
    for dataset in datasets:
        result = analyze_split_profile_stack(
            input_source=dataset["input_source"],
            output_svg_file=dataset["output_svg_file"],
            output_csv_file=dataset["output_csv_file"],
            num_bins=int(dataset.get("num_bins", num_bins)),
            normalize_intensity=bool(dataset.get("normalize_intensity", normalize_intensity)),
            plot_title=dataset["plot_title"],
            source_key=dataset["source_key"],
            condition_label=dataset.get("condition_label"),
            render_plot=bool(dataset.get("render_plot", True)),
        )
        results.append(result)

    combine_export_tables(
        Path(combined_csv_path),
        [
            {
                "path": result.output_csv,
                "metadata": {
                    "condition_key": result.source_key,
                    "condition_label": result.condition_label,
                },
            }
            for result in results
        ],
    )
    remove_files([result.output_csv for result in results])
    return tuple(results)


def _build_folded_dataset(
    output_dir: Path,
    *,
    source_key: str,
    stack_key: str,
    plot_title: str | None,
    condition_label: str | None = None,
    figure_stem: str,
    render_plot: bool = True,
) -> dict[str, object]:
    stack_map = build_straightened_stack_map(STRAIGHTENER_CONFIGS[source_key])
    return {
        "input_source": stack_map[stack_key],
        "plot_title": plot_title,
        "condition_label": condition_label if condition_label is not None else plot_title,
        "source_key": source_key,
        "output_svg_file": output_dir / f"{figure_stem}.svg",
        "output_csv_file": output_dir / f"{figure_stem}_profile.csv",
        "num_bins": DEFAULT_NUM_BINS,
        "normalize_intensity": True,
        "render_plot": render_plot,
    }


def _fig5g_datasets(output_dir: Path) -> list[dict[str, object]]:
    return [
        _build_folded_dataset(
            output_dir,
            source_key=FIG5G_MINUS_KEY,
            stack_key="c2_FtsZ",
            plot_title="- MinCDE",
            figure_stem="Fig5g_minus",
            render_plot=False,
        ),
        _build_folded_dataset(
            output_dir,
            source_key=FIG5G_PLUS_KEY,
            stack_key="c2_FtsZ",
            plot_title="+ MinCDE",
            figure_stem="Fig5g_plus",
            render_plot=False,
        ),
    ]


def _figs8d_datasets(output_dir: Path) -> list[dict[str, object]]:
    return [
        _build_folded_dataset(
            output_dir,
            source_key=FIGS8D_MINUS_KEY,
            stack_key="c2_FtsZ",
            plot_title="- MinCDE",
            figure_stem="FigS8d_minus",
        ),
        _build_folded_dataset(
            output_dir,
            source_key=FIGS8D_PLUS_KEY,
            stack_key="c2_FtsZ",
            plot_title="+ MinCDE",
            figure_stem="FigS8d_plus",
        ),
    ]


def _figs9a_datasets(output_dir: Path) -> list[dict[str, object]]:
    return [
        _build_folded_dataset(
            output_dir,
            source_key=FIGS9A_KEY,
            stack_key="c2_FtsZ",
            plot_title=None,
            condition_label="- MinC",
            figure_stem="FigS9a",
        )
    ]


def _figs9b_datasets(output_dir: Path) -> list[dict[str, object]]:
    return [
        _build_folded_dataset(
            output_dir,
            source_key=FIGS9B_KEY,
            stack_key="c3_FtsZ",
            plot_title=None,
            condition_label="- MinDE",
            figure_stem="FigS9b",
        )
    ]


def _figs12e_datasets(output_dir: Path) -> list[dict[str, object]]:
    return [
        _build_folded_dataset(
            output_dir,
            source_key=FIGS12C_KEY,
            stack_key="c2_FtsZ",
            plot_title="-MinC",
            figure_stem="FigS12c_folded",
            render_plot=False,
        ),
        _build_folded_dataset(
            output_dir,
            source_key=FIGS12D_KEY,
            stack_key="c3_FtsZ",
            plot_title="-MinDE",
            figure_stem="FigS12d_folded",
            render_plot=False,
        ),
    ]


def _split_profile_frame_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if str(column).startswith("frame_")
        and (str(column).endswith("_left") or str(column).endswith("_right"))
    ]


def build_folded_profile_se_audit_table(
    datasets: list[dict[str, object]],
    output_csv_path: Path | None = None,
    *,
    num_bins: int = DEFAULT_NUM_BINS,
) -> pd.DataFrame:
    audit_tables: list[pd.DataFrame] = []
    for dataset in datasets:
        profile_table, frame_count = calculate_folded_half_profile_table(
            dataset["input_source"],
            num_bins=int(dataset.get("num_bins", num_bins)),
            normalize_intensity=True,
        )
        condition_key = str(dataset["source_key"])
        condition_label = str(dataset["plot_title"])

        summary_records = []
        for distance_from_center, distance_df in profile_table.groupby(
            "distance_from_center", sort=True
        ):
            raw_values = distance_df["raw_intensity"]
            normalized_values = distance_df["normalized_intensity"]
            n_half_profiles = int(normalized_values.count())
            normalized_mean = float(normalized_values.mean())
            normalized_sd = float(normalized_values.std(ddof=1))
            normalized_se = float(_standard_error(normalized_values))
            raw_mean = float(raw_values.mean())
            raw_sd = float(raw_values.std(ddof=1))
            raw_se = float(_standard_error(raw_values))
            summary_records.append(
                {
                    "condition_key": condition_key,
                    "condition_label": condition_label,
                    "distance_from_center": float(distance_from_center),
                    "n_frames": int(frame_count),
                    "n_half_profiles": n_half_profiles,
                    "normalized_mean": normalized_mean,
                    "normalized_sd": normalized_sd,
                    "normalized_se": normalized_se,
                    "normalized_mean_minus_se": max(normalized_mean - normalized_se, 0.0),
                    "normalized_mean_plus_se": normalized_mean + normalized_se,
                    "normalized_min": float(normalized_values.min()),
                    "normalized_q25": float(normalized_values.quantile(0.25)),
                    "normalized_median": float(normalized_values.median()),
                    "normalized_q75": float(normalized_values.quantile(0.75)),
                    "normalized_max": float(normalized_values.max()),
                    "normalized_count_gt_0_5": int((normalized_values > 0.5).sum()),
                    "normalized_count_gt_0_8": int((normalized_values > 0.8).sum()),
                    "normalized_count_eq_1": int(np.isclose(normalized_values, 1.0).sum()),
                    "raw_mean": raw_mean,
                    "raw_sd": raw_sd,
                    "raw_se": raw_se,
                    "raw_min": float(raw_values.min()),
                    "raw_median": float(raw_values.median()),
                    "raw_max": float(raw_values.max()),
                }
            )

        summary_table = pd.DataFrame.from_records(summary_records).set_index(
            "distance_from_center"
        )
        raw_wide = profile_table.pivot(
            index="distance_from_center",
            columns="profile_name",
            values="raw_intensity",
        ).add_prefix("raw_")
        normalized_wide = profile_table.pivot(
            index="distance_from_center",
            columns="profile_name",
            values="normalized_intensity",
        ).add_prefix("normalized_")
        audit_table = (
            summary_table.join(raw_wide)
            .join(normalized_wide)
            .reset_index()
            .sort_values(["condition_key", "distance_from_center"])
        )
        audit_tables.append(audit_table)

    combined = pd.concat(audit_tables, ignore_index=True, sort=False)
    if output_csv_path is not None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_csv_path, index=False)
        print(f"SE audit table saved to: {output_csv_path}")
    return combined


def build_fig5g_se_audit_table(fig_dir: Path) -> pd.DataFrame:
    output_dir = Path(fig_dir) / "02_Fig5g_plot_output"
    return build_folded_profile_se_audit_table(
        _fig5g_datasets(output_dir),
        output_dir / "Fig5g_SE_audit_table.csv",
    )


def build_figs8d_se_audit_table(fig_dir: Path) -> pd.DataFrame:
    output_dir = Path(fig_dir) / "02_FigS8d_plot_output"
    return build_folded_profile_se_audit_table(
        _figs8d_datasets(output_dir),
        output_dir / "FigS8d_SE_audit_table.csv",
    )


def build_figs9a_se_audit_table(fig_dir: Path) -> pd.DataFrame:
    output_dir = Path(fig_dir) / "02_FigS9a_plot_output"
    return build_folded_profile_se_audit_table(
        _figs9a_datasets(output_dir),
        output_dir / "FigS9a_SE_audit_table.csv",
    )


def build_figs9b_se_audit_table(fig_dir: Path) -> pd.DataFrame:
    output_dir = Path(fig_dir) / "02_FigS9b_plot_output"
    return build_folded_profile_se_audit_table(
        _figs9b_datasets(output_dir),
        output_dir / "FigS9b_SE_audit_table.csv",
    )


def build_figs12e_se_audit_table(fig_dir: Path) -> pd.DataFrame:
    output_dir = Path(fig_dir) / "02_FigS12cde_plot_output"
    return build_folded_profile_se_audit_table(
        _figs12e_datasets(output_dir),
        output_dir / "FigS12e_SE_audit_table.csv",
    )


def build_figs9a_figs9b_se_audit_table(repo_root: Path) -> pd.DataFrame:
    repo_root = Path(repo_root)
    figs9a_output_dir = repo_root / "FigS9a/02_FigS9a_plot_output"
    figs9b_output_dir = repo_root / "FigS9b/02_FigS9b_plot_output"
    return build_folded_profile_se_audit_table(
        [
            *_figs9a_datasets(figs9a_output_dir),
            *_figs9b_datasets(figs9b_output_dir),
        ],
        figs9a_output_dir / "FigS9a_FigS9b_SE_audit_table.csv",
    )


def _load_folded_profile_source(source) -> pd.DataFrame:
    if isinstance(source, dict):
        path = Path(source["path"])
        metadata = dict(source.get("metadata", {}))
    else:
        path = Path(source)
        metadata = {}

    df = pd.read_csv(path)
    for key, value in metadata.items():
        if key not in df.columns:
            df.insert(0, key, value)
        else:
            df[key] = value
    return df


def load_folded_profile_summaries(csv_paths: list[object] | tuple[object, ...]) -> dict[str, dict[str, object]]:
    dataframes = []
    for source in csv_paths:
        df = _load_folded_profile_source(source)
        if "condition_key" not in df.columns or "condition_label" not in df.columns:
            raise ValueError(f"{source} is missing condition metadata.")
        dataframes.append(df)

    combined = pd.concat(dataframes, ignore_index=True, sort=False)
    summaries: dict[str, dict[str, object]] = {}
    for condition_key, condition_df in combined.groupby("condition_key", sort=False):
        condition_df = condition_df.sort_values("distance_from_center")
        frame_columns = _split_profile_frame_columns(condition_df)
        if not frame_columns:
            raise ValueError(f"No folded profile frame columns found for {condition_key}.")

        frame_values = condition_df[frame_columns].apply(pd.to_numeric, errors="coerce")
        mean_values = pd.to_numeric(condition_df["overall_mean"], errors="coerce")
        if mean_values.isna().all():
            mean_values = frame_values.mean(axis=1)

        summaries[str(condition_key)] = {
            "label": str(condition_df["condition_label"].iloc[0]),
            "distance": pd.to_numeric(
                condition_df["distance_from_center"], errors="coerce"
            ).to_numpy(dtype=float),
            "mean": mean_values.to_numpy(dtype=float),
            "se": _standard_error(frame_values, axis=1).to_numpy(dtype=float),
            "sd": frame_values.std(axis=1, ddof=1).fillna(0.0).to_numpy(dtype=float),
            "frames": frame_values.to_numpy(dtype=float),
            "half_profile_count": int(frame_values.notna().sum(axis=1).max()),
        }

    return summaries


def _style_merged_folded_axis(ax) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.tick_params(axis="both", which="major", width=1.8, length=6, labelsize=12)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def render_merged_folded_profiles(
    csv_paths: list[object] | tuple[object, ...],
    output_svg_path: Path,
    *,
    condition_order: tuple[str, ...],
    title: str | None = "Merged folded intensity profiles",
    colors: dict[str, str] | None = None,
    spread: str = "se",
) -> Path:
    spread = spread.lower()
    if spread not in {"se", "sd"}:
        raise ValueError(f"Unsupported spread type: {spread}")

    summaries = load_folded_profile_summaries(csv_paths)
    missing_keys = [condition_key for condition_key in condition_order if condition_key not in summaries]
    if missing_keys:
        raise ValueError(f"Missing folded profile condition(s): {', '.join(missing_keys)}")

    output_svg_path = Path(output_svg_path)
    output_svg_path.parent.mkdir(parents=True, exist_ok=True)
    colors = colors or {}
    default_colors = ("#1f77b4", "#ff7f0e", "#0F4D92", "#B64342")
    spread_label = spread.upper()

    with plt.rc_context(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 14,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    ):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        max_upper = 0.0
        for index, condition_key in enumerate(condition_order):
            summary = summaries[condition_key]
            x = summary["distance"]
            mean = summary["mean"]
            spread_values = summary[spread]
            color = colors.get(condition_key, default_colors[index % len(default_colors)])
            lower = np.clip(mean - spread_values, 0.0, None)
            upper = mean + spread_values
            max_upper = max(max_upper, float(np.nanmax(upper)))
            ax.fill_between(
                x,
                lower,
                upper,
                color=color,
                alpha=0.16,
                linewidth=0,
            )
            ax.plot(
                x,
                mean,
                color=color,
                linewidth=2.8,
                label=f"{summary['label']} mean & {spread_label}",
            )

        _style_merged_folded_axis(ax)
        ax.set_ylim(0.0, max(1.05, max_upper * 1.05))
        ax.set_xlabel("Normalized distance from equator", fontsize=14)
        ax.set_ylabel("Peak-normalized intensity", fontsize=14)
        if title:
            ax.set_title(title, fontsize=16, pad=10)
        ax.legend(loc="upper right", fontsize=12)
        fig.tight_layout(pad=1.2)
        fig.savefig(output_svg_path, format="svg", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    print(f"Merged folded profile ({spread_label}) SVG saved to: {output_svg_path}")
    return output_svg_path


def run_fig5g_figure(fig_dir: Path) -> tuple[SplitProfileResult, ...]:
    output_dir = Path(fig_dir) / "02_Fig5g_plot_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_svg in (
        output_dir / "Fig5g_minus.svg",
        output_dir / "Fig5g_plus.svg",
        output_dir / "Fig5g_merged_profiles_SD.svg",
    ):
        stale_svg.unlink(missing_ok=True)
    datasets = _fig5g_datasets(output_dir)
    combined_csv_path = output_dir / "Fig5g.csv"
    results = run_folded_split_profile_figure(datasets, combined_csv_path)
    render_merged_folded_profiles(
        [combined_csv_path],
        output_dir / "Fig5g_merged_profiles.svg",
        condition_order=(FIG5G_MINUS_KEY, FIG5G_PLUS_KEY),
        title=None,
    )
    return results


def run_figs12e_figure(fig_dir: Path) -> tuple[SplitProfileResult, ...]:
    """Render only the merged folded -MinC/-MinDE profile as FigS12e."""
    output_dir = Path(fig_dir) / "02_FigS12cde_plot_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_svg in (
        output_dir / "FigS12c_folded.svg",
        output_dir / "FigS12d_folded.svg",
        output_dir / "FigS12e_SD.svg",
    ):
        stale_svg.unlink(missing_ok=True)

    datasets = _figs12e_datasets(output_dir)
    combined_csv_path = output_dir / "FigS12e.csv"
    results = run_folded_split_profile_figure(datasets, combined_csv_path)
    render_merged_folded_profiles(
        [combined_csv_path],
        output_dir / "FigS12e.svg",
        condition_order=(FIGS12C_KEY, FIGS12D_KEY),
        title=None,
        colors={FIGS12C_KEY: "#1f77b4", FIGS12D_KEY: "#ff7f0e"},
    )
    return results


def run_figs8d_figure(fig_dir: Path) -> tuple[SplitProfileResult, ...]:
    output_dir = Path(fig_dir) / "02_FigS8d_plot_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = _figs8d_datasets(output_dir)
    combined_csv_path = output_dir / "FigS8d.csv"
    results = run_folded_split_profile_figure(datasets, combined_csv_path)
    render_merged_folded_profiles(
        [combined_csv_path],
        output_dir / "FigS8d_merged_profiles.svg",
        condition_order=(FIGS8D_MINUS_KEY, FIGS8D_PLUS_KEY),
        title="Merged folded intensity profiles",
    )
    return results


def run_figs9a_figure(fig_dir: Path) -> tuple[SplitProfileResult, ...]:
    output_dir = Path(fig_dir) / "02_FigS9a_plot_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _figs9a_datasets(output_dir)[0]
    return (
        analyze_split_profile_stack(
            input_source=dataset["input_source"],
            output_svg_file=dataset["output_svg_file"],
            output_csv_file=output_dir / "FigS9a.csv",
            num_bins=int(dataset.get("num_bins", DEFAULT_NUM_BINS)),
            normalize_intensity=bool(dataset.get("normalize_intensity", True)),
            plot_title=dataset["plot_title"],
            source_key=dataset["source_key"],
            condition_label=dataset["condition_label"],
        ),
    )


def run_figs9b_figure(fig_dir: Path) -> tuple[SplitProfileResult, ...]:
    output_dir = Path(fig_dir) / "02_FigS9b_plot_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _figs9b_datasets(output_dir)[0]
    return (
        analyze_split_profile_stack(
            input_source=dataset["input_source"],
            output_svg_file=dataset["output_svg_file"],
            output_csv_file=output_dir / "FigS9b.csv",
            num_bins=int(dataset.get("num_bins", DEFAULT_NUM_BINS)),
            normalize_intensity=bool(dataset.get("normalize_intensity", True)),
            plot_title=dataset["plot_title"],
            source_key=dataset["source_key"],
            condition_label=dataset["condition_label"],
        ),
    )


def render_figs9a_figs9b_merged_profiles(repo_root: Path) -> Path:
    repo_root = Path(repo_root)
    figs9a_csv = repo_root / "FigS9a/02_FigS9a_plot_output/FigS9a.csv"
    figs9b_csv = repo_root / "FigS9b/02_FigS9b_plot_output/FigS9b.csv"
    output_path = (
        repo_root
        / "FigS9a/02_FigS9a_plot_output/FigS9a_FigS9b_merged_profiles.svg"
    )
    return render_merged_folded_profiles(
        [
            {
                "path": figs9a_csv,
                "metadata": {
                    "condition_key": FIGS9A_KEY,
                    "condition_label": "- MinC",
                },
            },
            {
                "path": figs9b_csv,
                "metadata": {
                    "condition_key": FIGS9B_KEY,
                    "condition_label": "- MinDE",
                },
            },
        ],
        output_path,
        condition_order=(FIGS9A_KEY, FIGS9B_KEY),
        title="Merged folded intensity profiles",
        colors={FIGS9A_KEY: "#1f77b4", FIGS9B_KEY: "#ff7f0e"},
    )
