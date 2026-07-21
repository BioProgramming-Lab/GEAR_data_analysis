from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

from .kymograph_common import (
    align_projections_to_center,
    calculate_projections,
    save_heatmap,
)
from .straightener_common import (
    StraightenerConfig,
    extract_skeleton_and_longest_path,
    process_single_image_set,
)


MIN_PERIOD_SECONDS = 90.0
PEAK_PERIOD_CUTOFF_MULTIPLIER = 2.0
SCALE_BAR_LENGTH_UM = 2.0


@dataclass(frozen=True)
class FigS10DatasetConfig:
    sample_id: str
    source_group: str
    kymograph_panel: str
    fft_panel: str
    raw_subdir: str
    frame_interval_s: float
    expected_c1_frames: int
    expected_c2_frames: int
    expected_c3_frames: int
    expected_analysis_frames: int
    expected_length_um: float
    scale_reference_index: int
    start_preference: str = "right"


@dataclass(frozen=True)
class FigS10Result:
    output_dir: Path
    csv_path: Path
    svg_paths: tuple[Path, ...]
    sample_summary: pd.DataFrame
    regression_summary: pd.DataFrame


FIGS10_DATASETS = (
    FigS10DatasetConfig(
        sample_id="sample_01",
        source_group="oscillation-1",
        kymograph_panel="FigS10a",
        fft_panel="FigS10e",
        raw_subdir="sample_01",
        frame_interval_s=30.0,
        expected_c1_frames=57,
        expected_c2_frames=57,
        expected_c3_frames=57,
        expected_analysis_frames=57,
        expected_length_um=3.01715272673,
        scale_reference_index=1,
    ),
    FigS10DatasetConfig(
        sample_id="sample_02",
        source_group="oscillation-2",
        kymograph_panel="FigS10b",
        fft_panel="FigS10f",
        raw_subdir="sample_02",
        frame_interval_s=30.0,
        expected_c1_frames=74,
        expected_c2_frames=74,
        expected_c3_frames=74,
        expected_analysis_frames=74,
        expected_length_um=6.5239986032,
        scale_reference_index=2,
    ),
    FigS10DatasetConfig(
        sample_id="sample_04",
        source_group="oscillation-4",
        kymograph_panel="FigS10c",
        fft_panel="FigS10g",
        raw_subdir="sample_04",
        frame_interval_s=30.0,
        expected_c1_frames=62,
        expected_c2_frames=62,
        expected_c3_frames=62,
        expected_analysis_frames=62,
        expected_length_um=5.35629224325,
        scale_reference_index=4,
    ),
    FigS10DatasetConfig(
        sample_id="sample_06",
        source_group="oscillation-6",
        kymograph_panel="FigS10d",
        fft_panel="FigS10h",
        raw_subdir="sample_06",
        frame_interval_s=15.0,
        expected_c1_frames=149,
        expected_c2_frames=149,
        expected_c3_frames=134,
        expected_analysis_frames=134,
        expected_length_um=5.65879929925,
        scale_reference_index=6,
    ),
)


FRAME_PATTERN = re.compile(r"_t0*(\d+)c([123])$", re.IGNORECASE)


def _channel_frame_index(channel_dir: Path, channel_number: int) -> dict[int, Path]:
    frame_map: dict[int, Path] = {}
    for path in channel_dir.glob("*.tif"):
        match = FRAME_PATTERN.search(path.stem)
        if match is None or int(match.group(2)) != channel_number:
            continue
        frame_index = int(match.group(1))
        if frame_index in frame_map:
            raise ValueError(f"Duplicate frame {frame_index} in {channel_dir}")
        frame_map[frame_index] = path
    if not frame_map:
        raise FileNotFoundError(f"No channel-{channel_number} TIFF files found in {channel_dir}")
    return frame_map


def _read_filament_annotation(json_path: Path) -> tuple[dict, dict]:
    with json_path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)
    shapes = [
        shape
        for shape in annotation.get("shapes", [])
        if str(shape.get("label", "")).strip().casefold() == "filament"
    ]
    if len(shapes) != 1:
        raise ValueError(f"Expected one 'filament' polygon in {json_path}, found {len(shapes)}")
    return annotation, shapes[0]


def _load_scale_width_px(raw_dir: Path, index: int) -> float:
    json_path = raw_dir / "scale_ref" / f"ref-{index}.json"
    with json_path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)
    shapes = annotation.get("shapes", [])
    if len(shapes) != 1:
        raise ValueError(f"Expected one scale-bar annotation in {json_path}")
    x_coordinates = np.asarray(shapes[0]["points"], dtype=float)[:, 0]
    width_px = float(np.max(x_coordinates) - np.min(x_coordinates))
    if width_px <= 0:
        raise ValueError(f"Invalid scale-bar width in {json_path}")
    return width_px


def _measure_centerline_length_um(
    c1_path: Path,
    raw_dir: Path,
    scale_reference_index: int,
) -> tuple[float, float, float]:
    annotation, filament_shape = _read_filament_annotation(c1_path.with_suffix(".json"))
    height = int(annotation["imageHeight"])
    width = int(annotation["imageWidth"])
    polygon = np.asarray(filament_shape["points"], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    path = np.asarray(
        extract_skeleton_and_longest_path(mask, start_preference="top"),
        dtype=float,
    )
    if len(path) < 2:
        raise ValueError(f"Invalid centerline extracted from {c1_path}")
    length_px = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    scale_width_px = _load_scale_width_px(raw_dir, scale_reference_index)
    micrometers_per_pixel = SCALE_BAR_LENGTH_UM / scale_width_px
    return length_px * micrometers_per_pixel, length_px, micrometers_per_pixel


def _prepare_dataset(figure_dir: Path, config: FigS10DatasetConfig) -> dict:
    raw_dir = figure_dir / "00_FigS10_raw"
    sample_dir = raw_dir / config.raw_subdir
    channel_dirs = {
        1: sample_dir / "c1_membrane",
        2: sample_dir / "c2_MinC",
        3: sample_dir / "c3_MreB",
    }
    channel_maps = {
        channel: _channel_frame_index(directory, channel)
        for channel, directory in channel_dirs.items()
    }
    expected_counts = {
        1: config.expected_c1_frames,
        2: config.expected_c2_frames,
        3: config.expected_c3_frames,
    }
    measured_counts = {channel: len(frame_map) for channel, frame_map in channel_maps.items()}
    if measured_counts != expected_counts:
        raise ValueError(
            f"{config.sample_id} channel counts differ from the audited manifest: "
            f"measured={measured_counts}, expected={expected_counts}"
        )

    common_frames = sorted(set.intersection(*(set(frame_map) for frame_map in channel_maps.values())))
    if len(common_frames) != config.expected_analysis_frames:
        raise ValueError(
            f"{config.sample_id} has {len(common_frames)} common frames; "
            f"expected {config.expected_analysis_frames}"
        )
    if common_frames != list(range(common_frames[0], common_frames[-1] + 1)):
        raise ValueError(
            f"{config.sample_id} contains internal frame gaps; do not compress the time axis"
        )
    straightener_config = StraightenerConfig(
        key=config.sample_id,
        target_dir=channel_dirs[1],
        mapping_dirs=(channel_dirs[2],),
        start_preference=config.start_preference,
    )
    c2_projections: list[np.ndarray] = []
    for frame_index in common_frames:
        c1_path = channel_maps[1][frame_index]
        c1_json = c1_path.with_suffix(".json")
        if not c1_json.exists():
            raise FileNotFoundError(f"Missing filament annotation: {c1_json}")
        processed = process_single_image_set(
            {
                "target_img_path": c1_path,
                "target_json_path": c1_json,
                "mapping_img_paths": (channel_maps[2][frame_index],),
            },
            straightener_config,
        )
        c2_gray = cv2.cvtColor(processed.mapping_straight_rgb[0], cv2.COLOR_RGB2GRAY)
        c2_projections.append(calculate_projections(c2_gray, axis=0))

    aligned_time_by_position = align_projections_to_center(c2_projections)
    raw_matrix = np.transpose(aligned_time_by_position)
    raw_max = float(np.nanmax(raw_matrix))
    normalized_matrix = raw_matrix / raw_max if raw_max > 0 else raw_matrix.copy()
    times_s = np.arange(len(common_frames), dtype=float) * config.frame_interval_s

    length_um, length_px, micrometers_per_pixel = _measure_centerline_length_um(
        channel_maps[1][common_frames[0]],
        raw_dir,
        config.scale_reference_index,
    )
    if not np.isclose(length_um, config.expected_length_um, atol=5e-10):
        raise ValueError(
            f"{config.sample_id} length {length_um:.12g} um differs from "
            f"audited value {config.expected_length_um:.12g} um"
        )

    return {
        "config": config,
        "common_frames": common_frames,
        "times_s": times_s,
        "raw_matrix": raw_matrix,
        "normalized_matrix": normalized_matrix,
        "raw_global_max": raw_max,
        "length_um": length_um,
        "length_px": length_px,
        "micrometers_per_pixel": micrometers_per_pixel,
        "dropped_frames": {
            channel: sorted(set(frame_map) - set(common_frames))
            for channel, frame_map in channel_maps.items()
        },
    }


def _interpolate_nan_rows(matrix: np.ndarray) -> np.ndarray:
    prepared = np.asarray(matrix, dtype=float).copy()
    frame_axis = np.arange(prepared.shape[1])
    for row_index in range(prepared.shape[0]):
        row = prepared[row_index]
        valid = np.isfinite(row)
        if valid.all():
            continue
        if not valid.any():
            prepared[row_index] = 0.0
        elif valid.sum() == 1:
            prepared[row_index] = row[valid][0]
        else:
            prepared[row_index] = np.interp(frame_axis, frame_axis[valid], row[valid])
    return prepared


def calculate_center_of_mass_trace(matrix: np.ndarray) -> np.ndarray:
    prepared = _interpolate_nan_rows(matrix)
    weights = np.clip(prepared, a_min=0.0, a_max=None)
    position_axis = np.linspace(-1.0, 1.0, weights.shape[0])[:, None]
    denominator = np.sum(weights, axis=0)
    trace = np.sum(position_axis * weights, axis=0) / (denominator + 1e-12)
    valid = np.isfinite(trace)
    if not valid.any():
        raise ValueError("Center-of-mass trace contains no finite values")
    if not valid.all():
        trace = np.interp(np.arange(len(trace)), np.flatnonzero(valid), trace[valid])
    return trace


def calculate_fft(times_s: np.ndarray, trace: np.ndarray) -> dict:
    if len(times_s) < 8:
        raise ValueError("At least eight frames are required for FFT analysis")
    dt_values = np.diff(times_s)
    dt_seconds = float(np.median(dt_values))
    if not np.allclose(dt_values, dt_seconds, atol=1e-6):
        raise ValueError("FFT input must be evenly sampled")

    detrended_trace = signal.detrend(trace, type="linear")
    detrended_trace = detrended_trace - np.mean(detrended_trace)
    window = np.hanning(len(detrended_trace))
    frequencies_hz = np.fft.rfftfreq(len(detrended_trace), d=dt_seconds)
    power = np.abs(np.fft.rfft(detrended_trace * window)) ** 2

    valid_frequency = frequencies_hz > 0
    periods_s = np.full_like(frequencies_hz, np.inf, dtype=float)
    periods_s[valid_frequency] = 1.0 / frequencies_hz[valid_frequency]
    search_mask = valid_frequency & (periods_s >= MIN_PERIOD_SECONDS)
    search_indices = np.flatnonzero(search_mask)
    if len(search_indices) == 0:
        raise ValueError("No FFT bins satisfy the minimum period")
    peak_index = int(search_indices[np.argmax(power[search_mask])])
    display_max_period_s = float(periods_s[peak_index] * PEAK_PERIOD_CUTOFF_MULTIPLIER)
    display_mask = search_mask & (periods_s <= display_max_period_s)
    band_power = float(np.sum(power[display_mask]))

    return {
        "detrended_trace": detrended_trace,
        "frequencies_hz": frequencies_hz,
        "periods_s": periods_s,
        "power": power,
        "display_mask": display_mask,
        "peak_index": peak_index,
        "dt_seconds": dt_seconds,
        "observed_duration_s": float(times_s[-1] - times_s[0]),
        "fft_record_length_s": float(len(times_s) * dt_seconds),
        "peak_frequency_hz": float(frequencies_hz[peak_index]),
        "peak_period_s": float(periods_s[peak_index]),
        "peak_power": float(power[peak_index]),
        "peak_band_power_fraction": (
            float(power[peak_index] / band_power) if band_power > 0 else 0.0
        ),
        "display_max_period_s": display_max_period_s,
    }


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.tick_params(axis="both", direction="out", length=4.5, width=1.3)
    ax.grid(False)


def _plot_kymograph_panel(data: dict, output_svg: Path) -> None:
    config: FigS10DatasetConfig = data["config"]
    save_heatmap(
        projection_data=data["normalized_matrix"],
        save_path=output_svg,
        colormap="Reds_Transparent",
        time_interval=config.frame_interval_s,
        fig_aspect_ratio=4 / 1,
        time_xlabel="Time (s)",
        y_labels=("Pole 1", "Equator", "Pole 2"),
    )


def _relative_fft_coordinates(fft_data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(fft_data["display_mask"])
    periods_min = fft_data["periods_s"][indices] / 60.0
    power = fft_data["power"][indices]
    order = np.argsort(periods_min)
    periods_min = periods_min[order]
    power = power[order]
    indices = indices[order]
    relative_power = power / np.max(power) if float(np.max(power)) > 0 else power
    return periods_min, relative_power, indices


def _plot_fft_panel(fft_data: dict, output_svg: Path) -> None:
    periods_min, relative_power, _ = _relative_fft_coordinates(fft_data)
    peak_period_min = fft_data["peak_period_s"] / 60.0
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 1.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        ax.plot(
            periods_min,
            relative_power,
            color="#2F6F9F",
            linewidth=2.0,
            marker="o",
            markersize=4,
            markerfacecolor="white",
            markeredgewidth=1.0,
        )
        ax.axvline(
            peak_period_min,
            color="#A33F3F",
            linestyle="--",
            linewidth=1.5,
            label=f"Peak: {peak_period_min:.2f} min",
        )
        ax.set_xlabel("Period (min)")
        ax.set_ylabel("Relative FFT power")
        ax.set_xlim(float(np.min(periods_min)), float(np.max(periods_min)))
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=9)
        _style_axis(ax)
        fig.savefig(output_svg, format="svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)


def _plot_length_period_panel(source: pd.DataFrame, output_svg: Path) -> tuple[object, pd.DataFrame]:
    selected = source[source["selection_pass"].astype(bool)].copy()
    x = selected["gear_length_um"].to_numpy(dtype=float)
    y = selected["oscillation_period_s"].to_numpy(dtype=float)
    fit = stats.linregress(x, y)
    r_squared = float(fit.rvalue**2)
    x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    y_fit = fit.intercept + fit.slope * x_fit

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.linewidth": 1.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        # Keep the source workflow's 367.2 pt SVG width under current Matplotlib rounding.
        fig, ax = plt.subplots(figsize=(5.1000001, 4.5))
        ax.scatter(
            x,
            y,
            s=48,
            marker="o",
            color="black",
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        ax.plot(
            x_fit,
            y_fit,
            color="#555555",
            linestyle="--",
            linewidth=1.7,
            zorder=2,
        )
        ax.text(
            0.05,
            0.94,
            rf"$R^2$ = {r_squared:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
        )
        ax.set_xlabel(r"Filament length ($\mu$m)", fontsize=13, fontweight="bold", labelpad=8)
        ax.set_ylabel(
            "Oscillation period (s)",
            fontsize=13,
            fontweight="bold",
            labelpad=8,
        )
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 800)
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_yticks([0, 200, 400, 600, 800])
        ax.tick_params(axis="both", direction="out", length=6, width=1.5, labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.grid(False)
        fig.subplots_adjust(left=0.22, bottom=0.18, right=0.97, top=0.96)
        fig.savefig(output_svg, format="svg", facecolor="white")
        plt.close(fig)

    regression = pd.DataFrame(
        [
            {
                "model": "ordinary least-squares linear regression",
                "equation": "oscillation_period_s = intercept + slope * gear_length_um",
                "n": len(selected),
                "slope_seconds_per_um": float(fit.slope),
                "intercept_seconds": float(fit.intercept),
                "pearson_r": float(fit.rvalue),
                "r_squared": r_squared,
                "p_value_slope": float(fit.pvalue),
                "slope_standard_error": float(fit.stderr),
                "intercept_standard_error": float(fit.intercept_stderr),
            }
        ]
    )
    return fit, regression


def _common_identifiers(data: dict) -> dict:
    config: FigS10DatasetConfig = data["config"]
    return {
        "sample_id": config.sample_id,
        "source_group": config.source_group,
        "kymograph_panel": config.kymograph_panel,
        "fft_panel": config.fft_panel,
        "raw_data_in_repository": True,
    }


def _build_kymograph_table(data: dict) -> pd.DataFrame:
    matrix_raw = data["raw_matrix"]
    matrix_norm = data["normalized_matrix"]
    position_count, frame_count = matrix_raw.shape
    position_index = np.repeat(np.arange(position_count), frame_count)
    frame_offset = np.tile(np.arange(frame_count), position_count)
    frame_number = np.asarray(data["common_frames"], dtype=int)[frame_offset]
    time_s = np.asarray(data["times_s"], dtype=float)[frame_offset]
    table = pd.DataFrame(
        {
            "table_type": "kymograph",
            "channel": "c2_MinC",
            "position_index": position_index,
            "position_normalized": (
                position_index / (position_count - 1) if position_count > 1 else 0.0
            ),
            "frame_number": frame_number,
            "time_s": time_s,
            "intensity_raw": matrix_raw.reshape(-1),
            "intensity_normalized": matrix_norm.reshape(-1),
            "normalization": "full-series global maximum",
        }
    )
    for key, value in _common_identifiers(data).items():
        table[key] = value
    return table


def _build_trace_table(data: dict, center_of_mass: np.ndarray, fft_data: dict) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "table_type": "center_of_mass_trace",
            "frame_number": data["common_frames"],
            "time_s": data["times_s"],
            "center_of_mass_position": center_of_mass,
            "detrended_center_of_mass": fft_data["detrended_trace"],
            "trace_definition": "intensity-weighted center of mass across all c2_MinC position bins",
        }
    )
    for key, value in _common_identifiers(data).items():
        table[key] = value
    return table


def _build_fft_table(data: dict, fft_data: dict) -> pd.DataFrame:
    periods_min, relative_power, indices = _relative_fft_coordinates(fft_data)
    table = pd.DataFrame(
        {
            "table_type": "fft_spectrum",
            "fft_bin_index": indices,
            "frequency_hz": fft_data["frequencies_hz"][indices],
            "period_s": fft_data["periods_s"][indices],
            "period_min": periods_min,
            "fft_power": fft_data["power"][indices],
            "relative_fft_power": relative_power,
            "is_peak": indices == fft_data["peak_index"],
            "fft_preprocessing": "linear detrend; mean subtraction; Hann window; rFFT",
            "minimum_period_s": MIN_PERIOD_SECONDS,
            "display_cutoff_rule": "2x main peak period after peak selection",
        }
    )
    for key, value in _common_identifiers(data).items():
        table[key] = value
    return table


def _build_four_sample_summary(data: dict, fft_data: dict) -> dict:
    config: FigS10DatasetConfig = data["config"]
    dropped = data["dropped_frames"]
    return {
        "table_type": "sample_summary",
        **_common_identifiers(data),
        "frame_interval_s": config.frame_interval_s,
        "raw_c1_frame_count": config.expected_c1_frames,
        "raw_c2_frame_count": config.expected_c2_frames,
        "raw_c3_frame_count": config.expected_c3_frames,
        "analysis_frame_count": len(data["common_frames"]),
        "analysis_frame_start": data["common_frames"][0],
        "analysis_frame_end": data["common_frames"][-1],
        "dropped_c1_frames": ";".join(map(str, dropped[1])),
        "dropped_c2_frames": ";".join(map(str, dropped[2])),
        "dropped_c3_frames": ";".join(map(str, dropped[3])),
        "raw_global_max": data["raw_global_max"],
        "gear_length_um": data["length_um"],
        "centerline_length_px": data["length_px"],
        "micrometers_per_pixel": data["micrometers_per_pixel"],
        "peak_frequency_hz": fft_data["peak_frequency_hz"],
        "peak_period_s": fft_data["peak_period_s"],
        "peak_period_min": fft_data["peak_period_s"] / 60.0,
        "peak_fft_power": fft_data["peak_power"],
        "peak_band_power_fraction": fft_data["peak_band_power_fraction"],
        "fft_record_length_s": fft_data["fft_record_length_s"],
        "observed_duration_s": fft_data["observed_duration_s"],
        "channel_1_identity": "membrane geometry target",
        "channel_2_identity": "mCherry-MinC",
        "channel_3_identity": "MreB-msfGFP",
    }


def run_figs10_figure(figure_dir: Path | str) -> FigS10Result:
    figure_dir = Path(figure_dir).resolve()
    raw_dir = figure_dir / "00_FigS10_raw"
    output_dir = figure_dir / "02_FigS10_plot_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    length_period_source = pd.read_csv(raw_dir / "FigS10i_length_period_source.csv")
    prepared_datasets = []
    final_tables: list[pd.DataFrame] = []
    four_sample_summaries: list[dict] = []
    svg_paths: list[Path] = []

    for config in FIGS10_DATASETS:
        data = _prepare_dataset(figure_dir, config)
        center_of_mass = calculate_center_of_mass_trace(data["normalized_matrix"])
        fft_data = calculate_fft(data["times_s"], center_of_mass)
        if not np.isclose(fft_data["dt_seconds"], config.frame_interval_s, atol=1e-9):
            raise ValueError(f"{config.sample_id} frame interval validation failed")

        audited_row = length_period_source.loc[
            length_period_source["sample_id"] == config.sample_id
        ]
        if len(audited_row) != 1:
            raise ValueError(f"Expected one length-period source row for {config.sample_id}")
        audited_period = float(audited_row.iloc[0]["oscillation_period_s"])
        if not np.isclose(fft_data["peak_period_s"], audited_period, atol=1e-9):
            raise ValueError(
                f"{config.sample_id} FFT period {fft_data['peak_period_s']:.12g} s differs "
                f"from audited value {audited_period:.12g} s"
            )

        kymograph_svg = output_dir / f"{config.kymograph_panel}.svg"
        fft_svg = output_dir / f"{config.fft_panel}.svg"
        _plot_kymograph_panel(data, kymograph_svg)
        _plot_fft_panel(fft_data, fft_svg)
        svg_paths.extend([kymograph_svg, fft_svg])

        final_tables.extend(
            [
                _build_kymograph_table(data),
                _build_trace_table(data, center_of_mass, fft_data),
                _build_fft_table(data, fft_data),
            ]
        )
        four_sample_summaries.append(_build_four_sample_summary(data, fft_data))
        prepared_datasets.append(data)

    scatter_svg = output_dir / "FigS10i.svg"
    _, regression_summary = _plot_length_period_panel(length_period_source, scatter_svg)
    svg_paths.append(scatter_svg)

    sample_summary = length_period_source.copy()
    sample_summary.insert(0, "table_type", "sample_summary")
    four_by_sample = {row["sample_id"]: row for row in four_sample_summaries}
    for row_index, row in sample_summary.iterrows():
        detailed = four_by_sample.get(row["sample_id"])
        if detailed is not None:
            for key, value in detailed.items():
                sample_summary.loc[row_index, key] = value
        else:
            sample_summary.loc[row_index, "source_note"] = (
                "clean n=10 source row retained; raw time series not migrated in this FigS10 package"
            )

    regression_table = regression_summary.copy()
    regression_table.insert(0, "table_type", "regression_summary")
    final_tables.extend([sample_summary, regression_table])
    final_csv = pd.concat(final_tables, ignore_index=True, sort=False)
    leading_columns = [
        "table_type",
        "sample_id",
        "source_group",
        "kymograph_panel",
        "fft_panel",
        "raw_data_in_repository",
    ]
    ordered_columns = leading_columns + [
        column for column in final_csv.columns if column not in leading_columns
    ]
    final_csv = final_csv[ordered_columns]
    csv_path = output_dir / "FigS10.csv"
    final_csv.to_csv(csv_path, index=False, float_format="%.12g")

    for svg_path in svg_paths:
        svg_text = svg_path.read_text(encoding="utf-8")
        contains_numeric_nan = re.search(
            r"(?<![A-Za-z0-9])nan(?![A-Za-z0-9])",
            svg_text,
            flags=re.IGNORECASE,
        )
        if "<svg" not in svg_text or contains_numeric_nan:
            raise ValueError(f"Invalid SVG output: {svg_path}")

    return FigS10Result(
        output_dir=output_dir,
        csv_path=csv_path,
        svg_paths=tuple(svg_paths),
        sample_summary=sample_summary,
        regression_summary=regression_summary,
    )
