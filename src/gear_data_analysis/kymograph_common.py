from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .straightener_common import to_grayscale_frames


def create_transparent_cmap(base_cmap_name):
    base_cmap = plt.get_cmap(base_cmap_name)
    colors = base_cmap(np.linspace(0, 1, 256))
    colors[:, -1] = np.linspace(0, 1, 256)
    return LinearSegmentedColormap.from_list(f"{base_cmap_name}_Transparent", colors)


GREENS_TRANSPARENT = create_transparent_cmap("Greens")
REDS_TRANSPARENT = create_transparent_cmap("Reds")
PURPLES_TRANSPARENT = create_transparent_cmap("Purples")
CUSTOM_COLORMAPS = {
    "Greens_Transparent": GREENS_TRANSPARENT,
    "Reds_Transparent": REDS_TRANSPARENT,
    "Purples_Transparent": PURPLES_TRANSPARENT,
}


def calculate_projections(image, axis=0):
    return np.sum(image, axis=axis)


def normalize_data_0_to_max(data):
    max_val = np.max(data)
    if max_val > 0:
        return data / max_val
    return data


def align_projections_to_center(projections_list):
    max_width = max(len(proj) for proj in projections_list)

    aligned_projections = []
    for proj in projections_list:
        current_width = len(proj)
        if current_width == max_width:
            aligned_projections.append(proj)
        else:
            padding_total = max_width - current_width
            padding_left = padding_total // 2
            padding_right = padding_total - padding_left
            aligned_proj = np.pad(
                proj,
                (padding_left, padding_right),
                mode="constant",
                constant_values=0,
            )
            aligned_projections.append(aligned_proj)

    return np.array(aligned_projections)


def build_projection_data(stack_sources):
    frames = {channel: to_grayscale_frames(source) for channel, source in stack_sources.items()}
    first_channel = next(iter(stack_sources))
    num_frames = len(frames[first_channel])

    projection_series = {channel: [] for channel in stack_sources}

    for idx in range(num_frames):
        for channel in stack_sources:
            projection_series[channel].append(calculate_projections(frames[channel][idx], axis=0))

    aligned_data = {
        channel: align_projections_to_center(series)
        for channel, series in projection_series.items()
    }
    projection_data_final = {
        channel: np.transpose(data) for channel, data in aligned_data.items()
    }
    projection_data_norm = {
        channel: normalize_data_0_to_max(data)
        for channel, data in projection_data_final.items()
    }
    return projection_data_final, projection_data_norm


def save_heatmap(
    projection_data,
    save_path,
    colormap,
    time_interval,
    fig_aspect_ratio=4 / 1,
    time_xlabel="Time (s)",
    y_labels=("Pole 1", "Equator", "Pole 2"),
):
    if isinstance(colormap, str) and colormap in CUSTOM_COLORMAPS:
        colormap = CUSTOM_COLORMAPS[colormap]

    height, num_frames = projection_data.shape
    total_time = (num_frames - 1) * time_interval

    fig_width = 20
    fig_height = fig_width / fig_aspect_ratio
    plt.figure(figsize=(fig_width, fig_height))

    plt.imshow(
        projection_data,
        aspect="auto",
        cmap=colormap,
        origin="lower",
        vmin=0,
        vmax=1,
        extent=[0, total_time, 0, height],
        interpolation="none",
    )
    plt.colorbar(label="Normalized Intensity")
    plt.xlabel(time_xlabel)
    y_ticks = [0, height // 2, height - 1]
    plt.yticks(y_ticks, y_labels)
    plt.title("")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format="svg", transparent=True, dpi=300)
    plt.close()


def save_data_to_csv(
    raw_data_dict,
    normalized_data_dict,
    save_path,
    time_interval,
    channel_order,
    pole_labels=("Pole1", "Pole2"),
):
    height, num_frames = list(raw_data_dict.values())[0].shape
    time_points = [i * time_interval for i in range(num_frames)]
    csv_data = []
    header = ["Position", "Channel", "Data_Type"] + [f"Time_{time_value}s" for time_value in time_points]
    csv_data.append(header)

    pole_start, pole_end = pole_labels
    for pos in range(height):
        if pos == 0:
            pos_label = pole_start
        elif pos == height - 1:
            pos_label = pole_end
        elif pos == height // 2:
            pos_label = "Equator"
        else:
            pos_label = f"Pos_{pos}"

        for channel in channel_order:
            raw_row = [pos_label, channel, "Raw"] + list(raw_data_dict[channel][pos, :])
            csv_data.append(raw_row)
            norm_row = [pos_label, channel, "Normalized"] + list(normalized_data_dict[channel][pos, :])
            csv_data.append(norm_row)

    df = pd.DataFrame(csv_data[1:], columns=csv_data[0])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)


def run_kymograph_analysis(
    stack_sources,
    output_dir,
    time_interval_seconds,
    channel_outputs,
    csv_name,
    channel_order=None,
    time_xlabel="Time (s)",
    heatmap_y_labels=("Pole 1", "Equator", "Pole 2"),
    csv_pole_labels=("Pole1", "Pole2"),
):
    raw_data_dict, normalized_data_dict = build_projection_data(stack_sources)

    if channel_order is None:
        channel_order = list(stack_sources)

    saved_svg_paths = []
    for channel_output in channel_outputs:
        save_path = Path(output_dir) / channel_output["svg_name"]
        save_heatmap(
            normalized_data_dict[channel_output["channel"]],
            save_path,
            channel_output["colormap"],
            time_interval_seconds,
            time_xlabel=time_xlabel,
            y_labels=heatmap_y_labels,
        )
        saved_svg_paths.append(save_path)

    csv_save_path = Path(output_dir) / csv_name
    save_data_to_csv(
        raw_data_dict,
        normalized_data_dict,
        csv_save_path,
        time_interval_seconds,
        channel_order=channel_order,
        pole_labels=csv_pole_labels,
    )

    return {
        "raw_data": raw_data_dict,
        "normalized_data": normalized_data_dict,
        "svg_paths": saved_svg_paths,
        "csv_path": csv_save_path,
    }
