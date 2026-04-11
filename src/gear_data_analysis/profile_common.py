from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.interpolate import make_interp_spline

from .straightener_common import to_grayscale_frame


def load_single_image(image_source):
    return to_grayscale_frame(image_source)


def calculate_two_channel_projection(roi, axis=0):
    projection_raw = np.sum(roi, axis=axis).astype(np.float32)

    projection_normalized = projection_raw.copy()
    max_val = projection_raw.max()
    if max_val > 0:
        projection_normalized = projection_normalized / max_val

    return projection_raw, projection_normalized


def calculate_projection_raw(roi, axis=0):
    return np.sum(roi, axis=axis).astype(np.float32)


def normalize_projection(projection_raw):
    projection_normalized = projection_raw.copy()
    max_val = projection_normalized.max() if projection_normalized.size else 0.0
    if max_val > 0:
        projection_normalized = projection_normalized / max_val
    return projection_normalized


def smooth_data(x, y, smoothing_factor=300, require_min_points=False):
    if require_min_points and len(x) < 4:
        return x, y

    if smoothing_factor <= len(x):
        smoothing_factor = len(x) * 2

    x_smooth = np.linspace(x.min(), x.max(), smoothing_factor)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_smooth)

    return x_smooth, y_smooth


def analyze_two_channel_images(
    path_membrane,
    path_ftsz,
    apply_smoothing=True,
    smoothing_factor=300,
    save_path=None,
    membrane_label="Membrane",
    secondary_label=r"$\mathrm{FtsZ^{MTS}}$",
    membrane_color="#B200F4",
    secondary_color="#00C000",
):
    image_membrane = load_single_image(path_membrane)
    image_ftsz = load_single_image(path_ftsz)

    print(f"Membrane image shape: {image_membrane.shape}")
    print(f"FtsZ image shape: {image_ftsz.shape}")

    projection_membrane_raw, projection_membrane = calculate_two_channel_projection(
        image_membrane, axis=0
    )
    projection_ftsz_raw, projection_ftsz = calculate_two_channel_projection(
        image_ftsz, axis=0
    )

    x = np.arange(len(projection_membrane))

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if apply_smoothing:
        x_smooth_membrane, y_smooth_membrane = smooth_data(
            x, projection_membrane, smoothing_factor
        )
        x_smooth_ftsz, y_smooth_ftsz = smooth_data(x, projection_ftsz, smoothing_factor)

        ax.plot(
            x_smooth_membrane,
            y_smooth_membrane,
            label=membrane_label,
            color=membrane_color,
            linewidth=2,
        )
        ax.plot(
            x_smooth_ftsz,
            y_smooth_ftsz,
            label=secondary_label,
            color=secondary_color,
            linewidth=2,
        )
    else:
        ax.plot(x, projection_membrane, label=membrane_label, color=membrane_color, linewidth=2)
        ax.plot(x, projection_ftsz, label=secondary_label, color=secondary_color, linewidth=2)

    ax.set_xlim(0, len(x) - 1)
    ax.set_ylim(0, None)

    first_point = 0
    middle_point = len(x) // 2
    last_point = len(x) - 1

    ax.set_xticks([first_point, middle_point, last_point])
    ax.set_xticklabels(["Pole 1", "Equator", "Pole 2"])

    ax.set_ylabel("Normalized Intensity", fontsize=14)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.tick_params(axis="x", labelsize=14, width=2, length=8)
    ax.tick_params(axis="y", labelsize=14, width=2, length=8)

    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()

    return projection_membrane_raw, projection_membrane, projection_ftsz_raw, projection_ftsz


def save_two_channel_projections_to_csv(
    projection_membrane_raw,
    projection_ftsz_raw,
    projection_membrane,
    projection_ftsz,
    save_path,
):
    data_rows = []

    for idx in range(len(projection_membrane_raw)):
        row = [
            f"{projection_membrane_raw[idx]:.5f}",
            f"{projection_ftsz_raw[idx]:.5f}",
            "",
            f"{projection_membrane[idx]:.5f}",
            f"{projection_ftsz[idx]:.5f}",
        ]
        data_rows.append(row)

    with open(save_path, "w", encoding="utf-8") as handle:
        handle.write("Membrane_raw,FtsZ_raw,,Membrane_normalized,FtsZ_normalized\n")
        for row in data_rows:
            handle.write(",".join(row) + "\n")

    print(f"CSV saved to: {save_path}")


def analyze_three_channel_images(
    channel_configs,
    apply_smoothing=True,
    smoothing_factor=300,
    save_path=None,
    fig_width=6,
    fig_height=4,
    plot_title=None,
    y_upper_margin=100,
):
    if len(channel_configs) != 3:
        raise ValueError(
            "channel_configs must contain exactly three entries for Pre-bleach, Post-bleach, and Post-recovery."
        )

    data_length_um = 8.0
    x_offset = 1.0
    axis_pad = 2.0
    tick_step = 1.0
    y_tick_step = 200

    bleached_start = 3.0
    bleached_end = 10.0
    bleached_alpha = 0.2
    bleached_color = "#84888f"

    axis_end = x_offset + data_length_um + axis_pad

    channel_data = []
    max_y = 0.0

    for config in channel_configs:
        label = config.get("label")
        image_source = config.get("path")
        if not label or image_source is None:
            raise ValueError("Each channel config must include 'label' and 'path'.")

        image = load_single_image(image_source)
        print(f"{label} image shape: {image.shape}")

        projection_raw = calculate_projection_raw(image, axis=0)
        projection_normalized = normalize_projection(projection_raw)
        if projection_raw.size:
            max_y = max(max_y, projection_raw.max())

        if len(projection_raw) >= 2:
            x_vals = np.linspace(0, data_length_um, len(projection_raw)) + x_offset
        else:
            x_vals = np.array([], dtype=np.float32)

        if apply_smoothing and len(x_vals) >= 4:
            x_plot, y_plot = smooth_data(
                x_vals,
                projection_raw,
                smoothing_factor,
                require_min_points=True,
            )
        else:
            x_plot, y_plot = x_vals, projection_raw

        channel_data.append(
            {
                "label": label,
                "raw": projection_raw,
                "normalized": projection_normalized,
                "plot_x": x_plot,
                "plot_y": y_plot,
                "color": config.get("color", "#B200F4"),
                "linestyle": config.get("linestyle", "-"),
            }
        )

    if max_y <= 0:
        max_y = 1.0

    y_max_plot = np.ceil(max_y / y_tick_step) * y_tick_step
    y_ticks = np.arange(0, y_max_plot + y_tick_step, y_tick_step)
    y_limit = y_max_plot + y_upper_margin

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for channel in channel_data:
        x_plot = channel["plot_x"]
        y_plot = channel["plot_y"]

        if len(x_plot) == 0:
            continue

        ax.plot(
            x_plot,
            y_plot,
            label=channel["label"],
            color=channel["color"],
            linewidth=2,
            linestyle=channel["linestyle"],
        )

    b_start_disp = bleached_start + x_offset
    b_end_disp = bleached_end + x_offset
    ax.add_patch(
        Rectangle(
            (b_start_disp, 0),
            b_end_disp - b_start_disp,
            y_limit,
            facecolor=bleached_color,
            alpha=bleached_alpha,
            edgecolor="none",
        )
    )
    ax.text(
        b_start_disp + (b_end_disp - b_start_disp) / 2.0,
        y_limit * 0.98,
        "bleached region",
        ha="center",
        va="top",
        fontsize=16,
        color="#000000",
        alpha=1,
    )

    ax.set_xlim(0, axis_end)
    x_ticks = np.arange(0, axis_end + tick_step, tick_step)
    x_tick_labels = []
    for tick in x_ticks:
        label_value = tick - x_offset
        if label_value >= -1e-6 and np.isclose(label_value % 5, 0.0, atol=1e-6):
            x_tick_labels.append(str(int(round(label_value))))
        else:
            x_tick_labels.append("")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)

    ax.set_ylim(0, y_limit)
    ax.set_yticks(y_ticks)

    ax.set_xlabel("Length (μm)", fontsize=16)
    ax.set_ylabel("Intensity (a.u.)", fontsize=16)
    title_text = plot_title if plot_title is not None else "Intensity profile"
    ax.set_title(title_text, loc="left", fontsize=18, pad=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=10)

    for tick, label in zip(ax.xaxis.get_major_ticks(), ax.get_xticklabels()):
        if label.get_text():
            tick.tick1line.set_markersize(8)
            tick.tick2line.set_markersize(8)
        else:
            tick.tick1line.set_markersize(4)
            tick.tick2line.set_markersize(4)

    ax.tick_params(axis="x", labelsize=14, width=2, length=4)
    ax.tick_params(axis="y", labelsize=14, width=2, length=8)

    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()

    return channel_data


def save_three_channel_projections_to_csv(channel_results, save_path):
    if not channel_results:
        raise ValueError("channel_results cannot be empty.")

    max_points = max(len(channel["raw"]) for channel in channel_results)

    raw_headers = [f"{channel['label'].replace(' ', '_')}_raw" for channel in channel_results]
    norm_headers = [
        f"{channel['label'].replace(' ', '_')}_normalized" for channel in channel_results
    ]

    with open(save_path, "w", encoding="utf-8") as handle:
        handle.write(",".join(raw_headers + [""] + norm_headers) + "\n")
        for idx in range(max_points):
            row_raw = []
            row_norm = []
            for channel in channel_results:
                if idx < len(channel["raw"]):
                    row_raw.append(f"{channel['raw'][idx]:.5f}")
                    row_norm.append(f"{channel['normalized'][idx]:.5f}")
                else:
                    row_raw.append("")
                    row_norm.append("")

            handle.write(",".join(row_raw + [""] + row_norm) + "\n")

    print(f"CSV saved to: {save_path}")
