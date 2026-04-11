from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from scipy.sparse.csgraph import dijkstra
from skan import Skeleton as SkanSkeleton
from skimage.morphology import skeletonize as skel_func


@dataclass(frozen=True)
class StraightenerConfig:
    key: str
    target_dir: Path
    mapping_dirs: tuple[Path, ...]
    start_preference: str = "top"
    pad_to_canvas: bool = False
    output_width: int = 200
    output_height: int = 200


@dataclass(frozen=True)
class ProcessedImageSet:
    target_img_path: Path
    target_straight_rgb: np.ndarray
    mapping_img_paths: tuple[Path, ...]
    mapping_straight_rgb: tuple[np.ndarray, ...]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
REPO_ROOT = Path(__file__).resolve().parent
MATCH_SUFFIX_PATTERNS = (
    re.compile(r"(?:_|-)?c\d+$", re.IGNORECASE),
    re.compile(r"(?:_|-)[A-Za-z]$"),
)


def _resolve_config(config_or_key: StraightenerConfig | str) -> StraightenerConfig:
    if isinstance(config_or_key, StraightenerConfig):
        return config_or_key
    return STRAIGHTENER_CONFIGS[config_or_key]


def _convert_bgr_to_rgb(image: np.ndarray | None) -> np.ndarray:
    if image is None:
        raise ValueError("Expected an image array, got None.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_match_key(stem: str) -> str:
    normalized = stem
    changed = True
    while changed:
        changed = False
        for pattern in MATCH_SUFFIX_PATTERNS:
            updated = pattern.sub("", normalized)
            if updated != normalized:
                normalized = updated
                changed = True
    return normalized.lower()


def list_image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_match_index(image_paths: list[Path], directory: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for image_path in image_paths:
        match_key = normalize_match_key(image_path.stem)
        if match_key in index:
            raise ValueError(
                f"Ambiguous basename match in {directory}: "
                f"'{index[match_key].name}' and '{image_path.name}' both map to '{match_key}'."
            )
        index[match_key] = image_path
    return index


def select_skeleton_start_point(
    endpoint_coordinates: np.ndarray, preference: str = "top"
) -> int:
    if len(endpoint_coordinates) < 2:
        return 0

    if preference == "top":
        return int(np.argmin(endpoint_coordinates[:, 0]))
    if preference == "bottom":
        return int(np.argmax(endpoint_coordinates[:, 0]))
    if preference == "left":
        return int(np.argmin(endpoint_coordinates[:, 1]))
    if preference == "right":
        return int(np.argmax(endpoint_coordinates[:, 1]))
    return 0


def straighten_polygon_region(
    image: np.ndarray, polygon_mask: np.ndarray, skeleton_path_rc: np.ndarray
) -> np.ndarray | None:
    skeleton_path_rc = np.array(skeleton_path_rc, dtype=np.float32)
    if len(skeleton_path_rc) < 2:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    path_segments = np.diff(skeleton_path_rc, axis=0)
    segment_lengths = np.linalg.norm(path_segments, axis=1)
    arc_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
    total_skeleton_length = arc_lengths[-1]

    if total_skeleton_length < 1e-3:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    path_normals_rc = np.zeros_like(skeleton_path_rc)

    tangent_first = skeleton_path_rc[1] - skeleton_path_rc[0]
    norm_tangent_first = np.linalg.norm(tangent_first)
    if norm_tangent_first < 1e-6:
        tangent_first = np.array([1.0, 0.0])
    else:
        tangent_first /= norm_tangent_first
    path_normals_rc[0] = [-tangent_first[1], tangent_first[0]]

    for i in range(1, len(skeleton_path_rc) - 1):
        tangent = skeleton_path_rc[i + 1] - skeleton_path_rc[i - 1]
        norm_tangent = np.linalg.norm(tangent)
        if norm_tangent < 1e-6:
            path_normals_rc[i] = path_normals_rc[i - 1]
        else:
            tangent /= norm_tangent
            path_normals_rc[i] = [-tangent[1], tangent[0]]

    tangent_last = skeleton_path_rc[-1] - skeleton_path_rc[-2]
    norm_tangent_last = np.linalg.norm(tangent_last)
    if norm_tangent_last < 1e-6:
        tangent_last = np.array([1.0, 0.0])
    else:
        tangent_last /= norm_tangent_last
    path_normals_rc[-1] = [-tangent_last[1], tangent_last[0]]

    all_mask_points_rc = np.argwhere(polygon_mask > 0).astype(np.float32)
    if len(all_mask_points_rc) == 0:
        return None

    all_v_coords = []
    for p_rc in all_mask_points_rc:
        distances_to_skel_pts_sq = np.sum((skeleton_path_rc - p_rc) ** 2, axis=1)
        closest_skel_pt_idx = int(np.argmin(distances_to_skel_pts_sq))
        vec_skel_to_p = p_rc - skeleton_path_rc[closest_skel_pt_idx]
        perp_dist = np.dot(vec_skel_to_p, path_normals_rc[closest_skel_pt_idx])
        all_v_coords.append(float(perp_dist))

    if not all_v_coords:
        d_min, d_max = -20, 20
    else:
        d_min, d_max = float(np.min(all_v_coords)), float(np.max(all_v_coords))

    output_width_px = int(round(total_skeleton_length))
    output_height_px = int(round(d_max - d_min))
    if output_width_px <= 0:
        output_width_px = 1
    if output_height_px <= 0:
        output_height_px = 1

    v_offset_display = -d_min
    map_c_coords = np.zeros((output_height_px, output_width_px), dtype=np.float32)
    map_r_coords = np.zeros((output_height_px, output_width_px), dtype=np.float32)

    for r_new in range(output_height_px):
        for c_new in range(output_width_px):
            u_param = total_skeleton_length - float(c_new)
            v_param = float(r_new) - v_offset_display

            skel_pt_idx = np.searchsorted(arc_lengths, u_param, side="right") - 1
            skel_pt_idx = max(0, min(skel_pt_idx, len(arc_lengths) - 2))

            segment_start_arc_len = arc_lengths[skel_pt_idx]
            segment_len = arc_lengths[skel_pt_idx + 1] - segment_start_arc_len

            if segment_len < 1e-6:
                interp_ratio = 0.0
            else:
                interp_ratio = np.clip(
                    (u_param - segment_start_arc_len) / segment_len, 0.0, 1.0
                )

            pt_on_skel_rc = (
                (1 - interp_ratio) * skeleton_path_rc[skel_pt_idx]
                + interp_ratio * skeleton_path_rc[skel_pt_idx + 1]
            )
            normal_rc_interp = (
                (1 - interp_ratio) * path_normals_rc[skel_pt_idx]
                + interp_ratio * path_normals_rc[skel_pt_idx + 1]
            )

            norm_of_normal = np.linalg.norm(normal_rc_interp)
            if norm_of_normal < 1e-6:
                if np.linalg.norm(path_normals_rc[skel_pt_idx]) > 1e-6:
                    normal_rc_interp = path_normals_rc[skel_pt_idx]
                else:
                    normal_rc_interp = np.array([0.0, 1.0])
            else:
                normal_rc_interp /= norm_of_normal

            map_r_coords[r_new, c_new] = (
                pt_on_skel_rc[0] + v_param * normal_rc_interp[0]
            )
            map_c_coords[r_new, c_new] = (
                pt_on_skel_rc[1] + v_param * normal_rc_interp[1]
            )

    masked_source_image = cv2.bitwise_and(image, image, mask=polygon_mask)
    straightened_img_content = cv2.remap(
        masked_source_image,
        map_c_coords,
        map_r_coords,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    mask_float = polygon_mask.astype(np.float32) / 255.0
    straightened_mask_float = cv2.remap(
        mask_float,
        map_c_coords,
        map_r_coords,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    final_straightened_mask = (straightened_mask_float > 0.5).astype(np.uint8) * 255
    final_output = cv2.bitwise_and(
        straightened_img_content,
        straightened_img_content,
        mask=final_straightened_mask,
    )
    return cv2.flip(final_output, -1)


def pad_and_center_image(
    img_to_pad: np.ndarray | None, out_width: int, out_height: int
) -> np.ndarray:
    if img_to_pad is None:
        return np.zeros((out_height, out_width, 3), dtype=np.uint8)

    h, w = img_to_pad.shape[:2]
    canvas = np.zeros((out_height, out_width, 3), dtype=np.uint8)

    x_offset = (out_width - w) // 2
    y_offset = (out_height - h) // 2
    paste_w, paste_h = min(w, out_width), min(h, out_height)

    if x_offset < 0 or y_offset < 0:
        crop_x = (w - out_width) // 2 if w > out_width else 0
        crop_y = (h - out_height) // 2 if h > out_height else 0
        canvas[:, :] = img_to_pad[crop_y : crop_y + paste_h, crop_x : crop_x + paste_w]
    else:
        canvas[y_offset : y_offset + paste_h, x_offset : x_offset + paste_w] = (
            img_to_pad[:paste_h, :paste_w]
        )

    return canvas


def extract_skeleton_and_longest_path(
    mask: np.ndarray, start_preference: str = "top"
) -> np.ndarray:
    skeleton_rc = skel_func(mask > 0)
    graph = SkanSkeleton(skeleton_rc)
    pixel_graph, coordinates_rc = graph.graph, graph.coordinates

    if pixel_graph.shape[0] == 0:
        raise ValueError("Skeleton graph is empty.")

    n_nodes = pixel_graph.shape[0]
    degrees = np.bincount(pixel_graph.indices, minlength=n_nodes)
    endpoint_nodes_indices = np.where(degrees == 1)[0]

    longest_path_coords_rc = None

    if len(endpoint_nodes_indices) < 2:
        if graph.n_paths > 0:
            longest_path_idx = int(np.argmax(graph.path_lengths()))
            longest_path_coords_rc = graph.path_coordinates(longest_path_idx)
            if longest_path_coords_rc is None or len(longest_path_coords_rc) < 2:
                raise ValueError("Fallback skeleton path is invalid.")
        else:
            raise ValueError("No paths found and insufficient endpoints.")
    else:
        temp_start_node = int(endpoint_nodes_indices[0])
        distances_from_temp = dijkstra(
            csgraph=pixel_graph,
            directed=False,
            indices=temp_start_node,
            return_predecessors=False,
        )

        farthest_node_A_idx = -1
        max_dist = -1.0
        for ep_idx in endpoint_nodes_indices:
            distance = distances_from_temp[ep_idx]
            if distance > max_dist and np.isfinite(distance):
                max_dist = float(distance)
                farthest_node_A_idx = int(ep_idx)

        if farthest_node_A_idx == -1:
            raise ValueError("Could not find a reachable endpoint A.")

        distances_from_A, predecessors_from_A = dijkstra(
            csgraph=pixel_graph,
            directed=False,
            indices=farthest_node_A_idx,
            return_predecessors=True,
        )

        farthest_node_B_idx = -1
        max_dist_A_to_B = -1.0
        for ep_idx in endpoint_nodes_indices:
            distance = distances_from_A[ep_idx]
            if distance > max_dist_A_to_B and np.isfinite(distance):
                max_dist_A_to_B = float(distance)
                farthest_node_B_idx = int(ep_idx)

        if farthest_node_B_idx == -1:
            raise ValueError("Could not find a reachable endpoint B.")

        path_indices = []
        curr = farthest_node_B_idx
        while curr != -9999:
            path_indices.append(curr)
            if curr == farthest_node_A_idx:
                break
            curr = predecessors_from_A[curr]

        if not path_indices or path_indices[-1] != farthest_node_A_idx:
            raise ValueError("Path reconstruction failed.")

        path_indices.reverse()
        endpoint_A_coord = coordinates_rc[farthest_node_A_idx]
        endpoint_B_coord = coordinates_rc[farthest_node_B_idx]
        two_endpoints_coords = np.array([endpoint_A_coord, endpoint_B_coord])
        preferred_start_index = select_skeleton_start_point(
            two_endpoints_coords, start_preference
        )

        if preferred_start_index == 0:
            longest_path_coords_rc = coordinates_rc[path_indices]
        else:
            path_indices.reverse()
            longest_path_coords_rc = coordinates_rc[path_indices]

    if longest_path_coords_rc is None or len(longest_path_coords_rc) < 2:
        raise ValueError("Skeleton path is invalid after final check.")

    return longest_path_coords_rc


def validate_and_load_files(
    target_dir: Path, mapping_dirs: tuple[Path, ...]
) -> list[dict[str, object]]:
    target_files = []
    target_image_paths = list_image_paths(target_dir)
    for img_path in target_image_paths:
        filename = img_path.name
        name = img_path.stem
        json_path = target_dir / f"{name}.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Target image '{filename}' is missing its corresponding JSON file."
            )
        target_files.append(
            {
                "img": img_path,
                "json": json_path,
                "name": filename,
                "match_key": normalize_match_key(name),
            }
        )

    if not target_files:
        raise ValueError("No valid image/JSON pairs found in the target folder.")

    target_keys = {file_info["match_key"] for file_info in target_files}
    if len(target_keys) != len(target_files):
        raise ValueError(f"Target folder {target_dir} contains duplicate normalized basenames.")

    for m_dir in mapping_dirs:
        mapping_paths = list_image_paths(m_dir)
        mapping_index = build_match_index(mapping_paths, m_dir)
        mapping_keys = set(mapping_index)

        if len(mapping_index) != len(target_files):
            raise ValueError(
                f"File count mismatch for {m_dir}: target has {len(target_files)}, mapping has {len(mapping_index)}."
            )

        missing = sorted(target_keys - mapping_keys)
        extra = sorted(mapping_keys - target_keys)
        if missing or extra:
            message_parts = [f"Basename mismatch for {m_dir}."]
            if missing:
                message_parts.append(f"Missing keys: {missing}")
            if extra:
                message_parts.append(f"Extra keys: {extra}")
            raise ValueError(" ".join(message_parts))

        for target_file_info in target_files:
            target_img = cv2.imread(os.fspath(target_file_info["img"]))
            if target_img is None:
                raise OSError(f"Cannot read target image: {target_file_info['img']}")

            map_img_path = mapping_index[target_file_info["match_key"]]
            map_img = cv2.imread(os.fspath(map_img_path))
            if map_img is None:
                raise OSError(f"Cannot read mapping image: {map_img_path}")

            h, w = target_img.shape[:2]
            mh, mw = map_img.shape[:2]
            if (h, w) != (mh, mw):
                raise ValueError(
                    f"Size mismatch: '{target_file_info['name']}' ({w}x{h}) vs '{map_file}' ({mw}x{mh})"
                )

    file_list_map = []
    mapping_indexes = {
        m_dir: build_match_index(list_image_paths(m_dir), m_dir) for m_dir in mapping_dirs
    }

    for target_file_info in target_files:
        mapping_img_paths = []
        for m_dir in mapping_dirs:
            mapping_img_paths.append(mapping_indexes[m_dir][target_file_info["match_key"]])

        file_list_map.append(
            {
                "target_img_path": target_file_info["img"],
                "target_json_path": target_file_info["json"],
                "mapping_img_paths": tuple(mapping_img_paths),
            }
        )

    return file_list_map


def process_single_image_set(
    file_set: dict[str, object], config: StraightenerConfig
) -> ProcessedImageSet:
    target_img_path = Path(file_set["target_img_path"])
    target_json_path = Path(file_set["target_json_path"])
    mapping_img_paths = tuple(Path(p) for p in file_set["mapping_img_paths"])

    target_img_cv = cv2.imread(os.fspath(target_img_path))
    if target_img_cv is None:
        raise OSError(f"Cannot read target image: {target_img_path}")
    img_height, img_width = target_img_cv.shape[:2]

    with target_json_path.open("r", encoding="utf-8") as f:
        labelme_data = json.load(f)

    polygon_points_xy = labelme_data["shapes"][0]["points"]
    poly_np = np.array(polygon_points_xy, dtype=np.int32)
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_np], 255)

    longest_path_coords_rc = extract_skeleton_and_longest_path(
        mask, config.start_preference
    )

    straight_target_raw = straighten_polygon_region(target_img_cv, mask, longest_path_coords_rc)
    straight_mappings_raw = []
    for map_img_path in mapping_img_paths:
        map_img_cv = cv2.imread(os.fspath(map_img_path))
        if map_img_cv is None:
            raise OSError(f"Cannot read mapping image: {map_img_path}")
        straight_map = straighten_polygon_region(map_img_cv, mask, longest_path_coords_rc)
        straight_mappings_raw.append(straight_map)

    if config.pad_to_canvas:
        final_target = pad_and_center_image(
            straight_target_raw, config.output_width, config.output_height
        )
        final_mappings = tuple(
            pad_and_center_image(img, config.output_width, config.output_height)
            for img in straight_mappings_raw
        )
    else:
        final_target = straight_target_raw
        final_mappings = tuple(straight_mappings_raw)

    return ProcessedImageSet(
        target_img_path=target_img_path,
        target_straight_rgb=_convert_bgr_to_rgb(final_target),
        mapping_img_paths=mapping_img_paths,
        mapping_straight_rgb=tuple(_convert_bgr_to_rgb(img) for img in final_mappings),
    )


def process_all_image_sets(
    config_or_key: StraightenerConfig | str,
) -> tuple[ProcessedImageSet, ...]:
    config = _resolve_config(config_or_key)
    file_list_map = validate_and_load_files(config.target_dir, config.mapping_dirs)
    return tuple(process_single_image_set(file_set, config) for file_set in file_list_map)


def build_named_straightened_images(
    config_or_key: StraightenerConfig | str,
) -> dict[str, np.ndarray]:
    image_map: dict[str, np.ndarray] = {}
    for result_set in process_all_image_sets(config_or_key):
        target_path = result_set.target_img_path
        target_name = f"{target_path.stem}_{target_path.parent.name}_straight.tif"
        image_map[target_name] = result_set.target_straight_rgb

        for map_path, map_rgb in zip(
            result_set.mapping_img_paths, result_set.mapping_straight_rgb
        ):
            map_name = f"{map_path.stem}_{map_path.parent.name}_straight.tif"
            image_map[map_name] = map_rgb

    return image_map


def build_straightened_stack_map(
    config_or_key: StraightenerConfig | str,
) -> dict[str, tuple[np.ndarray, ...]]:
    config = _resolve_config(config_or_key)
    results = process_all_image_sets(config)

    stack_map: dict[str, list[np.ndarray]] = {config.target_dir.name: []}
    stack_map.update({mapping_dir.name: [] for mapping_dir in config.mapping_dirs})

    for result_set in results:
        stack_map[config.target_dir.name].append(result_set.target_straight_rgb)
        for mapping_dir, map_rgb in zip(config.mapping_dirs, result_set.mapping_straight_rgb):
            stack_map[mapping_dir.name].append(map_rgb)

    return {key: tuple(value) for key, value in stack_map.items()}


def to_grayscale_frame(image_source: str | os.PathLike[str] | Path | np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image_source, (str, os.PathLike, Path)):
        with Image.open(image_source) as img:
            return np.array(img.convert("L"))

    if isinstance(image_source, Image.Image):
        return np.array(image_source.convert("L"))

    if isinstance(image_source, np.ndarray):
        return np.array(Image.fromarray(image_source).convert("L"))

    raise TypeError(f"Unsupported image source type: {type(image_source)!r}")


def to_grayscale_frames(
    frame_source: str | os.PathLike[str] | Path | Iterable[np.ndarray],
) -> list[np.ndarray]:
    if isinstance(frame_source, (str, os.PathLike, Path)):
        img = Image.open(frame_source)
        frames = []
        try:
            while True:
                frames.append(np.array(img.convert("L")))
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        return frames

    return [to_grayscale_frame(frame) for frame in frame_source]


STRAIGHTENER_CONFIGS = {
    "Fig1h": StraightenerConfig(
        key="Fig1h",
        target_dir=REPO_ROOT / "Fig1h/00_Fig1h_raw/c2_membrane",
        mapping_dirs=(REPO_ROOT / "Fig1h/00_Fig1h_raw/c1_mBaojin",),
        start_preference="left",
    ),
    "Fig3d": StraightenerConfig(
        key="Fig3d",
        target_dir=REPO_ROOT / "Fig3d/00_Fig3c_raw/c2_membrane",
        mapping_dirs=(REPO_ROOT / "Fig3d/00_Fig3c_raw/c1_FtsZ",),
        start_preference="left",
    ),
    "Fig4c": StraightenerConfig(
        key="Fig4c",
        target_dir=REPO_ROOT / "Fig4c/00_Fig4b_raw/c1_membrane",
        mapping_dirs=(
            REPO_ROOT / "Fig4c/00_Fig4b_raw/c2_MinC",
            REPO_ROOT / "Fig4c/00_Fig4b_raw/c3",
        ),
        start_preference="bottom",
    ),
    "Fig4e": StraightenerConfig(
        key="Fig4e",
        target_dir=REPO_ROOT / "Fig4e/00_Fig4d_raw/c1_membrane",
        mapping_dirs=(
            REPO_ROOT / "Fig4e/00_Fig4d_raw/c2_MinC",
            REPO_ROOT / "Fig4e/00_Fig4d_raw/c3_FtsZ",
        ),
        start_preference="right",
    ),
    "Fig4f_plus": StraightenerConfig(
        key="Fig4f_plus",
        target_dir=REPO_ROOT / "Fig4f/00_Fig4f_raw_data/+MinCDE/c1_membrane",
        mapping_dirs=(REPO_ROOT / "Fig4f/00_Fig4f_raw_data/+MinCDE/c2_FtsZ",),
        start_preference="right",
    ),
    "Fig4f_minus": StraightenerConfig(
        key="Fig4f_minus",
        target_dir=REPO_ROOT / "Fig4f/00_Fig4f_raw_data/-MinCDE/c1_membrane",
        mapping_dirs=(REPO_ROOT / "Fig4f/00_Fig4f_raw_data/-MinCDE/c2_FtsZ",),
        start_preference="right",
    ),
    "Fig4g": StraightenerConfig(
        key="Fig4g",
        target_dir=REPO_ROOT / "Fig4g/00_Fig4g_raw/c1_membrane",
        mapping_dirs=(REPO_ROOT / "Fig4g/00_Fig4g_raw/c3_FtsZ",),
        start_preference="left",
    ),
    "FigS3f": StraightenerConfig(
        key="FigS3f",
        target_dir=REPO_ROOT / "FigS3f/00_FigS3f_raw/c2_membrane",
        mapping_dirs=(REPO_ROOT / "FigS3f/00_FigS3f_raw/c1_mBaojin",),
        start_preference="left",
    ),
    "FigS6d": StraightenerConfig(
        key="FigS6d",
        target_dir=REPO_ROOT / "FigS6d/00_FigS6c_raw/c1_membrane",
        mapping_dirs=(REPO_ROOT / "FigS6d/00_FigS6c_raw/c2_FtsZ",),
        start_preference="top",
    ),
    "FigS6f": StraightenerConfig(
        key="FigS6f",
        target_dir=REPO_ROOT / "FigS6f/00_FigS6e_raw/c1_membrane",
        mapping_dirs=(REPO_ROOT / "FigS6f/00_FigS6e_raw/c3_FtsZ",),
        start_preference="top",
    ),
    "FigS7e": StraightenerConfig(
        key="FigS7e",
        target_dir=REPO_ROOT / "FigS7e/00_FigS7d_raw/c1_membrane",
        mapping_dirs=(
            REPO_ROOT / "FigS7e/00_FigS7d_raw/c2_MinC",
            REPO_ROOT / "FigS7e/00_FigS7d_raw/c3",
        ),
        start_preference="right",
    ),
    "FigS9a": StraightenerConfig(
        key="FigS9a",
        target_dir=REPO_ROOT / "FigS9a/00_FigS9a_raw/c1_membrane",
        mapping_dirs=(REPO_ROOT / "FigS9a/00_FigS9a_raw/c2_FtsZ",),
        start_preference="right",
    ),
    "FigS9b": StraightenerConfig(
        key="FigS9b",
        target_dir=REPO_ROOT / "FigS9b/00_FigS9b_raw/c1_membrane",
        mapping_dirs=(REPO_ROOT / "FigS9b/00_FigS9b_raw/c3_FtsZ",),
        start_preference="right",
    ),
    "FigS8C": StraightenerConfig(
        key="FigS8C",
        target_dir=REPO_ROOT / "FigS8C/00_FigS8C_raw/c1_membrane",
        mapping_dirs=(REPO_ROOT / "FigS8C/00_FigS8C_raw/c1_FtsZ",),
        start_preference="right",
    ),
    "FigS8d_minus": StraightenerConfig(
        key="FigS8d_minus",
        target_dir=REPO_ROOT / "FigS8d/00_FigS8d_raw/-MinCDE/c1_membrane",
        mapping_dirs=(REPO_ROOT / "FigS8d/00_FigS8d_raw/-MinCDE/c2_FtsZ",),
        start_preference="right",
    ),
    "FigS8d_plus": StraightenerConfig(
        key="FigS8d_plus",
        target_dir=REPO_ROOT / "FigS8d/00_FigS8d_raw/+MinCDE/c1_membrane",
        mapping_dirs=(REPO_ROOT / "FigS8d/00_FigS8d_raw/+MinCDE/c2_FtsZ",),
        start_preference="right",
    ),
}
