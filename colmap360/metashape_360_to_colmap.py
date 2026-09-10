#!/usr/bin/env python3
"""Convert Metashape spherical-camera results to a COLMAP text dataset.

The implementation follows the camera-pose convention used by the original
Metashape_360_to_COLMAP_plane project: Metashape camera transforms are treated
as camera-to-world (C2W), each rectilinear view gets its canonical direction
rotation, and COLMAP receives the inverse world-to-camera (W2C) transform.

Masks are intentionally not generated here; SAM3 masks are projected by the
separate mask stage using the same direction convention.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image

try:
    import open3d as o3d
except ImportError:
    o3d = None


# Canonical cubemap directions used by the reference implementation.
# Keep these values identical for image projection, mask projection and COLMAP
# camera extrinsics.
DIRECTIONS = (
    ("front", 0.0, 0.0),
    ("right", -90.0, 0.0),
    ("back", 180.0, 0.0),
    ("left", 90.0, 0.0),
    ("top", 0.0, 90.0),
    ("bottom", 0.0, -90.0),
)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ply", type=Path, default=None)
    parser.add_argument("--crop-size", type=int, default=1920)
    parser.add_argument("--fov-deg", type=float, default=90.0)
    parser.add_argument("--max-images", type=int, default=10000)
    parser.add_argument("--yaw-offset", type=float, default=0.0)
    parser.add_argument("--rotate-z180", action="store_true")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Reserved for future parallel extraction; use 1 for the first test.",
    )
    return parser.parse_args()


def parse_floats(text: str | None) -> np.ndarray:
    if not text:
        return np.empty(0, dtype=np.float64)
    return np.asarray([float(x) for x in text.split()], dtype=np.float64)


def parse_metashape_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    chunk = root.find("chunk")
    if chunk is None:
        chunk = root[0]

    sensors = chunk.find("sensors")
    cameras = chunk.find("cameras")
    components = chunk.find("components")
    if sensors is None or cameras is None:
        raise ValueError("Metashape XML must contain sensors and cameras.")

    sensor_data: Dict[str, dict] = {}
    for sensor in sensors.findall("sensor"):
        if sensor.get("type") != "spherical":
            continue
        resolution = sensor.find("resolution")
        if resolution is None:
            continue
        sensor_data[sensor.get("id", "")] = {
            "width": int(resolution.get("width")),
            "height": int(resolution.get("height")),
        }

    if not sensor_data:
        raise ValueError("No spherical sensor found in Metashape XML.")

    component_data: Dict[str, np.ndarray] = {}
    if components is not None:
        for component in components.findall("component"):
            transform_node = component.find("transform")
            if transform_node is None:
                continue

            rotation = parse_floats(
                getattr(transform_node.find("rotation"), "text", None)
            )
            translation = parse_floats(
                getattr(transform_node.find("translation"), "text", None)
            )
            scale_node = transform_node.find("scale")
            scale = (
                float(scale_node.text)
                if scale_node is not None and scale_node.text
                else 1.0
            )

            transform = np.eye(4, dtype=np.float64)
            if rotation.size == 9:
                transform[:3, :3] = rotation.reshape(3, 3)
            if translation.size == 3:
                transform[:3, 3] = translation / scale
            component_data[component.get("id", "")] = transform

    camera_data: list[dict] = []
    for camera in cameras.findall("camera"):
        if camera.get("enabled", "true").lower() == "false":
            continue

        camera_id = camera.get("id")
        label = camera.get("label", camera_id or "camera")
        sensor_id = camera.get("sensor_id")
        transform_node = camera.find("transform")
        if (
            camera_id is None
            or sensor_id not in sensor_data
            or transform_node is None
            or transform_node.text is None
        ):
            continue

        values = parse_floats(transform_node.text)
        if values.size != 16:
            continue

        camera_data.append(
            {
                "id": camera_id,
                "label": label,
                "sensor_id": sensor_id,
                "component_id": camera.get("component_id"),
                "transform": values.reshape(4, 4),
            }
        )

    return sensor_data, component_data, camera_data


def direction_rotation_matrix(direction: str) -> np.ndarray:
    for name, yaw_deg, pitch_deg in DIRECTIONS:
        if name == direction:
            yaw = math.radians(yaw_deg)
            pitch = math.radians(pitch_deg)
            ry = np.array(
                [
                    [math.cos(yaw), 0.0, math.sin(yaw)],
                    [0.0, 1.0, 0.0],
                    [-math.sin(yaw), 0.0, math.cos(yaw)],
                ],
                dtype=np.float64,
            )
            rx = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, math.cos(pitch), -math.sin(pitch)],
                    [0.0, math.sin(pitch), math.cos(pitch)],
                ],
                dtype=np.float64,
            )
            return ry @ rx
    raise KeyError(f"Unknown direction: {direction}")


def yaw_rotation_matrix(yaw_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    return np.array(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ],
        dtype=np.float64,
    )


def perspective_map(
    width: int,
    height: int,
    size: int,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
):
    """Build an equirectangular -> perspective remap."""
    f = (size / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    x = (xx - size / 2.0) / f
    y = (size / 2.0 - yy) / f
    z = np.ones_like(x)

    dirs = np.stack((x, y, z), axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    rotation = yaw_rotation_matrix(yaw_deg) @ yaw_rotation_matrix(0.0)
    pitch = math.radians(pitch_deg)
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rotation = rotation @ rx
    dirs = dirs @ rotation.T

    longitude = np.arctan2(dirs[..., 0], dirs[..., 2])
    latitude = np.arcsin(np.clip(dirs[..., 1], -1.0, 1.0))

    map_x = ((longitude / (2.0 * math.pi)) + 0.5) * width
    map_y = (0.5 - latitude / math.pi) * height
    map_x = np.mod(map_x, width).astype(np.float32)
    map_y = np.clip(map_y, 0, height - 1).astype(np.float32)
    return map_x, map_y, f


def project_equirectangular(
    image: np.ndarray,
    size: int,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
) -> np.ndarray:
    h, w = image.shape[:2]
    map_x, map_y, _ = perspective_map(w, h, size, fov_deg, yaw_deg, pitch_deg)
    return cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def quaternion_from_rotation(r: np.ndarray) -> tuple[float, float, float, float]:
    """Return COLMAP quaternion components as (w, x, y, z)."""
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return w, x, y, z


def write_points3d(ply_path: Path | None, output_path: Path):
    if ply_path is None:
        output_path.write_text("", encoding="utf-8")
        return
    if o3d is None:
        raise RuntimeError("--ply requires Open3D. Install it with: pip install open3d")

    cloud = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors) if cloud.has_colors() else None

    with output_path.open("w", encoding="utf-8") as f:
        for index, point in enumerate(points, start=1):
            if colors is not None and index - 1 < len(colors):
                rgb = np.clip(colors[index - 1] * 255.0, 0, 255).astype(int)
                f.write(
                    f"{index} {point[0]} {point[1]} {point[2]} "
                    f"{rgb[0]} {rgb[1]} {rgb[2]}\n"
                )
            else:
                f.write(
                    f"{index} {point[0]} {point[1]} {point[2]} 255 255 255\n"
                )


def find_image_for_label(images_dir: Path, label: str) -> Path | None:
    candidates = [images_dir / label]
    for extension in SUPPORTED_EXTENSIONS:
        candidates.append(images_dir / f"{label}{extension}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    stem = Path(label).stem
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            if path.stem == stem:
                return path
    return None


def main() -> None:
    args = parse_args()
    if not args.images.is_dir():
        raise NotADirectoryError(args.images)
    if not args.xml.is_file():
        raise FileNotFoundError(args.xml)
    if args.crop_size <= 0 or not 0 < args.fov_deg < 180:
        raise ValueError("crop-size must be > 0 and fov-deg must be between 0 and 180.")

    sensor_data, component_data, cameras = parse_metashape_xml(args.xml)
    del sensor_data

    labels = cameras[: args.max_images]
    if not labels:
        raise RuntimeError("No usable spherical cameras found in Metashape XML.")

    args.output.mkdir(parents=True, exist_ok=True)
    output_images = args.output / "images"
    output_images.mkdir(parents=True, exist_ok=True)

    fx = (args.crop_size / 2.0) / math.tan(math.radians(args.fov_deg) / 2.0)
    camera_lines = [
        "# Camera list with one line of data per camera:",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        "# Number of cameras: 1",
        f"1 PINHOLE {args.crop_size} {args.crop_size} {fx:.12f} {fx:.12f} "
        f"{args.crop_size / 2.0:.12f} {args.crop_size / 2.0:.12f}",
    ]
    image_lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]

    image_id = 1
    source_count = 0

    for camera in labels:
        source = find_image_for_label(args.images, camera["label"])
        if source is None:
            print(f"WARNING: no image found for Metashape camera '{camera['label']}', skipping")
            continue

        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            print(f"WARNING: could not read {source}, skipping")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = camera["transform"].copy()
        component_id = camera.get("component_id")
        if component_id in component_data:
            transform = component_data[component_id] @ transform

        R_c2w = transform[:3, :3]
        t_c2w = transform[:3, 3]
        frame_yaw_offset = args.yaw_offset

        if source_count == 0:
            print(f"First camera: {camera['label']}")
            print(f"  component_id: {component_id}")
            print(f"  R_c2w:\n{R_c2w}")
            print(f"  C (camera center): {t_c2w}")

        for direction, yaw_deg, pitch_deg in DIRECTIONS:
            yaw = yaw_deg + frame_yaw_offset
            crop = project_equirectangular(
                image,
                args.crop_size,
                args.fov_deg,
                yaw,
                pitch_deg,
            )
            filename = f"{Path(camera['label']).stem}_{direction}.jpg"
            Image.fromarray(crop).save(output_images / filename, quality=95)

            R_dir = direction_rotation_matrix(direction)
            if frame_yaw_offset != 0.0:
                R_dir = yaw_rotation_matrix(frame_yaw_offset) @ R_dir

            # Metashape transform is C2W. COLMAP expects W2C.
            R_c2w_dir = R_c2w @ R_dir
            R_w2c = R_c2w_dir.T
            t_w2c = -R_w2c @ t_c2w

            # Keep this behavior identical to the reference project for users
            # who need the optional PostShot/scene-axis correction.
            if args.rotate_z180:
                R_z180 = np.array(
                    [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
                R_w2c = R_w2c @ R_z180

            q_w, q_x, q_y, q_z = quaternion_from_rotation(R_w2c)
            image_lines.append(
                f"{image_id} {q_w:.12f} {q_x:.12f} {q_y:.12f} {q_z:.12f} "
                f"{t_w2c[0]:.12f} {t_w2c[1]:.12f} {t_w2c[2]:.12f} 1 {filename}"
            )
            image_lines.append("")
            image_id += 1

        source_count += 1
        print(f"Processed: {source.name}")

    (args.output / "cameras.txt").write_text(
        "\n".join(camera_lines) + "\n", encoding="utf-8"
    )
    image_lines.insert(3, f"# Number of images: {image_id - 1}")
    (args.output / "images.txt").write_text(
        "\n".join(image_lines) + "\n", encoding="utf-8"
    )
    write_points3d(args.ply, args.output / "points3D.txt")

    print("\nDone.")
    print(f"Input panoramas: {source_count}")
    print(f"Perspective images: {image_id - 1}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
