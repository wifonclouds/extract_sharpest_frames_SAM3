#!/usr/bin/env python3
"""Convert Metashape spherical-camera results to a COLMAP text dataset.

This is a clean implementation for the SAM3 workflow. It intentionally does
not run YOLO or generate masks. Masks produced separately by SAM3 can be added
later without changing the image/camera conversion stage.

Input:
  - equirectangular images
  - Metashape Camera.xml exported from a spherical chunk
  - optional PLY point cloud

Output:
  - six 90-degree rectilinear views per panorama
  - COLMAP cameras.txt, images.txt and optionally points3D.txt

Example:
  python colmap360/metashape_360_to_colmap.py \
      --images sharp_frames \
      --xml Camera.xml \
      --output colmap_dataset

Dependencies:
  pip install numpy pillow opencv-python
  optional: pip install open3d
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


DIRECTIONS = (
    ("front", 0.0, 0.0),
    ("right", 90.0, 0.0),
    ("back", 180.0, 0.0),
    ("left", 270.0, 0.0),
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
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Reserved for future parallel extraction; use 1 for the first test.")
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

    component_transform = np.eye(4, dtype=np.float64)
    if components is not None:
        component = components.find("component")
        if component is not None:
            transform = component.find("transform")
            if transform is not None:
                rotation = parse_floats(getattr(transform.find("rotation"), "text", None))
                translation = parse_floats(getattr(transform.find("translation"), "text", None))
                scale_node = transform.find("scale")
                scale = float(scale_node.text) if scale_node is not None and scale_node.text else 1.0
                if rotation.size == 9:
                    component_transform[:3, :3] = rotation.reshape(3, 3)
                if translation.size == 3:
                    component_transform[:3, 3] = translation / scale

    camera_data: Dict[str, dict] = {}
    for camera in cameras.findall("camera"):
        if camera.get("enabled", "true").lower() == "false":
            continue
        camera_id = camera.get("id")
        label = camera.get("label", camera_id or "camera")
        sensor_id = camera.get("sensor_id")
        transform_node = camera.find("transform")
        if camera_id is None or sensor_id not in sensor_data or transform_node is None:
            continue
        transform = parse_floats(transform_node.text)
        if transform.size != 16:
            continue
        camera_data[label] = {
            "id": camera_id,
            "sensor_id": sensor_id,
            "transform": transform.reshape(4, 4),
        }

    return sensor_data, component_transform, camera_data


def perspective_map(width: int, height: int, size: int, fov_deg: float,
                    yaw_deg: float, pitch_deg: float):
    """Build an equirectangular -> perspective remap.

    Image coordinates use y-down. The ray-space convention uses y-up, so the
    image-space y coordinate is inverted here. The same mapping is used for
    SAM3 masks to keep masks pixel-aligned with generated views.
    """
    f = (size / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    x = (xx - size / 2.0) / f
    y = (size / 2.0 - yy) / f
    z = np.ones_like(x)

    dirs = np.stack((x, y, z), axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    ry = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                   [0, 1, 0],
                   [-math.sin(yaw), 0, math.cos(yaw)]])
    rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]])
    rotation = ry @ rx
    dirs = dirs @ rotation.T

    longitude = np.arctan2(dirs[..., 0], dirs[..., 2])
    latitude = np.arcsin(np.clip(dirs[..., 1], -1.0, 1.0))

    map_x = ((longitude / (2.0 * math.pi)) + 0.5) * width
    map_y = (0.5 - latitude / math.pi) * height
    map_x = np.mod(map_x, width).astype(np.float32)
    map_y = np.clip(map_y, 0, height - 1).astype(np.float32)
    return map_x, map_y, f


def project_equirectangular(image: np.ndarray, size: int, fov_deg: float,
                            yaw_deg: float, pitch_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    map_x, map_y, _ = perspective_map(w, h, size, fov_deg, yaw_deg, pitch_deg)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)


def quaternion_from_rotation(r: np.ndarray) -> tuple[float, float, float, float]:
    trace = np.trace(r)
    if trace > 0:
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
                f.write(f"{index} {point[0]} {point[1]} {point[2]} {rgb[0]} {rgb[1]} {rgb[2]}\n")
            else:
                f.write(f"{index} {point[0]} {point[1]} {point[2]} 255 255 255\n")


def find_image_for_label(images_dir: Path, label: str) -> Path | None:
    candidates = [images_dir / label]
    for extension in SUPPORTED_EXTENSIONS:
        candidates.append(images_dir / f"{label}{extension}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    stem_matches = [p for p in images_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                    and p.stem == Path(label).stem]
    return stem_matches[0] if stem_matches else None


def main() -> None:
    args = parse_args()
    if not args.images.is_dir():
        raise NotADirectoryError(args.images)
    if not args.xml.is_file():
        raise FileNotFoundError(args.xml)
    if args.crop_size <= 0 or args.fov_deg <= 0 or args.fov_deg >= 180:
        raise ValueError("crop-size must be > 0 and fov-deg must be between 0 and 180.")

    sensor_data, component_transform, cameras = parse_metashape_xml(args.xml)
    labels = list(cameras)[:args.max_images]
    if not labels:
        raise RuntimeError("No usable spherical cameras found in Metashape XML.")

    output_images = args.output / "images"
    output_images.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    fx = (args.crop_size / 2.0) / math.tan(math.radians(args.fov_deg) / 2.0)
    camera_lines = ["# Camera list with one line of data per camera:",
                    "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    image_lines = ["# Image list with two lines of data per image:",
                   "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]

    image_id = 1
    camera_id = 1
    camera_model_ids: Dict[int, int] = {}

    for label in labels:
        info = cameras[label]
        source = find_image_for_label(args.images, label)
        if source is None:
            print(f"WARNING: no image found for Metashape camera '{label}', skipping")
            continue

        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            print(f"WARNING: could not read {source}, skipping")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        model_key = (args.crop_size, args.crop_size)
        if model_key not in camera_model_ids:
            camera_model_ids[model_key] = camera_id
            camera_lines.append(
                f"{camera_id} PINHOLE {args.crop_size} {args.crop_size} {fx:.12f} {fx:.12f} "
                f"{args.crop_size / 2.0:.12f} {args.crop_size / 2.0:.12f}"
            )
            camera_id += 1
        current_camera_id = camera_model_ids[model_key]

        base_transform = component_transform @ info["transform"]
        if args.rotate_z180:
            rz = np.diag([-1.0, -1.0, 1.0, 1.0])
            base_transform = rz @ base_transform

        for direction, yaw, pitch in DIRECTIONS:
            yaw += args.yaw_offset
            crop = project_equirectangular(image, args.crop_size, args.fov_deg, yaw, pitch)
            filename = f"{Path(label).stem}_{direction}.jpg"
            out_path = output_images / filename
            Image.fromarray(crop).save(out_path, quality=95)

            view_rotation = np.eye(3)
            if direction == "right":
                a = math.radians(90 + args.yaw_offset)
                view_rotation = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
            elif direction == "back":
                a = math.radians(180 + args.yaw_offset)
                view_rotation = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
            elif direction == "left":
                a = math.radians(270 + args.yaw_offset)
                view_rotation = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
            elif direction == "top":
                a = math.radians(90)
                view_rotation = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
            elif direction == "bottom":
                a = math.radians(-90)
                view_rotation = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])

            world_rotation = base_transform[:3, :3] @ view_rotation
            q = quaternion_from_rotation(world_rotation)
            t = base_transform[:3, 3]
            image_lines.append(
                f"{image_id} {q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} "
                f"{t[0]:.12f} {t[1]:.12f} {t[2]:.12f} {current_camera_id} {filename}"
            )
            image_lines.append("")
            image_id += 1

        print(f"Processed: {source.name}")

    (args.output / "cameras.txt").write_text("\n".join(camera_lines) + "\n", encoding="utf-8")
    (args.output / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    write_points3d(args.ply, args.output / "points3D.txt")

    print("\nDone.")
    print(f"Input panoramas: {len(labels)}")
    print(f"Perspective images: {image_id - 1}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
