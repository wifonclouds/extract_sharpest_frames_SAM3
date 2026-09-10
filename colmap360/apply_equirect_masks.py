#!/usr/bin/env python3
"""Project original equirectangular SAM3 masks onto the six COLMAP views."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

DIRECTIONS = (
    ("front", 0.0, 0.0),
    ("right", 90.0, 0.0),
    ("back", 180.0, 0.0),
    ("left", 270.0, 0.0),
    ("top", 0.0, 90.0),
    ("bottom", 0.0, -90.0),
)
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def perspective_map(width: int, height: int, size: int, fov_deg: float, yaw_deg: float, pitch_deg: float):
    f = (size / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    x = (xx - size / 2.0) / f
    y = (yy - size / 2.0) / f
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
    dirs = dirs @ (ry @ rx).T
    longitude = np.arctan2(dirs[..., 0], dirs[..., 2])
    latitude = np.arcsin(np.clip(dirs[..., 1], -1.0, 1.0))
    map_x = ((longitude / (2.0 * math.pi)) + 0.5) * width
    map_y = (0.5 - latitude / math.pi) * height
    return (np.mod(map_x, width).astype(np.float32),
            np.clip(map_y, 0, height - 1).astype(np.float32))


def find_mask(mask_dir: Path, stem: str) -> Path | None:
    direct = [mask_dir / f"{stem}{ext}" for ext in MASK_EXTENSIONS]
    for path in direct:
        if path.is_file():
            return path
    for path in mask_dir.rglob("*"):
        if path.is_file() and path.stem == stem and path.suffix.lower() in MASK_EXTENSIONS:
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=1920)
    parser.add_argument("--fov-deg", type=float, default=90.0)
    parser.add_argument("--yaw-offset", type=float, default=0.0)
    args = parser.parse_args()

    if not args.masks.is_dir():
        raise NotADirectoryError(args.masks)
    out_masks = args.output / "masks"
    out_masks.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "images"
    generated = sorted(image_dir.glob("*.*"))
    if not generated:
        raise RuntimeError(f"No generated images found in {image_dir}")

    cache: dict[str, np.ndarray] = {}
    missing = 0
    written = 0
    for view in generated:
        parts = view.stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        stem, direction = parts
        direction_info = next((d for d in DIRECTIONS if d[0] == direction), None)
        if direction_info is None:
            continue
        mask_path = find_mask(args.masks, stem)
        if mask_path is None:
            missing += 1
            continue
        if stem not in cache:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                missing += 1
                continue
            cache[stem] = mask
        mask = cache[stem]
        _, yaw, pitch = direction_info
        yaw += args.yaw_offset
        map_x, map_y = perspective_map(mask.shape[1], mask.shape[0], args.crop_size, args.fov_deg, yaw, pitch)
        projected = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
        # Preserve the SAM3 convention: black = person, white = background.
        cv2.imwrite(str(out_masks / f"{view.stem}.png"), projected)
        written += 1

    print(f"SAM3 masks projected: {written}")
    if missing:
        print(f"WARNING: missing/unreadable source masks for {missing} generated views")


if __name__ == "__main__":
    main()
