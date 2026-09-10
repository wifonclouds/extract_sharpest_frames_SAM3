#!/usr/bin/env python3
"""Run the Metashape->COLMAP converter and optionally project SAM3 masks."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--masks", type=Path, default=None)
    p.add_argument("--xml", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ply", type=Path, default=None)
    p.add_argument("--crop-size", type=int, default=1920)
    p.add_argument("--fov-deg", type=float, default=90.0)
    p.add_argument("--max-images", type=int, default=10000)
    p.add_argument("--yaw-offset", type=float, default=0.0)
    p.add_argument("--rotate-z180", action="store_true")
    p.add_argument("--num-workers", type=int, default=1)
    a = p.parse_args()

    converter = Path(__file__).with_name("metashape_360_to_colmap.py")
    cmd = [sys.executable, str(converter), "--images", str(a.images), "--xml", str(a.xml), "--output", str(a.output),
           "--crop-size", str(a.crop_size), "--fov-deg", str(a.fov_deg), "--max-images", str(a.max_images),
           "--yaw-offset", str(a.yaw_offset), "--num-workers", str(a.num_workers)]
    if a.ply: cmd += ["--ply", str(a.ply)]
    if a.rotate_z180: cmd.append("--rotate-z180")

    print("[1/2] Running Metashape 360 -> COLMAP conversion...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    if a.masks:
        mask_script = Path(__file__).with_name("apply_equirect_masks.py")
        mask_cmd = [sys.executable, str(mask_script), "--images", str(a.images), "--masks", str(a.masks),
                    "--output", str(a.output), "--crop-size", str(a.crop_size), "--fov-deg", str(a.fov_deg),
                    "--yaw-offset", str(a.yaw_offset)]
        print("[2/2] Projecting SAM3 masks onto generated perspective views...")
        result = subprocess.run(mask_cmd)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    else:
        print("[2/2] No SAM3 mask folder supplied; skipping mask projection.")


if __name__ == "__main__":
    main()
