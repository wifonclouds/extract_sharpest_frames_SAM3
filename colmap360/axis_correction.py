#!/usr/bin/env python3
"""Apply the global axis convention expected by LichtFeld."""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np


def quaternion_from_rotation(r: np.ndarray) -> tuple[float, float, float, float]:
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


def apply_lichtfeld_axis(output_dir: Path) -> None:
    """Rotate points and camera orientations 180 degrees around world X."""
    s = np.diag([1.0, -1.0, -1.0])

    points_path = output_dir / "points3D.txt"
    if points_path.is_file():
        result = []
        for line in points_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 7 and not fields[0].startswith("#"):
                xyz = s @ np.asarray([float(fields[1]), float(fields[2]), float(fields[3])])
                fields[1:4] = [f"{v:.12f}" for v in xyz]
                line = " ".join(fields)
            result.append(line)
        points_path.write_text("\n".join(result) + "\n", encoding="utf-8")

    images_path = output_dir / "images.txt"
    if images_path.is_file():
        result = []
        for line in images_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            # Pose lines have >=10 fields; the following POINTS2D line has triples.
            if len(fields) >= 10 and not fields[0].startswith("#"):
                try:
                    qw, qx, qy, qz = map(float, fields[1:5])
                    r = np.array([
                        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
                    ], dtype=np.float64)
                    qw, qx, qy, qz = quaternion_from_rotation(r @ s)
                    fields[1:5] = [f"{v:.12f}" for v in (qw, qx, qy, qz)]
                    line = " ".join(fields)
                except ValueError:
                    pass
            result.append(line)
        images_path.write_text("\n".join(result) + "\n", encoding="utf-8")
