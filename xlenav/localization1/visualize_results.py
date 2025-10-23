import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import open3d as o3d


def _load_records(records_path: Path) -> List[dict]:
    with records_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _camera_frustum(
    cam_to_world: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    scale: float = 0.25,
) -> o3d.geometry.LineSet:
    z = scale
    corners_cam = np.array(
        [
            [0.0, 0.0, 0.0],
            [(-cx) * z / fx, (-cy) * z / fy, z],
            [(width - cx) * z / fx, (-cy) * z / fy, z],
            [(width - cx) * z / fx, (height - cy) * z / fy, z],
            [(-cx) * z / fx, (height - cy) * z / fy, z],
        ],
        dtype=np.float32,
    )
    corners_world = (cam_to_world[:3, :3] @ corners_cam.T).T + cam_to_world[:3, 3]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
    ]
    colors = [[1.0, 0.0, 0.0] for _ in lines]
    frustum = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(corners_world))
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector(colors)
    return frustum


def visualize(
    point_cloud_path: Path,
    records_path: Path,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    include_failed: bool = False,
) -> None:
    point_cloud = o3d.io.read_point_cloud(str(point_cloud_path))
    camera_geometries: List[o3d.geometry.Geometry] = []
    records = _load_records(records_path)
    for record in records:
        if not record.get("success") and not include_failed:
            continue
        if "cam_to_world" not in record or record["cam_to_world"] is None:
            continue
        cam_to_world = np.asarray(record["cam_to_world"], dtype=np.float32)
        frustum = _camera_frustum(
            cam_to_world=cam_to_world,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
        )
        camera_geometries.append(frustum)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0.0, 0.0, 0.0]
    )
    geometries: Iterable[o3d.geometry.Geometry] = [point_cloud, axis, *camera_geometries]
    o3d.visualization.draw_geometries(list(geometries))
