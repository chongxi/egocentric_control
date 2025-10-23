import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a PLY point cloud and camera poses to preview them in Open3D."
    )
    parser.add_argument(
        "--ply",
        type=Path,
        default=Path("points.ply"),
        help="Path to the colored PLY point cloud.",
    )
    parser.add_argument(
        "--poses",
        type=Path,
        default=Path("output/camera_poses.npz"),
        help="Path to the camera pose archive produced by run_localization.py.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Image width associated with the camera intrinsics.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height associated with the camera intrinsics.",
    )
    parser.add_argument(
        "--fx",
        type=float,
        default=576.0,
        help="Camera focal length in pixels along the x axis.",
    )
    parser.add_argument(
        "--fy",
        type=float,
        default=576.0,
        help="Camera focal length in pixels along the y axis.",
    )
    parser.add_argument(
        "--cx",
        type=float,
        default=320.0,
        help="Principal point x coordinate in pixels.",
    )
    parser.add_argument(
        "--cy",
        type=float,
        default=240.0,
        help="Principal point y coordinate in pixels.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.25,
        help="Length of the visualized camera frusta.",
    )
    return parser.parse_args()


def _camera_frustum(
    cam_to_world: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    scale: float,
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
    colors = [[1.0, 0.1, 0.1] for _ in lines]
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(corners_world.astype(np.float64))
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector(colors)
    return frustum


def load_camera_poses(path: Path) -> np.ndarray:
    archive = np.load(path)
    if "cam_to_world" not in archive:
        raise KeyError(f"{path} does not contain 'cam_to_world'")
    return archive["cam_to_world"]


def main() -> None:
    args = parse_args()
    if not args.poses.exists():
        raise FileNotFoundError(f"Camera pose archive not found: {args.poses}")
    if not args.ply.exists():
        raise FileNotFoundError(f"Point cloud not found: {args.ply}")

    print(f"[Preview] Loading point cloud from {args.ply}")
    point_cloud = o3d.io.read_point_cloud(str(args.ply))
    print(f"[Preview] Loaded point cloud with {np.asarray(point_cloud.points).shape[0]} points.")

    print(f"[Preview] Loading camera poses from {args.poses}")
    cam_to_world = load_camera_poses(args.poses)
    print(f"[Preview] Loaded {cam_to_world.shape[0]} poses.")

    geometries: List[o3d.geometry.Geometry] = [
        point_cloud,
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5),
    ]
    for pose in cam_to_world:
        frustum = _camera_frustum(
            cam_to_world=pose,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            width=args.width,
            height=args.height,
            scale=args.scale,
        )
        geometries.append(frustum)

    print("[Preview] Launching Open3D visualizer...")
    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    main()
