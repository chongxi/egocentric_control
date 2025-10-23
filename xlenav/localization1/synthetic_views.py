import math
import os
import pickle
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
import open3d as o3d


@dataclass
class SyntheticView:
    name: str
    color_image: np.ndarray
    depth_image: np.ndarray
    extrinsic: np.ndarray
    keypoints: np.ndarray
    descriptors: np.ndarray
    points3d: np.ndarray


@dataclass
class SyntheticViewDatabase:
    ply_path: str
    cache_path: str
    image_width: int = 640
    image_height: int = 480
    fx: float = None
    fy: float = None
    cx: float = None
    cy: float = None
    num_views: int = 80
    radius_scale: float = 1.3
    min_depth: float = 0.1
    max_depth: float = 10.0
    point_size: float = 3.0
    random_seed: int = 13
    views: List[SyntheticView] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.fx is None or self.fy is None:
            focal = 0.9 * max(self.image_width, self.image_height)
            self.fx = focal
            self.fy = focal
        if self.cx is None or self.cy is None:
            self.cx = self.image_width / 2.0
            self.cy = self.image_height / 2.0
        self._rng = np.random.default_rng(self.random_seed)

    def load_or_build(self, force_rebuild: bool = False) -> None:
        if os.path.exists(self.cache_path) and not force_rebuild:
            self._load()
            return
        self._build()
        self._save()

    def _build(self) -> None:
        point_cloud = o3d.io.read_point_cloud(self.ply_path)
        if not point_cloud.has_colors():
            raise ValueError("Point cloud must contain per-point colors.")
        center = point_cloud.get_center()
        radii = np.linalg.norm(np.asarray(point_cloud.points) - center, axis=1)
        radius = np.max(radii) * self.radius_scale

        renderer = o3d.visualization.rendering.OffscreenRenderer(
            self.image_width, self.image_height
        )
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = self.point_size
        renderer.scene.set_background([0.0, 0.0, 0.0, 0.0])
        renderer.scene.add_geometry("point_cloud", point_cloud, mat)

        camera = renderer.scene.camera
        vertical_fov = math.degrees(
            2.0 * math.atan(self.image_height / (2.0 * self.fy))
        )
        aspect = self.image_width / self.image_height
        camera.set_projection(
            vertical_fov,
            aspect,
            self.min_depth,
            self.max_depth,
            o3d.visualization.rendering.Camera.FovType.Vertical,
        )

        directions = list(self._sample_view_directions(self.num_views))
        orb = cv2.ORB_create(nfeatures=1200)

        self.views = []
        for idx, direction in enumerate(directions):
            position = center + radius * direction
            camera.look_at(center, position, [0.0, 0.0, 1.0])
            extrinsic = np.asarray(camera.get_view_matrix(), dtype=np.float32)
            color_o3d = renderer.render_to_image()
            depth_o3d = renderer.render_to_depth_image(True)
            color_rgba = np.asarray(color_o3d)
            color_rgb = color_rgba[..., :3]
            color = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            depth = np.asarray(depth_o3d)
            keypoints, descriptors, points3d = self._extract_features(
                color, depth, extrinsic, orb
            )
            view = SyntheticView(
                name=f"synthetic_{idx:03d}",
                color_image=color,
                depth_image=depth,
                extrinsic=extrinsic,
                keypoints=keypoints,
                descriptors=descriptors,
                points3d=points3d,
            )
            self.views.append(view)

    def _sample_view_directions(self, count: int):
        offset = 2.0 / count
        increment = math.pi * (3.0 - math.sqrt(5.0))
        for i in range(count):
            y = ((i * offset) - 1.0) + (offset / 2.0)
            r = math.sqrt(max(0.0, 1.0 - y * y))
            phi = ((i % count) * increment) % (math.tau)
            x = math.cos(phi) * r
            z = math.sin(phi) * r
            yield np.array([x, y, z], dtype=np.float32)

    def _extract_features(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        extrinsic: np.ndarray,
        orb: cv2.ORB,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        kps, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is None or len(kps) == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 32), dtype=np.uint8),
                np.zeros((0, 3), dtype=np.float32),
            )
        points_2d = []
        points_3d = []
        descriptor_list = []
        for kp, descriptor in zip(kps, descriptors):
            u, v = kp.pt
            u_i = int(round(u))
            v_i = int(round(v))
            if (
                u_i < 0
                or v_i < 0
                or u_i >= self.image_width
                or v_i >= self.image_height
            ):
                continue
            depth_value = depth[v_i, u_i]
            if depth_value <= 0.0 or not np.isfinite(depth_value):
                continue
            point_cam = self._pixel_to_camera(u, v, depth_value)
            point_world = self._camera_to_world(point_cam, extrinsic)
            points_2d.append([u, v])
            points_3d.append(point_world)
            descriptor_list.append(descriptor)
        if not points_2d:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 32), dtype=np.uint8),
                np.zeros((0, 3), dtype=np.float32),
            )
        valid_descriptors = np.asarray(descriptor_list, dtype=np.uint8)
        return (
            np.asarray(points_2d, dtype=np.float32),
            valid_descriptors,
            np.asarray(points_3d, dtype=np.float32),
        )

    def _pixel_to_camera(self, u: float, v: float, depth: float) -> np.ndarray:
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        return np.array([x, y, depth], dtype=np.float32)

    @staticmethod
    def _camera_to_world(point_cam: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
        return rotation.T @ (point_cam - translation)

    def _save(self) -> None:
        payload = {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "views": [
                {
                    "name": v.name,
                    "color_image": v.color_image,
                    "depth_image": v.depth_image,
                    "extrinsic": v.extrinsic,
                    "keypoints": v.keypoints,
                    "descriptors": v.descriptors,
                    "points3d": v.points3d,
                }
                for v in self.views
            ],
        }
        with open(self.cache_path, "wb") as handle:
            pickle.dump(payload, handle)

    def _load(self) -> None:
        with open(self.cache_path, "rb") as handle:
            payload = pickle.load(handle)
        self.image_width = payload["image_width"]
        self.image_height = payload["image_height"]
        self.fx = payload["fx"]
        self.fy = payload["fy"]
        self.cx = payload["cx"]
        self.cy = payload["cy"]
        self.views = [
            SyntheticView(
                name=item["name"],
                color_image=item["color_image"],
                depth_image=item["depth_image"],
                extrinsic=item["extrinsic"],
                keypoints=item["keypoints"],
                descriptors=item["descriptors"],
                points3d=item["points3d"],
            )
            for item in payload["views"]
        ]
