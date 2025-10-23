import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import cv2
import numpy as np

from .synthetic_views import SyntheticView


@dataclass
class LocalizationRecord:
    image_path: str
    success: bool
    rvec: Optional[np.ndarray]
    tvec: Optional[np.ndarray]
    cam_to_world: Optional[np.ndarray]
    reprojection_error: Optional[float]
    inliers: int
    total_matches: int
    best_view: Optional[str]
    elapsed_ms: float

    def to_json(self) -> dict:
        payload = asdict(self)
        if self.rvec is not None:
            payload["rvec"] = self.rvec.flatten().tolist()
        if self.tvec is not None:
            payload["tvec"] = self.tvec.flatten().tolist()
        if self.cam_to_world is not None:
            payload["cam_to_world"] = self.cam_to_world.tolist()
        return payload


class FrameLocalizer:
    def __init__(
        self,
        views: List[SyntheticView],
        camera_width: int,
        camera_height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        match_ratio: float = 0.85,
        min_correspondences: int = 8,
        min_inliers: int = 8,
        ransac_threshold: float = 12.0,
        refine: bool = True,
    ) -> None:
        self.views = views
        self.match_ratio = match_ratio
        self.min_correspondences = min_correspondences
        self.min_inliers = min_inliers
        self.ransac_threshold = ransac_threshold
        self.refine = refine
        self.camera_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.camera_width = camera_width
        self.camera_height = camera_height

    def localize_image(self, image_path: str) -> LocalizationRecord:
        start = time.perf_counter()
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            elapsed = (time.perf_counter() - start) * 1000.0
            return LocalizationRecord(
                image_path=image_path,
                success=False,
                rvec=None,
                tvec=None,
                cam_to_world=None,
                reprojection_error=None,
                inliers=0,
                total_matches=0,
                best_view=None,
                elapsed_ms=elapsed,
            )
        image = cv2.resize(image, (self.camera_width, self.camera_height))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) == 0:
            elapsed = (time.perf_counter() - start) * 1000.0
            return LocalizationRecord(
                image_path=image_path,
                success=False,
                rvec=None,
                tvec=None,
                cam_to_world=None,
                reprojection_error=None,
                inliers=0,
                total_matches=0,
                best_view=None,
                elapsed_ms=elapsed,
            )

        best_record = None
        best_score = -1.0
        kp_array = np.array([kp.pt for kp in keypoints], dtype=np.float32)

        for view in self.views:
            if view.descriptors.shape[0] == 0:
                continue
            matches = self.matcher.knnMatch(descriptors, view.descriptors, k=2)
            filtered = []
            for m, n in matches:
                if m.distance < self.match_ratio * n.distance:
                    filtered.append(m)
            if len(filtered) < self.min_correspondences:
                continue
            image_points = kp_array[[m.queryIdx for m in filtered]]
            object_points = view.points3d[[m.trainIdx for m in filtered]]
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                iterationsCount=5000,
                reprojectionError=self.ransac_threshold,
                confidence=0.999,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success or inliers is None or len(inliers) < self.min_inliers:
                continue
            inlier_pts_2d = image_points[inliers[:, 0]]
            inlier_pts_3d = object_points[inliers[:, 0]]
            if self.refine:
                rvec, tvec = cv2.solvePnPRefineLM(
                    inlier_pts_3d,
                    inlier_pts_2d,
                    self.camera_matrix,
                    self.dist_coeffs,
                    rvec,
                    tvec,
                )
            projected, _ = cv2.projectPoints(
                inlier_pts_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs
            )
            projected = projected.reshape(-1, 2)
            repro_error = float(np.mean(np.linalg.norm(projected - inlier_pts_2d, axis=1)))
            score = len(inliers) / max(1.0, repro_error)
            if score > best_score:
                elapsed = (time.perf_counter() - start) * 1000.0
                best_score = score
                cam_to_world = self._camera_pose_from_pnp(rvec, tvec)
                best_record = LocalizationRecord(
                    image_path=image_path,
                    success=True,
                    rvec=rvec,
                    tvec=tvec,
                    cam_to_world=cam_to_world,
                    reprojection_error=repro_error,
                    inliers=int(len(inliers)),
                    total_matches=len(filtered),
                    best_view=view.name,
                    elapsed_ms=elapsed,
                )

        if best_record is None:
            elapsed = (time.perf_counter() - start) * 1000.0
            return LocalizationRecord(
                image_path=image_path,
                success=False,
                rvec=None,
                tvec=None,
                cam_to_world=None,
                reprojection_error=None,
                inliers=0,
                total_matches=0,
                best_view=None,
                elapsed_ms=elapsed,
            )
        return best_record

    @staticmethod
    def _camera_pose_from_pnp(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        rotation, _ = cv2.Rodrigues(rvec)
        cam_to_world = np.eye(4, dtype=np.float32)
        cam_to_world[:3, :3] = rotation.T
        cam_to_world[:3, 3] = (-rotation.T @ tvec).flatten()
        return cam_to_world

    @staticmethod
    def save_results(records: List[LocalizationRecord], output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump([record.to_json() for record in records], handle, indent=2)
