import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np

from .localizer import FrameLocalizer, LocalizationRecord
from .synthetic_views import SyntheticViewDatabase
from .visualize_results import visualize


def _collect_images(image_dir: Path) -> List[Path]:
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []
    for pattern in patterns:
        image_paths.extend(image_dir.glob(pattern))
    return sorted(image_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-frame localization against a point cloud.")
    parser.add_argument("--ply", type=Path, default=Path("points.ply"), help="PLY point cloud path.")
    parser.add_argument("--images", type=Path, default=Path("images"), help="Directory with input images.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory for localization outputs.")
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Optional path to synthetic view cache (defaults to output/synthetic_views.pkl).",
    )
    parser.add_argument("--image-width", type=int, default=640, help="Internal image width for localization.")
    parser.add_argument("--image-height", type=int, default=480, help="Internal image height for localization.")
    parser.add_argument("--fx", type=float, default=None, help="Camera focal length in pixels (x axis).")
    parser.add_argument("--fy", type=float, default=None, help="Camera focal length in pixels (y axis).")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x in pixels.")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y in pixels.")
    parser.add_argument("--num-views", type=int, default=80, help="Number of synthetic viewpoints to render.")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force regeneration of synthetic viewpoint cache.",
    )
    parser.add_argument("--visualize", action="store_true", help="Launch Open3D viewer after localization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache if args.cache is not None else (output_dir / "synthetic_views.pkl")

    db = SyntheticViewDatabase(
        ply_path=str(args.ply),
        cache_path=str(cache_path),
        image_width=args.image_width,
        image_height=args.image_height,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        num_views=args.num_views,
    )
    print(f"[SyntheticDB] Loading or building cache at {cache_path}")
    db.load_or_build(force_rebuild=args.force_rebuild)
    print(f"[SyntheticDB] Loaded {len(db.views)} synthetic views.")

    localizer = FrameLocalizer(
        views=db.views,
        camera_width=db.image_width,
        camera_height=db.image_height,
        fx=db.fx,
        fy=db.fy,
        cx=db.cx,
        cy=db.cy,
    )

    image_paths = _collect_images(args.images)
    if not image_paths:
        print("[Localizer] No images found.")
        return

    results: List[LocalizationRecord] = []
    print(f"[Localizer] Processing {len(image_paths)} frames...")
    for path in image_paths:
        record = localizer.localize_image(str(path))
        results.append(record)
        status = "OK" if record.success else "FAIL"
        print(
            f"[Localizer] {path.name:>20} | {status} | "
            f"{record.elapsed_ms:6.1f} ms | inliers={record.inliers:3d} | matches={record.total_matches:3d}"
        )

    json_path = output_dir / "localization_results.json"
    FrameLocalizer.save_results(results, str(json_path))
    print(f"[Localizer] Saved results to {json_path}")

    csv_path = output_dir / "localization_timings.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image",
                "success",
                "elapsed_ms",
                "inliers",
                "total_matches",
                "reprojection_error",
                "best_view",
            ]
        )
        for record in results:
            writer.writerow(
                [
                    Path(record.image_path).name,
                    int(record.success),
                    f"{record.elapsed_ms:.3f}",
                    record.inliers,
                    record.total_matches,
                    f"{record.reprojection_error:.5f}" if record.reprojection_error else "",
                    record.best_view or "",
                ]
            )
    print(f"[Localizer] Wrote timings to {csv_path}")

    successful = [r for r in results if r.success and r.cam_to_world is not None]
    if successful:
        pose_names = np.array([Path(r.image_path).name for r in successful])
        pose_mats = np.stack([r.cam_to_world for r in successful], axis=0)
        poses_path = output_dir / "camera_poses.npz"
        np.savez_compressed(poses_path, names=pose_names, cam_to_world=pose_mats)
        print(f"[Localizer] Stored {pose_mats.shape[0]} camera poses in {poses_path}")
    else:
        print("[Localizer] No successful poses to store.")

    if args.visualize:
        print("[Visualizer] Launching Open3D preview...")
        visualize(
            point_cloud_path=args.ply,
            records_path=json_path,
            fx=db.fx,
            fy=db.fy,
            cx=db.cx,
            cy=db.cy,
            width=db.image_width,
            height=db.image_height,
        )


if __name__ == "__main__":
    main()
