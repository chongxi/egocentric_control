# vggt_loc.py
# VGGT localization for a NEW image within an existing 3D scene from keyframes.
# - Loads keyframes from a folder + a new query image
# - Runs VGGT to estimate the new image's pose in the same coordinate system
# - Prints detailed step-by-step progress and timing
# - Visualizes cameras (keyframes gray, new orange) and optional point cloud
# - Based on vggt_reconstruction_walkthrough.ipynb

import argparse
import os
from pathlib import Path
import time
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image
import open3d as o3d

print("=" * 80)
print("VGGT Localization - Camera Pose Estimation")
print("=" * 80)

# ----------------------- Helpers -----------------------

def add_repo_to_syspath(repo_dir: str | None):
    if repo_dir:
        import sys
        repo_dir = os.path.abspath(repo_dir)
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

def list_images(dir_path: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() in exts])

def make_camera_axes(c2w: np.ndarray, axis_len: float = 0.05, color=(0.6, 0.6, 0.6)) -> o3d.geometry.LineSet:
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    o = t
    x = t + R[:, 0] * axis_len
    y = t + R[:, 1] * axis_len
    z = t + R[:, 2] * axis_len
    pts = np.stack([o, x, y, z], axis=0)
    lines = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.array(color)[None, :], (3, 1)))
    return ls

def to_homogeneous_4x4(batch_3x4: torch.Tensor) -> torch.Tensor:
    # (S,3,4) -> (S,4,4)
    S = batch_3x4.shape[0]
    out = torch.zeros((S, 4, 4), dtype=batch_3x4.dtype, device=batch_3x4.device)
    out[:, :3, :4] = batch_3x4
    out[:, 3, 3] = 1.0
    return out

# ----------------------- Main -----------------------

def main():
    parser = argparse.ArgumentParser(description="VGGT localization: estimate new image pose in keyframe scene.")
    parser.add_argument("--keyframes_dir", type=str, required=True, help="Folder containing keyframe images")
    parser.add_argument("--new_image", type=str, required=True, help="Path to the NEW image to localize")
    parser.add_argument("--vggt_repo", type=str, default=None, help="Path to local vggt repo (if not pip-installed)")
    parser.add_argument("--axis_len", type=float, default=0.05, help="Axis length for camera triads in viewer")
    parser.add_argument("--show_points", action="store_true", help="Also render a filtered point cloud (slower)")
    parser.add_argument("--conf_percentile", type=float, default=25.0, help="Drop lowest-confidence depth pixels below this percentile")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save outputs (point cloud, poses)")
    parser.add_argument("--save_pcd", action="store_true", help="Save point cloud to file")
    parser.add_argument("--save_pose", action="store_true", help="Save new image pose to file")
    args = parser.parse_args()

    print("\n[STEP 1/8] Setting up environment...")
    add_repo_to_syspath(args.vggt_repo)

    # Import after sys.path tweak
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3
    print("    ✓ VGGT modules imported successfully")

    print("\n[STEP 2/8] Loading images...")
    keyframes_dir = Path(args.keyframes_dir)
    assert keyframes_dir.is_dir(), f"Not a directory: {keyframes_dir}"
    new_image_path = Path(args.new_image)
    assert new_image_path.is_file(), f"File not found: {new_image_path}"

    keyframe_paths = list_images(keyframes_dir)
    if len(keyframe_paths) == 0:
        raise RuntimeError(f"No images in keyframes_dir: {keyframes_dir}")

    print(f"    ✓ Found {len(keyframe_paths)} keyframe(s) in {keyframes_dir}")
    print(f"    ✓ Query image: {new_image_path.name}")

    # Order = [keyframes..., new]
    all_paths = [str(p) for p in keyframe_paths] + [str(new_image_path)]

    # Device + dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device.type == "cuda" else nullcontext()
    print(f"    ✓ Using device: {device}, dtype: {dtype}")

    print("\n[STEP 3/8] Preprocessing images...")
    t_preprocess = time.perf_counter()
    images_tensor = load_and_preprocess_images(all_paths).to(device)  # (S,3,H,W)
    batched = images_tensor.unsqueeze(0)  # (1,S,3,H,W)
    print(f"    ✓ Preprocessed tensor shape: {images_tensor.shape}")
    print(f"    ✓ Preprocessing time: {time.perf_counter() - t_preprocess:.4f}s")

    print("\n[STEP 4/8] Loading VGGT model...")
    t_model_load = time.perf_counter()
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    print(f"    ✓ Model loaded in {time.perf_counter() - t_model_load:.4f}s")

    # ---------- Inference (timed) ----------
    print("\n[STEP 5/8] Running VGGT inference...")
    print("    → Aggregating image tokens...")
    t0 = time.perf_counter()
    with torch.no_grad(), autocast_ctx:
        # aggregator tokens
        t_agg = time.perf_counter()
        tokens_list, patch_start_idx = model.aggregator(batched)
        print(f"      ✓ Aggregator done ({len(tokens_list)} refinement iterations) in {time.perf_counter() - t_agg:.4f}s")

        # camera (poses)
        print("    → Predicting camera poses...")
        t_cam = time.perf_counter()
        pose_enc_list = model.camera_head(tokens_list)
        pose_enc = pose_enc_list[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batched.shape[-2:])
        print(f"      ✓ Camera head done in {time.perf_counter() - t_cam:.4f}s")

        # depth for optional points
        print("    → Predicting depth maps...")
        t_depth = time.perf_counter()
        depth_map, depth_conf = model.depth_head(tokens_list, batched, patch_start_idx)
        print(f"      ✓ Depth head done in {time.perf_counter() - t_depth:.4f}s")

    elapsed = time.perf_counter() - t0
    print(f"    ✓ Total inference time: {elapsed:.4f}s")

    print("\n[STEP 6/8] Converting poses to world coordinates...")
    # extrinsic: (1,S,3,4) - world-to-camera transformation (OpenCV convention)
    extrinsic_b1 = extrinsic.squeeze(0).detach()        # (S,3,4)
    extrinsic_h = to_homogeneous_4x4(extrinsic_b1)      # (S,4,4)
    # convert to cam->world using closed-form inverse
    cam_to_world = closed_form_inverse_se3(extrinsic_h).cpu().numpy()  # (S,4,4)
    new_pose_c2w = cam_to_world[-1]
    print(f"    ✓ Converted {cam_to_world.shape[0]} camera poses")
    print(f"    ✓ New image is frame #{len(keyframe_paths)} (0-indexed)")

    # ---------- Print outputs ----------
    np.set_printoptions(precision=5, suppress=True)
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n[Timing] Total inference time: {elapsed:.4f}s")
    print(f"[Timing] Images processed: {len(all_paths)} ({len(keyframe_paths)} keyframes + 1 new)")
    print(f"\n[New Image Pose] Camera-to-World (4x4 matrix):")
    print(new_pose_c2w)
    print(f"\n[New Image Position] XYZ: {new_pose_c2w[:3, 3]}")

    # ---------- Save outputs ----------
    output_dir = Path(args.output_dir)
    if args.save_pose or args.save_pcd:
        print(f"\n[STEP 7/8] Saving outputs to {output_dir}...")
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.save_pose:
            # Save new image pose
            pose_file = output_dir / f"pose_{new_image_path.stem}.txt"
            np.savetxt(pose_file, new_pose_c2w, fmt='%.8f', header=f'Camera-to-World pose for {new_image_path.name}\nRotation (3x3) and Translation (3x1) matrix')
            print(f"    ✓ Saved pose to: {pose_file}")

            # Also save in JSON format for easier programmatic access
            import json
            pose_json = output_dir / f"pose_{new_image_path.stem}.json"
            pose_data = {
                "image_name": new_image_path.name,
                "camera_to_world": new_pose_c2w.tolist(),
                "position_xyz": new_pose_c2w[:3, 3].tolist(),
                "rotation_matrix": new_pose_c2w[:3, :3].tolist(),
                "keyframes_used": [p.name for p in keyframe_paths]
            }
            with open(pose_json, 'w') as f:
                json.dump(pose_data, f, indent=2)
            print(f"    ✓ Saved pose (JSON) to: {pose_json}")

    # ---------- Visualization ----------
    step_num = 8 if args.save_pose or args.save_pcd else 7
    print(f"\n[STEP {step_num}/8] Preparing visualization...")
    geoms = []
    pcd = None  # Initialize for later saving

    # Optional point cloud from depth (filter like your notebook)
    if args.show_points:
        print("    → Unprojecting depth to 3D points...")
        # Unproject *all* frames
        point_map_by_unproj = unproject_depth_map_to_point_map(
            depth_map.squeeze(0), extrinsic_b1, intrinsic.squeeze(0)
        )  # (S,H,W,3) numpy
        print(f"      ✓ Unprojected point map shape: {point_map_by_unproj.shape}")

        print("    → Filtering by confidence...")
        conf_flat = depth_conf[0].detach().cpu().numpy().reshape(-1)
        points_flat = point_map_by_unproj.reshape(-1, 3)

        # Colors from images
        colors_flat = images_tensor.detach().cpu().permute(0, 2, 3, 1).numpy().reshape(-1, 3)

        cutoff = np.percentile(conf_flat, args.conf_percentile)
        valid = (conf_flat >= cutoff) & (conf_flat > 1e-5)
        print(f"      ✓ Confidence cutoff (p{args.conf_percentile}): {cutoff:.6f}")
        print(f"      ✓ Keeping {valid.sum():,}/{len(valid):,} points ({100*valid.sum()/len(valid):.1f}%)")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_flat[valid])
        pcd.colors = o3d.utility.Vector3dVector(colors_flat[valid])

        # Optional: statistical outlier removal like in notebook
        print("    → Removing statistical outliers...")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
        print(f"      ✓ After outlier removal: {len(pcd.points):,} points")

        geoms.append(pcd)

    # Save point cloud if requested
    if args.save_pcd and pcd is not None:
        pcd_file = output_dir / "point_cloud.ply"
        o3d.io.write_point_cloud(str(pcd_file), pcd)
        print(f"    ✓ Saved point cloud to: {pcd_file}")
    elif args.save_pcd and pcd is None:
        print("    ⚠ Cannot save point cloud: use --show_points to generate it first")

    # Camera frames
    print("    → Creating camera visualizations...")
    S = cam_to_world.shape[0]
    for i in range(S - 1):
        geoms.append(make_camera_axes(cam_to_world[i], axis_len=args.axis_len, color=(0.6, 0.6, 0.6)))  # gray
    geoms.append(make_camera_axes(cam_to_world[-1], axis_len=args.axis_len, color=(1.0, 0.5, 0.0)))      # orange
    # world axes at origin
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_len * 3.0))
    print(f"    ✓ Created {S} camera frames (gray=keyframes, orange=new query)")

    # Save all camera poses if requested
    if args.save_pose:
        all_poses_file = output_dir / "all_camera_poses.txt"
        with open(all_poses_file, 'w') as f:
            f.write(f"# All camera poses (camera-to-world)\n")
            f.write(f"# Total frames: {S} ({len(keyframe_paths)} keyframes + 1 new)\n")
            f.write(f"# New image: {new_image_path.name}\n\n")
            for i in range(S - 1):
                f.write(f"# Frame {i}: {keyframe_paths[i].name}\n")
                np.savetxt(f, cam_to_world[i], fmt='%.8f')
                f.write("\n")
            f.write(f"# Frame {S-1}: {new_image_path.name} (NEW)\n")
            np.savetxt(f, cam_to_world[-1], fmt='%.8f')
        print(f"    ✓ Saved all camera poses to: {all_poses_file}")

    print(f"\n[STEP 8/8] Launching 3D viewer...")
    print("    (Close the viewer window to exit)")
    o3d.visualization.draw_geometries(
        geoms,
        window_name="VGGT Localization (gray=keyframes, orange=new)",
        width=1280,
        height=800,
        left=80,
        top=60,
    )

    print("\n" + "=" * 80)
    print("DONE!")
    if args.save_pose or args.save_pcd:
        print(f"\nOutputs saved to: {output_dir.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    main()
