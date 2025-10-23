# Localization Pipeline (Point Cloud ↔ Webcam Frames)

This directory contains a self–contained pipeline that localizes individual webcam frames inside the coordinate frame of the `points.ply` point cloud without relying on COLMAP. Each image is matched independently against a bank of synthetic renderings of the point cloud, and a PnP solve recovers the camera pose when enough correspondences are found. The same code logs per–frame timing and can optionally render an Open3D scene that overlays the point cloud with the recovered camera frusta.

## 1. Environment Setup

```bash
cd /mnt/g/GithubProject/egocentric_control
python3 -m venv .venv_localization
. .venv_localization/bin/activate
pip install -r xlenav/localization1/requirements.txt
```

You can leave the virtual environment active while running the localization and visualization steps. The requirements pin `numpy==1.26.4` to keep OpenCV compatible.

## 2. Running Per-Image Localization

```bash
. .venv_localization/bin/activate
python -m xlenav.localization1.run_localization \
  --ply xlenav/localization1/points.ply \
  --images xlenav/localization1/images \
  --output-dir xlenav/localization1/output \
  --num-views 80 \
  --image-width 640 \
  --image-height 480
```

Key options:
- `--num-views`: number of synthetic viewpoints rendered around the point cloud. More views improve coverage at the cost of precomputation time.
- `--image-width/--image-height`: internal resolution used for feature extraction. Adjust to match your webcam feed.
- `--fx/--fy/--cx/--cy`: override the default pinhole intrinsics if you have a calibrated camera.
- `--force-rebuild`: regenerate the synthetic view cache even if a cached copy exists.
- `--visualize`: trigger the Open3D viewer once localization finishes.

The first run builds `output/synthetic_views.pkl`, a cache of rendered color/depth images plus associated 2D–3D feature tracks. Subsequent runs reuse the cache unless `--force-rebuild` is supplied.

During localization the script prints a one-line status per frame, including elapsed time in milliseconds, the number of tentative matches, and the number of inliers returned by the PnP solver.

## 3. Outputs

All artifacts are written to the folder specified by `--output-dir`:

- `localization_results.json`: structured log with success flag, inliers, reprojection error, timing, raw PnP parameters, and a `4×4` camera-to-world matrix for each frame.
- `localization_timings.csv`: compact CSV (image, success, elapsed_ms, inliers, matches) for plotting or benchmarking.
- `camera_poses.npz`: stacked camera-to-world matrices (`names`, `cam_to_world`) for every successful localization, ready for downstream consumption. Omitted if no frame succeeds.
- `synthetic_views.pkl`: cached synthetic view feature database (created on demand).

## 4. Visualizing Point Cloud + Camera Poses

To overlay the point cloud and the recovered camera frusta:

```bash
. .venv_localization/bin/activate
python -m xlenav.localization1.run_localization \
  --ply xlenav/localization1/points.ply \
  --images xlenav/localization1/images \
  --output-dir xlenav/localization1/output \
  --visualize
```

The visualizer reads the JSON output, drops any failed frames (unless configured otherwise), builds a frustum mesh from the stored camera intrinsics, and launches the Open3D interactive viewer.

## 5. Reusing or Extending the Pipeline

- **Switching datasets**: point `--ply` and `--images` to new data. Delete the cached `synthetic_views.pkl` or pass `--force-rebuild` if the point cloud changes.
- **Real-time integration**: the localization core is contained in `localizer.py`. You can import `FrameLocalizer` inside your own application, feed frames from a live webcam, and reuse the timing metrics returned in each `LocalizationRecord`.
- **Tuning performance**: reduce `--num-views` or ORB feature counts (see `synthetic_views.py` and `localizer.py`) for faster but less robust matching; increase them for better coverage.
- **Improving accuracy**: replace ORB with SuperPoint/SuperGlue or plug in a global image retriever before the per-view matching loop. The current structure isolates feature extraction to make such swaps straightforward.

For any recurring deployment, collect a quick checkerboard calibration of the webcam to populate `--fx/--fy/--cx/--cy` with accurate intrinsics—this significantly tightens the PnP solves.
