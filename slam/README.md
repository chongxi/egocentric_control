# MASt3R-SLAM Debug Session Notes

These notes capture what was attempted in this workspace to run MASt3R-SLAM on the YouTube video stored at `slam/video/scan_20251009_155543.mp4`. They should help anyone revisiting this setup understand the current state, the modifications that were made, and why the pipeline still stops after the first frame.

## Environment Preparation

1. **Base environment**: `conda create -n mast3r-slam python=3.11` (per upstream README) and activated for all steps.
2. **PyTorch**: Installed CUDA 12.4 toolkit-compatible wheel:
   ```powershell
   pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
   pip install numpy==1.26.4  # downgrade to satisfy compiled deps
   ```
3. **Editable packages**: After resolving toolchain issues, the following succeeded:
   ```powershell
   pip install -e thirdparty/mast3r
   pip install -e thirdparty/in3d
   pip install --no-build-isolation -e .
   ```
4. **Torch codec import fix**: `mast3r_slam/dataloader.py` now tries both the legacy (`torchcodec.decoders.VideoDecoder`) and the new (`torchcodec.VideoDecoder`) import paths and prints which one succeeds. Without torchcodec it falls back to OpenCV with a warning. Despite installing `torchcodec 0.7.0`, the import still fails in this environment, so OpenCV is used.
5. **Checkpoint downloads**: All MASt3R checkpoints were fetched into `slam/MASt3R-SLAM/checkpoints/` with `Start-BitsTransfer`. The primary model file had to be redownloaded because the first copy was corrupted.

## Code Changes in This Workspace

| File | Purpose |
| ---- | ------- |
| `mast3r_slam/backend/src/*.cu` | Switched all `long` tensor accessors to `int64_t` to make the CUDA extension compile on MSVC. |
| `mast3r_slam/dataloader.py` | Added torchcodec import fallback and better diagnostics; prints the actual decoder path in use. |
| `main.py` | Added `--single-thread` CLI flag that flips the config value before startup. |
| `run_video_debug.py` _(new)_ | Stand-alone entry point with extra logging and a sequential `VideoCapture` loader to decouple video ingestion from torchcodec. |
| `config/base.yaml` | Adjusted several parameters for experimentation (see timeline below). |

No intrusive changes were made to core algorithms; everything is logged in-place without removing existing functionality.

## Attempted Workflow

1. **Standard entry (`python main.py ...`)**: Loads the model, renders one frame, then the UI freezes. Root cause: the multiprocessing backend crashes due to GPU out-of-memory (OOM) when visualization is active. Running with `--no-viz` still stalls because PyTorch/MASt3R consumes ~9.3 GB for the first frame, leaving little headroom for subsequent steps.
2. **Debug entry (`run_video_debug.py`)**:
   - Provides verbose logging (`--log-every N`, `--max-frames N`, `--no-viz`, `--single-thread`).  
   - Uses a sequential loader so `cv2.VideoCapture` isn’t reseeking for each frame.
   - Added single-threaded backend execution so we could step through optimization without forking subprocesses.
   - Despite the streamlined flow, frame 1 still takes several minutes and never finishes because the GPU remains at ~9.3 GB before MASt3R matching kicks in. With CUDA launch blocking enabled we saw no explicit error—work simply progresses extremely slowly.

## Parameter Tweaks That Were Tested

| Setting | Motivation | Result |
| ------- | ---------- | ------ |
| `dataset.img_downsample = 2`, then `4` | Reduce per-frame resolution (512→256→128). | VRAM dropped a little, but frame 1 still consumed most of the 12 GB budget once tracking began. |
| `matching.radius = 2`, `matching.dilation_max = 3` | Smaller correspondence search window. | No observable improvement; still GPU-bound. |
| `local_opt.use_cuda = False` | Move Gauss–Newton solve to CPU. | Not enough reduction; MASt3R front-end still exhausts VRAM. |
| `--single-thread` | Avoid spawning backend process and running a second CUDA context. | Keeps the app alive, but the first tracking step remains heavy. |
| `torchcodec` import | Wanted faster video access. | Package installs but still cannot be imported; falling back to OpenCV remains a minor slowdown but not the root cause. |

Even with the lightest configuration (downsample = 4, CPU optimizer), frame 1 never cleared within a reasonable time on an RTX 3060. `nvidia-smi` confirmed ~9.2 GB of VRAM in use and 0% util afterwards, indicating the GPU is simply at its limit for this model.

## Current Status (October 2025)

- Environment builds cleanly; all MASt3R-SLAM dependencies install, CUDA extensions compile, and checkpoints load.
- A custom debug entry (`run_video_debug.py`) exists to iterate quickly without the GUI.
- Video processing still stops after the first frame due to resource exhaustion on a 12 GB GPU. No additional coding bugs are apparent—the pipeline is just too memory-intensive at full MASt3R resolution.

## Suggested Next Steps

1. **Use a smaller backbone**: Check if the upstream project offers a lighter MASt3R checkpoint (e.g., base or tiny variants) that fit within 8–10 GB.
2. **Run on larger GPU or cloud instance**: An RTX 4090 (24 GB) is what the authors used; matching that hardware is the simplest fix.
3. **Aggressive downsampling**: If visuals can be lower fidelity, push `img_downsample` to 8 or preprocess the video externally to 256×256 before feeding it in.
4. **Disable retrieval / local optimization temporarily**: Cutting out global optimization (`local_opt.window_size=0`, disable retriever) might save memory enough to advance frames for debugging.
5. **Torchcodec fix**: If a local copy of torchcodec can be imported, MP4 decoding will be faster, though it won’t reduce VRAM usage.

All relevant commands and outputs are preserved in the PowerShell history. Whenever the hardware situation improves—or a lighter checkpoint is available—the debug runner should make it easy to verify progress by watching the `[DEBUG] Frame …` log lines advance.

---

## DROID-SLAM Experiments (October 2025)

Following the MASt3R-SLAM attempts we evaluated [DROID-SLAM](../slam/DROID-SLAM) on the same `scan_20251009_155543.mp4` video. This section records what was added, what worked, and the remaining blockers.

### Summary of work completed

- Cloned upstream DROID-SLAM into `slam/DROID-SLAM`.
- Rebuilt the CUDA extension with Windows-friendly flags (`/GL-`, `/LTCG:OFF`) so the MSVC linker no longer runs out of heap.
- Added a `video_demo.py` helper that streams frames straight from an MP4 while keeping the OpenGL viewer alive. It exposes `--log_interval` for periodic diagnostics.
- Enhanced `demo.py` with the same logging option and headless support (`--disable_vis`). The visualiser was instrumented to print the number of valid points as the map grows.
- Automatic timestamped logs (`demo_run_YYYYMMDD_HHMMSS.log`) are written on each run for later inspection. Reference logs live alongside the repo root.
- Expanded `slam/DROID-SLAM/README.md` to document the workflow and troubleshooting steps.

### Observations

- Sample datasets (ETH/EuRoC/TUM) run to completion when the native extension survives, producing reconstructions via `--reconstruction_path`.
- Diagnostics appear as expected, e.g.

  ```
  [demo] frame=640 processed=660 map=221 frontend_count=31 fps=26.12
  [demo] completed | frames=660 | map=221
  ```

- `video_demo.py` confirms the pipeline accepts MP4 footage and reports progress every *N* frames.

### Outstanding issues

- **CUDA access violation on Windows**: The `droid_backends` extension consistently crashes after ~20 frames with exit code `-1073741819`. The failure occurs in native code, so Python never surfaces a traceback. Recommended next steps are to rebuild the extension with debug symbols or run the project inside WSL2/Linux where DROID-SLAM is better supported.
- **Empty viewer on custom footage**: Because the crash happens before enough keyframes are accepted, the OpenGL window never displays a point cloud. Accurate intrinsics and relaxed thresholds are in place, but the native crash must be resolved first.

### Where to look

- Logs capturing successful runs: `slam/DROID-SLAM/demo_reference.log`, `demo_run_*.log`.
- Video workflow and usage instructions: `slam/DROID-SLAM/README.md`.
- Video helper script: `slam/DROID-SLAM/video_demo.py`.

Once the CUDA crash is addressed the logging infrastructure and video entry point should make it straightforward to rerun the experiment and monitor progress from the console.



