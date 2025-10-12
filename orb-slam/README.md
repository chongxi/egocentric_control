# ORB-SLAM3 on WSL – Quickstart & Video Demo

This guide documents how to build the bundled [`ORB_SLAM3`](ORB_SLAM3) tree under WSL, avoid the most common build traps, and replay a monocular SLAM demo from a recorded video. The instructions were validated on Windows 11 + WSL2 (Ubuntu 24.04).

---

## 1. Prerequisites

- **WSL2 distro** with a recent Ubuntu (22.04+ recommended, 24.04 used here).
- **WSLg enabled** (default on Windows 11) or an X‑server on Windows 10 (see [WSL GUI tips](#4-wsl-gui-notes)).
- **CMake ≥ 3.16** (ships with Ubuntu 24.04).
- **Compiler**: GCC 11+ supports the required C++17 features.

Install the required system packages:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git libeigen3-dev libopencv-dev \
    libboost-dev libboost-filesystem-dev libboost-system-dev libboost-thread-dev \
    libglew-dev libpython3-dev freeglut3-dev libxi-dev libxmu-dev ffmpeg
```

> `ffmpeg` is only needed for the video-to-frames conversion used in the demo.

Clone Pangolin (already present here in `orb-slam/Pangolin`) if you need to rebuild it from scratch:

```bash
git clone https://github.com/stevenlovegrove/Pangolin.git
```

---

## 2. Build Steps

### 2.1 Pangolin

```bash
cd orb-slam/Pangolin
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

### 2.2 ORB-SLAM3 core

```bash
cd /mnt/g/GithubProject/egocentric_control/orb-slam/ORB_SLAM3
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Artifacts land in:

- `ORB_SLAM3/lib/libORB_SLAM3.so`
- Example binaries under `ORB_SLAM3/Examples/...`

### 2.3 Known build gotchas (already addressed)

- **Modern Pangolin + GCC 13 needs C++17**. The `CMakeLists.txt` included in this repo already promotes the standard to C++17 (while keeping legacy macros for existing `#ifdef COMPILEDWITHC11` checks). If you base your setup on the upstream repo, apply the same change or you’ll hit `std::decay_t` / `std::enable_if_t` errors in `sigslot`.
- **`mnFullBAIdx++` with `bool`** fails under C++17. The `LoopClosing` member has been updated to `int` here. If you patch another clone, ensure `mnFullBAIdx` is an integer.
- **Pangolin missing**: if CMake cannot find it, verify `/usr/local/lib/cmake/Pangolin` exists and add `-DPangolin_DIR=/usr/local/lib/cmake/Pangolin`.
- **OpenCV version**: This tree expects OpenCV ≥ 4.4. Ubuntu 24.04 ships 4.6 which works out of the box.

---

## 3. Running the Monocular Demo on a Video

This demo extracts the first ~20 seconds of a 30 FPS 1080p video and feeds a synthetic TUM-style dataset to the monocular example.

### 3.1 Prepare frames and timestamps

```bash
VIDEO_ROOT=/mnt/g/GithubProject/egocentric_control/orb-slam/video
VIDEO=$VIDEO_ROOT/scan_20251009_155543.mp4
FRAME_DIR=$VIDEO_ROOT/scan_20251009_155543_frames
DATASET_DIR=$VIDEO_ROOT/scan_20251009_155543_tum

mkdir -p "$FRAME_DIR" "$DATASET_DIR/rgb"

# Extract the first 600 frames (~20 s @ 30 FPS)
ffmpeg -y -i "$VIDEO" -frames:v 600 "$FRAME_DIR/frame_%06d.png"

# Symlink frames into the dataset folder
for f in "$FRAME_DIR"/frame_*.png; do
    ln -sfn "../../scan_20251009_155543_frames/$(basename "$f")" \
        "$DATASET_DIR/rgb/$(basename "$f")"
done

# Generate TUM-style timestamps
python3 - <<'PY'
from pathlib import Path
root = Path("/mnt/g/GithubProject/egocentric_control/orb-slam/video/scan_20251009_155543_tum")
rgb = root / "rgb"
files = sorted(rgb.glob("frame_*.png"))
fps = 30.0
lines = ["# color images", "# timestamp filename", "#"]
for idx, path in enumerate(files):
    t = idx / fps
    lines.append(f"{t:.6f} rgb/{path.name}")
(root / "rgb.txt").write_text("\n".join(lines) + "\n")
PY
```

Create a basic calibration file (update the intrinsics if you have a true calibration):

```bash
cat <<'EOF' > $VIDEO_ROOT/scan_20251009_155543.yaml
%YAML:1.0
Camera.type: "PinHole"
Camera1.fx: 1200.0
Camera1.fy: 1200.0
Camera1.cx: 960.0
Camera1.cy: 540.0
Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0
Camera1.k3: 0.0
Camera.fps: 30
Camera.RGB: 1
Camera.width: 1920
Camera.height: 1080
ORBextractor.nFeatures: 2000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
EOF
```

### 3.2 Launch ORB-SLAM3

```bash
cd /mnt/g/GithubProject/egocentric_control/orb-slam/ORB_SLAM3
./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt \
    /mnt/g/GithubProject/egocentric_control/orb-slam/video/scan_20251009_155543.yaml \
    /mnt/g/GithubProject/egocentric_control/orb-slam/video/scan_20251009_155543_tum
```

You should see the Pangolin viewer (trajectory + tracking windows). When the sequence finishes:

- `KeyFrameTrajectory.txt` is written in the working directory.
- Console prints mean/median tracking times (~28 ms on the sample).

> To process a longer portion of the video, increase `-frames:v` or remove the switch and regenerate `rgb.txt` (timestamps must stay 1/30 s apart).

### 3.3 One-click entry point

Prefer an automated path? Run the helper script, which will extract frames (if missing), refresh timestamps, generate a default calibration file (if absent), and launch the viewer:

```bash
cd /mnt/g/GithubProject/egocentric_control/orb-slam
./run_video_demo.sh
```

You may override the video, settings, or frame limit:

```bash
FRAME_LIMIT=0 ./run_video_demo.sh /path/to/video.mp4 custom_settings.yaml
```

`FRAME_LIMIT=0` forces a fresh extraction of every frame in the video. Use `REEXTRACT=1` to rebuild a truncated cache, and `FPS_OVERRIDE` if your recording is not 30 FPS.

---

## 4. WSL GUI Notes

- **Windows 11 / WSLg**: GUI apps just work. `./mono_tum` connects to the built-in RDP-based compositor. Ensure `echo $DISPLAY` returns something like `:0` and no extra configuration is needed.
- **Windows 10**: Install an X server (e.g. [VcXsrv](https://sourceforge.net/projects/vcxsrv/)) on the host, launch it, and export `DISPLAY` from WSL:
  ```bash
  export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
  export LIBGL_ALWAYS_INDIRECT=1
  ```
  Then run the SLAM binary; the Pangolin window will appear on the Windows desktop.
- If the viewer refuses to open, verify:
  - `glxinfo` runs without errors (install `mesa-utils` if missing).
  - You are not SSHed into WSL without X forwarding.

---

## 5. Troubleshooting Checklist

- `make` dies in `sigslot/signal.hpp`: confirm you reconfigured after the CMakeLists update and you see `Using flag -std=c++17` during configure.
- `Pangolin` not found: rerun `sudo ldconfig`, and export `PKG_CONFIG_PATH=/usr/local/lib/pkgconfig` before CMake.
- `imread` fails in the demo: ensure the symlink tree points to actual frames (`file dataset/rgb/frame_000001.png` should resolve, not “broken symlink”).
- Viewer window invisible on Windows 10: start your X server first, confirm `xeyes` or `glxgears` opens, then launch ORB-SLAM3.

Happy mapping!
