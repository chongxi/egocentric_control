#!/usr/bin/env bash
# Helper entry-point for replaying a monocular ORB-SLAM3 demo from a video.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORB_DIR="$ROOT/ORB_SLAM3"
VOCAB="$ORB_DIR/Vocabulary/ORBvoc.txt"
VIDEO_ROOT="$ROOT/video"

if [[ ! -f "$VOCAB" ]]; then
  echo "ERROR: Vocabulary not found at $VOCAB" >&2
  exit 1
fi

VIDEO="${1:-$VIDEO_ROOT/scan_20251009_155543.mp4}"
if [[ ! -f "$VIDEO" ]]; then
  echo "ERROR: Video not found: $VIDEO" >&2
  echo "Usage: $0 [/path/to/video.mp4] [optional_settings.yaml]" >&2
  exit 1
fi

SETTINGS="${2:-$VIDEO_ROOT/scan_20251009_155543.yaml}"
VIDEO_BASENAME="$(basename "${VIDEO%.*}")"
FRAME_DIR="$VIDEO_ROOT/${VIDEO_BASENAME}_frames"
DATASET_DIR="$VIDEO_ROOT/${VIDEO_BASENAME}_tum"
RGB_DIR="$DATASET_DIR/rgb"

FRAME_LIMIT="${FRAME_LIMIT:-600}"          # set FRAME_LIMIT=0 to use the full video
FPS="${FPS_OVERRIDE:-30}"                  # override if your footage uses a different FPS

command -v ffmpeg >/dev/null 2>&1 || {
  echo "ERROR: ffmpeg not found. Install with 'sudo apt-get install -y ffmpeg'." >&2
  exit 1
}

mkdir -p "$FRAME_DIR" "$RGB_DIR"

extract_frames() {
  local frame_args=("$@")
  echo "Extracting frames from $VIDEO ..."
  rm -f "$FRAME_DIR"/frame_*.png
  ffmpeg -y -i "$VIDEO" "${frame_args[@]}" "$FRAME_DIR/frame_%06d.png"
}

if [[ "$FRAME_LIMIT" == "0" ]]; then
  # Always regenerate when processing the full clip.
  extract_frames
else
  existing_count=$(find "$FRAME_DIR" -maxdepth 1 -name 'frame_*.png' | wc -l)
  if [[ "$existing_count" -ge "$FRAME_LIMIT" && "${REEXTRACT:-0}" != "1" ]]; then
    echo "Reusing frames in $FRAME_DIR (found $existing_count frames)"
  else
    extract_frames -frames:v "$FRAME_LIMIT"
  fi
fi

echo "Refreshing rgb/ symlinks ..."
find "$RGB_DIR" -type l -delete
for frame in "$FRAME_DIR"/frame_*.png; do
  ln -sfn "../../$(basename "$FRAME_DIR")/$(basename "$frame")" \
    "$RGB_DIR/$(basename "$frame")"
done

echo "Generating rgb.txt ..."
python3 - "$RGB_DIR" "$FPS" <<'PY'
import sys
from pathlib import Path

rgb_dir = Path(sys.argv[1])
fps = float(sys.argv[2])
frames = sorted(rgb_dir.glob("frame_*.png"))
lines = ["# color images", "# timestamp filename", "#"]
for idx, frame in enumerate(frames):
    t = idx / fps
    lines.append(f"{t:.6f} rgb/{frame.name}")
(rgb_dir.parent / "rgb.txt").write_text("\n".join(lines) + "\n")
PY

if [[ ! -f "$SETTINGS" ]]; then
  echo "Creating default settings file at $SETTINGS"
  cat <<EOF >"$SETTINGS"
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
Camera.fps: $FPS
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
else
  echo "Using existing settings file: $SETTINGS"
fi

echo "Launching ORB-SLAM3 viewer ..."
pushd "$ORB_DIR" >/dev/null
./Examples/Monocular/mono_tum "$VOCAB" "$SETTINGS" "$DATASET_DIR"
popd >/dev/null

echo "KeyFrame trajectory written to: $ORB_DIR/KeyFrameTrajectory.txt"
