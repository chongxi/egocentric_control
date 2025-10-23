# Depth Generation Pipeline

This folder contains a Python pipeline for converting RGB images to monocular depth maps using the MiDaS models.

## Setup

1. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install --upgrade pip
   ```

2. Install dependencies (CPU builds of PyTorch + extras):

   ```bash
   .venv/bin/pip install torch==2.3.0+cpu torchvision==0.18.0+cpu torchaudio==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu
   .venv/bin/pip install opencv-python timm
   ```

## Usage

Process every image in `images/` (outputs written to `depth_outputs/` by default):

```bash
.venv/bin/python generate_depth.py --model-type DPT_Hybrid
```

Process a single frame and place results in a custom folder:

```bash
.venv/bin/python generate_depth.py \
  --image-path images/frame_000123.jpg \
  --output-dir depth_single
```

Downsample outputs (RGB + depth) to 480x800 resolution:

```bash
.venv/bin/python generate_depth.py \
  --mode downsample \
  --output-dir depth_downsample
```

### Key Options

- `--model-type` chooses the MiDaS backbone (`DPT_Large`, `DPT_Hybrid`, `MiDaS_small`).
- `--device` can be set to `cuda` if a GPU is available; defaults to CPU detection.
- `--output-dir` controls where depth PNG/NPZ files are stored.
- `--mode downsample` additionally writes a resized RGB frame and depth map at 480x800 (override size via `--downsample-size HEIGHT WIDTH`).

The script prints per-image runtime and overall throughput to help gauge performance.
