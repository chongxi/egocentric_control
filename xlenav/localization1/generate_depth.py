#!/usr/bin/env python3
"""
Generate monocular depth estimates for every image in the `images` folder or a
single image file. Supports full-resolution outputs or a downsampled 480x800
pipeline that also saves RGB frames.

The script uses the pre-trained MiDaS DPT-Large model to infer dense depth for
each RGB frame and stores the result as both a 16-bit PNG (for visualization or
RGB-D style consumption) and a compressed NumPy file with the raw floating point
depth values. Output files are written to `depth_outputs`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable
from time import perf_counter
import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RGB images to depth maps.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "images",
        help="Directory containing RGB images to process.",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        help="Process a single RGB image instead of a directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "depth_outputs",
        help="Directory to save depth predictions.",
    )
    parser.add_argument(
        "--model-type",
        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
        default="DPT_Large",
        help="Choose MiDaS backbone (quality vs speed trade-off).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run inference on.",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "downsample"],
        default="full",
        help="`full` keeps original resolution; `downsample` outputs 480x800 RGB/depth.",
    )
    parser.add_argument(
        "--downsample-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(480, 800),
        help="Target size (H W) when --mode downsample is used.",
    )
    return parser.parse_args()


def load_midas(model_type: str, device: str) -> tuple[torch.nn.Module, Callable]:
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ("DPT_Large", "DPT_Hybrid"):
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform

    return midas, transform


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth_min = np.min(depth)
    depth_max = np.max(depth)
    if depth_max - depth_min < 1e-6:
        return np.zeros_like(depth, dtype=np.uint16)

    depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    return (depth_normalized * 65535).astype(np.uint16)


def process_image(
    image_path: Path,
    output_dir: Path,
    midas: torch.nn.Module,
    transform,
    device: str,
    mode: str,
    downsample_size: tuple[int, int],
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_batch = transform(image_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    rgb_for_save = None
    if mode == "downsample":
        target_h, target_w = downsample_size
        if target_h <= 0 or target_w <= 0:
            raise ValueError("Downsample dimensions must be positive integers.")
        rgb_for_save = cv2.resize(
            image, (target_w, target_h), interpolation=cv2.INTER_AREA
        )
        depth = cv2.resize(
            depth,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )

    depth_uint16 = normalize_depth(depth)

    png_path = output_dir / f"{image_path.stem}_depth.png"
    npz_path = output_dir / f"{image_path.stem}_depth.npz"

    cv2.imwrite(str(png_path), depth_uint16)
    np.savez_compressed(npz_path, depth=depth)
    if rgb_for_save is not None:
        rgb_path = output_dir / f"{image_path.stem}_rgb.png"
        cv2.imwrite(str(rgb_path), rgb_for_save)


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image_path:
        if not args.image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {args.image_path}")
        image_paths = [args.image_path]
    else:
        image_dir: Path = args.image_dir
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        image_paths = sorted(
            p
            for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not image_paths:
            raise RuntimeError(f"No images found in {image_dir}")

    midas, transform = load_midas(args.model_type, args.device)

    total_start = perf_counter()
    processed = 0
    downsample_size = tuple(args.downsample_size)
    for image_path in image_paths:
        image_start = perf_counter()
        process_image(
            image_path,
            output_dir,
            midas,
            transform,
            args.device,
            args.mode,
            downsample_size,
        )
        processed += 1
        print(
            f"Processed {image_path.name} in {perf_counter() - image_start:.2f}s",
            flush=True,
        )

    total_elapsed = perf_counter() - total_start
    print(
        f"Processed {processed} image(s) in {total_elapsed:.2f}s "
        f"(avg {total_elapsed / processed:.2f}s each)",
        flush=True,
    )


if __name__ == "__main__":
    main()
