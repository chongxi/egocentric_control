import argparse
import math
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

DEFAULT_RESULT_PATH = Path("vggt-loc") / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect PLY point clouds under a result directory and report whether "
            "they appear to live in the same coordinate space. Optionally run ICP "
            "to estimate rigid transforms relative to a reference sample."
        )
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default=DEFAULT_RESULT_PATH,
        help=f"Directory that contains subfolders with PLY files (default: {DEFAULT_RESULT_PATH})",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Name of the subfolder to use as reference. Defaults to the first subfolder alphabetically.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=4000,
        help=(
            "Maximum number of points to sample from each cloud for ICP and statistics. "
            "Larger samples are truncated to this size."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        help="Maximum number of ICP iterations per comparison.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Convergence tolerance on the change in mean error between ICP iterations.",
    )
    parser.add_argument(
        "--skip-icp",
        action="store_true",
        help="Skip ICP and only report descriptive statistics.",
    )
    parser.add_argument(
        "--icp-device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "Device to run ICP on: 'cpu', 'cuda', or 'auto' (use cuda if available). "
            "Using 'cpu' avoids GPU->NumPy conversion issues in some PyTorch3D versions."
        ),
    )
    return parser.parse_args()


def find_ply_file(directory: Path) -> Optional[Path]:
    for candidate in sorted(directory.iterdir()):
        if candidate.suffix.lower() == ".ply" and candidate.is_file():
            return candidate
    return None


def load_point_cloud(file_path: Path) -> np.ndarray:
    loaders = (_load_with_open3d, _load_with_plyfile, _load_ascii_ply)
    errors = []
    for loader in loaders:
        try:
            points = loader(file_path)
            if points is not None and len(points) > 0:
                return points
        except Exception as exc:  # noqa: BLE001 - we want to aggregate any loader issues
            errors.append(f"{loader.__name__}: {exc}")
    raise RuntimeError(
        f"Failed to load '{file_path}'. Tried loaders {[l.__name__ for l in loaders]}:\n"
        + "\n".join(f"  - {item}" for item in errors)
    )


def _load_with_open3d(file_path: Path) -> Optional[np.ndarray]:
    try:
        import open3d as o3d  # type: ignore
    except ImportError:
        return None

    pcd = o3d.io.read_point_cloud(str(file_path))
    return np.asarray(pcd.points, dtype=np.float64)


def _load_with_plyfile(file_path: Path) -> Optional[np.ndarray]:
    try:
        from plyfile import PlyData  # type: ignore
    except ImportError:
        return None
    ply = PlyData.read(str(file_path))
    try:
        vertex_data = ply["vertex"].data
    except (KeyError, AttributeError):
        return None
    columns = []
    for axis in ("x", "y", "z"):
        if axis in vertex_data.dtype.names:
            columns.append(np.asarray(vertex_data[axis], dtype=np.float64))
        else:
            return None
    return np.column_stack(columns)


def _load_ascii_ply(file_path: Path) -> Optional[np.ndarray]:
    with file_path.open("r", encoding="utf-8") as fh:
        header = []
        line = fh.readline()
        if not line or not line.strip().startswith("ply"):
            raise ValueError("Not a PLY file")
        header.append(line)
        while True:
            line = fh.readline()
            if not line:
                raise ValueError("Unexpected end of header")
            header.append(line)
            if line.strip() == "end_header":
                break

        num_vertices = None
        property_names: list[str] = []
        for item in header:
            tokens = item.strip().split()
            if len(tokens) >= 3 and tokens[0] == "element" and tokens[1] == "vertex":
                num_vertices = int(tokens[2])
            if len(tokens) >= 3 and tokens[0] == "property" and tokens[-1] in {"x", "y", "z"}:
                property_names.append(tokens[-1])

        if num_vertices is None:
            raise ValueError("Could not determine number of vertices")
        if property_names != ["x", "y", "z"]:
            raise ValueError("ASCII fallback only supports x/y/z vertex properties")

        data = np.loadtxt(fh, dtype=np.float64, max_rows=num_vertices, usecols=(0, 1, 2))
        return np.asarray(data, dtype=np.float64)


def describe_points(points: np.ndarray) -> Dict[str, np.ndarray]:
    stats = {
        "count": int(points.shape[0]),
        "centroid": np.mean(points, axis=0),
        "min": np.min(points, axis=0),
        "max": np.max(points, axis=0),
        "std": np.std(points, axis=0),
    }
    stats["range"] = stats["max"] - stats["min"]
    return stats


def sample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= max_points:
        return points.copy()
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def nearest_neighbors(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(target)
        distances, indices = tree.query(source, k=1)
        return indices, distances
    except ImportError:
        # Naive fallback, O(n^2)
        diffs = source[:, None, :] - target[None, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        indices = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(distances.shape[0]), indices]
        return indices, min_distances


def run_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_iterations: int,
    tolerance: float,
    rng: np.random.Generator,  # Unused but kept for API compatibility
    device_preference: str = "auto",
) -> Dict[str, object]:
    if len(source_points) == 0 or len(target_points) == 0:
        raise ValueError("Cannot run ICP with empty point clouds")

    try:
        import torch
        from pytorch3d.ops import iterative_closest_point
        from pytorch3d.structures import Pointclouds
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch3D is required for ICP but is not installed. Install it or run with --skip-icp."
        ) from exc

    start_time = time.perf_counter()

    # Resolve device
    if device_preference == "cpu":
        device = torch.device("cpu")
    elif device_preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA for ICP, but CUDA is not available.")
        device = torch.device("cuda")
    else:  # auto
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_tensor = torch.from_numpy(source_points).float().to(device)
    target_tensor = torch.from_numpy(target_points).float().to(device)

    source_cloud = Pointclouds(points=[source_tensor])
    target_cloud = Pointclouds(points=[target_tensor])

    try:
        icp_result = iterative_closest_point(
            source_cloud,
            target_cloud,
            max_iterations=max_iterations,
            relative_rmse_threshold=tolerance,
        )
    except TypeError as exc:
        if "unexpected keyword argument 'relative_rmse_threshold'" not in str(exc):
            raise
        icp_result = iterative_closest_point(
            source_cloud,
            target_cloud,
            max_iterations=max_iterations,
        )

    def get_field(obj: object, name: str) -> Optional[object]:
        if obj is None:
            return None
        if isinstance(obj, dict):
            field = obj.get(name)
        elif hasattr(obj, name):
            field = getattr(obj, name)
        else:
            return None
        # If it's a tensor, detach and move to CPU for safe downstream numpy ops
        try:
            import torch  # type: ignore
            if isinstance(field, torch.Tensor):
                return field.detach().cpu()
        except Exception:
            pass
        return field

    if hasattr(icp_result, "_asdict"):
        result_dict = icp_result._asdict()
    elif isinstance(icp_result, dict):
        result_dict = icp_result
    else:
        result_dict = {}

    def tensor_to_numpy(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        if t is None:
            return None
        if not isinstance(t, torch.Tensor):
            return None
        arr = t.detach().cpu().numpy()
        if arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    rotation = tensor_to_numpy(
        get_field(result_dict, "R")
        or get_field(icp_result, "R")
        or get_field(result_dict, "rotation")
        or get_field(icp_result, "rotation")
    )
    translation = tensor_to_numpy(
        get_field(result_dict, "T")
        or get_field(icp_result, "T")
        or get_field(result_dict, "translation")
        or get_field(icp_result, "translation")
    )

    if rotation is None or translation is None:
        composite = get_field(result_dict, "RTs") or get_field(icp_result, "RTs")
        if composite is not None:
            composite_np = None
            # Handle several possibilities for composite
            try:
                import torch  # type: ignore
                if isinstance(composite, torch.Tensor):
                    composite_np = composite.detach().cpu().numpy()
                elif isinstance(composite, (list, tuple)) and len(composite) > 0:
                    first = composite[0]
                    if isinstance(first, torch.Tensor):
                        composite_np = first.detach().cpu().numpy()
                    else:
                        composite_np = np.asarray(composite)
                else:
                    composite_np = np.asarray(composite)
            except Exception as e:
                # Fall back by skipping composite if conversion fails
                composite_np = None

            if composite_np is not None:
                # Normalize to numpy array
                composite_np = np.asarray(composite_np)
                if composite_np.ndim >= 3:
                    mat = composite_np[0]
                else:
                    mat = composite_np

                # Accept (4,4), (3,4), or (3,3)
                if mat.shape == (4, 4):
                    rotation = mat[:3, :3]
                    translation = mat[:3, 3]
                elif mat.shape == (3, 4):
                    rotation = mat[:3, :3]
                    translation = mat[:3, 3]
                elif mat.shape == (3, 3):
                    rotation = mat
                    translation = np.zeros(3, dtype=np.float64)

    if rotation is None or translation is None:
        available_keys = list(result_dict.keys()) if result_dict else dir(icp_result)
        raise RuntimeError(
            "PyTorch3D ICP result did not include rotation/translation tensors. "
            f"Available fields: {available_keys}"
        )

    scale_tensor = (
        get_field(result_dict, "s")
        or get_field(icp_result, "s")
        or (
            torch.norm(rotation, dim=-1).mean() / math.sqrt(3.0)
            if isinstance(rotation, torch.Tensor)
            else None
        )
    )
    if isinstance(scale_tensor, torch.Tensor):
        scale = float(scale_tensor.detach().cpu().view(-1)[0].item())
    elif isinstance(scale_tensor, (float, int)):
        scale = float(scale_tensor)
    else:
        scale = 1.0

    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)

    if rotation.ndim == 1:
        rotation = rotation.reshape(3, 3)
    elif rotation.ndim > 2:
        # Best-effort squeeze for batched outputs
        rotation = rotation.reshape(3, 3)

    if not np.isclose(scale, 1.0):
        rotation = rotation * scale

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    aligned_points = None
    aligned_candidate = (
        get_field(result_dict, "X")
        or get_field(icp_result, "X")
        or get_field(result_dict, "aligned_source")
        or get_field(icp_result, "aligned_source")
        or get_field(result_dict, "transformed_source")
        or get_field(icp_result, "transformed_source")
    )
    if aligned_candidate is not None:
        if hasattr(aligned_candidate, "points_list"):
            aligned_list = aligned_candidate.points_list()
            if aligned_list:
                aligned_points = aligned_list[0].detach().cpu().numpy()
        elif isinstance(aligned_candidate, torch.Tensor):
            aligned_points = aligned_candidate.detach().cpu().numpy()
            if aligned_points.ndim >= 2 and aligned_points.shape[0] == 1:
                aligned_points = aligned_points.reshape(aligned_points.shape[1], -1)

    if aligned_points is None:
        aligned_points = (source_points @ rotation.T) + translation

    iterations = get_field(result_dict, "iters") or get_field(icp_result, "iters")
    if isinstance(iterations, torch.Tensor):
        iterations = int(iterations.detach().cpu().view(-1)[0].item())
    elif iterations is None:
        iterations = max_iterations
    else:
        iterations = int(iterations)

    rmse_tensor = get_field(result_dict, "rmse") or get_field(icp_result, "rmse")
    rmse_value = None
    if isinstance(rmse_tensor, torch.Tensor):
        rmse_value = float(rmse_tensor.detach().cpu().view(-1)[0].item())
    elif isinstance(rmse_tensor, (float, int)):
        rmse_value = float(rmse_tensor)

    converged_attr = get_field(result_dict, "converged") or get_field(icp_result, "converged")
    if isinstance(converged_attr, torch.Tensor):
        converged = bool(converged_attr.detach().cpu().view(-1)[0].item())
    elif isinstance(converged_attr, (bool, np.bool_)):
        converged = bool(converged_attr)
    else:
        converged = iterations < max_iterations

    residuals = compute_alignment_error(transform, source_points, target_points)
    mean_error = residuals["mean"]
    rmse = rmse_value if rmse_value is not None else residuals["rmse"]

    duration = time.perf_counter() - start_time

    return {
        "transform": transform,
        "iterations": iterations,
        "converged": converged,
        "mean_error": mean_error,
        "rmse": rmse,
        "max_error": residuals["max"],
        "rotation_angle_deg": rotation_angle(transform[:3, :3]),
        "translation_norm": float(np.linalg.norm(transform[:3, 3])),
        "aligned_points": aligned_points.astype(np.float64, copy=False),
        "duration_s": duration,
    }


def compute_alignment_error(transform: np.ndarray, source: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    if len(source) == 0 or len(target) == 0:
        return {"mean": math.inf, "rmse": math.inf, "max": math.inf}

    rotated = (source @ transform[:3, :3].T) + transform[:3, 3]
    indices, distances = nearest_neighbors(rotated, target)
    rms = float(np.sqrt(np.mean(distances**2)))
    return {"mean": float(np.mean(distances)), "rmse": rms, "max": float(np.max(distances))}


def rotation_angle(rotation_matrix: np.ndarray) -> float:
    trace = np.clip((np.trace(rotation_matrix) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(trace))


def format_vector(vector: Iterable[float]) -> str:
    return "[" + ", ".join(f"{value: .4f}" for value in vector) + "]"


def analyze_cloud(
    name: str,
    stats: Dict[str, np.ndarray],
    reference_stats: Optional[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    print(f"\n=== {name} ===")
    print(f"- points      : {stats['count']}")
    print(f"- centroid    : {format_vector(stats['centroid'])}")
    print(f"- min         : {format_vector(stats['min'])}")
    print(f"- max         : {format_vector(stats['max'])}")
    print(f"- range       : {format_vector(stats['range'])}")
    print(f"- std dev     : {format_vector(stats['std'])}")

    if reference_stats is not None:
        centroid_shift = stats["centroid"] - reference_stats["centroid"]
        range_delta = stats["range"] - reference_stats["range"]
        print(f"- centroid Δ  : {format_vector(centroid_shift)}")
        print(f"- range Δ     : {format_vector(range_delta)}")
    return stats


def preview_icp_alignment(records: list[Dict[str, object]], rng: np.random.Generator) -> None:
    try:
        import open3d as o3d  # type: ignore
    except ImportError:
        print("\n[preview] Skipped: open3d is not installed.")
        return

    selection = int(rng.integers(low=0, high=len(records)))
    record = records[selection]
    name = record["name"]
    sample_points = np.asarray(record["sample_points"], dtype=np.float64)
    reference_points = np.asarray(record["reference_points"], dtype=np.float64)
    transform = np.asarray(record["transform"], dtype=np.float64)
    aligned_points = record.get("aligned_points")
    if aligned_points is not None:
        transformed_points = np.asarray(aligned_points, dtype=np.float64)
    else:
        transformed_points = (sample_points @ transform[:3, :3].T) + transform[:3, 3]

    def make_cloud(points: np.ndarray, color: Tuple[float, float, float]) -> "o3d.geometry.PointCloud":
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        repeated = np.tile(np.asarray(color, dtype=np.float64), (points.shape[0], 1))
        cloud.colors = o3d.utility.Vector3dVector(repeated)
        return cloud

    before_color = (1.0, 0.0, 0.0)
    after_color = (0.0, 0.8, 0.0)
    reference_color = (0.0, 0.4, 1.0)

    before_cloud = make_cloud(sample_points, before_color)
    after_cloud = make_cloud(transformed_points, after_color)
    reference_cloud = make_cloud(reference_points, reference_color)

    print(f"\n[preview] Displaying ICP preview for '{name}'.")
    print("[preview] Colors -> before align: red, after align: green, reference: blue")
    print("[preview] Close the Open3D window to continue.")
    o3d.visualization.draw_geometries(
        [before_cloud, after_cloud, reference_cloud],
        window_name=f"ICP preview: {name}",
    )


def main() -> None:
    np.set_printoptions(precision=5, suppress=True)
    args = parse_args()
    result_path = args.result_path

    if not result_path.exists():
        raise FileNotFoundError(f"Result path '{result_path}' does not exist.")

    subfolders = sorted([item for item in result_path.iterdir() if item.is_dir()])
    if not subfolders:
        raise RuntimeError(f"No subdirectories found under '{result_path}'.")

    reference_folder = None
    if args.reference:
        for folder in subfolders:
            if folder.name == args.reference:
                reference_folder = folder
                break
        if reference_folder is None:
            raise ValueError(
                f"Reference folder '{args.reference}' not found. Available folders: "
                + ", ".join(folder.name for folder in subfolders)
            )
    else:
        reference_folder = subfolders[0]

    rng = np.random.default_rng(42)
    sample_cache: Dict[str, np.ndarray] = {}
    icp_records: list[Dict[str, object]] = []

    print(f"Result path : {result_path.resolve()}")
    print(f"Reference   : {reference_folder.name}")

    reference_file = find_ply_file(reference_folder)
    if reference_file is None:
        raise RuntimeError(f"No PLY file found in reference folder '{reference_folder}'.")

    reference_points = load_point_cloud(reference_file)
    reference_stats_full = describe_points(reference_points)
    analyze_cloud(reference_folder.name, reference_stats_full, None)
    sample_cache[reference_folder.name] = sample_points(reference_points, args.sample_size, rng)

    for folder in subfolders:
        if folder == reference_folder:
            continue

        ply_file = find_ply_file(folder)
        if ply_file is None:
            print(f"\n=== {folder.name} ===")
            print("- skipped: no PLY file found")
            continue

        points = load_point_cloud(ply_file)
        stats_full = describe_points(points)
        analyze_cloud(folder.name, stats_full, reference_stats_full)
        sampled_points = sample_points(points, args.sample_size, rng)
        sample_cache[folder.name] = sampled_points

        if args.skip_icp:
            continue

        try:
            icp_result = run_icp(
                sampled_points,
                sample_cache[reference_folder.name],
                max_iterations=args.max_iterations,
                tolerance=args.tolerance,
                rng=rng,
                device_preference=args.icp_device,
            )
            transform = icp_result["transform"]
            print("- ICP converged: {}".format("yes" if icp_result["converged"] else "no"))
            print(f"- ICP iterations : {icp_result['iterations']}")
            print(f"- ICP mean error : {icp_result['mean_error']:.6f}")
            print(f"- ICP RMSE       : {icp_result['rmse']:.6f}")
            print(f"- ICP max error  : {icp_result['max_error']:.6f}")
            print(f"- ICP rotation ° : {icp_result['rotation_angle_deg']:.4f}")
            print(f"- ICP translation: {icp_result['translation_norm']:.6f} (norm)")
            print(f"- ICP time       : {icp_result['duration_s']:.4f} s")
            print("- ICP transform :\n{}".format(np.array2string(transform, formatter={"float_kind": lambda x: f"{x: .6f}"})))
            icp_records.append(
                {
                    "name": folder.name,
                    "sample_points": sampled_points.copy(),
                    "reference_points": sample_cache[reference_folder.name].copy(),
                    "transform": transform.copy(),
                    "aligned_points": None
                    if icp_result["aligned_points"] is None
                    else np.asarray(icp_result["aligned_points"], dtype=np.float64).copy(),
                    "duration_s": icp_result["duration_s"],
                }
            )
        except Exception as exc:  # noqa: BLE001 - report failure but continue
            print(f"- ICP failed    : {exc}")

    if not args.skip_icp and icp_records:
        preview_icp_alignment(icp_records, rng)
    elif not args.skip_icp:
        print("\n[preview] Skipped: no successful ICP results to visualize.")

if __name__ == "__main__":
    main()
