import argparse
import math
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


def best_fit_transform(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)
    centered_source = source - centroid_source
    centered_target = target - centroid_target
    h = centered_source.T @ centered_target

    u, _, vh = np.linalg.svd(h)
    r = vh.T @ u.T
    if np.linalg.det(r) < 0:
        vh[-1, :] *= -1
        r = vh.T @ u.T
    t = centroid_target - centroid_source @ r.T
    return r, t


def run_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_iterations: int,
    tolerance: float,
    rng: np.random.Generator,
) -> Dict[str, object]:
    if len(source_points) == 0 or len(target_points) == 0:
        raise ValueError("Cannot run ICP with empty point clouds")

    source = source_points.copy()
    target = target_points.copy()

    transform = np.eye(4, dtype=np.float64)
    prev_error = None
    converged = False

    for iteration in range(1, max_iterations + 1):
        indices, distances = nearest_neighbors(source, target)
        matched_target = target[indices]
        rotation, translation = best_fit_transform(source, matched_target)

        transform_step = np.eye(4)
        transform_step[:3, :3] = rotation
        transform_step[:3, 3] = translation
        transform = transform_step @ transform

        source = (source @ rotation.T) + translation
        mean_error = float(np.mean(distances))
        if prev_error is not None and abs(prev_error - mean_error) < tolerance:
            converged = True
            break
        prev_error = mean_error

        jitter = 1e-6 * rng.standard_normal(size=source.shape)
        source += jitter

    residuals = compute_alignment_error(transform, source_points, target_points)
    return {
        "transform": transform,
        "iterations": iteration if converged else max_iterations,
        "converged": converged,
        "mean_error": residuals["mean"],
        "rmse": residuals["rmse"],
        "max_error": residuals["max"],
        "rotation_angle_deg": rotation_angle(transform[:3, :3]),
        "translation_norm": float(np.linalg.norm(transform[:3, 3])),
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
            )
            transform = icp_result["transform"]
            print("- ICP converged: {}".format("yes" if icp_result["converged"] else "no"))
            print(f"- ICP iterations : {icp_result['iterations']}")
            print(f"- ICP mean error : {icp_result['mean_error']:.6f}")
            print(f"- ICP RMSE       : {icp_result['rmse']:.6f}")
            print(f"- ICP max error  : {icp_result['max_error']:.6f}")
            print(f"- ICP rotation ° : {icp_result['rotation_angle_deg']:.4f}")
            print(f"- ICP translation: {icp_result['translation_norm']:.6f} (norm)")
            print("- ICP transform :\n{}".format(np.array2string(transform, formatter={"float_kind": lambda x: f"{x: .6f}"})))
            icp_records.append(
                {
                    "name": folder.name,
                    "sample_points": sampled_points.copy(),
                    "reference_points": sample_cache[reference_folder.name].copy(),
                    "transform": transform.copy(),
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
