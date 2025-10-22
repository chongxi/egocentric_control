"""
COLMAP Visualization with Open3D
Loads and visualizes PLY point cloud and camera poses from COLMAP sparse reconstruction
"""

import numpy as np
import open3d as o3d
import struct
import json
from pathlib import Path
from typing import Dict, Tuple


class ColmapDataLoader:
    """Load COLMAP binary format data"""

    @staticmethod
    def read_cameras_binary(path: str) -> Dict:
        """
        Read cameras.bin file

        Format:
        - num_cameras (uint64)
        - For each camera:
          - camera_id (uint32)
          - model_id (int32)
          - width (uint64)
          - height (uint64)
          - params (double array)
        """
        cameras = {}
        with open(path, "rb") as f:
            num_cameras = struct.unpack("<Q", f.read(8))[0]

            for _ in range(num_cameras):
                camera_id = struct.unpack("<I", f.read(4))[0]
                model_id = struct.unpack("<i", f.read(4))[0]
                width = struct.unpack("<Q", f.read(8))[0]
                height = struct.unpack("<Q", f.read(8))[0]

                # Read camera parameters (usually 4 for PINHOLE: fx, fy, cx, cy)
                # Model types: 0=SIMPLE_PINHOLE(3), 1=PINHOLE(4), 2=SIMPLE_RADIAL(4), etc.
                num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}
                params_count = num_params.get(model_id, 4)
                params = struct.unpack(f"<{params_count}d", f.read(8 * params_count))

                cameras[camera_id] = {
                    'id': camera_id,
                    'model': model_id,
                    'width': width,
                    'height': height,
                    'params': params
                }

        return cameras

    @staticmethod
    def read_images_binary(path: str) -> Dict:
        """
        Read images.bin file

        Format:
        - num_images (uint64)
        - For each image:
          - image_id (uint32)
          - qw, qx, qy, qz (double x4) - quaternion rotation
          - tx, ty, tz (double x3) - translation
          - camera_id (uint32)
          - name (string)
          - num_points2D (uint64)
          - points2D data
        """
        images = {}
        with open(path, "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]

            for _ in range(num_images):
                image_id = struct.unpack("<I", f.read(4))[0]

                # Quaternion (w, x, y, z)
                qvec = struct.unpack("<4d", f.read(32))

                # Translation
                tvec = struct.unpack("<3d", f.read(24))

                camera_id = struct.unpack("<I", f.read(4))[0]

                # Image name (null-terminated string)
                name = b''
                while True:
                    char = f.read(1)
                    if char == b'\x00':
                        break
                    name += char
                name = name.decode('utf-8')

                # Number of 2D points
                num_points2D = struct.unpack("<Q", f.read(8))[0]

                # Skip 2D points data (x, y, point3D_id for each point)
                f.read(num_points2D * 24)  # 8 + 8 + 8 bytes per point

                images[image_id] = {
                    'id': image_id,
                    'qvec': qvec,
                    'tvec': tvec,
                    'camera_id': camera_id,
                    'name': name
                }

        return images

    @staticmethod
    def qvec2rotmat(qvec: Tuple[float, float, float, float]) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        qw, qx, qy, qz = qvec

        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])

        return R


class ColmapVisualizer:
    """Visualize COLMAP reconstruction with Open3D"""

    def __init__(self, sparse_dir: str, config_path: str = None):
        self.sparse_dir = Path(sparse_dir)
        # Default config path is in the same directory as the script
        if config_path is None:
            config_path = Path(__file__).parent / "autoprojector_conf.json"
        self.config_path = config_path
        self.config = None
        self.cameras = {}
        self.images = {}
        self.point_cloud = None
        self.geometries = []
        self.plane_fitting_camera_indices = []  # Indices of cameras used for plane fitting
        self.camera_plane_params = None  # Store camera plane (normal, d)
        self.ground_plane_params = None  # Store ground plane (normal, d)

        # Load configuration
        self.load_config()

    def load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            print(f"Configuration loaded from {self.config_path}")
            print(f"  - RANSAC threshold: {self.config.get('RANSAC_threshold', 0.005)}")
            print(f"  - Max iterations: {self.config.get('max_iterations', 1000)}")
            print(f"  - Camera-ground distance: {self.config.get('camera_ground_distance', 1.6)}")
            print(f"  - Projection filter type: {self.config.get('projection_filter_type', 'camera')}")
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found, using defaults")
            self.config = {
                "RANSAC_threshold": 0.005,
                "max_iterations": 1000,
                "camera_ground_distance": 1.6,
                "projection_filter_type": "camera",
                "projection_filter_params": {
                    "camera": {"min_percent": 0.1, "max_percent": 1.0},
                    "ceiling": {"min_percent": 0.1, "max_percent": 0.8}
                }
            }

    def load_data(self):
        """Load all COLMAP data"""
        # Load cameras
        cameras_path = self.sparse_dir / "cameras.bin"
        if cameras_path.exists():
            self.cameras = ColmapDataLoader.read_cameras_binary(str(cameras_path))

        # Load images (camera poses)
        images_path = self.sparse_dir / "images.bin"
        if images_path.exists():
            self.images = ColmapDataLoader.read_images_binary(str(images_path))

        # Load point cloud
        ply_path = self.sparse_dir / "points.ply"
        if ply_path.exists():
            self.point_cloud = o3d.io.read_point_cloud(str(ply_path))

        # Print summary
        print("=" * 60)
        print("Dataset Summary:")
        print("-" * 60)

        # Point cloud count
        num_points = len(self.point_cloud.points) if self.point_cloud else 0
        print(f"Total points: {num_points:,}")

        # Camera poses count
        print(f"Camera poses: {len(self.images)}")

        # XYZ ranges
        if num_points > 0:
            points = np.asarray(self.point_cloud.points)
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            z_min, z_max = points[:, 2].min(), points[:, 2].max()

            print(f"X range: [{x_min:.4f}, {x_max:.4f}]")
            print(f"Y range: [{y_min:.4f}, {y_max:.4f}]")
            print(f"Z range: [{z_min:.4f}, {z_max:.4f}]")

        print("=" * 60)

    def create_camera_frustum(self, pose: Dict, camera: Dict, scale: float = 0.1, color: Tuple[float, float, float] = (1, 0, 0)) -> o3d.geometry.LineSet:
        """
        Create a camera frustum visualization

        Args:
            pose: Image pose dict with qvec and tvec
            camera: Camera dict with intrinsics
            scale: Size of the frustum
            color: RGB color tuple
        """
        # Get camera intrinsics
        width = camera['width']
        height = camera['height']

        # Create frustum in camera space
        aspect_ratio = width / height
        frustum_height = scale
        frustum_width = scale * aspect_ratio

        # Camera frustum corners in camera space
        points = np.array([
            [0, 0, 0],  # Camera center
            [-frustum_width/2, -frustum_height/2, scale],  # Top-left
            [frustum_width/2, -frustum_height/2, scale],   # Top-right
            [frustum_width/2, frustum_height/2, scale],    # Bottom-right
            [-frustum_width/2, frustum_height/2, scale],   # Bottom-left
        ])

        # Transform to world space
        R = ColmapDataLoader.qvec2rotmat(pose['qvec'])
        t = np.array(pose['tvec'])

        # COLMAP uses camera-to-world transformation
        # World point = R^T * (camera_point) + C
        # where C = -R^T * t
        C = -R.T @ t
        points_world = (R.T @ points.T).T + C

        # Define frustum lines
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from camera center to corners
            [1, 2], [2, 3], [3, 4], [4, 1],  # Rectangle at image plane
        ]

        # Create LineSet
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points_world)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

        return line_set

    def get_camera_center(self, pose: Dict) -> np.ndarray:
        """
        Get camera center in world coordinates

        Args:
            pose: Image pose dict with qvec and tvec

        Returns:
            Camera center as 3D numpy array
        """
        R = ColmapDataLoader.qvec2rotmat(pose['qvec'])
        t = np.array(pose['tvec'])

        # Camera center: C = -R^T * t
        C = -R.T @ t
        return C

    def fit_plane_least_squares(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Fit a plane to multiple points using least-squares method
        Plane equation: ax + by + cz + d = 0

        Args:
            points: Nx3 array of points

        Returns:
            normal: Normal vector [a, b, c] (unit vector)
            d: Plane constant
        """
        # Calculate centroid
        centroid = points.mean(axis=0)

        # Center the points
        centered = points - centroid

        # Perform SVD
        # The plane normal is the singular vector corresponding to the smallest singular value
        _, _, vh = np.linalg.svd(centered)

        # Normal vector is the last row of V^T (smallest singular value)
        normal = vh[2, :]

        # Normalize (should already be normalized from SVD, but just to be safe)
        normal = normal / np.linalg.norm(normal)

        # Calculate d using the centroid
        d = -np.dot(normal, centroid)

        return normal, d

    def fit_plane_ransac(self, points: np.ndarray, distance_threshold: float = 0.005, max_iterations: int = 1000) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Fit a plane to point cloud using RANSAC to find the dominant plane (e.g., ground)

        Args:
            points: Nx3 array of points
            distance_threshold: Maximum distance for a point to be considered an inlier (default: 0.005)
            max_iterations: Maximum number of RANSAC iterations

        Returns:
            normal: Normal vector [a, b, c] (unit vector)
            d: Plane constant
            inlier_mask: Boolean mask indicating which points are inliers
        """
        best_inliers = None
        best_normal = None
        best_d = None
        best_inlier_count = 0

        n_points = len(points)

        print(f"Running RANSAC plane fitting on {n_points:,} points...")
        print(f"Distance threshold: {distance_threshold}, Max iterations: {max_iterations}")

        for _ in range(max_iterations):
            # Randomly sample 3 points
            sample_indices = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[sample_indices]

            # Calculate plane from 3 points
            v1 = p2 - p1
            v2 = p3 - p1

            # Normal vector
            normal = np.cross(v1, v2)
            normal_length = np.linalg.norm(normal)

            # Skip if points are collinear
            if normal_length < 1e-6:
                continue

            normal = normal / normal_length
            d = -np.dot(normal, p1)

            # Calculate distances of all points to this plane
            distances = np.abs(np.dot(points, normal) + d)

            # Count inliers
            inliers = distances < distance_threshold
            inlier_count = np.sum(inliers)

            # Update best plane if this one has more inliers
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
                best_normal = normal
                best_d = d

        inlier_percentage = (best_inlier_count / n_points) * 100
        print(f"RANSAC found plane with {best_inlier_count:,} inliers ({inlier_percentage:.2f}% of points)")

        # Refine plane using all inliers with least-squares
        if best_inlier_count > 3:
            inlier_points = points[best_inliers]
            best_normal, best_d = self.fit_plane_least_squares(inlier_points)
            print(f"Refined plane using least-squares on inliers")

        return best_normal, best_d, best_inliers

    def project_point_to_plane(self, point: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
        """
        Project a point onto a plane

        Args:
            point: 3D point to project
            normal: Plane normal vector [a, b, c]
            d: Plane constant

        Returns:
            Projected point on the plane
        """
        # Distance from point to plane
        distance = np.dot(normal, point) + d

        # Project point onto plane
        projected = point - distance * normal

        return projected

    def get_plane_local_coords(self, points: np.ndarray, normal: np.ndarray, d: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a local 2D coordinate system on the plane and express points in it

        Args:
            points: Nx3 array of 3D points
            normal: Plane normal vector
            d: Plane constant

        Returns:
            coords_2d: Nx2 array of points in local plane coordinates
            basis_u: First basis vector of the plane
            basis_v: Second basis vector of the plane
        """
        # Project all points onto the plane first
        projected_points = np.array([
            self.project_point_to_plane(point, normal, d)
            for point in points
        ])

        # Create an orthonormal basis on the plane
        # Choose an arbitrary vector not parallel to the normal
        if abs(normal[0]) < 0.9:
            arbitrary = np.array([1.0, 0.0, 0.0])
        else:
            arbitrary = np.array([0.0, 1.0, 0.0])

        # First basis vector: perpendicular to normal
        basis_u = arbitrary - np.dot(arbitrary, normal) * normal
        basis_u = basis_u / np.linalg.norm(basis_u)

        # Second basis vector: perpendicular to both normal and basis_u
        basis_v = np.cross(normal, basis_u)
        basis_v = basis_v / np.linalg.norm(basis_v)

        # Find a reference point on the plane (centroid of projected points)
        origin = projected_points.mean(axis=0)

        # Express each projected point in the local 2D coordinate system
        coords_2d = []
        for point in projected_points:
            relative = point - origin
            u = np.dot(relative, basis_u)
            v = np.dot(relative, basis_v)
            coords_2d.append([u, v])

        return np.array(coords_2d), basis_u, basis_v, origin

    def calculate_plane_distance(self) -> float:
        """
        Calculate the distance between camera plane and ground plane

        Returns:
            Distance between the two planes (perpendicular distance)
        """
        if self.camera_plane_params is None or self.ground_plane_params is None:
            return None

        normal1, d1 = self.camera_plane_params
        normal2, d2 = self.ground_plane_params

        # Check if planes are parallel (dot product of normals close to 1 or -1)
        dot_product = abs(np.dot(normal1, normal2))

        if dot_product > 0.99:  # Planes are nearly parallel
            # Distance between parallel planes: |d1 - d2| / |normal|
            # Since normals are unit vectors, |normal| = 1
            distance = abs(d1 - d2)
            print(f"\nPlanes are parallel (dot product: {dot_product:.4f})")
        else:
            # Planes intersect - calculate distance at a reference point
            # Use a point on plane 1 and measure distance to plane 2
            # Find a point on plane 1: we can use any point, let's use origin if d1=0, else construct one
            if abs(normal1[2]) > 0.1:  # Normal has significant Z component
                # Point on plane 1: set x=0, y=0, solve for z
                point_on_plane1 = np.array([0, 0, -d1 / normal1[2]])
            elif abs(normal1[1]) > 0.1:  # Normal has significant Y component
                point_on_plane1 = np.array([0, -d1 / normal1[1], 0])
            else:  # Normal has significant X component
                point_on_plane1 = np.array([-d1 / normal1[0], 0, 0])

            # Distance from this point to plane 2
            distance = abs(np.dot(normal2, point_on_plane1) + d2)
            print(f"\nPlanes intersect at angle (dot product: {dot_product:.4f})")
            print(f"Distance measured at reference point on camera plane")

        return distance

    def create_surface_from_pointcloud_bounds(self, color: Tuple[float, float, float] = (0.0, 0.8, 0.2), percentage: float = 0.8) -> o3d.geometry.TriangleMesh:
        """
        Create a rectangular surface mesh based on the point cloud's X,Y extent,
        lying on the plane fitted to a percentage of camera positions using least-squares

        Args:
            color: RGB color for the surface
            percentage: Percentage of cameras to use for plane fitting (default: 0.8 = 80%)

        Returns:
            TriangleMesh representing the rectangular surface
        """
        if self.point_cloud is None or len(self.point_cloud.points) == 0:
            print("Warning: No point cloud available to create surface")
            return None

        if len(self.images) < 3:
            print("Warning: Need at least 3 cameras for plane fitting")
            return None

        # Get all point cloud points
        points = np.asarray(self.point_cloud.points)

        # Calculate how many cameras to use (80% of total)
        total_cameras = len(self.images)
        num_cameras_to_use = max(3, int(total_cameras * percentage))  # At least 3 cameras

        print(f"Selecting {num_cameras_to_use} out of {total_cameras} cameras ({percentage*100:.0f}%) - excluding outliers")

        # Step 1: Get all camera centers
        all_poses = list(self.images.values())
        all_camera_centers = []
        for pose in all_poses:
            center = self.get_camera_center(pose)
            all_camera_centers.append(center)
        all_camera_centers = np.array(all_camera_centers)

        # Step 2: Fit initial plane using all cameras
        initial_normal, initial_d = self.fit_plane_least_squares(all_camera_centers)

        # Step 3: Calculate distance of each camera to the initial plane
        distances = np.abs(np.dot(all_camera_centers, initial_normal) + initial_d)

        # Step 4: Select the cameras with smallest distances (closest to plane)
        # Sort by distance and take the closest 80%
        sorted_indices = np.argsort(distances)
        selected_indices = sorted_indices[:num_cameras_to_use]

        # Step 5: Get the selected camera centers
        camera_centers = all_camera_centers[selected_indices]

        # Step 6: Refit plane using only the selected cameras (non-outliers)
        normal, d = self.fit_plane_least_squares(camera_centers)

        # Store the plane parameters
        self.camera_plane_params = (normal, d)

        # Store the indices of cameras used for plane fitting
        self.plane_fitting_camera_indices = sorted(selected_indices.tolist())

        print(f"Selected cameras (excluding outliers): {len(selected_indices)} cameras")

        # Get local plane coordinates to find true surface extent
        print("Projecting all points onto camera plane to determine surface size...")
        coords_2d, basis_u, basis_v, origin = self.get_plane_local_coords(points, normal, d)

        # Find bounding box in local 2D coordinates
        u_min, u_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
        v_min, v_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()

        surface_width = u_max - u_min
        surface_length = v_max - v_min

        print(f"Camera surface width: {surface_width:.4f}, length: {surface_length:.4f} (in plane local coords)")

        # Create rectangle corners in local 2D coordinates, then convert to 3D
        corners_2d = np.array([
            [u_min, v_min],  # Bottom-left
            [u_max, v_min],  # Bottom-right
            [u_max, v_max],  # Top-right
            [u_min, v_max],  # Top-left
        ])

        # Convert corners from local 2D to global 3D coordinates
        vertices = np.array([
            origin + u * basis_u + v * basis_v
            for u, v in corners_2d
        ])

        # Define the two triangles that make up the rectangle
        # Each triangle defined in both directions for double-sided rendering
        triangles = np.array([
            # First triangle (bottom-left, bottom-right, top-right)
            [0, 1, 2],
            [0, 2, 1],
            # Second triangle (bottom-left, top-right, top-left)
            [0, 2, 3],
            [0, 3, 2],
        ])

        # Create mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

        # Compute normals for proper lighting
        mesh.compute_vertex_normals()

        # Set color
        mesh.paint_uniform_color(color)

        return mesh

    def create_ground_plane_surface(self, color: Tuple[float, float, float] = (0.8, 0.4, 0.2), distance_threshold: float = 0.005) -> o3d.geometry.TriangleMesh:
        """
        Create a ground plane surface using RANSAC on the point cloud

        Args:
            color: RGB color for the ground plane surface
            distance_threshold: RANSAC distance threshold for inliers

        Returns:
            TriangleMesh representing the ground plane surface
        """
        if self.point_cloud is None or len(self.point_cloud.points) == 0:
            print("Warning: No point cloud available for ground plane detection")
            return None

        # Get all points
        points = np.asarray(self.point_cloud.points)

        # Fit plane using RANSAC
        normal, d, _ = self.fit_plane_ransac(points, distance_threshold=distance_threshold)

        if normal is None:
            print("Warning: RANSAC failed to find a plane")
            return None

        # Store the ground plane parameters
        self.ground_plane_params = (normal, d)

        # Get local plane coordinates to find true surface extent
        print("Projecting all points onto ground plane to determine surface size...")
        coords_2d, basis_u, basis_v, origin = self.get_plane_local_coords(points, normal, d)

        # Find bounding box in local 2D coordinates
        u_min, u_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
        v_min, v_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()

        ground_width = u_max - u_min
        ground_length = v_max - v_min

        print(f"Ground plane width: {ground_width:.4f}, length: {ground_length:.4f} (in plane local coords)")

        # Create rectangle corners in local 2D coordinates, then convert to 3D
        corners_2d = np.array([
            [u_min, v_min],  # Bottom-left
            [u_max, v_min],  # Bottom-right
            [u_max, v_max],  # Top-right
            [u_min, v_max],  # Top-left
        ])

        # Convert corners from local 2D to global 3D coordinates
        vertices = np.array([
            origin + u * basis_u + v * basis_v
            for u, v in corners_2d
        ])

        # Define triangles for the rectangle
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 1],
            [0, 2, 3],
            [0, 3, 2],
        ])

        # Create mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)

        return mesh

    def create_camera_spheres(self, sphere_radius: float = 0.01, show_every_nth: int = 1) -> list:
        """
        Create spheres at all visible camera positions to mark them
        Cameras used for plane fitting get green color, others get yellow

        Args:
            sphere_radius: Radius of the spheres
            show_every_nth: Show sphere for every nth camera (same as frustum display)

        Returns:
            List of sphere meshes
        """
        spheres = []

        # Color coding:
        # Green: Cameras used for plane fitting
        # Yellow: Other cameras
        plane_fitting_color = (0.0, 1.0, 0.0)  # Green - used for plane fitting
        other_color = (1.0, 1.0, 0.0)  # Yellow - not used for plane fitting

        for idx, pose in enumerate(self.images.values()):
            # Only create sphere for cameras matching the show_every_nth pattern
            if idx % show_every_nth != 0:
                continue

            center = self.get_camera_center(pose)

            # Create sphere
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sphere.translate(center)

            # Choose color based on whether it's used for plane fitting
            if idx in self.plane_fitting_camera_indices:
                color = plane_fitting_color
            else:
                color = other_color

            sphere.paint_uniform_color(color)
            sphere.compute_vertex_normals()

            spheres.append(sphere)

        return spheres

    def create_visualization(self, camera_scale: float = 0.2, show_every_nth: int = 1, show_surface: bool = True, show_camera_spheres: bool = True, show_ground_plane: bool = True):
        """
        Create visualization geometries

        Args:
            camera_scale: Scale of camera frustums
            show_every_nth: Show every nth camera (to reduce clutter)
            show_surface: Whether to show the surface fitted to 80% of cameras
            show_camera_spheres: Whether to show spheres at all camera positions
            show_ground_plane: Whether to show the RANSAC-fitted ground plane
        """
        self.geometries = []

        # Add point cloud
        if self.point_cloud is not None:
            self.geometries.append(self.point_cloud)

        # Add spheres at all camera positions (matching show_every_nth)
        if show_camera_spheres and len(self.images) > 0:
            spheres = self.create_camera_spheres(sphere_radius=0.01, show_every_nth=show_every_nth)
            for sphere in spheres:
                self.geometries.append(sphere)

        # Add camera-based surface (green)
        if show_surface and self.point_cloud is not None:
            surface = self.create_surface_from_pointcloud_bounds(color=(0.0, 0.8, 0.3))
            if surface is not None:
                self.geometries.append(surface)

        # Add RANSAC ground plane (orange/brown)
        if show_ground_plane and self.point_cloud is not None:
            ground_plane = self.create_ground_plane_surface(color=(0.8, 0.4, 0.2), distance_threshold=0.005)
            if ground_plane is not None:
                self.geometries.append(ground_plane)

        # Calculate and print distance between the two surfaces
        if show_surface and show_ground_plane:
            distance = self.calculate_plane_distance()
            if distance is not None:
                print(f"Distance between camera plane and ground plane: {distance:.4f} units")
                print("=" * 60)

        # Add camera frustums
        camera_count = 0
        for idx, img_data in enumerate(self.images.values()):
            if idx % show_every_nth != 0:
                continue

            camera_id = img_data['camera_id']
            if camera_id in self.cameras:
                # Use different colors for different cameras
                color = (
                    (camera_count % 10) / 10.0,
                    0.5,
                    1.0 - (camera_count % 10) / 10.0
                )

                frustum = self.create_camera_frustum(
                    img_data,
                    self.cameras[camera_id],
                    scale=camera_scale,
                    color=color
                )
                self.geometries.append(frustum)
                camera_count += 1

        # Add coordinate frame at origin
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        self.geometries.append(coord_frame)

    def visualize(self, camera_scale: float = 0.2, show_every_nth: int = 1, show_surface: bool = True, show_camera_spheres: bool = True, show_ground_plane: bool = True):
        """
        Visualize the COLMAP reconstruction

        Args:
            camera_scale: Scale of camera frustums
            show_every_nth: Show every nth camera (1 = show all, 5 = show every 5th)
            show_surface: Whether to show the surface fitted to 80% of cameras
            show_camera_spheres: Whether to show spheres at all camera positions
            show_ground_plane: Whether to show the RANSAC-fitted ground plane
        """
        self.load_data()
        self.create_visualization(camera_scale, show_every_nth, show_surface, show_camera_spheres, show_ground_plane)

        o3d.visualization.draw_geometries(
            self.geometries,
            window_name="COLMAP Reconstruction Viewer",
            width=1280,
            height=720,
            left=50,
            top=50
        )

    def filter_points_by_height(self, points: np.ndarray) -> np.ndarray:
        """
        Filter points based on height between camera plane and ground plane

        Args:
            points: Nx3 array of 3D points

        Returns:
            Boolean mask indicating which points pass the filter
        """
        if self.camera_plane_params is None or self.ground_plane_params is None:
            print("Warning: Planes not fitted yet. Returning all points.")
            return np.ones(len(points), dtype=bool)

        camera_normal, camera_d = self.camera_plane_params
        ground_normal, ground_d = self.ground_plane_params

        # Get filter parameters from config
        filter_type = self.config.get('projection_filter_type', 'camera')
        filter_params = self.config.get('projection_filter_params', {}).get(filter_type, {})
        min_percent = filter_params.get('min_percent', 0.1)
        max_percent = filter_params.get('max_percent', 1.0)

        print(f"\nFiltering points using '{filter_type}' mode:")
        print(f"  - Height range: {min_percent*100:.1f}% to {max_percent*100:.1f}%")

        if filter_type == 'camera':
            # Camera mode: filter points between camera plane and ground plane
            # Calculate signed distance from each point to both planes
            dist_to_ground = np.dot(points, ground_normal) + ground_d
            dist_to_camera = np.dot(points, camera_normal) + camera_d

            # Total distance between planes (use absolute values)
            # We want points that are between the two planes
            plane_distance = abs(np.mean(dist_to_camera) - np.mean(dist_to_ground))

            # For each point, calculate its relative position between ground and camera
            # 0 = at ground, 1 = at camera plane
            # Use distance from ground as reference
            relative_height = np.abs(dist_to_ground) / plane_distance

            # Filter based on percentage range
            min_height = min_percent
            max_height = max_percent

            mask = (relative_height >= min_height) & (relative_height <= max_height)

        elif filter_type == 'ceiling':
            # Ceiling mode: filter points from ground upward
            # Calculate distance from each point to ground plane
            dist_to_ground = np.abs(np.dot(points, ground_normal) + ground_d)

            # Get max distance to define ceiling
            max_distance = np.max(dist_to_ground)

            # Calculate relative height from ground (0 = ground, 1 = highest point)
            relative_height = dist_to_ground / max_distance

            # Filter based on percentage range
            min_height = min_percent
            max_height = max_percent

            mask = (relative_height >= min_height) & (relative_height <= max_height)
        else:
            print(f"Warning: Unknown filter type '{filter_type}', returning all points")
            mask = np.ones(len(points), dtype=bool)

        filtered_count = np.sum(mask)
        total_count = len(points)
        print(f"  - Filtered points: {filtered_count:,} / {total_count:,} ({filtered_count/total_count*100:.1f}%)")

        return mask

    def project_points_to_2d_grid(self, points: np.ndarray, resolution: float = 0.05) -> Tuple[np.ndarray, float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Project filtered 3D points onto ground plane and create 2D occupancy grid

        Args:
            points: Nx3 array of 3D points to project
            resolution: Grid resolution in meters (default: 0.05m = 5cm per pixel)

        Returns:
            grid: 2D occupancy grid (0 = free, 100 = occupied)
            x_min, x_max, y_min, y_max: Bounds of the grid in world coordinates
            basis_u, basis_v, origin: Plane coordinate frame for 3D-2D transformation
        """
        if len(points) == 0:
            print("Warning: No points to project")
            return None, 0, 0, 0, 0, None, None, None

        ground_normal, ground_d = self.ground_plane_params

        # Project all points onto ground plane
        projected_points = np.array([
            self.project_point_to_plane(point, ground_normal, ground_d)
            for point in points
        ])

        # Get 2D coordinates on the ground plane
        coords_2d, basis_u, basis_v, origin = self.get_plane_local_coords(projected_points, ground_normal, ground_d)

        # Find bounds
        x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
        y_min, y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()

        print(f"\nProjecting points to 2D grid:")
        print(f"  - Resolution: {resolution} m/pixel ({resolution*100:.1f} cm/pixel)")
        print(f"  - X range: [{x_min:.3f}, {x_max:.3f}] ({x_max - x_min:.3f} m)")
        print(f"  - Y range: [{y_min:.3f}, {y_max:.3f}] ({y_max - y_min:.3f} m)")

        # Create grid
        grid_width = int(np.ceil((x_max - x_min) / resolution))
        grid_height = int(np.ceil((y_max - y_min) / resolution))

        print(f"  - Grid size: {grid_width} x {grid_height} pixels")

        # Initialize grid with 0 (free space)
        grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

        # Map each point to grid cell and mark as occupied (100)
        for x, y in coords_2d:
            grid_x = int((x - x_min) / resolution)
            grid_y = int((y - y_min) / resolution)

            # Clip to grid bounds
            grid_x = np.clip(grid_x, 0, grid_width - 1)
            grid_y = np.clip(grid_y, 0, grid_height - 1)

            grid[grid_y, grid_x] = 100  # Mark as occupied

        occupied_cells = np.sum(grid == 100)
        total_cells = grid_width * grid_height
        print(f"  - Occupied cells: {occupied_cells:,} / {total_cells:,} ({occupied_cells/total_cells*100:.2f}%)")

        return grid, x_min, x_max, y_min, y_max, basis_u, basis_v, origin

    def save_pgm(self, grid: np.ndarray, output_path: str, metadata: dict = None):
        """
        Save occupancy grid as PGM (Portable Gray Map) file

        Args:
            grid: 2D occupancy grid (values 0-255)
            output_path: Output file path (.pgm)
            metadata: Optional metadata dictionary to save alongside
        """
        if grid is None:
            print("Warning: No grid to save")
            return

        height, width = grid.shape

        # Save PGM file (binary format)
        with open(output_path, 'wb') as f:
            # PGM header
            f.write(b'P5\n')  # Magic number for binary PGM
            f.write(f'{width} {height}\n'.encode())
            f.write(b'255\n')  # Max gray value

            # Write pixel data (row by row)
            grid.tofile(f)

        print(f"\nOccupancy map saved to: {output_path}")
        print(f"  - Format: PGM (P5 binary)")
        print(f"  - Size: {width} x {height} pixels")

        # Optionally save metadata as YAML (ROS convention)
        if metadata:
            yaml_path = output_path.replace('.pgm', '.yaml')
            with open(yaml_path, 'w') as f:
                f.write(f"image: {Path(output_path).name}\n")
                f.write(f"resolution: {metadata.get('resolution', 0.05)}\n")
                f.write(f"origin: [{metadata.get('origin_x', 0)}, {metadata.get('origin_y', 0)}, 0.0]\n")
                f.write("occupied_thresh: 0.65\n")
                f.write("free_thresh: 0.196\n")
                f.write("negate: 0\n")

                # Add calibration info as comments
                if 'scale_factor' in metadata:
                    f.write(f"\n# Calibration Information\n")
                    f.write(f"# scale_factor: {metadata['scale_factor']:.6f} m/unit\n")
                    f.write(f"# measured_distance: {metadata['measured_distance']:.6f} units\n")
                    f.write(f"# real_distance: {metadata['real_distance']:.2f} meters\n")

            print(f"  - Metadata saved to: {yaml_path}")
            print(f"  - Real-world resolution: {metadata.get('resolution', 0.05)} m/pixel")

            # Save detailed transformation parameters as NPZ for 3D-2D mapping
            npz_path = output_path.replace('.pgm', '_transform.npz')
            np.savez(
                npz_path,
                # Ground plane parameters
                ground_plane_normal=metadata.get('ground_plane_normal'),
                ground_plane_d=metadata.get('ground_plane_d'),
                # Plane coordinate frame (for 3D to 2D transformation)
                plane_basis_u=metadata.get('plane_basis_u'),
                plane_basis_v=metadata.get('plane_basis_v'),
                plane_origin=metadata.get('plane_origin'),
                # Grid bounds in local plane coordinates (measured units)
                grid_x_min=metadata.get('grid_x_min'),
                grid_y_min=metadata.get('grid_y_min'),
                grid_x_max=metadata.get('grid_x_max'),
                grid_y_max=metadata.get('grid_y_max'),
                # Resolution and scale
                resolution_measured_units=metadata.get('resolution_measured_units'),
                resolution_meters=metadata.get('resolution', 0.05),
                scale_factor=metadata.get('scale_factor'),
                # Grid dimensions
                grid_width=metadata.get('grid_width'),
                grid_height=metadata.get('grid_height'),
            )
            print(f"  - Transformation parameters saved to: {npz_path}")
            print(f"    (Use this for 3D-2D coordinate mapping)")

    def generate_occupancy_map(self, output_path: str = "router/occupancy_map.pgm"):
        """
        Generate and save occupancy map from point cloud
        Uses camera_ground_distance from config to calculate accurate resolution

        Args:
            output_path: Output PGM file path
        """
        print("\n" + "="*60)
        print("Generating Occupancy Map")
        print("="*60)

        # Ensure data and planes are loaded
        if self.point_cloud is None:
            print("Error: No point cloud loaded")
            return

        if self.camera_plane_params is None or self.ground_plane_params is None:
            print("Error: Planes not fitted. Run visualization first.")
            return

        # Calculate measured distance between planes
        measured_distance = self.calculate_plane_distance()
        real_distance = self.config.get('camera_ground_distance', 1.6)

        # Calculate scale factor: real_world_units / measured_units
        scale_factor = real_distance / measured_distance

        print(f"\nScale Calibration:")
        print(f"  - Measured plane distance: {measured_distance:.4f} units")
        print(f"  - Real-world distance: {real_distance} meters")
        print(f"  - Scale factor: {scale_factor:.4f} m/unit")

        # Calculate resolution in measured units (5cm in real world)
        target_resolution_meters = 0.05  # 5cm per pixel
        resolution_measured_units = target_resolution_meters / scale_factor

        print(f"  - Target resolution: {target_resolution_meters} m/pixel ({target_resolution_meters*100:.1f} cm/pixel)")
        print(f"  - Resolution in measured units: {resolution_measured_units:.6f} units/pixel")

        # Get all points
        points = np.asarray(self.point_cloud.points)

        # Filter points by height
        mask = self.filter_points_by_height(points)
        filtered_points = points[mask]

        # Project to 2D grid using measured units
        grid, x_min, x_max, y_min, y_max, basis_u, basis_v, origin = self.project_points_to_2d_grid(filtered_points, resolution_measured_units)

        # Get ground plane parameters
        ground_normal, ground_d = self.ground_plane_params

        # Save PGM with real-world metadata and 3D-2D transformation parameters
        metadata = {
            # ROS standard parameters
            'resolution': target_resolution_meters,  # Real-world resolution
            'origin_x': x_min * scale_factor,  # Real-world coordinates
            'origin_y': y_min * scale_factor,
            'scale_factor': scale_factor,
            'measured_distance': measured_distance,
            'real_distance': real_distance,
            # 3D-2D transformation parameters
            'ground_plane_normal': ground_normal,
            'ground_plane_d': ground_d,
            'plane_basis_u': basis_u,
            'plane_basis_v': basis_v,
            'plane_origin': origin,
            'grid_x_min': x_min,  # In measured units
            'grid_y_min': y_min,
            'grid_x_max': x_max,
            'grid_y_max': y_max,
            'resolution_measured_units': resolution_measured_units,
            'grid_width': grid.shape[1],
            'grid_height': grid.shape[0],
        }
        self.save_pgm(grid, output_path, metadata)

        print("="*60)

        return grid

    def visualize_occupancy_map(self, pgm_path: str):
        """
        Visualize the generated occupancy map

        Args:
            pgm_path: Path to the PGM file to visualize
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("Warning: matplotlib not available, skipping visualization")
            print("Install with: pip install matplotlib")
            return

        # Read PGM file
        try:
            with open(pgm_path, 'rb') as f:
                # Read header
                magic = f.readline().strip()
                if magic != b'P5':
                    print(f"Error: Not a valid binary PGM file (magic: {magic})")
                    return

                # Skip comments and read dimensions
                line = f.readline()
                while line.startswith(b'#'):
                    line = f.readline()

                width, height = map(int, line.split())
                _ = f.readline()  # Skip max_val line

                # Read pixel data
                grid = np.frombuffer(f.read(), dtype=np.uint8).reshape((height, width))

        except Exception as e:
            print(f"Error reading PGM file: {e}")
            return

        # Read metadata if available
        yaml_path = pgm_path.replace('.pgm', '.yaml')
        resolution = 0.05
        origin_x = 0
        origin_y = 0

        try:
            with open(yaml_path, 'r') as f:
                for line in f:
                    if line.startswith('resolution:'):
                        resolution = float(line.split(':')[1].strip())
                    elif line.startswith('origin:'):
                        # Parse [x, y, z]
                        coords = line.split('[')[1].split(']')[0]
                        origin_x, origin_y = map(float, coords.split(',')[:2])
        except FileNotFoundError:
            print(f"Warning: Metadata file {yaml_path} not found, using defaults")

        # Calculate real-world dimensions
        map_width_m = width * resolution
        map_height_m = height * resolution

        # Create visualization
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Left plot: Occupancy map
        # Show map with origin at bottom-left (flip vertically)
        im1 = ax1.imshow(grid, cmap='gray_r', origin='lower', extent=[
            origin_x,
            origin_x + map_width_m,
            origin_y,
            origin_y + map_height_m
        ])

        ax1.set_title(f'Occupancy Map\n{width}x{height} pixels @ {resolution*100:.1f}cm/pixel', fontsize=14, fontweight='bold')
        ax1.set_xlabel(f'X (meters)\nMap width: {map_width_m:.2f}m', fontsize=11)
        ax1.set_ylabel(f'Y (meters)\nMap height: {map_height_m:.2f}m', fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_aspect('equal')

        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Occupancy (0=Free, 100=Occupied)', fontsize=10)

        # Add legend
        free_patch = mpatches.Patch(color='white', label='Free (0)')
        occupied_patch = mpatches.Patch(color='black', label='Occupied (100)')
        ax1.legend(handles=[free_patch, occupied_patch], loc='upper right', fontsize=10)

        # Right plot: Statistics
        ax2.axis('off')
        stats_text = f"""
Occupancy Map Statistics
{'='*40}

File Information:
  Path: {pgm_path}
  Format: PGM (P5 Binary)

Dimensions:
  Width: {width} pixels
  Height: {height} pixels
  Total cells: {width * height:,}

Real-World Scale:
  Resolution: {resolution} m/pixel ({resolution*100:.1f} cm/pixel)
  Map width: {map_width_m:.2f} meters
  Map height: {map_height_m:.2f} meters
  Map area: {map_width_m * map_height_m:.2f} m²

Origin (bottom-left):
  X: {origin_x:.3f} m
  Y: {origin_y:.3f} m

Occupancy Statistics:
  Free cells: {np.sum(grid == 0):,} ({np.sum(grid == 0)/grid.size*100:.1f}%)
  Occupied cells: {np.sum(grid == 100):,} ({np.sum(grid == 100)/grid.size*100:.1f}%)
  Unknown cells: {np.sum((grid > 0) & (grid < 100)):,} ({np.sum((grid > 0) & (grid < 100))/grid.size*100:.1f}%)

Coverage:
  Occupied area: {np.sum(grid == 100) * resolution * resolution:.2f} m²
  Free area: {np.sum(grid == 0) * resolution * resolution:.2f} m²
        """

        ax2.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax2.transAxes)

        plt.suptitle('COLMAP Occupancy Map Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        print(f"\nOccupancy map visualization complete!")


def main():
    """Main entry point"""
    import sys

    # Get the script directory
    script_dir = Path(__file__).parent

    # Default sparse directory (xlenav/mappipeline/sparse1/)
    sparse_dir = script_dir / "sparse1"

    # Parse command line arguments
    if len(sys.argv) > 1:
        sparse_dir = Path(sys.argv[1])

    # Check if directory exists
    if not sparse_dir.exists():
        print(f"Error: Directory '{sparse_dir}' not found")
        print(f"Usage: python {sys.argv[0]} [sparse_dir]")
        print(f"Default: {script_dir / 'sparse1'}")
        return

    # Create visualizer
    visualizer = ColmapVisualizer(str(sparse_dir))

    print("\n" + "="*60)
    print("COLMAP Point Cloud to Occupancy Map Pipeline")
    print("="*60)

    # Step 1: Load data
    print("\n[Step 1/5] Loading COLMAP data...")
    visualizer.load_data()

    # Step 2: Fit planes by creating visualization
    print("\n[Step 2/5] Fitting camera and ground planes...")
    visualizer.create_visualization(
        camera_scale=0.2,
        show_every_nth=1,
        show_surface=True,
        show_camera_spheres=True,
        show_ground_plane=True
    )

    # Step 3: Show 3D visualization
    print("\n[Step 3/5] Displaying 3D visualization (close window to continue)...")
    o3d.visualization.draw_geometries(
        visualizer.geometries,
        window_name="COLMAP Reconstruction Viewer - Close to continue",
        width=1280,
        height=720,
        left=50,
        top=50
    )

    # Step 4: Generate occupancy map
    print("\n[Step 4/5] Generating occupancy map...")
    # Output to parent directory (xlenav/occupancy_map.pgm)
    xlenav_dir = script_dir.parent
    output_path = xlenav_dir / "occupancy_map.pgm"
    visualizer.generate_occupancy_map(output_path=str(output_path))

    # Step 5: Preview occupancy map
    print("\n[Step 5/5] Previewing occupancy map (close window to exit)...")
    visualizer.visualize_occupancy_map(str(output_path))


if __name__ == "__main__":
    main()
