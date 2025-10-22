"""
COLMAP Visualization with Open3D
Loads and visualizes PLY point cloud and camera poses from COLMAP sparse reconstruction
"""

import numpy as np
import open3d as o3d
import struct
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

    def __init__(self, sparse_dir: str):
        self.sparse_dir = Path(sparse_dir)
        self.cameras = {}
        self.images = {}
        self.point_cloud = None
        self.geometries = []
        self.plane_fitting_camera_indices = []  # Indices of cameras used for plane fitting

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

        # Get point cloud bounds
        points = np.asarray(self.point_cloud.points)
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()

        surface_width = x_max - x_min
        surface_length = y_max - y_min

        # Calculate how many cameras to use (80% of total)
        total_cameras = len(self.images)
        num_cameras_to_use = max(3, int(total_cameras * percentage))  # At least 3 cameras

        print(f"Selecting {num_cameras_to_use} out of {total_cameras} cameras ({percentage*100:.0f}%) - excluding outliers")
        print(f"Surface width (X): {surface_width:.4f}, length (Y): {surface_length:.4f}")

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

        # Store the indices of cameras used for plane fitting
        self.plane_fitting_camera_indices = sorted(selected_indices.tolist())

        print(f"Selected cameras (excluding outliers): {len(selected_indices)} cameras")

        # Create rectangle corners in 3D space (initially at arbitrary Z)
        # We'll use the mean Z as a starting point, then project onto the plane
        z_mean = np.mean([c[2] for c in camera_centers])

        corners_initial = np.array([
            [x_min, y_min, z_mean],  # Bottom-left
            [x_max, y_min, z_mean],  # Bottom-right
            [x_max, y_max, z_mean],  # Top-right
            [x_min, y_max, z_mean],  # Top-left
        ])

        # Project each corner onto the plane defined by the 3 cameras
        vertices = np.array([
            self.project_point_to_plane(corner, normal, d)
            for corner in corners_initial
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
        normal, d, inlier_mask = self.fit_plane_ransac(points, distance_threshold=distance_threshold)

        if normal is None:
            print("Warning: RANSAC failed to find a plane")
            return None

        # Get inlier points to determine the extent of the ground plane
        inlier_points = points[inlier_mask]

        # Calculate bounding box of inliers in X,Y
        x_min, x_max = inlier_points[:, 0].min(), inlier_points[:, 0].max()
        y_min, y_max = inlier_points[:, 1].min(), inlier_points[:, 1].max()

        ground_width = x_max - x_min
        ground_length = y_max - y_min

        print(f"Ground plane width (X): {ground_width:.4f}, length (Y): {ground_length:.4f}")

        # Create rectangle corners at the extent of inliers
        # Start with corners at mean Z of inliers
        z_mean = inlier_points[:, 2].mean()

        corners_initial = np.array([
            [x_min, y_min, z_mean],  # Bottom-left
            [x_max, y_min, z_mean],  # Bottom-right
            [x_max, y_max, z_mean],  # Top-right
            [x_min, y_max, z_mean],  # Top-left
        ])

        # Project corners onto the RANSAC-fitted plane
        vertices = np.array([
            self.project_point_to_plane(corner, normal, d)
            for corner in corners_initial
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


def main():
    """Main entry point"""
    import sys

    # Default sparse directory
    sparse_dir = "router/sparse1"

    # Parse command line arguments
    if len(sys.argv) > 1:
        sparse_dir = sys.argv[1]

    # Check if directory exists
    if not Path(sparse_dir).exists():
        print(f"Error: Directory '{sparse_dir}' not found")
        print(f"Usage: python {sys.argv[0]} [sparse_dir]")
        return

    # Create visualizer
    visualizer = ColmapVisualizer(sparse_dir)

    # Visualize all cameras with spheres
    # show_every_nth=1 displays all cameras
    # Set show_surface=True to display a rectangular surface fitted to 80% of cameras (green)
    # Set show_ground_plane=True to display RANSAC-fitted ground plane from point cloud (orange/brown)
    # Set show_camera_spheres=True to mark all cameras with spheres (green: used for fitting, yellow: not used)
    visualizer.visualize(camera_scale=0.2, show_every_nth=1, show_surface=True, show_camera_spheres=True, show_ground_plane=True)


if __name__ == "__main__":
    main()
