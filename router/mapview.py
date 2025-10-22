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

    def load_data(self):
        """Load all COLMAP data"""
        print("Loading COLMAP data...")
        print("=" * 60)

        # Load cameras
        cameras_path = self.sparse_dir / "cameras.bin"
        if cameras_path.exists():
            self.cameras = ColmapDataLoader.read_cameras_binary(str(cameras_path))
            print(f"✓ Loaded {len(self.cameras)} cameras")

        # Load images (camera poses)
        images_path = self.sparse_dir / "images.bin"
        if images_path.exists():
            self.images = ColmapDataLoader.read_images_binary(str(images_path))
            print(f"✓ Loaded {len(self.images)} camera poses")

        # Load point cloud
        ply_path = self.sparse_dir / "points.ply"
        if ply_path.exists():
            self.point_cloud = o3d.io.read_point_cloud(str(ply_path))
            num_points = len(self.point_cloud.points)
            print(f"✓ Loaded point cloud with {num_points:,} points")

            # Print detailed point cloud statistics
            if num_points > 0:
                points = np.asarray(self.point_cloud.points)

                print("\nPoint Cloud Statistics:")
                print("-" * 60)

                # X axis stats
                x_min, x_max = points[:, 0].min(), points[:, 0].max()
                x_range = x_max - x_min
                print(f"X-axis: min={x_min:.4f}, max={x_max:.4f}, range={x_range:.4f}")

                # Y axis stats
                y_min, y_max = points[:, 1].min(), points[:, 1].max()
                y_range = y_max - y_min
                print(f"Y-axis: min={y_min:.4f}, max={y_max:.4f}, range={y_range:.4f}")

                # Z axis stats
                z_min, z_max = points[:, 2].min(), points[:, 2].max()
                z_range = z_max - z_min
                print(f"Z-axis: min={z_min:.4f}, max={z_max:.4f}, range={z_range:.4f}")

                # Centroid
                centroid = points.mean(axis=0)
                print(f"\nCentroid: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]")

                # Bounding box volume
                volume = x_range * y_range * z_range
                print(f"Bounding box volume: {volume:.4f} cubic units")

                # Check if colors exist
                if self.point_cloud.has_colors():
                    print(f"Point cloud has colors: Yes")
                else:
                    print(f"Point cloud has colors: No")

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

    def calculate_plane_from_3_points(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculate plane equation from 3 points: ax + by + cz + d = 0

        Args:
            p1, p2, p3: Three 3D points

        Returns:
            normal: Normal vector [a, b, c] (unit vector)
            d: Plane constant
        """
        # Calculate two vectors in the plane
        v1 = p2 - p1
        v2 = p3 - p1

        # Normal vector is cross product
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)  # Normalize

        # Calculate d using point p1
        d = -np.dot(normal, p1)

        return normal, d

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

    def create_surface_from_pointcloud_bounds(self, color: Tuple[float, float, float] = (0.0, 0.8, 0.2)) -> o3d.geometry.TriangleMesh:
        """
        Create a rectangular surface mesh based on the point cloud's X,Y extent,
        lying on the plane defined by cameras 2-4 positions

        Args:
            color: RGB color for the surface

        Returns:
            TriangleMesh representing the rectangular surface
        """
        if self.point_cloud is None or len(self.point_cloud.points) == 0:
            print("Warning: No point cloud available to create surface")
            return None

        if len(self.images) < 4:
            print("Warning: Need at least 4 cameras to use cameras 2-4")
            return None

        # Get point cloud bounds
        points = np.asarray(self.point_cloud.points)
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()

        print("\nCreating Surface:")
        print("=" * 60)
        print(f"Point cloud X range: [{x_min:.4f}, {x_max:.4f}]")
        print(f"Point cloud Y range: [{y_min:.4f}, {y_max:.4f}]")
        print(f"Point cloud Z range: [{z_min:.4f}, {z_max:.4f}]")

        # Get cameras 2-4 (indices 1, 2, 3)
        camera_poses_2_to_4 = list(self.images.values())[1:4]
        camera_centers = []

        print("\nCameras 2-4 Positions (defining the plane):")
        print("-" * 60)
        for i, pose in enumerate(camera_poses_2_to_4):
            center = self.get_camera_center(pose)
            camera_centers.append(center)
            image_name = pose.get('name', f'image_{pose["id"]}')
            print(f"  Camera {i+2} ({image_name}): [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")

        # Calculate plane from 3 camera positions
        p1, p2, p3 = camera_centers[0], camera_centers[1], camera_centers[2]
        normal, d = self.calculate_plane_from_3_points(p1, p2, p3)

        print("\nPlane Equation: {:.4f}x + {:.4f}y + {:.4f}z + {:.4f} = 0".format(
            normal[0], normal[1], normal[2], d
        ))

        # Verify that the 3 points lie on the plane (should be ~0)
        print("\nVerification (distance to plane, should be ~0):")
        for i, p in enumerate(camera_centers):
            dist = abs(np.dot(normal, p) + d)
            print(f"  Camera {i+1} distance to plane: {dist:.6f}")

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

        # Calculate actual surface dimensions and area on the plane
        # The surface might be tilted, so we calculate actual edge lengths
        edge1_length = np.linalg.norm(vertices[1] - vertices[0])
        edge2_length = np.linalg.norm(vertices[3] - vertices[0])

        # Calculate area using cross product for accuracy
        v1 = vertices[1] - vertices[0]
        v2 = vertices[3] - vertices[0]
        area = np.linalg.norm(np.cross(v1, v2))

        print("\nProjected Rectangle Corners on Plane:")
        print("-" * 60)
        for i, vertex in enumerate(vertices):
            corner_names = ["Bottom-left", "Bottom-right", "Top-right", "Top-left"]
            print(f"  {corner_names[i]}: [{vertex[0]:.4f}, {vertex[1]:.4f}, {vertex[2]:.4f}]")

        print(f"\nSurface Dimensions (on the tilted plane):")
        print(f"  Edge 1 length: {edge1_length:.4f}")
        print(f"  Edge 2 length: {edge2_length:.4f}")
        print(f"  Area:          {area:.4f} square units")

        # Calculate tilt angle of the plane
        z_axis = np.array([0, 0, 1])
        angle_rad = np.arccos(np.clip(np.dot(normal, z_axis), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        print(f"\nPlane tilt from horizontal: {angle_deg:.2f} degrees")

        print("=" * 60)

        return mesh

    def create_camera_spheres(self, sphere_radius: float = 0.01) -> list:
        """
        Create spheres at cameras 2-4 positions to mark them

        Args:
            sphere_radius: Radius of the spheres

        Returns:
            List of sphere meshes
        """
        if len(self.images) < 4:
            print("Warning: Need at least 4 cameras to mark cameras 2-4")
            return []

        spheres = []
        camera_poses_2_to_4 = list(self.images.values())[1:4]

        # Different colors for each sphere
        colors = [
            (1.0, 0.0, 0.0),  # Red - Camera 2
            (0.0, 1.0, 0.0),  # Green - Camera 3
            (0.0, 0.0, 1.0),  # Blue - Camera 4
        ]

        print("\nMarking Cameras 2-4 with Spheres:")
        print("-" * 60)

        for i, pose in enumerate(camera_poses_2_to_4):
            center = self.get_camera_center(pose)

            # Create sphere
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sphere.translate(center)
            sphere.paint_uniform_color(colors[i])
            sphere.compute_vertex_normals()

            spheres.append(sphere)

            # Get image name for better identification
            image_name = pose.get('name', f'image_{pose["id"]}')

            print(f"  Camera {i+2} ({image_name}):")
            print(f"    Position: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
            print(f"    Color: {'Red' if i == 0 else 'Green' if i == 1 else 'Blue'}")

        print("-" * 60)

        return spheres

    def create_visualization(self, camera_scale: float = 0.2, show_every_nth: int = 1, show_surface: bool = True, show_camera_spheres: bool = True):
        """
        Create visualization geometries

        Args:
            camera_scale: Scale of camera frustums
            show_every_nth: Show every nth camera (to reduce clutter)
            show_surface: Whether to show the surface from cameras 2-4
            show_camera_spheres: Whether to show spheres at cameras 2-4 positions
        """
        self.geometries = []

        # Add point cloud
        if self.point_cloud is not None:
            self.geometries.append(self.point_cloud)
            print(f"Added point cloud")

        # Add spheres at cameras 2-4 positions
        if show_camera_spheres and len(self.images) >= 4:
            spheres = self.create_camera_spheres(sphere_radius=0.01)
            for sphere in spheres:
                self.geometries.append(sphere)
            if spheres:
                print(f"✓ Added {len(spheres)} camera position spheres (cameras 2-4)")

        # Add rectangular surface based on point cloud bounds
        if show_surface and self.point_cloud is not None:
            surface = self.create_surface_from_pointcloud_bounds(color=(0.0, 0.8, 0.3))
            if surface is not None:
                self.geometries.append(surface)
                print("✓ Added rectangular surface to visualization")

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

        print(f"Added {camera_count} camera frustums")

        # Add coordinate frame at origin
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        self.geometries.append(coord_frame)

    def visualize(self, camera_scale: float = 0.2, show_every_nth: int = 1, show_surface: bool = True, show_camera_spheres: bool = True):
        """
        Visualize the COLMAP reconstruction

        Args:
            camera_scale: Scale of camera frustums
            show_every_nth: Show every nth camera (1 = show all, 5 = show every 5th)
            show_surface: Whether to show the surface from cameras 2-4
            show_camera_spheres: Whether to show spheres at cameras 2-4 positions
        """
        self.load_data()
        self.create_visualization(camera_scale, show_every_nth, show_surface, show_camera_spheres)

        print(f"\nVisualizing {len(self.geometries)} geometries...")
        print("Controls:")
        print("  - Mouse: Rotate view")
        print("  - Scroll: Zoom")
        print("  - Ctrl+Mouse: Pan")
        print("  - Q or ESC: Exit")

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

    # Visualize (show every 5th camera to reduce clutter)
    # Adjust show_every_nth based on your dataset size
    # Set show_surface=True to display a rectangular surface matching point cloud X,Y extent
    # Set show_camera_spheres=True to mark cameras 2-4 with colored spheres
    visualizer.visualize(camera_scale=0.2, show_every_nth=5, show_surface=True, show_camera_spheres=True)


if __name__ == "__main__":
    main()
