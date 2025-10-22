import open3d as o3d
import numpy as np
import yaml
from PIL import Image
from pathlib import Path


class OccupancyMapViewer:
    def __init__(self, pgm_path, yaml_path):
        """
        Initialize the occupancy map viewer.

        Args:
            pgm_path: Path to the .pgm occupancy map file
            yaml_path: Path to the .yaml metadata file
        """
        self.pgm_path = Path(pgm_path)
        self.yaml_path = Path(yaml_path)
        self.metadata = self._load_metadata()
        self.map_data = self._load_pgm()
        self.vis = None
        self.initial_view_set = False

    def _load_metadata(self):
        """Load metadata from YAML file."""
        with open(self.yaml_path, 'r') as f:
            metadata = yaml.safe_load(f)
        return metadata

    def _load_pgm(self):
        """Load PGM image file."""
        img = Image.open(self.pgm_path)
        map_data = np.array(img)
        return map_data

    def create_point_cloud(self):
        """
        Create a point cloud from the occupancy map.
        Each pixel becomes a point with z=0, colored based on occupancy value.
        """
        height, width = self.map_data.shape
        resolution = self.metadata['resolution']
        origin = self.metadata['origin']

        # Create point cloud
        points = []
        colors = []

        for i in range(height):
            for j in range(width):
                # Get occupancy value (0-255 in PGM)
                occupancy_value = self.map_data[i, j]

                # Convert pixel coordinates to world coordinates
                # Note: Image y-axis is inverted compared to map coordinates
                x = origin[0] + j * resolution
                y = origin[1] + (height - i - 1) * resolution
                z = 0.0  # Fixed z-axis for 2D map

                points.append([x, y, z])

                # Color based on occupancy (reversed):
                # Black (0 in PGM) = free space -> show as white
                # White (255 in PGM) = occupied space -> show as black
                # Gray = unknown
                normalized_value = 1.0 - (occupancy_value / 255.0)  # Invert colors
                colors.append([normalized_value, normalized_value, normalized_value])

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

        return pcd

    def visualize(self):
        """
        Visualize the occupancy map with locked 2D top-down view.
        The view allows zoom in/out and panning only (no rotation).
        """
        pcd = self.create_point_cloud()

        # Create visualizer with mouse callbacks
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Occupancy Map Viewer - 2D Mode", width=1920, height=1080)
        self.vis.add_geometry(pcd)

        # Calculate the center of the map
        points = np.asarray(pcd.points)
        center = points.mean(axis=0)

        # Calculate map bounds for proper viewport setup
        min_bounds = points.min(axis=0)
        max_bounds = points.max(axis=0)
        map_width = max_bounds[0] - min_bounds[0]
        map_height = max_bounds[1] - min_bounds[1]

        # Set up the view to look down at the XY plane
        ctr = self.vis.get_view_control()

        # Get render options
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.5, 0.5, 0.5])  # Gray background
        opt.point_size = 3.0  # Larger points for better visibility

        print("=== Occupancy Map Viewer ===")
        print(f"Map size: {self.map_data.shape}")
        print(f"Resolution: {self.metadata['resolution']} m/pixel")
        print(f"Origin: {self.metadata['origin']}")
        print(f"Map center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
        print(f"Map dimensions: {map_width:.3f}m × {map_height:.3f}m")
        print("\nControls:")
        print("  - Mouse scroll: Zoom in/out")
        print("  - Mouse drag: Pan")
        print("  - Q/ESC: Quit")
        print("\nNote: View locked to top-down 2D (Z-axis rotation disabled)")

        # Animation callback to lock the camera view
        def lock_view_to_2d(vis):
            """Lock camera to always look straight down (2D top-down view)."""
            ctr = vis.get_view_control()

            # Set initial view on first call
            if not self.initial_view_set:
                ctr.set_front([0, 0, -1])
                ctr.set_up([0, 1, 0])
                ctr.set_lookat(center)
                ctr.set_zoom(1)
                self.initial_view_set = True
                print("Initial view applied")

            # Always keep the camera locked to top-down
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, 1, 0])
            return False

        # Register animation callback (this will set initial view on first frame)
        self.vis.register_animation_callback(lock_view_to_2d)

        print(f"Waiting for initial view to be applied...")

        # Run visualizer
        self.vis.run()
        self.vis.destroy_window()


def main():
    """Main function to run the occupancy map viewer."""
    # Define paths
    base_dir = Path(__file__).parent
    pgm_path = base_dir / "occupancy_map.pgm"
    yaml_path = base_dir / "occupancy_map.yaml"

    # Check if files exist
    if not pgm_path.exists():
        print(f"Error: PGM file not found at {pgm_path}")
        return
    if not yaml_path.exists():
        print(f"Error: YAML file not found at {yaml_path}")
        return

    # Create viewer and visualize
    viewer = OccupancyMapViewer(pgm_path, yaml_path)
    viewer.visualize()


if __name__ == "__main__":
    main()
