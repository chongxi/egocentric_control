import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from PIL import Image


class Map2DViewer:
    def __init__(self, map_path):
        """
        Initialize the 2D map viewer.

        Args:
            map_path: Path to either the .pgm or .yaml file (both files must have the same base name)
        """
        map_path = Path(map_path)

        # Determine base path and find both files
        if map_path.suffix.lower() == '.pgm':
            self.pgm_path = map_path
            self.yaml_path = map_path.with_suffix('.yaml')
        elif map_path.suffix.lower() == '.yaml':
            self.yaml_path = map_path
            self.pgm_path = map_path.with_suffix('.pgm')
        else:
            # Assume no extension, try to find both files
            self.pgm_path = map_path.with_suffix('.pgm')
            self.yaml_path = map_path.with_suffix('.yaml')

        # Validate files exist
        if not self.pgm_path.exists():
            raise FileNotFoundError(f"PGM file not found: {self.pgm_path}")
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {self.yaml_path}")

        # Load map data
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
        Visualize the 2D map with locked top-down view.
        The view allows zoom in/out and panning only (no rotation).
        """
        pcd = self.create_point_cloud()

        # Create visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="2D Map Viewer", width=1920, height=1080)
        self.vis.add_geometry(pcd)

        # Calculate the center of the map
        points = np.asarray(pcd.points)
        center = points.mean(axis=0)

        # Calculate map bounds for proper viewport setup
        min_bounds = points.min(axis=0)
        max_bounds = points.max(axis=0)
        map_width = max_bounds[0] - min_bounds[0]
        map_height = max_bounds[1] - min_bounds[1]

        # Get render options
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.5, 0.5, 0.5])  # Gray background
        opt.point_size = 3.0  # Larger points for better visibility

        print("=== 2D Map Viewer ===")
        print(f"Loaded files:")
        print(f"  PGM: {self.pgm_path}")
        print(f"  YAML: {self.yaml_path}")
        print(f"\nMap information:")
        print(f"  Size: {self.map_data.shape[1]} × {self.map_data.shape[0]} pixels")
        print(f"  Resolution: {self.metadata['resolution']} m/pixel")
        print(f"  Origin: [{self.metadata['origin'][0]:.3f}, {self.metadata['origin'][1]:.3f}]")
        print(f"  Center: [{center[0]:.3f}, {center[1]:.3f}]")
        print(f"  Dimensions: {map_width:.3f}m × {map_height:.3f}m")
        print("\nControls:")
        print("  - Mouse scroll: Zoom in/out")
        print("  - Mouse drag: Pan")
        print("  - Q/ESC: Quit")
        print("\nView locked to top-down 2D mode")

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

            # Always keep the camera locked to top-down
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, 1, 0])
            return False

        # Register animation callback
        self.vis.register_animation_callback(lock_view_to_2d)

        # Run visualizer
        self.vis.run()
        self.vis.destroy_window()


def main():
    """Main function to run the 2D map viewer."""
    parser = argparse.ArgumentParser(
        description='Pure 2D Map Viewer for PGM/YAML occupancy maps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 2dmapviewer.py run occupancy_map.pgm
  python 2dmapviewer.py occupancy_map.pgm
  python 2dmapviewer.py occupancy_map.yaml
  python 2dmapviewer.py /path/to/map.pgm

Note: Both .pgm and .yaml files must exist with the same base name.
        """
    )
    parser.add_argument(
        'map_file',
        type=str,
        nargs='?',
        help='Path to the .pgm or .yaml map file (both files must have the same base name)'
    )

    # Allow optional "run" command prefix for CLI usage (e.g., python 2dmapviewer.py run map.pgm)
    argv = sys.argv[1:]
    command = 'run'
    if argv and not argv[0].startswith('-'):
        possible_command = argv[0].lower()
        if possible_command == 'run':
            command = possible_command
            argv = argv[1:]

    if command != 'run':
        parser.error(f"Unknown command '{command}'. Only 'run' is supported.")

    args = parser.parse_args(argv)

    # If no argument provided, try to find map in current directory
    if args.map_file is None:
        # Try common default names
        default_names = ['occupancy_map', 'map', 'grid_map']
        base_dir = Path.cwd()

        for name in default_names:
            pgm_path = base_dir / f"{name}.pgm"
            if pgm_path.exists():
                args.map_file = str(pgm_path)
                print(f"No file specified, using found map: {args.map_file}")
                break

        if args.map_file is None:
            print("Error: No map file specified and no default map found.")
            print("Please provide a path to a .pgm or .yaml map file.")
            parser.print_help()
            return

    try:
        # Create viewer and visualize
        viewer = Map2DViewer(args.map_file)
        viewer.visualize()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error loading or displaying map: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
