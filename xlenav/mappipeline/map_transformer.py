"""
3D-2D Map Transformer
Provides bidirectional transformation between 3D point cloud coordinates and 2D occupancy map grid coordinates
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class MapTransformer:
    """
    Handles transformation between 3D world coordinates and 2D grid coordinates

    Usage:
        # Load transformation parameters
        transformer = MapTransformer.from_file('router/occupancy_map_transform.npz')

        # Transform 3D point to 2D grid coordinates
        grid_x, grid_y = transformer.world_to_grid(point_3d)

        # Transform 2D grid coordinates back to 3D world point (on ground plane)
        point_3d = transformer.grid_to_world(grid_x, grid_y)
    """

    def __init__(self,
                 ground_plane_normal: np.ndarray,
                 ground_plane_d: float,
                 plane_basis_u: np.ndarray,
                 plane_basis_v: np.ndarray,
                 plane_origin: np.ndarray,
                 grid_x_min: float,
                 grid_y_min: float,
                 resolution_measured_units: float,
                 scale_factor: float,
                 grid_width: int,
                 grid_height: int):
        """
        Initialize transformer with parameters saved during map generation

        Args:
            ground_plane_normal: Normal vector of the ground plane [a, b, c]
            ground_plane_d: Plane constant (ax + by + cz + d = 0)
            plane_basis_u: First basis vector of the 2D plane coordinate system
            plane_basis_v: Second basis vector of the 2D plane coordinate system
            plane_origin: Origin of the 2D plane coordinate system in 3D space
            grid_x_min: Minimum X coordinate in local plane coordinates (measured units)
            grid_y_min: Minimum Y coordinate in local plane coordinates (measured units)
            resolution_measured_units: Grid resolution in measured units
            scale_factor: Scale factor for converting measured units to real-world meters
            grid_width: Width of the occupancy grid in pixels
            grid_height: Height of the occupancy grid in pixels
        """
        self.ground_plane_normal = ground_plane_normal
        self.ground_plane_d = ground_plane_d
        self.plane_basis_u = plane_basis_u
        self.plane_basis_v = plane_basis_v
        self.plane_origin = plane_origin
        self.grid_x_min = grid_x_min
        self.grid_y_min = grid_y_min
        self.resolution = resolution_measured_units
        self.scale_factor = scale_factor
        self.grid_width = grid_width
        self.grid_height = grid_height

    @classmethod
    def from_file(cls, npz_path: str) -> 'MapTransformer':
        """
        Load transformation parameters from NPZ file

        Args:
            npz_path: Path to the *_transform.npz file

        Returns:
            MapTransformer instance
        """
        data = np.load(npz_path)

        return cls(
            ground_plane_normal=data['ground_plane_normal'],
            ground_plane_d=float(data['ground_plane_d']),
            plane_basis_u=data['plane_basis_u'],
            plane_basis_v=data['plane_basis_v'],
            plane_origin=data['plane_origin'],
            grid_x_min=float(data['grid_x_min']),
            grid_y_min=float(data['grid_y_min']),
            resolution_measured_units=float(data['resolution_measured_units']),
            scale_factor=float(data['scale_factor']),
            grid_width=int(data['grid_width']),
            grid_height=int(data['grid_height'])
        )

    def project_point_to_plane(self, point: np.ndarray) -> np.ndarray:
        """
        Project a 3D point onto the ground plane

        Args:
            point: 3D point [x, y, z]

        Returns:
            Projected point on the ground plane
        """
        # Distance from point to plane
        distance = np.dot(self.ground_plane_normal, point) + self.ground_plane_d

        # Project point onto plane
        projected = point - distance * self.ground_plane_normal

        return projected

    def world_to_plane_coords(self, point_3d: np.ndarray) -> Tuple[float, float]:
        """
        Convert 3D world coordinates to 2D plane local coordinates

        Args:
            point_3d: 3D point in world coordinates [x, y, z]

        Returns:
            (u, v): Coordinates in the local 2D plane coordinate system
        """
        # Project point to ground plane
        projected = self.project_point_to_plane(point_3d)

        # Express in local plane coordinates
        relative = projected - self.plane_origin
        u = np.dot(relative, self.plane_basis_u)
        v = np.dot(relative, self.plane_basis_v)

        return u, v

    def plane_coords_to_world(self, u: float, v: float, on_plane: bool = True) -> np.ndarray:
        """
        Convert 2D plane local coordinates to 3D world coordinates

        Args:
            u: First coordinate in plane local system
            v: Second coordinate in plane local system
            on_plane: If True, return point on ground plane. If False, at origin height.

        Returns:
            3D point in world coordinates [x, y, z]
        """
        # Reconstruct 3D point from plane coordinates
        point_3d = self.plane_origin + u * self.plane_basis_u + v * self.plane_basis_v

        if on_plane:
            # Ensure point is exactly on the plane (project it)
            point_3d = self.project_point_to_plane(point_3d)

        return point_3d

    def world_to_grid(self, point_3d: np.ndarray) -> Tuple[int, int]:
        """
        Convert 3D world coordinates to 2D grid pixel coordinates

        Args:
            point_3d: 3D point in world coordinates [x, y, z]

        Returns:
            (grid_x, grid_y): Pixel coordinates in the occupancy grid
            Returns (-1, -1) if point is outside grid bounds
        """
        # Convert to plane coordinates
        u, v = self.world_to_plane_coords(point_3d)

        # Convert to grid coordinates
        grid_x = int((u - self.grid_x_min) / self.resolution)
        grid_y = int((v - self.grid_y_min) / self.resolution)

        # Check bounds
        if grid_x < 0 or grid_x >= self.grid_width or grid_y < 0 or grid_y >= self.grid_height:
            return -1, -1

        return grid_x, grid_y

    def grid_to_world(self, grid_x: int, grid_y: int) -> np.ndarray:
        """
        Convert 2D grid pixel coordinates to 3D world coordinates (on ground plane)

        Args:
            grid_x: X coordinate in grid (pixel column)
            grid_y: Y coordinate in grid (pixel row)

        Returns:
            3D point on the ground plane [x, y, z]
        """
        # Convert grid coordinates to plane local coordinates
        # Use center of pixel
        u = self.grid_x_min + (grid_x + 0.5) * self.resolution
        v = self.grid_y_min + (grid_y + 0.5) * self.resolution

        # Convert to 3D world coordinates
        point_3d = self.plane_coords_to_world(u, v, on_plane=True)

        return point_3d

    def grid_to_world_meters(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """
        Convert grid coordinates to real-world 2D coordinates in meters
        (useful for path planning, distance calculation, etc.)

        Args:
            grid_x: X coordinate in grid (pixel column)
            grid_y: Y coordinate in grid (pixel row)

        Returns:
            (x_meters, y_meters): 2D coordinates in meters
        """
        # Get plane coordinates in measured units
        u = self.grid_x_min + (grid_x + 0.5) * self.resolution
        v = self.grid_y_min + (grid_y + 0.5) * self.resolution

        # Scale to real-world meters
        x_meters = u * self.scale_factor
        y_meters = v * self.scale_factor

        return x_meters, y_meters

    def world_meters_to_grid(self, x_meters: float, y_meters: float) -> Tuple[int, int]:
        """
        Convert real-world 2D coordinates in meters to grid coordinates

        Args:
            x_meters: X coordinate in meters (in plane local system)
            y_meters: Y coordinate in meters (in plane local system)

        Returns:
            (grid_x, grid_y): Pixel coordinates in the occupancy grid
        """
        # Scale from real-world meters to measured units
        u = x_meters / self.scale_factor
        v = y_meters / self.scale_factor

        # Convert to grid coordinates
        grid_x = int((u - self.grid_x_min) / self.resolution)
        grid_y = int((v - self.grid_y_min) / self.resolution)

        # Check bounds
        if grid_x < 0 or grid_x >= self.grid_width or grid_y < 0 or grid_y >= self.grid_height:
            return -1, -1

        return grid_x, grid_y

    def get_info(self) -> str:
        """
        Get human-readable information about the transformation

        Returns:
            Formatted string with transformation parameters
        """
        info = f"""
MapTransformer Information:
{'='*60}
Ground Plane:
  Normal: {self.ground_plane_normal}
  D constant: {self.ground_plane_d}

Plane Coordinate Frame:
  Basis U: {self.plane_basis_u}
  Basis V: {self.plane_basis_v}
  Origin: {self.plane_origin}

Grid Properties:
  Size: {self.grid_width} x {self.grid_height} pixels
  Bounds (plane coords): X[{self.grid_x_min:.3f}, {self.grid_x_min + self.grid_width * self.resolution:.3f}]
                        Y[{self.grid_y_min:.3f}, {self.grid_y_min + self.grid_height * self.resolution:.3f}]
  Resolution: {self.resolution:.6f} measured units/pixel
  Scale factor: {self.scale_factor:.6f} m/unit
  Real-world resolution: {self.resolution * self.scale_factor:.6f} m/pixel
{'='*60}
"""
        return info


def demo_usage():
    """Demonstrate usage of the MapTransformer"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python map_transformer.py <path_to_transform.npz>")
        print("Example: python map_transformer.py router/occupancy_map_transform.npz")
        return

    npz_path = sys.argv[1]

    if not Path(npz_path).exists():
        print(f"Error: File '{npz_path}' not found")
        return

    # Load transformer
    print(f"Loading transformation parameters from: {npz_path}")
    transformer = MapTransformer.from_file(npz_path)

    # Print info
    print(transformer.get_info())

    # Example 1: Convert 3D point to grid
    print("\nExample 1: 3D World → 2D Grid")
    point_3d = np.array([1.0, 2.0, 0.5])
    grid_x, grid_y = transformer.world_to_grid(point_3d)
    print(f"  3D point: {point_3d}")
    print(f"  → Grid coords: ({grid_x}, {grid_y})")

    # Example 2: Convert grid to 3D point
    print("\nExample 2: 2D Grid → 3D World")
    grid_x, grid_y = 100, 150
    point_3d = transformer.grid_to_world(grid_x, grid_y)
    print(f"  Grid coords: ({grid_x}, {grid_y})")
    print(f"  → 3D point: {point_3d}")

    # Example 3: Grid to real-world meters
    print("\nExample 3: 2D Grid → Real-world Meters")
    x_m, y_m = transformer.grid_to_world_meters(grid_x, grid_y)
    print(f"  Grid coords: ({grid_x}, {grid_y})")
    print(f"  → Meters: ({x_m:.3f}, {y_m:.3f})")

    # Example 4: Round-trip test
    print("\nExample 4: Round-trip Test (3D → Grid → 3D)")
    original = np.array([0.5, 1.5, 0.3])
    grid_x, grid_y = transformer.world_to_grid(original)
    recovered = transformer.grid_to_world(grid_x, grid_y)
    print(f"  Original 3D: {original}")
    print(f"  → Grid: ({grid_x}, {grid_y})")
    print(f"  → Recovered 3D: {recovered}")
    print(f"  Distance error: {np.linalg.norm(original - recovered):.6f} units")


if __name__ == "__main__":
    demo_usage()
