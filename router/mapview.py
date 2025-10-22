import open3d as o3d
import trimesh
import numpy as np
import os


def load_glb_to_open3d(glb_path):
    """
    Load a GLB file using trimesh and convert to Open3D geometries.
    Handles both point clouds and meshes.

    Args:
        glb_path: Path to the GLB file

    Returns:
        tuple: (point_clouds, meshes) - Lists of Open3D geometries
    """
    # Load GLB file using trimesh
    print(f"Loading GLB file: {glb_path}")
    scene = trimesh.load(glb_path)

    point_clouds = []
    meshes = []

    # Handle both single geometry and scene with multiple geometries
    if isinstance(scene, trimesh.Scene):
        print(f"Found {len(scene.geometry)} geometries in scene")

        for name, geom in scene.geometry.items():
            # Handle PointCloud
            if isinstance(geom, trimesh.PointCloud):
                print(f"\n  {name}: PointCloud with {len(geom.vertices)} points")

                # Create Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.asarray(geom.vertices))

                # Add colors if available
                if hasattr(geom, 'colors') and geom.colors is not None:
                    colors = np.asarray(geom.colors)
                    if colors.shape[1] == 4:  # RGBA
                        colors = colors[:, :3]
                    if colors.max() > 1.0:  # If colors are in 0-255 range
                        colors = colors / 255.0
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    print(f"    Added {len(colors)} vertex colors")

                point_clouds.append(pcd)

            # Handle Mesh
            elif isinstance(geom, trimesh.Trimesh):
                vertices = np.asarray(geom.vertices)
                triangles = np.asarray(geom.faces)

                # Create Open3D mesh
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)

                # Add vertex colors if available
                if hasattr(geom.visual, 'vertex_colors'):
                    colors = np.asarray(geom.visual.vertex_colors)[:, :3] / 255.0
                    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

                # Compute normals for better visualization
                o3d_mesh.compute_vertex_normals()
                meshes.append(o3d_mesh)

    else:
        # Single geometry
        if isinstance(scene, trimesh.PointCloud):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(scene.vertices))
            if hasattr(scene, 'colors') and scene.colors is not None:
                colors = np.asarray(scene.colors)[:, :3]
                if colors.max() > 1.0:
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)
            point_clouds.append(pcd)
        else:
            vertices = np.asarray(scene.vertices)
            triangles = np.asarray(scene.faces)
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
            o3d_mesh.compute_vertex_normals()
            meshes.append(o3d_mesh)

    print(f"\nLoaded {len(point_clouds)} point clouds and {len(meshes)} meshes")
    return point_clouds, meshes


def visualize_glb(glb_path, show_cameras=True, show_point_cloud=True,
                  point_size=1.0, camera_scale=0.02):
    """
    Load and visualize a GLB file in Open3D viewer.

    Args:
        glb_path: Path to the GLB file
        show_cameras: Whether to show camera meshes
        show_point_cloud: Whether to show point clouds
        point_size: Size of points in visualization
        camera_scale: Scale factor for camera meshes
    """
    # Load the geometries
    point_clouds, meshes = load_glb_to_open3d(glb_path)

    # Prepare geometries to visualize
    geometries_to_show = []

    # Add point clouds
    if show_point_cloud and point_clouds:
        for pcd in point_clouds:
            geometries_to_show.append(pcd)
            print(f"Adding point cloud with {len(pcd.points)} points")

    # Add camera meshes
    if show_cameras and meshes:
        # Scale down camera meshes if needed
        for mesh in meshes:
            if camera_scale != 1.0:
                mesh.scale(camera_scale, center=mesh.get_center())
            geometries_to_show.append(mesh)
        print(f"Adding {len(meshes)} camera frustums")

    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )
    geometries_to_show.append(coord_frame)

    if not geometries_to_show:
        print("No geometries to display!")
        return

    # Visualize with custom render options
    print("\nOpening Open3D viewer...")
    print("Controls:")
    print("  - Left mouse: Rotate")
    print("  - Right mouse: Pan")
    print("  - Scroll: Zoom")
    print("  - '+'/'-': Adjust point size")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="GLB Viewer - Point Cloud & Cameras",
                      width=1920, height=1080)

    for geom in geometries_to_show:
        vis.add_geometry(geom)

    # Set render options
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark gray
    render_option.show_coordinate_frame = True

    # Run visualizer
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    glb_file = os.path.join(script_dir, "sample1.glb")

    if not os.path.exists(glb_file):
        print(f"Error: GLB file not found at {glb_file}")
    else:
        # Visualize with both point cloud and cameras
        visualize_glb(
            glb_file,
            show_cameras=True,      # Show camera frustums
            show_point_cloud=True,  # Show point cloud
            point_size=2.0,         # Point size for visualization
            camera_scale=1.0        # Scale for camera meshes
        )
